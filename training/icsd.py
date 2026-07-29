"""Adapter for loading the ICSD-YOLO fork with its release gaps restored.

The pinned submodule (external/icsd_yolo @ 47018ed) has six defects fixed here
without editing tracked submodule files (ADR 0002):
1. ``from einsum import einsum`` — a typo for the einops function; resolved by
   putting ``training/icsd_compat`` (which contains an ``einsum.py`` shim) on
   ``sys.path``.
2. ``models.nn.Add_modules`` was never published; the canonical restoration in
   ``training/icsd_compat/Add_modules`` is copied into the submodule tree on
   first use (idempotent; the copy is untracked in the submodule).
3. ``parse_model`` references the paper's custom blocks without importing them;
   injected by :func:`_inject_custom_blocks` (which also aliases the undefined
   ``FocusNetBlock`` to ``SparseFocusBlock``).
4. ``C3k2_AKConv`` is missing from ``parse_model``'s argument-handling sets;
   patched in memory by :func:`_patch_parse_model`.
5. ``parse_model`` reads the out-channel spec from ``args[1]`` although every
   shipped YAML (and stock Ultralytics) puts it at ``args[0]``; also patched by
   :func:`_patch_parse_model`.
6. ``yaml_model_load`` derives the compound scale from a ``yolo<digits><scale>``
   filename pattern that ``ICSD-YOLOs.yaml`` cannot match, silently collapsing
   every variant to 'n'; :func:`_variant_cfg` writes a verbatim copy under a
   pattern-compliant name.
7. Repeat handling deviates from stock Ultralytics in two places that chain
   incompatible module copies (``n = 2`` after the repeat insert; depth gain
   ``max(round(n * depth), 2)``); restored to stock semantics (``n = 1`` /
   ``min 1``) by :func:`_patch_parse_model`.
8. The C3k2 m/l/x flag is set via ``args[4]`` (the expansion-ratio slot, an
   IndexError for two-arg C3k2 layers); stock uses ``args[3]`` (the c3k slot).
9. The focus blocks are channel-preserving with signature ``(dim, ctx_dim,
   mlp_ratio, drop_path)``, so trailing YAML args would land on ``mlp_ratio``
   (e.g. ``SparseFocusBlock [512, 512]`` → a ``dim*512`` MLP);
   :func:`_patch_parse_model` maps them to ``(dim, ctx_dim=dim)`` — the only
   functional configuration, as the class's own ``ctx_dim=dim//4`` default
   crashes in ``forward``.

10. ``BaseTrainer.get_dataset`` ignores ``self.args.data`` and rebuilds the
    dataset path from the fork's private argparse config
    (``args.data_path + '/' + args.data``), which resolves to ``./datasets/``;
    :func:`_patch_get_dataset` restores stock resolution.
11. ``models/utils/__init__.py`` calls the fork's private ``parse_args()`` at
    import time, parsing ``sys.argv`` and crashing any host with unrecognised
    argv (pytest, notebooks); shadowed by the argv-safe
    ``icsd_compat/config_utils`` package placed earlier on ``sys.path``.

Even after restoration the release cannot reproduce the paper's reported
complexity (reconstructed X variant: 121M params / 486 GFLOPs vs the paper's
116.8 GFLOPs), so we use the restored detector as an honestly-documented
approximation, not a faithful reproduction. See ADR 0002.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ICSD_ROOT = PROJECT_ROOT / "external" / "icsd_yolo"
COMPAT_DIR = PROJECT_ROOT / "training" / "icsd_compat"
ICSD_MODEL_CFG_DIR = ICSD_ROOT / "models" / "cfg" / "models" / "ICSD"
GENERATED_CFG_DIR = PROJECT_ROOT / "training" / "generated"


def install_compat_shims() -> None:
    """Restore the missing Add_modules package inside the submodule (idempotent)."""
    target = ICSD_ROOT / "models" / "nn" / "Add_modules"
    source = COMPAT_DIR / "Add_modules"
    if not target.is_dir():
        shutil.copytree(source, target)
        logger.info("Restored missing Add_modules package into %s", target)
    # COMPAT_DIR must end up *before* ICSD_ROOT so its argv-safe config_utils
    # shadows the fork's (release gap 11); insert(0) reverses iteration order.
    for entry in (str(ICSD_ROOT), str(COMPAT_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _inject_custom_blocks() -> None:
    """Inject the paper's custom blocks into models.nn.tasks.

    tasks.py references ConvStemBlock/FocusBlock/GlobalContextExtractor/
    SparseFocusBlock/FocusNetBlock in parse_model() but never imports them —
    a third release gap. ``FocusNetBlock`` (used by the s/m/l/x YAML heads) is
    defined nowhere in the release; the paper describes FocusBlock as
    integrating CoordConv + multi-scale context fusion + sparse self-attention
    + MLP enhancement, which matches ``SparseFocusBlock`` (the plain
    ``FocusBlock`` class lacks the sparse attention), so FocusNetBlock is
    aliased to SparseFocusBlock. Recorded in ADR 0002.
    """
    import models.nn.tasks as tasks
    from models.nn.modules.conv import AKConv, C3k2_AKConv
    from models.nn.modules.ConvStemBlock import ConvStemBlock
    from models.nn.modules.FocusBlock import FocusBlock
    from models.nn.modules.GlobalContextExtractor import GlobalContextExtractor
    from models.nn.modules.SparseSelfAttenFocusBlock import SparseFocusBlock

    tasks.AKConv = AKConv
    tasks.C3k2_AKConv = C3k2_AKConv
    tasks.ConvStemBlock = ConvStemBlock
    tasks.FocusBlock = FocusBlock
    tasks.GlobalContextExtractor = GlobalContextExtractor
    tasks.SparseFocusBlock = SparseFocusBlock
    tasks.FocusNetBlock = SparseFocusBlock


def _patch_parse_model() -> None:
    """Re-exec ``parse_model`` with release gaps 4, 5, 7, 8 and 9 repaired.

    See the module docstring for the gap catalogue. The defects live in
    function-local literals of ``parse_model``, so namespace injection cannot
    reach them; instead the function source is patched in memory and
    re-executed into the module namespace, leaving the submodule checkout
    pristine. Idempotent (the C3k2_AKConv marker is present iff patched).
    """
    import inspect

    import models.nn.tasks as tasks

    src_lines, start_lineno = inspect.getsourcelines(tasks.parse_model)
    src = "".join(src_lines)
    if "C3k2_AKConv" in src:
        return
    replacements = [
        # gap 4: register C3k2_AKConv in both argument-handling sets
        (
            "            SparseFocusBlock,\n",
            "            SparseFocusBlock,\n            C3k2_AKConv,\n",
        ),
        (
            "            C2PSA,\n        }\n    )\n",
            "            C2PSA,\n            C3k2_AKConv,\n        }\n    )\n",
        ),
        # gap 5: out-channels come from args[0] in every shipped YAML
        (
            "            c1, c2 = ch[f], args[1]\n",
            "            c1, c2 = ch[f], args[0]\n",
        ),
        # gap 7: stock repeat semantics (fork chains n incompatible copies)
        (
            "                args.insert(2, n)  # number of repeats\n                n = 2\n",
            "                args.insert(2, n)  # number of repeats\n                n = 1\n",
        ),
        (
            "        n = n_ = max(round(n * depth), 2) if n > 1 else n",
            "        n = n_ = max(round(n * depth), 1) if n > 1 else n",
        ),
        # gap 8: c3k flag lives at args[3] after the repeat insert, not args[4]
        (
            "                    args[4] = True\n",
            "                    args[3] = True\n",
        ),
        # gap 9: focus blocks take (dim, ctx_dim); trailing YAML args must not
        # reach mlp_ratio, and ctx_dim must equal dim (dim//4 default crashes)
        (
            "            args = [c1, c2, *args[1:]]\n",
            "            args = [c1, c2, *args[1:]]\n"
            "            if m in {FocusNetBlock, FocusBlock, SparseFocusBlock}:\n"
            "                args = [c1, c2]\n",
        ),
    ]
    for old, new in replacements:
        if old not in src:
            raise RuntimeError(
                "parse_model source no longer matches the pinned ICSD commit; "
                f"refusing to patch (missing anchor: {old!r})."
            )
        src = src.replace(old, new, 1)
    # Pad so tracebacks report line numbers close to the real tasks.py
    # (inserted lines shift everything after the frozensets slightly).
    src = "\n" * (start_lineno - 1) + src
    exec(compile(src, tasks.__file__, "exec"), tasks.__dict__)
    logger.info("Patched parse_model in memory (release gaps 4, 5, 7, 8, 9)")


def _patch_get_dataset() -> None:
    """Restore stock dataset resolution in ``BaseTrainer`` (release gap 10).

    The fork's ``get_dataset`` discards ``self.args.data`` and rebuilds the
    path from its private argparse config (``args.data_path + '/' +
    args.data`` → ``./datasets/``), so no caller-supplied dataset can ever be
    trained on. Patched in memory like :func:`_patch_parse_model`.
    """
    import inspect
    import textwrap

    import models.engine.trainer as trainer_mod

    if getattr(trainer_mod, "_csa_get_dataset_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(trainer_mod.BaseTrainer.get_dataset))
    drop_anchor = "    args = parse_args()\n"
    path_anchor = (
        "            data = args.data_path + '/' + args.data\n"
        "            data = check_det_dataset(data)\n"
    )
    if drop_anchor not in src or path_anchor not in src:
        raise RuntimeError(
            "BaseTrainer.get_dataset source no longer matches the pinned ICSD "
            "commit; refusing to patch (update _patch_get_dataset anchors)."
        )
    src = src.replace(drop_anchor, "", 1)
    src = src.replace(path_anchor, "            data = check_det_dataset(self.args.data)\n", 1)
    namespace: dict = {}
    exec(compile(src, trainer_mod.__file__, "exec"), trainer_mod.__dict__, namespace)
    trainer_mod.BaseTrainer.get_dataset = namespace["get_dataset"]
    trainer_mod._csa_get_dataset_patched = True
    logger.info("Patched BaseTrainer.get_dataset in memory (release gap 10)")


def _variant_cfg(arch: str, scale: str) -> Path:
    """Return a config path whose filename encodes the compound scale (gap 6).

    ``yaml_model_load`` ignores any ``scale:`` key and re-derives the scale
    from the filename via ``re.search(r"yolo[v]?\\d+([nslmx])", stem)``.
    ``ICSD-YOLOs.yaml`` has no digits before the scale letter, so every
    variant silently collapses to the first scale ('n'). A verbatim copy named
    ``icsd-<arch>arch-yolo11<scale>.yaml`` satisfies the pattern.

    ``arch`` selects the source YAML (the release's s/m/l/x files are four
    *different* architectures, not one architecture at four scales); ``scale``
    selects the compound-scaling row applied to it.
    """
    src_cfg = ICSD_MODEL_CFG_DIR / f"ICSD-YOLO{arch}.yaml"
    if not src_cfg.is_file():
        raise FileNotFoundError(f"No ICSD config for arch '{arch}': {src_cfg}")
    if scale not in "nsmlx" or len(scale) != 1:
        raise ValueError(f"Invalid scale '{scale}' (expected one of n/s/m/l/x)")
    GENERATED_CFG_DIR.mkdir(exist_ok=True)
    dst_cfg = GENERATED_CFG_DIR / f"icsd-{arch}arch-yolo11{scale}.yaml"
    content = src_cfg.read_text()
    if not dst_cfg.is_file() or dst_cfg.read_text() != content:
        dst_cfg.write_text(content)
        logger.info("Wrote scale-encoding config copy %s", dst_cfg)
    return dst_cfg


try:  # torch is a hard dependency of the project; guard only for doc tooling
    import torch as _torch
    from torch import nn as _nn
    from torch.nn import functional as _F

    class MPSSafeAdaptiveAvgPool2d(_nn.Module):
        """AdaptiveAvgPool2d that routes non-divisible MPS inputs through CPU.

        Module-level (not function-local) so pickled checkpoints containing it
        can be saved and reloaded.
        """

        def __init__(self, output_size) -> None:
            super().__init__()
            self.output_size = (
                (output_size, output_size)
                if isinstance(output_size, int)
                else tuple(output_size)
            )

        def forward(self, x: "_torch.Tensor") -> "_torch.Tensor":
            if x.device.type == "mps":
                h, w = x.shape[-2:]
                oh, ow = self.output_size
                if h % oh or w % ow:
                    return _F.adaptive_avg_pool2d(x.cpu(), self.output_size).to(x.device)
            return _F.adaptive_avg_pool2d(x, self.output_size)

except ImportError:  # pragma: no cover
    MPSSafeAdaptiveAvgPool2d = None  # type: ignore[assignment]


def _make_mps_safe(module) -> int:
    """Replace non-global AdaptiveAvgPool2d instances with an MPS-safe wrapper.

    The focus blocks pool to PSP-style grids (e.g. 3×3, 6×6) that rarely divide
    the feature-map size; the MPS kernel rejects non-divisible cases with a
    RuntimeError (pytorch#96056), and ``PYTORCH_ENABLE_MPS_FALLBACK`` does not
    apply because the op *is* implemented. The wrapper routes only the
    non-divisible-on-MPS case through CPU — the tensors involved are tiny.

    Returns the number of replaced modules.
    """
    from torch import nn

    replaced = 0
    for parent in module.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.AdaptiveAvgPool2d) and child.output_size not in (1, (1, 1)):
                setattr(parent, name, MPSSafeAdaptiveAvgPool2d(child.output_size))
                replaced += 1
    return replaced


def load_icsd_yolo(scale: str = "n", arch: str = "x"):
    """Return a restored ICSD-YOLO model at the given compound scale.

    Imports the authors' fork (package name ``models``) after installing the
    compatibility shims and in-memory patches. The returned object has the
    standard Ultralytics ``YOLO`` API (train/val/predict).

    Defaults: ``arch='x'`` — the only shipped YAML matching the paper's
    described architecture (ConvStemBlock stem + FocusBlock refinement over a
    YOLO11 layout; the s/m/l files are earlier mutually-inconsistent
    experiments, and ICSD-YOLOn.yaml is a stale AKConv-only config).
    ``scale='n'`` — the size class of our YOLO11n baseline, for a fair
    same-budget comparison (5.67M params / 17.7 GFLOPs restored).
    """
    install_compat_shims()
    from models import YOLO  # the fork's renamed ultralytics package

    _inject_custom_blocks()
    _patch_parse_model()
    _patch_get_dataset()
    cfg = _variant_cfg(arch, scale)
    logger.info("Loading restored ICSD-YOLO (arch=%s, scale=%s) from %s", arch, scale, cfg)
    model = YOLO(str(cfg))
    replaced = _make_mps_safe(model.model)
    if replaced:
        logger.info("Wrapped %d adaptive-pool modules for MPS safety", replaced)

    def _rewrap(trainer) -> None:
        # The trainer rebuilds the network from YAML, discarding the wrappers.
        n = _make_mps_safe(trainer.model)
        if n:
            logger.info("Re-wrapped %d adaptive-pool modules on trainer model", n)

    model.add_callback("on_pretrain_routine_start", _rewrap)
    return model


if __name__ == "__main__":
    # Smoke test: instantiate the restored ICSD-YOLO and run one forward pass.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import torch

    model = load_icsd_yolo(scale="n", arch="x")
    from models.utils.torch_utils import get_flops

    n_params = sum(p.numel() for p in model.model.parameters())
    flops = get_flops(model.model, imgsz=640)
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 640, 640))
    shapes = [tuple(t.shape) for t in (out[0] if isinstance(out, tuple) else out)]
    print(f"Restored ICSD-YOLO (x-arch @ n-scale): {n_params / 1e6:.2f}M params, {flops:.1f} GFLOPs")
    print(f"Forward pass OK, output shapes: {shapes}")
