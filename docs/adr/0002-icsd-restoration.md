# ADR 0002: Restoration of the ICSD-YOLO release, and what it can honestly claim

Date: 2026-07-29 (Week 4)
Status: accepted
Depends on: [ADR 0001](0001-backbone-strategy.md) (ICSD-YOLO as base detector)

## Context

ADR 0001 selected ICSD-YOLO (Shi et al., *Expert Systems with Applications*,
2025 — "ICSD-YOLO: Intelligent detection for real-time industrial field
safety") as our base detector, using the authors' released fork of Ultralytics
8.3.74 pinned as a submodule at `external/icsd_yolo` (PrintSC/code
@ `47018ed`). No pretrained weights were released, so the code must at minimum
instantiate and train.

It does neither, as shipped. This ADR catalogues every defect found, how we
repaired it, and the honest limits of the result.

## Findings: release defects

All fixes live in `training/icsd.py` (adapter) and `training/icsd_compat/`
(shim package). **No tracked submodule file is modified**; two mechanisms are
used: an untracked package copy inside the submodule tree, and in-memory
source patching of `parse_model` at import time (re-exec into the module
namespace, anchored on exact source strings that fail loudly if the pinned
commit changes).

| # | Defect | Symptom | Fix |
|---|--------|---------|-----|
| 1 | `from einsum import einsum` in `block.py` (typo for the einops function) | `ModuleNotFoundError` | `icsd_compat/einsum.py` shim on `sys.path` |
| 2 | `models.nn.Add_modules` package imported but never published | `ModuleNotFoundError` | Restored from definitions found verbatim in `block.py:395-591` / `conv.py` (`icsd_compat/Add_modules/`, copied into the submodule untracked) |
| 3 | `parse_model` references `ConvStemBlock`/`FocusBlock`/`GlobalContextExtractor`/`SparseFocusBlock`/`FocusNetBlock` without importing them; `FocusNetBlock` is defined **nowhere** in the release | `NameError` on any model parse | Namespace injection; `FocusNetBlock` aliased to `SparseFocusBlock` (the paper describes sparse self-attention, which only `SparseFocusBlock` implements) |
| 4 | `C3k2_AKConv` absent from `parse_model`'s `base_modules`/`repeat_modules` sets | Raw YAML args reach the constructor (`c2=False` → `TypeError`) | In-memory patch: added to both sets |
| 5 | Out-channels read from `args[1]`; every shipped YAML (and stock Ultralytics) puts them at `args[0]` | Kernel sizes used as channel counts (zero-element convs); `IndexError` on single-arg modules | In-memory patch: `args[0]` |
| 6 | `yaml_model_load` derives compound scale from a `yolo<digits><scale>` filename regex that `ICSD-YOLO{s,m,l,x}.yaml` cannot match | **Every variant silently collapses to n-scale** — including, presumably, in the authors' own runs | Adapter writes a verbatim copy under a pattern-compliant name (`icsd-<arch>arch-yolo11<scale>.yaml`) |
| 7 | Repeat handling deviates from stock: `n = 2` after the repeat-arg insert (stock: `n = 1`) and depth gain `max(round(n*depth), 2)` (stock: min 1) | Chains two copies of c1→c2 blocks → channel mismatch `RuntimeError` | In-memory patch: stock semantics |
| 8 | C3k2 m/l/x flag set via `args[4]` (stock: `args[3]`) | `IndexError` for two-arg C3k2 layers at m/l/x scales; silently corrupts the expansion ratio otherwise | In-memory patch: `args[3]` |
| 9 | Focus blocks are channel-preserving with signature `(dim, ctx_dim, mlp_ratio, drop_path)`, but base-module arg insertion sends trailing YAML args into `mlp_ratio` (e.g. `SparseFocusBlock [512, 512]` → `mlp_ratio=512`, a `dim*512`-wide MLP) | Parameter explosion (l-arch @ n-scale: 21.9M) | In-memory patch: focus blocks receive `(dim, ctx_dim=dim)` only. The class's own `ctx_dim=dim//4` default is itself broken (channel mismatch in the residual add), so `ctx_dim=dim` is the only functional configuration |
| 10 | `BaseTrainer.get_dataset` discards `self.args.data` and rebuilds the path from the fork's private argparse config (`args.data_path + '/' + args.data` → `./datasets/`) | No caller-supplied dataset can ever be trained on (`IsADirectoryError`) | In-memory patch of the method: stock resolution `check_det_dataset(self.args.data)` |
| 11 | `models/utils/__init__.py` calls the fork's private `parse_args()` at import time, parsing `sys.argv` | Import crashes under any host with unrecognised argv (pytest exits with argparse error) | Argv-safe `config_utils` shadow package on `sys.path` ahead of the submodule root, delegating to the original parser with neutralised argv |

## Findings: the YAML variants are not one architecture

The release ships five ICSD YAMLs. They are **mutually inconsistent
architectures**, not one architecture at five scales:

- `ICSD-YOLOn.yaml` — stale AKConv-only experiment; none of the paper's blocks.
- `ICSD-YOLOs.yaml` — ConvStemBlock stem + `C3k2_AKConv` throughout + three
  extra head layers (`FocusNetBlock [512]`, `C3k2_AKConv [1024]`,
  `FocusNetBlock [1024]`).
- `ICSD-YOLOm.yaml` — as `s` but **without any focus block**.
- `ICSD-YOLOl.yaml` — `SparseFocusBlock` in backbone + head, stock `C3k2` in
  head, `GlobalContextExtractor` commented out.
- `ICSD-YOLOx.yaml` — cleanest and newest (current Ultralytics header format):
  ConvStemBlock ×2 stem, stock `C3k2`, `SparseFocusBlock` in backbone tail and
  head. **This is the only file matching the paper's described architecture**
  ("Information ConvStem FocusBlock Detector": ConvStemBlock for low-level
  feature extraction + FocusBlock for multi-level semantic refinement).

Because each file carries the full n/s/m/l/x `scales:` dict, architecture
(file) and compound scale are independent axes. The adapter exposes both:
`load_icsd_yolo(scale="n", arch="x")`.

## Findings: the paper's reported complexity is not reproducible

The paper's only published compute figure is ICSD-YOLO-X at **116.8 GFLOPs**
(−41.3% vs YOLOv12-X's 199G). Our restored x-arch at x-scale measures
**121.3M params / 486.0 GFLOPs** — 4× the paper figure. No reconstruction
choice we tested closes this gap: the focus blocks as released are far heavier
than the paper implies, and their lighter internal configuration
(`ctx_dim=dim//4`) crashes. The scale-collapse defect (#6) means the authors'
own training runs would have silently used n-scale regardless of the file
name, further disconnecting the release from the paper's tables.

Measured restored variants (640px):

| arch | scale | params | GFLOPs |
|------|-------|--------|--------|
| x | n | 5.67M | 17.7 |
| x | s | 22.4M | 66.2 |
| x | x | 121.3M | 486.0 |
| s | n | 5.64M | 16.1 |
| s | s | 20.97M | 54.4 |
| m | n | 2.99M | — |

(For reference: stock YOLO11n = 2.6M / 6.3 GFLOPs, YOLO11s = 9.4M / 21.7.)

## Decision

1. **Base detector**: restored ICSD-YOLO, **x-arch @ n-scale** (5.67M / 17.7
   GFLOPs) — the only architecture matching the paper's description, at the
   size class closest to our YOLO11n baseline for a same-budget comparison.
2. **Framing in the manuscript**: we describe our detector as a *restored*
   ICSD-YOLO and cite this catalogue. We do **not** claim to reproduce the
   published ICSD-YOLO results; the release does not permit it (no weights,
   non-running code, complexity mismatch). The restoration itself is
   reproducibility work and will be noted as such.
3. **Baselines unaffected**: YOLO11n (pretrained) remains the practical
   baseline; the ablation ladder (Baseline → +WPAM → +TSMM → +SCRM → full)
   is backbone-agnostic by design (Week 1 detection contract), so none of our
   novelty claims rest on ICSD-YOLO's specific numbers.

## Operational caveat: Apple MPS

The focus blocks use adaptive pooling with non-divisible output sizes, which
PyTorch has not implemented on MPS (pytorch#96056). Training launchers set
`PYTORCH_ENABLE_MPS_FALLBACK=1` (before importing torch) so the affected op
runs on CPU; the throughput cost is measured by `scripts/probe_throughput.py`
and feeds the local-vs-cloud decision.

## Consequences

- Anyone cloning our repo gets a working ICSD-YOLO by importing
  `training.icsd` — no manual submodule surgery; patches fail loudly if the
  pinned commit drifts.
- Reviewer-facing risk: "your base detector is not the published ICSD-YOLO."
  Accepted and pre-empted by this ADR and the manuscript framing; the
  alternative (silently training a broken or unscaled model) is worse.
- If the authors ever publish weights or corrected code, the adapter's
  variant/scale flags let us swap without touching research modules.
