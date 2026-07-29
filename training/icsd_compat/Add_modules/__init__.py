"""Restoration of the `models.nn.Add_modules` package missing from the ICSD-YOLO release.

The authors' repository (PrintSC/code @ 47018ed) imports this package from
`ConvStemBlock.py` and `block.py`, but never published it. Every symbol it must
provide is *defined verbatim elsewhere in the same repository* (block.py lines
395-591 and conv.py) — the authors evidently inlined the definitions and forgot
to remove the stale imports. This package restores the import surface:

- `BatchNorm.py`: standalone copies of the definitions from block.py (copying,
  not re-exporting, avoids a circular import since block.py imports this module
  before executing its own definitions).
- `Conv.py`: re-exports from models.nn.modules.conv (loaded before this module).

Installed into the submodule by training/icsd.py; documented in ADR 0002.
"""

from models.nn.modules.conv import autopad  # noqa: F401
