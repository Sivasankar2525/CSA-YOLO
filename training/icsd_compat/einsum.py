"""Compat shim: the ICSD-YOLO fork does `from einsum import einsum`, a typo for
the einops function with identical call signature (tensors first, pattern last).
Placed on sys.path by the training adapter so the pinned submodule stays unmodified."""
from einops import einsum  # noqa: F401
