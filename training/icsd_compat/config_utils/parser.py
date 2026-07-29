"""Argv-safe delegate for the fork's ``config_utils.parser`` (release gap 11)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ORIGINAL = (
    Path(__file__).resolve().parents[3]
    / "external"
    / "icsd_yolo"
    / "config_utils"
    / "parser.py"
)
_spec = importlib.util.spec_from_file_location("_icsd_original_config_parser", _ORIGINAL)
assert _spec is not None and _spec.loader is not None
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)


def parse_args():
    """Return the fork's default argparse.Namespace without touching real argv."""
    saved_argv = sys.argv
    sys.argv = saved_argv[:1]
    try:
        return _original.parse_args()
    finally:
        sys.argv = saved_argv
