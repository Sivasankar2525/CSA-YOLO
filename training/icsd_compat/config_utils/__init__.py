"""Argv-safe shadow of the ICSD-YOLO fork's ``config_utils`` package.

The fork calls ``parse_args()`` at import time (``models/utils/__init__.py``),
which parses ``sys.argv`` and crashes any embedding host whose argv it does
not recognise (pytest, notebooks, our launchers). This shadow package sits
earlier on ``sys.path`` than the submodule root and delegates to the original
parser with a neutralised argv, so only the fork's defaults are ever parsed.
Release gap 11; see ADR 0002.
"""
