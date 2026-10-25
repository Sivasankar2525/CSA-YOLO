"""Shared pytest configuration."""

import sys
from pathlib import Path

# Make the project root importable so ``evaluation`` and ``scripts`` resolve in tests.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: heavyweight tests (model instantiation); deselect with -m 'not slow'"
    )
