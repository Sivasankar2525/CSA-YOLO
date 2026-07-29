"""Shared pytest configuration."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: heavyweight tests (model instantiation); deselect with -m 'not slow'"
    )
