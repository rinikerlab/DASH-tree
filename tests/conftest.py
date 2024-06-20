"""Pytest configuration file."""

from collections.abc import Sequence

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add cli option to include slow tests.

    Args:
        parser (pytest.Parser): pytest parser.
    """
    parser.addoption("--run-slow", action="store_true", default=False, help="Include slow tests.")


def pytest_configure(config: pytest.Config) -> None:
    """Add slow marker to pytest.

    Args:
        config (pytest.Config): pytest configuration.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: pytest.Config, items: Sequence[pytest.Item]) -> None:
    """Skip slow tests if not included.

    Args:
        config (pytest.Config): pytest configuration.
        items (Sequence[pytest.Item]): pytest items.
    """
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run.")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
