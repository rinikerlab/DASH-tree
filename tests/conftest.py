"""Pytest configuration file."""

import pytest
from collections.abc import Sequence


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tier",
        action="store",
        default="min",
        choices=("min", "med", "max"),
        help="Run tests for a specific environment tier.",
    )
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Include slow tests."
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "min: expected to pass with tree_ony_env.yml")
    config.addinivalue_line("markers", "med: expected to pass with min_environment.yml")
    config.addinivalue_line("markers", "max: expected to pass with environment.yml")
    config.addinivalue_line("markers", "slow: slow running tests")


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Item]
) -> None:
    tier = config.getoption("--tier")
    run_slow = config.getoption("--run-slow")

    allowed_tiers = {"min"}
    if tier == "med":
        allowed_tiers.update({"med"})
    elif tier == "max":
        allowed_tiers.update({"med", "max"})

    for item in items:
        item_tiers = {
            mark.name
            for mark in item.iter_markers()
            if mark.name in {"min", "med", "max"}
        }

        if item_tiers and not (item_tiers & allowed_tiers):
            item.add_marker(
                pytest.mark.skip(reason=f"Test requires higher tier than '{tier}'")
            )
            continue

        if "slow" in item.keywords and not run_slow:
            item.add_marker(pytest.mark.skip(reason="Slow test: use --run-slow to run"))
