"""
Shared pytest configuration.

The only thing this file does: add a --run-regression command-line flag,
and automatically skip anything marked @pytest.mark.regression unless
that flag is passed. This is what keeps `pytest` (your everyday command)
fast, free, and offline, while still letting `pytest --run-regression`
opt in to the slow, costly, real-API tests when you actually want them.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-regression",
        action="store_true",
        default=False,
        help="Run prompt regression tests that call the real OpenAI API (costs money, requires network).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-regression"):
        return  # flag was passed, run everything including regression tests

    skip_regression = pytest.mark.skip(
        reason="Regression tests hit the real OpenAI API. Run with 'pytest --run-regression' to include them."
    )
    for item in items:
        if "regression" in item.keywords:
            item.add_marker(skip_regression)
