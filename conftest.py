"""
Shared pytest configuration.

Two things this file does: add a --run-regression command-line flag
(skipping anything marked @pytest.mark.regression unless that flag is
passed, so `pytest` stays fast/free/offline while `pytest --run-regression`
opts into the slow, costly, real-API tests), and reset the app's
in-memory rate limiter before every test.

The rate limiter (app/main.py, slowapi, keyed by remote address) is a
module-level singleton that persists for the life of the test process --
TestClient always uses the same fake remote address, so every test in
the suite that hits /process shares ONE rate-limit budget (20/minute by
default) rather than each test getting its own. Enough golden-conversation
tests across enough files, each sending several turns, adds up to more
than 20 real requests within the same 60-second test run and starts
failing with 429s that have nothing to do with what's actually being
tested. Resetting before each test gives every test its own clean budget,
matching what a real production customer session would actually see
(one customer's messages, not the whole test suite's, count against
their limit)."""

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


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give every test its own clean rate-limit budget -- see this
    module's docstring. Import app.main lazily, inside the fixture, so
    test files that never touch the FastAPI app at all (the large
    majority) don't pay for importing it, and so a test environment
    missing config this app needs at import time doesn't break
    collection of unrelated tests."""
    try:
        from app.main import limiter
    except Exception:
        yield
        return
    limiter.reset()
    yield
