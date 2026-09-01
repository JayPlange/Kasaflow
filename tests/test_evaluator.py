"""
Unit tests for the KasaFlow behavioural evaluator (scripts/evaluator/).

None of these hit the real OpenAI API -- route_customer() is mocked
throughout, standing in for router.py by logging the same
KASAFLOW_TURN_TRACE line the real function would, so run_turn()'s
log-capture wiring is genuinely exercised (not just its comparison
logic against a hand-built dict). What these tests CANNOT prove is
whether a real scenario passes against the real model -- that needs a
real OPENAI_API_KEY and scripts/run_evaluator.py, not this file. See
scripts/evaluator/runner.py's own module docstring.
"""

import json
import logging

import pytest

from scripts.evaluator.report import print_summary, to_json
from scripts.evaluator.runner import run_scenario, run_turn
from scripts.evaluator.schema import CATEGORIES, Scenario, Turn


def _fake_route_customer(trace_overrides):
    """Builds a stand-in for services.router.route_customer that logs a
    KASAFLOW_TURN_TRACE line the same shape router.py's real
    _log_turn_trace() produces, so run_turn()'s capture handler has
    something real to catch."""
    router_logger = logging.getLogger("services.router")

    def _fake(message, session_id):
        trace = {
            "session_id": session_id,
            "customer_message": message,
            "pre_tool_state": {},
            "llm_structured_output": {"tool": "get_product_price", "arguments": {"product_name": "Ring"}},
            "resolved_arguments": {"product_name": "Ring", "material": "18k"},
            "tool_result": {"product": "Ring", "material": "18k", "price": 1200.0},
            "final_result": {"product": "Ring", "material": "18k", "price": 1200.0},
            "post_tool_state": {},
        }
        trace.update(trace_overrides)
        router_logger.info("KASAFLOW_TURN_TRACE %s", json.dumps(trace))
        return trace["final_result"]

    return _fake


# ---------------------------------------------------------------------
# schema.py
# ---------------------------------------------------------------------

def test_scenario_rejects_unknown_category():
    with pytest.raises(ValueError):
        Scenario(id="x", category="NOT A REAL CATEGORY", description="d", turns=[Turn(message="hi")])


def test_scenario_rejects_empty_turns():
    with pytest.raises(ValueError):
        Scenario(id="x", category=CATEGORIES[0], description="d", turns=[])


def test_scenario_result_passed_is_false_when_error_set():
    from scripts.evaluator.schema import ScenarioResult

    scenario = Scenario(id="x", category=CATEGORIES[0], description="d", turns=[Turn(message="hi")])
    result = ScenarioResult(scenario=scenario, turn_results=[], error="boom")
    assert result.passed is False


# ---------------------------------------------------------------------
# runner.py -- run_turn()'s trace capture
# ---------------------------------------------------------------------

def test_run_turn_captures_the_logged_trace(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    trace = run_turn("how much is the ring", session_id="test-session")
    assert trace["customer_message"] == "how much is the ring"
    assert trace["llm_structured_output"]["tool"] == "get_product_price"


def test_run_turn_raises_when_nothing_was_logged(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        lambda message, session_id: {"product": "Ring"},  # never logs a trace
    )
    with pytest.raises(RuntimeError, match="No KASAFLOW_TURN_TRACE captured"):
        run_turn("how much is the ring", session_id="test-session")


def test_run_turn_does_not_leak_the_capture_handler(monkeypatch):
    # A handler left attached after run_turn() returns would silently
    # accumulate across every subsequent scenario in the same process,
    # and would keep capturing every REAL route_customer() call this
    # process ever makes afterwards -- both wrong. Confirm cleanup.
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    router_logger = logging.getLogger("services.router")
    handlers_before = list(router_logger.handlers)
    run_turn("hello", session_id="test-session")
    assert router_logger.handlers == handlers_before


# ---------------------------------------------------------------------
# runner.py -- run_scenario()'s comparison logic
# ---------------------------------------------------------------------

def test_run_scenario_passes_when_expectations_match(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    scenario = Scenario(
        id="unit-01",
        category="PRICE",
        description="unit test",
        turns=[Turn(message="how much is the ring", expected_tool="get_product_price", expected_fields={"material": "18k"})],
    )
    result = run_scenario(scenario)
    assert result.passed is True
    assert result.error is None


def test_run_scenario_fails_on_wrong_tool(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    scenario = Scenario(
        id="unit-02",
        category="PRICE",
        description="unit test",
        turns=[Turn(message="how much is the ring", expected_tool="propose_order")],
    )
    result = run_scenario(scenario)
    assert result.passed is False
    failing_checks = [c for tr in result.turn_results for c in tr.checks if not c.passed]
    assert any(c.label == "tool" for c in failing_checks)


def test_run_scenario_fails_on_wrong_field(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    scenario = Scenario(
        id="unit-03",
        category="PRICE",
        description="unit test",
        turns=[Turn(message="how much is the ring", expected_fields={"material": "14k"})],
    )
    result = run_scenario(scenario)
    assert result.passed is False


def test_run_scenario_checks_expected_contains_against_formatted_reply(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    scenario = Scenario(
        id="unit-04",
        category="PRICE",
        description="unit test",
        turns=[Turn(message="how much is the ring", expected_contains=["GH₵1,200.00"])],
    )
    result = run_scenario(scenario)
    assert result.passed is True


def test_run_scenario_expected_not_contains_catches_a_bad_reply(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    scenario = Scenario(
        id="unit-05",
        category="PRICE",
        description="unit test",
        turns=[Turn(message="how much is the ring", expected_not_contains=["GH₵1,200.00"])],
    )
    result = run_scenario(scenario)
    assert result.passed is False


def test_run_scenario_records_error_without_raising(monkeypatch):
    def _boom(message, session_id):
        raise RuntimeError("simulated LLM failure")

    monkeypatch.setattr("scripts.evaluator.runner.route_customer", _boom)
    scenario = Scenario(
        id="unit-06",
        category="PRICE",
        description="unit test",
        turns=[Turn(message="how much is the ring")],
    )
    result = run_scenario(scenario)
    assert result.passed is False
    assert "simulated LLM failure" in result.error


def test_run_scenario_shares_one_session_across_turns(monkeypatch):
    seen_session_ids = []

    def _fake(message, session_id):
        seen_session_ids.append(session_id)
        return _fake_route_customer({})(message, session_id)

    monkeypatch.setattr("scripts.evaluator.runner.route_customer", _fake)
    scenario = Scenario(
        id="unit-07",
        category="REFERENCES",
        description="unit test",
        turns=[Turn(message="first"), Turn(message="second")],
    )
    run_scenario(scenario)
    assert len(set(seen_session_ids)) == 1


def test_run_scenario_accepts_a_tuple_of_alternative_tools(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    scenario = Scenario(
        id="unit-08",
        category="PRICE",
        description="unit test",
        turns=[Turn(message="how much is the ring", expected_tool=("propose_order", "get_product_price"))],
    )
    result = run_scenario(scenario)
    assert result.passed is True


# ---------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------

def test_to_json_round_trips_and_leaves_manner_rating_unscored(monkeypatch):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    scenario = Scenario(
        id="unit-09",
        category="PRICE",
        description="unit test",
        turns=[Turn(message="how much is the ring", expected_tool="get_product_price", manner_note="should read naturally")],
    )
    result = run_scenario(scenario)
    payload = json.loads(to_json([result]))
    assert payload[0]["id"] == "unit-09"
    assert payload[0]["passed"] is True
    turn_payload = payload[0]["turns"][0]
    assert turn_payload["manner_note"] == "should read naturally"
    # Never fabricate an automated manner verdict -- see schema.py.
    assert turn_payload["manner_rating"] is None


def test_print_summary_does_not_raise(monkeypatch, capsys):
    monkeypatch.setattr(
        "scripts.evaluator.runner.route_customer",
        _fake_route_customer({}),
    )
    scenario = Scenario(
        id="unit-10",
        category="PRICE",
        description="unit test",
        turns=[Turn(message="how much is the ring", expected_tool="propose_order")],
    )
    result = run_scenario(scenario)
    print_summary([result])
    captured = capsys.readouterr()
    assert "unit-10" in captured.out
    assert "FAIL" in captured.out


# ---------------------------------------------------------------------
# scenarios.py -- the corpus itself is well-formed
# ---------------------------------------------------------------------

def test_scenario_corpus_covers_every_category_and_has_no_duplicate_ids():
    from scripts.evaluator.scenarios import SCENARIOS

    seen_categories = {s.category for s in SCENARIOS}
    assert seen_categories == set(CATEGORIES)

    ids = [s.id for s in SCENARIOS]
    assert len(ids) == len(set(ids))

    for scenario in SCENARIOS:
        assert scenario.source, f"{scenario.id} has no source"
        for turn in scenario.turns:
            assert turn.message.strip(), f"{scenario.id} has an empty turn message"
