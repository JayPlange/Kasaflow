"""
Runs Scenario objects (scripts/evaluator/schema.py) against the real
route_customer() pipeline and checks the result against each turn's
expectations.

Requires a real OPENAI_API_KEY (and, for scenarios using
delivery/geocoding, whatever else router.py's own dependencies need) --
this drives the actual LLM tool-selection call, same as a real customer
message would, not a mock. There is no offline mode for the pass/fail
result itself; see tests/test_evaluator.py for how the harness's own
comparison/reporting logic is unit tested instead, with route_customer()
mocked out.

Every scenario gets its own fresh session_id (a uuid4, not a customer
phone number) so scenarios never share memory state with each other or
with a real conversation, even when run against the same backing store
a live deployment uses.
"""

from __future__ import annotations

import json
import logging
import uuid

from scripts.evaluator.schema import CheckResult, Scenario, ScenarioResult, Turn, TurnResult
from services.response_formatter import format_for_customer
from services.router import route_customer

_TRACE_PREFIX = "KASAFLOW_TURN_TRACE "


class _TraceCapture(logging.Handler):
    """Captures router.py's own KASAFLOW_TURN_TRACE log lines (built
    2026-08-21) rather than re-deriving llm_structured_output/
    resolved_arguments by reaching into router.py's private functions a
    second time -- one source of truth for "what actually happened this
    turn", already battle-tested as the thing Webb reads to debug real
    failures."""

    def __init__(self) -> None:
        super().__init__()
        self.traces: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if not message.startswith(_TRACE_PREFIX):
            return
        try:
            self.traces.append(json.loads(message[len(_TRACE_PREFIX):]))
        except json.JSONDecodeError:
            # A malformed trace line is a router.py logging bug, not an
            # evaluator bug -- don't let it crash the run, just drop it
            # silently the same way _log_turn_trace() itself never lets
            # a logging failure touch the customer-facing turn.
            pass


def run_turn(message: str, session_id: str) -> dict:
    """Runs one turn through the real pipeline and returns the exact
    trace dict router.py logged for it. Raises RuntimeError if nothing
    was captured -- almost always means router.py's logger name changed
    or its level was raised above INFO somewhere else in the process,
    not that the turn silently did nothing."""
    router_logger = logging.getLogger("services.router")
    capture = _TraceCapture()
    capture.setLevel(logging.INFO)
    prior_level = router_logger.level
    prior_propagate = router_logger.propagate
    router_logger.addHandler(capture)
    router_logger.setLevel(logging.INFO)
    try:
        route_customer(message, session_id=session_id)
    finally:
        router_logger.removeHandler(capture)
        router_logger.setLevel(prior_level)
        router_logger.propagate = prior_propagate

    if not capture.traces:
        raise RuntimeError(
            f"No KASAFLOW_TURN_TRACE captured for message {message!r} -- "
            "is services.router's logger reachable at INFO level?"
        )
    # A multi-request message ("requests" plural, see llm.py) logs one
    # trace per sub-request plus none at the top level -- last one is
    # the most recent tool actually run this turn, matching what a
    # scenario asserting a single expected_tool means to check. Multi-
    # intent-in-one-message scenarios should be written as the intended
    # tool being the LAST distinct ask in the message, or should assert
    # on resolved_arguments/expected_contains instead of expected_tool.
    return capture.traces[-1]


def _final_text(trace: dict) -> str:
    # format_for_customer(None) is itself meaningful -- it's the "Hmm,
    # I couldn't find that one" no-match reply, a real, valid shape of
    # final_result, not an empty/nothing state. An earlier version of
    # this function short-circuited on final_result is None and
    # returned "" instead of calling format_for_customer, which quietly
    # broke every scenario checking a no-match reply's actual wording
    # (confirmed live, 2026-09-01, on ambiguity-03) -- always delegate,
    # exactly like router.py's own callers do.
    return format_for_customer(trace.get("final_result"))


def _check_turn(turn: Turn, trace: dict, final_text: str) -> list[CheckResult]:
    checks: list[CheckResult] = []
    llm_output = trace.get("llm_structured_output") or {}
    resolved_arguments = trace.get("resolved_arguments") or {}
    actual_tool = llm_output.get("tool")

    if turn.expected_tool is not None:
        expected = (
            turn.expected_tool if isinstance(turn.expected_tool, tuple) else (turn.expected_tool,)
        )
        checks.append(
            CheckResult(
                label="tool",
                passed=actual_tool in expected,
                detail=f"expected tool in {expected!r}, got {actual_tool!r}",
            )
        )

    if turn.expected_fields:
        for key, expected_value in turn.expected_fields.items():
            actual_value = resolved_arguments.get(key) if isinstance(resolved_arguments, dict) else None
            checks.append(
                CheckResult(
                    label=f"field:{key}",
                    passed=actual_value == expected_value,
                    detail=f"expected {key}={expected_value!r}, got {actual_value!r}",
                )
            )

    lowered_text = final_text.lower()
    for substring in turn.expected_contains:
        checks.append(
            CheckResult(
                label=f"contains:{substring!r}",
                passed=substring.lower() in lowered_text,
                detail=final_text,
            )
        )
    for substring in turn.expected_not_contains:
        checks.append(
            CheckResult(
                label=f"not_contains:{substring!r}",
                passed=substring.lower() not in lowered_text,
                detail=final_text,
            )
        )

    return checks


def run_scenario(scenario: Scenario, session_id: str | None = None) -> ScenarioResult:
    """Runs every turn in a scenario against ONE shared session, in
    order -- a multi-turn scenario is a single conversation, not several
    independent messages, so later turns see the memory state earlier
    turns left behind, same as a real customer would."""
    session_id = session_id or f"eval-{scenario.id}-{uuid.uuid4()}"
    turn_results: list[TurnResult] = []
    try:
        for turn in scenario.turns:
            trace = run_turn(turn.message, session_id)
            final_text = _final_text(trace)
            checks = _check_turn(turn, trace, final_text)
            turn_results.append(TurnResult(turn=turn, trace=trace, final_text=final_text, checks=checks))
    except Exception as e:  # noqa: BLE001 -- a scenario blowing up is itself a result, not a crash
        return ScenarioResult(scenario=scenario, turn_results=turn_results, error=str(e))
    return ScenarioResult(scenario=scenario, turn_results=turn_results)


def run_all(scenarios: list[Scenario]) -> list[ScenarioResult]:
    return [run_scenario(scenario) for scenario in scenarios]
