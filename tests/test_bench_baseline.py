"""
Tests for scripts/bench/ (stream_probe.py and baseline.py).

No real API calls. These pin down the definitions the benchmark reports:
TTFT is the first NON-EMPTY text delta (not the first event), decode time
is first to last text delta, concurrency is genuinely parallel, and
success is split into separate layers.
"""

import threading
import time
from types import SimpleNamespace

from scripts.bench import baseline
from scripts.bench.baseline import Case
from scripts.bench.stream_probe import Gauge, probe_once
from services.llm import ToolSelectionError


def _event(type_, **fields):
    return SimpleNamespace(type=type_, **fields)


def _completed(inp=1000, out=41, cached=768):
    usage = SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    return _event("response.completed", response=SimpleNamespace(usage=usage))


class _FakeClient:
    """responses.create(stream=True) returns a generator with timed events."""

    def __init__(self, script):
        self._script = script
        self.responses = SimpleNamespace(create=self._create)
        self.kwargs = None

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self._script()


def _timed_script(first_text_after=0.06, gap=0.03):
    def script():
        yield _event("response.created")
        yield _event("response.in_progress")
        yield _event("response.output_item.added")
        yield _event("response.content_part.added")
        time.sleep(first_text_after)
        yield _event("response.output_text.delta", delta="")  # empty: must not count
        yield _event("response.output_text.delta", delta='{"tool"')
        time.sleep(gap)
        yield _event("response.output_text.delta", delta=': "x"}')
        yield _event("response.output_text.done")
        yield _completed()
    return script


# ---------------------------------------------------------------------
# stream_probe
# ---------------------------------------------------------------------

def test_ttft_is_first_non_empty_text_not_first_event():
    client = _FakeClient(_timed_script(first_text_after=0.06, gap=0.03))

    row = probe_once(client, "m", "prompt", timeout=5)

    assert client.kwargs["stream"] is True
    assert row["outcome"] == "ok"
    assert row["first_event_ms"] < 20
    assert row["ttft_ms"] >= 55
    assert row["ttft_ms"] > row["first_event_ms"] + 40
    assert row["decode_ms"] >= 25
    assert row["total_ms"] >= row["ttft_ms"] + row["decode_ms"] - 1
    assert row["n_text_events"] == 2  # the empty delta is not counted
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 41
    assert row["cached_tokens"] == 768
    assert row["terminal_event"] == "response.completed"


def test_tokens_per_second_uses_decode_interval_only():
    row = probe_once(_FakeClient(_timed_script(0.05, 0.05)), "m", "p", timeout=5)

    decode_s = row["decode_ms"] / 1000
    assert abs(row["output_tps_decode"] - 41 / decode_s) / row["output_tps_decode"] < 0.02
    assert abs(row["output_tps_decode_excl_first"] - 40 / decode_s) / row["output_tps_decode_excl_first"] < 0.02


def test_keep_events_records_the_timeline():
    row = probe_once(_FakeClient(_timed_script(0.01, 0.01)), "m", "p", timeout=5, keep_events=True)

    types = [e["type"] for e in row["events"]]
    assert types[0] == "response.created"
    assert "response.output_text.delta" in types
    assert types[-1] == "response.completed"


def test_stream_with_no_text_is_not_a_success():
    def script():
        yield _event("response.created")
        yield _completed(out=0)

    row = probe_once(_FakeClient(script), "m", "p", timeout=5)

    assert row["outcome"] == "no_text"
    assert row["ttft_ms"] is None


def test_probe_classifies_exceptions_and_releases_gauge():
    def create(**_kwargs):
        raise ConnectionResetError("reset")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    gauge = Gauge()

    row = probe_once(client, "m", "p", timeout=5, gauge=gauge)

    assert row["outcome"] == "other_error"
    assert row["error_type"] == "ConnectionResetError"
    assert gauge.enter() == 1  # back to zero in flight before this call


# ---------------------------------------------------------------------
# load generation
# ---------------------------------------------------------------------

def _peak_task(gauge, peaks, hold=0.12):
    def task(item):
        peaks.append(gauge.enter())
        time.sleep(hold)
        gauge.leave()
        return {"item": item}
    return task


def test_closed_loop_reaches_the_stated_concurrency():
    gauge, peaks = Gauge(), []

    rows = baseline.run_closed_loop(_peak_task(gauge, peaks), list(range(8)), 4)

    assert max(peaks) == 4
    assert [r["item"] for r in rows] == list(range(8))  # results keep item order


def test_closed_loop_at_one_never_overlaps():
    gauge, peaks = Gauge(), []

    baseline.run_closed_loop(_peak_task(gauge, peaks, hold=0.02), list(range(4)), 1)

    assert max(peaks) == 1


def test_closed_loop_all_workers_start_together():
    starts = []
    lock = threading.Lock()

    def task(_item):
        with lock:
            starts.append(time.perf_counter())
        time.sleep(0.15)
        return {}

    baseline.run_closed_loop(task, list(range(8)), 8)

    assert max(starts) - min(starts) < 0.1


# ---------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------

def test_percentile_is_nearest_rank():
    data = [10.0, 20.0, 30.0, 40.0]
    assert baseline.percentile(data, 50) == 20.0
    assert baseline.percentile(data, 95) == 40.0
    assert baseline.percentile([7.0], 99) == 7.0


def test_summarise_ignores_none_and_handles_empty():
    assert baseline.summarise([None, None]) is None
    stat = baseline.summarise([30.0, None, 10.0, 20.0])
    assert (stat["n"], stat["min"], stat["p50"], stat["max"]) == (3, 10.0, 20.0, 30.0)


# ---------------------------------------------------------------------
# success layers
# ---------------------------------------------------------------------

def _collector_with(attempts):
    collector = baseline.AttemptCollector()
    for attempt in attempts:
        collector(dict(attempt, thread_id=threading.get_ident()))
    return collector


def test_prod_task_separates_api_success_from_parse_success():
    collector = baseline.AttemptCollector()
    outputs = {"good": '{"tool": "converse", "arguments": {}}', "unparseable": "not json"}

    def call_llm(message, *_):
        return outputs[message]

    def parse(raw):
        if raw == "not json":
            raise ToolSelectionError("LLM did not return valid JSON")
        return {"tool": "converse", "arguments": {}}

    task = baseline.make_prod_task(call_llm, parse, collector, ToolSelectionError)

    good, bad = task(Case("s-good", "good")), task(Case("s-bad", "unparseable"))

    assert (good["api_ok"], good["valid_response"], good["parse_ok"]) == (True, True, True)
    assert good["tools"] == ["converse"]
    assert (bad["api_ok"], bad["valid_response"], bad["parse_ok"]) == (True, True, False)


def test_prod_task_api_failure_leaves_parse_unmeasured():
    collector = baseline.AttemptCollector()

    def call_llm(*_):
        raise ToolSelectionError("LLM unreachable after 3 attempts")

    row = baseline.make_prod_task(call_llm, lambda raw: {}, collector, ToolSelectionError)(Case("s", "x"))

    assert row["api_ok"] is False
    assert row["parse_ok"] is None


def test_prod_task_folds_attempt_records():
    collector = baseline.AttemptCollector()

    def call_llm(*_):
        collector({"thread_id": threading.get_ident(), "outcome": "timeout", "latency_ms": 15000.0,
                   "in_flight_at_start": 1})
        collector({"thread_id": threading.get_ident(), "outcome": "ok", "latency_ms": 900.0,
                   "input_tokens": 2000, "output_tokens": 30, "cached_tokens": 1024,
                   "output_chars": 60, "in_flight_at_start": 1})
        return '{"tool": "converse", "arguments": {}}'

    row = baseline.make_prod_task(call_llm, lambda raw: {"tool": "converse"}, collector,
                                  ToolSelectionError)(Case("s", "x"))

    assert row["llm_attempts"] == 2
    assert row["llm_outcomes"] == ["timeout", "ok"]
    assert row["llm_latency_ms"] == 15900.0
    assert row["final_attempt_latency_ms"] == 900.0
    assert row["cached_tokens"] == 1024


def test_e2e_task_reports_route_errors_and_turns_without_a_model_call():
    collector = baseline.AttemptCollector()

    ok = baseline.make_e2e_task(lambda m, s: {"reply": "hi"}, collector)(Case("s", "x"))
    err = baseline.make_e2e_task(lambda m, s: {"error": "Something went wrong"}, collector)(Case("s", "x"))

    assert ok["route_ok"] is True
    assert ok["llm_called"] is False
    assert ok["api_ok"] is None  # no model call happened, so it is not "failed"
    assert err["route_ok"] is False
    assert err["error_type"] == "route_error_result"


def test_e2e_task_separates_llm_time_from_the_rest_of_the_turn():
    collector = baseline.AttemptCollector()

    def route(message, session_id):
        collector({"thread_id": threading.get_ident(), "outcome": "ok", "latency_ms": 50.0,
                   "output_chars": 40, "in_flight_at_start": 1})
        time.sleep(0.2)
        return {"reply": "hi"}

    row = baseline.make_e2e_task(route, collector)(Case("s", "x"))

    assert row["api_ok"] is True and row["valid_response"] is True
    assert row["e2e_ms"] >= 200
    assert row["non_llm_ms"] >= 150


# ---------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------

def test_stream_phase_runs_warmup_then_ladder_with_same_messages_per_level():
    seen = []

    def probe(message):
        seen.append(message)
        return {"outcome": "ok", "in_flight_at_start": 1}

    cases = [Case(f"s{i}", f"m{i}") for i in range(8)]
    result = baseline.run_stream_phase(lambda c: probe(c.message), cases, warmup=3, levels=(1, 2), per_level=4)

    assert [r["seq"] for r in result["warmup"]] == [1, 2, 3]
    assert all(r["phase"] == "stream-first" for r in result["warmup"])
    assert sorted(result["ladder"]) == [1, 2]
    per_level = {lvl: sorted(r["seq"] for r in b["rows"]) for lvl, b in result["ladder"].items()}
    assert per_level == {1: [4, 5, 6, 7], 2: [8, 9, 10, 11]}
    level_messages = [sorted(seen[3:7]), sorted(seen[7:11])]
    assert level_messages[0] == level_messages[1] == ["m0", "m1", "m2", "m3"]


def test_default_plan_is_within_the_agreed_request_budget():
    args = baseline.parse_args([])
    plan = baseline.build_plan(args)

    assert plan["total_model_requests"] == 45
    assert 30 <= plan["total_model_requests"] <= 50


def test_case_pool_is_fixed_for_a_seed():
    assert baseline.pick_cases(6, seed=0) == baseline.pick_cases(6, seed=0)
    assert baseline.pick_cases(6, seed=0) != baseline.pick_cases(6, seed=1)


def test_every_case_has_a_scenario_id_and_a_message():
    for case in baseline.pick_cases(60, seed=0):
        assert case.scenario_id and case.message
        assert case.scenario_id != case.message


# ---------------------------------------------------------------------
# privacy: rows must not carry message text, model output or error text
# ---------------------------------------------------------------------

SECRET_MESSAGE = "SENTINEL-CUSTOMER-MESSAGE 0244 123 456"
SECRET_OUTPUT = "SENTINEL-MODEL-OUTPUT"


def _assert_clean(*rows):
    import json

    blob = json.dumps(rows, default=str)
    assert SECRET_MESSAGE not in blob
    assert SECRET_OUTPUT not in blob


def test_prod_rows_never_hold_message_text_or_model_output():
    collector = baseline.AttemptCollector()
    case = Case("scn-1", SECRET_MESSAGE)

    def parse_missing_keys(raw):
        # Mirrors llm.py, whose error text embeds the parsed model output.
        raise ToolSelectionError(f"LLM response missing 'tool' or 'arguments': {{'x': '{SECRET_OUTPUT}'}}")

    def api_fails(*_):
        raise ToolSelectionError(f"LLM request failed: {SECRET_MESSAGE}")

    parse_failed = baseline.make_prod_task(lambda m, *_: SECRET_OUTPUT, parse_missing_keys, collector,
                                           ToolSelectionError)(case)
    api_failed = baseline.make_prod_task(api_fails, parse_missing_keys, collector, ToolSelectionError)(case)
    unexpected = baseline.make_prod_task(lambda *_: (_ for _ in ()).throw(RuntimeError(SECRET_OUTPUT)),
                                         parse_missing_keys, collector, ToolSelectionError)(case)

    _assert_clean(parse_failed, api_failed, unexpected)
    assert parse_failed["scenario_id"] == "scn-1"
    assert parse_failed["error_type"] == "ToolSelectionError"
    assert unexpected["error_type"] == "RuntimeError"


def test_e2e_rows_never_hold_message_text_or_error_text():
    collector = baseline.AttemptCollector()
    case = Case("scn-2", SECRET_MESSAGE)

    returned_error = baseline.make_e2e_task(lambda m, s: {"error": f"{SECRET_OUTPUT} {m}"}, collector)(case)
    raised = baseline.make_e2e_task(lambda m, s: (_ for _ in ()).throw(ValueError(m)), collector)(case)
    fine = baseline.make_e2e_task(lambda m, s: {"reply": m}, collector)(case)

    _assert_clean(returned_error, raised, fine)
    assert returned_error["scenario_id"] == "scn-2"
    assert raised["error_type"] == "ValueError"


def test_stream_probe_rows_never_hold_text():
    def script():
        yield _event("response.created")
        yield _event("response.output_text.delta", delta=SECRET_OUTPUT)
        yield _completed()

    row = probe_once(_FakeClient(script), "m", f"prompt containing {SECRET_MESSAGE}", timeout=5, keep_events=True)

    _assert_clean(row)
    assert row["text_chars"] == len(SECRET_OUTPUT)  # length only
