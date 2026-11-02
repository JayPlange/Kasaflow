"""Tests for scripts/evalset: trace parsing, anonymisation, replay, scoring, dashboard."""

import json
import uuid

import pytest

from scripts.evalset import anonymise, score as score_mod, traces
from scripts.evalset.dashboard import render
from scripts.evalset.run import RecordedModel, run_conversation
from scripts.evalset.sandbox import Sandbox


def _trace(session, message, tool="converse", final=None, source=None):
    output = {"tool": tool, "arguments": {}}
    if source:
        output["_source"] = source
    return {"session_id": session, "customer_message": message, "llm_structured_output": output,
            "final_result": final or {"conversation_reply": "hi"}}


# --- traces -------------------------------------------------------------------

def test_wrapped_powershell_log_is_parsed():
    record = json.dumps(_trace("s1", "hello there"))
    # Hard-wrap at 30 characters with CRLF, the way Tee-Object writes it.
    wrapped = "\r\n".join(record[i:i + 30] for i in range(0, len(record), 30))
    text = "noise line\r\nservices.router | KASAFLOW_TURN_TRACE " + wrapped + "\r\nmore noise"
    records, failures = traces.extract_records(text)
    assert failures == 0
    assert records[0]["data"]["customer_message"] == "hello there"


def test_utf16_log_file_is_read(tmp_path):
    path = tmp_path / "log.txt"
    path.write_bytes(("KASAFLOW_TURN_TRACE " + json.dumps(_trace("s", "yo"))).encode("utf-16"))
    records, _ = traces.extract_records(traces.read_log_text(path))
    assert records[0]["data"]["customer_message"] == "yo"


def test_multi_tool_turn_and_metrics_group_into_one_turn():
    records = [
        {"kind": "trace", "data": _trace("s", "price of ring and chain", "get_product_price")},
        {"kind": "trace", "data": _trace("s", "price of ring and chain", "get_product_price")},
        {"kind": "metrics", "data": {"session_id": "s", "turn_latency_ms": 900}},
        {"kind": "trace", "data": _trace("s", "thanks")},
        {"kind": "metrics", "data": {"session_id": "s", "turn_latency_ms": 400}},
    ]
    turns = traces.group_turns(records)
    assert [len(t.traces) for t in turns] == [2, 1]
    assert [t.metrics["turn_latency_ms"] for t in turns] == [900, 400]


def test_fast_path_and_guardrail_turns_have_no_model_output():
    assert traces.model_output_of(_trace("s", "14", "propose_order", source="awaiting_field:material")) is None
    assert traces.model_output_of({"llm_structured_output": {"tool": "guardrail:escalate"}}) is None
    assert traces.model_output_of({"llm_structured_output": {"error": "bad json"}}) is None
    assert traces.model_output_of(_trace("s", "hi"))["tool"] == "converse"


# --- anonymise ----------------------------------------------------------------

@pytest.mark.parametrize("text, must_not_contain", [
    ("call me on 024 123 4567", "4567"),
    ("my number is +233241234567", "241234567"),
    ("email kofi.mensah@example.com", "kofi.mensah"),
    ("my GPS is GA-183-8164", "GA-183-8164"),
    ("house 12, near the market", "house 12"),
    ("my name is Akosua Mensah", "Akosua"),
])
def test_personal_details_are_removed(text, must_not_contain):
    out = anonymise._scrub_text(text, anonymise.Redactions())
    assert must_not_contain not in out


@pytest.mark.parametrize("text", [
    "no 5 please", "2 of them", "this is Gye Nyame right?", "18k gold ring, 14g", "order #7001 status",
])
def test_ordinary_shop_text_is_kept(text):
    assert anonymise._scrub_text(text, anonymise.Redactions()) == text


def test_addresses_keep_known_places_only():
    log = anonymise.Redactions()
    assert anonymise._scrub_address("No 12 Ofankor Rd, East Legon", log) == "[ADDRESS] East Legon"
    assert anonymise._scrub_address("deliver to Kasoa", log) == "Kasoa"
    assert anonymise._scrub_address("unknown", log) == "unknown"
    assert anonymise._scrub_address("behind the blue kiosk", log) == "[ADDRESS]"


def test_build_conversations_hashes_sessions_and_reconstructs_multi_requests(tmp_path):
    phone = "233241234567"
    lines = [
        _trace(phone, "my name is Ama, price of ring and chain", "get_product_price",
               final={"price": 100}),
        _trace(phone, "my name is Ama, price of ring and chain", "get_product_price",
               final={"price": 200}),
        _trace(phone, "18", "propose_order", source="awaiting_field:material",
               final={"error": "How many?"}),
    ]
    log = tmp_path / "log.jsonl"
    log.write_text("".join("x | KASAFLOW_TURN_TRACE " + json.dumps(l) + "\n" for l in lines))

    conversations, redactions, stats = anonymise.build_conversations([log], "salt", "pilot")

    assert stats["turns"] == 2 and stats["parse_failures"] == 0
    conv = conversations[0]
    assert phone not in json.dumps(conv)
    assert "Ama" not in json.dumps(conv)
    assert conv["id"] == anonymise.session_hash(phone, "salt") != anonymise.session_hash(phone, "other")
    first, second = conv["turns"]
    assert len(first["recorded"]["model_output"]["requests"]) == 2
    assert second["recorded"]["model_called"] is False


def test_merge_never_overwrites_existing_conversations(tmp_path):
    path = tmp_path / "c.jsonl"
    anonymise.merge_into(path, [{"id": "a", "turns": [], "note": "original"}])
    added, total = anonymise.merge_into(path, [{"id": "a", "turns": [], "note": "new"}, {"id": "b", "turns": []}])
    assert (added, total) == (1, 2)
    assert json.loads(path.read_text().splitlines()[0])["note"] == "original"


# --- replay in the sandbox ----------------------------------------------------

def _catalogue_row():
    from app.config import settings

    rows = json.loads(settings.products_path.read_text())
    return next(r for r in rows if r.get("variation_id") and r.get("price"))


def _order_conversation(row):
    order_args = {
        "product_name": row["product"], "material": row["material"], "quantity": 2,
        "delivery_address": "East Legon", "delivery_option": "accra_rider",
    }
    return {
        "id": f"conv-test-{uuid.uuid4().hex[:6]}",
        "source": "synthetic",
        "turns": [
            {"customer_message": f"2 {row['product']} in {row['material']} to East Legon",
             "recorded": {"model_called": True, "model_output": {"tool": "propose_order", "arguments": order_args},
                          "result_kind": "proposal"}},
            {"customer_message": "CONFIRM",
             "recorded": {"model_called": False, "model_output": None, "result_kind": "order_confirmation"}},
        ],
    }


@pytest.fixture
def recorded_router(monkeypatch):
    from services import router

    model = RecordedModel()
    monkeypatch.setattr(router, "understand_customer", model)
    return model


def test_recorded_replay_places_order_in_sandbox_only(recorded_router, monkeypatch):
    from services import order_tool

    def no_network(*a, **k):
        raise AssertionError("real HTTP attempted")

    monkeypatch.setattr("requests.post", no_network)
    row = _catalogue_row()
    conv = _order_conversation(row)

    with Sandbox(allow_embeddings=False, allow_geocoding=False) as sandbox:
        run = run_conversation(conv, 0, "recorded", sandbox, recorded_router)
        assert sandbox.store.calls[0][0] == "POST"
    import requests as real_requests

    assert order_tool.requests is real_requests  # sandbox restored on exit

    assert [t["result_kind"] for t in run["turns"]] == ["proposal", "order_confirmation"]
    assert run["turns"][1]["replay"]["model_called_now"] is False
    order = run["orders"][0]
    assert (order["product"], order["material"], order["quantity"]) == (row["product"], row["material"], 2)
    assert float(order["store_total"]) == pytest.approx(row["price"] * 2)
    assert run["staff_messages"] == 1  # new-order notification, captured


def test_replay_marks_turns_that_now_need_a_model_answer(recorded_router):
    conv = {"id": "conv-missing", "turns": [
        {"customer_message": "tell me something nice",
         "recorded": {"model_called": False, "model_output": None, "result_kind": "conversation_reply"}},
    ]}
    with Sandbox(allow_embeddings=False, allow_geocoding=False) as sandbox:
        run = run_conversation(conv, 0, "recorded", sandbox, recorded_router)
    assert run["turns"][0]["replay"]["missing_model_answer"] is True


# --- scoring ------------------------------------------------------------------

def _run(conv_id, orders, turns=None, escalation=None, run_index=0):
    return {"conversation_id": conv_id, "run": run_index, "source": "pilot", "orders": orders,
            "escalation": escalation, "turns": turns or [{"turn_latency_ms": 1500, "result_kind": "proposal",
                                                          "llm_calls": [], "guardrail_events": []}]}


ORDER = {"product": "Gold Ring, 14g", "material": "18k", "quantity": 2, "store_total": "100.00"}


def test_judge_order_cases():
    label = {"reviewed": True, "expected_order": {"product": "gold ring 14g", "material": "18K", "quantity": 2}}
    assert score_mod.judge_order(_run("a", [ORDER]), label)["correct"]
    wrong_karat = {**ORDER, "material": "14k"}
    verdict = score_mod.judge_order(_run("a", [wrong_karat]), label)
    assert not verdict["correct"] and "material" in verdict["why"]
    assert not score_mod.judge_order(_run("a", []), label)["correct"]
    assert not score_mod.judge_order(_run("a", [ORDER, ORDER]), label)["correct"]
    mismatch_turns = [{"turn_latency_ms": 1, "result_kind": "order_confirmation", "llm_calls": [],
                       "guardrail_events": [{"event": "escalated", "reason": "total_mismatch"}]}]
    assert not score_mod.judge_order(_run("a", [ORDER], mismatch_turns), label)["correct"]
    none_label = {"reviewed": True, "expected_order": None}
    assert score_mod.judge_order(_run("b", []), none_label)["correct"]
    assert not score_mod.judge_order(_run("b", [ORDER]), none_label)["correct"]


def test_cost_uses_cached_and_uncached_prices():
    pricing = {"models": {"gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60}}}
    call = {"model": "gpt-4.1-mini-2025-04-14", "input_tokens": 3000, "cached_tokens": 2000, "output_tokens": 50}
    expected = (1000 * 0.40 + 2000 * 0.10 + 50 * 1.60) / 1e6
    assert score_mod.call_cost_usd(call, pricing) == pytest.approx(expected)
    assert score_mod.call_cost_usd({**call, "input_tokens": None}, pricing) is None
    assert score_mod.call_cost_usd({**call, "model": "unknown-model"}, pricing) is None


def test_wilson_interval_is_wide_for_small_samples():
    low, high = score_mod.wilson_interval(1, 2)
    assert low < 0.15 and high > 0.85
    assert score_mod.wilson_interval(0, 0) is None


def test_score_excludes_unreviewed_and_measures_consistency():
    report = {"mode": "live", "repeat": 2, "runs": [
        _run("a", [ORDER], run_index=0), _run("a", [], run_index=1), _run("u", [ORDER]),
    ]}
    labels = {"a": {"reviewed": True, "expected_order": {"product": ORDER["product"], "material": "18k", "quantity": 2}},
              "u": {"reviewed": False, "expected_order": None}}
    result = score_mod.score(report, labels, {"models": {}})
    assert result["order_accuracy"]["scored"] == 2 and result["order_accuracy"]["correct"] == 1
    assert result["unreviewed_conversations"] == ["u"]
    assert result["consistency"]["a"] == 0.5


def test_dashboard_renders_and_escapes():
    report = {"mode": "live", "repeat": 1, "started_utc": "20261102", "runs": [
        _run("conv-x", [], [{"index": 0, "customer_message": "<script>alert(1)</script>", "reply": "hi",
                             "result_kind": "conversation_reply", "turn_latency_ms": 1200,
                             "llm_calls": [{"latency_ms": 900, "input_tokens": 10, "output_tokens": 2}],
                             "guardrail_events": []}]),
    ]}
    scores = score_mod.score(report, {}, {"models": {}})
    page = render(report, scores)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "Order accuracy" in page and "not counted in order accuracy" in page
