"""
Scoring for eval runs: order accuracy, cost per conversation, latency.

Definitions (also shown on the dashboard):

order accuracy
    Per conversation run with a REVIEWED label. Correct when:
      - the label expects no order and none was created, or
      - the label expects an order and exactly one was created whose
        product, karat and quantity match the label, and whose store
        total matched what the customer was quoted (a mismatch is the
        totals guardrail firing, which means the order is not right).
    Reported as correct / scored with a 95% Wilson interval, because the
    eval set is small and a bare percentage would overstate certainty.

cost per conversation
    Sum over the tool-selection model calls in the conversation:
    (uncached input x input price + cached input x cached price +
    output x output price), using scripts/evalset/pricing.json. Unknown
    (None) when any call lacks token counts. Covers only that model
    call: embeddings, vision, voice and geocoding are not included.

latency
    Per customer turn, measured inside router.route_customer() from the
    turn lock being taken to the result being returned. Split into model
    time (sum of model call latencies) and the rest. The first turn of a
    process also pays one-off start-up costs (catalogue loading), so
    p50 is a better summary than the mean.

consistency
    With --repeat > 1, the share of a conversation's runs that ended in
    the most common outcome (same orders, same handoff). 1.0 means every
    run ended the same way.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

PRICING_PATH = Path(__file__).with_name("pricing.json")


def load_pricing(path: Path = PRICING_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_prices(pricing: dict, model: str | None) -> dict | None:
    models = pricing.get("models", {})
    if not model:
        return None
    if model in models:
        return models[model]
    for name in sorted(models, key=len, reverse=True):
        if model.startswith(name):
            return models[name]
    return None


def call_cost_usd(call: dict, pricing: dict) -> float | None:
    prices = _model_prices(pricing, call.get("model"))
    input_tokens, output_tokens = call.get("input_tokens"), call.get("output_tokens")
    if prices is None or input_tokens is None or output_tokens is None:
        return None
    cached = call.get("cached_tokens") or 0
    uncached = max(input_tokens - cached, 0)
    return (uncached * prices["input"] + cached * prices["cached_input"] + output_tokens * prices["output"]) / 1e6


def _calls_for_cost(run_mode: str, turn: dict) -> list[dict]:
    if run_mode == "recorded":
        return (turn.get("replay") or {}).get("original_llm_calls", [])
    return turn.get("llm_calls", [])


def conversation_cost(run: dict, run_mode: str, pricing: dict) -> float | None:
    total = 0.0
    any_call = False
    for turn in run["turns"]:
        for call in _calls_for_cost(run_mode, turn):
            any_call = True
            cost = call_cost_usd(call, pricing)
            if cost is None:
                return None
            total += cost
    if not any_call:
        # No model call at all is only "free" if the model genuinely wasn't
        # needed; in recorded mode with no logged tokens it means unknown.
        needed = any((t.get("replay") or {}).get("model_called_originally") for t in run["turns"])
        return None if (run_mode == "recorded" and needed) else 0.0
    return total


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def _norm(value) -> str:
    return " ".join(str(value or "").lower().replace(",", " ").split())


def judge_order(run: dict, label: dict) -> dict:
    expected = label.get("expected_order")
    orders = run["orders"]
    total_mismatch = any(
        e.get("event") == "escalated" and e.get("reason") == "total_mismatch"
        for t in run["turns"] for e in t.get("guardrail_events", [])
    )
    verdict = {"expected": expected, "created": orders, "fields": {}}
    if expected is None:
        verdict["correct"] = not orders
        verdict["why"] = "no order expected" + ("" if not orders else f", but {len(orders)} created")
        return verdict
    if len(orders) != 1:
        verdict["correct"] = False
        verdict["why"] = f"expected one order, {len(orders)} created"
        return verdict
    order = orders[0]
    fields = {
        "product": _norm(order.get("product")) == _norm(expected.get("product")),
        "material": _norm(order.get("material")) == _norm(expected.get("material")),
        "quantity": str(order.get("quantity")) == str(expected.get("quantity")),
    }
    verdict["fields"] = fields
    verdict["correct"] = all(fields.values()) and not total_mismatch
    wrong = [k for k, ok in fields.items() if not ok]
    if total_mismatch:
        wrong.append("store total differed from quote")
    verdict["why"] = "matches label" if verdict["correct"] else "wrong: " + ", ".join(wrong)
    return verdict


def _outcome_signature(run: dict) -> str:
    orders = sorted((_norm(o.get("product")), _norm(o.get("material")), str(o.get("quantity"))) for o in run["orders"])
    return json.dumps({"orders": orders, "handoff": (run.get("escalation") or {}).get("reason")})


def score(report: dict, labels: dict, pricing: dict | None = None) -> dict:
    pricing = pricing or load_pricing()
    mode = report["mode"]
    per_run = []
    turn_latencies, model_latencies, app_latencies = [], [], []
    events = Counter()
    divergences = 0
    missing_answers = 0
    embeddings_blocked = 0
    crashes = 0

    for run in report["runs"]:
        label = labels.get(run["conversation_id"])
        reviewed = bool(label and label.get("reviewed"))
        verdict = judge_order(run, label) if reviewed else None
        handoff_ok = None
        if reviewed and label.get("expect_handoff") is not None:
            handoff_ok = bool(run.get("escalation")) == bool(label["expect_handoff"])

        for turn in run["turns"]:
            turn_latencies.append(turn["turn_latency_ms"])
            model_ms = sum(c.get("latency_ms") or 0 for c in turn.get("llm_calls", []))
            model_latencies.append(model_ms)
            app_latencies.append(max(turn["turn_latency_ms"] - model_ms, 0))
            for e in turn.get("guardrail_events", []):
                key = e["event"] + (f":{e['reason']}" if e.get("reason") else "")
                events[key] += 1
            replay = turn.get("replay")
            if replay:
                if replay["missing_model_answer"]:
                    missing_answers += 1
                if replay["original_result_kind"] != turn["result_kind"]:
                    divergences += 1
            if turn.get("embeddings_blocked"):
                embeddings_blocked += 1
            if turn.get("error"):
                crashes += 1

        conv_latency = sum(t["turn_latency_ms"] for t in run["turns"])
        per_run.append({
            "conversation_id": run["conversation_id"],
            "run": run["run"],
            "source": run.get("source"),
            "turn_count": len(run["turns"]),
            "reviewed": reviewed,
            "order": verdict,
            "handoff_ok": handoff_ok,
            "escalation": run.get("escalation"),
            "cost_usd": conversation_cost(run, mode, pricing),
            "latency_ms_total": round(conv_latency, 1),
            "signature": _outcome_signature(run),
        })

    scored = [r for r in per_run if r["order"] is not None]
    correct = sum(1 for r in scored if r["order"]["correct"])
    field_totals = Counter()
    field_correct = Counter()
    for r in scored:
        for field, ok in r["order"]["fields"].items():
            field_totals[field] += 1
            field_correct[field] += int(ok)

    consistency = {}
    by_conv: dict[str, list[str]] = {}
    for r in per_run:
        by_conv.setdefault(r["conversation_id"], []).append(r["signature"])
    for conv_id, sigs in by_conv.items():
        if len(sigs) > 1:
            consistency[conv_id] = Counter(sigs).most_common(1)[0][1] / len(sigs)

    costs = [r["cost_usd"] for r in per_run if r["cost_usd"] is not None]
    handoff_scored = [r for r in per_run if r["handoff_ok"] is not None]

    return {
        "mode": mode,
        "model": report.get("model"),
        "repeat": report.get("repeat"),
        "started_utc": report.get("started_utc"),
        "conversations": len(by_conv),
        "runs": len(per_run),
        "unreviewed_conversations": sorted({r["conversation_id"] for r in per_run if not r["reviewed"]}),
        "order_accuracy": {
            "correct": correct,
            "scored": len(scored),
            "rate": correct / len(scored) if scored else None,
            "ci95": wilson_interval(correct, len(scored)),
            "by_field": {f: {"correct": field_correct[f], "of": field_totals[f]} for f in field_totals},
        },
        "handoff_accuracy": {
            "correct": sum(1 for r in handoff_scored if r["handoff_ok"]),
            "scored": len(handoff_scored),
        },
        "cost_usd": {
            "measured_runs": len(costs),
            "of_runs": len(per_run),
            "median": percentile(costs, 50),
            "p95": percentile(costs, 95),
            "mean": sum(costs) / len(costs) if costs else None,
            "total": sum(costs) if costs else None,
            "pricing_note": pricing.get("_checked"),
        },
        "latency_ms": {
            "turns": len(turn_latencies),
            "turn_p50": percentile(turn_latencies, 50),
            "turn_p95": percentile(turn_latencies, 95),
            "turn_max": max(turn_latencies) if turn_latencies else None,
            "model_p50": percentile(model_latencies, 50),
            "app_p50": percentile(app_latencies, 50),
            "app_p95": percentile(app_latencies, 95),
        },
        "consistency": consistency,
        "guardrail_events": dict(events),
        "replay": {
            "result_kind_divergences": divergences,
            "missing_model_answers": missing_answers,
        } if mode == "recorded" else None,
        "embeddings_blocked_turns": embeddings_blocked,
        "crashed_turns": crashes,
        "per_run": per_run,
    }
