"""
Baseline measurements for how KasaFlow talks to its model. Benchmark only.

Three phases, all against the real API (they cost real requests):

  1. stream  The same prompt KasaFlow builds, sent with stream=True through
             stream_probe. The only phase that can give TTFT and decode
             time. First a few warm-up requests, then a concurrency ladder.
  2. prod    The production call itself (llm._call_llm, non-streaming),
             then llm._parse_tool_request, at concurrency 1. Shows whether
             the instrumentation matches the streaming numbers, and splits
             "API answered" from "reply parsed as a tool request".
  3. e2e     route_customer(), timed from outside, fresh session per
             request. Shows how much of a whole turn is the model call.

Nothing here edits KasaFlow behaviour. The benchmark registers a sink on
services/llm_metrics.py (which records per-attempt data for phases 2 and
3) and times route_customer() from outside.

Success is reported in separate layers, because they are different things:

  api_ok          The API returned a response (no timeout or API error).
  valid_response  The response contained non-empty text.
  parse_ok        The text parsed as a KasaFlow tool request (prod phase).
  route_ok        route_customer() returned a result with no "error" key
                  (e2e phase). A "nothing found" result counts as ok.

Load shape. The ladder is closed-loop: N worker threads each take the next
request as soon as their last one finishes, and all start together behind
a barrier, so the stated concurrency is what is actually in flight.
max_in_flight in the report is the observed peak. If it is below the
stated level, the level was not reached.

Wording. "First request" and "later requests" describe order only. The
provider's infrastructure is not something this script can make cold or
warm. cached_tokens (the provider's prompt cache) is reported separately.

Small samples. With fewer than 20 requests, p95 is effectively the
maximum. The report marks those cells with *.

Privacy. Rows and reports carry the scenario id, never the message text.
Error strings are not stored (the parser's own errors can embed model
output); only the exception class name is. The tool names the model
chose are stored, as categories.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import queue
import random
import sys
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

SMALL_SAMPLE = 20
DEFAULT_LEVELS = (1, 2, 4, 8)


class Case(NamedTuple):
    """One benchmark input. Only scenario_id is ever written to a row or report."""

    scenario_id: str
    message: str


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already sorted, non-empty list."""
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def summarise(values: list[float | None]) -> dict | None:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    return {
        "n": len(clean),
        "min": clean[0],
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "max": clean[-1],
        "mean": round(sum(clean) / len(clean), 1),
    }


# ---------------------------------------------------------------------
# Load generation
# ---------------------------------------------------------------------

def run_closed_loop(task: Callable[[Any], dict], items: list, concurrency: int) -> list[dict]:
    """Run task(item) for every item with `concurrency` workers in flight.

    Workers start together (barrier), then each takes the next item as soon
    as it finishes the last. Results come back in item order.
    """
    workers = max(1, min(concurrency, len(items)))
    pending: queue.Queue = queue.Queue()
    for index, item in enumerate(items):
        pending.put((index, item))
    results: list[dict | None] = [None] * len(items)
    barrier = threading.Barrier(workers)

    def worker() -> None:
        barrier.wait()
        while True:
            try:
                index, item = pending.get_nowait()
            except queue.Empty:
                return
            results[index] = task(item)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [r for r in results if r is not None]


class AttemptCollector:
    """llm_metrics sink that files records by the thread that made the call."""

    def __init__(self) -> None:
        self._by_thread: dict[int, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def __call__(self, record: dict) -> None:
        with self._lock:
            self._by_thread[record["thread_id"]].append(record)

    def drain(self) -> list[dict]:
        """Take and clear this thread's records."""
        with self._lock:
            return self._by_thread.pop(threading.get_ident(), [])


def fold_attempts(attempts: list[dict]) -> dict:
    """One row's worth of facts from the attempts one request made."""
    if not attempts:
        return {"llm_called": False, "llm_attempts": 0}
    ok = [a for a in attempts if a.get("outcome") == "ok"]
    last = ok[-1] if ok else attempts[-1]
    return {
        "llm_called": True,
        "llm_attempts": len(attempts),
        "llm_outcomes": [a.get("outcome") for a in attempts],
        "llm_latency_ms": round(sum(a.get("latency_ms", 0.0) for a in attempts), 1),
        "final_attempt_latency_ms": last.get("latency_ms"),
        "input_tokens": last.get("input_tokens"),
        "output_tokens": last.get("output_tokens"),
        "cached_tokens": last.get("cached_tokens"),
        "reasoning_tokens": last.get("reasoning_tokens"),
        "output_chars": last.get("output_chars"),
        "in_flight_at_start": max(a.get("in_flight_at_start", 0) for a in attempts),
    }


def tool_names(parsed: Any) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    if "requests" in parsed and isinstance(parsed["requests"], list):
        return [str(r.get("tool")) for r in parsed["requests"] if isinstance(r, dict)]
    return [str(parsed["tool"])] if "tool" in parsed else []


# ---------------------------------------------------------------------
# Phase tasks. Dependencies are passed in so tests can use fakes.
# ---------------------------------------------------------------------

def make_prod_task(call_llm: Callable, parse: Callable, collector: AttemptCollector,
                   error_cls: type[Exception]) -> Callable[[Case], dict]:
    def task(case: Case) -> dict:
        collector.drain()
        row: dict[str, Any] = {"phase": "prod", "scenario_id": case.scenario_id, "ts": round(time.time(), 3)}
        t0 = time.perf_counter()
        try:
            raw = call_llm(case.message, None, None)
            row["api_ok"] = True
            row["valid_response"] = isinstance(raw, str) and bool(raw.strip())
            try:
                row["tools"] = tool_names(parse(raw))
                row["parse_ok"] = True
            except error_cls as exc:
                row["parse_ok"] = False
                row["error_type"] = type(exc).__name__
        except error_cls as exc:
            row.update(api_ok=False, valid_response=False, parse_ok=None, error_type=type(exc).__name__)
        except Exception as exc:  # one bad request must not end the run
            row.update(api_ok=False, valid_response=False, parse_ok=None, error_type=type(exc).__name__)
        row["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row.update(fold_attempts(collector.drain()))
        return row
    return task


def make_e2e_task(route: Callable[[str, str], dict], collector: AttemptCollector) -> Callable[[Case], dict]:
    def task(case: Case) -> dict:
        collector.drain()
        row: dict[str, Any] = {"phase": "e2e", "scenario_id": case.scenario_id, "ts": round(time.time(), 3)}
        t0 = time.perf_counter()
        try:
            result = route(case.message, f"bench-{uuid.uuid4().hex[:12]}")
            row["route_ok"] = isinstance(result, dict) and "error" not in result
            if isinstance(result, dict) and "error" in result:
                row["error_type"] = "route_error_result"
        except Exception as exc:
            row["route_ok"] = False
            row["error_type"] = type(exc).__name__
        row["e2e_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row.update(fold_attempts(collector.drain()))
        attempts_ok = [o for o in row.get("llm_outcomes", []) if o == "ok"]
        row["api_ok"] = bool(attempts_ok) if row["llm_called"] else None
        row["valid_response"] = (row.get("output_chars") or 0) > 0 if row["llm_called"] else None
        if row["llm_called"]:
            row["non_llm_ms"] = round(row["e2e_ms"] - row["llm_latency_ms"], 1)
        return row
    return task


# ---------------------------------------------------------------------
# Running the stream phase
# ---------------------------------------------------------------------

def run_stream_phase(probe: Callable[[Case], dict], cases: list[Case], warmup: int,
                     levels: tuple[int, ...], per_level: int) -> dict:
    """Warm-up requests in order, then the ladder. Same cases at every level."""
    seq = 0
    warm_rows = []
    for index in range(warmup):
        row = probe(cases[index % len(cases)])
        seq += 1
        row.update(phase="stream-first", seq=seq)
        warm_rows.append(row)

    ladder: dict[int, dict] = {}
    for level in levels:
        items = [cases[i % len(cases)] for i in range(per_level)]
        start = time.perf_counter()
        rows = run_closed_loop(probe, items, level)
        wall_s = time.perf_counter() - start
        for row in rows:
            seq += 1
            row.update(phase="stream-ladder", level=level, seq=seq)
        ladder[level] = {"wall_s": round(wall_s, 3), "rows": rows}
    return {"warmup": warm_rows, "ladder": ladder}


def summarise_level(level: int, wall_s: float, rows: list[dict]) -> dict:
    ok = [r for r in rows if r["outcome"] == "ok"]
    out_tokens = sum(r["output_tokens"] or 0 for r in ok)
    return {
        "level": level,
        "requests": len(rows),
        "ok": len(ok),
        "max_in_flight": max((r["in_flight_at_start"] or 0 for r in rows), default=0),
        "wall_s": wall_s,
        "requests_per_s": round(len(rows) / wall_s, 2) if wall_s else None,
        "output_tokens_per_s_aggregate": round(out_tokens / wall_s, 1) if wall_s else None,
        "ttft_ms": summarise([r["ttft_ms"] for r in ok]),
        "decode_ms": summarise([r["decode_ms"] for r in ok]),
        "total_ms": summarise([r["total_ms"] for r in ok]),
        "output_tps_decode": summarise([r["output_tps_decode"] for r in ok]),
        "input_tokens": summarise([r["input_tokens"] for r in ok]),
        "output_tokens": summarise([r["output_tokens"] for r in ok]),
        "cached_tokens": summarise([r["cached_tokens"] for r in ok]),
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def _f(stat: dict | None, key: str, small: bool = False) -> str:
    if not stat:
        return "-"
    text = f"{stat[key]:.0f}"
    return text + ("*" if small and key in ("p95",) else "")


def print_stream_report(stream: dict) -> None:
    print("\nFIRST REQUESTS (order only; cached_tokens is the provider's prompt cache)")
    print(f"{'seq':>3} {'outcome':<10} {'ttft':>7} {'first_ev':>8} {'decode':>7} {'total':>7} "
          f"{'in_tok':>7} {'out_tok':>7} {'cached':>7}")
    for r in stream["warmup"]:
        print(f"{r['seq']:>3} {r['outcome']:<10} {r['ttft_ms'] or 0:>7.0f} {r['first_event_ms'] or 0:>8.0f} "
              f"{r['decode_ms'] or 0:>7.0f} {r['total_ms'] or 0:>7.0f} {r['input_tokens'] or 0:>7} "
              f"{r['output_tokens'] or 0:>7} {r['cached_tokens'] if r['cached_tokens'] is not None else '-':>7}")

    print("\nCONCURRENCY LADDER (milliseconds unless stated; * = fewer than 20 samples, p95 is the max)")
    print(f"{'conc':>4} {'ok/n':>6} {'peak':>5} {'ttft p50':>9} {'p95':>6} {'decode p50':>10} {'tok/s p50':>9} "
          f"{'total p50':>9} {'p95':>6} {'max':>6} {'req/s':>6} {'agg tok/s':>9} {'in_tok':>7} {'cached':>7}")
    for level, block in stream["ladder"].items():
        s = summarise_level(level, block["wall_s"], block["rows"])
        small = s["requests"] < SMALL_SAMPLE
        peak_note = "" if s["max_in_flight"] >= min(level, s["requests"]) else " !"
        print(f"{level:>4} {str(s['ok']) + '/' + str(s['requests']):>6} {str(s['max_in_flight']) + peak_note:>5} "
              f"{_f(s['ttft_ms'], 'p50'):>9} {_f(s['ttft_ms'], 'p95', small):>6} {_f(s['decode_ms'], 'p50'):>10} "
              f"{_f(s['output_tps_decode'], 'p50'):>9} {_f(s['total_ms'], 'p50'):>9} {_f(s['total_ms'], 'p95', small):>6} {_f(s['total_ms'], 'max'):>6} "
              f"{s['requests_per_s']:>6} {s['output_tokens_per_s_aggregate']:>9} "
              f"{_f(s['input_tokens'], 'mean'):>7} {_f(s['cached_tokens'], 'mean'):>7}")
    print("peak = most requests observed in flight at once. '!' means the stated level was not reached.")
    print("tok/s p50 = per-request output tokens / decode time. agg tok/s = all output tokens / level wall time.")


def print_layer_report(title: str, rows: list[dict], keys: tuple[str, ...]) -> None:
    if not rows:
        return
    print(f"\n{title} ({len(rows)} requests)")
    for key in keys:
        values = [r.get(key) for r in rows if r.get(key) is not None]
        if values:
            print(f"  {key:<15} {sum(1 for v in values if v)}/{len(values)}")
    lat_key = "wall_ms" if "wall_ms" in rows[0] else "e2e_ms"
    stat = summarise([r.get(lat_key) for r in rows])
    if stat:
        print(f"  {lat_key:<15} p50 {stat['p50']:.0f}  max {stat['max']:.0f}")
    llm = summarise([r.get("llm_latency_ms") for r in rows])
    if llm:
        print(f"  {'llm_latency_ms':<15} p50 {llm['p50']:.0f}  max {llm['max']:.0f}")
    if rows[0]["phase"] == "e2e":
        non = summarise([r.get("non_llm_ms") for r in rows])
        skipped = sum(1 for r in rows if not r["llm_called"])
        if non:
            print(f"  {'non_llm_ms':<15} p50 {non['p50']:.0f}  max {non['max']:.0f}  (turn time outside the model call)")
        print(f"  turns that never called the model: {skipped}/{len(rows)}")
    retried = sum(1 for r in rows if r.get("llm_attempts", 0) > 1)
    print(f"  requests that needed a retry: {retried}/{len(rows)}")
    tools = defaultdict(int)
    for r in rows:
        for name in r.get("tools", []):
            tools[name] += 1
    if tools:
        print("  tools chosen: " + ", ".join(f"{k} x{v}" for k, v in sorted(tools.items())))


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def build_plan(args: argparse.Namespace) -> dict:
    levels = tuple(int(x) for x in args.levels.split(","))
    stream_n = 0 if args.skip_stream else args.warmup + len(levels) * args.per_level
    return {
        "warmup": 0 if args.skip_stream else args.warmup,
        "levels": list(levels),
        "per_level": args.per_level,
        "prod_requests": args.prod_requests,
        "e2e_requests": args.e2e_requests,
        "total_model_requests": stream_n + args.prod_requests + args.e2e_requests,
    }


def pick_cases(count: int, seed: int) -> list[Case]:
    """First-turn messages from the evaluator corpus: fixed, seeded, synthetic.

    Some scenarios share a first message. Each message is labelled with the
    alphabetically first scenario id that uses it.
    """
    from scripts.evaluator.scenarios import SCENARIOS

    first_id: dict[str, str] = {}
    for scenario in sorted(SCENARIOS, key=lambda s: s.id):
        if scenario.turns and scenario.turns[0].message:
            first_id.setdefault(scenario.turns[0].message, scenario.id)
    pool = sorted(Case(sid, msg) for msg, sid in first_id.items())
    random.Random(seed).shuffle(pool)
    return [pool[i % len(pool)] for i in range(count)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KasaFlow model-call baseline (real API requests).")
    parser.add_argument("--warmup", type=int, default=5, help="Sequential first requests (default 5).")
    parser.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)), help="Concurrency levels.")
    parser.add_argument("--per-level", type=int, default=8,
                        help="Requests per level (default 8). Must be at least the largest level.")
    parser.add_argument("--prod-requests", type=int, default=4, help="Production-path requests (default 4).")
    parser.add_argument("--e2e-requests", type=int, default=4, help="End-to-end requests (default 4; 0 skips).")
    parser.add_argument("--skip-stream", action="store_true", help="Skip the streaming phase.")
    parser.add_argument("--seed", type=int, default=0, help="Message selection seed.")
    parser.add_argument("--note", default="", help="Free text stored in the report (where you ran it, network).")
    parser.add_argument("--out", help="JSON report path (default bench_results/baseline-<time>.json).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit. No API requests.")
    parser.add_argument("--probe-one", action="store_true", help="Send one streaming request and exit.")
    parser.add_argument("--show-events", action="store_true", help="With --probe-one: print every stream event.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    if max(plan["levels"]) > args.per_level:
        print("--per-level must be at least the largest concurrency level.", file=sys.stderr)
        return 2

    print("Plan:", json.dumps(plan))
    if args.dry_run:
        print("Dry run: no requests sent.")
        return 0

    from services import llm, llm_metrics
    from scripts.bench.stream_probe import Gauge, probe_once

    model = llm.settings.openai_model
    timeout = llm.settings.llm_timeout_seconds

    def probe_with_prompt(keep_events: bool = False, gauge: Gauge | None = None) -> Callable[[str], dict]:
        def probe(case: Case) -> dict:
            prompt = llm._build_prompt(case.message, None, None)
            row = probe_once(llm.client, model, prompt, timeout=timeout, gauge=gauge, keep_events=keep_events)
            row["scenario_id"] = case.scenario_id
            return row
        return probe

    if args.probe_one:
        row = probe_with_prompt(keep_events=args.show_events)(pick_cases(1, args.seed)[0])
        for event in row.pop("events", []):
            print(f"{event['t_ms']:>8.1f} ms  {event['type']}")
        print(json.dumps(row, indent=2))
        return 0

    sample_prompt = llm._build_prompt(pick_cases(1, args.seed)[0].message, None, None)
    t0 = time.perf_counter()
    for _ in range(200):
        llm._build_prompt("x", None, None)
    build_ms = round((time.perf_counter() - t0) / 200 * 1000, 3)

    started = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "meta": {
            "started_utc": started.isoformat(timespec="seconds"),
            "plan": plan,
            "model": model,
            "timeout_s": timeout,
            "llm_max_retries": llm.settings.llm_max_retries,
            "prompt_chars_empty_state": len(sample_prompt),
            "prompt_build_ms_mean": build_ms,
            "state": "empty session state; real sessions add to the state lines",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "openai_sdk": __import__("openai").__version__,
            "seed": args.seed,
            "note": args.note,
        },
    }
    print(f"Model {model}. Prompt is {len(sample_prompt)} characters with empty state; "
          f"building it takes {build_ms} ms (inside the prod-path timer).")

    if not args.skip_stream:
        gauge = Gauge()
        cases = pick_cases(max(args.warmup, args.per_level), args.seed)
        stream = run_stream_phase(probe_with_prompt(gauge=gauge), cases, args.warmup,
                                  tuple(plan["levels"]), args.per_level)
        print_stream_report(stream)
        report["stream"] = {
            "warmup": stream["warmup"],
            "ladder": {str(k): {"wall_s": v["wall_s"], "rows": v["rows"],
                                "summary": summarise_level(k, v["wall_s"], v["rows"])}
                       for k, v in stream["ladder"].items()},
        }

    collector = AttemptCollector()
    llm_metrics.add_sink(collector)
    try:
        if args.prod_requests > 0:
            task = make_prod_task(llm._call_llm, llm._parse_tool_request, collector, llm.ToolSelectionError)
            rows = run_closed_loop(task, pick_cases(args.prod_requests, args.seed), 1)
            print_layer_report("PRODUCTION PATH (llm._call_llm + parse, non-streaming, concurrency 1)", rows,
                               ("api_ok", "valid_response", "parse_ok"))
            report["prod"] = rows

        if args.e2e_requests > 0:
            from services import router

            print("\nE2E uses fresh sessions and first-turn messages, so no order can be created. "
                  "Each turn still runs embeddings, catalogue and delivery lookups as normal.")
            task = make_e2e_task(router.route_customer, collector)
            rows = run_closed_loop(task, pick_cases(args.e2e_requests, args.seed), 1)
            print_layer_report("END TO END (route_customer, concurrency 1)", rows, ("api_ok", "valid_response", "route_ok"))
            report["e2e"] = rows
    finally:
        llm_metrics.remove_sink(collector)

    out = Path(args.out) if args.out else Path("bench_results") / f"baseline-{started:%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
