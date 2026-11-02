"""
Render an eval run as a self-contained HTML dashboard.

    python -m scripts.evalset.dashboard                    # latest run in eval_reports/
    python -m scripts.evalset.dashboard eval_reports/run-....json

Writes the HTML next to the run file. One file, no external requests, so
it can be opened offline or attached to a message. Light and dark mode.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from scripts.evalset.score import load_pricing, score

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "eval_reports"
LABELS_PATH = ROOT / "evalset" / "labels.json"

E = html.escape


def _fmt_pct(rate):
    return "n/a" if rate is None else f"{rate * 100:.0f}%"


def _fmt_ms(ms):
    if ms is None:
        return "n/a"
    if ms < 1:
        return "<1 ms"
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"


def _fmt_usd(value):
    if value is None:
        return "n/a"
    return f"${value:.4f}" if value < 0.1 else f"${value:.2f}"


def _verdict_badge(run: dict) -> str:
    order = run["order"]
    if order is None:
        return '<span class="badge muted">— Unscored</span>'
    if order["correct"]:
        return '<span class="badge good">✓ Correct</span>'
    return '<span class="badge critical">✗ Wrong</span>'


def _order_text(order: dict | None) -> str:
    if not order:
        return "none"
    return f"{order.get('quantity')} × {order.get('material')} {order.get('product')}"


def _latency_chart(turns: list[dict]) -> str:
    """Horizontal stacked bars, one per turn: model time then the rest."""
    if not turns:
        return ""
    rows = []
    for t in turns:
        model_ms = sum(c.get("latency_ms") or 0 for c in t.get("llm_calls", []))
        app_ms = max(t["turn_latency_ms"] - model_ms, 0)
        rows.append((t["index"] + 1, model_ms, app_ms))
    # Floor the scale at one second so sub-millisecond replay turns don't
    # stretch to full width and read like real waiting time.
    longest = max(max(m + a for _, m, a in rows), 1000.0)
    bar_h, gap, label_w, width = 14, 8, 52, 560
    plot_w = width - label_w - 70
    height = len(rows) * (bar_h + gap) + gap
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Latency per turn, model time and application time">']
    for i, (n, model_ms, app_ms) in enumerate(rows):
        y = gap + i * (bar_h + gap)
        parts.append(f'<text x="{label_w - 8}" y="{y + bar_h - 3}" class="axis" text-anchor="end">Turn {n}</text>')
        x = label_w
        model_w = model_ms / longest * plot_w
        app_w = app_ms / longest * plot_w
        tip = f"Turn {n}: model {_fmt_ms(model_ms)}, application {_fmt_ms(app_ms)}"
        if model_w >= 1:
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{max(model_w - 2, 1):.1f}" height="{bar_h}" rx="2" '
                         f'class="s1"><title>{E(tip)}</title></rect>')
            x += model_w
        if app_w >= 1 or model_w < 1:
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{max(app_w, 2):.1f}" height="{bar_h}" rx="4" '
                         f'class="s2"><title>{E(tip)}</title></rect>')
            x += max(app_w, 2)
        parts.append(f'<text x="{x + 6:.1f}" y="{y + bar_h - 3}" class="value">{E(_fmt_ms(model_ms + app_ms))}</text>')
    parts.append("</svg>")
    legend = ('<div class="legend"><span><i class="sw s1"></i>Model call</span>'
              '<span><i class="sw s2"></i>Application (tools, state, formatting)</span></div>')
    return legend + "".join(parts)


def _turn_table(run_report: dict, mode: str) -> str:
    rows = []
    for t in run_report["turns"]:
        tokens = sum((c.get("input_tokens") or 0) + (c.get("output_tokens") or 0) for c in t.get("llm_calls", []))
        notes = [e["event"] + (f" ({e['reason']})" if e.get("reason") else "") for e in t.get("guardrail_events", [])]
        replay = t.get("replay")
        if replay:
            if replay["missing_model_answer"]:
                notes.append("replay: model now called where it wasn't")
            if replay["original_result_kind"] != t["result_kind"]:
                notes.append(f"changed from {replay['original_result_kind']}")
        if t.get("embeddings_blocked"):
            notes.append("embeddings off")
        if t.get("error"):
            notes.append(t["error"])
        rows.append(
            "<tr>"
            f"<td class='num'>{t['index'] + 1}</td>"
            f"<td>{E(t['customer_message'])}</td>"
            f"<td>{E(t.get('reply') or '')}</td>"
            f"<td><code>{E(str(t.get('result_kind')))}</code></td>"
            f"<td class='num'>{_fmt_ms(t['turn_latency_ms'])}</td>"
            f"<td class='num'>{tokens if tokens else '–'}</td>"
            f"<td>{E('; '.join(notes))}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th>#</th><th>Customer</th><th>Assistant</th><th>Result</th>"
        "<th>Latency</th><th>Tokens</th><th>Notes</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def render(report: dict, scores: dict) -> str:
    mode = report["mode"]
    acc = scores["order_accuracy"]
    cost = scores["cost_usd"]
    lat = scores["latency_ms"]
    ci = acc["ci95"]
    ci_text = f"95% interval {ci[0] * 100:.0f}–{ci[1] * 100:.0f}%" if ci else "no reviewed labels yet"

    banners = []
    if scores["unreviewed_conversations"]:
        banners.append(
            f"{len(scores['unreviewed_conversations'])} of {scores['conversations']} conversations have no reviewed "
            "label, so they are not counted in order accuracy. Review evalset/labels.json."
        )
    sources = sorted({r.get("source") or "unknown" for r in scores["per_run"]})
    if any(s != "pilot" for s in sources):
        banners.append(f"Conversation source: {', '.join(sources)}. These are not real customer conversations.")
    if mode == "recorded":
        banners.append(
            "Recorded replay: the model's original answers were replayed, so latency is application time only "
            "and cost comes from the original turns' logged tokens (n/a where they weren't logged). "
            "Run --mode live to measure the model."
        )
    if scores["embeddings_blocked_turns"]:
        n = scores["embeddings_blocked_turns"]
        banners.append(f"{n} turn{'s' if n != 1 else ''} hit a disabled embeddings call (product search fallback "
                       "or policy answer); run with --allow-embeddings to include them.")

    events = scores["guardrail_events"]
    handoffs = {k.split(":", 1)[1]: v for k, v in events.items() if k.startswith("escalated:")}
    guard_other = {k: v for k, v in events.items() if not k.startswith("escalated:")}

    latency_label = "Turn latency (application only)" if mode == "recorded" else "Turn latency"
    tiles = f"""
<section class="tiles">
  <div class="tile"><div class="label">Order accuracy</div>
    <div class="big">{acc['correct']}/{acc['scored']}</div>
    <div class="sub">{_fmt_pct(acc['rate'])} · {E(ci_text)}</div></div>
  <div class="tile"><div class="label">Cost per conversation</div>
    <div class="big">{_fmt_usd(cost['median'])}</div>
    <div class="sub">median · p95 {_fmt_usd(cost['p95'])} · measured {cost['measured_runs']}/{cost['of_runs']} runs</div></div>
  <div class="tile"><div class="label">{latency_label}</div>
    <div class="big">{_fmt_ms(lat['turn_p50'])}</div>
    <div class="sub">p50 · p95 {_fmt_ms(lat['turn_p95'])} · model p50 {_fmt_ms(lat['model_p50'])}</div></div>
  <div class="tile"><div class="label">Handoffs to a person</div>
    <div class="big">{sum(handoffs.values())}</div>
    <div class="sub">{E(', '.join(f'{k} {v}' for k, v in sorted(handoffs.items())) or 'none')}</div></div>
  <div class="tile"><div class="label">Other guardrail events</div>
    <div class="big">{sum(guard_other.values())}</div>
    <div class="sub">{E(', '.join(f'{k} {v}' for k, v in sorted(guard_other.items())) or 'none')}</div></div>
</section>"""

    conv_rows = []
    for r in scores["per_run"]:
        order = r["order"]
        expected = _order_text(order["expected"]) if order else "–"
        created = "; ".join(_order_text(o) for o in order["created"]) if order else "–"
        why = order["why"] if order else "label not reviewed"
        consistency = scores["consistency"].get(r["conversation_id"])
        run_suffix = f" · run {r['run'] + 1}" if (scores["repeat"] or 1) > 1 else ""
        consistency_text = "–" if consistency is None else f"{consistency * 100:.0f}%"
        conv_rows.append(
            "<tr>"
            f"<td><code>{E(r['conversation_id'])}</code>{run_suffix}</td>"
            f"<td>{_verdict_badge(r)}<div class='why'>{E(why)}</div></td>"
            f"<td>{E(expected)}</td><td>{E(created or 'none')}</td>"
            f"<td>{E((r['escalation'] or {}).get('reason') or '–')}</td>"
            f"<td class='num'>{_fmt_usd(r['cost_usd'])}</td>"
            f"<td class='num'>{_fmt_ms(r['latency_ms_total'])}</td>"
            f"<td class='num'>{consistency_text}</td>"
            "</tr>"
        )

    drilldowns = []
    for run in report["runs"]:
        if run["run"] != 0:
            continue
        drilldowns.append(
            f"<details><summary><code>{E(run['conversation_id'])}</code> · {len(run['turns'])} turns</summary>"
            f"{_latency_chart(run['turns'])}{_turn_table(run, mode)}</details>"
        )

    replay_line = ""
    if scores.get("replay"):
        rp = scores["replay"]
        d, m = rp["result_kind_divergences"], rp["missing_model_answers"]
        replay_line = (f"<p class='sub'>Replay: {d} turn{'s' if d != 1 else ''} ended differently from the original "
                       f"conversation; {m} turn{'s' if m != 1 else ''} asked the model where the original didn't.</p>")

    title = f"{mode} run · {report.get('started_utc', '')} UTC"
    model_suffix = f" · model {E(scores['model'])}" if scores.get("model") else ""
    banner_html = "".join(f'<div class="banner">{E(b)}</div>' for b in banners)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KasaFlow Eval Dashboard</title>
<style>
:root{{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--s1:#2a78d6;--s2:#eb6834;
--good:#0a7d0a;--good-bg:#e3f4e3;--critical:#b42f2f;--critical-bg:#fbe6e6;--muted-bg:#efeeea;--warn-bg:#fff4dc}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--panel:#222220;--ink:#f0efec;--ink2:#c3c2b7;
--rule:#383835;--s1:#3987e5;--s2:#d95926;--good:#5fd35f;--good-bg:#1f3a1f;--critical:#f08a8a;--critical-bg:#45201f;--muted-bg:#2c2c2a;--warn-bg:#3d331a}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--panel:#222220;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835;--s1:#3987e5;--s2:#d95926;
--good:#5fd35f;--good-bg:#1f3a1f;--critical:#f08a8a;--critical-bg:#45201f;--muted-bg:#2c2c2a;--warn-bg:#3d331a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 64px}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:32px 0 10px}}
.sub,.why{{color:var(--ink2);font-size:13px}}.banner{{background:var(--warn-bg);border-radius:8px;padding:10px 14px;margin:8px 0;font-size:14px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-top:18px}}
.tile{{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px}}
.tile .label{{font-size:13px;color:var(--ink2)}}.tile .big{{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums;margin:2px 0}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:var(--panel);font-size:14px}}
th,td{{text-align:left;vertical-align:top;padding:8px 10px;border-bottom:1px solid var(--rule)}}th{{font-size:12px;color:var(--ink2);font-weight:600}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}code{{font-size:12.5px}}
.badge{{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12.5px;font-weight:600;white-space:nowrap}}
.badge.good{{background:var(--good-bg);color:var(--good)}}.badge.critical{{background:var(--critical-bg);color:var(--critical)}}
.badge.muted{{background:var(--muted-bg);color:var(--ink2)}}
details{{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:10px 14px;margin:10px 0}}
summary{{cursor:pointer;font-weight:600}}.chart{{width:100%;max-width:560px;height:auto;margin:8px 0 12px;display:block}}
.chart .axis,.chart .value{{fill:var(--ink2);font-size:11px}}.s1{{fill:var(--s1);background:var(--s1)}}.s2{{fill:var(--s2);background:var(--s2)}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin-top:10px}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
dl{{font-size:14px}}dt{{font-weight:600;margin-top:10px}}dd{{margin:2px 0 0;color:var(--ink2)}}
</style></head><body><main>
<h1>KasaFlow conversation eval</h1>
<div class="sub">{E(title)} · {scores['conversations']} conversations · {scores['runs']} runs{model_suffix}</div>
{banner_html}
{tiles}
{replay_line}
<h2>Conversations</h2>
<div class="scroll"><table><thead><tr><th>Conversation</th><th>Order outcome</th><th>Expected</th><th>Created</th>
<th>Handoff</th><th class="num">Cost</th><th class="num">Latency</th><th class="num">Consistency</th></tr></thead>
<tbody>{''.join(conv_rows)}</tbody></table></div>
<h2>Turn by turn</h2>
{''.join(drilldowns)}
<h2>How these are measured</h2>
<dl>
<dt>Order accuracy</dt><dd>Runs with a reviewed label only. Correct when no order was expected and none was placed, or when exactly one order was placed with the labelled product, karat and quantity and the store total matched the quote.</dd>
<dt>Cost per conversation</dt><dd>Tokens of the tool-selection model call priced with scripts/evalset/pricing.json ({E(cost.get('pricing_note') or '')}). Excludes embeddings, vision, voice and geocoding.</dd>
<dt>Turn latency</dt><dd>Measured inside the router for each customer message, split into model time and everything else. Excludes WhatsApp delivery time.</dd>
<dt>Consistency</dt><dd>With repeated runs, the share of runs of a conversation that ended in the most common outcome.</dd>
<dt>Guardrail events</dt><dd>injection_flagged, write_blocked, reply_replaced, price_changed, handoff_active and escalated (with reason), as logged by the router.</dd>
</dl>
</main></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    args = parser.parse_args(argv)

    run_path = args.run
    if run_path is None:
        candidates = sorted(REPORTS_DIR.glob("run-*.json"))
        if not candidates:
            print("No runs in eval_reports/. Run python -m scripts.evalset.run first.")
            return 1
        run_path = candidates[-1]
    report = json.loads(run_path.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels.exists() else {}
    scores = score(report, labels, load_pricing())
    out = run_path.with_suffix(".html")
    out.write_text(render(report, scores), encoding="utf-8")
    run_path.with_name(run_path.stem + "-scores.json").write_text(
        json.dumps({k: v for k, v in scores.items()}, indent=2, default=str), encoding="utf-8",
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
