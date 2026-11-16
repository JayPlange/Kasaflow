# KasaFlow — WhatsApp AI Customer Workflow Engine

> An AI operator being tested ahead of a pilot with a jewellery retailer. It handles product enquiries, pricing, delivery quotes and order workflows over WhatsApp. The LLM selects the tools; deterministic Python controls pricing, validation and order execution.

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)]()
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses%20API-black?style=flat&logo=openai&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?style=flat&logo=docker&logoColor=white)]()
[![Tests](https://img.shields.io/badge/tests-937%20passing-brightgreen?style=flat)]()

---

## What It Does

A customer messages the business on WhatsApp — text, a voice note in English or Twi, or a photo of a piece they like — and KasaFlow turns it into a structured action: a price, a delivery quote, a product recommendation, a policy answer, or a real order against the business's live WooCommerce store. The LLM's only job is deciding *which* tool to call and *what arguments* it needs. It never talks to the customer directly, never prices anything itself, and never writes to WooCommerce directly — those are Python's job.

That separation is the core design bet: prompts decide intent, Python executes it. If the model hallucinates or drifts, the blast radius is "picked the wrong tool," not "invented a price" or "created an order that shouldn't exist."

## Architecture

.
├── app/
├── services/
├── data/
├── docs/
├── tests/
├── scripts/
└── evaluator/

## Key Engineering Decisions

**The LLM proposes, the application owns state**
The model decides what a message means and which tool to call. It never validates state, calculates money, or decides whether an order can be created — that's `order_tool.py`'s job. Concretely: `propose_order()` is a pure, deterministic lookup and multiplication, never the model; `confirm_order()` only ever acts on a proposal that function already priced and stored, never on arguments the model hands it directly. A field the model can't determine comes back as the literal string `"unknown"`, not omitted or guessed, so the system's own memory resolves it rather than the model inventing a value.

**Orders are scoped to the customer who placed them**
`get_order_status`/`cancel_order` check that the order's stamped `billing.phone` matches the requesting WhatsApp session before acting, rather than trusting any order number a message happens to contain. Added 2026-09-05 after auditing `_resolve_order_id()` and finding it had no ownership check at all — any customer could look up or cancel any other customer's order by guessing a small sequential ID.

**WhatsApp webhook signatures are verified, not assumed**
`app/whatsapp_routes.py` recomputes Meta's `X-Hub-Signature-256` HMAC and compares in constant time before processing anything. Previously any POST that merely parsed as valid JSON was trusted — meaning anyone who found the webhook URL could fabricate a customer message and trigger a real, billed LLM call, or a real order.

**A per-session lock spans the whole turn, not just individual memory writes**
Two near-simultaneous messages from the same customer used to be able to race: both read the same starting state, and whichever's LLM call finished last could silently overwrite the other's actual last-stated value. `services/memory.py` now holds one lock per session for the full `route_customer()` call, not just individual `get()`/`set()` operations.

**Tool Registry + Tool Executor, not a big if/else**
Eleven tools are registered by name in one dict and dispatched generically. Adding a twelfth means writing the function and registering it — nothing else in the request path changes.

**Retrieval, not a bigger prompt**
Policy questions are answered by embedding the question and comparing against cached policy-document embeddings, returning only what's above a similarity threshold — a return-policy question genuinely can't get a warranty-policy answer.

**Don't retry a failed write blindly**
WooCommerce's REST API has no idempotency-key support. On a timed-out order-create, the system looks the order up by a token embedded in its own metadata before deciding whether a retry is safe, instead of risking a duplicate.

**Fail-fast configuration**
Every required environment variable is validated once at import time. A missing key crashes the app on startup, not on the first customer request.

## Known Limitations

The honest, current list of this is `docs/architecture_audit_2026-08-20.md` — a self-run audit covering state isolation, conversation continuity, turn concurrency, and order safety, each with evidence and a concrete fix. One item from it (the per-session turn lock above) is confirmed fixed since the audit; the rest are ranked and sequenced there rather than restated here, since this README would just go stale against it again.

Beyond what that document covers: memory is in-process only (no Redis, doesn't survive a restart or a second replica), CI runs tests but there's no deployment pipeline, and RAG is a flat six-document set, not a vector database.

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Run the standard test suite:

```bash
pytest
```

Run the regression tests (these make real API calls and may incur costs):

```bash
pytest --run-regression
```

### Behavioural evaluator

`scripts/evaluator/` is separate from pytest: real, multi-turn conversations run against the actual `route_customer()` pipeline, checked against an expected tool/fields/response per turn across fifteen categories. It answers a question pytest can't — not "does the code do what it's supposed to", but "does a real conversation resolve the way a customer would expect it to." Correctness and manner (does the reply actually read naturally) are checked separately on purpose; manner isn't auto-scored.

```bash
python3 scripts/run_evaluator.py
python3 scripts/run_evaluator.py --category "REFERENCES"
```

### Conversation eval set

`evalset/` and `scripts/evalset/` turn logged conversations into an anonymised eval set and replay it through the current code, scoring order accuracy, cost per conversation and latency, with a static HTML dashboard per run. A recorded mode replays the model's original answers (free, isolates the application); a live mode calls the model with `--repeat` to expose variance. See `evalset/README.md`. The set currently holds two operator test conversations; there are no real customer conversations yet.

## Guardrails

Deterministic checks around the model's choices, in code rather than the prompt:

- **Totals before committing.** `confirm_order()` re-prices the exact quoted variant against the catalogue and re-quotes if the price moved; compares WooCommerce's order total with the quoted one and hands off on a mismatch; optional `ORDER_AUTO_CONFIRM_MAX_TOTAL` and `ORDER_MAX_QUANTITY` send large orders to a person instead of placing them.
- **Prompt injection.** `services/guardrails.py` flags messages that try to override instructions, impersonate staff or the system, fake a tool call, set a price or extract the prompt. A flagged message is quoted as data in the prompt and cannot trigger `confirm_order` or `cancel_order` that turn. Converse replies that leak prompt text are replaced.
- **Human escalation.** `services/escalation.py`. Explicit requests for a person are caught before the model runs; the model can also choose `escalate_to_human`; repeated failures (`ESCALATION_FAILURE_THRESHOLD`, default 3) or repeated manipulation (`ESCALATION_INJECTION_THRESHOLD`, default 2) escalate automatically. An escalated session gets a holding reply and no model or tool calls, and staff are notified on `STAFF_NOTIFICATION_PHONE`.

Every turn logs `KASAFLOW_TURN_METRICS` (latency, tokens per model call, guardrail events) next to the existing `KASAFLOW_TURN_TRACE`.

## Running Locally

```bash
# .env (not committed) needs: OPENAI_API_KEY, APP_API_KEY, and, for the
# integrations you want active: WOOCOMMERCE_*, WHATSAPP_*, KHAYA_*, COHERE_API_KEY,
# GOOGLE_MAPS_API_KEY. Each integration degrades gracefully when unconfigured --
# see app/config.py for what each one unlocks.

docker compose up --build
```

Visit `http://localhost:8000/docs` for the interactive API.

## Tech Stack

| Layer | Tech |
|---|---|
| API | FastAPI, Uvicorn |
| LLM | OpenAI Responses API (GPT-4.1-mini) |
| Retrieval | OpenAI embeddings (`text-embedding-3-small`), in-memory cosine similarity |
| Commerce | WooCommerce REST API (live catalogue + order creation) |
| Messaging | WhatsApp Cloud API, HMAC-verified webhook |
| Voice | Khaya AI (ASR/TTS, English + Twi) |
| Vision | Cohere image embeddings (exact-item photo matching) |
| Geocoding | Google Maps API (delivery-zone inference) |
| Validation | Pydantic |
| Auth / rate limiting | Custom API key dependency, slowapi |
| Testing | pytest (937 passing), separate behavioural evaluator |
| Containerization | Docker (multi-stage, non-root user, healthcheck) |

## Status

Self-directed, actively developed as the engine behind a real WhatsApp-based customer workflow project for a jewellery retail client. Not yet deployed to a public host.
