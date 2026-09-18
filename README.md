# KasaFlow — AI Customer Workflow Engine

> **AI operator for a jewellery retailer, running over WhatsApp.**
> The LLM decides what to do. Deterministic Python does it — including the money.

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)]()
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses%20API-black?style=flat&logo=openai&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?style=flat&logo=docker&logoColor=white)]()
[![Tests](https://img.shields.io/badge/tests-809%20passing-brightgreen?style=flat)]()

---

## What It Does

A customer messages the business on WhatsApp — text, a voice note in English or Twi, or a photo of a piece they like — and KasaFlow turns it into a structured action: a price, a delivery quote, a product recommendation, a policy answer, or a real order against the business's live WooCommerce store. The LLM's only job is deciding *which* tool to call and *what arguments* it needs. It never talks to the customer directly, never prices anything itself, and never writes to WooCommerce directly — those are Python's job.

That separation is the core design bet: prompts decide intent, Python executes it. If the model hallucinates or drifts, the blast radius is "picked the wrong tool," not "invented a price" or "created an order that shouldn't exist."

## Architecture

app/
├── main.py # FastAPI app: /process endpoint, auth + rate limiting, conditional demo mount
├── auth.py # Shared-secret API key check (X-API-Key header)
├── config.py # Centralized, fail-fast settings (env vars validated at startup)
├── whatsapp_routes.py # WhatsApp Cloud API webhook, HMAC-SHA256 signature verification
├── demo_routes.py # Local-only demo dashboard, off by default (ENABLE_DEMO_DASHBOARD)
└── logging_config.py # Structured logging setup

services/
├── router.py # Orchestration: message -> LLM -> tool execution -> result
├── llm.py # LLM tool-selection layer (OpenAI Responses API)
├── memory.py # Per-session state, TTL-based, one lock per session spanning a full turn
├── tool_registry.py # Maps tool names -> functions (11 tools, 3 of them writes)
├── tool_executor.py # Calls the selected tool, normalizes failures
├── order_tool.py # propose_order / confirm_order / cancel_order / get_order_status
├── woocommerce_sync.py # Live catalogue sync against the real store
├── whatsapp_client.py # Outbound WhatsApp Cloud API sends
├── voice_tool.py # Khaya ASR/TTS for English/Twi voice notes
├── vision_tool.py # Describes a customer's product photo, routes it into the catalogue pipeline
├── photo_match_tool.py # Cohere image embeddings -- exact-item photo identification
├── image_search.py, image_embed_tool.py, image_embeddings_sync.py # Supporting image-matching pipeline
├── geocoding_tool.py # Delivery-zone inference from a free-text address
├── product_tool.py, product_search.py # Price/weight/karat lookups, semantic + keyword search
├── quote_service.py # Composes price + delivery into one quote
├── recommendation_service.py # Catalogue-filtered recommendations
├── knowledge_base.py, policy_tool.py # RAG-backed policy Q&A
└── response_formatter.py # Isolates safety-critical facts (price, order ID) from tone-only wording

data/products.json # Product catalogue, synced live from WooCommerce
data/policies.json # Policy/FAQ documents the RAG tool retrieves from
data/image_embeddings.json # Precomputed embeddings for photo-based product matching

docs/architecture_audit_2026-08-20.md # Self-authored failure-mode audit: 10 ranked issues,
# evidence + impact + fix sequence for each. See below.

tests/ # 809 passing, 29 gated behind --run-regression (real-API cost)
scripts/evaluator/ # Separate behavioural evaluator: real multi-turn conversations, 15 scenario categories


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
pytest                        # 809 passing, 29 skipped (gated behind --run-regression)
pytest --run-regression       # + real-OpenAI-API regression tests (costs money)
```

### Behavioural evaluator

`scripts/evaluator/` is separate from pytest: real, multi-turn conversations run against the actual `route_customer()` pipeline, checked against an expected tool/fields/response per turn across fifteen categories. It answers a question pytest can't — not "does the code do what it's supposed to", but "does a real conversation resolve the way a customer would expect it to." Correctness and manner (does the reply actually read naturally) are checked separately on purpose; manner isn't auto-scored.

```bash
python3 scripts/run_evaluator.py
python3 scripts/run_evaluator.py --category "REFERENCES"
```

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
| Testing | pytest (809 passing), separate behavioural evaluator |
| Containerization | Docker (multi-stage, non-root user, healthcheck) |

## Status

Self-directed, actively developed as the engine behind a real WhatsApp-based customer workflow project for a jewellery retail client. Not yet deployed to a public host.