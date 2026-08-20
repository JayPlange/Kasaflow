# KasaFlow — AI Customer Workflow Engine

> **Structured, tool-based AI customer service for a jewellery retailer.**
> LLM decides what to do. Deterministic Python does it. Never the other way round.

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)]()
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses%20API-black?style=flat&logo=openai&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?style=flat&logo=docker&logoColor=white)]()
[![Tests](https://img.shields.io/badge/tests-66%20passing-brightgreen?style=flat)]()

---

## What It Does

KasaFlow takes a customer's free-text message ("how much is a gold ring?", "what've you got in silver?", "what's your returns policy?") and turns it into a structured action: a price lookup, delivery info, a combined quote, a product recommendation, or a retrieval-grounded policy answer. The LLM's only job is deciding *which* tool to call and *what arguments* it needs — it never talks to the customer directly, and it never touches the business logic or the policy documents that actually answer them.

That separation is the core design bet: prompts decide intent, Python executes it. If the model hallucinates or drifts, the blast radius is "picked the wrong tool," not "invented a price."

## Architecture

```
app/
├── main.py             # FastAPI app: /process endpoint, auth + rate limiting wiring
├── auth.py              # Shared-secret API key check (X-API-Key header)
├── config.py             # Centralized, fail-fast settings (env vars validated at startup)
└── logging_config.py     # Structured logging setup

services/
├── router.py             # Orchestration: message -> LLM -> tool execution -> result
├── llm.py                # LLM tool-selection layer (OpenAI Responses API)
├── tool_registry.py       # Maps tool names -> functions
├── tool_executor.py       # Calls the selected tool, normalizes failures
├── product_tool.py        # get_product_price -- deterministic JSON lookup
├── delivery_tool.py       # get_delivery_information -- lists real delivery options (rider/shipping), not a price -- a human arranges the actual delivery
├── quote_service.py       # generate_quote -- composes price + delivery into one quote
├── recommendation_service.py  # recommend_products -- filters catalogue by material
├── memory.py              # Per-session context (product/material), keyed by session_id
├── embeddings_client.py    # Thin OpenAI embeddings wrapper, same retry policy as llm.py
├── knowledge_base.py       # RAG index: embeds + caches policy docs, retrieves by cosine similarity
└── policy_tool.py          # answer_policy_question -- the RAG-backed tool

data/products.json         # Product catalogue (deliberately external to app logic)
data/policies.json         # Policy/FAQ documents the RAG tool retrieves from
tests/                     # 66 tests: unit, integration, and prompt regression
```

**Request flow:** `POST /process` → API key + rate-limit check → `route_customer()` → `understand_customer()` asks the LLM which tool to use → `execute_tool()` runs it → structured JSON back to the caller. Every layer only knows about the one below it, so swapping the model, adding a tool, or changing the transport doesn't ripple through the whole system.

## Key Engineering Decisions

**The LLM proposes, the application owns state**
The model may decide what a customer's message means and which tool to call. It never validates state, calculates money, or decides whether an order can actually be created — that's `order_tool.py`'s job (see its own module docstring for the propose/confirm split this enforces), and `services/router.py`/`services/memory.py`'s for tracking what's already been said across turns. Concretely: `propose_order()` is a pure, deterministic lookup and multiplication, never the model; `confirm_order()` only ever acts on a proposal that function already priced and stored, never on arguments the model hands it directly; and a field the model can't determine is returned as the literal string `"unknown"`, not omitted or guessed, so the system's own memory can resolve it rather than the model inventing a value. This is a hard invariant, not a style preference — every "the AI got an order detail wrong" bug this project has hit traces back to somewhere that boundary was blurred.

**Tool Registry + Tool Executor, not a big if/else**
Five tools (`get_product_price`, `get_delivery_information`, `generate_quote`, `recommend_products`, `answer_policy_question`) are registered by name in one dict and dispatched generically. Adding a sixth tool means writing the function and registering it — nothing else in the request path changes.

**Retrieval, not a bigger prompt**
Policy questions (returns, warranty, sizing, care, engraving, payment) are answered by `answer_policy_question`, which embeds the customer's question and the six documents in `data/policies.json` with the OpenAI embeddings API, then returns only the document(s) above a similarity threshold. Documents are embedded once and cached in memory, not re-embedded per request. The alternative — pasting every policy into every prompt — would mean the model reads irrelevant policy on every call and still has to pick the right paragraph itself; retrieval means it only ever sees the one that actually answers the question, and a return-policy question genuinely can't get a warranty-policy answer.

**Composite tools reuse, not duplicate, logic**
`generate_quote` doesn't touch the products file directly — it calls `get_product_price` and `get_delivery_information` and combines the results. The alternative (reimplementing the lookup inside the quote function) would mean two places to fix the same bug.

**Retries on transient failures only**
`llm.py` retries `APIConnectionError`/`APITimeoutError` with exponential backoff, but never retries `APIError` (auth failures, bad requests) — those will never succeed on retry, so retrying just burns time and money.

**Defensive JSON parsing**
The model is told to return raw JSON, but models occasionally wrap it in markdown fences anyway. Rather than crash on that, the parser strips fences before decoding, and raises a typed `ToolSelectionError` (not a generic exception) on genuinely invalid output.

**Fail-fast configuration**
`app/config.py` reads and validates every required environment variable (`OPENAI_API_KEY`, `APP_API_KEY`) once at import time. A missing key crashes the app on startup, not on the first customer request — a broken deploy should never look like a working one.

**API key auth + per-IP rate limiting**
`/process` requires a shared secret in the `X-API-Key` header (checked with a timing-safe comparison) and is rate-limited per IP via `slowapi`, independent of the API key — a single leaked key can't let one caller monopolize the whole app's paid LLM budget. `/` stays unauthenticated so health checks don't need credentials.

**Contract-stable migration path**
The routing layer started as deterministic, rule-based matching (`V1`) and was swapped for LLM-driven tool selection without changing what callers of `route_customer()` receive back. The API contract didn't move even though the implementation underneath it did.

## Known Limitations

**Memory is in-process only.** `services/memory.py` now keys context by `session_id` and is safe under concurrency (each session gets its own locked entry, with TTL-based expiry), which closes the earlier gap. What it doesn't yet do is survive a restart or work across multiple app instances — the store lives in a single process's memory, so a redeploy or a second replica behind a load balancer starts with a clean slate. Moving it behind Redis (or similar) is the natural next step if KasaFlow ever runs multi-instance.

**CI runs tests, not deploys.** `.github/workflows/tests.yml` runs the test suite automatically on push, but there's no deployment pipeline yet — shipping a change still means someone manually building and pushing the container.

**RAG is a flat document set, not a vector database.** `data/policies.json` holds six short policy documents, embedded and compared in memory with plain cosine similarity — appropriate at this scale, but it doesn't chunk long documents and it would need a real vector store (and a re-embedding job when policies change) before the document set grows past a few dozen entries.

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                        # 66 tests: unit + integration, fast and free
pytest --run-regression       # + prompt regression tests against the real OpenAI API (costs money)
```

Regression tests are opt-in by design — the everyday test loop stays fast, offline, and free, while still allowing a real-API accuracy check when it's actually wanted.

## Running Locally

```bash
# .env (not committed):
# OPENAI_API_KEY=...
# APP_API_KEY=...   (shared secret for the X-API-Key header)

docker compose up --build
```

Then visit `http://localhost:8000/docs` for the interactive API — use the lock icon to authorize with `APP_API_KEY` before calling `/process`.

## Tech Stack

| Layer | Tech |
|---|---|
| API | FastAPI, Uvicorn |
| LLM | OpenAI Responses API (GPT-4.1-mini) |
| Retrieval | OpenAI embeddings API (`text-embedding-3-small`), in-memory cosine similarity |
| Validation | Pydantic |
| Auth / rate limiting | Custom API key dependency, slowapi |
| Config | python-dotenv, centralized fail-fast settings |
| Testing | pytest (unit, integration via `TestClient`, opt-in regression) |
| Containerization | Docker (multi-stage build, non-root user, healthcheck) |

## Roadmap

Docker ✅ · Auth & rate limiting ✅ · Per-session memory ✅ · RAG-backed policy Q&A ✅ · Deployment · Observability · Evaluation · Open-source LLM support (Ollama/vLLM) · LangGraph · Distributed session store (Redis) · Vector database for a larger document set · MCP · Multi-agent workflows

## Status

Self-directed, hardened. Not yet deployed publicly — actively developed as the engine behind a real WhatsApp-based customer workflow project for a jewellery retail client.
