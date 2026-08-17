import logging
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.auth import verify_api_key
from app.config import settings
from app.demo_routes import router as demo_router
from app.logging_config import configure_logging
from app.whatsapp_routes import router as whatsapp_router
from services.product_search import get_product_index
from services.router import route_customer

configure_logging()
logger = logging.getLogger(__name__)


def _warm_product_search_index() -> None:
    """product_search.py's ProductIndex embeds the whole catalogue (3,918
    products, 2 batched OpenAI calls) the first time anything falls
    through to semantic search, and caches it for the rest of the
    process's life. Left lazy, that cost lands on whichever customer's
    message happens to trigger the first fallback -- confirmed live,
    2026-08-16: a plain price-in-a-specific-karat question took over a
    minute to answer. fly.toml also scales this app to zero idle
    machines (min_machines_running = 0), so this isn't a one-off: it can
    recur after every cold start in production, not just on first boot
    locally.

    Run in a background thread rather than blocking startup: Fly's
    healthcheck (fly.toml, 5s grace period) hits "/" shortly after boot,
    and a synchronous multi-second embeddings call here would risk
    failing that healthcheck instead of just slowing down a search.
    Best-effort -- if it fails, the lazy path in product_search.py still
    catches it on the first real request, exactly as it did before this
    existed."""
    def _warm():
        try:
            get_product_index()._ensure_loaded()
            logger.info("Product search index warmed at startup")
        except Exception:
            logger.exception("Failed to warm product search index at startup -- will load lazily instead")

    threading.Thread(target=_warm, daemon=True).start()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _warm_product_search_index()
    yield
    # Nothing to clean up on shutdown -- the warm-up thread is a daemon
    # and the process exiting is enough.


# Per-IP request limiter. Keyed on remote address rather than the API
# key: a single leaked/shared key shouldn't let one caller monopolize
# the whole app's paid LLM budget.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="KasaFlow", version="0.1.0", lifespan=_lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(whatsapp_router)
# Local demo dashboard only -- unauthenticated by design, see
# demo_routes.py's module docstring. Never expose this publicly as-is.
app.include_router(demo_router)


class ProcessRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    # Optional: echo back the session_id from a previous response to let
    # the backend resolve references like "that one" against what this
    # same customer said earlier. Omit it to start a fresh conversation,
    # a new session_id is generated and returned either way.
    session_id: str | None = Field(default=None, max_length=100)


@app.get("/")
def home():
    return {"status": "KasaFlow is running", "model": settings.openai_model}


@app.post("/process")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
def process(
    request: Request,
    payload: ProcessRequest,
    api_key: str = Depends(verify_api_key),
):
    session_id = payload.session_id or str(uuid.uuid4())

    try:
        result = route_customer(payload.message, session_id)
    except Exception:
        # Anything that reaches here is a bug, not an expected failure
        # mode (expected failures are already handled inside
        # route_customer and returned as {"error": ...}).
        logger.exception("Unhandled error while processing request")
        raise HTTPException(status_code=500, detail="Internal server error")

    # Every response carries the session_id, whether the caller supplied
    # one or generated one just now, so a client can thread it through
    # their next request and get context continuity.
    return {**result, "session_id": session_id}
