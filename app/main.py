import logging
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.auth import verify_api_key
from app.config import settings
from app.logging_config import configure_logging
from services.router import route_customer

configure_logging()
logger = logging.getLogger(__name__)

# Per-IP request limiter. Keyed on remote address rather than the API
# key: a single leaked/shared key shouldn't let one caller monopolize
# the whole app's paid LLM budget.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="KasaFlow", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
