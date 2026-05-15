"""Turkish sentence sentiment model — FastAPI entry point."""
from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Suppress noisy HuggingFace Hub warnings during cached model loading.
# HF_HUB_DISABLE_IMPLICIT_TOKEN prevents the "unauthenticated requests" message.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backend.api import router as user_router  # noqa: E402
from backend.state import shutdown_clear, startup_load  # noqa: E402


# ── Request ID Middleware ─────────────────────────────────────────────────────
class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique request-id header to every request/response for tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        logger.info(
            "→ %s %s  request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "← %s %s  status=%d  request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_load()
    yield
    shutdown_clear()


app = FastAPI(
    title="Turkish Sentiment API",
    description=(
        "Prediction and visualization endpoints for end-users and integrations. "
        "Training, test metrics, and baseline scripts are not exposed via the API."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8001",
        "http://localhost:8001",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept", "X-Inference-Token", "X-Api-Key", "X-Request-Id"],
)

app.include_router(user_router)
