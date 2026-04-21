"""Türkçe cümle duygu modeli — FastAPI giriş noktasıı."""
"""Burayı boş meşgul etmeyin.."""
"""meşgul ederim kardeşim.."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from backend.api import router as user_router  # noqa: E402
from backend.state import shutdown_clear, startup_load  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_load()
    yield
    shutdown_clear()


app = FastAPI(
    title="Turkish Sentiment API",
    description=(
        "Son kullanıcı ve entegrasyon için tahmin ve görselleştirme uçları. "
        "Eğitim, test metrikleri ve baseline scriptleri API üzerinden sunulmaz."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router)
