"""HTTP request/response schemas (public endpoints)."""
from __future__ import annotations

from typing import Annotated, Optional, Union

from pydantic import BaseModel, Field, model_validator

MAX_BATCH_ITEMS = 500
MAX_VISUALIZE_TEXTS = 500
MAX_VISUALIZE_ROWS = 10_000
MAX_BATCH_UPLOAD_BYTES = 1024 * 1024


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: Optional[str] = None


class MetaResponse(BaseModel):
    model_name: str
    class_names: list[str]
    confidence_fallback_enabled: bool
    confidence_threshold: float
    confidence_fallback_label: str
    max_batch_items: int = MAX_BATCH_ITEMS
    max_visualize_texts: int = MAX_VISUALIZE_TEXTS
    max_batch_upload_bytes: int = MAX_BATCH_UPLOAD_BYTES


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class PredictResponse(BaseModel):
    sentiment: str
    raw_sentiment: str
    confidence: float
    fallback_applied: bool
    probabilities: dict[str, float]


class BatchItem(BaseModel):
    id: Optional[Union[str, int]] = None
    text: str = Field(..., min_length=1, max_length=8000)


class BatchPredictRequest(BaseModel):
    items: Annotated[list[BatchItem], Field(min_length=1, max_length=MAX_BATCH_ITEMS)]


class BatchPredictResponse(BaseModel):
    predictions: list[dict]


class BatchUploadResponse(BaseModel):
    predictions: list[dict]
    counts: dict[str, int]
    total: int
    topic_title: str
    keywords_subtitle: str
    source: str = "upload_csv"


class SentimentRow(BaseModel):
    sentiment: str = Field(..., min_length=1, max_length=64)


class VisualizeDistributionRequest(BaseModel):
    """Exactly one of: raw texts (requires model) or pre-labelled sentiment rows."""

    topic_title: Optional[str] = Field(None, max_length=500)
    keywords_subtitle: Optional[str] = Field(None, max_length=500)
    texts: Optional[list[str]] = None
    rows: Optional[list[SentimentRow]] = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> VisualizeDistributionRequest:
        has_t = bool(self.texts)
        has_r = bool(self.rows)
        if has_t == has_r:
            raise ValueError("Exactly one of 'texts' or 'rows' must be provided.")
        if has_t and len(self.texts or []) > MAX_VISUALIZE_TEXTS:
            raise ValueError(f"At most {MAX_VISUALIZE_TEXTS} texts can be sent.")
        if has_r and len(self.rows or []) > MAX_VISUALIZE_ROWS:
            raise ValueError(f"At most {MAX_VISUALIZE_ROWS} rows can be sent.")
        if has_t:
            for t in self.texts or []:
                if len(t) > 8000:
                    raise ValueError("Each text can be at most 8000 characters.")
        return self


class DistributionStatsResponse(BaseModel):
    counts: dict[str, int]
    total: int
    topic_title: str
    keywords_subtitle: str
    source: str  # "texts" | "rows"
