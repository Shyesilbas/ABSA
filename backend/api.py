"""Kullanıcıya açık REST uçları (eğitim/değerlendirme yok)."""
from __future__ import annotations

import io
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from app.batch_predict import predict_batch_entries
from app.visualize_results import (
    render_sentiment_distribution_png,
    sentiment_distribution_counts,
)
from backend import state
from backend.schemas import (
    MAX_BATCH_ITEMS,
    MAX_BATCH_UPLOAD_BYTES,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchUploadResponse,
    DistributionStatsResponse,
    HealthResponse,
    MetaResponse,
    PredictRequest,
    PredictResponse,
    VisualizeDistributionRequest,
)
from core.config import (
    CLASS_NAMES,
    CONFIDENCE_FALLBACK_ENABLED,
    CONFIDENCE_FALLBACK_LABEL,
    CONFIDENCE_THRESHOLD,
    MODEL_NAME,
)
from model.inference import predict_sentence_with_meta

router = APIRouter()
TEXT_COLUMN_CANDIDATES = (
    "text",
    "tweet",
    "tweets",
    "content",
    "metin",
    "sentence",
    "yorum",
    "review",
    "message",
)
ID_COLUMN_CANDIDATES = ("id", "row_id", "item_id")


def _ensure_model() -> None:
    err = state.load_error()
    if err is not None:
        raise HTTPException(status_code=503, detail=f"Model yüklenemedi: {err}")
    if not state.model_ready():
        raise HTTPException(status_code=503, detail="Model henüz hazır değil.")


def _resolved_topic_kw(body: VisualizeDistributionRequest) -> tuple[str, str]:
    from core.config import BATCH_TOPIC_KEYWORDS, BATCH_TOPIC_TITLE

    title = body.topic_title.strip() if body.topic_title else BATCH_TOPIC_TITLE
    kw = (
        body.keywords_subtitle.strip()
        if body.keywords_subtitle
        else ", ".join(BATCH_TOPIC_KEYWORDS)
    )
    return title, kw


def _resolved_topic_kw_from_params(topic_title: str | None, keywords_subtitle: str | None) -> tuple[str, str]:
    from core.config import BATCH_TOPIC_KEYWORDS, BATCH_TOPIC_TITLE

    title = topic_title.strip() if topic_title else BATCH_TOPIC_TITLE
    kw = keywords_subtitle.strip() if keywords_subtitle else ", ".join(BATCH_TOPIC_KEYWORDS)
    return title, kw


def _dataframe_for_visualize(body: VisualizeDistributionRequest) -> tuple[pd.DataFrame, str, str, str]:
    title, kw = _resolved_topic_kw(body)
    if body.texts is not None:
        _ensure_model()
        model, tokenizer, device = state.get_model_bundle()
        entries = [{"text": t, "id": i} for i, t in enumerate(body.texts)]
        preds = predict_batch_entries(model, tokenizer, device, entries)
        if not preds:
            raise HTTPException(status_code=422, detail="Geçerli tahmin üretilemedi (metinler çok kısa olabilir).")
        df = pd.DataFrame(preds)
        return df, title, kw, "texts"
    assert body.rows is not None
    for r in body.rows:
        if r.sentiment not in CLASS_NAMES:
            raise HTTPException(
                status_code=422,
                detail=f"Bilinmeyen sentiment: {r.sentiment!r}. Beklenen: {CLASS_NAMES}",
            )
    df = pd.DataFrame([{"sentiment": r.sentiment} for r in body.rows])
    return df, title, kw, "rows"


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _read_upload_csv(content: bytes) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            continue
    raise HTTPException(
        status_code=422,
        detail="CSV okunamadi. UTF-8/UTF-8-SIG kodlamali ve baslik satirli bir dosya yukleyin.",
    )


def _csv_entries_from_dataframe(df: pd.DataFrame, *, max_items: int) -> list[dict]:
    if df.empty:
        raise HTTPException(status_code=422, detail="CSV bos veya gecersiz satirlar iceriyor.")
    if len(df) > max_items:
        raise HTTPException(status_code=422, detail=f"CSV en fazla {max_items} satir icerebilir.")

    text_col = _pick_column(df.columns.tolist(), TEXT_COLUMN_CANDIDATES)
    if text_col is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "CSV icinde metin kolonu bulunamadi. Beklenen kolon isimleri: "
                f"{list(TEXT_COLUMN_CANDIDATES)}"
            ),
        )

    id_col = _pick_column(df.columns.tolist(), ID_COLUMN_CANDIDATES)
    entries: list[dict] = []
    for i, row in df.iterrows():
        text = row.get(text_col)
        if pd.isna(text):
            continue
        text_str = str(text).strip()
        if len(text_str) < 2:
            continue
        rid = row.get(id_col) if id_col else i
        entries.append({"id": rid, "text": text_str})
    return entries


@router.get("/health", response_model=HealthResponse, tags=["Kullanıcı — Sistem"])
def health() -> HealthResponse:
    err = state.load_error()
    if err:
        return HealthResponse(status="degraded", model_loaded=False, detail=err)
    if state.model_ready():
        return HealthResponse(status="ok", model_loaded=True)
    return HealthResponse(status="starting", model_loaded=False)


@router.get("/meta", response_model=MetaResponse, tags=["Kullanıcı — Sistem"])
def meta() -> MetaResponse:
    return MetaResponse(
        model_name=MODEL_NAME,
        class_names=list(CLASS_NAMES),
        confidence_fallback_enabled=CONFIDENCE_FALLBACK_ENABLED,
        confidence_threshold=float(CONFIDENCE_THRESHOLD),
        confidence_fallback_label=CONFIDENCE_FALLBACK_LABEL,
    )


@router.post("/predict", response_model=PredictResponse, tags=["Kullanıcı — Tahmin"])
def predict_one(body: PredictRequest) -> PredictResponse:
    _ensure_model()
    model, tokenizer, device = state.get_model_bundle()
    label, probs, meta = predict_sentence_with_meta(model, tokenizer, device, body.text)
    prob_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return PredictResponse(
        sentiment=label,
        raw_sentiment=meta["raw_label"],
        confidence=meta["confidence"],
        fallback_applied=bool(meta["fallback_applied"]),
        probabilities=prob_map,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["Kullanıcı — Tahmin"])
def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    _ensure_model()
    model, tokenizer, device = state.get_model_bundle()
    entries = [item.model_dump() for item in body.items]
    rows = predict_batch_entries(model, tokenizer, device, entries)
    if not rows:
        raise HTTPException(
            status_code=422,
            detail="Geçerli tahmin üretilemedi (metinler çok kısa veya boş olabilir).",
        )
    return BatchPredictResponse(predictions=rows)


@router.post("/predict/batch/upload", response_model=BatchUploadResponse, tags=["Kullanıcı — Tahmin"])
async def predict_batch_upload(
    file: UploadFile = File(...),
    topic_title: str | None = Form(default=None),
    keywords_subtitle: str | None = Form(default=None),
) -> BatchUploadResponse:
    _ensure_model()
    if not file.filename:
        raise HTTPException(status_code=422, detail="Dosya adi bos olamaz.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Sadece .csv uzantili dosyalar kabul edilir.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Yuklenen CSV bos.")
    if len(content) > MAX_BATCH_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV dosya boyutu en fazla {MAX_BATCH_UPLOAD_BYTES // 1024} KB olabilir.",
        )

    df = _read_upload_csv(content)
    entries = _csv_entries_from_dataframe(df, max_items=MAX_BATCH_ITEMS)
    if not entries:
        raise HTTPException(status_code=422, detail="Gecerli metin satiri bulunamadi (min 2 karakter).")

    model, tokenizer, device = state.get_model_bundle()
    rows = predict_batch_entries(model, tokenizer, device, entries)
    if not rows:
        raise HTTPException(status_code=422, detail="Toplu tahmin sonucu olusturulamadi.")

    title, kw = _resolved_topic_kw_from_params(topic_title, keywords_subtitle)
    counts, total = sentiment_distribution_counts(pd.DataFrame(rows))
    return BatchUploadResponse(
        predictions=rows,
        counts=counts,
        total=total,
        topic_title=title,
        keywords_subtitle=kw,
    )


@router.post(
    "/visualize/distribution",
    response_class=Response,
    tags=["Kullanıcı — Görselleştirme"],
    summary="Duygu dağılımı grafiği (PNG)",
)
def visualize_distribution_png(body: VisualizeDistributionRequest) -> Response:
    df, title, kw, _src = _dataframe_for_visualize(body)
    try:
        png = render_sentiment_distribution_png(df, topic_title=title, keywords_text=kw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return Response(content=png, media_type="image/png")


@router.post(
    "/visualize/distribution/stats",
    response_model=DistributionStatsResponse,
    tags=["Kullanıcı — Görselleştirme"],
    summary="Duygu dağılımı sayımları (JSON)",
)
def visualize_distribution_stats(body: VisualizeDistributionRequest) -> DistributionStatsResponse:
    df, title, kw, src = _dataframe_for_visualize(body)
    try:
        counts, total = sentiment_distribution_counts(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return DistributionStatsResponse(
        counts=counts,
        total=total,
        topic_title=title,
        keywords_subtitle=kw,
        source=src,
    )
