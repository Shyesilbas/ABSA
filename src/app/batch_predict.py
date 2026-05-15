from __future__ import annotations

import os

import pandas as pd

from core.config import BATCH_RESULTS_PATH, OUTPUTS_DIR, SAMPLE_TEXTS_PATH
from core.progress import track
from model.inference import load_classifier


def predict_batch_entries(model, tokenizer, device, entries: list[dict]) -> list[dict]:
    """API or programmatic batch: each entry contains `text` and optional `id`.

    Uses batched tokenisation + forward passes for efficiency.
    """
    import torch
    import torch.nn.functional as F
    from core.config import (
        CLASS_NAMES,
        CONFIDENCE_FALLBACK_ENABLED,
        CONFIDENCE_FALLBACK_LABEL,
        CONFIDENCE_THRESHOLD,
        MAX_LEN,
    )

    # Filter valid entries
    valid: list[tuple[int, str]] = []
    id_map: list = []
    for i, entry in enumerate(entries):
        text = entry.get("text")
        rid = entry.get("id", i)
        if not isinstance(text, str) or len(text.strip()) < 2:
            continue
        valid.append((i, text.strip()))
        id_map.append(rid)

    if not valid:
        return []

    BATCH_SIZE = 32
    results: list[dict] = []

    for start in range(0, len(valid), BATCH_SIZE):
        chunk = valid[start : start + BATCH_SIZE]
        chunk_ids = id_map[start : start + BATCH_SIZE]
        texts = [t for _, t in chunk]

        enc = tokenizer(
            texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits.float()
            probs = F.softmax(logits, dim=-1)
            confs, raw_idxs = probs.max(dim=-1)

        for j in range(len(texts)):
            raw_idx = int(raw_idxs[j].item())
            conf = float(confs[j].item())
            final_idx = raw_idx

            if CONFIDENCE_FALLBACK_ENABLED and conf < float(CONFIDENCE_THRESHOLD):
                if CONFIDENCE_FALLBACK_LABEL in CLASS_NAMES:
                    final_idx = CLASS_NAMES.index(CONFIDENCE_FALLBACK_LABEL)

            fallback_applied = final_idx != raw_idx
            results.append(
                {
                    "id": chunk_ids[j],
                    "text": texts[j],
                    "sentiment": CLASS_NAMES[final_idx],
                    "raw_sentiment": CLASS_NAMES[raw_idx],
                    "fallback_applied": fallback_applied,
                    "confidence": round(conf, 4),
                }
            )

    return results


def read_file_smart(filepath):
    try:
        df = pd.read_csv(filepath, sep=";")
        df.columns = df.columns.str.strip()
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    try:
        df = pd.read_csv(filepath, sep=",")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None


def process_batch(input_file, output_file):
    model, tokenizer, device = load_classifier()

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    df = read_file_smart(input_file)
    if df is None:
        return

    if "text" not in df.columns:
        for col in df.columns:
            if col.lower() in ("tweet", "tweets", "content", "metin", "sentence"):
                df = df.rename(columns={col: "text"})
                break

    if "text" not in df.columns:
        print(f"Missing 'text' column. Available columns: {df.columns.tolist()}")
        return

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Build entries list for the efficient batched pipeline
    entries = []
    for idx, row in df.iterrows():
        text = row.get("text")
        rid = row.get("id", idx)
        if isinstance(text, str) and len(text.strip()) >= 2:
            entries.append({"id": rid, "text": text.strip()})

    print(f"Processing {len(entries)} valid entries from {len(df)} rows...")
    results = predict_batch_entries(model, tokenizer, device, entries)

    fallback_count = sum(1 for r in results if r.get("fallback_applied", False))

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Rows written: {len(results)} -> {output_file}")
    print(f"Confidence fallback applied: {fallback_count}/{len(results)}")


def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    process_batch(SAMPLE_TEXTS_PATH, BATCH_RESULTS_PATH)


if __name__ == "__main__":
    main()
