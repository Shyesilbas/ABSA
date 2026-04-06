import os
import pandas as pd
from tqdm import tqdm

from config import SAMPLE_TEXTS_PATH, BATCH_RESULTS_PATH, OUTPUTS_DIR
from model_utils import load_classifier, predict_sentence


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
        print(f"Dosya okunamadı: {e}")
        return None


def process_batch(input_file, output_file):
    model, tokenizer, device = load_classifier()

    if not os.path.exists(input_file):
        print(f"Girdi yok: {input_file}")
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
        print(f"'text' sütunu yok. Sütunlar: {df.columns.tolist()}")
        return

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tahmin"):
        text = row.get("text")
        rid = row.get("id", row.name)
        if not isinstance(text, str) or len(text.strip()) < 2:
            continue
        label, probs = predict_sentence(model, tokenizer, device, text)
        conf = float(max(probs))
        results.append(
            {
                "id": rid,
                "text": text,
                "sentiment": label,
                "confidence": round(conf, 4),
            }
        )

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Satır: {len(results)}  -> {output_file}")


if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    process_batch(SAMPLE_TEXTS_PATH, BATCH_RESULTS_PATH)
