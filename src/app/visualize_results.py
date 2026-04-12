import io
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.config import (
    BATCH_RESULTS_PATH,
    BATCH_TOPIC_KEYWORDS,
    BATCH_TOPIC_TITLE,
    OUTPUTS_DIR,
)

matplotlib.use("Agg")


def prepare_batch_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Batch sonuç CSV/API çerçevesini grafik için normalize eder."""
    out = df.copy()
    out.columns = out.columns.str.strip()
    if "sentiment" not in out.columns:
        raise ValueError("Missing required column: sentiment")
    if "confidence" in out.columns:
        out = out[out["confidence"] >= 0.0]
    return out


def sentiment_distribution_counts(df: pd.DataFrame) -> tuple[dict[str, int], int]:
    """value_counts sözlüğü ve toplam satır sayısı."""
    prep = prepare_batch_results_dataframe(df)
    vc = prep["sentiment"].value_counts()
    counts = {str(k): int(v) for k, v in vc.items()}
    return counts, len(prep)


def render_sentiment_distribution_png(
    df: pd.DataFrame,
    *,
    topic_title: str | None = None,
    keywords_text: str | None = None,
) -> bytes:
    """Duygu dağılımı sütun grafiğini PNG baytları olarak üretir."""
    prep = prepare_batch_results_dataframe(df)
    title = topic_title if topic_title is not None else BATCH_TOPIC_TITLE
    kw = keywords_text if keywords_text is not None else ", ".join(BATCH_TOPIC_KEYWORDS)

    colors = {"Negative": "#FF4B4B", "Neutral": "#7D7D7D", "Positive": "#4CAF50"}
    order = ["Negative", "Neutral", "Positive"]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.countplot(
        x="sentiment",
        data=prep,
        palette=colors,
        order=[o for o in order if o in set(prep["sentiment"].unique())],
    )
    plt.title(f"Sentiment distribution | {title} (n={len(prep)})", fontsize=13, fontweight="bold")
    plt.suptitle(f"Keywords: {kw}", fontsize=10, y=0.98)
    plt.xlabel("Class")
    plt.ylabel("Count")

    for p in ax.patches:
        h = int(p.get_height())
        if h > 0:
            ax.annotate(
                f"{h}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=11,
                xytext=(0, 5),
                textcoords="offset points",
            )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close()
    return buf.getvalue()


def main():
    if not os.path.exists(BATCH_RESULTS_PATH):
        print(f"CSV not found: {BATCH_RESULTS_PATH}")
        print("Run batch_predict first.")
        return

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = pd.read_csv(BATCH_RESULTS_PATH)
    try:
        png = render_sentiment_distribution_png(df)
    except ValueError as e:
        print(str(e))
        return

    out = os.path.join(OUTPUTS_DIR, "chart_sentiment_distribution.png")
    with open(out, "wb") as f:
        f.write(png)
    print(f"Chart saved: {out}")
    prep = prepare_batch_results_dataframe(df)
    print(prep["sentiment"].value_counts())


if __name__ == "__main__":
    main()
