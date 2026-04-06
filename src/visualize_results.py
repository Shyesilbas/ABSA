import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import BATCH_RESULTS_PATH, OUTPUTS_DIR


def main():
    if not os.path.exists(BATCH_RESULTS_PATH):
        print(f"CSV yok: {BATCH_RESULTS_PATH}")
        print("Önce batch_predict çalıştırın.")
        return

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = pd.read_csv(BATCH_RESULTS_PATH)
    df.columns = df.columns.str.strip()

    if "sentiment" not in df.columns:
        print("Beklenen sütun: sentiment")
        return

    if "confidence" in df.columns:
        df = df[df["confidence"] >= 0.0]

    colors = {"Negative": "#FF4B4B", "Neutral": "#7D7D7D", "Positive": "#4CAF50"}
    order = ["Negative", "Neutral", "Positive"]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.countplot(
        x="sentiment",
        data=df,
        palette=colors,
        order=[o for o in order if o in set(df["sentiment"].unique())],
    )
    plt.title(f"Duygu dağılımı (n={len(df)})", fontsize=14, fontweight="bold")
    plt.xlabel("Sınıf")
    plt.ylabel("Adet")

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

    out = os.path.join(OUTPUTS_DIR, "chart_sentiment_distribution.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Grafik: {out}")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
