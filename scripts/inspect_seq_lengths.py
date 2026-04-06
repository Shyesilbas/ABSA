#!/usr/bin/env python3
"""train.csv üzerinde kelime uzunluğu özeti (Colab'de de çalıştırılabilir)."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "train.csv"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    if not path.exists():
        print(f"Dosya yok: {path}")
        sys.exit(1)

    sent_words = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s = str(row.get("Sentence") or row.get("text") or "")
            sent_words.append(len(s.split()))

    def stats(name, xs):
        if not xs:
            return
        print(f"{name}: max={max(xs)}, ortalama={sum(xs)/len(xs):.2f}, n={len(xs)}")

    print(f"Dosya: {path}")
    stats("Kelime — cümle (Sentence/text)", sent_words)
    print(
        "\nNot: BERT MAX_LEN alt-kelime (WordPiece) sayısıdır; "
        "kelime sayısından genelde biraz daha büyüktür."
    )


if __name__ == "__main__":
    main()
