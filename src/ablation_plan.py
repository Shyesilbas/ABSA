from __future__ import annotations

import itertools
import os

import pandas as pd

from config import ABLATION_REPORT_PATH


def build_plan() -> pd.DataFrame:
    rows = []
    run_id = 0
    for use_hf, merge_hard, fallback in itertools.product([False, True], repeat=3):
        run_id += 1
        rows.append(
            {
                "run_id": f"abl_{run_id:02d}",
                "use_hf_train_extra": use_hf,
                "merge_hard_examples": merge_hard,
                "confidence_fallback_enabled": fallback,
                "requires_retrain": bool(use_hf or merge_hard),
                "notes": "Train + evaluate required for fair comparison.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(os.path.dirname(ABLATION_REPORT_PATH), exist_ok=True)
    df = build_plan()
    df.to_csv(ABLATION_REPORT_PATH, index=False)
    print(df.to_string(index=False))
    print(f"\nAblation plan saved: {ABLATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
