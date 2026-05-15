"""Multi-seed training script for variance estimation.

Runs the full training pipeline across multiple random seeds to report
mean ± std on validation and test metrics, addressing the single-seed
limitation noted in the project report.

Usage:
    python -m src.app.multi_seed_train [--seeds 42 123 456]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src/ is importable.
_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed training runner")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds to iterate over (default: 42 123 456).",
    )
    args = parser.parse_args()

    # Lazy imports — heavy ML stack.
    import numpy as np

    from core import config

    results: list[dict] = []

    for seed in args.seeds:
        print(f"\n{'=' * 60}")
        print(f"  SEED: {seed}")
        print(f"{'=' * 60}\n")

        # Override the global seed.
        config.RANDOM_SEED = seed
        run_name = f"multi_seed_{seed}"
        config.OUTPUT_RUN_NAME = run_name

        # Re-run the full train pipeline for this seed.
        from app import train as train_module

        train_module.main()

        # Read the experiment log produced by train.main().
        log_path = config.OUTPUTS_DIR / "experiment_last_run.json"
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            results.append(
                {
                    "seed": seed,
                    "run_name": run_name,
                    "best_val_macro_f1": log.get("best_val_macro_f1"),
                    "epochs": log.get("epochs_run"),
                }
            )
        else:
            print(f"[WARN] No experiment log found for seed {seed}.")
            results.append({"seed": seed, "run_name": run_name})

    # Summary
    val_f1s = [r["best_val_macro_f1"] for r in results if r.get("best_val_macro_f1")]
    if val_f1s:
        mean_f1 = float(np.mean(val_f1s))
        std_f1 = float(np.std(val_f1s))
        print(f"\n{'=' * 60}")
        print(f"  MULTI-SEED SUMMARY ({len(val_f1s)} seeds)")
        print(f"{'=' * 60}")
        print(f"  Best Val Macro-F1: {mean_f1:.6f} ± {std_f1:.6f}")
        for r in results:
            print(f"    seed={r['seed']}: {r.get('best_val_macro_f1', 'N/A')}")
    else:
        print("\n[WARN] No valid results collected.")

    # Save summary
    summary_path = _ROOT / "data" / "outputs" / "multi_seed_summary.json"
    os.makedirs(summary_path.parent, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "seeds": args.seeds,
                "results": results,
                "mean_val_macro_f1": float(np.mean(val_f1s)) if val_f1s else None,
                "std_val_macro_f1": float(np.std(val_f1s)) if val_f1s else None,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
