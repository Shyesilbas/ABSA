# Legacy Code Archive

This directory contains the **original flat-module** versions of the project files
before the refactoring to the current `src/core`, `src/data`, `src/model`, `src/app`
package structure.

## Status: **ARCHIVED — DO NOT USE**

These files are kept for historical reference only. All active development
should target the refactored modules in `src/`.

To see the evolution from these files to the current codebase, use:

```bash
git log --follow --all -- src/config.py
```

## File Mapping

| Legacy File | Current Location |
|---|---|
| `config.py` | `src/core/config.py` |
| `data_download.py` | *(removed — integrated into training_data)* |
| `data_preprocessing.py` | `src/data/contracts.py` + `src/data/training_data.py` |
| `dataset_loader.py` | `src/data/dataset_loader.py` |
| `model_utils.py` | `src/model/inference.py` |
| `train.py` | `src/app/train.py` + `src/model/trainer.py` |
| `predict.py` | `src/app/predict.py` |
| `batch_predict.py` | `src/app/batch_predict.py` |
| `evaulate_metrics.py` | `src/app/evaluate_metrics.py` *(typo fixed)* |
| `visualize_results.py` | `src/app/visualize_results.py` |
| `auto_predict.py` | *(removed — functionality in batch_predict)* |
