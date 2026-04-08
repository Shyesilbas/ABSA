import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_ROOT_DIR = os.path.join(DATA_DIR, "outputs")
OUTPUT_RUN_NAME = os.getenv("OUTPUT_RUN_NAME", "run_default")
OUTPUTS_DIR = os.path.join(OUTPUTS_ROOT_DIR, OUTPUT_RUN_NAME)

# Raw dataset output
RAW_DATA_PATH = os.path.join(DATA_DIR, "turkish_absa_train.csv")

# Processed sentence-level CSVs (Sentence, Polarity)
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# Batch prediction
SAMPLE_TEXTS_PATH = os.path.join(DATA_DIR, "sample_tweets.csv")
BATCH_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "sentiment_batch_results.csv")
BASELINE_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "baseline_comparison.csv")
BASELINE_CLASS_REPORTS_PATH = os.path.join(OUTPUTS_DIR, "baseline_class_reports.csv")
BATCH_TOPIC_TITLE = "E-commerce campaign complaints"
BATCH_TOPIC_KEYWORDS = [
    "kargo",
    "indirim",
    "iade",
    "stok",
    "müşteri hizmetleri",
]

# Model
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MODEL_PATH = os.path.join(MODELS_DIR, "sentence_best_model.bin")

MAX_LEN = 160
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5

RANDOM_SEED = 42
USE_CLASS_WEIGHTS = True
NEUTRAL_CLASS_INDEX = 1
NEUTRAL_LOSS_BOOST = 1.0

EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-4

USE_AMP = True
WARMUP_RATIO = 0.1
DATALOADER_NUM_WORKERS = 2

# Training pool expansion
MERGE_RAW_ABSA_FOR_TRAIN = True
USE_HF_TRAIN_EXTRA = True
HF_DATASET_ID = "winvoker/turkish-sentiment-analysis-dataset"
HF_SAMPLE_SIZE = 10_000
HF_SEED = 42

# Manually curated hard examples override duplicate sentences
HARD_EXAMPLES_PATH = os.path.join(DATA_DIR, "hard_examples.csv")
MERGE_HARD_EXAMPLES = True

# Inference confidence policy
CONFIDENCE_FALLBACK_ENABLED = True
CONFIDENCE_THRESHOLD = 0.70
CONFIDENCE_FALLBACK_LABEL = "Neutral"

# Quality and experiment artifacts
LEAKAGE_GUARD_ENABLED = True
LEAKAGE_REPORT_PATH = os.path.join(OUTPUTS_DIR, "leakage_report.json")
MISCLASSIFIED_REPORT_PATH = os.path.join(OUTPUTS_DIR, "test_misclassified.csv")
CONFUSION_PAIRS_REPORT_PATH = os.path.join(OUTPUTS_DIR, "test_confusion_pairs.csv")
EXPERIMENT_ARTIFACT_PATH = os.path.join(OUTPUTS_DIR, "experiment_last_run.json")
ABLATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "ablation_plan.csv")
