import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

# Ham veri (indirme çıktısı)
RAW_DATA_PATH = os.path.join(DATA_DIR, "turkish_absa_train.csv")

# İşlenmiş cümle düzeyi CSV (Sentence, Polarity)
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# Toplu tahmin
SAMPLE_TEXTS_PATH = os.path.join(DATA_DIR, "sample_tweets.csv")
BATCH_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "sentiment_batch_results.csv")

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
NEUTRAL_LOSS_BOOST = 1.5

EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-4

USE_AMP = True
WARMUP_RATIO = 0.1
DATALOADER_NUM_WORKERS = 2

# Eğitim verisi genişletme (train.py; Colab script ile aynı mantık)
MERGE_RAW_ABSA_FOR_TRAIN = True
USE_HF_TRAIN_EXTRA = True
HF_DATASET_ID = "winvoker/turkish-sentiment-analysis-dataset"
HF_SAMPLE_SIZE = 10_000
HF_SEED = 42

# Elle seçilmiş zor örnekler (ironi, nötr konuşma vb.); train’de aynı cümle varsa bu etiket baskın
HARD_EXAMPLES_PATH = os.path.join(DATA_DIR, "hard_examples.csv")
MERGE_HARD_EXAMPLES = True
