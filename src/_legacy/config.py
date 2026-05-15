import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model_state.bin')

# Data Files
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_DATA_PATH = os.path.join(DATA_DIR, 'val.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'turkish_absa_train.csv')
SAMPLE_TWEETS_PATH = os.path.join(DATA_DIR, 'sample_tweets.csv')
FINAL_REPORT_PATH = os.path.join(DATA_DIR, 'final_report.csv')

# Model Settings
MODEL_NAME = 'dbmdz/bert-base-turkish-cased'
MAX_LEN = 128
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
