# src/config.py

# Model Settings
MODEL_NAME = 'dbmdz/bert-base-turkish-cased'
MAX_LEN = 128

# Output Labels
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5