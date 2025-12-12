import pandas as pd
from sklearn.model_selection import train_test_split
from config import RAW_DATA_PATH, DATA_DIR, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH

def process_data():
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Raw data len: {len(df)}")
    df = df.dropna()

    print("Label distribution:\n", df['Polarity'].value_counts())

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Polarity'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Polarity'])

    print("-" * 30)
    print(f"Train Set: {len(train_df)} raw")
    print(f"Validation Set: {len(val_df)} raw")
    print(f"Test Set: {len(test_df)} raw")

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"Folders saved to the '{DATA_DIR}' .")

if __name__ == "__main__":
    process_data()
