import pandas as pd
from sklearn.model_selection import train_test_split
import os


def process_data():
    base_path = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(base_path, 'data', 'turkish_absa_train.csv')

    df = pd.read_csv(input_path)

    print(f"Raw data len: {len(df)}")
    df = df.dropna()

    print("Label distribution:\n", df['Polarity'].value_counts())


    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Polarity'])

    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Polarity'])

    print("-" * 30)
    print(f"Train Set: {len(train_df)} raw")
    print(f"Validation Set: {len(val_df)} raw")
    print(f"Test Set: {len(test_df)} raw")

    data_dir = os.path.join(base_path, 'data')
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

    print(f"Folders saved to the '{data_dir}' .")


if __name__ == "__main__":
    process_data()