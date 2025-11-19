import pandas as pd
from datasets import load_dataset
import os

def download_and_save_data():
    print("Dataset is downloading...")
    dataset = load_dataset("Sengil/Turkish-ABSA-Wsynthetic")

    df_train = pd.DataFrame(dataset['train'])

    print("-" * 30)
    print("Column names:", df_train.columns.tolist())
    print(f"Total data length: {len(df_train)}")
    print("-" * 30)
    print("First 5 rows:")
    print(df_train.head())

    if 'label' in df_train.columns:
        print("-" * 30)
        print("Labels:")
        print(df_train['label'].value_counts())

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'turkish_absa_train.csv')

    df_train.to_csv(output_path, index=False)
    print(f"\nSuccessful! Data saved to: {output_path}")


if __name__ == "__main__":
    download_and_save_data()