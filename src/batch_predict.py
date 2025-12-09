import pandas as pd
import torch
from transformers import BertTokenizer
import spacy
import os
from tqdm import tqdm
from config import MODEL_NAME, MAX_LEN, CLASS_NAMES
from auto_predict import SentimentClassifier, extract_aspects, predict_sentiment


def load_resources_for_batch():
    print("⏳ Resources loading (Model + SpaCy)...")

    device = torch.device("cpu")
    model = SentimentClassifier(n_classes=3)

    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, 'models', 'best_model_state.bin')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model folder not found at PATH : {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Load SpaCy (Try different versions to be safe)
    try:
        nlp = spacy.load("tr_core_news_md")
    except:
        try:
            nlp = spacy.load("tr_core_news_tr")
        except:
            print("❌ SpaCy model not found! Please install 'tr_core_news_tr'.")
            exit()

    print("✅ All resources ready!")
    return model, tokenizer, nlp, device


def read_file_smart(filepath):
    """
    Tries to read CSV with different separators (semicolon or comma).
    Fixes the issue where data is separated by ';'.
    """
    try:
        # Priority 1: Try Semicolon (;) which is what your file uses
        df = pd.read_csv(filepath, sep=';')
        # Clean column names (remove spaces like 'text ' -> 'text')
        df.columns = df.columns.str.strip()

        # Check if it actually worked (does it have columns?)
        if len(df.columns) > 1:
            return df
    except:
        pass

    # Priority 2: Try Comma (,)
    try:
        df = pd.read_csv(filepath, sep=',')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        return None


def process_batch(input_file, output_file):
    model, tokenizer, nlp, device = load_resources_for_batch()

    if not os.path.exists(input_file):
        print(f"❌ Input file not found : {input_file}")
        return

    # Use the smart reader function
    df = read_file_smart(input_file)

    if df is None:
        return

    # If column is named 'Tweet' or 'Tweets', rename it to 'text'
    if 'text' not in df.columns:
        for col in df.columns:
            if col.lower() in ['tweet', 'tweets', 'content', 'metin']:
                df.rename(columns={col: 'text'}, inplace=True)
                break

    # Final check for text column
    if 'text' not in df.columns:
        print(f"❌ Error: Could not find a 'text' column. Available columns: {df.columns.tolist()}")
        return

    results = []
    print("🚀 Analysis Started...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        text = row.get('text')
        tweet_id = row.get('id', index)

        # Skip empty or very short texts
        if not isinstance(text, str) or len(str(text)) < 2:
            continue

        # 1. Extract Aspects
        aspects = extract_aspects(nlp, str(text))

        # 2. If no aspect found, label as General
        if not aspects:
            results.append({
                'id': tweet_id,
                'text': text,
                'detected_aspect': 'GENERAL',
                'sentiment': 'Neutral',
                'confidence_score': 0.0
            })
            continue

        # 3. Predict sentiment for each aspect
        for aspect in aspects:
            sentiment, conf = predict_sentiment(model, tokenizer, device, str(text), aspect)

            results.append({
                'id': tweet_id,
                'text': text,
                'detected_aspect': aspect,
                'sentiment': sentiment,
                'confidence_score': round(conf * 100, 2)
            })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print("-" * 50)
    print(f"Processed {len(df)} tweets, found {len(results_df)} sentiment targets.")
    print(f" Report saved to: {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_csv = os.path.join(base_dir, 'data', 'sample_tweets.csv')
    output_csv = os.path.join(base_dir, 'data', 'final_report.csv')

    process_batch(input_csv, output_csv)