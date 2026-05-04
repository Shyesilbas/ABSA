import pandas as pd
import os
from tqdm import tqdm
from config import SAMPLE_TWEETS_PATH, FINAL_REPORT_PATH
from model_utils import load_model_resources, extract_aspects, predict_sentiment_single

def read_file_smart(filepath):
    try:
        df = pd.read_csv(filepath, sep=';')
        df.columns = df.columns.str.strip()

        if len(df.columns) > 1:
            return df
    except:
        pass

    try:
        df = pd.read_csv(filepath, sep=',')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Could not read file: {e}")
        return None

def process_batch(input_file, output_file):
    model, tokenizer, nlp, device = load_model_resources()

    if not os.path.exists(input_file):
        print(f"Input file not found : {input_file}")
        return

    df = read_file_smart(input_file)

    if df is None:
        return

    if 'text' not in df.columns:
        for col in df.columns:
            if col.lower() in ['tweet', 'tweets', 'content', 'metin']:
                df.rename(columns={col: 'text'}, inplace=True)
                break

    if 'text' not in df.columns:
        print(f"Error: Could not find a 'text' column. Available columns: {df.columns.tolist()}")
        return

    results = []
    print("Analysis Started...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        text = row.get('text')
        tweet_id = row.get('id', index)

        if not isinstance(text, str) or len(str(text)) < 2:
            continue

        aspects = extract_aspects(nlp, str(text))

        if not aspects:
            results.append({
                'id': tweet_id,
                'text': text,
                'detected_aspect': 'GENERAL',
                'sentiment': 'Neutral',
                'confidence_score': 0.0
            })
            continue

        for aspect in aspects:
            sentiment, conf = predict_sentiment_single(model, tokenizer, device, str(text), aspect)

            results.append({
                'id': tweet_id,
                'text': text,
                'detected_aspect': aspect,
                'sentiment': sentiment,
                'confidence_score': round(conf * 100, 2)
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print("-" * 50)
    print(f"Processed {len(df)} tweets, found {len(results_df)} sentiment targets.")
    print(f"Report saved to: {output_file}")

if __name__ == "__main__":
    process_batch(SAMPLE_TWEETS_PATH, FINAL_REPORT_PATH)
