import warnings
from model_utils import load_model_resources, extract_aspects, predict_sentiment_single

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    model, tokenizer, nlp, device = load_model_resources()

    print("\n" + "=" * 60)
    print("ASPECT-BASED SENTIMENT ANALYSIS")
    print("Type a sentence to find topics and sentiments.")
    print("Type 'q' to quit.")
    print("=" * 60)

    while True:
        sentence = input("\nEnter a sentence: ")
        if sentence.lower() == 'q':
            print("Goodbye.")
            break

        found_aspects = extract_aspects(nlp, sentence)

        if not found_aspects:
            print("No specific aspects found.")
            continue

        print(f"Detected Aspects: {found_aspects}")
        print("-" * 40)

        for aspect in found_aspects:
            sentiment, conf = predict_sentiment_single(model, tokenizer, device, sentence, aspect)
            print(f"{aspect.ljust(15)} : {sentiment} (Confidence: {conf * 100:.1f}%)")
