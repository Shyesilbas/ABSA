from model_utils import load_model_resources, predict_sentiment_single

if __name__ == "__main__":
    model, tokenizer, _, device = load_model_resources()

    print("-" * 50)
    print("TEST PHASE (Press 'q' to quit.)")
    print("-" * 50)

    while True:
        sentence = input("\nEnter a sentence: ")
        if sentence.lower() == 'q':
            break

        aspect = input("Enter the aspect: ")

        sentiment, _ = predict_sentiment_single(model, tokenizer, device, sentence, aspect)
        print(f"Model Says: {sentiment}")
