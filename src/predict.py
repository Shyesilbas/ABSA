from model_utils import load_classifier, predict_sentence

if __name__ == "__main__":
    model, tokenizer, device = load_classifier()

    print("-" * 50)
    print("Cümle düzeyi tahmin (çıkmak için q)")
    print("-" * 50)

    while True:
        sentence = input("\nCümle: ").strip()
        if sentence.lower() == "q":
            break
        if not sentence:
            continue
        label, probs = predict_sentence(model, tokenizer, device, sentence)
        print(f"Tahmin: {label}  olasılıklar: {[round(p, 4) for p in probs]}")
