from model.inference import load_classifier, predict_sentence_with_meta


def main() -> None:
    model, tokenizer, device = load_classifier()

    print("-" * 50)
    print("Sentence-level prediction (type q to quit)")
    print("-" * 50)

    while True:
        sentence = input("\nSentence: ").strip()
        if sentence.lower() == "q":
            break
        if not sentence:
            continue
        label, probs, meta = predict_sentence_with_meta(model, tokenizer, device, sentence)
        fallback_note = " [fallback]" if meta["fallback_applied"] else ""
        print(
            f"Prediction: {label}{fallback_note}  raw={meta['raw_label']}  "
            f"confidence={meta['confidence']:.4f}  probabilities: {[round(p, 4) for p in probs]}"
        )


if __name__ == "__main__":
    main()
