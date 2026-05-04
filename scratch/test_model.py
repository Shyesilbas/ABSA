
import torch
from model.inference import load_classifier, predict_sentence_with_meta

def test():
    print("Loading model...")
    model, tokenizer, device = load_classifier()
    sentences = [
        "Bugün hava çok güzel, mutluyum.",
        "Kargo çok geç geldi, hiç memnun değilim.",
        "Ürün normal, idare eder.",
        "Berbat bir deneyimdi, sakın almayın.",
        "Harika bir ürün, kesinlikle tavsiye ederim."
    ]
    for s in sentences:
        label, probs, meta = predict_sentence_with_meta(model, tokenizer, device, s)
        print(f"Text: {s}")
        print(f"  Label: {label}, Confidence: {meta['confidence']:.4f}")
        print(f"  Probs: {probs}")

if __name__ == "__main__":
    test()
