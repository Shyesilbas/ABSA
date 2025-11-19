import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import spacy
import os

# --- CONFIGURATION ---
MODEL_NAME = 'dbmdz/bert-base-turkish-cased'
MAX_LEN = 128
# Customized labels with emojis for better UI
class_names = ['Negative 😡', 'Neutral 😐', 'Positive 😃']


# --- 1. MODEL ARCHITECTURE ---
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


# --- 2. LOADERS ---
def load_resources():
    print("⏳ Loading AI Models (BERT + spaCy)... Please wait.")

    # A. Load Sentiment Model (BERT)
    device = torch.device("cpu")  # Force CPU for your Mac
    model = SentimentClassifier(n_classes=3)

    # Path handling
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, 'models', 'best_model_state.bin')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # B. Load Aspect Extractor (spaCy - Turkish)
    try:
        nlp = spacy.load("tr_core_news_lg")
        except OSError:
        print("❌ ERROR: Turkish spaCy model not found.")
        print("Please run: python -m spacy download tr_core_news_tr")
        exit()

    print("✅ All systems ready!")
    return model, tokenizer, nlp, device


# --- 3. ASPECT EXTRACTION LOGIC ---
def extract_aspects(nlp, text):
    """
    Finds nouns and proper nouns in the sentence to use as potential aspects.
    """
    doc = nlp(text)
    aspects = []

    for token in doc:
        # Filter: Get Nouns/Proper Nouns, ignore punctuation and stop words
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and not token.is_punct:
            aspects.append(token.text)

    return list(set(aspects))  # Remove duplicates


# --- 4. PREDICTION LOGIC ---
def predict_sentiment(model, tokenizer, device, text, aspect):
    encoded_review = tokenizer.encode_plus(
        text,
        aspect,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, prediction = torch.max(output, dim=1)
        confidence = torch.max(probs).item()

    return class_names[prediction], confidence


# --- MAIN APP LOOP ---
if __name__ == "__main__":
    model, tokenizer, nlp, device = load_resources()

    print("\n" + "=" * 60)
    print("🤖 AUTOMATIC ASPECT-BASED SENTIMENT ANALYSIS")
    print("Type a sentence, and I will find the topics and judge them.")
    print("Type 'q' to quit.")
    print("=" * 60)

    while True:
        sentence = input("\n📝 Enter a sentence: ")
        if sentence.lower() == 'q':
            print("Goodbye! 👋")
            break

        # Step 1: Auto-detect aspects
        found_aspects = extract_aspects(nlp, sentence)

        if not found_aspects:
            print("⚠️ No specific aspects found in this sentence.")
            continue

        print(f"🔎 Detected Aspects: {found_aspects}")
        print("-" * 40)

        # Step 2: Analyze each aspect
        for aspect in found_aspects:
            sentiment, conf = predict_sentiment(model, tokenizer, device, sentence, aspect)

            # Formatting the output
            # Example: "Food    : Positive (98%)"
            print(f"👉 {aspect.ljust(15)} : {sentiment} (Confidence: {conf * 100:.1f}%)")