import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import spacy
import os
import warnings

warnings.filterwarnings("ignore")

MODEL_NAME = 'dbmdz/bert-base-turkish-cased'
MAX_LEN = 128
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']

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

def load_resources():
    print("Loading models...")

    device = torch.device("cpu")
    model = SentimentClassifier(n_classes=3)

    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, 'models', 'best_model_state.bin')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    try:
        nlp = spacy.load("tr_core_news_md")
    except OSError:
        try:
            nlp = spacy.load("tr_core_news_tr")
        except OSError:
            print("Turkish spaCy model not found.")
            exit()

    print("System ready.")
    return model, tokenizer, nlp, device

def extract_aspects(nlp, text):
    doc = nlp(str(text))
    aspects = []

    blacklist = {
        'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'bu', 'şu', 'kim', 'ne', 'biri', 'birisi',
        've', 'ile', 'ama', 'fakat', 'lakin', 'veya', 'hem', 'hemde', 'da', 'de', 'ki',
        'gibi', 'kadar', 'diye', 'için', 'üzere', 'yüzden', 'nedeniyle', 'dolayı',
        'bence', 'zaten', 'artık', 'bile', 'sadece', 'daha', 'çok', 'az', 'en', 'pek',
        'gün', 'zaman', 'saat', 'yıl', 'hafta', 'bugün', 'yarın', 'dün', 'sonra', 'önce',
        'şey', 'konu', 'taraf', 'kısım', 'bölüm', 'yer', 'kez', 'kere', 'sefer',
        'lazım', 'gerek', 'var', 'yok', 'tane', 'adet',
        'iyi', 'kötü', 'güzel', 'çirkin', 'fena', 'berbat', 'harika', 'muhteşem',
        'büyük', 'küçük', 'uzun', 'kısa', 'eski', 'yeni', 'zor', 'kolay', 'bozuk', 'sağlam'
    }

    for token in doc:
        lemma = token.lemma_.lower()
        text_lower = token.text.lower()

        if token.pos_ not in ["NOUN", "PROPN"]:
            continue

        if token.is_stop or token.is_punct:
            continue

        if lemma in blacklist or text_lower in blacklist:
            continue

        if len(text_lower) < 3:
            continue

        aspects.append(token.text)

    return list(set(aspects))

def predict_sentiment(model, tokenizer, device, text, aspect):
    text_lower = text.lower()
    if "ne " in text_lower and " ne " in text_lower:
        return "Neutral", 1.0

    encoded = tokenizer.encode_plus(
        text, aspect, max_length=MAX_LEN, add_special_tokens=True,
        return_token_type_ids=False, padding='max_length',
        truncation=True, return_attention_mask=True, return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, prediction = torch.max(output, dim=1)
        conf = torch.max(probs).item()

    return CLASS_NAMES[prediction], conf

if __name__ == "__main__":
    model, tokenizer, nlp, device = load_resources()

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
            sentiment, conf = predict_sentiment(model, tokenizer, device, sentence, aspect)
            print(f"{aspect.ljust(15)} : {sentiment} (Confidence: {conf * 100:.1f}%)")
