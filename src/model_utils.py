import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import spacy
import os
import sys
from config import MODEL_NAME, MODEL_PATH, CLASS_NAMES, MAX_LEN

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

def load_spacy_model():
    try:
        return spacy.load("tr_core_news_md")
    except OSError:
        try:
            return spacy.load("tr_core_news_tr")
        except OSError:
            print("Turkish spaCy model not found. Please install 'tr_core_news_tr' or 'tr_core_news_md'.")
            sys.exit(1)

def load_model_resources(device=None):
    if device is None:
        device = torch.device("cpu")

    print("Loading models...")
    model = SentimentClassifier(n_classes=len(CLASS_NAMES))
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    nlp = load_spacy_model()
    
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

def predict_sentiment_single(model, tokenizer, device, text, aspect):
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

