import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import os

MODEL_NAME = 'dbmdz/bert-base-turkish-cased'
MAX_LEN = 128
class_names = ['Negative', 'Neutral', 'Positive']

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

def load_model():
    print("Model is loading, please wait...")

    device = torch.device("cpu")

    model = SentimentClassifier(n_classes=3)

    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, 'models', 'best_model_state.bin')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder not found! Please check the PATH: '{model_path}' .")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Model Loaded successfully!")
    return model, device

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
        _, prediction = torch.max(output, dim=1)

    return class_names[prediction]

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model, device = load_model()

    print("-" * 50)
    print("TEST PHASE (Press 'q' to quit.)")
    print("-" * 50)

    while True:
        sentence = input("\nEnter a sentence: ")
        if sentence.lower() == 'q':
            break

        aspect = input("Enter the aspect: ")

        sentiment = predict_sentiment(model, tokenizer, device, sentence, aspect)
        print(f"Model Says: {sentiment}")
