import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import os

class ABSADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):

        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.sentences = self.df.Sentence.values
        self.aspects = self.df.Aspect.values
        self.labels = self.df.Polarity.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        review = str(self.sentences[item])
        aspect = str(self.aspects[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            review,
            aspect,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    MODEL_NAME = 'dbmdz/bert-base-turkish-cased'
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    base_path = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(base_path, 'data', 'train.csv')

    dataset = ABSADataset(train_path, tokenizer)

    sample = dataset[0]

    print("-" * 30)
    print("Sample Sentence:", sample['review_text'])
    print("Tokenize edition (Input IDs):", sample['input_ids'][:20])
    print("Target Label:", sample['targets'])
    print("-" * 30)
    print("Test Successful! Data is suitable for BERT format.")
