import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from config import (
    MODEL_NAME, MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    TRAIN_DATA_PATH, VAL_DATA_PATH, MODEL_PATH, CLASS_NAMES
)
from model_utils import SentimentClassifier
from dataset_loader import ABSADataset

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data using configured paths
    print("Loading data...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = ABSADataset(TRAIN_DATA_PATH, tokenizer, MAX_LEN)
    val_dataset = ABSADataset(VAL_DATA_PATH, tokenizer, MAX_LEN)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = SentimentClassifier(n_classes=len(CLASS_NAMES))
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    print("Training started!")
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        
        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_dataset)
        )
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

        val_acc, val_loss = eval_model(
            model, val_data_loader, loss_fn, device, len(val_dataset)
        )
        print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = val_acc
            print(f"=> Best model saved to {MODEL_PATH}!")

    print("Training finished!")

if __name__ == "__main__":
    main()

