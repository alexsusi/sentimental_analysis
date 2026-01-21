# sentiment_imdb_bert.py
# Required packages: transformers datasets evaluate torch scikit-learn

import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_SAMPLES = 2000
EVAL_SAMPLES = 1000
VAL_SPLIT = 0.1
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS_FROZEN = 2
NUM_EPOCHS_UNFROZEN = 1
LEARNING_RATE = 2e-5
SEED = 42
MODEL_NAME = "distilbert-base-uncased"  # Fast, efficient, good for CPU training
OUTPUT_DIR = "./sentiment_model_output"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_prepare_data():
    print(f"Loading IMDb dataset from stanfordnlp/imdb...")
    dataset = load_dataset("stanfordnlp/imdb")
    
    # Select subsets
    train_full = dataset["train"].shuffle(seed=SEED).select(range(min(TRAIN_SAMPLES, len(dataset["train"]))))
    test_subset = dataset["test"].shuffle(seed=SEED).select(range(min(EVAL_SAMPLES, len(dataset["test"]))))
    
    # Create validation split from train
    split_idx = int(len(train_full) * (1 - VAL_SPLIT))
    train_subset = train_full.select(range(split_idx))
    val_subset = train_full.select(range(split_idx, len(train_full)))
    
    print(f"Dataset sizes - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")
    
    return train_subset, val_subset, test_subset

def tokenize_dataset(dataset, tokenizer):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH
        )
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def freeze_base_model(model):
    # Freeze all parameters except classification head
    for name, param in model.named_parameters():
        if "classifier" not in name and "pre_classifier" not in name:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    print("All parameters unfrozen for fine-tuning")

def predict_texts(model, tokenizer, texts, device):
    model.eval()
    results = []
    
    print("\n" + "="*60)
    print("Testing custom reviews:")
    print("="*60)
    
    for text in texts:
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            pos_prob = probs[0][1].item()
        
        label_text = "POSITIVE" if pred_label == 1 else "NEGATIVE"
        results.append((text, label_text, pos_prob))
        
        print(f"\nReview: \"{text}\"")
        print(f"Prediction: {label_text}")
        print(f"Positive probability: {pos_prob:.4f}")
    
    print("="*60 + "\n")
    return results

def main():
    set_seed(SEED)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = torch.cuda.is_available()
    print(f"Using device: {device}")
    print(f"FP16 training: {use_fp16}")
    print(f"Model: {MODEL_NAME}\n")
    
    # Load data
    train_data, val_data, test_data = load_and_prepare_data()
    
    # Load tokenizer and model
    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    train_tokenized = tokenize_dataset(train_data, tokenizer)
    val_tokenized = tokenize_dataset(val_data, tokenizer)
    test_tokenized = tokenize_dataset(test_data, tokenizer)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Phase 1: Train with frozen base model
    print("\n" + "="*60)
    print("PHASE 1: Training classification head (base model frozen)")
    print("="*60)
    
    freeze_base_model(model)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS_FROZEN,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        seed=SEED,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Phase 2: Fine-tune with unfrozen model
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning entire model (all parameters unfrozen)")
    print("="*60)
    
    unfreeze_model(model)
    
    training_args_unfrozen = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE / 10,  # Lower LR for full fine-tuning
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS_UNFROZEN,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        seed=SEED,
        report_to="none",
    )
    
    trainer_unfrozen = Trainer(
        model=model,
        args=training_args_unfrozen,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer_unfrozen.train()
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    test_results = trainer_unfrozen.evaluate(test_tokenized)
    
    print("\nTest Set Metrics:")
    print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"  Precision: {test_results['eval_precision']:.4f}")
    print(f"  Recall:    {test_results['eval_recall']:.4f}")
    print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
    
    # Save model and tokenizer
    print(f"\nSaving model and tokenizer to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Test with custom reviews
    custom_reviews = [
        "Incredible film. Just great!",
        "What a waste of time! I'd like to have back those 2 hours of my life.",
        "This movie was absolutely fantastic! Best thing I've seen all year.",
        "Boring and predictable. Not worth watching.",
        "A masterpiece of cinema. Brilliant performances all around.",
        "Terrible acting and awful plot. Complete disaster.",
    ]
    
    predict_texts(model, tokenizer, custom_reviews, device)
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()