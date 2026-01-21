# sentiment_imdb_bert.py
# Required packages: transformers datasets evaluate torch scikit-learn

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
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

# Default configuration
DEFAULT_TRAIN_SAMPLES = 2000
DEFAULT_EVAL_SAMPLES = 1000
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS_FROZEN = 2
DEFAULT_NUM_EPOCHS_UNFROZEN = 1
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_SEED = 42
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
BASE_OUTPUT_DIR = "./sentiment_model_output"

def generate_run_id():
    """Generate timestamped run ID in local timezone"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def setup_logging(log_dir, run_id):
    """Setup logging to both console and file"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{run_id}.log"
    
    # Create logger
    logger = logging.getLogger("sentiment_trainer")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_run_config(config_path, args, device, use_fp16, dataset_sizes):
    """Save run configuration to JSON"""
    config = {
        "run_id": args.run_id,
        "mode": args.mode,
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "seed": args.seed,
        "device": str(device),
        "fp16_enabled": use_fp16,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "val_split": args.val_split,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "num_epochs_frozen": args.num_epochs_frozen,
        "num_epochs_unfrozen": args.num_epochs_unfrozen,
        "learning_rate": args.learning_rate,
        "dataset_sizes": dataset_sizes,
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def save_data_selection_info(output_path, args, train_indices, val_indices, test_indices):
    """Save information about data selection"""
    selection_info = {
        "seed": args.seed,
        "train_samples_requested": args.train_samples,
        "eval_samples_requested": args.eval_samples,
        "val_ratio": args.val_split,
        "method": "shuffle with seed, then select first N samples",
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
        "train_range": f"0-{len(train_indices)-1}",
        "val_range": f"{len(train_indices)}-{len(train_indices)+len(val_indices)-1}",
        "test_range": f"0-{len(test_indices)-1}",
    }
    
    with open(output_path, 'w') as f:
        json.dump(selection_info, f, indent=2)

def load_and_prepare_data(args, logger):
    """Load and prepare IMDb dataset"""
    logger.info(f"Loading IMDb dataset from stanfordnlp/imdb...")
    dataset = load_dataset("stanfordnlp/imdb")
    
    # Select subsets with tracking
    train_full = dataset["train"].shuffle(seed=args.seed)
    train_subset_size = min(args.train_samples, len(train_full))
    train_full = train_full.select(range(train_subset_size))
    
    test_full = dataset["test"].shuffle(seed=args.seed)
    test_subset_size = min(args.eval_samples, len(test_full))
    test_subset = test_full.select(range(test_subset_size))
    
    # Create validation split from train
    split_idx = int(len(train_full) * (1 - args.val_split))
    train_subset = train_full.select(range(split_idx))
    val_subset = train_full.select(range(split_idx, len(train_full)))
    
    logger.info(f"Dataset sizes - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")
    
    # Track indices
    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, len(train_full)))
    test_indices = list(range(test_subset_size))
    
    return train_subset, val_subset, test_subset, train_indices, val_indices, test_indices

def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize dataset"""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length
        )
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return tokenized

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
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

def freeze_base_model(model, logger):
    """Freeze all parameters except classification head"""
    for name, param in model.named_parameters():
        if "classifier" not in name and "pre_classifier" not in name:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

def unfreeze_model(model, logger):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("All parameters unfrozen for fine-tuning")

def save_metrics(output_path, val_metrics, test_metrics, run_id, model_name):
    """Save evaluation metrics to JSON"""
    metrics_data = {
        "run_id": run_id,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)

def save_predictions(output_path, predictions, run_id, model_name):
    """Save predictions to JSON"""
    predictions_data = {
        "run_id": run_id,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
    }
    
    with open(output_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)

def predict_texts(model, tokenizer, texts, device, logger, run_id, model_name):
    """Predict sentiment for given texts"""
    model.eval()
    results = []
    
    logger.info("="*60)
    logger.info("Running predictions:")
    logger.info("="*60)
    
    for text in texts:
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            pos_prob = probs[0][1].item()
        
        label_text = "POSITIVE" if pred_label == 1 else "NEGATIVE"
        
        result = {
            "input_text": text,
            "predicted_label": label_text,
            "prob_positive": round(pos_prob, 4),
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "run_id": run_id,
        }
        results.append(result)
        
        logger.info(f"\nReview: \"{text}\"")
        logger.info(f"Prediction: {label_text}")
        logger.info(f"Positive probability: {pos_prob:.4f}")
    
    logger.info("="*60)
    return results

def train_mode(args, logger):
    """Training mode"""
    # Setup directories
    run_output_dir = Path(BASE_OUTPUT_DIR) / args.run_id
    logs_dir = run_output_dir / "logs"
    model_dir = run_output_dir / "model"
    reports_dir = run_output_dir / "reports"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(logs_dir, args.run_id)
    
    set_seed(args.seed)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = torch.cuda.is_available()
    
    logger.info("="*60)
    logger.info("CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Device: {device}")
    logger.info(f"FP16 training: {use_fp16}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Train samples: {args.train_samples}")
    logger.info(f"Eval samples: {args.eval_samples}")
    logger.info(f"Validation split: {args.val_split}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs (frozen): {args.num_epochs_frozen}")
    logger.info(f"Epochs (unfrozen): {args.num_epochs_unfrozen}")
    logger.info("="*60)
    
    # Load data
    train_data, val_data, test_data, train_indices, val_indices, test_indices = load_and_prepare_data(args, logger)
    
    dataset_sizes = {
        "train": len(train_data),
        "validation": len(val_data),
        "test": len(test_data),
    }
    
    # Save configuration
    save_run_config(reports_dir / "run_config.json", args, device, use_fp16, dataset_sizes)
    save_data_selection_info(reports_dir / f"data_selection_{args.run_id}.json", args, train_indices, val_indices, test_indices)
    
    # Load tokenizer and model
    logger.info(f"\nLoading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    # Tokenize datasets
    logger.info("\nTokenizing datasets...")
    train_tokenized = tokenize_dataset(train_data, tokenizer, args.max_length)
    val_tokenized = tokenize_dataset(val_data, tokenizer, args.max_length)
    test_tokenized = tokenize_dataset(test_data, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Phase 1: Train with frozen base model
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Training classification head (base model frozen)")
    logger.info("="*60)
    
    freeze_base_model(model, logger)
    
    training_args = TrainingArguments(
        output_dir=str(model_dir / "phase1"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs_frozen,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        seed=args.seed,
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
    
    # Get validation metrics after phase 1
    val_metrics_phase1 = trainer.evaluate(val_tokenized)
    logger.info(f"\nPhase 1 Validation Metrics: {val_metrics_phase1}")
    
    # Phase 2: Fine-tune with unfrozen model
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Fine-tuning entire model (all parameters unfrozen)")
    logger.info("="*60)
    
    unfreeze_model(model, logger)
    
    training_args_unfrozen = TrainingArguments(
        output_dir=str(model_dir / "phase2"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate / 10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs_unfrozen,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        seed=args.seed,
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
    
    # Get final validation metrics
    val_metrics_final = trainer_unfrozen.evaluate(val_tokenized)
    logger.info(f"\nFinal Validation Metrics: {val_metrics_final}")
    
    # Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*60)
    
    test_metrics = trainer_unfrozen.evaluate(test_tokenized)
    
    logger.info("\nTest Set Metrics:")
    logger.info(f"  Accuracy:  {test_metrics['eval_accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['eval_precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['eval_recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['eval_f1']:.4f}")
    
    # Save metrics
    save_metrics(
        reports_dir / f"metrics_{args.run_id}.json",
        val_metrics_final,
        test_metrics,
        args.run_id,
        args.model_name
    )
    
    # Save training log history
    if hasattr(trainer_unfrozen.state, 'log_history'):
        with open(reports_dir / f"training_log_history_{args.run_id}.json", 'w') as f:
            json.dump(trainer_unfrozen.state.log_history, f, indent=2)
        logger.info(f"\nSaved training log history to {reports_dir / f'training_log_history_{args.run_id}.json'}")
    
    # Save model and tokenizer
    logger.info(f"\nSaving model and tokenizer to {model_dir}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Test with custom reviews
    custom_reviews = [
        "Incredible film. Just great!",
        "What a waste of time! I'd like to have back those 2 hours of my life.",
        "This movie was absolutely fantastic! Best thing I've seen all year.",
        "Boring and predictable. Not worth watching.",
        "A masterpiece of cinema. Brilliant performances all around.",
        "Terrible acting and awful plot. Complete disaster.",
    ]
    
    predictions = predict_texts(model, tokenizer, custom_reviews, device, logger, args.run_id, args.model_name)
    
    # Save predictions
    save_predictions(
        reports_dir / f"custom_predictions_{args.run_id}.json",
        predictions,
        args.run_id,
        args.model_name
    )
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {run_output_dir}")
    logger.info(f"Model saved to: {model_dir}")
    logger.info(f"Logs saved to: {logs_dir}")
    logger.info(f"Reports saved to: {reports_dir}")
    logger.info("="*60)

def predict_mode(args, logger):
    """Prediction mode"""
    # Setup directories
    run_output_dir = Path(BASE_OUTPUT_DIR) / args.run_id
    logs_dir = run_output_dir / "logs"
    reports_dir = run_output_dir / "reports"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(logs_dir, args.run_id)
    
    logger.info("="*60)
    logger.info("PREDICTION MODE")
    logger.info("="*60)
    logger.info(f"Run ID: {args.run_id}")
    logger.info(f"Loading model from: {args.load_dir}")
    logger.info("="*60)
    
    # Check if load_dir exists
    if not Path(args.load_dir).exists():
        logger.error(f"Model directory not found: {args.load_dir}")
        # Try to find the most recent run
        base_path = Path(BASE_OUTPUT_DIR)
        if base_path.exists():
            runs = sorted([d for d in base_path.iterdir() if d.is_dir()], reverse=True)
            if runs:
                args.load_dir = str(runs[0] / "model")
                logger.info(f"Using most recent run: {args.load_dir}")
            else:
                logger.error("No trained models found. Please train a model first.")
                return
        else:
            logger.error("No trained models found. Please train a model first.")
            return
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    logger.info(f"Loading model and tokenizer from {args.load_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.load_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.load_dir)
    model.to(device)
    
    # Get texts to predict
    texts = []
    
    if args.text:
        texts.append(args.text)
    
    if args.text_file:
        logger.info(f"Reading texts from {args.text_file}")
        try:
            with open(args.text_file, 'r') as f:
                file_texts = [line.strip() for line in f if line.strip()]
                texts.extend(file_texts)
        except FileNotFoundError:
            logger.error(f"Text file not found: {args.text_file}")
            return
    
    if not texts:
        logger.error("No texts provided. Use --text or --text_file")
        return
    
    logger.info(f"Predicting sentiment for {len(texts)} text(s)")
    
    # Make predictions
    predictions = predict_texts(model, tokenizer, texts, device, logger, args.run_id, args.model_name)
    
    # Save predictions
    output_path = reports_dir / f"predict_outputs_{args.run_id}.json"
    save_predictions(output_path, predictions, args.run_id, args.model_name)
    
    logger.info(f"\nPredictions saved to: {output_path}")
    logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description="Sentiment Classification with BERT")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"],
                        help="Mode: train or predict")
    
    # Training parameters
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Pretrained model name")
    parser.add_argument("--train_samples", type=int, default=DEFAULT_TRAIN_SAMPLES,
                        help="Number of training samples")
    parser.add_argument("--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES,
                        help="Number of evaluation samples")
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT,
                        help="Validation split ratio")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--num_epochs_frozen", type=int, default=DEFAULT_NUM_EPOCHS_FROZEN,
                        help="Number of epochs with frozen base model")
    parser.add_argument("--num_epochs_unfrozen", type=int, default=DEFAULT_NUM_EPOCHS_UNFROZEN,
                        help="Number of epochs with unfrozen model")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    
    # Prediction parameters
    parser.add_argument("--load_dir", type=str, default=None,
                        help="Directory to load model from (for predict mode)")
    parser.add_argument("--text", type=str, default=None,
                        help="Single text to predict")
    parser.add_argument("--text_file", type=str, default=None,
                        help="File with texts to predict (one per line)")
    
    args = parser.parse_args()
    
    # Generate run ID
    args.run_id = generate_run_id()
    
    # Create base logger for initial output
    logger = logging.getLogger("sentiment_trainer")
    
    if args.mode == "train":
        train_mode(args, logger)
    elif args.mode == "predict":
        if args.load_dir is None:
            # Try to find most recent run
            base_path = Path(BASE_OUTPUT_DIR)
            if base_path.exists():
                runs = sorted([d for d in base_path.iterdir() if d.is_dir()], reverse=True)
                if runs:
                    args.load_dir = str(runs[0] / "model")
        predict_mode(args, logger)

if __name__ == "__main__":
    main()
