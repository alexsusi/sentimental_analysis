"""
Data handling module for sentiment analysis pipeline.

This module provides comprehensive data loading, preprocessing, and
tokenization capabilities for sentiment analysis tasks.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import random
import numpy as np
import torch
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoTokenizer

try:
    from .config import SentimentConfig, DataConfig
except ImportError:
    from config import SentimentConfig, DataConfig


logger = logging.getLogger(__name__)


class DataManager:
    """Manages data loading, preprocessing, and tokenization for sentiment analysis."""

    def __init__(self, config: SentimentConfig):
        """
        Initialize DataManager with configuration.

        Args:
            config: SentimentConfig containing data settings
        """
        self.config = config
        self.data_config = config.data
        self.tokenizer: Optional[AutoTokenizer] = None
        self.dataset: Optional[DatasetDict] = None

        # Set random seed for reproducibility
        self.set_seed(config.system.seed)

    def set_seed(self, seed: int) -> None:
        """
        Set random seeds for reproducibility.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_dataset(self) -> DatasetDict:
        """
        Load and prepare the IMDb dataset.

        Returns:
            DatasetDict: Prepared dataset with train, validation, and test splits

        Raises:
            RuntimeError: If dataset loading fails
            ValueError: If invalid split names are provided
        """
        try:
            logger.info(f"Loading dataset '{self.data_config.dataset_name}'...")

            # Load the full dataset
            full_dataset = load_dataset(
                self.data_config.dataset_name, cache_dir=self.data_config.cache_dir
            )

            # Ensure required splits exist
            required_splits = ["train", "test"]
            for split in required_splits:
                if split not in full_dataset:
                    raise ValueError(f"Dataset missing required split: {split}")

            # Store original dataset
            self.dataset = full_dataset

            # Process and split the data
            processed_dataset = self._process_dataset(full_dataset)

            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  Train: {len(processed_dataset['train'])} samples")
            logger.info(f"  Validation: {len(processed_dataset['validation'])} samples")
            logger.info(f"  Test: {len(processed_dataset['test'])} samples")

            return processed_dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise RuntimeError(f"Dataset loading failed: {str(e)}")

    def _process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Process the raw dataset into train/val/test splits.

        Args:
            dataset: Raw dataset from HuggingFace

        Returns:
            DatasetDict: Processed dataset with proper splits
        """
        # Shuffle and select training subset
        train_data = dataset["train"].shuffle(seed=self.data_config.train_samples)
        train_subset_size = min(self.data_config.train_samples, len(train_data))
        train_data = train_data.select(range(train_subset_size))

        # Shuffle and select test subset
        test_data = dataset["test"].shuffle(seed=self.data_config.eval_samples)
        test_subset_size = min(self.data_config.eval_samples, len(test_data))
        test_data = test_data.select(range(test_subset_size))

        # Create validation split from training data
        val_size = int(len(train_data) * self.data_config.val_split)
        val_data = train_data.select(range(len(train_data) - val_size, len(train_data)))
        train_data = train_data.select(range(len(train_data) - val_size))

        # Create processed dataset dict
        processed_dataset = DatasetDict(
            {"train": train_data, "validation": val_data, "test": test_data}
        )

        # Store data selection info for reproducibility
        self._save_data_info(processed_dataset)

        return processed_dataset

    def _save_data_info(self, dataset: DatasetDict) -> None:
        """
        Save information about data selection for reproducibility.

        Args:
            dataset: Processed dataset
        """
        data_info = {
            "dataset_name": self.data_config.dataset_name,
            "seed": self.config.system.seed,
            "train_samples_requested": self.data_config.train_samples,
            "eval_samples_requested": self.data_config.eval_samples,
            "val_split": self.data_config.val_split,
            "method": "shuffle with seed, then select first N samples",
            "final_sizes": {
                "train": len(dataset["train"]),
                "validation": len(dataset["validation"]),
                "test": len(dataset["test"]),
            },
            "text_column": self.data_config.text_column,
            "label_column": self.data_config.label_column,
        }

        # This will be saved later in the main training function
        self.data_selection_info = data_info

    def initialize_tokenizer(self, model_name: str) -> AutoTokenizer:
        """
        Initialize tokenizer for the specified model.

        Args:
            model_name: Name of the pretrained model

        Returns:
            AutoTokenizer: Initialized tokenizer

        Raises:
            RuntimeError: If tokenizer initialization fails
        """
        try:
            logger.info(f"Initializing tokenizer for model: {model_name}")

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, cache_dir=self.data_config.cache_dir
            )

            # Validate tokenizer
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            self.tokenizer = tokenizer
            logger.info(
                f"Tokenizer initialized successfully. Vocabulary size: {tokenizer.vocab_size}"
            )

            return tokenizer

        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {str(e)}")
            raise RuntimeError(f"Tokenizer initialization failed: {str(e)}")

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Tokenize the dataset using the initialized tokenizer.

        Args:
            dataset: Dataset to tokenize

        Returns:
            DatasetDict: Tokenized dataset

        Raises:
            RuntimeError: If tokenization fails or tokenizer not initialized
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Call initialize_tokenizer() first."
            )

        try:
            logger.info("Tokenizing dataset...")

            def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
                """
                Tokenization function for dataset mapping.

                Args:
                    examples: Batch of examples from dataset

                Returns:
                    Dict[str, Any]: Tokenized examples
                """
                return self.tokenizer(
                    examples[self.data_config.text_column],
                    truncation=True,
                    padding=False,  # Will be handled by DataCollatorWithPadding
                    max_length=self.data_config.max_length,
                    return_attention_mask=True,
                )

            # Apply tokenization to all splits
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                remove_columns=[self.data_config.text_column],
                desc="Tokenizing",
            )

            # Rename label column if needed
            if self.data_config.label_column != "label":
                tokenized_dataset = tokenized_dataset.rename_column(
                    self.data_config.label_column, "label"
                )

            # Set dataset format for PyTorch
            tokenized_dataset.set_format("torch")

            logger.info("Dataset tokenized successfully")
            return tokenized_dataset

        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            raise RuntimeError(f"Tokenization failed: {str(e)}")

    def get_data_collator(self):
        """
        Get data collator for batching and padding.

        Returns:
            DataCollatorWithPadding: Configured data collator
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        from transformers import DataCollatorWithPadding

        return DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.data_config.max_length,
            return_tensors="pt",
        )

    def validate_dataset(self, dataset: DatasetDict) -> bool:
        """
        Validate dataset format and content.

        Args:
            dataset: Dataset to validate

        Returns:
            bool: True if valid

        Raises:
            ValueError: If dataset validation fails
        """
        required_splits = ["train", "validation", "test"]
        required_columns = ["input_ids", "attention_mask", "label"]

        # Check splits
        for split in required_splits:
            if split not in dataset:
                raise ValueError(f"Missing required dataset split: {split}")

        # Check columns in each split
        for split_name, split_data in dataset.items():
            missing_cols = [
                col for col in required_columns if col not in split_data.column_names
            ]
            if missing_cols:
                raise ValueError(
                    f"Split '{split_name}' missing columns: {missing_cols}"
                )

            # Check data types
            if len(split_data) > 0:
                example = split_data[0]

                # Check input_ids
                if not isinstance(example["input_ids"], torch.Tensor):
                    raise ValueError(
                        f"input_ids should be torch.Tensor in split '{split_name}'"
                    )

                # Check attention_mask
                if not isinstance(example["attention_mask"], torch.Tensor):
                    raise ValueError(
                        f"attention_mask should be torch.Tensor in split '{split_name}'"
                    )

                # Check label
                if not isinstance(example["label"], torch.Tensor):
                    raise ValueError(
                        f"label should be torch.Tensor in split '{split_name}'"
                    )

        logger.info("Dataset validation passed")
        return True

    def get_dataset_statistics(self, dataset: DatasetDict) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive statistics about the dataset.

        Args:
            dataset: Dataset to analyze

        Returns:
            Dict[str, Dict[str, Any]]: Statistics for each dataset split
        """
        stats = {}

        for split_name, split_data in dataset.items():
            split_stats = {
                "num_samples": len(split_data),
                "num_positive": 0,
                "num_negative": 0,
                "avg_text_length": 0,
                "max_text_length": 0,
                "min_text_length": float("inf"),
            }

            # Analyze text lengths and labels
            text_lengths = []

            for example in split_data:
                # Count labels
                if example["label"].item() == 1:
                    split_stats["num_positive"] += 1
                else:
                    split_stats["num_negative"] += 1

                # Calculate text length (number of tokens)
                if hasattr(example, "text") and example["text"]:
                    text_length = len(example["text"].split())
                else:
                    # Use input_ids length if original text not available
                    text_length = example["input_ids"].numel()

                text_lengths.append(text_length)

            if text_lengths:
                split_stats["avg_text_length"] = np.mean(text_lengths)
                split_stats["max_text_length"] = max(text_lengths)
                split_stats["min_text_length"] = min(text_lengths)

            # Calculate label distribution
            split_stats["positive_ratio"] = (
                split_stats["num_positive"] / split_stats["num_samples"]
            )
            split_stats["negative_ratio"] = (
                split_stats["num_negative"] / split_stats["num_samples"]
            )

            stats[split_name] = split_stats

        return stats

    def save_dataset_info(self, dataset: DatasetDict, output_path: Path) -> None:
        """
        Save dataset information and statistics to file.

        Args:
            dataset: Dataset to analyze
            output_path: Path to save the information
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get dataset statistics
        stats = self.get_dataset_statistics(dataset)

        # Combine with data selection info
        dataset_info = {
            "data_selection_info": getattr(self, "data_selection_info", {}),
            "dataset_statistics": stats,
            "tokenization_info": {
                "max_length": self.data_config.max_length,
                "tokenizer_model": self.tokenizer.name_or_path
                if self.tokenizer
                else None,
                "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, default=str)

        logger.info(f"Dataset information saved to: {output_path}")
