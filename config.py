"""
Configuration management module for sentiment analysis pipeline.

This module provides comprehensive configuration management with support for
YAML and JSON configuration files, command-line argument overrides, and
environment variable substitution.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import argparse


@dataclass
class DataConfig:
    """Configuration for data handling and preprocessing."""

    dataset_name: str = "stanfordnlp/imdb"
    train_samples: int = 2000
    eval_samples: int = 1000
    val_split: float = 0.1
    max_length: int = 256
    batch_size: int = 16
    cache_dir: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""

    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    dropout_rate: float = 0.1
    id2label: Dict[int, str] = None
    label2id: Dict[str, int] = None

    def __post_init__(self):
        if self.id2label is None:
            self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        if self.label2id is None:
            self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    num_epochs_frozen: int = 2
    num_epochs_unfrozen: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_clip_val: float = 1.0
    accumulation_steps: int = 1
    fp16: bool = True
    dataloader_num_workers: int = 0
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    logging_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and reporting."""

    compute_metrics: bool = True
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    threshold: float = 0.5

    def __post_init__(self):
        self.metrics_list = ["accuracy", "precision", "recall", "f1"]


@dataclass
class SystemConfig:
    """Configuration for system-level settings."""

    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    output_dir: str = "./sentiment_model_output"
    log_level: str = "INFO"
    experiment_tracking: bool = False
    checkpoint_resume: bool = True


@dataclass
class SentimentConfig:
    """Main configuration class that combines all sub-configurations."""

    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    system: SystemConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.system is None:
            self.system = SystemConfig()


class ConfigManager:
    """Manages configuration loading, saving, and validation."""

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> SentimentConfig:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            SentimentConfig: Loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid configuration file format: {e}")

        return ConfigManager._dict_to_config(config_dict)

    @staticmethod
    def save_config(config: SentimentConfig, config_path: Union[str, Path]) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            config_path: Output path for configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(config)

        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    @staticmethod
    def create_default_config(config_path: Union[str, Path]) -> SentimentConfig:
        """
        Create and save a default configuration file.

        Args:
            config_path: Path where to save the default config

        Returns:
            SentimentConfig: The default configuration
        """
        config = SentimentConfig()
        ConfigManager.save_config(config, config_path)
        return config

    @staticmethod
    def merge_cli_args(
        config: SentimentConfig, args: argparse.Namespace
    ) -> SentimentConfig:
        """
        Merge command-line arguments with configuration.

        Args:
            config: Base configuration
            args: Command-line arguments

        Returns:
            SentimentConfig: Merged configuration
        """
        # Data config overrides
        if hasattr(args, "train_samples") and args.train_samples:
            config.data.train_samples = args.train_samples
        if hasattr(args, "eval_samples") and args.eval_samples:
            config.data.eval_samples = args.eval_samples
        if hasattr(args, "val_split") and args.val_split:
            config.data.val_split = args.val_split
        if hasattr(args, "max_length") and args.max_length:
            config.data.max_length = args.max_length
        if hasattr(args, "batch_size") and args.batch_size:
            config.data.batch_size = args.batch_size

        # Model config overrides
        if hasattr(args, "model_name") and args.model_name:
            config.model.model_name = args.model_name

        # Training config overrides
        if hasattr(args, "num_epochs_frozen") and args.num_epochs_frozen:
            config.training.num_epochs_frozen = args.num_epochs_frozen
        if hasattr(args, "num_epochs_unfrozen") and args.num_epochs_unfrozen:
            config.training.num_epochs_unfrozen = args.num_epochs_unfrozen
        if hasattr(args, "learning_rate") and args.learning_rate:
            config.training.learning_rate = args.learning_rate

        # System config overrides
        if hasattr(args, "seed") and args.seed:
            config.system.seed = args.seed
        if hasattr(args, "output_dir") and args.output_dir:
            config.system.output_dir = args.output_dir

        return config

    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> SentimentConfig:
        """Convert dictionary to SentimentConfig object."""
        # Extract nested configurations
        data_dict = config_dict.get("data", {})
        model_dict = config_dict.get("model", {})
        training_dict = config_dict.get("training", {})
        evaluation_dict = config_dict.get("evaluation", {})
        system_dict = config_dict.get("system", {})

        # Create config objects
        data_config = DataConfig(**data_dict)
        model_config = ModelConfig(**model_dict)
        training_config = TrainingConfig(**training_dict)
        evaluation_config = EvaluationConfig(**evaluation_dict)
        system_config = SystemConfig(**system_dict)

        return SentimentConfig(
            data=data_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config,
            system=system_config,
        )

    @staticmethod
    def validate_config(config: SentimentConfig) -> bool:
        """
        Validate configuration parameters.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if valid

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate data config
        if config.data.train_samples <= 0:
            raise ValueError("train_samples must be positive")
        if config.data.eval_samples <= 0:
            raise ValueError("eval_samples must be positive")
        if not 0 < config.data.val_split < 1:
            raise ValueError("val_split must be between 0 and 1")
        if config.data.max_length <= 0:
            raise ValueError("max_length must be positive")
        if config.data.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Validate model config
        if config.model.num_labels <= 0:
            raise ValueError("num_labels must be positive")
        if not 0 <= config.model.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        # Validate training config
        if config.training.num_epochs_frozen < 0:
            raise ValueError("num_epochs_frozen must be non-negative")
        if config.training.num_epochs_unfrozen < 0:
            raise ValueError("num_epochs_unfrozen must be non-negative")
        if config.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if config.training.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        # Validate system config
        if config.system.seed < 0:
            raise ValueError("seed must be non-negative")

        return True


def setup_cli_parser() -> argparse.ArgumentParser:
    """
    Setup command-line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Sentiment Classification with BERT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "predict", "eval"],
        help="Operation mode",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON)",
    )

    # Data parameters
    parser.add_argument("--train_samples", type=int, help="Number of training samples")
    parser.add_argument("--eval_samples", type=int, help="Number of evaluation samples")
    parser.add_argument("--val_split", type=float, help="Validation split ratio")
    parser.add_argument("--max_length", type=int, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, help="Batch size")

    # Model parameters
    parser.add_argument("--model_name", type=str, help="Pretrained model name")

    # Training parameters
    parser.add_argument(
        "--num_epochs_frozen", type=int, help="Epochs with frozen base model"
    )
    parser.add_argument(
        "--num_epochs_unfrozen", type=int, help="Epochs with unfrozen model"
    )
    parser.add_argument("--learning_rate", type=float, help="Learning rate")

    # System parameters
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output_dir", type=str, help="Output directory")

    # Prediction parameters
    parser.add_argument("--load_dir", type=str, help="Directory to load model from")
    parser.add_argument("--text", type=str, help="Single text to predict")
    parser.add_argument("--text_file", type=str, help="File with texts to predict")

    # Evaluation parameters
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )

    return parser
