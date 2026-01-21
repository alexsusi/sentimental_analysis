"""
Enhanced Sentiment Analysis Package.

This package provides comprehensive sentiment analysis capabilities using
pretrained BERT models with advanced features including:

- Modular architecture with separate components for data, model, training, and evaluation
- Comprehensive configuration management with YAML/JSON support
- Advanced training with two-phase approach, early stopping, and checkpointing
- Detailed evaluation with multiple metrics and visualizations
- Command-line interface with subcommands for train, predict, evaluate, and config
- Experiment tracking and comprehensive logging

Example usage:
    # Basic training
    python -m divers train --model_name distilbert-base-uncased --train_samples 1000

    # With configuration file
    python -m divers train --config config.yaml

    # Prediction
    python -m divers predict --model_path ./model --text "This movie was great!"

    # Evaluation
    python -m divers evaluate --model_path ./model --split test
"""

from .config import (
    SentimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    SystemConfig,
    ConfigManager,
    setup_cli_parser,
)

from .data_module import DataManager
from .model_module import ModelManager
from .training_module import SentimentTrainer
from .evaluation_module import SentimentEvaluator, create_evaluation_report

__version__ = "2.0.0"
__author__ = "Enhanced Sentiment Analysis Team"
__email__ = "contact@sentiment-analysis.com"

__all__ = [
    # Configuration
    "SentimentConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "SystemConfig",
    "ConfigManager",
    "setup_cli_parser",
    # Core modules
    "DataManager",
    "ModelManager",
    "SentimentTrainer",
    "SentimentEvaluator",
    # Utilities
    "create_evaluation_report",
]
