"""
Model management module for sentiment analysis pipeline.

This module provides model loading and management capabilities for sentiment analysis models.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from .config import SentimentConfig, ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, training setup, and operations."""

    def __init__(self, config: SentimentConfig):
        """
        Initialize ModelManager with configuration.

        Args:
            config: SentimentConfig containing model and training settings
        """
        self.config = config
        self.model_config = config.model
        self.training_config = config.training
        self.system_config = config.system

        # Training state for resuming
        self.checkpoint_dir: Optional[Path] = None
        self.current_epoch: int = 0
        self.best_metric: float = float("-inf")

        # Device setup
        self.device = self._setup_device()

        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def _setup_device(self) -> torch.device:
        """
        Setup and configure the compute device.

        Returns:
            torch.device: Configured device for computation
        """
        if self.system_config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                logger.info(
                    f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
                )
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        elif self.system_config.device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")

        return device

    def load_model(
        self, model_name: Optional[str] = None, checkpoint_path: Optional[str] = None
    ) -> AutoModelForSequenceClassification:
        """
        Load the sentiment analysis model.

        Args:
            model_name: Name of pretrained model (overrides config)
            checkpoint_path: Path to checkpoint for resuming training

        Returns:
            AutoModelForSequenceClassification: Loaded model

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            model_name = model_name or self.model_config.model_name

            if checkpoint_path:
                logger.info(f"Loading model from checkpoint: {checkpoint_path}")

                # Validate checkpoint path
                self._validate_model_path(checkpoint_path)

                # Load model from local path
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint_path, local_files_only=True
                )

                # Automatically load tokenizer from the same local path
                if not self.tokenizer:
                    logger.info(f"Loading tokenizer from local path: {checkpoint_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        checkpoint_path, local_files_only=True
                    )
            else:
                logger.info(f"Loading pretrained model: {model_name}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=self.model_config.num_labels,
                    id2label=self.model_config.id2label,
                    label2id=self.model_config.label2id,
                    torch_dtype=torch.float16
                    if self.training_config.fp16
                    else torch.float32,
                    cache_dir=None,  # Will be set by config if needed
                )

            # Move model to device
            self.model.to(self.device)

            # Configure dropout
            self._configure_dropout()

            # Log model information
            self._log_model_info()

            return self.model

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def _validate_model_path(self, path: str) -> None:
        """
        Validate that the model path contains required files.

        Args:
            path: Path to validate

        Raises:
            FileNotFoundError: If required files are missing
        """
        from pathlib import Path

        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")

        # Check for required model files
        required_files = ["config.json"]
        optional_files = [
            "pytorch_model.bin",
            "model.safetensors",
        ]  # Either one is fine

        missing_required = [f for f in required_files if not (model_path / f).exists()]
        has_model_file = any((model_path / f).exists() for f in optional_files)

        if missing_required:
            raise FileNotFoundError(
                f"Model path missing required files: {missing_required}. "
                f"Found files: {list(model_path.glob('*'))}"
            )

        if not has_model_file:
            raise FileNotFoundError(
                f"Model path missing model file. Expected one of: {optional_files}. "
                f"Found files: {list(model_path.glob('*'))}"
            )

        logger.debug(f"Model path validation passed for: {path}")

    def _configure_dropout(self) -> None:
        """Configure dropout rates in the model."""
        if hasattr(self.model, "dropout"):
            if isinstance(self.model.dropout, torch.nn.Dropout):
                self.model.dropout.p = self.model_config.dropout_rate
            logger.info(f"Set model dropout rate to: {self.model_config.dropout_rate}")

        # Configure dropout in classifier if present
        if hasattr(self.model, "classifier"):
            if hasattr(self.model.classifier, "dropout"):
                if isinstance(self.model.classifier.dropout, torch.nn.Dropout):
                    self.model.classifier.dropout.p = self.model_config.dropout_rate
                    logger.info(
                        f"Set classifier dropout rate to: {self.model_config.dropout_rate}"
                    )

    def _log_model_info(self) -> None:
        """Log comprehensive model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logger.info("Model Information:")
        logger.info(f"  Model: {self.model_config.model_name}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  FP16 enabled: {self.training_config.fp16}")

        # Log model architecture summary
        if hasattr(self.model, "config"):
            config = self.model.config
            logger.info(f"  Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
            logger.info(f"  Num layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
            logger.info(
                f"  Num attention heads: {getattr(config, 'num_attention_heads', 'N/A')}"
            )
            logger.info(f"  Vocabulary size: {getattr(config, 'vocab_size', 'N/A')}")
            logger.info(
                f"  Max position embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}"
            )

    def setup_training_arguments(
        self, phase: str, output_dir: Path, learning_rate: Optional[float] = None
    ) -> TrainingArguments:
        """
        Setup training arguments for the specified phase.

        Args:
            phase: Training phase ("phase1" or "phase2")
            output_dir: Directory to save model outputs
            learning_rate: Override learning rate

        Returns:
            TrainingArguments: Configured training arguments
        """
        # Determine phase-specific settings
        if phase == "phase1":
            num_epochs = self.training_config.num_epochs_frozen
            lr = learning_rate or self.training_config.learning_rate
        elif phase == "phase2":
            num_epochs = self.training_config.num_epochs_unfrozen
            lr = learning_rate or (self.training_config.learning_rate / 10)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Create output directory
        phase_output_dir = output_dir / phase
        phase_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            # Output and logging
            output_dir=str(phase_output_dir),
            logging_dir=str(phase_output_dir / "logs"),
            # Training parameters
            num_train_epochs=num_epochs,
            learning_rate=lr,
            weight_decay=self.training_config.weight_decay,
            warmup_steps=self.training_config.warmup_steps,
            gradient_clip_val=self.training_config.gradient_clip_val,
            # Batch processing
            per_device_train_batch_size=self.config.data.batch_size,
            per_device_eval_batch_size=self.config.data.batch_size,
            dataloader_num_workers=getattr(
                self.training_config, "dataloader_num_workers", 0
            ),
            gradient_accumulation_steps=self.training_config.accumulation_steps,
            # Evaluation and saving
            eval_strategy=self.training_config.eval_strategy,
            save_strategy=self.training_config.save_strategy,
            logging_steps=self.training_config.logging_steps,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            # Optimization
            fp16=self.training_config.fp16 if torch.cuda.is_available() else False,
            seed=self.system_config.seed,
            # Reporting
            report_to="none",
        )

        logger.info(f"Training arguments for {phase}:")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Output dir: {phase_output_dir}")

        return training_args

    def save_model(self, output_dir: Path) -> None:
        """
        Save the model and tokenizer.

        Args:
            output_dir: Directory to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(output_dir)
        logger.info(f"Model saved to: {output_dir}")

        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to: {output_dir}")

    def load_model_from_checkpoint(
        self, checkpoint_path: Path
    ) -> AutoModelForSequenceClassification:
        """
        Load model from checkpoint for resuming training.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            AutoModelForSequenceClassification: Loaded model
        """
        self.checkpoint_dir = Path(checkpoint_path).parent
        return self.load_model(checkpoint_path=str(checkpoint_path))

    def save_training_state(self, output_dir: Path) -> None:
        """
        Save training state for potential resumption.

        Args:
            output_dir: Directory to save training state
        """
        import json
        from datetime import datetime

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_state = {
            "current_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "model_config": {
                "model_name": self.model_config.model_name,
                "num_labels": self.model_config.num_labels,
            },
            "timestamp": datetime.now().isoformat(),
        }

        state_file = output_dir / "training_state.json"
        with open(state_file, "w") as f:
            json.dump(training_state, f, indent=2)

        logger.info(f"Training state saved to: {state_file}")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the model.

        Returns:
            Dict[str, Any]: Model summary information
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        summary = {
            "model_name": self.model_config.model_name,
            "num_labels": self.model_config.num_labels,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "fp16_enabled": self.training_config.fp16,
            "model_architecture": str(type(self.model).__name__),
        }

        # Add model configuration if available
        if hasattr(self.model, "config"):
            config = self.model.config
            summary.update(
                {
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                    "num_attention_heads": getattr(config, "num_attention_heads", None),
                    "vocab_size": getattr(config, "vocab_size", None),
                    "max_position_embeddings": getattr(
                        config, "max_position_embeddings", None
                    ),
                }
            )

        return summary
