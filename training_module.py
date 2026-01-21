"""
Training module for sentiment analysis pipeline.

This module provides comprehensive training capabilities including
two-phase training, checkpointing, early stopping, and experiment tracking.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import torch
from transformers import Trainer, EarlyStoppingCallback
from datasets import DatasetDict, Dataset

try:
    from .config import SentimentConfig
    from .data_module import DataManager
    from .model_module import ModelManager
    from .evaluation_module import SentimentEvaluator, create_evaluation_report
except ImportError:
    from config import SentimentConfig
    from data_module import DataManager
    from model_module import ModelManager
    from evaluation_module import SentimentEvaluator, create_evaluation_report

logger = logging.getLogger(__name__)


class SentimentTrainer:
    """Comprehensive trainer for sentiment analysis models."""

    def __init__(self, config: SentimentConfig):
        """
        Initialize the trainer with configuration.

        Args:
            config: SentimentConfig containing all training settings
        """
        self.config = config
        self.data_config = config.data
        self.model_config = config.model
        self.training_config = config.training
        self.system_config = config.system

        # Initialize components
        self.data_manager = DataManager(config)
        self.model_manager = ModelManager(config)
        self.evaluator: Optional[SentimentEvaluator] = None

        # Training state
        self.output_dir = Path(self.system_config.output_dir)
        self.run_id = self._generate_run_id()
        self.run_dir = self.output_dir / self.run_id

        # Setup directories
        self._setup_directories()

        # Setup logging
        self.logger = self._setup_logging()

        # Data and model
        self.dataset: Optional[DatasetDict] = None
        self.tokenized_dataset: Optional[DatasetDict] = None
        self.trainer: Optional[Trainer] = None

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _setup_directories(self) -> None:
        """Setup output directories."""
        self.logs_dir = self.run_dir / "logs"
        self.model_dir = self.run_dir / "model"
        self.reports_dir = self.run_dir / "reports"
        self.checkpoints_dir = self.run_dir / "checkpoints"

        for directory in [
            self.logs_dir,
            self.model_dir,
            self.reports_dir,
            self.checkpoints_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        import sys

        # Create logger
        logger = logging.getLogger(f"sentiment_trainer_{self.run_id}")
        logger.setLevel(getattr(logging, self.system_config.log_level.upper()))
        logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        # File handler
        log_file = self.logs_dir / f"run_{self.run_id}.log"
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def setup_training(self) -> None:
        """
        Setup all components for training.

        Raises:
            RuntimeError: If setup fails at any step
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("SENTIMENT ANALYSIS TRAINING SETUP")
            self.logger.info("=" * 80)

            # Log configuration
            self._log_configuration()

            # Load and prepare data
            self.logger.info("Loading and preparing data...")
            self.dataset = self.data_manager.load_dataset()

            # Initialize tokenizer
            self.logger.info("Initializing tokenizer...")
            self.tokenizer = self.data_manager.initialize_tokenizer(
                self.model_config.model_name
            )
            self.model_manager.tokenizer = self.tokenizer

            # Tokenize dataset
            self.logger.info("Tokenizing dataset...")
            self.tokenized_dataset = self.data_manager.tokenize_dataset(self.dataset)

            # Validate dataset
            self.logger.info("Validating dataset...")
            self.data_manager.validate_dataset(self.tokenized_dataset)

            # Load model
            self.logger.info("Loading model...")
            self.model = self.model_manager.load_model()

            # Setup evaluator
            self.evaluator = SentimentEvaluator(
                self.model, self.tokenizer, self.model_manager.device, self.config
            )

            # Save dataset information
            dataset_info_path = self.reports_dir / f"dataset_info_{self.run_id}.json"
            self.data_manager.save_dataset_info(
                self.tokenized_dataset, dataset_info_path
            )

            self.logger.info("Training setup completed successfully")

        except Exception as e:
            self.logger.error(f"Training setup failed: {str(e)}")
            raise RuntimeError(f"Training setup failed: {str(e)}")

    def _log_configuration(self) -> None:
        """Log the complete training configuration."""
        self.logger.info("TRAINING CONFIGURATION:")
        self.logger.info(f"  Run ID: {self.run_id}")
        self.logger.info(f"  Device: {self.model_manager.device}")
        self.logger.info(f"  Model: {self.model_config.model_name}")
        self.logger.info(f"  Dataset: {self.data_config.dataset_name}")
        self.logger.info(f"  Train samples: {self.data_config.train_samples}")
        self.logger.info(f"  Eval samples: {self.data_config.eval_samples}")
        self.logger.info(f"  Validation split: {self.data_config.val_split}")
        self.logger.info(f"  Batch size: {self.data_config.batch_size}")
        self.logger.info(f"  Max length: {self.data_config.max_length}")
        self.logger.info(f"  Learning rate: {self.training_config.learning_rate}")
        self.logger.info(f"  Epochs (frozen): {self.training_config.num_epochs_frozen}")
        self.logger.info(
            f"  Epochs (unfrozen): {self.training_config.num_epochs_unfrozen}"
        )
        self.logger.info(
            f"  Early stopping patience: {self.training_config.early_stopping_patience}"
        )
        self.logger.info(f"  FP16 enabled: {self.training_config.fp16}")
        self.logger.info(f"  Seed: {self.system_config.seed}")
        self.logger.info("=" * 80)

    def train(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.

        Returns:
            Dict[str, Any]: Training results and metrics

        Raises:
            RuntimeError: If training fails
        """
        if self.dataset is None or self.tokenized_dataset is None:
            self.setup_training()

        try:
            training_results = {
                "run_id": self.run_id,
                "phase1_results": None,
                "phase2_results": None,
                "final_evaluation": None,
                "custom_predictions": None,
            }

            # Phase 1: Train classification head (frozen base model)
            self.logger.info("=" * 80)
            self.logger.info(
                "PHASE 1: Training Classification Head (Base Model Frozen)"
            )
            self.logger.info("=" * 80)

            phase1_results = self._train_phase("phase1", frozen=True)
            training_results["phase1_results"] = phase1_results

            # Phase 2: Fine-tune entire model
            self.logger.info("=" * 80)
            self.logger.info("PHASE 2: Fine-tuning Entire Model")
            self.logger.info("=" * 80)

            phase2_results = self._train_phase("phase2", frozen=False)
            training_results["phase2_results"] = phase2_results

            # Final evaluation
            self.logger.info("=" * 80)
            self.logger.info("FINAL EVALUATION")
            self.logger.info("=" * 80)

            final_evaluation = self.evaluator.evaluate_dataset(
                self.tokenized_dataset["test"], "test"
            )
            training_results["final_evaluation"] = final_evaluation

            # Test with custom reviews
            self.logger.info("Testing with custom reviews...")
            custom_predictions = self._test_custom_reviews()
            training_results["custom_predictions"] = custom_predictions

            # Save final model
            self.logger.info("Saving final model...")
            self.model_manager.save_model(self.model_dir)

            # Save training results
            self._save_training_results(training_results)

            # Generate comprehensive report
            self._generate_training_report(training_results)

            self.logger.info("=" * 80)
            self.logger.info("TRAINING COMPLETED SUCCESSFULLY")
            self.logger.info(f"Results saved to: {self.run_dir}")
            self.logger.info("=" * 80)

            return training_results

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")

    def _train_phase(self, phase_name: str, frozen: bool = True) -> Dict[str, Any]:
        """
        Execute a single training phase.

        Args:
            phase_name: Name of the training phase
            frozen: Whether to freeze base model parameters

        Returns:
            Dict[str, Any]: Phase training results
        """
        # Freeze/unfreeze model as needed
        if frozen:
            self.model_manager.freeze_base_model()
        else:
            self.model_manager.unfreeze_model()

        # Setup training arguments
        training_args = self.model_manager.setup_training_arguments(
            phase_name, self.checkpoints_dir
        )

        # Setup trainer
        self.trainer = self.model_manager.setup_trainer(
            training_args,
            self.tokenized_dataset["train"],
            self.tokenized_dataset["validation"],
            self.data_manager.get_data_collator(),
            self.evaluator.compute_metrics,
            self.tokenizer,
        )

        # Train the model
        self.logger.info(f"Starting {phase_name} training...")
        train_result = self.trainer.train()

        # Evaluate on validation set
        val_metrics = self.trainer.evaluate(self.tokenized_dataset["validation"])

        # Log phase results
        self.logger.info(f"{phase_name} training completed:")
        self.logger.info(f"  Final training loss: {train_result.training_loss:.4f}")
        self.logger.info(f"  Validation metrics: {val_metrics}")

        # Save training state
        self.model_manager.save_training_state(self.checkpoints_dir / phase_name)

        return {
            "phase_name": phase_name,
            "training_result": train_result,
            "validation_metrics": val_metrics,
            "model_frozen": frozen,
        }

    def _test_custom_reviews(self) -> List[Dict[str, Any]]:
        """Test the model with custom review examples."""
        custom_reviews = [
            "Incredible film. Just great!",
            "What a waste of time! I'd like to have back those 2 hours of my life.",
            "This movie was absolutely fantastic! Best thing I've seen all year.",
            "Boring and predictable. Not worth watching.",
            "A masterpiece of cinema. Brilliant performances all around.",
            "Terrible acting and awful plot. Complete disaster.",
            "It was okay, nothing special but not terrible either.",
            "I laughed, I cried, it was better than Cats!",
        ]

        predictions = self.evaluator.predict_texts(custom_reviews)

        # Log predictions
        self.logger.info("Custom Review Predictions:")
        self.logger.info("=" * 60)
        for pred in predictions:
            self.logger.info(f'Review: "{pred["text"][:60]}..."')
            self.logger.info(
                f"Prediction: {pred['predicted_label']} (confidence: {pred['confidence']:.4f})"
            )
            self.logger.info("-" * 40)
        self.logger.info("=" * 60)

        # Save predictions
        predictions_path = self.reports_dir / f"custom_predictions_{self.run_id}.json"
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, default=str)

        return predictions

    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save complete training results."""
        results_path = self.reports_dir / f"training_results_{self.run_id}.json"

        # Prepare results for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        self.logger.info(f"Training results saved to: {results_path}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, "__dict__"):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _generate_training_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive training report."""
        try:
            # Generate evaluation plots
            if results["final_evaluation"]:
                cm_plot_path = self.reports_dir / f"confusion_matrix_{self.run_id}.png"
                pr_plot_path = (
                    self.reports_dir / f"precision_recall_curve_{self.run_id}.png"
                )

                self.evaluator.plot_confusion_matrix(
                    results["final_evaluation"], cm_plot_path
                )
                self.evaluator.plot_precision_recall_curve(
                    results["final_evaluation"], pr_plot_path
                )

                # Create evaluation report
                create_evaluation_report(results["final_evaluation"], self.reports_dir)

            # Generate HTML training report
            html_report = self._generate_training_html(results)
            report_path = self.run_dir / "training_report.html"

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_report)

            self.logger.info(f"Training report generated: {report_path}")

        except Exception as e:
            self.logger.warning(f"Failed to generate training report: {str(e)}")

    def _generate_training_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML training report."""

        phase1_metrics = results.get("phase1_results", {}).get("validation_metrics", {})
        phase2_metrics = results.get("phase2_results", {}).get("validation_metrics", {})
        final_metrics = results.get("final_evaluation", {}).get("metrics", {})

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sentiment Analysis Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
                .phase {{ background-color: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Sentiment Analysis Training Report</h1>
            <p><strong>Run ID:</strong> {self.run_id}</p>
            <p><strong>Model:</strong> {self.model_config.model_name}</p>
            <p><strong>Timestamp:</strong> {datetime.now().isoformat()}</p>
            
            <div class="section">
                <h2>Phase 1 Results (Classification Head Training)</h2>
                <div class="phase">
                    <h3>Validation Metrics</h3>
                    <div class="metric">Accuracy: {phase1_metrics.get("eval_accuracy", 0):.4f}</div>
                    <div class="metric">F1: {phase1_metrics.get("eval_f1", 0):.4f}</div>
                    <div class="metric">Precision: {phase1_metrics.get("eval_precision", 0):.4f}</div>
                    <div class="metric">Recall: {phase1_metrics.get("eval_recall", 0):.4f}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Phase 2 Results (Full Model Fine-tuning)</h2>
                <div class="phase">
                    <h3>Validation Metrics</h3>
                    <div class="metric">Accuracy: {phase2_metrics.get("eval_accuracy", 0):.4f}</div>
                    <div class="metric">F1: {phase2_metrics.get("eval_f1", 0):.4f}</div>
                    <div class="metric">Precision: {phase2_metrics.get("eval_precision", 0):.4f}</div>
                    <div class="metric">Recall: {phase2_metrics.get("eval_recall", 0):.4f}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Final Test Evaluation</h2>
                <div class="metric">Accuracy: {final_metrics.get("accuracy", 0):.4f}</div>
                <div class="metric">F1: {final_metrics.get("f1", 0):.4f}</div>
                <div class="metric">Precision: {final_metrics.get("precision", 0):.4f}</div>
                <div class="metric">Recall: {final_metrics.get("recall", 0):.4f}</div>
                <div class="metric">ROC-AUC: {final_metrics.get("roc_auc", 0):.4f}</div>
            </div>
            
            <div class="section">
                <h2>Custom Review Predictions</h2>
                <p>Model tested with {len(results.get("custom_predictions", []))} custom reviews.</p>
            </div>
        </body>
        </html>
        """

        return html

    def resume_training(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (if None, uses latest)

        Returns:
            Dict[str, Any]: Training results
        """
        if checkpoint_path:
            self.model_manager.load_model_from_checkpoint(checkpoint_path)
        else:
            # Find latest checkpoint
            checkpoints = list(self.checkpoints_dir.glob("*/checkpoint-*"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found for resuming")

            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            self.model_manager.load_model_from_checkpoint(latest_checkpoint)

        self.logger.info(f"Resuming training from checkpoint")
        return self.train()
