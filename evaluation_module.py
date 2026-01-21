"""
Evaluation module for sentiment analysis pipeline.

This module provides comprehensive evaluation capabilities including
metrics computation, result visualization, and analysis tools.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class SentimentEvaluator:
    """Comprehensive evaluator for sentiment analysis models."""

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        device: torch.device,
        config: Any = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: Trained sentiment analysis model
            tokenizer: Tokenizer for the model
            device: Device for computation
            config: Configuration object
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        # Label mappings
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics for the trainer.

        Args:
            eval_pred: Tuple of (predictions, labels) from trainer

        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)

        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=1, zero_division=0
        )

        # Additional metrics
        # ROC-AUC (need probabilities)
        probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
        try:
            roc_auc = roc_auc_score(labels, probs)
        except ValueError:
            roc_auc = 0.0

        # Average Precision (PR-AUC)
        try:
            pr_auc = average_precision_score(labels, probs)
        except ValueError:
            pr_auc = 0.0

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        }

        return metrics

    def evaluate_dataset(
        self, dataset: Dataset, split_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation on a dataset.

        Args:
            dataset: Dataset to evaluate
            split_name: Name of the dataset split

        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        logger.info(f"Evaluating model on {split_name} set...")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for example in dataset:
                # Prepare inputs
                input_ids = example["input_ids"].unsqueeze(0).to(self.device)
                attention_mask = example["attention_mask"].unsqueeze(0).to(self.device)
                labels = example["label"].item()

                # Get predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred_label = torch.argmax(probs, dim=1).item()

                # Collect results
                all_predictions.append(pred_label)
                all_labels.append(labels)
                all_probs.append(probs[0][1].item())  # Probability of positive class

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probs = np.array(all_probs)

        # Compute comprehensive metrics
        metrics = self._compute_comprehensive_metrics(labels, predictions, probs)

        # Generate detailed analysis
        analysis = self._generate_detailed_analysis(labels, predictions, probs, dataset)

        results = {
            "split_name": split_name,
            "num_samples": len(dataset),
            "metrics": metrics,
            "analysis": analysis,
            "predictions": {
                "predicted_labels": predictions.tolist(),
                "true_labels": labels.tolist(),
                "positive_probabilities": probs.tolist(),
            },
        }

        logger.info(f"Evaluation completed for {split_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        return results

    def _compute_comprehensive_metrics(
        self, labels: np.ndarray, predictions: np.ndarray, probs: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""

        # Basic classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=1, zero_division=0
        )

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(labels, probs)
        except ValueError:
            roc_auc = 0.0

        # PR-AUC
        try:
            pr_auc = average_precision_score(labels, probs)
        except ValueError:
            pr_auc = 0.0

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(
                labels, predictions, average=None, zero_division=0
            )
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            # Per-class metrics
            "precision_negative": float(precision_per_class[0])
            if len(precision_per_class) > 0
            else 0.0,
            "recall_negative": float(recall_per_class[0])
            if len(recall_per_class) > 0
            else 0.0,
            "f1_negative": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            "support_negative": int(support_per_class[0])
            if len(support_per_class) > 0
            else 0,
            "precision_positive": float(precision_per_class[1])
            if len(precision_per_class) > 1
            else 0.0,
            "recall_positive": float(recall_per_class[1])
            if len(recall_per_class) > 1
            else 0.0,
            "f1_positive": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            "support_positive": int(support_per_class[1])
            if len(support_per_class) > 1
            else 0,
        }

        return metrics

    def _generate_detailed_analysis(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probs: np.ndarray,
        dataset: Dataset,
    ) -> Dict[str, Any]:
        """Generate detailed analysis of predictions."""

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        # Find misclassified examples
        misclassified_indices = np.where(labels != predictions)[0]
        misclassified_examples = []

        # Limit the number of misclassified examples to save
        max_misclassified = min(10, len(misclassified_indices))
        for idx in misclassified_indices[:max_misclassified]:
            example = {
                "index": int(idx),
                "true_label": int(labels[idx]),
                "predicted_label": int(predictions[idx]),
                "positive_probability": float(probs[idx]),
                "confidence": float(max(probs[idx], 1 - probs[idx])),
            }

            # Add text if available
            if hasattr(dataset[idx], "text"):
                example["text"] = dataset[idx]["text"]

            misclassified_examples.append(example)

        # Confidence analysis
        confidence_correct = []
        confidence_incorrect = []

        for i in range(len(labels)):
            confidence = max(probs[i], 1 - probs[i])
            if labels[i] == predictions[i]:
                confidence_correct.append(confidence)
            else:
                confidence_incorrect.append(confidence)

        # Threshold analysis
        thresholds = np.arange(0.1, 0.9, 0.1)
        threshold_metrics = []

        for threshold in thresholds:
            threshold_preds = (probs >= threshold).astype(int)
            if len(np.unique(threshold_preds)) > 1:  # Avoid division by zero
                tp = np.sum((threshold_preds == 1) & (labels == 1))
                fp = np.sum((threshold_preds == 1) & (labels == 0))
                tn = np.sum((threshold_preds == 0) & (labels == 0))
                fn = np.sum((threshold_preds == 0) & (labels == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                threshold_metrics.append(
                    {
                        "threshold": float(threshold),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "true_positives": int(tp),
                        "false_positives": int(fp),
                        "true_negatives": int(tn),
                        "false_negatives": int(fn),
                    }
                )

        analysis = {
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                labels,
                predictions,
                target_names=["NEGATIVE", "POSITIVE"],
                output_dict=True,
                zero_division=0,
            ),
            "misclassified_examples": misclassified_examples,
            "num_misclassified": len(misclassified_indices),
            "misclassification_rate": float(len(misclassified_indices) / len(labels)),
            "confidence_stats": {
                "avg_confidence_correct": float(np.mean(confidence_correct))
                if confidence_correct
                else 0.0,
                "avg_confidence_incorrect": float(np.mean(confidence_incorrect))
                if confidence_incorrect
                else 0.0,
                "num_correct": len(confidence_correct),
                "num_incorrect": len(confidence_incorrect),
            },
            "threshold_analysis": threshold_metrics,
        }

        return analysis

    def predict_texts(
        self, texts: List[str], batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on a list of texts.

        Args:
            texts: List of texts to predict
            batch_size: Batch size for processing

        Returns:
            List[Dict[str, Any]]: Prediction results
        """
        self.model.eval()
        results = []

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred_labels = torch.argmax(probs, dim=1)

                # Process each result
                for j, text in enumerate(batch_texts):
                    pred_label = pred_labels[j].item()
                    pos_prob = probs[j][1].item()

                    result = {
                        "text": text,
                        "predicted_label": self.id2label[pred_label],
                        "predicted_label_id": pred_label,
                        "positive_probability": round(pos_prob, 4),
                        "negative_probability": round(1 - pos_prob, 4),
                        "confidence": round(max(pos_prob, 1 - pos_prob), 4),
                    }
                    results.append(result)

        return results

    def save_evaluation_results(
        self, results: Dict[str, Any], output_path: Path
    ) -> None:
        """
        Save evaluation results to file.

        Args:
            results: Evaluation results to save
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Evaluation results saved to: {output_path}")

    def plot_confusion_matrix(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Plot and save confusion matrix.

        Args:
            results: Evaluation results containing confusion matrix
            output_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = np.array(results["analysis"]["confusion_matrix"])

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["NEGATIVE", "POSITIVE"],
                yticklabels=["NEGATIVE", "POSITIVE"],
            )
            plt.title(f"Confusion Matrix - {results['split_name']} Set")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            # Add metrics to the plot
            metrics = results["metrics"]
            plt.figtext(
                0.02,
                0.02,
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1']:.4f}",
                fontsize=10,
                ha="left",
            )

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Confusion matrix saved to: {output_path}")

        except ImportError as e:
            logger.warning(f"Matplotlib/Seaborn not available for plotting: {e}")
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")

    def plot_precision_recall_curve(
        self, results: Dict[str, Any], output_path: Path
    ) -> None:
        """
        Plot and save precision-recall curve.

        Args:
            results: Evaluation results
            output_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt

            labels = np.array(results["predictions"]["true_labels"])
            probs = np.array(results["predictions"]["positive_probabilities"])

            precision, recall, thresholds = precision_recall_curve(labels, probs)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, "b-", linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {results['split_name']} Set")
            plt.grid(True, alpha=0.3)

            # Add AUC score
            auc_score = results["metrics"]["pr_auc"]
            plt.figtext(0.5, 0.02, f"PR-AUC: {auc_score:.4f}", fontsize=12, ha="center")

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Precision-recall curve saved to: {output_path}")

        except ImportError as e:
            logger.warning(f"Matplotlib not available for plotting: {e}")
        except Exception as e:
            logger.error(f"Failed to plot precision-recall curve: {e}")


def create_evaluation_report(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create a comprehensive HTML evaluation report.

    Args:
        results: Evaluation results
        output_dir: Directory to save the report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate HTML report
    html_content = _generate_html_report(results)

    report_path = output_dir / "evaluation_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Evaluation report saved to: {report_path}")


def _generate_html_report(results: Dict[str, Any]) -> str:
    """Generate HTML content for evaluation report."""

    metrics = results["metrics"]
    analysis = results["analysis"]

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .confusion-matrix {{ text-align: center; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Sentiment Analysis Evaluation Report</h1>
        <h2>Dataset: {results["split_name"]} ({results["num_samples"]} samples)</h2>
        
        <h3>Overall Metrics</h3>
        <div class="metric">
            <div class="metric-value">{metrics["accuracy"]:.4f}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics["f1"]:.4f}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics["precision"]:.4f}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics["recall"]:.4f}</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric">
            <div class="metric-value">{metrics["roc_auc"]:.4f}</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        
        <h3>Confusion Matrix</h3>
        <div class="confusion-matrix">
            <table>
                <tr>
                    <th></th>
                    <th>Predicted Negative</th>
                    <th>Predicted Positive</th>
                </tr>
                <tr>
                    <th>Actual Negative</th>
                    <td>{analysis["confusion_matrix"][0][0]}</td>
                    <td>{analysis["confusion_matrix"][0][1]}</td>
                </tr>
                <tr>
                    <th>Actual Positive</th>
                    <td>{analysis["confusion_matrix"][1][0]}</td>
                    <td>{analysis["confusion_matrix"][1][1]}</td>
                </tr>
            </table>
        </div>
        
        <h3>Misclassification Analysis</h3>
        <p><strong>Total Misclassified:</strong> {analysis["num_misclassified"]} ({analysis["misclassification_rate"]:.2%})</p>
        <p><strong>Average Confidence (Correct):</strong> {analysis["confidence_stats"]["avg_confidence_correct"]:.4f}</p>
        <p><strong>Average Confidence (Incorrect):</strong> {analysis["confidence_stats"]["avg_confidence_incorrect"]:.4f}</p>
    </body>
    </html>
    """

    return html
