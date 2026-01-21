#!/usr/bin/env python3
"""
Fixed version of predict command that works with local model paths.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def cmd_predict_simple(args):
    """Simple predict command that works."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        logger = logging.getLogger(__name__)

        # Validate model path
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            sys.exit(1)

        logger.info(f"Loading model from: {model_path}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True
        )

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Get texts to predict
        texts = []
        if args.text:
            texts.append(args.text)
        elif args.text_file:
            logger.info(f"Reading texts from {args.text_file}")
            with open(args.text_file, "r") as f:
                texts.extend([line.strip() for line in f if line.strip()])
        else:
            logger.error("No texts provided. Use --text or --text_file")
            sys.exit(1)

        # Make predictions
        logger.info(f"Making predictions for {len(texts)} texts...")
        model.eval()

        results = []
        for text in texts:
            # Tokenize input
            inputs = tokenizer(
                text, truncation=True, padding=True, max_length=256, return_tensors="pt"
            ).to(device)

            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred_label = torch.argmax(probs, dim=1).item()
                pos_prob = probs[0][1].item()

            # Format result
            label_text = "POSITIVE" if pred_label == 1 else "NEGATIVE"
            result = {
                "text": text,
                "predicted_label": label_text,
                "predicted_label_id": pred_label,
                "positive_probability": round(pos_prob, 4),
                "negative_probability": round(1 - pos_prob, 4),
                "confidence": round(max(pos_prob, 1 - pos_prob), 4),
            }
            results.append(result)

            # Show result
            logger.info(f'Review: "{text[:60]}{"..." if len(text) > 60 else ""}"')
            logger.info(
                f"Prediction: {label_text} (confidence: {result['confidence']:.4f})"
            )

        # Output results
        if args.output_file:
            import json

            with open(args.output_file, "w") as f:
                json.dump({"predictions": results}, f, indent=2)
            logger.info(f"Predictions saved to: {args.output_file}")
        else:
            print("\n=== PREDICTION RESULTS ===")
            for result in results:
                print(f"Text: {result['text']}")
                print(f"Prediction: {result['predicted_label']}")
                print(f"Positive Probability: {result['positive_probability']}")
                print(f"Confidence: {result['confidence']}")
                print("-" * 40)

        return True

    except ImportError as e:
        logger.error(f"Missing required packages: {e}")
        logger.error("Please install: pip install torch transformers")
        return False
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for simple predict."""
    parser = argparse.ArgumentParser(description="Simple Sentiment Prediction")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--text", type=str, help="Single text to predict")
    parser.add_argument(
        "--text_file", type=str, help="File with texts to predict (one per line)"
    )
    parser.add_argument(
        "--output_file", type=str, help="Output file for predictions (JSON format)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Run prediction
    success = cmd_predict_simple(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
