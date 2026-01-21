# Enhanced Sentiment Analysis with BERT v2.0

A production-ready, modular sentiment analysis pipeline using pretrained BERT models with advanced features including two-phase training, early stopping, checkpointing, and comprehensive evaluation.

## 🚀 Quick Start

### Installation
```bash
# Install required packages
pip install torch transformers datasets scikit-learn matplotlib seaborn pyyaml numpy

# Navigate to the enhanced directory
cd divers/
```

### Basic Usage
```bash
# Train a model with default settings
python -m divers train

# Make predictions
python -m divers predict --model_path ./sentiment_model_output/latest/model --text "This movie was amazing!"

# Evaluate model performance
python -m divers evaluate --model_path ./sentiment_model_output/latest/model
```

## ✨ Key Features

### 🏗️ **Modular Architecture**
- **Clean Separation**: Dedicated modules for data, model, training, and evaluation
- **Type Safety**: Comprehensive type hints and dataclass validation
- **Extensible Design**: Easy to add new models, datasets, and metrics

### 🛠️ **Advanced Training**
- **Two-Phase Training**: Optimal strategy (frozen base → fine-tuning)
- **Early Stopping**: Prevent overfitting with configurable patience
- **Checkpointing**: Resume training from any checkpoint
- **Mixed Precision**: FP16 training for faster computation

### 📊 **Comprehensive Evaluation**
- **Multiple Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Visualization**: Automatic confusion matrices and precision-recall curves
- **Error Analysis**: Detailed misclassification analysis with confidence scores
- **Export Options**: JSON, CSV, and HTML reports

### 🎛️ **Professional CLI**
- **Subcommands**: `train`, `predict`, `evaluate`, `config`, `resume`
- **Configuration**: YAML/JSON files with CLI override support
- **Interactive Mode**: Real-time predictions
- **Batch Processing**: Handle multiple texts efficiently

## 📁 Project Structure

```
divers/
├── config.py              # Configuration management (dataclasses, YAML/JSON)
├── data_module.py          # Data loading, preprocessing, tokenization
├── model_module.py         # Model management, device setup, training args
├── training_module.py      # Training orchestration, experiment tracking
├── evaluation_module.py     # Evaluation, metrics, visualization
├── __main__.py           # CLI application with subcommands
├── __init__.py           # Package initialization
├── default_config.yaml     # Default configuration template
├── demo_enhancements.py   # Demo of new features
├── test_structure.py      # Structure validation tests
├── README.md             # This documentation
└── sentiment_imdb_V2.py  # Original monolithic implementation
```

## 🔧 Configuration Management

### Default Configuration
```yaml
# data configuration
data:
  dataset_name: "stanfordnlp/imdb"
  train_samples: 2000
  eval_samples: 1000
  val_split: 0.1
  max_length: 256
  batch_size: 16

# model configuration  
model:
  model_name: "distilbert-base-uncased"
  num_labels: 2
  dropout_rate: 0.1

# training configuration
training:
  num_epochs_frozen: 2
  num_epochs_unfrozen: 1
  learning_rate: 2e-5
  early_stopping_patience: 3
  fp16: true

# system configuration
system:
  seed: 42
  device: "auto"  # auto, cpu, cuda
  output_dir: "./sentiment_model_output"
```

### Configuration Commands
```bash
# Create default configuration
python -m divers config create my_config.yaml

# Validate configuration
python -m divers config validate my_config.yaml

# Show configuration
python -m divers config show my_config.yaml
```

## 🖥️ Command Reference

### Train Command
```bash
python -m divers train [OPTIONS]

Basic Usage:
  python -m divers train
  
Advanced Options:
  --config PATH              Configuration file path
  --model_name TEXT          Pretrained model (default: distilbert-base-uncased)
  --train_samples INT        Training samples (default: 2000)
  --eval_samples INT         Evaluation samples (default: 1000)
  --batch_size INT           Batch size (default: 16)
  --learning_rate FLOAT      Learning rate (default: 2e-5)
  --early_stopping_patience INT  Early stopping patience (default: 3)
  --seed INT                 Random seed (default: 42)
  --output_dir TEXT          Output directory
  --fp16                     Use mixed precision training
```

### Predict Command
```bash
python -m divers predict [OPTIONS] --model_path PATH

Single Text:
  python -m divers predict --model_path ./model --text "Great movie!"

Batch from File:
  python -m divers predict --model_path ./model --text_file reviews.txt

Interactive Mode:
  python -m divers predict --model_path ./model --interactive

Output Options:
  --output_file PATH         Save predictions to file
  --output_format FORMAT     Output format: json or csv
  --show_confidence          Show prediction confidence
```

### Evaluate Command
```bash
python -m divers evaluate [OPTIONS] --model_path PATH

Basic Evaluation:
  python -m divers evaluate --model_path ./model --split test

Advanced Options:
  --config_path PATH         Model configuration file
  --dataset_name TEXT        Dataset for evaluation
  --split SPLIT              Split: train/validation/test
  --max_samples INT          Limit number of samples
  --output_dir TEXT          Results directory
  --generate_plots           Create visualizations
  --save_predictions         Save all predictions
```

### Resume Command
```bash
python -m divers resume [OPTIONS] --checkpoint_path PATH

Example:
  python -m divers resume --checkpoint_path ./sentiment_model_output/latest/checkpoints/phase2/checkpoint-500
```

## 📊 Output Structure

Each training run creates a timestamped directory:

```
sentiment_model_output/
└── 2024-01-21_14-30-25/          # Run-specific directory
    ├── logs/
    │   └── run_2024-01-21_14-30-25.log
    ├── model/
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   ├── tokenizer_config.json
    │   └── vocab.txt
    ├── reports/
    │   ├── dataset_info_*.json
    │   ├── training_results_*.json
    │   ├── custom_predictions_*.json
    │   ├── confusion_matrix_*.png
    │   ├── precision_recall_curve_*.png
    │   └── evaluation_report.html
    ├── checkpoints/
    │   ├── phase1/
    │   │   └── checkpoint-*/
    │   └── phase2/
    │       └── checkpoint-*/
    └── training_report.html
```

## 🎯 Use Cases & Examples

### Quick Experiment (5 minutes)
```bash
python -m divers train \
    --train_samples 500 \
    --eval_samples 200 \
    --num_epochs_frozen 1 \
    --num_epochs_unfrozen 0 \
    --batch_size 32
```

### Production Training
```bash
python -m divers train \
    --config production_config.yaml \
    --experiment_name "sentiment_model_v1"
```

### Model Comparison
```bash
for model in "distilbert-base-uncased" "bert-base-uncased" "roberta-base"
do
    python -m divers train \
        --model_name $model \
        --train_samples 1000 \
        --output_dir "./results/$model"
done
```

### Custom Dataset
```yaml
# custom_config.yaml
data:
  dataset_name: "my_custom_dataset"
  text_column: "review_text"
  label_column: "sentiment"
  train_samples: 5000
```

```bash
python -m divers train --config custom_config.yaml
```

## 📈 Performance Tips

### Memory Optimization
```bash
# Reduce batch size for GPU memory limits
python -m divers train --batch_size 8

# Enable gradient accumulation for effective larger batches
python -m divers train --batch_size 4 --config large_batch_config.yaml
```

### Speed Optimization
```bash
# Use mixed precision (default for CUDA)
python -m divers train --fp16

# Increase dataloader workers
python -m divers train --config optimized_config.yaml
# With config:
# training:
#   dataloader_num_workers: 4
```

### Quality Optimization
```bash
# Enable early stopping
python -m divers train --early_stopping_patience 5

# Use smaller learning rate for fine-tuning
python -m divers train --learning_rate 1e-5
```

## 🧪 Testing and Validation

### Run Structure Tests
```bash
python -m divers test_structure
```

### Demo New Features
```bash
python -m divers demo_enhancements
```

### Configuration Validation
```bash
python -m divers config validate my_config.yaml
```

## 🔍 Advanced Features

### Two-Phase Training Strategy
1. **Phase 1**: Train classification head with frozen base model
2. **Phase 2**: Fine-tune entire model with smaller learning rate

This approach:
- ✅ Prevents catastrophic forgetting
- ✅ Faster initial convergence
- ✅ Better final performance
- ✅ More stable training

### Comprehensive Metrics
- **Classification**: Accuracy, Precision, Recall, F1 Score
- **Ranking**: ROC-AUC, PR-AUC
- **Calibration**: Confidence analysis
- **Error Analysis**: Misclassification patterns

### Visualization and Reporting
- **Confusion Matrix**: Visual classification performance
- **Precision-Recall Curve**: Trade-off analysis
- **Training Curves**: Loss and metrics over time
- **HTML Reports**: Interactive evaluation summaries

## 🚨 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Solutions:
# 1. Reduce batch size
python -m divers train --batch_size 8

# 2. Disable FP16
python -m divers train --no-fp16

# 3. Use gradient accumulation
python -m divers train --config gradient_accum_config.yaml
```

#### Dataset Loading Issues
```bash
# Clear cache
rm -rf ~/.cache/huggingface/datasets/

# Use specific cache directory
python -m divers train --config cache_config.yaml
# With config:
# data:
#   cache_dir: "./cache"
```

#### Model Loading Issues
```bash
# Verify model path
ls -la ./sentiment_model_output/latest/model/

# Check configuration
python -m divers config show config.yaml
```

### Debug Mode
```bash
# Enable verbose logging
python -m divers --verbose train --config debug_config.yaml

# With debug config:
# system:
#   log_level: "DEBUG"
#   experiment_tracking: true
```

## 🔄 Migration from Original

### From sentiment_imdb_V2.py

The enhanced version maintains full compatibility while adding improvements:

**Original:**
```bash
python sentiment_imdb_V2.py --mode train --train_samples 1000
```

**Enhanced:**
```bash
python -m divers train --train_samples 1000
```

**Key Differences:**
- ✅ Modular structure (easier to maintain)
- ✅ Configuration files (reproducible experiments)
- ✅ Subcommands (intuitive CLI)
- ✅ Advanced features (early stopping, checkpointing)
- ✅ Comprehensive evaluation
- ✅ Professional documentation

## 🤝 Contributing

### Development Setup
```bash
# Clone and setup
git clone <repository>
cd divers/
pip install -r requirements.txt

# Run tests
python -m divers test_structure

# Run demo
python -m divers demo_enhancements
```

### Adding New Features

**New Models:**
```python
# In model_module.py
def load_custom_model(model_name, config):
    # Implementation
    pass
```

**New Metrics:**
```python
# In evaluation_module.py
def compute_custom_metrics(labels, predictions):
    # Implementation
    pass
```

**New Datasets:**
```python
# In data_module.py
def load_custom_dataset(config):
    # Implementation
    pass
```

## 📄 License

This project is provided for educational and research purposes. See LICENSE file for details.

## 🆘 Support

### Getting Help
1. **Documentation**: Read this README thoroughly
2. **Configuration**: Use `python -m divers config create` for templates
3. **Validation**: Check configurations with `python -m divers config validate`
4. **Logs**: Review training logs in output directory

### Common Questions

**Q: How do I use a different model?**
A: Use `--model_name` or set in config: `model.model_name: "roberta-base"`

**Q: How do I change the dataset?**
A: Modify config: `data.dataset_name: "my_dataset"`

**Q: How do I tune hyperparameters?**
A: Edit config file under `training:` section or use CLI arguments

**Q: How do I resume interrupted training?**
A: Use `python -m divers resume --checkpoint_path ./path/to/checkpoint`

**Q: How do I get more evaluation metrics?**
A: Use `--generate_plots` for visualizations and check HTML reports

---

## 🎉 Summary

This enhanced sentiment analysis system provides:

- ✅ **Professional Code Quality**: Modular, type-safe, well-documented
- ✅ **Advanced Features**: Two-phase training, early stopping, checkpointing
- ✅ **User-Friendly CLI**: Intuitive subcommands and configuration
- ✅ **Comprehensive Evaluation**: Multiple metrics and visualizations
- ✅ **Production Ready**: Experiment tracking, error handling, logging
- ✅ **Extensible Design**: Easy to customize and extend

Perfect for research, prototyping, and production sentiment analysis tasks! 🚀