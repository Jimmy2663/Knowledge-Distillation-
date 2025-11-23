# Knowledge Distillation for Image Classification

A comprehensive PyTorch implementation demonstrating **Knowledge Distillation (KD)** for training efficient student models from larger teacher models. This project implements knowledge distillation using soft targets from a teacher network combined with hard labels (Cross-Entropy loss) to train lightweight student networks for image classification tasks.

## Project Overview

Knowledge Distillation is a model compression technique where a pre-trained large teacher model transfers its knowledge to a smaller, more efficient student model. This approach improves the student model's performance beyond what it could achieve through standard training.

### Key Features

- **Teacher-Student Framework**: Train a large teacher model (VGG16) and then transfer its knowledge to a smaller student model (ResNet18)
- **Hybrid Loss Function**: Combines soft target loss (from teacher) and hard label loss (cross-entropy) using temperature-based scaling
- **Comprehensive Metrics**: Tracks Top-1, Top-3 accuracy, precision, recall, F1-score, and AUC across training, validation, and test sets
- **Multiple Training Modes**:
  - Teacher model training (standard supervised learning)
  - Student model without KD (baseline comparison)
  - Student model with KD (enhanced learning)
- **Detailed Visualization**: Generates loss curves, accuracy plots, and confusion matrices
- **Device-Agnostic**: Supports both CPU and GPU training with automatic device detection

## Project Structure

```
├── experiment.py                # Main entry point - orchestrates full knowledge distillation pipeline
├── engine.py                    # Standard training/evaluation engine
├── KD_engine.py                # Knowledge distillation specific training/evaluation engine
├── model_builder.py            # Neural network architecture definitions
├── data_setup.py               # Data loading and preprocessing utilities
├── utils.py                    # Utility functions for metrics, plotting, and model management
├── prediction_for_one.py       # Script for single image prediction
├── res18_train.py             # Standalone ResNet18 training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## File Descriptions

### `experiment.py` **Main File**

The primary entry point orchestrating the complete knowledge distillation workflow:

- Trains a **VGG16 teacher model** (100 epochs) on the dataset
- Trains a **ResNet18 student model without KD** (30 epochs) as a baseline
- Trains a **ResNet18 student model with KD** (30 epochs) using teacher knowledge
- Saves all models, metrics, and visualizations to organized subdirectories
- Configurable hyperparameters: temperature (T), alpha (ALPHA), learning rates, batch size

### `engine.py`

Standard training and evaluation loop for models trained without knowledge distillation:

- `train_step()`: Single epoch training with comprehensive metrics
- `test_step()`: Model evaluation on validation/test sets
- `train_and_test()`: Complete training loop for multiple epochs
- Supports Top-1/Top-3 accuracy, precision, recall, F1-score, and AUC metrics

### `KD_engine.py`

Knowledge distillation-specific training engine:

- `train_step_knowledge_distillation()`: Implements distillation loss combining:
  - **Soft target loss**: KL divergence between teacher and student (with temperature T)
  - **Hard target loss**: Cross-entropy with ground truth labels
  - **Weighted combination**: `loss = α × CE_loss + (1-α) × KD_loss`
- `test_step()`: Same evaluation as standard engine
- `train_and_test_with_KD()`: Complete KD training loop

### `model_builder.py`

Neural network architecture implementations:

- **VGG16**: Deep convolutional network for teacher model (5 conv blocks + 3 FC layers)
- **ResNet18**: Lightweight residual network for student models (18 layers with residual blocks)
- **BasicBlock**: Building block for ResNet with batch normalization and skip connections

### `data_setup.py`

Data loading utilities:

- `create_dataloaders()`: Creates training, validation, and test DataLoaders
- Supports ImageFolder directory structure for multi-class datasets
- Handles image transforms, batching, and multi-worker data loading

### `utils.py`

Comprehensive utility functions:

- **Metrics**: `calculate_topk_accuracy()`, `calculate_comprehensive_metrics()`
- **Plotting**: `plot_training_curves()`, `plot_top1_top3_comparison()`, `generate_confusion_matrix()`
- **Reporting**: `generate_classification_report()`, `save_training_metrics()`, `save_detailed_metrics()`
- **Model Management**: `save_model()`, `save_model_summary()`
- **Time Formatting**: `format_time()` for readable time display

### `res18_train.py`

Standalone script for training ResNet18 without knowledge distillation (useful for baseline comparison).

### `prediction_for_one.py`

Inference script for making predictions on single images with a trained model.

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional but recommended)

### Installation

1. **Clone or download the project**

2. **Create and activate a virtual environment** (recommended)

   ```bash
   # Using conda
   conda create -n knowledge_distillation python=3.9
   conda activate knowledge_distillation

   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiment

```bash
python experiment.py
```

This will:

1. Train the teacher model (VGG16) for 100 epochs
2. Train student model without KD (ResNet18) for 30 epochs
3. Train student model with KD (ResNet18) for 30 epochs
4. Save all models, metrics, and visualizations

**Output Structure:**

```
Save_dir_T2_A50_pv/
├── teacher/
│   ├── Teacher_model.pth
│   ├── model_summary_teacher.txt
│   ├── training_metrics.txt
│   ├── detailed_metrics.txt
│   ├── results.json
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix_train.png
│   ├── confusion_matrix_val.png
│   ├── confusion_matrix_test.png
│   └── classification_report_*.txt
├── student1/  (without KD)
│   └── [same structure as teacher]
└── student2/  (with KD)
    └── [same structure as teacher]
```

## Configuration

Edit hyperparameters in `experiment.py`:

```python
# Training epochs
TEACHER_EPOCHS = 100      # Teacher training epochs
STUDENT_EPOCHS = 30       # Student training epochs

# Learning rates
TEACHER_LR = 0.0001       # Teacher learning rate
STUDENT_LR = 0.0001       # Student learning rate

# Knowledge Distillation parameters
T = 2                     # Temperature (higher = softer targets, 2-7 typical)
ALPHA = 0.50             # Weight balance: ALPHA × CE + (1-ALPHA) × KD
                         # ALPHA=0.5 means 50% CE loss, 50% KD loss

# Data parameters
BATCH_SIZE = 32          # Batch size for training
```

## Expected Outputs

### Metrics Tracked

For each epoch (training, validation, test):

- **Loss**: Training/validation loss
- **Top-1 Accuracy**: Standard accuracy
- **Top-3 Accuracy**: Sample correctly classified in top-3 predictions
- **Precision**: Micro, Macro, Weighted averages
- **Recall**: Micro, Macro, Weighted averages
- **F1-Score**: Micro, Macro, Weighted averages
- **AUC**: Area under ROC curve (Micro, Macro, Weighted)

### Visualizations Generated

- **Loss curves**: Training vs validation loss
- **Accuracy curves**: Training vs validation accuracy
- **Top-1 vs Top-3 comparison**: Dual accuracy metrics
- **Confusion matrices**: For train, validation, and test sets
- **Classification reports**: Precision/recall/F1 per class

##  Single Image Prediction

To make predictions on a single image:

```bash
python prediction_for_one.py --model_path path/to/model.pth
```

Edit the following in `prediction_for_one.py`:

```python
IMG_PATH = "/path/to/your/image.jpg"
class_names = ["class1", "class2", "class3", "class4"]
```

## Knowledge Distillation Deep Dive

### How It Works

1. **Teacher Model**: Trained first using standard supervised learning to achieve high accuracy
2. **Knowledge Transfer**: Teacher generates soft predictions (probabilities) on training data
3. **Temperature Scaling**: Soft targets are created using temperature T:

   ```
   soft_targets = softmax(teacher_logits / T)
   ```

   - Higher T → softer distributions → more information from non-target classes
   - Lower T → harder distributions → closer to one-hot encoding

4. **Student Training**: Student learns from combined loss:
   ```
   loss = α × CE_loss(student, ground_truth) + (1-α) × KD_loss(student, teacher)
   ```
   - CE loss ensures student learns from hard labels
   - KD loss transfers teacher's knowledge through soft targets

### Benefits

- Student models achieve higher accuracy than training alone
- More efficient inference (smaller model size, faster predictions)
- Improved generalization due to teacher's learned representations
- Reduced overfitting through soft target regularization

## Customization

### Adding New Models

Edit `model_builder.py` to add new architectures:

```python
class YourModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Define architecture

    def forward(self, x):
        # Forward pass
        return x
```

### Changing Loss Function Weights

Modify `ALPHA` in `experiment.py`:

- `ALPHA = 0.5`: Balanced between CE and KD (recommended starting point)
- `ALPHA = 0.7`: More emphasis on ground truth
- `ALPHA = 0.3`: More emphasis on teacher knowledge

### Custom Data Format

Update `data_setup.py` and `experiment.py` to use different data loaders or formats.

## Requirements

See `requirements.txt` for exact versions. Key dependencies:

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy
- scikit-learn
- matplotlib
- seaborn
- torchinfo (for model summaries)

## Notes

- **GPU Usage**: The code automatically detects GPU availability. Set `device = "cuda"` manually in scripts for GPU-only training
- **Data Paths**: Update train_dir, test_dir, real_test_dir in `experiment.py` to point to your dataset
- **Class Count**: Automatically inferred from dataset directories (ImageFolder format)
- **Model Checkpointing**: Best models are saved during training (can be extended in engine files)

## Use Cases

- **Model Compression**: Deploy efficient models in resource-constrained environments
- **Federated Learning**: Transfer knowledge to edge devices
- **Domain Adaptation**: Use teacher trained on source domain for student in target domain
- **Ensemble Learning**: Combine knowledge from multiple teachers
- **Curriculum Learning**: Progressive knowledge transfer from complex to simple models

## References

- Hinton, G. E., Vanhoucke, V., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network" - Original KD paper
- VGG16: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- ResNet18: He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition"
