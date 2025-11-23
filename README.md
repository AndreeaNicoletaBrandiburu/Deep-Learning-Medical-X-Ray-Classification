# Multimodal Veterinary-Inspired Radiograph Classifier

Deep learning pipeline for chest X-ray classification, designed as a proxy for veterinary radiographs. Implemented in PyTorch with CNN & Vision Transformer architectures, GPU acceleration, Grad-CAM explainability, and model compression techniques.

## 🎯 Project Overview

This project demonstrates a complete deep learning pipeline for medical image classification, using public human chest X-ray datasets as a proxy for veterinary radiographs. The approach simulates a realistic medical imaging workflow that could be adapted for veterinary applications.

**Key Motivation**: Due to limited access to labeled veterinary radiology datasets, this project uses the public ["Chest X-Ray Pneumonia"](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) human dataset as a proxy for veterinary thoracic radiographs. This allows for practical implementation and testing of deep learning techniques while maintaining clinical relevance.

## ✨ Features

- **Model Architectures**: ResNet-18 (CNN) and Vision Transformer (ViT)
- **GPU Acceleration**: CUDA-optimized training and inference
- **Mixed Precision Training**: FP16 support for faster training and reduced memory usage
- **Performance Benchmarking**: Throughput, latency, and GPU memory profiling
- **Explainability**: Grad-CAM visualization to identify model attention regions
- **Model Compression**: Pruning and dynamic quantization for deployment optimization
- **Medical Metrics**: ROC-AUC, accuracy, precision, recall, F1-score
- **Clean Code**: Well-structured, production-ready implementation

## 📁 Project Structure

```
multimodal-vet-radiology-dl/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│   └─ 01_exploration.ipynb
├─ src/
│   ├─ data.py              # Data loading utilities
│   ├─ models.py            # Model architectures (ResNet, ViT)
│   ├─ train.py             # Training loop and utilities
│   ├─ eval.py              # Evaluation metrics
│   ├─ explainability.py    # Grad-CAM implementation
│   └─ compress.py          # Model compression (pruning, quantization)
└─ data/                    # Dataset directory (not in repo)
    ├─ train/
    ├─ val/
    └─ test/
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle and organize it as:

```
data/
├─ train/
│   ├─ NORMAL/
│   └─ PNEUMONIA/
├─ val/
│   ├─ NORMAL/
│   └─ PNEUMONIA/
└─ test/
    ├─ NORMAL/
    └─ PNEUMONIA/
```

### 3. Training

#### Train ResNet-18:
```bash
python run_resnet.py
```

#### Train Vision Transformer:
```bash
python run_vit.py
```

### 4. Evaluation

```python
from src.eval import evaluate_model, print_evaluation_report
from src.models import build_resnet18
from src.data import get_dataloaders

# Load model and data
model = build_resnet18(num_classes=2, pretrained=False)
model.load_state_dict(torch.load('best_model.pth'))
train_dl, val_dl, test_dl, class_names = get_dataloaders('data/')

# Evaluate
metrics = evaluate_model(model, test_dl)
print_evaluation_report(metrics)
```

### 5. Explainability (Grad-CAM)

```python
from src.explainability import create_gradcam_for_resnet
from PIL import Image
import torch
from torchvision import transforms

# Load image and model
img = Image.open('path/to/image.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
input_tensor = transform(img).unsqueeze(0)

# Create Grad-CAM
gradcam = create_gradcam_for_resnet(model)
gradcam.visualize(
    input_tensor,
    np.array(img),
    class_names=['NORMAL', 'PNEUMONIA'],
    save_path='gradcam_result.png'
)
```

### 6. Model Compression

```python
from src.compress import compress_model_example

# Apply pruning and quantization
pruned_model, quantized_model, comparison = compress_model_example(
    model,
    pruning_ratio=0.3,
    apply_quantization=True
)
```

### 7. Performance Benchmarking

```python
from src.benchmark import benchmark_model
from src.data import get_dataloaders

# Load model and data
model = build_resnet18(num_classes=2, pretrained=False)
model.load_state_dict(torch.load('best_resnet18_model.pth'))
_, _, test_dl, _ = get_dataloaders('data/')

# Benchmark model performance
metrics = benchmark_model(
    model=model,
    dataloader=test_dl,
    device='cuda',
    use_amp=True,  # Benchmark with mixed precision
    num_runs=100
)
```

Or run the example script:
```bash
python example_benchmark.py
```

## 📊 Results

### ResNet-18 Performance (Test Set)

After training with regularization techniques (dropout, class weights, early stopping), the model achieved the following performance:

| Metric | Value |
|--------|-------|
| **Loss** | 0.3302 |
| **Accuracy** | 92.95% |
| **Precision** | 93.03% |
| **Recall** | 95.90% |
| **F1-Score** | 94.44% |
| **ROC-AUC** | 97.51% |
| **Optimal Threshold** | 0.8898 |

### Confusion Matrix

```
                Predicted
              Normal  Pneumonia
Actual Normal   206       28
      Pneumonia  16      374
```

**Interpretation:**
- **True Positives (TP)**: 374 - Correctly identified pneumonia cases
- **True Negatives (TN)**: 206 - Correctly identified normal cases
- **False Positives (FP)**: 28 - Normal cases misclassified as pneumonia
- **False Negatives (FN)**: 16 - Pneumonia cases missed

### Training Improvements

The model was trained with the following regularization techniques to reduce overfitting:
- **Dropout** (0.5) in final classification layer
- **Class weights** for imbalanced dataset handling
- **Weight decay** (1e-4) for L2 regularization
- **Early stopping** (patience=5) based on validation AUC
- **Enhanced data augmentation** (rotation, translation, color jitter)

**Result**: Reduced train-val accuracy gap from ~43% to ~5%, achieving better generalization.

## 🔬 Technical Details

### Model Architectures

1. **ResNet-18**: Pretrained on ImageNet, fine-tuned for binary classification
2. **Vision Transformer**: ViT-Base-Patch16-224 from `timm` library

### Training

- **Optimizer**: Adam with learning rate 1e-4, weight decay 1e-4
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)
- **Loss**: CrossEntropyLoss with class weights for imbalanced datasets
- **Regularization**: 
  - Dropout (0.5) in final classification layer
  - L2 regularization via weight decay
  - Class weights computed from training data distribution
- **Early stopping**: Based on validation AUC (patience=5)
- **Data augmentation**: Random horizontal flip, rotation (±15°), translation, color jitter
- **Mixed Precision Training (FP16)**: Enabled by default on GPU for faster training and reduced memory usage
  - Uses PyTorch's `autocast` and `GradScaler` for automatic mixed precision
  - Typically provides 1.5-2x speedup on modern GPUs with minimal accuracy loss

### Performance Benchmarking

- **Throughput**: Measures samples/images per second
- **Latency**: Average, P50, P95, P99 percentiles
- **GPU Memory**: Tracks allocated, reserved, and peak memory usage
- **Comparison**: Compare FP32 vs FP16 performance
- **Model Size**: Calculates model size in MB

### Compression

- **Pruning**: L1 unstructured pruning on final fully-connected layer
- **Quantization**: Dynamic INT8 quantization for inference speedup

## 🎓 Connection to Veterinary Background

This project bridges my dual expertise:

1. **Medical Knowledge**: Understanding of radiographic interpretation and anatomical structures (from veterinary medicine background)
2. **AI/ML Skills**: Deep learning implementation, model optimization, and deployment considerations

The use of human chest X-rays as a proxy allows for practical implementation while maintaining clinical relevance. The pipeline can be adapted for veterinary datasets when available, leveraging the same preprocessing, training, and evaluation framework.


## 🛠️ Technologies

- **PyTorch**: Deep learning framework
- **torchvision**: Pretrained models and transforms
- **timm**: Vision Transformer models
- **scikit-learn**: Metrics computation
- **matplotlib**: Visualization
- **numpy, pandas**: Data manipulation


## 👤 Author

**Andreea Nicoleta Brandiburu**  
MSc Data Science Student | Embedded SW Engineer  
*Bridging veterinary medicine and artificial intelligence*

---

**Note**: This project uses human chest X-ray datasets as a proxy for veterinary radiographs to demonstrate deep learning techniques in a medical imaging context. The pipeline is designed to be adaptable to veterinary datasets when available.
