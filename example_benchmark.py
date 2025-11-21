"""
Example script for benchmarking model performance.
Demonstrates throughput, latency, and GPU memory usage analysis.
"""

import torch
from src.models import build_resnet18
from src.data import get_dataloaders
from src.benchmark import benchmark_model, print_benchmark_comparison

if __name__ == '__main__':
    # Configuration
    data_dir = 'data'
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = 'best_resnet18_model.pth'
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\nLoading dataset...")
    train_dl, val_dl, test_dl, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=224,
        num_workers=4
    )
    
    # Load model
    print("\nLoading model...")
    model = build_resnet18(num_classes=len(class_names), pretrained=False, dropout=0.5)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Warning: {checkpoint_path} not found. Using untrained model.")
    
    model.eval()
    
    # Benchmark with FP32
    print("\n" + "="*60)
    print("BENCHMARKING WITH FP32 (Standard Precision)")
    print("="*60)
    metrics_fp32 = benchmark_model(
        model=model,
        dataloader=test_dl,
        device=device,
        use_amp=False,
        num_runs=100
    )
    
    # Benchmark with FP16 (Mixed Precision)
    if device == "cuda":
        print("\n" + "="*60)
        print("BENCHMARKING WITH FP16 (Mixed Precision)")
        print("="*60)
        metrics_fp16 = benchmark_model(
            model=model,
            dataloader=test_dl,
            device=device,
            use_amp=True,
            num_runs=100
        )
        
        # Comparison
        print_benchmark_comparison(
            [metrics_fp32, metrics_fp16],
            ['ResNet-18 (FP32)', 'ResNet-18 (FP16)']
        )
    else:
        print("\nMixed precision benchmarking skipped (CPU mode)")

