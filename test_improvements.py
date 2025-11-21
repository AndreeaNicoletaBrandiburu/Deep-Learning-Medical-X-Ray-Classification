"""
Quick test script to verify mixed precision and benchmarking work correctly.
"""

import torch
import torch.nn as nn
from src.models import build_resnet18
from src.train import train_one_epoch
from src.benchmark import benchmark_model, get_model_size_mb
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

def test_mixed_precision():
    """Test that mixed precision training works without errors."""
    print("Testing Mixed Precision Training...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create dummy model and data
    model = build_resnet18(num_classes=2, pretrained=False, dropout=0.5)
    model = model.to(device)
    
    # Create dummy dataset (ResNet expects 3 channels RGB)
    dummy_data = torch.randn(32, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (32,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    # Test with FP32
    print("  Testing FP32 (standard precision)...")
    try:
        loss1, acc1, auc1 = train_one_epoch(
            model, dummy_dataloader, criterion, optimizer, device,
            use_amp=False, scaler=None
        )
        print(f"    [OK] FP32 training successful: loss={loss1:.4f}")
    except Exception as e:
        print(f"    [FAIL] FP32 training failed: {e}")
        return False
    
    # Test with FP16 (if CUDA available)
    if device == "cuda":
        print("  Testing FP16 (mixed precision)...")
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        try:
            loss2, acc2, auc2 = train_one_epoch(
                model, dummy_dataloader, criterion, optimizer, device,
                use_amp=True, scaler=scaler
            )
            print(f"    [OK] FP16 training successful: loss={loss2:.4f}")
        except Exception as e:
            print(f"    [FAIL] FP16 training failed: {e}")
            return False
    else:
        print("  Skipping FP16 test (CPU mode)")
    
    return True

def test_benchmarking():
    """Test that benchmarking functions work."""
    print("\nTesting Benchmarking Functions...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy model
    model = build_resnet18(num_classes=2, pretrained=False, dropout=0.5)
    model = model.to(device)
    model.eval()
    
    # Test model size calculation
    print("  Testing model size calculation...")
    try:
        size_mb = get_model_size_mb(model)
        print(f"    [OK] Model size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"    [FAIL] Model size calculation failed: {e}")
        return False
    
    # Test GPU memory (if CUDA)
    if device == "cuda":
        print("  Testing GPU memory functions...")
        from src.benchmark import get_gpu_memory_usage
        try:
            mem_info = get_gpu_memory_usage()
            if mem_info['available']:
                print(f"    [OK] GPU memory check successful")
                print(f"      Allocated: {mem_info['allocated_gb']:.2f} GB")
            else:
                print(f"    [FAIL] GPU memory check failed")
                return False
        except Exception as e:
            print(f"    [FAIL] GPU memory check failed: {e}")
            return False
    
    return True

if __name__ == '__main__':
    print("="*60)
    print("TESTING IMPROVEMENTS")
    print("="*60)
    
    # Test mixed precision
    mp_ok = test_mixed_precision()
    
    # Test benchmarking
    bench_ok = test_benchmarking()
    
    print("\n" + "="*60)
    if mp_ok and bench_ok:
        print("[SUCCESS] ALL TESTS PASSED")
    else:
        print("[FAILED] SOME TESTS FAILED")
    print("="*60)

