"""
Model compression utilities: pruning and quantization.
Demonstrates model optimization for deployment scenarios.
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from torch.nn.utils import prune
from typing import Dict


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def apply_pruning(
    model: nn.Module,
    pruning_ratio: float = 0.3,
    method: str = 'l1_unstructured'
) -> nn.Module:
    """
    Apply pruning to the final fully-connected layer.
    
    Args:
        model: PyTorch model to prune
        pruning_ratio: Fraction of weights to prune (0.0 to 1.0)
        method: Pruning method ('l1_unstructured' or 'random_unstructured')
    
    Returns:
        Pruned model
    """
    # Find the final linear layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith('fc'):
            # Apply pruning
            if method == 'l1_unstructured':
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            elif method == 'random_unstructured':
                prune.random_unstructured(module, name='weight', amount=pruning_ratio)
            else:
                raise ValueError(f"Unknown pruning method: {method}")
            
            # Make pruning permanent
            prune.remove(module, 'weight')
            break
    
    return model


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply dynamic quantization (INT8) to the model.
    Quantization is performed during inference.
    
    Args:
        model: PyTorch model to quantize
    
    Returns:
        Quantized model
    """
    model.eval()
    
    # Quantize the model
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    return quantized_model


def compare_model_sizes(
    original_model: nn.Module,
    compressed_model: nn.Module
) -> Dict[str, float]:
    """
    Compare original and compressed model sizes.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model
    
    Returns:
        Dictionary with size comparisons
    """
    original_size = get_model_size_mb(original_model)
    compressed_size = get_model_size_mb(compressed_model)
    
    reduction = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
    
    return {
        'original_size_mb': original_size,
        'compressed_size_mb': compressed_size,
        'reduction_percent': reduction
    }


def compress_model_example(
    model: nn.Module,
    pruning_ratio: float = 0.3,
    apply_quantization: bool = True
) -> tuple:
    """
    Example of applying both pruning and quantization.
    
    Args:
        model: Model to compress
        pruning_ratio: Ratio for pruning
        apply_quantization: Whether to apply quantization
    
    Returns:
        Tuple of (pruned_model, quantized_model, comparison_dict)
    """
    print("Original model:")
    print(f"  Parameters: {count_parameters(model)['total']:,}")
    print(f"  Size: {get_model_size_mb(model):.2f} MB\n")
    
    # Apply pruning
    pruned_model = apply_pruning(model, pruning_ratio=pruning_ratio)
    print(f"After pruning ({pruning_ratio*100:.0f}%):")
    print(f"  Size: {get_model_size_mb(pruned_model):.2f} MB")
    
    quantized_model = None
    if apply_quantization:
        # Apply quantization
        quantized_model = apply_dynamic_quantization(pruned_model)
        print(f"\nAfter quantization:")
        print(f"  Size: {get_model_size_mb(quantized_model):.2f} MB")
    
    # Comparison
    comparison = compare_model_sizes(model, quantized_model if quantized_model else pruned_model)
    print(f"\nTotal reduction: {comparison['reduction_percent']:.2f}%")
    
    return pruned_model, quantized_model, comparison

