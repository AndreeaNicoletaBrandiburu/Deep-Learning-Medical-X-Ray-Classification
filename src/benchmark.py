"""
Performance benchmarking utilities for model evaluation.
Measures throughput, latency, GPU memory usage, and model size.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from typing import Dict, Optional


def benchmark_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    use_amp: bool = False
) -> Dict:
    """
    Benchmark model inference performance.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for inference
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark runs
        use_amp: Whether to use mixed precision
    
    Returns:
        Dictionary with performance metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    
    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= num_warmup:
                break
            x = x.to(device)
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
    
    # Synchronize GPU
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []
    total_samples = 0
    
    with torch.no_grad():
        start_time = time.time()
        for i, (x, _) in enumerate(dataloader):
            if i >= num_runs:
                break
            
            x = x.to(device)
            batch_size = x.size(0)
            
            # Measure latency
            if device == "cuda":
                torch.cuda.synchronize()
            
            iter_start = time.time()
            
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            iter_end = time.time()
            latency_ms = (iter_end - iter_start) * 1000
            latencies.append(latency_ms)
            total_samples += batch_size
        
        end_time = time.time()
    
    total_time = end_time - start_time
    throughput = total_samples / total_time  # samples per second
    
    metrics = {
        'throughput_samples_per_sec': throughput,
        'throughput_images_per_sec': throughput,
        'avg_latency_ms': np.mean(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'total_samples': total_samples,
        'total_time_sec': total_time,
        'device': device,
        'use_amp': use_amp
    }
    
    return metrics


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def get_gpu_memory_usage(device: int = 0) -> Dict:
    """Get GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {'available': False}
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    
    return {
        'available': True,
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated
    }


def benchmark_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = None,
    use_amp: bool = False,
    num_runs: int = 100
) -> Dict:
    """
    Comprehensive model benchmarking.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for benchmarking
        device: Device to run on
        use_amp: Whether to use mixed precision
        num_runs: Number of benchmark iterations
    
    Returns:
        Dictionary with all benchmark metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("MODEL BENCHMARKING")
    print("="*60)
    
    # Model size
    model_size_mb = get_model_size_mb(model)
    print(f"\nModel Size: {model_size_mb:.2f} MB")
    
    # GPU memory (before)
    if device == "cuda":
        gpu_mem_before = get_gpu_memory_usage()
        print(f"\nGPU Memory (before):")
        print(f"  Allocated: {gpu_mem_before['allocated_gb']:.2f} GB")
        print(f"  Reserved: {gpu_mem_before['reserved_gb']:.2f} GB")
    
    # Inference benchmark
    inference_metrics = benchmark_inference(
        model, dataloader, device=device, 
        num_runs=num_runs, use_amp=use_amp
    )
    
    # GPU memory (after)
    if device == "cuda":
        gpu_mem_after = get_gpu_memory_usage()
        print(f"\nGPU Memory (after):")
        print(f"  Allocated: {gpu_mem_after['allocated_gb']:.2f} GB")
        print(f"  Reserved: {gpu_mem_after['reserved_gb']:.2f} GB")
        print(f"  Max Allocated: {gpu_mem_after['max_allocated_gb']:.2f} GB")
    
    # Combine all metrics
    all_metrics = {
        'model_size_mb': model_size_mb,
        'inference': inference_metrics
    }
    
    if device == "cuda":
        all_metrics['gpu_memory'] = gpu_mem_after
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Throughput: {inference_metrics['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Average Latency: {inference_metrics['avg_latency_ms']:.2f} ms")
    print(f"P95 Latency: {inference_metrics['p95_latency_ms']:.2f} ms")
    print(f"P99 Latency: {inference_metrics['p99_latency_ms']:.2f} ms")
    print("="*60 + "\n")
    
    return all_metrics


def print_benchmark_comparison(metrics_list: list, model_names: list):
    """
    Print comparison table of multiple model benchmarks.
    
    Args:
        metrics_list: List of benchmark dictionaries
        model_names: List of model names
    """
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Throughput':<15} {'Avg Latency':<15} {'P95 Latency':<15} {'Size (MB)':<12}")
    print("-"*80)
    
    for name, metrics in zip(model_names, metrics_list):
        inf = metrics['inference']
        throughput = inf['throughput_samples_per_sec']
        avg_lat = inf['avg_latency_ms']
        p95_lat = inf['p95_latency_ms']
        size = metrics['model_size_mb']
        
        print(f"{name:<20} {throughput:>10.2f} img/s  {avg_lat:>10.2f} ms  {p95_lat:>10.2f} ms  {size:>8.2f} MB")
    
    print("="*80 + "\n")

