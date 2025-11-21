"""
Example script for model compression.
Demonstrates pruning and quantization.
"""

import torch
from src.models import build_resnet18
from src.compress import compress_model_example, count_parameters, get_model_size_mb

# Configuration
checkpoint_path = 'best_resnet18_model.pth'
num_classes = 2

# Load model
print("Loading model...")
model = build_resnet18(num_classes=num_classes, pretrained=False)
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("\n" + "="*60)
print("MODEL COMPRESSION DEMONSTRATION")
print("="*60)

# Apply compression
pruned_model, quantized_model, comparison = compress_model_example(
    model,
    pruning_ratio=0.3,
    apply_quantization=True
)

# Save compressed models
print("\nSaving compressed models...")
torch.save({
    'model_state_dict': pruned_model.state_dict(),
    'compression_type': 'pruned',
    'pruning_ratio': 0.3
}, 'compressed_pruned_model.pth')

if quantized_model:
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'compression_type': 'quantized'
    }, 'compressed_quantized_model.pth')

print("\nCompressed models saved!")
print(f"  - Pruned: compressed_pruned_model.pth")
print(f"  - Quantized: compressed_quantized_model.pth")

