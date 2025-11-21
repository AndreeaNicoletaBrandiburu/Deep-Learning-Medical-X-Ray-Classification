"""
Model architectures for chest X-ray classification.
CNN (ResNet) and Vision Transformer implementations.
"""

import torch
import torch.nn as nn
from torchvision import models
import timm


def build_resnet18(num_classes: int, pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    """
    Build ResNet-18 model for classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability for regularization
    
    Returns:
        ResNet-18 model
    """
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    # Add dropout before final layer to reduce overfitting
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_vit(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build Vision Transformer model for classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Vision Transformer model
    """
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model

