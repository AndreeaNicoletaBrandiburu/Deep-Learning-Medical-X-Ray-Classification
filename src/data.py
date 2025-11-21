"""
Data loading utilities for chest X-ray classification.
Uses public human chest X-ray dataset as proxy for veterinary radiographs.
"""

from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation transforms.
    
    Args:
        img_size: Target image size
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return train_tf, val_tf


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing train/val/test subdirectories
        batch_size: Batch size for dataloaders
        img_size: Target image size
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    train_tf, val_tf = get_transforms(img_size)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=val_tf)

    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dl, val_dl, test_dl, train_ds.classes


def compute_class_weights(dataloader):
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        dataloader: Training dataloader
    
    Returns:
        Tensor of class weights
    """
    import torch
    from collections import Counter
    
    all_labels = []
    for _, labels in dataloader:
        # Handle both tensor and numpy array labels
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.cpu().numpy())
        else:
            all_labels.extend(labels)
    
    class_counts = Counter(all_labels)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # Compute weights: inverse frequency
    weights = torch.zeros(num_classes)
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)
    
    return weights

