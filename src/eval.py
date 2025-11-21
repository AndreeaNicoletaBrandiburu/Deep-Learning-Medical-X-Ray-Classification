"""
Evaluation utilities for model assessment.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import numpy as np
from typing import Dict, Tuple


def find_optimal_threshold(y_true, y_probs) -> float:
    """
    Find optimal threshold that maximizes F1-score.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
    
    Returns:
        Optimal threshold value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = None,
    class_names: list = None,
    threshold: float = None
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained PyTorch model
        dataloader: Test dataloader
        device: Device to run on
        class_names: List of class names
        threshold: Classification threshold (if None, uses argmax or finds optimal)
    
    Returns:
        Dictionary of metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    
    all_preds, all_probs, all_labels = [], [], []
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Use optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(all_labels, all_probs)
        print(f"Optimal threshold found: {threshold:.4f}")
    
    all_preds = (all_probs >= threshold).astype(int)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, average='binary', zero_division=0),
        'threshold': threshold,
    }
    
    # ROC-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics['roc_auc'] = float('nan')
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
    
    if class_names:
        metrics['class_names'] = class_names
    
    return metrics


def print_evaluation_report(metrics: Dict):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    print(f"Loss:              {metrics['loss']:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1-Score:          {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    if 'threshold' in metrics:
        print(f"Threshold:         {metrics['threshold']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    print("="*50 + "\n")

