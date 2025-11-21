"""
Training utilities for deep learning models.
Includes training loop, validation, and metrics computation.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Tuple, Optional


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_amp=False, scaler=None):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        use_amp: Whether to use Automatic Mixed Precision (FP16)
        scaler: GradScaler for mixed precision (required if use_amp=True)
    
    Returns:
        Tuple of (average_loss, accuracy, auc)
    """
    model.train()
    losses, preds_all, labels_all = [], [], []

    for x, y in tqdm(dataloader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            # Mixed precision training
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds_all.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    acc = accuracy_score(labels_all, (torch.tensor(preds_all) > 0.5).int())
    try:
        auc = roc_auc_score(labels_all, preds_all)
    except ValueError:
        auc = float("nan")
    return sum(losses) / len(losses), acc, auc


def eval_one_epoch(model, dataloader, criterion, device):
    """
    Evaluate model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation/test dataloader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Tuple of (average_loss, accuracy, auc)
    """
    model.eval()
    losses, preds_all, labels_all = [], [], []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Eval", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            losses.append(loss.item())
            preds_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    acc = accuracy_score(labels_all, (torch.tensor(preds_all) > 0.5).int())
    try:
        auc = roc_auc_score(labels_all, preds_all)
    except ValueError:
        auc = float("nan")
    return sum(losses) / len(losses), acc, auc


def train_experiment(
    model,
    train_dl,
    val_dl,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = None,
    weight_decay: float = 1e-4,
    use_class_weights: bool = True,
    early_stopping_patience: int = 5,
    use_amp: bool = False,
):
    """
    Full training experiment with early stopping based on validation AUC.
    
    Args:
        model: PyTorch model to train
        train_dl: Training dataloader
        val_dl: Validation dataloader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to run on (auto-detects if None)
        weight_decay: L2 regularization strength
        use_class_weights: Whether to use class weights for imbalanced datasets
        early_stopping_patience: Number of epochs to wait before early stopping
        use_amp: Whether to use Automatic Mixed Precision (FP16) for faster training
    
    Returns:
        Trained model with best weights loaded
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    # Setup mixed precision if requested and CUDA available
    if use_amp and device == "cuda":
        scaler = GradScaler()
        print("  - Mixed Precision Training (FP16): enabled")
    else:
        scaler = None
        if use_amp and device != "cuda":
            print("  - Mixed Precision Training: disabled (requires CUDA)")
    
    # Compute class weights if requested
    if use_class_weights:
        from src.data import compute_class_weights
        class_weights = compute_class_weights(train_dl).to(device)
        print(f"Class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Add weight decay for L2 regularization
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_auc, best_state = -1, None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_auc = train_one_epoch(
            model, train_dl, criterion, optimizer, device, 
            use_amp=use_amp, scaler=scaler
        )
        val_loss, val_acc, val_auc = eval_one_epoch(model, val_dl, criterion, device)

        scheduler.step(val_loss)
        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_auc={tr_auc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best validation AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience: {early_stopping_patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nBest model loaded with validation AUC: {best_auc:.4f}")
    
    return model

