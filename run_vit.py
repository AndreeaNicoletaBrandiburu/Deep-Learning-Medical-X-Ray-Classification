"""
Quick training script for Vision Transformer model.
Run this to train a Vision Transformer baseline.
"""

import torch
from src.data import get_dataloaders
from src.models import build_vit
from src.train import train_experiment

if __name__ == '__main__':
    # Configuration
    data_dir = 'data'
    batch_size = 16  # ViT needs smaller batch size
    epochs = 5  # Start with 5 epochs for testing
    learning_rate = 1e-5  # Lower LR for ViT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
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
    
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_dl)}")
    print(f"Val batches: {len(val_dl)}")
    
    # Build model
    print("\nBuilding Vision Transformer model...")
    model = build_vit(num_classes=len(class_names), pretrained=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train
    print("\nStarting training...")
    trained_model = train_experiment(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=epochs,
        lr=learning_rate,
        device=device
    )
    
    # Save model
    save_path = 'best_vit_model.pth'
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names)
    }, save_path)
    print(f"\nModel saved to {save_path}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    from src.eval import evaluate_model, print_evaluation_report
    
    test_metrics = evaluate_model(trained_model, test_dl, device=device, class_names=class_names)
    print_evaluation_report(test_metrics)

