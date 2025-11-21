"""
Grad-CAM implementation for model explainability.
Visualizes which image regions the model focuses on for predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple
import os


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN models.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Save activation maps."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (None for highest prediction)
        
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # Weighted combination
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam
    
    def visualize(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_names: list,
        class_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize Grad-CAM on the original image.
        
        Args:
            input_tensor: Preprocessed input tensor
            original_image: Original image (numpy array, RGB)
            class_names: List of class names
            class_idx: Target class index
            save_path: Path to save visualization
        """
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Resize CAM to original image size
        h, w = original_image.shape[:2]
        cam = cv2.resize(cam, (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = np.float32(original_image) / 255
        heatmap_norm = np.float32(heatmap) / 255
        overlayed = 0.6 * overlayed + 0.4 * heatmap_norm
        overlayed = np.clip(overlayed, 0, 1)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            pred_prob = probs[0, pred_class].item()
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image\nPredicted: {class_names[pred_class]} ({pred_prob:.2f})')
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap\n(Red = High Attention)')
        axes[1].axis('off')
        
        axes[2].imshow(overlayed)
        axes[2].set_title('Overlayed Visualization\n(Model Focus Regions)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_gradcam_for_resnet(model: nn.Module, layer_name: str = 'layer4') -> GradCAM:
    """
    Create Grad-CAM for ResNet model.
    
    Args:
        model: ResNet model
        layer_name: Layer name to target (default: 'layer4' for ResNet)
    
    Returns:
        GradCAM object
    """
    target_layer = getattr(model, layer_name)
    # Get the last conv layer in the block
    target_layer = target_layer[-1].conv2 if hasattr(target_layer[-1], 'conv2') else target_layer[-1]
    return GradCAM(model, target_layer)

