"""
Example script for Grad-CAM visualization.
Run this after training a model to visualize attention regions.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models import build_resnet18
from src.explainability import create_gradcam_for_resnet

# Configuration
checkpoint_path = 'best_resnet18_model.pth'
image_path = 'data/test/PNEUMONIA/person1_bacteria_1.jpeg'  # Example path
class_names = ['NORMAL', 'PNEUMONIA']
save_path = 'gradcam_visualization.png'

# Load model
print("Loading model...")
model = build_resnet18(num_classes=len(class_names), pretrained=False)
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
print(f"Loading image: {image_path}")
img = Image.open(image_path).convert('RGB')
original_img = np.array(img)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
input_tensor = transform(img).unsqueeze(0)

# Create Grad-CAM
print("Generating Grad-CAM visualization...")
gradcam = create_gradcam_for_resnet(model)

# Visualize
gradcam.visualize(
    input_tensor,
    original_img,
    class_names,
    save_path=save_path
)

print(f"\nVisualization saved to {save_path}")

