import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.config import GATEKEEPER_MODEL_PATH, FIGURES_DIR, SAMPLE_IMAGES_DIR, DEVICE
from src.model import load_checkpoint, get_transforms

def generate_sample_gradcams():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model, classes, _ = load_checkpoint(GATEKEEPER_MODEL_PATH)
    transform = get_transforms(is_train=False)
    
    sample_files = [
        ("example_eczema.jpg", "Eczema"),
        ("example_melanoma.jpg", "Melanoma"),
        ("example_psoriasis.jpg", "Psoriasis")
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    for row, (fname, label) in enumerate(sample_files):
        img_path = os.path.join(SAMPLE_IMAGES_DIR, fname)
        if not os.path.exists(img_path):
            continue
            
        raw_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        rgb_img = np.float32(raw_img.resize((224, 224))) / 255
        vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        axes[row, 0].imshow(raw_img)
        axes[row, 0].set_title(f"Original: {label}", fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(vis)
        axes[row, 1].set_title(f"Grad-CAM Attention: {label}", fontsize=11, fontweight='bold', color='#0284c7')
        axes[row, 1].axis('off')
        
    plt.tight_layout()
    gradcam_fig_path = os.path.join(FIGURES_DIR, "05_gradcam_overlay.png")
    plt.savefig(gradcam_fig_path, dpi=200)
    plt.close()
    print(f"[EXPLAIN] Grad-CAM heatmaps saved to {gradcam_fig_path}")

if __name__ == "__main__":
    generate_sample_gradcams()
