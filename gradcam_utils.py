import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus   # Grad-CAM++ (Chattopadhay et al.)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# Input resolution must match the val_transform used at inference time
_INFER_SIZE = 320


def get_target_layer(model):
    """
    Auto-detect the best Grad-CAM++ target layer for the model architecture.

    DenseNet121 : model.features.denseblock4
        Dense connections preserve spatial texture features (consolidated
        lung areas, fluid lines) right up to the last feature block, making
        denseblock4 the richest source of class-discriminative activations.
        Grad-CAM++ is preferred over Grad-CAM for multi-instance localization
        (PMC11355845 survey: Grad-CAM++ is state-of-the-art for CXR XAI).

    ResNet50 (legacy fallback) : model.layer4[-1]
    """
    # DenseNet121
    if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
        return model.features.denseblock4
    # ResNet (legacy)
    if hasattr(model, 'layer4'):
        return model.layer4[-1]
    raise ValueError("Unknown model architecture — cannot auto-detect Grad-CAM++ target layer.")

def get_gradcam_heatmap(model, input_tensor, target_layer, img_path, target_category_idx=None):
    """
    Generates a Grad-CAM heatmap for a given input tensor, and identifies high-activation regions.
    """
    # 1. Initialize Grad-CAM++ (superior localization vs plain Grad-CAM for
    #    multi-class medical images per Chattopadhay et al. 2018)
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    
    # 2. Setup Targets
    targets = None
    if target_category_idx is not None:
        targets = [ClassifierOutputTarget(target_category_idx)]
    
    # 3. Generate Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 4. Load original image for overlay
    original_img = Image.open(img_path).convert('RGB')
    original_img = original_img.resize((_INFER_SIZE, _INFER_SIZE))
    original_img_np = np.array(original_img) / 255.0 # Normalize for show_cam_on_image
    
    # 5. Create Overlay
    visualization = show_cam_on_image(original_img_np, grayscale_cam, use_rgb=True)
    
    # 6. Determine high activation regions 
    h, w = grayscale_cam.shape
    mid_h, mid_w = h // 2, w // 2
    
    regions_activations = {
        "upper right": np.mean(grayscale_cam[:mid_h, mid_w:]),
        "upper left": np.mean(grayscale_cam[:mid_h, :mid_w]),
        "lower right": np.mean(grayscale_cam[mid_h:, mid_w:]),
        "lower left": np.mean(grayscale_cam[mid_h:, :mid_w])
    }
    
    active_regions = [region for region, act in regions_activations.items() if act > 0.4]
    if not active_regions:
        max_region = max(regions_activations, key=regions_activations.get)
        active_regions = [max_region]
        
    regions_text = ", ".join(active_regions)
    
    return visualization, regions_text

if __name__ == "__main__":
    print("Grad-CAM utility ready.")
