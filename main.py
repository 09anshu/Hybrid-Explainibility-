import os
import torch
from torchvision import transforms
from PIL import Image

from model import get_densenet121_model, COMPETITION_LABELS
from gradcam_utils import get_gradcam_heatmap

DEFAULT_WEIGHTS = "best_densenet121.pth"


def run_pipeline(img_path, model_weight_path=DEFAULT_WEIGHTS, gradcam_class_idx=0):
    """
    Run the full inference pipeline for one chest X-ray image.

    Parameters
    ----------
    img_path          : path to JPEG/PNG chest X-ray
    model_weight_path : path to trained .pth weights file
    gradcam_class_idx : which of the 5 competition labels to visualise
                        0=Atelectasis, 1=Cardiomegaly, 2=Consolidation,
                        3=Edema, 4=Pleural Effusion

    Returns
    -------
    predictions    : dict {label: probability}  — 5 competition labels
    heatmap_vis    : np.ndarray (H, W, 3) Grad-CAM overlay
    active_regions : str  — human-readable description of high-activation quadrants
    error_msg      : str or None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────
    model = get_densenet121_model(pretrained=False)
    if os.path.exists(model_weight_path):
        model.load_state_dict(
            torch.load(model_weight_path, map_location=device, weights_only=True)
        )
    model.to(device)
    model.eval()

    # ── Preprocess ────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None, None, None, f"Failed to load image: {e}"

    input_tensor = transform(image).unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        output_logits = model(input_tensor)
        probabilities = torch.sigmoid(output_logits)[0].cpu().numpy()

    predictions = {
        label: float(prob)
        for label, prob in zip(COMPETITION_LABELS, probabilities)
    }

    # ── Grad-CAM on requested class ───────────────────────────────────────
    target_layer = model.features.denseblock4
    heatmap_vis, active_regions = get_gradcam_heatmap(
        model, input_tensor.cpu(), target_layer, img_path,
        target_category_idx=gradcam_class_idx
    )

    # Return everything cleanly
    return predictions, heatmap_vis, active_regions, None


if __name__ == "__main__":
    # Quick smoke-test on a CheXpert frontal image
    sample_img = "train/patient00001/study1/view1_frontal.jpg"
    if os.path.exists(sample_img):
        preds, cam, regions, err = run_pipeline(sample_img, gradcam_class_idx=3)  # Edema
        if err:
            print(f"Error: {err}")
        else:
            print("Pipeline OK!")
            for label, prob in preds.items():
                bar = "#" * int(prob * 20)
                print(f"  {label:25s}: {prob:.3f}  [{bar:<20}]")
            print(f"  Grad-CAM active regions : {regions}")
    else:
        print(f"Sample image not found at {sample_img!r}. Run after CheXpert data extracted.")
