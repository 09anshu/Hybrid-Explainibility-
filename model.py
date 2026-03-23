import torch
import torch.nn as nn
from torchvision import models

# ────────────────────────────────────────────────────────────────────────────
#  7-label schema — 5 CheXpert competition labels + 2 specialized targets
#
#  Indices 0-4 : Original CheXpert competition labels (Irvin et al. 2019)
#  Index 5     : Pneumothorax  — air in the pleural space; life-threatening
#  Index 6     : Fracture      — rib / vertebral fractures; trauma / osteoporosis
#
#  All 7 labels are present in the CheXpert train.csv and valid.csv so no
#  additional dataset is required. The masked-BCE loss skips labels that are
#  not annotated in external datasets (NIH, RSNA, Kaggle).
# ────────────────────────────────────────────────────────────────────────────
COMPETITION_LABELS = [
    "Atelectasis",      # index 0
    "Cardiomegaly",     # index 1
    "Consolidation",    # index 2
    "Edema",            # index 3
    "Pleural Effusion", # index 4
    "Pneumothorax",     # index 5  ← specialized: air in pleural space
    "Fracture",         # index 6  ← specialized: rib / vertebral fracture
]
NUM_CLASSES = len(COMPETITION_LABELS)  # 7

# Subset views for the UI
CHEXPERT_5_LABELS   = COMPETITION_LABELS[:5]   # original 5 competition labels
SPECIALIZED_LABELS  = COMPETITION_LABELS[5:]   # Pneumothorax + Fracture


def get_densenet121_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    DenseNet121 adapted for 7-label multi-label classification.

    Labels 0-4 : CheXpert competition labels (AUC target ≥ 0.85)
    Label  5   : Pneumothorax  — specialized detection target
    Label  6   : Fracture      — rib/vertebral fracture detection

    Architecture:
      ImageNet pre-trained DenseNet121 → replace FC head with Linear(1024, num_classes).

    Why DenseNet121:
      - Reference architecture from CheXpert paper (Irvin et al. 2019) and CheXNet
        (Rajpurkar et al. 2017).  Dense connections preserve fine-grained texture
        gradients (lung opacity, consolidation, fluid lines, rib cortex) all the way to
        the classifier — ResNet skip-connections lose this spatial detail.
      - Stanford ensemble AUC 0.907 using DenseNet121.  Our target: ≥ 0.90 on
        all 5 CheXpert labels + Pneumothorax detection AUC ≥ 0.92.

    Output: raw logits (shape [B, num_classes]).
    Apply torch.sigmoid() at inference time to get probabilities per class.

    Grad-CAM target layer: model.features.denseblock4
    """
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.densenet121(weights=weights)

    # Replace classifier head: 1024-d features → num_classes logits
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model


def load_backbone_weights(model, weights_path, device='cpu'):
    """
    Load weights into model, falling back to backbone-only if the classifier
    head shape doesn't match (e.g. upgrading from 5-class → 7-class weights).

    Returns (model, mode) where mode is 'full' or 'backbone-only'.
    """
    import torch
    state = torch.load(weights_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
        return model, 'full'
    except RuntimeError:
        # Classifier dimension mismatch — load backbone only
        backbone_state = {k: v for k, v in state.items()
                          if not k.startswith('classifier')}
        model.load_state_dict(backbone_state, strict=False)
        return model, 'backbone-only'


# ────────────────────────────────────────────────────────────────────────────
#  Backward-compat alias
# ────────────────────────────────────────────────────────────────────────────
def get_resnet50_model(num_classes=NUM_CLASSES, pretrained=True):
    """Deprecated alias → transparently returns DenseNet121."""
    return get_densenet121_model(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    model       = get_densenet121_model()
    dummy_input = torch.randn(4, 3, 224, 224)
    output      = model(dummy_input)
    print(f"Output shape  : {output.shape}")           # Expected: [4, 5]
    print(f"Labels        : {COMPETITION_LABELS}")
    print(f"GradCAM layer : model.features.denseblock4")
