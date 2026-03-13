import torch
import torch.nn as nn
from torchvision import models

# ────────────────────────────────────────────────────────────────────────────
#  CheXpert 5 competition labels (Irvin et al. 2019) — model output order
# ────────────────────────────────────────────────────────────────────────────
COMPETITION_LABELS = [
    "Atelectasis",      # index 0
    "Cardiomegaly",     # index 1
    "Consolidation",    # index 2
    "Edema",            # index 3
    "Pleural Effusion", # index 4
]
NUM_CLASSES = len(COMPETITION_LABELS)  # 5


def get_densenet121_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    DenseNet121 adapted for CheXpert 5-label multi-label classification.

    Architecture:
      ImageNet pre-trained DenseNet121 → replace FC head with Linear(1024, num_classes).

    Why DenseNet121:
      - Reference architecture from CheXpert paper (Irvin et al. 2019) and CheXNet
        (Rajpurkar et al. 2017).  Dense connections preserve fine-grained texture
        gradients (lung opacity, consolidation, fluid lines) all the way to the
        classifier — ResNet skip-connections lose this spatial detail.
      - Stanford ensemble AUC 0.907 using DenseNet121.  Our target: ≥ 0.85 on
        GTX 1650 hardware.

    Output: raw logits (shape [B, num_classes]).
    Apply torch.sigmoid() at inference time to get probabilities per class.

    Grad-CAM target layer: model.features.denseblock4
    """
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.densenet121(weights=weights)

    # Replace classifier head: 1024-d features → num_classes logits
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model


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
