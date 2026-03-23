"""
train.py — 3-phase DenseNet121 training pipeline.
Optimised for local GTX 1650 (4 GB VRAM) with CUDA 12.9.

Phase 1  Pre-training  : CheXpert 5-label (+ optional NIH / MIMIC-CXR)
Phase 2  Fine-tuning   : RSNA Pneumonia Detection Challenge
Phase 3  Final-tuning  : Kaggle Binary Chest X-Ray

GTX 1650 settings used throughout:
  batch_size=8, grad_accum_steps=4  → effective batch = 32
  num_workers=4 with persistent_workers=True
  AMP (torch.amp) + cudnn.benchmark
  Checkpoint every epoch → safe resume

Masked BCE loss:
  Each sample carries a mask tensor from the dataset. Loss is only computed
  on labelled entries — allows mixing datasets with different label spaces.

AUC tracking:
  Per-class AUC (sklearn.metrics.roc_auc_score) computed on the official
  CheXpert validation set (234 radiologist-labeled images) at every epoch.
  Mean AUC of the 5 competition labels is the primary metric.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from model   import get_densenet121_model, COMPETITION_LABELS
from dataset import (
    get_phase1_dataloaders,
    get_phase2_dataloaders,
    get_phase3_dataloaders,
)

# ────────────────────────────────────────────────────────────────────────────
#  Global settings
# ────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True   # Tune convolutions for fixed input size


# ════════════════════════════════════════════════════════════════════════════
#  MASKED BCE LOSS
# ════════════════════════════════════════════════════════════════════════════

class MaskedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss that ignores label positions where mask == 0,
    with optional label smoothing (Müller et al. 2019).

    Forward:
      logits  : [B, C]
      targets : [B, C]  — float, 0.0 or 1.0
      masks   : [B, C]  — float, 1.0 = labelled, 0.0 = unknown

    label_smoothing ε:
      Soft targets: 1 → 1-ε,  0 → ε.
      Prevents overconfident sigmoid saturation; particularly helpful when
      training on uncertain/noisy CheXpert labels.
    """
    def __init__(self, pos_weight=None, label_smoothing=0.0):
        super().__init__()
        self.pos_weight     = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, masks):
        # Apply label smoothing
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            targets = targets * (1.0 - 2 * eps) + eps

        # Element-wise BCE (no reduction)
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            loss_elem = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw.unsqueeze(0), reduction='none'
            )
        else:
            loss_elem = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
        # Zero out unknown-label positions
        masked   = loss_elem * masks.to(logits.device)
        n_active = masks.to(logits.device).sum().clamp(min=1.0)
        return masked.sum() / n_active


# ════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(state, filepath):
    torch.save(state, filepath)
    print(f"  [Checkpoint] Saved → {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load checkpoint. Returns (start_epoch, best_mean_auc)."""
    if not os.path.exists(filepath):
        return 0, 0.0
    ckpt = torch.load(filepath, map_location=DEVICE, weights_only=False)
    resumed = True
    try:
        model.load_state_dict(ckpt['model_state'])
    except RuntimeError:
        # Classifier size mismatch (e.g. 5-label ckpt -> 7-label model):
        # load backbone only and restart epoch/lr states.
        partial = {
            k: v for k, v in ckpt['model_state'].items()
            if not k.startswith('classifier')
        }
        model.load_state_dict(partial, strict=False)
        resumed = False
        print("  [Checkpoint] Loaded backbone-only from legacy checkpoint; "
              "classifier head reinitialized for current label set.")

    if not resumed:
        return 0, 0.0

    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch   = ckpt.get('epoch', 0) + 1
    best_mean_auc = ckpt.get('best_mean_auc', 0.0)
    print(f"  [Checkpoint] Resumed from epoch {ckpt.get('epoch', 0) + 1} "
          f"(best mean AUC: {best_mean_auc:.4f})")
    return start_epoch, best_mean_auc


# ════════════════════════════════════════════════════════════════════════════
#  SINGLE-EPOCH TRAIN (with gradient accumulation)
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, grad_accum_steps=4):
    """
    Train for one epoch with gradient accumulation and gradient clipping.

    grad_accum_steps: accumulate gradients over N mini-batches before stepping.
    Effective batch size = loader.batch_size × grad_accum_steps = 8 × 4 = 32.

    Gradient clipping (max_norm=1.0) prevents rare exploding-gradient spikes
    that otherwise stall training when Fracture / Pneumothorax labels appear in
    very imbalanced mini-batches.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    loop = tqdm(loader, total=len(loader), leave=True, dynamic_ncols=True)
    for step, (images, labels, masks, _) in enumerate(loop):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE,  non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, labels, masks) / grad_accum_steps

        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        loop.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")

    return total_loss / len(loader)


# ════════════════════════════════════════════════════════════════════════════
#  VALIDATE  (loss + per-class AUC)
# ════════════════════════════════════════════════════════════════════════════

def validate(model, loader, criterion):
    """
    Validate and compute per-class AUC for the 5 competition labels.

    Returns (avg_loss, per_class_auc_dict, mean_auc).
    """
    model.eval()
    total_loss  = 0.0
    all_preds   = []   # list of [B, 5] numpy arrays
    all_targets = []   # list of [B, 5] numpy arrays
    all_masks   = []   # list of [B, 5] numpy arrays

    loop = tqdm(loader, total=len(loader), leave=True,
                desc="  Validating", dynamic_ncols=True)
    with torch.no_grad():
        for images, labels, masks, _ in loop:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE,  non_blocking=True)
            outputs = model(images)
            loss    = criterion(outputs, labels, masks)
            total_loss  += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # [N, 5]
    targets = np.concatenate(all_targets, axis=0)   # [N, 5]
    masks   = np.concatenate(all_masks,   axis=0)   # [N, 5]

    auc_per_class = {}
    valid_aucs    = []
    for i, label in enumerate(COMPETITION_LABELS):
        # Only evaluate on samples where this label is annotated
        valid_idx = masks[:, i] == 1.0
        if valid_idx.sum() < 2 or len(np.unique(targets[valid_idx, i])) < 2:
            auc_per_class[label] = float('nan')
        else:
            try:
                auc = roc_auc_score(targets[valid_idx, i], preds[valid_idx, i])
                auc_per_class[label] = auc
                valid_aucs.append(auc)
            except Exception:
                auc_per_class[label] = float('nan')

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return total_loss / len(loader), auc_per_class, mean_auc


# ════════════════════════════════════════════════════════════════════════════
#  GENERIC PHASE RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_phase(
    phase_name,
    model,
    train_loader,
    val_loader,
    pos_weight,
    epochs,
    lr,
    checkpoint_path,
    best_weights_path,
    freeze_backbone=False,
    grad_accum_steps=4,
):
    """
    Generic training loop for any phase.

    freeze_backbone=True:
      Epoch 0   : Only classifier head trained (warm-up).
      Epoch 1+  : Full model trained with differential LR:
                  backbone = lr / 10,  classifier = lr.

    grad_accum_steps:
      Number of gradient accumulation steps. Effective batch = batch_size × steps.
      Default 4 → effective batch = 32 on GTX 1650.
    """
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")
    print(f"  Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Grad accum steps : {grad_accum_steps}  "
          f"(effective batch = {train_loader.batch_size * grad_accum_steps})")

    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight, label_smoothing=0.1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    start_epoch, best_mean_auc = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler
    )

    # ── Training history ──────────────────────────────────────────────
    history_path = os.path.join('results', 'training_history.json')
    os.makedirs('results', exist_ok=True)
    if os.path.exists(history_path):
        with open(history_path) as _hf:
            history = json.load(_hf)
    else:
        history = {'epochs': [], 'train_loss': [], 'val_loss': [],
                   'mean_auc': [], 'per_class_auc': [], 'lr': [],
                   'phase': []}

    for epoch in range(start_epoch, epochs):
        # Unfreeze backbone after warm-up epoch in fine-tuning phases
        if freeze_backbone and epoch == 1:
            print("  [Fine-tune] Unfreezing backbone — differential LR "
                  f"(backbone={lr/10:.1e}, head={lr:.1e})")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam([
                {'params': model.features.parameters(),   'lr': lr / 10},
                {'params': model.classifier.parameters(), 'lr': lr},
            ], weight_decay=1e-5)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 1, eta_min=lr / 100)

        print(f"\n  Epoch {epoch + 1}/{epochs}")
        avg_train = train_one_epoch(
            model, train_loader, criterion, optimizer, grad_accum_steps
        )
        avg_val, auc_per_class, mean_auc = validate(model, val_loader, criterion)
        scheduler.step()

        current_lr = optimizer.param_groups[-1]['lr']
        print(f"  Train Loss : {avg_train:.4f}")
        print(f"  Val Loss   : {avg_val:.4f}  |  Mean AUC: {mean_auc:.4f}  |  LR: {current_lr:.2e}")
        print("  Per-class AUC:")
        for label, auc in auc_per_class.items():
            flag = " ← PRIMARY" if label in ("Edema", "Pleural Effusion") else ""
            print(f"    {label:20s}: {auc:.4f}{flag}" if not np.isnan(auc) else
                  f"    {label:20s}: n/a")

        # ── Record epoch history ─────────────────────────────────
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(round(avg_train, 6))
        history['val_loss'].append(round(avg_val, 6))
        history['mean_auc'].append(round(mean_auc, 6))
        history['per_class_auc'].append(
            {k: round(v, 6) if not np.isnan(v) else None
             for k, v in auc_per_class.items()}
        )
        history['lr'].append(round(current_lr, 8))
        history['phase'].append(phase_name)
        with open(history_path, 'w') as _hf:
            json.dump(history, _hf, indent=2)
        print(f"  [History] Saved training history → {history_path}")

        # Save best model by mean AUC (primary metric)
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            torch.save(model.state_dict(), best_weights_path)
            print(f"  ✓ New best mean AUC {best_mean_auc:.4f} → saved to {best_weights_path}")

        # Always save latest checkpoint AFTER updating best_mean_auc
        # (so resume uses the correct best value and doesn't overwrite
        # superior weights with an inferior epoch)
        save_checkpoint({
            'epoch':         epoch,
            'model_state':   model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_mean_auc': best_mean_auc,
            'auc_per_class': auc_per_class,
        }, checkpoint_path)

    print(f"\n  {phase_name} complete.  Best mean AUC: {best_mean_auc:.4f}")
    return model


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Pre-training on CheXpert
# ════════════════════════════════════════════════════════════════════════════

def phase1_pretrain(
    chexpert_train_csv = 'train.csv',
    chexpert_val_csv   = 'valid.csv',
    chexpert_root      = '.',
    nih_csv            = None,
    nih_img_dir        = None,
    mimic_chexpert_csv = None,
    mimic_records_csv  = None,
    mimic_img_root     = None,
    frontal_only       = True,
    u_label_soft       = True,       # MLMIP soft uncertainty labels — enabled by default
    batch_size         = 8,          # GTX 1650 safe with AMP
    epochs             = 21,         # run one extra epoch beyond 20
    lr                 = 1e-4,
    num_workers        = 4,
    grad_accum_steps   = 4,          # effective batch = 32
):
    train_loader, val_loader, pos_weight = get_phase1_dataloaders(
        chexpert_train_csv = chexpert_train_csv,
        chexpert_val_csv   = chexpert_val_csv,
        chexpert_root      = chexpert_root,
        nih_csv            = nih_csv,
        nih_img_dir        = nih_img_dir,
        mimic_chexpert_csv = mimic_chexpert_csv,
        mimic_records_csv  = mimic_records_csv,
        mimic_img_root     = mimic_img_root,
        frontal_only       = frontal_only,
        u_label_soft       = u_label_soft,
        batch_size         = batch_size,
        num_workers        = num_workers,
    )
    model = get_densenet121_model(pretrained=True).to(DEVICE)
    return run_phase(
        phase_name        = "Phase 1 — Pre-training (CheXpert 7-label)",
        model             = model,
        train_loader      = train_loader,
        val_loader        = val_loader,
        pos_weight        = pos_weight,
        epochs            = epochs,
        lr                = lr,
        checkpoint_path   = "checkpoint_phase1.pth",
        best_weights_path = "best_densenet121_phase1.pth",
        freeze_backbone   = False,
        grad_accum_steps  = grad_accum_steps,
    )


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Fine-tuning on RSNA
# ════════════════════════════════════════════════════════════════════════════

def phase2_finetune(
    rsna_csv,
    rsna_img_dir,
    phase1_weights    = 'best_densenet121_phase1.pth',
    batch_size        = 8,
    epochs            = 5,
    lr                = 1e-4,
    num_workers       = 4,
    grad_accum_steps  = 4,
):
    train_loader, val_loader, pos_weight = get_phase2_dataloaders(
        rsna_csv    = rsna_csv,
        rsna_img_dir = rsna_img_dir,
        batch_size  = batch_size,
        num_workers = num_workers,
    )
    model = get_densenet121_model(pretrained=False).to(DEVICE)
    if os.path.exists(phase1_weights):
        model.load_state_dict(
            torch.load(phase1_weights, map_location=DEVICE, weights_only=True)
        )
        print(f"  [Phase2] Loaded Phase 1 weights from {phase1_weights}")
    return run_phase(
        phase_name        = "Phase 2 — Fine-tuning (RSNA Pneumonia)",
        model             = model,
        train_loader      = train_loader,
        val_loader        = val_loader,
        pos_weight        = pos_weight,
        epochs            = epochs,
        lr                = lr,
        checkpoint_path   = "checkpoint_phase2.pth",
        best_weights_path = "best_densenet121_phase2.pth",
        freeze_backbone   = True,
        grad_accum_steps  = grad_accum_steps,
    )


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — Final tuning on Kaggle Binary
# ════════════════════════════════════════════════════════════════════════════

def phase3_final_tune(
    kaggle_train_dir,
    kaggle_val_dir    = None,
    phase2_weights    = 'best_densenet121_phase2.pth',
    batch_size        = 8,
    epochs            = 3,
    lr                = 5e-5,
    num_workers       = 4,
    grad_accum_steps  = 4,
):
    # Fall back to Phase 1 weights if Phase 2 doesn't exist
    weights_path = phase2_weights
    if not os.path.exists(weights_path):
        weights_path = 'best_densenet121_phase1.pth'

    train_loader, val_loader, pos_weight = get_phase3_dataloaders(
        kaggle_train_dir = kaggle_train_dir,
        kaggle_val_dir   = kaggle_val_dir,
        batch_size       = batch_size,
        num_workers      = num_workers,
    )
    model = get_densenet121_model(pretrained=False).to(DEVICE)
    if os.path.exists(weights_path):
        model.load_state_dict(
            torch.load(weights_path, map_location=DEVICE, weights_only=True)
        )
        print(f"  [Phase3] Loaded weights from {weights_path}")
    return run_phase(
        phase_name        = "Phase 3 — Final Tuning (Kaggle Binary)",
        model             = model,
        train_loader      = train_loader,
        val_loader        = val_loader,
        pos_weight        = pos_weight,
        epochs            = epochs,
        lr                = lr,
        checkpoint_path   = "checkpoint_phase3.pth",
        best_weights_path = "best_densenet121.pth",   # ← final production weights
        freeze_backbone   = True,
        grad_accum_steps  = grad_accum_steps,
    )


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT  — local GTX 1650 run (Phase 1 only, CheXpert)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Enforce GPU training for consistent high-AUC runs.
    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA GPU not available. Training is configured to run on GPU only. "
            "Enable CUDA and retry."
        )
    print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")

    # Hardware-safe defaults for GTX 1650 (4 GB VRAM)
    phase1_pretrain(
        chexpert_train_csv = 'train.csv',
        chexpert_val_csv   = 'valid.csv',
        chexpert_root      = '.',
        frontal_only       = True,
        batch_size         = 8,
        epochs             = 21,
        lr                 = 1e-4,
        num_workers        = 4,
        grad_accum_steps   = 4,
    )
