"""
evaluate_phase3.py — Evaluate the DenseNet121 (CheXpert) model on the
binary NORMAL/PNEUMONIA chest X-ray dataset from phase_3.

Strategy:
  The model outputs probabilities for 5 CheXpert pathologies.
  We derive a single "pneumonia score" = max(prob across 5 labels).
  If the max probability > threshold → PNEUMONIA, else NORMAL.

Usage:
  python evaluate_phase3.py
  python evaluate_phase3.py --data_dir phase_3/chest_xray/test
  python evaluate_phase3.py --weights best_densenet121_phase1.pth --save_dir results

Outputs (saved to --save_dir, default: results/):
  metrics.json           — Full metrics for the Streamlit dashboard
  roc_curves.png         — Binary ROC curve (Pneumonia vs Normal)
  precision_recall.png   — Precision-Recall curve
  confusion_matrices.png — Single confusion matrix
  auc_report.txt         — Plain-text summary
"""

import os
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

from model import get_densenet121_model, COMPETITION_LABELS


def get_eval_transform():
    """Same preprocessing as CheXpert validation."""
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def evaluate_phase3(
    weights_path='best_densenet121_phase1.pth',
    data_dir='phase_3/chest_xray/test',
    save_dir='results',
    batch_size=16,
    threshold=0.5,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────
    model = get_densenet121_model(pretrained=False).to(device)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded weights: {weights_path}")
    else:
        print(f"WARNING: {weights_path!r} not found — using random init.")
    model.eval()

    # ── Dataset (ImageFolder: NORMAL=0, PNEUMONIA=1) ─────────────────────
    dataset = ImageFolder(data_dir, transform=get_eval_transform())
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=True)
    class_names = dataset.classes  # ['NORMAL', 'PNEUMONIA']
    print(f"Classes: {class_names}")
    print(f"Evaluating on {len(dataset)} images from {data_dir}")

    # Ensure PNEUMONIA = positive class (label 1)
    pneumonia_idx = class_names.index('PNEUMONIA') if 'PNEUMONIA' in class_names else 1

    # ── Inference ─────────────────────────────────────────────────────────
    all_scores_per_label = []  # [N, 5] — per-pathology probabilities
    all_targets = []           # [N]    — 0=NORMAL, 1=PNEUMONIA

    use_amp = device.type == 'cuda'
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inference"):
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
            else:
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            # Replace NaN with 0
            probs = np.nan_to_num(probs, nan=0.0)
            all_scores_per_label.append(probs)
            all_targets.append(labels.numpy())

    scores_5   = np.concatenate(all_scores_per_label, axis=0)  # [N, 5]
    targets    = np.concatenate(all_targets, axis=0)            # [N]

    # Binary pneumonia score = max over the 5 pathology probabilities
    pneumonia_score = scores_5.max(axis=1)  # [N]
    binary_preds    = (pneumonia_score >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────
    acc  = accuracy_score(targets, binary_preds)
    f1   = f1_score(targets, binary_preds, zero_division=0)
    prec = precision_score(targets, binary_preds, zero_division=0)
    rec  = recall_score(targets, binary_preds, zero_division=0)

    if len(np.unique(targets)) >= 2:
        auc = roc_auc_score(targets, pneumonia_score)
        ap  = average_precision_score(targets, pneumonia_score)
    else:
        auc = 0.0
        ap  = 0.0

    print("\n" + "=" * 55)
    print("  Binary Pneumonia Classification — Phase 3 Test Set")
    print("=" * 55)
    print(f"  {'AUC':20s}  {auc:.4f}")
    print(f"  {'Accuracy':20s}  {acc:.4f}")
    print(f"  {'F1 Score':20s}  {f1:.4f}")
    print(f"  {'Precision':20s}  {prec:.4f}")
    print(f"  {'Recall':20s}  {rec:.4f}")
    print(f"  {'Avg Precision (AP)':20s}  {ap:.4f}")
    print("=" * 55)
    print()
    print(classification_report(targets, binary_preds,
                                target_names=class_names))

    # ── Per-pathology contribution scores ─────────────────────────────────
    per_class_metrics = {}
    for i, label in enumerate(COMPETITION_LABELS):
        y_score = scores_5[:, i]
        y_pred_i = (y_score >= threshold).astype(int)
        lbl_acc  = accuracy_score(targets, y_pred_i)
        lbl_f1   = f1_score(targets, y_pred_i, zero_division=0)
        lbl_prec = precision_score(targets, y_pred_i, zero_division=0)
        lbl_rec  = recall_score(targets, y_pred_i, zero_division=0)
        if len(np.unique(targets)) >= 2 and not np.all(y_score == y_score[0]):
            lbl_auc = roc_auc_score(targets, y_score)
        else:
            lbl_auc = None
        per_class_metrics[label] = {
            'auc': float(lbl_auc) if lbl_auc is not None else None,
            'accuracy': float(lbl_acc),
            'f1_score': float(lbl_f1),
            'precision': float(lbl_prec),
            'recall': float(lbl_rec),
        }

    # ── Save JSON metrics ─────────────────────────────────────────────────
    overall_metrics = {
        'mean_auc': float(auc),
        'accuracy': float(acc),
        'f1_score': float(f1),
        'precision': float(prec),
        'recall': float(rec),
    }
    json_metrics = {
        'model_weights': weights_path,
        'val_csv': data_dir,
        'num_samples': len(dataset),
        'threshold': threshold,
        'dataset': 'Phase 3 — Binary Pneumonia (NORMAL / PNEUMONIA)',
        'overall': overall_metrics,
        'per_class': per_class_metrics,
    }
    json_path = os.path.join(save_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"  JSON metrics saved → {json_path}")

    # ── Save text report ──────────────────────────────────────────────────
    report_path = os.path.join(save_dir, 'auc_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model weights : {weights_path}\n")
        f.write(f"Dataset       : {data_dir}  ({len(dataset)} samples)\n")
        f.write(f"Task          : Binary Pneumonia Classification\n\n")
        f.write(f"{'Metric':20s}  {'Value':>8}\n")
        f.write("-" * 35 + "\n")
        f.write(f"{'AUC':20s}  {auc:>8.4f}\n")
        f.write(f"{'Accuracy':20s}  {acc:>8.4f}\n")
        f.write(f"{'F1 Score':20s}  {f1:>8.4f}\n")
        f.write(f"{'Precision':20s}  {prec:>8.4f}\n")
        f.write(f"{'Recall':20s}  {rec:>8.4f}\n")
        f.write(f"{'Avg Precision':20s}  {ap:>8.4f}\n")
        f.write("\n" + "=" * 55 + "\n")
        f.write("Per-pathology breakdown (as pneumonia proxy):\n\n")
        f.write(f"{'Label':25s}  {'AUC':>6}  {'Acc':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}\n")
        f.write("-" * 75 + "\n")
        for label in COMPETITION_LABELS:
            m = per_class_metrics[label]
            auc_s = f"{m['auc']:.4f}" if m['auc'] is not None else "  n/a"
            f.write(f"{label:25s}  {auc_s:>6}  {m['accuracy']:.4f}  "
                    f"{m['f1_score']:.4f}  {m['precision']:.4f}  {m['recall']:.4f}\n")
    print(f"  Report saved → {report_path}")

    # ── ROC Curve ─────────────────────────────────────────────────────────
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']

    fig, axes = plt.subplots(1, min(6, 1 + len(COMPETITION_LABELS)), figsize=(22, 4))
    fig.suptitle("ROC Curves — Phase 3 Pneumonia Test Set", fontsize=13, y=1.02)

    # Combined binary ROC
    if len(np.unique(targets)) >= 2:
        fpr, tpr, _ = roc_curve(targets, pneumonia_score)
        axes[0].plot(fpr, tpr, color='#e41a1c', lw=2, label=f"AUC = {auc:.3f}")
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        axes[0].fill_between(fpr, tpr, alpha=0.08, color='#e41a1c')
        axes[0].legend(loc='lower right', fontsize=10)
    else:
        axes[0].text(0.5, 0.5, 'single class', ha='center', va='center',
                     transform=axes[0].transAxes)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title("Combined (max)", fontsize=10)
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.02]); axes[0].grid(True, alpha=0.3)

    # Per-pathology ROC
    for i, (label, ax) in enumerate(zip(COMPETITION_LABELS, axes[1:])):
        y_score = scores_5[:, i]
        if len(np.unique(targets)) >= 2 and not np.all(y_score == y_score[0]):
            fpr_l, tpr_l, _ = roc_curve(targets, y_score)
            lbl_auc = per_class_metrics[label]['auc'] or 0.0
            ax.plot(fpr_l, tpr_l, color=colors[i], lw=2, label=f"AUC = {lbl_auc:.3f}")
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.fill_between(fpr_l, tpr_l, alpha=0.08, color=colors[i])
            ax.legend(loc='lower right', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'constant preds', ha='center', va='center',
                    transform=ax.transAxes)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(label, fontsize=10)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02]); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC curves saved → {roc_path}")

    # ── Precision-Recall Curve ────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    fig.suptitle("Precision-Recall — Phase 3 Pneumonia Test Set", fontsize=13, y=1.02)
    if len(np.unique(targets)) >= 2:
        prec_c, rec_c, _ = precision_recall_curve(targets, pneumonia_score)
        ax.plot(rec_c, prec_c, color='#e41a1c', lw=2, label=f"AP = {ap:.3f}")
        base = targets.mean()
        ax.axhline(y=base, color='k', linestyle='--', lw=1, alpha=0.5,
                   label=f"Baseline = {base:.3f}")
        ax.fill_between(rec_c, prec_c, alpha=0.08, color='#e41a1c')
        ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02]); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(save_dir, 'precision_recall.png')
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PR curve saved → {pr_path}")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle("Confusion Matrix @ threshold 0.5", fontsize=13, y=1.02)
    cm = confusion_matrix(targets, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    return overall_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on Phase 3 pneumonia dataset")
    parser.add_argument('--weights',  default='best_densenet121_phase1.pth',
                        help="Path to model weights")
    parser.add_argument('--data_dir', default='phase_3/chest_xray/test',
                        help="Path to test folder with NORMAL/ and PNEUMONIA/ subfolders")
    parser.add_argument('--save_dir', default='results',
                        help="Directory to save outputs")
    parser.add_argument('--batch',    type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    # Try best weights, fall back
    weights = args.weights
    if not os.path.exists(weights):
        for fallback in ['best_densenet121.pth',
                         'best_densenet121_phase2.pth',
                         'best_densenet121_phase3.pth']:
            if os.path.exists(fallback):
                print(f"  {weights!r} not found — using {fallback!r}")
                weights = fallback
                break

    evaluate_phase3(
        weights_path=weights,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        batch_size=args.batch,
        threshold=args.threshold,
    )
