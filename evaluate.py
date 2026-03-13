"""
evaluate.py — Final evaluation on the official CheXpert validation set.

The CheXpert valid.csv contains 234 chest X-rays labelled by a consensus
of 3 radiologists — this is the only gold-standard set in the dataset.
All AUC numbers reported in publications use this set.

Usage
-----
  python evaluate.py                            # uses best_densenet121.pth
  python evaluate.py --weights best_densenet121_phase1.pth
  python evaluate.py --weights best_densenet121.pth --save_dir results/

Outputs (saved to --save_dir, default: results/):
  roc_curves.png         — ROC curve for each of the 5 competition labels
  precision_recall.png   — Precision-recall curves
  confusion_matrices.png — Confusion matrix per label (threshold 0.5)
  auc_report.txt         — Plain-text AUC summary
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # headless-safe backend
import matplotlib.pyplot as plt
import json
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score,
)
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

from model   import get_densenet121_model, COMPETITION_LABELS
from dataset import get_val_transform, U_POLICY, ChexpertDataset
from torch.utils.data import DataLoader


# ════════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION ROUTINE
# ════════════════════════════════════════════════════════════════════════════

def evaluate(
    weights_path  = 'best_densenet121.pth',
    val_csv       = 'valid.csv',
    data_root     = '.',
    save_dir      = 'results',
    batch_size    = 16,
    num_workers   = 4,
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
        print(f"WARNING: weights file {weights_path!r} not found — using random init.")
    model.eval()

    # ── Validation dataset ────────────────────────────────────────────────
    val_ds = ChexpertDataset(val_csv, data_root,
                             transform=get_val_transform(), frontal_only=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"Evaluating on {len(val_ds)} samples from {val_csv}")

    # ── Inference ─────────────────────────────────────────────────────────
    all_preds   = []
    all_targets = []

    # NOTE: AMP (fp16) intentionally disabled — DenseNet121's dense
    # concatenation paths overflow fp16 range, producing NaN.
    # Model was trained in fp32 (see train.py).
    with torch.no_grad():
        for images, labels, _masks, _ in tqdm(val_loader, desc="Inference"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # [N, 5]
    targets = np.concatenate(all_targets, axis=0)   # [N, 5]

    # ── Per-class AUC ─────────────────────────────────────────────────────
    auc_scores = {}
    print("\n" + "="*55)
    print("  CheXpert Competition Labels — AUC on Official Val Set")
    print("="*55)
    print(f"  {'Label':25s}  {'AUC':>6}  {'AP':>6}")
    print("-"*55)
    for i, label in enumerate(COMPETITION_LABELS):
        y_true = targets[:, i]
        y_score = preds[:, i]
        if len(np.unique(y_true)) < 2 or np.any(np.isnan(y_score)):
            auc_scores[label] = float('nan')
            reason = "single class" if len(np.unique(y_true)) < 2 else "NaN in preds"
            print(f"  {label:25s}  {'n/a':>6}  {'n/a':>6}  ({reason})")
            continue
        auc = roc_auc_score(y_true, y_score)
        ap  = average_precision_score(y_true, y_score)
        auc_scores[label] = auc
        print(f"  {label:25s}  {auc:.4f}  {ap:.4f}")

    valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
    mean_auc   = np.mean(valid_aucs) if valid_aucs else 0.0
    print("-"*55)
    print(f"  {'Mean AUC (5 labels)':25s}  {mean_auc:.4f}")
    print("="*55)

    # Stanford baseline reference (Irvin et al. 2019, ensemble)
    stanford = {
        "Atelectasis": 0.858, "Cardiomegaly": 0.832,
        "Consolidation": 0.899, "Edema": 0.924, "Pleural Effusion": 0.968,
    }
    print("\n  Comparison to Stanford DenseNet121 Ensemble (paper):")
    print(f"  {'Label':25s}  {'Ours':>6}  {'Stanford':>8}  {'Gap':>6}")
    print("-"*55)
    for label in COMPETITION_LABELS:
        our   = auc_scores.get(label, float('nan'))
        ref   = stanford.get(label, float('nan'))
        gap   = our - ref if not (np.isnan(our) or np.isnan(ref)) else float('nan')
        gap_s = f"{gap:+.4f}" if not np.isnan(gap) else "  n/a"
        our_s = f"{our:.4f}"  if not np.isnan(our) else "  n/a"
        print(f"  {label:25s}  {our_s:>6}  {ref:>8.4f}  {gap_s:>6}")
    print("="*55)

    # ── Per-class Accuracy, F1, Precision, Recall (threshold 0.5) ───────
    threshold = 0.5
    metrics_per_class = {}
    for i, label in enumerate(COMPETITION_LABELS):
        y_true = targets[:, i]
        y_score = preds[:, i]
        has_nan = np.any(np.isnan(y_score))
        single_class = len(np.unique(y_true)) < 2
        # Replace NaN predictions with 0.5 for threshold-based metrics
        y_score_clean = np.nan_to_num(y_score, nan=0.5)
        y_pred = (y_score_clean >= threshold).astype(int)
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        if not has_nan and not single_class:
            ap = average_precision_score(y_true, y_score)
        else:
            ap = 0.0
        metrics_per_class[label] = {
            'auc': float(auc_scores.get(label, 0.0)) if not np.isnan(auc_scores.get(label, float('nan'))) else None,
            'accuracy': float(acc),
            'f1_score': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'avg_precision': float(ap),
        }

    # Overall metrics (macro average across 5 labels)
    all_y_true = targets.flatten()
    all_y_pred = (np.nan_to_num(preds.flatten(), nan=0.5) >= threshold).astype(int)
    overall_metrics = {
        'mean_auc': float(mean_auc),
        'accuracy': float(accuracy_score(all_y_true, all_y_pred)),
        'f1_score': float(f1_score(all_y_true, all_y_pred, zero_division=0)),
        'precision': float(precision_score(all_y_true, all_y_pred, zero_division=0)),
        'recall': float(recall_score(all_y_true, all_y_pred, zero_division=0)),
    }

    # ── Save JSON metrics for frontend ────────────────────────────────────
    json_metrics = {
        'model_weights': weights_path,
        'val_csv': val_csv,
        'num_samples': len(val_ds),
        'threshold': threshold,
        'overall': overall_metrics,
        'per_class': metrics_per_class,
    }
    json_path = os.path.join(save_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"\n  JSON metrics saved → {json_path}")

    # ── Save text report ──────────────────────────────────────────────────
    report_path = os.path.join(save_dir, 'auc_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model weights : {weights_path}\n")
        f.write(f"Val set       : {val_csv}  ({len(val_ds)} samples)\n\n")
        f.write(f"{'Label':25s}  {'AUC':>6}  {'Acc':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}\n")
        f.write("-"*75 + "\n")
        for label in COMPETITION_LABELS:
            m = metrics_per_class[label]
            auc_s = f"{m['auc']:.4f}" if m['auc'] is not None else "  n/a"
            f.write(f"{label:25s}  {auc_s:>6}  {m['accuracy']:.4f}  "
                    f"{m['f1_score']:.4f}  {m['precision']:.4f}  {m['recall']:.4f}\n")
        f.write("-"*75 + "\n")
        f.write(f"\n{'Mean AUC':25s}  {mean_auc:.4f}\n")
        o = overall_metrics
        f.write(f"{'Overall Accuracy':25s}  {o['accuracy']:.4f}\n")
        f.write(f"{'Overall F1 Score':25s}  {o['f1_score']:.4f}\n")
        f.write(f"{'Overall Precision':25s}  {o['precision']:.4f}\n")
        f.write(f"{'Overall Recall':25s}  {o['recall']:.4f}\n")
    print(f"\n  Report saved → {report_path}")

    # ── ROC Curves ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle("ROC Curves — CheXpert Official Validation Set", fontsize=13, y=1.02)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']

    for i, (label, ax) in enumerate(zip(COMPETITION_LABELS, axes)):
        y_true  = targets[:, i]
        y_score = preds[:, i]
        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, 'single class', ha='center', va='center',
                    transform=ax.transAxes)
        elif np.any(np.isnan(y_score)):
            ax.text(0.5, 0.5, 'NaN in preds', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = auc_scores[label]
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.fill_between(fpr, tpr, alpha=0.08, color=colors[i])
            ax.legend(loc='lower right', fontsize=10)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(label, fontsize=10); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC curves saved → {roc_path}")

    # ── Precision-Recall Curves ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle("Precision-Recall Curves — CheXpert Official Validation Set",
                 fontsize=13, y=1.02)

    for i, (label, ax) in enumerate(zip(COMPETITION_LABELS, axes)):
        y_true  = targets[:, i]
        y_score = preds[:, i]
        if len(np.unique(y_true)) < 2:
            ax.text(0.5, 0.5, 'single class', ha='center', va='center',
                    transform=ax.transAxes)
        elif np.any(np.isnan(y_score)):
            ax.text(0.5, 0.5, 'NaN in preds', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ax.plot(rec, prec, color=colors[i], lw=2, label=f"AP = {ap:.3f}")
            base = y_true.mean()
            ax.axhline(y=base, color='k', linestyle='--', lw=1,
                       alpha=0.5, label=f"Baseline = {base:.3f}")
            ax.fill_between(rec, prec, alpha=0.08, color=colors[i])
            ax.legend(loc='upper right', fontsize=9)
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(label, fontsize=10); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pr_path = os.path.join(save_dir, 'precision_recall.png')
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PR curves saved → {pr_path}")

    # ── Confusion Matrices (threshold 0.5) ────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle("Confusion Matrices @ threshold 0.5", fontsize=13, y=1.02)

    for i, (label, ax) in enumerate(zip(COMPETITION_LABELS, axes)):
        y_true = targets[:, i]
        y_score_clean = np.nan_to_num(preds[:, i], nan=0.5)
        y_pred = (y_score_clean >= 0.5).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        disp   = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=['Neg', 'Pos'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(label, fontsize=10)

    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrices saved → {cm_path}")

    return auc_scores, mean_auc


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CheXpert evaluation script")
    parser.add_argument('--weights',   default='best_densenet121.pth',
                        help="Path to model weights .pth file")
    parser.add_argument('--val_csv',   default='valid.csv',
                        help="Path to CheXpert valid.csv")
    parser.add_argument('--data_root', default='.',
                        help="Root directory containing CheXpert images")
    parser.add_argument('--save_dir',  default='results',
                        help="Directory to save output plots and reports")
    parser.add_argument('--batch',     type=int, default=16,
                        help="Batch size for inference")
    args = parser.parse_args()

    # Try best final weights first, fall back to phase1
    weights = args.weights
    if not os.path.exists(weights):
        for fallback in ['best_densenet121_phase1.pth',
                         'best_densenet121_phase2.pth',
                         'best_densenet121_phase3.pth']:
            if os.path.exists(fallback):
                print(f"  {weights!r} not found — using {fallback!r} instead.")
                weights = fallback
                break

    evaluate(
        weights_path = weights,
        val_csv      = args.val_csv,
        data_root    = args.data_root,
        save_dir     = args.save_dir,
        batch_size   = args.batch,
        num_workers  = 4,
    )
