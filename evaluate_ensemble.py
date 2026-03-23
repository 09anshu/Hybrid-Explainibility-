import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChexpertDataset, get_val_transform
from model import COMPETITION_LABELS, get_densenet121_model, load_backbone_weights


def load_models(weight_paths, device):
    models = []
    for path in weight_paths:
        model = get_densenet121_model(pretrained=False).to(device)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights not found: {path}")
        model, mode = load_backbone_weights(model, path, device=device)
        model.eval()
        print(f"Loaded: {path} ({mode})")
        models.append(model)
    return models


def evaluate_ensemble(weight_paths, val_csv, data_root, save_dir, batch_size):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_models(weight_paths, device)

    ds = ChexpertDataset(val_csv, data_root, transform=get_val_transform(), frontal_only=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _masks, _paths in tqdm(loader, desc="Ensemble inference"):
            images = images.to(device, non_blocking=True)

            logits_sum = None
            for m in models:
                logits = m(images)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            logits_mean = logits_sum / float(len(models))

            probs = torch.sigmoid(logits_mean).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    print("\n" + "=" * 72)
    print("Ensemble AUC on CheXpert validation")
    print("=" * 72)
    print(f"{'Label':25s}  {'AUC':>8}  {'AP':>8}")
    print("-" * 72)

    per_class = {}
    aucs = []
    for i, label in enumerate(COMPETITION_LABELS):
        y_true = targets[:, i]
        y_prob = probs[:, i]

        if len(np.unique(y_true)) < 2:
            auc = None
            ap = 0.0
            print(f"{label:25s}  {'n/a':>8}  {'n/a':>8}")
        else:
            auc = float(roc_auc_score(y_true, y_prob))
            ap = float(average_precision_score(y_true, y_prob))
            aucs.append(auc)
            print(f"{label:25s}  {auc:8.4f}  {ap:8.4f}")

        y_pred = (y_prob >= 0.5).astype(int)
        per_class[label] = {
            "auc": auc,
            "avg_precision": ap,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    print("-" * 72)
    print(f"{'Mean AUC':25s}  {mean_auc:8.4f}")
    print("=" * 72)

    overall = {
        "mean_auc": mean_auc,
        "accuracy": float(accuracy_score(targets.flatten(), (probs.flatten() >= 0.5).astype(int))),
        "f1_score": float(f1_score(targets.flatten(), (probs.flatten() >= 0.5).astype(int), zero_division=0)),
        "precision": float(precision_score(targets.flatten(), (probs.flatten() >= 0.5).astype(int), zero_division=0)),
        "recall": float(recall_score(targets.flatten(), (probs.flatten() >= 0.5).astype(int), zero_division=0)),
    }

    payload = {
        "model_weights": weight_paths,
        "val_csv": val_csv,
        "num_samples": int(len(ds)),
        "threshold": 0.5,
        "overall": overall,
        "per_class": per_class,
    }

    json_path = os.path.join(save_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    txt_path = os.path.join(save_dir, "auc_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model weights : {', '.join(weight_paths)}\n")
        f.write(f"Val set       : {val_csv}  ({len(ds)} samples)\n\n")
        f.write(f"{'Label':25s}  {'AUC':>6}  {'Acc':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}\n")
        f.write("-" * 75 + "\n")
        for label in COMPETITION_LABELS:
            m = per_class[label]
            auc_str = f"{m['auc']:.4f}" if m["auc"] is not None else "  n/a"
            f.write(
                f"{label:25s}  {auc_str:>6}  {m['accuracy']:.4f}  {m['f1_score']:.4f}  "
                f"{m['precision']:.4f}  {m['recall']:.4f}\n"
            )
        f.write("-" * 75 + "\n")
        f.write(f"\n{'Mean AUC':25s}  {overall['mean_auc']:.4f}\n")
        f.write(f"{'Overall Accuracy':25s}  {overall['accuracy']:.4f}\n")
        f.write(f"{'Overall F1 Score':25s}  {overall['f1_score']:.4f}\n")
        f.write(f"{'Overall Precision':25s}  {overall['precision']:.4f}\n")
        f.write(f"{'Overall Recall':25s}  {overall['recall']:.4f}\n")

    print(f"Saved metrics: {json_path}")
    print(f"Saved report : {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ensemble of DenseNet121 checkpoints")
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="Paths to .pth checkpoints. Example: --weights w1.pth w2.pth",
    )
    parser.add_argument("--val-csv", default="valid.csv")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--save-dir", default="results_ensemble")
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_ensemble(
        weight_paths=args.weights,
        val_csv=args.val_csv,
        data_root=args.data_root,
        save_dir=args.save_dir,
        batch_size=args.batch,
    )
