import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm

from model import get_densenet121_model, load_backbone_weights, COMPETITION_LABELS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PTX_INDEX = COMPETITION_LABELS.index("Pneumothorax")


class SIIMPneumothoraxDataset(Dataset):
    def __init__(self, df, root_dir, tfm):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.tfm = tfm
        self.mask = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
        self.mask[PTX_INDEX] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["new_filename"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (1024, 1024), (0, 0, 0))

        if self.tfm is not None:
            img = self.tfm(img)

        y = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
        y[PTX_INDEX] = float(row["has_pneumo"])
        return img, y, self.mask.clone(), img_path


class MaskedBCE(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, masks):
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            loss_elem = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw.unsqueeze(0), reduction="none"
            )
        else:
            loss_elem = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
        masked = loss_elem * masks.to(logits.device)
        n_active = masks.to(logits.device).sum().clamp(min=1.0)
        return masked.sum() / n_active


def get_train_tfm(img_size=320):
    return transforms.Compose([
        transforms.Resize((img_size + 24, img_size + 24)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_tfm(img_size=320):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_sampler(labels):
    labels = np.asarray(labels)
    n = len(labels)
    pos = max(int((labels == 1).sum()), 1)
    neg = max(int((labels == 0).sum()), 1)
    w_pos = n / pos
    w_neg = n / neg
    weights = np.where(labels == 1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=n, replacement=True)


def evaluate_ptx_auc(model, loader, criterion):
    model.eval()
    losses, ys, ps = [], [], []
    with torch.no_grad():
        for x, y, m, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            m = m.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y, m)
            losses.append(loss.item())
            probs = torch.sigmoid(logits)[:, PTX_INDEX].cpu().numpy()
            ys.append(y[:, PTX_INDEX].cpu().numpy())
            ps.append(probs)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    return float(np.mean(losses)), auc


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA GPU not available. This training is configured for GPU.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    csv_path = "pneumo masks/siim-acr-pneumothorax/stage_1_test_images.csv"
    img_root = "pneumo masks/siim-acr-pneumothorax/png_masks"
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    weights_init = "best_densenet121_phase1_stopped_best_20260317.pth"
    ckpt_path = "checkpoint_pneumothorax_siim.pth"
    best_path = "best_densenet121_pneumothorax_siim.pth"
    history_path = out_dir / "pneumothorax_siim_history.json"

    batch_size = 16
    epochs = 8
    lr = 5e-5
    num_workers = 4

    df = pd.read_csv(csv_path)
    df = df[df["new_filename"].apply(lambda x: os.path.exists(os.path.join(img_root, x)))]

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["has_pneumo"],
    )

    train_ds = SIIMPneumothoraxDataset(train_df, img_root, get_train_tfm())
    val_ds = SIIMPneumothoraxDataset(val_df, img_root, get_val_tfm())

    sampler = build_sampler(train_df["has_pneumo"].values)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = get_densenet121_model(pretrained=False).to(DEVICE)
    if os.path.exists(weights_init):
        model, mode = load_backbone_weights(model, weights_init, device=DEVICE)
        print(f"[Init] Loaded {weights_init} ({mode})")

    pos = max(int((train_df["has_pneumo"] == 1).sum()), 1)
    neg = max(int((train_df["has_pneumo"] == 0).sum()), 1)
    pos_weight = torch.ones(len(COMPETITION_LABELS), dtype=torch.float32)
    pos_weight[PTX_INDEX] = float(neg / pos)

    criterion = MaskedBCE(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    start_epoch = 0
    best_auc = 0.0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "ptx_auc": [], "lr": []}

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        try:
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optimizer_state"])
            scheduler.load_state_dict(ck["scheduler_state"])
            start_epoch = ck.get("epoch", -1) + 1
            best_auc = ck.get("best_auc", 0.0)
            history = ck.get("history", history)
            print(f"[Resume] epoch={start_epoch}, best_auc={best_auc:.4f}")
        except Exception as e:
            print(f"[Resume] Failed to fully resume checkpoint: {e}")

    print("=" * 64)
    print("SIIM Pneumothorax Fine-Tuning")
    print("=" * 64)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print(f"Class balance train (neg/pos): {neg}/{pos}  pos_weight={pos_weight[PTX_INDEX]:.2f}")

    for ep in range(start_epoch, epochs):
        model.train()
        train_losses = []
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {ep+1}/{epochs}", dynamic_ncols=True)

        for x, y, m, _ in loop:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            m = m.to(DEVICE, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y, m)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        tr_loss = float(np.mean(train_losses))
        va_loss, va_auc = evaluate_ptx_auc(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(ep + 1)
        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(va_loss, 6))
        history["ptx_auc"].append(round(va_auc, 6) if not np.isnan(va_auc) else None)
        history["lr"].append(round(current_lr, 8))

        print(f"Epoch {ep+1}/{epochs} | train={tr_loss:.4f} val={va_loss:.4f} ptx_auc={va_auc:.4f} lr={current_lr:.2e}")

        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), best_path)
            print(f"  [Best] Pneumothorax AUC improved to {best_auc:.4f} -> {best_path}")

        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_auc": best_auc,
            "history": history,
        }, ckpt_path)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print("=" * 64)
    print(f"Done. Best SIIM Pneumothorax AUC: {best_auc:.4f}")
    print(f"Best weights: {best_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
