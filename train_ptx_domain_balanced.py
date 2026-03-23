import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from model import COMPETITION_LABELS, get_densenet121_model, load_backbone_weights


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PTX_INDEX = COMPETITION_LABELS.index("Pneumothorax")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_train_transform(img_size: int = 320) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size + 24, img_size + 24)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_val_transform(img_size: int = 320) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def safe_open_rgb(path: str, size: int = 320) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (size, size), (0, 0, 0))


class CheXpertPTXDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, img_size: int = 320):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        self.mask = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
        self.mask[PTX_INDEX] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = safe_open_rgb(row["abs_path"], self.img_size)
        if self.transform is not None:
            image = self.transform(image)

        y = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
        y[PTX_INDEX] = float(row["ptx_label"])
        return image, y, self.mask.clone(), row["abs_path"], "chexpert"


class SIIMPTXDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, img_size: int = 320):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        self.mask = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
        self.mask[PTX_INDEX] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = safe_open_rgb(row["abs_path"], self.img_size)
        if self.transform is not None:
            image = self.transform(image)

        y = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
        y[PTX_INDEX] = float(row["ptx_label"])
        return image, y, self.mask.clone(), row["abs_path"], "siim"


class MaskedBCE(nn.Module):
    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        elem = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device).unsqueeze(0),
            reduction="none",
        )
        masked = elem * masks.to(logits.device)
        n_active = masks.to(logits.device).sum().clamp(min=1.0)
        return masked.sum() / n_active


def build_chexpert_df(csv_path: str, data_root: str, frontal_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Path"] = df["Path"].str.replace(r"^CheXpert-v1\.0(?:-small)?/", "", regex=True)

    if frontal_only and "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

    if "Pneumothorax" not in df.columns:
        raise ValueError("Pneumothorax column not found in CheXpert CSV.")

    # Use U-Zeros for PTX to avoid uncertain-positive inflation.
    ptx = df["Pneumothorax"].fillna(0.0).replace(-1.0, 0.0).astype(np.float32)
    df = df.copy()
    df["ptx_label"] = ptx
    df["abs_path"] = df["Path"].apply(lambda p: os.path.join(data_root, p))
    df = df[df["abs_path"].apply(os.path.exists)].reset_index(drop=True)
    return df[["abs_path", "ptx_label"]]


def build_siim_df(csv_path: str, image_root: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"new_filename", "has_pneumo"}
    if not required.issubset(set(df.columns)):
        raise ValueError("SIIM CSV must contain columns: new_filename, has_pneumo")

    df = df.copy()
    df["ptx_label"] = df["has_pneumo"].astype(np.float32)
    df["abs_path"] = df["new_filename"].apply(lambda p: os.path.join(image_root, p))
    df = df[df["abs_path"].apply(os.path.exists)].reset_index(drop=True)
    return df[["abs_path", "ptx_label"]]


def maybe_downsample(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df
    per_class = max_samples // 2
    parts = []
    for cls in [0.0, 1.0]:
        g = df[df["ptx_label"] == cls]
        if len(g) == 0:
            continue
        parts.append(g.sample(n=min(len(g), per_class), random_state=seed))

    out = pd.concat(parts, ignore_index=True) if parts else df.sample(n=min(len(df), max_samples), random_state=seed)
    return out.reset_index(drop=True)


def build_domain_balanced_sampler(domain_tags: np.ndarray, labels: np.ndarray) -> WeightedRandomSampler:
    domain_tags = np.asarray(domain_tags)
    labels = np.asarray(labels)

    n = len(labels)
    domains = np.unique(domain_tags)
    domain_w = {d: 1.0 / max((domain_tags == d).sum(), 1) for d in domains}

    pos_count = max(int((labels == 1).sum()), 1)
    neg_count = max(int((labels == 0).sum()), 1)
    class_w = {1.0: n / pos_count, 0.0: n / neg_count}

    weights = np.array([
        domain_w[domain_tags[i]] * class_w[float(labels[i])]
        for i in range(n)
    ], dtype=np.float64)
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=n, replacement=True)


def evaluate_single_domain(model, loader, criterion):
    model.eval()
    losses = []
    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for x, y, m, _p, _d in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            m = m.to(DEVICE, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y, m)
            losses.append(loss.item())

            probs = torch.sigmoid(logits)[:, PTX_INDEX].cpu().numpy()
            y_true = y[:, PTX_INDEX].cpu().numpy()
            y_prob_all.append(probs)
            y_true_all.append(y_true)

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    return float(np.mean(losses)), auc


def train(args):
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA GPU not available. This training strategy requires GPU.")

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    chex_train = build_chexpert_df(args.chexpert_train_csv, args.data_root, frontal_only=True)
    chex_val = build_chexpert_df(args.chexpert_val_csv, args.data_root, frontal_only=False)
    siim_all = build_siim_df(args.siim_csv, args.siim_image_root)

    chex_train = maybe_downsample(chex_train, args.max_chexpert_train_samples, args.seed)
    siim_all = maybe_downsample(siim_all, args.max_siim_samples, args.seed)

    siim_train, siim_val = train_test_split(
        siim_all,
        test_size=args.siim_val_split,
        random_state=args.seed,
        stratify=siim_all["ptx_label"] if len(siim_all["ptx_label"].unique()) > 1 else None,
    )

    ds_chex_train = CheXpertPTXDataset(chex_train, make_train_transform(args.img_size), img_size=args.img_size)
    ds_siim_train = SIIMPTXDataset(siim_train, make_train_transform(args.img_size), img_size=args.img_size)
    ds_chex_val = CheXpertPTXDataset(chex_val, make_val_transform(args.img_size), img_size=args.img_size)
    ds_siim_val = SIIMPTXDataset(siim_val, make_val_transform(args.img_size), img_size=args.img_size)

    train_concat = ConcatDataset([ds_chex_train, ds_siim_train])
    domain_tags = np.array(["chexpert"] * len(ds_chex_train) + ["siim"] * len(ds_siim_train))
    ptx_labels = np.concatenate([
        chex_train["ptx_label"].values.astype(np.float32),
        siim_train["ptx_label"].values.astype(np.float32),
    ])

    sampler = build_domain_balanced_sampler(domain_tags, ptx_labels)

    train_loader = DataLoader(
        train_concat,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    chex_val_loader = DataLoader(
        ds_chex_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    siim_val_loader = DataLoader(
        ds_siim_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = get_densenet121_model(pretrained=False).to(DEVICE)
    if os.path.exists(args.init_weights):
        model, mode = load_backbone_weights(model, args.init_weights, device=DEVICE)
        print(f"[Init] Loaded {args.init_weights} ({mode})")

    pos = max(int((ptx_labels == 1).sum()), 1)
    neg = max(int((ptx_labels == 0).sum()), 1)
    pos_weight = torch.ones(len(COMPETITION_LABELS), dtype=torch.float32)
    pos_weight[PTX_INDEX] = float(neg / pos)

    criterion = MaskedBCE(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    checkpoint_path = os.path.join(args.save_dir, "checkpoint_ptx_domain_balanced.pth")
    best_path = os.path.join(args.save_dir, "best_densenet121_ptx_domain_balanced.pth")
    history_path = os.path.join(args.save_dir, "ptx_domain_balanced_history.json")

    history = {
        "epoch": [],
        "train_loss": [],
        "chex_val_loss": [],
        "chex_val_auc": [],
        "siim_val_loss": [],
        "siim_val_auc": [],
        "blended_score": [],
        "lr": [],
    }

    start_epoch = 0
    best_score = -1.0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        try:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_score = float(ckpt.get("best_score", -1.0))
            history = ckpt.get("history", history)
            print(f"[Resume] epoch={start_epoch}, best_score={best_score:.4f}")
        except Exception as ex:
            print(f"[Resume] Failed to load full state, starting fresh: {ex}")

    print("=" * 72)
    print("Domain-Balanced PTX Training (SIIM + CheXpert)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train counts: CheXpert={len(ds_chex_train)} | SIIM={len(ds_siim_train)}")
    print(f"Val counts:   CheXpert={len(ds_chex_val)} | SIIM={len(ds_siim_val)}")
    print(f"PTX pos_weight={pos_weight[PTX_INDEX]:.3f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        for x, y, m, _p, _d in loop:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            m = m.to(DEVICE, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y, m)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = float(np.mean(losses))
        chex_loss, chex_auc = evaluate_single_domain(model, chex_val_loader, criterion)
        siim_loss, siim_auc = evaluate_single_domain(model, siim_val_loader, criterion)

        # Prioritize CheXpert PTX ranking, with SIIM as regularizer signal.
        chex_part = 0.0 if np.isnan(chex_auc) else chex_auc
        siim_part = 0.0 if np.isnan(siim_auc) else siim_auc
        blended_score = 0.7 * chex_part + 0.3 * siim_part

        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(round(train_loss, 6))
        history["chex_val_loss"].append(round(chex_loss, 6))
        history["chex_val_auc"].append(None if np.isnan(chex_auc) else round(chex_auc, 6))
        history["siim_val_loss"].append(round(siim_loss, 6))
        history["siim_val_auc"].append(None if np.isnan(siim_auc) else round(siim_auc, 6))
        history["blended_score"].append(round(blended_score, 6))
        history["lr"].append(round(lr_now, 8))

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch {epoch+1}/{args.epochs} | train={train_loss:.4f} "
            f"chex_auc={chex_auc:.4f} siim_auc={siim_auc:.4f} "
            f"blend={blended_score:.4f} lr={lr_now:.2e}"
        )

        if blended_score > best_score:
            best_score = blended_score
            torch.save(model.state_dict(), best_path)
            print(f"[Best] blended_score={best_score:.4f} saved -> {best_path}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
                "history": history,
            },
            checkpoint_path,
        )

    print("=" * 72)
    print(f"Training complete. Best blended score: {best_score:.4f}")
    print(f"Best model: {best_path}")
    print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="Domain-balanced PTX training with SIIM + CheXpert")
    parser.add_argument("--chexpert-train-csv", default="train.csv")
    parser.add_argument("--chexpert-val-csv", default="valid.csv")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--siim-csv", default="pneumo masks/siim-acr-pneumothorax/stage_1_test_images.csv")
    parser.add_argument("--siim-image-root", default="pneumo masks/siim-acr-pneumothorax/png_masks")
    parser.add_argument("--init-weights", default="best_densenet121_phase1.pth")
    parser.add_argument("--save-dir", default="results_domain_balanced_ptx")

    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-chexpert-train-samples", type=int, default=50000)
    parser.add_argument("--max-siim-samples", type=int, default=0)
    parser.add_argument("--siim-val-split", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
