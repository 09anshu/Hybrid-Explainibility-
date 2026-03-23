"""
dataset.py — Multi-dataset support for 3-phase DenseNet121 training.

Phase 1  Pre-training   : CheXpert  +  NIH ChestX-ray14  (+  MIMIC-CXR optional)
Phase 2  Fine-tuning    : RSNA Pneumonia Detection Challenge
Phase 3  Final-tuning   : Kaggle Binary Chest X-Ray (NORMAL vs PNEUMONIA)

5 CheXpert competition labels (Irvin et al. 2019), model output index order:
  0  Atelectasis      — U-Ones   (-1 → 1 :  uncertain = positive)
  1  Cardiomegaly     — U-Ones   (-1 → 1)
  2  Consolidation    — U-Zeros  (-1 → 0 :  uncertain = negative)
  3  Edema            — U-Ones   (-1 → 1)
  4  Pleural Effusion — U-Ones   (-1 → 1)

Uncertainty policy reference: Table 4, Irvin et al. AAAI 2019.

Multi-dataset masked loss:
  Each sample carries a `mask` tensor (same shape as labels).
  mask[i] = 1  → label[i] is known, contribute to loss.
  mask[i] = 0  → label[i] is unknown / not annotated, skip in loss.
  CheXpert  → mask = all-ones (all 5 labels annotated).
  NIH       → mask = 1 only for labels present in NIH findings vocabulary.
  RSNA      → mask = [1,0,1,0,0]  (Atelectasis + Consolidation proxy for pneumonia).
  Kaggle    → mask = [1,0,1,0,0]  (same proxy).
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms

from model import COMPETITION_LABELS  # 7-label: 5 CheXpert + Pneumothorax + Fracture

# ════════════════════════════════════════════════════════════════════════════
#  UNCERTAINTY POLICIES  (Irvin et al. 2019, Table 4 + extensions)
# ════════════════════════════════════════════════════════════════════════════
# For each label: how to handle value -1 (uncertain)?
#   1  → U-Ones  (uncertain treated as positive)
#   0  → U-Zeros (uncertain treated as negative)
U_POLICY = {
    "Atelectasis":      1,   # U-Ones  — strong positive signal in uncertain cases
    "Cardiomegaly":     1,   # U-Ones
    "Consolidation":    0,   # U-Zeros — preserves specificity; most uncertain = normal
    "Edema":            1,   # U-Ones
    "Pleural Effusion": 1,   # U-Ones
    "Pneumothorax":     0,   # U-Zeros — true PTX is usually clear; uncertain = absent
    "Fracture":         1,   # U-Ones  — subtle fractures are easily missed; uncertain = present
}

# NIH ChestX-ray14 label → CheXpert label mapping (NIH has Pneumothorax, not Fracture)
NIH_TO_CHEXPERT = {
    "Atelectasis":   "Atelectasis",
    "Cardiomegaly":  "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema":         "Edema",
    "Effusion":      "Pleural Effusion",  # NIH uses "Effusion"
    "Pneumothorax":  "Pneumothorax",      # Direct match — NIH annotates PTX
}

# Masks per dataset — which of the 7 labels are annotated
#  Index:       0  1  2  3  4   5   6
#  Label:      At Ca Co Ed PE  PTX Fra
MASK_CHEXPERT = [1, 1, 1, 1, 1,  1,  1]  # CheXpert annotates all 7
MASK_NIH      = [1, 1, 1, 1, 1,  1,  0]  # NIH has no Fracture annotations
MASK_RSNA     = [1, 0, 1, 0, 0,  0,  0]  # Atelectasis + Consolidation (pneumonia proxy)
MASK_KAGGLE   = [1, 0, 1, 0, 0,  0,  0]  # Same proxy


# ════════════════════════════════════════════════════════════════════════════
#  GLOBAL SIZE CONSTANT
# ════════════════════════════════════════════════════════════════════════════
# 320×320 matches the original CheXpert paper (Irvin et al. 2019) and
# Stomper10 re-implementation (+0.012 AUC vs 224×224 small variant).
IMG_SIZE = 320


# ════════════════════════════════════════════════════════════════════════════
#  TRANSFORMS  (resize to IMG_SIZE+20 → crop IMG_SIZE for augmentation)
# ════════════════════════════════════════════════════════════════════════════

def get_train_transform():
    """Stronger augmentation pipeline targeting CXR variability + rib/PTX detection.

    Improvements over baseline:
      RandomRotation ±15°  — broader patient positioning variance (helps rib fractures)
      RandomAffine scale   — mild zoom-in/out (lesion-size invariance)
      GaussianBlur         — scanner sharpness variability; helps generalization
      RandomErasing        — simulates lead markers / body-part occlusion
      Increased ColorJitter — wider scanner contrast/brightness range
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.07, 0.07), scale=(0.90, 1.10)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.10, scale=(0.01, 0.04), ratio=(0.5, 2.0), value=0),
    ])


def get_val_transform():
    """Deterministic transform for validation / inference."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ════════════════════════════════════════════════════════════════════════════
#  DATASET 1 — CheXpert
# ════════════════════════════════════════════════════════════════════════════

class ChexpertDataset(Dataset):
    """
    CheXpert multi-label dataset — 5 competition labels.

    Parameters
    ----------
    csv_file      : path to train.csv or valid.csv
    root_dir      : directory containing the CheXpert-v1.0 tree
    transform     : torchvision transform
    frontal_only  : if True, drop lateral views (frontal views give better AUC)
    u_label_soft  : if True, use MLMIP soft-label strategy — uncertain (-1) labels
                    are replaced with random Uniform[0.55, 0.85] at sample time
                    instead of the hard U-Ones / U-Zeros mapping.
                    Reference: Serbin et al. (ooodmt/MLMIP, TU Berlin) —
                    adopted from a top CheXpert competition contributor.

    Returns (image_tensor, labels_tensor, mask_tensor, img_path)
      labels_tensor : float32 [5]  — values in {0, 1} or [0.55, 0.85] for uncertain
      mask_tensor   : float32 [5]  — all 1.0 (CheXpert annotates all 5 labels)
    """

    def __init__(self, csv_file, root_dir, transform=None,
                 frontal_only=True, u_label_soft=False):
        df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self._u_label_soft = u_label_soft

        # Strip dataset-version prefix from paths
        df['Path'] = df['Path'].str.replace(
            r'^CheXpert-v1\.0(?:-small)?/', '', regex=True
        )

        # Filter to frontal views only (better AUC on the 5 competition labels)
        if frontal_only and 'Frontal/Lateral' in df.columns:
            df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

        # Apply U-label policies
        for col in COMPETITION_LABELS:
            if col in df.columns:
                if u_label_soft:
                    # Keep -1 as sentinel; replaced at sample time in __getitem__
                    df[col] = df[col].fillna(0.0)   # NaN (missing) → 0
                else:
                    # Hard U-Ones / U-Zeros per Irvin et al. 2019 Table 4
                    df[col] = df[col].fillna(0.0).replace(-1.0, float(U_POLICY[col]))
            else:
                df[col] = 0.0
        df[COMPETITION_LABELS] = df[COMPETITION_LABELS].astype(np.float32)

        self.df = df
        self._mask = torch.tensor(MASK_CHEXPERT, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        raw = row[COMPETITION_LABELS].values.astype(np.float32)
        if self._u_label_soft:
            # MLMIP soft-label: uncertain (-1.0) → Uniform[0.55, 0.85]
            # Calibrated range keeps the loss signal diffuse, avoiding
            # the over-confident 0/1 collapse seen with hard U-Ones/Zeros.
            for i in range(len(raw)):
                if raw[i] == -1.0:
                    raw[i] = np.float32(np.random.uniform(0.55, 0.85))
        labels = torch.tensor(raw, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels, self._mask.clone(), img_path


# ════════════════════════════════════════════════════════════════════════════
#  DATASET 2 — NIH ChestX-ray14
# ════════════════════════════════════════════════════════════════════════════

class NIHDataset(Dataset):
    """
    NIH ChestX-ray14 dataset — maps to 5 CheXpert competition labels.

    Expected CSV (Data_Entry_2017.csv) columns:
        Image Index | Finding Labels | ...
    'Finding Labels' is a '|'-separated string e.g. 'Atelectasis|Effusion'.

    Returns (image_tensor, labels_tensor, mask_tensor, img_path)
    """

    def __init__(self, csv_file, img_dir, transform=None):
        self.df      = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self._mask   = torch.tensor(MASK_NIH, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Index'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        findings = set(str(row['Finding Labels']).split('|'))
        labels   = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
        for nih_label, chex_label in NIH_TO_CHEXPERT.items():
            if nih_label in findings:
                idx_c = COMPETITION_LABELS.index(chex_label)
                labels[idx_c] = 1.0

        if self.transform:
            image = self.transform(image)
        return image, labels, self._mask.clone(), img_path


# ════════════════════════════════════════════════════════════════════════════
#  DATASET 3 — RSNA Pneumonia Detection Challenge  (Phase 2)
# ════════════════════════════════════════════════════════════════════════════

class RSNADataset(Dataset):
    """
    RSNA Pneumonia Detection Challenge dataset.

    CSV (stage_2_train_labels.csv): patientId | x | y | width | height | Target
    Target 1 = Lung Opacity (Pneumonia positive), 0 = Normal.

    Masked labels: Atelectasis=Target, Consolidation=Target, others masked out.
    Images expected as PNG (pre-converted from DICOM).

    Returns (image_tensor, labels_tensor, mask_tensor, img_path)
    """

    def __init__(self, csv_file, img_dir, transform=None):
        df_raw       = pd.read_csv(csv_file)
        self.df      = df_raw.groupby('patientId', as_index=False)['Target'].max()
        self.img_dir = img_dir
        self.transform = transform
        self._mask   = torch.tensor(MASK_RSNA, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            p = os.path.join(self.img_dir, row['patientId'] + ext)
            if os.path.exists(p):
                img_path = p
                break
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            img_path = img_path or ''

        # Atelectasis(0) and Consolidation(2) — pneumonia manifests as both
        labels = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
        labels[0] = float(row['Target'])   # Atelectasis
        labels[2] = float(row['Target'])   # Consolidation

        if self.transform:
            image = self.transform(image)
        return image, labels, self._mask.clone(), img_path


# ════════════════════════════════════════════════════════════════════════════
#  DATASET 4 — Kaggle Binary Chest X-Ray  (Phase 3)
# ════════════════════════════════════════════════════════════════════════════

class KaggleBinaryDataset(Dataset):
    """
    Kaggle 'Chest X-Ray Images (Pneumonia)' binary dataset.

    Folder structure:
        <root>/
            NORMAL/     *.jpeg
            PNEUMONIA/  *.jpeg

    PNEUMONIA → Atelectasis=1, Consolidation=1 (mask=[1,0,1,0,0]).
    NORMAL    → all zeros.

    Returns (image_tensor, labels_tensor, mask_tensor, img_path)
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []  # (path, is_pneumonia)
        for label_name, label_val in [('NORMAL', 0.0), ('PNEUMONIA', 1.0)]:
            folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder, fname), label_val))
        self._mask = torch.tensor(MASK_KAGGLE, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pneu = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        labels    = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
        labels[0] = pneu   # Atelectasis
        labels[2] = pneu   # Consolidation

        if self.transform:
            image = self.transform(image)
        return image, labels, self._mask.clone(), img_path


# ════════════════════════════════════════════════════════════════════════════
#  DATASET 5 — MIMIC-CXR  (optional, requires PhysioNet access)
# ════════════════════════════════════════════════════════════════════════════

class MIMICDataset(Dataset):
    """
    MIMIC-CXR-JPG dataset.

    Required files:
      mimic-cxr-2.0.0-chexpert.csv  — label file
      mimic-cxr-2.0.0-split.csv     — train/validate/test split

    Image path format:
      <img_root>/files/p<sub2>/p<subject_id>/s<study_id>/<dicom_id>.jpg

    U-label policy: same as CheXpert.
    Returns (image_tensor, labels_tensor, mask_tensor, img_path)
    """

    def __init__(self, chexpert_csv, records_csv, img_root, split='train', transform=None):
        labels_df  = pd.read_csv(chexpert_csv)
        records_df = pd.read_csv(records_csv)
        merged = records_df[records_df['split'] == split].merge(
            labels_df, on=['subject_id', 'study_id']
        )
        for col in COMPETITION_LABELS:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0).replace(-1.0, float(U_POLICY[col]))
            else:
                merged[col] = 0.0
        merged[COMPETITION_LABELS] = merged[COMPETITION_LABELS].astype(np.float32)
        self.df        = merged.reset_index(drop=True)
        self.img_root  = img_root
        self.transform = transform
        self._mask     = torch.tensor(MASK_CHEXPERT, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        subject_id = str(row['subject_id'])
        study_id   = str(row['study_id'])
        dicom_id   = str(row.get('dicom_id', ''))
        img_path   = os.path.join(
            self.img_root, 'files',
            f'p{subject_id[:2]}', f'p{subject_id}',
            f's{study_id}', f'{dicom_id}.jpg',
        )
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        labels = torch.tensor(
            row[COMPETITION_LABELS].values.astype(np.float32), dtype=torch.float32
        )
        if self.transform:
            image = self.transform(image)
        return image, labels, self._mask.clone(), img_path


# ════════════════════════════════════════════════════════════════════════════
#  POS_WEIGHT UTILITIES  (CSV-based, no image loading — fast)
# ════════════════════════════════════════════════════════════════════════════

def compute_pos_weight_chexpert(csv_file, frontal_only=True):
    """
    Compute BCEWithLogitsLoss pos_weight for the 5 competition labels from
    the CheXpert train CSV without loading any images.

    pos_weight[i] = (n_neg[i] / n_pos[i]) — standard class-imbalance correction.
    Returns: torch.Tensor shape [5]
    """
    df = pd.read_csv(csv_file)
    if frontal_only and 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']

    for col in COMPETITION_LABELS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).replace(-1.0, float(U_POLICY[col]))
        else:
            df[col] = 0.0

    pw = []
    for col in COMPETITION_LABELS:
        n_pos = max(float((df[col] == 1.0).sum()), 1.0)
        n_neg = max(float((df[col] == 0.0).sum()), 1.0)
        pw.append(n_neg / n_pos)
    pos_weight = torch.tensor(pw, dtype=torch.float32)
    print("  [pos_weight]", {c: f"{v:.1f}" for c, v in zip(COMPETITION_LABELS, pw)})
    return pos_weight


def build_row_sampler(df, label_cols):
    """
    Build a WeightedRandomSampler from a DataFrame — no image I/O.

    Weight of each row = max inverse-class-frequency across all positive labels.
    Samples with no positive label get weight 1.0.
    """
    n = len(df)
    weights = np.ones(n, dtype=np.float64)
    for col in label_cols:
        if col not in df.columns:
            continue
        col_vals = df[col].values
        n_pos = max((col_vals == 1.0).sum(), 1)
        cw    = n / n_pos
        weights = np.maximum(weights, np.where(col_vals == 1.0, cw, 1.0))
    return WeightedRandomSampler(
        torch.DoubleTensor(weights), num_samples=n, replacement=True
    )


# ════════════════════════════════════════════════════════════════════════════
#  PHASE DATALOADER FACTORIES
# ════════════════════════════════════════════════════════════════════════════

def get_phase1_dataloaders(
    chexpert_train_csv='train.csv',
    chexpert_val_csv='valid.csv',     # Official 234-image radiologist-labeled set
    chexpert_root='.',
    nih_csv=None,
    nih_img_dir=None,
    mimic_chexpert_csv=None,
    mimic_records_csv=None,
    mimic_img_root=None,
    frontal_only=True,
    u_label_soft=False,               # MLMIP soft uncertainty labels (Uniform[0.55,0.85])
    batch_size=8,                     # Safe for GTX 1650 (4 GB VRAM) with AMP
    num_workers=4,
):
    """
    Phase 1: Pre-training on CheXpert + NIH (+optional MIMIC-CXR).

    Validation uses the official CheXpert valid.csv (234 radiologist-labeled
    images) — the only set with reliable gold-standard annotations.

    Returns (train_loader, val_loader, pos_weight)
    """
    train_tf = get_train_transform()
    val_tf   = get_val_transform()

    # ── CheXpert train set ────────────────────────────────────────────────
    train_datasets = [
        ChexpertDataset(chexpert_train_csv, chexpert_root,
                        transform=train_tf, frontal_only=frontal_only,
                        u_label_soft=u_label_soft)
    ]

    # ── CheXpert official validation set (radiologist-labeled) ────────────────
    val_ds = ChexpertDataset(chexpert_val_csv, chexpert_root,
                             transform=val_tf, frontal_only=False,
                             u_label_soft=False)  # val set: use hard labels always
    val_datasets = [val_ds]
    print(f"  [Phase1] CheXpert train: {len(train_datasets[0])} | "
          f"official val: {len(val_ds)} (radiologist-labeled)")

    # ── NIH (optional) ───────────────────────────────────────────────────
    if nih_csv and nih_img_dir and os.path.exists(nih_csv):
        nih_ds = NIHDataset(nih_csv, nih_img_dir, transform=train_tf)
        train_datasets.append(nih_ds)
        print(f"  [Phase1] NIH added: {len(nih_ds)} images")

    # ── MIMIC-CXR (optional) ─────────────────────────────────────────────
    if mimic_chexpert_csv and mimic_records_csv and mimic_img_root:
        if os.path.exists(mimic_chexpert_csv):
            train_datasets.append(
                MIMICDataset(mimic_chexpert_csv, mimic_records_csv,
                             mimic_img_root, split='train', transform=train_tf)
            )
            # MIMIC validate split → add to val
            val_datasets.append(
                MIMICDataset(mimic_chexpert_csv, mimic_records_csv,
                             mimic_img_root, split='validate', transform=val_tf)
            )
            print("  [Phase1] MIMIC-CXR added")

    combined_train = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    combined_val   = ConcatDataset(val_datasets)   if len(val_datasets)   > 1 else val_datasets[0]
    print(f"  [Phase1] Total train: {len(combined_train)} | val: {len(combined_val)}")

    # ── pos_weight (fast, CSV-based) ──────────────────────────────────────
    pos_weight = compute_pos_weight_chexpert(chexpert_train_csv, frontal_only=frontal_only)

    # ── WeightedRandomSampler (CSV-based, no image loading) ──────────────
    chex_df = pd.read_csv(chexpert_train_csv)
    if frontal_only and 'Frontal/Lateral' in chex_df.columns:
        chex_df = chex_df[chex_df['Frontal/Lateral'] == 'Frontal']
    for col in COMPETITION_LABELS:
        if col in chex_df.columns:
            chex_df[col] = chex_df[col].fillna(0.0).replace(-1.0, float(U_POLICY[col]))
    sampler = build_row_sampler(chex_df, COMPETITION_LABELS)

    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, pos_weight


def get_phase2_dataloaders(
    rsna_csv,
    rsna_img_dir,
    batch_size=8,
    num_workers=4,
):
    """
    Phase 2: Fine-tuning on RSNA dataset.
    Returns (train_loader, val_loader, pos_weight).
    pos_weight is computed for Atelectasis + Consolidation only (mask positions 0,2).
    """
    train_tf = get_train_transform()
    val_tf   = get_val_transform()

    ds    = RSNADataset(rsna_csv, rsna_img_dir, transform=None)
    n     = len(ds)
    tr_n  = int(0.9 * n)
    val_n = n - tr_n
    gen   = torch.Generator().manual_seed(42)
    tr_idx, val_idx = torch.utils.data.random_split(range(n), [tr_n, val_n], generator=gen)

    class _RSNASplit(Dataset):
        def __init__(self, base, indices, tf):
            self.base    = base
            self.indices = list(indices)
            self.tf      = tf
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            row = self.base.df.iloc[self.indices[i]]
            p   = None
            for ext in ('.png', '.jpg', '.jpeg'):
                c = os.path.join(self.base.img_dir, row['patientId'] + ext)
                if os.path.exists(c): p = c; break
            try:   img = Image.open(p).convert('RGB')
            except: img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0)); p = p or ''
            lbl      = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
            lbl[0]   = float(row['Target'])
            lbl[2]   = float(row['Target'])
            mask_t   = torch.tensor(MASK_RSNA, dtype=torch.float32)
            return self.tf(img), lbl, mask_t, p

    train_ds = _RSNASplit(ds, tr_idx,  train_tf)
    val_ds   = _RSNASplit(ds, val_idx, val_tf)

    # pos_weight for Atelectasis and Consolidation (positions 0 and 2)
    n_pos = max(float(ds.df['Target'].sum()), 1.0)
    n_neg = max(float((ds.df['Target'] == 0).sum()), 1.0)
    pw    = torch.ones(len(COMPETITION_LABELS), dtype=torch.float32)
    pw[0] = n_neg / n_pos
    pw[2] = n_neg / n_pos
    print(f"  [pos_weight RSNA] Atelectasis/Consolidation: {pw[0]:.1f}")

    # Sampler from target column
    targets  = ds.df.iloc[list(tr_idx)]['Target'].values.astype(np.float64)
    cw       = len(targets) / max(targets.sum(), 1)
    s_weights = np.where(targets == 1, cw, 1.0)
    sampler  = WeightedRandomSampler(torch.DoubleTensor(s_weights), len(s_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, drop_last=True, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    print(f"  [Phase2] RSNA train: {len(train_ds)} | val: {len(val_ds)}")
    return train_loader, val_loader, pw


def get_phase3_dataloaders(
    kaggle_train_dir,
    kaggle_val_dir=None,
    batch_size=8,
    num_workers=4,
):
    """
    Phase 3: Final tuning on Kaggle binary dataset.
    Returns (train_loader, val_loader, pos_weight).
    """
    train_tf = get_train_transform()
    val_tf   = get_val_transform()

    if kaggle_val_dir and os.path.isdir(kaggle_val_dir):
        train_ds = KaggleBinaryDataset(kaggle_train_dir, transform=train_tf)
        val_ds   = KaggleBinaryDataset(kaggle_val_dir,   transform=val_tf)
    else:
        all_ds = KaggleBinaryDataset(kaggle_train_dir, transform=None)
        tr_n   = int(0.85 * len(all_ds))
        val_n  = len(all_ds) - tr_n
        gen    = torch.Generator().manual_seed(42)
        tr_sub, val_sub = torch.utils.data.random_split(all_ds, [tr_n, val_n], generator=gen)

        class _KaggleSplit(Dataset):
            def __init__(self, sub, tf):
                self.sub = sub; self.tf = tf
            def __len__(self): return len(self.sub)
            def __getitem__(self, i):
                p, pneu = self.sub.dataset.samples[self.sub.indices[i]]
                try:   img = Image.open(p).convert('RGB')
                except: img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
                lbl    = torch.zeros(len(COMPETITION_LABELS), dtype=torch.float32)
                lbl[0] = pneu; lbl[2] = pneu
                mask_t = torch.tensor(MASK_KAGGLE, dtype=torch.float32)
                return self.tf(img), lbl, mask_t, p

        train_ds = _KaggleSplit(tr_sub,  train_tf)
        val_ds   = _KaggleSplit(val_sub, val_tf)

    # pos_weight
    all_samp = KaggleBinaryDataset(kaggle_train_dir).samples
    n_pneu   = sum(1 for _, v in all_samp if v == 1.0)
    n_norm   = sum(1 for _, v in all_samp if v == 0.0)
    pw       = torch.ones(len(COMPETITION_LABELS), dtype=torch.float32)
    pw[0]    = n_norm / max(n_pneu, 1)
    pw[2]    = n_norm / max(n_pneu, 1)
    print(f"  [pos_weight Kaggle] Atelectasis/Consolidation: {pw[0]:.1f}")

    s_weights = [pw[0].item() if s[1] == 1.0 else 1.0 for s in all_samp[:len(train_ds)]]
    sampler   = WeightedRandomSampler(torch.DoubleTensor(s_weights), len(s_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, drop_last=True, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    print(f"  [Phase3] Kaggle train: {len(train_ds)} | val: {len(val_ds)}")
    return train_loader, val_loader, pw


# ════════════════════════════════════════════════════════════════════════════
#  LEGACY SHIM
# ════════════════════════════════════════════════════════════════════════════

def get_dataloaders(csv_file='train.csv', root_dir='.', batch_size=8, num_workers=4):
    """Backward-compat wrapper → Phase 1 (CheXpert only)."""
    val_csv = csv_file.replace('train.csv', 'valid.csv')
    if not os.path.exists(val_csv):
        val_csv = 'valid.csv'
    return get_phase1_dataloaders(
        chexpert_train_csv=csv_file,
        chexpert_val_csv=val_csv,
        chexpert_root=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    tl, vl, pw = get_dataloaders(batch_size=4, num_workers=2)
    imgs, lbls, masks, paths = next(iter(tl))
    print(f"Batch  : {imgs.shape}")
    print(f"Labels : {lbls.shape}  — {COMPETITION_LABELS}")
    print(f"Masks  : {masks[0]}")
    print(f"Sample label row: {lbls[0]}")
    print(f"pos_weight: {pw}")
