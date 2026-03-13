# Radiology AI Explainer — Full Build Execution Log

**Project:** Radiology AI Explainer (DenseNet121 + Grad-CAM++ + Gemini)  
**Author:** Anshu  
**Date Range:** February 2026  
**Hardware:** NVIDIA GTX 1650 (4 GB VRAM) / Intel i5-10300H / WSL2 Ubuntu  
**Final Status:** Phase 1 Complete — Mean AUC 0.8273 on CheXpert Validation Set  

---

## 1. PROJECT INITIALIZATION

### 1.1 Environment Setup

The project was developed on Windows 11 running WSL2 (Ubuntu) to leverage native CUDA support through the Linux kernel driver passthrough. All training and inference ran under WSL2 for direct GPU access.

```
OS          : Windows 11 + WSL2 (Ubuntu)
Shell       : bash (WSL2)
Workspace   : /home/anshu/minor_project/
Disk free   : 915 GB (WSL2 ext4 VHD — fast I/O)
```

### 1.2 Python Version

```bash
$ python3 --version
Python 3.12.3
```

### 1.3 Virtual Environment Creation

```bash
cd /home/anshu/minor_project
python3 -m venv venv
source venv/bin/activate
```

### 1.4 CUDA / GPU Verification

```bash
$ nvidia-smi
# NVIDIA GeForce GTX 1650, CUDA 12.8, 4096 MiB VRAM

$ python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
True NVIDIA GeForce GTX 1650
```

CUDA toolkit:
```
CUDA Version         : 12.8
cuDNN                : bundled with PyTorch 2.10.0+cu128
torch.backends.cudnn : benchmark = True (set in train.py)
```

### 1.5 Required Libraries Installed

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install streamlit==1.54.0
pip install langchain==1.2.10 langchain-google-genai==4.2.1 langchain-huggingface==1.2.0
pip install scikit-learn==1.8.0
pip install grad-cam==1.5.5
pip install matplotlib==3.10.8
pip install numpy==2.4.2
pip install pandas==2.3.3
pip install pillow==12.1.1
pip install opencv-python==4.13.0.92
pip install python-dotenv==1.2.1
pip install tqdm==4.67.3
```

Full dependency manifest:

| Package | Version | Purpose |
|---|---|---|
| torch | 2.10.0+cu128 | Model training & inference |
| torchvision | 0.25.0 | DenseNet121 pretrained weights, transforms |
| streamlit | 1.54.0 | Web frontend |
| langchain | 1.2.10 | LLM pipeline orchestration |
| langchain-google-genai | 4.2.1 | Google Gemini integration |
| langchain-huggingface | 1.2.0 | HuggingFace Zephyr-7B (original LLM, later replaced) |
| scikit-learn | 1.8.0 | AUC, ROC, confusion matrix, F1/precision/recall |
| grad-cam | 1.5.5 | Grad-CAM++ heatmap generation |
| matplotlib | 3.10.8 | Plot generation (ROC curves, bar charts, radar charts) |
| numpy | 2.4.2 | Array operations |
| pandas | 2.3.3 | CSV parsing and data manipulation |
| pillow | 12.1.1 | Image loading and conversion |
| opencv-python | 4.13.0.92 | Image processing for heatmap overlay |
| python-dotenv | 1.2.1 | `.env` file API key loading |
| tqdm | 4.67.3 | Training progress bars |

---

## 2. DATASET PREPARATION

### 2.1 Dataset Used — CheXpert (Phase 1)

CheXpert (Stanford ML Group, Irvin et al. 2019) — the largest publicly available chest X-ray dataset with uncertainty labels.

| Property | Value |
|---|---|
| Dataset | CheXpert-v1.0-small |
| Total training images | 223,414 (train.csv: 223,415 rows including header) |
| Frontal-only (used) | ~191,027 (after filtering Lateral views) |
| Validation images | 234 (radiologist-labeled gold standard, valid.csv: 235 rows including header) |
| Image format | JPEG, variable resolution |
| CSV columns | 19 (14 pathology labels + 5 metadata columns) |
| Labels used | 5 competition labels out of 14 total |

### 2.2 CSV Structure

```
Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Enlarged Cardiomediastinum,
Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,
Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices
```

Sample row:
```
CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg,Female,68,Frontal,AP,
1.0,,,,,,,,,0.0,,,,1.0
```

### 2.3 CSV Parsing Logic

Implemented in `dataset.py → ChexpertDataset.__init__()`:

```python
df = pd.read_csv(csv_file)

# Strip dataset-version prefix from paths for local directory compatibility
df['Path'] = df['Path'].str.replace(
    r'^CheXpert-v1\.0(?:-small)?/', '', regex=True
)
# Result: "train/patient00001/study1/view1_frontal.jpg"
```

### 2.4 Frontal Image Filtering

Only frontal views (AP/PA) were used. Lateral views were excluded because frontal views give significantly better AUC on the 5 competition labels (consistent with the original CheXpert paper methodology).

```python
if frontal_only and 'Frontal/Lateral' in df.columns:
    df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)
```

This reduced the training set from 223,414 → ~191,027 images.

### 2.5 Uncertainty Handling Policy

CheXpert uses three label values: `1.0` (positive), `0.0` (negative), `-1.0` (uncertain), and `NaN` (not mentioned).

The uncertainty policy follows **Irvin et al. 2019, Table 4** — the "diff" policy where each label has its own mapping:

| Index | Label | Policy | Mapping | Rationale |
|---|---|---|---|---|
| 0 | Atelectasis | U-Ones | −1 → 1 | Strong positive signal in uncertain cases |
| 1 | Cardiomegaly | U-Ones | −1 → 1 | Enlarged heart rarely truly uncertain |
| 2 | Consolidation | U-Zeros | −1 → 0 | Preserves specificity; most uncertain = normal |
| 3 | Edema | U-Ones | −1 → 1 | Fluid accumulation has strong radiographic signs |
| 4 | Pleural Effusion | U-Ones | −1 → 1 | Well-defined meniscus sign |

Implementation in `dataset.py`:

```python
U_POLICY = {
    "Atelectasis":      1,   # U-Ones
    "Cardiomegaly":     1,   # U-Ones
    "Consolidation":    0,   # U-Zeros
    "Edema":            1,   # U-Ones
    "Pleural Effusion": 1,   # U-Ones
}

# Applied during CSV parsing:
for col in COMPETITION_LABELS:
    df[col] = df[col].fillna(0.0).replace(-1.0, float(U_POLICY[col]))
```

An **alternative soft-label strategy** (MLMIP, TU Berlin) was also implemented but not used by default:
```python
# u_label_soft=True: uncertain (-1.0) → Uniform[0.55, 0.85] at sample time
# Calibrated range keeps the loss signal diffuse, avoiding
# the over-confident 0/1 collapse seen with hard U-Ones/Zeros.
if raw[i] == -1.0:
    raw[i] = np.float32(np.random.uniform(0.55, 0.85))
```

### 2.6 Class Imbalance Calculation

Class imbalance was computed from `train.csv` without loading any images (CSV-based, fast):

```python
def compute_pos_weight_chexpert(csv_file, frontal_only=True):
    df = pd.read_csv(csv_file)
    if frontal_only and 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']
    # Apply uncertainty policy first
    for col in COMPETITION_LABELS:
        df[col] = df[col].fillna(0.0).replace(-1.0, float(U_POLICY[col]))
    pw = []
    for col in COMPETITION_LABELS:
        n_pos = max(float((df[col] == 1.0).sum()), 1.0)
        n_neg = max(float((df[col] == 0.0).sum()), 1.0)
        pw.append(n_neg / n_pos)
    return torch.tensor(pw, dtype=torch.float32)
```

### 2.7 pos_weight Computation Formula

$$\text{pos\_weight}[i] = \frac{n_{\text{neg}}[i]}{n_{\text{pos}}[i]}$$

Where $n_{\text{neg}}$ = count of `0.0` labels and $n_{\text{pos}}$ = count of `1.0` labels (after uncertainty mapping) for each of the 5 competition labels.

Computed values:

| Label | pos_weight | Imbalance Ratio |
|---|---|---|
| Atelectasis | ~2.2 | Mild |
| Cardiomegaly | ~5.3 | Moderate |
| Consolidation | ~13.7 | Severe |
| Edema | ~2.1 | Mild |
| Pleural Effusion | ~1.2 | Nearly balanced |

### 2.8 WeightedRandomSampler Implementation

To prevent class-imbalanced mini-batches, a `WeightedRandomSampler` was built from the CSV (no image I/O):

```python
def build_row_sampler(df, label_cols):
    n = len(df)
    weights = np.ones(n, dtype=np.float64)
    for col in label_cols:
        col_vals = df[col].values
        n_pos = max((col_vals == 1.0).sum(), 1)
        cw    = n / n_pos                 # inverse class frequency
        weights = np.maximum(weights, np.where(col_vals == 1.0, cw, 1.0))
    return WeightedRandomSampler(
        torch.DoubleTensor(weights), num_samples=n, replacement=True
    )
```

Logic: Each row's sampling weight = max inverse-class-frequency across all its positive labels. Rows with no positive labels get weight 1.0.

---

## 3. PREPROCESSING PIPELINE

### 3.1 Training Transform

```python
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((340, 340)),                    # Resize to IMG_SIZE+20
        transforms.RandomCrop(320),                       # Random crop to 320×320
        transforms.RandomHorizontalFlip(p=0.5),           # CXR laterality invariant
        transforms.RandomRotation(degrees=10),            # Patient positioning variance
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Translation robustness
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Scanner variation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
```

Augmentation rationale (from PMC11355845 survey on CXR deep learning):
- `RandomHorizontalFlip` — CXR laterality invariant augmentation  
- `RandomRotation(10°)` — mimics slight patient positioning variance  
- `RandomAffine(translate=5%)` — slight translation robustness (dataset bias mitigation)  
- `ColorJitter` — scanner contrast/brightness variation between institutions

### 3.2 Validation Transform

```python
def get_val_transform():
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
```

No augmentation. Deterministic for reproducible evaluation.

### 3.3 Image Resolution Decision — Why 320×320

**320×320** was chosen because:

1. **Matches the original CheXpert paper** (Irvin et al. 2019) — ensures direct comparability with Stanford baselines.
2. **Stomper10 re-implementation** measured **+0.012 AUC** vs the 224×224 "small" variant — a meaningful improvement.
3. **VRAM feasibility**: Verified at 2.2 GB peak (fp32) on GTX 1650 with batch_size=8, leaving 1.8 GB headroom.
4. **Trade-off**: 320×320 captures sufficient anatomical detail (pleural lines, fluid levels) without the VRAM cost of 512×512.

Constant defined in `dataset.py`:
```python
IMG_SIZE = 320
```

### 3.4 Normalization Values

ImageNet normalization statistics were used (standard for transfer learning from ImageNet-pretrained backbones):

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

These values normalize pixel intensities to match the distribution the DenseNet121 backbone was pre-trained on.

### 3.5 DataLoader Configuration

```python
train_loader = DataLoader(
    combined_train,
    batch_size=8,
    sampler=sampler,                        # WeightedRandomSampler
    num_workers=4,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,                # Keep workers alive across epochs
)

val_loader = DataLoader(
    combined_val,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
```

### 3.6 Batch Size & Gradient Accumulation

| Parameter | Value | Rationale |
|---|---|---|
| Physical batch_size | 8 | Maximum safe for 4 GB VRAM @ 320×320 fp32 |
| grad_accum_steps | 4 | Simulates larger batch without extra VRAM |
| Effective batch size | 32 | 8 × 4 = 32 (standard for medical imaging) |

Gradient accumulation logic in `train_one_epoch()`:
```python
loss = criterion(outputs, labels, masks) / grad_accum_steps
loss.backward()

if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
    optimizer.step()
    optimizer.zero_grad()
```

---

## 4. MODEL CONSTRUCTION

### 4.1 DenseNet121 Loading Method

```python
from torchvision import models

weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
model   = models.densenet121(weights=weights)
```

Uses `torchvision.models.densenet121` with the official `DenseNet121_Weights.IMAGENET1K_V1` enum (PyTorch 2.x API).

### 4.2 Pretrained Weights Source

- **Source:** ImageNet1K_V1 (ImageNet ILSVRC 2012, 1000-class classification)
- **Format:** Bundled with `torchvision==0.25.0`
- **Purpose:** Transfer learning — the ImageNet features (edges, textures, shapes) transfer well to medical imaging

### 4.3 Classifier Head Modification

The original DenseNet121 has a classifier `Linear(1024, 1000)` for ImageNet. This was replaced:

```python
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
# model.classifier = nn.Linear(1024, 5)
```

- **Input features:** 1024 (output of the final dense block's batch norm → ReLU → global average pool)
- **Output:** 5 (one logit per competition label)
- **No dropout** in the classifier — DenseNet121's dense connectivity acts as implicit regularization

### 4.4 Total Parameters

```
Total parameters : 6,958,981
Trainable        : 6,958,981 (all layers fine-tuned in Phase 1)
Classifier head  : 1024 × 5 + 5 = 5,125 parameters
Backbone         : 6,953,856 parameters
```

### 4.5 Output Format

- **Training:** Raw logits `[B, 5]` → fed directly to `BCEWithLogitsLoss` (numerically stable)
- **Inference:** `torch.sigmoid(logits)` → probabilities in `[0, 1]` per class
- **Multi-label:** Each pathology is independent — a patient can have multiple conditions simultaneously

---

## 5. LOSS & OPTIMIZATION

### 5.1 MaskedBCEWithLogitsLoss Logic

Custom loss function that supports mixing datasets with different label spaces. Each sample carries a mask tensor — loss is only computed on labelled entries.

```python
class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight  # [C] tensor or None

    def forward(self, logits, targets, masks):
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
```

**Mask tensors per dataset:**
| Dataset | Mask | Description |
|---|---|---|
| CheXpert | `[1, 1, 1, 1, 1]` | All 5 labels annotated |
| NIH ChestX-ray14 | `[1, 1, 1, 1, 1]` | All 5 mapped via NIH→CheXpert vocabulary |
| RSNA Pneumonia | `[1, 0, 1, 0, 0]` | Atelectasis + Consolidation only (pneumonia proxy) |
| Kaggle Binary | `[1, 0, 1, 0, 0]` | Same proxy |

### 5.2 pos_weight Integration

The `pos_weight` tensor is passed to `BCEWithLogitsLoss` to up-weight the positive class in each label, compensating for class imbalance. It is broadcast as `pos_weight.unsqueeze(0)` to match the `[B, C]` shape.

Formula applied internally by PyTorch:
$$\mathcal{L} = -\left[ \text{pos\_weight} \cdot y \cdot \log(\sigma(x)) + (1-y) \cdot \log(1-\sigma(x)) \right]$$

### 5.3 Optimizer Selection and Hyperparameters

```python
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | Adam | Standard for medical imaging fine-tuning; adaptive LR per-parameter |
| Learning rate | 1e-4 | Proven effective for DenseNet121 CheXpert training (multiple references) |
| Weight decay | 1e-5 | Light L2 regularization to prevent overfitting on imbalanced classes |
| Betas | (0.9, 0.999) | PyTorch defaults — no modification needed |

### 5.4 Learning Rate Scheduler Details

```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
# T_max=10, eta_min=1e-6
```

- **Type:** CosineAnnealingLR
- **T_max:** 10 (total epochs)
- **eta_min:** LR/100 = 1e-6
- **Behavior:** Smooth cosine decay from 1e-4 → 1e-6 over 10 epochs
- **Advantage:** Better final convergence vs StepLR; avoids sudden LR drops that destabilize medical image training

### 5.5 Precision Mode — fp32

**fp32 (full precision) was used, NOT fp16 (AMP).**

Reason: DenseNet121's architecture uses dense concatenation paths where feature maps from all preceding layers are concatenated. In fp16, this causes the accumulated values to overflow the float16 range (max ~65,504), producing **NaN** in forward pass outputs.

This was confirmed empirically:
- With `torch.amp.autocast('cuda')` → all predictions = NaN
- Without autocast (fp32) → correct predictions, 2.2 GB VRAM peak

Comment in `train.py`:
```python
# NOTE: AMP (fp16) disabled — DenseNet121's dense concatenation paths
# overflow fp16 range, producing NaN.  fp32 fits fine on GTX 1650
# (2.2 GB peak at batch=8, 320×320).
```

---

## 6. TRAINING EXECUTION (PHASE 1)

### 6.1 Command Used to Start Training

```bash
cd /home/anshu/minor_project
python3 train.py
```

Or via the orchestration script:
```bash
python3 run_local.py --phase 1 --epochs 10
```

The `train.py` entry point:
```python
if __name__ == "__main__":
    phase1_pretrain(
        chexpert_train_csv = 'train.csv',
        chexpert_val_csv   = 'valid.csv',
        chexpert_root      = '.',
        frontal_only       = True,
        batch_size         = 8,
        epochs             = 10,
        lr                 = 1e-4,
        num_workers        = 4,
        grad_accum_steps   = 4,
    )
```

### 6.2 Epoch Configuration

| Setting | Value |
|---|---|
| Total epochs | 10 |
| Effective batch size | 32 (8 × 4 grad accum) |
| Training samples per epoch | ~191,027 (frontal-only, with WeightedRandomSampler) |
| Validation samples | 234 (official CheXpert radiologist-labeled set) |
| Steps per epoch | ~23,878 (191,027 / 8) |
| Optimizer steps per epoch | ~5,970 (steps / grad_accum_steps) |

### 6.3 Checkpoint Saving Logic

**Every epoch**, a full checkpoint is saved (safe resume capability):

```python
save_checkpoint({
    'epoch':           epoch,
    'model_state':     model.state_dict(),
    'optimizer_state':  optimizer.state_dict(),
    'scheduler_state':  scheduler.state_dict(),
    'best_mean_auc':   best_mean_auc,
    'auc_per_class':   auc_per_class,
}, "checkpoint_phase1.pth")
```

Checkpoint file: `checkpoint_phase1.pth` (81 MB — includes model + optimizer + scheduler states).

### 6.4 Best Model Saving Condition

Best weights are saved only when mean AUC improves:

```python
if mean_auc > best_mean_auc:
    best_mean_auc = mean_auc
    torch.save(model.state_dict(), "best_densenet121_phase1.pth")
    print(f"  ✓ New best mean AUC {best_mean_auc:.4f} → saved")
```

Best weights file: `best_densenet121_phase1.pth` (28 MB — model state_dict only).

**Important:** The checkpoint stores the LAST epoch's `auc_per_class`, while the best weights file was saved at an EARLIER epoch. These contain different per-class values.

### 6.5 VRAM Usage

| Metric | Value |
|---|---|
| VRAM peak | 2.2 GB |
| VRAM total | 4.0 GB |
| Headroom | 1.8 GB |
| Precision | fp32 |
| Batch size | 8 |
| Image size | 320×320×3 |

### 6.6 Training Time Per Epoch

| Metric | Value |
|---|---|
| Time per epoch | ~6h 33min |
| Total Phase 1 (10 epochs) | ~65h |
| Images per second | ~8.1 img/s |
| GPU utilization | ~95% during forward/backward |

### 6.7 Training Progression

| Epoch | Train Loss | Val Loss | Mean AUC | Best? |
|---|---|---|---|---|
| 1 | — | — | 0.8091 | ✓ |
| 2 | — | — | 0.8126 | ✓ |
| 3 | — | — | 0.8223 | ✓ |
| **4** | **0.9244** | **0.8702** | **0.8273** | **✓ Best** |
| 5–10 | — | — | — | No improvement |

**Best epoch: 4** (at ~26h elapsed). Epochs 5–10 showed no improvement — typical with CosineAnnealingLR where the learning rate drops below useful levels in the second half.

### 6.8 Best Epoch Per-Class AUC

| Label | AUC | Status |
|---|---|---|
| Atelectasis | 0.7504 | 🟡 |
| Cardiomegaly | 0.7503 | 🟡 |
| Consolidation | 0.8393 | 🟡 |
| **Edema** | **0.9382** | 🟢 **Exceeds Stanford** |
| Pleural Effusion | 0.8584 | 🟢 |
| **Mean** | **0.8273** | Foundation complete |

---

## 7. EVALUATION

### 7.1 Evaluation Command

```bash
python3 evaluate.py --weights best_densenet121_phase1.pth --val_csv valid.csv --save_dir results/
```

### 7.2 AUC Calculation Method

Per-class AUC computed using `sklearn.metrics.roc_auc_score`:

```python
from sklearn.metrics import roc_auc_score

for i, label in enumerate(COMPETITION_LABELS):
    y_true = targets[:, i]       # ground-truth binary labels
    y_score = preds[:, i]        # sigmoid probabilities
    auc = roc_auc_score(y_true, y_score)
```

Mean AUC = arithmetic mean of the 5 per-class AUC values. This is the **primary metric** (consistent with the CheXpert competition leaderboard).

### 7.3 ROC Curve Generation

```python
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_true, y_score)
ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"AUC = {auc:.3f}")
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)  # diagonal baseline
ax.fill_between(fpr, tpr, alpha=0.08, color=colors[i])
```

One subplot per label. Saved to `results/roc_curves.png` at 150 DPI.

### 7.4 Confusion Matrix Logic

Binary confusion matrix per label at threshold 0.5:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = (preds[:, i] >= 0.5).astype(int)
cm     = confusion_matrix(y_true, y_pred)
disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Neg', 'Pos'])
```

Saved to `results/confusion_matrices.png`.

### 7.5 Per-Class Performance (Full Results)

Evaluation on 234 radiologist-labeled CheXpert validation images using `best_densenet121_phase1.pth`:

| Label | AUC | Accuracy | F1 | Precision | Recall | Avg Precision |
|---|---|---|---|---|---|---|
| Atelectasis | 0.7504 | 0.6410 | 0.5922 | 0.4841 | 0.7625 | 0.5933 |
| Cardiomegaly | 0.7503 | 0.7179 | 0.5714 | 0.5116 | 0.6471 | 0.6923 |
| Consolidation | 0.8393 | 0.6838 | 0.4394 | 0.2929 | 0.8788 | 0.4073 |
| Edema | 0.9382 | 0.8632 | 0.6981 | 0.6066 | 0.8222 | 0.7833 |
| Pleural Effusion | 0.8584 | 0.7650 | 0.6667 | 0.5612 | 0.8209 | 0.7162 |
| **Overall** | **0.8273** | **0.7342** | **0.5924** | **0.4809** | **0.7713** | — |

### 7.6 Stanford Comparison

| Label | Ours | Stanford (Irvin 2019) | Gap |
|---|---|---|---|
| Atelectasis | 0.7504 | 0.8580 | −0.1076 |
| Cardiomegaly | 0.7503 | 0.8320 | −0.0817 |
| Consolidation | 0.8393 | 0.8990 | −0.0597 |
| **Edema** | **0.9382** | **0.9240** | **+0.0142** |
| Pleural Effusion | 0.8584 | 0.9680 | −0.1096 |
| **Mean** | **0.8273** | **0.8962** | **−0.0689** |

**Edema (0.9382) genuinely exceeds Stanford's single-model baseline (0.924).** Note: Stanford uses an ensemble; our single-model performance is competitive.

### 7.7 Output Files Generated

```
results/
├── metrics.json             — Full JSON metrics (frontend consumption)
├── auc_report.txt           — Plain-text AUC summary
├── roc_curves.png           — ROC curve per label (150 DPI)
├── precision_recall.png     — PR curve per label (150 DPI)
└── confusion_matrices.png   — Confusion matrix per label (150 DPI)
```

---

## 8. EXPLAINABILITY IMPLEMENTATION

### 8.1 Grad-CAM++ Library Used

```python
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
```

Package: `grad-cam==1.5.5` (pytorch-grad-cam)

### 8.2 Target Layer Selection — denseblock4

```python
def get_target_layer(model):
    if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
        return model.features.denseblock4
```

**Why `denseblock4`:**
- DenseNet121's dense connections preserve spatial texture features (consolidated lung areas, fluid lines) all the way through the network
- `denseblock4` is the last feature block before global average pooling — richest source of class-discriminative activations
- Grad-CAM++ at this layer captures both high-level semantic regions and fine-grained spatial detail
- Validated by PMC11355845 survey: Grad-CAM++ on the last dense block is state-of-the-art for CXR XAI

### 8.3 Heatmap Generation Process

Step-by-step in `gradcam_utils.py → get_gradcam_heatmap()`:

```python
# 1. Initialize Grad-CAM++
cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

# 2. Setup target (specific pathology class)
targets = [ClassifierOutputTarget(target_category_idx)]

# 3. Generate grayscale heatmap
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]  # shape: [H, W]

# 4. Load and resize original image
original_img = Image.open(img_path).convert('RGB')
original_img = original_img.resize((320, 320))
original_img_np = np.array(original_img) / 255.0

# 5. Create overlay visualization
visualization = show_cam_on_image(original_img_np, grayscale_cam, use_rgb=True)
```

### 8.4 Quadrant-Based Region Activation Logic

The heatmap is divided into 4 quadrants. Mean activation per quadrant determines which regions the model focused on:

```python
h, w = grayscale_cam.shape
mid_h, mid_w = h // 2, w // 2

regions_activations = {
    "upper right": np.mean(grayscale_cam[:mid_h, mid_w:]),
    "upper left":  np.mean(grayscale_cam[:mid_h, :mid_w]),
    "lower right": np.mean(grayscale_cam[mid_h:, mid_w:]),
    "lower left":  np.mean(grayscale_cam[mid_h:, :mid_w]),
}
```

### 8.5 Threshold Value Used — 0.4

```python
active_regions = [region for region, act in regions_activations.items() if act > 0.4]
if not active_regions:
    max_region = max(regions_activations, key=regions_activations.get)
    active_regions = [max_region]  # Fallback: at least one region reported

regions_text = ", ".join(active_regions)
```

- Threshold `0.4` balances sensitivity (detecting all active regions) with specificity (avoiding false activations).
- Fallback ensures at least one region is always reported to the LLM.

### 8.6 Overlay Creation

Uses `pytorch-grad-cam`'s `show_cam_on_image()` which converts the grayscale heatmap to a JET colormap and blends it with the original image:

```python
visualization = show_cam_on_image(original_img_np, grayscale_cam, use_rgb=True)
```

Output: RGB numpy array ready for Streamlit `st.image()`.

---

## 9. LLM INTEGRATION

### 9.1 Model Used

**Primary (current):** Google Gemini (gemini-2.5-flash / gemini-2.5-flash-lite / gemini-2.0-flash)  
**Original (deprecated):** HuggingFaceH4/zephyr-7b-beta via HuggingFace Inference Endpoints

The system was originally built with Zephyr-7B but migrated to Google Gemini for superior medical reasoning and availability.

```python
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
```

### 9.2 API Integration Method

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.3,
    max_output_tokens=1024,
    google_api_key=api_key,
)
```

- **Framework:** LangChain (`langchain-google-genai==4.2.1`)
- **API key:** Loaded from `.env` file via `python-dotenv`, **not exposed in the UI**  
- **Fallback chain:** Tries `gemini-2.5-flash` → `gemini-2.5-flash-lite` → `gemini-2.0-flash` in order until one responds

### 9.3 Prompt Template Structure

LangChain `ChatPromptTemplate` with system/history/human slots:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm
```

The system prompt is dynamically constructed from the model's predictions and Grad-CAM regions:

```python
def format_system_prompt(prob_text, regions_text):
    return f"""You are a helpful expert radiologist AI assistant.
Discuss this chest X-ray with the patient using a structured, reasoning-based approach.

**Factual Deductions from AI Model:**
{prob_text}

**Actual Visual Evidence (Grad-CAM Focus):**
{regions_text}

**Your Task:**
Provide your response in the following structured format:

1. **Findings Summary:** Briefly summarize the model's factual deductions and risk labels.
2. **Visual Evidence Reasoning:** Explain *why* the model might have looked at the
   specific regions highlighted by Grad-CAM.
3. **Medical Impression:** Deliver a clear, empathetic, and professional explanation
   of what this means for the patient.
"""
```

### 9.4 Risk-Level Logic

```python
def get_diagnosis_verdicts(predictions):
    for disease, prob in predictions.items():
        if prob > 0.5:
            risk = "High Risk (Positive)"
        elif prob > 0.2:
            risk = "Moderate / Elevated Risk"
        else:
            risk = "Low Risk (Negative)"
```

| Probability Range | Risk Level | Color in UI |
|---|---|---|
| > 0.5 | High Risk (Positive) | Red |
| 0.2 – 0.5 | Moderate / Elevated Risk | Orange |
| < 0.2 | Low Risk (Negative) | Green |

### 9.5 Structured Output Format

The LLM is instructed to produce:
1. **Findings Summary** — Model's factual deductions and risk labels
2. **Visual Evidence Reasoning** — Why the model focused on specific Grad-CAM regions
3. **Medical Impression** — Clear, empathetic explanation for the patient

### 9.6 Session Chat Handling

```python
# Session state initialization
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""

# First message auto-generated on analysis
response = chain.invoke({
    "system_prompt": system_prompt,
    "history": [],
    "input": "Translate these findings into a clear and empathetic medical impression."
})

# Follow-up messages include full chat history
history_msgs = []
for m in st.session_state.chat_history[:-1]:
    if m["role"] == "user":
        history_msgs.append(HumanMessage(content=m["content"]))
    else:
        history_msgs.append(AIMessage(content=m["content"]))

response = chain.invoke({
    "system_prompt": system_prompt,
    "history": history_msgs,
    "input": user_prompt,
})
```

---

## 10. FRONTEND DEVELOPMENT

### 10.1 Streamlit Setup

```python
st.set_page_config(
    page_title="Radiology AI Explainer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)
```

Custom CSS injected for:
- Gradient hero banner
- Hover-animated stat cards
- Risk color badges (red/orange/green)
- Styled file uploader (dashed border)
- Hidden Streamlit hamburger menu and footer

### 10.2 Tab-Based Navigation

Three main tabs for clean UX:

```python
tab_scan, tab_perf, tab_about = st.tabs([
    "🔬  Scan & Diagnose",
    "📊  Model Performance",
    "ℹ️  About",
])
```

1. **Scan & Diagnose** — Image upload, inference, predictions, Grad-CAM, AI chat
2. **Model Performance** — Metric cards, AUC table, bar chart, radar chart, evaluation curves
3. **About** — Pathology descriptions, how-it-works, tech stack, disclaimer

### 10.3 File Upload Implementation

```python
uploaded_file = st.file_uploader(
    "Drop your X-Ray here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)
```

Uploaded file is saved to a temp file for Grad-CAM (which needs a file path):
```python
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    tmp_img_path = tmp_file.name
image = Image.open(uploaded_file).convert('RGB')
```

### 10.4 Probability Visualization

Risk cards row with HTML/CSS badges:
```python
risk_cols = st.columns(len(COMPETITION_LABELS))
for i, (disease, prob) in enumerate(preds.items()):
    if prob > 0.5:
        badge = "risk-high"    # red
    elif prob > 0.2:
        badge = "risk-mod"     # orange
    else:
        badge = "risk-low"     # green
```

Detailed probability bars in an expander:
```python
with st.expander("📋 Detailed probability breakdown"):
    for disease, prob in preds.items():
        st.markdown(f"**{disease}** — {prob:.1%}")
        st.progress(prob)
```

### 10.5 Heatmap Display Logic

Side-by-side original and Grad-CAM overlay:
```python
gcam_left, gcam_right = st.columns(2)
with gcam_left:
    st.image(image, caption="Original", use_container_width=True)
with gcam_right:
    st.image(heatmap_vis, caption=f"{target_disease} — {active_regions}",
             use_container_width=True)
```

Disease selector:
```python
target_disease = st.selectbox("Pathology:", COMPETITION_LABELS)
```

### 10.6 Chat UI Implementation

Uses Streamlit's native chat components:
```python
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a follow-up question…"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = chain.invoke({...})
            st.markdown(resp.content)
```

### 10.7 Running the App

```bash
cd /home/anshu/minor_project
streamlit run app.py --server.headless true
# Opens at http://localhost:8501
```

---

## 11. FILE STRUCTURE CREATED

```
/home/anshu/minor_project/
│
├── model.py                  (61 lines)   — DenseNet121 architecture, COMPETITION_LABELS
├── dataset.py                (705 lines)  — 5 Dataset classes, transforms, dataloaders,
│                                            pos_weight, WeightedRandomSampler
├── train.py                  (462 lines)  — MaskedBCE loss, 3-phase training pipeline,
│                                            checkpoint/resume, gradient accumulation
├── run_local.py              (285 lines)  — GTX 1650 orchestration, hardware checks,
│                                            CLI argument parsing, path configuration
├── evaluate.py               (336 lines)  — ROC/PR/CM curves, JSON metrics, Stanford comparison
├── evaluate_phase3.py        (315 lines)  — Binary pneumonia evaluation for Phase 3 dataset
├── gradcam_utils.py          (81 lines)   — Grad-CAM++ heatmap generation, region detection
├── llm_explainer.py          (86 lines)   — LangChain + Google Gemini pipeline
├── app.py                    (668 lines)  — Streamlit frontend (tabs, dashboard, chat)
├── main.py                   (91 lines)   — Single-image inference pipeline
├── test_pneumonia.py         (48 lines)   — Quick pneumonia detection test script
│
├── .env                                   — API key (gemini_api_key=...)
├── train.csv                              — CheXpert training labels (223,414 data rows)
├── valid.csv                              — CheXpert validation labels (234 data rows)
│
├── train/                                 — CheXpert training images
│   ├── patient00001/
│   ├── patient00002/
│   └── ... (~191K frontal images)
├── valid/                                 — CheXpert validation images (234 images)
│
├── best_densenet121_phase1.pth   (28 MB)  — Best Phase 1 weights (epoch 4, AUC 0.8273)
├── checkpoint_phase1.pth         (81 MB)  — Full checkpoint (epoch 10, model+opt+sched)
├── best_resnet50.pth             (90 MB)  — Legacy weights (unused, backward compat)
│
├── results/
│   ├── metrics.json                        — Full evaluation metrics (JSON)
│   ├── auc_report.txt                      — Plain-text AUC report
│   ├── roc_curves.png                      — ROC curves (5 subplots)
│   ├── precision_recall.png                — PR curves (5 subplots)
│   └── confusion_matrices.png              — Confusion matrices (5 subplots)
│
├── phase_3/                               — Kaggle binary dataset (NORMAL/PNEUMONIA)
│   └── chest_xray/
│       ├── train/ (5,216 images)
│       ├── val/   (16 images)
│       └── test/  (624 images)
│
├── implementation.md                       — Implementation report
├── build_execution_log.md                  — This file
├── colab_training.ipynb                    — Google Colab training notebook
└── __pycache__/                            — Python bytecode cache
```

**Total Python codebase: 3,137 lines across 11 files.**

---

## 12. HARDWARE DETAILS

### 12.1 GPU

| Property | Value |
|---|---|
| Model | NVIDIA GeForce GTX 1650 |
| Architecture | Turing (TU117) |
| VRAM | 4 GB GDDR5 |
| CUDA Cores | 896 |
| Memory Bandwidth | 128 GB/s |
| Compute Capability | 7.5 |

### 12.2 VRAM Peak Usage

| Phase | Peak VRAM | Configuration |
|---|---|---|
| Training (Phase 1) | 2.2 GB | batch=8, 320×320, fp32, DenseNet121 |
| Inference (single image) | ~1.0 GB | batch=1, 320×320, fp32 |
| Evaluation (batch=16) | ~1.5 GB | batch=16, 320×320, fp32, no_grad |
| Grad-CAM++ | ~1.2 GB | Requires gradient computation for target layer |

### 12.3 CPU

| Property | Value |
|---|---|
| Model | Intel Core i5-10300H |
| Cores/Threads | 4 / 8 |
| Base Clock | 2.5 GHz |
| Turbo | 4.5 GHz |
| DataLoader workers | 4 (persistent) |

### 12.4 System

| Property | Value |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu) |
| RAM | 7.7 GB |
| Disk | 915 GB free (WSL2 ext4 VHD) |
| CUDA | 12.8 |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu128 |

### 12.5 Total Training Time

| Phase | Duration |
|---|---|
| Phase 1 (10 epochs) | ~65 hours |
| Best epoch reached | Epoch 4 (~26h elapsed) |
| Evaluation | ~2 minutes |
| Grad-CAM per image | ~0.3 seconds |

---

## 13. DECISIONS & JUSTIFICATIONS

### 13.1 Why DenseNet121?

1. **Reference architecture** from both the CheXpert paper (Irvin et al. 2019) and CheXNet (Rajpurkar et al. 2017) — ensures direct comparability.
2. **Dense connections** preserve fine-grained texture gradients (lung opacity, consolidation, fluid lines) all the way to the classifier — ResNet skip-connections lose this spatial detail.
3. **Stanford ensemble achieved 0.907 mean AUC** using DenseNet121 — proven architecture for this task.
4. **Compact model** (6.96M params, 28 MB weights) fits comfortably on GTX 1650 (4 GB VRAM).
5. **Rich Grad-CAM++ features** — `denseblock4` preserves spatial activations ideal for heatmap visualization.

### 13.2 Why 320×320 Resolution?

1. **Matches the original CheXpert paper** — ensures reproducibility and fair comparison.
2. **Measured +0.012 AUC improvement** over 224×224 (Stomper10 re-implementation benchmark).
3. **VRAM feasible** — 2.2 GB peak at batch=8, well within 4 GB limit.
4. **Sufficient anatomical detail** — pleural lines, fluid levels, cardiac silhouette boundaries are resolved at 320px.
5. **Trade-off vs 512×512** — diminishing AUC returns don't justify the quadrupled VRAM cost.

### 13.3 Why Grad-CAM++?

1. **PMC11355845 survey** identified Grad-CAM++ as **state-of-the-art for chest X-ray explainability**.
2. **Multi-instance localization** — when multiple pathological regions exist in one image (e.g., bilateral pleural effusion), Grad-CAM++ produces more accurate per-region activations than plain Grad-CAM.
3. **Second-order gradient weighting** provides more mathematically sound importance maps.
4. **Library availability** — `pytorch-grad-cam` provides a clean API with `GradCAMPlusPlus` class.
5. **Validated specifically for medical imaging** — not just a general-purpose XAI method.

### 13.4 Why Gemini (originally Zephyr-7B)?

**Original choice — Zephyr-7B:**
- Free tier via HuggingFace Inference Endpoints
- Good instruction-following for structured medical prompts
- No API cost

**Migration to Google Gemini:**
- Superior medical reasoning capabilities
- `gemini-2.5-flash` provides fast, high-quality responses
- Free tier with generous limits
- Better availability and reliability than HuggingFace free endpoints
- Fallback chain (`flash` → `flash-lite` → `2.0-flash`) ensures resilience

### 13.5 Why fp32?

**DenseNet121's architecture makes fp16 (AMP) unsafe.**

Technical root cause: DenseNet121 uses dense concatenation — feature maps from ALL preceding layers within a dense block are concatenated along the channel dimension. With 121 layers, the intermediate concatenated tensors can exceed the fp16 representable range (max ~65,504), producing `NaN` in the forward pass.

This was confirmed empirically:
- With `torch.amp.autocast('cuda')` → **all predictions = NaN**
- Without autocast (fp32) → correct predictions
- `train.py` contains an explicit comment warning about this

fp32 is feasible because:
- Peak VRAM is only 2.2 GB (batch=8, 320×320)
- GTX 1650 has 4 GB total → 1.8 GB headroom
- No performance benefit from fp16 at this batch size (memory-bound, not compute-bound)

---

## 14. FINAL OUTPUT

### 14.1 Best Mean AUC

$$\text{Mean AUC} = 0.8273$$

Computed as the arithmetic mean of the 5 per-class AUC values on the official CheXpert validation set (234 radiologist-labeled images).

### 14.2 Per-Class Final Results

| Label | AUC | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|---|
| Atelectasis | 0.7504 | 0.6410 | 0.5922 | 0.4841 | 0.7625 |
| Cardiomegaly | 0.7503 | 0.7179 | 0.5714 | 0.5116 | 0.6471 |
| Consolidation | 0.8393 | 0.6838 | 0.4394 | 0.2929 | 0.8788 |
| **Edema** | **0.9382** | **0.8632** | **0.6981** | **0.6066** | **0.8222** |
| Pleural Effusion | 0.8584 | 0.7650 | 0.6667 | 0.5612 | 0.8209 |
| **Overall** | **0.8273** | **0.7342** | **0.5924** | **0.4809** | **0.7713** |

### 14.3 Saved Weight Files

| File | Size | Contents |
|---|---|---|
| `best_densenet121_phase1.pth` | 28 MB | Best model state_dict (epoch 4, AUC 0.8273) |
| `checkpoint_phase1.pth` | 81 MB | Full state (model + optimizer + scheduler + metadata) |

### 14.4 System State

- Streamlit app running at `http://localhost:8501`
- Three-tab UI: Scan & Diagnose, Model Performance, About
- Gemini AI chat functional with auto-loaded API key
- Grad-CAM++ heatmaps generating correctly
- All evaluation plots saved to `results/`

---

## Status: Phase 1 Complete — Ready for Phase 2 Fine-tuning.
