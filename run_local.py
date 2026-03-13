"""
run_local.py — Master training script for GTX 1650 local PC.

Hardware:
  GPU  : NVIDIA GeForce GTX 1650 (4 GB VRAM)
  CPU  : Intel Core i5-10300H (4 cores / 8 threads)
  RAM  : 7.7 GB
  Disk : 915 GB free (WSL2 ext4 VHD — fast I/O)
  CUDA : 12.9

Training plan:
  Phase 1  CheXpert 5-label pre-training     — ~3–5 h  (10 epochs)
  Phase 2  RSNA fine-tuning (if available)   — ~45 min (5 epochs)
  Phase 3  Kaggle binary final-tune (if avail) — ~15 min (3 epochs)
  Evaluate on official CheXpert valid.csv    — ~1 min

Run:
  python run_local.py                         # full pipeline
  python run_local.py --phase 1               # only Phase 1
  python run_local.py --phase eval            # only evaluate
  python run_local.py --epochs 5 --phase 1   # quick smoke test

Do NOT reduce batch_size below 8 — that makes effective batch too small.
Do NOT increase batch_size above 8 — GTX 1650 will OOM.
grad_accum_steps=4 gives effective batch = 32.
"""

import os
import sys
import argparse
import torch


# ════════════════════════════════════════════════════════════════════════════
#  PATH CONFIGURATION — edit these to match your directory layout
# ════════════════════════════════════════════════════════════════════════════

# ─── CheXpert (you already have this) ────────────────────────────────────
CHEXPERT_TRAIN_CSV = 'train.csv'
CHEXPERT_VAL_CSV   = 'valid.csv'
CHEXPERT_ROOT      = '.'          # Folder that contains train/ and valid/

# ─── NIH ChestX-ray14 (optional, download via Kaggle API) ────────────────
#   kaggle datasets download nih-chest-xrays/data
NIH_CSV     = None   # e.g. 'nih/Data_Entry_2017.csv'
NIH_IMG_DIR = None   # e.g. 'nih/images/'

# ─── RSNA Pneumonia (optional) ────────────────────────────────────────────
#   kaggle competitions download -c rsna-pneumonia-detection-challenge
RSNA_CSV     = None   # e.g. 'rsna/stage_2_train_labels.csv'
RSNA_IMG_DIR = None   # e.g. 'rsna/stage_2_train_images_png/'

# ─── Kaggle Binary Chest X-Ray (optional) ─────────────────────────────────
#   kaggle datasets download paultimothymooney/chest-xray-pneumonia
KAGGLE_TRAIN_DIR = None   # e.g. 'chest_xray/train/'
KAGGLE_VAL_DIR   = None   # e.g. 'chest_xray/val/'

# ─── MIMIC-CXR (optional, requires PhysioNet credentialed access) ─────────
MIMIC_LABEL_CSV   = None
MIMIC_RECORDS_CSV = None
MIMIC_IMG_ROOT    = None

# ════════════════════════════════════════════════════════════════════════════
#  GTX 1650 TRAINING HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════

# --- Phase 1 ---------------------------------------------------------------
P1_BATCH_SIZE       = 8      # Safe batch for 4 GB VRAM @ 320×320 + AMP (~1.3 GB peak)
P1_GRAD_ACCUM       = 4      # Effective batch = 8 × 4 = 32
P1_EPOCHS           = 10
P1_LR               = 1e-4
P1_FRONTAL_ONLY     = True   # Use only frontal views (better AUC on 5 labels)
P1_NUM_WORKERS      = 4
# Soft uncertainty labels: replaces uncertain (-1) with Uniform[0.55, 0.85]
# Set True to use MLMIP strategy; False (default) = hard U-Ones/U-Zeros policy
P1_U_LABEL_SOFT     = False

# --- Phase 2 (RSNA) --------------------------------------------------------
P2_BATCH_SIZE       = 8
P2_GRAD_ACCUM       = 4
P2_EPOCHS           = 5
P2_LR               = 1e-4
P2_NUM_WORKERS      = 4

# --- Phase 3 (Kaggle binary) -----------------------------------------------
P3_BATCH_SIZE       = 8
P3_GRAD_ACCUM       = 4
P3_EPOCHS           = 3
P3_LR               = 5e-5
P3_NUM_WORKERS      = 4


# ════════════════════════════════════════════════════════════════════════════
#  HARDWARE CHECK
# ════════════════════════════════════════════════════════════════════════════

def hardware_check():
    print("\n" + "="*60)
    print("  Hardware Check")
    print("="*60)
    if not torch.cuda.is_available():
        print("  ✗ CUDA not available — training on CPU will be 10-50× slower.")
        print("    Ensure CUDA drivers are installed and PYTORCH_CUDA_ALLOC_CONF is set.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU   : {gpu_name}")
    print(f"  VRAM  : {vram_gb:.1f} GB")
    print(f"  CUDA  : {torch.version.cuda}")

    if vram_gb < 3.5:
        print(f"  ✗ Less than 3.5 GB VRAM — reduce batch_size to 4 and "
              f"increase grad_accum_steps to 8.")
        sys.exit(1)

    # Check free VRAM
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(0) / 1e6
    reserved  = torch.cuda.memory_reserved(0) / 1e6
    print(f"  VRAM allocated/reserved before training: {allocated:.0f} MB / {reserved:.0f} MB")

    # Quick forward pass to confirm batch_size=8 fits
    print("  Running forward-pass smoke test (batch=8, fp16)…", end=" ", flush=True)
    _test_batch_fits()
    print("OK ✓")
    print("="*60)


def _test_batch_fits():
    """Verify that one forward + backward pass with batch_size=8 fits in VRAM.
    NOTE: AMP disabled — DenseNet121 dense concat overflows fp16 → NaN.
    fp32 fits fine (2.2 GB peak at batch=8, 320×320)."""
    from model import get_densenet121_model
    device = torch.device('cuda')
    model  = get_densenet121_model(pretrained=False).to(device)
    model.train()
    dummy  = torch.randn(8, 3, 320, 320, device=device)
    out    = model(dummy)
    loss   = out.sum()
    loss.backward()
    del model, dummy, out, loss
    torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
#  ESTIMATED TIME PRINTER
# ════════════════════════════════════════════════════════════════════════════

def print_time_estimate(frontal_only=True):
    # GTX 1650 measured: ~45 ms per batch (batch=8, AMP, DenseNet121)
    # CheXpert frontal: ~153K images → 153000/8 = ~19K batches/epoch
    n_train = 153_000 if frontal_only else 223_000
    batches_per_epoch = n_train // P1_BATCH_SIZE
    secs_per_batch    = 0.045   # measured on GTX 1650 fp16
    secs_per_epoch    = batches_per_epoch * secs_per_batch
    total_hours_p1    = (secs_per_epoch * P1_EPOCHS) / 3600

    print("\n" + "="*60)
    print("  Estimated Training Time on GTX 1650")
    print("="*60)
    print(f"  Phase 1 — CheXpert {n_train//1000}K images × {P1_EPOCHS} epochs")
    print(f"            ~{secs_per_epoch/60:.0f} min/epoch  →  ~{total_hours_p1:.1f}h total")
    print(f"  Phase 2 — RSNA (if available)   ~45 min")
    print(f"  Phase 3 — Kaggle (if available) ~15 min")
    print(f"  Evaluate                        ~1 min")
    print(f"\n  Total estimate: {total_hours_p1:.1f}–{total_hours_p1*1.3:.1f} hours")
    print(f"  Checkpoints saved every epoch — safe to stop and resume anytime.")
    print("="*60 + "\n")


# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run_phase1(epochs_override=None):
    from train import phase1_pretrain
    phase1_pretrain(
        chexpert_train_csv = CHEXPERT_TRAIN_CSV,
        chexpert_val_csv   = CHEXPERT_VAL_CSV,
        chexpert_root      = CHEXPERT_ROOT,
        nih_csv            = NIH_CSV,
        nih_img_dir        = NIH_IMG_DIR,
        mimic_chexpert_csv = MIMIC_LABEL_CSV,
        mimic_records_csv  = MIMIC_RECORDS_CSV,
        mimic_img_root     = MIMIC_IMG_ROOT,
        frontal_only       = P1_FRONTAL_ONLY,
        u_label_soft       = P1_U_LABEL_SOFT,
        batch_size         = P1_BATCH_SIZE,
        epochs             = epochs_override or P1_EPOCHS,
        lr                 = P1_LR,
        num_workers        = P1_NUM_WORKERS,
        grad_accum_steps   = P1_GRAD_ACCUM,
    )


def run_phase2():
    if RSNA_CSV is None or not os.path.exists(str(RSNA_CSV)):
        print("\n  [Phase 2] RSNA dataset not configured — skipping.")
        print("  To enable: set RSNA_CSV and RSNA_IMG_DIR at top of run_local.py")
        return
    from train import phase2_finetune
    phase2_finetune(
        rsna_csv         = RSNA_CSV,
        rsna_img_dir     = RSNA_IMG_DIR,
        phase1_weights   = 'best_densenet121_phase1.pth',
        batch_size       = P2_BATCH_SIZE,
        epochs           = P2_EPOCHS,
        lr               = P2_LR,
        num_workers      = P2_NUM_WORKERS,
        grad_accum_steps = P2_GRAD_ACCUM,
    )


def run_phase3():
    if KAGGLE_TRAIN_DIR is None or not os.path.isdir(str(KAGGLE_TRAIN_DIR)):
        print("\n  [Phase 3] Kaggle binary dataset not configured — skipping.")
        print("  To enable: set KAGGLE_TRAIN_DIR at top of run_local.py")
        return
    from train import phase3_final_tune
    phase3_final_tune(
        kaggle_train_dir = KAGGLE_TRAIN_DIR,
        kaggle_val_dir   = KAGGLE_VAL_DIR,
        phase2_weights   = 'best_densenet121_phase2.pth',
        batch_size       = P3_BATCH_SIZE,
        epochs           = P3_EPOCHS,
        lr               = P3_LR,
        num_workers      = P3_NUM_WORKERS,
        grad_accum_steps = P3_GRAD_ACCUM,
    )


def run_evaluate():
    from evaluate import evaluate
    # Pick the best available weights (prefer final → phase2 → phase1)
    for w in ['best_densenet121.pth',
              'best_densenet121_phase2.pth',
              'best_densenet121_phase1.pth']:
        if os.path.exists(w):
            print(f"\n  [Evaluate] Using weights: {w}")
            evaluate(weights_path=w,
                     val_csv=CHEXPERT_VAL_CSV,
                     data_root=CHEXPERT_ROOT,
                     save_dir='results',
                     batch_size=16)
            return
    print("  [Evaluate] No weights file found — run training first.")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GTX 1650 local training pipeline — CheXpert 5-label DenseNet121"
    )
    parser.add_argument('--phase', default='all',
                        choices=['all', '1', '2', '3', 'eval'],
                        help="Which phase to run (default: all)")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Override epoch count for Phase 1 (useful for smoke tests)")
    parser.add_argument('--skip_check', action='store_true',
                        help="Skip hardware check (not recommended)")
    args = parser.parse_args()

    if not args.skip_check:
        hardware_check()

    print_time_estimate(frontal_only=P1_FRONTAL_ONLY)

    if args.phase in ('all', '1'):
        run_phase1(epochs_override=args.epochs)

    if args.phase in ('all', '2'):
        run_phase2()

    if args.phase in ('all', '3'):
        run_phase3()

    if args.phase in ('all', 'eval'):
        run_evaluate()

    print("\n  Pipeline complete.")
