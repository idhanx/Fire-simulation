"""
Forest Fire — Training Pipeline (Final)

Key techniques:
  - Spatial train/val split (top 70% / bottom 30%)
  - Patch extraction (128x128, stride 64)
  - Augmentation (flips + 90-deg rotations)
  - Label smoothing (0.9/0.05)
  - Input noise (std=0.01)
  - Capped pos_weight (max 10.0)
  - Early stopping + LR scheduler
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.cnn import FireModel

# ── Config ───────────────────────────────────────────────────
PATCH      = 128
STRIDE     = 64
BATCH      = 4
EPOCHS     = 30
LR         = 1e-3
PATIENCE   = 7
TRAIN_FRAC = 0.7     # spatial split fraction
MAX_POS_W  = 10.0    # cap class weight to avoid over-correction
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Patch extraction (spatial) ───────────────────────────────
def extract_patches(features, labels, row_start, row_end):
    """Extract overlapping patches from rows [row_start, row_end)."""
    _, H, W = features.shape
    pf, pl = [], []
    for i in range(row_start, min(row_end, H - PATCH + 1), STRIDE):
        for j in range(0, W - PATCH + 1, STRIDE):
            pf.append(features[:, i:i+PATCH, j:j+PATCH])
            pl.append(labels[:, i:i+PATCH, j:j+PATCH])
    return np.array(pf), np.array(pl)


# ── Dataset ──────────────────────────────────────────────────
class PatchDataset(Dataset):
    """Patches with on-the-fly augmentation + label smoothing + input noise."""

    def __init__(self, feats, labs, augment=True):
        self.x = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(labs, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.augment:
            # Random flips
            if torch.rand(1) > 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) > 0.5:
                x, y = x.flip(-2), y.flip(-2)
            # Random 90-deg rotation
            k = torch.randint(0, 4, (1,)).item()
            if k:
                x = torch.rot90(x, k, [-2, -1])
                y = torch.rot90(y, k, [-2, -1])
            # Label smoothing: hard 0/1 → 0.05/0.95
            y = y * 0.9 + 0.05
            # Input noise
            x = x + 0.01 * torch.randn_like(x)
        return x, y


# ── Metrics helper ───────────────────────────────────────────
def binary_f1(preds, targets, thresh=0.5):
    """Quick F1 from logits + targets (both on device)."""
    with torch.no_grad():
        p = (torch.sigmoid(preds) > thresh).float()
        tp = ((p == 1) & (targets >= 0.5)).sum().item()
        fp = ((p == 1) & (targets < 0.5)).sum().item()
        fn = ((p == 0) & (targets >= 0.5)).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


# ── Main ─────────────────────────────────────────────────────
def train():
    print("=" * 55)
    print("  FOREST FIRE — TRAINING")
    print("=" * 55)

    # Load
    features = np.load("data/processed/features.npy")  # (2, 512, 512)
    labels   = np.load("data/labels/labels.npy")        # (1, 512, 512)
    _, H, W  = features.shape
    split    = int(H * TRAIN_FRAC)

    print(f"\n  Data      : {features.shape}")
    print(f"  Spatial   : train rows 0-{split}, val rows {split}-{H}")

    # Patches
    tr_f, tr_l = extract_patches(features, labels, 0, split)
    va_f, va_l = extract_patches(features, labels, split, H)
    print(f"  Patches   : {len(tr_f)} train, {len(va_f)} val (no overlap)")

    train_dl = DataLoader(PatchDataset(tr_f, tr_l, augment=True),
                          batch_size=BATCH, shuffle=True)
    val_dl   = DataLoader(PatchDataset(va_f, va_l, augment=False),
                          batch_size=BATCH, shuffle=False)

    # Class weight (capped)
    pos = (labels == 1).sum()
    neg = labels.size - pos
    raw_w = neg / max(pos, 1)
    pw = min(raw_w, MAX_POS_W)
    print(f"  pos_weight: {raw_w:.1f} → capped to {pw:.1f}")

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = FireModel().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Device    : {device}  |  Params: {params:,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float32).to(device)
    )
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=5)

    # Train loop
    print(f"\n  Epochs: {EPOCHS} (early stop={PATIENCE})")
    print("-" * 55)

    best_loss, wait = float("inf"), 0
    os.makedirs("outputs", exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        t_loss, t_f1_sum, t_n = 0.0, 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            t_loss += loss.item() * xb.size(0)
            t_f1_sum += binary_f1(out, yb) * xb.size(0)
            t_n += xb.size(0)

        # ── Val ──
        model.eval()
        v_loss, v_f1_sum, v_n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                v_loss += criterion(out, yb).item() * xb.size(0)
                v_f1_sum += binary_f1(out, yb) * xb.size(0)
                v_n += xb.size(0)

        tl, tf = t_loss/t_n, t_f1_sum/t_n
        vl, vf = v_loss/v_n, v_f1_sum/v_n

        print(f"  {ep:2d}/{EPOCHS}  TrL={tl:.4f} F1={tf:.3f}  |  "
              f"VaL={vl:.4f} F1={vf:.3f}  lr={optim.param_groups[0]['lr']:.0e}")

        sched.step(vl)

        if vl < best_loss:
            best_loss, wait = vl, 0
            torch.save(model.state_dict(), "outputs/model.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\n  Early stop at epoch {ep}")
                break

    print("-" * 55)
    print(f"  Model saved → outputs/model.pth  (best val_loss={best_loss:.4f})")


if __name__ == "__main__":
    train()