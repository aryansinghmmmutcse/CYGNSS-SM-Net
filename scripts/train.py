# ================================================================
# CYGNSS-SM-Net  –  Main Training Script
# ================================================================
# Usage:
#   python scripts/train.py --region narmada_tapti
#   python scripts/train.py --region godavari
#
# Edit configs/paths.py to set your data paths before running.
# ================================================================

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import CYGNSSSMNet
from src.losses import compute_total_loss, get_loss_weights, compute_metrics
from src.data_utils import (
    load_npz_chunks, apply_qc,
    preprocess_ddm, preprocess_waveforms,
    build_tabular_features, normalise_tabular,
    CYGNSSDataset,
)
from configs.paths import REGION_PATHS, DEFAULT_SAVE_DIR

# ================================================================
# Argument Parser
# ================================================================
parser = argparse.ArgumentParser(description="Train CYGNSS-SM-Net")
parser.add_argument("--region",    type=str,   default="narmada_tapti",
                    choices=list(REGION_PATHS.keys()),
                    help="Region to train on")
parser.add_argument("--epochs",    type=int,   default=100)
parser.add_argument("--batch",     type=int,   default=64)
parser.add_argument("--seed",      type=int,   default=42)
parser.add_argument("--save_dir",  type=str,   default=DEFAULT_SAVE_DIR)
parser.add_argument("--patience",  type=int,   default=25)
args = parser.parse_args()

SEED     = args.seed
EPOCHS   = args.epochs
BATCH    = args.batch
WARMUP   = 10
PATIENCE = args.patience

torch.manual_seed(SEED)
np.random.seed(SEED)

device  = "cuda" if torch.cuda.is_available() else "cpu"
USE_PIN = device == "cuda"
N_WORK  = 2 if device == "cuda" else 0
print(f"Device: {device}  |  Region: {args.region}")

save_dir = os.path.join(args.save_dir, args.region)
os.makedirs(save_dir, exist_ok=True)

# ================================================================
# 1. Load Data
# ================================================================
cfg     = REGION_PATHS[args.region]
df      = pd.read_csv(cfg["csv_path"])
ddm_raw = load_npz_chunks(cfg["ddm_dir"], "ddm").astype(np.float32)
wf_raw  = load_npz_chunks(cfg["wf_dir"],  "waveforms").astype(np.float32)
print(f"Loaded  CSV:{df.shape}  DDM:{ddm_raw.shape}  WF:{wf_raw.shape}")

# ================================================================
# 2. Quality Control
# ================================================================
df, ddm_raw, wf_raw = apply_qc(df, ddm_raw, wf_raw)
print(f"After QC: {len(df)} samples  "
      f"SM:[{df['soil_moisture'].min():.4f}, {df['soil_moisture'].max():.4f}]")

# ================================================================
# 3. Preprocessing
# ================================================================
print("Preprocessing DDM ...")
ddm_multi = preprocess_ddm(ddm_raw)

print("Preprocessing waveforms ...")
wf_proc, wf_phys_norm, wf_stats = preprocess_waveforms(wf_raw)

print("Building tabular features ...")
tab_raw, am_pm = build_tabular_features(df)
sm_target      = df["soil_moisture"].values.astype(np.float32)

# ================================================================
# 4. Stratified Train / Validation Split
# ================================================================
sm_tercile         = pd.qcut(sm_target, q=3, labels=False, duplicates="drop")
indices            = np.arange(len(df))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED,
    shuffle=True, stratify=sm_tercile)
print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

# Fit normalisation on training set only
tab_train          = tab_raw[train_idx]
tab_norm, tab_mu, tab_sd = normalise_tabular(tab_raw,
    mu=tab_train.mean(0, keepdims=True).astype(np.float32),
    sd=tab_train.std(0,  keepdims=True).astype(np.float32))

# ================================================================
# 5. DataLoaders
# ================================================================
train_ds = CYGNSSDataset(train_idx, ddm_multi, wf_proc,
                         tab_norm, wf_phys_norm, am_pm, sm_target)
val_ds   = CYGNSSDataset(val_idx,   ddm_multi, wf_proc,
                         tab_norm, wf_phys_norm, am_pm, sm_target)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=N_WORK, pin_memory=USE_PIN)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=N_WORK, pin_memory=USE_PIN)

# ================================================================
# 6. Model
# ================================================================
H_, W_ = ddm_multi.shape[2], ddm_multi.shape[3]
model  = CYGNSSSMNet(tab_dim=tab_norm.shape[1],
                     phys_dim=wf_phys_norm.shape[1],
                     H=H_, W=W_).to(device)
print(f"Parameters: {model.count_parameters():,}")

# ================================================================
# 7. Optimizer  (two LR groups: backbone vs head/projection)
# ================================================================
backbone_params, head_params = [], []
for name, p in model.named_parameters():
    if any(k in name for k in ["head", "proj", "film", "phys"]):
        head_params.append(p)
    else:
        backbone_params.append(p)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 5e-4},
    {"params": head_params,     "lr": 1.5e-3},
], weight_decay=5e-4)

def lr_lam(ep):
    if ep < WARMUP:
        return (ep + 1) / WARMUP
    return 0.5 * (1 + np.cos(np.pi * (ep - WARMUP) / (EPOCHS - WARMUP)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lam)

# ================================================================
# 8. Training Loop
# ================================================================
best_r2   = -999.0
best_met  = {}
no_imp    = 0
history   = []

for epoch in range(EPOCHS):
    W = get_loss_weights(epoch)

    # Curriculum temperature for soft peak localizer
    model.ddm.pk.temp = min(1.0 + (epoch / 80.0) * 19.0, 20.0)

    # ── Train ──────────────────────────────────────────────────
    model.train()
    tot_loss  = 0.0
    n_batches = 0

    for xd, xw, xt, xp, ampm, yb in train_loader:
        xd   = xd.to(device);   xw   = xw.to(device)
        xt   = xt.to(device);   xp   = xp.to(device)
        ampm = ampm.to(device); yb   = yb.to(device)

        if any(torch.isnan(t).any() for t in [xd, xw, xt, xp, yb]):
            continue

        gam, nu, al, be = model(xd, xw, xt, xp, ampm)
        if any(torch.isnan(t).any() for t in [gam, nu, al, be]):
            continue

        loss = compute_total_loss(gam, nu, al, be, yb, W)
        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss  += loss.item()
        n_batches += 1

    scheduler.step()

    # ── Validate ────────────────────────────────────────────────
    model.eval()
    preds, trues, uncs = [], [], []

    with torch.no_grad():
        for xd, xw, xt, xp, ampm, yb in val_loader:
            xd   = xd.to(device);   xw   = xw.to(device)
            xt   = xt.to(device);   xp   = xp.to(device)
            ampm = ampm.to(device)
            gam, nu, al, be = model(xd, xw, xt, xp, ampm)
            gam = torch.nan_to_num(gam, nan=0.25)
            unc = be / (nu * (al - 1) + 1e-6)
            preds.extend(gam.cpu().numpy())
            trues.extend(yb.numpy())
            uncs.extend(unc.cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    if np.isnan(preds).any():
        print(f"Ep{epoch+1}: NaN in predictions, skipping")
        continue

    m    = compute_metrics(trues, preds)
    unc_ = float(np.mean(uncs))
    ph   = 1 if epoch < 20 else (2 if epoch < 50 else 3)

    history.append({
        "epoch":      epoch + 1,
        "phase":      ph,
        "train_loss": round(tot_loss / max(n_batches, 1), 6),
        "val_RMSE":   round(m["RMSE"], 6),
        "val_R2":     round(m["R2"],   6),
        "val_NSE":    round(m["NSE"],  6),
        "val_KGE":    round(m["KGE"],  6),
        "val_unc":    round(unc_,      6),
    })

    if m["R2"] > best_r2:
        best_r2  = m["R2"]
        best_met = dict(**m, UNC=unc_)
        no_imp   = 0
        torch.save(model.state_dict(),
                   os.path.join(save_dir, "best_model.pt"))
    else:
        no_imp += 1

    print(f"Ep{epoch+1:3d}[Ph{ph}] "
          f"Loss:{tot_loss/max(n_batches,1):.5f} | "
          f"RMSE:{m['RMSE']:.5f}  R2:{m['R2']:.5f}  "
          f"NSE:{m['NSE']:.5f}  KGE:{m['KGE']:.5f} | "
          f"Unc:{unc_:.5f} | BestR2:{best_r2:.5f}")

    if no_imp >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

# ================================================================
# 9. Save Outputs
# ================================================================
pd.DataFrame(history).to_csv(
    os.path.join(save_dir, "training_history.csv"), index=False)

np.save(os.path.join(save_dir, "tab_mean.npy"),      tab_mu)
np.save(os.path.join(save_dir, "tab_std.npy"),       tab_sd)
np.save(os.path.join(save_dir, "wf_phys_mean.npy"),  wf_stats["wp_mu"])
np.save(os.path.join(save_dir, "wf_phys_std.npy"),   wf_stats["wp_sd"])

print("\n======== BEST VALIDATION RESULTS ========")
for k, v in best_met.items():
    print(f"  {k}: {v:.5f}")
print(f"\nAll outputs saved to: {save_dir}")
