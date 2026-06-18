# ================================================================
# CYGNSS-SM-Net  –  Cross-Region Generalisation Evaluation
# ================================================================
# Train on one basin and test on the other to assess
# out-of-region generalisation.
#
# Usage:
#   python scripts/cross_region.py --source godavari --target narmada_tapti
#   python scripts/cross_region.py --source narmada_tapti --target godavari
# ================================================================

import argparse
import copy
import json
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
parser = argparse.ArgumentParser(description="CYGNSS-SM-Net Cross-Region Evaluation")
parser.add_argument("--source",   type=str, required=True,
                    choices=list(REGION_PATHS.keys()),
                    help="Region to train on")
parser.add_argument("--target",   type=str, required=True,
                    choices=list(REGION_PATHS.keys()),
                    help="Region to test on")
parser.add_argument("--epochs",   type=int, default=100)
parser.add_argument("--batch",    type=int, default=64)
parser.add_argument("--seed",     type=int, default=42)
parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
parser.add_argument("--patience", type=int, default=25)
args = parser.parse_args()

assert args.source != args.target, "Source and target regions must differ"

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

run_name = f"{args.source}_to_{args.target}"
save_dir = os.path.join(args.save_dir, "cross_region", run_name)
os.makedirs(save_dir, exist_ok=True)
print(f"Device: {device}  |  {args.source.upper()} -> {args.target.upper()}")


# ================================================================
# Helper: load and preprocess one region
# ================================================================
def load_region(region_name):
    cfg     = REGION_PATHS[region_name]
    df      = pd.read_csv(cfg["csv_path"])
    ddm_raw = load_npz_chunks(cfg["ddm_dir"], "ddm").astype(np.float32)
    wf_raw  = load_npz_chunks(cfg["wf_dir"],  "waveforms").astype(np.float32)

    df, ddm_raw, wf_raw = apply_qc(df, ddm_raw, wf_raw)
    print(f"  {region_name}: {len(df)} samples after QC")

    ddm_multi               = preprocess_ddm(ddm_raw)
    wf_proc, wf_phys_raw, _ = preprocess_waveforms(wf_raw)
    tab_raw, am_pm          = build_tabular_features(df)
    sm_target               = df["soil_moisture"].values.astype(np.float32)

    return dict(
        name=region_name, df=df,
        ddm=ddm_multi, wf=wf_proc,
        tab_raw=tab_raw, wf_phys_raw=wf_phys_raw,
        am_pm=am_pm, target=sm_target,
    )


# ================================================================
# Load both regions
# ================================================================
print("\nLoading source region ...")
src = load_region(args.source)
print("Loading target region ...")
tgt = load_region(args.target)

# ================================================================
# Normalise using source training set statistics only
# ================================================================
sm_tercile         = pd.qcut(src["target"], q=3, labels=False, duplicates="drop")
indices            = np.arange(len(src["df"]))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED,
    shuffle=True, stratify=sm_tercile)

src_tab_train = src["tab_raw"][train_idx]
tab_mu        = src_tab_train.mean(0, keepdims=True).astype(np.float32)
tab_sd        = src_tab_train.std(0,  keepdims=True).astype(np.float32)
tab_sd[tab_sd < 1e-6] = 1.0

src_phys_train = src["wf_phys_raw"][train_idx]
wp_mu          = src_phys_train.mean(0, keepdims=True).astype(np.float32)
wp_sd          = src_phys_train.std(0,  keepdims=True).astype(np.float32)
wp_sd[wp_sd < 1e-6] = 1.0

# Apply to both source and target
src_tab,  _, _ = normalise_tabular(src["tab_raw"],     mu=tab_mu, sd=tab_sd)
tgt_tab,  _, _ = normalise_tabular(tgt["tab_raw"],     mu=tab_mu, sd=tab_sd)

src_phys = np.clip((src["wf_phys_raw"] - wp_mu) / wp_sd, -10, 10).astype(np.float32)
tgt_phys = np.clip((tgt["wf_phys_raw"] - wp_mu) / wp_sd, -10, 10).astype(np.float32)

# Save normalisation stats
np.save(os.path.join(save_dir, "tab_mean.npy"),     tab_mu)
np.save(os.path.join(save_dir, "tab_std.npy"),      tab_sd)
np.save(os.path.join(save_dir, "wf_phys_mean.npy"), wp_mu)
np.save(os.path.join(save_dir, "wf_phys_std.npy"),  wp_sd)

# ================================================================
# DataLoaders
# ================================================================
train_ds = CYGNSSDataset(train_idx, src["ddm"], src["wf"],
                         src_tab, src_phys, src["am_pm"], src["target"])
val_ds   = CYGNSSDataset(val_idx,   src["ddm"], src["wf"],
                         src_tab, src_phys, src["am_pm"], src["target"])
tgt_idx  = np.arange(len(tgt["target"]))
tgt_ds   = CYGNSSDataset(tgt_idx, tgt["ddm"], tgt["wf"],
                         tgt_tab, tgt_phys, tgt["am_pm"], tgt["target"])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=N_WORK, pin_memory=USE_PIN)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=N_WORK, pin_memory=USE_PIN)
tgt_loader   = DataLoader(tgt_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=N_WORK, pin_memory=USE_PIN)

# ================================================================
# Model + Optimizer
# ================================================================
H_, W_ = src["ddm"].shape[2], src["ddm"].shape[3]
model  = CYGNSSSMNet(tab_dim=src_tab.shape[1],
                     phys_dim=src_phys.shape[1],
                     H=H_, W=W_).to(device)
print(f"Parameters: {model.count_parameters():,}")

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
# Training Loop
# ================================================================
best_r2  = -999.0
no_imp   = 0
history  = []

for epoch in range(EPOCHS):
    W = get_loss_weights(epoch)
    model.ddm.pk.temp = min(1.0 + (epoch / 80.0) * 19.0, 20.0)

    model.train()
    tot_loss, n_batches = 0.0, 0

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

    if np.isnan(np.array(preds)).any():
        continue

    m    = compute_metrics(trues, preds)
    unc_ = float(np.mean(uncs))
    ph   = 1 if epoch < 20 else (2 if epoch < 50 else 3)

    history.append({"epoch": epoch+1, "phase": ph,
                    "src_val_R2": round(m["R2"], 5),
                    "src_val_RMSE": round(m["RMSE"], 5)})

    if m["R2"] > best_r2:
        best_r2 = m["R2"]
        no_imp  = 0
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    else:
        no_imp += 1

    print(f"Ep{epoch+1:3d}[Ph{ph}] "
          f"SrcVal RMSE:{m['RMSE']:.5f} R2:{m['R2']:.5f} "
          f"KGE:{m['KGE']:.5f} | BestR2:{best_r2:.5f}")

    if no_imp >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

pd.DataFrame(history).to_csv(
    os.path.join(save_dir, "training_history.csv"), index=False)

# ================================================================
# Target Region Evaluation
# ================================================================
print(f"\n{'='*60}")
print(f"  Evaluating on TARGET: {args.target.upper()}")
print(f"{'='*60}")

model.load_state_dict(
    torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device))
model.eval()

ap, at, au, agroup = [], [], [], []
with torch.no_grad():
    for xd, xw, xt, xp, ampm, yb in tgt_loader:
        xd   = xd.to(device);   xw   = xw.to(device)
        xt   = xt.to(device);   xp   = xp.to(device)
        ampm = ampm.to(device)
        gam, nu, al, be = model(xd, xw, xt, xp, ampm)
        gam = torch.nan_to_num(gam, nan=0.25)
        unc = be / (nu * (al - 1) + 1e-6)
        ap.extend(gam.cpu().numpy())
        at.extend(yb.numpy())
        au.extend(unc.cpu().numpy())
        agroup.extend(ampm.cpu().numpy())

ap     = np.array(ap);     at     = np.array(at)
au     = np.array(au);     agroup = np.array(agroup)
overall = compute_metrics(at, ap)
overall["UNC"] = float(np.mean(au))

print("\nOverall target metrics:")
for k, v in overall.items():
    print(f"  {k}: {v:.5f}")

# Save predictions
tdf = tgt["df"].copy().reset_index(drop=True)
tdf["pred_sm"]       = ap
tdf["true_sm"]       = at
tdf["uncertainty"]   = au
tdf["source_region"] = args.source
tdf["target_region"] = args.target
tdf.to_csv(os.path.join(save_dir, "target_predictions.csv"), index=False)

# AM/PM breakdown
ampm_rows = []
for g, label in [(0, "AM"), (1, "PM")]:
    mask = agroup == g
    if mask.sum() < 10:
        continue
    m = compute_metrics(at[mask], ap[mask])
    ampm_rows.append({"Window": label, "Samples": int(mask.sum()), **m})
pd.DataFrame(ampm_rows).to_csv(
    os.path.join(save_dir, "ampm_results.csv"), index=False)

# Year-wise breakdown
tdf["year"] = tdf["day"].astype(str).str[:4]
yr_rows = []
for yr in sorted(tdf["year"].unique()):
    t = tdf[tdf["year"] == yr]
    if len(t) < 20:
        continue
    m = compute_metrics(t["true_sm"], t["pred_sm"])
    yr_rows.append({"Year": str(yr), "Samples": len(t), **m})
pd.DataFrame(yr_rows).to_csv(
    os.path.join(save_dir, "year_results.csv"), index=False)

# Save summary JSON
summary = {"source": args.source, "target": args.target,
           "overall": overall, "ampm": ampm_rows, "yearly": yr_rows}
with open(os.path.join(save_dir, "summary.json"), "w") as f:
    json.dump({k: (v if not isinstance(v, float) else round(v, 6))
               for k, v in summary.items()}, f, indent=2, default=str)

print(f"\nAll outputs saved to: {save_dir}")
