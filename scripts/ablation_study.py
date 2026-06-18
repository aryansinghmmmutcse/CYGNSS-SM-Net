# ================================================================
# CYGNSS-SM-Net  –  ABLATION STUDY
# ================================================================
# Matches Table 6 in the paper:
#   Full model
#   No DDM branch
#   No waveform branch
#   No tabular branch
#   No AM/PM conditioning
# ================================================================

import os, glob, warnings, time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────
csv_path = "/content/drive/MyDrive/Overall_IGPlane/IGP_PunjabHaryana/samples_IndoGangetic_QC_stream.csv"
ddm_dir  = "/content/drive/MyDrive/Overall_IGPlane/IGP_PunjabHaryana/chunks_ddm"
wf_dir   = "/content/drive/MyDrive/Overall_IGPlane/IGP_PunjabHaryana/chunks_wf"
save_dir = "/content/IGP/cygnss_sm_ablation"
os.makedirs(save_dir, exist_ok=True)

device  = "cuda" if torch.cuda.is_available() else "cpu"
USE_PIN = device == "cuda"
N_WORK  = 2 if device == "cuda" else 0
EPOCHS  = 100
BATCH   = 64
SEED    = 42
print("Device:", device)

# ================================================================
# ABLATION CONFIG
# ================================================================
@dataclass
class AblationConfig:
    name:        str
    label:       str
    use_ddm:     bool   # if False: DDM branch replaced with zeros
    use_wf:      bool   # if False: waveform branch replaced with zeros
    use_tab:     bool   # if False: tabular branch replaced with zeros
    use_ampm:    bool   # if False: AM/PM FiLM conditioning disabled

# Exactly the 5 variants reported in Table 6
ABLATIONS = [
    AblationConfig("Full",    "Full model",          use_ddm=True,  use_wf=True,  use_tab=True,  use_ampm=True),
    AblationConfig("No_DDM",  "No DDM branch",       use_ddm=False, use_wf=True,  use_tab=True,  use_ampm=True),
    AblationConfig("No_WF",   "No waveform branch",  use_ddm=True,  use_wf=False, use_tab=True,  use_ampm=True),
    AblationConfig("No_Tab",  "No tabular branch",   use_ddm=True,  use_wf=True,  use_tab=False, use_ampm=True),
    AblationConfig("No_AMPM", "No AM/PM conditioning",use_ddm=True, use_wf=True,  use_tab=True,  use_ampm=False),
]

# ================================================================
# 1. LOAD DATA
# ================================================================
def load_npz_chunks(folder, key):
    paths = sorted(glob.glob(os.path.join(folder, "*.npz")))
    return np.concatenate([np.load(p)[key] for p in paths], axis=0)

df      = pd.read_csv(csv_path)
ddm_raw = load_npz_chunks(ddm_dir, "ddm").astype(np.float32)
wf_raw  = load_npz_chunks(wf_dir,  "waveforms").astype(np.float32)
print(f"CSV:{df.shape}  DDM:{ddm_raw.shape}  WF:{wf_raw.shape}")

# ================================================================
# 2. QC + ALIGN
# ================================================================
n       = min(len(df), len(ddm_raw), len(wf_raw))
df      = df.iloc[:n].reset_index(drop=True)
ddm_raw = ddm_raw[:n]
wf_raw  = wf_raw[:n]

df = df[df["ddma_3x3"]      > 0]
df = df[df["smap_dist_km"]  < 10]
df = df[df["soil_moisture"]  > 0.0]
df = df[df["soil_moisture"]  < 0.65]
df = df.reset_index(drop=True)
n  = len(df)
ddm_raw = ddm_raw[:n]
wf_raw  = wf_raw[:n]
print(f"After QC: {n} samples  "
      f"SM:[{df['soil_moisture'].min():.4f},{df['soil_moisture'].max():.4f}]")

H_, W_ = ddm_raw.shape[1], ddm_raw.shape[2]   # 17, 11

# ================================================================
# 3. FEATURE ENGINEERING
# ================================================================
day_str = df["day"].astype(str)
year    = day_str.str[:4].astype(int)
month   = day_str.str[4:6].astype(int)
dom     = day_str.str[6:8].astype(int)
doy     = pd.to_datetime(dict(year=year, month=month, day=dom)).dt.dayofyear.values
doy_sin = np.sin(2 * np.pi * doy / 365.0).astype(np.float32)
doy_cos = np.cos(2 * np.pi * doy / 365.0).astype(np.float32)

am_pm    = (df["smap_group"].str.upper() == "PM").astype(np.float32).values
log_peak = np.log1p(df["ddm_peak"].values).astype(np.float32)
log_3x3  = np.log1p(np.maximum(df["ddma_3x3"].values, 0)).astype(np.float32)
lat_r    = np.deg2rad(df["lat"].values)
lon_r    = np.deg2rad(df["lon"].values)

# Full 12-feature tabular vector (with seasonal encoding)
tab_full = np.stack([
    df["lat"].values, df["lon"].values,
    np.sin(lat_r), np.cos(lat_r),
    np.sin(lon_r), np.cos(lon_r),
    df["smap_dist_km"].values,
    log_peak, log_3x3,
    am_pm,
    doy_sin, doy_cos,
], axis=1).astype(np.float32)   # (N, 12)

def normalise_tab(tab, skip_cols):
    mu = tab.mean(0, keepdims=True)
    sd = tab.std(0,  keepdims=True)
    sd[sd < 1e-6] = 1.0
    t  = (tab - mu) / sd
    t[:, skip_cols] = tab[:, skip_cols]
    return np.clip(t, -5, 5)

# Columns that are already bounded/binary: cos/sin (3-8), am_pm (9), doy_sin/cos (10,11)
tab_norm = normalise_tab(tab_full, skip_cols=[3, 4, 5, 6, 9, 10, 11])
TAB_DIM  = tab_norm.shape[1]   # 12

# ================================================================
# 4. DDM PREPROCESSING  (3-channel: DDM + gradient + local variance)
# ================================================================
def per_sample_minmax(x, axis=None, eps=1e-6):
    xmin = x.min(axis=axis, keepdims=True)
    xmax = x.max(axis=axis, keepdims=True)
    return (x - xmin) / np.maximum(xmax - xmin, eps)

def ddm_gradient_map(ddm):
    gx = np.gradient(ddm, axis=2)
    gy = np.gradient(ddm, axis=1)
    return per_sample_minmax(
        np.sqrt(gx**2 + gy**2), axis=(1, 2)).astype(np.float32)

def ddm_localvar_map(ddm, w=3):
    x  = torch.from_numpy(ddm).unsqueeze(1)
    k  = torch.ones(1, 1, w, w) / (w * w)
    with torch.no_grad():
        mu  = F.conv2d(x, k, padding=w // 2)
        mu2 = F.conv2d(x ** 2, k, padding=w // 2)
    return per_sample_minmax(
        (mu2 - mu**2).clamp(0).squeeze(1).numpy(),
        axis=(1, 2)).astype(np.float32)

ddm_log  = np.log1p(np.maximum(ddm_raw, 0.0))
ddm_proc = per_sample_minmax(ddm_log, axis=(1, 2)).astype(np.float32)

print("Computing DDM gradient …")
ddm_grad  = ddm_gradient_map(ddm_proc)
print("Computing DDM local variance …")
ddm_lvar  = ddm_localvar_map(ddm_proc)
ddm_multi = np.stack([ddm_proc, ddm_grad, ddm_lvar], axis=1)  # (N,3,H,W)
assert not np.isnan(ddm_multi).any(), "NaN in DDM multi-channel array"

# ================================================================
# 5. WAVEFORM PREPROCESSING + PHYSICS FEATURES
# ================================================================
wf_log  = np.log1p(np.maximum(wf_raw, 0.0))
wf_proc = per_sample_minmax(wf_log, axis=(2,)).astype(np.float32)   # (N,3,17)

def wf_physics_features(wf):
    N, C, L = wf.shape
    out = np.zeros((N, C * 4), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            s  = wf[i, c]
            pk = int(np.argmax(s))
            pv = float(s[pk])
            b  = c * 4
            out[i, b]   = (pv - s[0]) / (pk + 1e-6)            if pk > 0   else 0.0
            out[i, b+1] = (float(s[-1]) - pv) / (L - pk + 1e-6) if pk < L-1 else 0.0
            ab = np.where(s >= pv / 2)[0]
            out[i, b+2] = float(ab[-1] - ab[0]) / L            if len(ab) > 1 else 0.0
            lm = float(s[:pk+1].mean()) if pk > 0   else 0.0
            tm = float(s[pk:].mean())   if pk < L-1 else 0.0
            out[i, b+3] = (lm - tm) / (lm + tm + 1e-6)
    return out

print("Computing waveform physics features …")
wf_phys      = wf_physics_features(wf_proc)
wp_mu        = wf_phys.mean(0, keepdims=True)
wp_sd        = wf_phys.std(0,  keepdims=True)
wp_sd[wp_sd < 1e-6] = 1.0
wf_phys_norm = np.clip((wf_phys - wp_mu) / wp_sd, -10, 10)
PHYS_DIM     = wf_phys_norm.shape[1]   # 12
print(f"WF physics features: {wf_phys_norm.shape}")

# ================================================================
# 6. TARGET + STRATIFIED SPLIT
# ================================================================
sm_target  = df["soil_moisture"].values.astype(np.float32)
sm_tercile = pd.qcut(sm_target, q=3, labels=False)
indices    = np.arange(n)
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED,
    shuffle=True, stratify=sm_tercile)
print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

# ================================================================
# 7. DATASET
# ================================================================
class CYGNSSDataset(Dataset):
    def __init__(self, idx):
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (
            torch.from_numpy(ddm_multi[j]).float(),
            torch.from_numpy(wf_proc[j]).float(),
            torch.from_numpy(tab_norm[j]).float(),
            torch.from_numpy(wf_phys_norm[j]).float(),
            torch.tensor(float(am_pm[j])).float(),
            torch.tensor(sm_target[j]).float(),
        )

train_ds = CYGNSSDataset(train_idx)
val_ds   = CYGNSSDataset(val_idx)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=N_WORK, pin_memory=USE_PIN)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=N_WORK, pin_memory=USE_PIN)

# ================================================================
# 8. MODEL BUILDING BLOCKS
# ================================================================

# ── CBAM ─────────────────────────────────────────────────────────
class ChannelAttn(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        mid = max(c // r, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(nn.Linear(c, mid), nn.ReLU(), nn.Linear(mid, c))

    def forward(self, x):
        b, c = x.shape[:2]
        w = torch.sigmoid(
            self.fc(self.avg(x).view(b, c)) +
            self.fc(self.mx(x).view(b, c)))
        return x * w.view(b, c, 1, 1)

class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        a    = x.mean(1, keepdim=True)
        m, _ = x.max(1,  keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([a, m], 1)))

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttn(c)
        self.sa = SpatialAttn()

    def forward(self, x):
        return self.sa(self.ca(x))

# ── Soft Peak Localizer ───────────────────────────────────────────
class SoftPeakLocalizer(nn.Module):
    def __init__(self, H, W, out_dim=32):
        super().__init__()
        self.proj = nn.Linear(5, out_dim)
        rg, cg = torch.meshgrid(
            torch.arange(H).float() / H,
            torch.arange(W).float() / W,
            indexing='ij')
        self.register_buffer('rf', rg.reshape(-1))
        self.register_buffer('cf', cg.reshape(-1))
        self.temp = 1.0

    def forward(self, x):
        B    = x.shape[0]
        flat = x[:, 0].reshape(B, -1)
        a    = F.softmax(flat * self.temp, dim=-1)
        pv   = (a * flat).sum(-1, keepdim=True)
        pr   = (a * self.rf).sum(-1, keepdim=True)
        pc   = (a * self.cf).sum(-1, keepdim=True)
        rv   = (a * (self.rf - pr) ** 2).sum(-1, keepdim=True)
        cv   = (a * (self.cf - pc) ** 2).sum(-1, keepdim=True)
        return self.proj(torch.cat([pv, pr, pc, rv, cv], -1))

# ── DDM Branch ───────────────────────────────────────────────────
class DDMBranch(nn.Module):
    def __init__(self, out_dim=128, H=17, W=11):
        super().__init__()

        def blk(ci, co, k, p, d=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, padding=p, dilation=d),
                nn.BatchNorm2d(co), nn.GELU(), CBAM(co))

        self.fine   = blk(3, 32, 3, 1)
        self.med    = blk(3, 32, 3, 2, d=2)
        self.coarse = blk(3, 32, 5, 2)
        self.merge  = nn.Sequential(
            nn.Conv2d(96, 64, 1), nn.BatchNorm2d(64), nn.GELU(), CBAM(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.GELU(), CBAM(128),
            nn.AdaptiveAvgPool2d((2, 2)))
        self.pk   = SoftPeakLocalizer(H, W, 32)
        self.proj = nn.Linear(128 * 4 + 32, out_dim)

    def forward(self, x):
        s = torch.cat([self.fine(x), self.med(x), self.coarse(x)], 1)
        h = self.merge(s).flatten(1)
        return self.proj(torch.cat([h, self.pk(x)], 1))

# ── AM/PM FiLM ───────────────────────────────────────────────────
class AMPMFilm(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.emb   = nn.Embedding(2, 16)
        self.gamma = nn.Linear(16, feat_dim)
        self.beta  = nn.Linear(16, feat_dim)

    def forward(self, h, ampm):
        e = self.emb(ampm.long())
        return h * torch.sigmoid(self.gamma(e)) + self.beta(e)

# ── Waveform Branch ───────────────────────────────────────────────
class WFBranch(nn.Module):
    def __init__(self, phys_dim=12, out_dim=128):
        super().__init__()

        def c1(ci, co, k, p):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, padding=p),
                nn.BatchNorm1d(co), nn.GELU())

        self.fine   = c1(3, 32, 3, 1)
        self.med    = c1(3, 32, 5, 2)
        self.coarse = c1(3, 32, 9, 4)
        self.merge  = nn.Sequential(c1(96, 64, 3, 1), nn.MaxPool1d(2))
        self.gru    = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.film   = AMPMFilm(128)
        self.phys   = nn.Sequential(
            nn.Linear(phys_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.proj   = nn.Sequential(
            nn.Linear(128 + 64, out_dim), nn.GELU())

    def forward(self, x, xp, ampm, use_ampm=True):
        h      = torch.cat([self.fine(x), self.med(x), self.coarse(x)], 1)
        h      = self.merge(h)
        go, _  = self.gru(h.transpose(1, 2))
        h      = go[:, -1, :]
        if use_ampm:
            h  = self.film(h, ampm)
        return self.proj(torch.cat([h, self.phys(xp)], 1))

# ── Tabular Branch ────────────────────────────────────────────────
class TabBranch(nn.Module):
    def __init__(self, in_dim=12, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 128),    nn.GELU(),
            nn.Linear(128, out_dim))

    def forward(self, x):
        return self.net(x)

# ── Cross-Modal Attention ─────────────────────────────────────────
class CMA(nn.Module):
    def __init__(self, d=128, h=4):
        super().__init__()
        self.h  = h
        self.dh = d // h
        self.sc = self.dh ** -0.5
        self.q  = nn.Linear(d, d, bias=False)
        self.k  = nn.Linear(d, d, bias=False)
        self.v  = nn.Linear(d, d, bias=False)
        self.o  = nn.Linear(d, d)
        self.n  = nn.LayerNorm(d)

    def forward(self, q, c):
        B, D = q.shape
        Q = self.q(q).view(B, self.h, self.dh)
        K = self.k(c).view(B, self.h, self.dh)
        V = self.v(c).view(B, self.h, self.dh)
        a = F.softmax((Q * K * self.sc).sum(-1), dim=-1)
        return self.n(q + self.o((a.unsqueeze(-1) * V).reshape(B, D)))

# ── Gated MoE Fusion ─────────────────────────────────────────────
class MoEFusion(nn.Module):
    def __init__(self, d=128, E=4):
        super().__init__()
        # Cross-modal attention pairs
        self.d2w = CMA(d); self.w2d = CMA(d)
        self.d2t = CMA(d); self.t2d = CMA(d)
        self.w2t = CMA(d); self.t2w = CMA(d)
        # Gated MoE
        self.gate = nn.Sequential(
            nn.Linear(d * 3, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, E), nn.Softmax(dim=-1))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d*3, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128)),
            nn.Sequential(nn.Linear(d*3, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 128)),
            nn.Sequential(nn.Linear(d*3, 128), nn.GELU(), nn.Linear(128, 128)),
            nn.Sequential(nn.Linear(d*3, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 128)),
        ])
        self.res  = nn.Linear(d * 3, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, f1, f2, f3):
        # Cross-modal attention
        f1 = self.d2w(f1, f2); f1 = self.d2t(f1, f3)
        f2 = self.w2d(f2, f1); f2 = self.w2t(f2, f3)
        f3 = self.t2d(f3, f1); f3 = self.t2w(f3, f2)
        z    = torch.cat([f1, f2, f3], 1)
        w    = self.gate(z)
        outs = torch.stack([e(z) for e in self.experts], 1)
        return self.norm((w.unsqueeze(-1) * outs).sum(1) + self.res(z))

# ── Evidential Regression Head ────────────────────────────────────
class EvidentialHead(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(128, 64),     nn.GELU(), nn.Dropout(0.15))
        self.gamma = nn.Linear(64, 1)
        self.lognu = nn.Linear(64, 1)
        self.logal = nn.Linear(64, 1)
        self.logbe = nn.Linear(64, 1)

    def forward(self, x):
        h   = self.shared(x)
        gam = torch.sigmoid(self.gamma(h)).squeeze(1) * 0.65
        nu  = F.softplus(self.lognu(h)).squeeze(1) + 1e-4
        al  = F.softplus(self.logal(h)).squeeze(1) + 1.0 + 1e-4
        be  = F.softplus(self.logbe(h)).squeeze(1) + 1e-4
        return gam, nu, al, be

# ================================================================
# 9. ABLATION MODEL
#    Branches are zeroed out (not removed) to keep architecture
#    comparable — each missing branch outputs a zero embedding.
# ================================================================
class AblationModel(nn.Module):
    def __init__(self, cfg: AblationConfig, H=17, W=11):
        super().__init__()
        self.cfg  = cfg
        self.ddm  = DDMBranch(128, H, W)
        self.wf   = WFBranch(PHYS_DIM, 128)
        self.tab  = TabBranch(TAB_DIM, 128)
        self.fuse = MoEFusion(128, 4)
        self.head = EvidentialHead(128)

    def forward(self, xd, xw, xt, xp, ampm):
        B = xd.shape[0]

        f1 = self.ddm(xd)  if self.cfg.use_ddm else torch.zeros(B, 128, device=xd.device)
        f2 = self.wf(xw, xp, ampm, use_ampm=self.cfg.use_ampm) \
             if self.cfg.use_wf  else torch.zeros(B, 128, device=xd.device)
        f3 = self.tab(xt) if self.cfg.use_tab else torch.zeros(B, 128, device=xd.device)

        return self.head(self.fuse(f1, f2, f3))

# ================================================================
# 10. LOSSES
# ================================================================
def nig_nll_loss(y, gam, nu, al, be):
    tbl = 2 * be * (1 + nu)
    nll = (0.5 * torch.log(np.pi / nu)
           - al * torch.log(tbl)
           + (al + 0.5) * torch.log(nu * (y - gam) ** 2 + tbl)
           + torch.lgamma(al) - torch.lgamma(al + 0.5))
    return nll.mean()

def nig_reg_loss(y, gam, nu, al, be):
    return (torch.abs(y - gam) * (2 * nu + al)).mean()

def huber_loss(pred, target, delta=0.05):
    return F.huber_loss(pred, target, delta=delta)

def kge_loss(pred, target):
    r = torch.corrcoef(torch.stack([pred, target]))[0, 1].clamp(-1, 1)
    b = pred.mean() / (target.mean() + 1e-8)
    g = (pred.std() / (pred.mean() + 1e-8)) / \
        (target.std() / (target.mean() + 1e-8) + 1e-8)
    return 1 - (1 - torch.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2))

def get_loss_weights(epoch):
    if   epoch < 20: return dict(huber=1.0, kge=0.0, nig=0.0, nig_reg=0.0)
    elif epoch < 50: return dict(huber=1.0, kge=0.3, nig=0.0, nig_reg=0.0)
    else:            return dict(huber=0.8, kge=0.2, nig=0.1, nig_reg=0.05)

# ================================================================
# 11. METRICS
# ================================================================
def compute_metrics(obs, pred):
    obs  = np.array(obs)
    pred = np.array(pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    r2   = r2_score(obs, pred)
    nse  = 1 - np.sum((obs - pred) ** 2) / \
               (np.sum((obs - obs.mean()) ** 2) + 1e-8)
    r    = np.corrcoef(obs, pred)[0, 1]
    b    = pred.mean() / (obs.mean() + 1e-8)
    g    = (pred.std() / (pred.mean() + 1e-8)) / \
           (obs.std()  / (obs.mean()  + 1e-8) + 1e-8)
    kge  = 1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)
    return dict(RMSE=rmse, R2=r2, NSE=nse, KGE=kge)

# ================================================================
# 12. SINGLE RUN
# ================================================================
def run_ablation(cfg: AblationConfig):
    torch.manual_seed(SEED)
    model = AblationModel(cfg, H_, W_).to(device)
    nparams = sum(p.numel() for p in model.parameters())

    # Two parameter groups: backbone vs head/projection
    backbone_params = []
    head_params     = []
    for name, p in model.named_parameters():
        if any(k in name for k in ["head", "proj", "fuse.res", "fuse.norm"]):
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 5e-4},
        {"params": head_params,     "lr": 1.5e-3},
    ], weight_decay=5e-4)

    WARMUP = 10
    def lr_lam(ep):
        if ep < WARMUP:
            return (ep + 1) / WARMUP
        return 0.5 * (1 + np.cos(np.pi * (ep - WARMUP) / (EPOCHS - WARMUP)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lam)

    best_r2  = -999.0
    best_met = {}
    t0       = time.time()

    for epoch in range(EPOCHS):
        W = get_loss_weights(epoch)

        # Temperature schedule for soft peak localizer
        if hasattr(model.ddm, 'pk') and cfg.use_ddm:
            model.ddm.pk.temp = min(1.0 + (epoch / 80.0) * 19.0, 20.0)

        # ── Train ──
        model.train()
        tot_loss = 0.0
        n_batches = 0
        for xd, xw, xt, xp, ampm, yb in train_loader:
            xd   = xd.to(device);   xw   = xw.to(device)
            xt   = xt.to(device);   xp   = xp.to(device)
            ampm = ampm.to(device); yb   = yb.to(device)

            gam, nu, al, be = model(xd, xw, xt, xp, ampm)
            if torch.isnan(gam).any():
                continue

            loss = W["huber"] * huber_loss(gam, yb)
            if W["kge"]     > 0: loss += W["kge"]     * kge_loss(gam, yb)
            if W["nig"]     > 0: loss += W["nig"]     * nig_nll_loss(yb, gam, nu, al, be)
            if W["nig_reg"] > 0: loss += W["nig_reg"] * nig_reg_loss(yb, gam, nu, al, be)
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss  += loss.item()
            n_batches += 1

        scheduler.step()

        # ── Validate ──
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xd, xw, xt, xp, ampm, yb in val_loader:
                xd   = xd.to(device);   xw   = xw.to(device)
                xt   = xt.to(device);   xp   = xp.to(device)
                ampm = ampm.to(device)
                gam, _, _, _ = model(xd, xw, xt, xp, ampm)
                gam = torch.nan_to_num(gam, nan=0.25)
                preds.extend(gam.cpu().numpy())
                trues.extend(yb.numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        if np.isnan(preds).any():
            continue

        m = compute_metrics(trues, preds)
        if m["R2"] > best_r2:
            best_r2  = m["R2"]
            best_met = m
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"{cfg.name}_best.pt"))

        print(f"  [{cfg.name}] Ep{epoch+1:3d}/{EPOCHS} "
              f"Loss:{tot_loss/max(n_batches,1):.5f}  "
              f"RMSE:{m['RMSE']:.5f}  R²:{m['R2']:.5f}  "
              f"NSE:{m['NSE']:.5f}  KGE:{m['KGE']:.5f}  "
              f"Best R²:{best_r2:.5f}")

    elapsed = time.time() - t0
    return {**best_met, "Params": nparams, "Time_s": elapsed}

# ================================================================
# 13. RUN ALL 5 VARIANTS
# ================================================================
all_results = {}

print("\n" + "=" * 70)
print("  CYGNSS-SM-Net  ABLATION STUDY  (Table 6 variants)")
print("=" * 70)

for cfg in ABLATIONS:
    print(f"\n{'─'*70}")
    print(f"  Running: {cfg.name}  —  {cfg.label}")
    print(f"{'─'*70}")
    result = run_ablation(cfg)
    all_results[cfg.name] = {"Label": cfg.label, **result}
    print(f"  ✓ Done  |  Best R²={result['R2']:.5f}  "
          f"RMSE={result['RMSE']:.5f}  Time={result['Time_s']:.0f}s")

# ================================================================
# 14. RESULTS TABLE  (matches Table 6 in paper)
# ================================================================
print("\n\n" + "=" * 80)
print("  ABLATION RESULTS  (Table 6)  —  best validation checkpoint per variant")
print("=" * 80)

rows = []
for name, res in all_results.items():
    rows.append({
        "Variant":  res["Label"],
        "RMSE":     round(res["RMSE"], 5),
        "R2":       round(res["R2"],   5),
        "NSE":      round(res["NSE"],  5),
        "KGE":      round(res["KGE"],  5),
        "Params":   f'{res["Params"]:,}',
        "Time_s":   f'{res["Time_s"]:.0f}s',
    })

result_df = pd.DataFrame(rows)
result_df.to_csv(os.path.join(save_dir, "ablation_results_table6.csv"), index=False)

header = (f"{'Variant':<28} {'RMSE':>8} {'R²':>8} "
          f"{'NSE':>8} {'KGE':>8}  {'Params':>12}  {'Time':>7}")
print(header)
print("─" * len(header))

best_r2_global = max(r["R2"] for r in all_results.values())
for r in rows:
    marker = "  ◄ BEST" if float(r["R2"]) == round(best_r2_global, 5) else ""
    print(f"{r['Variant']:<28} {r['RMSE']:>8} {r['R2']:>8} "
          f"{r['NSE']:>8} {r['KGE']:>8}  "
          f"{r['Params']:>12}  {r['Time_s']:>7}{marker}")

print("─" * len(header))
print(f"\nResults saved to: {save_dir}/ablation_results_table6.csv")
print("Model checkpoints saved to:", save_dir)
