# ================================================================
# CYGNSS-SM-Net  –  Data Loading and Preprocessing Utilities
# ================================================================

import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ================================================================
# Data Loading
# ================================================================

def load_npz_chunks(folder, key):
    """
    Load and concatenate all .npz chunk files from a directory.

    Args:
        folder (str): Directory containing .npz files
        key    (str): Key to extract from each .npz file (e.g. 'ddm', 'waveforms')

    Returns:
        numpy array concatenated along axis 0
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No .npz files found in: {folder}")
    return np.concatenate([np.load(p)[key] for p in paths], axis=0)


# ================================================================
# Quality Control
# ================================================================

def apply_qc(df, ddm_raw, wf_raw):
    """
    Apply standard CYGNSS quality control filters.

    Filters:
        - ddma_3x3 > 0          : positive coherent neighbourhood amplitude
        - smap_dist_km < 10     : SMAP collocation distance < 10 km
        - 0 < soil_moisture < 0.65 : physical SM bounds (removes sentinels)

    Returns:
        df, ddm_raw, wf_raw after filtering
    """
    n       = min(len(df), len(ddm_raw), len(wf_raw))
    df      = df.iloc[:n].reset_index(drop=True)
    ddm_raw = ddm_raw[:n]
    wf_raw  = wf_raw[:n]

    df = df[df["ddma_3x3"]     > 0]
    df = df[df["smap_dist_km"] < 10]
    df = df[df["soil_moisture"] > 0.0]
    df = df[df["soil_moisture"] < 0.65]
    df = df.reset_index(drop=True)

    n       = len(df)
    ddm_raw = ddm_raw[:n]
    wf_raw  = wf_raw[:n]

    return df, ddm_raw, wf_raw


# ================================================================
# DDM Preprocessing
# ================================================================

def per_sample_minmax(x, axis=None, eps=1e-6):
    """Per-sample min-max normalisation along specified axes."""
    xmin = x.min(axis=axis, keepdims=True)
    xmax = x.max(axis=axis, keepdims=True)
    return (x - xmin) / np.maximum(xmax - xmin, eps)


def ddm_gradient_map(ddm):
    """
    Compute gradient magnitude map of DDM.
    Captures peak sharpness and spatial transition information.
    """
    gx = np.gradient(ddm, axis=2)
    gy = np.gradient(ddm, axis=1)
    return per_sample_minmax(
        np.sqrt(gx ** 2 + gy ** 2), axis=(1, 2)).astype(np.float32)


def ddm_localvar_map(ddm, w=3):
    """
    Compute local variance map of DDM using a sliding window.
    Captures neighbourhood scattering texture.
    """
    x = torch.from_numpy(ddm).unsqueeze(1)
    k = torch.ones(1, 1, w, w) / (w * w)
    with torch.no_grad():
        mu  = F.conv2d(x, k, padding=w // 2)
        mu2 = F.conv2d(x ** 2, k, padding=w // 2)
    return per_sample_minmax(
        (mu2 - mu ** 2).clamp(0).squeeze(1).numpy(),
        axis=(1, 2)).astype(np.float32)


def preprocess_ddm(ddm_raw):
    """
    Full DDM preprocessing pipeline:
        1. Log compression: log(1 + max(D, 0))
        2. Per-sample min-max normalisation
        3. Gradient magnitude channel
        4. Local variance channel
        5. Stack to 3-channel tensor (N, 3, H, W)

    Returns:
        ddm_multi (N, 3, H, W) float32 array
    """
    ddm_log  = np.log1p(np.maximum(ddm_raw, 0.0))
    ddm_proc = per_sample_minmax(ddm_log, axis=(1, 2)).astype(np.float32)
    ddm_grad = ddm_gradient_map(ddm_proc)
    ddm_lvar = ddm_localvar_map(ddm_proc)
    ddm_multi = np.stack([ddm_proc, ddm_grad, ddm_lvar], axis=1)
    assert not np.isnan(ddm_multi).any(), "NaN detected in DDM multi-channel array"
    return ddm_multi


# ================================================================
# Waveform Preprocessing
# ================================================================

def wf_physics_features(wf):
    """
    Extract physics-motivated waveform descriptors.

    For each of 3 waveform channels, computes:
        - Leading-edge slope     (power rise rate before peak)
        - Trailing-edge slope    (power decay rate after peak)
        - Half-power width       (normalised width at 50% of peak)
        - Asymmetry index        (leading vs trailing mean power ratio)

    Args:
        wf: (N, 3, L) normalised waveform array

    Returns:
        (N, 12) float32 physics feature array
    """
    N, C, L = wf.shape
    out = np.zeros((N, C * 4), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            s  = wf[i, c]
            pk = int(np.argmax(s))
            pv = float(s[pk])
            b  = c * 4
            out[i, b]   = (pv - s[0]) / (pk + 1e-6)              if pk > 0   else 0.0
            out[i, b+1] = (float(s[-1]) - pv) / (L - pk + 1e-6)  if pk < L-1 else 0.0
            ab = np.where(s >= pv / 2)[0]
            out[i, b+2] = float(ab[-1] - ab[0]) / L              if len(ab) > 1 else 0.0
            lm = float(s[:pk+1].mean()) if pk > 0   else 0.0
            tm = float(s[pk:].mean())   if pk < L-1 else 0.0
            out[i, b+3] = (lm - tm) / (lm + tm + 1e-6)
    return out


def preprocess_waveforms(wf_raw):
    """
    Full waveform preprocessing pipeline:
        1. Log compression: log(1 + max(W, 0))
        2. Per-sample min-max normalisation along waveform axis
        3. Extract and z-score normalise physics features

    Returns:
        wf_proc     : (N, 3, 17) normalised waveform tensor
        wf_phys_norm: (N, 12)    normalised physics features
        stats       : dict with wp_mu, wp_sd for later use
    """
    wf_log  = np.log1p(np.maximum(wf_raw, 0.0))
    wf_proc = per_sample_minmax(wf_log, axis=(2,)).astype(np.float32)

    wf_phys = wf_physics_features(wf_proc)
    wp_mu   = wf_phys.mean(0, keepdims=True)
    wp_sd   = wf_phys.std(0,  keepdims=True)
    wp_sd[wp_sd < 1e-6] = 1.0
    wf_phys_norm = np.clip((wf_phys - wp_mu) / wp_sd, -10, 10).astype(np.float32)

    return wf_proc, wf_phys_norm, {"wp_mu": wp_mu, "wp_sd": wp_sd}


# ================================================================
# Tabular Feature Engineering
# ================================================================

def build_tabular_features(df):
    """
    Build 12-feature physics-informed tabular vector.

    Features:
        0-1 : latitude, longitude
        2-5 : sin/cos of lat, sin/cos of lon  (circular encoding)
        6   : smap_dist_km                    (collocation quality)
        7   : log(1 + ddm_peak)               (DDM peak power)
        8   : log(1 + ddma_3x3)               (3x3 neighbourhood power)
        9   : AM/PM flag (0=AM, 1=PM)
        10  : sin(DOY * 2pi/365)              (seasonal encoding)
        11  : cos(DOY * 2pi/365)

    Returns:
        tab_raw : (N, 12) float32 array (unnormalised)
        am_pm   : (N,)    float32 AM/PM flag
    """
    day_str = df["day"].astype(str).str.zfill(8)
    year    = day_str.str[:4].astype(int)
    month   = day_str.str[4:6].astype(int)
    dom     = day_str.str[6:8].astype(int)
    doy     = pd.to_datetime(dict(year=year, month=month, day=dom)).dt.dayofyear.values
    doy_sin = np.sin(2 * np.pi * doy / 365.0).astype(np.float32)
    doy_cos = np.cos(2 * np.pi * doy / 365.0).astype(np.float32)

    am_pm    = (df["smap_group"].astype(str).str.upper() == "PM").astype(np.float32).values
    log_peak = np.log1p(df["ddm_peak"].values).astype(np.float32)
    log_3x3  = np.log1p(np.maximum(df["ddma_3x3"].values, 0)).astype(np.float32)
    lat_r    = np.deg2rad(df["lat"].values)
    lon_r    = np.deg2rad(df["lon"].values)

    tab_raw = np.stack([
        df["lat"].values, df["lon"].values,
        np.sin(lat_r), np.cos(lat_r),
        np.sin(lon_r), np.cos(lon_r),
        df["smap_dist_km"].values,
        log_peak, log_3x3,
        am_pm, doy_sin, doy_cos,
    ], axis=1).astype(np.float32)

    return tab_raw, am_pm


def normalise_tabular(tab_raw, mu=None, sd=None):
    """
    Z-score normalise tabular features.
    Columns that are already bounded (sin/cos, AM/PM flag) are kept as-is.

    Args:
        tab_raw : (N, 12) raw tabular array
        mu      : (1, 12) pre-computed mean (if None, computed from tab_raw)
        sd      : (1, 12) pre-computed std  (if None, computed from tab_raw)

    Returns:
        tab_norm : (N, 12) normalised array clipped to [-5, 5]
        mu       : (1, 12) mean used
        sd       : (1, 12) std used
    """
    # Columns to skip normalisation: sin/cos lat/lon (2-5), AM/PM (9), doy sin/cos (10,11)
    skip = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)

    if mu is None:
        mu = tab_raw.mean(0, keepdims=True).astype(np.float32)
    if sd is None:
        sd = tab_raw.std(0,  keepdims=True).astype(np.float32)
        sd[sd < 1e-6] = 1.0

    tab_norm = (tab_raw - mu) / sd
    tab_norm[:, skip] = tab_raw[:, skip]
    return np.clip(tab_norm, -5, 5).astype(np.float32), mu, sd


# ================================================================
# PyTorch Dataset
# ================================================================

class CYGNSSDataset(Dataset):
    """
    PyTorch Dataset for CYGNSS soil moisture retrieval.

    Args:
        idx        : array of sample indices to include
        ddm_multi  : (N, 3, H, W) preprocessed DDM array
        wf_proc    : (N, 3, 17)   preprocessed waveform array
        tab_norm   : (N, 12)      normalised tabular features
        wf_phys    : (N, 12)      normalised waveform physics features
        am_pm      : (N,)         AM/PM flag
        sm_target  : (N,)         SMAP soil moisture targets (cm3/cm3)
    """
    def __init__(self, idx, ddm_multi, wf_proc, tab_norm,
                 wf_phys, am_pm, sm_target):
        self.idx       = np.array(idx)
        self.ddm       = ddm_multi
        self.wf        = wf_proc
        self.tab       = tab_norm
        self.wf_phys   = wf_phys
        self.am_pm     = am_pm
        self.sm_target = sm_target

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (
            torch.from_numpy(self.ddm[j]).float(),
            torch.from_numpy(self.wf[j]).float(),
            torch.from_numpy(self.tab[j]).float(),
            torch.from_numpy(self.wf_phys[j]).float(),
            torch.tensor(float(self.am_pm[j])).float(),
            torch.tensor(float(self.sm_target[j])).float(),
        )
