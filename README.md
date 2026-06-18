# CYGNSS-SM-Net

**Physics-informed Multimodal Deep Learning for SMAP-referenced Soil Moisture Estimation from CYGNSS Observations**

> Aryan Singh · Hari Om · Sunil Jha  
> Department of Computer Science and Engineering, IIT (ISM) Dhanbad, India  
> *Computers & Geosciences* (2025)

---

## Overview

CYGNSS-SM-Net is a physics-informed multimodal deep learning framework that jointly processes raw Delay-Doppler Maps (DDMs), waveform sequences, and physics-informed tabular features from CYGNSS GNSS-R observations to estimate SMAP-equivalent soil moisture with calibrated per-sample uncertainty.

Most existing CYGNSS soil moisture methods reduce DDMs to scalar reflectivity indices before learning, discarding the rich spatial and temporal structure of the raw signal. CYGNSS-SM-Net directly ingests all three input modalities through dedicated encoding branches, fuses them via cross-modal attention and gated mixture-of-experts, and produces both a physics-bounded soil moisture estimate and a calibrated epistemic uncertainty — without Monte Carlo sampling.

### Key Results (Computers & Geosciences, 2025)

| Region | RMSE (m³/m³) | R² | NSE | KGE |
|---|---|---|---|---|
| Godavari | 0.01991 | 0.9586 | 0.9586 | 0.9741 |
| Narmada–Tapti | 0.01898 | 0.9624 | 0.9624 | 0.9802 |

Validated against SMAP Level-3 enhanced passive product (9 km) over 2021–2024. Results are consistent across years, seasons, AM/PM overpasses, and spatial subsets.

---

## Architecture

```
DDM (3×17×11)  ──► DDMBranch    ──► 128-dim embedding ──►┐
WF  (3×17)     ──► WFBranch     ──► 128-dim embedding ──►├──► MoEFusion ──► EvidentialHead
Tab (12,)      ──► TabBranch    ──► 128-dim embedding ──►┘
                                                          ▼
                                             SM estimate + uncertainty
```

**DDMBranch** — Multi-scale dilated CNN (fine/medium/coarse receptive fields) with CBAM channel+spatial attention and a differentiable soft peak localizer with curriculum temperature annealing.

**WFBranch** — Multi-scale 1D CNN (k=3,5,9) + Bidirectional GRU + AM/PM FiLM conditioning + explicit waveform physics features (leading-edge slope, trailing-edge slope, half-power width, asymmetry).

**TabBranch** — 3-layer MLP encoding 12 physics-informed features: spatial coordinates, circular geographic encodings, SMAP collocation distance, log DDM power terms, AM/PM flag, and seasonal sin/cos encodings.

**MoEFusion** — Six bidirectional cross-modal attention pairs (DDM↔WF, DDM↔Tab, WF↔Tab) followed by gated mixture-of-experts fusion (4 experts, learned soft router) with residual skip and LayerNorm.

**EvidentialHead** — Normal-Inverse-Gamma regression head producing physics-bounded SM estimates (sigmoid constrained to [0, 0.65] cm³/cm³) and calibrated epistemic uncertainty U = β / (ν(α−1)) without Monte Carlo sampling.

---

## Repository Structure

```
CYGNSS-SM-Net/
├── src/
│   ├── model.py          # Full architecture (DDMBranch, WFBranch, TabBranch, MoEFusion, EvidentialHead)
│   ├── losses.py         # Loss functions (Huber, KGE, NIG-NLL) + evaluation metrics (RMSE, R², NSE, KGE)
│   └── data_utils.py     # Data loading, QC, DDM/waveform preprocessing, dataset class
├── scripts/
│   ├── train.py          # Single-region training script
│   ├── cross_region.py   # Cross-region generalisation evaluation
│   └── ablation_study.py # Table 6 ablation variants (Full / No DDM / No WF / No Tab / No AM-PM)
├── configs/
│   └── paths.py          # Data paths configuration
├── results/              # Saved model checkpoints and CSV outputs (created at runtime)
├── requirements.txt
└── LICENSE               # MIT
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/CYGNSS-SM-Net.git
cd CYGNSS-SM-Net
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, NumPy, Pandas, scikit-learn.

---

## Data

This work uses the following publicly available datasets:

| Dataset | Source | Access |
|---|---|---|
| CYGNSS Level 1 DDM (v3.2) | NASA PO.DAAC | [doi:10.5067/CYGNS-L1X32](https://doi.org/10.5067/CYGNS-L1X32) |
| SMAP Level 3 Enhanced Passive (9 km) | NASA NSIDC | [SMAP L3 Product](https://nsidc.org/data/spl3smp_e) |

### Expected Input Format

Each region requires three files/directories:

```
samples_<region>_QC_stream.csv      # Metadata CSV with columns:
                                    #   lat, lon, day (YYYYMMDD), smap_group (AM/PM),
                                    #   ddm_peak, ddma_3x3, smap_dist_km, soil_moisture

chunks_ddm/                         # Directory of .npz files
    chunk_0000.npz                  #   key: 'ddm'       shape: (N_chunk, 17, 11)
    chunk_0001.npz
    ...

chunks_wf/                          # Directory of .npz files
    chunk_0000.npz                  #   key: 'waveforms' shape: (N_chunk, 3, 17)
    chunk_0001.npz
    ...
```

---

## Usage

### 1. Configure Paths

Edit `configs/paths.py` to point to your data directories:

```python
REGION_PATHS = {
    "narmada_tapti": {
        "csv_path": "/path/to/samples_NarmadaTapti_QC_stream.csv",
        "ddm_dir":  "/path/to/chunks_ddm/",
        "wf_dir":   "/path/to/chunks_wf/",
    },
    "godavari": {
        "csv_path": "/path/to/samples_Godavari_QC_stream.csv",
        "ddm_dir":  "/path/to/chunks_ddm/",
        "wf_dir":   "/path/to/chunks_wf/",
    },
}
```

### 2. Train

```bash
# Train on Narmada-Tapti basin
python scripts/train.py --region narmada_tapti --epochs 100

# Train on Godavari basin
python scripts/train.py --region godavari --epochs 100
```

Outputs saved to `results/<region>/`:
- `best_model.pt` — best checkpoint by validation R²
- `training_history.csv` — epoch-wise metrics
- `tab_mean.npy`, `tab_std.npy` — normalisation statistics
- `wf_phys_mean.npy`, `wf_phys_std.npy`

### 3. Cross-Region Evaluation

```bash
# Train on Godavari, test on Narmada-Tapti
python scripts/cross_region.py --source godavari --target narmada_tapti

# Train on Narmada-Tapti, test on Godavari
python scripts/cross_region.py --source narmada_tapti --target godavari
```

### 4. Ablation Study (Table 6)

```bash
python scripts/ablation_study.py
```

Runs 5 variants: Full model, No DDM branch, No waveform branch, No tabular branch, No AM/PM conditioning. Results saved to `results/ablation/ablation_results_table6.csv`.

### 5. Use Model Directly

```python
import torch
from src.model import CYGNSSSMNet

model = CYGNSSSMNet(tab_dim=12, phys_dim=12, H=17, W=11)
model.load_state_dict(torch.load("results/narmada_tapti/best_model.pt"))
model.eval()

# Inputs: batch of 8 samples
xd   = torch.randn(8, 3, 17, 11)   # DDM (3-channel)
xw   = torch.randn(8, 3, 17)       # Waveforms
xt   = torch.randn(8, 12)          # Tabular features
xp   = torch.randn(8, 12)          # Waveform physics features
ampm = torch.zeros(8)              # 0=AM, 1=PM

with torch.no_grad():
    gam, nu, al, be = model(xd, xw, xt, xp, ampm)
    sm_estimate  = gam                         # (B,) soil moisture in cm³/cm³
    uncertainty  = be / (nu * (al - 1))        # (B,) epistemic uncertainty
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Backbone LR | 5×10⁻⁴ |
| Head/Projection LR | 1.5×10⁻³ |
| Weight decay | 5×10⁻⁴ |
| Batch size | 64 |
| Max epochs | 100 |
| LR schedule | 10-epoch warmup + cosine decay |
| Gradient clipping | max norm 1.0 |
| Early stopping patience | 25 epochs |
| Train/Val split | 80:20 stratified by SM tercile |

**Three-phase curriculum:**

| Phase | Epochs | Loss |
|---|---|---|
| 1 | 1–20 | Huber (δ=0.05) |
| 2 | 21–50 | Huber + 0.3×KGE |
| 3 | 51–100 | 0.8×Huber + 0.2×KGE + 0.1×NIG-NLL + 0.05×NIG-Reg |

---

## Citation

If you use CYGNSS-SM-Net in your research, please cite:

```bibtex
@article{singh2025cygnss,
  title   = {Soil Moisture Retrieval from CYGNSS Observations Using CYGNSS-SM-Net:
             A Physics-informed Multimodal Deep Learning Framework},
  author  = {Singh, Aryan and Om, Hari and Jha, Sunil},
  journal = {Computers \& Geosciences},
  year    = {2025},
  publisher = {Elsevier}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
