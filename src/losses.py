# ================================================================
# CYGNSS-SM-Net  –  Loss Functions and Evaluation Metrics
# ================================================================
# Three-phase training curriculum:
#   Phase 1 (ep  1-20): Huber only          -> stable regression
#   Phase 2 (ep 21-50): Huber + KGE         -> hydrological consistency
#   Phase 3 (ep 51+):   Huber + KGE + NIG   -> calibrated uncertainty
# ================================================================

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score


# ================================================================
# Loss Functions
# ================================================================

def huber_loss(pred, target, delta=0.05):
    """
    Huber loss with delta=0.05 (robust to outliers in SM range).
    Used in all three training phases.
    """
    return F.huber_loss(pred, target, delta=delta)


def kge_loss(pred, target):
    """
    Differentiable Kling-Gupta Efficiency loss.
    KGE = 1 - sqrt((r-1)^2 + (beta-1)^2 + (gamma-1)^2)
    We minimise 1-KGE to improve hydrological distributional agreement.
    """
    r = torch.corrcoef(torch.stack([pred, target]))[0, 1]
    r = torch.nan_to_num(r, nan=0.0).clamp(-1, 1)
    b = pred.mean() / (target.mean() + 1e-8)
    g = (pred.std() / (pred.mean() + 1e-8)) / \
        (target.std() / (target.mean() + 1e-8) + 1e-8)
    kge = 1 - torch.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)
    return 1 - kge


def nig_nll_loss(y, gam, nu, al, be):
    """
    Normal-Inverse-Gamma negative log-likelihood.
    Penalises both prediction error and overconfidence simultaneously.
    """
    two_be_lam = 2 * be * (1 + nu)
    nll = (
        0.5 * torch.log(torch.tensor(np.pi, device=y.device) / nu)
        - al * torch.log(two_be_lam)
        + (al + 0.5) * torch.log(nu * (y - gam) ** 2 + two_be_lam)
        + torch.lgamma(al) - torch.lgamma(al + 0.5)
    )
    return nll.mean()


def nig_reg_loss(y, gam, nu, al, be):
    """
    Evidential regularisation: penalise evidence on incorrectly estimated samples.
    Prevents the model from being confidently wrong.
    """
    err  = torch.abs(y - gam)
    evid = 2 * nu + al
    return (err * evid).mean()


def get_loss_weights(epoch):
    """
    Three-phase curriculum loss weights.

    Phase 1 (epochs  1-20): Huber only
    Phase 2 (epochs 21-50): Huber + KGE (weight 0.3)
    Phase 3 (epochs 51+  ): Huber + KGE + NIG-NLL + NIG-Reg
    """
    if epoch < 20:
        return dict(huber=1.0, kge=0.0, nig=0.0, nig_reg=0.0)
    elif epoch < 50:
        return dict(huber=1.0, kge=0.3, nig=0.0, nig_reg=0.0)
    else:
        return dict(huber=0.8, kge=0.2, nig=0.1, nig_reg=0.05)


def compute_total_loss(gam, nu, al, be, yb, weights):
    """
    Compute weighted sum of active loss terms.

    Args:
        gam, nu, al, be : NIG parameters from EvidentialHead
        yb              : ground truth soil moisture (B,)
        weights         : dict from get_loss_weights(epoch)

    Returns:
        total loss tensor
    """
    loss = weights["huber"] * huber_loss(gam, yb)
    if weights["kge"]     > 0:
        loss += weights["kge"]     * kge_loss(gam, yb)
    if weights["nig"]     > 0:
        loss += weights["nig"]     * nig_nll_loss(yb, gam, nu, al, be)
    if weights["nig_reg"] > 0:
        loss += weights["nig_reg"] * nig_reg_loss(yb, gam, nu, al, be)
    return loss


# ================================================================
# Evaluation Metrics
# ================================================================

def compute_metrics(obs, pred):
    """
    Compute RMSE, R2, NSE, and KGE between observations and predictions.

    Args:
        obs  : array-like, observed soil moisture values
        pred : array-like, predicted soil moisture values

    Returns:
        dict with keys: RMSE, R2, NSE, KGE
    """
    obs  = np.array(obs,  dtype=np.float64)
    pred = np.array(pred, dtype=np.float64)

    rmse = float(np.sqrt(mean_squared_error(obs, pred)))
    r2   = float(r2_score(obs, pred))
    nse  = float(1 - np.sum((obs - pred) ** 2) /
                 (np.sum((obs - obs.mean()) ** 2) + 1e-8))

    r = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 else np.nan
    b = pred.mean() / (obs.mean() + 1e-8)
    g = (pred.std() / (pred.mean() + 1e-8)) / \
        (obs.std()  / (obs.mean()  + 1e-8) + 1e-8)
    kge = float(1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)) \
          if not np.isnan(r) else float("nan")

    return dict(RMSE=rmse, R2=r2, NSE=nse, KGE=kge)
