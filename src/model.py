# ================================================================
# CYGNSS-SM-Net: Physics-informed Multimodal Deep Learning
# for SMAP-referenced Soil Moisture Estimation from CYGNSS
# ================================================================
# Architecture overview:
#   DDMBranch     -- multi-channel DDM + CBAM + soft peak localizer
#   WFBranch      -- multi-scale 1D CNN + BiGRU + AM/PM FiLM
#   TabBranch     -- physics-informed tabular MLP
#   MoEFusion     -- cross-modal attention + gated mixture-of-experts
#   EvidentialHead-- NIG distribution for calibrated uncertainty
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# DDM Branch Building Blocks
# ================================================================

class ChannelAttn(nn.Module):
    """Channel attention module (part of CBAM)."""
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
    """Spatial attention module (part of CBAM)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        a    = x.mean(1, keepdim=True)
        m, _ = x.max(1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([a, m], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttn(c)
        self.sa = SpatialAttn()

    def forward(self, x):
        return self.sa(self.ca(x))


class SoftPeakLocalizer(nn.Module):
    """
    Differentiable soft-argmax over the DDM.
    Learns peak location, value, and spread.
    Temperature annealed 1 -> 20 over training (curriculum schedule).
    """
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


class DDMBranch(nn.Module):
    """
    DDM encoding branch.
    Input : 3-channel DDM tensor (original + gradient + local variance), shape (B, 3, H, W)
    Output: 128-dimensional embedding
    Components:
        - Multi-scale dilated convolutions (fine/medium/coarse)
        - CBAM channel + spatial attention
        - Differentiable soft peak localizer
    """
    def __init__(self, in_ch=3, out_dim=128, H=17, W=11):
        super().__init__()

        def blk(ci, co, k, p, d=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, padding=p, dilation=d),
                nn.BatchNorm2d(co), nn.GELU())

        self.fine   = nn.Sequential(blk(in_ch, 32, 3, 1), blk(32, 32, 3, 1), CBAM(32))
        self.med    = nn.Sequential(blk(in_ch, 32, 3, 2, d=2), CBAM(32))
        self.coarse = nn.Sequential(blk(in_ch, 32, 5, 2), CBAM(32))
        self.merge  = nn.Sequential(
            nn.Conv2d(96, 64, 1), nn.BatchNorm2d(64), nn.GELU(), CBAM(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), CBAM(128),
            nn.AdaptiveAvgPool2d((2, 2)))
        self.pk   = SoftPeakLocalizer(H, W, 32)
        self.proj = nn.Linear(128 * 4 + 32, out_dim)

    def forward(self, x):
        s = torch.cat([self.fine(x), self.med(x), self.coarse(x)], 1)
        h = self.merge(s).flatten(1)
        return self.proj(torch.cat([h, self.pk(x)], 1))


# ================================================================
# Waveform Branch Building Blocks
# ================================================================

class AMPMFilm(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on AM/PM overpass timing.
    AM/PM affects surface temperature -> dielectric constant -> SM signal.
    Uses a learnable embedding (2 classes: AM=0, PM=1).
    """
    def __init__(self, feat_dim):
        super().__init__()
        self.emb   = nn.Embedding(2, 16)
        self.gamma = nn.Linear(16, feat_dim)
        self.beta  = nn.Linear(16, feat_dim)

    def forward(self, h, ampm):
        e = self.emb(ampm.long())
        return h * torch.sigmoid(self.gamma(e)) + self.beta(e)


class WFBranch(nn.Module):
    """
    Waveform encoding branch.
    Input : 3-channel waveform tensor (B, 3, 17) + physics features (B, 12) + AM/PM flag
    Output: 128-dimensional embedding
    Components:
        - Multi-scale 1D convolutions (k = 3, 5, 9)
        - Bidirectional GRU for sequential LE->TE patterns
        - AM/PM FiLM conditioning
        - Explicit waveform physics features (LE/TE slope, half-power width, asymmetry)
    """
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
        self.proj   = nn.Sequential(nn.Linear(192, out_dim), nn.GELU())

    def forward(self, x, xp, ampm):
        h      = torch.cat([self.fine(x), self.med(x), self.coarse(x)], 1)
        h      = self.merge(h)
        go, _  = self.gru(h.transpose(1, 2))
        hg     = go[:, -1, :]
        hg     = self.film(hg, ampm)
        p      = self.phys(xp)
        return self.proj(torch.cat([hg, p], 1))


# ================================================================
# Tabular Branch
# ================================================================

class TabBranch(nn.Module):
    """
    Tabular feature encoding branch (3-layer MLP).
    Input : 12 physics-informed features including spatial coordinates,
            collocation distance, log DDM power terms, AM/PM flag,
            and seasonal sin/cos encodings.
    Output: 128-dimensional embedding
    """
    def __init__(self, in_dim=12, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 128),    nn.GELU(),
            nn.Linear(128, out_dim))

    def forward(self, x):
        return self.net(x)


# ================================================================
# Fusion: Cross-Modal Attention + Gated MoE
# ================================================================

class CMA(nn.Module):
    """
    Cross-Modal Attention with residual connection and LayerNorm.
    Applied bidirectionally across all 3 modality pairs:
        DDM <-> Waveform, DDM <-> Tabular, Waveform <-> Tabular
    """
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


class MoEFusion(nn.Module):
    """
    Gated Mixture-of-Experts fusion module.
    Steps:
        1. Cross-modal attention enriches each modality embedding
        2. Concatenate enriched embeddings -> (B, 384)
        3. Soft router assigns weights to 4 expert networks
        4. Weighted expert outputs + residual skip -> LayerNorm
    """
    def __init__(self, d=128, E=4):
        super().__init__()
        # Bidirectional cross-modal attention pairs
        self.d2w = CMA(d); self.w2d = CMA(d)
        self.d2t = CMA(d); self.t2d = CMA(d)
        self.w2t = CMA(d); self.t2w = CMA(d)
        # Soft router
        self.gate = nn.Sequential(
            nn.Linear(d * 3, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, E), nn.Softmax(dim=-1))
        # 4 diverse experts with different hidden sizes
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d*3, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128)),
            nn.Sequential(nn.Linear(d*3, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 128)),
            nn.Sequential(nn.Linear(d*3, 128), nn.GELU(), nn.Linear(128, 128)),
            nn.Sequential(nn.Linear(d*3, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 128)),
        ])
        self.res  = nn.Linear(d * 3, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, f1, f2, f3):
        # Bidirectional cross-modal enrichment
        f1 = self.d2w(f1, f2); f1 = self.d2t(f1, f3)
        f2 = self.w2d(f2, f1); f2 = self.w2t(f2, f3)
        f3 = self.t2d(f3, f1); f3 = self.t2w(f3, f2)
        z    = torch.cat([f1, f2, f3], 1)
        w    = self.gate(z)
        outs = torch.stack([e(z) for e in self.experts], 1)
        return self.norm((w.unsqueeze(-1) * outs).sum(1) + self.res(z))


# ================================================================
# Evidential Regression Head
# ================================================================

class EvidentialHead(nn.Module):
    """
    Deep Evidential Regression head.
    Outputs Normal-Inverse-Gamma parameters (gamma, nu, alpha, beta).
    - Soil moisture estimate: SM_hat = 0.65 * sigmoid(gamma)
      -> physics-bounded to [0.0, 0.65] cm3/cm3
    - Predictive uncertainty: U = beta / (nu * (alpha - 1))
      -> calibrated epistemic uncertainty without Monte Carlo sampling
    Reference: Amini et al. (2020), NeurIPS.
    """
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
# Full CYGNSS-SM-Net
# ================================================================

class CYGNSSSMNet(nn.Module):
    """
    CYGNSS-SM-Net: Physics-informed Multimodal Deep Learning Framework
    for SMAP-referenced Soil Moisture Estimation from CYGNSS Observations.

    Args:
        tab_dim  (int): Number of tabular input features (default: 12)
        phys_dim (int): Number of waveform physics features (default: 12)
        H        (int): DDM height in delay bins (default: 17)
        W        (int): DDM width in Doppler bins (default: 11)

    Inputs:
        xd   (B, 3, H, W): 3-channel DDM tensor
        xw   (B, 3, 17)  : 3-channel waveform tensor
        xt   (B, tab_dim): Tabular feature vector
        xp   (B, 12)     : Waveform physics feature vector
        ampm (B,)        : AM/PM overpass flag (0=AM, 1=PM)

    Outputs:
        gam  (B,): Soil moisture estimate (cm3/cm3), bounded [0, 0.65]
        nu   (B,): NIG nu parameter
        al   (B,): NIG alpha parameter
        be   (B,): NIG beta parameter
        Uncertainty: U = be / (nu * (al - 1))
    """
    def __init__(self, tab_dim=12, phys_dim=12, H=17, W=11):
        super().__init__()
        self.ddm  = DDMBranch(3, 128, H, W)
        self.wf   = WFBranch(phys_dim, 128)
        self.tab  = TabBranch(tab_dim, 128)
        self.fuse = MoEFusion(128, 4)
        self.head = EvidentialHead(128)

    def forward(self, xd, xw, xt, xp, ampm):
        f1    = self.ddm(xd)
        f2    = self.wf(xw, xp, ampm)
        f3    = self.tab(xt)
        fused = self.fuse(f1, f2, f3)
        return self.head(fused)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
