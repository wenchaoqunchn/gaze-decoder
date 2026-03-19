# shared/models.py
# ─────────────────────────────────────────────────────────────────────────────
# GazeDecoder V3 — All model definitions
#
# V3.1 upgrades (targeting F1 > 0.93):
#   1. IIB upgraded from concat-FFN to MHA cross-attention (Q=behav, KV=ctx)
#      → per-timestep selective context injection instead of uniform blending
#   2. ctx_proj upgraded from 768→d_model (1 layer) to 768→2d→d_model (2 layer)
#      → reduces information bottleneck in semantic compression
#   3. Training: val/test leak fixed — val set now uses a held-out participant
#      from train split; test indices never used for checkpoint selection
#
# ── Ablation goal ──────────────────────────────────────────────────────────
# Find the optimal ChronosXOfficial variant by exploring different behavioral
# input channel (b_in) compositions.
#
# V2 exploration revealed that what goes into b_in critically affects IIB's
# ability to model temporal dynamics.  Three channel sources are available:
#
#   Spatial (2d)  : normalised gaze position (x, y) — always time-varying
#   Layer1  (8d)  : micro-window stats (fixation_ratio, mean_fix_dur,
#                   saccade_amplitude, regression_rate, direction_entropy,
#                   revisit_density, sf_ratio, velocity) — per timestep
#   Layer2  (8d)  : window-level macro stats (fixation_count_norm,
#                   saccade_count_norm, total_fix_time_ratio, spatial_density,
#                   convex_hull_area, attention_switch_freq, velocity_trend,
#                   scan_linearity) — broadcast constant for the whole window
#
# Two ways Layer2 can be injected:
#   "bc"  (broadcast) : L2 values repeated along T dimension → enter b_in
#   "oib" (OIB gate)  : L2 projected and fused into OIB's context summary
#
# ChronosXOfficial ablation variants (CHRONOSX_VARIANTS):
#
#  Group A — vary b_in, OIB uses plain semantic ctx only (no L2)
#   Bchan_Spatial           b_in = Spatial(2)             OIB cond = mean(ctx)
#   Bchan_L1                b_in = L1(8)                  OIB cond = mean(ctx)
#   Bchan_Spatial_L1        b_in = Spatial+L1(10)         OIB cond = mean(ctx)  ← V2 BCBest
#   Bchan_Spatial_L1_L2bc   b_in = Spatial+L1+L2bc(18)   OIB cond = mean(ctx)  ← V2 BC-Full
#
#  Group B — vary b_in, OIB fuses semantic ctx with L2 projection
#   Bchan_Spatial_L2oib     b_in = Spatial(2)             OIB cond = fused(ctx+L2) ← V2 OIBBest
#   Bchan_Spatial_L1_L2oib  b_in = Spatial+L1(10)        OIB cond = fused(ctx+L2) ← V2 Hybrid
#   Bchan_Full_L2oib        b_in = Spatial+L1+L2bc(18)   OIB cond = fused(ctx+L2)
#
# Baselines (12 total):
#   ML : XGBoost, RandomForest, LightGBM
#   DL : BiLSTM, 1D-CNN, TransformerEnc, PatchTST, iTransformer,
#        TimesNet, DLinear, Mamba, TimesBERT
# ─────────────────────────────────────────────────────────────────────────────
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.config import FEAT_DIM, WINDOW_SIZE, SEED

# ─────────────────────────────────────────────────────────────────────────────
# §A  ChronosX variants
# ─────────────────────────────────────────────────────────────────────────────


class _CovariateInjectionBlock(nn.Module):
    """Lightweight covariate injection (V1 style): concat → ReLU → Linear."""

    def __init__(
        self, behav_dim: int, covariate_dim: int, hidden_dim: int, model_dim: int
    ):
        super().__init__()
        self.behav_in = nn.Linear(behav_dim, hidden_dim)
        self.cov_in = nn.Linear(covariate_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(
        self, behav_embeds: torch.Tensor, covariates: torch.Tensor
    ) -> torch.Tensor:
        z_b = self.behav_in(behav_embeds)
        z_c = self.cov_in(covariates)
        out = self.fusion(torch.cat([z_b, z_c], dim=-1))
        return behav_embeds + out  # residual


class ChronosXV1(nn.Module):
    """
    ChronosXV1 — faithful port of V1 best-result architecture.
    3-layer standard Conv1d + CovariateInjectionBlock + mean-pool head.
    No Transformer, no cross-attention.

    b_in = Spatial(2) + Layer1(8) = 10d  (V3 aligned)
    """

    _BEHAV_DIM = 10  # Spatial(2) + Layer1(8)
    _CTX_DIM = 768  # TextEmbed(384) + CodeEmbed(384)

    def __init__(self, d_model: int = 128, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.behav_cnn = nn.Sequential(
            nn.Conv1d(self._BEHAV_DIM, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.injection = _CovariateInjectionBlock(
            behav_dim=d_model,
            covariate_dim=self._CTX_DIM,
            hidden_dim=hidden_dim,
            model_dim=d_model,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        behavioral = torch.cat(
            [x[:, :, :2], x[:, :, 770:778]], dim=-1
        )  # Spatial+L1=10d
        covariates = x[:, :, 2:770]
        z = self.behav_cnn(behavioral.transpose(1, 2)).transpose(1, 2)
        z = self.norm(self.injection(z, covariates))
        return self.head(z.mean(dim=1))


class ChronosX(nn.Module):
    """
    ChronosX — Cross-attention fusion (V2 design).
    Behavioral pathway: 1D-CNN.
    Context pathway: Self-Attention.
    Fusion: Cross-Attention (Q=behav, KV=context).

    b_in = Spatial(2) + Layer1(8) = 10d  (V3 aligned)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_ctx_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.behav_proj = nn.Sequential(nn.Linear(10, d_model), nn.LayerNorm(d_model))
        self.behav_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model // 4),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model // 4),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.behav_norm = nn.LayerNorm(d_model)
        self.ctx_proj = nn.Linear(768, d_model)
        _enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.ctx_encoder = nn.TransformerEncoder(_enc, num_layers=n_ctx_layers)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_in = torch.cat([x[:, :, :2], x[:, :, 770:778]], dim=-1)  # Spatial+L1=10d
        c_in = x[:, :, 2:770]
        b = self.behav_proj(b_in)
        b_cnn = self.behav_cnn(b.transpose(1, 2))
        b = self.behav_norm(b + b_cnn.transpose(1, 2))
        c = self.ctx_encoder(self.ctx_proj(c_in))
        b_attn, _ = self.cross_attn(query=b, key=c, value=c)
        b_attn = self.cross_norm(b + b_attn)
        fused = self.fusion_norm(
            self.fusion(torch.cat([b_attn.mean(1), c.mean(1)], dim=-1))
        )
        return self.head(fused)


class IIB(nn.Module):
    """
    Input Injection Block (arXiv:2503.12107 §3.1), upgraded to cross-attention.

    V3.1 change: replaces concat-FFN with MHA cross-attention (Q=behav, KV=ctx),
    so each behavioral timestep attends to the most relevant context positions
    rather than receiving a uniform context injection.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Cross-attention: behavioral tokens query into context tokens
        nhead = max(1, d_model // 16)  # e.g. d_model=128 → 8 heads
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        # e      : [B, T, d]  behavioral tokens
        # x_cov  : [B, T, d]  context tokens (already projected to d_model)
        attn_out, _ = self.cross_attn(query=e, key=x_cov, value=x_cov)
        e = self.norm1(e + self.dropout(attn_out))  # residual + norm
        e = self.norm2(e + self.ffn(e))  # FFN + residual + norm
        return e


class OIB(nn.Module):
    """Output Injection Block (arXiv:2503.12107 §3.2, adapted for classification)."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_h = nn.Linear(d_model, hidden_dim)
        self.W_f = nn.Linear(d_model, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_last: torch.Tensor, ctx_summary: torch.Tensor) -> torch.Tensor:
        z_h = self.W_h(h_last)
        z_f = self.W_f(ctx_summary)
        refined = self.proj(torch.cat([z_h, z_f], dim=-1))
        return self.norm(h_last + refined)


class ChronosXOfficial(nn.Module):
    """
    ChronosXOfficial — IIB + Transformer Backbone + OIB
    (arXiv:2503.12107, adapted for gaze classification).

    Pipeline:
      1. Split: behavioral [B,T,14→10] / semantic [B,T,768]
         b_in = Spatial(2) + Layer1(8) = 10d   (V3 default; L2 excluded from b_in)
         c_in = Text(384) + Code(384) = 768d
      2. Tokenise both → d_model tokens
      3. IIB: inject context into behavioral tokens
      4. Backbone: CLS + TransformerEncoder
      5. OIB: context summary modulates CLS output
      6. MLP head → logits [B,2]

    Note: this class is used by baselines.ipynb as the "proposed method" reference.
    It uses Spatial+L1 as b_in (V2 BCBest configuration).
    For the full b_in ablation see CHRONOSX_VARIANTS in ablation.ipynb.
    """

    _BEHAV_DIM = 10  # Spatial(2) + Layer1(8)
    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.behav_proj = nn.Sequential(
            nn.Linear(self._BEHAV_DIM, d_model), nn.LayerNorm(d_model)
        )
        # Two-stage context projection: 768 → 256 → d_model
        # Reduces information bottleneck from direct 768→128 compression
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=n_backbone_layers)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # b_in = Spatial(2) + Layer1(8) = 10d  (V3 default; avoids L2 DC bias)
        b_in = torch.cat([x[:, :, :2], x[:, :, 770:778]], dim=-1)
        c_in = x[:, :, 2:770]
        e = self.behav_proj(b_in)
        v = self.ctx_proj(c_in)
        e_prime = self.iib(e, v)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone(torch.cat([cls, e_prime], dim=1))
        h_last = self.backbone_norm(h[:, 0, :])
        h_prime = self.oib(h_last, v.mean(dim=1))
        return self.head(h_prime)


# ─────────────────────────────────────────────────────────────────────────────
# §A2  ChronosXOfficial behavioral-channel (b_in) ablation variants
#
# Goal: find the optimal b_in composition for ChronosXOfficial.
#
# Feature layout in the raw input tensor x [B, T, FEAT_DIM]:
#   x[:, :,   0:2  ] = Spatial   (normalised gaze x, y)
#   x[:, :,   2:770] = Semantic  (Text-384 + Code-384 embeddings)  → ctx only
#   x[:, :, 770:778] = Layer1    (8d per-timestep micro-window stats)
#   x[:, :, 778:786] = Layer2    (8d window-level macro stats, broadcast)
#
# NOTE: V3 dataset FEAT_DIM = 786 (adds Layer1+Layer2 on top of V2's 782).
#       If the dataset still has 782 dims, Layer1/Layer2 slices will be zeros.
#       The dataset builder (features.py) must output 786-dim vectors.
#
# Two L2 injection modes:
#   "bc"  (broadcast) : L2 is part of b_in directly (constant per window)
#   "oib" (gate)      : L2 is projected and fused into OIB's context summary
# ─────────────────────────────────────────────────────────────────────────────

# Feature slice constants (shared across all variant classes)
_S_SPATIAL = slice(0, 2)  # 2d  gaze position
_S_SEMANTIC = slice(2, 770)  # 768d text+code embeddings  (ctx stream)
_S_L1 = slice(770, 778)  # 8d  Layer-1 micro-window stats
_S_L2 = slice(778, 786)  # 8d  Layer-2 window macro stats
_N_L1 = 8
_N_L2 = 8


def _make_backbone(
    d_model: int, nhead: int, n_layers: int, dropout: float
) -> nn.TransformerEncoder:
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(enc_layer, num_layers=n_layers)


def _init_weights_v3(module: nn.Module, cls_token: nn.Parameter):
    nn.init.trunc_normal_(cls_token, std=0.02)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class _BchanBase(nn.Module):
    """
    Shared skeleton for all b_in channel ablation variants.

    Parameters
    ----------
    behav_dim      : int   – total dimension of b_in (determined by which
                             channels are included)
    use_l2_oib     : bool  – if True, project L2 and fuse into OIB condition
                             (Group B variants); if False, OIB uses mean(ctx) only
    d_model / nhead / n_backbone_layers / iib_hidden / oib_hidden / dropout :
                             standard architecture hyper-parameters (same as
                             ChronosXOfficial defaults)
    """

    _CTX_DIM = 768

    def __init__(
        self,
        behav_dim: int,
        use_l2_oib: bool = False,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = behav_dim
        self.use_l2_oib = use_l2_oib
        self.d_model = d_model

        self.behav_proj = nn.Sequential(
            nn.Linear(behav_dim, d_model), nn.LayerNorm(d_model)
        )
        # Two-stage context projection: 768 → 256 → d_model
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)

        if use_l2_oib:
            # L2 → d_model projection; fused with mean(ctx) before OIB
            self.l2_proj = nn.Sequential(
                nn.Linear(_N_L2, d_model), nn.LayerNorm(d_model)
            )
            self.l2_fusion = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU())

        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def _build_b_in(self, x: torch.Tensor) -> torch.Tensor:
        """Subclasses override this to select which channels enter b_in."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        b_in = self._build_b_in(x)  # [B, T, behav_dim]
        c_in = x[:, :, _S_SEMANTIC]  # [B, T, 768]

        e = self.behav_proj(b_in)  # [B, T, d]
        v = self.ctx_proj(c_in)  # [B, T, d]
        e_p = self.iib(e, v)  # IIB: inject context into behavioral

        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(
            self.backbone(torch.cat([cls, e_p], dim=1))[:, 0]
        )  # CLS token after backbone

        if self.use_l2_oib:
            # L2 projected → fuse with semantic summary → OIB condition
            l2 = x[:, :, _S_L2].mean(dim=1)  # [B, 8]  window-level mean of L2
            mac = self.l2_proj(l2)  # [B, d]
            cond = self.l2_fusion(torch.cat([v.mean(dim=1), mac], dim=-1))  # [B, d]
        else:
            cond = v.mean(dim=1)  # plain semantic ctx summary

        return self.head(self.oib(h, cond))


# ── Group A — plain OIB (OIB cond = mean(ctx) only) ──────────────────────────


class Bchan_Spatial(_BchanBase):
    """b_in = Spatial(2) only.  Minimum possible behavioral input."""

    def __init__(self, **kw):
        super().__init__(behav_dim=2, use_l2_oib=False, **kw)

    def _build_b_in(self, x):
        return x[:, :, _S_SPATIAL]


class Bchan_L1(_BchanBase):
    """b_in = Layer1(8) only.  Micro-window stats without spatial anchor."""

    def __init__(self, **kw):
        super().__init__(behav_dim=_N_L1, use_l2_oib=False, **kw)

    def _build_b_in(self, x):
        return x[:, :, _S_L1]


class Bchan_Spatial_L1(_BchanBase):
    """
    b_in = Spatial(2) + Layer1(8) = 10d.
    OIB cond = mean(ctx).
    Matches V2 BCBest — removing L2-broadcast from b_in gave +0.012 F1.
    """

    def __init__(self, **kw):
        super().__init__(behav_dim=2 + _N_L1, use_l2_oib=False, **kw)

    def _build_b_in(self, x):
        return torch.cat([x[:, :, _S_SPATIAL], x[:, :, _S_L1]], dim=-1)


class Bchan_Spatial_L1_L2bc(_BchanBase):
    """
    b_in = Spatial(2) + Layer1(8) + Layer2-broadcast(8) = 18d.
    OIB cond = mean(ctx).
    Matches V2 BC-Full — constant L2 values in b_in hurt IIB (DC offset problem).
    """

    def __init__(self, **kw):
        super().__init__(behav_dim=2 + _N_L1 + _N_L2, use_l2_oib=False, **kw)

    def _build_b_in(self, x):
        return torch.cat(
            [
                x[:, :, _S_SPATIAL],
                x[:, :, _S_L1],
                x[:, :, _S_L2],
            ],
            dim=-1,
        )


# ── Group B — L2-OIB injection (OIB cond = fused(ctx + L2 projection)) ───────


class Bchan_Spatial_L2oib(_BchanBase):
    """
    b_in = Spatial(2) only.  OIB cond = fused(ctx + L2).
    Matches V2 OIBBest — L1 near-redundant once Spatial is present.
    """

    def __init__(self, **kw):
        super().__init__(behav_dim=2, use_l2_oib=True, **kw)

    def _build_b_in(self, x):
        return x[:, :, _S_SPATIAL]


class Bchan_Spatial_L1_L2oib(_BchanBase):
    """
    b_in = Spatial(2) + Layer1(8) = 10d.  OIB cond = fused(ctx + L2).
    Combines BCBest's b_in with OIBBest's L2 gate.
    Matches V2 Hybrid / OIB-Full (F1≈0.9258 in exploratory round).
    """

    def __init__(self, **kw):
        super().__init__(behav_dim=2 + _N_L1, use_l2_oib=True, **kw)

    def _build_b_in(self, x):
        return torch.cat([x[:, :, _S_SPATIAL], x[:, :, _S_L1]], dim=-1)


class Bchan_Full_L2oib(_BchanBase):
    """
    b_in = Spatial(2) + Layer1(8) + Layer2-broadcast(8) = 18d.
    OIB cond = fused(ctx + L2).
    Tests whether adding broadcast L2 to b_in on top of OIB gate helps or hurts.
    """

    def __init__(self, **kw):
        super().__init__(behav_dim=2 + _N_L1 + _N_L2, use_l2_oib=True, **kw)

    def _build_b_in(self, x):
        return torch.cat(
            [
                x[:, :, _S_SPATIAL],
                x[:, :, _S_L1],
                x[:, :, _S_L2],
            ],
            dim=-1,
        )


# ── Group D — Architecture depth variants (motivated by V3.1 cross-attn gain) ─
#
# Finding from V3.1: the IIB cross-attention upgrade drove most of the 0.92→0.94
# improvement.  Three depth extensions are tested:
#
#   Bchan_Spatial_IIB2      IIB stacked ×2 (iterative ctx refinement)
#   Bchan_Spatial_CtxSA     ctx tokens pass through 1-layer self-attn before IIB KV
#   Bchan_Spatial_OIBxattn  OIB upgraded from concat-FFN to cross-attn (Q=CLS, KV=ctx)
#
# All three use Spatial(2) as b_in (confirmed best b_in from Group A–C ablation).


class IIBStack(nn.Module):
    """
    Two stacked IIB cross-attention blocks.

    Block 1: e attends to original ctx tokens v  (coarse semantic alignment)
    Block 2: enriched e attends to v again        (fine-grained refinement)

    Using the same v for both blocks avoids gradient vanishing and keeps
    the ctx signal stable; each block adds a residual so information is
    never lost.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block1 = IIB(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
        self.block2 = IIB(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, e: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        e = self.block1(e, v)
        e = self.block2(e, v)
        return e


class Bchan_Spatial_IIB2(nn.Module):
    """
    b_in = Spatial(2).  IIB stacked ×2.  OIB cond = mean(ctx).

    Motivation: single IIB cross-attn makes one pass of Q→KV alignment.
    Stacking two IIB blocks allows iterative refinement: the first block
    provides coarse semantic grounding, the second performs fine-grained
    per-timestep selection on the already-enriched behavioral tokens.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.iib = IIBStack(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, _S_SPATIAL])  # [B, T, d]
        v = self.ctx_proj(x[:, :, _S_SEMANTIC])  # [B, T, d]
        e_p = self.iib(e, v)  # 2× cross-attn
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v.mean(dim=1)))


class Bchan_Spatial_CtxSA(nn.Module):
    """
    b_in = Spatial(2).  ctx tokens pass through 1-layer self-attn before IIB KV.
    OIB cond = mean(enriched ctx).

    Motivation: ctx_proj currently maps each frame's 768d embedding independently.
    Frame-level embeddings from the same window share lexical/structural context
    but have no inter-frame interaction after projection.  Adding one self-attention
    layer lets ctx tokens exchange information across T timesteps, producing richer
    KV representations for the IIB cross-attention query.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        # 1-layer self-attention to enrich ctx tokens before they become IIB KV
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, _S_SPATIAL])  # [B, T, d]
        v = self.ctx_proj(x[:, :, _S_SEMANTIC])  # [B, T, d]
        v = self.ctx_sa(v)  # ctx self-attn: KV enrichment
        e_p = self.iib(e, v)  # IIB cross-attn
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v.mean(dim=1)))


class OIBCrossAttn(nn.Module):
    """
    OIB upgraded from concat-FFN to cross-attention.

    Q = h_last (CLS token, [B,1,d]), KV = v (ctx sequence, [B,T,d]).
    The CLS token attends to the full ctx sequence rather than receiving
    a fixed mean(ctx) summary, letting it focus on the most decision-relevant
    context positions at the output stage.
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        nhead = max(1, d_model // 16)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_last: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # h_last : [B, d]   CLS output
        # v      : [B, T, d] ctx token sequence
        q = h_last.unsqueeze(1)  # [B, 1, d]
        attn_out, _ = self.cross_attn(query=q, key=v, value=v)
        q = self.norm1(q + self.dropout(attn_out))  # [B, 1, d]
        q = self.norm2(q + self.ffn(q))  # [B, 1, d]
        return q.squeeze(1)  # [B, d]


class Bchan_Spatial_OIBxattn(nn.Module):
    """
    b_in = Spatial(2).  OIB replaced by cross-attention (Q=CLS, KV=ctx).
    IIB is the standard single cross-attn block.

    Motivation: OIB currently refines the CLS token using a fixed mean(ctx)
    summary vector.  Replacing it with cross-attention lets the CLS token
    selectively attend to the T most relevant ctx positions, mirroring the
    IIB upgrade from V3.0→V3.1.  This closes the "asymmetry" where IIB uses
    full-sequence KV but OIB still uses a single pooled vector.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        # Replace OIB with cross-attention block
        self.oib = OIBCrossAttn(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, _S_SPATIAL])  # [B, T, d]
        v = self.ctx_proj(x[:, :, _S_SEMANTIC])  # [B, T, d]
        e_p = self.iib(e, v)  # IIB cross-attn
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v))  # OIB cross-attn (KV=full ctx)


class Bchan_Spatial_CtxSA_OIBxattn(nn.Module):
    """
    b_in = Spatial(2).
    CtxSA  : ctx tokens enriched by 1-layer self-attn before IIB KV.
    OIBxattn: OIB replaced by cross-attn (Q=CLS, KV=enriched ctx).

    Combination of the two best Group-D variants:
      - CtxSA  (+0.0039 over Bchan_Spatial, best F1_issue)
      - OIBxattn (+0.0032 over Bchan_Spatial, best F1_macro & precision)

    Both improvements target the ctx stream independently:
      CtxSA  acts on ctx BEFORE it reaches IIB (enriches KV for query alignment)
      OIBxattn acts on ctx AFTER backbone (enriches OIB condition for output refinement)
    There is no overlap in parameter scope, so their gains should be additive.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        # CtxSA: 1-layer self-attn to enrich ctx tokens before IIB KV
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        # OIBxattn: cross-attn OIB (Q=CLS, KV=enriched ctx sequence)
        self.oib = OIBCrossAttn(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, _S_SPATIAL])  # [B, T, d]
        v = self.ctx_proj(x[:, :, _S_SEMANTIC])  # [B, T, d]
        v = self.ctx_sa(v)  # CtxSA: enrich KV
        e_p = self.iib(e, v)  # IIB cross-attn (enriched KV)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v))  # OIBxattn (KV=enriched ctx)


# ── Group C — Fine-grained channel variants (motivated by V3.1 ablation) ─────
#
# Finding: Bchan_Spatial (2d) beats all richer b_in variants because L1 features
# are largely redundant derivatives of (x,y): velocity, dispersion, saccade_amp
# are all recoverable from the raw trajectory.  However two L1 channels are
# genuinely orthogonal to Spatial:
#   L1[6] direction_change  — angular turn-rate (not linearly deducible from x,y)
#   L1[7] acceleration      — jerk proxy (2nd derivative of speed, high-freq signal)
#
# Bchan_Spatial_DirAcc tests whether these two truly-new dimensions help.
#
# Bchan_Spatial_FrameDiff replaces all L1 statistics with explicit frame
# differences (Δx, Δy), giving the model raw velocity without any micro-window
# averaging that smears temporal resolution.  Unlike L1 statistics it preserves
# per-frame sharpness, which matters for cross-attention query alignment.


class Bchan_Spatial_DirAcc(_BchanBase):
    """
    b_in = Spatial(2) + direction_change(1) + acceleration(1) = 4d.
    OIB cond = mean(ctx).

    Motivation: of the 8 L1 channels, only direction_change (L1[6]) and
    acceleration (L1[7]) are orthogonal to (x,y) — they capture angular
    turn-rate and jerk that cannot be recovered from position alone.
    All other L1 channels are statistics of distance/velocity, which are
    first-order derivatives of Spatial and therefore redundant as IIB queries.

    Hypothesis: adding exactly these two channels extends query expressivity
    without introducing the redundancy noise that degrades Bchan_Spatial_L1.
    """

    def __init__(self, **kw):
        super().__init__(behav_dim=4, use_l2_oib=False, **kw)

    def _build_b_in(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x[:, :, _S_SPATIAL]  # [B, T, 2]  (x, y)
        dir_chg = x[:, :, _S_L1][:, :, 6:7]  # [B, T, 1]  direction_change
        accel = x[:, :, _S_L1][:, :, 7:8]  # [B, T, 1]  acceleration
        return torch.cat([spatial, dir_chg, accel], dim=-1)  # [B, T, 4]


class Bchan_Spatial_FrameDiff(_BchanBase):
    """
    b_in = Spatial(x,y) + FrameDiff(Δx,Δy) = 4d.
    OIB cond = mean(ctx).

    Motivation: L1 velocity/saccade features derive from a micro-window of
    16 frames, losing per-frame temporal resolution.  FrameDiff provides the
    same velocity signal at full per-frame resolution (Δx = x_t − x_{t−1}),
    which better preserves the sharp onset of issue-related gaze transitions.
    The first frame's diff is set to zero (causal padding).

    Unlike Bchan_Spatial_DirAcc this variant encodes velocity directly in
    Cartesian form rather than polar statistics, keeping the spatial
    coordinate frame consistent throughout b_in.
    """

    def __init__(self, **kw):
        super().__init__(behav_dim=4, use_l2_oib=False, **kw)

    def _build_b_in(self, x: torch.Tensor) -> torch.Tensor:
        s = x[:, :, _S_SPATIAL]  # [B, T, 2]  (x, y)
        diff = torch.zeros_like(s)  # [B, T, 2]  (Δx, Δy)
        diff[:, 1:, :] = s[:, 1:, :] - s[:, :-1, :]  # causal: frame 0 → 0
        return torch.cat([s, diff], dim=-1)  # [B, T, 4]


# ── Group F — Context-stream ablation (based on Bchan_Spatial_CtxSA) ─────────
#
# The 768d ctx stream is the concatenation of two independently trained
# sentence-transformers:
#   [2:386]   embed_text  (384d) — natural-language description of the AOI element
#   [386:770] embed_code  (384d) — source-code embedding of the element
#
# Both halves are projected jointly by ctx_proj → d_model, then enriched by
# ctx_sa (1-layer self-attn) before serving as IIB KV.
#
# Group F asks: which half of the ctx stream drives the observed improvement?
#
#   CtxSA_NoCode  — keep only embed_text (384d); zero out embed_code
#   CtxSA_NoText  — keep only embed_code (384d); zero out embed_text
#   CtxSA_NoCtx     — remove ctx stream entirely; IIB receives zeros (pure-behav)
#
# All three share the same architecture as Bchan_Spatial_CtxSA so that any
# performance difference isolates the contribution of the removed sub-stream.

_S_TEXT = slice(2, 386)  # 384d  natural-language description embedding
_S_CODE = slice(386, 770)  # 384d  source-code embedding


class CtxSA_NoCode(nn.Module):
    """
    w/o Code embedding: ctx = embed_text (384d) only; embed_code slot zeroed.

    Naming: "NoCode" = Text stream retained, Code stream ablated.
    Tests whether the natural-language description of the viewed element
    alone matches full-ctx performance.  If so, the code embedding is
    redundant for gaze-issue prediction.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2  # Spatial(x, y) — full gaze trajectory retained
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def _mask_ctx(self, x: torch.Tensor) -> torch.Tensor:
        """Keep embed_text [2:386]; zero-fill embed_code [386:770]."""
        ctx = x[:, :, _S_SEMANTIC].clone()  # [B, T, 768]
        ctx[:, :, 384:] = 0.0  # zero embed_code half
        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, _S_SPATIAL])
        v = self.ctx_proj(self._mask_ctx(x))
        v = self.ctx_sa(v)
        e_p = self.iib(e, v)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v.mean(dim=1)))


class CtxSA_NoText(nn.Module):
    """
    w/o Text embedding: ctx = embed_code (384d) only; embed_text slot zeroed.

    Naming: "NoText" = Code stream retained, Text stream ablated.
    Tests whether the source-code embedding of the viewed element alone
    matches full-ctx performance.  Code embeddings capture structural
    syntax patterns; if they suffice, semantic descriptions add no value.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2  # Spatial(x, y) — full gaze trajectory retained
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def _mask_ctx(self, x: torch.Tensor) -> torch.Tensor:
        """Keep embed_code [386:770]; zero-fill embed_text [2:386]."""
        ctx = x[:, :, _S_SEMANTIC].clone()  # [B, T, 768]
        ctx[:, :, :384] = 0.0  # zero embed_text half
        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, _S_SPATIAL])
        v = self.ctx_proj(self._mask_ctx(x))
        v = self.ctx_sa(v)
        e_p = self.iib(e, v)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v.mean(dim=1)))


class CtxSA_NoCtx(nn.Module):
    """
    w/o Semantic: ctx stream replaced by zeros (pure behavioral model).

    This is the strongest ablation: IIB and OIB still exist structurally
    but receive no real context signal (KV = projected zeros).  Any gap
    between this and Bchan_Spatial_CtxSA measures the total contribution
    of the exogenous context stream to classification performance.

    Note: this is architecturally distinct from Bchan_Spatial (which has no
    IIB/OIB at all).  If CtxSA_NoCtx ≈ Bchan_Spatial, the IIB/OIB modules
    themselves add no overhead when ctx is absent; if it is worse, the
    injected zeros harm the backbone representation.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2  # Spatial(x, y) only
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        e = self.behav_proj(x[:, :, _S_SPATIAL])
        # Replace semantic stream with zeros — no exogenous context
        v_zeros = torch.zeros(B, T, self._CTX_DIM, device=x.device, dtype=x.dtype)
        v = self.ctx_proj(v_zeros)
        v = self.ctx_sa(v)
        e_p = self.iib(e, v)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v.mean(dim=1)))


# ── Group G — Module-contribution ablation (based on Bchan_Spatial_CtxSA) ────
#
# Each variant removes or bypasses exactly ONE module from the full pipeline:
#
#   Full pipeline:
#     Spatial → behav_proj → e  ─────────────────────────────────────────┐
#     Semantic → ctx_proj → ctx_sa → v  ──→  IIB(Q=e, KV=v) → e_p       │
#                                             backbone(CLS+e_p) → h       │
#                                             OIB(h, mean(v)) → head      │
#
#   CtxSA_NoBehav    : zero b_in  → e = behav_proj(zeros)
#                      Measures the total contribution of the gaze trajectory.
#                      If performance collapses, gaze is the primary signal.
#
#   CtxSA_NoIIB      : skip IIB  → e_p = e  (no ctx injection into behav tokens)
#                      Measures IIB cross-attention contribution.
#
#   CtxSA_NoOIB      : skip OIB  → head(h) directly  (no output refinement)
#                      Measures OIB output-stage refinement contribution.
#
#   CtxSA_1StageProj : replace two-stage ctx_proj (768→256→128) with
#                      single linear (768→128).
#                      Isolates the contribution of the V3.1 two-stage bottleneck.


class CtxSA_NoBehav(nn.Module):
    """
    w/o behavioral stream: b_in replaced with zeros.

    The IIB Q-queries are projections of a zero tensor, so IIB receives no
    positional/trajectory information.  Any residual performance comes purely
    from the ctx stream (text+code embeddings) flowing through the backbone
    via IIB's KV attention and the OIB gate.

    Expected outcome: large drop → gaze trajectory is the primary discriminator.
    If drop is moderate, semantic context alone carries partial signal.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2  # Spatial(x, y) — zeroed at runtime
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # Replace behavioral stream with zeros — no gaze trajectory
        e_zeros = torch.zeros(B, T, 2, device=x.device, dtype=x.dtype)
        e = self.behav_proj(e_zeros)
        v = self.ctx_proj(x[:, :, _S_SEMANTIC])
        v = self.ctx_sa(v)
        e_p = self.iib(e, v)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v.mean(dim=1)))


class CtxSA_NoIIB(nn.Module):
    """
    w/o IIB: behavioral tokens enter the backbone directly without ctx injection.

    The ctx stream (v) still flows through ctx_proj → ctx_sa → OIB, so the
    model retains ctx-conditioned output refinement.  Only the frame-level
    cross-attention between gaze queries and ctx keys is removed.

    If performance drops significantly, IIB's per-frame context alignment is
    the critical innovation; if OIB alone suffices, the output-stage gate is
    the more important component.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2  # Spatial(x, y) only
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # No IIB — behavioral tokens go straight to backbone
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, _S_SPATIAL])  # [B, T, d]  — no IIB injection
        v = self.ctx_proj(x[:, :, _S_SEMANTIC])
        v = self.ctx_sa(v)
        # Skip IIB: e_p = e (behavioral tokens unchanged by ctx)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e], dim=1))[:, 0])
        return self.head(self.oib(h, v.mean(dim=1)))


class CtxSA_NoOIB(nn.Module):
    """
    w/o OIB: backbone CLS token feeds the head directly (no output refinement).

    The ctx stream still flows through ctx_proj → ctx_sa → IIB, so frame-level
    context injection is preserved.  Only the output-stage ctx modulation is
    removed — the classifier head sees raw CLS without ctx-gated refinement.

    Complements CtxSA_NoIIB: together they decompose ctx contributions into
    input-stage (IIB) vs output-stage (OIB) effects.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2  # Spatial(x, y) only
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        # No OIB — head receives backbone CLS directly
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, _S_SPATIAL])
        v = self.ctx_proj(x[:, :, _S_SEMANTIC])
        v = self.ctx_sa(v)
        e_p = self.iib(e, v)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        # Skip OIB: head receives CLS directly
        return self.head(h)


class CtxSA_1StageProj(nn.Module):
    """
    Single-stage ctx projection: 768 → d_model (one Linear + LayerNorm).

    V3.1 introduced a two-stage projection (768 → 256 → 128) to reduce the
    information bottleneck when compressing 768d embeddings to 128d.  This
    variant reverts to the V3.0 single-stage design to isolate that change's
    contribution.

    If performance drops vs Bchan_Spatial_CtxSA, the two-stage projection
    is a meaningful improvement; if parity holds, the single layer suffices.
    """

    _CTX_DIM = 768

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2  # Spatial(x, y) only
        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        # Single-stage: 768 → d_model directly (no intermediate 256d layer)
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, _S_SPATIAL])
        v = self.ctx_proj(x[:, :, _S_SEMANTIC])  # single-stage projection
        v = self.ctx_sa(v)
        e_p = self.iib(e, v)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v.mean(dim=1)))


# ─────────────────────────────────────────────────────────────────────────────
# §B  Classic DL baselines
# ─────────────────────────────────────────────────────────────────────────────


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = FEAT_DIM,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.head(h)


class CNN1DClassifier(nn.Module):
    def __init__(
        self, input_dim: int = FEAT_DIM, d_model: int = 128, dropout: float = 0.2
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, 7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(self.proj(x).transpose(1, 2)).mean(dim=-1)
        return self.head(z)


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = FEAT_DIM,
        d_model: int = 128,
        nhead: int = 8,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.proj(x)).mean(dim=1))


class DLinearClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = FEAT_DIM,
        seq_len: int = WINDOW_SIZE,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.linear = nn.Linear(seq_len, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(self.proj(x).transpose(1, 2)).squeeze(-1)
        return self.head(z)


class PatchTSTClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = FEAT_DIM,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(input_dim * patch_len, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        patches = []
        for start in range(0, T - self.patch_len + 1, self.stride):
            patches.append(x[:, start : start + self.patch_len, :].reshape(B, -1))
        p = torch.stack(patches, dim=1)
        z = torch.cat([self.cls_token.expand(B, -1, -1), self.proj(p)], dim=1)
        cls = self.norm(self.encoder(z)[:, 0])
        return self.head(cls)


class iTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = FEAT_DIM,
        d_model: int = 128,
        nhead: int = 8,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(WINDOW_SIZE, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.proj(x.transpose(1, 2))).mean(dim=1)
        return self.head(z)


class TimesBlock(nn.Module):
    def __init__(self, d_model: int = 128, top_k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xf = torch.fft.rfft(x, dim=1)
        freq_amp = xf.abs().mean(dim=-1)
        top_k = min(self.top_k, freq_amp.shape[1] - 1)
        _, top_idx = torch.topk(freq_amp[:, 1:], top_k, dim=1)
        top_idx = top_idx + 1
        agg = torch.zeros_like(x)
        for k in range(top_k):
            p = max(2, int(round(T / top_idx[:, k].float().mean().item())))
            pad_len = math.ceil(T / p) * p - T
            xp = F.pad(x, (0, 0, 0, pad_len))
            xp = xp.reshape(B, -1, p, D).permute(0, 3, 1, 2)
            xp = self.conv(xp).permute(0, 2, 3, 1).reshape(B, -1, D)
            agg = agg + xp[:, :T, :]
        return self.norm(x + self.dropout(agg / top_k))


class TimesNetClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = FEAT_DIM,
        d_model: int = 128,
        depth: int = 3,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [TimesBlock(d_model, top_k, dropout) for _ in range(depth)]
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        for b in self.blocks:
            z = b(z)
        return self.head(z.mean(dim=1))


class S4Layer(nn.Module):
    """Simplified diagonal S4 layer (stable, log-space A)."""

    def __init__(self, d_model: int = 128, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.log_A_diag = nn.Parameter(torch.randn(d_state) * 0.5 - 1.0)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model) * 0.1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        A_disc = torch.exp(-torch.exp(self.log_A_diag))
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            h = h * A_disc.unsqueeze(0) + torch.matmul(x_t, self.B.T)
            outputs.append(torch.matmul(h, self.C.T) + x_t * self.D)
        output = torch.stack(outputs, dim=1)
        return self.norm(x + self.dropout(output))


class MambaClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = FEAT_DIM,
        d_model: int = 128,
        d_state: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.layers = nn.ModuleList(
            [S4Layer(d_model, d_state, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        for layer in self.layers:
            z = layer(z)
        return self.head(self.norm(z.mean(dim=1)))


class TimesBERTClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = FEAT_DIM,
        d_model: int = 128,
        nhead: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.proj = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        z = torch.cat([self.cls.expand(B, -1, -1), self.proj(x)], dim=1)
        return self.head(self.encoder(z)[:, 0])


# ─────────────────────────────────────────────────────────────────────────────
# §C  ML baseline builders
# ─────────────────────────────────────────────────────────────────────────────


def build_rf():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )


def build_xgb():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def build_lgbm():
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §D  Model registries
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelSpec:
    name: str
    kind: str  # 'dl' or 'ml'
    build: Callable


# ── ChronosXOfficial b_in channel ablation variants (used by ablation notebook) ─
#
# Group A — OIB cond = mean(ctx) only (standard path):
#   Bchan_Spatial           b_in = Spatial(2)            ← spatial-only baseline
#   Bchan_L1                b_in = L1(8)                 ← behavioral-only baseline
#   Bchan_Spatial_L1        b_in = Spatial + L1 (10)     ← V2 BCBest reference
#   Bchan_Spatial_L1_L2bc   b_in = Spatial + L1 + L2 (18) ← V2 BC-Full reference
#
# Group B — OIB cond = fused(ctx + L2 projection) instead of mean(ctx):
#   Bchan_Spatial_L2oib     b_in = Spatial (2)           ← V2 OIBBest reference
#   Bchan_Spatial_L1_L2oib  b_in = Spatial + L1 (10)     ← V2 Hybrid reference
#   Bchan_Full_L2oib        b_in = Spatial + L1 + L2 (18) ← new variant
#
# Group C — fine-grained channel tuning (V3.1 post-ablation):
#   Bchan_Spatial_DirAcc    b_in = Spatial + dir_change + accel (4) ← orthogonal L1 only
#   Bchan_Spatial_FrameDiff b_in = Spatial + Δx + Δy (4)           ← full-res velocity
#
CHRONOSX_VARIANTS: dict = {
    # ── Group A: b_in channel ablation, OIB cond = mean(ctx) only ────────────
    #    Purpose: find optimal behavioral channel set for b_in stream
    "Bchan_Spatial": ModelSpec("Bchan_Spatial", "dl", lambda: Bchan_Spatial()),
    "Bchan_L1": ModelSpec("Bchan_L1", "dl", lambda: Bchan_L1()),
    "Bchan_Spatial_L1": ModelSpec("Bchan_Spatial_L1", "dl", lambda: Bchan_Spatial_L1()),
    "Bchan_Spatial_L1_L2bc": ModelSpec(
        "Bchan_Spatial_L1_L2bc", "dl", lambda: Bchan_Spatial_L1_L2bc()
    ),
    # ── Group B: vary b_in, OIB cond = fused(ctx + L2 projection) ────────────
    #    Purpose: test whether routing L2 through OIB gate instead of b_in helps
    "Bchan_Spatial_L2oib": ModelSpec(
        "Bchan_Spatial_L2oib", "dl", lambda: Bchan_Spatial_L2oib()
    ),
    "Bchan_Spatial_L1_L2oib": ModelSpec(
        "Bchan_Spatial_L1_L2oib", "dl", lambda: Bchan_Spatial_L1_L2oib()
    ),
    "Bchan_Full_L2oib": ModelSpec("Bchan_Full_L2oib", "dl", lambda: Bchan_Full_L2oib()),
    # ── Group C: fine-grained channel tuning (V3.1 post-ablation) ────────────
    #    Purpose: test whether orthogonal L1 sub-channels or full-res velocity
    #             can push beyond Bchan_Spatial without redundancy noise
    "Bchan_Spatial_DirAcc": ModelSpec(
        "Bchan_Spatial_DirAcc", "dl", lambda: Bchan_Spatial_DirAcc()
    ),
    "Bchan_Spatial_FrameDiff": ModelSpec(
        "Bchan_Spatial_FrameDiff", "dl", lambda: Bchan_Spatial_FrameDiff()
    ),
    # ── Group D: architecture depth variants (IIB/ctx/OIB deepening) ─────────
    #    Purpose: test whether deeper cross-attention (IIB×2, ctx self-attn,
    #             OIB cross-attn) can further improve upon Bchan_Spatial baseline
    "Bchan_Spatial_IIB2": ModelSpec(
        "Bchan_Spatial_IIB2", "dl", lambda: Bchan_Spatial_IIB2()
    ),
    "Bchan_Spatial_CtxSA": ModelSpec(
        "Bchan_Spatial_CtxSA", "dl", lambda: Bchan_Spatial_CtxSA()
    ),
    "Bchan_Spatial_OIBxattn": ModelSpec(
        "Bchan_Spatial_OIBxattn", "dl", lambda: Bchan_Spatial_OIBxattn()
    ),
    # ── Group E: combination (CtxSA + OIBxattn) ──────────────────────────────
    "Bchan_Spatial_CtxSA_OIBxattn": ModelSpec(
        "Bchan_Spatial_CtxSA_OIBxattn", "dl", lambda: Bchan_Spatial_CtxSA_OIBxattn()
    ),
    # ── Group F: context-stream ablation (Text / Code / No-Ctx) ──────────────
    #    Purpose: decompose the 768d ctx stream into its two 384d halves
    #    (embed_text vs embed_code) and measure each sub-stream's contribution.
    #    Naming convention: "NoX" = stream X is zeroed, all else retained.
    #    CtxSA_NoCtx is the hard lower-bound: same architecture, zero ctx.
    "CtxSA_NoCode": ModelSpec("CtxSA_NoCode", "dl", lambda: CtxSA_NoCode()),
    "CtxSA_NoText": ModelSpec("CtxSA_NoText", "dl", lambda: CtxSA_NoText()),
    "CtxSA_NoCtx": ModelSpec("CtxSA_NoCtx", "dl", lambda: CtxSA_NoCtx()),
    # ── Group G: module-contribution ablation ─────────────────────────────────
    #    Purpose: measure each module's individual contribution by removing it
    #    while keeping all other components identical to Bchan_Spatial_CtxSA.
    #    Forms a complete decomposition: behav stream / IIB / OIB / ctx_proj.
    "CtxSA_NoBehav": ModelSpec("CtxSA_NoBehav", "dl", lambda: CtxSA_NoBehav()),
    "CtxSA_NoIIB": ModelSpec("CtxSA_NoIIB", "dl", lambda: CtxSA_NoIIB()),
    "CtxSA_NoOIB": ModelSpec("CtxSA_NoOIB", "dl", lambda: CtxSA_NoOIB()),
    "CtxSA_1StageProj": ModelSpec("CtxSA_1StageProj", "dl", lambda: CtxSA_1StageProj()),
}


# ─────────────────────────────────────────────────────────────────────────────
# §E  Wide-context variant — Bchan_Spatial_CtxSA_Wide
#
# Motivation: EyeSeqDatasetV2 uses random projection to squeeze 1152-2688d
# multimodal vectors into the fixed 768d ctx slot.  This introduces
# irreversible information loss (~33 % when 1152→768).
#
# This variant avoids projection entirely by using EyeSeqDatasetV2_Wide:
#   • Keeps old embed_text(384) + embed_code(384) = 768d at [2:770]
#   • Appends 4 new 384d channels (G3a, G3b, G4, G5) at [770:2306]
#   • ctx_proj receives 2304d input (no information loss)
#
# All other architecture details are identical to Bchan_Spatial_CtxSA:
#   behav_proj, backbone(3L Transformer), IIB, CtxSA layer, OIB, head.
# Only _CTX_DIM and the feature slice change.
#
# Feature slice used: x[:,:,2:2306]  (WIDE_CTX_DIM = 2304)
# ─────────────────────────────────────────────────────────────────────────────


# Import wide-layout constants from dataset module at runtime to avoid
# circular imports at module-load time.
def _get_wide_ctx_slice():
    """Return the context slice for the Wide layout (2304d, [2:2306])."""
    try:
        from shared.dataset import WIDE_CTX_DIM

        return slice(2, 2 + WIDE_CTX_DIM)
    except ImportError:
        return slice(2, 2306)  # fallback: 2304d = 6×384


class Bchan_Spatial_CtxSA_Wide(nn.Module):
    """
    Wide-context version of Bchan_Spatial_CtxSA.

    Uses EyeSeqDatasetV2_Wide (2322d feature tensor).
    ctx_proj receives the full 2304d context block [2:2306] — no random
    projection, no information loss compared to EyeSeqDatasetV2.

    Architecture is identical to Bchan_Spatial_CtxSA except:
      _CTX_DIM = 2304  (vs 768 in the original)
      reads x[:,:,2:2306]  (vs x[:,:,2:770])

    Context composition:
      [2:386]    embed_text(384)             old JSON  (MiniLM, text desc)
      [386:770]  embed_code(384)             old JSON  (sentence-transformer)
      [770:1154] embed_text_src(384)         new JSON  G3a (visible source text)
      [1154:1538] embed_text_ocr(384)        new JSON  G3b (OCR)
      [1538:1922] embed_img_origin_patch(384) new JSON G4 (ViT patch)
      [1922:2306] embed_img_origin_page(384)  new JSON G5 (ViT page)
    """

    _CTX_DIM = 2304  # 6 × 384 (all fields at full dimension, no projection)

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        n_backbone_layers: int = 3,
        iib_hidden: int = 64,
        oib_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = 2
        self._ctx_slice = _get_wide_ctx_slice()

        self.behav_proj = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model))
        # ctx_proj: 2304 → d_model×2 → d_model  (same two-stage design)
        self.ctx_proj = nn.Sequential(
            nn.Linear(self._CTX_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        # 1-layer self-attention to enrich ctx tokens (identical to original)
        self.ctx_sa = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.iib = IIB(d_model=d_model, hidden_dim=iib_hidden, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.backbone = _make_backbone(d_model, nhead, n_backbone_layers, dropout)
        self.backbone_norm = nn.LayerNorm(d_model)
        self.oib = OIB(d_model=d_model, hidden_dim=oib_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        _init_weights_v3(self, self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        e = self.behav_proj(x[:, :, :2])  # [B, T, d]  spatial only
        v = self.ctx_proj(x[:, :, self._ctx_slice])  # [B, T, d]  2304d → d
        v = self.ctx_sa(v)  # ctx self-attn
        e_p = self.iib(e, v)  # IIB cross-attn
        cls = self.cls_token.expand(B, -1, -1)
        h = self.backbone_norm(self.backbone(torch.cat([cls, e_p], dim=1))[:, 0])
        return self.head(self.oib(h, v.mean(dim=1)))


# ── All baselines (used by baselines notebook) ────────────────────────────────
BASELINE_MODELS: dict = {
    "XGBoost": ModelSpec("XGBoost", "ml", build_xgb),
    "RandomForest": ModelSpec("RandomForest", "ml", build_rf),
    "LightGBM": ModelSpec("LightGBM", "ml", build_lgbm),
    "BiLSTM": ModelSpec("BiLSTM", "dl", lambda: BiLSTMClassifier()),
    "1D-CNN": ModelSpec("1D-CNN", "dl", lambda: CNN1DClassifier()),
    "TransformerEnc": ModelSpec(
        "TransformerEnc", "dl", lambda: TransformerEncoderClassifier()
    ),
    "PatchTST": ModelSpec("PatchTST", "dl", lambda: PatchTSTClassifier()),
    "iTransformer": ModelSpec("iTransformer", "dl", lambda: iTransformerClassifier()),
    "TimesNet": ModelSpec("TimesNet", "dl", lambda: TimesNetClassifier()),
    "DLinear": ModelSpec("DLinear", "dl", lambda: DLinearClassifier()),
    "Mamba": ModelSpec("Mamba", "dl", lambda: MambaClassifier(d_state=64)),
    "TimesBERT": ModelSpec("TimesBERT", "dl", lambda: TimesBERTClassifier()),
}
