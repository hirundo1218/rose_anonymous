"""
ROSE: multi-scale CNN encoder without temporal pooling.

Each expert branch keeps the full conv feature map (B, hidden, T). Forecast is
produced by a 1D conv head on that map instead of pooling + MLP. The gate still
uses a lightweight per-branch summary for routing alpha.

Interface: forward(x) -> (pred, alphas) with
  pred   : (B, forecast_len, C)
  alphas : (B, C, K)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import model as base
from . import utils as ut

DEVICE = ut.DEVICE
WINDOW = ut.WINDOW
TARGET_LEN = WINDOW
K = ut.K
BATCH = ut.BATCH
EPOCHS = ut.EPOCHS
LR = ut.LR
PAPER_ENCODER_KERNEL_SIZES = base.PAPER_ENCODER_KERNEL_SIZES
LAMBDA_LB_PAPER = base.LAMBDA_LB_PAPER
LAMBDA_DIST_PAPER = base.LAMBDA_DIST_PAPER
LAMBDA_S = base.LAMBDA_S
LAMBDA_ES = base.LAMBDA_ES

DEFAULT_HEAD_KERNEL = 3
DEFAULT_HEAD_LAYERS = 2
DEFAULT_GATE_POOL = 'mean_std'

# Gate summary candidates for routing (experts always keep full h).
GATE_POOL_CHOICES = (
    'mean_max',      # mean + max
    'mean_std',      # current default: mean + std
    'mean',          # mean only
    'last',          # last timestep
    'max',           # max only
    'std',           # std only
    'max_std',       # max + std
    'mean_max_std',  # mean + max + std
)


def gate_feat_multiplier(gate_pool: str) -> int:
    gp = str(gate_pool)
    if gp in ('mean_max', 'mean_std', 'max_std'):
        return 2
    if gp == 'mean_max_std':
        return 3
    if gp in ('mean', 'last', 'max', 'std'):
        return 1
    raise ValueError(f'Unknown gate_pool: {gate_pool!r}; choose from {GATE_POOL_CHOICES}')


def gate_features(h: torch.Tensor, gate_pool: str = DEFAULT_GATE_POOL) -> torch.Tensor:
    """(B, hidden, T) -> (B, gate_feat) for routing only (experts keep full h)."""
    gp = str(gate_pool)
    if gp == 'mean_max':
        return torch.cat([h.mean(dim=-1), h.max(dim=-1).values], dim=-1)
    if gp == 'mean_std':
        return torch.cat([h.mean(dim=-1), h.std(dim=-1, unbiased=False)], dim=-1)
    if gp == 'max_std':
        return torch.cat([h.max(dim=-1).values, h.std(dim=-1, unbiased=False)], dim=-1)
    if gp == 'mean_max_std':
        return torch.cat(
            [h.mean(dim=-1), h.max(dim=-1).values, h.std(dim=-1, unbiased=False)],
            dim=-1,
        )
    if gp == 'mean':
        return h.mean(dim=-1)
    if gp == 'last':
        return h[..., -1]
    if gp == 'max':
        return h.max(dim=-1).values
    if gp == 'std':
        return h.std(dim=-1, unbiased=False)
    raise ValueError(f'Unknown gate_pool: {gate_pool!r}; choose from {GATE_POOL_CHOICES}')


# Back-compat alias used by experiments that import _gate_features
def _gate_features(h: torch.Tensor, gate_pool: Optional[str] = None) -> torch.Tensor:
    return gate_features(h, gate_pool if gate_pool is not None else DEFAULT_GATE_POOL)


class EncoderBranchSeq(nn.Module):
    """Multi-scale encoder branch: conv only, keeps temporal length (no pooling)."""

    def __init__(self, in_ch: int, hidden: int, kernel_size: int):
        super().__init__()
        p = kernel_size // 2
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=kernel_size, padding=p),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=p),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, hidden, T)


class MultiScaleEncoderSeq(nn.Module):
    """One branch per kernel size; each outputs (B, hidden, T)."""

    def __init__(
        self,
        in_ch: int,
        hidden: int = 64,
        kernel_sizes: Optional[list] = None,
        gate_pool: str = DEFAULT_GATE_POOL,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes if kernel_sizes is not None else ut.EXPERT_KERNEL_SIZES
        self.hidden = int(hidden)
        self.kernel_sizes = list(kernel_sizes)
        self.gate_pool = str(gate_pool)
        self.branches = nn.ModuleList([
            EncoderBranchSeq(in_ch, hidden, k) for k in self.kernel_sizes
        ])

    @property
    def gate_feat_dim(self) -> int:
        return self.hidden * gate_feat_multiplier(self.gate_pool)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [br(x) for br in self.branches]


class ConvHeadExpert(nn.Module):
    """
    1D conv forecast head on encoder feature map (B, hidden, T) -> (B, forecast_len).

    Same-length setup (T == forecast_len): two (or more) conv layers preserve T,
    then 1x1 conv maps channels to a single forecast channel.
    """

    def __init__(
        self,
        hidden: int,
        forecast_len: int,
        head_kernel: int = DEFAULT_HEAD_KERNEL,
        n_layers: int = DEFAULT_HEAD_LAYERS,
    ):
        super().__init__()
        self.forecast_len = int(forecast_len)
        p = head_kernel // 2
        layers: List[nn.Module] = []
        for _ in range(max(1, int(n_layers))):
            layers.extend([
                nn.Conv1d(hidden, hidden, kernel_size=head_kernel, padding=p),
                nn.ReLU(),
            ])
        layers.append(nn.Conv1d(hidden, 1, kernel_size=1))
        self.head = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.head(h).squeeze(1)  # (B, T)
        if y.shape[1] != self.forecast_len:
            y = F.interpolate(
                y.unsqueeze(1),
                size=self.forecast_len,
                mode='linear',
                align_corners=False,
            ).squeeze(1)
        return y


class GroupMoESeqConvHead(nn.Module):
    """Per-channel MoE block: seq encoder + conv-head experts + gate on summaries."""

    def __init__(
        self,
        in_ch: int,
        K_experts: int,
        hidden: int = 64,
        forecast_len: int = TARGET_LEN,
        kernel_sizes: Optional[list] = None,
        head_kernel: int = DEFAULT_HEAD_KERNEL,
        head_layers: int = DEFAULT_HEAD_LAYERS,
        gate_pool: str = DEFAULT_GATE_POOL,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes if kernel_sizes is not None else PAPER_ENCODER_KERNEL_SIZES[:K_experts]
        if len(kernel_sizes) != K_experts:
            kernel_sizes = (kernel_sizes * (K_experts // len(kernel_sizes) + 1))[:K_experts]
        self.in_ch = int(in_ch)
        self.forecast_len = int(forecast_len)
        self.K = int(K_experts)
        self.kernel_sizes = list(kernel_sizes)
        self.gate_pool = str(gate_pool)
        self.encoder = MultiScaleEncoderSeq(
            in_ch, hidden, self.kernel_sizes, gate_pool=self.gate_pool,
        )
        gate_in = self.encoder.gate_feat_dim * self.K
        self.gate = nn.Linear(gate_in, self.K)
        self.experts = nn.ModuleList([
            ConvHeadExpert(hidden, self.forecast_len * self.in_ch, head_kernel, head_layers)
            for _ in range(self.K)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hs = self.encoder(x)  # list of (B, hidden, T)
        gate_parts = [gate_features(h, self.gate_pool) for h in hs]
        h_gate = torch.cat(gate_parts, dim=-1)
        alpha = F.softmax(self.gate(h_gate), dim=-1)
        outs = torch.stack([ex(hs[k]) for k, ex in enumerate(self.experts)], dim=-1)
        y = torch.sum(outs * alpha.unsqueeze(1), dim=-1)
        if self.in_ch > 1:
            y = y.view(y.shape[0], self.in_ch, self.forecast_len).sum(dim=1)
        return y, alpha


class ROSE(nn.Module):
    """Channel-wise MoE with conv-head experts (no encoder pooling)."""

    def __init__(
        self,
        num_channels: int,
        K_experts: int = K,
        hidden: int = 64,
        forecast_len: int = TARGET_LEN,
        head_kernel: int = DEFAULT_HEAD_KERNEL,
        head_layers: int = DEFAULT_HEAD_LAYERS,
        gate_pool: str = DEFAULT_GATE_POOL,
    ):
        super().__init__()
        self.C = int(num_channels)
        self.K = int(K_experts)
        self.forecast_len = int(forecast_len)
        self.head_kernel = int(head_kernel)
        self.head_layers = int(head_layers)
        self.gate_pool = str(gate_pool)
        self.channel_moes = nn.ModuleList([
            GroupMoESeqConvHead(
                in_ch=1,
                K_experts=self.K,
                hidden=hidden,
                forecast_len=self.forecast_len,
                head_kernel=self.head_kernel,
                head_layers=self.head_layers,
                gate_pool=self.gate_pool,
            )
            for _ in range(self.C)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T_in, C = x.shape
        if C != self.C:
            raise ValueError(f'Expected C={self.C}, got {C}')
        pred = x.new_zeros((B, self.forecast_len, C))
        alpha = x.new_zeros((B, C, self.K))
        for c in range(self.C):
            xc = x[:, :T_in, c].unsqueeze(1)  # (B, 1, T)
            yc, ac = self.channel_moes[c](xc)
            pred[:, :, c] = yc
            alpha[:, c, :] = ac
        return pred, alpha


def train_rose(
    model: ROSE,
    Xw_train: np.ndarray,
    Yt_train: np.ndarray,
) -> None:
    """Same loss as ROSE (Huber + load-balance + BW); no grad-loss term."""
    base.train_rose(model, Xw_train, Yt_train, lambda_grad=0.0)


def build_checkpoint_dict(
    model: ROSE,
    data_dir: str,
    alpha_train_mean_ck: Optional[np.ndarray] = None,
) -> dict:
    ckpt = {
        'state_dict': model.state_dict(),
        'model_type': 'rose',
        'model_name': 'ROSE',
        'C': int(model.C),
        'K': int(model.K),
        'data_dir': str(data_dir),
        'encoder': 'multiscale_cnn_seq',
        'encoder_mode': 'convhead',
        'head_kernel': int(model.head_kernel),
        'head_layers': int(model.head_layers),
        'gate_pool': str(model.gate_pool),
        'encoder_kernel_sizes': list(PAPER_ENCODER_KERNEL_SIZES[:model.K]),
        'lambda_s': float(LAMBDA_S),
        'lambda_es': float(LAMBDA_ES),
        'score_aggregation': base.DEFAULT_SCORE_AGGREGATION,
        'point_score_align': base.DEFAULT_POINT_SCORE_ALIGN,
        'window': int(WINDOW),
        'target_len': int(model.forecast_len),
    }
    if alpha_train_mean_ck is not None:
        ckpt['alpha_train_mean'] = np.asarray(alpha_train_mean_ck, dtype=np.float64)
    return ckpt


def is_convhead_checkpoint(ckpt: dict) -> bool:
    mtype = str(ckpt.get('model_type', '')).lower()
    if mtype in ('rose', 'rose_convhead'):
        return True
    return str(ckpt.get('encoder_mode', '')).lower() == 'convhead'


def load_from_checkpoint(ckpt: dict) -> ROSE:
    C = int(ckpt['C'])
    K_ex = int(ckpt['K'])
    forecast_len = int(ckpt.get('target_len', TARGET_LEN))
    gate_pool = str(ckpt.get('gate_pool', DEFAULT_GATE_POOL))
    model = ROSE(
        num_channels=C,
        K_experts=K_ex,
        forecast_len=forecast_len,
        head_kernel=int(ckpt.get('head_kernel', DEFAULT_HEAD_KERNEL)),
        head_layers=int(ckpt.get('head_layers', DEFAULT_HEAD_LAYERS)),
        gate_pool=gate_pool,
    ).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


# Module-level alias kept for older imports / train_eval meta dumps
GATE_POOL = DEFAULT_GATE_POOL
