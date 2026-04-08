"""
VisNet Unified Framework - SEPARATE FREQUENCY CHANNELS + WAVELET

This variant adds wavelet processing to improve learning, binding, and multiscale representation:

  - Transformation: 2D Haar wavelet decomposition of opponent channels gives multiscale
    (LL, LH, HL, HH) subbands; each frequency stream is augmented with the corresponding
    wavelet band (Gabor + wavelet per stream) for richer, scale-aligned features.

  - Binding: Holographic (HRR) binding uses **unitary FFT** circular convolution, blended with
    circular **correlation** (dual unbinding direction) for a richer commutator-free code.
    Wavelet-domain binding still binds patch with y in the Haar basis (low/high bands).

  - Neural type: Optional wavelet denoising (soft-threshold in 1D Haar) of the patch
    before Hebbian learning promotes sparse, stable representations and reduces noise
    in the weight update.

Core pipeline:
  RGB -> opponent (3 ch, no DoG) -> Gabor on luminance + Wavelet2D
   -> Per stream: L1_i → L2_i (V4-like) → ventral L3_i → L4_i (TE) in parallel with MT_i → PP_i (dorsal).
   Dorsal MT inputs are gated by a **static** saliency map (centre-surround DoG + texture residual + local z-score figure-ground), not motion.
   Readout concat(TE, PP) when dorsal is on.
   -> Classifier: 2 × num_freqs × spatial_size² (dorsal) or 1 × (ventral-only).

Learning: Unified plasticity + holographic + hyperbolic + wavelet_binding + optional wavelet_denoise.
Dataset: CIFAR-10.

**Neuron–glia coupling** is on by default (``--no-neuron-glia`` to disable): a slow extra-synaptic
state (EMA of layer activity summaries) modulates local Hebbian variance/inhibition per L1–L4—
homeostatic regulation without backprop through the hierarchy.
"""

from __future__ import annotations

import math
import os
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def _gaussian_kernel_2d(ksz: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized 2D Gaussian for depthwise blur: shape [1, 1, ksz, ksz]."""
    ax = torch.arange(-(ksz // 2), ksz // 2 + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) ** 2))
    g = g / (g.sum() + 1e-8)
    return g.view(1, 1, ksz, ksz)


# =============================================================================
# HYPERBOLIC OPERATIONS (Poincaré ball) - reused from hyperbolic variants
# =============================================================================


class HyperbolicOps:
    """
    Utility class implementing common operations on the Poincaré ball model.
    Adapted from the Hyperbolic3_* variants so we can reuse the same binding /
    distance-gradient terms inside the local plasticity updates.
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-5, max_norm: float = 0.999):
        self.c = float(c)
        self.eps = float(eps)
        self.max_norm = float(max_norm)

    def norm(self, x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        return torch.norm(x, dim=dim, keepdim=keepdim)

    # -------------------------------------------------------------------------
    # Exponential / logarithmic maps at the origin
    # -------------------------------------------------------------------------
    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        v_norm = torch.clamp(self.norm(v, dim=-1, keepdim=True), min=self.eps)
        sqrt_c = math.sqrt(self.c)
        second_term = torch.tanh(sqrt_c * v_norm) * (v / (sqrt_c * v_norm))
        return torch.clamp(second_term, min=-self.max_norm, max=self.max_norm)

    def log_map_zero(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.clamp(self.norm(x, dim=-1, keepdim=True), min=self.eps, max=self.max_norm)
        sqrt_c = math.sqrt(self.c)
        return (1.0 / sqrt_c) * torch.atanh(sqrt_c * x_norm) * (x / x_norm)

    def to_poincare(self, v: torch.Tensor) -> torch.Tensor:
        return self.exp_map_zero(v)

    def from_poincare(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_map_zero(x)

    # -------------------------------------------------------------------------
    # Möbius addition / scalar multiplication
    # -------------------------------------------------------------------------
    def project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.norm(x, dim=-1, keepdim=True)
        scale = torch.clamp(self.max_norm / (norm + self.eps), max=1.0)
        return x * scale

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + (c ** 2) * x2 * y2
        return torch.clamp(numerator / torch.clamp(denominator, min=self.eps), min=-self.max_norm, max=self.max_norm)

    # -------------------------------------------------------------------------
    # Hyperbolic distance and its gradient w.r.t the weights
    # -------------------------------------------------------------------------
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm_sq = torch.clamp((x * x).sum(dim=-1), min=0, max=1.0 / self.c - self.eps)
        y_norm_sq = torch.clamp((y * y).sum(dim=-1), min=0, max=1.0 / self.c - self.eps)
        diff = x - y
        dist_sq = (diff * diff).sum(dim=-1)
        denom = (1 - self.c * x_norm_sq) * (1 - self.c * y_norm_sq)
        denom = torch.clamp(denom, min=self.eps)
        arg = 1 + 2 * self.c * dist_sq / denom
        arg = torch.clamp(arg, min=1.0 + self.eps, max=1e10)
        return (1.0 / math.sqrt(self.c)) * torch.acosh(arg)

    def distance_gradient_wrt_w(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Gradient of d_h(x, w) wrt w. x: [B, N, D], w: [N, D].
        Returns a tensor shaped like x (same B, N, D); caller typically averages over B.
        """
        B, N, D = x.shape
        if chunk_size is not None and N > chunk_size:
            grad = torch.zeros_like(x)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                grad[:, start:end, :] = self._distance_gradient_chunk(x[:, start:end, :], w[start:end, :])
            return grad
        return self._distance_gradient_chunk(x, w)

    def _distance_gradient_chunk(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        w_expanded = w.unsqueeze(0).expand(B, -1, -1)
        diff = w_expanded - x
        x_norm_sq = torch.clamp((x * x).sum(dim=-1, keepdim=True), min=0, max=1.0 / self.c - self.eps)
        w_norm_sq = torch.clamp((w_expanded * w_expanded).sum(dim=-1, keepdim=True), min=0, max=1.0 / self.c - self.eps)
        dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
        denom_x = torch.clamp(1 - self.c * x_norm_sq, min=self.eps)
        denom_w = torch.clamp(1 - self.c * w_norm_sq, min=self.eps)
        denom = denom_x * denom_w
        arg = 1 + 2 * self.c * dist_sq / denom
        arg = torch.clamp(arg, min=1.0 + self.eps, max=1e10)
        d_arcosh = 1.0 / torch.sqrt(torch.clamp(arg * arg - 1.0, min=self.eps))
        sqrt_c = math.sqrt(self.c)
        d_dist_sq = 2 * diff
        d_denom = -2 * self.c * w_expanded * denom_x
        d_arg = 2 * self.c * (d_dist_sq / denom - dist_sq * d_denom / (denom * denom + self.eps))
        grad = (1.0 / sqrt_c) * d_arcosh * d_arg
        return torch.clamp(grad, min=-10.0, max=10.0)


# Shared ops instance
hyp_ops = HyperbolicOps(c=1.0)


# =============================================================================
# PREPROCESSING: RGB opponent (no DoG) + Gabor (12 orientations on luminance)
# =============================================================================

def _gaussian2d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    ax = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return g / (g.sum() + 1e-8)


def _mexican_hat_kernel2d(
    size: int,
    sigma_center: float,
    sigma_surround: float,
    surround_gain: float,
    device: torch.device,
) -> torch.Tensor:
    """Difference-of-Gaussians (center − gain×surround), zero-mean, L1-normalized for conv inhibition."""
    c = _gaussian2d(size, sigma_center, device)
    s = _gaussian2d(size, sigma_surround, device)
    k = c - float(surround_gain) * s
    k = k - k.mean()
    return k / (k.abs().sum() + 1e-8)


class EntropyDropout(nn.Module):
    """
    Dropout with probability modulated by activation entropy: higher entropy (more
    uniform activations) -> higher dropout p. p = base_p * (1 + entropy_scale * norm_entropy).
    """

    def __init__(self, base_p: float = 0.1, entropy_scale: float = 0.5):
        super().__init__()
        self._p = float(base_p)
        self.entropy_scale = float(entropy_scale)

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, value: float) -> None:
        self._p = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self._p <= 0:
            return x
        flat = x.reshape(x.size(0), -1)
        probs = F.softmax(flat, dim=1)
        eps = 1e-8
        entropy = -(probs * (probs + eps).log()).sum(dim=1)
        max_entropy = math.log(flat.size(1) + eps)
        norm_entropy = (entropy / (max_entropy + eps)).clamp(0, 1)
        mean_ne = norm_entropy.mean().item()
        p = min(1.0, self._p * (1.0 + self.entropy_scale * mean_ne))
        return F.dropout(x, p=p, training=True)


class ModernHopfieldPFC(nn.Module):
    """
    Lightweight Modern-Hopfield style associative retrieval block for PFC-like
    representation stabilization after IT/L4 features.
    """

    def __init__(
        self,
        feature_dim: int,
        num_patterns: int = 64,
        beta: float = 1.0,
        temperature: float = 2.0,
        blend: float = 0.005,
        ema_lr: float = 0.0,
        unsup_update: bool = False,
        use_cosine_similarity: bool = True,
        soft_ema_update: bool = True,
        normalize_memory: bool = False,
        sparsity: float = 0.9,
        sparse_update: bool = True,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_patterns = int(max(2, num_patterns))
        self.beta = float(max(1e-6, beta))
        self.temperature = float(max(1e-6, temperature))
        self.blend = float(max(0.0, min(1.0, blend)))
        self.ema_lr = float(max(0.0, min(1.0, ema_lr)))
        self.unsup_update = bool(unsup_update)
        self.use_cosine_similarity = bool(use_cosine_similarity)
        self.soft_ema_update = bool(soft_ema_update)
        self.normalize_memory = bool(normalize_memory)
        self.sparsity = float(max(0.0, min(1.0, sparsity)))
        self.sparse_update = bool(sparse_update)
        self.patterns = nn.Parameter(torch.empty(self.num_patterns, self.feature_dim))
        nn.init.xavier_uniform_(self.patterns)
        if self.normalize_memory:
            with torch.no_grad():
                self.patterns.copy_(F.normalize(self.patterns, p=2, dim=1))
        self.norm = nn.LayerNorm(self.feature_dim) if bool(use_layernorm) else nn.Identity()
        self._last_consistency_loss: Optional[torch.Tensor] = None

    def _apply_feature_sparsity(self, feat: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0.0:
            return feat
        keep_frac = max(0.0, min(1.0, 1.0 - self.sparsity))
        keep_k = int(math.ceil(keep_frac * feat.size(1)))
        if keep_k <= 0:
            return torch.zeros_like(feat)
        if keep_k >= feat.size(1):
            return feat
        kth = torch.topk(feat.abs(), k=keep_k, dim=1, largest=True, sorted=False).values.min(dim=1, keepdim=True).values
        return feat * (feat.abs() >= kth).to(feat.dtype)

    def forward(self, x: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.feature_dim:
            raise ValueError(
                f"ModernHopfieldPFC expects [B, {self.feature_dim}], got {tuple(x.shape)}"
            )
        do_unsup_ema = self.training and self.unsup_update and bool(update_memory) and self.ema_lr > 0.0
        # In unsupervised EMA mode, read detached memory to avoid in-place/grad conflicts.
        patterns_read = self.patterns.detach() if do_unsup_ema else self.patterns
        if self.use_cosine_similarity:
            x_for_sim = F.normalize(x, p=2, dim=1)
            p_for_sim = F.normalize(patterns_read, p=2, dim=1)
            logits = (x_for_sim @ p_for_sim.t()) * (self.beta / self.temperature)
        else:
            scale = math.sqrt(float(self.feature_dim))
            logits = (x @ patterns_read.t()) * (self.beta / max(1e-8, scale * self.temperature))
        attn = F.softmax(logits, dim=1)
        retrieved = attn @ patterns_read
        self._last_consistency_loss = (
            F.mse_loss(retrieved, x.detach()) if do_unsup_ema else F.mse_loss(retrieved, x)
        )

        if do_unsup_ema:
            with torch.no_grad():
                x_det = x.detach()
                x_upd = self._apply_feature_sparsity(x_det) if self.sparse_update else x_det
                if self.soft_ema_update:
                    w = attn.detach()  # [B, K]
                    denom = w.sum(dim=0)  # [K]
                    valid = denom > 1e-8
                    if valid.any():
                        weighted_sum = w.t() @ x_upd  # [K, D]
                        means = weighted_sum / denom.clamp_min(1e-8).unsqueeze(1)
                        v_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                        self.patterns[v_idx].mul_(1.0 - self.ema_lr).add_(means[v_idx], alpha=self.ema_lr)
                else:
                    assign = attn.detach().argmax(dim=1)
                    for cls_id in torch.unique(assign):
                        idx = int(cls_id.item())
                        cls_feat = x_upd[assign == cls_id]
                        if cls_feat.numel() == 0:
                            continue
                        batch_mean = cls_feat.mean(dim=0)
                        self.patterns[idx].mul_(1.0 - self.ema_lr).add_(batch_mean, alpha=self.ema_lr)
                if self.normalize_memory:
                    self.patterns.copy_(F.normalize(self.patterns, p=2, dim=1))

        out = (1.0 - self.blend) * x + self.blend * retrieved
        return self.norm(out)

    def get_last_consistency_loss(self) -> Optional[torch.Tensor]:
        return self._last_consistency_loss


class HebbianSelfAttentionPFC(nn.Module):
    """
    Transformer-style softmax attention over K memory slots (values = row-wise patterns).
    Q/K are linear projections of the input and of memory; W_q, W_k and patterns are updated
    with Hebbian outer-product rules (no backprop). Same interface as ModernHopfieldPFC for PP flat [B,D].
    """

    def __init__(
        self,
        feature_dim: int,
        num_patterns: int = 64,
        head_dim: int = 32,
        blend: float = 0.005,
        temperature: float = 1.0,
        ema_lr: float = 1e-3,
        unsup_update: bool = True,
        use_layernorm: bool = False,
        hebbian_lr: float = 1e-4,
        hebbian_decay: float = 1e-5,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_patterns = int(max(2, num_patterns))
        hd = int(max(4, min(head_dim, feature_dim)))
        self.head_dim = hd
        self.blend = float(max(0.0, min(1.0, blend)))
        self.temperature = float(max(1e-6, temperature))
        self.ema_lr = float(max(0.0, min(1.0, ema_lr)))
        self.unsup_update = bool(unsup_update)
        self.hebbian_lr = float(max(0.0, hebbian_lr))
        self.hebbian_decay = float(max(0.0, hebbian_decay))
        self.norm = nn.LayerNorm(self.feature_dim) if bool(use_layernorm) else nn.Identity()
        self._last_consistency_loss: Optional[torch.Tensor] = None

        self.W_q = nn.Parameter(torch.empty(self.feature_dim, hd))
        self.W_k = nn.Parameter(torch.empty(self.feature_dim, hd))
        self.patterns = nn.Parameter(torch.empty(self.num_patterns, self.feature_dim))
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.patterns)
        for p in (self.W_q, self.W_k, self.patterns):
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, update_memory: bool = True) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.feature_dim:
            raise ValueError(
                f"HebbianSelfAttentionPFC expects [B, {self.feature_dim}], got {tuple(x.shape)}"
            )
        B = x.size(0)
        Wq = self.W_q
        Wk = self.W_k
        P = self.patterns
        scale = 1.0 / math.sqrt(float(self.head_dim))
        q = x @ Wq
        k_mem = P @ Wk
        logits = (q @ k_mem.t()) * (scale / self.temperature)
        attn = F.softmax(logits, dim=1)
        retrieved = attn @ P
        self._last_consistency_loss = F.mse_loss(retrieved, x.detach())

        do_learn = self.training and self.unsup_update and bool(update_memory) and self.ema_lr > 0.0
        do_hebb = self.training and self.unsup_update and bool(update_memory) and self.hebbian_lr > 0.0
        if do_hebb or do_learn:
            with torch.no_grad():
                q_det = (x @ Wq).detach()
                k_det = (P @ Wk).detach()
                if do_hebb:
                    dq = (x.t() @ q_det) / float(B)
                    dk = (P.t() @ k_det) / float(max(1, self.num_patterns))
                    decay = self.hebbian_decay
                    lr_h = self.hebbian_lr
                    Wq.mul_(1.0 - decay).add_(dq, alpha=lr_h)
                    Wk.mul_(1.0 - decay).add_(dk, alpha=lr_h)
                if do_learn:
                    P_r = P.detach()
                    attn_d = attn.detach()
                    denom = attn_d.sum(dim=0)
                    valid = denom > 1e-8
                    if valid.any():
                        weighted_sum = attn_d.t() @ x.detach()
                        means = weighted_sum / denom.clamp_min(1e-8).unsqueeze(1)
                        v_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                        P_r = P_r.clone()
                        P_r[v_idx].mul_(1.0 - self.ema_lr).add_(means[v_idx], alpha=self.ema_lr)
                        P.copy_(P_r)

        out = (1.0 - self.blend) * x + self.blend * retrieved
        return self.norm(out)

    def get_last_consistency_loss(self) -> Optional[torch.Tensor]:
        return self._last_consistency_loss


def _normalized_entropy_activations(y: torch.Tensor, dim: int = -1) -> float:
    """Compute mean normalized entropy (in [0,1]) over the last dimension."""
    if y.numel() == 0:
        return 0.0
    flat = y.reshape(-1, y.size(dim)) if y.dim() > 1 else y.unsqueeze(0)
    probs = F.softmax(flat, dim=dim)
    eps = 1e-8
    entropy = -(probs * (probs + eps).log()).sum(dim=dim).mean()
    max_h = math.log(flat.size(dim) + eps)
    return (entropy / (max_h + eps)).clamp(0.0, 1.0).item()


class DoGRGB(nn.Module):
    """RGB -> opponent (L, RG, BY) -> DoG per channel."""

    def __init__(
        self,
        device: torch.device,
        ksz: int = 3,
        sigma_c: float = 1.0,
        sigma_s: float = 1.6,
        k: float = 0.6,
    ):
        super().__init__()
        assert ksz % 2 == 1, "DoG kernel size must be odd"
        self.pad = ksz // 2

        g_c = _gaussian2d(ksz, sigma_c, device)
        g_s = _gaussian2d(ksz, sigma_s, device)
        dog = g_c - k * g_s
        dog = dog / (dog.abs().sum() + 1e-8)
        w = dog.view(1, 1, ksz, ksz).repeat(3, 1, 1, 1)  # group conv over 3 opponent chans
        self.register_buffer("w", w)

    @staticmethod
    def rgb_to_opponent(x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        lum = (r + g + b) / 3.0
        rg = r - g
        by = b - (r + g) / 2.0
        return torch.cat([lum, rg, by], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 3:
            raise ValueError(f"DoGRGB expects RGB input with 3 channels, got {x.size(1)}")
        opp = self.rgb_to_opponent(x)
        return F.conv2d(opp, self.w, padding=self.pad, groups=3)


def make_gabor_kernels(
    freqs: Tuple[float, ...],
    oris: Tuple[int, ...],
    phs: Tuple[float, ...],
    size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (real, imag) kernels with shape [n_filters, 1, kH, kW]."""
    ax = torch.linspace(-1, 1, size, device=device)
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    Rs, Is = [], []
    for f in freqs:
        sigma = 0.56 / f
        G = torch.exp(-(X * X + Y * Y) / (2 * sigma * sigma))
        for theta_deg in oris:
            t = theta_deg * math.pi / 180.0
            Xp = X * math.cos(t) + Y * math.sin(t)
            for ph in phs:
                R = G * torch.cos(2 * math.pi * f * Xp + ph)
                I = G * torch.sin(2 * math.pi * f * Xp + ph)
                Rs.append(R / (R.abs().sum() + 1e-8))
                Is.append(I / (I.abs().sum() + 1e-8))
    R = torch.stack(Rs).unsqueeze(1)
    I = torch.stack(Is).unsqueeze(1)
    return R, I


class GaborBank(nn.Module):
    """Grouped Gabor bank: in_ch groups, 32 filters per group -> out_ch = in_ch * 32."""

    def __init__(
        self,
        in_ch: int,
        freqs: Tuple[float, ...],
        oris: Tuple[int, ...],
        phs: Tuple[float, ...],
        ksz: int,
        device: torch.device,
    ):
        super().__init__()
        R, I = make_gabor_kernels(freqs, oris, phs, ksz, device)
        # Repeat kernels per input channel for grouped conv
        self.register_buffer("real_w", R.repeat(in_ch, 1, 1, 1))  # [in_ch*32,1,k,k]
        self.register_buffer("imag_w", I.repeat(in_ch, 1, 1, 1))
        self.groups = in_ch
        self.pad = ksz // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = F.conv2d(x, self.real_w, padding=self.pad, groups=self.groups)
        i = F.conv2d(x, self.imag_w, padding=self.pad, groups=self.groups)
        return torch.sqrt(r * r + i * i + 1e-6)


# =============================================================================
# WAVELET: 2D Haar decomposition + 1D Haar for binding/denoising
# =============================================================================

def haar2d_one_level(x: torch.Tensor) -> torch.Tensor:
    """
    2D Haar wavelet, one level. x: [B, C, H, W] (H, W even).
    Returns [B, C*4, H//2, W//2]: (LL, LH, HL, HH) per channel in order.
    """
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    # Row transform: low = (c0+c1)/sqrt2, high = (c0-c1)/sqrt2
    x_even = x[:, :, 0::2, :]   # [B,C,H/2,W]
    x_odd = x[:, :, 1::2, :]
    row_low = (x_even + x_odd) * (1.0 / math.sqrt(2))
    row_high = (x_even - x_odd) * (1.0 / math.sqrt(2))
    # Col transform on row_low and row_high
    ll = (row_low[:, :, :, 0::2] + row_low[:, :, :, 1::2]) * (1.0 / math.sqrt(2))
    lh = (row_low[:, :, :, 0::2] - row_low[:, :, :, 1::2]) * (1.0 / math.sqrt(2))
    hl = (row_high[:, :, :, 0::2] + row_high[:, :, :, 1::2]) * (1.0 / math.sqrt(2))
    hh = (row_high[:, :, :, 0::2] - row_high[:, :, :, 1::2]) * (1.0 / math.sqrt(2))
    # Stack: [B, C, 4, H/2, W/2] -> [B, C*4, H/2, W/2]
    out = torch.stack([ll, lh, hl, hh], dim=2)
    return out.view(B, C * 4, H // 2, W // 2)


class Wavelet2D(nn.Module):
    """
    2D Haar wavelet front-end. Input [B, C, H, W] -> output [B, C*4, H, W]
    by doing one-level decomp then upsizing subbands back to H, W so they can
    be concatenated with Gabor at same resolution.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 1), mode="reflect")
            H, W = x.shape[2], x.shape[3]
        sub = haar2d_one_level(x)   # [B, C*4, H/2, W/2]
        # Upsample each subband back to H, W for alignment with Gabor maps
        sub = F.interpolate(sub, size=(H, W), mode="bilinear", align_corners=False)
        return sub


def haar1d_one_level(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """1D Haar, one level. x: [..., L] with L even. Returns (low, high) each [..., L/2]."""
    low = (x[..., 0::2] + x[..., 1::2]) * (1.0 / math.sqrt(2))
    high = (x[..., 0::2] - x[..., 1::2]) * (1.0 / math.sqrt(2))
    return low, high


def inverse_haar1d_one_level(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Inverse 1D Haar. low, high: [..., L]. Returns [..., 2*L]."""
    x0 = (low + high) * (1.0 / math.sqrt(2))
    x1 = (low - high) * (1.0 / math.sqrt(2))
    return torch.stack([x0, x1], dim=-1).flatten(-2)


def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft thresholding: sign(x) * max(|x| - tau, 0)."""
    return torch.sign(x) * F.relu(x.abs() - tau)


def wavelet_circular_conv_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular convolution of last dim. a: [..., N], b: [..., N] or [N]."""
    # Keep FFT in fp32 under AMP; cuFFT half precision requires power-of-two sizes.
    orig_dtype = a.dtype
    n = a.size(-1)
    a32 = a.float()
    b32 = b.float()
    af = torch.fft.rfft(a32, n=n, dim=-1)
    bf = torch.fft.rfft(b32, n=n, dim=-1)
    out = torch.fft.irfft(af * bf, n=n, dim=-1).real
    return out.to(orig_dtype)


# =============================================================================
# UTIL: local receptive-field extraction for L1 (7x7x96)
# =============================================================================

def extract_rf_patches_all_channels(x: torch.Tensor, rf_size: int) -> torch.Tensor:
    """
    x: [B, C, H, W] -> patches: [B, L, C*rf*rf] where L = H*W.
    Uses reflect padding (paper allows zero padding; reflect is stable).
    """
    pad = rf_size // 2
    x_padded = F.pad(x, [pad] * 4, mode="reflect")
    patches = F.unfold(x_padded, kernel_size=rf_size)  # [B, C*rf*rf, L]
    return patches.transpose(1, 2)  # [B, L, C*rf*rf]


def build_rf_gaussian_sparse_mask(
    in_channels: int,
    rf_size: int,
    *,
    sigma_frac: float,
    keep_fraction: float,
    sparse_quantile: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Binary mask {0,1} per input dim [in_dim], aligned with ``F.unfold``.

    Isotropic Gaussian scores on the ``rf_size×rf_size`` grid pick **which** spatial sites are
    connected (center-most under the Gaussian). ``keep_fraction`` sets how many sites are **1**
    (e.g. 0.6 → ~60% of RF spatial positions are 1, rest 0); each selected site repeats for all
    ``in_channels`` at that kernel cell.

    - ``keep_fraction`` < 1: top-``keep_fraction`` sites by Gaussian score → **1**, others **0**.
    - ``keep_fraction`` == 1 and ``sparse_quantile`` > 0: binary mask via quantile threshold (legacy).
    - Otherwise: all **1** (full RF connectivity).
    """
    rf = int(rf_size)
    ic = int(in_channels)
    if rf < 1 or ic < 1:
        return torch.ones(max(1, ic * rf * rf), device=device, dtype=dtype)
    sigma_frac = float(max(1e-4, sigma_frac))
    yy, xx = torch.meshgrid(
        torch.arange(rf, device=device, dtype=torch.float32),
        torch.arange(rf, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy = (rf - 1) * 0.5
    cx = (rf - 1) * 0.5
    sigma = sigma_frac * float(rf)
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    g = torch.exp(-0.5 * dist2 / (sigma * sigma + 1e-8))
    g = g / (g.max() + 1e-8)
    scores = g.reshape(-1)
    kf = float(max(0.0, min(1.0, keep_fraction)))
    K = int(scores.numel())
    if kf < 1.0 - 1e-9:
        keep_n = max(1, min(K, int(math.ceil(kf * float(K)))))
        _, topi = torch.topk(scores, keep_n, largest=True)
        g_flat = torch.zeros(K, device=device, dtype=torch.float32)
        g_flat.scatter_(0, topi, torch.ones(keep_n, device=device, dtype=torch.float32))
    else:
        sq = float(max(0.0, min(1.0, sparse_quantile)))
        if sq > 0.0:
            thr = torch.quantile(scores, sq)
            g_flat = (scores >= thr).to(dtype=torch.float32)
        else:
            g_flat = torch.ones(K, device=device, dtype=torch.float32)
    # Unfold layout: each spatial kernel cell has ``in_channels`` consecutive values.
    mask = g_flat.unsqueeze(1).expand(-1, ic).reshape(-1)
    return mask.to(dtype=dtype)


def soft_threshold_signed(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Signed soft-threshold (soft inhibition): sign(x)*relu(|x|-tau)."""
    if tau <= 0:
        return x
    return torch.sign(x) * F.relu(torch.abs(x) - tau)


def _fft_holographic_binding(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    corr_blend: float,
    in_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    HRR-style binding in the Fourier domain (Plate / convolutional HRR).

    - ``norm="ortho"`` keeps energy scale stable (Parseval) vs unnormalized FFT products.
    - ``corr_blend`` mixes circular **convolution** (X ⊛ Y) with circular **correlation**
      (dual to binding, closer to unbinding in retrieval literature).

    ``x`` is ``[in_dim]`` (global patch mean) or ``[N, in_dim]`` (per-neuron RFs); ``y`` is ``[in_dim]``.
    """
    rho = float(max(0.0, min(1.0, corr_blend)))
    d = int(x.shape[-1] if in_dim is None else in_dim)
    y_n = y / (y.norm() + 1e-8)
    y32 = y_n.float()
    Yf = torch.fft.rfft(y32, norm="ortho")
    if x.dim() == 1:
        x_n = x / (x.norm() + 1e-8)
        x32 = x_n.float()
        Xf = torch.fft.rfft(x32, norm="ortho")
        conv = torch.fft.irfft(Xf * Yf, n=d, norm="ortho")
        if rho > 0.0:
            corr = torch.fft.irfft(Xf * torch.conj(Yf), n=d, norm="ortho")
            conv = (1.0 - rho) * conv + rho * corr
        out = conv.to(x.dtype)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    x_n = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    x32 = x_n.float()
    Xf = torch.fft.rfft(x32, dim=1, norm="ortho")
    prod = Xf * Yf.view(1, -1)
    conv = torch.fft.irfft(prod, n=d, dim=1, norm="ortho")
    if rho > 0.0:
        prod_c = Xf * torch.conj(Yf).view(1, -1)
        corr = torch.fft.irfft(prod_c, n=d, dim=1, norm="ortho")
        conv = (1.0 - rho) * conv + rho * corr
    return torch.nan_to_num(conv.to(x.dtype), nan=0.0, posinf=0.0, neginf=0.0)


def smooth_inhibition_saturation(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Smooth, differentiable saturation without hard thresholding:
      scale * tanh(x / scale)
    Preserves sign; small |x| pass through ~linearly; large |x| compress.
    Set scale <= 0 to return x unchanged (caller should skip when scale is 0).
    """
    s = float(scale)
    if s <= 0.0:
        return x
    return s * torch.tanh(x / (s + 1e-8))


# =============================================================================
# UNIFIED LOCAL PLASTICITY LAYERS
# - Topographic 2D layer: 32x32 map (1024 neurons), local RFs via unfold
# =============================================================================

@dataclass
class UnifiedCoeffs:
    # NOTE: Local plasticity LR (Hebbian step scale; also used as lr_lateral base).
    # The reference (ConsistencyLearning5) uses ~1e-6 but
    # our implementation normalizes several terms; a slightly larger default is stable.
    eta: float = 5e-9
    alpha: float = 0.1
    # Holographic binding controls
    holo_update_freq: int = 20
    holo_fast_lr: float = 0.1
    holo_fast_decay: float = 0.1
    # Blend circular correlation with convolution in FFT HRR (0 = pure conv, 1 = pure corr).
    holo_corr_blend: float = 0.2
    # Cap ||M_holo_fast||_F to limit long-run drift (0 = uncapped).
    holo_fast_norm_cap: float = 12.0

    # Hyperbolic-specific coefficients (mirrors Hyperbolic3_* scripts)
    distance_gradient_lr: float = 0.01  # ΔW -= lr * ∇_w d_h(x, w) when enabled
    beta_hyp: float = 0.01             # Möbius/hyperbolic binding strength

    # Wavelet: binding in wavelet domain + optional denoising before Hebbian
    gamma_wavelet: float = 0.01        # wavelet binding strength (scale-aligned binding)
    wavelet_threshold: float = 0.02    # soft-threshold for wavelet denoising (0 = off)
    use_wavelet_denoise: bool = True   # soft-threshold patch in 1D Haar before Hebbian
    use_wavelet_binding: bool = True  # bind patch with y in wavelet domain
    # If False: Gabor-only per stream (24 ch); no Haar wavelet bands concatenated at the input.
    use_wavelet_input: bool = True

    # Regularization / decay coefficients for unified plasticity
    lambda_a: float = 0.001   # anti-Hebbian (decorrelation)
    lambda_c: float = 0.001   # consistency (temporal)
    lambda_r: float = 0.001   # recursive (attractor)
    lambda_F: float = 0.001   # free-energy (predictive)
    lambda_d: float = 0.1    # weight decay on W (plasticity decay)

    # Entropy-based dropout: scale dropout p by activation entropy (higher entropy -> higher p)
    use_entropy_dropout: bool = False
    entropy_dropout_scale: float = 0.5

    # Entropy-modulated plasticity decay: scale lambda_d by activation entropy
    use_entropy_plasticity_decay: bool = False
    entropy_decay_scale: float = 0.5

    # Inhibition options
    #inhibition_mode: str = "global"  # "global", "kernel", "averaging", or "mixed"
    inhibition_mode: str = "kernel"
    #use_local_inhibition: bool = True

    use_soft_inhibition: bool = False
    # If True: no hard/soft threshold on inhibition signals — raw conv/L outputs (still scaled by inhibition_strength).
    # Optional inhibition_smooth_scale applies tanh saturation only (still no τ cut-off).
    inhibition_no_threshold: bool = True
    inhibition_smooth_scale: float = 0.0
    averaging_inhibition_size: int = 7
    kernel_inhibition_size: int = 7
    kernel_inhibition_sigma: float = 1.5
    mixed_inhibition_w_global: float = 1.0 / 3.0
    mixed_inhibition_w_kernel: float = 1.0 / 3.0
    mixed_inhibition_w_averaging: float = 1.0 / 3.0
    inhibition_dropout: float = 0.0
    # On by default; runs after competitive inhibition (if any), independent of inhibition strength.
    use_divisive_inhibition_norm: bool = True
    # "global": mean |y| per image (broadcast to units)
    # "local": Gaussian spatial pool of |y| per unit
    # "both" / "all": mix global + local (use both terms; weights normalized in layer __init__)
    # Tuned defaults: 9×9 Gaussian σ=2.0; α,β for stable gain after L2 norm.
    # See TUNING_DIVISIVE_PFC9.md for a short empirical check.
    divisive_mode: str = "both"
    divisive_w_global: float = 0.5
    divisive_w_local: float = 0.5
    divisive_local_size: int = 9
    divisive_local_sigma: float = 2.0
    divisive_alpha: float = 0.85
    # Tuned (6 ep, 1.5% train, seed 42): β=0.36, w_g=w_l=0.5 gave best BestVal vs nearby β and weights.
    divisive_beta: float = 0.36
    adaptive_inhibition: bool = False
    target_active_frac: float = 0.01
    inhibition_adapt_lr: float = 0.01
    inhibition_min: float = 0.0
    inhibition_max: float = 1.0
    inhibition_learning_sparsity: float = 0.0

    # k-Winners with Homeostatic Adaptive Threshold (per-neuron theta_i)
    # Replace fixed WTA score y with y - theta and update theta_i so each neuron
    # wins at about the expected sparsity rate.
    use_homeostatic_threshold: bool = False
    homeostatic_threshold_lr: float = 0.02
    homeostatic_threshold_theta_init: float = 0.0
    homeostatic_threshold_min: float = -2.0
    homeostatic_threshold_max: float = 2.0

    # Oscillatory / phase inhibition (optional gating for competition)
    use_oscillatory_inhibition: bool = False
    phase_period: int = 10
    phase_gate_sharpness: float = 1.0

    # Cascade + skip connections (PFC17): deeper layers see concatenated earlier maps.
    # L2: [L1_out, mean(L1_input)] ; L3: [L2_out, L1_out] ; L4: [L3_out, L2_out, L1_out] (all [B,1,H,W] per part).
    cascade_skip_connections: bool = True

    # Receptive-field feedforward: Gaussian taper within RF (center-strong) + optional hard sparsity.
    # ``rf_connectivity_keep_frac`` (default 0.6): binary {0,1} mask; fraction of RF **spatial** sites = 1
    # (top sites by Gaussian score → center blob); repeated per channel → ~60% of input dims are 1.
    rf_connectivity_gaussian: bool = False
    rf_gaussian_sigma_frac: float = 0.28
    rf_connectivity_keep_frac: float = 0.6
    rf_connectivity_sparse_quantile: float = 0.0

    # Excitatory/Inhibitory (EI) neuron populations with mutual inhibition.
    # Implemented as E = relu(y), I = relu(-y), then mutual learned-L coupling in global mode.
    use_ei_neurons: bool = False
    ei_mutual_inhibition: bool = False
    # If True: I->E uses L_ei (<=0), E->I uses L_ie (>=0). If False: shared L for both (legacy).
    ei_separate_lateral: bool = True
    # EI matrix initialization (for separate mode)
    ei_l_ei_init: float = -0.01
    ei_l_ie_init: float = 0.01

    # Learned lateral L in global/mixed: multi-term plasticity (weights on each term; 0 = off).
    # ΔL = lr_lateral * inh_scale * ( lateral_w_hebb*corr + lateral_w_anti*(-corr) + lateral_w_cov*cov
    #       + lateral_w_holo*holo_assoc + lateral_w_hyp*hyp_assoc + lateral_w_wave*wave_assoc
    #       - inhibition_decay*L - lateral_w_oja*oja_stab ), then threshold/sparsity/clamp.
    # corr = y^T y (sum over batch, not /B); cov = centered batch outer product; oja_stab = L * mean(y_i^2+y_j^2)/2.
    # Legacy behavior: lateral_w_hebb=1, others 0, inhibition_decay as before.
    lateral_w_hebb: float = 1.0
    lateral_w_anti: float = 0.0
    lateral_w_cov: float = 0.0
    lateral_w_holo: float = 0.05
    lateral_w_hyp: float = 0.05
    lateral_w_wave: float = 0.05
    lateral_w_oja: float = 0.0

    # --- SOM-inspired inhibition schedules (shrink neighborhood, decay lateral LR, soft competition) ---
    # Off by default: SOM schedules + softmax competition can destabilize early CE; opt-in via CLI.
    som_enabled: bool = False
    # Interpolate kernel Gaussian σ from start (broad) to end (sharp) over training progress [0,1].
    kernel_inhibition_sigma_start: float = 2.5
    kernel_inhibition_sigma_end: float = 1.0
    # Lateral matrix learning rate: full scale until warmup fraction, then linearly to lr_lateral_schedule_end_scale.
    som_lr_lateral_warmup_fraction: float = 0.2
    lr_lateral_schedule_end_scale: float = 0.5
    # Softmax spatial competition (BMU-like soft assignment); temperature → 0 sharpens winners.
    use_inhibition_softmax: bool = False
    inhibition_softmax_temp_start: float = 1.5
    inhibition_softmax_temp_end: float = 0.45
    # WTA sparsity scales from start→end × layer base (higher sparsity = fewer survivors).
    som_wta_scale_start: float = 0.88
    som_wta_scale_end: float = 1.0
    # Mexican-hat (DoG) inhibition: narrow center vs wide surround (σ and gain scheduled).
    mexican_center_sigma_start: float = 0.65
    mexican_center_sigma_end: float = 0.45
    mexican_surround_sigma_start: float = 2.4
    mexican_surround_sigma_end: float = 1.75
    mexican_surround_gain_start: float = 1.0
    mexican_surround_gain_end: float = 1.15

    # Mixture rule for unsupervised local update terms.
    # fixed: deterministic weighted sum using base weights
    # adaptive: "smart" balancing using base weights and inverse term norms
    # random: sample weights from Dirichlet around base weights
    unsup_mix_mode: str = "adaptive"
    unsup_mix_normalize_terms: bool = True
    unsup_mix_temperature: float = 1.0
    unsup_mix_random_alpha: float = 20.0
    unsup_mix_w_hebb: float = 0.30
    unsup_mix_w_holo: float = 0.10
    unsup_mix_w_hyp: float = 0.10
    unsup_mix_w_wave: float = 0.15
    unsup_mix_w_anti: float = 0.10
    unsup_mix_w_cons: float = 0.08
    unsup_mix_w_rec: float = 0.08
    unsup_mix_w_free: float = 0.06
    unsup_mix_w_decay: float = 0.03
    unsup_mix_w_dist: float = 0.00

    # Structural plasticity: periodic prune/grow schedule on local synapses.
    use_structural_plasticity: bool = True
    # Lower frequency reduces catastrophic accumulation of prune/grow steps on long runs.
    structural_update_freq: int = 1200
    structural_prune_threshold: float = 1e-4
    structural_prune_max_frac: float = 0.002
    structural_grow_threshold: float = 0.02
    structural_grow_max_frac: float = 0.001
    structural_grow_init_scale: float = 0.02

    # Per-neuron predictive-coding plasticity: row-wise gain on η·dW — blend of
    # (this layer's spatial activity) + (PFC TE map per freq) + (local trace surprise).
    pc_per_neuron_plasticity: bool = True
    pc_per_neuron_layer_weight: float = 0.34
    pc_per_neuron_pfc_weight: float = 0.33
    pc_per_neuron_trace_weight: float = 0.33
    pc_per_neuron_gain_min: float = 0.5
    pc_per_neuron_gain_max: float = 1.5


class UnifiedPlasticityLayer(nn.Module):
    """
    A vector layer with N neurons:
      y = ReLU(LN(W x + b - inh*(L y)))   (with optional recurrent refinement)
    and local plasticity implementing the manuscript's unified update.
    """

    def __init__(
        self,
        in_dim: int,
        n_neurons: int = 8100,
        name: str = "L?",
        coeffs: UnifiedCoeffs = UnifiedCoeffs(),
        beta_trace: float = 0.8,
        use_hebbian: bool = True,
        use_antihebbian: bool = True,
        use_holographic: bool = True,
        use_consistency: bool = True,
        use_recursive: bool = True,
        use_free_energy: bool = True,
        use_active_inference: bool = True,
        recursive_iters: int = 5,
        inh_strength: float = 0.1,
        lr_lateral: float = 1e-3,
        lateral_decay: float = 1e-4,
        w_clip: float = 5.0,
        l_clip: float = 1.0,
        active_inference_steps: int = 3,
        active_inference_lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.name = name
        self.in_dim = in_dim
        self.n = n_neurons
        self.coeffs = coeffs
        self.beta_trace = beta_trace
        self.use_hebbian = use_hebbian
        self.use_antihebbian = use_antihebbian
        self.use_holographic = use_holographic
        self.use_consistency = use_consistency
        self.use_recursive = use_recursive
        self.use_free_energy = use_free_energy
        self.use_active_inference = use_active_inference
        self.recursive_iters = recursive_iters
        self.inh_strength = inh_strength
        self.lr_lateral = lr_lateral
        self.lateral_decay = lateral_decay
        self.w_clip = w_clip
        self.l_clip = l_clip
        self.active_inference_steps = active_inference_steps
        self.active_inference_lr = active_inference_lr

        dev = device or torch.device("cpu")

        # Main synapses (NO gradients)
        self.W = nn.Parameter(torch.empty(n_neurons, in_dim), requires_grad=False)
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(n_neurons), requires_grad=False)

        # Learned lateral inhibition weights (anti-Hebbian, NO gradients)
        # Reference-style init: -0.1 off-diagonal, diagonal = 0.
        self.L = nn.Parameter(torch.full((n_neurons, n_neurons), -0.1, device=dev), requires_grad=False)
        with torch.no_grad():
            self.L.fill_diagonal_(0.0)

        # Holographic embedding to resolve dim mismatch (y is N, x is in_dim).
        # Fixed random projection y -> in_dim so both are length in_dim for convolution.
        P = torch.randn(in_dim, n_neurons, device=dev) / math.sqrt(n_neurons)
        self.register_buffer("P_holo", P)

        # Generator for free-energy/predictive coding (NO gradients)
        self.G = nn.Parameter(torch.empty(in_dim, n_neurons), requires_grad=False)  # x_hat = G y
        nn.init.xavier_uniform_(self.G)

        # State buffers
        self.register_buffer("trace", torch.zeros(n_neurons))
        self.register_buffer("prev_y", torch.zeros(1))  # set on first batch

        # For LayerNorm-like stabilization (running stats, NO gradients)
        self.register_buffer("running_mean", torch.zeros(n_neurons))
        self.register_buffer("running_var", torch.ones(n_neurons))
        self.momentum = 0.1

        # Activation (match Best_Ever style): LayerNorm -> L2 normalize
        self.layer_norm = nn.LayerNorm(n_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        y = x @ self.W.t() + self.b  # [B, N]
        # Activation (requested): LayerNorm -> L2 normalization
        y = self.layer_norm(y)
        y = F.normalize(y, p=2, dim=1)

        # Lateral inhibition (learned)
        if self.inh_strength > 0:
            y = y - self.inh_strength * (y @ self.L.t())
            y = self.layer_norm(y)
            y = F.normalize(y, p=2, dim=1)

        # Recursive attractor refinement (paper: iterative refinement)
        if self.use_recursive and self.recursive_iters > 0:
            y0 = y
            yt = y
            for _ in range(self.recursive_iters):
                yt = yt - self.inh_strength * (yt @ self.L.t())
                yt = self.layer_norm(yt)
                yt = F.normalize(yt, p=2, dim=1)
            y = yt
            # Keep y0 for plasticity term
            self._last_y0 = y0.detach()
            self._last_yT = y.detach()
        else:
            self._last_y0 = y.detach()
            self._last_yT = y.detach()

        return y

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Apply unified local update. Uses batch averages for stability.
        x: [B, in_dim], y: [B, N]
        """
        B = x.size(0)
        x_b = x
        y_b = y

        # Trace (paper Eq. trace): uses previous activity
        # If beta_trace == 0: disable temporal smoothing and use instantaneous y_mean.
        y_mean = y_b.mean(dim=0)
        if self.beta_trace > 0.0:
            self.trace.mul_(1 - self.beta_trace).add_(self.beta_trace * y_mean)

        # Optionally do active inference on y (optimize latent to minimize free energy).
        # IMPORTANT: we enable gradients only for the latent optimization step.
        if self.use_active_inference and self.use_free_energy and self.active_inference_steps > 0:
            with torch.enable_grad():
                z = y_mean.detach().clone().requires_grad_(True)
                x_target = x_b.mean(dim=0).detach()
                for _ in range(self.active_inference_steps):
                    x_hat = self.G.detach() @ z  # treat weights as constants
                    F_energy = (x_target - x_hat).pow(2).mean() + 1e-3 * z.pow(2).mean()
                    (grad,) = torch.autograd.grad(F_energy, z, retain_graph=False, create_graph=False)
                    z = (z - self.active_inference_lr * grad).detach().requires_grad_(True)
                y_eff = z.detach()
        else:
            y_eff = y_mean.detach() if self.beta_trace <= 0.0 else self.trace.detach()
            # trace-modulated effective post-syn activity (or instantaneous y_mean if beta_trace==0)

        # Hebbian term: y_eff x^T
        dW = torch.zeros_like(self.W)
        if self.use_hebbian:
            x_mean = x_b.mean(dim=0)
            dW.add_(torch.ger(y_eff, x_mean) / (self.in_dim + 1e-8))

        # Holographic binding term: HRR-style circular bind( P y, x ) -> outer with y_eff
        if self.use_holographic:
            x_mean = x_b.mean(dim=0)
            y_embed = (self.P_holo @ y_eff).contiguous()
            rho = float(getattr(self.coeffs, "holo_corr_blend", 0.2))
            conv = _fft_holographic_binding(x_mean, y_embed, corr_blend=rho, in_dim=self.in_dim)
            dW.add_(self.coeffs.alpha * torch.ger(y_eff, conv) / (self.in_dim + 1e-8))

        # Consistency term (paper: temporal stability)
        if self.use_consistency and self.prev_y.numel() == self.n:
            # Encourage similarity across time by penalizing diff
            diff = y_mean - self.prev_y
            x_mean = x_b.mean(dim=0)
            dW.add_(-self.coeffs.lambda_c * torch.ger(diff, x_mean) / (self.in_dim + 1e-8))

        # Recursive term (paper Eq. recurse): (yT - y0) x^T
        if self.use_recursive:
            dy = (self._last_yT.mean(dim=0) - self._last_y0.mean(dim=0))
            x_mean = x_b.mean(dim=0)
            dW.add_(self.coeffs.lambda_r * torch.ger(dy, x_mean) / (self.in_dim + 1e-8))

        # Free energy term (paper Eq. FE): (x - x_hat) y^T (here mapped into W update)
        if self.use_free_energy:
            x_mean = x_b.mean(dim=0)
            y_mean = y_b.mean(dim=0)
            x_hat = self.G @ y_mean  # [in_dim]
            err = (x_mean - x_hat)   # [in_dim]
            # Update generator (local Hebbian on generator weights)
            self.G.add_(0.001 * torch.ger(err, y_mean))
            # Feedforward update component from prediction error
            dW.add_(self.coeffs.lambda_F * torch.ger(y_mean, err) / (self.in_dim + 1e-8))

        # Weight decay (optionally scaled by activation entropy)
        lambda_d = self.coeffs.lambda_d
        if getattr(self.coeffs, "use_entropy_plasticity_decay", False) and lambda_d > 0:
            norm_entropy = _normalized_entropy_activations(y_eff.unsqueeze(0))
            scale = 1.0 + getattr(self.coeffs, "entropy_decay_scale", 0.5) * norm_entropy
            lambda_d = lambda_d * scale
        dW.add_(-lambda_d * self.W)

        # Apply update (weights are no-grad parameters)
        with torch.no_grad():
            dW = torch.nan_to_num(dW, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip update magnitude to prevent blow-ups
            dW_norm = dW.norm()
            if torch.isfinite(dW_norm) and dW_norm > 1.0:
                dW.mul_(1.0 / (dW_norm + 1e-8))
            self.W.add_(self.coeffs.eta * dW)
            # Hard clip only (no post-update normalization per user request)
            self.W.clamp_(-self.w_clip, self.w_clip)

        # Anti-Hebbian decorrelation on lateral weights: ΔL = -η y y^T (off-diagonal)
        if self.use_antihebbian:
            with torch.no_grad():
                corr = (y_b.t() @ y_b) / float(B)  # [N,N]
                corr.fill_diagonal_(0.0)
                # L is an inhibitory weight matrix used as: y - inh*(y @ L^T)
                # So it must be NON-NEGATIVE. Increase it for co-active pairs.
                self.L.mul_(1.0 - self.lateral_decay).add_(self.lr_lateral * corr)
                self.L.fill_diagonal_(0.0)
                self.L.clamp_(0.0, self.l_clip)

        # Save prev_y
        with torch.no_grad():
            self.prev_y = y_mean.detach().clone()


class TopographicUnifiedLayer(nn.Module):
    """
    2D topographic layer with 32x32 layout (1024 neurons), each neuron has a local RF.

    - Input: x_map [B, C_in, H, W]
    - Output: y_map [B, 1, H, W]  (1024 neurons)
    - RF extraction uses unfold: patches [B, N, C_in*rf*rf]

    Unified local update is applied per-neuron using its own patch mean.
    """

    def __init__(
        self,
        in_channels: int,
        rf_size: int,
        spatial_size: int = 90,
        name: str = "L?",
        coeffs: UnifiedCoeffs = UnifiedCoeffs(),
        beta_trace: float = 0.9,
        use_hebbian: bool = True,
        use_antihebbian: bool = True,
        use_holographic: bool = True,
        use_consistency: bool = True,
        use_recursive: bool = True,
        use_free_energy: bool = True,
        use_active_inference: bool = True,
        recursive_iters: int = 5,
        inhibition_strength: float = 0.1,
        inhibition_decay: float = 0.01,
        competition_threshold: float = 0.1,
        lateral_update_freq: int = 10,
        lr_lateral: float = 1e-3,
        # Reference clamps weights to [-1, 1]; keep that as default for stability.
        w_clip: float = 1.0,
        active_inference_steps: int = 3,
        active_inference_lr: float = 1e-1,
        wta_sparsity: float = 0.9,
        # Reference computes holographic term less frequently.
        holo_update_freq: Optional[int] = None,
        use_hyperbolic_binding: bool = True,
        use_distance_gradient: bool = False,
        gradient_chunk_size: Optional[int] = 512,
        use_wavelet_binding: bool = True,
        use_wavelet_denoise: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.rf_size = rf_size
        self.spatial_size = spatial_size
        self.N = spatial_size * spatial_size
        self.in_dim = in_channels * rf_size * rf_size
        self.pad = rf_size // 2

        self.coeffs = coeffs
        self.beta_trace = beta_trace
        self.use_hebbian = use_hebbian
        self.use_antihebbian = use_antihebbian
        self.use_holographic = use_holographic
        self.use_consistency = use_consistency
        self.use_recursive = use_recursive
        self.use_free_energy = use_free_energy
        self.use_active_inference = use_active_inference
        self.use_hyperbolic_binding = bool(use_hyperbolic_binding)
        self.use_distance_gradient = bool(use_distance_gradient)
        self.gradient_chunk_size = gradient_chunk_size
        self.use_wavelet_binding = bool(use_wavelet_binding)
        self.use_wavelet_denoise = bool(use_wavelet_denoise)
        self.recursive_iters = recursive_iters
        # Inhibition (match ConsistencyLearning5-style)
        self.inhibition_strength = float(inhibition_strength)
        self.current_inhibition_strength = float(inhibition_strength)
        self.inhibition_decay = float(inhibition_decay)
        self.competition_threshold = float(competition_threshold)
        self.lateral_update_freq = int(lateral_update_freq)
        self.lr_lateral = lr_lateral
        self.inhibition_mode = str(getattr(coeffs, "inhibition_mode", "global")).lower()
        self.use_ei_neurons = bool(getattr(coeffs, "use_ei_neurons", False))
        self.ei_mutual_inhibition = bool(getattr(coeffs, "ei_mutual_inhibition", False))
        self.ei_separate_lateral = bool(getattr(coeffs, "ei_separate_lateral", True))
        self.ei_l_ei_init = float(getattr(coeffs, "ei_l_ei_init", -0.1))
        self.ei_l_ie_init = float(getattr(coeffs, "ei_l_ie_init", 0.1))
        self.use_soft_inhibition = bool(getattr(coeffs, "use_soft_inhibition", False))
        self.inhibition_no_threshold = bool(getattr(coeffs, "inhibition_no_threshold", False))
        self.inhibition_smooth_scale = float(max(0.0, getattr(coeffs, "inhibition_smooth_scale", 0.0)))
        self.use_divisive_inhibition_norm = bool(getattr(coeffs, "use_divisive_inhibition_norm", False))
        _dm = str(getattr(coeffs, "divisive_mode", "both")).lower().strip()
        if _dm == "all":
            _dm = "both"
        self.divisive_mode = _dm
        wg = float(getattr(coeffs, "divisive_w_global", 0.5))
        wl = float(getattr(coeffs, "divisive_w_local", 0.5))
        wsum = max(1e-8, abs(wg) + abs(wl))
        self.divisive_w_global = abs(wg) / wsum
        self.divisive_w_local = abs(wl) / wsum
        self.divisive_alpha = float(getattr(coeffs, "divisive_alpha", 0.85))
        self.divisive_beta = float(getattr(coeffs, "divisive_beta", 0.36))
        self.inhibition_dropout = float(max(0.0, min(1.0, getattr(coeffs, "inhibition_dropout", 0.0))))
        self.adaptive_inhibition = bool(getattr(coeffs, "adaptive_inhibition", False))
        self.target_active_frac = float(getattr(coeffs, "target_active_frac", 0.01))
        self.inhibition_adapt_lr = float(getattr(coeffs, "inhibition_adapt_lr", 0.01))
        self.inhibition_min = float(getattr(coeffs, "inhibition_min", 0.0))
        self.inhibition_max = float(getattr(coeffs, "inhibition_max", 1.0))
        self.inhibition_learning_sparsity = float(
            max(0.0, min(1.0, getattr(coeffs, "inhibition_learning_sparsity", 0.0)))
        )
        # Homeostatic adaptive threshold for k-Winners gating.
        self.use_homeostatic_threshold = bool(getattr(coeffs, "use_homeostatic_threshold", False))
        self.homeostatic_threshold_lr = float(getattr(coeffs, "homeostatic_threshold_lr", 0.02))
        self.homeostatic_threshold_theta_init = float(getattr(coeffs, "homeostatic_threshold_theta_init", 0.0))
        self.homeostatic_threshold_min = float(getattr(coeffs, "homeostatic_threshold_min", -2.0))
        self.homeostatic_threshold_max = float(getattr(coeffs, "homeostatic_threshold_max", 2.0))
        self.lateral_w_hebb = float(getattr(coeffs, "lateral_w_hebb", 1.0))
        self.lateral_w_anti = float(getattr(coeffs, "lateral_w_anti", 0.0))
        self.lateral_w_cov = float(getattr(coeffs, "lateral_w_cov", 0.0))
        self.lateral_w_holo = float(getattr(coeffs, "lateral_w_holo", 0.05))
        self.lateral_w_hyp = float(getattr(coeffs, "lateral_w_hyp", 0.05))
        self.lateral_w_wave = float(getattr(coeffs, "lateral_w_wave", 0.05))
        self.lateral_w_oja = float(getattr(coeffs, "lateral_w_oja", 0.0))
        wg = float(getattr(coeffs, "mixed_inhibition_w_global", 1.0 / 3.0))
        wk = float(getattr(coeffs, "mixed_inhibition_w_kernel", 1.0 / 3.0))
        wa = float(getattr(coeffs, "mixed_inhibition_w_averaging", 1.0 / 3.0))
        wsum = max(1e-8, wg + wk + wa)
        self.mixed_inh_w_global = wg / wsum
        self.mixed_inh_w_kernel = wk / wsum
        self.mixed_inh_w_averaging = wa / wsum
        self.w_clip = w_clip
        self.active_inference_steps = active_inference_steps
        self.active_inference_lr = active_inference_lr
        # Global WTA: keep top (1 - wta_sparsity) fraction active (per sample)
        self.wta_sparsity = float(wta_sparsity)
        self.holo_update_freq = int(
            getattr(coeffs, "holo_update_freq", 20) if holo_update_freq is None else holo_update_freq
        )

        dev = device or torch.device("cpu")

        _rfg = bool(getattr(coeffs, "rf_connectivity_gaussian", False))
        _sff = float(getattr(coeffs, "rf_gaussian_sigma_frac", 0.28))
        _rkf = float(getattr(coeffs, "rf_connectivity_keep_frac", 0.6))
        _rsq = float(getattr(coeffs, "rf_connectivity_sparse_quantile", 0.0))
        if _rfg:
            _rm = build_rf_gaussian_sparse_mask(
                self.in_channels,
                self.rf_size,
                sigma_frac=_sff,
                keep_fraction=_rkf,
                sparse_quantile=_rsq,
                device=dev,
                dtype=torch.float32,
            )
            self.register_buffer("rf_connectivity_mask", _rm)
        else:
            self.register_buffer("rf_connectivity_mask", torch.ones(self.in_dim, device=dev, dtype=torch.float32))

        # One weight vector per neuron/location (VisNet-style), NO gradients
        self.W = nn.Parameter(torch.empty(self.N, self.in_dim), requires_grad=False)
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(self.N), requires_grad=False)
        # Per-neuron adaptive winner thresholds (buffer; updated in forward during training).
        # Used as: scores = y - theta; select top-k winners in scores space.
        self.register_buffer(
            "winner_theta",
            torch.full((self.N,), self.homeostatic_threshold_theta_init, device=dev, dtype=torch.float32),
        )

        # Learned lateral inhibition weights (NEGATIVE values), NO gradients
        # Reference-style init: -0.1, diag 0, clamp to [-1, 0]
        self.L = nn.Parameter(torch.full((self.N, self.N), -0.1, device=dev), requires_grad=False)
        with torch.no_grad():
            self.L.fill_diagonal_(0.0)
        # E/I-specific laterals (only used when ei_separate_lateral + mutual EI):
        #   L_ei: I -> E (inhibitory, same sign convention as L)
        #   L_ie: E -> I (excitatory drive, nonnegative)
        self.L_ei = nn.Parameter(torch.full((self.N, self.N), self.ei_l_ei_init, device=dev), requires_grad=False)
        self.L_ie = nn.Parameter(torch.full((self.N, self.N), self.ei_l_ie_init, device=dev), requires_grad=False)
        with torch.no_grad():
            self.L_ei.fill_diagonal_(0.0)
            self.L_ie.fill_diagonal_(0.0)

        self.register_buffer("update_counter", torch.tensor(0, dtype=torch.long))
        self._cache_yE_lateral: Optional[torch.Tensor] = None
        self._cache_yI_lateral: Optional[torch.Tensor] = None
        self.som_enabled = bool(getattr(coeffs, "som_enabled", False))
        self.use_inhibition_softmax = bool(getattr(coeffs, "use_inhibition_softmax", False))
        self.lr_lateral_base = float(lr_lateral)
        self._wta_sparsity_base = float(wta_sparsity)
        self._softmax_competition_temp = float(
            getattr(coeffs, "inhibition_softmax_temp_start", 1.5)
        )
        # Oscillatory / phase inhibition: per-neuron preferred phase + global time.
        self.use_oscillatory_inhibition = bool(getattr(coeffs, "use_oscillatory_inhibition", False))
        self.phase_period = int(max(1, getattr(coeffs, "phase_period", 10)))
        self.phase_gate_sharpness = float(getattr(coeffs, "phase_gate_sharpness", 1.0))
        if self.use_oscillatory_inhibition:
            # Preferred phase per neuron in [0, 2pi)
            phase_pref = torch.rand(self.N, device=dev, dtype=torch.float32) * (2.0 * math.pi)
            self.register_buffer("phase_pref", phase_pref)
            self.register_buffer("phase_t", torch.zeros((), device=dev, dtype=torch.long))
        if self.inhibition_mode in ("kernel", "mixed"):
            ksz = int(getattr(coeffs, "kernel_inhibition_size", 7))
            if ksz % 2 == 0:
                ksz += 1
            sigma = float(
                getattr(coeffs, "kernel_inhibition_sigma_start", coeffs.kernel_inhibition_sigma)
                if self.som_enabled
                else getattr(coeffs, "kernel_inhibition_sigma", 1.5)
            )
            kernel = _gaussian2d(ksz, sigma, dev).view(1, 1, ksz, ksz)
            self.register_buffer("inhibition_kernel", kernel)
        if self.inhibition_mode == "mexican":
            ksz = int(getattr(coeffs, "kernel_inhibition_size", 7))
            if ksz % 2 == 0:
                ksz += 1
            sc = float(getattr(coeffs, "mexican_center_sigma_start", 0.65))
            ss = float(getattr(coeffs, "mexican_surround_sigma_start", 2.4))
            sg = float(getattr(coeffs, "mexican_surround_gain_start", 1.0))
            mk = _mexican_hat_kernel2d(ksz, sc, ss, sg, dev).view(1, 1, ksz, ksz)
            self.register_buffer("mexican_kernel", mk)
        if self.inhibition_mode in ("averaging", "mixed"):
            ksz = int(getattr(coeffs, "averaging_inhibition_size", 7))
            if ksz % 2 == 0:
                ksz += 1
            kernel = torch.ones((1, 1, ksz, ksz), device=dev, dtype=torch.float32) / float(ksz * ksz)
            self.register_buffer("averaging_inhibition_kernel", kernel)

        # Local / mixed divisive: Gaussian pool of |y| (same-size padding as inhibition convs)
        if self.use_divisive_inhibition_norm and self.divisive_mode in ("local", "both"):
            dsz = int(getattr(coeffs, "divisive_local_size", 7))
            if dsz % 2 == 0:
                dsz += 1
            dsig = float(getattr(coeffs, "divisive_local_sigma", 1.5))
            self.register_buffer(
                "divisive_local_kernel",
                _gaussian2d(dsz, dsig, dev).view(1, 1, dsz, dsz),
            )

        # Holographic embedding y (N) -> in_dim
        P = torch.randn(self.in_dim, self.N, device=dev) / math.sqrt(self.N)
        self.register_buffer("P_holo", P)

        # Fast HRR weights for hyperbolic binding (mirrors Hyperbolic3_* scripts)
        self.register_buffer("M_holo_fast", torch.zeros(self.N, self.in_dim, device=dev))
        self.holo_fast_decay = float(getattr(coeffs, "holo_fast_decay", 0.1))
        self.holo_fast_lr = float(getattr(coeffs, "holo_fast_lr", 0.2))

        # Generator for free-energy (predict patch-vector mean), NO gradients
        self.G = nn.Parameter(torch.empty(self.in_dim, self.N), requires_grad=False)
        nn.init.xavier_uniform_(self.G)

        # Running stats + traces
        self.register_buffer("running_mean", torch.zeros(self.N))
        self.register_buffer("running_var", torch.ones(self.N))
        self.momentum = 0.1
        self.register_buffer("trace", torch.zeros(self.N))
        self.register_buffer("prev_y", torch.zeros(1))
        # Last batch's |G^T err| per neuron for shared FE ↔ PFC top-down coupling (see update_from_patches).
        self.register_buffer("last_fe_signal_per_neuron", torch.zeros(self.N))

        # Activation (match Best_Ever style): LayerNorm -> L2 normalize
        self.layer_norm = nn.LayerNorm(self.N)

        # For recursive term
        self._last_y0 = None
        self._last_yT = None

    @torch.no_grad()
    def apply_som_schedule(self, progress: float, coeffs: "UnifiedCoeffs") -> None:
        """
        SOM-style schedules: shrinking Gaussian σ, Mexican-hat σ/gain, lateral lr, softmax T, WTA scale.
        progress in [0, 1] over training (epoch 1 -> 0, last epoch -> 1).
        """
        if not getattr(coeffs, "som_enabled", False):
            return
        p = float(max(0.0, min(1.0, progress)))

        # Lateral learning rate (anti-Hebbian on L): decay after warmup fraction
        wf = float(max(0.0, min(1.0, getattr(coeffs, "som_lr_lateral_warmup_fraction", 0.2))))
        end_scale = float(getattr(coeffs, "lr_lateral_schedule_end_scale", 0.5))
        if p < wf:
            lr_scale = 1.0
        else:
            p2 = (p - wf) / (1.0 - wf + 1e-8)
            lr_scale = 1.0 + (end_scale - 1.0) * p2
        self.lr_lateral = float(self.lr_lateral_base * lr_scale)

        # Softmax competition temperature
        t0 = float(getattr(coeffs, "inhibition_softmax_temp_start", 1.5))
        t1 = float(getattr(coeffs, "inhibition_softmax_temp_end", 0.45))
        self._softmax_competition_temp = t0 + (t1 - t0) * p

        # WTA sparsity scale (increase sparsity toward end of training)
        ws0 = float(getattr(coeffs, "som_wta_scale_start", 0.88))
        ws1 = float(getattr(coeffs, "som_wta_scale_end", 1.0))
        wscale = ws0 + (ws1 - ws0) * p
        self.wta_sparsity = float(min(0.99, max(0.0, self._wta_sparsity_base * wscale)))

        dev = self.W.device
        ksz = int(getattr(coeffs, "kernel_inhibition_size", 7))
        if ksz % 2 == 0:
            ksz += 1

        sig0 = float(getattr(coeffs, "kernel_inhibition_sigma_start", 2.5))
        sig1 = float(getattr(coeffs, "kernel_inhibition_sigma_end", 1.0))
        sigma = sig0 + (sig1 - sig0) * p

        if self.inhibition_mode in ("kernel", "mixed") and hasattr(self, "inhibition_kernel"):
            self.inhibition_kernel.copy_(_gaussian2d(ksz, sigma, dev).view(1, 1, ksz, ksz))

        if self.inhibition_mode == "mexican" and hasattr(self, "mexican_kernel"):
            sc0 = float(getattr(coeffs, "mexican_center_sigma_start", 0.65))
            sc1 = float(getattr(coeffs, "mexican_center_sigma_end", 0.45))
            ss0 = float(getattr(coeffs, "mexican_surround_sigma_start", 2.4))
            ss1 = float(getattr(coeffs, "mexican_surround_sigma_end", 1.75))
            sg0 = float(getattr(coeffs, "mexican_surround_gain_start", 1.0))
            sg1 = float(getattr(coeffs, "mexican_surround_gain_end", 1.15))
            sc = sc0 + (sc1 - sc0) * p
            ss = ss0 + (ss1 - ss0) * p
            sg = sg0 + (sg1 - sg0) * p
            self.mexican_kernel.copy_(
                _mexican_hat_kernel2d(ksz, sc, ss, sg, dev).view(1, 1, ksz, ksz)
            )

    def _extract_patches(self, x_map: torch.Tensor) -> torch.Tensor:
        # x_map: [B, C, H, W] -> patches: [B, N, in_dim]
        x_pad = F.pad(x_map, [self.pad] * 4, mode="reflect")
        patches = F.unfold(x_pad, kernel_size=self.rf_size)  # [B, in_dim, N]
        return patches.transpose(1, 2)

    def _mask_rf_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Apply binary {0,1} RF mask (Gaussian picks which sites are 1) to patch vectors."""
        m = self.rf_connectivity_mask.to(device=patches.device, dtype=patches.dtype)
        return patches * m.view(1, 1, -1)

    def _apply_inhibition_dropout(self, inhibition: torch.Tensor) -> torch.Tensor:
        if self.training and self.inhibition_dropout > 0.0:
            keep_prob = 1.0 - self.inhibition_dropout
            mask = (torch.rand_like(inhibition) < keep_prob).to(inhibition.dtype)
            inhibition = inhibition * mask / max(keep_prob, 1e-8)
        return inhibition

    def _ei_mutual_terms(self, yE: torch.Tensor, yI: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """I->E and E->I lateral drives. Separate L_ei (<=0) / L_ie (>=0) or shared L (legacy)."""
        if self.ei_separate_lateral:
            inh_E = yI @ self.L_ei.t()
            inh_I = yE @ self.L_ie.t()
        else:
            inh_E = yI @ self.L.t()
            inh_I = yE @ self.L.t()
        return inh_E, inh_I

    def _apply_divisive_inhibition_norm(
        self, y: torch.Tensor, B: int, H: int, W: int
    ) -> torch.Tensor:
        """
        Divisive normalization (optional; independent of inhibition strength):
          y <- y / (alpha + beta * denom)
        - global: denom = mean_j |y_ij| per batch item (broadcast)
        - local:  denom_ij = Gaussian spatial pool of |y|
        - both:   denom = w_g * global + w_l * local (needs local kernel)
        """
        if not self.use_divisive_inhibition_norm:
            return y
        abs_y = torch.abs(y)
        k = getattr(self, "divisive_local_kernel", None)
        global_mean = torch.mean(abs_y, dim=1, keepdim=True)  # [B, 1]
        if self.divisive_mode == "both" and k is not None:
            pad = k.size(-1) // 2
            pooled = F.conv2d(abs_y.view(B, 1, H, W), k, padding=pad).view(B, self.N)
            mixed = self.divisive_w_global * global_mean + self.divisive_w_local * pooled
            denom = self.divisive_alpha + self.divisive_beta * mixed
        elif self.divisive_mode == "local" and k is not None:
            pad = k.size(-1) // 2
            pooled = F.conv2d(abs_y.view(B, 1, H, W), k, padding=pad)
            denom = self.divisive_alpha + self.divisive_beta * pooled.view(B, self.N)
        else:
            denom = self.divisive_alpha + self.divisive_beta * global_mean
        return y / (denom + 1e-8)

    def _shape_inhibition_signal(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform raw inhibition drives before applying them to y.

        - inhibition_no_threshold=True: no hard/soft τ cut-off (fully linear in the drive).
          Optional inhibition_smooth_scale>0 applies tanh saturation only (smooth, no zeros).
        - Else: legacy hard threshold or soft_threshold_signed(competition_threshold).
        """
        if self.inhibition_no_threshold:
            if self.inhibition_smooth_scale > 0.0:
                return smooth_inhibition_saturation(z, self.inhibition_smooth_scale)
            return z
        if self.use_soft_inhibition:
            return soft_threshold_signed(z, self.competition_threshold)
        return torch.where(
            torch.abs(z) < self.competition_threshold,
            torch.zeros_like(z),
            z,
        )

    def forward(self, x_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          y_map:   [B, 1, H, W]
          y_flat:  [B, N]
          patches: [B, N, in_dim]
        """
        B, C, H, W = x_map.shape
        if H != self.spatial_size or W != self.spatial_size:
            raise ValueError(f"{self.name} expects {self.spatial_size}x{self.spatial_size}, got {H}x{W}")
        if C != self.in_channels:
            raise ValueError(f"{self.name} expects {self.in_channels} channels, got {C}")

        # Phase bookkeeping for oscillatory inhibition.
        if self.use_oscillatory_inhibition:
            with torch.no_grad():
                self.phase_t += 1

        patches = self._mask_rf_patches(self._extract_patches(x_map))  # [B, N, in_dim]
        y = torch.einsum("bni,ni->bn", patches, self.W) + self.b.unsqueeze(0)  # [B,N]

        # Activation (requested): LayerNorm -> L2 normalization
        y = self.layer_norm(y)
        y = F.normalize(y, p=2, dim=1)

        # Optional EI split + mutual inhibition (only when inhibition_mode is global or mixed).
        # E = relu(y), I = relu(-y), then L_ei / L_ie (or shared L) mutual coupling.
        yE = None
        yI = None
        if self.use_ei_neurons and self.ei_mutual_inhibition and self.inhibition_mode in ("global", "mixed"):
            yE = F.relu(y)
            yI = F.relu(-y)
            yE = yE / (yE.norm(dim=1, keepdim=True) + 1e-8)
            yI = yI / (yI.norm(dim=1, keepdim=True) + 1e-8)
            y = yE

        # Competitive inhibition: global, kernel, averaging, or mixed combination
        if self.current_inhibition_strength > 0:
            if yI is not None:
                # Mutual learned inhibition: shared L (legacy) or L_ei (I->E) + L_ie (E->I).
                inh_E, inh_I = self._ei_mutual_terms(yE, yI)
                inh_E = self._shape_inhibition_signal(inh_E)
                inh_E = self._apply_inhibition_dropout(inh_E)
                yE = yE + self.current_inhibition_strength * inh_E

                inh_I = self._shape_inhibition_signal(inh_I)
                inh_I = self._apply_inhibition_dropout(inh_I)
                yI = yI + self.current_inhibition_strength * inh_I

                yE = yE / (yE.norm(dim=1, keepdim=True) + 1e-8)
                yI = yI / (yI.norm(dim=1, keepdim=True) + 1e-8)
                y = yE
            if self.inhibition_mode == "mixed":
                y_map0 = y.view(B, 1, H, W)
                inh_kernel = F.conv2d(
                    y_map0,
                    self.inhibition_kernel,
                    padding=self.inhibition_kernel.size(-1) // 2,
                ).view(B, self.N)
                inh_avg = F.conv2d(
                    y_map0,
                    self.averaging_inhibition_kernel,
                    padding=self.averaging_inhibition_kernel.size(-1) // 2,
                ).view(B, self.N)
                inh_global = y @ self.L.t()
                inh_kernel = self._shape_inhibition_signal(inh_kernel)
                inh_avg = self._shape_inhibition_signal(inh_avg)
                inh_global = self._shape_inhibition_signal(inh_global)
                inh_kernel = self._apply_inhibition_dropout(inh_kernel)
                inh_avg = self._apply_inhibition_dropout(inh_avg)
                inh_global = self._apply_inhibition_dropout(inh_global)
                y = y + self.current_inhibition_strength * (
                    self.mixed_inh_w_global * inh_global
                    - self.mixed_inh_w_kernel * inh_kernel
                    - self.mixed_inh_w_averaging * inh_avg
                )
            elif self.inhibition_mode == "kernel":
                y_map0 = y.view(B, 1, H, W)
                inh = F.conv2d(y_map0, self.inhibition_kernel, padding=self.inhibition_kernel.size(-1) // 2)
                inhibition = inh.view(B, self.N)
                inhibition = self._shape_inhibition_signal(inhibition)
                inhibition = self._apply_inhibition_dropout(inhibition)
                # Kernel inhibition is positive local energy, subtract to suppress activity.
                y = y - self.current_inhibition_strength * inhibition
            elif self.inhibition_mode == "mexican":
                y_map0 = y.view(B, 1, H, W)
                mk = self.mexican_kernel
                inh = F.conv2d(y_map0, mk, padding=mk.size(-1) // 2)
                inhibition = inh.view(B, self.N)
                inhibition = self._shape_inhibition_signal(inhibition)
                inhibition = self._apply_inhibition_dropout(inhibition)
                y = y - self.current_inhibition_strength * inhibition
            elif self.inhibition_mode == "averaging":
                y_map0 = y.view(B, 1, H, W)
                inh = F.conv2d(
                    y_map0,
                    self.averaging_inhibition_kernel,
                    padding=self.averaging_inhibition_kernel.size(-1) // 2,
                )
                inhibition = inh.view(B, self.N)
                inhibition = self._shape_inhibition_signal(inhibition)
                inhibition = self._apply_inhibition_dropout(inhibition)
                y = y - self.current_inhibition_strength * inhibition
            elif self.inhibition_mode == "predictive":
                # Predictive inhibition: inhibit predictable components based on generator reconstruction.
                with torch.no_grad():
                    y_mean = y.mean(dim=0)  # [N]
                    x_hat = self.G.detach() @ y_mean  # [in_dim]
                    pred_back = self.G.detach().t() @ x_hat  # [N]
                    predictive_drive = pred_back.abs().unsqueeze(0).expand(B, self.N)  # [B,N]
                inhibition = self._shape_inhibition_signal(predictive_drive)
                inhibition = self._apply_inhibition_dropout(inhibition)
                y = y - self.current_inhibition_strength * inhibition
            else:
                # Global learned L; L is negative so inhibition is negative and adding suppresses y.
                inhibition = y @ self.L.t()
                inhibition = self._shape_inhibition_signal(inhibition)
                inhibition = self._apply_inhibition_dropout(inhibition)
                y = y + self.current_inhibition_strength * inhibition

        if self.use_inhibition_softmax and self._softmax_competition_temp > 0:
            t = self._softmax_competition_temp
            if yI is not None:
                scaleE = yE.norm(dim=1, keepdim=True).clamp(min=1e-8)
                scaleI = yI.norm(dim=1, keepdim=True).clamp(min=1e-8)
                yE = F.softmax(yE / t, dim=1) * scaleE
                yI = F.softmax(yI / t, dim=1) * scaleI
                y = yE
            else:
                scale = y.norm(dim=1, keepdim=True).clamp(min=1e-8)
                y = F.softmax(y / t, dim=1) * scale

        if self.use_divisive_inhibition_norm:
            y = self._apply_divisive_inhibition_norm(y, B, H, W)
            if yI is not None:
                yI = self._apply_divisive_inhibition_norm(yI, B, H, W)

        # Recursive refinement (attractor)
        y0 = y
        if self.use_recursive and self.recursive_iters > 0:
            yt = y
            ytI = yI if yI is not None else None
            for _ in range(self.recursive_iters):
                if self.current_inhibition_strength > 0:
                    if self.inhibition_mode == "mixed":
                        yt_map = yt.view(B, 1, H, W)
                        inh_kernel = F.conv2d(
                            yt_map,
                            self.inhibition_kernel,
                            padding=self.inhibition_kernel.size(-1) // 2,
                        ).view(B, self.N)
                        inh_avg = F.conv2d(
                            yt_map,
                            self.averaging_inhibition_kernel,
                            padding=self.averaging_inhibition_kernel.size(-1) // 2,
                        ).view(B, self.N)
                        inh_global = yt @ self.L.t()
                        inh_kernel = self._shape_inhibition_signal(inh_kernel)
                        inh_avg = self._shape_inhibition_signal(inh_avg)
                        inh_global = self._shape_inhibition_signal(inh_global)
                        inh_kernel = self._apply_inhibition_dropout(inh_kernel)
                        inh_avg = self._apply_inhibition_dropout(inh_avg)
                        inh_global = self._apply_inhibition_dropout(inh_global)
                        yt = yt + self.current_inhibition_strength * (
                            self.mixed_inh_w_global * inh_global
                            - self.mixed_inh_w_kernel * inh_kernel
                            - self.mixed_inh_w_averaging * inh_avg
                        )
                        if ytI is not None:
                            # Mutual EI inhibition in mixed branch (global part via L, plus kernel/avg inhibition).
                            inh_E, inh_I = self._ei_mutual_terms(yt, ytI)
                            inh_E = self._shape_inhibition_signal(inh_E)
                            inh_E = self._apply_inhibition_dropout(inh_E)
                            yt = yt + self.current_inhibition_strength * inh_E

                            inh_I = self._shape_inhibition_signal(inh_I)
                            inh_I = self._apply_inhibition_dropout(inh_I)
                            ytI = ytI + self.current_inhibition_strength * inh_I
                    elif self.inhibition_mode == "kernel":
                        yt_map = yt.view(B, 1, H, W)
                        inh = F.conv2d(yt_map, self.inhibition_kernel, padding=self.inhibition_kernel.size(-1) // 2)
                        inhibition = inh.view(B, self.N)
                        inhibition = self._shape_inhibition_signal(inhibition)
                        inhibition = self._apply_inhibition_dropout(inhibition)
                        yt = yt - self.current_inhibition_strength * inhibition
                    elif self.inhibition_mode == "mexican":
                        yt_map = yt.view(B, 1, H, W)
                        mk = self.mexican_kernel
                        inh = F.conv2d(yt_map, mk, padding=mk.size(-1) // 2)
                        inhibition = inh.view(B, self.N)
                        inhibition = self._shape_inhibition_signal(inhibition)
                        inhibition = self._apply_inhibition_dropout(inhibition)
                        yt = yt - self.current_inhibition_strength * inhibition
                    elif self.inhibition_mode == "averaging":
                        yt_map = yt.view(B, 1, H, W)
                        inh = F.conv2d(
                            yt_map,
                            self.averaging_inhibition_kernel,
                            padding=self.averaging_inhibition_kernel.size(-1) // 2,
                        )
                        inhibition = inh.view(B, self.N)
                        inhibition = self._shape_inhibition_signal(inhibition)
                        inhibition = self._apply_inhibition_dropout(inhibition)
                        yt = yt - self.current_inhibition_strength * inhibition
                    elif self.inhibition_mode == "predictive":
                        # Predictive inhibition during recursion: recompute predictability from current yt.
                        with torch.no_grad():
                            y_mean_t = yt.mean(dim=0)  # [N]
                            x_hat_t = self.G.detach() @ y_mean_t  # [in_dim]
                            pred_back_t = self.G.detach().t() @ x_hat_t  # [N]
                            predictive_drive_t = pred_back_t.abs().unsqueeze(0).expand(B, self.N)  # [B,N]
                        inhibition = self._shape_inhibition_signal(predictive_drive_t)
                        inhibition = self._apply_inhibition_dropout(inhibition)
                        yt = yt - self.current_inhibition_strength * inhibition
                    else:
                        if ytI is not None:
                            # Mutual EI inhibition in global learned-L branch.
                            inh_E, inh_I = self._ei_mutual_terms(yt, ytI)
                            inh_E = self._shape_inhibition_signal(inh_E)
                            inh_E = self._apply_inhibition_dropout(inh_E)
                            yt = yt + self.current_inhibition_strength * inh_E

                            inh_I = self._shape_inhibition_signal(inh_I)
                            inh_I = self._apply_inhibition_dropout(inh_I)
                            ytI = ytI + self.current_inhibition_strength * inh_I
                        else:
                            inhibition = yt @ self.L.t()
                            inhibition = self._shape_inhibition_signal(inhibition)
                            inhibition = self._apply_inhibition_dropout(inhibition)
                            yt = yt + self.current_inhibition_strength * inhibition
                # Match prior dynamics: re-apply divisive only when inhibition refines yt (avoid N× divisive when strength=0)
                if self.use_divisive_inhibition_norm and self.current_inhibition_strength > 0:
                    yt = self._apply_divisive_inhibition_norm(yt, B, H, W)
                    if ytI is not None:
                        ytI = self._apply_divisive_inhibition_norm(ytI, B, H, W)
            y = yt
            if ytI is not None:
                yI = ytI
        self._last_y0 = y0.detach()
        self._last_yT = y.detach()

        if (
            self.use_ei_neurons
            and self.ei_mutual_inhibition
            and self.inhibition_mode in ("global", "mixed")
            and yI is not None
        ):
            self._cache_yE_lateral = y.detach()
            self._cache_yI_lateral = yI.detach()
        else:
            self._cache_yE_lateral = None
            self._cache_yI_lateral = None

        if self.training and self.adaptive_inhibition:
            with torch.no_grad():
                active_frac = (y > 0).float().mean().item()
                delta = self.inhibition_adapt_lr * (active_frac - self.target_active_frac)
                self.current_inhibition_strength = float(
                    max(self.inhibition_min, min(self.inhibition_max, self.current_inhibition_strength + delta))
                )

        # Global WTA with sparsity (paper can be updated to match this)
        # wta_sparsity = fraction to zero out. keep_k = ceil((1 - sparsity) * N)
        if self.wta_sparsity > 0.0:
            keep_frac = max(0.0, min(1.0, 1.0 - self.wta_sparsity))
            keep_k = int(math.ceil(keep_frac * self.N))
            keep_k = max(1, min(self.N, keep_k))
            phase_gate = None
            if self.use_oscillatory_inhibition:
                # Phase gate in [0,1] computed from each neuron's preferred phase.
                with torch.no_grad():
                    t = float(self.phase_t.item())
                    omega = (2.0 * math.pi) / float(self.phase_period)
                    phase_gate = 0.5 * (1.0 + torch.cos(omega * t + self.phase_pref))  # [N]
                    if self.phase_gate_sharpness != 1.0:
                        phase_gate = phase_gate.pow(self.phase_gate_sharpness)
                    phase_gate = phase_gate.unsqueeze(0)  # [1,N]
            if self.use_homeostatic_threshold:
                # Adaptive k-winner gating:
                #   scores = y - theta_i
                #   mask selects top-k scores per sample, then applies it to y.
                scores = y - self.winner_theta.unsqueeze(0)
                if phase_gate is not None:
                    scores = scores * phase_gate.to(dtype=scores.dtype)
                kth_s = (
                    torch.topk(scores, k=keep_k, dim=1, largest=True, sorted=False)
                    .values.min(dim=1, keepdim=True)
                    .values
                )
                mask = scores >= kth_s
                if self.training:
                    # Homeostatic update: if neuron wins too often => increase theta, else decrease.
                    r_i = mask.float().mean(dim=0)  # [N] win frequency in the batch (keep float32 for AMP safety)
                    r_target = float(keep_k) / float(self.N)
                    delta = self.homeostatic_threshold_lr * (r_i - r_target)
                    self.winner_theta.add_(delta)
                    self.winner_theta.clamp_(self.homeostatic_threshold_min, self.homeostatic_threshold_max)
                y = y * mask.to(y.dtype)
            else:
                if phase_gate is not None:
                    y = y * phase_gate.to(dtype=y.dtype)
                # Rank by |y| so signed LayerNorm codes still pick salient units (nonnegative path unchanged).
                mag = y.abs()
                kth = torch.topk(mag, k=keep_k, dim=1, largest=True, sorted=False).values.min(
                    dim=1, keepdim=True
                ).values
                y = y * (mag >= kth).to(y.dtype)

        y_map = y.view(B, 1, H, W)
        return y_map, y, patches

    def update_from_patches(
        self,
        patches: torch.Tensor,
        y_flat: torch.Tensor,
        variance_scale: Union[float, torch.Tensor] = 1.0,
        inhibition_scale: float = 1.0,
        layer_spatial_gain: Optional[torch.Tensor] = None,
        pfc_spatial_gain: Optional[torch.Tensor] = None,
    ):
        """
        patches: [B, N, in_dim]
        y_flat:  [B, N]
        variance_scale: scalar or length-[N] tensor (per-neuron top-down × predictive × glia when enabled).
        layer_spatial_gain: optional [N] from this layer/stream's batch-mean |y| (independent per layer).
        pfc_spatial_gain: optional [N] from PFC TE state for this frequency band.
        """
        B = y_flat.size(0)
        inh_scale = float(max(0.0, inhibition_scale))
        # With mutual EI, use cached excitatory map for plasticity (matches L/L_ei dynamics).
        y_plas = y_flat
        cye = self._cache_yE_lateral
        if cye is not None and cye.shape == y_flat.shape:
            y_plas = cye.to(device=y_flat.device, dtype=y_flat.dtype)
        with torch.no_grad():
            self.update_counter += 1
            self.last_fe_signal_per_neuron.zero_()
            # Top-down control also steers effective inhibition strength.
            target_inh = float(self.inhibition_strength) * inh_scale
            blended = 0.9 * float(self.current_inhibition_strength) + 0.1 * target_inh
            self.current_inhibition_strength = float(
                max(self.inhibition_min, min(self.inhibition_max, blended))
            )

        patches = self._mask_rf_patches(patches)

        y_mean = y_plas.mean(dim=0)              # [N]
        # Match reference scaling more closely: use SUM over batch (their hebb uses x summed over batch).
        patch_sum = patches.sum(dim=0)           # [N, in_dim]
        patch_mean = patch_sum / float(B)        # [N, in_dim] (keep mean for places that need it)

        use_pc = bool(getattr(self.coeffs, "pc_per_neuron_plasticity", True))
        trace_prev: Optional[torch.Tensor] = None
        with torch.no_grad():
            if use_pc and self.beta_trace > 0.0 and self.trace.numel() == self.N:
                trace_prev = self.trace.detach().clone()

        # Trace
        with torch.no_grad():
            if self.beta_trace > 0.0:
                self.trace.mul_(1 - self.beta_trace).add_(self.beta_trace * y_mean)

        # Active inference on y_mean (optional)
        if self.use_active_inference and self.use_free_energy and self.active_inference_steps > 0:
            with torch.enable_grad():
                z = y_mean.detach().clone().requires_grad_(True)
                x_target = patch_mean.mean(dim=0).detach()  # [in_dim]
                for _ in range(self.active_inference_steps):
                    x_hat = self.G.detach() @ z
                    F_energy = (x_target - x_hat).pow(2).mean() + 1e-3 * z.pow(2).mean()
                    (grad,) = torch.autograd.grad(F_energy, z, retain_graph=False, create_graph=False)
                    z = (z - self.active_inference_lr * grad).detach().requires_grad_(True)
                y_eff = z.detach()
        else:
            y_eff = y_mean.detach() if self.beta_trace <= 0.0 else self.trace.detach()

        # Build per-neuron unsupervised term updates, then mix them with a configurable strategy.
        term_updates: Dict[str, torch.Tensor] = {}

        def _add_term(name: str, value: torch.Tensor) -> None:
            if name in term_updates:
                term_updates[name] = term_updates[name] + value
            else:
                term_updates[name] = value

        # Optional wavelet denoising: soft-threshold in 1D Haar before Hebbian (sparser, stabler updates)
        patch_for_hebb = patch_sum
        if self.use_wavelet_denoise and getattr(self.coeffs, "use_wavelet_denoise", True):
            tau = getattr(self.coeffs, "wavelet_threshold", 0.02)
            if tau > 0 and self.in_dim >= 2:
                D = self.in_dim + (self.in_dim % 2)
                pad_width = D - self.in_dim
                pm = F.pad(patch_mean, (0, pad_width), mode="reflect")
                low, high = haar1d_one_level(pm)
                low = soft_threshold(low, tau)
                high = soft_threshold(high, tau)
                rec = inverse_haar1d_one_level(low, high)
                patch_denoised = rec[..., : self.in_dim]
                patch_for_hebb = patch_denoised * float(B)

        if self.use_hebbian:
            _add_term("hebb", y_eff.unsqueeze(1) * patch_for_hebb)

        if int(self.update_counter.item()) % max(1, self.holo_update_freq) == 0:
            if self.use_holographic:
                y_embed = (self.P_holo @ y_eff).contiguous()
                rho = float(getattr(self.coeffs, "holo_corr_blend", 0.2))
                conv = _fft_holographic_binding(patch_mean, y_embed, corr_blend=rho, in_dim=self.in_dim)
                holo_term = y_eff.unsqueeze(1) * conv
                _add_term("holo", self.coeffs.alpha * holo_term)
                with torch.no_grad():
                    self.M_holo_fast.mul_(1.0 - self.holo_fast_decay)
                    self.M_holo_fast.add_(self.holo_fast_lr * holo_term)
                    cap = float(getattr(self.coeffs, "holo_fast_norm_cap", 12.0))
                    if cap > 0.0:
                        n = self.M_holo_fast.norm()
                        if torch.isfinite(n) and float(n) > cap:
                            self.M_holo_fast.mul_(cap / (n + 1e-8))
                _add_term("holo", self.coeffs.alpha * self.M_holo_fast)

            if self.use_hyperbolic_binding:
                beta_hyp = getattr(self.coeffs, "beta_hyp", 0.01)
                x_hyp = hyp_ops.project_to_ball(patch_mean)
                y_scale = torch.abs(y_eff).unsqueeze(1)
                y_modulated = y_scale * patch_mean
                y_hyp = hyp_ops.project_to_ball(y_modulated)
                binding_hyp = hyp_ops.mobius_add(x_hyp, y_hyp)
                binding_tang = hyp_ops.from_poincare(binding_hyp)
                binding_tang = torch.nan_to_num(binding_tang, nan=0.0, posinf=0.0, neginf=0.0)
                _add_term("hyp", beta_hyp * (y_eff.unsqueeze(1) * binding_tang))

        if self.use_distance_gradient and getattr(self.coeffs, "distance_gradient_lr", 0.0) > 0:
            x_hyp = hyp_ops.to_poincare(patch_mean)
            w_hyp = hyp_ops.to_poincare(self.W)
            grad = hyp_ops.distance_gradient_wrt_w(
                x_hyp.unsqueeze(0),
                w_hyp,
                chunk_size=self.gradient_chunk_size,
            )
            grad = grad.mean(dim=0)
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            _add_term("dist", -self.coeffs.distance_gradient_lr * grad)

        # Wavelet-domain binding: bind patch with y in 1D Haar (scale-aligned neural binding)
        if self.use_wavelet_binding and getattr(self.coeffs, "use_wavelet_binding", True):
            gamma_w = getattr(self.coeffs, "gamma_wavelet", 0.01)
            if gamma_w > 0 and self.in_dim >= 2:
                D = self.in_dim + (self.in_dim % 2)
                pad_w = D - self.in_dim
                pm = F.pad(patch_mean, (0, pad_w), mode="reflect")
                y_embed = (self.P_holo @ y_eff).contiguous()
                y_embed = y_embed / (y_embed.norm() + 1e-8)
                # F.pad reflect needs 2D+; pad as (1, D) then squeeze
                ye = F.pad(y_embed.unsqueeze(0), (0, pad_w), mode="reflect").squeeze(0)
                low_p, high_p = haar1d_one_level(pm)
                low_y, high_y = haar1d_one_level(ye)
                bind_low = wavelet_circular_conv_1d(low_p, low_y)
                bind_high = wavelet_circular_conv_1d(high_p, high_y)
                binding = inverse_haar1d_one_level(bind_low, bind_high)[..., : self.in_dim]
                binding = torch.nan_to_num(binding, nan=0.0, posinf=0.0, neginf=0.0)
                _add_term("wave", gamma_w * (y_eff.unsqueeze(1) * binding))

        # Anti-Hebbian decorrelation on feedforward weights (match ConsistencyLearning5):
        #   w_flat: [N, in_dim]
        #   sim:    [N, N] (cosine similarity between neuron weight vectors)
        #   anti:   sim @ w_flat
        if self.use_antihebbian:
            with torch.no_grad():
                w_flat = self.W
                w_n = F.normalize(w_flat, p=2, dim=1)
                sim = w_n @ w_n.t()
                sim.fill_diagonal_(0.0)
                anti_hebb = sim @ w_flat
            _add_term("anti", -self.coeffs.lambda_a * anti_hebb)

        if self.use_consistency and self.prev_y.numel() == self.N:
            diff = (y_mean - self.prev_y)  # [N]
            _add_term("cons", -self.coeffs.lambda_c * diff.unsqueeze(1) * patch_sum)

        if self.use_recursive and self._last_y0 is not None and self._last_yT is not None:
            dy = (self._last_yT.mean(dim=0) - self._last_y0.mean(dim=0))  # [N]
            _add_term("rec", self.coeffs.lambda_r * dy.unsqueeze(1) * patch_sum)

        if self.use_free_energy:
            # Predict a global patch vector from y_mean (predictive coding)
            x_target = patch_mean.mean(dim=0)  # [in_dim]
            x_hat = self.G @ y_mean            # [in_dim]
            err = (x_target - x_hat)           # [in_dim]
            with torch.no_grad():
                self.G.add_(0.001 * torch.ger(err, y_mean))
                # ∇_y 0.5||x - G y||^2 = -G^T err — magnitude per neuron for shared energy with PFC top-down.
                self.last_fe_signal_per_neuron.copy_((self.G.t() @ err).abs())
            _add_term("free", self.coeffs.lambda_F * (y_mean.unsqueeze(1) * err.unsqueeze(0)))

        # Weight decay (optionally scaled by activation entropy)
        lambda_d = self.coeffs.lambda_d
        if getattr(self.coeffs, "use_entropy_plasticity_decay", False) and lambda_d > 0:
            norm_entropy = _normalized_entropy_activations(y_plas)
            scale = 1.0 + getattr(self.coeffs, "entropy_decay_scale", 0.5) * norm_entropy
            lambda_d = lambda_d * scale
        _add_term("decay", -lambda_d * self.W)

        # ---------------------------
        # Mixture of unsupervised terms
        # ---------------------------
        dW = torch.zeros_like(self.W)
        if term_updates:
            mode = str(getattr(self.coeffs, "unsup_mix_mode", "adaptive")).lower()
            normalize_terms = bool(getattr(self.coeffs, "unsup_mix_normalize_terms", True))
            temperature = float(max(1e-6, getattr(self.coeffs, "unsup_mix_temperature", 1.0)))
            random_alpha = float(max(1e-6, getattr(self.coeffs, "unsup_mix_random_alpha", 20.0)))

            base_weight_map = {
                "hebb": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_hebb", 0.30))),
                "holo": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_holo", 0.10))),
                "hyp": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_hyp", 0.10))),
                "wave": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_wave", 0.15))),
                "anti": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_anti", 0.10))),
                "cons": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_cons", 0.08))),
                "rec": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_rec", 0.08))),
                "free": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_free", 0.06))),
                "decay": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_decay", 0.03))),
                "dist": float(max(0.0, getattr(self.coeffs, "unsup_mix_w_dist", 0.00))),
            }

            names = list(term_updates.keys())
            eps = 1e-8
            norms = torch.tensor(
                [float(term_updates[n].norm().detach().cpu().item()) for n in names],
                device=self.W.device,
                dtype=self.W.dtype,
            )

            if normalize_terms:
                for n in names:
                    term = term_updates[n]
                    term_updates[n] = term / (term.norm() + eps)

            base = torch.tensor([base_weight_map.get(n, 0.0) for n in names], device=self.W.device, dtype=self.W.dtype)
            if float(base.sum().item()) <= 0.0:
                base = torch.ones_like(base) / float(max(1, base.numel()))
            else:
                base = base / (base.sum() + eps)

            if mode == "fixed":
                mix_w = base
            elif mode == "random":
                alpha = torch.clamp(random_alpha * base, min=1e-3)
                mix_w = torch.distributions.Dirichlet(alpha.float()).sample().to(self.W.dtype)
            else:
                # Smart/adaptive weighting: base prior + inverse-norm balancing.
                # Large terms get down-weighted automatically to avoid domination.
                inv_norm = 1.0 / (norms + eps)
                inv_norm = inv_norm / (inv_norm.sum() + eps)
                score = torch.log(base + eps) + temperature * torch.log(inv_norm + eps)
                mix_w = torch.softmax(score, dim=0)

            for i, n in enumerate(names):
                dW.add_(mix_w[i] * term_updates[n])

        # Apply update with stability clamps
        with torch.no_grad():
            dW = torch.nan_to_num(dW, nan=0.0, posinf=0.0, neginf=0.0)
            dW_norm = dW.norm()
            if torch.isfinite(dW_norm) and dW_norm > 1.0:
                dW.mul_(1.0 / (dW_norm + 1e-8))
            Nloc = self.N
            dev, dt = self.W.device, self.W.dtype
            if isinstance(variance_scale, torch.Tensor):
                vs_t = variance_scale.to(device=dev, dtype=dt).flatten()
                if vs_t.numel() == Nloc:
                    vs_vec = vs_t.clamp(min=0.0)
                elif vs_t.numel() == 1:
                    vs_vec = torch.full((Nloc,), float(vs_t.item()), device=dev, dtype=dt).clamp(min=0.0)
                else:
                    vs_vec = torch.full(
                        (Nloc,), float(vs_t.mean().item()), device=dev, dtype=dt
                    ).clamp(min=0.0)
            else:
                vs_vec = None
                vs_f = float(max(0.0, variance_scale))
            if use_pc:
                gmn = float(getattr(self.coeffs, "pc_per_neuron_gain_min", 0.5))
                gmx = float(getattr(self.coeffs, "pc_per_neuron_gain_max", 1.5))
                if trace_prev is not None and trace_prev.numel() == Nloc:
                    local_g = (y_mean - trace_prev).abs()
                else:
                    local_g = (y_mean - y_mean.mean()).abs()
                local_g = local_g / (local_g.mean() + 1e-8)
                local_g = torch.clamp(local_g, gmn, gmx)
                layer_g = layer_spatial_gain
                pfc = pfc_spatial_gain
                layer_ok = layer_g is not None and layer_g.numel() == Nloc
                pfc_ok = pfc is not None and pfc.numel() == Nloc
                wl = float(getattr(self.coeffs, "pc_per_neuron_layer_weight", 0.34))
                wp = float(getattr(self.coeffs, "pc_per_neuron_pfc_weight", 0.33))
                wt = float(getattr(self.coeffs, "pc_per_neuron_trace_weight", 0.33))
                devg, dtg = local_g.device, local_g.dtype
                acc = torch.zeros(Nloc, device=devg, dtype=dtg)
                wsum = 0.0
                if wt > 0.0:
                    acc = acc + wt * local_g
                    wsum += wt
                if layer_ok and wl > 0.0:
                    acc = acc + wl * layer_g.to(device=devg, dtype=dtg)
                    wsum += wl
                if pfc_ok and wp > 0.0:
                    acc = acc + wp * pfc.to(device=devg, dtype=dtg)
                    wsum += wp
                if wsum > 0.0:
                    eff = acc / float(wsum)
                else:
                    eff = local_g
                eff = eff / (eff.mean() + 1e-8)
                if vs_vec is not None:
                    row_scale = vs_vec * eff
                else:
                    row_scale = vs_f * eff
            else:
                if vs_vec is not None:
                    row_scale = vs_vec * torch.ones(Nloc, device=dev, dtype=dt)
                else:
                    row_scale = vs_f * torch.ones(Nloc, device=dev, dtype=dt)
            self.W.add_(self.coeffs.eta * row_scale.unsqueeze(1) * dW)
            self.W.clamp_(-self.w_clip, self.w_clip)

            # Structural plasticity schedule: prune weak synapses, regrow useful ones.
            if (
                bool(getattr(self.coeffs, "use_structural_plasticity", False))
                and int(self.update_counter.item()) % max(1, int(getattr(self.coeffs, "structural_update_freq", 200))) == 0
            ):
                total_syn = self.W.numel()
                prune_max = int(max(0, round(float(getattr(self.coeffs, "structural_prune_max_frac", 0.001)) * total_syn)))
                grow_max = int(max(0, round(float(getattr(self.coeffs, "structural_grow_max_frac", 0.001)) * total_syn)))
                prune_thr = float(max(0.0, getattr(self.coeffs, "structural_prune_threshold", 1e-4)))
                grow_thr = float(max(0.0, getattr(self.coeffs, "structural_grow_threshold", 0.02)))
                grow_init = float(max(0.0, getattr(self.coeffs, "structural_grow_init_scale", 0.01)))

                # Prune near-zero weights (bounded by prune_max for stability).
                if prune_max > 0:
                    abs_w = torch.abs(self.W)
                    prune_candidates = (abs_w < prune_thr) & (self.W != 0)
                    cand_idx = torch.nonzero(prune_candidates.view(-1), as_tuple=False).squeeze(1)
                    if cand_idx.numel() > 0:
                        if cand_idx.numel() > prune_max:
                            cand_vals = abs_w.view(-1)[cand_idx]
                            sel_local = torch.topk(cand_vals, k=prune_max, largest=False).indices
                            sel_idx = cand_idx[sel_local]
                        else:
                            sel_idx = cand_idx
                        self.W.view(-1)[sel_idx] = 0.0

                # Grow new synapses where Hebbian drive is high (bounded by grow_max).
                if grow_max > 0 and grow_init > 0.0:
                    hebb_drive = torch.abs(y_eff.unsqueeze(1) * patch_mean)
                    grow_candidates = (self.W == 0) & (hebb_drive > grow_thr)
                    cand_idx = torch.nonzero(grow_candidates.view(-1), as_tuple=False).squeeze(1)
                    if cand_idx.numel() > 0:
                        if cand_idx.numel() > grow_max:
                            cand_vals = hebb_drive.view(-1)[cand_idx]
                            sel_local = torch.topk(cand_vals, k=grow_max, largest=True).indices
                            sel_idx = cand_idx[sel_local]
                        else:
                            sel_idx = cand_idx
                        grow_values = grow_init * torch.sign((y_eff.unsqueeze(1) * patch_mean).view(-1)[sel_idx])
                        self.W.view(-1)[sel_idx] = grow_values

                self.W.clamp_(-self.w_clip, self.w_clip)
            # No post-update weight normalization (per user request)

        # Lateral inhibition learning for modes that use learned global L
        # (global and mixed; kernel/averaging-only use fixed local kernels).
        if (
            self.inhibition_mode in ("global", "mixed")
            and self.use_antihebbian
            and (int(self.update_counter.item()) % self.lateral_update_freq == 0)
        ):
            with torch.no_grad():
                # Match reference: raw outer product is NOT averaged by B.
                corr = y_plas.t() @ y_plas  # [N,N]
                corr.fill_diagonal_(0.0)
                # Centered covariance over batch (same sum convention as corr).
                y_mean_b = y_plas.mean(dim=0, keepdim=True)  # [1,N]
                yc = y_plas - y_mean_b
                cov_mat = yc.t() @ yc
                cov_mat.fill_diagonal_(0.0)
                # Holographic-inspired associative term on centered neuron activity.
                # Uses cosine-normalized co-activity across the batch to keep scale stable.
                yc_norm = F.normalize(yc, p=2, dim=0, eps=1e-8)
                holo_assoc = yc_norm.t() @ yc_norm
                holo_assoc.fill_diagonal_(0.0)
                # Hyperbolic-inspired lateral term: Poincaré embed each sample's centered map (row of yc),
                # then Gram structure (ambient inner product after exp-map), zero diagonal.
                hyp_assoc = torch.zeros_like(corr)
                if self.lateral_w_hyp > 0.0:
                    y_rows = F.normalize(yc.float(), p=2, dim=1, eps=1e-8)
                    hyp_emb = hyp_ops.to_poincare(y_rows)
                    hyp_emb = torch.nan_to_num(hyp_emb, nan=0.0, posinf=0.0, neginf=0.0)
                    hyp_assoc = hyp_emb.transpose(0, 1) @ hyp_emb
                    hyp_assoc = hyp_assoc.to(dtype=corr.dtype)
                    hyp_assoc.fill_diagonal_(0.0)
                # Wavelet-inspired lateral term: 1D Haar low-pass along batch index per neuron (column of yc),
                # cosine-normalized low bands, then neuron×neuron Gram; captures slow co-fluctuation across batch.
                wave_assoc = torch.zeros_like(corr)
                if self.lateral_w_wave > 0.0:
                    wt = yc.transpose(0, 1).contiguous().float()  # [N, B]
                    if wt.size(1) >= 2:
                        if wt.size(1) % 2 == 1:
                            wt = F.pad(wt, (0, 1), mode="replicate")
                        low_w, _ = haar1d_one_level(wt)
                        wf = F.normalize(low_w, p=2, dim=1, eps=1e-8)
                        wf = torch.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0)
                        wave_assoc = wf @ wf.transpose(0, 1)
                        wave_assoc = wave_assoc.to(dtype=corr.dtype)
                        wave_assoc.fill_diagonal_(0.0)
                # Oja-like stabilization: scale L by mean squared activity (off-diagonal dynamics).
                y2 = (y_plas.pow(2)).mean(dim=0)  # [N]
                oja_stab = self.L * (y2.unsqueeze(0) + y2.unsqueeze(1)) * 0.5
                oja_stab.fill_diagonal_(0.0)
                anti = -corr
                lateral_drive = (
                    self.lateral_w_hebb * corr
                    + self.lateral_w_anti * anti
                    + self.lateral_w_cov * cov_mat
                    + self.lateral_w_holo * holo_assoc
                    + self.lateral_w_hyp * hyp_assoc
                    + self.lateral_w_wave * wave_assoc
                    - self.inhibition_decay * self.L
                    - self.lateral_w_oja * oja_stab
                )
                lateral_update = (self.lr_lateral * inh_scale) * lateral_drive
                lateral_update = torch.where(
                    torch.abs(lateral_update) < self.competition_threshold,
                    torch.zeros_like(lateral_update),
                    lateral_update,
                )
                if self.inhibition_learning_sparsity > 0.0:
                    keep_frac = max(0.0, min(1.0, 1.0 - self.inhibition_learning_sparsity))
                    if keep_frac <= 0.0:
                        lateral_update.zero_()
                    else:
                        keep_k = int(math.ceil(keep_frac * lateral_update.numel()))
                        keep_k = max(1, min(lateral_update.numel(), keep_k))
                        if keep_k < lateral_update.numel():
                            flat_abs = torch.abs(lateral_update).view(-1)
                            kth = torch.topk(flat_abs, k=keep_k, largest=True, sorted=False).values.min()
                            lateral_update = torch.where(
                                torch.abs(lateral_update) >= kth,
                                lateral_update,
                                torch.zeros_like(lateral_update),
                            )
                self.L.add_(lateral_update)
                self.L.fill_diagonal_(0.0)
                self.L.clamp_(-1.0, 0.0)

                # Separate I->E (L_ei <= 0) and E->I (L_ie >= 0) plasticity from cached E/I maps.
                if (
                    self.ei_separate_lateral
                    and self._cache_yE_lateral is not None
                    and self._cache_yI_lateral is not None
                ):
                    yE = self._cache_yE_lateral
                    yI = self._cache_yI_lateral
                    corr_ie = yI.t() @ yE
                    corr_ie.fill_diagonal_(0.0)
                    corr_ei = yE.t() @ yI
                    corr_ei.fill_diagonal_(0.0)
                    ymb_e = yE.mean(dim=0, keepdim=True)
                    ymb_i = yI.mean(dim=0, keepdim=True)
                    yc_e = yE - ymb_e
                    yc_i = yI - ymb_i
                    cov_ie = yc_i.t() @ yc_e
                    cov_ie.fill_diagonal_(0.0)
                    cov_ei = yc_e.t() @ yc_i
                    cov_ei.fill_diagonal_(0.0)
                    yc_mix = (yc_e + yc_i) * 0.5
                    yc_norm_m = F.normalize(yc_mix, p=2, dim=0, eps=1e-8)
                    holo_m = yc_norm_m.t() @ yc_norm_m
                    holo_m.fill_diagonal_(0.0)
                    hyp_m = torch.zeros_like(corr_ie)
                    if self.lateral_w_hyp > 0.0:
                        y_rows = F.normalize(yc_mix.float(), p=2, dim=1, eps=1e-8)
                        hyp_emb = hyp_ops.to_poincare(y_rows)
                        hyp_emb = torch.nan_to_num(hyp_emb, nan=0.0, posinf=0.0, neginf=0.0)
                        hyp_m = hyp_emb.transpose(0, 1) @ hyp_emb
                        hyp_m = hyp_m.to(dtype=corr_ie.dtype)
                        hyp_m.fill_diagonal_(0.0)
                    wave_m = torch.zeros_like(corr_ie)
                    if self.lateral_w_wave > 0.0:
                        wt = yc_mix.transpose(0, 1).contiguous().float()
                        if wt.size(1) >= 2:
                            if wt.size(1) % 2 == 1:
                                wt = F.pad(wt, (0, 1), mode="replicate")
                            low_w, _ = haar1d_one_level(wt)
                            wf = F.normalize(low_w, p=2, dim=1, eps=1e-8)
                            wf = torch.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0)
                            wave_m = wf @ wf.transpose(0, 1)
                            wave_m = wave_m.to(dtype=corr_ie.dtype)
                            wave_m.fill_diagonal_(0.0)
                    y2e = (yE.pow(2)).mean(dim=0)
                    y2i = (yI.pow(2)).mean(dim=0)
                    oja_ei = self.L_ei * (y2i.unsqueeze(0) + y2e.unsqueeze(1)) * 0.5
                    oja_ei.fill_diagonal_(0.0)
                    oja_ie = self.L_ie * (y2e.unsqueeze(0) + y2i.unsqueeze(1)) * 0.5
                    oja_ie.fill_diagonal_(0.0)
                    drive_ei = (
                        self.lateral_w_hebb * corr_ie
                        + self.lateral_w_anti * (-corr_ie)
                        + self.lateral_w_cov * cov_ie
                        + self.lateral_w_holo * holo_m
                        + self.lateral_w_hyp * hyp_m
                        + self.lateral_w_wave * wave_m
                        - self.inhibition_decay * self.L_ei
                        - self.lateral_w_oja * oja_ei
                    )
                    drive_ie = (
                        self.lateral_w_hebb * corr_ei
                        + self.lateral_w_anti * (-corr_ei)
                        + self.lateral_w_cov * cov_ei
                        + self.lateral_w_holo * holo_m
                        + self.lateral_w_hyp * hyp_m
                        + self.lateral_w_wave * wave_m
                        - self.inhibition_decay * self.L_ie
                        - self.lateral_w_oja * oja_ie
                    )
                    upd_ei = (self.lr_lateral * inh_scale) * drive_ei
                    upd_ie = (self.lr_lateral * inh_scale) * drive_ie
                    upd_ei = torch.where(
                        torch.abs(upd_ei) < self.competition_threshold,
                        torch.zeros_like(upd_ei),
                        upd_ei,
                    )
                    upd_ie = torch.where(
                        torch.abs(upd_ie) < self.competition_threshold,
                        torch.zeros_like(upd_ie),
                        upd_ie,
                    )
                    if self.inhibition_learning_sparsity > 0.0:
                        keep_frac = max(0.0, min(1.0, 1.0 - self.inhibition_learning_sparsity))
                        if keep_frac <= 0.0:
                            upd_ei.zero_()
                            upd_ie.zero_()
                        else:
                            keep_k = int(math.ceil(keep_frac * upd_ei.numel()))
                            keep_k = max(1, min(upd_ei.numel(), keep_k))
                            if keep_k < upd_ei.numel():
                                for u in (upd_ei, upd_ie):
                                    flat_abs = torch.abs(u).view(-1)
                                    kth = torch.topk(flat_abs, k=keep_k, largest=True, sorted=False).values.min()
                                    u.copy_(
                                        torch.where(
                                            torch.abs(u) >= kth,
                                            u,
                                            torch.zeros_like(u),
                                        )
                                    )
                    self.L_ei.add_(upd_ei)
                    self.L_ei.fill_diagonal_(0.0)
                    self.L_ie.add_(upd_ie)
                    self.L_ie.fill_diagonal_(0.0)

        with torch.no_grad():
            self.prev_y = y_mean.detach().clone()


# =============================================================================
# VISNET UNIFIED MODEL (paper-aligned)
# =============================================================================

class VisNetUnified(nn.Module):
    """
    Hierarchical VisNet with per-frequency streams.

    When ``coeffs.cascade_skip_connections`` is True (default in PFC17), L2–L4 use a
    cascade with skip connections: each deeper layer receives channel-concatenated
    maps from earlier stages (multi-channel input to ``TopographicUnifiedLayer``).

    Optional **dorsal stream** (``use_dorsal_stream=True``): after ventral L2, **MT** then **PP**
    (per frequency). MT inputs are **spatially gated** on luminance using a **static** dorsal
    map: centre-surround (difference-of-Gaussians) saliency, a high-frequency **texture**
    residual, and a local **z-score** figure-ground term — not optic flow or Sobel-as-motion.
    **PFC Hopfield** input (dorsal) uses ``pfc_pre_hopfield_fusion``: ``pp_only`` | ``blend`` | ``gate`` | ``all``
    (``all`` = equal mix of PP-only, blend, and gated TE–PP). Fusion gate matrices use **unsupervised**
    LMS-style updates (``use_pfc_fusion_gate_unsup``, ``pfc_fusion_gate_unsup_lr`` / ``decay``), like
    ``pfc_topdown_W`` — no classifier BP.
    Optional **PC layer-output masks**: ``sigmoid(PFC @ W + b)`` per ventral layer (L1–L4) multiplies
    layer flats before readout; LMS trained on prediction error vs max-normalized per-sample |activity|.
    The classifier sees ``concat(TE, second)`` with
    ``pfc_post_readout_fusion``: ``concat`` | ``gate`` | ``all``. With ``pfc_mode="hebbian_sa"``, the PFC block is
    softmax self-attention over memory slots with Hebbian Q/K updates (no backprop on PFC).

    Optional **PFC spatial readout gate** (``pfc_spatial_readout_gate``): after the Hopfield/PFC
    loop, builds a **spatial attention map** from the PFC state (mean absolute activity over frequency
    streams, per location), and **down-weights** TE and PP features at low-saliency locations before
    the linear classifier — a lightweight top-down–biased **figure vs. background** emphasis that
    complements the bottom-up dorsal saliency map earlier in the net.

    Optional **two-step recurrent feedback** (``pfc_recurrent_feedback_steps=2``): on the **last**
    top-down iteration, after Hopfield refines the PFC state, a spatial saliency map gates the
    **L1 frequency inputs** and the full ventral (+ dorsal) stack is run **again**, so higher-level
    state can influence early-layer activations in a second pass (approximation of cortical feedback).

    Optional **dense PFC→L1 feedback** (``use_pfc_dense_feedback``): additive injection of the
    post-Hopfield PFC map (reshaped to ``[B,F,H,W]``, per-location normalized) into **every L1
    channel** before that second stack — an explicit top-down connection from PFC state to early
    layers (no backprop). Can be used alone (second pass with additive only) or combined with
    recurrent multiplicative gating.

    Optional **deep PFC feedback** (``use_pfc_deep_feedback``): on the second stack, a bottleneck
    projection of the post-Hopfield PFC vector yields spatial maps added to **L2, L3, ventral L4**,
    and (with dorsal) **MT** and **PP** outputs (fixed weights, no backprop).

    Optional **vertical symmetry prior** (``use_symmetry_gate_prior``): multiplies the static dorsal
    saliency map by a bilateral-consistency map (left vs. mirrored right on luminance). No
    backprop; optional ``symmetry_prior_unsup_lr`` EMA-updates a scalar scale from batch mean
    symmetry (unsupervised attention gain, not gradient-based).

    Optional **IT↔PP bidirectional interface** (``it_pp_cross_gate``): after TE and PP flats are built,
    each stream is multiplicatively emphasized at locations where the *other* stream is salient
    (spatial maps from mean absolute activity per frequency). No learned query/key matrices;
    optional ``it_pp_cross_iters`` repeats the mutual update (recurrence at the interface).
    """

    def __init__(
        self,
        device: str = "cpu",
        num_classes: int = 10,
        spatial_size: int = 64,
        auto_resize_input: bool = True,
        coeffs: UnifiedCoeffs = UnifiedCoeffs(),
        use_hebbian: bool = True,
        use_antihebbian: bool = True,
        use_holographic: bool = True,
        use_consistency: bool = True,
        use_recursive: bool = True,
        use_free_energy: bool = True,
        use_active_inference: bool = True,
        dropout: float = 0.1,
        inhibition_decay: float = 0.01,
        use_pfc_hopfield: bool = False,
        pfc_hopfield_patterns: int = 64,
        pfc_hopfield_beta: float = 1.0,
        pfc_hopfield_temperature: float = 2.0,
        pfc_hopfield_blend: float = 0.001,
        pfc_hopfield_ema_lr: float = 1e-3,
        pfc_hopfield_unsup_update: bool = False,
        pfc_hopfield_cosine: bool = True,
        pfc_hopfield_soft_ema: bool = True,
        pfc_hopfield_normalize_memory: bool = False,
        pfc_hopfield_sparsity: float = 0.95,
        pfc_hopfield_sparse_update: bool = True,
        pfc_hopfield_layernorm: bool = False,
        pfc_mode: str = "hopfield",
        pfc_hebbian_lr: float = 1e-4,
        pfc_hebbian_decay: float = 1e-5,
        pfc_sa_head_dim: int = 0,
        pfc_topdown_attention: bool = True,
        pfc_topdown_strength: float = 0.25,
        pfc_topdown_min_scale: float = 0.6,
        pfc_topdown_max_scale: float = 1.4,
        pfc_topdown_unsup_lr: float = 3e-3,
        pfc_topdown_decay: float = 1e-4,
        pfc_topdown_per_neuron: bool = True,
        pfc_topdown_neuron_use_bias: bool = False,
        pfc_topdown_shared_fe_blend: float = 0.35,
        pfc_inhibition_feedback_unsup_lr: float = 3e-3,
        pfc_inhibition_feedback_decay: float = 1e-4,
        pfc_topdown_iters: int = 2,
        pfc_predictive_feedback: bool = True,
        pfc_predictive_strength: float = 0.15,
        pfc_predictive_min_scale: float = 0.6,
        pfc_predictive_max_scale: float = 1.4,
        pfc_predictive_unsup_lr: float = 3e-3,
        pfc_predictive_decay: float = 1e-4,
        pfc_l1_lambda: float = 0.0,
        pfc_l1_prox_step: float = 1.0,
        local_l1_lambda: float = 0.0,
        local_l1_prox_step: float = 1.0,
        local_l1_warmup_steps: int = 0,
        local_l1_apply_every: int = 1,
        wta_l1: float = 1e-2,
        wta_l234: float = 1e-2,
        rf_l1: int = 7,
        rf_l2: int = 7,
        rf_l3: int = 7,
        rf_l4: int = 7,
        recursive_iters: int = 5,
        use_dorsal_stream: bool = True,
        pfc_spatial_readout_gate: bool = True,
        pfc_spatial_gate_strength: float = 0.65,
        pfc_spatial_gate_floor: float = 0.2,
        pfc_recurrent_feedback_steps: int = 2,
        pfc_recurrent_feedback_strength: float = 0.35,
        use_pfc_dense_feedback: bool = True,
        pfc_dense_feedback_strength: float = 0.12,
        use_pfc_deep_feedback: bool = True,
        pfc_deep_feedback_rank: int = 32,
        pfc_deep_fb_strength_l2: float = 0.06,
        pfc_deep_fb_strength_l3: float = 0.06,
        pfc_deep_fb_strength_l4: float = 0.08,
        pfc_deep_fb_strength_mt: float = 0.05,
        pfc_deep_fb_strength_pp: float = 0.05,
        use_symmetry_gate_prior: bool = True,
        symmetry_gate_alpha: float = 0.5,
        symmetry_prior_unsup_lr: float = 0.0,
        it_pp_cross_gate: bool = True,
        it_pp_cross_pp_to_te: float = 0.35,
        it_pp_cross_te_to_pp: float = 0.35,
        it_pp_cross_iters: int = 1,
        pfc_pre_hopfield_fusion: str = "all",
        pfc_pre_blend_w_te: float = 0.5,
        pfc_pre_blend_w_pp: float = 0.5,
        pfc_post_readout_fusion: str = "all",
        use_pfc_fusion_gate_unsup: bool = True,
        pfc_fusion_gate_unsup_lr: float = 3e-3,
        pfc_fusion_gate_decay: float = 1e-4,
        pfc_fusion_lms_chunk_rows: int = 256,
        use_pfc_pc_layer_output_mask: bool = True,
        pfc_pc_layer_mask_lr: float = 0.001,
        pfc_pc_layer_mask_decay: float = 1e-4,
        use_neuron_glia: bool = True,
        glia_state_dim: int = 8,
        glia_ema: float = 0.995,
        glia_neuron_strength: float = 0.12,
        glia_gate_min: float = 0.75,
        glia_gate_max: float = 1.25,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.use_free_energy = bool(use_free_energy)
        self.spatial_size = int(spatial_size)
        self.auto_resize_input = bool(auto_resize_input)
        self.coeffs = coeffs
        self.dropout_p = float(dropout)
        self.rf_l1 = int(rf_l1)
        self.rf_l2 = int(rf_l2)
        self.rf_l3 = int(rf_l3)
        self.rf_l4 = int(rf_l4)
        self.recursive_iters = int(max(0, recursive_iters))
        self.use_dorsal_stream = bool(use_dorsal_stream)
        self.use_pfc_spatial_readout_gate = bool(pfc_spatial_readout_gate)
        self.pfc_spatial_gate_strength = float(max(0.0, min(1.0, pfc_spatial_gate_strength)))
        self.pfc_spatial_gate_floor = float(max(0.0, min(1.0, pfc_spatial_gate_floor)))
        self.pfc_recurrent_feedback_steps = int(max(1, min(2, int(pfc_recurrent_feedback_steps))))
        self.pfc_recurrent_feedback_strength = float(max(0.0, pfc_recurrent_feedback_strength))
        self.use_pfc_dense_feedback = bool(use_pfc_dense_feedback)
        self.pfc_dense_feedback_strength = float(max(0.0, pfc_dense_feedback_strength))
        self.use_pfc_deep_feedback = bool(use_pfc_deep_feedback)
        self.pfc_deep_feedback_rank = int(max(4, min(256, int(pfc_deep_feedback_rank))))
        self.pfc_deep_fb_strength_l2 = float(max(0.0, pfc_deep_fb_strength_l2))
        self.pfc_deep_fb_strength_l3 = float(max(0.0, pfc_deep_fb_strength_l3))
        self.pfc_deep_fb_strength_l4 = float(max(0.0, pfc_deep_fb_strength_l4))
        self.pfc_deep_fb_strength_mt = float(max(0.0, pfc_deep_fb_strength_mt))
        self.pfc_deep_fb_strength_pp = float(max(0.0, pfc_deep_fb_strength_pp))
        self.use_symmetry_gate_prior = bool(use_symmetry_gate_prior)
        self.symmetry_gate_alpha = float(max(0.0, symmetry_gate_alpha))
        self.symmetry_prior_unsup_lr = float(max(0.0, symmetry_prior_unsup_lr))
        self.it_pp_cross_gate = bool(it_pp_cross_gate)
        self.it_pp_cross_pp_to_te = float(max(0.0, it_pp_cross_pp_to_te))
        self.it_pp_cross_te_to_pp = float(max(0.0, it_pp_cross_te_to_pp))
        self.it_pp_cross_iters = int(max(1, it_pp_cross_iters))
        _pf = str(pfc_pre_hopfield_fusion).strip().lower()
        if _pf not in ("pp_only", "blend", "gate", "all"):
            raise ValueError('pfc_pre_hopfield_fusion must be "pp_only", "blend", "gate", or "all"')
        self.pfc_pre_hopfield_fusion = _pf
        self.pfc_pre_blend_w_te = float(max(0.0, pfc_pre_blend_w_te))
        self.pfc_pre_blend_w_pp = float(max(0.0, pfc_pre_blend_w_pp))
        _pr = str(pfc_post_readout_fusion).strip().lower()
        if _pr not in ("concat", "gate", "all"):
            raise ValueError('pfc_post_readout_fusion must be "concat", "gate", or "all"')
        self.pfc_post_readout_fusion = _pr
        self.use_pfc_fusion_gate_unsup = bool(use_pfc_fusion_gate_unsup)
        self.pfc_fusion_gate_unsup_lr = float(max(0.0, pfc_fusion_gate_unsup_lr))
        self.pfc_fusion_gate_decay = float(max(0.0, pfc_fusion_gate_decay))
        self.pfc_fusion_lms_chunk_rows = int(max(8, min(4096, int(pfc_fusion_lms_chunk_rows))))

        use_entropy_dropout = getattr(coeffs, "use_entropy_dropout", False)
        entropy_scale = getattr(coeffs, "entropy_dropout_scale", 0.5)
        if use_entropy_dropout:
            self.dropout = EntropyDropout(base_p=self.dropout_p, entropy_scale=entropy_scale)
        else:
            # Dropout on output of each layer (only active in training)
            self.dropout = nn.Dropout(p=self.dropout_p)

        # Preprocessing: no DoG; opponent only. Gabor on luminance (1 ch) with 7 freqs/7 orientations.
        freqs = tuple(i / 3.5 for i in range(1, 8))
        oris = tuple(i * 25 for i in range(7))  # 7 orientations: 0, 25, 50, 75, 100, 125, 150
        #phs = (0, math.pi / 2)
        phs = (0, math.pi)
        
        self.gabor = GaborBank(1, freqs, oris, phs, 7, self.device)  # 1 ch (luminance) -> 96 ch
        self.num_freqs = len(freqs)
        self.gabor_ch_per_freq = 1 * len(oris) * len(phs)  # 24 (12 ori × 2 phase)
        self.use_wavelet_input = bool(getattr(coeffs, "use_wavelet_input", True))
        if self.use_wavelet_input:
            self.wavelet = Wavelet2D()
            self.wavelet_ch_per_band = 3
            self.channels_per_freq = self.gabor_ch_per_freq + self.wavelet_ch_per_band  # 27
        else:
            self.wavelet = None  # Gabor-only streams (24 ch)
            self.wavelet_ch_per_band = 0
            self.channels_per_freq = self.gabor_ch_per_freq

        self.cascade_skip_connections = bool(getattr(coeffs, "cascade_skip_connections", True))
        # L2/L3/L4 input channel counts: with cascade skips, patches stack along the channel dim.
        _l2_in = 2 if self.cascade_skip_connections else 1
        _l3_in = 2 if self.cascade_skip_connections else 1
        _l4_in = 3 if self.cascade_skip_connections else 1

        # SEPARATE FREQUENCY CHANNELS: L1 has 4 separate layers (one per frequency)
        # Each L1: Gabor 24 + optional wavelet 3 = ``channels_per_freq`` (27 or 24)
        self.l1_freq_layers = nn.ModuleList([
            TopographicUnifiedLayer(
                in_channels=self.channels_per_freq,
                rf_size=self.rf_l1,
                spatial_size=self.spatial_size,
                name=f"L1_freq{i}",
                coeffs=coeffs,
                beta_trace=0.0,
                recursive_iters=self.recursive_iters,
                inhibition_strength=0.1,
                inhibition_decay=float(inhibition_decay),
                lr_lateral=coeffs.eta,
                use_hebbian=use_hebbian,
                use_antihebbian=use_antihebbian,
                use_holographic=use_holographic,
                use_consistency=use_consistency,
                use_recursive=use_recursive,
                use_free_energy=use_free_energy,
                use_active_inference=use_active_inference,
                wta_sparsity=float(wta_l1),
                use_wavelet_binding=getattr(coeffs, "use_wavelet_binding", True),
                use_wavelet_denoise=getattr(coeffs, "use_wavelet_denoise", True),
                device=self.device,
            )
            for i in range(self.num_freqs)
        ])

        # L2: 1 ch (plain) or 2 ch (cascade: L1 activity + skip from raw freq input)
        self.l2_freq_layers = nn.ModuleList([
            TopographicUnifiedLayer(
                in_channels=_l2_in,
                rf_size=self.rf_l2,
                spatial_size=self.spatial_size,
                name=f"L2_freq{i}",
                coeffs=coeffs,
                beta_trace=0.9,
                recursive_iters=self.recursive_iters,
                inhibition_strength=0.1,
                inhibition_decay=float(inhibition_decay),
                lr_lateral=coeffs.eta,
                use_hebbian=use_hebbian,
                use_antihebbian=use_antihebbian,
                use_holographic=use_holographic,
                use_consistency=use_consistency,
                use_recursive=use_recursive,
                use_free_energy=use_free_energy,
                use_active_inference=False,
                wta_sparsity=float(wta_l234),
                device=self.device,
            )
            for i in range(self.num_freqs)
        ])

        # L3: 1 ch or 2 ch (cascade: L2 activity + skip from L1)
        self.l3_freq_layers = nn.ModuleList([
            TopographicUnifiedLayer(
                in_channels=_l3_in,
                rf_size=self.rf_l3,
                spatial_size=self.spatial_size,
                name=f"L3_freq{i}",
                coeffs=coeffs,
                beta_trace=0.9,
                recursive_iters=self.recursive_iters,
                inhibition_strength=0.1,
                inhibition_decay=float(inhibition_decay),
                lr_lateral=coeffs.eta,
                use_hebbian=use_hebbian,
                use_antihebbian=use_antihebbian,
                use_holographic=use_holographic,
                use_consistency=use_consistency,
                use_recursive=use_recursive,
                use_free_energy=use_free_energy,
                use_active_inference=False,
                wta_sparsity=float(wta_l234),
                device=self.device,
            )
            for i in range(self.num_freqs)
        ])

        # L4: 1 ch or 3 ch (cascade: L3 + skips from L2 and L1)
        self.l4_freq_layers = nn.ModuleList([
            TopographicUnifiedLayer(
                in_channels=_l4_in,
                rf_size=self.rf_l4,
                spatial_size=self.spatial_size,
                name=f"L4_freq{i}",
                coeffs=coeffs,
                beta_trace=0.9,
                recursive_iters=self.recursive_iters,
                inhibition_strength=0.1,
                inhibition_decay=float(inhibition_decay),
                lr_lateral=coeffs.eta,
                use_hebbian=use_hebbian,
                use_antihebbian=use_antihebbian,
                use_holographic=use_holographic,
                use_consistency=use_consistency,
                use_recursive=use_recursive,
                use_free_energy=use_free_energy,
                use_active_inference=False,
                wta_sparsity=float(wta_l234),
                device=self.device,
            )
            for i in range(self.num_freqs)
        ])

        if self.use_dorsal_stream:
            # MT: same input geometry as ventral L3 (from L2 / "V4"); PP: same as ventral L4 but MT replaces L3 in the cascade.
            self.mt_freq_layers = nn.ModuleList([
                TopographicUnifiedLayer(
                    in_channels=_l3_in,
                    rf_size=self.rf_l3,
                    spatial_size=self.spatial_size,
                    name=f"MT_freq{i}",
                    coeffs=coeffs,
                    beta_trace=0.9,
                    recursive_iters=self.recursive_iters,
                    inhibition_strength=0.1,
                    inhibition_decay=float(inhibition_decay),
                    lr_lateral=coeffs.eta,
                    use_hebbian=use_hebbian,
                    use_antihebbian=use_antihebbian,
                    use_holographic=use_holographic,
                    use_consistency=use_consistency,
                    use_recursive=use_recursive,
                    use_free_energy=use_free_energy,
                    use_active_inference=False,
                    wta_sparsity=float(wta_l234),
                    device=self.device,
                )
                for i in range(self.num_freqs)
            ])
            self.pp_freq_layers = nn.ModuleList([
                TopographicUnifiedLayer(
                    in_channels=_l4_in,
                    rf_size=self.rf_l4,
                    spatial_size=self.spatial_size,
                    name=f"PP_freq{i}",
                    coeffs=coeffs,
                    beta_trace=0.9,
                    recursive_iters=self.recursive_iters,
                    inhibition_strength=0.1,
                    inhibition_decay=float(inhibition_decay),
                    lr_lateral=coeffs.eta,
                    use_hebbian=use_hebbian,
                    use_antihebbian=use_antihebbian,
                    use_holographic=use_holographic,
                    use_consistency=use_consistency,
                    use_recursive=use_recursive,
                    use_free_energy=use_free_energy,
                    use_active_inference=False,
                    wta_sparsity=float(wta_l234),
                    device=self.device,
                )
                for i in range(self.num_freqs)
            ])
        else:
            self.mt_freq_layers = nn.ModuleList()
            self.pp_freq_layers = nn.ModuleList()

        # Classifier reads L4 across frequency channels (optionally ×2 if dorsal stream is fused)
        _l4_single_stream = self.num_freqs * self.spatial_size * self.spatial_size
        total_l4_neurons = _l4_single_stream * (2 if self.use_dorsal_stream else 1)
        self.use_pfc_hopfield = bool(use_pfc_hopfield)
        _pm = str(pfc_mode).strip().lower()
        if self.use_pfc_hopfield:
            if _pm not in ("hopfield", "hebbian_sa"):
                raise ValueError('pfc_mode must be "hopfield" or "hebbian_sa"')
            self.pfc_mode = _pm
        else:
            self.pfc_mode = "hopfield"
        self.use_pfc_topdown_attention = bool(pfc_topdown_attention)
        self.pfc_topdown_strength = float(max(0.0, pfc_topdown_strength))
        self.pfc_topdown_min_scale = float(max(0.0, pfc_topdown_min_scale))
        self.pfc_topdown_max_scale = float(max(self.pfc_topdown_min_scale, pfc_topdown_max_scale))
        self.pfc_topdown_unsup_lr = float(max(0.0, pfc_topdown_unsup_lr))
        self.pfc_topdown_decay = float(max(0.0, pfc_topdown_decay))
        self.pfc_inhibition_feedback_unsup_lr = float(max(0.0, pfc_inhibition_feedback_unsup_lr))
        self.pfc_inhibition_feedback_decay = float(max(0.0, pfc_inhibition_feedback_decay))
        self.pfc_topdown_iters = int(max(1, pfc_topdown_iters))
        self.use_pfc_predictive_feedback = bool(pfc_predictive_feedback)
        self.pfc_predictive_strength = float(max(0.0, pfc_predictive_strength))
        self.pfc_predictive_min_scale = float(max(0.0, pfc_predictive_min_scale))
        self.pfc_predictive_max_scale = float(max(self.pfc_predictive_min_scale, pfc_predictive_max_scale))
        self.pfc_predictive_unsup_lr = float(max(0.0, pfc_predictive_unsup_lr))
        self.pfc_predictive_decay = float(max(0.0, pfc_predictive_decay))
        self.pfc_l1_lambda = float(max(0.0, pfc_l1_lambda))
        self.pfc_l1_prox_step = float(max(0.0, pfc_l1_prox_step))
        self.local_l1_lambda = float(max(0.0, local_l1_lambda))
        self.local_l1_prox_step = float(max(0.0, local_l1_prox_step))
        self.local_l1_warmup_steps = int(max(0, local_l1_warmup_steps))
        self.local_l1_apply_every = int(max(1, local_l1_apply_every))
        self.local_l1_update_counter = 0
        self.use_neuron_glia = bool(use_neuron_glia)
        self.glia_state_dim = int(max(4, glia_state_dim))
        self.glia_ema = float(min(0.9999, max(0.0, glia_ema)))
        self.glia_neuron_strength = float(max(0.0, glia_neuron_strength))
        _gmn = float(glia_gate_min)
        _gmx = float(glia_gate_max)
        self.glia_gate_min = float(min(_gmn, _gmx))
        self.glia_gate_max = float(max(_gmn, _gmx))
        # Unsupervisedly trained top-down attention mapper (no backprop).
        self.pfc_topdown_W = nn.Parameter(torch.eye(4, dtype=torch.float32), requires_grad=False)
        self.pfc_topdown_b = nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad=False)
        # Learnable feedback mapper specifically for inhibition control.
        self.pfc_inhibition_feedback_W = nn.Parameter(torch.eye(4, dtype=torch.float32), requires_grad=False)
        self.pfc_inhibition_feedback_b = nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad=False)
        # Learnable predictive-feedback mapper (PFC prediction -> per-layer error scales).
        self.pfc_predictive_feedback_W = nn.Parameter(torch.eye(4, dtype=torch.float32), requires_grad=False)
        self.pfc_predictive_feedback_b = nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad=False)
        # Learnable top-down: one weight per (layer, neuron) = 4 × (F·H·W) = neurons in L1–L4 concatenated.
        # D = num_freqs × spatial_size² (e.g. F × 32 × 32). Optional bias adds another 4×D parameters.
        _d_flat = int(self.num_freqs * self.spatial_size * self.spatial_size)
        self.use_pfc_topdown_per_neuron = bool(pfc_topdown_per_neuron)
        self.pfc_topdown_neuron_use_bias = bool(pfc_topdown_neuron_use_bias)
        self.pfc_topdown_shared_fe_blend = float(max(0.0, min(1.0, pfc_topdown_shared_fe_blend)))
        _dev_td = self.device
        self.pfc_topdown_neuron_w = nn.Parameter(
            torch.empty(4, _d_flat, device=_dev_td, dtype=torch.float32), requires_grad=False
        )
        nn.init.normal_(self.pfc_topdown_neuron_w, std=0.02)
        if self.pfc_topdown_neuron_use_bias:
            self.register_parameter(
                "pfc_topdown_neuron_b",
                nn.Parameter(torch.zeros(4, _d_flat, device=_dev_td, dtype=torch.float32), requires_grad=False),
            )
        else:
            self.register_parameter("pfc_topdown_neuron_b", None)
        # PFC / Hopfield sees one topographic flat per stream: ventral TE only, or dorsal PP only (not TE+PP).
        _pfc_hf_dim = _l4_single_stream
        if self.use_pfc_hopfield:
            if self.pfc_mode == "hebbian_sa":
                _hd = int(pfc_sa_head_dim)
                if _hd <= 0:
                    _hd = min(64, max(8, _pfc_hf_dim // 8))
                self.pfc_hopfield = HebbianSelfAttentionPFC(
                    feature_dim=_pfc_hf_dim,
                    num_patterns=int(pfc_hopfield_patterns),
                    head_dim=_hd,
                    blend=float(pfc_hopfield_blend),
                    temperature=float(pfc_hopfield_temperature),
                    ema_lr=float(pfc_hopfield_ema_lr),
                    unsup_update=True,
                    use_layernorm=bool(pfc_hopfield_layernorm),
                    hebbian_lr=float(pfc_hebbian_lr),
                    hebbian_decay=float(pfc_hebbian_decay),
                )
            else:
                self.pfc_hopfield = ModernHopfieldPFC(
                    feature_dim=_pfc_hf_dim,
                    num_patterns=int(pfc_hopfield_patterns),
                    beta=float(pfc_hopfield_beta),
                    temperature=float(pfc_hopfield_temperature),
                    blend=float(pfc_hopfield_blend),
                    ema_lr=float(pfc_hopfield_ema_lr),
                    unsup_update=bool(pfc_hopfield_unsup_update),
                    use_cosine_similarity=bool(pfc_hopfield_cosine),
                    soft_ema_update=bool(pfc_hopfield_soft_ema),
                    normalize_memory=bool(pfc_hopfield_normalize_memory),
                    sparsity=float(pfc_hopfield_sparsity),
                    sparse_update=bool(pfc_hopfield_sparse_update),
                    use_layernorm=bool(pfc_hopfield_layernorm),
                )
        else:
            self.pfc_hopfield = None
        # Predictive-coding–trained gates: PFC state -> sigmoid mask per ventral layer (L1–L4); LMS on error vs normalized |activity|.
        _d_pc_mask = _pfc_hf_dim
        self.use_pfc_pc_layer_output_mask = bool(use_pfc_pc_layer_output_mask)
        self.pfc_pc_layer_mask_lr = float(max(0.0, pfc_pc_layer_mask_lr))
        self.pfc_pc_layer_mask_decay = float(max(0.0, pfc_pc_layer_mask_decay))
        _dev = self.device
        self.pfc_pc_layer_mask_W = nn.Parameter(
            torch.empty(_d_pc_mask, 4, device=_dev, dtype=torch.float32), requires_grad=False
        )
        self.pfc_pc_layer_mask_b = nn.Parameter(torch.zeros(4, device=_dev, dtype=torch.float32), requires_grad=False)
        nn.init.xavier_uniform_(self.pfc_pc_layer_mask_W)

        self.clf = nn.Linear(total_l4_neurons, num_classes)
        nn.init.xavier_uniform_(self.clf.weight)
        nn.init.zeros_(self.clf.bias)

        _gd = self.glia_state_dim
        _gdev, _gf = self.device, torch.float32
        self.register_buffer("_glia_trace", torch.zeros(_gd, device=_gdev, dtype=_gf))
        self.register_buffer(
            "_glia_proj_in", torch.randn(_gd, 4, device=_gdev, dtype=_gf).mul_(0.07)
        )
        self.register_buffer(
            "_glia_proj_out", torch.randn(4, _gd, device=_gdev, dtype=_gf).mul_(0.07)
        )

        # Static dorsal spatial gate: DoG blur kernels (centre-surround + texture vs structure).
        _dgk = 15
        self._dorsal_gate_pad = _dgk // 2
        dtyp = torch.float32
        dev = self.device
        self.register_buffer("_gauss_k_small", _gaussian_kernel_2d(_dgk, 1.0, dev, dtyp))
        self.register_buffer("_gauss_k_large", _gaussian_kernel_2d(_dgk, 4.0, dev, dtyp))
        # Unsupervised (no-BP) running scale for symmetry prior: EMA of mean(symmetry_map) per batch.
        self.register_buffer("symmetry_prior_scale_ema", torch.tensor(1.0, device=dev, dtype=dtyp))

        # PFC bottleneck -> spatial maps for deep top-down (L2/L3/L4/MT/PP); no backprop.
        _D = int(self.num_freqs * self.spatial_size * self.spatial_size)
        _r = int(self.pfc_deep_feedback_rank)
        _fh = _D
        self.register_parameter("pfc_fb_Wd", nn.Parameter(torch.empty(_D, _r, device=dev, dtype=dtyp)))
        self.register_parameter("pfc_fb_bd", nn.Parameter(torch.zeros(_r, device=dev, dtype=dtyp)))
        for _name in ("l2", "l3", "l4", "mt", "pp"):
            self.register_parameter(
                f"pfc_fb_U_{_name}", nn.Parameter(torch.empty(_r, _fh, device=dev, dtype=dtyp))
            )
        nn.init.xavier_uniform_(self.pfc_fb_Wd)
        for _name in ("l2", "l3", "l4", "mt", "pp"):
            nn.init.xavier_uniform_(getattr(self, f"pfc_fb_U_{_name}"))
        for _p in (
            self.pfc_fb_Wd,
            self.pfc_fb_bd,
            self.pfc_fb_U_l2,
            self.pfc_fb_U_l3,
            self.pfc_fb_U_l4,
            self.pfc_fb_U_mt,
            self.pfc_fb_U_pp,
        ):
            _p.requires_grad_(False)

        # Fixed (no-BP) attention-like gates: TE↔PP before Hopfield; TE↔PFC for classifier second half.
        self.register_parameter(
            "pfc_pre_gate_W", nn.Parameter(torch.empty(_D, 2 * _D, device=dev, dtype=dtyp))
        )
        self.register_parameter("pfc_pre_gate_b", nn.Parameter(torch.zeros(_D, device=dev, dtype=dtyp)))
        self.register_parameter(
            "pfc_post_gate_W", nn.Parameter(torch.empty(_D, 2 * _D, device=dev, dtype=dtyp))
        )
        self.register_parameter("pfc_post_gate_b", nn.Parameter(torch.zeros(_D, device=dev, dtype=dtyp)))
        nn.init.xavier_uniform_(self.pfc_pre_gate_W)
        nn.init.xavier_uniform_(self.pfc_post_gate_W)
        for _p in (self.pfc_pre_gate_W, self.pfc_pre_gate_b, self.pfc_post_gate_W, self.pfc_post_gate_b):
            _p.requires_grad_(False)

        self.to(self.device)

    @property
    def _single_stream_flat_dim(self) -> int:
        return int(self.num_freqs * self.spatial_size * self.spatial_size)

    @property
    def num_learnable_topdown_neuron_weights(self) -> int:
        """Count of PFC→neuron multiplicative couplings for L1–L4: ``4 × num_freqs × spatial_size²``."""
        if not getattr(self, "use_pfc_topdown_per_neuron", True):
            return 0
        return int(4 * self._single_stream_flat_dim)

    @property
    def l4_feature_dim(self) -> int:
        """Classifier input width: ventral L4 + optional PP (dorsal), each ``num_freqs×spatial²``."""
        n = self._single_stream_flat_dim
        return n * (2 if self.use_dorsal_stream else 1)

    @property
    def unsup_layer_feature_dims(self) -> Dict[str, int]:
        """Per-layer flat feature sizes for variance memory (L4 doubles when PP is fused)."""
        n = self._single_stream_flat_dim
        if not self.use_dorsal_stream:
            return {"l1": n, "l2": n, "l3": n, "l4": n}
        return {"l1": n, "l2": n, "l3": n, "l4": 2 * n}

    @torch.no_grad()
    def _neuron_glia_per_layer_gates(
        self, layer_targets: Dict[str, float], update_trace: bool = True
    ) -> List[float]:
        """
        Slow glial trace from mean |activity| per layer (L1–L4) → per-layer gates on plasticity.
        Neuron → glia: EMA of projected log-activity; glia → neuron: bounded multipliers on
        variance/inhibition scales. Fixed random projections (buffers); no CE backprop.
        """
        if not self.use_neuron_glia:
            return [1.0, 1.0, 1.0, 1.0]
        dev = self._glia_trace.device
        s = torch.tensor(
            [
                float(layer_targets["l1"]),
                float(layer_targets["l2"]),
                float(layer_targets["l3"]),
                float(layer_targets["l4"]),
            ],
            device=dev,
            dtype=torch.float32,
        )
        s = torch.log1p(s.clamp_min(1e-8))
        inc = s @ self._glia_proj_in.T
        if self.training and bool(update_trace):
            self._glia_trace.mul_(self.glia_ema).add_(inc, alpha=1.0 - self.glia_ema)
        delta = torch.tanh(self._glia_trace @ self._glia_proj_out.T)
        g = (1.0 + self.glia_neuron_strength * delta).clamp(self.glia_gate_min, self.glia_gate_max)
        return [float(g[i].item()) for i in range(4)]

    def _ventral_l1_l2(
        self,
        freq_inputs: List[torch.Tensor],
        l1_layers: nn.ModuleList,
        l2_layers: nn.ModuleList,
        l2_pfc_feedback: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        l1_outputs: List[torch.Tensor] = []
        l1_flats: List[torch.Tensor] = []
        l1_patches: List[torch.Tensor] = []
        for i, freq_input in enumerate(freq_inputs):
            y1_map, y1_flat, p1 = l1_layers[i](freq_input)
            l1_outputs.append(y1_map)
            l1_flats.append(y1_flat)
            l1_patches.append(p1)

        l2_outputs: List[torch.Tensor] = []
        l2_flats: List[torch.Tensor] = []
        l2_patches: List[torch.Tensor] = []
        for i, y1_map in enumerate(l1_outputs):
            y1_out = self.dropout(y1_map)
            if self.cascade_skip_connections:
                skip0 = freq_inputs[i].mean(dim=1, keepdim=True)
                l2_in = torch.cat([y1_out, skip0], dim=1)
            else:
                l2_in = y1_out
            y2_map, y2_flat, p2 = l2_layers[i](l2_in)
            if l2_pfc_feedback is not None:
                g = l2_pfc_feedback[i]
                if g.shape[2:] != y2_map.shape[2:]:
                    g = F.interpolate(g, size=y2_map.shape[2:], mode="bilinear", align_corners=False)
                y2_map = y2_map + g.expand_as(y2_map)
                B = y2_map.size(0)
                y2_flat = y2_map.reshape(B, -1)
            l2_outputs.append(y2_map)
            l2_flats.append(y2_flat)
            l2_patches.append(p2)

        return l1_outputs, l1_flats, l1_patches, l2_outputs, l2_flats, l2_patches

    def _l3_inputs_from_l12(self, l1_outputs: List[torch.Tensor], l2_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Same L3 input tensor for ventral L3 and dorsal MT (parallel from L2 / V4-like)."""
        l3_inputs: List[torch.Tensor] = []
        for i, y2_map in enumerate(l2_outputs):
            y2_out = self.dropout(y2_map)
            if self.cascade_skip_connections:
                l3_in = torch.cat([y2_out, l1_outputs[i]], dim=1)
            else:
                l3_in = y2_out
            l3_inputs.append(l3_in)
        return l3_inputs

    def _vertical_symmetry_map(self, L: torch.Tensor) -> torch.Tensor:
        """
        Bilateral consistency map for vertical mirror symmetry: left half vs. flipped right half
        on luminance (per-channel z-score, then correlation-like product), in ~[0, 1].
        """
        B, C, H, W = L.shape
        if W < 4:
            return torch.ones(B, 1, H, W, device=L.device, dtype=L.dtype)
        w_left = W // 2
        w_right = W - w_left
        m = int(min(w_left, w_right))
        if m < 2:
            return torch.ones(B, 1, H, W, device=L.device, dtype=L.dtype)
        left = L[:, :, :, :m]
        right = L[:, :, :, W - m :].flip(-1)
        left_n = (left - left.mean(dim=(2, 3), keepdim=True)) / (left.std(dim=(2, 3), keepdim=True) + 1e-6)
        right_n = (right - right.mean(dim=(2, 3), keepdim=True)) / (right.std(dim=(2, 3), keepdim=True) + 1e-6)
        sym_half = (left_n * right_n).mean(dim=1, keepdim=True)
        sym_half = (sym_half + 1.0) * 0.5
        sym_half = sym_half.clamp(0.0, 1.0)
        sym_full = torch.cat([sym_half, sym_half.flip(-1)], dim=-1)
        if sym_full.size(-1) != W:
            sym_full = F.interpolate(sym_full, size=(H, W), mode="bilinear", align_corners=False)
        return sym_full

    @torch.no_grad()
    def _maybe_update_symmetry_prior_unsup(self, sym_map: torch.Tensor) -> None:
        """EMA of batch-mean symmetry map; scales prior strength without backprop."""
        if self.symmetry_prior_unsup_lr <= 0.0 or (not self.training):
            return
        lr = float(self.symmetry_prior_unsup_lr)
        m = float(sym_map.detach().mean().clamp(0.0, 1.0))
        v = (1.0 - lr) * float(self.symmetry_prior_scale_ema) + lr * m
        self.symmetry_prior_scale_ema.fill_(min(1.0, max(0.0, v)))

    def _dorsal_spatial_gate_map(self, L: torch.Tensor) -> torch.Tensor:
        """
        Static-image dorsal attention: centre-surround DoG saliency, texture (high-pass) energy,
        and local z-score figure-ground — combined into a soft map in ~[0, 1] per image.
        Optionally multiply by a vertical-symmetry prior (unsupervised EMA scale, no BP).
        """
        pad = self._dorsal_gate_pad
        g1 = F.conv2d(L, self._gauss_k_small, padding=pad)
        g4 = F.conv2d(L, self._gauss_k_large, padding=pad)
        dog = g1 - g4
        dog_n = dog / (dog.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
        s_dog = torch.sigmoid(3.0 * dog_n)

        texture = L - g4
        tex_e = texture.abs()
        tex_e = tex_e / (tex_e.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))

        kloc = 15
        ploc = kloc // 2
        local_mean = F.avg_pool2d(L, kloc, stride=1, padding=ploc)
        local_sq = F.avg_pool2d(L * L, kloc, stride=1, padding=ploc)
        local_var = (local_sq - local_mean * local_mean).clamp_min(0.0)
        local_std = torch.sqrt(local_var + 1e-6)
        z = (L - local_mean) / local_std
        s_fg = torch.sigmoid(1.5 * z)

        gate = s_dog * s_fg * (0.5 + 0.5 * tex_e)
        gate = gate / (gate.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))

        if self.use_symmetry_gate_prior:
            sym = self._vertical_symmetry_map(L)
            self._maybe_update_symmetry_prior_unsup(sym)
            scale = float(self.symmetry_prior_scale_ema) if self.symmetry_prior_unsup_lr > 0.0 else 1.0
            a = float(self.symmetry_gate_alpha) * scale
            gate = gate * (1.0 + a * sym)
            gate = gate / (gate.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
        return gate

    def _mt_inputs_from_l3_inputs(
        self, l3_inputs: List[torch.Tensor], spatial_gate: torch.Tensor
    ) -> List[torch.Tensor]:
        """Gate ventral L3-sized tensors toward salient / figure-like regions for MT (same layout as L3)."""
        out: List[torch.Tensor] = []
        for t in l3_inputs:
            g = spatial_gate
            if g.shape[2:] != t.shape[2:]:
                g = F.interpolate(g, size=t.shape[2:], mode="bilinear", align_corners=False)
            out.append(t * (1.0 + 0.75 * g))
        return out

    def _ventral_l3_l4(
        self,
        l3_inputs: List[torch.Tensor],
        l1_outputs: List[torch.Tensor],
        l2_outputs: List[torch.Tensor],
        l3_layers: nn.ModuleList,
        l4_layers: nn.ModuleList,
        l3_pfc_feedback: Optional[List[torch.Tensor]] = None,
        l4_pfc_feedback: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        l3_outputs: List[torch.Tensor] = []
        l3_flats: List[torch.Tensor] = []
        l3_patches: List[torch.Tensor] = []
        for i, l3_in in enumerate(l3_inputs):
            y3_map, y3_flat, p3 = l3_layers[i](l3_in)
            if l3_pfc_feedback is not None:
                g = l3_pfc_feedback[i]
                if g.shape[2:] != y3_map.shape[2:]:
                    g = F.interpolate(g, size=y3_map.shape[2:], mode="bilinear", align_corners=False)
                y3_map = y3_map + g.expand_as(y3_map)
                B = y3_map.size(0)
                y3_flat = y3_map.reshape(B, -1)
            l3_outputs.append(y3_map)
            l3_flats.append(y3_flat)
            l3_patches.append(p3)

        l4_outputs: List[torch.Tensor] = []
        l4_flats: List[torch.Tensor] = []
        l4_patches: List[torch.Tensor] = []
        for i, y3_map in enumerate(l3_outputs):
            y3_out = self.dropout(y3_map)
            if self.cascade_skip_connections:
                l4_in = torch.cat([y3_out, l2_outputs[i], l1_outputs[i]], dim=1)
            else:
                l4_in = y3_out
            y4_map, y4_flat, p4 = l4_layers[i](l4_in)
            if l4_pfc_feedback is not None:
                g = l4_pfc_feedback[i]
                if g.shape[2:] != y4_map.shape[2:]:
                    g = F.interpolate(g, size=y4_map.shape[2:], mode="bilinear", align_corners=False)
                y4_map = y4_map + g.expand_as(y4_map)
                B = y4_map.size(0)
                y4_flat = y4_map.reshape(B, -1)
            l4_outputs.append(y4_map)
            l4_flats.append(y4_flat)
            l4_patches.append(p4)

        return l3_outputs, l3_flats, l3_patches, l4_outputs, l4_flats, l4_patches

    def _run_mt_pp_from_l12(
        self,
        l3_inputs: List[torch.Tensor],
        l1_outputs: List[torch.Tensor],
        l2_outputs: List[torch.Tensor],
        mt_pfc_feedback: Optional[List[torch.Tensor]] = None,
        pp_pfc_feedback: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Dorsal: MT from spatially gated L3 inputs, then PP (cascade mirrors ventral L4 with MT substituting L3)."""
        mt_flats: List[torch.Tensor] = []
        mt_patches: List[torch.Tensor] = []
        pp_flats: List[torch.Tensor] = []
        pp_patches: List[torch.Tensor] = []
        for i in range(self.num_freqs):
            l3_in = l3_inputs[i]
            y_mt, mt_flat, mt_p = self.mt_freq_layers[i](l3_in)
            y_mt_out = self.dropout(y_mt)
            if mt_pfc_feedback is not None:
                g = mt_pfc_feedback[i]
                if g.shape[2:] != y_mt_out.shape[2:]:
                    g = F.interpolate(g, size=y_mt_out.shape[2:], mode="bilinear", align_corners=False)
                y_mt_out = y_mt_out + g.expand_as(y_mt_out)
                B = y_mt_out.size(0)
                mt_flat = y_mt_out.reshape(B, -1)
            if self.cascade_skip_connections:
                pp_in = torch.cat([y_mt_out, l2_outputs[i], l1_outputs[i]], dim=1)
            else:
                pp_in = y_mt_out
            y_pp, pp_flat, pp_p = self.pp_freq_layers[i](pp_in)
            if pp_pfc_feedback is not None:
                g = pp_pfc_feedback[i]
                if g.shape[2:] != y_pp.shape[2:]:
                    g = F.interpolate(g, size=y_pp.shape[2:], mode="bilinear", align_corners=False)
                y_pp = y_pp + g.expand_as(y_pp)
                B = y_pp.size(0)
                pp_flat = y_pp.reshape(B, -1)
            mt_flats.append(mt_flat)
            mt_patches.append(mt_p)
            pp_flats.append(pp_flat)
            pp_patches.append(pp_p)
        return mt_flats, mt_patches, pp_flats, pp_patches

    def _run_stream_from_freq_inputs(
        self,
        freq_inputs: List[torch.Tensor],
        l1_layers: nn.ModuleList,
        l2_layers: nn.ModuleList,
        l3_layers: nn.ModuleList,
        l4_layers: nn.ModuleList,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """Full ventral L1–L4 for a list of per-frequency inputs."""
        l1_outputs, l1_flats, l1_patches, l2_outputs, l2_flats, l2_patches = self._ventral_l1_l2(
            freq_inputs, l1_layers, l2_layers
        )
        l3_inputs = self._l3_inputs_from_l12(l1_outputs, l2_outputs)
        l3_outputs, l3_flats, l3_patches, l4_outputs, l4_flats, l4_patches = self._ventral_l3_l4(
            l3_inputs, l1_outputs, l2_outputs, l3_layers, l4_layers
        )
        return (
            l1_outputs,
            l1_flats,
            l1_patches,
            l2_outputs,
            l2_flats,
            l2_patches,
            l3_outputs,
            l3_flats,
            l3_patches,
            l4_outputs,
            l4_flats,
            l4_patches,
        )

    def iter_topographic_layers(self):
        """All L1–L4 TopographicUnifiedLayer modules (for SOM schedules, etc.)."""
        for lst in (self.l1_freq_layers, self.l2_freq_layers, self.l3_freq_layers, self.l4_freq_layers):
            for layer in lst:
                yield layer
        if self.use_dorsal_stream:
            for lst in (self.mt_freq_layers, self.pp_freq_layers):
                for layer in lst:
                    yield layer

    @torch.no_grad()
    def apply_som_schedules(self, epoch: int, total_epochs: int, coeffs: UnifiedCoeffs) -> None:
        """Update per-layer SOM-style inhibition schedules once per training epoch."""
        te = max(1, int(total_epochs))
        progress = (float(epoch) - 1.0) / float(te - 1) if te > 1 else 0.0
        for layer in self.iter_topographic_layers():
            layer.apply_som_schedule(progress, coeffs)

    @torch.no_grad()
    def apply_joint_plasticity_lr_scale(self, scale: float, coeffs: UnifiedCoeffs) -> None:
        """Scale local plasticity + PFC unsupervised rates by ``scale = lr_clf / lr_clf_initial`` (cosine schedule)."""
        s = float(max(0.0, scale))
        if not hasattr(self, "_joint_lr_snapshot") or self._joint_lr_snapshot is None:
            layers_snap = []
            for layer in self.iter_topographic_layers():
                layers_snap.append(
                    {
                        "lr_lateral_base": float(layer.lr_lateral_base),
                        "holo_fast_lr": float(layer.holo_fast_lr),
                        "active_inference_lr": float(layer.active_inference_lr),
                    }
                )
            hop_ema = float(self.pfc_hopfield.ema_lr) if self.pfc_hopfield is not None else None
            hop_hebb = (
                float(getattr(self.pfc_hopfield, "hebbian_lr"))
                if self.pfc_hopfield is not None and hasattr(self.pfc_hopfield, "hebbian_lr")
                else None
            )
            self._joint_lr_snapshot = {
                "eta": float(coeffs.eta),
                "alpha": float(coeffs.alpha),
                "holo_fast_lr": float(coeffs.holo_fast_lr),
                "distance_gradient_lr": float(coeffs.distance_gradient_lr),
                "beta_hyp": float(coeffs.beta_hyp),
                "gamma_wavelet": float(coeffs.gamma_wavelet),
                "lambda_a": float(coeffs.lambda_a),
                "lambda_c": float(coeffs.lambda_c),
                "lambda_r": float(coeffs.lambda_r),
                "lambda_F": float(coeffs.lambda_F),
                "inhibition_adapt_lr": float(coeffs.inhibition_adapt_lr),
                "homeostatic_threshold_lr": float(coeffs.homeostatic_threshold_lr),
                "layers": layers_snap,
                "pfc_topdown_unsup_lr": float(self.pfc_topdown_unsup_lr),
                "pfc_inhibition_feedback_unsup_lr": float(self.pfc_inhibition_feedback_unsup_lr),
                "pfc_predictive_unsup_lr": float(self.pfc_predictive_unsup_lr),
                "pfc_fusion_gate_unsup_lr": float(self.pfc_fusion_gate_unsup_lr),
                "hopfield_ema_lr": hop_ema,
                "hopfield_hebbian_lr": hop_hebb,
            }
        snap = self._joint_lr_snapshot
        coeffs.eta = snap["eta"] * s
        coeffs.alpha = snap["alpha"] * s
        coeffs.holo_fast_lr = snap["holo_fast_lr"] * s
        coeffs.distance_gradient_lr = snap["distance_gradient_lr"] * s
        coeffs.beta_hyp = snap["beta_hyp"] * s
        coeffs.gamma_wavelet = snap["gamma_wavelet"] * s
        coeffs.lambda_a = snap["lambda_a"] * s
        coeffs.lambda_c = snap["lambda_c"] * s
        coeffs.lambda_r = snap["lambda_r"] * s
        coeffs.lambda_F = snap["lambda_F"] * s
        coeffs.inhibition_adapt_lr = snap["inhibition_adapt_lr"] * s
        coeffs.homeostatic_threshold_lr = snap["homeostatic_threshold_lr"] * s
        for layer, b in zip(self.iter_topographic_layers(), snap["layers"]):
            layer.lr_lateral_base = b["lr_lateral_base"] * s
            layer.holo_fast_lr = b["holo_fast_lr"] * s
            layer.active_inference_lr = b["active_inference_lr"] * s
        self.pfc_topdown_unsup_lr = snap["pfc_topdown_unsup_lr"] * s
        self.pfc_inhibition_feedback_unsup_lr = snap["pfc_inhibition_feedback_unsup_lr"] * s
        self.pfc_predictive_unsup_lr = snap["pfc_predictive_unsup_lr"] * s
        self.pfc_fusion_gate_unsup_lr = float(
            max(0.0, float(snap.get("pfc_fusion_gate_unsup_lr", self.pfc_fusion_gate_unsup_lr)) * s)
        )
        if self.pfc_hopfield is not None and snap["hopfield_ema_lr"] is not None:
            self.pfc_hopfield.ema_lr = float(min(1.0, max(0.0, snap["hopfield_ema_lr"] * s)))
        if (
            self.pfc_hopfield is not None
            and snap.get("hopfield_hebbian_lr") is not None
            and hasattr(self.pfc_hopfield, "hebbian_lr")
        ):
            self.pfc_hopfield.hebbian_lr = float(max(0.0, float(snap["hopfield_hebbian_lr"]) * s))

    @staticmethod
    @torch.no_grad()
    def _tensor_soft_threshold_l1_(w: torch.Tensor, tau: float, chunk_elems: int = 2_097_152) -> None:
        """
        Apply w := sign(w) * max(|w| - tau, 0). Small tensors use one vectorized pass (fast).
        Very large fusion gates [D, 2D] are chunked in big slices to limit peak memory and Python loop cost.
        """
        if tau <= 0.0:
            return
        n = w.numel()
        # Fast path: small enough that ~2×W temporaries stay modest (e.g. PFC 4×4 maps, local W).
        if n <= 4_194_304:
            abs_part = w.abs()
            abs_part.sub_(tau).clamp_(min=0.0)
            torch.mul(w.sign(), abs_part, out=w)
            return
        flat = w.reshape(-1)
        for start in range(0, n, chunk_elems):
            sl = flat[start : start + chunk_elems]
            abs_part = sl.abs()
            abs_part.sub_(tau).clamp_(min=0.0)
            torch.mul(sl.sign(), abs_part, out=sl)

    @torch.no_grad()
    def _apply_pfc_l1_prox(self) -> None:
        if self.pfc_l1_lambda <= 0.0 or self.pfc_l1_prox_step <= 0.0:
            return
        tau = float(self.pfc_l1_lambda * self.pfc_l1_prox_step)
        _pfc_td_neuron = [self.pfc_topdown_neuron_w]
        if self.pfc_topdown_neuron_b is not None:
            _pfc_td_neuron.append(self.pfc_topdown_neuron_b)
        for w in (
            self.pfc_topdown_W,
            self.pfc_inhibition_feedback_W,
            self.pfc_predictive_feedback_W,
            *_pfc_td_neuron,
            self.pfc_pc_layer_mask_W,
            self.pfc_pre_gate_W,
            self.pfc_post_gate_W,
        ):
            self._tensor_soft_threshold_l1_(w, tau)

    @torch.no_grad()
    def _apply_local_l1_prox(self) -> None:
        if self.local_l1_lambda <= 0.0 or self.local_l1_prox_step <= 0.0:
            return
        tau = float(self.local_l1_lambda * self.local_l1_prox_step)
        stacks = (
            *self.l1_freq_layers,
            *self.l2_freq_layers,
            *self.l3_freq_layers,
            *self.l4_freq_layers,
        )
        if self.use_dorsal_stream:
            stacks = (*stacks, *self.mt_freq_layers, *self.pp_freq_layers)
        for layer in stacks:
            self._tensor_soft_threshold_l1_(layer.W, tau)
            layer.W.clamp_(-layer.w_clip, layer.w_clip)

    @torch.no_grad()
    def _maybe_apply_local_l1_prox(self) -> None:
        if (not self.training) or self.local_l1_lambda <= 0.0 or self.local_l1_prox_step <= 0.0:
            return
        self.local_l1_update_counter += 1
        if self.local_l1_update_counter <= self.local_l1_warmup_steps:
            return
        if (self.local_l1_update_counter % self.local_l1_apply_every) != 0:
            return
        self._apply_local_l1_prox()

    def _pfc_attention_vector(self, pfc_features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if (
            pfc_features is None
            or pfc_features.ndim != 2
            or pfc_features.size(0) <= 0
            or pfc_features.size(1) < 4
            or (not bool(self.use_pfc_hopfield))
            or (not (bool(self.use_pfc_topdown_attention) or bool(self.use_pfc_predictive_feedback)))
        ):
            return None
        chunks = torch.chunk(pfc_features.detach(), 4, dim=1)
        scores = torch.stack([c.abs().mean(dim=1) for c in chunks], dim=1)  # [B,4]
        return F.softmax(scores, dim=1).mean(dim=0).to(self.pfc_topdown_W.dtype)  # [4]

    def _stack_layer_flat_abs_mean(self, l_flats: List[torch.Tensor]) -> torch.Tensor:
        """Batch-mean |y| per frequency map, concatenated to length F·H·W."""
        parts = [torch.abs(l_flats[i]).mean(dim=0).flatten() for i in range(len(l_flats))]
        return torch.cat(parts, dim=0)

    def _stack_layer_fe_signal_from_stack(self, freq_layers: nn.ModuleList) -> torch.Tensor:
        """Concatenate ``last_fe_signal_per_neuron`` from each frequency layer → length F·H·W."""
        parts = [layer.last_fe_signal_per_neuron.detach().flatten() for layer in freq_layers]
        return torch.cat(parts, dim=0)

    def _layer_pc_fe_mean_per_stack(self, freq_layers: nn.ModuleList) -> float:
        """Mean |G^T err| over neurons in this layer stack (same PC signal as ``last_fe_signal_per_neuron``)."""
        if not bool(getattr(self, "use_free_energy", True)):
            return 0.0
        tot = 0.0
        n = 0
        with torch.no_grad():
            for layer in freq_layers:
                s = layer.last_fe_signal_per_neuron
                tot += float(s.sum().item())
                n += int(s.numel())
        return tot / max(1, n)

    def _blend_layer_targets_for_pfc_unsup(
        self,
        l1_a: float,
        l2_a: float,
        l3_a: float,
        l4_a: float,
    ) -> Dict[str, float]:
        """
        Scalar per-layer targets for PFC softmax LMS (4×4 paths) and ``pred_scales`` input:
        blends mean |y| with mean PC |G^T err| per layer, same α as ``pfc_topdown_shared_fe_blend``.
        """
        alpha = float(getattr(self, "pfc_topdown_shared_fe_blend", 0.0))
        if alpha <= 0.0 or not bool(getattr(self, "use_free_energy", True)):
            return {"l1": l1_a, "l2": l2_a, "l3": l3_a, "l4": l4_a}
        stacks = (
            self.l1_freq_layers,
            self.l2_freq_layers,
            self.l3_freq_layers,
            self.l4_freq_layers,
        )
        fe = torch.tensor(
            [self._layer_pc_fe_mean_per_stack(stacks[i]) for i in range(4)],
            dtype=torch.float64,
        )
        act = torch.tensor([l1_a, l2_a, l3_a, l4_a], dtype=torch.float64)
        fe_n = fe / (fe.max() + 1e-8)
        act_n = act / (act.max() + 1e-8)
        if float(fe.sum()) <= 1e-12:
            return {"l1": l1_a, "l2": l2_a, "l3": l3_a, "l4": l4_a}
        mixed = (1.0 - alpha) * act_n + alpha * fe_n
        scale = float(act.max().item()) + 1e-12
        names = ("l1", "l2", "l3", "l4")
        return {names[i]: float(mixed[i].item() * scale) for i in range(4)}

    def _pfc_neuron_topdown_scales(self, pfc_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        One learned weight per (layer, neuron index): ``4 × D`` with ``D = num_freqs × H × W``
        (total weights = all neurons across L1–L4). Returns per-layer tensors [D] mapped to
        ``[pfc_topdown_min_scale, pfc_topdown_max_scale]``.
        """
        dev, dt = pfc_state.device, pfc_state.dtype
        d_exp = int(self._single_stream_flat_dim)
        one = torch.ones(d_exp, device=dev, dtype=dt)
        default = {"l1": one.clone(), "l2": one.clone(), "l3": one.clone(), "l4": one.clone()}
        if (
            (not bool(self.use_pfc_topdown_attention))
            or pfc_state.ndim != 2
            or int(pfc_state.size(1)) != d_exp
        ):
            return default
        pfm = pfc_state.detach().mean(dim=0)
        mn = float(self.pfc_topdown_min_scale)
        mx = float(self.pfc_topdown_max_scale)
        out: Dict[str, torch.Tensor] = {}
        for ik, name in enumerate(("l1", "l2", "l3", "l4")):
            raw = pfm * self.pfc_topdown_neuron_w[ik]
            if self.pfc_topdown_neuron_b is not None:
                raw = raw + self.pfc_topdown_neuron_b[ik]
            # Map to [mn, mx] then mean-normalize to ~1 so global η is not biased vs scalar top-down (was ~0.85–0.95).
            g = torch.sigmoid(raw) * (mx - mn) + mn
            out[name] = g / (g.mean() + 1e-8)
        return out

    @torch.no_grad()
    def _update_pfc_neuron_topdown_unsup(
        self,
        pfc_state: torch.Tensor,
        l1_flats: List[torch.Tensor],
        l2_flats: List[torch.Tensor],
        l3_flats: List[torch.Tensor],
        l4_flats: List[torch.Tensor],
    ) -> None:
        """LMS-style update for ``pfc_topdown_neuron_*``; target blends |activity| with shared FE |G^T err|."""
        if (
            (not getattr(self, "use_pfc_topdown_per_neuron", True))
            or (not bool(self.use_pfc_topdown_attention))
            or float(self.pfc_topdown_unsup_lr) <= 0.0
            or (not self.training)
            or pfc_state.ndim != 2
            or int(pfc_state.size(1)) != int(self._single_stream_flat_dim)
        ):
            return
        pfm = pfc_state.detach().mean(dim=0)
        targets_act = {
            "l1": self._stack_layer_flat_abs_mean(l1_flats),
            "l2": self._stack_layer_flat_abs_mean(l2_flats),
            "l3": self._stack_layer_flat_abs_mean(l3_flats),
            "l4": self._stack_layer_flat_abs_mean(l4_flats),
        }
        layer_stacks = {
            "l1": self.l1_freq_layers,
            "l2": self.l2_freq_layers,
            "l3": self.l3_freq_layers,
            "l4": self.l4_freq_layers,
        }
        alpha = float(getattr(self, "pfc_topdown_shared_fe_blend", 0.0))
        lr = float(self.pfc_topdown_unsup_lr)
        decay = float(self.pfc_topdown_decay)
        for ik, name in enumerate(("l1", "l2", "l3", "l4")):
            t_act = targets_act[name] / (targets_act[name].max() + 1e-8)
            if alpha > 0.0:
                t_fe = self._stack_layer_fe_signal_from_stack(layer_stacks[name])
                t_fe = t_fe / (t_fe.max() + 1e-8)
                if float(t_fe.sum()) > 1e-12:
                    t = (1.0 - alpha) * t_act + alpha * t_fe
                else:
                    t = t_act
            else:
                t = t_act
            raw = self.pfc_topdown_neuron_w[ik] * pfm
            if self.pfc_topdown_neuron_b is not None:
                raw = raw + self.pfc_topdown_neuron_b[ik]
            pred = torch.sigmoid(raw)
            err = t - pred
            self.pfc_topdown_neuron_w[ik].mul_(1.0 - decay).add_(err * pfm, alpha=lr)
            if self.pfc_topdown_neuron_b is not None:
                self.pfc_topdown_neuron_b[ik].mul_(1.0 - decay).add_(err, alpha=lr)
        self.pfc_topdown_neuron_w.clamp_(-2.0, 2.0)
        if self.pfc_topdown_neuron_b is not None:
            self.pfc_topdown_neuron_b.clamp_(-2.0, 2.0)

    @torch.no_grad()
    def _update_pfc_topdown_unsup(self, attn_vec: Optional[torch.Tensor], layer_targets: Dict[str, float]) -> None:
        """
        Local unsupervised update for top-down attention weights.
        Learns to map current PFC attention to desired layer-control distribution.
        """
        if getattr(self, "use_pfc_topdown_per_neuron", True):
            return
        if (
            attn_vec is None
            or self.pfc_topdown_unsup_lr <= 0.0
            or (not self.training)
            or (not bool(self.use_pfc_topdown_attention))
        ):
            return
        target = torch.tensor(
            [
                float(layer_targets.get("l1", 1.0)),
                float(layer_targets.get("l2", 1.0)),
                float(layer_targets.get("l3", 1.0)),
                float(layer_targets.get("l4", 1.0)),
            ],
            device=attn_vec.device,
            dtype=attn_vec.dtype,
        )
        target = target.clamp_min(1e-6)
        target = target / (target.sum() + 1e-8)
        pred_logits = self.pfc_topdown_W @ attn_vec + self.pfc_topdown_b
        pred = F.softmax(pred_logits, dim=0)
        err = target - pred
        lr = float(self.pfc_topdown_unsup_lr)
        decay = float(self.pfc_topdown_decay)
        self.pfc_topdown_W.mul_(1.0 - decay).add_(torch.outer(err, attn_vec), alpha=lr)
        self.pfc_topdown_b.mul_(1.0 - decay).add_(err, alpha=lr)
        self._apply_pfc_l1_prox()
        self.pfc_topdown_W.clamp_(-2.0, 2.0)
        self.pfc_topdown_b.clamp_(-2.0, 2.0)

    def _pfc_topdown_layer_scales(self, pfc_features: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        Derive top-down control scales for L1-L4 from current PFC features.
        Non-parametric attention over 4 equal feature chunks keeps this stable and cheap.
        """
        default_scales = {"l1": 1.0, "l2": 1.0, "l3": 1.0, "l4": 1.0}
        attn_vec = self._pfc_attention_vector(pfc_features)
        if attn_vec is None:
            return default_scales
        logits = self.pfc_topdown_W @ attn_vec + self.pfc_topdown_b
        attn = F.softmax(logits, dim=0)
        centered = (attn - 0.25) * 4.0  # roughly [-1, 1]
        scales = 1.0 + float(self.pfc_topdown_strength) * centered
        scales = torch.clamp(
            scales,
            min=float(self.pfc_topdown_min_scale),
            max=float(self.pfc_topdown_max_scale),
        )
        return {
            "l1": float(scales[0].item()),
            "l2": float(scales[1].item()),
            "l3": float(scales[2].item()),
            "l4": float(scales[3].item()),
        }

    @torch.no_grad()
    def _update_pfc_inhibition_feedback_unsup(
        self, attn_vec: Optional[torch.Tensor], layer_targets: Dict[str, float]
    ) -> None:
        if (
            attn_vec is None
            or self.pfc_inhibition_feedback_unsup_lr <= 0.0
            or (not self.training)
            or (not bool(self.use_pfc_topdown_attention))
        ):
            return
        target = torch.tensor(
            [
                float(layer_targets.get("l1", 1.0)),
                float(layer_targets.get("l2", 1.0)),
                float(layer_targets.get("l3", 1.0)),
                float(layer_targets.get("l4", 1.0)),
            ],
            device=attn_vec.device,
            dtype=attn_vec.dtype,
        )
        target = target.clamp_min(1e-6)
        target = target / (target.sum() + 1e-8)
        pred_logits = self.pfc_inhibition_feedback_W @ attn_vec + self.pfc_inhibition_feedback_b
        pred = F.softmax(pred_logits, dim=0)
        err = target - pred
        lr = float(self.pfc_inhibition_feedback_unsup_lr)
        decay = float(self.pfc_inhibition_feedback_decay)
        self.pfc_inhibition_feedback_W.mul_(1.0 - decay).add_(torch.outer(err, attn_vec), alpha=lr)
        self.pfc_inhibition_feedback_b.mul_(1.0 - decay).add_(err, alpha=lr)
        self._apply_pfc_l1_prox()
        self.pfc_inhibition_feedback_W.clamp_(-2.0, 2.0)
        self.pfc_inhibition_feedback_b.clamp_(-2.0, 2.0)

    def _pfc_inhibition_layer_scales(self, pfc_features: Optional[torch.Tensor]) -> Dict[str, float]:
        default_scales = {"l1": 1.0, "l2": 1.0, "l3": 1.0, "l4": 1.0}
        attn_vec = self._pfc_attention_vector(pfc_features)
        if attn_vec is None:
            return default_scales
        logits = self.pfc_inhibition_feedback_W @ attn_vec + self.pfc_inhibition_feedback_b
        attn = F.softmax(logits, dim=0)
        centered = (attn - 0.25) * 4.0
        scales = 1.0 + float(self.pfc_topdown_strength) * centered
        scales = torch.clamp(
            scales,
            min=float(self.pfc_topdown_min_scale),
            max=float(self.pfc_topdown_max_scale),
        )
        return {
            "l1": float(scales[0].item()),
            "l2": float(scales[1].item()),
            "l3": float(scales[2].item()),
            "l4": float(scales[3].item()),
        }

    @torch.no_grad()
    def _update_pfc_predictive_feedback_unsup(
        self, attn_vec: Optional[torch.Tensor], layer_targets: Dict[str, float]
    ) -> None:
        if (
            attn_vec is None
            or self.pfc_predictive_unsup_lr <= 0.0
            or (not self.training)
            or (not bool(self.use_pfc_predictive_feedback))
        ):
            return
        target = torch.tensor(
            [
                float(layer_targets.get("l1", 1.0)),
                float(layer_targets.get("l2", 1.0)),
                float(layer_targets.get("l3", 1.0)),
                float(layer_targets.get("l4", 1.0)),
            ],
            device=attn_vec.device,
            dtype=attn_vec.dtype,
        )
        target = target.clamp_min(1e-6)
        target = target / (target.sum() + 1e-8)
        pred_logits = self.pfc_predictive_feedback_W @ attn_vec + self.pfc_predictive_feedback_b
        pred = F.softmax(pred_logits, dim=0)
        err = target - pred
        lr = float(self.pfc_predictive_unsup_lr)
        decay = float(self.pfc_predictive_decay)
        self.pfc_predictive_feedback_W.mul_(1.0 - decay).add_(torch.outer(err, attn_vec), alpha=lr)
        self.pfc_predictive_feedback_b.mul_(1.0 - decay).add_(err, alpha=lr)
        self._apply_pfc_l1_prox()
        self.pfc_predictive_feedback_W.clamp_(-2.0, 2.0)
        self.pfc_predictive_feedback_b.clamp_(-2.0, 2.0)

    @staticmethod
    @torch.no_grad()
    def _pfc_fusion_lms_update_linear_(
        W: torch.Tensor,
        b: torch.Tensor,
        err: torch.Tensor,
        x: torch.Tensor,
        lr: float,
        decay: float,
        chunk_rows: int = 256,
    ) -> None:
        """
        LMS gate update without materializing full err.T @ x (can be [D,2D] GiB-scale on large maps).
        Accumulates (err[:,rs:re].T @ x) / B per row chunk.
        """
        bsz = float(x.size(0))
        out_d = W.size(0)
        W.mul_(1.0 - decay)
        b.mul_(1.0 - decay)
        for rs in range(0, out_d, chunk_rows):
            re = min(rs + chunk_rows, out_d)
            dW = (err[:, rs:re].T @ x) / bsz
            W[rs:re, :].add_(dW, alpha=lr)
        b.add_(err.mean(dim=0), alpha=lr)

    @torch.no_grad()
    def _update_pfc_fusion_gates_unsup(
        self,
        te_for_pre: Optional[torch.Tensor],
        pp_for_pre: Optional[torch.Tensor],
        te_readout: torch.Tensor,
        pfc_readout: torch.Tensor,
    ) -> None:
        """
        Local LMS / Hebbian-style update on pre/post fusion gates (fixed matrices, no classifier BP).
        Pre-gate uses TE and PP **before** ``it_pp`` cross-gate (same tensors as ``_pfc_fuse_te_pp_pre_hopfield``
        in ``_run_ventral_dorsal_stack``). Post-gate uses TE and post-Hopfield PFC after the spatial readout gate.
        Target per dimension is relative saliency: |TE| / (|TE| + |other|) in [0, 1], matching the
        role of g in g*TE + (1-g)*other. This does not replace ``it_pp_cross_gate`` (spatial map coupling);
        it learns a separate channel-wise fusion for the Hopfield input and classifier readout.
        """
        if (
            (not self.training)
            or (not bool(self.use_pfc_fusion_gate_unsup))
            or float(self.pfc_fusion_gate_unsup_lr) <= 0.0
            or (not bool(self.use_dorsal_stream))
        ):
            return
        lr = float(self.pfc_fusion_gate_unsup_lr)
        decay = float(self.pfc_fusion_gate_decay)
        eps = 1e-6
        if te_for_pre is not None and pp_for_pre is not None:
            te_d = te_for_pre.detach()
            pp_d = pp_for_pre.detach()
            x = torch.cat([te_d, pp_d], dim=1)
            g = torch.sigmoid(F.linear(x, self.pfc_pre_gate_W, self.pfc_pre_gate_b))
            target = te_d.abs() / (te_d.abs() + pp_d.abs() + eps)
            err = target - g
            self._pfc_fusion_lms_update_linear_(
                self.pfc_pre_gate_W,
                self.pfc_pre_gate_b,
                err,
                x,
                lr,
                decay,
                chunk_rows=int(self.pfc_fusion_lms_chunk_rows),
            )
            self._apply_pfc_l1_prox()
            self.pfc_pre_gate_W.clamp_(-4.0, 4.0)
            self.pfc_pre_gate_b.clamp_(-4.0, 4.0)
        te_r = te_readout.detach()
        pfc_d = pfc_readout.detach()
        x2 = torch.cat([te_r, pfc_d], dim=1)
        g2 = torch.sigmoid(F.linear(x2, self.pfc_post_gate_W, self.pfc_post_gate_b))
        target2 = te_r.abs() / (te_r.abs() + pfc_d.abs() + eps)
        err2 = target2 - g2
        self._pfc_fusion_lms_update_linear_(
            self.pfc_post_gate_W,
            self.pfc_post_gate_b,
            err2,
            x2,
            lr,
            decay,
            chunk_rows=int(self.pfc_fusion_lms_chunk_rows),
        )
        self._apply_pfc_l1_prox()
        self.pfc_post_gate_W.clamp_(-4.0, 4.0)
        self.pfc_post_gate_b.clamp_(-4.0, 4.0)

    def _pfc_predictive_feedback_layer_scales(
        self, pfc_features: Optional[torch.Tensor], layer_targets: Dict[str, float]
    ) -> Dict[str, float]:
        default_scales = {"l1": 1.0, "l2": 1.0, "l3": 1.0, "l4": 1.0}
        if not bool(self.use_pfc_predictive_feedback):
            return default_scales
        attn_vec = self._pfc_attention_vector(pfc_features)
        if attn_vec is None:
            return default_scales
        target = torch.tensor(
            [
                float(layer_targets.get("l1", 1.0)),
                float(layer_targets.get("l2", 1.0)),
                float(layer_targets.get("l3", 1.0)),
                float(layer_targets.get("l4", 1.0)),
            ],
            device=attn_vec.device,
            dtype=attn_vec.dtype,
        )
        target = target.clamp_min(1e-6)
        target = target / (target.sum() + 1e-8)
        pred_logits = self.pfc_predictive_feedback_W @ attn_vec + self.pfc_predictive_feedback_b
        pred = F.softmax(pred_logits, dim=0)
        err = (target - pred) * 4.0  # roughly in [-1, 1]
        scales = 1.0 + float(self.pfc_predictive_strength) * err
        scales = torch.clamp(
            scales,
            min=float(self.pfc_predictive_min_scale),
            max=float(self.pfc_predictive_max_scale),
        )
        return {
            "l1": float(scales[0].item()),
            "l2": float(scales[1].item()),
            "l3": float(scales[2].item()),
            "l4": float(scales[3].item()),
        }

    def _pfc_spatial_predictive_gain(self, pfc_state: torch.Tensor, freq_idx: int) -> Optional[torch.Tensor]:
        """
        Per-neuron spatial gain [N] from batch-mean PFC state reshaped as [F,H,W], slice ``freq_idx``.
        Mean-normalized to ~1; used with local trace surprise in ``TopographicUnifiedLayer.update_from_patches``.
        Returns None when disabled or when ``pfc_state`` is not TE-shaped [B, F*H*W].
        """
        coeffs = getattr(self, "coeffs", None)
        if coeffs is None or not bool(getattr(coeffs, "pc_per_neuron_plasticity", True)):
            return None
        wp = float(getattr(coeffs, "pc_per_neuron_pfc_weight", 0.5))
        if wp <= 0.0:
            return None
        D = int(pfc_state.size(1))
        f, h, w = int(self.num_freqs), int(self.spatial_size), int(self.spatial_size)
        fh = f * h * w
        if D != fh or freq_idx < 0 or freq_idx >= f:
            return None
        gmn = float(getattr(coeffs, "pc_per_neuron_gain_min", 0.5))
        gmx = float(getattr(coeffs, "pc_per_neuron_gain_max", 1.5))
        with torch.no_grad():
            x = pfc_state.detach().mean(0).view(f, h, w)[freq_idx].flatten().abs()
            x = x / (x.mean() + 1e-8)
            x = torch.clamp(x, gmn, gmx)
            x = x / (x.mean() + 1e-8)
            return x.to(device=pfc_state.device, dtype=pfc_state.dtype)

    def _layer_flat_spatial_plasticity_gain(self, y_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Per-neuron [N] gain from **this layer's** activations only: batch-mean |y| per map location.
        One map per layer/stream/frequency — independent of other layers' feedback.
        """
        coeffs = getattr(self, "coeffs", None)
        if coeffs is None or not bool(getattr(coeffs, "pc_per_neuron_plasticity", True)):
            return None
        wl = float(getattr(coeffs, "pc_per_neuron_layer_weight", 0.34))
        if wl <= 0.0:
            return None
        if y_flat.dim() != 2 or int(y_flat.size(0)) < 1:
            return None
        gmn = float(getattr(coeffs, "pc_per_neuron_gain_min", 0.5))
        gmx = float(getattr(coeffs, "pc_per_neuron_gain_max", 1.5))
        with torch.no_grad():
            x = y_flat.detach().abs().mean(dim=0).flatten()
            x = x / (x.mean() + 1e-8)
            x = torch.clamp(x, gmn, gmx)
            x = x / (x.mean() + 1e-8)
            return x.to(device=y_flat.device, dtype=y_flat.dtype)

    def _pfc_pc_layer_output_gates(self, pfc_state: torch.Tensor) -> torch.Tensor:
        """Per-sample multiplicative gates in (0, 1) from post-Hopfield PFC state [B, D]."""
        return torch.sigmoid(pfc_state @ self.pfc_pc_layer_mask_W + self.pfc_pc_layer_mask_b)

    @torch.no_grad()
    def _update_pfc_pc_layer_output_mask(
        self,
        pfc_state: torch.Tensor,
        l1_flats: List[torch.Tensor],
        l2_flats: List[torch.Tensor],
        l3_flats: List[torch.Tensor],
        l4_flats: List[torch.Tensor],
    ) -> None:
        """
        Predictive coding (LMS): predict normalized per-layer mean |activity| from PFC with sigmoid(Wx+b).
        Target is max-normalized per sample across the four layers so each channel is in [0, 1].
        """
        if (
            (not bool(self.use_pfc_pc_layer_output_mask))
            or float(self.pfc_pc_layer_mask_lr) <= 0.0
            or (not self.training)
        ):
            return
        B = float(pfc_state.size(0))
        if B < 1:
            return
        t1 = torch.cat(l1_flats, dim=1).abs().mean(dim=1, keepdim=True)
        t2 = torch.cat(l2_flats, dim=1).abs().mean(dim=1, keepdim=True)
        t3 = torch.cat(l3_flats, dim=1).abs().mean(dim=1, keepdim=True)
        t4 = torch.cat(l4_flats, dim=1).abs().mean(dim=1, keepdim=True)
        target = torch.cat([t1, t2, t3, t4], dim=1)
        target = target / (target.max(dim=1, keepdim=True).values.clamp_min(1e-6))
        z = pfc_state.detach()
        pred = torch.sigmoid(z @ self.pfc_pc_layer_mask_W + self.pfc_pc_layer_mask_b)
        err = target - pred
        lr = float(self.pfc_pc_layer_mask_lr)
        decay = float(self.pfc_pc_layer_mask_decay)
        self.pfc_pc_layer_mask_W.mul_(1.0 - decay).add_(z.t() @ err / B, alpha=lr)
        self.pfc_pc_layer_mask_b.mul_(1.0 - decay).add_(err.mean(dim=0), alpha=lr)
        self.pfc_pc_layer_mask_W.clamp_(-2.0, 2.0)
        self.pfc_pc_layer_mask_b.clamp_(-2.0, 2.0)

    def _apply_pfc_pc_layer_output_masks_to_flats(
        self,
        l1_flats: List[torch.Tensor],
        l2_flats: List[torch.Tensor],
        l3_flats: List[torch.Tensor],
        l4_flats: List[torch.Tensor],
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """In-place multiply each ventral layer flat by its gate [B,1]. Returns updated TE flat."""
        g1, g2, g3, g4 = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3], gates[:, 3:4]
        for i in range(len(l1_flats)):
            l1_flats[i].mul_(g1)
            l2_flats[i].mul_(g2)
            l3_flats[i].mul_(g3)
            l4_flats[i].mul_(g4)
        return torch.cat(l4_flats, dim=1)

    def _maybe_pfc_pc_mask_layer_outputs(
        self,
        pfc_state: torch.Tensor,
        l1_flats: List[torch.Tensor],
        l2_flats: List[torch.Tensor],
        l3_flats: List[torch.Tensor],
        l4_flats: List[torch.Tensor],
    ) -> torch.Tensor:
        """PC-LMS update (train) + apply gates to ventral L1–L4; returns TE flat for downstream readout."""
        if not bool(self.use_pfc_pc_layer_output_mask):
            return torch.cat(l4_flats, dim=1)
        if self.training and float(self.pfc_pc_layer_mask_lr) > 0.0:
            self._update_pfc_pc_layer_output_mask(pfc_state, l1_flats, l2_flats, l3_flats, l4_flats)
        gates = self._pfc_pc_layer_output_gates(pfc_state)
        return self._apply_pfc_pc_layer_output_masks_to_flats(l1_flats, l2_flats, l3_flats, l4_flats, gates)

    def _pfc_spatial_attention_map(self, stream_flat: torch.Tensor) -> torch.Tensor:
        """Map flat PFC-sized stream [B, F*H*W] -> spatial saliency [B, 1, H, W] in ~[0, 1]."""
        B, D = stream_flat.shape
        f, h, w = self.num_freqs, self.spatial_size, self.spatial_size
        if D != f * h * w:
            raise ValueError(f"Expected flat dim {f * h * w}, got {D}")
        x = stream_flat.view(B, f, h, w)
        m = x.detach().abs().mean(dim=1, keepdim=True)
        return m / (m.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))

    def _apply_spatial_gate_to_flat(self, stream_flat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Multiply each frequency plane by the same spatial gate (blend with identity using strength/floor)."""
        B, D = stream_flat.shape
        f, h, w = self.num_freqs, self.spatial_size, self.spatial_size
        if gate.shape[2:] != (h, w):
            gate = F.interpolate(gate, size=(h, w), mode="bilinear", align_corners=False)
        s = float(self.pfc_spatial_gate_strength)
        fl = float(self.pfc_spatial_gate_floor)
        g = (1.0 - s) + s * (fl + (1.0 - fl) * gate)
        x = stream_flat.view(B, f, h, w) * g
        return x.view(B, D)

    def _apply_pfc_spatial_readout_gate(
        self, te_flat: torch.Tensor, pfc_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-down spatial readout bias: suppress locations where PFC state is weak (background proxy).
        Attention is derived from the post-Hopfield PFC map (dorsal: PP; ventral-only: TE).
        """
        if not self.use_pfc_spatial_readout_gate or float(self.pfc_spatial_gate_strength) <= 0.0:
            return te_flat, pfc_state
        attn = self._pfc_spatial_attention_map(pfc_state)
        te_g = self._apply_spatial_gate_to_flat(te_flat, attn)
        pfc_g = self._apply_spatial_gate_to_flat(pfc_state, attn)
        return te_g, pfc_g

    def _pfc_fuse_te_pp_pre_hopfield(self, te_flat: torch.Tensor, pp_flat: torch.Tensor) -> torch.Tensor:
        """Hopfield input [B,D] from ventral TE and dorsal PP (same flat dim)."""
        mode = self.pfc_pre_hopfield_fusion
        if mode == "pp_only":
            return pp_flat
        wte, wpp = self.pfc_pre_blend_w_te, self.pfc_pre_blend_w_pp
        s = wte + wpp
        blend = (wte * te_flat + wpp * pp_flat) / s if s > 0.0 else pp_flat
        cat = torch.cat([te_flat, pp_flat], dim=1)
        g_pre = torch.sigmoid(F.linear(cat, self.pfc_pre_gate_W, self.pfc_pre_gate_b))
        gate = g_pre * te_flat + (1.0 - g_pre) * pp_flat
        if mode == "blend":
            return blend
        if mode == "gate":
            return gate
        if mode == "all":
            return (pp_flat + blend + gate) / 3.0
        return gate

    def _pfc_fuse_te_pfc_post_readout(self, te_flat: torch.Tensor, pfc_state: torch.Tensor) -> torch.Tensor:
        """Second half of dorsal classifier input [B,D]: raw post-Hopfield PFC or gated mix with TE."""
        mode = self.pfc_post_readout_fusion
        if mode == "concat":
            return pfc_state
        cat = torch.cat([te_flat, pfc_state], dim=1)
        g = torch.sigmoid(F.linear(cat, self.pfc_post_gate_W, self.pfc_post_gate_b))
        gated = g * te_flat + (1.0 - g) * pfc_state
        if mode == "gate":
            return gated
        if mode == "all":
            return 0.5 * (pfc_state + gated)
        return gated

    def _pfc_classifier_readout(self, te_flat: torch.Tensor, pfc_state: torch.Tensor) -> torch.Tensor:
        """Classifier input: ventral-only [B,D] or dorsal ``concat(TE, fused_pfc_slot)`` [B,2D]."""
        if not self.use_dorsal_stream:
            return pfc_state
        second = self._pfc_fuse_te_pfc_post_readout(te_flat, pfc_state)
        return torch.cat([te_flat, second], dim=1)

    def _multiply_flat_by_spatial_attention(
        self, stream_flat: torch.Tensor, attn_map: torch.Tensor, strength: float
    ) -> torch.Tensor:
        """Reshape flat [B,F*H*W] -> maps, multiply by (1 + strength * attn), same layout as TE/PP."""
        if strength <= 0.0:
            return stream_flat
        B, D = stream_flat.shape
        f, h, w = self.num_freqs, self.spatial_size, self.spatial_size
        g = attn_map
        if g.shape[2:] != (h, w):
            g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        x = stream_flat.view(B, f, h, w) * (1.0 + strength * g)
        return x.view(B, D)

    def _it_pp_bidirectional_cross_gate(
        self, te_flat: torch.Tensor, pp_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Explicit IT↔PP coupling: TE (ventral L4) gains spatial emphasis from PP saliency; PP gains from TE.
        Maps are fixed (mean-abs per location); no backprop through the routing rule.
        """
        if not self.use_dorsal_stream or not self.it_pp_cross_gate:
            return te_flat, pp_flat
        a = float(self.it_pp_cross_pp_to_te)
        b = float(self.it_pp_cross_te_to_pp)
        if a <= 0.0 and b <= 0.0:
            return te_flat, pp_flat
        te = te_flat
        pp = pp_flat
        n_it = int(max(1, self.it_pp_cross_iters))
        for _ in range(n_it):
            m_pp = self._pfc_spatial_attention_map(pp.detach())
            m_te = self._pfc_spatial_attention_map(te.detach())
            if a > 0.0:
                te = self._multiply_flat_by_spatial_attention(te, m_pp, a)
            if b > 0.0:
                pp = self._multiply_flat_by_spatial_attention(pp, m_te, b)
        return te, pp

    def _modulate_freq_inputs_pfc_feedback(
        self, freq_inputs: List[torch.Tensor], attn_map: torch.Tensor
    ) -> List[torch.Tensor]:
        """Spatially gate L1 inputs using PFC-derived saliency (second recurrent pass)."""
        s = float(self.pfc_recurrent_feedback_strength)
        if s <= 0.0:
            return freq_inputs
        out: List[torch.Tensor] = []
        for t in freq_inputs:
            g = attn_map
            if g.shape[2:] != t.shape[2:]:
                g = F.interpolate(g, size=t.shape[2:], mode="bilinear", align_corners=False)
            out.append(t * (1.0 + s * g))
        return out

    def _apply_pfc_dense_feedback_to_freq_inputs(
        self, freq_inputs: List[torch.Tensor], pfc_state: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Explicit top-down: add a spatial map derived from the post-Hopfield PFC state to each L1
        frequency band (broadcast across all input channels). Uses detached PFC; no gradient into PFC.
        """
        alpha = float(self.pfc_dense_feedback_strength)
        if alpha <= 0.0:
            return freq_inputs
        B, D = pfc_state.shape
        f, h, w = self.num_freqs, self.spatial_size, self.spatial_size
        if D != f * h * w:
            raise ValueError(
                f"PFC dense feedback expects flat dim {f * h * w} (F*H*W), got {D}"
            )
        m = pfc_state.detach().view(B, f, h, w)
        m = m / (m.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
        out: List[torch.Tensor] = []
        for i in range(f):
            g = m[:, i : i + 1, :, :]
            if g.shape[2:] != freq_inputs[i].shape[2:]:
                g = F.interpolate(g, size=freq_inputs[i].shape[2:], mode="bilinear", align_corners=False)
            out.append(freq_inputs[i] + alpha * g)
        return out

    def _pfc_deep_feedback_any_strength(self) -> bool:
        return bool(self.use_pfc_deep_feedback) and any(
            float(x) > 0.0
            for x in (
                self.pfc_deep_fb_strength_l2,
                self.pfc_deep_fb_strength_l3,
                self.pfc_deep_fb_strength_l4,
                self.pfc_deep_fb_strength_mt,
                self.pfc_deep_fb_strength_pp,
            )
        )

    def _pfc_run_second_ventral_stack(self, td_i: int) -> bool:
        """Whether to re-run L1–L4 (+ dorsal) after PFC refinement on the last top-down iteration."""
        if td_i != self.pfc_topdown_iters - 1:
            return False
        if bool(self.use_pfc_dense_feedback) and float(self.pfc_dense_feedback_strength) > 0.0:
            return True
        if self._pfc_deep_feedback_any_strength():
            return True
        return int(self.pfc_recurrent_feedback_steps) >= 2 and float(self.pfc_recurrent_feedback_strength) > 0.0

    def _build_pfc_second_pass_freq_inputs(
        self, freq_inputs: List[torch.Tensor], pfc_state: torch.Tensor
    ) -> List[torch.Tensor]:
        """Optional multiplicative saliency then additive PFC→L1 dense feedback."""
        freq_fb = freq_inputs
        if int(self.pfc_recurrent_feedback_steps) >= 2 and float(self.pfc_recurrent_feedback_strength) > 0.0:
            attn_fb = self._pfc_spatial_attention_map(pfc_state)
            freq_fb = self._modulate_freq_inputs_pfc_feedback(freq_fb, attn_fb)
        if bool(self.use_pfc_dense_feedback) and float(self.pfc_dense_feedback_strength) > 0.0:
            freq_fb = self._apply_pfc_dense_feedback_to_freq_inputs(freq_fb, pfc_state)
        return freq_fb

    def _pfc_feedback_lists_from_spatial_maps(self, m: torch.Tensor) -> List[torch.Tensor]:
        """Split [B,F,H,W] into per-frequency [B,1,H,W] maps."""
        return [m[:, i : i + 1, :, :] for i in range(self.num_freqs)]

    def _pfc_feedback_state_for_second_pass(self, pfc_state: torch.Tensor) -> Optional[torch.Tensor]:
        """Post-Hopfield PFC vector for deep additive feedback on the second stack (no grad)."""
        if not self._pfc_deep_feedback_any_strength():
            return None
        return pfc_state.detach()

    def _pfc_deep_feedback_spatial_maps(self, pfc_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Bottleneck: flat PFC [B,D] -> z [B,r] -> separate U_head -> [B,F,H,W] per target stage.
        Detached; no gradient through feedback weights.
        """
        B, D = pfc_state.shape
        d_exp = self.num_freqs * self.spatial_size * self.spatial_size
        if D != d_exp:
            raise ValueError(f"PFC deep feedback expects flat dim {d_exp}, got {D}")
        ps = pfc_state.detach()
        z = torch.tanh(ps @ self.pfc_fb_Wd + self.pfc_fb_bd)
        out: Dict[str, torch.Tensor] = {}
        f, h, w = self.num_freqs, self.spatial_size, self.spatial_size
        for name in ("l2", "l3", "l4", "mt", "pp"):
            U = getattr(self, f"pfc_fb_U_{name}")
            raw = (z @ U).view(B, f, h, w)
            raw = raw / (raw.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
            out[name] = raw
        return out

    def _run_ventral_dorsal_stack(
        self,
        freq_inputs: List[torch.Tensor],
        L: torch.Tensor,
        pfc_feedback_state: Optional[torch.Tensor] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        torch.Tensor,
    ]:
        """One full ventral L1–L4 pass plus optional dorsal MT→PP; returns tensors for PFC / plasticity."""
        l2_fb: Optional[List[torch.Tensor]] = None
        l3_fb: Optional[List[torch.Tensor]] = None
        l4_fb: Optional[List[torch.Tensor]] = None
        mt_fb: Optional[List[torch.Tensor]] = None
        pp_fb: Optional[List[torch.Tensor]] = None
        if pfc_feedback_state is not None and self.use_pfc_deep_feedback and self._pfc_deep_feedback_any_strength():
            dm = self._pfc_deep_feedback_spatial_maps(pfc_feedback_state)
            if self.pfc_deep_fb_strength_l2 > 0.0:
                l2_fb = self._pfc_feedback_lists_from_spatial_maps(dm["l2"] * self.pfc_deep_fb_strength_l2)
            if self.pfc_deep_fb_strength_l3 > 0.0:
                l3_fb = self._pfc_feedback_lists_from_spatial_maps(dm["l3"] * self.pfc_deep_fb_strength_l3)
            if self.pfc_deep_fb_strength_l4 > 0.0:
                l4_fb = self._pfc_feedback_lists_from_spatial_maps(dm["l4"] * self.pfc_deep_fb_strength_l4)
            if self.use_dorsal_stream:
                if self.pfc_deep_fb_strength_mt > 0.0:
                    mt_fb = self._pfc_feedback_lists_from_spatial_maps(dm["mt"] * self.pfc_deep_fb_strength_mt)
                if self.pfc_deep_fb_strength_pp > 0.0:
                    pp_fb = self._pfc_feedback_lists_from_spatial_maps(dm["pp"] * self.pfc_deep_fb_strength_pp)

        l1_outputs, l1_flats, l1_patches, l2_outputs, l2_flats, l2_patches = self._ventral_l1_l2(
            freq_inputs, self.l1_freq_layers, self.l2_freq_layers, l2_pfc_feedback=l2_fb
        )
        l3_inputs = self._l3_inputs_from_l12(l1_outputs, l2_outputs)
        _l3o, l3_flats, l3_patches, _l4o, l4_flats, l4_patches = self._ventral_l3_l4(
            l3_inputs,
            l1_outputs,
            l2_outputs,
            self.l3_freq_layers,
            self.l4_freq_layers,
            l3_pfc_feedback=l3_fb,
            l4_pfc_feedback=l4_fb,
        )
        te_flat = torch.cat(l4_flats, dim=1)
        mt_flats: Optional[List[torch.Tensor]] = None
        mt_patches: Optional[List[torch.Tensor]] = None
        pp_flats: Optional[List[torch.Tensor]] = None
        pp_patches: Optional[List[torch.Tensor]] = None
        if self.use_dorsal_stream:
            L_map = L
            if L_map.size(2) != l3_inputs[0].size(2) or L_map.size(3) != l3_inputs[0].size(3):
                L_map = F.interpolate(
                    L_map,
                    size=(l3_inputs[0].size(2), l3_inputs[0].size(3)),
                    mode="bilinear",
                    align_corners=False,
                )
            spatial_gate = self._dorsal_spatial_gate_map(L_map)
            mt_inputs = self._mt_inputs_from_l3_inputs(l3_inputs, spatial_gate)
            mt_flats, mt_patches, pp_flats, pp_patches = self._run_mt_pp_from_l12(
                mt_inputs, l1_outputs, l2_outputs, mt_pfc_feedback=mt_fb, pp_pfc_feedback=pp_fb
            )
            pp_flat = torch.cat(pp_flats, dim=1)
            pfc_state = (
                pp_flat
                if self.pfc_pre_hopfield_fusion == "pp_only"
                else self._pfc_fuse_te_pp_pre_hopfield(te_flat, pp_flat)
            )
        else:
            pfc_state = te_flat
        return (
            l1_outputs,
            l1_flats,
            l1_patches,
            l2_outputs,
            l2_flats,
            l2_patches,
            l3_inputs,
            l3_flats,
            l3_patches,
            l4_flats,
            l4_patches,
            te_flat,
            mt_flats,
            mt_patches,
            pp_flats,
            pp_patches,
            pfc_state,
        )

    def _rgb_to_freq_inputs(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Opponent → Gabor; optionally concat Haar wavelet bands per stream (27 ch) or Gabor-only (24 ch)."""
        x_opp = DoGRGB.rgb_to_opponent(x)
        L = x_opp[:, 0:1]
        g = self.gabor(L)
        if self.auto_resize_input and (g.size(2) != self.spatial_size or g.size(3) != self.spatial_size):
            g = F.interpolate(g, size=(self.spatial_size, self.spatial_size), mode="bilinear", align_corners=False)
        gabor_bands = torch.chunk(g, self.num_freqs, dim=1)
        if not self.use_wavelet_input:
            return list(gabor_bands), L
        assert self.wavelet is not None
        w = self.wavelet(x_opp)
        if w.size(2) != self.spatial_size or w.size(3) != self.spatial_size:
            w = F.interpolate(w, size=(self.spatial_size, self.spatial_size), mode="bilinear", align_corners=False)
        wavelet_base = torch.chunk(w, 4, dim=1)
        wavelet_bands = [wavelet_base[i % 4] for i in range(self.num_freqs)]
        freq_inputs = [torch.cat([gb, wb], dim=1) for gb, wb in zip(gabor_bands, wavelet_bands)]
        return freq_inputs, L

    def forward(
        self,
        x: torch.Tensor,
        local_update: bool = True,
        plasticity_gain_vec: Optional[torch.Tensor] = None,
        layer_scale_vec: Optional[torch.Tensor] = None,
        readout_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optional ``plasticity_gain_vec`` [4] scales local Hebbian gains (L1–L4) for one forward.
        Optional ``layer_scale_vec`` [4] multiplies detached L1–L4 flats (per stream) before the PC
        mask and readout so a meta-network can receive CE gradients for each layer without backbone BP.
        Optional scalar ``readout_scale`` multiplies detached readout before the classifier."""
        if x.size(1) != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {x.size(1)}")

        freq_inputs, L = self._rgb_to_freq_inputs(x)

        (
            l1_outputs,
            l1_flats,
            l1_patches,
            l2_outputs,
            l2_flats,
            l2_patches,
            _l3_inputs_unused,
            l3_flats,
            l3_patches,
            l4_flats,
            l4_patches,
            te_flat,
            mt_flats,
            mt_patches,
            pp_flats,
            pp_patches,
            pfc_state,
        ) = self._run_ventral_dorsal_stack(freq_inputs, L)

        fusion_te_pre = te_flat
        fusion_pp_pre = torch.cat(pp_flats, dim=1) if self.use_dorsal_stream and pp_flats is not None else None
        if self.use_dorsal_stream:
            te_flat, pfc_state = self._it_pp_bidirectional_cross_gate(te_flat, pfc_state)

        l1_activity = float(torch.cat(l1_flats, dim=1).detach().abs().mean().item())
        l2_activity = float(torch.cat(l2_flats, dim=1).detach().abs().mean().item())
        l3_activity = float(torch.cat(l3_flats, dim=1).detach().abs().mean().item())
        for td_i in range(self.pfc_topdown_iters):
            if self.pfc_hopfield is not None and bool(self.use_pfc_hopfield):
                pfc_state = self.pfc_hopfield(pfc_state, update_memory=bool(local_update and self.training))
            if self._pfc_run_second_ventral_stack(td_i):
                freq_fb = self._build_pfc_second_pass_freq_inputs(freq_inputs, pfc_state)
                (
                    l1_outputs,
                    l1_flats,
                    l1_patches,
                    l2_outputs,
                    l2_flats,
                    l2_patches,
                    _,
                    l3_flats,
                    l3_patches,
                    l4_flats,
                    l4_patches,
                    te_flat,
                    mt_flats,
                    mt_patches,
                    pp_flats,
                    pp_patches,
                    pfc_state,
                ) = self._run_ventral_dorsal_stack(
                    freq_fb, L, self._pfc_feedback_state_for_second_pass(pfc_state)
                )
                fusion_te_pre = te_flat
                fusion_pp_pre = (
                    torch.cat(pp_flats, dim=1) if self.use_dorsal_stream and pp_flats is not None else None
                )
                if self.use_dorsal_stream:
                    te_flat, pfc_state = self._it_pp_bidirectional_cross_gate(te_flat, pfc_state)
                l1_activity = float(torch.cat(l1_flats, dim=1).detach().abs().mean().item())
                l2_activity = float(torch.cat(l2_flats, dim=1).detach().abs().mean().item())
                l3_activity = float(torch.cat(l3_flats, dim=1).detach().abs().mean().item())
            td_scales = (
                self._pfc_neuron_topdown_scales(pfc_state)
                if getattr(self, "use_pfc_topdown_per_neuron", True)
                else self._pfc_topdown_layer_scales(pfc_state)
            )
            inh_scales = self._pfc_inhibition_layer_scales(pfc_state)
            l4_activity = float(pfc_state.detach().abs().mean().item())
            layer_targets = self._blend_layer_targets_for_pfc_unsup(
                l1_activity, l2_activity, l3_activity, l4_activity
            )
            pred_scales = self._pfc_predictive_feedback_layer_scales(pfc_state, layer_targets)
            if getattr(self, "use_pfc_topdown_per_neuron", True):
                td_total = {k: td_scales[k] * float(pred_scales[k]) for k in ("l1", "l2", "l3", "l4")}
            else:
                td_total = {k: float(td_scales[k] * pred_scales[k]) for k in ("l1", "l2", "l3", "l4")}
            inh_total = {k: float(inh_scales[k] * pred_scales[k]) for k in ("l1", "l2", "l3", "l4")}
            if plasticity_gain_vec is not None:
                pg = plasticity_gain_vec.view(4).detach()
                for ik, k in enumerate(("l1", "l2", "l3", "l4")):
                    g = float(pg[ik].item())
                    td_total[k] = td_total[k] * g
                    inh_total[k] = inh_total[k] * g
            if td_i == self.pfc_topdown_iters - 1:
                _glia_g = self._neuron_glia_per_layer_gates(layer_targets, update_trace=True)
                for ik, k in enumerate(("l1", "l2", "l3", "l4")):
                    td_total[k] = td_total[k] * _glia_g[ik]
                    inh_total[k] = inh_total[k] * _glia_g[ik]
            attn_vec = self._pfc_attention_vector(pfc_state)
            self._update_pfc_neuron_topdown_unsup(pfc_state, l1_flats, l2_flats, l3_flats, l4_flats)
            self._update_pfc_topdown_unsup(
                attn_vec,
                layer_targets,
            )
            self._update_pfc_inhibition_feedback_unsup(
                attn_vec,
                layer_targets,
            )
            self._update_pfc_predictive_feedback_unsup(attn_vec, layer_targets)
            # One local plasticity step per batch, after final top-down pass (gates + Hopfield state).
            if local_update and self.training and td_i == self.pfc_topdown_iters - 1:
                _n = int(self.spatial_size * self.spatial_size)
                for i in range(self.num_freqs):
                    sl = slice(i * _n, (i + 1) * _n)
                    _pg = self._pfc_spatial_predictive_gain(pfc_state, i)
                    if getattr(self, "use_pfc_topdown_per_neuron", True):
                        v1, v2, v3, v4 = (
                            td_total["l1"][sl],
                            td_total["l2"][sl],
                            td_total["l3"][sl],
                            td_total["l4"][sl],
                        )
                    else:
                        v1 = v2 = v3 = v4 = None
                    self.l1_freq_layers[i].update_from_patches(
                        l1_patches[i],
                        l1_flats[i],
                        variance_scale=v1 if v1 is not None else td_total["l1"],
                        inhibition_scale=inh_total["l1"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l1_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    self.l2_freq_layers[i].update_from_patches(
                        l2_patches[i],
                        l2_flats[i],
                        variance_scale=v2 if v2 is not None else td_total["l2"],
                        inhibition_scale=inh_total["l2"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l2_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    self.l3_freq_layers[i].update_from_patches(
                        l3_patches[i],
                        l3_flats[i],
                        variance_scale=v3 if v3 is not None else td_total["l3"],
                        inhibition_scale=inh_total["l3"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l3_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    self.l4_freq_layers[i].update_from_patches(
                        l4_patches[i],
                        l4_flats[i],
                        variance_scale=v4 if v4 is not None else td_total["l4"],
                        inhibition_scale=inh_total["l4"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l4_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    if self.use_dorsal_stream and mt_patches is not None and pp_patches is not None:
                        self.mt_freq_layers[i].update_from_patches(
                            mt_patches[i],
                            mt_flats[i],
                            variance_scale=v3 if v3 is not None else td_total["l3"],
                            inhibition_scale=inh_total["l3"],
                            layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(mt_flats[i]),
                            pfc_spatial_gain=_pg,
                        )
                        self.pp_freq_layers[i].update_from_patches(
                            pp_patches[i],
                            pp_flats[i],
                            variance_scale=v4 if v4 is not None else td_total["l4"],
                            inhibition_scale=inh_total["l4"],
                            layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(pp_flats[i]),
                            pfc_spatial_gain=_pg,
                        )
                self._maybe_apply_local_l1_prox()
        if layer_scale_vec is not None:
            ls = layer_scale_vec.view(4).to(device=l1_flats[0].device, dtype=l1_flats[0].dtype)
            for i in range(self.num_freqs):
                l1_flats[i] = l1_flats[i].detach() * ls[0]
                l2_flats[i] = l2_flats[i].detach() * ls[1]
                l3_flats[i] = l3_flats[i].detach() * ls[2]
                l4_flats[i] = l4_flats[i].detach() * ls[3]
            te_flat = torch.cat(l4_flats, dim=1)
        if bool(self.use_pfc_pc_layer_output_mask):
            te_flat = self._maybe_pfc_pc_mask_layer_outputs(pfc_state, l1_flats, l2_flats, l3_flats, l4_flats)
            if self.use_dorsal_stream:
                te_flat, pfc_state = self._it_pp_bidirectional_cross_gate(te_flat, pfc_state)
        te_flat, pfc_state = self._apply_pfc_spatial_readout_gate(te_flat, pfc_state)
        self._update_pfc_fusion_gates_unsup(fusion_te_pre, fusion_pp_pre, te_flat, pfc_state)
        readout = self._pfc_classifier_readout(te_flat, pfc_state)
        if readout_scale is not None:
            rs = readout_scale.view(()).to(device=readout.device, dtype=readout.dtype)
            l4_combined = self.dropout(readout.detach() * rs)
            return self.clf(l4_combined)
        l4_combined = self.dropout(readout)

        # Only the classifier is trained by gradient descent. L1–L4 use local plasticity only.
        # Detach so no gradients flow into L1–L4 when training the classifier.
        return self.clf(l4_combined.detach() if self.training else l4_combined)

    def forward_features(self, x: torch.Tensor, local_update: bool = True) -> torch.Tensor:
        """Run the full pipeline and return fused readout features (ventral TE + optional PP).
        Shape is ``[B, num_freqs*spatial²]`` or doubled when the dorsal (MT→PP) path is enabled.
        Used for serial stacking: first VisNet outputs these features for the second VisNet.
        """
        if x.size(1) != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {x.size(1)}")

        freq_inputs, L = self._rgb_to_freq_inputs(x)

        (
            l1_outputs,
            l1_flats,
            l1_patches,
            l2_outputs,
            l2_flats,
            l2_patches,
            _,
            l3_flats,
            l3_patches,
            l4_flats,
            l4_patches,
            te_flat,
            mt_flats,
            mt_patches,
            pp_flats,
            pp_patches,
            pfc_state,
        ) = self._run_ventral_dorsal_stack(freq_inputs, L)

        fusion_te_pre = te_flat
        fusion_pp_pre = torch.cat(pp_flats, dim=1) if self.use_dorsal_stream and pp_flats is not None else None
        if self.use_dorsal_stream:
            te_flat, pfc_state = self._it_pp_bidirectional_cross_gate(te_flat, pfc_state)

        l1_activity = float(torch.cat(l1_flats, dim=1).detach().abs().mean().item())
        l2_activity = float(torch.cat(l2_flats, dim=1).detach().abs().mean().item())
        l3_activity = float(torch.cat(l3_flats, dim=1).detach().abs().mean().item())
        for td_i in range(self.pfc_topdown_iters):
            if self.pfc_hopfield is not None and bool(self.use_pfc_hopfield):
                pfc_state = self.pfc_hopfield(pfc_state, update_memory=bool(local_update and self.training))
            if self._pfc_run_second_ventral_stack(td_i):
                freq_fb = self._build_pfc_second_pass_freq_inputs(freq_inputs, pfc_state)
                (
                    l1_outputs,
                    l1_flats,
                    l1_patches,
                    l2_outputs,
                    l2_flats,
                    l2_patches,
                    _,
                    l3_flats,
                    l3_patches,
                    l4_flats,
                    l4_patches,
                    te_flat,
                    mt_flats,
                    mt_patches,
                    pp_flats,
                    pp_patches,
                    pfc_state,
                ) = self._run_ventral_dorsal_stack(
                    freq_fb, L, self._pfc_feedback_state_for_second_pass(pfc_state)
                )
                fusion_te_pre = te_flat
                fusion_pp_pre = (
                    torch.cat(pp_flats, dim=1) if self.use_dorsal_stream and pp_flats is not None else None
                )
                if self.use_dorsal_stream:
                    te_flat, pfc_state = self._it_pp_bidirectional_cross_gate(te_flat, pfc_state)
                l1_activity = float(torch.cat(l1_flats, dim=1).detach().abs().mean().item())
                l2_activity = float(torch.cat(l2_flats, dim=1).detach().abs().mean().item())
                l3_activity = float(torch.cat(l3_flats, dim=1).detach().abs().mean().item())
            td_scales = (
                self._pfc_neuron_topdown_scales(pfc_state)
                if getattr(self, "use_pfc_topdown_per_neuron", True)
                else self._pfc_topdown_layer_scales(pfc_state)
            )
            inh_scales = self._pfc_inhibition_layer_scales(pfc_state)
            l4_activity = float(pfc_state.detach().abs().mean().item())
            layer_targets = self._blend_layer_targets_for_pfc_unsup(
                l1_activity, l2_activity, l3_activity, l4_activity
            )
            pred_scales = self._pfc_predictive_feedback_layer_scales(pfc_state, layer_targets)
            if getattr(self, "use_pfc_topdown_per_neuron", True):
                td_total = {k: td_scales[k] * float(pred_scales[k]) for k in ("l1", "l2", "l3", "l4")}
            else:
                td_total = {k: float(td_scales[k] * pred_scales[k]) for k in ("l1", "l2", "l3", "l4")}
            inh_total = {k: float(inh_scales[k] * pred_scales[k]) for k in ("l1", "l2", "l3", "l4")}
            if td_i == self.pfc_topdown_iters - 1:
                _glia_g = self._neuron_glia_per_layer_gates(layer_targets, update_trace=True)
                for ik, k in enumerate(("l1", "l2", "l3", "l4")):
                    td_total[k] = td_total[k] * _glia_g[ik]
                    inh_total[k] = inh_total[k] * _glia_g[ik]
            attn_vec = self._pfc_attention_vector(pfc_state)
            self._update_pfc_neuron_topdown_unsup(pfc_state, l1_flats, l2_flats, l3_flats, l4_flats)
            self._update_pfc_topdown_unsup(
                attn_vec,
                layer_targets,
            )
            self._update_pfc_inhibition_feedback_unsup(
                attn_vec,
                layer_targets,
            )
            self._update_pfc_predictive_feedback_unsup(attn_vec, layer_targets)
            if local_update and self.training and td_i == self.pfc_topdown_iters - 1:
                _n = int(self.spatial_size * self.spatial_size)
                for i in range(self.num_freqs):
                    sl = slice(i * _n, (i + 1) * _n)
                    _pg = self._pfc_spatial_predictive_gain(pfc_state, i)
                    if getattr(self, "use_pfc_topdown_per_neuron", True):
                        v1, v2, v3, v4 = (
                            td_total["l1"][sl],
                            td_total["l2"][sl],
                            td_total["l3"][sl],
                            td_total["l4"][sl],
                        )
                    else:
                        v1 = v2 = v3 = v4 = None
                    self.l1_freq_layers[i].update_from_patches(
                        l1_patches[i],
                        l1_flats[i],
                        variance_scale=v1 if v1 is not None else td_total["l1"],
                        inhibition_scale=inh_total["l1"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l1_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    self.l2_freq_layers[i].update_from_patches(
                        l2_patches[i],
                        l2_flats[i],
                        variance_scale=v2 if v2 is not None else td_total["l2"],
                        inhibition_scale=inh_total["l2"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l2_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    self.l3_freq_layers[i].update_from_patches(
                        l3_patches[i],
                        l3_flats[i],
                        variance_scale=v3 if v3 is not None else td_total["l3"],
                        inhibition_scale=inh_total["l3"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l3_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    self.l4_freq_layers[i].update_from_patches(
                        l4_patches[i],
                        l4_flats[i],
                        variance_scale=v4 if v4 is not None else td_total["l4"],
                        inhibition_scale=inh_total["l4"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l4_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    if self.use_dorsal_stream and mt_patches is not None and pp_patches is not None:
                        self.mt_freq_layers[i].update_from_patches(
                            mt_patches[i],
                            mt_flats[i],
                            variance_scale=v3 if v3 is not None else td_total["l3"],
                            inhibition_scale=inh_total["l3"],
                            layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(mt_flats[i]),
                            pfc_spatial_gain=_pg,
                        )
                        self.pp_freq_layers[i].update_from_patches(
                            pp_patches[i],
                            pp_flats[i],
                            variance_scale=v4 if v4 is not None else td_total["l4"],
                            inhibition_scale=inh_total["l4"],
                            layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(pp_flats[i]),
                            pfc_spatial_gain=_pg,
                        )
                self._maybe_apply_local_l1_prox()
        if bool(self.use_pfc_pc_layer_output_mask):
            te_flat = self._maybe_pfc_pc_mask_layer_outputs(pfc_state, l1_flats, l2_flats, l3_flats, l4_flats)
            if self.use_dorsal_stream:
                te_flat, pfc_state = self._it_pp_bidirectional_cross_gate(te_flat, pfc_state)
        te_flat, pfc_state = self._apply_pfc_spatial_readout_gate(te_flat, pfc_state)
        self._update_pfc_fusion_gates_unsup(fusion_te_pre, fusion_pp_pre, te_flat, pfc_state)
        readout = self._pfc_classifier_readout(te_flat, pfc_state)
        return self.dropout(readout)

    def count_parameters(self) -> Dict[str, int]:
        # Count parameters by component and report gradient vs local-learning split.
        counts: Dict[str, int] = {}

        def nparams(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        # Fixed preprocessing kernels are buffers; count them as "fixed" for reporting
        gabor_fixed = self.gabor.real_w.numel() + self.gabor.imag_w.numel()
        counts["DoG_fixed"] = 0  # no DoG in this variant (12-orientation Gabor on luminance)
        counts["Gabor_fixed"] = gabor_fixed
        counts["Wavelet_fixed"] = 0  # Wavelet2D is parameter-free (Haar filters)

        # Local layers (no-grad weights) - sum across all frequency channels (ventral + optional dorsal)
        def _sum_wb(layers: nn.ModuleList) -> int:
            return sum(layer.W.numel() + layer.b.numel() for layer in layers)

        def _sum_lat(layers: nn.ModuleList) -> int:
            return sum(layer.L.numel() for layer in layers)

        def _sum_gen(layers: nn.ModuleList) -> int:
            return sum(layer.G.numel() for layer in layers)

        counts["L1_Wb"] = _sum_wb(self.l1_freq_layers)
        counts["L2_Wb"] = _sum_wb(self.l2_freq_layers)
        counts["L3_Wb"] = _sum_wb(self.l3_freq_layers) + (_sum_wb(self.mt_freq_layers) if self.use_dorsal_stream else 0)
        counts["L4_Wb"] = _sum_wb(self.l4_freq_layers) + (_sum_wb(self.pp_freq_layers) if self.use_dorsal_stream else 0)

        counts["L1_lateral"] = _sum_lat(self.l1_freq_layers)
        counts["L2_lateral"] = _sum_lat(self.l2_freq_layers)
        counts["L3_lateral"] = _sum_lat(self.l3_freq_layers) + (
            _sum_lat(self.mt_freq_layers) if self.use_dorsal_stream else 0
        )
        counts["L4_lateral"] = _sum_lat(self.l4_freq_layers) + (
            _sum_lat(self.pp_freq_layers) if self.use_dorsal_stream else 0
        )

        counts["L1_generator"] = _sum_gen(self.l1_freq_layers)
        counts["L2_generator"] = _sum_gen(self.l2_freq_layers)
        counts["L3_generator"] = _sum_gen(self.l3_freq_layers) + (
            _sum_gen(self.mt_freq_layers) if self.use_dorsal_stream else 0
        )
        counts["L4_generator"] = _sum_gen(self.l4_freq_layers) + (
            _sum_gen(self.pp_freq_layers) if self.use_dorsal_stream else 0
        )

        # Gradient-based classifier
        counts["Classifier_grad"] = self.clf.weight.numel() + self.clf.bias.numel()
        counts["PFC_grad"] = nparams(self.pfc_hopfield) if self.pfc_hopfield is not None else 0
        counts["PFC_topdown_local"] = (
            self.pfc_topdown_W.numel()
            + self.pfc_topdown_b.numel()
            + self.pfc_topdown_neuron_w.numel()
            + (self.pfc_topdown_neuron_b.numel() if self.pfc_topdown_neuron_b is not None else 0)
            + self.pfc_pre_gate_W.numel()
            + self.pfc_pre_gate_b.numel()
            + self.pfc_post_gate_W.numel()
            + self.pfc_post_gate_b.numel()
        )
        counts["PFC_predictive_local"] = (
            self.pfc_predictive_feedback_W.numel()
            + self.pfc_predictive_feedback_b.numel()
            + self.pfc_pc_layer_mask_W.numel()
            + self.pfc_pc_layer_mask_b.numel()
        )

        total = sum(counts.values())
        grad = counts["Classifier_grad"] + counts["PFC_grad"]
        counts["TOTAL"] = total
        counts["Grad_%"] = float(100.0 * grad / max(1, total))
        return counts

    def classifier_parameters(self):
        """Parameters trained by gradient descent (for optimizer and clipping)."""
        if self.pfc_hopfield is None or bool(getattr(self.pfc_hopfield, "unsup_update", False)):
            return self.clf.parameters()
        return list(self.clf.parameters()) + list(self.pfc_hopfield.parameters())

    def forward_with_unsupervised_features(
        self,
        x: torch.Tensor,
        local_update: bool = True,
        variance_class_memory: Optional[Dict[str, torch.Tensor]] = None,
        variance_memory_lr: float = 0.05,
        variance_intra_weight: float = 1.0,
        variance_inter_weight: float = 1.0,
        variance_reg_weight: float = 0.0,
        use_step_b_predictive: bool = False,
        step_b_weight: float = 0.1,
        use_step_c_slot: bool = False,
        step_c_weight: float = 0.05,
        step_c_num_slots: int = 4,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass that also returns combined features from all 4 unsupervised layers.

        Returns:
          logits,
          {"l1": [B,D], "l2": [B,D], "l3": [B,D], "l4": [B,D]} — L4 is TE+PP when dorsal is on,
          {"l1": scalar_obj, "l2": scalar_obj, "l3": scalar_obj, "l4": scalar_obj}
        """
        if x.size(1) != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {x.size(1)}")

        freq_inputs, L = self._rgb_to_freq_inputs(x)

        (
            l1_outputs,
            l1_flats,
            l1_patches,
            l2_outputs,
            l2_flats,
            l2_patches,
            _,
            l3_flats,
            l3_patches,
            l4_flats,
            l4_patches,
            _te_flat_u,
            mt_flats,
            mt_patches,
            pp_flats,
            pp_patches,
            _pfc_u,
        ) = self._run_ventral_dorsal_stack(freq_inputs, L)

        l1_combined = torch.cat(l1_flats, dim=1)
        l2_combined = torch.cat(l2_flats, dim=1)
        l3_combined = torch.cat(l3_flats, dim=1)
        l4_te = torch.cat(l4_flats, dim=1)
        pp_flat_raw: Optional[torch.Tensor] = None
        fusion_te_pre = l4_te
        fusion_pp_pre: Optional[torch.Tensor] = None
        if self.use_dorsal_stream and pp_flats is not None:
            pp_flat_raw = torch.cat(pp_flats, dim=1)
            fusion_pp_pre = pp_flat_raw
            l4_te, pp_flat_raw = self._it_pp_bidirectional_cross_gate(l4_te, pp_flat_raw)
            l4_combined = torch.cat([l4_te, pp_flat_raw], dim=1)
        else:
            l4_combined = l4_te

        layer_features = {"l1": l1_combined, "l2": l2_combined, "l3": l3_combined, "l4": l4_combined}
        layer_objectives: Dict[str, torch.Tensor] = {
            "l1": l1_combined.new_zeros(()),
            "l2": l2_combined.new_zeros(()),
            "l3": l3_combined.new_zeros(()),
            "l4": l4_combined.new_zeros(()),
        }
        layer_update_scales = {"l1": 1.0, "l2": 1.0, "l3": 1.0, "l4": 1.0}
        layer_denoms = {"l1": 1.0, "l2": 1.0, "l3": 1.0, "l4": 1.0}

        if (
            self.training
            and variance_class_memory is not None
            and float(variance_reg_weight) > 0.0
        ):
            for layer_name in ("l1", "l2", "l3", "l4"):
                if layer_name not in variance_class_memory:
                    raise ValueError(f"Missing class memory for layer '{layer_name}'.")
                layer_obj, intra_var, inter_var = hebbian_unsupervised_proto_variance_objective(
                    layer_features[layer_name],
                    class_memory=variance_class_memory[layer_name],
                    memory_lr=float(variance_memory_lr),
                    intra_weight=float(variance_intra_weight),
                    inter_weight=float(variance_inter_weight),
                )
                layer_objectives[layer_name] = layer_obj
                denom = intra_var.detach().abs() + inter_var.detach().abs() + 1e-8
                layer_denoms[layer_name] = float(denom.item())

        # Step B option: predictive-coding style consistency across adjacent layers.
        if bool(use_step_b_predictive):
            w_b = float(max(0.0, step_b_weight))
            pc12 = (l1_combined - l2_combined.detach()).pow(2).mean()
            pc23 = (l2_combined - l3_combined.detach()).pow(2).mean()
            # Match L3 to ventral TE only (same dim as L3); PP is separate stream.
            pc34 = (l3_combined - l4_te.detach()).pow(2).mean()
            layer_objectives["l1"] = layer_objectives["l1"] + w_b * pc12
            layer_objectives["l2"] = layer_objectives["l2"] + w_b * 0.5 * (pc12 + pc23)
            layer_objectives["l3"] = layer_objectives["l3"] + w_b * 0.5 * (pc23 + pc34)
            layer_objectives["l4"] = layer_objectives["l4"] + w_b * pc34

        # Step C option: slot-style object-centric binding objective on top layer.
        if bool(use_step_c_slot):
            w_c = float(max(0.0, step_c_weight))
            slot_obj = unsupervised_slot_binding_objective(l4_combined, num_slots=int(step_c_num_slots))
            layer_objectives["l4"] = layer_objectives["l4"] + w_c * slot_obj
            # Small spillover to L3 for smoother hierarchical shaping.
            layer_objectives["l3"] = layer_objectives["l3"] + 0.5 * w_c * slot_obj

        # Convert layer objectives to variance-driven local update scales.
        for layer_name in ("l1", "l2", "l3", "l4"):
            denom = float(max(1e-8, layer_denoms[layer_name]))
            pressure = (layer_objectives[layer_name].detach() / denom).clamp(-1.0, 1.0)
            layer_update_scales[layer_name] = float(torch.clamp(1.0 + 0.5 * pressure, 0.5, 1.5).item())

        base_update_scales = dict(layer_update_scales)
        te_flat = l4_te
        if self.use_dorsal_stream and fusion_pp_pre is not None:
            pfc_state = (
                fusion_pp_pre
                if self.pfc_pre_hopfield_fusion == "pp_only"
                else self._pfc_fuse_te_pp_pre_hopfield(fusion_te_pre, fusion_pp_pre)
            )
        else:
            pfc_state = fusion_te_pre
        l1_activity = float(l1_combined.detach().abs().mean().item())
        l2_activity = float(l2_combined.detach().abs().mean().item())
        l3_activity = float(l3_combined.detach().abs().mean().item())
        for td_i in range(self.pfc_topdown_iters):
            if self.pfc_hopfield is not None and bool(self.use_pfc_hopfield):
                pfc_state = self.pfc_hopfield(pfc_state, update_memory=bool(local_update and self.training))
            if self._pfc_run_second_ventral_stack(td_i):
                freq_fb = self._build_pfc_second_pass_freq_inputs(freq_inputs, pfc_state)
                (
                    l1_outputs,
                    l1_flats,
                    l1_patches,
                    l2_outputs,
                    l2_flats,
                    l2_patches,
                    _,
                    l3_flats,
                    l3_patches,
                    l4_flats,
                    l4_patches,
                    _,
                    mt_flats,
                    mt_patches,
                    pp_flats,
                    pp_patches,
                    pfc_state,
                ) = self._run_ventral_dorsal_stack(
                    freq_fb, L, self._pfc_feedback_state_for_second_pass(pfc_state)
                )
                l1_combined = torch.cat(l1_flats, dim=1)
                l2_combined = torch.cat(l2_flats, dim=1)
                l3_combined = torch.cat(l3_flats, dim=1)
                l4_te = torch.cat(l4_flats, dim=1)
                if self.use_dorsal_stream and pp_flats is not None:
                    pp_flat_raw = torch.cat(pp_flats, dim=1)
                    fusion_te_pre = l4_te
                    fusion_pp_pre = pp_flat_raw
                    l4_te, pp_flat_raw = self._it_pp_bidirectional_cross_gate(l4_te, pp_flat_raw)
                    l4_combined = torch.cat([l4_te, pp_flat_raw], dim=1)
                else:
                    l4_combined = l4_te
                layer_features = {"l1": l1_combined, "l2": l2_combined, "l3": l3_combined, "l4": l4_combined}
                l1_activity = float(l1_combined.detach().abs().mean().item())
                l2_activity = float(l2_combined.detach().abs().mean().item())
                l3_activity = float(l3_combined.detach().abs().mean().item())
            td_scales = (
                self._pfc_neuron_topdown_scales(pfc_state)
                if getattr(self, "use_pfc_topdown_per_neuron", True)
                else self._pfc_topdown_layer_scales(pfc_state)
            )
            inh_scales = self._pfc_inhibition_layer_scales(pfc_state)
            l4_activity = float(pfc_state.detach().abs().mean().item())
            layer_targets = self._blend_layer_targets_for_pfc_unsup(
                l1_activity, l2_activity, l3_activity, l4_activity
            )
            pred_scales = self._pfc_predictive_feedback_layer_scales(pfc_state, layer_targets)
            if getattr(self, "use_pfc_topdown_per_neuron", True):
                iter_update_scales = {
                    layer_name: float(base_update_scales[layer_name])
                    * td_scales[layer_name]
                    * float(pred_scales[layer_name])
                    for layer_name in ("l1", "l2", "l3", "l4")
                }
            else:
                iter_update_scales = {
                    layer_name: float(
                        base_update_scales[layer_name]
                        * td_scales[layer_name]
                        * pred_scales[layer_name]
                    )
                    for layer_name in ("l1", "l2", "l3", "l4")
                }
            inh_total = {
                layer_name: float(inh_scales[layer_name] * pred_scales[layer_name])
                for layer_name in ("l1", "l2", "l3", "l4")
            }
            if td_i == self.pfc_topdown_iters - 1:
                _glia_g = self._neuron_glia_per_layer_gates(layer_targets, update_trace=True)
                for ik, layer_name in enumerate(("l1", "l2", "l3", "l4")):
                    iter_update_scales[layer_name] = iter_update_scales[layer_name] * _glia_g[ik]
                    inh_total[layer_name] = float(inh_total[layer_name] * _glia_g[ik])
            attn_vec = self._pfc_attention_vector(pfc_state)
            self._update_pfc_neuron_topdown_unsup(pfc_state, l1_flats, l2_flats, l3_flats, l4_flats)
            self._update_pfc_topdown_unsup(attn_vec, layer_targets)
            self._update_pfc_inhibition_feedback_unsup(attn_vec, layer_targets)
            self._update_pfc_predictive_feedback_unsup(attn_vec, layer_targets)
            if local_update and self.training and td_i == self.pfc_topdown_iters - 1:
                _n = int(self.spatial_size * self.spatial_size)
                for i in range(self.num_freqs):
                    sl = slice(i * _n, (i + 1) * _n)
                    _pg = self._pfc_spatial_predictive_gain(pfc_state, i)
                    if getattr(self, "use_pfc_topdown_per_neuron", True):
                        s1 = iter_update_scales["l1"][sl]
                        s2 = iter_update_scales["l2"][sl]
                        s3 = iter_update_scales["l3"][sl]
                        s4 = iter_update_scales["l4"][sl]
                    else:
                        s1 = iter_update_scales["l1"]
                        s2 = iter_update_scales["l2"]
                        s3 = iter_update_scales["l3"]
                        s4 = iter_update_scales["l4"]
                    self.l1_freq_layers[i].update_from_patches(
                        l1_patches[i],
                        l1_flats[i],
                        variance_scale=s1,
                        inhibition_scale=inh_total["l1"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l1_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    self.l2_freq_layers[i].update_from_patches(
                        l2_patches[i],
                        l2_flats[i],
                        variance_scale=s2,
                        inhibition_scale=inh_total["l2"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l2_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    self.l3_freq_layers[i].update_from_patches(
                        l3_patches[i],
                        l3_flats[i],
                        variance_scale=s3,
                        inhibition_scale=inh_total["l3"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l3_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    self.l4_freq_layers[i].update_from_patches(
                        l4_patches[i],
                        l4_flats[i],
                        variance_scale=s4,
                        inhibition_scale=inh_total["l4"],
                        layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l4_flats[i]),
                        pfc_spatial_gain=_pg,
                    )
                    if self.use_dorsal_stream and mt_patches is not None and pp_patches is not None:
                        self.mt_freq_layers[i].update_from_patches(
                            mt_patches[i],
                            mt_flats[i],
                            variance_scale=s3,
                            inhibition_scale=inh_total["l3"],
                            layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(mt_flats[i]),
                            pfc_spatial_gain=_pg,
                        )
                        self.pp_freq_layers[i].update_from_patches(
                            pp_patches[i],
                            pp_flats[i],
                            variance_scale=s4,
                            inhibition_scale=inh_total["l4"],
                            layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(pp_flats[i]),
                            pfc_spatial_gain=_pg,
                        )
                self._maybe_apply_local_l1_prox()
        te_flat = l4_te
        if bool(self.use_pfc_pc_layer_output_mask):
            te_flat = self._maybe_pfc_pc_mask_layer_outputs(pfc_state, l1_flats, l2_flats, l3_flats, l4_flats)
            if self.use_dorsal_stream:
                te_flat, pfc_state = self._it_pp_bidirectional_cross_gate(te_flat, pfc_state)
            l1_combined = torch.cat(l1_flats, dim=1)
            l2_combined = torch.cat(l2_flats, dim=1)
            l3_combined = torch.cat(l3_flats, dim=1)
            l4_te = te_flat
            if self.use_dorsal_stream:
                l4_combined = torch.cat([te_flat, pfc_state], dim=1)
            else:
                l4_combined = l4_te
            layer_features = {"l1": l1_combined, "l2": l2_combined, "l3": l3_combined, "l4": l4_combined}
        te_flat, pfc_state = self._apply_pfc_spatial_readout_gate(te_flat, pfc_state)
        self._update_pfc_fusion_gates_unsup(fusion_te_pre, fusion_pp_pre, te_flat, pfc_state)
        l4_for_clf = self._pfc_classifier_readout(te_flat, pfc_state)
        l4_for_clf = self.dropout(l4_for_clf)
        logits = self.clf(l4_for_clf.detach() if self.training else l4_for_clf)

        return logits, layer_features, layer_objectives

    def get_pfc_consistency_loss(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.pfc_hopfield is None or not bool(self.use_pfc_hopfield):
            dev = device if device is not None else next(self.clf.parameters()).device
            return torch.zeros((), device=dev)
        loss = self.pfc_hopfield.get_last_consistency_loss()
        if loss is None:
            dev = device if device is not None else next(self.clf.parameters()).device
            return torch.zeros((), device=dev)
        return loss


# =============================================================================
# TRAINING (CIFAR-10 default; paper mentions multiple datasets)
# =============================================================================

def hebbian_unsupervised_proto_variance_objective(
    features: torch.Tensor,
    class_memory: torch.Tensor,
    memory_lr: float = 0.05,
    intra_weight: float = 1.0,
    inter_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hebbian-like objective with online class memory (prototype) updates.

    Unsupervised prototype update:
      assign each sample to nearest prototype in class_memory, then
      m_c <- (1 - eta) * m_c + eta * mean(features[assign==c])

    Objective (layer-local):
      intra_term: minimize spread to assigned prototype
      inter_term: maximize pairwise prototype separation
      total = intra_weight * intra_term - inter_weight * inter_term

    Returns:
      objective, intra_var, inter_var
    """
    if features.ndim != 2:
        raise ValueError(f"Expected [B, D] features, got shape {tuple(features.shape)}")
    if class_memory.ndim != 2 or class_memory.size(1) != features.size(1):
        raise ValueError(
            f"Expected class_memory [C, D] with D={features.size(1)}, got {tuple(class_memory.shape)}"
        )

    zero = features.new_zeros(())
    B = features.size(0)
    C = class_memory.size(0)
    if B == 0 or C == 0:
        return zero, zero, zero

    # Bootstrap empty prototype memory from first batch (unsupervised).
    if float(class_memory.abs().sum().item()) == 0.0:
        init_k = min(C, B)
        class_memory[:init_k].copy_(features[:init_k].detach())

    # Assign each sample to nearest prototype (pseudo-classes).
    # distances: [B, C]
    distances = (features[:, None, :] - class_memory[None, :, :]).pow(2).mean(dim=-1)
    assign = distances.argmin(dim=1)

    # Hebbian-like online prototype update (no gradient through memory update).
    eta = float(max(0.0, min(1.0, memory_lr)))
    unique_classes = torch.unique(assign)
    for cls_id in unique_classes:
        cls_idx = int(cls_id.item())
        cls_feat = features[assign == cls_id]
        if cls_feat.size(0) == 0:
            continue
        batch_mean = cls_feat.detach().mean(dim=0)
        class_memory[cls_idx].mul_(1.0 - eta).add_(batch_mean, alpha=eta)

    # Intra-cluster spread term (minimize): keep samples close to assigned prototype.
    sample_memory = class_memory[assign]
    intra_var = (features - sample_memory).pow(2).mean()

    # Inter-prototype dispersion term (maximize): push prototypes apart.
    active_means = class_memory[unique_classes]
    inter_var = zero
    if active_means.size(0) > 1:
        means = active_means
        pairwise_sqdist = (means[:, None, :] - means[None, :, :]).pow(2).mean(dim=-1)  # [K, K]
        upper = torch.triu(
            torch.ones_like(pairwise_sqdist, dtype=torch.bool, device=pairwise_sqdist.device), diagonal=1
        )
        inter_var = pairwise_sqdist[upper].mean() if upper.any() else zero

    objective = float(intra_weight) * intra_var - float(inter_weight) * inter_var
    return objective, intra_var, inter_var


def unsupervised_slot_binding_objective(
    features: torch.Tensor,
    num_slots: int = 4,
) -> torch.Tensor:
    """
    Lightweight slot-style objective:
      minimize within-slot compactness while maximizing slot separation.
    Uses nearest-slot assignments from current batch (no labels).
    """
    if features.ndim != 2:
        raise ValueError(f"Expected [B, D] features, got {tuple(features.shape)}")
    B, _ = features.shape
    if B < 2:
        return features.new_zeros(())

    K = int(max(2, min(num_slots, B)))
    slots = features[:K].detach().clone()
    dists = (features[:, None, :] - slots[None, :, :]).pow(2).mean(dim=-1)  # [B,K]
    assign = dists.argmin(dim=1)

    compact = (features - slots[assign]).pow(2).mean()
    inter = features.new_zeros(())
    if K > 1:
        pair = (slots[:, None, :] - slots[None, :, :]).pow(2).mean(dim=-1)
        upper = torch.triu(torch.ones_like(pair, dtype=torch.bool), diagonal=1)
        inter = pair[upper].mean() if upper.any() else inter

    return compact - 0.1 * inter


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    show_progress: bool = True,
    use_amp: bool = False,
    scaler=None,
    label_smoothing: float = 0.0,
    variance_reg_weight: float = 0.0,
    variance_intra_weight: float = 1.0,
    variance_inter_weight: float = 1.0,
    variance_memory_lr: float = 0.05,
    variance_class_memory: Optional[Dict[str, torch.Tensor]] = None,
    variance_only_unsup: bool = False,
    use_step_b_predictive: bool = False,
    step_b_weight: float = 0.1,
    use_step_c_slot: bool = False,
    step_c_weight: float = 0.05,
    step_c_num_slots: int = 4,
    pfc_consistency_weight: float = 0.0,
    local_plasticity: bool = True,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_ce_loss, total_var_objective, total_pfc_consistency = 0.0, 0.0, 0.0
    last_pred, last_actual = None, None
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}", disable=not show_progress)
    _nb = device.type == "cuda"
    for x, y in pbar:
        x, y = x.to(device, non_blocking=_nb), y.to(device, non_blocking=_nb)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            if float(variance_reg_weight) > 0.0:
                out, unsup_features, layer_terms = model.forward_with_unsupervised_features(
                    x,
                    local_update=bool(local_plasticity),
                    variance_class_memory=variance_class_memory,
                    variance_memory_lr=float(variance_memory_lr),
                    variance_intra_weight=float(variance_intra_weight),
                    variance_inter_weight=float(variance_inter_weight),
                    variance_reg_weight=float(variance_reg_weight),
                    use_step_b_predictive=bool(use_step_b_predictive),
                    step_b_weight=float(step_b_weight),
                    use_step_c_slot=bool(use_step_c_slot),
                    step_c_weight=float(step_c_weight),
                    step_c_num_slots=int(step_c_num_slots),
                )
            else:
                out = model(x, local_update=bool(local_plasticity))
                unsup_features = None
                layer_terms = {}
            ce_loss = F.cross_entropy(out, y, label_smoothing=float(label_smoothing))
            var_objective = ce_loss.new_zeros(())
            pfc_consistency = model.get_pfc_consistency_loss(device=x.device)
            if float(variance_reg_weight) > 0.0:
                if not layer_terms:
                    raise ValueError("Layer-wise variance objectives were not produced.")
                # Independent per-layer objectives; combined without averaging.
                var_objective = layer_terms["l1"] + layer_terms["l2"] + layer_terms["l3"] + layer_terms["l4"]
            # Label loss + optional unsupervised objectives.
            total_objective = (
                ce_loss
                + float(variance_reg_weight) * var_objective
                + float(max(0.0, pfc_consistency_weight)) * pfc_consistency
            )
            loss = ce_loss + float(max(0.0, pfc_consistency_weight)) * pfc_consistency
            if not loss.requires_grad:
                loss = ce_loss
        if use_amp and scaler is not None:
            did_amp_step = False
            if loss.requires_grad:
                scaler.scale(loss).backward()
                has_optimizer_grads = any(
                    p.grad is not None for group in optimizer.param_groups for p in group["params"]
                )
                if has_optimizer_grads:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
                    scaler.step(optimizer)
                    did_amp_step = True
            if did_amp_step:
                scaler.update()
        else:
            if loss.requires_grad:
                loss.backward()
                has_optimizer_grads = any(
                    p.grad is not None for group in optimizer.param_groups for p in group["params"]
                )
                if has_optimizer_grads:
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
                    optimizer.step()

        total_loss += total_objective.item() * x.size(0)
        total_ce_loss += ce_loss.item() * x.size(0)
        total_var_objective += var_objective.item() * x.size(0)
        total_pfc_consistency += pfc_consistency.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
        last_pred = out.argmax(1).detach().cpu().tolist()
        last_actual = y.detach().cpu().tolist()
        postfix = {"loss": f"{total_loss/total:.4f}", "acc": f"{100*correct/total:.2f}%"}
        if variance_reg_weight > 0.0:
            postfix["var"] = f"{total_var_objective/total:.4f}"
            postfix["ce(eval)"] = f"{total_ce_loss/total:.4f}"
        if pfc_consistency_weight > 0.0:
            postfix["pfc_cons"] = f"{total_pfc_consistency/total:.4f}"
        pbar.set_postfix(postfix)
    return total_loss / total, 100.0 * correct / total, last_pred, last_actual


@torch.no_grad()
def test_epoch(model: nn.Module, loader, device, use_amp: bool = False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    _nb = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=_nb), y.to(device, non_blocking=_nb)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            out = model(x, local_update=False)
            # Keep test loss on the same scale as train loss (mean per sample)
            loss = F.cross_entropy(out, y)
        total_loss += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return total_loss / total, 100.0 * correct / total


@dataclass
class TrainingRunResult:
    """Metrics returned after a full CIFAR-10 training run (programmatic API)."""

    best_val_acc: float
    best_epoch: int
    best_test_at_best_val: Optional[float]
    final_val_loss: float
    final_val_acc: float
    final_test_loss: float
    final_test_acc: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisNet Unified - CIFAR-10, Separate Freqs")

    def add_bool_arg(name: str, default: bool, help_text: str) -> None:
        """Backwards-compatible bool flag with explicit --foo / --no-foo."""
        dest = name.replace("-", "_")
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
        group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable: {help_text}")
        parser.set_defaults(**{dest: default})

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars (faster logging on Windows).")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed-precision training.")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for classifier cross-entropy.")
    parser.add_argument(
        "--variance-reg-weight",
        type=float,
        default=0.05,
        help="Weight for Hebbian variance regularizer across L1-L4: +inter_memory_variance - intra_class_spread (default 0=off).",
    )
    parser.add_argument(
        "--variance-intra-weight",
        type=float,
        default=1.5,
        help="Scale for intra-class variance term (subtracted in objective to maximize intra variance).",
    )
    parser.add_argument(
        "--variance-inter-weight",
        type=float,
        default=0.5,
        help="Scale for inter-class variance term (added in objective to minimize inter variance).",
    )
    parser.add_argument(
        "--variance-memory-lr",
        type=float,
        default=0.06,
        help="Hebbian update rate for class memory prototypes used in the variance term.",
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=42,
        help="Topographic map size (default 64). Use 32 to match CIFAR input resolution for speed; coarser maps.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=-1,
        help="DataLoader workers: -1 = min(4, CPU count), 0 = load in main process (slowest).",
    )
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--rf-l1", type=int, default=15, help="Receptive field size for L1.")
    parser.add_argument("--rf-l2", type=int, default=15, help="Receptive field size for L2.")
    parser.add_argument("--rf-l3", type=int, default=15, help="Receptive field size for L3.")
    parser.add_argument("--rf-l4", type=int, default=15, help="Receptive field size for L4.")
    add_bool_arg(
        "rf-gaussian-connectivity",
        default=False,
        help_text="Gaussian center-weighted RF mask; keep top fraction of sites by Gaussian (--rf-connectivity-keep-frac, default 0.6).",
    )
    parser.add_argument(
        "--rf-gaussian-sigma-frac",
        type=float,
        default=0.28,
        help="Gaussian σ as a fraction of rf_size (isotropic); larger RF → use ~0.25–0.35 for a tight center bump.",
    )
    parser.add_argument(
        "--rf-connectivity-keep-frac",
        type=float,
        default=0.6,
        help="Binary mask: fraction of RF spatial sites set to 1 (top by Gaussian score ≈ center); 0.6 ≈ 60%% of sites are 1. 1.0 = all 1 (full RF).",
    )
    parser.add_argument(
        "--rf-sparse-quantile",
        type=float,
        default=0.0,
        help="Only if --rf-connectivity-keep-frac=1.0: zero below this quantile of the Gaussian (legacy). Otherwise use keep-frac.",
    )
    parser.add_argument(
        "--recursive-iters",
        type=int,
        default=0,
        help="Number of recursive inhibition refinement iterations per layer forward pass.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of train split to use for training (default 0.10 = 10%).",
    )
    parser.add_argument("--no-auto-resize", action="store_true", help="Disable auto-resizing gabor maps to spatial-size")
    parser.add_argument(
        "--no-dorsal-stream",
        action="store_true",
        help="Disable dorsal stream (parallel high-pass luminance hierarchy); use ventral-only L4 size.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root directory for CIFAR-10 download/cache.",
    )
    parser.add_argument("--val-split", type=float, default=0.02, help="Fraction of CIFAR-10 train used for validation (0.1 = 5k val)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for train/val split reproducibility")
    parser.add_argument(
        "--eval-test-each-epoch",
        action="store_true",
        help="If set, also evaluate test each epoch (do NOT use it for model selection).",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="If set, save full model checkpoint when validation improves.",
    )
    parser.add_argument(
        "--save-best-path",
        type=str,
        default="best_model_cifar10_val.pth",
        help="Path for best-by-val checkpoint (only used if --save-best).",
    )
    parser.add_argument(
        "--no-eval-test-on-best-val",
        action="store_true",
        help="Disable reporting test when validation improves (enabled by default).",
    )
    parser.add_argument(
        "--lambda-d",
        type=float,
        default=1e-4,
        help="Weight decay for plasticity layers (default 0.03). Try [1e-4, 0.1] if tuning.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.01,
        #default=0.04,
        help="Max dropout at epoch 1 when scheduling; decays to 0 by last epoch (default 0.05).",
    )
    parser.add_argument(
        "--clf-lr",
        type=float,
        default=5e-4,
        help="Learning rate for classifier optimizer.",
    )
    parser.add_argument(
        "--no-schedule-decay-dropout",
        action="store_true",
        help="Disable scheduling: use constant --lambda-d and --dropout instead of decaying to 0.",
    )
    parser.add_argument(
        "--lambda-d-floor-ratio",
        type=float,
        default=0.15,
        help="When scheduling, never decay plasticity lambda_d below this fraction of --lambda-d (0=old behavior: decay to 0). Helps avoid late-epoch Hebbian blow-ups.",
    )
    parser.add_argument(
        "--dropout-floor-ratio",
        type=float,
        default=0.15,
        help="When scheduling, never decay classifier dropout below this fraction of --dropout (0=decay to 0).",
    )
    parser.add_argument(
        "--freeze-local-plasticity-after-epoch",
        type=int,
        default=3,
        help="If >0, disable L1-L4 local Hebbian updates after this epoch (classifier still trains). Stops late collapse from plasticity drift. 0=off.",
    )
    add_bool_arg(
        "auto-freeze-on-collapse",
        default=True,
        help_text="Automatically freeze local plasticity if validation accuracy collapses far below best.",
    )
    parser.add_argument(
        "--collapse-val-drop-threshold",
        type=float,
        default=8.0,
        help="Auto-freeze trigger: val_acc must drop by at least this many points below best_val.",
    )
    parser.add_argument(
        "--collapse-patience-epochs",
        type=int,
        default=1,
        help="Auto-freeze trigger: wait this many epochs after best epoch before checking collapse.",
    )
    parser.add_argument(
        "--entropy-dropout",
        action="store_true",
        help="Use entropy-based dropout: p = base_p * (1 + scale * norm_entropy).",
    )
    parser.add_argument(
        "--entropy-dropout-scale",
        type=float,
        default=0.5,
        help="Scale for entropy term in entropy-based dropout (default 0.5).",
    )
    parser.add_argument(
        "--entropy-plasticity-decay",
        action="store_true",
        help="Scale plasticity weight decay (lambda_d) by activation entropy.",
    )
    parser.add_argument(
        "--entropy-decay-scale",
        type=float,
        default=0.5,
        help="Scale for entropy term in plasticity decay (default 0.5).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=3e-4,
        help="L2 weight decay for classifier (Adam). Plasticity uses --lambda-d (default 0).",
    )
    add_bool_arg(
        "use-pfc-hopfield",
        default=True,
        help_text="Enable a PFC-like Modern Hopfield block after L4 before classification.",
    )
    parser.add_argument("--pfc-hopfield-patterns", type=int, default=96, help="Number of associative memory patterns in PFC Hopfield block.")
    parser.add_argument("--pfc-hopfield-beta", type=float, default=1.5, help="Softmax sharpness in PFC Hopfield retrieval.")
    parser.add_argument("--pfc-hopfield-temperature", type=float, default=1.5, help="Temperature for Hopfield retrieval/assignment softmax.")
    parser.add_argument("--pfc-hopfield-blend", type=float, default=0.0005, help="Blend ratio between raw L4 and retrieved PFC state.")
    parser.add_argument(
        "--pfc-warmup-epochs",
        type=int,
        default=5,
        help="Number of initial epochs to keep PFC disabled before enabling it (0 = no warmup).",
    )
    parser.add_argument("--pfc-hopfield-ema-lr", type=float, default=0.001, help="EMA rate for unsupervised Hopfield memory updates from L4 features.")
    add_bool_arg(
        "pfc-hopfield-unsup-update",
        default=True,
        help_text="Enable unsupervised EMA memory updates for PFC Hopfield patterns.",
    )
    add_bool_arg(
        "pfc-hopfield-cosine",
        default=True,
        help_text="Use cosine similarity for Hopfield retrieval and assignment.",
    )
    add_bool_arg(
        "pfc-hopfield-soft-ema",
        default=True,
        help_text="Use soft assignment (attention-weighted) EMA update for Hopfield memory.",
    )
    add_bool_arg(
        "pfc-hopfield-normalize-memory",
        default=False,
        help_text="L2-normalize Hopfield memory patterns after updates.",
    )
    parser.add_argument(
        "--pfc-hopfield-sparsity",
        type=float,
        default=0.91,
        help="Sparsity for competitive pre-update gating in PFC Hopfield memory updates (higher = sparser).",
    )
    add_bool_arg(
        "pfc-hopfield-sparse-update",
        default=True,
        help_text="Apply PFC sparsity to features before Hopfield memory EMA updates.",
    )
    parser.add_argument(
        "--pfc-consistency-weight",
        type=float,
        default=0.01,
        help="Weight of label-free retrieval-consistency loss for PFC Hopfield.",
    )
    add_bool_arg(
        "pfc-hopfield-layernorm",
        default=False,
        help_text="Apply LayerNorm at the output of PFC Hopfield block.",
    )
    parser.add_argument(
        "--pfc-mode",
        type=str,
        #default="hopfield",
        default="hebbian_sa",
        choices=("hopfield", "hebbian_sa"),
        help='PFC memory: "hopfield" (Modern Hopfield) or "hebbian_sa" (softmax self-attention over slots; Q/K Hebbian, no BP).',
    )
    parser.add_argument(
        "--pfc-hebbian-lr",
        type=float,
        default=1e-4,
        help="Hebbian learning rate for Q/K projections when --pfc-mode hebbian_sa.",
    )
    parser.add_argument(
        "--pfc-hebbian-decay",
        type=float,
        default=1e-5,
        help="Multiplicative decay per step on Q/K when --pfc-mode hebbian_sa.",
    )
    parser.add_argument(
        "--pfc-sa-head-dim",
        type=int,
        default=0,
        help="Self-attention head dim for hebbian_sa (0 = auto from L4 flat dim).",
    )
    add_bool_arg(
        "pfc-topdown-attention",
        default=True,
        help_text="Use PFC-derived top-down attention gates to modulate local updates in L1-L4.",
    )
    parser.add_argument(
        "--pfc-topdown-strength",
        type=float,
        default=0.22,
        help="Strength of PFC top-down modulation around baseline scale 1.0.",
    )
    parser.add_argument(
        "--pfc-topdown-min-scale",
        type=float,
        default=0.7,
        help="Lower clamp for PFC top-down local-update scaling.",
    )
    parser.add_argument(
        "--pfc-topdown-max-scale",
        type=float,
        default=1.2,
        help="Upper clamp for PFC top-down local-update scaling.",
    )
    parser.add_argument(
        "--pfc-topdown-unsup-lr",
        type=float,
        default=0.002,
        help="Unsupervised local learning rate for PFC top-down attention weights.",
    )
    parser.add_argument(
        "--pfc-topdown-decay",
        type=float,
        default=5e-5,
        help="Decay for unsupervised PFC top-down attention weights.",
    )
    add_bool_arg(
        "pfc-topdown-per-neuron",
        default=True,
        help_text="One learned PFC→neuron weight per (L1–L4, neuron): 4×num_freqs×spatial² = all ventral neurons.",
    )
    add_bool_arg(
        "pfc-topdown-neuron-bias",
        default=False,
        help_text="Optional 4×D bias (extra params). Default off so learnable connections = 4×F×H×W weights only.",
    )
    parser.add_argument(
        "--pfc-topdown-shared-fe-blend",
        type=float,
        default=0.35,
        help="Blend α in [0,1] for PFC unsup targets: per-neuron top-down LMS and 4-way softmax paths use (1-α)·norm(activity)+α·norm(|G^T err|) per layer. 0=activity only.",
    )
    parser.add_argument(
        "--pfc-inhibition-feedback-unsup-lr",
        type=float,
        default=0.00125,
        help="Unsupervised local learning rate for PFC inhibition-feedback weights.",
    )
    parser.add_argument(
        "--pfc-inhibition-feedback-decay",
        type=float,
        default=5e-5,
        help="Decay for unsupervised PFC inhibition-feedback weights.",
    )
    parser.add_argument(
        "--pfc-topdown-iters",
        type=int,
        default=2,
        help="Number of iterative PFC top-down feedback cycles per forward pass.",
    )
    parser.add_argument(
        "--dorsal4-attention-preset",
        type=str,
        default="active",
        choices=("active", "minimal"),
        help="active (default): symmetry prior + IT-PP cross-gate on. minimal: both off (overrides after parse).",
    )
    add_bool_arg(
        "pfc-spatial-readout-gate",
        default=True,
        help_text="After PFC/Hopfield, gate TE and PP flat maps by a spatial saliency map from PFC state (figure vs background readout bias).",
    )
    parser.add_argument(
        "--pfc-spatial-gate-strength",
        type=float,
        default=0.65,
        help="0=no spatial gating, 1=full blend toward saliency map (default 0.65).",
    )
    parser.add_argument(
        "--pfc-spatial-gate-floor",
        type=float,
        default=0.2,
        help="Minimum multiplicative gain at any location when strength=1 (avoids zeroing background entirely).",
    )
    parser.add_argument(
        "--pfc-recurrent-feedback-steps",
        type=int,
        default=2,
        choices=[1, 2],
        help="1=single ventral pass unless dense/deep feedback; 2=second pass with multiplicative PFC saliency on L1 (default 2). Use --no-pfc-dense-feedback etc. to simplify.",
    )
    parser.add_argument(
        "--pfc-recurrent-feedback-strength",
        type=float,
        default=0.35,
        help="Second-pass emphasis: freq_inputs *= (1 + strength * PFC saliency map); 0 disables modulation.",
    )
    add_bool_arg(
        "pfc-dense-feedback",
        default=True,
        help_text="Explicit PFC→L1 top-down: on the last td iter, add a normalized PFC spatial map to each L1 band (second stack). Can be used without recurrent steps=2 if strength>0.",
    )
    parser.add_argument(
        "--pfc-dense-feedback-strength",
        type=float,
        default=0.12,
        help="Additive scale for dense PFC feedback onto L1 (broadcast across channels per frequency).",
    )
    add_bool_arg(
        "pfc-deep-feedback",
        default=True,
        help_text="Second-stack bottleneck PFC→L2/L3/L4/(MT/PP) additive maps (no BP on feedback weights).",
    )
    parser.add_argument(
        "--pfc-deep-feedback-rank",
        type=int,
        default=32,
        help="Bottleneck width r for PFC deep feedback (flat D -> r -> F*H*W maps per stage).",
    )
    parser.add_argument(
        "--pfc-deep-fb-l2",
        type=float,
        default=0.06,
        help="Strength of additive PFC feedback to ventral L2 outputs (second stack).",
    )
    parser.add_argument(
        "--pfc-deep-fb-l3",
        type=float,
        default=0.06,
        help="Strength of additive PFC feedback to ventral L3 outputs (second stack).",
    )
    parser.add_argument(
        "--pfc-deep-fb-l4",
        type=float,
        default=0.08,
        help="Strength of additive PFC feedback to ventral L4 (TE) outputs (second stack).",
    )
    parser.add_argument(
        "--pfc-deep-fb-mt",
        type=float,
        default=0.05,
        help="Strength of additive PFC feedback to MT outputs when dorsal stream is on.",
    )
    parser.add_argument(
        "--pfc-deep-fb-pp",
        type=float,
        default=0.05,
        help="Strength of additive PFC feedback to PP outputs when dorsal stream is on.",
    )
    add_bool_arg(
        "symmetry-gate-prior",
        default=True,
        help_text="Multiply dorsal saliency gate by vertical bilateral-consistency map (no BP). Default: on.",
    )
    parser.add_argument(
        "--symmetry-gate-alpha",
        type=float,
        default=0.5,
        help="Strength of symmetry prior: gate *= (1 + alpha * scale * sym_map); scale from EMA if unsup lr > 0.",
    )
    parser.add_argument(
        "--symmetry-prior-unsup-lr",
        type=float,
        default=0.0,
        help="EMA rate for batch-mean symmetry (scales alpha); 0 = fixed scale 1.0, no unsupervised update.",
    )
    add_bool_arg(
        "it-pp-cross-gate",
        default=True,
        help_text="Bidirectional IT↔PP: TE gains spatial emphasis from PP saliency map, PP from TE (no learned Q/K). Default: on.",
    )
    parser.add_argument(
        "--it-pp-cross-pp-to-te",
        type=float,
        default=0.35,
        help="TE *= (1 + strength * PP spatial attention map).",
    )
    parser.add_argument(
        "--it-pp-cross-te-to-pp",
        type=float,
        default=0.35,
        help="PP *= (1 + strength * TE spatial attention map).",
    )
    parser.add_argument(
        "--it-pp-cross-iters",
        type=int,
        default=1,
        help="Number of mutual TE↔PP gating iterations at the interface.",
    )
    parser.add_argument(
        "--pfc-pre-hopfield-fusion",
        type=str,
        default="all",
        choices=("pp_only", "blend", "gate", "all"),
        help='PFC Hopfield input when dorsal is on: pp_only, blend, gate, or all (mean of pp_only, blend, gate).',
    )
    parser.add_argument(
        "--pfc-pre-blend-w-te",
        type=float,
        default=0.5,
        help="Blend weight for ventral TE when --pfc-pre-hopfield-fusion blend.",
    )
    parser.add_argument(
        "--pfc-pre-blend-w-pp",
        type=float,
        default=0.5,
        help="Blend weight for dorsal PP when --pfc-pre-hopfield-fusion blend.",
    )
    parser.add_argument(
        "--pfc-post-readout-fusion",
        type=str,
        default="all",
        choices=("concat", "gate", "all"),
        help='Classifier second half: concat, gate (TE–PFC mix), or all (mean of concat and gate).',
    )
    add_bool_arg(
        "pfc-fusion-gate-unsup",
        default=True,
        help_text="Unsupervised LMS/EMA updates on pre/post TE–PP and TE–PFC fusion gates (no classifier BP).",
    )
    parser.add_argument(
        "--pfc-fusion-gate-unsup-lr",
        type=float,
        default=3e-3,
        help="Learning rate for unsupervised fusion-gate updates (scaled with joint plasticity schedule).",
    )
    parser.add_argument(
        "--pfc-fusion-gate-decay",
        type=float,
        default=1e-4,
        help="Multiplicative decay per step on fusion-gate weights (same role as pfc_topdown_decay).",
    )
    parser.add_argument(
        "--pfc-fusion-lms-chunk-rows",
        type=int,
        default=256,
        help="Fusion-gate LMS row-chunk size (larger = fewer matmuls, slightly more peak memory per chunk).",
    )
    add_bool_arg(
        "compile-model",
        default=False,
        help_text="Use torch.compile on CUDA (warmup on first steps; can help steady-state throughput).",
    )
    add_bool_arg(
        "pfc-predictive-feedback",
        default=True,
        help_text="Enable explicit PFC predictive-coding feedback to L1-L4.",
    )
    parser.add_argument(
        "--pfc-predictive-strength",
        type=float,
        default=0.15,
        help="Strength of PFC predictive error feedback scaling.",
    )
    parser.add_argument(
        "--pfc-predictive-min-scale",
        type=float,
        default=0.5,
        help="Lower clamp for PFC predictive feedback scaling.",
    )
    parser.add_argument(
        "--pfc-predictive-max-scale",
        type=float,
        default=1.3,
        help="Upper clamp for PFC predictive feedback scaling.",
    )
    parser.add_argument(
        "--pfc-predictive-unsup-lr",
        type=float,
        default=1.5e-3,
        help="Unsupervised local learning rate for PFC predictive-feedback weights.",
    )
    parser.add_argument(
        "--pfc-predictive-decay",
        type=float,
        default=5e-5,
        help="Decay for unsupervised PFC predictive-feedback weights.",
    )
    add_bool_arg(
        "pfc-pc-layer-output-mask",
        default=True,
        help_text="PFC predictive-coding LMS masks ventral L1–L4 flats (sigmoid(W·PFC+b)) before TE readout; no classifier BP.",
    )
    parser.add_argument(
        "--pfc-pc-layer-mask-lr",
        type=float,
        default=0.001,
        help="LMS lr for PC layer-output mask weights (target = max-normalized mean |activity| per layer).",
    )
    parser.add_argument(
        "--pfc-pc-layer-mask-decay",
        type=float,
        default=1e-4,
        help="Weight decay per step for PC layer-output mask matrices.",
    )
    parser.add_argument(
        "--pfc-l1-lambda",
        type=float,
        default=1.2e-4,
        help="L1 sparsity strength for PFC feedback matrices (proximal shrinkage).",
    )
    parser.add_argument(
        "--pfc-l1-prox-step",
        type=float,
        default=1.5,
        help="Proximal step size used for L1 soft-threshold on PFC feedback matrices.",
    )
    parser.add_argument(
        "--local-l1-lambda",
        type=float,
        default=1.2e-6,
        help="L1 sparsity strength for local unsupervised layer feedforward weights (W only).",
    )
    parser.add_argument(
        "--local-l1-prox-step",
        type=float,
        default=1.0,
        help="Proximal step size used for local-layer L1 soft-threshold on W.",
    )
    parser.add_argument(
        "--local-l1-warmup-steps",
        type=int,
        default=1000,
        help="Number of local update steps before enabling local-layer L1 prox.",
    )
    parser.add_argument(
        "--local-l1-apply-every",
        type=int,
        default=2,
        help="Apply local-layer L1 prox every N local update steps.",
    )
    parser.add_argument(
        "--inhibition-mode",
        type=str,
        default="mixed",
        choices=["global", "kernel", "averaging", "mixed", "mexican", "predictive"],
        help="Inhibition: global / kernel / averaging / mixed / mexican / predictive (inhibit generator-predictable components).",
    )

    add_bool_arg(
        "use-ei-neurons",
        default=False,
        help_text="Enable E/I populations (requires --ei-mutual-inhibition for split dynamics; global/mixed only).",
    )
    add_bool_arg(
        "ei-mutual-inhibition",
        default=False,
        help_text="Mutual E/I coupling (L_ei, L_ie). Opt-in; default off for stable training.",
    )
    add_bool_arg(
        "ei-separate-lateral",
        default=True,
        help_text="Use separate learned L_ei (I->E, <=0) and L_ie (E->I, >=0); disable for legacy shared L.",
    )
    parser.add_argument(
        "--ei-l-ei-init",
        type=float,
        default=-0.01,
        help="Initialization value for L_ei (I->E) when EI separate lateral is enabled.",
    )
    parser.add_argument(
        "--ei-l-ie-init",
        type=float,
        default=0.01,
        help="Initialization value for L_ie (E->I) when EI separate lateral is enabled.",
    )
    add_bool_arg(
        "som-inhibition-schedules",
        default=False,
        help_text="SOM-style schedules (kernel/Mexican σ, lateral LR, softmax T, WTA). Off by default for stable training; use --som-inhibition-schedules to enable.",
    )
    parser.add_argument(
        "--kernel-inhibition-sigma-start",
        type=float,
        default=2.5,
        help="(SOM) Starting Gaussian σ for kernel/mixed inhibition (wide neighborhood early).",
    )
    parser.add_argument(
        "--kernel-inhibition-sigma-end",
        type=float,
        default=1.0,
        help="(SOM) Ending Gaussian σ for kernel/mixed inhibition (sharp late).",
    )
    parser.add_argument(
        "--som-lr-lateral-warmup-fraction",
        type=float,
        default=0.2,
        help="Fraction of training with full lateral LR before decay (0–1).",
    )
    parser.add_argument(
        "--lr-lateral-schedule-end-scale",
        type=float,
        default=0.5,
        help="Scale for lateral (L-matrix) LR at end of schedule vs start (after warmup).",
    )
    add_bool_arg(
        "inhibition-softmax",
        default=False,
        help_text="Spatial softmax competition softmax(y/T) after inhibition. Off by default; use --inhibition-softmax to enable.",
    )
    parser.add_argument(
        "--inhibition-softmax-temp-start",
        type=float,
        default=1.5,
        help="Softmax temperature at start (softer assignment).",
    )
    parser.add_argument(
        "--inhibition-softmax-temp-end",
        type=float,
        default=0.45,
        help="Softmax temperature at end (sharper winners).",
    )
    parser.add_argument(
        "--som-wta-scale-start",
        type=float,
        default=0.88,
        help="Multiply layer WTA sparsity at start of schedule.",
    )
    parser.add_argument(
        "--som-wta-scale-end",
        type=float,
        default=1.0,
        help="Multiply layer WTA sparsity at end of schedule.",
    )
    parser.add_argument(
        "--mexican-center-sigma-start",
        type=float,
        default=0.65,
        help="Mexican-hat: narrow Gaussian σ at schedule start.",
    )
    parser.add_argument(
        "--mexican-center-sigma-end",
        type=float,
        default=0.45,
        help="Mexican-hat: narrow Gaussian σ at schedule end.",
    )
    parser.add_argument(
        "--mexican-surround-sigma-start",
        type=float,
        default=2.4,
        help="Mexican-hat: wide Gaussian σ at schedule start.",
    )
    parser.add_argument(
        "--mexican-surround-sigma-end",
        type=float,
        default=1.75,
        help="Mexican-hat: wide Gaussian σ at schedule end.",
    )
    parser.add_argument(
        "--mexican-surround-gain-start",
        type=float,
        default=1.0,
        help="Mexican-hat: surround gain at schedule start.",
    )
    parser.add_argument(
        "--mexican-surround-gain-end",
        type=float,
        default=1.15,
        help="Mexican-hat: surround gain at schedule end.",
    )
    add_bool_arg(
        "soft-inhibition",
        default=True,
        #default=False,
        help_text="Use signed soft-threshold τ on inhibition (ignored if --inhibition-no-threshold).",
    )
    add_bool_arg(
        "inhibition-no-threshold",
        default=True,
        help_text="No τ cut-off on inhibition: use raw kernel/global/averaging drives (soft linear competition). Optional --inhibition-smooth-scale adds tanh saturation only.",
    )
    parser.add_argument(
        "--inhibition-smooth-scale",
        type=float,
        default=0.0,
        help="If >0 with --inhibition-no-threshold: smooth saturation scale*tanh(x/scale) (no hard zeros). 0 = fully linear.",
    )
    parser.add_argument(
        "--averaging-inhibition-size",
        type=int,
        default=7,
        help="Kernel size for averaging inhibition mode (odd preferred).",
    )
    parser.add_argument(
        "--kernel-inhibition-size",
        type=int,
        default=7,
        help="Kernel size for kernel-based inhibition mode (odd preferred).",
    )
    parser.add_argument(
        "--kernel-inhibition-sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma for kernel-based inhibition.",
    )
    parser.add_argument(
        "--mixed-inhibition-w-global",
        type=float,
        default=0.5,
        help="Relative weight of global inhibition in mixed mode.",
    )
    parser.add_argument(
        "--mixed-inhibition-w-kernel",
        type=float,
        default=0.25,
        help="Relative weight of kernel inhibition in mixed mode.",
    )
    parser.add_argument(
        "--mixed-inhibition-w-averaging",
        type=float,
        default=0.25,
        help="Relative weight of averaging inhibition in mixed mode.",
    )
    parser.add_argument(
        "--inhibition-dropout",
        type=float,
        default=0.01,
        help="Dropout probability on inhibition signals during training (0 disables).",
    )
    add_bool_arg(
        "divisive-inhibition-norm",
        default=True,
        help_text="Apply divisive normalization after LayerNorm+L2 and optional inhibition (runs even if inhibition strength is 0).",
    )
    parser.add_argument(
        "--divisive-alpha",
        type=float,
        default=0.85,
        help="Additive term in divisive denominator (stability; tuned default 0.85 for local mode).",
    )
    parser.add_argument(
        "--divisive-beta",
        type=float,
        default=0.36,
        help="Gain on pooled |y| in divisive denominator (default tuned for combined global+local).",
    )
    parser.add_argument(
        "--divisive-mode",
        type=str,
        default="both",
        choices=["global", "local", "both", "all"],
        help="Divisive norm: global / local / both (mix) / all (=both). Default both = global+local.",
    )
    parser.add_argument(
        "--divisive-w-global",
        type=float,
        default=0.5,
        help="Weight of global mean(|y|) in 'both' mode (normalized with divisive-w-local).",
    )
    parser.add_argument(
        "--divisive-w-local",
        type=float,
        default=0.5,
        help="Weight of local Gaussian pool in 'both' mode (normalized with divisive-w-global).",
    )
    parser.add_argument(
        "--divisive-local-size",
        type=int,
        default=9,
        help="Odd kernel size for local divisive pooling (even values are bumped to odd; default 9).",
    )
    parser.add_argument(
        "--divisive-local-sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma for local divisive pooling (default 2.0 for 9×9 surround).",
    )
    add_bool_arg(
        "adaptive-inhibition",
        default=True,
        help_text="Adapt inhibition strength online to match target activity fraction.",
    )
    parser.add_argument("--target-active-frac", type=float, default=0.03, help="Target fraction of positive activations.")
    parser.add_argument("--inhibition-adapt-lr", type=float, default=0.002, help="Adaptation rate for inhibition strength.")
    add_bool_arg(
        "use-homeostatic-threshold",
        default=True,
        help_text="Use per-neuron homeostatic adaptive threshold for k-winner gating (replaces fixed WTA threshold).",
    )
    parser.add_argument(
        "--homeostatic-threshold-lr",
        type=float,
        default=0.02,
        help="Learning rate for homeostatic per-neuron threshold update (theta_i).",
    )
    parser.add_argument(
        "--homeostatic-threshold-theta-init",
        type=float,
        default=0.0,
        help="Initial value for per-neuron winner thresholds (theta_i).",
    )
    parser.add_argument(
        "--homeostatic-threshold-min",
        type=float,
        default=-2.0,
        help="Lower clamp for per-neuron winner thresholds (theta_i).",
    )
    parser.add_argument(
        "--homeostatic-threshold-max",
        type=float,
        default=2.0,
        help="Upper clamp for per-neuron winner thresholds (theta_i).",
    )
    add_bool_arg(
        "oscillatory-inhibition",
        default=True,
        help_text="Enable oscillatory/phase gating for competition (phase-based winners).",
    )
    parser.add_argument(
        "--phase-period",
        type=int,
        default=10,
        help="Oscillation period (in forward calls) for phase inhibition.",
    )
    parser.add_argument(
        "--phase-gate-sharpness",
        type=float,
        default=1.0,
        help="Sharpness of phase gate; higher => more winner-only-in-one-phase behavior.",
    )
    parser.add_argument("--inhibition-decay", type=float, default=0.001, help="Decay factor for lateral inhibition updates.")
    parser.add_argument(
        "--lateral-w-hebb",
        type=float,
        default=1.0,
        help="Global/mixed L: weight on raw co-activation outer product y^T y (legacy default 1).",
    )
    parser.add_argument(
        "--lateral-w-anti",
        type=float,
        default=0.0,
        help="Global/mixed L: weight on anti-Hebbian term -y^T y (decorrelation vs raw co-activation).",
    )
    parser.add_argument(
        "--lateral-w-cov",
        type=float,
        default=0.0,
        help="Global/mixed L: weight on batch-centered covariance (y-ȳ)^T(y-ȳ).",
    )
    parser.add_argument(
        "--lateral-w-holo",
        type=float,
        default=0.05,
        help="Global/mixed L: weight on holographic-inspired cosine associative co-activity term.",
    )
    parser.add_argument(
        "--lateral-w-hyp",
        type=float,
        default=0.05,
        help="Global/mixed L: weight on hyperbolic-inspired Gram after Poincaré embedding of batch-centered rows.",
    )
    parser.add_argument(
        "--lateral-w-wave",
        type=float,
        default=0.05,
        help="Global/mixed L: weight on wavelet-inspired Gram of Haar low-pass (along batch) per neuron.",
    )
    parser.add_argument(
        "--lateral-w-oja",
        type=float,
        default=0.0,
        help="Global/mixed L: weight on Oja-like stabilizer L * mean(y_i^2+y_j^2)/2 (scales decay with activity).",
    )
    parser.add_argument("--inhibition-min", type=float, default=0.0, help="Lower clamp for adaptive inhibition strength.")
    parser.add_argument("--inhibition-max", type=float, default=1.0, help="Upper clamp for adaptive inhibition strength.")
    parser.add_argument("--wta-l1", type=float, default=0.08, help="WTA sparsity for L1 (higher = sparser).")
    parser.add_argument(
        "--inhibition-learning-sparsity",
        type=float,
        default=0.5,
        help="Sparsity on lateral inhibition learning updates (higher = sparser).",
    )
    parser.add_argument("--wta-l234", type=float, default=0.04, help="WTA sparsity for L2-L4 (higher = sparser).")
    parser.add_argument("--holo-alpha", type=float, default=0.15, help="Strength multiplier for holographic binding updates.")
    parser.add_argument("--holo-update-freq", type=int, default=20, help="Apply expensive holographic updates every N local updates.")
    parser.add_argument("--holo-fast-lr", type=float, default=0.1, help="Learning rate for fast holographic memory trace.")
    parser.add_argument("--holo-fast-decay", type=float, default=0.1, help="Decay for fast holographic memory trace.")
    parser.add_argument(
        "--holo-corr-blend",
        type=float,
        default=0.2,
        help="HRR: mix circular correlation with convolution in FFT domain (0=conv only, 1=corr only); orthonormal FFT.",
    )
    parser.add_argument(
        "--holo-fast-norm-cap",
        type=float,
        default=12.0,
        help="Cap Frobenius norm of running fast holographic trace M_holo_fast (0=disable).",
    )
    parser.add_argument("--gamma-wavelet", type=float, default=0.01, help="Wavelet binding strength (default 0.01).")
    parser.add_argument("--wavelet-threshold", type=float, default=0.02, help="Soft-threshold for wavelet denoising (0=off).")
    parser.add_argument("--no-wavelet-denoise", action="store_true", help="Disable wavelet denoising before Hebbian.")
    parser.add_argument("--no-wavelet-binding", action="store_true", help="Disable wavelet-domain binding term.")
    add_bool_arg(
        "wavelet-input",
        default=True,
        help_text="Concatenate Haar wavelet bands with Gabor per stream (27 ch). Off = Gabor-only input (24 ch).",
    )
    parser.add_argument(
        "--unsup-mix-mode",
        type=str,
        default="adaptive",
        choices=["fixed", "adaptive", "random"],
        help="How to mix unsupervised local terms: fixed, adaptive (smart), or random.",
    )
    parser.add_argument(
        "--no-unsup-mix-normalize-terms",
        action="store_true",
        help="Disable per-term normalization before unsupervised mixing.",
    )
    parser.add_argument(
        "--unsup-mix-temperature",
        type=float,
        default=1.0,
        help="Adaptive mixing strength; larger values favor inverse-norm balancing more.",
    )
    parser.add_argument(
        "--unsup-mix-random-alpha",
        type=float,
        default=20.0,
        help="Dirichlet concentration for random mixing (higher=closer to base weights).",
    )
    add_bool_arg(
        "use-step-b-predictive",
        default=False,
        help_text="Enable Step B predictive-coding consistency term.",
    )
    parser.add_argument("--step-b-weight", type=float, default=0.1, help="Weight for Step B predictive-coding term.")
    add_bool_arg(
        "use-step-c-slot",
        default=False,
        help_text="Enable Step C lightweight slot-binding term.",
    )
    parser.add_argument("--step-c-weight", type=float, default=0.05, help="Weight for Step C slot-binding objective.")
    parser.add_argument("--step-c-num-slots", type=int, default=4, help="Number of slots for Step C slot-binding objective.")
    parser.add_argument("--unsup-mix-w-hebb", type=float, default=0.30, help="Base mixture weight for Hebbian term.")
    parser.add_argument("--unsup-mix-w-holo", type=float, default=0.10, help="Base mixture weight for holographic term.")
    parser.add_argument("--unsup-mix-w-hyp", type=float, default=0.10, help="Base mixture weight for hyperbolic term.")
    parser.add_argument("--unsup-mix-w-wave", type=float, default=0.15, help="Base mixture weight for wavelet-binding term.")
    parser.add_argument("--unsup-mix-w-anti", type=float, default=0.10, help="Base mixture weight for anti-Hebbian term.")
    parser.add_argument("--unsup-mix-w-cons", type=float, default=0.08, help="Base mixture weight for consistency term.")
    parser.add_argument("--unsup-mix-w-rec", type=float, default=0.08, help="Base mixture weight for recursive term.")
    parser.add_argument("--unsup-mix-w-free", type=float, default=0.06, help="Base mixture weight for free-energy term.")
    parser.add_argument("--unsup-mix-w-decay", type=float, default=0.03, help="Base mixture weight for decay term.")
    parser.add_argument("--unsup-mix-w-dist", type=float, default=0.00, help="Base mixture weight for distance-gradient term.")
    add_bool_arg(
        "cascade-skip-connections",
        default=True,
        help_text="Cascade L2-L4 with skip maps: L2<-[L1,mean(L1 input)], L3<-[L2,L1], L4<-[L3,L2,L1].",
    )
    add_bool_arg(
        "neuron-glia",
        default=True,
        help_text="Slow glial EMA from L1–L4 mean |activity| gates local Hebbian variance/inhibition per layer (no CE backprop).",
    )
    parser.add_argument(
        "--glia-state-dim",
        type=int,
        default=8,
        help="Dimension of the glial slow state (per VisNet stack).",
    )
    parser.add_argument(
        "--glia-ema",
        type=float,
        default=0.995,
        help="EMA decay for glial trace (higher = slower homeostasis).",
    )
    parser.add_argument(
        "--glia-neuron-strength",
        type=float,
        default=0.12,
        help="How strongly glial state tilts per-layer plasticity gates (before clamp).",
    )
    parser.add_argument("--glia-gate-min", type=float, default=0.75, help="Lower clamp on glia→neuron multipliers.")
    parser.add_argument("--glia-gate-max", type=float, default=1.25, help="Upper clamp on glia→neuron multipliers.")
    add_bool_arg(
        "structural-plasticity",
        default=True,
        help_text="Enable structural plasticity (periodic prune/grow on local W).",
    )
    parser.add_argument(
        "--structural-update-freq",
        type=int,
        default=1200,
        help="Apply structural plasticity every N local updates (higher = gentler on long runs).",
    )
    parser.add_argument("--structural-prune-threshold", type=float, default=1e-4, help="Prune synapses with |w| below this threshold.")
    parser.add_argument("--structural-prune-max-frac", type=float, default=0.002, help="Max fraction of synapses pruned per structural step.")
    parser.add_argument("--structural-grow-threshold", type=float, default=0.02, help="Grow synapses where Hebbian drive exceeds this threshold.")
    parser.add_argument("--structural-grow-max-frac", type=float, default=0.001, help="Max fraction of synapses grown per structural step.")
    parser.add_argument("--structural-grow-init-scale", type=float, default=0.02, help="Initial magnitude for newly grown synapses.")
    add_bool_arg(
        "pc-per-neuron-plasticity",
        default=True,
        help_text="Per-neuron PC gain: blend layer-local |y| map + PFC TE map + trace surprise (weights sum in update).",
    )
    parser.add_argument(
        "--pc-per-neuron-layer-weight",
        type=float,
        default=0.34,
        help="Blend weight for this layer's spatial |activity| feedback (independent per L1–L4 / MT / PP).",
    )
    parser.add_argument(
        "--pc-per-neuron-pfc-weight",
        type=float,
        default=0.33,
        help="Blend weight for PFC topographic map (per frequency band).",
    )
    parser.add_argument(
        "--pc-per-neuron-trace-weight",
        type=float,
        default=0.33,
        help="Blend weight for local trace prediction error (per neuron).",
    )
    parser.add_argument("--pc-per-neuron-gain-min", type=float, default=0.5, help="Lower clamp for per-neuron plasticity gains.")
    parser.add_argument("--pc-per-neuron-gain-max", type=float, default=1.5, help="Upper clamp for per-neuron plasticity gains.")
    return parser


def run_training(args: argparse.Namespace) -> TrainingRunResult:
    if not hasattr(args, "eval_test_on_best_val"):
        args.eval_test_on_best_val = not bool(getattr(args, "no_eval_test_on_best_val", False))
    if str(getattr(args, "dorsal4_attention_preset", "active")).lower() == "minimal":
        args.symmetry_gate_prior = False
        args.it_pp_cross_gate = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    print(f"Running: {__file__}")
    print(f"Device: {device}")
    print("Dataset: CIFAR-10")

    # CIFAR-10: 32x32 RGB, 10 classes. Standard normalization.
    transform_train = transforms.Compose(
        [
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    data_root = os.path.expanduser(str(args.data_dir))
    os.makedirs(data_root, exist_ok=True)

    # CIFAR-10: 50k train, 10k test (official split)
    full_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    # Split training set into train/val (reproducible)
    val_frac = float(max(0.0, min(0.5, args.val_split)))
    n_train_total = len(full_train)
    n_val = int(round(val_frac * n_train_total))
    n_train = n_train_total - n_val

    g = torch.Generator().manual_seed(int(args.seed))
    perm = torch.randperm(n_train_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    train_frac = float(max(0.0, min(1.0, args.train_fraction)))
    if train_frac < 1.0:
        keep_n = max(1, int(round(train_frac * len(train_idx))))
        train_idx = train_idx[:keep_n]

    train_ds = torch.utils.data.Subset(full_train, train_idx)
    val_ds = torch.utils.data.Subset(
        datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_test),
        val_idx,
    )

    num_classes = 10

    # Plasticity coefficients (weight decay, wavelet terms)
    coeffs = UnifiedCoeffs(
        alpha=float(args.holo_alpha),
        holo_update_freq=int(args.holo_update_freq),
        holo_fast_lr=float(args.holo_fast_lr),
        holo_fast_decay=float(args.holo_fast_decay),
        holo_corr_blend=float(getattr(args, "holo_corr_blend", 0.2)),
        holo_fast_norm_cap=float(getattr(args, "holo_fast_norm_cap", 12.0)),
        lambda_d=float(args.lambda_d),
        gamma_wavelet=float(args.gamma_wavelet),
        wavelet_threshold=float(args.wavelet_threshold),
        use_wavelet_denoise=not args.no_wavelet_denoise,
        use_wavelet_binding=not args.no_wavelet_binding,
        use_wavelet_input=bool(args.wavelet_input),
        use_entropy_dropout=args.entropy_dropout,
        entropy_dropout_scale=float(args.entropy_dropout_scale),
        use_entropy_plasticity_decay=args.entropy_plasticity_decay,
        entropy_decay_scale=float(args.entropy_decay_scale),
        inhibition_mode=str(args.inhibition_mode),
        use_ei_neurons=bool(args.use_ei_neurons),
        ei_mutual_inhibition=bool(args.ei_mutual_inhibition),
        ei_separate_lateral=bool(args.ei_separate_lateral),
        ei_l_ei_init=float(args.ei_l_ei_init),
        ei_l_ie_init=float(args.ei_l_ie_init),
        use_soft_inhibition=bool(args.soft_inhibition),
        inhibition_no_threshold=bool(args.inhibition_no_threshold),
        inhibition_smooth_scale=float(args.inhibition_smooth_scale),
        averaging_inhibition_size=int(args.averaging_inhibition_size),
        kernel_inhibition_size=int(args.kernel_inhibition_size),
        kernel_inhibition_sigma=float(args.kernel_inhibition_sigma),
        mixed_inhibition_w_global=float(args.mixed_inhibition_w_global),
        mixed_inhibition_w_kernel=float(args.mixed_inhibition_w_kernel),
        mixed_inhibition_w_averaging=float(args.mixed_inhibition_w_averaging),
        inhibition_dropout=float(args.inhibition_dropout),
        use_divisive_inhibition_norm=bool(args.divisive_inhibition_norm),
        divisive_mode=str(args.divisive_mode),
        divisive_w_global=float(args.divisive_w_global),
        divisive_w_local=float(args.divisive_w_local),
        divisive_local_size=int(args.divisive_local_size),
        divisive_local_sigma=float(args.divisive_local_sigma),
        divisive_alpha=float(args.divisive_alpha),
        divisive_beta=float(args.divisive_beta),
        adaptive_inhibition=bool(args.adaptive_inhibition),
        target_active_frac=float(args.target_active_frac),
        inhibition_adapt_lr=float(args.inhibition_adapt_lr),
        inhibition_min=float(args.inhibition_min),
        inhibition_max=float(args.inhibition_max),
        use_homeostatic_threshold=bool(args.use_homeostatic_threshold),
        homeostatic_threshold_lr=float(args.homeostatic_threshold_lr),
        homeostatic_threshold_theta_init=float(args.homeostatic_threshold_theta_init),
        homeostatic_threshold_min=float(args.homeostatic_threshold_min),
        homeostatic_threshold_max=float(args.homeostatic_threshold_max),
        use_oscillatory_inhibition=bool(args.oscillatory_inhibition),
        phase_period=int(args.phase_period),
        phase_gate_sharpness=float(args.phase_gate_sharpness),
        inhibition_learning_sparsity=float(args.inhibition_learning_sparsity),
        lateral_w_hebb=float(args.lateral_w_hebb),
        lateral_w_anti=float(args.lateral_w_anti),
        lateral_w_cov=float(args.lateral_w_cov),
        lateral_w_holo=float(args.lateral_w_holo),
        lateral_w_hyp=float(args.lateral_w_hyp),
        lateral_w_wave=float(args.lateral_w_wave),
        lateral_w_oja=float(args.lateral_w_oja),
        unsup_mix_mode=str(args.unsup_mix_mode),
        unsup_mix_normalize_terms=not bool(args.no_unsup_mix_normalize_terms),
        unsup_mix_temperature=float(args.unsup_mix_temperature),
        unsup_mix_random_alpha=float(args.unsup_mix_random_alpha),
        unsup_mix_w_hebb=float(args.unsup_mix_w_hebb),
        unsup_mix_w_holo=float(args.unsup_mix_w_holo),
        unsup_mix_w_hyp=float(args.unsup_mix_w_hyp),
        unsup_mix_w_wave=float(args.unsup_mix_w_wave),
        unsup_mix_w_anti=float(args.unsup_mix_w_anti),
        unsup_mix_w_cons=float(args.unsup_mix_w_cons),
        unsup_mix_w_rec=float(args.unsup_mix_w_rec),
        unsup_mix_w_free=float(args.unsup_mix_w_free),
        unsup_mix_w_decay=float(args.unsup_mix_w_decay),
        unsup_mix_w_dist=float(args.unsup_mix_w_dist),
        use_structural_plasticity=bool(args.structural_plasticity),
        structural_update_freq=int(args.structural_update_freq),
        structural_prune_threshold=float(args.structural_prune_threshold),
        structural_prune_max_frac=float(args.structural_prune_max_frac),
        structural_grow_threshold=float(args.structural_grow_threshold),
        structural_grow_max_frac=float(args.structural_grow_max_frac),
        structural_grow_init_scale=float(args.structural_grow_init_scale),
        pc_per_neuron_plasticity=bool(args.pc_per_neuron_plasticity),
        pc_per_neuron_layer_weight=float(args.pc_per_neuron_layer_weight),
        pc_per_neuron_pfc_weight=float(args.pc_per_neuron_pfc_weight),
        pc_per_neuron_trace_weight=float(args.pc_per_neuron_trace_weight),
        pc_per_neuron_gain_min=float(args.pc_per_neuron_gain_min),
        pc_per_neuron_gain_max=float(args.pc_per_neuron_gain_max),
        som_enabled=bool(args.som_inhibition_schedules),
        kernel_inhibition_sigma_start=float(args.kernel_inhibition_sigma_start),
        kernel_inhibition_sigma_end=float(args.kernel_inhibition_sigma_end),
        som_lr_lateral_warmup_fraction=float(args.som_lr_lateral_warmup_fraction),
        lr_lateral_schedule_end_scale=float(args.lr_lateral_schedule_end_scale),
        use_inhibition_softmax=bool(args.inhibition_softmax),
        inhibition_softmax_temp_start=float(args.inhibition_softmax_temp_start),
        inhibition_softmax_temp_end=float(args.inhibition_softmax_temp_end),
        som_wta_scale_start=float(args.som_wta_scale_start),
        som_wta_scale_end=float(args.som_wta_scale_end),
        mexican_center_sigma_start=float(args.mexican_center_sigma_start),
        mexican_center_sigma_end=float(args.mexican_center_sigma_end),
        mexican_surround_sigma_start=float(args.mexican_surround_sigma_start),
        mexican_surround_sigma_end=float(args.mexican_surround_sigma_end),
        mexican_surround_gain_start=float(args.mexican_surround_gain_start),
        mexican_surround_gain_end=float(args.mexican_surround_gain_end),
        cascade_skip_connections=bool(args.cascade_skip_connections),
        rf_connectivity_gaussian=bool(args.rf_gaussian_connectivity),
        rf_gaussian_sigma_frac=float(args.rf_gaussian_sigma_frac),
        rf_connectivity_keep_frac=float(args.rf_connectivity_keep_frac),
        rf_connectivity_sparse_quantile=float(args.rf_sparse_quantile),
    )
    # Keep all unsupervised methods active; variance objective is an additional influence.
    variance_only_unsup = False

    _nw = int(getattr(args, "dataloader_workers", -1))
    if _nw < 0:
        _nw = min(4, os.cpu_count() or 1)
    num_workers = max(0, _nw)
    _pin = device.type == "cuda"
    _dl_common = {"num_workers": num_workers, "pin_memory": _pin}
    if num_workers > 0:
        _dl_common["persistent_workers"] = True
        _dl_common["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **_dl_common)
    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size, shuffle=False, **_dl_common)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, **_dl_common)

    _vn_kwargs = dict(
        device=str(device),
        num_classes=num_classes,
        spatial_size=args.spatial_size,
        auto_resize_input=not args.no_auto_resize,
        coeffs=coeffs,
        use_hebbian=True,
        use_antihebbian=True,
        use_holographic=True,
        use_consistency=True,
        use_recursive=True,
        use_free_energy=True,
        use_active_inference=True,
        dropout=args.dropout,
        inhibition_decay=float(args.inhibition_decay),
        use_pfc_hopfield=bool(args.use_pfc_hopfield),
        pfc_hopfield_patterns=int(args.pfc_hopfield_patterns),
        pfc_hopfield_beta=float(args.pfc_hopfield_beta),
        pfc_hopfield_temperature=float(args.pfc_hopfield_temperature),
        pfc_hopfield_blend=float(args.pfc_hopfield_blend),
        pfc_hopfield_ema_lr=float(args.pfc_hopfield_ema_lr),
        pfc_hopfield_unsup_update=bool(args.pfc_hopfield_unsup_update),
        pfc_hopfield_cosine=bool(args.pfc_hopfield_cosine),
        pfc_hopfield_soft_ema=bool(args.pfc_hopfield_soft_ema),
        pfc_hopfield_normalize_memory=bool(args.pfc_hopfield_normalize_memory),
        pfc_hopfield_sparsity=float(args.pfc_hopfield_sparsity),
        pfc_hopfield_sparse_update=bool(args.pfc_hopfield_sparse_update),
        pfc_hopfield_layernorm=bool(args.pfc_hopfield_layernorm),
        pfc_mode=str(args.pfc_mode),
        pfc_hebbian_lr=float(args.pfc_hebbian_lr),
        pfc_hebbian_decay=float(args.pfc_hebbian_decay),
        pfc_sa_head_dim=int(args.pfc_sa_head_dim),
        pfc_topdown_attention=bool(args.pfc_topdown_attention),
        pfc_topdown_strength=float(args.pfc_topdown_strength),
        pfc_topdown_min_scale=float(args.pfc_topdown_min_scale),
        pfc_topdown_max_scale=float(args.pfc_topdown_max_scale),
        pfc_topdown_unsup_lr=float(args.pfc_topdown_unsup_lr),
        pfc_topdown_decay=float(args.pfc_topdown_decay),
        pfc_topdown_per_neuron=bool(args.pfc_topdown_per_neuron),
        pfc_topdown_neuron_use_bias=bool(args.pfc_topdown_neuron_bias),
        pfc_topdown_shared_fe_blend=float(args.pfc_topdown_shared_fe_blend),
        pfc_inhibition_feedback_unsup_lr=float(args.pfc_inhibition_feedback_unsup_lr),
        pfc_inhibition_feedback_decay=float(args.pfc_inhibition_feedback_decay),
        pfc_topdown_iters=int(args.pfc_topdown_iters),
        pfc_predictive_feedback=bool(args.pfc_predictive_feedback),
        pfc_predictive_strength=float(args.pfc_predictive_strength),
        pfc_predictive_min_scale=float(args.pfc_predictive_min_scale),
        pfc_predictive_max_scale=float(args.pfc_predictive_max_scale),
        pfc_predictive_unsup_lr=float(args.pfc_predictive_unsup_lr),
        pfc_predictive_decay=float(args.pfc_predictive_decay),
        use_pfc_pc_layer_output_mask=bool(args.pfc_pc_layer_output_mask),
        pfc_pc_layer_mask_lr=float(args.pfc_pc_layer_mask_lr),
        pfc_pc_layer_mask_decay=float(args.pfc_pc_layer_mask_decay),
        use_neuron_glia=bool(getattr(args, "neuron_glia", True)),
        glia_state_dim=int(getattr(args, "glia_state_dim", 8)),
        glia_ema=float(getattr(args, "glia_ema", 0.995)),
        glia_neuron_strength=float(getattr(args, "glia_neuron_strength", 0.12)),
        glia_gate_min=float(getattr(args, "glia_gate_min", 0.75)),
        glia_gate_max=float(getattr(args, "glia_gate_max", 1.25)),
        pfc_l1_lambda=float(args.pfc_l1_lambda),
        pfc_l1_prox_step=float(args.pfc_l1_prox_step),
        local_l1_lambda=float(args.local_l1_lambda),
        local_l1_prox_step=float(args.local_l1_prox_step),
        local_l1_warmup_steps=int(args.local_l1_warmup_steps),
        local_l1_apply_every=int(args.local_l1_apply_every),
        wta_l1=float(args.wta_l1),
        wta_l234=float(args.wta_l234),
        rf_l1=int(args.rf_l1),
        rf_l2=int(args.rf_l2),
        rf_l3=int(args.rf_l3),
        rf_l4=int(args.rf_l4),
        recursive_iters=int(args.recursive_iters),
        use_dorsal_stream=not bool(args.no_dorsal_stream),
        pfc_spatial_readout_gate=bool(args.pfc_spatial_readout_gate),
        pfc_spatial_gate_strength=float(args.pfc_spatial_gate_strength),
        pfc_spatial_gate_floor=float(args.pfc_spatial_gate_floor),
        pfc_recurrent_feedback_steps=int(args.pfc_recurrent_feedback_steps),
        pfc_recurrent_feedback_strength=float(args.pfc_recurrent_feedback_strength),
        use_pfc_dense_feedback=bool(args.pfc_dense_feedback),
        pfc_dense_feedback_strength=float(args.pfc_dense_feedback_strength),
        use_pfc_deep_feedback=bool(args.pfc_deep_feedback),
        pfc_deep_feedback_rank=int(args.pfc_deep_feedback_rank),
        pfc_deep_fb_strength_l2=float(args.pfc_deep_fb_l2),
        pfc_deep_fb_strength_l3=float(args.pfc_deep_fb_l3),
        pfc_deep_fb_strength_l4=float(args.pfc_deep_fb_l4),
        pfc_deep_fb_strength_mt=float(args.pfc_deep_fb_mt),
        pfc_deep_fb_strength_pp=float(args.pfc_deep_fb_pp),
        use_symmetry_gate_prior=bool(args.symmetry_gate_prior),
        symmetry_gate_alpha=float(args.symmetry_gate_alpha),
        symmetry_prior_unsup_lr=float(args.symmetry_prior_unsup_lr),
        it_pp_cross_gate=bool(args.it_pp_cross_gate),
        it_pp_cross_pp_to_te=float(args.it_pp_cross_pp_to_te),
        it_pp_cross_te_to_pp=float(args.it_pp_cross_te_to_pp),
        it_pp_cross_iters=int(args.it_pp_cross_iters),
        pfc_pre_hopfield_fusion=str(args.pfc_pre_hopfield_fusion),
        pfc_pre_blend_w_te=float(args.pfc_pre_blend_w_te),
        pfc_pre_blend_w_pp=float(args.pfc_pre_blend_w_pp),
        pfc_post_readout_fusion=str(args.pfc_post_readout_fusion),
        use_pfc_fusion_gate_unsup=bool(args.pfc_fusion_gate_unsup),
        pfc_fusion_gate_unsup_lr=float(args.pfc_fusion_gate_unsup_lr),
        pfc_fusion_gate_decay=float(args.pfc_fusion_gate_decay),
        pfc_fusion_lms_chunk_rows=int(args.pfc_fusion_lms_chunk_rows),
    )
    model = VisNetUnified(**_vn_kwargs)
    if bool(getattr(args, "neuron_glia", True)):
        print(
            "Neuron–glia coupling ON (default): slow EMA gates per-layer local plasticity "
            f"(dim={int(getattr(args, 'glia_state_dim', 8))}, ema={float(getattr(args, 'glia_ema', 0.995)):.4f}, "
            f"strength={float(getattr(args, 'glia_neuron_strength', 0.12)):.3g}, clamp "
            f"[{float(getattr(args, 'glia_gate_min', 0.75)):.2f}, {float(getattr(args, 'glia_gate_max', 1.25)):.2f}]). "
            "Use --no-neuron-glia to disable."
        )
    if bool(getattr(args, "compile_model", False)) and device.type == "cuda":
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("torch.compile: enabled on model.")
        except Exception as e:
            print(f"torch.compile: skipped ({e}).")
    counts = model.count_parameters()
    print(f"Cascade skip connections: {bool(getattr(coeffs, 'cascade_skip_connections', True))}")
    print("\nPARAMETER SUMMARY (paper-aligned)")
    for k in [
        "DoG_fixed",
        "Gabor_fixed",
        "Wavelet_fixed",
        "L1_Wb",
        "L2_Wb",
        "L3_Wb",
        "L4_Wb",
        "Classifier_grad",
        "PFC_grad",
        "PFC_topdown_local",
        "PFC_predictive_local",
        "TOTAL",
        "Grad_%",
    ]:
        print(f"  {k}: {counts[k]}")
    print()

    # AdamW on classifier only with cosine LR decay for smoother convergence.
    clf_lr0 = float(args.clf_lr)
    optimizer = optim.AdamW(model.classifier_parameters(), lr=clf_lr0, weight_decay=float(args.weight_decay))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=max(1e-6, clf_lr0 * 0.1),
    )
    use_amp = (device.type == "cuda") and (not bool(args.no_amp))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Model selection should be based on validation, not test.
    best_val_acc = -1.0
    best_epoch = -1
    best_path = str(args.save_best_path)
    best_test_at_best_val = None

    # Print split info
    print(
        f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"(val_frac={val_frac:.3f}, train_frac={train_frac:.3f}, seed={args.seed})"
    )
    print(f"DataLoader: num_workers={num_workers}, pin_memory={_pin}")
    print(f"RF sizes: L1={args.rf_l1}, L2={args.rf_l2}, L3={args.rf_l3}, L4={args.rf_l4}")
    print(f"Recursive inhibition iterations per layer: {int(args.recursive_iters)}")
    if not args.save_best:
        print("Note: --save-best is OFF (recommended for speed). Test will be evaluated at the final epoch only.")
        if args.eval_test_on_best_val:
            print("Note: --eval-test-on-best-val is ON (this peeks at test occasionally; do not use for paper reporting).")

    # CIFAR-10 class names for printing last-batch labels
    cifar10_classes = getattr(
        getattr(train_loader.dataset, "dataset", train_loader.dataset),
        "classes",
        [str(i) for i in range(10)],
    )

    lambda_d_max = float(args.lambda_d)
    dropout_max = float(args.dropout)
    schedule = not args.no_schedule_decay_dropout
    ld_floor_r = max(0.0, float(getattr(args, "lambda_d_floor_ratio", 0.0)))
    do_floor_r = max(0.0, float(getattr(args, "dropout_floor_ratio", 0.0)))
    if schedule:
        print(
            f"Schedule: lambda_d and dropout decay from {lambda_d_max}/{dropout_max} "
            f"toward floor ratios {ld_floor_r:.2g}/{do_floor_r:.2g} of max over {args.epochs} epochs "
            f"(not to absolute 0 unless floors are 0)."
        )
    fe = int(getattr(args, "freeze_local_plasticity_after_epoch", 0))
    if fe > 0:
        print(f"Local plasticity (Hebbian) will freeze after epoch {fe} (classifier + val/test still run).")
    auto_freeze_on_collapse = bool(getattr(args, "auto_freeze_on_collapse", True))
    collapse_drop_thr = float(max(0.0, getattr(args, "collapse_val_drop_threshold", 8.0)))
    collapse_patience = int(max(0, getattr(args, "collapse_patience_epochs", 1)))
    forced_freeze = False
    if auto_freeze_on_collapse:
        print(
            "Auto-freeze-on-collapse: "
            f"enabled (trigger if val drops >= {collapse_drop_thr:.2f} pts after {collapse_patience} epoch patience)."
        )
    print(
        "Unsupervised mixer: "
        f"mode={args.unsup_mix_mode}, "
        f"normalize_terms={not bool(args.no_unsup_mix_normalize_terms)}, "
        f"temp={float(args.unsup_mix_temperature):.3g}, "
        f"rand_alpha={float(args.unsup_mix_random_alpha):.3g}"
    )
    print(
        "Structural plasticity: "
        f"enabled={bool(args.structural_plasticity)}, "
        f"freq={int(args.structural_update_freq)}, "
        f"prune_thr={float(args.structural_prune_threshold):.3g}, "
        f"grow_thr={float(args.structural_grow_threshold):.3g}"
    )
    print(
        "Step options: "
        f"B_predictive={bool(args.use_step_b_predictive)}(w={float(args.step_b_weight):.3g}), "
        f"C_slot={bool(args.use_step_c_slot)}(w={float(args.step_c_weight):.3g}, k={int(args.step_c_num_slots)})"
    )
    print(
        "PFC Hopfield: "
        f"enabled={bool(args.use_pfc_hopfield)}, "
        f"mode={str(args.pfc_mode)}, "
        f"warmup_epochs={int(max(0, args.pfc_warmup_epochs))}, "
        f"patterns={int(args.pfc_hopfield_patterns)}, "
        f"hebbian_lr={float(args.pfc_hebbian_lr):.3g}, "
        f"hebbian_decay={float(args.pfc_hebbian_decay):.3g}, "
        f"sa_head_dim={int(args.pfc_sa_head_dim)}, "
        f"beta={float(args.pfc_hopfield_beta):.3g}, "
        f"temp={float(args.pfc_hopfield_temperature):.3g}, "
        f"blend={float(args.pfc_hopfield_blend):.3g}, "
        f"ema_lr={float(args.pfc_hopfield_ema_lr):.3g}, "
        f"unsup_update={bool(args.pfc_hopfield_unsup_update)}, "
        f"cosine={bool(args.pfc_hopfield_cosine)}, "
        f"soft_ema={bool(args.pfc_hopfield_soft_ema)}, "
        f"norm_mem={bool(args.pfc_hopfield_normalize_memory)}, "
        f"sparsity={float(args.pfc_hopfield_sparsity):.3g}, "
        f"sparse_update={bool(args.pfc_hopfield_sparse_update)}, "
        f"cons_w={float(args.pfc_consistency_weight):.3g}, "
        f"topdown_attn={bool(args.pfc_topdown_attention)}, "
        f"td_strength={float(args.pfc_topdown_strength):.3g}, "
        f"td_range=[{float(args.pfc_topdown_min_scale):.3g},{float(args.pfc_topdown_max_scale):.3g}], "
        f"td_unsup_lr={float(args.pfc_topdown_unsup_lr):.3g}, "
        f"td_decay={float(args.pfc_topdown_decay):.3g}, "
        f"td_shared_fe={float(args.pfc_topdown_shared_fe_blend):.3g}, "
        f"inh_unsup_lr={float(args.pfc_inhibition_feedback_unsup_lr):.3g}, "
        f"inh_decay={float(args.pfc_inhibition_feedback_decay):.3g}, "
        f"td_iters={int(max(1, args.pfc_topdown_iters))}, "
        f"spatial_readout_gate={bool(args.pfc_spatial_readout_gate)}, "
        f"sp_gate_s={float(args.pfc_spatial_gate_strength):.3g}, "
        f"sp_gate_floor={float(args.pfc_spatial_gate_floor):.3g}, "
        f"recurrent_steps={int(max(1, min(2, args.pfc_recurrent_feedback_steps)))}, "
        f"recurrent_fb_strength={float(args.pfc_recurrent_feedback_strength):.3g}, "
        f"dense_pfc_fb={bool(args.pfc_dense_feedback)}, "
        f"dense_pfc_fb_s={float(args.pfc_dense_feedback_strength):.3g}, "
        f"deep_pfc_fb={bool(args.pfc_deep_feedback)}, "
        f"deep_r={int(args.pfc_deep_feedback_rank)}, "
        f"deep_l2={float(args.pfc_deep_fb_l2):.3g}, "
        f"deep_l3={float(args.pfc_deep_fb_l3):.3g}, "
        f"deep_l4={float(args.pfc_deep_fb_l4):.3g}, "
        f"deep_mt={float(args.pfc_deep_fb_mt):.3g}, "
        f"deep_pp={float(args.pfc_deep_fb_pp):.3g}, "
        f"d4_attn_preset={str(getattr(args, 'dorsal4_attention_preset', 'active'))}, "
        f"sym_gate_prior={bool(args.symmetry_gate_prior)}, "
        f"sym_alpha={float(args.symmetry_gate_alpha):.3g}, "
        f"sym_unsup_lr={float(args.symmetry_prior_unsup_lr):.3g}, "
        f"it_pp_cross={bool(args.it_pp_cross_gate)}, "
        f"it_pp_a={float(args.it_pp_cross_pp_to_te):.3g}, "
        f"it_pp_b={float(args.it_pp_cross_te_to_pp):.3g}, "
        f"it_pp_K={int(max(1, args.it_pp_cross_iters))}, "
        f"pfc_pre_fusion={str(args.pfc_pre_hopfield_fusion)}, "
        f"pfc_post_readout={str(args.pfc_post_readout_fusion)}, "
        f"pfc_fusion_unsup={bool(args.pfc_fusion_gate_unsup)}, "
        f"pfc_fusion_lr={float(args.pfc_fusion_gate_unsup_lr):.3g}, "
        f"pfc_fusion_decay={float(args.pfc_fusion_gate_decay):.3g}, "
        f"pred_fb={bool(args.pfc_predictive_feedback)}, "
        f"pred_strength={float(args.pfc_predictive_strength):.3g}, "
        f"pred_range=[{float(args.pfc_predictive_min_scale):.3g},{float(args.pfc_predictive_max_scale):.3g}], "
        f"pred_unsup_lr={float(args.pfc_predictive_unsup_lr):.3g}, "
        f"pred_decay={float(args.pfc_predictive_decay):.3g}, "
        f"pc_layer_mask={bool(args.pfc_pc_layer_output_mask)}, "
        f"pc_mask_lr={float(args.pfc_pc_layer_mask_lr):.3g}, "
        f"pc_mask_decay={float(args.pfc_pc_layer_mask_decay):.3g}, "
        f"pfc_l1_lambda={float(args.pfc_l1_lambda):.3g}, "
        f"pfc_l1_prox_step={float(args.pfc_l1_prox_step):.3g}, "
        f"local_l1_lambda={float(args.local_l1_lambda):.3g}, "
        f"local_l1_prox_step={float(args.local_l1_prox_step):.3g}, "
        f"local_l1_warmup_steps={int(max(0, args.local_l1_warmup_steps))}, "
        f"local_l1_apply_every={int(max(1, args.local_l1_apply_every))}, "
        f"neuron_glia={bool(getattr(args, 'neuron_glia', True))}, "
        f"glia_dim={int(getattr(args, 'glia_state_dim', 8))}, "
        f"glia_ema={float(getattr(args, 'glia_ema', 0.995)):.3g}, "
        f"glia_strength={float(getattr(args, 'glia_neuron_strength', 0.12)):.3g}"
    )
    if float(args.variance_reg_weight) > 0.0:
        print(
            "Hebbian variance regularizer ON (applied to L1-L4): "
            f"lambda={float(args.variance_reg_weight):.4g}, "
            f"intra_w={float(args.variance_intra_weight):.4g}, "
            f"inter_w={float(args.variance_inter_weight):.4g}, "
            f"memory_lr={float(args.variance_memory_lr):.4g}"
        )
        print("Unsupervised mode: full local unsupervised mixture active + variance modulation.")

    _ud = model.unsup_layer_feature_dims
    variance_class_memory = None
    if float(args.variance_reg_weight) > 0.0:
        variance_class_memory = {
            "l1": torch.zeros(num_classes, int(_ud["l1"]), device=device),
            "l2": torch.zeros(num_classes, int(_ud["l2"]), device=device),
            "l3": torch.zeros(num_classes, int(_ud["l3"]), device=device),
            "l4": torch.zeros(num_classes, int(_ud["l4"]), device=device),
        }

    for epoch in range(1, args.epochs + 1):
        # Optional PFC warmup: keep PFC disabled for first N epochs, then enable.
        if hasattr(model, "use_pfc_hopfield") and model.pfc_hopfield is not None:
            warmup_epochs = int(max(0, args.pfc_warmup_epochs))
            pfc_active_now = bool(args.use_pfc_hopfield) and (epoch > warmup_epochs)
            model.use_pfc_hopfield = bool(pfc_active_now)

        # Variable weight decay and dropout: linear schedule from max (epoch 1) toward floor (last epoch)
        if schedule and args.epochs > 1:
            progress = (epoch - 1) / (args.epochs - 1)  # 0 at start, 1 at end
            current_lambda_d = max(
                lambda_d_max * ld_floor_r,
                lambda_d_max * (1.0 - progress),
            )
            current_dropout = max(
                dropout_max * do_floor_r,
                dropout_max * (1.0 - progress),
            )
            coeffs.lambda_d = current_lambda_d
            if hasattr(model, "dropout"):
                model.dropout.p = current_dropout

        scale_lr = float(optimizer.param_groups[0]["lr"]) / max(1e-12, clf_lr0)
        if hasattr(model, "apply_joint_plasticity_lr_scale"):
            model.apply_joint_plasticity_lr_scale(scale_lr, coeffs)

        plasticity_on = (not forced_freeze) and (fe <= 0 or epoch <= fe)

        # SOM-style inhibition schedules (kernel σ, Mexican-hat, lateral LR, softmax T, WTA)
        if hasattr(model, "apply_som_schedules"):
            model.apply_som_schedules(epoch, args.epochs, coeffs)

        tr_loss, tr_acc, last_pred, last_actual = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            show_progress=not bool(args.no_tqdm),
            use_amp=use_amp,
            scaler=scaler,
            label_smoothing=float(args.label_smoothing),
            variance_reg_weight=float(args.variance_reg_weight),
            variance_intra_weight=float(args.variance_intra_weight),
            variance_inter_weight=float(args.variance_inter_weight),
            variance_memory_lr=float(args.variance_memory_lr),
            variance_class_memory=variance_class_memory,
            variance_only_unsup=variance_only_unsup,
            use_step_b_predictive=bool(args.use_step_b_predictive),
            step_b_weight=float(args.step_b_weight),
            use_step_c_slot=bool(args.use_step_c_slot),
            step_c_weight=float(args.step_c_weight),
            step_c_num_slots=int(args.step_c_num_slots),
            pfc_consistency_weight=float(args.pfc_consistency_weight),
            local_plasticity=plasticity_on,
        )
        va_loss, va_acc = test_epoch(model, val_loader, device, use_amp=use_amp)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            if args.save_best:
                torch.save(model.state_dict(), best_path)
            if args.eval_test_on_best_val:
                te_loss_now, te_acc_now = test_epoch(model, test_loader, device, use_amp=use_amp)
                best_test_at_best_val = float(te_acc_now)
                print(
                    f"  NEW BEST VAL @ epoch {best_epoch}: "
                    f"Val {va_loss:.4f}/{best_val_acc:.2f}% | "
                    f"Test {te_loss_now:.4f}/{best_test_at_best_val:.2f}%"
                )
        elif (
            auto_freeze_on_collapse
            and (not forced_freeze)
            and plasticity_on
            and best_epoch > 0
            and epoch > (best_epoch + collapse_patience)
            and va_acc <= (best_val_acc - collapse_drop_thr)
        ):
            forced_freeze = True
            print(
                "  AUTO-FREEZE TRIGGERED: validation dropped "
                f"{best_val_acc - va_acc:.2f} pts from best ({best_val_acc:.2f}% -> {va_acc:.2f}%). "
                "Disabling local plasticity for remaining epochs."
            )

        if args.eval_test_each_epoch:
            te_loss, te_acc = test_epoch(model, test_loader, device, use_amp=use_amp)
            print(
                f"Epoch {epoch:3d} | Train: {tr_loss:.4f}/{tr_acc:.2f}% | "
                f"Val: {va_loss:.4f}/{va_acc:.2f}% | "
                f"Test: {te_loss:.4f}/{te_acc:.2f}% | BestVal: {best_val_acc:.2f}% @ epoch {best_epoch}"
            )
        else:
            print(
                f"Epoch {epoch:3d} | Train: {tr_loss:.4f}/{tr_acc:.2f}% | "
                f"Val: {va_loss:.4f}/{va_acc:.2f}% | BestVal: {best_val_acc:.2f}% @ epoch {best_epoch}"
            )

        # Last batch of epoch: predicted vs actual labels
        if last_pred is not None and last_actual is not None:
            pred_names = [cifar10_classes[i] for i in last_pred]
            actual_names = [cifar10_classes[i] for i in last_actual]
            print(f"  Last batch pred: {last_pred} ({pred_names})")
            print(f"  Last batch actual: {last_actual} ({actual_names})")

        scheduler.step()

    # Final reporting
    if args.save_best and os.path.exists(best_path):
        # Paper-style: load best-by-val checkpoint and evaluate test ONCE
        # Load state dict directly with weights_only for safer, warning-free checkpoint restore.
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        final_val_loss, final_val_acc = test_epoch(model, val_loader, device, use_amp=use_amp)
        final_te_loss, final_te_acc = test_epoch(model, test_loader, device, use_amp=use_amp)
        print(f"\nBest-by-val epoch: {best_epoch} | Val: {final_val_loss:.4f}/{final_val_acc:.2f}%")
        print(f"Final test (best-by-val): {final_te_loss:.4f}/{final_te_acc:.2f}%")
    else:
        # Fast path: test only once at the end (last-epoch model)
        final_val_loss, final_val_acc = test_epoch(model, val_loader, device, use_amp=use_amp)
        final_te_loss, final_te_acc = test_epoch(model, test_loader, device, use_amp=use_amp)
        print(f"\nBestVal: {best_val_acc:.2f}%@{best_epoch}")
        if best_test_at_best_val is not None:
            print(f"Test at best-val epoch (peeked): {best_test_at_best_val:.2f}%")
        print(f"Last-epoch Val: {final_val_loss:.4f}/{final_val_acc:.2f}%")
        print(f"Last-epoch Test: {final_te_loss:.4f}/{final_te_acc:.2f}%")

    if (
        best_epoch > 0
        and int(args.epochs) > int(best_epoch) + 1
        and (not args.save_best)
        and float(final_val_acc) < float(best_val_acc) - 15.0
    ):
        print(
            "\nSTABILITY NOTE: Last-epoch val collapsed vs best epoch (Hebbian/plasticity drift is common). "
            "Mitigations: --save-best; --freeze-local-plasticity-after-epoch set near your best epoch; "
            "defaults now use --lambda-d-floor-ratio/--dropout-floor-ratio 0.15, structural-update-freq 1200, "
            "and auto-freeze-on-collapse."
        )

    return TrainingRunResult(
        best_val_acc=float(best_val_acc),
        best_epoch=int(best_epoch),
        best_test_at_best_val=best_test_at_best_val,
        final_val_loss=float(final_val_loss),
        final_val_acc=float(final_val_acc),
        final_test_loss=float(final_te_loss),
        final_test_acc=float(final_te_acc),
    )


def main(argv: Optional[list[str]] = None) -> TrainingRunResult:
    """CLI entry: parse args (optionally from `argv`), run training, return metrics."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.eval_test_on_best_val = not bool(args.no_eval_test_on_best_val)
    return run_training(args)


if __name__ == "__main__":
    main()

