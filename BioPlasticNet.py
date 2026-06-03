"""
VisNet Unified Framework - SEPARATE FREQUENCY CHANNELS + WAVELET
=================================================================
Full fixed version.  Changes vs BioPlasticNet3_pinv2.py:

  1. fit_pseudoinverse_classifier is INSIDE VisNetUnified (correct indentation,
     uses self.clf and self.forward_features).
  2. _build_model() centralises model construction; supports:
       --inhibition-mode none   → forces inhibition_strength=0 everywhere
       --disable-pfc            → disables all PFC/predictive-coding paths
  3. ddp_worker actually calls fit_pseudoinverse_classifier when
     --readout-type pinv, and broadcasts clf weights to all ranks.
  4. All dist.* calls are guarded by world_size > 1 (single-GPU safe).
  5. TrainingRunResult constructed with all 7 required fields.
"""

from __future__ import annotations

import math
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

if not hasattr(torch.amp, "GradScaler"):
    torch.amp.GradScaler = torch.cuda.amp.GradScaler          # type: ignore
if not hasattr(torch.amp, "autocast"):
    torch.amp.autocast = torch.cuda.amp.autocast              # type: ignore


# ---------------------------------------------------------------------------
# Utility kernels
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(ksz: int, sigma: float,
                         device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ax = torch.arange(-(ksz // 2), ksz // 2 + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) ** 2))
    return (g / (g.sum() + 1e-8)).view(1, 1, ksz, ksz)


def _gaussian2d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    ax = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return g / (g.sum() + 1e-8)


def _mexican_hat_kernel2d(size: int, sigma_center: float, sigma_surround: float,
                           surround_gain: float, device: torch.device) -> torch.Tensor:
    c = _gaussian2d(size, sigma_center, device)
    s = _gaussian2d(size, sigma_surround, device)
    k = c - float(surround_gain) * s
    k = k - k.mean()
    return k / (k.abs().sum() + 1e-8)


# ---------------------------------------------------------------------------
# Hyperbolic operations (Poincaré ball)
# ---------------------------------------------------------------------------

class HyperbolicOps:
    def __init__(self, c: float = 1.0, eps: float = 1e-5, max_norm: float = 0.999):
        self.c = float(c); self.eps = float(eps); self.max_norm = float(max_norm)

    def norm(self, x, dim=-1, keepdim=False):
        return torch.norm(x, dim=dim, keepdim=keepdim)

    def exp_map_zero(self, v):
        v_norm = torch.clamp(self.norm(v, dim=-1, keepdim=True), min=self.eps)
        sqrt_c = math.sqrt(self.c)
        return torch.clamp(torch.tanh(sqrt_c * v_norm) * (v / (sqrt_c * v_norm)),
                           min=-self.max_norm, max=self.max_norm)

    def log_map_zero(self, x):
        x_norm = torch.clamp(self.norm(x, dim=-1, keepdim=True),
                              min=self.eps, max=self.max_norm)
        sqrt_c = math.sqrt(self.c)
        return (1.0 / sqrt_c) * torch.atanh(sqrt_c * x_norm) * (x / x_norm)

    def to_poincare(self, v):   return self.exp_map_zero(v)
    def from_poincare(self, x): return self.log_map_zero(x)

    def project_to_ball(self, x):
        norm = self.norm(x, dim=-1, keepdim=True)
        return x * torch.clamp(self.max_norm / (norm + self.eps), max=1.0)

    def mobius_add(self, x, y):
        c = self.c
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2*c*xy + c*y2)*x + (1 - c*x2)*y
        den = 1 + 2*c*xy + (c**2)*x2*y2
        return torch.clamp(num / torch.clamp(den, min=self.eps),
                           min=-self.max_norm, max=self.max_norm)

    def distance(self, x, y):
        x2 = torch.clamp((x*x).sum(dim=-1), min=0, max=1.0/self.c - self.eps)
        y2 = torch.clamp((y*y).sum(dim=-1), min=0, max=1.0/self.c - self.eps)
        d2 = ((x-y)*(x-y)).sum(dim=-1)
        den = torch.clamp((1 - self.c*x2)*(1 - self.c*y2), min=self.eps)
        arg = torch.clamp(1 + 2*self.c*d2/den, min=1.0+self.eps, max=1e10)
        return (1.0/math.sqrt(self.c)) * torch.acosh(arg)

    def distance_gradient_wrt_w(self, x, w, chunk_size=None):
        B, N, D = x.shape
        if chunk_size is not None and N > chunk_size:
            grad = torch.zeros_like(x)
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                grad[:, s:e, :] = self._distance_gradient_chunk(x[:, s:e, :], w[s:e, :])
            return grad
        return self._distance_gradient_chunk(x, w)

    def _distance_gradient_chunk(self, x, w):
        B = x.size(0)
        we = w.unsqueeze(0).expand(B, -1, -1)
        diff = we - x
        x2 = torch.clamp((x*x).sum(dim=-1, keepdim=True), min=0, max=1.0/self.c-self.eps)
        w2 = torch.clamp((we*we).sum(dim=-1, keepdim=True), min=0, max=1.0/self.c-self.eps)
        d2 = (diff*diff).sum(dim=-1, keepdim=True)
        dx = torch.clamp(1-self.c*x2, min=self.eps)
        dw = torch.clamp(1-self.c*w2, min=self.eps)
        den = dx * dw
        arg = torch.clamp(1+2*self.c*d2/den, min=1.0+self.eps, max=1e10)
        da = 1.0/torch.sqrt(torch.clamp(arg*arg-1.0, min=self.eps))
        dd = 2*diff
        ddenom = -2*self.c*we*dx
        darg = 2*self.c*(dd/den - d2*ddenom/(den*den+self.eps))
        return torch.clamp((1.0/math.sqrt(self.c))*da*darg, min=-10.0, max=10.0)


hyp_ops = HyperbolicOps(c=1.0)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class EntropyDropout(nn.Module):
    def __init__(self, base_p=0.1, entropy_scale=0.5):
        super().__init__(); self._p = float(base_p); self.entropy_scale = float(entropy_scale)

    @property
    def p(self): return self._p
    @p.setter
    def p(self, v): self._p = float(v)

    def forward(self, x):
        if not self.training or self._p <= 0: return x
        flat = x.reshape(x.size(0), -1)
        probs = F.softmax(flat, dim=1)
        entropy = -(probs*(probs+1e-8).log()).sum(dim=1)
        ne = (entropy/(math.log(flat.size(1)+1e-8)+1e-8)).clamp(0,1).mean().item()
        return F.dropout(x, p=min(1.0, self._p*(1.0+self.entropy_scale*ne)), training=True)


class ModernHopfieldPFC(nn.Module):
    def __init__(self, feature_dim, num_patterns=64, beta=1.0, temperature=2.0,
                 blend=0.005, ema_lr=0.0, unsup_update=False,
                 use_cosine_similarity=True, soft_ema_update=True,
                 normalize_memory=False, sparsity=0.9, sparse_update=True,
                 use_layernorm=False):
        super().__init__()
        self.feature_dim = int(feature_dim); self.num_patterns = int(max(2,num_patterns))
        self.beta = float(max(1e-6,beta)); self.temperature = float(max(1e-6,temperature))
        self.blend = float(max(0.,min(1.,blend))); self.ema_lr = float(max(0.,min(1.,ema_lr)))
        self.unsup_update = bool(unsup_update); self.use_cosine_similarity = bool(use_cosine_similarity)
        self.soft_ema_update = bool(soft_ema_update); self.normalize_memory = bool(normalize_memory)
        self.sparsity = float(max(0.,min(1.,sparsity))); self.sparse_update = bool(sparse_update)
        self.patterns = nn.Parameter(torch.empty(self.num_patterns, self.feature_dim))
        nn.init.xavier_uniform_(self.patterns)
        if self.normalize_memory:
            with torch.no_grad(): self.patterns.copy_(F.normalize(self.patterns,p=2,dim=1))
        self.norm = nn.LayerNorm(self.feature_dim) if bool(use_layernorm) else nn.Identity()
        self._last_consistency_loss: Optional[torch.Tensor] = None

    def _apply_feature_sparsity(self, feat):
        if self.sparsity <= 0.: return feat
        keep_k = int(math.ceil(max(0.,min(1.,1.-self.sparsity))*feat.size(1)))
        if keep_k <= 0: return torch.zeros_like(feat)
        if keep_k >= feat.size(1): return feat
        kth = torch.topk(feat.abs(),k=keep_k,dim=1,largest=True,sorted=False).values.min(dim=1,keepdim=True).values
        return feat*(feat.abs()>=kth).to(feat.dtype)

    def forward(self, x, update_memory=True):
        if x.ndim!=2 or x.size(1)!=self.feature_dim:
            raise ValueError(f"ModernHopfieldPFC expects [B,{self.feature_dim}], got {tuple(x.shape)}")
        do_ema = self.training and self.unsup_update and bool(update_memory) and self.ema_lr>0.
        pr = self.patterns.detach() if do_ema else self.patterns
        if self.use_cosine_similarity:
            logits = (F.normalize(x,p=2,dim=1)@F.normalize(pr,p=2,dim=1).t())*(self.beta/self.temperature)
        else:
            logits = (x@pr.t())*(self.beta/max(1e-8,math.sqrt(float(self.feature_dim))*self.temperature))
        attn = F.softmax(logits,dim=1); retrieved = attn@pr
        self._last_consistency_loss = F.mse_loss(retrieved,x.detach()) if do_ema else F.mse_loss(retrieved,x)
        if do_ema:
            with torch.no_grad():
                xu = self._apply_feature_sparsity(x.detach()) if self.sparse_update else x.detach()
                if self.soft_ema_update:
                    w=attn.detach(); denom=w.sum(dim=0); valid=denom>1e-8
                    if valid.any():
                        means=(w.t()@xu)/denom.clamp_min(1e-8).unsqueeze(1)
                        idx=torch.nonzero(valid,as_tuple=False).squeeze(1)
                        self.patterns[idx].mul_(1.-self.ema_lr).add_(means[idx],alpha=self.ema_lr)
                else:
                    for cid in torch.unique(attn.detach().argmax(dim=1)):
                        cf=xu[attn.detach().argmax(dim=1)==cid]
                        if cf.numel(): self.patterns[int(cid)].mul_(1.-self.ema_lr).add_(cf.mean(0),alpha=self.ema_lr)
                if self.normalize_memory: self.patterns.copy_(F.normalize(self.patterns,p=2,dim=1))
        return self.norm((1.-self.blend)*x+self.blend*retrieved)

    def get_last_consistency_loss(self): return self._last_consistency_loss


class HebbianSelfAttentionPFC(nn.Module):
    def __init__(self, feature_dim, num_patterns=64, head_dim=32, blend=0.005,
                 temperature=1.0, ema_lr=1e-3, unsup_update=True,
                 use_layernorm=False, hebbian_lr=1e-4, hebbian_decay=1e-5):
        super().__init__()
        self.feature_dim=int(feature_dim); self.num_patterns=int(max(2,num_patterns))
        hd=int(max(4,min(head_dim,feature_dim))); self.head_dim=hd
        self.blend=float(max(0.,min(1.,blend))); self.temperature=float(max(1e-6,temperature))
        self.ema_lr=float(max(0.,min(1.,ema_lr))); self.unsup_update=bool(unsup_update)
        self.hebbian_lr=float(max(0.,hebbian_lr)); self.hebbian_decay=float(max(0.,hebbian_decay))
        self.norm=nn.LayerNorm(self.feature_dim) if bool(use_layernorm) else nn.Identity()
        self._last_consistency_loss: Optional[torch.Tensor]=None
        self.W_q=nn.Parameter(torch.empty(self.feature_dim,hd))
        self.W_k=nn.Parameter(torch.empty(self.feature_dim,hd))
        self.patterns=nn.Parameter(torch.empty(self.num_patterns,self.feature_dim))
        nn.init.xavier_uniform_(self.W_q); nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.patterns)
        for p in (self.W_q,self.W_k,self.patterns): p.requires_grad_(False)

    def forward(self, x, update_memory=True):
        if x.ndim!=2 or x.size(1)!=self.feature_dim:
            raise ValueError(f"HebbianSelfAttentionPFC expects [B,{self.feature_dim}], got {tuple(x.shape)}")
        B=x.size(0); scale=1./math.sqrt(float(self.head_dim))
        q=x@self.W_q; k_mem=self.patterns@self.W_k
        attn=F.softmax((q@k_mem.t())*(scale/self.temperature),dim=1)
        retrieved=attn@self.patterns
        self._last_consistency_loss=F.mse_loss(retrieved,x.detach())
        do_l=self.training and self.unsup_update and bool(update_memory) and self.ema_lr>0.
        do_h=self.training and self.unsup_update and bool(update_memory) and self.hebbian_lr>0.
        if do_h or do_l:
            with torch.no_grad():
                qd=(x@self.W_q).detach(); kd=(self.patterns@self.W_k).detach()
                if do_h:
                    self.W_q.mul_(1.-self.hebbian_decay).add_((x.t()@qd)/float(B),alpha=self.hebbian_lr)
                    self.W_k.mul_(1.-self.hebbian_decay).add_((self.patterns.t()@kd)/float(max(1,self.num_patterns)),alpha=self.hebbian_lr)
                if do_l:
                    Pr=self.patterns.detach(); ad=attn.detach(); denom=ad.sum(dim=0); valid=denom>1e-8
                    if valid.any():
                        means=(ad.t()@x.detach())/denom.clamp_min(1e-8).unsqueeze(1)
                        idx=torch.nonzero(valid,as_tuple=False).squeeze(1)
                        Pr=Pr.clone(); Pr[idx].mul_(1.-self.ema_lr).add_(means[idx],alpha=self.ema_lr)
                        self.patterns.copy_(Pr)
        return self.norm((1.-self.blend)*x+self.blend*retrieved)

    def get_last_consistency_loss(self): return self._last_consistency_loss


def _normalized_entropy_activations(y, dim=-1):
    if y.numel()==0: return 0.0
    flat=y.reshape(-1,y.size(dim)) if y.dim()>1 else y.unsqueeze(0)
    probs=F.softmax(flat,dim=dim)
    entropy=-(probs*(probs+1e-8).log()).sum(dim=dim).mean()
    return (entropy/(math.log(flat.size(dim)+1e-8)+1e-8)).clamp(0.,1.).item()


class DoGRGB(nn.Module):
    def __init__(self, device, ksz=3, sigma_c=1.0, sigma_s=1.6, k=0.6):
        super().__init__(); assert ksz%2==1; self.pad=ksz//2
        g_c=_gaussian2d(ksz,sigma_c,device); g_s=_gaussian2d(ksz,sigma_s,device)
        dog=g_c-k*g_s; dog=dog/(dog.abs().sum()+1e-8)
        self.register_buffer("w",dog.view(1,1,ksz,ksz).repeat(3,1,1,1))

    @staticmethod
    def rgb_to_opponent(x):
        r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
        return torch.cat([(r+g+b)/3.,r-g,b-(r+g)/2.],dim=1)

    def forward(self, x):
        if x.size(1)!=3: raise ValueError(f"DoGRGB expects 3 channels, got {x.size(1)}")
        return F.conv2d(self.rgb_to_opponent(x),self.w,padding=self.pad,groups=3)


def make_gabor_kernels(freqs, oris, phs, size, device):
    ax=torch.linspace(-1,1,size,device=device)
    X,Y=torch.meshgrid(ax,ax,indexing="ij")
    Rs,Is=[],[]
    for f in freqs:
        sigma=0.56/f; G=torch.exp(-(X*X+Y*Y)/(2*sigma*sigma))
        for theta_deg in oris:
            t=theta_deg*math.pi/180.; Xp=X*math.cos(t)+Y*math.sin(t)
            for ph in phs:
                R=G*torch.cos(2*math.pi*f*Xp+ph); I=G*torch.sin(2*math.pi*f*Xp+ph)
                Rs.append(R/(R.abs().sum()+1e-8)); Is.append(I/(I.abs().sum()+1e-8))
    return torch.stack(Rs).unsqueeze(1), torch.stack(Is).unsqueeze(1)


class GaborBank(nn.Module):
    def __init__(self, in_ch, freqs, oris, phs, ksz, device):
        super().__init__()
        R,I=make_gabor_kernels(freqs,oris,phs,ksz,device)
        self.register_buffer("real_w",R.repeat(in_ch,1,1,1))
        self.register_buffer("imag_w",I.repeat(in_ch,1,1,1))
        self.groups=in_ch; self.pad=ksz//2

    def forward(self, x):
        r=F.conv2d(x,self.real_w,padding=self.pad,groups=self.groups)
        i=F.conv2d(x,self.imag_w,padding=self.pad,groups=self.groups)
        return torch.sqrt(r*r+i*i+1e-6)


# ---------------------------------------------------------------------------
# Wavelet utilities
# ---------------------------------------------------------------------------

def haar2d_one_level(x):
    B,C,H,W=x.shape; assert H%2==0 and W%2==0
    rl=(x[:,:,0::2,:]+x[:,:,1::2,:])/math.sqrt(2)
    rh=(x[:,:,0::2,:]-x[:,:,1::2,:])/math.sqrt(2)
    ll=(rl[:,:,:,0::2]+rl[:,:,:,1::2])/math.sqrt(2)
    lh=(rl[:,:,:,0::2]-rl[:,:,:,1::2])/math.sqrt(2)
    hl=(rh[:,:,:,0::2]+rh[:,:,:,1::2])/math.sqrt(2)
    hh=(rh[:,:,:,0::2]-rh[:,:,:,1::2])/math.sqrt(2)
    return torch.stack([ll,lh,hl,hh],dim=2).view(B,C*4,H//2,W//2)


class Wavelet2D(nn.Module):
    def forward(self, x):
        B,C,H,W=x.shape
        if H%2!=0 or W%2!=0: x=F.pad(x,(0,1,0,1),mode="reflect"); H,W=x.shape[2],x.shape[3]
        sub=haar2d_one_level(x)
        return F.interpolate(sub,size=(H,W),mode="bilinear",align_corners=False)


def haar1d_one_level(x):
    low=(x[...,0::2]+x[...,1::2])/math.sqrt(2)
    high=(x[...,0::2]-x[...,1::2])/math.sqrt(2)
    return low,high


def inverse_haar1d_one_level(low, high):
    x0=(low+high)/math.sqrt(2); x1=(low-high)/math.sqrt(2)
    return torch.stack([x0,x1],dim=-1).flatten(-2)


def soft_threshold(x, tau):
    return torch.sign(x)*F.relu(x.abs()-tau)


def wavelet_circular_conv_1d(a, b):
    od=a.dtype; n=a.size(-1)
    af=torch.fft.rfft(a.float(),n=n,dim=-1); bf=torch.fft.rfft(b.float(),n=n,dim=-1)
    return torch.fft.irfft(af*bf,n=n,dim=-1).real.to(od)


def soft_threshold_signed(x, tau):
    if tau<=0: return x
    return torch.sign(x)*F.relu(torch.abs(x)-tau)


def smooth_inhibition_saturation(x, scale):
    s=float(scale)
    if s<=0.: return x
    return s*torch.tanh(x/(s+1e-8))


# ---------------------------------------------------------------------------
# HRR / holographic binding
# ---------------------------------------------------------------------------

def _fft_holographic_binding(x, y, *, corr_blend, in_dim=None):
    rho=float(max(0.,min(1.,corr_blend)))
    d=int(x.shape[-1] if in_dim is None else in_dim)
    yn=y/(y.norm()+1e-8); Yf=torch.fft.rfft(yn.float(),norm="ortho")
    if x.dim()==1:
        xn=x/(x.norm()+1e-8); Xf=torch.fft.rfft(xn.float(),norm="ortho")
        conv=torch.fft.irfft(Xf*Yf,n=d,norm="ortho")
        if rho>0.: conv=(1.-rho)*conv+rho*torch.fft.irfft(Xf*torch.conj(Yf),n=d,norm="ortho")
        return torch.nan_to_num(conv.to(x.dtype))
    xn=x/(x.norm(dim=1,keepdim=True)+1e-8); Xf=torch.fft.rfft(xn.float(),dim=1,norm="ortho")
    conv=torch.fft.irfft(Xf*Yf.view(1,-1),n=d,dim=1,norm="ortho")
    if rho>0.: conv=(1.-rho)*conv+rho*torch.fft.irfft(Xf*torch.conj(Yf).view(1,-1),n=d,dim=1,norm="ortho")
    return torch.nan_to_num(conv.to(x.dtype))


def _fft_holographic_lateral_binding(yc, L, *, corr_blend=0.2):
    B,N=yc.shape; rho=float(max(0.,min(1.,corr_blend)))
    yc_t=yc.transpose(0,1).contiguous().float()
    yc_n=yc_t/yc_t.norm(dim=1,keepdim=True).clamp(min=1e-8)
    Yf=torch.fft.rfft(yc_n,n=B,dim=1,norm="ortho")
    Yi=Yf.unsqueeze(1); Yj=Yf.unsqueeze(0)
    if rho<=0.: prod=Yi*Yj
    elif rho>=1.: prod=Yi*torch.conj(Yj)
    else: prod=(1.-rho)*(Yi*Yj)+rho*(Yi*torch.conj(Yj))
    assoc=torch.fft.irfft(prod,n=B,dim=2,norm="ortho").mean(dim=2)
    assoc=torch.nan_to_num(assoc.to(L.dtype)); assoc.fill_diagonal_(0.)
    oja=L*assoc; oja.fill_diagonal_(0.); return oja


def _oja_hyperbolic_lateral_binding(yc, L, hyp_ops_ref, *, use_distance=False):
    B,N=yc.shape
    cols=yc.transpose(0,1).contiguous().float()
    emb=hyp_ops_ref.to_poincare(cols/cols.norm(dim=1,keepdim=True).clamp(min=1e-8))
    emb=torch.nan_to_num(emb.to(L.dtype))
    if use_distance:
        diff=emb.unsqueeze(0)-emb.unsqueeze(1); d2=diff.pow(2).sum(dim=-1)
        nu=emb.norm(dim=-1).clamp(max=1-1e-5)
        den=((1-nu.unsqueeze(0).pow(2))*(1-nu.unsqueeze(1).pow(2))).clamp(min=1e-8)
        assoc=-torch.acosh((1.+2.*d2/den).clamp(min=1.+1e-6))
    else:
        assoc=emb@emb.transpose(0,1)
    assoc=torch.nan_to_num(assoc); assoc.fill_diagonal_(0.)
    oja=L*assoc; oja.fill_diagonal_(0.); return oja


def _oja_wavelet_lateral_binding(yc, L, haar1d_fn, *, n_levels=1):
    B,N=yc.shape; wt=yc.transpose(0,1).contiguous().float()
    if wt.size(1)<2: return torch.zeros_like(L)
    low=wt
    for _ in range(n_levels):
        if low.size(1)<2: break
        if low.size(1)%2==1: low=F.pad(low,(0,1),mode="replicate")
        low,_=haar1d_fn(low)
    wf=torch.nan_to_num(F.normalize(low,p=2,dim=1,eps=1e-8).to(L.dtype))
    assoc=wf@wf.transpose(0,1); assoc.fill_diagonal_(0.)
    oja=L*assoc; oja.fill_diagonal_(0.); return oja


# ---------------------------------------------------------------------------
# RF mask
# ---------------------------------------------------------------------------

def build_rf_gaussian_sparse_mask(in_channels, rf_size, *, sigma_frac, keep_fraction,
                                   sparse_quantile, device, dtype):
    rf=int(rf_size); ic=int(in_channels)
    if rf<1 or ic<1: return torch.ones(max(1,ic*rf*rf),device=device,dtype=dtype)
    sigma_frac=float(max(1e-4,sigma_frac))
    yy,xx=torch.meshgrid(torch.arange(rf,device=device,dtype=torch.float32),
                          torch.arange(rf,device=device,dtype=torch.float32),indexing="ij")
    cy,cx=(rf-1)*.5,(rf-1)*.5; sigma=sigma_frac*float(rf)
    scores=torch.exp(-0.5*((yy-cy)**2+(xx-cx)**2)/(sigma*sigma+1e-8))
    scores=scores/scores.max().clamp(min=1e-8); scores=scores.reshape(-1)
    K=int(scores.numel()); kf=float(max(0.,min(1.,keep_fraction)))
    if kf<1.-1e-9:
        keep_n=max(1,min(K,int(math.ceil(kf*float(K)))))
        _,topi=torch.topk(scores,keep_n,largest=True)
        g=torch.zeros(K,device=device,dtype=torch.float32)
        g.scatter_(0,topi,torch.ones(keep_n,device=device,dtype=torch.float32))
    else:
        sq=float(max(0.,min(1.,sparse_quantile)))
        g=(scores>=torch.quantile(scores,sq)).float() if sq>0. else torch.ones(K,device=device,dtype=torch.float32)
    return g.unsqueeze(1).expand(-1,ic).reshape(-1).to(dtype=dtype)


# ---------------------------------------------------------------------------
# UnifiedCoeffs dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnifiedCoeffs:
    eta: float = 5e-9
    alpha: float = 0.1
    holo_update_freq: int = 20
    holo_fast_lr: float = 0.1
    holo_fast_decay: float = 0.1
    holo_corr_blend: float = 0.2
    holo_fast_norm_cap: float = 12.0
    distance_gradient_lr: float = 0.01
    beta_hyp: float = 0.01
    gamma_wavelet: float = 0.01
    wavelet_threshold: float = 0.02
    use_wavelet_denoise: bool = True
    use_wavelet_binding: bool = True
    use_wavelet_input: bool = True
    lambda_a: float = 0.001
    lambda_c: float = 0.001
    lambda_r: float = 0.001
    lambda_F: float = 0.001
    lambda_d: float = 0.1
    use_entropy_dropout: bool = False
    entropy_dropout_scale: float = 0.5
    use_entropy_plasticity_decay: bool = False
    entropy_decay_scale: float = 0.5
    inhibition_mode: str = "kernel"
    use_soft_inhibition: bool = False
    inhibition_no_threshold: bool = True
    inhibition_smooth_scale: float = 0.0
    averaging_inhibition_size: int = 7
    kernel_inhibition_size: int = 7
    kernel_inhibition_sigma: float = 1.5
    mixed_inhibition_w_global: float = 1./3.
    mixed_inhibition_w_kernel: float = 1./3.
    mixed_inhibition_w_averaging: float = 1./3.
    inhibition_dropout: float = 0.0
    use_divisive_inhibition_norm: bool = True
    divisive_mode: str = "both"
    divisive_w_global: float = 0.5
    divisive_w_local: float = 0.5
    divisive_local_size: int = 9
    divisive_local_sigma: float = 2.0
    divisive_alpha: float = 0.85
    divisive_beta: float = 0.36
    adaptive_inhibition: bool = False
    target_active_frac: float = 0.01
    inhibition_adapt_lr: float = 0.01
    inhibition_min: float = 0.0
    inhibition_max: float = 1.0
    inhibition_learning_sparsity: float = 0.0
    use_homeostatic_threshold: bool = False
    homeostatic_threshold_lr: float = 0.02
    homeostatic_threshold_theta_init: float = 0.0
    homeostatic_threshold_min: float = -2.0
    homeostatic_threshold_max: float = 2.0
    use_oscillatory_inhibition: bool = False
    phase_period: int = 10
    phase_gate_sharpness: float = 1.0
    cascade_skip_connections: bool = True
    rf_connectivity_gaussian: bool = False
    rf_gaussian_sigma_frac: float = 0.28
    rf_connectivity_keep_frac: float = 0.6
    rf_connectivity_sparse_quantile: float = 0.0
    use_ei_neurons: bool = False
    ei_mutual_inhibition: bool = False
    ei_separate_lateral: bool = True
    ei_l_ei_init: float = -0.01
    ei_l_ie_init: float = 0.01
    lateral_w_hebb: float = 1.0
    lateral_w_anti: float = 0.0
    lateral_w_cov: float = 0.0
    lateral_w_holo: float = 0.05
    lateral_w_hyp: float = 0.05
    lateral_w_wave: float = 0.05
    lateral_w_oja: float = 0.0
    lateral_w_oja_holo: float = 0.0
    holo_lateral_corr_blend: float = 1.0
    lateral_w_oja_hyp: float = 0.0
    lateral_oja_hyp_use_distance: bool = False
    lateral_w_oja_wave: float = 0.0
    lateral_oja_wave_levels: int = 1
    som_enabled: bool = False
    kernel_inhibition_sigma_start: float = 2.5
    kernel_inhibition_sigma_end: float = 1.0
    som_lr_lateral_warmup_fraction: float = 0.2
    lr_lateral_schedule_end_scale: float = 0.5
    use_inhibition_softmax: bool = False
    inhibition_softmax_temp_start: float = 1.5
    inhibition_softmax_temp_end: float = 0.45
    som_wta_scale_start: float = 0.88
    som_wta_scale_end: float = 1.0
    mexican_center_sigma_start: float = 0.65
    mexican_center_sigma_end: float = 0.45
    mexican_surround_sigma_start: float = 2.4
    mexican_surround_sigma_end: float = 1.75
    mexican_surround_gain_start: float = 1.0
    mexican_surround_gain_end: float = 1.15
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
    unsup_mix_w_oja_holo: float = 0.0
    unsup_mix_w_oja_hyp: float = 0.0
    unsup_mix_w_oja_wave: float = 0.0
    use_structural_plasticity: bool = True
    structural_update_freq: int = 1200
    structural_prune_threshold: float = 1e-4
    structural_prune_max_frac: float = 0.002
    structural_grow_threshold: float = 0.02
    structural_grow_max_frac: float = 0.001
    structural_grow_init_scale: float = 0.02
    pc_per_neuron_plasticity: bool = True
    pc_per_neuron_layer_weight: float = 0.34
    pc_per_neuron_pfc_weight: float = 0.33
    pc_per_neuron_trace_weight: float = 0.33
    pc_per_neuron_gain_min: float = 0.5
    pc_per_neuron_gain_max: float = 1.5
    
    
    
    
    
    
    
    
    
    
    
    
    

    
# ---------------------------------------------------------------------------
# TopographicUnifiedLayer
# ---------------------------------------------------------------------------

class TopographicUnifiedLayer(nn.Module):
    def __init__(self, in_channels, rf_size, spatial_size=90, name="L?",
                 coeffs=None, beta_trace=0.9, use_hebbian=True, use_antihebbian=True,
                 use_holographic=True, use_consistency=True, use_recursive=True,
                 use_free_energy=True, use_active_inference=True, recursive_iters=5,
                 inhibition_strength=0.1, inhibition_decay=0.01, competition_threshold=0.1,
                 lateral_update_freq=10, lr_lateral=1e-3, w_clip=1.0,
                 active_inference_steps=3, active_inference_lr=0.1, wta_sparsity=0.9,
                 holo_update_freq=None, use_hyperbolic_binding=True,
                 use_distance_gradient=False, gradient_chunk_size=512,
                 use_wavelet_binding=True, use_wavelet_denoise=True, device=None):
        super().__init__()
        if coeffs is None: coeffs = UnifiedCoeffs()
        self.name = name; self.in_channels = in_channels; self.rf_size = rf_size
        self.spatial_size = spatial_size; self.N = spatial_size*spatial_size
        self.in_dim = in_channels*rf_size*rf_size; self.pad = rf_size//2
        self.coeffs = coeffs; self.beta_trace = beta_trace
        self.use_hebbian = use_hebbian; self.use_antihebbian = use_antihebbian
        self.use_holographic = use_holographic; self.use_consistency = use_consistency
        self.use_recursive = use_recursive; self.use_free_energy = use_free_energy
        self.use_active_inference = use_active_inference
        self.use_hyperbolic_binding = bool(use_hyperbolic_binding)
        self.use_distance_gradient = bool(use_distance_gradient)
        self.gradient_chunk_size = gradient_chunk_size
        self.use_wavelet_binding = bool(use_wavelet_binding)
        self.use_wavelet_denoise = bool(use_wavelet_denoise)
        self.recursive_iters = recursive_iters
        self.inhibition_strength = float(inhibition_strength)
        self.current_inhibition_strength = float(inhibition_strength)
        self.inhibition_decay = float(inhibition_decay)
        self.competition_threshold = float(competition_threshold)
        self.lateral_update_freq = int(lateral_update_freq)
        self.lr_lateral = lr_lateral; self.lr_lateral_base = float(lr_lateral)
        self.inhibition_mode = str(getattr(coeffs,"inhibition_mode","kernel")).lower()
        self.use_ei_neurons = bool(getattr(coeffs,"use_ei_neurons",False))
        self.ei_mutual_inhibition = bool(getattr(coeffs,"ei_mutual_inhibition",False))
        self.ei_separate_lateral = bool(getattr(coeffs,"ei_separate_lateral",True))
        self.ei_l_ei_init = float(getattr(coeffs,"ei_l_ei_init",-0.1))
        self.ei_l_ie_init = float(getattr(coeffs,"ei_l_ie_init",0.1))
        self.use_soft_inhibition = bool(getattr(coeffs,"use_soft_inhibition",False))
        self.inhibition_no_threshold = bool(getattr(coeffs,"inhibition_no_threshold",False))
        self.inhibition_smooth_scale = float(max(0.,getattr(coeffs,"inhibition_smooth_scale",0.)))
        self.use_divisive_inhibition_norm = bool(getattr(coeffs,"use_divisive_inhibition_norm",False))
        _dm = str(getattr(coeffs,"divisive_mode","both")).lower().strip()
        if _dm=="all": _dm="both"
        self.divisive_mode = _dm
        wg=float(getattr(coeffs,"divisive_w_global",0.5)); wl=float(getattr(coeffs,"divisive_w_local",0.5))
        ws=max(1e-8,abs(wg)+abs(wl)); self.divisive_w_global=abs(wg)/ws; self.divisive_w_local=abs(wl)/ws
        self.divisive_alpha=float(getattr(coeffs,"divisive_alpha",0.85))
        self.divisive_beta=float(getattr(coeffs,"divisive_beta",0.36))
        self.inhibition_dropout=float(max(0.,min(1.,getattr(coeffs,"inhibition_dropout",0.))))
        self.adaptive_inhibition=bool(getattr(coeffs,"adaptive_inhibition",False))
        self.target_active_frac=float(getattr(coeffs,"target_active_frac",0.01))
        self.inhibition_adapt_lr=float(getattr(coeffs,"inhibition_adapt_lr",0.01))
        self.inhibition_min=float(getattr(coeffs,"inhibition_min",0.))
        self.inhibition_max=float(getattr(coeffs,"inhibition_max",1.))
        self.inhibition_learning_sparsity=float(max(0.,min(1.,getattr(coeffs,"inhibition_learning_sparsity",0.))))
        self.use_homeostatic_threshold=bool(getattr(coeffs,"use_homeostatic_threshold",False))
        self.homeostatic_threshold_lr=float(getattr(coeffs,"homeostatic_threshold_lr",0.02))
        self.homeostatic_threshold_theta_init=float(getattr(coeffs,"homeostatic_threshold_theta_init",0.))
        self.homeostatic_threshold_min=float(getattr(coeffs,"homeostatic_threshold_min",-2.))
        self.homeostatic_threshold_max=float(getattr(coeffs,"homeostatic_threshold_max",2.))
        self.lateral_w_hebb=float(getattr(coeffs,"lateral_w_hebb",1.)); self.lateral_w_anti=float(getattr(coeffs,"lateral_w_anti",0.))
        self.lateral_w_cov=float(getattr(coeffs,"lateral_w_cov",0.)); self.lateral_w_holo=float(getattr(coeffs,"lateral_w_holo",0.05))
        self.lateral_w_hyp=float(getattr(coeffs,"lateral_w_hyp",0.05)); self.lateral_w_wave=float(getattr(coeffs,"lateral_w_wave",0.05))
        self.lateral_w_oja=float(getattr(coeffs,"lateral_w_oja",0.))
        wg=float(getattr(coeffs,"mixed_inhibition_w_global",1./3.))
        wk=float(getattr(coeffs,"mixed_inhibition_w_kernel",1./3.))
        wa=float(getattr(coeffs,"mixed_inhibition_w_averaging",1./3.))
        ws=max(1e-8,wg+wk+wa)
        self.mixed_inh_w_global=wg/ws; self.mixed_inh_w_kernel=wk/ws; self.mixed_inh_w_averaging=wa/ws
        self.w_clip=w_clip; self.active_inference_steps=active_inference_steps
        self.active_inference_lr=active_inference_lr; self.wta_sparsity=float(wta_sparsity)
        self._wta_sparsity_base=float(wta_sparsity)
        self.holo_update_freq=int(getattr(coeffs,"holo_update_freq",20) if holo_update_freq is None else holo_update_freq)
        dev=device or torch.device("cpu")
        _rfg=bool(getattr(coeffs,"rf_connectivity_gaussian",False))
        if _rfg:
            _rm=build_rf_gaussian_sparse_mask(self.in_channels,self.rf_size,
                sigma_frac=float(getattr(coeffs,"rf_gaussian_sigma_frac",0.28)),
                keep_fraction=float(getattr(coeffs,"rf_connectivity_keep_frac",0.6)),
                sparse_quantile=float(getattr(coeffs,"rf_connectivity_sparse_quantile",0.)),
                device=dev,dtype=torch.float32)
            self.register_buffer("rf_connectivity_mask",_rm)
        else:
            self.register_buffer("rf_connectivity_mask",torch.ones(self.in_dim,device=dev,dtype=torch.float32))
        self.W=nn.Parameter(torch.empty(self.N,self.in_dim),requires_grad=False)
        nn.init.xavier_uniform_(self.W)
        self.b=nn.Parameter(torch.zeros(self.N),requires_grad=False)
        self.register_buffer("winner_theta",torch.full((self.N,),self.homeostatic_threshold_theta_init,device=dev,dtype=torch.float32))
        self._alloc_dense_lateral=self.inhibition_mode in ("global","mixed")
        if self._alloc_dense_lateral:
            self.L=nn.Parameter(torch.full((self.N,self.N),-0.1,device=dev,dtype=torch.float32),requires_grad=False)
            with torch.no_grad(): self.L.fill_diagonal_(0.)
            self.L_ei=nn.Parameter(torch.full((self.N,self.N),self.ei_l_ei_init,device=dev,dtype=torch.float32),requires_grad=False)
            self.L_ie=nn.Parameter(torch.full((self.N,self.N),self.ei_l_ie_init,device=dev,dtype=torch.float32),requires_grad=False)
            with torch.no_grad(): self.L_ei.fill_diagonal_(0.); self.L_ie.fill_diagonal_(0.)
        else:
            z=torch.zeros(1,1,device=dev,dtype=torch.float32)
            self.L=nn.Parameter(z,requires_grad=False)
            self.L_ei=nn.Parameter(torch.zeros(1,1,device=dev,dtype=torch.float32),requires_grad=False)
            self.L_ie=nn.Parameter(torch.zeros(1,1,device=dev,dtype=torch.float32),requires_grad=False)
        self.register_buffer("update_counter",torch.tensor(0,dtype=torch.long))
        self._cache_yE_lateral: Optional[torch.Tensor]=None
        self._cache_yI_lateral: Optional[torch.Tensor]=None
        self.som_enabled=bool(getattr(coeffs,"som_enabled",False))
        self.use_inhibition_softmax=bool(getattr(coeffs,"use_inhibition_softmax",False))
        self._softmax_competition_temp=float(getattr(coeffs,"inhibition_softmax_temp_start",1.5))
        self.use_oscillatory_inhibition=bool(getattr(coeffs,"use_oscillatory_inhibition",False))
        self.phase_period=int(max(1,getattr(coeffs,"phase_period",10)))
        self.phase_gate_sharpness=float(getattr(coeffs,"phase_gate_sharpness",1.))
        if self.use_oscillatory_inhibition:
            self.register_buffer("phase_pref",torch.rand(self.N,device=dev,dtype=torch.float32)*(2.*math.pi))
            self.register_buffer("phase_t",torch.zeros((),device=dev,dtype=torch.long))
        if self.inhibition_mode in ("kernel","mixed"):
            ksz=int(getattr(coeffs,"kernel_inhibition_size",7))
            if ksz%2==0: ksz+=1
            sigma=float(getattr(coeffs,"kernel_inhibition_sigma_start",coeffs.kernel_inhibition_sigma)
                        if self.som_enabled else getattr(coeffs,"kernel_inhibition_sigma",1.5))
            self.register_buffer("inhibition_kernel",_gaussian2d(ksz,sigma,dev).view(1,1,ksz,ksz))
        if self.inhibition_mode=="mexican":
            ksz=int(getattr(coeffs,"kernel_inhibition_size",7))
            if ksz%2==0: ksz+=1
            self.register_buffer("mexican_kernel",_mexican_hat_kernel2d(ksz,
                float(getattr(coeffs,"mexican_center_sigma_start",0.65)),
                float(getattr(coeffs,"mexican_surround_sigma_start",2.4)),
                float(getattr(coeffs,"mexican_surround_gain_start",1.)),dev).view(1,1,ksz,ksz))
        if self.inhibition_mode in ("averaging","mixed"):
            ksz=int(getattr(coeffs,"averaging_inhibition_size",7))
            if ksz%2==0: ksz+=1
            self.register_buffer("averaging_inhibition_kernel",
                torch.ones((1,1,ksz,ksz),device=dev,dtype=torch.float32)/float(ksz*ksz))
        if self.use_divisive_inhibition_norm and self.divisive_mode in ("local","both"):
            dsz=int(getattr(coeffs,"divisive_local_size",7))
            if dsz%2==0: dsz+=1
            self.register_buffer("divisive_local_kernel",
                _gaussian2d(dsz,float(getattr(coeffs,"divisive_local_sigma",1.5)),dev).view(1,1,dsz,dsz))
        P=torch.randn(self.in_dim,self.N,device=dev)/math.sqrt(self.N)
        self.register_buffer("P_holo",P)
        self.register_buffer("M_holo_fast",torch.zeros(self.N,self.in_dim,device=dev))
        self.holo_fast_decay=float(getattr(coeffs,"holo_fast_decay",0.1))
        self.holo_fast_lr=float(getattr(coeffs,"holo_fast_lr",0.2))
        self.G=nn.Parameter(torch.empty(self.in_dim,self.N),requires_grad=False)
        nn.init.xavier_uniform_(self.G)
        self.register_buffer("running_mean",torch.zeros(self.N))
        self.register_buffer("running_var",torch.ones(self.N))
        self.momentum=0.1
        self.register_buffer("trace",torch.zeros(self.N))
        self.register_buffer("prev_y",torch.zeros(1))
        self.register_buffer("last_fe_signal_per_neuron",torch.zeros(self.N))
        self.layer_norm=nn.LayerNorm(self.N)
        self._last_y0=None; self._last_yT=None

    @torch.no_grad()
    def apply_som_schedule(self, progress, coeffs):
        if not getattr(coeffs,"som_enabled",False): return
        p=float(max(0.,min(1.,progress)))
        wf=float(max(0.,min(1.,getattr(coeffs,"som_lr_lateral_warmup_fraction",0.2))))
        end_scale=float(getattr(coeffs,"lr_lateral_schedule_end_scale",0.5))
        if p<wf: lr_scale=1.
        else: lr_scale=1.+(end_scale-1.)*(p-wf)/(1.-wf+1e-8)
        self.lr_lateral=float(self.lr_lateral_base*lr_scale)
        t0=float(getattr(coeffs,"inhibition_softmax_temp_start",1.5))
        t1=float(getattr(coeffs,"inhibition_softmax_temp_end",0.45))
        self._softmax_competition_temp=t0+(t1-t0)*p
        ws0=float(getattr(coeffs,"som_wta_scale_start",0.88))
        ws1=float(getattr(coeffs,"som_wta_scale_end",1.))
        self.wta_sparsity=float(min(0.99,max(0.,self._wta_sparsity_base*(ws0+(ws1-ws0)*p))))
        dev=self.W.device
        ksz=int(getattr(coeffs,"kernel_inhibition_size",7))
        if ksz%2==0: ksz+=1
        sig0=float(getattr(coeffs,"kernel_inhibition_sigma_start",2.5))
        sig1=float(getattr(coeffs,"kernel_inhibition_sigma_end",1.))
        sigma=sig0+(sig1-sig0)*p
        if self.inhibition_mode in ("kernel","mixed") and hasattr(self,"inhibition_kernel"):
            self.inhibition_kernel.copy_(_gaussian2d(ksz,sigma,dev).view(1,1,ksz,ksz))
        if self.inhibition_mode=="mexican" and hasattr(self,"mexican_kernel"):
            sc=float(getattr(coeffs,"mexican_center_sigma_start",0.65))+(float(getattr(coeffs,"mexican_center_sigma_end",0.45))-float(getattr(coeffs,"mexican_center_sigma_start",0.65)))*p
            ss=float(getattr(coeffs,"mexican_surround_sigma_start",2.4))+(float(getattr(coeffs,"mexican_surround_sigma_end",1.75))-float(getattr(coeffs,"mexican_surround_sigma_start",2.4)))*p
            sg=float(getattr(coeffs,"mexican_surround_gain_start",1.))+(float(getattr(coeffs,"mexican_surround_gain_end",1.15))-float(getattr(coeffs,"mexican_surround_gain_start",1.)))*p
            self.mexican_kernel.copy_(_mexican_hat_kernel2d(ksz,sc,ss,sg,dev).view(1,1,ksz,ksz))

    def _extract_patches(self, x_map):
        x_pad=F.pad(x_map,[self.pad]*4,mode="reflect")
        return F.unfold(x_pad,kernel_size=self.rf_size).transpose(1,2)

    def _mask_rf_patches(self, patches):
        m=self.rf_connectivity_mask.to(device=patches.device,dtype=patches.dtype)
        return patches*m.view(1,1,-1)

    def _apply_inhibition_dropout(self, inhibition):
        if self.training and self.inhibition_dropout>0.:
            kp=1.-self.inhibition_dropout
            inhibition=inhibition*(torch.rand_like(inhibition)<kp).to(inhibition.dtype)/max(kp,1e-8)
        return inhibition

    def _ei_mutual_terms(self, yE, yI):
        if self.ei_separate_lateral:
            return yI@self.L_ei.t(), yE@self.L_ie.t()
        return yI@self.L.t(), yE@self.L.t()

    def _apply_divisive_inhibition_norm(self, y, B, H, W):
        if not self.use_divisive_inhibition_norm: return y
        abs_y=torch.abs(y); k=getattr(self,"divisive_local_kernel",None)
        gm=torch.mean(abs_y,dim=1,keepdim=True)
        if self.divisive_mode=="both" and k is not None:
            pad=k.size(-1)//2
            pooled=F.conv2d(abs_y.view(B,1,H,W),k,padding=pad).view(B,self.N)
            denom=self.divisive_alpha+self.divisive_beta*(self.divisive_w_global*gm+self.divisive_w_local*pooled)
        elif self.divisive_mode=="local" and k is not None:
            pad=k.size(-1)//2
            denom=self.divisive_alpha+self.divisive_beta*F.conv2d(abs_y.view(B,1,H,W),k,padding=pad).view(B,self.N)
        else:
            denom=self.divisive_alpha+self.divisive_beta*gm
        return y/(denom+1e-8)

    def _shape_inhibition_signal(self, z):
        if self.inhibition_no_threshold:
            return smooth_inhibition_saturation(z,self.inhibition_smooth_scale) if self.inhibition_smooth_scale>0. else z
        if self.use_soft_inhibition: return soft_threshold_signed(z,self.competition_threshold)
        return torch.where(torch.abs(z)<self.competition_threshold,torch.zeros_like(z),z)

    def _apply_inhibition(self, y, B, H, W, yI=None):
        """Apply one round of competitive inhibition. Returns updated y (and yI if EI)."""
        if self.current_inhibition_strength<=0: return y, yI
        inh_s=self.current_inhibition_strength
        if yI is not None:
            inh_E,inh_I=self._ei_mutual_terms(y,yI)
            y=y+inh_s*self._apply_inhibition_dropout(self._shape_inhibition_signal(inh_E))
            yI=yI+inh_s*self._apply_inhibition_dropout(self._shape_inhibition_signal(inh_I))
            y=y/(y.norm(dim=1,keepdim=True)+1e-8)
            yI=yI/(yI.norm(dim=1,keepdim=True)+1e-8)
        if self.inhibition_mode=="kernel":
            inh=F.conv2d(y.view(B,1,H,W),self.inhibition_kernel,padding=self.inhibition_kernel.size(-1)//2).view(B,self.N)
            y=y-inh_s*self._apply_inhibition_dropout(self._shape_inhibition_signal(inh))
        elif self.inhibition_mode=="mexican":
            mk=self.mexican_kernel
            inh=F.conv2d(y.view(B,1,H,W),mk,padding=mk.size(-1)//2).view(B,self.N)
            y=y-inh_s*self._apply_inhibition_dropout(self._shape_inhibition_signal(inh))
        elif self.inhibition_mode=="averaging":
            inh=F.conv2d(y.view(B,1,H,W),self.averaging_inhibition_kernel,padding=self.averaging_inhibition_kernel.size(-1)//2).view(B,self.N)
            y=y-inh_s*self._apply_inhibition_dropout(self._shape_inhibition_signal(inh))
        elif self.inhibition_mode=="mixed":
            ym=y.view(B,1,H,W)
            ik=F.conv2d(ym,self.inhibition_kernel,padding=self.inhibition_kernel.size(-1)//2).view(B,self.N)
            ia=F.conv2d(ym,self.averaging_inhibition_kernel,padding=self.averaging_inhibition_kernel.size(-1)//2).view(B,self.N)
            ig=y@self.L.t()
            y=y+inh_s*(self.mixed_inh_w_global*self._apply_inhibition_dropout(self._shape_inhibition_signal(ig))
                       -self.mixed_inh_w_kernel*self._apply_inhibition_dropout(self._shape_inhibition_signal(ik))
                       -self.mixed_inh_w_averaging*self._apply_inhibition_dropout(self._shape_inhibition_signal(ia)))
        elif self.inhibition_mode=="predictive":
            with torch.no_grad():
                pb=self.G.detach().t()@(self.G.detach()@y.mean(dim=0))
                pd=pb.abs().unsqueeze(0).expand(B,self.N)
            y=y-inh_s*self._apply_inhibition_dropout(self._shape_inhibition_signal(pd))
        else:  # global
            if yI is None:
                inh=y@self.L.t()
                y=y+inh_s*self._apply_inhibition_dropout(self._shape_inhibition_signal(inh))
        return y, yI

    def forward(self, x_map):
        B,C,H,W=x_map.shape
        if H!=self.spatial_size or W!=self.spatial_size:
            raise ValueError(f"{self.name} expects {self.spatial_size}x{self.spatial_size}, got {H}x{W}")
        if C!=self.in_channels:
            raise ValueError(f"{self.name} expects {self.in_channels} channels, got {C}")
        if self.use_oscillatory_inhibition:
            with torch.no_grad(): self.phase_t+=1
        patches=self._mask_rf_patches(self._extract_patches(x_map))
        y=torch.einsum("bni,ni->bn",patches,self.W)+self.b.unsqueeze(0)
        y=F.normalize(self.layer_norm(y),p=2,dim=1)
        yE=yI=None
        if self.use_ei_neurons and self.ei_mutual_inhibition and self.inhibition_mode in ("global","mixed"):
            yE=F.relu(y)/(F.relu(y).norm(dim=1,keepdim=True)+1e-8)
            yI=F.relu(-y)/(F.relu(-y).norm(dim=1,keepdim=True)+1e-8)
            y=yE
        y,yI=self._apply_inhibition(y,B,H,W,yI)
        if self.use_inhibition_softmax and self._softmax_competition_temp>0:
            t=self._softmax_competition_temp
            scale=y.norm(dim=1,keepdim=True).clamp(min=1e-8)
            y=F.softmax(y/t,dim=1)*scale
        if self.use_divisive_inhibition_norm:
            y=self._apply_divisive_inhibition_norm(y,B,H,W)
        y0=y
        if self.use_recursive and self.recursive_iters>0:
            yt=y
            for _ in range(self.recursive_iters):
                yt,yI=self._apply_inhibition(yt,B,H,W,yI)
                if self.use_divisive_inhibition_norm and self.current_inhibition_strength>0:
                    yt=self._apply_divisive_inhibition_norm(yt,B,H,W)
            y=yt
        self._last_y0=y0.detach(); self._last_yT=y.detach()
        if self.use_ei_neurons and self.ei_mutual_inhibition and self.inhibition_mode in ("global","mixed") and yI is not None:
            self._cache_yE_lateral=y.detach(); self._cache_yI_lateral=yI.detach()
        else:
            self._cache_yE_lateral=None; self._cache_yI_lateral=None
        if self.training and self.adaptive_inhibition:
            with torch.no_grad():
                delta=self.inhibition_adapt_lr*((y>0).float().mean().item()-self.target_active_frac)
                self.current_inhibition_strength=float(max(self.inhibition_min,min(self.inhibition_max,self.current_inhibition_strength+delta)))
        if self.wta_sparsity>0.:
            keep_k=max(1,min(self.N,int(math.ceil(max(0.,min(1.,1.-self.wta_sparsity))*self.N))))
            phase_gate=None
            if self.use_oscillatory_inhibition:
                with torch.no_grad():
                    t=float(self.phase_t.item()); omega=(2.*math.pi)/float(self.phase_period)
                    pg=0.5*(1.+torch.cos(omega*t+self.phase_pref))
                    if self.phase_gate_sharpness!=1.: pg=pg.pow(self.phase_gate_sharpness)
                    phase_gate=pg.unsqueeze(0)
            if self.use_homeostatic_threshold:
                scores=y-self.winner_theta.unsqueeze(0)
                if phase_gate is not None: scores=scores*phase_gate.to(dtype=scores.dtype)
                kth=torch.topk(scores,k=keep_k,dim=1,largest=True,sorted=False).values.min(dim=1,keepdim=True).values
                mask=scores>=kth
                if self.training:
                    ri=mask.float().mean(dim=0)
                    self.winner_theta.add_(self.homeostatic_threshold_lr*(ri-keep_k/float(self.N)))
                    self.winner_theta.clamp_(self.homeostatic_threshold_min,self.homeostatic_threshold_max)
                y=y*mask.to(y.dtype)
            else:
                if phase_gate is not None: y=y*phase_gate.to(dtype=y.dtype)
                mag=y.abs()
                kth=torch.topk(mag,k=keep_k,dim=1,largest=True,sorted=False).values.min(dim=1,keepdim=True).values
                y=y*(mag>=kth).to(y.dtype)
        return y.view(B,1,H,W), y, patches

    def update_from_patches(self, patches, y_flat, variance_scale=1.0,
                             inhibition_scale=1.0, layer_spatial_gain=None, pfc_spatial_gain=None):
        B=y_flat.size(0); inh_scale=float(max(0.,inhibition_scale))
        y_plas=y_flat
        if self._cache_yE_lateral is not None and self._cache_yE_lateral.shape==y_flat.shape:
            y_plas=self._cache_yE_lateral.to(device=y_flat.device,dtype=y_flat.dtype)
        with torch.no_grad():
            self.update_counter+=1
            self.last_fe_signal_per_neuron.zero_()
            blended=0.9*float(self.current_inhibition_strength)+0.1*float(self.inhibition_strength)*inh_scale
            self.current_inhibition_strength=float(max(self.inhibition_min,min(self.inhibition_max,blended)))
        patches=self._mask_rf_patches(patches)
        y_mean=y_plas.mean(dim=0); patch_sum=patches.sum(dim=0); patch_mean=patch_sum/float(B)
        use_pc=bool(getattr(self.coeffs,"pc_per_neuron_plasticity",True))
        trace_prev=None
        with torch.no_grad():
            if use_pc and self.beta_trace>0. and self.trace.numel()==self.N:
                trace_prev=self.trace.detach().clone()
        with torch.no_grad():
            if self.beta_trace>0.: self.trace.mul_(1-self.beta_trace).add_(self.beta_trace*y_mean)
        if self.use_active_inference and self.use_free_energy and self.active_inference_steps>0:
            with torch.enable_grad():
                z=y_mean.detach().clone().requires_grad_(True)
                xt=patch_mean.mean(dim=0).detach()
                for _ in range(self.active_inference_steps):
                    xh=self.G.detach()@z
                    Fe=(xt-xh).pow(2).mean()+1e-3*z.pow(2).mean()
                    (grad,)=torch.autograd.grad(Fe,z,retain_graph=False,create_graph=False)
                    z=(z-self.active_inference_lr*grad).detach().requires_grad_(True)
                y_eff=z.detach()
        else:
            y_eff=y_mean.detach() if self.beta_trace<=0. else self.trace.detach()
        term_updates: Dict[str,torch.Tensor]={}
        def _add(n,v):
            term_updates[n]=term_updates[n]+v if n in term_updates else v
        patch_for_hebb=patch_sum
        if self.use_wavelet_denoise and getattr(self.coeffs,"use_wavelet_denoise",True):
            tau=getattr(self.coeffs,"wavelet_threshold",0.02)
            if tau>0 and self.in_dim>=2:
                D=self.in_dim+(self.in_dim%2); pw=D-self.in_dim
                pm=F.pad(patch_mean,(0,pw),mode="reflect")
                low,high=haar1d_one_level(pm)
                rec=inverse_haar1d_one_level(soft_threshold(low,tau),soft_threshold(high,tau))
                patch_for_hebb=rec[...,:self.in_dim]*float(B)
        if self.use_hebbian: _add("hebb",y_eff.unsqueeze(1)*patch_for_hebb)
        if int(self.update_counter.item())%max(1,self.holo_update_freq)==0:
            if self.use_holographic:
                ye=(self.P_holo@y_eff).contiguous()
                rho=float(getattr(self.coeffs,"holo_corr_blend",0.2))
                ht=y_eff.unsqueeze(1)*_fft_holographic_binding(patch_mean,ye,corr_blend=rho,in_dim=self.in_dim)
                _add("holo",self.coeffs.alpha*ht)
                with torch.no_grad():
                    self.M_holo_fast.mul_(1.-self.holo_fast_decay).add_(self.holo_fast_lr*ht)
                    cap=float(getattr(self.coeffs,"holo_fast_norm_cap",12.))
                    if cap>0.:
                        n=self.M_holo_fast.norm()
                        if torch.isfinite(n) and float(n)>cap: self.M_holo_fast.mul_(cap/(n+1e-8))
                _add("holo",self.coeffs.alpha*self.M_holo_fast)
            if self.use_hyperbolic_binding:
                bh=float(getattr(self.coeffs,"beta_hyp",0.01))
                xhp=hyp_ops.project_to_ball(patch_mean)
                yhp=hyp_ops.project_to_ball(torch.abs(y_eff).unsqueeze(1)*patch_mean)
                bt=torch.nan_to_num(hyp_ops.from_poincare(hyp_ops.mobius_add(xhp,yhp)))
                _add("hyp",bh*(y_eff.unsqueeze(1)*bt))
        if self.use_distance_gradient and getattr(self.coeffs,"distance_gradient_lr",0.)>0:
            xhp=hyp_ops.to_poincare(patch_mean); whp=hyp_ops.to_poincare(self.W)
            grad=torch.nan_to_num(hyp_ops.distance_gradient_wrt_w(xhp.unsqueeze(0),whp,chunk_size=self.gradient_chunk_size).mean(dim=0))
            _add("dist",-self.coeffs.distance_gradient_lr*grad)
        if self.use_wavelet_binding and getattr(self.coeffs,"use_wavelet_binding",True):
            gw=getattr(self.coeffs,"gamma_wavelet",0.01)
            if gw>0 and self.in_dim>=2:
                D=self.in_dim+(self.in_dim%2); pw=D-self.in_dim
                pm=F.pad(patch_mean,(0,pw),mode="reflect")
                ye=(self.P_holo@y_eff).contiguous(); ye=ye/(ye.norm()+1e-8)
                ye=F.pad(ye.unsqueeze(0),(0,pw),mode="reflect").squeeze(0)
                lp,hp=haar1d_one_level(pm); ly,hy=haar1d_one_level(ye)
                binding=torch.nan_to_num(inverse_haar1d_one_level(wavelet_circular_conv_1d(lp,ly),wavelet_circular_conv_1d(hp,hy))[...,:self.in_dim])
                _add("wave",gw*(y_eff.unsqueeze(1)*binding))
        if self.use_antihebbian:
            with torch.no_grad():
                wn=F.normalize(self.W,p=2,dim=1); sim=wn@wn.t(); sim.fill_diagonal_(0.)
                _add("anti",-self.coeffs.lambda_a*(sim@self.W))
        if self.use_consistency and self.prev_y.numel()==self.N:
            _add("cons",-self.coeffs.lambda_c*(y_mean-self.prev_y).unsqueeze(1)*patch_sum)
        if self.use_recursive and self._last_y0 is not None and self._last_yT is not None:
            dy=(self._last_yT.mean(dim=0)-self._last_y0.mean(dim=0))
            _add("rec",self.coeffs.lambda_r*dy.unsqueeze(1)*patch_sum)
        if self.use_free_energy:
            xt=patch_mean.mean(dim=0); xh=self.G@y_mean; err=xt-xh
            with torch.no_grad():
                self.G.add_(0.001*torch.ger(err,y_mean))
                self.last_fe_signal_per_neuron.copy_((self.G.t()@err).abs())
            _add("free",self.coeffs.lambda_F*(y_mean.unsqueeze(1)*err.unsqueeze(0)))
        ld=self.coeffs.lambda_d
        if getattr(self.coeffs,"use_entropy_plasticity_decay",False) and ld>0:
            ld=ld*(1.+getattr(self.coeffs,"entropy_decay_scale",0.5)*_normalized_entropy_activations(y_plas))
        _add("decay",-ld*self.W)
        dW=torch.zeros_like(self.W)
        if term_updates:
            mode=str(getattr(self.coeffs,"unsup_mix_mode","adaptive")).lower()
            nt=bool(getattr(self.coeffs,"unsup_mix_normalize_terms",True))
            temp=float(max(1e-6,getattr(self.coeffs,"unsup_mix_temperature",1.)))
            ra=float(max(1e-6,getattr(self.coeffs,"unsup_mix_random_alpha",20.)))
            bwm={"hebb":float(getattr(self.coeffs,"unsup_mix_w_hebb",0.30)),
                 "holo":float(getattr(self.coeffs,"unsup_mix_w_holo",0.10)),
                 "hyp":float(getattr(self.coeffs,"unsup_mix_w_hyp",0.10)),
                 "wave":float(getattr(self.coeffs,"unsup_mix_w_wave",0.15)),
                 "anti":float(getattr(self.coeffs,"unsup_mix_w_anti",0.10)),
                 "cons":float(getattr(self.coeffs,"unsup_mix_w_cons",0.08)),
                 "rec":float(getattr(self.coeffs,"unsup_mix_w_rec",0.08)),
                 "free":float(getattr(self.coeffs,"unsup_mix_w_free",0.06)),
                 "decay":float(getattr(self.coeffs,"unsup_mix_w_decay",0.03)),
                 "dist":float(getattr(self.coeffs,"unsup_mix_w_dist",0.00))}
            names=list(term_updates.keys()); eps=1e-8
            norms=torch.tensor([float(term_updates[n].norm().item()) for n in names],device=self.W.device,dtype=self.W.dtype)
            if nt:
                for n in names: term_updates[n]=term_updates[n]/(term_updates[n].norm()+eps)
            base=torch.tensor([max(0.,bwm.get(n,0.)) for n in names],device=self.W.device,dtype=self.W.dtype)
            base=base/base.sum().clamp(min=eps) if base.sum()>0 else torch.ones_like(base)/max(1,len(names))
            if mode=="fixed": mix_w=base
            elif mode=="random":
                alpha=torch.clamp(ra*base,min=1e-3)
                mix_w=torch.distributions.Dirichlet(alpha.float()).sample().to(self.W.dtype)
            else:
                inv_n=1./(norms+eps); inv_n=inv_n/(inv_n.sum()+eps)
                mix_w=torch.softmax(torch.log(base+eps)+temp*torch.log(inv_n+eps),dim=0)
            for i,n in enumerate(names): dW.add_(mix_w[i]*term_updates[n])
        with torch.no_grad():
            dW=torch.nan_to_num(dW); dn=dW.norm()
            if torch.isfinite(dn) and dn>1.: dW.mul_(1./(dn+1e-8))
            Nloc=self.N; dev,dt=self.W.device,self.W.dtype
            if isinstance(variance_scale,torch.Tensor):
                vs_t=variance_scale.to(device=dev,dtype=dt).flatten()
                if vs_t.numel()==Nloc: vs_vec=vs_t.clamp(min=0.)
                elif vs_t.numel()==1: vs_vec=torch.full((Nloc,),float(vs_t.item()),device=dev,dtype=dt).clamp(min=0.)
                else: vs_vec=torch.full((Nloc,),float(vs_t.mean().item()),device=dev,dtype=dt).clamp(min=0.)
            else:
                vs_vec=None; vs_f=float(max(0.,variance_scale))
            if use_pc:
                gmn=float(getattr(self.coeffs,"pc_per_neuron_gain_min",0.5))
                gmx=float(getattr(self.coeffs,"pc_per_neuron_gain_max",1.5))
                lg=(y_mean-trace_prev).abs() if trace_prev is not None and trace_prev.numel()==Nloc else (y_mean-y_mean.mean()).abs()
                lg=torch.clamp(lg/(lg.mean()+1e-8),gmn,gmx)
                wl=float(getattr(self.coeffs,"pc_per_neuron_layer_weight",0.34))
                wp=float(getattr(self.coeffs,"pc_per_neuron_pfc_weight",0.33))
                wt=float(getattr(self.coeffs,"pc_per_neuron_trace_weight",0.33))
                dg,dtg=lg.device,lg.dtype
                acc=torch.zeros(Nloc,device=dg,dtype=dtg); wsum=0.
                if wt>0.: acc=acc+wt*lg; wsum+=wt
                if layer_spatial_gain is not None and layer_spatial_gain.numel()==Nloc and wl>0.:
                    acc=acc+wl*layer_spatial_gain.to(device=dg,dtype=dtg); wsum+=wl
                if pfc_spatial_gain is not None and pfc_spatial_gain.numel()==Nloc and wp>0.:
                    acc=acc+wp*pfc_spatial_gain.to(device=dg,dtype=dtg); wsum+=wp
                eff=acc/float(wsum) if wsum>0. else lg
                eff=eff/(eff.mean()+1e-8)
                row_scale=vs_vec*eff if vs_vec is not None else vs_f*eff
            else:
                row_scale=vs_vec*torch.ones(Nloc,device=dev,dtype=dt) if vs_vec is not None else vs_f*torch.ones(Nloc,device=dev,dtype=dt)
            self.W.add_(self.coeffs.eta*row_scale.unsqueeze(1)*dW)
            self.W.clamp_(-self.w_clip,self.w_clip)
            if bool(getattr(self.coeffs,"use_structural_plasticity",False)) and int(self.update_counter.item())%max(1,int(getattr(self.coeffs,"structural_update_freq",1200)))==0:
                ts=self.W.numel()
                pm=int(max(0,round(float(getattr(self.coeffs,"structural_prune_max_frac",0.002))*ts)))
                gm=int(max(0,round(float(getattr(self.coeffs,"structural_grow_max_frac",0.001))*ts)))
                pt=float(max(0.,getattr(self.coeffs,"structural_prune_threshold",1e-4)))
                gt=float(max(0.,getattr(self.coeffs,"structural_grow_threshold",0.02)))
                gi=float(max(0.,getattr(self.coeffs,"structural_grow_init_scale",0.02)))
                if pm>0:
                    ci=torch.nonzero((torch.abs(self.W)<pt)&(self.W!=0),as_tuple=False).squeeze(1) if self.W.dim()>1 else torch.nonzero((torch.abs(self.W.view(-1))<pt)&(self.W.view(-1)!=0),as_tuple=False).squeeze(1)
                    wf=self.W.view(-1)
                    if ci.numel()>0:
                        ci2=torch.nonzero((torch.abs(wf)<pt)&(wf!=0),as_tuple=False).squeeze(1)
                        if ci2.numel()>pm:
                            si=torch.topk(torch.abs(wf)[ci2],k=pm,largest=False).indices
                            wf[ci2[si]]=0.
                        else: wf[ci2]=0.
                if gm>0 and gi>0.:
                    hd=torch.abs(y_eff.unsqueeze(1)*patch_mean)
                    wf=self.W.view(-1); hdf=hd.view(-1)
                    ci=torch.nonzero((wf==0)&(hdf>gt),as_tuple=False).squeeze(1)
                    if ci.numel()>0:
                        if ci.numel()>gm:
                            si=torch.topk(hdf[ci],k=gm,largest=True).indices; ci=ci[si]
                        wf[ci]=gi*torch.sign(hdf[ci])
                self.W.clamp_(-self.w_clip,self.w_clip)
        if self.inhibition_mode in ("global","mixed") and self.use_antihebbian and int(self.update_counter.item())%self.lateral_update_freq==0:
            with torch.no_grad():
                corr=y_plas.t()@y_plas; corr.fill_diagonal_(0.)
                ymb=y_plas.mean(dim=0,keepdim=True); yc=y_plas-ymb
                cov=yc.t()@yc; cov.fill_diagonal_(0.)
                ycn=F.normalize(yc,p=2,dim=0,eps=1e-8); ha=ycn.t()@ycn; ha.fill_diagonal_(0.)
                hya=torch.zeros_like(corr)
                if self.lateral_w_hyp>0.:
                    yr=F.normalize(yc.float(),p=2,dim=1,eps=1e-8)
                    he=hyp_ops.to_poincare(yr); he=torch.nan_to_num(he.to(dtype=corr.dtype))
                    hya=he.transpose(0,1)@he; hya.fill_diagonal_(0.)
                wa=torch.zeros_like(corr)
                if self.lateral_w_wave>0.:
                    wt=yc.transpose(0,1).contiguous().float()
                    if wt.size(1)>=2:
                        if wt.size(1)%2==1: wt=F.pad(wt,(0,1),mode="replicate")
                        lw,_=haar1d_one_level(wt)
                        wf=torch.nan_to_num(F.normalize(lw,p=2,dim=1,eps=1e-8).to(dtype=corr.dtype))
                        wa=wf@wf.transpose(0,1); wa.fill_diagonal_(0.)
                y2=(y_plas.pow(2)).mean(dim=0)
                oja_stab=self.L*(y2.unsqueeze(0)+y2.unsqueeze(1))*0.5; oja_stab.fill_diagonal_(0.)
                lwoh=float(getattr(self.coeffs,"lateral_w_oja_holo",0.))
                oha=_fft_holographic_lateral_binding(yc,self.L,corr_blend=float(getattr(self.coeffs,"holo_lateral_corr_blend",1.))) if lwoh>0. and yc.size(0)>=2 else torch.zeros_like(corr)
                lwoyp=float(getattr(self.coeffs,"lateral_w_oja_hyp",0.))
                ohya=_oja_hyperbolic_lateral_binding(yc,self.L,hyp_ops,use_distance=bool(getattr(self.coeffs,"lateral_oja_hyp_use_distance",False))) if lwoyp>0. and yc.size(0)>=2 else torch.zeros_like(corr)
                lwoyw=float(getattr(self.coeffs,"lateral_w_oja_wave",0.))
                owya=_oja_wavelet_lateral_binding(yc,self.L,haar1d_one_level,n_levels=int(getattr(self.coeffs,"lateral_oja_wave_levels",1))) if lwoyw>0. and yc.size(0)>=2 else torch.zeros_like(corr)
                drive=(self.lateral_w_hebb*corr+self.lateral_w_anti*(-corr)+self.lateral_w_cov*cov
                       +self.lateral_w_holo*ha+self.lateral_w_hyp*hya+self.lateral_w_wave*wa
                       +lwoh*oha+lwoyp*ohya+lwoyw*owya
                       -self.inhibition_decay*self.L-self.lateral_w_oja*oja_stab)
                lupd=(self.lr_lateral*inh_scale)*drive
                lupd=torch.where(torch.abs(lupd)<self.competition_threshold,torch.zeros_like(lupd),lupd)
                if self.inhibition_learning_sparsity>0.:
                    kf=max(0.,min(1.,1.-self.inhibition_learning_sparsity))
                    if kf<=0.: lupd.zero_()
                    else:
                        kk=max(1,min(lupd.numel(),int(math.ceil(kf*lupd.numel()))))
                        if kk<lupd.numel():
                            kth=torch.topk(torch.abs(lupd).view(-1),k=kk,largest=True,sorted=False).values.min()
                            lupd=torch.where(torch.abs(lupd)>=kth,lupd,torch.zeros_like(lupd))
                self.L.add_(lupd); self.L.fill_diagonal_(0.); self.L.clamp_(-1.,0.)
        with torch.no_grad(): self.prev_y=y_mean.detach().clone()
        
        
        
        
        
        
        
        
 
 
 
 # ---------------------------------------------------------------------------
# VisNetUnified
# ---------------------------------------------------------------------------

class VisNetUnified(nn.Module):
    def __init__(self, device="cpu", num_classes=10, spatial_size=80,
                 auto_resize_input=True, coeffs=None,
                 use_hebbian=True, use_antihebbian=True, use_holographic=True,
                 use_consistency=True, use_recursive=True, use_free_energy=True,
                 use_active_inference=True, dropout=0.1, inhibition_decay=0.01,
                 use_pfc_hopfield=False, pfc_hopfield_patterns=64,
                 pfc_hopfield_beta=1.0, pfc_hopfield_temperature=2.0,
                 pfc_hopfield_blend=0.001, pfc_hopfield_ema_lr=1e-3,
                 pfc_hopfield_unsup_update=False, pfc_hopfield_cosine=True,
                 pfc_hopfield_soft_ema=True, pfc_hopfield_normalize_memory=False,
                 pfc_hopfield_sparsity=0.95, pfc_hopfield_sparse_update=True,
                 pfc_hopfield_layernorm=False, pfc_mode="hopfield",
                 pfc_hebbian_lr=1e-4, pfc_hebbian_decay=1e-5, pfc_sa_head_dim=0,
                 pfc_topdown_attention=True, pfc_topdown_strength=0.25,
                 pfc_topdown_min_scale=0.6, pfc_topdown_max_scale=1.4,
                 pfc_topdown_unsup_lr=3e-3, pfc_topdown_decay=1e-4,
                 pfc_topdown_per_neuron=True, pfc_topdown_neuron_use_bias=False,
                 pfc_topdown_shared_fe_blend=0.35,
                 pfc_inhibition_feedback_unsup_lr=3e-3, pfc_inhibition_feedback_decay=1e-4,
                 pfc_topdown_iters=2, pfc_predictive_feedback=True,
                 pfc_predictive_strength=0.15, pfc_predictive_min_scale=0.6,
                 pfc_predictive_max_scale=1.4, pfc_predictive_unsup_lr=3e-3,
                 pfc_predictive_decay=1e-4, pfc_l1_lambda=0., pfc_l1_prox_step=1.,
                 local_l1_lambda=0., local_l1_prox_step=1., local_l1_warmup_steps=0,
                 local_l1_apply_every=1, wta_l1=1e-2, wta_l234=1e-2,
                 rf_l1=7, rf_l2=7, rf_l3=7, rf_l4=7, recursive_iters=5,
                 use_dorsal_stream=True, pfc_spatial_readout_gate=True,
                 pfc_spatial_gate_strength=0.65, pfc_spatial_gate_floor=0.2,
                 pfc_recurrent_feedback_steps=2, pfc_recurrent_feedback_strength=0.35,
                 use_pfc_dense_feedback=True, pfc_dense_feedback_strength=0.12,
                 use_pfc_deep_feedback=True, pfc_deep_feedback_rank=32,
                 pfc_deep_fb_strength_l2=0.06, pfc_deep_fb_strength_l3=0.06,
                 pfc_deep_fb_strength_l4=0.08, pfc_deep_fb_strength_mt=0.05,
                 pfc_deep_fb_strength_pp=0.05, use_symmetry_gate_prior=True,
                 symmetry_gate_alpha=0.5, symmetry_prior_unsup_lr=0.,
                 it_pp_cross_gate=True, it_pp_cross_pp_to_te=0.35,
                 it_pp_cross_te_to_pp=0.35, it_pp_cross_iters=1,
                 pfc_pre_hopfield_fusion="all", pfc_pre_blend_w_te=0.5,
                 pfc_pre_blend_w_pp=0.5, pfc_post_readout_fusion="all",
                 use_pfc_fusion_gate_unsup=True, pfc_fusion_gate_unsup_lr=3e-3,
                 pfc_fusion_gate_decay=1e-4, pfc_fusion_lms_chunk_rows=256,
                 use_pfc_pc_layer_output_mask=True, pfc_pc_layer_mask_lr=0.001,
                 pfc_pc_layer_mask_decay=1e-4, use_neuron_glia=True,
                 glia_state_dim=8, glia_ema=0.995, glia_neuron_strength=0.12,
                 glia_gate_min=0.75, glia_gate_max=1.25):
        super().__init__()
        if coeffs is None: coeffs = UnifiedCoeffs()
        self.device = torch.device(device)
        self.num_classes = num_classes; self.use_free_energy = bool(use_free_energy)
        self.spatial_size = int(spatial_size); self.auto_resize_input = bool(auto_resize_input)
        self.coeffs = coeffs; self.dropout_p = float(dropout)
        self.rf_l1=int(rf_l1); self.rf_l2=int(rf_l2); self.rf_l3=int(rf_l3); self.rf_l4=int(rf_l4)
        self.recursive_iters=int(max(0,recursive_iters))
        self.use_dorsal_stream=bool(use_dorsal_stream)
        self.use_pfc_spatial_readout_gate=bool(pfc_spatial_readout_gate)
        self.pfc_spatial_gate_strength=float(max(0.,min(1.,pfc_spatial_gate_strength)))
        self.pfc_spatial_gate_floor=float(max(0.,min(1.,pfc_spatial_gate_floor)))
        self.pfc_recurrent_feedback_steps=int(max(1,min(2,pfc_recurrent_feedback_steps)))
        self.pfc_recurrent_feedback_strength=float(max(0.,pfc_recurrent_feedback_strength))
        self.use_pfc_dense_feedback=bool(use_pfc_dense_feedback)
        self.pfc_dense_feedback_strength=float(max(0.,pfc_dense_feedback_strength))
        self.use_pfc_deep_feedback=bool(use_pfc_deep_feedback)
        self.pfc_deep_feedback_rank=int(max(4,min(256,pfc_deep_feedback_rank)))
        self.pfc_deep_fb_strength_l2=float(max(0.,pfc_deep_fb_strength_l2))
        self.pfc_deep_fb_strength_l3=float(max(0.,pfc_deep_fb_strength_l3))
        self.pfc_deep_fb_strength_l4=float(max(0.,pfc_deep_fb_strength_l4))
        self.pfc_deep_fb_strength_mt=float(max(0.,pfc_deep_fb_strength_mt))
        self.pfc_deep_fb_strength_pp=float(max(0.,pfc_deep_fb_strength_pp))
        self.use_symmetry_gate_prior=bool(use_symmetry_gate_prior)
        self.symmetry_gate_alpha=float(max(0.,symmetry_gate_alpha))
        self.symmetry_prior_unsup_lr=float(max(0.,symmetry_prior_unsup_lr))
        self.it_pp_cross_gate=bool(it_pp_cross_gate)
        self.it_pp_cross_pp_to_te=float(max(0.,it_pp_cross_pp_to_te))
        self.it_pp_cross_te_to_pp=float(max(0.,it_pp_cross_te_to_pp))
        self.it_pp_cross_iters=int(max(1,it_pp_cross_iters))
        _pf=str(pfc_pre_hopfield_fusion).strip().lower()
        if _pf not in ("pp_only","blend","gate","all"): _pf="all"
        self.pfc_pre_hopfield_fusion=_pf
        self.pfc_pre_blend_w_te=float(max(0.,pfc_pre_blend_w_te))
        self.pfc_pre_blend_w_pp=float(max(0.,pfc_pre_blend_w_pp))
        _pr=str(pfc_post_readout_fusion).strip().lower()
        if _pr not in ("concat","gate","all"): _pr="all"
        self.pfc_post_readout_fusion=_pr
        self.use_pfc_fusion_gate_unsup=bool(use_pfc_fusion_gate_unsup)
        self.pfc_fusion_gate_unsup_lr=float(max(0.,pfc_fusion_gate_unsup_lr))
        self.pfc_fusion_gate_decay=float(max(0.,pfc_fusion_gate_decay))
        self.pfc_fusion_lms_chunk_rows=int(max(8,min(4096,pfc_fusion_lms_chunk_rows)))
        if getattr(coeffs,"use_entropy_dropout",False):
            self.dropout=EntropyDropout(base_p=self.dropout_p,entropy_scale=getattr(coeffs,"entropy_dropout_scale",0.5))
        else:
            self.dropout=nn.Dropout(p=self.dropout_p)
        freqs=tuple(i/3.5 for i in range(1,4))
        oris=tuple(i*25 for i in range(4))
        phs=(0,math.pi)
        self.gabor=GaborBank(1,freqs,oris,phs,7,self.device)
        self.num_freqs=len(freqs); self.gabor_ch_per_freq=1*len(oris)*len(phs)
        self.use_wavelet_input=bool(getattr(coeffs,"use_wavelet_input",True))
        if self.use_wavelet_input:
            self.wavelet=Wavelet2D(); self.wavelet_ch_per_band=3
            self.channels_per_freq=self.gabor_ch_per_freq+self.wavelet_ch_per_band
        else:
            self.wavelet=None; self.wavelet_ch_per_band=0
            self.channels_per_freq=self.gabor_ch_per_freq
        self.cascade_skip_connections=bool(getattr(coeffs,"cascade_skip_connections",True))
        _l2_in=2 if self.cascade_skip_connections else 1
        _l3_in=2 if self.cascade_skip_connections else 1
        _l4_in=3 if self.cascade_skip_connections else 1

        def _make_layers(in_ch, rf, name_prefix, beta, wta, ai):
            return nn.ModuleList([
                TopographicUnifiedLayer(
                    in_channels=in_ch, rf_size=rf, spatial_size=self.spatial_size,
                    name=f"{name_prefix}{i}", coeffs=coeffs, beta_trace=beta,
                    recursive_iters=self.recursive_iters, inhibition_strength=0.1,
                    inhibition_decay=float(inhibition_decay), lr_lateral=coeffs.eta,
                    use_hebbian=use_hebbian, use_antihebbian=use_antihebbian,
                    use_holographic=use_holographic, use_consistency=use_consistency,
                    use_recursive=use_recursive, use_free_energy=use_free_energy,
                    use_active_inference=ai, wta_sparsity=float(wta),
                    use_wavelet_binding=getattr(coeffs,"use_wavelet_binding",True),
                    use_wavelet_denoise=getattr(coeffs,"use_wavelet_denoise",True),
                    device=self.device)
                for i in range(self.num_freqs)])

        self.l1_freq_layers=_make_layers(self.channels_per_freq,rf_l1,"L1_freq",0.0,wta_l1,use_active_inference)
        self.l2_freq_layers=_make_layers(_l2_in,rf_l2,"L2_freq",0.9,wta_l234,False)
        self.l3_freq_layers=_make_layers(_l3_in,rf_l3,"L3_freq",0.9,wta_l234,False)
        self.l4_freq_layers=_make_layers(_l4_in,rf_l4,"L4_freq",0.9,wta_l234,False)
        if self.use_dorsal_stream:
            self.mt_freq_layers=_make_layers(_l3_in,rf_l3,"MT_freq",0.9,wta_l234,False)
            self.pp_freq_layers=_make_layers(_l4_in,rf_l4,"PP_freq",0.9,wta_l234,False)
        else:
            self.mt_freq_layers=nn.ModuleList(); self.pp_freq_layers=nn.ModuleList()

        _l4s=self.num_freqs*self.spatial_size*self.spatial_size
        total_l4=_l4s*(2 if self.use_dorsal_stream else 1)
        self.use_pfc_hopfield=bool(use_pfc_hopfield)
        _pm=str(pfc_mode).strip().lower()
        self.pfc_mode=_pm if self.use_pfc_hopfield else "hopfield"
        self.use_pfc_topdown_attention=bool(pfc_topdown_attention)
        self.pfc_topdown_strength=float(max(0.,pfc_topdown_strength))
        self.pfc_topdown_min_scale=float(max(0.,pfc_topdown_min_scale))
        self.pfc_topdown_max_scale=float(max(self.pfc_topdown_min_scale,pfc_topdown_max_scale))
        self.pfc_topdown_unsup_lr=float(max(0.,pfc_topdown_unsup_lr))
        self.pfc_topdown_decay=float(max(0.,pfc_topdown_decay))
        self.pfc_inhibition_feedback_unsup_lr=float(max(0.,pfc_inhibition_feedback_unsup_lr))
        self.pfc_inhibition_feedback_decay=float(max(0.,pfc_inhibition_feedback_decay))
        self.pfc_topdown_iters=int(max(1,pfc_topdown_iters))
        self.use_pfc_predictive_feedback=bool(pfc_predictive_feedback)
        self.pfc_predictive_strength=float(max(0.,pfc_predictive_strength))
        self.pfc_predictive_min_scale=float(max(0.,pfc_predictive_min_scale))
        self.pfc_predictive_max_scale=float(max(self.pfc_predictive_min_scale,pfc_predictive_max_scale))
        self.pfc_predictive_unsup_lr=float(max(0.,pfc_predictive_unsup_lr))
        self.pfc_predictive_decay=float(max(0.,pfc_predictive_decay))
        self.pfc_l1_lambda=float(max(0.,pfc_l1_lambda)); self.pfc_l1_prox_step=float(max(0.,pfc_l1_prox_step))
        self.local_l1_lambda=float(max(0.,local_l1_lambda)); self.local_l1_prox_step=float(max(0.,local_l1_prox_step))
        self.local_l1_warmup_steps=int(max(0,local_l1_warmup_steps))
        self.local_l1_apply_every=int(max(1,local_l1_apply_every)); self.local_l1_update_counter=0
        self.use_neuron_glia=bool(use_neuron_glia); self.glia_state_dim=int(max(4,glia_state_dim))
        self.glia_ema=float(min(0.9999,max(0.,glia_ema))); self.glia_neuron_strength=float(max(0.,glia_neuron_strength))
        self.glia_gate_min=float(min(glia_gate_min,glia_gate_max)); self.glia_gate_max=float(max(glia_gate_min,glia_gate_max))
        self.pfc_topdown_W=nn.Parameter(torch.eye(4,dtype=torch.float32),requires_grad=False)
        self.pfc_topdown_b=nn.Parameter(torch.zeros(4,dtype=torch.float32),requires_grad=False)
        self.pfc_inhibition_feedback_W=nn.Parameter(torch.eye(4,dtype=torch.float32),requires_grad=False)
        self.pfc_inhibition_feedback_b=nn.Parameter(torch.zeros(4,dtype=torch.float32),requires_grad=False)
        self.pfc_predictive_feedback_W=nn.Parameter(torch.eye(4,dtype=torch.float32),requires_grad=False)
        self.pfc_predictive_feedback_b=nn.Parameter(torch.zeros(4,dtype=torch.float32),requires_grad=False)
        _df=int(self.num_freqs*self.spatial_size*self.spatial_size)
        self.use_pfc_topdown_per_neuron=bool(pfc_topdown_per_neuron)
        self.pfc_topdown_neuron_use_bias=bool(pfc_topdown_neuron_use_bias)
        self.pfc_topdown_shared_fe_blend=float(max(0.,min(1.,pfc_topdown_shared_fe_blend)))
        self.pfc_topdown_neuron_w=nn.Parameter(torch.empty(4,_df,device=self.device,dtype=torch.float32),requires_grad=False)
        nn.init.normal_(self.pfc_topdown_neuron_w,std=0.02)
        if self.pfc_topdown_neuron_use_bias:
            self.register_parameter("pfc_topdown_neuron_b",nn.Parameter(torch.zeros(4,_df,device=self.device,dtype=torch.float32),requires_grad=False))
        else:
            self.register_parameter("pfc_topdown_neuron_b",None)
        _phd=_l4s
        if self.use_pfc_hopfield:
            if self.pfc_mode=="hebbian_sa":
                _hd=int(pfc_sa_head_dim) if pfc_sa_head_dim>0 else min(64,max(8,_phd//8))
                self.pfc_hopfield=HebbianSelfAttentionPFC(feature_dim=_phd,num_patterns=int(pfc_hopfield_patterns),
                    head_dim=_hd,blend=float(pfc_hopfield_blend),temperature=float(pfc_hopfield_temperature),
                    ema_lr=float(pfc_hopfield_ema_lr),unsup_update=True,use_layernorm=bool(pfc_hopfield_layernorm),
                    hebbian_lr=float(pfc_hebbian_lr),hebbian_decay=float(pfc_hebbian_decay))
            else:
                self.pfc_hopfield=ModernHopfieldPFC(feature_dim=_phd,num_patterns=int(pfc_hopfield_patterns),
                    beta=float(pfc_hopfield_beta),temperature=float(pfc_hopfield_temperature),
                    blend=float(pfc_hopfield_blend),ema_lr=float(pfc_hopfield_ema_lr),
                    unsup_update=bool(pfc_hopfield_unsup_update),use_cosine_similarity=bool(pfc_hopfield_cosine),
                    soft_ema_update=bool(pfc_hopfield_soft_ema),normalize_memory=bool(pfc_hopfield_normalize_memory),
                    sparsity=float(pfc_hopfield_sparsity),sparse_update=bool(pfc_hopfield_sparse_update),
                    use_layernorm=bool(pfc_hopfield_layernorm))
        else:
            self.pfc_hopfield=None
        self.use_pfc_pc_layer_output_mask=bool(use_pfc_pc_layer_output_mask)
        self.pfc_pc_layer_mask_lr=float(max(0.,pfc_pc_layer_mask_lr))
        self.pfc_pc_layer_mask_decay=float(max(0.,pfc_pc_layer_mask_decay))
        self.pfc_pc_layer_mask_W=nn.Parameter(torch.empty(_phd,4,device=self.device,dtype=torch.float32),requires_grad=False)
        self.pfc_pc_layer_mask_b=nn.Parameter(torch.zeros(4,device=self.device,dtype=torch.float32),requires_grad=False)
        nn.init.xavier_uniform_(self.pfc_pc_layer_mask_W)
        self.clf=nn.Linear(total_l4,num_classes)
        nn.init.xavier_uniform_(self.clf.weight); nn.init.zeros_(self.clf.bias)
        _gd=self.glia_state_dim; dev=self.device; f32=torch.float32
        self.register_buffer("_glia_trace",torch.zeros(_gd,device=dev,dtype=f32))
        self.register_buffer("_glia_proj_in",torch.randn(_gd,4,device=dev,dtype=f32).mul_(0.07))
        self.register_buffer("_glia_proj_out",torch.randn(4,_gd,device=dev,dtype=f32).mul_(0.07))
        _dgk=15; self._dorsal_gate_pad=_dgk//2
        self.register_buffer("_gauss_k_small",_gaussian_kernel_2d(_dgk,1.,dev,f32))
        self.register_buffer("_gauss_k_large",_gaussian_kernel_2d(_dgk,4.,dev,f32))
        self.register_buffer("symmetry_prior_scale_ema",torch.tensor(1.,device=dev,dtype=f32))
        _D=int(self.num_freqs*self.spatial_size*self.spatial_size); _r=int(self.pfc_deep_feedback_rank)
        self.register_parameter("pfc_fb_Wd",nn.Parameter(torch.empty(_D,_r,device=dev,dtype=f32)))
        self.register_parameter("pfc_fb_bd",nn.Parameter(torch.zeros(_r,device=dev,dtype=f32)))
        for _n in ("l2","l3","l4","mt","pp"):
            self.register_parameter(f"pfc_fb_U_{_n}",nn.Parameter(torch.empty(_r,_D,device=dev,dtype=f32)))
        nn.init.xavier_uniform_(self.pfc_fb_Wd)
        for _n in ("l2","l3","l4","mt","pp"): nn.init.xavier_uniform_(getattr(self,f"pfc_fb_U_{_n}"))
        for _p in [self.pfc_fb_Wd,self.pfc_fb_bd]+[getattr(self,f"pfc_fb_U_{_n}") for _n in ("l2","l3","l4","mt","pp")]:
            _p.requires_grad_(False)
        self.register_parameter("pfc_pre_gate_W",nn.Parameter(torch.empty(_D,2*_D,device=dev,dtype=f32)))
        self.register_parameter("pfc_pre_gate_b",nn.Parameter(torch.zeros(_D,device=dev,dtype=f32)))
        self.register_parameter("pfc_post_gate_W",nn.Parameter(torch.empty(_D,2*_D,device=dev,dtype=f32)))
        self.register_parameter("pfc_post_gate_b",nn.Parameter(torch.zeros(_D,device=dev,dtype=f32)))
        nn.init.xavier_uniform_(self.pfc_pre_gate_W); nn.init.xavier_uniform_(self.pfc_post_gate_W)
        for _p in (self.pfc_pre_gate_W,self.pfc_pre_gate_b,self.pfc_post_gate_W,self.pfc_post_gate_b):
            _p.requires_grad_(False)
        self.to(self.device)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def _single_stream_flat_dim(self): return int(self.num_freqs*self.spatial_size*self.spatial_size)

    @property
    def l4_feature_dim(self): return self._single_stream_flat_dim*(2 if self.use_dorsal_stream else 1)

    # ── glia ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _neuron_glia_per_layer_gates(self, layer_targets, update_trace=True):
        if not self.use_neuron_glia: return [1.,1.,1.,1.]
        dev=self._glia_trace.device
        s=torch.tensor([float(layer_targets["l1"]),float(layer_targets["l2"]),
                         float(layer_targets["l3"]),float(layer_targets["l4"])],device=dev,dtype=torch.float32)
        inc=torch.log1p(s.clamp_min(1e-8))@self._glia_proj_in.T
        if self.training and bool(update_trace):
            self._glia_trace.mul_(self.glia_ema).add_(inc,alpha=1.-self.glia_ema)
        g=(1.+self.glia_neuron_strength*torch.tanh(self._glia_trace@self._glia_proj_out.T)).clamp(self.glia_gate_min,self.glia_gate_max)
        return [float(g[i].item()) for i in range(4)]

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _rgb_to_freq_inputs(self, x):
        x_opp=DoGRGB.rgb_to_opponent(x); L=x_opp[:,0:1]
        g=self.gabor(L)
        if self.auto_resize_input and (g.size(2)!=self.spatial_size or g.size(3)!=self.spatial_size):
            g=F.interpolate(g,size=(self.spatial_size,self.spatial_size),mode="bilinear",align_corners=False)
        gabor_bands=torch.chunk(g,self.num_freqs,dim=1)
        if not self.use_wavelet_input: return list(gabor_bands),L
        w=self.wavelet(x_opp)
        if w.size(2)!=self.spatial_size or w.size(3)!=self.spatial_size:
            w=F.interpolate(w,size=(self.spatial_size,self.spatial_size),mode="bilinear",align_corners=False)
        wb=torch.chunk(w,4,dim=1)
        return [torch.cat([gb,wb[i%4]],dim=1) for i,gb in enumerate(gabor_bands)],L

    # ── dorsal gate ───────────────────────────────────────────────────────────

    def _vertical_symmetry_map(self, L):
        B,C,H,W=L.shape
        if W<4: return torch.ones(B,1,H,W,device=L.device,dtype=L.dtype)
        m=int(min(W//2,W-W//2))
        if m<2: return torch.ones(B,1,H,W,device=L.device,dtype=L.dtype)
        lft=L[:,:,:,:m]; rgt=L[:,:,:,W-m:].flip(-1)
        ln=(lft-lft.mean(dim=(2,3),keepdim=True))/(lft.std(dim=(2,3),keepdim=True)+1e-6)
        rn=(rgt-rgt.mean(dim=(2,3),keepdim=True))/(rgt.std(dim=(2,3),keepdim=True)+1e-6)
        sh=((ln*rn).mean(dim=1,keepdim=True)+1.)*0.5
        sf=torch.cat([sh,sh.flip(-1)],dim=-1)
        if sf.size(-1)!=W: sf=F.interpolate(sf,size=(H,W),mode="bilinear",align_corners=False)
        return sf.clamp(0.,1.)

    @torch.no_grad()
    def _maybe_update_symmetry_prior_unsup(self, sym_map):
        if self.symmetry_prior_unsup_lr<=0. or not self.training: return
        m=float(sym_map.detach().mean().clamp(0.,1.))
        v=(1.-self.symmetry_prior_unsup_lr)*float(self.symmetry_prior_scale_ema)+self.symmetry_prior_unsup_lr*m
        self.symmetry_prior_scale_ema.fill_(min(1.,max(0.,v)))

    def _dorsal_spatial_gate_map(self, L):
        pad=self._dorsal_gate_pad
        g1=F.conv2d(L,self._gauss_k_small,padding=pad); g4=F.conv2d(L,self._gauss_k_large,padding=pad)
        dog=g1-g4; dn=dog/(dog.abs().amax(dim=(2,3),keepdim=True).clamp_min(1e-6))
        s_dog=torch.sigmoid(3.*dn)
        tex=L-g4; te=tex.abs()/(tex.abs().amax(dim=(2,3),keepdim=True).clamp_min(1e-6))
        kloc=15; ploc=kloc//2
        lm=F.avg_pool2d(L,kloc,stride=1,padding=ploc)
        lsq=F.avg_pool2d(L*L,kloc,stride=1,padding=ploc)
        ls=torch.sqrt((lsq-lm*lm).clamp_min(0.)+1e-6)
        s_fg=torch.sigmoid(1.5*(L-lm)/ls)
        gate=s_dog*s_fg*(0.5+0.5*te)
        gate=gate/(gate.amax(dim=(2,3),keepdim=True).clamp_min(1e-6))
        if self.use_symmetry_gate_prior:
            sym=self._vertical_symmetry_map(L); self._maybe_update_symmetry_prior_unsup(sym)
            scale=float(self.symmetry_prior_scale_ema) if self.symmetry_prior_unsup_lr>0. else 1.
            gate=gate*(1.+float(self.symmetry_gate_alpha)*scale*sym)
            gate=gate/(gate.amax(dim=(2,3),keepdim=True).clamp_min(1e-6))
        return gate

    # ── ventral stack helpers ─────────────────────────────────────────────────

    def _ventral_l1_l2(self, freq_inputs, l2_pfc_feedback=None):
        l1_out,l1_fl,l1_pa=[],[],[]
        for i,fi in enumerate(freq_inputs):
            y1m,y1f,p1=self.l1_freq_layers[i](fi)
            l1_out.append(y1m); l1_fl.append(y1f); l1_pa.append(p1)
        l2_out,l2_fl,l2_pa=[],[],[]
        for i,y1m in enumerate(l1_out):
            y1d=self.dropout(y1m)
            l2_in=torch.cat([y1d,freq_inputs[i].mean(dim=1,keepdim=True)],dim=1) if self.cascade_skip_connections else y1d
            y2m,y2f,p2=self.l2_freq_layers[i](l2_in)
            if l2_pfc_feedback is not None:
                g=l2_pfc_feedback[i]
                if g.shape[2:]!=y2m.shape[2:]: g=F.interpolate(g,size=y2m.shape[2:],mode="bilinear",align_corners=False)
                y2m=y2m+g.expand_as(y2m); y2f=y2m.reshape(y2m.size(0),-1)
            l2_out.append(y2m); l2_fl.append(y2f); l2_pa.append(p2)
        return l1_out,l1_fl,l1_pa,l2_out,l2_fl,l2_pa

    def _l3_inputs(self, l1_out, l2_out):
        res=[]
        for i,y2m in enumerate(l2_out):
            y2d=self.dropout(y2m)
            res.append(torch.cat([y2d,l1_out[i]],dim=1) if self.cascade_skip_connections else y2d)
        return res

    def _ventral_l3_l4(self, l3_inputs, l1_out, l2_out, l3_fb=None, l4_fb=None):
        l3_out,l3_fl,l3_pa=[],[],[]
        for i,l3i in enumerate(l3_inputs):
            y3m,y3f,p3=self.l3_freq_layers[i](l3i)
            if l3_fb is not None:
                g=l3_fb[i]
                if g.shape[2:]!=y3m.shape[2:]: g=F.interpolate(g,size=y3m.shape[2:],mode="bilinear",align_corners=False)
                y3m=y3m+g.expand_as(y3m); y3f=y3m.reshape(y3m.size(0),-1)
            l3_out.append(y3m); l3_fl.append(y3f); l3_pa.append(p3)
        l4_out,l4_fl,l4_pa=[],[],[]
        for i,y3m in enumerate(l3_out):
            y3d=self.dropout(y3m)
            l4i=torch.cat([y3d,l2_out[i],l1_out[i]],dim=1) if self.cascade_skip_connections else y3d
            y4m,y4f,p4=self.l4_freq_layers[i](l4i)
            if l4_fb is not None:
                g=l4_fb[i]
                if g.shape[2:]!=y4m.shape[2:]: g=F.interpolate(g,size=y4m.shape[2:],mode="bilinear",align_corners=False)
                y4m=y4m+g.expand_as(y4m); y4f=y4m.reshape(y4m.size(0),-1)
            l4_out.append(y4m); l4_fl.append(y4f); l4_pa.append(p4)
        return l3_out,l3_fl,l3_pa,l4_out,l4_fl,l4_pa

    def _run_mt_pp(self, l3_inputs, l1_out, l2_out, mt_fb=None, pp_fb=None):
        mt_fl,mt_pa,pp_fl,pp_pa=[],[],[],[]
        for i in range(self.num_freqs):
            ymt,mf,mp=self.mt_freq_layers[i](l3_inputs[i])
            ymtd=self.dropout(ymt)
            if mt_fb is not None:
                g=mt_fb[i]
                if g.shape[2:]!=ymtd.shape[2:]: g=F.interpolate(g,size=ymtd.shape[2:],mode="bilinear",align_corners=False)
                ymtd=ymtd+g.expand_as(ymtd); mf=ymtd.reshape(ymtd.size(0),-1)
            ppi=torch.cat([ymtd,l2_out[i],l1_out[i]],dim=1) if self.cascade_skip_connections else ymtd
            ypp,pf,pp=self.pp_freq_layers[i](ppi)
            if pp_fb is not None:
                g=pp_fb[i]
                if g.shape[2:]!=ypp.shape[2:]: g=F.interpolate(g,size=ypp.shape[2:],mode="bilinear",align_corners=False)
                ypp=ypp+g.expand_as(ypp); pf=ypp.reshape(ypp.size(0),-1)
            mt_fl.append(mf); mt_pa.append(mp); pp_fl.append(pf); pp_pa.append(pp)
        return mt_fl,mt_pa,pp_fl,pp_pa

    # ── deep PFC feedback ─────────────────────────────────────────────────────

    def _pfc_deep_feedback_any_strength(self):
        return bool(self.use_pfc_deep_feedback) and any(float(x)>0. for x in (
            self.pfc_deep_fb_strength_l2,self.pfc_deep_fb_strength_l3,
            self.pfc_deep_fb_strength_l4,self.pfc_deep_fb_strength_mt,self.pfc_deep_fb_strength_pp))

    def _pfc_deep_feedback_spatial_maps(self, pfc_state):
        B,D=pfc_state.shape; f,h,w=self.num_freqs,self.spatial_size,self.spatial_size
        ps=pfc_state.detach(); z=torch.tanh(ps@self.pfc_fb_Wd+self.pfc_fb_bd)
        out={}
        for name in ("l2","l3","l4","mt","pp"):
            raw=(z@getattr(self,f"pfc_fb_U_{name}")).view(B,f,h,w)
            out[name]=raw/(raw.abs().amax(dim=(2,3),keepdim=True).clamp_min(1e-6))
        return out

    def _fb_lists(self, m): return [m[:,i:i+1,:,:] for i in range(self.num_freqs)]

    # ── PFC spatial attention ─────────────────────────────────────────────────

    def _pfc_spatial_attention_map(self, flat):
        B,D=flat.shape; f,h,w=self.num_freqs,self.spatial_size,self.spatial_size
        x=flat.view(B,f,h,w); m=x.detach().abs().mean(dim=1,keepdim=True)
        return m/(m.amax(dim=(2,3),keepdim=True).clamp_min(1e-6))

    def _apply_spatial_gate_to_flat(self, flat, gate):
        B,D=flat.shape; f,h,w=self.num_freqs,self.spatial_size,self.spatial_size
        if gate.shape[2:]!=(h,w): gate=F.interpolate(gate,size=(h,w),mode="bilinear",align_corners=False)
        s=float(self.pfc_spatial_gate_strength); fl=float(self.pfc_spatial_gate_floor)
        return (flat.view(B,f,h,w)*((1.-s)+s*(fl+(1.-fl)*gate))).view(B,D)

    def _apply_pfc_spatial_readout_gate(self, te_flat, pfc_state):
        if not self.use_pfc_spatial_readout_gate or self.pfc_spatial_gate_strength<=0.:
            return te_flat,pfc_state
        attn=self._pfc_spatial_attention_map(pfc_state)
        return self._apply_spatial_gate_to_flat(te_flat,attn),self._apply_spatial_gate_to_flat(pfc_state,attn)

    # ── IT↔PP cross-gate ──────────────────────────────────────────────────────

    def _it_pp_bidirectional_cross_gate(self, te, pp):
        if not self.use_dorsal_stream or not self.it_pp_cross_gate: return te,pp
        a=float(self.it_pp_cross_pp_to_te); b=float(self.it_pp_cross_te_to_pp)
        if a<=0. and b<=0.: return te,pp
        B,D=te.shape; f,h,w=self.num_freqs,self.spatial_size,self.spatial_size
        for _ in range(max(1,self.it_pp_cross_iters)):
            mpp=self._pfc_spatial_attention_map(pp.detach())
            mte=self._pfc_spatial_attention_map(te.detach())
            if a>0.: te=(te.view(B,f,h,w)*(1.+a*mpp)).view(B,D)
            if b>0.: pp=(pp.view(B,f,h,w)*(1.+b*mte)).view(B,D)
        return te,pp

    # ── pre-Hopfield fusion ───────────────────────────────────────────────────

    def _pfc_fuse_te_pp_pre_hopfield(self, te, pp):
        m=self.pfc_pre_hopfield_fusion
        if m=="pp_only": return pp
        wte,wpp=self.pfc_pre_blend_w_te,self.pfc_pre_blend_w_pp
        s=wte+wpp; blend=(wte*te+wpp*pp)/s if s>0. else pp
        cat=torch.cat([te,pp],dim=1)
        g=torch.sigmoid(F.linear(cat,self.pfc_pre_gate_W,self.pfc_pre_gate_b))
        gate=g*te+(1.-g)*pp
        if m=="blend": return blend
        if m=="gate": return gate
        return (pp+blend+gate)/3.

    def _pfc_fuse_te_pfc_post_readout(self, te, pfc):
        m=self.pfc_post_readout_fusion
        if m=="concat": return pfc
        cat=torch.cat([te,pfc],dim=1)
        g=torch.sigmoid(F.linear(cat,self.pfc_post_gate_W,self.pfc_post_gate_b))
        gated=g*te+(1.-g)*pfc
        return gated if m=="gate" else 0.5*(pfc+gated)

    def _pfc_classifier_readout(self, te, pfc):
        if not self.use_dorsal_stream: return pfc
        return torch.cat([te,self._pfc_fuse_te_pfc_post_readout(te,pfc)],dim=1)

    # ── PFC top-down helpers ──────────────────────────────────────────────────

    def _pfc_attention_vector(self, pfc_features):
        if (pfc_features is None or pfc_features.ndim!=2 or pfc_features.size(0)<=0
                or pfc_features.size(1)<4 or not bool(self.use_pfc_hopfield)
                or not (bool(self.use_pfc_topdown_attention) or bool(self.use_pfc_predictive_feedback))):
            return None
        chunks=torch.chunk(pfc_features.detach(),4,dim=1)
        scores=torch.stack([c.abs().mean(dim=1) for c in chunks],dim=1)
        return F.softmax(scores,dim=1).mean(dim=0).to(self.pfc_topdown_W.dtype)

    def _pfc_neuron_topdown_scales(self, pfc_state):
        dev,dt=pfc_state.device,pfc_state.dtype; d=int(self._single_stream_flat_dim)
        one=torch.ones(d,device=dev,dtype=dt)
        default={"l1":one.clone(),"l2":one.clone(),"l3":one.clone(),"l4":one.clone()}
        if not bool(self.use_pfc_topdown_attention) or pfc_state.ndim!=2 or int(pfc_state.size(1))!=d:
            return default
        pfm=pfc_state.detach().mean(dim=0); mn=float(self.pfc_topdown_min_scale); mx=float(self.pfc_topdown_max_scale)
        out={}
        for ik,name in enumerate(("l1","l2","l3","l4")):
            raw=pfm*self.pfc_topdown_neuron_w[ik]
            if self.pfc_topdown_neuron_b is not None: raw=raw+self.pfc_topdown_neuron_b[ik]
            g=torch.sigmoid(raw)*(mx-mn)+mn; out[name]=g/(g.mean()+1e-8)
        return out

    def _pfc_topdown_layer_scales(self, pfc_features):
        default={"l1":1.,"l2":1.,"l3":1.,"l4":1.}
        av=self._pfc_attention_vector(pfc_features)
        if av is None: return default
        attn=F.softmax(self.pfc_topdown_W@av+self.pfc_topdown_b,dim=0)
        scales=torch.clamp(1.+float(self.pfc_topdown_strength)*(attn-0.25)*4.,
                           min=float(self.pfc_topdown_min_scale),max=float(self.pfc_topdown_max_scale))
        return {k:float(scales[i].item()) for i,k in enumerate(("l1","l2","l3","l4"))}

    def _pfc_inhibition_layer_scales(self, pfc_features):
        default={"l1":1.,"l2":1.,"l3":1.,"l4":1.}
        av=self._pfc_attention_vector(pfc_features)
        if av is None: return default
        attn=F.softmax(self.pfc_inhibition_feedback_W@av+self.pfc_inhibition_feedback_b,dim=0)
        scales=torch.clamp(1.+float(self.pfc_topdown_strength)*(attn-0.25)*4.,
                           min=float(self.pfc_topdown_min_scale),max=float(self.pfc_topdown_max_scale))
        return {k:float(scales[i].item()) for i,k in enumerate(("l1","l2","l3","l4"))}

    def _pfc_predictive_feedback_layer_scales(self, pfc_features, layer_targets):
        default={"l1":1.,"l2":1.,"l3":1.,"l4":1.}
        if not bool(self.use_pfc_predictive_feedback): return default
        av=self._pfc_attention_vector(pfc_features)
        if av is None: return default
        target=torch.tensor([float(layer_targets.get(k,1.)) for k in ("l1","l2","l3","l4")],
                             device=av.device,dtype=av.dtype).clamp_min(1e-6)
        target=target/(target.sum()+1e-8)
        pred=F.softmax(self.pfc_predictive_feedback_W@av+self.pfc_predictive_feedback_b,dim=0)
        err=(target-pred)*4.
        scales=torch.clamp(1.+float(self.pfc_predictive_strength)*err,
                           min=float(self.pfc_predictive_min_scale),max=float(self.pfc_predictive_max_scale))
        return {k:float(scales[i].item()) for i,k in enumerate(("l1","l2","l3","l4"))}

    @torch.no_grad()
    def _update_pfc_neuron_topdown_unsup(self, pfc_state, l1fl, l2fl, l3fl, l4fl):
        if (not getattr(self,"use_pfc_topdown_per_neuron",True) or not bool(self.use_pfc_topdown_attention)
                or float(self.pfc_topdown_unsup_lr)<=0. or not self.training
                or pfc_state.ndim!=2 or int(pfc_state.size(1))!=int(self._single_stream_flat_dim)):
            return
        pfm=pfc_state.detach().mean(dim=0); lr=float(self.pfc_topdown_unsup_lr); decay=float(self.pfc_topdown_decay)
        for ik,(name,fls) in enumerate(zip(("l1","l2","l3","l4"),(l1fl,l2fl,l3fl,l4fl))):
            t=torch.cat([torch.abs(fls[i]).mean(dim=0).flatten() for i in range(len(fls))],dim=0)
            t=t/(t.max()+1e-8)
            raw=self.pfc_topdown_neuron_w[ik]*pfm
            if self.pfc_topdown_neuron_b is not None: raw=raw+self.pfc_topdown_neuron_b[ik]
            err=t-torch.sigmoid(raw)
            self.pfc_topdown_neuron_w[ik].mul_(1.-decay).add_(err*pfm,alpha=lr)
            if self.pfc_topdown_neuron_b is not None:
                self.pfc_topdown_neuron_b[ik].mul_(1.-decay).add_(err,alpha=lr)
        self.pfc_topdown_neuron_w.clamp_(-2.,2.)

    @torch.no_grad()
    def _update_pfc_topdown_unsup(self, av, layer_targets):
        if getattr(self,"use_pfc_topdown_per_neuron",True): return
        if av is None or self.pfc_topdown_unsup_lr<=0. or not self.training: return
        target=torch.tensor([float(layer_targets.get(k,1.)) for k in ("l1","l2","l3","l4")],
                             device=av.device,dtype=av.dtype).clamp_min(1e-6)
        target=target/(target.sum()+1e-8)
        pred=F.softmax(self.pfc_topdown_W@av+self.pfc_topdown_b,dim=0); err=target-pred
        lr=float(self.pfc_topdown_unsup_lr); decay=float(self.pfc_topdown_decay)
        self.pfc_topdown_W.mul_(1.-decay).add_(torch.outer(err,av),alpha=lr)
        self.pfc_topdown_b.mul_(1.-decay).add_(err,alpha=lr)
        self.pfc_topdown_W.clamp_(-2.,2.); self.pfc_topdown_b.clamp_(-2.,2.)

    @torch.no_grad()
    def _update_pfc_inhibition_feedback_unsup(self, av, layer_targets):
        if av is None or self.pfc_inhibition_feedback_unsup_lr<=0. or not self.training: return
        target=torch.tensor([float(layer_targets.get(k,1.)) for k in ("l1","l2","l3","l4")],
                             device=av.device,dtype=av.dtype).clamp_min(1e-6)
        target=target/(target.sum()+1e-8)
        pred=F.softmax(self.pfc_inhibition_feedback_W@av+self.pfc_inhibition_feedback_b,dim=0); err=target-pred
        lr=float(self.pfc_inhibition_feedback_unsup_lr); decay=float(self.pfc_inhibition_feedback_decay)
        self.pfc_inhibition_feedback_W.mul_(1.-decay).add_(torch.outer(err,av),alpha=lr)
        self.pfc_inhibition_feedback_b.mul_(1.-decay).add_(err,alpha=lr)

    @torch.no_grad()
    def _update_pfc_predictive_feedback_unsup(self, av, layer_targets):
        if av is None or self.pfc_predictive_unsup_lr<=0. or not self.training: return
        target=torch.tensor([float(layer_targets.get(k,1.)) for k in ("l1","l2","l3","l4")],
                             device=av.device,dtype=av.dtype).clamp_min(1e-6)
        target=target/(target.sum()+1e-8)
        pred=F.softmax(self.pfc_predictive_feedback_W@av+self.pfc_predictive_feedback_b,dim=0); err=target-pred
        lr=float(self.pfc_predictive_unsup_lr); decay=float(self.pfc_predictive_decay)
        self.pfc_predictive_feedback_W.mul_(1.-decay).add_(torch.outer(err,av),alpha=lr)
        self.pfc_predictive_feedback_b.mul_(1.-decay).add_(err,alpha=lr)

    # ── PC layer-output mask ──────────────────────────────────────────────────

    def _pfc_pc_layer_output_gates(self, pfc_state):
        return torch.sigmoid(pfc_state@self.pfc_pc_layer_mask_W+self.pfc_pc_layer_mask_b)

    @torch.no_grad()
    def _update_pfc_pc_layer_output_mask(self, pfc_state, l1fl, l2fl, l3fl, l4fl):
        if not bool(self.use_pfc_pc_layer_output_mask) or float(self.pfc_pc_layer_mask_lr)<=0. or not self.training: return
        B=float(pfc_state.size(0))
        if B<1: return
        t1=torch.cat(l1fl,dim=1).abs().mean(dim=1,keepdim=True)
        t2=torch.cat(l2fl,dim=1).abs().mean(dim=1,keepdim=True)
        t3=torch.cat(l3fl,dim=1).abs().mean(dim=1,keepdim=True)
        t4=torch.cat(l4fl,dim=1).abs().mean(dim=1,keepdim=True)
        target=torch.cat([t1,t2,t3,t4],dim=1)
        target=target/(target.max(dim=1,keepdim=True).values.clamp_min(1e-6))
        z=pfc_state.detach(); pred=torch.sigmoid(z@self.pfc_pc_layer_mask_W+self.pfc_pc_layer_mask_b)
        err=target-pred; lr=float(self.pfc_pc_layer_mask_lr); decay=float(self.pfc_pc_layer_mask_decay)
        self.pfc_pc_layer_mask_W.mul_(1.-decay).add_(z.t()@err/B,alpha=lr)
        self.pfc_pc_layer_mask_b.mul_(1.-decay).add_(err.mean(dim=0),alpha=lr)
        self.pfc_pc_layer_mask_W.clamp_(-2.,2.); self.pfc_pc_layer_mask_b.clamp_(-2.,2.)

    def _maybe_pfc_pc_mask_layer_outputs(self, pfc_state, l1fl, l2fl, l3fl, l4fl):
        if not bool(self.use_pfc_pc_layer_output_mask): return torch.cat(l4fl,dim=1)
        if self.training and float(self.pfc_pc_layer_mask_lr)>0.:
            self._update_pfc_pc_layer_output_mask(pfc_state,l1fl,l2fl,l3fl,l4fl)
        gates=self._pfc_pc_layer_output_gates(pfc_state)
        g1,g2,g3,g4=gates[:,0:1],gates[:,1:2],gates[:,2:3],gates[:,3:4]
        for i in range(len(l1fl)):
            l1fl[i].mul_(g1); l2fl[i].mul_(g2); l3fl[i].mul_(g3); l4fl[i].mul_(g4)
        return torch.cat(l4fl,dim=1)

    # ── fusion gate unsup update ──────────────────────────────────────────────

    @torch.no_grad()
    def _update_pfc_fusion_gates_unsup(self, te_pre, pp_pre, te_r, pfc_r):
        if not self.training or not bool(self.use_pfc_fusion_gate_unsup) or float(self.pfc_fusion_gate_unsup_lr)<=0. or not bool(self.use_dorsal_stream): return
        lr=float(self.pfc_fusion_gate_unsup_lr); decay=float(self.pfc_fusion_gate_decay); eps=1e-6; cr=int(self.pfc_fusion_lms_chunk_rows)
        def _lms_update(W,b,err,x):
            bsz=float(x.size(0)); W.mul_(1.-decay); b.mul_(1.-decay)
            for rs in range(0,W.size(0),cr):
                re=min(rs+cr,W.size(0)); W[rs:re,:].add_((err[:,rs:re].T@x)/bsz,alpha=lr)
            b.add_(err.mean(dim=0),alpha=lr)
        if te_pre is not None and pp_pre is not None:
            te_d=te_pre.detach(); pp_d=pp_pre.detach(); x=torch.cat([te_d,pp_d],dim=1)
            g=torch.sigmoid(F.linear(x,self.pfc_pre_gate_W,self.pfc_pre_gate_b))
            _lms_update(self.pfc_pre_gate_W,self.pfc_pre_gate_b,te_d.abs()/(te_d.abs()+pp_d.abs()+eps)-g,x)
            self.pfc_pre_gate_W.clamp_(-4.,4.); self.pfc_pre_gate_b.clamp_(-4.,4.)
        te_r2=te_r.detach(); pfc_d=pfc_r.detach(); x2=torch.cat([te_r2,pfc_d],dim=1)
        g2=torch.sigmoid(F.linear(x2,self.pfc_post_gate_W,self.pfc_post_gate_b))
        _lms_update(self.pfc_post_gate_W,self.pfc_post_gate_b,te_r2.abs()/(te_r2.abs()+pfc_d.abs()+eps)-g2,x2)
        self.pfc_post_gate_W.clamp_(-4.,4.); self.pfc_post_gate_b.clamp_(-4.,4.)

    # ── spatial predictive gain ───────────────────────────────────────────────

    def _pfc_spatial_predictive_gain(self, pfc_state, freq_idx):
        coeffs=getattr(self,"coeffs",None)
        if coeffs is None or not bool(getattr(coeffs,"pc_per_neuron_plasticity",True)): return None
        if float(getattr(coeffs,"pc_per_neuron_pfc_weight",0.5))<=0.: return None
        D=int(pfc_state.size(1)); f,h,w=int(self.num_freqs),int(self.spatial_size),int(self.spatial_size)
        if D!=f*h*w or freq_idx<0 or freq_idx>=f: return None
        gmn=float(getattr(coeffs,"pc_per_neuron_gain_min",0.5)); gmx=float(getattr(coeffs,"pc_per_neuron_gain_max",1.5))
        with torch.no_grad():
            x=pfc_state.detach().mean(0).view(f,h,w)[freq_idx].flatten().abs()
            x=torch.clamp(x/(x.mean()+1e-8),gmn,gmx)
            return (x/(x.mean()+1e-8)).to(device=pfc_state.device,dtype=pfc_state.dtype)

    def _layer_flat_spatial_plasticity_gain(self, y_flat):
        coeffs=getattr(self,"coeffs",None)
        if coeffs is None or not bool(getattr(coeffs,"pc_per_neuron_plasticity",True)): return None
        if float(getattr(coeffs,"pc_per_neuron_layer_weight",0.34))<=0.: return None
        if y_flat.dim()!=2 or int(y_flat.size(0))<1: return None
        gmn=float(getattr(coeffs,"pc_per_neuron_gain_min",0.5)); gmx=float(getattr(coeffs,"pc_per_neuron_gain_max",1.5))
        with torch.no_grad():
            x=y_flat.detach().abs().mean(dim=0).flatten()
            x=torch.clamp(x/(x.mean()+1e-8),gmn,gmx)
            return (x/(x.mean()+1e-8)).to(device=y_flat.device,dtype=y_flat.dtype)

    # ── L1 prox ───────────────────────────────────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def _tensor_soft_threshold_l1_(w, tau, chunk_elems=2097152):
        if tau<=0.: return
        n=w.numel()
        if n<=4194304:
            ap=w.abs(); ap.sub_(tau).clamp_(min=0.); torch.mul(w.sign(),ap,out=w); return
        flat=w.reshape(-1)
        for s in range(0,n,chunk_elems):
            sl=flat[s:s+chunk_elems]; ap=sl.abs(); ap.sub_(tau).clamp_(min=0.); torch.mul(sl.sign(),ap,out=sl)

    @torch.no_grad()
    def _apply_pfc_l1_prox(self):
        if self.pfc_l1_lambda<=0. or self.pfc_l1_prox_step<=0.: return
        tau=float(self.pfc_l1_lambda*self.pfc_l1_prox_step)
        for w in [self.pfc_topdown_W,self.pfc_inhibition_feedback_W,self.pfc_predictive_feedback_W,
                  self.pfc_topdown_neuron_w,self.pfc_pc_layer_mask_W,self.pfc_pre_gate_W,self.pfc_post_gate_W]:
            self._tensor_soft_threshold_l1_(w,tau)

    @torch.no_grad()
    def _maybe_apply_local_l1_prox(self):
        if not self.training or self.local_l1_lambda<=0. or self.local_l1_prox_step<=0.: return
        self.local_l1_update_counter+=1
        if self.local_l1_update_counter<=self.local_l1_warmup_steps: return
        if (self.local_l1_update_counter%self.local_l1_apply_every)!=0: return
        tau=float(self.local_l1_lambda*self.local_l1_prox_step)
        stacks=(*self.l1_freq_layers,*self.l2_freq_layers,*self.l3_freq_layers,*self.l4_freq_layers)
        if self.use_dorsal_stream: stacks=(*stacks,*self.mt_freq_layers,*self.pp_freq_layers)
        for layer in stacks:
            self._tensor_soft_threshold_l1_(layer.W,tau); layer.W.clamp_(-layer.w_clip,layer.w_clip)

    # ── SOM schedules ─────────────────────────────────────────────────────────

    def iter_topographic_layers(self):
        for lst in (self.l1_freq_layers,self.l2_freq_layers,self.l3_freq_layers,self.l4_freq_layers):
            for layer in lst: yield layer
        if self.use_dorsal_stream:
            for lst in (self.mt_freq_layers,self.pp_freq_layers):
                for layer in lst: yield layer

    @torch.no_grad()
    def apply_som_schedules(self, epoch, total_epochs, coeffs):
        te=max(1,int(total_epochs)); progress=(float(epoch)-1.)/float(te-1) if te>1 else 0.
        for layer in self.iter_topographic_layers(): layer.apply_som_schedule(progress,coeffs)

    # ── main ventral+dorsal stack ─────────────────────────────────────────────

    def _run_ventral_dorsal_stack(self, freq_inputs, L, pfc_feedback_state=None):
        l2_fb=l3_fb=l4_fb=mt_fb=pp_fb=None
        if pfc_feedback_state is not None and self._pfc_deep_feedback_any_strength():
            dm=self._pfc_deep_feedback_spatial_maps(pfc_feedback_state)
            if self.pfc_deep_fb_strength_l2>0.: l2_fb=self._fb_lists(dm["l2"]*self.pfc_deep_fb_strength_l2)
            if self.pfc_deep_fb_strength_l3>0.: l3_fb=self._fb_lists(dm["l3"]*self.pfc_deep_fb_strength_l3)
            if self.pfc_deep_fb_strength_l4>0.: l4_fb=self._fb_lists(dm["l4"]*self.pfc_deep_fb_strength_l4)
            if self.use_dorsal_stream:
                if self.pfc_deep_fb_strength_mt>0.: mt_fb=self._fb_lists(dm["mt"]*self.pfc_deep_fb_strength_mt)
                if self.pfc_deep_fb_strength_pp>0.: pp_fb=self._fb_lists(dm["pp"]*self.pfc_deep_fb_strength_pp)
        l1_out,l1_fl,l1_pa,l2_out,l2_fl,l2_pa=self._ventral_l1_l2(freq_inputs,l2_fb)
        l3_inp=self._l3_inputs(l1_out,l2_out)
        _,l3_fl,l3_pa,_,l4_fl,l4_pa=self._ventral_l3_l4(l3_inp,l1_out,l2_out,l3_fb,l4_fb)
        te_flat=torch.cat(l4_fl,dim=1)
        mt_fl=mt_pa=pp_fl=pp_pa=None
        if self.use_dorsal_stream:
            Lm=L
            if Lm.size(2)!=l3_inp[0].size(2) or Lm.size(3)!=l3_inp[0].size(3):
                Lm=F.interpolate(Lm,size=(l3_inp[0].size(2),l3_inp[0].size(3)),mode="bilinear",align_corners=False)
            sg=self._dorsal_spatial_gate_map(Lm)
            mt_inp=[l3_inp[i]*(1.+0.75*(sg if sg.shape[2:]==l3_inp[i].shape[2:] else F.interpolate(sg,size=l3_inp[i].shape[2:],mode="bilinear",align_corners=False))) for i in range(self.num_freqs)]
            mt_fl,mt_pa,pp_fl,pp_pa=self._run_mt_pp(mt_inp,l1_out,l2_out,mt_fb,pp_fb)
            pp_flat=torch.cat(pp_fl,dim=1)
            pfc_state=pp_flat if self.pfc_pre_hopfield_fusion=="pp_only" else self._pfc_fuse_te_pp_pre_hopfield(te_flat,pp_flat)
        else:
            pfc_state=te_flat
        return l1_out,l1_fl,l1_pa,l2_out,l2_fl,l2_pa,l3_inp,l3_fl,l3_pa,l4_fl,l4_pa,te_flat,mt_fl,mt_pa,pp_fl,pp_pa,pfc_state

    def _pfc_run_second_ventral_stack(self, td_i):
        if td_i!=self.pfc_topdown_iters-1: return False
        if bool(self.use_pfc_dense_feedback) and float(self.pfc_dense_feedback_strength)>0.: return True
        if self._pfc_deep_feedback_any_strength(): return True
        return int(self.pfc_recurrent_feedback_steps)>=2 and float(self.pfc_recurrent_feedback_strength)>0.

    def _build_pfc_second_pass_freq_inputs(self, freq_inputs, pfc_state):
        fi=freq_inputs
        if int(self.pfc_recurrent_feedback_steps)>=2 and float(self.pfc_recurrent_feedback_strength)>0.:
            attn=self._pfc_spatial_attention_map(pfc_state); s=float(self.pfc_recurrent_feedback_strength)
            B,D=pfc_state.shape; f,h,w=self.num_freqs,self.spatial_size,self.spatial_size
            fi=[t*(1.+s*(attn if attn.shape[2:]==t.shape[2:] else F.interpolate(attn,size=t.shape[2:],mode="bilinear",align_corners=False))) for t in fi]
        if bool(self.use_pfc_dense_feedback) and float(self.pfc_dense_feedback_strength)>0.:
            B,D=pfc_state.shape; f,h,w=self.num_freqs,self.spatial_size,self.spatial_size
            m=pfc_state.detach().view(B,f,h,w)
            m=m/(m.abs().amax(dim=(2,3),keepdim=True).clamp_min(1e-6)); alpha=float(self.pfc_dense_feedback_strength)
            fi=[fi[i]+alpha*(m[:,i:i+1,:,:] if m[:,i:i+1,:,:].shape[2:]==fi[i].shape[2:] else F.interpolate(m[:,i:i+1,:,:],size=fi[i].shape[2:],mode="bilinear",align_corners=False)) for i in range(f)]
        return fi

    # ── plasticity update helper ──────────────────────────────────────────────

    def _do_plasticity_updates(self, pfc_state, td_total, inh_total,
                                l1_fl,l1_pa,l2_fl,l2_pa,l3_fl,l3_pa,l4_fl,l4_pa,
                                mt_fl,mt_pa,pp_fl,pp_pa):
        _n=int(self.spatial_size*self.spatial_size)
        for i in range(self.num_freqs):
            sl=slice(i*_n,(i+1)*_n); _pg=self._pfc_spatial_predictive_gain(pfc_state,i)
            v1=td_total["l1"][sl] if isinstance(td_total["l1"],torch.Tensor) else td_total["l1"]
            v2=td_total["l2"][sl] if isinstance(td_total["l2"],torch.Tensor) else td_total["l2"]
            v3=td_total["l3"][sl] if isinstance(td_total["l3"],torch.Tensor) else td_total["l3"]
            v4=td_total["l4"][sl] if isinstance(td_total["l4"],torch.Tensor) else td_total["l4"]
            self.l1_freq_layers[i].update_from_patches(l1_pa[i],l1_fl[i],variance_scale=v1,inhibition_scale=inh_total["l1"],layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l1_fl[i]),pfc_spatial_gain=_pg)
            self.l2_freq_layers[i].update_from_patches(l2_pa[i],l2_fl[i],variance_scale=v2,inhibition_scale=inh_total["l2"],layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l2_fl[i]),pfc_spatial_gain=_pg)
            self.l3_freq_layers[i].update_from_patches(l3_pa[i],l3_fl[i],variance_scale=v3,inhibition_scale=inh_total["l3"],layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l3_fl[i]),pfc_spatial_gain=_pg)
            self.l4_freq_layers[i].update_from_patches(l4_pa[i],l4_fl[i],variance_scale=v4,inhibition_scale=inh_total["l4"],layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(l4_fl[i]),pfc_spatial_gain=_pg)
            if self.use_dorsal_stream and mt_pa is not None and pp_pa is not None:
                self.mt_freq_layers[i].update_from_patches(mt_pa[i],mt_fl[i],variance_scale=v3,inhibition_scale=inh_total["l3"],layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(mt_fl[i]),pfc_spatial_gain=_pg)
                self.pp_freq_layers[i].update_from_patches(pp_pa[i],pp_fl[i],variance_scale=v4,inhibition_scale=inh_total["l4"],layer_spatial_gain=self._layer_flat_spatial_plasticity_gain(pp_fl[i]),pfc_spatial_gain=_pg)
        self._maybe_apply_local_l1_prox()

    def _blend_layer_targets(self, l1a, l2a, l3a, l4a):
        return {"l1":l1a,"l2":l2a,"l3":l3a,"l4":l4a}

    def _compute_td_inh_totals(self, pfc_state, layer_targets, plasticity_gain_vec=None):
        td_scales=(self._pfc_neuron_topdown_scales(pfc_state) if getattr(self,"use_pfc_topdown_per_neuron",True)
                   else self._pfc_topdown_layer_scales(pfc_state))
        inh_scales=self._pfc_inhibition_layer_scales(pfc_state)
        pred_scales=self._pfc_predictive_feedback_layer_scales(pfc_state,layer_targets)
        td_total={k:td_scales[k]*float(pred_scales[k]) for k in ("l1","l2","l3","l4")}
        inh_total={k:float(inh_scales[k]*pred_scales[k]) for k in ("l1","l2","l3","l4")}
        if plasticity_gain_vec is not None:
            pg=plasticity_gain_vec.view(4).detach()
            for ik,k in enumerate(("l1","l2","l3","l4")):
                td_total[k]=td_total[k]*float(pg[ik].item())
                inh_total[k]=inh_total[k]*float(pg[ik].item())
        return td_total,inh_total

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x, local_update=True, plasticity_gain_vec=None,
                layer_scale_vec=None, readout_scale=None):
        if x.size(1)!=3: raise ValueError(f"Expected RGB, got {x.size(1)} channels")
        freq_inputs,L=self._rgb_to_freq_inputs(x)
        (l1_out,l1_fl,l1_pa,l2_out,l2_fl,l2_pa,_,l3_fl,l3_pa,
         l4_fl,l4_pa,te_flat,mt_fl,mt_pa,pp_fl,pp_pa,pfc_state)=self._run_ventral_dorsal_stack(freq_inputs,L)
        fusion_te_pre=te_flat
        fusion_pp_pre=torch.cat(pp_fl,dim=1) if self.use_dorsal_stream and pp_fl is not None else None
        if self.use_dorsal_stream: te_flat,pfc_state=self._it_pp_bidirectional_cross_gate(te_flat,pfc_state)
        l1a=float(torch.cat(l1_fl,dim=1).detach().abs().mean().item())
        l2a=float(torch.cat(l2_fl,dim=1).detach().abs().mean().item())
        l3a=float(torch.cat(l3_fl,dim=1).detach().abs().mean().item())
        for td_i in range(self.pfc_topdown_iters):
            if self.pfc_hopfield is not None:
                pfc_state=self.pfc_hopfield(pfc_state,update_memory=bool(local_update and self.training))
            if self._pfc_run_second_ventral_stack(td_i):
                ff=self._build_pfc_second_pass_freq_inputs(freq_inputs,pfc_state)
                (l1_out,l1_fl,l1_pa,l2_out,l2_fl,l2_pa,_,l3_fl,l3_pa,
                 l4_fl,l4_pa,te_flat,mt_fl,mt_pa,pp_fl,pp_pa,pfc_state)=self._run_ventral_dorsal_stack(
                    ff,L,pfc_state.detach() if self._pfc_deep_feedback_any_strength() else None)
                fusion_te_pre=te_flat
                fusion_pp_pre=torch.cat(pp_fl,dim=1) if self.use_dorsal_stream and pp_fl is not None else None
                if self.use_dorsal_stream: te_flat,pfc_state=self._it_pp_bidirectional_cross_gate(te_flat,pfc_state)
                l1a=float(torch.cat(l1_fl,dim=1).detach().abs().mean().item())
                l2a=float(torch.cat(l2_fl,dim=1).detach().abs().mean().item())
                l3a=float(torch.cat(l3_fl,dim=1).detach().abs().mean().item())
            l4a=float(pfc_state.detach().abs().mean().item())
            layer_targets=self._blend_layer_targets(l1a,l2a,l3a,l4a)
            td_total,inh_total=self._compute_td_inh_totals(pfc_state,layer_targets,plasticity_gain_vec)
            if td_i==self.pfc_topdown_iters-1:
                gg=self._neuron_glia_per_layer_gates(layer_targets,update_trace=True)
                for ik,k in enumerate(("l1","l2","l3","l4")):
                    td_total[k]=td_total[k]*gg[ik]; inh_total[k]=inh_total[k]*gg[ik]
            av=self._pfc_attention_vector(pfc_state)
            self._update_pfc_neuron_topdown_unsup(pfc_state,l1_fl,l2_fl,l3_fl,l4_fl)
            self._update_pfc_topdown_unsup(av,layer_targets)
            self._update_pfc_inhibition_feedback_unsup(av,layer_targets)
            self._update_pfc_predictive_feedback_unsup(av,layer_targets)
            if local_update and self.training and td_i==self.pfc_topdown_iters-1:
                self._do_plasticity_updates(pfc_state,td_total,inh_total,l1_fl,l1_pa,l2_fl,l2_pa,l3_fl,l3_pa,l4_fl,l4_pa,mt_fl,mt_pa,pp_fl,pp_pa)
        if layer_scale_vec is not None:
            ls=layer_scale_vec.view(4).to(device=l1_fl[0].device,dtype=l1_fl[0].dtype)
            for i in range(self.num_freqs):
                l1_fl[i]=l1_fl[i].detach()*ls[0]; l2_fl[i]=l2_fl[i].detach()*ls[1]
                l3_fl[i]=l3_fl[i].detach()*ls[2]; l4_fl[i]=l4_fl[i].detach()*ls[3]
            te_flat=torch.cat(l4_fl,dim=1)
        if bool(self.use_pfc_pc_layer_output_mask):
            te_flat=self._maybe_pfc_pc_mask_layer_outputs(pfc_state,l1_fl,l2_fl,l3_fl,l4_fl)
            if self.use_dorsal_stream: te_flat,pfc_state=self._it_pp_bidirectional_cross_gate(te_flat,pfc_state)
        te_flat,pfc_state=self._apply_pfc_spatial_readout_gate(te_flat,pfc_state)
        self._update_pfc_fusion_gates_unsup(fusion_te_pre,fusion_pp_pre,te_flat,pfc_state)
        readout=self._pfc_classifier_readout(te_flat,pfc_state)
        if readout_scale is not None:
            return self.clf(self.dropout(readout.detach()*readout_scale.view(()).to(device=readout.device,dtype=readout.dtype)))
        return self.clf((readout.detach() if self.training else readout) if not self.dropout.training else self.dropout(readout.detach() if self.training else readout))

    def forward_features(self, x, local_update=True):
        if x.size(1)!=3: raise ValueError(f"Expected RGB, got {x.size(1)} channels")
        freq_inputs,L=self._rgb_to_freq_inputs(x)
        (l1_out,l1_fl,l1_pa,l2_out,l2_fl,l2_pa,_,l3_fl,l3_pa,
         l4_fl,l4_pa,te_flat,mt_fl,mt_pa,pp_fl,pp_pa,pfc_state)=self._run_ventral_dorsal_stack(freq_inputs,L)
        fusion_te_pre=te_flat
        fusion_pp_pre=torch.cat(pp_fl,dim=1) if self.use_dorsal_stream and pp_fl is not None else None
        if self.use_dorsal_stream: te_flat,pfc_state=self._it_pp_bidirectional_cross_gate(te_flat,pfc_state)
        l1a=float(torch.cat(l1_fl,dim=1).detach().abs().mean().item())
        l2a=float(torch.cat(l2_fl,dim=1).detach().abs().mean().item())
        l3a=float(torch.cat(l3_fl,dim=1).detach().abs().mean().item())
        for td_i in range(self.pfc_topdown_iters):
            if self.pfc_hopfield is not None:
                pfc_state=self.pfc_hopfield(pfc_state,update_memory=bool(local_update and self.training))
            if self._pfc_run_second_ventral_stack(td_i):
                ff=self._build_pfc_second_pass_freq_inputs(freq_inputs,pfc_state)
                (l1_out,l1_fl,l1_pa,l2_out,l2_fl,l2_pa,_,l3_fl,l3_pa,
                 l4_fl,l4_pa,te_flat,mt_fl,mt_pa,pp_fl,pp_pa,pfc_state)=self._run_ventral_dorsal_stack(
                    ff,L,pfc_state.detach() if self._pfc_deep_feedback_any_strength() else None)
                fusion_te_pre=te_flat
                fusion_pp_pre=torch.cat(pp_fl,dim=1) if self.use_dorsal_stream and pp_fl is not None else None
                if self.use_dorsal_stream: te_flat,pfc_state=self._it_pp_bidirectional_cross_gate(te_flat,pfc_state)
                l1a=float(torch.cat(l1_fl,dim=1).detach().abs().mean().item())
                l2a=float(torch.cat(l2_fl,dim=1).detach().abs().mean().item())
                l3a=float(torch.cat(l3_fl,dim=1).detach().abs().mean().item())
            l4a=float(pfc_state.detach().abs().mean().item())
            layer_targets=self._blend_layer_targets(l1a,l2a,l3a,l4a)
            td_total,inh_total=self._compute_td_inh_totals(pfc_state,layer_targets)
            if td_i==self.pfc_topdown_iters-1:
                gg=self._neuron_glia_per_layer_gates(layer_targets,update_trace=True)
                for ik,k in enumerate(("l1","l2","l3","l4")):
                    td_total[k]=td_total[k]*gg[ik]; inh_total[k]=inh_total[k]*gg[ik]
            av=self._pfc_attention_vector(pfc_state)
            self._update_pfc_neuron_topdown_unsup(pfc_state,l1_fl,l2_fl,l3_fl,l4_fl)
            self._update_pfc_topdown_unsup(av,layer_targets)
            self._update_pfc_inhibition_feedback_unsup(av,layer_targets)
            self._update_pfc_predictive_feedback_unsup(av,layer_targets)
            if local_update and self.training and td_i==self.pfc_topdown_iters-1:
                self._do_plasticity_updates(pfc_state,td_total,inh_total,l1_fl,l1_pa,l2_fl,l2_pa,l3_fl,l3_pa,l4_fl,l4_pa,mt_fl,mt_pa,pp_fl,pp_pa)
        if bool(self.use_pfc_pc_layer_output_mask):
            te_flat=self._maybe_pfc_pc_mask_layer_outputs(pfc_state,l1_fl,l2_fl,l3_fl,l4_fl)
            if self.use_dorsal_stream: te_flat,pfc_state=self._it_pp_bidirectional_cross_gate(te_flat,pfc_state)
        te_flat,pfc_state=self._apply_pfc_spatial_readout_gate(te_flat,pfc_state)
        self._update_pfc_fusion_gates_unsup(fusion_te_pre,fusion_pp_pre,te_flat,pfc_state)
        return self.dropout(self._pfc_classifier_readout(te_flat,pfc_state))

    # ── pseudo-inverse classifier (FIXED: inside class, uses self.clf) ────────

    @torch.no_grad()
    def fit_pseudoinverse_classifier(self, loader, device, ridge=1e-4, max_batches=-1):
        """Dual-form ridge: invert [N,N] not [D,D]. Exact, fast when N<<D."""
        self.eval()
        Phi_list, Y_list = [], []
        for nb, (x, y) in enumerate(loader):
            if max_batches > 0 and nb >= max_batches: break
            x = x.to(device, non_blocking=True)
            Phi_list.append(self.forward_features(x, local_update=False).float().cpu())
            Y_list.append(y.cpu())
        if not Phi_list: return
        Phi = torch.cat(Phi_list, 0)              # [N, D]
        Y   = torch.cat(Y_list, 0).long()         # [N]
        N, D = Phi.shape; C = self.clf.out_features

        mu  = Phi.mean(0, keepdim=True)
        std = Phi.std(0, keepdim=True).clamp_min(1e-6)
        Phi_s = ((Phi - mu) / std).double()       # [N, D]

        Y_oh = torch.zeros(N, C, dtype=torch.float64)
        Y_oh.scatter_(1, Y.unsqueeze(1), 1.0)
        ymean = Y_oh.mean(0, keepdim=True)
        Y_c = Y_oh - ymean

        # dual solve: W = Phi_s^T (Phi_s Phi_s^T + ridge*N I)^{-1} Y_c
        G = Phi_s @ Phi_s.T                        # [N, N]  <-- small!
        G.diagonal().add_(ridge * N)
        alpha = torch.linalg.solve(G, Y_c)         # [N, C]  fast
        Ws = Phi_s.T @ alpha                        # [D, C]

        W_real = (Ws / std.T.double())
        b_real = (ymean - (mu.double()/std.double()) @ Ws).squeeze(0)
        self.clf.weight.data.copy_(W_real.T.float().to(device))
        self.clf.bias.data.copy_(b_real.float().to(device))
    # ── misc ──────────────────────────────────────────────────────────────────

    def get_pfc_consistency_loss(self, device=None):
        if self.pfc_hopfield is None:
            dev=device if device is not None else next(self.clf.parameters()).device
            return torch.zeros((),device=dev)
        loss=self.pfc_hopfield.get_last_consistency_loss()
        if loss is None:
            dev=device if device is not None else next(self.clf.parameters()).device
            return torch.zeros((),device=dev)
        return loss

    def classifier_parameters(self):
        if self.pfc_hopfield is None or bool(getattr(self.pfc_hopfield,"unsup_update",False)):
            return self.clf.parameters()
        return list(self.clf.parameters())+list(self.pfc_hopfield.parameters())

    def count_parameters(self):
        def np_(m): return sum(p.numel() for p in m.parameters())
        def swb(ls): return sum(l.W.numel()+l.b.numel() for l in ls)
        c={"Classifier_grad":self.clf.weight.numel()+self.clf.bias.numel(),
           "PFC_grad":np_(self.pfc_hopfield) if self.pfc_hopfield else 0,
           "L1_Wb":swb(self.l1_freq_layers),"L2_Wb":swb(self.l2_freq_layers),
           "L3_Wb":swb(self.l3_freq_layers)+swb(self.mt_freq_layers),
           "L4_Wb":swb(self.l4_freq_layers)+swb(self.pp_freq_layers)}
        total=sum(c.values()); grad=c["Classifier_grad"]+c["PFC_grad"]
        c["TOTAL"]=total; c["Grad_%"]=float(100.*grad/max(1,total))
        return c
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# ---------------------------------------------------------------------------
# TrainingRunResult
# ---------------------------------------------------------------------------

@dataclass
class TrainingRunResult:
    best_val_acc: float
    best_epoch: int
    best_test_at_best_val: Optional[float]
    final_val_loss: float
    final_val_acc: float
    final_test_loss: float
    final_test_acc: float


# ---------------------------------------------------------------------------
# _build_model — centralised construction with ablation flags
# ---------------------------------------------------------------------------

def _build_model(args: argparse.Namespace, device: torch.device) -> VisNetUnified:
    """
    Build VisNetUnified from parsed args.
    Supports:
      --inhibition-mode none  → forces inhibition_strength=0, wta=0 everywhere
      --disable-pfc           → disables all PFC / predictive-coding paths
    """
    inh_mode     = str(getattr(args, "inhibition_mode", "kernel")).lower()
    no_inhibition = (inh_mode == "none")
    disable_pfc   = bool(getattr(args, "disable_pfc", False))

    coeffs = UnifiedCoeffs(
        eta                         = float(getattr(args, "lambda_d",    5e-9)),
        lambda_d                    = float(getattr(args, "lambda_d",    1e-4)),
        alpha                       = float(getattr(args, "holo_alpha",  0.15)),
        inhibition_mode             = "kernel" if no_inhibition else inh_mode,
        use_soft_inhibition         = bool(getattr(args, "soft_inhibition",          True)),
        inhibition_no_threshold     = bool(getattr(args, "inhibition_no_threshold",  True)),
        inhibition_smooth_scale     = float(getattr(args, "inhibition_smooth_scale", 0.0)),
        averaging_inhibition_size   = int(getattr(args, "averaging_inhibition_size", 7)),
        kernel_inhibition_size      = int(getattr(args, "kernel_inhibition_size",    7)),
        kernel_inhibition_sigma     = float(getattr(args, "kernel_inhibition_sigma", 1.5)),
        inhibition_dropout          = float(getattr(args, "inhibition_dropout",      0.01)),
        adaptive_inhibition         = False if no_inhibition else bool(getattr(args, "adaptive_inhibition",       False)),
        use_homeostatic_threshold   = False if no_inhibition else bool(getattr(args, "use_homeostatic_threshold", False)),
        homeostatic_threshold_lr    = float(getattr(args, "homeostatic_threshold_lr",    0.02)),
        homeostatic_threshold_theta_init = float(getattr(args, "homeostatic_threshold_theta_init", 0.0)),
        homeostatic_threshold_min   = float(getattr(args, "homeostatic_threshold_min",   -2.0)),
        homeostatic_threshold_max   = float(getattr(args, "homeostatic_threshold_max",    2.0)),
        use_oscillatory_inhibition  = False if no_inhibition else bool(getattr(args, "oscillatory_inhibition",    False)),
        phase_period                = int(getattr(args, "phase_period",            10)),
        phase_gate_sharpness        = float(getattr(args, "phase_gate_sharpness",  1.0)),
        use_divisive_inhibition_norm = False if no_inhibition else bool(getattr(args, "divisive_inhibition_norm", True)),
        divisive_mode               = str(getattr(args, "divisive_mode",           "both")),
        divisive_w_global           = float(getattr(args, "divisive_w_global",     0.5)),
        divisive_w_local            = float(getattr(args, "divisive_w_local",      0.5)),
        divisive_local_size         = int(getattr(args, "divisive_local_size",     9)),
        divisive_local_sigma        = float(getattr(args, "divisive_local_sigma",  2.0)),
        divisive_alpha              = float(getattr(args, "divisive_alpha",        0.85)),
        divisive_beta               = float(getattr(args, "divisive_beta",         0.36)),
        target_active_frac          = float(getattr(args, "target_active_frac",    0.03)),
        inhibition_adapt_lr         = float(getattr(args, "inhibition_adapt_lr",   0.002)),
        inhibition_min              = float(getattr(args, "inhibition_min",         0.0)),
        inhibition_max              = float(getattr(args, "inhibition_max",         1.0)),
        inhibition_learning_sparsity = float(getattr(args, "inhibition_learning_sparsity", 0.5)),
        mixed_inhibition_w_global   = float(getattr(args, "mixed_inhibition_w_global",   0.5)),
        mixed_inhibition_w_kernel   = float(getattr(args, "mixed_inhibition_w_kernel",   0.25)),
        mixed_inhibition_w_averaging = float(getattr(args, "mixed_inhibition_w_averaging", 0.25)),
        cascade_skip_connections    = bool(getattr(args, "cascade_skip_connections", True)),
        use_wavelet_input           = bool(getattr(args, "wavelet_input",            True)),
        use_wavelet_binding         = not bool(getattr(args, "no_wavelet_binding",   False)),
        use_wavelet_denoise         = not bool(getattr(args, "no_wavelet_denoise",   False)),
        wavelet_threshold           = float(getattr(args, "wavelet_threshold",       0.02)),
        gamma_wavelet               = float(getattr(args, "gamma_wavelet",           0.01)),
        holo_update_freq            = int(getattr(args, "holo_update_freq",          20)),
        holo_fast_lr                = float(getattr(args, "holo_fast_lr",            0.1)),
        holo_fast_decay             = float(getattr(args, "holo_fast_decay",         0.1)),
        holo_corr_blend             = float(getattr(args, "holo_corr_blend",         0.2)),
        holo_fast_norm_cap          = float(getattr(args, "holo_fast_norm_cap",      12.0)),
        use_structural_plasticity   = bool(getattr(args, "structural_plasticity",    True)),
        structural_update_freq      = int(getattr(args, "structural_update_freq",    1200)),
        structural_prune_threshold  = float(getattr(args, "structural_prune_threshold", 1e-4)),
        structural_prune_max_frac   = float(getattr(args, "structural_prune_max_frac",  0.002)),
        structural_grow_threshold   = float(getattr(args, "structural_grow_threshold",  0.02)),
        structural_grow_max_frac    = float(getattr(args, "structural_grow_max_frac",   0.001)),
        structural_grow_init_scale  = float(getattr(args, "structural_grow_init_scale", 0.02)),
        unsup_mix_mode              = str(getattr(args, "unsup_mix_mode",             "adaptive")),
        unsup_mix_normalize_terms   = not bool(getattr(args, "no_unsup_mix_normalize_terms", False)),
        unsup_mix_temperature       = float(getattr(args, "unsup_mix_temperature",   1.0)),
        unsup_mix_random_alpha      = float(getattr(args, "unsup_mix_random_alpha",  20.0)),
        unsup_mix_w_hebb            = float(getattr(args, "unsup_mix_w_hebb",        0.30)),
        unsup_mix_w_holo            = float(getattr(args, "unsup_mix_w_holo",        0.10)),
        unsup_mix_w_hyp             = float(getattr(args, "unsup_mix_w_hyp",         0.10)),
        unsup_mix_w_wave            = float(getattr(args, "unsup_mix_w_wave",        0.15)),
        unsup_mix_w_anti            = float(getattr(args, "unsup_mix_w_anti",        0.10)),
        unsup_mix_w_cons            = float(getattr(args, "unsup_mix_w_cons",        0.08)),
        unsup_mix_w_rec             = float(getattr(args, "unsup_mix_w_rec",         0.08)),
        unsup_mix_w_free            = float(getattr(args, "unsup_mix_w_free",        0.06)),
        unsup_mix_w_decay           = float(getattr(args, "unsup_mix_w_decay",       0.03)),
        unsup_mix_w_dist            = float(getattr(args, "unsup_mix_w_dist",        0.00)),
        pc_per_neuron_plasticity    = bool(getattr(args, "pc_per_neuron_plasticity", True)),
        pc_per_neuron_layer_weight  = float(getattr(args, "pc_per_neuron_layer_weight", 0.34)),
        pc_per_neuron_pfc_weight    = float(getattr(args, "pc_per_neuron_pfc_weight",   0.33)),
        pc_per_neuron_trace_weight  = float(getattr(args, "pc_per_neuron_trace_weight", 0.33)),
        pc_per_neuron_gain_min      = float(getattr(args, "pc_per_neuron_gain_min",    0.5)),
        pc_per_neuron_gain_max      = float(getattr(args, "pc_per_neuron_gain_max",    1.5)),
        lateral_w_hebb              = float(getattr(args, "lateral_w_hebb",  1.0)),
        lateral_w_anti              = float(getattr(args, "lateral_w_anti",  0.0)),
        lateral_w_cov               = float(getattr(args, "lateral_w_cov",   0.0)),
        lateral_w_holo              = float(getattr(args, "lateral_w_holo",  0.05)),
        lateral_w_hyp               = float(getattr(args, "lateral_w_hyp",   0.05)),
        lateral_w_wave              = float(getattr(args, "lateral_w_wave",  0.05)),
        lateral_w_oja               = float(getattr(args, "lateral_w_oja",   0.0)),
        som_enabled                 = bool(getattr(args, "som_inhibition_schedules", False)),
        kernel_inhibition_sigma_start = float(getattr(args, "kernel_inhibition_sigma_start", 2.5)),
        kernel_inhibition_sigma_end   = float(getattr(args, "kernel_inhibition_sigma_end",   1.0)),
        use_inhibition_softmax      = bool(getattr(args, "inhibition_softmax", False)),
        inhibition_softmax_temp_start = float(getattr(args, "inhibition_softmax_temp_start", 1.5)),
        inhibition_softmax_temp_end   = float(getattr(args, "inhibition_softmax_temp_end",   0.45)),
    )

    model = VisNetUnified(
        device                      = str(device),
        num_classes                 = 10,
        spatial_size                = int(args.spatial_size),
        auto_resize_input           = not bool(getattr(args, "no_auto_resize", False)),
        coeffs                      = coeffs,
        dropout                     = float(getattr(args, "dropout",          0.01)),
        inhibition_decay            = float(getattr(args, "inhibition_decay", 1e-3)),
        wta_l1                      = 0.0 if no_inhibition else float(getattr(args, "wta_l1",   0.08)),
        wta_l234                    = 0.0 if no_inhibition else float(getattr(args, "wta_l234",  0.04)),
        rf_l1                       = int(getattr(args, "rf_l1", 15)),
        rf_l2                       = int(getattr(args, "rf_l2", 15)),
        rf_l3                       = int(getattr(args, "rf_l3", 15)),
        rf_l4                       = int(getattr(args, "rf_l4", 15)),
        recursive_iters             = int(getattr(args, "recursive_iters", 0)),
        use_dorsal_stream           = not bool(getattr(args, "no_dorsal_stream", False)),
        use_symmetry_gate_prior     = bool(getattr(args, "symmetry_gate_prior",     True)),
        symmetry_gate_alpha         = float(getattr(args, "symmetry_gate_alpha",    0.5)),
        symmetry_prior_unsup_lr     = float(getattr(args, "symmetry_prior_unsup_lr",0.0)),
        it_pp_cross_gate            = bool(getattr(args, "it_pp_cross_gate",        True)),
        it_pp_cross_pp_to_te        = float(getattr(args, "it_pp_cross_pp_to_te",   0.35)),
        it_pp_cross_te_to_pp        = float(getattr(args, "it_pp_cross_te_to_pp",   0.35)),
        it_pp_cross_iters           = int(getattr(args, "it_pp_cross_iters",        1)),
        use_neuron_glia             = bool(getattr(args, "neuron_glia",             True)),
        glia_state_dim              = int(getattr(args, "glia_state_dim",           8)),
        glia_ema                    = float(getattr(args, "glia_ema",               0.995)),
        glia_neuron_strength        = float(getattr(args, "glia_neuron_strength",   0.12)),
        glia_gate_min               = float(getattr(args, "glia_gate_min",          0.75)),
        glia_gate_max               = float(getattr(args, "glia_gate_max",          1.25)),
        # PFC flags — all disabled when disable_pfc=True
        use_pfc_hopfield            = False if disable_pfc else bool(getattr(args, "use_pfc_hopfield",          True)),
        pfc_hopfield_patterns       = int(getattr(args, "pfc_hopfield_patterns",    96)),
        pfc_hopfield_beta           = float(getattr(args, "pfc_hopfield_beta",      1.5)),
        pfc_hopfield_temperature    = float(getattr(args, "pfc_hopfield_temperature",1.5)),
        pfc_hopfield_blend          = float(getattr(args, "pfc_hopfield_blend",     5e-4)),
        pfc_hopfield_ema_lr         = float(getattr(args, "pfc_hopfield_ema_lr",    1e-3)),
        pfc_hopfield_unsup_update   = False if disable_pfc else bool(getattr(args, "pfc_hopfield_unsup_update", True)),
        pfc_hopfield_cosine         = bool(getattr(args, "pfc_hopfield_cosine",     True)),
        pfc_hopfield_soft_ema       = bool(getattr(args, "pfc_hopfield_soft_ema",   True)),
        pfc_hopfield_normalize_memory = bool(getattr(args, "pfc_hopfield_normalize_memory", False)),
        pfc_hopfield_sparsity       = float(getattr(args, "pfc_hopfield_sparsity",  0.91)),
        pfc_hopfield_sparse_update  = bool(getattr(args, "pfc_hopfield_sparse_update", True)),
        pfc_hopfield_layernorm      = bool(getattr(args, "pfc_hopfield_layernorm",  False)),
        pfc_mode                    = str(getattr(args, "pfc_mode",                 "hebbian_sa")),
        pfc_hebbian_lr              = float(getattr(args, "pfc_hebbian_lr",         1e-4)),
        pfc_hebbian_decay           = float(getattr(args, "pfc_hebbian_decay",      1e-5)),
        pfc_sa_head_dim             = int(getattr(args, "pfc_sa_head_dim",          0)),
        pfc_topdown_attention       = False if disable_pfc else bool(getattr(args, "pfc_topdown_attention",     True)),
        pfc_topdown_strength        = float(getattr(args, "pfc_topdown_strength",   0.22)),
        pfc_topdown_min_scale       = float(getattr(args, "pfc_topdown_min_scale",  0.7)),
        pfc_topdown_max_scale       = float(getattr(args, "pfc_topdown_max_scale",  1.2)),
        pfc_topdown_unsup_lr        = 0.0 if disable_pfc else float(getattr(args, "pfc_topdown_unsup_lr",      2e-3)),
        pfc_topdown_decay           = float(getattr(args, "pfc_topdown_decay",      5e-5)),
        pfc_topdown_per_neuron      = bool(getattr(args, "pfc_topdown_per_neuron",  True)),
        pfc_topdown_neuron_use_bias = bool(getattr(args, "pfc_topdown_neuron_bias", False)),
        pfc_topdown_shared_fe_blend = float(getattr(args, "pfc_topdown_shared_fe_blend", 0.35)),
        pfc_inhibition_feedback_unsup_lr = 0.0 if disable_pfc else float(getattr(args, "pfc_inhibition_feedback_unsup_lr", 1.25e-3)),
        pfc_inhibition_feedback_decay    = float(getattr(args, "pfc_inhibition_feedback_decay", 5e-5)),
        pfc_topdown_iters           = 1 if disable_pfc else int(getattr(args, "pfc_topdown_iters",             2)),
        pfc_predictive_feedback     = False if disable_pfc else bool(getattr(args, "pfc_predictive_feedback",   True)),
        pfc_predictive_strength     = float(getattr(args, "pfc_predictive_strength",0.15)),
        pfc_predictive_min_scale    = float(getattr(args, "pfc_predictive_min_scale",0.5)),
        pfc_predictive_max_scale    = float(getattr(args, "pfc_predictive_max_scale",1.3)),
        pfc_predictive_unsup_lr     = 0.0 if disable_pfc else float(getattr(args, "pfc_predictive_unsup_lr",   1.5e-3)),
        pfc_predictive_decay        = float(getattr(args, "pfc_predictive_decay",   5e-5)),
        pfc_l1_lambda               = float(getattr(args, "pfc_l1_lambda",          1.2e-4)),
        pfc_l1_prox_step            = float(getattr(args, "pfc_l1_prox_step",       1.5)),
        local_l1_lambda             = float(getattr(args, "local_l1_lambda",        1.2e-6)),
        local_l1_prox_step          = float(getattr(args, "local_l1_prox_step",     1.0)),
        local_l1_warmup_steps       = int(getattr(args, "local_l1_warmup_steps",    1000)),
        local_l1_apply_every        = int(getattr(args, "local_l1_apply_every",     2)),
        use_pfc_pc_layer_output_mask = False if disable_pfc else bool(getattr(args, "pfc_pc_layer_output_mask", True)),
        pfc_pc_layer_mask_lr        = 0.0 if disable_pfc else float(getattr(args, "pfc_pc_layer_mask_lr",      1e-3)),
        pfc_pc_layer_mask_decay     = float(getattr(args, "pfc_pc_layer_mask_decay",1e-4)),
        pfc_spatial_readout_gate    = False if disable_pfc else bool(getattr(args, "pfc_spatial_readout_gate",  True)),
        pfc_spatial_gate_strength   = float(getattr(args, "pfc_spatial_gate_strength",0.65)),
        pfc_spatial_gate_floor      = float(getattr(args, "pfc_spatial_gate_floor",  0.2)),
        pfc_recurrent_feedback_steps    = 1 if disable_pfc else int(getattr(args, "pfc_recurrent_feedback_steps",  2)),
        pfc_recurrent_feedback_strength = 0.0 if disable_pfc else float(getattr(args, "pfc_recurrent_feedback_strength", 0.35)),
        use_pfc_dense_feedback      = False if disable_pfc else bool(getattr(args, "pfc_dense_feedback",        True)),
        pfc_dense_feedback_strength = float(getattr(args, "pfc_dense_feedback_strength", 0.12)),
        use_pfc_deep_feedback       = False if disable_pfc else bool(getattr(args, "pfc_deep_feedback",         True)),
        pfc_deep_feedback_rank      = int(getattr(args, "pfc_deep_feedback_rank",   32)),
        pfc_deep_fb_strength_l2     = float(getattr(args, "pfc_deep_fb_l2",         0.06)),
        pfc_deep_fb_strength_l3     = float(getattr(args, "pfc_deep_fb_l3",         0.06)),
        pfc_deep_fb_strength_l4     = float(getattr(args, "pfc_deep_fb_l4",         0.08)),
        pfc_deep_fb_strength_mt     = float(getattr(args, "pfc_deep_fb_mt",         0.05)),
        pfc_deep_fb_strength_pp     = float(getattr(args, "pfc_deep_fb_pp",         0.05)),
        use_pfc_fusion_gate_unsup   = False if disable_pfc else bool(getattr(args, "pfc_fusion_gate_unsup",    True)),
        pfc_fusion_gate_unsup_lr    = 0.0 if disable_pfc else float(getattr(args, "pfc_fusion_gate_unsup_lr",  3e-3)),
        pfc_fusion_gate_decay       = float(getattr(args, "pfc_fusion_gate_decay",  1e-4)),
        pfc_fusion_lms_chunk_rows   = int(getattr(args, "pfc_fusion_lms_chunk_rows",256)),
        pfc_pre_hopfield_fusion     = str(getattr(args, "pfc_pre_hopfield_fusion",  "all")),
        pfc_pre_blend_w_te          = float(getattr(args, "pfc_pre_blend_w_te",     0.5)),
        pfc_pre_blend_w_pp          = float(getattr(args, "pfc_pre_blend_w_pp",     0.5)),
        pfc_post_readout_fusion     = str(getattr(args, "pfc_post_readout_fusion",  "all")),
    )

    # Force inhibition to zero in every layer when --inhibition-mode none
    if no_inhibition:
        for layer in model.iter_topographic_layers():
            layer.inhibition_strength         = 0.0
            layer.current_inhibition_strength = 0.0
            layer.wta_sparsity                = 0.0
            layer._wta_sparsity_base          = 0.0

    return model


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def average_all_weights(model: torch.nn.Module) -> None:
    if not dist.is_initialized(): return
    ws = dist.get_world_size()
    with torch.no_grad():
        for p in model.parameters():
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM); p.data /= ws
        for b in model.buffers():
            if b.is_floating_point():
                dist.all_reduce(b.data, op=dist.ReduceOp.SUM); b.data /= ws


# ---------------------------------------------------------------------------
# run_training / ddp_worker
# ---------------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> TrainingRunResult:
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = max(1, n_gpu)
    if world_size < 2:
        print(f"Running single-process (world_size={world_size}).", flush=True)
        return ddp_worker(0, 1, args)
    print(f"Spawning DDP across {world_size} GPUs...", flush=True)
    import torch.multiprocessing as mp
    mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
    return TrainingRunResult(
        best_val_acc=-1., best_epoch=-1, best_test_at_best_val=-1.,
        final_val_loss=-1., final_val_acc=-1., final_test_loss=-1., final_test_acc=-1.)


def get_all_hebbian_weights(model):
    weights = []
    for name, module in model.named_modules():
        if hasattr(module, "W"):
            weights.append((name, module.W))
    return weights

def ddp_worker(rank: int, world_size: int, args: argparse.Namespace) -> TrainingRunResult:
    # ── runtime ──────────────────────────────────────────────────────────────
    torch.backends.cudnn.enabled   = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    is_master = (rank == 0)
    if is_master:
        print(f"Running: {__file__}", flush=True)

    # ── data ─────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    data_root = os.path.expanduser(str(args.data_dir))
    if is_master: os.makedirs(data_root, exist_ok=True)
    if world_size > 1: dist.barrier()

    full_train = datasets.CIFAR10(root=data_root, train=True,  download=is_master, transform=transform)
    test_ds    = datasets.CIFAR10(root=data_root, train=False, download=is_master, transform=transform)
    if world_size > 1: dist.barrier()

    val_frac  = float(max(0., min(0.5, args.val_split)))
    n_total   = len(full_train)
    n_val     = int(round(val_frac * n_total))
    g         = torch.Generator().manual_seed(int(args.seed))
    perm      = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_total - n_val]
    val_idx   = perm[n_total - n_val:]
    if args.train_fraction < 1.0:
        train_idx = train_idx[:max(1, int(round(args.train_fraction * len(train_idx))))]

    train_ds = torch.utils.data.Subset(full_train, train_idx)
    val_ds   = torch.utils.data.Subset(
        datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform), val_idx)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True,  drop_last=True)
        val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)
        test_sampler  = DistributedSampler(test_ds,  world_size, rank, shuffle=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=2, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,  batch_size=args.test_batch_size,
                              sampler=val_sampler,   shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.test_batch_size,
                              sampler=test_sampler,  shuffle=False)

    # ── model ────────────────────────────────────────────────────────────────
    model = _build_model(args, device).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    raw_model: VisNetUnified = model.module if world_size > 1 else model  # type: ignore

    # ── optimiser ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        raw_model.classifier_parameters(),
        lr=float(args.clf_lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs),
        eta_min=max(1e-6, float(args.clf_lr) * 0.1))
    criterion = torch.nn.CrossEntropyLoss()

    # ── config ───────────────────────────────────────────────────────────────
    readout_type     = str(getattr(args, "readout_type",    "sgd")).lower()
    pinv_ridge       = float(getattr(args, "pinv_ridge",     1e-4))
    pinv_fit_every   = int(getattr(args, "pinv_fit_every",    1))
    pinv_max_batches = int(getattr(args, "pinv_max_batches", -1))
    freeze_after     = int(getattr(args, "freeze_local_plasticity_after_epoch", 0))

    best_val_acc = best_epoch = -1; best_val_acc = -1.
    best_test_at_best = final_val = final_test = -1.

    _pinv_done = False
    # ── epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        #local_plasticity_on = (freeze_after <= 0) or (epoch <= freeze_after)
        local_plasticity_on = (freeze_after < 0) or (epoch <= freeze_after)
        
        if is_master:
            print(f"\n--- Epoch {epoch}/{args.epochs}  "
                  f"[plasticity={'ON ' if local_plasticity_on else 'OFF'}  "
                  f"readout={readout_type}  "
                  f"inhibition={getattr(args,'inhibition_mode','kernel')}  "
                  f"pfc={'off' if getattr(args,'disable_pfc',False) else 'on'}] ---", flush=True)

        # ---- train ----------------------------------------------------------
        model.train()
        run_loss = run_correct = run_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_master)
        for i, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out  = model(x, local_update=local_plasticity_on)
            loss = criterion(out, y)

            use_pinv = (args.readout_type == "pinv")
            if not use_pinv:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(raw_model.classifier_parameters()), 1.0)
                optimizer.step()

            if world_size > 1: average_all_weights(model)
            run_loss    += loss.item()
            run_correct += out.argmax(1).eq(y).sum().item()
            run_total   += y.size(0)
            if is_master:
                pbar.set_postfix({"loss": f"{run_loss/(i+1):.4f}",
                                  "acc":  f"{100.*run_correct/run_total:.2f}%"})
        if is_master:
            print(f"✅ Train | Loss: {run_loss/len(train_loader):.4f} "
                  f"| Acc: {100.*run_correct/run_total:.2f}%", flush=True)

        # ---- pseudo-inverse readout -----------------------------------------
        use_pinv = (readout_type == "pinv")

        # ---- train (skip batch loop only if pinv + frozen) ----
        if not (use_pinv and not local_plasticity_on):
            model.train()
            run_loss = run_correct = run_total = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_master)
            for i, (x, y) in enumerate(pbar):
                ...
        else:
            if is_master:
                print("  [pinv] Skipping SGD train loop (frozen features).", flush=True)

        # ---- pinv fit (MUST run regardless of the skip above) ----
        if use_pinv and epoch > args.freeze_local_plasticity_after_epoch and not _pinv_done:
            if is_master:
                print("  [pinv] Fitting (once, on frozen features)...", flush=True)
                full_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)
                raw_model.fit_pseudoinverse_classifier(full_loader, device,
                                                       ridge=pinv_ridge, max_batches=pinv_max_batches)
            _pinv_done = True

        # ---- validation -----------------------------------------------------
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Val", leave=False, disable=not is_master):
                x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                vc += model(x, local_update=False).argmax(1).eq(y).sum().item(); vt += y.size(0)
        if world_size > 1:
            vct = torch.tensor(vc, device=device); vtt = torch.tensor(vt, device=device)
            dist.all_reduce(vct, op=dist.ReduceOp.SUM); dist.all_reduce(vtt, op=dist.ReduceOp.SUM)
            val_acc = vct.item() / vtt.item()
        else:
            val_acc = vc / max(1, vt)
        final_val = val_acc
        if is_master:
            print(f"✅ Val Acc: {val_acc:.4f}", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc; best_epoch = epoch
                if getattr(args, "save_best", False):
                    torch.save(raw_model.state_dict(), args.save_best_path)
                    print("  💾 Saved best checkpoint.", flush=True)

        # ---- test (every 10 epochs or last) ---------------------------------
        if (epoch % 10 == 0) or (epoch == args.epochs):
            model.eval()
            tc = tt = 0
            with torch.no_grad():
                for x, y in tqdm(test_loader, desc="Test", leave=False, disable=not is_master):
                    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                    tc += model(x, local_update=False).argmax(1).eq(y).sum().item(); tt += y.size(0)
            if world_size > 1:
                tct = torch.tensor(tc, device=device); ttt = torch.tensor(tt, device=device)
                dist.all_reduce(tct, op=dist.ReduceOp.SUM); dist.all_reduce(ttt, op=dist.ReduceOp.SUM)
                test_acc = tct.item() / ttt.item()
            else:
                test_acc = tc / max(1, tt)
            final_test = test_acc
            if val_acc >= best_val_acc: best_test_at_best = test_acc
            if is_master:
                print(f"🧪 Test Acc (epoch {epoch}): {test_acc:.4f}", flush=True)


        # ── 1. INITIALIZE TRACKING STATE OUTSIDE THE LOOP ──────────────────────────
        W_before_list = None  # <-- Explicitly placed BEFORE entering the loop
    
        # ── 3. CAPTURE BASELINE SNAPSHOT WHEN PLASTICITY TURNS OFF ────────────
        local_plasticity_on = (args.freeze_local_plasticity_after_epoch <= 0) or (epoch <= args.freeze_local_plasticity_after_epoch)
        
        if not local_plasticity_on and W_before_list is None:
            with torch.no_grad():
                raw_model = model.module if world_size > 1 else model
                W_before_list = [
                    (name, W.clone().detach())
                    for name, W in get_all_hebbian_weights(raw_model)
                ]
            if is_master:
                print(f"\n[CHECK] Epoch {epoch}: Plasticity transitioned to OFF. Stored baseline feature weights.", flush=True)

        # ── 4. VERIFY DRIFT VIA CONTEXT GUARANTEES ────────────────────────────
        # This will now run perfectly without throwing an UnboundLocalError!
        if W_before_list is not None and epoch > args.freeze_local_plasticity_after_epoch + 1:
            with torch.no_grad():
                raw_model = model.module if world_size > 1 else model
                current_weights = get_all_hebbian_weights(raw_model)
                
                total_drift = 0.0
                significant_drift_detected = False
                
                for (name, W_before), (_, W_after) in zip(W_before_list, current_weights):
                    diff = (W_after - W_before).abs().mean()
                    total_drift += diff.item()
                    
                    if diff.item() > 0.001:
                        significant_drift_detected = True
                        if is_master:
                            print(f"[CHECK] Warning: {name} parameter drift detected! Value: {diff.item():.6e}", flush=True)
                
                if not significant_drift_detected and is_master:
                    print(f"✅ No significant drift of frozen features at epoch {epoch} (Total mean drift: {total_drift:.6e})", flush=True)




    if world_size > 1: dist.destroy_process_group()

    return TrainingRunResult(
        best_val_acc=best_val_acc, best_epoch=best_epoch,
        best_test_at_best_val=best_test_at_best,
        final_val_loss=-1., final_val_acc=final_val,
        final_test_loss=-1., final_test_acc=final_test)


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisNet Unified - CIFAR-10")

    def add_bool(name, default, help_text):
        dest = name.replace("-", "_")
        g = parser.add_mutually_exclusive_group()
        g.add_argument(f"--{name}",    dest=dest, action="store_true",  help=help_text)
        g.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable: {help_text}")
        parser.set_defaults(**{dest: default})

    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--train-fraction",   type=float, default=1.0)
    parser.add_argument("--batch-size",       type=int,   default=16)
    parser.add_argument("--test-batch-size",  type=int,   default=4)
    parser.add_argument("--spatial-size",     type=int,   default=32)
    parser.add_argument("--data-dir",         type=str,   default="./data")
    parser.add_argument("--val-split",        type=float, default=0.02)
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--clf-lr",           type=float, default=5e-4)
    parser.add_argument("--weight-decay",     type=float, default=3e-4)
    parser.add_argument("--dropout",          type=float, default=0.01)
    parser.add_argument("--lambda-d",         type=float, default=1e-4)
    parser.add_argument("--rf-l1",            type=int,   default=15)
    parser.add_argument("--rf-l2",            type=int,   default=15)
    parser.add_argument("--rf-l3",            type=int,   default=15)
    parser.add_argument("--rf-l4",            type=int,   default=15)
    parser.add_argument("--recursive-iters",  type=int,   default=0)
    parser.add_argument("--freeze-local-plasticity-after-epoch", type=int, default=0)
    parser.add_argument("--inhibition-decay", type=float, default=1e-3)
    parser.add_argument("--inhibition-mode",  type=str,   default="kernel",
                        choices=["global","kernel","averaging","mixed","mexican","predictive","none"])
    parser.add_argument("--wta-l1",           type=float, default=0.08)
    parser.add_argument("--wta-l234",         type=float, default=0.04)
    add_bool("adaptive-inhibition",         False, "Adapt inhibition strength online")
    add_bool("use-homeostatic-threshold",   False, "Per-neuron homeostatic threshold for WTA")
    parser.add_argument("--homeostatic-threshold-lr",         type=float, default=0.02)
    parser.add_argument("--homeostatic-threshold-theta-init", type=float, default=0.0)
    parser.add_argument("--homeostatic-threshold-min",        type=float, default=-2.0)
    parser.add_argument("--homeostatic-threshold-max",        type=float, default=2.0)
    parser.add_argument("--target-active-frac",   type=float, default=0.03)
    parser.add_argument("--inhibition-adapt-lr",  type=float, default=0.002)
    parser.add_argument("--inhibition-min",        type=float, default=0.0)
    parser.add_argument("--inhibition-max",        type=float, default=1.0)
    parser.add_argument("--inhibition-learning-sparsity", type=float, default=0.5)
    parser.add_argument("--kernel-inhibition-size",  type=int,   default=7)
    parser.add_argument("--kernel-inhibition-sigma", type=float, default=1.5)
    parser.add_argument("--averaging-inhibition-size", type=int, default=7)
    parser.add_argument("--inhibition-dropout",   type=float, default=0.01)
    add_bool("soft-inhibition",             True,  "Signed soft-threshold on inhibition")
    add_bool("inhibition-no-threshold",     True,  "Raw linear inhibition drives")
    parser.add_argument("--inhibition-smooth-scale", type=float, default=0.0)
    add_bool("divisive-inhibition-norm",    True,  "Divisive normalization")
    parser.add_argument("--divisive-mode",        type=str,   default="both",
                        choices=["global","local","both","all"])
    parser.add_argument("--divisive-alpha",       type=float, default=0.85)
    parser.add_argument("--divisive-beta",        type=float, default=0.36)
    parser.add_argument("--divisive-w-global",    type=float, default=0.5)
    parser.add_argument("--divisive-w-local",     type=float, default=0.5)
    parser.add_argument("--divisive-local-size",  type=int,   default=9)
    parser.add_argument("--divisive-local-sigma", type=float, default=2.0)
    add_bool("oscillatory-inhibition",      False, "Phase-based oscillatory gating")
    parser.add_argument("--phase-period",         type=int,   default=10)
    parser.add_argument("--phase-gate-sharpness", type=float, default=1.0)
    add_bool("cascade-skip-connections",    True,  "Skip connections L2->L4")
    add_bool("no-dorsal-stream",            False, "Disable dorsal MT/PP path")
    add_bool("neuron-glia",                 True,  "Slow glial homeostasis")
    parser.add_argument("--glia-state-dim",      type=int,   default=8)
    parser.add_argument("--glia-ema",            type=float, default=0.995)
    parser.add_argument("--glia-neuron-strength",type=float, default=0.12)
    parser.add_argument("--glia-gate-min",       type=float, default=0.75)
    parser.add_argument("--glia-gate-max",       type=float, default=1.25)
    add_bool("structural-plasticity",       True,  "Prune/grow synapses periodically")
    parser.add_argument("--structural-update-freq",      type=int,   default=1200)
    parser.add_argument("--structural-prune-threshold",  type=float, default=1e-4)
    parser.add_argument("--structural-prune-max-frac",   type=float, default=0.002)
    parser.add_argument("--structural-grow-threshold",   type=float, default=0.02)
    parser.add_argument("--structural-grow-max-frac",    type=float, default=0.001)
    parser.add_argument("--structural-grow-init-scale",  type=float, default=0.02)
    parser.add_argument("--holo-alpha",          type=float, default=0.15)
    parser.add_argument("--holo-update-freq",    type=int,   default=20)
    parser.add_argument("--holo-fast-lr",        type=float, default=0.1)
    parser.add_argument("--holo-fast-decay",     type=float, default=0.1)
    parser.add_argument("--holo-corr-blend",     type=float, default=0.2)
    parser.add_argument("--holo-fast-norm-cap",  type=float, default=12.0)
    parser.add_argument("--gamma-wavelet",       type=float, default=0.01)
    parser.add_argument("--wavelet-threshold",   type=float, default=0.02)
    parser.add_argument("--no-wavelet-denoise",  action="store_true")
    parser.add_argument("--no-wavelet-binding",  action="store_true")
    add_bool("wavelet-input",               True,  "Concat Haar wavelet to Gabor")
    parser.add_argument("--lateral-w-hebb", type=float, default=1.0)
    parser.add_argument("--lateral-w-anti", type=float, default=0.0)
    parser.add_argument("--lateral-w-cov",  type=float, default=0.0)
    parser.add_argument("--lateral-w-holo", type=float, default=0.05)
    parser.add_argument("--lateral-w-hyp",  type=float, default=0.05)
    parser.add_argument("--lateral-w-wave", type=float, default=0.05)
    parser.add_argument("--lateral-w-oja",  type=float, default=0.0)
    parser.add_argument("--unsup-mix-mode", type=str,   default="adaptive",
                        choices=["fixed","adaptive","random"])
    parser.add_argument("--no-unsup-mix-normalize-terms", action="store_true")
    parser.add_argument("--unsup-mix-temperature",  type=float, default=1.0)
    parser.add_argument("--unsup-mix-random-alpha", type=float, default=20.0)
    parser.add_argument("--unsup-mix-w-hebb",  type=float, default=0.30)
    parser.add_argument("--unsup-mix-w-holo",  type=float, default=0.10)
    parser.add_argument("--unsup-mix-w-hyp",   type=float, default=0.10)
    parser.add_argument("--unsup-mix-w-wave",  type=float, default=0.15)
    parser.add_argument("--unsup-mix-w-anti",  type=float, default=0.10)
    parser.add_argument("--unsup-mix-w-cons",  type=float, default=0.08)
    parser.add_argument("--unsup-mix-w-rec",   type=float, default=0.08)
    parser.add_argument("--unsup-mix-w-free",  type=float, default=0.06)
    parser.add_argument("--unsup-mix-w-decay", type=float, default=0.03)
    parser.add_argument("--unsup-mix-w-dist",  type=float, default=0.00)
    add_bool("pc-per-neuron-plasticity",    True,  "Per-neuron PC gain")
    parser.add_argument("--pc-per-neuron-layer-weight", type=float, default=0.34)
    parser.add_argument("--pc-per-neuron-pfc-weight",   type=float, default=0.33)
    parser.add_argument("--pc-per-neuron-trace-weight", type=float, default=0.33)
    parser.add_argument("--pc-per-neuron-gain-min",     type=float, default=0.5)
    parser.add_argument("--pc-per-neuron-gain-max",     type=float, default=1.5)
    parser.add_argument("--local-l1-lambda",       type=float, default=1.2e-6)
    parser.add_argument("--local-l1-prox-step",    type=float, default=1.0)
    parser.add_argument("--local-l1-warmup-steps", type=int,   default=1000)
    parser.add_argument("--local-l1-apply-every",  type=int,   default=2)
    # PFC args
    add_bool("disable-pfc",                 False, "Disable all PFC/predictive-coding modules")
    add_bool("use-pfc-hopfield",            True,  "Modern Hopfield PFC block")
    parser.add_argument("--pfc-hopfield-patterns",    type=int,   default=96)
    parser.add_argument("--pfc-hopfield-beta",        type=float, default=1.5)
    parser.add_argument("--pfc-hopfield-temperature", type=float, default=1.5)
    parser.add_argument("--pfc-hopfield-blend",       type=float, default=5e-4)
    parser.add_argument("--pfc-hopfield-ema-lr",      type=float, default=1e-3)
    add_bool("pfc-hopfield-unsup-update",   True,  "Unsupervised Hopfield EMA updates")
    add_bool("pfc-hopfield-cosine",         True,  "Cosine similarity in Hopfield")
    add_bool("pfc-hopfield-soft-ema",       True,  "Soft EMA update for Hopfield")
    add_bool("pfc-hopfield-normalize-memory", False, "L2-normalize Hopfield memory")
    parser.add_argument("--pfc-hopfield-sparsity",    type=float, default=0.91)
    add_bool("pfc-hopfield-sparse-update",  True,  "Sparse Hopfield memory update")
    add_bool("pfc-hopfield-layernorm",      False, "LayerNorm on Hopfield output")
    parser.add_argument("--pfc-mode",             type=str,   default="hebbian_sa",
                        choices=["hopfield","hebbian_sa"])
    parser.add_argument("--pfc-hebbian-lr",       type=float, default=1e-4)
    parser.add_argument("--pfc-hebbian-decay",    type=float, default=1e-5)
    parser.add_argument("--pfc-sa-head-dim",      type=int,   default=0)
    add_bool("pfc-topdown-attention",       True,  "PFC top-down attention on L1-L4")
    parser.add_argument("--pfc-topdown-strength",    type=float, default=0.22)
    parser.add_argument("--pfc-topdown-min-scale",   type=float, default=0.7)
    parser.add_argument("--pfc-topdown-max-scale",   type=float, default=1.2)
    parser.add_argument("--pfc-topdown-unsup-lr",    type=float, default=2e-3)
    parser.add_argument("--pfc-topdown-decay",       type=float, default=5e-5)
    add_bool("pfc-topdown-per-neuron",      True,  "Per-neuron PFC top-down weights")
    add_bool("pfc-topdown-neuron-bias",     False, "Bias on per-neuron top-down weights")
    parser.add_argument("--pfc-topdown-shared-fe-blend", type=float, default=0.35)
    parser.add_argument("--pfc-inhibition-feedback-unsup-lr", type=float, default=1.25e-3)
    parser.add_argument("--pfc-inhibition-feedback-decay",    type=float, default=5e-5)
    parser.add_argument("--pfc-topdown-iters",    type=int,   default=2)
    add_bool("pfc-predictive-feedback",     True,  "PFC predictive-coding feedback")
    parser.add_argument("--pfc-predictive-strength",    type=float, default=0.15)
    parser.add_argument("--pfc-predictive-min-scale",   type=float, default=0.5)
    parser.add_argument("--pfc-predictive-max-scale",   type=float, default=1.3)
    parser.add_argument("--pfc-predictive-unsup-lr",    type=float, default=1.5e-3)
    parser.add_argument("--pfc-predictive-decay",       type=float, default=5e-5)
    parser.add_argument("--pfc-l1-lambda",  type=float, default=1.2e-4)
    parser.add_argument("--pfc-l1-prox-step", type=float, default=1.5)
    add_bool("pfc-pc-layer-output-mask",    True,  "PC-LMS masks on ventral layer flats")
    parser.add_argument("--pfc-pc-layer-mask-lr",    type=float, default=1e-3)
    parser.add_argument("--pfc-pc-layer-mask-decay", type=float, default=1e-4)
    add_bool("pfc-spatial-readout-gate",    True,  "Spatial saliency gate at readout")
    parser.add_argument("--pfc-spatial-gate-strength", type=float, default=0.65)
    parser.add_argument("--pfc-spatial-gate-floor",    type=float, default=0.2)
    parser.add_argument("--pfc-recurrent-feedback-steps",    type=int,   default=2,
                        choices=[1, 2])
    parser.add_argument("--pfc-recurrent-feedback-strength", type=float, default=0.35)
    add_bool("pfc-dense-feedback",          True,  "Dense PFC->L1 additive feedback")
    parser.add_argument("--pfc-dense-feedback-strength", type=float, default=0.12)
    add_bool("pfc-deep-feedback",           True,  "Bottleneck PFC->L2/L3/L4/MT/PP")
    parser.add_argument("--pfc-deep-feedback-rank", type=int,   default=32)
    parser.add_argument("--pfc-deep-fb-l2", type=float, default=0.06)
    parser.add_argument("--pfc-deep-fb-l3", type=float, default=0.06)
    parser.add_argument("--pfc-deep-fb-l4", type=float, default=0.08)
    parser.add_argument("--pfc-deep-fb-mt", type=float, default=0.05)
    parser.add_argument("--pfc-deep-fb-pp", type=float, default=0.05)
    add_bool("symmetry-gate-prior",         True,  "Vertical symmetry prior on dorsal gate")
    parser.add_argument("--symmetry-gate-alpha",      type=float, default=0.5)
    parser.add_argument("--symmetry-prior-unsup-lr",  type=float, default=0.0)
    add_bool("it-pp-cross-gate",            True,  "Bidirectional IT<->PP cross-gate")
    parser.add_argument("--it-pp-cross-pp-to-te", type=float, default=0.35)
    parser.add_argument("--it-pp-cross-te-to-pp", type=float, default=0.35)
    parser.add_argument("--it-pp-cross-iters",    type=int,   default=1)
    parser.add_argument("--pfc-pre-hopfield-fusion", type=str, default="all",
                        choices=["pp_only","blend","gate","all"])
    parser.add_argument("--pfc-pre-blend-w-te",   type=float, default=0.5)
    parser.add_argument("--pfc-pre-blend-w-pp",   type=float, default=0.5)
    parser.add_argument("--pfc-post-readout-fusion", type=str, default="all",
                        choices=["concat","gate","all"])
    add_bool("pfc-fusion-gate-unsup",       True,  "Unsupervised LMS fusion gate updates")
    parser.add_argument("--pfc-fusion-gate-unsup-lr",  type=float, default=3e-3)
    parser.add_argument("--pfc-fusion-gate-decay",     type=float, default=1e-4)
    parser.add_argument("--pfc-fusion-lms-chunk-rows", type=int,   default=256)
    # readout
    parser.add_argument("--readout-type",   type=str,   default="sgd",
                        choices=["sgd","pinv"])
    parser.add_argument("--pinv-ridge",       type=float, default=1e-4)
    parser.add_argument("--pinv-fit-every",   type=int,   default=1)
    parser.add_argument("--pinv-max-batches", type=int,   default=-1)
    # misc
    parser.add_argument("--save-best",      action="store_true", default=False)
    parser.add_argument("--save-best-path", type=str, default="best_model.pth")
    parser.add_argument("--no-auto-resize", action="store_true", default=False)
    parser.add_argument("--no-eval-test-on-best-val", action="store_true", default=False)
    add_bool("som-inhibition-schedules",    False, "SOM-style inhibition schedules")
    add_bool("inhibition-softmax",          False, "Softmax spatial competition")
    parser.add_argument("--kernel-inhibition-sigma-start", type=float, default=2.5)
    parser.add_argument("--kernel-inhibition-sigma-end",   type=float, default=1.0)

    return parser


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> TrainingRunResult:
    parser = build_arg_parser()
    args   = parser.parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
    
    
    
    
    
    
    
    
    

#python BioPlasticNet3_pinv5.py   --epochs 30   --train-fraction 0.05   --batch-size 16   --test-batch-size 16   --spatial-size 32   --freeze-local-plasticity-after-epoch 2   --i
#nhibition-mode none   --disable-pfc  --readout-type pinv
