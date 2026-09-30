from __future__ import annotations
import os
import random
from collections import deque
from typing import Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
DEFAULT_DATA_DIR = 'datasets/MTS/PSM/AllInOne'
WINDOW = 96
EXPERT_KERNEL_SIZES = [3, 7, 15, 31]
K = 4
BATCH = 256
EPOCHS = 20
LR = 0.001
EPS = 1e-08
USE_FREQ_SCORE = False
LAMBDA_FREQ_SCORE = 0.5
FREQ_EXCLUDE_DC = True
FREQ_USE_LOG_MAG = True
FREQ_BATCH = 2048
USE_SSL = False
LAMBDA_SSL = 0.1
SSL_MASK_RATIO = 0.2
THRESHOLD_QUANTILE = 0.995
THRESHOLD_METHOD = 'quantile'
MAD_K = 4.0
STD_K = 3.0
USE_FIXED_THRESHOLD = True
REF_BUF = 5000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'[Seed] {seed} (deterministic cudnn=True)')

def load_data(data_dir: Optional[str]=None):
    root = os.path.normpath(data_dir if data_dir is not None else DEFAULT_DATA_DIR)
    paths = {'train': os.path.join(root, 'train.npy'), 'train_label': os.path.join(root, 'train_label.npy'), 'test': os.path.join(root, 'test.npy'), 'test_label': os.path.join(root, 'test_label.npy')}
    for name, p in paths.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f'Not found: {p}\nExpected KAN-AD layout under data_dir, e.g. datasets/MTS/PSM/AllInOne/')
    X_train = np.load(paths['train']).astype(np.float32)
    y_train = np.load(paths['train_label']).astype(np.int64)
    X_test = np.load(paths['test']).astype(np.float32)
    y_test = np.load(paths['test_label']).astype(np.int64)
    if X_train.ndim != 2:
        raise ValueError('train.npy must be 2D: (T, C)')
    for tag, arr in (('train', X_train), ('test', X_test)):
        n_bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
        if n_bad > 0:
            print(f'[WARN] {tag} has {n_bad} NaN/Inf -> 0.0')
            if tag == 'train':
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    print(f'[Data] {root} | C={X_train.shape[1]}')
    return (X_train, y_train, X_test, y_test)

def _covariance(Z: torch.Tensor, eps: float=1e-06) -> Tuple[torch.Tensor, torch.Tensor]:
    B, D = Z.shape
    mu = Z.mean(dim=0, keepdim=True)
    X = Z - mu
    cov = X.T @ X / (B + 1e-12)
    cov = 0.5 * (cov + cov.T)
    cov = cov + eps * torch.eye(D, device=Z.device, dtype=Z.dtype)
    return (mu.squeeze(0), cov)

def _sqrtm_spd(A: torch.Tensor, eps: float=0.0001, n_iters: int=15) -> torch.Tensor:
    A = 0.5 * (A + A.T) + eps * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    D = A.shape[0]
    scale = (torch.norm(A, p='fro') / D ** 0.5 + eps).clamp(min=eps)
    M = A / scale
    Y = M.clone()
    Z = torch.eye(D, device=A.device, dtype=A.dtype)
    I = Z.clone()
    for _ in range(n_iters):
        ZY = Z @ Y
        Y = 0.5 * Y @ (3.0 * I - ZY)
        Z = 0.5 * (3.0 * I - ZY) @ Z
    out = Y * scale ** 0.5
    if not torch.isfinite(out).all():
        out = scale ** 0.5 * I
    return out

def bures_wasserstein_gaussian(mu1: torch.Tensor, Sigma1: torch.Tensor, mu2: torch.Tensor, Sigma2: torch.Tensor, eps: float=0.0001) -> torch.Tensor:
    mean_term = torch.sum((mu1 - mu2) ** 2)
    S1_sqrt = _sqrtm_spd(Sigma1, eps=eps)
    middle = S1_sqrt @ Sigma2 @ S1_sqrt
    middle = 0.5 * (middle + middle.T) + eps * torch.eye(middle.shape[0], device=middle.device, dtype=middle.dtype)
    middle_sqrt = _sqrtm_spd(middle, eps=eps)
    trace_term = torch.trace(Sigma1) + torch.trace(Sigma2) - 2.0 * torch.trace(middle_sqrt)
    return torch.clamp(mean_term + trace_term, min=0.0)

class OTAdaptiveThreshold:

    def __init__(self, ref_scores: np.ndarray, quantile: float, ref_buf: int):
        ref_scores = np.asarray(ref_scores, dtype=np.float64)
        if len(ref_scores) < 100:
            raise ValueError('Need enough reference scores to initialize threshold.')
        self.quantile = float(quantile)
        self.ref = deque(ref_scores[-ref_buf:].tolist(), maxlen=ref_buf)
        self.tau = float(np.quantile(np.array(self.ref), self.quantile))

    def step(self, cur_scores: np.ndarray, delta: float=0.15, guard: float=0.85):
        if USE_FIXED_THRESHOLD:
            return self.tau
        cur_scores = np.asarray(cur_scores, dtype=np.float64)
        if len(cur_scores) < 50:
            return self.tau
        ref_arr = np.array(self.ref, dtype=np.float64)
        d = wasserstein_distance(ref_arr, cur_scores)
        if d > delta and (cur_scores < self.tau).mean() >= guard:
            cur_sorted = np.sort(cur_scores)
            keep = max(10, int(len(cur_sorted) * 0.85))
            for s in cur_sorted[:keep]:
                self.ref.append(float(s))
            self.tau = float(np.quantile(np.array(self.ref), self.quantile))
        return self.tau

class FixedThreshold:

    def __init__(self, tau: float):
        self.tau = float(tau)

    def step(self, cur_scores: np.ndarray, delta: float=0.15, guard: float=0.85):
        return self.tau

def build_threshold(ref_scores: np.ndarray, method: str=THRESHOLD_METHOD, quantile: float=THRESHOLD_QUANTILE, mad_k: float=MAD_K, std_k: float=STD_K, ref_buf: int=REF_BUF) -> tuple:
    ref_scores = np.asarray(ref_scores, dtype=np.float64)
    if ref_scores.size == 0:
        raise ValueError('ref_scores is empty')
    method = str(method).lower()
    if method == 'quantile':
        thr = OTAdaptiveThreshold(ref_scores=ref_scores, quantile=quantile, ref_buf=ref_buf)
        mode = 'fixed' if USE_FIXED_THRESHOLD else 'adaptive'
        info = f'Threshold[{method}] {mode} | tau (q={quantile:.3f}) = {thr.tau:.4f}'
        return (thr, info)
    if method == 'mad':
        med = float(np.median(ref_scores))
        mad = float(np.median(np.abs(ref_scores - med)))
        tau = med + float(mad_k) * 1.4826 * mad
        thr = FixedThreshold(tau=tau)
        return (thr, f'Threshold[mad] tau={thr.tau:.4f}')
    if method == 'std':
        mu = float(np.mean(ref_scores))
        sd = float(np.std(ref_scores) + 1e-12)
        thr = FixedThreshold(tau=mu + float(std_k) * sd)
        return (thr, f'Threshold[std] tau={thr.tau:.4f}')
    raise ValueError(f'Unknown threshold method: {method}')

def _fft_features_batch(Xw: np.ndarray) -> np.ndarray:
    Xw = np.asarray(Xw, dtype=np.float64)
    mag = np.abs(np.fft.rfft(Xw, axis=1))
    if FREQ_USE_LOG_MAG:
        mag = np.log1p(mag)
    return mag

def build_fft_reference(Xw_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, W, C = Xw_ref.shape
    F = W // 2 + 1
    freq_idx = np.arange(F, dtype=np.int64)
    if FREQ_EXCLUDE_DC and F > 1:
        freq_idx = freq_idx[1:]
    sum_ = np.zeros((len(freq_idx), C), dtype=np.float64)
    sumsq = np.zeros((len(freq_idx), C), dtype=np.float64)
    n_total = 0
    for s in range(0, N, FREQ_BATCH):
        mag = _fft_features_batch(Xw_ref[s:s + FREQ_BATCH])[:, freq_idx, :]
        sum_ += mag.sum(axis=0)
        sumsq += (mag ** 2).sum(axis=0)
        n_total += mag.shape[0]
    mu = sum_ / max(n_total, 1)
    var = np.maximum(sumsq / max(n_total, 1) - mu ** 2, 0.0)
    return (mu.astype(np.float64), np.sqrt(var) + 1e-08, freq_idx)

def fft_deviation_channel_scores(Xw: np.ndarray, mu: np.ndarray, std: np.ndarray, freq_idx: np.ndarray) -> np.ndarray:
    N, _, C = Xw.shape
    out = np.zeros((N, C), dtype=np.float64)
    for s in range(0, N, FREQ_BATCH):
        mag = _fft_features_batch(Xw[s:s + FREQ_BATCH])[:, freq_idx, :]
        z = np.abs((mag - mu[None, :, :]) / std[None, :, :])
        out[s:s + mag.shape[0]] = z.mean(axis=1)
    return out

def run_kanad_evals(point_scores: np.ndarray, point_labels: np.ndarray):
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointAuprcPA, PointF1PA, PointKthF1PA
    y_true = np.asarray(point_labels, dtype=np.float64)
    margins = [0]
    results = {}
    m = PointF1PA().calc(point_scores, y_true, margins)
    results['best f1 under pa'] = {'f1': m.f1, 'precision': m.p, 'recall': m.r}
    m = EventF1PA(mode='squeeze').calc(point_scores, y_true, margins)
    results['event-based f1 under pa with mode squeeze'] = {'f1': m.f1, 'precision': m.p, 'recall': m.r}
    m = PointKthF1PA(k=5).calc(point_scores, y_true, margins)
    results['best f1 under 5-delay pa'] = {'f1': m.f1, 'precision': m.p, 'recall': m.r}
    m = PointAuprcPA().calc(point_scores, y_true, margins)
    results['point-based auprc pa'] = m.value
    return results

def run_catch_evals(point_scores: np.ndarray, point_preds: np.ndarray, point_labels: np.ndarray) -> dict:
    from ts_benchmark.evaluation.metrics import classification_metrics_label, classification_metrics_score
    y_true = np.asarray(point_labels, dtype=np.float64)
    y_score = np.asarray(point_scores, dtype=np.float64)
    y_pred = np.asarray(point_preds, dtype=np.float64)
    out_score, out_label = ({}, {})

    def _safe(fn, a, b):
        try:
            return float(fn(a, b))
        except Exception:
            return float('nan')
    for name in classification_metrics_score.__all__:
        out_score[name] = _safe(getattr(classification_metrics_score, name), y_true, y_score)
    for name in classification_metrics_label.__all__:
        out_label[name] = _safe(getattr(classification_metrics_label, name), y_true, y_pred)
    return {'score': out_score, 'label': out_label}

def plot_anomaly_result(X_test: np.ndarray, y_test: np.ndarray, point_scores: np.ndarray, save_path: str='anomaly_result.png', max_points: Optional[int]=None, channel: Optional[int]=None, point_taus: Optional[np.ndarray]=None) -> None:
    T, _ = X_test.shape
    signal = np.mean(X_test, axis=1) if channel is None else X_test[:, channel]
    if max_points is None or max_points >= T:
        t_start, t_end = (0, T)
    else:
        ano_idx = np.where(y_test > 0.5)[0]
        if len(ano_idx) == 0:
            t_start, t_end = (0, max_points)
        else:
            first_ano = int(ano_idx[0])
            half = max_points // 2
            t_start = max(0, first_ano - half // 2)
            t_end = min(T, t_start + max_points)
    t = np.arange(t_start, t_end)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6.8), sharex=True)
    ax1.plot(t, signal[t_start:t_end], color='steelblue', linewidth=0.6)
    ax2.plot(t, point_scores[t_start:t_end], color='darkgreen', linewidth=0.8, label='score')
    if point_taus is not None:
        ax2.plot(t, point_taus[t_start:t_end], '--', color='orangered', label='tau')
    ax1.set_title('(a) Signal')
    ax2.set_title('(b) Anomaly score')
    ax2.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot] {save_path}')