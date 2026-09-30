from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils as base
DEVICE = base.DEVICE
WINDOW = base.WINDOW
TARGET_LEN = WINDOW
HORIZON = TARGET_LEN
K = base.K
BATCH = base.BATCH
EPOCHS = base.EPOCHS
LR = base.LR
EPS = base.EPS
PAPER_ENCODER_KERNEL_SIZES = [3, 7, 15, 31]
DEFAULT_ENCODER_MODE = 'attn'  # 'default' | 'sharp' | 'attn'
DEFAULT_SCORE_AGGREGATION = 'zsum'  # sum | max | topk_mean | zsum | zmax | ztopk_mean
DEFAULT_POINT_SCORE_ALIGN = 'step_ovl_mean'
DEFAULT_SCORE_TOPK = 3
STEP_ALIGN_SCHEMES = frozenset({
    'step_ovl_mean', 'step_ovl_max', 'step_ovl_wfront', 'step_ovl_wback', 'step_paper',
})
LAMBDA_GRAD = 0.0  # >0 weights first-difference forecast error (sharp mode)
USE_FREQ_SCORE = base.USE_FREQ_SCORE
LAMBDA_FREQ_SCORE = base.LAMBDA_FREQ_SCORE
LAMBDA_LB_PAPER = 0.3
LAMBDA_DIST_PAPER = 0.1
BW_COV_EPS = 0.01
BW_SQRT_EPS = 0.0001
LAMBDA_S = 0.5
LAMBDA_ES = 0.0  # interaction e·s disabled; score = e + λ_s·s
ROUTING_REF_NORMAL_ONLY = True
USE_SSL = base.USE_SSL
LAMBDA_SSL = base.LAMBDA_SSL
SSL_MASK_RATIO = base.SSL_MASK_RATIO
load_data = base.load_data
set_seed = base.set_seed
run_kanad_evals = base.run_kanad_evals
run_catch_evals = base.run_catch_evals
plot_anomaly_result = base.plot_anomaly_result
build_threshold = base.build_threshold
PAPER_SCORE_METRIC_ALIASES = {'auc_roc': 'A-R', 'auc_pr': 'A-P', 'VUS_ROC': 'V-R', 'VUS_PR': 'V-P', 'R_AUC_ROC': 'R_AUC_ROC', 'R_AUC_PR': 'R_AUC_PR'}
PAPER_SCORE_PRIMARY_KEYS = ('auc_roc', 'auc_pr', 'VUS_ROC', 'VUS_PR')
build_fft_reference = base.build_fft_reference
fft_deviation_channel_scores = base.fft_deviation_channel_scores

def create_windows_subsequent(X: np.ndarray, y: Optional[np.ndarray], window: int=WINDOW, target_len: int=TARGET_LEN) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    T, C = X.shape
    N = T - window - target_len + 1
    if N <= 0:
        raise ValueError(f'Time series too short: T={T}, need >= window+target_len+1={window + target_len + 1}')
    Xw = np.stack([X[i:i + window] for i in range(N)], axis=0).astype(np.float32)
    Yt = np.stack([X[i + window:i + window + target_len] for i in range(N)], axis=0).astype(np.float32)
    yw = None
    if y is not None:
        yw = np.array([np.max(y[i + window:i + window + target_len]) for i in range(N)], dtype=np.int64)
    return (Xw, Yt, yw)

def window_scores_to_point_scores_paper(test_scores: np.ndarray, T_test: int, window: int=WINDOW) -> np.ndarray:
    N = len(test_scores)
    fill = float(np.min(test_scores)) if N > 0 else 0.0
    point_scores = np.full(T_test, fill, dtype=np.float64)
    if window > 0:
        point_scores[:window] = fill
    for i in range(N):
        t = i + window
        if t < T_test:
            point_scores[t] = test_scores[i]
    return point_scores

def window_scores_to_point_scores_subsequent(test_scores: np.ndarray, T_test: int, window: int=WINDOW, target_len: int=TARGET_LEN) -> np.ndarray:
    N = len(test_scores)
    fill = float(np.min(test_scores)) if N > 0 else 0.0
    point_scores = np.full(T_test, fill, dtype=np.float64)
    for i in range(N):
        t0 = i + window
        t1 = min(i + window + target_len, T_test)
        if t0 < t1:
            point_scores[t0:t1] = np.maximum(point_scores[t0:t1], test_scores[i])
    return point_scores

def _point_score_start_index(align: str, window: int, target_len: int = TARGET_LEN) -> int:
    if align in ('paper', 'step_ovl_mean', 'step_ovl_max', 'step_ovl_wfront', 'step_ovl_wback', 'step_paper'):
        return int(window)
    if align in ('subsequent_max', 'subsequent'):
        return int(window + target_len - 1)
    raise ValueError(f'Unknown point_score align: {align!r}')


def map_step_scores_to_points(
    step_scores: np.ndarray,
    T: int,
    window: int = WINDOW,
    align: str = 'step_ovl_mean',
    weight_tau: float = 24.0,
) -> np.ndarray:
    """Map per-step window scores (N, H) to point scores (T,)."""
    N, H = step_scores.shape
    fill_ref = float(step_scores.min()) if step_scores.size else 0.0

    if align == 'step_paper':
        pt = np.full(T, fill_ref, dtype=np.float64)
        idx = np.arange(N) + window
        ok = idx < T
        pt[idx[ok]] = step_scores[ok, 0]
        return pt

    if align in ('step_ovl_mean', 'step_ovl_wfront', 'step_ovl_wback'):
        acc = np.zeros(T, dtype=np.float64)
        wsum = np.zeros(T, dtype=np.float64)
        for h in range(H):
            t = np.arange(N) + window + h
            ok = t < T
            if align == 'step_ovl_mean':
                w = 1.0
            elif align == 'step_ovl_wfront':
                w = float(np.exp(-h / weight_tau))
            else:
                w = float(np.exp(-(H - 1 - h) / weight_tau))
            np.add.at(acc, t[ok], step_scores[ok, h] * w)
            np.add.at(wsum, t[ok], w)
        pt = np.full(T, fill_ref, dtype=np.float64)
        nz = wsum > 0
        pt[nz] = acc[nz] / wsum[nz]
        return pt

    if align == 'step_ovl_max':
        pt = np.full(T, -np.inf, dtype=np.float64)
        for h in range(H):
            t = np.arange(N) + window + h
            ok = t < T
            np.maximum.at(pt, t[ok], step_scores[ok, h])
        pt[~np.isfinite(pt)] = fill_ref
        return pt

    raise ValueError(f'Unknown step align: {align!r}')


def window_scores_to_point_scores(
    test_scores: np.ndarray,
    T_test: int,
    window: int = WINDOW,
    target_len: int = TARGET_LEN,
    align: str = DEFAULT_POINT_SCORE_ALIGN,
) -> np.ndarray:
    if align == 'paper':
        return window_scores_to_point_scores_paper(test_scores, T_test, window=window)
    if align in ('subsequent_max', 'subsequent'):
        return window_scores_to_point_scores_subsequent(test_scores, T_test, window=window, target_len=target_len)
    if align in STEP_ALIGN_SCHEMES:
        raise ValueError(
            f'align={align!r} requires per-step scores; use compute_point_scores() instead.'
        )
    raise ValueError(f'Unknown point_score align: {align!r}')


def _topk_mean_channels(contrib: np.ndarray, k: int) -> np.ndarray:
    k = int(min(max(1, k), contrib.shape[-1]))
    part = np.partition(contrib, -k, axis=-1)[..., -k:]
    return part.mean(axis=-1)


def aggregate_step_channel_contrib(
    contrib_nhc: np.ndarray,
    scheme: str = DEFAULT_SCORE_AGGREGATION,
    mu: Optional[np.ndarray] = None,
    sd: Optional[np.ndarray] = None,
    topk: int = DEFAULT_SCORE_TOPK,
) -> np.ndarray:
    """Per-step channel contributions (N, H, C) -> per-step window scores (N, H)."""
    x = np.asarray(contrib_nhc, dtype=np.float64)
    if scheme.startswith('z'):
        if mu is None or sd is None:
            raise ValueError(f'scheme={scheme!r} requires train channel stats (mu, sd).')
        x = (x - mu[None, None, :]) / (sd[None, None, :] + EPS)
    if scheme in ('sum', 'zsum'):
        return x.sum(axis=-1)
    if scheme in ('max', 'zmax'):
        return x.max(axis=-1)
    if scheme in ('topk_mean', 'ztopk_mean'):
        return _topk_mean_channels(x, topk)
    raise ValueError(f'Unknown score aggregation: {scheme!r}')

def load_balance_loss(alphas: torch.Tensor, K_experts: int) -> torch.Tensor:
    alpha_mean = alphas.mean(dim=1)
    target_uniform = torch.full_like(alpha_mean, 1.0 / float(K_experts))
    return ((alpha_mean - target_uniform) ** 2).sum(dim=1).mean()

def eta_barrier_loss(alphas: torch.Tensor, eta0: float) -> torch.Tensor:
    """Pointwise Prop. 1 floor: penalize min_n α_{c,n} below η0.

    alphas : (B, C, K)
    Loss is mean over (B, C) of (ReLU(η0 - min_n α) / η0)^2, so a full
    collapse (min α = 0) costs ~1 per channel and is comparable to L_pred
    once weighted by λ_lb · λ_η.
    """
    if float(eta0) <= 0.0:
        return alphas.new_zeros(())
    min_a = alphas.min(dim=-1).values
    return ((F.relu(float(eta0) - min_a) / float(eta0)) ** 2).mean()

def load_balance_loss_with_eta(
    alphas: torch.Tensor,
    K_experts: int,
    eta0: float,
    lambda_eta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_lb' = L_usage + λ_η · L_η  (usage balance + pointwise η floor)."""
    l_usage = load_balance_loss(alphas, K_experts)
    l_eta = eta_barrier_loss(alphas, eta0)
    total = l_usage + float(lambda_eta) * l_eta
    return total, l_usage, l_eta

def _gaussian_fit_window_over_time(Y: torch.Tensor, cov_eps: float=BW_COV_EPS) -> Tuple[torch.Tensor, torch.Tensor]:
    if Y.dim() != 3:
        raise ValueError(f'Expected Y (B, W, C), got shape {tuple(Y.shape)}')
    B, W, C = Y.shape
    Z = Y.reshape(B * W, C)
    return base._covariance(Z, eps=cov_eps)

def bures_wasserstein_window_loss(Y_true: torch.Tensor, Y_pred: torch.Tensor, cov_eps: float=BW_COV_EPS, sqrt_eps: float=BW_SQRT_EPS) -> torch.Tensor:
    mu_t, sig_t = _gaussian_fit_window_over_time(Y_true, cov_eps=cov_eps)
    mu_p, sig_p = _gaussian_fit_window_over_time(Y_pred, cov_eps=cov_eps)
    l_dist = base.bures_wasserstein_gaussian(mu_t, sig_t, mu_p, sig_p, eps=sqrt_eps)
    if not torch.isfinite(l_dist):
        l_dist = torch.zeros((), device=Y_true.device, dtype=Y_true.dtype)
    return l_dist

def paper_training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alphas: torch.Tensor,
    K_experts: int,
    lambda_lb: float = LAMBDA_LB_PAPER,
    lambda_dist: float = LAMBDA_DIST_PAPER,
    lambda_grad: float = LAMBDA_GRAD,
    eta0: float = 0.0,
    lambda_eta: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    l_pred = F.huber_loss(pred, target, delta=1.0)
    if float(eta0) > 0.0:
        l_lb, l_usage, l_eta = load_balance_loss_with_eta(
            alphas, K_experts, eta0=float(eta0), lambda_eta=float(lambda_eta),
        )
    else:
        l_lb = load_balance_loss(alphas, K_experts)
        l_usage = l_lb
        l_eta = pred.new_zeros(())
    l_dist = bures_wasserstein_window_loss(target, pred)
    total = l_pred + float(lambda_lb) * l_lb + float(lambda_dist) * l_dist
    logs = {
        'L_pred': float(l_pred.detach().cpu().item()),
        'L_lb': float(l_lb.detach().cpu().item()),
        'L_usage': float(l_usage.detach().cpu().item()),
        'L_eta': float(l_eta.detach().cpu().item()),
        'L_dist': float(l_dist.detach().cpu().item()),
        'L_grad': 0.0,
        'total': float(total.detach().cpu().item()),
    }
    if float(lambda_grad) > 0.0 and pred.shape[1] > 1:
        dp = pred[:, 1:] - pred[:, :-1]
        dt = target[:, 1:] - target[:, :-1]
        l_grad = F.huber_loss(dp, dt, delta=1.0)
        total = total + float(lambda_grad) * l_grad
        logs['L_grad'] = float(l_grad.detach().cpu().item())
        logs['total'] = float(total.detach().cpu().item())
    return (total, logs)

def train_rose(
    model: nn.Module,
    Xw_train: np.ndarray,
    Yt_train: np.ndarray,
    lambda_grad: float = LAMBDA_GRAD,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    N = Xw_train.shape[0]
    idx = np.arange(N)
    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        logs_epoch = {'L_pred': [], 'L_lb': [], 'L_dist': [], 'L_grad': [], 'total': []}
        for s in range(0, N, BATCH):
            batch_idx = idx[s:s + BATCH]
            xb = torch.from_numpy(Xw_train[batch_idx]).to(DEVICE)
            yb = torch.from_numpy(Yt_train[batch_idx]).to(DEVICE)
            if USE_SSL:
                raise NotImplementedError('USE_SSL is disabled for subsequent-window ROSE.')
            pred, alphas = model(xb)
            loss, logs = paper_training_loss(
                pred, yb, alphas, model.K, lambda_grad=lambda_grad,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in logs_epoch:
                logs_epoch[k].append(logs[k])
        msg = (
            f"[epoch {ep + 1:02d}] total={np.mean(logs_epoch['total']):.4f} "
            f"L_pred={np.mean(logs_epoch['L_pred']):.4f} "
            f"L_lb={np.mean(logs_epoch['L_lb']):.4f} "
            f"L_dist={np.mean(logs_epoch['L_dist']):.4f} "
            f"L_grad={np.mean(logs_epoch['L_grad']):.4f} "
            f"(λ1={LAMBDA_LB_PAPER}, λ2={LAMBDA_DIST_PAPER}, λ_grad={lambda_grad})"
        )
        print(msg)
        if ep == 0 or (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            with torch.no_grad():
                mean_alpha = alphas.mean(dim=(0, 1)).cpu().numpy()
            s = ' '.join([f'e{k}:{mean_alpha[k]:.2f}' for k in range(model.K)])
            print(f'  [Expert usage] global mean alpha (ideal 1/K={1.0 / model.K:.2f}): {s}')

@torch.no_grad()
def get_routing_weights(model: nn.Module, Xw: np.ndarray, batch_size: int=BATCH) -> np.ndarray:
    model.eval()
    chunks = []
    N = Xw.shape[0]
    for s in range(0, N, batch_size):
        xb = torch.from_numpy(Xw[s:s + batch_size]).to(DEVICE)
        _, alphas = model(xb)
        chunks.append(alphas.cpu().numpy())
    return np.concatenate(chunks, axis=0)

@torch.no_grad()
def compute_alpha_train_mean(model: nn.Module, Xw_tr: np.ndarray, yw_tr: Optional[np.ndarray]=None, normal_only: bool=ROUTING_REF_NORMAL_ONLY, batch_size: int=BATCH) -> np.ndarray:
    alphas = get_routing_weights(model, Xw_tr, batch_size=batch_size)
    if normal_only and yw_tr is not None:
        mask = yw_tr == 0
        n_norm = int(mask.sum())
        if n_norm == 0:
            raise ValueError('No normal train windows (yw_tr==0) for alpha_train_mean; check train labels or set ROUTING_REF_NORMAL_ONLY=False.')
        alphas = alphas[mask]
        print(f'[alpha_train_mean] normal train windows only: {n_norm}/{len(yw_tr)}')
    mean_a = np.mean(alphas, axis=0).astype(np.float64)
    return mean_a

def routing_shift(alpha_nck: np.ndarray, alpha_train_mean_ck: np.ndarray) -> np.ndarray:
    diff = alpha_nck - alpha_train_mean_ck[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float64)

@torch.no_grad()
def compute_scores(model: nn.Module, Xw: np.ndarray, Yt: np.ndarray, sigma: np.ndarray, alpha_train_mean_ck: np.ndarray, lambda_s: float=LAMBDA_S, lambda_es: float=LAMBDA_ES) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    sigma_t = torch.from_numpy(np.asarray(sigma, dtype=np.float32)).to(DEVICE).view(1, 1, -1)
    alpha_ref = torch.from_numpy(np.asarray(alpha_train_mean_ck, dtype=np.float32)).to(DEVICE)
    score_chunks = []
    e_chunks = []
    s_chunks = []
    contrib_chunks = []
    alpha_chunks = []
    N = Xw.shape[0]
    for start in range(0, N, BATCH):
        xb = torch.from_numpy(Xw[start:start + BATCH]).to(DEVICE)
        yb = torch.from_numpy(Yt[start:start + BATCH]).to(DEVICE)
        pred, alphas = model(xb)
        e = torch.linalg.norm((yb - pred) / (sigma_t + EPS), dim=1)
        s = torch.linalg.norm(alphas - alpha_ref.unsqueeze(0), dim=-1)
        contrib = e + float(lambda_s) * s + float(lambda_es) * e * s
        score = contrib.sum(dim=1)
        score_chunks.append(score.cpu().numpy())
        e_chunks.append(e.cpu().numpy())
        s_chunks.append(s.cpu().numpy())
        contrib_chunks.append(contrib.cpu().numpy())
        alpha_chunks.append(alphas.cpu().numpy())
    return (np.concatenate(score_chunks).astype(np.float64), np.concatenate(e_chunks).astype(np.float64), np.concatenate(s_chunks).astype(np.float64), np.concatenate(contrib_chunks).astype(np.float64), np.concatenate(alpha_chunks).astype(np.float64))


@torch.no_grad()
def compute_per_step_channel_contrib(
    model: nn.Module,
    Xw: np.ndarray,
    Yt: np.ndarray,
    sigma: np.ndarray,
    alpha_train_mean_ck: np.ndarray,
    lambda_s: float = LAMBDA_S,
    lambda_es: float = LAMBDA_ES,
    batch_size: int = BATCH,
) -> np.ndarray:
    """Per-window, per-horizon-step, per-channel score: (N, H, C)."""
    model.eval()
    sig = torch.from_numpy(np.asarray(sigma, dtype=np.float32)).to(DEVICE).view(1, 1, -1)
    alpha_ref = torch.from_numpy(np.asarray(alpha_train_mean_ck, dtype=np.float32)).to(DEVICE)
    chunks: List[np.ndarray] = []
    for start in range(0, Xw.shape[0], batch_size):
        xb = torch.from_numpy(Xw[start:start + batch_size]).to(DEVICE)
        yb = torch.from_numpy(Yt[start:start + batch_size]).to(DEVICE)
        pred, alphas = model(xb)
        err = torch.abs((yb - pred) / (sig + EPS))
        s = torch.linalg.norm(alphas - alpha_ref.unsqueeze(0), dim=-1).unsqueeze(1)
        contrib = err + float(lambda_s) * s + float(lambda_es) * err * s
        chunks.append(contrib.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def compute_point_scores(
    model: nn.Module,
    Xw_tr: np.ndarray,
    Yt_tr: np.ndarray,
    Xw: np.ndarray,
    Yt: np.ndarray,
    sigma: np.ndarray,
    alpha_train_mean_ck: np.ndarray,
    score_aggregation: str = DEFAULT_SCORE_AGGREGATION,
    point_score_align: str = DEFAULT_POINT_SCORE_ALIGN,
    topk: int = DEFAULT_SCORE_TOPK,
    return_train_point_scores: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Full scoring pipeline: per-step channel contrib -> aggregation -> point mapping.

    Returns test point scores, optional train point scores (for threshold), and meta.
    """
    _, _, _, contrib_tr, _ = compute_scores(model, Xw_tr, Yt_tr, sigma, alpha_train_mean_ck)
    mu = contrib_tr.mean(axis=0)
    sd = contrib_tr.std(axis=0)

    step_tr = compute_per_step_channel_contrib(model, Xw_tr, Yt_tr, sigma, alpha_train_mean_ck)
    step_te = compute_per_step_channel_contrib(model, Xw, Yt, sigma, alpha_train_mean_ck)
    step_tr_agg = aggregate_step_channel_contrib(
        step_tr, scheme=score_aggregation, mu=mu, sd=sd, topk=topk,
    )
    step_te_agg = aggregate_step_channel_contrib(
        step_te, scheme=score_aggregation, mu=mu, sd=sd, topk=topk,
    )

    T_te = int(Xw.shape[0] + WINDOW + model.forecast_len - 1)
    T_tr = int(Xw_tr.shape[0] + WINDOW + model.forecast_len - 1)

    if point_score_align in STEP_ALIGN_SCHEMES:
        pt_te = map_step_scores_to_points(step_te_agg, T_te, WINDOW, align=point_score_align)
        pt_tr = map_step_scores_to_points(step_tr_agg, T_tr, WINDOW, align=point_score_align) if return_train_point_scores else None
    else:
        win_te = np.linalg.norm(step_te_agg, axis=1)
        win_tr = np.linalg.norm(step_tr_agg, axis=1)
        pt_te = window_scores_to_point_scores(win_te, T_te, WINDOW, model.forecast_len, align=point_score_align)
        pt_tr = window_scores_to_point_scores(win_tr, T_tr, WINDOW, model.forecast_len, align=point_score_align) if return_train_point_scores else None

    meta = {
        'score_aggregation': str(score_aggregation),
        'point_score_align': str(point_score_align),
        'contrib_mu': mu,
        'contrib_sd': sd,
    }
    return pt_te.astype(np.float64), pt_tr, meta


def compute_scores_with_optional_freq(model: nn.Module, Xw_tr: np.ndarray, Yt_tr: np.ndarray, Xw: np.ndarray, Yt: np.ndarray, sigma: np.ndarray, alpha_train_mean_ck: np.ndarray, yw_tr: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores, e, s, contrib, _ = compute_scores(model, Xw, Yt, sigma, alpha_train_mean_ck)
    if USE_FREQ_SCORE:
        fft_mu, fft_std, fft_idx = build_fft_reference(Xw_tr)
        ch_sc = fft_deviation_channel_scores(Xw, fft_mu, fft_std, fft_idx)
        scores = scores + LAMBDA_FREQ_SCORE * ch_sc.sum(axis=1)
    return (scores, e, s, contrib)

def _safe_metric_call(metric_fn, actual: np.ndarray, predicted: np.ndarray) -> float:
    try:
        return float(metric_fn(actual, predicted))
    except Exception:
        return float('nan')

def run_paper_score_evals(point_scores: np.ndarray, point_labels: np.ndarray) -> dict:
    y_true = np.asarray(point_labels, dtype=np.float64)
    y_score = np.asarray(point_scores, dtype=np.float64)
    margins = [0]
    out: dict = {'source': [], 'metrics': {}, 'primary': {}}
    try:
        from ts_benchmark.evaluation.metrics import classification_metrics_score
        for name in classification_metrics_score.__all__:
            fn = getattr(classification_metrics_score, name)
            out['metrics'][name] = _safe_metric_call(fn, y_true, y_score)
        out['source'].append('ts_benchmark.score')
    except Exception as e:
        print(f'[Eval] ts_benchmark score metrics unavailable: {e}')
    if not out['metrics']:
        try:
            from EasyTSAD.Evaluations.Protocols import PointAurocPA, PointAuprcPA, VUS_ROC, VUS_PR, R_AUC
            out['metrics']['auc_roc'] = float(PointAurocPA().calc(y_score, y_true, margins).value)
            out['metrics']['auc_pr'] = float(PointAuprcPA().calc(y_score, y_true, margins).value)
            out['metrics']['VUS_ROC'] = float(VUS_ROC().calc(y_score, y_true, margins).value)
            out['metrics']['VUS_PR'] = float(VUS_PR().calc(y_score, y_true, margins).value)
            r_auc = R_AUC().calc(y_score, y_true, margins)
            if hasattr(r_auc, 'value'):
                out['metrics']['R_AUC_ROC'] = float(r_auc.value)
            out['source'].append('EasyTSAD')
        except Exception as e:
            print(f'[Eval] EasyTSAD score metrics unavailable: {e}')
    for key in PAPER_SCORE_PRIMARY_KEYS:
        if key in out['metrics']:
            alias = PAPER_SCORE_METRIC_ALIASES.get(key, key)
            out['primary'][alias] = out['metrics'][key]
    out['kanad_extra'] = run_kanad_evals(point_scores, point_labels)
    return out

def threshold_from_train_scores(train_scores: np.ndarray, quantile: float=0.99, method: str='quantile') -> Tuple[float, str]:
    ref = np.asarray(train_scores, dtype=np.float64)
    tau, info = build_threshold(ref, method=method, quantile=float(quantile), mad_k=base.MAD_K, std_k=base.STD_K)
    return (float(tau), info)

def eval_optional_threshold_labels(point_scores: np.ndarray, point_labels: np.ndarray, tau: float, window: int=WINDOW, target_len: int=TARGET_LEN, point_score_align: str=DEFAULT_POINT_SCORE_ALIGN) -> dict:
    T = len(point_labels)
    point_taus = np.zeros(T, dtype=np.float64)
    n_win = max(0, T - window - target_len + 1)
    start_pt = _point_score_start_index(point_score_align, window, target_len)
    point_taus[start_pt:start_pt + n_win] = float(tau)
    point_taus[:start_pt] = float(tau)
    point_preds = (point_scores > point_taus).astype(np.int64)
    try:
        catch = run_catch_evals(point_scores, point_preds, point_labels)
        return {'tau': float(tau), 'label_metrics': catch.get('label', {}), 'score_metrics': catch.get('score', {})}
    except Exception as e:
        return {'tau': float(tau), 'error': str(e)}

def plot_routing_heatmap(alphas_nc: np.ndarray, save_path: str='routing_weights.png', title: str='Mean alpha per channel', max_channels_plot: int=55) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mean_a = np.mean(alphas_nc, axis=0)
    C, K_ex = mean_a.shape
    C_show = min(C, max_channels_plot)
    fig, ax = plt.subplots(figsize=(1.2 * K_ex + 2, max(4, 0.15 * C_show + 2)))
    im = ax.imshow(mean_a[:C_show], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(np.arange(K_ex))
    ax.set_xticklabels([f'E{k}' for k in range(K_ex)])
    ax.set_yticks(np.arange(C_show))
    ax.set_yticklabels([f'ch{c}' for c in range(C_show)], fontsize=6)
    ax.set_xlabel('expert')
    ax.set_ylabel('channel')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot] {save_path}')

def build_checkpoint_dict(model: nn.Module, data_dir: str, alpha_train_mean_ck: Optional[np.ndarray]=None) -> dict:
    from . import convhead_model as ch
    return ch.build_checkpoint_dict(model, data_dir, alpha_train_mean_ck=alpha_train_mean_ck)


def load_from_checkpoint(ckpt: dict) -> nn.Module:
    from . import convhead_model as ch
    return ch.load_from_checkpoint(ckpt)


train_model_channelwise = train_rose
load_channelwise_model_from_checkpoint = load_from_checkpoint
compute_scores_channelwise_paper = compute_scores
get_expert_weights_channelwise = get_routing_weights