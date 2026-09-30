#!/usr/bin/env python3
"""
Train and evaluate ROSE (1D conv forecast head, no encoder pooling).

Usage (from repo root):
    uv run python -m rose.train_eval_convhead \\
        --data_dir datasets/MTS/MSL/AllInOne \\
        --checkpoint checkpoints/rose_msl.pt \\
        --out_dir outputs/rose_msl \\
        --seed 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from . import convhead_model as ch
from . import model as m
from .utils import DEFAULT_DATA_DIR


def _point_taus_for_align(T_test: int, n_win: int, tau: float, window: int, target_len: int, align: str) -> np.ndarray:
    point_taus = np.zeros(T_test, dtype=np.float64)
    start_pt = m._point_score_start_index(align, window, target_len)
    point_taus[start_pt:start_pt + n_win] = float(tau)
    point_taus[:start_pt] = float(tau)
    return point_taus


def _save_metrics(path_json: str, path_csv: str, meta: dict, primary: dict, all_metrics: dict):
    payload = {'meta': meta, 'primary_score_metrics': primary, 'all_score_metrics': all_metrics}
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    row = {**meta}
    for k, v in primary.items():
        row[f'primary::{k}'] = v
    for k, v in all_metrics.items():
        row[f'score::{k}'] = v
    with open(path_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f'[Saved] {path_json}\n[Saved] {path_csv}')


def parse_args():
    p = argparse.ArgumentParser(description='ROSE — train and evaluate')
    p.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    p.add_argument('--checkpoint', type=str, default='checkpoints/rose.pt')
    p.add_argument('--out_dir', type=str, default='outputs/rose')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--with_threshold_eval', action='store_true')
    p.add_argument('--threshold_quantile', type=float, default=0.99)
    p.add_argument('--head_kernel', type=int, default=ch.DEFAULT_HEAD_KERNEL)
    p.add_argument('--head_layers', type=int, default=ch.DEFAULT_HEAD_LAYERS)
    p.add_argument('--point_score_align', type=str, default=m.DEFAULT_POINT_SCORE_ALIGN,
                   choices=('paper', 'subsequent_max', 'step_ovl_mean', 'step_ovl_max',
                            'step_ovl_wfront', 'step_ovl_wback', 'step_paper'))
    p.add_argument('--score_aggregation', type=str, default=m.DEFAULT_SCORE_AGGREGATION,
                   choices=('sum', 'max', 'topk_mean', 'zsum', 'zmax', 'ztopk_mean'))
    p.add_argument('--score_topk', type=int, default=m.DEFAULT_SCORE_TOPK)
    p.add_argument(
        '--gate_pool', type=str, default=ch.DEFAULT_GATE_POOL,
        choices=list(ch.GATE_POOL_CHOICES),
        help='Gate summary for routing alpha (experts keep full feature map)',
    )
    return p.parse_args()


def main(
    data_dir: str,
    checkpoint: str,
    out_dir: str = 'outputs/rose',
    seed: int = 42,
    with_threshold_eval: bool = False,
    threshold_quantile: float = 0.99,
    head_kernel: int = ch.DEFAULT_HEAD_KERNEL,
    head_layers: int = ch.DEFAULT_HEAD_LAYERS,
    point_score_align: str = m.DEFAULT_POINT_SCORE_ALIGN,
    score_aggregation: str = m.DEFAULT_SCORE_AGGREGATION,
    score_topk: int = m.DEFAULT_SCORE_TOPK,
    gate_pool: str = ch.DEFAULT_GATE_POOL,
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint) or '.', exist_ok=True)
    m.set_seed(int(seed))
    print(
        f'[ROSE] head_kernel={head_kernel} head_layers={head_layers} '
        f'| gate_pool={gate_pool} | score_aggregation={score_aggregation} '
        f'| point_score_align={point_score_align} | lambda_es={m.LAMBDA_ES}'
    )

    X_train, y_train, X_test, y_test = m.load_data(data_dir=data_dir)
    C = X_train.shape[1]
    print(f'Train {X_train.shape}, Test {X_test.shape}, C={C}')
    Xw_tr, Yt_tr, yw_tr = m.create_windows_subsequent(X_train, y_train)
    Xw_te, Yt_te, _ = m.create_windows_subsequent(X_test, y_test)
    print(f'[Windows] ({m.WINDOW}, C) -> predict ({m.TARGET_LEN}, C) via conv head (no pooling)')

    normal_mask = y_train == 0
    Xn = X_train[normal_mask] if normal_mask.mean() >= 0.5 else X_train
    sigma = np.std(Xn, axis=0).astype(np.float32) + 1e-06

    net = ch.ROSE(
        num_channels=C,
        K_experts=m.K,
        forecast_len=m.TARGET_LEN,
        head_kernel=head_kernel,
        head_layers=head_layers,
        gate_pool=gate_pool,
    ).to(m.DEVICE)

    print(f'[Train] L_pred + {m.LAMBDA_LB_PAPER}*L_lb + {m.LAMBDA_DIST_PAPER}*L_dist')
    ch.train_rose(net, Xw_tr, Yt_tr)

    alpha_train_mean = m.compute_alpha_train_mean(net, Xw_tr, yw_tr=yw_tr)
    np.save(os.path.join(out_dir, 'alpha_train_mean.npy'), alpha_train_mean)
    torch.save(ch.build_checkpoint_dict(net, data_dir, alpha_train_mean_ck=alpha_train_mean), checkpoint)
    print(f'[Saved] {checkpoint}')

    train_scores, _, _, _ = m.compute_scores_with_optional_freq(
        net, Xw_tr, Yt_tr, Xw_tr, Yt_tr, sigma, alpha_train_mean,
    )
    test_scores, test_e, test_s, test_contrib = m.compute_scores_with_optional_freq(
        net, Xw_tr, Yt_tr, Xw_te, Yt_te, sigma, alpha_train_mean,
    )
    point_scores, train_point_scores, score_meta = m.compute_point_scores(
        net, Xw_tr, Yt_tr, Xw_te, Yt_te, sigma, alpha_train_mean,
        score_aggregation=score_aggregation,
        point_score_align=point_score_align,
        topk=score_topk,
        return_train_point_scores=True,
    )
    if len(point_scores) != len(y_test):
        point_scores = point_scores[:len(y_test)]
    if train_point_scores is not None and len(train_point_scores) != len(y_train):
        train_point_scores = train_point_scores[:len(y_train)]

    print('============= [ROSE] Raw score metrics =============')
    paper_eval = m.run_paper_score_evals(point_scores, y_test)
    for alias, val in paper_eval.get('primary', {}).items():
        print(f'  {alias}: {val:.4f}')

    meta = {
        'model': 'ROSE',
        'model_type': 'rose',
        'data_dir': data_dir,
        'C': C,
        'K': m.K,
        'window': m.WINDOW,
        'head_kernel': int(head_kernel),
        'head_layers': int(head_layers),
        'gate_pool': str(gate_pool),
        'lambda_s': float(m.LAMBDA_S),
        'lambda_es': float(m.LAMBDA_ES),
        'point_score_align': point_score_align,
        'score_aggregation': score_aggregation,
        'score_topk': int(score_topk),
    }
    _save_metrics(
        os.path.join(out_dir, 'metrics.json'),
        os.path.join(out_dir, 'metrics.csv'),
        meta,
        paper_eval.get('primary', {}),
        paper_eval.get('metrics', {}),
    )
    np.savez(
        os.path.join(out_dir, 'scores.npz'),
        train_window_scores=train_scores,
        test_window_scores=test_scores,
        point_scores=point_scores,
        train_point_scores=train_point_scores,
        alpha_train_mean=alpha_train_mean,
        test_e=test_e,
        test_s=test_s,
        test_channel_contrib=test_contrib,
        contrib_mu=score_meta.get('contrib_mu'),
        contrib_sd=score_meta.get('contrib_sd'),
    )

    point_taus = None
    if with_threshold_eval:
        thresh_ref = train_point_scores
        tau, info = m.threshold_from_train_scores(thresh_ref, quantile=threshold_quantile)
        print(f'[Threshold] {info}')
        opt = m.eval_optional_threshold_labels(point_scores, y_test, tau, point_score_align=point_score_align)
        for name, val in sorted(opt.get('label_metrics', {}).items()):
            print(f'  {name}: {val:.4f}')
        point_taus = _point_taus_for_align(
            len(y_test), len(test_scores), tau, m.WINDOW, m.TARGET_LEN, point_score_align,
        )

    m.plot_anomaly_result(
        X_test, y_test, point_scores,
        save_path=os.path.join(out_dir, 'anomaly_plot.png'),
        max_points=3000, channel=0, point_taus=point_taus,
    )


if __name__ == '__main__':
    args = parse_args()
    main(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        seed=args.seed,
        with_threshold_eval=args.with_threshold_eval,
        threshold_quantile=args.threshold_quantile,
        head_kernel=args.head_kernel,
        head_layers=args.head_layers,
        point_score_align=args.point_score_align,
        score_aggregation=args.score_aggregation,
        score_topk=args.score_topk,
        gate_pool=args.gate_pool,
    )
