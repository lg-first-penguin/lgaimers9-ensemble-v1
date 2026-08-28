# code/experiment_yudam_ple_trackman_collapse.py
"""[레버 A - Step 1d] "2025 추론 트랙맨 상수붕괴" 시뮬레이션.

로컬 cutoff7 val(2024)은 트랙맨 10-key 조인이 살아있어 full_ple가 MLP solo +23을
낸다. 그러나 팀원 실전에선 PLE + 트랙맨64 = 968.15 -> 879.54 (-88). 이유 가설:
2025 추론 때 season==2025가 trackman(2019-2024)에 없어 트랙맨64 전 컬럼이
NaN -> SimpleImputer(median) 로 단일 상수로 붕괴하고, PLE가 StandardScaler보다
이 붕괴에 취약(팀원 진단 ~4x). 로컬은 이 조건을 재현 못함 -> 여기서 강제 재현.

방법: cutoff7 split 정상 학습. val 예측 시 두 가지로:
  normal    : val 그대로 (트랙맨64 조인 살아있음)
  collapsed : val 의 트랙맨64 컬럼 전부 NaN -> 전처리기(train median)로 채움
              = 실제 2025 추론에서 벌어지는 일
arm: baseline(전 raw) vs full_ple(전 PLE). CatBoost(유담 v2 HP)도 동일하게
collapsed val 로 별도 예측(NaN native).

핵심 지표: degradation = solo(normal) - solo(collapsed).
  full_ple 의 degradation 이 baseline 보다 크게 나쁘면 -> 팀원 -88 메커니즘 확인,
  트랙맨64 존재 하에 PLE 재도입 위험.

사용법:
  python -m code.experiment_yudam_ple_trackman_collapse --seeds 3
  python -m code.experiment_yudam_ple_trackman_collapse --seeds 7
"""
import argparse
import gc

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    fit_quantile_edges, compute_bss, get_device,
)
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS
from code.experiment_yudam_hybrid_mlp import HybridMLP, train_one, TRACKMAN64_RE


def mlp_predict(mdl, Xc, Xn, device):
    with torch.no_grad():
        return mdl(Xc.to(device), Xn.to(device)).cpu().numpy()


def run_arm(name, use_ple, tr_p, va_norm_p, va_coll_p, ordered, cat_dims, embed_dims,
            seeds, device, y_val):
    n = len(ordered)
    n_ple, n_raw = (n, 0) if use_ple else (0, n)
    Xtc = torch.tensor(tr_p[CAT_COLS].values.astype(np.float32))
    Xtn = torch.tensor(tr_p[ordered].values.astype(np.float32))
    ytr = torch.tensor(tr_p[TARGET].values.astype(np.float32))
    Xvc = torch.tensor(va_norm_p[CAT_COLS].values.astype(np.float32))
    Xvn_norm = torch.tensor(va_norm_p[ordered].values.astype(np.float32))
    Xvn_coll = torch.tensor(va_coll_p[ordered].values.astype(np.float32))
    ple_edges = fit_quantile_edges(Xtn[:, :n_ple]) if n_ple > 0 else None

    p_norm, p_coll = [], []
    for s in seeds:
        mdl = train_one(Xtc, Xtn, ytr, Xvc, Xvn_norm, y_val, n_ple, n_raw,
                        cat_dims, embed_dims, ple_edges, s, device)
        p_norm.append(mlp_predict(mdl, Xvc, Xvn_norm, device))
        p_coll.append(mlp_predict(mdl, Xvc, Xvn_coll, device))
        del mdl; gc.collect()
    mn = np.mean(p_norm, axis=0)
    mc = np.mean(p_coll, axis=0)
    s_norm = compute_bss(mn, y_val)[2]
    s_coll = compute_bss(mc, y_val)[2]
    print(f"[{name}] MLP solo  normal={s_norm:.2f}  collapsed={s_coll:.2f}  "
          f"degradation={s_norm - s_coll:+.2f}", flush=True)
    return dict(mlp_norm=mn, mlp_coll=mc, s_norm=s_norm, s_coll=s_coll)


def blend(cat_p, mlp_p, y):
    clf = LogisticRegression().fit(np.column_stack([cat_p, mlp_p]), y)
    wc, wm = (float(c) for c in clf.coef_[0]); b = float(clf.intercept_[0])
    return compute_bss(1 / (1 + np.exp(-(wc * cat_p + wm * mlp_p + b))), y)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = YUDAM_ENSEMBLE_SEEDS[:args.seeds]
    cb_seeds = YUDAM_CATBOOST_SEEDS[:args.seeds]
    device = get_device()

    train_split, val_split, num_cols, cat_feature_cols, all_cols = build_split(regime="cutoff7")
    y_val = val_split[TARGET].values
    trk = [c for c in num_cols if TRACKMAN64_RE.match(c)]
    nontrk = [c for c in num_cols if not TRACKMAN64_RE.match(c)]
    ordered = nontrk + trk           # 트랙맨64를 뒤로 몰아둠 (arm 로직 단순화)
    print(f"[cols] num {len(num_cols)} = 트랙맨64 {len(trk)} + 나머지 {len(nontrk)}", flush=True)

    # 전처리기는 정상 train 에 fit
    tr_p, ce, ni, ns, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    va_norm_p = apply_preprocessing(val_split, CAT_COLS, num_cols, ce, ni, ns)

    # collapsed val: 트랙맨64 전부 NaN -> imputer(median) 이 train median 으로 채움
    val_coll = val_split.copy()
    val_coll[trk] = np.nan
    va_coll_p = apply_preprocessing(val_coll, CAT_COLS, num_cols, ce, ni, ns)

    # sanity: collapsed 후 트랙맨64가 실제로 상수인지
    uniq = va_coll_p[trk].nunique().max()
    print(f"[sanity] collapsed val 트랙맨64 컬럼당 고유값 max={uniq} (1이어야 상수붕괴)", flush=True)

    # CatBoost: 정상 학습, 정상/collapsed val 각각 예측 (NaN native)
    cb = train_catboost_ensemble(
        train_split[cat_feature_cols], train_split[TARGET].values,
        val_split[cat_feature_cols], y_val, seeds=cb_seeds, verbose=False,
        params=_yudam_catboost_params(),
    )
    cb_models = [m for m, _ in cb]
    cat_norm = predict_catboost_ensemble(cb_models, val_split[cat_feature_cols])
    val_cb_coll = val_split.copy()
    val_cb_coll[[c for c in trk if c in val_cb_coll.columns]] = np.nan
    cat_coll = predict_catboost_ensemble(cb_models, val_cb_coll[cat_feature_cols])
    print(f"[CatBoost] solo  normal={compute_bss(cat_norm, y_val)[2]:.2f}  "
          f"collapsed={compute_bss(cat_coll, y_val)[2]:.2f}  "
          f"degradation={compute_bss(cat_norm, y_val)[2] - compute_bss(cat_coll, y_val)[2]:+.2f}", flush=True)
    del train_split, val_split, val_coll, val_cb_coll; gc.collect()

    res = {}
    for name, use_ple in [("baseline", False), ("full_ple", True)]:
        res[name] = run_arm(name, use_ple, tr_p, va_norm_p, va_coll_p, ordered,
                            cat_dims, embed_dims, seeds, device, y_val)

    # 블렌드 (normal cat + normal mlp) vs (collapsed cat + collapsed mlp)
    print(f"\n{'='*66}\n=== 트랙맨64 상수붕괴 시뮬 요약 | cutoff7 | {args.seeds}-seed ===\n{'='*66}", flush=True)
    for name in ("baseline", "full_ple"):
        r = res[name]
        bl_n = blend(cat_norm, r["mlp_norm"], y_val)
        bl_c = blend(cat_coll, r["mlp_coll"], y_val)
        print(f"  {name:9s}  MLP solo  {r['s_norm']:8.2f} -> {r['s_coll']:8.2f}  "
              f"(deg {r['s_norm']-r['s_coll']:+7.2f}) | blend {bl_n:8.2f} -> {bl_c:8.2f}  "
              f"(deg {bl_n-bl_c:+7.2f})", flush=True)
    b, f = res["baseline"], res["full_ple"]
    print(f"\n  full_ple vs baseline:", flush=True)
    print(f"    normal    MLP Δ = {f['s_norm']-b['s_norm']:+.2f}  (로컬 cutoff7 기존 관측 +23 근처여야)", flush=True)
    print(f"    collapsed MLP Δ = {f['s_coll']-b['s_coll']:+.2f}  (이게 음수/급감이면 팀원 -88 메커니즘 확인)", flush=True)
    print(f"    degradation 차이 = full_ple {f['s_norm']-f['s_coll']:+.2f} vs baseline {b['s_norm']-b['s_coll']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
