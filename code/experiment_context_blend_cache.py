# code/experiment_context_blend_cache.py
"""[Context 블렌드 실험 1/2] fold별 base 예측 + context 캐싱.

constrained context-dependent blend weight 실험용. yudam 의 `cache_fold_predictions.py`
대응. 4개 레짐(cutoff7/2023/2022/2021)에 대해 현행 1117 베이스라인(트랙맨64 제거 +
raw-concat MLP + v2 CatBoost) base 모델을 학습하고, val 셋의 (p_cat, p_mlp, y) +
context 컬럼을 `scratchpad/ctxblend_fold_{regime}.pkl` 로 저장한다. 저장 후엔
`experiment_context_blend_constrained.py` 가 재학습 없이 LOFO 메타만 반복.

스크리닝 시드 수(MLP 3-seed / CatBoost 3-seed) 기본 — context 축이 살아남으면
7/5-seed 로 재캐싱.

사용법:  python -m code.experiment_context_blend_cache --seeds 3
"""
import argparse
import gc
import os
import pickle

import numpy as np

from code.experiment_yudam_common import build_split
from code.experiment_yudam_trackman64_prune import _train_mlp
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.experiment_yudam_common import _yudam_catboost_params
from code.mlp_model import predict_bundle, compute_bss, get_device
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS, TE_RESIDUAL_COLS

TARGET = "control_success"
OUT_DIR = "scratchpad"
REGIMES = ["cutoff7", "2023", "2022", "2021"]

# 메타에 넣을 후보 context 축 (+ 참조용 season/month). 전부 build_split 의 val_split 에 존재.
CTX_COLS = [
    "li", "count_pressure", "count_diff", "strikes_before", "balls_before",
    "outs_before", "is_full_count", "same_hand", "num_runners_on",
    "home_win_expectancy", "score_diff_pitcher_team", "inning",
    "season", "game_month",
]


def run_regime(regime, mlp_seeds, cb_seeds, device):
    print(f"\n{'='*68}\n=== cache regime={regime} ===\n{'='*68}", flush=True)
    ts, vs, num_cols, cat_feature_cols, all_cols = build_split(regime, verbose=True)
    y_val = vs[TARGET].values.astype(np.float64)

    mlp_bundle = _train_mlp(ts, vs, num_cols, all_cols, mlp_seeds, device)
    p_mlp = predict_bundle(mlp_bundle, vs[all_cols], device=device).astype(np.float64)

    cb = train_catboost_ensemble(
        ts[cat_feature_cols], ts[TARGET].values, vs[cat_feature_cols], y_val,
        seeds=cb_seeds, verbose=False, params=_yudam_catboost_params(),
    )
    p_cat = predict_catboost_ensemble([m for m, _ in cb], vs[cat_feature_cols]).astype(np.float64)

    cat_solo = compute_bss(p_cat, y_val)[2]
    mlp_solo = compute_bss(p_mlp, y_val)[2]
    ctx = vs[[c for c in CTX_COLS if c in vs.columns]].reset_index(drop=True).copy()
    missing = [c for c in CTX_COLS if c not in vs.columns]
    print(f"[{regime}] p_cat solo={cat_solo:.2f} | p_mlp solo={mlp_solo:.2f} | "
          f"n_val={len(y_val)} | ctx cols={list(ctx.columns)}{' MISSING '+str(missing) if missing else ''}", flush=True)

    out = dict(regime=regime, p_cat=p_cat, p_mlp=p_mlp, y=y_val, ctx=ctx,
               cat_solo=cat_solo, mlp_solo=mlp_solo,
               n_mlp_seed=len(mlp_seeds), n_cb_seed=len(cb_seeds))
    path = os.path.join(OUT_DIR, f"ctxblend_fold_{regime}.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[{regime}] saved -> {path}", flush=True)

    del ts, vs, mlp_bundle, cb
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--regimes", type=str, default=",".join(REGIMES))
    args = ap.parse_args()
    mlp_seeds = list(YUDAM_ENSEMBLE_SEEDS[:args.seeds])
    cb_seeds = list(YUDAM_CATBOOST_SEEDS[:args.seeds])
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)
    for rg in [r.strip() for r in args.regimes.split(",") if r.strip()]:
        run_regime(rg, mlp_seeds, cb_seeds, device)
        gc.collect()
    print("\n[cache 완료] 다음: python -m code.experiment_context_blend_constrained", flush=True)


if __name__ == "__main__":
    main()
