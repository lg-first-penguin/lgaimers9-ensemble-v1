# code/experiment_yudam_trackman64_prune.py
"""[Pruning 실험 1] 트랙맨64 상황조인 완전 제거.

배경: process_trackman_features_safe 의 match_cols 에 season 이 포함된다 -> 2025 추론
때 season=2025 가 trackman(2019-2024)에 매칭 0건 -> 64컬럼 전부 per-column 상수로
붕괴한다 (code/train.py, submit/script.py 둘 다 확인). 즉 candidate B raw(실전 1093)는
이 64컬럼이 실전에서 죽은 채로 나온 점수. cutoff7 val 에서만 조인이 인위적으로
살아있어 §92 ablation 이 "load-bearing" 으로 나왔던 것.

가설: 추론 때 상수로 죽는 피처를 학습에 넣으면 모델이 학습 중 거기 기대 -> 나머지
가중치가 miscalibrate (repo 핵심교훈: 추론 때 새 값 갖는 컬럼은 학습에 넣지 말라).
64컬럼을 학습·추론 양쪽에서 완전히 빼면 실전에서 오히려 개선될 수 있다.

검증 (레짐당, 같은 build_split):
  BASE_norm       : 트랙맨64 살린 채 학습, val 정상 채점
                    (cutoff7 에서 부풀려진 값 = 2025 에서 안 벌어지는 조건)
  BASE_coll       : 트랙맨64 살린 채 학습, val 의 트랙맨64 -> NaN -> 전처리기 median
                    (MLP) / NaN native (CatBoost) 로 채점. 메타는 BASE_norm 예측에
                    fit(=stage1 관례) 한 걸 collapsed 예측에 적용 -> 실제 2025 추론 재현.
  PRUNE           : 트랙맨64 를 num_cols / cat_feature_cols 양쪽에서 제거하고 학습,
                    val 정상 채점.

결정 지표 : PRUNE.blend - BASE_coll.blend   (>0 이면 pruning 이 실전에서 이득)
참고 지표 : PRUNE.blend - BASE_norm.blend   (조인이 살아있을 때의 비용)

레짐: cutoff7 / 2023 / 2022 / 2021.  MLP 3-seed + CatBoost 3-seed 기본.

사용법:
  python -m code.experiment_yudam_trackman64_prune --seeds 3 --regimes cutoff7,2023,2022,2021
"""
import argparse
import gc

import numpy as np
from sklearn.linear_model import LogisticRegression

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.experiment_yudam_hybrid_mlp import TRACKMAN64_RE
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, make_bundle, predict_bundle, compute_bss, get_device,
)
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS


def _fit_meta(cat_p, mlp_p, y):
    clf = LogisticRegression().fit(np.column_stack([cat_p, mlp_p]), y)
    w_cat, w_mlp = (float(c) for c in clf.coef_[0])
    b = float(clf.intercept_[0])
    return w_cat, w_mlp, b


def _blend_score(w, cat_p, mlp_p, y):
    w_cat, w_mlp, b = w
    p = 1.0 / (1.0 + np.exp(-(w_cat * cat_p + w_mlp * mlp_p + b)))
    return compute_bss(p, y)[2]


def _train_mlp(train_split, val_split, num_cols, all_cols, seeds, device):
    """raw-concat MLP 앙상블 학습 -> bundle 반환 (predict_bundle 로 임의 val df 채점)."""
    train_proc, ce, ni, ns, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, ce, ni, ns)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)
    y_val = val_proc[TARGET].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims, bin_edges=None,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
        seeds=seeds, device=device, verbose=False,
    )
    bundle = make_bundle(members, CAT_COLS, num_cols, cat_dims, embed_dims, ce, ni, ns, bin_edges=None)
    del train_proc, val_proc
    gc.collect()
    return bundle


def run_regime(regime, mlp_seeds, cb_seeds, device):
    print(f"\n{'='*70}\n=== regime={regime} ===\n{'='*70}", flush=True)
    cb_params = _yudam_catboost_params()

    # ---- BASE: 트랙맨64 포함 ----
    train_split, val_split, num_cols, cat_feature_cols, all_cols = build_split(regime)
    y_val = val_split[TARGET].values
    trk_num = [c for c in num_cols if TRACKMAN64_RE.match(c)]
    trk_cat = [c for c in cat_feature_cols if TRACKMAN64_RE.match(c)]
    print(f"[cols] num {len(num_cols)} (트랙맨64 {len(trk_num)}) | "
          f"cat_feat {len(cat_feature_cols)} (트랙맨64 {len(trk_cat)})", flush=True)

    mlp_bundle = _train_mlp(train_split, val_split, num_cols, all_cols, mlp_seeds, device)
    mlp_norm = predict_bundle(mlp_bundle, val_split[all_cols], device=device)
    val_coll = val_split.copy()
    val_coll[trk_num] = np.nan
    mlp_coll = predict_bundle(mlp_bundle, val_coll[all_cols], device=device)

    cb = train_catboost_ensemble(
        train_split[cat_feature_cols], train_split[TARGET].values,
        val_split[cat_feature_cols], y_val, seeds=cb_seeds, verbose=False, params=cb_params,
    )
    cb_models = [m for m, _ in cb]
    cat_norm = predict_catboost_ensemble(cb_models, val_split[cat_feature_cols])
    val_cb_coll = val_split.copy()
    val_cb_coll[trk_cat] = np.nan
    cat_coll = predict_catboost_ensemble(cb_models, val_cb_coll[cat_feature_cols])

    # sanity: collapsed 후 트랙맨64 상수 확인 (MLP 전처리 후)
    _, _ce, _ni, _ns, _ = fit_preprocessing(train_split, CAT_COLS, num_cols)
    _chk = apply_preprocessing(val_coll, CAT_COLS, num_cols, _ce, _ni, _ns)
    uniq = int(_chk[trk_num].nunique().max()) if trk_num else 0
    print(f"[sanity] collapsed val 트랙맨64 전처리후 컬럼당 고유값 max={uniq} (1이어야 상수붕괴)", flush=True)
    del _chk

    base_norm_solo = (compute_bss(cat_norm, y_val)[2], compute_bss(mlp_norm, y_val)[2])
    base_coll_solo = (compute_bss(cat_coll, y_val)[2], compute_bss(mlp_coll, y_val)[2])

    meta_norm = _fit_meta(cat_norm, mlp_norm, y_val)         # stage1 관례: 정상 예측에 fit
    base_norm_blend = _blend_score(meta_norm, cat_norm, mlp_norm, y_val)
    base_coll_blend = _blend_score(meta_norm, cat_coll, mlp_coll, y_val)   # 그 메타를 collapsed 에 적용 = 실전
    # 참고: 메타를 collapsed 예측에 refit 했을 때 (상한)
    meta_coll = _fit_meta(cat_coll, mlp_coll, y_val)
    base_coll_blend_refit = _blend_score(meta_coll, cat_coll, mlp_coll, y_val)

    del mlp_bundle, val_coll, val_cb_coll, train_split, val_split
    gc.collect()

    # ---- PRUNE: 트랙맨64 완전 제거 ----
    def _drop(cols):
        return [c for c in cols if TRACKMAN64_RE.match(c)]

    tr_p, va_p, num_p, catf_p, all_p = build_split(regime, drop_cols=_drop)
    assert not [c for c in num_p if TRACKMAN64_RE.match(c)], "PRUNE num_cols 에 트랙맨64 잔존"
    assert not [c for c in catf_p if TRACKMAN64_RE.match(c)], "PRUNE cat_feature_cols 에 트랙맨64 잔존"
    yv_p = va_p[TARGET].values

    mlp_bundle_p = _train_mlp(tr_p, va_p, num_p, all_p, mlp_seeds, device)
    mlp_p_pred = predict_bundle(mlp_bundle_p, va_p[all_p], device=device)
    cb_p = train_catboost_ensemble(
        tr_p[catf_p], tr_p[TARGET].values, va_p[catf_p], yv_p,
        seeds=cb_seeds, verbose=False, params=cb_params,
    )
    cat_p_pred = predict_catboost_ensemble([m for m, _ in cb_p], va_p[catf_p])
    prune_solo = (compute_bss(cat_p_pred, yv_p)[2], compute_bss(mlp_p_pred, yv_p)[2])
    meta_p = _fit_meta(cat_p_pred, mlp_p_pred, yv_p)
    prune_blend = _blend_score(meta_p, cat_p_pred, mlp_p_pred, yv_p)

    del mlp_bundle_p, tr_p, va_p
    gc.collect()

    # ---- 요약 ----
    print(f"\n--- regime={regime} 요약 (mlp{len(mlp_seeds)}seed / cb{len(cb_seeds)}seed) ---", flush=True)
    print(f"  BASE_norm        CatBoost {base_norm_solo[0]:8.2f} | MLP {base_norm_solo[1]:8.2f} | blend {base_norm_blend:8.2f}", flush=True)
    print(f"  BASE_coll(실전)  CatBoost {base_coll_solo[0]:8.2f} | MLP {base_coll_solo[1]:8.2f} | blend {base_coll_blend:8.2f}  (메타 refit 시 {base_coll_blend_refit:.2f})", flush=True)
    print(f"  PRUNE            CatBoost {prune_solo[0]:8.2f} | MLP {prune_solo[1]:8.2f} | blend {prune_blend:8.2f}", flush=True)
    print(f"  >>> 결정지표  PRUNE - BASE_coll(실전)  blend = {prune_blend - base_coll_blend:+.2f}", flush=True)
    print(f"  >>> 참고지표  PRUNE - BASE_norm        blend = {prune_blend - base_norm_blend:+.2f}", flush=True)
    return dict(
        regime=regime,
        base_norm_blend=base_norm_blend, base_coll_blend=base_coll_blend,
        base_coll_blend_refit=base_coll_blend_refit, prune_blend=prune_blend,
        d_decision=prune_blend - base_coll_blend, d_reference=prune_blend - base_norm_blend,
        base_norm_solo=base_norm_solo, base_coll_solo=base_coll_solo, prune_solo=prune_solo,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--regimes", type=str, default="cutoff7,2023,2022,2021")
    args = ap.parse_args()
    mlp_seeds = list(YUDAM_ENSEMBLE_SEEDS[:args.seeds])
    cb_seeds = list(YUDAM_CATBOOST_SEEDS[:args.seeds])
    device = get_device()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    rows = []
    for rg in regimes:
        rows.append(run_regime(rg, mlp_seeds, cb_seeds, device))
        gc.collect()

    print(f"\n{'='*78}\n=== 트랙맨64 완전제거 pruning — 종합 (mlp{args.seeds}seed/cb{args.seeds}seed) ===\n{'='*78}", flush=True)
    print(f"{'regime':>9} | {'BASE_norm':>10} | {'BASE_coll':>10} | {'PRUNE':>10} | "
          f"{'Δ decision':>11} | {'Δ ref':>8}", flush=True)
    for r in rows:
        print(f"{r['regime']:>9} | {r['base_norm_blend']:10.2f} | {r['base_coll_blend']:10.2f} | "
              f"{r['prune_blend']:10.2f} | {r['d_decision']:+11.2f} | {r['d_reference']:+8.2f}", flush=True)
    dd = [r["d_decision"] for r in rows]
    print(f"\nΔ decision (PRUNE - 실전BASE): mean {np.mean(dd):+.2f}  std {np.std(dd):.2f}  "
          f"min {np.min(dd):+.2f}  max {np.max(dd):+.2f}", flush=True)
    print("판정 가이드: 모든 레짐 Δdecision >= ~0 (+ 크면 명확) 이면 실전 제출 후보. "
          "cutoff7 만 크게 +이고 나머지 -이면 또 regime-flip.", flush=True)


if __name__ == "__main__":
    main()
