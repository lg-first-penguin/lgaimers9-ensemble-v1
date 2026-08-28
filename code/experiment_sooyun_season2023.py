# code/experiment_sooyun_season2023.py
"""§85(손수연 레시피 cutoff7 멀티시드 재검증, 블렌드 3/3승 평균+5.18)의 다음 단계 —
season2023 레짐(train<2023/val=2023) 재검증. §84/§85에서 재사용했던
`open/reference/best_model.pkl`의 MLP 번들은 cutoff7 컨벤션(2019~2023 전부 학습
포함)이라 season2023 홀드아웃엔 구조적으로 리크되므로 여기선 새 MLP를
`train<2023/val=2023` 스플릿 전용으로 처음부터 재학습한다(자원 절약을 위해 프로덕션
20-seed 대신 7-seed로 축소 — baseline과 손수연 후보 둘 다 이 동일한 fresh MLP를
재사용하므로 상대비교에는 지장 없음).

베이스라인(우리 자신의 프로덕션 CatBoost)은 `code/thirdmodel_common.py::build_split`
(cutoff7=False, holdout=2023, apply_f1=True)로 만든다 — 이 함수는 `code/train.py::main()`을
그대로 재현(F1필터+TE-residual+시즌진행분+coarse pitchmix+tier제거 상태)하므로 season2023
전용 "진짜 프로덕션 레시피" 비교 기준을 제공한다.

사용법: python -m code.experiment_sooyun_season2023
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, embed_dim_for_cardinality,
    fit_preprocessing, apply_preprocessing, fit_quantile_edges, to_tensors,
    train_ensemble, make_bundle, predict_bundle, get_device, compute_bss,
)
from code.blend_model import fit_meta_model
from code.catboost_model import train_catboost
from code.thirdmodel_common import build_split
from code.experiment_sooyun_recipe import (
    build_sooyun_features, SOOYUN_CATBOOST_PARAMS, SOOYUN_ITERATIONS, SOOYUN_CAT_COLS,
    check_memory_or_abort, DATA_DIR, TARGET,
)

SEASON2023_MLP_SEEDS = ENSEMBLE_SEEDS[:7]  # 자원/시간 절약, baseline/후보 공통 재사용


def main():
    device = get_device()
    print(f"device={device}", flush=True)

    print("[1/4] 우리 프로덕션 스플릿 (season2023, F1필터 적용)", flush=True)
    t0 = time.time()
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(
        cutoff7=False, holdout=2023, apply_f1=True,
    )
    check_memory_or_abort("build_split 완료")
    y_train = train_split[TARGET]
    y_val = val_split[TARGET].values
    print(f"[build_split] train={len(train_split)} val={len(val_split)} ({time.time()-t0:.1f}s)", flush=True)

    print("[2/4] 우리 프로덕션 CatBoost (season2023)", flush=True)
    t0 = time.time()
    cat_model, best_iter = train_catboost(
        train_split[cat_feature_cols], y_train, val_split[cat_feature_cols], y_val,
    )
    cat_preds_ours = cat_model.predict_proba(val_split[cat_feature_cols])[:, 1]
    _, _, cat_score_ours = compute_bss(cat_preds_ours, y_val)
    print(f"[우리 CatBoost] season2023 Val Score={cat_score_ours:.2f} best_iter={best_iter} ({time.time()-t0:.1f}s)", flush=True)

    print("[3/4] Fresh MLP (season2023 전용, 7-seed)", flush=True)
    check_memory_or_abort("MLP 학습 전")
    t0 = time.time()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_t,
        seeds=SEASON2023_MLP_SEEDS, embed_dims=embed_dims, bin_edges=bin_edges,
        device=device, verbose=False,
    )
    mlp_bundle = make_bundle(
        members, CAT_COLS, mlp_num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
    )
    print(f"[Fresh MLP season2023] 학습 완료 ({time.time()-t0:.1f}s)", flush=True)

    mlp_preds_ours = predict_bundle(mlp_bundle, val_split, device=device)
    _, _, mlp_score_ours = compute_bss(mlp_preds_ours, y_val)
    print(f"[우리 MLP] season2023 Val Score={mlp_score_ours:.2f}", flush=True)

    _, _, _, blend_score_ours, _ = fit_meta_model(cat_preds_ours, mlp_preds_ours, y_val)
    print(f"[우리 프로덕션 블렌드] season2023 Val Score={blend_score_ours:.2f}", flush=True)

    print("[4/4] 손수연 레시피 CatBoost (season2023, F1 미적용) + 같은 fresh MLP", flush=True)
    check_memory_or_abort("손수연 피처 빌드 전")
    df_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df_raw = df_raw.dropna(subset=[TARGET]).reset_index(drop=True)
    for c in df_raw.select_dtypes(include="float64").columns:
        df_raw[c] = df_raw[c].astype(np.float32)
    for c in df_raw.select_dtypes(include="int64").columns:
        if c not in ("row_id",):
            df_raw[c] = df_raw[c].astype(np.int32)

    train_mask = df_raw["season"] < 2023
    val_mask = df_raw["season"] == 2023

    sooyun_full = build_sooyun_features(df_raw)
    train_sy = sooyun_full.loc[train_mask].reset_index(drop=True)
    val_sy = sooyun_full.loc[val_mask].reset_index(drop=True)
    del sooyun_full

    for c in SOOYUN_CAT_COLS:
        train_sy[c] = train_sy[c].astype(str)
        val_sy[c] = val_sy[c].astype(str)

    drop_cols = ["row_id", TARGET]
    feature_cols = [c for c in train_sy.columns if c not in drop_cols]
    y_val_sy = val_sy[TARGET].values
    assert len(val_sy) == len(val_split), f"val 행 개수 불일치: sooyun={len(val_sy)} ours={len(val_split)}"

    check_memory_or_abort("손수연 CatBoost 학습 전")
    t0 = time.time()
    train_pool = Pool(train_sy[feature_cols], train_sy[TARGET], cat_features=SOOYUN_CAT_COLS)
    val_pool = Pool(val_sy[feature_cols], val_sy[TARGET], cat_features=SOOYUN_CAT_COLS)
    sy_model = CatBoostClassifier(**SOOYUN_CATBOOST_PARAMS, iterations=SOOYUN_ITERATIONS, random_seed=42)
    sy_model.fit(train_pool)
    cat_preds_sy = sy_model.predict_proba(val_pool)[:, 1]
    _, _, cat_score_sy = compute_bss(cat_preds_sy, y_val_sy)
    print(f"[손수연 CatBoost] season2023 Val Score={cat_score_sy:.2f} ({time.time()-t0:.1f}s)", flush=True)

    _, _, _, blend_score_sy, _ = fit_meta_model(cat_preds_sy, mlp_preds_ours, y_val_sy)
    print(f"[손수연+우리MLP 블렌드] season2023 Val Score={blend_score_sy:.2f}", flush=True)

    print("\n" + "=" * 70)
    print("season2023 결과 (fresh 7-seed MLP 공통 재사용)")
    print(f"  우리 프로덕션:   cat={cat_score_ours:.2f}  mlp={mlp_score_ours:.2f}  blend={blend_score_ours:.2f}")
    print(f"  손수연+우리MLP: cat={cat_score_sy:.2f}  mlp={mlp_score_ours:.2f}  blend={blend_score_sy:.2f}")
    print(f"  블렌드 delta: {blend_score_sy - blend_score_ours:+.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
