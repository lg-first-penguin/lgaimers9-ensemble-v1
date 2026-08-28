# code/experiment_yudam_full_replica.py
"""사용자 질문: "조유담님 모델과의 격차는 뭐가 원인이야? 피처들간의 상호작용이
다른건가?" -- 이 프로젝트가 지금까지 조유담과의 구조적 차이를 하나씩(CatBoost
하이퍼파라미터만, 트랙맨64만, quantile 제거만, reverse_rate만) 이식해봤지만 전부
개별로는 격차를 설명 못 했다(§78-81, `code/experiment_teammate_combo.py`가 이미
하이퍼파라미터+트랙맨64 둘만 조합해서 테스트했었고 그것도 기각). 이번엔 **그의 실제
전체 조합**(F1필터+TrackA+coarse pitchmix+시즌진행분[전부 우리도 이미 있음] +
reverse_rate 시즌분해 + 트랙맨64 + 그의 CatBoost 하이퍼파라미터 + MLP quantile
PLE 제거)을 하나의 번들로 우리 자신의 cutoff7/season2023 검증 프레임워크에서
재현해, "상호작용 때문에 개별 이식으로는 안 보이던 이득이 조합에서는 나타나는가"를
직접 테스트한다.

**2026-08-27 수정: v1(cutoff7 -20.66/season2023 +11.04)이 두 가지를 놓치고 있었음이
그의 실제 저장소(`teammate/yudam/`)를 직접 코드 대조해서 드러남**:
  1. CatBoost 하이퍼파라미터가 STALE했다 -- `experiment_teammate_catboost_hparams.py`의
     `TEAMMATE_PARAMS`는 "1059.72 레시피" 시절(시즌진행분+TrackA까지만 반영된 상태)에
     튜닝된 구버전(`depth=7, lr=0.02034, l2=14.806, min_data_in_leaf=35, iterations=1462`)
     인데, 그의 실제 최신(~1090/1092 실전) 번들은 reverse_rate+same_hand까지 전부 반영된
     피처셋으로 2026-08-26 재탐색한 신버전(`depth=7, lr=0.046773, l2=19.391490,
     random_strength=8.254100, bagging_temperature=0.127747, border_count=179,
     min_data_in_leaf=1, best_iteration=684`, `teammate/yudam/EXPERIMENTS.md` 972-994줄,
     `full_retrain_blend_f1.py`에 실제로 박혀 있는 값)를 쓴다. 학습률 2배 이상, 리프
     정규화(min_data_in_leaf 35->1)가 사실상 해제, iteration은 절반 -- 완전히 다른
     정규화 레짐.
  2. same_hand/same_hand_advantage(MLP 전용)가 아예 빠져 있었다 -- `build_split()`
     (`code/thirdmodel_common.py`)이 이 피처를 포함하지 않는데, v1이 그 위에 그대로
     쌓기만 하고 수동으로도 추가하지 않았음. 그의 `full_retrain_blend_f1.py` 127줄
     `add_same_hand(train_df)`가 실제 ~1090/1092 번들에 들어 있고, 그의 자체 dual-fold
     검증에서 MLP 전용으로 평균 +28 수준(+6.72/+49.81)의 꽤 큰 양의 신호로 재확인된
     피처다. 참고로 이 피처 자체는 우리 프로덕션(`code/train.py`)에는 이미 MLP전용으로
     들어가 있음 -- 빠진 건 이 재현 실험 스크립트뿐이었다.
  아래 `TEAMMATE_PARAMS_V2`/`TEAMMATE_ITERATIONS_V2`와 same_hand 적용으로 수정.

기존 재사용 컴포넌트:
  - F1필터+TrackA+시즌진행분+coarse pitchmix: `code/thirdmodel_common.py::build_split`
  - reverse_rate 시즌분해: `code/experiment_reverse_rate_season_progression.py`
  - 트랙맨64(season 조인, 실전 100% 상수 위험 이미 문서화됨): `code/experiment_teammate_trackman64.py`
  - 조유담 CatBoost 하이퍼파라미터: 아래 `TEAMMATE_PARAMS_V2` (v1은 stale이라 폐기)
  - MLP quantile 제거(raw-concat, bin_edges=None): `code/mlp_model.py::train_ensemble`가 이미 지원

트랙맨64/reverse_rate 둘 다 CatBoost와 MLP 양쪽에 먹인다 -- 조유담의 실제 라우팅
(`full_retrain_blend_f1.py`: `num_cols = [c for c in all_cols if c not in CAT_COLS
and c not in TE_RESIDUAL_COLS]`, 즉 TE-residual만 CatBoost 전용이고 나머지는 전부
MLP에도 들어감)과 동일하게 맞춘다.

사용법: python -m code.experiment_yudam_full_replica
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import CAT_FEATURES
from code.blend_model import fit_meta_model
from code.thirdmodel_common import build_split, TARGET_COL
from code.trackman_pitcher_features import clean_trackman
from code.train import apply_same_hand, SAME_HAND_COLS
from code.experiment_teammate_trackman64 import merge_trackman64, TM64_COLS
from code.experiment_reverse_rate_season_progression import (
    build_reverse_season_end_lookup, apply_reverse_season_progression, REV_COLS,
)
from code.experiment_sooyun_recipe import check_memory_or_abort

DATA_DIR = "./open/data"
SEEDS = [42, 123, 7]
MLP_SEEDS = [42, 123, 7]

# 2026-08-26 조유담 실제 재탐색값(teammate/yudam/EXPERIMENTS.md 972-994줄,
# full_retrain_blend_f1.py에 실제로 반영된 값) -- reverse_rate+same_hand까지
# 전부 포함된 피처셋 기준으로 재탐색된 "현재" 하이퍼파라미터. v1이 썼던
# `experiment_teammate_catboost_hparams.py::TEAMMATE_PARAMS`는 그보다 이전
# (1059.72 레시피 시절, TrackA+시즌진행분까지만) 튜닝값이라 STALE -- 폐기.
TEAMMATE_PARAMS_V2 = dict(
    depth=7,
    learning_rate=0.046773,
    l2_leaf_reg=19.391490,
    random_strength=8.254100,
    bagging_temperature=0.127747,
    border_count=179,
    min_data_in_leaf=1,
    bootstrap_type="Bayesian",
    loss_function="Logloss",
    eval_metric="BrierScore",
    verbose=False,
)
TEAMMATE_ITERATIONS_V2 = 684 + 50  # best_iteration + buffer, 그의 full_retrain 관례와 동일


def build_yudam_features(train_split, val_split, cat_feature_cols, mlp_num_cols, df_trm_clean, trm_holdout):
    lookup = build_reverse_season_end_lookup(train_split)
    train_split = apply_reverse_season_progression(train_split, lookup)
    val_split = apply_reverse_season_progression(val_split, lookup)

    train_split, train_match = merge_trackman64(train_split, df_trm_clean, holdout=trm_holdout)
    val_split, val_match = merge_trackman64(val_split, df_trm_clean, holdout=trm_holdout)
    print(f"  트랙맨64 매칭률: train={train_match:.1%} val={val_match:.1%}", flush=True)

    # same_hand/same_hand_advantage: MLP 전용(v1에서 누락됐던 부분, CatBoost엔 안 먹임 --
    # 그의 실제 라우팅과 동일)
    train_split = apply_same_hand(train_split)
    val_split = apply_same_hand(val_split)

    cat_cols_yudam = cat_feature_cols + REV_COLS + TM64_COLS
    mlp_cols_yudam = mlp_num_cols + REV_COLS + TM64_COLS + SAME_HAND_COLS
    return train_split, val_split, cat_cols_yudam, mlp_cols_yudam


def catboost_yudam_score(train_split, val_split, cat_cols, seed):
    params = dict(TEAMMATE_PARAMS_V2)
    params["iterations"] = TEAMMATE_ITERATIONS_V2
    params["random_seed"] = seed
    model = CatBoostClassifier(**params)
    train_pool = Pool(train_split[cat_cols], train_split[TARGET_COL], cat_features=CAT_FEATURES)
    model.fit(train_pool)
    preds = model.predict_proba(val_split[cat_cols])[:, 1]
    y_val = val_split[TARGET_COL].values
    return preds, compute_bss(preds, y_val)[2]


def mlp_yudam_score(train_split, val_split, mlp_cols, device, seeds):
    """quantile PLE 없이(bin_edges=None, raw-concat) -- 조유담 MLP와 동일."""
    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, mlp_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, mlp_cols, TARGET_COL)
    X_va_cat, X_va_num, y_va = to_tensors(va_proc, CAT_COLS, mlp_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric_feats=len(mlp_cols),
        embed_dims=embed_dims, bin_edges=None,
        X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_va, seeds=seeds, device=device, verbose=False,
    )
    preds = predict_ensemble(members, cat_dims, len(mlp_cols), embed_dims, X_va_cat, X_va_num, bin_edges=None, device=device)
    y_val = y_va.numpy()
    return preds, compute_bss(preds, y_val)[2]


def run_regime(cutoff7, holdout, device, df_trm_clean):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (조유담 전체 조합 재현) ===\n{'='*70}", flush=True)

    check_memory_or_abort(f"{label} build_split 전")
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    trm_holdout = 2024 if cutoff7 else holdout

    check_memory_or_abort(f"{label} 조유담 피처 빌드 전")
    train_y, val_y, cat_cols_y, mlp_cols_y = build_yudam_features(
        train_split, val_split, cat_feature_cols, mlp_num_cols, df_trm_clean, trm_holdout,
    )
    y_val = val_y[TARGET_COL].values
    print(f"[{label}] CatBoost 피처={len(cat_cols_y)}개 MLP 피처={len(mlp_cols_y)}개", flush=True)

    check_memory_or_abort(f"{label} MLP(quantile없음) 학습 전")
    t0 = time.time()
    mlp_preds, mlp_score = mlp_yudam_score(train_y, val_y, mlp_cols_y, device, MLP_SEEDS)
    print(f"[{label}][MLP quantile없음, {len(MLP_SEEDS)}-seed] Val Score={mlp_score:.2f} ({time.time()-t0:.1f}s)", flush=True)

    rows = []
    for seed in SEEDS:
        check_memory_or_abort(f"{label} CatBoost seed={seed} 학습 전")
        t0 = time.time()
        cat_preds, cat_score = catboost_yudam_score(train_y, val_y, cat_cols_y, seed)
        _, _, _, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val)
        print(f"[{label}][seed={seed}] cat={cat_score:.2f} blend={blend_score:.2f} ({time.time()-t0:.1f}s)", flush=True)
        rows.append((seed, cat_score, blend_score))

    avg_c = np.mean([r[1] for r in rows])
    avg_b = np.mean([r[2] for r in rows])
    print(f"[{label}] 평균: cat={avg_c:.2f} mlp={mlp_score:.2f} blend={avg_b:.2f}", flush=True)
    return {"rows": rows, "mlp_score": mlp_score, "avg_cat": avg_c, "avg_blend": avg_b}


def main():
    device = get_device()
    print(f"device={device}", flush=True)

    print("[트랙맨 로드+클렌징]", flush=True)
    t0 = time.time()
    df_trm_full = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    print(f"완료 ({time.time()-t0:.1f}s)", flush=True)
    check_memory_or_abort("트랙맨 로드 직후")

    result_cutoff7 = run_regime(cutoff7=True, holdout=2024, device=device, df_trm_clean=df_trm_clean)
    result_2023 = run_regime(cutoff7=False, holdout=2023, device=device, df_trm_clean=df_trm_clean)

    print("\n" + "=" * 70)
    print("조유담 전체 조합 재현 -- 최종 요약")
    print(f"  cutoff7:    cat={result_cutoff7['avg_cat']:.2f}  mlp={result_cutoff7['mlp_score']:.2f}  blend={result_cutoff7['avg_blend']:.2f}")
    print(f"  season2023: cat={result_2023['avg_cat']:.2f}  mlp={result_2023['mlp_score']:.2f}  blend={result_2023['avg_blend']:.2f}")
    print("  비교 기준 -- 우리 프로덕션: cutoff7 blend=753.37 / season2023 blend=734.94")
    print("=" * 70)


if __name__ == "__main__":
    main()
