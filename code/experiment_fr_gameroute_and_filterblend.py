# code/experiment_fr_gameroute_and_filterblend.py
"""사용자 제안 2건, cutoff7 CatBoost-only 1차 스크리닝:

1) F1 필터 여부로 갈라 학습한 두 CatBoost를 50:50 평균 블렌드
   - Model A: F1 필터 적용(프로덕션과 동일)
   - Model B: F1 필터 미적용(2022 이하 game_type=='F' 행도 그대로 포함)
   - pred = 0.5*A + 0.5*B
   사전 우려: game_type이 CatBoost 피처중요도 1위이고, F의 성공률이 R 대비 2023년
   기점으로 역전됐다는 게 F1 필터 채택의 근거였다(code/train.py::apply_f1_filter
   docstring). Model B는 정확히 그 오염된 관계를 다시 학습하므로, 단순 평균 블렌드가
   필터의 이득을 희석시킬 가능성이 높다는 게 사전 가설 — 그래도 실측한다.

2) game_type별로 완전히 분리된 두 CatBoost(R전용/F전용)를 학습해 라우팅
   - CatBoost_R: game_type=='R' 학습행만(F1 필터 이후 기준)
   - CatBoost_F: game_type=='F' 학습행만(F1 필터 이후라 표본이 매우 적음 — cutoff7
     기준 42,942행, §57 참고)
   - 검증행은 자기 game_type에 맞는 모델로 예측, 합쳐서 전체 스코어 계산
   사전 우려: F 전용 모델의 학습 표본이 전체의 ~3%뿐이라 이 프로젝트에서 반복 확인된
   "표본 부족 구간 특화 모델은 취약하다"(핵심 교훈 #9, #25) 패턴이 재현될 가능성.

두 아이디어 모두 사전 우려와 무관하게 실측으로 판단한다.

사용법: python -m code.experiment_fr_gameroute_and_filterblend
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.train import apply_f1_filter, add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS
from code.mlp_model import compute_bss
from code.catboost_model import CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.trackman_pitcher_features import merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
THREAD_COUNT = 4
CAT_FEATURES = ["game_type", "base_state"]


def load_base():
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    print(f"[load_base] 완료 (경과 {time.time()-t0:.1f}s)")
    return train_df, df_trm


def build_split(train_df, df_trm, holdout, train_mask_fn, val_mask_fn):
    df = train_df.copy()
    df = merge_coarse_pitchmix(df, df_trm, holdout=holdout)
    train_mask = train_mask_fn(df)
    val_mask = val_mask_fn(df)
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)
    features = [c for c in df.columns if c not in ['row_id', TARGET_COL]]
    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    return train_split, val_split, features


def with_te(source_for_te, train_split, val_split, features):
    te_prior = source_for_te[TARGET_COL].mean()
    tr = apply_te_residual_features(source_for_te, train_split, te_prior)
    va = apply_te_residual_features(source_for_te, val_split, te_prior)
    return tr, va, features + TE_RESIDUAL_COLS


def fit_catboost(X_train, y_train, X_val, y_val, seed=42):
    params = dict(CATBOOST_PARAMS)
    params.update(iterations=MAX_ITERATIONS, thread_count=THREAD_COUNT, random_seed=seed,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(X_val, y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def run_idea1_filter_blend(train_split, val_split, features):
    print("\n--- 아이디어 1: F1필터 적용/미적용 50:50 블렌드 ---")
    y_val = val_split[TARGET_COL].values

    # Model A: F1 필터 적용(프로덕션과 동일)
    tr_a = apply_f1_filter(train_split)
    tr_a, va_a, cols_a = with_te(tr_a, tr_a, val_split, features)
    t0 = time.time()
    model_a = fit_catboost(tr_a[cols_a], tr_a[TARGET_COL].values, va_a[cols_a], y_val)
    pred_a = model_a.predict_proba(va_a[cols_a])[:, 1]
    _, _, score_a = compute_bss(pred_a, y_val)
    print(f"  [Model A: F1필터 적용] n_train={len(tr_a)} score={score_a:.2f} ({time.time()-t0:.1f}s)")

    # Model B: F1 필터 미적용
    tr_b, va_b, cols_b = with_te(train_split, train_split, val_split, features)
    t0 = time.time()
    model_b = fit_catboost(tr_b[cols_b], tr_b[TARGET_COL].values, va_b[cols_b], y_val)
    pred_b = model_b.predict_proba(va_b[cols_b])[:, 1]
    _, _, score_b = compute_bss(pred_b, y_val)
    print(f"  [Model B: F1필터 미적용] n_train={len(tr_b)} score={score_b:.2f} ({time.time()-t0:.1f}s)")

    blend_pred = 0.5 * pred_a + 0.5 * pred_b
    _, _, score_blend = compute_bss(blend_pred, y_val)
    print(f"  [50:50 블렌드] score={score_blend:.2f}  (A 대비 delta={score_blend-score_a:+.2f})")
    return score_a, score_b, score_blend


def run_idea2_gameroute(train_split, val_split, features):
    print("\n--- 아이디어 2: game_type별 완전 분리 라우팅 ---")
    tr = apply_f1_filter(train_split)
    tr, va, cols = with_te(tr, tr, val_split, features)
    y_val = va[TARGET_COL].values

    # baseline: 통합 단일 모델(= 아이디어1의 Model A와 동일 구성)
    t0 = time.time()
    model_unified = fit_catboost(tr[cols], tr[TARGET_COL].values, va[cols], y_val)
    pred_unified = model_unified.predict_proba(va[cols])[:, 1]
    _, _, score_unified = compute_bss(pred_unified, y_val)
    print(f"  [통합 baseline] score={score_unified:.2f} ({time.time()-t0:.1f}s)")

    tr_r = tr[tr["game_type"] == "R"].reset_index(drop=True)
    tr_f = tr[tr["game_type"] == "F"].reset_index(drop=True)
    print(f"  [분리] train R={len(tr_r)}행 F={len(tr_f)}행")

    t0 = time.time()
    model_r = fit_catboost(tr_r[cols], tr_r[TARGET_COL].values, va[cols], y_val)
    model_f = fit_catboost(tr_f[cols], tr_f[TARGET_COL].values, va[cols], y_val)
    print(f"  [분리 학습 완료] ({time.time()-t0:.1f}s)")

    pred_r_all = model_r.predict_proba(va[cols])[:, 1]
    pred_f_all = model_f.predict_proba(va[cols])[:, 1]
    is_f_val = (va["game_type"] == "F").values
    routed_pred = np.where(is_f_val, pred_f_all, pred_r_all)
    _, _, score_routed = compute_bss(routed_pred, y_val)
    print(f"  [라우팅(R전용+F전용)] score={score_routed:.2f}  (통합 대비 delta={score_routed-score_unified:+.2f})")

    for name, mask in [("R subgroup", ~is_f_val), ("F subgroup", is_f_val)]:
        _, _, s_unified = compute_bss(pred_unified[mask], y_val[mask])
        routed_sub = pred_f_all[mask] if name == "F subgroup" else pred_r_all[mask]
        _, _, s_routed = compute_bss(routed_sub, y_val[mask])
        print(f"    {name} (n={mask.sum()}): 통합={s_unified:.2f} 라우팅={s_routed:.2f} delta={s_routed-s_unified:+.2f}")

    return score_unified, score_routed


def main():
    train_df, df_trm = load_base()
    train_split, val_split, features = build_split(
        train_df, df_trm, holdout=2024,
        train_mask_fn=lambda df: (df['season'] < 2024) | ((df['season'] == 2024) & (df['game_month'] < 7)),
        val_mask_fn=lambda df: (df['season'] == 2024) & (df['game_month'] >= 7),
    )
    print(f"[cutoff7] n_train(필터전)={len(train_split)} n_val={len(val_split)}")

    run_idea1_filter_blend(train_split, val_split, features)
    run_idea2_gameroute(train_split, val_split, features)


if __name__ == "__main__":
    main()
