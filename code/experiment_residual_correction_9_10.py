# code/experiment_residual_correction_9_10.py
"""사용자 제안: "9~10월만 경향성이 다르니 이것만 맞추는 서브모델을 만들자."

이 프로젝트에서 9~10월 약세 자체는 이미 여러 번 다뤄졌다 (PROJECT_HISTORY.md §34/§35,
EXPERIMENTS.md §35.5-§35.6):
  - "가을야구 콜업" 가설은 반박됨(신규 등판 투수 수가 다른 달과 비슷).
  - "ABS 도입" 가설은 채택되어 cutoff=7 스플릿(2024 상반기를 학습에 포함)으로 이미 대응.
  - weight=5 표본 재가중 라우팅/게이팅은 표본이 큰 검증셋(2023 10월 n=17,638)에서
    역전(-16.60)해 기각됨.
  - isotonic 순수 재보정(main_pred만으로 recalibration)은 9~10월 fold에서 -13.48로
    손해 — "예측값의 순위만 다시 매핑"하는 방식은 이미 실패가 확인됨(§34).

이번 시도는 메커니즘이 다르다: main_pred 하나만 보는 재보정이 아니라, 상황 피처
(카운트/이닝/li/score_diff/asof_* 등)까지 함께 보고 "9~10월에 국소적으로 나타나는
조건부 편향"을 잡는 2단계 잔차 보정 모델(작은 CatBoostRegressor)이다. isotonic이
실패했다고 이 방식도 실패한다는 보장은 없지만, 같은 프로젝트에서 구간 특화
모델/피처가 반복적으로 실패해왔다는 일반 패턴(PROJECT_HISTORY.md 핵심 교훈)은 여전히
경계 신호다.

1차 시도(--mode other_months, 검증-구간 내 다른 달로 학습)는 두 레짐이 정반대로
갈렸다: cutoff7(프로덕션 레짐)에서 9~10월 단독 -8.42/전체 -2.68로 손해, season==2023
에서는 +269.35/+60.09로 대박이 났지만 corr_fit 표본이 2023쪽만 8개월(19만행) vs
cutoff7쪽 2개월(7.5만행)로 크게 차이 나고, 보정 후 9~10월 점수(666.31)가 2023 연간
평균(564.09)보다도 높아지는 등 과적합 의심 신호가 있어 신뢰하지 않음.

사용자가 재요청한 --mode targeted(기본값): "9~10월일 때만 잔차모델을 학습시켜서
라우팅"하는 방식으로 재구현. train_split 안에 있는 과거 시즌들의 9~10월 행만 잔차모델
학습에 쓴다(다른 달 데이터로 훈련해 9~10월에 적용하는 것이 아니라, 9~10월 자체의
과거 패턴만 학습). 다만 train_split의 9~10월 행은 이미 메인 CatBoost/MLP 학습에
쓰였으므로 그대로 predict하면 in-sample이라 잔차가 인위적으로 작아진다(과최적화된
잔차를 학습하는 꼴) — 이를 피하려고 K-fold OOF를 쓴다: train_split의 9~10월 행을
5-fold로 쪼개, 각 폴드마다 "그 폴드만 뺀 나머지 전체(다른 달 전부 + 9~10월 4/5)"로
CatBoost를 재학습해 그 폴드의 9~10월 행을 예측 → 이렇게 모은 OOF 예측이 "메인 모델이
한 번도 보지 않은 9~10월 행에 대한 정직한 예측"이 되고, 그 잔차(y - oof_pred)로
9~10월 전용 잔차모델을 학습한다. 실제 검증(val_split)의 9~10월 행에는 진짜 메인
블렌드 예측 위에 이 잔차모델을 라우팅해서 적용한다 — 정답은 어디에도 유출되지 않는다.
(OOF 생성 단계는 계산량 때문에 CatBoost만 쓴다 — 메인 블렌드는 CatBoost+MLP 스태킹
이지만 CatBoost가 메타 가중치의 절반 이상을 차지하고, 5-fold x MLP 앙상블까지 돌리면
시간이 너무 오래 걸림.)

사용법:
  python -m code.experiment_residual_correction_9_10 --holdout 2023
  python -m code.experiment_residual_correction_9_10 --cutoff7
  python -m code.experiment_residual_correction_9_10 --cutoff7 --mode other_months  # 1차 시도 재현용
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import KFold, train_test_split

from code.blend_model import fit_meta_model, predict_meta
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, predict_catboost, train_catboost
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, make_bundle, predict_bundle,
    to_tensors, train_ensemble,
)
from code.train import add_engineered_features
from code.trackman_pitcher_features import clean_trackman, add_all_tiers
from code.experiment_coarse_pitchmix import merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]
TIER_FEED = {"a": "mlp"}  # 현재 프로덕션 구성
TARGET_MONTHS = {9, 10}


def build_split(holdout, cutoff7, apply_f1=True):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm_full = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TIER_FEED), holdout=holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TIER_FEED[tier] == "cat" for c in cols]

    df_trm_pmix = df_trm_full[["season", "balls_before", "strikes_before", "pitcher_hand",
                                "batter_hand", "pitch_type_group"]].copy()
    hand_map = {"Left": 1, "Right": 2}
    df_trm_pmix["pitcher_hand"] = df_trm_pmix["pitcher_hand"].map(hand_map)
    df_trm_pmix["batter_hand"] = df_trm_pmix["batter_hand"].map(hand_map)
    df = merge_coarse_pitchmix(df, df_trm_pmix, holdout=holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    trk_all_cols = trk_mlp_cols + trk_cat_cols
    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in trk_all_cols]
    features = base_features + trk_all_cols
    cat_features = base_features + trk_cat_cols
    mlp_num_cols = [c for c in base_features if c not in CAT_COLS] + trk_mlp_cols

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)

    if apply_f1:
        before = len(train_split)
        train_split = train_split[~((train_split["game_type"] == "F") & (train_split["season"] <= 2022))].reset_index(drop=True)
        print(f"[F1 필터] {before} -> {len(train_split)}행")

    print(f"훈련: {len(train_split)}행 | 검증: {len(val_split)}행 (season 구성: {sorted(val_split['game_month'].unique())})")
    return train_split, val_split, features, cat_features, mlp_num_cols


def train_main_blend(train_split, val_split, cat_features, mlp_num_cols):
    device = get_device()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    y_val_np = val_proc[TARGET_COL].values

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
        seeds=SCREEN_SEEDS, device=device,
    )
    print(f"MLP({len(SCREEN_SEEDS)}-seed) 학습 완료 ({time.time() - t0:.1f}s)")
    mlp_bundle = make_bundle(members, CAT_COLS, mlp_num_cols, cat_dims, embed_dims,
                              cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges)
    mlp_val_preds = predict_bundle(mlp_bundle, val_split, device=device)

    X_train_raw, y_train_raw = train_split[cat_features], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[cat_features], val_split[TARGET_COL].values
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    print(f"CatBoost 완료 (best_iteration={catboost_best_iteration}, {time.time() - t0:.1f}s)")
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)

    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
    w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)
    blend_val_preds = predict_meta(w_cat, w_mlp, intercept, cat_val_preds, mlp_val_preds)
    print(f"[메인 블렌드 전체 검증] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f} | Blend={blend_score:.2f}")
    return blend_val_preds, y_val_raw, catboost_best_iteration


def run_residual_correction(val_split, blend_val_preds, y_val_raw, cat_features):
    is_target = val_split["game_month"].isin(TARGET_MONTHS).values
    corr_fit_idx = np.where(~is_target)[0]
    corr_eval_idx = np.where(is_target)[0]
    print(f"\ncorrection-fit(9~10월 제외): {len(corr_fit_idx)}행 | correction-eval(9~10월, 목표): {len(corr_eval_idx)}행")

    baseline_eval_score = compute_bss(blend_val_preds[corr_eval_idx], y_val_raw[corr_eval_idx])[2]
    print(f"[보정 전] 9~10월 단독 Blend score: {baseline_eval_score:.2f}")

    residual = y_val_raw[corr_fit_idx] - blend_val_preds[corr_fit_idx]
    X_fit = val_split.loc[corr_fit_idx, cat_features].copy()
    X_fit["main_pred"] = blend_val_preds[corr_fit_idx]
    X_eval = val_split.loc[corr_eval_idx, cat_features].copy()
    X_eval["main_pred"] = blend_val_preds[corr_eval_idx]

    X_sub_tr, X_sub_va, r_sub_tr, r_sub_va = train_test_split(X_fit, residual, test_size=0.2, random_state=42)
    train_pool = Pool(data=X_sub_tr, label=r_sub_tr, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_sub_va, label=r_sub_va, cat_features=CAT_FEATURES)
    resid_model = CatBoostRegressor(
        iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=10,
        loss_function="RMSE", eval_metric="RMSE", random_seed=42,
        early_stopping_rounds=50, verbose=False,
    )
    resid_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"[잔차모델] best_iteration={resid_model.get_best_iteration()}")

    pred_resid = resid_model.predict(X_eval)
    corrected_preds = np.clip(blend_val_preds[corr_eval_idx] + pred_resid, 0.0, 1.0)
    corrected_eval_score = compute_bss(corrected_preds, y_val_raw[corr_eval_idx])[2]
    print(f"[보정 후] 9~10월 단독 Blend+잔차보정 score: {corrected_eval_score:.2f}")
    print(f"delta(보정후-보정전): {corrected_eval_score - baseline_eval_score:+.2f}")

    full_preds = blend_val_preds.copy()
    full_preds[corr_eval_idx] = corrected_preds
    full_baseline_score = compute_bss(blend_val_preds, y_val_raw)[2]
    full_corrected_score = compute_bss(full_preds, y_val_raw)[2]
    print(f"\n[검증셋 전체] 보정 전 Blend={full_baseline_score:.2f} | 보정 후(9~10월만 교체)={full_corrected_score:.2f} | delta={full_corrected_score - full_baseline_score:+.2f}")


def build_910_oof_residual(train_split, cat_features, catboost_best_iteration, n_splits=5, seed=42):
    """train_split 안의 과거 시즌 9~10월 행에 대해서만 K-fold OOF 잔차를 만든다.
    각 폴드는 "그 폴드의 9~10월 행만 제외한 나머지 전체"(다른 달 전부 + 9~10월 4/5)로
    학습해, 메인 모델이 한 번도 보지 않은 9~10월 행에 대한 정직한 예측을 얻는다."""
    is_target = train_split["game_month"].isin(TARGET_MONTHS).values
    idx_910 = np.where(is_target)[0]
    idx_other = np.where(~is_target)[0]
    y = train_split[TARGET_COL].values
    oof_pred = np.zeros(len(idx_910))

    params = dict(CATBOOST_PARAMS)
    params["iterations"] = catboost_best_iteration
    params["verbose"] = False

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_pos, va_pos) in enumerate(kf.split(idx_910)):
        fold_train_910 = idx_910[tr_pos]
        fold_val_910 = idx_910[va_pos]
        fold_train_idx = np.concatenate([idx_other, fold_train_910])

        X_fold_train = train_split.loc[fold_train_idx, cat_features]
        y_fold_train = y[fold_train_idx]
        X_fold_val = train_split.loc[fold_val_910, cat_features]

        model = CatBoostClassifier(**params)
        pool = Pool(data=X_fold_train, label=y_fold_train, cat_features=CAT_FEATURES)
        model.fit(pool)
        oof_pred[va_pos] = predict_catboost(model, X_fold_val)
        print(f"  [OOF fold {fold + 1}/{n_splits}] train={len(fold_train_idx)}행(9~10월 {len(fold_train_910)}행 포함) "
              f"| 예측한 9~10월={len(fold_val_910)}행")

    residual_910 = y[idx_910] - oof_pred
    X_910 = train_split.loc[idx_910, cat_features].copy()
    X_910["main_pred"] = oof_pred
    return X_910, residual_910


def run_residual_correction_targeted(train_split, val_split, blend_val_preds, y_val_raw, cat_features,
                                      catboost_best_iteration, recency_weight=1.0):
    is_target_val = val_split["game_month"].isin(TARGET_MONTHS).values
    val_910_idx = np.where(is_target_val)[0]
    baseline_score = compute_bss(blend_val_preds[val_910_idx], y_val_raw[val_910_idx])[2]
    print(f"\n[보정 전] 9~10월 단독 Blend score: {baseline_score:.2f}")

    print("\n[9~10월 전용 OOF 잔차 생성 - KFold]")
    X_910, residual_910 = build_910_oof_residual(train_split, cat_features, catboost_best_iteration)
    print(f"9~10월 전용 학습 표본: {len(X_910)}행 (train_split 내 과거 시즌들의 9~10월만)")

    weight_910 = np.ones(len(X_910))
    if recency_weight != 1.0:
        # 실제 2024년 9~10월 데이터는 검증용으로 홀드아웃돼 있어 학습에 못 씀 — 학습 가능한
        # 과거 9~10월(2019~2023) 중 2024에 가장 가까운 season(=max)에 가중치를 준다.
        nearest_season = X_910["season"].max()
        is_nearest = (X_910["season"].values == nearest_season)
        weight_910 = np.where(is_nearest, recency_weight, 1.0)
        print(f"[재가중] season=={nearest_season}(2024에 가장 가까움) 행 {is_nearest.sum()}개에 weight={recency_weight} 적용")

    X_sub_tr, X_sub_va, r_sub_tr, r_sub_va, w_sub_tr, w_sub_va = train_test_split(
        X_910, residual_910, weight_910, test_size=0.2, random_state=42)
    train_pool = Pool(data=X_sub_tr, label=r_sub_tr, weight=w_sub_tr, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_sub_va, label=r_sub_va, weight=w_sub_va, cat_features=CAT_FEATURES)
    resid_model = CatBoostRegressor(
        iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=10,
        loss_function="RMSE", eval_metric="RMSE", random_seed=42,
        early_stopping_rounds=50, verbose=False,
    )
    resid_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"[잔차모델] best_iteration={resid_model.get_best_iteration()}")

    X_eval = val_split.loc[val_910_idx, cat_features].copy()
    X_eval["main_pred"] = blend_val_preds[val_910_idx]
    pred_resid = resid_model.predict(X_eval)
    corrected_preds = np.clip(blend_val_preds[val_910_idx] + pred_resid, 0.0, 1.0)
    corrected_score = compute_bss(corrected_preds, y_val_raw[val_910_idx])[2]
    print(f"[보정 후, 라우팅] 9~10월 단독 score: {corrected_score:.2f}")
    print(f"delta(보정후-보정전): {corrected_score - baseline_score:+.2f}")

    full_preds = blend_val_preds.copy()
    full_preds[val_910_idx] = corrected_preds
    full_baseline = compute_bss(blend_val_preds, y_val_raw)[2]
    full_corrected = compute_bss(full_preds, y_val_raw)[2]
    print(f"\n[검증셋 전체] 보정 전={full_baseline:.2f} | 보정 후(라우팅)={full_corrected:.2f} | delta={full_corrected - full_baseline:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--mode", choices=["targeted", "other_months"], default="targeted")
    parser.add_argument("--recency-weight", type=float, default=1.0,
                         help="9~10월 학습 표본 중 2024에 가장 가까운 season(2023)에 줄 가중치 배수 (1.0=재가중 없음)")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    label = "cutoff7" if args.cutoff7 else f"holdout={holdout}"
    print(f"=== 잔차 보정 실험 ({label}, mode={args.mode}, recency_weight={args.recency_weight}) ===")

    train_split, val_split, features, cat_features, mlp_num_cols = build_split(holdout, args.cutoff7)
    blend_val_preds, y_val_raw, catboost_best_iteration = train_main_blend(train_split, val_split, cat_features, mlp_num_cols)
    if args.mode == "targeted":
        run_residual_correction_targeted(train_split, val_split, blend_val_preds, y_val_raw, cat_features,
                                          catboost_best_iteration, recency_weight=args.recency_weight)
    else:
        run_residual_correction(val_split, blend_val_preds, y_val_raw, cat_features)


if __name__ == "__main__":
    main()
