# code/experiment_mlp_ensemble_weighting.py
"""MLP 고도화 1단계: 7-seed 앙상블을 단순 평균 대신 "가중 결합"으로 바꿔보는 실험.
재학습이 필요 없다 — 이미 학습된 open/reference/best_model.pkl의 7개 멤버를 그대로 로드해
val split에 대한 멤버별(seed별) 예측 7개를 뽑고, 이를 재조합하는 방법만 바꾼다.

재조합 방법(로지스틱 회귀, 7-멤버 예측 -> y)은 val split 자체에다 fit하면 "그 val을 보고
맞춘 가중치로 그 val을 채점"하는 낙관 편향이 생기므로, val을 3-fold로 나눠 OOF(각 fold는
나머지 2-fold로 학습한 가중치로 예측)로 전체 val 예측을 만든 뒤 채점한다 — 이것이 이번
실험의 유일한 공정성 장치.

두 콤보 모두 CatBoost는 그대로 두고(cat_preds 고정), 2-피처 메타모델(w_cat/w_mlp/intercept)만
mlp_pred를 바꿔가며 재fit해 블렌드 점수를 비교한다(현재 프로덕션과 동일하게 val 전체로 fit
— 메타모델 fit 자체를 OOF로 바꾸는 실험은 별도 5단계 과제).

사용법:
  python -m code.experiment_mlp_ensemble_weighting
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost
from code.mlp_model import TabularMLP, apply_preprocessing, compute_bss, get_device, to_tensors
from code.train import (
    TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED, add_engineered_features, apply_f1_filter,
    apply_te_residual_features,
)
from code.trackman_pitcher_features import add_all_tiers, clean_trackman, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
REF_MODEL_PATH = "./open/reference/best_model.pkl"
N_OOF_FOLDS = 3
SEED = 42


def build_val_split():
    """code/test.py와 완전히 동일한 로직으로 cutoff7 val split을 재구성한다."""
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = (train_df["season"] < 2024) | ((train_df["season"] == 2024) & (train_df["game_month"] < 7))
    val_mask = (train_df["season"] == 2024) & (train_df["game_month"] >= 7)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    train_df, _ = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=2024)
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=2024)

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    features = [c for c in train_df.columns if c not in ["row_id", TARGET_COL]]
    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    val_split = apply_te_residual_features(train_split, val_split, te_prior)

    X_val = val_split[features + TE_RESIDUAL_COLS]
    y_val = val_split[TARGET_COL].values
    return X_val, y_val


def predict_members(mlp_bundle, df, device=None):
    """predict_ensemble(code/mlp_model.py)의 "평균 내기 직전" 버전 — 멤버별 예측을
    (n_rows, n_members) 행렬로 반환한다."""
    device = device or get_device()
    df_proc = apply_preprocessing(
        df, mlp_bundle["cat_cols"], mlp_bundle["num_cols"],
        mlp_bundle["cat_encoder"], mlp_bundle["num_imputer"], mlp_bundle["num_scaler"],
    )
    X_cat, X_num = to_tensors(df_proc, mlp_bundle["cat_cols"], mlp_bundle["num_cols"])
    bin_edges = mlp_bundle.get("bin_edges")
    quantile_d = mlp_bundle.get("quantile_d", 8)

    preds = []
    with torch.no_grad():
        for member in mlp_bundle["members"]:
            model = TabularMLP(
                num_numeric_feats=len(mlp_bundle["num_cols"]), cat_dims=mlp_bundle["cat_dims"],
                embed_dims=mlp_bundle["embed_dims"], bin_edges=bin_edges, quantile_d=quantile_d,
            ).to(device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            preds.append(model(X_cat.to(device), X_num.to(device)).cpu().numpy())
    return np.column_stack(preds)  # (n_rows, n_members)


def oof_logistic_combine(member_preds, y_val, n_folds=N_OOF_FOLDS, seed=SEED):
    """member_preds(n_rows, n_members) -> y_val 로지스틱 회귀를 n_folds OOF로 fit/predict."""
    n = len(y_val)
    oof = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, holdout_idx in kf.split(member_preds):
        model = LogisticRegression(max_iter=1000)
        model.fit(member_preds[train_idx], y_val[train_idx])
        oof[holdout_idx] = model.predict_proba(member_preds[holdout_idx])[:, 1]
    return oof


def main():
    print("[1/3] val split 재구성 + reference 번들 로드")
    X_val, y_val = build_val_split()
    with open(REF_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    mlp_bundle = bundle["mlp_bundle"]
    n_members = len(mlp_bundle["members"])
    print(f"  val: {len(y_val)}행 | MLP 멤버 수: {n_members}")

    cat_feature_cols = bundle.get("cat_feature_cols")
    cat_df = X_val[cat_feature_cols] if cat_feature_cols is not None else X_val
    cat_preds = predict_catboost(bundle["catboost_model"], cat_df)
    cat_score = compute_bss(cat_preds, y_val)[2]
    print(f"  CatBoost 단독 Score={cat_score:.2f} (참고, 변경 없음)")

    print("[2/3] MLP 멤버별 예측 추출")
    device = get_device()
    member_preds = predict_members(mlp_bundle, X_val, device=device)  # (n, 7)

    print("[3/3] 재조합 방법 비교")
    results = {}

    # baseline: 현재 프로덕션과 동일한 단순 평균
    mean_pred = member_preds.mean(axis=1)
    mean_mlp_score = compute_bss(mean_pred, y_val)[2]
    w_cat, w_mlp, intercept, mean_blend_score, _ = fit_meta_model(cat_preds, mean_pred, y_val)
    results["baseline(단순평균)"] = (mean_mlp_score, mean_blend_score)
    print(f"  [baseline(단순평균)] MLP={mean_mlp_score:.2f} | Blend={mean_blend_score:.2f}")

    # 참고용(낙관 편향 있음): val 전체로 fit한 로지스틱 가중 결합 — OOF와 비교해 편향 크기 확인용
    full_fit_model = LogisticRegression(max_iter=1000)
    full_fit_model.fit(member_preds, y_val)
    full_fit_pred = full_fit_model.predict_proba(member_preds)[:, 1]
    full_fit_mlp_score = compute_bss(full_fit_pred, y_val)[2]
    _, _, _, full_fit_blend_score, _ = fit_meta_model(cat_preds, full_fit_pred, y_val)
    results["참고: val전체fit(낙관편향)"] = (full_fit_mlp_score, full_fit_blend_score)
    print(f"  [참고: val전체fit(낙관편향)] MLP={full_fit_mlp_score:.2f} | Blend={full_fit_blend_score:.2f} "
          f"| 가중치={np.round(full_fit_model.coef_[0], 3).tolist()}")

    # 진짜 비교 대상: OOF 로지스틱 가중 결합 — KFold 시드 5개로 강건성 확인
    # (재학습 없이 예측만 재조합하는 실험이라 이 정도 반복은 사실상 공짜)
    base_mlp, base_blend = results["baseline(단순평균)"]
    oof_mlp_deltas, oof_blend_deltas = [], []
    for kfold_seed in [42, 1, 2, 3, 4]:
        oof_pred = oof_logistic_combine(member_preds, y_val, seed=kfold_seed)
        oof_mlp_score = compute_bss(oof_pred, y_val)[2]
        _, _, _, oof_blend_score, _ = fit_meta_model(cat_preds, oof_pred, y_val)
        oof_mlp_deltas.append(oof_mlp_score - base_mlp)
        oof_blend_deltas.append(oof_blend_score - base_blend)
        print(f"  [OOF 가중결합(3-fold, kfold_seed={kfold_seed})] MLP={oof_mlp_score:.2f} ({oof_mlp_score-base_mlp:+.2f}) | Blend={oof_blend_score:.2f} ({oof_blend_score-base_blend:+.2f})")

    print(f"\n{'='*70}\nOOF delta 5회 평균: MLP {np.mean(oof_mlp_deltas):+.2f} (std {np.std(oof_mlp_deltas):.2f}) | "
          f"Blend {np.mean(oof_blend_deltas):+.2f} (std {np.std(oof_blend_deltas):.2f})\n{'='*70}")


if __name__ == "__main__":
    main()
