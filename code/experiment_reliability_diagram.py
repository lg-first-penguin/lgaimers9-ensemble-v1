# code/experiment_reliability_diagram.py
"""프로덕션 모델(open/reference/best_model.pkl)의 cutoff7 검증셋에 대한
Murphy 분해(REL/RES/UNC) + reliability diagram 진단.

BS = REL - RES + UNC, BSS = 1 - REL/UNC + RES/UNC 이므로, 점수를 올리는
레버는 REL(캘리브레이션 오차, 작을수록 좋음)과 RES(분별력, 클수록 좋음)뿐이다.
지금까지의 실험 대부분은 RES를 늘리는 시도(새 피처)였고, REL을 직접 겨냥한
시도는 메타모델 자체 말고는 없었다 — 이 스크립트는 REL이 실제로 어디서
새고 있는지(어떤 확률 구간에서 과신/과소신인지) 확인하기 위한 것이다.

test.py와 동일한 파이프라인으로 cutoff7 검증 스플릿을 재현하고, 그 위에서
CatBoost 단독 / MLP 단독 / 블렌드 세 가지 예측 각각에 대해 동일 개수(quantile)
구간으로 나눠 REL/RES/UNC를 계산한다.
"""
import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import numpy as np
import pandas as pd

from code.train import add_engineered_features, apply_f1_filter, apply_te_residual_features, TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED
from code.mlp_model import compute_bss, predict_bundle
from code.catboost_model import predict_catboost
from code.blend_model import predict_meta
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix

ID_COL = "row_id"
TARGET_COL = "control_success"


def build_val_split():
    DATA_DIR = "./open/data"
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = (train_df['season'] < 2024) | ((train_df['season'] == 2024) & (train_df['game_month'] < 7))
    val_mask = (train_df['season'] == 2024) & (train_df['game_month'] >= 7)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    train_df, _ = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=2024)
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=2024)

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    features = [col for col in train_df.columns if col not in [ID_COL, TARGET_COL]]

    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    val_split = apply_te_residual_features(train_split, val_split, te_prior)

    X_val = val_split[features + TE_RESIDUAL_COLS]
    y_val = val_split[TARGET_COL].values
    return X_val, y_val


def murphy_decompose(preds, y, n_bins=10):
    """분위수(동일 개수) 구간으로 나눈 Murphy 분해. (UNC, REL, RES, bin_table) 반환."""
    n = len(y)
    obar = y.mean()
    unc = obar * (1.0 - obar)

    order = np.argsort(preds)
    preds_sorted = preds[order]
    y_sorted = y[order]

    edges = np.linspace(0, n, n_bins + 1).astype(int)
    rows = []
    rel = 0.0
    res = 0.0
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        if hi <= lo:
            continue
        nk = hi - lo
        pbar_k = preds_sorted[lo:hi].mean()
        obar_k = y_sorted[lo:hi].mean()
        rel += nk * (pbar_k - obar_k) ** 2
        res += nk * (obar_k - obar) ** 2
        rows.append({
            "bin": k,
            "n": nk,
            "pred_range": f"[{preds_sorted[lo]:.3f}, {preds_sorted[hi-1]:.3f}]",
            "mean_pred": pbar_k,
            "actual_rate": obar_k,
            "gap": pbar_k - obar_k,
        })
    rel /= n
    res /= n
    return unc, rel, res, pd.DataFrame(rows)


def report(name, preds, y, n_bins=10):
    unc, rel, res, table = murphy_decompose(preds, y, n_bins=n_bins)
    brier, bss, score = compute_bss(preds, y)
    bs_check = rel - res + unc
    print(f"\n{'='*78}")
    print(f"=== {name} ===")
    print(f"{'='*78}")
    print(f"  UNC={unc:.6f}  REL={rel:.6f}  RES={res:.6f}  (REL-RES+UNC={bs_check:.6f} vs BS={brier:.6f}, diff={bs_check-brier:.2e})")
    print(f"  BSS={bss:.5f}  Score={score:.2f}   [REL/UNC={rel/unc*100000:.2f}pt, RES/UNC={res/unc*100000:.2f}pt of the 100000 scale]")
    print(table.to_string(index=False, formatters={
        "mean_pred": "{:.4f}".format, "actual_rate": "{:.4f}".format, "gap": "{:+.4f}".format,
    }))
    return unc, rel, res, score


def main():
    print("검증 스플릿(cutoff7, 2024 7~10월) 재구성 중...")
    X_val, y_val = build_val_split()
    print(f"검증 표본 수: {len(y_val)}, 실제 성공률(ō)={y_val.mean():.4f}")

    with open("./open/reference/best_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    cat_feature_cols = bundle.get("cat_feature_cols")
    cat_df = X_val[cat_feature_cols] if cat_feature_cols is not None else X_val
    cat_preds = predict_catboost(bundle["catboost_model"], cat_df)
    mlp_preds = predict_bundle(bundle["mlp_bundle"], X_val)
    meta = bundle["meta_model"]
    blend_preds = predict_meta(meta["w_cat"], meta["w_mlp"], meta["intercept"], cat_preds, mlp_preds)

    print(f"\n메타모델 가중치: w_cat={meta['w_cat']:.4f}  w_mlp={meta['w_mlp']:.4f}  intercept={meta['intercept']:.4f}")

    n_bins = 10
    report("CatBoost 단독", cat_preds, y_val, n_bins)
    report("MLP 7-seed 앙상블 단독", mlp_preds, y_val, n_bins)
    unc, rel, res, score = report("블렌드 (프로덕션 최종 예측)", blend_preds, y_val, n_bins)

    print(f"\n{'='*78}")
    print("=== 20구간(finer) 블렌드 재확인 ===")
    print(f"{'='*78}")
    report("블렌드, n_bins=20", blend_preds, y_val, 20)


if __name__ == "__main__":
    main()
