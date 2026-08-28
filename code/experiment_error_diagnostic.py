# code/experiment_error_diagnostic.py
"""프로덕션 reference 블렌드(open/reference/best_model.pkl, 실전 1027.54 확정 구성)의
cutoff7 검증셋 예측을 상황 변수별로 쪼개서, 어디서 모델이 가장 많이 틀리는지(캘리브레이션
갭)와 어디서 baseline 대비 여전히 여지(group BSS)가 남아있는지 진단한다. 새 피처를
찍어보는 대신, 실제 잔차가 어디 몰려있는지부터 데이터로 확인하는 게 목적 — 다음 피처
아이디어를 근거 있게 좁히기 위한 진단 단계 (code/test.py의 검증 스플릿 재구성 로직을
그대로 재사용해 실제 채점에 쓰이는 모델·데이터와 동일 조건을 보장한다).

사용법:
  python -m code.experiment_error_diagnostic
"""
import os
import pickle

import numpy as np
import pandas as pd

from code.train import add_engineered_features, apply_f1_filter, apply_te_residual_features, TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED
from code.blend_model import predict_blend_bundle
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
REF_MODEL_PATH = "./open/reference/best_model.pkl"


def build_val_split():
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

    features = [col for col in train_df.columns if col not in ["row_id", TARGET_COL]]
    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    val_split = apply_te_residual_features(train_split, val_split, te_prior)

    X_val = val_split[features + TE_RESIDUAL_COLS]
    y_val = val_split[TARGET_COL].values
    return val_split, X_val, y_val


def qbin(series, q=4, label_prefix=""):
    try:
        return pd.qcut(series, q=q, duplicates="drop").astype(str)
    except ValueError:
        return series.astype(str)


def group_report(name, group_key, y_val, pred, min_n=500):
    dfe = pd.DataFrame({"g": group_key, "y": y_val, "p": pred})
    rows = []
    for g, sub in dfe.groupby("g"):
        n = len(sub)
        if n < min_n:
            continue
        r = sub["y"].mean()
        mean_pred = sub["p"].mean()
        brier = ((sub["p"] - sub["y"]) ** 2).mean()
        baseline_brier = r * (1 - r)
        bss = max(0.0, 1 - brier / baseline_brier) if baseline_brier > 0 else np.nan
        rows.append({
            "axis": name, "group": str(g), "n": n, "actual_rate": r,
            "mean_pred": mean_pred, "calib_gap": mean_pred - r,
            "brier": brier, "group_score": bss * 100000,
        })
    return pd.DataFrame(rows)


def main():
    print("검증 스플릿 재구성 중 (code/test.py와 동일 로직)...")
    val_split, X_val, y_val = build_val_split()

    with open(REF_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    pred = predict_blend_bundle(bundle, X_val)

    overall_r = y_val.mean()
    overall_brier = ((pred - y_val) ** 2).mean()
    overall_score = max(0, 100000 * (1 - overall_brier / (overall_r * (1 - overall_r))))
    print(f"전체: n={len(y_val)}, r={overall_r:.4f}, Brier={overall_brier:.6f}, Score={overall_score:.2f}\n")

    axes = {
        "game_month": val_split["game_month"],
        "inning": val_split["inning"].clip(upper=9),
        "balls_strikes": val_split["balls_before"].astype(str) + "-" + val_split["strikes_before"].astype(str),
        "outs_before": val_split["outs_before"],
        "base_state": val_split["base_state"],
        "num_runners_on": val_split["num_runners_on"],
        "top_bottom": val_split["top_bottom"],
        "game_type": val_split["game_type"],
        "pitcher_hand": val_split["pitcher_hand"],
        "batter_hand": val_split["batter_hand"],
        "li_bucket": qbin(val_split["li"], q=5),
        "score_diff_pitcher_team_bucket": qbin(val_split["score_diff_pitcher_team"], q=5),
        "asof_pitcher_n_bucket(경력)": qbin(val_split["asof_pitcher_n"], q=5),
        "asof_batter_n_bucket(경력)": qbin(val_split["asof_batter_n"], q=5),
        "pitcher_season_n_bucket(시즌누적)": qbin(val_split["pitcher_season_n"], q=5),
    }

    all_reports = []
    for name, key in axes.items():
        rep = group_report(name, key, y_val, pred)
        all_reports.append(rep)
    full = pd.concat(all_reports, ignore_index=True)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 200)

    print("=== 캘리브레이션 갭 절대값 기준 상위 25 (mean_pred - actual_rate) ===")
    top_calib = full.reindex(full["calib_gap"].abs().sort_values(ascending=False).index).head(25)
    print(top_calib[["axis", "group", "n", "actual_rate", "mean_pred", "calib_gap", "group_score"]].to_string(index=False))

    print("\n=== 그룹별 BSS(group_score) 하위 25 (baseline 대비 여지가 가장 많이 남은 구간, n>=500) ===")
    worst_bss = full.sort_values("group_score").head(25)
    print(worst_bss[["axis", "group", "n", "actual_rate", "mean_pred", "calib_gap", "group_score"]].to_string(index=False))

    full.to_csv("./open/temp/error_diagnostic_full.csv", index=False)
    print("\n전체 표는 open/temp/error_diagnostic_full.csv 에 저장됨.")


if __name__ == "__main__":
    main()
