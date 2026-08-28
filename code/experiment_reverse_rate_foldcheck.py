# code/experiment_reverse_rate_foldcheck.py
"""reverse_rate 시즌분해(§81, 5-seed 재검증으로 dual-regime 레짐반전이 사라지고
Blend가 두 레짐 다 소폭 플러스로 나온 뒤) 최종 판단을 위한 rolling-origin
재검증이다. `code/experiment_te_residual_foldcheck.py`와 동일한 방법론 —
game_type=='R'만 써서 F1 필터가 이른 cutoff의 학습 구간을 통째로 지우는 함정을
피하고(핵심 교훈 #20/#28), train<val_season / val==val_season으로 2021/2022/2023
세 시즌을 순서대로 val로 삼는다.

baseline은 "지금 프로덕션 CatBoost 피처셋"(시즌진행분 포함 add_engineered_features
+ coarse pitchmix + TE-residual 6축, `code/train.py`와 동일 함수 재사용) 그대로,
여기에 reverse_rate 시즌분해 2컬럼(season_rate/season_rate_gap — 원래 실험의
season_n/season_count는 CatBoost엔 안 먹였던 REV_COLS 그대로 사용)을 추가한
버전을 비교한다. 노이즈를 줄이기 위해 CatBoost는 5-seed 앙상블(우리 프로덕션과
동일 `CATBOOST_SEED_POOL`)로 학습한다. TE-residual fold check와 같은 이유로
MLP/블렌드는 생략(신호 자체의 시간 일반화만 본다).

사용법: python -m code.experiment_reverse_rate_foldcheck
"""
import time

import numpy as np
import pandas as pd

from code.catboost_model import CAT_FEATURES, CATBOOST_SEED_POOL, train_catboost_ensemble
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix
from code.experiment_reverse_rate_season_progression import (
    build_reverse_season_end_lookup, apply_reverse_season_progression, REV_COLS,
)

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023]


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)  # F1 트랩 회피
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def ensemble_score(cols, train_split, val_split, y_val, tag):
    t0 = time.time()
    X_train, y_train = train_split[cols], train_split[TARGET_COL].values
    X_val = val_split[cols]
    results = train_catboost_ensemble(X_train, y_train, X_val, y_val, seeds=CATBOOST_SEED_POOL, verbose=False)
    preds_list = [m.predict_proba(X_val)[:, 1] for m, _ in results]
    ensemble_pred = np.mean(preds_list, axis=0)
    score = compute_bss(ensemble_pred, y_val)[2]
    print(f"    [{tag}] 5-seed 앙상블 Val Score={score:.2f} ({time.time()-t0:.1f}s)", flush=True)
    return score


def run_fold(df_all, df_trm, val_season):
    print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===", flush=True)
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)  # 시즌진행분 포함, 프로덕션과 동일하게 전체 df로 lookup 생성
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in PITCHMIX_COLS]
    all_cols = base_features + PITCHMIX_COLS + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values
    print(f"    n_train={len(train_split)} n_val={len(val_split)}", flush=True)

    te_prior = train_split[TARGET_COL].mean()
    train_split = apply_te_residual_features(train_split, train_split, te_prior)
    val_split = apply_te_residual_features(train_split, val_split, te_prior)
    baseline_cols = base_features + PITCHMIX_COLS + TE_RESIDUAL_COLS

    rev_lookup = build_reverse_season_end_lookup(train_split)
    tr_rev = apply_reverse_season_progression(train_split, rev_lookup)
    va_rev = apply_reverse_season_progression(val_split, rev_lookup)

    base_score = ensemble_score(baseline_cols, train_split, val_split, y_val, "baseline(현재 프로덕션)")
    new_score = ensemble_score(baseline_cols + REV_COLS, tr_rev, va_rev, y_val, "+reverse_season(2)")
    delta = new_score - base_score
    print(f"    delta = {delta:+.2f}", flush=True)
    return base_score, new_score, delta


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행", flush=True)

    rows = []
    for val_season in FOLD_SEASONS:
        b, n, d = run_fold(df_all, df_trm, val_season)
        rows.append((val_season, b, n, d))

    deltas = [d for _, _, _, d in rows]
    print("\n" + "=" * 70)
    print(f"{'val_season':<12}{'baseline':>12}{'+reverse':>12}{'delta':>10}")
    for val_season, b, n, d in rows:
        print(f"{val_season:<12}{b:>12.2f}{n:>12.2f}{d:>+10.2f}")
    wins = sum(1 for d in deltas if d > 0)
    print(f"\n{wins}/{len(deltas)} fold 승, 평균 delta = {np.mean(deltas):+.2f} (std={np.std(deltas):.2f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
