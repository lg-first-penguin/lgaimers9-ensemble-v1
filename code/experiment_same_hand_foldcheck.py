# code/experiment_same_hand_foldcheck.py
"""code/experiment_same_hand.py의 dual-regime 결과(cutoff7 CatBoost -0.67/MLP -4.36,
season==2023 CatBoost +24.18/MLP +71.63)를 rolling-origin 3-fold로 재검증한다.

사용자 지적: cutoff7은 노이즈 범위 안이라 쳐도(거의 0), season==2023의 MLP +71.63은
이 프로젝트에서 흔히 보는 "2023만 좋아 보이는" 노이즈치고는 폭이 커서 rolling-origin으로
한 번 더 확인해볼 가치가 있다는 판단 — `experiment_career_trajectory_foldcheck.py`/
`experiment_li_zero_filter_foldcheck.py`와 동일한 방법론(game_type=='R'만 써서 F1
필터가 이른 cutoff의 학습 구간 F행을 통째로 지우는 함정 회피, train<val_season
rolling-origin)을 CatBoost뿐 아니라 MLP(3-seed)에도 적용한다 — 기존 foldcheck
스크립트들은 전부 CatBoost 단독이었지만, 이번 신호는 MLP 쪽이 핵심이라 MLP도 반드시
포함해야 결론이 난다.

사용법:
  python -m code.experiment_same_hand_foldcheck
"""
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.experiment_same_hand import SAME_HAND_COLS, apply_same_hand, train_mlp_once, MLP_SEEDS
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, to_tensors,
)
from code.train import add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix

torch.use_deterministic_algorithms(True, warn_only=True)

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023]


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)  # F1 트랩 회피
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def train_catboost(X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def run_mlp(train_split, val_split, num_cols, device):
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val = val_proc[TARGET_COL].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    preds_list = []
    for seed in MLP_SEEDS:
        model, _ = train_mlp_once(
            X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
            X_val_cat, X_val_num, y_val, seed, device,
        )
        with torch.no_grad():
            preds_list.append(model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy())
    ens_preds = np.mean(preds_list, axis=0)
    return compute_bss(ens_preds, y_val)[2]


def run_fold(df_all, df_trm, val_season, device):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)
    df = apply_same_hand(df)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols]
    mlp_num_base = [c for c in base_features if c not in CAT_COLS and c not in PITCHMIX_COLS and c not in SAME_HAND_COLS]
    cat_base = [c for c in base_features if c not in SAME_HAND_COLS]

    all_cols = base_features + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    prior = train_split[TARGET_COL].mean()
    train_split_te = apply_te_residual_features(train_split, train_split, prior)
    val_split_te = apply_te_residual_features(train_split, val_split, prior)

    results = {}
    for tag, use_sh in [("baseline", False), ("+same_hand", True)]:
        cat_cols = cat_base + TE_RESIDUAL_COLS + (SAME_HAND_COLS if use_sh else [])
        mlp_cols = mlp_num_base + (SAME_HAND_COLS if use_sh else [])

        t0 = time.time()
        X_train, y_train = train_split_te[cat_cols], train_split_te[TARGET_COL].values
        X_val = val_split_te[cat_cols]
        preds, best_iter = train_catboost(X_train, y_train, X_val, y_val)
        cat_score = compute_bss(preds, y_val)[2]
        print(f"  [val={val_season}][CatBoost {tag}] Val Score={cat_score:.2f} "
              f"(best_iter={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_cols)})")

        t0 = time.time()
        mlp_score = run_mlp(train_split_te, val_split_te, mlp_cols, device)
        print(f"  [val={val_season}][MLP {tag}] 3-seed ensemble Val Score={mlp_score:.2f} "
              f"(n_features={len(mlp_cols)}, {time.time()-t0:.1f}s)")

        results[tag] = (cat_score, mlp_score)

    cat_delta = results["+same_hand"][0] - results["baseline"][0]
    mlp_delta = results["+same_hand"][1] - results["baseline"][1]
    print(f"  [val={val_season}] CatBoost delta={cat_delta:+.2f}  MLP delta={mlp_delta:+.2f}")
    return cat_delta, mlp_delta


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")
    device = get_device()
    print(f"[Device] {device}")

    cat_deltas, mlp_deltas = [], []
    for val_season in FOLD_SEASONS:
        print(f"\n{'='*70}\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===\n{'='*70}")
        cd, md = run_fold(df_all, df_trm, val_season, device)
        cat_deltas.append(cd)
        mlp_deltas.append(md)

    print(f"\n{'='*70}")
    print(f"CatBoost: {sum(d > 0 for d in cat_deltas)}/{len(cat_deltas)} fold 승리, "
          f"평균 delta={np.mean(cat_deltas):+.2f}, deltas={[f'{d:+.2f}' for d in cat_deltas]}")
    print(f"MLP:      {sum(d > 0 for d in mlp_deltas)}/{len(mlp_deltas)} fold 승리, "
          f"평균 delta={np.mean(mlp_deltas):+.2f}, deltas={[f'{d:+.2f}' for d in mlp_deltas]}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
