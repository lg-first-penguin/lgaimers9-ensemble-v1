# code/experiment_walk4_filter.py
"""'순수 4구 연속 볼(스트라이크 0개)' 볼넷 시퀀스의 마지막(ball-four) 투구 18,053건 중
control_success=1(9,159건)만 빼고 훈련 / control_success=0(8,894건)만 빼고 훈련해서
cutoff7 로컬 점수가 어떻게 바뀌는지 확인한다 (고의사구 오라벨링 가설 검증 후속).

row_id 식별은 code/pitcher_crosswalk.py::segment_train_games로 게임 경계를 잡고,
같은 타자·같은 게임 내에서 balls_before가 0->1->2->3으로 이어지며 strikes_before가
계속 0으로 유지된 마지막 투구를 찾는 방식(세션 내 find_walk4_rowids.py와 동일 로직,
여기서는 훈련 스플릿 구성과 한 파일에서 재사용하려고 인라인으로 다시 구현).

훈련 파이프라인은 code/thirdmodel_common.py::build_split을 그대로 따르되(F1 필터,
cutoff7 스플릿, 트랙맨 tier 비활성 상태, season-progression/TE-residual 전부 동일),
row_id를 끝까지 보존해 train_split에서 골라낸 두 후보군을 각각 제거한 뒤 CatBoost는
전체 재학습, MLP는 3-seed 스크리닝(SCREEN_SEEDS)으로 프로덕션 7-seed 레퍼런스 대비
delta를 보는 기존 스크리닝 관례(code/experiment_thirdmodel_base.py 등)를 따른다.

사용법: python -m code.experiment_walk4_filter
"""
import os

import numpy as np
import pandas as pd

from code.pitcher_crosswalk import segment_train_games
from code.mlp_model import (
    CAT_COLS, QUANTILE_D, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model
from code.train import (
    TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED, add_engineered_features, apply_f1_filter,
    apply_te_residual_features,
)
from code.trackman_pitcher_features import add_all_tiers, clean_trackman, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]

# 프로덕션 7-seed cutoff7 레퍼런스 (code/experiment_thirdmodel_base.py 캐시 기준, 다른
# 스크리닝 실험들과 동일한 비교 기준선)
REF_CAT = 706.56
REF_MLP7 = 738.55
REF_BLEND = 753.37


def find_walk4_rowids():
    cols = ["row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
            "balls_before", "strikes_before", "outs_before",
            "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id", "control_success"]
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=cols)
    df = df.sort_values("row_id").reset_index(drop=True)
    seg = segment_train_games(df)
    seg = seg.sort_values(["game_uid", "row_id"]).reset_index(drop=True)

    b = seg
    same_batter = [b["batter_id"] == b["batter_id"].shift(k) for k in (1, 2, 3)]
    same_game = [b["game_uid"] == b["game_uid"].shift(k) for k in (1, 2, 3)]
    is_ball4 = (
        (b["balls_before"] == 3) & (b["strikes_before"] == 0)
        & (b["balls_before"].shift(1) == 2) & (b["strikes_before"].shift(1) == 0) & same_batter[0] & same_game[0]
        & (b["balls_before"].shift(2) == 1) & (b["strikes_before"].shift(2) == 0) & same_batter[1] & same_game[1]
        & (b["balls_before"].shift(3) == 0) & (b["strikes_before"].shift(3) == 0) & same_batter[2] & same_game[2]
    )
    walk4 = b[is_ball4]
    rowids_cs1 = set(walk4.loc[walk4["control_success"] == 1, "row_id"].values.tolist())
    rowids_cs0 = set(walk4.loc[walk4["control_success"] == 0, "row_id"].values.tolist())
    print(f"[walk4] 총 {len(walk4)}건 (cs=1: {len(rowids_cs1)}, cs=0: {len(rowids_cs0)})")
    return rowids_cs1, rowids_cs0


def build_split_with_rowid(cutoff7=True, holdout=2024, apply_f1=True):
    """thirdmodel_common.build_split과 동일하지만 row_id를 끝까지 보존한다."""
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
        trk_holdout = 2024
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout
        trk_holdout = holdout

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=trk_holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]

    df = merge_coarse_pitchmix(df, df_trm, holdout=trk_holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    features = [c for c in df.columns if c not in (TARGET_COL,)]  # row_id 보존
    mlp_num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols
                    and c not in trk_mlp_cols and c != "row_id"]
    cat_feature_cols = [c for c in features if c not in trk_mlp_cols and c != "row_id"]

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    if apply_f1:
        before = len(train_split)
        train_split = apply_f1_filter(train_split)
        print(f"[F1 필터] {before} -> {len(train_split)}행")

    te_prior = train_split[TARGET_COL].mean()
    train_split = apply_te_residual_features(train_split, train_split, te_prior)
    val_split = apply_te_residual_features(train_split, val_split, te_prior)
    cat_feature_cols = cat_feature_cols + TE_RESIDUAL_COLS

    return train_split, val_split, mlp_num_cols, cat_feature_cols


def run_one(name, train_split, val_split, mlp_num_cols, cat_feature_cols, exclude_rowids):
    print(f"\n{'='*20} 실험: {name} (제외 {len(exclude_rowids)}행) {'='*20}")
    ts = train_split[~train_split["row_id"].isin(exclude_rowids)].reset_index(drop=True)
    print(f"훈련 행수: {len(train_split)} -> {len(ts)}")

    X_train_raw, y_train_raw = ts[cat_feature_cols], ts[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split[TARGET_COL].values
    cat_model, cat_best_iter = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(cat_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[CatBoost] best_iter={cat_best_iter} score={cat_score:.2f} (ref={REF_CAT:.2f}, delta={cat_score-REF_CAT:+.2f})")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(ts, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    device = get_device()

    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=val_proc[TARGET_COL].values,
        seeds=SCREEN_SEEDS, device=device,
    )
    ens_pred = predict_ensemble(
        members, cat_dims, len(mlp_num_cols), embed_dims,
        X_val_cat, X_val_num, bin_edges=bin_edges, quantile_d=QUANTILE_D, device=device,
    )
    mlp_score = compute_bss(ens_pred, y_val_raw)[2]
    print(f"[MLP {len(SCREEN_SEEDS)}-seed] score={mlp_score:.2f} (ref 7-seed={REF_MLP7:.2f}, delta={mlp_score-REF_MLP7:+.2f})")

    w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val_preds, ens_pred, y_val_raw)
    print(f"[2-way blend] score={blend_score:.2f} (ref={REF_BLEND:.2f}, delta={blend_score-REF_BLEND:+.2f}) "
          f"weights cat={w_cat:.3f} mlp={w_mlp:.3f} intercept={intercept:.3f}")
    return dict(name=name, n_removed=len(exclude_rowids), cat_score=cat_score, mlp_score=mlp_score, blend_score=blend_score)


def main():
    rowids_cs1, rowids_cs0 = find_walk4_rowids()

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split_with_rowid(cutoff7=True)
    print(f"[build_split] train={len(train_split)} val={len(val_split)} "
          f"mlp_num_cols={len(mlp_num_cols)} cat_feature_cols={len(cat_feature_cols)}")

    results = []
    results.append(run_one("cs=1 제외", train_split, val_split, mlp_num_cols, cat_feature_cols, rowids_cs1))
    results.append(run_one("cs=0 제외", train_split, val_split, mlp_num_cols, cat_feature_cols, rowids_cs0))

    print(f"\n{'='*20} 요약 {'='*20}")
    print(f"baseline(무필터, 7-seed 레퍼런스): CatBoost={REF_CAT:.2f} MLP={REF_MLP7:.2f} blend={REF_BLEND:.2f}")
    for r in results:
        print(f"{r['name']:12s} (n={r['n_removed']}): CatBoost={r['cat_score']:.2f} "
              f"MLP(3-seed)={r['mlp_score']:.2f} blend={r['blend_score']:.2f}")


if __name__ == "__main__":
    main()
