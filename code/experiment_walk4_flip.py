# code/experiment_walk4_flip.py
"""code/experiment_walk4_filter.py의 후속. 순수 4구 연속 볼(스트라이크 0개) 볼넷의
마지막 투구 18,053건을 control_success로 그냥 "제거"하면 balls_before==3 &
strikes_before==0 조합이 인위적으로 거의 한쪽 라벨로 완전분리돼 CatBoost/MLP가 통째로
붕괴한다는 게 확인됐다(code/experiment_walk4_filter.py 결과, 행수 유지 안 됨).

이번엔 행을 지우지 않고 **라벨만 뒤집는다** — 완전분리 문제 없이 "고의사구스러운
행들의 현재 라벨이 실제로 학습에 방해가 되는지"를 직접 검증한다.
  - 실험 A: walk4 중 control_success=1인 9,159행 -> 0으로 뒤집어서 훈련
  - 실험 B: walk4 중 control_success=0인 8,894행 -> 1으로 뒤집어서 훈련
뒤집기는 TRAIN 스플릿(cutoff7) 행에만 적용하고, VAL 스플릿(검증 정답)은 원본 그대로
둔다 — 원래 공식 라벨 기준으로 점수를 매겨야 의미가 있다. 또한 뒤집기는 파생피처
계산(season-progression, TE-residual 등 전부 control_success에 의존) **이전에** 원본
df에 적용해, 이 행들을 참조하는 다른 행의 asof/TE 피처에도 인과적으로 일관되게
반영되게 한다.

사용법: python -m code.experiment_walk4_flip
"""
import numpy as np
import pandas as pd

from code.experiment_walk4_filter import (
    DATA_DIR, REF_BLEND, REF_CAT, REF_MLP7, SCREEN_SEEDS, TARGET_COL, find_walk4_rowids,
)
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


def build_split_with_flip(flip_rowids, flip_to, cutoff7=True, holdout=2024, apply_f1=True):
    """thirdmodel_common.build_split과 동일한 절차를 따르되, train 스플릿에 속한
    flip_rowids 행들의 control_success를 flip_to로 덮어쓴 뒤(파생피처 계산 전에)
    나머지는 동일하게 진행한다."""
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
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

    flip_mask = df["row_id"].isin(flip_rowids) & train_mask
    n_flipped = int(flip_mask.sum())
    before_dist = df.loc[flip_mask, TARGET_COL].value_counts().to_dict()
    df.loc[flip_mask, TARGET_COL] = flip_to
    print(f"[flip] train 스플릿 내 {n_flipped}행의 control_success -> {flip_to}로 뒤집음 "
          f"(뒤집기 전 분포: {before_dist})")

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=trk_holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]

    df = merge_coarse_pitchmix(df, df_trm, holdout=trk_holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    features = [c for c in df.columns if c not in (TARGET_COL,)]
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


def run_one(name, train_split, val_split, mlp_num_cols, cat_feature_cols):
    print(f"\n{'='*20} 실험: {name} {'='*20}")
    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split[TARGET_COL].values
    cat_model, cat_best_iter = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(cat_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[CatBoost] best_iter={cat_best_iter} score={cat_score:.2f} (ref={REF_CAT:.2f}, delta={cat_score-REF_CAT:+.2f})")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
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
    return dict(name=name, cat_score=cat_score, mlp_score=mlp_score, blend_score=blend_score)


def main():
    rowids_cs1, rowids_cs0 = find_walk4_rowids()

    results = []

    ts_a, vs_a, num_a, catcols_a = build_split_with_flip(rowids_cs1, flip_to=0, cutoff7=True)
    results.append(run_one("cs=1 -> 0 뒤집기", ts_a, vs_a, num_a, catcols_a))

    ts_b, vs_b, num_b, catcols_b = build_split_with_flip(rowids_cs0, flip_to=1, cutoff7=True)
    results.append(run_one("cs=0 -> 1 뒤집기", ts_b, vs_b, num_b, catcols_b))

    print(f"\n{'='*20} 요약 {'='*20}")
    print(f"baseline(원본 라벨, 7-seed 레퍼런스): CatBoost={REF_CAT:.2f} MLP={REF_MLP7:.2f} blend={REF_BLEND:.2f}")
    for r in results:
        print(f"{r['name']:16s}: CatBoost={r['cat_score']:.2f} "
              f"MLP(3-seed)={r['mlp_score']:.2f} blend={r['blend_score']:.2f}")


if __name__ == "__main__":
    main()
