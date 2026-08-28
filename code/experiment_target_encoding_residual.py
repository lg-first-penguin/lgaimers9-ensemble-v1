# code/experiment_target_encoding_residual.py
"""팀원 제보 Track A(target-encoding 잔차) 6개 피처를 검증한다.

투수/타자의 "상황별 성공률 - 주효과"를 스무딩해서 넣는다. asof_pitcher_success_rate
(주효과, 이미 공식 피처로 제공됨)와 중복을 줄이려고 상황별 인코딩에서 주효과를 뺀
잔차만 쓴다(팀원 설계와 동일). 시즌<s만으로 인코딩을 만드는 "시즌-causal" 방식으로
(1) 학습 파티션 내 자기 자신/동일 시즌 데이터가 자기 인코딩에 섞이는 self-leakage와
(2) 팀원이 처음 겪은 F1 오염 재유입(인코딩 통계에도 F1 필터를 적용해야 함, 인코딩
소스가 이미 F1 필터된 train_split이므로 자동으로 해결됨)을 동시에 막는다.

시즌-causal 계산은 pd.merge_asof(direction='backward', allow_exact_matches=False)로
그룹별 "이 시즌보다 앞선 시즌들의 누적 합/표본수"를 구현했다 — 행 자신의 시즌이
인코딩 소스(학습 파티션)에 존재하지 않는 경우(예: holdout=2023처럼 검증 시즌 전체가
학습 파티션 밖에 있는 경우)도 merge_asof가 "그 이전에 가장 가까운 관측 시즌까지의
누적"을 자동으로 찾아주므로 별도 폴백 코드가 필요 없다.

사용법:
  python -m code.experiment_target_encoding_residual --cutoff7
  python -m code.experiment_target_encoding_residual --holdout 2023
"""
import argparse
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_season_progression import TARGET_COL, build_split
from code.train import apply_f1_filter
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors,
    train_ensemble,
)

K_SMOOTH = 200

# (피처명, 그룹 컬럼, 주효과와 뺄지 여부(주효과 컬럼명))
TE_AXES = [
    ("te_p_cnt", ["pitcher_id", "balls_before", "strikes_before"], "p_main"),
    ("te_p_bhand", ["pitcher_id", "batter_hand"], "p_main"),
    ("te_p_run", ["pitcher_id", "num_runners_on"], "p_main"),
    ("te_p_inn", ["pitcher_id", "inning"], "p_main"),
    ("te_b_cnt", ["batter_id", "balls_before", "strikes_before"], "b_main"),
]
MAIN_AXES = [("p_main", ["pitcher_id"]), ("b_main", ["batter_id"])]

TE_RESIDUAL_COLS = [f"{name}_res" for name, *_ in TE_AXES] + ["te_covered"]


def causal_smoothed_encode(source_df, query_df, group_cols, prior, k=K_SMOOTH):
    """source_df(학습 파티션, 이미 F1 필터 적용됨)로 시즌-causal 스무딩 인코딩을 만들어
    query_df(아무 행이나: train/val/test)에 적용한다. query_df 행 자신의 시즌보다
    엄격히 앞선 시즌들의 데이터만 쓴다 — allow_exact_matches=False가 핵심."""
    agg = source_df.groupby(group_cols + ["season"])[TARGET_COL].agg(["sum", "count"]).reset_index()
    agg = agg.sort_values("season")
    agg["cum_sum"] = agg.groupby(group_cols)["sum"].cumsum()
    agg["cum_n"] = agg.groupby(group_cols)["count"].cumsum()
    agg = agg[group_cols + ["season", "cum_sum", "cum_n"]].sort_values("season")

    query = query_df[group_cols + ["season"]].reset_index()
    query_sorted = query.sort_values("season")
    merged = pd.merge_asof(
        query_sorted, agg, on="season", by=group_cols,
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("index")
    cum_sum = merged["cum_sum"].fillna(0.0).values
    cum_n = merged["cum_n"].fillna(0.0).values
    enc = (cum_sum + prior * k) / (cum_n + k)
    covered = (cum_n > 0).astype(np.int64)
    return enc, covered


def add_te_residual_features(source_df, query_df, prior):
    """source_df로 인코딩을 만들어 query_df에 잔차 피처 6개를 계산해 붙인다."""
    query_df = query_df.copy()
    mains = {}
    covered_any = np.zeros(len(query_df), dtype=np.int64)
    for name, group_cols in MAIN_AXES:
        enc, covered = causal_smoothed_encode(source_df, query_df, group_cols, prior)
        mains[name] = enc
        covered_any = np.maximum(covered_any, covered)

    for name, group_cols, main_key in TE_AXES:
        enc, covered = causal_smoothed_encode(source_df, query_df, group_cols, prior)
        query_df[f"{name}_res"] = enc - mains[main_key]
        covered_any = np.maximum(covered_any, covered)

    query_df["te_covered"] = covered_any
    return query_df


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(holdout, cutoff7)
    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in trk_mlp_cols and c not in trk_cat_cols]

    all_cols = base_features + trk_mlp_cols + trk_cat_cols + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    y_val_raw = val_split[TARGET_COL].values

    prior = train_split[TARGET_COL].mean()
    t0 = time.time()
    te_source = train_split  # 인코딩 소스는 F1 필터가 이미 적용된 원본 train_split(잔차 컬럼 추가 전)
    train_split = add_te_residual_features(te_source, train_split, prior)
    val_split = add_te_residual_features(te_source, val_split, prior)
    print(f"[TE 잔차 피처 생성] {time.time()-t0:.1f}s (prior={prior:.4f})")

    results = {}
    for tag, use_new in [("baseline", False), ("+TE잔차6개", True)]:
        extra = TE_RESIDUAL_COLS if use_new else []
        cat_feature_cols = base_features + trk_cat_cols + extra
        mlp_num_cols = [c for c in base_features if c not in CAT_COLS] + trk_mlp_cols + extra

        X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
        X_val, y_val = val_split[cat_feature_cols], y_val_raw

        t0 = time.time()
        cat_model, cat_best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        cat_preds = cat_model.predict_proba(X_val)[:, 1]
        cat_score = compute_bss(cat_preds, y_val)[2]
        print(f"[{tag}][CatBoost] Val Score={cat_score:.2f} (best_iteration={cat_best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_feature_cols)})")

        device = get_device()
        train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
        val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
        X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
        X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
        embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
        bin_edges = fit_quantile_edges(X_tr_num)

        t0 = time.time()
        members = train_ensemble(
            X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
            seeds=ENSEMBLE_SEEDS, device=device, verbose=False,
        )
        mlp_preds = predict_ensemble(members, cat_dims, len(mlp_num_cols), embed_dims, X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
        mlp_score = compute_bss(mlp_preds, y_val)[2]
        print(f"[{tag}][MLP(7-seed)] Val Score={mlp_score:.2f} ({time.time()-t0:.1f}s)")

        from code.blend_model import fit_meta_model
        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val)
        print(f"[{tag}][Blend] w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f} -> Val Score={blend_score:.2f}")

        results[tag] = (cat_score, mlp_score, blend_score)

    base = results["baseline"]
    print(f"\n--- {label} 요약 (vs baseline) ---")
    print(f"  {'variant':<14}{'CatBoost':>12}{'MLP(7seed)':>14}{'Blend':>12}")
    for tag in ["baseline", "+TE잔차6개"]:
        c, m, b = results[tag]
        print(f"  {tag:<14}{c:>8.2f}({c-base[0]:+.2f}) {m:>8.2f}({m-base[1]:+.2f}) {b:>8.2f}({b-base[2]:+.2f})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
