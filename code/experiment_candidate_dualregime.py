# code/experiment_candidate_dualregime.py
"""§86 다음 단계 — 사용자 지시: TrackA를 적용한 뒤에도 dual-regime(cutoff7+season2023)
검증을 해보고, 손수연의 진짜 기여 포인트를 찾는다. 두 후보를 cutoff7/season2023
양쪽에서 3-seed로 검증한다:

candidate A -- 손수연 레시피(no_both/wide-categorical/season-1트랙맨/그의 formula) +
  Track A(TE-residual, 이 repo 검증됨) + F1 필터. §86-①(TrackA만 추가, F1 미적용)이
  cutoff7에서 +17.21(블렌드)로 가장 강한 신호였지만 F1이 빠져있어 season2023에서
  안전한지 모른다. F1을 추가해서 두 레짐 다 통과하는지 확인.

candidate B -- 우리 자신의 프로덕션 레시피(F1+TrackA+시즌진행분+coarse pitchmix, 이미
  전부 있음, `code/thirdmodel_common.py::build_split`) 위에 손수연의 진짜 신규 레버인
  season-1 앵커 트랙맨 std5/gap4(§86-②에서 no_both/wide-categorical은 노이즈로
  판정됐으므로 이것만 남음)를 추가. row_id 없이 6-key(match_season+inning+top_bottom+
  balls_before+strikes_before+pitcher_hand+batter_hand) 조인으로 붙인다(top_bottom을
  잠깐 'T'/'B' 문자열로 되돌려서 손수연의 lookup 테이블과 맞춤).

season2023 레짐은 fresh 7-seed MLP를 한 번만 학습해 pickle로 저장(재사용 목적,
§86-③에서는 저장 안 해서 재학습해야 했음 -- 이번엔 저장해서 다음 후보 테스트 때
재사용 가능).

사용법: python -m code.experiment_candidate_dualregime
"""
import pickle
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, embed_dim_for_cardinality,
    fit_preprocessing, apply_preprocessing, fit_quantile_edges, to_tensors,
    train_ensemble, make_bundle, predict_bundle, get_device, compute_bss,
)
from code.blend_model import fit_meta_model
from code.catboost_model import train_catboost, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.thirdmodel_common import build_split
from code.train import apply_te_residual_features, TE_RESIDUAL_COLS, apply_f1_filter
from code.experiment_sooyun_recipe import (
    build_sooyun_features, SOOYUN_CATBOOST_PARAMS, SOOYUN_ITERATIONS, SOOYUN_CAT_COLS,
    check_memory_or_abort, DATA_DIR, TARGET, add_merge_features, MATCH_COLS_6,
    STD_FEATURES, GAP_FEATURES, SOOYUN_DIR,
)

SEEDS = [42, 123, 7]
SEASON2023_MLP_SEEDS = ENSEMBLE_SEEDS[:7]
SEASON2023_MLP_CACHE = "/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/00986cdc-e8b5-4c84-88c6-ad5668236ca7/scratchpad/season2023_mlp_bundle.pkl"


def train_catboost_variant(X_train, y_train, X_val, y_val, cat_features, seed, iterations=None):
    if iterations is None:
        params = dict(CATBOOST_PARAMS)
        params["iterations"] = MAX_ITERATIONS
        params["random_seed"] = seed
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        model = CatBoostClassifier(**params)
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    else:
        model = CatBoostClassifier(**SOOYUN_CATBOOST_PARAMS, iterations=iterations, random_seed=seed)
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        model.fit(train_pool)
    preds = model.predict_proba(X_val)[:, 1]
    _, _, score = compute_bss(preds, y_val)
    return preds, score


def build_candidate_a(df_raw, train_mask, val_mask, apply_f1):
    """손수연 레시피 + TrackA (+옵션 F1)."""
    sooyun_full = build_sooyun_features(df_raw)
    sooyun_full = sooyun_full.merge(df_raw[["row_id", "pitcher_id", "batter_id"]], on="row_id", how="left")
    train_sy = sooyun_full.loc[train_mask].reset_index(drop=True)
    val_sy = sooyun_full.loc[val_mask].reset_index(drop=True)
    del sooyun_full
    for c in SOOYUN_CAT_COLS:
        train_sy[c] = train_sy[c].astype(str)
        val_sy[c] = val_sy[c].astype(str)

    te_prior = train_sy[TARGET].mean()
    train_sy = apply_te_residual_features(train_sy, train_sy, te_prior)
    val_sy = apply_te_residual_features(train_sy, val_sy, te_prior)
    train_sy = train_sy.drop(columns=["pitcher_id", "batter_id"])
    val_sy = val_sy.drop(columns=["pitcher_id", "batter_id"])

    if apply_f1:
        before = len(train_sy)
        train_sy = apply_f1_filter(train_sy)
        print(f"  [candidate A, F1적용] {before} -> {len(train_sy)}행", flush=True)

    drop_cols = ["row_id", TARGET]
    feature_cols = [c for c in train_sy.columns if c not in drop_cols]
    return train_sy, val_sy, feature_cols, SOOYUN_CAT_COLS


def build_candidate_b_extra(df_raw):
    """우리 프로덕션 레시피 위에 얹을 season-1앵커 트랙맨 std5/gap4 (row_id keyed)."""
    std_table = pd.read_csv(f"{SOOYUN_DIR}/model/trackman_match_table.csv")
    gap_table = pd.read_csv(f"{SOOYUN_DIR}/model/trackman_match_table_gap.csv")
    key_df = df_raw[["row_id", "season", "inning", "top_bottom", "balls_before",
                      "strikes_before", "pitcher_hand", "batter_hand"]].copy()
    key_df = add_merge_features(key_df, std_table, MATCH_COLS_6)
    key_df = add_merge_features(key_df, gap_table, MATCH_COLS_6)
    extra_cols = ["row_id"] + STD_FEATURES + GAP_FEATURES
    return key_df[extra_cols]


def run_cutoff7(df_raw, mlp_bundle, device):
    print("\n" + "=" * 70 + "\n=== cutoff7 ===\n" + "=" * 70, flush=True)
    train_mask = (df_raw["season"] < 2024) | ((df_raw["season"] == 2024) & (df_raw["game_month"] < 7))
    val_mask = (df_raw["season"] == 2024) & (df_raw["game_month"] >= 7)

    df_ours = df_raw.copy()
    df_ours["top_bottom"] = df_ours["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    from code.train import add_engineered_features, apply_same_hand
    league_mean = df_ours.loc[train_mask, TARGET].mean()
    df_ours_eng = add_engineered_features(df_ours.copy(), league_mean)
    df_ours_eng = apply_same_hand(df_ours_eng)
    val_ours = df_ours_eng.loc[val_mask].reset_index(drop=True)
    mlp_preds = predict_bundle(mlp_bundle, val_ours, device=device)

    results = {}

    # --- candidate A: 손수연+TrackA+F1 ---
    check_memory_or_abort("cutoff7 candidate A 빌드 전")
    train_a, val_a, feats_a, cats_a = build_candidate_a(df_raw, train_mask, val_mask, apply_f1=True)
    y_val_a = val_a[TARGET].values
    for seed in SEEDS:
        check_memory_or_abort(f"cutoff7 candidate A seed={seed}")
        t0 = time.time()
        preds, cat_score = train_catboost_variant(
            train_a[feats_a], train_a[TARGET], val_a[feats_a], y_val_a, cats_a, seed,
            iterations=SOOYUN_ITERATIONS,
        )
        _, _, _, blend_score, _ = fit_meta_model(preds, mlp_preds, y_val_a)
        print(f"[cutoff7][candidateA-F1][seed={seed}] cat={cat_score:.2f} blend={blend_score:.2f} ({time.time()-t0:.1f}s)", flush=True)
        results.setdefault("A", []).append((seed, cat_score, blend_score))
    del train_a, val_a

    # --- candidate B: 프로덕션 + std5/gap4 ---
    check_memory_or_abort("cutoff7 candidate B 빌드 전")
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=True, apply_f1=True)
    extra = build_candidate_b_extra(df_raw)
    # train_split/val_split엔 row_id가 없다 -- season-1앵커 std/gap는 row_id 불필요(6-key로 자기 자신에
    # 붙이면 되므로) train_split/val_split 자체에 직접 병합한다(6-key가 이미 컬럼으로 존재).
    std_table = pd.read_csv(f"{SOOYUN_DIR}/model/trackman_match_table.csv")
    gap_table = pd.read_csv(f"{SOOYUN_DIR}/model/trackman_match_table_gap.csv")
    for split_df in (train_split, val_split):
        split_df["top_bottom"] = split_df["top_bottom"].map({0: "T", 1: "B"})
    train_split = add_merge_features(train_split, std_table, MATCH_COLS_6)
    train_split = add_merge_features(train_split, gap_table, MATCH_COLS_6)
    val_split = add_merge_features(val_split, std_table, MATCH_COLS_6)
    val_split = add_merge_features(val_split, gap_table, MATCH_COLS_6)
    for split_df in (train_split, val_split):
        split_df["top_bottom"] = split_df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    cat_feature_cols_b = cat_feature_cols + STD_FEATURES + GAP_FEATURES
    y_val_b = val_split[TARGET].values
    cat_features_base = ["game_type", "base_state"]
    for seed in SEEDS:
        check_memory_or_abort(f"cutoff7 candidate B seed={seed}")
        t0 = time.time()
        preds, cat_score = train_catboost_variant(
            train_split[cat_feature_cols_b], train_split[TARGET], val_split[cat_feature_cols_b], y_val_b,
            cat_features_base, seed,
        )
        _, _, _, blend_score, _ = fit_meta_model(preds, mlp_preds, y_val_b)
        print(f"[cutoff7][candidateB][seed={seed}] cat={cat_score:.2f} blend={blend_score:.2f} ({time.time()-t0:.1f}s)", flush=True)
        results.setdefault("B", []).append((seed, cat_score, blend_score))
    del train_split, val_split

    return results


def get_or_train_season2023_mlp(train_split, val_split, mlp_num_cols, y_val, device):
    import os
    if os.path.exists(SEASON2023_MLP_CACHE):
        print(f"[season2023 MLP] 캐시 발견, 재사용: {SEASON2023_MLP_CACHE}", flush=True)
        with open(SEASON2023_MLP_CACHE, "rb") as f:
            mlp_bundle = pickle.load(f)
        return mlp_bundle

    check_memory_or_abort("season2023 MLP 학습 전")
    t0 = time.time()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_t,
        seeds=SEASON2023_MLP_SEEDS, embed_dims=embed_dims, bin_edges=bin_edges,
        device=device, verbose=False,
    )
    mlp_bundle = make_bundle(
        members, CAT_COLS, mlp_num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
    )
    print(f"[season2023 MLP] 학습 완료 ({time.time()-t0:.1f}s), 캐시에 저장", flush=True)
    with open(SEASON2023_MLP_CACHE, "wb") as f:
        pickle.dump(mlp_bundle, f)
    return mlp_bundle


def run_season2023(df_raw, device):
    print("\n" + "=" * 70 + "\n=== season2023 ===\n" + "=" * 70, flush=True)
    train_mask = df_raw["season"] < 2023
    val_mask = df_raw["season"] == 2023

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=False, holdout=2023, apply_f1=True)
    y_val_prod = val_split[TARGET].values
    mlp_bundle = get_or_train_season2023_mlp(train_split, val_split, mlp_num_cols, y_val_prod, device)
    mlp_preds = predict_bundle(mlp_bundle, val_split, device=device)
    _, _, mlp_score = compute_bss(mlp_preds, y_val_prod)
    print(f"[season2023][우리 MLP] Val Score={mlp_score:.2f}", flush=True)

    results = {}

    # --- candidate A ---
    check_memory_or_abort("season2023 candidate A 빌드 전")
    train_a, val_a, feats_a, cats_a = build_candidate_a(df_raw, train_mask, val_mask, apply_f1=True)
    y_val_a = val_a[TARGET].values
    assert len(val_a) == len(val_split), f"val 행 개수 불일치: A={len(val_a)} prod={len(val_split)}"
    for seed in SEEDS:
        check_memory_or_abort(f"season2023 candidate A seed={seed}")
        t0 = time.time()
        preds, cat_score = train_catboost_variant(
            train_a[feats_a], train_a[TARGET], val_a[feats_a], y_val_a, cats_a, seed,
            iterations=SOOYUN_ITERATIONS,
        )
        _, _, _, blend_score, _ = fit_meta_model(preds, mlp_preds, y_val_a)
        print(f"[season2023][candidateA-F1][seed={seed}] cat={cat_score:.2f} blend={blend_score:.2f} ({time.time()-t0:.1f}s)", flush=True)
        results.setdefault("A", []).append((seed, cat_score, blend_score))
    del train_a, val_a

    # --- candidate B ---
    check_memory_or_abort("season2023 candidate B 빌드 전")
    std_table = pd.read_csv(f"{SOOYUN_DIR}/model/trackman_match_table.csv")
    gap_table = pd.read_csv(f"{SOOYUN_DIR}/model/trackman_match_table_gap.csv")
    for split_df in (train_split, val_split):
        split_df["top_bottom"] = split_df["top_bottom"].map({0: "T", 1: "B"})
    train_split = add_merge_features(train_split, std_table, MATCH_COLS_6)
    train_split = add_merge_features(train_split, gap_table, MATCH_COLS_6)
    val_split = add_merge_features(val_split, std_table, MATCH_COLS_6)
    val_split = add_merge_features(val_split, gap_table, MATCH_COLS_6)
    for split_df in (train_split, val_split):
        split_df["top_bottom"] = split_df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    cat_feature_cols_b = cat_feature_cols + STD_FEATURES + GAP_FEATURES
    y_val_b = val_split[TARGET].values
    cat_features_base = ["game_type", "base_state"]
    for seed in SEEDS:
        check_memory_or_abort(f"season2023 candidate B seed={seed}")
        t0 = time.time()
        preds, cat_score = train_catboost_variant(
            train_split[cat_feature_cols_b], train_split[TARGET], val_split[cat_feature_cols_b], y_val_b,
            cat_features_base, seed,
        )
        _, _, _, blend_score, _ = fit_meta_model(preds, mlp_preds, y_val_b)
        print(f"[season2023][candidateB][seed={seed}] cat={cat_score:.2f} blend={blend_score:.2f} ({time.time()-t0:.1f}s)", flush=True)
        results.setdefault("B", []).append((seed, cat_score, blend_score))

    return results


def summarize(name, results):
    print(f"\n--- {name} 요약 ---")
    for key, rows in results.items():
        avg_c = np.mean([r[1] for r in rows])
        avg_b = np.mean([r[2] for r in rows])
        print(f"  candidate {key}: cat_avg={avg_c:.2f} blend_avg={avg_b:.2f}")


def main():
    print("[데이터 로드]", flush=True)
    t0 = time.time()
    df_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df_raw = df_raw.dropna(subset=[TARGET]).reset_index(drop=True)
    for c in df_raw.select_dtypes(include="float64").columns:
        df_raw[c] = df_raw[c].astype(np.float32)
    for c in df_raw.select_dtypes(include="int64").columns:
        if c not in ("row_id",):
            df_raw[c] = df_raw[c].astype(np.int32)
    print(f"train.csv 로드 완료: {len(df_raw)}행 ({time.time()-t0:.1f}s)", flush=True)
    check_memory_or_abort("train.csv 로드 직후")

    with open("./open/reference/best_model.pkl", "rb") as f:
        ref_bundle = pickle.load(f)
    mlp_bundle_cutoff7 = ref_bundle["mlp_bundle"]
    device = get_device()

    results_cutoff7 = run_cutoff7(df_raw, mlp_bundle_cutoff7, device)
    summarize("cutoff7", results_cutoff7)

    results_season2023 = run_season2023(df_raw, device)
    summarize("season2023", results_season2023)

    print("\n" + "=" * 70)
    print("최종 요약 (참고: cutoff7 프로덕션=753.37, 손수연+TrackA-noF1=775.76 / season2023 프로덕션=734.94)")
    summarize("cutoff7", results_cutoff7)
    summarize("season2023", results_season2023)
    print("=" * 70)


if __name__ == "__main__":
    main()
