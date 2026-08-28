# code/fix_full_retrain_5cat20mlp.py
"""dopip.py 실행 버그 수정용 원샷 스크립트.

무슨 일이 있었나: 2026-08-24 dopip.py 실행에서 Step1(train.py, 5-seed CatBoost +
20-seed MLP)이 로컬 cutoff7 검증에서 751.36을 냈는데, 기존 reference(pre-DeepFM
2-way, 7-seed MLP)가 753.37로 근소하게 더 높아 test.py가 정상적으로(!) KEEP_REF를
선택했다 — 이 자체는 노이즈밴드 안의 정상적인 안전장치 동작이다. 문제는 그 다음:
dopip.py의 [Full Retrain] 단계가 "reference의 멤버 수(7) == 현재 ENSEMBLE_SEEDS
길이(20)"가 거짓이라 이 reference를 "호환 안 되는 레거시 포맷"으로 오판하고,
전부 기본값(MLP 시드당 균일 20+5=25epoch, CatBoost 시드당 균일 500+50=550
iteration, meta_model={w_cat:1,w_mlp:1,intercept:0} 미학습 동일가중치)으로
폴백해버렸다 — Step1이 실제로 학습해 만든 진짜 값(시드별 best_epoch, catboost
시드별 best_iteration, 제대로 fit된 스태킹 가중치)은 KEEP_REF 처리 때 latest_model.pkl
이 삭제되면서 함께 날아갔다. 사용자는 이 특정 설정(5cat+20mlp)을 reference 승격
여부와 무관하게 그대로 실전 제출해보기로 결정했으므로, 이 스크립트는 Step1 로그에
남아있던 실제 값들을 하드코딩해 Full Retrain만 다시 올바르게 수행한다.

Step1 로그에서 추출한 실제 값(재실행 없이 그대로 재사용):
  - MLP 시드별 best_epoch (ENSEMBLE_SEEDS 순서): 아래 PER_SEED_BEST_EPOCH
  - CatBoost 시드별 best_iteration (CATBOOST_SEED_POOL 순서): 아래 PER_SEED_BEST_ITER
  - 메타모델: w_cat=1.8727523834540176, w_mlp=1.9774001599340556,
    intercept=-1.9458592941994106 (Step1 cutoff7 val Score 751.36 당시 fit)

pitchmix_lookup.csv/season_end_lookup.csv/pair_matchup_lookup.csv는 이미 이번
dopip.py 실행에서 정상적으로(버그 영향 없이) 저장돼 있으므로 재계산하지 않는다.
`dopip.py`의 Full Retrain 블록(피처 엔지니어링 부분)을 그대로 복제하되, 모델
학습/번들 조립 부분만 하드코딩된 실제 값으로 교체했다.
"""
import os
import pickle

import numpy as np
import pandas as pd

from code.train import (
    apply_f1_filter, add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS,
    TRACKMAN_TIER_FEED, build_season_end_lookup, apply_pair_matchup_rowlevel,
    build_pair_matchup_lookup, PAIR_COLS,
)
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, to_tensors, train_mlp, make_bundle, get_device,
)
from code.catboost_model import train_catboost_ensemble, CATBOOST_SEED_POOL
from code.blend_model import make_blend_bundle
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS

ID_COL = "row_id"
TARGET_COL = "control_success"
DATA_DIR = "./open/data"
FULL_RETRAIN_EPOCH_BUFFER = 5
CATBOOST_ITERATION_BUFFER = 50

# Step1 로그에서 추출 (ENSEMBLE_SEEDS 순서와 정확히 일치)
PER_SEED_BEST_EPOCH = {
    42: 4, 123: 6, 7: 4, 2024: 7, 99: 6, 555: 4, 31337: 4,
    1: 5, 2: 3, 3: 5, 4: 2, 5: 5, 6: 3, 8: 3, 9: 2,
    10: 2, 11: 2, 12: 2, 13: 4, 14: 3,
}
# Step1 로그에서 추출 (CATBOOST_SEED_POOL 순서와 정확히 일치: [42,123,7,2024,99])
PER_SEED_BEST_ITER = {42: 1097, 123: 978, 7: 1013, 2024: 905, 99: 762}
REAL_META_MODEL = {
    "w_cat": 1.8727523834540176,
    "w_mlp": 1.9774001599340556,
    "intercept": -1.9458592941994106,
}

assert set(PER_SEED_BEST_EPOCH) == set(ENSEMBLE_SEEDS), "PER_SEED_BEST_EPOCH가 현재 ENSEMBLE_SEEDS와 다릅니다"
assert set(PER_SEED_BEST_ITER) == set(CATBOOST_SEED_POOL), "PER_SEED_BEST_ITER가 현재 CATBOOST_SEED_POOL과 다릅니다"


def main():
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    train_df_raw["top_bottom"] = train_df_raw["top_bottom"].map({"T": 0, "B": 1}).astype("int64")
    train_df = train_df_raw.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    train_df, trk_tier_cols = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=2025)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]

    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=None)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = train_df[TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)
    train_df = apply_f1_filter(train_df)

    te_prior = train_df[TARGET_COL].mean()
    train_df = apply_te_residual_features(train_df, train_df, te_prior)

    train_df = apply_pair_matchup_rowlevel(train_df)
    # (pair_matchup_lookup.csv는 이미 이전 실행에서 저장됨 — 재저장 불필요)

    drop_cols = [ID_COL, TARGET_COL]
    full_features = [col for col in train_df.columns if col not in drop_cols and col not in TE_RESIDUAL_COLS]
    num_cols = [c for c in full_features if c not in CAT_COLS and c not in trk_cat_cols and c not in PAIR_COLS]
    cat_feature_cols = [c for c in full_features if c not in trk_mlp_cols and c not in PAIR_COLS] + TE_RESIDUAL_COLS

    print(f"[Fix Full Retrain] 총 {len(train_df)}행, MLP {len(ENSEMBLE_SEEDS)}시드, CatBoost {len(CATBOOST_SEED_POOL)}시드")

    full_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_df, CAT_COLS, num_cols)
    X_full_cat, X_full_num, y_full = to_tensors(full_proc, CAT_COLS, num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_full_num, n_bins=QUANTILE_N_BINS)

    device = get_device()
    full_members = []
    for seed in ENSEMBLE_SEEDS:
        full_epochs = max(PER_SEED_BEST_EPOCH[seed], 1) + FULL_RETRAIN_EPOCH_BUFFER
        model, _ = train_mlp(
            X_full_cat, X_full_num, y_full,
            cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
            max_epochs=full_epochs, device=device, seed=seed,
        )
        full_members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": full_epochs,
            "seed": seed,
        })
        print(f"[Fix Full Retrain] MLP seed={seed} {full_epochs} epoch 학습 완료 ({len(full_members)}/{len(ENSEMBLE_SEEDS)})")

    final_mlp_bundle = make_bundle(
        full_members, CAT_COLS, num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
    )

    per_seed_iterations = [PER_SEED_BEST_ITER[s] + CATBOOST_ITERATION_BUFFER for s in CATBOOST_SEED_POOL]
    print(f"[Fix Full Retrain] CatBoost {len(CATBOOST_SEED_POOL)}-seed, iterations={per_seed_iterations}")
    X_full_raw, y_full_raw = train_df[cat_feature_cols], train_df[TARGET_COL].values
    catboost_results = train_catboost_ensemble(
        X_full_raw, y_full_raw, seeds=CATBOOST_SEED_POOL,
        per_seed_iterations=per_seed_iterations, verbose=True,
    )
    final_catboost_models = [m for m, _ in catboost_results]
    print("[Fix Full Retrain] CatBoost 5-seed 학습 완료")

    final_bundle = make_blend_bundle(final_catboost_models, final_mlp_bundle, REAL_META_MODEL, cat_feature_cols=cat_feature_cols)
    final_bundle["catboost_best_iteration"] = per_seed_iterations[0]
    final_bundle["catboost_best_iterations"] = per_seed_iterations

    FINAL_SUBMIT_MODEL_PATH = "./submit/model/final_retained_model.pkl"
    os.makedirs(os.path.dirname(FINAL_SUBMIT_MODEL_PATH), exist_ok=True)
    with open(FINAL_SUBMIT_MODEL_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"[Fix Full Retrain 완료] 올바른 meta_model/epoch/iteration으로 재구성 -> {FINAL_SUBMIT_MODEL_PATH}")
    print(f"  meta_model={REAL_META_MODEL}")


if __name__ == "__main__":
    main()
