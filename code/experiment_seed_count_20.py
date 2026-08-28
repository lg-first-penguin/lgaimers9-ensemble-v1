# code/experiment_seed_count_20.py
"""MLP 앙상블 시드 수 재검증 2탄: 7-seed(현재 프로덕션) vs 15-seed vs 20-seed.

`code/experiment_seed_count_1521.py`(7->15)가 이미 이 repo 조건(cutoff7 스플릿,
현재 프로덕션 전체 피처셋, CatBoost+MLP 블렌드)으로 재검증됐고, 그 결과는 실전
리더보드까지 제출되어 **-6.46 리그레션으로 기각/원복**됐다
([[mlp_15seed_ensemble_adopted]]). 그런데 옆 세션(다른 repo, 순수 MLP)에서
"20시드까지 로컬 점수가 단조 증가"라는 새 보고가 들어와, 사용자 요청으로 이
repo 자체 조건에서 20시드까지 직접 재검증한다.

7-seed(`ENSEMBLE_SEEDS`) + 15-seed(`experiment_seed_count_1521.NEW_SEEDS`)에
새 시드 5개([10,11,12,13,14], 기존 15개와 전부 distinct)를 더해 총 20개를 한
번에 학습하고, 7/15/20개 평균을 MLP 솔로/CatBoost+MLP 블렌드 양쪽에서 비교한다.
CatBoost는 시드 수와 무관하므로 한 번만 학습해 재사용한다.

사용법:
  python -m code.experiment_seed_count_20 --cutoff7
  python -m code.experiment_seed_count_20 --holdout 2023
"""
import argparse
import time

from code.blend_model import fit_meta_model
from code.catboost_model import train_catboost
from code.experiment_seed_count_1521 import NEW_SEEDS as NEW_SEEDS_8
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS,
    embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    fit_quantile_edges, to_tensors, train_ensemble, predict_ensemble, compute_bss,
)
from code.thirdmodel_common import build_split, TARGET_COL

NEW_SEEDS_5 = [10, 11, 12, 13, 14]
SEEDS_20 = ENSEMBLE_SEEDS + NEW_SEEDS_8 + NEW_SEEDS_5
assert len(set(SEEDS_20)) == 20, f"시드 중복: {SEEDS_20}"


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (20-seed) ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout)
    y_val = val_split[TARGET_COL].values

    t0 = time.time()
    cat_model, cat_best_iter = train_catboost(
        train_split[cat_feature_cols], train_split[TARGET_COL].values,
        val_split[cat_feature_cols], y_val,
    )
    cat_preds = cat_model.predict_proba(val_split[cat_feature_cols])[:, 1]
    cat_score = compute_bss(cat_preds, y_val)[2]
    print(f"[CatBoost] Val Score={cat_score:.2f} (best_iter={cat_best_iter}, {time.time()-t0:.1f}s)")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
        seeds=SEEDS_20, verbose=True,
    )
    print(f"[MLP 20-seed 학습 완료] {time.time()-t0:.1f}s")

    results = {}
    for tag, member_subset in [("7-seed(현재 프로덕션)", members[:7]), ("15-seed", members[:15]), ("20-seed", members)]:
        mlp_preds = predict_ensemble(
            member_subset, cat_dims, len(mlp_num_cols), embed_dims,
            X_val_cat, X_val_num, bin_edges=bin_edges,
        )
        mlp_score = compute_bss(mlp_preds, y_val)[2]

        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val)
        print(f"[{tag}] MLP solo={mlp_score:.2f} | Blend={blend_score:.2f} "
              f"(w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f}, n_members={len(member_subset)})")
        results[tag] = (mlp_score, blend_score)

    (s7, b7), (s15, b15), (s20, b20) = results["7-seed(현재 프로덕션)"], results["15-seed"], results["20-seed"]
    print(f"\n--- {label} 요약 --- CatBoost={cat_score:.2f}")
    print(f"  solo:  7-seed={s7:.2f}  15-seed={s15:.2f}(delta7={s15-s7:+.2f})  20-seed={s20:.2f}(delta15={s20-s15:+.2f}, delta7={s20-s7:+.2f})")
    print(f"  blend: 7-seed={b7:.2f}  15-seed={b15:.2f}(delta7={b15-b7:+.2f})  20-seed={b20:.2f}(delta15={b20-b15:+.2f}, delta7={b20-b7:+.2f})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(args.cutoff7, holdout)


if __name__ == "__main__":
    main()
