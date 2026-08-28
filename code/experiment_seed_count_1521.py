# code/experiment_seed_count_1521.py
"""MLP 앙상블 시드 수 재검증: 7-seed(현재 프로덕션) vs 15-seed.

이 repo(`code/mlp_model.py::ENSEMBLE_SEEDS` 주석)에는 이미 "15-seed로 늘려도 추가
개선 없음(768.93, 정체) -> 7개로 확정"이라는 결론이 있지만, 그 실측은 이 프로젝트의
아주 초기 단계(§3.1, CatBoost->MLP 전환 직후) 기록이다 — cutoff7 스플릿도, F1
필터/season-progression/TE-residual 같은 지금 피처셋도, `torch.use_deterministic_
algorithms` 논디터미니즘 픽스(2026-08-21, [[post_1041_mlp_upgrade_track]] §64)도 전부
그 시점엔 없었다. 즉 그 기각 근거 자체가 지금 기준으로 보면 스킴/스플릿이 다른 데다
GPU 논디터미니즘이 섞였을 가능성이 있는 낡은 측정값이다.

한편 peer 세션(다른 lgaimers9 계열 repo, 스플릿/피처셋/모델계열이 이 repo와 다름 —
season==2024 전체 홀드아웃, TE-residual/coarse pitchmix 없음, MLP 솔로만)에서 독립적
으로 "7-seed 786.81 -> 15-seed(14 distinct) 800.17, +13.36, 재현됨"이라는 결과를
보고했다. 조건이 다른 repo의 결과라 숫자를 그대로 이식할 근거는 없지만, "시드를
늘리면 이 프로젝트의 통상적 노이즈 플로어(~20pt)를 넘는 이득이 있을 수 있다"는
가설 자체는 이 repo 자체 조건(cutoff7 스플릿, 현재 프로덕션 전체 피처셋, CatBoost+MLP
블렌드, determinism 픽스 적용 상태)으로 독립 재검증할 가치가 있다.

기존 7-seed(`ENSEMBLE_SEEDS`)에 새 8개 시드([1,2,3,4,5,6,8,9], 기존 목록과 전부
distinct)를 추가해 총 15개를 한 번에 학습하고, 첫 7개 평균(현재 프로덕션과 동일)과
15개 전체 평균을 MLP 솔로/CatBoost+MLP 블렌드 양쪽에서 비교한다. CatBoost는 시드
수와 무관하므로 한 번만 학습해 재사용한다.

1단계: dual-regime(cutoff7 + season==2023) 전체(솔로+블렌드) 스크리닝.

사용법:
  python -m code.experiment_seed_count_1521 --cutoff7
  python -m code.experiment_seed_count_1521 --holdout 2023
"""
import argparse
import time

from code.blend_model import fit_meta_model
from code.catboost_model import train_catboost
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS,
    embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    fit_quantile_edges, to_tensors, train_ensemble, predict_ensemble, compute_bss,
)
from code.thirdmodel_common import build_split, TARGET_COL

NEW_SEEDS = [1, 2, 3, 4, 5, 6, 8, 9]
SEEDS_15 = ENSEMBLE_SEEDS + NEW_SEEDS
assert len(set(SEEDS_15)) == 15, f"시드 중복: {SEEDS_15}"


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout)
    y_val = val_split[TARGET_COL].values

    # --- CatBoost (시드 수와 무관, 한 번만 학습) ---
    t0 = time.time()
    cat_model, cat_best_iter = train_catboost(
        train_split[cat_feature_cols], train_split[TARGET_COL].values,
        val_split[cat_feature_cols], y_val,
    )
    cat_preds = cat_model.predict_proba(val_split[cat_feature_cols])[:, 1]
    cat_score = compute_bss(cat_preds, y_val)[2]
    print(f"[CatBoost] Val Score={cat_score:.2f} (best_iter={cat_best_iter}, {time.time()-t0:.1f}s)")

    # --- MLP 15-seed 한 번에 학습 ---
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
        seeds=SEEDS_15, verbose=True,
    )
    print(f"[MLP 15-seed 학습 완료] {time.time()-t0:.1f}s")

    results = {}
    for tag, member_subset in [("7-seed(현재 프로덕션)", members[:7]), ("15-seed", members)]:
        mlp_preds = predict_ensemble(
            member_subset, cat_dims, len(mlp_num_cols), embed_dims,
            X_val_cat, X_val_num, bin_edges=bin_edges,
        )
        mlp_score = compute_bss(mlp_preds, y_val)[2]

        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val)
        print(f"[{tag}] MLP solo={mlp_score:.2f} | Blend={blend_score:.2f} "
              f"(w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f}, n_members={len(member_subset)})")
        results[tag] = (mlp_score, blend_score)

    (solo7, blend7), (solo15, blend15) = results["7-seed(현재 프로덕션)"], results["15-seed"]
    print(f"\n--- {label} 요약 --- CatBoost={cat_score:.2f} | "
          f"solo: 7-seed={solo7:.2f} 15-seed={solo15:.2f} delta={solo15-solo7:+.2f} | "
          f"blend: 7-seed={blend7:.2f} 15-seed={blend15:.2f} delta={blend15-blend7:+.2f}")
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
