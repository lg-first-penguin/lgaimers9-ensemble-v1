# code/experiment_pitchmix_only_reverify.py
"""tier A(투수 크로스워크 x pitcher-std, MLP feed)를 빼고 coarse pitchmix(CatBoost feed)만
남긴 구성을 7-seed 프로덕션 앙상블 + 두 레짐(cutoff7/season==2023)으로 재검증한다.

배경: §41(실전 950.81, 982.22 대비 -31.41)의 원인 후보였던 "하이퍼파라미터 미재탐색"은
`code/experiment_hparam_reverify.py`로 배제됐다(재탐색해도 평균적으로 더 나쁘고 레짐 간
부호가 반전). 남은 유력 후보는 tier A 자체다 — §37에서 tier A/B/C 통합판이 실전에서
-112.70 하락했을 때, 원인 조사(§38)는 `clean_trackman()` 버그를 찾아 대부분 설명했지만
"검증(train.csv 기반)에서 측정한 크로스워크 커버리지가 실제 test.csv보다 낙관적으로
편향됐을 가능성"(PROJECT_HISTORY.md 핵심 교훈 #21, 2순위 용의자)은 끝내 확정도 배제도
되지 않은 채 남아있었다. tier A는 그 크로스워크(`pitcher_map.csv`)에 여전히 의존하므로
이 위험을 그대로 물려받는다 — coarse pitchmix는 투수 정체성/season을 조인 키로 쓰지
않아 이 문제를 구조적으로 피해간다.

팀원의 독립 실험 보고서(2026-08-18 세션에서 전달받음)도 같은 패턴을 시사한다:
"선수 단위" 파생 피처(릴리스 표준편차, 구종 엔트로피, FB-BR 물리량 gap)는 전부 로컬에서도
효과가 없어 기각됐고, "투수 정체성 x 카운트"로 세분화한 count-mix 피처는 로컬에서는
개선됐지만 실제 리더보드에서 하락해 기각됐다(우리 tier A/B/C 첫 시도의 실패와 정확히
같은 모양). 반면 정체성 없는 coarse pitchmix는 XGBoost/CatBoost 양쪽에서 견고했고
팀원 쪽 실전 927까지 확인됐다.

이 스크립트는 tier A를 완전히 빼고 coarse pitchmix만 남긴 구성(모듈 상수
`experiment_residual_correction_9_10.TIER_FEED`를 `{}`로 임시 monkeypatch)을 프로덕션
7-seed 앙상블로 재확인한다.

사용법:
  python -m code.experiment_pitchmix_only_reverify --cutoff7
  python -m code.experiment_pitchmix_only_reverify --holdout 2023
"""
import argparse
import time

import code.experiment_residual_correction_9_10 as split_mod
from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors,
    train_ensemble,
)

TARGET_COL = "control_success"


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (pitchmix-only, tier A 제외) ===\n{'='*70}")

    orig_tier_feed = split_mod.TIER_FEED
    split_mod.TIER_FEED = {}
    try:
        train_split, val_split, features, cat_features, mlp_num_cols = split_mod.build_split(holdout, cutoff7)
    finally:
        split_mod.TIER_FEED = orig_tier_feed

    device = get_device()
    X_train_raw = train_split[cat_features]
    y_train_raw = train_split[TARGET_COL].values
    X_val_raw = val_split[cat_features]
    y_val_raw = val_split[TARGET_COL].values

    t0 = time.time()
    catboost_model, best_iter = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_preds, y_val_raw)[2]
    print(f"[CatBoost] Val Score={cat_score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s)")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num)

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_raw,
        seeds=ENSEMBLE_SEEDS, device=device, verbose=False,
    )
    mlp_preds = predict_ensemble(members, cat_dims, len(mlp_num_cols), embed_dims, X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
    mlp_score = compute_bss(mlp_preds, y_val_raw)[2]
    print(f"[MLP(7-seed)] Val Score={mlp_score:.2f} ({time.time()-t0:.1f}s)")

    w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val_raw)
    print(f"[Blend] Score={blend_score:.2f} (w_cat={w_cat:.3f}, w_mlp={w_mlp:.3f}, intercept={intercept:.3f})")
    return cat_score, mlp_score, blend_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
