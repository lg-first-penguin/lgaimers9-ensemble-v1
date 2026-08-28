# code/experiment_mlp_ensemble_role.py
"""2026-08-24 세션: "MLP의 앙상블 내 역할"과 "MLP만의 특성"(QuantileEmbedding 기반 PLE
수치 인코딩)을 겨냥한 신규 피처 스크리닝.

배경 — 왜 단순 교차항(explicit interaction)이 아니라 이 3가지를 고르는가:
이 프로젝트에서 "명시적 교차/interaction 피처"는 지금까지 CatBoost/MLP 양쪽 다 거의 항상
손해였다(§14.2, §26, §27 배터강화/pitch-mix 교차 -13.43~-44, 팀원 B/D/H/E/F 전부 듀얼레짐
sign-flip, career_trajectory/career_acceleration/month-level-trend 전부 기각). 이번엔 그
패턴과 다른 종류의 3가지 가설만 고른다:

1. **pitchmix -> MLP도**: coarse pitchmix(4열)는 이미 채택된 유일한 트랙맨 피처(CatBoost
   전용, cutoff7 +34.43/2023 +9.57)로 "새 정보가 있다"는 게 이미 검증됐다 — 아직 한 번도
   MLP 쪽에 먹여본 적이 없다(TRACKMAN_TIER_FEED 관례상 tier는 mlp/cat 중 하나만 골랐지만
   pitchmix는 애초에 항상 cat 전용으로만 하드코딩돼 있었음). "새 정보 자체가 없어서
   실패"가 아니라 "라우팅을 안 해봐서 미검증"인 유일한 케이스.
2. **TE-residual 중 1개 축만 MLP에**: 전체 6열을 MLP에 먹였을 때 cutoff7 solo -24.51로
   무너진 적이 있다(apply_te_residual_features 문서 참고) — 그런데 "정보 자체가 MLP에
   해롭다"와 "6개 축을 한�   먹여서 용량/노이즈 과부하가 왔다"는 서로 다른 가설이다.
   가장 강한 축 하나(te_p_cnt_res, 투수x카운트)만 줄여서 재확인한 적은 없다.
3. **season-progression 내부 교차**: 지금까지의 "교차항은 대체로 중복" 패턴은 전부
   서로 무관하거나 약한 축끼리의 교차였다. 반면 season-progression(8열)은 이 프로젝트
   역사상 가장 크게 검증된 단일 피처(§45, 실전 +45.32 기여) — "이미 강하다고 증명된
   축끼리의 교차"는 아직 아무도 안 해봤다. pitcher_season_rate_gap과
   batter_season_rate_gap의 곱(둘 다 동시에 시즌 컨디션이 좋은/나쁜 경우를 포착)을 시험한다.

평가: `code/thirdmodel_common.py::build_split`(현재 프로덕션과 완전히 동일한 파이프라인,
TE-residual/coarse pitchmix 포함)을 그대로 쓰고, CatBoost는 레짐당 1번만 고정 학습해
재사용(피처가 안 바뀌므로), MLP는 SCREEN_SEEDS(3-seed)로 변형마다 새로 학습해 solo+블렌드
점수를 함께 본다 (§28의 교훈 — 블렌드가 실제 최적화 대상이므로 solo만 보면 착시 가능).

사용법:
  python -m code.experiment_mlp_ensemble_role --regime cutoff7
  python -m code.experiment_mlp_ensemble_role --regime 2023
  python -m code.experiment_mlp_ensemble_role --regime both
"""
import argparse

import numpy as np

from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.thirdmodel_common import build_split, TARGET_COL
from code.trackman_pitcher_features import PITCHMIX_COLS
from code.train import causal_smoothed_te_encode, TE_MAIN_AXES

SCREEN_SEEDS = [42, 123, 7]

TE_STRONGEST_AXIS = ("te_p_cnt", ["pitcher_id", "balls_before", "strikes_before"], "p_main")


def add_te_single_axis(train_split, val_split, prior):
    """TE-residual 6열 중 가장 강한 1개 축(투수x카운트)만 계산해 MLP 전용 열
    (`te_p_cnt_res_mlp`)로 추가한다. CatBoost용 `te_p_cnt_res`(cat_feature_cols에 이미
    포함됨)와 이름이 겹치지 않도록 별도 컬럼명을 쓴다 — 값 자체는 동일한 계산이라도
    "MLP에 먹이는 열"과 "CatBoost에 먹이는 열"을 독립적으로 다루기 위함(관례 유지)."""
    name, group_cols, main_key = TE_STRONGEST_AXIS
    out_col = "te_p_cnt_res_mlp"

    tr_main, _ = causal_smoothed_te_encode(train_split, train_split, ["pitcher_id"], prior)
    tr_axis, _ = causal_smoothed_te_encode(train_split, train_split, group_cols, prior)
    train_split = train_split.copy()
    train_split[out_col] = tr_axis - tr_main

    va_main, _ = causal_smoothed_te_encode(train_split, val_split, ["pitcher_id"], prior)
    va_axis, _ = causal_smoothed_te_encode(train_split, val_split, group_cols, prior)
    val_split = val_split.copy()
    val_split[out_col] = va_axis - va_main
    return train_split, val_split, [out_col]


def add_season_interaction(df):
    df = df.copy()
    p_gap = df["pitcher_season_rate_gap"].fillna(0.0)
    b_gap = df["batter_season_rate_gap"].fillna(0.0)
    df["season_gap_product"] = p_gap * b_gap
    conf = np.minimum(np.log1p(df["pitcher_season_n"]), np.log1p(df["batter_season_n"]))
    df["season_gap_product_weighted"] = p_gap * b_gap * conf
    return df, ["season_gap_product", "season_gap_product_weighted"]


def run_variant(train_split, val_split, num_cols, device, seeds=SCREEN_SEEDS):
    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, num_cols, TARGET_COL)
    X_va_cat, X_va_num, y_va = to_tensors(va_proc, CAT_COLS, num_cols, TARGET_COL)

    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_va,
        seeds=seeds, device=device, verbose=False,
    )
    preds = predict_ensemble(
        members, cat_dims, len(num_cols), embed_dims, X_va_cat, X_va_num,
        bin_edges=bin_edges, device=device,
    )
    return compute_bss(preds, y_va.numpy())[2], preds


def run_regime(cutoff7, holdout, seeds=SCREEN_SEEDS):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    y_val = val_split[TARGET_COL].values
    device = get_device()

    from code.catboost_model import train_catboost, predict_catboost
    from code.blend_model import fit_meta_model

    X_train_raw = train_split[cat_feature_cols]
    y_train_raw = train_split[TARGET_COL].values
    X_val_raw = val_split[cat_feature_cols]
    catboost_model, _ = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=False)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val)[2]
    print(f"[CatBoost(고정)] Val Score: {cat_score:.2f}")

    te_prior = train_split[TARGET_COL].mean()
    tr_te, va_te, te_cols = add_te_single_axis(train_split, val_split, te_prior)

    tr_season, season_cols = add_season_interaction(train_split)
    va_season, _ = add_season_interaction(val_split)

    variants = {
        "baseline": (train_split, val_split, mlp_num_cols),
        "+pitchmix(4)": (train_split, val_split, mlp_num_cols + PITCHMIX_COLS),
        "+te_single_axis(1)": (tr_te, va_te, mlp_num_cols + te_cols),
        "+season_interaction(2)": (tr_season, va_season, mlp_num_cols + season_cols),
    }
    combo_tr, combo_cols = train_split.copy(), list(mlp_num_cols)
    combo_tr = combo_tr.copy()
    for c in PITCHMIX_COLS:
        combo_tr[c] = train_split[c]
    combo_tr[te_cols[0]] = tr_te[te_cols[0]]
    for c in season_cols:
        combo_tr[c] = tr_season[c]
    combo_va = val_split.copy()
    for c in PITCHMIX_COLS:
        combo_va[c] = val_split[c]
    combo_va[te_cols[0]] = va_te[te_cols[0]]
    for c in season_cols:
        combo_va[c] = va_season[c]
    combo_cols = mlp_num_cols + PITCHMIX_COLS + te_cols + season_cols
    variants["+all(7)combined"] = (combo_tr, combo_va, combo_cols)

    results = {}
    for name, (tr, va, cols) in variants.items():
        mlp_score, mlp_preds = run_variant(tr, va, cols, device, seeds=seeds)
        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val_preds, mlp_preds, y_val)
        results[name] = (mlp_score, blend_score)
        print(f"[{name}] MLP solo: {mlp_score:.2f} | Blend: {blend_score:.2f} (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f})")

    base_mlp, base_blend = results["baseline"]
    print(f"\n--- {label} 요약 (baseline MLP solo={base_mlp:.2f}, Blend={base_blend:.2f}) ---")
    for name, (mlp_score, blend_score) in results.items():
        if name == "baseline":
            continue
        print(f"  {name}: MLP solo delta={mlp_score - base_mlp:+.2f} | Blend delta={blend_score - base_blend:+.2f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["cutoff7", "2023", "both"], default="both")
    parser.add_argument("--ensemble", action="store_true", default=False, help="7-seed(ENSEMBLE_SEEDS)로 재검증 (기본은 3-seed 스크리닝)")
    args = parser.parse_args()
    seeds = ENSEMBLE_SEEDS if args.ensemble else SCREEN_SEEDS

    if args.regime in ("cutoff7", "both"):
        run_regime(cutoff7=True, holdout=2024, seeds=seeds)
    if args.regime in ("2023", "both"):
        run_regime(cutoff7=False, holdout=2023, seeds=seeds)
