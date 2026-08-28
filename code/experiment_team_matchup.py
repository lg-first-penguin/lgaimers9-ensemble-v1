# code/experiment_team_matchup.py
"""pitcher_id x batter_team_id / batter_id x pitcher_team_id "팀 단위 상성" 피처
스크리닝 — 2026-08-22 세션, 페어매치업(pitcher_id x batter_id) 기각의 교훈을 반영해
설계한 새 피처. 자세한 설계 근거는 code/train.py::apply_team_matchup_features 문서 참고.

핵심: 이번엔 causal_smoothed_te_encode(TE-residual, 실전 +13.86 확인)를 그대로 재사용해서
시즌-내 row-level 리크 버그가 구조적으로 불가능하다 — 처음부터 검증된 causal 패턴으로
시작한다(페어매치업처럼 "일단 짜고 나중에 리크 발견" 순서를 반복하지 않음).

thirdmodel_common.build_split()을 기반으로 쓴다(F1 필터, cutoff7/holdout 스플릿,
season-progression, TE-residual, tier A 제거+coarse pitchmix 전부 이미 적용된 상태) —
거기에 팀 매치업 2개 컬럼만 추가로 얹는다.

사용법: python -m code.experiment_team_matchup [--holdout 2023|2024] [--cutoff7]
"""
import argparse

import numpy as np

from code.mlp_model import (
    CAT_COLS, QUANTILE_D, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model
from code.train import apply_team_matchup_features, TEAM_MATCHUP_RESIDUAL_COLS
from code.thirdmodel_common import build_split, TARGET_COL

SCREEN_SEEDS = [42, 123, 7]

REF_CAT = 706.56
REF_MLP7 = 738.55
REF_BLEND = 753.37


def build_split_with_team_matchup(cutoff7=True, holdout=2024, apply_f1=True):
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(
        cutoff7=cutoff7, holdout=holdout, apply_f1=apply_f1,
    )
    prior = train_split[TARGET_COL].mean()
    train_split = apply_team_matchup_features(train_split, train_split, prior)
    val_split = apply_team_matchup_features(train_split, val_split, prior)
    print(f"[build_split_with_team_matchup] train={len(train_split)} val={len(val_split)} "
          f"mlp_num_cols={len(mlp_num_cols)} cat_feature_cols={len(cat_feature_cols)}")
    return train_split, val_split, mlp_num_cols, cat_feature_cols


def report_coverage(train_split, val_split):
    tr_cov = train_split["team_matchup_covered"].mean()
    val_cov = val_split["team_matchup_covered"].mean()
    print(f"[커버리지] train: team_matchup_covered 비율={tr_cov:.4f} "
          f"| val: team_matchup_covered 비율={val_cov:.4f}")
    return tr_cov, val_cov


def run_one(name, train_split, val_split, mlp_num_cols, cat_feature_cols):
    print(f"\n{'='*20} 실험: {name} {'='*20}")
    print(f"mlp_num_cols={len(mlp_num_cols)} cat_feature_cols={len(cat_feature_cols)}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()

    regime = "cutoff7" if args.cutoff7 else f"season=={args.holdout}"
    train_split, val_split, base_mlp_cols, base_cat_cols = build_split_with_team_matchup(
        cutoff7=args.cutoff7, holdout=args.holdout,
    )
    report_coverage(train_split, val_split)

    results = []
    results.append(run_one(
        "baseline(팀매치업 미포함)", train_split, val_split, base_mlp_cols, base_cat_cols,
    ))
    results.append(run_one(
        "team_matchup -> MLP", train_split, val_split, base_mlp_cols + TEAM_MATCHUP_RESIDUAL_COLS, base_cat_cols,
    ))
    results.append(run_one(
        "team_matchup -> CatBoost", train_split, val_split, base_mlp_cols, base_cat_cols + TEAM_MATCHUP_RESIDUAL_COLS,
    ))
    results.append(run_one(
        "team_matchup -> both", train_split, val_split,
        base_mlp_cols + TEAM_MATCHUP_RESIDUAL_COLS, base_cat_cols + TEAM_MATCHUP_RESIDUAL_COLS,
    ))

    print(f"\n{'='*20} 요약 ({regime}) {'='*20}")
    base = results[0]
    for r in results:
        d_cat = r["cat_score"] - base["cat_score"]
        d_mlp = r["mlp_score"] - base["mlp_score"]
        d_blend = r["blend_score"] - base["blend_score"]
        print(f"{r['name']:28s}: CatBoost={r['cat_score']:.2f}({d_cat:+.2f}) "
              f"MLP(3-seed)={r['mlp_score']:.2f}({d_mlp:+.2f}) blend={r['blend_score']:.2f}({d_blend:+.2f})")


if __name__ == "__main__":
    main()
