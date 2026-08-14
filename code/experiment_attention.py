# code/experiment_attention.py
"""3번째 모델 후보(attention 계열) 스크리닝 실험.

CatBoost(GBDT)+Tabular MLP(concat) 2-way 스태킹에 구조적으로 다른 계열을 하나 더
넣어 앙상블 독립성을 높일 수 있는지 본다 (PROJECT_HISTORY.md "다음으로 시도해볼 만한
방향" 항목). 먼저 단일 시드로 빠르게 스크리닝해 (a) solo BSS가 CatBoost/MLP와 경쟁이
되는지, (b) 기존 두 모델과 예측이 얼마나 다른지(상관계수 — 낮을수록 스태킹에 유리)
확인한 뒤, 유의미하면 7-seed 앙상블 + 3-way 스태킹으로 확장한다.

두 아키텍처를 비교한다:
  --model ft     FT-Transformer (Gorishniy et al. 2021) — code/ft_transformer_model.py
  --model excel  ExcelFormer 근사 구현 — code/excelformer_model.py (semi-permeable
                 attention을 CatBoost feature importance 기반 랭킹으로 근사; 논문의
                 beta-mixup 증강은 포함하지 않음, 순수 아키텍처 비교 목적)

사용법:
  python -m code.experiment_attention --model ft --holdout 2024
  python -m code.experiment_attention --model excel --holdout 2023
  python -m code.experiment_attention --model excel --holdout 2024 --ensemble   # 7-seed
  python -m code.experiment_attention --model excel --holdout 2024 --ensemble --foldcheck
      # 위 실행에 더해 2024 안에서 rolling-origin 3-fold(월 버킷, code/experiment_3way_stack.py
      # ::step_foldcheck와 동일 FOLD_BOUNDARIES)로 2-way vs 3-way 스태킹 이득을 재검증한다.
      # §24.5에서 단일 holdout(2023/2024) 2-point 비교만으로는 표본이 너무 적어(n=2)
      # "이득이 진짜 불안정한지 우연인지" 구분이 안 됐던 문제를 보완한다. holdout=2024
      # 전용 — fold 경계가 한 시즌 안의 월(5~10월)을 나누는 방식이라 다른 holdout에는
      # 적용되지 않는다.
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import predict_catboost, train_catboost
from code.train import add_engineered_features

DATA_DIR = "./open/data"
TARGET_COL = "control_success"


def build_split(holdout, apply_f1):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    train_mask = df["season"] < holdout
    val_mask = df["season"] == holdout
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS]

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)

    if apply_f1:
        before = len(train_split)
        train_split = train_split[~((train_split["game_type"] == "F") & (train_split["season"] <= 2022))].reset_index(drop=True)
        print(f"[F1 필터] {before} -> {len(train_split)}행")

    return train_split, val_split, features, num_cols


def run_foldcheck(model_name, val_split, cat_val_preds, mlp_val_preds, att_val_preds, y_val_raw):
    """code/experiment_3way_stack.py::step_foldcheck와 동일한 rolling-origin 3-fold
    (월 버킷 확장 윈도우)로 season==2024 안에서 2-way vs 3-way 스태킹을 재검증한다.
    CatBoost/MLP/attention 모델 자체는 재학습하지 않고(season<2024로 이미 학습됨), 각
    fold에서 메타모델(로지스틱 회귀)만 그 fold의 meta_train 구간으로 다시 피팅한다."""
    from code.blend_model import fit_meta_model, predict_meta
    from code.experiment_3way_stack import FOLD_BOUNDARIES, fit_meta_model_n

    game_month = val_split["game_month"].values
    deltas = []
    print(f"\n[{model_name}-foldcheck] {'val 구간':<10} {'n_train':>8} {'n_val':>8} {'2-way':>10} {model_name+' 3-way':>14} {'delta':>8}")
    for lo, hi in FOLD_BOUNDARIES:
        val_mask = (game_month == lo) | (game_month == hi)
        train_mask = game_month < lo

        cat_tr, mlp_tr, att_tr, y_tr = cat_val_preds[train_mask], mlp_val_preds[train_mask], att_val_preds[train_mask], y_val_raw[train_mask]
        cat_va, mlp_va, att_va, y_va = cat_val_preds[val_mask], mlp_val_preds[val_mask], att_val_preds[val_mask], y_val_raw[val_mask]

        w_cat, w_mlp, intercept, _, _ = fit_meta_model(cat_tr, mlp_tr, y_tr)
        two_way_preds = predict_meta(w_cat, w_mlp, intercept, cat_va, mlp_va)
        two_way_score = compute_bss(two_way_preds, y_va)[2]

        weights, intercept3, _, _ = fit_meta_model_n([cat_tr, mlp_tr, att_tr], y_tr)
        z = sum(w * p for w, p in zip(weights, [cat_va, mlp_va, att_va])) + intercept3
        three_way_preds = 1.0 / (1.0 + np.exp(-z))
        three_way_score = compute_bss(three_way_preds, y_va)[2]

        delta = three_way_score - two_way_score
        deltas.append(delta)
        print(f"{lo}~{hi}월    {train_mask.sum():>8} {val_mask.sum():>8} {two_way_score:>10.2f} {three_way_score:>14.2f} {delta:>+8.2f}")

    wins = sum(1 for d in deltas if d > 0)
    print(f"\n[{model_name}-foldcheck] {wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ft", "excel"], default="ft")
    parser.add_argument("--holdout", type=int, default=2024, choices=[2023, 2024])
    parser.add_argument("--apply-f1", dest="apply_f1", action="store_true", default=True)
    parser.add_argument("--no-f1", dest="apply_f1", action="store_false")
    parser.add_argument("--ensemble", action="store_true", default=False,
                         help="attention 모델을 단일 시드가 아니라 ENSEMBLE_SEEDS 7개로 학습")
    parser.add_argument("--foldcheck", action="store_true", default=False,
                         help="season==2024 안에서 rolling-origin 3-fold로 2-way vs 3-way 스태킹 이득을 재검증 (holdout=2024 전용)")
    args = parser.parse_args()
    if args.foldcheck and args.holdout != 2024:
        raise ValueError("--foldcheck은 holdout=2024 전용입니다 (fold 경계가 2024 시즌 내 월을 나누는 방식)")
    label = f"model={args.model} holdout={args.holdout} f1={'ON' if args.apply_f1 else 'OFF'}"

    train_split, val_split, features, num_cols = build_split(args.holdout, args.apply_f1)
    print(f"[{label}] 훈련: {len(train_split)}행 | 검증: {len(val_split)}행 | 피처: {len(features)}")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val_np = val_proc[TARGET_COL].values

    device = get_device()
    print(f"[Device] {device}")

    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    # CatBoost도 같이 학습해서(가벼움) attention 모델의 예측과 상관계수를 바로 비교한다.
    X_train_raw, y_train_raw = train_split[features], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[features], val_split[TARGET_COL].values
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[{label}] CatBoost(대조군) 학습 완료 ({time.time() - t0:.1f}s) | 점수: {cat_score:.2f}")

    t0 = time.time()
    if args.model == "ft":
        from code.ft_transformer_model import batched_forward
        if args.ensemble:
            from code.ft_transformer_model import ENSEMBLE_SEEDS as FT_SEEDS, train_ft_ensemble, predict_ft_ensemble
            members = train_ft_ensemble(
                X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, bin_edges=bin_edges,
                X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
                seeds=FT_SEEDS, device=device, verbose=False,
            )
            att_val_preds = predict_ft_ensemble(members, cat_dims, bin_edges, X_val_cat, X_val_num, device=device)
            best_epoch = [m["best_epoch"] for m in members]
        else:
            from code.ft_transformer_model import train_ft
            model, best_epoch = train_ft(
                X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, bin_edges=bin_edges,
                X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
                device=device, verbose=True,
            )
            att_val_preds = batched_forward(model, X_val_cat, X_val_num, device=device)
    else:
        from code.excelformer_model import compute_feature_ranking
        from code.ft_transformer_model import batched_forward
        ranking = compute_feature_ranking(catboost_model, features, CAT_COLS, num_cols)
        if args.ensemble:
            from code.excelformer_model import ENSEMBLE_SEEDS as EXCEL_SEEDS, train_excel_ensemble, predict_excel_ensemble
            members = train_excel_ensemble(
                X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, bin_edges=bin_edges,
                cat_cols=CAT_COLS, num_cols=num_cols, feature_ranking=ranking,
                X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
                seeds=EXCEL_SEEDS, device=device, verbose=False,
            )
            att_val_preds = predict_excel_ensemble(members, cat_dims, bin_edges, ranking, X_val_cat, X_val_num, device=device)
            best_epoch = [m["best_epoch"] for m in members]
        else:
            from code.excelformer_model import train_excel
            model, best_epoch = train_excel(
                X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, bin_edges=bin_edges,
                cat_cols=CAT_COLS, num_cols=num_cols, feature_ranking=ranking,
                X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
                device=device, verbose=True,
            )
            att_val_preds = batched_forward(model, X_val_cat, X_val_num, device=device)

    att_score = compute_bss(att_val_preds, y_val_raw)[2]
    print(f"[{label}] {args.model.upper()} 학습 완료 (best_epoch={best_epoch}, {time.time() - t0:.1f}s) | 점수: {att_score:.2f}")

    corr = np.corrcoef(cat_val_preds, att_val_preds)[0, 1]
    print(f"\n[RESULT {label}] CatBoost={cat_score:.2f} | {args.model.upper()}={att_score:.2f} | corr(catboost,{args.model})={corr:.4f}")

    # MLP은 항상 이 holdout에 맞춰 새로 학습한다(7-seed 앙상블). 프로덕션
    # reference bundle(open/reference/best_model.pkl)은 항상 season<2024로
    # 학습돼 있어 holdout=2024에서는 우연히 올바른 out-of-sample 비교가 되지만,
    # holdout=2023에서는 그 bundle이 이미 season==2023을 학습에서 본 상태라
    # in-sample 평가가 되어버린다(실측 시도: MLP=1403.42라는 비정상 고득점이 나옴).
    t0 = time.time()
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    mlp_members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
        seeds=ENSEMBLE_SEEDS, device=device, verbose=False,
    )
    mlp_val_preds = predict_ensemble(
        mlp_members, cat_dims, len(num_cols), embed_dims, X_val_cat, X_val_num,
        bin_edges=bin_edges, device=device,
    )
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
    corr_mlp = np.corrcoef(mlp_val_preds, att_val_preds)[0, 1]
    print(f"[{label}] MLP 앙상블(이 holdout 전용, 7-seed) 학습 완료 ({time.time() - t0:.1f}s)")
    print(f"[RESULT {label}] MLP={mlp_score:.2f} | corr(mlp,{args.model})={corr_mlp:.4f}")

    from code.blend_model import fit_meta_model
    from code.experiment_3way_stack import fit_meta_model_n

    w_cat, w_mlp, intercept, blend2_score, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)
    print(f"[RESULT {label}] 2-way(CatBoost+MLP) Blend={blend2_score:.2f} (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f})")

    weights, intercept3, blend3_score, _ = fit_meta_model_n([cat_val_preds, mlp_val_preds, att_val_preds], y_val_raw)
    print(f"[RESULT {label}] 3-way(CatBoost+MLP+{args.model.upper()}) Blend={blend3_score:.2f} (weights={weights} intercept={intercept3:.3f})")
    print(f"[RESULT {label}] 3-way 대비 2-way 이득: {blend3_score - blend2_score:+.2f}")

    if args.foldcheck:
        run_foldcheck(args.model, val_split, cat_val_preds, mlp_val_preds, att_val_preds, y_val_raw)

    cache_path = f"./open/temp/experiment_attention_{args.model}_h{args.holdout}_preds.pkl"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "cat_val_preds": cat_val_preds, "mlp_val_preds": mlp_val_preds,
            "att_val_preds": att_val_preds, "y_val_raw": y_val_raw,
        }, f)
    print(f"[{label}] 예측값 캐시 저장: {cache_path}")


if __name__ == "__main__":
    main()
