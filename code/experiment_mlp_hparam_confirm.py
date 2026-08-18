# code/experiment_mlp_hparam_confirm.py
"""`code/tune_mlp.py`의 3-seed Optuna 스크리닝에서 나온 최적 하이퍼파라미터
(lr=0.008696, weight_decay=0.008255, dropout=0.488, batch_size=2048, Best trial 010,
Score=809.97 vs 다른 저성능 트라이얼들 576~705)를 7-seed 앙상블 + CatBoost 블렌드로,
2024/2023 두 홀드아웃 모두에서 재검증한다. 이 프로젝트의 반복된 패턴(§22/§28)대로
3-seed 스크리닝 결과는 그 자체로 프로덕션에 반영하지 않는다.

사용법:
  python -m code.experiment_mlp_hparam_confirm --holdout 2024
  python -m code.experiment_mlp_hparam_confirm --holdout 2023
"""
import argparse

import code.mlp_model as mlp_model
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, LR, WEIGHT_DECAY, DROPOUT, BATCH_SIZE,
    apply_preprocessing, compute_bss, embed_dim_for_cardinality, fit_preprocessing,
    fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model
from code.experiment_attention import build_split

BASELINE_PARAMS = dict(lr=LR, weight_decay=WEIGHT_DECAY, dropout=DROPOUT, batch_size=BATCH_SIZE)
TUNED_PARAMS = dict(lr=0.008696142413290007, weight_decay=0.008255473227402412, dropout=0.48781484431514655, batch_size=2048)


def run_mlp_variant(train_split, val_split, num_cols, device, params, seeds):
    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, num_cols, "control_success")
    X_va_cat, X_va_num, y_va = to_tensors(va_proc, CAT_COLS, num_cols, "control_success")

    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    mlp_model.DROPOUT = params["dropout"]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_va,
        seeds=seeds, lr=params["lr"], weight_decay=params["weight_decay"], batch_size=params["batch_size"],
        device=device, verbose=False,
    )
    preds = predict_ensemble(
        members, cat_dims, len(num_cols), embed_dims, X_va_cat, X_va_num,
        bin_edges=bin_edges, device=device,
    )
    return compute_bss(preds, y_va.numpy())[2], preds


def run(holdout, seeds=ENSEMBLE_SEEDS):
    train_split, val_split, features, num_cols = build_split(holdout, apply_f1=True)
    y_val = val_split["control_success"].values

    device = get_device()
    print(f"[Device] {device}")

    X_train_raw = train_split[features]
    y_train_raw = train_split["control_success"].values
    X_val_raw = val_split[features]
    catboost_model, _ = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=False)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val)[2]
    print(f"[CatBoost(고정)] Val Score: {cat_score:.2f}\n")

    results = {}
    for name, params in [("baseline", BASELINE_PARAMS), ("tuned", TUNED_PARAMS)]:
        mlp_score, mlp_val_preds = run_mlp_variant(train_split, val_split, num_cols, device, params, seeds)
        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val)
        results[name] = (mlp_score, blend_score)
        print(f"[{name}] params={params}")
        print(f"[{name}] MLP solo: {mlp_score:.2f} | Blend: {blend_score:.2f} (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f})")

    (base_mlp, base_blend), (tuned_mlp, tuned_blend) = results["baseline"], results["tuned"]
    print(f"\nMLP solo Delta: {tuned_mlp - base_mlp:+.2f}")
    print(f"Blend Delta: {tuned_blend - base_blend:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024)
    args = parser.parse_args()
    run(args.holdout)
