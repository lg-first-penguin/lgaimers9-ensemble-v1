# code/experiment_attention_mlp.py
"""MLP 고도화 3단계: code/attention_mlp_model.py::AttentionTabularMLP(멤버 자체에 경량
self-attention 블록)가 현재 프로덕션 TabularMLP(단순 concat)보다 나은지 3-seed로
방향성부터 스크리닝한다. 데이터 준비는 code/experiment_nbins_finesweep.py::build_mlp_data와
동일(현재 프로덕션 cutoff7/F1필터/season-progression/coarse pitchmix 피처셋).

reverify 스텝은 3-seed 스크리닝에서 +16.39(앙상블 점수 기준, 노이즈 하한선 ~20점 바로
아래)로 애매하게 나온 뒤, ENSEMBLE_SEEDS(7개)로 attention 모델만 재학습해서 결론 낸다.
baseline은 이미 open/reference/best_model.pkl에 7-seed로 학습되어 있으므로 재학습하지
않고 그 번들의 CatBoost/MLP 예측을 그대로 재사용 — CatBoost 예측도 고정값으로 재사용해
블렌드 점수까지 비교한다(code/experiment_mlp_ensemble_weighting.py::build_val_split과
동일한 val 재구성 로직).

사용법:
  python -m code.experiment_attention_mlp --step screen
  python -m code.experiment_attention_mlp --step reverify
"""
import argparse
import pickle
import time

import numpy as np

from code.attention_mlp_model import predict_attention, train_mlp_attention
from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.experiment_nbins_finesweep import build_mlp_data
from code.experiment_season_progression import build_split
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
    train_mlp, QUANTILE_N_BINS,
)
from code.train import TE_RESIDUAL_COLS, apply_f1_filter, apply_te_residual_features

SCREEN_SEEDS = [42, 123, 7]
REF_MODEL_PATH = "./open/reference/best_model.pkl"


def run_screen():
    print("[data] 전처리된 텐서 준비 중...")
    data = build_mlp_data()
    device = get_device()
    bin_edges = fit_quantile_edges(data["X_tr_num"], n_bins=QUANTILE_N_BINS)

    # 주의: 개별 시드 점수의 "평균"이 아니라, 프로덕션과 동일하게 "예측값을 평균한 뒤
    # 하나의 앙상블 점수"를 낸다 — 개별-스코어 평균은 앙상블의 분산 감소 효과를 반영하지
    # 못해 n_bins 스윕 등 다른 실험과 비교가 안 되는 다른 지표가 된다(1차 실행에서 발견한
    # 버그: baseline 개별평균 623.25 vs 같은 n_bins=24의 3-seed 앙상블 점수 730.18).
    print(f"\n=== baseline(TabularMLP, 단순 concat) 3-seed ===")
    base_preds_list = []
    for seed in SCREEN_SEEDS:
        t0 = time.time()
        model, best_epoch = train_mlp(
            data["X_tr_cat"], data["X_tr_num"], data["y_tr"],
            cat_dims=data["cat_dims"], embed_dims=data["embed_dims"], bin_edges=bin_edges,
            X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
            device=device, verbose=False, seed=seed,
        )
        preds = predict_ensemble(
            [{"state_dict": model.state_dict()}], data["cat_dims"], len(data["num_cols"]), data["embed_dims"],
            data["X_val_cat"], data["X_val_num"], bin_edges=bin_edges, device=device,
        )
        base_preds_list.append(preds)
        solo_score = compute_bss(preds, data["y_val"])[2]
        print(f"  [baseline] seed={seed} best_epoch={best_epoch} 개별 Val Score={solo_score:.2f} ({time.time()-t0:.1f}s)")
    base_ensemble_score = compute_bss(np.mean(base_preds_list, axis=0), data["y_val"])[2]
    print(f"[baseline] 3-seed 앙상블 점수: {base_ensemble_score:.2f}")

    print(f"\n=== AttentionTabularMLP(d_token=8, n_heads=2, 1-layer) 3-seed ===")
    attn_preds_list = []
    for seed in SCREEN_SEEDS:
        t0 = time.time()
        model, best_epoch = train_mlp_attention(
            data["X_tr_cat"], data["X_tr_num"], data["y_tr"],
            cat_dims=data["cat_dims"], embed_dims=data["embed_dims"], bin_edges=bin_edges,
            X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
            device=device, verbose=False, seed=seed,
        )
        preds = predict_attention(model, data["X_val_cat"], data["X_val_num"], device=device)
        attn_preds_list.append(preds)
        solo_score = compute_bss(preds, data["y_val"])[2]
        print(f"  [attention] seed={seed} best_epoch={best_epoch} 개별 Val Score={solo_score:.2f} ({time.time()-t0:.1f}s)")
    attn_ensemble_score = compute_bss(np.mean(attn_preds_list, axis=0), data["y_val"])[2]
    print(f"[attention] 3-seed 앙상블 점수: {attn_ensemble_score:.2f}")

    delta = attn_ensemble_score - base_ensemble_score
    print(f"\n{'='*70}\ndelta(attention vs baseline, 앙상블 점수 기준): {delta:+.2f}\n{'='*70}")


def run_reverify():
    from code.experiment_mlp_ensemble_weighting import build_val_split

    print("[data] MLP 텐서 준비 중...")
    data = build_mlp_data()
    device = get_device()
    bin_edges = fit_quantile_edges(data["X_tr_num"], n_bins=QUANTILE_N_BINS)

    print("[data] CatBoost/reference용 val split 재구성 중...")
    X_val_df, y_val_check = build_val_split()
    assert np.array_equal(y_val_check, data["y_val"]), "build_mlp_data와 build_val_split의 y_val 행 순서가 다릅니다"

    with open(REF_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    cat_feature_cols = bundle.get("cat_feature_cols")
    cat_df = X_val_df[cat_feature_cols] if cat_feature_cols is not None else X_val_df
    cat_preds = predict_catboost(bundle["catboost_model"], cat_df)
    cat_score = compute_bss(cat_preds, data["y_val"])[2]

    ref_mlp_bundle = bundle["mlp_bundle"]
    n_ref_members = len(ref_mlp_bundle["members"])
    ref_mlp_preds = predict_ensemble(
        ref_mlp_bundle["members"], ref_mlp_bundle["cat_dims"], len(ref_mlp_bundle["num_cols"]),
        ref_mlp_bundle["embed_dims"], data["X_val_cat"], data["X_val_num"],
        bin_edges=ref_mlp_bundle.get("bin_edges"), device=device,
    )
    ref_mlp_score = compute_bss(ref_mlp_preds, data["y_val"])[2]
    _, _, _, ref_blend_score, _ = fit_meta_model(cat_preds, ref_mlp_preds, data["y_val"])
    print(f"[baseline(reference 번들 재사용, {n_ref_members}-seed)] CatBoost={cat_score:.2f} | MLP={ref_mlp_score:.2f} | Blend={ref_blend_score:.2f}")

    print(f"\n=== AttentionTabularMLP {len(ENSEMBLE_SEEDS)}-seed 재학습 ===")
    attn_preds_list = []
    for seed in ENSEMBLE_SEEDS:
        t0 = time.time()
        model, best_epoch = train_mlp_attention(
            data["X_tr_cat"], data["X_tr_num"], data["y_tr"],
            cat_dims=data["cat_dims"], embed_dims=data["embed_dims"], bin_edges=bin_edges,
            X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
            device=device, verbose=False, seed=seed,
        )
        preds = predict_attention(model, data["X_val_cat"], data["X_val_num"], device=device)
        attn_preds_list.append(preds)
        solo_score = compute_bss(preds, data["y_val"])[2]
        print(f"  [attention] seed={seed} best_epoch={best_epoch} 개별 Val Score={solo_score:.2f} ({time.time()-t0:.1f}s)")
    attn_ensemble_pred = np.mean(attn_preds_list, axis=0)
    attn_mlp_score = compute_bss(attn_ensemble_pred, data["y_val"])[2]
    _, _, _, attn_blend_score, _ = fit_meta_model(cat_preds, attn_ensemble_pred, data["y_val"])
    print(f"[attention({len(ENSEMBLE_SEEDS)}-seed)] CatBoost={cat_score:.2f}(동일) | MLP={attn_mlp_score:.2f} | Blend={attn_blend_score:.2f}")

    print(f"\n{'='*70}")
    print(f"delta: MLP {attn_mlp_score-ref_mlp_score:+.2f} | Blend {attn_blend_score-ref_blend_score:+.2f}")
    print(f"{'='*70}")


def run_screen_2023():
    """season==2023 홀드아웃 듀얼레짐 확인 — cutoff7과 달리 재사용할 reference 번들이
    없으므로 CatBoost/baseline MLP를 이 스크립트 안에서 3-seed로 새로 학습한다."""
    print("[data] season==2023 홀드아웃 split 구성 중...")
    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(2023, cutoff7=False)

    drop_cols = ["row_id", "control_success"]
    features = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols]
    cat_feature_cols = [c for c in features if c not in trk_mlp_cols]

    train_split = df.loc[train_mask, features + ["control_success"]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + ["control_success"]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    print(f"[data] 훈련 {len(train_split)}행 | 검증 {len(val_split)}행")

    te_prior = train_split["control_success"].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)
    cat_feature_cols = cat_feature_cols + TE_RESIDUAL_COLS

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, "control_success")
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, num_cols, "control_success")
    y_val = val_proc["control_success"].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)
    device = get_device()

    print("\n=== CatBoost baseline 학습 ===")
    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split["control_success"].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split["control_success"].values
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_preds, y_val)[2]
    print(f"[CatBoost] Score={cat_score:.2f} (best_iteration={catboost_best_iteration}, {time.time()-t0:.1f}s)")

    print("\n=== baseline(TabularMLP) 3-seed ===")
    base_preds_list = []
    for seed in SCREEN_SEEDS:
        t0 = time.time()
        members = train_ensemble(
            X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val, seeds=[seed], device=device,
        )
        preds = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
        base_preds_list.append(preds)
        print(f"  [baseline] seed={seed} 개별 Val Score={compute_bss(preds, y_val)[2]:.2f} ({time.time()-t0:.1f}s)")
    base_mlp_pred = np.mean(base_preds_list, axis=0)
    base_mlp_score = compute_bss(base_mlp_pred, y_val)[2]
    _, _, _, base_blend_score, _ = fit_meta_model(cat_preds, base_mlp_pred, y_val)
    print(f"[baseline] MLP={base_mlp_score:.2f} | Blend={base_blend_score:.2f}")

    print("\n=== AttentionTabularMLP 3-seed ===")
    attn_preds_list = []
    for seed in SCREEN_SEEDS:
        t0 = time.time()
        model, best_epoch = train_mlp_attention(
            X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val, device=device, verbose=False, seed=seed,
        )
        preds = predict_attention(model, X_val_cat, X_val_num, device=device)
        attn_preds_list.append(preds)
        print(f"  [attention] seed={seed} best_epoch={best_epoch} 개별 Val Score={compute_bss(preds, y_val)[2]:.2f} ({time.time()-t0:.1f}s)")
    attn_mlp_pred = np.mean(attn_preds_list, axis=0)
    attn_mlp_score = compute_bss(attn_mlp_pred, y_val)[2]
    _, _, _, attn_blend_score, _ = fit_meta_model(cat_preds, attn_mlp_pred, y_val)
    print(f"[attention] MLP={attn_mlp_score:.2f} | Blend={attn_blend_score:.2f}")

    print(f"\n{'='*70}")
    print(f"[season==2023] delta: MLP {attn_mlp_score-base_mlp_score:+.2f} | Blend {attn_blend_score-base_blend_score:+.2f}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["screen", "reverify", "screen2023"])
    args = parser.parse_args()
    if args.step == "screen":
        run_screen()
    elif args.step == "reverify":
        run_reverify()
    else:
        run_screen_2023()


if __name__ == "__main__":
    main()
