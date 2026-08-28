# code/build_ple_submit_bundle.py
"""B-with-PLE 제출 번들 빌드 (2026-08-28).

candidate B(유담 레시피 포트) 와 **MLP 인코딩만** 다르다:
  candidate B : 수치형 raw StandardScaler concat (bin_edges=None)
  이 번들     : 수치형 QuantileEmbedding / PLE (fit_quantile_edges)
나머지(피처셋=트랙맨64+reverse_rate+TE-residual+same_hand, CatBoost v2 HP 5-seed 배깅,
F1 필터, 메타모델 2-input 로지스틱, 정적 lookup CSV)는 전부 동일.

목적: "우리 파이프라인 실전 2025에서 PLE 가 득이냐 실이냐" 를 실전 리더보드로 판정.
로컬 근거는 엇갈림 — cutoff7 blend +20.90 / season2023 −17.41 / rolling-origin 2/4 fold
(memory: ple_removal_resweep_0of4_rejected). dual-regime 기준 REJECT 지만 신뢰 레짐
cutoff7 이 크게 +이고 붕괴-취약성 메커니즘은 반증돼서 "제출로만 결판나는 후보".

2단계:
  1) cutoff7 split 으로 PLE MLP 7-seed + CatBoost 5-seed 학습 -> val 로 메타 가중치,
     시드별 best_epoch, CatBoost best_iteration 확정 (train.py::main() cutoff7 경로와 동일,
     단 bin_edges 를 fit_quantile_edges 로).
  2) 전체 데이터 재학습(홀드아웃 없음) -> submit/model/final_retained_model.pkl.
     정적 lookup(season_end/rate_end/te_source)도 재생성해 submit/model/ 에 저장
     (레시피 동일이라 candidate B 것과 내용 동일, 자기완결성 위해 재생성).

실행 전 submit/ 를 백업할 것. 실행:
  python -m code.build_ple_submit_bundle
"""
import gc
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, train_mlp, make_bundle, predict_bundle,
    fit_quantile_edges, get_device, compute_bss,
)
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.blend_model import fit_meta_model, make_blend_bundle
from code.experiment_yudam_common import build_split
from code.train import (
    process_trackman_features_safe, apply_f1_filter, add_engineered_features,
    apply_te_residual_features, TE_RESIDUAL_COLS, build_season_end_lookup,
    build_rate_end_lookup, apply_same_hand, SAME_HAND_COLS,
    YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS,
)
from code.trackman_pitcher_features import merge_coarse_pitchmix

TARGET = "control_success"
ID = "row_id"
DATA_DIR = "./open/data"
EPOCH_BUFFER = 5
CATBOOST_ITER_BUFFER = 50
FINAL_PATH = "./submit/model/final_retained_model.pkl"


def _v2_params():
    with open("./teammate/yudam/model_py311/best_catboost_hparams_v2.json") as f:
        t = json.load(f)["best_params"]
    return dict(
        depth=t["depth"], learning_rate=t["learning_rate"], l2_leaf_reg=t["l2_leaf_reg"],
        random_strength=t["random_strength"], bagging_temperature=t["bagging_temperature"],
        border_count=t["border_count"], min_data_in_leaf=t["min_data_in_leaf"],
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    )


def stage1_cutoff7(device):
    print("\n" + "=" * 72 + "\n[1단계] cutoff7 split -> PLE 메타가중치 / best_epoch / cb_iter 확정\n" + "=" * 72, flush=True)
    ts, vs, num_cols, cat_feature_cols, all_cols = build_split(regime="cutoff7", verbose=True)

    tr_proc, ce, ni, nsc, cat_dims = fit_preprocessing(ts, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(vs, CAT_COLS, num_cols, ce, ni, nsc)
    Xtc, Xtn, ytr = to_tensors(tr_proc, CAT_COLS, num_cols, TARGET)
    Xvc, Xvn, _ = to_tensors(va_proc, CAT_COLS, num_cols, TARGET)
    y_val = va_proc[TARGET].values
    del tr_proc, va_proc
    gc.collect()

    bin_edges = fit_quantile_edges(Xtn)
    print(f"[1단계] bin_edges shape={tuple(bin_edges.shape)} (num_numeric x n_bins+1)", flush=True)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    members = train_ensemble(
        Xtc, Xtn, ytr, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
        embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=Xvc, X_val_num=Xvn, y_val=y_val,
        seeds=YUDAM_ENSEMBLE_SEEDS, device=device, verbose=True,
    )
    mlp_bundle_s1 = make_bundle(members, CAT_COLS, num_cols, cat_dims, embed_dims,
                                ce, ni, nsc, bin_edges=bin_edges)
    mlp_val = predict_bundle(mlp_bundle_s1, vs[all_cols], device=device)
    per_seed_epochs = [m["best_epoch"] for m in members]

    cb_res = train_catboost_ensemble(
        ts[cat_feature_cols], ts[TARGET].values, vs[cat_feature_cols], y_val,
        seeds=YUDAM_CATBOOST_SEEDS, verbose=True, params=_v2_params(),
    )
    cat_val = predict_catboost_ensemble([m for m, _ in cb_res], vs[cat_feature_cols])
    cb_best_iters = [it for _, it in cb_res]

    w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val, mlp_val, y_val)
    mlp_solo = compute_bss(mlp_val, y_val)[2]
    cat_solo = compute_bss(cat_val, y_val)[2]
    print(f"\n[1단계 결과] cutoff7  CatBoost solo={cat_solo:.2f} | PLE-MLP solo={mlp_solo:.2f} | "
          f"blend={blend_score*100000:.2f}", flush=True)
    print(f"[1단계 결과] meta w_cat={w_cat:.4f} w_mlp={w_mlp:.4f} intercept={intercept:.4f}", flush=True)
    print(f"[1단계 결과] per_seed_epochs={per_seed_epochs} | cb_best_iters={cb_best_iters}", flush=True)

    del ts, vs, members, mlp_bundle_s1, Xtc, Xtn, Xvc, Xvn, cb_res
    gc.collect()
    return dict(meta={"w_cat": w_cat, "w_mlp": w_mlp, "intercept": intercept},
                per_seed_epochs=per_seed_epochs, cb_best_iters=cb_best_iters)


def stage2_full_retrain(s1, device):
    print("\n" + "=" * 72 + "\n[2단계] 전체 데이터 재학습 -> submit/model/final_retained_model.pkl\n" + "=" * 72, flush=True)
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    tr_final, _mc, trackman_cols = process_trackman_features_safe(train_df_raw, df_trm, is_train_split=False)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    del tr_final, train_df_raw
    gc.collect()
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=None)
    print(f"[2단계][트랙맨] 물리조인 {len(trackman_cols)}컬럼 + coarse pitchmix 4컬럼", flush=True)

    season_end_lookup = build_season_end_lookup(train_df)
    season_end_lookup.to_csv("./submit/model/season_end_lookup.csv", index=False)
    rate_end_lookup = build_rate_end_lookup(train_df)
    rate_end_lookup.to_csv("./submit/model/rate_end_lookup.csv", index=False)
    print(f"[2단계] 시즌진행분 lookup 재저장 ({len(season_end_lookup)}행 / {len(rate_end_lookup)}행)", flush=True)

    league_success_mean = train_df[TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)
    train_df = apply_same_hand(train_df)
    train_df = apply_f1_filter(train_df)

    te_prior = train_df[TARGET].mean()
    te_source_cols = ["pitcher_id", "batter_id", "balls_before", "strikes_before",
                      "batter_hand", "num_runners_on", "inning", "season", TARGET]
    te_source = train_df[te_source_cols].copy()
    te_source.to_csv("./submit/model/te_source.csv", index=False)
    train_df = apply_te_residual_features(te_source, train_df, te_prior)
    print(f"[2단계] TE 소스 재저장 ({len(te_source)}행)", flush=True)
    del df_trm
    gc.collect()

    all_cols = [c for c in train_df.columns if c not in (ID, TARGET)]
    cat_feature_cols = [c for c in all_cols if c not in SAME_HAND_COLS]
    num_cols = [c for c in all_cols if c not in CAT_COLS and c not in TE_RESIDUAL_COLS]
    print(f"[2단계] {len(train_df)}행 | CatBoost {len(cat_feature_cols)}피처 / "
          f"MLP {len(num_cols) + len(CAT_COLS)}피처", flush=True)

    full_proc, ce, ni, nsc, cat_dims = fit_preprocessing(train_df, CAT_COLS, num_cols)
    Xfc, Xfn, yf = to_tensors(full_proc, CAT_COLS, num_cols, TARGET)
    del full_proc
    gc.collect()
    bin_edges = fit_quantile_edges(Xfn)
    print(f"[2단계] bin_edges shape={tuple(bin_edges.shape)}", flush=True)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    full_members = []
    for seed, base_epoch in zip(YUDAM_ENSEMBLE_SEEDS, s1["per_seed_epochs"]):
        full_epochs = max(base_epoch or 1, 1) + EPOCH_BUFFER
        model, _ = train_mlp(
            Xfc, Xfn, yf, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
            embed_dims=embed_dims, bin_edges=bin_edges, max_epochs=full_epochs,
            device=device, seed=seed,
        )
        full_members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": full_epochs, "seed": seed,
        })
        print(f"[2단계] MLP seed={seed} {full_epochs} epoch 완료", flush=True)
        del model
        gc.collect()

    final_mlp_bundle = make_bundle(full_members, CAT_COLS, num_cols, cat_dims, embed_dims,
                                   ce, ni, nsc, bin_edges=bin_edges)

    cb_full_iters = [it + CATBOOST_ITER_BUFFER for it in s1["cb_best_iters"]]
    print(f"[2단계] CatBoost 5-seed 재학습 iterations={cb_full_iters}", flush=True)
    cb_full = train_catboost_ensemble(
        train_df[cat_feature_cols], train_df[TARGET].values, seeds=YUDAM_CATBOOST_SEEDS,
        per_seed_iterations=cb_full_iters, verbose=True, params=_v2_params(),
    )
    final_catboost_models = [m for m, _ in cb_full]

    final_bundle = make_blend_bundle(final_catboost_models, final_mlp_bundle, s1["meta"],
                                     cat_feature_cols=cat_feature_cols)
    final_bundle["catboost_best_iteration"] = cb_full_iters[0]
    final_bundle["catboost_best_iterations"] = cb_full_iters

    os.makedirs(os.path.dirname(FINAL_PATH), exist_ok=True)
    with open(FINAL_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"\n[완료] {FINAL_PATH} 저장. meta={s1['meta']}", flush=True)
    print(f"[완료] mlp_bundle 에 bin_edges 포함: {final_bundle['mlp_bundle'].get('bin_edges') is not None}", flush=True)


def main():
    torch.manual_seed(0)
    device = get_device()
    print(f"device={device}", flush=True)
    s1 = stage1_cutoff7(device)
    gc.collect()
    stage2_full_retrain(s1, device)


if __name__ == "__main__":
    main()
