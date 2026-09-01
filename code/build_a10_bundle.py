# code/build_a10_bundle.py
"""A + CatBoost 10-seed 헤지 번들 (2026-09-01, 마감).

현행 프로덕션(cat_team, 실전 1126.77) 대비 두 가지만 다르다 — 둘 다 순수 분산감소,
구조적 주장 없음:
  A  : MLP 7-seed full-retrain epoch 을 per-seed(cutoff7 best_epoch + 5, 평균 ~21)
       -> **flat 25** (유담 full_retrain_blend_f1.py 관례). 4개 시드가 19~21 -> 25.
  10 : CatBoost 배깅 5-seed -> **10-seed**.
메타(w_cat=2.0117/w_mlp=1.9848/int=−2.0284) · lookup 3종 · 피처셋 · v2 HP · team_id
categorical · F1 필터 · 트랙맨64 제거 전부 프로덕션과 동일. cutoff7 stage1 없음
(full-data 스케일 효과라 홀드아웃 스크린 불가) -> 바로 빌드 후 실전 제출.

빌드 파이프라인은 code/build_catteam_bundle.py::stage2_full_retrain 과 한 글자도
다르지 않게(트랙맨 조인 -> merge_coarse_pitchmix -> lookup 재생성 -> add_engineered
-> same_hand -> F1 -> TE-residual) 재현하고, MLP epoch 과 CatBoost seed 수만 바꾼다.

실행:
  python -m code.build_a10_bundle                 # 빌드 -> ./scratchpad/submit_a10/
  python -m code.build_a10_bundle --cb-seeds 7    # CatBoost 7-seed (시간 절약)
  python -m code.build_a10_bundle --promote-only  # 스테이징 -> submit/ 반영 + 백업
"""
import argparse
import gc
import json
import os
import pickle
import shutil
import time

import pandas as pd
import torch

from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, to_tensors,
    train_mlp, make_bundle, get_device,
)
from code.catboost_model import train_catboost_ensemble
from code.blend_model import make_blend_bundle
from code.train import (
    process_trackman_features_safe, apply_f1_filter, add_engineered_features,
    apply_te_residual_features, apply_same_hand, SAME_HAND_COLS, TE_RESIDUAL_COLS,
    is_trackman64, build_season_end_lookup, build_rate_end_lookup,
    YUDAM_ENSEMBLE_SEEDS,
)
from code.trackman_pitcher_features import merge_coarse_pitchmix

TARGET = "control_success"
ID = "row_id"
DATA_DIR = "./open/data"
PROD_PKL = "./submit/model/final_retained_model.pkl"
EXTRA_CAT = ["pitcher_team_id", "batter_team_id"]
CB_CAT_FEATURES = ["game_type", "base_state"] + EXTRA_CAT
CATBOOST_ITER_BUFFER = 50
MLP_FLAT_EPOCHS = 25
CB_SEED_POOL = [42, 123, 7, 2024, 99, 555, 31337, 1, 2, 3]   # 앞 N개
STAGING_DIR = "./scratchpad/submit_a10"
STAGING_PATH = os.path.join(STAGING_DIR, "final_retained_model.pkl")
MLP_CKPT = os.path.join(STAGING_DIR, "mlp_bundle_ckpt.pkl")   # stage 분리용 중간 체크포인트


def _v2_params():
    with open("./teammate/yudam/model_py311/best_catboost_hparams_v2.json") as f:
        t = json.load(f)["best_params"]
    return dict(
        depth=t["depth"], learning_rate=t["learning_rate"], l2_leaf_reg=t["l2_leaf_reg"],
        random_strength=t["random_strength"], bagging_temperature=t["bagging_temperature"],
        border_count=t["border_count"], min_data_in_leaf=t["min_data_in_leaf"],
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    )


def _cast_team_str(df, cols):
    df = df.copy()
    for c in EXTRA_CAT:
        if c in cols:
            df[c] = df[c].astype(str)
    return df


def _build_train_df():
    """build_catteam_bundle.stage2_full_retrain 과 동일한 피처 파이프라인.
    lookup/te_source CSV 는 STAGING_DIR 에 저장. (train_df, cat_feature_cols, num_cols, cb_cat) 반환."""
    os.makedirs(STAGING_DIR, exist_ok=True)
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    tr_final, _mc, _tc = process_trackman_features_safe(train_df_raw, df_trm, is_train_split=False)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    del tr_final, train_df_raw
    gc.collect()
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=None)

    build_season_end_lookup(train_df).to_csv(os.path.join(STAGING_DIR, "season_end_lookup.csv"), index=False)
    build_rate_end_lookup(train_df).to_csv(os.path.join(STAGING_DIR, "rate_end_lookup.csv"), index=False)

    league_success_mean = train_df[TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)
    train_df = apply_same_hand(train_df)
    train_df = apply_f1_filter(train_df)

    te_prior = train_df[TARGET].mean()
    te_source_cols = ["pitcher_id", "batter_id", "balls_before", "strikes_before",
                      "batter_hand", "num_runners_on", "inning", "season", TARGET]
    te_source = train_df[te_source_cols].copy()
    te_source.to_csv(os.path.join(STAGING_DIR, "te_source.csv"), index=False)
    train_df = apply_te_residual_features(te_source, train_df, te_prior)
    del df_trm
    gc.collect()

    all_cols = [c for c in train_df.columns if c not in (ID, TARGET)]
    cat_feature_cols = [c for c in all_cols if c not in SAME_HAND_COLS and not is_trackman64(c)]
    num_cols = [c for c in all_cols if c not in CAT_COLS and c not in TE_RESIDUAL_COLS and not is_trackman64(c)]
    assert not any(is_trackman64(c) for c in cat_feature_cols + num_cols)
    cb_cat = [c for c in CB_CAT_FEATURES if c in cat_feature_cols]

    with open(PROD_PKL, "rb") as f:
        prod = pickle.load(f)
    assert len(cat_feature_cols) == len(prod["cat_feature_cols"]), \
        f"cat_feature_cols {len(cat_feature_cols)} != prod {len(prod['cat_feature_cols'])}"
    assert len(num_cols) == len(prod["mlp_bundle"]["num_cols"]), \
        f"num_cols {len(num_cols)} != prod {len(prod['mlp_bundle']['num_cols'])}"
    print(f"[A10] {len(train_df)}행 | CatBoost {len(cat_feature_cols)}피처 / MLP num {len(num_cols)} | cat_features={cb_cat}", flush=True)
    return train_df, cat_feature_cols, num_cols, cb_cat


def stage_mlp(device):
    """MLP 7-seed flat-25 만 학습 -> MLP_CKPT 저장. (CUDA 컨텍스트는 이 프로세스 종료 시 정리)"""
    print("\n" + "=" * 72 + f"\n[A10 stage=mlp] MLP flat-{MLP_FLAT_EPOCHS} {len(YUDAM_ENSEMBLE_SEEDS)}-seed\n" + "=" * 72, flush=True)
    train_df, cat_feature_cols, num_cols, _cb_cat = _build_train_df()
    full_proc, ce, ni, nsc, cat_dims = fit_preprocessing(train_df, CAT_COLS, num_cols)
    Xfc, Xfn, yf = to_tensors(full_proc, CAT_COLS, num_cols, TARGET)
    del full_proc, train_df
    gc.collect()
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    members = []
    for seed in YUDAM_ENSEMBLE_SEEDS:
        model, _ = train_mlp(
            Xfc, Xfn, yf, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
            embed_dims=embed_dims, bin_edges=None, max_epochs=MLP_FLAT_EPOCHS, device=device, seed=seed,
        )
        members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": MLP_FLAT_EPOCHS, "seed": seed,
        })
        print(f"[A10 stage=mlp] seed={seed} flat {MLP_FLAT_EPOCHS} epoch 완료", flush=True)
        del model
        gc.collect()
    mlp_bundle = make_bundle(members, CAT_COLS, num_cols, cat_dims, embed_dims, ce, ni, nsc, bin_edges=None)
    with open(MLP_CKPT, "wb") as f:
        pickle.dump({"mlp_bundle": mlp_bundle, "num_cols": num_cols, "cat_feature_cols": cat_feature_cols}, f)
    print(f"[A10 stage=mlp] 체크포인트 저장 -> {MLP_CKPT} (num_cols {len(num_cols)})", flush=True)


def stage_cb(cb_nseed):
    """MLP_CKPT 로드 + CatBoost cb_nseed-seed 학습(CPU) + 최종 번들 assemble. torch/CUDA 미사용."""
    print("\n" + "=" * 72 + f"\n[A10 stage=cb] CatBoost {cb_nseed}-seed + assemble\n" + "=" * 72, flush=True)
    assert os.path.exists(MLP_CKPT), f"{MLP_CKPT} 없음 — 먼저 --stage mlp"
    with open(MLP_CKPT, "rb") as f:
        ck = pickle.load(f)
    mlp_bundle = ck["mlp_bundle"]
    with open(PROD_PKL, "rb") as f:
        prod = pickle.load(f)
    meta = prod["meta_model"]
    prod_iters = prod["catboost_best_iterations"]
    base_iters = [it + CATBOOST_ITER_BUFFER for it in prod_iters]
    cb_seeds = CB_SEED_POOL[:cb_nseed]
    cb_full_iters = [base_iters[i % len(base_iters)] for i in range(cb_nseed)]
    print(f"[A10 stage=cb] 메타 재사용={meta} | seeds={cb_seeds} | iters={cb_full_iters}", flush=True)

    train_df, cat_feature_cols, num_cols, cb_cat = _build_train_df()
    assert cat_feature_cols == ck["cat_feature_cols"], "cat_feature_cols 가 MLP 체크포인트와 불일치"
    assert num_cols == ck["num_cols"], "num_cols 가 MLP 체크포인트와 불일치"

    cb_full = train_catboost_ensemble(
        _cast_team_str(train_df[cat_feature_cols], cat_feature_cols), train_df[TARGET].values,
        seeds=cb_seeds, per_seed_iterations=cb_full_iters, verbose=True,
        params=_v2_params(), cat_features=cb_cat,
    )
    final_catboost_models = [m for m, _ in cb_full]

    final_bundle = make_blend_bundle(final_catboost_models, mlp_bundle, meta, cat_feature_cols=cat_feature_cols)
    final_bundle["catboost_best_iteration"] = cb_full_iters[0]
    final_bundle["catboost_best_iterations"] = cb_full_iters
    final_bundle["catboost_extra_cat"] = list(EXTRA_CAT)
    with open(STAGING_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"\n[A10 완료] {STAGING_PATH}", flush=True)
    print(f"[A10 완료] MLP {len(mlp_bundle['members'])}-seed flat{MLP_FLAT_EPOCHS} / CatBoost {len(final_catboost_models)}-seed "
          f"/ cat_feature_cols {len(cat_feature_cols)} / num_cols {len(num_cols)} / "
          f"bin_edges None={final_bundle['mlp_bundle'].get('bin_edges') is None} / meta={meta}", flush=True)


def promote():
    ts = time.strftime("%Y%m%d_%H%M%S")
    bdir = f"./open/former_model/submit_pre_a10_{ts}"
    os.makedirs(bdir, exist_ok=True)
    for fn in ("final_retained_model.pkl", "season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        src = os.path.join("./submit/model", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(bdir, fn))
    shutil.copy2("./submit/script.py", os.path.join(bdir, "script.py"))
    print(f"[promote] 기존 submit/ 백업 -> {bdir}", flush=True)
    for fn in ("final_retained_model.pkl", "season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        src = os.path.join(STAGING_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join("./submit/model", fn))
    with open("./submit/script.py") as fh:
        assert "catboost_extra_cat" in fh.read(), "submit/script.py 에 catboost_extra_cat 패치 없음 — cat_team 빌드 먼저"
    print("[promote] STAGING -> submit/model/ 복사 완료. (script.py = cat_team 패치 그대로)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["mlp", "cb", "all"], default="all",
                   help="mlp=MLP만 학습+체크포인트 / cb=체크포인트 로드+CatBoost+assemble / all=순차")
    ap.add_argument("--cb-seeds", type=int, default=7)
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--promote-only", action="store_true")
    args = ap.parse_args()
    if args.promote_only:
        promote()
        return
    assert 1 <= args.cb_seeds <= len(CB_SEED_POOL)
    torch.manual_seed(0)
    print(f"stage={args.stage} | cb_seeds={args.cb_seeds}", flush=True)
    if args.stage in ("mlp", "all"):
        device = get_device()
        print(f"device={device}", flush=True)
        stage_mlp(device)
    if args.stage in ("cb", "all"):
        stage_cb(args.cb_seeds)
    if args.promote:
        promote()


if __name__ == "__main__":
    main()
