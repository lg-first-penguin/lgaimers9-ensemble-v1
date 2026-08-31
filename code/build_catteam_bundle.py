# code/build_catteam_bundle.py
"""cat_team 제출 번들 빌드 (2026-08-31).

현행 프로덕션(1117.03 = candidate B raw − 트랙맨64) 과 **pitcher_team_id /
batter_team_id 를 CatBoost cat_features 로 선언(str 캐스팅)** 한 점만 다르다.
나머지(raw-concat MLP 7-seed, CatBoost v2 HP 5-seed, coarse pitchmix, season-
progression, reverse_rate 진행분, Track A TE-residual→CatBoost, same_hand→MLP,
F1 필터, 트랙맨64 제거, 2-input 로지스틱 메타)는 전부 동일.

근거 (`code/experiment_yudam_id_cat_sweep.py` + `code/experiment_yudam_catteam_pidmlp.py`):
team_id 는 카디널리티 ~10 + 팀당 ~13만 행이라 CatBoost ordered target-statistic 이
raw-numeric 분할을 이긴다. CatBoost solo Δ 가 데이터量에 단조 (+21.87@1.26M / −0.01@871k
/ −32.20@654k / −101.86@432k) — 2021/2022 음수는 소량-데이터 아티팩트, 프로덕션(137만)
엔 안 옮겨감. blend Δ cutoff7 +5.53 / 2023 +2.47 (실전near 두 레짐 다 +). MLP 임베딩엔
team_id 그대로 유지(빼면 cutoff7 −14~−31). pid MLP 제거는 2023 마이너스라 미포함.

2단계 (build_trackman64_removed_bundle.py 와 동일 구조):
  1) cutoff7 split -> raw MLP 7-seed + CatBoost 5-seed(team categorical) -> 메타가중치 /
     시드별 best_epoch / cb best_iteration 확정.
  2) 전체 1.37M 재학습 -> STAGING 저장. 정적 lookup 3종 재생성(내용은 프로덕션과 동일).

번들에 catboost_extra_cat=["pitcher_team_id","batter_team_id"] 키 추가 -> submit/script.py
가 추론 시 그 컬럼을 .astype(str) 해서 학습때와 표현 일치시킴.

실행:  python -m code.build_catteam_bundle
       python -m code.build_catteam_bundle --promote   # submit/ 반영(백업+script.py 패치)
"""
import argparse
import gc
import json
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd
import torch

from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, train_mlp, make_bundle, predict_bundle,
    get_device, compute_bss,
)
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.blend_model import fit_meta_model, make_blend_bundle
from code.experiment_yudam_common import build_split
from code.train import (
    process_trackman_features_safe, apply_f1_filter, add_engineered_features,
    apply_te_residual_features, TE_RESIDUAL_COLS, build_season_end_lookup,
    build_rate_end_lookup, apply_same_hand, SAME_HAND_COLS, is_trackman64,
    YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS,
)
from code.trackman_pitcher_features import merge_coarse_pitchmix

TARGET = "control_success"
ID = "row_id"
DATA_DIR = "./open/data"
EPOCH_BUFFER = 5
CATBOOST_ITER_BUFFER = 50
EXTRA_CAT = ["pitcher_team_id", "batter_team_id"]
CB_CAT_FEATURES = ["game_type", "base_state"] + EXTRA_CAT
STAGING_DIR = "./scratchpad/submit_catteam"
STAGING_PATH = os.path.join(STAGING_DIR, "final_retained_model.pkl")


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


def stage1_cutoff7(device):
    print("\n" + "=" * 72 + "\n[1단계] cutoff7 -> 메타 / best_epoch / cb_iter (team categorical)\n" + "=" * 72, flush=True)
    ts, vs, num_cols, cat_feature_cols, all_cols = build_split(regime="cutoff7", verbose=True)
    assert not any(is_trackman64(c) for c in num_cols + cat_feature_cols), "트랙맨64 가 아직 피처목록에"
    cb_cat = [c for c in CB_CAT_FEATURES if c in cat_feature_cols]
    print(f"[1단계] MLP num {len(num_cols)} / CatBoost {len(cat_feature_cols)} | cat_features={cb_cat}", flush=True)

    tr_proc, ce, ni, nsc, cat_dims = fit_preprocessing(ts, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(vs, CAT_COLS, num_cols, ce, ni, nsc)
    Xtc, Xtn, ytr = to_tensors(tr_proc, CAT_COLS, num_cols, TARGET)
    Xvc, Xvn, _ = to_tensors(va_proc, CAT_COLS, num_cols, TARGET)
    y_val = va_proc[TARGET].values
    del tr_proc, va_proc
    gc.collect()

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        Xtc, Xtn, ytr, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
        embed_dims=embed_dims, bin_edges=None,
        X_val_cat=Xvc, X_val_num=Xvn, y_val=y_val,
        seeds=YUDAM_ENSEMBLE_SEEDS, device=device, verbose=True,
    )
    mlp_bundle_s1 = make_bundle(members, CAT_COLS, num_cols, cat_dims, embed_dims, ce, ni, nsc, bin_edges=None)
    mlp_val = predict_bundle(mlp_bundle_s1, vs[all_cols], device=device)
    per_seed_epochs = [m["best_epoch"] for m in members]

    cb_res = train_catboost_ensemble(
        _cast_team_str(ts[cat_feature_cols], cat_feature_cols), ts[TARGET].values,
        _cast_team_str(vs[cat_feature_cols], cat_feature_cols), y_val,
        seeds=YUDAM_CATBOOST_SEEDS, verbose=True, params=_v2_params(), cat_features=cb_cat,
    )
    cat_val = predict_catboost_ensemble([m for m, _ in cb_res], _cast_team_str(vs[cat_feature_cols], cat_feature_cols))
    cb_best_iters = [it for _, it in cb_res]

    w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val, mlp_val, y_val)
    mlp_solo = compute_bss(mlp_val, y_val)[2]
    cat_solo = compute_bss(cat_val, y_val)[2]
    print(f"\n[1단계 결과] cutoff7 CatBoost solo={cat_solo:.2f} | raw-MLP solo={mlp_solo:.2f} | blend={blend_score * 100000:.2f}", flush=True)
    print(f"[1단계 결과] (참고: 프로덕션 1117 cutoff7 stage1 blend ~729, team categorical solo는 +약20 기대)", flush=True)
    print(f"[1단계 결과] meta w_cat={w_cat:.4f} w_mlp={w_mlp:.4f} intercept={intercept:.4f}", flush=True)
    print(f"[1단계 결과] per_seed_epochs={per_seed_epochs} | cb_best_iters={cb_best_iters}", flush=True)

    del ts, vs, members, mlp_bundle_s1, Xtc, Xtn, Xvc, Xvn, cb_res
    gc.collect()
    return dict(meta={"w_cat": w_cat, "w_mlp": w_mlp, "intercept": intercept},
                per_seed_epochs=per_seed_epochs, cb_best_iters=cb_best_iters)


def stage2_full_retrain(s1, device):
    print("\n" + "=" * 72 + f"\n[2단계] 전체 재학습 (team categorical) -> {STAGING_PATH}\n" + "=" * 72, flush=True)
    os.makedirs(STAGING_DIR, exist_ok=True)
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    tr_final, _mc, trackman_cols = process_trackman_features_safe(train_df_raw, df_trm, is_train_split=False)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    del tr_final, train_df_raw
    gc.collect()
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=None)

    season_end_lookup = build_season_end_lookup(train_df)
    season_end_lookup.to_csv(os.path.join(STAGING_DIR, "season_end_lookup.csv"), index=False)
    rate_end_lookup = build_rate_end_lookup(train_df)
    rate_end_lookup.to_csv(os.path.join(STAGING_DIR, "rate_end_lookup.csv"), index=False)
    print(f"[2단계] lookup 저장 ({len(season_end_lookup)}행 / {len(rate_end_lookup)}행)", flush=True)

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
    print(f"[2단계] {len(train_df)}행 | CatBoost {len(cat_feature_cols)}피처 / MLP {len(num_cols) + len(CAT_COLS)}피처 | cat_features={cb_cat}", flush=True)

    full_proc, ce, ni, nsc, cat_dims = fit_preprocessing(train_df, CAT_COLS, num_cols)
    Xfc, Xfn, yf = to_tensors(full_proc, CAT_COLS, num_cols, TARGET)
    del full_proc
    gc.collect()
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    full_members = []
    for seed, base_epoch in zip(YUDAM_ENSEMBLE_SEEDS, s1["per_seed_epochs"]):
        full_epochs = max(base_epoch or 1, 1) + EPOCH_BUFFER
        model, _ = train_mlp(
            Xfc, Xfn, yf, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
            embed_dims=embed_dims, bin_edges=None, max_epochs=full_epochs, device=device, seed=seed,
        )
        full_members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": full_epochs, "seed": seed,
        })
        print(f"[2단계] MLP seed={seed} {full_epochs} epoch 완료", flush=True)
        del model
        gc.collect()

    final_mlp_bundle = make_bundle(full_members, CAT_COLS, num_cols, cat_dims, embed_dims, ce, ni, nsc, bin_edges=None)

    cb_full_iters = [it + CATBOOST_ITER_BUFFER for it in s1["cb_best_iters"]]
    print(f"[2단계] CatBoost 5-seed 재학습 iterations={cb_full_iters}", flush=True)
    cb_full = train_catboost_ensemble(
        _cast_team_str(train_df[cat_feature_cols], cat_feature_cols), train_df[TARGET].values,
        seeds=YUDAM_CATBOOST_SEEDS, per_seed_iterations=cb_full_iters, verbose=True,
        params=_v2_params(), cat_features=cb_cat,
    )
    final_catboost_models = [m for m, _ in cb_full]

    final_bundle = make_blend_bundle(final_catboost_models, final_mlp_bundle, s1["meta"], cat_feature_cols=cat_feature_cols)
    final_bundle["catboost_best_iteration"] = cb_full_iters[0]
    final_bundle["catboost_best_iterations"] = cb_full_iters
    final_bundle["catboost_extra_cat"] = list(EXTRA_CAT)

    with open(STAGING_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"\n[완료] {STAGING_PATH} 저장. meta={s1['meta']}", flush=True)
    print(f"[완료] catboost_extra_cat={final_bundle['catboost_extra_cat']} | cat_feature_cols {len(cat_feature_cols)} / num_cols {len(num_cols)}", flush=True)
    print(f"[완료] bin_edges 없음(raw concat): {final_bundle['mlp_bundle'].get('bin_edges') is None}", flush=True)


SCRIPT_PATCH_ANCHOR = "    catboost_models = bundle.get(\"catboost_models\") or [bundle[\"catboost_model\"]]"
SCRIPT_PATCH_BLOCK = (
    "    # cat_team 번들: 학습때 team_id 를 categorical(str) 로 넣었으므로 추론도 동일 캐스팅\n"
    "    _extra_cat = bundle.get(\"catboost_extra_cat\", [])\n"
    "    if _extra_cat:\n"
    "        X_test_cb = X_test_cb.copy()\n"
    "        for _c in _extra_cat:\n"
    "            if _c in X_test_cb.columns:\n"
    "                X_test_cb[_c] = X_test_cb[_c].astype(str)\n"
    "    catboost_models = bundle.get(\"catboost_models\") or [bundle[\"catboost_model\"]]"
)


def promote():
    ts = time.strftime("%Y%m%d_%H%M%S")
    bdir = f"./open/former_model/submit_pre_catteam_{ts}"
    os.makedirs(bdir, exist_ok=True)
    for f in ("final_retained_model.pkl", "season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        src = os.path.join("./submit/model", f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(bdir, f))
    shutil.copy2("./submit/script.py", os.path.join(bdir, "script.py"))
    print(f"[promote] 기존 submit/ 백업 -> {bdir}", flush=True)

    for f in ("final_retained_model.pkl", "season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        src = os.path.join(STAGING_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join("./submit/model", f))
    # script.py 패치 (idempotent)
    with open("./submit/script.py") as fh:
        sp = fh.read()
    if "catboost_extra_cat" not in sp:
        sp = sp.replace(SCRIPT_PATCH_ANCHOR, SCRIPT_PATCH_BLOCK, 1)
        with open("./submit/script.py", "w") as fh:
            fh.write(sp)
        print("[promote] submit/script.py 패치 완료 (team_id str 캐스팅)", flush=True)
    else:
        print("[promote] submit/script.py 이미 패치됨", flush=True)
    print("[promote] STAGING -> submit/model/ 복사 완료.", flush=True)


def stage2_fast(device):
    """--fast: cat_team 은 CatBoost 만 바꾸므로 현행 프로덕션 번들의 MLP(이미 137만
    전체학습된 7-seed raw-concat)·메타·lookup 3종을 그대로 재사용하고 CatBoost 5-seed
    만 team-categorical 로 전체 재학습한다. stage1(cutoff7) 완전 생략."""
    print("\n" + "=" * 72 + f"\n[FAST] MLP/메타/lookup 재사용 + CatBoost 5-seed(team cat)만 재학습 -> {STAGING_PATH}\n" + "=" * 72, flush=True)
    os.makedirs(STAGING_DIR, exist_ok=True)
    with open("./submit/model/final_retained_model.pkl", "rb") as f:
        prod = pickle.load(f)
    meta = prod["meta_model"]
    mlp_bundle = prod["mlp_bundle"]
    prod_cat_iters = prod.get("catboost_best_iterations") or [prod["catboost_best_iteration"]] * len(YUDAM_CATBOOST_SEEDS)
    print(f"[FAST] 프로덕션 재사용: meta={meta} | mlp num_cols={len(mlp_bundle['num_cols'])} | cb_iters={prod_cat_iters}", flush=True)
    for f in ("season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        shutil.copy2(os.path.join("./submit/model", f), os.path.join(STAGING_DIR, f))
    print("[FAST] lookup 3종 프로덕션에서 복사(내용 동일)", flush=True)

    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    tr_final, _mc, _tc = process_trackman_features_safe(train_df_raw, df_trm, is_train_split=False)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    del tr_final, train_df_raw
    gc.collect()
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=None)
    league_success_mean = train_df[TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)
    train_df = apply_same_hand(train_df)
    train_df = apply_f1_filter(train_df)
    te_prior = train_df[TARGET].mean()
    te_source_cols = ["pitcher_id", "batter_id", "balls_before", "strikes_before",
                      "batter_hand", "num_runners_on", "inning", "season", TARGET]
    train_df = apply_te_residual_features(train_df[te_source_cols].copy(), train_df, te_prior)
    del df_trm
    gc.collect()

    all_cols = [c for c in train_df.columns if c not in (ID, TARGET)]
    cat_feature_cols = [c for c in all_cols if c not in SAME_HAND_COLS and not is_trackman64(c)]
    cb_cat = [c for c in CB_CAT_FEATURES if c in cat_feature_cols]
    cb_full_iters = [it + CATBOOST_ITER_BUFFER for it in prod_cat_iters]
    print(f"[FAST] {len(train_df)}행 | CatBoost {len(cat_feature_cols)}피처 | cat_features={cb_cat} | iters={cb_full_iters}", flush=True)

    cb_full = train_catboost_ensemble(
        _cast_team_str(train_df[cat_feature_cols], cat_feature_cols), train_df[TARGET].values,
        seeds=YUDAM_CATBOOST_SEEDS, per_seed_iterations=cb_full_iters, verbose=True,
        params=_v2_params(), cat_features=cb_cat,
    )
    final_catboost_models = [m for m, _ in cb_full]

    final_bundle = make_blend_bundle(final_catboost_models, mlp_bundle,
                                     {"w_cat": meta["w_cat"], "w_mlp": meta["w_mlp"], "intercept": meta["intercept"]},
                                     cat_feature_cols=cat_feature_cols)
    final_bundle["catboost_best_iteration"] = cb_full_iters[0]
    final_bundle["catboost_best_iterations"] = cb_full_iters
    final_bundle["catboost_extra_cat"] = list(EXTRA_CAT)
    with open(STAGING_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"\n[FAST 완료] {STAGING_PATH} | meta 재사용={meta} | catboost_extra_cat={final_bundle['catboost_extra_cat']}", flush=True)
    print(f"[FAST 완료] MLP 재사용(num_cols {len(final_bundle['mlp_bundle']['num_cols'])}, bin_edges None={final_bundle['mlp_bundle'].get('bin_edges') is None})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="stage1 생략 + 프로덕션 MLP/메타/lookup 재사용, CatBoost만 재학습")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--promote-only", action="store_true")
    args = ap.parse_args()
    if args.promote_only:
        promote()
        return
    torch.manual_seed(0)
    device = get_device()
    print(f"device={device}", flush=True)
    if args.fast:
        stage2_fast(device)
    else:
        s1 = stage1_cutoff7(device)
        gc.collect()
        stage2_full_retrain(s1, device)
    if args.promote:
        promote()


if __name__ == "__main__":
    main()
