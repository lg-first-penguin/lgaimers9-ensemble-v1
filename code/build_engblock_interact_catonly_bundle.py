# code/build_engblock_interact_catonly_bundle.py
"""G_interact(3컬럼)를 CatBoost 피처목록에서만 제거한 제출 번들 빌드 (2026-08-31).

현행 프로덕션(candidate B raw − 트랙맨64, 실전 1117.03)에서 **엔지니어링 interaction
3컬럼 `pitcher_count_advantage_raw` / `pitcher_count_advantage_rel` / `count_pressure` 를
CatBoost 의 cat_feature_cols 에서만 빼고 MLP num_cols 에는 그대로 둔다**(= "catonly" /
MLP-only feed). 나머지(raw-concat MLP 7-seed, CatBoost v2 HP 5-seed, coarse pitchmix,
season-progression + reverse_rate 진행분, Track A TE-residual→CatBoost, same_hand→MLP,
F1 필터, 트랙맨64 제거, 2-input 로지스틱 메타)는 전부 1117 레시피와 동일.

로컬 근거는 약하다 (`code/experiment_yudam_engblock_pergroup_rolling.py`): catonly 4레짐
blend Δ = cutoff7 +9.01 / 2023 +3.79 / 2022 −10.74 / 2021 +2.81, mean +1.2 (노이즈밴드 안).
결정적 신호인 CatBoost-solo Δ 는 +10.0 / −7.9 / −7.7 / +12.7 로 fold마다 부호반전.
→ 실전 이득 확률 낮음. 사용자 판단으로 "실전 데이터포인트 1개" 목적 빌드.
submit/ 는 사용자가 직접 교체·제출. 기본은 스테이징 경로에만 쓴다.

2단계 (build_trackman64_removed_bundle.py 와 동일 구조, 트랙맨64 제거 + interact→cat 제외 추가):
  1) cutoff7 split (트랙맨64 drop + interact 를 cat 에서만 제외) -> raw MLP 7-seed +
     CatBoost 5-seed -> val 메타가중치 / 시드별 best_epoch / cb best_iteration 확정.
     (pergroup 실험의 catonly cutoff7 blend ≈ BASE 730.87 + 9.01 = ~740 근방이면 정상)
  2) 전체 1.37M 재학습(홀드아웃 없음) -> STAGING_PATH 저장. 정적 lookup 3종도 재생성
     (내용은 candidate B / 1117 과 동일 — 피처목록만 다름).

실행:  python -m code.build_engblock_interact_catonly_bundle
       python -m code.build_engblock_interact_catonly_bundle --promote   # submit/ 반영(백업 후)
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
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, train_mlp, make_bundle, predict_bundle,
    get_device, compute_bss,
)
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.blend_model import fit_meta_model, make_blend_bundle
from code.experiment_yudam_common import build_split
from code.experiment_yudam_hybrid_mlp import TRACKMAN64_RE
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

# G_interact — code/experiment_yudam_engblock_prune.py::GROUPS["G_interact"]
INTERACT3 = ["pitcher_count_advantage_raw", "pitcher_count_advantage_rel", "count_pressure"]

SP = "/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/09ea49a2-3198-4db3-b121-b70299cef422/scratchpad"
STAGING_DIR = os.path.join(SP, "submit_interact_catonly")
STAGING_PATH = os.path.join(STAGING_DIR, "final_retained_model.pkl")


def _tm64(cols):
    return [c for c in cols if TRACKMAN64_RE.match(c)]


def _exclude_interact_from_cat(df, holdout):
    """build_split add_features_fn 훅: interact 3컬럼을 CatBoost 에서만 제외, MLP 는 유지."""
    return df, set(INTERACT3), set()


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
    print("\n" + "=" * 72 + "\n[1단계] cutoff7 (트랙맨64 drop + interact→cat 제외) -> 메타/epoch/iter\n" + "=" * 72, flush=True)
    ts, vs, num_cols, cat_feature_cols, all_cols = build_split(
        regime="cutoff7", add_features_fn=_exclude_interact_from_cat, verbose=True)
    assert not _tm64(num_cols) and not _tm64(cat_feature_cols), "트랙맨64 가 아직 피처목록에 있음"
    assert not [c for c in cat_feature_cols if c in INTERACT3], "interact 가 아직 cat_feature_cols 에 있음"
    assert all(c in num_cols for c in INTERACT3), "interact 가 MLP num_cols 에서 빠졌음(있어야 함)"
    print(f"[1단계] interact 3컬럼: CatBoost 제외 / MLP 유지 확인 | "
          f"MLP num {len(num_cols)} / CatBoost {len(cat_feature_cols)}", flush=True)

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
    mlp_bundle_s1 = make_bundle(members, CAT_COLS, num_cols, cat_dims, embed_dims,
                               ce, ni, nsc, bin_edges=None)
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
    print(f"\n[1단계 결과] cutoff7  CatBoost solo={cat_solo:.2f} | raw-MLP solo={mlp_solo:.2f} | "
          f"blend={blend_score * 100000:.2f}  (BASE 730.87 + catonly Δ ~+9 = ~740 근방이면 정상)", flush=True)
    print(f"[1단계 결과] meta w_cat={w_cat:.4f} w_mlp={w_mlp:.4f} intercept={intercept:.4f}", flush=True)
    print(f"[1단계 결과] per_seed_epochs={per_seed_epochs} | cb_best_iters={cb_best_iters}", flush=True)

    del ts, vs, members, mlp_bundle_s1, Xtc, Xtn, Xvc, Xvn, cb_res
    gc.collect()
    return dict(meta={"w_cat": w_cat, "w_mlp": w_mlp, "intercept": intercept},
                per_seed_epochs=per_seed_epochs, cb_best_iters=cb_best_iters)


def stage2_full_retrain(s1, device):
    print("\n" + "=" * 72 + f"\n[2단계] 전체 재학습 -> {STAGING_PATH}\n" + "=" * 72, flush=True)
    os.makedirs(STAGING_DIR, exist_ok=True)
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    tr_final, _mc, trackman_cols = process_trackman_features_safe(train_df_raw, df_trm, is_train_split=False)
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    del tr_final, train_df_raw
    gc.collect()
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=None)
    print(f"[2단계][트랙맨] 물리조인 {len(trackman_cols)}컬럼 산출됨(제거 대상) + coarse pitchmix 4컬럼 유지", flush=True)

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
    # 1117 레시피: trackman64 + same_hand -> cat 제외. 여기 추가: interact 3컬럼 -> cat 만 제외.
    cat_feature_cols = [c for c in all_cols
                        if c not in SAME_HAND_COLS and not TRACKMAN64_RE.match(c)
                        and c not in INTERACT3]
    # MLP num_cols: interact 3컬럼 유지 (trackman64 / TE-residual / cat 만 제외).
    num_cols = [c for c in all_cols
                if c not in CAT_COLS and c not in TE_RESIDUAL_COLS and not TRACKMAN64_RE.match(c)]
    assert not _tm64(cat_feature_cols) and not _tm64(num_cols)
    assert not [c for c in cat_feature_cols if c in INTERACT3]
    assert all(c in num_cols for c in INTERACT3)
    print(f"[2단계] {len(train_df)}행 | CatBoost {len(cat_feature_cols)}피처 (interact 3 제외) / "
          f"MLP {len(num_cols) + len(CAT_COLS)}피처 (interact 3 포함)", flush=True)

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
            embed_dims=embed_dims, bin_edges=None, max_epochs=full_epochs,
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
                                   ce, ni, nsc, bin_edges=None)

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

    with open(STAGING_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"\n[완료] {STAGING_PATH} 저장. meta={s1['meta']}", flush=True)
    print(f"[완료] bin_edges 없음(raw concat) 확인: {final_bundle['mlp_bundle'].get('bin_edges') is None}", flush=True)
    print(f"[완료] cat_feature_cols {len(cat_feature_cols)} (interact 0개) / num_cols {len(num_cols)} "
          f"(interact {sum(c in num_cols for c in INTERACT3)}개)", flush=True)


def promote():
    ts = time.strftime("%Y%m%d_%H%M%S")
    bdir = f"./open/former_model/submit_pre_interact_catonly_{ts}"
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
    print("[promote] STAGING -> submit/model/ 복사 완료. script.py 는 무변경(번들 피처목록으로 선택).", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--promote-only", action="store_true")
    args = ap.parse_args()
    if args.promote_only:
        promote()
        return
    torch.manual_seed(0)
    device = get_device()
    print(f"device={device}", flush=True)
    s1 = stage1_cutoff7(device)
    gc.collect()
    stage2_full_retrain(s1, device)
    if args.promote:
        promote()


if __name__ == "__main__":
    main()
