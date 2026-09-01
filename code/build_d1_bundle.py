# code/build_d1_bundle.py
"""D1 / B / D1+B 제출 번들 빌드 (2026-09-01, 마감 스피드빌드).

전부 현행 프로덕션(cat_team, 실전 1126.77) 위에서 **CatBoost 5-seed 만** 바꾼다.
프로덕션 번들의 MLP(137만 전체학습 7-seed raw-concat) / 메타 / lookup 3종은 그대로 재사용.

variant:
  d1  : asof_pitcher_success_rate / asof_batter_success_rate 를 CatBoost 피처목록에서만
        제거 (MLP num_cols 는 유지). 2025 베테랑의 career-누적 성공률은 pre-ABS 투구가
        지배 -> stale·레짐오염. season-progression *_season_rate 가 라이브 신호로 이미 있음.
  b   : 방향 명확한 rate 피처 14개에 CatBoost monotone_constraints (+1 성공률류 / -1
        middle·reverse류). HP 프리서치가 아니라 용량을 줄이는 구조적 정규화.
  d1b : d1 + b 동시.

--meta-from-json : 스크린 결과 JSON 에서 cutoff7 해당 config 의 메타가중치를 대신 사용
                   (기본은 프로덕션 메타 재사용 — cat_team fast build 와 동일 관례).

실행:
  python -m code.build_d1_bundle --variant d1
  python -m code.build_d1_bundle --variant b   --meta-from-json scratchpad/local_screen_result.json
  python -m code.build_d1_bundle --variant d1b
  python -m code.build_d1_bundle --variant d1  --promote-only
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

from code.mlp_model import get_device
from code.catboost_model import train_catboost_ensemble
from code.blend_model import make_blend_bundle
from code.train import (
    process_trackman_features_safe, apply_f1_filter, add_engineered_features,
    apply_te_residual_features, apply_same_hand, SAME_HAND_COLS, is_trackman64,
    YUDAM_CATBOOST_SEEDS,
)
from code.trackman_pitcher_features import merge_coarse_pitchmix

TARGET = "control_success"
ID = "row_id"
DATA_DIR = "./open/data"
CATBOOST_ITER_BUFFER = 50
EXTRA_CAT = ["pitcher_team_id", "batter_team_id"]
CB_CAT_FEATURES = ["game_type", "base_state"] + EXTRA_CAT
PROD_PKL = "./submit/model/final_retained_model.pkl"

D1_CAT_DROP = ["asof_pitcher_success_rate", "asof_batter_success_rate"]

# B(monotone_constraints) — code/experiment_structural_batch.py 와 동일.
B_MONO_POS = [
    "asof_pitcher_success_rate", "asof_batter_success_rate",
    "pitcher_season_success_rate", "batter_season_success_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]
B_MONO_NEG = [
    "asof_pitcher_middle_rate", "asof_batter_middle_rate", "asof_pitcher_reverse_rate",
    "pitcher_reverse_season_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]

VARIANTS = {"d1", "b", "d1b"}
SCREEN_CFG_KEY = {"d1": "d1_career_catdrop", "b": "b_monotone", "d1b": "d1_career_catdrop"}


def _staging_dir(variant):
    return f"./scratchpad/submit_{variant}"


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


def _meta_from_screen(path, variant):
    with open(path) as f:
        d = json.load(f)
    key = SCREEN_CFG_KEY[variant]
    r = d["regimes"]["cutoff7"][key]
    m = {"w_cat": r["w_cat"], "w_mlp": r["w_mlp"], "intercept": r["intercept"]}
    print(f"[meta] {key} cutoff7 스크린에서 가져옴: {m}", flush=True)
    return m


def _mono_constraints(cat_feature_cols):
    cols = set(cat_feature_cols)
    d = {c: 1 for c in B_MONO_POS if c in cols}
    d.update({c: -1 for c in B_MONO_NEG if c in cols})
    missing = [c for c in (B_MONO_POS + B_MONO_NEG) if c not in cols]
    return d, missing


def build(variant, device, meta_override=None):
    drop_career = variant in ("d1", "d1b")
    do_mono = variant in ("b", "d1b")
    staging_dir = _staging_dir(variant)
    staging_path = os.path.join(staging_dir, "final_retained_model.pkl")
    print("\n" + "=" * 72 +
          f"\n[{variant.upper()} FAST] MLP/메타/lookup 재사용 + CatBoost 5-seed 만 재학습 "
          f"(career drop={drop_career}, monotone={do_mono})\n" + "=" * 72, flush=True)
    os.makedirs(staging_dir, exist_ok=True)

    with open(PROD_PKL, "rb") as f:
        prod = pickle.load(f)
    meta = meta_override or prod["meta_model"]
    mlp_bundle = prod["mlp_bundle"]
    prod_cat_iters = prod.get("catboost_best_iterations") or [prod["catboost_best_iteration"]] * len(YUDAM_CATBOOST_SEEDS)
    print(f"[{variant}] 프로덕션 재사용: meta={meta} | mlp num_cols={len(mlp_bundle['num_cols'])} | "
          f"cb_iters={prod_cat_iters}", flush=True)
    if drop_career:
        assert all(c in mlp_bundle["num_cols"] for c in D1_CAT_DROP), \
            "career 컬럼이 프로덕션 MLP num_cols 에 없음 — 번들 이상"

    for fn in ("season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        shutil.copy2(os.path.join("./submit/model", fn), os.path.join(staging_dir, fn))
    print(f"[{variant}] lookup 3종 프로덕션에서 복사(내용 동일)", flush=True)

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
    if drop_career:
        cat_feature_cols = [c for c in cat_feature_cols if c not in D1_CAT_DROP]
        assert not any(c in cat_feature_cols for c in D1_CAT_DROP)
    assert all(c in cat_feature_cols for c in EXTRA_CAT)
    cb_cat = [c for c in CB_CAT_FEATURES if c in cat_feature_cols]
    cb_full_iters = [it + CATBOOST_ITER_BUFFER for it in prod_cat_iters]

    params = _v2_params()
    if do_mono:
        mono, missing = _mono_constraints(cat_feature_cols)
        assert mono, "monotone 제약이 0개 — 컬럼명 확인 필요"
        params["monotone_constraints"] = mono
        print(f"[{variant}] monotone {len(mono)}개 적용: {mono}", flush=True)
        if missing:
            print(f"[{variant}] ⚠️ 목록에 없어 건너뜀: {missing}", flush=True)

    print(f"[{variant}] {len(train_df)}행 | CatBoost {len(cat_feature_cols)}피처 "
          f"(cat_team 79 {'- 2 = 77' if drop_career else '그대로'}) | cat_features={cb_cat} | iters={cb_full_iters}", flush=True)

    cb_full = train_catboost_ensemble(
        _cast_team_str(train_df[cat_feature_cols], cat_feature_cols), train_df[TARGET].values,
        seeds=YUDAM_CATBOOST_SEEDS, per_seed_iterations=cb_full_iters, verbose=True,
        params=params, cat_features=cb_cat,
    )
    final_catboost_models = [m for m, _ in cb_full]

    final_bundle = make_blend_bundle(
        final_catboost_models, mlp_bundle,
        {"w_cat": meta["w_cat"], "w_mlp": meta["w_mlp"], "intercept": meta["intercept"]},
        cat_feature_cols=cat_feature_cols)
    final_bundle["catboost_best_iteration"] = cb_full_iters[0]
    final_bundle["catboost_best_iterations"] = cb_full_iters
    final_bundle["catboost_extra_cat"] = list(EXTRA_CAT)
    with open(staging_path, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"\n[{variant} 완료] {staging_path} | meta={meta} | catboost_extra_cat={EXTRA_CAT}", flush=True)
    print(f"[{variant} 완료] CatBoost cat_feature_cols {len(cat_feature_cols)} / MLP num_cols "
          f"{len(mlp_bundle['num_cols'])} (재사용) | bin_edges None={final_bundle['mlp_bundle'].get('bin_edges') is None}", flush=True)
    return staging_dir


def promote(variant):
    staging_dir = _staging_dir(variant)
    ts = time.strftime("%Y%m%d_%H%M%S")
    bdir = f"./open/former_model/submit_pre_{variant}_{ts}"
    os.makedirs(bdir, exist_ok=True)
    for fn in ("final_retained_model.pkl", "season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        src = os.path.join("./submit/model", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(bdir, fn))
    shutil.copy2("./submit/script.py", os.path.join(bdir, "script.py"))
    print(f"[promote:{variant}] 기존 submit/ 백업 -> {bdir}", flush=True)
    for fn in ("final_retained_model.pkl", "season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        src = os.path.join(staging_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join("./submit/model", fn))
    with open("./submit/script.py") as fh:
        assert "catboost_extra_cat" in fh.read(), \
            "submit/script.py 에 catboost_extra_cat 패치가 없음 — cat_team 빌드 먼저 필요"
    print(f"[promote:{variant}] STAGING -> submit/model/ 복사 완료. (script.py 는 cat_team 패치 그대로 사용)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--meta-from-json", default=None,
                    help="스크린 결과 JSON 에서 cutoff7 해당 config 메타가중치 사용")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--promote-only", action="store_true")
    args = ap.parse_args()
    if args.promote_only:
        promote(args.variant)
        return
    torch.manual_seed(0)
    device = get_device()
    print(f"device={device} | variant={args.variant}", flush=True)
    meta_override = _meta_from_screen(args.meta_from_json, args.variant) if args.meta_from_json else None
    build(args.variant, device, meta_override=meta_override)
    if args.promote:
        promote(args.variant)


if __name__ == "__main__":
    main()
