# code/build_catteam_nopidmlp_bundle.py
"""cat_team + (pitcher_id/batter_id 를 MLP num_cols 에서 제거) 초스피드 빌드.

프로덕션(1126.77 = cat_team) 대비 딱 하나만 다르다: MLP 의 raw-numeric 입력에서
pitcher_id / batter_id 2개를 뺀다 (익명 ID 를 표준화 스칼라로 먹이던 것 제거).
CatBoost 쪽은 전혀 안 건드림 (pid 는 CatBoost numeric 으로 유지, team_id categorical 유지).

초스피드: 프로덕션 번들의 CatBoost 5-seed(전체학습 완료) + 메타가중치 + lookup 3종을
그대로 재사용하고, MLP 7-seed 만 pid 뺀 채 1.37M 전체 재학습한다 (per-seed epoch =
프로덕션 best_epoch + 5). stage1(cutoff7) 및 메타 refit 은 시간상 생략 —> 메타는
team-numeric-MLP 기준으로 fit 된 값이라 근사치.

실행:  python -m code.build_catteam_nopidmlp_bundle
       python -m code.build_catteam_nopidmlp_bundle --promote-only
"""
import argparse
import gc
import os
import pickle
import shutil
import time

import pandas as pd
import torch

from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing,
    to_tensors, train_mlp, make_bundle, get_device,
)
from code.blend_model import make_blend_bundle
from code.train import (
    process_trackman_features_safe, apply_f1_filter, add_engineered_features,
    apply_te_residual_features, apply_same_hand, SAME_HAND_COLS, is_trackman64,
    TE_RESIDUAL_COLS, YUDAM_ENSEMBLE_SEEDS,
)
from code.trackman_pitcher_features import merge_coarse_pitchmix

TARGET = "control_success"
ID = "row_id"
DATA_DIR = "./open/data"
EPOCH_BUFFER = 5
DROP_FROM_MLP = ["pitcher_id", "batter_id"]
PROD_PKL = "./submit/model/final_retained_model.pkl"
STAGING_DIR = "./scratchpad/submit_catteam_nopidmlp"
STAGING_PATH = os.path.join(STAGING_DIR, "final_retained_model.pkl")


def build(device):
    print("\n" + "=" * 72, flush=True)
    print(f"[초스피드] cat_team CatBoost/메타/lookup 재사용 + MLP 7-seed(pid 제거)만 재학습 -> {STAGING_PATH}", flush=True)
    print("=" * 72, flush=True)
    os.makedirs(STAGING_DIR, exist_ok=True)

    with open(PROD_PKL, "rb") as f:
        prod = pickle.load(f)
    meta = prod["meta_model"]
    prod_mlp = prod["mlp_bundle"]
    prod_seeds = [m["seed"] for m in prod_mlp["members"]]
    prod_epochs = [m["best_epoch"] for m in prod_mlp["members"]]
    assert prod_seeds == list(YUDAM_ENSEMBLE_SEEDS), f"seed 불일치 {prod_seeds} vs {list(YUDAM_ENSEMBLE_SEEDS)}"
    prod_num_cols = prod_mlp["num_cols"]
    assert all(c in prod_num_cols for c in DROP_FROM_MLP), "프로덕션 MLP num_cols 에 pid 가 없음?!"
    print(f"[초스피드] 프로덕션 재사용: meta={meta}", flush=True)
    print(f"[초스피드] CatBoost {len(prod['catboost_models'])}-seed / cb_iters={prod.get('catboost_best_iterations')} / extra_cat={prod.get('catboost_extra_cat')}", flush=True)
    print(f"[초스피드] MLP seeds={prod_seeds} epochs={prod_epochs} -> full epochs +{EPOCH_BUFFER}", flush=True)

    for f in ("season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        shutil.copy2(os.path.join("./submit/model", f), os.path.join(STAGING_DIR, f))
    print("[초스피드] lookup 3종 프로덕션에서 복사(내용 동일)", flush=True)

    # ---- 피처 파이프라인 (build_catteam_bundle.stage2_fast 와 동일) ----
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
    # num_cols: 표준 정의에서 pid 2개를 추가로 제거
    num_cols = [c for c in all_cols
                if c not in CAT_COLS and c not in TE_RESIDUAL_COLS
                and not is_trackman64(c) and c not in DROP_FROM_MLP]
    cat_feature_cols = prod["cat_feature_cols"]  # CatBoost 쪽 그대로
    assert not any(c in num_cols for c in DROP_FROM_MLP)
    assert len(num_cols) == len(prod_num_cols) - len(DROP_FROM_MLP), \
        f"num_cols {len(num_cols)} != {len(prod_num_cols)}-{len(DROP_FROM_MLP)}"
    print(f"[초스피드] {len(train_df)}행 | MLP num_cols {len(num_cols)} (프로덕션 {len(prod_num_cols)} - pid {len(DROP_FROM_MLP)}) | CatBoost {len(cat_feature_cols)}(그대로)", flush=True)

    full_proc, ce, ni, nsc, cat_dims = fit_preprocessing(train_df, CAT_COLS, num_cols)
    Xfc, Xfn, yf = to_tensors(full_proc, CAT_COLS, num_cols, TARGET)
    del full_proc
    gc.collect()
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    full_members = []
    for seed, base_epoch in zip(YUDAM_ENSEMBLE_SEEDS, prod_epochs):
        full_epochs = max(base_epoch or 1, 1) + EPOCH_BUFFER
        t0 = time.time()
        model, _ = train_mlp(
            Xfc, Xfn, yf, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
            embed_dims=embed_dims, bin_edges=None, max_epochs=full_epochs, device=device, seed=seed,
        )
        full_members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": full_epochs, "seed": seed,
        })
        print(f"[초스피드] MLP seed={seed} {full_epochs} epoch 완료 ({time.time() - t0:.0f}s)", flush=True)
        del model
        gc.collect()

    final_mlp_bundle = make_bundle(full_members, CAT_COLS, num_cols, cat_dims, embed_dims, ce, ni, nsc, bin_edges=None)

    final_bundle = make_blend_bundle(prod["catboost_models"], final_mlp_bundle,
                                    {"w_cat": meta["w_cat"], "w_mlp": meta["w_mlp"], "intercept": meta["intercept"]},
                                    cat_feature_cols=cat_feature_cols)
    final_bundle["catboost_best_iteration"] = prod["catboost_best_iteration"]
    final_bundle["catboost_best_iterations"] = prod.get("catboost_best_iterations")
    final_bundle["catboost_extra_cat"] = list(prod.get("catboost_extra_cat", []))

    with open(STAGING_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"\n[완료] {STAGING_PATH}", flush=True)
    print(f"[완료] MLP num_cols {len(num_cols)} / CatBoost {len(cat_feature_cols)} / extra_cat={final_bundle['catboost_extra_cat']} / meta 재사용={meta}", flush=True)


def promote():
    ts = time.strftime("%Y%m%d_%H%M%S")
    bdir = f"./open/former_model/submit_pre_nopidmlp_{ts}"
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
    # script.py 는 이미 catboost_extra_cat 패치가 되어 있어야 함(cat_team 프로덕션 상태)
    with open("./submit/script.py") as fh:
        sp = fh.read()
    assert "catboost_extra_cat" in sp, "submit/script.py 에 catboost_extra_cat 패치가 없음! cat_team 프로덕션 상태가 아님"
    print("[promote] submit/script.py 는 이미 catboost_extra_cat 패치됨 (변경 불필요)", flush=True)
    print("[promote] STAGING -> submit/model/ 복사 완료.", flush=True)


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
    build(device)
    if args.promote:
        promote()


if __name__ == "__main__":
    main()
