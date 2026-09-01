# code/build_mlpdropseason_bundle.py
"""mlp_drop_season 제출 번들 (2026-09-01, 마감 — 사용자 지시).

현행 프로덕션(cat_team, 실전 1126.77) 대비 딱 하나:
  MLP num_cols 에서 `season` 제거 (68 -> 67). CatBoost 는 그대로 season 포함(79).
가설: season=2025 는 학습(2019-2024)에 없는 값 -> MLP StandardScaler 가 모든 2025 행을
학습범위 밖 z 로 밀어 계통편향. 로컬 스크린 −4.98 / Colab 스크린 +13.29 로 부호 불일치
(few-seed MLP 분산) -> 실전 데이터포인트로 확정.

빌드: build_a10_bundle._build_train_df 로 피처 파이프라인 재현(= stage2_full_retrain).
  MLP 7-seed 재학습(season 제거, per-seed epoch = 프로덕션 best_epoch + 5).
  CatBoost 5-seed / 메타 / lookup 3종은 프로덕션에서 그대로 재사용.

실행:
  python -m code.build_mlpdropseason_bundle
  python -m code.build_mlpdropseason_bundle --promote-only
"""
import argparse
import gc
import os
import pickle
import shutil
import time

import torch

from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, to_tensors,
    train_mlp, make_bundle, get_device,
)
from code.blend_model import make_blend_bundle
from code.train import YUDAM_ENSEMBLE_SEEDS
from code.build_a10_bundle import _build_train_df, MLP_FLAT_EPOCHS  # 피처 파이프라인 재사용

TARGET = "control_success"
PROD_PKL = "./submit/model/final_retained_model.pkl"
EPOCH_BUFFER = 5
STAGING_DIR = "./scratchpad/submit_mlpdropseason"
STAGING_PATH = os.path.join(STAGING_DIR, "final_retained_model.pkl")
MLP_CKPT = os.path.join(STAGING_DIR, "mlp_bundle_ckpt.pkl")


def build(device):
    print("\n" + "=" * 72 + "\n[mlp_drop_season] MLP 7-seed 재학습(season 제거) + CatBoost/메타/lookup 프로덕션 재사용\n" + "=" * 72, flush=True)
    os.makedirs(STAGING_DIR, exist_ok=True)

    with open(PROD_PKL, "rb") as f:
        prod = pickle.load(f)
    meta = prod["meta_model"]
    prod_cat_models = prod["catboost_models"]
    prod_epochs = [m["best_epoch"] for m in prod["mlp_bundle"]["members"]]
    prod_num_cols = prod["mlp_bundle"]["num_cols"]
    assert "season" in prod_num_cols, "프로덕션 MLP num_cols 에 season 이 없음 — 이미 제거됨?"
    print(f"[mlp_drop_season] 재사용: meta={meta} | CatBoost {len(prod_cat_models)}-seed | "
          f"프로덕션 MLP epochs={prod_epochs} -> +{EPOCH_BUFFER}", flush=True)

    for fn in ("season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv"):
        shutil.copy2(os.path.join("./submit/model", fn), os.path.join(STAGING_DIR, fn))
    print("[mlp_drop_season] lookup 3종 프로덕션에서 복사(내용 동일)", flush=True)

    train_df, cat_feature_cols, num_cols_full, _cb_cat = _build_train_df()
    assert cat_feature_cols == prod["cat_feature_cols"], "cat_feature_cols 가 프로덕션과 불일치"
    num_cols = [c for c in num_cols_full if c != "season"]
    assert "season" not in num_cols and len(num_cols) == len(prod_num_cols) - 1, \
        f"num_cols {len(num_cols)} != prod {len(prod_num_cols)} - 1"
    print(f"[mlp_drop_season] {len(train_df)}행 | MLP num_cols {len(num_cols)} (프로덕션 {len(prod_num_cols)} - season) | "
          f"CatBoost {len(cat_feature_cols)} (그대로, season 포함)", flush=True)

    full_proc, ce, ni, nsc, cat_dims = fit_preprocessing(train_df, CAT_COLS, num_cols)
    Xfc, Xfn, yf = to_tensors(full_proc, CAT_COLS, num_cols, TARGET)
    del full_proc, train_df
    gc.collect()
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    members = []
    for seed, base_epoch in zip(YUDAM_ENSEMBLE_SEEDS, prod_epochs):
        ep = max(base_epoch or 1, 1) + EPOCH_BUFFER
        model, _ = train_mlp(
            Xfc, Xfn, yf, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
            embed_dims=embed_dims, bin_edges=None, max_epochs=ep, device=device, seed=seed,
        )
        members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": ep, "seed": seed,
        })
        print(f"[mlp_drop_season] MLP seed={seed} {ep} epoch 완료", flush=True)
        del model
        gc.collect()
    mlp_bundle = make_bundle(members, CAT_COLS, num_cols, cat_dims, embed_dims, ce, ni, nsc, bin_edges=None)
    with open(MLP_CKPT, "wb") as f:
        pickle.dump(mlp_bundle, f)

    final_bundle = make_blend_bundle(list(prod_cat_models), mlp_bundle, meta, cat_feature_cols=cat_feature_cols)
    final_bundle["catboost_best_iteration"] = prod["catboost_best_iteration"]
    final_bundle["catboost_best_iterations"] = prod["catboost_best_iterations"]
    final_bundle["catboost_extra_cat"] = prod["catboost_extra_cat"]
    with open(STAGING_PATH, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"\n[mlp_drop_season 완료] {STAGING_PATH}", flush=True)
    print(f"  MLP {len(members)}-seed / num_cols {len(num_cols)} (season 제거) / "
          f"CatBoost {len(final_bundle['catboost_models'])}-seed (프로덕션 재사용) / "
          f"cat_feature_cols {len(cat_feature_cols)} / bin_edges None={final_bundle['mlp_bundle'].get('bin_edges') is None} / meta={meta}", flush=True)


def promote():
    ts = time.strftime("%Y%m%d_%H%M%S")
    bdir = f"./open/former_model/submit_pre_mlpdropseason_{ts}"
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
        assert "catboost_extra_cat" in fh.read(), "submit/script.py 에 catboost_extra_cat 패치 없음"
    print("[promote] STAGING -> submit/model/ 복사 완료. (script.py = cat_team 패치 그대로)", flush=True)


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
