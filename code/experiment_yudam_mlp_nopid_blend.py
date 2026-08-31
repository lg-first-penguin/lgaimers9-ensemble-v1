# code/experiment_yudam_mlp_nopid_blend.py
"""[id_cat_sweep 후속] pitcher_id/batter_id 를 MLP num_cols 에서만 제거했을 때의
blend 점수. 기존 스윕은 base_cat×mlp_nopid 를 안 재고 버렸다.

레짐당 3개만 학습 (MLP 2개 + CatBoost base 1개, 전부 3-seed). 예측 배열을
scratchpad/nopid_blend_preds/*.npz 로 저장(재실행 방지).

blend (2-input 로지스틱, val 전체 fit+채점 = repo 관례):
  base    = base_cat  × base_mlp     (기준)
  MLP만제거 = base_cat  × nopid_mlp    ← 사용자가 원한 값
  양쪽제거  = nopid_cat × nopid_mlp    (참고, id_cat_sweep 와 동일해야 함)  -- nopid_cat 도 학습
"""
import gc
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, make_bundle, predict_bundle, get_device, compute_bss,
)
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS

REGIMES = os.environ.get("REGIMES", "cutoff7,2023,2022,2021").split(",")
REALISTIC = {"cutoff7", "2023"}
MLP_SEEDS = list(YUDAM_ENSEMBLE_SEEDS[:3])
CB_SEEDS = list(YUDAM_CATBOOST_SEEDS[:3])
IDS = ["pitcher_id", "batter_id"]
TEAMS = ["pitcher_team_id", "batter_team_id"]
OUT_PATH = "./scratchpad/mlp_nopid_blend_result.json"
PRED_DIR = "./scratchpad/nopid_blend_preds"


def _blend(p_cat, p_mlp, y):
    clf = LogisticRegression().fit(np.column_stack([p_cat, p_mlp]), y)
    wc, wm = (float(c) for c in clf.coef_[0]); b = float(clf.intercept_[0])
    bp = 1.0 / (1.0 + np.exp(-(wc * p_cat + wm * p_mlp + b)))
    return compute_bss(bp, y)[2], (wc, wm, b)


def _mlp_val(ts, vs, num_cols, all_cols, device):
    tp, ce, ni, ns, cd = fit_preprocessing(ts, CAT_COLS, num_cols)
    vp = apply_preprocessing(vs, CAT_COLS, num_cols, ce, ni, ns)
    Xtc, Xtn, ytr = to_tensors(tp, CAT_COLS, num_cols, TARGET)
    Xvc, Xvn, _ = to_tensors(vp, CAT_COLS, num_cols, TARGET)
    yv = vp[TARGET].values
    del tp, vp; gc.collect()
    ed = [embed_dim_for_cardinality(d) for d in cd]
    mem = train_ensemble(Xtc, Xtn, ytr, cat_dims=cd, num_numeric_feats=len(num_cols),
                         embed_dims=ed, bin_edges=None, X_val_cat=Xvc, X_val_num=Xvn, y_val=yv,
                         seeds=MLP_SEEDS, device=device, verbose=False)
    bd = make_bundle(mem, CAT_COLS, num_cols, cd, ed, ce, ni, ns, bin_edges=None)
    return predict_bundle(bd, vs[all_cols], device=device).astype(np.float64)


def _cb_val(ts, vs, cols, ytr, yv):
    res = train_catboost_ensemble(ts[cols], ytr, vs[cols], yv, seeds=CB_SEEDS,
                                  verbose=False, params=_yudam_catboost_params())
    return predict_catboost_ensemble([m for m, _ in res], vs[cols]).astype(np.float64)


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    device = get_device()
    print(f"[mlp_nopid_blend] regimes={REGIMES} | MLP/CB {len(MLP_SEEDS)}-seed | device={device}", flush=True)
    out = []
    for rg in REGIMES:
        rg = rg.strip()
        print(f"\n{'='*64}\n=== regime={rg} ===\n{'='*64}", flush=True)
        ts, vs, num_cols, catf, all_cols = build_split(rg, verbose=True)
        ytr = ts[TARGET].values; yv = vs[TARGET].values.astype(np.float64)
        num_nopid = [c for c in num_cols if c not in IDS]
        catf_nopid = [c for c in catf if c not in IDS]
        catf_noteam = [c for c in catf if c not in TEAMS]

        p_mlp_base = _mlp_val(ts, vs, num_cols, all_cols, device)
        p_mlp_nopid = _mlp_val(ts, vs, num_nopid, all_cols, device)
        p_cat_base = _cb_val(ts, vs, catf, ytr, yv)
        p_cat_nopid = _cb_val(ts, vs, catf_nopid, ytr, yv)
        p_cat_noteam = _cb_val(ts, vs, catf_noteam, ytr, yv)
        np.savez(os.path.join(PRED_DIR, f"{rg}.npz"),
                 y=yv, p_mlp_base=p_mlp_base, p_mlp_nopid=p_mlp_nopid,
                 p_cat_base=p_cat_base, p_cat_nopid=p_cat_nopid, p_cat_noteam=p_cat_noteam)

        s_base, _ = _blend(p_cat_base, p_mlp_base, yv)
        s_mlponly, w1 = _blend(p_cat_base, p_mlp_nopid, yv)
        s_both, w2 = _blend(p_cat_nopid, p_mlp_nopid, yv)
        s_catonly, w3 = _blend(p_cat_nopid, p_mlp_base, yv)
        s_noteam, w4 = _blend(p_cat_noteam, p_mlp_base, yv)
        s_noteam_mlpnopid, w5 = _blend(p_cat_noteam, p_mlp_nopid, yv)
        mlp_b = compute_bss(p_mlp_base, yv)[2]; mlp_n = compute_bss(p_mlp_nopid, yv)[2]
        cat_b = compute_bss(p_cat_base, yv)[2]; cat_n = compute_bss(p_cat_nopid, yv)[2]
        cat_nt = compute_bss(p_cat_noteam, yv)[2]
        row = dict(regime=rg, n_val=int(len(yv)),
                   mlp_base_solo=mlp_b, mlp_nopid_solo=mlp_n,
                   cat_base_solo=cat_b, cat_nopid_solo=cat_n, cat_noteam_solo=cat_nt,
                   blend_base=s_base, blend_mlponly=s_mlponly,
                   blend_catonly=s_catonly, blend_both=s_both,
                   blend_noteam=s_noteam, blend_noteam_mlpnopid=s_noteam_mlpnopid,
                   d_mlponly=s_mlponly - s_base, d_catonly=s_catonly - s_base, d_both=s_both - s_base,
                   d_noteam=s_noteam - s_base, d_noteam_mlpnopid=s_noteam_mlpnopid - s_base)
        out.append(row)
        print(f"[{rg}] MLP solo base={mlp_b:.2f} nopid={mlp_n:.2f} ({mlp_n-mlp_b:+.2f}) | "
              f"CatBoost solo base={cat_b:.2f} nopid={cat_n:.2f} ({cat_n-cat_b:+.2f}) noteam={cat_nt:.2f} ({cat_nt-cat_b:+.2f})", flush=True)
        print(f"[{rg}] blend base={s_base:.2f} | MLP만제거 Δ{s_mlponly-s_base:+.2f} | "
              f"pid_CatBoost만 Δ{s_catonly-s_base:+.2f} | pid_양쪽 Δ{s_both-s_base:+.2f} | "
              f"no_team Δ{s_noteam-s_base:+.2f} | no_team+MLP_nopid Δ{s_noteam_mlpnopid-s_base:+.2f}", flush=True)
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2, default=float)
        del ts, vs; gc.collect()

    print(f"\n{'='*76}\n=== 종합: blend Δ vs base ===\n{'='*76}", flush=True)
    print(f"{'':>14} | " + " | ".join(f"{r['regime']:>9}" for r in out) + " |    mean | real min", flush=True)
    for key, lab in [("d_mlponly", "pid MLP만"), ("d_catonly", "pid CatBoost만"), ("d_both", "pid 양쪽"),
                     ("d_noteam", "no_team"), ("d_noteam_mlpnopid", "no_team+pidMLP")]:
        ds = [r[key] for r in out]
        rmin = min(r[key] for r in out if r["regime"] in REALISTIC)
        print(f"{lab:>14} | " + " | ".join(f"{d:+9.2f}" for d in ds) + f" | {np.mean(ds):+7.2f} | {rmin:+.2f}", flush=True)
    print(f"\n[저장] {OUT_PATH} / 예측배열 {PRED_DIR}/*.npz", flush=True)


if __name__ == "__main__":
    main()
