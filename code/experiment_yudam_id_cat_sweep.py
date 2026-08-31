# code/experiment_yudam_id_cat_sweep.py
"""[D 후속] pitcher_id/batter_id/team_id 를 CatBoost 에서 어떻게 다룰지 스윕.

현행 1117 프로덕션(트랙맨64 제거된 yudam 레시피)에서:
  - pitcher_id/batter_id/pitcher_team_id/batter_team_id 는 전부 int64 → CatBoost 에
    raw 숫자로 들어간다 (categorical 선언은 game_type/base_state 뿐, yudam 와 동일).
  - pitcher_id/batter_id 는 MLP num_cols 에도 표준화된 정수로 들어간다 (사실상 죽은 입력).
  - pitcher_team_id/batter_team_id 는 MLP CAT_COLS 에 임베딩된다.

과거 검증(PROJECT_HISTORY §3.1): pitcher_id/batter_id 를 CatBoost **categorical 선언**
→ solo -171 / blend -63 (큰 손해). "제거"(no_both) 는 손수연 레시피에서 다른 구성으로
격리돼 "노이즈" 판정. team_id 완전 제거·team_id categorical 은 미검증.

이 스크립트가 재는 것 (regime 별, CatBoost 3-seed / MLP 3-seed):
  CatBoost 쪽 (MLP 는 base 재사용):
    base        : 현행 (cat_features = game_type, base_state)
    no_pid      : CatBoost X 에서 pitcher_id, batter_id 제거
    no_team     : CatBoost X 에서 pitcher_team_id, batter_team_id 제거
    cat_team    : cat_features += pitcher_team_id, batter_team_id (str 캐스팅)
    cat_hand    : cat_features += pitcher_hand, batter_hand (sanity, ≈0 예상)
    cat_teamhand: cat_features += team_id 2개 + hand 2개 (손수연 wide-ish, pid/bid 제외)
  MLP 쪽:
    mlp_no_pid  : MLP num_cols 에서 pitcher_id, batter_id 제거 → blend(no_pid_cat×no_pid_mlp)

판정: cutoff7·2023(실전近) 안 지고 4-regime 평균 blend Δ>0 이어야 후보. 아니면 기각.
regime 별 build_split 1회, Pool/텐서 재사용.

사용법: python -m code.experiment_yudam_id_cat_sweep
        REGIMES=cutoff7,2023 python -m code.experiment_yudam_id_cat_sweep
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
OUT_PATH = "./scratchpad/id_cat_sweep_result.json"

IDS = ["pitcher_id", "batter_id"]
TEAMS = ["pitcher_team_id", "batter_team_id"]
HANDS = ["pitcher_hand", "batter_hand"]


def _blend_bss(p_cat, p_mlp, y):
    clf = LogisticRegression().fit(np.column_stack([p_cat, p_mlp]), y)
    wc, wm = (float(c) for c in clf.coef_[0])
    b = float(clf.intercept_[0])
    bp = 1.0 / (1.0 + np.exp(-(wc * p_cat + wm * p_mlp + b)))
    return compute_bss(bp, y)[2], (wc, wm, b)


def _train_mlp_get_val(train_split, val_split, num_cols, all_cols, device):
    train_proc, ce, ni, ns, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, ce, ni, ns)
    Xtc, Xtn, ytr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    Xvc, Xvn, _ = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)
    y_val = val_proc[TARGET].values
    del train_proc, val_proc
    gc.collect()
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        Xtc, Xtn, ytr, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
        embed_dims=embed_dims, bin_edges=None,
        X_val_cat=Xvc, X_val_num=Xvn, y_val=y_val,
        seeds=MLP_SEEDS, device=device, verbose=False,
    )
    bundle = make_bundle(members, CAT_COLS, num_cols, cat_dims, embed_dims, ce, ni, ns, bin_edges=None)
    p = predict_bundle(bundle, val_split[all_cols], device=device)
    return p.astype(np.float64), y_val.astype(np.float64)


def _train_cb_get_val(train_split, val_split, cat_feature_cols, y_tr, y_val, cat_features):
    Xtr = train_split[cat_feature_cols].copy()
    Xva = val_split[cat_feature_cols].copy()
    for c in cat_features:
        if c not in ("game_type", "base_state") and c in Xtr.columns:
            Xtr[c] = Xtr[c].astype(str)
            Xva[c] = Xva[c].astype(str)
    res = train_catboost_ensemble(
        Xtr, y_tr, Xva, y_val, seeds=CB_SEEDS, verbose=False,
        params=_yudam_catboost_params(), cat_features=list(cat_features),
    )
    models = [m for m, _ in res]
    preds = predict_catboost_ensemble(models, Xva).astype(np.float64)
    return preds


def run_regime(regime, device):
    print(f"\n{'='*70}\n=== regime={regime} ===\n{'='*70}", flush=True)
    ts, vs, num_cols, cat_feature_cols, all_cols = build_split(regime, verbose=True)
    y_tr = ts[TARGET].values
    y_val = vs[TARGET].values.astype(np.float64)

    # --- MLP: base + no_pid ---
    p_mlp_base, _ = _train_mlp_get_val(ts, vs, num_cols, all_cols, device)
    num_cols_nopid = [c for c in num_cols if c not in IDS]
    p_mlp_nopid, _ = _train_mlp_get_val(ts, vs, num_cols_nopid, all_cols, device)
    mlp_base_solo = compute_bss(p_mlp_base, y_val)[2]
    mlp_nopid_solo = compute_bss(p_mlp_nopid, y_val)[2]
    print(f"[{regime}] MLP solo base={mlp_base_solo:.2f} | no_pid={mlp_nopid_solo:.2f} "
          f"(Δ{mlp_nopid_solo - mlp_base_solo:+.2f})", flush=True)

    # --- CatBoost configs ---
    cb_configs = {
        "base":        (cat_feature_cols, ["game_type", "base_state"]),
        "no_pid":      ([c for c in cat_feature_cols if c not in IDS], ["game_type", "base_state"]),
        "no_team":     ([c for c in cat_feature_cols if c not in TEAMS], ["game_type", "base_state"]),
        "cat_team":    (cat_feature_cols, ["game_type", "base_state"] + TEAMS),
        "cat_hand":    (cat_feature_cols, ["game_type", "base_state"] + HANDS),
        "cat_teamhand": (cat_feature_cols, ["game_type", "base_state"] + TEAMS + HANDS),
    }
    rows = {}
    p_cat = {}
    for name, (cols, catf) in cb_configs.items():
        catf = [c for c in catf if c in cols]
        p = _train_cb_get_val(ts, vs, cols, y_tr, y_val, catf)
        p_cat[name] = p
        solo = compute_bss(p, y_val)[2]
        blend, w = _blend_bss(p, p_mlp_base, y_val)
        rows[name] = dict(cb_solo=solo, blend=blend, w=w)
        print(f"[{regime}] CB {name:12s} solo={solo:8.2f} | blend(x mlp_base)={blend:8.2f} "
              f"w=({w[0]:+.2f},{w[1]:+.2f},b={w[2]:+.2f})", flush=True)

    # remove-from-both: no_pid CatBoost x no_pid MLP
    blend_nopid_both, w_nb = _blend_bss(p_cat["no_pid"], p_mlp_nopid, y_val)
    print(f"[{regime}] CB no_pid x MLP no_pid  blend={blend_nopid_both:8.2f} "
          f"w=({w_nb[0]:+.2f},{w_nb[1]:+.2f},b={w_nb[2]:+.2f})", flush=True)

    base_blend = rows["base"]["blend"]
    base_solo = rows["base"]["cb_solo"]
    summary = {"regime": regime, "n_val": int(len(y_val)),
               "mlp_base_solo": mlp_base_solo, "mlp_nopid_solo": mlp_nopid_solo,
               "base_cb_solo": base_solo, "base_blend": base_blend}
    for name in cb_configs:
        summary[f"{name}_d_solo"] = rows[name]["cb_solo"] - base_solo
        summary[f"{name}_d_blend"] = rows[name]["blend"] - base_blend
    summary["nopid_both_d_blend"] = blend_nopid_both - base_blend

    del ts, vs, p_cat
    gc.collect()
    return summary


def main():
    device = get_device()
    print(f"[id_cat_sweep] regimes={REGIMES} | MLP {len(MLP_SEEDS)}-seed / CB {len(CB_SEEDS)}-seed | device={device}", flush=True)
    all_summ = []
    for rg in REGIMES:
        all_summ.append(run_regime(rg.strip(), device))
        with open(OUT_PATH, "w") as f:
            json.dump(all_summ, f, indent=2, default=float)
        gc.collect()

    print(f"\n{'='*78}\n=== 종합 (blend Δ vs base, per regime) ===\n{'='*78}", flush=True)
    configs = ["no_pid", "no_team", "cat_team", "cat_hand", "cat_teamhand"]
    hdr = f"{'config':>13} | " + " | ".join(f"{s['regime']:>9}" for s in all_summ) + " |    mean | realistic"
    print(hdr, flush=True)
    for name in configs:
        ds = [s[f"{name}_d_blend"] for s in all_summ]
        rmin = min((s[f"{name}_d_blend"] for s in all_summ if s["regime"] in REALISTIC), default=float("nan"))
        v = "후보" if (np.mean(ds) > 0 and rmin > -2) else "기각"
        print(f"{name:>13} | " + " | ".join(f"{d:+9.2f}" for d in ds) + f" | {np.mean(ds):+7.2f} | {v} (real min {rmin:+.2f})", flush=True)
    ds = [s["nopid_both_d_blend"] for s in all_summ]
    print(f"{'no_pid_BOTH':>13} | " + " | ".join(f"{d:+9.2f}" for d in ds) + f" | {np.mean(ds):+7.2f} |", flush=True)
    print(f"\n{'config':>13} | CatBoost solo Δ per regime", flush=True)
    for name in configs:
        ds = [s[f"{name}_d_solo"] for s in all_summ]
        print(f"{name:>13} | " + " | ".join(f"{s['regime']}={d:+.2f}" for s, d in zip(all_summ, ds)), flush=True)
    md = [s["mlp_nopid_solo"] - s["mlp_base_solo"] for s in all_summ]
    print(f"\n{'mlp no_pid':>13} solo Δ | " + " | ".join(f"{s['regime']}={d:+.2f}" for s, d in zip(all_summ, md)), flush=True)
    print(f"\n[저장] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
