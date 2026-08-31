# code/experiment_yudam_catteam_pidmlp.py
"""[Step 0] cat_team × pid-MLP-removal 조합 로컬 확인 (게이트 아님, 가산성 확인용).

저장된 scratchpad/nopid_blend_preds/{regime}.npz 에서 p_mlp_base / p_mlp_nopid /
p_cat_base / y 재사용. 레짐당 CatBoost 1개만 새로 학습:
  p_cat_catteam : cat_features 에 pitcher_team_id / batter_team_id 추가 (str 캐스팅)

blend (2-input 로지스틱, val 전체):
  base            = p_cat_base    × p_mlp_base   (기준)
  cat_team        = p_cat_catteam × p_mlp_base
  pid_MLP만       = p_cat_base    × p_mlp_nopid  (교차확인, mlp_nopid_blend 와 일치해야)
  cat_team+pidMLP = p_cat_catteam × p_mlp_nopid  ← 두 레버 다 cutoff7 양수인 조합

선행: python -m code.experiment_yudam_mlp_nopid_blend (npz 생성)
"""
import gc
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.mlp_model import compute_bss
from code.train import YUDAM_CATBOOST_SEEDS

REGIMES = os.environ.get("REGIMES", "cutoff7,2023,2022,2021").split(",")
REALISTIC = {"cutoff7", "2023"}
CB_SEEDS = list(YUDAM_CATBOOST_SEEDS[:3])
TEAMS = ["pitcher_team_id", "batter_team_id"]
PRED_DIR = "./scratchpad/nopid_blend_preds"
OUT_PATH = "./scratchpad/catteam_pidmlp_result.json"


def _blend(p_cat, p_mlp, y):
    clf = LogisticRegression().fit(np.column_stack([p_cat, p_mlp]), y)
    wc, wm = (float(c) for c in clf.coef_[0]); b = float(clf.intercept_[0])
    bp = 1.0 / (1.0 + np.exp(-(wc * p_cat + wm * p_mlp + b)))
    return compute_bss(bp, y)[2], (wc, wm, b)


def main():
    print(f"[catteam_pidmlp] regimes={REGIMES} | CatBoost {len(CB_SEEDS)}-seed", flush=True)
    out = []
    for rg in REGIMES:
        rg = rg.strip()
        npz = os.path.join(PRED_DIR, f"{rg}.npz")
        if not os.path.exists(npz):
            print(f"[{rg}] SKIP - {npz} 없음", flush=True)
            continue
        d = np.load(npz)
        y = d["y"]; p_mlp_base = d["p_mlp_base"]; p_mlp_nopid = d["p_mlp_nopid"]; p_cat_base = d["p_cat_base"]

        print(f"\n{'='*64}\n=== regime={rg} ===\n{'='*64}", flush=True)
        ts, vs, _num, catf, _all = build_split(rg, verbose=True)
        cat_features = ["game_type", "base_state"] + [c for c in TEAMS if c in catf]
        Xtr = ts[catf].copy(); Xva = vs[catf].copy()
        for c in TEAMS:
            if c in Xtr.columns:
                Xtr[c] = Xtr[c].astype(str); Xva[c] = Xva[c].astype(str)
        res = train_catboost_ensemble(Xtr, ts[TARGET].values, Xva, vs[TARGET].values,
                                      seeds=CB_SEEDS, verbose=False,
                                      params=_yudam_catboost_params(), cat_features=cat_features)
        p_cat_catteam = predict_catboost_ensemble([m for m, _ in res], Xva).astype(np.float64)

        s_base, _ = _blend(p_cat_base, p_mlp_base, y)
        s_catteam, w1 = _blend(p_cat_catteam, p_mlp_base, y)
        s_pidmlp, _ = _blend(p_cat_base, p_mlp_nopid, y)
        s_combo, w2 = _blend(p_cat_catteam, p_mlp_nopid, y)
        cat_b = compute_bss(p_cat_base, y)[2]; cat_ct = compute_bss(p_cat_catteam, y)[2]
        row = dict(regime=rg, base_blend=s_base,
                   cat_base_solo=cat_b, cat_catteam_solo=cat_ct,
                   blend_catteam=s_catteam, blend_pidmlp=s_pidmlp, blend_combo=s_combo,
                   d_catteam=s_catteam - s_base, d_pidmlp=s_pidmlp - s_base, d_combo=s_combo - s_base)
        out.append(row)
        print(f"[{rg}] CatBoost solo base={cat_b:.2f} cat_team={cat_ct:.2f} ({cat_ct-cat_b:+.2f})", flush=True)
        print(f"[{rg}] blend base={s_base:.2f} | cat_team Δ{s_catteam-s_base:+.2f} | "
              f"pid_MLP만 Δ{s_pidmlp-s_base:+.2f} | cat_team+pidMLP Δ{s_combo-s_base:+.2f} "
              f"(가산예측 {(s_catteam-s_base)+(s_pidmlp-s_base):+.2f})", flush=True)
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2, default=float)
        del ts, vs; gc.collect()

    print(f"\n{'='*74}\n=== 종합: blend Δ vs base ===\n{'='*74}", flush=True)
    print(f"{'':>18} | " + " | ".join(f"{r['regime']:>9}" for r in out) + " |    mean | real(cutoff7,2023)", flush=True)
    for key, lab in [("d_catteam", "cat_team"), ("d_pidmlp", "pid MLP만"), ("d_combo", "cat_team+pidMLP")]:
        ds = [r[key] for r in out]
        reals = [r[key] for r in out if r["regime"] in REALISTIC]
        print(f"{lab:>18} | " + " | ".join(f"{x:+9.2f}" for x in ds)
              + f" | {np.mean(ds):+7.2f} | {'/'.join(f'{x:+.2f}' for x in reals)} (min {min(reals):+.2f})", flush=True)
    print(f"\n[저장] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
