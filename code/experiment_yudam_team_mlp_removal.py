# code/experiment_yudam_team_mlp_removal.py
"""[mlp_nopid_blend 후속] team_id 2개를 MLP 임베딩(CAT_COLS)에서도 빼는 config 추가.

mlp_nopid_blend.py 가 저장한 scratchpad/nopid_blend_preds/{regime}.npz 를 재사용
(y, p_cat_base, p_cat_nopid, p_cat_noteam, p_mlp_base, p_mlp_nopid). 레짐당 새로
학습하는 건 MLP 2개뿐:
  p_mlp_noteam        : MLP CAT_COLS 에서 team_id 2개 제거 (num_cols 는 그대로)
  p_mlp_nopid_noteam  : MLP 에서 pitcher_id/batter_id(num) + team_id(cat) 전부 제거

새 blend (2-input 로지스틱, val 전체):
  team MLP만       = p_cat_base   × p_mlp_noteam
  team 양쪽        = p_cat_noteam × p_mlp_noteam
  pid+team MLP양쪽 = p_cat_noteam × p_mlp_nopid_noteam

선행: python -m code.experiment_yudam_mlp_nopid_blend (npz 생성)
"""
import gc
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from code.experiment_yudam_common import build_split, TARGET
from code.mlp_model import (
    embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, make_bundle, predict_bundle, get_device, compute_bss,
)
from code.mlp_model import CAT_COLS as CAT_COLS_FULL
from code.train import YUDAM_ENSEMBLE_SEEDS

REGIMES = os.environ.get("REGIMES", "cutoff7,2023,2022,2021").split(",")
REALISTIC = {"cutoff7", "2023"}
MLP_SEEDS = list(YUDAM_ENSEMBLE_SEEDS[:3])
IDS = ["pitcher_id", "batter_id"]
TEAMS = ["pitcher_team_id", "batter_team_id"]
PRED_DIR = "./scratchpad/nopid_blend_preds"
OUT_PATH = "./scratchpad/team_mlp_removal_result.json"


def _blend(p_cat, p_mlp, y):
    clf = LogisticRegression().fit(np.column_stack([p_cat, p_mlp]), y)
    wc, wm = (float(c) for c in clf.coef_[0]); b = float(clf.intercept_[0])
    bp = 1.0 / (1.0 + np.exp(-(wc * p_cat + wm * p_mlp + b)))
    return compute_bss(bp, y)[2]


def _mlp_val(ts, vs, cat_cols, num_cols, all_cols, device):
    tp, ce, ni, ns, cd = fit_preprocessing(ts, cat_cols, num_cols)
    vp = apply_preprocessing(vs, cat_cols, num_cols, ce, ni, ns)
    Xtc, Xtn, ytr = to_tensors(tp, cat_cols, num_cols, TARGET)
    Xvc, Xvn, _ = to_tensors(vp, cat_cols, num_cols, TARGET)
    yv = vp[TARGET].values
    del tp, vp; gc.collect()
    ed = [embed_dim_for_cardinality(d) for d in cd]
    mem = train_ensemble(Xtc, Xtn, ytr, cat_dims=cd, num_numeric_feats=len(num_cols),
                         embed_dims=ed, bin_edges=None, X_val_cat=Xvc, X_val_num=Xvn, y_val=yv,
                         seeds=MLP_SEEDS, device=device, verbose=False)
    bd = make_bundle(mem, cat_cols, num_cols, cd, ed, ce, ni, ns, bin_edges=None)
    return predict_bundle(bd, vs[all_cols], device=device).astype(np.float64)


def main():
    device = get_device()
    cat_noteam = [c for c in CAT_COLS_FULL if c not in TEAMS]
    print(f"[team_mlp_removal] regimes={REGIMES} | MLP {len(MLP_SEEDS)}-seed | "
          f"CAT_COLS full={CAT_COLS_FULL}\n  -> noteam CAT_COLS={cat_noteam}", flush=True)
    out = []
    for rg in REGIMES:
        rg = rg.strip()
        npz_path = os.path.join(PRED_DIR, f"{rg}.npz")
        if not os.path.exists(npz_path):
            print(f"[{rg}] SKIP - {npz_path} 없음 (mlp_nopid_blend 먼저 실행)", flush=True)
            continue
        d = np.load(npz_path)
        y = d["y"]; p_cat_base = d["p_cat_base"]; p_cat_noteam = d["p_cat_noteam"]; p_mlp_base = d["p_mlp_base"]
        s_base = _blend(p_cat_base, p_mlp_base, y)

        print(f"\n{'='*64}\n=== regime={rg} (base blend={s_base:.2f}) ===\n{'='*64}", flush=True)
        ts, vs, num_cols, _catf, all_cols = build_split(rg, verbose=True)
        num_nopid = [c for c in num_cols if c not in IDS]

        p_mlp_noteam = _mlp_val(ts, vs, cat_noteam, num_cols, all_cols, device)
        p_mlp_nopid_noteam = _mlp_val(ts, vs, cat_noteam, num_nopid, all_cols, device)

        s_team_mlponly = _blend(p_cat_base, p_mlp_noteam, y)
        s_team_both = _blend(p_cat_noteam, p_mlp_noteam, y)
        s_pidteam_mlpboth = _blend(p_cat_noteam, p_mlp_nopid_noteam, y)
        mlp_nt_solo = compute_bss(p_mlp_noteam, y)[2]
        mlp_pnt_solo = compute_bss(p_mlp_nopid_noteam, y)[2]
        mlp_b_solo = compute_bss(p_mlp_base, y)[2]

        row = dict(regime=rg, base_blend=s_base,
                   mlp_base_solo=mlp_b_solo, mlp_noteam_solo=mlp_nt_solo, mlp_nopid_noteam_solo=mlp_pnt_solo,
                   blend_team_mlponly=s_team_mlponly, blend_team_both=s_team_both,
                   blend_pidteam_mlpboth=s_pidteam_mlpboth,
                   d_team_mlponly=s_team_mlponly - s_base, d_team_both=s_team_both - s_base,
                   d_pidteam_mlpboth=s_pidteam_mlpboth - s_base)
        out.append(row)
        print(f"[{rg}] MLP solo base={mlp_b_solo:.2f} noteam={mlp_nt_solo:.2f} ({mlp_nt_solo-mlp_b_solo:+.2f}) "
              f"nopid+noteam={mlp_pnt_solo:.2f} ({mlp_pnt_solo-mlp_b_solo:+.2f})", flush=True)
        print(f"[{rg}] blend  team_MLP만 Δ{s_team_mlponly-s_base:+.2f} | team_양쪽 Δ{s_team_both-s_base:+.2f} | "
              f"pid+team_MLP양쪽(+CB team제거) Δ{s_pidteam_mlpboth-s_base:+.2f}", flush=True)
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2, default=float)
        del ts, vs; gc.collect()

    print(f"\n{'='*72}\n=== 종합: blend Δ vs base ===\n{'='*72}", flush=True)
    print(f"{'':>22} | " + " | ".join(f"{r['regime']:>9}" for r in out) + " |    mean | real min", flush=True)
    for key, lab in [("d_team_mlponly", "team MLP만 제거"), ("d_team_both", "team 양쪽 제거"),
                     ("d_pidteam_mlpboth", "pid+team MLP전부+CBteam")]:
        ds = [r[key] for r in out]
        rmin = min(r[key] for r in out if r["regime"] in REALISTIC)
        print(f"{lab:>22} | " + " | ".join(f"{x:+9.2f}" for x in ds) + f" | {np.mean(ds):+7.2f} | {rmin:+.2f}", flush=True)
    print(f"\n[저장] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
