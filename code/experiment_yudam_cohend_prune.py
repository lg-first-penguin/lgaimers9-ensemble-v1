# code/experiment_yudam_cohend_prune.py
"""[Pruning 실험 2 - 1단계] Cohen's d + CatBoost 순열중요도 기반 프루닝 후보 선정.

목적: "실무/논문에서 쓰는 제거 기준값"으로 candidate B(유담) 피처셋의 저신호 피처를
골라낸다. 실제 제거·재검증은 2단계(experiment_yudam_cohend_prune_ablate.py)에서.

측정 (build_split(cutoff7) 의 F1-필터된 train_split 위에서):
  1. Cohen's d = (mean[y=1] - mean[y=0]) / s_pooled   (수치형 피처별, nan-aware)
     관례 기준: |d| < 0.10 = negligible(무시 가능), 0.10~0.20 = very small,
                0.20 = small, 0.50 = medium, 0.80 = large  (Cohen 1988 / Sawilowsky 2009)
  2. |point-biserial r| = |d| / sqrt(d^2 + 4)   (d 의 단조변환, 참고용)
  3. CatBoost(유담 v2 HP, 400iter 고정) feature_importance (PredictionValuesChange)
  4. null importance: control_success 를 셔플해 3회 재학습한 importance 의 피처별 최댓값
     -> 실제 importance 가 이 값 이하면 "트리가 우연 이상으로 안 씀"

프루닝 후보 셋 (JSON 으로 저장):
  S_d_negligible : |d| < 0.02
  S_d_small      : |d| < 0.10
  S_lowimp       : CatBoost importance 하위 25%
  S_both         : S_d_small ∩ S_lowimp   (가장 안전 — 단변량 신호도 약하고 트리도 안 씀)
  S_null         : real importance <= null importance 최댓값
카테고리형(game_type/base_state)은 d 를 안 재고 항상 보존.
트랙맨64 는 [Pruning 실험 1] 소관이라 여기선 별도 태깅만.

출력: scratchpad/cohend_prune_sets.json + 랭킹 테이블(로그).

사용법: python -m code.experiment_yudam_cohend_prune
"""
import gc
import json
import os

import numpy as np
import pandas as pd

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.experiment_yudam_hybrid_mlp import TRACKMAN64_RE
from code.catboost_model import train_catboost
from code.mlp_model import CAT_COLS

OUT = "/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/64d8ecb5-0a28-4eef-9550-ce16083f96e8/scratchpad/cohend_prune_sets.json"


def cohens_d(x, y):
    x = np.asarray(x, dtype=np.float64)
    m = ~np.isnan(x)
    x, y = x[m], np.asarray(y)[m]
    x1, x0 = x[y == 1], x[y == 0]
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    n1, n0 = len(x1), len(x0)
    s1, s0 = x1.std(ddof=1), x0.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    if sp == 0:
        return 0.0
    return (x1.mean() - x0.mean()) / sp


def main():
    train_split, val_split, num_cols, cat_feature_cols, all_cols = build_split("cutoff7")
    del val_split
    gc.collect()
    y = train_split[TARGET].values.astype(np.int64)
    print(f"[data] train_split={len(train_split)} | num_cols={len(num_cols)} "
          f"cat_feature_cols={len(cat_feature_cols)}", flush=True)

    # ---- 1. Cohen's d (수치형 전부: MLP num_cols ∪ CatBoost 수치형) ----
    numeric_feats = sorted(set(num_cols) | {c for c in cat_feature_cols if c not in CAT_COLS})
    rows = []
    for c in numeric_feats:
        d = cohens_d(train_split[c].values, y)
        r = abs(d) / np.sqrt(d**2 + 4) if np.isfinite(d) else np.nan
        rows.append(dict(feat=c, cohens_d=d, abs_d=abs(d) if np.isfinite(d) else np.nan,
                         pb_r=r, is_trackman64=bool(TRACKMAN64_RE.match(c))))
    tab = pd.DataFrame(rows)

    # ---- 3. CatBoost importance (실제) ----
    Xc = train_split[cat_feature_cols].copy()
    model, _ = train_catboost(Xc, y, iterations=400, verbose=False, params=_yudam_catboost_params())
    imp = pd.Series(model.get_feature_importance(), index=cat_feature_cols, name="cb_imp")
    del model
    gc.collect()

    # ---- 4. null importance (target 셔플 3회) ----
    rng = np.random.default_rng(0)
    null_imps = []
    for i in range(3):
        y_sh = rng.permutation(y)
        m, _ = train_catboost(Xc, y_sh, iterations=400, verbose=False, params=_yudam_catboost_params())
        null_imps.append(pd.Series(m.get_feature_importance(), index=cat_feature_cols))
        del m
        gc.collect()
        print(f"  null run {i+1}/3 done", flush=True)
    null_max = pd.concat(null_imps, axis=1).max(axis=1)
    del Xc
    gc.collect()

    tab = tab.merge(imp.rename("cb_imp").reset_index().rename(columns={"index": "feat"}), on="feat", how="left")
    tab = tab.merge(null_max.rename("null_imp_max").reset_index().rename(columns={"index": "feat"}), on="feat", how="left")
    tab["beats_null"] = tab["cb_imp"] > tab["null_imp_max"]
    tab = tab.sort_values("abs_d", na_position="last").reset_index(drop=True)

    # ---- 프루닝 후보 셋 ----
    non_tm = tab[~tab["is_trackman64"]].copy()   # 트랙맨64는 실험1 소관, 여기선 제외
    imp_q25 = non_tm["cb_imp"].quantile(0.25)
    S_d_negligible = sorted(non_tm.loc[non_tm["abs_d"] < 0.02, "feat"])
    S_d_small = sorted(non_tm.loc[non_tm["abs_d"] < 0.10, "feat"])
    S_lowimp = sorted(non_tm.loc[non_tm["cb_imp"] <= imp_q25, "feat"])
    S_both = sorted(set(S_d_small) & set(S_lowimp))
    S_null = sorted(non_tm.loc[~non_tm["beats_null"].fillna(False), "feat"])

    sets = dict(
        S_d_negligible=S_d_negligible, S_d_small=S_d_small, S_lowimp=S_lowimp,
        S_both=S_both, S_null=S_null,
        thresholds=dict(d_negligible=0.02, d_small=0.10, imp_q25=float(imp_q25)),
        note="트랙맨64 64개는 실험1 소관이라 이 셋들에서 제외. 카테고리형 game_type/base_state 보존.",
    )
    with open(OUT, "w") as f:
        json.dump(sets, f, ensure_ascii=False, indent=2)

    # ---- 출력 ----
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    print("\n=== 전체 랭킹 (|d| 오름차순, 트랙맨64 제외) ===", flush=True)
    print(non_tm[["feat", "cohens_d", "pb_r", "cb_imp", "null_imp_max", "beats_null"]]
          .round(4).to_string(index=False), flush=True)

    print("\n=== 트랙맨64 64개 요약 (실험1 소관) ===", flush=True)
    tm = tab[tab["is_trackman64"]]
    print(f"  |d| 중앙값 {tm['abs_d'].median():.4f} / 최대 {tm['abs_d'].max():.4f} | "
          f"cb_imp 합 {tm['cb_imp'].sum():.2f} (전체 {tab['cb_imp'].sum():.2f}) | "
          f"beats_null {int(tm['beats_null'].sum())}/64", flush=True)

    print(f"\n=== 프루닝 후보 셋 (-> {OUT}) ===", flush=True)
    for k in ("S_d_negligible", "S_d_small", "S_lowimp", "S_both", "S_null"):
        print(f"  {k:15s} ({len(sets[k]):3d}) : {sets[k]}", flush=True)


if __name__ == "__main__":
    main()
