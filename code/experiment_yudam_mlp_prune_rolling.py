# code/experiment_yudam_mlp_prune_rolling.py
"""[Pruning 실험 v2] MLP 전용 '정확한 대수적 중복' 피처 제거 + logit-space 메타 비교.

지난 FE/Pruning 라인(memory: fe_pruning_line_closed_2026_08_29) 실패 진단:
후보를 단변량 저신호(Cohen's d / point-biserial r / marginal importance / null-importance)
로 골라서, 그 피처들이 실은 '상호작용'·'저데이터 fold 에서 MLP-solo 지탱' 역할을
하고 있던 걸 놓쳤다 (blk3 에서 runner_on_1b 빼니 2021/2022 fold MLP solo -68/-135 붕괴,
교훈 #9). 이번엔 후보를 '정보가 다른 피처에 이미 정확히 들어있는' 피처(정확한 대수적
중복)만으로 좁힌다 — MLP 1층이 선형이라 정확한 선형결합 피처는 표현력 손실 0 으로
제거 가능(저데이터 fold 붕괴 메커니즘 자체가 없음). 이 repo 에서 유일하게 성공한
pruning(트랙맨64)이 '저신호'가 아니라 '구조적 degenerate' 였던 것과 같은 모양.

후보 (MLP num_cols 에서만 제외, CatBoost cat_feature_cols 은 그대로 — CatBoost 는
규제로 중복을 잘 견디고 제거 upside 가 없음):
  run_total_before    = run_top_before + run_bot_before          (정확한 합)
  away_win_expectancy ~ 100 - home_win_expectancy                (corr ~ -1)
  num_runners_on      = runner_on_1b + 2b + 3b = popcount(base_state)
  count_diff          = strikes_before - balls_before            (engineered, 원천 둘 다 numeric)
개별 4 + 묶음(ALL4) 1.

검증: 3-seed x 4-fold rolling-origin (cutoff7 / 2023 / 2022 / 2021).
  - 단일시드 cutoff7 Phase 1 은 생략 (노이즈 필터라 무가치, 교훈 #26 / 지난 19-of-19 전멸).
  - CatBoost 3-seed 는 fold 당 1회만 학습 후 재사용 (후보가 전부 MLP-only 라 예측 불변).
  - 메타: (a) raw 확률 입력(현행) (b) logit-space 입력 -- 같은 예측으로 둘 다 채점.

판정 (하나라도 어기면 기각):
  - blend Δ(raw meta) 4-fold 중 3/4 이상 승
  - 어느 fold 에서도 MLP-solo Δ -20 이상 붕괴 없음  (blk3 가 걸린 지점)
  - cutoff7 blend Δ 가 노이즈밴드(±7) 밖

사용법:  python -m code.experiment_yudam_mlp_prune_rolling
출력:    scratchpad/mlp_prune_rolling.json + 로그
"""
import gc
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, make_bundle, predict_bundle, get_device, compute_bss,
)
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble

REGIMES = ["cutoff7", "2023", "2022", "2021"]
MLP_SEEDS = (42, 123, 7)
CB_SEEDS = (42, 123, 7)

CANDIDATES = {
    "run_total_before":    ["run_total_before"],
    "away_win_expectancy": ["away_win_expectancy"],
    "num_runners_on":      ["num_runners_on"],
    "count_diff":          ["count_diff"],
    "ALL4":                ["run_total_before", "away_win_expectancy", "num_runners_on", "count_diff"],
}

OUT = os.path.join(
    "/tmp/claude-1000/-home-user-contest-mlp-lgaimers9",
    "80d8950c-489c-47b5-93de-fb7a77e4f443/scratchpad/mlp_prune_rolling.json",
)


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _fit_meta(cat_v, mlp_v, y, space):
    if space == "logit":
        X = np.column_stack([_logit(cat_v), _logit(mlp_v)])
    else:
        X = np.column_stack([cat_v, mlp_v])
    clf = LogisticRegression()
    clf.fit(X, y)
    w0, w1 = (float(c) for c in clf.coef_[0])
    b = float(clf.intercept_[0])
    z = w0 * X[:, 0] + w1 * X[:, 1] + b
    pred = 1.0 / (1.0 + np.exp(-z))
    _, _, score = compute_bss(pred, y)
    return score, (w0, w1, b)


def _train_mlp_preds(train_split, val_split, num_cols, all_cols, device):
    """raw-concat MLP 3-seed 학습 -> val 예측 반환 (build_split/train.py 와 동일 전처리)."""
    tp, ce, ni, ns, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    vp = apply_preprocessing(val_split, CAT_COLS, num_cols, ce, ni, ns)
    Xtc, Xtn, ytr = to_tensors(tp, CAT_COLS, num_cols, TARGET)
    Xvc, Xvn, _ = to_tensors(vp, CAT_COLS, num_cols, TARGET)
    yv = vp[TARGET].values
    del tp, vp
    gc.collect()
    eds = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        Xtc, Xtn, ytr, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
        embed_dims=eds, bin_edges=None, X_val_cat=Xvc, X_val_num=Xvn, y_val=yv,
        seeds=list(MLP_SEEDS), device=device, verbose=False,
    )
    bundle = make_bundle(members, CAT_COLS, num_cols, cat_dims, eds, ce, ni, ns, bin_edges=None)
    preds = predict_bundle(bundle, val_split[all_cols], device=device)
    del members, bundle, Xtc, Xtn, Xvc, Xvn
    gc.collect()
    return preds


def main():
    t0 = time.time()
    device = get_device()
    print(f"device={device} | MLP_SEEDS={MLP_SEEDS} CB_SEEDS={CB_SEEDS}", flush=True)
    results = {}

    for regime in REGIMES:
        print(f"\n{'=' * 66}\nregime={regime}\n{'=' * 66}", flush=True)
        tr, va, num_cols, cat_cols, all_cols = build_split(regime, verbose=True)
        yv = va[TARGET].values

        missing = [c for cols in CANDIDATES.values() for c in cols if c not in num_cols]
        assert not missing, f"후보 컬럼이 num_cols 에 없음: {sorted(set(missing))}"

        # ---- CatBoost 3-seed : fold 당 1회 (MLP-only 후보라 예측 불변) ----
        cb_res = train_catboost_ensemble(
            tr[cat_cols], tr[TARGET].values, va[cat_cols], yv,
            seeds=list(CB_SEEDS), verbose=False, params=_yudam_catboost_params(),
        )
        cat_v = predict_catboost_ensemble([m for m, _ in cb_res], va[cat_cols])
        _, _, cat_solo = compute_bss(cat_v, yv)
        del cb_res
        gc.collect()

        # ---- baseline MLP ----
        mlp_v_base = _train_mlp_preds(tr, va, num_cols, all_cols, device)
        _, _, mlp_solo_base = compute_bss(mlp_v_base, yv)
        base_raw, w_base_raw = _fit_meta(cat_v, mlp_v_base, yv, "raw")
        base_logit, w_base_logit = _fit_meta(cat_v, mlp_v_base, yv, "logit")
        print(f"[{regime}] BASE  cat_solo={cat_solo:.2f} | mlp_solo={mlp_solo_base:.2f} | "
              f"blend_raw={base_raw:.2f} (w={w_base_raw[0]:.3f}/{w_base_raw[1]:.3f}/{w_base_raw[2]:.3f}) | "
              f"blend_logit={base_logit:.2f}", flush=True)

        reg = {
            "n_val": int(len(yv)), "cat_solo": cat_solo,
            "base": dict(mlp_solo=mlp_solo_base, blend_raw=base_raw, blend_logit=base_logit,
                         w_raw=w_base_raw, w_logit=w_base_logit),
            "candidates": {},
        }

        for name, cols in CANDIDATES.items():
            drop = set(cols)
            c_num = [c for c in num_cols if c not in drop]
            mlp_v = _train_mlp_preds(tr, va, c_num, all_cols, device)
            _, _, mlp_solo = compute_bss(mlp_v, yv)
            c_raw, w_raw = _fit_meta(cat_v, mlp_v, yv, "raw")
            c_logit, w_logit = _fit_meta(cat_v, mlp_v, yv, "logit")
            reg["candidates"][name] = dict(
                dropped=cols, n_num=len(c_num),
                mlp_solo=mlp_solo, blend_raw=c_raw, blend_logit=c_logit,
                d_mlp=mlp_solo - mlp_solo_base,
                d_blend_raw=c_raw - base_raw,
                d_blend_logit=c_logit - base_logit,
                w_raw=w_raw, w_logit=w_logit,
            )
            print(f"[{regime}] {name:20s} nnum={len(c_num):3d} | "
                  f"mlp Δ{mlp_solo - mlp_solo_base:+7.2f} | "
                  f"blend_raw Δ{c_raw - base_raw:+7.2f} | "
                  f"blend_logit Δ{c_logit - base_logit:+7.2f}", flush=True)
            del mlp_v
            gc.collect()

        results[regime] = reg
        del tr, va, mlp_v_base, cat_v
        gc.collect()
        _dump(results)

    _dump(results)
    _summary(results)
    print(f"\n총 소요 {time.time() - t0:.1f}s -> {OUT}", flush=True)


def _dump(results):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _summary(results):
    print(f"\n{'=' * 66}\n요약: blend_raw Δ (프로덕션 메타 = raw)\n{'=' * 66}", flush=True)
    names = list(CANDIDATES)
    hdr = f"{'candidate':20s} " + " ".join(f"{r:>9s}" for r in REGIMES) + "   wins  mean"
    print(hdr, flush=True)
    for name in names:
        ds = [results[r]["candidates"][name]["d_blend_raw"] for r in REGIMES if r in results]
        wins = sum(1 for d in ds if d > 0)
        row = f"{name:20s} " + " ".join(f"{d:+9.2f}" for d in ds)
        print(f"{row}   {wins}/{len(ds)}  {np.mean(ds):+.2f}", flush=True)

    print(f"\n{'-' * 66}\nMLP-solo Δ (붕괴 감시: 어느 fold든 -20 이하면 기각)\n{'-' * 66}", flush=True)
    for name in names:
        ds = [results[r]["candidates"][name]["d_mlp"] for r in REGIMES if r in results]
        flag = "  <-- 붕괴!" if any(d <= -20 for d in ds) else ""
        print(f"{name:20s} " + " ".join(f"{d:+9.2f}" for d in ds) + flag, flush=True)

    print(f"\n{'-' * 66}\nlogit-space 메타 vs raw 메타 (baseline blend, fold별)\n{'-' * 66}", flush=True)
    for r in REGIMES:
        if r not in results:
            continue
        b = results[r]["base"]
        print(f"{r:>9s}  raw={b['blend_raw']:.2f}  logit={b['blend_logit']:.2f}  "
              f"Δ(logit-raw)={b['blend_logit'] - b['blend_raw']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
