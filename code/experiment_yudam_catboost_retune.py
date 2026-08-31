# code/experiment_yudam_catboost_retune.py
"""[Step 3] 유담 피처셋(트랙맨64 복원 + reverse_rate + TE-residual + same_hand,
cutoff7) 위에서 CatBoost 하이퍼파라미터 재탐색.

구 `code/tune.py`는 (1) import 가 깨졌고(`thirdmodel_common`), (2) cutoff7 val 한
창에서만 목적함수를 재서 이 프로젝트가 5회 문서화한 "CatBoost HP 서치가 cutoff7 val
에 과적합" 패턴(memory: catboost_te_residual_retune_rejected)에 그대로 빠졌다.

이 스크립트는 그 실패모드를 목적함수에 직접 넣어 회피한다:
  objective(trial) = mean( BSS[cutoff7 val], BSS[2022 fold val] )   # rolling-origin 내장
각 fold 1-seed CatBoost. 두 split 은 1회만 build 해 Pool 재사용.

탐색범위는 유담 v2(depth7 / lr0.0468 / l2 19.39 / rs8.25 / bt0.128 / bc179 / mdl1)
주변으로 살짝만 넓게 잡는다 — 유담이 이미 v1->v2 두 번 튜닝해 1092 를 낸 값이라
크게 벗어날 이유가 없고, 벗어날수록 과적합 위험만 커진다.

Optuna 후: best params 와 유담 v2 를 두 fold + season2023 게이트에서 3-seed 로 재확인.
채택 조건: cutoff7 / 2022 / 2023 **세 fold 전부** 유담 v2 이상.

사용법:
  TUNE_TRIALS=24 python -m code.experiment_yudam_catboost_retune
  TUNE_TRIALS=8  python -m code.experiment_yudam_catboost_retune   # 빠른 확인
"""
import gc
import json
import os

import numpy as np
import optuna

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.mlp_model import compute_bss
from code.train import YUDAM_CATBOOST_SEEDS

N_TRIALS = int(os.environ.get("TUNE_TRIALS", 24))
OUT_PATH = "./scratchpad/catboost_retune_result.json"
TUNE_FOLDS = ["cutoff7", "2022"]
GATE_FOLDS = ["cutoff7", "2022", "2023"]


def _load_fold(regime):
    ts, vs, _num, catf, _all = build_split(regime=regime, verbose=True)
    return dict(
        Xtr=ts[catf], ytr=ts[TARGET].values,
        Xva=vs[catf], yva=vs[TARGET].values, catf=catf,
    )


def _score(fold, params, seeds):
    res = train_catboost_ensemble(
        fold["Xtr"], fold["ytr"], fold["Xva"], fold["yva"],
        seeds=seeds, verbose=False, params=params,
    )
    models = [m for m, _ in res]
    preds = predict_catboost_ensemble(models, fold["Xva"])
    return compute_bss(preds, fold["yva"])[2]


def _params_from_trial(trial):
    v2 = _yudam_catboost_params()
    p = dict(v2)
    p["depth"] = trial.suggest_int("depth", 6, 8)
    p["learning_rate"] = trial.suggest_float("learning_rate", 0.025, 0.09, log=True)
    p["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 6.0, 40.0, log=True)
    p["random_strength"] = trial.suggest_float("random_strength", 3.0, 14.0)
    p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.2)
    p["border_count"] = trial.suggest_int("border_count", 96, 254)
    p["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 40, log=True)
    return p


def main():
    print(f"[retune] N_TRIALS={N_TRIALS} | tune folds={TUNE_FOLDS} (mean BSS objective)", flush=True)
    folds = {r: _load_fold(r) for r in TUNE_FOLDS}
    gc.collect()

    # 유담 v2 기준선 (1-seed, 목적함수와 동일 조건)
    v2 = _yudam_catboost_params()
    v2_1seed = {r: _score(folds[r], v2, [YUDAM_CATBOOST_SEEDS[0]]) for r in TUNE_FOLDS}
    v2_mean = np.mean(list(v2_1seed.values()))
    print(f"[retune] 유담 v2 1-seed: " + " ".join(f"{r}={v2_1seed[r]:.2f}" for r in TUNE_FOLDS)
          + f" | mean={v2_mean:.2f}", flush=True)

    def objective(trial):
        p = _params_from_trial(trial)
        s = {r: _score(folds[r], p, [YUDAM_CATBOOST_SEEDS[0]]) for r in TUNE_FOLDS}
        for r in TUNE_FOLDS:
            trial.set_user_attr(f"bss_{r}", s[r])
        m = float(np.mean(list(s.values())))
        print(f"[trial {trial.number:02d}] " + " ".join(f"{r}={s[r]:.2f}" for r in TUNE_FOLDS)
              + f" | mean={m:.2f} (v2 mean={v2_mean:.2f}) params={trial.params}", flush=True)
        return m

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial({k: v for k, v in {
        "depth": 7, "learning_rate": 0.046773, "l2_leaf_reg": 19.39149,
        "random_strength": 8.2541, "bagging_temperature": 0.127747,
        "border_count": 179, "min_data_in_leaf": 1,
    }.items()})  # 유담 v2 자체를 trial 0 로
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\n{'='*66}\n[retune] Optuna best mean={study.best_value:.2f} (유담 v2 mean={v2_mean:.2f}, "
          f"Δ={study.best_value - v2_mean:+.2f})\nbest params={study.best_params}\n{'='*66}", flush=True)

    if study.best_value <= v2_mean + 1.0:
        print("[retune] Optuna best 가 유담 v2 대비 +1.0 미만 -> 게이트 스킵, 재튜닝 이득 없음(예상대로).", flush=True)
        _dump(study, v2_mean, v2_1seed, gate=None)
        return

    # 게이트: best params vs v2, 3-seed, 3 fold (cutoff7/2022 재사용 + 2023 추가)
    best_p = dict(v2); best_p.update(study.best_params)
    seeds3 = list(YUDAM_CATBOOST_SEEDS[:3])
    if "2023" not in folds:
        folds["2023"] = _load_fold("2023")
    gate = {}
    for r in GATE_FOLDS:
        b = _score(folds[r], best_p, seeds3)
        v = _score(folds[r], v2, seeds3)
        gate[r] = dict(best=b, v2=v, delta=b - v)
        print(f"[gate {r}] best={b:.2f}  v2={v:.2f}  Δ={b - v:+.2f}", flush=True)
    wins = sum(1 for r in GATE_FOLDS if gate[r]["delta"] > 0)
    verdict = "ADOPT 후보" if wins == 3 else f"REJECT ({wins}/3 fold만 우세)"
    print(f"\n[retune] 게이트 {wins}/3 -> {verdict}", flush=True)
    _dump(study, v2_mean, v2_1seed, gate=gate, best_p=best_p, verdict=verdict)


def _dump(study, v2_mean, v2_1seed, gate, best_p=None, verdict=None):
    out = dict(
        n_trials=N_TRIALS, tune_folds=TUNE_FOLDS,
        optuna_best_mean=study.best_value, optuna_best_params=study.best_params,
        v2_mean_1seed=v2_mean, v2_1seed=v2_1seed,
        gate=gate, best_params_full=best_p, verdict=verdict,
    )
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(f"[retune] 결과 저장: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
