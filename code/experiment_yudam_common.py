# code/experiment_yudam_common.py
"""2026-08-27 유담님 레시피 전면 교체 이후의 공용 실험 하네스.

기존 `code/thirdmodel_common.py::build_split`은 옛 파이프라인(PLE / tier-feed /
TE-residual 도입 이전) 스냅샷이라, 새 베이스라인(raw-concat MLP + 트랙맨64 복원 +
reverse_rate 진행분 + v2 CatBoost HP) 위에서 피처를 재검증하려면 쓸 수 없다.
이 파일의 `build_split`은 현재 `code/train.py::main()`을 한 글자도 다르지 않게
재현한다. 피처 재검증(특히 "PLE 없으니 예전에 기각된 트랙맨/시즌 분해 피처를 다시")
스크립트들이 이걸 공통 하네스로 쓴다.

레짐:
- ``cutoff7`` : train = season<2024 | (season==2024 & month<7), val = 2024/07~10.
               프로덕션 주 기준. 트랙맨 holdout=2024.
- ``2023``    : train = season<2023, val = season==2023. F1 필터가 2023 train에서
               F리그를 100% 제거하는데 val엔 ~10% 섞여 구조적으로 왜곡됨 — 참고용.
               (memory: cutoff7_season2023_regime_flip_diagnosed)

실험 훅 (`build_split` 인자):
  ``add_features_fn(df_full, holdout) -> (df_full, exclude_from_cat, exclude_from_mlp)``
      ``apply_same_hand`` 직후 · 컬럼 목록 확정 전에 호출된다. 새 컬럼을 df_full에
      직접 추가하고, CatBoost/MLP 각각에서 빼야 할 컬럼 집합을 돌려준다(기본 both).
      트랙맨류는 holdout(=val 시즌)을 받아 within-season leak을 피할 것.
  ``drop_cols(all_cols) -> iterable``  (ablation용, 예: 트랙맨64 컬럼 전체 제거)

리소스: 항상 단일 fold. 기본 MLP 3-seed + CatBoost 3-seed 스크리닝.
``seeds=(1,1)``이면 1-seed씩(가장 가벼움). train.csv+trackman 로드 ~4GB,
조인/학습 포함 peak ~7GB — `free -h`로 8GB+ 여유 확인 후 실행.
"""
import gc
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, make_bundle, predict_bundle, get_device, compute_bss,
)
from code.catboost_model import predict_catboost_ensemble, train_catboost_ensemble
from code.train import (
    process_trackman_features_safe, add_engineered_features, apply_f1_filter,
    apply_same_hand, apply_te_residual_features, SAME_HAND_COLS, TE_RESIDUAL_COLS,
    YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS, is_trackman64,
)
from code.trackman_pitcher_features import merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET = "control_success"
V2_HPARAMS_PATH = "./teammate/yudam/model_py311/best_catboost_hparams_v2.json"

REGIMES = {
    #  name    : (train_mask_fn,                                            holdout_season)
    "cutoff7": (lambda d: (d["season"] < 2024) | ((d["season"] == 2024) & (d["game_month"] < 7)), 2024),
    "2023":    (lambda d: d["season"] < 2023, 2023),
    # expand-window rolling-origin fold들 (train = season<Y, val = season==Y).
    "2022":    (lambda d: d["season"] < 2022, 2022),
    "2021":    (lambda d: d["season"] < 2021, 2021),
}


def _yudam_catboost_params():
    with open(V2_HPARAMS_PATH) as f:
        t = json.load(f)["best_params"]
    return dict(
        depth=t["depth"], learning_rate=t["learning_rate"], l2_leaf_reg=t["l2_leaf_reg"],
        random_strength=t["random_strength"], bagging_temperature=t["bagging_temperature"],
        border_count=t["border_count"], min_data_in_leaf=t["min_data_in_leaf"],
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    )


def build_split(regime="cutoff7", add_features_fn=None, drop_cols=None, apply_f1=True, verbose=True,
                cutoff_month=None, fixed_val_month=None, f1_boundary=2022, keep_trackman64=False):
    """현재 code/train.py::main()의 전처리를 그대로 재현해 (train_split, val_split,
    num_cols, cat_feature_cols, all_cols)를 돌려준다.

    add_features_fn / drop_cols 로 피처를 추가·제거해 재검증한다.

    Step 2 스윕용 knob (regime="cutoff7" 일 때만):
      cutoff_month    : 2024 행이 train 으로 들어가는 월 경계 (기본 7 = 프로덕션).
                        train = season<2024 | (season==2024 & month < cutoff_month).
      fixed_val_month : val = season==2024 & month >= 이 값. None 이면 cutoff_month 와 동일
                        (=프로덕션). 스윕에서 cutoff_month 를 바꿔도 val 셋을 고정해 점수를
                        비교가능하게 하려면 여기에 고정값(예: 8)을 준다.
      f1_boundary     : F1 필터 상한 시즌 (기본 2022). None 이면 F1 필터 완전 비활성.

    keep_trackman64 : 기본 False = 트랙맨64 상황조인 물리량 컬럼을 CatBoost/MLP 피처목록
                      양쪽에서 제외 (실전 1117.03 레시피 = 현행 프로덕션. code/train.py 와 동일).
                      컬럼 자체는 all_cols/train_split/val_split 에 남는다. True 로 주면
                      옛 candidate B raw(143피처) 동작 — collapse-sim 등 트랙맨64 를 살려서
                      비교해야 하는 스크립트 전용.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {list(REGIMES)}")
    _default_train_mask_fn, holdout = REGIMES[regime]
    val_season = holdout

    if regime == "cutoff7" and (cutoff_month is not None or fixed_val_month is not None):
        cm = 7 if cutoff_month is None else int(cutoff_month)
        train_mask_fn = (lambda d, _cm=cm: (d["season"] < 2024)
                         | ((d["season"] == 2024) & (d["game_month"] < _cm)))
    else:
        train_mask_fn = _default_train_mask_fn

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    tr_final, _match_cols, _trackman_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    del df
    gc.collect()
    train_df = tr_final.dropna(subset=[TARGET]).reset_index(drop=True)
    del tr_final
    gc.collect()

    # league_success_mean 은 스칼라 평균이라 행 순서와 무관 — add_features_fn 이
    # 행을 재정렬할 수 있으므로(트랙맨 asof 조인 등) 마스크는 mutation 이후에 다시 만든다.
    _pre_train_mask = train_mask_fn(train_df)
    league_success_mean = train_df.loc[_pre_train_mask, TARGET].mean()
    train_df = add_engineered_features(train_df, league_success_mean)
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=holdout)
    train_df = apply_same_hand(train_df)

    exclude_from_cat, exclude_from_mlp = set(), set()
    if add_features_fn is not None:
        train_df, exc_cat, exc_mlp = add_features_fn(train_df, holdout)
        exclude_from_cat |= set(exc_cat or ())
        exclude_from_mlp |= set(exc_mlp or ())
    train_df = train_df.reset_index(drop=True)

    del df_trm
    gc.collect()

    # 마스크는 train_df 자신의 season/game_month 컬럼에서 재계산(위치 기반, 재정렬 안전).
    train_mask = train_mask_fn(train_df).to_numpy()
    val_mask = (train_df["season"] == val_season).to_numpy()
    if regime == "cutoff7":
        if fixed_val_month is not None:
            _vm = int(fixed_val_month)
        elif cutoff_month is not None:
            _vm = int(cutoff_month)
        else:
            _vm = 7
        val_mask = val_mask & (train_df["game_month"] >= _vm).to_numpy()

    dropped = set(drop_cols(list(train_df.columns))) if drop_cols is not None else set()
    base_drop = {"row_id", TARGET} | dropped
    all_cols = [c for c in train_df.columns if c not in base_drop]

    # code/train.py 와 동일한 라우팅: same_hand -> MLP만, TE-residual -> CatBoost만,
    # 트랙맨64(상황조인 물리량) -> 양쪽 모두 제외 (실전 1117.03 레시피, 2025 추론때
    # season 매칭 0건으로 상수붕괴 → CatBoost miscalibrate). 컬럼 자체는 all_cols 에
    # 남겨 df/train_split/val_split 에는 존재 (collapse-sim 등이 keep_trackman64=True 로
    # 되살려 쓸 수 있게).
    _tm64_excl = (lambda c: False) if keep_trackman64 else is_trackman64
    cat_feature_cols = [c for c in all_cols
                        if c not in SAME_HAND_COLS and c not in exclude_from_cat
                        and not _tm64_excl(c)]
    num_cols = [c for c in all_cols
                if c not in CAT_COLS and c not in TE_RESIDUAL_COLS and c not in exclude_from_mlp
                and not _tm64_excl(c)]

    train_split = train_df.loc[train_mask, all_cols + [TARGET]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, all_cols + [TARGET]].reset_index(drop=True)
    del train_df
    gc.collect()

    if apply_f1 and f1_boundary is not None:
        if int(f1_boundary) == 2022:
            train_split = apply_f1_filter(train_split)
        else:
            _b = int(f1_boundary)
            _before = len(train_split)
            train_split = train_split[
                ~((train_split["game_type"] == "F") & (train_split["season"] <= _b))
            ].reset_index(drop=True)
            print(f"[F1 필터] game_type=='F' & season<={_b} 제거: {_before} -> {len(train_split)}행 "
                  f"({_before - len(train_split)}행 제거)")
    elif f1_boundary is None:
        print("[F1 필터] 비활성 (f1_boundary=None)")

    te_prior = train_split[TARGET].mean()
    te_source = train_split  # code/train.py 와 동일: TE-residual 추가 전 train_split 을 source 로 고정
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)
    cat_feature_cols = cat_feature_cols + TE_RESIDUAL_COLS

    if verbose:
        _tm64_n = sum(1 for c in all_cols if is_trackman64(c))
        _tm64_note = (f" | 트랙맨64 {_tm64_n}개 " + ("포함(keep_trackman64=True)" if keep_trackman64 else "제외")) if _tm64_n else ""
        print(f"[build_split:{regime}] train={len(train_split)} val={len(val_split)} | "
              f"CatBoost {len(cat_feature_cols)}피처 / MLP {len(num_cols) + len(CAT_COLS)}피처 "
              f"(num {len(num_cols)} + cat {len(CAT_COLS)}){_tm64_note}", flush=True)
        if dropped:
            print(f"[build_split] dropped: {sorted(dropped)}", flush=True)
        if exclude_from_cat or exclude_from_mlp:
            print(f"[build_split] cat 제외 {sorted(exclude_from_cat)} | mlp 제외 {sorted(exclude_from_mlp)}", flush=True)
    return train_split, val_split, num_cols, cat_feature_cols, all_cols


def run_experiment(train_split, val_split, num_cols, cat_feature_cols, all_cols,
                   mlp_seeds=None, cb_seeds=None, label="", verbose=True):
    """raw-concat MLP(bin_edges=None) + v2 CatBoost 를 학습하고 val 전체로 채점한다.
    메타모델은 val 전체 fit + val 전체 채점(이 repo 관례, EXPERIMENTS.md 숫자와 직접 비교).
    반환: dict(cat_solo, mlp_solo, blend, w_cat, w_mlp, intercept, cb_best_iters)."""
    mlp_seeds = list(mlp_seeds) if mlp_seeds is not None else list(YUDAM_ENSEMBLE_SEEDS)
    cb_seeds = list(cb_seeds) if cb_seeds is not None else list(YUDAM_CATBOOST_SEEDS)
    device = get_device()

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, num_cols, TARGET)
    y_val = val_proc[TARGET].values
    del train_proc, val_proc
    gc.collect()

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims, bin_edges=None,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
        seeds=mlp_seeds, device=device, verbose=False,
    )
    mlp_bundle = make_bundle(members, CAT_COLS, num_cols, cat_dims, embed_dims,
                             cat_encoder, num_imputer, num_scaler, bin_edges=None)
    mlp_val = predict_bundle(mlp_bundle, val_split[all_cols], device=device)

    cb_results = train_catboost_ensemble(
        train_split[cat_feature_cols], train_split[TARGET].values,
        val_split[cat_feature_cols], val_split[TARGET].values,
        seeds=cb_seeds, verbose=False, params=_yudam_catboost_params(),
    )
    cb_models = [m for m, _ in cb_results]
    cb_best_iters = [it for _, it in cb_results]
    cat_val = predict_catboost_ensemble(cb_models, val_split[cat_feature_cols])

    _, _, cat_solo = compute_bss(cat_val, y_val)
    _, _, mlp_solo = compute_bss(mlp_val, y_val)

    clf = LogisticRegression()
    clf.fit(np.column_stack([cat_val, mlp_val]), y_val)
    w_cat, w_mlp = (float(c) for c in clf.coef_[0])
    intercept = float(clf.intercept_[0])
    blend_pred = 1.0 / (1.0 + np.exp(-(w_cat * cat_val + w_mlp * mlp_val + intercept)))
    _, _, blend = compute_bss(blend_pred, y_val)

    res = dict(cat_solo=cat_solo, mlp_solo=mlp_solo, blend=blend,
               w_cat=w_cat, w_mlp=w_mlp, intercept=intercept, cb_best_iters=cb_best_iters,
               n_mlp_seed=len(mlp_seeds), n_cb_seed=len(cb_seeds))
    if verbose:
        tag = f"[{label}] " if label else ""
        print(f"{tag}CatBoost solo={cat_solo:.2f} | MLP solo={mlp_solo:.2f} | blend={blend:.2f} "
              f"(w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} b={intercept:.3f}, "
              f"mlp{len(mlp_seeds)}seed/cb{len(cb_seeds)}seed)", flush=True)
    return res


def compare(regime="cutoff7", add_features_fn=None, drop_cols=None, mlp_seeds=None, cb_seeds=None,
            label="candidate"):
    """baseline(피처 변경 없음) vs candidate 를 같은 레짐에서 순차 실행해 델타를 출력한다."""
    print(f"\n==================== regime={regime} ====================", flush=True)
    base = run_experiment(*build_split(regime), mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label="baseline")
    gc.collect()
    cand = run_experiment(*build_split(regime, add_features_fn=add_features_fn, drop_cols=drop_cols),
                          mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label=label)
    d_cat = cand["cat_solo"] - base["cat_solo"]
    d_mlp = cand["mlp_solo"] - base["mlp_solo"]
    d_bl = cand["blend"] - base["blend"]
    print(f"\n[Δ {label} vs baseline | {regime}]  CatBoost {d_cat:+.2f} | MLP {d_mlp:+.2f} | blend {d_bl:+.2f}", flush=True)
    return dict(regime=regime, baseline=base, candidate=cand, d_cat=d_cat, d_mlp=d_mlp, d_blend=d_bl)
