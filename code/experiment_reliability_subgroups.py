# code/experiment_reliability_subgroups.py
"""reliability diagram 진단(code/experiment_reliability_diagram.py)의 후속.
남은 두 질문을 검사한다:

(2) 메타모델 이후에 season/game_type 등 서브그룹별 추가 재보정 레이어가 필요한가?
    -> cutoff7 검증셋(단일 시즌이라 season 자체는 못 가름)에서 game_month/game_type별
       REL을 보고, "시즌 자체"에 대해서는 rolling-origin 3-fold(train<val_season,
       val_season in {2021,2022,2023}, R-only, code/experiment_catboost_seed_ensemble_foldcheck.py와
       동일 컨벤션)로 fold(=미래 미지 시즌)마다 REL이 커지는 추세가 있는지 본다.

(3) cold-start(asof_pitcher_n 작은 표본) 구간을 학습 시 다운웨이팅(방금 기각,
    레짐 완전 반전)이 아니라 사후 재보정으로 다루면 다를까?
    -> 다운웨이팅이 레짐마다 반대로 터졌다는 것 자체가 "cold-start 미보정 방향이
       레짐마다 다르다"는 가설과 일치한다. 이걸 학습을 거치지 않고 순수하게
       예측값의 REL만으로 직접 검증한다: cold-start 버킷의 REL/gap 부호가
       fold(레짐)마다 같은 방향인지 반대 방향인지 본다. 반대 방향이면 고정
       방향의 사후 재보정(예: cold-start를 평균 쪽으로 shrink)도 다운웨이팅과
       똑같은 이유로 일반화에 실패할 것이라고 예상할 수 있다.
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS
from code.trackman_pitcher_features import merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023]


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def train_one(seed, X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params["random_seed"] = seed
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def murphy(preds, y):
    obar = y.mean()
    unc = obar * (1 - obar)
    brier, bss, score = compute_bss(preds, y)
    rel_res = brier - unc  # = REL - RES (부호 포함, score 계산엔 굳이 분리 불필요)
    return unc, brier, score, (preds.mean() - obar)


def subgroup_table(preds, y, group_key, name):
    df = pd.DataFrame({"pred": preds, "y": y, "g": group_key})
    out = []
    for g, sub in df.groupby("g"):
        n = len(sub)
        pbar = sub["pred"].mean()
        obar = sub["y"].mean()
        _, brier, score, _ = murphy(sub["pred"].values, sub["y"].values)
        out.append({name: g, "n": n, "mean_pred": pbar, "actual_rate": obar, "gap": pbar - obar, "score": score})
    return pd.DataFrame(out)


def run_fold(df_all, df_trm, val_season, seed=42):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols]

    all_cols = base_features + [TARGET_COL, "game_month", "asof_pitcher_n", "asof_batter_n"]
    all_cols = list(dict.fromkeys(all_cols))  # 중복 제거(이미 base_features에 포함된 컬럼들)
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, prior)
    val_split = apply_te_residual_features(te_source, val_split, prior)
    cat_feature_cols = base_features + TE_RESIDUAL_COLS

    X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]

    t0 = time.time()
    preds, best_iter = train_one(seed, X_train, y_train, X_val, y_val)
    unc, brier, score, mean_gap = murphy(preds, y_val)
    print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only, n_train={len(X_train)}, n_val={len(X_val)}) ===")
    print(f"  전체: Score={score:.2f}  ō={y_val.mean():.4f}  전체평균gap(pred-actual)={mean_gap:+.4f}  ({time.time()-t0:.1f}s)")

    print("  -- game_month별 --")
    print(subgroup_table(preds, y_val, val_split["game_month"].values, "month").to_string(index=False, formatters={
        "mean_pred": "{:.4f}".format, "actual_rate": "{:.4f}".format, "gap": "{:+.4f}".format, "score": "{:.1f}".format}))

    # cold-start 버킷: asof_pitcher_n을 val_split 내에서 4분위로 나눔 (표본 적을수록 콜드스타트)
    n_pitcher = val_split["asof_pitcher_n"].values
    q = pd.qcut(n_pitcher, 4, labels=["Q1(콜드)", "Q2", "Q3", "Q4(웜)"], duplicates="drop")
    print("  -- asof_pitcher_n 4분위(콜드스타트) --")
    print(subgroup_table(preds, y_val, q, "n_bucket").to_string(index=False, formatters={
        "mean_pred": "{:.4f}".format, "actual_rate": "{:.4f}".format, "gap": "{:+.4f}".format, "score": "{:.1f}".format}))

    cold_mask = n_pitcher <= np.quantile(n_pitcher, 0.25)
    cold_gap = preds[cold_mask].mean() - y_val[cold_mask].mean()
    warm_gap = preds[~cold_mask].mean() - y_val[~cold_mask].mean()
    print(f"  -- 콜드(하위25%) gap={cold_gap:+.4f} (n={cold_mask.sum()})  vs  웜(상위75%) gap={warm_gap:+.4f} (n={(~cold_mask).sum()}) --")

    return {"val_season": val_season, "score": score, "cold_gap": cold_gap, "warm_gap": warm_gap}


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")

    results = []
    for val_season in FOLD_SEASONS:
        results.append(run_fold(df_all, df_trm, val_season))

    print(f"\n{'='*78}\n=== fold별 콜드스타트 gap 방향 요약 ===\n{'='*78}")
    for r in results:
        print(f"  val={r['val_season']}: score={r['score']:.2f}  cold_gap={r['cold_gap']:+.4f}  warm_gap={r['warm_gap']:+.4f}")


if __name__ == "__main__":
    main()
