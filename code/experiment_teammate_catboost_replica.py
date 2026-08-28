# code/experiment_teammate_catboost_replica.py
"""사용자 지적: 팀원의 CatBoost 단독 1,000점대 레시피(A~H, 앞서
code/experiment_teammate_features_b.py에서 B'/D/H/E/F 개별·조합 스크리닝 했던 그
피처들)를 재검증할 때, 우리 프로덕션 피처셋(F1필터 ON, TE-residual/coarse
pitchmix/pair-matchup 포함, 단일시드) 위에 얹어서 테스트했다 — 그런데 팀원은
(1) F1 필터를 아예 안 쓰고 2019~2022 F리그 데이터를 그대로 학습에 포함시켰고,
(2) 5-seed CatBoost bagging을 썼다고 확인됨. 이 두 가지는 이전 스크리닝에서
전혀 격리되지 않은 변수였다 — B'/D/H/E/F가 기각된 건 맞지만, "F1필터 끄기"와
"5시드 배깅"의 개별/결합 효과는 아직 미측정.

이 스크립트는 두 갈래로 구성된다.

1) `run_isolation_grid`: 팀원 A~H 레시피를 최대한 충실히 재현(A/C/G는
   `code/train.py::add_engineered_features` 재사용, B'/D/H/E/F는
   `code/experiment_teammate_features_b.py`의 헬퍼 재사용, TE-residual/coarse
   pitchmix/pair-matchup은 포함하지 않음 — 팀원 레시피에 없던 우리 전용 피처라서
   제외, pitcher_id/batter_id는 팀원 문서의 "최종 피처에서 제외" 명시를 따름)한
   뒤, F1필터 on/off × 1seed/5seed 배깅의 2x2 그리드를 dual-regime으로 스크리닝.
   CatBoost는 시드별 반복 학습이 필요하므로, seed 목록[42,123,7,2024,99] 5개를
   한 번씩만 학습하고 누적 평균으로 1~5-seed 전부를 얻는다(재학습 없음).

2) `--save-preds`: 가장 유력한 변형(F1 OFF + 5seed)의 검증 예측을 .npy로 저장
   —  이후 우리 프로덕션 CatBoost(F1 ON) + 이 replica + MLP N-seed 3-way
   스태킹 실험에서 재사용하기 위함. train.csv를 매번 새로 읽고 동일한 val_mask
   로직을 쓰므로, 다른 스크립트(`code/thirdmodel_common.py::build_split`)가
   만드는 val_split과 행 순서가 일치한다(둘 다 원본 df를 그대로 읽어 동일한
   season/game_month 마스크만 적용, 별도 셔플/정렬 없음).

사용법:
  python -m code.experiment_teammate_catboost_replica --cutoff7
  python -m code.experiment_teammate_catboost_replica --holdout 2023
  python -m code.experiment_teammate_catboost_replica --cutoff7 --save-preds
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_f1_filter
from code.trackman_pitcher_features import clean_trackman
from code.experiment_teammate_features_b import (
    prep_trackman_for_ef, build_std5_table, build_gap4_table, apply_table,
    build_prev_season_league_table, apply_season_relative, apply_hand_features,
    apply_hand_matchup, STD5_COLS, GAP4_COLS, HAND_COLS, HAND_MATCHUP_COL,
)

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SEED_POOL = [42, 123, 7, 2024, 99]


def build_teammate_split(cutoff7, holdout, apply_f1):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)  # A(원본) + C(인터랙션) + G(시즌진행분)

    train_split = df.loc[train_mask].reset_index(drop=True)
    val_split = df.loc[val_mask].reset_index(drop=True)

    if apply_f1:
        before = len(train_split)
        train_split = apply_f1_filter(train_split)
        print(f"  [F1 필터] {before} -> {len(train_split)}행")
    else:
        print(f"  [F1 필터 OFF] {len(train_split)}행 그대로 (19-22 F리그 포함)")

    # B'(시즌상대 리그베이스라인)
    season_table = build_prev_season_league_table(train_split, target_col=TARGET_COL)
    fallback_mean = train_split[TARGET_COL].mean()
    train_split = apply_season_relative(train_split, season_table, fallback_mean)
    val_split = apply_season_relative(val_split, season_table, fallback_mean)
    # D(same_hand) + H(hand_matchup)
    train_split = apply_hand_features(train_split)
    val_split = apply_hand_features(val_split)
    train_split = apply_hand_matchup(train_split)
    val_split = apply_hand_matchup(val_split)

    # E(std5) + F(gap4) - 트랙맨, season-1 앵커, 6키 매칭
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    trm_prepped = prep_trackman_for_ef(df_trm_clean)
    std5_table, std5_fallback = build_std5_table(trm_prepped)
    gap4_table, gap4_fallback = build_gap4_table(trm_prepped)
    train_split = apply_table(train_split, std5_table, std5_fallback, STD5_COLS)
    val_split = apply_table(val_split, std5_table, std5_fallback, STD5_COLS)
    train_split = apply_table(train_split, gap4_table, gap4_fallback, GAP4_COLS)
    val_split = apply_table(val_split, gap4_table, gap4_fallback, GAP4_COLS)

    # 팀원 문서 A 그룹 각주: pitcher_id/batter_id는 시즌진행분 계산에만 쓰고 최종 피처에서 제외
    drop_cols = ["row_id", TARGET_COL, "pitcher_id", "batter_id"]
    feature_cols = [c for c in train_split.columns if c not in drop_cols]
    cat_features = ["game_type", "base_state", HAND_MATCHUP_COL]

    return train_split, val_split, feature_cols, cat_features


def train_one(seed, X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["random_seed"] = seed
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def run_isolation_grid(cutoff7, holdout, n_seeds, save_preds):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 팀원 A-H replica 격리 그리드: {label} ===\n{'='*70}")

    results = {}
    for apply_f1 in [True, False]:
        f1_tag = "F1=ON" if apply_f1 else "F1=OFF"
        print(f"\n--- {f1_tag} ---")
        train_split, val_split, feature_cols, cat_features = build_teammate_split(cutoff7, holdout, apply_f1)
        X_train, y_train = train_split[feature_cols], train_split[TARGET_COL].values
        X_val, y_val = val_split[feature_cols], val_split[TARGET_COL].values

        preds_list = []
        for seed in SEED_POOL[:n_seeds]:
            t0 = time.time()
            preds, best_iter = train_one(seed, X_train, y_train, X_val, y_val, cat_features)
            score = compute_bss(preds, y_val)[2]
            print(f"    seed={seed}: Val Score={score:.2f} (best_iter={best_iter}, {time.time()-t0:.1f}s)")
            preds_list.append(preds)

        for k in [1, n_seeds]:
            ens_pred = np.mean(preds_list[:k], axis=0)
            ens_score = compute_bss(ens_pred, y_val)[2]
            results[(apply_f1, k)] = ens_score
            print(f"  [{f1_tag}, {k}-seed] Val Score={ens_score:.2f}")

        if save_preds and not apply_f1:
            out = f"/tmp/teammate_replica_{label}_f1off_{n_seeds}seed_preds.npy"
            np.save(out, np.mean(preds_list, axis=0))
            np.save(f"/tmp/teammate_replica_{label}_yval.npy", y_val)
            print(f"  [저장] {out}")

    print(f"\n--- {label} 요약 (단독 CatBoost score) ---")
    for apply_f1 in [True, False]:
        f1_tag = "F1=ON" if apply_f1 else "F1=OFF"
        s1 = results[(apply_f1, 1)]
        sn = results[(apply_f1, n_seeds)]
        print(f"  {f1_tag}: 1seed={s1:.2f}  {n_seeds}seed={sn:.2f}  (배깅효과={sn-s1:+.2f})")
    f1on_1, f1off_1 = results[(True, 1)], results[(False, 1)]
    f1on_n, f1off_n = results[(True, n_seeds)], results[(False, n_seeds)]
    print(f"  F1필터 효과(1seed 기준): {f1off_1-f1on_1:+.2f}")
    print(f"  F1필터 효과({n_seeds}seed 기준): {f1off_n-f1on_n:+.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--save-preds", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_isolation_grid(args.cutoff7, holdout, args.n_seeds, args.save_preds)


if __name__ == "__main__":
    main()
