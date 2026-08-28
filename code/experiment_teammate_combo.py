# code/experiment_teammate_combo.py
"""팀원(조유담) 1059.72 레시피의 두 구조적 차이 — CatBoost 하이퍼파라미터
(`experiment_teammate_catboost_hparams.py`, cutoff7 solo -13.14)와 트랙맨64
(`experiment_teammate_trackman64.py`, cutoff7 solo -17.94, val 매칭률 0.0%로
실전에서 100% 상수임이 확인됨) — 를 **함께** 넣었을 때만 시너지가 나는지 확인한다.
둘 다 단독으로는 cutoff7에서 손해였으므로, 상호작용이 없다면 조합도 마이너스일
것으로 예상된다(개별 손해가 상쇄되기보다 누적될 가능성이 더 높음) — 그래도 실제로
확인해야 다음 결론(팀원 실전 우위의 원인은 이 둘이 아니다)을 자신 있게 낼 수 있다.

사용법:
  python -m code.experiment_teammate_combo --cutoff7
  python -m code.experiment_teammate_combo --holdout 2023
"""
import argparse
import time

import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, TARGET_COL
from code.trackman_pitcher_features import clean_trackman
from code.experiment_teammate_catboost_hparams import TEAMMATE_PARAMS, TEAMMATE_ITERATIONS
from code.experiment_teammate_trackman64 import merge_trackman64, TM64_COLS

DATA_DIR = "./open/data"


def train_one(overrides, iterations, X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params.update(overrides)
    params["iterations"] = iterations
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (팀원 하이퍼파라미터+트랙맨64 조합 테스트) ===\n{'='*70}")

    train_split, val_split, _mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    y_train = train_split[TARGET_COL].values
    y_val = val_split[TARGET_COL].values

    df_trm_full = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    trm_holdout = 2024 if cutoff7 else holdout
    train_tm, train_match_rate = merge_trackman64(train_split, df_trm_clean, holdout=trm_holdout)
    val_tm, val_match_rate = merge_trackman64(val_split, df_trm_clean, holdout=trm_holdout)
    print(f"  트랙맨64 매칭률: train={train_match_rate:.1%}, val={val_match_rate:.1%}")

    cat_feature_cols_tm = cat_feature_cols + TM64_COLS
    X_train_base = train_split[cat_feature_cols]
    X_val_base = val_split[cat_feature_cols]
    X_train_tm = train_tm[cat_feature_cols_tm]
    X_val_tm = val_tm[cat_feature_cols_tm]

    configs = [
        ("우리 baseline (하이퍼파라미터=우리, 트랙맨64=없음)", {}, 1500, X_train_base, X_val_base),
        ("하이퍼파라미터만 팀원", TEAMMATE_PARAMS, TEAMMATE_ITERATIONS, X_train_base, X_val_base),
        ("트랙맨64만 추가", {}, 1500, X_train_tm, X_val_tm),
        ("하이퍼파라미터+트랙맨64 둘 다 팀원", TEAMMATE_PARAMS, TEAMMATE_ITERATIONS, X_train_tm, X_val_tm),
    ]

    results = {}
    for name, overrides, iterations, X_train, X_val in configs:
        t0 = time.time()
        preds, best_iter = train_one(overrides, iterations, X_train, y_train, X_val, y_val)
        score = compute_bss(preds, y_val)[2]
        results[name] = score
        print(f"  [{name}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s)")

    base_score = results[configs[0][0]]
    print(f"\n  --- baseline({base_score:.2f}) 대비 delta ---")
    for name, score in results.items():
        if name == configs[0][0]:
            continue
        print(f"  {name}: delta={score-base_score:+.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(args.cutoff7, holdout)


if __name__ == "__main__":
    main()
