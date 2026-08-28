# code/experiment_trackman_solo.py
"""트랙맨 데이터'만'으로 control_success를 예측하는 solo 진단 모델.

목적: CatBoost/MLP 블렌드에 트랙맨을 피처로 얹는 게 아니라, tier A 스타일의 투수x구종군
물리 지표(mean/std, code/trackman_pitcher_features.py::merge_asof_pitcher_std 재사용)만을
단독 입력으로 써서 "트랙맨에 control_success와 관련된 신호가 애초에 얼마나 있는지"를
CatBoost/MLP 없이 순수하게 측정한다. asof9key/situational-fingerprint 계열(핑거프린트로
train.csv 특정 행을 찾아 label을 끌어오는 방식)은 로컬-실전 skew가 구조적으로 확인돼
이미 3번 기각됐으므로 여기선 쓰지 않는다 — 대신 tier A가 원래 하던 "투수 정체성 크로스워크
+ 구종군별 물리 지표 집계"만으로 새 df_main(트랙맨 유래 컬럼만)을 만들어 그 자체로 학습한다.

새로 추가한 부분: 트랙맨 소스 데이터에도 F1 필터에 대응하는 필터를 적용한다.
trackman_history.csv에는 game_type 컬럼이 없지만, pitcher_team 값이 'MIN_' 접두사인
행(파생팀 코드, MIN_DOO/MIN_NCD/... <-> 1군 DOO_BEA/NC_DIN/...과 1:1 대응 확인됨)이
퓨처스(2군) 로그로 보여 이를 game_type=='F'의 근사 대응물로 사용한다.

듀얼레짐(cutoff7 / season==2023) 검증, CPU thread_count 캡(동시에 돌고 있는 다른 세션의
CatBoost 학습과 코어를 나눠 쓰기 위함), 단일 시드(순수 진단이라 7-seed 앙상블 불필요).

사용법: python -m code.experiment_trackman_solo
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.train import apply_f1_filter
from code.mlp_model import compute_bss
from code.trackman_pitcher_features import clean_trackman, merge_asof_pitcher_std

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
THREAD_COUNT = 2
ITERATIONS = 800
EARLY_STOPPING_ROUNDS = 40


def load_base():
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")

    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)

    is_futures = df_trm_clean["pitcher_team"].str.startswith("MIN_")
    before = len(df_trm_clean)
    f1_mask = is_futures & (df_trm_clean["season"] <= 2022)
    df_trm_clean = df_trm_clean[~f1_mask].reset_index(drop=True)
    print(f"  트랙맨 F1-equiv 필터(MIN_접두사 팀 x season<=2022 제거): "
          f"{before} -> {len(df_trm_clean)}행 ({f1_mask.sum()}행 제거)")
    print(f"[load_base] 완료 (경과 {time.time()-t0:.1f}s)")
    return train_df, pitcher_map, df_trm_clean


def run_regime(name, train_df, pitcher_map, df_trm_clean, holdout, train_mask_fn, val_mask_fn):
    t0 = time.time()
    df_main = apply_f1_filter(train_df.copy())
    df_main, feature_cols = merge_asof_pitcher_std(
        df_main, df_trm_clean, pitcher_map, tier="a", holdout=holdout
    )

    train_mask = train_mask_fn(df_main)
    val_mask = val_mask_fn(df_main)
    X_train, y_train = df_main.loc[train_mask, feature_cols], df_main.loc[train_mask, TARGET_COL]
    X_val, y_val = df_main.loc[val_mask, feature_cols], df_main.loc[val_mask, TARGET_COL]

    print(f"\n[{name}] n_features={len(feature_cols)} n_train={len(X_train)} n_val={len(X_val)}")

    model = CatBoostClassifier(
        iterations=ITERATIONS,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="BrierScore",
        random_seed=42,
        thread_count=THREAD_COUNT,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    preds = model.predict_proba(X_val)[:, 1]
    brier, bss, score = compute_bss(preds, y_val.values)

    # 트랙맨 물리량이 전혀 신호가 없을 때의 참조선: 모든 val 행에 train 성공률 상수 예측
    const_pred = np.full(len(y_val), y_train.mean())
    const_brier, const_bss, const_score = compute_bss(const_pred, y_val.values)

    print(f"[{name}] best_iter={model.get_best_iteration()} | "
          f"트랙맨-only score={score:.2f} (BSS={bss:.5f}) | "
          f"상수예측 참조선 score={const_score:.2f} | 경과 {time.time()-t0:.1f}s")

    importances = pd.Series(model.get_feature_importance(train_pool), index=feature_cols)
    print(f"[{name}] top-5 피처 중요도:\n{importances.sort_values(ascending=False).head(5)}")

    return dict(name=name, score=score, const_score=const_score, n_val=len(X_val),
                n_features=len(feature_cols), row_id=df_main.loc[val_mask, "row_id"].values,
                preds=preds, y_val=y_val.values)


def main():
    train_df, pitcher_map, df_trm_clean = load_base()

    results = []

    # 레짐 1: cutoff7 (프로덕션 스플릿) — holdout=2024로 own-season 트랙맨 클램프
    results.append(run_regime(
        "cutoff7", train_df, pitcher_map, df_trm_clean, holdout=2024,
        train_mask_fn=lambda df: (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7)),
        val_mask_fn=lambda df: (df["season"] == 2024) & (df["game_month"] >= 7),
    ))

    # 레짐 2: season==2023 홀드아웃 (pre-ABS 감사용 레짐) — holdout=2023
    results.append(run_regime(
        "season2023", train_df, pitcher_map, df_trm_clean, holdout=2023,
        train_mask_fn=lambda df: df["season"] < 2023,
        val_mask_fn=lambda df: df["season"] == 2023,
    ))

    print("\n" + "=" * 60)
    print(f"{'레짐':<12}{'n_val':>10}{'트랙맨-only':>14}{'상수 참조선':>14}{'delta':>10}")
    for r in results:
        delta = r["score"] - r["const_score"]
        print(f"{r['name']:<12}{r['n_val']:>10}{r['score']:>14.2f}{r['const_score']:>14.2f}{delta:>+10.2f}")
    print("=" * 60)

    os.makedirs("./open/temp/experiment_trackman_solo", exist_ok=True)
    for r in results:
        np.savez(
            f"./open/temp/experiment_trackman_solo/{r['name']}_preds.npz",
            row_id=r["row_id"], preds=r["preds"], y_val=r["y_val"],
        )
    print("[main] 예측 저장 완료 (상관관계 후속 체크용): ./open/temp/experiment_trackman_solo/")


if __name__ == "__main__":
    main()
