# code/experiment_abs_regime.py
"""KBO ABS(자동 볼 판정 시스템) 도입(2024시즌부터, 2020~2023엔 퓨처스리그에서만 시범
운영)이 2019~2023과 2024 사이의 진짜 레짐 시프트(regime shift)일 수 있다는 가설을
검증하는 실험. 근거: 시즌별 control_success 성공률이 2019(0.5647)->2024(0.4861)로
단조 하락하며, 특히 2022->2023(-2.9%p)이 가장 크고 2023->2024(-1.4%p)도 이어짐 —
§18의 F1 필터(2023년 game_type='F' 관계 역전)와 겹치는 시점대이며, 사용자의 별도
앙상블 실험에서도 CatBoost 단독 성능이 2023(751)->2024(642)로 급락했다는 보고가 있음.

가설이 맞다면, "2019~2023(구 체제) 데이터가 아무리 많아도 실제 평가(2025시즌,
2024와 같은 ABS 체제)와는 체제 자체가 다르다"는 뜻이 되어, 학습에 2024(신 체제)
데이터를 일부라도 포함시키거나 가중치를 주는 게 데이터 양보다 더 중요할 수 있다.

현재 프로덕션 분할(학습 2019~2023 / 검증 2024 전체)로는 이 가설을 못 본다 — 2024
데이터가 학습에 전혀 없기 때문. 그래서 2024를 시간순으로 쪼개 아래 3가지 설정을
**동일한 검증셋**(2024년 9~10월 — 월별 분해에서 성능이 가장 크게 무너졌던 구간)으로
공정 비교한다:

  A (baseline, 현재 방식과 동일한 학습 레짐): 2019~2023만 학습
  B (2024 앞부분 포함): 2019~2023 + 2024 3~8월 학습 (2024 데이터 노출, 가중치 없음)
  C (2024 가중치 강화): B와 동일한 학습행 구성이지만 2024 3~8월 행에 sample_weight 부여

CatBoost만으로 먼저 빠르게 스크리닝한다(F1 필터는 세 설정 모두 동일하게 적용 —
season<=2022인 F행만 제거하므로 2024 데이터를 학습에 넣어도 영향 없음, §29에서
확인된 "F1 필터가 학습 구간의 F행을 통째로 지우는" 함정과는 무관).

사용법:
  python -m code.experiment_abs_regime --step run
  python -m code.experiment_abs_regime --step run --weight 5.0
"""
import argparse
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.train import apply_f1_filter, add_engineered_features
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
VAL_MONTH_CUTOFF = 9  # 기본값: 2024년 9~10월을 검증셋으로 고정 (월별 분해에서 가장 나빴던 구간).
# build_splits(val_month_cutoff=...)로 다른 컷오프도 검증 가능 (예: 7 -> 2024년 7~10월 검증).


def compute_score(preds, y):
    r = y.mean()
    brier = ((preds - y) ** 2).mean()
    baseline = r * (1 - r)
    return max(0, 100000 * (1 - brier / baseline))


def build_splits(val_month_cutoff=VAL_MONTH_CUTOFF):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    full_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    pre2024_mask = full_df["season"] < 2024
    is2024_head_mask = (full_df["season"] == 2024) & (full_df["game_month"] < val_month_cutoff)
    is2024_tail_mask = (full_df["season"] == 2024) & (full_df["game_month"] >= val_month_cutoff)

    league_mean = full_df.loc[pre2024_mask, TARGET_COL].mean()
    fdf = add_engineered_features(full_df, league_mean)
    for c in CAT_FEATURES:
        fdf[c] = fdf[c].astype(str)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in fdf.columns if c not in drop_cols]

    pre2024 = fdf.loc[pre2024_mask, features + [TARGET_COL]].reset_index(drop=True)
    pre2024 = apply_f1_filter(pre2024)  # season<=2022 F행만 제거, 2024 데이터엔 영향 없음
    head2024 = fdf.loc[is2024_head_mask, features + [TARGET_COL]].reset_index(drop=True)
    tail2024 = fdf.loc[is2024_tail_mask, features + [TARGET_COL]].reset_index(drop=True)

    print(f"[split] 2019~2023(F1필터 적용): {len(pre2024)}행 | 2024 3~{val_month_cutoff-1}월(head): {len(head2024)}행 "
          f"| 2024 {val_month_cutoff}~10월(검증, tail): {len(tail2024)}행")
    return pre2024, head2024, tail2024, features


def train_and_score(train_df, features, val_df, weight_2024=None, pre_n=None, verbose_label=""):
    X_tr, y_tr = train_df[features], train_df[TARGET_COL].values
    X_va, y_va = val_df[features], val_df[TARGET_COL].values

    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    params["verbose"] = False

    sample_weight = None
    if weight_2024 is not None:
        sample_weight = np.ones(len(train_df))
        sample_weight[pre_n:] = weight_2024  # 2024 head 행들(뒤에 concat됨)에만 가중치

    train_pool = Pool(data=X_tr, label=y_tr, cat_features=CAT_FEATURES, weight=sample_weight)
    val_pool = Pool(data=X_va, label=y_va, cat_features=CAT_FEATURES)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    bi = int(model.get_best_iteration())
    preds = model.predict_proba(X_va)[:, 1]
    score = compute_score(preds, y_va)
    print(f"[{verbose_label}] n_train={len(train_df)} | best_iter={bi} | score={score:.2f}")
    return score


def step_run(weight_2024, val_month_cutoff=VAL_MONTH_CUTOFF):
    pre2024, head2024, tail2024, features = build_splits(val_month_cutoff=val_month_cutoff)

    # A: 2019~2023만 학습
    score_a = train_and_score(pre2024, features, tail2024, verbose_label="A (2019~2023만)")

    # B: 2019~2023 + 2024 head, 가중치 없음
    train_b = pd.concat([pre2024, head2024], ignore_index=True)
    score_b = train_and_score(train_b, features, tail2024, verbose_label="B (+2024 head, 가중치無)")

    # C: 2019~2023 + 2024 head, 2024 head에 가중치
    score_c = train_and_score(train_b, features, tail2024, weight_2024=weight_2024, pre_n=len(pre2024),
                               verbose_label=f"C (+2024 head, weight={weight_2024})")

    print(f"\n=== 결과 요약 (검증: 2024년 {val_month_cutoff}~10월 고정) ===")
    print(f"{'설정':<35} {'score':>10} {'vs A':>10}")
    print(f"{'A (2019~2023만, 현재 방식과 동일 레짐)':<35} {score_a:>10.2f} {'-':>10}")
    print(f"{'B (+2024 head, 가중치 없음)':<35} {score_b:>10.2f} {score_b-score_a:>+10.2f}")
    print(f"{'C (+2024 head, weight='+str(weight_2024)+')':<35} {score_c:>10.2f} {score_c-score_a:>+10.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["run"])
    parser.add_argument("--weight", type=float, default=5.0, help="설정 C에서 2024 head 행에 줄 가중치 배수")
    parser.add_argument("--val-cutoff", type=int, default=VAL_MONTH_CUTOFF, help="검증 시작 월 (예: 7 -> 2024년 7~10월 검증)")
    args = parser.parse_args()
    step_run(weight_2024=args.weight, val_month_cutoff=args.val_cutoff)


if __name__ == "__main__":
    main()
