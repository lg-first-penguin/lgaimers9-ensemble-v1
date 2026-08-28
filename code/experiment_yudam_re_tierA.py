# code/experiment_yudam_re_tierA.py
"""[재검증 #1] 트랙맨 tier A (투수 크로스워크 × 구종군별 물리 지표 mean/std) 를
raw-concat MLP 베이스라인(유담님 레시피, PLE 없음) 위에서 MLP 전용으로 다시 측정한다.

배경: tier A 는 2026-08-18 실전 리더보드 950.81 (−31.41) 로 기각됐다. 당시 로컬
dual-regime 은 애매했고(cutoff7 은 tier A 선호, season2023 은 제거 선호), 실전 회귀의
유력한 원인 가설은 "PLE × 2025 추론 시 상황조인 붕괴 → 64개 물리량 컬럼이 한 quantile
bin 으로 degenerate" 였다. 현재 프로덕션 MLP 는 PLE 를 안 쓰므로(bin_edges=None) 그
실패 모드가 사라졌을 수 있다 — 사실이면 tier A 가 최소 중립~소폭 + 로 돌아온다.

천장 주의: control_success 는 최종 공 위치 vs 존 + 포수 요구방향으로 정의되고 트랙맨
30개 컬럼은 전부 릴리스 물리량이라 구조적 상한이 있다
(memory: control_success_definition_vs_trackman_columns_gap). 큰 승리는 기대하지 말 것.

사용법:
  python -m code.experiment_yudam_re_tierA --regime cutoff7            # MLP/CB 3-seed 스크리닝(기본)
  python -m code.experiment_yudam_re_tierA --regime cutoff7 --quick   # 1-seed (가장 가벼움)
  python -m code.experiment_yudam_re_tierA --regime 2023
"""
import argparse
import os

import pandas as pd

from code.experiment_yudam_common import compare, DATA_DIR
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS
from code.trackman_pitcher_features import clean_trackman, add_all_tiers

_STATE = {}


def add_tierA(df_full, holdout):
    """tier A 물리 지표 컬럼을 df_full 에 추가하고 (df, exclude_from_cat, exclude_from_mlp)
    를 돌려준다. tier A 는 MLP 전용이므로 CatBoost feature 에서는 제외한다."""
    if "trm_clean" not in _STATE:
        df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
        _STATE["trm_clean"] = clean_trackman(df_trm)
        _STATE["pitcher_map"] = pd.read_csv("./open/temp/pitcher_map.csv")
        del df_trm
    df_out, tier_cols = add_all_tiers(df_full, _STATE["trm_clean"], _STATE["pitcher_map"], ["a"], holdout=holdout)
    cols = tier_cols["a"]
    print(f"[tier A] {len(cols)}개 컬럼 추가 -> MLP 전용", flush=True)
    return df_out, set(cols), set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="cutoff7", choices=["cutoff7", "2023"])
    ap.add_argument("--quick", action="store_true", help="MLP/CatBoost 1-seed")
    ap.add_argument("--seeds", type=int, default=3, help="스크리닝 시드 수 (기본 3)")
    args = ap.parse_args()

    n = 1 if args.quick else args.seeds
    mlp_seeds = YUDAM_ENSEMBLE_SEEDS[:n]
    cb_seeds = YUDAM_CATBOOST_SEEDS[:n]
    compare(regime=args.regime, add_features_fn=add_tierA,
            mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label="tierA->MLP")


if __name__ == "__main__":
    main()
