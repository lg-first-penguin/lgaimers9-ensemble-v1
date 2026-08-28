# code/experiment_yudam_re_rowlevel.py
"""[재검증 #3] row-level pseudo-label 조건화 트랙맨 피처(86.73% 매칭, pseudo-label로
투수×구종군 성공/실패 조건 물리량 mean/std 압축)를 raw-concat MLP 베이스라인 위에서
MLP 전용으로 다시 측정한다.

배경: 이 피처의 "좁힌 2컬럼"(induced_vert_break × breaking) 변형은 PLE 시절 3-seed에서
양쪽 레짐 blend + 로 보였으나 7-seed 재검증에서 부호반전으로 기각됐다
(memory: trackman_rowlevel_pseudolabel_rejected — cutoff7 blend +11.21→−7.12,
season2023 +15.78→−0.19). tier A(#1)와 달리 메커니즘이 다르다 — 크로스워크 물리량
평균이 아니라 pseudo-label 조건부 분포. PLE 제거로 계산이 달라질 여지가 남아있어
한 번은 확인한다.

사용법:
  python -m code.experiment_yudam_re_rowlevel --regime cutoff7           # 좁힌 2컬럼, 3-seed
  python -m code.experiment_yudam_re_rowlevel --regime cutoff7 --full    # 64컬럼 전체
  python -m code.experiment_yudam_re_rowlevel --regime cutoff7 --quick
"""
import argparse

from code.experiment_yudam_common import compare
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS
from code.trackman_conditioned_features_v2 import merge_trackman_conditioned_v2

_NARROW = dict(metrics=["induced_vert_break"], pitch_types=["breaking"])


def make_adder(full):
    kw = {} if full else _NARROW

    def add_rowlevel(df_full, holdout):
        df_out, cond_cols = merge_trackman_conditioned_v2(df_full, holdout=holdout, **kw)
        print(f"[rowlevel{'(64)' if full else '(narrow2)'}] {len(cond_cols)}개 컬럼 추가 -> MLP 전용", flush=True)
        return df_out, set(cond_cols), set()

    return add_rowlevel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="cutoff7", choices=["cutoff7", "2023"])
    ap.add_argument("--full", action="store_true", help="64컬럼 전체 (기본: 좁힌 2컬럼)")
    ap.add_argument("--quick", action="store_true", help="MLP/CatBoost 1-seed")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    n = 1 if args.quick else args.seeds
    compare(regime=args.regime, add_features_fn=make_adder(args.full),
            mlp_seeds=YUDAM_ENSEMBLE_SEEDS[:n], cb_seeds=YUDAM_CATBOOST_SEEDS[:n],
            label=f"rowlevel{'64' if args.full else '-narrow2'}->MLP")


if __name__ == "__main__":
    main()
