# code/thirdmodel_common.py
"""2026-08-21 세션: §50에서 프로젝트 전체적으로 종결됐던 "3rd-모델 스태킹" 라인을
사용자 판단으로 재개하면서 쓰는 공용 split/전처리 유틸리티.

**왜 새로 만드나 (기존 build_split들을 못 쓰는 이유)**: 이 프로젝트에 이미 3개의
"3rd 모델용 build_split"이 있다(`code/experiment_attention.py`,
`code/experiment_tabnet_correlation.py`가 재사용하는
`code/experiment_residual_correction_9_10.py::build_split`) — 그런데 셋 다 지금
프로덕션(real 1041.40, TE-residual §45 + tier A 제거 이후)보다 오래된 스냅샷이다:
  - `experiment_attention.py::build_split`은 `add_engineered_features`만 부르고
    `apply_te_residual_features`를 아예 호출하지 않는다 — TE-residual 피처(실전
    +13.86 확인, `code/train.py::TE_RESIDUAL_COLS`) 도입 이전 코드.
  - `experiment_residual_correction_9_10.py::build_split`은 `TIER_FEED = {"a": "mlp"}`로
    트랙맨 tier A가 아직 켜져 있다 — tier A는 2026-08-18 실전 리더보드 950.81(-31.41)로
    확인되어 프로덕션에서 완전히 빠졌다(`code/train.py::TRACKMAN_TIER_FEED = {}`).
  - 즉 §22-24(FT-Transformer/ExcelFormer)와 §42/§50(TabNet 등)의 "3rd 모델은 항상
    CatBoost/MLP와 상관 0.85+" 결론은 전부 **그 당시 피처셋**(TE-residual 없음,
    tier A 있음 또는 asof-only 서브셋) 기준이었다 — 지금 피처셋으로 검증된 적은
    없다. 이번 재개의 실질적 근거이기도 하다(§50의 "새로운 메커니즘 없이 재시도
    금지" 원칙과 별개로, 피처셋 자체가 바뀌었다는 사실 근거).

이 파일의 `build_split`은 `code/train.py::main()`을 한 글자도 다르지 않게 그대로
재현한다(F1 필터, cutoff7 스플릿, add_engineered_features/season-progression,
TE-residual, tier A 제거 상태의 add_all_tiers, coarse pitchmix) — 다만 모델 학습은
하지 않고 스플릿된 DataFrame과 컬럼 리스트만 반환한다.

3rd 모델(TabNet/FT-Transformer/GrowNet 등)에는 기존 `experiment_attention.py`/
`experiment_tabnet_correlation.py`와 동일한 관례를 따라 **MLP와 같은 피처셋**
(CAT_COLS + mlp_num_cols)을 준다 — CatBoost 전용 피처(TE-residual, coarse
pitchmix)는 주지 않는다(3번째 축이 어느 한쪽에 유리하지 않도록).
"""
import os

import numpy as np
import pandas as pd

from code.mlp_model import CAT_COLS
from code.train import (
    TE_RESIDUAL_COLS,
    TRACKMAN_TIER_FEED,
    add_engineered_features,
    apply_f1_filter,
    apply_te_residual_features,
    apply_pair_matchup_rowlevel,
    build_pair_matchup_lookup,
    apply_pair_matchup_static,
)
from code.trackman_pitcher_features import add_all_tiers, clean_trackman, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"


def build_split(cutoff7=True, holdout=2024, apply_f1=True):
    """`code/train.py::main()`과 동일한 절차. cutoff7=True면 프로덕션 스플릿
    (season<2024 | (season==2024 & game_month<7) 학습, season==2024 & game_month>=7
    검증). cutoff7=False면 예전 관례(season<holdout 학습, season==holdout 검증,
    `--holdout {2023,2024}` 수동 감사용)로 대체.

    반환: train_split, val_split, mlp_num_cols(3rd 모델/MLP용), cat_feature_cols
    (CatBoost용, TE-residual+pitchmix 포함), cat_val_raw/train_raw는 호출부에서
    train_split[cat_feature_cols]로 바로 슬라이스."""
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
        trk_holdout = 2024
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout
        trk_holdout = holdout

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=trk_holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]

    df = merge_coarse_pitchmix(df, df_trm, holdout=trk_holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in df.columns if c not in drop_cols]
    mlp_num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols
                    and c not in trk_mlp_cols]
    cat_feature_cols = [c for c in features if c not in trk_mlp_cols]

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    if apply_f1:
        before = len(train_split)
        train_split = apply_f1_filter(train_split)
        print(f"[F1 필터] {before} -> {len(train_split)}행")

    te_prior = train_split[TARGET_COL].mean()
    train_split = apply_te_residual_features(train_split, train_split, te_prior)
    val_split = apply_te_residual_features(train_split, val_split, te_prior)
    cat_feature_cols = cat_feature_cols + TE_RESIDUAL_COLS

    # 페어 매치업 5개(->CatBoost 전용) — TE-residual과 동일하게 train_split만 causal
    # source로 쓰고 val_split엔 그 source로 얼린 정적 lookup을 적용한다
    # (code/train.py::apply_pair_matchup_rowlevel 문서의 "첫 채택판의 버그" 참고). 실전
    # 리더보드 1028.75(-12.65)로 기각된 첫 채택판의 버그를 고친 버전 — 재검증 전까지는
    # feed 비활성 유지(cat_feature_cols에 추가하지 않음).
    train_split = apply_pair_matchup_rowlevel(train_split)
    pair_lookup = build_pair_matchup_lookup(train_split)
    val_split = apply_pair_matchup_static(val_split, pair_lookup)

    print(f"[thirdmodel_common.build_split] train={len(train_split)} val={len(val_split)} "
          f"mlp_num_cols={len(mlp_num_cols)} cat_feature_cols={len(cat_feature_cols)}")
    return train_split, val_split, mlp_num_cols, cat_feature_cols
