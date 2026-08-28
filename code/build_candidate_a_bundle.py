# code/build_candidate_a_bundle.py
"""candidate A(손수연 레시피+TrackA+F1) 실전 제출용 최종 번들 생성.

사용자 승인 배경: `code/experiment_candidate_dualregime.py`(EXPERIMENTS.md §87)에서
candidate A가 cutoff7 blend +10.63(3/3승)/season2023 blend +23.93(3/3승)로 양쪽 레짐
전부 견고하게 이겨 이 프로젝트 교차팀 비교 역사상 최강 신호로 확인됨. 사용자가 롤링
검증을 생략하고 바로 제출하기로 결정.

절차 (dopip.py의 "레퍼런스 검증 -> 전체 재학습" 관례를 그대로 따름):
1. cutoff7 split으로 candidate A CatBoost(3-seed, F1필터 적용)를 학습해 기존 프로덕션
   MLP(재학습 없이 순수 추론 재사용, `open/reference/best_model.pkl`의 mlp_bundle)와
   블렌드, 메타모델 가중치(w_cat/w_mlp/intercept)를 이 시점에 확정한다 -- 전체 재학습
   단계엔 val이 없으므로 dopip.py Full Retrain 관례와 동일하게 여기서 고정한 가중치를
   그대로 재사용한다.
2. 전체 train.csv(모든 시즌, 홀드아웃 없음)로 candidate A CatBoost를 3-seed 재학습
   (F1필터 적용)한다. MLP는 손대지 않고 기존 submit/model/final_retained_model.pkl의
   mlp_bundle(이미 전체 데이터로 재학습된 20-seed 앙상블)을 그대로 재사용한다 --
   candidate A가 검증한 그대로("우리 자신의 MLP, 재학습 없이 재사용").
3. candidate A 전용 정적 추론 아티팩트(TE-residual 소스/prior, 리그평균 dict, 트랙맨
   std5/gap4 매치테이블 사본)와 최종 번들을 submit_candidate_a/model/ 에 저장한다.

사용법: python -m code.build_candidate_a_bundle
"""
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.mlp_model import compute_bss, predict_bundle, get_device
from code.blend_model import fit_meta_model
from code.train import (
    add_engineered_features, apply_same_hand, apply_te_residual_features, apply_f1_filter,
)
from code.experiment_sooyun_recipe import (
    build_sooyun_features, SOOYUN_CATBOOST_PARAMS, SOOYUN_ITERATIONS, SOOYUN_CAT_COLS,
    check_memory_or_abort, DATA_DIR, TARGET, SOOYUN_DIR,
)

# 사용자 지시(2026-08-27): 3-seed([42,123,7], 이 repo의 dual-regime 검증에 쓴 시드)
# 대신 손수연 본인의 실제 5-seed 배깅(그의 .cbm 파일 5개: model_seed{42,123,777,999,2024}.cbm,
# 실전 CatBoost 솔로 1007.52)과 정확히 동일한 시드로 맞춘다 -- 그의 실제 레시피에 더 가깝다.
SEEDS = [42, 123, 777, 999, 2024]
OUT_DIR = "./submit_candidate_a"
MODEL_DIR = os.path.join(OUT_DIR, "model")

TE_SOURCE_COLS = ["pitcher_id", "batter_id", "balls_before", "strikes_before",
                   "batter_hand", "num_runners_on", "inning", "season", TARGET]


def prep_sooyun_for_catboost(df_raw, row_mask=None):
    """build_sooyun_features + TE-residual(no_both 이전에 적용) + (옵션) F1필터.
    row_mask=None이면 전체 df_raw가 곧 학습 파티션(전체 재학습)."""
    sooyun_full = build_sooyun_features(df_raw)
    sooyun_full = sooyun_full.merge(df_raw[["row_id", "pitcher_id", "batter_id"]], on="row_id", how="left")
    if row_mask is None:
        train_sy = sooyun_full
        val_sy = None
    else:
        train_mask, val_mask = row_mask
        train_sy = sooyun_full.loc[train_mask].reset_index(drop=True)
        val_sy = sooyun_full.loc[val_mask].reset_index(drop=True)
    del sooyun_full
    for c in SOOYUN_CAT_COLS:
        train_sy[c] = train_sy[c].astype(str)
        if val_sy is not None:
            val_sy[c] = val_sy[c].astype(str)

    te_prior = train_sy[TARGET].mean()
    te_source = train_sy.copy()
    train_sy = apply_te_residual_features(te_source, train_sy, te_prior)
    if val_sy is not None:
        val_sy = apply_te_residual_features(te_source, val_sy, te_prior)

    return train_sy, val_sy, te_source, te_prior


def step1_fit_meta_weights(df_raw, ref_mlp_bundle, device):
    """cutoff7 split으로 candidate A CatBoost(3-seed) + 기존 프로덕션 MLP를 블렌드해
    메타모델 가중치를 확정한다."""
    print("\n" + "=" * 70 + "\n[1단계] cutoff7으로 메타모델 가중치 확정\n" + "=" * 70, flush=True)
    train_mask = (df_raw["season"] < 2024) | ((df_raw["season"] == 2024) & (df_raw["game_month"] < 7))
    val_mask = (df_raw["season"] == 2024) & (df_raw["game_month"] >= 7)

    check_memory_or_abort("1단계 candidate A 피처 빌드 전")
    train_sy, val_sy, _, _ = prep_sooyun_for_catboost(df_raw, row_mask=(train_mask, val_mask))
    train_sy = train_sy.drop(columns=["pitcher_id", "batter_id"])
    val_sy = val_sy.drop(columns=["pitcher_id", "batter_id"])
    before = len(train_sy)
    train_sy = apply_f1_filter(train_sy)
    print(f"  [F1필터] {before} -> {len(train_sy)}행", flush=True)

    drop_cols = ["row_id", TARGET]
    feature_cols = [c for c in train_sy.columns if c not in drop_cols]
    y_val = val_sy[TARGET].values

    cat_preds_list = []
    for seed in SEEDS:
        check_memory_or_abort(f"1단계 CatBoost seed={seed}")
        t0 = time.time()
        model = CatBoostClassifier(**SOOYUN_CATBOOST_PARAMS, iterations=SOOYUN_ITERATIONS, random_seed=seed)
        train_pool = Pool(train_sy[feature_cols], train_sy[TARGET], cat_features=SOOYUN_CAT_COLS)
        model.fit(train_pool)
        preds = model.predict_proba(val_sy[feature_cols])[:, 1]
        cat_preds_list.append(preds)
        _, _, score = compute_bss(preds, y_val)
        print(f"  [seed={seed}] cat_solo={score:.2f} ({time.time()-t0:.1f}s)", flush=True)
    cat_preds = np.mean(cat_preds_list, axis=0)
    _, _, cat_avg_score = compute_bss(cat_preds, y_val)
    print(f"  [3-seed 평균] cat_solo={cat_avg_score:.2f}", flush=True)

    # 기존 프로덕션 MLP -- 재학습 없이 순수 추론 재사용 (candidate A 검증과 동일)
    df_ours = df_raw.copy()
    df_ours["top_bottom"] = df_ours["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    league_mean_cutoff = df_ours.loc[train_mask, TARGET].mean()
    df_ours_eng = add_engineered_features(df_ours.copy(), league_mean_cutoff)
    df_ours_eng = apply_same_hand(df_ours_eng)
    val_ours = df_ours_eng.loc[val_mask].reset_index(drop=True)
    mlp_preds = predict_bundle(ref_mlp_bundle, val_ours, device=device)
    _, _, mlp_score = compute_bss(mlp_preds, y_val)
    print(f"  [기존 MLP, 재학습없음] mlp_solo={mlp_score:.2f}", flush=True)

    w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val)
    print(f"  [메타모델] w_cat={w_cat:.4f} w_mlp={w_mlp:.4f} intercept={intercept:.4f} | blend={blend_score:.2f}", flush=True)
    return {"w_cat": w_cat, "w_mlp": w_mlp, "intercept": intercept}


def step2_full_retrain(df_raw):
    """전체 train.csv(홀드아웃 없음)로 candidate A CatBoost 3-seed 재학습."""
    print("\n" + "=" * 70 + "\n[2단계] 전체 데이터로 candidate A CatBoost 재학습\n" + "=" * 70, flush=True)
    check_memory_or_abort("2단계 candidate A 피처 빌드 전")
    train_sy, _, te_source, te_prior = prep_sooyun_for_catboost(df_raw, row_mask=None)
    te_source_slim = te_source[TE_SOURCE_COLS].copy()
    train_sy = train_sy.drop(columns=["pitcher_id", "batter_id"])
    before = len(train_sy)
    train_sy = apply_f1_filter(train_sy)
    print(f"  [F1필터] {before} -> {len(train_sy)}행", flush=True)

    drop_cols = ["row_id", TARGET]
    feature_cols = [c for c in train_sy.columns if c not in drop_cols]
    print(f"  CatBoost 피처={len(feature_cols)}개 (categorical {len(SOOYUN_CAT_COLS)}개)", flush=True)

    models = []
    for seed in SEEDS:
        check_memory_or_abort(f"2단계 CatBoost seed={seed}")
        t0 = time.time()
        model = CatBoostClassifier(**SOOYUN_CATBOOST_PARAMS, iterations=SOOYUN_ITERATIONS, random_seed=seed)
        train_pool = Pool(train_sy[feature_cols], train_sy[TARGET], cat_features=SOOYUN_CAT_COLS)
        model.fit(train_pool)
        models.append(model)
        print(f"  [seed={seed}] 학습 완료 ({time.time()-t0:.1f}s)", flush=True)

    # 리그평균 dict -- build_sooyun_features가 df_raw(전체) 기준으로 내부에서 쓰는 것과
    # 동일하게 전체 df_raw로 재계산 (season -> mean, 소규모라 CSV로 정적 동봉)
    league_mean_dict = df_raw.groupby("season")[TARGET].mean().to_dict()

    return models, feature_cols, te_source_slim, te_prior, league_mean_dict


def main():
    print("[데이터 로드]", flush=True)
    t0 = time.time()
    df_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df_raw = df_raw.dropna(subset=[TARGET]).reset_index(drop=True)
    for c in df_raw.select_dtypes(include="float64").columns:
        df_raw[c] = df_raw[c].astype(np.float32)
    for c in df_raw.select_dtypes(include="int64").columns:
        if c not in ("row_id",):
            df_raw[c] = df_raw[c].astype(np.int32)
    print(f"train.csv 로드 완료: {len(df_raw)}행 ({time.time()-t0:.1f}s)", flush=True)
    check_memory_or_abort("train.csv 로드 직후")

    # 1단계(메타가중치 확정)는 cutoff7 val을 "못 본" 레퍼런스 MLP(open/reference/best_model.pkl,
    # season<2024|2024년 3~6월까지만 학습)를 써야 한다 -- 전체 데이터로 재학습된
    # submit/model/final_retained_model.pkl의 MLP를 여기 쓰면 그 MLP가 이미 cutoff7 val
    # 구간(2024 7~10월)까지 학습에 포함해서 본 상태라 리크(blend=2080이라는 비정상 점수 +
    # 깨진 음수 계수로 실제 발견됨, 2026-08-27). 2단계(최종 번들)는 반대로 전체 재학습된
    # MLP를 쓰는 게 맞다(dopip.py의 "레퍼런스=검증용, 전체재학습=실제 제출용" 관례와 동일).
    with open("./open/reference/best_model.pkl", "rb") as f:
        ref_bundle = pickle.load(f)
    ref_mlp_bundle = ref_bundle["mlp_bundle"]

    with open("./submit/model/final_retained_model.pkl", "rb") as f:
        prod_bundle = pickle.load(f)
    full_mlp_bundle = prod_bundle["mlp_bundle"]

    device = get_device()
    print(f"device={device}, 레퍼런스(cutoff7 val 미포함) MLP 멤버 수={len(ref_mlp_bundle['members'])}, "
          f"전체재학습 MLP 멤버 수={len(full_mlp_bundle['members'])}", flush=True)

    meta = step1_fit_meta_weights(df_raw, ref_mlp_bundle, device)
    models, feature_cols, te_source_slim, te_prior, league_mean_dict = step2_full_retrain(df_raw)
    mlp_bundle = full_mlp_bundle  # 최종 번들엔 전체재학습 MLP 사용

    os.makedirs(MODEL_DIR, exist_ok=True)

    final_bundle = {
        "catboost_models": models,
        "mlp_bundle": mlp_bundle,
        "meta_model": meta,
        "cat_feature_cols": feature_cols,
        "cat_cols_categorical": SOOYUN_CAT_COLS,
        "te_prior": float(te_prior),
        "league_mean_dict": {int(k): float(v) for k, v in league_mean_dict.items()},
        "recipe": "candidate_a_sooyun_tracka_f1",
    }
    out_path = os.path.join(MODEL_DIR, "final_retained_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"[저장] {out_path}", flush=True)

    te_source_slim.to_csv(os.path.join(MODEL_DIR, "te_source.csv"), index=False)
    print(f"[저장] {MODEL_DIR}/te_source.csv ({len(te_source_slim)}행)", flush=True)

    shutil.copy(f"{SOOYUN_DIR}/model/trackman_match_table.csv", os.path.join(MODEL_DIR, "trackman_match_table.csv"))
    shutil.copy(f"{SOOYUN_DIR}/model/trackman_match_table_gap.csv", os.path.join(MODEL_DIR, "trackman_match_table_gap.csv"))
    shutil.copy("./submit/model/season_end_lookup.csv", os.path.join(MODEL_DIR, "season_end_lookup.csv"))
    print("[저장] 트랙맨 std5/gap4 매치테이블 + season_end_lookup.csv 사본 완료", flush=True)

    print("\n" + "=" * 70)
    print("candidate A 번들 생성 완료")
    print(f"  메타모델: {meta}")
    print(f"  CatBoost 피처 수: {len(feature_cols)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
