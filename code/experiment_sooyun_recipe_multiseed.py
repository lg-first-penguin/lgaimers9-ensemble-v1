# code/experiment_sooyun_recipe_multiseed.py
"""§84(code/experiment_sooyun_recipe.py, 단일시드 cutoff7 블렌드 +10.63)의 다음
단계 — 사용자 지시: "순서대로 진행" (1) 멀티시드 재검증 (2) 그의 레시피 위에 F1
필터를 추가해서 재검증. 두 가지를 한 번에 처리한다: 손수연 피처(build_sooyun_features,
F1 미적용 원본)를 한 번만 만들고, 시드 3개(42/123/7) x {F1 미적용, F1 적용} 2가지
변형을 모두 학습해 cutoff7에서 비교한다. MLP는 여전히 우리 레퍼런스 번들을
재학습 없이 고정 재사용(시드와 무관하게 예측은 1번만 계산).

season2023은 여기서도 안 다룬다 — §84에 기록된 대로 재사용하는 MLP 번들이 cutoff7
컨벤션(2019~2023 전부 학습에 포함)이라 season2023 홀드아웃엔 구조적으로 리크된다.
그 확인은 별도로 MLP를 season2023 스플릿 전용으로 재학습해야 하는 다음 단계다.

사용법: python -m code.experiment_sooyun_recipe_multiseed
"""
import pickle
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.mlp_model import compute_bss, predict_bundle, get_device
from code.blend_model import fit_meta_model
from code.train import add_engineered_features, apply_same_hand, apply_f1_filter
from code.experiment_sooyun_recipe import (
    build_sooyun_features, SOOYUN_CATBOOST_PARAMS, SOOYUN_ITERATIONS, SOOYUN_CAT_COLS,
    check_memory_or_abort, DATA_DIR, TARGET,
)

SEEDS = [42, 123, 7]


def train_and_score(train_df, val_df, feature_cols, y_val, seed):
    train_pool = Pool(train_df[feature_cols], train_df[TARGET], cat_features=SOOYUN_CAT_COLS)
    val_pool = Pool(val_df[feature_cols], val_df[TARGET], cat_features=SOOYUN_CAT_COLS)
    model = CatBoostClassifier(**SOOYUN_CATBOOST_PARAMS, iterations=SOOYUN_ITERATIONS, random_seed=seed)
    model.fit(train_pool)
    preds = model.predict_proba(val_pool)[:, 1]
    _, _, score = compute_bss(preds, y_val)
    return preds, score


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

    with open("./open/reference/best_model.pkl", "rb") as f:
        ref_bundle = pickle.load(f)
    mlp_bundle = ref_bundle["mlp_bundle"]
    device = get_device()

    df_ours = df_raw.copy()
    df_ours["top_bottom"] = df_ours["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    cutoff_train_mask = (df_raw["season"] < 2024) | ((df_raw["season"] == 2024) & (df_raw["game_month"] < 7))
    cutoff_val_mask = (df_raw["season"] == 2024) & (df_raw["game_month"] >= 7)

    league_mean_cutoff = df_ours.loc[cutoff_train_mask, TARGET].mean()
    df_ours_cutoff = add_engineered_features(df_ours.copy(), league_mean_cutoff)
    df_ours_cutoff = apply_same_hand(df_ours_cutoff)
    val_ours_cutoff = df_ours_cutoff.loc[cutoff_val_mask].reset_index(drop=True)
    del df_ours, df_ours_cutoff

    # 우리 MLP는 시드/F1변형과 무관하게 1번만 예측 (고정 재사용)
    mlp_preds = predict_bundle(mlp_bundle, val_ours_cutoff, device=device)

    check_memory_or_abort("손수연 피처 빌드 전")
    sooyun_full = build_sooyun_features(df_raw)
    train_sy_noF1 = sooyun_full.loc[cutoff_train_mask].reset_index(drop=True)
    val_sy = sooyun_full.loc[cutoff_val_mask].reset_index(drop=True)
    del sooyun_full

    for c in SOOYUN_CAT_COLS:
        train_sy_noF1[c] = train_sy_noF1[c].astype(str)
        val_sy[c] = val_sy[c].astype(str)

    train_sy_f1 = apply_f1_filter(train_sy_noF1)

    drop_cols = ["row_id", TARGET]
    feature_cols = [c for c in train_sy_noF1.columns if c not in drop_cols]
    y_val = val_sy[TARGET].values
    print(f"[손수연 피처] train(F1없음)={len(train_sy_noF1)} train(F1적용)={len(train_sy_f1)} val={len(val_sy)}", flush=True)

    rows = []
    for seed in SEEDS:
        check_memory_or_abort(f"seed={seed} F1없음 학습 전")
        t0 = time.time()
        cat_preds_noF1, cat_score_noF1 = train_and_score(train_sy_noF1, val_sy, feature_cols, y_val, seed)
        _, _, _, blend_noF1, _ = fit_meta_model(cat_preds_noF1, mlp_preds, y_val)
        print(f"[seed={seed}][F1없음] cat={cat_score_noF1:.2f} blend={blend_noF1:.2f} ({time.time()-t0:.1f}s)", flush=True)

        check_memory_or_abort(f"seed={seed} F1적용 학습 전")
        t0 = time.time()
        cat_preds_f1, cat_score_f1 = train_and_score(train_sy_f1, val_sy, feature_cols, y_val, seed)
        _, _, _, blend_f1, _ = fit_meta_model(cat_preds_f1, mlp_preds, y_val)
        print(f"[seed={seed}][F1적용] cat={cat_score_f1:.2f} blend={blend_f1:.2f} ({time.time()-t0:.1f}s)", flush=True)

        rows.append((seed, cat_score_noF1, blend_noF1, cat_score_f1, blend_f1))

    print("\n" + "=" * 90)
    print(f"{'seed':<8}{'cat(noF1)':>12}{'blend(noF1)':>14}{'cat(F1)':>12}{'blend(F1)':>12}")
    for seed, c0, b0, c1, b1 in rows:
        print(f"{seed:<8}{c0:>12.2f}{b0:>14.2f}{c1:>12.2f}{b1:>12.2f}")
    avg_c0 = np.mean([r[1] for r in rows]); avg_b0 = np.mean([r[2] for r in rows])
    avg_c1 = np.mean([r[3] for r in rows]); avg_b1 = np.mean([r[4] for r in rows])
    print(f"{'평균':<8}{avg_c0:>12.2f}{avg_b0:>14.2f}{avg_c1:>12.2f}{avg_b1:>12.2f}")
    print("=" * 90)
    print("우리 프로덕션 레퍼런스: cat_solo=706.56, blend=753.37 (cutoff7)")


if __name__ == "__main__":
    main()
