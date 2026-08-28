# code/experiment_sooyun_te_residual.py
"""손수연 vs 조유담 비교(대화 세션)에서 나온 가설 검증: 손수연 레시피(77피처, F1
미적용, no_both, wide-categorical, season-1앵커 트랙맨)엔 이 repo가 실전 +13.86으로
검증한 Track A(TE-residual, `code/train.py::apply_te_residual_features`)가 빠져있고,
조유담 레시피(실전 1085.24)엔 있다 — 이게 두 외부 레시피 간 ~30점 격차의 일부를
설명하는지 cutoff7에서 3-seed로 검증한다.

베이스: §85(멀티시드 재검증, F1 미적용이 F1 적용보다 항상 나음)의 결론에 따라 F1
필터는 걸지 않는다. TE-residual은 CatBoost 전용(§45 원 채택과 동일 라우팅) —
train_sy를 자기 자신의 causal source로 써서(merge_asof allow_exact_matches=False라
리크 불가) 6개 잔차 컬럼을 추가하고, SOOYUN_CAT_COLS는 그대로, feature_cols에만
TE_RESIDUAL_COLS를 얹는다.

사용법: python -m code.experiment_sooyun_te_residual
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.mlp_model import compute_bss, predict_bundle, get_device
from code.blend_model import fit_meta_model
from code.train import apply_te_residual_features, TE_RESIDUAL_COLS
from code.experiment_sooyun_recipe import (
    build_sooyun_features, SOOYUN_CATBOOST_PARAMS, SOOYUN_ITERATIONS, SOOYUN_CAT_COLS,
    check_memory_or_abort, DATA_DIR, TARGET,
)

SEEDS = [42, 123, 7]


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

    import pickle
    with open("./open/reference/best_model.pkl", "rb") as f:
        ref_bundle = pickle.load(f)
    mlp_bundle = ref_bundle["mlp_bundle"]
    device = get_device()

    from code.train import add_engineered_features, apply_same_hand
    df_ours = df_raw.copy()
    df_ours["top_bottom"] = df_ours["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    cutoff_train_mask = (df_raw["season"] < 2024) | ((df_raw["season"] == 2024) & (df_raw["game_month"] < 7))
    cutoff_val_mask = (df_raw["season"] == 2024) & (df_raw["game_month"] >= 7)
    league_mean_cutoff = df_ours.loc[cutoff_train_mask, TARGET].mean()
    df_ours_cutoff = add_engineered_features(df_ours.copy(), league_mean_cutoff)
    df_ours_cutoff = apply_same_hand(df_ours_cutoff)
    val_ours_cutoff = df_ours_cutoff.loc[cutoff_val_mask].reset_index(drop=True)
    mlp_preds = predict_bundle(mlp_bundle, val_ours_cutoff, device=device)

    check_memory_or_abort("손수연 피처 빌드 전")
    sooyun_full = build_sooyun_features(df_raw)
    # build_sooyun_features가 마지막에 no_both로 pitcher_id/batter_id를 지우는데,
    # TE-residual(Track A)의 causal groupby가 이 두 ID를 그룹키로 쓰므로 row_id 기준
    # 원본 df_raw에서 다시 붙여준다 -- TE-residual 계산 후에는 다시 지워서(아래) 최종
    # CatBoost 입력엔 no_both가 그대로 유지되게 한다.
    sooyun_full = sooyun_full.merge(df_raw[["row_id", "pitcher_id", "batter_id"]], on="row_id", how="left")
    train_sy = sooyun_full.loc[cutoff_train_mask].reset_index(drop=True)
    val_sy = sooyun_full.loc[cutoff_val_mask].reset_index(drop=True)
    del sooyun_full

    for c in SOOYUN_CAT_COLS:
        train_sy[c] = train_sy[c].astype(str)
        val_sy[c] = val_sy[c].astype(str)

    # Track A(TE-residual) 추가: train_sy를 자기 자신의 causal source로 사용
    te_prior = train_sy[TARGET].mean()
    train_sy = apply_te_residual_features(train_sy, train_sy, te_prior)
    val_sy = apply_te_residual_features(train_sy, val_sy, te_prior)
    train_sy = train_sy.drop(columns=["pitcher_id", "batter_id"])  # no_both 복원
    val_sy = val_sy.drop(columns=["pitcher_id", "batter_id"])

    drop_cols = ["row_id", TARGET]
    feature_cols = [c for c in train_sy.columns if c not in drop_cols]
    y_val = val_sy[TARGET].values
    print(f"[손수연+TrackA] train={len(train_sy)} val={len(val_sy)} features={len(feature_cols)} "
          f"(TE-residual {len(TE_RESIDUAL_COLS)}개 포함)", flush=True)

    rows = []
    for seed in SEEDS:
        check_memory_or_abort(f"seed={seed} 학습 전")
        t0 = time.time()
        train_pool = Pool(train_sy[feature_cols], train_sy[TARGET], cat_features=SOOYUN_CAT_COLS)
        val_pool = Pool(val_sy[feature_cols], val_sy[TARGET], cat_features=SOOYUN_CAT_COLS)
        model = CatBoostClassifier(**SOOYUN_CATBOOST_PARAMS, iterations=SOOYUN_ITERATIONS, random_seed=seed)
        model.fit(train_pool)
        cat_preds = model.predict_proba(val_pool)[:, 1]
        _, _, cat_score = compute_bss(cat_preds, y_val)
        _, _, _, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val)
        print(f"[seed={seed}] cat={cat_score:.2f} blend={blend_score:.2f} ({time.time()-t0:.1f}s)", flush=True)
        rows.append((seed, cat_score, blend_score))

    print("\n" + "=" * 60)
    print(f"{'seed':<8}{'cat':>12}{'blend':>12}")
    for seed, c, b in rows:
        print(f"{seed:<8}{c:>12.2f}{b:>12.2f}")
    avg_c = np.mean([r[1] for r in rows]); avg_b = np.mean([r[2] for r in rows])
    print(f"{'평균':<8}{avg_c:>12.2f}{avg_b:>12.2f}")
    print("=" * 60)
    print("비교 기준 -- §85 손수연(TrackA 없음) 3-seed 평균: cat=699.17 blend=758.55")
    print("비교 기준 -- 우리 프로덕션 레퍼런스: cat_solo=706.56 blend=753.37")


if __name__ == "__main__":
    main()
