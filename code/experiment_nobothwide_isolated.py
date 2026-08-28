# code/experiment_nobothwide_isolated.py
"""손수연 vs 조유담 비교에서 나온 발견: `no_both`(CatBoost에서 pitcher_id/batter_id
완전 제거)와 넓은 categorical 선언(8개, team_id/hand 포함)은 손수연 고유의 레버다 --
조유담은 코드로 직접 확인한 결과 둘 다 안 쓴다(teammate/yudam/full_retrain_blend_f1.py:
features에 ID 그대로 남음, categorical_features_cb는 object dtype만 자동 추출되는데
원본 데이터에서 object인 컬럼은 top_bottom/game_type/base_state뿐이고 top_bottom도
곧바로 int 매핑되어 실질 2개(game_type/base_state) -- 우리 repo와 동일).

즉 이 두 레버는 이 repo에서도, 조유담 repo에서도 한번도 테스트된 적이 없다. 손수연
레시피 전체를 이식하지 않고 이 두 레버만 **이 repo 자신의 프로덕션 CatBoost
레시피**(F1필터+Track A+시즌진행분+coarse pitchmix, `code/thirdmodel_common.py`가
그대로 재현) 위에 얹어서 순수하게 이 레버들만의 효과를 cutoff7에서 격리 검증한다.

4개 변형 x 3-seed:
  baseline        -- 현재 프로덕션 그대로 (categorical=game_type,base_state 2개, ID 유지)
  no_both         -- pitcher_id/batter_id를 피처에서 제거
  wide_cat        -- ID는 유지하되 pitcher_team_id/batter_team_id/pitcher_hand/batter_hand도
                     categorical로 추가 선언(4개 -> 총 6개, 손수연의 8개에서 hand_matchup/
                     top_bottom string 유지는 제외 -- 그건 그의 레시피 고유 조합이라 이번엔
                     안 건드림, 순수하게 "ID를 categorical로도 선언하면"에 집중)
  combined        -- no_both + wide_cat

사용법: python -m code.experiment_nobothwide_isolated
"""
import time

import numpy as np
from catboost import CatBoostClassifier, Pool

from code.mlp_model import compute_bss, predict_bundle, get_device
from code.blend_model import fit_meta_model
from code.catboost_model import CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.thirdmodel_common import build_split
from code.experiment_sooyun_recipe import check_memory_or_abort

SEEDS = [42, 123, 7]
TARGET = "control_success"
WIDE_EXTRA_CATS = ["pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand"]

VARIANTS = {
    "baseline": dict(drop_ids=False, wide_cat=False),
    "no_both": dict(drop_ids=True, wide_cat=False),
    "wide_cat": dict(drop_ids=False, wide_cat=True),
    "combined": dict(drop_ids=True, wide_cat=True),
}


def train_variant(train_split, val_split, base_features, cat_features_base, y_val, seed, drop_ids, wide_cat):
    features = list(base_features)
    cat_features = list(cat_features_base)
    if drop_ids:
        features = [c for c in features if c not in ("pitcher_id", "batter_id")]
    if wide_cat:
        cat_features = cat_features + WIDE_EXTRA_CATS

    X_train = train_split[features].copy()
    X_val = val_split[features].copy()
    if wide_cat:
        for c in WIDE_EXTRA_CATS:
            X_train[c] = X_train[c].astype(str)
            X_val[c] = X_val[c].astype(str)

    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["random_seed"] = seed
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, train_split[TARGET], cat_features=cat_features)
    val_pool = Pool(X_val, val_split[TARGET], cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    preds = model.predict_proba(X_val)[:, 1]
    _, _, score = compute_bss(preds, y_val)
    return preds, score


def main():
    print("[1/2] 우리 프로덕션 스플릿 (cutoff7, F1필터 적용)", flush=True)
    t0 = time.time()
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=True, apply_f1=True)
    check_memory_or_abort("build_split 완료")
    y_val = val_split[TARGET].values
    print(f"[build_split] train={len(train_split)} val={len(val_split)} cat_features={len(cat_feature_cols)} "
          f"({time.time()-t0:.1f}s)", flush=True)

    import pickle
    with open("./open/reference/best_model.pkl", "rb") as f:
        ref_bundle = pickle.load(f)
    mlp_bundle = ref_bundle["mlp_bundle"]
    device = get_device()
    mlp_preds = predict_bundle(mlp_bundle, val_split, device=device)
    _, _, mlp_score = compute_bss(mlp_preds, y_val)
    print(f"[우리 MLP, 고정 재사용] Val Score={mlp_score:.2f}", flush=True)

    cat_features_base = ["game_type", "base_state"]

    results = {name: [] for name in VARIANTS}
    for variant_name, cfg in VARIANTS.items():
        for seed in SEEDS:
            check_memory_or_abort(f"{variant_name} seed={seed} 학습 전")
            t0 = time.time()
            cat_preds, cat_score = train_variant(
                train_split, val_split, cat_feature_cols, cat_features_base, y_val, seed,
                drop_ids=cfg["drop_ids"], wide_cat=cfg["wide_cat"],
            )
            _, _, _, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val)
            print(f"[{variant_name}][seed={seed}] cat={cat_score:.2f} blend={blend_score:.2f} ({time.time()-t0:.1f}s)", flush=True)
            results[variant_name].append((seed, cat_score, blend_score))

    print("\n" + "=" * 70)
    print(f"{'variant':<12}{'cat_avg':>12}{'blend_avg':>12}")
    for variant_name, rows in results.items():
        avg_c = np.mean([r[1] for r in rows])
        avg_b = np.mean([r[2] for r in rows])
        print(f"{variant_name:<12}{avg_c:>12.2f}{avg_b:>12.2f}")
    print("=" * 70)
    print("비교 기준 -- 우리 프로덕션 레퍼런스: cat_solo=706.56 blend=753.37")


if __name__ == "__main__":
    main()
