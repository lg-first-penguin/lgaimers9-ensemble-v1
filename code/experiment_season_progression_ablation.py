# code/experiment_season_progression_ablation.py
"""팀원 제보: 시즌진행분 8개(투수4+타자4) 중 투수만 남기면(타자 4개 제거) 오히려
점수가 더 오를 수 있다는 실험 결과가 있었다고 함 — 이 저장소에서 재검증한다.

`code/train.py::add_engineered_features`가 이미 프로덕션에서 8개 전부를 df에
추가하므로(무조건), build_split()이 반환하는 df에는 이미 8개 컬럼이 다 들어있다.
이 스크립트는 그 컬럼들을 다시 계산하지 않고, CatBoost feature 목록에서 어떤
컬럼을 넣고 뺄지만 다르게 해서 3-way(full8 / pitcher-only 4 / none)를 비교한다.

기존 code/experiment_season_progression_blend.py는 "use_new=False가 곧 시즌진행분
제외"라고 가정하고 짜여 있는데, add_engineered_features가 무조건 8개를 추가하도록
바뀐 지금은 그 가정이 깨져 baseline에도 8개가 섞여 들어간다 — 그래서 재사용하지
않고 이 스크립트를 새로 작성했다.

사용법:
  python -m code.experiment_season_progression_ablation --cutoff7
  python -m code.experiment_season_progression_ablation --holdout 2023
"""
import argparse
import time

from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_season_progression import SEASON_PROGRESSION_COLS, TARGET_COL, build_split
from code.mlp_model import compute_bss
from code.train import apply_f1_filter

PITCHER_COLS = [c for c in SEASON_PROGRESSION_COLS if c.startswith("pitcher_")]
BATTER_COLS = [c for c in SEASON_PROGRESSION_COLS if c.startswith("batter_")]

VARIANTS = [
    ("none (8개 전부 제외)", SEASON_PROGRESSION_COLS),
    ("pitcher-only (타자4 제외)", BATTER_COLS),
    ("full8 (현재 프로덕션)", []),
]


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(holdout, cutoff7)
    drop_cols = ["row_id", TARGET_COL]

    results = {}
    for tag, exclude_cols in VARIANTS:
        base_features = [
            c for c in df.columns
            if c not in drop_cols and c not in trk_mlp_cols and c not in trk_cat_cols and c not in exclude_cols
        ]
        cat_features = base_features + trk_cat_cols

        train_split = df.loc[train_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        val_split = df.loc[val_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        train_split = apply_f1_filter(train_split)

        X_train, y_train = train_split[cat_features], train_split[TARGET_COL].values
        X_val, y_val = val_split[cat_features], val_split[TARGET_COL].values

        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"[{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_features)})")
        results[tag] = score

    base = results["none (8개 전부 제외)"]
    print(f"\n--- {label} 요약 (vs 시즌진행분 완전 제외) ---")
    for tag, _ in VARIANTS:
        print(f"  {tag}: {results[tag]:.2f} ({results[tag]-base:+.2f})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
