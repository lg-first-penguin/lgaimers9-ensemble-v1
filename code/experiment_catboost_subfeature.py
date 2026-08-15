# code/experiment_catboost_subfeature.py
"""CatBoost 전용 피처 제거 스크리닝 실험.

`code/experiment_mlp_subfeature.py::run_removal_blend`가 "MLP에서 add_engineered_features
12개를 빼면 어떨까"를 CatBoost는 고정한 채 검증했다(§28: 3-seed에선 유망했지만 7-seed+2023
홀드아웃에서 완전히 뒤집혀 기각). 이번엔 반대로 **CatBoost 쪽에서 같은 12개를 빼고 MLP는
고정**한 채 블렌드 점수를 본다 — §14.2는 CatBoost에 새 피처를 "추가"만 테스트했지 기존
피처를 "제거"하는 방향은 한 번도 테스트되지 않았다.

MLP는 이번 실험에서 변하지 않으므로 한 번만 학습해 재사용한다(CatBoost 변형마다 다시
학습하지 않음 — §28과 정반대 구조). CatBoost는 학습이 빨라(~80초) 시드 앙상블이 필요
없다(`CATBOOST_PARAMS`의 `random_seed=42` 고정, MLP처럼 epoch간 고분산 문제가 없음).

`code/experiment_attention.py::build_split`(트랙맨 없음, F1 필터 적용)을 재사용한다.

사용법:
  python -m code.experiment_catboost_subfeature --holdout 2024
  python -m code.experiment_catboost_subfeature --holdout 2023 --ensemble   # MLP 7-seed로 고정 학습
"""
import argparse

from code.mlp_model import ENSEMBLE_SEEDS, compute_bss, get_device
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model
from code.experiment_attention import build_split
from code.experiment_mlp_subfeature import ENGINEERED_FEATURES, SCREEN_SEEDS, run_variant


def run(holdout, seeds=SCREEN_SEEDS):
    train_split, val_split, features, num_cols = build_split(holdout, apply_f1=True)
    y_val = val_split["control_success"].values

    device = get_device()
    print(f"[Device] {device}")

    # MLP는 고정 — 두 CatBoost 변형에서 항상 동일 피처(baseline num_cols)로 한 번만 학습
    mlp_score, mlp_val_preds = run_variant(train_split, val_split, num_cols, device, seeds=seeds)
    print(f"[MLP(고정, seeds={len(seeds)})] Val Score: {mlp_score:.2f}\n")

    variants = {
        "baseline": features,
        "-engineered(12)": [c for c in features if c not in ENGINEERED_FEATURES],
    }

    results = {}
    for name, cols in variants.items():
        X_train_raw, y_train_raw = train_split[cols], train_split["control_success"].values
        X_val_raw = val_split[cols]
        catboost_model, best_iter = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=False)
        cat_val_preds = predict_catboost(catboost_model, X_val_raw)
        cat_score = compute_bss(cat_val_preds, y_val)[2]

        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val)
        results[name] = (cat_score, blend_score)
        print(f"[{name}] CatBoost solo: {cat_score:.2f} (best_iter={best_iter}) | Blend: {blend_score:.2f} (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f})")

    (base_cat, base_blend), (rm_cat, rm_blend) = results["baseline"], results["-engineered(12)"]
    print(f"\nCatBoost solo Delta: {rm_cat - base_cat:+.2f}")
    print(f"Blend Delta: {rm_blend - base_blend:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024)
    parser.add_argument("--ensemble", action="store_true", default=False, help="MLP를 7-seed(ENSEMBLE_SEEDS)로 고정 학습 (기본은 3-seed)")
    args = parser.parse_args()
    seeds = ENSEMBLE_SEEDS if args.ensemble else SCREEN_SEEDS
    run(args.holdout, seeds=seeds)
