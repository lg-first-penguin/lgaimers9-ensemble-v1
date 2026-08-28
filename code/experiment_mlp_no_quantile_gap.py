# code/experiment_mlp_no_quantile_gap.py
"""팀원(조유담) 파이프라인과의 3가지 구조적 차이 중 2번째: MLP quantile PLE 임베딩
유무. 우리 MLP는 이걸 쓰고(§15.3, +50~56pt/7-7 검증됨), 팀원 쪽은 과거 실전
862~880대 회귀 사고의 용의자로 의심받아 롤백한 채 재도입을 보류 중이다(이후
"무죄" 쪽 정황 증거만 나옴, `teammate/yudam` EXPERIMENTS.md 참고).

이 스크립트는 우리 *현재* 프로덕션 피처셋(F1필터+시즌진행분+TrackA+coarse
pitchmix, tier A 없음) 위에서 quantile PLE 유무만 isolate해서 dual-regime으로
재확인한다 — §15의 원래 검증은 이 피처셋이 생기기 전이었으므로 "지금도" 여전히
이기는지 재확인하는 의미도 있다.

CatBoost는 quantile과 무관하므로 레짐당 baseline 1개만 학습해서 재사용,
MLP만 quantile on/off 두 변형을 3-seed로 비교한다.

사용법: python -m code.experiment_mlp_no_quantile_gap --both
"""
import argparse
import time

from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model
from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.thirdmodel_common import build_split, TARGET_COL

SCREEN_SEEDS = [42, 123, 7]


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (MLP quantile PLE 유무) ===\n{'='*70}", flush=True)

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    y_val = val_split[TARGET_COL].values
    device = get_device()

    t0 = time.time()
    X_train = train_split[cat_feature_cols]
    y_train = train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]
    cat_model, cat_best_iter = train_catboost(X_train, y_train, X_val, y_val, verbose=False)
    cat_preds = predict_catboost(cat_model, X_val)
    cat_score = compute_bss(cat_preds, y_val)[2]
    print(f"  [CatBoost baseline] Val Score={cat_score:.2f} (best_iter={cat_best_iter}, {time.time()-t0:.1f}s)", flush=True)

    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_va_cat, X_va_num, y_va = to_tensors(va_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    results = {}
    for tag, bin_edges in [("quantile(현재)", fit_quantile_edges(X_tr_num)), ("no_quantile(팀원 방식)", None)]:
        t0 = time.time()
        members = train_ensemble(
            X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric_feats=len(mlp_num_cols),
            embed_dims=embed_dims, bin_edges=bin_edges,
            X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_va,
            seeds=SCREEN_SEEDS, device=device, verbose=False,
        )
        preds = predict_ensemble(members, cat_dims, len(mlp_num_cols), embed_dims, X_va_cat, X_va_num,
                                  bin_edges=bin_edges, device=device)
        mlp_score = compute_bss(preds, y_va.numpy())[2]
        _, _, _, blend_score, _ = fit_meta_model(cat_preds, preds, y_val)
        print(f"  [MLP {tag}] solo={mlp_score:.2f} blend={blend_score:.2f} ({time.time()-t0:.1f}s)", flush=True)
        results[tag] = (mlp_score, blend_score)

    mlp_delta = results["quantile(현재)"][0] - results["no_quantile(팀원 방식)"][0]
    blend_delta = results["quantile(현재)"][1] - results["no_quantile(팀원 방식)"][1]
    print(f"\n  delta(quantile - no_quantile): MLP solo={mlp_delta:+.2f} | Blend={blend_delta:+.2f}", flush=True)
    return {"mlp_delta": mlp_delta, "blend_delta": blend_delta, **results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    results = {}
    if args.both:
        results["cutoff7"] = run_regime(True, 2024)
        results["season2023"] = run_regime(False, 2023)
    else:
        holdout = 2024 if args.cutoff7 else args.holdout
        label = "cutoff7" if args.cutoff7 else f"season{holdout}"
        results[label] = run_regime(args.cutoff7, holdout)

    print("\n" + "=" * 70)
    print(f"{'레짐':<14}{'MLP delta':>14}{'Blend delta':>14}")
    for name, r in results.items():
        print(f"{name:<14}{r['mlp_delta']:>+14.2f}{r['blend_delta']:>+14.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
