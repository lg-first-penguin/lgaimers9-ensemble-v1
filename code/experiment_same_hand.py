# code/experiment_same_hand.py
"""새 피처 후보: `same_hand`/`same_hand_advantage` (팀원 조유담님 저장소, 미채택 상태로
남아있던 아이디어 — `teammate/yudam/EXPERIMENTS.md` §1-6 참고).

투수-타자 손이 같은지(둘 다 좌 또는 둘 다 우) 플래그 + "리그 평균 대비 잘하는 투수가
같은 손 매치업에서 유독 다른지" 교차항. 팀원 repo에서의 원래 결과(2023+2024 평균,
그들 파이프라인 기준): XGBoost +14.85, CatBoost -16.26(기각), **MLP +47.90(2024와
비슷한 구간만 보면 +80.52로 특히 큼)**. 다만 팀원 쪽은 이 결과를 확정 채택하지 못하고
이후 원인불명 리더보드 regression 소동에 휘말려 통째로 롤백했다(same_hand 자체가 나쁘다고
결론 난 게 아니라 디버깅 편의상 되돌린 것) — 우리 repo에서는 한 번도 시도한 적 없는
피처라 이번 세션에서 직접 재현/검증한다.

두 컬럼:
  - `same_hand`: (pitcher_hand == batter_hand) 정수 플래그. train.csv에 이미 1(좌)/2(우)로
    인코딩되어 있어 팀원 repo처럼 문자열 매핑이 필요 없다(직접 확인, `pitcher_hand`/
    `batter_hand` value_counts로 1/2만 존재함을 확인).
  - `same_hand_advantage`: `pitcher_relative_success * same_hand` — `add_engineered_features`가
    이미 계산해두는 `pitcher_relative_success`(투수 성공률 - 리그 평균)를 그대로 재사용.

dual-regime(cutoff7 + season==2023 holdout) 스크리닝. CatBoost는 단일모델(결정적),
MLP는 3-seed 앙상블(프로젝트 관례). 두 모델 다 정식 프로덕션 피처셋
(`thirdmodel_common.build_split` — 시즌진행분+TE-residual+coarse pitchmix, tier A 없음)
위에 same_hand 2컬럼만 추가해서 비교. CatBoost/MLP 각각 독립적으로 추가 여부를 테스트하므로
팀원처럼 모델별로 다르게 라우팅(승자만 채택)하는 것도 이 스크립트 결과로 바로 판단 가능.

사용법:
  python -m code.experiment_same_hand --cutoff7
  python -m code.experiment_same_hand --holdout 2023
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostClassifier, Pool
from torch.utils.data import DataLoader, TensorDataset

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, TabularMLP, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device, to_tensors,
)
from code.thirdmodel_common import build_split

torch.use_deterministic_algorithms(True, warn_only=True)

SAME_HAND_COLS = ["same_hand", "same_hand_advantage"]
MLP_SEEDS = [42, 123, 7]
BATCH_SIZE = 4096
LR = 0.003
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 60
PATIENCE = 7


def apply_same_hand(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.int64)
    df["same_hand_advantage"] = df["pitcher_relative_success"] * df["same_hand"]
    return df


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_catboost_variant(tag, train_split, val_split, cat_cols):
    X_train, y_train = train_split[cat_cols], train_split["control_success"].values
    X_val, y_val = val_split[cat_cols], val_split["control_success"].values
    t0 = time.time()
    model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
    preds = model.predict_proba(X_val)[:, 1]
    score = compute_bss(preds, y_val)[2]
    print(f"  [CatBoost {tag}] Val Score={score:.2f} (best_iteration={best_iter}, "
          f"{time.time()-t0:.1f}s, n_features={len(cat_cols)})")
    return score


def train_mlp_once(X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                    X_val_cat, X_val_num, y_val, seed, device):
    torch.manual_seed(seed)
    model = TabularMLP(cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best_state, best_epoch, best_val_brier, epochs_no_improve = None, 0, float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_cat, batch_num, batch_y in loader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_cat, batch_num)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
        val_brier, _, _ = compute_bss(val_preds, y_val)

        if val_brier < best_val_brier - 1e-9:
            best_val_brier, best_epoch = val_brier, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def run_mlp_variant(tag, train_split, val_split, num_cols, device):
    t0 = time.time()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, "control_success")
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, "control_success")
    y_val = val_proc["control_success"].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    preds_list, best_epochs = [], []
    for seed in MLP_SEEDS:
        model, best_epoch = train_mlp_once(
            X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
            X_val_cat, X_val_num, y_val, seed, device,
        )
        with torch.no_grad():
            preds_list.append(model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy())
        best_epochs.append(best_epoch)
    ens_preds = np.mean(preds_list, axis=0)
    _, _, score = compute_bss(ens_preds, y_val)
    print(f"  [MLP {tag}] 3-seed ensemble Val Score={score:.2f} "
          f"best_epochs={best_epochs} (n_features={len(num_cols)}, {time.time()-t0:.1f}s)")
    return score


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout)
    train_sh = apply_same_hand(train_split)
    val_sh = apply_same_hand(val_split)

    device = get_device()
    results = {}

    results["CatBoost baseline"] = run_catboost_variant("baseline", train_split, val_split, cat_feature_cols)
    results["CatBoost +same_hand"] = run_catboost_variant(
        "+same_hand", train_sh, val_sh, cat_feature_cols + SAME_HAND_COLS)

    results["MLP baseline"] = run_mlp_variant("baseline", train_split, val_split, mlp_num_cols, device)
    results["MLP +same_hand"] = run_mlp_variant(
        "+same_hand", train_sh, val_sh, mlp_num_cols + SAME_HAND_COLS, device)

    print(f"\n--- {label} 요약 ---")
    print(f"  CatBoost: baseline={results['CatBoost baseline']:.2f} "
          f"+same_hand={results['CatBoost +same_hand']:.2f} "
          f"delta={results['CatBoost +same_hand']-results['CatBoost baseline']:+.2f}")
    print(f"  MLP:      baseline={results['MLP baseline']:.2f} "
          f"+same_hand={results['MLP +same_hand']:.2f} "
          f"delta={results['MLP +same_hand']-results['MLP baseline']:+.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(args.cutoff7, holdout)


if __name__ == "__main__":
    main()
