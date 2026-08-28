# code/experiment_yudam_hybrid_mlp.py
"""[레버 A] 하이브리드 수치형 인코딩. 현재 프로덕션 MLP(유담 레시피)는 모든 수치형을
raw StandardScaler concat 한다. 가설: quantile PLE 자체는 이득(§79 solo +48.76 / blend
+18.41)이지만 트랙맨64(상황 10-key 물리량 조인)가 2025 추론에서 상수 fallback 으로
붕괴할 때 PLE 가 취약해 유담님 쪽에서 실측 회귀(968.15→879.54)를 냈다. 그렇다면
**트랙맨64만 raw 로 격리하고 나머지 안정 피처엔 PLE 복원**하면 두 이득을 다 가질 수 있다.

3-arm 비교 (같은 커스텀 학습 루프, 인코딩만 다름 — apples-to-apples):
  baseline : 전 수치형 raw concat  (= 현재 프로덕션 MLP)
  hybrid   : PLE(트랙맨64 제외) + raw(트랙맨64)
  full_ple : 전 수치형 PLE          (§79 재현 + "PLE×트랙맨64" 가설 직접 확인)

CatBoost(유담 v2 HP)는 한 번만 학습해 세 arm 블렌드에 공유. 메타모델 val 전체 fit/채점.

사용법:
  python -m code.experiment_yudam_hybrid_mlp --regime cutoff7            # 3-seed
  python -m code.experiment_yudam_hybrid_mlp --regime cutoff7 --arms baseline,hybrid
  python -m code.experiment_yudam_hybrid_mlp --regime 2023
"""
import argparse
import gc
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    fit_quantile_edges, compute_bss, get_device, QuantileEmbedding,
    HIDDEN1, HIDDEN2, DROPOUT, LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE, QUANTILE_D,
)
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS

# 트랙맨64 컬럼: process_trackman_features_safe 산출물, 이름은 {metric}_{mean|std}_mean_{group}
# (phase1 agg[mean,std] 후 phase3 agg[mean] 이 붙어 "_mean_" 이 한 번 더 들어감 — 실측 확인).
# 8 metric × {mean,std} × {fastball,breaking,offspeed,other} = 64.
TRACKMAN64_RE = re.compile(
    r"^(rel_speed|spin_rate|induced_vert_break|horz_break|extension|rel_height|rel_side|zone_speed)"
    r"_(mean|std)_mean_(fastball|breaking|offspeed|other)$"
)


class HybridMLP(nn.Module):
    """앞 n_ple 개 수치형 컬럼은 QuantileEmbedding(PLE), 뒤 n_raw 개는 raw passthrough."""
    def __init__(self, n_ple, n_raw, cat_dims, embed_dims, ple_bin_edges, quantile_d=QUANTILE_D):
        super().__init__()
        self.n_ple, self.n_raw = n_ple, n_raw
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim + 2, edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        if n_ple > 0:
            self.quantile = QuantileEmbedding(ple_bin_edges, d_embed=quantile_d)
            numeric_out = n_ple * quantile_d + n_raw
        else:
            self.quantile = None
            numeric_out = n_raw
        self.mlp = nn.Sequential(
            nn.Linear(sum(self.embed_dims) + numeric_out, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2), nn.BatchNorm1d(HIDDEN2), nn.ReLU(),
            nn.Linear(HIDDEN2, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        emb = torch.cat([e(x_cat[:, i].long()) for i, e in enumerate(self.embeddings)], dim=1)
        if self.quantile is not None:
            x_ple = self.quantile(x_num[:, :self.n_ple])
            parts = [x_ple, x_num[:, self.n_ple:]] if self.n_raw > 0 else [x_ple]
            x_num_enc = torch.cat(parts, dim=1)
        else:
            x_num_enc = x_num
        return self.sig(self.mlp(torch.cat([emb, x_num_enc], dim=1))).squeeze(-1)


def train_one(X_tr_cat, X_tr_num, y_tr, X_va_cat, X_va_num, y_va, n_ple, n_raw,
              cat_dims, embed_dims, ple_edges, seed, device):
    torch.manual_seed(seed)
    m = HybridMLP(n_ple, n_raw, cat_dims, embed_dims, ple_edges).to(device)
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCELoss()
    loader = DataLoader(TensorDataset(X_tr_cat, X_tr_num, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    best_brier, best_state, no_imp = float("inf"), None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        m.train()
        for bc, bn, by in loader:
            opt.zero_grad()
            loss = crit(m(bc.to(device), bn.to(device)), by.to(device))
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vp = m(X_va_cat.to(device), X_va_num.to(device)).cpu().numpy()
        br = ((vp - y_va) ** 2).mean()
        if br < best_brier - 1e-9:
            best_brier, best_state, no_imp = br, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m


def run_arm(name, ple_cols, raw_cols, tr, va, cat_dims, embed_dims, seeds, device, cat_val, y_val):
    ordered = ple_cols + raw_cols
    n_ple, n_raw = len(ple_cols), len(raw_cols)
    tr_p, ce, ni, ns = tr
    # tr/va already preprocessed dfs; build tensors in the ordered layout
    Xtc = torch.tensor(tr_p[CAT_COLS].values.astype(np.float32))
    Xtn = torch.tensor(tr_p[ordered].values.astype(np.float32))
    ytr = torch.tensor(tr_p[TARGET].values.astype(np.float32))
    Xvc = torch.tensor(va[CAT_COLS].values.astype(np.float32))
    Xvn = torch.tensor(va[ordered].values.astype(np.float32))
    ple_edges = fit_quantile_edges(Xtn[:, :n_ple]) if n_ple > 0 else None

    preds = []
    for s in seeds:
        mdl = train_one(Xtc, Xtn, ytr, Xvc, Xvn, y_val, n_ple, n_raw, cat_dims, embed_dims, ple_edges, s, device)
        with torch.no_grad():
            preds.append(mdl(Xvc.to(device), Xvn.to(device)).cpu().numpy())
        del mdl; gc.collect()
    mlp_val = np.mean(preds, axis=0)
    mlp_solo = compute_bss(mlp_val, y_val)[2]

    clf = LogisticRegression().fit(np.column_stack([cat_val, mlp_val]), y_val)
    wc, wm = (float(c) for c in clf.coef_[0]); b = float(clf.intercept_[0])
    blend = compute_bss(1 / (1 + np.exp(-(wc * cat_val + wm * mlp_val + b))), y_val)[2]
    print(f"[{name}] MLP solo={mlp_solo:.2f} | blend={blend:.2f} (n_ple={n_ple} n_raw={n_raw}, "
          f"w_cat={wc:.3f} w_mlp={wm:.3f})", flush=True)
    return mlp_solo, blend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="cutoff7", choices=["cutoff7", "2023", "2022", "2021"])
    ap.add_argument("--arms", default="baseline,hybrid,full_ple")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    arms = args.arms.split(",")
    seeds = YUDAM_ENSEMBLE_SEEDS[:args.seeds]
    cb_seeds = YUDAM_CATBOOST_SEEDS[:args.seeds]
    device = get_device()

    train_split, val_split, num_cols, cat_feature_cols, all_cols = build_split(regime=args.regime)
    y_val = val_split[TARGET].values

    trk = [c for c in num_cols if TRACKMAN64_RE.match(c)]
    nontrk = [c for c in num_cols if not TRACKMAN64_RE.match(c)]
    print(f"[cols] num 총 {len(num_cols)} = 트랙맨64 {len(trk)} + 나머지 {len(nontrk)}", flush=True)

    # 전처리는 전 num_cols 공통(StandardScaler). arm별로 순서만 재배치.
    tr_p, ce, ni, ns, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    va_p = apply_preprocessing(val_split, CAT_COLS, num_cols, ce, ni, ns)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    cb = train_catboost_ensemble(
        train_split[cat_feature_cols], train_split[TARGET].values,
        val_split[cat_feature_cols], y_val, seeds=cb_seeds, verbose=False, params=_yudam_catboost_params(),
    )
    cat_val = predict_catboost_ensemble([m for m, _ in cb], val_split[cat_feature_cols])
    print(f"[CatBoost] solo={compute_bss(cat_val, y_val)[2]:.2f}", flush=True)
    del train_split, val_split; gc.collect()

    ARM_DEF = {
        "baseline": ([], nontrk + trk),        # 전부 raw
        "hybrid":   (nontrk, trk),              # PLE(비트랙맨) + raw(트랙맨64)
        "full_ple": (nontrk + trk, []),         # 전부 PLE
    }
    res = {}
    for a in arms:
        ple_cols, raw_cols = ARM_DEF[a]
        res[a] = run_arm(a, ple_cols, raw_cols, (tr_p, ce, ni, ns), va_p, cat_dims, embed_dims, seeds, device, cat_val, y_val)

    print(f"\n{'='*64}\n=== 하이브리드 인코딩 요약 | regime={args.regime} | {args.seeds}-seed ===\n{'='*64}", flush=True)
    if "baseline" in res:
        bs_solo, bs_bl = res["baseline"]
        for a in arms:
            s, bl = res[a]
            print(f"  {a:10s} MLP solo={s:8.2f} ({s-bs_solo:+.2f}) | blend={bl:8.2f} ({bl-bs_bl:+.2f})", flush=True)
    else:
        for a in arms:
            s, bl = res[a]
            print(f"  {a:10s} MLP solo={s:8.2f} | blend={bl:8.2f}", flush=True)


if __name__ == "__main__":
    main()
