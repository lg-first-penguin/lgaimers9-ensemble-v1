# code/excelformer_model.py
"""ExcelFormer(Chen et al. 2023) 근사 구현 — 3번째 앙상블 후보 스크리닝용.

논문 전체(특히 beta-mixup 데이터 증강, 학습된 AFI 상호작용 레이어)를 그대로
재현하지는 않습니다. 시간 대비 리스크가 커서(자체 재현 난이도 높음, 우리 피처 수가
~50개라 oversmoothing이 실전에서 문제 될 정도로 깊은 네트워크가 필요하지도 않을
가능성), 핵심 아이디어인 **semi-permeable attention**(정보가 "덜 중요한 피처 ->
더 중요한 피처" 방향으로만 흐르게 제한해 얕은 층에서도 oversmoothing을 줄이는 마스크)
만 근사해서 `code/ft_transformer_model.py::FTTransformer`와 직접 비교합니다.

- 피처 중요도 랭킹은 CatBoost의 `get_feature_importance()`로 근사합니다(논문은 자체
  랭킹 절차를 쓰지만, 우리는 이미 학습된 CatBoost가 있어 재사용). 랭킹 0이 가장 중요.
- attention mask: 토큰 i(피처, rank r_i)는 자기 자신, CLS, 그리고 자기보다 같거나 더
  중요한(rank가 같거나 낮은) 토큰에만 attend할 수 있습니다. CLS는 모든 토큰에 attend.
  이 마스크는 배치/레이어에 무관하게 고정이라 모델 초기화 시 한 번만 계산합니다.
- 토크나이저(NumericTokenizer/CategoricalTokenizer)와 학습 루프는
  `code/ft_transformer_model.py`와 동일한 패턴을 그대로 씁니다.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.ft_transformer_model import CategoricalTokenizer, NumericTokenizer, batched_forward
from code.mlp_model import compute_bss, get_device

D_TOKEN = 32
N_LAYERS = 3
N_HEADS = 8
FFN_MULT = 2
DROPOUT = 0.1
LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2048
MAX_EPOCHS = 60
PATIENCE = 7
SEED = 42

ENSEMBLE_SEEDS = [42, 123, 7, 2024, 99, 555, 31337]


def compute_feature_ranking(catboost_model, features, cat_cols, num_cols):
    """CatBoost feature importance로 [cat_cols..., num_cols...] 토큰 순서에 맞는
    중요도 랭킹(0=가장 중요)을 반환합니다. `features`는 CatBoost 학습 시 사용한
    컬럼 순서(catboost_model.get_feature_importance()와 정렬이 일치해야 함)."""
    importances = catboost_model.get_feature_importance()
    imp_map = dict(zip(features, importances))
    token_order = list(cat_cols) + list(num_cols)
    imp_values = np.array([imp_map[c] for c in token_order], dtype=np.float64)
    order = np.argsort(-imp_values)
    rank = np.empty(len(token_order), dtype=np.int64)
    rank[order] = np.arange(len(token_order))
    return rank


def build_semi_permeable_mask(rank_tokens):
    """(1+num_tokens, 1+num_tokens) 가산(additive) attention 마스크. 인덱스 0=CLS.
    피처 토큰 i는 자신/CLS/자신보다 중요하거나 같은(rank<=) 토큰에만 attend 가능."""
    n = len(rank_tokens)
    S = n + 1
    mask = torch.zeros(S, S)
    ranks = torch.tensor(rank_tokens, dtype=torch.float32)
    ri = ranks.view(n, 1)
    rj = ranks.view(1, n)
    disallow = rj > ri  # j가 i보다 덜 중요하면 i->j 금지
    disallow.fill_diagonal_(False)
    sub_mask = torch.zeros(n, n)
    sub_mask[disallow] = float("-inf")
    mask[1:, 1:] = sub_mask
    return mask  # row 0(CLS)/col 0(CLS)는 전부 0 = 항상 허용


class ExcelFormer(nn.Module):
    def __init__(self, cat_dims, bin_edges, feature_ranking, d_token=D_TOKEN, n_layers=N_LAYERS,
                 n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT):
        super().__init__()
        self.cat_tokenizer = CategoricalTokenizer(cat_dims, d_token)
        self.num_tokenizer = NumericTokenizer(bin_edges, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        self.register_buffer("attn_mask", build_semi_permeable_mask(feature_ranking))
        self.n_heads = n_heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        cat_tokens = self.cat_tokenizer(x_cat)
        num_tokens = self.num_tokenizer(x_num)
        tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens, mask=self.attn_mask)
        cls_out = self.norm(encoded[:, 0])
        return self.sigmoid(self.head(cls_out)).squeeze(-1)


def train_excel(
    X_tr_cat, X_tr_num, y_tr, cat_dims, bin_edges, cat_cols, num_cols, feature_ranking,
    X_val_cat=None, X_val_num=None, y_val=None,
    d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
    lr=LR, weight_decay=WEIGHT_DECAY, device=None, verbose=True, seed=SEED,
):
    torch.manual_seed(seed)
    device = device or get_device()
    model = ExcelFormer(
        cat_dims=cat_dims, bin_edges=bin_edges, feature_ranking=feature_ranking,
        d_token=d_token, n_layers=n_layers, n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    has_val = X_val_cat is not None and y_val is not None and len(y_val) > 0

    best_state = None
    best_epoch = 0
    best_val_brier = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_cat, batch_num, batch_y in loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_cat, batch_num)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)

        if has_val:
            model.eval()
            val_preds = batched_forward(model, X_val_cat, X_val_num, device=device)
            val_brier, val_bss, val_score = compute_bss(val_preds, y_val)
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f} | Val Brier: {val_brier:.6f} | Val Score: {val_score:.2f}")

            if val_brier < best_val_brier - 1e-9:
                best_val_brier = val_brier
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"[EarlyStopping] Val Brier 개선 없음 {patience} epoch 지속 -> epoch {epoch}에서 종료 (best epoch: {best_epoch})")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f}")
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def train_excel_ensemble(
    X_tr_cat, X_tr_num, y_tr, cat_dims, bin_edges, cat_cols, num_cols, feature_ranking,
    X_val_cat=None, X_val_num=None, y_val=None,
    seeds=ENSEMBLE_SEEDS, d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS,
    ffn_mult=FFN_MULT, dropout=DROPOUT, max_epochs=MAX_EPOCHS, patience=PATIENCE,
    batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, device=None, verbose=True,
):
    device = device or get_device()
    members = []
    for seed in seeds:
        model, best_epoch = train_excel(
            X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, bin_edges=bin_edges,
            cat_cols=cat_cols, num_cols=num_cols, feature_ranking=feature_ranking,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
            d_token=d_token, n_layers=n_layers, n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay, device=device, verbose=verbose, seed=seed,
        )
        members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": best_epoch,
            "seed": seed,
        })
        if verbose:
            print(f"[Ensemble] seed={seed} 학습 완료 (best_epoch={best_epoch})")
    return members


def predict_excel_ensemble(members, cat_dims, bin_edges, feature_ranking, X_cat, X_num,
                            d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_mult=FFN_MULT,
                            dropout=DROPOUT, device=None):
    device = device or get_device()
    preds_list = []
    for member in members:
        model = ExcelFormer(
            cat_dims=cat_dims, bin_edges=bin_edges, feature_ranking=feature_ranking,
            d_token=d_token, n_layers=n_layers, n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
        ).to(device)
        model.load_state_dict(member["state_dict"])
        model.eval()
        preds_list.append(batched_forward(model, X_cat, X_num, device=device))
    return np.mean(preds_list, axis=0)


def make_excel_bundle(members, cat_cols, num_cols, cat_dims, bin_edges, feature_ranking,
                       cat_encoder, num_imputer, num_scaler,
                       d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT):
    avg_best_epoch = int(round(np.mean([m["best_epoch"] for m in members]))) if members else 0
    return {
        "members": members,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_dims": cat_dims,
        "bin_edges": bin_edges,
        "feature_ranking": feature_ranking,
        "cat_encoder": cat_encoder,
        "num_imputer": num_imputer,
        "num_scaler": num_scaler,
        "d_token": d_token,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ffn_mult": ffn_mult,
        "dropout": dropout,
        "best_epoch_": avg_best_epoch,
    }


def predict_excel_bundle(bundle, df, device=None):
    from code.mlp_model import apply_preprocessing, to_tensors
    device = device or get_device()
    df_proc = apply_preprocessing(
        df, bundle["cat_cols"], bundle["num_cols"],
        bundle["cat_encoder"], bundle["num_imputer"], bundle["num_scaler"],
    )
    X_cat, X_num = to_tensors(df_proc, bundle["cat_cols"], bundle["num_cols"])
    return predict_excel_ensemble(
        bundle["members"], bundle["cat_dims"], bundle["bin_edges"], bundle["feature_ranking"], X_cat, X_num,
        d_token=bundle.get("d_token", D_TOKEN), n_layers=bundle.get("n_layers", N_LAYERS),
        n_heads=bundle.get("n_heads", N_HEADS), ffn_mult=bundle.get("ffn_mult", FFN_MULT),
        dropout=bundle.get("dropout", DROPOUT), device=device,
    )
