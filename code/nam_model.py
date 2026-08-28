# code/nam_model.py
"""3rd-모델 탐색 4번 후보: NAM(Neural Additive Model, Agarwal et al. 2021 "Neural Additive
Models: Interpretable Machine Learning with Neural Nets"). CatBoost/MLP/DeepFM 전부 피처 간
상호작용(트리 분할, dense 결합, order-2 곱)을 어떤 형태로든 학습하는데, NAM은 정반대로
"피처 하나당 독립된 작은 신경망(shape function)의 합"이라는 가법성(additivity) 제약을
구조적으로 강제한다 — 상호작용을 원천적으로 표현할 수 없는 유일한 후보.

튜닝판은 원 논문이 제시한 3가지 핵심 기법을 그대로 적용한다:
  1. ExU(exp-centered unit) 첫 레이어: relu((x-b)*exp(w)) — 표준 relu(wx+b)보다 날카로운
     구간별 변화(jagged function)를 더 잘 근사한다고 논문이 보고 (본 데이터의 볼카운트/
     이닝처럼 정수형 임계값이 많은 피처에 특히 적합할 수 있다는 가설).
  2. Feature dropout: 미니배치마다 피처 전체(shape function 출력 하나)를 통째로 무작위
     드롭 — 논문의 핵심 정칙화, 특정 피처(특히 고카디널리티 pitcher_id/batter_id)에 대한
     과도한 의존을 줄인다.
  3. Output penalty: 피처별 shape function 출력의 제곱합에 작은 페널티를 걸어 각 피처
     기여도가 과도하게 커지는 것(=암묵적 과적합)을 억제.

피처별 독립 소형 MLP를 파이썬 반복문 없이 벡터화하려고 QuantileEmbedding과 같은 einsum
패턴("bnf,nfd->bnd" 스타일, mlp_model.py 참고)으로 "필드별 독립 2-layer MLP"를 구현한다."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.mlp_model import compute_bss, get_device

NAM_HIDDEN = 32
NAM_LR = 0.003
NAM_WEIGHT_DECAY = 0.01
NAM_BATCH_SIZE = 4096
NAM_MAX_EPOCHS = 60
NAM_PATIENCE = 7


class NAM(nn.Module):
    """카테고리 필드는 임베딩(dim=1) 룩업 shape function, 수치 필드는 필드별 독립
    2-layer MLP(1->hidden->hidden->1) shape function. 전부 더해 로짓을 만든다 — 교차항 없음.
    `use_exu=True`면 1층을 ExU(exp-centered unit)로, `feature_dropout>0`이면 피처 단위
    구조적 드롭아웃을 적용한다."""

    def __init__(self, cat_dims, num_numeric, hidden=NAM_HIDDEN, use_exu=True,
                 feature_dropout=0.0, subnet_dropout=0.1):
        super().__init__()
        self.cat_embed = nn.ModuleList([nn.Embedding(dim + 2, 1) for dim in cat_dims])
        for emb in self.cat_embed:
            nn.init.zeros_(emb.weight)

        self.n_num = num_numeric
        self.hidden = hidden
        self.use_exu = use_exu
        self.feat_dropout = nn.Dropout(feature_dropout) if feature_dropout > 0 else None
        self.subnet_dropout = nn.Dropout(subnet_dropout) if subnet_dropout > 0 else None
        if num_numeric > 0:
            self.w1 = nn.Parameter(torch.empty(num_numeric, hidden))           # 1 -> hidden, 필드별
            self.b1 = nn.Parameter(torch.zeros(num_numeric, hidden))
            self.w2 = nn.Parameter(torch.empty(num_numeric, hidden, hidden))   # hidden -> hidden, 필드별
            self.b2 = nn.Parameter(torch.zeros(num_numeric, hidden))
            self.w3 = nn.Parameter(torch.empty(num_numeric, hidden))           # hidden -> 1, 필드별
            self.b3 = nn.Parameter(torch.zeros(num_numeric))
            if use_exu:
                # ExU 초기화(논문 권장): 가중치는 N(4,0.5) 부근(exp(4)~55 스케일)이 아니라
                # 프로젝트 특성상 입력이 이미 표준화돼 있으므로 N(0,0.5)로 보수적으로 시작.
                nn.init.normal_(self.w1, mean=0.0, std=0.5)
            else:
                nn.init.normal_(self.w1, std=0.3)
            nn.init.normal_(self.w2, std=0.1)
            nn.init.normal_(self.w3, std=0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_cat, x_num, return_contribs=False):
        if self.cat_embed:
            cat_terms = torch.cat([emb(x_cat[:, i].long()) for i, emb in enumerate(self.cat_embed)], dim=1)  # (batch, n_cat)
        else:
            cat_terms = torch.zeros(x_num.shape[0], 0, device=x_num.device)

        if self.n_num > 0:
            if self.use_exu:
                h1 = torch.relu((x_num.unsqueeze(-1) - self.b1) * torch.exp(self.w1))
            else:
                h1 = torch.relu(x_num.unsqueeze(-1) * self.w1 + self.b1)          # (batch, n_num, hidden)
            if self.subnet_dropout is not None:
                h1 = self.subnet_dropout(h1)
            h2 = torch.relu(torch.einsum("bnh,nhg->bng", h1, self.w2) + self.b2)  # (batch, n_num, hidden)
            if self.subnet_dropout is not None:
                h2 = self.subnet_dropout(h2)
            num_terms = torch.einsum("bnh,nh->bn", h2, self.w3) + self.b3         # (batch, n_num)
        else:
            num_terms = torch.zeros(x_cat.shape[0], 0, device=x_cat.device)

        all_terms = torch.cat([cat_terms, num_terms], dim=1)  # (batch, n_fields) — 피처 단위 드롭아웃 대상
        if self.feat_dropout is not None:
            all_terms = self.feat_dropout(all_terms)

        logit = all_terms.sum(dim=1) + self.bias.squeeze(-1)
        if return_contribs:
            return logit, all_terms
        return logit


def train_nam(
    X_tr_cat, X_tr_num, y_tr, cat_dims, hidden=NAM_HIDDEN,
    X_val_cat=None, X_val_num=None, y_val=None,
    max_epochs=NAM_MAX_EPOCHS, patience=NAM_PATIENCE, batch_size=NAM_BATCH_SIZE,
    lr=NAM_LR, weight_decay=NAM_WEIGHT_DECAY, device=None, verbose=True, seed=42,
    use_exu=True, feature_dropout=0.0, subnet_dropout=0.1, output_penalty=0.0,
):
    torch.manual_seed(seed)
    device = device or get_device()
    model = NAM(
        cat_dims=cat_dims, num_numeric=X_tr_num.shape[1], hidden=hidden, use_exu=use_exu,
        feature_dropout=feature_dropout, subnet_dropout=subnet_dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(TensorDataset(X_tr_cat, X_tr_num, y_tr), batch_size=batch_size, shuffle=True, num_workers=0)
    has_val = X_val_cat is not None and y_val is not None and len(y_val) > 0

    best_state, best_epoch, best_val_brier, no_improve = None, 0, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bc, bn, by in loader:
            bc, bn, by = bc.to(device), bn.to(device), by.to(device)
            optimizer.zero_grad()
            logits, contribs = model(bc, bn, return_contribs=True)
            loss = criterion(logits, by)
            if output_penalty > 0:
                loss = loss + output_penalty * (contribs ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(model(X_val_cat.to(device), X_val_num.to(device))).cpu().numpy()
            val_brier, val_bss, val_score = compute_bss(val_preds, y_val)
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f} | Val Brier: {val_brier:.6f} | Val Score: {val_score:.2f}")
            if val_brier < best_val_brier - 1e-9:
                best_val_brier, best_epoch, no_improve = val_brier, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"[EarlyStopping] epoch {epoch}에서 종료 (best epoch: {best_epoch})")
                    break
        else:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch
