# 실험 기록 (Feature Engineering & Hyperparameter Tuning)

`control_success` 예측 CatBoost 모델에 적용한 피처 엔지니어링과 하이퍼파라미터 튜닝 내역을 정리합니다. 기준은 이번 작업 시작 시점의 `open/reference/best_model.pkl` (BSS 0.00737 / 점수 736.61)입니다.

> **참고**: 이 로그 전체를 관통하는 "왜" 서사는 `PROJECT_HISTORY.md`에 정리돼 있습니다. 아래 본문 곳곳에 남아있는 `TABULAR_MLP_REPORT.md` 인용은 그 문서가 존재하던 시점에 작성된 원문 그대로 남겨뒀지만(로그이므로 소급 수정하지 않음), 해당 문서는 `PROJECT_HISTORY.md`로 대체되어 삭제되었습니다 — `TABULAR_MLP_REPORT.md §N`을 보게 되면 `PROJECT_HISTORY.md`의 대응 절을 대신 참고하세요.

## 점수 변화 요약

| 단계 | BSS | 환산 점수 | 비고 |
| --- | --- | --- | --- |
| 시작 시점 reference | 0.00737 | 736.61 | 이번 작업 이전 모델 |
| + 파생 피처 12개 추가 (기본 하이퍼파라미터) | 0.00766 | 765.99 | +29.4 |
| + 하이퍼파라미터 튜닝 (Optuna 40 trials) | **0.00819** | **818.54** | +52.5 (누적 +81.9) |

검증은 `code/test.py`의 기존 방식 그대로 — `season == 2024` 전체를 홀드아웃, `season < 2024`로 학습.

## 1. 추가된 피처 (`code/train.py::add_engineered_features`)

새 함수 `add_engineered_features(df, league_success_mean)`으로 12개 피처를 추가했습니다. `code/train.py`, `code/test.py`, `dopip.py`(full retrain), `submit/script.py`(코드 중복, 수동 동기화) 네 곳에 모두 반영했습니다.

| 피처명 | 계산식 | 의도 |
| --- | --- | --- |
| `pitcher_recent1_gap` | `asof_pitcher_prev1_game_success_rate − asof_pitcher_success_rate` | 최근 1경기 컨디션 변화 |
| `pitcher_recent3_gap` | `asof_pitcher_prev3_game_success_rate − asof_pitcher_success_rate` | 최근 3경기 흐름 |
| `pitcher_recent5_gap` | `asof_pitcher_prev5_game_success_rate − asof_pitcher_success_rate` | 최근 5경기 추세 |
| `pitcher_relative_success` | `asof_pitcher_success_rate − league_success_mean` | 리그 평균 대비 상대 실력 |
| `count_diff` | `strikes_before − balls_before` | 현재 카운트가 투수에게 유리한지 |
| `is_full_count` | `(balls_before==3) & (strikes_before==2)` → 0/1 | 풀카운트 여부 |
| `pitcher_count_advantage_raw` | `asof_pitcher_success_rate × count_diff` | 투수 실력 × 현재 카운트 |
| `pitcher_count_advantage_rel` | `pitcher_relative_success × count_diff` | 리그 대비 실력 × 현재 카운트 |
| `pitcher_trend` | `prev1_game_success_rate − prev5_game_success_rate` | 단기(1경기) vs 중기(5경기) 흐름 차이 |
| `pitcher_consistency` | `std([prev1, prev3, prev5]_game_success_rate)` | 최근 제구 안정성 |
| `matchup` | `asof_pitcher_success_rate − asof_batter_success_rate` | 투수·타자 상대 우위 |
| `count_pressure` | `pitcher_relative_success × li × 1[strikes_before≥2 or balls_before≥3]` | 압박 카운트(2스트라이크/3볼)에서의 상대실력 × 상황 중요도 |

**`league_success_mean`은 리크 방지를 위해 학습 시점마다 다르게 계산합니다:**
- `code/train.py`, `code/test.py` (평가용): `season < 2024` 학습 split의 `control_success` 평균만 사용. 검증에 쓰는 2024 시즌 데이터는 이 평균 계산에서 제외 — 홀드아웃을 분리한 의미를 지키기 위함.
- `dopip.py` full retrain, `submit/script.py`: 홀드아웃이 없는 단계이므로 `train.csv` 전체(2019~2024) 평균 사용.

**논의 중 제외/단순화된 부분:**
- 원래 아이디어는 "직전 시즌 리그 평균"(`prev_season_league_success`, 시즌별 lookup)이었으나, 구현 복잡도 대비 이득이 크지 않다고 판단해 **단일 스칼라 상수(전체 평균)**로 단순화함.
- `pitcher_relative_success`를 유지하기로 하면서 여기 의존하는 `pitcher_count_advantage_rel`, `count_pressure`도 함께 유지.

### Feature Importance (튜닝 전 reference 모델 기준, PredictionValuesChange)

| 피처 | Importance |
| --- | --- |
| `pitcher_relative_success` | 3.98 (전체 4위) |
| `pitcher_recent5_gap` | 0.96 |
| `pitcher_count_advantage_raw` | 0.82 |
| `count_diff` | 0.72 |
| `pitcher_recent1_gap` | 0.42 |
| `pitcher_consistency` | 0.36 |
| `count_pressure` | 0.35 |
| `pitcher_recent3_gap` | 0.30 |
| `pitcher_trend` | 0.27 |
| `pitcher_count_advantage_rel` | 0.23 |
| `matchup` | 0.18 |
| `is_full_count` | 0.04 |

새 피처 12개 합산 importance ≈ 8.6% (전체 대비). `pitcher_relative_success`가 압도적으로 기여도가 높고, `is_full_count`는 거의 쓰이지 않음 (제거 후보로 남겨둠, 아직 제거하지 않음). 참고로 `game_type`(25.4%), `season`(16.3%)이 여전히 가장 큰 비중을 차지함.

## 2. 하이퍼파라미터 튜닝

새 스크립트 `code/tune.py`로 Optuna(TPE sampler, seed=42) 기반 탐색을 40 trials 수행. 데이터 로딩·피처 엔지니어링은 1회만 수행하고 트라이얼 간 재사용(속도 최적화). 목적함수는 `season==2024` 홀드아웃 BSS 최대화, `early_stopping_rounds=50`으로 트라이얼별 반복수는 가변.

### 탐색 공간

| 파라미터 | 범위 |
| --- | --- |
| `depth` | 4 ~ 8 |
| `learning_rate` | 0.01 ~ 0.2 (log) |
| `l2_leaf_reg` | 1 ~ 30 (log) |
| `random_strength` | 0 ~ 10 |
| `bagging_temperature` | 0 ~ 5 |
| `border_count` | 32 ~ 254 |
| `min_data_in_leaf` | 1 ~ 200 (log) |
| `bootstrap_type` | `Bayesian` (고정) |

### 적용된 최적값 (Trial 28, BSS 0.00819)

| 파라미터 | 튜닝 전 (train.py 원래값) | 튜닝 후 (적용값) |
| --- | --- | --- |
| `depth` | 6 | **7** |
| `learning_rate` | 0.05 | **0.05040411253232039** |
| `l2_leaf_reg` | (미지정, 기본값 3) | **3.14659036827521** |
| `random_strength` | (미지정, 기본값 1) | **3.715568024268865** |
| `bagging_temperature` | (미지정, 기본값 1) | **0.4609270457436248** |
| `border_count` | (미지정, 기본값 254) | **106** |
| `min_data_in_leaf` | (미지정, 기본값 1) | **81** |
| `bootstrap_type` | (미지정, 기본값) | **Bayesian** (명시) |
| best_iteration | 337 (기존 reference) | **519** |

`code/train.py`의 `CatBoostClassifier`와 `dopip.py` full retrain 단계의 `CatBoostClassifier`에 동일하게 반영. 이전에는 두 곳의 하이퍼파라미터가 서로 달랐던 것(예: `learning_rate` 0.05 vs 0.03, `l2_leaf_reg` 3 vs 10)도 이번에 통일함.

## 3. 변경된 파일

- `code/train.py` — `add_engineered_features()` 추가, 파생 피처 반영, 튜닝된 하이퍼파라미터 적용
- `code/test.py` — 동일 파생 피처 로직 반영 (학습 모델과 동일 입력 셋 유지 목적)
- `dopip.py` — full retrain 단계에 동일 파생 피처 + 튜닝된 하이퍼파라미터 반영
- `submit/script.py` — 코드 중복으로 동일 파생 피처 로직 반영 (train.csv를 추가로 로드해 `league_success_mean` 계산)
- `code/tune.py` — 신규. Optuna 하이퍼파라미터 탐색 스크립트 (제출용 코드에는 포함되지 않음, 로컬 실험 전용)

## 4. 남은 이슈 / 다음 후보

- `is_full_count`는 importance가 거의 0에 가까워 제거해도 무방해 보이지만 아직 유지 중.
- `code/test.py`의 `X_val['top_bottom'] = ...` 라인에서 나는 `SettingWithCopyWarning`은 이번 작업과 무관한 기존 이슈, 미수정.
- 다음 방향으로는 앙상블보다 피처 엔지니어링을 우선하기로 함 — 특히 `asof_batter_*`(현재 3개뿐)가 상대적으로 덜 활용되어 있고, trackman 쪽 pitch-mix/상황 교차 피처도 아직 없음. 앙상블은 대회가 10분 추론 제한이 있는 코드 제출 방식이라 리스크 대비 이득이 낮다고 판단해 후순위로 미룸.

## 5. 모델 교체: CatBoost → Tabular MLP (PyTorch)

위 3장까지의 튜닝 결과가 CatBoost 기준 **BSS 0.00819 / 환산 818.54점**이었던 상황에서, 사용자가 Colab에서 아래와 같은 훨씬 단순한 피처셋(트랙맨 병합·파생 피처 없음, `test.csv` 원본 47개 컬럼만 사용, 범주형은 `top_bottom`/`game_type`/`base_state` 3개만 임베딩)으로 PyTorch Tabular MLP(임베딩 + `Linear(128)→BN→ReLU→Dropout→Linear(64)→BN→ReLU→Linear(1)→Sigmoid`, 3 epoch, `season==2024` 홀드아웃)를 돌렸더니 **환산 1,000점을 넘겼다**는 보고를 받고 모델을 CatBoost에서 Tabular MLP로 전면 교체함.

- 이 1,000점대 숫자는 이번 저장소의 트랙맨 병합/파생 피처 파이프라인이 아니라, 훨씬 단순한 피처셋으로 얻은 것이라는 점에 유의 — 직접적인 비교는 아님. 전체 파이프라인(트랙맨 병합 + 12개 파생 피처 포함)으로 재학습한 실제 BSS/점수는 아직 기록하지 않았으므로, `python dopip.py` 실행 후 `code/test.py` 출력을 확인해 이 표에 추가할 것.
- 기존 트랙맨 병합(`process_trackman_features_safe`)과 12개 파생 피처(`add_engineered_features`)는 그대로 유지한 채, **모델만** CatBoost → TabularMLP로 교체함 (Colab 프로토타입처럼 피처셋을 단순화하지는 않음). 범주형/수치형 분리는 Colab 프로토타입과 동일하게 `CAT_COLS = ['top_bottom', 'game_type', 'base_state']`만 임베딩하고 `pitcher_id`/`batter_id`/팀 ID 및 나머지 모든 `asof_*`/파생/트랙맨 피처는 수치형(스케일링)으로 처리.
- CatBoost의 `early_stopping_rounds`/`best_iteration_` 개념을 `code/mlp_model.py::train_mlp`의 validation-Brier 기준 early stopping(patience 7, 최대 60 epoch)과 `best_epoch_`로 대체. `dopip.py`의 전체 데이터 재학습 단계는 `best_epoch_ + 5` epoch만큼 재학습(기존 `best_iteration_ + 10` 관례와 동일한 취지).
- 모델·전처리기는 커스텀 클래스 인스턴스가 아니라 `dict`(state_dict + 범주형/수치형 컬럼 목록 + `OrdinalEncoder`/`SimpleImputer`/`StandardScaler` + `best_epoch_`)로 pickle — `submit.zip`에는 `code/` 패키지가 포함되지 않으므로 프로젝트 전용 클래스를 pickle하면 대회 서버에서 unpickle이 실패하기 때문. `submit/script.py`는 여전히 `code/`를 import하지 않는 독립 실행 파일이며, `TabularMLP` 클래스 정의와 전처리 로직을 수동으로 복제해 둠 (CatBoost 시절 `add_engineered_features` 중복 관례와 동일).
- `open/reference/best_model.pkl`에 남아있던 기존 CatBoost 모델(raw `CatBoostClassifier` pickle)은 새 dict 번들 포맷과 호환되지 않음. `code/test.py`가 reference 파일이 dict 번들이 아니면 비교 없이 "레거시로 추정"하고 신규 모델을 채택하도록 방어 코드를 추가해, `dopip.py`를 한 번만 돌리면 자동으로 새 MLP 모델이 reference로 승격되고 기존 CatBoost 모델은 `open/former_model/`로 백업되도록 함.
- **변경된 파일**: `code/mlp_model.py`(신규, `TabularMLP` 정의 + 학습/전처리 공용 유틸), `code/train.py`, `code/test.py`, `dopip.py`, `submit/script.py`, `submit/requirements.txt`(catboost 제거 — torch/pandas/numpy/scikit-learn은 평가 서버 기본 설치 패키지라 명시하지 않음).
- **변경하지 않은 파일**: `code/train_x30.py`/`code/test_x30.py`/`code/train_rd30.py`/`code/test_rd30.py`/`code/tune.py`/`code/train.last.py` — 여전히 CatBoost 기반, 이번 마이그레이션 범위 밖.
- 검증: 실제 대회 데이터 없이 스키마가 동일한 합성(랜덤) 데이터로 `code/train.py` → `code/test.py` → `dopip.py` → `submit/script.py`(단독 실행, `code/` 없이도 동작 확인)까지 전체 파이프라인이 에러 없이 끝까지 도는 것을 확인함. 합성 데이터라 BSS 수치 자체는 의미 없음 — 실제 데이터로 재실행해 점수를 확인 필요.

## 6. 실제 데이터 검증: Colab 1,500점의 진실, 그리고 MLP 자체 튜닝

### 6.1 실제 데이터로 첫 실행 — BSS 660.61, Colab 1,500점과 큰 괴리

실제 `open/data/`(train.csv 1,475,092행)로 5장의 최초 MLP(단일 모델, `CAT_COLS=['top_bottom','game_type','base_state']`, `embed_dim=8` 고정)를 학습한 결과 **실측 BSS 0.00661 / 환산 660.61점** — Colab에서 보고된 ~1,500점과 크게 차이남. 아래 순서로 원인을 규명함.

- **트랙맨 데이터 제외 비교**: 트랙맨 병합을 완전히 뺀 버전으로 재학습 → BSS 626.66. 포함 버전(660.61)이 근소하게 더 높음 — 트랙맨이 점수를 깎아먹는 게 아님을 확인, 괴리의 원인에서 배제.
- **시간 필터 재확인**: `process_trackman_features_safe`의 `is_train_split=True` "Time Filter"가 실제로는 아무 행도 제거하지 않음(`train.csv`·`trackman_history.csv` 둘 다 2024년 최대 `game_month`=10로 동일)을 확인. 그러나 `match_cols`에 `season`/`game_month`가 포함되어 있어 애초에 시즌을 넘나드는 병합 자체가 불가능하므로 리크 없음 — 마찬가지로 괴리의 원인이 아님.
- **Colab 1,500점의 실제 원인 (사용자 확인 완료)**: Colab 코드에 `if val_data.shape[0] == 0: ... 무작위 인덱스 분할(Hold-out 20%)`라는 폴백이 있었고, 이게 실제로 트리거되어 시간 기준(`season==2024`) 홀드아웃이 아니라 **무작위 20% 분할**로 검증했던 것으로 확인됨. 무작위 분할 + `pitcher_id`/`batter_id`/팀 ID를 스케일링된 숫자로 그대로 넣는 조합은 동일 선수의 투구가 train/val 양쪽에 섞여 들어가는 선수-정체성 리크를 유발하기 쉬움 → **Colab의 1,500점은 무효한 벤치마크였음**. 이후 목표는 "1,500점 재현"이 아니라 "리크 없는 `season==2024` 홀드아웃 기준으로 CatBoost(818.54)를 넘기기"로 재설정.

### 6.2 임베딩 확장 시도 — 실패, 그리고 원인

CatBoost feature importance 상위(`game_type` 25.4%, `season` 16.3%)를 참고해 `CAT_COLS`를 10개(`season`/`pitcher_hand`/`batter_hand`/`pitcher_team_id`/`batter_team_id`/`pitcher_id`/`batter_id` 추가)로 넓히고 `embed_dim_for_cardinality`(fastai 스타일 카디널리티 휴리스틱)로 컬럼별 임베딩 크기를 부여해봤으나, 실측 점수가 **660.61 → 303.79로 급락**(best_epoch=3에서 조기 종료 — 명백한 과적합 패턴). 두 원인을 분리해서 진단함:

- `pitcher_id`(792)/`batter_id`(830) 임베딩: 학습 데이터(~122만 행) 대비 카디널리티가 높아 표본이 적은 개별 선수 임베딩이 노이즈를 암기해버림. 이 둘만 빼도 351.53으로 여전히 낮음 — 두 번째 원인이 더 큼을 시사.
- `season` 임베딩: 검증셋(2024)이 학습셋(2019~2023)에 아예 없던 시즌값이라, 임베딩으로 두면 검증 데이터 전원이 "미학습 카테고리"의 랜덤 임베딩을 공유하게 되어 예측 전체가 체계적으로 miscalibrate됨. `season`만 다시 빼자(=`pitcher_id`/`batter_id`/`season` 모두 제외, `pitcher_hand`/`batter_hand`/`pitcher_team_id`/`batter_team_id`만 임베딩 유지) **690.62**로 회복 — 원래 660.61보다도 소폭 개선. **`season`처럼 미래에 반드시 새 값이 나오는(실제 평가 `test.csv`도 2025 시즌으로 `train.csv`엔 없는 값) 순서형 컬럼은 임베딩이 아니라 숫자로 다뤄야 외삽이 가능하다는 일반 교훈.**

### 6.3 하이퍼파라미터 스윕 — 유의미한 개선 없음

`CAT_COLS = ['top_bottom','game_type','base_state','pitcher_hand','batter_hand','pitcher_team_id','batter_team_id']` 기준으로 추가 튜닝:

| 설정 | 실측 점수 |
| --- | --- |
| 기준 (lr=0.003, wd=0.01) | 690.62 |
| lr=0.001 | 647.43 (하락) |
| wd=0.05 | 684.66 (유의미한 차이 없음) |

단일 모델은 동일 설정으로도 epoch마다 Val Score가 크게 흔들림(같은 run 안에서 490~690점대 오르내림) — 하이퍼파라미터보다 **분산 자체**가 병목이라고 판단, 앙상블로 방향 전환.

### 6.4 시드 앙상블 — 채택된 최종 방향

서로 다른 시드로 학습한 모델 여러 개의 예측 확률을 평균하는 방식으로 분산을 줄임 (전처리기는 1회만 fit, 시드별로 가중치 초기화·배치 순서만 다름):

| 설정 | 실측 점수 |
| --- | --- |
| 단일 모델 최고치 (참고용) | 715~740 |
| **7-seed 앙상블** | **780.92** (ad-hoc), **789.58** (파이프라인 실측) |
| 15-seed 앙상블 | 768.93 (추가 개선 없음, 정체) |
| 7-seed 앙상블 + 네트워크 확장(256/128, dropout 0.4, patience 10) | 777.92 (개선 없음) |

7-seed 앙상블(`ENSEMBLE_SEEDS = [42, 123, 7, 2024, 99, 555, 31337]`)을 최종 구조로 채택. `code/mlp_model.py::train_ensemble`/`predict_ensemble`로 구현하고, 번들 포맷을 `state_dict` 단일 키에서 `members`(리스트, 각 `{state_dict, best_epoch, seed}`) 키로 변경 — `code/test.py`의 레거시 감지도 `"members" in bundle` 기준으로 갱신. `dopip.py`의 full retrain 단계도 7개 시드 각각을 (해당 시드의 eval-run best_epoch + 5 buffer)만큼 전체 데이터로 재학습해 7-멤버 앙상블을 `submit/model/final_retained_model.pkl`로 저장. `submit/script.py`도 동일하게 7개 멤버를 로드해 예측을 평균하도록 갱신(코드 중복, 수동 동기화 유지).

### 6.5 최종 결과 요약

| 모델 | 실측 BSS 점수 (season==2024 홀드아웃) |
| --- | --- |
| CatBoost (튜닝됨, 3장 기준) | **818.54** |
| Tabular MLP 단일 모델 (최초 마이그레이션, CAT_COLS 3개) | 660.61 |
| Tabular MLP 단일 모델 (CAT_COLS 3개, 트랙맨 제외) | 626.66 |
| Tabular MLP 단일 모델 (CAT_COLS 10개 — 실패한 시도) | 303.79 |
| Tabular MLP 단일 모델 (CAT_COLS 7개, season/ID 제외) | 690.62 |
| **Tabular MLP 7-seed 앙상블 (최종 채택)** | **789.58** |

MLP 앙상블(789.58)은 CatBoost(818.54)를 아직 넘지 못함(-28.96점). 여러 아키텍처/하이퍼파라미터 실험이 690→789 구간에서 정체된 것으로 보아, 순수 아키텍처 튜닝만으로 이 격차를 메우기는 어려워 보임 — 사용자와 논의 후 "789.58 앙상블 MLP를 최종 확정"하기로 결정(수료 기준 549.51은 여유 있게 통과). CatBoost와의 블렌딩/스태킹이 다음으로 시도해볼 만한 방향으로 논의됨(아직 미착수).

## 7. 실제 리더보드 역전, TabM 시도, CatBoost 블렌딩 (최종)

MLP 단독 버전을 실제로 제출한 뒤, **로컬 검증과 실제 대시보드 순위가 뒤바뀌는 것**을 확인함 — 로컬은 CatBoost(818.54) > MLP(789.58)였지만, 실제 리더보드에서는 MLP가 CatBoost보다 약 +10점 높게 나옴. 이후 로컬 홀드아웃과 실전 결과가 상충하면 실전을 신뢰하는 방침으로 전환.

이어서 Deep Ensemble(시드별 완전 독립 모델 7개)의 학습 비용을 줄이기 위해 BatchEnsemble/TabM 스타일의 효율적 앙상블(공유 백본 + 멤버별 rank-1 승수 벡터)을 자체 구현해 시도했으나, k=32로도 로컬 597~608점에 그쳐(단일 MLP 690.62보다도 낮음) 근본 원인(추정: r/s 초기화가 항등에 가까워 멤버 분화 부족)을 못 찾고 보류.

마지막으로 CatBoost와 MLP 앙상블의 예측 확률을 단순 가중 평균(`alpha * CatBoost + (1-alpha) * MLP`)하는 블렌딩을 시도 — alpha를 그리드 서치한 결과 **alpha≈0.55~0.59에서 로컬 851~854점**으로 CatBoost·MLP 단독 성능을 모두 크게 상회함(오차 상관계수 0.9991로 매우 높았음에도). 이를 파이프라인에 정식 반영(`code/catboost_model.py`, `code/blend_model.py` 신규, 번들 스키마를 `{"catboost_model", "mlp_bundle", "alpha"}`로 변경)하고 실제 `dopip.py` 전체 파이프라인으로 재검증(alpha=0.59, 851.19)한 뒤 최종 제출본으로 채택.

CatBoost 하이퍼파라미터는 기존 튜닝된 reference 모델의 `model.get_params()`로 그대로 추출해 재사용 — 이 값 자체가 팀의 자체 Optuna 탐색(공식 데이터 기준) 결과이므로 대회 규칙상 문제 없음.

**실제 제출 결과: 대회 대시보드 924점.** 로컬 평가 점수(851.19) 대비 +72.81 — 로컬 홀드아웃이 실전 성능을 과소평가하는 패턴(§7 서두)이 블렌드에서도 재현, MLP 단독 사례(+10점)보다 격차가 더 커짐.

전체 배경·진단 과정·교훈은 `TABULAR_MLP_REPORT.md` §4~9에 상세 기록.

## 8. CatBoost `pitcher_id`/`batter_id` 카테고리 지정 시도 — 실패

두 모델 간 예측 상관관계가 0.9991로 매우 높아(§7) 앙상블 다양성 이득이 제한적이라는 가설 아래, CatBoost가 target-statistic 기반이라 고카디널리티 카테고리를 MLP 임베딩보다 잘 다룰 것으로 보고 `code/catboost_model.py::CAT_FEATURES`에 `pitcher_id`(792)/`batter_id`(830)를 추가해봄 (기존엔 `game_type`, `base_state`만 카테고리 지정 — 두 컬럼 자체는 이미 `features`에 숫자로 포함돼 있었음, "빠져있던" 게 아니라 "카테고리로 선언 안 된" 상태였음).

`code/train.py` 단독 재실행(season==2024 홀드아웃)으로 검증한 결과:

| 모델 | 변경 전 | 변경 후 |
| --- | --- | --- |
| CatBoost 단독 | 818.54 | **647.20** (-171.3) |
| MLP 단독 (참고, 영향 없어야 함) | 789.58 | 766.20 (정상 epoch 분산 범위) |
| 블렌드 (alpha 자동 재탐색) | 851.19 | **788.12** (-63.1), alpha 0.59→0.28 |

CatBoost 단독이 크게 하락했고 블렌드도 alpha가 MLP 쪽으로 크게 이동했음에도 방어하지 못함. §6.2에서 MLP 임베딩에 `pitcher_id`/`batter_id`를 추가했을 때(660.61→304)와 동일한 원인으로 판단: 검증셋(season==2024)이 학습셋(season<2024)에 없던 시즌이라, CatBoost의 ordered target-statistic 인코딩도 이 정도의 선수 구성 시프트 앞에서는 견고하지 않음 — target-statistic이 MLP의 랜덤 임베딩보다는 낫지만 여전히 훈련 시즌의 투수/타자별 통계에 의존하기 때문. **결론: `CAT_FEATURES`는 원복(`["game_type", "base_state"]`) — 시즌마다 값이 바뀌는 식별자성 컬럼은 모델 종류(트리 vs 신경망)와 무관하게 카테고리로 선언하지 말 것.**

## 9. CatBoost × MLP 스태킹 실험 (고정 alpha 블렌드 대신 메타모델)

alpha 가중평균 대신, `[cat_pred, mlp_pred]`를 입력으로 하는 `LogisticRegression` 메타모델로 두 예측을 비선형 결합하면 이득이 있는지 검증. `season==2024` 검증셋 전체를 메타모델 학습에 쓰면 과적합 평가가 되므로, 시간순으로 다시 쪼갬: `game_month<=8`을 `meta_train`(21.9만행, 메타모델 학습), `game_month>=9`를 `meta_val`(3.5만행, 평가)로 분리. alpha도 공정 비교를 위해 `meta_train`에서만 재탐색.

`meta_val`(9~10월) 구간은 두 모델 다 시즌 전체 대비 성능이 크게 낮았음(CatBoost 818.54→397.00, MLP 763.03→327.37 — 원인 미조사, 시즌 막바지 표본 특성으로 추정) — 그래도 상대 비교는 유효.

| 메타모델 구성 | meta_val 점수 |
| --- | --- |
| alpha 블렌드 (meta_train에서 재탐색, alpha=0.64) | 413.28 |
| **`[cat_pred, mlp_pred]` 로지스틱 회귀 스태킹** | **421.28** (+8.0) |
| + 상황 피처 9개(`balls_before`/`strikes_before`/`outs_before`/`inning`/`runner_on_1b~3b`/`li`/`score_diff_pitcher_team`, 표준화) | 418.25 (−3.0, 역효과) |
| + 상황 피처 9개 + `cat_pred*mlp_pred` 교호작용항 | 417.71 (추가 역효과) |

상황 피처를 넣었을 때 회귀 계수가 전부 -0.014~0.014로 사실상 0에 수렴 — CatBoost/MLP 둘 다 이미 카운트/이닝/주자 상황을 직접 입력으로 학습하므로, `cat_pred`/`mlp_pred`에 이미 그 정보가 녹아 있어 메타모델이 얻을 잔여 신호가 없었던 것으로 판단. 오히려 무의미한 피처가 늘면서 L2 정규화가 핵심 두 피처(`cat_pred`, `mlp_pred`)의 계수를 희석시켜 소폭 손해.

**결론(잠정)**: 순수 `[cat_pred, mlp_pred]` 2피처 스태킹이 alpha 블렌드보다 이 슬라이스에서 +8.0점(약 +1.9%) 우세. 상황 피처 추가는 기각. 단일 시점 분할(9~10월)만으로는 노이즈인지 실제 개선인지 판단하기 이르므로 여러 시점 슬라이스로 안정성 재검증(§9.1) 진행.

### 9.1 안정성 재검증 — rolling-origin 3-fold

단일 분할의 우연일 가능성을 배제하기 위해, `season==2024` 안에서 "과거→미래" 방향을 유지한 rolling-origin 분할 3개로 반복 검증(각 fold마다 alpha·메타모델 모두 해당 fold의 train 구간에서만 재학습):

| val 구간 | n_train | n_val | CatBoost | MLP | alpha 블렌드 | 스태킹 | delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5~6월 | 54,949 | 88,592 | 1015.62 | 960.08 | 1031.79 (α=0.86) | 1034.89 | +3.11 |
| 7~8월 | 143,541 | 74,990 | 753.34 | 773.40 | 795.22 (α=0.73) | 826.77 | **+31.55** |
| 9~10월 | 218,531 | 34,976 | 397.00 | 327.37 | 413.28 (α=0.64) | 421.28 | +8.00 |

3개 fold 전부 스태킹 승리(3/3), 평균 **+14.22점**. fold별 alpha가 0.64~0.86으로 시점에 따라 꽤 다르게 나오는 것도 눈에 띔 — 고정 alpha 하나로는 못 잡는 시점 의존성이 실제로 있고, 스태킹의 이득이 이 변동성을 일부 흡수하는 것으로 보임. 단일 분할 결과가 우연이 아니라는 근거로 충분하다고 판단, **정식 파이프라인 반영 채택 및 완료**(`code/blend_model.py`의 `sweep_alpha`를 `fit_meta_model`로 교체, 번들 스키마 `"alpha"` → `"meta_model": {"w_cat","w_mlp","intercept"}`, `code/train.py`/`code/test.py`/`dopip.py`/`submit/script.py` 전부 갱신, `dopip.py` 전체 파이프라인 재실행으로 검증 완료, 검증 스코어 855.31).

## 10. MLP 앙상블 부트스트랩(복원추출) 시도 — 실패

CatBoost·MLP 예측 상관관계(0.87~0.89대)를 더 낮춰 스태킹 이득을 키울 수 있는지 검증하기 위해, 7-seed MLP 앙상블의 시드별 학습 데이터를 "전체 학습셋 그대로 공유"가 아니라 시드마다 복원추출(bootstrap, N은 그대로 121만행 유지, 구성만 다르게)로 바꿔봄. 사용자가 과거에 "pitcher_id당 케이스 수 제한 → 줄인 양에 비례해 점수 하락, 편향 감소 효과보다 데이터량 감소 효과가 더 큼" 경험한 것을 근거로, "N을 유지하는 복원추출이면 하드 서브샘플링과 달리 안전하지 않을까"라는 가설을 세우고 검증.

| | Val Score |
| --- | --- |
| MLP 단독 (baseline, 시드 전체 동일 데이터) | 777.71 |
| MLP 단독 (bootstrap, 시드별 복원추출) | **724.96** (-52.75) |
| CatBoost 단독 (참고) | 818.54 |
| CatBoost-MLP 상관관계 (baseline → bootstrap) | 0.8939 → 0.8739 |
| 스태킹: CatBoost + baseline MLP | 861.50 |
| 스태킹: CatBoost + bootstrap MLP | **853.90** (-7.6) |

가설대로 상관관계는 소폭 줄었지만(0.894→0.874), MLP 개별 성능 손실(-52.75)이 그 이득을 압도해 최종 스태킹도 손해. 복원추출이라 N은 유지했어도, 유니크 행 기준으로는 평균 약 63.2%만 학습에 노출되는 효과(부트스트랩 표본의 기대 커버리지)라 사실상 하드 서브샘플링과 비슷하게 데이터량이 줄어든 것으로 해석됨. **결론: 기각. `code/mlp_model.py::train_ensemble`은 변경하지 않음.**

## 11. TabM(BatchEnsemble) 재도전 — Rademacher(±1) 초기화

§7/TABULAR_MLP_REPORT.md §5에서 보류된 TabM 프로토타입(공유 백본 + 멤버별 rank-1 승수 벡터 `r_k`/`s_k`)의 잔여 가설 — "r/s를 N(1,0.1)로 초기화하면 32개 가상 멤버가 조기종료 시점(~epoch 7)까지 충분히 분화하지 못한다" — 을 원 BatchEnsemble 논문 방식인 Rademacher(±1) 초기화로 재검증. 기존 코드는 남아있지 않아 새로 구현(공유 `nn.Embedding` + `BatchEnsembleLinear` 3층, BatchNorm은 K개 가상 멤버의 통계를 한 super-batch로 섞어버리는 문제가 있어 제외하고 plain Linear→ReLU→Dropout만 사용).

| 시도 | Val Score |
| --- | --- |
| 1차 시도 (r/s ~ N(1, 0.1), 기존 보류분) | 607.82 |
| 2차 시도 (r/s를 weight decay 대상에서 제외) | 597.32 |
| **3차 시도 (r/s ~ Rademacher(±1), 이번 실험)** | **708.39** (best_epoch=5) |

가설이 맞았음을 확인 — Rademacher 초기화가 N(1,0.1) 대비 **+100.57점** 개선. 다만 여전히 단일 MLP(690.62)는 근소하게 넘지만 7-seed Deep Ensemble(777~790)에는 못 미침. CatBoost와의 상관관계도 0.8835로 Deep Ensemble MLP(0.87~0.89대)와 비슷한 수준이라 다양성 이득도 크지 않았음. 스태킹(CatBoost+TabM) 832.83으로 CatBoost 단독(818.54)은 넘지만 현재 프로덕션 스태킹(CatBoost+Deep Ensemble MLP, 855~861)에는 못 미침.

**결론**: §5의 잔여 가설(초기화 문제)은 검증 완료 — 맞았음. 그러나 개선된 TabM도 Deep Ensemble을 대체할 만큼은 아니라서 **프로덕션 미채택**, `TabM 재도전` 항목은 이걸로 종결(추가 조사 불필요). 학습 비용 절감(1회 학습 vs 7회 독립 학습)이라는 원래 동기는 여전히 유효하지만, 이 데이터셋에서는 정확도 손실이 그 이득보다 커서 실익이 없다고 판단.

## 12. 세 번째 모델 계열 추가 — LightGBM / XGBoost / RandomForest 3-way 스태킹 시도 (보류)

TABULAR_MLP_REPORT.md "다음으로 시도해볼 만한 방향"에 남아있던 "블렌딩 확장 — 세 번째 모델 계열 추가"를 검증. CatBoost+MLP 2-way 스태킹(§9, 855.31)에 GBDT 계열 세 종류(LightGBM, XGBoost)와 배깅 계열(RandomForest)을 각각 하나씩 추가해, `[cat_pred, mlp_pred, third_pred]` 3피처 로지스틱 회귀 스태킹으로 이득이 있는지 확인. 하이퍼파라미터는 튜닝 없이 합리적인 기본값(`code/lightgbm_model.py`/`code/xgboost_model.py`/`code/randomforest_model.py` 참고) — 이득이 확인되면 그때 튜닝 여부를 판단하는 방침.

CatBoost/MLP 자체를 매번 재학습하지 않기 위해, 이미 완성된 `open/reference/best_model.pkl`을 그대로 불러와 season==2024 검증셋에 대한 `cat_pred`/`mlp_pred`를 추론만으로 재현하고(재학습 없이 CatBoost=818.54, MLP=767.93, 기존 meta_model 그대로 블렌드=855.31 — 기록된 값과 일치 확인), 그 위에 세 번째 모델만 학습해 얹는 방식으로 실험 비용을 크게 줄임(`code/experiment_3way_stack.py`).

| 3번째 모델 | 단독 Val Score | vs CatBoost 상관 | vs MLP 상관 | 3-way 스태킹 | 2-way(855.31) 대비 delta |
| --- | --- | --- | --- | --- | --- |
| LightGBM | 697.51 (best_iteration=88) | 0.921 | 0.865 | 854.29 | -1.03 |
| **XGBoost** | 729.46 (best_iteration=90) | 0.915 | 0.855 | **862.67** | **+7.36** |
| RandomForest | 521.88 (best_iteration=150) | 0.830 | 0.806 | 848.95 | -6.36 |

RandomForest가 상관관계는 세 후보 중 가장 낮았음(가장 "이질적")에도 3-way 스태킹은 가장 크게 손해를 봄 — 단독 성능이 너무 약해서(521.88) 이질성 이득보다 노이즈를 더 많이 끌어들인 것으로 해석. 반대로 XGBoost는 LightGBM과 상관관계가 비슷한데도(0.915 vs 0.921) 단독 성능이 더 좋아(729.46 vs 697.51) 유일하게 순이득을 냄. §11의 "이질성은 다른 알고리즘을 쓴다가 아니라 실제로 다르게 틀린다에서 나온다"는 결론에 더해, **단독 성능이 어느 정도 뒷받침되지 않으면 이질성만으로는 스태킹 이득을 내지 못한다**는 것을 보여준 사례.

### 12.1 XGBoost 3-way 안정성 재검증 — rolling-origin 3-fold

유일하게 순이득을 낸 XGBoost에 대해서만, §9.1과 동일한 월 버킷 확장 윈도우(3~4월 train-only, 5~6/7~8/9~10월을 순서대로 val)로 재검증. CatBoost/MLP/XGBoost 자체는 재학습하지 않고, 메타모델(2-way/3-way 로지스틱 회귀)만 각 fold의 train 구간에서 재학습.

| val 구간 | n_train | n_val | 2-way | 3-way(+XGBoost) | delta |
| --- | --- | --- | --- | --- | --- |
| 5~6월 | 54,949 | 88,592 | 1032.77 | 1030.71 | -2.06 |
| 7~8월 | 143,541 | 74,990 | 824.02 | 830.32 | +6.30 |
| 9~10월 | 218,531 | 34,976 | 424.68 | 435.67 | +10.98 |

2/3 fold 승리, 평균 **+5.07점**. §9.1에서 2-way 스태킹을 정식 채택할 때의 기준(3/3 fold 전부 승리, 평균 +14.22점)에는 못 미치는 약한 증거 — 방향은 대체로 긍정적이고 특히 시즌 후반(9~10월, §9.1에서도 스태킹 이득이 가장 컸던 구간)에서 이득이 크지만, 초반 fold(5~6월)는 오히려 소폭 손해라 우연이 아니라고 단정하기엔 이르다고 판단.

**결론**: **프로덕션 미채택, 보류.** `code/experiment_3way_stack.py`, `code/lightgbm_model.py`, `code/xgboost_model.py`, `code/randomforest_model.py`는 실험 스크립트로 리포지토리에 남겨두되(`code/train_x30.py` 등과 동일한 관례), `dopip.py`/`code/train.py`/`code/blend_model.py`/`submit/script.py`는 변경하지 않음. XGBoost 쪽만 폴드를 더 늘리거나(현재 3-fold는 표본이 작아 노이즈 여지가 있음) 하이퍼파라미터를 가볍게라도 튜닝해 증거를 보강하면 재검토 여지가 있음 — "블렌딩 확장" 항목은 완전히 종결하지 않고 이 상태로 열어둠.

### 12.2 팀원의 실제 튜닝된 XGBoost로 재시도 — 오히려 더 나쁨

§12의 XGBoost는 CatBoost 파라미터를 대충 유추한 튜닝 없는 기본값이었음. 팀원이 별도 트랙에서 이 저장소의 트랙맨 조인+`add_engineered_features` 12개 파생피처를 그대로 이식하고 Optuna 2라운드(총 55 trials: 넓은 탐색 30 + 좁힌 탐색 25)로 튜닝해 로컬 752.70을 낸 XGBoost가 있어, 그 하이퍼파라미터(`max_depth=6, learning_rate=0.0144, subsample=0.637, colsample_bytree=0.992, reg_lambda=0.0605, reg_alpha=4.368, min_child_weight=12, gamma=0.00897`, `n_estimators=3000`, 조기종료 기준도 logloss 대신 대회 채점 지표에 맞춘 커스텀 Brier eval_metric, `early_stopping_rounds=50`)를 `code/xgboost_model.py::train_xgboost_tuned`로 그대로 재현해 같은 실험을 반복.

이 저장소 파이프라인 위에서 재현한 단독 성능은 744.38(best_iteration=462) — 팀원이 보고한 752.70에 근접(차이는 미세한 환경/버전 차이로 추정, 트랙맨 조인·파생피처 코드 자체는 동일). 이 정도로도 §12의 튜닝 없는 XGBoost(729.46)보다 명확히 강한 모델.

| | 단독 Val Score | vs CatBoost 상관 | vs MLP 상관 | 3-way 스태킹(단일 분할) | 2-way(855.31) 대비 |
| --- | --- | --- | --- | --- | --- |
| §12 튜닝 없는 XGBoost | 729.46 | 0.915 | 0.855 | 862.67 | +7.36 |
| **팀원 튜닝 XGBoost (재현)** | **744.38** | **0.934** | 0.871 | 861.61 | +6.29 |

단독 성능은 더 좋아졌는데 CatBoost와의 상관관계도 같이 올라가서(0.915→0.934) 단일 분할 스태킹 이득은 오히려 근소하게 더 작음(+7.36→+6.29) — §10/§12에서 이미 확인한 "이질성과 단독 성능이 같이 움직이면 순이득이 상쇄된다"는 패턴이 그대로 재현됨.

rolling-origin 3-fold 재검증에서는 격차가 훨씬 크게 벌어짐:

| val 구간 | n_train | n_val | 2-way | 3-way(+튜닝 XGBoost) | delta |
| --- | --- | --- | --- | --- | --- |
| 5~6월 | 54,949 | 88,592 | 1032.77 | 1004.40 | **-28.37** |
| 7~8월 | 143,541 | 74,990 | 824.02 | 821.17 | -2.85 |
| 9~10월 | 218,531 | 34,976 | 424.68 | 433.26 | +8.58 |

**1/3 fold 승리, 평균 -7.55점** — §12의 튜닝 없는 버전(2/3, +5.07)보다 뚜렷하게 나쁨. 특히 메타모델 학습 표본이 가장 작은 첫 fold(train 54,949행, `[cat_pred, mlp_pred, third_pred]` 3피처)에서 -28.37로 크게 무너짐. CatBoost와의 상관관계가 더 높아진 만큼(0.934) `cat_pred`와 `third_pred` 간 공선성이 커져, 메타모델(로지스틱 회귀) 계수가 학습 표본이 작을 때 더 불안정해지는 것으로 추정 — 다만 정확한 메커니즘은 추가로 진단하지 않음(가설 수준).

**결론**: 팀원의 실전 검증된(로컬 752.70, 대시보드 795) 더 강한 XGBoost를 가져와도 3-way 스태킹에는 **오히려 더 나쁜 결과**. "더 좋은 단독 모델을 넣으면 스태킹도 더 좋아질 것"이라는 직관이 이번엔 틀렸음 — 단독 성능과 상관관계가 함께 올라가는 경우, 튜닝이 스태킹 관점에서는 역효과를 낼 수 있다는 사례. §12의 "프로덕션 미채택, 보류" 결론을 더 강하게 뒷받침. `code/xgboost_model.py::train_xgboost_tuned`는 참고용으로 남겨두되 추가 조사는 우선순위를 낮춤.

## 13. rolling-origin 3-fold의 "5~6월 붕괴"는 실제로 무엇이었나 — 소표본 아티팩트 진단, RandomForest 자체 성능 개선 시도

§12.2의 rolling-origin 재검증에서 튜닝 XGBoost 3-way가 5~6월 fold에서 -28.37이라는 큰 손해를 봤던 것에 대해, "5~6월이 실제로 특별히 어려운 구간이라 그런지, 아니면 그 fold의 메타모델 학습 데이터가 3~4월치(54,949행)뿐이라 생긴 소표본 불안정 아티팩트인지"를 구분하는 진단과, "RandomForest 자체 성능을 키우면 3-way가 될 수 있지 않을까"라는 두 가지 후속 질문을 검증.

### 13.1 소표본 아티팩트 진단 — "5~6월이 나쁜 게 아니라 표본이 적어서였다"

메타모델을 (A) 기존 방식대로 3~4월치(54,949행)만으로 학습시킨 경우와, (B) 5~6월만 빼고 시즌 2024 나머지 전체(164,915행)로 학습시킨 경우를 비교해, 둘 다 동일하게 5~6월 구간만 채점:

| 모델 | (A) 3~4월만 학습(n=54,949) | (B) 5~6월 제외 전체 학습(n=164,915) |
| --- | --- | --- |
| quick XGBoost | 1030.71 (2-way 대비 -2.06) | 1040.20 (2-way 대비 -3.56) |
| 튜닝 XGBoost | 1004.40 (2-way 대비 **-28.37**, w_mlp=0.285로 붕괴) | 1036.67 (2-way 대비 **-7.09**, w_mlp=1.432로 정상 회복) |

튜닝 XGBoost의 경우 학습 표본을 3배로 늘리자(A→B) `w_mlp` 계수가 0.285(사실상 죽음)에서 1.432(정상)로 회복되고, 손해도 -28.37 → -7.09로 대부분(약 21점) 줄어들었다. **즉 원래 관찰된 -28.37의 대다수는 "5~6월이 특별히 나쁜 달"이라서가 아니라, rolling-origin CV의 첫 fold가 구조적으로 메타모델 학습 표본이 작을 수밖에 없어(§12.1의 fold 경계상 3~4월 데이터만 가용) 생긴 아티팩트였다.** 실제 프로덕션(`dopip.py`)은 메타모델을 검증셋 전체(25만행)로 한 번에 학습시키므로 이 정도의 소표본 불안정은 애초에 재현되지 않는다.

다만 (B)에서도 여전히 -3.56~-7.09의 잔여 손해가 남는다 — 완전히 착시는 아니고, 5~6월에 3-way가 근소하게 불리한 무언가가 진짜 있을 가능성은 남아있다. 하지만 이는 시즌 2024 단 한 번의 관측치이고(2025년에 재현될지 불명), 이 정도 크기(단일 자릿수~한 자릿수 초반)의 월별 효과를 근거로 "특정 달만 3-way 예외 처리" 같은 규칙을 프로덕션에 넣는 것은 §9(상황 피처 추가 역효과)·TABULAR_MLP_REPORT.md 인사이트 6(로컬 홀드아웃 과최적화 위험)과 같은 함정에 해당한다고 판단해 **채택하지 않음**.

**교훈**: rolling-origin CV로 안정성을 검증할 때, 초반 fold(학습 표본이 작은 fold)의 극단적인 결과는 "그 시점 데이터의 특성"이 아니라 "표본 부족에 의한 메타모델 추정 불안정"일 수 있다 — 특히 피처 간 공선성이 높을 때(상관관계 0.93+) 이 효과가 크다. 극단적인 fold 결과를 볼 때는 그 fold의 학습 표본 크기부터 확인할 것.

### 13.2 RandomForest 자체 성능 개선 — objective를 Brier(MSE)로 바꾸기

§12에서 RandomForest가 3-way를 가장 크게 깎아먹은(-6.36) 이유가 "단독 성능이 너무 낮아서(521.88)"였다는 점에 착안해, RandomForest 자체를 개선할 수 있는지 검증.

**시도 1 — 용량 확대(더 깊은 트리, 더 작은 리프)는 역효과였다**: `max_depth`를 14→28로 늘리고 `min_samples_leaf`를 50→5로 줄이는 방향으로(모델 용량↑) 튜닝해봤지만 오히려 계속 나빠짐(521.88→483.37→364.15→212.41) — 과적합 방향. 반대로 더 강하게 정규화해도(leaf 100/200, depth 8~10) 마찬가지로 나빠짐(437.05~506.63) — 즉 기존 `max_depth=14, min_samples_leaf=50` 설정이 이미 이 방향(깊이/리프 크기)으로는 국소 최적점에 가까웠고, `criterion="log_loss"`로 바꿔봐도 도움 안 됨(505.91~517.92, gini 기준선 521.88보다 낮음).

**시도 2 — objective를 Brier로 교체(RandomForestRegressor) — 효과 있음**: RandomForest는 배깅이라 XGBoost처럼 손실함수의 그래디언트를 직접 타겟으로 학습하지 않고, 각 트리가 노드별 불순도 감소(기본 Gini)만 보고 독립적으로 자란다 — `criterion="brier"` 같은 옵션은 없음. 다만 Brier score(`(p-y)²`의 평균)는 이진 타겟에 대한 MSE와 수학적으로 동일하므로, `RandomForestClassifier` 대신 **`RandomForestRegressor`를 0/1 타겟에 그대로 학습**시키면 각 노드 분할이 기본 criterion `squared_error`(MSE)를 직접 줄이는 방향으로 이뤄져 Gini보다 Brier에 훨씬 가까운 목적함수가 된다.

`n_estimators=150, max_depth=14`에 `min_samples_leaf`만 바꿔가며 검증:

| min_samples_leaf | 단독 Val Score |
| --- | --- |
| 50 | 628.85 |
| 100 | 641.96 |
| 200 | 663.09 |
| **500** | **667.94** (수확체감 시작) |

Classifier 기준선(521.88) 대비 **+146.06(+28%)** 개선 — RandomForestClassifier에서는 아무리 depth/leaf를 조정해도 안 되던 것이, objective를 바꾸는 것만으로 크게 개선됨. leaf=200→500 구간의 개선폭(+4.85)이 leaf=100→200(+21.13)보다 확연히 작아 수확체감에 들어선 것으로 판단, 추가 탐색은 하지 않음.

이 개선된 RandomForest(leaf=100, leaf=500)로 3-way 스태킹을 다시 검증:

| | 단독 Val Score | vs CatBoost 상관 | vs MLP 상관 | 3-way(단일분할) | rolling-origin |
| --- | --- | --- | --- | --- | --- |
| §12 RandomForestClassifier(원본) | 521.88 | 0.830 | 0.806 | 848.95 (-6.36) | (미실시) |
| RandomForestRegressor(leaf=100) | 641.96 | 0.850 | 0.798 | 856.58 (+1.27) | 2/3 fold, 평균 **-2.93** |
| RandomForestRegressor(leaf=500) | 667.94 | 0.867 | 0.817 | 856.76 (+1.45) | 2/3 fold, 평균 **-2.75** |

| val 구간 | 2-way | 3-way(leaf=100) | 3-way(leaf=500) |
| --- | --- | --- | --- |
| 5~6월 | 1032.77 | 1015.14 (-17.64) | 1018.99 (-13.78) |
| 7~8월 | 824.02 | 827.01 (+2.99) | 824.28 (+0.25) |
| 9~10월 | 424.68 | 430.53 (+5.85) | 429.97 (+5.28) |

단독 성능을 521.88→667.94로 28% 끌어올렸음에도(상관관계는 0.83→0.87로 소폭만 상승, 여전히 네 후보 중 가장 낮은 축) 3-way 스태킹은 여전히 애매함 — 단일분할로는 근소한 플러스(+1.3~1.5)지만 rolling-origin 평균은 마이너스(-2.75~-2.93), 2/3 fold는 이겨도 5~6월의 손해가 나머지 이득을 상쇄. §13.1의 소표본 아티팩트를 감안하더라도(5~6월 fold 자체가 과장된 수치일 수 있음) 순이득이라고 부르기엔 여전히 근거가 얇음.

**결론**: RandomForest 자체 성능은 실제로 크게 개선 가능했다(objective 교체가 핵심 레버였음, "용량을 늘린다"는 직관적인 방향은 오히려 틀렸음). 하지만 그 개선이 3-way 스태킹을 채택 기준 위로 끌어올릴 만큼 크지는 않았다 — §12의 "프로덕션 미채택, 보류" 결론 유지. `RandomForestRegressor` 기반 구현은 참고용으로만 남기고 `code/randomforest_model.py`(Classifier 버전)는 변경하지 않음.

## 14. CatBoost/MLP 개별 성능 개선 시도 — 트랙맨 매칭 세분화, 파생 피처, loss function 교체 (모두 실패)

3번째 모델 추가가 §12~§13에서 보류로 마무리된 뒤, "새 모델을 추가하는 대신 기존 CatBoost/MLP 각각의 성능을 먼저 올려보자"는 방향으로 5개 후보를 검증. 스크리닝은 속도를 위해 CatBoost 단독(season==2024 홀드아웃, baseline 818.54) 또는 MLP 단독(3-seed 평균, `open/temp/experiment_stack3`에 캐시된 동일 split 재사용)으로 진행.

### 14.1 트랙맨 매칭 지문(fingerprint) coarsening — 세분화가 이미 최적, coarsening은 전부 손해

현재 `process_trackman_features_safe`의 매칭 지문은 10개 컬럼 조합(`match_cols`)이라 카디널리티가 매우 높고, 지문+`pitch_type_group` 조합당 트랙맨 매칭 건수 중앙값이 1건(평균 2.06)에 불과함을 확인했다. 언뜻 "표본 부족 → 노이즈"로 보이지만, 이는 선수 ID 조인이 애초에 불가능한 상황(`pitcher_id`/`batter_id`와 트랙맨의 `pitcher_trackman_id`/`batter_trackman_id`는 서로 다른 익명화 공간이라 값 범위조차 겹치지 않음 — 20만 행 샘플에서 overlap 0건 확인)에서 최대한 비슷한 상황의 과거 투구를 찾아 근사하려는 **의도적 설계**일 수 있다. 즉 bias(세분화 시 낮음)-variance(세분화 시 높음) 트레이드오프이지 확정된 버그가 아니므로, 이를 "고쳐야 할 문제"로 단정하지 않고 coarsening 3단계를 baseline과 나란히 CatBoost로 비교했다:

| variant | 지문+구종군당 median n | Val Score | delta |
| --- | --- | --- | --- |
| baseline(현재, 세분화) | 1 | **818.54** | +0.00 |
| `game_dayofweek` 제거 | 4 | 699.73 | -118.81 |
| +`inning` 제거 | 20 | 691.38 | -127.16 |
| +`game_month` 제거 | 138 | 659.97 | -158.57 |

coarsening은 표본 안정성을 확실히 높였음에도(median 1→138) 전부 큰 폭으로 손해를 봤다 — 세분화된 "근접 매칭"이 실제로 유효한 신호였고, coarsening은 그 신호를 뭉개기만 했다. 현재 설계가 이미 이 축에서는 최적에 가까움. **결론: 매칭 지문 변경 없음, 현행 유지.**

### 14.2 `asof_batter_*` 강화 / pitch-mix × 상황 교차 피처 — 둘 다 손해

투수 쪽(`pitcher_relative_success`, `pitcher_trend`, `pitcher_consistency` 등)에 비해 타자 쪽 파생 피처가 `matchup` 하나뿐으로 상대적으로 빈약하다는 점에 착안해 두 묶음을 각각 CatBoost로 테스트했다:

- **배터 강화** (`batter_relative_success`, `batter_relative_middle`, `batter_experience_log`, `pitcher_experience_log`, `matchup_confidence`, `batter_pressure`, `batter_middle_vs_pitcher_middle`, 7개 추가): 774.64 (**-43.89**)
- **pitch-mix 교차** (`pitchmix_confidence`, `fastball_pressure`, `breaking_fullcount`, `offspeed_ahead`, `mix_entropy`, `fastball_rate_x_situational_speed`, `breaking_rate_x_situational_break`, 7개 추가): 779.48 (**-39.06**)

두 경우 모두 개별 feature importance는 0.2~1.7 수준으로 "쓰이긴" 했지만(CatBoost `PredictionValuesChange` 기준), 순효과는 마이너스였다. 공통적으로 `best_iteration`이 519 → 430 전후로 앞당겨지는 패턴이 두 실험 모두에서 반복됐다. 원인으로 의심되는 것: (a) `batter_relative_success` 등 일부는 기존 피처의 단조변환(상수를 빼거나 곱하기만 함)이라 트리 분할 관점에서 새 정보가 거의 없고, (b) CatBoost가 `random_strength≈3.7`(123개 피처 기준으로 Optuna 튜닝된 값)로 분할 후보 선택에 무작위성을 주입하는데, 정보 없는 중복 피처가 늘어나면 분할 후보 풀이 희석되어 좋은 분할이 상대적으로 덜 뽑힐 수 있다. 이 가설을 확인하려고 단조변환 피처 1개만 추가하는 대조군 실험(`feat_control_single.py`)을 준비했으나 실행 직전 작업 방향이 바뀌어 미실시 — 메커니즘은 확정하지 못했지만, 두 후보 모두 결과가 이미 명확히 마이너스라 어차피 **채택하지 않는다.**

### 14.3 loss function을 Brier(MSE)로 교체 — MLP는 노이즈 이내, CatBoost는 손해

§13.2에서 RandomForest가 objective를 Gini/log_loss에서 Brier와 수학적으로 동일한 MSE(`RandomForestRegressor`)로 바꿔 크게 개선됐던 것에 착안해, CatBoost(`Logloss`→`RMSE`)와 MLP(`BCELoss`→`MSELoss`)에도 같은 아이디어를 적용해봤다.

**MLP** (단일모델, seed 3개[42, 123, 7] 평균):

| config | seed별 Val Score | 평균 |
| --- | --- | --- |
| baseline(`BCELoss`, `base_state` embed_dim=5) | 677.93 / 678.83 / 693.45 | 683.40 |
| `MSELoss`(embed_dim=5) | 708.87 / 652.28 / 683.11 | 681.42 (**-1.98**) |
| `BCELoss`, `base_state` embed_dim=8(카디널리티 8에 맞춰 확장) | 700.18 / 673.50 / 691.48 | 688.39 (**+4.99**) |

두 변화 모두 config 내부의 시드 간 변동폭(최대 56점)보다 작아 노이즈 수준이다 — 단일 MLP는 원래 epoch/시드에 따라 편차가 크다는 게 이미 알려진 사실(§6.4, 500~740점대). `base_state`는 카디널리티가 8뿐이라 기존 embed_dim=5로도 8개 범주를 선형독립적으로 표현하기엔 이미 충분한 용량일 가능성이 높고, 이후 MLP 레이어가 임베딩+수치형을 비선형으로 결합하므로 병목이 embed_dim 자체는 아닐 수 있다 — 다만 3-seed 스크리닝만으로는 결론을 내리기엔 검정력이 약하다. 둘 다 뚜렷한 방향성이 없는 상태라 7-seed 풀 앙상블 재검증 비용을 들일 근거가 부족해 여기서 중단.

**CatBoost** (`loss_function`만 `RMSE`로 교체, `eval_metric`도 `RMSE`로 맞춤 — 나머지 하이퍼파라미터 동일):

| loss | Val Score | delta |
| --- | --- | --- |
| `Logloss`(baseline) | 818.54 | +0.00 |
| `RMSE`(Brier와 수학적으로 동일) | 782.57 | **-35.97** |

RandomForest와 정반대 결과다. RandomForest는 배깅이라 그래디언트 없이 노드별 불순도 감소만 보고 트리를 독립적으로 키우므로, "불순도 기준(Gini)"과 "평가지표(Brier)"의 정합성 자체가 목적함수 선택의 전부였다. 반면 CatBoost/GBDT는 그래디언트 기반 최적화이고, 이진 분류에서 `Logloss`(cross-entropy)는 예측이 정답에서 멀수록(확률이 극단으로 틀릴수록) 그래디언트가 커지는 반면 squared error(MSE/Brier)는 같은 상황에서 그래디언트가 오히려 평탄해지는 구간이 있어(0/1 근처) 최적화 지형이 더 나쁘다 — 로지스틱 회귀류가 확률 보정이 최종 목표여도 log-loss로 학습하는 이유이기도 하다. 이 프로젝트는 이미 `eval_metric="BrierScore"`로 early stopping은 Brier 기준으로 하면서 `loss_function="Logloss"`로 학습해 두 장점을 모두 취하고 있었던 셈이다. **결론: 현행 유지, 변경 없음.**

### 14.4 종합

이번 라운드에서 시도한 5개 후보(트랙맨 coarsening 3단계, 배터 강화, pitch-mix 교차, MLP loss/embedding 2종, CatBoost RMSE loss) 전부 baseline을 넘지 못했다. §12~§13(3번째 모델 추가)에 이어 이번에도 "직관적으로 그럴듯한 방향"이 실측에서는 대부분 손해였다 — 이 프로젝트의 CatBoost/MLP 조합은 이미 상당히 성숙한 로컬 최적점 근방에 있는 것으로 보인다. `TABULAR_MLP_REPORT.md` §9의 "트랙맨 피처 고도화" 항목은 이번 실험으로 종결하고, "수치형 피처 임베딩(periodic/quantile embedding)"은 여전히 미시도 상태로 남긴다.

## 15. 수치형 피처 주기함수(periodic) 임베딩 — 3-seed 스크리닝 (잠정 노이즈 수준, 재검증 예정)

§14.4에서 미시도로 남겨둔 "수치형 피처 임베딩"을 검증. 현재 `TabularMLP`는 수치형 116개 컬럼(트랙맨 결합 + 파생 피처 포함, 대회 원본 44개보다 많음)을 표준화만 해서 그대로 concat하는데, Gorishniy et al.("On Embeddings for Numerical Features in Tabular Deep Learning")의 PLR(Periodic-Linear-ReLU) 임베딩으로 대체해봤다: 피처마다 학습 가능한 주파수 `c_j ~ N(0, sigma^2)`로 `[sin(2*pi*c_j*x), cos(2*pi*c_j*x)]`(주파수 개수 k=8)를 만들고, 피처별 독립 Linear(비공유 가중치)로 d_embed=8차원 임베딩을 얻은 뒤 ReLU를 거쳐 concat — 범주형 임베딩과 동일한 역할을 수치형에도 부여하는 셈이다. 구현은 `code/periodic_mlp_model.py`(`PeriodicEmbedding`, `TabularMLPPeriodic`, `train_mlp_periodic`), 실험 스크립트는 `code/experiment_3way_stack.py`와 동일한 관례로 `code/experiment_periodic_embed.py` — 둘 다 `dopip.py` 메인 파이프라인에는 아직 미반영.

§14.3(loss function 스크리닝)과 같은 이유로 7-seed 풀 앙상블 대신 3-seed(`[42, 123, 7]`) 스크리닝으로 방향성만 먼저 확인. sigma(주파수 초기화 스케일)가 논문에서도 가장 성능을 좌우하는 하이퍼파라미터로 지목되어 그리드로 스윕:

| sigma | 3-seed 평균 | std | delta (baseline 677.06) |
| --- | --- | --- | --- |
| baseline(표준화만, PeriodicEmbedding 없음) | 677.06 | 23.69 | +0.00 |
| 0.001 | 705.51 | 11.21 | +28.45 |
| 0.003 | 661.42 | 30.81 | −15.64 |
| 0.01 | 693.26 | 28.72 | +16.20 |
| 0.02 | 702.69 | 46.33 | +25.63 |
| 0.05 | 681.90 | 13.77 | +4.84 |
| 0.1 | 680.79 | 27.13 | +3.73 |
| 1.0 | 289.66 | 220.17 | **−387.40** (붕괴, 시드 하나는 Val Score 0.00으로 학습 실패) |

**sigma=1.0의 붕괴는 명확한 신호다** — 주파수가 너무 크면 sin/cos 곡면이 지나치게 고주파로 요동쳐 최적화 지형이 깨지고(3개 시드 중 1개는 아예 학습 실패), 이는 논문에서 지적한 sigma 민감도가 그대로 재현된 것으로 해석.

**반면 sigma가 작은 구간(0.001~0.1)은 노이즈로 보인다.** 인접한 값끼리 매끄럽게 이어지지 않고 부호가 뒤집힌다 — 0.001(+28.45) → 0.003(**−15.64**) → 0.01(+16.20) → 0.02(+25.63) → 0.05(+4.84). 진짜 sigma 효과라면 이 정도로 촘촘한 그리드에서 바로 옆 지점끼리 부호가 반대로 나오기 어렵다. delta 크기(4~28점)도 시드 간 자연 변동폭(std 11~46, baseline 자체도 std 23.69)과 겹치는 수준이라, §14.3의 MLP loss/embedding 실험과 동일하게 "3-seed로는 진짜 개선인지 우연인지 구분 불가" 판정.

### 15.1 sigma=0.001, 7-seed 재검증 — 노이즈였음을 확인

가장 좋았던 sigma=0.001(3-seed delta +28.45, std 11.21로 그리드 내 가장 낮음)이 우연인지 재현되는 신호인지 확인하기 위해, `ENSEMBLE_SEEDS`(7-seed, `[42, 123, 7, 2024, 99, 555, 31337]`)의 나머지 4개 시드(`2024, 99, 555, 31337`)를 baseline과 sigma=0.001 양쪽에 추가로 학습해 7-seed 전체로 재비교:

| | 7-seed 평균 | std | delta |
| --- | --- | --- | --- |
| baseline(표준화만) | 688.14 | 23.35 | +0.00 |
| periodic sigma=0.001 | 692.29 | 28.02 | **+4.15** |

추가된 4개 시드 중 하나(31337)가 628.72로 크게 떨어지면서 3-seed 때의 +28.45가 +4.15로 주저앉았다 — baseline std(23.35)보다도 작은 크기라 완전히 노이즈 안에 파묻힌다. §14.3에서 우려했던 "3-seed 스크리닝만으로는 검정력이 약하다"는 경고가 정확히 재현된 사례.

**결론(최종): 미채택.** sigma=1.0급 고주파 임베딩은 명확히 위험(학습 붕괴)하고, sigma를 작게 잡아 논문의 "안전 구간"에 두더라도 이 데이터셋/아키텍처 조합에서는 baseline(수치형을 표준화만 해서 concat) 대비 유의미한 개선이 없다. `code/mlp_model.py::TabularMLP`는 변경하지 않는다. `code/periodic_mlp_model.py`/`code/experiment_periodic_embed.py`는 향후 다른 k/d 조합이나 quantile 임베딩 등을 시도할 때 재사용할 수 있도록 실험용으로만 남겨둔다.

### 15.2 quantile 기반 piecewise-linear 인코딩(PLE) — 대조 실험, 역시 노이즈 수준 (재개 예정)

periodic이 노이즈로 판정된 뒤, 같은 "수치형 피처 임베딩" 아이디어를 다른 인코딩으로 재검증. Gorishniy et al.의 Q-LR(Quantile 기반 piecewise-linear 인코딩 + 피처별 독립 Linear + ReLU) 변형을 구현 — `code/quantile_mlp_model.py`(`QuantileEmbedding`, `TabularMLPQuantile`, `train_mlp_quantile`), 실험 스크립트는 `code/experiment_quantile_embed.py`(`code/experiment_periodic_embed.py`의 `prepare_tensors`/`run_baseline`을 그대로 재사용). periodic의 "주파수 개수 k"에 대응하는 하이퍼파라미터는 "quantile 구간 개수 n_bins"로, 공정 비교를 위해 periodic과 동일하게 n_bins=8, d=8로 시작. 구간 경계는 트레인 스플릿에서만 quantile로 계산(리크 방지), 중복 경계(저카디널리티 피처)는 1e-6 epsilon으로 단조 증가를 보장.

이번엔 §15.1의 교훈을 반영해 3-seed 스크리닝 직후 바로 나머지 4-seed(`2024, 99, 555, 31337`)까지 채워 7-seed로 재검증:

| | 7-seed 평균 | std | delta | baseline 대비 시드별 승률 |
| --- | --- | --- | --- | --- |
| baseline(표준화만, §15.1과 동일) | 688.14 | 23.35 | +0.00 | — |
| quantile n_bins=8, d=8 | 696.31 | 17.31 | +8.17 | **3/7** (42, 2024, 555, 31337에서 패, 그중 31337은 −0.04로 사실상 동률) |

periodic(+4.15)보다는 delta가 크지만(+8.17) 여전히 baseline std(23.35) 안에 들어가고, 짝지은 시드별 비교에서도 과반(4/7)이 baseline 승리라 방향성이 뚜렷하지 않다. periodic과 마찬가지로 "노이즈 수준, 채택 근거 부족"에 가까운 상태.

**상태: 결론 보류, 재개 예정.** n_bins=8 한 지점만으로는 quantile 인코딩 자체를 완전히 기각하기엔 이르다고 판단해 세션을 일시 중단 — 다음 재개 시 n_bins 그리드(`code/experiment_quantile_embed.py --step sweep --binlist 4,16,32`, 이미 구현됨)로 이어서 확인할 예정. 지금까지 나온 패턴(periodic도 노이즈, quantile도 약한 신호)을 볼 때 그리드를 넓혀도 비슷한 결과일 가능성이 높다고 보지만, 확정 짓지 않고 이어서 검증하기로 함.

### 15.3 n_bins 그리드 재개 — n_bins=16~24에서 뚜렷한 신호 확인, n_bins=24 채택

§15.2를 재개해 n_bins 그리드(`code/experiment_quantile_embed.py --step sweep`)를 확장했다. 이번엔 baseline도 **같은 7-seed로 다시 학습**해서(저장된 평균값 재사용이 아니라) seed별 paired 비교(승/패)까지 가능하게 했다 — §15.1에서 "3-seed 스크리닝만으로는 검정력이 약하다"를 확인한 뒤로, 평균 delta만으로는 우연과 신호를 구분하기 어렵다는 게 이 프로젝트의 반복된 교훈이기 때문이다.

1차로 n_bins=16, 32를 스윕(baseline은 §15.2의 저장값 688.14 재사용)한 뒤, paired 승률 확인을 위해 baseline을 동일 7-seed로 재학습(680.89, std 17.35 — §15.2의 688.14와는 다른 실행이라 절대값 차이가 나지만 GPU 논디터미니즘 범위 안). 이 680.89 기준 seed별 실측값으로 n_bins=16/32를 다시 paired 비교하고, 8~16/16~32 사이를 좁히기 위해 n_bins=12, 24를 추가 스윕:

| n_bins | 7-seed 평균 | std | delta | baseline 대비 승률 |
| --- | --- | --- | --- | --- |
| baseline (재학습, paired) | 680.89 | 17.35 | +0.00 | — |
| 4 | 690.08 | 44.74 | +1.94 | 노이즈(승률 불명, std가 baseline보다 큼) |
| 12 | 702.31 | 28.57 | +21.42 | 5/7 (99, 31337에서 패) |
| **16** | 730.95 | 24.96 | **+50.06** | **7/7** |
| **24** | 736.67 | 23.49 | **+55.78** | **7/7** |
| 32 | 735.34 | 32.70 | +54.45 | 6/7 (555에서 −22.72로 크게 패) |

(n_bins=4/16/32는 §15.2와 동일 스크립트·동일 seed 순서로 학습해 baseline 재학습분과 정확히 짝지어 승패를 셀 수 있었다. n_bins=8은 §15.2 실행분만 있고 이번 paired baseline과 짝지어 재확인하지 않았다 — 표에서 제외.)

패턴이 뚜렷하다: n_bins=16~24 구간이 delta 크기(+50~+56, baseline std의 2~3배)·승률(7/7)·분산(std 23~25로 오히려 baseline보다 안정) 세 지표 모두에서 이전 periodic/quantile n_bins=8 실험과 확연히 다르다 — 그때는 델타가 baseline std 안에 묻히거나 승률이 절반 근처였다. n_bins=12는 다시 애매해지고(5/7, std 커짐) n_bins=32는 32.70으로 분산이 커지며 세부 seed(555)에서 크게 밀린다. 즉 "좋은 구간"은 16~24이고 그 안에서는 24가 delta·승률·std 세 지표 모두 최우수.

**결론: 채택.** n_bins=24, d=8(Q-LR, ReLU 포함)을 `code/mlp_model.py::TabularMLP`의 수치형 처리 기본 경로로 정식 반영했다 — `QuantileEmbedding`/`fit_quantile_edges`를 `code/mlp_model.py`로 옮기고, `TabularMLP`/`train_mlp`/`train_ensemble`/`predict_ensemble`/`make_bundle`/`predict_bundle`에 `bin_edges` 파라미터를 추가했다(주어지면 quantile 임베딩 경로, `None`이면 예전 raw-concat 경로 — `code/experiment_periodic_embed.py::run_baseline` 등 과거 임베딩 대조 실험의 raw-concat 기준선 재현용으로 하위 호환 유지). `code/train.py`와 `dopip.py`의 full retrain 단계 양쪽에서 학습 split(또는 전체 데이터)의 표준화된 수치형 값으로 `bin_edges`를 새로 fit해 사용하고, 번들에 `"bin_edges"`/`"quantile_d"` 키로 저장한다. `submit/script.py`에도 `QuantileEmbedding` 클래스와 갱신된 `TabularMLP`를 수동 동기화로 복제했다.

**주의(미해결 채로 남기는 부분)**: 이 채택 판단은 여전히 단일 시간 분할(season==2024 홀드아웃) 기준이다 — §9(스태킹 메타모델)처럼 rolling-origin 여러 fold로 재검증하지는 않았고, 실제 리더보드에도 아직 이 아키텍처로 제출한 적이 없다. 이 프로젝트에서 로컬 홀드아웃이 실제 성능을 과소평가해온 이력(TABULAR_MLP_REPORT.md §4)과 반대로, 소표본/단일분할 개선이 fold를 늘리면 사라진 사례(§13.1)도 있었던 만큼, 다음 `dopip.py` 전체 파이프라인 실행과 실제 제출 결과로 이 채택을 재확인할 필요가 있다.

### 15.4 전체 파이프라인 재확인 — CatBoost+MLP 블렌드 BSS 879.34로 승격

§15.3의 "미해결" 사항대로 실제 `dopip.py` 전체 파이프라인(개별 실험 스크립트가 아니라 `code/train.py` → `code/test.py` reference 비교 → full retrain)을 quantile n_bins=24가 반영된 코드로 실행해 재확인했다.

결과: 신규 블렌드 모델 BSS 0.00879 (환산 879.34) vs 기존 reference(알파 블렌드 시절 스태킹, 855.31) — **+24.03점**, 기존 reference를 넘어서 `open/reference/best_model.pkl`로 승격됨. 개별 MLP 시드는 이전과 마찬가지로 시드 간 편차가 크지만(단일 시드 Val Score 277~767 범위) 7-시드 앙상블 + CatBoost 블렌드 수준에서는 §15.3의 실험 스크립트 스크리닝 결과와 같은 방향(quantile 임베딩이 raw-concat 대비 우위)으로 재현됐다 — 실험 스크립트와 프로덕션 파이프라인(트랙맨 피처 포함 116개 수치형 컬럼, 전체 학습 파이프라인)이 서로 다른 코드 경로임에도 결론이 일치한다는 점에서 §15.3의 채택 판단에 대한 독립적인 재확인으로 본다.

Full retrain으로 `submit/model/final_retained_model.pkl`도 갱신됨. 다만 이 879.34는 여전히 로컬 season==2024 단일 홀드아웃 기준이며, TABULAR_MLP_REPORT.md §4에서 반복 확인된 대로 로컬 홀드아웃이 실제 리더보드 성능을 과소평가해온 이력이 있으므로, 실제 제출 결과로 다시 한번 검증이 필요하다 — 이 아키텍처로는 아직 실제 리더보드 제출 이력 없음.

### 15.5 실제 리더보드 제출 — 957.80417점, 로컬 과소평가 패턴 재확인

§15.4에서 만든 `submit.zip`(quantile n_bins=24 반영 블렌드)을 실제 대회 대시보드에 제출. 결과: **957.80417점** — 로컬 홀드아웃(879.34) 대비 +78.5점 높게 나왔고, 이 아키텍처 도입 전 마지막 실제 제출 기록인 924점(알파 블렌드 시절, CLAUDE.md 참고)보다도 +33.8점 높다.

로컬이 실제 리더보드를 과소평가해온 이 프로젝트의 반복 패턴(TABULAR_MLP_REPORT.md §4)이 이번에도 재현됐다 — 로컬 단일 시간분할(season==2024) 홀드아웃과 실제 평가 데이터(2025 시즌) 사이의 분포 차이가 매번 같은 방향(로컬이 낮게 나옴)으로 나타난다는 점은 이제 우연이라기보다 이 프로젝트의 검증 방식 자체의 구조적 특성으로 봐도 될 만큼 누적됐다. **결론: quantile (PLE) 수치형 임베딩 채택이 실제 리더보드에서도 확정적으로 검증됨.** 현재 프로덕션 모델 = CatBoost + (quantile embedding 적용) MLP 앙상블, 스태킹 메타모델 블렌드, 실제 대시보드 957.80417점.

## 16. 트랙맨 매칭 'season' 등호 문제 — 실제 제출에서 64개 파생 피처가 상시 0으로 죽어있었음, fallback으로 수정

957.80417점 제출 직후, 트랙맨 병합 로직을 다시 들여다보다 발견한 문제. `process_trackman_features_safe`의 `match_cols`(10개: `season`, `game_month`, `game_dayofweek`, `inning`, `top_bottom`, `balls_before`, `strikes_before`, `outs_before`, `pitcher_hand`, `batter_hand`)에 `season`이 포함돼 있는데, **실제 평가 데이터(`test.csv`)는 항상 season==2025이고 `trackman_history.csv`는 2019~2024만 있다.** `season` 등호 조건 때문에 실제 제출 시 매칭이 구조적으로 100% 실패한다.

로컬의 5행짜리 포맷 확인용 `test.csv`(season 전부 2025)로 직접 재현:

```
trackman season range: 2019 ~ 2024
test season values: [2025]
트랙맨-파생 피처 개수: 64
전체 test 행 5개 중 트랙맨 매칭 전부 실패(전체 NaN)한 행: 5개  (100%)
```

즉 실제 대시보드 제출(245,789개 샘플) 전체에서 64개 트랙맨 파생 피처가 전부 상수 0이었다(`submit/script.py`의 `fillna(0)` 방어 처리 덕에 크래시는 안 났지만, 해당 피처 블록이 완전히 죽은 채로 957.80417점이 나온 것).

**왜 로컬 검증이 못 잡았나**: 로컬 홀드아웃은 `season==2024`인데 트랙맨 데이터의 최대 시즌도 2024라서, 로컬 val 행은 실제로 매칭이 된다. 즉 지금까지의 모든 로컬 검증(§14.1의 트랙맨 fingerprint coarsening 실험 포함)은 "트랙맨이 val 시즌을 볼 수 있는" 조건에서 측정된 것이라, 실제 배포 조건(트랙맨이 eval 시즌을 원천적으로 볼 수 없음)과 다르다. §14.1의 "세분화가 이미 최적"이라는 결론 자체는 그 조건 안에서는 유효하지만, 실제 배포에서는 애초에 매칭이 안 되므로 그 결론이 실제 배포 성능에 얼마나 기여했는지는 별개 문제다.

### 16.1 로컬 시뮬레이션으로 fallback 효과 검증

`code/experiment_trackman_season_fallback.py`로 실제 배포 조건을 흉내냈다: 로컬 val(season==2024) 피처를 만들 때 `trackman_history.csv`에서 `season==2024` 데이터를 통째로 제거(1,793,078 → 1,458,852행, 남은 시즌 2019~2023)해 "트랙맨이 eval 시즌을 못 보는" 실제 상황을 재현하고, 재학습 없이 `open/reference/best_model.pkl`(CatBoost+MLP+메타모델)을 그대로 불러와 세 가지 조건을 비교했다:

| 조건 | 트랙맨 피처 전부 0인 행 비율 | CatBoost | MLP | Blend |
| --- | --- | --- | --- | --- |
| as-is (참고, 트랙맨이 2024 포함 — 지금까지의 모든 로컬 검증 조건) | 1.5% | 820.42 | 837.62 | **880.38** |
| broken (season 포함 10-key, 실제 2025 제출 상황 재현) | 100% | 711.27 | 735.43 | **730.25** |
| fallback (season 등호 실패 시 season 제외 9-key로 재매칭) | 2.9% | 670.35 | 731.92 | **752.42** |

- **broken vs as-is**: delta −150.14. 트랙맨이 eval 시즌을 못 보게 되는 것만으로 로컬 기준 150점 가까이 잃는다 — §14.1에서 매칭 키 1개만 성기게 해도(9-key, 여전히 실제 매칭은 됨) CatBoost 단독 -118.81이 났던 것과 같은 방향(트랙맨 파생 피처가 실제로 큰 비중을 차지한다는 뜻)이고, 완전히 죽이면 그보다도 더 크게 손해를 본다.
- **fallback vs broken**: delta **+22.17**. season을 뺀 9-key 재매칭이 97.1%의 행을 복구했고(2.9%만 잔여 미매칭), 시뮬레이션 기준으로 확실한 순이득을 보였다. season 등호 없이 5개 시즌(2019~2023)을 한데 묶어 집계하다 보니 as-is(같은 시즌만 매칭)만큼 정밀하진 않아 이론적 상한(+150.14) 전부를 회복하진 못했지만, 방향과 크기 모두 채택할 만한 수준.

**리크 안전성**: 이 fallback은 **추론 전용 코드(`submit/script.py`)에만** 적용했다 — `code/train.py::process_trackman_features_safe`(학습/검증 피처 생성)는 그대로 둔다. 평가 시점의 season(2025)은 `trackman_history.csv`가 커버하는 모든 시즌(2019~2024)보다 항상 미래이므로 season을 빼고 매칭해도 미래 정보가 섞일 위험이 없지만, 학습 시점에는 이전 시즌 행이 이후 시즌 트랙맨 데이터를 보게 되는 진짜 리크가 생길 수 있다(현재 `season` 등호가 사실상 유일한 시간 리크 방지 장치 — `is_train_split`의 글로벌 시간 필터는 이미 무력하다는 게 문서화돼 있었음, CLAUDE.md 참고). 그래서 재학습은 필요 없다 — `submit/model/final_retained_model.pkl`은 그대로 두고 `submit/script.py`의 추론 로직만 수정했다.

**결론: 채택.** `submit/script.py::merge_trackman_features`에 season fallback을 구현했다(`build_trackman_lookup` 헬퍼로 분리해 1차 10-key 매칭 실패 행에 한해 2차 9-key 매칭). 행 독립성(대회 규칙 핵심 원칙 — 어떤 행의 예측값은 다른 test.csv 행의 존재 여부와 무관해야 함)도 재검증했다 — fallback도 `trackman_history.csv`(공식 데이터)만 조회하는 룩업이라 여전히 행 단위로 독립적임을 "행 1개짜리 test.csv vs 전체 test.csv" 비교로 확인(최대 오차 1.4e-08, 부동소수점 수준). 모델 재학습 없이 `submit.zip`만 재빌드해 제출 가능. 실제 리더보드에서 이 fallback이 로컬 시뮬레이션(+22.17)만큼, 혹은 이 프로젝트의 "로컬이 실제를 과소평가" 패턴을 감안하면 그 이상 개선될 가능성이 있다 — 아직 실제 제출로 확정 검증되지 않음, 다음 제출로 확인 필요.

### 16.2 파생 문제 — "지금까지 모든 로컬 실험이 as-is(트랙맨 살아있는) 조건에서 나왔는데, 3rd 모델 스태킹(§12) 결론도 다시 봐야 하지 않나?"

§16의 발견 직후 나온 타당한 질문: §12/§12.2(LightGBM/XGBoost/RandomForest 3-way 스태킹, "프로덕션 미채택, 보류")를 포함해 이 프로젝트의 모든 로컬 실험이 트랙맨이 실제로 매칭되는 "as-is" 조건에서 측정됐다면, 실제 배포 조건(fallback)에서 결론이 뒤집히는 실험이 있지 않을까?

`code/experiment_3way_stack.py`에 `--val-mode {as-is, broken, fallback}`를 추가해(train_split은 그대로 — 학습 행은 season 등호로 항상 자기 시즌만 매칭되므로 val 시즌의 트랙맨 존재 여부와 무관) LightGBM/XGBoost/XGBoost(튜닝)/RandomForest를 두 축으로 재검증했다: (1) 트랙맨 조건(as-is vs fallback), (2) 레퍼런스 모델 버전(§12 당시의 알파블렌드 시절 vs 현재의 quantile-embedding 스태킹, §15.4). 두 축을 분리해야 "무엇 때문에 결론이 바뀌었는지" 오귀인하지 않는다.

| 3번째 모델 | 레퍼런스 | 트랙맨 조건 | 단독 Val | vs CatBoost | vs MLP | 2-way 대비 delta |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM | §12(구) | as-is | 697.51 | 0.921 | 0.865 | −1.03 |
| LightGBM | 현재 | as-is | 697.51 | 0.9210 | 0.8938 | **−10.04** |
| LightGBM | 현재 | fallback | 590.38 | 0.9007 | 0.8836 | −2.90 |
| XGBoost(미튜닝) | §12(구) | as-is | 729.46 | 0.915 | 0.855 | **+7.36** |
| XGBoost(미튜닝) | 현재 | as-is | 729.46 | 0.9152 | 0.8882 | **−2.91** |
| XGBoost(미튜닝) | 현재 | fallback | 604.33 | 0.8933 | 0.8770 | −0.88 |
| XGBoost(팀원 튜닝) | §12.2(구) | as-is | 744.38 | 0.934 | 0.871 | +6.29 (rolling −7.55) |
| XGBoost(팀원 튜닝) | 현재 | as-is | 744.38 | 0.9343 | 0.9034 | −4.31 |
| XGBoost(팀원 튜닝) | 현재 | fallback | 633.05 | 0.9119 | 0.8930 | −0.04 |
| RandomForest | §12(구) | as-is | 521.88 | 0.830 | 0.806 | −6.36 |
| RandomForest | 현재 | fallback | 420.53 | 0.8229 | 0.8130 | −6.05 |

(RandomForest는 학습이 7분 이상 걸리고 두 조건 모두 이미 뚜렷하게 마이너스라 "현재 레퍼런스 + as-is" 조합은 생략 — 값을 채워도 결론이 바뀔 가능성이 낮다고 판단.)

**결과를 두 축으로 분리해서 읽으면**:
- **레퍼런스 모델 축이 진짜 원인이었다.** §12에서 유일하게 순이득(+7.36)이었던 미튜닝 XGBoost가, 트랙맨 조건은 그대로(as-is) 두고 레퍼런스만 현재 버전(quantile embedding 채택 후, 2-way 879.34)으로 바꾸자 곧바로 마이너스(−2.91)로 뒤집혔다. 팀원 튜닝 XGBoost도 마찬가지(+6.29 → −4.31). 즉 **트랙맨 fallback과는 무관하게, quantile embedding으로 2-way 베이스라인 자체가 더 강해지면서 3rd 모델이 비집고 들어갈 여지가 이미 줄어들어 있었다.**
- **트랙맨 조건(fallback) 축은 오히려 반대 방향으로 움직였다.** 같은 현재 레퍼런스 기준으로 as-is→fallback을 비교하면 모든 delta가 0에 더 가까워졌다(LightGBM −10.04→−2.90, XGBoost −2.91→−0.88, 튜닝 XGBoost −4.31→−0.04) — fallback 조건에서 트랙맨 신호가 흐려지면서 CatBoost/MLP의 우위도 함께 줄어든 것으로 보이나, 어느 경우도 플러스로 전환시키지는 못했다.
- **결론은 바뀌지 않았고, 오히려 더 단단해졌다.** 네 가지 조합(레퍼런스×트랙맨조건) 전부에서 3-way 스태킹은 마이너스이거나 사실상 0(−0.04)이다. §12에서 "결론 미확정, 열어둠"으로 남겨뒀던 유일한 근거(미튜닝 XGBoost의 +7.36)가 지금은 두 축 중 어느 쪽으로 봐도 재현되지 않는다.

**최종 결론: §12/§12.2의 "3번째 모델 계열 추가, 프로덕션 미채택" 판단을 유지하며, 이번 재검증으로 더 확실하게 닫는다.** 트랙맨 매칭 버그는 이 결론을 뒤집지 않았다 — 오히려 원래 열려 있던 유일한 여지(XGBoost)가 트랙맨과 무관한 다른 이유(2-way 베이스라인 자체의 개선)로 이미 닫혀 있었음을 확인한 셈. `code/experiment_3way_stack.py --val-mode`는 향후 트랙맨 신뢰도가 바뀌는 다른 실험에도 재사용 가능하도록 남겨둔다.

## 17. 트랙맨 최종 결론 — ablation·asof9key 재설계 끝에 완전 드롭 확정

§16의 fallback 패치(752.42)로 일단 봉합한 뒤, "애초에 트랙맨을 학습에 넣지 않았다면?" 질문을 검증했다.

### 17.1 ablation — 트랙맨 피처를 학습에서 통째로 제거

`code/experiment_trackman_ablation.py`: `process_trackman_features_safe` 호출 자체를 생략하고 CatBoost+MLP를 처음부터 재학습(season==2024 홀드아웃, F1 필터 미적용).

| 조건 | CatBoost | MLP | Blend |
| --- | --- | --- | --- |
| broken (§16, season skew로 트랙맨 전부 0) | 711.27 | 735.43 | 730.25 |
| fallback (§16, season 실패시만 9-key 재매칭) | 670.35 | 731.92 | 752.42 |
| **ablation (트랙맨 피처 자체를 제거)** | **693.74** | **765.14** | **795.13** |
| as-is (참고, 실제 불가능) | 820.42 | 837.62 | 880.38 |

**트랙맨을 아예 빼는 게 fallback 패치보다 +42.71점 더 좋았다.** broken(전부 0)보다도 낫다 — 원인은 학습/서빙 분포 불일치(train-serve skew): 학습 시점엔 season이 항상 정확히 매칭되는 "날카로운" 분포로 트랙맨 피처를 학습하는데, 서빙 시점엔 항상 season 매칭에 실패해 9-key로 재매칭된 "뭉뚱그려진" 분포를 보게 되어, 모델이 학습한 것과 실제로 받는 입력의 의미가 어긋난다. "일부라도 매칭시켜주자"는 patch가 오히려 역효과였던 것.

### 17.2 asof9key — 학습도 서빙과 같은 분포로 맞추면?

skew의 근본 원인이 "학습=season-exact, 서빙=season-drop"의 비대칭이라면, 애초에 학습 때도 서빙과 같은 방식으로 매칭하면 되지 않을까. `code/experiment_trackman_asof9key.py`: 매칭 키에서 season을 아예 빼고(9-key), 대신 각 행의 season 이하(≤)로 컷오프한 트랙맨 데이터만 사용하는 asof 누적 방식으로 재설계했다(리크 없음 — 이후 시즌 정보를 보지 않음). 검증 행(season==2024)과 실배포 행(season==2025) 둘 다 "컷오프=트랙맨 전체(2019~2024)"로 수렴하므로, §16처럼 트랙맨에서 특정 시즌을 인위적으로 지우는 블라인드 트릭도 필요 없다.

season==2024, F1 미적용 기준: **asof9key 804.04** (CatBoost 724.10, MLP 780.23) — ablation(795.13) 대비 **+8.91**. 트랙맨 병합 방식 중에는 이게 최선으로 나왔다.

### 17.3 F1 필터 도입 후 2023 홀드아웃 재검증 — 시즌 평균 사실상 0, 드롭 확정

§18의 F1 필터를 채택하기로 하면서, season==2024 단일 홀드아웃의 신뢰성 문제(§18.1)를 감안해 asof9key도 F1 필터 + 2023 홀드아웃으로 다시 봤다.

| | 2023 (F1=ON) | 2024 (F1=ON) | 평균 |
| --- | --- | --- | --- |
| no-trackman | 550.15 | 775.57 | 662.86 |
| asof9key | 521.04 | 801.99 | 661.52 |
| Δ (asof9key − no-trackman) | **−29.11** | **+26.42** | **−1.35** |

같은 F1 조건·같은 2023 홀드아웃에서 asof9key가 no-trackman보다 **−29.11 더 나쁘다** — §17.2에서 봤던 2024·F1=OFF의 +8.91 우위가 정확히 뒤집힌다. 시즌별로는 ±26~29로 크게 흔들리지만 2023+2024 평균은 사실상 0(−1.35)으로 수렴한다. 개별 시즌 결과는 노이즈였고, 트랙맨(어떤 매칭 방식이든)에는 실제로 안정적인 신호가 없다는 뜻이다.

이 결론은 팀원이 독립적으로 검증한 3가지 트랙맨 접근(상황 지문 조인 −40.4, 투수 ID 기반 통산 물리량 −38~−46, 경기 중 실시간 상태 상관 0.014 — §18 참고)과도 정확히 수렴한다.

**최종 결론: 트랙맨은 프로덕션에서 완전히 드롭한다.** `code/train.py`의 `process_trackman_features_safe` 호출과 함수 자체를 제거하고, `submit/script.py`의 `merge_trackman_features`/`build_trackman_lookup`과 `trackman_history.csv` 로드도 제거했다. 남은 것: `top_bottom`을 T/B→0/1로 매핑하는 로직만 (CatBoost가 `game_type`/`base_state`만 범주형으로 선언하고 `top_bottom`은 수치형으로 기대하기 때문 — 예전에는 이 매핑이 트랙맨 병합 함수 안에 있었는데, 병합 함수가 사라지면서 `main()`으로 옮겼다).

## 18. 팀원 제보 — F1 필터(2022 이하 시즌 game_type=='F' 제거) 채택

팀원이 독립적으로 진행한 검증에서 두 가지를 제보받았다: (1) 이 프로젝트의 검증 방법론(season==2024 단일 홀드아웃) 자체의 맹점, (2) 그 맹점을 2023 홀드아웃으로 추가 검증하다 발견한 `game_type` F(퓨처스/2군)/R(1군) 관계의 시즌별 역전과 그 수정(F1 필터). 리포트 원문은 크로스워크 복원(개막전 투구수 대조로 팀 매핑, 신뢰도 0.9999)까지 포함한 상당히 엄밀한 방법론으로 작성되었다.

### 18.1 검증 방법론 문제 — season==2024 단일 홀드아웃의 맹점

리포트 주장: season==2024만으로 검증하면 그 해의 리그 기준선 변동폭(r, 즉 그 시즌의 평균 성공률)에 점수가 지배당한다. 동일 모델을 season==2023으로 재검증하면 완전히 붕괴(팀원 리포트 기준 0.0점)한다는 것.

### 18.2 F/R 성공률 관계 역전 — 우리 데이터에서 재현 확인

우리 `train.csv`로 직접 재현했다 (팀원 수치와 소수점까지 일치):

| season | F | R | F−R |
| --- | --- | --- | --- |
| 2019 | 0.6893 | 0.5495 | +14.0%p |
| 2020 | 0.5878 | 0.5269 | +6.1%p |
| 2021 | 0.7038 | 0.5128 | +19.1%p |
| 2022 | 0.7087 | 0.5037 | +20.5%p |
| **2023** | **0.4729** | 0.5031 | **−3.0%p** |
| **2024** | **0.4593** | 0.4897 | **−3.0%p** |

2022년까지 F가 R보다 최대 +20.5%p 높다가, 2023년부터 −3.0%p로 역전되어 2024까지 유지된다 — 일회성이 아니라 새 상태로 정착한 것으로 보인다. `game_type`은 CatBoost 피처 중요도 1위(팀원 리포트 기준 25.4%)라 이 역전이 학습에 미치는 영향이 크다.

**조치 (팀원 제안, `code/train.py::apply_f1_filter`로 구현)**: `train = train[~((train['game_type'] == 'F') & (train['season'] <= 2022))]`. 학습 데이터에만 적용하고 검증/추론 데이터는 원본을 유지한다. 2023년 이후 F는 새 관계가 유효하므로 남긴다 — 팀원 리포트에 따르면 F를 전량 제거하는 대안은 2024에서 -105점 손해를 보였다. 가중치로 희석하는 방식(2023 이후 F에 2배 가중)도 실패했다(조기 종료가 첫 트리에서 멈춤) — 오염 구간은 제거가 유일한 해법.

### 18.3 우리 아키텍처로 직접 검증

팀원 리포트는 우리의 최신 아키텍처(quantile embedding MLP, 트랙맨 제거)를 기준으로 작성된 게 아니므로, 값을 그대로 믿지 않고 `code/experiment_f1_filter.py`로 우리 코드 위에서 재현했다(트랙맨은 §17의 결론에 따라 이미 제거된 상태로 테스트 — F1 변수만 격리).

| 홀드아웃 | F1 | CatBoost | MLP | Blend |
| --- | --- | --- | --- | --- |
| 2024 | OFF | 693.74 | 765.14 | 795.13 |
| 2024 | ON | 689.35 | 761.61 | **775.57** |
| 2023 | OFF | 12.54 | 0.00 | 72.24 |
| 2023 | ON | 543.71 | 513.65 | **550.15** |

| | F1 OFF | F1 ON | Δ |
| --- | --- | --- | --- |
| 2023 | 72.24 (거의 붕괴) | 550.15 | **+477.91** |
| 2024 | 795.13 | 775.57 | −19.56 |
| **평균** | 433.69 | **662.86** | **+229.18** |

팀원 리포트가 예측한 패턴과 정확히 일치한다 — **2024만 보면 F1이 오히려 손해(-19.56)처럼 보이지만, 2023의 붕괴를 압도적으로 만회하면서 평균은 크게 개선된다(+229.18).** 2023의 F1=OFF 결과(72.24)는 §18.1이 경고한 "리그 기준선 변동폭에 지배당하는 검증"의 실물 증거이기도 하다 — 모델 품질이 아니라 그 해 F/R 역전이라는 데이터 오염이 점수를 결정하고 있었다.

**결론: 채택.** `code/train.py`, `dopip.py`(전체 재학습 단계)의 학습 데이터 로드 직후에 F1 필터를 적용한다. `code/test.py`/`submit/script.py`(검증·추론 데이터)는 수정하지 않는다.

## 19. 트랙맨 드롭 + F1 필터 — 프로덕션 파이프라인 반영

§17.3(트랙맨 완전 드롭)과 §18.3(F1 필터 채택)을 `code/train.py`, `code/test.py`, `dopip.py`, `submit/script.py`에 실제로 반영하고 `dopip.py` 전체 파이프라인을 재실행했다.

**작업 중 발견한 버그**: `code/test.py`가 신규 모델을 기존 reference와 비교할 때, 두 모델의 dict 키 구성(`catboost_model`/`mlp_bundle`/`meta_model`)만 확인하고 그 안의 실제 피처 스키마(컬럼 구성)가 호환되는지는 확인하지 않았다. 트랙맨 피처 제거로 피처 집합 자체가 바뀌자, 기존 reference의 CatBoost가 `predict_proba` 단계에서 컬럼 불일치 예외를 던지며 파이프라인 전체가 죽었다. 기존에 있던 "레거시 포맷은 비교 건너뛰기" 분기와 같은 맥락으로, reference 스코어링을 try/except로 감싸 스키마 불일치도 "비교 불가 → 신규 모델 채택"으로 처리하도록 수정했다.

**실행 결과**: F1 필터가 학습 데이터에서 1,221,585 → 1,116,277행(105,308행 제거)으로 적용됨. season==2024 검증 기준 블렌드 **777.61**(w_cat=2.017, w_mlp=2.287, intercept=-2.177) — §18.3의 독립 실험값(775.57)과 거의 일치해 정상 재현 확인. 기존 reference는 스키마 불일치로 자동 스킵되어 신규 모델이 그대로 승격됨. 전체 재학습(F1 필터 적용 후 1,369,784행)으로 `submit/model/final_retained_model.pkl` 갱신 완료, `submit.zip` 재구성 후 5행짜리 샘플 `test.csv`로 스모크 테스트 통과(정상 범위의 확률 출력, 트랙맨 관련 에러 없음).

이 777.61은 여전히 season==2024 단일 홀드아웃 기준이라는 한계가 있지만(§18.1의 문제 제기 자체가 이 방식의 신뢰성에 의문을 던진 것이므로), 트랙맨 드롭·F1 필터 각각의 순효과는 2023+2024 두 시즌 평균으로 이미 별도 검증됐다(§17.3, §18.3). 실제 리더보드 재제출로 최종 확인 필요 — 아직 제출 전.

## 20. 2023+2024 검증 컨벤션 검토 — 감사용 도구로만 유지, F1 필터와의 충돌로 프로덕션 미반영

§18.1에서 제기된 "season==2024 단일 홀드아웃이 신뢰할 수 없다"는 문제를, 실제로 `code/train.py`/`code/test.py`/`dopip.py`의 기본 스플릿 자체를 `train<2023` / `val∈{2023,2024}`로 바꿔서 해결할지 검토했다.

**결론: 하지 않는다.** 진짜 2023+2024 듀얼 홀드아웃을 만들려면 season 2023을 학습에서 완전히 빼야 하는데(그래야 2023 검증이 실제 홀드아웃이지 in-sample 평가가 아니게 됨), 그러면 남는 학습 구간(2019~2022)에 F1 필터(`season<=2022`인 F 행 제거)를 적용하는 순간 **F 데이터가 학습에서 전량 사라진다** — 이는 §18.2에서 이미 F1보다 못하다고 확인된 "F2(전량 제거)" 방식과 동치가 된다. 즉 "검증을 제대로 하려고 2023을 떼어내면" "F1 필터가 애초에 지키려던 바로 그 2023 데이터를 잃는" 딜레마가 생긴다. F1 필터의 이득(2023+2024 평균 +229.18, §18.3)이 검증 방식 개선의 이득보다 훨씬 크고 이미 실증됐으므로, 프로덕션 스플릿은 `train<2024`/`val==2024`를 유지한다.

대신 2023+2024 검증은 **개별 후보 변경을 판단할 때 쓰는 감사 도구**로 남긴다 — 이번 세션에서 트랙맨(§17.3)과 F1 필터(§18.3) 자체를 검증할 때 이미 이 방식으로 썼다. `code/experiment_f1_filter.py --holdout {2023,2024}`, `code/experiment_trackman_asof9key.py --holdout {2023,2024}`가 이 용도의 재사용 가능한 인프라다. `code/test.py`의 자동 승격 판단(`NEW_BEST`/`KEEP_REF`)은 여전히 season==2024 단일 기준이라는 한계를 안고 간다는 뜻이지만, 트레이드오프를 인지한 채로 내린 의도적 선택이다.

## 21. CatBoost 재튜닝 — F1 필터로 학습 데이터가 바뀐 뒤 Optuna 재탐색, +32.31 (단독), +17.16 (블렌드)

기존 `code/catboost_model.py::CATBOOST_PARAMS`는 F1 필터도 트랙맨 제거도 없던 훨씬 이전 시점의 Optuna 탐색 결과였다. F1 필터로 학습 데이터에서 105,308행(전체의 약 8.6%)이 빠졌으니, 최적 하이퍼파라미터도 달라졌을 가능성을 검증했다.

`code/tune.py`를 현재 프로덕션 피처 구성(트랙맨 없음, F1 필터 적용, season<2024 학습/season==2024 검증)에 맞게 갱신하고 — 기존에는 삭제된 `process_trackman_features_safe`를 import하고 있어 그대로는 실행 불가능한 상태였다 — Optuna 40 trials(TPE sampler)를 재실행했다.

| | 이전 (구 데이터 기준 튜닝) | 신규 (F1 필터 적용 후 재튜닝) | Δ |
| --- | --- | --- | --- |
| CatBoost 단독 (season==2024) | 689.35 | **721.66** | **+32.31** |

새 최적 파라미터(`depth=6, learning_rate=0.0275, l2_leaf_reg=2.361, border_count=32, random_strength=9.996, bagging_temperature=0.466, min_data_in_leaf=89` — `min_data_in_leaf`는 이전 탐색 공간에 없던 신규 파라미터, best_iteration=761)를 `CATBOOST_PARAMS`에 반영하고 `dopip.py` 전체 파이프라인을 재실행했다.

**결과**: season==2024 블렌드 **794.77**(w_cat=2.119, w_mlp=2.366, intercept=-2.259) — 직전 reference(777.61, §19) 대비 **+17.16**. 정상적으로 새 reference로 승격, 전체 재학습(CatBoost 811 iteration = best_iteration 761 + 버퍼 50) 완료, `submit.zip` 재구성 및 5행 샘플 스모크 테스트 통과. 아직 2023+2024 듀얼 검증이나 실제 리더보드로는 확인하지 않았다 — 다음 제출 기회에 확인 필요.

## 22. 실제 리더보드 제출 — 971점, 트랙맨 드롭+F1 필터+CatBoost 재튜닝 종합 확인

§21의 `submit.zip`(트랙맨 완전 제거 + F1 필터 + 재튜닝된 CatBoost)을 실제 대회 대시보드에 제출. 결과: **971점** — 이 프로젝트 역대 최고 실제 제출 점수로, 이 아키텍처 이전의 마지막 제출 기록인 957.80417점(§15.5, 트랙맨 버그가 숨어있던 quantile-embedding 버전)보다도 +13.2점 높다.

로컬 season==2024 단일 홀드아웃(794.77)만 보면 §15.5의 879.34보다 오히려 낮아 보이지만, §17.3/§18.3/§19에서 반복 확인했듯 이 로컬 수치는 이번 변경들의 진짜 효과를 보여주는 지표가 아니다(트랙맨 드롭·F1 필터의 순효과는 season==2024 단일 지표로는 과소평가되거나 방향이 뒤집혀 보일 수 있다는 게 §18.1 이후 이 세션 전체의 핵심 발견이었다). 실제 리더보드가 오히려 더 높게 나온 것은, 로컬이 실제 성능을 과소평가해온 이 프로젝트의 반복 패턴(TABULAR_MLP_REPORT.md §4, §15.5)과도, 그리고 트랙맨 버그가 실제로 죽어있던 64개 피처였다는 §16의 발견과도 일치한다 — 이번 제출은 그 죽은 피처들을 걷어내고 F1 필터로 오염 구간까지 정리한 뒤 나온 결과이므로, 971점은 트랙맨/F1/재튜닝 세 변경의 종합 실사용 효과를 실제 데이터로 확인한 것으로 본다.

## 23. 트랙맨 재도입 4차 시도 — '배경 상황(context)' 뭉뚱그린 집계, 2023+2024 평균 사실상 0 (재확인, 드롭 유지)

§17에서 트랙맨을 완전히 드롭한 뒤, 사용자가 근본적으로 다른 설계를 제안했다: 정밀 매칭 대신 **볼/스트라이크/아웃카운트 + 시즌구간(초/중/후반) + 좌우타/좌우투**로만 뭉뚱그려 그룹핑하고, 그룹별로 [rel_speed, spin_rate, induced_vert_break, horz_break, extension, rel_height, rel_side, zone_speed] 8개 물리 지표와 신규 피처 `stint_pitch_no`(경기 내 투수 교체 시마다 1로 재시작하는 등판 내 투구수, "이 상황이 보통 몇 구째쯤 나오는가"라는 피로도 근사치)의 mean/std를 냈다(`code/experiment_trackman_context.py`).

이전 시도들(§14 상황 지문 조인, §17.2 asof9key 9-key 정밀 매칭)과의 핵심 차이:
- 시즌을 정확한 `game_month`가 아니라 3구간(early=3~5월/mid=6~8월/late=9~11월)으로 뭉개, test.csv(항상 2025)가 trackman_history의 2019~2024 중 아무 시즌에서나 자연스럽게 매칭되게 함 — season 등호 skew(§16) 문제가 애초에 구조적으로 생기지 않도록 설계.
- `pitch_type_group`/`auto_pitch_type`로 피벗하지 않고 상황 전체를 하나로 묶어 집계(§17.2는 64개 컬럼으로 피벗해 상황별 표본을 잘게 쪼갰음).
- trackman_history.csv에 `base_state`/`runner_on_*` 컬럼이 없어 주자상황은 애초에 그룹핑 키에 넣을 수 없었다(데이터에 없는 정보).
- season asof 컷오프(학습 행은 season<=그 행의 season, 검증/배포는 season<=2024)는 §17.2와 동일하게 유지해 미래 정보 leak 방지.

동일한 재튜닝된 CatBoost 파라미터·F1 필터로 컨텍스트 피처 유무만 격리해 2023/2024 양쪽 홀드아웃으로 검증했다(`--no-context` 플래그로 기준선 재현):

| | 2024 ctx=OFF | 2024 ctx=ON | Δ | 2023 ctx=OFF | 2023 ctx=ON | Δ | 평균 Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost | 721.66 | 763.98 | **+42.32** | 541.91 | 535.94 | −5.97 | **+18.18** |
| MLP | 759.80 | 780.39 | +20.59 | 545.85 | 513.49 | **−32.36** | −5.89 |
| Blend | 792.93 | 809.75 | +16.82 | 558.97 | 538.33 | **−20.64** | **−1.91** |

2024만 보면 CatBoost·MLP·Blend 전부 뚜렷한 개선처럼 보이지만(+16.82~+42.32), 2023에서 전부 역전된다(−5.97~−32.36). 이번에도 §17.3에서 봤던 것과 정확히 같은 패턴 — 서로 다른 3번째 설계(정밀 조인 → 9-key asof → 뭉뚱그린 context 집계)로 3번 반복 확인했으니 개별 시즌 결과가 노이즈라는 결론이 더 굳어진다. 최종 채택 지표인 **Blend의 2023+2024 평균 Δ는 −1.91로 사실상 0** — §17.3의 asof9key(−1.35)와 같은 수준.

흥미로운 점 하나: CatBoost 단독 평균 Δ는 +18.18로 순양이다(MLP만 −5.89로 순음, 블렌드를 깎아먹음). 즉 CatBoost는 이 컨텍스트 피처에서 뭔가 durable한 걸 뽑아내고 MLP는 못 뽑아내는 것처럼 보인다. 그러나 (a) 현재 파이프라인은 두 모델이 피처셋을 공유하는 구조라 모델별로 다른 피처셋을 주려면 구조 변경이 필요하고, (b) +18.18은 이 프로젝트에서 반복 관찰된 시즌 간 변동폭(±20~40)에 비해 두드러지게 크지 않다 — 그래서 이번 세션에서는 추가로 파고들지 않고 기록만 남긴다.

**결론: 드롭 유지.** 프로덕션 파이프라인은 변경하지 않는다. `code/experiment_trackman_context.py`는 실험 스크립트로 보관.

**결론**: 트랙맨 완전 제거, F1 필터, CatBoost 재튜닝 모두 실제 리더보드에서 최종 확정됐다. 현재 프로덕션 모델 = CatBoost(재튜닝) + MLP 앙상블(quantile embedding) 스태킹 블렌드, 트랙맨 미사용, F1 필터 적용, 실제 대시보드 971점.

## 24. 세 번째 모델(attention) 스크리닝 — FT-Transformer/ExcelFormer, 단일 시드는 유망했으나 7-seed 앙상블에서 노이즈로 판명 (드롭)

CatBoost(GBDT)+MLP(concat) 2-way 스태킹에 구조적으로 다른 계열(피처-간 self-attention)을 3번째로 추가하면 이질성이 늘어 스태킹 이득이 있을지 스크리닝했다(사용자가 다른 AI의 추천표를 근거로 FT-Transformer/ExcelFormer/AMFormer 등을 제안, 둘 다 구현해 비교하기로 결정).

### 24.1 구현

- `code/ft_transformer_model.py`: FT-Transformer(Gorishniy et al. 2021). 범주형은 `nn.Embedding(card+2, d_token)`, 수치형은 `code/mlp_model.py::QuantileEmbedding`과 동일한 PLE 로직이지만 flatten하지 않고 `(batch, num_numeric, d_token)`으로 유지하는 `NumericTokenizer`. 학습 가능한 [CLS] 토큰 + 표준 Transformer 인코더(pre-norm, GELU FFN, d_token=32, 3 layer, 8 head) + `Linear(d_token,1)+Sigmoid`.
- `code/excelformer_model.py`: ExcelFormer(Chen et al. 2023) **근사** 구현. 논문 전체(특히 beta-mixup 증강, AFI 상호작용 레이어)는 재현하지 않고, 핵심 아이디어인 semi-permeable attention만 근사했다 — CatBoost `get_feature_importance()`로 피처 중요도 랭킹을 매기고, 토큰 i는 자신/CLS/자신보다 같거나 중요한 토큰에만 attend하도록 가산(additive) 마스크를 씌운다(정보가 "덜 중요 → 더 중요" 방향으로만 흐르게 제한해 oversmoothing 억제).
- 둘 다 `code/mlp_model.py`의 전처리 유틸(`fit_preprocessing`/`apply_preprocessing`/`to_tensors`/`fit_quantile_edges`)을 그대로 재사용, TabularMLP와 동일한 early-stopping(Val Brier, patience 7) + 시드 앙상블(`ENSEMBLE_SEEDS`) 패턴을 따른다.

### 24.2 버그 두 개 (구현 과정에서 발견/수정)

1. **`code/experiment_3way_stack.py`의 죽은 import**: 트랙맨 제거 세션(§17) 때 삭제된 `process_trackman_features_safe`를 여전히 import하고 있어 그 파일의 `fit_meta_model_n`(N-피처 스태킹 메타모델)을 재사용하려는 순간 `ImportError`. `add_engineered_features`만 남기고 정리.
2. **어텐션 모델의 검증/추론이 배치 없이 통째로 forward되는 버그**: `train_ft`/`train_excel`이 매 epoch 검증 시 `X_val_cat`/`X_val_num` 전체(holdout=2024면 253,507행)를 한 번에 forward했다. self-attention의 forward 메모리는 O(batch·heads·seq²)로 스케일해서, seq=60(1 CLS+7 cat+52 num) 기준 253,507행을 한 번에 넣으면 이론상 약 29GB가 필요하다 — 마스크 없는 FT-Transformer는 이 GPU(로컬 8GB)에서 우연히 PyTorch의 fused/flash 커널 경로를 타 통과했지만, 커스텀 `attn_mask`를 쓰는 ExcelFormer는 느린 경로로 빠지며 실제로 `Tried to allocate 27.20 GiB` OOM이 났다. **실전 배포(24.6만 행 test.csv 추론)에서도 똑같이 터질 수 있는 잠재 버그**였다. `code/ft_transformer_model.py::batched_forward()`(청크 단위 `torch.no_grad()` forward)를 추가해 두 모델의 검증/앙상블 추론(`predict_ft_ensemble`/`predict_excel_ensemble`) 모두에 적용해 수정.

### 24.3 단일 시드 스크리닝 (season==2024)

같은 조건(F1 필터 ON, 트랙맨 미사용, 재튜닝 CatBoost)에서 solo 성능과 기존 두 모델과의 예측 상관계수를 봤다(`code/experiment_attention.py`).

| 모델 | solo 점수 | corr(vs CatBoost) | corr(vs MLP) |
| --- | --- | --- | --- |
| CatBoost | 721.66 | — | — |
| MLP(7-seed) | 769.23 | — | — |
| FT-Transformer(1 seed) | 589.35 | 0.8714 | 0.8309 |
| ExcelFormer(1 seed) | 538.21 | 0.8739 | 0.8321 |

두 attention 모델 다 solo 성능이 CatBoost/MLP보다 130~230점 낮다. 상관계수는 CatBoost-MLP 자체 상관(~0.999, §6)보다는 뚜렷이 낮아 구조적 독립성은 있어 보였다.

### 24.4 3-way 스태킹 이득 — 단일 시드는 두 시즌 모두 양수(트랙맨과 다른 패턴)

`code/experiment_3way_stack.py::fit_meta_model_n`으로 [CatBoost, MLP, attention] 3-피처 로지스틱 회귀를 매 홀드아웃에서 새로 학습해 2-way 대비 이득을 측정했다. (초기 구현은 MLP 비교 대상으로 프로덕션 reference bundle을 재사용했는데, holdout=2023에서는 그 bundle이 이미 season==2023을 학습에서 본 상태라 in-sample 평가가 되어 MLP=1403.42라는 비정상 값이 나왔다 — MLP도 매 holdout 전용으로 7-seed 새로 학습하도록 수정 후 재실행.)

| 모델 | 2024 Δ | 2023 Δ | 평균 Δ |
| --- | --- | --- | --- |
| FT-Transformer(1 seed) | −1.14 | +8.55 | +3.71 |
| ExcelFormer(1 seed) | +3.39 | +25.56 | **+14.48** |

트랙맨 실험들과 달리 부호가 두 시즌 모두 양수였다(FT도 +3.71, ExcelFormer는 +14.48로 CatBoost 재튜닝의 블렌드 이득(+17.16, §21)과 맞먹는 규모). 유망해 보여 ExcelFormer를 7-seed 앙상블로 확장해 재검증했다.

### 24.5 7-seed 앙상블 재검증 — 이득이 홀드아웃 간 부호가 뒤집힘, 노이즈로 판정 (드롭)

| | 2024 Δ | 2023 Δ | 평균 Δ |
| --- | --- | --- | --- |
| ExcelFormer 1-seed | +3.39 | +25.56 | +14.48 |
| ExcelFormer 7-seed | **+25.63** | **−20.53** | +2.55 |

2024는 이득이 오히려 더 커졌지만(+3.39→+25.63), 2023은 완전히 뒤집혔다(+25.56→−20.53, 46점 차) — solo 점수는 이 홀드아웃에서 앙상블로 실제 개선됐음에도(323.89→424.42) 스태킹 이득은 정반대로 갔다. 앙상블 평균(+2.55)은 단일 시드 평균(+14.48)의 5분의 1 수준으로 쪼그라들었고, 이 프로젝트에서 반복 관찰된 시즌 간 변동폭(±20~40) 안에 들어간다 — 즉 신호가 아니라 노이즈일 가능성이 높다.

**추정 원인**: ExcelFormer의 예측이 CatBoost/MLP와 상관계수 0.88~0.93으로 이미 높은 상태에서, 3-피처 로지스틱 회귀를 시즌 하나짜리 검증셋(24.5만~25.3만 행)에 매번 새로 피팅한다. 상관이 높은 피처들의 회귀 계수는 다중공선성 때문에 표본에 따라 부호까지 흔들릴 수 있다 — 실제로 h2023 1-seed 실행에서는 ExcelFormer 가중치가 음수(−1.72, 오차 보정 신호)였는데 7-seed 실행에서는 양수(+1.19)로 바뀌었다. 즉 attention 모델 자체의 품질 문제라기보다, "이미 상관 높은 3번째 피처를 작은 검증셋 하나로 스태킹"하는 절차 자체가 원래 불안정하다는 뜻으로 보인다.

**결론: 드롭.** FT-Transformer/ExcelFormer 둘 다 프로덕션에 편입하지 않는다. `code/ft_transformer_model.py`, `code/excelformer_model.py`, `code/experiment_attention.py`는 실험 스크립트/재사용 가능 인프라로 보관한다(버그 수정된 `batched_forward`는 향후 어텐션 계열을 다시 시도할 때도 유효).

### 24.6 메타모델 검증 방식 자체를 고쳐 재검증 — 정규화(LogisticRegressionCV) + rolling-origin 3-fold, 그래도 드롭 재확인

§24.5의 원인 분석("ExcelFormer 예측이 CatBoost/MLP와 상관 0.88~0.93으로 이미 높은 상태에서, 시즌 하나짜리 검증셋에 3-피처 로지스틱 회귀를 매번 새로 피팅하니 다중공선성으로 계수 부호까지 흔들린다")이 맞다면, 검증 절차 자체를 고치면 숨어 있던 진짜 신호가 드러날 수도 있다는 가설을 세워 재검증했다.

**코드 변경**:
- `code/experiment_3way_stack.py::fit_meta_model_n`: `LogisticRegression()`(C=1.0 고정) → `LogisticRegressionCV(cv=5, Cs=10)`로 교체, 정규화 강도를 내부 5-fold CV가 직접 고르게 함.
- `code/experiment_attention.py`: `--foldcheck` 옵션 추가(`run_foldcheck()`) — `code/experiment_3way_stack.py::step_foldcheck`/§9.1과 동일한 `FOLD_BOUNDARIES=[(5,6),(7,8),(9,10)]` 월 버킷 확장 윈도우로, season==2024 안에서 2-way vs 3-way 스태킹을 재검증한다(시즌 2개짜리 point-check보다 표본이 많은 out-of-sample 검증).

**결과** (`python -m code.experiment_attention --model excel --holdout 2024 --ensemble --foldcheck`, F1 필터 ON, EXCEL 7-seed 학습 6184.8s):

| 모델 | solo 점수 | corr(vs CatBoost) | corr(vs MLP) |
| --- | --- | --- | --- |
| CatBoost | 721.66 | — | — |
| MLP(7-seed) | 755.36 | — | — |
| ExcelFormer(7-seed) | 538.76 | 0.9165 | 0.8429 |

2-way(CatBoost+MLP) Blend=782.42, 3-way(season 전체로 메타모델 한 번 피팅)=789.30(+6.88, weight_excel=−0.434) — 다만 이 +6.88은 메타모델을 season==2024 전체로 학습하고 같은 season==2024로 평가한 것이라 사실상 인샘플에 가깝다(§24.5가 겪었던 것과 같은 함정 구조).

rolling-origin 3-fold(진짜 out-of-sample — 각 fold는 그 fold 이전 월들로만 메타모델을 재학습):

| val 구간 | n_train | n_val | 2-way | 3-way(+EXCEL) | delta |
| --- | --- | --- | --- | --- | --- |
| 5~6월 | 54,949 | 88,592 | 945.78 | 804.83 | **−140.95** |
| 7~8월 | 143,541 | 74,990 | 752.36 | 719.02 | −33.35 |
| 9~10월 | 218,531 | 34,976 | 326.82 | 329.70 | +2.88 |

**1/3 fold 승리, 평균 delta −57.14.**

**결론**: 다중공선성을 완화하는 정규화(`LogisticRegressionCV`)를 적용하고, 시즌 2개짜리 point-check 대신 진짜 rolling-origin fold로 검증해도 3-way 스태킹 이득은 살아나지 않고 오히려 뚜렷하게 마이너스로 나온다. "검증 절차의 불안정성이 진짜 신호를 가렸을 수도 있다"는 가설은 기각됐다 — 절차를 고쳤더니 오히려 더 명확하게 손해라는 게 드러났다. **드롭 결정 재확인, 최종.**

## 25. 팀원의 독립적 트랙맨 3종 재검증 상세 — 릴리스 산포(재현성) 격리 테스트까지 포함, 전부 실패 재확인

§17.3/§18/§23에서 이미 여러 각도로 트랙맨 드롭을 확정했는데, 사용자가 "trackman의 `rel_speed`/`spin_rate`/release point 등 물리 지표, 특히 그 표준편차(재현성)는 `asof_*` 결과-기반 피처에는 없는 새로운 정보 아니냐"고 이의를 제기했다. 검증 과정에서 두 가지가 드러났다.

### 25.1 이 리포지토리 안의 관련 코드는 애초에 무효였다

`code/train_x30.py`/`code/train_rd30.py`/`code/train.last.py`의 `build_pitcher_consistency_features`는 정확히 이 아이디어(투수·구종군별 `horz_break`/`induced_vert_break`/`extension`/`rel_speed`의 std)를 구현한 코드였지만, `pitcher_trackman_id`로 집계한 컬럼을 `'pitcher_id'`로 개명한 뒤 메인 데이터의 진짜 `pitcher_id`와 그대로 `pd.merge`한다. 직접 확인한 결과 두 ID 공간은 완전히 분리돼 있다(메인 792개, 범위 20700~24633 / 트랙맨 906개, 범위 50008~71775155, **overlap 0**) — 이 merge는 항상 전부 NaN이었을 것이고, 이 피처의 단독 결과가 `EXPERIMENTS.md`/`PROJECT_HISTORY.md` 어디에도 기록돼 있지 않았던 이유가 이거였다. 즉 이 리포지토리 자체 코드로는 "재현성" 가설이 한 번도 유효하게 테스트된 적이 없었다.

### 25.2 팀원이 별도 크로스워크로 실제로 테스트했다 — 산포까지 포함, 격리해도 실패

팀원은 값 겹침이 아니라 **시퀀스 재구성** 방식으로 유효한 선수 단위 크로스워크를 만들었다(`work/pitcher_map.csv`, 팀원 로컬 보관):
1. `(season, game_month, game_dayofweek, 홈팀, 원정팀)` 5개 키로 경기 후보를 잡고, 1:1로 유일하게 대응되는 경우만 채택(88.7%).
2. 짝지어진 두 경기의 총 투구수(행 수)가 정확히 일치하는지 대조(596경기 중 505경기, 84.7% 통과).
3. `trackman_history.csv`는 저장 순서가 무작위지만 `pitch_no`로 정렬하면 시간순이 복원된다는 점을 이용해, 정렬 후 같은 위치의 `inning`/`balls_before`/`strikes_before`/`outs_before` 시퀀스 전체가 일치하는지 검증(505경기 전부 일치, 불일치 0건 — 300여 구의 볼카운트 흐름이 우연히 통째로 일치할 확률은 사실상 0이므로 결정적 증거).
4. 통과한 경기들에서 같은 위치의 `pitcher_id`/`pitcher_trackman_id`를 다수결로 누적(다수 득표 비율 0.90 이상, 총 100구 이상만 채택). 팀명(정수 ID ↔ 트랙맨 문자열)도 같은 원리(투구수 대조)로 확정.

**결과: 확정 매핑 493명, 확신도 평균 0.9999(490명은 1.0), 1:1 중복 0건, 커버리지 투수 기준 62% / 행 기준 약 67%.** 외부 데이터 없이 두 공식 CSV의 컬럼만으로 복원한 것이라 대회 규칙 위반도 아니다.

이 유효한 크로스워크로 3가지 접근을 테스트했다(F1 필터 적용, 2023/2024 홀드아웃):

**접근 1 — 투수별 통산 물리 특성(18개 피처: 릴리스 산포·구속·회전수 등)**

| | 2023 | 2024 | 평균 | Δ |
| --- | --- | --- | --- | --- |
| 기준선(F1) | 545.2 | 638.5 | 591.8 | — |
| trackman 전체(18개) | 473.7 | 618.8 | 546.2 | **−45.6** |
| **릴리스 관련만(산포 포함)** | 492.2 | 615.1 | 553.6 | **−38.2** |

**바로 이 "릴리스 관련만" 행이 사용자가 제기한 가설의 격리 테스트다** — speed/spin 등 나머지를 빼고 release point/release spread(산포=표준편차 기반 재현성 지표)만 남겨도 여전히 뚜렷한 손해다. 전체 18개(−45.6)보다는 덜 나쁘지만 기준선 대비 마이너스라는 결론은 그대로다.

**진단**: feature importance 합계 15.01%로 모델이 활발히 사용했지만 점수는 하락했고, 중요도 1위는 `tm_speed_mean`(평균 구속)이었다. 팀원 해석: 이건 제구력 지표가 아니라 **투수 개인 식별자**에 가깝다 — 한 투수의 시즌 평균 구속·산포는 경기마다 거의 고정된 값이라, 모델이 "이 투구가 제구될 확률"이 아니라 "이게 누구 투구인가"를 알아내는 데 이 피처를 쓴다는 것. 그런데 그 정체성 정보는 이미 `asof_pitcher_success_rate`가 실제 결과 이력이라는 훨씬 정확한 형태로 담고 있어서, 물리 지표는 같은 정보의 더 노이즈 낀 버전이 된다. 재현성(산포) 계열도 결국 투수 개인의 고유 특성이라 같은 함정에 빠지는 것으로 보인다.

**접근 2 — 경기 중 실시간 상태** ("오늘 릴리스가 평소와 다른가"를 겨냥, 통산 지표로는 못 담는 축)

`groupby().transform(lambda s: s.expanding().mean().shift(1))`로 각 투구 시점 직전까지만 참조하는 누수 방지 설계(구현 중 `expanding().shift(1)`을 그룹 밖에서 적용하면 경기 경계를 넘어 이전 경기 값이 유입되는 버그를 팀원이 발견·수정했다). 860,214투구 결합, 443,379행 분석:

| 지표 | 상관 |
| --- | --- |
| today_extension_mean | 0.0137 |
| today_spin_rate_mean | −0.0118 |
| today_bauer_mean | −0.0100 |
| today_release_spread | 0.0083 |
| (비교) `asof_pitcher_success_rate` | 0.0846 |

최고 상관 0.0137은 `asof_pitcher_success_rate`(0.0846)의 6분의 1 수준 — 유의미한 신호 없음.

**접근 3 — 상황 지문 기반 조인(`trm_n_*`)**: 이 팀이 기존에 시도한 것과 동일 계열, −40.4 (§14.1 재확인).

### 25.3 종결 판정

| 접근 | 결과 |
| --- | --- |
| 상황 지문 기반 조인 | −40.4 |
| 투수 ID 기반 통산 물리량(산포 포함, 격리해도) | −38.2 ~ −45.6 |
| 경기 중 실시간 상태 | 상관 0.014 |

3가지 독립 접근 모두 실패, **재현성(산포) 가설도 격리 테스트에서 기각됐다.** 크로스워크 파일(`work/pitcher_map.csv`, 팀원 보관)은 향후 다른 각도가 필요하면 재사용 가능하지만, 트랙맨 활용 자체는 이 리포지토리와 팀원 양쪽에서 최종 종결로 본다.

### 25.4 부수 발견 — asof 시차 보정 시도도 실패, 선발/불펜 구분 피처도 실패 (참고)

팀원이 별도로 확인: `asof_pitcher_success_rate`는 누적 지표라 F/R 역전(§18.2)을 뒤늦게 반영한다(2023년 실제 F 성공률 0.4729 vs asof가 나타내는 값 0.5450, 오차 +0.072 / 2024년 오차 +0.052). 이를 명시적으로 보정하는 피처(`asof` 지표를 `season × game_type` 평균 대비로 재계산)를 만들었으나 **−39.8**(2023 460.3 / 2024 549.0, 기준선 498.9/589.9 대비)로 오히려 가장 큰 손해였다 — F1 필터가 이미 오염의 근본 원인(2022 이하 시즌 F 데이터)을 제거했기 때문에, 보정 피처는 중복 노이즈로 작용한 것으로 해석된다. F1 필터 자체의 정당성을 다시 한번 뒷받침하는 방증이다.

같은 라운드에서 `is_late_inning`/`inning × skill`/`reliever_proxy`로 만든 **선발/불펜 구분 피처도 −7.0**(2023 486.6 / 2024 588.1)로 손해였다 — 사용자가 별도로 제안했던 "선발/중계/마무리 역할 피처" 아이디어의 실현 가능한 근사 버전 하나가 이미 팀원 쪽에서 테스트되고 기각된 셈이다.

팀원 리포트의 종합(F1 필터 위에 얹은 파생·상호작용 피처 20종 시도 — 전패, trackman 결합 3종 — 전패, 데이터 정제(F1 필터) 1종 — +270.5/리더보드 +18)은 이 프로젝트 자체의 반복 패턴(§14.4 "직관적으로 그럴듯한 방향이 실측에서는 대부분 손해")과 독립적으로 수렴한다 — 타겟과의 최대 상관이 0.0846(asof_pitcher_success_rate)인 데이터에서는 새 신호를 만들어내기보다 기존 신호의 분산을 줄이거나 오염을 제거하는 쪽(F1 필터, CatBoost 재튜닝, 시드 앙상블)이 실제로 통하는 방향이었다는 게 두 독립 트랙 모두에서 재확인된 결론이다.

## 26. 2023 fold 붕괴 최종 분해 — R은 완전히 안정적, F가 100% 원인, BSS가 r≈0.5에서 유독 가혹해지는 이유

§18.1/§18.2에서 "season==2024 단일 홀드아웃은 그 시즌 r에 지배당한다"는 문제를 짚었지만, "그럼 2023 fold가 왜 그렇게까지 나쁜가"는 F1 필터 도입 이후에도 완전히 분해되지 않은 채 남아 있었다. 팀원이 별도로 13단계 조사를 진행했다:

1. 2023 전체 성공률 하락(52.9%→50.0%)은 완만한 추세일 뿐 급격한 단절이 아니라 붕괴를 설명 못 함.
2. 신규 투수 유입 증가 없음, 신규/기존 투수 성공률도 비슷 — 선수 구성 문제 아님.
3. 최근 2~3년만 학습해도 2023 성능 거의 개선 안 됨 — 오래된(2019~2020) 데이터 탓 아님.
4. **`game_type=='F'` 성공률만 유독 붕괴**: 2022 70.9% → 2023 47.3% (R은 50.4%→50.3%로 그대로) — §18.2의 우리 자체 수치(2022 0.7087→2023 0.4729)와 정확히 일치.
5. 같은 투수가 F/R을 둘 다 던진 경우만 비교해도 F−R 격차가 2022 +24.8%p → 2023 +2.3%p → 2024 +1.1%p로 붕괴 — "못하는 투수가 F로 몰려서"가 아니라 **F라는 카테고리 자체의 효과가 사라진 것.**
6. F 데이터 수집 구조(투구 수, 월별 구성, 팀 수, F/R 겸업 비율, 트랙맨 매칭률)엔 2023에 특이사항 없음.
7. 다른 `asof_*`/상황 변수 중 F처럼 뚜렷하게 관계가 뒤집힌 건 거의 없음 — 2023 전체가 이상한 게 아니라 `game_type` 하나가 특히 강하게 흔들림.
8. 스트라이크존 관련 이력(ball/strike/middle)은 2022→2023 큰 변화 없는데 F 실제 성공률만 23.6%p 하락 — 리그 전체 스트라이크존 변화만으로는 설명 안 됨.
9~11. F1 필터(2019~2022 F 제거) 실험, 베이스(44피처) vs 현재(68피처) 모델 비교, 2×2(베이스/현재 × F1 유무) 비교 — F1 필터 효과는 베이스 모델부터 이미 존재(베이스 F1: 2023 8.5→490.0, 현재 F1: 2023 5.4→591.3). 추가된 트랙맨/파생 피처가 2023 문제를 새로 만든 게 아니라 원래부터 있던 문제라는 것도 확인.
12. **1차 분해에서 나온 "2023 Brier 손실의 89.5%가 R에서 나온다"는 결과** — 처음엔 "F만으로 2023 붕괴를 설명 못 한다, R도 무너진 것 아니냐"는 가설로 이어졌으나(이 세션에서도 이 지점을 짚어 재확인을 요청했다), 후속 확인 결과 **손실 비중이 정확히 표본 비중(R 89.5% data / F 10.5%)과 같았을 뿐인 산술적 인공물**이었다.
13. F 관련 파생 상호작용 피처(`f_x_reverse`, `f_x_success`, `f_x_prev3_success`, `f_x_middle` = `is_f × asof_pitcher_*`) 4종 시도 — **기각.** 정확한 수치는 남아있지 않지만, F1 필터 적용 이후 F−R 성공률 격차가 2023부터 이미 +2.3%p/+1.1%p로 거의 0에 수렴해(위 5번) `is_f`가 최근 시즌에서는 거의 정보가 없는 플래그가 돼버린다는 점, 그리고 이미 §14.2(상황 교차 피처 −39.06)·§25.4(asof 시차 보정 −39.8)에서 반복 확인된 "기존 신호끼리의 명시적 곱은 CatBoost엔 중복 정보로 작용" 패턴과 일치해 기각이 타당해 보인다.

### 최종 확인 — R을 절대 지표로 격리하면 완전히 안정적

12번의 "R 89.5%" 결과가 진짜 R의 이상 신호인지, 표본 비중에 따른 당연한 결과인지 확인을 요청한 결과:

> "R만 따로 보면, 2023의 절대 예측 정확도(Brier)는 2022/2024와 거의 동일해요(0.2485/0.2499/0.2481). R 자체는 레짐 시프트의 영향을 거의 안 받았어요. relative_score가 낮아 보이는 건 2023의 실제 성공률이 정확히 0.5에 가까워서 baseline_brier(분모)가 최대치라 그런 계산상의 특성이고, 손실 집중도로 보면 R/F 둘 다 정확히 표본 비율(1.00배)만큼만 손실을 만들어요. 즉 2023 fold 전체 붕괴는 순전히 F(relative_score=0, baseline보다도 못 맞힘) 때문이고, R은 완전히 안정적이었어요."

**R의 절대 Brier(0.2485/0.2499/0.2481)는 세 시즌 내내 사실상 동일 — R은 레짐 시프트의 영향을 전혀 받지 않았다.** 2023 fold 전체가 나빠 보이는 건 100% F 때문이고, "R도 무너진 게 아닌가"는 표본 비중=손실 비중이라는 산술적 착시였다.

**메커니즘**: `baseline_brier = r(1-r)`은 r=0.5에서 최댓값을 갖는다. 2023의 전체 성공률이 마침 0.5 근방이라 분모(baseline_brier)가 최대가 되고, 같은 절대 오차(Brier)라도 `BSS = 100000×(1 - Brier/baseline_brier)` 공식상 상대 점수가 더 낮게 환산된다 — 절대 예측 품질이 나빠서가 아니라, 그 시즌의 r이 마침 BSS가 가장 가혹해지는 지점에 있었기 때문이다. 이건 §18.1("season==2024 단일 홀드아웃은 그 시즌 r에 지배당한다")의 일반적 경고를 훨씬 정밀하게 만든다 — 단순히 "시즌마다 r이 달라서 불안정하다"가 아니라, **r이 0.5에 가까운 시즌/서브셋일수록 절대 성능 변화 없이도 BSS가 구조적으로 더 낮게 나온다**는 구체적 메커니즘이다.

**결론**: 2023 fold 붕괴는 F 하나로 완전히 설명되고, F1 필터가 정확히 그 원인을 겨냥한 올바른 수정이었다는 게 재확인됐다. R 쪽에 남은 미해결 문제는 없다.

## 27. MLP 전용 서브피처 스크리닝 — §14.2의 14개 교차 피처, MLP에서도 손해 (기각)

§14.2("asof_batter_* 강화" 7개 + "pitch-mix 교차" 7개, 총 14개)는 CatBoost 단독으로만 스크리닝됐고(-43.89 / -39.06), MLP에서는 한 번도 테스트되지 않았었다. "CatBoost는 트리 분기로 상호작용을 스스로 찾아내니 명시적 교차항이 중복이지만, MLP는 층으로 상호작용을 근사해야 하니 오히려 도움이 될 수 있다"는 가설(§23의 "CatBoost 평균 Δ+18.18/MLP −5.89로 갈렸다"는 힌트에서 출발)을 `code/experiment_mlp_subfeature.py`로 검증했다.

원본 스크립트가 리포지토리에 없어 이름으로부터 재구성했고(정확한 수식 미상), 트랙맨 의존으로 보이는 `fastball_rate_x_situational_speed`/`breaking_rate_x_situational_break` 2개는 트랙맨이 §25에서 완전히 닫혔으므로 순수 상황 변수(`li`/`num_runners_on`) 기반으로 대체했다. `code/experiment_attention.py::build_split`(트랙맨 없음, F1 필터 적용, season==2024 홀드아웃)을 재사용하고, §14.3/§15와 동일하게 3-seed(`[42, 123, 7]`) MLP 스크리닝으로 방향성만 먼저 확인했다.

| variant | Val Score |
| --- | --- |
| baseline | 744.21 |
| +subfeatures(14개) | 730.78 |

**Delta: −13.43.** CatBoost(−39~−44)보다는 덜 나쁘지만 방향은 동일하게 마이너스다. "MLP는 상호작용을 스스로 못 찾으니 명시적 교차항이 유리할 것"이라는 가설은 기각됐다 — 이 14개 피처 조합 자체가 (CatBoost/MLP 모델 종류와 무관하게) 신호가 없거나 기존 피처와 중복이라는 뜻으로 보인다. 7-seed 재검증 없이 여기서 기각. **결론: MLP 서브피처로도 채택하지 않는다.**

## 28. MLP에서 기존 피처(`add_engineered_features` 12개)를 빼보는 반대 방향 — 3-seed에서는 유망, 7-seed+2023 홀드아웃에서 완전히 뒤집힘 (기각)

§27이 "MLP에 뭘 더 주면 도움이 될까"였다면, 이번엔 반대로 "MLP에서 뭘 빼면 오히려 나아질까"를 검증했다 — 이 프로젝트에서 실제로 통했던 유일한 성공 패턴(새 신호 추가가 아니라 기존 신호의 분산/노이즈 축소, F1 필터가 그 실례, §25/§26/§27 결론 참고)과 같은 방향이라는 문제의식에서 출발했다. 후보는 `code/train.py::add_engineered_features`의 12개(`pitcher_recent1_gap`/`pitcher_recent3_gap`/`pitcher_recent5_gap`/`pitcher_relative_success`/`count_diff`/`is_full_count`/`pitcher_count_advantage_raw`/`pitcher_count_advantage_rel`/`pitcher_trend`/`pitcher_consistency`/`matchup`/`count_pressure`) — §14.2가 CatBoost 기준으로 "일부는 기존 피처의 단조변환이라 새 정보가 거의 없다"고 진단했던 바로 그 그룹.

평가 방식도 개선했다: solo MLP 점수만 보지 않고, **CatBoost(고정, 두 변형에서 항상 동일 피처)와 스태킹한 블렌드 점수**를 직접 비교하도록 `code/experiment_mlp_subfeature.py::run_removal_blend`를 작성했다 — 프로덕션이 실제로 최적화하는 지표는 solo가 아니라 블렌드이기 때문.

**1차: 3-seed 스크리닝, holdout=2024**

| | MLP solo | Blend |
| --- | --- | --- |
| baseline | 730.89 | 773.83 |
| −engineered(12) | 746.13 | 787.44 |
| Δ | **+15.25** | **+13.60** |

solo·블렌드 둘 다 뚜렷하게 개선 — 이번 세션에서 처음으로 양쪽 다 확실히 좋아진 결과였다.

**2차: 7-seed 재검증(`ENSEMBLE_SEEDS`), holdout=2024**

| | MLP solo | Blend |
| --- | --- | --- |
| baseline | 764.33 | 790.75 |
| −engineered(12) | 773.92 | 788.27 |
| Δ | +9.59 | **−2.47** |

시드를 7개로 늘리자 solo Δ는 축소(+15.25→+9.59)됐고 **블렌드 Δ는 부호가 뒤집혔다(+13.60→−2.47)**. §22(ExcelFormer)에서 이미 확인된 "시드 앙상블이 분산을 줄이면 스태킹 이득의 부호까지 바뀔 수 있다"(핵심 교훈 #14)가 여기서도 그대로 재현됐다.

**3차: 7-seed, holdout=2023 (듀얼 홀드아웃 감사)**

| | MLP solo | Blend |
| --- | --- | --- |
| baseline | 530.28 | 555.07 |
| −engineered(12) | 504.67 | 539.68 |
| Δ | **−25.61** | **−15.39** |

2023은 처음부터 뚜렷하게 마이너스.

**종합 (2023+2024 평균, 7-seed)**:

| | MLP solo Δ | Blend Δ |
| --- | --- | --- |
| 2024 | +9.59 | −2.47 |
| 2023 | −25.61 | −15.39 |
| **평균** | **−8.01** | **−8.93** |

3-seed·단일 시즌(2024)에서만 보였던 +13.60은 완전한 착시였다 — 시드를 늘리고 2023을 같이 보니 평균 −8.93으로 뚜렷하게 마이너스다. §17.3/§18.1/§22/§23/§26에서 반복된 "단일 시즌·적은 시드 결과를 과신하지 말라"는 패턴이 이번엔 "피처 추가"가 아니라 "피처 제거" 방향에서도 그대로 재현됐다. **결론: `add_engineered_features` 12개는 MLP에서도 그대로 유지한다. 제거하지 않는다.**

## 29. CatBoost에서도 같은 12개 피처 제거 — 대칭 실험, 평균 −2.58로 역시 기각

§28이 "MLP에서 빼보기"였다면 이번엔 반대로 **CatBoost에서 `add_engineered_features` 12개를 빼고 MLP는 고정**한 채 블렌드 점수를 비교했다(`code/experiment_catboost_subfeature.py`). §14.2가 CatBoost에 새 피처를 "추가"만 테스트했지 기존 피처를 "제거"하는 방향은 한 번도 테스트되지 않았던 빈틈. CatBoost는 `random_seed=42` 고정이라 MLP 같은 시드 분산 문제가 없어(§28처럼 3-seed→7-seed로 부호가 뒤집힐 위험이 구조적으로 작음) 처음부터 2024/2023 두 홀드아웃을 모두 확인했다(MLP는 3-seed로 고정, 비교 기준용).

| holdout | CatBoost solo Δ | Blend Δ |
| --- | --- | --- |
| 2024 | −8.83 | **−6.61** |
| 2023 | +2.92 | **+1.45** |
| **평균** | **−2.96** | **−2.58** |

시즌별로 부호가 갈렸지만(2024 마이너스, 2023 소폭 플러스) 평균은 −2.58로 마이너스 쪽이다. §28(MLP 제거, 평균 −8.93)만큼 크진 않지만 방향은 같다 — **CatBoost 쪽도 이미 지금 피처셋이 최적에 가깝다는 뜻.** 모델별 서브피처 분리(§23에서 시작된 문제의식) 축에서 "제거" 방향은 MLP·CatBoost 양쪽 다 기각으로 마무리됐다. **결론: 현행 피처셋 유지, 양쪽 모델 모두 변경 없음.**

## 30. 진짜 다중시즌 OOF 스태킹 — 실행은 했지만 F1 필터와 fold 설계가 구조적으로 충돌해 무효, 이 데이터셋에서는 애초에 성립 불가능함을 확인

이 세션 초반에 논의만 하고 "비용이 커서 3번째 모델을 편입하기로 결정되면 투자할 규모"라며 보류했던 방향 — 3번째 모델이 전부 닫힌(§22~24) 뒤 실제로 구현·실행했다(`code/experiment_oof_stacking.py`). `code/experiment_attention.py::build_split(val_season, apply_f1=True)`를 재사용해 expanding-window fold(train 2019-2020→val 2021, train 2019-2021→val 2022, train 2019-2022→val 2023)마다 CatBoost+MLP(7-seed)를 새로 학습해 OOF 예측을 모으고, 세 fold를 합친 74만 행짜리 메타 학습셋에 메타모델을 피팅한 뒤, 프로덕션급(season<2024 학습) CatBoost+MLP의 season==2024 예측에 그 메타모델을 그대로 적용해 기존 방식(season==2024 자체에 메타모델 피팅)과 비교했다.

**1차 결과 (표면상)**:

| fold | CatBoost | MLP |
| --- | --- | --- |
| val=2021 | 602.67 | **0.00** |
| val=2022 | 909.72 | 648.92 |
| val=2023 | 541.91 | 533.37 |

OOF 메타모델: `w_cat=6.747, w_mlp=-1.999`(부호 반전, 이 프로젝트에서 나온 적 없는 극단값) — 이걸 2024에 적용하니 Blend 586.72로 기존 방식(784.64) 대비 **−197.92.**

**원인 진단 — fold=2021의 MLP가 완전히 붕괴(0.00)한 게 이상해서 직접 확인**: F1 필터(`season<=2022`인 `game_type=='F'` 행 제거)와 fold의 학습 구간이 구조적으로 충돌한다.

| holdout(val) | train 시즌 | F1 필터 전 F행 | F1 필터 후 F행 | val F행 |
| --- | --- | --- | --- | --- |
| 2021 | 2019-2020 | 48,999 | **0** | 25,861 |
| 2022 | 2019-2021 | 74,860 | **0** | 30,448 |
| 2023 | 2019-2022 | 105,308 | **0** | 25,686 |
| 2024(프로덕션) | 2019-2023 | 130,994 | 25,686 | 30,010 |

**세 fold 전부 학습 데이터에 `game_type='F'` 행이 정확히 0개였다** — F1 필터가 "season≤2022인 F행 제거"인데, 세 fold의 학습 구간이 전부 2022 이하 시즌으로만 이루어져 F가 통째로 사라진다(오직 프로덕션 fold(2024 검증)만 학습에 2023을 포함해 F가 일부 남는다). MLP의 `game_type` 임베딩이 훈련 중 한 번도 보지 못한 카테고리를 검증에서 2만5천~3만 행씩 마주쳐 미학습(사실상 무작위) 임베딩으로 예측한 것 — `season`을 임베딩하면 안 되는 이유(핵심 교훈 #2, §3.1/§7)와 정확히 같은 메커니즘이 이번엔 `F1 필터 × game_type` 조합에서 재현됐다. CatBoost는 카테고리 처리 방식이 달라(임베딩이 아니라 target-statistic 기반) 상대적으로 덜 무너졌지만 그래도 눈에 띄게 낮았다.

**결론**: `−197.92`라는 수치는 "OOF 스태킹 자체가 나쁘다"는 증거가 아니라 fold 설계 버그의 산물이다. 더 중요한 발견은, **F1 필터와 양립하는 fold가 이 데이터셋(2019~2024)에서는 사실상 프로덕션 fold(2024 검증) 하나뿐**이라는 것 — 학습 구간에 2023을 포함해야 F가 안 사라지는데, 2023을 포함하면서 2024보다 이전인 검증 시즌은 존재하지 않는다. 즉 "여러 개의 다양한 fold로 큰 메타 학습셋을 만든다"는 OOF 스태킹의 핵심 전제 자체가 이 데이터·이 F1 필터 조합에서는 구조적으로 성립하지 않는다. F1 필터를 fold별로 다르게(예: 그 fold의 val 시즌 기준으로 컷오프 조정) 조정하는 우회로도 생각해볼 수 있지만, 그러면 각 fold가 서로 다른 데이터 처리 규칙으로 학습돼 "동일한 파이프라인의 OOF"라는 전제가 깨진다. **재설계 비용 대비 기대 이득이 불확실해 여기서 더 투자하지 않는다. 프로덕션의 현재 2-way 메타모델(season==2024 단일 홀드아웃 피팅)을 유지한다.**

## 31. MLP 하이퍼파라미터 재탐색 — quantile embedding 도입 후 한 번도 재검증 안 된 lr/weight_decay/dropout/batch_size, Optuna로 재탐색

`code/mlp_model.py`의 `LR=0.003`/`WEIGHT_DECAY=0.01`/`DROPOUT=0.3`/`BATCH_SIZE=4096`는 §3.1에서 스윕된 값인데, 그 스윕은 수치형 피처를 표준화만 해서 concat하던 구조(quantile embedding 도입 이전)에서 나온 결론("lr/wd 조정 무의미")이었다. §13/§15에서 수치형 처리를 PLE(quantile embedding)로 완전히 바꾼 뒤로는 재검증된 적이 없다 — CatBoost가 F1 필터 도입 후 재튜닝해 +32.31을 얻었던 것(§21)과 같은 상황. `code/tune_mlp.py`(Optuna, TPESampler)로 lr/weight_decay/dropout/batch_size 4개를 재탐색했다(아키텍처는 고정 — HIDDEN1/HIDDEN2/QUANTILE_D/n_bins 불변).

### 31.1 3-seed 스크리닝 (20 trials, holdout=2024)

Best trial(010): `lr=0.008696, weight_decay=0.008255, dropout=0.488, batch_size=2048` — **Score 809.97**, baseline 3-seed 기준값(730~765대, 실행마다 변동)보다 뚜렷하게 높음. 저성능 트라이얼(lr<0.001)은 576~705로 큰 폭 하회 — lr을 높이고(0.003 초과) dropout을 높이고(0.4~0.5) batch_size를 2048로 낮추는 방향이 상위권에 일관되게 나타남(단일 우연 트라이얼이 아니라 20개 트라이얼 전반의 경향).

### 31.2 7-seed + 블렌드, 2024/2023 듀얼 홀드아웃 재검증

`code/experiment_mlp_hparam_confirm.py`로 최적 파라미터를 7-seed 앙상블 + CatBoost 블렌드로 재확인했다(CatBoost는 각 holdout에서 고정, 두 variant 모두 동일 피처 사용).

| holdout | MLP solo Δ | Blend Δ |
| --- | --- | --- |
| 2024 | **+39.62** | **+28.23** |
| 2023 | **−23.01** | **−10.44** |
| **평균** | **+8.31** | **+8.90** |

2024는 3-seed보다 오히려 델타가 커졌다(baseline 790.46 → tuned **818.69**, 현재 프로덕션 로컬 기준 794.77(CLAUDE.md)보다도 +23.92). 하지만 2023은 뚜렷하게 마이너스다. 평균은 플러스(+8.90)지만, 이 스윙 폭(2024 +28.23 vs 2023 −10.44, 격차 38.67)은 §23에서 관찰된 "시즌 간 자연 변동폭"(CatBoost 기준 ±20~40) 범위 안에 들어와 순수 노이즈인지 실제 개선인지 이 결과만으로는 단정하기 어렵다. 다만 F1 필터도 정확히 같은 패턴(한 시즌 마이너스, 다른 시즌 크게 플러스, 평균은 확실히 플러스, §18.3)으로 채택된 전례가 있어 완전히 기각하기도 애매한 경계선 결과다.

**결론: 보류(경계선).** 프로덕션 하이퍼파라미터를 아직 변경하지 않는다. 추가 확신이 필요하면 rolling-origin fold check(`--foldcheck` 인프라, §24.6과 동일 방식)로 season==2024 안에서의 out-of-sample 안정성을 한 번 더 확인하거나, 실제 리더보드 제출로 판단하는 게 다음 단계다.

### 31.3 실제 리더보드 제출로 최종 판단 — 기각

경계선 결과였으므로 실제로 `dopip.py`를 튜닝된 하이퍼파라미터로 돌려(local season==2024 블렌드 818.69, §31.2와 동일 수치 재현) `submit.zip`을 만들어 제출했다. **실제 리더보드 957.85** — 기존 971 대비 **−13.15 하락**.

2024 단일 홀드아웃에서 관찰된 +23.92는 실제 개선이 아니라 시즌 변동폭 안의 노이즈였던 것으로 최종 확인됐다. §31.2에서 이미 경고했던 "이 스윙 폭이 시즌 간 자연 변동폭과 겹쳐 판단 불가"라는 우려가 그대로 적중한 사례 — F1 필터(§18.3)가 같은 패턴에서 실제로는 유효했던 것과 달리, 이번엔 경계선 로컬 결과가 실전에서 뒤집혔다. **핵심 교훈에 추가할 사례**: 2024 단일 시즌 홀드아웃에서의 큰 델타는, 특히 그 델타가 다른 시즌에서 부호가 반전될 때, 실제 개선을 보장하지 않는다 — 반드시 실제 리더보드로 교차검증해야 한다.

**최종 결론: 기각.** `code/mlp_model.py`의 `LR`/`WEIGHT_DECAY`/`DROPOUT`/`BATCH_SIZE`를 원래 값(`0.003`/`0.01`/`0.3`/`4096`)으로 되돌렸다(git checkout). `open/reference/best_model.pkl`, `submit/model/final_retained_model.pkl`도 튜닝 이전 커밋(971점 버전, 3b823f2)으로 복원했다. `code/tune_mlp.py`, `code/experiment_mlp_hparam_confirm.py`는 향후 재탐색을 위한 참고 스크립트로 유지(untracked).

## 32. 2-way 메타모델을 비선형으로 교체 — poly2/MLP/GBM 세 후보 전부 rolling-origin foldcheck에서 명확히 기각

지금까지 "N번째 모델을 늘리는" 방향(§12/§12.1/§12.2, §22~24)은 전부 기각됐다. 모델 개수를 늘리는 대신, 기존 2개 모델(CatBoost+MLP)의 예측을 결합하는 방식 자체를 선형 로지스틱 회귀에서 비선형으로 바꾸면 이득이 있는지가 아직 제대로 검증된 적 없는 방향이었다 — §9에서 `cat_pred*mlp_pred` 교호작용항을 한 번 시도한 적 있지만 단일 슬라이스+상황피처와 섞임+구버전 피처셋이라는 한계가 있었다.

`code/experiment_meta_nonlinear.py`로 재검증: `open/reference/best_model.pkl`(CatBoost+MLP)을 재학습 없이 그대로 로드해 `cat_pred`/`mlp_pred`만 추론(base 캐시, 검증: 기존 794.77과 정확히 일치)하고, 입력은 `[cat_pred, mlp_pred]` 2개로 고정한 채 메타모델만 세 가지 비선형 후보로 교체해 §9.1/§12.1과 동일한 rolling-origin 3-fold(월 버킷 확장 윈도우)로 선형 베이스라인과 비교했다.

- **poly2**: `[cat, mlp, cat², mlp², cat×mlp]` + `LogisticRegressionCV`(정규화 강도 CV로 자동 선택, §23과 동일한 정규화 원칙)
- **mlp**: 은닉 4유닛짜리 아주 작은 `MLPClassifier`(강한 L2, early_stopping)
- **gbm**: `max_depth=2`, 트리 30개짜리 `HistGradientBoostingClassifier`

| 후보 | 5~6월 | 7~8월 | 9~10월 | fold 승수 | 평균 delta |
| --- | --- | --- | --- | --- | --- |
| poly2 | -404.06 | -93.73 | +44.83 | 1/3 | **-150.99** |
| mlp | -940.74 | -766.59 | -330.12 | 0/3 | **-679.15** |
| gbm | -125.64 | -76.40 | +21.36 | 1/3 | **-60.23** |

mlp 후보는 두 fold에서 예측이 사실상 상수(~0.434, 표준편차 0.0008)로 붕괴했음을 직접 확인(강한 정규화+[0,1] 범위 미표준화 입력 조합이 원인으로 추정) — 다만 poly2/gbm은 붕괴 없이도 똑같이 뚜렷한 손해를 보여, 이 결론이 mlp 후보의 구현 디테일 때문만은 아니다. 세 계열 모두, 그리고 특히 메타 학습 표본이 가장 작은 첫 fold(5~6월, train 54,949행)에서 손해가 가장 컸다는 점이 §12.1/§12.2("메타모델 학습 표본이 작을수록 파라미터가 많은 모델이 먼저 무너진다")와 정확히 같은 패턴이다.

**결론: 기각, 경계선 아님.** 세 계열 모두 방향이 일관되게 마이너스이고 손해 폭도 크며(-60~-679), F1 필터·MLP 하이퍼파라미터 재탐색처럼 "한 시즌만 마이너스, 평균은 플러스"인 경계선 패턴과 다르다 — 실제 리더보드 확인 없이 로컬만으로 기각을 확정한다. 프로덕션 메타모델(`code/blend_model.py::fit_meta_model`, 순수 2피처 선형 로지스틱 회귀)은 변경하지 않는다. `code/experiment_meta_nonlinear.py`는 향후 다른 비선형 후보(예: 온전히 튜닝된 MLP)를 재탐색할 때 참고용으로 리포지토리에 남겨둔다.

## 33. CatBoost Optuna 재탐색 확대 — GPU 가속, depth/l2_leaf_reg/min_data_in_leaf 범위 확대, "탐색·프로덕션 디바이스 불일치"라는 새로운 함정을 발견 후 디바이스를 일치시켜 채택

§21의 CatBoost 재튜닝(40 trials, CPU, 721.66)이 F1 필터 적용 이후 한 번만 이뤄졌고, 그 뒤로 트라이얼 수·탐색범위 모두 재검토된 적이 없었다. `code/tune.py`에 `TUNE_TASK_TYPE` 환경변수(GPU 가속 옵션)를 추가하고, 탐색범위를 확대(`depth` 4~8→4~10, `learning_rate` 0.01~0.2→0.005~0.3, `l2_leaf_reg` 1~30→1~50, `min_data_in_leaf` 1~200→1~300)해 150 trials로 재실행했다.

### 33.1 GPU 150-trial 탐색

같은 하이퍼파라미터로 timing 테스트한 결과 GPU가 CPU 대비 약 3배 빠름(iterations=300 고정 기준 12.5s vs 36.4s) — 다만 CatBoost GPU는 `BrierScore`(우리 eval_metric)를 네이티브 지원하지 않아 "5-iteration 주기로 CPU에서 계산" 폴백이 걸리고, `depth=10`+낮은 learning_rate 조합은 조기종료까지 1500 iteration 가까이 쓰는 경우가 많아 실제 평균 속도는 예상(26s/trial)보다 느렸다(전체 150 trials, 약 3시간 소요).

**Best trial**: BSS=0.00764 → **Score 763.90**, `depth=10, learning_rate=0.005905, l2_leaf_reg=42.93, random_strength=1.420, bagging_temperature=0.106, border_count=46, min_data_in_leaf=299`. 상위권 대부분이 `depth=10`(탐색범위 상한), `l2_leaf_reg` 35~50대(상한 근처), `learning_rate` 0.006~0.012, `min_data_in_leaf` 200대 이상으로 뚜렷하게 수렴 — §21의 CPU 튜닝(depth=6)과는 완전히 다른 영역.

### 33.2 CPU로 재현 시도 — 전이 실패, "탐색·프로덕션 디바이스 불일치"라는 새로운 함정 발견

프로덕션 학습은 CPU(`train_catboost`, task_type 미지정)이므로, GPU 탐색으로 찾은 params를 그대로 CPU로 재학습해 season 2024/2023 듀얼 홀드아웃으로 검증했다.

| holdout | 기존 CATBOOST_PARAMS (CPU) | GPU 탐색 신규 params (CPU 재학습) | delta |
| --- | --- | --- | --- |
| 2024 | 721.66 (best_iter=761) | 713.31 (best_iter=1490) | **-8.35** |
| 2023 | 541.91 (best_iter=698) | 538.67 (best_iter=1016) | **-3.24** |
| 평균 | | | **-5.80** |

GPU 탐색 중 최고 763.90이었던 후보가 CPU로 재학습하면 두 시즌 다 기존값보다 나쁘다 — 경계선이 아니라 일관되게 마이너스라 최초엔 기각 판단. 그런데 곧바로 "GPU가 원래 더 나쁜 모델을 만드는 건지, 탐색·배포 디바이스가 다른 게 문제인지"를 분리해서 볼 필요가 제기됐다(사용자 제안: 어차피 MLP도 GPU를 쓰는데, CatBoost도 프로덕션 자체를 GPU로 돌리면 되지 않느냐).

### 33.3 디바이스 일치 검증 — 기존 params를 GPU로 재학습(순수 디바이스 효과 격리)

먼저 **동일 하이퍼파라미터**(기존 CATBOOST_PARAMS)를 CPU vs GPU로 학습해 순수 디바이스 효과만 봤다:

| | 2024 (CPU 대비) |
| --- | --- |
| CPU | 721.66 (best_iter=761) |
| GPU | 721.51 (best_iter=968) |

2024만 보면 거의 동일(delta -0.15) — "GPU가 원래 더 나쁘다"는 가설은 기각되는 듯 보였다. 그런데 2023 홀드아웃(학습 데이터가 더 적음, 870,752행)까지 3-seed(42/123/2024)로 넓혀 확인하니 그림이 달라졌다:

| holdout | 기존 params, CPU | 기존 params, GPU (3-seed) |
| --- | --- | --- |
| 2024 | 721.66 | 712.19 ~ 728.71 (평균 718.86) — CPU와 거의 동일 |
| 2023 | 541.91 | 486.05 ~ 510.36 (평균 494.96) — **3/3 시드 모두 -31~-56, 일관되게 나쁨** |

즉 GPU 학습은 학습 데이터가 적을수록(iteration 수가 적게 끝날수록) 불안정해진다 — BrierScore 조기종료가 GPU에서 5-iteration 주기로만 체크되는 게 원인으로 추정(§33.1). `depth=6`짜리 기존 params는 보통 400~850 iteration에서 끝나 이 5-iteration 오차가 상대적으로 크게 작용하지만, `depth=10`+매우 낮은 learning_rate인 신규 params는 900~1400대까지 학습해 상대오차가 작다.

신규탐색 params도 같은 3-seed×2-holdout으로 GPU 재학습해 비교(둘 다 GPU, 공정 비교):

| holdout | 기존(GPU) 평균 | 신규탐색(GPU) 평균 | fold/seed 승수 |
| --- | --- | --- | --- |
| 2024 | 718.86 | 750.31 | 3/3 |
| 2023 | 494.96 | 539.38 | 3/3 |

6/6 시드·시즌 조합 전부 신규탐색 승리 (2024: +14.6~+46.2, 2023: +20.8~+57.2) — GPU 조건에서는 신규탐색 params가 확실히 더 낫다.

가장 중요한 비교는 **"GPU+신규params"가 실제 현재 프로덕션 "CPU+기존params"(721.66/541.91, 971점의 원천)를 이기는가**다:

| holdout | seed42 | seed123 | seed2024 |
| --- | --- | --- | --- |
| 2024 | +36.70 | +27.61 | +7.05 |
| 2023 | -0.61 | +3.78 | -10.77 |

2024는 3/3 시드 모두 뚜렷하게 플러스(+7~+37), 2023은 3개 시드 모두 ±11 이내로 GPU 자체의 시드 변동폭(±13, 위 표 참고)과 비슷한 크기 — MLP 하이퍼파라미터 재탐색(§30/§31, 한 시즌이 뚜렷하게 마이너스)과 달리 "한쪽은 확실한 개선, 한쪽은 노이즈 수준 중립"인 패턴이라 경계선으로 보지 않고 채택하기로 판단.

### 33.4 채택 — `CATBOOST_PARAMS`를 GPU+신규탐색 params로 교체, `dopip.py` 전체 재실행

`code/catboost_model.py::CATBOOST_PARAMS`를 §33.1의 신규탐색 값으로 교체하고 `task_type="GPU", devices="0"`을 추가(`train_catboost`가 `CATBOOST_PARAMS`를 그대로 복사해 쓰므로 이 한 곳만 바꾸면 검증 학습·전체 재학습 둘 다 자동 반영). `dopip.py` 전체 파이프라인(7-seed MLP 앙상블 + GPU CatBoost + 스태킹) 재실행:

- CatBoost 검증 학습: `best_iteration=1376`
- 스태킹 메타모델: `w_cat=2.130, w_mlp=2.317, intercept=-2.245`
- **블렌드 Val Score: 812.38** — 기존 reference(794.77) 대비 **+17.61**, `NEW_BEST` 판정으로 승격
- 전체 재학습(CatBoost 1426 iteration = 1376+버퍼50, MLP 7-seed 7~11 epoch)까지 정상 완료, `submit/model/final_retained_model.pkl` 갱신

**추론 호환성 확인**: GPU로 학습된 `CatBoostClassifier`를 `CUDA_VISIBLE_DEVICES=""`(GPU 없는 환경 시뮬레이션)에서 `predict_proba`로 직접 호출해 정상 동작 확인 — 트리 구조는 디바이스 무관하게 직렬화되므로 학습 디바이스와 무관하게 추론 가능. `submit/script.py`도 로컬 데이터로 5행 스모크 테스트(`submit/data`에 실데이터 심볼릭 링크로 임시 연결) 통과, `submission.csv`에 다양한 실수 확률값 정상 출력 확인 후 `submit.zip` 재구성 완료.

**결론: 로컬 기준 채택.** `code/catboost_model.py::CATBOOST_PARAMS`가 GPU+신규탐색 값으로 교체됐고, `open/reference/best_model.pkl`·`submit/model/final_retained_model.pkl` 모두 새 블렌드로 갱신됐다. 다만 이번 세션의 다른 교훈(§30/§31: 로컬 개선이 실전에서 뒤집힌 전례)을 감안해 **실제 리더보드 제출로 최종 확인 전까지는 "잠정 채택"** 상태로 취급한다. 이번 케이스는 §30/§31과 달리 2/2 홀드아웃이 손해가 아니라 "확실한 개선 + 노이즈 수준 중립"이라는 더 안전한 패턴이라 신중하게 낙관적이다.

### 33.5 실제 리더보드 제출 — 967.36, 근소하게 하락 (최종 기각, 복원)

`submit.zip`을 실제로 제출한 결과 **967.3619755738** — 기존 971 대비 **−3.64 하락**. 로컬은 분명히 올랐지만(794.77→812.38, +17.61) 실전은 근소하게 떨어졌다 — §30/§31(MLP 하이퍼파라미터 재탐색, −13.15)에 이어 이번 세션 **두 번째로 "로컬 상승·실전 하락"이 재현된 사례**다. 다만 하락폭이 훨씬 작다(−3.64 vs −13.15) — 완전한 손해라기보다 "거의 동률인데 근소하게 밑도는" 수준에 가깝다.

§33.3에서 관찰한 "기존 params를 GPU로 학습하면 2023에서 시드 간 ±13, 최대 −56까지 흔들린다"는 사실 자체가, 이 실험이 원래부터 순수 하이퍼파라미터 개선이 아니라 **디바이스 전환에 따른 학습 노이즈**를 상당 부분 포함하고 있었음을 시사한다 — GPU 학습의 시드 변동폭이 실제 대시보드 단일 지점 관측값(2025 시즌, 새로운 unseen 데이터)에서도 비슷한 크기로 나타났다고 보는 게 합리적인 해석이다. season==2024/2023 두 홀드아웃 모두에서 우호적이었던 것치고는 실전 하락폭이 작았다는 점이, "디바이스 자체 노이즈가 진짜 하이퍼파라미터 개선분을 상당 부분 상쇄했다"는 가설과 부합한다.

**최종 결론: 기각, 복원.** `code/catboost_model.py`(`CATBOOST_PARAMS`+GPU 설정), `code/tune.py`(GPU 옵션+확대된 탐색범위)를 `git checkout`으로 원복했다. `open/reference/best_model.pkl`, `submit/model/final_retained_model.pkl`도 971점 커밋 상태로 복원, `submit.zip` 재구성 완료. `code/tune.py`의 GPU 지원 자체(`TUNE_TASK_TYPE` 옵션)는 유용한 인프라이지만, 이번 세션 결론상 "탐색·배포 디바이스를 일치시켜도 CPU 대비 실전 우위를 만들지 못했다"는 것이 최종 판단이라 커밋 전 상태로 되돌렸다 — 향후 재도전 시 이 세션의 diff를 참고할 수 있다(git reflog/이 세션 로그).

**핵심 교훈에 추가**: GPU CatBoost 학습은 (특히 학습 데이터가 작을 때) 조기종료 체크 주기가 성긴 탓에 CPU보다 시드 간 변동폭이 크다 — 이 노이즈가 실제 대시보드 결과에도 비슷한 크기로 반영될 수 있어, "GPU로 탐색+배포를 일치시키면 안전하다"는 가정은 절반만 맞다(탐색-배포 전이 문제는 해결되지만, 디바이스 자체의 학습 불안정성은 남는다). 순수 학습 속도 때문에 GPU를 쓰려면 이 노이즈를 감안한 멀티시드 앙상블(CatBoost 쪽도 MLP처럼) 등 추가 안정화 장치가 필요해 보인다 — 다음에 시도해볼 만한 방향으로 남겨둔다.

## 34. 블렌드 확률 사후 보정(calibration) — isotonic/platt 둘 다 기각

CatBoost+MLP를 어떻게 합칠지(모델 개수 확장 §12/§22~24, 메타모델 비선형화 §32)가 아니라, 이미 합쳐진 최종 확률이 잘 보정(calibrated)돼 있는지를 검증. BSS(Brier)는 확률 보정 품질에 직접 민감한 지표라서, 모델 결합 방식과 독립적으로 시도할 가치가 있었다. `code/experiment_calibration.py`가 `code/experiment_meta_nonlinear.py`의 캐시(재학습 불필요)를 재사용해, §9.1과 동일한 rolling-origin 3-fold로 "기존 선형 스태킹" vs "그 위에 보정 함수 한 겹 추가"를 비교.

- **isotonic**(비모수 단조회귀): 1/3 fold 승리, 평균 **−2.91** — meta_train 구간의 세부 확률 곡선에 과적합해 시즌 후반 fold(9~10월)에서 −13.48로 크게 손해.
- **platt**(logit 위에 1차원 로지스틱 회귀): 3/3 fold 승리, 평균 **+1.92**(+0.01~+4.12) — 방향은 일관되지만 이 프로젝트에서 채택된 개선들(+14~+78)에 비해 크기가 노이즈 경계선 수준.

**결론: 둘 다 기각.** 지금 쓰는 선형 스태킹(`sigmoid(w_cat·cat+w_mlp·mlp+intercept)`) 자체가 이미 로지스틱 회귀 기반이라 상당히 잘 보정돼 있어, 그 위에 얹는 보정이 고칠 잔차가 거의 안 남아있던 것으로 해석된다(Guo et al. "On Calibration of Modern Neural Networks" 관점에서도, 우리 MLP는 은닉층 2개짜리 저용량 네트워크에 `weight_decay=0.01`이 걸려있어 그 논문이 지적하는 "깊고 weight decay 없는 과신 모델" 상황과는 거리가 멀다는 것과 일관됨). 프로덕션 메타모델은 변경하지 않는다. `code/experiment_calibration.py`는 향후 다른 보정 후보(예: 온전히 정규화된 platt 변형) 재탐색용으로 남겨둔다.

## 35. ABS(자동 볼 판정 시스템) 레짐 시프트 가설 — 2024 head 데이터 포함이 견고한 개선, cutoff=7·weight=1 채택 (§37에서 실제 리더보드 982.22로 확정 채택)

### 35.1 배경 — 두 가설(9월 콜업 vs ABS 도입) 검증

`code/experiment_calibration.py`의 rolling-origin fold check에서 9~10월 fold(330.12)가 다른 fold(766.59, 961.95)보다 유독 나쁜 게 반복 관찰됨. 사용자가 "가을야구(2군/신인 콜업)" 가설을 제기했으나, 데이터로 반박됨: 2024시즌 9월 신규 등판 투수 수(23명)가 8월(25명)과 비슷하고, `asof_pitcher_n<50` 비율도 9월(1.1%)·10월(0%)로 오히려 다른 달보다 낮음(§표는 대화 로그 참고, 문서 미기록).

대신 사용자가 제기한 "ABS(자동 볼 판정 시스템) 도입" 가설을 웹 검색으로 확인: KBO는 실제로 **2024시즌부터 ABS를 전면 도입**(2020~2023년엔 퓨처스리그에서만 시범 운영, 91%→95~96% 판정 정확도 개선 목표). `control_success`가 스트라이크존 위치 기준 라벨이라, 판정 시스템이 바뀌면 라벨링 자체가 달라질 수 있음. 시즌별 성공률(r)도 2019(0.5647)→2024(0.4861)로 단조 하락, 특히 2022→2023(−2.9%p)이 가장 크고 2023→2024(−1.4%p)도 이어짐. 사용자의 별도 앙상블 실험에서도 CatBoost 단독이 2023(751)→2024(642)로 급락했다는 보고와 방향이 일치한다고 봤으나, **§36에서 이 "751→642" 보고 자체가 팀원 코드의 데이터 leak 버그로 밝혀져 정정됨** — 이 근거는 무효, ABS 가설 자체는 §35.2~35.5의 독립적 실험으로 별도 지지됨.

### 35.2 head-inclusion 실험 설계

가설: "2019~2023(구 판정체계) 데이터가 아무리 많아도 실제 평가(2025시즌, 2024와 같은 ABS체제)와는 체제가 다르다"면, **학습에 2024(ABS체제) 데이터를 일부 포함시키는 것**이 도움될 것. `code/experiment_abs_regime.py`로 2024를 월 기준(`cutoff`)으로 head(학습 포함)/tail(검증)로 나눠 세 가지 비교:
- A: `season<2024`만 학습
- B: `season<2024` + `season==2024 & month<cutoff`, 가중치 없음(weight=1)
- C: B와 동일 데이터, `season==2024` 행에 sample_weight 부여(weight=5 등)

F1 필터는 학습 구간에 그대로 적용(`season<=2022`인 F행만 제거, 2024 데이터엔 영향 없음).

### 35.3 CatBoost 단독 스윕 — cutoff 4~10 × weight 1/5

| cutoff | 검증행수 | A(baseline) | weight=1 | delta | weight=5 | delta |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 241,146 | 702.78 | 715.17 | +12.39 | 632.78 | −70.00 |
| 5 | 198,558 | 721.97 | 749.76 | +27.79 | 696.53 | −25.44 |
| 6 | 154,480 | 666.33 | 718.17 | +51.84 | 668.54 | +2.21 |
| **7** | 109,966 | 564.32 | **634.32** | **+70.00** | 615.55 | +51.23 |
| **8** | 76,896 | 457.64 | **529.73** | **+72.09** | 389.26 | −68.38 |
| 9 | 34,976 | 269.18 | 310.04 | +40.86 | 335.96 | +66.78 |
| 10 | 1,671 | 151.04 | 131.99 | −19.05 | 168.14 | +17.10 |

**weight=1은 cutoff 4~9 전부 플러스**로 매우 견고함(delta가 4→8까지 매끄럽게 커지다 9~10에서 검증행수 급감과 함께 꺾이는 산 모양 — 표본이 작아질수록 노이즈가 커지는 자연스러운 패턴). **weight=5는 방향이 들쭉날쭉**(−70~+67, 뚜렷한 추세 없음) — "r 추세가 반전되는 지점마다 weight=5가 진다"는 가설도 검토했으나 4/7 케이스만 들어맞고 나머지는 반례라 기각, 대신 "head 표본이 작을수록(cutoff=4,5,8) weight=5가 그 노이즈를 증폭시킨다"는 설명이 더 일관됨.

### 35.4 MLP+블렌드 검증 — cutoff 4~10 (weight=1, 3-seed 스크리닝)

`code/experiment_abs_mlp.py`로 CatBoost 단독 결론이 블렌드(프로덕션이 최적화하는 실제 지표)에서도 재현되는지 확인. Config A는 재학습 없이 기존 `open/reference/best_model.pkl`(7-seed 프로덕션) 예측을 tail로 마스킹, Config C는 3-seed(`[42,123,7]`) 스크리닝 + `WeightedRandomSampler`(물리적 행 복제 대신 가중 샘플링, epoch 비용 유지).

| cutoff | 검증행수 | A(블렌드) | C(블렌드, weight=1) | delta |
| --- | --- | --- | --- | --- |
| 4 | 241,146 | 776.22 | 740.16 | **−36.06** |
| 5 | 198,558 | 819.76 | 809.78 | **−9.98** |
| 6 | 154,480 | 754.79 | 764.82 | +10.04 |
| **7** | 109,966 | 657.94 | **715.97** | **+58.03** |
| 8 | 76,896 | 550.68 | 572.25 | +21.57 |
| 9 | 34,976 | 392.62 | 408.75 | +16.13 |
| 10 | 1,671 | 69.97 | 257.37 | +187.40 (표본 극소, baseline 자체가 0에 가까워 신뢰 불가) |

CatBoost 단독과 달리 **cutoff=4,5는 블렌드에서 마이너스로 뒤집힘** — head 데이터가 아주 작을 때(12,361~54,949행) CatBoost의 작은 이득을, 3-seed MLP 스크리닝의 시드 노이즈(프로덕션은 7-seed)가 집어삼킨 것으로 추정. **cutoff=7이 전체 cutoff 중 가장 크고 가장 신뢰할 만한 이득**(+58.03, 표본도 10을 제외하면 최대).

같은 스윕을 weight=5(2024 head 행에 5배 가중치, CatBoost는 `sample_weight`, MLP는 `WeightedRandomSampler`)로도 전체 cutoff 재실행:

| cutoff | 검증행수 | A(블렌드) | C(블렌드, weight=5) | delta | (참고) weight=1 delta |
| --- | --- | --- | --- | --- | --- |
| 4 | 241,146 | 776.22 | 690.11 | **−86.10** | −36.06 |
| 5 | 198,558 | 819.76 | 759.95 | **−59.82** | −9.98 |
| 6 | 154,480 | 754.79 | 700.67 | **−54.12** | +10.04 |
| **7** | 109,966 | 657.94 | 671.58 | +13.64 | **+58.03** |
| 8 | 76,896 | 550.68 | 503.61 | **−47.07** | +21.57 |
| 9 | 34,976 | 392.62 | 430.28 | +37.66 | +16.13 |
| 10 | 1,671 | 69.97 | 318.21 | +248.24 (표본 극소, 신뢰 불가) | +187.40 (표본 극소, 신뢰 불가) |

**weight=5는 블렌드 기준으로 cutoff=4,5,6,8 전부 weight=1보다 나쁘고, 심지어 baseline A 대비도 마이너스**(4,5,6,8 네 cutoff 모두 A보다 낮음) — CatBoost 단독 스윕(§35.3)에서 본 "weight=5는 방향이 들쭉날쭉"이 블렌드에서는 아예 "표본이 충분히 큰 cutoff에서 일관되게 손해"로 더 나쁘게 나타남. 유일하게 이기는 cutoff=7도 weight=1(+58.03)보다 크게 못함(+13.64). cutoff=9,10의 plus는 검증 표본이 각각 34,976/1,671로 작아 노이즈일 가능성이 높다는 §35.3 해석과 일치. **weight=5는 어떤 cutoff에서도 weight=1을 이기지 못함 → weight=5 채택 근거 없음, weight=1 채택을 재확인.**

### 35.5 2023 교차검증 — 다른 해에도 재현되는지, F1 필터 트랩 회피하며 검증

2023을 검증셋으로 쓰려면 학습을 `season<2023`(=2019~2022)으로 잘라야 하는데, 이러면 F1 필터(`season<=2022`인 F행 제거)가 학습 구간의 **F행을 100% 제거**한다(§29의 트랩과 동일 메커니즘). 세 단계로 접근:

1. **F1 필터 완전 미적용**(pre_df에 F1 필터 자체를 안 걺, F 성공률 역전 이전 관계가 그대로 학습됨): baseline이 `best_iter=3, score=29.27`로 거의 즉시 붕괴 — §18에서 F1 필터가 고치려던 문제(2022 이전 F/R 관계를 그대로 2023에 적용하면 스코어가 0에 수렴)가 재현된 것. 월별 breakdown도 0.00~46.83으로 전 구간 균일하게 나쁨(4월/9월 특유 굴곡 없음) — baseline 자체가 통째로 망가진 상태라 세부 패턴을 볼 수 없는 상태였음.
2. **F1 필터 정상 적용**(프로덕션 `apply_f1_filter` 함수 그대로, pre-2023 구간은 여전히 F 전체 제거되지만 head로 포함되는 2023 앞부분은 `season<=2022` 조건에 안 걸려 F가 유지됨): baseline이 정상 범위로 회복(cutoff=7: 493.08, cutoff=9: 410.98). head-inclusion delta는 **cutoff=7: +256.12, cutoff=9: +96.46**로 2024보다도 훨씬 큼 — 다만 F1이 못 잡는 잔여 F/R 효과가 섞였을 가능성.
3. **`game_type='R'`만 사용**(F/R 이슈 완전 배제): head-inclusion delta가 **cutoff=7: +65.15, cutoff=9: +64.37**로 크기는 줄었지만 여전히 견고한 플러스 — F1 문제와 무관하게 "같은 해 데이터 포함"이 2023에도 진짜로 도움이 된다는 뜻. R전용 월별 breakdown(2023)은 `4월(559.91) > 평균, 8월(620.97) 최고, 9월(391.74) 최저, 10월(407.49)`로 **4월 dip이 없고 9~10월만 약함** — 2024(4월·9~10월 둘 다 dip)와는 다른 패턴.

**해석**: "같은 해 데이터 포함이 도움된다"는 효과는 2023·2024 둘 다에서 F1/F-R 이슈와 무관하게 재현되는 일반적 패턴으로 보인다. 다만 **4월 dip은 2024 고유 현상**으로 보임(2023 R전용엔 없음) — "매년 반복되는 초반 콜드스타트"보다는 "2024년 4월(ABS 전면 도입 직후 적응기) 특유의 사건"에 가깝다는 가설에 좀 더 무게가 실림. 9~10월 약세는 2023·2024 공통이라, 매년 반복되는 후반부 일반 패턴(피로 누적, 순위 결정 국면 등)일 가능성.

### 35.6 검토했으나 기각한 방향 — weight=5 기반 시즌별 라우팅/게이팅

사용자가 "9~10월은 weight=5 모델이, 나머지는 weight=1 모델이 맞추도록 메타모델/게이팅으로 라우팅하면 어떤가"를 제안. §35.3(CatBoost 단독)과 §35.4(블렌드) 양쪽의 weight=5 전체 cutoff 스윕으로 검토한 결과 기각:
1. weight=5의 cutoff=9 우위는 §35.3/§35.4에서 이미 "표본이 작아서 생기는 노이즈"로 잠정 결론난 상태 — 노이즈 위에 라우팅 구조를 얹으면 노이즈를 고정시키는 꼴. 블렌드 기준으로는 weight=5가 표본이 큰 cutoff(4,5,6,8)에서 baseline보다도 나쁘다는 게 §35.4에서 명확히 확인됨 — "9~10월만 weight=5로 맞추자"는 제안의 전제 자체("weight=5가 특정 구간에서 더 낫다")가 표본 큰 구간에서는 성립하지 않음.
2. `game_month`가 이미 CatBoost/MLP 양쪽의 원시 입력 피처라, "9~10월이 다르다"는 신호를 단일 모델도 이미 학습할 수 있는 구조.
3. 이 프로젝트에서 "구간별 특화 모델/피처"(선발-불펜 구분 −7.0, F/R 상호작용 등)는 반복적으로 실패, 반면 "단일 모델에 데이터를 잘 넣어주는" 접근(F1 필터, 이번 head-inclusion)은 성공한 패턴과 상충.
4. §32에서 메타모델에 복잡도(비선형성)를 추가하는 시도가 전부 실패한 것과 같은 리스크가 게이팅에도 적용됨.

**사용자 재검토 요청 — "w=5가 9~10월 검증에서 (같은 cutoff끼리 비교하면) 항상 w=1을 이긴다"는 관찰을 2023으로 재검증**: 2024 데이터만 보면 실제로 cutoff=9(+37.66)와 cutoff=10(+248.24) 모두 w=5가 w=1을 이겼다는 사용자 지적은 정확했다. 다만 cutoff=10은 검증행이 1,671개뿐이라 신뢰도가 의심스러웠던 값(§35.4에서 이미 "표본 극소, 신뢰 불가"로 표시). `code/experiment_abs_routing_2023.py`로 같은 비교를 2023(`game_type='R'`만 사용, F/R 이슈 완전 배제, §35.5와 동일 방식)의 CatBoost+MLP 블렌드로 재현:

| 년도 | cutoff | 검증행수 | w=1 블렌드 | w=5 블렌드 | delta(w5−w1) |
| --- | --- | --- | --- | --- | --- |
| 2024 | 9 (9~10월) | 34,976 | 408.75 | 430.28 | +37.66 |
| 2024 | 10 (10월만) | 1,671 | 257.37 | 318.21 | **+248.24** |
| 2023 | 9 (9~10월) | 51,074 | 664.96 | 670.46 | +5.50 |
| 2023 | 10 (10월만) | **17,638** | 900.20 | 883.60 | **−16.60** |

**2023 10월은 2024 10월보다 검증행이 10배 이상 크고(17,638 vs 1,671) 정상적인 표본인데, 여기서 w=5의 우위가 뒤집힌다(−16.60)**. cutoff=9(9~10월 묶음)도 표본이 더 큰 2023 쪽에서 이득이 +37.66 → +5.50으로 급격히 줄어든다 — "표본이 클수록 w=5의 이득이 작아지거나 사라진다"는 뚜렷한 패턴으로, §35.3/§35.4의 "head 표본이 작을수록 weight=5가 노이즈를 증폭시킨다" 해석과 정확히 부합한다. **"9~10월 전용 w=5 모델" 아이디어는 2023 재검증으로도 뒷받침되지 않아 기각을 재확인.**

### 35.7 최종 결론 — cutoff=7·weight=1 채택 (프로덕션 반영 완료, §37에서 실제 리더보드 확인)

**채택한 프로덕션 분할 변경**:
- 학습: `season<2024` + `season==2024 & game_month<7`
- 검증: `season==2024 & game_month>=7`
- 가중치: 없음(weight=1) — weight=5 계열은 cutoff 전반에서 불안정해 기각

`code/train.py`/`code/test.py`의 `train_mask`/`val_mask`를 위 조건으로 교체하고(`dopip.py`는 항상 전체 데이터로 재학습하는 구조라 수정 불필요) `dopip.py` 전체 파이프라인을 실행 → `NEW_BEST`로 승격 → 실제 리더보드 제출까지 완료. 결과는 §37.

## 36. "2024만 유독 절벽" 여부 재검증 — 우리 아키텍처에서는 재현 안 됨, 팀원 리포트는 데이터 leak 버그로 판명

§35.1에서 인용한 "사용자의 별도 앙상블 실험, CatBoost 단독 2023(751)→2024(642) 급락"이 실제로 우리 프로덕션 아키텍처에서도 재현되는지 사용자가 직접 재확인을 요청했다.

### 36.1 연도별 r 추세 기술통계 (2019~2024, `game_type='R'`만)

| season | n | r_mean | YoY Δ | r_std(월간) | 월별기울기 | 전반(3-6월) | 후반(7-10월) | 후반−전반 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 211,627 | 0.5495 | — | 0.0177 | +0.0035 | 0.5497 | 0.5493 | −0.0004 |
| 2020 | 220,874 | 0.5269 | −0.0226 | 0.0074 | +0.0001 | 0.5273 | 0.5268 | −0.0005 |
| 2021 | 221,227 | 0.5128 | −0.0141 | 0.0133 | −0.0048 | 0.5199 | 0.5055 | −0.0144 |
| 2022 | 217,024 | 0.5037 | −0.0091 | 0.0155 | −0.0055 | 0.5138 | 0.4932 | −0.0206 |
| 2023 | 219,839 | 0.5031 | −0.0006 | 0.0086 | +0.0031 | 0.4975 | 0.5086 | +0.0111 |
| 2024 | 223,497 | 0.4897 | −0.0134 | 0.0153 | +0.0004(*) | 0.4946 | 0.4833 | −0.0113 |

(*) 2024 10월(n=526, R전용 기준)이 극소 표본이라 월별기울기가 왜곡됨 — 3~9월만 보면 0.5038→0.4749로 뚜렷한 하락.

r_mean은 2019→2024까지 단조 하락하되 감소폭이 2023(−0.0006, 거의 정체)까지 줄다가 2024(−0.0134)에 다시 확대된다. 다만 **시즌 내 월별 추세(하락/상승 방향) 자체는 2024가 특이 케이스가 아니다** — 2021·2022도 시즌 내내 하락 추세였고, 오히려 **2023이 유일하게 상승 추세(+0.0031)**로 이질적이다. 9월만 보면 2024(0.4749)가 시즌 최저지만 2021(0.5128 근처)·2022(0.4881)도 9~10월이 시즌 최저 구간이었던 반면 2020(0.5364)·2023(0.5064)은 오히려 9월이 시즌 상위권 — "9~10월 하락"이 2024 고유 현상은 아니고, 2023만 예외적으로 없었다.

### 36.2 연도별 홀드아웃 재현 — `train<season → eval==season`, R-only, CatBoost+MLP(3-seed)+블렌드

`game_type='R'`만 사용(F1 필터의 "학습 구간 F 전량 제거" 트랩을 피하기 위해, §29/§35와 동일 방식)해 2021~2024 4개 연도를 프로덕션과 동일한 `CATBOOST_PARAMS`/피처 엔지니어링/블렌드 스태킹으로 재현했다:

| holdout | CatBoost | MLP(3-seed) | Blend |
| --- | --- | --- | --- |
| 2021 | 515.19 | 451.90 | 519.37 |
| 2022 | 584.10 | 575.66 | 600.64 |
| 2023 | 534.39 | 502.89 | 557.49 |
| **2024** | **721.94** | **773.26** | **797.66** |

**2024가 CatBoost·MLP·Blend 세 지표 모두에서 4개 연도 중 최고점 — "2024만 절벽"은 이 아키텍처에서 전혀 재현되지 않는다.** 오히려 이 프로젝트가 역사적으로 반복 관찰해온 패턴(§23의 트랙맨 context 실험 기준선: CatBoost 721.66/541.91, MLP 759.80/545.85, Blend 792.93/558.97, 2024/2023 순)과 이번 재현 결과(721.94/534.39, 773.26/502.89, 797.66/557.49)가 거의 정확히 일치한다 — **이 프로젝트의 모든 2023/2024 듀얼 홀드아웃 기록에서 2024가 2023보다 항상 높았다.**

§26에서 이미 확인했듯, 2023이 낮은 이유는 "모델이 못 맞혀서"가 아니라 **2023의 r(0.5031, R-only)이 0.5에 거의 정확히 맞아떨어져 baseline_brier가 최댓값 근처에 있고, 이 지점에서 BSS 공식이 절대 오차의 미세한 차이에도 극도로 민감해지기 때문**이다. 실제로 이번 재현의 절대 CatBoost Brier는 2023 0.248654 vs 2024 0.248090으로 차이가 0.0006에 불과했다(§36.1 대비 §26 재확인).

### 36.3 팀원 리포트 코드 분석 — "2023(751)→2024(642)" 원인은 데이터 leak 버그

사용자가 제공한 팀원의 5-모델 스태킹 스크립트를 직접 리뷰했다. 핵심 버그:

```python
df_cleaned = preprocess_pipeline(train_raw, is_train=True)
is_val = df_cleaned['season'] == holdout_year
train_df = df_cleaned[~is_val].reset_index(drop=True)   # <- 문제
val_df = df_cleaned[is_val].reset_index(drop=True)
```

`train_df = df_cleaned[~is_val]`는 "holdout_year가 *아닌* 모든 행"이지 "holdout_year *이전* 모든 행"이 아니다. `train_raw`(`train.csv`)는 2019~2024 전 시즌을 포함하므로:
- `holdout_year=2023` 호출: `train_df` = {2019,2020,2021,2022,**2024**} — **미래(2024) 데이터가 2023 예측용 학습에 leak**된다.
- `holdout_year=2024` 호출: `train_df` = {2019~2023} — 2024가 데이터셋의 마지막 시즌이라 우연히 leak이 없는 정직한 검증이 된다.

즉 **751점(2023)은 미래 데이터로 부풀려진 값이고, 642점(2024)이 leak 없는 진짜 기준선**이다 — "2024로 갈수록 급락"이 아니라 "2023 쪽이 비정상적으로 부풀려졌다"가 맞는 해석이며, §36.2의 우리 재현 결과(2023 낮음/2024 높음, leak 없는 정직한 분할)와 방향이 정확히 일치한다.

부차적 요인(방향을 만드는 주범은 아니지만 절대 점수를 깎음): (1) CatBoost에 `cat_features` 미지정 — target-statistic 처리 없이 순수 숫자로 학습, (2) `eval_set`/`early_stopping_rounds` 없이 고정 1200 iteration — 시즌마다 다른 최적 iteration(우리 재현: holdout=2023일 때 177~254, holdout=2024는 739~768)을 못 맞춤, (3) `season`을 피처에서 완전히 제외, (4) L0 OOF용 `StratifiedKFold(shuffle=True)`가 시간 순서를 무시 — 팀원이 스스로 지적한 문제("타임 슬라이스 없이 K-Fold")이지만, 실제로 해를 끼치는 건 2024가 섞인 학습 풀을 랜덤 폴드로 나누는 **2023 실행 쪽**이라 원인 진단 방향이 반대로 되어 있었다.

### 36.4 결론

이 프로젝트의 아키텍처(F1 필터, 트랙맨 드롭, 68개 피처, 재튜닝 CatBoost, quantile-embedding MLP 앙상블, 스태킹 블렌드)에서는 "2024가 다른 해보다 유독 예측하기 어렵다"는 절벽 현상이 **전혀 관찰되지 않는다** — 정반대로 2024가 항상 가장 높은 점수를 받는다. §35의 ABS 레짐 시프트 가설(head-inclusion이 2024 후반부 예측에 도움된다는 §35.2~35.5의 실험)은 이 결과와 모순되지 않는다 — "2024를 예측하기 어렵다"가 아니라 "같은 해 데이터를 학습에 포함시키면 그 해 후반부를 더 잘 맞힌다"는, 방향이 다른 별개의 관찰이기 때문이다. 다만 §35.1에서 보조 근거로 인용했던 팀원의 "751→642" 수치는 그 자체가 leak 버그의 산물로 밝혀졌으므로 ABS 가설의 근거 목록에서 제외한다.

## 37. cutoff=7·weight=1 프로덕션 반영 및 실제 리더보드 확인 — 982.22, 971 대비 +11.22로 최종 채택

§35.7의 제안을 실제로 반영했다. `code/train.py`/`code/test.py`의 `train_mask`/`val_mask`를 `(season<2024) | (season==2024 & game_month<7)` / `(season==2024 & game_month>=7)`로 교체(`dopip.py`는 Step 3 full retrain이 항상 전체 데이터를 쓰는 구조라 수정 불필요 — 분할은 오직 reference 모델의 epoch/iteration 수·메타모델 가중치를 얼마로 정할지에만 영향을 준다).

### 37.1 `dopip.py` 실행 결과

- **Step 1 (train.py)**: 학습 1,259,818행(2019~2023 F1필터 + 2024 3~6월) | 검증 109,966행(2024 7~10월) — §35.4의 3-seed 스크리닝과 행 수 정확히 일치. 7-seed MLP 앙상블 학습(best_epoch: 2,4,3,2,2,3,5), CatBoost `best_iteration=1025`(§35.4 스크리닝의 931보다 더 큼 — 7-seed 프로덕션 학습이라 조기종료 시점이 다름), 메타모델 `w_cat=1.988, w_mlp=1.976, intercept=-2.011`(기존 `w_cat=2.119, w_mlp=2.366`보다 CatBoost·MLP 비중이 더 균형에 가까움) | **블렌드 Val Score: 705.34**.
- **Step 2 (test.py)**: 신규 모델(705.34) vs 기존 reference를 새 검증셋(2024 7~10월)에 재평가한 값(644.49, §35.4의 A=657.94와 비슷한 수준 — 다른 채점 경로라 근소한 차이) 비교 → **NEW_BEST 승격 확정, +60.85**(§35.4의 3-seed 스크리닝 추정 +58.03과 거의 일치, 실제 7-seed 프로덕션 학습으로도 재현됨).
- **Step 3 (Full Retrain)**: 전체 1,369,784행(F1필터 적용, 전 시즌)으로 7-seed MLP(epoch 7~10, buffer 5 포함) + CatBoost(1075 iteration = 1025+buffer 50) 재학습, reference의 메타모델 가중치 그대로 재사용 → `submit/model/final_retained_model.pkl` 생성.

### 37.2 `submit.zip` 재구성 및 스모크 테스트

`submit/script.py`는 분할 로직에 의존하지 않으므로(피처셋 불변, `league_success_mean`도 항상 `train.csv` 전체 기준) 코드 수정 없이 그대로 재사용. `submit.zip`을 `model/`, `script.py`, `requirements.txt` 최상위 구조로 재구성 후, 실제 zip을 풀어 5행 샘플 `test.csv`로 `script.py`를 통째로 실행하는 스모크 테스트 통과(예측값 0.42~0.50 범위, 정상).

### 37.3 실제 리더보드 결과 — 982.22, 최종 채택

실제 제출 점수 **982.22**, 기존 최고 971 대비 **+11.22**. 이 세션에서 실제 리더보드로 최종 확인한 세 번째 케이스이자(§32의 CatBoost GPU 재탐색 −3.64, §31의 MLP 하이퍼파라미터 재탐색 −13.15는 모두 로컬은 개선처럼 보였지만 실전에서 하락해 반려됨), **처음으로 로컬 개선이 실전에서도 그대로 유지된 케이스**다. cutoff=7·weight=1은 최종 채택 확정.

**핵심 차이점**(사용자 질문에 대한 답): 최종 제출 모델의 학습 *데이터*는 이전과 동일하게 항상 전체(2019~2024 전부)를 쓴다 — 이건 cutoff을 바꿔도 안 바뀌는 `dopip.py`의 기존 구조다. 바뀌는 건 그 전체 데이터로 **몇 개의 트리/epoch을 쓸지, CatBoost·MLP를 얼마 비율로 섞을지**를 결정하는 reference 모델의 학습 조건이다 — 이번에 CatBoost iteration이 761(구)→1025(신)로, 메타모델 가중치가 `w_cat=2.119/w_mlp=2.366`(구, MLP 우세)→`w_cat=1.988/w_mlp=1.976`(신, 균형)으로 바뀌었고, 이 차이가 실제 리더보드 +11.22로 이어졌다.

## 38. 트랙맨 재도전 — 투수 크로스워크 x pitcher-std/pressure 4개 티어 스크리닝, cutoff=7 확인 후 A/B/C 프로덕션 통합 → 실제 리더보드 869.52로 대폭 하락(−112.70), 크로스워크 커버리지 선택 편향으로 잠정 진단 (미해결, 롤백 대기)

`code/pitcher_crosswalk.py`(§14.1/§25.1에서 이미 만들어 둔 시퀀스 재구성 기반 `pitcher_id ↔ pitcher_trackman_id` 크로스워크, train.csv 792명 중 568명/71.7% 매칭, 외부데이터 아님)를 이용해 "진짜 투수 정체성 x 상황(압박/타자손) x 구종군별 물리 지표(mean+std)"를 다시 시도했다 — §14/§17/§25에서 실패했던 두 갈래(상황 지문 매칭만/정체성만, 상황 조건 없음)를 합친, 이번 세션 이전엔 아직 안 해본 조합.

### 38.1 4개 티어 설계

팀원이 표본 크기를 사전 조사(A=투수x구종군 중앙값 431, B=+압박 202, C=+타자손 212, F=+둘다 100, 완전카운트 축은 40%가 n<10이라 배제)한 결과를 바탕으로 `code/experiment_trackman_pitcherstd_pressure.py`에 4개 티어를 구현:

| Tier | 그룹핑 축 | 피처 수 | 코드 |
|---|---|---|---|
| A | `pitcher_id`×`pitch_type_group` | 64 | `groupby(["pitcher_id","pitch_type_group"])[METRICS].agg(["mean","std"])` |
| B | +`pressure`=`(balls_before>=3)\|(strikes_before>=2)` | 128 | 위 + pressure 축 |
| C | +`batter_hand`(원본 컬럼) | 128 | 위 + batter_hand 축 |
| F | +`batter_hand`×`pressure` 둘 다 | 256 | 위 + 두 축 모두 |

`METRICS`(8개): `rel_speed, spin_rate, induced_vert_break, horz_break, extension, rel_height, rel_side, zone_speed`. 클렌징: `inning>=1`, `balls/strikes/outs_before` 범위, `extension>0`, `zone_speed<=rel_speed`, 완전중복 제거, `pitcher_hand` 다수결 단일화(§14.3와 동일).

### 38.2 asof 컷오프 버그 발견 및 수정

최초 구현은 holdout 시즌(예: holdout=2023이면 검증행 season=2023)에 대해서도 `trackman season<=season`(자기 시즌 포함)을 썼는데, 사용자가 "홀드아웃=2023이면 트랙맨도 2022까지만 써야 하는 거 아니냐"고 지적해 확인한 결과 실제 버그였다 — 실배포(test=2025)는 트랙맨이 2025를 아예 커버 안 해 항상 "목표 시즌 자기 자신은 못 봄" 상태인데, 기존 구현은 그 간극을 재현 못 하고 있었다(시즌 내 미래 경기가 검증에 새는 look-ahead도 겸함). `merge_asof_pitcher_std`를 `cutoff_season = min(row_season, holdout-1)`로 수정해 holdout 시즌 행(학습/검증 무관)은 항상 자기 시즌 트랙맨을 못 보게 클램프했다. `code/experiment_trackman_asof9key.py`도 동일 설계(holdout=2024 검증행에 `season<=2024`=전체 파일)라 같은 결함이 있음을 확인했으나, 그 실험은 이미 기각된 결과라 결론에 영향 없어 미수정.

### 38.3 4티어 × 2홀드아웃 × 3feed 전수 스크리닝 (26개 런, 3-seed)

`--feed-to {both,cat,mlp}`로 CatBoost/MLP 중 어느 쪽에 트랙맨을 줄지도 함께 스윕(단일 모델에만 주는 게 유리할 수 있다는 가설 검증). baseline(tier=none): 2024 Blend=762.40(Cat 723.36/MLP 727.53), 2023 Blend=554.76(Cat 546.52/MLP 514.16).

| Tier | Feed | 2024 Blend (Δ) | 2023 Blend (Δ) |
|---|---|---|---|
| A | both | 798.69 (+36.29) | 523.58 (−31.18) |
| A | cat | 786.35 (+23.95) | 550.95 (−3.81) |
| A | mlp | **825.76 (+63.36)** | 551.28 (−3.48) |
| B | both | 756.75 (−5.65) | 506.52 (−48.24) |
| B | cat | 808.21 (+45.81) | 544.28 (−10.48) |
| B | mlp | 791.43 (+29.03) | 543.55 (−11.21) |
| C | both | 771.33 (+8.93) | 514.87 (−39.89) |
| C | cat | 806.74 (+44.34) | 543.81 (−10.95) |
| C | mlp | 798.68 (+36.28) | 544.27 (−10.49) |
| F | both | 725.80 (−36.60) | 505.46 (−49.30) |
| F | cat | 801.32 (+38.92) | 526.67 (−28.09) |
| F | mlp | 795.07 (+32.67) | 549.74 (−5.02) |

**핵심 관찰 두 가지.** (1) 12개 조합 전부 2023 홀드아웃에서 마이너스 — 예외 없음. (2) `both`(CatBoost+MLP 동시 피딩)는 8개 tier×홀드아웃 쌍 중 7개에서 최악의 feed(평균 −20.7, cat 평균 +12.5, mlp 평균 +16.4) — 한쪽 모델에만 몰아주는 게 낫다는 가설이 강하게 확인됨. tier 순위(cat/mlp 평균): A(+20.0) > C(+14.8) > B(+13.3) > F(+9.6, 파생피처 256개로 제일 무겁고 제일 약함).

### 38.4 ABS 레짐 힌트 — cutoff=7(프로덕션 실측 스플릿)로 재검증

사용자가 "2023은 pre-ABS 레짐이라 다르게 나온 거고, ABS 레짐인 2024/cutoff=7이 더 신뢰할 만한 것 아니냐"고 제안(§34/§37의 ABS 가설과 동일 논리). `--cutoff7` 플래그를 추가해 순수 시즌 홀드아웃 대신 실제 프로덕션 스플릿(학습=season<2024+2024 1~6월, 검증=2024 7~10월)으로 상위 후보 4개를 재검증. baseline(tier=none, cutoff7): Blend=708.06(Cat 622.78/MLP 696.31).

| 조합 | Blend | Δ |
|---|---|---|
| A / mlp | 744.21 | +36.15 |
| B / cat | 719.84 | +11.78 |
| C / cat | 734.46 | +26.40 |
| F / mlp | 698.95 | **−9.11** |

A/B/C 세 후보 전부 cutoff=7에서도 견고하게 플러스 — 2023 실패가 노이즈가 아니라 ABS 레짐 차이였을 가능성에 무게가 실림. F는 cutoff=7에서도 유일하게 마이너스+파생피처 256개로 스크리닝 중 스왑 6.9GB까지 튐(메모리 리스크) → **F 제외, A(→MLP만)+B(→CatBoost만)+C(→CatBoost만) 세 개를 하나의 통합 모델로 프로덕션 반영 결정.**

### 38.5 프로덕션 통합 — 신규 모듈 `code/trackman_pitcher_features.py` + 4개 파일 수정

- `code/trackman_pitcher_features.py`(신규): `clean_trackman`/`build_pitcher_lookup`/`merge_asof_pitcher_std`/`add_all_tiers`(여러 티어 동시 병합)/`build_lookup_full_history`(실전 추론 전용, asof 루프 없이 전체 트랙맨 1개 lookup만).
- `code/blend_model.py`: `make_blend_bundle`/`predict_blend_bundle`에 `cat_feature_cols` 추가 — CatBoost가 학습 안 쓴 A(MLP전용) 컬럼까지 받아 스키마가 어긋나는 걸 막기 위해, 번들에 CatBoost 전용 컬럼 목록을 저장하고 추론 시 그걸로 서브셋.
- `code/train.py`/`code/test.py`: `TRACKMAN_TIER_FEED = {"a":"mlp","b":"cat","c":"cat"}`, `holdout=2024`로 병합(검증 시즌=2024 행은 학습/검증 무관 자기 시즌 트랙맨 배제, §38.2와 동일 원칙).
- `dopip.py` Step 3(Full Retrain): `holdout=2025`로 병합 — **검증 단계와 의도적으로 다른 조건**(모든 학습 행이 자기 시즌까지의 트랙맨을 그대로 봄, `cutoff=min(season,2024)=season`). 실배포(test=2025, 트랙맨 완전 공백)와 분포를 맞추려는 의도였으나 §38.7에서 이게 용의자로 지목됨.
- `submit/script.py`: 위 함수들을 인라인 복제(code/ 패키지 의존 없이 독립 실행) + `submit/model/pitcher_map.csv`를 정적 파일로 동봉(대회 규칙상 문제없음 — train.csv+trackman_history.csv만으로 만든 파생 lookup, test.csv 미사용, 외부데이터 아님).

### 38.6 로컬 검증 — NEW_BEST 승격, Full Retrain 정상 완료

`code/train.py` 단독 실행: Blend Val Score **750.79**(기존 reference 705.34 대비 재현). `dopip.py` 전체 파이프라인 재실행: Step1 753.77(재현 오차 범위 내) → Step2 NEW_BEST(reference 705.34 대비 +48.43) → Step3 Full Retrain(전체 1,369,784행, 7-seed epoch 8~11, CatBoost 1003 iteration) 정상 완료, `submit/model/final_retained_model.pkl`(5.7MB) 생성. `submit.zip` 재구성(`model/final_retained_model.pkl`+`model/pitcher_map.csv`+`script.py`+`requirements.txt`) 후 실제 eval 디렉토리 구조(`./data`+`./model`+`./output`)로 5행 샘플 스모크 테스트 통과.

### 38.7 실제 리더보드: 869.52 — 982.22 대비 −112.70으로 대폭 하락 (기각, 원인 미확정)

로컬은 확실한 개선(+48.43)이었는데 실전은 이 세션에서 가장 큰 폭으로 하락했다 — §31(−13.15)·§33(−3.64)보다 한 자릿수 더 큰 규모. 사후 조사로 확인한 것:

- **크로스워크 커버리지 선택 편향(1순위 용의자)**: 로컬 `open/data/test.csv` 5개 샘플(실제 test.csv 형식 확인용 샘플)로 직접 대조하니 `pitcher_id` 5개 중 **4개(80%)가 `pitcher_map.csv`에 없음**. 스크리닝 때 봤던 "행 기준 커버리지 95.3%"는 train.csv 검증구간(고빈도·주전급 투수가 크로스워크도 잘되고 트랙맨 표본도 풍부해 std가 신뢰할 만한 쪽으로 쏠린 표본) 기준이라 낙관적으로 편향됐을 가능성이 크다. n=5라 확정은 아니나(§18.1의 "단일 소표본 과신 경계"가 여기도 적용), 미매칭 행은 A/B/C 트랙맨 피처가 전부 0으로 채워져 학습 때 본 "0이 아닌 유의미한 값" 분포와 실제 테스트 분포가 어긋났을 개연성이 높다.
- **검증-배포 asof 조건 불일치(2순위 용의자)**: §38.5에서 의도적으로 다르게 설계한 대로, 로컬 753.77을 낸 모델(`train.py`, holdout=2024, season=2024 행은 자기 시즌 트랙맨 배제)과 실제 제출된 모델(`dopip.py` Full Retrain, holdout=2025, season=2024 행도 자기 시즌 포함)이 트랙맨 학습 조건 자체가 다르다 — 검증 점수가 실제 배포 모델의 성능을 보장하지 못했을 수 있다.
- 부차: std=0 fillna가 "표본부족"과 "크로스워크 미매칭"을 구분 못 함, 11만행 검증셋 규모에서 CatBoost B+C(256피처) 국소 과적합 가능성, 애초에 트랙맨 자체가 무신호(§14-16)라는 재확인일 가능성.

**미해결 상태로 세션 종료.** `open/reference/best_model.pkl`/`submit/model/final_retained_model.pkl`은 현재 이번 트랙맨 통합판(869.52 버전)이며 둘 다 미커밋 상태 — git 기준 마지막 커밋(`0906e93`, 982.22 버전)은 그대로 보존돼 있어 `git checkout -- open/reference/best_model.pkl submit/model/final_retained_model.pkl`로 즉시 복원 가능(또는 `open/former_model/former_best_model_v13.pkl`도 동일한 백업). **다음 세션 시작 시 먼저 결정할 것: (a) 982.22로 롤백할지, (b) 크로스워크 커버리지를 실측/개선(예: train.csv 전체가 아니라 test.csv 실제 pitcher_id 분포로 재확인)해서 원인을 마저 특정할지.** 상세 코드/로그: `code/trackman_pitcher_features.py`, `code/experiment_trackman_pitcherstd_pressure.py`, `code/pitcher_crosswalk.py`.

### 38.8 (다음 세션, 2026-08-17) 팀원 제보 "수동 검토 대상" 문서 대조 — `clean_trackman()`의 실제 버그 발견

세션 시작 시 (a) 롤백 (b) 원인 추가 규명 중 (b)를 선택. 팀원이 공유한 트랙맨 수동 검토 문서(행 단위 이상치 6종 + 경기 구조 이슈 4종 + 다중손값 8명 목록)를 `open/data/trackman_history.csv` 1,793,078행 전체와 직접 대조 — 문서의 카운트(이닝<1 1행, 볼/스트라이크/아웃범위밖 1/1/95행, extension≤0 3행, zone_speed>rel_speed 1행, 완전중복 4행, 다중손값 8명)가 전부 정확히 일치함을 확인(문서 자체는 정확).

대조 과정에서 문서에 없던 별개의 실제 버그를 발견: `clean_trackman()`의 `(df["extension"] > 0) & (df["zone_speed"] <= df["rel_speed"])` 조건은 pandas에서 NaN 피연산자를 항상 `False`로 평가하므로, **결측치(extension 7,716행/zone_speed 7,921행/rel_speed 7,617행)를 전부 "이상치"로 오인해 삭제**하고 있었다. 실제 `clean_trackman()` 로그의 "8184행 제거" 중 진짜 이상치는 102행뿐이고 나머지 ~8,119행은 결측치 오필터링. `groupby(...).agg(["mean","std"])`는 컬럼별로 NaN을 자동 스킵하므로(skipna=True 기본값) 이 오필터링은 불필요했다.

수정: `(df["extension"].isna() | (df["extension"] > 0))` 및 `(df["zone_speed"].isna() | df["rel_speed"].isna() | (df["zone_speed"] <= df["rel_speed"]))`로 결측은 통과시키도록 변경(`code/trackman_pitcher_features.py`, `code/experiment_trackman_pitcherstd_pressure.py`, `submit/script.py` 3곳 동기화). 수정 후 제거 행 수: 8184 → **169행**(진짜 이상치 102 + 완전중복 2 + 소수손 마이너리티 65).

### 38.9 버그 수정 후 2023+cutoff7 듀얼체크 재실행 — A/B는 반전, C는 여전히 실패

`code/experiment_trackman_pitcherstd_pressure.py`(수정된 `clean_trackman` 사용)로 채택된 3개 조합을 2023 홀드아웃과 cutoff=7 양쪽에서 재검증(baseline 2023=554.76, cutoff7=708.06):

| 조합 | 버그 있던 버전 2023 Δ | **버그 수정 후 2023 Δ** | 버그 있던 버전 cutoff7 Δ | **버그 수정 후 cutoff7 Δ** |
|---|---|---|---|---|
| A→mlp | −3.48 | **+7.58** | +36.15 | **+32.31** |
| B→cat | −10.48 | **+16.33** | +11.78 | **+28.64** |
| C→cat | −10.95 | **−14.64** | +26.40 | (재검증 불필요, 아래서 최종 제외) |

A/B는 2023(pre-ABS)에서도 확실히 플러스로 반전 — 결측치 오필터링이 진짜 원인의 상당 부분이었음을 시사. C는 버그 수정 후에도 여전히 마이너스(타자손×구종군 축은 표본이 더 희소해 결측치 문제와 무관하게 원래 약했던 것으로 추정) → **C 최종 제외**.

참고로 이 재검증 전에 "0-fill이 문제 아니냐"는 별도 가설(크로스워크에 아예 없는 신인/미매칭 투수 행을 0 대신 NaN으로 남겨 CatBoost 네이티브 처리·MLP는 `SimpleImputer(median)`에 맡기는 방식, `code/experiment_trackman_nanfill.py`)도 2023 홀드아웃으로 테스트했으나 기각됐다 — A→mlp는 −3.48→+1.43(노이즈 수준), B→cat는 −10.48→**−23.65**(악화, CatBoost 단독 546.52→464.65로 무너짐), C→cat는 −10.95→−15.94(악화). 진짜 원인은 0-fill이 아니라 `clean_trackman()`의 결측치 오필터링이었다.

### 38.10 B의 정체 — "CatBoost는 죽는데 메타모델이 구제"하는 가짜 신호였음이 드러남

B→cat의 2023 결과를 모델별로 뜯어보니: **CatBoost 단독 점수는 546.52→490.80으로 오히려 −55.72 악화**됐는데, MLP가 우연히 좋게 나와(544.10) 메타모델이 CatBoost 비중을 낮추는 방향으로 재가중치를 줘서 블렌드만 +16.33으로 구제된 것이었다. 반면 pitchmix(§38.12)는 CatBoost 단독도 546.52→560.13으로 진짜 개선. 즉 B의 "블렌드 플러스"는 진짜 상호보완 신호가 아니라 메타모델의 재가중치 트릭 + MLP 시드 운에 기댄 취약한 결과였다 — §38.11에서 실제로 드러남.

### 38.11 A+B+pitchmix 통합 테스트 — 2023에서 −25.51로 대폭 악화, B가 원흉으로 확인

A(→mlp)+B(→cat)+coarse pitchmix(→cat, §38.12)를 한꺼번에 합쳐 2023 홀드아웃 재검증(`code/experiment_combined_ab_pitchmix.py`): CatBoost=499.63(baseline 546.52 대비 **−46.89**), MLP=508.71(baseline 514.16 대비 −5.45, 노이즈 범위), **Blend=529.25(baseline 554.76 대비 −25.51)**. 개별 검증 델타(+7.58/+16.33/+12.84)가 전혀 더해지지 않고 오히려 이 세션에서 가장 큰 폭으로 악화됐다 — B가 CatBoost에 pitchmix와 동시에 먹여지자 §38.10에서 드러난 CatBoost 훼손이 다시 나타났고, 이번엔 MLP도 좋은 시드가 아니라 메타모델이 구제 못했다. **B 최종 제외 확정.**

### 38.12 팀원 제보 coarse pitchmix — 독립 검증, cat-only로 채택

팀원(XGBoost 트랙)이 `trackman_history.csv`를 `(balls_before, strikes_before, pitcher_hand, batter_hand)` 4축으로만 그룹핑해 `pitch_type_group`(fastball/breaking/offspeed/other) 비율 4개를 피처로 만드는 "coarse pitchmix"를 보고(그쪽 로컬: XGBoost +38.20/CatBoost +25.16/MLP +0.47, 실전 리더보드 927). 투수 정체성도 season도 조인 키가 아니라(볼카운트×손 조합만 사용) 이번 세션 내내 실패 원인이었던 크로스워크 커버리지 문제와 season 구조적 불일치를 둘 다 구조적으로 피해간다는 점에서 A/B/C와 근본적으로 다른 메커니즘.

`code/experiment_coarse_pitchmix.py`로 우리 파이프라인(cutoff=7, F1필터)에서 독립 재검증. **주의**: train.csv는 `pitcher_hand`/`batter_hand`를 정수코드(1/2)로 익명화하지만 trackman_history.csv는 "Right"/"Left" 문자열 — 크로스워크(`pitcher_map.csv`/`batter_map.csv`)로 방향을 역산(투수 568명 중 1명, 타자 506명 중 1명만 불일치): **train 1↔Left, 2↔Right**. 리크 방지: val 시즌 자기 자신의 트랙맨은 테이블 계산에서 제외(`holdout` 파라미터로 `trm[season<holdout]`만 사용, 다른 asof 피처와 동일 관례).

| feed | cutoff7 Blend (Δ) | 2023 Blend (Δ) |
|---|---|---|
| cat | 721.11 (+13.05) | 567.60 (+12.84) |
| both | 715.24 (+7.18) | (미검증, cat이 더 나아 스킵) |

feed=cat이 both보다 나음(MLP는 거의 무변화, 섞으면 오히려 희석) — B/C와 같은 패턴. CatBoost 단독도 cutoff7 622.78→654.14(+31.36), 2023 546.52→560.13(+13.61)로 **진짜 개선**(B와 달리 메타모델 재가중치에 기대지 않음). 2023/cutoff7 양쪽 통과 → **cat-only로 채택**.

### 38.13 최종 조합 확정 — A(→MLP)+pitchmix(→CatBoost), B/C/F 전부 제외

B를 뺀 A+pitchmix만 `code/experiment_combined_ab_pitchmix.py --tiers a`로 재검증:

| | 2023 Blend (Δ) | cutoff7 Blend (Δ) |
|---|---|---|
| A+pitchmix | 564.33 (+9.57) | 742.49 (+34.43) |

A+B+pitchmix(§38.11, −25.51)와 달리 서로 잡아먹지 않고 양쪽 레짐 모두 견고하게 플러스. CatBoost/MLP 각각 진짜로 개선되는 조합만 남기고(A→MLP, pitchmix→CatBoost), CatBoost를 훼손하며 메타모델 재가중치에만 기대던 B를 뺀 것이 핵심. **A+pitchmix를 프로덕션 최종 채택.**

### 38.14 프로덕션 반영 및 로컬 파이프라인 재실행 — Blend 705.34 → 756.01

`code/trackman_pitcher_features.py`에 `compute_coarse_pitchmix`/`merge_coarse_pitchmix` 추가, `TRACKMAN_TIER_FEED = {"a": "mlp"}`로 축소(B/C 제거), `code/train.py`/`test.py`/`dopip.py`/`submit/script.py` 4곳 모두 pitchmix 병합 추가 + `clean_trackman` 버그 수정 반영. `dopip.py` Full Retrain은 pitchmix 테이블을 `holdout=None`(트랙맨 전체 2019~2024)으로 계산해 `submit/model/pitchmix_lookup.csv`(50개 조합)로 정적 동봉(`pitcher_map.csv`와 동일한 관례) — 실전 추론(`submit/script.py`)은 재계산 없이 이 CSV를 그대로 병합만 한다.

`dopip.py` 전체 파이프라인 실행: Step1 Blend Val Score **756.01**(기존 reference 705.34 대비 **+50.67**, 수동 `code/train.py` 단독 실행 시엔 762.57 — 7-seed 시드 노이즈 범위 내) → Step2 NEW_BEST 승격(구 reference가 어제의 869.52 B/C 포함판이라 스키마 자체가 안 맞아 "비교 불가, 신규 채택"으로 처리됨 — 705.34와의 직접 비교는 train.py 로그로 별도 확인) → Step3 Full Retrain(전체 1,369,784행, 7-seed epoch 8~11, CatBoost 919 iteration) 정상 완료. 실제 eval 디렉토리 구조(`./data`+`./model`+`./output`)로 5행 샘플 스모크 테스트 통과(예측값 0.43~0.52). 어제의 869.52 버전은 `open/former_model/former_best_model_v14.pkl`/`former_latest_model_v19.pkl`로 백업 보존.

**실제 리더보드 제출은 사용자가 세션 종료 시점 기준 제출 시간이 아니라 보류.** 다음 제출 가능 시점에 이 버전(로컬 756~763)을 제출해 실전 검증할 것 — 이번엔 §38.11에서 확인했듯 개별 델타가 아니라 "합쳐도 서로 잡아먹지 않는지"와 "개별 모델 스코어가 진짜로 개선되는지"(메타모델 재가중치에 기대지 않는지)까지 확인한 뒤 반영했다는 점이 어제와의 핵심 차이.

## 39. 9~10월 전용 잔차 보정 서브모델 (라우팅/게이팅) — 세 가지 변형 전부 기각

사용자 제안: "9~10월만 경향성이 다르니 이것만 맞추는 서브모델을 만들자." 9~10월 약세 자체는 §34/§35에서 이미 다뤄졌고(ABS 가설 채택→cutoff=7로 대응, weight=5 라우팅은 표본 큰 검증에서 역전해 기각, isotonic 순수 재보정도 §34에서 -13.48로 손해) 이번엔 "main_pred 하나가 아니라 상황 피처까지 보고 국소 편향을 잡는 2단계 잔차 보정 모델"이라는, 이전과 다른 메커니즘으로 재시도. `code/experiment_residual_correction_9_10.py` (production 피처셋: tier A→MLP + coarse pitchmix→CatBoost 기준).

공통 설계: 메인 CatBoost+MLP 블렌드는 정상적으로 train_split 전체로 학습(production과 동일). 검증은 항상 실제 라우팅과 동일하게, 9~10월 예측만 `main_pred + correction_model.predict(...)`으로 교체하고 다른 달은 원래 블렌드 예측 그대로 둔 뒤 전체/9~10월 단독 두 기준으로 채점.

**변형 1 (`--mode other_months`)**: 검증셋(val_split) 내에서 9~10월 이전 달(cutoff7=7~8월, 2023=1~8월)로 잔차모델을 학습해 9~10월에 적용(같은 검증셋 안에서 시간분할, 정답 유출 없음).

| 레짐 | 보정 전(9~10월 단독) | 보정 후 | Δ(9~10월) | 보정 전(전체) | 보정 후 | Δ(전체) |
|---|---|---|---|---|---|---|
| cutoff7(프로덕션) | 495.01 | 486.59 | **-8.42** | 750.39 | 747.71 | **-2.68** |
| season==2023 | 396.97 | 666.31 | **+269.35** | 564.09 | 624.18 | **+60.09** |

두 레짐이 정반대로 갈렸다. 2023 쪽 correction-fit이 8개월(19만행)인 반면 cutoff7 쪽은 2개월(7.5만행)뿐이라 표본 차이가 크고, 보정 후 9~10월 점수(666.31)가 2023 연간 평균(564.09)보다도 높아지는 등 과적합 의심 신호가 뚜렷해 2023 결과를 신뢰하지 않음. **실제 제출에 쓰이는 cutoff7 기준으로 손해 → 기각.**

**변형 2 (`--mode targeted`, 사용자 재요청)**: "9~10월일 때만 학습"을 문자 그대로 구현 — train_split 안의 과거 시즌(2019~2023) 9~10월 행만으로 잔차모델을 학습. 단, 이 행들은 이미 메인 CatBoost 학습에 쓰였으므로 그대로 predict하면 in-sample이라, 9~10월 행을 5-fold로 쪼개 "그 폴드만 뺀 나머지 전체"로 매번 CatBoost를 재학습해 OOF 예측을 만들고(`build_910_oof_residual`), 그 잔차로 CatBoostRegressor를 학습해 실제 val 9~10월에 라우팅.

cutoff7: 보정 전 516.34 → 보정 후 **300.82**(9~10월 단독 Δ **-215.52**), 전체 766.01 → 697.50(Δ **-68.51**) — 변형 1보다 훨씬 크게 악화. 원인: 학습 표본 28만행이 전부 2019~2023년(pre-ABS 또는 ABS 이전) 9~10월이라, ABS 레짐 시프트 이후인 2024년 9~10월에는 그 패턴 자체가 구조적으로 안 맞음 — F1 필터가 고치려던 "오래된 레짐 관계를 새 레짐에 그대로 적용" 함정과 동일한 메커니즘이 새 형태로 재현된 것.

**변형 3 (`--mode targeted --recency-weight 5.0`, 사용자 재요청)**: 변형 2의 원인 진단(오래된 레짐 데이터의 지배)에 대한 처방으로, 2019~2023 중 2024에 가장 가까운 season(2023)의 9~10월 행(54,779행)에 weight=5를 줘서 잔차모델이 최신 패턴을 더 반영하도록 재가중.

cutoff7: 보정 전 513.35 → 보정 후 **217.87**(9~10월 단독 Δ **-295.48**), 전체 765.54 → 671.61(Δ **-93.92**) — 변형 2보다도 더 크게 악화. 재가중이 문제를 완화하기는커녕 키웠다: 2023 한 시즌에 가중치를 몰아주자 여러 연도가 주던 평균화·완충 효과가 줄고, 잔차모델이 (여전히 pre-full-ABS인) 2023 하나의 패턴에 더 좁고 강하게 맞춰져 2024 ABS 레짐과의 괴리가 오히려 커졌다.

**결론 — 세 변형 전부 프로덕션 레짐(cutoff7)에서 명확히 손해, 전부 기각.** weight 유무나 학습 데이터 범위(다른 달 vs 9~10월 전용)를 바꿔도 방향이 바뀌지 않았다 — 근본 문제는 "9~10월 전용으로 쓸 수 있는 학습 데이터가 전부 2024 ABS 레짐 이전 것"이라는 구조적 제약이라, 재가중치 같은 하이퍼파라미터 튜닝으로는 해결되지 않는다.

**핵심 교훈 #25(신규)**: 레짐 시프트 직후 구간(9~10월처럼)을 과거 데이터만으로 특화 학습하는 시도는, 그 학습 데이터 자체가 전부 구(舊) 레짐이라는 근본 문제 때문에 재가중치·데이터 범위 조정 등 어떤 변형을 시도해도 구조적으로 취약하다 — 오히려 최신에 가까운 단일 시즌에 가중치를 몰아줄수록 다양한 연도가 주던 평균화 효과가 줄어 더 나빠질 수 있다. 진짜 신규 레짐 데이터(여기서는 실제 2024 9~10월)가 없는 한 이런 구간 특화 서브모델은 시도하지 않는 게 낫다.

## 40. MLP capacity(깊이) / calibration 스크리닝 — 레이어 추가 전부 기각, weight_decay 대신 시드 수는 부분 효과

팀원이 공유한 calibration 논문(Guo et al. 류, capacity가 커질수록 raw error는 줄지만 ECE는 커진다 / BatchNorm이 calibration을 낮춘다 / weight decay가 클수록 calibration이 좋아진다는 주장)을 계기로, 현재 프로덕션 MLP(`code/mlp_model.py`, 2층: 128→64, wd=0.01)에 레이어를 늘리는 게 이득인지 `code/experiment_mlp_capacity_calibration.py`로 스크리닝. BSS(MLP 단독/블렌드) 외에 ECE(15-bin)도 같이 측정해 논문 메커니즘이 우리 데이터에서 재현되는지 직접 확인. CatBoost는 변형 간 고정, train/val 구성은 §38의 production 피처셋(tier A→MLP + coarse pitchmix→CatBoost) 그대로.

**Round 1 (3-seed 스크리닝, hidden=[128,64,32] 3층 "deeper" + wd=0.05 "deeper_wd"):**

| 레짐 | baseline MLP/ECE/Blend | deeper Δ(MLP/ECE/Blend) | deeper_wd Δ(MLP/ECE/Blend) |
|---|---|---|---|
| cutoff7(프로덕션) | 732.22 / 0.0048 / 744.44 | -16.13 / 악화(0.0054) / **+18.07** | -24.13 / 더 악화(0.0079) / +3.41 |
| season==2023 | 527.91 / 0.0025 / 568.39 | -48.42 / 개선(0.0019) / **-16.48** | -28.00 / 악화(0.0044) / -4.55 |

MLP 단독 점수는 두 레짐 모두 일관되게 나빠졌지만(capacity 증가 자체는 이득 없음), Blend만 레짐마다 정반대로 움직였다(cutoff7 +18.07 vs 2023 -16.48) — §38의 tier B 사례(핵심 교훈 #23, 서브모델 자체는 나빠졌는데 메타모델 재가중으로 블렌드만 좋아 보이는 착시)와 형태가 같아 신뢰하지 않음. weight_decay를 키워도(deeper_wd) ECE가 오히려 더 나빠져(cutoff7: 0.0079, 2023: 0.0044) 논문의 "wd↑→calibration↑" 주장이 우리 설정에서 재현되지 않음 — 조기종료(2~8 epoch, 심지어 2023에선 1~2 epoch)로 끊는 학습 체제가 논문이 가정한 "training loss를 끝까지 밀어붙인 뒤 overconfident해지는" 상황과 다르기 때문으로 추정.

**Round 2 (cutoff7 한정, `--round2`): depth를 4층까지, weight_decay 대신 시드 수로 대조**

- `deep4` = hidden=[128,96,64,32] 4층, wd=0.01, 3-seed. Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data"(2021)의 강한 MLP baseline도 3층(각 256)까지만 쓰고 그 이상은 skip connection이 있는 ResNet 블록으로 넘어간다는 점에 근거해 4층에서 스윕을 멈춤(skip connection 없는 plain MLP는 그 이상 깊이에서 학습이 어려워지는 게 정설).
- `deeper_7seed` = hidden=[128,64,32] 3층, wd=0.01이지만 SCREEN_SEEDS(3)이 아닌 프로덕션 ENSEMBLE_SEEDS(7)로 학습. Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"(2017) — 서로 다른 랜덤 초기화 앙상블 평균이 calibration 개선의 "gold standard"라는 근거로, weight_decay 대신 시드 수를 capacity 증가의 대응 축으로 시도.

| variant | MLP solo | ECE | Blend | ΔMLP | ΔBlend |
|---|---|---|---|---|---|
| baseline(재실행) | 713.56 | 0.0085 | 756.68 | — | — |
| deep4 | 690.74 | 0.0066 | 751.58 | -22.82 | -5.11 |
| deeper_7seed | 708.97 | 0.0066 | 738.56 | -4.59 | **-18.12** |

**중요한 부수 발견 — 3-seed 스크리닝 자체의 노이즈가 크다.** 이번 baseline 재실행(713.56 / ECE 0.0085 / 756.68)이 Round 1의 baseline(732.22 / ECE 0.0048 / 744.44)과 완전히 동일한 시드([42,123,7])인데도 MLP 단독 ~19점, ECE는 거의 2배 차이가 났다(GPU 연산 비결정성 — embedding backward의 atomic add 등으로 추정). 즉 Round 1에서 관찰한 deeper의 "+18.07 Blend 개선"은 이 노이즈 범위 안에 있을 가능성이 크다.

**deeper_7seed가 이 의심을 정리해줬다**: 시드를 3→7개로 늘리자 MLP 단독 하락폭이 -16.13(Round1, 3-seed)에서 **-4.59**로 뚜렷이 줄었다 — Lakshminarayanan의 "앙상블이 개별 모델의 불안정성을 평균으로 상쇄한다" 메커니즘이 실제로 관찰된 것. 하지만 정작 Blend는 -18.12로 더 나빠졌다 — Round 1의 +18.07이 시드를 늘려 더 안정적으로 측정하니 사라지고 season==2023(-16.48)과 같은 방향(악화)으로 수렴했다. 즉 **두 가지 독립적인 방법(다른 레짐 확인 / 시드 수 확대)이 같은 결론(Round 1의 개선은 노이즈)으로 수렴**했다.

**결론 — 4번의 독립 체크(cutoff7 3-seed / season2023 3-seed / cutoff7 4층 / cutoff7 3층-7seed) 전부 depth 증가에 반대되는 신호. MLP 레이어 추가(3층·4층 모두)는 기각.** weight_decay 강화도 두 라운드 모두 도움이 안 됨. 시드 수 확대는 "capacity 증가로 생긴 solo 모델의 불안정성을 완화"하는 데는 논문 예측대로 유효했지만, 그 자체로 아키텍처 변경을 정당화할 만큼 net-positive를 만들어주지는 못했다 — 즉 이 프로젝트에서 시드 수 확대는 여전히 "기존 아키텍처의 분산을 줄이는 도구"이지 "새 capacity를 정당화하는 도구"는 아니다.

**핵심 교훈 #26(신규)**: 3-seed 스크리닝 단계의 run-to-run 노이즈(동일 시드라도 재실행 시 GPU 비결정성으로 MLP 단독 점수 ~19점, ECE 거의 2배까지 벌어짐)는, 스크리닝에서 관찰되는 중간 크기 델타(수십 점 이내)를 그대로 신뢰하기엔 부족하다는 근거가 된다. 특히 "서브모델 자체는 나빠졌는데 Blend만 좋아지는" 패턴(핵심 교훈 #23)이 한 레짐에서만 관찰됐다면, 그건 실제 개선이 아니라 이 노이즈이거나 메타모델 재가중 착시일 가능성이 높다 — 두 번째 레짐 확인이나 시드 수 확대 중 하나로 반드시 재확인해야 한다.

**Round 3 (cutoff7 한정, `--resnet`): skip connection이 있는 진짜 ResNet 블록으로 재시도**

Round 1/2에서 plain deeper/deep4가 손해였던 게 "skip connection 부재로 깊은 네트워크가 잘 학습되지 않아서"인지 직접 확인하기 위해, Gorishniy et al.(2021)의 tabular ResNet 블록(BN → Linear(d→d·hidden_factor) → ReLU → Dropout → Linear(d·hidden_factor→d) → Dropout → residual add, 여기선 d=128, hidden_factor=2)을 2/4/6개 스윕(`ResNetTabularMLP`, `ResBlock`).

| variant | MLP solo | ECE | Blend | ΔMLP | ΔBlend |
|---|---|---|---|---|---|
| baseline(재실행) | 733.84 | 0.0033 | 760.52 | — | — |
| resnet2(2블록) | 696.65 | 0.0041 | 746.59 | -37.19 | -13.93 |
| resnet4(4블록) | 691.04 | 0.0042 | 728.09 | -42.80 | -32.43 |
| resnet6(6블록) | 678.26 | 0.0058 | 721.98 | -55.59 | -38.54 |

Round 1의 plain deeper와 달리 이번엔 **레짐 반전 없이 블록 수가 늘수록 MLP 단독/Blend/ECE 전부 단조롭게 나빠졌다** — 노이즈나 메타모델 재가중 착시로 설명하기 어려운, 그 자체로 일관되고 명확한 부정 신호라 season==2023 교차검증 없이도 결론을 내림. skip connection을 넣어도 이 문제에서는 capacity 증가 자체가 득이 안 됨을 확인.

**결론(Round 1~3 종합) — plain 3·4층, ResNet 2·4·6블록 등 시도한 모든 depth 증가 변형이 기각.** 이 파이프라인에서 MLP의 표현력은 병목이 아닌 것으로 보인다: CatBoost가 이미 비선형 상호작용을 잘 잡고 있고(메타모델 가중치 w_cat≈w_mlp로 균형), MLP는 블렌드에 "다르게 틀리는" 다양성을 더하는 역할이 커서 solo 표현력을 키우는 것 자체가 큰 의미가 없는 것으로 추정. (LR/weight_decay/dropout을 2층 baseline 값 그대로 재사용했다는 한계는 있음 — 깊은 아키텍처 전용 하이퍼파라미터 재튜닝은 시도하지 않았다.)

**핵심 교훈 #27(신규)**: 이 프로젝트의 MLP는 "표현력을 늘리면 더 잘 맞힌다"는 일반적인 딥러닝 직관이 통하지 않는 영역이다 — plain depth 증가와 ResNet 스타일 skip connection 둘 다, 정규화(weight_decay)나 구조적 개선(residual)으로 문제를 해결하지 못하고 오히려 일관되게 손해였다. 이 프로젝트에서 MLP의 역할은 "CatBoost와 다르게 틀려서 스태킹 블렌드에 다양성을 더하는 것"에 가깝고, MLP 자체의 단독 표현력을 늘리는 방향(레이어 추가, 폭 확장 — 이전에도 기각됨)은 반복적으로 기각되어 왔다. 앞으로 "MLP를 더 크게/깊게 만들자"는 제안이 나오면 이 결과부터 참고할 것.

## 41. §38 tier A+coarse pitchmix 실전 제출 — 950.81 (982.22 대비 −31.41)

**제출물**: `submit.zip`, 2026-08-17 14:07 빌드 (`final_retained_model.pkl` 14:00, `script.py` 13:00, `pitchmix_lookup.csv` 13:42, `pitcher_map.csv` 8/17 00:22). 구조 검증(최상위 `model/`+`script.py`+`requirements.txt`, 추가 폴더 없음) 및 `TRACKMAN_TIER_FEED = {"a": "mlp"}` + `PITCHMIX_COLS`→CatBoost 반영 확인 완료 후 2026-08-18 제출.

**로컬 vs 실전**:

| | cutoff7 로컬 블렌드 | 실제 대시보드 |
|---|---|---|
| 982.22 버전(트랙맨 없음, §36/§37 커밋) | 705.34 | 982.22 |
| tier A+coarse pitchmix(§38, 미커밋) | 756.01 | **950.81** |

Δ로컬 +50.67 vs Δ실전 **−31.41** — 로컬 개선이 실전에서 뒤집힌 세 번째 사례(§30 MLP 재탐색 −13.15, §32 CatBoost GPU 재탐색 −3.64에 이어)이자, 트랙맨 재도입 자체로는 두 번째 실전 하락(§37의 tier A+B+C, −112.70)이다. §38에서 두 레짐(2023/cutoff7) 교차검증과 서브모델별 개별 개선(메타모델 재가중 트릭이 아님을 확인)까지 거쳤다는 점에서 §37보다 훨씬 신중하게 검증된 변경이었는데도 다시 마이너스로 나왔다 — 다만 하락폭 자체는 §37(−112.70)보다 훨씬 작다.

**하이퍼파라미터 재탐색 여부 점검(결과 자체에 대한 조치는 아직 안 함)**: `CATBOOST_PARAMS`(§21, F1 필터 직후 재탐색값)와 MLP `LR`/`WEIGHT_DECAY`/`DROPOUT`/`BATCH_SIZE`(§30 재탐색 기각 후 원복된 기본값) 둘 다 cutoff=7 분할 도입(§36)과 tier A+pitchmix 도입(§38) 이후 한 번도 재탐색되지 않은 채 이번 제출에 쓰였다. 원인 후보로 기록만 해두고 이번 세션에서는 검증하지 않았다.

**미해결 상태로 세션 종료**: `open/reference/best_model.pkl`/`submit/model/final_retained_model.pkl`은 950.81 버전(미커밋)으로 남아있고, 982.22 버전(마지막 커밋)은 `git checkout`으로 즉시 복원 가능. 롤백 여부·원인 규명 둘 다 다음 세션 과제.

## 42. TabNet 3rd-model 상관관계 진단 + 본검증 — 기각

`code/experiment_tabnet_correlation.py`. 전날 밤 시도가 컴퓨터 재시작으로 CatBoost 학습 중(iteration 946/1500) 죽어 결과가 전혀 안 남았던 것을 재실행했다. 1-seed 진단(이미 스크립트 docstring에 기록돼 있던 결과, cutoff7): TabNet Val=548.53, corr(TabNet,CatBoost)=0.8198, corr(TabNet,MLP)=0.7898 — "확실히 낮음(≤0.7)" 기준 미달의 애매한 지대라 본검증(`--ensemble`, 7-seed) 투자를 진행.

**cutoff7 (프로덕션 레짐, `--cutoff7 --ensemble`)**:
```
[CatBoost] Val Score=654.14 (best_iteration=896, 113.6s)
[MLP(7-seed)] Val Score=741.83 (970.6s)
[TabNet(7-seed)] Val Score=642.41 (3540.0s)
[상관관계] TabNet vs CatBoost: 0.8952 | TabNet vs MLP: 0.9026
[스태킹 이득] 2-way Blend=759.09 | 3-way(+TabNet) Blend=759.97 (Δ+0.88, weights=['1.257', '2.066', '0.828'])
```

**season==2023 (`--holdout 2023 --ensemble`)**:
```
[CatBoost] Val Score=560.13 (best_iteration=742, 69.1s)
[MLP(7-seed)] Val Score=519.21 (567.9s)
[TabNet(7-seed)] Val Score=465.31 (2215.0s)
[상관관계] TabNet vs CatBoost: 0.8533 | TabNet vs MLP: 0.8382
[스태킹 이득] 2-way Blend=562.28 | 3-way(+TabNet) Blend=574.20 (Δ+11.92, weights=['1.922', '1.249', '1.243'])
```

두 레짐 다 부호는 플러스(ExcelFormer §22처럼 부호 반전은 없음)지만, 상관관계가 둘 다 0.85 안팎 — 이미 실패로 판명난 ExcelFormer의 0.88~0.93 구간과 사실상 겹친다. 델타 크기(+0.88, +11.92)도 §40에서 실측한 순수 GPU 비결정성 노이즈(동일 시드 재실행만으로 MLP 단독 스코어가 ~19점 벌어짐)보다 작거나 비슷한 수준이라, 노이즈와 구분되는 신호로 보기 어렵다.

**결론 — 기각.** TabNet은 sparsemax 기반 sequential feature masking으로 FT-Transformer/ExcelFormer(dense self-attention)와는 다른 메커니즘이지만, "신경망 계열 3번째 모델은 MLP와 상관관계가 구조적으로 높다"(핵심 교훈 #7)는 결론에는 도달 방식이 달라도 동일하게 수렴했다. 사용자 판단으로 로컬 스크리닝 단계에서 종료 — 실전 제출, rolling-origin fold check는 진행하지 않음(추가로 필요하면 §23과 동일한 절차로 재개 가능).

## 43. §41 하이퍼파라미터 재탐색 — CatBoost/MLP 둘 다 재검증에서 기각

`code/tune.py`/`code/tune_mlp.py`의 `build_data()`를 `code/experiment_residual_correction_9_10.py::build_split(2024, cutoff7=True)` 기준(현재 프로덕션 피처/분할)으로 교체하고 Optuna 재탐색을 돌렸다.

**CatBoost (40 trials)**: Best solo BSS=0.00660(Score 659.84), `depth=6, learning_rate=0.0508, l2_leaf_reg=1.83, random_strength=3.67, bagging_temperature=0.558, border_count=222, min_data_in_leaf=1, best_iteration=549`. 프로덕션 값(solo 654.14) 대비 +5.70.

**MLP (20 trials×3-seed screen)**: Best Score(3-seed)=749.46, `lr=0.004138, weight_decay=0.00082, dropout=0.139, batch_size=1024`.

**7-seed+두 레짐 재검증** (`code/experiment_hparam_reverify.py`, 신규 작성 — CatBoost는 `CATBOOST_TUNABLE_KEYS` 7개만 override, MLP는 `mlp_model.DROPOUT` 모듈 전역을 트라이얼처럼 임시 override):

```
=== cutoff7 ===
[CatBoost-baseline] Val Score=654.14 | [CatBoost-tuned] Val Score=659.84
[MLP(7-seed)-baseline] Val Score=768.98 | [MLP(7-seed)-tuned] Val Score=768.41
CatBoost=baseline+MLP=baseline -> Blend=775.72
CatBoost=baseline+MLP=tuned   -> Blend=777.75
CatBoost=tuned+MLP=baseline   -> Blend=779.22
CatBoost=tuned+MLP=tuned      -> Blend=780.61  (Δ+4.89 vs baseline+baseline)

=== season==2023 ===
[CatBoost-baseline] Val Score=560.13 | [CatBoost-tuned] Val Score=530.91
[MLP(7-seed)-baseline] Val Score=518.79 | [MLP(7-seed)-tuned] Val Score=503.80
CatBoost=baseline+MLP=baseline -> Blend=562.84
CatBoost=baseline+MLP=tuned   -> Blend=553.99
CatBoost=tuned+MLP=baseline   -> Blend=548.80
CatBoost=tuned+MLP=tuned      -> Blend=539.13  (Δ−23.71 vs baseline+baseline)
```

season==2023에서 CatBoost 튜닝(−29.22)·MLP 튜닝(−14.99) 둘 다 단독으로도 명확히 악화 — cutoff7 단일 홀드아웃(11만행)에 대한 Optuna 과최적화로 판단된다. 평균 Δ(+4.89, −23.71 평균 ≈ −9.4)도 순손실이고 레짐 간 부호가 갈린다(ExcelFormer §22-23과 같은 실패 모양).

**결론 — 하이퍼파라미터 재탐색 전부 기각, 프로덕션 값 유지.** §41의 원인 후보였던 "하이퍼파라미터 미재탐색"은 이걸로 배제된다 — 재탐색이 원인이 아니라, 재탐색 자체가 (제대로 검증하면) 오히려 손해였다.

## 44. tier A 제거 검토 — 팀원 리포트 2건 교차확인 + pitchmix-only 7-seed 재검증 (미완료)

사용자가 tier A(투수 크로스워크 x pitcher-std, →MLP) 제거를 제안하며 팀원의 독립 실험 리포트 2건을 공유했다.

**리포트 1 (선수단위 파생 피처)**: 릴리스 일관성(rel_h_std/rel_s_std/ext_std/release_dispersion, 구종군×4=12개) — "거의 변동 없음" 기각. 구종 엔트로피/dominant pitch rate — 기각. FB-BR 물리량 gap(speed/spin/IVB/hbreak, 4개) — "전년도 집계로는 중요 정보 안 됨" 기각. **count-mix(투수 정체성×볼카운트별 구종 비율)** — "로컬에서 올랐으나 LB에서 떨어짐" 기각. 마지막 것이 구조적으로 우리 tier A(정체성×상황)와 동형 — §37 tier A/B/C 첫 시도(로컬 +48.43→실전 −112.70)와 같은 실패 모양을 팀원이 독립 재현.

**리포트 2 (coarse pitchmix 상세)**: 우리가 채택한 것과 동일 피처. XGBoost +38.20/CatBoost +25.16/MLP +0.47(2023+2024 이중검증, 리크수정판), 실전 927(구종비중만, F1필터 없음). MLP 무변화 이유("이미 balls/strikes/hand를 원재료로 받아 상호작용을 스스로 학습")가 §38.12 우리 결론과 정확히 일치 — 독립 교차확인.

**pitchmix-only(tier A 제거) 7-seed 재검증** (`code/experiment_pitchmix_only_reverify.py`, 신규 작성 — `experiment_residual_correction_9_10.TIER_FEED`를 `{}`로 monkeypatch):

```
=== cutoff7 (pitchmix-only) ===
[CatBoost] Val Score=654.14 (best_iteration=896, 207.3s)
[MLP(7-seed)] Val Score=699.35 (856.0s)
[Blend] Score=716.73

=== season==2023 (pitchmix-only) ===
[CatBoost] Val Score=560.13 (best_iteration=742, 157.2s)
[MLP(7-seed)] Val Score=534.64 (544.1s)
[Blend] Score=566.30
```

tier A 포함 버전(§41/§43 baseline, 동일 스크립트 기준: cutoff7 Blend=775.72, 2023 Blend=562.28)과 비교하면 tier A를 뺄 때 **cutoff7은 −58.99(A가 크게 도움), 2023은 +4.02(A가 오히려 손해)** — 정반대. CatBoost는 tier A를 애초에 안 받으므로(→MLP만) 두 구성에서 완전히 동일(654.14/560.13), 이건 스크립트 정합성 확인(sanity check)이기도 하다.

**핵심 함정**: 이 로컬 비교로는 §37의 크로스워크 커버리지 편향 가설(핵심 교훈 #21: "검증 train.csv 기준 커버리지가 실제 test.csv보다 낙관적으로 편향됐을 가능성")을 판별할 수 없다. cutoff7이든 2023이든 검증셋과 `pitcher_map.csv` 크로스워크가 전부 train.csv 유래라, 레짐(연도)을 바꿔도 이 편향 구조 자체는 동일하게 남는다 — cutoff7에서 관측된 큰 tier A 이득이 진짜 test.csv(2025, 신인/은퇴로 커버리지가 더 나쁠 가능성)에서 재현될지는 로컬 검증으로 원천적으로 확인 불가능하다.

**pitchmix-only 전용 MLP 하이퍼파라미터 재탐색**: `tune_mlp.py`를 `TIER_FEED={}` 기준으로 재조정해 20 trials×3-seed 재탐색 — Best Score(3-seed)=725.14, `lr=0.008922, weight_decay=0.007795, dropout=0.4982, batch_size=2048`(`open/temp/best_mlp_hparams_pitchmix_only.json`). CatBoost는 tier A 유무와 무관하게 `cat_features`가 동일해 재탐색 불필요(§43 결과 그대로 적용, 이미 기각 확정이라 이번 재검증에는 baseline만 사용).

이 3-seed 결과의 7-seed+두 레짐 재검증(`experiment_hparam_reverify.py --pitchmix-only --skip-catboost-tuned --mlp-json .../best_mlp_hparams_pitchmix_only.json`)을 cutoff7/2023 양쪽에서 시작했으나 **컴퓨터 종료를 위해 완료 전 중단** — 로그 파일 남아있지 않음(프로세스만 시작하고 SIGTERM으로 정지). 다음 세션에서 재실행 필요.

**미해결 상태로 세션 종료.** 다음 세션 과제: (1) 위 재검증 마무리, (2) tier A 포함/제외 중 실전 제출로 최종 확인 — 로컬 레짐이 정반대로 갈리는 상황에서는 실전 제출 외에 판단 방법이 마땅치 않다.

## 45. §44 재검증 마무리(하이퍼파라미터 재탐색 전부 기각 확정) + tier A 제거 결정 + 투수/타자 "시즌 진행분" 8개 채택 (실전 제출 대기)

**1) §44 중단된 재검증 완료.** `experiment_hparam_reverify.py --pitchmix-only --skip-catboost-tuned --mlp-json .../best_mlp_hparams_pitchmix_only.json`을 cutoff7/season==2023 양쪽으로 재실행:

| 레짐 | Blend baseline hparam | Blend 재탐색 hparam | Δ |
|---|---|---|---|
| cutoff7 | 715.23 | 728.37 | **+13.14** |
| season==2023 | 562.37 | 549.33 | **−13.04** |

tier A 포함 버전(§43)과 정확히 같은 패턴 — cutoff7 단일 홀드아웃 과최적화. **pitchmix-only에 대해서도 재탐색 하이퍼파라미터 기각, 프로덕션 baseline 유지.** 이걸로 하이퍼파라미터 재탐색 경로는 tier A 유무와 무관하게 완전히 닫혔다.

**2) tier A 유지/제거 결정 — 실전 증거로 확정.** §44에서 로컬은 레짐마다 정반대(cutoff7: tier A 있는 쪽이 +58.99 유리 / season==2023: 없는 쪽이 +4.02 유리)라 로컬만으로 판단 불가했다. 그런데 tier A+pitchmix 조합이 이미 **실전 리더보드에 제출되어 950.81(−31.41, 982.22 대비)**로 확인된 상태였다 — 이 실전 결과 자체가 tier A 제거 방향(season==2023 레짐이 시사하는 방향)과 일치하는 직접 증거이므로, **tier A 제거(pitchmix-only)로 확정**했다. `code/train.py`/`submit/script.py`의 `TRACKMAN_TIER_FEED`를 `{"a": "mlp"}` → `{}`로 변경(coarse pitchmix는 유지 — tier A와 무관하게 별도 병합).

**3) 팀원 제보 피처 — 투수/타자 "시즌 진행분" 8개.** 팀원이 CatBoost 단독(F1필터, depth=6, 466 trees)으로 로컬 813.303을 보고한 피처: `asof_{pitcher,batter}_success_rate`는 커리어 전체 누적이라 베테랑일수록 최근 시즌 컨디션 신호가 희석되는데, 각 행마다 "직전 시즌 마지막 행"의 누적치(+그 행 자신의 결과)를 시즌 시작 시점 기준값으로 잡고 빼서 "이번 시즌만의" 성공률(`{role}_season_success_rate`)과 커리어 누적 대비 격차(`{role}_season_rate_gap`)를 분리해낸다 (`{role}_season_n`, `_success_count` 포함 총 8개). 직전 시즌은 이미 완결된 과거라 스플릿 경계(cutoff7의 2024 H2, season==2023 홀드아웃)를 넘어 참조해도 미래 정보 누출이 아니다.

teammate 숫자는 우리 스플릿/피처셋과 다를 수 있어 직접 재현 검증(`code/experiment_season_progression.py`, pitchmix-only 트랙맨 구성, CatBoost 단독):

| 레짐 | 피처 없음 | +시즌진행분 8개 | Δ |
|---|---|---|---|
| cutoff7 | 654.14 | 688.80 | **+34.66** |
| season==2023 | 560.13 | 710.89 | **+150.76** |

baseline 수치가 §44/§43의 동일 조건 수치와 정확히 일치해 스크립트 신뢰성 확인됨. 프로덕션 블렌드(CatBoost+7-seed MLP)로 재검증(`code/experiment_season_progression_blend.py`):

| 레짐 | CatBoost Δ | MLP(7-seed) Δ | Blend Δ |
|---|---|---|---|
| cutoff7 | +34.67 | +47.55 | **+35.90** (709.78→745.69) |
| season==2023 | +150.76 | +156.66 | **+164.90** (563.03→727.93) |

**이 프로젝트에서 트랙맨류 시도는 거의 전부 레짐마다 방향이 갈렸는데, 이 피처는 두 모델·두 레짐 전부 같은 방향으로 크게 개선** — 지금까지 시도 중 가장 신뢰도 높은 피처 추가. 채택.

**4) 코드 통합 — 정적 lookup 분리(중요 버그 예방).** 처음엔 "lookup 테이블 구성 + 적용"을 하나의 함수(`add_season_progression_features`)로 합쳐뒀는데, 이대로 `submit/script.py`에 그대로 복제하면 심각한 문제가 있었다: 추론 시 `test.csv`(season 2025)는 과거 시즌 행도 `control_success` 정답도 없어서, "직전 시즌 마지막 행" lookup을 **test.csv 자기 자신으로부터** 계산하게 되는 구조가 된다 — 이는 대회 규칙이 명시적으로 금지하는 "평가 데이터 내 다른 행을 이용한 피처 생성"에 해당할 뻔했다. `pitcher_map.csv`/`pitchmix_lookup.csv`와 동일한 관례로 **"lookup 구성(`build_season_end_lookup`, train.csv 전체로만 가능)"과 "적용(`apply_season_progression_features`, df 자신에 의존하지 않음)"을 분리**해 해결. `dopip.py` Full Retrain 단계에서 train.csv 전체(2019~2024)로 lookup을 계산해 `submit/model/season_end_lookup.csv`로 정적 동봉하고, `submit/script.py`는 재계산 없이 이 테이블을 병합만 한다(`add_season_progression` 함수, 수동 동기화).

**5) 파이프라인 실행 및 검증.** `code/train.py` 실행(cutoff7, pitchmix-only+시즌진행분) → Blend Val Score **748.87** (실험 스크립트의 745.69와 근사, 정상 범위 차이). `code/test.py`가 기존 reference(950.81, tier A 컬럼 보유)와 피처 스키마 불일치를 감지해 자동으로 `NEW_BEST` 채택(트랙맨 tier A 컬럼이 새 실행엔 존재하지 않음 — dict 포맷은 호환되지만 컬럼 집합이 달라 비교 불가, `code/test.py`의 기존 "호환 안 되면 신규 채택" 로직이 이 경우도 정상 처리). `dopip.py` 전체 파이프라인 실행 → `open/reference/best_model.pkl` 승격(구 버전은 `open/former_model/former_best_model_v15.pkl`로 백업), `submit/model/final_retained_model.pkl` 재생성(3.5MB), `season_end_lookup.csv`(4653행) 신규 동봉. 경쟁 서버와 동일한 폴더 구조(`data/`+`model/`+`output/`)로 `submit/script.py`를 직접 실행해(sample_submission.csv은 test.csv row_id로 합성) end-to-end 드라이런 — 정상적으로 `output/submission.csv` 생성 확인(5개 예측값 0.41~0.51 범위, 합리적).

**상태: 로컬 후보 완성, 실전 리더보드 미제출.** 로컬 cutoff7 블렌드 748.87은 트랙맨 도입 전 최고 기록(756.01, tier A+pitchmix 포함, §38)에 근접하면서도 tier A의 실전 리스크(950.81 하락 전례, §41)가 빠진 구성이다. 다음 실전 제출로 실제 효과를 확인해야 한다 — 특히 tier A 제거 자체는 여전히 "실전 증거 기반 추정"이지 로컬로 재확인된 게 아니므로(위 2번 항목), 이번 제출이 그 결정에 대한 첫 실전 확인이기도 하다.

**6) 검증 보강 — 트랙맨 유무 x 시즌진행분 유무 2x2 분해.** 위 3번 검증은 pitchmix-only 배경 하나로만 이뤄져 "시즌진행분 이득이 pitchmix와의 상호작용에서 나온 착시가 아닌지"를 배제하지 못한다는 지적이 나와, `code/experiment_season_progression_no_trackman.py`로 "트랙맨 완전 미사용(순정 982.22 피처셋)" 배경에서도 CatBoost 단독으로 재확인했다:

| | 트랙맨 없음(순정 982.22) | pitchmix-only |
|---|---|---|
| cutoff7, 시즌진행분 없음 | 622.78 | 654.14 |
| cutoff7, +시즌진행분 | 656.80 (Δ+34.02) | 688.80 (Δ+34.66) |
| season==2023, 시즌진행분 없음 | 546.52 | 560.13 |
| season==2023, +시즌진행분 | 648.10 (Δ+101.58) | 710.89 (Δ+150.76) |

시즌진행분의 증분 효과는 pitchmix 유무와 무관하게 양쪽 다 크게 양수(cutoff7은 거의 동일한 크기, 34.02 vs 34.66 — 상호작용 거의 없음). pitchmix의 증분 효과도 시즌진행분 유무와 무관하게 양쪽 다 양수이고, season==2023에서는 시즌진행분이 있을 때 오히려 더 커진다(+13.61→+62.79) — 상쇄가 아니라 보완 관계. **네 조합 중 "pitchmix+시즌진행분"이 두 레짐 모두 최고점** — 3번 결과가 pitchmix 배경의 착시가 아님을 확인했고, 현재 채택된 구성(pitchmix-only+시즌진행분)이 이 4개 조합 중 로컬 최선임도 재확인됐다.

**7) 실전 제출 결과 (2026-08-19) — 확정.** 이 후보(`submit.zip`, pitchmix-only+시즌진행분)를 실전 리더보드에 제출: **1027.54**. 이전 최고 982.22 대비 **+45.32**, tier A 포함 버전 950.81 대비 **+76.73**. 위 2번(tier A 제거)과 3번(시즌진행분 채택) 결정 둘 다 실전으로 확정 확인됨 — 특히 시즌진행분은 로컬(cutoff7 블렌드 +35.90)보다 실전에서 오히려 더 크게 개선됐다. 랭킹권 진입 기준은 약 1150점으로 전달받음 — 다음 목표는 이 ~122점 격차를 메우는 것.
