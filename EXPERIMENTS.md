# 실험 기록 (Feature Engineering & Hyperparameter Tuning)

`control_success` 예측 CatBoost 모델에 적용한 피처 엔지니어링과 하이퍼파라미터 튜닝 내역을 정리합니다. 기준은 이번 작업 시작 시점의 `open/reference/best_model.pkl` (BSS 0.00737 / 점수 736.61)입니다.

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
