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
