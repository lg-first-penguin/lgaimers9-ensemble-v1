# Trackman 피처 활용 전략 실험 계획

작성일: 2026-08-21  
상태: 계획 수립 완료, 1단계 실행 대기

## 1. 목표

Train과 고신뢰로 연결된 2019~2024 Trackman 이력을 활용해 제구 성공확률 예측의 Brier score를 개선한다. 단순히 Trackman 평균값을 기존 CatBoost에 대량으로 붙이는 것이 아니라, 다음 두 가지를 검증한다.

1. 투수의 구종별 동작 반복성·레퍼토리·상황별 선택 패턴이 제구와 연결되는가
2. Trackman 신호가 전체 예측을 바꾸기보다, 기존 모델의 불확실한 부분만 보정할 수 있는가

최종 제출 추론은 행별 pre-pitch 정보만으로 독립적으로 실행되어야 한다.

## 2. 반드시 지킬 규칙과 정보 시점

- 현재 투구의 Trackman 측정값, 실제 구종, 실제 위치·결과는 최종 추론 피처로 사용하지 않는다.
- 행의 시즌이 `Y`이면 Trackman 통계는 `season < Y` 자료만 사용한다.
- 2024 검증 행에는 2019~2023 Trackman만 사용한다.
- 실제 평가 제출에서는 2025 Trackman을 사용하지 않는다.
- test의 다른 행을 이용한 집계·보정·정렬·rolling을 하지 않는다.
- 고신뢰 매핑만 사용한다. 충돌·미확정·추정 보완 행은 결측으로 처리하거나 별도 분석한다.
- Trackman 표본이 부족한 투수·구종군은 무리하게 값을 만들지 않고 shrinkage 또는 결측 fallback을 사용한다.
- 모든 피처는 해당 행의 투구 시점 이전에 알 수 있는 정보인지 코드와 리포트에서 확인한다.

공식 규칙·최신 운영진 답변이 이 문서보다 우선한다. 특히 현재 투구 Trackman을 auxiliary/teacher target으로 사용하는 방식은 공식 허용 범위를 다시 확인하기 전까지 제출 후보에서 분리한다.

## 3. 데이터와 매핑 기준

주요 입력은 다음과 같다.

- Train: `open/data/train.csv`
- Trackman 매칭 결과: OneDrive `phase2/train_trackman_result_share_20260820`
- 실행 코드: `code/`, `tools/trackman/`
- 실험 산출물: `open/experiments/`

현재 확인된 매핑은 Train 약 147만 행 중 약 127.5만 행이 연결되고, Trackman ID 중복 확정과 공통 10개 상태 컬럼의 불일치는 제거된 상태다. 다만 행 단위 연결률과 투수별 과거 표본 수는 피처별로 별도 보고한다.

## 4. 현재까지의 근거와 가설

### 4.1 이미 시도했으나 채택하지 않은 방향

- 누적 투수 평균·최근 1/3/5/10구 요약
- 직전 시즌 고정 투수 프로필
- Trackman 평균 구속·회전수·무브먼트·구종 비율의 직접 추가
- exact-only 및 매칭 여부 기반 변형
- CatBoost distillation, teacher disagreement weighting, student blend

이 계열은 기준 CatBoost보다 안정적인 개선을 만들지 못했다. 특히 공식 pre-pitch 피처의 구종 비율과 Trackman 과거 구종 비율도 높은 상관을 보여, 단순 평균 프로필은 이미 모델이 가진 정보를 반복할 가능성이 높다.

### 4.2 아직 검증하지 않은 핵심 가설

- 전체 구종을 섞은 평균보다 구종군 내부의 변동성이 제구와 가깝다.
- 투수 고정 프로필보다 현재 카운트·타자 손잡이·game_type에 조건부인 예상 구종과 물리값이 행별 신호를 만든다.
- 투구 자체의 평균 능력보다 구종 간 릴리스·무브먼트 관계와 터널링 proxy가 중요할 수 있다.
- 절대값보다 직전 시즌 대비 변화와 반복성 악화가 중요할 수 있다.
- 약한 신호를 전체 CatBoost에 직접 넣기보다 OOF 잔차를 작은 Trackman 모델로 보정하는 편이 안전할 수 있다.

## 5. 공통 검증 설계

### 5.0 고정 기준 모델: 0821_2

앞으로 이 계획의 모든 Trackman 피처 실험은 0821_2 레시피를 기본 모델로 사용한다. Trackman 피처를 추가한 후보와 기준 모델은 동일한 시간 분할·입력 행·random seed 조건에서 비교한다.

- 입력: 기존 63개 pre-pitch 피처
- 모델 A: Full 학습 정책
- 모델 B: F1-filtered 학습 정책
- 각 모델 내부: CatBoost 70% + 3층 MLP 30%
- 모델 A와 B의 최종 예측: 50:50
- 최종 확률 보정: `clip(prediction - 0.01114, 0, 1)`
- 기준 2024 검증값: Brier `0.247699`, 로컬 환산 `843.77점`

이 기준값은 2019~2023년으로 학습하고 2024년을 검증한 결과다. 이후 모든 Trackman 실험은 다음 두 값을 반드시 함께 보고한다.

1. 0821_2 고정 기준 모델: Brier `0.247699`, 로컬 환산 `843.77점`
2. Trackman 추가 후보: 동일 조건의 Brier와 로컬 환산 점수

기준 모델의 Full/F1 정책, CatBoost·MLP 가중치, outer 50:50 비율, offset을 실험별로 임의 변경하지 않는다. 이 항목들을 바꾸는 실험은 Trackman 피처 실험과 별도의 모델링 실험으로 기록한다.

### 5.1 주 검증 분할

시간 순서를 지키는 3개 rolling fold를 기본으로 한다.

| Fold | 학습 | 검증 |
|---|---|---|
| 1 | 2019~2021 | 2022 |
| 2 | 2019~2022 | 2023 |
| 3 | 2019~2023 | 2024 |

2024 단독 결과는 빠른 screening에만 사용하고, 채택 판단은 3-fold 평균을 기준으로 한다.

### 5.2 기준 모델

Trackman 실험마다 위의 0821_2 기준 모델, 같은 pre-pitch 입력, 같은 Full/F1 처리 정책, 같은 seed와 조기 종료 기준을 사용한다. 기준 피처 수·파라미터·학습 행 필터·offset은 실행 시 `config.json`과 `report.md`에 고정 기록한다. 기존의 약한 55개 baseline은 보조 진단용으로만 사용하고, Trackman 후보의 채택 비교에는 사용하지 않는다.

### 5.3 기록할 지표

- 전체 Brier와 BSS
- fold별 Brier, 평균, 표준편차
- Trackman 매칭 여부별 Brier
- 과거 표본 수 구간별 Brier
- game_type별 Brier
- 결측 fallback 비율과 coverage
- 기준 대비 예측 변화량 및 calibration 변화

## 6. 단계별 실험 계획

### Phase 0. 데이터·시간 안전성 감사

첫 실행은 모델보다 피처 원료를 검증한다.

- 매칭 상태·시즌·투수·구종군별 coverage 집계
- 행별 cutoff date 적용 확인
- `pitch_type_group` 표준화와 비정상 값 처리
- 경기 ID 기준 within-game 집계 가능 여부 확인
- 표본 수, 결측률, 극단값, 연도별 장비 분포 확인
- season × game_type × pitcher_hand × pitch_type_group 그룹의 train-only z-score 기준표 생성

산출물은 원료 통계, cutoff 감사 결과, 사용 가능 컬럼 목록이다. 이 단계에서 시간 누수나 매칭 충돌이 발견되면 피처 구현을 중단한다.

### Phase 1. 구종군별 투구 동작 반복성

가장 먼저 검증할 피처군이다. 모든 구종을 섞지 않고 `pitcher × pitch_type_group` 내부에서 계산한다.

#### 릴리스 반복성

- `rel_height`, `rel_side`: 중앙값, 표준편차, MAD, IQR
- 릴리스 위치 2차원 공분산의 두 고유값, 공분산 행렬의 determinant
- 릴리스 타원 면적 proxy
- `extension`의 표준편차, MAD, IQR

#### 구종 내부 물리값 반복성

- 구속, 회전수, induced vertical break, horizontal break의 표준편차·MAD·IQR
- 평균 대비 변동계수와 robust z-score 절대값
- 경기별 변동성의 중앙값과 경기 간 변동성
- 투구 수·경기 수·최근 관측 이후 경과일 등 신뢰도 피처

#### 시간·이닝 변화

- 경기 내부 변동성과 경기 간 변동성 분리
- 이닝 후반의 구속·릴리스 위치 변화 기울기
- 같은 경기 초반 대비 후반의 robust 평균 차이

표본 수가 적은 구종군은 투수 전체 또는 리그 기준으로 단계적 shrinkage한다. 단순히 표본이 적다는 이유로 0을 넣지 않는다.

### Phase 2. 상황별 예상 구종·물리 특성

이 단계부터는 현재 Train 행의 pre-pitch 상황과 과거 Trackman 이력을 결합해 행마다 다른 피처를 만든다.

#### 조건부 분포

다음 순서로 표본을 축소한다.

`투수 × 카운트 × 타자손`  
→ `투수 × 카운트`  
→ `투수`  
→ `리그`

game_type은 Regular/Futures를 분리한다. 각 단계는 최소 표본 수와 smoothing을 적용한다.

#### 생성 피처

- 현재 카운트의 fastball/breaking/offspeed 예상 확률
- 예상 구속·회전수·IVB·HB·extension·릴리스 위치
- 현재 조건부 구종 비율과 평상시 구종 비율의 차이
- 3볼·2스트라이크 등 압박 상황의 구종 다양성
- 같은 손잡이·반대 손잡이 타자 상대 구종 변화
- 조건부 표본 수·fallback 단계·신뢰도

현재 행의 실제 Trackman 값이나 실제 구종은 절대 사용하지 않는다. 현재 카운트와 타자손 등 Train/test에서 사전적으로 알 수 있는 값만 사용한다.

### Phase 3. 레퍼토리의 기하학적 구조

투수의 절대 평균보다 구종 간 관계를 표현한다.

- fastball 대비 breaking/off-speed 구속 차이
- 구종군 중심점 사이 IVB/HB 거리
- 구종군 중심점 사이 릴리스 위치 거리
- 릴리스 위치는 유사하지만 무브먼트가 다른 정도
- 구종군 중심점의 최소·평균·최대 pairwise 거리
- 구종 다양성 entropy와 effective pitch count
- 구종별 표본 신뢰도와 중심점 안정성
- 시즌별 중심점 이동 거리

구종이 없는 투수나 표본이 부족한 구종은 pairwise 값을 결측으로 두고, 사용 가능한 쌍의 수를 별도 피처로 저장한다.

### Phase 4. 시즌 변화량

절대값 대신 변화량을 계산한다.

- 직전 시즌 평균 − 과거 커리어 평균
- 직전 시즌 반복성 지표 − 과거 평균 반복성
- 최근 2~3시즌 구속·릴리스·무브먼트 추세 기울기
- 구속은 유지되지만 릴리스 일관성이 악화된 정도
- 레퍼토리 중심점 이동량
- 직전 시즌 표본 수와 변화량 신뢰도

연도별 측정 차이를 줄이기 위해 원시값 비교와 함께 `season × game_type × pitcher_hand × pitch_type_group` 내부 z-score 변화량도 비교한다.

### Phase 5. OOF 잔차 보정 모델

Trackman 피처를 기존 CatBoost에 한꺼번에 추가하는 실험은 Phase 1~4에서 독립적인 신호가 확인된 뒤에만 수행한다.

1. 각 rolling fold에서 기준 모델의 OOF 예측 `p_base`를 생성한다.
2. 학습 행의 잔차 `r = y - p_base`를 계산한다.
3. Trackman 피처와 현재 pre-pitch 상황만 입력하는 작은 residual 모델을 학습한다.
4. 검증 시 `p_final = clip(p_base + lambda * r_trackman, 0, 1)`을 계산한다.

초기 residual 모델은 과도한 복잡도를 피하기 위해 얕은 CatBoost 또는 정규화 선형 모델로 시작한다. `lambda`, 최소 표본 수, fallback 규칙은 OOF에서만 선택한다.

#### Gating

- 안전한 pitcher ID 매핑
- 고신뢰 Trackman 연결
- 투수 전체 최소 표본 수
- 사용한 구종군별 최소 표본 수
- Regular/Futures 분리 가능 여부
- 피처 결측이 적은 행

게이트를 통과하지 못한 행은 `p_base`를 그대로 사용한다. 게이트별 coverage와 성능을 함께 보고, 소수 행의 큰 개선만으로 채택하지 않는다.

### Phase 6. Auxiliary/Teacher 방식은 별도 트랙

현재 투구 Trackman을 학습 보조 target으로 쓰는 auxiliary head나 privileged teacher는 기존 실험에서 작은 긍정 신호가 있었지만, 제출 규칙 해석 위험과 불안정성이 있다. 따라서 Phase 1~5와 섞지 않고 별도 실험으로 보관한다.

- 운영진 허용 여부를 먼저 확인한다.
- 허용 전에는 제출 후보로 사용하지 않는다.
- 사용하더라도 최종 추론은 pre-pitch Student만 실행한다.
- 현재 투구 Trackman을 직접 피처로 저장해 제출 ZIP에 포함하지 않는다.

## 7. 실험 운영 원칙

- 한 번에 한 피처군만 추가한다.
- Phase 1의 반복성에서 개선이 없으면 세부 피처를 쪼개 원인을 확인한 뒤 Phase 2로 이동한다.
- 유망한 단일 피처군만 조합한다. 처음부터 수십 개를 모두 CatBoost에 넣지 않는다.
- 각 실험은 `open/experiments/<experiment_id>/`에 설정, 피처 목록, fold metrics, report를 남긴다.
- OneDrive `phase2/EDA/실험_요약/`에는 실험 종료 후 채택·보류·기각 결론만 요약한다.
- 기존 `reference`, `champion`, `submit`은 실험 중 직접 변경하지 않는다.
- 사용자 요청이 있었던 `iterations=500`, `early_stopping_rounds=100` 조건은 실제 후보 검증 단계에서 적용하되, 빠른 screening은 별도 명시한다.

## 8. 채택·중단 기준

### 채택 후보

- 우선 2024 로컬 검증에서 0821_2 기준 Brier `0.247699`보다 낮다.
- 최종 채택 판단에서는 3개 rolling fold에서도 각 fold의 동일 레시피 기준보다 낮다.
- 최소 3개 중 2개 fold에서 개선 방향이다.
- 평균 개선이 최소 `2e-5` 이상이거나, 더 작더라도 coverage·calibration·세그먼트에서 일관된 근거가 있다.
- 결측·저표본 행의 성능을 크게 훼손하지 않는다.
- test 행 간 의존성이 없고, 모든 원료가 투구 시점 이전 정보다.

### 보류 또는 기각

- 2024 한 fold에서만 좋아지고 다른 fold에서 악화된다.
- 매칭 행에서만 좋아지며 실제 제출 대상 전체에서는 효과가 사라진다.
- 단순 missing flag만 남고 물리적 피처의 설명력이 없다.
- 표본 수를 낮추거나 게이트를 풀었을 때만 좋아진다.
- 현재 투구 Trackman 또는 평가 데이터의 다른 행이 필요하다.
- 500회 학습에서 100회 동안 검증 성능이 개선되지 않아 early stopping된 뒤에도 기준보다 나쁘다.

## 9. 단계별 실행 목록

- [x] Phase 0: 시간 cutoff·매핑·표본·정규화 감사
- [x] Phase 1-A: 구종군별 릴리스 반복성
- [x] Phase 1-B: 구종군별 물리값 반복성
- [x] Phase 1-C: within-game / between-game 분리와 후반 이닝 변화 — 2024 Brier 0.247789로 기준 0.247699보다 악화, 보류
- [x] Phase 1-D: Phase 1 단독 rolling 검증 — 2024 미세 개선, 2022·2023 악화로 전체 보류
- [x] Phase 1-E: 반복성 그룹 ablation 및 `std` rolling 재검증 — 2024에서는 `std` 21개가 Brier 0.247693으로 가장 유망했지만, 2022·2023 악화로 최종 보류 (`open/experiments/trackman_repeatability_validation/20260822/ablation_report.md`, `std_rolling_report.md`)
- [x] Phase 2: 조건부 예상 구종·물리 특성 — game_type 매핑 보정 후에도 Brier 0.247748로 기준 0.247699보다 악화, 최종 보류
- [x] Phase 3: 레퍼토리 기하 구조 — 2024 Brier 0.247719로 기준 0.247699보다 악화, 보류
- [x] Phase 4: 시즌 변화량 — 2024 Brier 0.247794로 기준 0.247699보다 악화, 보류
- [x] Phase 5-A: OOF residual model — 2024 미세 개선, 2023 악화로 보류
- [x] Phase 5-B: 고신뢰·고표본 gating — 반복성 availability gate에서만 2024 +0.42점, 안정성 부족
- [x] Phase 5-C: 500 iterations / early stopping 100 최종 확인 — 2024 best iteration 0, 기준 개선 없음
- [x] Phase 6: 규칙 감사 완료 — 공식 가이드가 현재 투구의 실제 구종·Trackman 측정값을 금지하므로 auxiliary/teacher 재실험 및 제출 후보 반영을 보류. 기존 auxiliary 결과는 3-fold 평균 Brier 0.24778488, 2023 fold 악화이며 0821_2와 동일 레시피가 아니므로 참고용으로만 보관 (`open/experiments/privileged_rule_audit/20260822/report.md`)

## 10. 실행 완료 상태

Phase 0부터 Phase 6까지의 계획된 검증을 모두 완료했다. 각 단계는 0821_2 고정 레시피와 비교했으며, Trackman 피처 후보는 2024 단일 검증 또는 rolling fold에서 안정적인 개선 기준을 충족하지 못해 제출 후보로 채택하지 않았다.

- Phase 0: 시간 cutoff·매핑·표본·정규화 감사 완료
- Phase 1: 반복성, 물리값 변동성, within/between-game 분해 및 rolling 검증 완료
- Phase 2: game_type을 분리한 조건부 예상 구종·물리 특성 검증 완료
- Phase 3: 레퍼토리 기하 구조 및 tunneling proxy 검증 완료
- Phase 4: 직전 시즌 대비 변화량·추세 검증 완료
- Phase 5: Ridge/CatBoost residual 보정 및 신뢰도 gating 검증 완료
- Phase 6: auxiliary/teacher 방식 규정 감사 완료. 현재 투구 Trackman 사용은 보류

최종 기준은 0821_2의 Brier `0.247699`, local score `843.77`이며, 이후 Trackman 실험은 새로운 가설이 생길 때 별도 실험으로 추가한다.

## 11. 실험별 필수 산출물

- 실행 스크립트와 입력 경로
- `config.json`
- 사용 피처 목록과 각 피처의 정보 시점
- fold별 Brier/BSS와 segment metrics
- coverage·결측·표본 수 분포
- 기준 대비 차이와 결론이 포함된 `report.md`
- 채택·보류·기각 상태
