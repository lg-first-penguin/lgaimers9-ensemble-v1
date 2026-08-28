# Trackman 활용 전략 v2 실험 계획
작성일: 2026-08-22  
상태: 새 전략 수립 단계  
기준 제출 후보: \`0822(std21)\`

## 1. 목적

기존 Trackman 전략은 투수의 과거 물리 프로필을 기존 63개 피처에 직접 추가하는 방식으로 대부분 실패했다. 새 전략의 목표는 Trackman 값을 일반적인 투수 능력 피처로 사용하는 것이 아니라, 다음 두 가지를 검증하는 것이다.

1. 기존 pre-pitch 모델이 약한 행에서 Trackman 과거 이력이 추가 정보를 제공하는가
2. Trackman이 예측 확률 자체보다 baseline의 불확실성·신뢰도·상태 변화를 설명하는가

Trackman은 현재 투구의 실제 구종·위치·결과·측정값으로 사용하지 않는다. 평가행의 현재 입력과 공식 학습 데이터, 공식 2019~2024 Trackman 과거 로그로만 행별 피처를 만든다.

기존 \`0822(std21)\`은 비교 기준 후보로 보존하며, 새 실험 후보는 별도 디렉터리에 저장한다.

## 2. 기존 전략의 실패 진단

### 2.1 투수 프로필과 투구 단위 target의 불일치

\`control_success\`는 투구 단위 target이지만, 기존 Trackman 피처 대부분은 다음과 같은 투수 고정 프로필이었다.

\`pitcher × pitch_type_group × prior seasons\`

따라서 같은 투수의 여러 행에 거의 같은 값이 반복되었다. 카운트·타자손·경기 중요도·최근 상태에 따라 달라지는 투구 단위 제구 변동을 충분히 표현하지 못했다.

### 2.2 기존 \`asof_*\` 정보와의 중복

기존 모델은 이미 투수의 과거 제구율·투구 수·최근 이력·상황별 pre-pitch 정보를 사용한다. Trackman의 평균 구속·구종 비율·장기 프로필은 투수 ID와 기존 이력 피처가 간접적으로 표현하는 정보를 반복했을 가능성이 크다.

### 2.3 측정값보다 신뢰도와 coverage 문제가 큼

투수·구종군별 Trackman 표본 수, 시즌별 측정량, 매칭 가능 여부가 크게 다르다. 값 자체보다 다음이 중요한 정보일 수 있다.

- 과거 Trackman 관측량
- 마지막 관측 시점
- 구종군별 관측 불균형
- 매칭된 물리 프로필의 신뢰도
- 기존 모델의 과거 투수 표본 수와 Trackman 표본 수의 불일치

### 2.4 전체 모델에 약한 피처를 직접 투입

Trackman 피처를 전체 CatBoost와 MLP에 동시에 넣으면 약한 신호와 결측 구조가 전체 예측을 흔든다. 실제 결과에서도 \`std21\`이 전체 76개보다 일관되게 좋았지만 baseline을 안정적으로 넘지는 못했다.

### 2.5 단일 2024 개선의 불안정성

\`std21\`의 rolling 결과:

| 검증연도 | 0821_2 기준 | std21 | 결과 |
|---|---:|---:|---|
| 2022 | 0.244431 | 0.244761 | 악화 |
| 2023 | 0.248775 | 0.248954 | 악화 |
| 2024 | 0.247699 | 0.247693 | 소폭 개선 |

따라서 Trackman의 장기 std 프로필 자체를 일반적인 예측 피처로 채택하지 않는다.

## 3. 규칙·정보 시점 기준

공식 원문인 \`대회 규칙.md\`, \`평가 데이터 독립 예측 원칙 위반 사례.md\`, \`data_description.md\`를 우선한다. 내부 통합 정리 문서는 보조 참고용으로만 사용한다.

### 허용 원칙

- 현재 평가행에 포함된 입력 변수
- 현재 평가행 입력만 이용한 결정론적 파생변수
- 공식 train.csv로 학습한 통계·모델
- 공식 2019~2024 Trackman history의 과거 요약
- 학습 데이터로 고정한 전처리·확률 보정

### 금지 원칙

- 현재 투구의 실제 구종·위치·판정·결과·Trackman 측정값
- 2025년 Trackman 데이터
- 외부 데이터와 외부 API
- 다른 test 행을 이용한 누적·rolling·lag·빈도·분포·target encoding
- test 전체를 이용한 사후 보정·정렬·순위 기반 보정
- 평가 시점 이후 정보를 과거 행에 섞는 집계

현재 평가행의 \`season=2025\`는 행의 입력 변수일 수 있다. 이때 사용하는 Trackman 원료는 제공된 2019~2024 과거 로그뿐이어야 하며, 2025년 Trackman 측정값을 사용하지 않는다.

## 4. 고정 기준

모든 새 실험은 0821_2와 비교한다.

- 기본 입력: 기존 63개 pre-pitch 피처
- Full 모델 + F1-filtered 모델: 50:50
- 각 모델 내부: CatBoost 70% + 3층 MLP 30%
- 최종 offset: \`-0.01114\`
- 기준 2024 Brier: \`0.247699\`
- 기준 2024 환산 점수: \`843.77\`
- 기본 rolling:
  - 2019~2021 → 2022
  - 2019~2022 → 2023
  - 2019~2023 → 2024

모델 구조·offset·Full/F1 정책을 바꾸는 실험은 Trackman 피처 실험과 별도 모델링 실험으로 기록한다.

## 5. 새 전략의 핵심 구조

Trackman 피처를 전체 행에 일괄 추가하지 않는다. 먼저 baseline의 약한 구간과 Trackman coverage를 확인한 뒤, 효과가 있을 가능성이 있는 구간에서만 사용한다.

권장 구조:

\`\`\`text
p_base = 0821_2 baseline prediction

trackman_signal =
    historical_trackman_profile
    × trackman_confidence
    × baseline_uncertainty
    × low_history_gate

p_final = clip(p_base + lambda × trackman_signal, 0, 1)
\`\`\`

초기에는 복잡한 end-to-end 모델보다 단순한 gate·interaction·residual 보정을 우선한다.

## 6. Phase 7-A. 약한 구간 진단

새 피처를 만들기 전에 baseline과 Trackman 효과를 구간별로 분해한다.

### 필수 구간

- \`asof_pitcher_n\`: 0, 1~10, 11~50, 51~100, 101~300, 300+
- \`asof_batter_n\` 구간
- Trackman 과거 투구 수
- Trackman 과거 경기 수
- 구종군별 최소 표본 수
- 매칭 가능 여부
- game_type
- 시즌
- baseline 예측값 구간
- Full/F1 예측 차이
- CatBoost/MLP 예측 차이

### 산출물

- 각 구간의 baseline Brier
- Trackman 후보 Brier
- 행 수와 coverage
- calibration 변화
- 평균 예측 변화량
- 개선이 특정 소수 그룹에만 집중되는지 여부

이 단계에서 모든 주요 구간이 악화되면 Trackman 피처 개발을 중단하고 baseline을 유지한다.

## 7. Phase 7-B. 저이력 투수 전용 Trackman branch

Trackman이 가장 유용할 가능성이 높은 대상은 기존 투수 이력이 부족한 행이다.

### 후보 gate

- \`asof_pitcher_n < 50\`
- \`asof_pitcher_n < 100\`
- \`asof_pitcher_n < 300\`
- 시즌 첫 구간 또는 투수의 이전 시즌 이력이 부족한 행
- baseline Full/F1 예측 차이가 큰 행

### 피처

- 구종군별 std21
- Trackman 투구 수·경기 수
- 구종군별 표본 수
- 마지막 Trackman 관측 시즌
- 최근 관측과 장기 평균의 차이
- Trackman profile missing flag
- Trackman 표본 신뢰도

### 모델링

처음부터 전체 CatBoost에 추가하지 않고 다음 세 가지를 비교한다.

1. baseline 전체 유지
2. gate 통과 행만 Trackman branch 사용
3. baseline과 Trackman branch의 고정 혼합

gate 기준과 혼합 비율은 rolling OOF에서만 정한다. gate 밖의 행은 정확히 baseline을 사용한다.

## 8. Phase 7-C. 행별 예상 구종 비율로 물리값 가중합

현재 투구의 실제 구종은 사용하지 않는다. 대신 현재 행의 pre-pitch 상황으로부터 과거 구종 선택 확률을 계산하고, Trackman 물리 프로필을 행별로 가중합한다.

### 예상 구종 확률

- 현재 카운트
- 타자손
- 투수손
- game_type
- 투수 과거 구종 사용 패턴

표본 축소 순서:

\`\`\`text
pitcher × count × batter_hand
→ pitcher × count
→ pitcher
→ league
\`\`\`

### 생성 피처

- 예상 구종 가중 평균 release std
- 예상 구종 가중 평균 physical std
- 예상 사용 구종 중 최대 불안정성
- 예상 구종 entropy × physical variability
- 예상 구종 확률과 Trackman 구종 표본의 불일치
- 상황별 expected mechanical risk
- 가중합의 표본 신뢰도

기존 Phase 2처럼 조건부 Trackman 값을 대량으로 추가하는 것이 아니라, 최종적으로 5~15개 정도의 행별 risk·confidence 피처로 압축한다.

## 9. Phase 7-D. 리그 상대적 기계적 이상치

절대적인 구속·릴리스 변동성 대신 다음 기준으로 정규화한다.

- season
- game_type
- pitcher_hand
- pitch_type_group

### 후보 피처

- 리그 percentile
- robust z-score
- 투수의 과거 평균 대비 변화
- 구종군 간 안정성 순위
- 구속 안정성 대비 릴리스 안정성의 불일치
- 릴리스는 안정적이나 무브먼트만 불안정한 정도
- 표본 수로 shrink한 anomaly score

원시값과 정규화값을 동시에 대량으로 넣지 않는다. 각 가설별로 소수 피처만 만든다.

## 10. Phase 7-E. baseline 불확실성 조건부 residual 보정

기존 residual 실험은 전체 행에 Trackman residual을 적용했기 때문에 효과가 희석됐을 수 있다.

### baseline 불확실성

- \`abs(full_prediction - f1_prediction)\`
- \`abs(catboost_prediction - mlp_prediction)\`
- \`abs(p_base - 0.5)\`
- rolling OOF 예측의 변동성
- Trackman 표본 신뢰도

### 보정식

\`\`\`text
p_final = clip(
    p_base
    + lambda
    × residual_trackman
    × baseline_uncertainty
    × trackman_confidence,
    0,
    1
)
\`\`\`

residual 모델은 Ridge 또는 얕은 CatBoost로 시작한다. \`lambda\`, gate, 최소 표본 수는 검증 fold의 미래 정보를 보지 않는 OOF 방식으로 고정한다.

## 11. Phase 7-F. 이전 시즌 말의 상태 변화

시즌 전체 평균보다 이전 시즌의 마지막 상태가 다음 시즌 제구와 연결되는지 검증한다.

### 후보 피처

- 이전 시즌 마지막 10경기 vs 시즌 전체
- 이전 시즌 마지막 30일 vs 시즌 전체
- 이전 시즌 후반 구속 변화
- 이전 시즌 후반 릴리스 변동성
- 구속은 유지되지만 릴리스가 흔들린 정도
- 마지막 관측 경기와 시즌 평균 차이
- 마지막 관측 시점으로부터의 경과 기간
- 다음 시즌 carry-over profile의 신뢰도

검증 시 목표 시즌보다 이전 Trackman만 사용한다. 예를 들어 2024 검증은 2019~2023 Trackman으로만 만든다.

## 12. 모델링 우선순위

가장 가능성이 높은 순서:

1. 저이력 투수 구간별 Trackman 효과 진단
2. 저이력·baseline 불확실성 gate
3. 예상 구종 확률 기반 mechanical risk 가중합
4. 리그 상대 percentile·anomaly
5. baseline uncertainty 조건부 residual
6. 이전 시즌 말 상태 변화
7. 위에서 rolling으로 살아남은 단일 그룹만 조합

처음부터 모든 후보를 84개 이상으로 합치지 않는다.

## 13. 채택 기준

### 채택

- 2022·2023·2024 중 최소 2개 fold 개선
- 전체 Brier 악화 없이 특정 구간 개선
- 개선 구간 coverage가 충분함
- 2024 단일 개선폭이 최소 \`2e-5\`이거나 calibration·segment 개선이 일관됨
- test 행 간 의존성 없음
- 공식 데이터와 pre-pitch 정보만 사용
- 결측·저표본 행에서 baseline fallback이 작동함

### 보류·기각

- 2024에서만 개선
- 소수 매칭 행에서만 개선
- gate를 풀었을 때만 개선
- 결측 행을 제거했을 때만 개선
- 기존 \`asof_*\` 피처와 거의 같은 정보를 반복
- Trackman 표본 신뢰도에 따라 방향이 바뀜
- 현재 투구 정보나 다른 test 행이 필요함

## 14. 실험 산출물

각 Phase는 다음을 남긴다.

- 실행 코드
- config.json
- 사용 피처 목록
- 정보 시점과 cutoff 규칙
- fold별 Brier/BSS
- segment별 coverage와 Brier
- missing·표본 수 분포
- 기준 대비 prediction shift
- 채택·보류·기각 결론
- 원본 report.md

기존 \`0822(std21)\` ZIP은 수정하지 않고, 새 후보는 \`open/experiments/trackman_strategy_v2/\` 아래에 저장한다.

## 15. 진행 상태

- [x] Phase 7-A: baseline·Trackman 표본·투수 이력·game_type·baseline 불확실성 세그먼트 진단
  - `std21` 전체 적용은 rolling 전체 개선이 아니었음
  - `game_type=F` gate screening은 2022/2023/2024 Brier를 각각 0.244803→0.244737, 0.249009→0.248934, 0.247695→0.247674로 모두 개선
  - F gate coverage는 각각 12.30%, 10.46%, 11.84%
  - 상세 결과: `open/experiments/trackman_strategy_v2/phase7a_segment_diagnostic_20260822/`
- [x] Phase 7-B: F 전용 Trackman branch와 OOF gate 검증
  - F 행만 `std21`을 포함한 84개 피처로 별도 학습하고, R 행은 0821_2 baseline을 유지
  - 2022·2023 평균 Brier delta로 선택한 고정 혼합 비율은 `lambda=0.10`이었음
  - 전체 Brier: 2022 `0.244803→0.244494` 개선, 2023 `0.249009→0.249251` 악화
  - 선택 lambda를 적용한 2024는 `0.247695→0.247694`로 사실상 동일하며, lambda=1.0은 `0.247798`로 악화
  - F branch 단독 Brier는 2022 `0.206460`, 2023 `0.296120`, 2024 `0.248196`으로 fold 방향이 불안정
  - F coverage는 2022/2023/2024 각각 12.30%/10.46%/11.84%
  - 결론: 2022·2023에서 일관 개선이 아니고 2024 개선폭도 기준 미달이므로 제출 후보로 채택하지 않음
  - 상세 결과: `open/experiments/trackman_strategy_v2/phase7b_f_branch_validation_20260822_rerun/`
- [x] Phase 7-C: 행별 예상 구종 비율 기반 mechanical risk
  - `asof_pitcher_fastball/breaking/offspeed_rate`를 가중치로 사용해 구종군별 Trackman 변동성 15개와 entropy·관측 신뢰도 3개를 행별로 생성
  - 총 81개 피처(기존 63개와 시즌·타자 보강 포함)에 추가하고 0821_2와 동일한 Full/F1·CatBoost/MLP·offset 레시피로 검증
  - 전체 Brier: 2022 `0.244803→0.244477`, 2023 `0.249009→0.248743` 개선
  - 2024 `0.247695→0.247722`로 `2.6e-5` 악화
  - 결론: 과거 2개 fold에서는 개선됐지만 2024에서 악화되어 제출 후보로 채택하지 않음
  - 상세 결과: `open/experiments/trackman_strategy_v2/phase7c_expected_mix_validation_20260822/`
- [x] Phase 7-D: 리그 상대적 anomaly 피처
  - 각 행의 이전 시즌 Trackman 분포를 `game_type × pitcher_hand` 기준으로 만들고, 구종별 변동성의 절대 robust z-score를 구종 사용률로 가중합
  - 2022 `0.244803→0.244554`, 2023 `0.249009→0.248775` 개선
  - 2024 `0.247695→0.247762`로 `6.6e-5` 악화
  - 결론: 2024에서 악화폭이 커 제출 후보로 채택하지 않음
  - 상세 결과: `open/experiments/trackman_strategy_v2/phase7d_anomaly_validation_20260822_rerun2/`
- [x] Phase 7-E: baseline 불확실성 조건부 residual
  - Ridge residual 모델에 std21·Trackman 표본 신뢰도·`abs(Full-F1)` 불확실성을 입력
  - gate는 Trackman available이면서 source fold의 baseline 불확실성 상위 25%인 행
  - 2023에서 lambda `0.10/0.25/0.50/0.75/1.0`이 모두 악화되어 선택 lambda는 `0.0`
  - 2024에서도 lambda `0.10`부터 악화(`0.247695→0.247700`), 선택 lambda 0은 baseline 유지
  - 결론: Trackman residual 보정은 현재 피처와 baseline 불확실성 조합에서 신호를 확인하지 못해 보류
  - 상세 결과: `open/experiments/trackman_strategy_v2/phase7e_residual_validation_20260822/`
- [x] Phase 7-F: 이전 시즌 말 상태 변화
  - 이전 시즌 마지막 상태−커리어 평균, 변동성 delta, 최근 3시즌 slope, 이전 시즌 표본 신뢰도 등 51개 피처를 검증
  - 정확한 0821_2 baseline 대비 2022 `0.244803→0.244748`로 `5.5e-5` 개선
  - 2023 `0.249009→0.249074`로 `6.5e-5` 악화
  - 2024 `0.247695→0.247794`로 `9.9e-5` 악화
  - 결론: 시즌 말 상태 변화도 방향이 불안정해 제출 후보로 채택하지 않음
  - 상세 결과: `open/experiments/trackman_strategy_v2/phase7f_state_change_validation_20260822/`

## 16. 전체 결론

Phase 7-A~7-F의 모든 제안 구조를 rolling 검증했지만, 2024를 포함한 최소 2개 fold에서 안정적으로 개선되는 Trackman 후보는 확인하지 못했다.

- Trackman 피처를 전체 모델에 직접 추가하면 2024 단일 개선 또는 과거 fold와 2024의 방향 반전이 반복되었다.
- F gate, 예상 구종 가중 risk, 리그 상대 anomaly, residual 보정, 시즌 말 상태 변화 모두 제출 기준을 충족하지 못했다.
- 따라서 Trackman을 추가한 후보로 `0822(std21)`을 교체하지 않는다.
- 현재 제출 기준은 기존 0821_2 레시피를 사용하는 `0822(std21)`이며, 새 Trackman 실험은 보류 상태로 보존한다.
- 이후 Trackman을 다시 사용하려면 새로운 정보 구조나 독립적인 검증 근거가 먼저 필요하다.
