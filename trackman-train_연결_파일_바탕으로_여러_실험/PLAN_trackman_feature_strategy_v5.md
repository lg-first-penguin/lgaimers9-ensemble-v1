# Trackman 활용 전략 v5

- 작성일: 2026-08-22
- 상태: Phase 0~5 완료; Trackman 제출 후보 보류
- 목적: v1~v4의 실패 원인을 반영해 Trackman을 `물리 프로필`이 아니라 `현재 상황에서 선택 가능한 구종의 command_success 결과`를 추정하는 보조 정보로 재정의하고, 규칙·시간 순서·평가 독립성을 지키면서 0821_2 기준을 넘을 수 있는지 검증한다.

## 1. 이번 버전의 핵심 결론

v1~v4에서 반복적으로 확인된 사실은 Trackman의 평균 구속, 회전수, 무브먼트, 릴리스 위치, 반복성, 신뢰도 메타 피처를 행에 직접 추가하는 방식이 안정적인 개선으로 이어지지 않았다는 것이다.

가장 중요한 새 가설은 다음과 같다.

> Trackman은 “투수가 어떤 물리적 특성을 가진가”보다 “현재 상황에서 선택될 수 있는 각 구종이 `control_success`에 어떤 결과를 내는가”를 모델링할 때 활용될 가능성이 높다.

따라서 이번에는 다음 순서를 고정한다.

1. `pitcher × pitch_type_group × count × batter_hand`별 time-safe command target encoding
2. 예상 구종 확률과 command target을 결합한 `p_trackman_command` 생성
3. 2023 검증에서 기준 모델과 비교
4. 2023에서 개선될 때만 historical pitch mixture teacher 실행
5. R branch부터 검증
6. R에서 안정적인 증거가 생긴 뒤 F branch를 별도 teacher 또는 보수적인 lambda로 검증
7. 마지막에 월별·팀별 recent profile과 역할/컨디션 정보를 추가 검토

이번 문서는 실행 순서와 탈락 기준을 고정하는 계획서다. 계획 작성 단계에서는 코드, 제출 ZIP, 활성 챔피언을 변경하지 않는다.

## 2. v1~v4 실패 분석

### 2.1 Trackman 피처가 행별 신호가 되지 못했다

대부분의 기존 피처는 투수×구종군×과거 시즌 단위로 계산된 고정 프로필이었다. 같은 투수의 많은 행이 동일한 값을 공유하므로, 이미 강한 63개 사전 투구 피처가 표현하는 투수·상황 효과에 추가 정보를 거의 제공하지 못했다.

### 2.2 물리적 불안정성이 곧 제구 실패는 아니다

릴리스 위치나 구속 변동성이 커지는 이유는 구종 변화, 역할 변화, 경기 환경, 표본 구성 변화일 수 있다. 따라서 “변동성이 크다 → control_success가 낮다”라는 단일 방향 가정이 성립하지 않았다.

### 2.3 평균 물리값의 기대값은 결과 예측과 다르다

기존 expected mix/mechanical 접근은 상황별 예상 구종과 예상 물리값을 만들었지만, 그 물리값이 실제 `control_success`에 미치는 조건부 결과를 직접 학습하지 않았다. 물리 특성의 평균을 넣는 것과 그 물리 특성에서 발생하는 제구 성공확률을 넣는 것은 다른 문제다.

### 2.4 단순 gating·신뢰도·shrinkage가 신호를 만들지 못했다

표본 수, 매핑 신뢰도, profile coverage를 이용한 gate와 shrinkage는 결측·저신뢰 행을 보호하는 효과는 있을 수 있지만, Trackman 자체가 제공하는 방향성 있는 command 신호를 만들지는 못했다. 2023에서 안정적으로 개선되지 않았으므로 주력 전략으로 반복하지 않는다.

### 2.5 teacher에는 신호가 있지만 pre-pitch로 옮기기 어려웠다

현재 투구의 실제 Trackman 정보까지 볼 수 있는 privileged teacher는 매칭된 행에서 baseline보다 크게 개선됐다. 그러나 기존 Phase 6-B의 단순 pre-pitch Ridge 재구성은 teacher 개선분을 충분히 복원하지 못했다.

이는 Trackman에 신호가 없다는 뜻보다 다음을 의미한다.

- teacher가 본 정보가 현재 투구의 물리값에 지나치게 직접 의존한다.
- 단순한 teacher delta 회귀만으로는 상황별 구종 선택과 물리 프로필의 혼합 구조를 표현하기 어렵다.
- teacher 결과를 그대로 추론에 쓰는 방식은 최종 평가 입력 경계와 맞지 않는다.
- teacher를 다시 시도한다면 `command target → pitch mixture → teacher 평균` 또는 OOF soft-label distillation처럼 구조를 바꿔야 한다.

### 2.6 연도·game type별 regime이 섞여 있다

F branch는 연도별 target rate와 표본 구조의 변화가 커서 R과 같은 보정값을 적용하면 안 된다. Trackman candidate가 한 연도에서 좋아 보여도 다른 연도·branch에서 반대 방향이 될 수 있으므로 R/F와 연도별 결과를 분리해서 본다.

## 3. 규칙과 정보 경계

### 3.1 공식 규칙을 우선한다

공식 `대회 규칙.md`를 최우선 기준으로 한다. 비공식 분석 문서는 구현 점검에 참고할 수 있지만 공식 규칙의 근거로 취급하지 않는다.

### 3.2 최종 평가 행 독립성

최종 추론에서 한 평가 행의 예측은 다음만 사용해야 한다.

- 해당 행의 입력 변수
- 해당 행의 입력 변수로 만든 파생변수
- 주최 측이 제공한 공식 학습 데이터
- 공식 학습 데이터만으로 학습·생성한 통계, 모델, 파생변수

다음은 금지 목록으로 고정한다.

- `test.csv`의 다른 행을 이용한 집계, rolling, lag, 평균, 분포, 순위, calibration
- 평가 데이터 내 같은 선수·팀·월·경기 단위의 다른 행 집계
- 평가 행의 현재 투구 이후 결과나 현재 투구 Trackman 값 사용
- 2025년 Trackman 데이터 또는 외부 API·외부 데이터 사용

### 3.3 Trackman history 사용 범위

공식 제공 `trackman_history.csv`의 2019~2024 과거 기록만 사용한다. 목표 시즌 Y의 피처는 원칙적으로 시즌 Y보다 이전 시즌의 공식 학습 데이터와 역사 Trackman에서만 만든다.

최종 2025 평가 추론에서는 2019~2024 역사 Trackman을 사용할 수 있지만, 평가 행의 2025 현재 투구 Trackman은 사용하지 않는다.

### 3.4 privileged teacher의 취급

공식 규칙에 “privileged teacher”라는 용어가 직접 금지되어 있다고 단정하지 않는다. 다만 teacher가 현재 투구 Trackman을 이용해 얻은 예측을 최종 평가 행에 직접 적용하면 평가 입력 경계를 벗어날 수 있으므로 제출용 추론에는 사용하지 않는다.

teacher branch는 다음 두 단계로 분리한다.

1. 로컬 진단: 공식 train과 과거 Trackman이 매칭된 학습 행에서 teacher의 상한과 distillation 가능성을 확인한다.
2. 제출 후보: teacher/current-pitch Trackman 없이 실행되는 student만 포함한다.

teacher로 만든 soft label을 student 학습에 사용하는 방식은 공식 규칙상 명시적 허용 여부를 별도로 운영진에게 확인한다. 확인 전에는 로컬 연구 후보로만 보관하고 리더보드 제출 후보로 승격하지 않는다.

## 4. 고정 baseline과 검증 프로토콜

### 4.1 고정 baseline

모든 비교의 기준은 다음 0821_2 recipe로 고정한다.

- 사전 투구 피처 63개
- Full 모델과 F1-filtered 모델을 50:50 블렌드
- 각 모델 내부: CatBoost 70% + 3-layer MLP 30%
- 최종 확률 offset: `-0.01114`
- 0821_2의 2024 local Brier: `0.247699`
- 기준 환산 점수: `843.77`

Trackman 실험은 baseline의 피처, 모델 구조, 블렌드 비율, offset을 임의로 바꾸지 않는다. 바꾸려면 별도 ablation으로 먼저 등록한다.

### 4.2 두 개의 사전 고정 검증

검증은 반드시 다음 두 번만으로 설정 선택을 판단한다.

- 2023 검증: 2019~2022로 학습하고 2023을 테스트
- 2024 검증: 2019~2023으로 학습하고 2024를 테스트

운영 원칙:

- 2023 결과로 피처·lambda·gate·계층·teacher 구조를 선택한다.
- 설정을 고정한 뒤 2024를 한 번의 확인용 holdout으로 평가한다.
- 2024 결과를 보고 피처, lambda, gate, branch 정책을 다시 고르지 않는다.
- 2024에만 좋아진 candidate는 개선으로 채택하지 않는다.
- 보고서에는 fold별 전체·R·F·매핑 coverage·표본 신뢰도별 Brier를 함께 기록한다.

### 4.3 채택 기준

주력 candidate가 되려면 최소한 다음을 만족해야 한다.

- 2023 전체 Brier가 baseline보다 개선
- 2023 R branch가 개선되거나 적어도 악화되지 않음
- 2023의 고신뢰 매칭 구간에서 개선 방향이 일관됨
- low-coverage·low-sample 행에서 baseline보다 나빠지지 않도록 fallback이 작동
- 고정된 설정으로 2024를 평가했을 때 전체 Brier가 baseline보다 개선되거나, 전체가 거의 동일하면서 R·고신뢰 구간에서 명확한 개선
- 단일 행 또는 2024 결과에만 의존한 tuning이 아님

`2.6e-5` 수준처럼 한 holdout에서만 나타나는 극소 개선은 안정적인 채택 근거로 보지 않는다.

## 5. Phase 0 — command label과 매칭 품질 점검

### 목표

Trackman 물리값 자체가 아니라, 과거 각 투구의 `control_success`와 Trackman 구종·상황을 시간 순서에 맞게 연결할 수 있는지 확인한다.

### 작업

1. 기존 공식 train target과 exact sequence matching된 Trackman 행만 1차 source로 사용한다.
2. exact match 여부, pitcher ID, season, game type, game date, pitch number, batter hand, count 일치율을 집계한다.
3. unmatched, ambiguous, duplicate, low-confidence mapping을 별도 표시한다.
4. target이 있는 공식 train 행만 command target 계산에 사용한다.
5. 실제 target row의 `pitch_type_group`은 역사 command rate를 계산하는 source key로만 사용한다. 평가 추론에서는 현재 투구의 실제 구종을 입력으로 가정하지 않는다.

### 산출물

- `open/experiments/trackman_command_te/<run_id>/lineage_report.md`
- 매칭 coverage 및 confidence 표
- season·game_type·pitcher_hand·pitch_type_group별 표본 수
- command label의 전체 평균과 baseline target rate 비교

### 탈락 기준

exact match가 아닌 매핑을 사실로 확장하거나, 매칭 불확실 행을 모두 같은 품질로 사용하는 경우 Phase 1로 진행하지 않는다.

## 6. Phase 1 — time-safe command target encoding

### 목표

현재 상황에서 각 구종군이 `control_success`와 어떤 관계를 보였는지를 과거 공식 데이터만으로 추정한다.

### source 시점 규칙

목표 시즌을 Y라고 할 때 source는 반드시 `season < Y`다.

- 2023 검증: 2019~2022 source만 사용
- 2024 검증: 2019~2023 source만 사용
- 최종 2025 평가: 2019~2024 source만 사용

현재 시즌 Y의 다른 학습 행 target을 사용해 Y 행의 Trackman TE를 만들지 않는다. 이 규칙을 적용해야 2023·2024 검증과 최종 추론의 시간 구조가 일치한다.

### 계층

가장 세밀한 계층부터 다음 순서로 축소한다.

1. `pitcher × game_type × pitch_type_group × balls_before × strikes_before × batter_hand`
2. `pitcher × game_type × pitch_type_group × count`
3. `pitcher × game_type × pitch_type_group`
4. `pitcher × pitch_type_group`
5. `game_type × pitch_type_group`
6. 전체 league prior

필요한 경우 `pitcher × game_type × count × batter_hand`의 구종 선택 계층과 동일한 키를 사용하되, 각 계층의 표본 부족 시 상위 계층으로 즉시 fallback한다.

### 계산 방법

각 계층에서 다음을 저장한다.

- 성공 수와 전체 수
- empirical-Bayes 또는 Beta prior를 적용한 posterior mean
- raw count, effective count
- 사용된 fallback level
- source season 범위와 가장 최근 source season

권장 초기 smoothing은 사전에 정한 한 가지 설정으로 시작하고, 2023 결과가 나온 뒤에만 최소한의 민감도 실험을 한다. `alpha`, `beta`, minimum count를 2024에 맞춰 고르지 않는다.

### 생성 피처

처음부터 많은 물리 피처를 추가하지 않고 다음 저차원 피처만 만든다.

- 구종군별 posterior command rate
- command rate의 raw/effective sample count
- fallback level
- pitcher 상황 command와 league 상황 command의 차이
- 구종군별 command rate의 best/worst 및 spread
- source 최신성

### 검증

먼저 TE 자체가 과거 source에서 target을 설명하는지 확인한다. 그 다음 TE를 baseline에 추가하거나, 아래 Phase 2의 `p_trackman_command`만 추가해 2023에서 평가한다.

## 7. Phase 2 — 예상 구종 확률과 `p_trackman_command`

### 목표

평가 행에는 현재 실제 구종이 없으므로, 현재 상황에서 각 구종군이 선택될 확률과 구종군별 command rate를 결합한다.

### 예상 구종 확률

다음 계층으로 time-safe `P(pitch_type_group | pitcher, game_type, count, batter_hand)`를 추정한다.

1. pitcher × game_type × count × batter_hand
2. pitcher × game_type × count
3. pitcher × game_type
4. pitcher
5. game_type × count × batter_hand
6. league prior

각 계층에 동일한 시간 경계, smoothing, sample metadata를 적용한다. 공식 train의 사전 투구 구종 비율과 Trackman history의 구종 비율을 합칠 때도 source season < Y만 사용하고, 어느 source가 사용됐는지 기록한다.

### 핵심 결합값

각 현재 행에 대해:

`p_trackman_command = Σ_g P(g | current_context) × command_rate(g | current_context, pitcher_history)`

여기서 `g`는 fastball/breaking/offspeed 등 pitch type group이다. 현재 평가 행의 실제 `pitch_type_group` 또는 현재 투구 Trackman 값을 사용하지 않는다.

추가 후보는 다음으로 제한한다.

- `p_trackman_command - p_asof_pitcher_command`
- 예상 command rate의 분산 또는 best/worst spread
- 예상 구종 entropy
- 예상 구종 확률의 effective sample count
- command TE의 fallback/reliability flag

### 첫 적용 방식

처음에는 baseline 예측에 직접 많은 피처를 넣지 않고, 다음 두 방식을 별도로 비교한다.

1. low-lambda probability blend: `p = clip(p_base + λ × centered(p_trackman_command), 0, 1)`
2. exact baseline recipe에 `p_trackman_command`와 최소 메타 피처만 추가

lambda와 중심화 방식은 2023에서만 고르고 2024에서는 고정한다. low-sample·low-confidence 행은 lambda=0 또는 baseline fallback으로 둔다.

## 8. Phase 3 — R branch 우선 검증

### 목표

F의 regime 변화가 R 신호를 가리지 않도록 먼저 R에서만 command target 가설을 검증한다.

### 순서

1. 2023 R 전체 baseline 측정
2. Phase 1 TE만 적용
3. Phase 2 `p_trackman_command` 적용
4. R 고신뢰 mapping 및 표본 충분 행만 적용
5. low-confidence 행 baseline 유지
6. 설정을 고정하고 2024 R 평가

### R 채택 기준

R에서 2023 개선이 없으면 historical pitch mixture teacher와 F branch를 즉시 진행하지 않는다. R에서 개선이 확인된 경우에만 다음 Phase로 이동한다.

## 9. Phase 4 — historical pitch mixture teacher

### 실행 조건

Phase 1~3의 command target/pitch mixture가 2023 R에서 baseline 대비 개선된 경우에만 실행한다.

### teacher의 역할

teacher는 과거 매칭 학습 행에서 현재 투구의 privileged Trackman 정보를 사용해 `P(control_success | pre_pitch_context, actual_pitch_physics)`를 학습한다. 기존의 물리 평균 피처와 달리, 실제 물리값이 command 결과에 미치는 조건부 관계를 직접 학습한다.

### 평가 행에 적용하는 방법

평가 행의 현재 투구 Trackman을 넣지 않는다. 대신 과거 source에서 현재 매핑된 투수·상황·구종군의 historical pitch prototype 또는 quantile 대표값을 만들고, 예상 구종 확률로 teacher 예측을 평균한다.

`p_teacher_mix = Σ_g P(g | context) × E_x[P_teacher(control_success | pre_pitch_context, g, x)]`

여기서 `x`는 source history에서 얻은 과거 prototype이며, 평가 행의 현재 투구 관측값이 아니다.

### 초기 teacher 구성

- teacher 입력: 고정 63개 pre-pitch 피처 + historical pitch group + historical Trackman 물리값
- source: 목표 시즌보다 이전 시즌의 exact match 행
- prototype: 구종군·투수·상황별 median/quantile 대표값, 저표본은 상위 계층 fallback
- 출력: group별 teacher probability와 mixture mean
- 후보 보정: 사전 등록한 보수적 lambda `.02`, `.05`, `.10`

teacher 자체가 좋아 보인다는 이유만으로 제출하지 않는다. 최종 제출에는 teacher 또는 현재 투구 Trackman이 들어가지 않아야 한다.

## 10. Phase 5 — OOF soft-label teacher distillation

### 목적

Phase 4의 mixture teacher가 pre-pitch 행별 예측으로 옮겨질 수 있는지, 기존 단순 Ridge delta보다 정확한 방식으로 확인한다.

### 시간 안전 OOF 구조

- 각 source year의 teacher는 그보다 이전 시즌 source로만 학습한다.
- teacher의 출력은 해당 source 학습 행의 OOF 또는 strictly past prediction으로 생성한다.
- 목표 연도 행의 target 또는 target-year Trackman을 student feature 생성에 사용하지 않는다.

### student

student는 정확히 0821_2의 구조를 사용한다.

- 63개 pre-pitch feature baseline
- Full/F1-filtered 50:50
- 각 모델 CatBoost 70% + 3-layer MLP 30%
- final offset `-0.01114`
- teacher soft label은 매칭된 source 행에만 소량의 고정 weight로 사용
- unmatched/low-confidence 행은 hard target baseline 학습 또는 baseline 추론 유지

R과 F는 처음부터 별도 student 또는 별도 distillation weight를 검토한다. F를 R의 teacher lambda로 보정하지 않는다.

### 규칙 gate

teacher soft label distillation은 공식 규칙상 명시적 허용 여부를 운영진에게 확인한다. 확인 전에는 실험 결과를 제출 후보로 승격하지 않는다. 제출 후보 ZIP에는 teacher 모델, current-pitch Trackman, evaluation-row aggregation 로직을 넣지 않는다.

## 11. Phase 6 — F branch 별도 처리

### 실행 조건

R branch에서 안정적인 개선이 먼저 확인된 경우에만 F를 검토한다.

### 원칙

- F 전용 expected pitch mixture 또는 F 전용 teacher를 사용
- F는 별도 calibration 또는 더 작은 lambda로 시작
- F 고신뢰·충분 표본 행만 적용
- 나머지 F 행은 baseline 유지
- 2023 F에서 악화되면 F Trackman candidate는 즉시 폐기

F branch 개선을 위해 2024 결과를 보고 특수 gate를 새로 추가하지 않는다.

## 12. Phase 7 — 월별·팀별 recent profile

### 실행 조건

command target 또는 teacher mixture가 2023에서 먼저 유효하다는 증거가 있을 때 마지막으로 실행한다. 기존 연간 pooled profile이 실패했으므로 월별·팀별 정보를 독립된 가설로 취급한다.

### 후보 피처

- prior season × month × team × pitch group command rate
- pitcher 최근 월별 command rate과 prior season 평균의 차이
- 현재 팀과 source 팀의 일치 여부
- 최근 source game date, recentness decay, effective sample count
- 팀 이동 전후 profile 차이
- 월별 expected pitch mix와 command rate의 변화

### 시간 경계

목표 시즌 Y의 월별 profile은 Y 이전 season의 날짜까지만 사용한다. 최종 2025 평가에서는 2024 historical Trackman까지 사용할 수 있지만, `test.csv`의 다른 평가 행으로 현재 월 profile을 만들지 않는다.

## 13. Phase 8 — 선택적 역할·컨디션 상태 피처

월별·팀별 profile 이후에도 여유가 있을 때만 검토한다.

- 경기별 평균 투구 수
- 과거 outing length와 현재 이닝의 차이
- 선발/불펜 비율
- 통상 등판 이닝과 현재 이닝의 차이
- 최근 등판 간격과 workload

기존 late-inning physical delta는 실패했으므로, 물리값의 후반 변화량을 다시 대규모로 추가하지 않고 역할·사용량 상태를 저차원으로만 시험한다.

## 14. 실험 매트릭스와 중단 규칙

각 Phase는 다음 순서로만 실행한다.

| 순서 | 실험 | 2023 개선 없을 때 |
|---|---|---|
| 1 | command TE 자체 | 중단 |
| 2 | `p_trackman_command` | 중단 |
| 3 | R-only baseline blend | 중단 |
| 4 | historical pitch mixture teacher | 중단 |
| 5 | OOF soft-label student | 중단 |
| 6 | F 별도 branch | 중단 |
| 7 | 월별·팀별 recent profile | 중단 |
| 8 | 역할·컨디션 상태 | 최종 후보 없음 |

다음 현상이 발생하면 해당 가설을 중단한다.

- 2023 전체와 R 모두 baseline보다 악화
- normal mapping보다 negative control이 더 좋음
- 고신뢰 subset만 좋아지고 전체 fallback 포함 결과가 악화
- 효과가 2024 한 번에만 존재
- source season 경계가 깨짐
- 매핑 coverage가 낮은데도 외삽으로 결과를 유지함
- teacher가 현재 평가 행 Trackman에 의존함

## 15. 최종 검증 및 제출 gate

제출 후보로 만들기 전에 다음을 모두 확인한다.

1. 2023 설정이 고정되어 있고 2024 결과를 보고 재선택하지 않았는가
2. 2025 추론은 2019~2024 공식 history만 사용하는가
3. 평가 행의 현재 Trackman을 읽지 않는가
4. `test.csv`의 다른 행을 집계하지 않는가
5. R/F branch별 source와 fallback이 독립적으로 동작하는가
6. teacher는 최종 추론에 포함되지 않거나, 규칙 확인이 완료된 distillation student만 남아 있는가
7. 제출 ZIP 안에서 상대경로·출력 형식·확률 범위·행 수가 정상인가
8. 기존 제출 오류 기록과 동일한 실수가 없는가
9. candidate는 Public 점수 확인 전 `submit/` 활성 챔피언으로 승격하지 않는가

규칙·독립성 검사를 통과하지 못하면 점수가 좋아도 제출하지 않는다.

## 16. 기록할 결과

각 실험은 다음 파일을 저장한다.

- 원본 report: `C:\projects\lgaimers9\open\experiments\trackman_command_te\<run_id>\report.md`
- fold metrics: 같은 디렉터리의 `metrics.csv`
- lineage/coverage: `lineage_report.md`
- model/config hash: `config.json`
- 실행 로그: `open/temp/` 또는 실험 디렉터리

OneDrive의 `phase2\EDA\실험_요약\`에는 채택·보류·기각 결론만 기록하며, 각 요약에 원본 report의 절대 경로, 가설, 설정, Brier/BSS, coverage, 결론을 남긴다.

## 17. 현재 체크리스트

- [x] Phase 0 command label 및 exact-match lineage audit
- [x] Phase 1 time-safe command target encoding
- [x] Phase 2 expected pitch probability + `p_trackman_command`
- [x] Phase 3 R-only 2023 selection / frozen 2024 check
- [x] Phase 4 historical pitch mixture teacher (진단 완료, 제출 후보 보류)
- [x] Phase 5 OOF soft-label student distillation (2023 개선, 2024 악화; 제출 보류)
- [ ] Phase 6 F separate teacher/lambda (R 안정 gate 미충족으로 중단)
- [ ] Phase 7 monthly/team recent profile (R 안정 gate 미충족으로 중단)
- [ ] Phase 8 role/conditioning state (R 안정 gate 미충족으로 중단)
- [ ] final rule and test-row independence audit
- [ ] candidate ZIP generation only after all gates

## 17.1 Phase 0 실행 결과 — 2026-08-22

`exact_full10_sequence` 매칭만 사용해 command label source의 lineage를 재검증했다.

- exact source: 773,446행
- `trackman_history.csv` 직접 1:1 join: 773,446/773,446
- 시즌·월·요일·이닝·공수·볼·스트라이크·아웃 상태 일치율: 모두 100%
- 목표 시즌보다 이전 시즌 source 여부: 100%
- 유효 `pitch_type_group` (`fastball`, `breaking`, `offspeed`): 765,122행, 98.92%
- `other` 구종군: 8,324행, Phase 1 source에서 제외
- 세밀한 `pitcher × game_type × pitch_type_group × count × batter_hand` group: 95,044개
- 세밀한 group 표본 수 중앙값: 4행
- 표본 수 30 이상 group: 4,823개
- 표본 수 100 이상 group: 214개

결론:

1. exact sequence와 Trackman row 연결은 수열·시점 기준에서 일관적이다.
2. 그러나 세밀한 command key는 희소하므로 raw rate를 직접 사용할 수 없다.
3. Phase 1은 Beta/empirical-Bayes smoothing, 계층 fallback, effective sample count를 필수로 포함한다.
4. 이 결과는 진단용 매칭을 검증한 것이며 공식 ID 매핑 승인이나 제출 허가를 의미하지 않는다.

원본 결과:

- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_phase0\report.md`
- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_phase0\exact_command_source.csv`
- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_phase0\source_group_summary.csv`

## 17.2 Phase 1~3 실행 결과 — 2026-08-22

### command TE 피처 직접 추가

R branch만 Trackman 피처를 활성화하고, 0821_2의 Full/F1 50:50·CatBoost 70% + 3-layer MLP 30%·offset `-0.01114`를 그대로 유지했다.

| 검증 | baseline | command TE 모델 | delta |
|---|---:|---:|---:|
| 2023 전체 | 0.249009 | 0.249014 | +0.000005 |
| 2023 R | 0.248420 | 0.248444 | +0.000024 |

결론: command TE 피처를 CatBoost/MLP 입력에 직접 추가하는 방식은 2023 R에서 탈락했다.

### `p_trackman_command` 직접 보정

2023 R에서 lambda grid를 선택하고, `lambda=0.2`를 고정한 뒤 2024를 확인했다. 보정식은 다음과 같다.

`p = clip(p_base + lambda × (p_trackman_command - p_base), 0, 1)`

| 검증 | baseline | 고정 lambda=0.2 | delta |
|---|---:|---:|---:|
| 2023 전체 | 0.249009 | 0.248930 | -0.000079 |
| 2023 R | 0.248420 | 0.248332 | -0.000088 |
| 2024 전체 | 0.247695 | 0.247789 | +0.000094 |
| 2024 R | 0.247744 | 0.247851 | +0.000106 |

F는 두 연도 모두 baseline을 유지했다. 2023에서만 개선되고 2024에서 악화됐으므로 제출 후보로 채택하지 않는다. 이는 Trackman command signal이 전혀 없다는 뜻이 아니라, 현재의 단순 command mixture가 연도 변화에 안정적이지 않다는 뜻이다.

### Phase 4 진행 조건

2023 R에서 command mixture의 방향성은 확인됐으므로 v5 계획의 조건부 다음 단계인 historical pitch mixture teacher를 R-only 진단으로 진행한다. 단, teacher가 2023에서 좋아 보여도 2024 frozen holdout에서 안정적이지 않으면 제출 후보가 되지 않는다.

원본 결과:

- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_phase1_validation_2023_R\report.md`
- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_command_blend\report.md`
- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_command_blend\selection_grid_2023.csv`
- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_command_blend\fixed_lambda_metrics.csv`

## 17.3 Phase 4 historical pitch mixture teacher — 2023 결과

2023 R에서만 historical pitch mixture teacher를 실행했다. teacher 학습 source는 2019~2022의 exact_full10_sequence 매칭 R 행이고, 각 2023 R 행에는 현재 투구 Trackman 대신 과거 투수×구종군 median prototype을 넣었다. 최종 teacher 예측은 time-safe 예상 구종확률로 세 구종군 예측을 가중 평균했다.

| 검증 | baseline R | teacher mixture R | delta |
|---|---:|---:|---:|
| teacher 직접 예측 | 0.248420 | 0.248234 | -0.000186 |
| lambda=0.30 보정 | 0.248420 | 0.248263 | -0.000157 |

전체 2023 Brier는 lambda 0.30에서 0.249009 → 0.248868로 개선됐다. lambda 0.30은 2023 결과만 보고 고정하며, 2024에서는 변경하지 않는다.

이 결과는 기존 단순 물리 평균과 다르게, 과거 물리 prototype을 teacher의 조건부 control_success 함수에 통과시킨 뒤 구종 선택확률로 적분했다는 점에서 별도 가설이다. 다만 teacher 학습 과정에는 privileged historical Trackman이 사용되므로, 규칙 확인 전 제출 후보로 승격하지 않는다.

원본 결과:

- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_historical_teacher_2023_R\report.md`
- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_historical_teacher_2023_R\metrics.csv`

2024 frozen 확인 결과:

| 검증 | baseline | teacher mixture lambda=0.30 | delta |
|---|---:|---:|---:|
| 2024 전체 | 0.247695 | 0.247720 | +0.000024 |
| 2024 R | 0.247744 | 0.247772 | +0.000028 |
| 2024 R teacher 직접 | 0.247744 | 0.248096 | +0.000352 |

결론: historical pitch mixture teacher는 2023에서 방향성 있는 개선을 보였지만 2024 frozen holdout에서 재현되지 않았다. teacher와 prototype을 제출 후보로 채택하지 않는다. 다음 Phase 5는 이 teacher의 privileged 신호를 OOF soft label로 pre-pitch student에 옮길 수 있는지 확인하는 연구용 진단으로만 진행한다.

원본 결과:

- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_historical_teacher_2024_R\report.md`
- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_historical_teacher_2024_R\metrics.csv`

## 17.4 Phase 5 OOF soft-label student — frozen 결과

historical pitch mixture teacher의 soft label을 목표연도 이전 exact-match R 행에만 만들고, 63개 pre-pitch student 학습에 alpha 0.1로 반영했다. Full/F1 50:50과 각 모델의 70:30 구조를 유지했으며, fractional target 때문에 CatBoost는 CrossEntropy, MLP는 동일 3층 구조의 MLPRegressor를 사용했다.

| 검증 | baseline 전체 | soft-label student 전체 | delta |
|---|---:|---:|---:|
| 2023 | 0.249009 | 0.248960 | -0.000049 |
| 2024 | 0.247695 | 0.247725 | +0.000030 |

| 검증 | baseline R | soft-label student R | delta |
|---|---:|---:|---:|
| 2023 | 0.248420 | 0.248366 | -0.000054 |
| 2024 | 0.247744 | 0.247777 | +0.000033 |

결론:

1. teacher의 privileged 신호는 OOF soft label에서도 2023에는 일부 전달됐다.
2. 2024 frozen holdout에서 방향이 재현되지 않아 제출 후보로 채택하지 않는다.
3. R branch에서 안정적인 개선이 확인되지 않았으므로 F 전용 teacher/lambda와 월별·팀별 recent profile은 이번 v5 실행에서 중단한다.
4. teacher 모델, soft label, historical prototype을 이용한 제출 ZIP은 만들지 않는다.

원본 결과:

- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_teacher_distillation_2023_R_retry\report.md`
- `C:\projects\lgaimers9\open\experiments\trackman_command_te\20260822_teacher_distillation_2024_R\report.md`

## 18. 성공 기준

이번 v5의 성공은 Trackman 피처를 많이 추가하는 것이 아니다. 다음 중 하나를 시간 안전하게 입증하는 것이다.

1. `p_trackman_command`가 2023에서 baseline보다 개선되고 2024에서도 방향이 유지된다.
2. command mixture 기반 teacher의 신호가 OOF student로 옮겨져 최종 pre-pitch 예측을 개선한다.
3. 개선이 없더라도 Trackman이 효과를 보이는 조건과 실패하는 조건을 표본·branch·시간 구조로 설명해 이후 실험의 범위를 줄인다.

반대로 2023 개선 없이 2024만 좋아지는 candidate, 전체 평균은 같지만 규칙 경계를 침범하는 candidate, current-pitch Trackman 없이는 실행할 수 없는 teacher는 성공으로 기록하지 않는다.
