# Trackman 활용 전략 v6: 기계적 상태 전이·역사적 유사 투수 검색 계획

작성일: 2026-08-22  
상태: Phase 0~3 실행 완료, Phase 4 판단 대기  
기준 후보: `0822(std21)`  
기준 모델: `0821_2` 고정 레시피

## 0. 이번 계획의 결론부터

v1~v5에서 실패한 것은 Trackman에 어떤 정보도 없어서가 아니다. 더 정확한 결론은 다음과 같다.

> 현재 투구의 Trackman 측정값에는 제구와 연결되는 정보가 있지만, 평가행에는 그 측정값이 없다. 투수의 과거 평균·분산·구종 비율·신뢰도만으로는 그 정보를 시간 안정적으로 복원하지 못했다.

따라서 v6에서는 다음을 하지 않는다.

- Trackman 원시값·평균·표준편차를 기존 63개 피처에 다시 대량 추가
- 현재 평가행의 실제 구종·위치·Trackman 측정값을 teacher나 보정값으로 사용
- `p_trackman_command`와 같은 단일 행별 물리 평균을 다시 확장
- 2024 결과를 본 뒤 lambda·gate·피처를 재선택

v6의 새 가설은 다음이다.

> Trackman의 가치가 투수의 고정 능력값이 아니라, 투수-시즌의 기계적 상태와 상태 전이가 과거 어떤 투수-시즌과 유사한지를 알려주는 데 있다면, Trackman은 숫자 피처가 아니라 역사적 유사 사례 검색과 모델 라우팅 신호로 사용해야 한다.

즉, 현재 행의 Trackman을 예측하는 것이 아니라 `현재 행의 투수에게 가장 비슷한 과거 기계적 상태가 이후 시즌에 어떤 baseline 잔차를 보였는가`를 time-safe하게 검색한다.

## 1. v1~v5 결과를 합친 증거

### 1.1 버전별로 실제로 확인된 것

| 버전 | 핵심 시도 | 관찰 | 의미 |
|---|---|---|---|
| v1 | 구종군별 반복성·geometry·시즌 변화량·residual 보정 | 2024에서만 미세 개선하거나 2022·2023과 방향 반전 | 정적 투수 profile은 다음 연도 제구를 안정적으로 설명하지 못함 |
| v2 | F gate, 예상 구종 mixture, anomaly, residual, 이전 시즌 상태 | 2022·2023 일부 개선 후 2024 악화. expected mix는 2024 `+2.6e-5`, anomaly는 `+6.6e-5`, 상태 변화는 `+9.9e-5` 악화 | 특정 regime에서 보이는 효과가 미래 연도로 이동하지 않음 |
| v3 | coverage 감사, permutation/null, Trackman-only, residual 방향 | 정상 profile이 층화 null보다 우수하지 않음. 18개 세그먼트 중 residual 방향 안정은 2개뿐 | physical player-specific incremental signal을 입증하지 못함 |
| v4 | regime·표준화·reliability·calibration·row-level teacher | static profile·reliability·calibration은 실패. 반면 현재 투구 Trackman teacher는 matched 행에서 개선 | 정보는 있으나 평가행에서 이용할 수 없는 현재 투구 수준의 privileged signal일 가능성 |
| v5 | command TE, 예상 구종 command mixture, historical teacher, soft-label student | 2023은 개선했지만 frozen 2024에서 모두 악화. 예: command blend `0.247695→0.247789`, teacher `0.247695→0.247720`, student `0.247695→0.247725` | teacher를 pre-pitch student로 옮기는 단순 distillation도 시간 안정적 proxy가 되지 않음 |

### 1.2 가장 중요한 양성·음성 증거

#### 양성 증거

v4의 row-level privileged teacher는 현재 투구 Trackman을 본 matched 행에서 다음과 같이 개선됐다.

- 2022 matched Brier: `0.247469 → 0.246154`
- 2023 matched Brier: `0.248571 → 0.248173`
- 2024 matched Brier: `0.247696 → 0.247138`

따라서 Trackman 자체가 무의미하다는 결론은 성급하다.

#### 음성 증거

그러나 teacher 보정량을 pre-pitch 입력만으로 복원한 결과는 악화됐다.

- 2023: baseline 대비 `+0.000388`
- 2024: baseline 대비 `+0.000615`
- teacher 보정량과 pre-pitch 복원값의 상관: 2023 `0.384`, 2024 `0.246`

이 차이는 다음을 뜻한다.

```text
현재 투구의 실제 물리 상태
        ↓  (teacher에서는 관측 가능)
투구별 제구 결과

과거 투수 평균·분산·구종 비율
        ↓  (평가행에서 사용 가능)
현재 투구 물리 상태를 안정적으로 복원하지 못함
```

v6는 이 정보 병목을 억지로 복원하지 않는다. 대신 과거 상태가 이후 시즌에서 어떻게 전개됐는지를 유사 사례로 이용한다.

## 2. v1~v5가 계속 실패한 이유에 대한 통합 추론

### 2.1 고정 profile은 target의 단위와 맞지 않다

`control_success`는 투구 단위 target이다. 반면 v1~v4의 대부분 피처는 `pitcher × pitch_type_group × prior seasons`의 고정 profile이었다. 같은 투수의 여러 행에 거의 같은 값이 반복된다.

현재 투구의 결과는 다음 요소에 동시에 영향을 받을 수 있다.

- 같은 시즌 안에서의 기계적 상태 변화
- 실제로 선택된 구종과 그 구종의 순간적인 실행 상태
- 카운트·타자손·상황에 따른 구종 선택
- 경기·역할·리그 수준에 따른 regime

과거 평균은 이 구조를 압축하는 과정에서 가장 중요한 변동을 잃는다.

### 2.2 기존 63개 baseline이 이미 투수 profile의 상당 부분을 가진다

`pitcher_id`, `asof_*`, 최근 경기, 구종 사용률, game_type, 카운트·타자 상황이 이미 baseline에 들어 있다. 따라서 Trackman 평균·구종 비율·표본 수를 숫자 피처로 붙이면 다음 세 가지 중 하나가 된다.

1. 기존 입력의 중복
2. pitcher ID의 간접 재표현
3. 매칭 coverage·결측·선택 효과의 대리변수

강한 baseline에서는 약한 새 신호보다 이 세 가지 잡음이 더 크게 작동한다.

### 2.3 target regime가 이동한다

Train target rate는 특히 Futures에서 크게 바뀐다.

| 연도 | Futures | Regular |
|---|---:|---:|
| 2022 | 0.7087 | 0.5037 |
| 2023 | 0.4729 | 0.5031 |
| 2024 | 0.4593 | 0.4897 |

2022→2023 Futures는 약 23.6%p 하락했다. 이 환경에서는 과거 command rate나 teacher 보정량의 절대값을 다음 시즌에 그대로 옮길 수 없다. v2~v5에서 2022·2023 개선이 2024에 반전된 현상은 물리값보다 regime 이동과 결합된 것으로 해석해야 한다.

### 2.4 command TE는 희소성과 선택 편향을 동시에 가진다

v5 Phase 0의 세밀한 key는 다음과 같았다.

- `pitcher × game_type × pitch_type_group × count × batter_hand` 그룹 수: 95,044개
- group 표본 수 중앙값: 4
- 표본 수 30 이상 group: 4,823개
- 표본 수 100 이상 group: 214개

계층 fallback과 smoothing은 분산을 줄이지만, 존재하지 않는 투구 단위 정보를 만들어주지는 않는다. v5 command TE가 직접 모델에 들어갔을 때 2023 R이 `0.248420→0.248444`로 악화된 것은 이 한계를 보여준다.

### 2.5 매칭률과 물리 profile의 유효 coverage는 다르다

v4에서 선수 ID 기반 profile 매핑 가능 행은 `99.29%`였지만, repeatability 사용 가능 행은 `68.54%`였다. 2024 profile coverage도 Futures 좌투수 `51.35%`부터 Regular 우투수 `83.86%`까지 크게 달랐다.

또한 v3의 정상 Trackman profile과 층화 permutation null의 성능이 거의 같았다. 따라서 “매칭됐다”는 것만으로 player-specific 물리 signal이 검증된 것은 아니다.

### 2.6 일부 실험은 2024 개선을 과대평가하기 쉽다

Brier 차이가 `1e-5~1e-4`인 실험이 많고, 일부는 offset·gate·branch 선택이 결합되어 있다. 2024 한 fold의 미세 개선은 다음 중 하나일 수 있다.

- 예측 평균 이동이 해당 연도 target rate에 우연히 맞은 것
- 특정 game_type 표본 구성 효과
- Trackman availability selection 효과
- 작은 calibration 차이
- 실제 physical signal

따라서 v6에서는 “2024 점수가 낮다”만으로 성공을 선언하지 않고, 같은 알고리즘의 층화 null·random analog·source regime 대조군을 반드시 함께 통과시킨다.

## 3. v6의 새 핵심 가설

### 3.1 상태 전이 가설

투수의 Trackman을 단순한 커리어 평균이 아니라 다음과 같이 본다.

```text
투수 p의 시즌 s 기계적 상태
        +
시즌 s-1 → s 변화량
        ↓
다음 시즌 s+1에서 나타나는 제구 residual
```

예를 들어 다음 두 투수가 모두 평균 구속 145km/h라고 해도, 한 명은 구속·릴리스가 안정적이고 다른 한 명은 최근 릴리스 위치가 이동 중일 수 있다. v1의 변화량 피처는 이 값을 직접 입력했지만, v6는 변화량이 비슷했던 과거 투수들의 “다음 시즌 결과”를 검색한다.

### 3.2 유사 사례 검색 가설

현재 target 시즌 Y의 투수 p에 대해, Y 이전의 마지막 Trackman 상태와 전이 벡터를 만든다. 그 후 과거에 다음 조건을 만족한 투수-시즌들을 검색한다.

- 상태 벡터가 유사함
- 같은 투수손
- 가능하면 같은 game_type 결과 regime
- 목표 시즌보다 충분히 이전에 다음 시즌 결과가 관측됨

검색된 과거 사례들의 다음 시즌 baseline residual을 가중 평균해 `r_mechanical_analog`를 만든다.

```text
p_final = clip(
    p_base
    + lambda × gate × r_mechanical_analog,
    0,
    1,
)
```

핵심은 Trackman 원시 profile을 CatBoost 입력으로 넣지 않는다는 점이다. Trackman은 “어떤 유사 사례를 참고할지”만 결정한다.

### 3.3 왜 v1의 변화량과 다른가

v1의 시즌 변화량은 현재 투수의 변화량을 숫자 피처로 직접 모델에 주입했다. v6는 다음을 추가한다.

- 변화량의 비선형 조합을 거리 기반으로 표현
- 같은 상태에서 실제로 다음 시즌에 어떻게 되었는지 target residual로 연결
- regime 평균 변화에 덜 민감하도록 absolute target rate가 아니라 baseline residual 사용
- confidence, neighbor distance, neighbor disagreement로 무리한 외삽 차단
- 같은 투수의 과거값에만 의존하지 않고 다른 투수의 역사적 analog로 일반화

이는 “새로운 물리 피처를 더 만드는 실험”이 아니라 “Trackman으로 historical prior를 검색하는 모델링 방식”이다.

## 4. 규칙·정보 시점·매핑 경계

원문 기준:

- `C:\Users\idong\OneDrive\바탕 화면\공모전\2026 LG Aimers 9기\phase2\대회 설명\대회 규칙.md`
- `C:\Users\idong\OneDrive\바탕 화면\공모전\2026 LG Aimers 9기\phase2\대회 설명\평가 데이터 독립 예측 원칙 위반 사례.md`
- `C:\Users\idong\OneDrive\바탕 화면\공모전\2026 LG Aimers 9기\phase2\대회 설명\data_description.md`

### 허용 범위

- 현재 평가행에 포함된 pre-pitch 입력
- 공식 `train.csv`의 target과 pre-pitch 입력
- 공식 `trackman_history.csv`의 2019~2024 과거 로그
- 목표 시즌보다 이전 시즌으로 제한한 Trackman profile·상태 전이
- 공식 학습 데이터로만 만든 baseline model·residual·검색 index

### 반드시 금지

- 현재 평가행의 실제 구종·위치·판정·결과·Trackman 측정값
- 2025년 Trackman 데이터라는 가정 또는 외부 자료
- test의 다른 행을 이용한 집계·분포·rolling·lag·보정
- 현재 평가행을 먼저 Trackman에 매칭한 뒤 그 값을 이용한 prediction
- teacher의 현재 투구 Trackman을 최종 제출 추론에 포함

### 매핑을 사실로 가정하지 않는 방법

Trackman ID 연결은 기술적으로 join이 되는 것과 공식적으로 검증된 ID mapping이라는 것이 다르다. v6는 다음 세 가지 조건을 분리 기록한다.

1. `exact_full10_sequence`와 모든 상태 컬럼 일치 여부
2. pitcher/team crosswalk의 1:1·충돌 여부
3. profile을 만들 때 사용된 source row의 confidence grade

추정·후보·global fallback mapping은 주 실험에서 제외한다. strict exact source와 층화 permutation null이 구분되지 않으면 Trackman 후보를 폐기한다. 이는 점수가 좋아도 제출 후보로 승격하지 않는 규칙 gate다.

## 5. 고정 baseline과 검증 설계

모든 v6 후보는 다음을 변경하지 않는다.

- 63개 pre-pitch 피처
- Full 모델 + F1-filtered 모델 `50:50`
- 각 모델 내부 CatBoost `70%` + 3층 MLP `30%`
- final offset `-0.01114`
- 기준 2024 Brier `0.247699`
- 기준 local 환산 점수 `843.77`

### 시간 rolling

| fold | 학습 | 검증 | Trackman source |
|---|---|---|---|
| A | 2019~2021 | 2022 | 2019~2021만 |
| B | 2019~2022 | 2023 | 2019~2022만 |
| C | 2019~2023 | 2024 | 2019~2023만 |

각 검증행 Y의 모든 profile·state·analog response는 `season < Y` 자료만 사용한다. 2024 결과를 본 후 2025용 설정을 바꾸지 않는다.

### 개발·확인·최종 holdout

- 2022: `k`, 거리 함수, 최소 neighbor 수, lambda의 개발 선택
- 2023: 2022에서 고정한 설정의 독립 확인. 재선택하지 않음
- 2024: 2022·2023에서 유지된 설정의 최종 holdout. 결과를 보고 수정하지 않음

2022만으로 선택한 설정이 2023에서 악화되면 v6 후보를 중단한다. 2023에서 재튜닝하지 않는다.

## 6. Phase 0 — 재현·lineage·negative control 고정

새 index를 만들기 전에 기존 결과를 재현하고 입력 lineage를 고정한다.

### 확인할 것

- 0821_2 baseline의 동일 행·동일 offset·동일 seed
- v1~v5 report와 실제 실행 config의 일치 여부
- Trackman history profile의 source season cutoff
- source row의 exact match·중복·충돌·pitcher crosswalk 상태
- profile coverage를 `complete / partial / missing / strict / fallback`으로 분해
- profile이 동일한 train 행에 반복되는 횟수

### 산출물

- `lineage_report.md`
- `profile_coverage_by_season_game_hand.csv`
- `mapping_grade_summary.csv`
- `baseline_reproduction.json`
- `negative_control_spec.json`

Phase 0에서 baseline 또는 cutoff가 기존 결과와 다르면 Phase 1을 시작하지 않는다.

## 7. Phase 1 — 투수-시즌 기계적 상태 문서 생성

### 7.1 상태의 기본 단위

정적 커리어 profile 대신 다음 단위로 만든다.

```text
pitcher × season
```

가능한 경우 `pitch_type_group` 내부를 유지하고, sample이 부족한 구종은 상태 벡터에서 shrinkage 또는 missing mask로 처리한다. 모든 구종을 섞은 표준편차 하나로 합치지 않는다.

### 7.2 상태 벡터

초기 버전은 20개 안팎으로 제한한다.

- 구종군별 평균·중앙값: rel_speed, spin_rate, induced_vert_break, horz_break, extension
- 구종군별 릴리스 위치: rel_height, rel_side
- 구종군별 robust spread: MAD 또는 IQR 중 하나
- 구종군별 관측 수와 유효 구종 수
- 구종군 비율·repertoire entropy
- 상태의 complete/partial mask

원시값을 모두 넣는 것이 아니라, 상태 벡터의 목적은 검색 거리 계산이다. CatBoost 피처 후보가 아니다.

### 7.3 전이 벡터

연속된 두 시즌의 상태가 모두 충분할 때만 다음을 계산한다.

```text
delta_s = standardized_state_s - standardized_state_{s-1}
```

표준화 기준은 해당 target 시즌보다 이전의 source history에서만 만든다. 표본이 부족하면 해당 성분을 거리 계산에서 제외하고 유효 차원 수를 기록한다.

### 7.4 시간 규칙

target 시즌 Y에 대해 사용할 수 있는 target pitcher state는 최대 Y-1 시즌이다.

- Y=2022: 2021까지
- Y=2023: 2022까지
- Y=2024: 2023까지
- 최종 평가 시즌 2025: 2024까지

Y 시즌의 Trackman row, Y 시즌의 현재 투구 Trackman, test 행은 상태 생성에 사용하지 않는다.

## 8. Phase 2 — 역사적 다음 시즌 residual index 생성

### 8.1 analog 사례의 정의

과거의 한 사례는 다음 단위다.

```text
(pitcher p, origin season s state, next season s+1 response)
```

origin state는 시즌 s의 Trackman profile과 s-1→s 전이로 구성한다. response는 s+1 시즌의 공식 train 행에서 계산한 baseline residual이다.

### 8.2 response를 absolute success rate로 사용하지 않는 이유

Futures의 연도별 target regime 이동이 크므로 raw success rate를 그대로 옮기지 않는다. 다음을 우선 사용한다.

```text
r_i = control_success_i - p_base_i
r_centered_i = r_i - mean(r | source_season, game_type)
```

`p_base_i`는 해당 source response 시즌보다 이전에 학습된 0821_2 rolling model의 예측이어야 한다. response를 만들 때 그 row의 target 또는 future season 정보로 baseline을 학습하지 않는다.

### 8.3 response의 상황 계층

충분한 표본이 있는 경우에만 다음 순서로 분해한다.

1. `game_type × pitcher_hand × balls_before × strikes_before × batter_hand`
2. `game_type × pitcher_hand × count`
3. `game_type × pitcher_hand`
4. `game_type`
5. 같은 pitcher_hand의 전체 source prior

각 response에는 `n`, effective sample size, source season 범위, fallback level을 함께 저장한다. target에 맞는 과거 사례가 없다고 해서 raw rate를 억지로 만들지 않는다.

### 8.4 현재 투수의 과거 response 처리

주 실험은 baseline과의 중복을 측정하기 위해 같은 pitcher의 historical response를 analog pool에서 제외한다. 별도 진단으로만 same-pitcher 포함 결과를 보고한다.

이렇게 해야 다음을 구분할 수 있다.

- Trackman 기계적 상태가 다른 투수에게도 transfer되는가
- 단순히 같은 pitcher의 과거 성적을 다시 넣은 것인가

## 9. Phase 3 — 기계적 analog 검색기

### 9.1 거리

초기 검색기는 복잡한 deep metric learning을 사용하지 않는다.

```text
d(x, z) = weighted_mean(abs(x_j - z_j))
```

구종군별 상태·전이 block에 동일한 최대 weight를 두고, missing 성분은 제외한다. 유효 차원이 적으면 confidence를 낮춘다.

후보 검색은 다음 순서로 제한한다.

1. 동일 pitcher_hand
2. 가능한 경우 동일 source game_type response
3. 최소 유효 차원 충족
4. 거리순 top-k

### 9.2 검색 결과

각 현재 row에 대해 다음만 생성한다.

- `r_mechanical_analog`: 거리 가중 residual 평균
- `analog_distance`: top-k weighted distance
- `analog_n_eff`: effective neighbor sample size
- `analog_disagreement`: neighbor residual의 robust spread
- `analog_fallback_level`
- `analog_profile_complete`

이 값들은 Trackman 원시값이 아니라 역사적 response 검색 결과다.

### 9.3 confidence gate

다음 조건을 모두 만족할 때만 보정한다.

- strict profile 또는 명시된 허용 grade
- 유효 차원 비율이 사전 기준 이상
- `n_eff` 최소 기준 충족
- `analog_distance`가 개발 fold percentile 이내
- `analog_disagreement`가 개발 fold 기준 이하
- same-pitcher-only가 아닌 외부 analog가 존재

그 밖의 행은 `p_base`를 그대로 사용한다. gate는 2022에서 정하고 2023·2024에 고정한다.

## 10. Phase 4 — residual 보정과 null 검정

### 10.1 본 후보

초기 후보는 단 하나로 시작한다.

```text
p_final = clip(p_base + lambda × gate × r_mechanical_analog, 0, 1)
```

lambda는 2022에서만 `{0.03, 0.05, 0.10}` 중 선택한다. 선택 결과가 없으면 `lambda=0`으로 종료한다. 2023에서 재선택하지 않는다.

### 10.2 반드시 비교할 대조군

동일한 검색기·동일한 gate·동일한 response를 유지하고 state만 바꾼다.

1. 정상 state-transition analog
2. `season × game_type × pitcher_hand` 내부 state shuffle
3. static profile만 사용한 analog
4. transition delta만 사용한 analog
5. Trackman profile 없이 `asof_pitcher_n`·game_type만 사용한 analog
6. random neighbor

정상 analog가 null·random보다 우수하지 않으면 Trackman physical signal로 인정하지 않는다. 이 경우 Trackman은 후보에서 제외한다.

### 10.3 기록할 지표

- 전체 Brier/BSS
- 2022·2023·2024 전체, R, F
- gate 안팎 Brier와 coverage
- analog distance·n_eff 분위수별 Brier
- baseline 대비 paired loss
- 예측 평균 변화량과 calibration slope/intercept
- same-pitcher 제외/포함 차이
- normal/null/random 차이

## 11. Phase 5 — 상태 ID 기반 mixture-of-experts

Phase 4에서 정상 analog가 null보다 우수하고 2022→2023 방향이 유지될 때만 실행한다.

### 가설

CatBoost에 연속적인 물리값을 20개 추가하는 대신, 기계적 상태 전이를 소수의 안정적인 상태 ID로 압축해 모델 라우팅에 사용한다.

### 구성

1. source history에서 상태·전이 벡터를 표준화
2. 사전에 정한 작은 K로 cluster 또는 prototype 생성
3. 각 target row의 prior state를 가장 가까운 prototype에 할당
4. 상태별로 baseline residual expert를 학습하거나 baseline과의 작은 delta expert를 학습
5. test row는 자기 state ID에 해당하는 expert만 호출

expert 입력에는 현재 행의 63개 pre-pitch 피처만 넣는다. Trackman 원시값을 expert 입력에 넣지 않는다. Trackman은 expert 선택만 담당한다.

### 안전장치

- cluster/prototype은 target season 이전 source만으로 fit
- 작은 cluster는 global baseline으로 fallback
- expert별 학습 표본 수와 calibration을 기록
- cluster ID만으로 좋아지는지, shuffled state ID에서도 같은지 비교
- cluster 수·expert lambda를 2022에서만 선택

Phase 4가 실패하면 Phase 5는 실행하지 않는다.

## 12. Phase 6 — 선택적 training-domain weighting

Phase 4·5가 실패했을 때 또 다른 raw feature sweep으로 가지 않는다. 마지막으로, Trackman이 예측값이 아니라 “어떤 historical training rows가 target pitcher와 비슷한가”를 알려주는 domain adaptation만 진단한다.

### 방법

- target row의 mechanical state와 가까운 historical pitcher-season을 찾음
- 그 사례가 속한 공식 train rows에 source weight를 부여
- Trackman 원시값 없이 63개 pre-pitch baseline을 weighted training
- target row는 자기 state와 가까운 expert 또는 weight profile을 사용

이 방식은 추론 시 test 행을 서로 집계하지 않고, 각 행이 공식 train history에서 독립적으로 source weight를 조회한다. 다만 row별 재학습은 10분 제한에 맞지 않으므로 실제 후보는 사전 학습한 소수 expert 방식으로만 구현한다.

### 중단 조건

- expert 수가 많아져 추론 10분 제한을 위협
- source weight가 사실상 pitcher ID를 재현
- random state와 차이가 없음
- 2022·2023 방향이 유지되지 않음

## 13. teacher는 어떻게 취급할 것인가

v4·v5 결과에서 teacher는 연구적으로 중요한 단서를 제공하지만, 제출 해법으로 재사용하지 않는다.

### teacher가 알려준 것

- 현재 투구의 물리 상태는 제구와 연결될 수 있음
- 그 연결은 투수 고정 profile보다 투구별 실행 상태에 가까움
- pre-pitch student가 teacher delta를 안정적으로 복원하지 못함

### v6에서 금지하는 것

- 현재 평가행의 Trackman을 읽는 teacher inference
- teacher prediction을 제출 확률에 직접 섞는 방식
- teacher soft label을 공식 허용 여부 확인 없이 제출 student에 반영

teacher는 필요할 때 “상태 전이 analog가 teacher error regime와 연관되는가”를 분석하는 진단용으로만 사용한다. 제출 후보는 반드시 pre-pitch 입력과 공식 과거 history만으로 실행되어야 한다.

## 14. 채택·보류·기각 기준

### 후보 승격 조건

다음 조건을 모두 만족해야 한다.

1. 2022에서 사전 설정한 lambda·k·gate로 baseline보다 개선
2. 2023에서 재튜닝 없이 같은 방향
3. 2024에서 설정 고정 후 baseline보다 개선
4. normal analog가 stratified null·random보다 우수
5. 전체 Brier 악화 없이 R/F 어느 한 branch도 치명적 악화가 없음
6. gate coverage가 충분하고 개선이 소수 표본에만 집중되지 않음
7. same-pitcher 포함 효과가 아니어도 남는 신호가 있음
8. exact/strict linkage와 cutoff audit 통과
9. test 행 독립성·single-row equivalence 통과

`1e-5` 수준의 단일 연도 개선은 단독 채택 근거로 사용하지 않는다. 최소 2개 rolling fold의 방향 일치와 null 대조군 우위가 필요하다.

### 즉시 중단 조건

- 정상 analog가 null보다 좋지 않음
- 2022 개선이 2023에서 반전
- 2024에서만 좋아짐
- 개선이 random neighbor에도 동일
- same pitcher를 포함할 때만 개선
- low-confidence·partial mapping에 의존
- analog가 baseline보다 평균 확률만 이동시키고 paired loss가 좋아지지 않음
- 현재 투구 Trackman 또는 test 다른 행이 필요함

## 15. 제출 독립성 검사

후보 ZIP을 만들 경우 다음을 자동 검사한다.

### 정적 검사

- `test.groupby`, `rolling`, `shift`, `expanding`, `rank`, 전체 test 평균·분포 사용 여부
- test 행을 이용한 target encoding·calibration map 여부
- 현재 행의 Trackman join 여부
- 2025 Trackman 파일 또는 외부 API 참조 여부

### 동적 검사

- 전체 test와 한 행 test의 해당 row prediction `abs diff = 0`
- test 행 순서 shuffle 후 원래 row_id별 prediction 동일
- test를 chunk 단위로 추론해도 full 추론과 동일
- 모델·index·prototype이 모두 ZIP 내부에 존재
- `output/submission.csv` 행 수·row_id·확률 범위 검증

## 16. 실행 순서

1. Phase 0: v1~v5 baseline·lineage 재현
2. Phase 1: pitcher-season state와 transition index 생성
3. Phase 2: strict historical next-season residual index 생성
4. Phase 3: normal/null/random analog 검색기 비교
5. Phase 4: 2022에서만 lambda·gate 선택 후 2023·2024 고정 검증
6. Phase 5: 신호가 유지될 때만 state-ID mixture-of-experts
7. Phase 6: 필요할 때만 소수 expert 기반 domain weighting 진단
8. Phase 7: 독립성·규칙·속도·ZIP 검증

### 시작하지 않을 실험

- raw Trackman 피처 10개 이상 추가
- 2024에서만 lambda/gate를 고르는 실험
- F branch만 별도로 최적화하는 실험
- teacher current-pitch Trackman 기반 제출
- monthly/team profile을 먼저 추가하는 실험
- same-pitcher 과거 target을 단순 재주입하는 실험

## 17. 예상되는 결과와 해석

### 경우 A — analog가 null보다 우수하고 2022~2024 방향이 유지

Trackman이 직접 확률을 예측하는 것이 아니라, 투수의 상태 전이가 이후 시즌 residual regime를 선택하는 데 유용하다는 결론이다. 이 경우 Phase 5 mixture-of-experts를 제한적으로 시도한다.

### 경우 B — analog가 2022·2023에서만 우수하고 2024에서 반전

v1~v5와 같은 nonstationarity가 재현된 것이다. Trackman physical information은 존재하지만 final pre-pitch prediction으로 transfer되지 않는다고 결론 내리고 `0822(std21)`을 유지한다.

### 경우 C — normal analog가 null/random과 동일

현재 제공된 Trackman history와 매핑으로는 player-specific mechanical signal을 사용할 수 없다는 결론이다. Trackman 피처를 더 늘리지 않는다.

### 경우 D — same-pitcher 포함에서만 개선

Trackman이 새로운 정보라기보다 기존 `asof_*`·pitcher history의 중복이라는 뜻이다. 제출 후보로 승격하지 않는다.

## 18. 산출물 경로

원본 결과는 다음에 저장한다.

```text
C:\projects\lgaimers9\open\experiments\trackman_strategy_v6\
```

예상 파일:

- `phase0_reproduction/`
- `phase1_state_transition/`
- `phase2_next_season_residual_index/`
- `phase3_analog_null_validation/`
- `phase4_frozen_rolling/`
- `phase5_state_expert/`
- `phase7_submission_audit/`

각 실행 디렉터리는 최소한 다음을 포함한다.

- `config.json`
- `feature_lineage.md`
- `metrics.csv`
- `coverage.csv`
- `null_comparison.csv`
- `report.md`

OneDrive `phase2\EDA\실험_요약\`에는 실행 종료 후 채택·보류·기각 결론만 요약한다. `0822(std21)`, `submit/`, `open/champion/`, `reference`는 후보가 공개 점수로 확인되기 전 변경하지 않는다.

## 19. 최종 판단 기준

v6의 목적은 Trackman을 반드시 제출 모델에 넣는 것이 아니다. 목표는 v1~v5의 실패를 설명하는 더 강한 검증을 수행하는 것이다.

최종적으로 다음 중 하나를 명확히 결론 내린다.

1. **Transferable signal:** 상태 전이 analog가 null보다 우수하고 rolling에서 안정적이므로 제한적 후보를 만든다.
2. **Unavailable signal:** 현재 투구 teacher에는 정보가 있지만 pre-pitch history로는 transfer되지 않으므로 제출에는 사용할 수 없다.
3. **No validated signal:** strict mapping과 null 대조군을 통과하는 Trackman signal이 없으므로 `0822(std21)`을 마지막 후보로 유지한다.

현재 v1~v5의 근거만으로는 2번이 가장 가능성이 높다. 그럼에도 v6를 시도할 가치는 있다. 지금까지 검증하지 않은 것은 “물리값이 좋은가”가 아니라 “비슷한 상태의 과거 투수들이 다음 시즌에 비슷한 방향으로 움직였는가”이기 때문이다.

## 20. 진행 상태

- [x] Phase 0: v1~v5 baseline·lineage·cutoff 재현
  - Train `1,475,092`행, Trackman history `1,793,078`행 확인
  - 0821_2 기준 2024 Brier `0.2476991213`, local score `843.7738` 확인
  - 기존 profile lineage의 mapping available `1,464,612`행, strict mapping `1,424,897`행 확인
  - 2019 prior profile coverage `0%`; 2024 game_type·pitcher_hand별 profile coverage `68.19%~83.86%` 확인
  - v5 exact command source `773,446`행, 상태 컬럼 8개 일치율 `100%`, source season < target season `100%` 확인
  - 세밀한 command group 표본 수 중앙값 `4` 확인
  - 원본 실행 코드: `C:\projects\lgaimers9\code\trackman_strategy_v6_phase0_reproduction.py`
  - 원본 결과: `C:\projects\lgaimers9\open\experiments\trackman_strategy_v6\phase0_reproduction\report.md`
  - 결론: Phase 1 상태 전이 index 생성을 진행할 수 있음
- [x] Phase 1: 투수-시즌 기계적 상태·전이 index 생성
  - [x] strict exact crosswalk 기반 상태 생성 완료
  - Trackman history strict crosswalk 통과 행 `1,726,374`
  - 투수-시즌 상태 `2,272`개, 연속 시즌 전이 `1,459`개 생성
  - 상태 차원 `37`개: 구종군별 median 물리값·MAD·구종 비율·전체 릴리스 요약
  - Trackman ID 충돌 `3`건, train 투수 ID 충돌 `1`건은 모두 제외하고 별도 기록
  - current evaluation pitch Trackman과 test 행은 사용하지 않음
  - 원본 실행 코드: `C:\projects\lgaimers9\code\trackman_strategy_v6_phase1_state_transition.py`
  - 원본 결과: `C:\projects\lgaimers9\open\experiments\trackman_strategy_v6\phase1_state_transition\report.md`
  - 결론: 충돌을 제거한 strict state index로 Phase 2 진행
- [x] Phase 2: 역사적 다음 시즌 residual index 생성
  - [x] 2021~2024 응답 시즌의 고정 0821_2 OOF 생성
  - 2024 baseline Brier `0.247699`로 기준값과 일치
  - 전이 상태가 있는 응답행 `623,947`개, response coverage `60.31%~64.31%`
  - historical next-season residual case `40,963`개 생성
  - 원본 실행 코드: `C:\projects\lgaimers9\code\trackman_strategy_v6_phase2_baseline_oof.py`, `C:\projects\lgaimers9\code\trackman_strategy_v6_phase2_next_season_residual.py`
  - 원본 결과: `C:\projects\lgaimers9\open\experiments\trackman_strategy_v6\phase2_baseline_oof\`, `C:\projects\lgaimers9\open\experiments\trackman_strategy_v6\phase2_next_season_residual\`
  - 결론: Phase 3 analog 검색을 진행할 수 있음
- [x] Phase 3: analog 검색·null 비교
  - state/delta 71차원, top-k `25`, 동일 상태·상황 계층을 사용한 normal analog와 층화 state-shuffle null을 비교
  - analog coverage `60.36%~62.43%`
  - 2022에서 lambda `0.10`이 baseline `0.244522` 대비 `0.244515`로 미세 개선
  - 2023은 같은 lambda에서 `0.248949`로 baseline `0.248944`보다 악화
  - 2024는 같은 lambda에서 `0.247700`으로 baseline `0.247699`보다 악화
  - normal analog는 null보다 각 연도에서 근소하게 우수했지만, 절대 Brier 개선이 rolling에서 유지되지 않음
  - 원본 실행 코드: `C:\projects\lgaimers9\code\trackman_strategy_v6_phase3_analog_null.py`
  - 원본 결과: `C:\projects\lgaimers9\open\experiments\trackman_strategy_v6\phase3_analog_null\report.md`
  - 결론: Trackman 상태가 analog 선택에는 약한 정보가 있으나, 2022에서 고정한 보정이 2023·2024로 transfer되지 않는지 Phase 4에서 고정 검증
- [x] Phase 4: 2022 선택 lambda·gate의 2023·2024 고정 rolling 검증
  - 2022에서만 선택한 lambda `0.10`을 2023·2024에 고정 적용
  - 2022: `0.244522 → 0.244515` (`-6.64e-6`)
  - 2023: `0.248944 → 0.248949` (`+4.62e-6`)
  - 2024: `0.247699 → 0.247700` (`+0.76e-6`)
  - 정상 analog의 null 대비 우위는 있으나 rolling fold 채택 조건을 만족하지 못함
  - 원본 실행 코드: `C:\projects\lgaimers9\code\trackman_strategy_v6_phase4_frozen_rolling.py`
  - 원본 결과: `C:\projects\lgaimers9\open\experiments\trackman_strategy_v6\phase4_frozen_rolling\report.md`
  - 결론: v6 analog 보정은 제출 후보로 승격하지 않으며 Phase 5 mixture-of-experts는 실행하지 않음
- [ ] Phase 5: state-ID mixture-of-experts
- [ ] Phase 6~7: domain weighting·제출 독립성 audit
