# 최종 Train–Trackman 팀 매핑 검증 보고서

기준일: 2026-08-20  
대상: 최종 `trackman_id`가 연결된 Train 1,275,274행  
목적: Train의 `season + game_type + team_id`와 Trackman의 `season + team_name`이 투수·타자 역할 모두에서 일관되게 대응하는지 검증

## 1. 결론

최종 Trackman ID 연결 행의 팀 정보는 전수 검증을 통과했다.

- Train–Trackman 시즌 불일치: **0행**
- 팀 ID·팀명 결측: **0행**
- exact 경기에서 만든 팀 대응표의 정방향 충돌: **0개**
- exact 경기에서 만든 팀 대응표의 역방향 충돌: **0개**
- exact 기준표로 검증 가능한 팀 역할 비교의 불일치: **0건**
- 최종 전체 매칭에서 정방향 팀 매핑 충돌: **0개**
- 최종 전체 매칭에서 역방향 팀 매핑 충돌: **0개**

따라서 최종 ID가 연결된 모든 행은 팀 정보 관점에서 모순이 없다.

다만 8,956개 팀 역할 비교는 해당 `시즌+게임타입+팀 ID` 조합이 exact 경기에서 한 번도 나오지 않아 exact 기준표로 독립 검증할 수 없었다. 이 조합들도 최종 전체 매칭 안에서는 각각 한 팀명과만 일대일로 대응했다.

## 2. 입력과 검증 범위

### 입력

- Train 투구 기록과 Trackman 투구 기록을 기준으로 검증했다.

### 사용 컬럼

Train:

```text
row_id
season
game_type
pitcher_team_id
batter_team_id
trackman_id
trackman_match_status
trackman_match_basis
```

Trackman:

```text
trackman_id
season
pitcher_team
batter_team
trackman_game_id
pitch_no
```

최종 `trackman_id`가 결측이 아닌 Train 행만 Trackman 원본과 일대일로 결합했다.

## 3. 검증 방법

같은 최종 매칭 결과에서 팀 대응표를 만들고 그 결과를 다시 확인하면 순환 검증이 될 수 있다. 이를 피하기 위해 다음 두 단계로 나눴다.

### 3.1 exact 경기 기준표

가장 강한 연결인 `trackman_match_basis=exact_full10_sequence` 773,446행에서 투수와 타자를 각각 하나의 팀 역할 관측으로 펼쳤다.

정방향 기준표:

```text
(Train season, Train game_type, Train team_id)
→ Trackman team_name
```

역방향 기준표:

```text
(Trackman season, Train game_type, Trackman team_name)
→ Train team_id
```

정방향에서 한 Train 키가 여러 팀명으로 연결되거나, 역방향에서 한 Trackman 팀명이 여러 Train 팀 ID로 연결되면 충돌로 판정했다.

### 3.2 나머지 자동·구간 매칭 검증

exact 기준표가 유일한 키에 대해 모든 최종 ID 연결 행의 투수팀과 타자팀을 비교했다.

exact 기준표에 없는 키는 불일치로 세지 않고 `exact 기준 미포함`으로 분리했다. 그 뒤 전체 최종 매칭 안에서 정방향·역방향 일대일성이 유지되는지 별도로 확인했다.

## 4. 전체 수치

| 항목 | 결과 |
|---|---:|
| 최종 Trackman ID 연결 Train 행 | 1,275,274 |
| 투수·타자 팀 역할 비교 | 2,550,548 |
| exact 경기 행 | 773,446 |
| exact 팀 역할 관측 | 1,546,892 |
| exact 기준표 키 | 114 |
| exact 정방향 충돌 키 | 0 |
| exact 역방향 충돌 키 | 0 |
| exact 기준표 적용 가능 역할 관측 | 2,541,592 |
| exact 기준표와 일치 | 2,541,592 |
| exact 기준표와 불일치 | 0 |
| exact 기준표 미포함 역할 관측 | 8,956 |
| 투수·타자 양쪽 모두 exact 기준표 적용 가능 행 | 1,266,318 |
| 한 팀 이상 exact 기준표 미포함인 행 | 8,956 |

최종 행의 약 99.30%는 투수팀과 타자팀 모두 exact 경기에서 만든 팀 대응표로 직접 검증됐고 모두 일치했다.

## 5. exact 기준표 미포함 11개 조합

다음 11개 `시즌+게임타입+Train 팀 ID` 조합은 final 매칭에는 존재하지만 exact 경기에는 없었다.

| season | game_type | Train team_id | Trackman team_name | 투수 역할 행 | 타자 역할 행 | 역할 관측 합계 | 포함 경기 |
|---:|---|---:|---|---:|---:|---:|---:|
| 2019 | F | 22 | KBO_POL | 128 | 144 | 272 | 1 |
| 2020 | F | 15 | MIN_LOT | 252 | 317 | 569 | 2 |
| 2020 | F | 18 | MIN_SAM | 374 | 417 | 791 | 3 |
| 2020 | F | 19 | MIN_NCD | 501 | 408 | 909 | 3 |
| 2020 | F | 20 | MIN_KTW | 463 | 455 | 918 | 3 |
| 2022 | F | 14 | MIN_HER | 769 | 875 | 1,644 | 6 |
| 2022 | F | 15 | MIN_LOT | 421 | 439 | 860 | 3 |
| 2022 | F | 16 | MIN_KIA | 354 | 395 | 749 | 3 |
| 2022 | F | 23 | KBO_ARM | 74 | 115 | 189 | 1 |
| 2023 | F | 20 | MIN_KTW | 498 | 607 | 1,105 | 4 |
| 2023 | F | 23 | KBO_ARM | 465 | 485 | 950 | 3 |
| **합계** | | | | **4,299** | **4,657** | **8,956** | |

이 11개 조합도 전체 최종 매칭에서는 각 Train 팀 ID가 표의 Trackman 팀명 하나로만 연결됐다. 반대로 각 Trackman 팀명도 같은 시즌·게임타입에서 Train 팀 ID 하나에만 대응했다.

## 6. 최종 매칭에 등장한 팀 범위

Train 팀 ID는 12개다.

```text
12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
```

최종 연결 행에서 확인된 Trackman 팀명은 24개다.

```text
DOO_BEA
HAN_EAG
KBO_ARM
KBO_POL
KIA_TIG
KIW_HER
KT_WIZ
LG_TWI
LOT_GIA
MIN_DOO
MIN_HAN
MIN_HER
MIN_KIA
MIN_KTW
MIN_LGT
MIN_LOT
MIN_NCD
MIN_SAM
MIN_SKW
MIN_SSG
NC_DIN
SAM_LIO
SK_WYV
SSG_LAN
```

전체 최종 매칭에서 관찰된 정방향 팀 키는 125개다.

- exact 기준표에 포함: 114개
- exact 기준표 미포함: 11개
- 전체 정방향 다중 팀명 충돌: 0개
- 전체 역방향 다중 팀 ID 충돌: 0개

## 7. 해석

이 결과는 다음 오류를 강하게 배제한다.

- Train 투수팀이 Trackman의 다른 팀으로 연결된 경우
- Train 타자팀이 Trackman의 다른 팀으로 연결된 경우
- 같은 시즌·게임타입의 Train 팀 ID가 여러 Trackman 팀명으로 갈라진 경우
- 같은 Trackman 팀명이 여러 Train 팀 ID로 역매핑된 경우
- Train과 Trackman의 시즌이 다른 행을 연결한 경우

다만 팀 검증만으로는 같은 시즌에 같은 두 팀이 치른 서로 다른 경기를 바꿔 연결한 오류를 잡을 수 없다. 따라서 팀 검증은 경기 수열·`pitch_no`·투수·타자 흐름 검증을 보강하는 독립 검증이며, 이를 대체하지 않는다.

## 8. 기존 팀 매핑 문서와의 차이

기존 2026-08-16 팀 매핑 보고서는 당시 확정된 2,700경기를 제외한 잔여 Trackman 경기에서 팀명으로 추가 경기 후보를 찾기 위한 분석이다.

이 문서는 목적과 기준이 다르다.

- 기존 문서: 팀 매핑으로 추가 경기 후보 탐색
- 현재 문서: 최종 `trackman_id` 1,275,274행의 팀 일관성 전수 검증

따라서 최종 매칭의 팀 검증 결과는 이 문서를 기준으로 한다.

## 9. 최종 판정

최종 Trackman ID 연결 행은 팀 기준 검증을 통과했다. exact 기준표로 독립 검증된 역할 관측 2,541,592건은 모두 일치했고, exact에 없던 8,956건도 최종 전체 매핑에서 일대일성을 유지했다. 팀 정보로 발견된 최종 매칭 오류는 0건이다.

## 10. 현재 기준 팀 대응표

### 10.1 간단 조회용 대응표

상세 검증 컬럼이 필요하지 않고 `season + game_type + train_team_id`에서 Trackman 팀명만 조회할 때는 간단 팀 대응표를 사용한다. 컬럼은 `season`, `game_type`, `train_team_id`, `trackman_team_name`, `mapping_status`로 구성된다. 원본 상세 대응표는 근거·후보·검증 정보를 포함하므로 그대로 보존한다.

현재 팀 ID 대응표는 Train에 등장하는 전체 `season + game_type + team_id` 126개 조합을 한 행씩 정리한다.

| 상태 | 조합 수 | `trackman_team_name` |
|---|---:|---|
| `confirmed_exact` | 114 | exact 경기에서 양방향 일대일로 확인된 팀명 |
| `consistent_final_only` | 11 | 최종 연결에서는 일대일이나 exact 근거가 없는 팀명 |
| `unresolved_no_final_link` | 1 | 현재 연결 근거가 없어 공란 |

`consistent_final_only`는 최종 연결 내부의 일관성을 뜻하며 exact 확정과 같은 등급으로 해석하지 않는다. `unresolved_no_final_link`는 `2024 + F + team_id 25` 조합이다.

팀·선수 대응표를 함께 보려면 통합 대응표 엑셀의 `Team_Mapping` 시트를 사용한다.
