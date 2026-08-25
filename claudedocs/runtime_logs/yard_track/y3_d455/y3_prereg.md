# y3_d455 prereg — 정책 비교층 v1 (rule-policy comparison on identical initial pile)

날짜: 2026-08-16 (69th). 사용자 지시 verbatim: "(b4) y3 진행해. H_max 강제/해상도/
release 높이는 너가 권장안 제시하고 시작해."

**이번 case의 신규 변수 (D322): [① place 행동 계약 v2(존 3×3 + H_max 관측-마스크
+ release 벽-클리어런스 하한 — 3개 사전 결정의 묶음, 전 arm 공통 상수),
② 정책군 스윕(pick 5종 / place 3종)] — 2개.**
정책 비교가 측정 대상 변수, place 계약 v2는 전 arm 동일 적용 상수(스윕 아님).
release 변경만은 a8 브리지 run으로 y2 대비 격리 측정.

## 0. 승계 핀 (변경 금지)

- 초기 더미 = y1_d453 yt3 웨이브 스폰 **verbatim** (SEED=45300, 2×2 ±27.5mm,
  z=0.20, 45 step 간격, 8웨이브×4개) — D453 bit-재현 규약.
- `y1_design.json` sha256 = `a045c414…` / manifest = `a1127acc…` (p29 PINS 승계).
- pick 프리미티브 = source 높이맵 셀 수직 레이 hit 물체 RemovePrim (파지 물리
  추상화 — 비주장 유지). 합동 정착 streak 30 / cap 400. 관측 = 13×13@10mm
  레이캐스트 (권위 표면 = cooked hull, D453).
- env 핀: numpy 1.26.0 / psutil 5.9.8 / rerun-sdk 0.34.1 (D326).

## 1. 사전 결정 3건 (D454 ③⑤가 요구한 결정 — 근거 포함 선언)

### 결정 1 — H_max 강제 = 관측-기반 사전 마스크 + 사후 위반 회계 유지

- 존 z 유효 ⟺ `pred(z) = H_bin_obs[voronoi(z)].max() + class_mm/1000 ≤ H_MAX + H_TOL`
  (class_mm = 이번에 집은 물체의 클래스 공칭 크기 — 증분 추정자로 선언;
  추정자 오차는 사후 위반 회계와 pred-실측 대조로 검증).
- 전 존 무효 시: `argmin pred` 존 선택 + `mask_exhausted` 카운트 (에피소드
  실패 아님 — 완주가 RQ2 지표이므로).
- 재시도-비용 방식 대신 마스크인 이유: (i) 모든 규칙 정책에 동일 적용되는
  공정 필터, (ii) 향후 RL의 표준 형식(invalid action masking), (iii) 실조업
  대응 — 적치 한계는 회피 대상이지 시행착오 대상이 아님.
- 사후 위반(착지 분산으로 인한 초과)은 게이트가 아니라 **회계**로 기록.

### 결정 2 — place 행동 해상도 = 3×3 존 (간격 40mm)

- 존 중심 셀 = (2,2),(2,6),(2,10),(6,2),(6,6),(6,10),(10,2),(10,6),(10,10)
  — **yp1 9지점 라스터와 동일 좌표** (연속성). Voronoi 분할 행/열 =
  {0..3},{4..8},{9..12} (전 169셀 완전 분할 — H_max 마스크 커버리지 완전).
- 근거: 착지 분산 p95 43.1mm(yp1) ≫ 셀 10mm — 셀 단위 목표 지정은 분산
  아래의 허위 정밀도(D454 ③). 관측 해상도는 10mm 셀 유지(행동≠관측 해상도).

### 결정 3 — release 높이 = 벽-클리어런스 하한 (v2)

- v2: `drop_z = max(H_wall, H_bin_obs.max()) + r_bound + 8mm`, H_wall=0.080.
  (y2의 `max(0.20, global_max + r + 8mm)`에서 **0.20 floor만 교체** —
  global-max 항은 유지해 스폰 겹침 위험 불변.)
- 근거: 0.20m은 y2에서 "조건부 상한" 인공값으로 명시(D454 ②). 실기 release는
  벽 상단 + 후행 체적 클리어런스(D449) 아래로 불가 → 벽+r+8mm가 현실-정합
  하한. 전형 drop_z ≈ 0.10~0.115m (낙하고 ~3~6cm vs y2 ~15cm).
- **a8 브리지 run** = 동일 정책(a1)에서 y2 release 공식만 사용 → release
  효과를 침묵 교체가 아닌 동일-정책 대조로 격리.
- 비주장 유지: 실기 분산 절대값 (arm 기구학 없음 — 하한 근사일 뿐).

## 2. 정책군 (arms — 물리 run 8 + rep2 1)

pick 정책 (place 고정 = masked_raster):
| arm | pick | 정의 (전부 결정론, tie = row-major 첫 인덱스) |
|---|---|---|
| a1 | greedy_high | argmax H_src (y2 yp1 pick 재현) — **주축, rep2 대상** |
| a2 | scan_gated | 라스터 포인터 순환, 다음 h≥5mm 셀 |
| a3 | greedy_low | argmin over {h≥5mm} |
| a4 | random_gated | rng(45301) 균등 among {h≥5mm} — 정책 RNG는 스폰 SEED와 분리 선언 |
| a5 | blind_raster | 무관측: 포인터 셀 무조건 시도. 레이 hit∉rocks → **no-op**(동작 1 소모, 물리 0 step, 관측 재사용) |

place 정책 (pick 고정 = greedy_high):
| arm | place | 정의 |
|---|---|---|
| a1 | masked_raster | 존 순환 포인터, 유효 존만; 전무효→argmin pred |
| a6 | masked_argmin | 유효 존 중 argmin pred (부하 균형) |
| a7 | stack_masked | (6,6) 고정, 무효 시 argmin pred — **yp2(비마스크 stack) 대조: 마스크의 강등-저지 효과** |

release 브리지:
| arm | 구성 |
|---|---|
| a8 | a1과 동일 정책 + **y2 release 공식**(0.20 floor + global max) |

- 에피소드 종료 = H_src_obs 전 셀 < 5mm (harness 판정) 또는 ACTION_CAP=400.
- 물리 사이클은 arm당 정확히 ≤32 (pick 성공 시에만 물리) — T_ALLOC = p29 동일.
- rep2 = a1 cold-start 별도 프로세스: 최종 bin obs bit·분산 수열·총 동작수 대조.

## 3. 게이트 (기계) vs 측정 (연구)

게이트 — 하나라도 FAIL이면 arm verdict FAIL (게이트-쇼핑 금지, FAIL 보존):
- G-initial: 정착 + 32/32 src (p29 승계).
- G-action-contract: obs-gated arm(a1~a4,a6~a8)에서 no-op 0건; no-op 발생 시
  해당 셀 h_obs < 5mm (blind 포함 전 arm — 관측-레이 일관성).
- G-cycle-settle: 전 물리 사이클 settled (streak 30, cap 400).
- G-complete: cap 내 src 0 / bin 32 / out 0 (전 arm — blind 포함).
- G-final-hmap: 전 338셀 중 ≥95%가 |obs−GT| ≤ 0.5mm (p29 승계, p95 게이트).
- G-repro (a1 rep2): bin obs max|Δ| ≤ 2mm → 분기 (i); 목표 = Δ 0.0.

측정 (게이트 아님): 총 동작수·낭비 동작수(no-op)·마스크 이벤트 수·
mask_exhausted·H_max 위반(final/any)·pred-실측 증분 오차·재형성 합계·분산
stats·bin 최종 hmax·클래스 소진 순서·a8 vs a1 분산 대조.

## 4. 예측 (사전 기재 — 반증 가능)

1. obs-gated arm 총 동작수 = 정확 32; a5(blind) > 32 (관측의 가치 정량 채널).
2. a7 최종 H_max 위반 ≈ 0~2셀 vs yp2 비마스크 11셀 (마스크의 강등-저지).
3. a8 분산(mean·p95) > a1 (release 하향 = 분산 축소 — 방향 예측).
4. pick 순서별 재형성 합계는 다를 것 — 방향 무예측 (탐색적 측정).

## 5. 비주장 (non-claims)

파지 물리(추상화) / 실기 분산 절대값 / RL 우월성(규칙군 비교일 뿐, "RL이
이긴다" 주장 금지 — 파일럿 프레임과 동일 규율) / 단일 초기 더미(통계적
일반화 비주장 — 다중 시드는 후속 case) / Kinect 충실도 / 실물 질량(15%
infill 추정) / 실조업 목적("높이 우선"은 대리 휴리스틱 — START_HERE 규율).

## 6. D341 계약

arm당(rep2 제외): save-only RRD 0.34.1 + footer verify + exact entity/
timeline/component + blueprint+rbl + inspection.png + 육안 기록. p29 엔티티
집합 승계 + `plots/masked_zones`·`plots/total_actions` 추가, no-op·mask
이벤트는 `events/phase` TextLog. Authority = `{arm}_results.json` + trace.npz
(f-string TAG — yt3 하드코딩 오기 재발 방지 확인 항목).

## 7. 산출물

`claudedocs/runtime_logs/yard_track/y3_d455/` 하위:
`{arm}_{results.json,trace.npz,timeline.rrd,timeline.rbl,rerun_validation.json,
inspection.png,final_heightmap.png,stdout.log,stderr.log,exit_status.txt,
argv.txt}`, a1은 `+script.py.txt`, rep2는 `a1_rep2_*` + `a1_repeat_compare.json`,
교차 요약 = `y3_policy_summary.json` + `y3_policy_summary.png` (p32).
러너 = `sim_scripts/p31_y3_policy_compare_probe.py`, 요약 =
`sim_scripts/p32_y3_policy_summary.py`. WRITE_GUARD·failure.json(D447)·
PRE_CLOSE_SENTINEL·argv/script 사본 p29 승계.
