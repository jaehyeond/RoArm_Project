# y2_d454 prereg — pick-place 전이: 재형성·착지 분산·H_max 회계 (yp1 spread / yp2 stack)

작성: 2026-08-16 68th. 실행 전 작성·동결 (수정은 REV append + 실행 전 선언).
상위: `y1_d453` (D453 — 테스트베드 기하·스폰 규약·높이맵 파이프라인 승계, SHA 핀).

## SS0 — 목적과 신규 변수

**이번 case의 신규 변수: [① pick 프리미티브(관측 기반 물체 제거),
② place 프리미티브(지정 셀 낙하 적치 + H_max 회계)] — 정확히 2개** (D322~).
yp1/yp2의 place 패턴 차이는 스윕 조건(fg2 사다리 전례)이지 신규 변수 아님.

- **질문 3 (연구 전제 직결)**: (a) pick마다 source 더미가 제거 footprint 밖에서
  재형성되는가·얼마나 (b) 지정 place 셀 대비 실제 착지 분산은 몇 mm인가
  (c) place 선택(분산 vs 적층)이 bin 상태·H_max 위반을 측정 가능하게 바꾸는가.
- **비목표**: 판단 정책 비교(greedy/OT/RL — 후속 case), 파지 물리(동결 grasp
  track·D452 실기 처방 별도), 로봇 팔, Kinect 관측 현실화, H_max **강제**
  (본 case는 회계=사후 계수만; 행동 무효화는 정책층 case).

## SS1 — 승계 자산 (SHA 핀, preflight 검증)

| 자산 | 값 |
|---|---|
| 기하 | `y1_d453/y1_design.json` `a045c414bfeb381e` (source/bin 13×13@10mm, 벽 80mm, **H_max=80mm**) |
| 물체 | o1 manifest `a1127acc` subset 32 (클래스당 8, y1 동일) |
| 초기 상태 | **yt3 웨이브 스폰 규약 verbatim** (2×2±27.5mm×8웨이브, z=0.20, 45step, seed 45300 — D453: cold-start bit-재현 확인) |
| 물리/정착 | dt 1/60·마찰 0.40/0.30·반발 0.1·solver p22 승계·정착 |v|<5mm/s ∧ |ω|<0.1rad/s |
| 하네스 | p27 검증 블록(부트/실패 캡처/selfcheck/런타임 dynamic 저작/레이캐스트) |

## SS2 — 전이 프리미티브 정의 (프로토콜)

- **pick(관측 기반, closed-loop)**: source 높이맵(레이캐스트) argmax 셀(동률 =
  row-major 첫 인덱스) → 그 셀 수직 레이의 hit prim = 집는 물체. **제거 =
  RemovePrim** (조언 프레임의 "RGB-D 관측 매 동작 후 갱신" 그대로 — 파지
  물리는 추상화, 비주장).
- **place**: 대상 물체를 bin 목표 셀 상공에 **dynamic 신규 저작**(pick 시점
  자세 유지 — 정체성 보존; kinematic 텔레포트 금지, D453 ④).
  낙하 z = **max(0.20, bin 높이맵 max + r_bound(물체) + 8mm)** — 관측의
  결정론적 함수(스폰-더미 교차 방지).
- **cycle**: pick 제거 + place 저작(동일 시점) → **합동 정착**(양 트레이 전
  물체, streak 30, cap 400 step) → 양 영역 높이맵 관측 → 지표 기록. 32 cycle
  = 전량 이송 에피소드.

## SS3 — 프로브 2종

- **yp1 (spread)**: place 목표 = bin 셀 (r,c)∈{2,6,10}² 9지점 라스터 순환
  (cycle c → pattern[c mod 9]; 4,4,4,4,3,3,3,3,3회 배분).
- **yp2 (stack)**: place 목표 = 전 cycle 중앙 셀 (6,6) 고정 — 나쁜 place
  정책의 귀결(H_max 위반·기둥 붕괴·이탈)을 측정하는 대조군.
- **yp1 rep2**: cold-start 동일 시드 재실행 — 에피소드 수준 재현성 분기
  (i) max|Δ| ≤ 2mm → 에피소드 재현 가능 / (ii) 초과 → cycle별 상태 기록 필수.

## SS4 — 게이트(기계 성립)와 측정(연구 지표)의 분리

**게이트 (yp1·yp2 공통 = 기계)**:
- G-initial: 초기 스폰 32/32 유지 + 정착 (yt3 게이트 승계).
- G-pick-valid: 32/32 — argmax 셀 레이 hit가 물체 prim (벽/바닥 무효).
- G-cycle-settle: 전 cycle 합동 정착 ≤ 400 step.
- G-final-hmap: 종료 상태 obs vs numpy GT — **양 영역 합산 95% 셀 |Δ|≤0.5mm**
  (max 절대 게이트는 D453 ②에 따라 제외 — 경사 tan(θ) 증폭은 게이트가 아니라
  p28식 분류로 보고).
**게이트 (yp1 전용)**: G-transfer-complete — 종료 시 source 물체 0·bin 내부
COM 32/32 (spread에서 이탈=기계·설계 결함). **yp2는 이탈·미완주가 측정값**
(나쁜 정책의 귀결) — 기계 게이트만 적용, 완주 여부는 지표로 보고.
**측정 (게이트 아님)**:
- 재형성: cycle별 |ΔH_src|>2mm 셀 수(제거 물체 pre-pick AABB+1셀 footprint
  제외). 분기: 평균 ≈0 → 전제 약화 정직 기록 / >0 → 전제 실증.
- 착지 분산: |실현 COM xy − 목표 셀 중심| [mm] — 분포(mean/p95/max).
  p95 > 30mm(3셀)면 10mm 행동 해상도 재설계 입력으로 기록.
- H_max 회계: cycle별 H_bin>80mm+0.5mm 셀 수(사후 계수), bin h_max 추이.
- bin 이탈 수·source 통로 낙하 수, cycle별 정착 step, 최종 heightmap 2종.

## SS5 — D341 계획

RRD 필수(전이 궤적+기하 verdict): save-only 0.34.1, 전 실행 step 타임라인
(rocks/centers·max_speed) + cycle 타임라인(PICK/PLACE 이벤트·분산·재형성·
H_max 스칼라), 최종 힐 와이어+높이맵 포인트, 고정 blueprint+rbl, footer
verify, inspection.png, 육안 기록. yp1 rep1 = RRD 권위; rep2·yp2도 각자 RRD
(yp2는 대조군 시각 증거 필요 — 적층 vs 분산 3D 비교).
**RRD 요약 텍스트의 권위 파일명은 f-string으로 TAG 반영** (yt3 오기 재발 방지).

## SS6 — 순응·산출

- 로봇 0 / lerobot-train 0 / git 0(사용자) / HANDOFF 0 / 동결 case 편집 0
  (y1 자산 read-only 핀 참조). env 핀 numpy 1.26.0/psutil 5.9.8/rerun 0.34.1.
- 러너 `sim_scripts/p29_y2_yp_transfer_probe.py` (--probe {yp1,yp2} --rep {1,2}),
  preflight 단독 선행, 실패 캡처(D447), 실패 attempt 증거 보존 규약(67th 방식).
- 산출: `yp{1,2}_{results.json,trace.npz,timeline.rrd/.rbl,rerun_validation.json,
  inspection.png,stdout,stderr,exit_status,argv,script.py.txt}` +
  `yp1_rep2_*` + `yp1_repeat_compare.json` + 최종 높이맵 PNG 2종 +
  대조 요약 `y2_contrast_summary.json` (yp1 vs yp2 — 분석 스크립트 p30).
- 실패 가능성 실재: G-cycle-settle(연쇄 붕괴 미수렴)·G-pick-valid(관측-물체
  귀속 오류)·G-transfer-complete(yp1 이탈)·G-final-hmap 어느 것도 보장 없음.
