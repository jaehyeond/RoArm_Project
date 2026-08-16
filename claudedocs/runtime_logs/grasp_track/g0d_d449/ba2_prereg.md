# `g0d_d449` / `ba2` preregistration — B601 full-arm side pick → carry → **place+release**: 팔 전체가 원통을 잡아 pedestal B로 옮겨 내려놓고 손을 떼는가

- Date: 2026-08-14 KST (62nd session)
- User authority: 62nd 채팅 사용자 명시 승인 — 원문 **"ba2 place 진행해 — side 파지
  먼저. b601구매는 보류."** (61st 말미 선택지 4 "ba2~ 확장"의 place 사다리 Step A,
  BACKLOG `b601_stacking_long_horizon_ladder` Step A 승격). 구매 보류 = sim 트랙만.
- 이번 case의 신규 변수 (2개): `[place 시퀀스 추가: carry→descend→release(개방)→retreat]`,
  `[제2 지지대 pedestal B + 착지 목표 지점]`. 그 외 전부 ba1에서 상속 (동일 자산+수리
  2종, 동일 원통/마찰/드라이브 게인, 동일 φ=225° side 파지 기하, 동일 게이트 상수 계열).
- Scope: 물리 실행 O (실패 가능 — carry/place/release 각 층위 신규), Isaac Sim 5.1
  로컬 4090 headless, RTX **키프레임 정지컷만** (9장 — **mp4 없음**: D324 궤적 영상
  금지는 ba1 승인 1건 한정이었고 본 case에 영상 승인 없음. 사후 영상이 필요하면
  trace의 q_meas/obj pose로 bg1v 계열 kinematic 재생 렌더 가능 — 별도 승인 필요),
  로봇 하드웨어 0, RunPod 0, lerobot-train 0, `g0a_*`/`g0b_*`/`g0c_*`/`g0d_d448`
  기존 산출물 편집 0, D427~D448 재판정 0.

## 1. Decision question / branch semantics / non-claims

- 질문: ba1에서 성립한 side 파지+리프트(1 pose)가 **운반(carry)→하강(descend)→
  개방(release)→후퇴(retreat)**까지 이어져, 원통이 pedestal B 위 목표 지점에
  직립·정지 상태로 놓이고 로봇이 무접촉으로 물러나는가 — 즉 pick-and-place 전 주기의
  sim 필요조건. (스태킹 사다리 Step A: 다음 ba3 "원통 위 원통"의 전제.)
- Branch semantics (1 pose 단판, 게이트 우선순위 순):
  - `BA2_TCP_TRACK_FAIL` — G-track-A(파지 전) 또는 G-track-B(release 전) TCP 오차
    >10 mm: 파지/배치 판정 이전에 제어 층위 실패로 분류 (이후 기록만).
  - `BA2_NO_BILATERAL` — G1 실패 (ba1 재현 실패 — 예상 밖, 원인 분석 필수).
  - `BA2_LIFT_FAIL` — G2 실패 (들어올리기 실패).
  - `BA2_CARRY_DROP` — G4 실패: 운반~하강 중 3D 상대 슬립 ≥6 mm (동적 유지 실패).
  - `BA2_PLACE_MISS` — G5 실패: 최종 물체 xy가 B 중심에서 >10 mm.
  - `BA2_TOPPLE_OR_UNSETTLED` — G6 실패: 기울어짐 >5° 또는 z 이탈 >3 mm 또는 미정착.
  - `BA2_RETREAT_CONTACT` — G7 실패: 최종 settle 중 로봇이 물체에 접촉 잔존.
  - `BA2_FULL_ARM_SIDE_PLACE_RELEASE_SUCCESS` — G1∧G2∧G4∧G5∧G6∧G7 전부 PASS.
  - IK/게이트 abort — Isaac 진입 전 종료 (실패 시 같은 태그 재실행 금지 → ba3 forward-only).
- Non-claims: 실물 B601 (게인·마찰 하네스 저작, D448 ④ 유지), top-down full-arm,
  타 방위각/배치 일반화, **원통-원통 적층 안정성 (ba3 영역 — 본 case는 pedestal 위
  place까지)**, 마찰 현실성, D445~D448 재판정. 키프레임 그림 단독 제시 금지 —
  G 수치 캡션 의무 (bg1v/ba1 규율 상속).

## 2. Method authority + 스모크 면제 근거

- 물리/제어/캡처/라이프사이클 전부 ba1 prereg §2 verbatim 상속: 명시적 scene-int
  스테핑(PhysxSceneAPI update Disabled + simulate_scene/fetch_results_scene, CPU
  PhysX, dt 1/60), DriveAPI targetPosition 매 step 저작(min-jerk), JointStateAPI 초기
  상태 = standoff_A, `rep.orchestrator.step()` 키프레임 캡처(D447 ③), BaseException
  캡처 + failure.json + rc sentinel + fsync → app.close() (D447 ①), 확장 enable 사이
  app.update() 펌프 3회 (61st 데드락 교훈).
- **별도 하네스 스모크 생략 (사전 선언)**: ba2의 하네스 메커니즘은 전부 ba1 스모크로
  기실증 — 계층 수리(S1), 정적 유지/스텝 추종(S2/S2b), FK-vs-USD(S3), **손가락
  전폐→재개방(S4 — 본 case OPEN phase의 메커니즘)**, contact report 캘리브레이션(S5),
  캡처 무간섭(S6). 신규는 궤적 내용과 phase 표뿐(하네스 아님). abort 층위는 preflight
  IK 재계산 게이트 + SETTLE 스모크 게이트(ba1 try-3이 기하 오류를 여기서 발각한 전례)
  + SETTLE2/SETTLE3 track 게이트가 담당.
- OPEN 정책 (사전 선언): OPEN phase 90 step 동안 손가락 목표를 0→0.0715 m min-jerk
  램프 (스텝 개방의 튕김 방지). 팔 목표는 place_B 고정.
- PLACE_GAP (사전 선언): place 시 TCP 목표 z = B 중심 z + **2 mm** (물체 바닥이
  pedestal 상면 2 mm 위 → 개방 시 2 mm 자유 낙하로 안착; restitution 0).

## 3. Frozen inputs / pins

- 자산/수리/드라이브/물체/마찰/articulation/env 핀: **ba1 prereg §3 전부 verbatim 상속**
  — 9파일 SHA(`b601_asset/UPSTREAM.md`), split2 bit-copy 이식+게이트 4종(census
  2+1/SHA/blade 극값 dev<1e-9/조인트 verbatim), 계층 수리(world-bake+resetXformStack,
  dev<1e-9), 팔 게인 1e5/500 + maxForce 공식 27/27/27/7/7/7, 그리퍼 5e3/2e2 +
  maxForce 100 N, 원통 D29×H50 0.02483 kg 마찰 0.40/0.30, sleepThreshold 0 전면,
  FilteredPairs 좌우 핑거 1쌍 + ground↔base, `numpy==1.26.0`/`psutil==5.9.8`/
  rerun 0.34.1. (ffmpeg 불사용 — mp4 없음.)
- 운동학 모델/IK 게이트: ba1 §3-2 verbatim — 조인트 테이블 repr-일치 게이트,
  FK-vs-USD < 1e-3 m (SETTLE 종료), preflight IK 재계산 vs 설계 핀 max|Δq| < 0.1°,
  수렴 pos<1e-6 ∧ ori<1e-6 ∧ 한계 여유 >5°.
- 지지: pedestal A ba1 verbatim (0.05×0.05×0.095 @ A 직하, 마찰 1.0) + **pedestal B
  동일 규격 @ (0.40, 0.08) 직하** + 바닥 슬래브 2×2×0.02. Pedestal A/B 비중첩 게이트:
  중심 이격 (dx,dy)=(0.06,0.08) → x축 면간 10 mm (≥5 mm gate).

## 4. 배치 + waypoint (오프라인 IK 스캔으로 설계 확정 — 62nd, 물리 0)

- Pick 기하 = ba1 §4 REV-1 verbatim: A = (0.34, 0, 0.12), φ=225°, X_TCP=−0.02448,
  standoff 0.05, lift +0.08. standoff/grasp/lift 설계 q 3종도 ba1 verbatim (재계산 일치).
- **Place 목표 B = (0.40, 0.08, 0.12)** (pedestal B 상면 0.095 + H/2). 선정 = 그리드
  스캔 (x 0.24~0.40, y −0.20~+0.20, step 0.02; 이격 ≥0.10, 베이스 이격 ≥0.20) 중
  feasible 13곳에서 **최대 한계 여유** (min margin 12.49°, 구간 최대 이동 24.1°).
- 신규 waypoint (IK 설계값 [j1..j6]°, preflight 재계산 일치 게이트):
  - hover_B (B 위 +0.08 m): `[−6.42028, −99.03786, −33.51464, −65.52375, −51.42028, 0.00027]`
  - place_B (TCP z = 0.122 = B z + gap 2 mm): `[−6.42022, −120.89471, −31.26793, −89.62732, −51.42022, 0.00027]`
  - retreat_B (place − approach·0.05): `[−15.02370, −119.18791, −24.53600, −94.65265, −60.02370, 0.00044]`
- 경로 검증 (설계 스캔 + preflight 재계산 게이트, 121점 dense FK/구간):
  - TRANSFER (lift_A→hover_B): TCP z = 0.200 상수 (min 0.19999999946 — j1·j5가 −45°
    오프셋 동조로 자세 편차 5.8e-5° 원호 경로), 자세 편차 gate <12°.
  - DESCEND/RETREAT 자세 편차 ~0° (gate <8°).
  - 게이트: TRANSFER TCP z_min > 0.155 (물체 바닥 >0.13 = 양 pedestal 상면 +35 mm).
- 스폰/착지 기하 무접촉 검증 (해석적, ba1 정정 공식): blade z 밴드 [0.100,0.140] (파지)
  / [0.102,0.142] (place) > 상면 0.095+3 mm; 팜 최근접 수평 0.0485 m > 반폭 대각
  0.0354 m (+13.1 mm, A·B 동일 — φ 동일·규격 동일).

## 5. Phases + gates

dt=1/60, 총 **1020 step (sim 17.0 s)**. 팔 = min-jerk 보간, 손가락 = 상수(OPEN 램프 제외).

| Phase | steps | 구간 [t] | 팔 목표 | 손가락 | 비고 |
|---|---|---|---|---|---|
| SETTLE | 30 | 0–29 | standoff_A 고정 | 0.0715 | 스모크 겸용 게이트 (ba1 §5) |
| APPROACH | 120 | 30–149 | standoff→grasp | 0.0715 | |
| SETTLE2 | 30 | 150–179 | grasp 고정 | 0.0715 | 끝: G-track-A TCP 실측 |
| CLOSE | 120 | 180–299 | grasp 고정 | **0.0** | ba1 verbatim |
| LIFT | 120 | 300–419 | grasp→lift_A | 0.0 | 끝: G2 판정점 |
| TRANSFER | 180 | 420–599 | lift_A→hover_B | 0.0 | 원호 경로 z=0.20 |
| DESCEND | 120 | 600–719 | hover_B→place_B | 0.0 | |
| SETTLE3 | 30 | 720–749 | place_B 고정 | 0.0 | 끝: G-track-B + G4 판정점 |
| OPEN | 90 | 750–839 | place_B 고정 | 0→0.0715 램프 | release |
| RETREAT | 120 | 840–959 | place_B→retreat_B | 0.0715 | |
| SETTLE4 | 60 | 960–1019 | retreat_B 고정 | 0.0715 | 끝: G5/G6/G7 판정점 |

- 게이트 (SUCCESS = G1∧G2∧G4∧G5∧G6∧G7; G-track 2종은 분류용):
  - **G1 close_bilateral**: CLOSE 중 같은 step min(F_L,F_R) > 0.01 N (ba1 verbatim).
  - **G2 lift_follow**: obj z 상승(LIFT 끝−CLOSE 끝) ≥ glf z 상승 −6 mm ∧ glf 상승 ≥60 mm.
  - **G4 carry_hold**: ‖Δ(p_glf−p_obj)‖ (CLOSE 끝→SETTLE3 끝, 3D) < 6 mm.
  - **G5 place_xy**: SETTLE4 끝 물체 xy − B xy ≤ 10 mm.
  - **G6 upright_rest**: 물체 축 기울기 ≤5° ∧ |z−0.12| ≤3 mm ∧ 마지막 30 step 이동 <1 mm.
  - **G7 release_clear**: SETTLE4 중 핑거·팜→물체 접촉력 최대 < 0.01 N.
  - **G-track-A/B**: SETTLE2/SETTLE3 끝 TCP 오차 ≤10 mm (초과 시 `BA2_TCP_TRACK_FAIL`).
  - measurement_valid: ba1 verbatim (SETTLE drift <2 mm, FK dev <1e-3, base 부동,
    전 step 유한, step 카운트 정합) + 지지 접촉력 m·g 캘리브레이션 기록.
- 진단 기록 (게이트 아님): RETREAT 중 물체 접촉 피크, 팔 무관 접촉 피크, OPEN 중
  물체 z 낙하 프로파일, HOLD-미사용 사유 = place가 대체.

## 6. Outputs (forward-only, `g0d_d449/`만 쓰기; 실패 시 같은 태그 재실행 금지 → ba3)

`ba2_prereg.md`(본 문서) / `ba2_script.py.txt` / `ba2_argv.txt` / `ba2_results.json` /
`ba2_trace.npz` / `ba2_timeline.rrd` / `ba2_timeline.rbl` / `ba2_rerun_validation.json` /
`ba2_inspection.png` / `ba2_key_*.png` (phase 경계 9장: settle/pregrasp/close/lift/
transfer/preplace/open/retreat/final) / `ba2_stdout.log` / `ba2_stderr.log` /
`ba2_exit_status.txt` / (실패 시) `ba2_failure.json`. **mp4 없음** (§0 Scope).

## 7. D341 observability

- verdict가 궤적·접촉·시간에 의존 → **RRD 의무**: 전 1020 step의 q(8), F_L/F_R/
  bilateral-min/팜, obj z·glf z·obj-xy-오차(B 기준), phase TextLog, 키프레임 3D
  (핑거 점군+원통 wire+파지·착지 목표 axes+pedestal A/B wire), verdict TextLog.
  rerun 0.34.1 핀, footer verify, exact entity/timeline/component 계약, 고정
  blueprint + `.rbl`, headless `ba2_inspection.png`, **실제 육안 검수 관찰 기록**
  (세션 doc). 문자열 내 수치는 상수에서 f-string 생성 (ba1 RRD 제목 "phi=135"
  오기 재발 방지).
