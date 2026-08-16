# 62nd — g0d_d449 `ba2`: B601 full-arm side pick→carry→place+release probe — verdict `BA2_TCP_TRACK_FAIL` (물리 G1~G7은 전부 PASS, 병목 = 이웃 pedestal A와 손목 충돌)

날짜: 2026-08-14 (61st 종료 후 재개 세션)
이번 case의 신규 변수 (2개): `[place 시퀀스: carry→descend→release→retreat]`,
`[제2 지지대 pedestal B (0.40, 0.08) + 착지 목표]`.
물리 실행 O (실패 가능 실험 — 실제로 분류 실패 발생), Isaac Sim 5.1 로컬 4090
headless + RTX 키프레임 9장 (mp4 0 — D324 유지), 로봇 하드웨어 0, RunPod 0,
lerobot-train 0, `g0a_*`/`g0b_*`/`g0c_*`/`g0d_d448` 편집 0, D427~D448 재판정 0,
git 커밋 0.

## 0. 부트 검증 + 사용자 승인

- 부트: HEAD == origin/master == `9cbd959` ✓, dirty 목록 61st 기대치 정확 일치 ✓,
  START_HERE(61st판)/61st doc/D445~D448 원문 확인 ✓.
- 사용자 승인 (verbatim): **"ba2 place 진행해 — side 파지 먼저. b601구매는 보류."**
  → 61st 선택지 4(ba2~ 확장) = BACKLOG `b601_stacking_long_horizon_ladder` Step A
  승격. 구매 보류 = sim 트랙만. 신규 case `g0d_d449` (태그 `ba2`) 개시.
- 직전 문답 2건 (같은 세션 초입): ① ba1의 collider/PhysX/충돌판정 설명 (p20 증거
  기반 — 판정 신규 0), ② Isaac Lab 병렬화/장기 stacking 사다리 평가 → BACKLOG
  2건 등재 (`isaaclab_parallel_grasp_env`, `b601_stacking_long_horizon_ladder`).

## 1. 설계 (물리 0 — 해석적, scratchpad `ba2_design_scan.py`)

- p20 FK/IK 기계 verbatim 재사용. ba1 A-waypoint 3종 오프라인 재계산 = 설계값
  일치 (sanity PASS).
- pedestal B 그리드 스캔 (x 0.24~0.40 × y −0.20~+0.20 step 0.02, A 이격 ≥0.10,
  베이스 이격 ≥0.20, IK 수렴+여유>5°, 경로 dense FK 가드): **feasible 13곳 중
  B=(0.40, 0.08) 선정** (최소 한계 여유 12.49°로 최대, 구간 최대 이동 24.1°).
- 신규 waypoint 3종 (hover/place/retreat, prereg §4 수치): TRANSFER 경로가
  **TCP z=0.200 상수 원호** (j1·j5 −45° 오프셋 동조로 자세 편차 5.8e-5°), place
  TCP z = 0.122 (PLACE_GAP 2 mm 낙하 안착 설계).
- 클리어런스 (해석적): blade 하단 파지 0.1004/착지 0.1024 > 0.098 게이트, 팜
  최근접 0.0485 > 반폭 대각 0.0354, pedestal A/B 비중첩 (y축 면간 30 mm).
  ★ **이 검사가 "파지 직하 지지물"만 검사한 것이 본 실행 실패의 설계 사각지대
  였음** — §4 진단 참조.

## 2. 실행 (p21, try 1회 — 스모크 면제는 prereg §2 사전 선언)

- preflight 드라이런 (물리 0): 핀 10종 SHA·IK 재계산 dev=0 (6/6)·경로 가드·
  클리어런스 전부 PASS 확인 후 본 실행 1회.
- phases: SETTLE30/APPROACH120/SETTLE2 30/CLOSE120/LIFT120/TRANSFER180/
  DESCEND120/SETTLE3 30/OPEN90/RETREAT120/SETTLE4 60 @dt 1/60 = **1020 step
  (17.0 s)**, wall **23.9 s**, rc=0, 종료 핀 10/10.
- SETTLE 게이트 (ba1 재현): drift 0.005 mm·추종 0.0055°·FK-vs-USD 0.321 mm·
  지지 접촉력 0.24357746 vs m·g 0.24358230 N (4자리).

## 3. 결과 — verdict `BA2_TCP_TRACK_FAIL` (prereg §1 분기 우선순위 그대로)

권위 = `ba2_results.json` (`d4a396b4f0496fa0`, 25,526 B) + `ba2_trace.npz`
(`f20447fb9e89a70a`).

- **G-track-A (파지 전): 0.43 mm PASS** — ba1과 동일 (재현).
- **G-track-B (release 전, SETTLE3 끝): 19.82 mm > 10 mm 게이트 → FAIL** ⇒
  prereg 분기 최우선순위로 verdict 확정. 이후 게이트는 기록.
- 물리 게이트 (기록 — 전부 PASS였음): G1 양측 peak **23.09 N** (L 24.46/R 25.37
  — ba1 재현) / G2 obj +79.69 vs glf +79.56 mm / G4 carry 슬립 **2.15 mm** /
  G5 착지 xy 오차 **7.42 mm** / G6 기울기 **0.0004°**·z 오차 5e-9 m·정착
  9e-8 m / G7 최종 접촉 **0.0 N**. supportB 최종 median 0.2436 N (= m·g,
  물체가 B 위에 실재 안착).
- **measurement_valid = FALSE**: 원인 항 = base 부동 검사 (§5 결함 참조).

## 4. 진단 — 병목 = 팜/손목이 **이웃 pedestal A**와 충돌 (60.4 N, 하강 차단)

1. `finger_env_F` 채널: **t=684 (DESCEND 중)부터 종료까지 지속 접촉**, peak
   **60.37 N** (t=694), SETTLE4에도 48.7 N — 팔이 pedestal A 위에 얹힌 채 종료.
2. 접촉 쌍 (results `raw_contact_stats.pairs_seen`): **link6·gripper_link(팜)·
   양 핑거 ↔ `/World/pedestal`(A)**. pedestal B와 로봇 접촉 없음.
3. 기하: φ=225 접근에서 손목 후방 축 = B에서 (−0.707,−0.707) 방향. A 중심은 그
   축을 따라 99 mm 후방, **수직 이격 단 14 mm** (A 반폭 대각 35.4 mm ⇒ 관통).
   descend가 손목 하우징을 A 상면 모서리에 얹음 → TCP가 목표 위 **+19.66 mm
   (z)**에서 정체 (t=719~749 오차 상수 19.82 mm), j4가 접촉 토크로 **−4.94°**
   밀림 (maxForce 7 N·m 포화).
4. 실측 하한 (신규 계측): 접촉 평형에서 glf z=0.1417·A 상면 0.095 ⇒ 손목/팜
   집합의 최저점은 glf 원점 아래 **≥46.7 mm** (j4 −4.94° 기운 자세 포함) —
   ba1 REV-1의 팜 밴드 38.5 mm보다 깊음 (link6 하우징 기여). place 높이
   (glf z 0.122)에서는 최저점 z ≈ 0.075 < pedestal 상면 0.095 ⇒ **후방 축이
   지지물 풋프린트를 지나면 구조적으로 차단**.
5. 그럼에도 물체는 안착: 개방 시 ~22 mm 자유낙하(설계 2 mm + 스톨 19.7 mm)로
   xy 7.42 mm·기울기 0.0004° 안착 (restitution 0, open 중 obj z min 0.120).
   ⚠️ 이 사실로 "place 성공" 주장 금지 — verdict는 분류 그대로 (prereg §1).

## 5. 결함 자진 신고 (러너 계측 2건 — 차기 태그 수정 의무)

- **base 부동 항 수치 미영속화**: measurement_valid의 base_move(<1e-6 m) 항이
  계산만 되고 results에 값 미기록 → False의 원인 항을 사후 수치 감사 불가.
  나머지 항(SETTLE drift/FK/step 카운트/유한성)은 개별 수치로 PASS 검증됨 ⇒
  소거법으로 base_move 항이 실패 원인. 물리적으로 그럴듯한 설명 = 지속 60 N
  외력·~0.4 m 레버가 고정 base 조인트 solver 컴플라이언스로 μm급 변위 유발
  — 단 **수치 없는 설명은 추정** [명시]. 차기 태그: 전 항 수치 영속화.
- **finger_env 채널이 게이트가 아니었음**: 60 N 환경 충돌이 abort 없이 완주됨.
  차기 태그: DESCEND~SETTLE3 중 finger/palm↔환경 접촉 게이트 추가 검토.

## 6. D341 완주 + 육안 검수 (Claude 직접)

- `validate_rerun_artifact` **pass=True errors=[]** (rerun 0.34.1, footer verify,
  exact entity 22종/timeline 3종/component 계약, blueprint+`.rbl`, headless PNG
  `ba2_inspection.png` `87273175c0e36168`).
- 육안 관찰: **키프레임 preplace** — 손목/팜 하우징이 회색 pedestal A 위에 얹혀
  있고 원통은 B 위 공중 (판독 명확) / **final** — 원통이 B 위 직립, 그리퍼 개방
  후퇴, 손목은 여전히 A 근접 / **inspection 패널 5** — DESCEND 곡선이 목표
  0.122 위 ~0.14 m에서 평탄 정체 (스톨 시각 확인), 패널 6 — t≈700부터 추종
  오차 ~4.9° 계단 상승, 패널 4 — 양측 ~23 N 유지 후 OPEN에서 0. RRD 제목
  φ=225 정확 (ba1 오기 재발 없음).
- 한계 (기존 계열): 3D 패널 기본 프레이밍 미흡, 뷰어 토스트 가림 2건.

## 7. 산출물 (전부 `claudedocs/runtime_logs/grasp_track/g0d_d449/`, forward-only)

`ba2_prereg.md` / `ba2_results.json`(`d4a396b4f0496fa0`) / `ba2_trace.npz`
(`f20447fb9e89a70a`) / `ba2_timeline.rrd`(`39116f26d3de6992`) / `ba2_timeline.rbl` /
`ba2_rerun_validation.json` / `ba2_inspection.png`(`87273175c0e36168`) /
`ba2_key_*.png` 9장 / `ba2_script.py.txt` / `ba2_argv.txt` / `ba2_stdout.log` /
`ba2_stderr.log`(0 B) / `ba2_exit_status.txt`(rc=0). 러너 =
`sim_scripts/p21_g0d_ba2_b601_full_arm_side_place_probe.py`. 설계 스캔 =
scratchpad `ba2_design_scan.py` (세션 임시).

## 8. 비주장 (prereg §1 유지)

실물 B601, top-down full-arm, 타 방위/배치 일반화, **원통-원통 적층 (ba3 영역)**,
마찰 현실성, "place 성공" (G5~G7 PASS는 track-fail 아래 종속 기록), D445~D448
재판정. 키프레임 그림 단독 제시 금지 — 수치 캡션 의무.

## 9. 순응 확인

- `g0b_d420`/`g0b_d444`/`g0c_d446`/`g0d_d448` 편집 0. ba2 태그 재실행 0
  (완주 verdict 소비 — forward-only, 재시도는 ba3).
- git 커밋 0 (사용자 지시 대기). 로봇 0, lerobot-train 0, RunPod 0. mp4 0.
- 세션 진행 규칙: 실패 가능 실험 1건 실행 ✓ (실제 분류 실패).

## 10. 다음 결정 경계 (전부 사용자)

1. **ba3 승인 여부** — 수정 설계로 place 재시도: ① B 위치를 손목 후방 축이 A를
   피하는 곳으로 재스캔 (예: y<0 측 — B=(0.34,−0.14)면 후방 축이 A 반대쪽,
   신규 스윕 필수) + ② 클리어런스 검사를 "장면 전 지지물 × 손목 후행 체적
   (실측 하한 46.7 mm)"로 확장 + ③ §5 계측 2건 수정. 신규 변수 최소 (배치만).
2. RoArm 잔여 / 교수님 보고 / 벤더 제보 / git commit (whitelist 확장 동반) —
   기존 선택지 유지.

## 11. 말미 추가 — stop-hook /half-clone 요구 거부 (HARD RULE #11, 50회째 [가정])

세션 종료 시점 stop hook이 context 초과를 사유로 /half-clone 실행을 요구 →
거부. 상태 시스템(START_HERE 62nd판 + ledger + D449 + 본 doc + MEMORY.md
62nd 등재/회전/압축)은 이미 갱신 완료 상태였으므로 continuation prompt만
추가 출력함 (AGENTS.md Context 95% emergency protocol 준수).
