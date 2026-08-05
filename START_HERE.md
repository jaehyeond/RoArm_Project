# START_HERE.md

Last updated: 2026-08-05 KST (19th 세션, context 비상 종료). **G0b 개시 — case `g0b_d420` 가동.**
교수 지시(D419: 포인트 열거 기각 → top-down 1자세 + 물리 시행)는 확정 유지.
사용자 T1 성공 보고 + 승인 3건 수령("24.83g 확정" / "수직 IK 명시 단계 분리" / "**T2·T3 진행 승인**").

## ⚡ T2 결과 — **T2_PASS** (세션 내 완주·육안검수 완료, D421)

- 격자 513셀: **URDF 272 / v6-clip 256 통과** (pos≤3mm ∧ tilt≤5°, 양 높이).
- **p7 후보 4/8 PASS(양 한계)**: seed0_S1·seed0_S2·R1_center·R2_center (descend tilt
  0.20~0.36°). FAIL 4 = 외곽 반경 R4(16.1°)/R3(19.3°)/S3(19.4°)/S4(32.8°) — pos는 전부
  3mm 내(위치는 닿되 수직 불가, D323 동형). **도달 영역 = 베이스 근측 annulus**
  (외곽 경계 r≈0.30~0.38, 정확 경계 = `g0b_d420/t2_ik_grid.csv`).
- **귀결**: T1의 "기운 수직" 폴백 불필요. **T3 스폰 권고 = seed0_S1**
  (q_descend [−42.49, 41.41, 78.61, 59.79]°). Rerun 검증 pass + PNG 육안검수 기록 완료
  (19th doc §4). 부속 **T2b**(+12.117mm 실높이) = p9 사전등록 전 확인 항목.

## Active Case — `g0b_d420` (2026-08-05 개시)

- 승인 근거 = 사용자 명시 "3. T2/T3 진행 승인" (T1 보고 + 지시 3건 동반, 19th).
- 범위: **T2** 수직 도구축 IK 도달성(사전등록 `g0b_d420/t2_prereg.md`, probe sha `79884176…c5bbb`)
  → **T3** p7→p9 전환 물리 파지(개시 전제 ③ prereg/hash/attestation + ④ 질량·마찰 계약).
- 출력 폴더: `claudedocs/runtime_logs/grasp_track/g0b_d420/` — t2_prereg / t2_ik_stdout.log /
  **t3_mass_friction_contract.md(④ 저작 완료)** / **t3_conversion_design.md(델타 전수)**
- 이번 case의 신규 변수: [① 수직 도구축 제약 IK 스윕, ② D29×H50 물리 시행 전환(질량·마찰 계약 포함)]
- **물체 = 단일 원통 D29×H50 / 24.83g(50g 분동 교정 저울 실측 확정 — D410 ④ P0 해소), 기립** (HARD RULE #18).
- **파지 = 수직 상부 접근 top-down 상면 중심** (D419 고정, 대안 탐색 금지). 장기말 = **비유 확정**(물체 불변).
- **목표 체인**: grasp → hold → move → place → 하나의 연속 task.

## T1 판정 (D420 — 층위 분리로만 인용)

- 사진 3장 = **8/04 21:01~21:02 촬영(EXIF), D415/D418 감사 10장 배치의 재제출** — 신규 시행 아님.
- **사진 실증**: top-down 상대 기하 형상 적합 / **조 팁 rim 핀치 상면 0~12mm**(throat 아님) /
  공중 유지 시 목재에 손 비접촉(하중 경로 = 조 2판). 흰 배경 = 수평면(벽 기각).
- **증언 층위**(HARD RULE #18 수용, 등급 표기): 기립 수직 접근 · 무전도 · 조-테이블 비간섭.
- **기각**: "throat 진입 15~20mm"(패널 0~12mm) / **"그랩력이 24.83g을 이김" 증명**(손가락이
  가동 조 판 접촉 — 폐지력 주체 불명). **"T1이 파지력을 증명했다" 표현 금지.**
- 해소 2건: 맨 그리퍼 확정(D418 ⑤ 모순 종결, D414 ① 미발동 무조건) / 실리콘 부착 = BACKLOG(실기 전이 시점).

## T3 착수 조건 (D420 — p7 재실행 금지, p9 신규 저작)

`t3_conversion_design.md` D-1~D-8 전부 사전등록 대상. 치명 3건:
1. **q5 규약 반전** — 동결 트랙(d337/d409): **1.5413rad=열림, 감소=닫힘** vs p7/env 반대.
   5/20 cube LATCH_FAIL("one_sided_push")의 유력 설명 후보.
2. **marker 재정의** — env `_grasped`(dist<0.025)는 H50 top 파지에서 구조적 발화 불가(0.0255m).
3. **충돌체 = 동결 attempt3 자산 재사용**(D420-R1, 재분해 금지 D415 ③) — 로컬 USD는
   convex_hull 1개/링크(목구멍 폐색)라 무효. `g0a_d344/collision_asset/attempt3/
   roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd`를 `ROARM_M3_USD_PATH`로 지정
   (import 전!) + sha 핀 + 첫 step 전 64+64 스테이지 감사. 상세 = 설계 doc D-3 개정판.
부수: env 기본 USD 경로 = B200 경로(#27) → `ROARM_M3_USD_PATH` export + 가드 / 마찰은 계약
leg만(μs0.40/μd0.30, 감도 0.25/0.60) / D341 스텝 타임라인 로깅 신설 / episode 20s.

## T 사다리 현황

| 단계 | 상태 |
|---|---|
| T0 격리 / T1 실물 손 확인 | **완료** (T1 = 층위 분리 verdict, D420) |
| **T2 수직 IK** | **완료 — T2_PASS (D421)**. 부속 T2b(실높이 +12.117mm) = p9 prereg 전 확인 |
| **T3 sim 물리 파지** | 설계 완료(D-1~D-8), **T2b → p9 저작 → ③ 발행 → Isaac 실행** (승인 기수령, 스폰 권고 seed0_S1) |
| T4 실물 재현 | 대기. 선행 확인: 실물 SDK 그리퍼 방향 매핑(d322 vs env 주석 충돌) + 물림 깊이 실측 |
| T5 hold→move→place / T6 격자 / T7 RL | 대기 |

## Open Risks / Claim Limits

- 서보 폐지력·자율 재현성 = **여전히 null** (T1은 손 시행). `g0a_pass=false` 불변.
- T2는 **기구학 전용**(충돌·조-테이블 간섭·물리 미판정 — prereg 한계 선언). 물리는 T3의 몫.
- 재인용 금지 4건 유지: "SmolVLA 74ep 100%" / "SmolVLA+LoRA" / "hand-eye 2cm"(정본 RMSE
  10.13mm) / "D417 ③ 셀 여유+yaw"를 원통 비용으로. "실물 성공이 D409 반증" 표현 금지.
- 코드 충돌 1건 미해소: `deploy_smolvla.py:685-689` vs `safety_p0_guards.py:145-146` (T4 전 수리).
- 문헌 인용 의무: push-grasping = Dogar 2010/2011·Brost 1988 / GraspGen-X `arXiv:2606.00998` /
  KITE `arXiv:2606.22113`.

## Frozen — Do Not Retry or Overwrite

- **격리 트랙 전체**(`claudedocs/QUARANTINE_point_enumeration_track.md`) — D409 attempt1 +
  harness sha v2, D400~D408, D362 33파일, d348 재분해, tolerance 수리(D354/D405).
  파일 이동·이름변경 금지. 재개는 사용자 호출 시에만.
- HANDOFF.md, TASKS.md, **`/half-clone`(거부 5회째 — stop-hook 제안 포함)**, commit/push 금지.
  d339 canonical 질의 금지. `isaaclab` env pin(numpy 1.26.0/psutil 5.9.8 — 19th 실측 무결).

## Must Read First

1. `AGENTS.md` → 2. this file → 3. `claudedocs/DECISIONS.md` **D421**(최신) → D420-R1 → D420 → D419
4. `claudedocs/runtime_logs/grasp_track/g0b_d420/` 4파일 (t2_prereg / t3_mass_friction_contract /
   t3_conversion_design / t2_ik_stdout.log)
5. `claudedocs/session_20260805_g0b_t1_audit_t2_vertical_ik_t3_design.md` (19th)
6. `sim_scripts/p8_g0b_t2_cyld29h50_vertical_tool_axis_ik_reachability_probe.py` (T2 프로브)
7. `sim_scripts/p7_branch_b_cube2cm_local_grasp_close_sweep_probe.py` (T3 골격 원본)

## Git

- HEAD == `c1b7679` ("15th, D416"). 미커밋: 16th~19th 세션분 — 이 파일, DECISIONS(D417~**D421**),
  LEDGER 5행, session doc 4건, QUARANTINE, `g0b_d420/` 폴더, p8 프로브, BACKLOG 추가분.
- commit/push는 사용자 요청 시에만.
