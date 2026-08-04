# START_HERE.md

Last updated: 2026-08-04 KST (10th session). **교수님 지시로 타깃 후보가 원통 →
젠가형 직육면체(75×25×15mm, 17.45g 실측)로 제안됨. 종이 분석 결과 D411:
직육면체로 바꿔도 top-first 실패는 그대로(납작 360셀 중 357 top-edge, 완전
수직 접근에서도 동일). 바뀌는 것은 실패의 결과(전도→밀림-포획)와 판정식의
유효성. 물체 교체 자체는 채택 가치 있으나 새 case = 사용자 승인 사안.
g0a_pass=false 불변.**

## Current Truth

- Pivot: RoArm grasp-track G0a. `q5=0` CLOSED, frozen OPEN `1.5413 rad`.
  Real-first funnel. D407 FAIL-STOP(`g0a_pass=false`), D408 동결 — 불변.
- **D409 attempt1 완주·소모** (8th doc): A 665 / **B 0 / A∧B 0** / POSES 1239 /
  PINCH 1146 / FULL 0. 전 격자 top-rim. **재실행 절대 금지 (DECISIONS D409)**.
  evidence `ccc8197b…f16750` / completion `6ce9218c…d638a8` / tuple `de79bc78…efcc9a60`.
  산출물 20파일 = `claudedocs/runtime_logs/grasp_track/g0a_d409/attempt1_.../`
  harness 3파일 sha v2 동결. manual_pass=false (D341: 과학 권위=evidence JSON).
- **D410** (9th doc, 종이): top-first는 TCP z·접근 기울기·물체 높이의 함수가
  아니라 **가동(스윙) 조 구조의 함수**. z 단독 재설계·수직 접근·H100 접합 무효.
- **D411** (10th doc, 종이, NEW): **타깃을 직육면체로 바꿔도 top-first 불변.**
  납작 배치(15mm) 360셀 중 357 top-edge(여유 0.000mm); tool 기울기 34.0°→
  **0.2°(완전 수직)** 에서도 전부 top-edge. 단 전도 임계 μ* 0.29→**0.83**,
  **Δh 28.94/15.23mm → 0.00~6.35mm**, 조 면 법선 36.6~46.3°→**21.8~23.2°**
  → 실패 모드가 전도에서 **밀림-포획**으로 전환. 세로 배치(25mm)에서 **A∧B
  통과 셀 최초 발생**(여유 2.014mm, rho 30mm, dfix 4.95mm, 테이블 8.56mm)이나
  법선 45~68°(눌러 찍기)·전도 임계 μ* 0.33. **yaw 허용 ±10°**(신규 비용).
  lead 재구현은 동결 D409 수치 재현 게이트 선통과(TCP 3축/2.021359845mm/
  0.6097505rad/top-rim witness 일치). **적대 검증 워크플로우는 context 한계로
  미완 — 단일 계산자 권위.**
- 최신 상세: 10th doc
  `claudedocs/session_20260804_grasp_g0a_rect_block_target_swap_paper_analysis.md`

## Active Case

- **없음 (case 미개시).** D409 종결·소모. 신규 타깃(직육면체) 채택 여부와
  후속 case는 **사용자 결정 대기**. Claude 단독 개시 금지.

## Next Concrete Action

1. **(사용자 결정)** 10th doc §9 메뉴: (a) 납작 블록 채택 + **닫힘 동역학**
   case (zero-step 기하 열거 반복은 무가치 — 답이 top-edge로 확정),
   (b) **선행 저비용 실측: 마찰 μ 경사판 손측정 + 캘리퍼 실측** (로봇 불필요,
   전 판정의 분기점), (c) 세로 배치 A∧B 셀 기하 case화(전도 μ* 0.33 리스크),
   (d) 정지.
2. **(교수 확인 3건)** ① 배치면 지정 필요 — 지시에 없음, 세로 세움(75mm)은
   원통보다 나쁨(μ* 0.10). ② yaw 랜덤화 포함 여부(±10° 제약). ③ 평면 물체용
   판정식 재설계 승인(D354와 별건).
3. (사용자 병행) 기울임/전도힘 손측정 — 2nd doc §2. 보고 시 기록만.
4. 차기 attempt 저작 시에만: 관측성 수리 3건 (8th doc §5).

## D408 Frozen Static/Runtime Authority (요약)

- prereg `0c0f1c03…c8d0d`; attestation `fa5a3cf2…50dd`; tuple-file
  `97c7ca51…e1ff`; manual 11/11 true `bf917eb4…af18`; terminal
  `48626366…dd37`. 동결: `claudedocs/runtime_logs/grasp_track/g0a_d408/`.

## Open Risks / Claim Limits

- stable grasp / force closure / 밀어넘김 부재 보증 / 동역학 / SDF 우월성 /
  타 물체·배치 전이: 전부 null. D362/D407 물리 증거는 D34×H90 전용 (D379).
- **D411은 기하 계산 권위이며 물리 verdict 아님.** μ **미실측**(블록·테이블)
  → 전도-미끄럼 판정 전부 μ 의존. 블록 치수는 제품 스펙 명목값(캘리퍼 미실측).
- 상완 링크 메시 부재(d348 = link5+gripper_link 64+64) → 상완 충돌 미지.
- 기하 라벨 단독 학습 승격 금지 (P3). part 마스크 face 수준 구분 불가 (W-FRZ1).

## Frozen — Do Not Retry or Overwrite

- **D409 attempt1 전체** (20파일 + harness 3파일 sha v2), D400~D408 attempt1,
  D362 33파일, D334 sidecar 수정·재실행 금지.
- target/IK/path, geometry, material/mass/actuator/physics, `isaaclab` env pin
  변경 금지. HANDOFF.md, TASKS.md, `/half-clone`, commit/push 금지.
- d339 canonical 2파일 = 역사적 cook witness (질의 사용 금지 — P1).

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/session_20260804_grasp_g0a_rect_block_target_swap_paper_analysis.md`
   (10th — 신규 타깃 §1 / 대조군 §2 / 납작 §3 / 실패모드 전환 §4 / 세로 §5 /
   yaw §6 / 판정식 축퇴 §7 / 한계 §8 / 메뉴 §9; D411 근거)
4. `claudedocs/session_20260804_grasp_g0a_d409_followup_approach_h100_paper_analysis.md`
   (9th — 접근방향 §1 / H100 §2 / 손측정 §5; D410 근거)
5. `claudedocs/session_20260804_grasp_g0a_d409_warning_disposition_static_fixture_attempt1_runtime_complete.md`
   (8th — runtime §3 / manual §4 / 다음 §5)
6. `claudedocs/DECISIONS.md` D407~D411
7. `claudedocs/EXPERIMENT_LEDGER.md` tail

## Git

- HEAD == origin/master == `40ec3ac`. 미커밋: 기존 세션 변경분 + 9th/10th
  세션 추가 (START_HERE, 10th doc 신규, LEDGER/DECISIONS append, attempt1 20파일).
- commit/push는 사용자 요청 시에만.
