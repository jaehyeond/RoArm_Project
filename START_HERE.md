# START_HERE.md

Last updated: 2026-08-04 KST (11th session). **사용자 결정으로 타깃 물체 =
젠가형 목재 블록(75×25×15mm, 17.45g) 확정** (커밋명 "젠가로 물체 변경",
HARD RULE #18). **D412: 이 그리퍼의 "두 영역"은 물체 옆면 대향 패치가 아니라
평행한 두 상단 모서리 쌍이며, 지배 변수는 Δh다. "밀림=실패"·"고정 조 먼저"
제약은 전복된다.** case는 아직 미개시. g0a_pass=false 불변.

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
- **D411** (10th doc, 종이): 직육면체로 바꿔도 top-first 불변(납작 360셀 중
  357 top-edge, 완전 수직 접근에서도 동일). 단 전도 임계 μ\* 0.29→**0.83**,
  **Δh 28.94/15.23mm → 0.00~6.35mm**, 법선 36.6~46.3°→**21.8~23.2°** →
  실패 모드가 전도에서 **밀림-포획**으로 전환. yaw 허용 **±10°**.
- **D412** (11th doc, 재해석, NEW): 두 영역의 정의를 스윙 조 운동학에 맞추면
  **평행한 두 상단 모서리 쌍**이 답이다. Δh=0 → 회전 짝힘 0, μ\*=0.83 → 통상
  목재 마찰(0.3~0.6)에서 전도 불가, 접촉선이 CoM보다 7.5mm 위 → 진자 안정.
  **보유력은 병목 아님**: 필요 0.171~0.285N vs D362 실측 peak **43.86N**.
  판정식 재설계 사유 2개(기하 축퇴 = D411 ③ / 밀림≠실패 = D412 ②).
  **"36.033mm = 정확도 하한" 인용 금지** (D328 자유공간 1.512mm).
- 최신 상세: 11th doc
  `claudedocs/session_20260804_grasp_g0a_jenga_two_region_reframe_plan.md`

## Active Case

- **없음 (case 미개시).** D409 종결·소모. Claude 단독 개시 금지.
- **확정 (사용자, 2026-08-04)**: 대상 물체 = 젠가형 목재 블록 75×25×15mm /
  17.45g. 원통 D29×H50(24.83g)은 대조군·역사 증거로만 유지.
- **미확정 (승인 대기)**: ① 배치면(납작 15mm 권고 vs 세로 25mm) ② yaw 랜덤화
  여부 ③ 평면 물체용 판정식 재설계 ④ 후속 case 개시.

## Next Concrete Action

1. **P1 — 로봇 없는 실물 파일럿 (사용자 손 작업, 승인 불필요, ~40분)**
   - P1-a 캘리퍼 실측 75/25/15mm (3면 × 3회)
   - P1-b 경사판 μ **2종**(블록↔테이블 / 블록↔조 표면) 각 5회. 납작 블록은
     전도 임계 기울기 ≈59°라 미끄러짐 우세 → μ = tan(φ) 직독
   - **P1-c 밀기-포획 모의시험 (최중요)**: 고정벽 + 카드를 ~22° 기울여 반대편
     윗모서리 밀기, 초기 간격 3종 × 3회. 평행 미끄러짐(포획) vs 비틀림(yaw
     위험) vs 타고 오름·튕김(실패) vs 압박 후 들어올림 유지
2. **P2 — 자세 1개 선정 (case 개시 승인 필요; Isaac 0 / 로봇 0 / 코드 무변경)**
   동결 D409 재현 게이트 선통과 후: ① **납작 배치 테이블 여유**(10th doc 누락,
   실패 시 납작 무효 → 최우선) ② 접근 corridor 무충돌 ③ **조 면 법선 부호**
   (10th doc 미기재) ④ 자세 (tau, rho, zo) + 예측
   신규 변수 2 = [물체 교체 / 판정식 = 상단 모서리 쌍 대칭 포획]
3. **P3 — 실물 로봇 close-lift-hold (HW 명시 승인 필요)**. 첫 "잡았다" 후보.
4. **교수 확인 3건**: 배치면 지정 / yaw 랜덤화 / 판정식 재설계 승인.

## D408 Frozen Static/Runtime Authority (요약)

- prereg `0c0f1c03…c8d0d`; attestation `fa5a3cf2…50dd`; tuple-file
  `97c7ca51…e1ff`; manual 11/11 true `bf917eb4…af18`; terminal
  `48626366…dd37`. 동결: `claudedocs/runtime_logs/grasp_track/g0a_d408/`.

## Open Risks / Claim Limits

- stable grasp / force closure / 밀어넘김 부재 보증 / 동역학 / SDF 우월성 /
  타 물체·배치 전이: 전부 null. D362/D407 물리 증거는 D34×H90 전용 (D379).
- **D411·D412는 기하·재해석 권위이며 물리 verdict 아님.** μ **미실측**(블록·
  테이블·조 표면) → 전도-미끄럼 판정 전부 μ 의존. 블록 치수는 제품 명목값.
- D412 §4 물체 클래스 술어는 **데이터점 2개 기반 가설** (Δh 임계 높이 미측정).
  push-grasping 문헌 연결은 **미검색·미검증** — 인용 전 HARD RULE #4 절차 필수.
- 상완 링크 메시 부재(d348 = link5+gripper_link 64+64) → 상완 충돌 미지.
- 기하 라벨 단독 학습 승격 금지 (P3). part 마스크 face 수준 구분 불가 (W-FRZ1).
- **D409 실행 이전 작성 브리핑 재사용 금지 without 실행 전/후 구분** (D412 ④).

## Frozen — Do Not Retry or Overwrite

- **D409 attempt1 전체** (20파일 + harness 3파일 sha v2), D400~D408 attempt1,
  D362 33파일, D334 sidecar 수정·재실행 금지.
- target/IK/path, geometry, material/mass/actuator/physics, `isaaclab` env pin
  변경 금지. HANDOFF.md, TASKS.md, `/half-clone`, commit/push 금지.
- d339 canonical 2파일 = 역사적 cook witness (질의 사용 금지 — P1).

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/session_20260804_grasp_g0a_jenga_two_region_reframe_plan.md`
   (11th — 과거 브리핑 대조 §2 / 두 영역 재해석 §3 / 물체 클래스 술어 §4 /
   판정식 전복 §5 / 36mm 정정 §6 / P1-P3 계획 §7; D412 근거)
4. `claudedocs/session_20260804_grasp_g0a_rect_block_target_swap_paper_analysis.md`
   (10th — 신규 타깃 §1 / 납작 §3 / 실패모드 전환 §4 / 세로 §5 / yaw §6 /
   판정식 축퇴 §7; D411 근거)
5. `claudedocs/session_20260804_grasp_g0a_d409_followup_approach_h100_paper_analysis.md`
   (9th — 접근방향 §1 / H100 §2 / 손측정 §5; D410 근거)
6. `claudedocs/session_20260804_grasp_g0a_d409_warning_disposition_static_fixture_attempt1_runtime_complete.md`
   (8th — runtime §3 / manual §4)
7. `claudedocs/DECISIONS.md` D407~D412
8. `claudedocs/EXPERIMENT_LEDGER.md` tail

## Git

- HEAD == `ceb6c98` ("젠가로 물체변경", 사용자 커밋 2026-08-04) — 8th~10th 세션
  산출물(10th doc, attempt1 20파일, sim_scripts 3종, BACKLOG append) 체크포인트됨.
- 미커밋: 11th 세션 추가분 = `START_HERE.md`, `claudedocs/DECISIONS.md`(D412),
  `claudedocs/EXPERIMENT_LEDGER.md`, 11th doc 신규.
- commit/push는 사용자 요청 시에만 (Claude 실행 금지).
