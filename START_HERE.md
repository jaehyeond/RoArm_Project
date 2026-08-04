# START_HERE.md

Last updated: 2026-08-04 KST (15th session). **D416: 사용자가 물체 축을 원통으로 확정(젠가 철회).
외부 AI "프리미너리 테스트 설계" 문서를 5-lens 적대 감사 → 5/5 PARTIAL_ADOPT. sim 원통은 이미
D29×H50이었고, 문서의 "sim-real 해결됨" 진단·novelty 4건·"확보된 것" 4건이 무너졌다.** case 미개시.
`g0a_pass=false` 불변.

## Current Truth

- Pivot: RoArm grasp-track G0a. `q5=0` CLOSED, frozen OPEN `1.5413 rad`. Real-first funnel.
  D407 FAIL-STOP(`g0a_pass=false`), D408 동결 — 불변.
- **D409 attempt1 완주·소모**: A 665 / **B 0 / A∧B 0** / POSES 1239 / PINCH 1146 / FULL 0.
  전 격자 top-rim. **재실행 절대 금지**. `moving_witness_top_margin_mm` 1239/1239 = 0.000(경계 등호),
  barrel 분류기 1개 제거 시 B=787 · **A∧B=505**.
- **D410** top-first = 스윙 조 **구조**의 함수(물체 높이 아님) / **D411** 직육면체도 top-first 불변 /
  **D412** 두 영역 = 평행한 두 상단 모서리 / **D413** 지배 변수 = 조 면 법선 **부호** /
  **D414** 배치면 보류 + 문헌(push-grasping = Dogar/Brost 기성, cross-embodiment = GraspGen-X 반증) /
  **D415** D341은 차단 요건 아님 · collider 64/64 유지 · 배치면 2해.
- **D416** (15th, 외부 문서 감사 + 물체 축 확정, NEW):
  ① **sim 원통은 이미 D29×H50이다** — `cyld29h50_..._worker.py:250-251` `CYL_RADIUS_M=0.0145` /
     `CYL_HEIGHT_M=0.050`, `:297-298`이 D34×H90을 `OLD_..., calibration-only, no D362 physics
     transfer`로 강등. 외부 문서 §6 "치수 변경안"은 **이미 끝난 일의 재제안**. 실제 공백은 **질량·마찰 계약**뿐.
  ② **"sim-real 해결됨"은 4겹으로 붕괴** — D362 실접촉 z=`0.07778545469045639`m = 테이블 위
     **89.90mm = 전체 높이** → r/h **0.189**(문서 0.378의 절반) / `r/h vs μ`는 β=0 특수해(D414 ① supersede) /
     문서 자체 모순 2건 / **D410 정면 충돌 — 마찰·형상비를 고쳐도 첫 접촉은 윗테두리**.
  ③ **novelty 4건 전부 선행연구** — Dex-Net 2.0 = **4-DOF 평면 파지**, ACRONYM/CGN = 팔 없는
     flying gripper, GET(arXiv:2604.26212, 2026-04), ICINCO 2005 cascading filters, GraspGen-X + Eppner.
     살아남는 좁은 형태 1개 = "고정 조 + **단일 호 스윙 조** × **sub-6-DOF 팔**"(확신도 LOW-MEDIUM).
  ④ **§1 "확보된 것" 4건 지위 역전** — 8mm는 D350 legacy proxy(Δ17.027mm) / cooked-hull은 D379
     identity FAIL / 자세 family는 10/10·0/10 실패 / **(7,11)mm은 정렬 standoff이고 파지 flush는 `D/2−8`**.
  ⑤ **물체 = 단일 D29×H50 확정, 접합 H100 폐기**(§Active Case).
  ⑥ **lead 자체 정정**: 직전 턴 "h=50 넣으면 실물도 전도로 역전"은 **틀렸다** — 동결 격자 h를 손
     파일럿에 전이한 것. 올바른 판정 = **"접촉 높이 미기록 → 판정 불가"**.
- 최신 상세: 15th doc
  `claudedocs/session_20260804_grasp_g0a_external_prelim_design_audit_cylinder_target_lock.md`

## Active Case

- **없음 (case 미개시).** Claude 단독 개시 금지. 승인 획득 0건.
- **대상 물체 = 단일 원통 D29×H50 / 24.83g (사용자 명시 "물체는 cylinder로 고정" — HARD RULE #18로
  커밋 `ceb6c98` 젠가 전환 철회).** 젠가 75×25×15는 **G2로 이월** → 배치면 15/25mm 질문은 지금 답 불필요.
- **접합 H100 폐기** — 동결 z에서 그리퍼를 열 수조차 없고(침투 onset 19.53mm), barrel-strict 창이
  dz 0~+70 빈 집합이며, **접합체 실측 43.74g ≠ 24.83g×2 = 49.66g → 타깃 원통 2개가 아니다.**
  쓰려면 같은 dowel 2개 재제작 + 재실측 선행.
- **사용자 관측(2026-08-04, HARD RULE #18)**: ① 원통이 생각보다 잘 집힌다 ② 실리콘 테이프 후 hold 개선
  ③ 젠가 세로·눕힘 둘 다 실패. **원통 기립 여부·테이프 전후는 여전히 판별 불가.**
- **사용자 지시**: "8/20에 매이지 말고 더 빨리 진행".

## Next Concrete Action

1. **확인 1건 최우선** — **원통 성공 조건 3분할**: 서 있었나/누웠나 · 로봇 자율인가 손 보조인가 ·
   **테이프 전인가 후인가**. 테이프 **후**면 D414 ①의 "그리퍼·팔 변경 → 동결 D409 evidence +
   d348(64+64) **전부 무효**, G-ladder 재시작"이 **지금 발동 중일 수 있다** — 기립 여부보다 파급이 크다.
2. **다음 최소 case 후보 C1** (승인 대기): 단일 D29×H50 · 동결 자세 **1개** · PhysX **상대 스크린**으로
   "닫힘 → 5cm 들어올림" **1회**. 신규 변수 1~2개(물리 스텝 재개 + 실물 질량 계약).
   뼈대 = 물리 경로는 **D407 worker**(ContactSensor/AppLauncher), 열거·5-DOF DLS IK·hppfcl BVH
   서명거리·sha 게이트는 **D409 harness S0~S2** 재사용. 변위 임계·5cm·성공 판정은 **사전등록 대상**.
3. 또는 **실물 캠페인 C1**(접근+닫힘+정지 hold) — HW 명시 승인만 받으면 착수 가능. D341은 차단 요건 아님.

## 승인 대기 (8건)

①노선 세부(sim C1 먼저 vs 실물 먼저) ②실물 캠페인 HW 승인(`AGENTS.md:266`) ③물리 재개(prereg +
hash tuple + attestation 3종) ④질량 24.83g·마찰 계약 = `Frozen` 해제 ⑤그리퍼 속도 정책
(`speed=1000` vs G10 `200`) + `gripper_angle_ctrl` G10 구멍 선수리 ⑥판정식 재설계 사전등록(**505셀 공개**)
⑦t(테이프 두께) 실측 [C]→[B] ⑧(접합 사용 시) 같은 dowel 2개 재제작 + 재실측.

## 교수 확인 항목

배치면 지정(G2로 이월) / yaw 랜덤화 / 판정식 재설계 / 젠가 채택이 G2 선취인지 / 테이프 개조가 지시 #8
위반인지 / G0b 트리거를 손·서보 시연으로 인정하는가 / **눕힌 원통을 쓰면 조건절("눕힐 거면 직사각형으로")이
발동하는가**.

## Open Risks / Claim Limits

- stable grasp / force closure / 동역학 / SDF 우월성 / 타 물체·배치 전이: 전부 null.
- **D410~D416은 기하·정정·감사 권위이며 물리 verdict 아님.**
- **n=3으로 지배 변수 순위를 정하지 말 것** — 함께 단조 변하는 양이 최소 5개라 판별력 0.
- **"실물 성공이 D409를 반증한다" 표현 금지** — D409의 B는 첫 접촉 위치 분류기다.
- **판정식 재설계는 505셀 사전등록 없이 착수 금지**(`p_z<top` → `p_z<=top` 한 글자로 전 격자 반전).
- **PhysX는 "상대 스크린" 허용, "절대 라벨" 금지**(`direction:123-129`). 외부 문서 §8의 "실패 라벨
  전체 표가 **학습의 입력이 된다**" 문장은 `direction:125-126` + D409 Implication ④ 위반이므로 **삭제**.
- **접합 반대의 정량 논거 3건 폐기(재인용 금지)**: Δh 2.1배 / 전도각 악화의 물리 verdict / P_crit ≪ 43.86N.
- 미해소 repo 결함 3건: (a) 같은 H50 원통 **Δh 3값 공존**(22.1 / 28.94 / 15.23mm) (b) "D362 실측
  43.86N" 오기가 `DECISIONS.md:24400-24401`·`:24450`·11th doc:82 잔존 (c) 개방각 매핑 3중 충돌
  (88.3° / 1.571rad / 1.5413rad / 86.6°).
- 코드 충돌 1건 미해소: `deploy_smolvla.py:685-689`(speed=1000) vs `safety_p0_guards.py:145-146`(>200 ValueError).
  **`safety_p0_guards`에 `gripper_angle_ctrl` 래퍼 없음 = G10 구멍.**
- 파지 성공 자동 판정기: 접촉 감지만 있음, **hold 판정은 사람**. `roarm_rl` grasp는 전부 kinematic attach라
  물리 파지 증거 불가(단 W2 골격·transport/release 커리큘럼은 이미 존재).
- 문헌 판독 깊이 = abstract/HTML 스니펫. "없다"는 전부 "검색 범위 내 미발견"(LOW-MEDIUM).

## Frozen — Do Not Retry or Overwrite

- **D409 attempt1 전체**(20파일 + harness 3파일 sha v2), D400~D408 attempt1, D362 33파일,
  D334 sidecar 수정·재실행 금지. tolerance 수리 금지(D354/D405). **d348 재분해 금지.**
- target/IK/path, geometry, material/mass/actuator/physics, `isaaclab` env pin 변경 금지.
- HANDOFF.md, TASKS.md, `/half-clone`, commit/push 금지. d339 canonical 질의 금지.

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/session_20260804_grasp_g0a_external_prelim_design_audit_cylinder_target_lock.md` (15th — D416)
4. `claudedocs/session_20260804_grasp_g0a_real_grasp_photos_placement_ambiguity_tape_collider.md` (14th — D415)
5. `claudedocs/session_20260804_grasp_g0a_d409_followup_approach_h100_paper_analysis.md` (9th — 접합 H100 원분석)
6. `claudedocs/session_20260804_grasp_g0a_friction_deferral_normal_sign_placement_reexam.md` (12th — D413)
7. `claudedocs/DECISIONS.md` D407~**D416**
8. `claudedocs/direction_20260708_grasp_pivot.md` (교수 지시 + G-사다리 + 라벨 사다리 3단)

## Git

- HEAD == `e08784b` ("D413", 사용자 push 2026-08-04 22:48 KST). 커밋 메시지는 "D413"이지만
  **내용은 D415까지 전부 포함**(12th·13th·14th doc 3파일 신규).
- 미커밋: **15th 세션분**(START_HERE 갱신, DECISIONS D416, LEDGER 1행, 15th doc 신규).
- commit/push는 사용자 요청 시에만 (Claude 실행 금지).
