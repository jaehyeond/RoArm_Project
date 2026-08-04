# START_HERE.md

Last updated: 2026-08-04 KST (14th session). **D415: 사용자 실물 사진 10장 + 관측 3건 수령. D341은 실물
시행의 차단 요건이 아니었고(조문 오인용), 배치면 지정 문장은 여전히 2해로 갈리며, collider 64/64는
유지가 정답이다. 적대 검증 4/4가 lead 1차 종합을 반박.** case 미개시. g0a_pass=false 불변.

## Current Truth

- Pivot: RoArm grasp-track G0a. `q5=0` CLOSED, frozen OPEN `1.5413 rad`.
  Real-first funnel. D407 FAIL-STOP(`g0a_pass=false`), D408 동결 — 불변.
- **D409 attempt1 완주·소모**: A 665 / **B 0 / A∧B 0** / POSES 1239 / PINCH 1146 /
  FULL 0. 전 격자 top-rim. **재실행 절대 금지**. evidence `ccc8197b…f16750`.
- **D410** top-first는 스윙 조 구조의 함수 / **D411** 직육면체도 top-first 불변 /
  **D412** 두 영역 = 평행한 두 상단 모서리 / **D413** 지배 변수 = 조 면 법선 부호 /
  **D414** 배치면 판정 보류 + 문헌(push-grasping = Dogar/Brost 기성, cross-embodiment = GraspGen-X 반증).
- **D415** (14th, 사진+감사+정정, NEW):
  ① **D341은 실물 파지의 차단 요건이 아니다** — `AGENTS.md:120-121`에 면제 문언이 실재하므로
     "면제 경로 없음"은 조문 오인용이고, `:142-144` "contract fails **without overriding the
     scientific verdict**" + D409 manual 4/11 정직 FAIL 선례(`DECISIONS.md:24373-24375`)로
     **"시행 → 실패 정직 기록 → verdict 유지"** 제3 경로가 이미 존재한다.
  ② **배치면은 여전히 2해** — A(높이15/radial75/조25 = 납작 yaw0) vs B(높이25/radial75/조15 = 세로 yaw0).
     "1.5를 바닥으로"가 양쪽으로 읽히고 사용자는 둘 다 실패라 했다. **D414 ① 보류 미해소.**
     "75mm radial = yaw 0"만 2경로 교차확인 확정. 필요한 답 = **높이 숫자 1개**.
  ③ **collider 64/64 유지가 정답, 재분해는 해롭다** — 프로브가 d348 분해 파트를 BVHModelOBBRSS로 썼고
     (`worker:1050-1058`,`:734-744`,`:1065-1069`) 재분해 시 `part_027/029/030/031`+inner17 동결 참조가 붕괴.
     단 sha 통과 ≠ 실물 대표성(**false pass**). 경계: 실물 시연 진행 가능 / **mm 여유 주장은 t 실측 후**.
  ④ **D414 ② "여유 전량 소모"는 과했다** — "측면 엄격 여유"는 수직량(`worker:909` z만).
     `Δz = R(cosθ−cosθ′)` = **0.336~0.388mm** = 17~30% 소모·양수 생존. 신규 비용: **yaw 허용 ±10° → ±6~9°**.
  ⑤ **테이프**: 24mm는 **폭** 확정(두께면 48mm > 개구 40~45mm로 안 닫힘). 두께 명목 **0.5mm(20 mil)** —
     동급 5출처 수렴 **추정**(OKONG 미공표). 자기융착 = **접착제 없음**(부착 형태 확인 필요).
  ⑥ **lead 정정 3**: `roarm_sdk/common.py` `tB~tR` **주석 처리**(힘 API 구조적 부재 아님) /
     hold 판정은 **사람**이고 `drift_check`는 유효성 필터(극성 반대) /
     `safety_p0_guards`에 **`gripper_angle_ctrl` 래퍼 없음 = G10 구멍**.
  ⑦ **D409 정밀화**: `moving_witness_top_margin_mm` 1239/1239 = **0.000**(경계 등호),
     barrel 분류기 1개만 빼면 **B=787 · A∧B=505**, `worker:1187`로 **z 고정** 2축 격자였다.
- 최신 상세: 14th doc
  `claudedocs/session_20260804_grasp_g0a_real_grasp_photos_placement_ambiguity_tape_collider.md`

## Active Case

- **없음 (case 미개시).** Claude 단독 개시 금지. 승인 획득 0건.
- **사용자 관측 (2026-08-04, HARD RULE #18)**: ① 원통이 생각보다 잘 집힌다 ② 실리콘 부착 후 hold 개선
  ③ 젠가 세로(2.5cm)·눕힘(1.5cm) 둘 다 실패 ④ 정면 배치 = "1.5를 바닥, 2.5×1.5 면이 로봇을 향해".
- **사진 10장 판독**: 원통 barrel 파지 6장 / **빨간 직육면체 막대** 4장(정체·치수 미확인, 최소 2장은
  손이 안 닿음 = 서보 유지 추정). **테이프 부착 여부·원통 기립 여부는 판별 불가.**
- 대상 물체 = 젠가형 목재 블록 75×25×15mm / 17.45g (커밋 `ceb6c98`). 원통 D29×H50은
  `direction:17`상 **G0b 트리거 물체** — `START_HERE` 과거 "대조군" 강등은 사다리와 충돌(D415 ⑦).
- **사용자 지시**: "8/20에 매이지 말고 **더 빨리** 진행".

## Next Concrete Action

1. **확인 6건**(14th doc §8) — 최우선 2건:
   ① **테이블에서 블록 윗면까지 높이가 15mm인가 25mm인가** (자로 1초, 배치면 확정)
   ④ **원통 성공이 서 있었나 누웠나 / 로봇 자율인가 손 보조인가 / 테이프 전인가 후인가**
      (전이면 D413 ② 쐐기 모델이 원통에 대해 내는 문턱 μ>0.74~1.05가 실물로 반증)
   나머지: 실패 시 75mm 축 방향 / 실패 양상 / 빨간 막대 정체·치수 / 테이프 부착 형태.
2. **승인 6건**(14th doc §9): ①노선 지정 ②실물 캠페인 C1 개시(HW 승인) ③그리퍼 속도 정책
   (`speed=1000` vs G10 `200`) ④t 실측([C]→[B]) ⑤D341 처리 ⑥판정식 재설계 사전등록(**505셀 공개 필수**).
3. 승인 1·2를 받으면 **오늘 착수 가능** — D341은 차단 요건이 아니다(D415 ①).
   신규 스크립트는 `safety_p0_guards`를 뼈대로, `replay_sim_demo_real.py:219-320`에서 **루프 구조만** 이식
   (두 안전 체계 혼용 금지). §10의 `gripper_angle_ctrl` 구멍을 먼저 막을 것.

## 교수 확인 항목 (5건 유지 + 1건 신규)

배치면 지정 / yaw 랜덤화 / 판정식 재설계 / 젠가 채택이 G2 선취인지 / 테이프 개조가 지시 #8 위반인지
/ **NEW: G0b 트리거를 손·서보 시연으로 인정하는가, 자율 명령 파지를 요구하는가**(`direction:17`은 자율성 미명시).

## Open Risks / Claim Limits

- stable grasp / force closure / 동역학 / SDF 우월성 / 타 물체·배치 전이: 전부 null.
- **D410~D415는 기하·정정·재해석·감사 권위이며 물리 verdict 아님.**
- **n=3(성공 1·실패 2)로 지배 변수 순위를 정하지 말 것** — 함께 단조 변하는 양이 최소 5개
  (상단 높이 37.883/12.883/2.883mm · μ\* · Δh · 곡면vs평면 · 회전대칭)라 판별력 0.
- **"실물 성공이 D409를 반증한다" 표현 금지** — D409의 B는 성공 판정식이 아니라 첫 접촉 위치 분류기다.
- **판정식 재설계는 505셀 사전등록 없이 착수 금지**(`p_z<top` → `p_z<=top` 한 글자로 전 격자 반전 = D354 전형).
- 코드 충돌 1건 미해소: `deploy_smolvla.py:685-689`(speed=1000) vs `safety_p0_guards.py:145-146`(>200 ValueError).
  **추가 위험**: `safety_p0_guards`에 `gripper_angle_ctrl` 래퍼가 없어 직접 호출 시 G10 미발동.
- 파지 성공 자동 판정기: **접촉 감지만 있음**(`trajectory_p0_gripper_sweep.py:73` gap), **hold 판정은 사람**.
  실물 영상 녹화 코드 없음(1프레임 PNG 유틸 `capture_kinect_for_sponge_check.py`는 있으나 D341 스크린샷 대체 불가).
- 납작 배치 **테이블 여유는 여전히 미계산**(11th doc:139 "최우선"). 세로 3셀 법선 **부호 미기재**.
- 문헌 부재 주장(스윙 호 조용 데이터셋 / sub-6-DOF 도달성 1급 제약)은 확신도 **LOW-MEDIUM**, 인용 전 재검색.
- 문서 결함 2건(미수리): "세로" 용어 충돌(75mm 세움 μ\*=0.10 vs 25mm 세로 0.326) /
  "D362 실측" 오기가 `DECISIONS.md:24401`, 10th:65, 9th:53·59, `LEDGER:460`에 잔존.

## Frozen — Do Not Retry or Overwrite

- **D409 attempt1 전체** (20파일 + harness 3파일 sha v2), D400~D408 attempt1,
  D362 33파일, D334 sidecar 수정·재실행 금지. tolerance 수리 금지(D354/D405).
- **d348 재분해 금지** — 파트 번호 재배열이 D409 evidence 참조를 전부 무효화한다(D415 ③).
- target/IK/path, geometry, material/mass/actuator/physics, `isaaclab` env pin 변경 금지.
- HANDOFF.md, TASKS.md, `/half-clone`, commit/push 금지. d339 canonical 질의 금지.

## Must Read First

1. `AGENTS.md`
2. this file
3. `claudedocs/session_20260804_grasp_g0a_real_grasp_photos_placement_ambiguity_tape_collider.md` (14th — D415)
4. `claudedocs/session_20260804_grasp_g0a_placement_audit_literature_verification_photo_pending.md` (13th — D414)
5. `claudedocs/session_20260804_grasp_g0a_friction_deferral_normal_sign_placement_reexam.md` (12th — D413)
6. `claudedocs/session_20260804_grasp_g0a_rect_block_target_swap_paper_analysis.md` (10th — 배치면 원수치)
7. `claudedocs/DECISIONS.md` D407~**D415**
8. `claudedocs/direction_20260708_grasp_pivot.md` (교수 지시 + G-사다리 원문)

## Git

- HEAD == `149965e` ("검증후 수정", 사용자 커밋 2026-08-04 20:17 KST — 11th 세션분 포함).
  (이전 START_HERE의 `ceb6c98` 표기는 오기였음 — 14th에서 정정.)
- 미커밋: 12th·13th·**14th 세션분**(START_HERE 갱신, DECISIONS D415, LEDGER, 14th doc 신규).
- commit/push는 사용자 요청 시에만 (Claude 실행 금지).
