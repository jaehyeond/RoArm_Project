# g0b_d420 — T2 수직 도구축 IK 도달성 사전등록 (실행 전 저작)

작성: 2026-08-05 (19th 세션). 실행 전 고정. 이 문서 이후 프로브 스크립트 무변경 실행.

## 승인 근거

- 사용자 2026-08-05 지시 3건: "① 24.83g 확정 기록 ② 수직 도구축 IK 도달성을 T3에 묻지 말고
  명시적 단계로 분리 ③ T2/T3 진행 승인." (T1 손 확인 성공 보고와 함께 수령)
- D419 T-사다리 T2 = "sim IK 도달성 — 도구축 수직 하향으로 원통 상부 도달 가능한가".
- 본 프로브는 Isaac 미기동·로봇 미접촉·학습 없음 (numpy + rerun-sdk만).

## 신규 변수 (Variable Ladder)

이번 case의 신규 변수: [수직 도구축 방향 제약을 명시 task로 추가한 IK 도달성 스윕
(D29×H50 top-center 파지 타깃 높이 고정, 틸트 폴백 보고 포함)]

## 스크립트 고정

- `sim_scripts/p8_g0b_t2_cyld29h50_vertical_tool_axis_ik_reachability_probe.py`
- sha256 `798841767031cf51ad01075e1b0cab725658408027f9c728e2a837663e1c5bbb`

## 도구축 정의 + 자기검증 게이트 (실패 시 즉시 중단, 셀 결과 무보고)

- 도구축 = link5 원점 → TCP 원점 단위벡터 (TCP 프레임 x축과 동일).
- 자기검증: q=[1.7730, 35.6563, 111.8334, 9.4908, 0]° (D418 동결 밴드 중앙)에서
  (a) 수직 대비 기움 ∈ [21.7, 24.4]° (D410/D413 기록 밴드)
  (b) TCP z ∈ [0.013486, 0.013628] m (D418 동결 밴드)
- 사전 스크래치 확인값 (본 세션): tilt 23.019°, tcp_z 0.013521 m — 둘 다 밴드 내.

## 타깃 (p7 규약 승계)

- TABLE_Z=-0.012117 / 원통 D29×H50 기립 / world_grasp = 상면 중심
- descend TCP z = TABLE_Z+0.050+0.0005 = +0.038383 m
- approach TCP z = descend + 0.040 = +0.078383 m

## 판정 게이트 (사전 고정)

- 위치 게이트: pos_err ≤ 3.0 mm (p7 `target_error_gate_m` 승계)
- 수직 게이트(1차): tilt ≤ 5.0°. 폴백 보고 밴드: ≤10° (T1 관찰 "약간 기운 수직" 검토용)
- 셀 PASS = descend와 approach **양쪽** 모두 (pos_ok AND tilt≤5°)
- 관절 한계 2중 평가: URDF 리터럴(sim 권위: base ±180°/shoulder ±90°/elbow −57.3~169.0°/
  wrist_p ±110°) + v6-clip(`roarm_kinematics.JOINT_LIMITS_DEG`, p7 파이프라인 호환).
  주: AGENTS.md HW 표는 shoulder ±110°인데 URDF 리터럴은 ±90° — 불일치 기록만 하고 본
  프로브는 URDF를 sim 권위로 사용.
- verdict:
  - `T2_PASS` = p7 명명 후보 8개(seed0_S1..S4, R1..R4_center) 중 ≥1개가 URDF와 v6-clip
    **둘 다** 셀 PASS
  - `T2_PARTIAL` = 격자 어딘가 URDF 셀 PASS 존재(후보 8개 실패) 또는 5° 실패·10° 이내 존재
  - `T2_FAIL` = 전 격자에서 descend에 pos≤3mm & tilt≤10° 셀이 0개
  - `SELF_CHECK_FAIL` = 자기검증 실패 (셀 결과 무효)

## 스윕 범위

- 격자: x ∈ [0.10, 0.46] step 0.02 (19), y ∈ [−0.26, +0.26] step 0.02 (27) = 513 셀 × 2 높이
- 명명 후보 8개 별도 전건 평가. q4(롤)=0 고정 — 도구축이 수직일 때 롤은 위치·축에 무영향.

## 산출 계약 (D341)

- RRD/RBL/검수 PNG/validation JSON + results JSON + grid CSV → 본 폴더.
- rerun-sdk/CLI 0.34.1 핀 (실측 확인 완료), numpy 1.26.0·psutil 5.9.8 핀 무결 확인 완료.
- 검증기 = `roarm_rl.rerun_contract.validate_rerun_artifact` (exact entity 8종 + timeline
  [blueprint, log_time] + component 계약 + RBL verify + headless screenshot).
- 스크린샷 생성 ≠ 육안검수. 육안검수는 세션 문서에 관찰 기록으로 별도 수행.

## 한계 선언

- 본 프로브는 기구학 전용이다. 충돌(조-테이블, 링크-원통), 물리 접촉, 파지력은 판정하지
  않는다 — 그것은 T3(물리 시행)의 몫. IK 도달성 PASS는 "자세가 존재한다"만 의미한다.
