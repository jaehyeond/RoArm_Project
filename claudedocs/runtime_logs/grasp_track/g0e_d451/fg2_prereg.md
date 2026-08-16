# fg2 prereg — g0e_d451: 폭-정지 닫힘 정책 단독 sim 시험 (W-step)

날짜: 2026-08-16 (64th). 실행 전 작성·동결 (forward-only).

## SS0 — 승인 근거

64th 사용자 지시 verbatim: "후보들 중에 어떤것들 순서대로 해야할지 보고
loop돌려서 진행해. step-by-step으로 순차적으로 사고하면서 말이야."
→ Claude가 제시한 순서(①프로포절→②W-step→③G-step→④O-step)에 대한 순차 진행
승인. 본 case = ② W-step. 63rd doc §9-3 및 START_HERE "W-step: 폭-정지 닫힘
정책 단독 sim 시험 (HW 0 변경)"과 동일 항목. HW 변경 0, 로봇 0.

## SS1 — 질문과 분기 (D445 ② 미시험 분기 해소)

D445는 순정 RoArm 조(수렴 쐐기)가 **완전 닫힘(14.0°)** 스윕에서 0/13임을
확정했으나, "폭-정지 닫힘 정책(17~20°)은 미시험 — '어떤 정책으로도 불가'
아님"을 명시 유보했다(D445 ②). 본 probe의 질문:

> **닫힘을 물체 폭 대응각 근처에서 정지·유지(position drive hold)하면, 순정
> 조가 D29×H50 원통을 양측 접촉으로 쥐고 hang에서 유지할 수 있는가?**

분기 (사전 선언):
- (i) **40/40 전패** → 폭-정지로도 sim 유지 불가 → 병목은 정책이 아니라
  접촉면 기하 → G-step 슬리브(기하 수리) 필요성이 정량 강화됨.
- (ii) **1+ SUCCESS** → SW-only 정책으로 sim 유지 가능 → G-step 설계 입력
  (슬리브 없이/with 병행 검토, 실기 검증 우선순위 조정).
어느 분기든 G-step 설계의 입력이며, D427~D450 재판정 없음.

## SS2 — 프로토콜 (fg1 verbatim + 신규 변수 정확히 1개)

- 기반: `sim_scripts/p17_*.py` (fg1, 58th) 프로토콜 verbatim — 동일 stage
  구성(support/object/materials/physics scene), 동일 flying-gripper(팔 제거,
  fixed-root articulation), 동일 phase 구조 PREGRASP(개방 88.30998°, 60 step)
  → CLOSE(120 step) → HANG(support collider off, 240 step, 30-step 청크 =
  fg1 DEV-4 승계), dt 1/60, 동일 마찰(물체 0.40/0.30, 지지면 1.0/1.0), 동일
  질량 0.02483 kg, 동일 GraspingManager default-scene 모드(fg1 DEV-2 승계).
- **신규 변수 (정확히 1개): side CLOSE 목표각.** fg1의 14.0°(완전 닫힘,
  D431(6) 최심값) 대신 **폭-정지 목표각 {23, 21, 19, 17, 15}°** 5레벨.
  CLOSE 종료 후 drive 목표가 정지각에 남아 HANG 동안 position hold = 폭-정지
  정책의 sim 구현. 정지각 스윕 사유: 조-물체 접촉각(폭 대응각)의 정밀값이
  미지(fg1 증거로 ~14-22° 대역 추정, D431(6) 대역 14-22°)이므로 단일값
  선택은 자의적 — 대역 상부를 괄호치는 5레벨 스윕이 견고.
- **pose = fg1 side 8 (sdg2 candidates verbatim)** × 5 정지각 = **40 평가**.
  **rim 5행 제외** — 사유: rim 실패 기전은 닫힘 스윕 중 접촉 소멸(D445,
  θ15 순간 0.250 N 외 접촉 0 step)이고 rim close 목표는 이미 stop-short
  (q5_row−2°)였으므로 폭-정지 변수와 무관. rim 제외는 프로토콜 축소이지
  변수 추가가 아님.
- 정지각 그룹 순서: 23 → 21 → 19 → 17 → 15 (개방→닫힘 순), 그룹당 side 8.

## SS3 — 자산·핀 (전부 기존 동결 자산, 신규 기하 0)

- 그리퍼: `g0b_d444/fg1_gripper_only.usd`를 **in-place 참조** (동결 폴더
  read-only 사용, 편집 0; SHA-256 `0e9f…dd76` 핀). stage측 루트만
  `/World/fg2_gripper`로 명명.
- attempt3 원본 5 USD 핀 + `isaacsim.replicator.grasping` extension.toml 핀 +
  `t3s_side_sdg2_candidates.json` 핀 = p17 PINS에서 소비분만 승계 (n8b/n8
  rim 핀은 rim 제외로 소비 없음 → 제외).
- env 핀: numpy 1.26.0 / psutil 5.9.8 / rerun 0.34.1 (실행 전 확인 완료).
- 자산 게이트 (a)(b)(c) fg1 verbatim: hull census 64+1/64+1, q5 joint 15속성
  bit-일치, inline mesh SHA 일치.

## SS4 — 게이트 (fg1 verbatim)

- SUCCESS = 같은 step 양측 접촉력 > 0.01 N (CLOSE 중) AND hang 낙하 < 6 mm.
- taxonomy: SUCCESS / BILATERAL_NO_HOLD / PRECLOSE_COLLISION / ONE_JAW_ONLY_* /
  NO_JAW_CONTACT (폭-정지가 접촉각보다 개방이면 NO_JAW_CONTACT 예상 — 그
  자체가 접촉각 위치 정보).
- measurement_valid = root 도달 오차 <1e-6 m AND link5-root 일치 <1e-6 m AND
  hang 중 root 드리프트 <1e-6 m AND spawn 드리프트 <2e-3 m.
- verdict 코드: 전패 → `FG2_ALL_40_FAIL_WIDTH_STOP_INSUFFICIENT_SIM`,
  1+ 성공 → `FG2_WIDTH_STOP_SOME_HOLD_SW_POLICY_VIABLE_SIM`.

## SS5 — D341 계약

save-only RecordingStream(0.34.1) + footer, 고정 blueprint 임베드 + .rbl,
`validate_rerun_artifact` exact entity/timeline/component 계약, headless
inspection.png, 실행 후 육안 검수 기록. 권위 = fg2_results.json +
fg2_trace.npz (RRD는 검수 증거, D341).

## SS6 — non-claims

실로봇 파지·서보 토크/전류 거동·마찰 현실성·IK 도달성·rim 일반화·슬리브
설계 수치에 대한 주장 없음. sim 필요조건 판별만. D419 격리 유지.

## 산출물 (전부 `g0e_d451/fg2_*`)

results.json / trace.npz / timeline.rrd / timeline.rbl / rerun_validation.json /
inspection.png / stdout.log / exit_status.txt / script.py.txt / argv.txt
(+ 실패 시 failure.json). 러너 = `sim_scripts/p22_g0e_fg2_cyld29h50_width_stop_close_probe.py`.
