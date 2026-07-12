# Backlog

These are deferred ideas. They are not active unless `START_HERE.md` names them
as the current Active Case.

- TCP/EEF dual recording.
- Caliper mm correction before real transfer.
- Render resolution: store 448x448, train/input 224x224.
- Direction diversity / goal-conditioned primitive. See `claudedocs/design_d321_goal_conditioned_primitive.md`.
- Upper friction bin remains an RL target reserve.
- Overshoot 167ep HER relabel.
- Data spec v1.
- VLA training deferred.

- 2026-07-11 (D331): `tool_surface_union` (D231) — 가동 조 `gripper_link`
  collision이 4mm proxy(convexHull, pxr 직접 확인)라 G0a alignment PASS 후에도
  G0b 파지 물리 불성립. G0b 진입 전 필수 collision 재저작 항목. (착수 금지,
  기록만.)

- 2026-07-12 (D335): `target_family_continuous_feasibility` — registered finite
  r/t set에서 raw-clear `0/2629`; 연속 domain 전체 불가능 증명은 아님. 새 물리
  변수 전에 승인된 별도 case로 top basins의 continuous/finer discriminator를
  실행할 수 있음. 현재 착수 금지.
- 2026-07-12 (D335): `target_family_orientation` — finer feasibility도 실패할
  경우 동일 anti-retreat r/t domain에 reachable wrist/tool-orientation 변수 1건을
  추가하는 non-retreat 후속안. D323 strict-axis 재도입 금지; 사용자 case 승인 전
  착수 금지.
- 2026-07-12 (D335): `grasp_depth_semantics_r_gt_17mm` — `r>17mm`는 TCP를 near
  face 밖으로 후퇴시키므로 clearance만 green이 되는 퇴행 가능성. 명시적인 새
  grasp-depth/bracketing 계약과 사용자 승인 없이는 domain 확장 금지.

- 2026-07-12 (D336 감사): `rerun_pipeline_upgrade` — 감사 결과 rerun은 D325 이후
  전 런에서 .rrd를 저장하지만 (12개) ①live viewer 0회 사용 (D326-D330은
  `--live_viewer` 플래그 보유, D332-D336은 False 하드코딩) ②D333 이후 RRD를
  열어본 증거 없음 (ok+nonzero gate만) ③물체가 solid로 추적 안 됨 (frame axes
  마커만) ④접촉력이 TextDocument로 묻힘 (Arrows3D/rr.Scalars 미사용) ⑤10-trial
  중 trial 1만 기록 ⑥D333 400 step 중 target 200만 RRD 수록. 업그레이드 후보:
  물리 런은 `rr.set_sinks(GrpcSink, FileSink)` 하이브리드(현업 표준), 원통
  solid 추적, 힘 화살표, 스칼라 타임시리즈, RRD 열람 확인 의무화. rerun-sdk
  버전 pin 주의 (D326 numpy 사건, .rrd는 인접 minor 호환만). 사용자 승인 전
  착수 금지, 기록만.
