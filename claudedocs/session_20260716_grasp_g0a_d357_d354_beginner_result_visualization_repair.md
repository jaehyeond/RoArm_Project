# Session 2026-07-16 - D357 D354 beginner-readable result visualization repair

## 1. What and why

D354의 실제 Isaac 실행은 성공했지만 Viewer PNG는 OPEN과 raw last-clear만 보존했고,
첫 overlap을 같은 카메라로 남기지 않았다. D355 failure-only Rerun은 글자 겹침,
잘림, 실제 로봇 부재 때문에 사용자 설명용으로 부적합했다. D357은 immutable D354
결과를 다시 판정하지 않고 초보자가 한눈에 구분할 수 있는 표시 증거만 만든다.

`이번 case의 신규 변수: [d354_beginner_result_visualization_contract]`

`new physical variables: []`

Output: `claudedocs/runtime_logs/grasp_track/g0a_d357/`

Harness: `sim_scripts/cyl34_top_view_d357_d354_beginner_result_visualization_repair.py`

## 2. Frozen authority

- Base Git: `HEAD == origin/master ==
  161f6d9d185bb41eb29259349ee0fd897a3c6de8`.
- D354 measurement SHA-256:
  `fd0d43c2a47abefad939fc0e980456cc396bbf3ba3bb104b28bcd999100f23ed`.
- D354 zero-step attestation SHA-256:
  `1975df11b13a774b89f953991d7fdac6e38d795e81a9535850b147d823740d20`.
- D354 completion SHA-256:
  `5cc70c8aa1e50532fa4ec27756496d6b9f9447156c56ef700084b44c16226f86`.
- Frozen display q5:
  - OPEN `1.5413000583648682rad`
  - last-clear `1.0269782543182373rad`
  - first-overlap `1.0269775390625rad`
- Frozen raw distance: `+0.0010050812803802547mm` ->
  `-0.000988475720559677mm`.
- Frozen live distance: `+0.0010049780471806762mm` ->
  `-0.0009864198978583663mm`.
- Frozen clear/overlap cylinder-local z difference:
  `0.0003806054099525502mm`.

These values are copied from D354 JSON. D357 does not call a distance/contact
query or classifier and cannot change `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`.

## 3. Registered execution order

1. `--stage prepare` verifies Git base, frozen hashes, pins, font, harness/session,
   forward-only output, and records one registered GUI command.
2. `--stage run` creates exactly one supervised `_worker`; inactivity timeout
   `120s`, total timeout `300s`, no automatic retry.
3. Worker preflight runs before `AppLauncher`. Then `headless=false`, `DISPLAY=:1`,
   `cuda:0` creates `SimulationApp`; Omni/PXR imports occur only after that.
4. Build the same frozen D354 environment and reset once. Apply the inherited
   conditional `timeline.pause()` -> `Timeline.commit()` bridge only if PAUSE is
   still pending. No next-frame or physics step is used.
5. Bind the worker to the one exclusive invocation marker with a one-time nonce
   and supervisor PID. Trap D351 evaluator, first-contact certification,
   classification, overlap/contact audit, `SimulationContext.step()`, and the
   explicit physics-step helper; attempted calls are measured rather than merely
   reported as zero.
6. Direct-write exactly three frozen full display-state reassertions (q0-q4 and
   object state remain bit-exact; only q5 differs) in order `OPEN -> last-clear ->
   first-overlap`. Each state uses identical side camera
   `eye=[0.285,-0.42,0.09]`, `target=[0.285,0,0.055]`, actual PhysX collider debug
   view, and one explicitly fixed and verified `1280x720` viewport capture.
7. Close Isaac normally. Only after process close, decode/hash all PNGs, build one
   Korean three-pose sheet, separate actual-scale and `Z x50000` display-only
   cross-sections, and write a minimal D357 RRD/RBL plus headless screenshot.
   The screenshot uses an isolated Rerun `--port auto` so the user's persistent
   Viewer on the default port remains running and cannot steal the render. A
   D357-only fixed eye/target blueprint shows the actual robot and cylinder,
   hides the duplicate commanded robot, and uses ASCII event text because the
   Rerun 0.34.1 embedded font does not render Korean glyphs reliably. The exact
   Rerun inventory is preregistered as 62 non-system entities with path-list
   SHA-256 `2a872536c08c44ed9ba00a82f82ad72f4109039bb86df521eed0c92857866ef2`
   and timelines `[blueprint, log_time, step]`.
8. Open all images at original resolution. Manual PASS must check actual robot and
   cylinder visibility, label non-overlap, no clipping/missing glyph, exact three
   roles, actual/magnified separation, and the explicit no-force/no-grasp warning.
9. `--stage finalize` may run only after the manual record exists. It creates the
   completion summary without another Isaac process.

## 4. Exact counters and guards

- Isaac visualization invocation: exactly `1`.
- Display-only frozen full-state write attempts/successes: exactly `3/3`; only
  q5 differs between the three states.
- q5 science evaluator invocation: `0`.
- raw/live distance query: `0`.
- overlap/contact query: `0`.
- new cap/rim classification: `0`.
- controlled physics steps: `0`.
- target/IK/path change: `0`.
- q0-q4 and object pose/velocity must equal frozen Float32 bits at every capture.
- custom counter, timeline time, and SimulationContext time/index must not advance.
- joint/object Float32 bits must also remain unchanged across the pause/commit
  bridge before the first display-state write.
- `simulation_app.update()` is UI/render pumping only; any observed PLAY/clock
  advance fails closed.
- The original `/app/player/playSimulations` and PhysX collider-display settings
  are restored exactly even on exception.

## 5. Visual deliverables

- `d357_open_same_camera_actual_physx.png`
- `d357_last_clear_same_camera_actual_physx.png`
- `d357_first_overlap_same_camera_actual_physx.png`
- `d357_three_pose_beginner_sheet_ko.png`
- `d357_contact_section_actual_scale_ko.png`
- `d357_contact_section_z50000_display_only_ko.png`
- `d357_d354_result_visualization.rrd`
- `d357_d354_result_visualization.rbl`
- `d357_d354_result_visualization_rerun.png`

The main Korean sheet, not Rerun, is the beginner-facing authority for visual
communication. Rerun is a replay/contract companion and never feeds a hash or
geometry gate.

## 6. Frozen prohibitions

- No D354/D355 rerun or overwrite.
- No new geometry/contact/cap-rim result.
- No asset/decomposition/gate/tolerance/material/mass/actuator/renderer/solver/
  physics setting change.
- No target/IK/path, settle, ten-trial, two-jaw grasp, hold/lift, G0b, RL/PPO,
  VLA, or ladder promotion.
- No D334 sidecar write, hardware, B200, commit, or push.
- A failed single invocation is preserved; the same D357 path is not retried.

## 7. Session-progress justification

This is a reactive observability repair caused by the observed D354 missing-overlap
capture and D355 unreadable Rerun. The single visualization contract can fail on
no-advance, capture, exact inventory, RRD, Korean glyph, clipping, or manual
readability checks. It deliberately cannot change the frozen geometry decision.

## 8. Preregistration status

아직 prepare/Isaac 실행 전이다. 실행 결과는 이 아래에 append한다.

## 9. Reactive manual-inspection addendum preregistration

원본 해상도 수동 검사에서 세 실제 Isaac 캡처 모두 움직이는 죠가 원통 뒤에
가려져 세 상태가 거의 같은 그림으로 보임을 확인했다. 따라서 기존 PNG/RRD를
덮어쓰거나 Isaac을 재실행하지 않고, 이 관찰 한계 자체를 초보자에게 보여 주는
forward-only 보충 그림 1개만 추가한다.

- 신규 변수: `[기존 캡처의 카메라 가림 설명 레이아웃 1개]`
- 입력: 기존 D357 `first_overlap` Isaac PNG와 기존 D357 Rerun screenshot만 읽기
- 출력:
  `claudedocs/runtime_logs/grasp_track/g0a_d357/d357_isaac_camera_occlusion_addendum_ko.png`
  및 대응 summary JSON
- 현재 harness SHA-256:
  `5f87dfb5a3ee9a4f36b8ad00399ed3df8d60196ddcad73686f7f00343a8100ea`
- 명령: `DISPLAY=:1 /home/cgxr/miniconda3/envs/isaaclab/bin/python -B
  sim_scripts/cyl34_top_view_d357_d354_beginner_result_visualization_repair.py
  --stage addendum`
- 정확한 금지: 새 Isaac/Kit/PhysX 실행, q5 query, physics step, 새 contact/cap-rim
  분류, 원 파일 overwrite, D354 판정 변경은 모두 0회
- 성공 기준: 새 PNG가 `2400x1500 RGB`로 decode되고, 원 invocation marker는
  여전히 1이며 D354 frozen hash와 `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`이
  그대로여야 한다.

이 보충은 과학 판정이 아니라, “실제 Isaac 캡처에서 무엇이 보이고 무엇이
가려졌는가”를 정직하게 설명하는 관찰성 수리다.

## 10. 실행 결과 — 정상 완료

등록한 Isaac GUI invocation은 정확히 1회였고 정상 종료했다.

- worker exit code: `0`
- watchdog: `false`
- supervisor elapsed: `84.53322536998894s`
- first-overlap Viewer hold: `60.02044868003577s`
- UI/render update: `565`
- frozen full-state display write attempts/successes: `3/3`
- timeline bridge `Timeline.commit()`: 조건부 `1`회, 모든 no-advance check PASS
- q5 science evaluator / distance query / overlap-contact query / 새 classification:
  모두 `0`
- controlled physics step attempts/steps: `0/0`
- 마지막 snapshot: q0-q4/q5, object pose, object linear/angular velocity bits exact;
  timeline PAUSED-not-STOPPED; SimulationContext time/index와 timeline time은
  display/hold 동안 불변

여기서 “정상 종료”는 engine log가 깨끗했다는 뜻이 아니다. worker log에는
non-fatal `[Error] Failed to clone in Fabric` 1줄과 존재하지 않는
`d338_convex_parts` path warning 2줄이 있다. 그 뒤 scene setup, 세 capture,
60초 hold, state restoration, close, exit `0`가 완료됐으므로 D357 표시 계약을
중단시킨 exception은 아니지만 향후 보고에서 숨기지 않는다. 또한 worker check
이름 `timeline_time_zero_unchanged`는 부정확하다. 실제 baseline은 timeline
`0.029999999329447746s`, SimulationContext `0.009999999776482582s/index 2`이고,
증명한 것은 절대 0이 아니라 그 baseline에서 display/hold 동안 더 진행하지
않았다는 뜻이다. registered post-reset controlled physics step은 `0`이다.

근거:

- `claudedocs/runtime_logs/grasp_track/g0a_d357/d357_worker_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d357/d357_supervisor_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d357/d357_automated_summary.json`

초기 resource sample은 RTX 4090 Laptop GPU compute capability `8.9`,
VRAM `2676/16376MiB` 사용이었다. 전체 82 samples에서 VRAM 사용 범위는
`2676..8321MiB`, GPU utilization은 `0..36%`(평균
`9.963414634146341%`), 최소 available system RAM은 `11334754304 bytes`였다.
이 case는 10Hz 정지 화면 hold라 GPU를 포화시키는 계산 workload가 아니며,
임의 Warp/SM tuning은 넣지 않았다.

## 11. Rerun 및 수동 시각 검사

RRD/RBL/footer verify, exact 62 non-system entity inventory, exact timelines
`[blueprint, log_time, step]`, fixed blueprint export, isolated-port headless
screenshot이 모두 PASS했다. 실제 로봇·두 죠·원통과 세 event row가 보이며,
Rerun은 배치/재생 증거일 뿐 접촉력 또는 bit-exact 판정 권위가 아니다.

원본 해상도 검사에서 중요한 제한을 새로 발견했다. 동일 side camera의 세
실제 Isaac PNG에서는 움직이는 죠가 원통 뒤에 가려져 세 상태가 거의 같아
보인다. 따라서 이 PNG만으로 접촉을 봤다고 주장할 수 없다. 기존 산출물은
그대로 보존하고, 사전등록한 forward-only 보충 그림
`d357_isaac_camera_occlusion_addendum_ko.png`를 추가해 다음을 분리했다.

1. 보이는 것: 실제 Isaac 실행, 팔/원통 배치, 세 q5 표시 상태 저장.
2. 가려져 안 보이는 것: 죠-원통 경계와 어느 표면의 선접촉인지.
3. 아직 시험하지 않은 것: PhysX 접촉력/마찰, 양 죠 동시 접촉, 버티기/들기,
   grasp 성공/실패.

보충 PNG는 `2400x1500 RGB`, SHA-256
`567aab0e719c3cef52470c8b275b46b3f3b492b8eaadb9213c5b1e726309294f`이고,
원본 해상도에서 한국어 glyph, clipping, text overlap을 수동 PASS했다.

근거:

- `claudedocs/runtime_logs/grasp_track/g0a_d357/d357_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d357/d357_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d357/d357_isaac_camera_occlusion_addendum_summary.json`

## 12. 실행 도중 상태 해석 정정

supervisor command가 terminal output을 즉시 반환하지 않은 직후, summary 파일이
아직 보이지 않는 짧은 시점에 외부 종료로 잘못 의심했다. 그러나 곧 도착한
phase/worker/supervisor/postprocess 파일을 timestamp와 함께 다시 읽은 결과,
60초 hold와 postprocess까지 정상 완료했음이 확정됐다. 잘못 가정한 salvage는
기존 worker/supervisor summary 존재 guard에서 산출물 쓰기 전에 중단됐고,
external-termination/salvage artifact는 생성되지 않았다. 최종 권위는 위 정상
completion JSON이다.

실제 GUI invocation에 결속된 preregistered harness SHA-256은
`cdb6fc3fec5884050e34b19cf6e4943874f2d36c8f79b0bb90e8430816bb870e`다.
실행 후 수동 검사에서 발견한 camera occlusion을 설명하기 위한 addendum stage
추가와 잘못 가정했던 salvage branch 제거 후 현재 harness SHA-256은
`5f87dfb5a3ee9a4f36b8ad00399ed3df8d60196ddcad73686f7f00343a8100ea`다.
이 차이는 숨기지 않으며, 새 Isaac/q5/physics 실행을 포함하지 않는다.

## 13. 최종 verdict와 다음 경계

최종 operational verdict:
`D357_D354_BEGINNER_VISUALIZATION_REPAIR_COMPLETE`.

completion SHA-256:
`89a20139c12d6936ae052d0069829f0381e6935ba5dcb1b3dcbf581fc3581e71`.

D354 scientific verdict는 계속
`D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`, `g0a_pass=false`다. 즉 팔은 원통
위치까지 갔고 정지 상태 세 자세를 실제 Isaac에 표시했지만, 실제 PhysX로 죠를
닫아 접촉력/마찰/파지를 시험한 적은 아직 없다.

다음 승인 범위는 별도 forward-only D358 offline derived moving-jaw patch hash
provenance audit다. bundled standalone core-PXR 환경에서 dtype/unit/ordering/
canonicalization provenance만 감사하며 Isaac/q5/physics/cap-rim 판정은 실행하지
않는다.
