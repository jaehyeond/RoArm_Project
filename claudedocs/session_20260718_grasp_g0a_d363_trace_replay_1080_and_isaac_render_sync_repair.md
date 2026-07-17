# D363 — D362 trace replay 1080p and zero-step Isaac render-sync repair

Date: 2026-07-18 KST

Case: `g0a_d363`

Status at preregistration: `USER_APPROVED_IMPLEMENTATION_IN_PROGRESS_NO_D363_ISAAC_INVOCATION`

이번 case의 신규 변수:

1. `exact_1920x1080_trace_replay_encoding`
2. `zero_step_explicit_fabric_forward_capture_sync`

신규 target/IK/path/physics/geometry 변수: `[]`

## 1. 무엇을 왜 고치는가

D362의 canonical 물리 증거는 완결됐다. OPEN 200행과 q5-close 300행, 총 500행이
durable prefix와 `d362_physics_trace.json`에 보존됐고, moving `gripper_link` 접촉 확인은
closure step 32, 원통 운동 확인은 step 42, final 원통 변화는 XY 60.619mm와 약 90도
전도였다. 그러나 표시층에는 서로 독립적인 두 결함이 있었다.

1. 1920x1080으로 만든 원 frame을 imageio-ffmpeg가 `macro_block_size=16` 때문에
   1920x1088로 자동 확대했다.
2. 실제 Isaac PNG는 네 decision state에서 사실상 같은 세워진 원통을 보였다. D362의
   physics-only step은 default `update_fabric=False`였고 capture 전에 `forward()`가 없어,
   PhysX tensor pose가 Fabric/Hydra 화면으로 flush됐다는 증거가 없었다.

D363은 이 두 **관측 결함만** 고친다. D362의 접촉·운동 결과를 다시 계산하지 않고,
immutable trace를 사람이 정확히 볼 수 있는 영상과 실제 Isaac renderer에 재생한다.

## 2. 사용자 승인과 현재 Git 교차검사

- 사용자 승인: “D363 `[d362_trace_replay_1080_and_isaac_render_sync_repair]`
  observability-only case 진행을 승인한다. D362 immutable trace만 읽어 exact 1920x1080
  재생성과 zero-step Fabric forward() 캡처 검증만 수행하고, q5/physics 과학 재실행 및
  target/IK/path·물리 설정 변경은 하지 마라.”
- boot에서 실제 `HEAD == origin/master ==
  f085463d2e994a633cd1bcefe0c98c0b6c19e18e`.
- 최근 commit: `f085463 D363 observability-only 전 저장`.
- D363 편집 전 worktree: clean.
- `START_HERE.md` Git 절의 `68f2ff0...`와 “D362 uncommitted” 설명은 stale하므로 이번
  forward-only state update에서 실제 Git 명령 결과로 교정한다.
- commit/push는 승인되지 않았고 수행하지 않는다.

## 3. 권위 입력과 immutable 경계

- canonical time/state 입력은 D362 trace SHA-256
  `9483146c4941e6518614c63acbf221128a564bafa7a9928d41e633ee6e4e2044`의 500행뿐이다.
- decision row는 zero-based JSON index `199/232/242/499`, 즉 global step
  `200/233/243/500`의 precommand/contact-confirmation/motion-confirmation/final이다.
- D362 33-file tree는 새 D363 output 밖에서 읽기 전용이다. 사전 filename+size manifest
  digest `4b14fb9bde888f5ad63f215477fc298efc300b5f63ebcac7f710b48798ec36d8`,
  file-SHA manifest digest
  `33a147c7fa2c02b90a4d972a158aba3cfbbffe0b19814535d267336a92f057be`를 실행 뒤 다시
  계산한다.
- D351-D362와 사용자 소유
  `claudedocs/lab_meeting/20260715/d334_collision_table/`은 add/overwrite/rename/rerun하지
  않는다.
- 영상의 64+64 surface는 D362가 이미 입력으로 고정했던 D348 callback-topology evidence
  SHA-256 `83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6`에서
  vertices/triangles만 읽는 표시용 입력이다. 새 corrected/live USD·PhysX inventory,
  hppfcl/property/volume query는 하지 않는다. 시간·상태 권위는 D362 JSON이며
  Rerun/MP4/PNG Float32 표시는 과학 gate로 역해시하지 않는다.

## 4. 신규 변수 1 — exact 1920x1080 trace replay

- D362 `_render_trace_replay_frame()`과 500행→250 frame index, 20fps, libx264,
  yuv420p, quality 7을 그대로 상속한다.
- writer의 유일한 encoding 변화는 `macro_block_size=16 → 1`이다. 1920과 1080은 모두
  yuv420p의 짝수 치수 조건을 만족하므로 자동 1088 scale이 필요 없다.
- 새 `g0a_d363/` MP4만 생성하며 D362 MP4/report를 수정하지 않는다.
- exact width/height, 250 frames, 20fps, H.264, yuv420p, bundled FFmpeg full decode,
  OpenCV first/middle/final nonblank를 gate한다. Source indices는 endpoint/unique만 보지 않고
  D362 report와 `np.linspace(0,499,250,dtype=int64)` 전체 배열이 exact여야 한다.
- 영상은 “D362 canonical trace replay; physics not recomputed”라고 명시한다.

## 5. 신규 변수 2 — zero-step explicit Fabric forward 동기화

- 새 worker는 frozen D362/D360 scene을 1회만 연다. reset 내부 warm-up은 별도 기록하고
  D363 controlled step으로 세지 않는다.
- reset 뒤 timeline을 `PAUSED-not-STOPPED`, `/app/player/playSimulations=false`로 고정한다.
- 네 trace row마다 actual joint position/velocity와 object world pose/quaternion/linear+
  angular velocity를 Float32로 PhysX view에 직접 쓴다. Drive target, q5 command,
  `scene.write_data_to_sim()`, sensor/contact query는 호출하지 않는다.
- 각 row의 exact 순서는 다음이다.

  `direct state write → before-forward primary/opposite capture → inner.sim.forward() exactly 1
  → after-forward primary/opposite capture`

- `inner.sim.render()`는 내부에서 추가 `forward()`를 호출하므로 금지한다. Capture는
  guarded raw `simulation_app.update()`만 사용한다.
- 모든 direct write/forward/app-update/capture 전후에 custom step counter, SimulationContext
  time/index, timeline time, joint/object Float32 bits를 검사한다. 등록 trace state를 쓴 뒤
  이 값 중 physics clock/counter가 하나라도 전진하면 FAIL_STOP이다.
- explicit forward count는 정확히 `4`, controlled physics step `0`, q5 science sample/target
  update `0`, contact/sensor query `0`이어야 한다.

## 6. 화면-상태 판정과 음성/양성 대조군

- 각 PNG는 원본 1280x720로 decode하고 D362와 같은 HSV yellow largest-component에서
  bbox, centroid, area, PCA axis angle을 저장한다. Nonempty PNG만으로 PASS하지 않는다.
- canonical trace state의 Float32 bits와 worker direct-write readback bits가 exact여야 한다.
- precommand trace는 거의 upright, final trace는 약 90도 toppled이다. 따라서
  precommand-after와 final-after의 yellow silhouette가 bbox/centroid/axis에서 실질적으로
  달라야 한다.
- 결정적 positive control은 motion state가 화면에 남은 `final-before`와 toppled final을
  flush한 `final-after`의 차이다. 단, Kit가 raw app update에서 이미 새 PhysX transform을
  동기화하면 before가 이미 final일 수 있으므로 이 차이 자체를 mandatory PASS gate로
  두지는 않는다. 이를 `stale_reproduced/already_synced/ambiguous`로 별도 분류한다.
  Mandatory gate는 `final-after`가 D362 stale final 및 `precommand-after`와 다르고,
  canonical final의 moved-and-toppled category와 일치하는지다.
- precommand은 reset 화면과 거의 같을 수 있으므로 네 before/after pair 모두가 달라야
  한다는 잘못된 gate는 두지 않는다.
- 실제 Isaac pixel을 D362의 display-only 48deg projection에 절대 보정됐다고 주장하지
  않는다. 대신 양 camera에서 precommand→final의 상대 screen 이동방향 cosine `>=0.8`을
  mandatory로 하고, projected-axis delta 오차 `<=15deg`, observed 이동 `>=15px`와 함께
  3개 중 적어도 2개가 참이어야 한다. Expected pre/final role을 뒤집은 음성 대조는 반드시
  FAIL해야 한다. 이는 trace pose와 임의 yellow 물체를 혼동하지 않기 위한 상대 대응 gate다.
- offline negative controls는 reference video/no-advance 양성 2개와 wrong middle-order,
  wrong decision row, 1088 resolution, duplicate/missing index, step/clock increment, fifth
  forward, q5 target update, D362 tree mutation 거부를 포함해 `12/12`여야 한다.

정량 기준도 실행 전에 다음처럼 고정한다. 두 yellow mask 비교에서 centroid 거리
`>=15px`, PCA axis 최소각 차이 `>=15deg`, mask IoU `<=0.85` 중 적어도 두 항목이
참이면 `materially_different=true`다. Upright는 bbox `height/width >=1.5`, toppled는
`width/height >=1.15`로 분류한다. 모든 largest component area는 `>=500px`여야 한다.
이 기준은 final-after 대 precommand-after 및 final-after 대 D362 stale-final에 적용한다.
접촉/운동 중간 두 상태 사이의 subpixel silhouette 차이는 gate하지 않는다.

## 7. Rerun·수동검사·완료 순서

D363 판정은 공간·시간 동기화에 의존하므로 D362 RRD를 재사용하는 것만으로 끝내지 않는다.
새 RRD/RBL에는 네 state × before/after actual capture, paired beginner sheet, canonical cylinder
pose, rendered silhouette metrics, forward count와 simulation time/index·timeline·custom-counter
delta를 `sync_step` timeline에 기록한다. Footer-enabled verify, exact entity/timeline/component
contract(`blueprint/log_time/sync_step` exact), fixed RBL, headless screenshot을 gate한다.

실행 순서는 다음과 같다.

1. 이 사전등록과 새 harness를 정적 검토한다.
2. CPU-only prepare가 새 `g0a_d363/`를 exclusive create하고 입력 hashes, Git, pins,
   display/GPU/RAM, D362 tree, negative controls를 검증한다.
3. 승인된 host `--stage run`을 정확히 1회 실행한다. 내부 AppLauncher/worker도 1회이며
   자동 retry는 없다.
4. worker close 뒤 MP4/16 actual PNG/RRD/RBL/screenshot/report를 자동 검증한다.
5. 원본 해상도 PNG, MP4 전체 재생, Rerun screenshot을 실제로 검사한 manual JSON 뒤에만
   CPU-only finalize를 실행한다. Automated artifact가 묶은 전체 pre-manual inventory/hash를
   finalize에서 다시 대조한다. Manual은 MP4 SHA, 실제 재생시간, first/last source row와
   한글 관찰문을 기록하고, completion은 automated/manual을 포함한 최종 precompletion
   hash map과 그 canonical digest를 보존한다.

## 8. 동결·금지 범위와 현재 residue

- q5 science sample/command/update, controlled physics step, contact/force query: `0`
- target/IK/path, initial-science question, asset/cook/decomposition/gate/tolerance,
  material/mass/actuator/solver/physics setting change: `0`
- cap/rim/barrel, exact face/manifold, force closure, grasp/hold/lift/G0a 판정: `0`
- settle, ten-trial, G0b, RL/PPO/VLA, ladder promotion: `0`
- hardware, B200/SSH, package install, commit/push: `0`
- automatic retry/resume/overwrite/unapproved signal: `0`

잔류 D342 PID `1729639`는 2026-07-18 preflight에서 parent user-systemd PID 1123 아래
4일 이상 살아 있고 GPU 320MiB를 쓰는 것으로 확인됐다. 전체 GPU는 RTX 4090 Laptop
SM 8.9, total/used/free `16376/2051/13894MiB`였다. D363 승인은 이 잔류 PID에 signal을
보내는 권한으로 확대하지 않는다. Worker는 이를 입력 risk로 기록하되, resource gate와
단일 D363 invocation을 독립 검증한다.

D363은 D362 평가에서 실제로 발생한 observability 실패에 대한 reactive repair이며,
영상 resolution과 Fabric-state synchronization 모두 PASS/FAIL이 바뀔 수 있는 검증이다.
따라서 새 q5/physics 평가 없이도 AGENTS Session progress rule의 reactive control-contract
예외를 충족한다. 이 문서는 prepare와 D363 Isaac invocation 전에 작성됐다.

## 9. 실제 실행 순서와 단일 실행 계약

1. preregistration 전에 `HEAD == origin/master ==
   f085463d2e994a633cd1bcefe0c98c0b6c19e18e`, clean worktree를 확인했다.
2. harness SHA-256
   `63b307137405b2a343af88e046e992ef4ee996aff3bc467e2bf58390e4e18a14`를
   preregistration에 결박했다. 세 차례 독립 정적 검토에서 실행 blocker가 없음을 확인한
   뒤 harness는 수정하지 않았다.
3. CPU-only prepare를 1회 실행했다. 입력/Git/D362 33-file manifest/D348 frozen display
   topology/D334 sidecar/pins/display/GPU/RAM/FFmpeg/Rerun/static audit/negative controls를
   포함한 `25/25`가 PASS였다. worker 진입 전 preflight도 `15/15` PASS였다.
4. 승인된 Isaac worker를 정확히 1회 실행했다. worker exit `0`, watchdog `null`, 자동
   retry `false`, 총 wall time `54.4547970489366s`였다. 외부 supervisor exit `2`는
   worker crash가 아니라 후처리 render-sync gate FAIL을 반영한다.
5. 16개 실제 PNG와 MP4/RRD/RBL/종합 그림을 생성·자동 검사했다. 이어 원본 PNG 16개,
   종합 그림 5개, MP4의 250프레임 전체를 직접 검사했다. MP4는 0.9배속 전체 decode
   playback `13.21s`로 first source row `0`과 last row `499`를 확인했다.
6. 수동 결과를 FAIL로 기록한 다음 CPU-only finalize를 정확히 1회 실행했다. 종료 verdict는
   `D363_OBSERVABILITY_OR_INTEGRITY_FAIL_STOP`, completion `pass=false`다.

근거:

- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_prepare_preflight.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_worker_preflight.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_supervisor_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_completion_summary.json`

## 10. 신규 변수 1 결과 — exact 1920x1080 재생은 PASS

D362의 immutable canonical trace를 시간·상태 권위로 읽고, D362가 이미 입력으로 결박한
frozen D348 64+64 topology는 로봇 표면 표시용으로만 상속해 새 MP4를 만들었다. 영상은 정확히
`1920x1080`, `250 frames`, `20.0fps`, `12.5s`, H.264/yuv420p였고 bundled FFmpeg
full decode와 first/middle/last nonblank 검사가 모두 PASS였다. 등록 source-index 배열과
관찰 배열의 digest도 모두
`2524ce072276f0726a0c110b91572624abb71941426a82745334861ceecd4347`로 같았다.
영상 SHA-256은
`2385fc89094acb03a7e8c3aa0b203e73a1e9a110dc9eb63c1116601c51a951e4`다.

수동 전 프레임 검사에서 row 0의 세워진 원통, 접촉 확인 뒤 기울어지는 구간, row 499의
누워서 로봇에서 멀어진 원통이 연속적으로 보였다. 각 frame에는
`PHYSICS NOT RECOMPUTED` 경고가 유지됐다. 즉 이 영상은 D362 기록의 정확한 시각 재생이며,
D363에서 물리를 새로 계산한 영상이 아니다. D362의 기존 1920x1088 파일은 그대로 동결했다.

근거:

- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_d362_trace_replay_report.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_d362_trace_replay_exact_1920x1080.mp4`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_d362_trace_replay_storyboard_1920x1080.png`

## 11. 신규 변수 2 결과 — actual Isaac render sync는 FAIL

네 기록 상태마다 direct display-state write `1회`와 explicit `SimulationContext.forward()`
`1회`를 수행해 합계 write `4`, forward `4`를 기록했다. controlled physics step,
q5 science sample, q5 target update, contact query는 모두 `0`이었다. 16개 PNG는 모두
decode/yellow-mask 조건을 통과했고 capture 전후 simulation clock/counter/state-cache bits는
전진하지 않았다.

하지만 실제 화면은 바뀌지 않았다.

- primary precommand-after 대 final-after:
  centroid `0.04910539000102259px`, axis `0.049759416015120905deg`,
  IoU `0.9965942454492073`; materially different `false`.
- opposite precommand-after 대 final-after:
  centroid `0.16152357284396698px`, axis `0.11867675711681613deg`,
  IoU `0.9947105495151337`; materially different `false`.
- primary final-before 대 final-after:
  centroid `0.011875451918142982px`, axis `0.003690711386610701deg`,
  IoU `0.9994713656387665`.
- D362 stale-final 대 D363 primary final-after:
  centroid `0.11118761940639038px`, axis `0.08459511120113916deg`,
  IoU `0.996066459226208`; materially different `false`.
- primary/opposite final-after는 둘 다 `toppled=false`였다.

등록 trace-render 상대 대응도 primary/opposite에서 각각 direction cosine
`-0.9555898628769772/-0.996726805586088`, observed displacement
`0.04910539000102259/0.16152357284396698px`로 FAIL했다. Expected 역할을 뒤집은 음성
대조도 PASS로 오인되지 않았으므로 discriminator 자체는 완화되지 않았다.

원본 16개를 눈으로 검사한 결과도 수치와 같았다. precommand/contact confirmation/
motion confirmation/final의 before/after와 양 camera 모두 원통이 거의 같은 자리에서
계속 수직으로 서 있었다. 따라서 explicit `forward()` 호출 사실은 증명됐지만 actual
Hydra viewport에 D362 final rigid-cylinder pose가 표시됐다는 증거는 없다.

근거:

- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_fabric_render_sync_report.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_fabric_sync_primary_storyboard_ko.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_fabric_sync_opposite_storyboard_ko.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_beginner_result_sheet_ko.png`

## 12. bit-exact readback의 의미와 아직 모르는 경계

자동 report의 `all_write_readback_bits_exact=true`를 독립 PhysX backend readback으로
해석하면 안 된다. D363 harness는 write 직후 `inner._sponge.data.root_pos_w`와
`root_quat_w`를 읽었다. 설치 IsaacLab `RigidObject.write_root_link_pose_to_sim()`은 입력을
먼저 AssetData의 `root_link_pose_w` cache에 복사한 뒤
`root_physx_view.set_transforms()`를 호출한다. `RigidObjectData.root_link_pose_w`는 data
timestamp가 simulation timestamp보다 오래됐을 때만
`root_physx_view.get_transforms()`로 backend를 다시 읽는다. D363은 timestamp를 전진시키는
asset/scene update를 하지 않았으므로 이 bit-exact 결과는 방금 쓴 **AssetData cache의
self-read**를 확정할 뿐이다.

설치 `SimulationContext.forward()`는 `_fabric_iface`가 있을 때만 작동한다. timeline이
paused이면 articulation kinematic update는 건너뛰고 `_update_fabric(0,0)`만 호출한다.
D363은 `cfg.use_fabric`, `_fabric_iface` 존재, 선택된 `force_update/update` callable,
independent `root_physx_view.get_transforms()`, USDRT/Fabric world matrix를 run artifact에
계측하지 않았다. 따라서 현재 확정된 단절은
`AssetData cache → PhysX backend → Fabric/USDRT → Hydra` 중 어딘가이며, 어느 화살표인지
아직 판별할 수 없다.

따라서 D362에서 확인한 “capture 전에 `forward()` 호출이 없었다”는 구현 결손은 사실이지만,
그 한 호출을 추가하면 화면이 복구된다는 충분원인 추론은 D363 결과로 superseded다.

worker log의 `Failed to clone in Fabric`는 실제 line에 존재한다. 그러나 같은 generic
오류가 D352, D353, D354, D357, D360, D362에도 반복됐고 D350/D357의 별도 static display
성공과도 공존할 수 있다. 이 메시지를 D363 stale cylinder의 단일 원인으로 승격하지 않는다.
CPU powersave/PCIe Gen1 경고도 성능 위험일 뿐 결정론적으로 같은 원통 pose가 남은 직접
원인 증거가 아니다.

관련 코드/로그:

- `sim_scripts/cyl34_top_view_d363_d362_trace_replay_1080_and_isaac_render_sync_repair.py:819`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/assets/rigid_object/rigid_object.py:225`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/assets/rigid_object/rigid_object_data.py:126`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py:466`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_worker_stdout_stderr.log:340`

## 13. Rerun·수동검사·자원 결과

새 RRD/RBL은 Rerun SDK/CLI `0.34.1`, exact timelines
`blueprint/log_time/sync_step`, exact entity/component contracts, footer verify, RBL verify,
headless screenshot을 모두 통과했다. RRD SHA-256은
`1b9ea16488970bed1385de7334cd9258c2115bfa1942e36dde613d65f4773f89`다.
Rerun screenshot에는 actual primary/opposite Isaac image와 판정표가 실제로 보였다.
actual image는 계속 upright였고 별도 canonical trace 3D panel은 toppled였으므로 두
관측층의 불일치를 숨기지 않았다.

수동 검사는 required `22/22` paths와 SHA를 exact 검증했다. `precommand_upright_seen=true`,
`final_moved_and_toppled_seen=false`, `d362_stale_vs_d363_final_difference_seen=false`,
`final_before_forward_classification=stale_reproduced`, manual `pass=false`다. 글자 겹침 없이
한글 beginner sheet와 양 storyboard를 읽을 수 있었다.

자원 telemetry는 53 samples였다. GPU used max/free min은 `7760/8185MiB`, utilization
max `42%`, worker RSS max `7,166,943,232B`였다. worker exit0 및 안정된 동일 화면과 함께
보면 VRAM OOM이나 GPU 미사용이 이번 FAIL의 근거는 아니다. 이 utilization은 Warp occupancy
또는 SM efficiency를 직접 측정한 값도 아니다.

근거:

- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_fabric_render_sync_rerun.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_supervisor_summary.json`

## 14. 무결성, 과학 범위, 최종 판정

finalize에서 harness/input/D362 33-file manifest/D334 sidecar/D363-D362 inode separation,
37개 automated-bound artifact SHA, precompletion inventory/hash map이 모두 exact였다.
completion precompletion hash-manifest digest는
`09acea8e7fc6ad3751a57a58fd06c2d357c7591491e65e872735a5e8a9a39138`다.

최종 operational verdict:

`D363_OBSERVABILITY_OR_INTEGRITY_FAIL_STOP`

이를 초보자 관점에서 풀면 다음과 같다.

- 성공: D362가 기록한 500-step 결과를 정확한 1920x1080 영상과 검증된 Rerun 파일로
  다시 볼 수 있게 됐다.
- 실패: 그 기록 자세를 실제 Isaac viewport에 직접 써서 `forward()`한 화면은 갱신되지
  않았다. 실제 Isaac 캡처로 D362의 쓰러진 final pose를 재현하지 못했다.
- 불변: D363은 물리 실험을 다시 하지 않았으므로 D362의 physical sub-verdict
  `D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED`는 상속만 한다. exact face/manifold,
  cap/rim/barrel order, force closure, stable grasp, hold/lift, target/IK repair justification는
  여전히 `null`, `g0a_pass=false`다.
- 금지 범위 준수: controlled physics/q5 science/q5 target/contact query는 `0`; target/IK/path,
  asset/decomposition/gate/material/mass/actuator/solver/physics 설정은 변경하지 않았다.

## 15. 다음 승인 경계

다음 가장 좁은 후보는 별도 승인 observability-only D364
`[paused_render_state_layer_localization]`이다. 같은 frozen D362 네 상태에서 physics step 없이
`forward()` 전후를 다음 네 층으로 독립 계측하는 범위만 제안한다.

1. AssetData cache
2. `root_physx_view.get_transforms()` 독립 backend read
3. USD local/world와 USDRT/Fabric world matrix
4. Hydra yellow-mask centroid/axis

동시에 `cfg.use_fabric`, `is_fabric_enabled()`, `_fabric_iface is not None`,
`force_update/update` 선택 callable을 attestation해야 한다. 이 결과가 있어야 cache→PhysX,
PhysX→Fabric, Fabric→Hydra 중 정확한 끊긴 경계를 말할 수 있다.

D364는 아직 승인되지 않았다. D363 output은 이 completion 뒤 동결하며 rerun/overwrite하지
않는다. q5/physics science 재실행, cap/rim discriminator, target/IK/path 변경, grasp/settle/
hold/lift/G0a, ten-trial, G0b, RL/PPO/VLA는 각각 다시 별도 명시 승인이 필요하다.

## 16. 핵심 SHA-256

- harness: `63b307137405b2a343af88e046e992ef4ee996aff3bc467e2bf58390e4e18a14`
- preregistration: `2f0beee3ca22c028493a4eaf1e6c534e776eb4213ed11d275593e859c3f412c0`
- prepare: `1c36ac5022c2a7ecfd716bf74f9dba88ac6e05328779d7dee135f87a0d3c10d5`
- worker summary: `634e77fd177d143a637a1042480ec533f770b0436b460a8fb7b73f4318677995`
- supervisor summary: `8f643dbe734e83c0c181997ff3a88ddc37b3f162ccdaacfebea4c437e6da597f`
- video report: `a53224f7ff4e075e987bce2adbd077fe6b1c2732fc4ec657a0ff1ced93852da5`
- sync report: `4cd5dd401b4eaea687549c5f5279b71e0f7fb0ad67a70f4d555f94b793653b3c`
- RRD: `1b9ea16488970bed1385de7334cd9258c2115bfa1942e36dde613d65f4773f89`
- Rerun validation: `a7cf1b8b9b0929a2900de336dd3e8627c2fb8acd791c2987b01f8198891c58c5`
- manual inspection: `46d7aae5dd7245a90af78deec3f54d18f14e62d999fd568c6f5893ba686bccc1`
- completion: `e55a155b814dabdb90ce6b219c36318431f695331342624d2d2780d7b7b4f078`
