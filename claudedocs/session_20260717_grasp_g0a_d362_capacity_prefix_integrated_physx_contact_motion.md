# D362 — capacity/prefix-integrated current-pose PhysX contact-motion rerun

Date: 2026-07-17 KST

Case: `g0a_d362`

Status at initial preregistration: `USER_APPROVED_IMPLEMENTATION_IN_PROGRESS_NO_INVOCATION`

이번 case의 신규 변수:

1. `runtime_contact_capacity_and_durable_prefix_integration`
2. `interface_visible_trace_replay_video`

신규 target/IK/path/geometry/material/actuator/solver 변수: `[]`

## 1. 무엇을 왜 다시 실행하는가

D360은 실제 OPEN baseline `200/200` step 뒤 q5 target을 `0.0rad`로 한 번 바꾸고
closure 43개 row까지 계산했다. 그러나 상세 접촉점 총 capacity가 `16`뿐이어서
`>16` warning 뒤 ATen index-range assertion과 PhysX CUDA device assert가 발생했다.
243개 row도 실행 끝에 한꺼번에 쓰도록 구현돼 body/value가 디스크에 남지 않았다.
따라서 D360은 어떤 robot body가 닿았는지, 힘이 얼마인지, 원통이 실제로 얼마나
움직였는지 판정하지 못했다.

D361은 설치 PhysX `5.6.1`과 동결 collider inventory에서 보수적 total capacity를
`1 × (1+1+64+64) × 256 = 33,280`으로 근거화했고, 매 step의 시작과 완성된
body/value observation을 append-only + `fsync` + exact reread하는 protocol을
failure injection `17/17`로 검증했다. 그러나 실제 PhysX는 실행하지 않아
`runtime_sufficiency=null`이었다.

D362는 D360의 물리 질문을 바꾸지 않고 이 두 operational 결함만 실제 worker에
통합한다. 질문은 다음 하나다.

> 동결된 현재 자세와 물리 설정에서 q5 target만 OPEN에서 CLOSED로 한 번 바꾸면,
> 어떤 robot body가 먼저 `0.1N` 이상 2-step 접촉을 만들고, 그 뒤 원통이 OPEN
> baseline 끝보다 XY `0.5mm` 이상 또는 tilt `1deg` 이상 2-step 움직이는가?

## 2. 사용자 승인과 Git 기준

- 사용자 승인: 2026-07-17 “승인할게 진행해봐. 그리고 git은 내가 D361완결 및
  동결이라는 이름으로 push했어.”
- 실제 boot Git: `HEAD == origin/master ==
  68f2ff040831c13b0198fe68ef88fe84a76a9df3`.
- 최근 commit: `68f2ff0 D361완결 및 동결`.
- D362 편집 전 worktree: clean.
- `START_HERE.md`의 이전 Git 절은 `e7ed71c...`와 D361 dirty worktree를 적어 stale했다.
  D362는 실제 위 Git 명령 결과를 권위로 사용한다.
- commit/push는 승인되지 않았다.

## 3. 동결된 D360 물리 계약

- seed `33201`, environment `1`, device `cuda:0`, GUI `headless=false`, `DISPLAY=:1`.
- q0-q5 OPEN Float32:
  `[0.03750238195061684, 0.542945146560669, 1.9687392711639404,
  0.18299327790737152, 0.0, 1.5413000583648682]rad`.
- q5 `0=CLOSED`, `1.5413000583648682rad=OPEN`; q5 target change는 OPEN→`0.0rad`
  정확히 한 번이다.
- cylinder position `[0.30000001192092896,0.0,0.03288299962878227]m`,
  quaternion `[1,0,0,0]`, radius `0.017m`, height `0.090m`, mass `0.72kg`.
- material static/dynamic friction `1.5/1.2`, restitution `0.0`.
- physics `dt=0.005s`; gripper actuator stiffness/damping/effort/velocity limit
  `80/4/2.5/3.14`.
- OPEN baseline `200` step, closure maximum이자 정상 horizon `300` step.
- contact threshold `0.1N`, object motion threshold XY `0.5mm` 또는 tilt `1deg`,
  모두 같은 phase의 2개 연속 row가 필요하다.
- D348 corrected callback-topology `256/256` channels, `128/128` parts와 live
  link5/gripper `64+64` binding은 physics 전에 다시 PASS해야 한다.
- actual Isaac camera는 D360 primary/opposite oblique view를 그대로 쓴다.

## 4. 신규 변수 1 — runtime capacity + durable prefix 통합

- ContactSensor의 물리 force/contact 의미는 바꾸지 않고 observability 설정
  `max_contact_data_count_per_prim`만 D361 등록값 `33,280`으로 바꾼다.
- runtime에서 sensor body `1`, environment `1`, filter order/path, filter count `4`,
  actual collider `1+1+64+64`, cfg `33,280`을 첫 controlled step 전에 exact 검증한다.
- 첫 controlled step 전에 exclusive prefix를 만든다. 각 step은 반드시
  `step_begin append+fsync` → physics step → full D360 state/body/filter force/contact
  point + raw count/start/high-water + event body/value `step_observation append+fsync+
  exact reread` 순서다.
- 정상 500-row horizon 또는 합법적 200-row baseline fail-stop만 seal할 수 있다.
  예외면 파일을 고치거나 resume하지 않고 valid prefix, partial tail, terminal inflight
  step을 그대로 감사한다.
- optional image/video/RRD/summary는 해당 step observation의 durable write 뒤에만 허용한다.
- 성공 worker는 final trace에서 display-only `event_masks`를 제외한 canonical state와
  durable prefix를 독립 reconciliation해야 한다.

## 5. 신규 변수 2 — interface-visible trace replay video

- 실제 Isaac decision-state는 initial/open-precommand/contact-if-present/motion-if-present/
  final을 primary와 opposite 두 view로 저장한다.
- full step timeline은 RRD/RBL에 실제 live 64+64 collider, cylinder pose, per-body force,
  contact points, q5와 motion scalar를 기록한다.
- 추가 MP4는 동일 canonical trace와 live collider stream을 두 anti-occlusion view로
  replay하고 q5/body force/object motion/step overlay를 표시한다. 이 replay는
  `physics not recomputed`라고 명시한다.
- MP4 생성은 물리 step을 추가하지 않으며 JSON/PhysX sensor보다 낮은 표시 권위다.
  ffprobe decode contract와 event storyboard를 만들고 원본 해상도로 직접 검사한다.

## 6. 실행·판정 순서

1. 새 `g0a_d362/`를 exclusive create하는 prepare preflight를 실행한다.
2. frozen hashes, Git scope, D360/D361 evidence, Python pins, DISPLAY, RTX4090, VRAM/RAM,
   ffmpeg/ffprobe/Rerun을 확인한다.
3. 두 독립 정적 검토와 offline negative controls가 PASS한 뒤에만 invocation marker를 쓴다.
4. supervisor가 실제 GUI Isaac worker를 정확히 한 번 시작하고 telemetry와 bounded
   watchdog를 기록한다. retry/overwrite는 없다.
5. worker는 exact runtime inventory/capacity/prefix header를 확인한 뒤 200-step OPEN
   baseline을 실행한다. baseline hard gate가 실패하면 q5를 바꾸지 않고 200-row prefix를
   seal한다.
6. baseline PASS일 때만 q5 target을 `0.0rad`로 한 번 바꾸고 closure 300 step을 모두
   실행한다. success early-stop은 없다.
7. trace/prefix/runtime capacity/actual Isaac captures/RRD/video를 각각 검증한다.
8. 원본 영상과 이미지를 직접 검사한 manual artifact 뒤에만 finalize한다.

### Verdict 경계

- moving `gripper_link`가 link4/link5보다 엄격히 먼저 2-step contact하고 그 뒤 threshold
  object motion이 있으면 body-level positive witness다.
- link4/link5가 먼저 또는 같은 solver row이면 clean moving-jaw attribution은 confounded다.
- contact만 있고 threshold motion이 없으면 contact-without-threshold-motion이다.
- baseline, control, capacity overflow, prefix, runtime, inventory, visualization이 실패하면
  각각 operational/observability FAIL_STOP이며 물리 질문을 과장하지 않는다.
- 어떤 결과도 exact triangle/face, cap/rim/barrel, 양면 force closure, stable grasp,
  hold/lift 또는 G0a를 판정하지 않는다. `g0a_pass=false`다.

## 7. 금지·동결 범위

- target/IK/path, initial q0-q5/object state 변경 `0`
- asset/cook/decomposition/gate/tolerance/material/mass/actuator/solver/physics/renderer 변경 `0`
- exact cap/rim/barrel discriminator, force closure, grasp/hold/lift/G0a 판정 `0`
- settle, ten-trial, G0b, RL/PPO, VLA, ladder promotion `0`
- real robot/hardware, B200/SSH, package install `0`
- D351-D361 overwrite/rerun/rename/add `0`
- user-owned `claudedocs/lab_meeting/20260715/d334_collision_table/` write `0`
- unapproved signal, commit, push `0`

D362는 q5-close perturbation이 실제로 접촉/운동 여부를 바꿀 수 있는 실패 가능한 평가다.
따라서 AGENTS.md Session progress rule을 충족한다. 이 절은 harness freeze, prepare, 실제
Isaac invocation 전 작성됐다. 결과와 exact hashes는 forward-only 후속 절에만 추가한다.

## 8. 실행 전 도구·복구 계약 정밀화

이 절도 D362 prepare와 Isaac invocation 전에 append했다. 위 §5의 `ffprobe decode
contract` 표현은 이 머신의 실제 설치 상태와 맞지 않아 다음 exact 구현으로 정정한다.

- standalone `ffprobe`를 새로 설치하거나 호출하지 않는다.
- 설치된 `imageio_ffmpeg==0.6.0`가 번들한 FFmpeg `7.0.2`와 `libx264` encoder를
  prepare에서 확인한다.
- 생성 MP4는 번들 FFmpeg로 파일 전체를 decode하고, 출력 stream이 H.264 및
  `yuv420p`인지 확인한다.
- OpenCV `4.11.0`으로 첫·중간·마지막 frame을 독립 decode하여 해상도, frame 수,
  FPS, nonblank를 확인한다.
- 따라서 이후의 정확한 영상 gate는 “bundled FFmpeg full-decode + OpenCV endpoint
  decode”다. package install은 0이다.

D360 원 worker log
`claudedocs/runtime_logs/grasp_track/g0a_d360/d360_worker_stdout_stderr.log`의 SHA-256은
`1bd0aa5a6060da283c8f84b83a0608156624de68aea47eba8c08b5878fc3ecf5`이며, line 367에
`Incomplete contact data ... more contact data points ... maxContactDataCount = 16` 경고가
정확히 남아 있다. D361의 `runtime_sufficiency=null`을 이번 실제 실행에서 닫으려면 다음
두 성분이 모두 필요하다.

1. runtime cfg뿐 아니라 실제 PhysX backend max가 `33,280`이고 상세 6개 buffer shape가
   `[33280,1]`, `[33280,3]`, `[33280,3]`, `[33280,1]`, `[1,4]`, `[1,4]`인지 첫
   controlled step 전에 zero-step으로 확인한다.
2. worker 종료 후 fsync된 전체 log를 supervisor가 독립 감사하여 `Incomplete contact
   data`, `more contact data points`, `maxContactDataCount` 관련 overflow warning이
   0건인지 확인한다. 1건이라도 있으면 D362 science/operational PASS를 금지한다.

hard exit/SIGTERM/device abort에서는 worker의 Python `except/finally`를 믿지 않는다.
supervisor가 process wait와 log fsync 직후 prefix를 CPU/file-only로 다시 검증한다.
정상 경로는 sealed complete prefix와 worker audit hash를 대조하고, 비정상 경로는 valid
record/observation 수, terminal inflight step, trailing bytes를 보존한다. resume/retry/
overwrite는 계속 금지다.

두 독립 read-only 정적 검토는 다음 STOP 결함을 발견해 실행 전에 수정했다.

- old D333 capacity-16 factory 호출, prefix helper 미연결, begin/observe 순서, D360 exact
  state-row에 trace-only field가 섞이던 문제
- backend allocation 미확인, baseline-FAIL phase 순서, hard-exit supervisor recovery 누락,
  partial JSON/JSONL parsing, missing-inventory 누락, observability 자동 PASS 누수, instance
  proxy collider 누락, 영상 수동검사 경로 누락, overflow warning 부재 gate 누락

수정 후 독립 prefix/API 검토는 blocking defect 없음으로 종료됐다. 별도 machine gate는
D360에서 물리·판정에 사용한 14개 함수의 source를 verdict label `D360/D362`만 정규화한
뒤 모두 exact equality로 확인한다. offline synthetic video layout negative control은
5/5 PASS였고 1920x1080 frame pixel 표준편차는 `30.504630488559773`이었다. 이 값들은
표시 파이프라인 검증일 뿐 접촉/운동 과학 증거가 아니다.

trace replay의 동결 live surface는 D348/D349의 link5+moving-gripper `64+64`뿐이다.
따라서 link4 surface를 새로 꾸며내지 않는다. link4는 authoritative aggregated sensor
contact point/force marker로 표시하며, 전체 robot 형상은 actual Isaac 두-view capture로
검사한다. 이 한계를 영상 report와 화면에 명시한다.

## 9. 실행 전 harness 동결

- frozen D362 harness SHA-256:
  `80fb5f47ec01de67c23b11f92fc6b46f3bff7063fc9474436a7863cf1c9df11c`
- 두 번째 독립 정적 검토는 이 exact blob에서 blocking defect 없음으로 종료됐다.
- 이 pin 이후 harness를 수정하면 prepare/run hash gate가 즉시 STOP한다.
- 아직 prepare output 생성, invocation marker, AppLauncher, Isaac/PhysX step은 모두 0이다.

## 10. prepare와 승인된 단 1회 실제 실행

Prepare는 Isaac을 import/launch하지 않은 상태에서 PASS했다.

- prepare SHA-256:
  `71b1b16bebc3c2a0fe1829ce8e22e8c46dd88ae8daaaad656b882ae2ac0f7aa3`
- preregistration SHA-256:
  `1eebbf12cf23c99a83e1640ce409efe8e0cde9ab2c0b5db43c06485ac7969e91`
- 실제 GPU는 RTX 4090 Laptop, compute capability `8.9`, total `16376MiB`,
  prepare 직전 used/free `2050/13894MiB`였다.
- `numpy==1.26.0`, `psutil==5.9.8`, Rerun SDK/CLI `0.34.1`, OpenCV
  `4.11.0`, imageio-ffmpeg `0.6.0`과 bundled FFmpeg `7.0.2/libx264`가
  모두 등록값과 일치했다. package install은 0이다.

사용자 승인 뒤 다음 host command를 정확히 한 번 실행했다.

```bash
DISPLAY=:1 /home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d362_current_pose_capacity_prefix_integrated_physx_contact_motion.py \
  --stage run
```

- supervisor elapsed: `74.08850605098996s`
- worker exit: `0`; outer supervisor exit: `2`
- watchdog/worker exception/automatic retry: `false/false/false`
- 실제 invocation 수: `1`; 재실행·resume·overwrite: `0`
- worker operational `pass=true`, supervisor/case `pass=false`

이 exit `2`는 아래 §13의 영상/phase observability gate가 만든 것이다. Worker는
예외 없이 실제 500 controlled physics step을 마쳤다. 따라서 outer exit와 worker
physics 종료 상태를 혼동하면 안 된다.

## 11. 실제 PhysX body-level 접촉·운동 원자료

### 11.1 OPEN baseline

200-step OPEN baseline은 모든 등록 hard gate를 PASS했다.

- q5 actual: `1.5412994623184204 → 1.5412912368774414rad`
- robot filter max: link4/link5/gripper 모두 `0.0N`
- cylinder XY 최대 이동: `0.003773643762621384mm`
- cylinder 최대 tilt: `0.003364520785190337deg`
- 마지막 50-step table support median z-force: `7.063635349273682N`
- precommand robot-contact/object-motion confound: 둘 다 `false`

그 뒤 q5 target만 `1.5413000583648682 → 0.0rad`로 정확히 한 번 바꿨다.
q0-q4 target bit는 바뀌지 않았고 target/IK/path·asset·material·mass·actuator·solver·
physics 설정 변경은 0이다.

### 11.2 closure event 순서

`dt=0.005s`에서 등록 threshold를 두 연속 row로 확인한 순서는 다음과 같다.

| Event | closure step / global step | q5 actual | 등록 sensor 값 |
|---|---:|---:|---:|
| moving `gripper_link` force onset | `31 / 232` | `1.0388332605rad` | `0.2780868631N` |
| moving `gripper_link` 2-row confirmation | `32 / 233` | `1.0231319666rad` | `2.8690685090N` |
| cylinder motion onset | `41 / 242` | `0.8818207383rad` | XY `0.5184094935mm` |
| cylinder motion 2-row confirmation | `42 / 243` | `0.8661201596rad` | XY `0.6365670514mm`, tilt delta `0.8153707093deg` |
| fixed `link5` force onset | `45 / 246` | `0.8190184236rad` | `3.3532596889N` |
| fixed `link5` 2-row confirmation | `46 / 247` | `0.8033178449rad` | `7.9014611099N` |

즉 moving jaw의 등록 양의 force onset 뒤 `10` physics step, 정확히 `0.05s` 뒤
cylinder motion threshold onset이 왔다. `link5` 양의 force는 moving jaw보다
`14` step 늦었다. `link4`는 500 row 전체에서 등록 count/force가 `0/0N`이었다.
이는 이 sensor/filter 계약에서 양의 link4 event가 없었다는 뜻이지, 모든 종류의
link4 접촉이 절대 없었다는 일반 명제는 아니다.

접촉점·힘의 주요 값은 다음과 같다.

- moving confirmation point:
  `[0.2902432978153229,-0.013921898789703846,0.07778545469045639]m`
- moving gripper peak: closure step `54`, `43.85833992858175N`
- fixed link5 peak: closure step `55`, `23.227865254723564N`
- link4 peak: `0.0N`
- detailed contact-point high-water: `22/33,280`; 당시 table/link4/link5/gripper
  count는 `4/0/13/5`, 남은 등록 headroom은 `33,258`이다.

500번째 controlled row에서 q5 actual은 `2.3607312393987556e-13rad`였고 cylinder는
OPEN baseline 끝 대비 다음처럼 바뀌었다.

- XY displacement: `60.61899778989994mm`
- tilt delta: `89.99777464743418deg`
- z delta: `-28.000520542263985mm`

따라서 body-level 물리 하위판정은 worker가 등록한
`D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED`다. 쉽게 말하면 현재 자세에서
moving jaw가 먼저 원통에 힘을 가했고 원통은 뒤이어 크게 밀리고 넘어졌다. 이것은
원통을 안정적으로 잡았다는 결과가 아니다. Exact face/manifold, cap/rim/barrel 순서,
양면 force closure, hold/lift, grasp 성공은 모두 `null`이고 `g0a_pass=false`다.

또한 q0-q4 target은 동결됐지만 actual 최대 drift는 closure step 59에서
`0.04380311071872711rad`였다. 이는 접촉 하중 아래 실제 articulation의 유한 변위를
그대로 보고한 진단값이며, target 변경으로 해석하지 않는다.

## 12. D361 capacity와 durable prefix의 실제 runtime 검증

첫 controlled step 전에 실제 backend와 buffer를 읽어 다음을 확인했다.

- sensor cfg / derived total / backend max: 모두 `33,280`
- actual collider count: cylinder/table/link4/link5/gripper = `1/1/1/64/64`
- backend detailed buffer shapes:
  `[33280,1]`, `[33280,3]`, `[33280,3]`, `[33280,1]`, `[1,4]`, `[1,4]`
- paused buffer probe가 추가한 controlled physics step: `0`

Durable prefix는 다음 상태로 완결·봉인됐다.

- SHA-256:
  `aa7f7419516f4dda723290d89389df680ef3336f2b16984d4f467b76eee41a8e`
- bytes/records: `2,744,124 / 1,002`
- record arithmetic: header `1` + begin `500` + observation `500` + seal `1`
- completed observations: `500`; terminal inflight: `null`; trailing bytes: `0`
- final seal: `full_500_step_horizon_complete`
- trace/state reconciliation, semantic audit, independent supervisor recovery: PASS
- recovery classification: `COMPLETE_SEALED_PREFIX`

개별 1,002 receipt의 offset/hash 목록 자체는 final artifact에 보존하지 않았고,
receipt count와 aggregate `fsync+reread` PASS만 보존했다. 따라서 최종 wire/hash-chain과
산술은 독립 재검증했지만 과거 각 fsync syscall을 사후 재현했다고 과장하지 않는다.

전체 worker log `22,255B`를 supervisor가 독립 검사한 결과 등록 overflow token
(`Incomplete contact data`, `more contact data points`, `maxContactDataCount`)은 `0건`이다.
D361의 `runtime_sufficiency=null`은 **이 exact inventory/run에 한해** high-water
`22 <= 33,280`과 warning 0으로 닫혔다. Inventory나 PhysX version이 바뀌면 다시
산정해야 한다.

## 13. 전체 case가 FAIL_STOP인 이유 — 두 observability 문제

### 13.1 MP4 exact-resolution gate

Canonical trace-replay MP4는 500 row에서 250 frame을 만들었고 H.264/libx264,
`yuv420p`, `20fps`, `12.5s`, full decode, frame count, first/middle/last nonblank는 모두
PASS했다. 그러나 imageio writer의 `macro_block_size=16`이 입력 `1920x1080`을
`1920x1088`로 자동 확장했다. Worker log line 367과 video report가 이 값을 직접
기록한다.

- registered resolution: `1920x1080`
- encoded/decoded resolution: `1920x1088`
- 유일한 automatic video check FAIL: `resolution_1920x1080=false`
- video report `pass=false`

원본 MP4의 first/middle/last frame을 decode하고 4개 quadrant로 잘라 직접 검사했다.
제목, 두 view label, q5 chart, numeric/event panel은 읽을 수 있었고 서로 겹치지 않았다.
Beginner sheet와 storyboard도 원본 해상도에서 글자 겹침 없이 읽을 수 있었다. 즉
내용이 공백/파손된 실패가 아니라 정확한 출력 해상도 계약 위반이다. 그래도 등록
gate를 사후 완화하거나 기존 MP4/report를 덮어쓰지 않는다.

### 13.2 actual Isaac PNG와 physics state의 화면 동기화 실패

더 중요한 수동검사 결과가 따로 있다. Capture report는 PNG가 `1280x720 RGBA`로
decode되는지만 PASS했다. 그러나 actual Isaac primary PNG를 원본 해상도로 비교하면
precommand/contact/motion/final에서 원통이 계속 세워져 보인다. Trace의 final
`60.619mm/약 90deg` 전도와 일치하지 않는다.

동일 raw PNG에 HSV yellow mask(`H 15..45`, `S 40..255`, `V 80..255`)와 largest
connected component를 적용한 read-only 수치 교차검사는 다음과 같았다.

| Capture | yellow bbox `(x,y,w,h)` | centroid `(x,y)` px |
|---|---|---|
| precommand | `(628,299,90,209)` | `(671.074,400.917)` |
| contact | `(628,299,90,209)` | `(671.135,400.932)` |
| motion | `(628,299,90,209)` | `(671.091,400.860)` |
| final | `(628,299,90,209)` | `(671.109,400.958)` |

즉 physics tensor/prefix는 서로 다른 object pose를 보존했지만, 이 네 actual interface
PNG의 cylinder silhouette는 사실상 같은 pose다. Capture guard가 증명한 것은 캡처가
추가 physics step을 만들지 않고 tensor state bit를 바꾸지 않았다는 것뿐이며, renderer가
그 tensor pose를 화면에 반영했다는 증명은 아니었다.

설치 소스와 실제 harness를 다시 읽어 renderer가 stale했던 구현 결손은 다음처럼
국소화했다.

1. D362 `_physics_step_checked()`는 D332 `_physics_step()`에 위임하고, 그 함수는
   `inner.sim.step(render=False)`만 호출한다.
2. 설치 Isaac Sim 5.1 `SimulationContext.step(render=False,
   update_fabric=False)`의 기본은 physics-only step이며, 문서도 Fabric interface에서
   갱신 transform을 읽으려면 `update_fabric=True`가 필요하다고 명시한다.
3. 설치 IsaacLab `SimulationContext.render()`는 Hydra texture를 갱신하기 전에
   `forward()`로 Fabric data를 flush한다. D362 capture는 timeline을 pause한 뒤
   `simulation_app.update()`만 반복하고 `inner.sim.forward()`/`render()`를 호출하지 않았다.
4. 반대로 D350이 상속한 D332 exact-state 정적 경로는 state write 뒤
   `inner.sim.forward()`를 호출한다. 그러므로 D350의 한 pose가 정상 표시된 사실과
   D362 동적 capture가 stale했던 사실은 모순이 아니다.

Source locations:

- D362 delegate/capture: current harness lines `2133-2145`, `1882-1933`
- D332 state-write/physics step: `sim_scripts/cyl34_top_view_d332_grasp_g0a_static_collision_discriminator.py:578-609`
- D350 call into that exact-state path:
  `sim_scripts/cyl34_top_view_d350_fixed_jaw_geometry_viewer.py:1795`
- Isaac Sim core step contract:
  `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py:672-714`
- PhysX conditional Fabric update:
  `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/physics_context/physics_context.py:563-574`
- IsaacLab render flush:
  `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py:575-597`

따라서 D362 actual PNG 실패의 직접 구현 결손은 **physics-only step 뒤 capture 전에
PhysX pose를 Fabric/renderer로 force-update하지 않은 것**이다. Log line 349의
`[Error] Failed to clone in Fabric`은 D360/D362 모두에 있지만, 이 error를 stale PNG의
별도 단일 원인이라고 확정하지는 않는다. Pause 자체도 원인으로 확정하지 않는다.

Rerun RRD/RBL 자체의 schema/entity/timeline/footer 검증은 PASS했고 replay screenshot과
trace video에서는 원통 전도가 보인다. 하지만 둘은 canonical trace를 표시한 replay
층이지 actual Isaac renderer 동기화의 대체 증거가 아니다.

## 14. 최종 판정, 동결, 다음 승인 경계

- physical sub-verdict:
  `D362_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED`
- overall operational verdict:
  `D362_SINGLE_INVOCATION_PHYSX_TRACE_COMPLETE_OBSERVABILITY_FAIL_STOP`
- controlled physics: `500`; q5 target update: `1`; invocation/retry: `1/0`
- body-level current-pose contact+motion support: `true`
- stable grasp / cap-rim-barrel order / target-IK repair justification: `null`
- `g0a_pass=false`

Video completion marker가 `pass=false`여서 supervisor phase contract와 overall pass가
FAIL했다. `d362_automated_summary.json`, manual PASS artifact, finalize/completion summary는
생성하지 않았다. D362 output은 supervisor summary까지 정확히 33개이며, 이 33개를
immutable failure evidence로 동결한다. D362 physics를 재실행하거나 artifact를 덮어쓰지
않는다. Commit/push도 하지 않았다.

다음 가장 좁은 후보는 별도 승인 D363
`[d362_trace_replay_1080_and_isaac_render_sync_repair]`다. 범위는 observability-only다.

1. D362 immutable JSON/prefix만 읽어 exact `1920x1080` H.264/yuv420p replay를 새
   forward-only output에 재생성·검증한다. Physics/q5는 재실행하지 않는다.
2. precommand/contact/motion/final 네 recorded state를 zero-step으로 표시하고, 등록할
   단 하나의 Fabric flush 경로 전/후 rendered cylinder pose를 trace pose와 수치
   대조한다. 이 단계도 controlled physics step과 q5 science sample은 0이다.

D363에서 target/IK/path, asset/decomposition/gate/material/mass/actuator/solver/physics,
cap-rim science, settle/grasp/G0a/RL/PPO/VLA를 변경하거나 실행하지 않는다. D363 뒤
cap/rim discriminator나 target/IK repair를 논의하려면 다시 별도 명시 승인이 필요하다.
