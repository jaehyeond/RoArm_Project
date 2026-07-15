# D353 — timeline pause pending-state commit bridge

Date: 2026-07-15 KST

## 1. 승인 질문과 단일 신규 변수

승인된 case는 D353 `[timeline_pause_pending_state_commit_bridge]`다. D352가 확인한
pending PAUSE 요청을 main thread의 명시적 `Timeline.commit()`으로 적용할 수 있는지,
그리고 그 적용이 timeline/world 상태를 한 step도 전진시키지 않는지를 q5 전에 한 번만
검증한다.

이번 case의 신규 변수:

1. `explicit_timeline_commit_after_pause`

신규 과학 변수는 `[]`, 신규 물리 변수는 `[]`다. D352의 외부 `120s` inactivity /
`300s` total watchdog와 GPU/CPU telemetry cadence는 동결 상속한다. marker의 append/fsync
mechanism은 상속하되, 사전등록된 D353 commit phase/attempt/call counter를 read-only로
추가한다. Timeline all-event callback 기록과 exact snapshot도 commit의 결과를 읽는
mutation-free 측정 채널이며 별도 control intervention이 아니다.

## 2. 왜 Isaac Sim 경로가 q5 전에 막혔는가

D352 원 JSON의 다섯 snapshot은 모두 `timeline_playing=true`였지만 custom counter `0`,
timeline time `0.029999999329447746`, SimulationContext
`{current_time=0.009999999776482582, current_time_step_index=2}`, joint/object Float32
bits가 불변이었다. live/raw 경계는 각각 `timeline.pause()`를 세 번 요청했으나 frame이나
commit 없이 즉시 이전 committed state를 읽었다.

설치된 `omni.timeline 1.0.14`는 state 변경이 다음 frame에 적용된다고 설명하고,
`Timeline.commit()`은 pending state 전부를 즉시 적용하면서 callback을 호출한다고
명시한다. 따라서 현재 확정 원인은 GPU 부족이나 collider 장기 binding이 아니라
**deferred state를 적용하기 전에 동기식 PLAY 검사를 한 control contract**다.

로컬 공식 근거:

- `.../omni.timeline-1.0.14+69cbf6ad.../docs/USAGE_PYTHON.md:12-29`
- `.../docs/FRAME_INTEGRITY.md:70-88`
- `.../omni/timeline/_timeline.pyi:299-314`
- `.../omni/timeline/tests/tests.py:296-305`

## 3. 사전등록 실행 경로

- Base Git HEAD/origin: `1f235b8a310afeb9f4f6734d69aba2a5430b7602`.
- 출력: `claudedocs/runtime_logs/grasp_track/g0a_d353/`.
- 실제 worker: `headless=false`, `DISPLAY=:1`, `cuda:0`, seed `33201`.
- prepare 뒤 effective validate는 정확히 1회다. 자동/수동 same-path retry는 없다.
- D352의 source/hash, AppLauncher, `_make_runtime_env`, reset, corrected D348 audit,
  live `0..127`, payload O_EXCL write, raw binding, five-snapshot bridge, cleanup 순서를
  q5 직전까지 재사용한다.
- initial reset pause는 기존 1회 conditional request를 유지한다. live/raw는 D351
  attempt2의 기존 3-iteration pause helper를 그대로 호출한다.
- 실제 pause request가 1회 이상 발행된 boundary에서만 바로 뒤에 `Timeline.commit()`을
  정확히 1회 호출한다. 이미 `(is_playing,is_stopped)=(false,false)`인 boundary에는 bare
  commit이나 redundant pause를 추가하지 않는다.

## 4. commit-only PASS 계약

각 boundary는 before-pause, post-pause/pre-commit, synchronous post-commit을 기록한다.
최소 한 boundary, 사전등록상 initial boundary는 다음 discriminating transition을 보여야 한다.

1. before: `is_playing=true`, `is_stopped=false`
2. pause request 뒤 commit 전: `true/false` 그대로
3. commit 뒤: `false/false` — STOP이 아니라 PAUSE
4. initial 단일 pause request의 commit callback delta: 정확히 PAUSE 1개,
   main-thread callback, tick/PLAY/STOP 없음

live/raw의 상속 helper가 예기치 않게 PLAY를 다시 관찰해 pause request를 여러 번 발행한
경우에는 설치 API가 coalescing 개수를 보장하지 않으므로 callback은 `1..request_count`개를
허용한다. 단 전부 같은 commit 호출 구간 안, 같은 phase/caller main thread의 PAUSE여야 하며
전체 callback stream과 세 boundary-local slice가 한 행도 남김없이 exact하게 일치해야 한다.

`Timeline.commit()`은 pause만이 아니라 모든 pending state와 callback을 flush하므로 다음
sentinel이 before/pending/post와 canonical five snapshots 전체에서 exact해야 한다.

- custom `_sim_step_counter == 0`
- timeline current time Float64 bits와 current tick 불변
- SimulationContext `current_time` / `current_time_step_index` exact 불변
- timeline start/end/tentative/time-code/tick/loop/auto-update/preroll/zoom metadata 불변
- director `None`, commit call과 callback 모두 main thread
- 6-joint, object position/quaternion Float32 bits/SHA exact 불변
- `/app/player/playSimulations`는 모든 권위 snapshot에서 readable이며 exact `false`
- geometry evaluator count `0`, q5 science count `0`, q5 trap count `0`
- corrected D348 `128/128`, live trace `128 start + 128 end`, body `64+64`, raw PASS
- D352 첫 pending-PLAY snapshot과 D353 initial pre-commit의 time/clock/joint/object bits exact
- D352 raw payload exact, D351/D352/input/source/user-sidecar hashes와 inventory exact
- before/pending/post 9개와 canonical 5개, 총 14개 snapshot의 단일 global mutation baseline exact
- prereg 뒤 harness/START/session hash와 성공 output inventory를 final supervisor까지 재검증

전체 prerequisite와 artifact hash binding까지 PASS할 때만 D353 controlled physics steps를
`0`으로 기록한다. 그 전에 timeout, 예외, director/non-main-thread, callback 추가 event,
PAUSE/불변량/raw/live/hash/discrimination 실패가 하나라도 있으면 `null`이다. D351 attempt2와
D352의 기존 `null`은 소급 변경하지 않는다.

## 5. q5 fail-closed 경계

D353는 `d351._evaluate_q5`와 attempt2에 저장된
`d351a2._ORIGINAL_EVALUATE_Q5` alias를 모두 원 함수보다 앞선 trap으로 교체한다. 호출 시
marker를 남기고 즉시 예외로 정지한다. base/attempt2 validate, counted evaluator, q5 grid,
direct state write는 AST gate에서 금지한다. 정상 PASS와 정상 contract FAIL 모두 q5 science
evaluation count와 trap count가 `0`이어야 한다.

## 6. 동결 및 금지 항목

asset/decomposition, target/IK/path, q0-q5/object, gate/tolerance, material/mass/actuator,
physics/solver/renderer 설정을 변경하지 않는다. `simulation_app.update()`, Kit next-update,
`forward_one_frame()`, rewind, `commit_silently()`, `inner/sim.step`, render, physics step은
금지한다. moving-surface measurement, q5 sweep, geometry/current-pose/grasp 판정, Viewer,
RRD/RBL, settle, ten-trial, G0b, RL/PPO, VLA, ladder도 없다.

GPU는 D352와 같은 RTX 4090 Laptop exact gate를 통과해야 하지만 clocks, power,
persistence, profiler, env 수, batch, renderer, solver를 바꾸지 않는다. 이 single-env
zero-step control case에는 76 SM을 포화시킬 batched physics workload가 없으므로
`nvidia-smi` active-time을 warp occupancy나 PASS threshold로 해석하지 않는다.

## 7. Rerun 및 session-progress 정당화

이 verdict는 geometry/pose/contact/trajectory/sensor time을 해석하지 않고 Timeline API
state와 canonical no-mutation sentinel만 판정한다. joint/object bits는 공간 판정이 아니라
변경 감시값이다. 따라서 D353는 새 RRD/RBL을 만들지 않는다.

또한 D353는 임의 control hardening이 아니다. D351 perturbation 진입 실패 뒤 D352가 직접
관찰한 pending-state defect에 대한 reactive single-variable perturbation이며, PASS/FAIL이
다음 q5 승인 가능 여부를 실제로 바꾼다. 따라서 '결정을 바꿀 수 없는 validation'이 아니다.

## 8. 사전등록 verdict taxonomy

1. q5 trap: `D353_Q5_SCOPE_BREACH_STOP`
2. watchdog: `D353_PHASE_WATCHDOG_STOP`
3. supervisor/worker validate preflight: `D353_VALIDATE_PREFLIGHT_STOP`
4. commit precondition: `D353_TIMELINE_COMMIT_PRECONDITION_STOP`
5. 실제 runtime exception: `D353_RUNTIME_EXCEPTION_STOP`
6. 완료된 bridge contract FAIL: `D353_TIMELINE_COMMIT_BRIDGE_CONTRACT_FAIL_STOP`
7. bridge PASS 뒤 cleanup/telemetry/process/hash/inventory gate FAIL:
   `D353_POST_BRIDGE_OBSERVABILITY_STOP`
8. summary 없는 비정상 종료: `D353_ABNORMAL_EXIT_STOP`
9. 전체 PASS: `D353_TIMELINE_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE`

exit code `0`, GPU activity, 또는 마지막 `SimulationApp.close:start` marker만으로 PASS나
runtime exception을 만들지 않는다. cleanup start 뒤 Python return marker가 없어도 worker와
process group이 종료되고 cleanup exception이 없으면 이를 active stall로 분류하지 않는다.

모든 outcome에서 scientific/geometry/current-pose/grasp/target-IK verdict는 `null`,
`g0a_pass=false`다. D353 결과 브리핑이 끝난 뒤에만 사용자가 새 forward-only q5 closure
science case를 별도 문장으로 승인할 수 있다.

D353 summary에는 D353의 현재값을 `d352_*` alias로 쓰지 않고 controlled-step 후보만
기록한다. 상속 D352 raw classifier의 PASS는 D353 판정에 사용하지 않으며, D351
attempt2/D352 controlled-step 역사값은 계속 `null`로 보존한다. D353
`controlled_physics_steps=0` 권위는 summary O_EXCL/fsync와 재읽기 exact가 끝난 뒤 생성한
별도 `d353_timeline_commit_bridge_attestation.json`에서만 열린다.

## 9. 실행 결과

### 9.1 실행 순서와 1회성

1. 사전등록 뒤 `--stage prepare`를 한 번 실행했다. parameter/GPU/preregistration
   세 artifact만 생성했고 모두 PASS했다. prepare process nonce는
   `5463503bcf7d5f6a16528e7d1ce89997`, 사전등록된 validate run nonce는
   `4126d64957e5851e893c7a30d4724951`이다.
2. 정적 AST/source/hash/output-path gate와 세 독립 read-only review를 통과한 뒤 실제
   `headless=false`, `DISPLAY=:1`, `cuda:0` validate를 정확히 한 번 실행했다.
3. worker PID `3168880`은 exit `0`으로 reap됐고 PID/process group 모두 사라졌다.
   watchdog, cleanup signal, runtime exception, automatic retry는 모두 없었다. 최종 output은
   예상된 성공 경로 17 files와 exact 일치했다.
4. 실행 당시 frozen source hash는 START
   `f31444347a824c7ee0d4bbf472241c50988a98155dcd84e21539cffd3ffff34b`, 이 session
   `b08890805d8d771057326509ed60f9c77829e240a254e2761928ea42c32d8dce`, harness
   `ab37141d721f5ca9571e9008a065344b3fb818ac9164fd56cda3c5617952cda9`였고 final
   supervisor까지 exact였다. 이 결과 append로 START/session의 현재 hash가 달라지는 것은
   의도된 사후 state-doc 갱신이며 실행 당시 frozen contract를 바꾸지 않는다.

Supervisor/worker validate preflight는 각각 `25/25`, `35/35` PASS였다. raw monotonic
timing은 AppLauncher `13.277207705s`, `_make_runtime_env` `3.171491627s`, reset(그 안의
commit 포함) `0.061736489s`, explicit commit 구간 `0.012992069s`, corrected audit
`0.095320871s`, live builder `1.833469931s`, raw binding `1.907043553s`였다. bridge
authority는 worker elapsed `21.847929378s`에 완료됐고 `SimulationApp.close` start는
`22.440568720s`였다. 이후 exit `0` process-terminal audit가 PASS했다. marker는 supervisor
1 + worker 303 = `304/304` valid, invalid `0`이었다.

### 9.2 단일 `Timeline.commit()` 판정

`d353_timeline_commit_event_contract.json`은 PASS했다. explicit commit attempt/call과
discriminating transition은 각각 `1/1/1`이었다.

- before: `timeline_playing=true`, `timeline_stopped=false`
- pause request 뒤 commit 전: `true/false` — D352 pending PLAY를 exact 재현
- commit 뒤: `false/false` — STOP이 아닌 PAUSE
- callback: 정확히 PAUSE 1개, caller와 같은 MainThread
- callback monotonic ns `794216641458132`는 commit call window
  `[794216641029502, 794216641483941]` 안이었다.

세 상태 모두 timeline time `0.029999999329447746`(bits
`000000e051b89e3f`), SimulationContext time/index
`0.009999999776482582/2`(time bits `00000040e17a843f`), custom step `0`, geometry
evaluation `0`, q5 science evaluation/state-write/trap `0`, joint/object Float32 bits가
exact였다. live/raw 경계에서는 timeline이 이미 PAUSE였으므로 pause intervention과 commit이
각각 `0/false`였고 callback도 없었다.

`d353_zero_step_bridge_contract.json`도 PASS했다. before/pending/post 상세 9개와 canonical
5개, 총 14개 snapshot에서 timeline/SimulationContext metadata와 clock, joint/object,
`/app/player/playSimulations=false`, director `None`, main-thread, custom step/q5/geometry
counter가 등록 baseline과 exact였다. `simulation_app.update()`나 physics step은 호출되지
않았다. D352 first pending snapshot match는 전 항목 PASS였고, raw source payload는 D352와
같은 SHA-256
`325004fdc98f01bc01e5534d96ce1e2abe410b47d21029f5961446f2b53f243b`였다.

Corrected live binding은 `128/128`, body subcheck는 link5 `64/64` + gripper_link
`64/64`, raw binding은 PASS였다. summary 자체의 controlled-step 값은 prereg대로 `null`이고
후보만 `0`이었다. summary O_EXCL write/fsync와 SHA-256
`7b8740cb176b3450936e796e6aa7dae72489fe625d08bef71da245e1b0be299a` exact reread 뒤
생성된 별도 attestation만 D353 controlled physics steps를 권위 있게 `0`으로 확정했다.
D351 attempt2와 D352의 역사값은 계속 `null`이다.

### 9.3 Isaac Sim과 GPU 해석

D353는 D351/D352가 막힌 이유가 Isaac 전체나 RTX4090 계산 불능이 아니라, deferred PAUSE를
적용하지 않은 채 이전 committed PLAY를 즉시 읽은 control contract였음을 직접 지지한다.
한 번의 explicit commit으로 PAUSE가 동기 확정됐고, 이후 corrected/live/raw 계약과 종료가
정상 완료됐다. 다만 D351 당시 장기 실행에는 함수 marker/stack이 없으므로 그 과거 실행의
정확한 함수-level 원인을 소급 확정하지 않는다.

GPU telemetry는 `26/26` valid, invalid/UUID mismatch `0/0`이었다. GPU utilization
min/mean/max `0/2.6923076923/21%`, VRAM `2052/4013/7437MiB`, SM clock
`210/1704.8077/2385MHz`, power `6.63/30.9292/44.09W`, temperature
`48/50.6538/52C`, worker CPU `9.3/153.644/740.2%`였다. 이는 device active-time이며
warp occupancy 측정값이 아니다. single-env zero-step control-only case에는 76 SM을 채울
batched physics workload가 없으므로 낮은 GPU 사용률은 병목 또는 설정 실패의 증거가 아니며,
GPU/renderer/solver/env 설정을 바꾸지 않았다.

worker log에는 `[Error] Failed to clone in Fabric` 한 줄과 관련 visual-path warning이 있었지만
traceback/runtime exception은 없었다. 그 뒤 corrected audit, live `128/128`, raw, bridge와
terminal audit가 모두 PASS했으므로 이 run에서는 실패 권위가 아니었다. 일반적으로 무해하다고
확대 해석하지 않으며 후속 science artifact에 실제 결손이 생기면 별도 원인으로 다시 판정한다.

### 9.4 최종 verdict와 과학 승인 경계

최종 operational verdict는
`D353_TIMELINE_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE`다. final supervisor와 commit
attestation SHA-256은 각각
`65c57e69f017d7d7afbb5fd03b10b56e87bb1bbc442b1351a25c18a0a55a31a5`,
`4758e9b09b3298ae0dd292f327bb37b474a624d3f0190629968c55cb091393d5`다.

q5 science evaluation/state-write/trap은 `0/0/0`이고, moving-surface measurement, q5
sweep, geometry/current-pose/grasp 판정, target/IK repair justification, Viewer, RRD/RBL은
전부 미실행/null이다. `g0a_pass=false`; D350이 마지막 scientific + observability case다.
D353는 immutable이며 다시 실행하거나 덮어쓰지 않는다. 새 forward-only q5 closure-science
case는 이 결과 브리핑 뒤 사용자의 새 명시 승인과 별도 사전등록을 받아야만 시작할 수 있다.
