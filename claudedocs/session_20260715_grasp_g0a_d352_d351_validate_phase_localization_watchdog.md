# D352 — D351 validate phase localization watchdog

Date: 2026-07-15 KST

## 1. 승인 질문과 신규 변수

승인된 case는 `D352 [d351_validate_phase_localization_watchdog]`다. D351 attempt2가
실제 RTX 4090 GUI를 띄운 뒤 live-binding 산출물 전에서 3693.302초 장기 실행된 위치를,
과학 측정 전에 forward-only marker와 외부 watchdog로 한 번만 국소화한다.

이번 case의 신규 변수:

1. `durable_phase_marker_stream`
2. `external_bounded_wall_clock_watchdog`

신규 과학 변수는 `[]`, 신규 물리 변수는 `[]`다.

## 2. 사전등록 범위

- 출력: `claudedocs/runtime_logs/grasp_track/g0a_d352/`
- worker는 실제 `headless=false`, `DISPLAY=:1`, `cuda:0` GUI 경로를 사용한다.
- marker: AppLauncher, `_make_runtime_env`, reset, corrected audit, live part
  `0..127` start/end, D351 attempt2 zero-step bridge의 각 전후, full live payload
  deepcopy/serialization/write, raw binding, close.
- watchdog: 마지막 유효 marker 뒤 `120s` 무진전 또는 worker 총 `300s`; timeout 시
  `SIGUSR1` faulthandler 5초, `SIGTERM` 30초, 생존 시 `SIGKILL`; 자동 재시도 없음.
- GPU/CPU telemetry: 명목 1초 간격 device active-time utilization, memory activity/usage,
  clocks, P-state, power, temperature와 worker CPU/RSS/thread/I/O를 marker 시간축에 묶는다.

`nvidia-smi`의 GPU/SM 사용률은 sampling 구간의 GPU active time이지 warp occupancy가
아니다. 현재 장치는 compute capability `8.9`, `76 SM`, warp `32`, SM당 최대
`1536 threads`라 최대 resident capacity는 `48 warps/SM`이지만, 이는 달성 occupancy가
아니다. D352에서는 profiler, kernel replay, clock/power/persistence 설정 변경을 하지 않는다.
`nvidia-smi` query와 part별 durable `fsync`는 실행 timing을 교란할 수 있으므로 D352
wall time은 D351 성능의 bit-exact 재현이나 benchmark 권위가 아니다.

## 3. 과학 진입 방지와 동결 경계

D352 전용 runner는 D351 base `_run_validate`로 낙하하지 않는다. 원 `_evaluate_q5`는
fail-closed trap으로 바꿔 호출 시 원 함수보다 먼저 정지하고, 정상 완료 조건은 trap count
`0`이다. worker는 five-snapshot bridge 직후 반환한다.

D351 attempt2의 기존 playback suppression(`/app/player/playSimulations=false`와 timeline
pause)은 같은 순서로 재현하지만, D352가 새 playback/physics-solver 설정을 추가하지 않는다.
target/IK/path, asset/decomposition, gate/tolerance, material/mass/actuator/physics 설정,
q0-q4/object, q5 science sample, moving-surface 측정, geometry verdict, controlled physics step,
`simulation_app.update`, Viewer/Rerun, settle, 10-trial, G0b, RL/PPO, VLA, ladder는 금지한다.

bridge가 counter/clock/state 불변을 완전히 증명한 경우에만 D352 controlled physics steps를
`0`으로 기록한다. timeout/예외가 bridge 전에 나면 `null`이다. D351 attempt2의 기존 값은
계속 `null`이며 소급 변경하지 않는다.

## 4. GPU 효율 해석 경계

D351 attempt2의 CPU `201~216%`, GPU SM `36%`, VRAM `2482~2486MiB`는 살아 있는
프로세스 활동만 뜻한다. 이것만으로 GPU idle, CPU bottleneck, warp occupancy를 판정하지
않는다. D352는 stalled phase와 여러 연속 telemetry 표본을 함께 제시한다. causal
bottleneck이나 true occupancy가 필요하면 D352 뒤 별도 profiler 승인이 필요하다.

## 5. Rerun 및 session-progress 예외 정당화

이 case는 geometry, pose, contact, simulation trajectory, sensor synchronization을 판정하지
않는 operational control-flow/file-log localization이다. 따라서 새 RRD/RBL은 만들지 않는다.
D351에서 실제 실패한 장기 실행에 대한 reactive instrumentation이며 과학 perturbation을
일부러 회피하는 case가 아니다. q5 science는 D352 결과 브리핑 뒤 별도 경계에서만 재개한다.

## 6. 실행 전 상태

- Git HEAD/origin: `c2cfa5f41d4c15fec15330cfad38b9b14e4c4f61`
- D351 attempt1/attempt2와 사용자 소유
  `claudedocs/lab_meeting/20260715/d334_collision_table/`는 hash-exact read-only로 고정한다.
- prepare, 정적 검증, 독립 검토 뒤 실제 validate는 정확히 1회만 실행한다.

## 7. 실행 결과

prepare는 한 번 PASS했고 실제 `headless=false`, `DISPLAY=:1`, `cuda:0` validate도
정확히 한 번 실행했다. 자동/수동 재실행은 하지 않았다.

### 7.1 prepare와 GPU exact gate

- preregistration run nonce:
  `23b968cb59e433786d4fc678aac61151`
- harness SHA-256:
  `3f770200d3ca729f106a74ba9e22390d4db5fdd4547e7091cf36ad46a59f65d0`
- parameter / GPU / preregistration SHA-256:
  `3c41109a92398fa9b5f19cb22cb07095a52863686f6f628b57a6bca34e3bb39e` /
  `77770159639b7ab2b0b2a9aae3e109a95564e51d93e2cd491c9259a706b15696` /
  `37843ebbf2708cd664bfa2b0b418ccad76e681da401c90a0d098e18a54430e8f`
- GPU exact gate는 RTX 4090 Laptop, UUID
  `GPU-05b1a3f8-b7cf-dc57-06aa-741fe2daa4b4`, compute capability `8.9`,
  `76 SM`, warp `32`, `1536 threads/SM`, `16376MiB`, driver `580.159.03`,
  Torch `2.7.0+cu128`, CUDA `12.8`을 모두 PASS했다.

원 근거는
`claudedocs/runtime_logs/grasp_track/g0a_d352/d352_gpu_hardware_contract.json`과
`d352_preregistration.json`이다.

### 7.2 marker 실행 순서와 시간

worker marker 기준 실행은 다음처럼 전진했다.

1. AppLauncher: `1.087807255 -> 19.584432350s`, duration
   `18.496625182s` (raw monotonic ns)
2. `_make_runtime_env`: `19.647432649 -> 22.876241273s`, duration
   `3.228808587s` (raw monotonic ns)
3. reset: `22.883068045 -> 22.921419065s`, duration `0.038351051s`
4. corrected audit: `22.928191508 -> 23.024816099s`, duration
   `0.096624583s`, PASS
5. live builder: `23.055121198 -> 24.937441036s`, duration
   `1.882319808s`; part start/end `128/128`, link5 `64/64`, gripper_link
   `64/64` subchecks PASS
6. live payload deepcopy/serialization/write: `25.003798837 -> 25.028278999s`,
   duration `0.024480171s`
7. raw binding: `25.034949915 -> 26.972454659s`, duration
   `1.937504730s`, PASS
8. localization summary write: process elapsed `27.093244882s`
9. `inner.close`: `27.099930123 -> 27.701076172s`; Kit shutdown log
   `27.925s`

따라서 D351 attempt2의 `3693.302s` 장기 실행은 재현되지 않았다.
`_make_runtime_env`, reset, corrected audit, live builder, serialization, raw binding 중
어느 하나가 매번 정지하는 deterministic blocker라는 가설은 이 1회 재현에서
지지되지 않는다. 그러나 D351 attempt2에는 phase marker나 stack dump가 없으므로 그
과거 1회의 정확한 함수-level 원인은 여전히 `null`이다.

원 marker SHA-256은
`09371d795a3e0214e3ddf335e7ff4a9bf78955c5b51ad85048e7b1700de389ce`다.

## 8. 실제 Isaac timeline 실패 원인

다섯 snapshot은 정확한 순서로 모두 기록됐다. 모든 snapshot에서 다음이 동시에
성립했다.

- `timeline_playing=true`
- `/app/player/playSimulations=false`
- custom step counter `0`
- timeline time `0.029999999329447746` 불변
- SimulationContext clock
  `{current_time=0.009999999776482582, current_time_step_index=2}` 불변
- joint/object Float32 bits exact 불변
- q5 evaluation count `0`

D351 attempt2의 `_pause_without_update`는 live 뒤와 raw 뒤 각각 최대 세 번
`timeline.pause()`를 호출했지만, Kit frame update나 `Timeline.commit()` 없이 즉시
`is_playing()`을 읽었다. 두 event 모두 `playing_before=true`, `playing_after=true`,
`pause_interventions=3`, `timeline_paused_after=false`였다.

설치된 `omni.timeline-1.0.14+69cbf6ad`의 로컬 공식
`docs/USAGE_PYTHON.md`는 timeline state change가 **다음 frame**에 적용되므로 즉시
조회가 이전 committed state를 읽는다고 명시한다. 같은 패키지 `_timeline.pyi`는
`Timeline.commit()`이 pending state를 적용하고 callback을 호출한다고 명시한다.
파일 SHA-256은 각각
`3e3ad641192893c47d9bf09c73b3a1f220dfbefb5b8f754fc9b83c8e3f85a6dd`와
`c5a431d83c24de23aefca0912ef819ae2f3322418264b81aba5279d4fe4ac35e`다.

따라서 현재 q5 진입을 막는 원인은 GPU나 collider binding이 아니라
**pause request가 pending인 상태에서 commit/next-frame 없이 즉시 검사한 control
contract**다. 반복 `pause()`는 pending request를 적용하지 않는다. 반대로
`simulation_app.update()`나 `forward_one_frame()`은 zero-step 범위를 깨뜨릴 수 있어
승인 없이 해결책으로 쓰지 않는다. `Timeline.commit()`도 아직 실행하지 않은 후보이며,
callback이 state/time/step에 미치는 영향을 별도 zero-step case에서 증명해야 한다.

bridge SHA-256은
`26a05d8c76ceaf83a0ebf57324b50c7853d38ce2e6bd58c25e4484c13f9a0036`다.

## 9. verdict 분류 교정

원 localization summary는
`D352_LOCALIZATION_CONTRACT_FAIL_STOP`을 기록했다. supervisor는 timeout도 아니고
localization PASS도 아니면 모두 `D352_LOCALIZATION_EXCEPTION_STOP`으로 묶는
catch-all 분기를 사용했다. 이번 worker는 exit `0`, watchdog 미작동, runtime-exception
field `null`, runtime-exception 파일 없음이고 localization summary를 정상 기록했다.
그러므로 supervisor의 `EXCEPTION_STOP` 문자열은 실제 예외 증거가 아니라 후처리
분류 오류다.

원 산출물을 수정하지 않고 별도
`d352_postrun_classification_audit.json`에서 다음처럼 교정했다.

`D352_LOCALIZATION_COMPLETE_TIMELINE_PAUSE_PENDING_STATE_STOP`

이 교정 audit의 SHA-256은
`92c186a7a4175101e7a3890f6bedf4cb6125bc5a78f13f38b79004a9b6035594`다.
D352 case PASS는 여전히 `false`다. 이는 geometry FAIL/PASS가 아니라 q5 전
control-contract STOP이다.

## 10. GPU/CPU telemetry 해석

31개 표본 모두 valid였고 invalid `0`, UUID mismatch `0`이었다. 표본 시작 간격
min/mean/max는 `0.999856593 / 1.000005827 / 1.000248643s`였다.

- device GPU active-time: min/mean/max `0 / 3.870967742 / 15%`
- memory utilization: `0 / 0.193548387 / 2%`
- VRAM: `2052 / 3863.967742 / 7430MiB`
- SM clock: `210 / 1785.483871 / 2385MHz`
- power: `7.04 / 32.590645 / 44.17W`
- worker CPU: `4.0 / 127.935484 / 724.7%`; `100%`는 논리 CPU 약 1개

이는 장치 전체 `nvidia-smi` active-time이며 achieved warp occupancy가 아니다. D352는
single-env GUI startup + zero-step serial audit/binding이고 q5 sweep, batched physics,
training이 없으므로 76 SM을 채울 GPU workload 자체가 없다. 이 case에서 occupancy를
높이려고 env 수, physics, renderer, solver, batch를 바꾸는 것은 동결 범위 위반이며
pending timeline state의 원인도 해결하지 않는다. 외부 process/desktop/renderer가
device 표본에 섞일 수 있어 CPU/GPU bottleneck 인과도 주장하지 않는다.

원 telemetry SHA-256은
`04dc29f902afc1ef3f142d424d0a98918defc735aa70ce5cfb698824b5684a2c`다.

## 11. 최종 과학 경계

- D352 q5 evaluation count: `0`
- D352 controlled physics steps: bridge 미완료이므로 `null`
- D351 attempt2 q5 count / controlled steps: `0 / null` 그대로
- moving-surface measurement / q5 sweep / Viewer / RRD / RBL: 없음
- geometry PASS/FAIL, current-pose support/rejection, grasp feasibility,
  target/IK repair justification: 모두 `null`
- `g0a_pass=false`
- D351 attempt1/attempt2 및 사용자 sidecar hash exact, commit/push 없음

supervisor / localization / postrun audit SHA-256은 각각
`b1bacc589d63d5dff60746c4030635ad100f9a86b701598c1dece159004f70be` /
`2548cadc18b098680b8c5500237e4e363258df99ef96bd29d8327f0955c47d60` /
`92c186a7a4175101e7a3890f6bedf4cb6125bc5a78f13f38b79004a9b6035594`다.

## 12. 다음 승인 경계

가장 좁은 다음 후보는 별도 D353
`[timeline_pause_pending_state_commit_bridge]`다. 신규 변수는
`explicit_timeline_commit_after_pause` 하나만 허용하고, q5 평가 전에 PAUSE와
timeline time, SimulationContext clock, custom step counter, joint/object Float32 bits
불변을 증명한다. q5 science, moving-surface 측정, target/IK/path, asset/decomposition,
physics/renderer/solver 변경은 D353에 포함하지 않는다.

D353가 PASS한 뒤에만 사용자가 이미 밝힌 q5-science 진행 의사를 결과 뒤의 별도 명시
승인으로 확인하고 closure science를 새 forward-only case에서 재개한다.
