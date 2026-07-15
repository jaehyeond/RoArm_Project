# D351 attempt2 — zero-step representation-binding timeline pause repair

날짜: 2026-07-15 KST  
상태: attempt2 1회 실행 / `D351_ATTEMPT2_PRE_SCIENCE_RUNTIME_LONG_RUN_STOP` / 과학 판정 `null` / D352 미승인  
이번 attempt의 신규 과학 변수: `[]`  
이번 attempt의 신규 물리 변수: `[]`  
상속한 D351 case 변수:
`[moving_jaw_actual_contact_surface_binding, frozen_pose_q5_closure_sweep]`

## 1. 무엇이 실제로 일어났는가

사용자가 승인한 D351
`[zero_step_moving_jaw_closure_geometry_discriminator]`의 첫 validate는 실제
`headless=False` Isaac GUI를 시작했고, D348 방식의 callback-topology live 표현을
`link5 64 + gripper_link 64 = 128/128`로 결합했다. 그러나 이 결합 뒤 prerequisite에서
Isaac timeline이 `PLAY`인 것이 관찰되어 다음 오류로 안전 중단했다.

```text
RuntimeError: D351 runtime prerequisites STOP: {'counter_after_reset_zero': True,
'timeline_paused': False, 'corrected_d348_128_of_128': True,
'live_binding_64_plus_64': True, 'raw_source_contract': True, 'launcher': True}
```

정확한 표현은 “builder 내부의 어느 호출이 timeline을 재생시켰는지 확정했다”가 아니라,
“representation build 뒤 prerequisite에서 예상 밖 `PLAY`를 관찰했다”이다.

attempt1은 q5 계산 함수가 호출되는 prerequisite를 통과하지 못했다. 원 harness에서
q5 evaluation은 prerequisite PASS 뒤에만 생성되며, attempt1에는 measurement, sweep
CSV, moving-surface binding, Viewer capture, RRD, automated summary가 없다. 따라서
attempt1의 q5 geometry sample은 `0`, 과학 판정은 `없음`, controlled physics step은
사후 추정으로 `0`이라 고쳐 쓰지 않고 exception의 `null`을 그대로 보존한다.

## 2. attempt1 불변 증거

- 원 harness:
  `sim_scripts/cyl34_top_view_d351_zero_step_closure_geometry.py`
  SHA-256 `3c4501885af7590f5883b36666c984ce88728a24d40451ea0a600660a386107d`
- 원 session:
  `claudedocs/session_20260715_grasp_g0a_d351_zero_step_closure_geometry.md`
  SHA-256 `20367375e05ce8cffb47f86ff0c1645a3544f5bf62516fe2e16a98919c356a06`
- attempt1 root:
  `claudedocs/runtime_logs/grasp_track/g0a_d351/`
- parameter freeze:
  `98b5778e826d411f37606dd724093a1ff292040d8c1d350db3781508735502e2`
- preregistration:
  `d0639f51485b96395de88b0942ea4af13a768f31db89a400df7af97a25df1456`
- validate preflight:
  `3e3172ff595bdc48b4216ab0bbb30386a2fdf29f0786ab8d950881114d434660`
- live binding 128/128:
  `9bc8d1c95f3c235816eb1c3c11516f3f27416e45b302cf8b6f9d5ee01ad6ec05`
- runtime exception:
  `138097cee4a471b84202572639fd19c0cba6103d5a628d89a2af49bcbde71914`

attempt2 wrapper는 위 다섯 root 파일의 exact inventory/hash, 원 harness/session hash,
attempt1 validate PID 종료, 과학·Viewer 출력 부재를 모두 hard gate로 묶는다. 기존
파일은 수정하거나 덮어쓰지 않는다.

## 3. reactive repair의 정확한 범위

새 출력은 다음 forward-only 경로만 사용한다.

`claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/`

새 wrapper:

`sim_scripts/cyl34_top_view_d351_attempt2_timeline_pause_repair.py`

수리는 representation binding 뒤의 재생 억제에만 한정한다.

1. `/app/player/playSimulations=false`를 playback control로 재확인한다.
2. timeline이 재생 중일 때만 `timeline.pause()`를 호출한다.
3. 이 bridge에서는 `simulation_app.update`, physics step, `dt>0` update를 호출하지 않는다.
4. 자산, `64+64` 분해, raw/live 표면 정의, q0-q4/object, q5 grid, root-search,
   target/IK/path, 허용값, 재질, 질량, 구동기, physics configuration을 바꾸지 않는다.

다음 다섯 시점의 상태를 Float32 bits 및 exact scalar로 기록한다.

1. reset/초기 pause 뒤, live binding 직전
2. live binding 직후, 재-pause 직전
3. live 재-pause 직후
4. live+raw binding 뒤, 최종 재-pause 직전
5. 최종 재-pause 직후

모든 시점에서 custom step counter `0`, timeline time 불변, Isaac
`current_time/current_time_step_index` 존재 및 불변, 관절 6개와 원통 position/quaternion
Float32 bits 불변, q5 evaluation call count `0`이어야 한다. 초기·최종 timeline도
paused여야 한다. 이 전체 bridge contract가 raw prerequisite에 직접 연결되므로 하나라도
실패하면 q5 sample 전에 STOP한다. 기존 per-sample 및 Viewer zero-step guard는 유지한다.

## 4. 이미 알려진 observability Boolean 수리

원 D351에는 D350 attempt1과 같은 집계 결함이 있다. `asset_write=false`는 “asset을
쓰지 않았다”는 필수 상태인데, 이를 그대로 `all(immutability.values())`에 넣으면 다른
검사가 모두 참이어도 automated result가 항상 거짓이 된다.

attempt2는 원 validator가 이 exact 패턴으로 exit `2`를 만든 경우에만 initial automated
JSON/Markdown 쓰기를 보류하고 다음을 별도 repair JSON으로 증명한다.

- `asset_write is False`
- 나머지 positive immutability checks 전부 true
- science result가 기록됨
- overlay, Rerun, real Viewer, launcher, timeline repair 모두 true
- controlled physics steps `0`

그때만 `asset_write_forbidden_and_absent=true`로 극성을 바로 해석해 automated aggregate와
exit code를 맞춘다. geometry, science verdict, 거리, 허용값, Viewer/Rerun payload는
재계산하거나 변경하지 않는다. 다른 실패 패턴에는 이 수리를 적용하지 않는다.

## 5. 실행·중단 경계

- attempt2는 prepare 검토 뒤 validate `1회`만 허용한다.
- attempt2 실패 시 자동 재시도, attempt3, gate 완화, target/IK 변경을 하지 않는다.
- 성공한 뒤에만 original-resolution 실제 Isaac Viewer/Rerun 수동 검사와 finalize를 한다.
- target/IK geometry repair, settle, 10-trial, G0b, RL/PPO, ladder promotion은 계속 금지한다.
- `g0a_pass=false`를 유지한다.
- commit/push를 하지 않는다.

attempt1의 실패는 AGENTS.md Session progress rule이 허용하는 reactive control hardening의
직접 원인이다. attempt2의 OPEN→CLOSED q5 sweep 자체는 결과에 따라 현재 자세를 유지할지
향후 별도 target/IK repair를 추천할지가 바뀌는 perturbation evaluation이다.

## 6. 현재 비과학적 작업공간 경계

사용자 소유
`claudedocs/lab_meeting/20260715/d334_collision_table/`의 README/HTML/ignored PNG가
attempt2 준비 중 외부에서 다시 갱신되는 것이 관찰됐다. 이 sidecar는 과학 입력이
아니며 수정하지 않는다. 변경이 끝난 뒤 시간 간격을 둔 안정성 표본을 확인하고,
attempt2 prepare→validate→finalize 동안만 현재 hash와 git role을 read-only로 동결한다.
안정되기 전에는 Isaac을 실행하지 않는다.

## 7. prepare 결과

attempt2 prepare는 Isaac을 시작하지 않고 PASS했다.

- parameter freeze:
  `d351_parameter_freeze_audit.json`, SHA-256
  `98b5778e826d411f37606dd724093a1ff292040d8c1d350db3781508735502e2`
  — attempt1과 exact 동일
- preregistration:
  `d351_preregistration.json`, SHA-256
  `eb05905a683842693dd5a0f7dff717cdae9c8bc4d9d6c51a9e5e7b21eba64fc1`
- prepare prechecks: 전부 PASS
- environment: registered Python, `numpy==1.26.0`, `psutil==5.9.8`,
  `rerun-sdk==0.34.1` PASS
- attempt1 immutability, active sidecar, Git scope, source/input/state hash PASS
- attempt2 신규 과학/물리 변수 `[] / []`

## 8. validate에서 실제로 관찰한 순서

1. 사용자 승인 범위의 validate를 딱 한 번 실행했다.
2. fresh validate PID `1994061`이 실제 `headless=False`, `DISPLAY=:1`, RTX 4090
   Isaac GUI를 시작했다.
3. validate preflight는 `20/20 PASS`였다. 파일 SHA-256은
   `035113da2ae94ec7d458d8f5e9a675bdac79f443fb3827f8555dfd4c37166334`다.
4. Kit log는 app ready `13.360s`, Simulation App Startup Complete `15.953s`,
   초기 PhysX/Fabric 활동 `16.393s`까지 기록했다.
5. 그 뒤 live-binding 파일이 생기지 않은 채 프로세스가 계속 실행됐다. 표본 시
   CPU는 평균 `201~216%`, process memory `7.9%`, GPU memory `2482~2486MiB`,
   GPU SM `36%`였고 Python/Kit 및 TBB worker가 활동했다. 따라서 단순 종료·zombie나
   빈 shell 대기는 아니었지만, 정확한 호출 위치는 국소화하지 못했다.
6. 읽기 전용 GDB stack attach는 Ubuntu Yama ptrace 정책으로 거부됐다. attach가
   성립하지 않았으므로 실행 프로세스 상태에는 변화가 없었다.
7. 경과 `01:00:17` 시점에도 attempt2 폴더는 parameter/prereg/preflight 세 파일뿐이었다.
   같은 attempt1에서 preflight→live-binding은 약 `22.83s`였으므로 정상 변동 범위를
   크게 벗어난 pre-science long run으로 판단했다.
8. 사용자에게 이 경계와 무산출 상태를 보고하고 별도 승인을 받아 PID `1994061`에
   `SIGTERM`을 보냈다. Isaac은 signal을 받아 graceful `SimulationApp.close`를 수행했고
   Kit log `3693.302s`에서 종료했다. 프로세스와 GPU context가 사라진 것을 확인했다.
9. 자동 재시도, attempt3, target/IK 변경, gate 변경은 하지 않았다.

## 9. 정량 결과와 증거

외부 종료 감사:

`claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/d351_external_termination_audit.json`

최종 현재 SHA-256:
`af17995b40d5818055388f97e38cbb50f0895f3a2aa4d2cb7f5cf1df3b6166fe`

Kit log:

`/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20260715_171836.log`

- bytes `1,275,110`
- SHA-256 `b4eb319c2b19638f6e263e6b654fb517f494a847e073b573c24d4563e7f72e20`
- shutdown elapsed `3693.302s`

최종 도달 상태:

| 항목 | 결과 |
|---|---:|
| prepare | PASS |
| validate preflight | `20/20 PASS` |
| real Isaac GUI launch | true |
| attempt2 live-binding file | absent |
| five-snapshot zero-step bridge | not reached / null |
| raw binding | not reached |
| q5 geometry evaluation | `0` by program order |
| moving-surface binding / measurement / sweep | absent |
| D351 Viewer capture / RRD | absent |
| controlled physics steps | `null` — 사후 `0`으로 고쳐 쓰지 않음 |
| scientific verdict | `null` |
| `g0a_pass` | false |

q5 evaluation `0`은 시간 추정이 아니라 원 program order로 보장된다. base harness는
live-binding write와 raw prerequisite 뒤에만 counted `_evaluate_q5`를 호출한다.
attempt2 live-binding 자체가 없으므로 q5 evaluation은 시작하지 않았다. 반대로 custom
counter와 simulation clock의 five-snapshot bridge는 파일로 완료되지 않았으므로
controlled physics step을 `0`이라고 소급 주장하지 않고 `null`로 둔다.

## 10. 최종 판정과 일상어 번역

최종 operational 판정은
`D351_ATTEMPT2_PRE_SCIENCE_RUNTIME_LONG_RUN_STOP`이다.

그리퍼가 닫힐 때 원통 옆면을 제대로 만나는지 재어 보려 했지만, 이번에는 그 자를
대기 전 단계에서 Isaac이 한 시간 넘게 내부 작업을 계속했다. 실제 Viewer 창은
열렸지만 D351용 collider 화면이나 거리 계산까지 도달하지 못했다. 그래서 “현재
자세로 잡을 수 있다”도 아니고 “현재 자세는 안 된다”도 아니다.

따라서 target/IK repair를 지지하거나 반대할 새 기하 증거가 없다. 현재 pose closure
geometry는 `null`, target/IK repair justification도 `null`이다. D350/D349 수치와
`g0a_pass=false`는 그대로다.

다음으로 가장 좁은 후보는 별도 승인 D352
`[d351_validate_phase_localization_watchdog]`다. 과학 변수를 추가하지 않고
`_make_runtime_env` 시작/끝, reset 시작/끝, corrected audit, live part `0..127`, bridge
경계에 진행 marker와 wall-clock watchdog를 두어 장기 실행 위치를 먼저 국소화해야 한다.
이 후보는 아직 승인되지 않았으며 구현·실행하지 않는다. target/IK geometry repair,
settle, 10-trial, G0b, RL/PPO, ladder도 계속 막혀 있다. D352는 localization-only이며,
그 결과 뒤 q5 과학 측정을 다시 실행하는 것도 별도 명시 승인이 필요하다.

## 11. Session progress rule 명시적 정당화

이 session은 사용자가 승인한 OPEN→CLOSED zero-step perturbation evaluation을 실제 fresh
GUI validate로 시도했다. 그러나 runtime이 live representation binding artifact 전에서
장기 실행되어 perturbation sample에 진입하지 못했다. 같은 attempt 자동 재시도는
forward-only/one-run 계약을 깨고, target/IK나 과학 gate 변경은 승인 범위 밖이다.
따라서 이번 session에 perturbation 결과가 없는 이유는 실행 회피가 아니라 관찰된
pre-science runtime STOP이며, 안전한 추가 진전은 별도 승인된 phase-localization case가
필요하다. `NO_PPO_PROMOTION`은 본 runtime 실패와 G0a 미통과 때문에 유지한다.
