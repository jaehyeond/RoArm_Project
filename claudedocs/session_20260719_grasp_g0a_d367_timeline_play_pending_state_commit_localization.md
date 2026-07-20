# D367 — timeline PLAY pending-state commit localization

상태: 실행 전 사전등록. Actual worker `0`, retry `0`.

이번 case의 신규 변수:

1. `explicit_timeline_commit_after_raw_play_request`

## 1. 무엇을 왜 확인하는가

D366은 PAUSED 기준점에서 raw `timeline.play()`를 한 번 요청했지만, 바로 뒤
`is_playing()`이 `false`여서 cylinder pose write 전에 안전정지했다. D366은 요청 직후의
raw `(is_playing,is_stopped)` tuple과 timeline event를 직접 보존하지 않았으므로, 그 결과를
“pending PLAY가 확정됐다”라고 승격할 수 없다.

D367의 질문은 하나뿐이다. 같은 D366 환경에서 raw `timeline.play()`가 현재 frame 안에서는
pending으로 남고, main thread의 explicit `Timeline.commit()` 한 번이 physics frame을
진전시키지 않은 채 그 PLAY만 적용하는가?

이것은 control API의 적용 경계를 검증하는 case다. Cylinder 접촉, q5 closure, grasp,
PhysX→Fabric/Hydra 전달 또는 G0a 과학 판정이 아니다.

## 2. 승인 범위와 commit 카운터의 정직한 분리

사용자는 2026-07-19 KST에 D367
`[timeline_play_pending_state_commit_localization]` zero-step control-only case를 승인했다.

- actual `headless=false`, `DISPLAY=:1`, `cuda:0` worker: `1`
- automatic retry: `0`
- raw PLAY request: `1`
- D367 판별용 PLAY `Timeline.commit()`: attempt/call/return `1/1/1`
- cylinder pose write / controlled step / public forward: `0/0/0`
- q5 science sample / q5 target update / contact query: `0/0/0`

새 process에서 D366과 같은 `(playing,stopped)=(false,false)` 기준점을 만들려면 D353에서 이미
검증된 초기 PAUSE request+commit을 상속해야 한다. 실제 D366도 이 초기 PAUSE commit을
`1`회 사용했다. 따라서 숨기지 않고 다음처럼 별도 카운터로 사전등록한다.

- inherited initial PAUSE commit: `1` — 동결 prerequisite, controlled PLAY window 밖
- D367 discriminating PLAY commit: `1` — 이번 case의 유일한 신규 변수
- worker 전체 raw `Timeline.commit()` 합계: `2`

“전체 commit 1회”라고 쓰면 거짓이다. 반대로 초기 PAUSE와 신규 PLAY를 한 commit에 섞으면
D366 기준점을 재현하지 못하는 새 confound가 된다. Terminal cleanup에서는 pause/stop,
추가 commit, `commit_silently()`를 호출하지 않는다.

## 3. Git 및 frozen evidence

실행 전 로컬 교차검사 기준:

- `HEAD == origin/master == 9f956a42db1bb43c817ffe435a4e9698707049f1`
- commit subject: `D366`
- boot 시 worktree: clean
- commit/push: 금지

동결 SHA-256:

- D353 harness: `ab37141d721f5ca9571e9008a065344b3fb818ac9164fd56cda3c5617952cda9`
- D353 event contract: `0b1d47671fe31206398961dc18f4d66912ba3fd59cf1634059c326a5e67a0b61`
- D353 zero-step attestation: `4758e9b09b3298ae0dd292f327bb37b474a624d3f0190629968c55cb091393d5`
- D366 harness: `27f6c55e77d62bddca760e0309078029f213f054ee4a7b3537d798188e4a4f61`
- D366 runtime prerequisite: `e7974c5870b9e753a6668b36b78cbcd1aa13a036fd0613a7c7c875efd2ba0c2c`
- D366 worker exception: `4ad206fdf93c4ce6bb31c3ec083419f788bb29b2f858ee19282c93a95e5c2d60`
- D366 completion: `cc061e07fca358d18467255a5ad02a6460513eae6bed5cccc463870b9ecf2f7d`
- D366 correction audit: `2f55c13bbd127654260f318798b2a6a76b0a8dc39cce1efe13502f0137a86312`

D353-D366 evidence tree와 사용자 소유
`claudedocs/lab_meeting/20260715/d334_collision_table/`은 manifest/hash로 전후 대조하며
수정하지 않는다. 새 파일은 `claudedocs/runtime_logs/grasp_track/g0a_d367/`에만 만든다.

## 4. 설치 API 근거

설치된 `omni.timeline` 문서는 상태 변경이 같은 frame 안에서는 deferred되어 raw
`play()` 직후 `is_playing()==false`이고, `commit()`은 pending state와 callback을 즉시
적용한다고 설명한다. `commit()`은 thread-safe하지 않으므로 main thread에서만 호출한다.

고정할 설치 파일 SHA-256:

- `docs/FRAME_INTEGRITY.md`: `1db84fba636fa743bcf98a38561132323ad13fcb89dc91629a06812b123b2e37`
- `omni/timeline/_timeline.pyi`: `c5a431d83c24de23aefca0912ef819ae2f3322418264b81aba5279d4fe4ac35e`
- `omni/timeline/tests/tests.py`: `570b36d310e3f3a307c8a35c38ba277da051e6e1e8fc25da6889e794ad270638`
- Isaac Core `simulation_context.py`: `ebafc6bcb30a454925fe21b96dcdbd4637c922a3fa9d5a6947308c9796ba5028`

Core public `SimulationContext.play()`는 raw `play(); commit()`을 묶고 문서상 내부 physics-handle
propagation step을 경고하므로 D367에서 호출하지 않는다. `app.update()`, next-frame,
`forward_one_frame()`, `sim.step()`, public `forward()`도 모두 금지한다.

## 5. 실제 관찰 순서

1. Frozen D362 environment를 만들고 reset한다. Reset 내부 time/index는 기록하지만 D367
   controlled physics step에는 포함하지 않는다.
2. Timeline event observer를 붙이고, D366 helper를 정확히 한 번 호출해 inherited PAUSE
   request+commit을 수행한다. Before는 `(true,false)`, after는 `(false,false)`, PAUSE event는
   정확히 1개여야 하며 clock/joint/object bits가 변하면 중단한다.
3. Physics callback을 정확히 한 번 등록하고 count/dt가 `0/[]`인지 확인한다.
4. D366과 같이 `/app/player/playSimulations=True`를 한 번 설정한다. 이 setting write 전후에도
   physics clock과 state bits가 불변이어야 한다.
5. `before_play_request` snapshot을 기록한다. Raw tuple은 `(false,false)`여야 한다.
6. raw `timeline.play()`를 정확히 한 번 호출하고 즉시 `post_play_request_pre_commit`을 기록한다.
   예상 tuple은 계속 `(false,false)`이고 callback delta는 `0`이다.
7. 같은 main thread에서 D367 explicit `timeline.commit()`을 정확히 한 번 호출한다.
8. `post_play_commit`과 즉시 canonical reread를 기록한다. Tuple은 `(true,false)`여야 한다.
9. Commit window 안 event는 PLAY(type `0`) 정확히 1개, PAUSE/STOP/time-tick은 0개여야 한다.
10. Worker summary를 O_EXCL+fsync하고 exact reread한다. 별도 attestation이 summary hash와 모든
    no-advance prerequisite를 묶은 뒤에만 `controlled_physics_steps=0`을 권위값으로 기록한다.
11. Cleanup safety로 `playSimulations=False`를 유지하고, observer release, physics callback
    removal, `inner.close()`, `SimulationApp.close()` 각각에 start/end/error marker를 남긴다.
    추가 timeline state request/commit은 없다. Cleanup 진입 직전과 safety-false 직후의 raw
    tuple·clock·joint/object bits를 별도 O_EXCL 파일로 먼저 내구화한다. PLAY request 뒤
    precondition이 실패하면 구조용 “rescue commit”을 호출하지 않고, 같은 marker와 90초
    phase-inactivity/300초 total watchdog로 종료 위치를 한정한다.

## 6. bit-exact no-advance 판정

Before/request/post/canonical snapshot은 다음을 모두 직렬화한다.

- raw `(is_playing,is_stopped)`
- timeline current-time Float64 bits, current tick, metadata
- SimulationContext current time Float64 bits와 step index
- custom `_sim_step_counter`
- physics callback count와 dt 배열
- 6-joint Float32 values/bits/SHA-256
- cylinder position/quaternion Float32 values/bits/SHA-256
- `/app/player/playSimulations` readable/value
- director, caller/callback thread ID와 MainThread 여부

6-joint tensor read는 사용자가 요구한 mutation sentinel이다. q5를 closure geometry나 접촉
의미로 평가하지 않으며 q5 science sample count는 `0`이다.

PASS에는 다음이 모두 필요하다.

- raw tuple: `(false,false) → (false,false) → (true,false)`
- PLAY commit attempt/call/return: `1/1/1`, return value `None`
- commit window의 MainThread PLAY callback: 정확히 `1`
- timeline/SimulationContext/custom counter/callback/joint/object bit 불변
- write/step/forward/q5/target/contact: 전부 `0`
- summary reread+hash attestation, cleanup, watchdog/process-residue gate PASS

PASS verdict는 `D367_TIMELINE_PLAY_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE`다.

Precondition, commit 실행, post PLAY, callback/thread, clock advance, state bit mutation,
scope counter, durable summary/hash, cleanup/watchdog 실패를 서로 다른 fail class로 보존한다.
Bridge가 저장된 뒤 cleanup만 실패해도 전체 D367 PASS로 승격하지 않는다.

## 7. 실패 가능한 negative controls

Prepare는 positive fixture 한 개와 다음 perturbation이 모두 판정을 FAIL로 바꾸는지 확인한다.

- PLAY request/commit `0` 또는 `2`
- before가 이미 PLAY/STOP, request 뒤 즉시 PLAY, post가 PAUSE/STOP
- callback missing/duplicate/wrong PAUSE·STOP·TICK/outside-window/wrong thread
- timeline bit/tick/metadata, SimulationContext time/index, custom counter, callback 변화
- joint/object 한 bit 변화
- setting unreadable/false, director 존재, non-main-thread
- write/step/forward/q5/target/contact/app-update/next-frame가 하나라도 nonzero
- summary hash tamper, cleanup end 누락, watchdog/retry/invocation `2`, process residue

이 대조군은 실제 PASS/FAIL 결정을 바꿀 수 있으므로 Session Progress Rule을 충족한다.

## 8. Rerun 생략 정당화

D367은 geometry, pose 의미, contact, trajectory 또는 sensor synchronization을 판정하지 않는다.
오직 동일 process/frame 안 API Boolean, event, clock, counter, opaque state bits의 bit-exact
control audit다. Rerun Float32 spatial copy는 이 판정을 강화하지 못하고 오히려 새 app/render
update를 유발할 수 있으므로 RRD/RBL/PNG를 생성하지 않는다. 원 JSON/JSONL이 권위다.

## 9. 실행 전 승인 경계

- Actual invocation: 아직 `0`
- D367 output: 아직 생성 전
- D366 one-write/one-step/one-forward 재개: 이번 turn에서 실행하지 않음
- q5/contact/cap-rim/grasp science: 실행하지 않음

D367이 PASS해도 결과를 먼저 보고한다. 이후 physics/Fabric 측정 재개는 새 forward-only case와
별도 명시 승인 뒤에만 가능하다.

## 10. Prepare 결과와 actual invocation

Prepare는 실제 Isaac worker를 열기 전에 끝났고 `18/18` check가 모두 PASS했다. Positive
fixture 한 개와 판정을 실제로 뒤집는 negative control `34/34`도 PASS했다. 주요 gate는
HEAD/origin, cached patch, D351-D366 frozen hash/manifest, D334 sidecar, DISPLAY, RTX 4090
Laptop, GPU/RAM 여유, `numpy==1.26.0`, `psutil==5.9.8`, 설치 API 및 active-callsite 정적
감사였다.

- preregistration SHA-256:
  `03ae93395ff7ca715bd6f960107e6786db35527fb44e70562b0d50f5122eaf17`
- prepare SHA-256:
  `ccc3113a1a1341844e29690381936d3d93bf704ebb9099b1591083987c17d608`
- actual GUI worker: 정확히 `1`; automatic retry: `0`
- worker PID/process group: `2456586/2456586`

Actual worker는 `AppLauncher → _make_runtime_env → reset → inherited PAUSE → physics callback
등록 → playSimulations=true → raw PLAY/explicit commit → durable summary/attestation → cleanup`
순서로 한 번 진행했다. 새 cylinder write, physics step, public forward, q5 또는 contact 경로는
호출하지 않았다.

Sources:

- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_prepare_preflight.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_isaac_invocation_marker.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_worker_phase_markers.jsonl`

## 11. raw PLAY와 explicit commit의 직접 관찰

이번 case에서 `(playing, stopped)` tuple은 다음처럼 기록됐다.

1. D366-equivalent PAUSED baseline: `(false,false)`
2. raw `timeline.play()` 직후, commit 전: `(false,false)`
3. explicit `timeline.commit()` 반환 직후: `(true,false)`
4. 즉시 canonical reread: `(true,false)`

즉 D366의 raw PLAY request가 틀린 명령이었던 것이 아니라, 이 설치 버전에서는 같은 frame의
pending request로 남고 explicit commit이 있어야 재생 상태가 적용된다. 이는 추론이 아니라 네
snapshot의 직접 관찰이다.

Commit 시도/호출/반환은 `1/1/1`, raw PLAY request는 `1`이었다. D366 baseline 재현용 inherited
PAUSE commit `1`은 새 변수와 분리했다. 따라서 worker 전체 `Timeline.commit()`은 거짓으로 `1`이
아니라 `2 = PAUSE 1 + PLAY 1`이다. PLAY commit 호출 구간은 `7,010,253ns`였고, 정확히 한 개의
PLAY callback(type `0`)이 시작 후 `635,400ns`에 같은 `MainThread`
(`ident=139973347566656`)에서 발생했다. Callback은 commit 구간 안에서 `(true,false)`를 봤다.

Contract check는 precondition `9/9`, event `10/10`, main `12/12`, counter `4/4` PASS였다.

Source:

- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_play_commit_contract.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_timeline_event_rows.jsonl`

## 12. 왜 zero-step이라고 확정할 수 있는가

Raw PLAY 전, request 직후, commit 직후, canonical reread 네 지점에서 다음 값이 bit-exact로
같았다.

- timeline time: `0.029999999329447746s`, Float64 bits `000000e051b89e3f`, tick `0`
- SimulationContext time/index: `0.009999999776482582s / 2`, time bits
  `00000040e17a843f`
- custom controlled-step counter: `0`
- physics callback count/dt: `0 / []`
- joint SHA-256:
  `752f9028218e5a3b4f226feb5dbdd4ced84c783742e904bdbdd8d8323f5939f5`
- cylinder position SHA-256:
  `7c92fa4e59547af82b9b3f4d3f659982bdc9df6256eba57de08880aa2e2b5489`
- cylinder quaternion SHA-256:
  `ccaf6f183579497e8bfcd71045c04286fd3c2e60f3641e3eea164b761a4494b7`

Worker summary는 먼저 `controlled_physics_steps=null`과 candidate `0`을 내구화했다. 그 summary를
exact reread하고 SHA-256으로 contract와 묶은 별도 attestation이 `7/7` PASS한 뒤에만 권위값
`controlled_physics_steps=0`을 기록했다. Cylinder write/controlled step/public forward/q5 science/
q5 target/contact query/app update/next frame는 모두 `0`이다.

따라서 제어-다리 subresult는
`D367_TIMELINE_PLAY_COMMIT_ZERO_STEP_BRIDGE_PASS_NO_SCIENCE`다. 이것은 “PLAY 상태를 한 물리
step 없이 적용했다”는 뜻이다. 원통 접촉, cap/rim, 파지, PhysX→Fabric→Hydra 전달을 다시
측정했다는 뜻은 아니다.

Sources:

- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_worker_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_zero_step_attestation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_postrun_classification_audit.json`

## 13. Cleanup 관찰과 전체 completion이 FAIL인 이유

제어-다리 뒤 cleanup 진입 snapshot과 `playSimulations=false` 적용 뒤 snapshot은 모두
`(true,false)`였고 timeline/SimulationContext/custom counter/physics callback/joint/object bits가
그대로였다. Cleanup entry artifact 자체는 `4/4` PASS했다.

단계표식은 다음까지 직접 남았다.

1. Timeline observer release `start/end`
2. Physics callback remove `start/end`
3. `inner.close()` `start/end`, elapsed `0.67826319s`
4. `SimulationApp.close()` `start`
5. worker log의 `[17.608s] Simulation App Shutting Down`

그 뒤 `SimulationApp.close():end`, `worker_finally_complete`, 그리고 그 end marker로 만드는
`d367_cleanup_localization.json`은 없다. Supervisor는 worker exit `0`, watchdog `null`,
SIGTERM/SIGKILL `false/false`, process-group 없음, worker GPU allocation 없음을 독립 확인했다.

설치된 `SimulationApp.close()` 소스는 line 821에서 `Simulation App Shutting Down`을 출력하고,
line 835에서 Python process가 곧 종료된다고 명시한 뒤 line 838에서
`shutdown_and_release_framework()`를 호출한다. 따라서 이번 실행에서 post-return Python marker가
없는 것은 worker hang이나 Isaac 계산 예외가 아니라, **종료 함수 뒤에 도달 가능한 Python
코드가 있다고 잘못 가정한 사전등록 cleanup 계약**과 설치 동작의 불일치다. 분류는
`SIMULATION_APP_CLOSE_TERMINAL_NONRETURNING_POST_MARKER_CONTRACT_MISMATCH`다.

그러나 사전등록은 cleanup end marker까지 전체 PASS gate로 요구했다. 실행 뒤 그 gate를 느슨하게
바꾸면 안 되므로 원 completion은 다음처럼 보존한다.

- overall `pass=false`
- final verdict: `D367_MEASUREMENT_OR_INTEGRITY_FAIL_STOP`
- bridge subresult: PASS, but overall completion: FAIL
- original completion overwrite: `false`

Sources:

- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_cleanup_entry_state.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_worker_phase_markers.jsonl`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_worker_stdout_stderr.log`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_supervisor_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_completion_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d367/d367_postrun_classification_audit.json`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py:763-838`

## 14. 자원, 시각화, 과학 경계

Supervisor elapsed는 `19.401926056016237s`, telemetry sample은 `10`이었다. Device GPU used
max/free min은 `8185/7760MiB`, GPU utilization max는 `8%`, worker RSS max는
`6,949,773,312B`였다. 이 값들은 실행이 bounded였음을 보여 주지만 SM/Warp occupancy나 어떤
과학적 원인의 증거는 아니다.

D367은 공간 geometry나 접촉/trajectory를 판정하지 않았으므로 preregistration대로 PNG,
RRD/RBL, 동영상을 만들지 않았다. 이번에는 시각화할 물체 이동도 없었다. q5 science sample,
cylinder write, controlled physics, contact query가 모두 `0`이므로 cap/rim, 현재 자세 파지,
PhysX/Fabric/Hydra 동시점, target/IK/path justification는 모두 `null`; `g0a_pass=false`다.

## 15. 최종 해시와 다음 승인 경계

독립 `sha256sum` 재계산 결과:

- completion:
  `828defad2aa5720574c31fbd125e9a314534a0ebc4c3107151bfe3413c7065b4`
- supervisor:
  `8b48e3454a37761f98d6cdce579297231db103aa456faa404b005ad6a52c05ed`
- PLAY contract:
  `017a4ad8439e599789ac1eca2bd6b149c2d1aa34da3db3c6b66efd2e045a6030`
- zero-step attestation:
  `219f4350403f1a84ee18962560c016f1de2cebbf74356b32e8c80aeb96f810cd`
- cleanup entry:
  `db96d7ae384f54f9f370ce8315b520af140f838a775306dd4cdb8387fc09001a`
- postrun classification audit:
  `d79408724d83ee60ad9150c043260dd3f88087074761bb05eb9484d8c5863311`

사용자의 조건은 “D367이 PASS하면 D366 one-write/one-step/one-forward 측정을 재개”였다. 제어
다리만 분리하면 PASS지만, 사전등록된 D367 overall completion은 FAIL이다. 따라서 그 조건을
충족했다고 임의 해석하지 않고 D366 측정을 이번 turn에서 재개하지 않았다.

다음에는 둘 중 하나가 새 명시 승인을 필요로 한다.

1. Immutable D367 evidence와 설치 소스만 읽어, terminal/non-returning close의 completion
   authority를 `pre-close sentinel + supervisor exit/no-watchdog/no-residue`로 검증하는 별도
   offline control-contract repair. Isaac/q5/physics는 `0`이다.
2. 사용자가 overall cleanup FAIL과 bridge PASS를 명시적으로 구분해 받아들인 뒤, 새
   forward-only 경로에서 D366의 frozen one-write/one-step/one-forward 측정을 다시 승인한다.

어느 쪽도 이 문서 작성만으로 승인된 것으로 간주하지 않는다. D367 경로는 동결하며 재실행,
retry, overwrite하지 않는다.

## 16. 최종 continuity 교차검사

- `HEAD == origin/master == 9f956a42db1bb43c817ffe435a4e9698707049f1`, subject `D366`.
- D367 boot 전 worktree는 clean이었다. 현재 변경은 D367 harness/session/output과
  `START_HERE.md`, `BACKLOG.md`, `DECISIONS.md`, `EXPERIMENT_LEDGER.md`뿐이다.
- D351-D366 output과 사용자 소유 D334 sidecar diff는 `0`이다.
- D367 output root에는 파일 `16`개가 있고 모든 `.json`은 `jq empty` PASS했다.
- Harness SHA-256:
  `10802e9a4395903aeacc0a7d1536a82e0b435ff86a17b4b2b37e903e1ebafd5d`.
- `git diff --check` PASS; state docs의 숫자와 raw JSON을 다시 대조했다.
- `d367_worker_stdout_stderr.log`는 실제로 존재하고 SHA-256은
  `897a5a5653acc31d9e00a2a9ddde36f10056b3cce77eb75ba30bb4131938e0cf`지만 repo의
  `.gitignore:105` `*.log` 규칙에 걸린다. 일반 add/push에는 자동 포함되지 않으므로, 향후
  사용자가 commit을 승인할 때 이 exact path의 보존 여부를 명시적으로 확인해야 한다. 이번
  turn에서는 staging, commit, push를 하지 않았다.
