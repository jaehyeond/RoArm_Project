# D366 — tensor-step Fabric visibility commit

상태: `D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP` — PLAY prerequisite에서 pre-step 안전정지,
post-step PhysX/Fabric/Hydra 판정 `null`, 재실행 없음

이번 case의 신규 변수:

1. `timeline_play_tensor_write_contract`
2. `one_controlled_physics_step_before_inherited_public_forward`

## 1. 무엇을 왜 측정하는가

D365는 frozen D362 row 499 자세를 cylinder tensor writer에 한 번 썼다. AssetData cache와
독립 `root_physx_view.get_transforms()`는 즉시 TARGET을 반환했지만, paused zero-step
`SimulationContext.forward()` 뒤에도 Fabric root/mesh current, render cache, Hydra pixels는
BASELINE에 남았다. D365는 전달 단절을 찾았지만 tensor 값이 solver scene에 commit됐다고
증명하지는 않았다.

현재 설치된 Isaac Lab 2.3.0 writer는 tensor가 simulation에 들어가는 시점을 step 뒤로
명시한다. D366은 D365 측정기를 상속하고 public ordering contract에 물리 step 한 번만
추가한다. 목적은 파지 결과가 아니라 다음 경로가 현재 RoArm scene에서 실제로 열리는지다.

`tensor pose write → one physics step → public Fabric forward → Hydra`

## 2. 사용자 승인과 절대 한도

사용자는 2026-07-19 KST에 D366
`[tensor_step_fabric_visibility_commit]` observability-only control case를 승인했다.

- actual `headless=false` Isaac worker: `1`
- automatic retry: `0`
- frozen cylinder root pose write: `1`
- controlled `sim.step(render=False)`: `1`
- inherited public `SimulationContext.forward()`: `1`
- q5 science sample / q5 target update / contact query: `0/0/0`

Target/IK/path 및 asset/decomposition/gate/material/mass/actuator/physics/renderer 설정은
변경하지 않는다. D351-D365와 사용자 소유
`claudedocs/lab_meeting/20260715/d334_collision_table/`은 읽기 전용으로 봉인한다.
Cap/rim, exact contact face/manifold, force closure, stable grasp, hold/lift, G0a는 판정하지
않으며 `g0a_pass=false`를 유지한다.

여기서 q5/contact `0`은 D366 controlled window가 q5 state를 science sample로 읽거나 q5
target을 쓰거나 contact API를 조회한 횟수가 0이라는 뜻이다. 한 physics step 동안 실제
joint가 미세 이동했는지 또는 물리 접촉이 발생했는지는 조회하지 않으므로 `null/unknown`이며,
이를 “실제 접촉 0”으로 해석하지 않는다.

## 3. Git와 frozen 입력

실행 전 기준은 다음이다.

- `HEAD == origin/master == ce99a2cc24bd7a3112418739edc1b4ce1c6ef8c9`
- commit subject: `D365완료`
- D365 evidence-continuity 준비 전 worktree clean
- 현재 staged user-owned continuity set: `START_HERE.md` correction + D365 PNG 8개 +
  worker raw log 1개

D366은 이 staged set을 보존하며 commit/push하지 않는다. Frozen D365 핵심 SHA-256은
prepare artifact에서 원 파일과 다시 대조한다.

- harness: `719011a6171e27ae6b759903ef060397c1fad10c5c822d4c24a207ccdc59d834`
- completion: `efb2ece5bb30fa987bfa8a6ed229d282efdecff6c359beaa8034448ff1c3752d`
- localization report: `b82065bf89930f80d2c6e4a38bdaf9a323b2604b25efb7cb840b29b0ac4c5420`
- worker summary: `3babf239358f48ae5d2edd2124e3bcdb29d76ef5464a82b958c9fddd3a2e8c2e`
- RRD: `73a2a1d5954e6dfadfb7e562ea2ac4de8dcd80413e94b58db8abab069378a056`

## 4. 사전등록 실행 순서

1. Frozen D362 environment를 만들고 reset한다. Reset 내부 transition은 controlled step에
   포함하지 않는다.
2. Timeline을 paused-not-stopped로 만들고 D365 raw baseline과 새 baseline의 independent
   PhysX/Fabric hierarchy/render cache를 대조한다.
3. 양 카메라 baseline PNG를 clock-guarded capture로 남긴다.
4. Timeline을 PLAY로 전환하되 physics clock과 custom counter가 움직이지 않았음을 확인한다.
5. Frozen D362 row 499 cylinder pose를 정확히 한 번 쓴다.
6. `post_write_pre_step` checkpoint에서 cache, independent PhysX, Fabric 계층을 읽는다.
   PhysX는 commanded pose여야 한다. Fabric root/mesh/cache가 이미 선행 변경되면 actual
   step은 등록대로 수행하되 최종 성공으로 승격하지 않고 integrity STOP으로 분리한다.
7. Public `inner.sim.step(render=False)`를 정확히 한 번 호출한다. Custom counter,
   SimulationContext step index/time, timeline 상태로 실제 한 step을 독립 확인한다.
8. `post_step_pre_forward` checkpoint를 읽는다. 이 independent PhysX pose를 이후 동시점
   equality의 권위로 고정한다.
9. PLAY 상태에서 inherited public `inner.sim.forward()`를 정확히 한 번 호출하고 추가
   physics clock 진전이 없음을 확인한다.
10. `post_forward_pre_pause` checkpoint를 읽은 뒤 timeline을 pause한다. Pause가 state/clock을
    추가 진전시키지 않았고 root/mesh/cache도 바꾸지 않았음을 확인한다.
11. 양 카메라 `post_step_forward` PNG를 남기고 terminal checkpoint를 읽는다.
12. Append-only layer journal, worker/supervisor phase markers, report, sheet, RRD/RBL을
    봉인하고 원본 PNG와 Rerun screenshot을 실제로 검사한 경우에만 finalize한다.

## 5. 동시점 판정 기준

Commanded D362 row 499 pose와 one-step 뒤 pose의 bit-exact 일치는 요구하지 않는다. 동적
원통은 0.005s 동안 중력이나 기존 접촉을 해소할 수 있기 때문이다. Terminal 기준은
`post_step_pre_forward` independent PhysX pose다.

- Fabric root current가 post-step PhysX pose와 일치하지 않으면
  `D366_POST_STEP_PHYSX_TO_FABRIC_NOT_PROPAGATED`.
- Root는 일치하지만 render-mesh current가 post-step root로 재구성한 mesh와 다르면
  `D366_FABRIC_ROOT_TO_RENDER_MESH_NOT_PROPAGATED`.
- Mesh current는 일치하지만 Boundable cached worldMatrix가 다르면
  `D366_RENDER_MESH_TO_CACHE_NOT_PROPAGATED`.
- Numeric Fabric chain은 일치하지만 실제 두 camera pixels가 baseline에 남으면
  `D366_FABRIC_TO_HYDRA_NOT_PROPAGATED`.
- Post-step PhysX/Fabric/mesh/cache와 두-view Hydra visual class가 모두 같은 toppled state를
  지지하면 `D366_ONE_STEP_PHYSX_FABRIC_HYDRA_VISIBLE`.
- 필수 측정 누락, downstream-ahead, getter/forward/pause side effect, 두 camera 불일치,
  counter 위반은 `D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP`.

Hydra는 tensor와 bit-exact 비교할 수 없으므로 두 독립 camera에서 (a) baseline 대비
material pixel difference, (b) cylinder mask의 upright/toppled class, (c) Rerun에 함께 기록한
post-step pose geometry와 사람의 원본 해상도 검사를 결합한다. Rerun Float32 geometry는
inspection-only이며 numeric equality authority로 역사용하지 않는다.

## 6. 실패 가능한 사전검증

Prepare는 다음 negative controls를 모두 통과해야 한다.

- post-step PhysX→root, root→mesh, mesh→cache, cache→Hydra 각 단절 fixture
- missing/OTHER/downstream-ahead/two-view disagreement fixture
- commanded target과 post-step authority를 의도적으로 바꾼 fixture
- quaternion sign equivalence, xyzw↔wxyz swap, 10mm translation 대조
- step 0/2, write 0/2, forward 0/2, q5/contact nonzero counter injection
- getter/forward/pause가 추가 physics clock을 진전시키는 fixture

이 perturbation evaluation이 판정을 실제로 FAIL시킬 수 있으므로 Session Progress Rule을
충족한다. Actual worker는 prepare PASS 뒤 한 번만 실행하며 실패해도 retry하지 않는다.

## 7. 설치 소스와 구현 사전감사

실행에 사용할 설치 소스 자체를 SHA-256으로 고정했다.

- IsaacLab `SimulationContext`: `340450726276d321c48b57de35f846c5a231c30a358b4922b5b7dbb8d42ec80e`
- Isaac Core `SimulationContext`: `ebafc6bcb30a454925fe21b96dcdbd4637c922a3fa9d5a6947308c9796ba5028`
- Isaac Core `PhysicsContext`: `7486ec315d2525bdef6e1cef294d67d419511e3367e7f40001a8324e2b6a7751`

이 설치판에서 `sim.step(render=False)`는 PLAY일 때 `simulate→fetch_results`를 수행하고
기본 `update_fabric=False`를 유지한다. Public `forward()`는 PLAY 상태에서 articulation
kinematic update 뒤 bound Fabric callable을 한 번 호출한다. `render()`는 내부에서
`forward()`를 다시 부를 수 있으므로 D366 controlled path에서는 호출하지 않는다.

Active worker AST 사전감사는 root pose writer, `step(render=False)`, public `forward()`,
physics callback add/remove, timeline `play()` callsite를 각각 정확히 1개로 제한한다.
34개 failure-capable negative-control check가 모두 PASS해야만 prepare가 PASS한다. Rerun은
8개 numeric checkpoint 전체 timeline을 기록하되 실제 RTX 이미지는 baseline/final 두
시점 × 두 view를 기록한다.

## 8. 실행 전 상태

- D366 output path는 아직 존재하지 않는다.
- D366 actual Isaac invocation count는 `0`이다.
- 이 문서 이후 결과 절은 actual run과 수동 inspection/finalize가 끝난 뒤 append한다.

## 9. 실제 실행 — 관찰 순서

### 9.1 prepare와 단일 invocation

1. `--stage prepare`는 등록된 `22/22` check를 모두 통과했다. 여기에는
   `HEAD==origin/master`, D362-D365 immutable manifest, staged D365 continuity set, D334
   sidecar, 설치 소스 SHA, static source audit, 34개 negative-control 계열, pinned
   `numpy==1.26.0`/`psutil==5.9.8`, Rerun 0.34.1, `DISPLAY=:1`, RTX 4090, VRAM/RAM gate가
   포함된다.
2. actual `headless=false`, `cuda:0` worker를 정확히 한 번 시작했고 자동 retry는 없었다.
   Invocation marker의 supervisor/worker PID는 `1308152/1308264`다.
3. Worker preflight `20/20`, launcher contract `7/7`, runtime prerequisite `17/17`이 PASS했다.
   Reset 뒤 내부 clock은 SimulationContext time/index
   `0.009999999776482582s/2`, timeline time `0.029999999329447746s`, custom step/callback
   `0/0`이었다. Reset 내부 transition은 registered controlled step에서 제외했다.
4. D353에서 검증된 PAUSE bridge는 commit `1`회로 paused-not-stopped를 만들었고 물리 clock을
   진전시키지 않았다. Independent PhysX, Fabric hierarchy current, mesh current/cache의
   D365 baseline 상속과 두 camera baseline 캡처도 완료했다.

### 9.2 PLAY guard에서의 안전정지

5. Physics callback을 등록한 뒤 count/dt는 계속 `0/[]`였다.
6. Registered raw `timeline.play()` request를 한 번 호출했다. D353의 PAUSE commit을 PLAY로
   일반화하지 않는 동결 규칙 때문에 이 구간의 `Timeline.commit()`은 `0`회였다.
7. 직후 guard는 다음을 기록했다.

   - `playing_not_stopped=false`
   - `play_commit_not_used=true`
   - `physics_state_unchanged=true`

   따라서 worker는 pose write보다 먼저
   `RuntimeError: D366 PLAY transition STOP`으로 안전정지했다. D366 artifact는 raw
   `(is_playing,is_stopped)` tuple 자체를 직렬화하지 않았으므로, “PLAY request가 pending이었다”는
   D352-D353 계보와 일치하는 해석이지만 D366의 직접 관찰값으로 과장하지 않는다. 직접 관찰은
   required `playing-not-stopped` 상태가 입증되지 않았다는 것까지다.
8. Program order상 exception 직전 실제 count는 다음과 같다.

   - cylinder pose write `0`
   - controlled `sim.step(render=False)` call/return `0/false`
   - physics callback count/dt `0/[]`
   - public `forward()` call/return `0/false`
   - q5 science sample/q5 target/contact query `0/0/0`

   Registered bridge가 완료되지 않았으므로 `controlled_physics_steps=null`을 유지한다. Reset 내부
   동작을 여기에 더하지 않는다. Contact query `0`은 실제 접촉이 없었다는 뜻이 아니라 접촉을
   조회하지 않았다는 뜻이다.

### 9.3 종료와 잔류 프로세스

9. Worker exception marker는 preflight에서 `20.252107s` 뒤 기록됐다. 그 뒤 cleanup 함수별
   marker가 없어서 정확히 `inner.close()`인지 `simulation_app.close()`인지 특정할 수 없는 종료
   구간에서 새 phase가 나오지 않았다.
10. Supervisor의 phase-inactivity watchdog가 전체 `204.97441041003913s`에 process group을
    종료했다. Raw log의 마지막 engine 행은 `[202.572s] Simulation App Shutting Down`이다.
    Worker exit code는 `0`이지만 exception 존재, worker summary 부재, watchdog 발생 때문에 PASS
    권위가 아니다.
11. 종료 뒤 host process를 다시 조회한 결과 supervisor `1308152`, worker `1308264`, NGX child
    `1310087` 모두 없었고 worker PID도 NVIDIA compute-app 표에 없었다. 수동 signal이나 retry는
    없었으며 D342 잔류 PID `1729639` 등 다른 process는 건드리지 않았다.

## 10. 정량 결과와 권위 구분

- Prepare/worker preflight/runtime prerequisite: `22/22`, `20/20`, `17/17` PASS.
- Worker phase journal: `15` rows. 마지막은 `worker_exception:stop`이다.
- Supervisor phase journal: `4` rows, 순서/횟수 PASS. Watchdog는
  `phase_inactivity`, elapsed `204.97441041003913s`다.
- Layer journal은 등록 `8`개 중 baseline 두 row만 남았다. Hash chain과 두 getter clock guard는
  PASS했지만 exact label/count gate는 FAIL했다. 이는 post-step 층이 없다는 정직한 실패다.
- GPU device-level telemetry는 total used max/free min `8576/7369MiB`, utilization max `68%`였고,
  worker RSS max `7,104,565,248B`였다. 이는 Warp occupancy/SM 효율 측정이 아니며 PLAY Boolean
  guard 실패의 원인으로 쓰지 않는다.
- Engine log의 generic `Failed to clone in Fabric`는 이전 case들에도 반복된 비특이 행이며,
  이번 guard의 직접 원인으로 입증되지 않았다.

Supervisor summary는 worker summary가 없어 write/step/forward/q5/contact 필드를 `null`로 보존했다.
반면 worker exception은 실패 지점의 program-order count를 `0`으로 기록했다. 이 둘은 모순이
아니다. `null`은 완결된 worker summary 부재를, exception의 `0`은 해당 callsite에 아직
진입하지 않았음을 뜻한다. 단, registered bridge 전체의 권위값인
`controlled_physics_steps`는 계속 `null`이다.

`d366_automated_summary.json`은 manual/process audit 이전에 생성된 immutable immediate-postprocess
snapshot이므로 `completion_pending=true`, `manual_visual_inspection_pending=true`를 그대로 가진다.
이를 덮어쓰지 않았으며, 최종 상태는 뒤에 추가한 forward-only completion이 원본 exception과
supervisor null의 역할을 분리해 보정한다.

최종 독립 일관성 감사에서 새 postrun draft 두 항목을 사용자 handoff 전에 교정했다. 첫째,
원본 prepare `.checks|length`는 `22`이고 전부 true인데 draft completion/state docs가 `21/21`로
잘못 셌다. 둘째, post-PLAY raw Boolean tuple이 없는데 manual audit가 “pending PLAY”를 직접
관찰처럼 썼다. 현재 파일은 각각 `22/22`, “PLAY-state guard failure + tuple 미직렬화”로
교정됐고, 이전/현재 SHA 계보는 별도
`d366_postcompletion_correction_audit.json`에 forward-only로 남겼다. 런타임 결과·null·verdict는
바뀌지 않았다.

## 11. 실제 화면 검사와 시각화 한계

두 원본 `1280x720 RGBA` Isaac PNG를 실제 해상도로 검사했다.

- primary SHA-256:
  `b82ed94f8b72e2ba7165abed854c5c95930faa7b49a4d1be70b428c610f3784f`
- opposite SHA-256:
  `4c26cbdf2e25208bdd84aaa9eb90cf29a1bf7c8594522b33d7fd7959dd9be8cc`

양쪽 모두 노란 원통이 upright이고 흰 RoArm gripper와 함께 보였다. 그러나 두 장은 PLAY request
전 baseline이다. Pose write, post-step, post-forward 이미지가 아니므로 “one-step 뒤에도 원통이
안 움직였다”거나 Fabric/Hydra가 실패했다고 판정할 수 없다. Post-step decision subject가 생기기
전에 멈췄으므로 report/sheet/RRD/RBL은 생성하지 않았고 visualization completion은 FAIL이다.

## 12. 최종 판정

최종 operational verdict는 `D366_MEASUREMENT_OR_INTEGRITY_FAIL_STOP`이다.

- D366이 승인받은 질문인 post-step independent PhysX와 Fabric root/mesh/cache/Hydra의 동시점
  일치 여부: `null`
- D365의 `D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED`: 그대로 상속
- D362 physical sub-result: 그대로 상속
- q5/contact/cap-rim/grasp/G0a science 재계산: 없음
- `g0a_pass=false`
- D366 geometry/physics/render propagation PASS 또는 FAIL: 아님

실패한 것은 Isaac 자체 실행도, GPU 메모리도, PhysX step 결과도 아니다. Isaac scene·baseline
PhysX/Fabric 읽기·실제 Hydra 캡처까지 정상 도달한 뒤, pre-step PLAY 상태 전이 계약을 즉시
확정하지 못해 의도적으로 멈춘 것이다.

## 13. 다음 승인 경계

같은 D366 경로를 재실행하거나 덮어쓰지 않는다. 가장 좁은 다음 후보는 별도 승인 control-only
case에서 raw `timeline.play()` 직후 explicit `Timeline.commit()` 한 변수만 검증하는 것이다.
Before/request/post의 raw `(is_playing,is_stopped)`, timeline time, SimulationContext time/index,
custom counter, physics callback, object/joint bits를 모두 직렬화해야 한다. 또한 cleanup의
callback removal, `inner.close()`, `simulation_app.close()` 시작/끝 marker를 두어 종료 watchdog
위치도 분리한다.

이 bridge가 PASS하더라도 cylinder pose write, controlled physics step, public forward와
post-step PhysX/Fabric/Hydra 비교는 다시 별도 명시 승인을 받아야 한다. Public
`SimulationContext.play()`는 설치 소스상 내부에서 `timeline.play(); timeline.commit()`을 호출하고
문서가 physics-handle propagation을 위한 내부 step을 경고하므로, exact-one-step case에 조용히
대체하지 않는다.

## 14. 주요 증거

- `claudedocs/runtime_logs/grasp_track/g0a_d366/d366_worker_exception.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d366/d366_worker_phase_markers.jsonl`
- `claudedocs/runtime_logs/grasp_track/g0a_d366/d366_supervisor_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d366/d366_runtime_step_fabric_attestation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d366/d366_pre_step_play_guard_safe_stop_completion.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d366/d366_baseline_only_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d366/d366_postrun_process_cleanup_audit.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d366/d366_postcompletion_correction_audit.json`

Forward-only completion SHA-256:
`cc061e07fca358d18467255a5ad02a6460513eae6bed5cccc463870b9ecf2f7d`.
Postcompletion correction audit SHA-256:
`2f55c13bbd127654260f318798b2a6a76b0a8dc39cce1efe13502f0137a86312`.
