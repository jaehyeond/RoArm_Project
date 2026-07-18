# D364 — paused render-state layer localization

Date: 2026-07-18 KST

Case: `g0a_d364`

Status at preregistration: `USER_APPROVED_IMPLEMENTATION_IN_PROGRESS_NO_D364_ISAAC_INVOCATION`

이번 case의 신규 변수:

1. `zero_step_independent_physx_and_scene_layer_readback`
2. `single_final_pose_layer_localization_with_hydra_pixel_witness`

신규 q5/target/IK/path/asset/physics 변수: `[]`

## 1. 무엇을 왜 측정하는가

D363은 frozen D362 trace의 네 자세를 IsaacLab tensor API로 직접 쓴 뒤 명시적
`SimulationContext.forward()`를 네 번 호출했지만 실제 viewport의 원통은 계속 수직으로
남았다. D363의 write/read bit-exact 결과는 독립 PhysX readback이 아니라 writer가 먼저
갱신한 `AssetData` cache를 같은 timestamp에서 다시 읽은 결과였다. 따라서 현재 알려진
단절 범위는 다음 전체 구간뿐이다.

`AssetData cache → PhysX backend → Fabric/USDRT → Hydra pixel`

사용자는 이 단절을 찾는 다음 한 단계의 측정·검증을 승인했다. D364는 render-sync를
고치거나 물리 결과를 다시 실행하지 않는다. D362의 frozen final row 하나를 D363과 같은
direct display-state write로 1회 기록하고, 각 층을 독립적으로 읽어 최초 불일치 화살표를
국소화한다.

## 2. 현재 Git과 forward-only 경계

- preregistration 직전 실제 `HEAD == origin/master ==
  94c0644ef3d4e69278bc864f0f8c2f3a40908dc8`, commit `D363test`다.
- preregistration 직전 worktree는 clean이었다.
- 기존 `START_HERE.md`의 `f085463d...`/D363-uncommitted 설명은 사용자 push 전 상태라
  stale하다. D364 state update에서 실제 Git 명령 결과로 교정한다.
- 새 output은
  `claudedocs/runtime_logs/grasp_track/g0a_d364/`만 exclusive-create한다.
- D351-D363 output, 특히 immutable D362 33-file tree와 D363 40-file tree를 수정·추가·
  rename·rerun하지 않는다.
- 사용자 소유
  `claudedocs/lab_meeting/20260715/d334_collision_table/` sidecar는 읽기/hash 대조만 한다.
- commit/push는 승인되지 않았고 수행하지 않는다.

## 3. frozen 입력

- D362 canonical trace:
  `claudedocs/runtime_logs/grasp_track/g0a_d362/d362_physics_trace.json`, SHA-256
  `9483146c4941e6518614c63acbf221128a564bafa7a9928d41e633ee6e4e2044`.
- 유일한 target state는 zero-based row `499`, global step `500`인 final state다.
- D362 worker summary SHA-256:
  `10f7bd39b67f9bd254827fab580396c9a8089304f904c20dc3efd908296b217d`.
- D363 harness SHA-256:
  `63b307137405b2a343af88e046e992ef4ee996aff3bc467e2bf58390e4e18a14`.
- D363 render-sync report SHA-256:
  `4cd5dd401b4eaea687549c5f5279b71e0f7fb0ad67a70f4d555f94b793653b3c`.
- D363 completion SHA-256:
  `e55a155b814dabdb90ce6b219c36318431f695331342624d2d2780d7b7b4f078`.

## 4. 정확한 측정 순서

Reset 내부 warm-up은 D364 controlled step에 포함하지 않는다. Reset 뒤 D353의
`Timeline.commit()` bridge를 상속해 timeline을 `PAUSED-not-STOPPED`로 확정하고
`/app/player/playSimulations=false`를 유지한다. 그 뒤 순서는 다음 하나뿐이다.

1. **baseline**: cache, 독립 `root_physx_view.get_transforms()/get_velocities()`, authored
   USD world transform, attached USDRT/Fabric prim의 직접 world/local attributes,
   Fabric interface/callable과 physics/timeline clock을 읽고 primary viewport를 캡처한다.
2. **direct write**: D362 row 499의 actual joint position/velocity와 cylinder
   position/quaternion/linear+angular velocity를 Float32로 정확히 1회 쓴다. Drive target,
   q5 command, `scene.write_data_to_sim()`, scene/sensor update는 호출하지 않는다.
3. **post-write immediate**: 어떠한 app update나 `forward()` 전 cache/PhysX/USD/USDRT를
   다시 읽는다.
4. **post-write app-update**: `forward()` 없이 guarded raw `simulation_app.update()`로
   primary viewport를 캡처한 뒤 네 층을 다시 읽는다.
5. **explicit forward**: run에서 확인된 `cfg.use_fabric`, `_fabric_iface`,
   `force_update/update` 선택 callable을 기록한 상태로 `inner.sim.forward()`를 정확히
   1회 호출한다.
6. **post-forward immediate**: app update 전 cache/PhysX/USD/USDRT를 다시 읽는다.
7. **post-forward app-update**: guarded raw app update로 primary viewport를 캡처하고
   네 층을 마지막으로 읽는다.

모든 단계에서 custom step counter, SimulationContext current time/index, timeline time과
PLAY/STOP 상태를 기록한다. Direct write로 의도한 state 외에 clock/counter가 하나라도
전진하면 operational FAIL_STOP이다.

## 5. 계층별 권위와 판정표

- `AssetData cache`: setter가 받은 값을 보존했는지만 확인한다. backend 권위가 아니다.
- `PhysX backend`: `root_physx_view.get_transforms()`의 독립 getter만 사용한다. PhysX는
  xyzw quaternion을 반환하므로 wxyz로 명시 변환한 뒤 frozen target과 비교한다.
- `authored USD`: standard USD stage의 `ComputeLocalToWorldTransform()`을 읽는다. Fabric
  사용 중에는 stale authored value일 수 있으므로 관찰값이지 dynamic-state gate가 아니다.
- `Fabric/USDRT`: 현재 USD stage id에 attach한 USDRT stage에서 exact cylinder prim의
  validity, `HasWorldXform/HasLocalXform`, world position/orientation 또는 local matrix를
  직접 읽는다. Standard USD fallback 값은 Fabric PASS로 세지 않는다.
- `Hydra pixel`: 원본 1280x720 PNG의 yellow largest component bbox/centroid/PCA axis/IoU와
  upright/toppled category를 사용한다. Nonempty PNG만으로 PASS하지 않는다.

Target match 허용치는 Float32 backend/display 경계만 수용하도록 position max-abs
`<=1e-6m`, quaternion angular error `<=0.01deg`로 고정한다. Hydra의 baseline→final
material change는 D363 기준을 그대로 상속한다: centroid `>=15px`, PCA axis delta
`>=15deg`, IoU `<=0.85` 중 2개 이상이고 final `width/height>=1.15`여야 한다.

최초 단절 판정은 다음처럼 forward-only로 고정한다.

| 관찰 | localization verdict |
|---|---|
| cache가 target 불일치 | `D364_DIRECT_WRITE_OR_CACHE_FAIL` |
| cache target, PhysX getter 불일치 | `D364_CACHE_TO_PHYSX_PENDING_OR_FAILED` |
| PhysX target, post-forward Fabric 불일치/부재 | `D364_PHYSX_TO_FABRIC_NOT_PROPAGATED` |
| Fabric target, Hydra pixel 불일치 | `D364_FABRIC_TO_HYDRA_NOT_PROPAGATED` |
| cache/PhysX/Fabric/Hydra 모두 target 대응 | `D364_END_TO_END_ZERO_STEP_VISIBLE` |
| 필요한 독립 getter/prim/attribute 자체를 읽지 못함 | `D364_MEASUREMENT_INCOMPLETE_FAIL_STOP` |

이 판정은 render synchronization 위치만 답한다. D362의 접촉·운동 물리 결과와 D354의
cap/rim science를 재판정하지 않는다.

## 6. 사전 대조군과 artifact 계약

- Offline decision-table positive/negative fixtures가 위 여섯 verdict를 서로 혼동하지
  않아야 한다.
- Cache 값을 PhysX 값으로 대체한 self-read fixture, USD fallback을 Fabric 값으로
  가장한 fixture, missing prim/attribute, clock/step increment, second write/second forward,
  q5 target/contact query와 D362/D363 mutation을 모두 거부한다.
- 실제 artifact는 preregistration, prepare, invocation marker, worker preflight/runtime,
  append-only phase markers, worker log/summary, layer localization report, 원본 PNG 3개,
  초보자용 Korean diagnostic sheet, RRD/RBL/validation/headless screenshot, supervisor/
  automated/manual/completion summary다.
- 공간·viewport 판정이 있으므로 Rerun 0.34.1 RRD를 새로 만든다. File sink는 첫 user log
  전에 연결하고 context exit 뒤 footer/RBL/entity/timeline/component/headless screenshot을
  검증한다. Rerun Float32 표시값은 layer 판정에 역사용하지 않는다.
- 원본 PNG와 Rerun screenshot을 실제로 검사한 manual JSON 뒤에만 finalize한다.

## 7. 실행 횟수와 watchdog

1. preregistration과 harness 정적 검토
2. CPU-only prepare 1회, 새 output exclusive create
3. 실제 `headless=false`, `DISPLAY=:1`, `cuda:0` Isaac worker 정확히 1회
4. automatic retry/resume/overwrite 0
5. bounded total/inactivity watchdog와 supervisor resource telemetry
6. worker 종료 뒤 offline layer/pixel/Rerun 검증
7. 실제 visual inspection 뒤 CPU-only finalize 1회

이 case는 D363에서 실제 관찰된 viewport synchronization 실패에 대한 reactive
localization이다. 결과에 따라 다음 render repair 선택이 달라지므로 AGENTS.md의
failure-capable/reactive control-contract 조건을 충족한다. 새 q5/physics perturbation은
수행하지 않는다.

## 8. 동결·금지 범위

- controlled physics step, q5 science sample/command/target update, contact/force query: `0`
- target/IK/path, asset/cook/decomposition/gate/tolerance, material/mass/actuator/solver/
  physics/renderer setting 변경: `0`
- cap/rim/barrel, exact face/manifold, force closure, grasp/hold/lift/G0a 판정: `0`
- settle, ten-trial, G0b, RL/PPO/VLA, ladder promotion: `0`
- dependency/package install, GPU clock/power/persistence/Warp/SM mutation: `0`
- real robot/hardware, B200/SSH, commit/push, unapproved signal: `0`
- D351-D363 또는 D334 sidecar write: `0`

실제 harness hash, prepare 결과, 단일 invocation과 계층별 관찰값은 실행 전/후
forward-only 절에만 추가한다.

## 9. 실행 전 계층 측정 정밀화

독립 API·harness·판정표 검토 뒤, 실제 invocation 전 다음을 더 좁게 고정한다. 이 절은
위 범위를 확장하지 않고 cylinder render-transform 단절에 불필요한 state write와
false Fabric fallback을 제거한다.

1. Direct write는 D362 row 499의 **cylinder root pose만**
   `write_root_pose_to_sim()`으로 1회 기록한다. Robot joint/q5 state, cylinder velocity,
   drive target은 쓰지 않는다. D363의 stale subject가 cylinder pose였고 paused
   articulation visual은 별도 kinematic 경로이므로 cylinder 하나로 격리한다.
2. `root_physx_view.get_transforms()`는 AssetData와 독립인 **PhysX tensor-view getter**로
   부른다. 이것을 solver 내부 committed-state getter라고 과장하지 않는다. Getter 전
   passive Fabric snapshot과 getter 후 passive Fabric snapshot을 각각 남겨 CUDA sync/getter가
   관찰 자체를 바꿨는지 검출한다.
3. Fabric은 rigid root `/World/envs/env_0/Sponge`뿐 아니라 실제 render leaf
   `/World/envs/env_0/Sponge/geometry/mesh`를 함께 읽는다. 각 prim에서 다음을 구분한다.
   - OmniHydra compatibility world position/orientation/scale attributes
   - Fabric hierarchy local matrix와 `hier.get_world_xform()` current-computed matrix
   - `omni:fabric:worldMatrix` cached/render-driving matrix
4. `isaacsim.core.utils.xforms`의 Fabric helper는 필요한 attribute가 없으면 authored USD로
   fallback할 수 있으므로 Fabric authority로 사용하지 않는다. Direct USDRT prim/attribute가
   없으면 명시적 measurement-incomplete다. `XFormPrim` view 생성, attribute Create/Set,
   `IFabricHierarchy.update_world_xforms()`도 read-only localization을 바꾸므로 금지한다.
5. Runtime은 `cfg.use_fabric`, `is_fabric_enabled()`, `_fabric_iface`, selected
   `_update_fabric` callable과 함께 `/app/useFabricSceneDelegate`,
   `/rtx/hydra/readTransformsFromFabricInRenderDelegate`를 기록한다. Direct callable을 따로
   호출하지 않고 `inner.sim.forward()`만 정확히 1회 호출한다.
6. Mesh target matrix는 baseline root↔mesh relative matrix를 고정해 final root pose와
   compose한다. Baseline에서 compose가 원 mesh matrix를 재구성하지 못하면 runtime
   prerequisite FAIL_STOP이며 임의 multiplication order를 선택하지 않는다.
7. Primary와 opposite 두 camera는 독립 관측 복제로 사용한다. 각 view에서 baseline,
   post-write/no-forward, post-forward PNG를 한 장씩 남겨 총 6장이다. View는 신규 물리
   변수가 아니며 원통 visibility와 false mask를 교차검증한다.

따라서 최종 동적 판정층은 다음 다섯 단계다.

`AssetData cache → independent PhysX tensor view → Fabric rigid root → Fabric rendered mesh → Hydra pixel`

Fabric root가 target인데 rendered mesh hierarchy가 target이 아니면 기존 §5 판정표보다
정밀한 `D364_FABRIC_ROOT_TO_RENDER_PRIM_HIERARCHY_NOT_PROPAGATED`를 사용한다. Downstream이
target인데 upstream이 old이거나 phase 사이 비단조/OTHER가 생기면 최초 화살표를 억지로
고르지 않고 `D364_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP`으로 종료한다. Authored USD가
baseline에 머무는 것은 Fabric-enabled runtime에서 예상 가능한 control이며 FAIL이 아니다.

## 10. 실제 invocation 직전 모순 제거와 최종 판정층 동결

§9가 §4와 §6의 넓은 초안을 명시적으로 축소했으므로, 실행 harness와 artifact gate는
다음 최종 계약만 사용한다.

- write 대상은 cylinder root pose 하나뿐이다. Joint position/velocity,
  `get_velocities()`, cylinder velocity는 쓰거나 읽지 않는다.
- 실제 Isaac PNG는 3장이 아니라 `3 phases × 2 views = 6장`이다.
- `simulation_app.update()`는 physics clock 불변 guard 안에서 viewport/capture event loop만
  진행한다. 물리 step이나 renderer 설정 변경으로 세지 않는다.
- 최종 순서는 cache → PhysX tensor view → Fabric root current → Fabric root cached/render
  → Fabric mesh current → Fabric mesh cached/render → Hydra pixel이다.
- Fabric root current가 target인데 root cached/render가 baseline이면
  `D364_FABRIC_ROOT_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED`다.
- Fabric mesh current가 target인데 mesh cached/render가 baseline이면
  `D364_FABRIC_RENDER_PRIM_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED`다.
- 이 두 세부 verdict도 기존 최초 불일치 원칙을 좁힌 이름일 뿐 신규 mutation이나
  후속 repair 승인이 아니다.

이 절 이후에 harness hash를 고정하고 CPU-only prepare를 실행한다. 이 시점까지
`g0a_d364/` output과 실제 Isaac invocation은 아직 없다.

## 11. 공식 USDRT 계층 의미 교차검증에 따른 최종 사전등록 보정

실행 전 설치된 `usdrt.scenegraph 7.6.1` 공식 문서
`.../docs/fabric_hierarchy.rst`를 다시 대조했다. 문서상 `_worldPosition/_worldOrientation`
compatibility 속성, hierarchy `localMatrix`, current-computed world, cached `worldMatrix`는
서로 같은 값의 별명이 아니다. 또한 cached `worldMatrix` 자동 생성은 Boundable prim에
보장되지만 rigid root Xform에는 보장되지 않는다. 따라서 §10의 root cached 선형 gate와
그 verdict는 폐기하고 다음을 최종 계약으로 삼는다.

`AssetData cache → PhysX tensor view → Fabric root compatibility pose →`
`Fabric hierarchy root current → Fabric mesh current → Fabric mesh cached/render → Hydra`

- root cached `worldMatrix`는 읽되 optional diagnostic-only다. 없거나 baseline이어도 그
  자체로 FAIL 또는 단절 verdict를 만들지 않는다.
- PhysX가 target인데 root compatibility가 baseline이면
  `D364_PHYSX_TO_FABRIC_COMPATIBILITY_NOT_PROPAGATED`다.
- root compatibility가 target인데 hierarchy root current가 baseline이면
  `D364_FABRIC_COMPATIBILITY_TO_HIERARCHY_NOT_PROPAGATED`다.
- hierarchy root current가 target인데 mesh current가 baseline이면
  `D364_FABRIC_ROOT_TO_RENDER_PRIM_HIERARCHY_NOT_PROPAGATED`다.
- mesh current가 target인데 Boundable mesh cached/render가 baseline이면
  `D364_FABRIC_RENDER_PRIM_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED`다.
- mesh cached/render가 target인데 두 Hydra view가 baseline이면
  `D364_FABRIC_TO_HYDRA_NOT_PROPAGATED`다.
- 두 Hydra view가 서로 다르거나 partial-change이고, getter 전후 전체 Fabric snapshot이
  달라지거나, 어느 계층이 TARGET에서 BASELINE/OTHER로 회귀하면
  `D364_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP`이다.

Mesh target transform은 더 이상 `mesh_world @ inverse(root_world)`로 자기 자신을 다시
복원하는 순환식을 쓰지 않는다. Authored static hierarchy에 대해
`UsdGeom.XformCache.ComputeRelativeTransform(mesh, root)`를 독립적으로 구하고, 설치된
USD row-vector 계약인 `relative @ root_world`로 baseline Fabric mesh world를 재구성해
허용치 `1e-5` 안에서 먼저 검증한 뒤 target mesh를 계산한다. Authored USD는 이 고정된
root-to-mesh relative geometry에만 쓰며 dynamic pose 권위로 승격하지 않는다.

Layer journal payload와 worker checkpoint JSON은 exact equality로 대조하고, phase marker는
존재 횟수뿐 아니라 실행 순서까지 검사한다. Human preregistration session의 SHA-256과 byte
count도 prepare JSON에 봉인하며, actual invocation/finalize 전까지 바뀌면 STOP한다.

이 §11이 §4·§5·§6·§9·§10 중 충돌하는 초안을 최종적으로 supersede한다. 여전히 신규
mutation은 cylinder root pose write 1회와 public `SimulationContext.forward()` 1회뿐이며,
controlled physics/q5/contact query는 모두 0이다. 이 보정 시점까지 실제 Isaac invocation은
여전히 0회다.

## 12. 실행 전 잔류 프로세스와 resource gate

실제 드라이버 namespace에서 GPU process를 다시 대조했다. PID `1729639`는 부모
`systemd --user` PID `1123`, PGID/SID `1729601`, 시작 `2026-07-13 21:38:33 KST`, 명령
`python sim_scripts/cyl34_top_view_d342_grasp_g0a_authored_coordinate_stream_repair.py
--headless`인 D342 잔류임을 재확인했다. 앞선 사용자 승인에 따라 PID 하나에 SIGTERM을
1회 보냈고 command 자체는 성공했지만, 재조회에서 process는 여전히 `Sl` 상태였다.
SIGKILL이나 다른 signal은 승인 범위가 아니므로 사용하지 않는다.

이 잔류 process의 GPU 사용은 `320MiB`였고, D364 prepare 직전 전체 GPU는 RTX 4090
Laptop `16376MiB` 중 used/free `2051/13894MiB`, utilization `20~21%`, P-state `P8`였다.
따라서 D364의 preregistered free-VRAM gate `>=8192MiB`는 충분히 통과하며, D342 process는
별도 process/stage이므로 D364 state 계층 값의 권위에는 포함하지 않는다. D364는 이 사실을
숨기지 않고 supervisor resource telemetry에 다시 남긴다.

이제 harness/session hash를 동결해 CPU-only prepare를 1회 실행한다. 아직 actual Isaac
worker invocation은 0회다.

## 13. CPU-only prepare 결과

Prepare는 1회 실행됐고 전 항목 PASS했다.

- Harness SHA-256:
  `4377203b9756b503b4f8a80f955db93ed9301edaa84e3113d81cfdee4c558925`
- Human preregistration prefix SHA-256/bytes:
  `f8d2ff508033fd34ab23927fb16a7c623fdb3f219c3eda31dc8f4bfc42548afb` /
  `18,394B`
- `HEAD == origin/master == 94c0644ef3d4e69278bc864f0f8c2f3a40908dc8`
- D362/D363 file count `33/40`, frozen input hashes, D334 sidecar, NumPy `1.26.0`,
  psutil `5.9.8`, Rerun `0.34.1`, `DISPLAY=:1` 모두 PASS했다.
- Prepare 시 GPU used/free `2,051/13,894MiB`, available RAM `15,954,116,608B`였다.
- 등록된 실제 worker invocation/write/forward/controlled-step 수는 `1/1/1/0`, automatic
  retry `0`이었다.

Sources:

- `claudedocs/runtime_logs/grasp_track/g0a_d364/d364_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d364/d364_prepare_preflight.json`

## 14. 실제 단일 invocation — 실행 순서와 안전 정지

실제 `headless=false`, `cuda:0`, `DISPLAY=:1` worker는 정확히 1회 시작됐다. 자동 retry는
없다. 실행 순서는 phase marker 기준 다음과 같다.

1. Worker preflight PASS.
2. AppLauncher start/complete PASS.
3. `_make_runtime_env` start/complete PASS.
4. Reset start/complete. Reset 내부 SimulationContext clock은 이미
   `time=0.009999999776482582s`, step index `2`였고 D364 controlled step에서는 제외했다.
5. Timeline을 PAUSED-not-STOPPED로 고정한 bridge PASS.
6. Baseline pre-capture layer journal 1개 기록.
7. Primary/opposite 실제 Isaac viewport를 각각 16 app updates 안에서 1280x720로 캡처.
   각 update 전후 physics/timeline clock 불변을 확인했다.
8. Baseline post-capture layer journal 1개 기록.
9. Runtime prerequisite에서 compatibility root attribute 부재를 확인하고 pose write 전에
   FAIL_STOP했다.

따라서 실제 등록 카운터는 다음과 같다.

| 항목 | 실제 수 |
|---|---:|
| actual Isaac worker invocation | 1 |
| cylinder root pose write | 0 |
| explicit `SimulationContext.forward()` | 0 |
| controlled physics step | 0 |
| q5 science sample / q5 target update | 0 / 0 |
| contact query | 0 |
| automatic retry | 0 |

Worker exception의 정확한 prerequisite 결과는 16개 중 다음 두 항목만 false였다.

- `root_compatibility_and_computed_available=false`
- `baseline_physx_matches_fabric_compatibility=false`

나머지 핵심 항목—Fabric enabled/interface, selected bound `force_update`, Fabric Scene
Delegate, Hydra Fabric transform read, exact PhysX prim path, root/mesh Fabric prim validity,
mesh cached/current availability, independent authored-relative compose, PhysX↔hierarchy-current
baseline equality, getter clock guard—는 true였다.

## 15. 원 계층값이 실제로 말하는 것

이번 FAIL은 Isaac이나 Fabric 전체가 실행되지 않았다는 뜻이 아니다. 오히려 실제 scene의
root 표현을 구체적으로 확인했다.

- 독립 PhysX getter baseline pose wxyz:
  `[0.30000001192092896, 0.0, 0.03288299962878227, 1.0, 0.0, 0.0, 0.0]`
- Root prim `/World/envs/env_0/Sponge`는 valid `Xform`이었다.
- Root `_worldPosition`, `_worldOrientation`, `_worldScale` compatibility attributes는 모두
  `valid=false`, `value_present=false`였다.
- 하지만 같은 root의 hierarchy local matrix, cached world matrix,
  `IFabricHierarchy.get_world_xform()` current matrix는 모두 valid/present였고 PhysX baseline과
  position/quaternion이 정확히 같았다.
- Render leaf `/World/envs/env_0/Sponge/geometry/mesh`도 hierarchy local/current/cached
  matrix가 모두 valid/present였다.
- Authored static root→mesh relative는 identity, reset-stack `false`; independent baseline
  reconstruction max-abs error는 `0.0`이었다.
- PhysX getter 전후 clock은 두 journal record 모두 불변이었다. Root/mesh current/cached
  matrix도 getter 전후 동일했다.

즉 실제 D364가 반증한 것은 “Fabric이 없다”가 아니라 **이 scene에서 optional compatibility
`_world*` 속성을 필수 직렬 계층으로 둔 측정 모델**이다. 이 속성을 건너뛰고 이미 실제로
존재하며 PhysX baseline과 일치한 hierarchy local/current/cached 경로를 직접 측정해야 한다.

## 16. 실제 baseline 시각 검사

원본 두 PNG를 original resolution으로 직접 검사했다.

- Primary: 1280x720, yellow component area `17,003px`, bbox
  `[628,299,90,209]`, PCA axis `90.983737deg`, upright true, toppled false.
- Opposite: 1280x720, area `17,010px`, bbox `[562,299,90,209]`, PCA axis
  `89.036711deg`, upright true, toppled false.

두 시점 모두 노란 원통과 로봇 말단을 선명하게 보여 주며 같은 baseline upright 자세를
확인한다. Pose write 전에 멈췄으므로 post-write/post-forward image, Korean sheet, RRD/RBL,
Rerun screenshot은 없다. 이번 operational verdict는 공간 판정이 아니라 attribute schema
prerequisite FAIL에 의존하므로, 존재하지 않는 final spatial subject를 꾸며 Rerun을 만들지
않았다.

Sources:

- `claudedocs/runtime_logs/grasp_track/g0a_d364/d364_baseline_primary_actual_isaac.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d364/d364_baseline_opposite_actual_isaac.png`
- `claudedocs/runtime_logs/grasp_track/g0a_d364/d364_manual_baseline_visual_inspection.json`

## 17. 종료 판정과 다음 승인 경계

Operational verdict:

`D364_PREWRITE_OPTIONAL_FABRIC_COMPATIBILITY_ATTRIBUTE_MISMODELED_FAIL_STOP`

D364는 render propagation PASS/FAIL을 판정하지 못했다. `localization_verdict=null`이다.
Pose write와 explicit forward가 모두 0이므로 D363의 실제 단절 위치는 여전히 미확정이다.
D362 physical sub-verdict는 바뀌지 않고, q5/contact/cap-rim/grasp/target-IK science도 모두
재실행·재판정되지 않았다. `g0a_pass=false`다.

Supervisor가 OS worker exit code `0`을 관찰했어도 worker exception artifact와 terminal
`worker_exception:stop`, worker summary 부재가 실제 완료 권위다. Kit shutdown이 process
code를 0으로 끝낼 수 있으므로 이후 case는 **exit code 단독을 성공으로 쓰지 않는다.**

다음 최소 후보는 별도 승인 D365
`[hierarchy_current_render_cache_propagation_localization]`이다. D364 raw baseline을 상속하고
없는 compatibility `_world*`는 negative/optional diagnostic으로만 기록한다. 선형 경로는
다음 네 화살표만 사용한다.

`AssetData cache → independent PhysX tensor view → IFabricHierarchy root/mesh current →`
`Boundable mesh cached worldMatrix → Hydra pixels`

D365도 frozen D362 final pose, cylinder root pose write 1회, public `SimulationContext.forward()`
1회, controlled physics/q5/contact `0`, actual worker 1회/no retry만 허용해야 한다. D364와 같은
output을 다시 실행하거나 덮어쓰지 않는다. D365 승인 전에는 코드 작성이나 실행을 하지
않는다.

Primary failure completion:

- `claudedocs/runtime_logs/grasp_track/g0a_d364/d364_prewrite_fail_completion.json`
  SHA-256 `845f3b392b986f4354618f9191c953ee3919e6917ed1db1f382e7f6654810211`
