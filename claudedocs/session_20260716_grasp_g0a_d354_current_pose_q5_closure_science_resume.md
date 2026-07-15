# D354 — current-pose q5 closure-science resume

날짜: 2026-07-16 KST  
사전등록 당시 상태: 사용자 승인 / 아직 prepare·Isaac 실행 전  
이번 case의 신규 운영·관측 변수:
`[positive_asset_write_immutability_aggregation]`  
이번 case의 신규 과학 변수: `[]`  
이번 case의 신규 물리 변수: `[]`  
상속한 D351 과학 변수:
`[moving_jaw_actual_contact_surface_binding, frozen_pose_q5_closure_sweep]`

## 1. 무엇을 왜 확인하는가

D353는 새 process의 reset 뒤 pending PLAY에 conditional main-thread
`Timeline.commit()`을 정확히 한 번 적용하면 timeline/world를 전진시키지 않고 PAUSE를
확정할 수 있음을 증명했다. 그러나 q5 과학은 실행하지 않았다.

D354는 그 control bridge를 새 forward-only process에서 다시 검증한 뒤, D351이
사전등록했지만 한 번도 진입하지 못한 다음 질문만 재개한다.

1. D350 q0-q4와 원통 pose를 동결하고 q5만 OPEN에서 CLOSED로 zero-time 기록할 때,
   실제 moving inner jaw가 원통 `barrel_interior`를 먼저 만나는가?
2. raw authored mesh와 D348 callback-topology live proxy가 첫 접촉 bracket·surface·feature에
   합의하고, 그 전 corridor에서 table과 엄격히 분리되는가?

positive 결과도 single-close 물리 case의 후보일 뿐, 양면 동시접촉·힘/마찰·force
closure·grasp·hold/lift·G0a PASS가 아니다. `g0a_pass=false`를 유지한다.

## 2. 동결 입력과 과학 계약

- Base `HEAD == origin/master`:
  `b7beb91997859a5ddb2b0407388e80aed45898dc`.
- 출력: `claudedocs/runtime_logs/grasp_track/g0a_d354/`.
- seed `33201`; q5 convention `0=CLOSED`, `1.5413rad=OPEN`.
- q0-q5 OPEN Float32:
  `[0.03750238195061684,0.542945146560669,1.9687392711639404,
  0.18299327790737152,0.0,1.5413000583648682]rad`.
- 원통 Float32 pose:
  position `[0.30000001192092896,0.0,0.03288299962878227]m`,
  quaternion `[1,0,0,0]`.
- q5 anchors: Float32 `linspace(OPEN,CLOSED,33)`, 단조 감소.
- adaptive bracket: `2*Rmax*sin(abs(delta_q)/2)`, 종료 폭 `<=1e-6rad`, 최대 깊이 `32`.
- raw first, live second. Live는 D348 callback polygon topology surface proxy이며 direct
  PhysX narrowphase라고 부르지 않는다.
- inherited gates: OPEN clear `0.1mm`, raw/live spatial fidelity `0.5mm`, table strict
  `>0`; 새 grasp/alignment 허용값은 없다.
- moving inner authored faces `672..1164`, paired outer negative control
  `13205..13697`; live-inner `40` triangles / `17` parts / frozen key hash를 그대로 쓴다.
- target/IK/path, q0-q4/object, asset, decomposition, gate/tolerance, material, mass,
  actuator, renderer, solver, physics configuration을 바꾸지 않는다.

원 D351 harness·session·parameter SHA-256은 각각
`3c4501885af7590f5883b36666c984ce88728a24d40451ea0a600660a386107d`,
`20367375e05ce8cffb47f86ff0c1645a3544f5bf62516fe2e16a98919c356a06`,
`98b5778e826d411f37606dd724093a1ff292040d8c1d350db3781508735502e2`로
prepare에서 다시 확인한다.

## 3. D353 bridge 상속과 q5 gate

D353 attestation SHA-256
`4758e9b09b3298ae0dd292f327bb37b474a624d3f0190629968c55cb091393d5`와
supervisor SHA-256
`65c57e69f017d7d7afbb5fd03b10b56e87bb1bbc442b1351a25c18a0a55a31a5`를
immutable prior evidence로 고정한다. 과거 attestation의 step `0`을 D354에 복사하지 않는다.

새 worker의 두 q5 evaluator 진입점은 처음에 같은 fail-closed gate 뒤에 둔다.

1. reset 뒤 `/app/player/playSimulations=false`를 쓴다.
2. 실제 PLAY일 때만 `timeline.pause()`를 한 번 요청한다.
3. before와 pending가 `(playing,stopped)=(true,false)`이고 D353 reset baseline의
   timeline/SimulationContext/joint/object bits와 exact인지 확인한다.
4. caller MainThread에서 `Timeline.commit()`을 정확히 한 번 호출한다.
5. PAUSE callback 한 개가 commit window 안에 있고 post가 `(false,false)`인지 확인한다.
6. custom counter, timeline time/tick, SimulationContext time/index, joint/object bits,
   q5 invocation/state-write 수가 전부 불변인지 확인한다.
7. corrected audit, live `128/128`, raw binding 뒤에도 같은 pre-science baseline과 PAUSE를
   확인한다. 이미 PAUSE인 경계에는 pause나 commit을 반복하지 않는다.
8. 위 계약과 immutable input이 모두 PASS일 때 첫 evaluator invocation에서만
   `science_arm=false -> true`를 durable marker와 bridge JSON에 기록한다.

`simulation_app.update`, next-frame, `commit_silently`, frame forward/rewind 또는 physics
step으로 PAUSE를 확정하지 않는다. Viewer UI pump는 과학 측정과 attestation 후보가 만들어진
뒤에만 허용하며 매 pump의 zero-step guard를 유지한다.

## 4. 등록된 실행 순서

1. 최신 Git, D351 core, D353 17-file inventory/attestation, D348-D350 input, sidecar,
   Python pins, RTX4090 GUI launcher를 확인한다.
2. fresh worker를 외부 supervisor가 process group으로 한 번만 시작한다. `120s` marker
   inactivity / `900s` total watchdog, SIGUSR1 faulthandler `5s`, SIGTERM `30s`, 최종
   SIGKILL 정책을 사용하고 자동 재시도하지 않는다.
3. D353 conditional commit bridge와 pre-science live/raw binding을 완료한다.
4. q5 gate를 한 번 arm한 뒤 원 D351 evaluator를 호출한다. 33 anchors, raw/live adaptive
   first-contact bracket, table corridor, moving surface binding, D350 fixed digest, CLOSED→OPEN
   endpoint repeat query, classification 순서를 바꾸지 않는다.
5. 각 unique q5 row는 q0-q4/object bits, requested q5 bits, custom counter `0`, PAUSE,
   timeline time, SimulationContext time/index 불변을 요구한다.
6. evaluator invocation 수, cache-miss state-write 수, primary unique measurement rows,
   repeat-cache rows, auxiliary `_set_state_only` writes를 별도 counter로 기록한다.
7. measurement JSON과 sweep CSV를 쓴 뒤 D351의 Viewer/Rerun 계약을 실행한다.
8. 실제 Isaac Viewer에서 OPEN과 resolved raw last-clear pose 또는 명시적 OPEN fallback의
   PhysX collider, colored 64+64, inner patch/chord/axis를 네 장 캡처한다. collider/guide
   visibility는 session-only UI 표시이며 prior value로 exact 복원한다.
9. finalized RRD/RBL은 매 unique q5 row의 dense live surface, moving patch, pivot/axis,
   raw/live witness/gap, 다섯 scalar와 event를 `step=0..N-1`에 정확히 한 번 보존한다.
10. runtime summary를 O_EXCL write/fsync하고 exact reread한 별도 D354 attestation에서만
    controlled physics steps 후보 `0`을 권위값 `0`으로 연다. 중단/예외면 `null`이다.
11. worker 종료 후 Viewer 4장과 Rerun 1장을 original resolution으로 실제 열어 검사한다.
    finalize는 PID absence, PNG 세 번 안정성/full RGBA decode, RRD machine contract,
    evidence hash chain, manual inspection을 모두 요구한다.

## 5. verdict 어휘

- `D354_CURRENT_PREGRASP_BARREL_CLOSURE_ELIGIBLE`: D351 positive inner+barrel, raw/live
  agreement, pinch-facing/order, continuous table-clear certificate가 모두 완전함.
- `D354_CURRENT_POSE_CLOSURE_GEOMETRY_REPAIR_RECOMMENDED`: inner+barrel과 competitor
  exclusion은 완전하지만 pinch-facing/order 또는 strict table-clear가 실패함.
- 그 밖의 input/order/binding/competitor/EPA/surface identity/zero-step/observability
  불완전은 해당 `D354_*_FAIL_STOP`; 대칭 negative certificate 없는 cap/rim/non-inner
  endpoint를 REPAIR로 확대하지 않는다.

위 어휘는 원 D351 verdict를 D354 forward-only 실행에 이름만 매핑한다. 과학 gate와
분류 논리는 바꾸지 않는다.

## 6. count와 설정 의미

- D351의 원 aggregate는 금지 사실인 `asset_write=false`를 그대로
  `all(values)`에 넣어 나머지 관측 계약이 전부 참이어도
  `immutability_pass=false`가 되는 부호 오류가 있다. D354는 원 payload와
  false trigger를 먼저 확인한 경우에만 같은 사실을 양수 표현
  `asset_write_forbidden_and_absent=true`로 바꾸어 observability aggregate와
  process exit를 다시 계산한다. 이것이 유일한 신규 운영·관측 변수이며,
  거리·surface·contact order·table corridor·과학 verdict 로직은 바꾸지 않는다.
- `evaluator_invocations`: cache hit를 포함한 wrapper 진입 수.
- `evaluator_state_writes`: cache miss로 원 evaluator가 실제 state-only write를 한 수.
- `primary_unique_rows`: measurement의 `execution_count`.
- `repeat_unique_rows`: 별도 CLOSED→OPEN repeat cache의 두 row.
- `auxiliary_state_only_writes`: surface binding/display/Viewer가 같은 zero-step primitive로
  pose를 기록한 수; 과학 sample 수로 부풀리지 않는다.
- `sim.forward()`와 `scene.update(dt=0)`은 state propagation이며 `sim.step()`이 아니다.
  nevertheless D354 own counter/time/clock attestation이 불완전하면 steps는 `null`이다.
- GPU telemetry는 active-time 관찰일 뿐 warp occupancy 또는 과학 판정 권위가 아니다.

## 7. 불변·금지 경계

- D351-D353 기존 파일 overwrite/rerun `0`; D354 output만 생성한다.
- asset write/cook/property query/decomposition 변경 `0`.
- target/IK/path/gate/material/mass/actuator/renderer/solver/physics 변경 `0`.
- controlled physics step/timeline PLAY/dt>0 update `0`.
- settle/ten-trial/G0b/RL/PPO/VLA/ladder `0`; `g0a_pass=false`.
- user-owned `claudedocs/lab_meeting/20260715/d334_collision_table/`는 read-only exact이며
  수정하지 않는다.
- commit/push `0`.

이번 session은 결과에 따라 current-pose closure eligibility, repair recommendation 또는
FAIL_STOP이 달라지는 실제 q5 perturbation evaluation이다. 실패 가능한 평가이므로
AGENTS.md Session progress rule을 충족한다.

## 8. 실행 결과

### 8.1 prepare와 단일 실행

- Base `HEAD == origin/master`는 실행 전후
  `b7beb91997859a5ddb2b0407388e80aed45898dc`였다.
- `--stage prepare`는 PASS했고 preregistration SHA-256은
  `dc3defad370f394008927a6dc261fcf9aac8b7b2b60a5d34c950c696c1f45349`다.
  prepare가 동결한 harness / `START_HERE.md` / 이 session의 당시 SHA-256은
  `f9676df74a61cdeadaccdbf3437f0304e4333c7dcc65d00b5fd40a2a4344b1ae` /
  `4be742fff700ab752bf62b983ed84d40aa446353f1d5bbb5db7dc45af87342d9` /
  `227eaa21f523a9012f64afc0ee87f298e5cc9aeb9b2cc990b2ec0e4eda914cb5`였다.
- 실제 validate는 `DISPLAY=:1`, `headless=false`, `cuda:0`, RTX 4090으로 정확히 한
  번 실행했다. worker PID `3690728`은 exit `0`으로 reap됐고 process group도 사라졌다.
  watchdog, SIGTERM/SIGKILL, runtime exception, automatic retry는 모두 없었다.
  marker는 `1242/1242` valid였고 마지막 `SimulationApp.close` start는 worker elapsed
  `153.70643517s`였다.
- 로그에는 startup `12.825s`의 비치명적
  `isaacsim.core.cloner.impl.cloner: Failed to clone in Fabric` 한 줄이 있다. 그러나 그 뒤
  환경 구성, 70-row 과학 측정, Viewer/RRD, attestation, 정상 close가 모두 완료됐다.
  따라서 이 메시지는 보존하지만 이번 실행의 실패 원인으로 분류하지 않는다.

### 8.2 D353 bridge와 zero-step 권위

- 새 process에서 commit attempt/call은 정확히 `1/1`이었고 conditional
  `PLAY -> pending PLAY -> PAUSE` bridge가 PASS했다. 이미 PAUSE인 live/raw 경계에는
  추가 pause/commit을 하지 않았다.
- evaluator wrapper 진입은 cache hit 포함 `377`, 실제 cache-miss state write는
  `72/72`, primary unique measurement row는 `70`, repeat row는 `2`이며 그중 distinct
  repeat cache는 `1`이었다. auxiliary state-only write는 `13/13`이었다.
- `d354_science_resume_summary.json`과 inherited raw audit의 legacy key
  `d352_q5_evaluation_count=377`은 이름이 잘못 붙은 비권위 필드다. 그 값은 바로 옆의
  D354 evaluator invocation `377`을 복제한 것이며 D352 과거 q5 count는 계속 `0`이다.
  D354 권위 count는 `d354_q5_evaluator_invocations=377`, D353 상속 과학 count는
  attestation의 `inherited_d353_science_evaluation_count=0`으로 읽는다.
- Viewer UI update는 `11161`, zero-step guard failure는 `0`이었다. 모든 measurement
  counter/state guard, 최종 PAUSE-not-STOP, timeline/SimulationContext/joint/object
  sentinel이 PASS했다.
- runtime summary는 먼저 controlled steps를 `null`, 후보를 `0`으로 기록했다. 그 JSON을
  O_EXCL write/fsync하고 exact reread한 별도 attestation만 D354 controlled physics steps를
  권위값 `0`으로 확정했다. attestation SHA-256은
  `1975df11b13a774b89f953991d7fdac6e38d795e81a9535850b147d823740d20`다.

### 8.3 q5 closure 측정

- raw와 live 모두 같은 Float32 first-contact bracket을 얻었다.
  - clear q5 `1.0269782543182373rad`
  - overlap q5 `1.0269775390625rad`
  - 폭 `7.152557373046875e-7rad`, adaptive depth `16`
- raw signed distance는 clear `+0.0010050812803802547mm`, overlap
  `-0.000988475720559677mm`였다. live는 각각
  `+0.0010049780471806762mm`, `-0.0009864198978583663mm`였다.
- raw/live contact q 차이는 `0rad`, contact surface travel 차이는
  `0.00004817170331236983mm`, 두 contact endpoint 최대 거리 차이는
  `0.000002055822701310661mm`였다. 두 표현의 contact-order certificate는 각각 PASS했다.
- OPEN에서 raw/live가 모두 clear였고, q5 decrease rotation axis/pivot 계약과 fixed link5
  q5 invariance도 PASS했다.
- first contact 전 table corridor는 OPEN부터 clear endpoint까지 연속 인증됐다. endpoint
  구간의 최소 실제 분류 clearance는 `65.42070265676648mm`, 보수적 연속 인증 최소 strict
  margin은 `63.22081483325994mm`였다. 전체 sweep 최소 `40.86601149340871mm`는
  precontact 판정이 아닌 진단값이다.

### 8.4 과학 FAIL_STOP의 정확한 원인

- raw와 live의 clear endpoint는 모두 cylinder-local `z=+0.045m`에 정확히 놓여
  `cap_or_rim_boundary`로 분류됐다. 바로 다음 overlap endpoint는 raw
  `z=0.044999618601561694m`, live `z=0.044999619394590046m`의
  `barrel_interior`였다.
- frozen 분류기는 새 tolerance 없이 strict z 순서만 사용한다. 따라서 raw/live 모두
  clear/overlap cylinder feature consensus가 false이고,
  `noninner_and_cap_competitors_excluded_over_full_bracket=false`였다.
- moving 접점이 frozen distal inner patch에 있다는 점, raw/live surface identity가
  unambiguous라는 점, 두 죠가 원통 중심의 반대편에 있다는 점, q5 decrease가 inward/고정
  surface 방향이라는 점은 PASS했다. 그러나 cap/rim competitor를 배제하지 못했으므로
  이 양성 항목만으로 barrel-first나 pinch/order PASS를 선언할 수 없다.
- 별도 binding gate도 false였다. immutable authored point/index/count stream과 face ID/order,
  vertex/edge/face set은 exact였지만 파생 identity가 불일치했다. 대표적으로 authored
  paired-XZ SHA는 frozen `917b7154...bcaf9`와 exact였으나 raw-derived paired-XZ vertex
  SHA는 `98ef77e6...18bbae`였고, derived vertex/triangle/patch digest 계약도 FAIL했다.
  이는 보존된 증거에서 asset mutation을 뜻하지 않으며, provenance/직렬화 의미를 별도로
  확인하기 전에는 surface binding PASS로 고쳐 쓰지도 않는다.
- 따라서 scientific contract는 false이고 최종 과학 verdict는
  `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`이다. frozen 어휘상
  `D354_CURRENT_PREGRASP_BARREL_CLOSURE_ELIGIBLE`도
  `D354_CURRENT_POSE_CLOSURE_GEOMETRY_REPAIR_RECOMMENDED`도 아니다.

### 8.5 실제 Viewer, Rerun, GPU 관찰성

- 실제 Isaac Viewer hold는 `120.00741406402085s`, hold 중 UI update는 `11072`, timeline
  intervention은 `0`이었다. OPEN과 resolved raw last-clear decision pose의 PhysX/colored
  64+64/side geometry 네 PNG를 `1280x720 RGBA`로 보존했다.
- Rerun SDK/CLI `0.34.1`, RRD `2,103,021B`, RBL `43,616B`는 footer-enabled verify가
  PASS했다. exact contract는 dynamic sample `70`, mesh `131`, point row `350`, arrow row
  `280`, scalar row `350`, event row `70`, entity/component path `279/279`였다. headless
  screenshot은 `4800x2800 RGBA`로 보존했다.
- 다섯 이미지는 `view_image original_resolution`으로 직접 검사했다. 실제 Isaac pose와
  collider, 두 64-part 집합, inner patch/chord/cylinder feature, Rerun의 full q5 동적
  surface/witness/timeline이 보였고 빈/손상 panel은 없었다. manual PASS는 과학 판정을
  override하지 않는다.
- GPU hardware contract는 RTX 4090 Laptop, compute capability `8.9`, `76` SM, warp size
  `32`, max threads/SM `1536`을 확인했다. telemetry `157/157` valid에서 GPU active-time
  min/mean/max는 `0/12.2356687898/42%`, VRAM은
  `2052/7025.3312102/7601MiB`, SM clock은 `210/1489.2993631/2385MHz`였다.
  clock/power/persistence/kernel/profiler 변경은 없었다. 이는 warp occupancy 측정이 아니며,
  one-env zero-step case에서 SM 포화를 위해 batch/physics를 바꾸는 것은 승인 범위가 아니다.

### 8.6 최종 상태와 해석 경계

- completion은 evidence hash chain, prior immutability, sidecar read-only, process absence,
  post-close PNG 안정성/해독, RRD/RBL, manual inspection을 모두 PASS했다.
  `completion_pass=true`, controlled physics steps `0`, `g0a_pass=false`다.
- completion summary SHA-256은
  `5cc70c8aa1e50532fa4ec27756496d6b9f9447156c56ef700084b44c16226f86`다.
  measurement / moving-binding / supervisor SHA-256은 각각
  `fd0d43c2a47abefad939fc0e980456cc396bbf3ba3bb104b28bcd999100f23ed` /
  `548d45ec4eb1dacbb4cbdefe2b64a3ed99ce72f4f5ffaaa6a9ee1e2b38756b15` /
  `ea54be6c636e49096ebef02d7b2a1ce903df9c8d7806ef771dd19a60196e57c7`다.
- `completion_pass=true`는 승인된 실패 가능한 측정과 관찰성 계약이 끝났다는 뜻이지
  current pose가 성공했다는 뜻이 아니다. 이 결과는 현재 pose의 barrel-first eligibility를
  인증하지 못하지만 물리적 grasp 불가능, target/IK repair 정당화, force closure 또는 G0a
  rejection까지 증명하지도 않는다.
- D354는 재실행하지 않는다. 다음 가장 좁은 후보는 새 Isaac/target 변경 없이 derived
  moving-jaw patch hash provenance를 원 stream으로 감사하는 별도 승인 offline case다.
  cap/rim boundary의 추가 기하 discriminator나 target/IK/path 변경은 그 뒤에도 각각 별도
  승인과 사전등록이 필요하다.
