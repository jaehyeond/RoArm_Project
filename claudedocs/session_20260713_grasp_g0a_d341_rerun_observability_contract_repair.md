# Session 2026-07-13 - D341: Rerun observability completion-contract repair

Pre-runtime status: `D341_PRE_REGISTERED_OBSERVABILITY_PENDING`

이번 case의 신규 변수:
`[rerun_observability_completion_contract]` (정확히 1개, measurement-only)

## 1. 무엇을 왜 하는가

D340의 과학 판정은
`D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP`이며 그대로 유지한다.
D340은 13개 fixed-point 후보에 대한 수치/해시 JSON과 PNG를 남겼지만,
기존 RRD는 다음 관측성 완료조건을 만족하지 못했다.

- 실제 파일은 `2,480,049` bytes이며 sha256은
  `8eb3d6130330334b9d6b457468cd4bb59097114c693cb7caa2e33a8f5993fe47`이다.
- 기본 `rrd verify --check-footers true`는 FAIL한다.
- RRD 안에는 D340 판정 대상인 13개 part의
  source/instance/prototype/candidate geometry가 없다.
- 파일 생성과 PNG 생성은 확인됐지만, RRD의 spatial subject를 실제로
  열어 판독했다는 별도 관찰 증거가 없다.
- D340 session의 RRD hash 문자열은 63자 오기이다. 원문은 immutable하게
  보존하고 D341에서 실제 파일/summary의 일치하는 64자 hash를 정정 기록한다.

D341은 이 실패에 대한 reactive measurement-contract repair다. Immutable
D340 JSON에서 표시용 geometry를 읽어 새로운 D341 RRD를 만들 뿐, D340을
rerun하거나 collision geometry를 다시 계산하지 않는다. Rerun은 관찰 계층이며
원본 callback 배열/JSON/hash가 계속 bit-exact 과학 권위다.

## 2. Rerun 강제 계약과 정확한 실행 타이밍

공간 또는 시간 판단이 verdict에 영향을 주는 case는 다음 순서를 모두
통과해야 한다.

1. 정확한 `rerun-sdk/CLI==0.34.1`과 Isaac 호환 pin을 확인한다.
2. 첫 user log 전에 전용 `RecordingStream`에 footer-enabled file sink를
   연결한다. Batch/Isaac 기본은 save-only이며 live Viewer는 선택사항이다.
3. 고정 Blueprint를 RRD에 active로 embed한다. 같은 Blueprint의 `.rbl`
   export도 별도로 남긴다.
4. generic frame만이 아니라 실제 decision subject를 기록한다. Cook case는
   source/instance/prototype/candidate를 별도 entity로, physics case는 실행한
   전체 step timeline과 tool/object/contact/force/decision scalar를 기록한다.
5. 각 spatial entity는 선언된 named coordinate frame을 가져야 한다.
6. flush 후 `RecordingStream` context를 종료해 sink/footer를 확정한다.
   이보다 앞서 artifact gate나 screenshot을 실행하면 안 된다.
7. 확정된 RRD에 대해 footer verify, stats, verbose schema, exact non-system
   entity set, exact timeline set, required component set을 검증한다.
8. embedded active Blueprint로 headless screenshot을 만든다. 외부 `.rbl`은
   검증된 layout export이지, 이미 active인 embedded Blueprint를 CLI가
   override했다는 증거로 사용하지 않는다.
9. screenshot 생성 성공은 renderability일 뿐 inspection이 아니다. 이미지를
   실제로 열어 관찰한 항목과 한계를 별도 report에 기록한 뒤에만
   `completion_contract_pass=true`로 분류한다.

Pure file/hash/schema audit처럼 공간·시간 판단이 전혀 없는 경우만 Rerun을
생략할 수 있고, 그 이유를 session doc에 적어야 한다. Training scalar 전체는
전용 tracker가 담당하고, Rerun은 sampled spatial rollout/trajectory를 담당한다.

## 3. 신규 변수 및 파라미터 증가 감사

- 신규 변수 수: `1` (Variable Ladder 제한 내).
- 신규 물리 변수: `0`.
- 기존 물리/target/controller/solver/object/contact 파라미터 증가·변경: `0/0`.
- decomposition 파라미터 증가·변경: `0/0`.
- tolerance/threshold 완화: `0`.
- controlled physics steps: `0`.
- collision asset writes: `0`.
- attempt3: absent and forbidden.

D341 harness는 `isaac`, `omni`, `pxr`, simulation step API를 import하지 않고,
`maxConvexHulls`, `hullVertexLimit`, `voxelResolution`, `errorPercentage`,
`minThickness`, `shrinkWrap`를 읽거나 설정하지 않는다. 입력은 D340의
canonical JSON/RRD와 session file뿐이며 출력은 새 `g0a_d341/` 아래로 제한한다.
따라서 52 meshes는 `13 parts x 4 variants`의 Float32 display copies이지
52개 새 geometry 변수나 decomposition 시도 횟수가 아니다.

## 4. Immutable inputs와 pre-runtime pins

- Boot HEAD: `2c8a25f689bd7c7f7f3927a956755c8642764d81`.
- D340 output inventory: `33` files, path/bytes/mtime/hash digest
  `ce77a75e9ee8ba559e57bf443e4eee587352498bbb154f91f06bb81b4462c8ab`.
- D340 session sha256:
  `24cbb2cf0a718d119bfd30121a536d4e6a1007738718eef44f0664820fc49762`.
- D340 candidate JSON:
  `f288d5232f039e58ccd209f332ebfabbf9fec137e746e97d9a3c58688420ef86`.
- D340 summary JSON:
  `ba62c879e78bfa7db47b003e1bfdd4ee2bd4ff250b083e423f94ebec67992163`.
- D340 postrun audit JSON:
  `2d6bc90a71d9ec407206ade7c32069ad209f8a0ba40cb44a40f42281318a1207`.
- D340 RRD:
  `8eb3d6130330334b9d6b457468cd4bb59097114c693cb7caa2e33a8f5993fe47`.
- D341 harness:
  `5ad636e93824b6a30e54619c3dec2d3bce5a6c446dc07fef53fcdd9bb09254e3`.
- `roarm_rl/viz_debug.py`:
  `6c9c41cfd68978738f103832c760a1e2c202cdebec6f8a0c4ad94bbd5d796517`.
- `roarm_rl/rerun_contract.py`:
  `90559c931bc753be97def463841d41426a2f1bd8e5ddd15a2a2ab08fb54a2e60`.
- Rerun contract tests:
  `1a8c186c356fe93ffe204cc19b3f43eeb613f9e081ca2935233016f8d921b1d1`.
- Exact RRD contract digest:
  `a0c9d7eccaf6585e34cd5c70494fb441be6e24df9b7f34c30fd84d7166bee6bb`.
- Pins: `rerun==0.34.1`, `numpy==1.26.0`, `psutil==5.9.8`.

The preregistration additionally pins this pre-runtime session document and
the active `START_HERE.md`. The harness refuses to start if any pin differs.

## 5. Registered scientific subject and exact archive contract

- source parts: `13`.
- variants per part: `4` (`source`, `instance`, `prototype`, `candidate`).
- mesh entities: `52`.
- Float64 scalar rows: `143`.
- event rows: `67`.
- exact non-system entities: `254`.
- exact timelines: `blueprint`, `event_idx`, `log_time`, `part_idx`.
- body-local frames: `link5_body_local`, `gripper_link_body_local`, each with
  `parent_frame=tf#/`.
- required schemas include `Transform3D`, `CoordinateFrame`, `Mesh3D`,
  `Scalars`, `TextLog`, and mesh metadata `TextDocument` components.

The fixed Blueprint exposes eight independent decision views: link5 and
gripper each have separate source/instance/prototype/candidate panels. Metrics
and events are visible below. The registered headless size is `2400x1400`.
Exact entity/schema validation, not a single screenshot, proves that all 52
registered rows exist; the independent panels make every variant visibly
reviewable without entity-overlay ambiguity.

## 6. Registered sequential procedure

1. Require the output folder to contain only `d341_preregistration.json`.
2. Verify HEAD, executable hashes, active-state/session hashes, D340 inventory,
   source hashes, version pins, absence of attempt3, subject counts, and exact
   RRD-contract digest.
3. Read immutable D340 candidate/postrun JSON and build 52 display meshes,
   143 scalars, and 67 events. Do not recook or transform scientific evidence.
4. Record one save-only D341 RRD with pre-log sink and embedded fixed Blueprint;
   finalize it before any validation. Export and verify the matching RBL.
5. Require footer, exact entities/timelines/components, forbidden escaped-path
   absence, and headless renderability all PASS.
6. Copy the finalized D341 RRD, truncate exactly
   `min(4096, max(1, len(good_rrd)//10))` trailing bytes, and require the same
   validator to return `pass=false`, `footer_manifest_present=false`, and the
   explicit `RRD footer verification failed` error. Never truncate D340 or the
   good D341 RRD.
7. Recompute the exact D340 inventory/session hashes and require no change.
8. Preserve the automated summary with
   `manual_visual_inspection_pending=true`; do not overwrite it.
9. Open the generated screenshot, record bounded observations in separate
   JSON/Markdown, then write a separate final completion summary.

## 7. Failure-capable experiment and no-physics justification

The finalized-copy truncation is a registered perturbation that can fail the
case: if the validator accepts it, the new completion contract is not trusted.
This is reactive hardening caused by D340's observed missing-footer and missing-
subject evidence. Running Isaac physics could not test this measurement defect,
would introduce an unregistered physical variable, and is therefore explicitly
forbidden. This is the session-progress-rule justification for no training or
physical perturbation in D341.

## 8. Pre-runtime checks

- Python compile: PASS.
- `git diff --check`: PASS.
- Dedicated Rerun unit/integration suite: `6/6` PASS.
- Normal finalized footer positive control: PASS.
- Footer-truncated negative control: rejected.
- Unexpected entity exact-set control: rejected.
- Undeclared spatial coordinate frame control: rejected before sink creation.
- URDF leading-slash escape regression control: PASS.
- Actual D340-shaped scratch artifact: `52/143/67`, exact `254` entities and
  all four timelines PASS; all eight independent spatial views plus the metric
  dataframe and events rendered at `2400x1400`.
- No physical/decomposition/threshold-setting token or Isaac/Omni/PXR import
  found in the D341 harness.

The base-conda `requests` compatibility warning and Rerun's NumPy-1 deprecation
warning were non-fatal. The mandated Isaac pins remain unchanged and all six
contract tests passed inside `isaaclab`.

## 9. Registered command

```bash
conda run -n isaaclab --no-capture-output python \
  sim_scripts/cyl34_top_view_d341_rerun_observability_contract_repair.py
```

## 10. Scope guards and result placeholder

- D340 evidence and verdict are immutable.
- D341 does not repair the D340 authored-coordinate proof and does not create
  attempt3. That separately approved physical path moves to D342 reserve.
- `g0a_pass=false`; G0b/RL/PPO/ladder remain blocked.
- No commit or push without explicit user request.

Runtime result: pending one registered execution and separate visual inspection.

## 11. Reactive invocation amendment after attempt0

The exact registered command was invoked once, but Python exited at the first
project import with `ModuleNotFoundError: No module named 'roarm_rl'`. Running a
file under `sim_scripts/` made that directory `sys.path[0]`; the repo root was
not on the import path. No D341 harness function ran, no RRD/RBL/screenshot or
summary was created, and the D341 folder still contained only preregistration.
D340 inventory and attempt3 absence remained unchanged.

This observed invocation-contract failure licenses one reactive control fix:
insert the script's resolved repo root at the front of `sys.path` before the
two `roarm_rl` imports. No experiment variable, geometry, parameter, gate,
output, or command changes. The original pre-runtime harness hash above is
preserved as attempt0 evidence; the amended harness/session hashes are pinned
in the preregistration amendment before the one effective execution.

## 12. Effective runtime result

Final status: `D341_RERUN_OBSERVABILITY_COMPLETION_CONTRACT_PASS`.

After the reactive import-path amendment, the same registered command entered
the harness exactly once and returned exit code `0`. The automated stage ended
at the intentionally intermediate verdict
`D341_RERUN_OBSERVABILITY_AUTOMATED_PASS_MANUAL_INSPECTION_PENDING`; its summary
was preserved without overwrite. A separate visual report and separate final
summary then closed the completion gate.

### 12.1 Automated archive evidence

- Good RRD: `742,647` bytes, sha256
  `48d76107c92e3fde53ae0281aafa8d0e8b08b5315a7ad80c385afbc9486de593`.
- RBL export: `96,376` bytes, sha256
  `33dd1413f58f00c5c3bec0cf8a03eb4dff84d507c4364b6dc27dafa8cf5bde0a`.
- Sink attached before user logs: PASS; flush/context finalization: PASS;
  footer manifest: PASS.
- Exact non-system entity set: `254/254`, no unexpected entity.
- Exact timelines: `blueprint`, `event_idx`, `log_time`, `part_idx`.
- Required per-path component contract: PASS.
- Registered subject: `13 x 4 = 52` meshes, `143` Float64 scalar rows,
  `67` event rows.
- Embedded fixed Blueprint headless render: PASS; external RBL footer verify:
  PASS. The RBL is a verified export, not claimed as a CLI override.
- Screenshot: `10,146,168` bytes, sha256
  `dc3f1d82e05d324f2fd032a0caca5077a04cbd3dc145f936bf897e0c1c8450ee`;
  registered logical size `2400x1400`, raster `4800x2800`.

Automated summary:
`claudedocs/runtime_logs/grasp_track/g0a_d341/d341_rerun_observability_automated_summary.json`
(sha256
`ffdb799d7f46086654c7e72c28f3a6a02a9db6a08b1d943c6dcc704790c04597`).

### 12.2 Failure-capable negative control

The harness copied the already-finalized good D341 RRD and removed exactly
`4,096` trailing bytes. The validator returned all three required predicates:

- overall validation `pass=false`;
- `footer_manifest_present=false`;
- errors contain `RRD footer verification failed`.

The negative control therefore PASSed by being correctly rejected. Its sha256
is `0dcfd7012de29510676aee7f43b4034f5200932ee8bd77ce35ae3c6b1b4eb0d5`.
Neither D340 nor the good D341 RRD was truncated.

### 12.3 Actual visual inspection

The generated PNG was opened at original detail. Observed:

- all eight independent titles and non-empty spatial panels were visible:
  source/live-instance/prototype/candidate for link5 and gripper;
- the three x1 variants were not overlaid in one panel;
- the Float64 metric Dataframe showed `part_idx` plus numeric scalar cells;
- the event table showed INFO/WARN rows including the retained D340 FAIL and
  the D341 no-physics/no-asset-mutation stop message;
- the upper-right Viewer loading notification did not hide a required title or
  the displayed candidate geometry.

This inspection does not infer bit equality from pixels and does not claim that
all 52 parts or 67 events are simultaneously visible. Exact archive schema and
canonical JSON/hashes carry those claims. Manual reports:

- `d341_manual_visual_inspection.json`, sha256
  `1b18b272d139cd989d08f0359a07235bfb9075f42b4ff0fe97334f95797448d9`;
- `d341_manual_visual_inspection.md`, sha256
  `5f014aee926b968cbd61708d6d9f0b666cc10e8bd0209d4c09ff3b76f9a7386b`.

### 12.4 D340 correction and immutable boundary

D340's actual RRD and canonical D340 summary agree on sha256
`8eb3d6130330334b9d6b457468cd4bb59097114c693cb7caa2e33a8f5993fe47`.
The 63-character session literal remains untouched as historical evidence.
The legacy RRD is decodable/non-empty but fails the footer-completion contract
and lacks D340's 13-part scientific subject, so the earlier broad phrase
"PNG/RRD inspected" is superseded: the D340 PNG was inspected; the RRD was
generated but was not completion-certified or visually inspected.

D340 remained `33 -> 33` files with identical full path/bytes/mtime/hash rows
and digest
`ce77a75e9ee8ba559e57bf443e4eee587352498bbb154f91f06bb81b4462c8ab`.
D340 session hash also remained exact. No attempt3 appeared.

### 12.5 Parameter/variable audit

- case variables: exactly one measurement-only variable;
- physical/decomposition/target/control/solver/tolerance parameter increases:
  `0`;
- existing parameter changes: `0`;
- threshold relaxations: `0`;
- collision asset writes: `0`;
- simulation started: `false`;
- controlled physics steps: `0`.

The 8-panel layout, `2400x1400` logical capture size, and metric Dataframe are
display settings inside the one registered observability variable. They are not
scientific parameter increases.

### 12.6 Verdict in everyday language

Rerun was unnecessary inside D340's numerical hash decision, but once D340
claimed a replayable visual artifact it needed a complete observability proof.
D341 now makes that proof enforceable: record the real subject before closing,
finalize the file, reject missing footers and schema surprises, render a fixed
decision layout, and require someone to actually look at it before saying
"complete".

This is an observability-contract PASS only. It does not repair D340's
authored-coordinate hash gate, does not certify or change a collision asset,
and does not advance the robot task. D340 remains
`D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP`; `g0a_pass=false` and
G0b/RL/ladder remain blocked. The physical critical path is the separately
approved D342 authored-coordinate-stream repair.

Final completion summary:
`claudedocs/runtime_logs/grasp_track/g0a_d341/d341_rerun_observability_completion_summary.json`
(sha256
`18d38552f30a2d3908fd10341eb2fb244a6733f37a51abe4467745b44abb0746`).
