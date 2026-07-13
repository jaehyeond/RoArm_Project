# Session 2026-07-13 - D342 authored-coordinate-stream contract repair

Pre-runtime status: `D342_PRE_REGISTERED_AUTHORED_STREAM_PENDING`

이번 case의 신규 변수:
`[authored_geometry_frame_contract]` (정확히 1개, measurement-only)

## 1. 무엇을 왜 하는가

D340은 13개 failing part의 instance/prototype fixed-point capture 자체는
통과했지만, `authored_hash_matches_d339_manifest`가 `0/13`이라 attempt3 전에
정지했다. D340의 post-run audit는 그 원인이 geometry 불일치가 아니라 서로
다른 coordinate/value stream을 exact hash로 비교한 데 있음을 보였다.

- D339 manifest는 USD에 직접 authored된 `Vec3f` points와 triangles를
  transform 전에 판독한 stream을 기준으로 한다.
- D340은 같은 points를 prim-local에서 body-local로 옮긴 뒤 float64
  canonicalization한 stream을 D339 hash와 비교했다.
- 변환 차이는 최대 `2.220446049250313e-16m`였지만 exact geometry hash는
  `13/13`, Qhull topology hash는 `10/13` 바뀌었다.

D342는 이 measurement contract만 수리한다. D339 attempt2 USDC를 직접
readback해 authored stream을 transform 전에 exact 비교하고, D340의
body-mapped/candidate arrays는 numerical containment/proximity에만 사용한다.
이번 case는 attempt3를 만들거나 collision representation을 변경하지 않는다.

## 2. 권위 있는 stream 정의

### 2.1 Direct authored exact domain

각 registered part의 authoritative direct stream은 다음과 같다.

1. D339 attempt2 physics layer의
   `/colliders/{body}/d338_convex_parts/{part}` prim을 PXR로 직접 읽는다.
2. `points`는 authored 순서를 유지한 `VtArray<Gf.Vec3f>`를 contiguous
   little-endian `<f4>` bytes로 해시한다.
3. `faceVertexCounts`는 모두 `3`이어야 하며, `faceVertexIndices`는 authored
   순서를 유지한 `<i8>` triangle stream으로 해시한다.
4. 동일 point 값을 transform/sort/dedup/Qhull 없이 `<f8>`로 승격한 bytes,
   triangles `<i8>` bytes, 그리고 둘의 결합 hash를 D339 hull manifest의
   `vertex_stream_sha256`, `topology_sha256`, `geometry_sha256`와 비교한다.
5. expected raw `<f4>` baseline은 D339이 실제 authoring에 사용한 cold1
   canonical arrays를 float32로 cast해 preregistration에 고정한다. cold2와도
   배열/triangle equality를 재확인한다.

JSON 문자열, Python float repr, transform 후 배열, Rerun Float32 display를
exact-hash 권위로 사용하지 않는다.

### 2.2 Body-mapped numerical domain

D339 live audit의 prim-to-body matrix를 direct points에 적용한 뒤 D340
`authored_x0`와 assignment/surface distance로 비교한다. 다음은 모두
`<=1e-9m` numeric gate다.

- prim-to-body identity delta
- mapped direct vs D340 x0 assignment-matched coordinate delta
- mapped direct vs D340 x0 symmetric solid distance
- D340 candidate x1의 x0 containment violation
- live consensus vs float32 candidate surface delta

`x0 -> candidate` directed distance는 수축량으로 기록하며 equality gate가
아니다. Body-mapped geometry hash는 positive evidence가 아니다.

## 3. 실패 가능한 대조군

두 negative control을 사전등록한다.

1. D340 legacy mixed-stream hash를 그대로 재현한다. Expected는 direct
   `13/13` exact PASS와 동시에 transformed D340 x0의 manifest hash match
   `0/13`, numeric frame equivalence `13/13`이다. Validator가 두 domain을
   구분하지 못하면 D342는 FAIL한다.
2. `link5/part_011`의 x-max authored vertex를 메모리 안에서만 `+10um`
   이동한다. Source 파일은 쓰지 않는다. Raw `<f4>` hash, manifest geometry
   hash, `1e-9m` proximity gate가 모두 이 perturbation을 거부해야 한다.

이는 session-progress rule의 perturbation evaluation이다. Physics는 이
measurement defect를 판별하지 못하고 unregistered physical variable을
추가하므로 금지한다.

## 4. 신규 변수와 파라미터 증가 감사

- 신규 변수 수: `1` (measurement-only).
- 신규/변경 physical variable: `0/0`.
- existing parameter increases/changes: `0/0`.
- decomposition parameter changes: `0`.
- threshold relaxations: `0`.
- collision asset writes: `0`.
- recook requests: `0`.
- controlled physics steps: `0`.
- attempt3: absent and forbidden in D342.

동결 decomposition은 `hullVertexLimit=64`, `maxConvexHulls=64`,
`voxelResolution=1,000,000`, `errorPercentage=1.0`,
`minThickness=0.0001m`, `shrinkWrap=true`다. Target/control도
`q5=1.5413rad`, `(radial,tangent)=(7,11)mm`, seed `33201`, HOME,
position-only IK, solver/object/table/mass를 그대로 둔다.

Machine audit:
`claudedocs/runtime_logs/grasp_track/g0a_d342/d342_parameter_freeze_audit.json`.

## 5. Immutable inputs와 실행 경계

- Boot HEAD: `b1476d1acc681f392eb3478da5192f3b3898085e`.
- Boot worktree: clean. D342 preregistration edits부터 intentionally
  uncommitted가 된다.
- D338 attempt1, D339 attempt2, D340, D341 outputs are immutable.
- Input inventories are pinned by path/bytes/sha256 before runtime and compared
  exact again after Rerun finalization.
- PXR/Kit은 binary USDC readback에만 사용한다. SimulationContext, task env,
  PhysX cook, simulation step은 생성/호출하지 않는다.
- Output은 새 forward-only
  `claudedocs/runtime_logs/grasp_track/g0a_d342/` 아래로만 쓴다.

## 6. Rerun 완료 계약

D342는 frame/geometry/containment 판단을 포함하므로 Rerun이 필수다.
과학 결과를 원본 배열/JSON/hash로 먼저 동결한 뒤 display copy만 Rerun에
one-way로 보낸다.

- subject: `13 parts x 3 meshes = 39`.
  - direct authored x0, USD prim-local
  - body-mapped D340 x0, body-local
  - D340 candidate x1, body-local
- named transforms: `2` body frames + `13` prim frames = `15`.
- scalars: `10 metrics x 13 + 13 gates = 143`.
- shared event rows: `16`.
- exact non-system entities: `238`.
- exact timelines: `blueprint`, `log_time`, `part_idx`.
- fixed Blueprint: link5/gripper 각 direct, mapped x0, overlay,
  x0-vs-x1 네 panel + 동시에 보이는 metrics/gates/events.
- registered screenshot: `2400x1400` logical size.

Footer, exact entity/timeline/component contract, RBL, headless screenshot를
통과한 뒤 screenshot을 실제로 열어 별도 report를 써야 completion PASS다.
Pixels는 bit equality나 `1e-16m` 차이를 증명하지 않는다.

## 7. 순차 실행 절차

1. Output folder에 preregistration/parameter audit 두 파일만 있는지 확인.
2. HEAD, session/START/source hashes, D339 attempt2/D340 inventories,
   environment pins, 13-part allowlist, raw `<f4>` baselines, Rerun exact
   contract를 확인한다.
3. Headless Kit을 열어 immutable D339 USDC의 13 part points/faces만 직접
   읽고 즉시 닫는다.
4. Direct stream exact checks를 먼저 동결한다.
5. Body-mapped containment/proximity checks와 두 negative control을 실행한다.
6. Scientific evidence JSON을 먼저 저장한다.
7. 39 display meshes와 numeric scalars/events를 footer-enabled RRD에 기록,
   finalize 후 exact validation 및 headless screenshot을 실행한다.
8. D339/D340 inventories와 attempt3 absence를 재검증한다.
9. Automated summary는 manual inspection pending으로 보존한다.
10. Screenshot을 실제 열어 별도 manual report/final summary를 작성한다.

## 8. Registered command

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab --no-capture-output python \
  sim_scripts/cyl34_top_view_d342_grasp_g0a_authored_coordinate_stream_repair.py \
  --headless
```

## 9. Pre-runtime verification

- Python compile: PASS.
- `git diff --check`: PASS.
- Rerun unit/integration suite: `7/7` PASS.
- Pins: `rerun==0.34.1`, `numpy==1.26.0`, `psutil==5.9.8`.
- Existing parameter increases/changes: `0/0`.
- No asset-authoring, recook, SimulationContext, physics-step, settle, or
  10-trial path is registered.

## 10. Result placeholder

Not run yet. Any PASS licenses only the authored-coordinate-stream proof.
Attempt3 authoring plus fresh live validation remains a separate, explicitly
approved follow-up case (recommended D343). `g0a_pass=false` throughout D342.

## 11. Invocation amendment 001 (launch-only attempt0 보존)

최초 registered command의 두 호출은 과학적 평가를 시작하지 못했다.

1. Sandbox 호출은 GPU/Vulkan 초기화에서 Kit이 중단됐다. D342 scientific
   harness, perturbation, Rerun logging은 진입하지 않았고 새 output은 0개다.
2. 동일 명령의 host 호출은 immutable D339 USDC read까지 성공했으나 기존
   harness가 그 직후 `SimulationApp.close()`를 호출했다. 이 환경에서 close가
   Python caller로 복귀하지 않아 `_evaluate()` 이전에 종료됐다. 보존된 Kit
   log는 `kit_20260713_213834.log` (380,538 bytes,
   sha256 `2a5e43d2d096d45445f555db3e8e568d8a88db59014af548e1eb39fd3e7d63cd`)다.

두 호출 뒤 D342 output folder에는 원 preregistration과 parameter audit 두
파일만 있었으며 evidence/RRD/RBL/screenshot/summary는 없었다. 따라서 유효
scientific execution count는 `0`이다. 원 preregistration은 덮어쓰지 않고
`d342_preregistration_amendment_001.json`을 forward-only로 추가한다.

Reactive 수정은 실행 순서 하나뿐이다. Kit을 연 상태에서 immutable USDC
read, exact/numeric evaluation, perturbation, evidence write, Rerun finalize와
validation, automated summary write/flush를 끝낸 뒤 마지막에 app을 닫는다.
예외 종료에서도 app close가 실행되도록 exit cleanup만 등록한다. Direct/mapped
stream 정의, 13-part subject, hash baseline, `1e-9m` tolerance, `+10um`
perturbation, Rerun entity/timeline/component 계약은 모두 동일하다.

- 추가 신규 변수: `0` (전체는 기존 1개 그대로).
- physical variable/parameter/tolerance/decomposition 변화: `0/0/0/0`.
- asset write/recook/physics/attempt3: `0/0/0/absent`.
- 이 amendment 뒤 동일 registered command의 다음 호출이 최초 유효 실행이다.

## 12. 최초 유효 실행 결과

동일 registered command를 host GPU에서 한 번 유효 실행했다. 자동 verdict는
`D342_AUTHORED_COORDINATE_STREAM_CONTRACT_FAIL_STOP`이다. 이후 재실행하지
않았고 attempt3도 만들지 않았다.

통과한 sub-evidence:

- Direct authored raw `<f4>` bytes, cold1/cold2 arrays, triangles, manifest
  vertex/topology/geometry hashes: 전부 `13/13` exact.
- Body-mapped numeric: `13/13`; assignment max
  `1.1796119636642288e-16m`, symmetric surface `0.0m`, candidate containment
  violation `0.0m`, consensus/candidate surface `0.0m`.
- Legacy transformed-stream manifest hash는 예상대로 `0/13`, numeric frame
  equivalence는 `13/13`; negative domain discriminator `13/13` PASS.
- `+10um` perturbation actual float32 delta
  `1.0000541806221008e-05m`; raw/geometry hash와 `1e-9m` proximity가 모두
  거부했다. Source write는 없었다.
- D339 `18->18`, D340 `33->33` inventories exact; parameter audit, attempt3
  absence, zero physics/recook/asset writes 모두 PASS.

유일한 실패는 direct check의 `min_thickness_frozen_1e_4m=0/13`이었다.
D339이 같은 immutable attempt2에서 기록한 실제 float readback은
`9.999999747378752e-05m`; 요청값과 차이는
`2.526212488436659e-12m`이다. D339/D340이 고정한 readback tolerance는
`1e-10m`이므로 정상 PASS지만, D342 harness가 이 한 check에 `1e-12m`을
하드코딩했다. 실제 parameter increase/change는 `0/0`이나 validator
tolerance가 사전등록과 달리 `100x` 엄격해졌다.

독립 audit에서 이 attr는 explicit authored USD `float`이고 bit pattern은
`0x38d1b717`로 확인됐다. Stage `metersPerUnit=1.0`이며 schema default
`0.001` fallback도 아니므로 unit/default 원인은 배제된다. D342 evidence가
raw typed value/type/authored-opinion/bits를 보존하지 않고 predicate boolean만
남긴 것도 다음 case에서 고칠 evidence-schema 결함이다.

따라서 전체 D342 PASS로 소급 재분류하지 않는다. Final bounded verdict는
`D342_AUTHORED_COORDINATE_STREAM_HARNESS_TOLERANCE_DRIFT_FAIL_STOP`이다.
좌표 stream 가설은 positive sub-evidence로만 보존한다.

## 13. Rerun 실제 검사

- RRD/RBL: `685,882/92,739` bytes, footer verify PASS.
- Exact non-system entities `238`; timelines exactly
  `blueprint/log_time/part_idx`; required component schemas PASS.
- Subject: frames `15`, meshes `39`, scalars `143`, events `16`.
- Headless screenshot logical `2400x1400`, actual raster `4800x2800`,
  `9,806,497` bytes.
- 이미지를 original detail로 실제 열었다. link5/gripper 각 direct, mapped,
  overlay, x0-vs-x1의 여덟 spatial panel이 모두 non-empty였고 metrics, gate,
  events도 보였다. Event table의 `direct=False`, `numeric=True`,
  `legacy_mixed_hash_rejected=True`가 FAIL 귀속과 일치했다.
- Pixels는 13개 exact enumeration이나 bit equality 권위로 쓰지 않았다.
- 경미한 non-decision anomaly도 기록했다. Viewer toast가 우상단 panel 일부를
  가리고 metrics가 가로로 잘렸으며, stats는 event `part_idx`를 UNSORTED로
  표시했다. 또한 automated summary의 manual-pending false와 log-status의
  inspection-incomplete가 의미상 충돌한다. Footer/entity/component/render와
  FAIL 귀속에는 영향이 없다.

Rerun observability 자체 verdict는 `D342_RERUN_OBSERVABILITY_INSPECTION_PASS`다.
이는 scientific FAIL을 뒤집지 않는다.

## 14. 실행 및 다음 경계

Kit app close 뒤 shell exit code가 `0`으로 관측됐지만, close 전에 flush된
summary와 stdout은 scientific FAIL을 명시한다. `SimulationApp.close()`가
Python의 계산된 failure return code `2` 전에 process를 종료한 것이므로 shell
status는 이번 verdict의 권위가 아니다. 다음 harness는 verdict를 먼저
영속화하고 Kit shutdown 뒤에도 nonzero failure status를 보존해야 한다.

D342는 여기서 정지한다. 다음 권장 D343은 별도 승인된 proof-only USD
typed-float/readback-contract repair다. 신규 변수는
`[usd_float_parameter_readback_contract]` 1개로 제한하고 raw value/type/
authored opinion/f32 bits를 보존한다. `np.float32(0.0001)` exact bits 또는
이미 동결된 `1e-10m`을 single source로 사용하고 adjacent-float32 negative를
거부해야 한다. 순수 schema/hash audit이므로 새 Rerun 생략 사유를 명시할 수
있다. D343 통과 전 attempt3는 금지하며, 그 뒤에도 collision-asset authoring은
별도 D344 승인 경계다. `g0a_pass=false`; G0b/RL/ladder blocked.
