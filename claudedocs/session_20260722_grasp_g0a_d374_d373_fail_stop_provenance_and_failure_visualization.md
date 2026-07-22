# D374 D373 fail-stop 계보 및 16+18 충돌체 시각화

Date: 2026-07-22 KST  
Case: `g0a_d374`  
Attempt: `attempt1_d373_fail_stop_provenance_and_failure_visualization`  
Final verdict: `D374_D373_FAIL_STOP_PROVENANCE_AND_FAILURE_VISUALIZATION_PASS`  
Preserved upstream verdict: `D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP`  
`g0a_pass=false`

이번 case의 신규 변수:

1. `d373_fail_stop_provenance_contract_v1`
2. `d373_failure_and_p34_visualization_projection_v1`

## 1. 무엇을 왜 했는가

D373은 frozen P34를 새 USD에 올리고 PhysX callback을 받는 데는 성공했지만, 전체 live
identity 인증 전에 네 계약 오류로 fail-stop했다. 정상 D373 분석 단계가 실행되지 않아
실패를 설명하는 1920×1080 그림과 34개 실제 callback collider 시각자료도 없었다.

D374의 목적은 두 가지뿐이었다.

1. immutable D373 원본을 읽어 Float32, instance-proxy property query, traversal,
   worker-supervisor 권위 단절의 정확한 계보를 확정한다.
2. D373에서 PhysX가 실제 반환한 instance callback polygon을 link5 16개와
   gripper_link 18개로 분리해 사람이 볼 수 있게 만든다.

D374는 P34를 고치거나 다시 cook하는 case가 아니다. Isaac/PhysX 재실행, USD write,
physics step, q5, contact, cylinder, target/IK/path, collider regeneration은 모두 0이다.

## 2. 부팅 및 Git 교차검사

- 실행 전 `HEAD == origin/master ==
  548d3517f5a7936529646c5d8b0009427eb936ab`를 확인했다.
- commit subject는 `D373_P34_LIVE_ASSET_IDENTITY_FAIL_STOP`이다.
- 사용자가 D373을 push한 뒤 시작했으므로 D374 전 baseline worktree는 clean이었다.
- D374는 commit/push하지 않았다.
- PNG 5개는 exact path에 존재하지만 `.gitignore:110`의 `*.png` 규칙 때문에 일반
  `git status`에는 나오지 않는다. 후속 push에 포함하려면 사용자 승인 후 force-add가 필요하다.
- 사용자 소유
  `claudedocs/lab_meeting/20260715/d334_collision_table/`의 전체 inventory를 전후 비교했고
  bit-exact였다.

## 3. 사전등록

사전등록은 다음을 모두 PASS했다.

- Git HEAD/origin exact
- D373 핵심 파일 7개 exact SHA-256
- D343 typed-Float32 계약 4개 exact SHA-256
- D373 callback witness JSON 정확히 68개
- `numpy==1.26.0`, `psutil==5.9.8`, `rerun-sdk==0.34.1`
- Noto CJK fonts와 absolute Rerun CLI
- Isaac/PhysX/pxr/Warp 계열 모듈 미로딩
- 신규 변수 2개

Failure-capable offline 음성대조군은 `4/4` PASS했다.

1. return code만 보면 잘못 PASS하는 대조군을 거부했다.
2. default traversal의 P34 0개를 asset 부재로 승격하지 않았다.
3. `ERROR_PARSING` row의 0 kg/0 m³ sentinel을 측정값으로 승격하지 않았다.
4. callback protocol PASS만으로 full identity를 승격하지 않았다.

Source:

- `claudedocs/runtime_logs/grasp_track/g0a_d374/attempt1_d373_fail_stop_provenance_and_failure_visualization/d374_preregistration.json`

## 4. 실행 순서

1. D373 전체 파일 inventory와 D343/D373 핵심 해시를 재확인했다.
2. D343 typed-Float32 계약은 읽기만 했고, scalar readback/packing/ULP 재시험은 0회였다.
3. D373 raw summary의 34 callback row와 68 witness JSON을 body/prim/channel key로
   bijection했다.
4. 각 part의 instance/prototype callback이 각 1회, `RESULT_VALID`, convex 1개,
   serialization error 0인지 확인했다.
5. instance와 prototype의 원 polygon payload가 part별 exact인지 확인했다.
6. instance 원 polygon을 화면 표시용 fan triangle로 변환했다. 이 triangle은 시각화에만
   사용했고 원 callback JSON/hash를 과학 권위로 유지했다.
7. 실패 원인판, 전체 조립/분리판, link5 16개 세부판, gripper_link 18개 세부판을
   각각 정확히 1920×1080으로 만들었다.
8. 동일 34개를 save-only RRD에 기록하고 fixed RBL, footer/entity/timeline/component,
   headless screenshot을 검증했다.
9. 다섯 PNG를 원본 해상도로 직접 열어 글자 겹침, 누락, 카메라와 범위 라벨을 검사했다.
10. D373과 D334 sidecar의 전후 inventory가 exact인지 다시 확인하고 finalize했다.

Actual offline audit/retry는 `1/0`, headless Viewer capture는 `1`회였다.

## 5. 네 실패 원인의 확정 결과

### 5.1 Float32 거짓 실패

- D343 frozen authority는 typed value
  `0.00009999999747378752m`, bits `0x38d1b717`, little-endian bytes
  `17b7d138`이다.
- D373 direct 34개와 live 34개의 numeric 값은 모두 그 D343 typed value와 같았다.
- D373에서 각 row의 유일한 false check는 `min_thickness_frozen`이었다.
- decimal `0.0001m`과의 차이 `2.526212488436659e-12m`를 `1e-12m` comparator로
  검사한 것이 거짓 실패 원인이다.
- D374는 D343 결과를 상속했다. D373 raw에는 `typeName`이나 raw bits가 직접 저장되지
  않았으므로 “D373이 bits를 직접 34/34 관측했다”고 승격하지 않는다.
- D374 typed scalar 재시험 및 Float packing/ULP 재계산은 `0/0`이다.

### 5.2 whole-robot instance proxy property query

- D373 live P34 row 34개는 모두 instance proxy였다.
- D373 stdout 45-51행과 55-56행에서 설치 PhysX는
  `RigidBodyAPI on an instance proxy not supported`를 기록했다.
- link5/gripper_link 두 query는 모두 `ERROR_PARSING(5)`였다.
- elapsed는 각각 `0.004140967968851328s`와 `0.002627589041367173s`로 timeout이 아니다.
- error row의 empty path, path ID 0, mass 0, volume 0은 sentinel이며 실제 property
  측정값이 아니다.

후속 live repair 계약은 dynamic articulation rigid-body owner를 non-instance로 유지하고,
collider geometry leaf/prototype instancing은 설치 runtime이 허용하는 범위에서만 별도로
다루도록 고정했다.

### 5.3 traversal 사각지대

- D345 default stage traversal이 본 variant P34 Mesh는 `0`개였다.
- proxy-aware D373 live inventory는 active P34 path `34`개였다.
- callback part row도 `34`개였다.
- 따라서 default traversal의 0은 asset absence가 아니라 scope blind spot이다.
- 후속 계약은 proxy-aware traversal과 direct authored-layer audit를 함께 요구한다.

### 5.4 worker-supervisor 권위

- supervisor process return은 `0`, old `pass=true`였다.
- raw summary와 hash-bound preclose sentinel은 모두 `worker_protocol_pass=false`였다.
- preclose summary SHA는 raw summary SHA와 exact였다.
- 수리된 effective-pass 식을 D373 원자료에 적용하면 `false`다.
- 후속 계약은 `returncode==0`, operational supervisor PASS, raw worker PASS, preclose
  worker PASS, preclose↔raw hash match가 모두 true일 때만 PASS하도록 고정했다.

## 6. 실제 callback 충돌체 16+18

D373의 instance callback 원 polygon 기준 결과는 다음과 같다.

| body | 역할 | 개수 |
|---|---|---:|
| link5 | 몸통(`structural_body`) | 1 |
| link5 | 연결/회전축 지지(`connector_support`) | 3 |
| link5 | 고정 턱 접촉 조각(`fixed_jaw`) | 10 |
| link5 | 고정 턱 뒷지지(`fixed_jaw_backbone`) | 2 |
| gripper_link | 움직이는 연결 지지(`moving_support`) | 4 |
| gripper_link | 움직이는 턱 접촉 조각(`moving_jaw`) | 12 |
| gripper_link | 움직이는 턱 뒷지지(`moving_jaw_backbone`) | 2 |
| 합계 |  | 34 |

- witness protocol: `68/68` PASS
- raw witness SHA: `68/68` exact
- instance↔prototype 원 polygon payload: `34/34` exact
- instance 34개 합계: vertices `314`, original polygons `262`
- part당 최대: vertices `13`, original polygons `17`
- 표시용 triangles 합계: `492`

그림은 link5와 gripper_link를 각각 자기 owner-local 좌표에 둔다. D373은 q5/world pose를
측정하지 않았으므로 두 body를 한 world pose로 합치지 않았다. 분리 배열은 각 조각을 쉽게
보려고 옮긴 표시 배치이며 실제 물리 자세가 아니다.

## 7. 시각화 결과

정확한 1920×1080 PNG 네 장을 원본 해상도에서 검사했다.

1. `d374_failure_provenance_1920x1080.png`: 네 실패 카드와 null/g0a footer가 겹치거나
   잘리지 않았다.
2. `d374_p34_assembled_and_exploded_1920x1080.png`: 두 owner-local 조립 형상과 두
   display-only 분리 배열이 모두 보였다.
3. `d374_link5_16_colliders_1920x1080.png`: p000-p015, 4×4 정확히 16칸이 보였다.
4. `d374_gripper_link_18_colliders_1920x1080.png`: p000-p017, 3×6 정확히 18칸이 보였다.

Rerun strict validation은 PASS했다.

- RRD bytes/SHA: `548234` /
  `f3003403fc9deed661558feea48c932c6ada06c5bc674cac8db8e47e92af2cf3`
- RBL bytes/SHA: `92791` /
  `eeff7a85715efba43fddbe02c69bf812820ca82f18d587706b1fb3043a255c40`
- fixed blueprint: link5/gripper assembled 2개, exploded 2개, D374 event log
- requested logical window는 1920×1080이지만 desktop DPR 2 때문에 screenshot 파일은
  3840×2160이다. exact 1920×1080 발표 권위는 위 네 PNG다.
- startup notification은 upper-right background 일부를 덮지만 gripper decision subject를
  가리지 않았다.

Rerun geometry는 Float32 inspection copy이고 callback JSON/hash를 다시 판정하는 권위가 아니다.

## 8. 범위 카운터

| 항목 | 값 |
|---|---:|
| offline audit invocation | 1 |
| automatic retry | 0 |
| D343 typed-Float32 retest | 0 |
| Isaac launch / PhysX call / USD write | 0 / 0 / 0 |
| physics step / q5 command / q5 sample | 0 / 0 / 0 |
| contact / cylinder create-write | 0 / 0 |
| target-IK-path change | 0 |
| collider regeneration / decomposition sweep | 0 / 0 |
| headless Rerun Viewer capture | 1 |

## 9. 판정과 해석 경계

Final verdict:

`D374_D373_FAIL_STOP_PROVENANCE_AND_FAILURE_VISUALIZATION_PASS`

이 PASS의 쉬운 뜻은 다음과 같다.

- D373이 왜 멈췄는지를 원본 증거로 일관되게 설명했다.
- 당시 PhysX가 실제 돌려준 P34 조각 34개를 누락 없이 사람이 볼 수 있게 만들었다.
- 후속 live worker가 지켜야 할 traversal/property/supervisor 계약을 machine-readable하게
  고정했다.

이 PASS는 P34 live identity나 물리 성능 PASS가 아니다. 다음은 계속 `null`이다.

- full P34 live identity
- authored↔callback surface/bounds/original-polygon topology-volume
- live property mass/COM/inertia/axes
- physics equivalence and runtime speed
- tipping causality
- grasp feasibility

D373 verdict는 그대로 FAIL_STOP이고 `g0a_pass=false`다.

## 10. 다음 승인 경계

다음 최소 후보는 아직 미승인인 별도 forward-only live repair case다.

`D375 [p34_live_asset_identity_contract_repair]`

이 후보가 승인된다면 D343 typed-Float32와 D374 repair contract를 상속하고, dynamic
articulation owner를 non-instance로 유지하며, proxy-aware/direct authored traversal과
hash-bound supervisor authority로 P34 live identity만 one-worker/no-retry로 다시 검증해야 한다.

그 live identity가 PASS하기 전에는 A64, link5-only P34, gripper-only P34, both-P34의 실제
원통 물리 비교로 넘어가지 않는다. 물리 비교와 target/IK/path/중앙 높이/손목 pose 변경은
각각 다시 별도 승인을 받아야 한다.

## 11. NVIDIA 공식 근거와 로컬 적용 증거

D374는 새 NVIDIA 의미를 도입하지 않고 D373의 version-matched 근거를 상속했다.

- Omni Physics 107.3, **Rigid Bodies**:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html
  - 적용 버전: installed `omni.physx 107.3.26`
  - 적용 의미: articulation link를 scenegraph-instance로 두지 않는다.
- 로컬 runtime 증거:
  `claudedocs/runtime_logs/grasp_track/g0a_d373/attempt1_p34_live_asset_identity_preflight/d373_worker_stdout.log:45-56`
- 로컬 property 결과:
  `claudedocs/runtime_logs/grasp_track/g0a_d373/attempt1_p34_live_asset_identity_preflight/d373_worker_raw_summary.json`
- 설치 schema pin:
  D373 preregistration의 `nvidia_contract.schema_sha256 =
  fe075bce4bde5ba7db69c6ccef0c4c26909336ab34c619129fc276f7cb4d7abc`

## 12. 핵심 산출물 및 SHA-256

- preregistration:
  `8a62eb6ba73e2d8f6325fc1df2f457de66dc3880e3f2bcdd95a0d12e20e09e78`
- failure provenance evidence:
  `2de72cb64033ffa9bf71a42b1c7cb1b1edf340635894b5d0bea05f54f2120ced`
- live repair contract:
  `09d95e78f4bf7ec617a2dc330c83dcb96a9cbc26512679a569cf3ef6e7a5ce88`
- automated summary:
  `0aa48a202854a9c33fae0a037da8c368515a2b8fc2f84c05ddee60d904169d97`
- manual inspection:
  `5c047981c82e5478369d6d8c357a7f714a579b263fd0c2e8441414b34f943590`
- completion summary:
  `0540cf4183b75d89fb649fe596212a881bfdee0cbd739bbce5e9fec932148d5b`

Primary source directory:

`claudedocs/runtime_logs/grasp_track/g0a_d374/attempt1_d373_fail_stop_provenance_and_failure_visualization/`
