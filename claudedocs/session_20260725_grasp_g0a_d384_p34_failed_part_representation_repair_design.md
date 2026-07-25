# Session — 2026-07-25 — Grasp G0a D384 P34 failed-part representation repair design

## 1. 무엇을 왜 확인했는가

사용자 승인 case:

`D384 [p34_failed_part_representation_repair_design]`

이번 case의 신규 변수:

1. `failed_profile_prism_authored_subpartition_v1`
2. `failed_source_hull_authored_recursive_partition_v1`

D379에서 수동 설계 P34 충돌체 `34`개 중 `17`개만 원 작성 형상과 PhysX
cook 뒤 형상이 허용 오차 안에서 같았다. D380은 실패한 나머지 `17`개가
바깥으로 커진 것이 아니라, 원 작성 꼭짓점 일부가 사라지며 안쪽으로 깎인
형상이라고 판정했다.

D384의 질문은 다음 하나였다.

> 실패한 17개만 원 작성 형상으로 정확히 다시 나누면, 현재 A64 비교 기준
> `128`개보다 적은 충돌체로 형상을 보존할 수 있는가?

이 질문에 답하기 전에는 새 USD나 실제 PhysX 충돌체를 만들 이유가 없다. 따라서
D384는 immutable D379/D380 JSON/CSV만 읽는 오프라인 설계 case로 제한했다.

## 2. 승인 범위와 동결값

허용:

- D379/D380 동결 JSON/CSV 읽기
- 실패한 17개 분류
- 원 작성 꼭짓점과 면만 사용하는 정확 분할 설계
- 정적 `1920x1080` 비교판
- save-only RRD/RBL과 Viewer 육안검사

금지하고 실제 `0`으로 유지:

- Isaac/Kit/PhysX 실행
- asset/USD 읽기 또는 쓰기
- collider materialization/regeneration
- 자동 convex-decomposition sweep
- 새 원통 생성 또는 변경
- controlled physics step
- q5 sample
- contact query
- target/IK/path 변경

동결 게이트:

- surface tolerance `0.1mm`
- authored-topology volume relative tolerance `0.5%`
- D372 void/source-coverage/contact-seed 의미론
- 통과한 기존 P34 부품 `17`개
- 비교용 총개수 기준 `<128`

`128`은 NVIDIA 한계나 최적값이 아니다. 현재 A64가 link5 `64` +
gripper_link `64`이므로 만든 프로젝트 비교 기준이다.

## 3. 부팅 및 Git 교차검사

- `HEAD`: `b880bc8f28c269f56f05a757dc725619d88c77b1`
- `origin/master`: `b880bc8f28c269f56f05a757dc725619d88c77b1`
- subject: `모델 change(grap당하는 원기둥)`
- 승인 전부터 존재한 D382/D383 미커밋 변경을 dirty baseline으로 등록하고 보존했다.
- commit/push는 실행하지 않았다.

중요: 위 commit subject만으로 새 `29x50mm` 원통이 만들어졌다고 판정하지 않았다.
아래 12절에서 실제 파일과 런타임 증거를 따로 감사했다.

## 4. 동결 입력

| 입력 | SHA-256 |
|---|---|
| D379 full live-identity evidence | `8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5` |
| D380 failed-part metrics CSV | `885806a2164c0703d8ecf2594ff19afacd86a11fdb648bb593415e6281ec1d9c` |
| D380 provenance evidence | `4c64d08e117501dd15a5836ce56ef8b963d188044beac465e645e53a17710bd1` |

사전등록은 `11/11` 조건을 만족했다. 실험이 실패할 수 있도록 다음 8개
음성대조군을 등록했다.

- 공개 USD direct-polygon selector가 있다고 거짓 주장
- profile child 하나 삭제
- zero/sliver tetrahedron 포함
- exact-cell 상한이 count gate를 통과한다고 거짓 주장
- authored vertex 수가 64를 넘었다고 거짓 주장
- surface gate를 `0.7mm`로 완화
- volume gate를 `7%`로 완화
- cooked vertex를 새 authored geometry로 재사용

최종 음성대조군 결과는 `8/8 PASS`였다.

## 5. NVIDIA 공식 문서와 설치본 교차검사

설치본:

- Isaac Sim `5.1.0.0` — D379 inventory 상속, D384에서는 실행하지 않음
- Kit `107.3.3`
- Omni PhysX schema `107.3.26`
- Rerun SDK/CLI `0.34.1`

사용한 NVIDIA 공식 자료:

1. **Omni Physics 107.3 — Colliders**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html  
   primitive collider는 대응 형상이 정해져 있지만, mesh의 `convexHull`과
   `convexDecomposition`은 cook을 거치는 근사 표현이라는 구분에 사용했다.
   단, 같은 문서는 `SETTING_COLLISION_APPROXIMATE_CYLINDERS=true`이면 cylinder를
   convex mesh로 근사한다고 명시한다. 따라서 새 원통 case에서는 이 setting과
   실제 collision representation을 함께 읽어야 한다.

2. **PhysX SDK 5.6.1 — Geometry**  
   https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Geometry.html  
   QuickHull 결과 꼭짓점이 입력 꼭짓점의 subset이며 `planeTolerance` 안의
   꼭짓점이 생략될 수 있다는 설명과, 저수준 SDK에는 points+polygons 입력
   경로가 있다는 사실에 사용했다.

3. **PhysX SDK 5.6.1 — PxCooking.h**  
   https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/107.3-omni-and-physx-5.6.1/physx/include/cooking/PxCooking.h  
   저수준 points+polygons cook 경로의 API 형태를 확인하는 보조 공식 소스로
   사용했다.

버전 경계:

- Omni Physics 문서는 설치 계열과 같은 `107.3`이다.
- PhysX 문서는 NVIDIA가 공개한 `5.6.1` 지원 문서다. 설치된 패키지는
  `omni.physx/schema 107.3.26`이지만 내부 PhysX 정확한 semver가 별도 runtime
  field로 노출된 것은 아니므로, `5.6.1` 문서를 설치 엔진의 bit-exact
  attestation으로 표현하지 않는다.

설치 스키마 교차검사:

- `PhysxConvexHullCollisionAPI` 공개 필드는
  `hullVertexLimit`, `minThickness`뿐이다.
- points/polygons를 직접 선택하는 공개 USD selector는 찾지 못했다.
- `PhysxCookedDataAPI`는 polygon 배열이 아니라 opaque `uchar[] buffer`를
  노출한다.

로컬 근거:

- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/plugins/PhysxSchema/resources/schema.usda:852`
- 같은 파일 `:2102`
- 설치 스키마 SHA-256:
  `fe075bce4bde5ba7db69c6ccef0c4c26909336ab34c619129fc276f7cb4d7abc`

따라서 저수준 C++ 경로의 존재와 공개 USD authoring 경로의 존재를 같은 것으로
취급하지 않았다. lower-level/private bridge가 절대 불가능하다고 판정한 것도
아니다.

## 6. 실패한 17개의 형상 분류

D372 생성 규칙과 D380 실패 집합을 교차검사한 결과:

- manual profile prism `9`
  - fixed jaw `2`
  - moving jaw `7`
- source 3-D convex hull `8`
  - moving support `4`
  - moving-jaw backbone `2`
  - fixed-jaw backbone `2`

이 분류가 필요한 이유는 두 형상에 같은 분할법을 억지로 적용하면 불필요하게
부품 수가 폭증하기 때문이다.

## 7. 후보 R1 — 등록된 원 작성 형상 정확 분할

### 7.1 profile prism 9개

저장된 2-D cap 삼각형을 그대로 사용해 triangular prism으로 나눴다.

- 원 실패 부품: `9`
- exact triangular-prism children: `46`

### 7.2 source 3-D hull 8개

원 작성 꼭짓점으로 Delaunay tetrahedralization을 만든 뒤, 면을 공유하고 합친
결과가 여전히 convex이며 꼭짓점이 최대 `8`개인 이웃 cell만 결정론적으로
합쳤다.

- 원 실패 부품: `8`
- merged convex children: `205`

### 7.3 총개수

- 그대로 두는 통과 부품: `17`
- profile children: `46`
- source-hull children: `205`
- 합계: `268`

형상 보존 검사 과정은 통과했지만 `268 < 128`은 거짓이다. 따라서 교수님용
저비용 후보로 기각했다.

상태:

`REJECTED_PART_COUNT_ABOVE_A64_REFERENCE`

## 8. 후보 R2 — 정확 사면체 상한 대조군

source 3-D hull을 합치지 않고 양의 부피 사면체로 그대로 유지했다.

- profile triangular prisms: `46`
- positive tetrahedra: `495`
- zero/sliver tetrahedra rejected: `40`
- unchanged passing parts: `17`
- 합계: `558`

이 후보는 “정확히 보존하려면 얼마나 커질 수 있는가”를 보여주는 보수적
상한 대조군이다. 효율 후보가 아니다.

상태:

`REJECTED_PART_COUNT_EXPLOSION`

## 9. 후보 R0 — 원 points+polygons 직접 경로

실패한 17개 원 작성 polygon payload가 저수준 direct-polygon 입력을 위한 구조
조건을 만족하는지 오프라인으로 검사했다.

- input preconditions: `17/17 PASS`
- 이론상 총 충돌체 수: 통과 17 + 직접 입력 17 = `34`
- public USD direct-polygon selector: `false`
- live runtime capability: `null`
- live identity: `null`
- materialization-ready: `false`

공개 USD selector가 없기 때문에 C++ 확장 또는 opaque cooked-data bridge가
필요하다. 이는 “실패 부품만 최소 분할”보다 훨씬 큰 새 표현 파이프라인이므로
이번 case에서는 reserve-only로 남겼다.

상태:

`RESERVE_ONLY_NEW_BRIDGE_REQUIRED`

## 10. 실행 순서와 forward-only 실패 기록

### Attempt1 — 동결

- preregistration `11/11 PASS`
- canonical 계산 전 `KeyError: live_callback_vertex_count`
- D379 실제 field는 `vertex_count`
- 분류: harness field-name error, 과학/형상 결과 아님

### Attempt2 — canonical 계산

새 설계 변수를 추가하지 않았다. immutable D379 callback `34/34`에 대해:

- `vertex_count` 존재
- obsolete `live_callback_vertex_count` 부재
- `vertex_count == len(live_callback_vertices_m)`

를 전부 확인한 뒤 메모리 안에서만 호환 alias를 제공했다.

- worker/retry `1/0`
- return `0`
- elapsed `3.729200662812218s`
- timeout/TERM/KILL/residue 모두 false
- canonical evidence SHA-256:
  `16ed5696d7198913367806e3ee13cf17a2b3f83c0c28d139115aa1d51c40822f`

계산은 완료됐지만 첫 비교판의 제목이 겹치고 Rerun 알림이 판정문을 가려
presentation completion은 fail-closed였다.

### Attempt3 — 동결

- 동결 attempt2 증거만 읽는 presentation-only repair
- worker/retry `1/0`, return `1`
- elapsed `0.7159916770178825s`
- `numpy.bool_`가 JSON-native boolean으로 변환되지 않아 RRD 전 중단
- Isaac/PhysX/Rerun 실패가 아닌 JSON serialization harness failure

### Attempt4 — 동결

- JSON-native scalar normalization과 exact `1920x1080` board gate PASS
- worker/retry `1/0`, return `1`
- elapsed `0.816509174881503s`
- direct-script 실행에서 `ModuleNotFoundError: roarm_rl`
- Rerun Viewer 실행 `0`
- repo-local import bootstrap 이전의 Python import-context failure

### Attempt5 — 동결

- repository root를 `sys.path[0]`에 정확히 한 번 넣어 import 복구
- worker/retry `1/0`, return `0`, elapsed `2.6707289579790086s`
- Viewer/retry `1/0`, return `0`, elapsed `1.336257654009387s`
- fixed camera, right-side notification buffer, RRD archive validation PASS
- 정적 board는 정상
- Rerun 한글 title/TextDocument가 네모 glyph로 표시되어 manual FAIL

### Attempt6 — 최종 presentation PASS

형상, 수치, board, camera, layout은 바꾸지 않고 Rerun view title과
TextDocument만 같은 뜻의 ASCII English로 바꿨다.

- worker/retry `1/0`
- return `0`
- elapsed `2.321078158915043s`
- Viewer/retry `1/0`
- return `0`
- elapsed `1.0074184099212289s`
- timeout/TERM/KILL/process residue 없음
- strict RRD/RBL validation PASS
- original-resolution manual checks `9/9 PASS`
- finalization checks `15/15 PASS`

이 presentation repair는 canonical 계산을 다시 실행하거나 바꾸지 않았다.

## 11. 정량 결과와 판정

Canonical design:

- `design_audit_pass=true`
- `repair_design_pass=false`
- `admissible_low_count_candidate_found=false`
- `p34_authored_to_cooked_identity_pass=false`
- `live_identity_pass=null`
- `cylinder_29x50_rendered_or_measured=false`
- `g0a_pass=false`

Canonical verdict:

`D384_REPRESENTATION_REPAIR_DESIGN_NO_ADMISSIBLE_LOW_COUNT_CANDIDATE_FAIL_STOP`

쉬운 뜻:

- 계산 절차와 실패 검출 장치는 정상 작동했다.
- 그러나 이번에 등록한 두 정확 분할 방식은 충돌체를 너무 많이 만들었다.
- 따라서 수리 asset을 만들지 않고 멈춘 것이 올바른 결과다.
- 이것은 Isaac Sim이 고장 났다는 뜻도, 로봇이 원통을 못 잡는다는 물리 판정도
  아니다. D384에는 물리를 실행하지 않았다.

Presentation verdict:

`D384_PRESENTATION_CONTRACT_REPAIRED_PASS`

## 12. 새 실제 제품 원통 시각자료 감사

### 12.1 확인된 제품 명목값

D379 vendor-page 감사에 기록된 값:

- diameter `29mm`
- height `50mm`
- material option: zelkova 또는 walnut

아직 없는 값:

- mass
- dimension tolerance
- COM/inertia
- static/dynamic friction
- bottom flatness
- exact roundness/bevel/taper

명목 geometry를 만들 때 쓸 값은 radius/height `14.5/50mm`다. 현재 table plane
`z=-12.117481868996972mm`에 세운다고 가정하면:

- bottom z: `-12.117481868996972mm`
- center z: `+12.882518131003028mm`
- top z: `+37.88251813100303mm`

이 값은 아직 authoring/readback/render를 수행한 결과가 아니라, 명목 치수로
계산한 다음 case 후보값이다.

### 12.2 실제로 존재하는 렌더와 수치

검색과 원본 스크립트/JSON 교차검사 결과, 새 `29x50mm` 원통을 실제 Isaac
scene에 만든 이미지, 영상, USD, live readback JSON, RRD는 없다.

현재 존재하는 실제 Isaac cylinder 자료는 옛 `34x90mm` 물체다.

- D332/D362 설정:
  - radius `0.017m`
  - height `0.090m`
  - mass `0.72kg`
  - static/dynamic friction `1.5/1.2`
  - restitution `0`
- D362 actual Isaac initial image:
  `claudedocs/runtime_logs/grasp_track/g0a_d362/d362_initial_open_actual_physx_interface_primary.png`
- D362 actual Isaac contact-confirmation image:
  `claudedocs/runtime_logs/grasp_track/g0a_d362/d362_contact_confirmation_actual_physx_interface.png`
- D363 trace replay video:
  `claudedocs/runtime_logs/grasp_track/g0a_d363/d363_d362_trace_replay_exact_1920x1080.mp4`
  (`1920x1080`, `20fps`, `12.5s`, `250` frames)
- D372 frozen-open board도 old `34x90mm` evidence를 투영한 offline Matplotlib
  시각화이며 새 실제 제품의 actual Isaac render가 아니다.

D384 자체의 `cylinder_creates_or_writes=0` 및
`cylinder_29x50_rendered_or_measured=false`와도 일치한다.

### 12.3 필요한 별도 시각화 case

P34 수리 방향을 정한 뒤 별도 명시 승인을 받아 다음을 zero-step으로 만들 수
있다.

- `UsdGeom.Cylinder`/Isaac primitive radius `0.0145m`, height `0.050m`, Z axis
- authored 값과 live readback 값의 exact JSON
- `SETTING_COLLISION_APPROXIMATE_CYLINDERS` 값과 실제 collision
  representation
- table-relative bottom/center/top 숫자
- actual Isaac 전체 장면과 jaw 근접 장면
- 숫자 패널이 포함된 exact `1920x1080` 비교판
- save-only RRD/RBL과 실제 Viewer 육안검사

질량/COM/inertia/friction은 실측 전까지 `null`로 두거나, 나중에 명시적으로
가정값과 실측값을 분리해야 한다. 이 case에서는 실행하지 않았다.

## 13. 시각 산출물

최종 attempt6:

- exact `1920x1080` board  
  `d384_attempt6_presentation_board_1920x1080.png`  
  SHA-256 `308f9890b0677a65ef6422081d753649d1ea27256df75e1c52fad90dd250b007`
- presentation RRD  
  SHA-256 `dff22645aa932fb5419c9797b5c96b5e3e8947c843095d9d52ab100c3269ac8a`
- fixed blueprint RBL  
  SHA-256 `5f0f93c4dbd32b0c42949ffafeff9aca1a2ad3a5b4d5d313831136ca86199786`
- native HiDPI Viewer screenshot `3840x2160`  
  SHA-256 `2356a58f8fb1229860c4b80704c9195fd14cdc36b8bf57758ce47e98a64eea78`
- manual inspection JSON  
  SHA-256 `7432f13b9c4b39bc3c6a5a59ee0486e89401ac25344191f5217df7a9a016836a`

Viewer 명령의 logical window는 `1920x1080`이고, HiDPI desktop의 native
physical screenshot이 `3840x2160`인 것을 분리해 기록했다. exact
`1920x1080` claim은 정적 board에만 적용한다.

### 13.1 Postcompletion artifact-label 계보 감사

독립 read-only 최종 감사에서 attempt6 JSON 일부의 내부 `artifact` string이
`ATTEMPT3`를 유지한다는 사실을 발견했다. 전체 `13`개를 전수 대조했다.

- inherited `ATTEMPT3` schema labels: `10/13`
- explicit `ATTEMPT6` labels: `3/13`
- forward-only attempt6 파일명/경로: `13/13`
- explicit `attempt` field가 있는 파일의 값: attempt6
- 등록 SHA-256 재검사: `13/13 exact`

이것은 attempts3-6이 같은 presentation schema를 상속하면서 schema identifier를
run identifier처럼 보이게 만든 이름 결함이다. 다음 원칙으로 fail-closed 기록했다.

- `artifact_label_clean=false`
- `attempt6_content_or_numeric_verdict_changed=false`
- `presentation_verdict_changed=false`
- immutable attempt6 파일 수정 `0`
- 추가 Viewer/Isaac/PhysX 실행 `0`

실행 계보 권위는 `artifact` string 하나가 아니라 forward-only directory/file
path, explicit `attempt` field, SHA-256의 결합이다. 기존 파일을 고쳐 해시 사슬을
깨지 않고 별도 감사 JSON을 남겼다.

감사 verdict:

`D384_ATTEMPT6_ARTIFACT_LABEL_PROVENANCE_AUDIT_PASS_WITH_INHERITED_ATTEMPT3_SCHEMA_LABELS`

감사 파일:

`claudedocs/runtime_logs/grasp_track/g0a_d384/postcompletion_attempt6_artifact_label_provenance_audit/d384_attempt6_artifact_label_provenance_audit.json`

SHA-256:

`47061dbbd2e3995474fafee7aad1fc39caa7fe1340eca51bbc355808bccd24ce`

## 14. 다음 경계

권장하지만 아직 미승인:

`D385 [p34_source_hull_semantic_low_count_redesign]`

목적:

- 실패한 general 3-D source hull `8`개만
- 의미 영역을 보존하는 low-count primitive/convex parts로 재설계
- profile exact split `46`, 통과 part `17`, D372 void/source coverage/contact
  seed, D379/D380 gate를 동결
- total `<128`과 형상 게이트를 오프라인으로만 판정

D385가 후보를 찾지 못하면 direct-polygon C++ bridge를 자동 실행하지 않고
다시 설계 방향을 승인받아야 한다.

별도 승인 항목:

- D385
- repaired asset/USD materialization
- Isaac/PhysX live identity
- actual product `29x50mm` zero-step target visualization/rebase
- mass/COM/inertia/friction 설정
- A64/P34 physics, q5/contact, hold/lift, grasp

D384 attempts1-6은 모두 동결한다. 기존 경로를 재실행하거나 덮어쓰지 않는다.

## 15. 주요 파일

- `sim_scripts/cyl34_top_view_d384_p34_failed_part_representation_repair_design.py`
- `sim_scripts/cyl34_top_view_d384_attempt2_callback_vertex_count_field_repair.py`
- `sim_scripts/cyl34_top_view_d384_attempt3_presentation_contract_repair.py`
- `sim_scripts/cyl34_top_view_d384_attempt4_json_native_layout_serialization_repair.py`
- `sim_scripts/cyl34_top_view_d384_attempt5_project_root_import_bootstrap_repair.py`
- `sim_scripts/cyl34_top_view_d384_attempt6_rerun_ascii_glyph_compatibility_repair.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d384/attempt2_callback_vertex_count_field_preflight_repair/d384_p34_representation_repair_design_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d384/attempt6_rerun_ascii_glyph_compatibility_repair/d384_attempt6_completion_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d384/attempt6_rerun_ascii_glyph_compatibility_repair/d384_attempt6_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d384/postcompletion_attempt6_artifact_label_provenance_audit/d384_attempt6_artifact_label_provenance_audit.json`
- `START_HERE.md`
