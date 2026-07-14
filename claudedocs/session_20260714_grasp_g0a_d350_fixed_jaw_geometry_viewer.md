# D350 — 동결 자세의 실제 고정-jaw 기하와 Isaac Viewer collider 시각화

날짜: 2026-07-14 KST  
상태: 사용자 승인 후 사전등록 / 실행 전  
이번 case의 신규 변수:
`[fixed_jaw_semantic_surface_binding, frozen_target_fixed_jaw_centerline_measurement]`  
신규 물리 변수: `[]`

## 1. 무엇을 왜 확인하는가

D349는 동결된 열린 자세에서 raw mesh와 D348 callback-topology live surface가
원통과 떨어져 있고 서로 합의함을 물리 step 전에 증명했다. 그러나 기존 정렬 판정의
고정-jaw 위치는 `TCP - link5 local x * 8mm`라는 한 점 proxy였고, 실제 고정 손가락
표면의 중심축·높이·법선은 측정하지 않았다. 또한 D347/D349 Rerun은 `link5`와
`gripper_link`를 분리된 여러 패널로 보여 주므로 조립된 실제 자세와 64+64 분해를
직관적으로 확인하기 어렵다.

D350의 질문은 다음 두 개뿐이다.

1. D349 raw 최단점이 속한 실제 `link5` 연결 표면을 결정적으로 다시 찾아 고정-jaw
   기하로 결합할 수 있는가?
2. D349의 exact Float32 자세를 바꾸지 않은 상태에서 그 표면의 중심축·높이·법선·
   원통 간격을 측정하고, 실제 Isaac Viewer 한 조립 화면에서 64+64 live collider와
   함께 확인할 수 있는가?

이 결과가 현재 target의 기하 불일치를 보여도 D350 안에서 target, IK 또는 경로를
고치지 않는다. 그것은 별도 사용자 승인 case의 입력이다.

## 2. 동결 입력

- Base Git HEAD: `647dfe6ba8e13c781b39850bf7228010fd1683b4` (`D349완료`)
- 출력 경로: `claudedocs/runtime_logs/grasp_track/g0a_d350/`
- D344 attempt3 robot USD SHA-256:
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`
- D344 attempt3 physics USD SHA-256:
  `043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503`
- D348 callback-topology evidence SHA-256:
  `83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6`
- D349 corrected audit SHA-256:
  `7e3d79f36e54fec4940bc58ecb81d4d13113329129b9d0926e0c65436cb5c079`
- D349 runtime binding SHA-256:
  `9bc8d1c95f3c235816eb1c3c11516f3f27416e45b302cf8b6f9d5ee01ad6ec05`
- D349 measurement SHA-256:
  `5de6d14e37d6b74b202d1bb668120a6bb57221eac24ea5c751457ce9823b6300`
- D349 completion SHA-256:
  `6ec883c4ebf4dd25aa2795006699b1d09e3b554412e2dcfa86277de541bd677e`
- target contract: `(radial,tangent)=(7,11)mm`, sign `-1`, seed `33201`,
  HOME-seeded position-only IK, q5 OPEN `1.5413rad`
- exact target joint Float32 radians:
  `[0.03750238195061684, 0.542945146560669, 1.9687392711639404,
  0.18299327790737152, 0.0, 1.5413000583648682]`
- exact object Float32 position:
  `[0.30000001192092896, 0.0, 0.03288299962878227]`
- exact object quaternion: `[1.0, 0.0, 0.0, 0.0]`
- live decomposition: `link5 64 + gripper_link 64`
- 기존 안전/일치 gate: clear `>=0.1mm`, raw/live delta `<=0.5mm`

## 3. 실제 고정-jaw 의미 결합 규칙

`gripper_left_link.stl`의 장착 가설이나 기존 8mm proxy를 actual surface로 사용하지
않는다. 다음 순서를 결과를 보기 전에 고정한다.

1. D349 measurement의 authoritative raw `link5` 최단점 world 좌표를 seed로 쓴다.
2. exact target의 runtime `link5` pose로 seed를 body-local 좌표로 바꾼다.
3. retained raw `link5` 삼각형 표면에서 seed와 가장 가까운 triangle을 찾는다.
4. raw mesh의 동일 좌표 vertex를 exact하게 weld한 뒤 edge/vertex 공유 triangle의
   연결 성분을 만든다.
5. 최소거리 triangle들이 하나의 연결 성분에만 속하고, seed-to-surface residual이
   `<=0.01mm`이며, 두 번의 독립 결합 digest가 exact해야 한다.
6. 선택 성분의 rigid-body owner는 `link5`여야 한다. q5 child인 `gripper_link`를
   고정-jaw로 넣는 음성 대조군은 owner/joint-parent 계약에서 거부돼야 한다.

하나라도 실패하면 `D350_FIXED_JAW_SEMANTIC_BINDING_FAIL_STOP`이다.

## 4. 사전등록 측정값과 판정 범위

Float64 원본에서 다음을 기록한다.

- base `B`, cylinder center `C`, 수평 radial/tangent
- 실제 고정-jaw 연결 성분의 unique vertex/triangle 수와 bounds/hash
- 연결 성분 unique vertex PCA의 가장 긴 축(실제 중심축)과 link5 `+z`의 관계
- 중심축의 수평 투영과 base→cylinder radial 사이 각도
- 중심축이 cylinder radial station을 지날 때의 world 높이와 `C.z` 차이
- D349 actual raw surface witness와 cylinder witness, 실제 간격
- seed triangle의 actual surface normal과 surface→cylinder-center 방향 각도
- 기존 8mm proxy와 actual raw witness의 3D 차이
- q5/target/object Float32 bits, raw/live 거리, 64+64 binding, simulation counter

중심 높이·중심축 각도의 새 성공 허용값은 D350에서 만들지 않는다. 따라서 결과는
`MEASURED` 또는 `FAIL_STOP`으로만 부르고 `ALIGNED_PASS`라고 부르지 않는다. 기존
D323의 폐기된 3deg strict-axis gate를 재사용하지 않는다.

## 5. Viewer와 collider 표시 계약

- 실제 launcher: `headless=False`, `livestream=0`, `xr=False`, `cuda:0`
- timeline은 계속 정지하며 `simulation_app.update()` UI/render pump만 허용한다.
- `sim.step`, `env.step`, `world.step`, `timeline.play`, `dt>0 update`는 금지한다.
- D348 callback topology를 world-space display-only mesh로 복사한다.
- display copy에는 CollisionAPI, rigid-body API, physics material을 적용하지 않는다.
- 원본 collider/material/asset은 수정하거나 저장하지 않는다.
- 한 조립 화면에 전체 robot/tool/cylinder를 두고 다음을 함께 표시한다.
  - `link5` 64개: 파랑 계열 part별 색
  - `gripper_link` 64개: 주황 계열 part별 색
  - actual fixed-jaw raw 연결 성분: 별도 강조색
  - target cylinder collider, base→center radial/tangent, 실제 중심축,
    actual witness/normal, 기존 proxy, TCP/link5/object frame
- actual Isaac viewport PNG: PhysX whole-oblique + PhysX tool-oblique 근접 화면,
  colored whole-oblique + tool-top + tool-side + tool-oblique (총 6장)
- 초기 interactive Viewer는 tool-oblique 카메라로 유지한다.
- Viewer capture 전후 counter `0→0`, target/object bits exact, source hash exact가
  아니면 STOP한다.

Rerun은 사용자 주 화면이 아니라 의무 replay/관찰 증거다. 결합된 128개 part와 실제
고정-jaw 기하를 기록하고, SDK `0.34.1`, finalized RRD/RBL footer, exact entity /
timeline/component 계약, embedded blueprint/export, headless screenshot, 원본 해상도
육안 검사를 모두 통과해야 한다. Rerun Float32 display copy는 수치 권위가 아니다.

## 6. 불변·금지 경계

- 자산 write `0`, decomposition 변경 `0`, fresh cook/callback/property query `0`
- target/허용값/재질/질량/구동기/물리 설정 변경 `0`
- controlled physics step `0`
- settle/10-trial/G0b/RL/PPO/ladder `0`
- `g0a_pass=false`
- D347-D349 산출물 수정·덮어쓰기·silent rerun 금지
- commit/push 금지

이번 session에서 RL update나 perturbation evaluation을 실행하지 않는 이유는 사용자
승인 질문이 zero-step 정적 의미 결합/기하 측정이고, 물리 perturbation은 측정 자세를
오염시키며 승인된 신규 변수 범위를 넘기기 때문이다. 대신 actual surface 결합과
동결 target 재측정은 실패 가능한 판정이며, 기존 proxy 해석을 유지하거나 반증해 다음
결정을 바꿀 수 있다.

## 7. 등록 실행 순서

1. `prepare`: Git/input hash/parameter/output/Rerun 계약을 기록하고 PASS 확인
2. `validate`: fresh GUI Isaac process에서 exact-write, 기하 측정, overlay, viewport
   capture, RRD 생성 후 interactive Viewer 유지
3. 실제 Viewer와 여섯 Viewer PNG/Rerun PNG(총 7장)를 원본 해상도로 육안 검사
4. `finalize`: 모든 과학/시각/immutability 계약을 결합해 최종 판정

실행 결과는 아직 없다.

## 8. 실행 결과 — attempt1 과학 측정 완료, 관찰성 실패 보존

사용자 승인 뒤 등록 순서대로 `prepare`와 fresh GUI `validate`를 실행했다. 실제
launcher는 `headless=False`, `livestream=0`, `xr=false`, `cuda:0`였고 Isaac Viewer를
`180.0072537449887s` 동안 `8,222`회 UI/render update로 유지했다. 이는 Rerun 캡처를
Viewer라고 부른 것이 아니라 실제 Isaac GUI 실행이다. timeline은 정지 상태였으며
global simulation counter는 `0->0`, controlled physics step은 `0`, joint/object state
최대 변화는 `0`이었다. frozen target과 object Float32 bits도 Viewer 종료 뒤까지
exact였다.

### 8.1 실제 fixed-jaw 결합과 수치

- D349의 authoritative raw `link5` witness를 포함하는 connected component가 하나로
  결정됐고 semantic binding은 PASS했다.
- 선택 성분은 `7,250` faces / `3,519` unique vertices이며 digest는
  `8f64ddb03308521ce905d0714def9b72e1e69871d2f9f13ea3bd2a3f07559a4d`다.
- q5 joint child인 `gripper_link`를 fixed jaw로 부르는 음성 대조군은 owner/parent
  계약에 따라 거부됐다.
- fixed-jaw principal axis의 수평 투영과 cylinder radial 사이 각도는
  `1.1234550133840087deg`, link5 `+z`와의 각도는 `3.158094445844178deg`, world pitch는
  `-67.5238792369183deg`였다.
- cylinder radial station에서 중심선의 cylinder-center 상대 오차는 tangent
  `-24.055688140880633mm`, height `-20.360856323322935mm`, radial residual
  `-0.4717441272236722mm`였다.
- actual nearest witness height error는 `+15.894261043514184mm`; legacy
  `TCP - link5 local x * 8mm` proxy와 actual witness의 3D 차이는
  `17.027401111623742mm`였다.
- surface normal은 actual gap 방향과 `0.020327446324421534deg`였지만
  surface-to-cylinder-center 방향과는 `36.765887128711086deg`였다.
- D349 raw/live 값은 link5 `4.2726455336106985/4.272736580324082mm`,
  gripper `11.175088374613944/11.340262326338637mm`로 exact 재현됐다.

측정 계약은 PASS했지만 중심선/높이 성공 허용값을 등록하지 않았으므로 결과 어휘는
`D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED`이고 `aligned_pass=null`이다. 축 투영이 radial에
가깝다는 값 하나를 실제 중심선 정렬 PASS로 확대하지 않는다.

권위 파일:

- `claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_semantic_binding.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_geometry_measurement.json`
- measurement SHA-256:
  `4fe91e4cd37f5b0f064c7e9c91480881973ca51e651132af2c8feb57750e8446`

### 8.2 실제 Viewer 및 64+64 표시

Isaac viewport에서 actual PhysX collider whole/tool oblique 두 장과 display-only colored
`link5 64 + gripper_link 64` whole-oblique/tool-top/tool-side/tool-oblique 네 장을 캡처했다.
모두 process close 뒤 `1280x720 RGBA`, CRC/decode/load PASS, nonzero, distinct SHA로
재검증됐다. 64+64 exact 주장은 화면 눈대중이 아니라 runtime binding 및 archive
entity/component 계약에서 나온다.

### 8.3 attempt1 관찰성 FAIL과 별도 repair 경계

attempt1의 scientific binding/measurement는 성공했지만 automated aggregate는 다음
세 구현 결함으로 FAIL했다.

1. viewport capture token은 6/6 성공했으나 비동기 sink가 끝나기 전에 세 PNG를 stat했다.
2. static Rerun Mesh3D는 사전등록된 `part_idx` timeline row를 만들지 않았다.
3. 성공 상태인 `asset_write=false`를 그대로 `all()`에 넣어 immutability aggregate가
   false가 됐다.

원 `d350_automated_summary.json`의 FAIL과 원 RRD/RBL은 수정하지 않았다. 이 실제 실패에
반응한 no-Isaac/no-physics attempt2는 별도 session과 하위 출력 경로에서만 관찰성을
수리했다. 최종 통합 판정은 attempt2 completion에 기록하며, D350 전체에서도
`g0a_pass=false`, `settle_authorized=false`, settle/10-trial/G0b/RL/PPO/ladder 미실행을
유지한다.

## 9. 최종 연결 판정

Reactive repair와 일곱 이미지 원본 해상도 육안 검사까지 완료한 최종 판정은
`D350_FROZEN_FIXED_JAW_GEOMETRY_MEASURED_AND_VIEWER_SUPPORTED`이며
`completion_pass=true`다. 이는 실제 fixed-jaw를 측정하고 실제 Viewer/64+64 관찰 계약을
완료했다는 뜻이지 alignment 또는 grasp PASS가 아니다. 최종 권위:

- `claudedocs/session_20260714_grasp_g0a_d350_observability_repair.md`
- `claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/d350_completion_summary.json`
- completion SHA-256:
  `7866886a49ecfca1c16bd1283c89e920613a4c25581dadf5ebaa195e1303cedb`

다음 후보는 actual connected jaw surface를 권위로 삼아 cylinder centerline placement를
새로 설계하는 별도 target/IK geometry-repair case다. target/IK/path 변경과 물리 step은
D350 범위 밖이며 사용자 별도 승인 전에는 시작하지 않는다.
