# 2026-07-21 — D371 offline collider candidate Pareto comparison

## 1. 무엇을 왜 비교했는가

사용자는 기존 D370 표시 보정만 다시 하는 대신, 새 충돌체 후보 비교 자체를 교수님에게
보여 줄 결과물로 만들도록 D371을 승인했다. 실제 원통 물리시험은 D371 결과를 먼저 보고한
뒤 별도 승인하기로 했다.

이번 case의 신규 변수:

1. `offline_collider_candidate_family`
2. `professor_facing_candidate_comparison_capture_contract`

D371의 질문은 다음처럼 제한했다.

- 기존 `64+64`가 최적이라는 근거가 있는가?
- NVIDIA 기본 상한 `32`를 같은 원본·같은 설정에서 적용하면 형상 여유를 얼마나
  잃으면서 충돌체 수를 줄이는가?
- 교수님이 제안한 “몸통은 적게, 접촉 턱은 보존” 아이디어의 가장 단순한 시제품은
  열린 자세부터 성립하는가?

여기서 “오프라인”은 저장된 형상과 자세를 파일에서 읽어 정적 거리와 형상 차이를 계산했다는
뜻이다. 로봇 관절을 움직이거나 물리 시간을 진행한 것이 아니다.

## 2. 승인·Git·동결 경계

- 준비 시 `HEAD == origin/master ==
  4a1120b801e808071583136e78954c78ca941dc8`, subject `370 test`; worktree clean.
- 신규 출력 경로:
  `claudedocs/runtime_logs/grasp_track/g0a_d371/`
- q5 target/sample, controlled physics step, live contact query, cylinder pose write,
  SimulationContext/reset, target/IK/path, material/mass/actuator/physics setting, canonical/live
  asset write는 모두 금지했다.
- D334 사용자 sidecar 세 파일은 준비 전 SHA-256을 등록하고 종료 후 동일성을 검사했다.
- cook-only worker는 정확히 1회, 자동 retry 0회로 사전등록했다.
- 실제 물리 동등성, 속도, 전도 인과, 파지 가능성은 판정하지 않도록 등록했다.

사전등록:

- `claudedocs/runtime_logs/grasp_track/g0a_d371/d371_preregistration.json`
- SHA-256:
  `fb726a04ac63a1e4d0638f51192da3ebef8fda61df006e19782a66e3fc7c9922`
- prepare checks `11/11 PASS`

## 3. 후보 정의

| 후보 | 뜻 | link5 + moving jaw 예상/실제 개수 | 계보 |
|---|---|---:|---|
| A | 현재 수리된 64-cap 기준 후보 | `64+64=128` | D347/D348의 기존 callback topology를 읽기만 함 |
| R64 | 원본 full mesh를 현재 환경에서 새로 cap64 분해 | `64+64=128` | frozen raw mesh |
| R32 | R64와 같은 원본·공통 설정, cap만 32 | `32+32=64` | frozen raw mesh |
| C1 | 인증된 접촉 운반 part를 보존하고 나머지를 body별 단일 convex로 합친 공격적 시제품 | `5+18=23` | A contact carriers + remainder maxHulls=1 |
| C2 | C1보다 넓은 인접 guard part를 보존하고 나머지를 body별 단일 convex로 합친 시제품 | `11+24=35` | A contact/nearest guard + remainder maxHulls=1 |

중요한 해석 경계:

- `R64 ↔ R32`만 `maxConvexHulls` 하나의 효과를 격리한다.
- A와 R64는 같은 개수지만 계보가 달라서 cap 하나의 인과 비교가 아니다.
- C1/C2는 정확한 CAD 의미 분할이 아니라 “접촉부 보존 + 나머지 단일 외피”라는 빠른
  탈락용 시제품이다.
- 어느 후보도 전역 최적이라고 사전 가정하지 않았다.

공통 cook 설정은 `errorPercentage=1.0`, `hullVertexLimit=64`,
`minThickness=0.0001m`, `shrinkWrap=true`, `voxelResolution=1,000,000`이다.

## 4. NVIDIA 근거의 역할

설치된 Isaac Sim/Omni Physics schema와 UI 데이터베이스를 다시 읽었다.

- `maxConvexHulls` schema 기본값: `32`
- `hullVertexLimit` schema 기본값: `64`
- 설치 UI 범위: hull 수 `1..2048`, hull당 vertex `8..64`

이 값들의 의미를 분리했다.

- `32`는 NVIDIA의 기본 설정이지 이 RoArm 형상에 대한 최적값이 아니다.
- `64 vertices`는 GPU convex geometry 제약과 직접 연결되는 값이다.
- 따라서 D371은 32를 채택하지 않고, 동일 원본 R64/R32를 실제로 비교했다.

참조한 NVIDIA 공식 자료:

1. PhysxConvexDecompositionCollisionAPI:
   https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.1/api/classusdrt_1_1_physx_schema_physx_convex_decomposition_collision_a_p_i.html
2. Isaac Sim Physics Simulation Fundamentals:
   https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html
3. Isaac Sim Performance Optimization Handbook:
   https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html
4. PhysX 5.6.1 GPU Rigid Bodies:
   https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html

마지막 자료는 제약의 보조 근거이며 설치된 SDK와 동일 버전이라고 주장하지 않는다.

## 5. 실행 절차

1. frozen raw mesh, A callback geometry, D368 contact-carrier 집합, D349 frozen-OPEN 자세를
   hash로 고정했다.
2. A의 정적 거리를 독립 재계산해 D349 값과 정확히 일치하는지 먼저 확인했다.
3. cook-only worker를 1회 시작했다.
4. R64/R32/C1/C2의 두 body를 cold1/cold2로 각각 cook하여
   `4 candidates × 2 bodies × 2 repetitions = 16 callbacks`를 수행했다.
5. callback 원 polygon index/count/vertex와 canonical geometry를 먼저 파일에 썼다.
6. controller가 32개 callback/canonical 파일을 다시 hash하여 기록된 hash와 비교했다.
7. frozen-OPEN 정적 거리, 1mm occupancy, raw exterior-to-candidate surface distance,
   contact-carrier 보존을 계산했다.
8. 실패 가능한 대조군 4개를 실행했다: Qhull 대체 거부, candidate order 교란,
   retained carrier 삭제, 두 상자 사이 빈 공간을 단일 convex가 메우는 synthetic control.
9. 수치 evidence를 시각화 전에 먼저 원자적으로 기록했다.
10. 1920x1080 교수용 비교판 3장과 RRD/RBL을 만들고 직접 화면 검사했다.

worker 결과:

- worker/retry: `1/0`
- return code: `0`
- elapsed: `27.346966849872842s`
- cook callbacks: `16/16`
- pre-close sentinel: PASS
- process-group residue: 없음
- SIGTERM/SIGKILL: 없음

## 6. 권위 수치 결과

권위 파일:

- `claudedocs/runtime_logs/grasp_track/g0a_d371/d371_offline_collider_comparison_evidence.json`
- SHA-256:
  `e300063d37de44d895da3b96ea6ac95c0d108d217f6f74c458bab218d7bccdf5`

### 6.1 frozen-OPEN 거리와 개수

단위는 millimeter이며, 양수는 원통과 떨어져 있음, 음수는 겹침이다.

| 후보 | link5 개수 | moving 개수 | link5 거리 | moving 거리 | raw 대비 최대 변화 | 오프라인 gate |
|---|---:|---:|---:|---:|---:|---|
| A | 64 | 64 | `4.2727365803` | `11.3402623263` | `0.1651739517` | PASS |
| R64 | 64 | 64 | `4.2727365803` | `11.3402623264` | `0.1651739518` | PASS |
| R32 | 32 | 32 | `4.2727365803` | `11.4715195927` | `0.2964312181` | PASS |
| C1 | 5 | 18 | `-6.2759049387` | `9.1129776374` | `10.5485504723` | FAIL |
| C2 | 11 | 24 | `-6.0535204177` | `9.0605912703` | `10.3261659513` | FAIL |

A 재현은 D349 live-topology 값과 두 body 모두 `0.0mm` 차이로 PASS했다.

C1/C2의 link5 충돌 part는 둘 다 `collapsed_structural_remainder`다. 즉 보존한 고정 턱
part가 아니라, 나머지 몸통을 한 convex 외피로 묶은 부분이 빈 공간을 가로질러 원통까지
침범했다.

### 6.2 A 기준 1mm occupancy와 표면 오차

occupancy는 1mm 격자 근사 진단이며 callback 표면/부피의 권위값은 아니다.

| 후보 | A 밖 추가 부피(mm³ 근사) | A 영역 누락(mm³ 근사) | raw exterior→후보 최대오차(mm) |
|---|---:|---:|---:|
| A | 0 | 0 | `2.5366950019` |
| R64 | 55 | 0 | `2.5366950019` |
| R32 | 1,836 | 196 | `2.6814841507` |
| C1 | 108,618 | 1,023 | `14.1690950622` |
| C2 | 102,217 | 910 | `10.2869800470` |

### 6.3 접촉 운반 부분

A에서 D368 contact patch를 운반한 표면/part는 다음과 같이 재현됐다.

- fixed jaw: `12 faces / 4 parts`
- moving inner: `40 faces / 17 parts`
- moving outer: `36 faces / 16 parts`

C1/C2는 이 운반 part를 bit-exact로 보존했다. 그런데도 실패했다. 이는 접촉면 보존만으로
충분하지 않고, 몸통을 만드는 나머지 convex가 그리퍼의 빈 공간을 메우지 않아야 한다는
추가 조건을 보여 준다.

### 6.4 Pareto 결과

모든 목적은 작을수록 좋다고 두었다: 총 hull 수, frozen-OPEN raw 거리 변화,
A 밖 추가 점유, A 영역 누락, raw exterior-to-candidate 최대오차.

- 오프라인 적격: `A, R64, R32`
- 부적격: `C1, C2`
- 오프라인 gate 적격 후보 안에서의 비지배 후보: `A, R32`
- R64는 같은 총 개수와 거의 같은 형상 지표에서 A보다 추가 점유가 있어 A에 지배된다.
- scalar score와 global optimum은 `null`.

따라서 “32가 최적”이 아니라, **이번 후보 중 R32가 수를 줄이면서 다음 단계로 넘길 수 있는
유일한 reduced-count 후보**라는 결론만 허용한다.

## 7. 무변경·무실행 확인

권위 evidence의 scope counter:

- AppLauncher cook-only: `1`
- PhysX cook callback: `16`
- offline hppfcl static part query: `378`
- SimulationContext: `0`
- environment reset: `0`
- controlled physics step: `0`
- q5 target/sample: `0`
- live contact query: `0`
- cylinder pose write: `0`
- canonical/live asset write: `0`
- target/IK/path change: `0`
- material/mass/actuator/physics change: `0`
- hardware: `0`

D334 sidecar current SHA-256은 사전등록과 동일하다.

- `README.md`: `35e39f58...e18783`
- `d334_collision_table_academic.html`: `6d38933f...a679c`
- `d334_collision_table_academic.png`: `ddc9db27...8a183`

## 8. 교수님용 시각자료

원본 해상도로 직접 연 세 파일은 모두 `1920x1080`, 글자 겹침/잘림 없음, 실제 후보
개수와 “물리/파지 미판정” 경계를 읽을 수 있었다.

1. cap64와 cap32 비교:
   `claudedocs/runtime_logs/grasp_track/g0a_d371/d371_cap_comparison_1920x1080.png`
   (`bc4e6afc...15d3a`)
2. C1/C2의 빈 공간 연결 문제:
   `claudedocs/runtime_logs/grasp_track/g0a_d371/d371_semantic_comparison_1920x1080.png`
   (`7887650e...8791`)
3. 원 접촉면(cyan)과 보존 part(yellow) 상세:
   `claudedocs/runtime_logs/grasp_track/g0a_d371/d371_contact_detail_1920x1080.png`
   (`5f5d13aa...ffe86`)

교수님용 표:

- `claudedocs/runtime_logs/grasp_track/g0a_d371/visual_repair_attempt1/d371_professor_comparison_report_repaired.md`
- SHA-256: `1c2402d3...ce4f0`

## 9. Rerun 표시 실패를 수치 결과와 분리

원 실행은 RRD/RBL 저장을 끝냈지만 ambient PATH에서 `rerun` CLI를 찾지 못해 자동
validation 단계에서 정지했다.

- original exception SHA-256: `08545700...92ec`
- original completion verdict:
  `D371_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`
- 이 예외 전에 measurement JSON이 기록됐고
  `measurement_evidence_valid=true`가 보존됐다.
- 원 worker/cook를 다시 실행하지 않았다.

반응형 visual repair attempt1은 새 forward-only 하위 경로에서 RRD/RBL을 재생성하지 않고,
설치된 Rerun CLI의 absolute path 한 변수만 검증했다.

- RRD/RBL hash unchanged: PASS
- footer verify, exact `787` non-system entities/components,
  `blueprint/log_time` timelines, RBL: PASS
- Viewer invocation: `1`; return code: `0`
- Isaac/PhysX launch, cook, physics, q5, contact: 모두 `0`

그러나 요청 `3840x2160`이 HiDPI에서 `7680x4320`으로 캡처됐고, Rerun 패널의 한글
글리프 일부가 네모 상자로 보였다. 따라서:

- Rerun data contract: PASS
- professor boards: PASS
- Rerun presentation contract: FAIL
- repair overall: FAIL

이 attempt도 retry하지 않았다. 발표에는 위 1080p 직접 비교판을 주 자료로 쓰고 RRD는
보조 회전/검사용으로만 사용한다.

## 10. 최종 판정과 다음 승인 경계

수치 판정:

`D371_OFFLINE_COLLIDER_PARETO_MEASURED_NO_PHYSICS`

쉬운 말로 정리하면:

1. 64+64가 최적이라는 증거는 없다.
2. 같은 원본에서 32+32로 줄인 R32는 열린 자세에서 원통과 겹치지 않았고, 총 충돌체 수를
   절반으로 줄였다.
3. 몸통 나머지를 무조건 한 덩어리 convex로 묶은 C1/C2는 빈 공간을 메워 이미 원통과
   겹쳤으므로 실제 물리시험에 보낼 가치가 없다.
4. 이것은 R32가 실제로 더 빠르거나 원통을 잡고, 덜 쓰러뜨린다는 증명이 아니다.

보류 필드:

- `physics_equivalence=null`
- `actual_gpu_contact_execution=null`
- `collider_count_tipping_causality=null`
- `grasp_feasibility=null`
- `current64_optimal=null`
- `g0a_pass=false`

가장 좁은 다음 순서:

1. 별도 승인 D372 `[raw64_raw32_live_asset_identity_preflight]`:
   fresh R64와 R32 callback polygon을 서로 다른 새 forward-only candidate asset으로
   만들고 live readback이 각각 정확히 `64+64`, `32+32`인지 확인한다.
   physics/q5/contact는 0.
2. D372가 PASS한 뒤 다시 별도 승인:
   R64와 R32를 같은 frozen pose/trajectory/contact 기록 조건에서 물리 비교한다.
   첫 인과 비교에서는 target/IK/path를 바꾸지 않는다. 이후 필요할 때 A와 선택 후보를
   비교하되, 그 비교는 실제 기준선 비교이지 cap 하나만의 인과시험이라고 부르지 않는다.

현재는 두 단계 모두 미승인이다. D371 경로와 visual repair attempt1은 동결한다.

## 11. 주요 파일과 해시

- main harness:
  `sim_scripts/cyl34_top_view_d371_offline_collider_candidate_pareto_comparison.py`
  — `a907d650...20242`
- cook worker:
  `sim_scripts/cyl34_top_view_d371_offline_collider_cook_worker.py`
  — `c6778f8f...ce40d`
- reactive Rerun repair:
  `sim_scripts/cyl34_top_view_d371_rerun_absolute_cli_visual_repair.py`
  — `802ec424...a144`
- RRD:
  `claudedocs/runtime_logs/grasp_track/g0a_d371/d371_collider_comparison.rrd`
  — `d13e02da...c5a50`
- RBL:
  `claudedocs/runtime_logs/grasp_track/g0a_d371/d371_collider_comparison.rbl`
  — `95ba94b1...b282`
- Rerun absolute-CLI validation:
  `claudedocs/runtime_logs/grasp_track/g0a_d371/visual_repair_attempt1/d371_rerun_absolute_cli_validation.json`
  — `845562b7...2a79`
- manual inspection:
  `claudedocs/runtime_logs/grasp_track/g0a_d371/visual_repair_attempt1/d371_visual_repair_manual_inspection.json`
  — `b059969e...7d6`

No commit or push was performed.
