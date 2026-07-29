# D403 actual run — 인프라 전부 PASS, 동결 authored-derivative 게이트의 계약 결함 4건으로 FAIL_STOP

Date: 2026-07-28 저녁 KST (이어서). 이번 case의 신규 변수:
`[git_baseline_repin_to_a69a96d36219268e4bc5e25065cc234da9d99674,
gpu_visible_host_boundary_execution_v1]` (정확히 2개, prereg 그대로) — **둘 다
runtime에서 실제 검증 성공**. 이번 세션의 승인된 1회 runtime attempt는 소모됨.

## 1. 무엇을 왜

이전 세션이 준비한 D403 정적 산출물(prereg `fd403c66...e919f0`, wrapper 2종,
3-lens 리뷰 blocker 0)을 이어받아, 사용자 순차 고속 실행 지시(2026-07-28)에
따라 attestation → tuple → 호스트 경계 1회 실행을 수행했다. 목적은 동결 D402
계약(D400 gripper_link A64→SDF res256 load/cook/readback preflight + D402
harness 수리 2건)의 첫 실제 실행이었다.

## 2. 실행 순서 (감사 가능 step-by-step)

1. **부팅 검증**: 상속 해시 31개(체인 13 + 동결 attempt 증거 13 + 설치 NVIDIA
   5) 전부 이 세션에서 재계산·prereg pin과 bit-identical 확인. D403 wrapper
   2종 AST 재파싱 PASS, 금지 top-level import 0, `__pycache__` 0.
2. **`d403_reviewed_script_attestation.json` 작성** (sha
   `416492f27a1f98fba457036732dc71098f1169005c24156dce51d48baa4004d9`).
   근거 분리 명기: 18종 science fixture = 전 체인 해시 동일성 기반 재승인,
   D402-layer 14종 + positive 5종 = 2026-07-28 재실행 전사(이전 세션 doc §2.3).
   작성 전 `_validate_approval_tuple`(D400 controller 462-595행) 로직을
   오프라인으로 복제해 전 항목 선검증 PASS.
3. **`d403_proposed_runtime_hash_tuple.json` 작성** (4필드 정확한 순서).
   tuple sha = `ed5f016bedba7c318aca9e37eb4952d83584d6d145ce0f0ebf6b6392e08ec122`.
4. **실행 전 점검**: nvidia-smi RTX 4090 Laptop / driver 580.173.02 / free
   15081 MiB(요구 8192+) / 잔존 Isaac·Kit·omni 프로세스 0 / D403 문자열
   conflict 프로세스 0 / `HEAD==origin/master==a69a96d` / dirty 10경로 전부
   prereg allowed 목록 내. 실행 셸 호스트 경계 확인(pid 961721, /dev/nvidia*
   3종 가시).
5. **호스트 경계 1회 실행** (isaaclab python -B, background, retry 0):
   총 소요 **28.7s**. 결과 exit 1, verdict
   `D400_GRIPPER_LINK_SDF_RES256_PREFLIGHT_FAIL_STOP` (technical FAIL →
   manual inspection 단계 미도달, 정상 분기).

## 3. Runtime에서 실제로 PASS한 것 (이번이 최초)

- **host_boundary_gate v1 통과 후 위임** — worker pid 993256(호스트 범위),
  D402의 pid=2 샌드박스 문제 재현 없음. D402-R1 메커니즘 runtime 검증 완료.
- **git repin baseline** — runtime freeze manifest PASS(`3c1e0fc9...`),
  post-worker authority binding gate 15개 체크 전부 PASS(`head_still_exact`
  포함, supervisor `post_worker_runtime_authority_binding_gate.pass=true`).
- **offline negative controls 18/18** runtime 재실행 PASS (phase ordinal 4).
- **SimulationApp headless GPU 기동 5.3s** (phase 6→7), kit log 57행
  ERROR/FATAL/WARNING 0 (`d400_kit_log.txt`).
- **D402 harness 수리 2건 최초 라이브 PASS**: runtime stack probe
  `omni_physx_extension_version=107.3.26` Item-호환 접근 성공
  (`d400_worker_raw_summary.json` runtime_stack_probe.pass=true — D401 actual의
  blocker 해소 확인), counter registered-projection 권위 동작
  (`serialized_order_used_as_pass_authority=false`).
- **derivative authoring 자체 성공**: bit-exact copy, root layer만 변경,
  instanceable=false 2회, A64 disable 64회, API Apply 3종 authored+saved
  (root layer listOp `Prepended [PhysicsCollisionAPI, PhysicsMeshCollisionAPI,
  PhysxSDFMeshCollisionAPI]` 확인), SDF 속성 7종 bit-정확 authored
  (`0x3c23d70a`, `0x3f800000` 일치).

## 4. FAIL 지점과 근본 원인 — 동결 계약 결함 4건 (Isaac/PhysX/SDF 실패 0건)

Worker가 `_author_sdf_derivative`(worker.py:1378)에서 raise. 13개 체크 중 2개
실패: `sdf_typed_readback_pass=false`, `composed_semantic_diff_allowlist_exact=false`.
PhysX attach 이전이므로 cook/SDF는 시작도 안 함. 사후 read-only 오프라인
진단(§5)으로 하위 원인 4건 분해:

| # | 결함 | 증거 |
|---|------|------|
| 1 | **`physics:collisionEnabled` uniform 기대는 스키마와 모순.** 체크가 `uniform=True`를 요구하나, 설치 usdPhysics 스키마는 `bool physics:collisionEnabled = true`로 **varying** 선언(uniform 키워드 없음) → 원리상 통과 불가 | 설치 `omni.usd.libs-1.0.1.../bin/usd/usdPhysics/resources/usdPhysics/schema.usda:285`; probe 관측 `variability=VariabilityVarying` |
| 2 | **float32 왕복 등호 버그.** `sdfNarrowBandThickness`/`sdfMargin` 기대값 0.01(double)을 `value == expected`로 비교. float32 저장 후 Get()은 0.009999999776482582 → 항상 불일치. 스펙 자체의 bits 체크(`0x3c23d70a`)는 **PASS** — 의도한 float32가 정확히 저장됐다는 증거 | worker.py:1447-1460 (`row["value"] == expected_value`), SDF_ATTRIBUTE_SPECS(worker.py:174-179); probe attribute rows |
| 3 | **normalizer가 attr `value`만 마스킹, `default` metadata 미마스킹.** 64개 A64 part의 collisionEnabled=false authoring이 attr metadata의 `default`(true→false)에도 반영 → allowlist 마스킹에도 row 해시 64건 불일치 | worker.py:1072-1075 (value만 마스킹) vs `_composed_stage_rows`가 `GetAllMetadata()` 인코딩(worker.py:997); probe part_000 diff: metadata default true→false |
| 4 | **relationship allowlist 부재.** mesh에 PhysicsCollisionAPI를 Apply하면 builtin `rel physics:simulationOwner`가 composed prim definition에 추가되는데 normalizer는 attributes/applied_schemas/apiSchemas-metadata만 모델링 → mesh row 1건 불일치 | 설치 usdPhysics `schema.usda:293`; probe mesh diff: rel physics:simulationOwner base=null → deriv=builtin |

semantic mismatch 총 65행 = 결함3(64행) + 결함4(1행). 결함 1·2는
`sdf_typed_readback_pass`의 실패 하위 체크 전부이며 **어떤 입력에서도 통과
불가능한 기대치**였다. 이 게이트 코드는 D400(정적만)→D401(freeze에서
정지)→D402(GPU preflight 정지)에서 한 번도 라이브 실행된 적이 없고, 정적
fixture·3중 적대적 리뷰도 못 잡았다 — **첫 라이브 실행 자체가 실험이었고
실패 가능했으며 실제로 실패했다** (session progress rule 충족).

부수: supervisor의 counter gate FAIL(`exact_14=false`: physx_property_queries
등 0)은 조기 fail-stop의 하류 결과이지 독립 결함 아님. `safe_to_close_app=false`
동일. worker returncode 0이지만 protocol pass false로 정확히 강등됨(등록된
`worker_internal_fail_return_zero_rejected` 패턴의 정상 동작).

## 5. 사후 진단 방법 (read-only, 재실행·수정 0)

- 디스크에 남은 derivative
  (`g0a_d403/attempt1_*/collision_asset/roarm_m3_link5_a64_gripper_sdf_res256/`)를
  Kit 기동 없이 열어, **동결 worker의 게이트 함수를 그대로 재적용**:
  isaaclab env python -B + `omni.usd.libs` pxr(USD 24.05) + PhysxSchema
  plugInfo 정식 등록(`Plug.Registry().RegisterPlugins`). repo 쓰기 0,
  isaaclab env 변경 0 (PYTHONPATH/LD_LIBRARY_PATH만).
- **registry 아티팩트 판별**: plugInfo 미등록 상태에선
  `required_apis_applied=false`(composed applied에서 PhysxSDFMeshCollisionAPI
  누락)로 보였으나, 등록 후 정상 포함 → in-worker(Kit)에선 이 체크 PASS로
  추정 [INF: worker 내부 하위 체크 상태는 exception이 top-level만 기록해
  직접 관측 불가; registry-correct 복제 + raw listOp authored 증거 기반 추론].
- probe 산출물(휘발, scratchpad): `d403_failure_diagnosis_result.json`,
  `d403_failure_diagnosis_result2.json` — 핵심 수치는 본 문서에 전사.

## 6. Verdict와 상태

- Canonical (동결 코드 출력): `D400_GRIPPER_LINK_SDF_RES256_PREFLIGHT_FAIL_STOP`
- Descriptive root-cause label:
  `D403_D400_AUTHORED_DERIVATIVE_GATE_CONTRACT_FAIL_STOP` (계약 결함 4건,
  NVIDIA 스택 실패 0건)
- `scientific_or_physics_verdict=null`, `g0a_pass=false` (변동 없음). SDF
  load/cook/readback 과학 질문은 여전히 미측정.
- **D403 attempt1 동결** — 같은 경로 retry/덮어쓰기 금지. attempt 예산(1회)
  소모.

## 7. 다음 case 설계 — D404 [d403_authored_derivative_gate_contract_repair] (승인 대기)

Reactive control-contract 수리(관측된 실패에 대한 대응 — AGENTS.md 규칙상
허용). 신규 변수 1: `authored_derivative_gate_contract_repair_v1` (수리 4건
일괄 = 본 세션 관측 실패의 최소 대응):

1. float 속성 등호를 **bits 권위**로: `float32_bits_hex == expected_bits`가
   있으면 value 이중 등호 제거(또는 float32 왕복 비교).
2. `collision_enabled_authored_uniform_noncustom_default_only`의 uniform
   요구를 설치 스키마 선언(varying)에 맞춤 — authored/non-custom/default-only
   유지.
3. normalizer: allowlist된 `physics:collisionEnabled` 경로에서 attr
   `metadata`의 `default` 항목도 동일 마커로 마스킹.
4. normalizer: mesh 경로의 relationship 중 세 applied API의 builtin
   (`physics:simulationOwner`)을 allowlist 필터에 추가.

구현 패턴: D402의 `_install_counter_authority_repair`와 동일 — D404 thin
wrapper가 동결 체인 로드 후 **동결 D400 worker 모듈의
`_sdf_prim_readback`/`_normalize_allowlisted_semantics` 함수 객체만 교체**
(동결 파일 수정 0). 새 prereg/attestation/tuple/정적 fixture(수리 4건 각각의
양성·음성 대조) + 리뷰 후 1회 실행. cook/PhysX 이후 단계는 D400 계약 그대로.

## 8. 경고 (다음 세션 필독)

- D400/D401/D402/D403 전 attempt 동결. D403 경로 재실행 금지.
- 이번 FAIL은 **Isaac/PhysX/SDF/드라이버 실패가 아님** — 인프라 계보(D401
  kit 정상, D402-R1 샌드박스 정정, D403 호스트 실행 성공)와 혼동 금지.
- preflight.py:1420의 `"true" is not True` SyntaxWarning은 동결 코드의 기존
  경고 — 수정 금지(동결), 실행 영향 없음.
- D404 prereg 작성 시 allowed_dirty_paths에 본 세션 산출 문서들 포함할 것.
- commit/push는 사용자 요청 시에만 (현재 미요청 상태 유지).
