# D402 — runtime stack Item / counter-order authority repair

Date: 2026-07-28 KST

## 1. Case와 승인 범위

Case:
`D402 [d401_runtime_stack_item_and_counter_order_authority_repair]`

이번 case의 신규 변수:

1. `carb_dictionary_item_compatible_extension_version_accessor_v1`
2. `serialized_counter_registered_projection_authority_v1`

사용자 승인은 offline/static-only였다. 새 forward-only preregistration,
Controller/Worker thin wrapper, 정적 attestation, four-hash proposed runtime
tuple까지만 허용됐다. Controller runtime, actual Worker, Isaac Sim, Kit,
PhysX, Warp, CUDA/GPU, USD, Rerun, physics, q5, contact, cylinder,
target/IK/path 및 설정 변경은 승인되지 않았다.

사전등록:

- path:
  `claudedocs/runtime_logs/grasp_track/g0a_d402/attempt1_d401_runtime_stack_item_and_counter_order_authority_repair/d402_preregistration.json`
- SHA-256:
  `9868b1f60035682295610ce9e38e23d8fa1df37804a69386b00aaf3cf1fdfc4e`
- 신규 변수와 금지 범위:
  preregistration lines 8-30

## 2. 부팅 및 현재 기준점 재검증

Current-State Protocol 순서로 `AGENTS.md`, `START_HERE.md`,
`DECISIONS.md`, ledger tail, START_HERE가 가리킨 D401 session/evidence를
다시 읽었다.

실제 Git:

- `HEAD == origin/master ==
  9dd14ebb32421d93e3b46e2912cd3e67e6daff20`
- subject: `D401까지, D402는 미승인`
- case 시작 worktree: clean
- `codex`는 더 이상 untracked가 아니라 tracked-clean empty file이다.

기존 START_HERE Git 절의 `e9fa300...`/dirty 설명은 D401 이전 상태라
stale이었다. D402 preregistration은 실제 `9dd14ebb...`를 등록했다
(`d402_preregistration.json:32-40`).

## 3. D401 원증거로 다시 확인한 두 결함

### 3.1 첫 실제 중단점: 버전 accessor의 타입 가정

D401 attempt1은 runtime-freeze PASS, package/GPU/process PASS,
runtime negative controls `18/18`, Worker/SimulationApp/retry `1/1/0`까지
도달했다. Isaac은 RTX 4090/cuda:0에서 시작했다.

runtime stack probe는 Isaac Sim/Lab 버전, `omni.physx` extension ID,
active root, native plugin hash를 통과했지만
`omni_physx_extension_version=null`만 실패했다.

동결 D400 Worker는 `get_extension_dict()` 결과가 built-in `dict`일 때만
`package.version`을 읽었다
(`cyl34_top_view_d400_..._worker.py:397-423`). 설치 Kit 107.3.3 API는
그 반환형을 `carb.dictionary.Item`으로 선언하고, 설치 Isaac Sim help
menu도 `manager.get_extension_dict(ext_id)["package"]["version"]`을
사용한다.

따라서 D401은 Isaac/PhysX/SDF 실패가 아니라 harness의 타입 계약 실패다.
asset/SDF write, property query, physics, q5, contact, cylinder는 모두 0이고
science 판정은 null이다.

### 3.2 잠복 결함: 저장된 JSON key 순서를 의미로 오인

Worker 메모리 안의 36개 counter 순서는 등록 순서와 정확히 같았다. 그러나
전역 JSON writer가 `sort_keys=True`로 저장하므로 외부 Controller가 다시
읽은 mapping은 알파벳순이었다. 동결 Controller는 그 물리적인 저장 순서가
등록 순서와 같아야 한다고 요구해 false FAIL을 만들었다
(`cyl34_top_view_d400_..._preflight.py:986-1040`).

JSON object의 물리적 key 순서는 counter 의미가 아니다. 정확한 key 집합,
타입, 값과 등록 순서로 다시 만든 projection이 권위여야 한다.

## 4. NVIDIA 공식/설치 근거

설치 기준:

- Isaac Sim `5.1.0.0`
- Isaac Lab `2.3.0`
- Kit `107.3.3`
- omni.physx `107.3.26`

버전 일치 공식 문서:

1. NVIDIA Omniverse Kit 107.3.0,
   [Extensions in-depth](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.3.0/guide/extensions_advanced.html)
   — runtime config는 `get_extension_dict`, `data["package"]`, 필요 시
   `data.get_dict()`로 읽는다.
2. NVIDIA Omniverse Kit 107.3.0,
   [ExtensionManager Python API](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.3.0/omni.ext/omni.ext.ExtensionManager.html)
   — `get_extension_dict(ext_id) -> carb.dictionary.Item`.
3. NVIDIA Omniverse Kit
   [carb.dictionary.Item API](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/carb.dictionary/carb.dictionary.Item.html)
   — Item은 dictionary tree node handle이며 `get_dict()`로 Python object로
   직렬화할 수 있다. 이 page는 latest이므로 설치 107.3.3 stub과 위
   version-matched 107.3.0 문서를 우선 교차검증했다.

로컬 설치 1차 근거:

- `_extensions.pyi:116`:
  `get_extension_dict(...) -> carb.dictionary._dictionary.Item`
- `python_api.md:145`: 같은 반환형
- `help_menu.py:121-128`: nested indexing 후 version `.split()`
- omni.physx `extension.toml:1-6`: version `107.3.26`

다섯 설치 근거의 SHA는 preregistration lines 200-223에 등록했고, D402
Controller가 미래 runtime-freeze manifest의
`installed_primary_source_hashes`에 전부 병합한다
(`D402 Controller:236-269`). 정적 재계산은 5/5 exact였다.

## 5. 구현 — 기존 실행 흐름을 복사하지 않은 두 수리

### 5.1 Worker: Item-compatible version read

새 Worker:
`sim_scripts/cyl34_top_view_d402_d401_runtime_stack_item_and_counter_order_authority_repair_worker.py`

- D401 Worker SHA를 먼저 고정 확인한다.
- D401/D400 파일은 수정하지 않는다.
- D400의 동결 stack probe를 먼저 실행한다.
- frozen probe가 resolve한 같은 active extension ID로 config를 한 번
  재조회한다.
- `extension_config["package"]["version"]` nested indexing만 사용한다.
- 최종 값은 exact built-in `str`일 때만 허용한다.
- ID/path/TOML fallback, broad `str()` coercion, broad exception swallow는
  없다.
- frozen read 1 + Item-compatible re-read 1 = resolved 경로 총 2회임을
  artifact에 명시한다.
- version field, version exact check, top-level probe pass만 다시 계산하고
  나머지 frozen checks는 그대로 둔다.

근거: Worker lines 63-124, install/delegate lines 127-188.

### 5.2 Controller: serialized-order-independent counter authority

새 Controller:
`sim_scripts/cyl34_top_view_d402_d401_runtime_stack_item_and_counter_order_authority_repair_controller.py`

외부 supervisor의 counter gate는 다음을 모두 요구한다.

- built-in JSON mapping
- 정확히 36개인 exact key set
- 모든 값 `type(value) is int`; `bool` 거부
- 기존 exact 14개 값
- 기존 zero 21개 값
- 기존 update-pump 범위
- frozen `COUNTER_KEYS` 순서로 만든 `[[key, value], ...]` projection
- 위 projection의 canonical SHA-256

serialized iteration order는 별도 diagnostic으로만 보존되고 PASS checks에
들어가지 않는다. JSON writer의 `sort_keys=True`는 변경하지 않았다.

근거: Controller lines 81-163. 교체 함수는 D401가 반환한 같은 D400 module에
설치되므로 inherited offline controls와 `_supervise_worker()`의 raw JSON
재판정 양쪽에 도달한다(Controller lines 221-233).

D401의 pre-write Git snapshot 흐름은 복사하지 않고 hash-gated module로
그대로 호출한다(Controller lines 166-218, 272-284).

## 6. 정적 검증을 실행한 순서와 결과

1. preregistration JSON parse와 exact hash: PASS.
2. 두 wrapper AST parse: `2/2 PASS`.
3. top-level Isaac/Kit/PhysX/pxr/Warp/CUDA/torch/Rerun import: `0`.
4. `python -B`로 inert wrapper definition과 pure helper만 평가했다.
   D401/D400 main, Controller runtime, actual Worker는 호출하지 않았다.
5. Item positive controls:
   non-dict FakeItem와 built-in dict 모두 exact `107.3.26` PASS.
6. Counter positive controls:
   registered order, `sort_keys=True` JSON roundtrip, reverse insertion order
   모두 PASS. sorted roundtrip의 물리 순서는 등록 순서와 실제로 달랐다.
7. Item/counter/AST/provenance combined controls: `43/43 PASS`.
8. inherited D400 approval-schema까지 포함한 최종 tuple/static contract
   재검산: `49/49 PASS`.
9. independent read-only adversarial review 3건 후 남은 blocker `0`.

독립 검토가 attestation 전에 찾고 수리한 것은:

1. 설치 NVIDIA 근거가 미래 runtime manifest에 병합되지 않던 provenance
   누락
2. 실제 extension config read가 총 2회인데 한 번처럼 보이던 기록
3. extension ID 미해결 실패 경로에서도 requery-ID boolean이 true가 될 수
   있던 기록 모순

정적 attestation:

- path:
  `claudedocs/runtime_logs/grasp_track/g0a_d402/attempt1_d401_runtime_stack_item_and_counter_order_authority_repair/d402_reviewed_script_attestation.json`
- SHA-256:
  `c112a18d51e238ec3bd8520f5dea52452a11e060840e8038e789a6a22279561d`
- positive controls `5/5`, inherited approval-compatible negative fixtures
  `32/32`, combined D402 controls `43/43`
- evidence lines: attestation 55-313, final checks 314-403

## 7. 발급한 four-hash tuple

Tuple contents, exact field order:

| Field | SHA-256 |
|---|---|
| preregistration | `9868b1f60035682295610ce9e38e23d8fa1df37804a69386b00aaf3cf1fdfc4e` |
| reviewed static attestation | `c112a18d51e238ec3bd8520f5dea52452a11e060840e8038e789a6a22279561d` |
| D402 Controller | `af1940a57b05ad9f8afdf8359fc099437360a7ff43eb97259e1ada9eb158da52` |
| D402 Worker | `214d6dcf8e330aa3a6da8a01a614275092462fa337bb1c1fea649c3ec0d654c3` |

Tuple file:

`claudedocs/runtime_logs/grasp_track/g0a_d402/attempt1_d401_runtime_stack_item_and_counter_order_authority_repair/d402_proposed_runtime_hash_tuple.json`

Tuple-file SHA-256:

`898c91551e9b724e0d8d7114128ccfb14563f16c4e6b22aa796d07e805c012ce`

## 8. 실행하지 않은 것과 판정

이번 static stage 실제 수치:

- Controller runtime invocation: `0`
- actual Worker invocation: `0`
- Isaac/Kit/PhysX launch: `0`
- SimulationApp: `0`
- GPU runtime job: `0`
- derivative USD/SDF/collision asset write: `0`
- physics/public forward: `0/0`
- q5 command/sample: `0/0`
- contact query/cylinder: `0/0`
- target/IK/path/settings change: `0`
- RRD/RBL/Viewer: `0`

Rerun을 생략한 이유는 geometry/pose/contact/time 판정이 없는 순수
control/hash/schema 수리이기 때문이다. D341 omission 조건에 해당한다.

Session Progress Rule은 D401에서 실제로 관찰된 harness failure에 대한
reactive repair라는 점과, 결정을 바꿀 수 있는 failure-capable static
fixtures `32/32`를 실행한 것으로 충족한다.

Verdict:

`D402_RUNTIME_STACK_ITEM_AND_COUNTER_ORDER_AUTHORITY_REPAIR_STATIC_ATTESTATION_PASS_RUNTIME_NOT_APPROVED`

이는 Isaac, PhysX, SDF, collision, contact, tipping, grasp PASS가 아니다.
`scientific_or_physics_verdict=null`, `g0a_pass=false`다.

## 9. 다음 승인 경계

Actual runtime은 아직 미승인이다. 진행하려면 사용자가 tuple-file SHA
`898c91551e9b724e0d8d7114128ccfb14563f16c4e6b22aa796d07e805c012ce`
를 정확히 명시해 D402 Controller one-shot runtime을 별도로 승인해야 한다.

이 tuple은 현재 등록 Git
`HEAD==origin/master==9dd14ebb32421d93e3b46e2912cd3e67e6daff20`
에만 유효하다. 그 전에 commit/push로 HEAD가 바뀌면 이 tuple을 실행에
사용하지 않고 새 forward-only baseline/tuple을 발급해야 한다.

Future runtime이 승인되더라도 범위는 동결 D400의 asset
load/cook/readback preflight뿐이다. physics, q5, contact, cylinder,
target/IK/path는 계속 0이며 이후 과학/물리 단계는 다시 별도 승인이다.

Commit/push는 수행하지 않았다. D334 sidecar와 D400/D401 frozen evidence는
수정하지 않았다.
