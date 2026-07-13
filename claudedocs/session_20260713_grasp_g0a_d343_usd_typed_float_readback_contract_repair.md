# 2026-07-13 Grasp G0a D343 — USD typed-float/readback-contract repair

이번 case의 신규 변수: `[usd_float_parameter_readback_contract]`

상태: **COMPLETED — `D343_USD_TYPED_FLOAT_READBACK_CONTRACT_PASS`**

## 1. 무엇을 왜 하는가

D342는 direct authored coordinate/hash와 body-mapped geometry를 `13/13`으로
지지했지만, 등록된 전체 판정은 FAIL이었다. 유일한 false predicate는
`min_thickness_frozen_1e_4m=0/13`이었다. immutable D339의 USD `float`
readback은 `9.999999747378752e-05m` (`0x38d1b717`)인데, D342 harness가
동결된 `1e-10m` 대신 미등록 `1e-12m` comparator를 사용했기 때문이다.

D342를 소급 PASS로 바꾸거나 곧바로 attempt3를 만들 수는 없다. 전자는
등록 verdict를 지우고, 후자는 proof-contract repair와 collision-asset mutation을
한 case에 섞는다. D343은 immutable D339 attempt2에서 USD scalar의 direct
authorship, type, composed resolve source, typed value와 bits를 별도로 증명한 뒤
정지한다. PASS는 별도 승인 D344의 eligibility만 연다.

## 2. 왜 D343가 다음인지 재검증

세 독립 관점과 로컬 API scoping을 대조했다.

1. **Case-boundary challenge:** D342의 registered verdict는 FAIL이며 evidence가
   raw type/authored opinion/bits를 보존하지 않았다. posthoc 재분류나 direct
   attempt3는 허용할 수 없다.
2. **Float-contract challenge:** `0.0001` 부근 float32 1 ULP는
   `7.275957614183426e-12m`이다. lower/upper adjacent float 모두 기존
   `1e-10m`을 통과하므로 tolerance는 typed identity authority가 될 수 없다.
3. **Rerun scope challenge:** 본 case는 spatial/temporal 판단이 없는 scalar
   schema/type/bit audit이다. 새 Rerun은 서면 예외로 생략하되 D342 RRD를
   D343 완료 artifact로 재사용하지 않는다.
4. **Installed API scoping:** standalone OpenUSD 0.24.5에서 direct Sdf spec과
   composed Usd attr를 읽을 수 있다. core-only PXR은 등록되지 않은 PhysX
   schema를 `GetAppliedSchemas()`에서 생략하므로 direct `apiSchemas` metadata가
   API-authorship authority다.

곧바로 D344로 가지 않는 이유는 proof repair와 asset authoring의 독립 실패
가능성을 보존하기 위해서다. D343이 실패하면 attempt3를 만들 이유가 없고,
D343이 통과해도 asset mutation은 여전히 별도 명시 승인이 필요하다.

## 3. 변수와 파라미터 감사

- 신규 변수: measurement-only 1개
  `[usd_float_parameter_readback_contract]`.
- physical variable 변경: `0`.
- existing parameter increase/change: `0/0`.
- decomposition parameter change: `0`.
- threshold relaxation: `0`.
- target/controller/solver change: `0`.
- collision asset write, recook, physics step, attempt3: `0/0/0/absent`.

동결 decomposition은 그대로다:

- `hullVertexLimit=64`
- `maxConvexHulls=64`
- `voxelResolution=1,000,000`
- `errorPercentage=1.0`
- `minThickness=0.0001m`
- `shrinkWrap=true`

동결 target/control도 `q5=1.5413rad`, `(radial,tangent)=(7,11)mm`,
`tangent_sign=-1`, seed `33201`, HOME, position-only IK 그대로다.

관측 subject는 D342 failure subset 13개에서 같은 authored scalar를 가진 D339
전체 128개로 확장한다. 이는 변수를 추가하거나 값을 올린 것이 아니다. 향후
attempt3는 13개만 교체하고 115개를 보존하므로, 보존될 part 하나라도 scalar
계약이 다르면 D344 eligibility가 바뀐다. 동일 변수의 decision-relevant sample
coverage를 완성하는 것이다.

권위 있는 감사 파일:
`claudedocs/runtime_logs/grasp_track/g0a_d343/d343_parameter_freeze_audit.json`.

## 4. Exact typed-float contract

요청값과 typed representation을 분리한다.

- requested decimal: `0.0001m`
- USD storage type: `float`
- expected typed value: `9.999999747378752e-05m`
- expected uint32: `953267991`
- expected bits: `0x38d1b717`
- little-endian bytes: `17b7d138`
- decimal representation delta: `2.526212488436659e-12m`
- historical compatibility tolerance: `1e-10m`

주 hard gate는 exact float32 bits다. `1e-10m`은 D339/D340/D342 compatibility와
D342 root-cause 재현을 위한 보조 diagnostic일 뿐 identity authority가 아니다.

설치된 PhysX schema source는 이 attr를 `float`, fallback `0.001`,
`Units: distance`로 선언한다. fallback typed bits는 `0x3a83126f`이며 expected
authored bits와 다르다. schema plugin의 core 등록 여부에 의존하지 않고 direct
Sdf authored default와 Usd resolve source를 함께 확인한다.

## 5. Per-part 판독 순서

각 128 part에서 다음을 모두 요구한다.

1. `Sdf.Layer.FindOrOpen`으로 immutable D339 physics layer를 연다.
2. direct prim의 `apiSchemas` metadata에
   `PhysxConvexHullCollisionAPI`가 authored됐는지 확인한다.
3. direct `Sdf.AttributeSpec`의 exact path, `float` type, authored default field,
   `0x38d1b717` bits를 확인한다.
4. 독립 `Usd.Stage.Open` composed attr의 exact name/type/value/bits를 확인한다.
5. `HasAuthoredValueOpinion`, `HasAuthoredValue`, unblocked, zero time samples,
   non-varying을 확인한다.
6. resolve source가 authored `Default`이고 schema `Fallback`이 아님을 확인한다.
7. property stack가 정확히 1개이며 동일 immutable D339 physics layer의 direct
   spec/type/default bits인지 확인한다.
8. physics layer에 `metersPerUnit=1.0`이 direct authored됐고 composed 값도
   `1.0`인지 확인한다.
9. D339 live readback의 check와 bits를 교차 확인한다.
10. D342 failure subset 13개는 D342 direct-hash/numeric anchors와 추가 교차한다.

Geometry point/index array는 판정에 사용하지 않는다.

## 6. 실패 가능한 perturbation

원본 파일을 쓰지 않고 memory에서 expected float32의 양쪽 adjacent 값을 만든다.

- lower: `0x38d1b716` = `9.99999901978299e-05m`
- upper: `0x38d1b718` = `0.00010000000474974513m`

두 값 모두 exact-bit validator가 거부하고, 동시에 historical `1e-10m`
comparator는 둘 다 받아들여야 discriminator가 PASS한다. 또한 올바른 expected
typed 값이 `1e-10m`은 통과하고 D342의 executed `1e-12m`은 실패하는지 재현한다.
이 perturbation은 D343 verdict를 실제로 FAIL로 바꿀 수 있으므로 session
progress rule의 failure-capable evaluation이다. 이 case는 D342에서 관측된
실패에 대한 reactive contract hardening이며 training/physics를 수행하지 않는
이유도 active proof-only scope에 있다.

## 7. Rerun 생략 계약

D343은 geometry, pose, contact, tool frame, motion, event sequence 또는 time을
해석하지 않는다. Exact Sdf/Usd type, authored opinion, resolve source, raw bits는
JSON과 immutable hash가 권위이며 scene viewer가 증거를 강화하지 않는다.

따라서 새 RRD/RBL/PNG는 만들지 않는다. D342 RRD는 context hash만 pin하고
D343 completion artifact로 재사용하지 않는다. 공간·시간 판단, Kit/physics,
geometry decision 중 하나라도 생기면 이 생략은 무효다.

권위 있는 생략 파일:
`claudedocs/runtime_logs/grasp_track/g0a_d343/d343_rerun_omission_justification.json`.

## 8. Runtime/exit 경계

- standalone PXR core only; Isaac Kit/GPU 불필요.
- `SimulationApp`, `SimulationContext`, PhysX cook, physics step 없음.
- source/attempt1/attempt2/D340/D342 write 없음.
- evidence와 summary는 새 D343 forward-only 파일로만 쓴다.
- PASS exit `0`, scientific FAIL exit `2`.
- effective run은 preregistration 통과 후 정확히 1회만 한다.

## 9. Registered command

```bash
env \
  PYTHONPATH=/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
  LD_LIBRARY_PATH=/home/cgxr/miniconda3/envs/isaaclab/lib:/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin \
  /home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/cyl34_top_view_d343_grasp_g0a_usd_typed_float_readback_contract_repair.py
```

## 10. PASS/FAIL and stop rule

PASS requires all of:

- preregistration and parameter/Rerun-omission contracts PASS.
- direct Sdf + composed Usd typed contract `128/128`.
- D342 subset anchors `13/13`.
- exact-bit adjacent negative PASS.
- D342 `1e-10` vs `1e-12` comparator reproduction PASS.
- D339/D340/D342 inventories exact.
- source writes/recook/physics/new Rerun/attempt3 all zero or absent.

하나라도 실패하면 `D343_USD_TYPED_FLOAT_READBACK_CONTRACT_FAIL_STOP`이다.
전체 통과 시 `D343_USD_TYPED_FLOAT_READBACK_CONTRACT_PASS`지만 D343은 즉시
정지한다. D342 verdict와 `g0a_pass=false`는 유지한다. PASS는 별도 승인 D344의
attempt3 authoring + fresh live-validation eligibility만 만든다.

## 11. Preserved preregistration-time result placeholder

사전등록 시점의 기록: `Not run yet. Preregistration hash/inventory lock and
standalone-PXR preflight remain.` 실제 실행 결과는 아래 §12-16에 순서대로
append했다.

## 12. Preflight 결과

위 placeholder 뒤에 preflight와 effective run을 순서대로 완료했다. Preflight는
기계 check `35/35`를 통과했다.

- 신규 변수/개수: exact `[usd_float_parameter_readback_contract]` / `1`.
- registered command/env, HEAD, script/session/START/audit hashes: exact.
- D339 attempt2: `18` files, inventory digest
  `0dae41fd3937a0a8aea18488019c74f097d32f7b8de916943ff31334e30464a1`.
- D340: `33` files, digest
  `def37cc3c4d10cad8919ce71175211cc34fe2e8b567dbc107f13de151a92940d`.
- D342: `13` files, digest
  `7c205d7f6222a2a091a70bb1cf784b339512efbfe8d50bbb3b5ee8c2fed35232`.
- `numpy==1.26.0`, `psutil==5.9.8`, OpenUSD `0.24.5`.
- 128-part manifest allowlist, 13-part D342 subset, schema/type/default/unit,
  exact bits, frozen tolerance, Rerun omission, attempt3 absence: all exact.

## 13. 최초이자 유일한 effective run

등록 명령을 한 번 실행했다. wall time은 약 `0.21s`, shell exit은 `0`이었다.
재실행하지 않았다. Verdict는
`D343_USD_TYPED_FLOAT_READBACK_CONTRACT_PASS`다.

### 13.1 128-part typed contract

- typed attribute PASS: `128/128`.
- per-part predicates: `32`개씩, 총 `4,096`; false predicate `0`.
- direct Sdf `AttributeSpec`, direct default field, `float` type,
  `PhysxConvexHullCollisionAPI` authored token: `128/128`.
- composed Usd attr valid/name/type, authored value/opinion: `128/128`.
- actual/direct/live unique bits: 모두 `0x38d1b717` 한 종류.
- actual typed value: 전부 `9.999999747378752e-05m`.
- resolve source: 전부 `Usd.ResolveInfoSourceDefault`; schema fallback `0/128`.
- blocked/time-varying/nonzero time sample: 각각 `0/128`.
- property stack: 전부 정확히 `1`개이며 immutable D339 physics layer의 동일
  spec/type/default bits.
- direct-authored/composed `metersPerUnit=1.0`; layer dirty before/after false.
- D342 failure subset anchor: `13/13`.

PhysX schema fallback은 typed `0.0010000000474974513m`, bits
`0x3a83126f`로 실제 authored 값과 명확히 달랐다.

### 13.2 Failure-capable discriminator

Expected typed bits는 `0x38d1b717`, uint32 `953267991`, LE bytes
`17b7d138`; 1 ULP는 `7.275957614183426e-12m`이다.

- lower `0x38d1b716`: exact-bit reject, `1e-10m` accept.
- upper `0x38d1b718`: exact-bit reject, `1e-10m` accept.

따라서 adjacent negative는 PASS했다. 이는 `1e-10m`이 compatibility에는
충분하지만 typed identity authority가 될 수 없음을 실제 반증으로 보였다.

D342 comparator도 재현됐다.

- 올바른 typed 값의 decimal delta:
  `2.526212488436659e-12m`.
- frozen `1e-10m`: PASS.
- D342 executed `1e-12m`: FAIL.

즉 D342 실패 원인은 실제 minThickness 값이나 geometry가 아니라 미등록
comparator tightening이라는 기존 root-cause와 일치한다. D342 자체는 소급
PASS로 바꾸지 않는다.

### 13.3 Scope와 불변성

- physical variables changed: `0`.
- existing parameter increases/changes: `0/0`.
- decomposition changes/threshold relaxations: `0/0`.
- collision asset writes/recook/SimulationContext/physics: `0/0/false/0`.
- attempt3: absent.
- new Rerun/RBL/PNG: `0/0/0`; 등록된 scalar/schema/bit exception 사용.
- D342 RRD는 context hash만 사용했고 D343 completion artifact로 재사용하지
  않았다.
- D339/D340/D342 inventory는 위 count/digest로 before==after exact.
- `g0a_pass=false`; G0b/RL/ladder promotion 없음.

## 14. Final verdict와 다음 경계

Final verdict:
`D343_USD_TYPED_FLOAT_READBACK_CONTRACT_PASS`.

일상어로는 D339 attempt2의 `minThickness=0.0001m`가 전체 128개 part에 정확히
float32로 authored돼 있고, D342가 정상 값을 너무 빡빡한 미등록 비교 기준으로
잘못 떨어뜨렸음을 독립적으로 확인한 것이다. 물리 자산을 고쳤거나 grasp가
성공했다는 뜻은 아니다.

D343은 여기서 정지한다. D343 PASS는 별도 승인 D344가 attempt3를 author하고
fresh live validation을 수행할 **자격만** 만든다. D344는 아직 승인·실행되지
않았다. D342 verdict, `g0a_pass=false`, G0b/RL/ladder block은 유지한다.

## 15. Evidence

- `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_parameter_freeze_audit.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_rerun_omission_justification.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_evidence.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_report.md`
- `sim_scripts/cyl34_top_view_d343_grasp_g0a_usd_typed_float_readback_contract_repair.py`

## 16. Final cross-validation

- Evidence actual SHA-256:
  `95bb4e3787d300071f1bac22037814b732781cd72a69a0334a34a05a50ac920b`,
  summary가 기록한 값과 exact.
- Evidence JSON의 모든 per-part check를 독립 집계: false `0`; direct API
  authored `128`, authored value+opinion `128`, fallback/blocked/time-varying
  각각 `0`, property-stack length `1` count `128`.
- D343 output은 preregistration/parameter audit/Rerun omission/evidence/
  summary/report의 6개 forward-only 파일뿐이며 RRD/RBL/PNG가 없다.
- `find` 기준 grasp-track 전체에 `collision_asset/attempt3`가 없다.
- D343 harness `py_compile`, 모든 D343 JSON parse, `git diff --check`: PASS.
- Rerun completion-contract regression test는 정확한 `isaaclab` Python과
  `isaaclab/bin` PATH에서 `7 passed` (`1` numpy-compat deprecation warning).
  앞선 base-shell 시도들은 각각 repo import path, `gymnasium`, Rerun CLI PATH
  누락으로 collection/validation 전에 중단된 launch-only 진단이며 소스,
  dependency, evidence, verdict를 바꾸지 않았다.
- D343 scientific harness는 재실행하지 않았다.
