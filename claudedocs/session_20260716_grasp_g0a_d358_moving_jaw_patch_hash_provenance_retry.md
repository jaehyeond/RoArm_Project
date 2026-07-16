# D358 moving-jaw patch-hash provenance retry — preregistration

Date: 2026-07-16 KST

Case: `g0a_d358`

이번 case의 신규 변수:

1. `bundled_standalone_core_pxr_execution_contract`
2. `derived_moving_jaw_patch_hash_provenance_semantics`

신규 physical variable: `[]`

## 1. What and why

D354의 moving-jaw binding은 authored point/index/count stream과 face ID/order는
exact였지만 derived patch hash와 runtime roundtrip exactness가 달랐다. D355는 이
차이가 dtype, unit, ordering, canonicalization 중 어디서 생겼는지 offline으로
감사하려 했으나, plain `isaaclab` Python에 bundled PXR 경로를 등록하지 않아 첫
`from pxr import ...`에서 멈췄다. D356은 D343/D345가 이미 같은 설치의 bundled
`omni.usd.libs`를 standalone core-PXR로 성공시켰음을 확인해 원인을 정정했다.

D358의 질문은 하나다.

> D351에 고정된 여덟 expected derived hash와 D354에서 다시 계산된 authored/raw
> hash가 정확히 어떤 coordinate source, dtype, byte order, memory layout, vertex/
> triangle canonicalization, digest blob order에서 만들어졌는가?

이 질문은 cap/rim 접촉 순서, q5 geometry, PhysX contact, grasp feasibility를 다시
판정하지 않는다.

## 2. Frozen inputs

- authoring USD SHA-256:
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`
- D339 asset manifest:
  `3b46cb39a1f0ff655dcd46172ebaa84f727d833773275b18f944397007ae2589`
- D354 binding / measurement / completion / attestation:
  - `548d45ec4eb1dacbb4cbdefe2b64a3ed99ce72f4f5ffaaa6a9ee1e2b38756b15`
  - `fd0d43c2a47abefad939fc0e980456cc396bbf3ba3bb104b28bcd999100f23ed`
  - `5cc70c8aa1e50532fa4ec27756496d6b9f9447156c56ef700084b44c16226f86`
  - `1975df11b13a774b89f953991d7fdac6e38d795e81a9535850b147d823740d20`
- frozen D355 helper harness:
  `b1fe5bf0f42c3d30a2b56d6809e17cfe4785eb7dcb610e2cf6fc05fb57c50d46`
- D343 prereg / D345 prereg / D345 worker-A standalone-PXR evidence:
  - `fb8f9c292042001aeb05d9b693d910797bd4a214d9e01427ccd54b7e2c387ce8`
  - `9c31b8070d2051c00ebd6789facd6c8a59256cb9beefe8645a63ff41a277b6a3`
  - `99991b382bf881502dc73009877cd09a5617be8d3a5a5610a0d047f741756974`
- D357 completion is context-only and pinned at
  `89a20139c12d6936ae052d0069829f0381e6935ba5dcb1b3dcbf581fc3581e71`.

D334 user-owned sidecar는 prepare 전 exact file inventory와 hash를 읽고 audit 뒤
동일성을 검사한다. 쓰기는 0회다.

## 3. Exact standalone core-PXR runtime

- Python: `/home/cgxr/miniconda3/envs/isaaclab/bin/python`
- `PYTHONPATH`:
  `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311`
- `LD_LIBRARY_PATH`:
  `/home/cgxr/miniconda3/envs/isaaclab/lib:/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin`
- required OpenUSD version: `[0,24,5]`
- required package pins: `numpy==1.26.0`, `psutil==5.9.8`
- PXR Sdf/Usd binary SHA-256:
  - `b4e3056cf5622e0f3036a74876b180c019e46beddc10beb821987020c0c7bbbc`
  - `0071f15c896e2252f647384d2276c9b7c9211c0c08f6553701a2432451d2d3c4`

`prepare`와 sole `audit` invocation 모두 exact environment를 검사한다.
`pxr/Gf`, `pxr/Usd`, `pxr/UsdGeom` module origin, in-memory smoke stage, binary
hash가 모두 등록 root와 일치해야 한다. `isaacsim`, `omni`, `physx`, `carb`,
`warp`, `torch` Python module이 하나라도 load되면 fail-closed다.

## 4. Registered audit grid and independent checks

Coordinate source 9개와 다음 axes의 Cartesian product `20,736` recipes를 finite
search한다.

- authored/raw, meter/millimeter, Float32/Float64와 Float32 roundtrip
- face ascending/descending
- winding preserve/flip
- vertex lexicographic unique/stable first occurrence
- triangle preserve/cyclic-min/unoriented-sort
- triangle row face-order/lexicographic
- signed-zero preserve/normalize
- vertex little-endian Float32/Float64
- triangle little-endian Int32/Int64
- digest order `FVT/FTV/VFT/VTF/TFV/TVF`

NumPy vectorized 결과를 그대로 믿지 않는다. 별도 Python tuple/dict/
`struct.pack` 구현이 다음을 independently replay해야 한다.

1. current authored/raw inner+outer vertex/triangle/patch/paired-XZ와 full raw stream
2. frozen target 각각을 처음 재현한 exact recipe
3. coherent eight-field recipe가 발견되면 그 전체 bundle

또한 각 target-matching recipe의 실제 canonical arrays를 little-endian C,
big-endian C, little-endian Fortran layout으로 다시 serialize한다. registered
little-endian C만 target을 맞고 BE/F alternative는 target에서 달라야 한다.

기존 7 perturbation controls도 모두 PASS해야 한다: wrong unit, wrong dtype,
big endian, Fortran order, reverse face order, flip winding, digest order FTV.

## 5. Decision rule

- `D358_HASH_PROVENANCE_LOCALIZED_COHERENT_RECIPE`: 모든 frozen/current stream,
  independent replay, negative control이 PASS하고 한 recipe가 여덟 expected field를
  동시에 재현한다.
- `D358_HASH_PROVENANCE_LOCALIZED_INCOHERENT_FROZEN_BUNDLE`: 모든 stream과
  controls가 PASS하고 각 expected field는 재현되지만, 한 recipe로 여덟 field를
  동시에 재현할 수 없다. 이는 historical expected bundle이 서로 다른 provenance
  semantics를 섞었다는 뜻이다.
- `D358_HASH_PROVENANCE_UNRESOLVED_FAIL_STOP`: stream, independent replay,
  negative control, 또는 expected field 재현 중 하나라도 실패한다.
- input/runtime가 먼저 멈추면 `D358_OFFLINE_INPUT_OR_RUNTIME_FAIL_STOP`이다.

어떤 outcome도 D354의 `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`을 바꾸거나
binding gate를 수정하지 않는다.

## 6. Exact execution order

1. `--stage prepare`: HEAD/origin, all frozen hashes, pins, PXR environment,
   20,736 grid count, D334 sidecar를 검사하고 prereg JSON만 exclusive-create한다.
2. `--stage audit`: prereg PASS, current Git, harness/session/input/env를 다시 검사한
   뒤에만 invocation marker를 exclusive-create한다.
3. 300초 wall-clock alarm 아래 frozen USD streams를 1회 읽고 recipe grid,
   independent replay, byte-layout alternatives, negative controls를 실행한다.
4. authoritative evidence JSON과 report를 쓴다.
5. completion 전 exact inventory를 검사하고 completion JSON을 마지막에 쓴다.
6. retry, overwrite, manual environment injection, second audit invocation은 없다.

Registered commands:

```bash
env PYTHONPATH=/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
  LD_LIBRARY_PATH=/home/cgxr/miniconda3/envs/isaaclab/lib:/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin \
  /home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d358_moving_jaw_patch_hash_provenance_retry.py --stage prepare

env PYTHONPATH=/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
  LD_LIBRARY_PATH=/home/cgxr/miniconda3/envs/isaaclab/lib:/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin \
  /home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d358_moving_jaw_patch_hash_provenance_retry.py --stage audit
```

Harness SHA-256:
`e5fdc1fc98c85db1b2533ae45e080999681a9b88c59c593bf5cb8869ee87757a`.

## 7. Frozen prohibitions and observability

- SimulationApp/Kit/Isaac GUI/GPU/RTX/Warp/nvidia-smi: `0`
- q5 science/distance/contact/overlap/cap-rim classification: `0`
- controlled physics steps: `0`
- asset/decomposition/gate/tolerance/material/mass/actuator/physics change: `0`
- target/IK/path, settle, ten-trial, hold/lift, G0b, RL/PPO/VLA/ladder: `0`
- package install and D334 sidecar write: `0`
- commit/push: `0`

D358은 pure file/hash/schema audit이고 spatial/temporal judgment가 없으므로 Rerun을
의도적으로 생략한다. 원 byte와 canonical JSON이 authority다.

## 8. Session-progress rule

이 case는 20,736 finite recipe search와 unit/dtype/byte-layout/order/winding/digest
perturbation negative controls를 포함하므로 실제로 FAIL할 수 있는 perturbation
evaluation이다. validation-only 예외가 아니다.

## 9. Preregistration-time status

이 절을 작성한 시점에는 `prepare`와 sole audit invocation 전이었다. 실제 결과는
시간 순서를 보존해 이 아래에 append한다.

## 10. Actual execution result

### 10.1 Prepare

등록된 standalone core-PXR 환경으로 `--stage prepare`를 실행했다. 결과는
`pass=true`였고 다음 prerequisite가 모두 exact였다.

- `HEAD == origin/master == 161f6d9d185bb41eb29259349ee0fd897a3c6de8`
- harness SHA-256
  `e5fdc1fc98c85db1b2533ae45e080999681a9b88c59c593bf5cb8869ee87757a`
- 이 session 문서의 preregistration-time SHA-256
  `bca21d3404411c420a17741d41b1c80ce1f9453c0aa4f790424052355ced6d53`
- candidate grid `20,736`
- OpenUSD `[0,24,5]`, NumPy `1.26.0`, psutil `5.9.8`
- registered `pxr.Gf`, `pxr.Usd`, `pxr.UsdGeom` origins와 Sdf/Usd binary hash
- 모든 frozen input hash와 D334 sidecar inventory

Prepare artifact SHA-256은
`23e6d0e6c6fb1d963896b6fe6d7e46b97937da1b82c9d87bd2aa89182d93af4f`다.

### 10.2 Sole audit invocation

사전등록 명령을 정확히 한 번 실행했다. process exit code는 `0`, audit invocation
count는 `1`, watchdog/timeout은 없었다. evidence의 recipe search elapsed는
`90.19101224502083s`, completion phase marker는 `90.21373272209894s`였다.

Phase order는 다음과 같이 forward-only였다.

1. `audit_started`
2. `frozen_usd_streams_loaded`
3. `recipe_grid_started` (`20,736`)
4. `recipe_grid_finished` (`2/8`, coherent bundle false)
5. `authoritative_evidence_written`
6. `completion_ready`

## 11. Quantified provenance result

최종 verdict는
`D358_HASH_PROVENANCE_UNRESOLVED_FAIL_STOP`이다. exit `0`, prereg/current-input/env
재검사, 순방향 phase 6개, exact final inventory와 completion 도달을 함께 볼 때 등록된
프로그램은 정상 완주했다. completion의 `operational_pass=true`는 이 상태를 요약하는
필드이지만 harness에서 completion 도달 시 literal로 기록되므로 그 필드 하나만을 근거로
삼지 않는다. top-level `pass=false`는 동결된 provenance 질문을 해결하지 못했다는
뜻이다. 둘을 혼동하면 안 된다.

### 11.1 무엇이 재현됐는가

- frozen authored USD의 point/count/index 원 stream hash는 모두 exact였다.
- D354가 실제 authored stream에서 관측한 inner/outer 8-field bundle은 registered
  recipe로 전부 재현됐다. 해시가 같은 equivalent recipe가 4개이므로, 이것만으로
  signed-zero 처리나 authored Float32→Float64→Float32 중 하나를 유일한 provenance로
  역추론하지는 않는다.
- D354 raw full vertex stream SHA-256
  `522a4f0fe91a04bf54c5c8be6492748c7490fc557fa8c0867200d97332dfa9db`와
  raw inner paired-XZ
  `98ef77e6c5080e96f763eab04c48d4d6c06c9bc1a8b79995bd0fffa32618bbae`도
  재현됐다.
- NumPy 계산과 별도 Python tuple/dict/`struct.pack` 구현은 current authored/raw
  inner+outer vertex/triangle/patch/paired-XZ 및 full raw stream `17/17`에서 일치했다.
- wrong unit/dtype, big endian, Fortran order, reverse face order, flip winding,
  digest order FTV 음성 대조군은 `7/7` PASS했다.

### 11.2 무엇이 재현되지 않았는가

D351에 상수로 고정된 historical expected bundle 중 paired-XZ 두 필드만 등록된
recipe로 재현됐다. 나머지 여섯 필드는 `20,736`개 어느 recipe에도 없었다.

| field | D351 frozen expected | D354/current authored | expected reproduced |
|---|---|---|---|
| inner vertex | `13c65ee4...74d55` | `caa7d967...398f9` | no |
| outer vertex | `0d9f7f85...2772a` | `3e24cab3...6131` | no |
| inner triangle | `5644e9a6...b9e17` | `d90afe18...d2349` | no |
| outer triangle | `5644e9a6...b9e17` | `bd024ff9...ad241` | no |
| inner patch | `c927e8c6...3531b` | `7478ac18...5a877` | no |
| outer patch | `9b430c7d...6486` | `e0a8f4cc...c7361` | no |
| inner paired-XZ | `917b7154...bcaf9` | `917b7154...bcaf9` | yes |
| outer paired-XZ | `917b7154...bcaf9` | `917b7154...bcaf9` | yes |

따라서 `reproduced_expected_field_count=2/8`,
`all_expected_fields_reproduced=false`,
`coherent_eight_field_recipe_found=false`다. 미재현 6개에는 target-matching recipe가
없으므로 그 recipe의 독립 replay와 endian/layout 대조를 수행할 대상 자체가 없었다.
그래서 해당 8-field aggregate check도 false다. 일치한 paired-XZ 2개 row는 독립
replay와 little-endian-C/BE/Fortran 대조를 모두 PASS했으며, aggregate false를 계산
구현이나 음성 대조군 실패로 해석하면 안 된다.

### 11.3 Float roundtrip의 실제 크기

authored Float32 millimeter point와 runtime Float64 meter point를 다시 millimeter로
환산한 배열은 bit-exact가 아니었다.

- total components: `123,282`
- mismatched components: `58,506`
- mismatched vertices: `36,519 / 41,094`
- max absolute delta: `0.0000031862526839177008mm`
- mean absolute delta: `0.0000009393762372878338mm`

이는 매우 작은 수치 차이가 실제로 존재함을 보여 주지만, 그 차이를 포함한 등록
roundtrip recipe도 D351의 미재현 6개 expected hash를 만들지 못했다. 따라서 단순히
“meter/Float64 변환 때문”이라고 결론 내릴 수 없다.

## 12. Critical interpretation

이번 감사가 증명한 것은 다음 두 문장이다.

1. 현재 authoring USD와 현재 D354 canonicalization은 내부적으로 재현 가능하다.
2. D351에 적힌 historical expected vertex/triangle/patch hash 6개를 만든 provenance는
   이번에 사전등록한 dtype/unit/order/canonicalization family 안에서 복원되지 않았다.

이는 geometry가 바뀌었다는 증명도, D351 상수가 틀렸다는 증명도 아니다. 예를 들어
기록되지 않은 다른 source array, 전혀 다른 remap/serialization 알고리즘, 별도 전처리
산출물에서 상수가 왔을 가능성은 남아 있다. `20,736`은 사전등록된 유한 grid이지 가능한
모든 알고리즘의 exhaustive universe가 아니다. 원 생성 근거를 찾기 전에는 expected
hash를 현재값으로 교체하거나 binding gate를 완화하면 안 된다.

D354의 `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP`, `g0a_pass=false`는 그대로다.
현재 자세의 barrel-first 접촉, 실제 접촉력, 파지 가능성, target/IK repair 필요성은 모두
계속 미판정이다.

## 13. Scope and artifact closure

- standalone core-PXR preflight: PASS; D358 process의 forbidden Isaac/Kit/PhysX/GPU
  Python modules before/after `[]`
- `SimulationApp`/Kit/Isaac GUI/GPU/nvidia-smi/Warp case-local contract count: `0`
- q5 science, distance/contact query, new cap/rim classification: `0`
- controlled physics steps: `0`
- asset/dependency/gate/target/IK/path change: `0`
- D334 sidecar writes: `0`; before/after inventory exact
- Rerun: pure file/hash/schema audit이므로 사전등록대로 생략
- retry/overwrite/second invocation: `0`

위 scope-guard 숫자는 harness의 case-local contract literal이며 시스템 전체 process
telemetry가 아니다. 따라서 이 숫자 하나로 사용자가 별도로 켜 둔 persistent Isaac Full
GUI까지 꺼져 있었다고 주장하지 않는다. D358 자체의 무사용 판단은 등록 명령, import
목록, exact environment, phase/inventory, 그리고 harness static scope를 함께 근거로 한다.

Final artifact SHA-256:

- invocation: `3096f10465767fd1c6f2836eb4f8df0a610969cbf5ab606eb5ef2343db671c8b`
- evidence: `6c19cf6c3cd99b9567db65bf065afcb95872c4cfa6940c6584a97717638af3ff`
- automated report: `8693034d00541724a06a498139448581441d5b03229a9c8d5c9e83723beafe15`
- phase markers: `0bdb46459d2a626e0ae53f069e03302618eb4b71c218550029f54e35f51d5904`
- completion summary: `9ea631942cab32708cbc2f58e2b8351ad03dd2f45ff8c6f699caa44079e875f7`

## 14. Next authorization boundary

D358은 완료됐고 재실행하지 않는다. 다음 case는 자동 승인되지 않았다.

가장 좁은 증거 복구 후보는 D351 expected 6개 상수가 최초로 만들어진 source/commit/
generator를 읽기 전용으로 추적하는 forward-only historical provenance case다. 실제
PhysX jaw-close/contact-force/object-motion case는 별도 선택지이며, 실행하려면 새로운
승인·사전등록과 죠-원통 interface가 보이는 camera가 필요하다. 어느 쪽도 이 session에서
실행하지 않는다.
