# Session 2026-07-16 - D356 D355 PXR root-cause correction

## 1. What and why

D355의 실행 사실은 보존하되, 그 실패 원인을 `SimulationApp`이 유일한 PXR
접근 경로라는 설명으로 좁힌 것은 기존 D343/D345 반증과 충돌한다. 이 correction-only
case는 immutable D343/D345/D355 증거를 다시 교차해 원인 귀속과 다음 경로만 바로잡는다.
D355를 재실행하거나 기존 session/DECISIONS/ledger row를 수정하지 않는다.

`이번 case의 신규 변수: [d355_pxr_root_cause_counterevidence_interpretation]`

`new physical variables: []`

최종 판정: `D356_D355_PXR_ROOT_CAUSE_CORRECTED_NO_RERUN`.

## 2. Git and immutable inputs

- 감사 시작 시 `HEAD == origin/master ==
  161f6d9d185bb41eb29259349ee0fd897a3c6de8`, worktree clean이었다.
- D355 harness와 runtime exception SHA-256:
  - `b1fe5bf0f42c3d30a2b56d6809e17cfe4785eb7dcb610e2cf6fc05fb57c50d46`
  - `48bcad5c5740651f7aa8157616b64a639b79f67af5033e60dfe939da4bfdebde`
- D343 preregistration / summary SHA-256:
  - `fb8f9c292042001aeb05d9b693d910797bd4a214d9e01427ccd54b7e2c387ce8`
  - `880601aac768df38675603828258850aea796b6436a299c46f8cc489ed8b00da`
- D345 preregistration / worker-A SHA-256:
  - `9c31b8070d2051c00ebd6789facd6c8a59256cb9beefe8645a63ff41a277b6a3`
  - `99991b382bf881502dc73009877cd09a5617be8d3a5a5610a0d047f741756974`

## 3. Evidence reconciliation in execution order

1. D355 prepare는 Git, frozen input hash, `numpy==1.26.0`, `psutil==5.9.8`,
   `rerun-sdk==0.34.1`만 검사했다. PXR 경로, `from pxr import Usd`,
   `Usd.GetVersion()` preflight는 없었다.
2. D355는 invocation marker를 먼저 쓴 뒤 `_source_arrays()`의 첫
   `from pxr import Gf, Usd, UsdGeom`에서 `ModuleNotFoundError`로 멈췄다.
   USD/hash/q5/physics/Isaac 실행 수는 모두 0이다.
3. D343은 동일 `isaaclab` Python에 설치된
   `isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311`를
   `PYTHONPATH`에, 그 `bin`과 Conda lib를 `LD_LIBRARY_PATH`에 사전등록했다.
   결과는 `isaac_kit_started=false`, `standalone_pxr_only=true`, OpenUSD
   `[0,24,5]`, PASS였다.
4. D345도 같은 bundled-core-PXR 환경을 등록했고 두 독립 worker가 OpenUSD
   `[0,24,5]`, `runtime_environment_created=false`, PASS를 기록했다.
5. 따라서 plain Conda Python이 자동으로 `pxr`를 노출하지 않는다는 D355 관찰은
   맞지만, `SimulationApp`만이 유일한 해결이라는 추론은 거짓이다.

## 4. Corrected cause and preserved facts

직접 원인은 Isaac/RTX/PhysX/GPU 실패가 아니라 다음 두 사전등록 누락이다.

1. 이미 검증된 D343/D345 bundled standalone-core-PXR
   `PYTHONPATH`/`LD_LIBRARY_PATH` 상속 누락
2. audit invocation 전에 PXR import와 OpenUSD version을 확인하는 preflight 누락

D355의 invocation 1회, no-retry, exception, zero counts,
`D355_OFFLINE_INPUT_OR_OBSERVABILITY_FAIL_STOP`, provenance result `null`은 모두
그대로 유효하다. 이미 봉인된 D355 실행 중 즉석 경로 주입이나 Kit launch를 하지 않은
것도 옳았다. 수정되는 것은 과거 결과가 아니라 원인 설명과 다음 후보뿐이다.

## 5. Why no new experiment was run

이번 session은 새 로봇 측정이 아니라 잘못된 원인 귀속을 기존 immutable 반례로
정정한다. D355 동일 경로 재실행은 금지되어 있고 D343/D345가 이미 실패 가능했던
standalone-PXR 실행을 각각 완료했다. 같은 환경을 다시 띄우는 검사는 이 정정 결정을
바꾸지 못하므로 AGENTS.md의 불필요한 validation 금지에 따라 실행하지 않았다.
Isaac launch, PXR import 재실행, q5, physics, cap/rim, asset write는 모두 0이다.

## 6. Next authorized sequential case

사용자는 정정 뒤 시각화, provenance 재시도, 실제 PhysX 접촉 시험을 순서대로
진행하도록 승인했다. 다음 Active Case는 D357
`[d354_beginner_result_visualization_repair]`이다. D357은 D354의 frozen OPEN /
last-clear / first-overlap 상태를 같은 실제 Isaac 카메라로 표시하는 관찰성 전용
case이며, 거리·feature를 재판정하거나 physics step을 실행하지 않는다.

그 뒤 D358은 D343/D345 bundled-core-PXR 환경과 import/version preflight를 상속해
frozen D355 provenance 계약을 새 output에서 한 번 실행한다. Kit bootstrap은 reserve
alternative이고 새 OpenUSD package 설치는 필요하지 않다.

## 7. Sources

- `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d343/d343_usd_typed_float_readback_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d345/d345_worker_a.json`
- `sim_scripts/cyl34_top_view_d355_moving_jaw_patch_hash_provenance_audit.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d355/d355_runtime_exception.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d355/d355_postrun_operational_audit.json`
