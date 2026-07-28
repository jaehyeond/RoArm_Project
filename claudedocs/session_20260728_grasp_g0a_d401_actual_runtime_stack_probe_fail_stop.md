# 2026-07-28 — Grasp G0a D401 actual runtime stack-probe fail-stop

## 1. 무엇을 왜 실행했는가

Case:
`D401 [d400_runtime_freeze_snapshot_order_repair]`

승인된 tuple SHA-256:
`7097134b350cf1641f2585c150cba45bc56ba0e9792d6549f0ae9c2f9e72cd2e`

이번 case의 신규 변수는 D401 정적 단계에서 등록한 다음 두 개뿐이다.

1. `git_baseline_repin_to_e9fa30088be7477ce5d6305aa5fdf68323e79adc`
2. `git_snapshot_capture_before_first_runtime_write_v1`

D400 attempt1은 Controller가 자기 첫 phase 파일을 쓰고 나서 Git 상태를
읽었기 때문에 Worker 시작 전에 멈췄다. D401은 Git snapshot을 첫 파일
작성 전에 메모리에 끝까지 캡처하도록 수리했다. 이번 실행의 목적은 그
수리가 실제 1회 실행에서도 통과하는지 확인한 뒤, 동결 D400 Worker가
moving jaw인 `gripper_link`의 A64 충돌체를 SDF resolution 256 입력으로
작성·load/cook/readback할 수 있는지 확인하는 것이었다.

사용자 승인 경계는 다음과 같았다.

- D401 Controller `python -B` 1회
- inherited Worker 1회, retry 0
- Isaac/PhysX는 asset load/cook/readback에만 사용
- controlled physics step, public `forward()`, q5, contact, cylinder는 0
- target/IK/path와 material/mass/actuator/physics 설정 변경 0

## 2. 실행 전 재검증

실제 Git 기준점:

- `HEAD = e9fa30088be7477ce5d6305aa5fdf68323e79adc`
- `origin/master = e9fa30088be7477ce5d6305aa5fdf68323e79adc`
- commit subject: `D400 gpu승인전`

승인 tuple의 네 입력은 실행 직전 실제 파일과 다시 일치했다.

| 입력 | SHA-256 |
|---|---|
| D401 preregistration | `c010578e7307c21e305a3db499fa25204297ceb08192233dbc42023bfd5de5c8` |
| reviewed-script attestation | `8d44072ae389475b50c7be2f297108d35fadb0937089f29e564bf3f20b68c9bb` |
| Controller | `2807353bb36f3309ed7592bdd3b24f4214ebde8b204ab3e253443f51bf63296e` |
| Worker | `fc019d0d74bc868a5f2cac928824f5de875e05783472f288873f01342775673d` |

호스트 GPU 사전검사는 다음을 확인했다.

- NVIDIA GeForce RTX 4090 Laptop GPU
- driver `580.173.02`
- compute capability `8.9`
- total VRAM `16376MiB`
- free VRAM `15887MiB`
- 기존 D400/D401 충돌 프로세스 `0`

기본 sandbox에서는 GPU device가 숨겨져 `nvidia-smi`가 NVML을 열 수
없었다. 이것은 Isaac 실행 실패가 아니며, 승인된 실제 실행은 GPU device가
보이는 호스트 경계에서 정확히 한 번 수행했다.

## 3. 실제 명령과 관찰 순서

실행 명령:

```bash
/home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d401_d400_runtime_freeze_snapshot_order_repair_controller.py \
  --approved-tuple-sha256 \
  7097134b350cf1641f2585c150cba45bc56ba0e9792d6549f0ae9c2f9e72cd2e
```

실행 횟수는 정확히 `1`, 자동 또는 수동 retry는 `0`이다.

관찰된 program order:

1. Controller가 첫 출력 전에 Git snapshot을 메모리에 캡처했다.
2. D401 runtime-freeze manifest가 전체 PASS했다.
3. package/GPU/existing-process gate가 PASS했다.
4. runtime 음성 대조군 `18/18`이 PASS했다.
5. Worker를 정확히 한 번 spawn했다.
6. Worker가 headless Isaac `SimulationApp`을 시작했다.
7. Isaac은 `cuda:0`과 RTX 4090을 정상 인식하고 약
   `7.101854029s` 뒤 launch-end marker를 남겼다.
8. Worker가 실행 stack의 PhysX extension 버전을 확인했다.
9. extension ID, extension root, native plugin hash는 맞았지만 버전 값만
   `null`로 기록되어 fail-stop했다.
10. 아직 stage를 만들기 전이었으므로 attach/detach 없이 raw summary와
    pre-close sentinel을 쓰고 앱 정리를 시작했다.
11. Controller는 Worker OS return code가 `0`이어도 내부
    `worker_protocol_pass=false`를 권위로 삼아 전체 결과를 FAIL로 닫았다.

Phase marker는 총 `15`개이며 registered order와 단조 증가 시각 계약을
통과했다.

## 4. D401 수리 자체의 결과

`d400_runtime_freeze_manifest.json`은 `pass=true`다. 특히 다음이 모두
참이다.

- `snapshot_captured_before_first_phase_write`
- `captured_dirty_regular_file_observations_complete`
- `captured_dirty_regular_files_still_exact_before_manifest`
- `head_exact`
- `origin_master_exact`
- `head_equals_origin_master`
- `no_unexpected_dirty_paths`
- `phase_path_absent_from_captured_status`
- frozen repo/installed-primary hashes exact
- D334 sidecar untouched
- approved tuple exact

따라서 D401이 고치기로 한 “Git snapshot 순서” 결함은 실제 실행에서도
PASS했다.

## 5. Isaac Sim은 실패했는가

아니다. 이번 증거에서 Isaac Sim 자체는 정상 기동했다.

Kit log는 다음을 기록했다.

- device: `cuda:0`
- experience:
  `isaaclab/apps/isaaclab.python.headless.kit`
- driver: `580.173.02`
- active GPU: `NVIDIA GeForce RTX 4090 Laptop`
- GPU memory: `16376MB`
- `ERROR`/`FATAL`: `0`
- 등록된 subject warning: `0`

CPU powersave, IOMMU, Intel iGPU skip 같은 일반 warning은 있었지만 이번
fail-stop의 직접 원인이 아니다. watchdog timeout, SIGTERM, SIGKILL,
process-group residue, GPU Worker PID residue도 모두 없었다.

## 6. 최초 실패의 정확한 원인

### 6.1 실행에서 관찰된 값

Worker runtime-stack probe:

| 검사 | 결과 |
|---|---|
| Isaac Sim distribution `5.1.0.0` | PASS |
| Isaac Lab distribution `2.3.0` | PASS |
| PhysX extension ID resolved | PASS |
| observed extension ID | `omni.physx-107.3.26` |
| active extension root exact | PASS |
| native plugin SHA exact | PASS |
| observed `package.version` | `null` |
| expected extension version `107.3.26` | FAIL |

설치된 native plugin SHA는 등록값과 같은
`03fbf17e6f0dc3f9006c8c00aa0ca572a72fd69498874df6dd900dac726c9909`
였다. 설치 extension의 `config/extension.toml`에도
`version = "107.3.26"`이 실제로 존재한다.

### 6.2 코드 결함

동결 D400 Worker는 다음 논리를 사용한다.

```python
physx_extension = manager.get_extension_dict(physx_extension_id)
physx_extension_version = (
    physx_extension.get("package", {}).get("version")
    if isinstance(physx_extension, dict)
    else None
)
```

그러나 설치된 Kit `107.3.3` Python API는
`get_extension_dict()`의 반환형을
`carb.dictionary._dictionary.Item`으로 명시한다. 즉 일반 Python
`dict`가 아니다. NVIDIA Kit 107.3 공식 예제도 반환값에서
`data["package"]`로 직접 접근하거나 `data.get_dict()`로 Python dict로
변환한다.

따라서 이번 최초 실패는 다음으로 분류한다.

`D401_D400_RUNTIME_STACK_VERSION_PROBE_HARNESS_TYPE_CONTRACT_FAIL_STOP`

이 이름은 설명용 operational label이다. 원 completion JSON의 canonical
verdict는 동결된
`D400_GRIPPER_LINK_SDF_RES256_PREFLIGHT_FAIL_STOP`이다.

핵심은 “PhysX 107.3.26이 설치되지 않았다”가 아니라 “검증기가 정상
`carb.dictionary.Item`을 일반 `dict`가 아니라고 버리고 버전을
`null`로 만들었다”는 것이다.

## 7. 별도로 발견된 잠재 control 결함

이번 최초 fail-stop과 독립적으로, 다음 runtime 전에 고쳐야 할 두 번째
false-fail 경로가 확인됐다.

1. Worker 메모리 안의 counter mapping은 등록 순서를 지켜
   `keys_exact_in_registered_order=true`였다.
2. 하지만 Worker JSON writer는 `sort_keys=True`로 key를 알파벳순 저장한다.
3. Supervisor는 JSON을 다시 읽은 뒤 물리적인 key iteration 순서가 등록
   순서와 같은지 비교한다.
4. 그래서 supervisor의 `exact_36_keys_in_order=false`가 됐다.

JSON object key의 저장 순서는 counter schema의 의미가 아니다. 다음
repair에서는 exact key set, exact value, registered-order projection을
검사해야 하며, writer의 정렬 부작용을 실행 의미로 오해해서는 안 된다.

또한 Worker process return code는 `0`이지만 내부 protocol은 false였다.
이번 Controller는 raw summary와 pre-close sentinel을 읽고 올바르게
전체 FAIL로 판정했다. 이후에도 OS return code 단독을 성공 권위로 쓰지
않는다.

## 8. 도달한 것과 도달하지 못한 것

도달한 항목:

- D401 snapshot-order runtime manifest PASS
- GPU/process gate PASS
- runtime 음성 대조군 `18/18 PASS`
- Worker claim/command/nonce/parent authority PASS
- Worker invocation `1`, retry `0`
- Isaac `SimulationApp` launch `1`
- runtime stack provenance probe

도달하지 못한 항목:

- derivative SDF USD/collision asset materialization
- live USD stage
- PhysX stage attach/detach
- SDF API/attribute writes
- global cook queue update pumps
- PhysX property query
- rigid-owner evidence
- mass/COM/inertia readback
- Rerun RRD/RBL/1920x1080 board/manual inspection

정확한 scope counters:

| Counter | Value |
|---|---:|
| actual Worker invocations | 1 |
| automatic retries | 0 |
| SimulationApp launches | 1 |
| derivative asset materializations | 0 |
| SDF API/attribute writes | 0 |
| PhysX stage attach/detach | 0 / 0 |
| PhysX property queries | 0 |
| SimulationContext constructions/resets | 0 / 0 |
| controlled physics steps/public forwards | 0 / 0 |
| q5 commands/samples | 0 / 0 |
| contact queries/cylinder writes | 0 / 0 |
| target/IK/path changes | 0 |
| source geometry/link5 representation changes | 0 / 0 |

기술 gate가 geometry 생성 전에 실패했기 때문에 Rerun과 화면은 생성하지
않았다. 이는 “그림을 만들다 실패”한 것이 아니라, D400 계약이 기술 PASS
뒤에만 Rerun을 허용하기 때문이다. 이번에는 시각화할 SDF candidate 자체가
없다.

## 9. 결과와 과학적 경계

Canonical completion:

- verdict:
  `D400_GRIPPER_LINK_SDF_RES256_PREFLIGHT_FAIL_STOP`
- `technical_pass=false`
- `runtime_preflight_pass=false`
- `observability_pass=false`
- `scientific_or_physics_verdict=null`
- `g0a_pass=false`

이 실행은 SDF가 RoArm articulation에서 작동한다거나 실패한다는 증거가
아니다. SDF asset 작성 함수에 들어가기 전에 멈췄기 때문이다. 원통을
만들거나, 움직이거나, 접촉시키거나, 잡는 시험도 하지 않았다. D362가
여전히 마지막 실제 physics run이다.

D401 attempt1 경로는 불변으로 보존하고 같은 경로에서 재실행하거나
덮어쓰지 않는다.

## 10. 다음 미승인 최소 단계

다음 후보:

`D402 [d401_runtime_stack_item_and_counter_order_authority_repair]`

offline/static-only 신규 변수 두 개:

1. `carb_dictionary_item_compatible_extension_version_accessor_v1`
2. `serialized_counter_registered_projection_authority_v1`

범위:

- NVIDIA Kit 107.3 공식 API와 설치 API에 맞는 version read 수리
- serialization/reload 뒤 exact key set/value/registered projection 수리
- AST/static/negative-control 검증
- 새 forward-only Controller/Worker와 새 tuple 작성
- Controller runtime, Worker, Isaac/Kit/PhysX 실행 0

새 tuple SHA를 보고한 뒤 actual one-worker runtime은 다시 별도 승인을
받아야 한다.

## 11. Session-progress-rule justification

이번 세션은 승인된 one-worker runtime을 실제로 1회 실행했고,
runtime-stack probe가 실패할 수 있는 fail-capable evaluation이었다.
D401 control hardening은 D400 attempt1에서 관찰된 실제 freeze 실패에 대한
reactive repair였다. 실패 뒤에는 같은 경로 재시도나 science 변수 변경을
하지 않았다.

## 12. NVIDIA 공식 자료와 로컬 근거

설치 stack:

- Isaac Sim `5.1.0.0`
- Isaac Lab `2.3.0`
- Kit `107.3.3`
- omni.physx `107.3.26`
- NVIDIA driver `580.173.02`

NVIDIA primary sources:

- Omniverse Kit 107.3.0, **Extensions in-depth**:
  <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.3.0/guide/extensions_advanced.html>
  — `get_extension_dict`, `data["package"]`, `data.get_dict()` 사용 예제.
  설치 Kit은 patch `107.3.3`이며 공개 문서는 같은 `107.3` 계열이다.
- Omniverse Kit 107.3.0, **ExtensionManager Python API**:
  <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.3.0/omni.ext/omni.ext.ExtensionManager.html>
  — `get_extension_dict(...) -> carb.dictionary._dictionary.Item`.
- Omniverse Kit, **carb.dictionary.Item**:
  <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/carb.dictionary/carb.dictionary.Item.html>
  — `get_dict()`가 subtree를 Python object로 직렬화함.

로컬 근거:

- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d400_runtime_freeze_manifest.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d400_phase_markers.jsonl`
- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d400_kit_log.txt`
- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d400_worker_raw_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d400_worker_preclose_sentinel.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d400_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d400_completion_summary.json`
- `sim_scripts/cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_worker.py:397-423`
- `sim_scripts/cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_worker.py:381-392`
- `sim_scripts/cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_preflight.py:986-1040`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/kit/launcher.toml:5`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/kit/kernel/py/omni/ext/_extensions.pyi:116`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/kit/kernel/config/docs/python_api.md:145`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.gui.menu/isaacsim/gui/menu/help_menu.py:121-128`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/config/extension.toml:1-6`

## 13. Exact runtime artifact hashes

| Artifact | SHA-256 |
|---|---|
| runtime freeze manifest | `89d94b119f0018cc20849a786a7cbb89a07a0ac0bb3d68566f6ea2a961bdb732` |
| Worker invocation | `387249746ce8117a21655955834f8e9acac350ae863b983225bb4babbe20937b` |
| Kit log | `7acfd02ecbfea1301b8d4b8bdfc5f5d5fdbbaf9e8072e74aa41afbdb800470ae` |
| Worker raw summary | `f07fa7b56ce9ee6e992fff9e9e6c6f2602510630316dfa14e37ff2405693c2e4` |
| Worker pre-close sentinel | `8f674cc27cf424dd3f087a034ebf1e44ef941100c925db13fb702a56a13085f6` |
| Worker supervisor | `f9c2bda41b9ded5f06f3ad65988f3b904a00c47fbdbfc531fd22a020955bd160` |
| completion summary | `394ab4318d400b960b5d5e5c71fddeb55448329054a5fe7569d6162f5fb6068c` |
