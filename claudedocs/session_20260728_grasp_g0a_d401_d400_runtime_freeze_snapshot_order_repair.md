# 2026-07-28 — Grasp G0a D401 D400 runtime-freeze snapshot-order repair

## 1. What and why

Case:
`D401 [d400_runtime_freeze_snapshot_order_repair]`

Status:
`STATIC_ATTESTATION_PASS_RUNTIME_NOT_APPROVED`

이번 case의 신규 변수:

1. `git_baseline_repin_to_e9fa30088be7477ce5d6305aa5fdf68323e79adc`
2. `git_snapshot_capture_before_first_runtime_write_v1`

D400 attempt1은 SDF, Isaac, PhysX 또는 GPU 문제로 실패한 것이 아니다.
Controller가 Worker를 시작하기 전에 실행하는 Git 동결 검사에서 멈췄다.
원인은 두 가지였다.

1. D400 사전등록의 기대 Git 기준점은 이전 `4c88865...`였지만, 실제
   `HEAD`와 `origin/master`는 모두
   `e9fa30088be7477ce5d6305aa5fdf68323e79adc`였다.
2. D400 Controller가 Git 상태를 읽기 전에 자신의
   `d400_phase_markers.jsonl`을 먼저 썼다. 따라서 자신의 첫 출력 파일을
   예상 밖 untracked 파일로 판정했다.

D401은 이 두 control/provenance 결함만 고친다. D400의 SDF 입력 형상,
USD mutation allowlist, owner/property/cook 판정, watchdog, counter,
Rerun, scientific-authority 계약은 바꾸지 않는다.

## 2. D400 attempt1 frozen observation

불변 경로:

`claudedocs/runtime_logs/grasp_track/g0a_d400/attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight/`

원 JSON을 다시 확인한 결과:

- 최종 verdict:
  `D400_GRIPPER_LINK_SDF_RES256_PREFLIGHT_FAIL_STOP`
- failed stage: `runtime_freeze_manifest`
- `head_exact=false`
- `origin_master_exact=false`
- `head_equals_origin_master=true`
- `no_unexpected_dirty_paths=false`
- approval tuple, frozen repo inputs, installed primary-source hashes,
  D334 sidecar checks는 모두 PASS
- Worker spawn request: `false`
- Worker claim observed: `false`
- `actual_worker_invocations`: 원시 authority가 `null`
- runtime/scientific/physics verdict: `null`
- scope counters: `null`
- `g0a_pass=false`

`actual_worker_invocations`를 임의로 숫자 `0`으로 다시 쓰지 않는다. 다만
Controller program order상 spawn request와 claim이 모두 없으므로 Worker,
Isaac, PhysX에는 도달하지 않았다는 운영 사실은 확정된다.

D400 attempt1의 네 runtime evidence는 D401 사전등록에 SHA-256으로
고정했으며 수정·덮어쓰기·같은 경로 재실행을 금지했다.

## 3. Approved boundary

이번 사용자 승인은 다음만 허용했다.

- 새 D401 사전등록
- 새 forward-only Controller/Worker wrapper 작성
- AST/source 정적 검토와 실패 가능한 음성 대조군
- 새 four-hash proposed runtime tuple 작성
- 상태 문서 연속성 갱신

다음은 실행하지 않았다.

- D401 Controller runtime
- D400/D401 Worker
- Isaac Sim, Kit, PhysX, Warp, CUDA/GPU runtime
- derivative USD/collision asset
- physics step, public `forward()`, q5 sample, contact query, cylinder
- target/IK/path/pose/material/mass/actuator/solver/physics 설정 변경
- Rerun Viewer

## 4. Procedure in observable order

### Step 1 — Current Git and frozen failure evidence

- 실제 `HEAD`와 `origin/master`를 다시 읽어 둘 다
  `e9fa30088be7477ce5d6305aa5fdf68323e79adc`임을 확인했다.
- D400 attempt1 네 출력과 D400 controller/worker/preregistration/
  attestation/tuple을 다시 SHA-256 대조했다.
- 사용자 소유 untracked `codex`는 0 byte,
  SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  로 preserve-only 등록했고 수정·삭제·stage하지 않았다.

### Step 2 — Forward-only preregistration

새 경로:

`claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/`

사전등록은 신규 변수를 정확히 두 개로 제한하고, D400 attempt1과
science/physics 계약을 모두 동결했다. 과거 제안에서 “D401”로 부르던
non-product articulation contact positive control은 이번 explicit D401
repair가 번호를 차지했으므로 실행하지 않았고 자동 재번호도 하지 않았다.

### Step 3 — Thin wrapper implementation

새 D401 Controller와 Worker는 동결 D400 구현을 복사·수정하지 않는 얇은
wrapper다.

미래 Controller의 등록 순서는 다음과 같다.

1. `python -B`/`dont_write_bytecode` fail-closed 검사
2. tuple과 동결 파일 read-only 검증
3. 미래 runtime 경로가 아직 없는지 read-only 검사
4. Worker command와 nonce를 메모리에서 구성
5. `git status`, `HEAD`, `origin/master`, 모든 dirty regular file의
   SHA-256/byte size, 단조 시각을 메모리에 한 번 캡처
6. 첫 runtime write로 phase row 작성
7. 캡처된 immutable snapshot만으로 manifest 작성
8. 캡처된 dirty file을 다시 hash/stat하여 중간 변조가 없을 때만 PASS
9. 그 뒤에만 동결 D400의 package/GPU/process 및 Worker 경로로 진행

`python -B`는 wrapper import가 Git snapshot보다 먼저 `__pycache__`를
만드는 숨은 write를 차단하기 위해 필수다. Manifest 안에서는 phase write
뒤 `git status`, `HEAD`, `origin/master`를 다시 묻지 않는다. 동결 D400의
post-worker live Git integrity 검사는 그대로 유지한다.

미래 runtime artifact basename과 D400 내부 Rerun entity/environment
identifier는 동결 계약 보존을 위해 `d400_*`, `/d400/...`,
`D400_INVOCATION_SHA256`을 그대로 쓴다. 단, 폴더는 새 D401 forward-only
경로이므로 D400 attempt1을 덮어쓰지 않는다.

### Step 4 — Static and adversarial review

두 wrapper를 import/execute하지 않고 AST와 source만 검사했다.
검사 결과:

- AST parse: `2/2 PASS`
- forbidden direct top-level imports
  (Isaac/Kit/PhysX/pxr/Warp/CUDA/torch/Rerun): `0`
- controller/worker exact path-rebinding contract: PASS
- snapshot-before-first-write source order: PASS
- manifest post-write live Git re-query: `0`
- dirty-file hash/size recheck: present
- D400 attempt1 immutable hashes: exact
- D334 sidecar mutation: `0`
- failure-capable negative fixtures: `35/35 PASS`

독립 역검토에서 정적 attestation 전에 세 결함을 찾아 고쳤다.

1. Python bytecode가 pre-snapshot write가 될 수 있음
2. snapshot 뒤 dirty file content drift를 다시 검증하지 않음
3. snapshot completion timestamp가 모든 HEAD/origin read보다 먼저
   기록될 수 있음

최종 역검토는 남은 runtime-blocking static defect가 없다고 판정했다.
이것은 실행 성공 판정이 아니라, 별도 승인으로 실제 1회 runtime을
시도할 수 있는 정적 준비 완료 판정이다.

## 5. Exact artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| D401 preregistration | `c010578e7307c21e305a3db499fa25204297ceb08192233dbc42023bfd5de5c8` |
| D401 reviewed-script attestation | `8d44072ae389475b50c7be2f297108d35fadb0937089f29e564bf3f20b68c9bb` |
| D401 Controller | `2807353bb36f3309ed7592bdd3b24f4214ebde8b204ab3e253443f51bf63296e` |
| D401 Worker | `fc019d0d74bc868a5f2cac928824f5de875e05783472f288873f01342775673d` |
| **D401 proposed runtime tuple file** | **`7097134b350cf1641f2585c150cba45bc56ba0e9792d6549f0ae9c2f9e72cd2e`** |

Tuple field order and values:

1. `preregistration_sha256`
2. `reviewed_script_attestation_sha256`
3. `controller_script_sha256`
4. `worker_script_sha256`

각 값은 실제 파일 SHA-256과 일치한다.

Static-stage counters:

| Counter | Value |
|---|---:|
| Controller files created | 1 |
| Worker wrapper files created | 1 |
| Static attestations created | 1 |
| Proposed runtime tuples created | 1 |
| Actual Controller runtime invocations | 0 |
| Actual Worker invocations | 0 |
| Isaac/Kit/PhysX imports or launches | 0 |
| GPU runtime jobs | 0 |
| Derivative asset/USD writes | 0 |
| Physics steps / public forwards | 0 / 0 |
| q5 samples / contact queries / cylinder writes | 0 / 0 / 0 |

## 6. Verdict and authorization boundary

Operational verdict:

`D401_RUNTIME_FREEZE_SNAPSHOT_ORDER_REPAIR_STATIC_ATTESTATION_PASS_RUNTIME_NOT_APPROVED`

Scientific/physics verdict:
`null`

`g0a_pass=false`

쉽게 말하면, D400에서 Isaac이 실패한 것이 아니라 Isaac을 켜기 전의
“작업 파일과 Git 기준점 확인 절차”가 잘못되어 멈췄고, D401은 그 확인
절차만 정적으로 수리했다. SDF가 실제 RoArm articulation에서 load/cook
되는지는 아직 한 번도 다시 측정하지 않았다.

다음 실제 Controller/Worker 1회는 새 별도 승인이 필요하며, 승인 문장에
tuple file SHA-256
`7097134b350cf1641f2585c150cba45bc56ba0e9792d6549f0ae9c2f9e72cd2e`
가 정확히 들어가야 한다.

이 tuple은 `HEAD == origin/master ==
e9fa30088be7477ce5d6305aa5fdf68323e79adc`이고 위 네 입력 파일이 그대로인
동안만 유효하다. 실제 runtime 전에 commit/push로 HEAD가 바뀌면 이
tuple로 실행하지 않고 새 forward-only baseline/tuple을 다시 발급한다.

## 7. Session-progress-rule justification

이번 control hardening은 임의의 예방 작업이 아니라 D400 actual
Controller attempt1에서 관측된 fail-stop에 대한 반응형 수리다. 사용자
승인이 offline/static-only로 Worker/Isaac/PhysX 실행을 명시적으로
금지했으므로 이 세션에서는 새 runtime experiment를 실행하지 않았다.
35개 음성 대조군은 정적 계약이 실제로 잘못된 입력을 거부할 수 있음을
검증했지만, SDF/physics 실험을 대신하지 않는다.

## 8. Sources

- `claudedocs/runtime_logs/grasp_track/g0a_d400/attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight/d400_runtime_freeze_manifest.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d400/attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight/d400_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d400/attempt1_gripper_link_sdf_res256_live_cook_articulation_preflight/d400_completion_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d401_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d401_reviewed_script_attestation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d401/attempt1_d400_runtime_freeze_snapshot_order_repair/d401_proposed_runtime_hash_tuple.json`
- `sim_scripts/cyl34_top_view_d401_d400_runtime_freeze_snapshot_order_repair_controller.py`
- `sim_scripts/cyl34_top_view_d401_d400_runtime_freeze_snapshot_order_repair_worker.py`
- `START_HERE.md`
- `claudedocs/DECISIONS.md#d401`
- `claudedocs/EXPERIMENT_LEDGER.md`
