# D361 — contact-point total-capacity and durable prefix-trace repair

Date: 2026-07-16 KST

Case: `g0a_d361`

Status at preregistration: `APPROVED_OFFLINE_CONTROL_REPAIR_IN_PROGRESS`

이번 case의 신규 변수:

1. `version_aligned_total_contact_point_capacity_budget`
2. `durable_framed_step_prefix_protocol`

## 1. 무엇을 왜 고치는가

D360은 실제 OPEN baseline 200 step과 closure 43개 완료 row까지 진행했지만, 상속한
ContactSensor의 상세 접촉점 총 capacity가 16개뿐이었다. 실제 접촉점 수가 16을 넘은
직후 contact-position aggregation의 CUDA index 범위 오류가 발생했다. 더구나 수치 row와
event body/value를 실행 끝에서 한꺼번에 파일로 쓰도록 구현했기 때문에, 이미 계산된
243개 row도 프로세스와 함께 사라졌다.

D361은 물리 결과를 다시 측정하는 case가 아니다. 다음 두 제어 결함만 고친다.

1. 현재 동결 장면의 실제 collision-shape 계보와 설치 PhysX 버전에 맞춰 상세 접촉점
   **총 capacity**를 산정하고, 미래 실행이 그 값을 정확히 상속하도록 machine-readable
   contract를 만든다.
2. 각 physics step의 시작과 완성된 body/value observation을 실행 중에 append-only
   JSONL로 즉시 `fsync`하여, 다음 실패가 나더라도 마지막 완성 step과 실행 중이던
   step을 구분할 수 있게 한다.

## 2. 사전등록 시점의 Git·입력 경계

- `HEAD == origin/master == e7ed71ca80768df9037c16e53a12d3c032af3d5d`
- boot 시 worktree: clean
- D360 session SHA-256:
  `97ef49eb31a754be4f12ea0c5f961ddfede4d19656df7147e3f10a63b21f9291`
- D360 harness SHA-256:
  `86bd2af855effb3bc31f067fd6cc7a4cb7088c422da3ae828d48f4e31d92fd5a`
- D360 runtime prerequisites SHA-256:
  `bfd20dc2ab678d9929c0dd8f54323f8f4b17707c6c89d099b1374b814c467dae`
- Installed IsaacLab ContactSensor config SHA-256:
  `adb530a2d26ec0ca21160a20c2491c921267764915c3b29108a4ad1bd88171f8`
- Installed IsaacLab ContactSensor implementation SHA-256:
  `c2b039eb46d55416a8699d82b2385abae563c3c4ab4404ad08fa68310ffa6c64`
- Installed `omni.physics.tensors` API SHA-256:
  `5dd16f8a37eccc94ac82338d6c1127e785cccf761cae1f6f18ef03d55b0f325f`
- Frozen composed physics USD SHA-256:
  `043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503`

`START_HERE.md`의 preregistration 전 Git 절은 D359/D360 편집 전 기준
`d4671d4...`와 dirty 상태를 적고 있어 현재 Git과 stale하다. D361은 실제 Git 명령의
`e7ed71c...` clean 결과를 권위로 사용한다.

### Prepare preflight attempt 1 — output 생성 전 fail-stop

첫 prepare 명령은 output directory, invocation marker, Isaac/PhysX/q5 실행을 만들기 전에
`worktree_scope_only_d361=false`로 중단했다. 원인은 `git status --short`의 첫 tracked row
` M START_HERE.md` 앞 공백을 `_git().strip()`이 제거하여 scope parser가 경로 첫 글자를
잃은 제어코드 결함이었다. 과학·capacity·prefix 평가가 아니며 본 failure-injection 1회는
시작되지 않았다. `_git()` 반환만 `rstrip()`으로 좁게 고쳐 porcelain XY 선행 열을 보존한
뒤, output 부재를 다시 확인하고 prepare preflight를 재시도한다.

## 3. total contact-point capacity의 근거

### 3.1 IsaacLab에서 이 값이 뜻하는 것

설치된 `ContactSensorCfg.max_contact_data_count_per_prim` 문서는 이 값이 filter마다의
개수가 아니라 모든 environment와 sensor body를 합친 총 상세 접촉점 한도라고 명시한다.
실제 구현은 다음 값을 PhysX tensor view에 전달한다.

```text
total capacity = max_contact_data_count_per_prim
                 × number of sensor bodies
                 × number of environments
```

D360은 sensor body 1개(`Sponge`)와 environment 1개이므로, 설정값과 실제 총 capacity가
같다. 네 filter는 이 수를 네 배로 늘려 주지 않는다.

Local sources:

- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py:26-35`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py:282-288`
- `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py:380-405`

### 3.2 동결 장면의 shape-pair 수

미래 실행 전 반드시 다시 exact 검증할 동결 count는 다음과 같다.

| 쪽 | collision shape 수 | 근거 |
|---|---:|---|
| sensor cylinder | 1 | `CylinderCfg`가 한 `Cylinder` geometry prim을 만들고 그 한 mesh prim에 collision을 적용 |
| TapTable filter | 1 | D360 runtime prerequisite의 enabled `Cube` collider 1개 |
| link4 filter | 1 | frozen composed physics USD의 enabled authored collision prim 1개 |
| link5 filter | 64 | D360 corrected live binding |
| moving `gripper_link` filter | 64 | D360 corrected live binding |

따라서 sensor/filter collision geometry pair의 최대 동시 후보 수는
`1 × (1 + 1 + 64 + 64) = 130`이다.

설치된 PhysX plugin binary에는 null-terminated `5.6.1` 문자열이 byte offset
`9,753,628`에 정확히 한 번 존재한다. 이와 정확히 맞는 NVIDIA-Omniverse 공식 source tag
`107.3-omni-and-physx-5.6.1`의 `PxContactBuffer.h`는 geometry pair 하나의
`MAX_CONTACTS = 256`을 선언한다. 따라서 다른 구버전의 64 상수를 가져오지 않고 이
version-aligned pair envelope를 사용한다.

- Exact official source:
  `https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/107.3-omni-and-physx-5.6.1/physx/include/geomutils/PxContactBuffer.h`
- Installed plugin:
  `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/bin/libomni.physx.plugin.so`
  SHA-256 `03fbf17e6f0dc3f9006c8c00aa0ca572a72fd69498874df6dd900dac726c9909`

사전등록 capacity는 임의의 반올림이나 filter-count 곱이 아니라 다음 exact product다.

```text
sensor shapes                         = 1
enabled filter shapes                 = 1 + 1 + 64 + 64 = 130
PhysX 5.6 contacts per geometry pair  = 256
registered total capacity             = 1 × 130 × 256 = 33,280
max_contact_data_count_per_prim       = 33,280  (1 env × 1 sensor body)
```

`omni.physics.tensors`가 공개적으로 만드는 force 1 + point 3 + normal 3 + separation 1의
Float32 상세 배열은 contact당 32 bytes다. 따라서 33,280점의 명시적 frontend payload는
`1,064,960 bytes = 1.015625 MiB`이고, 4 filter의 count/start UInt32 배열 32 bytes가
별도로 붙는다. Backend 내부 메모리는 이 계산에 포함하지 않으며 미래 runtime telemetry로
따로 관찰한다.

### 3.3 이 계산이 증명하지 않는 것

D361은 이 capacity가 실제 future PhysX run에서 충분했다는 것을 증명하지 않는다.
Isaac/PhysX를 실행하지 않기 때문이다. D361이 증명할 수 있는 것은 다음뿐이다.

- 숫자 33,280이 현재 version·shape inventory에서 기계적으로 재계산된다.
- inventory가 달라지면 실행 전에 fail-stop한다.
- old 16 및 filter 수를 단순 곱한 64가 거부된다.
- visible tensor memory 산술이 독립 재계산과 일치한다.

실제 overflow warning 부재, 실제 사용 contact count, q5 접촉/운동 값은 후속 별도 승인
physics run에서만 판단한다.

## 4. durable prefix 프로토콜

미래 worker가 sensor/filter/capacity contract를 확인한 직후, 첫 controlled physics step
전에 새 prefix 파일을 다음 flag로 한 번만 만든다.

```text
O_WRONLY | O_CREAT | O_EXCL | O_APPEND | O_CLOEXEC
```

같은 경로가 이미 있으면 resume/overwrite하지 않고 fail-stop한다. JSONL record kind는
`header`, `step_begin`, `step_observation`, `seal` 네 가지다.

각 record는 canonical JSON core의 SHA-256과 직전 record SHA-256을 포함한다. 매
append는 short-write loop 뒤 file `fsync`, 새 read-only descriptor의 exact `pread`
재검증을 수행한다. 첫 header 뒤에는 parent directory도 `fsync`한다.

physics step별 순서는 다음으로 고정한다.

1. `step_begin` append + `fsync`
2. physics step 시도
3. 기존 D360 전체 `_state_row()`와 contact-point count diagnostic 취득
4. 현재/직전 완료 row로 instantaneous 및 같은-phase two-step event 계산
5. 전체 state, body별 force/contact point, event body/value를 한
   `step_observation` payload에 함께 append + `fsync` + exact reread
6. 그 뒤에만 in-memory list 추가나 optional image/RRD 작업 수행

정상 horizon 또는 합법적 baseline stop에서는 full JSON/CSV/RRD보다 먼저 `seal`을
append/fsync한다. crash로 마지막 line이 반만 남으면 원본은 truncate/수정하지 않는다.
감사기는 newline·JSON·sequence·previous hash·self hash가 처음부터 연속으로 맞는
prefix까지만 신뢰하고, trailing bytes와 unmatched `step_begin`을 별도 보고한다. 해시가
모두 다시 계산되어 맞더라도 header의 sensor/filter path/index, 전체 D360 state-row field,
force vector/norm, 현재·직전 row에서 재계산한 event body/value, 누적 high-water mark,
header에 사전등록된 seal reason/count가 다르면 semantic FAIL이다.

## 5. D361 실행·검증 순서

1. 새 output `claudedocs/runtime_logs/grasp_track/g0a_d361/`를 exclusive create한다.
2. Git base, frozen hashes, installed source hashes, D334 sidecar, forbidden module 부재를
   preflight한다.
3. capacity budget을 두 독립 산식으로 재계산하고 old-16/filter×16 및 shape-count
   perturbation을 음성 대조군으로 실행한다.
4. prefix writer/verifier를 synthetic row로 정상 실행한다.
5. child-process crash/partial-tail/tamper/reorder/delete/duplicate/reconciliation mismatch
   failure injection을 실행해 마지막 유효 prefix와 body/value 보존을 검증한다. 별도로
   해시를 다시 유효하게 만든 wrong-header/wrong-event/premature-seal/truncated-state
   음성 대조군도 semantic verifier가 거부해야 한다.
6. 모든 결과를 새 D361 경로에만 쓰고, 소스/결과 inventory와 SHA-256을 고정한다.

Prepare와 run은 둘 다 다음 exact interpreter로만 허용한다. Base conda Python은
`isaacsim` distribution metadata가 없으므로 invocation 뒤 늦게 실패할 수 있어 금지한다.

```text
/home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d361_contact_point_capacity_and_prefix_trace_repair.py --stage prepare
/home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d361_contact_point_capacity_and_prefix_trace_repair.py --stage run
```

Capacity/PXR core-schema inventory는 invocation marker 전에 prepare에서 검증하고 payload를
동결한다. Run에서는 모든 frozen hash와 exact output inventory를 다시 확인한 뒤 marker를
쓴다. Failure-injection 각 test 결과는 aggregate 판단 전에 별도 append+fsync JSONL에 즉시
남긴다. 중간 예외가 나면 성공을 가장하지 않고 forward-only exception artifact를 남긴다.

이 offline failure injection은 실제 과학 데이터를 생성하지 않으면서 결과가 FAIL할 수
있는 perturbation evaluation이므로 AGENTS.md Session progress rule을 충족한다.

## 6. 명시적 금지·동결 범위

- Isaac Sim/Kit/IsaacLab application launch 금지
- q5 명령·sample·과학 재실행 금지
- PhysX simulation/physics step/contact query 금지
- 새 접촉 이미지·영상·RRD/RBL 생성 금지
- target/IK/path 및 초기 q0-q5/object state 변경 금지
- asset/decomposition/gate/tolerance/material/mass/actuator/solver/physics/renderer/dependency 변경 금지
- cap/rim/barrel, 접촉 body, grasp/hold/lift/G0a 과학 판정 금지
- D360 폴더 수정·재실행·finalize 금지
- user-owned `claudedocs/lab_meeting/20260715/d334_collision_table/` 수정 금지
- commit/push 금지

순수 파일/schema/control audit이므로 Rerun Observability Completion Contract의 spatial 또는
temporal 판단 대상이 없다. D361에서 RRD/RBL을 만들지 않는 것이 사전등록된 정당화다.

## 7. 결과

### 7.1 실행 식별과 최종 verdict

- 성공 prepare preregistration SHA-256:
  `87250fc30146887dd4c372092bed95a9491c67e7812426834ee37467c7c6de76`
- prepare SHA-256:
  `b0ade378ab1b673b2dceca18141b91fdcb451b0706b06c6c4f095b3a67022c72`
- frozen harness SHA-256:
  `6f41dee2cbe5ce651ef09e0f540a23c008d83f5e7fb6407ec152dd2e972cb608`
- offline invocation marker SHA-256:
  `f48084086398f6e4d2fc888fb294f764f0003a311f23fff8e1dbee07efd3f867`
- 실행은 등록한 exact `isaaclab/bin/python`으로 1회였고 retry는 없었다.
- 최종 verdict:
  `D361_CONTACT_CAPACITY_AND_PREFIX_TRACE_REPAIR_PASS_NO_PHYSICS`

첫 prepare attempt는 앞서 기록한 status-parser 결함으로 output 생성 전에 멈췄다. 좁은
수정과 두 독립 정적 재검토 뒤 prepare가 PASS했고, 그 뒤 failure-injection invocation은
정확히 한 번만 실행됐다.

### 7.2 capacity 결과

원 JSON의 actual collision-shape inventory는 다음과 같다.

| 항목 | count |
|---|---:|
| sensor cylinder | 1 |
| support table | 1 |
| link4 | 1 |
| link5 | 64 |
| moving gripper_link | 64 |

설치 plugin의 null-terminated PhysX `5.6.1` offset, cylinder source contract, D360
sensor body/environment/filter contract, link4 core-PXR inventory가 모두 일치했다. 따라서
두 독립 산식은 모두 다음 exact total을 냈다.

```text
direct       = 1 × (1 + 1 + 64 + 64) × 256 = 33,280
per-filter   = table 256 + link4 256 + link5 16,384 + gripper 16,384
             = 33,280
```

Capacity check `13/13`과 음성 대조군 `9/9`가 PASS했다. 명시적 frontend 상세 배열은
contact당 32 bytes, 합계 `1,064,960 bytes = 1.015625 MiB`; count/start 배열은 별도
32 bytes다. Backend 내부 할당과 실제 최고 사용량은 포함하지 않는다.

중요하게도 `runtime_sufficiency=null`이다. D361은 33,280을 설치 version과 동결 shape
inventory가 요구하는 보수적 reported-contact allocation envelope로 근거화했지만, 실제
PhysX에서 overflow warning이 사라지는지 또는 실제 count가 얼마인지는 실행하지 않았다.

Source:

- `claudedocs/runtime_logs/grasp_track/g0a_d361/d361_contact_capacity_budget.json`
  SHA-256 `ca5edc818fee321dad257dcf3d1ba4574b6c446471a7ff7abc7c8ab790bf79f5`

### 7.3 정상 prefix와 body/value 의미 검증

정상 synthetic reference는 다음 8개 record를 남겼다.

```text
header,
step_begin, step_observation,
step_begin, step_observation,
step_begin, step_observation,
seal
```

- sequence `0..7`, valid records `8`, observation `3`, terminal inflight `null`,
  trailing bytes `0`, sealed/complete/hash-chain 모두 PASS했다.
- 8개 append receipt 전부 file `fsync` 뒤 새 descriptor exact reread가 PASS했다.
- contact-point reported count/high-water는 `17/17`, `18/18`, `19/19`였다.
- 독립 hard-coded oracle `19/19`가 PASS했다.
- synthetic body/value oracle는 step 0에서 gripper instantaneous threshold만 true,
  step 1에서 gripper `0.11 -> 0.12N`, step 2에서 `0.12 -> 0.13N` two-row event를
  확인했다. Synthetic object-motion oracle는 step 2에서 XY `0.6 -> 0.7mm`를
  확인했다. 이 값들은 verifier 시험용 입력이지 실제 RoArm/원통 측정값이 아니다.
- Reference file SHA-256:
  `4fec59e0332621355fd4c52f2be16768ee9afce19dcc6b936528cd859e35c95a`.

Header는 sensor `Sponge`, body/filter order
`support_table/link4/link5/gripper_link`, index `0/1/2/3`, resolved prim path,
capacity 33,280, 전체 D360 state-row top-level field contract, prereg/prepare/invocation/
harness/capacity/protocol SHA-256, resume/overwrite false, 합법적 seal reason/count를
한 번에 고정했다.

### 7.4 실패 가능한 perturbation 평가

전체 `17/17` test가 기대한 방향으로 PASS했다. 여기서 PASS는 “고장 입력을 정상으로
오인하지 않았다”는 뜻이다.

1. Abrupt child termination은 `os._exit()`로 검증했으며 SIGKILL이라고 부르지 않는다.
   - exit `73`: header+`step_begin`만 신뢰, observation `0`, terminal inflight step 보존.
   - exit `74`: observation `1`까지 신뢰, inflight `null`.
   - exit `75`: observation `1`까지 신뢰하고 미완성 trailing `147 bytes`를 수정 없이 분리.
2. 원 byte flip, record reorder, middle delete, duplicate sequence 네 경우는 hash/sequence
   prefix gate가 모두 거부했다.
3. 공격자가 record hash를 전부 다시 맞춘 네 경우도 wire hash-chain 자체는 true였지만
   semantic verifier가 모두 거부했다.
   - wrong event body/value:
     `event body/value projection differs from the current and previous state rows`
   - wrong header body order: `header body label order mismatch`
   - observation 1개 뒤 premature seal:
     `seal is not registered ... observation_count=1`
   - full state에서 `q5_actual_rad` 삭제:
     `state_row top-level field set mismatch`
4. Actual `reconcile_prefix()` positive는 difference `null`; 외부 projection의 gripper force를
   `0.12 -> 8.12N`으로 바꾼 negative는 최초 차이를
   `$[1].state_row.contact.by_filter.gripper_link.force_norm_n`에서 정확히 보고했다.
5. Missing body와 JSON NaN도 writer/schema가 거부했다.

각 test 결과는 aggregate 판정 전에 별도 append+fsync journal에 sequence `0..16`으로
즉시 기록됐다. Journal 17줄 모두 pass이며 phase stream도 sequence `0..21` exact다.

Source:

- `claudedocs/runtime_logs/grasp_track/g0a_d361/d361_failure_injection_results.json`
  SHA-256 `817a2d7de2c6d567ac963ff0bb9fbc3f08bc7e341dea2d5aec7b3294f18c3502`
- `claudedocs/runtime_logs/grasp_track/g0a_d361/d361_failure_injection_results.jsonl`
  SHA-256 `7eb9d248ee69d60d7c3f5c4c0617fa043d6dabaa369e28c2febcb34f134bbc8b`
- `claudedocs/runtime_logs/grasp_track/g0a_d361/d361_phase_markers.jsonl`
  SHA-256 `6005becca3705ba310dd6d2ba644cda40bc9c44f5e74311452eb1a3b8deaa0e4`

### 7.5 Scope·inventory·불변성

- 사전등록 exact artifact `23/23`이 존재하고 unexpected/missing file은 없다.
- Completion이 고정한 precompletion inventory의 bytes/SHA-256은 독립 재계산과 exact다.
- Runtime exception artifact는 없고 이미지/영상/RRD/RBL도 `0`개다.
- Isaac launch, PhysX science run, physics step, q5 command/sample, target/IK/path change,
  asset/physics-setting change는 모두 `0`이다.
- D360 output tree 17 files와 D334 user-owned sidecar 3 files는 before/after exact다.
- Main process forbidden module list는 빈 배열이다.
- Rerun은 순수 파일/schema/control audit이라는 preregistered 이유로 생략했다.

Completion SHA-256:
`3a1c9ce273182e320e3b90dd82c2ab28d8f356b892b5e4b5b346ffee8912b095`.

### 7.6 쉬운 최종 판정과 다음 승인 경계

D361이 고친 것은 두 가지다. 첫째, D360의 16-slot 상세접촉 버퍼를 다시 쓰지 않도록
현재 설치 version과 collider 수에서 33,280-slot contract를 근거화했다. 둘째, 다음 실행이
중간에 죽어도 매 step의 시작 여부와 완성된 전체 state/body/value를 디스크 prefix에서
구분하고, 해시만 다시 맞춘 거짓 내용도 거부할 수 있음을 오프라인에서 검증했다.

D361은 원통에 누가 먼저 닿았는지, 힘이 얼마인지, 원통이 움직였는지, 현재 자세가
성공/실패인지 답하지 않았다. 따라서 다음 값은 그대로다.

- `contacting_body=null`
- `contact_force=null`
- `object_motion=null`
- `current_pose_support_or_rejection=null`
- `grasp_feasibility=null`
- `g0a_pass=false`

실제 q5/PhysX 재실행과 새 접촉 영상은 여전히 별도 명시 승인이 필요하다. 후속 case가
승인되면 D360의 동결된 자세/target/asset/physics 계약을 상속하고, D361의 capacity와
prefix를 실행 전에 통합·사전등록해야 한다. D361 자체는 재실행·덮어쓰기하지 않는다.
