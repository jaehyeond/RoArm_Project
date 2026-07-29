# RoArm cylinder grasp-track G0a D407 actual runtime

Date: 2026-07-29 KST

Status: **A/B 물리 worker는 각 1회 완주했으나, 필수 live manual inspection이
300초 안에 게시되지 않아 전체 attempt는
`D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP`으로 동결**

이번 case의 신규 변수:
[`gripper_link_collision_representation_a64_to_sdf_res256_v1`]

## 1. 무엇을 왜 실행했는가

사용자가 tuple-file SHA-256
`c7001b76fa0a6c3393d9df744bb1fc0fb419400d46d84fc69e531730400d4b99`
를 정확히 인용해 controller 1회, leg A/B worker 각 1회, retry 0의
Isaac/PhysX A/B runtime을 승인했다.

목적은 D362의 동결 34×90 mm, 0.72 kg cylinder 및 200-step OPEN +
300-step q5 close 계약을 그대로 두고 **등록된 신규 과학 변수 1개**를
재측정하는 것이었다. 단 B derivative에는 link5 collision scope의
`instanceable=false` authored-metadata confound가 남으므로, 실제 모든
B−A 차이가 이 한 변수에만 귀속된다는 뜻은 아니다.

- A control: link5 A64 64 + gripper A64 64.
- B treatment: link5 A64 64 + gripper SDF res256 mesh 1.

실행 명령:

```text
/home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d407_sdf_physics_ab_d362_remeasure_controller.py \
  --approved-tuple-sha256 \
  c7001b76fa0a6c3393d9df744bb1fc0fb419400d46d84fc69e531730400d4b99
```

재시도, 별도 worker 직접 실행, D400~D406/D362 수정·재실행, commit/push는
하지 않았다. 실제 A/B perturbation 평가를 수행했으므로 AGENTS.md의
Session progress rule도 충족한다.

## 2. 실행 직전 동결과 admission

- `HEAD == origin/master ==
  a69a96d36219268e4bc5e25065cc234da9d99674`.
- freeze 당시 dirty 91, prereg allowlist 138, unexpected 0.
- 승인 tuple, prereg, controller/worker SHA, frozen inputs, A/B 각 7개
  asset, D334 sidecar, science 14함수, numpy `1.26.0`, psutil `5.9.8`,
  Rerun SDK/CLI `0.34.1` 모두 PASS.
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU, compute capability 8.9,
  VRAM total/free/used `16376/14265/1680 MiB`, host RAM available
  `31,532,904,448 bytes`.
- freeze manifest SHA-256:
  `be98eb5ec7b8fa1fa43e8384080c74726d57d10a2aeb017da28b24b5039bb464`.
- controller PID `1170119`.

근거:
`d407_runtime_freeze_manifest.json`,
`d407_controller_phase_markers.jsonl:1`.

## 3. 실제 실행 순서

| 순서 | UTC | 관측 |
|---:|---|---|
| 1 | 01:25:04.455 | freeze manifest complete |
| 2 | 01:25:04.464 | leg A tuple recheck PASS |
| 3 | 01:25:04.477 | leg A worker 시작 |
| 4 | 01:25:30.634 | leg A exit 0, PASS, `26.156553861 s` |
| 5 | 01:25:35.689 | inter-leg GPU/process settle PASS, `5.047558479 s` |
| 6 | 01:25:35.708 | leg B tuple recheck PASS |
| 7 | 01:25:35.723 | leg B worker 시작 |
| 8 | 01:26:26.283 | leg B exit 0, PASS, `50.559581534 s` |
| 9 | 01:26:26.293 | post-run asset contract PASS |
| 10 | 01:26:26.458 | A/B delta summary 게시 |
| 11 | 01:26:26.773 | A/B comparison sheet 구조 게이트 PASS |
| 12 | 01:26:26.782 | live manual inspection prompt |
| 13 | 01:31:27.035 | `received=false`, `pass=false` |
| 14~16 | 01:31:27.051~.157 | manual-inspection FAIL-STOP, completion 게시 |

controller phase는 16행 exact sequence/monotonic contract PASS이고 SHA-256은
`21b93848e2c950485a3d2f04c8e67749f1010c7fff8a30d7ee4631030ae2512f`다.
A/B invocation marker는 각 1개, `automatic_retry=false`; watchdog trigger와
강제 cleanup은 0이다. 각 process group은 성공 종료 뒤 member 0이고 현재
D407 controller/worker 프로세스도 0이다.

근거: `d407_controller_phase_markers.jsonl:1-16`,
`d407_supervisor_summary.json`.

## 4. 동결 물리 조건과 OPEN baseline

양 leg 공통:

- cylinder radius `0.017 m`, height `0.09 m`, mass `0.72 kg`.
- physics dt `0.005 s`.
- static/dynamic friction `1.5/1.2`, restitution `0.0`.
- actuator stiffness/damping/effort/velocity limit
  `80/4/2.5/3.14`.
- solver TGS, gravity magnitude `9.8100004196167`.
- 36개 gravity/solver/contact/rest-offset 열거 payload A/B 동일
  (record-only 비교).

OPEN 200-step baseline은 A/B bit-identical했다.

- max XY displacement `0.003773643762621384 mm`.
- max tilt `0.003364520785190337 deg`.
- support-table Fz last-50 median `7.063635349273682 N`.
- robot-filter force 전부 `0.0 N`.
- precommand robot-contact/object-motion confound 모두 false.

근거: 양 leg `d407_runtime_prerequisites.json`, 양 leg
`d407_worker_summary.json:3-41`,
`d407_ab_delta_summary.json:201` 이후 record-only payload.

## 5. 300-step closure의 실제 측정값

### 5.1 이벤트와 peak force

| 측정값 | A: gripper A64 | B: gripper SDF | B−A |
|---|---:|---:|---:|
| moving-gripper onset/confirm | 31/32 | 21/22 | −10/−10 step |
| object-motion onset/confirm | 41/42 | 23/24 | −18/−18 step |
| link5 onset/confirm | 45/46 | 25/26 | −20/−20 step |
| gripper peak | `43.858339929 N @54` | `464.002511438 N @38` | `+420.144171509 N` |
| link5 peak | `23.227865255 N @55` | `357.175438290 N @157` | `+333.947573035 N` |
| link4 peak/event | `0.0 N` / 미관측 | `0.0 N` / 미관측 | `0.0 N` |

`link4=0`은 이 필터에서 양성 이벤트가 없었다는 뜻이며, 접촉 부재의
일반 증명은 아니다.

### 5.2 step 500 최종 row

| 측정값 | A | B | B−A |
|---|---:|---:|---:|
| XY displacement | `60.618997790 mm` | `46.183877345 mm` | `−14.435120445 mm` |
| tilt delta | `89.997774647 deg` | `58.162202071 deg` | `−31.835572577 deg` |
| z delta | `−28.000520542 mm` | `+1.721024513 mm` | `+29.721545056 mm` |
| descriptive tilt >45° | true | true | — |

A의 event와 final scalar는 frozen D362와 차이 0으로 재현됐다.
두 leg의 제한된 physical sub-verdict는 모두
`D407_MOVING_JAW_CONTACT_AND_OBJECT_MOTION_OBSERVED`다. 이는 움직이는 jaw
body의 양성 2-step force event 뒤 물체 운동을 관측했다는 뜻까지만 갖는다.

근거: `d407_ab_delta_summary.json:3-199`, 양 leg
`d407_worker_summary.json:69-153`.

## 6. 중요한 과학 해석 경계

B의 step-500 tilt가 A보다 작다고 해서 SDF가 더 안정적이거나 전도를
방지했다고 해석하면 안 된다.

- 둘 다 prereg의 설명용 `tilt>45°` 조건에서는 toppled다. 이것은
  improvement gate가 아니다.
- B의 step 500은 settled state가 아니다. gripper/link5 force가 각각
  `415.498018905/165.253139100 N`이고, angular velocity
  `[-13.852924347, 3.591230631, -4.203969479] rad/s`, upward velocity
  `0.080504529 m/s`, table gap `8.542064191 mm`, q5 velocity
  `3.140322685 rad/s`다.
- q5 effort saturation fraction은 A `0.31`, B `0.916666667`.
- q0~q4 max actual drift는 A `0.043803111 rad`, B `0.106023550 rad`;
  final drift는 A `0.002159178 rad`, B `0.078026652 rad`.
- contact-point high-water/capacity는 A `22/33280`, B `73/17152`.
  양 log의 overflow warning은 0이지만 B의 256 contacts/pair는 documented
  engine limit가 아니라 project assumption이다.
- B derivative는 link5 collision scope도 `instanceable=false`이므로 특히
  link5-side B−A를 gripper representation 하나의 순수 인과효과로 주장할
  수 없다.

따라서 stable grasp, force closure, grasp feasibility, exact face/manifold,
cap/rim/barrel 순서, SDF 일반 우월성, 29×50 mm cylinder 전이,
collider-count tipping causality, per-prim cooked SDF internal identity는
모두 null을 유지한다. 또한 B의 live ContactSensor binding, SDF shape
inventory, property-to-sensor path는 이번이 첫 live 관측이어서 사전
보장이 없었다. 한 번의 결정론적 run에서 통계적 일반화도 하지 않는다.

근거: B `d407_physics_trace.json:107472-107572`, 양 leg
`d407_worker_summary.json:270-382`,
`d407_completion_summary.json:48-69`.

## 7. 관측성 산출물과 수동검수 실패

구조적 observability 산출물은 양 leg 모두 PASS했다.

- Rerun `0.34.1` exact.
- RRD footer, RRD/RBL verify rc 0.
- exact entity/component 및 `physics_step`/`sim_time_s` timeline PASS.
- Rerun screenshot exact `1920×1080`.
- beginner sheet exact `3840×1720`.
- A/B comparison sheet exact `3840×1080`.

그러나 구조 PASS와 사람이 읽을 수 있는 화면 PASS는 다르다. 원본 해상도로
5 PNG를 직접 다시 검사한 결과:

- A/B Rerun PNG 모두 jaw/cylinder/trajectory는 보이지만 초기 connection,
  loading, headless 알림과 hover legend가 log/timeseries를 가린다.
- A beginner sheet는 정량·장면을 판독할 수 있다.
- B beginner sheet와 A/B sheet는 정량 비교가 가능하지만 B의 긴 force
  arrow가 일부 header/panel을 침범한다.

따라서 `timeseries_legible=true`와 `no_text_overlap=true`를 정직하게
게시할 수 없다. `d407_manual_visual_inspection.json`은 존재하지 않으며
controller의 durable truth는 `received=false`, `timeout=true`,
`manual_inspection_sha256=null`이다. 사후 검사는 trace를 설명하는
관측 증거일 뿐 live gate나 전체 PASS를 소급 복구하지 않는다.
두 차례 postrun input/asset/hash recheck는 PASS했지만, 최종
`root_artifact_integrity`는 manual failure 때문에 null이며 final root
seal로 확대 해석하지 않는다.

Prompt 중 준비된 scratch writer는
`/tmp/d407_static_prep_20260729/d407_manual_writer_live.py`,
SHA-256
`6313ad37c85f3190578805dd8d6f3323ab89d8feb19fd7cecaa577c4b242dcd9`,
mtime `2026-07-29 10:28:20.575983006 +0900`이다. manual JSON, publish
scratch directory, receipt/log는 모두 부재하므로 성공 게시나 정확한
실패 시각을 이 파일로 주장하지 않는다.

## 8. 로그 warning

각 worker log에
`[Error] [isaacsim.core.cloner.impl.cloner] Failed to clone in Fabric`
한 줄이 있다. D360/D362와 여러 선행 case에도 반복된 generic startup
행이며, 이번에는 그 뒤 두 worker가 각각 500 finite row, prefix seal,
RRD/RBL, summary까지 완주했다. 따라서 기록은 보존하되 이번 manual timeout
또는 물리 결과의 단일 원인으로 확정하지 않는다.

근거:
양 leg `d407_worker_stdout_stderr.log:55`,
`claudedocs/session_20260717_grasp_g0a_d362_capacity_prefix_integrated_physx_contact_motion.md:402-407`.

## 9. SHA-256 결박

### 9.1 승인에 사용한 static 4-SHA + tuple file

- preregistration:
  `6deb6779a18619f547952de9119eee599ea5dd40ac466d57d6a813988afb1269`
- reviewed attestation:
  `86d587e687b4d139083137913bd15b57cf8f394e2fd5114bafa38567415bae91`
- controller:
  `c758ffad7199c425e87526cad54dbf7e100dbed004460d44f908421ad6a13dc1`
- worker:
  `2f6da11cc9d074d7fa626eaadfb9a638b3cc74e7acdb2ae99fe07780041101cc`
- tuple-file:
  `c7001b76fa0a6c3393d9df744bb1fc0fb419400d46d84fc69e531730400d4b99`
- static fixture results:
  `568e7df1fdcb5bdd5117fc418bdeb55c284131e21f6e77dd782ac583f22ee1ea`

Static 수치: stages `13/13`, checks `58/58`, accept `10/10`, reject
`59/59`, M-late accept `1/1` + reject `15/15`, science `14/14`,
pre-runtime runtime-counter 10종 `0`. 최종 harness는 worker `4,334행`,
controller `1,862행`; prereg frozen inputs `36`, A/B assets `7+7`,
D334 sidecars `3`, allowlist `138`이다. source-derived capacity는
A `33,280`, B `17,152`; B SDF mesh는 `41,094 vertices /
13,698 triangles`다.

### 9.2 runtime root artifacts

- completion:
  `acad525927d4c8dcdd8d1eb498aa9dbe9a3933f3843b10a67bd7d28d87c6d15b`
- supervisor:
  `66a2af2e297794740a3caf03ae8258eb13405b3e8879cfb79c119511f991b7ff`
- delta:
  `4cb532236aaaf06118e54be870c25b26d1979b4aa59594ce02631ebece484948`
- controller phase:
  `21b93848e2c950485a3d2f04c8e67749f1010c7fff8a30d7ee4631030ae2512f`
- A/B sheet:
  `df3d97156cc6ea373d8c3f40c231ea36714e32ef9cc557bbb03f2411ddd9c0d9`

### 9.3 per-leg core observability

| 파일 | A SHA-256 | B SHA-256 |
|---|---|---|
| worker summary | `c3f96a10a1a4de8ddbeec84fd86f13765f66bc067fbc694367aaa9d039d637b3` | `744244dc5f591a4f36a253392e2cd82d9e344cb65f73506968cff1fb105130e7` |
| RRD | `6bca0b5e065d84ebfc823b3ece9722b4aa17bc49c453447eb183652f8b7f65a5` | `284c3c6d85bfb3f036a0eb4be0fe53d8b411ee464b12801a5ca50d32e35e498b` |
| RBL | `85131ed01e576400a78a17368bbe063092dc6d40361b636002f3ead763603033` | `c7c9de3b55d4735f59d88a680d2d5e9c2d05bfdb0cc6f831079625562ed22f40` |
| Rerun PNG | `2da8f53524f1142c93102e7fdde1adffd7c14a7f716ad8df67bc3ebaf796188b` | `42811cde03408e0f01c9d0a708c75db5bbedbf37ed378b7a78aa76ac154ea8af` |
| beginner PNG | `eb1ea7d887ef3f94149465029b68956b8298de4fdba9c7dca4ffa52e217ae611` | `174447d211d8666015b515e4497b69431131e81ffc2083f85b4aa36658f3d6b8` |

attempt root에는 static+runtime 총 44파일이 있다. state-doc update 직전
git dirty는 122행, 그중 D407은 35행이며 현재 dirty의 prereg allowlist
이탈은 0이었다.

## 10. 최종 판정과 다음 승인 경계

최종 controller 판정:

```text
D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP
failure_classification=manual_inspection
pass=false
g0a_pass=false
```

이것은 A/B physics worker 실패가 아니다. 두 물리 trace와 제한된
contact→motion sub-verdict는 보존되지만, 필수 live manual gate가
완료되지 않아 D407 전체 PASS를 선언할 수 없다. attempt1은 소진·동결하며
같은 경로를 재실행하거나 사후 manual JSON으로 덮어쓰지 않는다.

다음 최소 후보는 새 case D408에서 immutable D407 trace/RRD를 입력으로
overlay 없는 capture, bounded force-arrow 표시, 사전 준비된 atomic
writer/handshake를 검증하는 **observability-only repair**다. 이는 D407을
소급 PASS로 바꾸지 않고 physics를 재실행하지 않는다. 설계·정적 준비도
사용자 별도 승인 전에는 시작하지 않는다.

제안 승인 문장:

`D408 [d407_manual_observability_completion_repair]의 설계와 정적 준비를 승인합니다. D407 물리 trace/RRD는 읽기 전용으로 재사용하고 Isaac/PhysX physics step, q5/contact 재실행은 금지합니다.`
