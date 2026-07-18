# D365 — hierarchy current/render-cache propagation localization

상태: `COMPLETE_D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED`

이번 case의 신규 변수:

1. `optional_compatibility_removed_from_linear_gate`
2. `physx_hierarchy_mesh_cache_hydra_single_pose_localization`

## 1. 무엇을 왜 측정하는가

D362의 실제 물리 trace에서는 원통이 최종적으로 XY `60.61899778989994mm` 이동하고
`89.99777464743418deg` 기울었다. 그러나 D363의 actual Isaac viewport는 원통을 계속
수직으로 표시했다. D364는 이 표시 단절을 계층별로 찾으려 했지만, 현재 scene에 없는
compatibility `_worldPosition/_worldOrientation/_worldScale` 속성을 필수 전달층으로
잘못 두어 pose write 전에 정지했다.

D365는 D364를 재실행하지 않는다. D364가 실제로 확인한 baseline을 그대로 상속하고,
현재 Isaac 5.1/Fabric 계층에서 권위가 있는 아래 직렬 경로만 한 번 측정한다.

`AssetData cache → independent PhysX tensor view → IFabricHierarchy root current →`
`IFabricHierarchy render-mesh current → Boundable mesh cached worldMatrix → Hydra pixels`

여기서 independent PhysX tensor view는 AssetData cache를 다시 읽는 것이 아니라
`root_physx_view.get_transforms()`를 직접 읽는 별도 backend view다. 다만 solver 내부
commit 자체를 독립 증명한다고 과장하지 않는다.

## 2. 승인과 절대 경계

사용자는 D365 `[hierarchy_current_render_cache_propagation_localization]`를
observability-only case로 승인했다. 정확한 실행 한도는 다음과 같다.

- actual `headless=false` Isaac worker: `1`회
- automatic retry: `0`
- cylinder root pose write: `1`회
- public `SimulationContext.forward()`: `1`회
- controlled physics step: `0`
- q5 science sample/target update: `0/0`
- contact query: `0`

Target/IK/path, asset/decomposition/gate/material/mass/actuator/physics/renderer 설정은
바꾸지 않는다. D362-D364와 사용자 소유
`claudedocs/lab_meeting/20260715/d334_collision_table/`은 읽기 전용으로 봉인한다.
이 case는 cap/rim, contact, grasp, hold/lift, G0a를 재판정하지 않는다.

## 3. Git와 frozen 입력

실행 전 교차검사 기준:

- `HEAD == origin/master == 94c0644ef3d4e69278bc864f0f8c2f3a40908dc8`
- commit subject: `D363test`
- 현재 worktree의 기존 변경은 D364 state/harness/output뿐이며 commit/push하지 않는다.

D364의 원 증거:

- `d364_prewrite_fail_completion.json` SHA-256
  `845f3b392b986f4354618f9191c953ee3919e6917ed1db1f382e7f6654810211`
- D364 harness SHA-256
  `4377203b9756b503b4f8a80f955db93ed9301edaa84e3113d81cfdee4c558925`
- D364 session SHA-256
  `e931a37ced51a05c793574e0f9940534e5e2d0b0e2ca9489390c90c4bb203e69`
- D364 runtime attestation SHA-256
  `777ea962cea18fd9d9c2bbcf2306f70adf4a7f068c0439f2be4941a546703eb4`
- D364 worker exception SHA-256
  `f1a9d5334586bda31d197440ad6503dddc6f04b43bdbca7d4a2747318ed4cd6b`

D364 raw baseline의 PhysX pose는 position
`[0.30000001192092896, 0.0, 0.03288299962878227]m`, quaternion wxyz
`[1.0, 0.0, 0.0, 0.0]`다. Root/mesh hierarchy local/current와 mesh cached
worldMatrix가 valid였고, root current는 PhysX와 일치했다. D365는 새 worker의 baseline
PhysX/root current/mesh current/mesh cached를 이 raw 값들과 다시 대조한다. PNG는 Kit
세션 간 bit-exact를 요구하지 않고 D365 한 실행 안의 fresh baseline을 픽셀 권위로 쓴다.

## 4. optional diagnostic과 필수 계층의 분리

다음 값은 계속 원 JSON에 기록하지만 직렬 verdict, terminal availability,
BASELINE/TARGET 이진성, 시간축 regression, nonmonotonic 판정에 넣지 않는다.

- root compatibility `_worldPosition/_worldOrientation/_worldScale`
- compatibility `_localMatrix`
- non-Boundable root cached `omni:fabric:worldMatrix`

반면 root/mesh hierarchy local attr의 valid/present 여부는
`get_world_xform()` 입력 계층을 실제로 읽고 있다는 측정 무결성 prerequisite로 확인한다.
직렬 상태값에는 root/mesh current와 Boundable render-mesh cached worldMatrix만 들어간다.

설치된 공식 구현/문서의 교차근거:

- `isaaclab/sim/simulation_context.py:466-472`: paused 상태에서도 public `forward()`는
  `_update_fabric(0,0)`을 호출하며 physics simulate를 호출하지 않는다.
- 같은 파일 `:826-841`: physics step 없이도 갱신하도록 `force_update`를 binding한다.
- `usdrt.scenegraph .../docs/fabric_hierarchy.rst:77-99,109-129`: hierarchy current는
  local hierarchy에서 계산하며 Boundable prim의 cached `worldMatrix`가 Hydra 쪽 렌더
  전달값이다.

Runtime에서 `cfg.use_fabric`, interface enabled, FSD, Hydra-read setting,
선택 callable 이름 `force_update`, 그리고 그 callable이 실제 interface에 bound됐는지를
모두 필수로 확인한다.

## 5. 실제 한 번의 관측 순서

1. Launcher와 frozen D362 환경을 생성하고 reset한다. Reset 내부 transition은
   controlled physics counter에서 제외한다.
2. Timeline을 playing도 stopped도 아닌 paused 상태로 만들고 clock/counter를 봉인한다.
3. D364 raw baseline과 새 baseline의 PhysX/root current/mesh current/mesh cached를 대조한다.
4. `baseline_pre_capture`, 두 camera PNG, `baseline_post_capture`를 기록한다.
5. frozen D362 row 499의 최종 원통 pose를 cylinder root에 정확히 한 번 쓴다.
6. 어떠한 forward/app update 전 `post_write_immediate`를 읽는다.
7. clock-guarded app update로 primary/opposite `post_write_no_forward` PNG를 남긴 뒤
   `post_write_after_app_update`를 읽는다.
8. public `inner.sim.forward()`를 정확히 한 번 호출하고 clock 무진전을 확인한다.
9. app update 전 `post_forward_immediate`를 읽는다.
10. clock-guarded app update로 primary/opposite `post_forward` PNG를 남긴 뒤
    `post_forward_after_app_update`를 읽는다.
11. 총 6 checkpoint, 6 PNG, append-only layer journal, worker/supervisor marker를 봉인한다.
12. 자동 report/sheet/RRD/RBL을 만든 뒤 원본 PNG와 Rerun screenshot을 사람이 확인한
    경우에만 finalize한다.

모든 checkpoint에서 PhysX getter 전후 clock과 필수 Fabric
root-current/mesh-current/mesh-cached 값이 바뀌지 않아야 한다. Optional `_world*`가
생기거나 사라지는 것은 진단에 남기되 getter side-effect FAIL로 만들지 않는다.

## 6. 사전등록 verdict tree

필수 terminal 계층에 missing/OTHER가 있거나, 앞 계층은 BASELINE인데 뒤 계층이 TARGET인
비단조 상태, TARGET 후 BASELINE 회귀, 두 camera 불일치가 있으면
`D365_INCONSISTENT_OR_UNLOCALIZED_FAIL_STOP`이다. 필수 측정 자체가 없으면
`D365_MEASUREMENT_INCOMPLETE_FAIL_STOP`이다.

그 외 최초 BASELINE 경계는 다음처럼 판정한다.

| 관측 | verdict |
|---|---|
| cache가 target 아님 | `D365_DIRECT_WRITE_OR_CACHE_FAIL` |
| cache target, PhysX target 아님 | `D365_CACHE_TO_PHYSX_PENDING_OR_FAILED` |
| PhysX target, root current target 아님 | `D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED` |
| root current target, mesh current target 아님 | `D365_FABRIC_ROOT_TO_RENDER_PRIM_HIERARCHY_NOT_PROPAGATED` |
| mesh current target, mesh cached target 아님 | `D365_FABRIC_RENDER_PRIM_CURRENT_TO_RENDER_CACHE_NOT_PROPAGATED` |
| mesh cached target, Hydra target 아님 | `D365_FABRIC_TO_HYDRA_NOT_PROPAGATED` |
| 전 필수 계층 target | `D365_END_TO_END_ZERO_STEP_VISIBLE` |

Optional compatibility가 `UNAVAILABLE`, `BASELINE`, `OTHER`인 음성 대조군과 root cached
`UNAVAILABLE/BASELINE` 대조군은 위 verdict를 바꾸지 않아야 한다. 각 필수 계층은
`first_target_phase`도 기록해 write, app update, public forward 중 어느 경계에서 처음
전달됐는지 분리한다.

## 7. 실패 가능한 사전검증과 완료 권위

이 observability case의 실패 가능한 perturbation evaluation은 다음을 포함한다.

- cache/PhysX/root current/mesh current/mesh cache/Hydra 각 단절 fixture
- optional compatibility unavailable/baseline/other fixture
- root 또는 mesh downstream-ahead 비단조 fixture
- missing/OTHER/시간축 회귀 fixture
- quaternion sign equivalence, xyzw↔wxyz swap, 10mm translation 음성 대조
- 두 camera 및 full Fabric getter 전후 변화 대조

D364에서 OS worker exit code `0`만으로는 성공을 알 수 없었다. D365의 완료 권위는
동시에 다음을 요구한다.

1. worker exception artifact 부재
2. worker summary 존재 및 `pass=true`
3. worker phase stream의 마지막 marker가 `worker_summary:complete`, `pass=true`
4. worker exit `0`, watchdog null
5. write/forward `1/1`, controlled physics/q5/contact `0/0/0`

한 항목이라도 실패하면 report/RRD가 부분 생성되어도 정상 완료로 승격하지 않으며,
actual worker를 retry하지 않는다.

## 8. 실행 전 상태

- D365 output은 아직 존재하지 않는다.
- D365 actual Isaac invocation count는 `0`이다.
- Harness 구현과 CPU-only 정적/negative-control 검토 뒤 prepare를 한 번 실행한다.
- Prepare PASS 뒤에만 actual GUI worker를 한 번 실행한다.
- 이 문서의 이후 결과 절은 actual run/finalize가 끝난 뒤 append-only로 추가한다.

## 9. Prepare attempt1 — sandbox host-access gate FAIL, actual invocation 0

CPU-only prepare를 처음 호출했을 때 과학/코드/lineage 항목은 모두 PASS했지만 다음
세 host-access 항목만 FAIL했다.

- `display_available=false`, raw `xdpyinfo` return code `1`
- `gpu_exact=false`
- `gpu_free_gate=false`, raw `nvidia-smi` return code `9`

원 artifact는 덮어쓰지 않고 아래에 그대로 보존한다.

- `claudedocs/runtime_logs/grasp_track/g0a_d365/d365_preregistration.json`
  SHA-256 `2b81e3609bb89d64d48061235e8baa8541f787d873a319fd0dd000695803e668`
- `claudedocs/runtime_logs/grasp_track/g0a_d365/d365_prepare_preflight.json`
  SHA-256 `8e676a4f1ece7b4d6714f9230fc2bf15206d6324116654a1d0fe3548e6b06301`

이는 actual Isaac worker 실패나 retry가 아니다. Invocation marker, AppLauncher,
environment/reset, pose write, forward는 모두 아직 `0`이다. 같은 시점 호스트 명령은
`xdpyinfo -display :1` PASS와 다음 GPU row를 반환했다.

`NVIDIA GeForce RTX 4090 Laptop GPU, total 16376MiB, used 2051MiB, free 13894MiB, util 18%`

따라서 원인은 D365 계약이나 GPU 부족이 아니라 샌드박스 안 Python 자식 프로세스의
X/NVIDIA 장치 접근 차이다. Developer execution rule에 따라 중요한 host-access check는
호스트 권한으로 다시 수행하되, 실패한 두 파일은 수정하지 않는다. 새 forward-only
output은 다음으로 고정한다.

`claudedocs/runtime_logs/grasp_track/g0a_d365/attempt2_host_access_prepare_repair/`

Attempt2 prepare는 attempt1 두 hash, 정확히 세 access gate만 실패했다는 사실, invocation
marker 부재를 입력으로 봉인한다. 이 운영상 prepare repair는 신규 과학 변수가 아니며,
actual worker 등록 한도는 여전히 정확히 `1`회/no retry다.

## 10. Attempt2 prepare와 실제 1회 실행

Host-access repair output은
`claudedocs/runtime_logs/grasp_track/g0a_d365/attempt2_host_access_prepare_repair/`에만
새로 생성했다. Attempt2 prepare는 `19/19` checks PASS였다. 이때 GPU는 정확히
`NVIDIA GeForce RTX 4090 Laptop GPU`, total/used/free `16376/2459/13486MiB`, available
RAM `12,163,997,696B`였다. 사전등록 SHA-256은
`1637fde5eab5b1b6dbb73bb764f623e2bb4ab855bd5ed86b2df646b2890caa0e`다.

그 뒤 actual `headless=false` worker를 정확히 한 번만 실행했고 retry하지 않았다.
Worker exit는 `0`, elapsed는 `28.81899581302423s`, watchdog reason은 `null`이었다.
관측 순서는 preregistration과 같은 여섯 checkpoint였다.

1. `baseline_pre_capture`
2. `baseline_post_capture`
3. `post_write_immediate`
4. `post_write_after_app_update`
5. `post_forward_immediate`
6. `post_forward_after_app_update`

Append-only journal은 여섯 record, exact label/order/hash-chain, 모든 getter clock guard를
PASS했다. 등록된 cylinder pose write는 `1`회이며 return도 true였다. Public
`SimulationContext.forward()`도 `1`회이며 return true였다. Controlled physics step,
q5 science sample, q5 target update, contact query는 각각 `0/0/0/0`이었다. Timeline은
playing도 stopped도 아닌 paused 상태였고 custom step counter와 simulation/timeline clock은
write, getter, app update, forward 전후에 진전하지 않았다.

## 11. 계층별 원 관측 결과

Baseline position/quaternion wxyz는 D364 raw baseline과 exact하게 일치했다.

`[0.30000001192092896, 0.0, 0.03288299962878227, 1.0, 0.0, 0.0, 0.0]`

한 번 쓴 목표는 immutable D362 row `499`, global step `500`의 최종 pose다.

`[0.35721367597579956, 0.020038070157170296, 0.004883049055933952,`
` 0.6990910172462463, -0.1845378279685974, 0.6826027631759644,`
` 0.1061641052365303]`

Baseline과 target의 최대 축 위치 차이는 `0.057213664054870605m`, quaternion angular
차이는 `91.29174613972452deg`라서 두 상태를 오인할 수 없는 충분한 separation이었다.
Terminal classification은 다음과 같다.

| 직렬 층 | terminal class | 최초 TARGET phase |
|---|---:|---:|
| AssetData cache | `TARGET` | `post_write_immediate` |
| independent PhysX tensor view | `TARGET` | `post_write_immediate` |
| IFabricHierarchy root current | `BASELINE` | `null` |
| IFabricHierarchy render-mesh current | `BASELINE` | `null` |
| Boundable mesh cached worldMatrix | `BASELINE` | `null` |
| Hydra pixels | `BASELINE` | `null` |
| optional root compatibility `_world*` | `UNAVAILABLE` | verdict 제외 |
| optional root cached worldMatrix | `BASELINE` | verdict 제외 |

즉 setter의 내부 cache만 바뀐 것이 아니다. 별도의
`root_physx_view.get_transforms()`도 write 직후 target을 반환했다. 그러나 root current부터
render mesh current, cached render matrix, 실제 Hydra 화면까지는 baseline에 남았다. Public
`forward()`가 선택한 Fabric callable은 실제 interface에 bound된 `force_update`였고,
`cfg.use_fabric`, Fabric interface/enabled, FSD, Hydra Fabric-transform settings도 모두 true였다.
그럼에도 이 한 번의 paused zero-step 경로에서는 PhysX tensor state가 Fabric hierarchy
current로 전달되지 않았다.

이 결과가 증명하는 것은 **단절 경계**다. 왜 그 전달이 일어나지 않았는지의 더 내부 원인,
예를 들어 어떤 scene-update/commit event가 필요한지 또는 특정 clone 경고가 원인인지는
D365만으로 확정하지 않는다.

## 12. 실제 Isaac 픽셀과 시각검사

두 camera에서 baseline, write 뒤/forward 전, forward 뒤의 원본 `1280x720` PNG 여섯 장을
남겼다. 모두 `upright=true`, `toppled=false`였다. Baseline→post-forward 비교는 다음과 같다.

| 시점 | centroid delta | PCA axis delta | mask IoU | materially different |
|---|---:|---:|---:|---:|
| primary | `0.014426487842385519px` | `0.019178983364014357deg` | `0.999118113939679` | false |
| opposite | `0.013410813923302773px` | `0.006548856853399343deg` | `0.9976521688090626` | false |

원본 여섯 PNG, `4800x3000` 한국어 비교 sheet, `6400x3600` Rerun screenshot의 정확히
여덟 경로를 원본 해상도로 수동 검사했다. 실제 Isaac 여섯 화면은 모두 선 원통이고,
Rerun에는 그 실제 이미지들과 쓰러진 commanded target geometry가 함께 보여 단절을
숨기지 않았다. 일부 Rerun UI 한글 entity label은 네모 글꼴로 보였지만 실제 이미지,
영문 식별자, 수치 판정은 읽을 수 있었다. Manual checks는 `14/14` PASS였다.

Rerun SDK/CLI `0.34.1`, RRD footer, exact non-system entity/timeline/component, RBL,
headless screenshot 검증은 모두 PASS했다. RRD/RBL/screenshot bytes는 각각
`9,806,970/65,842/7,055,899B`였다.

## 13. 자원과 무결성

28개 supervisor sample에서 GPU used max/free min은 `8576/7369MiB`, utilization max는
`37%`, worker RSS max는 `7,131,017,216B`였다. VRAM 고갈, watchdog, worker exception,
postprocess exception은 없었다. 이 utilization은 SM/Warp occupancy 측정값이 아니며,
단절 원인으로 사용하지 않는다.

D362/D363/D364 manifests, frozen inputs, D334 sidecar는 모두 unchanged였다. Completion의
integrity checks `19/19`, output inventory `27/27`가 PASS했다. 핵심 SHA-256은 다음과 같다.

- harness: `719011a6171e27ae6b759903ef060397c1fad10c5c822d4c24a207ccdc59d834`
- worker summary: `3babf239358f48ae5d2edd2124e3bcdb29d76ef5464a82b958c9fddd3a2e8c2e`
- localization report: `b82065bf89930f80d2c6e4a38bdaf9a323b2604b25efb7cb840b29b0ac4c5420`
- RRD: `73a2a1d5954e6dfadfb7e562ea2ac4de8dcd80413e94b58db8abab069378a056`
- manual inspection: `ce0c8e02f033ca99032ce6c0dd6855e24259c951b966537252603575f3149a72`
- completion: `efb2ece5bb30fa987bfa8a6ed229d282efdecff6c359beaa8034448ff1c3752d`

## 14. 최종 판정과 과학적 범위

최종 operational verdict는
`D365_PHYSX_TO_FABRIC_HIERARCHY_NOT_PROPAGATED`, completion `pass=true`다. 여기서 PASS는
국소화 측정과 증거 계약이 완결됐다는 뜻이지 grasp 성공을 뜻하지 않는다.

D365는 physics/q5/contact를 한 번도 실행하지 않았으므로 D362의 물리 sub-result를
바꾸지 않는다. Cap/rim science, exact contact face/manifold, force closure, stable grasp,
hold/lift, target/IK repair justification는 계속 `null`, `g0a_pass=false`다. D365의
failure-capable evaluation은 preregistered cache/PhysX/Fabric/Hydra 단절 fixtures와
ordering/quaternion/translation/getter-side-effect 음성 대조군이었다. 승인 범위가
observability-only였기 때문에 새 물리 실험을 하지 않은 것은 의도된 경계다.

## 15. 동결과 다음 승인 경계

D365 두 output 경로를 동결한다. Attempt1 sandbox-access failure와 attempt2 completed run을
합치거나 덮어쓰거나 재실행하지 않는다. 다음에는 추측성 render repair를 바로 넣지 말고,
먼저 installed IsaacLab/Isaac Sim의 rigid-body tensor setter 이후 PhysX scene update와
Fabric hierarchy update가 어떤 public ordering/commit 계약을 요구하는지 source/API
계보로 좁혀야 한다. 그 뒤에도 bridge 하나만 별도 case로 사전등록해 검증해야 한다.
어떤 q5/physics/contact/cap-rim/grasp science 재개도 다시 명시 승인을 받아야 한다.

원 증거:

- `claudedocs/runtime_logs/grasp_track/g0a_d365/attempt2_host_access_prepare_repair/d365_state_layer_localization_report.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d365/attempt2_host_access_prepare_repair/d365_layer_readback_journal.jsonl`
- `claudedocs/runtime_logs/grasp_track/g0a_d365/attempt2_host_access_prepare_repair/d365_supervisor_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d365/attempt2_host_access_prepare_repair/d365_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d365/attempt2_host_access_prepare_repair/d365_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d365/attempt2_host_access_prepare_repair/d365_completion_summary.json`
