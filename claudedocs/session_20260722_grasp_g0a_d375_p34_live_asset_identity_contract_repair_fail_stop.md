# Session 2026-07-22 — D375 P34 live asset identity contract repair FAIL_STOP

## 1. 무엇을 왜 확인했나

사용자는 `D375 [p34_live_asset_identity_contract_repair]`를 승인했다. D373은 P34
충돌체 34개를 만들고 콜백까지 받았지만, 전체 로봇을 instanceable로 만들어
`link5`와 `gripper_link`가 지원되지 않는 articulation instance proxy가 되었다. D375의
질문은 동결된 P34 형상을 바꾸지 않고 이 구조만 수리했을 때 실제 PhysX가 다음을
정상적으로 읽는지였다.

- `link5` P34 충돌체 16개
- `gripper_link` P34 충돌체 18개
- 두 강체의 유효한 property query(질량·질량중심·관성 및 자식 collider)
- 동결 authored Float32와 실제 callback 원 다각형의 identity

이번 case의 신규 변수:

- `whole_robot_noninstance_direct_live_identity_contract_v1`

물리 step, q5, 접촉, 원통, target/IK/path, asset 재생성, 자동 convex decomposition,
material/mass/actuator/physics 설정 변경은 모두 금지했다.

## 2. 부팅·Git·동결 입력

- 시작 시 `HEAD == origin/master == 3d71aac219ba16f3262dc94b1898a459eaa534e7`.
- 시작 worktree는 clean이었다.
- 동결 D372 geometry, D373 asset/raw/fail, D374 repair/evidence/completion, D343 typed
  Float32 및 D348 callback polygon topology 해시를 사전등록에서 다시 확인했다.
- D373 asset root와 physics USD는 각각
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`,
  `1284fe48686baf1746d3a1537cb4774f3f32292f87fafb5eacf1e69772c8a9e8`로 유지됐다.
- 사용자 소유 `claudedocs/lab_meeting/20260715/d334_collision_table/`은 수정하지 않았다.

## 3. NVIDIA 공식 문서와 설치 버전 교차검증

설치 스택은 Isaac Sim `5.1.0.0`, Isaac Lab `2.3.0`, Omni PhysX/schema
`107.3.26`, RTX 4090 Laptop GPU(16,376 MiB, compute capability 8.9), driver
`580.159.03`이다. `numpy==1.26.0`, `psutil==5.9.8` pin도 유지됐다.

버전 일치 공식 근거:

1. NVIDIA, **Omni Physics 107.3 — Rigid Bodies**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/rigid_bodies.html  
   articulation link는 scenegraph instance 또는 PointInstancer가 될 수 없고, 한 rigid body
   아래 여러 collider를 두는 것은 지원한다고 명시한다.
2. NVIDIA, **Omni Physics 107.3 — Query The Mass and Volume**  
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/mass_inertia_queries.html  
   rigid-body callback, 자식 collider callback, finished callback 순서와 `VALID` 결과만
   측정 권위로 쓰는 계약을 명시한다.
3. NVIDIA, **Isaac Sim 5.1.0 — isaacsim.simulation_app**  
   https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html  
   `close(skip_cleanup=False)`는 full cleanup의 graceful shutdown,
   `skip_cleanup=True`는 immediate exit라고 설명한다.
4. NVIDIA, **Isaac Sim 5.1.0 Release Notes**  
   https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html  
   `SimulationApp`에 `skip_cleanup`이 추가됐고 shutdown/exit hang 수정도 기재되어 있다.

설치 소스도 `SimulationApp.close()`가 stage를 닫은 뒤 마지막에
`shutdown_and_release_framework()`를 호출함을 확인했다:
`/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py:763-838`.
공식 문서는 일반 계약을 설명할 뿐, 이번 RoArm/P34 workload에서 정상 반환을 보장한다는
뜻은 아니다. 이번 hang의 구체 원인은 후속 localization 없이는 확정하지 않는다.

## 4. 사전등록 순서

### attempt1 — 실제 worker 0회

`attempt1_whole_robot_noninstance_direct_live_identity_repair`는 prepare에서 멈췄다.
샌드박스 안의 Python 자식 프로세스가 실행한 `nvidia-smi`만 return `9`였고 나머지
검사는 전부 통과했다. Isaac worker는 실행하지 않았다. 이 경로는 그대로 동결했다.

### attempt2 — 외부 GPU 증명 경로만 수리

직접 host 조회는 다음을 정상 반환했다.

- GPU: `NVIDIA GeForce RTX 4090 Laptop GPU`
- driver: `580.159.03`
- VRAM total/used/free: `16376/480/15465 MiB`
- compute capability: `8.9`
- 별도 Isaac 프로세스: 없음

이를 `d375_external_gpu_attestation.json`에 고정하고 해시
`a2471bebba6f068846c49dd7cd9b9a8afbe6e36c2d90a4ea095c6428fdb53e96`로
사전등록했다. attempt2 prepare는 checks `21/21`, failure-capable negative controls
`4/4` PASS했다. preregistration SHA-256은
`4048cc8201029e4f4d196fe6f68e1f0fdfe90907627b20edeb57ca9a6709744b`다.

## 5. 실제 실행 — 관찰 순서

actual Isaac worker는 정확히 1회, automatic retry는 0회였다.

1. headless Isaac AppLauncher를 `cuda:0`으로 1회 시작했다.
2. 동결 D373 P34 asset을 메모리 stage에 reference했다. 새 USD materialization/write는 0이다.
3. `/World/Robot`, `link5`, `gripper_link`를 모두 non-instance/non-proxy로 확인했다.
4. direct authored Float32 P34 `16+18`을 읽었다.
5. live P34 경로 34개를 확인했다.
6. 각 live 경로에 PhysX callback을 정확히 1회씩, 합계 34회 요청했다.
7. `link5`, `gripper_link` property query를 각각 1회 요청했다.
8. PhysX stage를 detach하고 raw summary와 preclose sentinel을 해시로 묶었다.
9. worker는 PASS JSON을 stdout에 출력한 뒤 `launcher.app.close()`를 호출하는 `finally` 구간에
   들어갔다. 이후 process가 exit하지 않았지만 post-close marker가 없으므로 `close()` 내부와
   interpreter teardown 중 정확히 어디가 block됐는지는 미확정이다.
10. 900초 watchdog 후 supervisor가 SIGTERM을 보냈고, 추가 20초에도 종료되지 않아
    SIGKILL했다. 전체 elapsed는 `920.3908159369603s`, returncode는 `-9`다.

## 6. 종료 전에 확보된 raw subresults

다음은 full D375 identity PASS가 아니라 종료 전에 해시로 보존된 제한적 subresult다.

- owner structure: `/World/Robot`, `link5`, `gripper_link` 모두 valid,
  non-instance, non-instance-proxy, non-instanceable; 전체-로봇 `SetInstanceable(True)` 호출 `0`.
- direct authored readback: `34/34` PASS (`link5 16`, `gripper_link 18`).
- D343 typed Float32 authority 상속; 재시험 `0`; observed bits 계약은 `0x38d1b717`.
- live inventory: `34/34` PASS, active A64 `0`, disabled known legacy `2`.
- callback protocol: 실제 live path `34/34` PASS, 오류 `0`.
- property query:
  - `link5`: rigid result `VALID`, collider `17/17 VALID`, mass
    `0.015392799861729145kg`.
  - `gripper_link`: rigid result `VALID`, collider `19/19 VALID`, mass
    `0.0028707999736070633kg`.
  - 각 count는 활성 P34 `16/18`과 비활성 legacy collider 1개를 포함한다.
- authored MassAPI base↔P34 delta: 두 body의 mass/COM/inertia/principal axes 모두 `0.0`.
- raw/preclose worker protocol: `true/true`; preclose summary SHA, counters, timeline exact.
- timeline은 전후 STOP, time `0.0s`; physics step/q5/contact/cylinder/public forward는 모두 `0`.
- P34 asset file hashes는 전후 동일, materialization/write `0/0`.

원본 해시:

- raw summary: `74f959b765860d06ca1d892823d47dc395cad3aea92d0250e21ff706263fc21e`
- preclose sentinel: `1352d49f63b1ba58c75c1e5ad4d0bcb2d000510f1fc060938d672c53288d5203`
- supervisor: `69f5f8ec5760e7804f3d076c377fc0ea597bde902f3d8ec7d941f36208f4f51c`
- fail-stop attestation: `c3fb645ae9ca918e433bdf1734561504aab01a63d97d50086393c16b5d6f8fc7`

## 7. 최종 판정과 해석

최종 verdict:

`D375_P34_LIVE_ASSET_IDENTITY_CONTRACT_REPAIR_FAIL_STOP`

이 판정의 이유는 형상·property subresult 실패가 아니라 **프로세스 종료 권위 실패**다.
사전등록 supervisor 식은 returncode `0`, no timeout/signal, raw/preclose PASS와 해시 일치를
모두 요구한다. 이번에는 hash authority는 PASS했지만 timeout/SIGTERM/SIGKILL/return `-9`로
operational/effective PASS가 false다.

따라서 다음 표현만 허용한다.

- whole-robot instancing 제거는 D373의 `ERROR_PARSING`을 실제 `VALID` property rows로
  바꾼 raw subresult를 만들었다.
- P34 34개의 live acquisition은 종료 전에 성공적으로 기록됐다.
- 그러나 D375 full authored↔callback surface/bounds/original-polygon topology-volume
  classification은 supervisor fail-closed 때문에 실행하지 않았고 identity PASS는 아니다.
- A64/P34 물리 비교, 전도, 파지 가능성은 모두 `null`; `g0a_pass=false`다.

## 8. 시각화 상태

정확한 1920×1080 board와 save-only RRD/RBL은 생성하지 않았다. 이는 누락을 숨긴 것이
아니라 분석기가 hash-bound supervisor FAIL이면 geometry 판정을 시각화로 우회하지 않도록
사전등록됐기 때문이다. D375 실패를 교수님과 사용자가 볼 수 있게 만드는 작업은 immutable
D375 JSON만 읽는 별도 offline observability case로 해야 한다.

## 9. 다음 승인 경계

다음 최소 후보는 아직 미승인이다.

`D376 [d375_terminal_close_provenance_and_failure_visualization]`

- immutable D375 evidence만 읽는 offline-only case
- Isaac/PhysX 재실행 0
- 900초 shutdown hang의 프로그램 순서, supervisor 신호, raw subresult/전체 FAIL 분리를
  정확한 1920×1080 board와 save-only RRD/RBL로 시각화
- 설치 5.1 `SimulationApp.close()` 소스와 공식 문서의 graceful/immediate 경로를 대조하되
  구체 hang 원인을 추측으로 확정하지 않음

D376 뒤 실제 live retry가 필요하면 `skip_cleanup`, supervisor-owned terminal exit 또는 다른
종료 계약 중 **한 변수**를 다시 사전등록하고 별도 승인을 받아야 한다. 그 repaired live
identity가 full PASS하기 전에는 A64/P34 원통 물리 비교로 가지 않는다.

## 10. 주요 경로

- `sim_scripts/cyl34_top_view_d375_p34_live_asset_identity_contract_repair.py`
- `sim_scripts/cyl34_top_view_d375_p34_live_asset_identity_contract_repair_worker.py`
- `claudedocs/runtime_logs/grasp_track/g0a_d375/d375_external_gpu_attestation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d375/attempt1_whole_robot_noninstance_direct_live_identity_repair/d375_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d375/attempt2_external_gpu_attestation_repair/d375_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d375/attempt2_external_gpu_attestation_repair/d375_worker_raw_summary.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d375/attempt2_external_gpu_attestation_repair/d375_worker_preclose_sentinel.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d375/attempt2_external_gpu_attestation_repair/d375_worker_supervisor.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d375/attempt2_external_gpu_attestation_repair/d375_fail_stop_attestation.json`
