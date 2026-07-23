# Session 2026-07-22 — Grasp G0a D376 D375 terminal-close provenance and failure visualization

## 1. 무엇을 왜 확인했는가

D375는 P34 충돌체를 Isaac/PhysX에서 실제로 읽었지만 worker process가 끝나지 않아
`D375_P34_LIVE_ASSET_IDENTITY_CONTRACT_REPAIR_FAIL_STOP`으로 종료됐다. D376의 목적은
“Isaac Sim이 또 실패했다”는 포괄적 표현을 버리고, 동결 D375 JSON과 실제 Kit 로그만으로
어디까지 완료됐고 어디서 멈췄는지를 순서대로 국소화하는 것이었다. Isaac/PhysX나 원통
물리를 재실행하지 않았고, 실패를 설명하는 시각자료도 같은 immutable evidence에서 만들었다.

이번 case의 신규 변수:

1. `d375_terminal_close_provenance_contract_v1`
2. `d375_terminal_failure_visualization_projection_v1`

최종 verdict:
`D376_D375_TERMINAL_CLOSE_PROVENANCE_AND_FAILURE_VISUALIZATION_PASS`.
이는 종료 계보와 관찰자료의 PASS일 뿐, D375 full identity나 P34 물리의 PASS가 아니다.

## 2. 부팅, Git, 범위

- 부팅 때 `HEAD == origin/master == e30f7f99d44252f509e383627738f3ad7967ea93`,
  subject `D375`, clean worktree를 확인했다.
- D376은 offline process `1`, automatic retry `0`으로 실행했다.
- Isaac launch, PhysX call, physics step, q5 command/sample, contact query, cylinder write,
  USD write, collider regeneration, automatic decomposition sweep, target/IK/path change는 모두 `0`이다.
- D375와 사용자 소유 `claudedocs/lab_meeting/20260715/d334_collision_table/`은 전후
  bit-exact였다.
- D376 PNG 3개는 exact path에 존재하지만 repo `.gitignore`의 `*.png` 규칙에 걸린다.
  자동 stage하지 않았다.
- commit/push는 하지 않았다.

사전등록은 `18/18` PASS했고, 다음 registered evidence/logic guard는 `6/6` PASS했다:
정상 종료 D367을 hang으로 분류하지 않기, 여유 GPU를 OOM으로 바꾸지 않기, 6.0 fix를
D375의 thread dump로 과장하지 않기, StageCache Erase 누락을 확정 원인으로 만들지 않기,
raw PASS가 terminal timeout을 덮지 못하게 하기, 가짜 return 0이 timeout/signal을 덮지
못하게 하기. 이 중 spoofed-return은 실제 perturbation이지만 두 인과 경계 guard는 사전등록된
literal 논리 명제이므로 여섯 개 모두를 failure-capable perturbation이라고 부르지 않는다.

이번 session은 D375의 실제 terminal failure에 대한 reactive provenance case이고, 사용자가
Isaac/PhysX 재실행을 `0`으로 제한했다. 대신 frozen hash/program-order/spoofed-return/Rerun/
육안 gate가 틀리면 D376 자체가 FAIL_STOP이 되도록 했다. 이것이 offline-only 수행의 명시적
Session Progress Rule 정당화다.

근거:

- `claudedocs/runtime_logs/grasp_track/g0a_d376/attempt1_d375_terminal_close_provenance_and_failure_visualization/d376_preregistration.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d376/attempt1_d375_terminal_close_provenance_and_failure_visualization/d376_invocation.json`

## 3. 관찰 순서와 확정된 경계

### 3.1 D375 측정 본체는 종료 전에 완료됐다

동결 D375 raw/preclose 증거를 다시 읽어 다음을 확인했다.

- PhysX callback row `34/34` PASS.
- property-query collider row는 link5 `17`, gripper_link `19`; 모두 등록된 VALID 계약을
  통과했다.
- raw summary와 preclose sentinel의 protocol/hash binding이 exact였다.
- physics step, q5, contact는 `0/0/0`이었다.
- PhysX detach와 worker cleanup-end 표식까지 도달했다.

따라서 P34 readback·callback·property 측정 단계는 완료됐고, 관측된 non-exit 경계는 그
이후다. 이것만으로 shutdown interval의 GPU 상태나 callback workload의 간접 영향을 절대
배제하지는 않는다. 종료 후 classifier가 실행되지 않았으므로 full authored↔callback
surface/bounds/original-polygon topology-volume identity는 여전히 `null`이다.

### 3.2 실제 Kit 로그가 닫힌 순서를 더 좁혔다

D375의 원 Kit 로그 SHA-256은
`6522efde45e776fabf3186ddf362d509a6a3b04f999adc5024c28f41dce1ccc9`이며 D376 snapshot과
bit-exact다. 마지막 행은 다음 순서다.

- line 2282, `5.504s`: `SimulationApp.close: Closing application`
- line 2283, `5.504s`: Replicator shutdown finished
- line 2284, `5.522s`: Stage closed
- line 2285, `5.523s`: Simulation App Shutting Down
- line 2286, `5.523s`: `shutting down app and releasing framework`
- 그 뒤 로그가 없다.

근거:
`claudedocs/runtime_logs/grasp_track/g0a_d376/attempt1_d375_terminal_close_provenance_and_failure_visualization/d376_frozen_d375_kit_log.txt`.

외부 supervisor에서는 `900.0s` watchdog이 만료됐고, SIGTERM 뒤 등록된 `20.0s` grace에도
끝나지 않아 SIGKILL했다. 총 elapsed는 `920.3908159369603s`, return은 `-9`다.
worker cleanup-end부터 supervisor exit까지는 `914.65653135s`였다. 즉 확정 가능한 가장
좁은 경계는 **현재 stage가 닫힌 뒤 terminal framework-release/process-exit 경계**다.
네이티브 함수 안의 어느 thread/plugin이 막혔는지는 D375 thread dump가 없어 `null`이다.

원 수치와 program-order authority:
`claudedocs/runtime_logs/grasp_track/g0a_d376/attempt1_d375_terminal_close_provenance_and_failure_visualization/d376_terminal_close_provenance_evidence.json`.

## 4. NVIDIA 공식 자료와 설치 소스 교차검사

설치 스택은 Isaac Sim `5.1.0.0`, Isaac Lab `2.3.0`, SimulationApp extension `2.12.2`,
Kit `107.3.3`, Carbonite kernel `206.6`, Omni PhysX/schema `107.3.26`, NVIDIA driver
`580.159.03`, RTX 4090 Laptop GPU, compute capability `8.9`다.

1. NVIDIA **Isaac Sim 5.1.0 `isaacsim.simulation_app` API**는
   `close(skip_cleanup=False)`를 전체 정리를 하는 graceful shutdown,
   `skip_cleanup=True`를 immediate exit로 설명하고, launcher 기본값에서
   `fast_shutdown=True`를 명시한다.
   URL: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html
2. NVIDIA **Isaac Sim 5.1.0 Release Notes**는 `skip_cleanup` 추가와 stage-close/exit hang
   수리를 기록한다. 그러나 D375는 stage close 자체를 이미 통과했으므로 이 문구만으로
   이번 원인을 특정할 수 없다.
   URL: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html
3. NVIDIA **Isaac Sim 6.0.0 Release Notes**는 이후
   `shutdown_and_release_framework()`를 `app.shutdown()`으로 교체했으며, main thread가
   GIL을 가진 상태에서 `carb.tasking` worker가 기다리는 plugin teardown deadlock을 피하기
   위한 fix `5948099`라고 설명한다.
   URL: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html

로컬 설치 `simulation_app.py` SHA-256은
`7cbaa6f00e935a6f14bf1c28ec0db089fd924e931f3b0deee07a822f9b7d0090`이다.
`close()`는 line 763에서 시작하고, graceful path는 line 814에서 current stage를 닫은 뒤
line 838에서 `shutdown_and_release_framework()`를 부른다. `skip_cleanup=True` path도 line
793에서 같은 네이티브 함수를 부른다. D375 실행에는 이미 `fastShutdown=True`가 적용됐다.

따라서 다음 세 문장을 분리한다.

- **증명됨:** D375가 stage close 뒤 framework release/process exit 경계에서 끝나지 않았다.
- **가장 강한 메커니즘 후보:** 설치 5.1이 쓰는 같은 종료 함수에 대해 NVIDIA가 6.0에서
  공개한 GIL/`carb.tasking` teardown deadlock.
- **아직 증명 안 됨:** D375가 정확히 bug 5948099였다는 동일성. 6.0 자료는 later-version
  mechanism evidence이며 설치 5.1의 실제 native stack을 대신하지 않는다.

공식자료 attestation:
`claudedocs/runtime_logs/grasp_track/g0a_d376/attempt1_d375_terminal_close_provenance_and_failure_visualization/d376_nvidia_official_source_attestation.json`.

## 5. StageCache 가설의 정확한 지위

D375 worker는 custom in-memory stage를 `UsdUtils.StageCache`에 Insert하고 PhysX detach는 했지만
`StageCache.Erase(stage)`는 하지 않았다. 설치 Omni PhysX helper
`omni/physx/scripts/utils.py:582-599`의 `new_memory_stage`/`release_memory_stage` 쌍은
detach 뒤 `cache.Erase(stage)`까지 수행한다. 그래서 남은 유효 articulation object가 종료
교착을 촉발했을 가능성은 있다.

그러나 이것을 원인으로 확정하면 안 된다. D373도 Insert+detach 후 Erase하지 않았지만
`7.671346287010238s`, return `0`으로 정상 종료했다. 두 case의 차이는 D373의 articulation
property가 instance-proxy `ERROR_PARSING`이었던 반면 D375에는 non-instance VALID PhysX
object가 실제로 존재했다는 점이다. 그러므로 Erase 누락은 **조건부 촉발 후보**이고 단일
변수 인과 결과는 `null`이다.

## 6. “Isaac Sim이 계속 실패했다”가 아닌 이유

- D351 attempt1: Isaac과 corrected live binding `128/128`은 성공했으나 timeline pause
  선행조건이 false였다. attempt2는 `3693.302s` 뒤 사용자 승인 SIGTERM으로 끝났고 내부
  장기 실행 원인은 국소화되지 않았다.
- D367: worker가 `19.401926056s`, return `0`으로 정상 종료했다. 실패 표시는 우리
  post-close marker 계약의 오분류였다.
- D373: `7.671346287010238s`, return `0`으로 Isaac은 정상 종료했다. 실패는 Float32 비교,
  instance-proxy, traversal, supervisor identity 계약이었다.
- D375: raw/preclose PASS 뒤 실제 process non-exit가 외부 watchdog/signal로 증명된 유일한
  terminal shutdown case다.

따라서 서로 다른 네 현상을 “Isaac Sim 문제” 하나로 묶으면 수리 대상을 잘못 고르게 된다.

## 7. GPU/메모리 판단

D375 시작 전 GPU attestation은 RTX 4090 Laptop, driver `580.159.03`, free VRAM
`15465MiB`, 다른 Isaac process 없음이었다. Kit 로그에는 OOM 문자열이 없다. 이것은
**시작 용량이 건강했고 OOM 증거가 없었다**는 뜻이다. 종료 대기 900초 구간의 연속 GPU
telemetry는 없으므로 “GPU/VRAM 원인을 절대 배제했다”거나 Warp/SM 효율이 원인이라고
말할 수는 없다. 현재 로그의 마지막 경계와 NVIDIA 종료 결함 자료는 GPU 커널보다 CPU/native
plugin teardown 쪽을 훨씬 강하게 가리킨다.

## 8. 시각화와 육안검사

- 종료 순서 설명판: `d376_d375_terminal_close_timeline_1920x1080.png`, exact
  `1920x1080`, SHA-256 `36fa3111ec4250e1e866fe4680f0de9a1d036b40416d38113865728dc531a0a1`.
- 실패 분류 설명판: `d376_isaac_failure_classification_1920x1080.png`, exact
  `1920x1080`, SHA-256 `c32f39d9ef370bc850f7d0b773c6290ff18bd758f477543c9f4cd8e10b51cfdf`.
- save-only RRD/RBL: `63,882/43,641B`, SHA-256
  `08f181a6...d778b` / `c219c829...324f`; Rerun SDK/CLI `0.34.1`, footer, exact entity,
  timeline, component, embedded blueprint, exported RBL validation PASS.
- Rerun headless capture는 logical `1920x1080` 요청이 HiDPI 2배 physical `3840x2160` PNG가
  됐다. sandbox message-proxy 권한 경고가 보이지만 RRD load, required rows, screenshot
  return `0`은 통과했다. 정확한 1920×1080 발표 권위는 위 두 설명판이다.

세 이미지 모두 원본 해상도로 직접 검사했고, 두 설명판은 글자 겹침/잘림이 없으며 성공한
측정 구간과 실패한 종료 구간이 분리돼 보였다. Rerun 시간축의 1970 날짜 표시는 초보자에게
덜 직관적이므로 정확한 elapsed authority는 JSON과 설명판으로 유지했다.

근거:

- `claudedocs/runtime_logs/grasp_track/g0a_d376/attempt1_d375_terminal_close_provenance_and_failure_visualization/d376_rerun_validation.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d376/attempt1_d375_terminal_close_provenance_and_failure_visualization/d376_manual_visual_inspection.json`
- `claudedocs/runtime_logs/grasp_track/g0a_d376/attempt1_d375_terminal_close_provenance_and_failure_visualization/d376_completion_summary.json`

## 9. 결론과 다음 승인 경계

D376 PASS는 “D375의 측정은 끝났고 종료만 끝나지 않았다”는 경계, 서로 다른 과거 실패의
분류, 공식자료 기반 원인 후보, 실패 시각화를 완결했다. D375 effective identity는 false이고,
full classifier, physics equivalence/tipping, cylinder grasp feasibility, exact native blocker,
bug 5948099 exact identity는 모두 `null`; `g0a_pass=false`다.

다음 최소 후보는 아직 미승인
`D377 [d375_stagecache_erase_before_close_localization]`이다. D375의 성공한 측정 계약을
그대로 두고 PhysX detach 뒤 `StageCache.Erase(stage)` 한 변수만 추가하며, erase 전/후와
close 전/후 표식, one worker/no retry, bounded watchdog로 종료 여부만 판정한다. D373
counterexample 때문에 이는 “이미 아는 수리”가 아니라 조건부 촉발 가설의 단일 변수 시험이다.
D377이 lifecycle PASS해도 별도 full P34 classifier 승인이 필요하고, 그 PASS 전에는 A64/P34
원통 물리 비교로 가지 않는다.
