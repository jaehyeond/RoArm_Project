# D402 actual runtime attempt 1 — GPU preflight fail-stop

Date: 2026-07-28 KST

## 1. 무엇을 왜 실행했나

사용자가 승인한 D402 tuple을 사용해 Controller를 정확히 1회 실행했다. 목적은
D401에서 발견된 두 harness 수리(Item 버전 읽기, counter 권위)를 실제 실행 직전에
검증하고, 통과할 때만 Worker가 고정된 D400 SDF asset load/cook/readback으로
진입하게 하는 것이었다. 물리, q5, 접촉, 원통, public forward는 사전등록상 금지였다.

## 2. 실행 순서와 관찰

1. 승인 tuple SHA `898c9155...012ce`와 Git 기준을 재확인했다.
2. Controller가 runtime-freeze manifest를 만들었고 Git/tuple/설치 소스 해시 게이트는
   통과했다 (`d400_runtime_freeze_manifest.json`).
3. 다음 GPU/기존 프로세스 게이트에서 중단됐다. `minimum_free_vram`, `gpu_model_exact`,
   `gpu_compute_capability_exact`가 모두 false였고, `d400_process_conflicts=0`이었다.
4. `nvidia-smi --query-gpu=...`도 “couldn't communicate with the NVIDIA driver”로
   실패했다. 따라서 GPU 정보를 읽지 못한 상태에서 fail-closed 된 것이다.
5. Worker spawn request/claim은 false, 실제 Worker/Isaac/Kit/PhysX 실행은 0이다.
   신호 전송과 정리 대상 프로세스도 0이다.

## 3. 원본 증거

- phase 순서와 중단 단계: `.../d400_phase_markers.jsonl`
- supervisor 권위: `.../d400_worker_supervisor.json`
- 최종 요약: `.../d400_completion_summary.json`
- 산출물 SHA-256은 이 폴더의 파일 목록과 함께 보존했다.

## 4. 판정

`D402_RUNTIME_STACK_ITEM_AND_COUNTER_ORDER_AUTHORITY_REPAIR_RUNTIME_GPU_PREFLIGHT_FAIL_STOP`.
이는 D402 코드 수리 실패도, Isaac/PhysX/SDF 실패도, 과학/물리 결과도 아니다. NVIDIA
드라이버가 현재 프로세스에서 보이지 않아 Worker 이전의 안전 게이트가 실행을 차단한
인프라 fail-stop이다. `scientific_or_physics_verdict=null`, `g0a_pass=false`를 유지한다.

## 5. 다음 경계

D402 attempt1 경로는 immutable이며 같은 경로 재실행·재시도하지 않는다. 먼저 별도
승인된 읽기 전용 호스트 GPU/드라이버 가용성 진단과 복구가 필요하다. 복구 뒤에도
새 Git snapshot과 새 forward-only tuple을 발급한 별도 runtime case가 필요하다.

이번 세션은 AGENTS.md의 “실패할 수 있는 실험 또는 왜 실행하지 않았는지 명시” 규칙에
따라 실제 실행을 시도했으며, Worker 이전 fail-stop으로 과학 측정은 수행하지 않았다.
