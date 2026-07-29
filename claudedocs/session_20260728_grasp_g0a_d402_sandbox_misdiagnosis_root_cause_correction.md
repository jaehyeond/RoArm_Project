# D402 GPU fail-stop 근본원인 정정 — 샌드박스 오진 (호스트 드라이버 정상)

Date: 2026-07-28 오후 KST. 이 문서는 같은 날짜의 세 문서
(`session_20260728_grasp_g0a_d402_actual_runtime_gpu_preflight_fail_stop.md`,
`..._host_gpu_driver_readonly_diagnostic.md`,
`..._direct_isaac_gpu_initialization_fail_stop.md`)의 **인프라 결론을 정정**한다.
세 문서는 append-only 원칙으로 수정하지 않고 보존한다. 과학 결론(없음)은 불변.

## 정정 요지

15:01/15:11/15:18의 "NVIDIA driver 통신 불가 / /dev/nvidia* 부재 / NVML
DRIVER_NOT_LOADED"는 **호스트 고장이 아니라, 해당 세션의 명령이 Claude Code
Bash 샌드박스(사설 mount namespace의 /dev에 GPU 노드 없음 + 사설 PID
namespace) 안에서 실행된 관측**이었다. 호스트 복구는 필요 없었고 수행되지도
않았다 — 검증 후 그대로 정상이었다.

## 확정 증거 (4-agent 적대적 교차검증, 반박 실패)

1. **PID 2 스모킹건**: 실패한 D402 controller가 phase marker에 자기 pid를
   `2`로 기록 (`g0a_d402/attempt1_*/d400_phase_markers.jsonl`). 호스트 PID 2는
   kthreadd(커널 스레드)라 유저 프로세스 불가 → 사설 PID namespace 확정.
   성공한 D401 run은 호스트 PID 240041/240215 기록.
2. 호스트 `/dev/nvidia0|nvidiactl|nvidia-uvm` Birth=Change=13:32:11~12
   (부팅 13:32:07 직후), 이후 재생성 0회 (stat, devtmpfs).
3. Xorg PID 2641이 13:33부터 `/dev/nvidiactl`+`/dev/nvidia0`를 연 채 실패
   구간 내내 GPU 사용; nvidia-persistenced 13:32:12부터 연속 active; 커널
   저널 NVRM 오류/unload/Xid 0건.
4. 같은 부팅 14:13 D401 kit log: `cuda:0`, driver `580.173.02`, RTX 4090
   (`g0a_d401/attempt1_*/d400_kit_log.txt:2,20,26`).
5. 15:18 실행의 부수 증상(설치폴더 EROFS OSError30 + 같은 파일시스템의
   프로젝트 폴더 쓰기 성공 + dmesg/systemd 차단)은 경로별 bind-mount 샌드박스
   서명 — 호스트 단일 파일시스템에서 불가능한 ro/rw 분리.
6. D402 preflight 게이트: 파일시스템 pin 14/14 True, GPU 조회 3개만 False —
   파일은 보이고 장치만 안 보이는 샌드박스 프로파일.
7. 15:33 호스트 경계 최소 `SimulationApp(headless=True)` 재실행: GPU
   Foundation driver 580.173.02/Vulkan/RTX4090, torch CUDA True, Warp cuda:0
   sm_89, NVML/CUDA 오류 0. (한계: `app.close()`가 셰이더 캐시 재컴파일 후
   futex 대기로 hang → SIGTERM 정리, GPU 상태 청정 확인. 렌더러 device 생성과
   클린 종료는 이 최소실행에서 미검증 — D401 실사용 경로는 헤드리스 kit으로
   정상 종료 전례 있음.)

## 지난주 "자꾸 터진" 실체 (타임라인 감사)

- **실고장 1건뿐**: 7/25 06:46 unattended-upgrade가 **가동 중**(7/6 부팅
  유지) nvidia userspace+dkms를 580.159.03→580.173.02 교체 → 메모리의 커널
  모듈(159.03)과 skew → **7/25 06:47~7/28 04:02 (~69h) CUDA/NVML 실불능**.
  7/28 04:03 재부팅으로 해소.
- 7/28 06:09 unattended-upgrade가 kernel 6.8.0-136 설치(dkms 06:10 빌드) →
  사용자 재부팅 클러스터 13:12/13:13/13:30/13:32로 채택. 이후 호스트 연속 정상.
- 오늘 15:01~15:18 실패 3건 = 위 샌드박스 관측 (호스트 무관).

## 교훈 (DECISIONS D402-R1로 등재)

GPU preflight/진단은 반드시 **호스트 경계**(GPU device node가 보이는
비샌드박스 셸)에서 실행한다. 샌드박스 내 `nvidia-smi` 실패 ≠ 호스트 고장.
산출물에 기록된 자기 PID가 한 자릿수면 namespace 실행 증거다. D401 세션 문서
57-59행이 이미 이 메커니즘을 기록했었다 — D402 세션이 이를 놓친 것이 오진의
직접 원인.

## 검증 산출물

- 4-agent 검증 journal: `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/cfceb787-c5e1-46ac-bc3e-1a26fc41bc07/subagents/workflows/wf_528ea7fc-097/journal.jsonl`
- 최소 Isaac 로그(세션 scratchpad, 휘발): 핵심 수치는 본 문서에 전사됨.
