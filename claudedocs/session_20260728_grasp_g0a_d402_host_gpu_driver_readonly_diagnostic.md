# D402 후속 호스트 GPU/드라이버 읽기 전용 진단

Date: 2026-07-28 KST

## 범위

Isaac Sim, Kit, PhysX, USD, Worker, physics, q5, contact, cylinder를 실행하지
않고 NVIDIA 드라이버와 GPU 상태만 읽었다. 드라이버 재시작, 모듈 재적재,
장치 파일 생성, 재부팅, 설정 변경은 하지 않았다.

## 순차 검사 결과

1. `nvidia-smi`와 GPU query는 모두 exit code 9로 `couldn't communicate with
   the NVIDIA driver`를 반환했다.
2. 커널의 `nvidia`, `nvidia_drm`, `nvidia_modeset`, `nvidia_uvm` 모듈은
   적재되어 있었다.
3. `/proc/driver/nvidia/version`은 NVIDIA UNIX Open Kernel Module
   `580.173.02`를 보고했다.
4. `/proc/driver/nvidia/gpus/0000:01:00.0/information`과 PCI/udev는
   `NVIDIA GeForce RTX 4090 Laptop GPU`가 `nvidia`에 연결됐다고 보고했다.
5. 그러나 `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`은 모두 없었다.
   `/proc/devices`에는 해당 커널 장치 번호 등록은 남아 있었다.
6. `dmesg`는 이 세션의 권한으로 유의미한 NVIDIA 오류를 보여주지 않았고,
   systemd 상태 조회도 버스 권한 제한으로 확인하지 못했다.

## 판정

`D402_HOST_GPU_DRIVER_READONLY_DIAGNOSTIC_DEVICE_NODES_MISSING`.

드라이버 패키지나 커널 모듈이 완전히 없는 상태는 아니다. 커널과 PCI 계층은
GPU를 보지만, 사용자 공간이 CUDA/NVML을 열 때 필요한 장치 노드가 없어
`nvidia-smi`가 통신하지 못한다. 장치 노드 생성/udev 또는 드라이버 초기화가
불완전한 상태라는 데까지 좁혔지만, 이 읽기 전용 검사만으로 단일 원인을 확정하지는
않았다.

Isaac/Kit/PhysX/Worker/렌더링/물리/q5/contact/cylinder와 드라이버 재시작,
모듈 재적재, `mknod`, 재부팅은 모두 0회다.

## 다음 경계

D402 실제 경로는 동결한다. 호스트 복구 절차는 별도 승인 후에만 수행하며,
복구 뒤 `nvidia-smi` 정상 확인과 새 Git snapshot/tuple을 가진 새 runtime case가
필요하다. 같은 D402 출력 경로는 재실행하지 않는다.
