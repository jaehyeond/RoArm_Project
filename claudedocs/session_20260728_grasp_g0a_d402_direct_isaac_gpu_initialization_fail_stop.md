# D402 후속 직접 Isaac Sim GPU 초기화 진단

Date: 2026-07-28 KST

## 범위

기존 D402 출력 경로를 재실행하지 않고, 설치된 Isaac Sim 5.1.0 Python을 최소
`SimulationApp(headless=True)`로 1회 기동했다. 앱이 초기화되면 즉시 닫도록 했고,
Worker/asset/USD/PhysX scene/physics/q5/contact/cylinder는 실행하지 않았다.

## 순차 결과

1. Kit 로그가 생성되고 Isaac Sim 확장들이 로드되었다. 앱은 약 6.9초 후
   startup complete까지 도달했다.
2. 시작 초기에 `omni.cudainterop`가 `NVML_ERROR_DRIVER_NOT_LOADED`를 두 번
   보고했다. 표시된 Driver Version은 `0`이었다.
3. GPU Foundation이 장치를 만들지 못했고 `Failed to create any GPU devices`
   를 보고했다.
4. PhysX tensors와 Warp가 각각 `no CUDA-capable device is detected`와
   CUDA initialization error를 보고했다.
5. SimulationApp은 예외 없이 반환되어 종료했지만, 이는 GPU가 정상이라는
   뜻이 아니다. 렌더러/ CUDA/PhysX GPU 경로가 초기화되지 않은 상태에서 앱
   셸만 올라온 것이다.

## 부수 관찰

- `omni.datastore`의 DerivedDataCache 독점 lock 실패와 KVDB의 다른 Kit lock
  경고가 있었다.
- Material Library는 설치 경로 cache가 read-only라 `OSError(30)`을 냈다.
- 현재 `ps` 재확인에서는 살아 있는 Isaac/Kit/Omniverse 프로세스가 없었다.

## 판정

`D402_DIRECT_ISAAC_GPU_INITIALIZATION_FAIL_STOP`.

직접 실행으로 원인이 확정됐다. Isaac Sim 코드나 RoArm 충돌체가 먼저 실패한
것이 아니라, NVML이 드라이버를 열지 못해 GPU Foundation, CUDA, PhysX GPU,
Warp가 모두 초기화되지 않았다. 앱의 종료 코드 0은 GPU 성공을 의미하지 않는다.

이번 실행은 과학/물리 실험이 아니며 `scientific_or_physics_verdict=null`,
`g0a_pass=false`를 유지한다.

## 다음 경계

호스트 NVIDIA 장치 노드/드라이버 복구가 먼저다. 복구 전에는 Isaac 기반 연구를
재시작하지 않는다. 복구 후 `nvidia-smi`와 최소 Isaac 기동에서 GPU 이름·Driver
Version·CUDA 장치가 확인되어야 하며, 그 뒤 새 Git snapshot과 새 tuple을 가진
별도 runtime case 승인이 필요하다.
