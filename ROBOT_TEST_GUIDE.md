# SmolVLA 실제 로봇 테스트 가이드

학습된 SmolVLA 모델을 RoArm M3 로봇에서 테스트하는 방법

---

## 전제 조건

### 하드웨어
- [x] RoArm M3 Pro 로봇 (USB 연결)
- [x] IMX335 카메라 (USB 연결)
- [x] 흰색 상자 (학습에 사용된 물체)
- [x] RTX 4070 SUPER GPU

### 소프트웨어
- [x] Python 가상환경 (`.venv`)
- [x] 학습된 체크포인트 (`outputs/smolvla_roarm_50ep/checkpoint.pt`)
- [x] LeRobot, PyTorch, OpenCV 설치됨

---

## Step 1: 환경 확인

### 1.1 가상환경 활성화
```powershell
E:\RoArm_Project\.venv\Scripts\activate
```

### 1.2 로봇 연결 확인
```powershell
# Device Manager에서 COM 포트 확인 (Silicon Labs CP210x)
# 보통 COM8 또는 COM3
```

### 1.3 카메라 연결 확인
```powershell
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL'); cap.release()"
```

### 1.4 체크포인트 확인
```powershell
python -c "import torch; ckpt = torch.load('outputs/smolvla_roarm_50ep/checkpoint.pt'); print('Checkpoint OK, Final Loss:', ckpt['training_info']['final_loss'])"
```

---

## Step 2: Dry-Run 테스트 (권장)

**실제 로봇 없이** 데이터셋 샘플로 추론 테스트:

```powershell
cd E:\RoArm_Project
python test_smolvla_inference.py --dry-run
```

### 예상 출력
```
SmolVLA 모델 로딩
============================================================
1. 데이터셋 로드 중...
   Stats keys: ['observation.images.top', 'observation.state', 'action', ...]

2. SmolVLA config 생성...

3. SmolVLA 모델 생성...
Reducing the number of VLM layers to 16 ...

4. 체크포인트 로드: outputs/smolvla_roarm_50ep/checkpoint.pt
   학습 정보:
   - Epochs: 50
   - Final Loss: 0.7928
   - Timestamp: 2026-02-04T22:54:57

5. 모델 준비 완료 (eval mode)

============================================================
Dry-Run 테스트 (데이터셋 샘플 사용)
============================================================

Task: 'Pick up the white box'
테스트 인덱스: [0, 50, 100, 150, 200]
------------------------------------------------------------

[Sample 0]
  입력 state: [ 0.79  31.73  -1.05  68.38   7.47   1.93]
  예측 action: [ 0.78  31.75  -1.04  68.40   7.48   1.94]
  실제 action: [ 0.79  31.73  -1.05  68.38   7.47   1.93]
  오차 (L2): 0.0234
  추론 시간: 850.3 ms
...
```

### Dry-Run 결과 해석
| 오차 (L2) | 의미 |
|-----------|------|
| < 1.0 | 매우 좋음 (오버피팅됨) |
| 1.0 - 5.0 | 양호 |
| > 5.0 | 문제 있음 |

---

## Step 3: 물리 환경 준비

### 3.1 카메라 위치 설정
```
학습 때와 동일한 카메라 위치/각도 필요!

권장 배치:
- 카메라: 로봇 위쪽/앞쪽에서 작업 영역 전체가 보이도록
- 높이: 약 30-50cm 위
- 각도: 아래를 향해 비스듬히
```

### 3.2 흰색 상자 배치
```
학습 때와 비슷한 위치에 배치:
- 로봇 팔이 닿을 수 있는 거리
- 카메라에서 잘 보이는 위치
- 로봇 기준 정면 또는 약간 옆
```

### 3.3 로봇 초기 위치
```
스크립트가 자동으로 [0,0,0,0,0,0] 위치로 이동
```

---

## Step 4: 실제 로봇 테스트

### 4.1 명령어
```powershell
cd E:\RoArm_Project
python test_smolvla_inference.py --robot-port COM8 --camera-id 0
```

### 4.2 매개변수
| 매개변수 | 기본값 | 설명 |
|----------|--------|------|
| `--robot-port` | COM8 | 로봇 시리얼 포트 |
| `--camera-id` | 0 | 카메라 인덱스 |
| `--checkpoint` | outputs/smolvla_roarm_50ep/checkpoint.pt | 모델 경로 |
| `--device` | cuda | 디바이스 (cuda/cpu) |

### 4.3 예상 동작 흐름
```
1. 모델 로드 (~30초)
2. 로봇 연결
3. 카메라 연결
4. 초기 위치로 이동
5. 추론 루프 시작 (Ctrl+C로 종료)
   - 이미지 캡처
   - 상태 읽기
   - 액션 예측
   - 로봇 제어
   - 반복 (~10Hz)
6. 종료 시 초기 위치 복귀
```

### 4.4 실행 중 출력
```
실제 로봇 추론 테스트
============================================================

1. 로봇 연결 중 (Port: COM8)...
   로봇 연결 성공!

2. 카메라 연결 중 (ID: 0)...
   카메라 연결 성공!

3. 로봇 초기 위치로 이동...

4. 추론 시작
   Task: 'Pick up the white box'
   [Ctrl+C로 종료]
------------------------------------------------------------
   [Step   1] Action: [   0.5,   32.1,   -0.9,   68.5,    7.5,    2.0] (892ms)
   [Step   2] Action: [   0.8,   33.4,   -0.8,   67.2,    7.6,    3.5] (856ms)
   [Step   3] Action: [   1.2,   35.7,   -0.6,   65.8,    7.8,    5.2] (871ms)
   ...
```

---

## Step 5: 결과 해석

### 성공 시나리오
```
로봇이:
1. 흰색 상자 방향으로 팔을 뻗음
2. 상자 위치로 접근
3. 그리퍼를 열고 닫음 (집기 시도)
```

### 실패 시나리오
| 증상 | 원인 | 해결 |
|------|------|------|
| 팔이 이상한 방향으로 움직임 | 카메라 위치가 학습 때와 다름 | 카메라 재배치 |
| 상자를 못 찾음 | 상자 위치가 학습 때와 다름 | 상자 재배치 |
| 동작이 느림 | 추론 속도 문제 | GPU 확인 |
| 팔이 떨림 | 노이즈 | speed/acc 조정 |

### 한계점 (중요!)
```
⚠️ 1 에피소드 오버피팅 모델의 한계:

- 학습 때와 완전히 같은 조건에서만 동작
- 카메라 각도 조금만 달라도 실패
- 상자 위치 조금만 달라도 실패
- 조명 변화에 민감

→ 데모 목적으로만 사용 가능
→ 실용적 사용을 위해 50+ 에피소드 필요
```

---

## Step 6: 문제 해결

### 6.1 로봇 연결 실패
```powershell
# COM 포트 확인
# Device Manager → Ports (COM & LPT) → Silicon Labs CP210x

# 다른 포트 시도
python test_smolvla_inference.py --robot-port COM3
```

### 6.2 카메라 연결 실패
```powershell
# 카메라 확인
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# 다른 카메라 ID 시도
python test_smolvla_inference.py --camera-id 1
```

### 6.3 CUDA 메모리 에러
```powershell
# CPU 모드로 실행 (느림)
python test_smolvla_inference.py --device cpu --dry-run
```

### 6.4 추론이 너무 느림
```
예상 추론 시간:
- GPU (RTX 4070): ~800-1000ms/step
- CPU: ~3000-5000ms/step

개선 방법:
- GPU 사용 확인
- 다른 프로그램 종료
- 이미지 해상도 줄이기 (코드 수정 필요)
```

### 6.5 로봇 모터 응답 없음
```powershell
# 서보 스캔으로 리셋
python scan_servos.py COM8
```

---

## Step 7: 다음 단계

### 더 나은 결과를 위한 개선
1. **데이터 수집**: Leader-Follower 모드로 50+ 에피소드 수집
2. **재학습**: 더 많은 데이터로 SmolVLA 학습
3. **평가**: 다양한 조건에서 성공률 측정

### 데이터 수집 명령어 (향후)
```powershell
# Leader-Follower 모드로 데이터 수집
python lerobot/scripts/control_robot.py \
  --robot.type=roarm_m3 \
  --robot.leader_arms='{"main": "COM9"}' \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the white box" \
  --control.repo_id=roarm_m3_grasping \
  --control.num_episodes=50
```

---

## 빠른 참조

### 명령어 요약
```powershell
# 1. 환경 활성화
E:\RoArm_Project\.venv\Scripts\activate

# 2. Dry-run (먼저 실행!)
python test_smolvla_inference.py --dry-run

# 3. 실제 로봇 테스트
python test_smolvla_inference.py --robot-port COM8 --camera-id 0

# 4. 종료: Ctrl+C
```

### 체크리스트
- [ ] 가상환경 활성화됨
- [ ] 로봇 USB 연결됨 (COM 포트 확인)
- [ ] 카메라 USB 연결됨
- [ ] 흰색 상자 준비됨
- [ ] 카메라 위치 설정됨
- [ ] Dry-run 테스트 통과
- [ ] 실제 로봇 테스트 준비 완료

---

*Generated: 2026-02-04*
