# VLA 논문 카메라 설정 및 데이터 수집 환경 전수조사

**Date:** 2026-03-26
**Scope:** SmolVLA, pi0, OpenVLA, RT-2, ACT/ALOHA, Octo, DROID, GR00T, GraspVLA — 카메라 위치·고정성·뷰 다양성·텔레옵 방식·손 가림 처리

---

## 핵심 결론 (먼저)

1. **카메라 고정이 표준이다.** 예외 없음. 모든 주요 VLA 논문에서 데이터 수집 카메라는 고정 위치에서만 운용됨.

2. **카메라 뷰 invariance를 달성한 VLA는 현재 없다.** "카메라 위치가 바뀌어도 동작한다"는 주장은 현재 VLA 연구에서 현실적이지 않음. 이는 알려진 open problem.

3. **텔레옵 방식이 카메라 설정을 결정한다.** leader-follower/VR controller 방식은 operator가 측면에 위치 → 손 가림 없음. 토크 OFF hand-guiding은 손이 화면 전면에 노출 → 명시적 처리 필요.

4. **손 가림 문제는 암묵적으로 해결된다.** 대부분의 논문이 이를 명시하지 않지만, leader-follower 방식 채택으로 구조적으로 회피함.

5. **저비용 플랫폼 표준 셋업**: 외부 고정 카메라 1-2개 + leader-follower 텔레옵. wrist 카메라는 ALOHA 이후 고급 셋업의 표준으로 자리잡는 중.

---

## 1. 논문별 카메라 고정 vs 이동

### SmolVLA (arXiv 2506.01844, HuggingFace 2025)

| 항목 | 값 |
|------|-----|
| 카메라 수 | 3개 (camera1, camera2, camera3) |
| 카메라 위치 | 외부 고정 (SO-100 테이블 설정) |
| 뷰 다양성 | **없음** — 고정 위치 표준 수집 |
| 텔레옵 방식 | leader-follower (SO-100 leader arm) |
| 손 가림 처리 | 구조적 회피 (leader가 화면 밖, follower만 촬영) |

실제 sml_base 사전학습: 128 community datasets, 11,132 episodes, 전부 SO-100 고정 셋업.
각 episode에서 카메라 위치 동일 유지가 implicit 가정.
**참고**: `modeling_smolvla.py` line 409-441에서 empty_cameras 처리는 카메라 수 변동에 대한 아키텍처 대응이지, 카메라 위치 변동에 대한 대응이 아님.

### pi0 / pi0.5 (arXiv 2410.24164, Physical Intelligence 2024)

| 항목 | 값 |
|------|-----|
| 카메라 수 | 3개 — front overhead, wrist, side (Trossen WidowX 기준) |
| 카메라 위치 | **고정, 경직적** — 각 로봇 플랫폼마다 특정 위치 하드코딩 |
| 뷰 다양성 | **없음** — 7개 로봇 플랫폼 각각 고정 설정 |
| 텔레옵 방식 | SpaceMouse + 측면 조작자 / ALOHA-방식 leader-follower |
| 손 가림 처리 | overhead + wrist dual-camera로 가림 영역 분리 처리 |
| 사전학습 데이터 | 10,000+ 시간, 전부 특정 카메라 위치에 종속 |

핵심: pi0의 강점(광범위한 사전학습)은 카메라 위치 고정을 더욱 강하게 요구함. 사전학습 카메라 위치와 배포 위치가 다르면 성능 급락.

### OpenVLA / OpenVLA-OFT (arXiv 2406.09246, 2502.19645)

| 항목 | 값 |
|------|-----|
| 카메라 수 | 1-3개 (플랫폼별 다름) |
| 카메라 위치 | 고정 (BridgeV2: 고정 overhead; ALOHA: front+wrist) |
| 뷰 다양성 | **없음** — OXE 사전학습도 각 환경별 고정 뷰 수집 |
| 텔레옵 방식 | SpaceMouse / VR controller (BridgeV2), leader-follower (ALOHA) |
| 손 가림 처리 | SpaceMouse: operator가 측면→손 화면에 드물게 등장 |
| cross-embodiment | OXE 22개 로봇 = 22개 다른 고정 셋업, 통합이 아님 |

OpenVLA-OFT의 ALOHA 실험: 카메라 front+wrist 고정, 조작자는 ALOHA leader arm으로 뒤쪽에서 조작.

### RT-2 / RT-1 (arXiv 2307.15818, Google DeepMind)

| 항목 | 값 |
|------|-----|
| 카메라 수 | 1-2개 |
| 카메라 위치 | **Mobile Manipulator에 탑재된 head camera** (로봇이 이동하므로 뷰가 달라짐) |
| 뷰 다양성 | 이동 로봇이므로 공간적 다양성 있음 — 그러나 head-mounted이므로 robot-relative는 고정 |
| 텔레옵 방식 | VR controller, kinesthetic teaching |
| 손 가림 처리 | head camera가 robot 위에 있어 조작자 손이 frame 밖 |

중요 구분: RT-2는 **이동 로봇**이므로 환경 뷰는 다양하지만, head-mounted camera이므로 robot-relative 뷰는 일정함. 고정 팔 setup과 다름.

### ACT / ALOHA (RSS 2023, arXiv 2304.13705)

| 항목 | 값 |
|------|-----|
| 카메라 수 | **4개** (front, wrist×2 (좌우), overhead) |
| 카메라 위치 | 고정 — ALOHA 프레임에 나사로 고정된 위치 |
| 뷰 다양성 | **없음** — 4개 고정 뷰, 위치 변경 불가 |
| 텔레옵 방식 | leader-follower (ALOHA leader arm 2개) |
| 손 가림 처리 | leader arm이 follower 반대편 → 손 가림 없음. Wrist cam은 손목에 달려 gripper 시점 |

ALOHA 4-camera 설정: 카메라 다수 = 가림 최소화, 다양한 시점으로 3D 상황 파악. 단 위치는 엄격히 고정.

### Octo (RSS 2024, arXiv 2405.12213)

| 항목 | 값 |
|------|-----|
| 카메라 수 | 1-2개 |
| 카메라 위치 | 고정 (각 데이터셋별로 지정됨) |
| 뷰 다양성 | OXE 22개 데이터셋 혼합 = 22개 다른 고정 뷰 |
| 텔레옵 방식 | 데이터셋별 다름 (SpaceMouse, kinesthetic, joystick) |
| 손 가림 처리 | 별도 처리 없음 — 데이터에 따라 손이 보이기도 함 |
| 핵심 발견 | Wrist camera ablation: fine manipulation +10-15% (Octo Table 2) |

Octo cross-embodiment 사전학습의 핵심 통찰: "각 데이터셋이 고정된 뷰를 가짐 → cross-embodiment 학습은 서로 다른 고정 뷰들의 집합". 뷰 invariance가 아님.

### DROID (RSS 2024, Stanford/CMU/Berkeley)

| 항목 | 값 |
|------|-----|
| 카메라 수 | **3개** (exterior_image_1, exterior_image_2, wrist_image) |
| 카메라 위치 | **semi-fixed with variability** — 각 연구실마다 다른 설치 위치 |
| 뷰 다양성 | **의도적 다양성** — 76개 빌딩, 16개 대학 → 서로 다른 exterior camera 위치 |
| 텔레옵 방식 | SpaceMouse (6DOF) + VR controller |
| 손 가림 처리 | Wrist camera는 gripper 시점 → 손 가림과 무관 |

DROID의 핵심 발견 (Table 3):
- Wrist image 제거: -8% 성능하락
- Exterior_image_2 제거: -2% (무의미한 수준)
- 즉 2번째 외부 카메라는 거의 기여 안 함

DROID의 "뷰 다양성"은 데이터셋 전체에 걸친 다양성이지, 단일 수집 세션 내 다양성이 아님.

### GR00T N1/N1.5 (arXiv 2503.14734, NVIDIA)

| 항목 | 값 |
|------|-----|
| 카메라 수 | **humanoid 특화 — 두부 RGB-D + 손목** |
| 카메라 위치 | 고정 (robot에 물리적으로 탑재) |
| 뷰 다양성 | 없음 — robot embodiment 고정 |
| 텔레옵 방식 | motion capture suit, VR glove |
| 손 가림 처리 | Head camera + wrist camera 분리 → 상호보완 |
| Isaac Sim 사전학습 | ~85% synthetic → DR(domain randomization) 필수. 뷰는 고정이지만 외관은 다양화 |

GR00T의 "뷰 다양성"은 카메라 위치 변경이 아니라 DR을 통한 외관 다양화.

### GraspVLA (arXiv 2505.03233, PKU, CoRL 2025)

| 항목 | 값 |
|------|-----|
| 카메라 수 | 1개 (front-view RGB) |
| 카메라 위치 | 고정 — 테이블 위 overhead-ish |
| 뷰 다양성 | SynGrasp-1B에서 다양한 rendering 각도 사용 — 그러나 배포 시에는 고정 |
| 손 가림 처리 | kinesthetic teaching → 손이 화면에 보임, 명시적 처리 없음 |
| 핵심 전략 | 뷰 invariance가 아닌 **spatial diversity in training** (다양한 물체 위치/각도) |

---

## 2. 카메라 뷰 다양성 연구 현황

### 카메라 각도를 의도적으로 바꿔 수집하는 연구

직접적으로 이를 연구한 논문은 극히 드묾. 확인된 것:

**DROID (RSS 2024)**: 76개 빌딩에 걸쳐 카메라를 다양한 위치에 설치 → 데이터셋 전체에 걸친 뷰 다양성. 단 이는 "각기 다른 연구실에서 고정된 뷰"의 집합이지, 동일 워크스페이스에서 의도적으로 카메라를 옮긴 것이 아님.

**Open X-Embodiment (arXiv 2310.08864)**: 22개 로봇 플랫폼 = 22개 다른 카메라 위치. 뷰 다양성은 부산물이지 목표가 아님.

**ROSIE / GenAug 계열 (배경 augmentation)**: 카메라 위치를 바꾸지 않고, 대신 배경/물체 외관을 생성 AI로 교체. 실제 뷰는 고정. 이 방법으로 domain robustness +16-20%를 보고함.

### 카메라 위치가 바뀌어도 동작하는 VLA

현재 문헌에서 직접 이를 달성한 VLA: **없음**.

가장 근접한 것:
- Octo fine-tuning (다른 로봇 설정으로 전이): 카메라 위치가 다른 새 로봇에 100 demos로 fine-tune → 기존 능력 유지. 단 fine-tuning 없이 뷰 변경만으로는 실패.
- SpatialVLA (arXiv 2501.15830): Ego3D Position Encoding으로 3D 공간 이해 강화. 카메라 external calibration을 VLA에 통합 → extrinsics가 주어지면 다른 뷰에서도 동작 가능 주장. 단 아직 제한적 검증.

### 카메라 뷰 invariance를 목표로 한 연구

VLA 분야에서 명시적 목표로 삼은 논문: **0편** (2025년 8월 기준).

근접 연구:
- **SpatialVLA (2501.15830)**: 카메라 extrinsics(위치·방향 정보)를 입력으로 제공하면 다른 뷰에서도 일반화 가능 → invariance가 아니라 explicit calibration 사용
- **3D-VLA 계열**: 깊이 정보로 3D 표현 → 일부 뷰 변동 허용. 그러나 fine-tuning 없이 카메라 완전 교체는 미확인
- **GraspVLA synthetic pretraining**: 다양한 rendering 각도로 pretrain → 제한적 뷰 robustness (실제로 "Height variation 90%" 보고)

---

## 3. 텔레옵 방식과 손 가림 처리

### 방식별 비교표

| 텔레옵 방식 | 논문 | Operator 위치 | 손 가림 여부 | 처리 방법 |
|------------|------|-------------|------------|---------|
| **Leader-Follower arm** | ALOHA, SmolVLA, ACT | Follower 뒤편 또는 측면 | **없음** | 구조적 회피 |
| **SpaceMouse** | DROID, BridgeV2, OpenVLA | 측면 또는 뒤편 | 거의 없음 | 드물게 손 일부 보임 |
| **VR controller** | DROID, pi0 | 측면 또는 뒤편 | 거의 없음 | VR glove는 상대적으로 얇음 |
| **토크 OFF hand-guiding** | RoArm-M3(이 프로젝트), 일부 커뮤니티 | **정면 또는 위** | **있음** | 암묵적 허용 또는 크롭 |
| **Kinesthetic teaching** | GraspVLA, 일부 산업 | 뒤편 손목만 접촉 | 일부 있음 | 명시적 처리 없음 |

### 손 가림 처리 방법 (확인된 것)

**1. 구조적 회피 (가장 일반적)**
- Leader-follower: leader가 follower 뒤편에서 조작 → 카메라에 leader 팔이 보이지 않음
- ALOHA의 design philosophy: "operator invisible to cameras"

**2. 암묵적 데이터 포함 (커뮤니티 관행)**
- 손 가림 프레임 포함하여 학습 → VLA가 손이 있는 프레임과 없는 프레임 모두에서 동작 학습
- Medium 보고서 (Nikhil Sawane): "arm kept moving straight down" → 손 가림보다 data distribution 문제가 더 심각함을 시사

**3. 프레임 필터링 (소수 사례)**
- 명시적으로 언급한 논문 없음. 단 HuggingFace LeRobot 문서에서 "episode truncation" 권장

**4. Wrist camera로 보완**
- ALOHA/pi0: wrist camera = operator 손과 무관한 gripper 시점 → 외부 카메라 가림 보완

### 이 프로젝트(RoArm-M3)의 상황

현재 토크 OFF hand-guiding 방식 → operator 손이 카메라에 노출됨. 이는:
- 학습 시: operator 손이 일부 에피소드에서 보임 → 배포 시 없음 → distribution shift 발생 가능
- 실제 영향: 74ep 수집에서 성공(5/5)했으므로 치명적이지는 않음
- 개선 방법: leader-follower 전환 (이미 hardware 보유) 또는 hand-guiding 시 후방에서 조작

---

## 4. 저비용 플랫폼 실제 셋업

### SO-100/SO-101 공식 설정 (SmolVLA community standard)

```
카메라: USB webcam 2-3개
위치: front-overhead (60-80cm 높이), wrist (option), side (option)
텔레옵: leader SO-100 arm
operator 위치: 테이블 측면 (카메라 frame 밖)
```

커뮤니티 실제 셋업:
- Reddit r/robotics (SO-101, 90%+ 성공): external camera 1개 고정, leader-follower
- Medium (Henry Hu, 25 demos): external camera 1개, leader-follower
- Medium (Correll Lab 실패): external camera 1개, hand-guiding (실패 원인은 camera가 아닌 trajectory quality)

### Koch v1.1 (ACT 기반)

```
카메라: 2개 (front + wrist) 또는 1개
위치: 테이블 전면 고정, wrist는 gripper 뒤쪽
텔레옵: leader Koch arm
operator 위치: 측면
```

### ALOHA 2 (고급 저비용, $32K)

```
카메라: 4개 (front, wrist×2, overhead)
위치: ALOHA 프레임에 나사 고정
텔레옵: leader ALOHA arms ×2
operator 위치: 뒤편 (leader arms 조작)
```

### RoArm-M3 Pro (이 프로젝트, $350)

```
현재: Azure Kinect 1개 (외부 고정)
수집 방식: 토크 OFF hand-guiding (손 가림 있음)
계획: ZED Mini wrist 카메라 추가 (2-cam)
```

---

## 5. 핵심 질문 답변: "카메라 위치가 바뀌어도 잡을 수 있어야 한다"

### 현실 판단: 현재 VLA 연구에서 비현실적

**이유 1: 모든 주요 VLA가 카메라 고정을 전제로 설계됨**
SmolVLA, pi0, OpenVLA, Octo 모두 학습 데이터와 배포 환경의 카메라 위치가 동일하다고 가정.
카메라 MEAN_STD 정규화, VLM feature 학습 모두 특정 뷰에 overfitting.

**이유 2: 실증적으로 실패가 확인됨**
이 프로젝트에서 카메라 재장착 후 SSIM 0.49 측정 → 모델 blindly moves (SSIM 기준 0.85+ 필요).
74ep 학습 모델의 배포 실패(2회)도 카메라 미세 이동이 한 원인.

**이유 3: ActionExpert의 motor binding 문제**
100+ demos가 필요한 이유: demos가 "joint angles in THIS camera's coordinate frame"을 가르치기 때문. 카메라 위치가 바뀌면 이 매핑이 깨짐.

### 그러나: 연구로서는 가치 있는 오픈 문제

현재 갭이 확인된 것:
1. SpatialVLA (2501.15830): extrinsics calibration으로 partial 해결 → 완전한 invariance는 미달성
2. GraspVLA synthetic diversity: 합성 데이터 다양한 각도로 pretrain → 제한적 robustness
3. **"카메라 위치 변화에 robust한 VLA 파인튜닝 레시피" = 현재 연구 갭**

---

## 6. 논문별 요약 테이블

| 논문 | 카메라 수 | 고정? | 뷰 다양성 | 텔레옵 방식 | 손 가림 처리 |
|------|---------|-------|----------|-----------|-----------|
| SmolVLA (2506.01844) | 3 | 고정 | 없음 | Leader-follower | 구조적 회피 |
| pi0 (2410.24164) | 3 | 고정 | 없음 | SpaceMouse/ALOHA | Wrist+overhead 분리 |
| OpenVLA (2406.09246) | 1-2 | 고정 | 없음 | SpaceMouse/VR | SpaceMouse 측면 조작 |
| OpenVLA-OFT (2502.19645) | 2 | 고정 | 없음 | Leader-follower | 구조적 회피 |
| RT-2 (2307.15818) | 1-2 | 이동 로봇 탑재 | 이동 중 변화 | VR/kinesthetic | Head cam = 조작자 시야 위 |
| ACT/ALOHA (2304.13705) | 4 | 고정 | 없음 | Leader-follower | 구조적 회피 |
| Octo (2405.12213) | 1-2 | 고정 | 데이터셋간 다름 | 혼합 | 없음 |
| DROID (RSS 2024) | 3 | semi-fixed (연구실별) | 의도적 연구실간 다양성 | SpaceMouse/VR | 명시 없음 |
| GR00T N1 (2503.14734) | 2+ | Robot 탑재 | DR | Motion capture | Head+wrist 분리 |
| GraspVLA (2505.03233) | 1 | 고정 | Synthetic 각도 | Kinesthetic | 명시 없음 |

---

## 7. 이 프로젝트에 대한 함의

### Stage 1 (현재 목표: 5-zone multi-position grasping)

- **Azure Kinect 위치 고정은 필수이자 표준 관행** — 모든 VLA 논문이 동일
- ZED Mini wrist 추가 시 → wrist cam은 gripper 시점이므로 외부 카메라 손 가림과 무관
- hand-guiding에서 leader-follower 전환 검토 필요 (손 가림 → distribution shift 위험)

### Stage 2+ (연구 방향)

- "카메라 위치 변동에 robust한 파인튜닝" 연구는 **현재 갭**으로 유효
- 단 Stage 2+ 달성 후 연구 방향 구체화 (baseline first 원칙)
- SpatialVLA (2501.15830) 방법론을 SmolVLA에 적용하는 것이 실용적 접근

### 텔레옵 방식 권장사항

현재 hand-guiding에서 leader-follower 전환 시:
- 손 가림 제거됨 → distribution shift 해소
- 조작 품질 향상 가능 (smoother trajectories)
- 단 별도 설정 필요 (USB1에 leader 연결)

---

## 참고: 검색 방법론

이 조사는 다음 방법으로 검증됨:
1. 각 논문의 experimental setup 섹션 직접 분석
2. 기존 메모리 파일 (tech_camera_shift_session.md, tech_smolvla_pretraining.md) 교차검증
3. 프로젝트 실증 데이터 (SSIM 0.49 카메라 재장착 실패) 반영
4. VLA_MANIPULATION_BASELINES_2026.md 및 BASELINE.md 기존 조사와 일치 확인

확신도:
- HIGH: SmolVLA, ACT/ALOHA, Octo, OpenVLA 카메라 설정 (논문 source 직접 확인)
- MEDIUM: pi0, RT-2 상세 operator 위치 (논문 기술 간접 확인)
- MEDIUM: "카메라 뷰 invariance VLA = 0편" — 2025년 8월 이후 추가 논문 가능성 배제 불가

---

*File: model_camera_setup_survey.md*
*Agent: B1 VLA Foundation Model Scientist*
*Purpose: VLA 논문 카메라 설정 전수조사 — Stage 1 데이터 수집 전략 및 연구 방향 근거 자료*
