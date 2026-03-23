# 새 컨텍스트용 프롬프트 — SmolVLA 데이터 수집 전략 학습

아래 프롬프트를 새 Claude 대화에 붙여넣기하면 이전 연구 맥락을 유지한 채 공부/질문할 수 있다.

---

## 프롬프트 (복사해서 사용)

```
나는 RoArm-M3-Pro (6-DOF) 로봇팔 + SmolVLA(450M VLA 모델) 프로젝트를 진행 중이야.
현재 상태와 맥락을 먼저 읽고, 내 질문에 답해줘.

## 현재 상태
- 74개 에피소드로 SmolVLA fine-tuning 완료 (50K steps, batch_size=64)
- 배포 성공: open-loop 4-chunk, 5/5 성공 (단, RIGHT_FAR 단일 위치에서만 테스트)
- 문제: CENTER 44개(59.5%), LEFT 0개 → 공간 편향 심각
- 목표: 150개 에피소드 (5 zone × 30개), LEFT 구역 보충 최우선

## 기술 스택
- GPU: RTX 4090 Laptop (15.6GB VRAM), CUDA 12.6
- Camera: Azure Kinect DK (1280x720 RGB + Depth, 30fps)
- Framework: LeRobot 0.4.4 + SmolVLA (HuggingFace lerobot/smolvla_base)
- 데이터 수집: 토크 OFF hand-guiding (leader-follower 아님)
- Python 3.11, PyTorch 2.7.1+cu126

## 핵심 제약조건 (반드시 지켜야 함)
1. 커스텀 학습 스크립트 금지 → `lerobot-train` CLI만 사용
2. 사전학습 `lerobot/smolvla_base` 필수 (from-scratch하면 평균 액션만 출력)
3. 카메라 위치 변경 = 모든 데이터 무효 = 재수집 필수
4. SmolVLA VLM은 고정(freeze), Action Expert만 학습됨
5. 새 데이터 추가 시 stats.json 변경 → 기존 체크포인트에서 이어 학습 불가

## 배운 교훈 (이전 실험에서 검증됨)
- 50 episodes (68% SHALLOW) → 0% 배포 성공 (v1 실패)
- 74 episodes (ALL DEEP, 좋은 품질) → 100% 배포 성공 (v3)
- 에피소드 품질 > 에피소드 수량
- Closed-loop n=1 실패, Open-loop 4-chunk 성공 (bimodal gripper 노이즈 때문)
- 이미지 augmentation 불필요 → 실제 공간 다양성이 augmentation
- SmolVLA 공식: 5 positions × 10 reps = 50 episodes (SO-100 in-distribution)
- 우리는 OOD embodiment → 100-150 에피소드 필요 (추정)
- 7-Phase 에피소드: 시작→접근+열기→호버→하강→잡기→들기→복귀 (5-6초)

## 관련 도메인 비교 (이미 조사됨)
- SmolVLA 공식: 50ep (SO-100, in-dist)
- ACT/ALOHA: 50 demos (leader-follower, 높은 품질)
- Diffusion Policy: 90-284 demos (사전학습 없음)
- LoRA-VLA: 200 demos
- 결론: 사전학습 규모 ∝ 1/필요 데이터량

## 미해결 의심 (검증 필요)
1. LEFT zone에서도 RIGHT와 동일한 성공률이 나올까?
2. 100개에서 중간 테스트 → 통과하면 150까지 불필요?
3. Closed-loop이 더 많은 데이터로 개선되나?
4. 물체 변경 시 데이터 최소량은? (물체당 20? 50?)
5. PaliGemma 3B 로컬이 Gemini API를 대체 가능한가?

## 장기 비전
- Baby AI "비비" + SmolVLA = 호기심 기반 로봇 학습 (CDRL)
- Cloud VLM (Gemini) → 씬 이해 / Local VLA (SmolVLA) → 모터 제어
- 목표: 정적 정책이 아닌 "성장하는 인지 아키텍처"

이 맥락을 기반으로 내 질문에 답해줘.
항상 step-by-step으로, 의심과 검증을 포함해서.
도메인 비교와 논문 근거가 있으면 같이 알려줘.
```

---

## 사용 예시 질문들

이 프롬프트를 붙여넣은 후 아래 질문들을 활용할 수 있다:

### 데이터 수집 기초
- "에피소드 수집 시 정지 프레임을 줄이려면 어떻게 해야 해?"
- "zone별 물체 배치 기준점(테이프 마커)을 어떻게 설계하는 게 좋아?"
- "5개 zone 대신 7개로 나누면 장점이 있어?"
- "에피소드 길이를 10초로 늘리면 뭐가 달라져?"

### 품질 검증
- "100개 모은 후 중간 배포 테스트에서 뭘 확인해야 해?"
- "zone별 성공률이 불균등하면 어떤 zone을 보충해야 해?"
- "정지 프레임 비율을 25% 아래로 낮추는 실용적 방법은?"

### 도메인 비교
- "ACT처럼 action chunk k=100을 쓰면 SmolVLA보다 나을까?"
- "Diffusion Policy가 90개 필요한데 SmolVLA가 50개면 되는 이유를 더 자세히 설명해줘"
- "OpenVLA-OFT의 LoRA fine-tuning을 SmolVLA에 적용할 수 있어?"

### 다중 물체 확장
- "스펀지 + 컵을 하나의 데이터셋으로 합치면 어떤 문제가 생겨?"
- "물체별 task text를 다르게 하면 Action Expert가 정말 구분할 수 있어?"
- "multi-task 학습 시 batch_size나 steps을 어떻게 조절해야 해?"

### Baby AI 연동
- "비비가 SmolVLA에 명령을 내리는 인터페이스를 어떻게 설계해?"
- "비비의 호기심 → 물체 선택 → SmolVLA 실행 → 피드백 루프를 구체적으로 설계해줘"
- "PaliGemma 3B를 로컬에서 씬 이해용으로 쓰면 SmolVLA VRAM과 충돌 안 나?"

### 논문/연구 방향
- "우리가 발견한 'open-loop이 closed-loop보다 나은 경우'를 논문으로 쓸 수 있어?"
- "CDRL (Concept-Driven Robot Learning) 아이디어의 novelty를 검증해줘"
- "한국어 VLA 데이터셋(KoRobo)의 학술적 가치가 실제로 있어?"
