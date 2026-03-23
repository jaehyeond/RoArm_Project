# SmolVLA 에피소드 데이터 수집 전략 (종합 가이드)

> **작성일**: 2026-02-28
> **현재 상태**: 74 에피소드 보유, 5/5 배포 성공 (RIGHT_FAR 단일 위치)
> **목표**: 150+ 에피소드, 5개 공간 구역 커버, 다중 물체 대응

---

## 1. 현재 상황 진단 — "74개로 성공했는데, 왜 더 필요한가?"

### 1.1 현실 직시

| 사실 | 의미 |
|------|------|
| 5/5 성공 (100%) | 하지만 **같은 위치, 같은 물체**에서만 테스트 |
| Base ~45° (RIGHT_FAR) | 74개 중 19개가 이 구역 → 이 위치만 잘 작동하는 것 |
| LEFT 에피소드 0개 | 왼쪽에 스펀지 놓으면 **아마 실패** |
| Open-loop 4-chunk | Closed-loop은 아직 실패 → 모델이 실시간 보정 못함 |

### 1.2 의심 #1: "74개면 공식 50개보다 많은데?"

**답변: 양의 문제가 아니라 분포의 문제다.**

```
공식 SmolVLA: 5개 위치 × 10회 = 50개 (각 위치에 10개씩 균등)
우리 v3:     74개인데 CENTER 44개(59.5%), RIGHT 11개, LEFT 0개
             → 사실상 CENTER 단일 위치 44개 + 나머지 산발적
```

공식 50개가 작동하는 이유:
- SO-100 로봇 = SmolVLA 사전학습 데이터에 **포함됨** (in-distribution)
- 5개 위치에 **균등 분배**
- 우리는 OOD embodiment + 불균등 분배 → 더 많이 필요

### 1.3 의심 #2: "150개는 어디서 나온 숫자인가?"

| 출처 | 에피소드 수 | 맥락 |
|------|------------|------|
| SmolVLA 공식 | 50 (최소) | SO-100 (in-distribution) |
| SmolVLA 공식 | 25 → "부족, 나쁜 성능" | 유일한 공식 ablation |
| MEMORY.md 기록 | "OOD 로봇은 150+ episodes + 200K steps" | 사전학습 분석에서 추정 |
| Diffusion Policy (Chi 2023) | 90-284 | 사전학습 없이 from-scratch |
| LoRA-VLA (2512.11921) | 200 | Consumer GPU fine-tuning |
| ACT/ALOHA | 50 | Leader-follower (높은 데모 품질) |

**솔직한 평가**: 150이라는 숫자에 확실한 근거는 없다.
- 25 < 필요한 양 < ∞ 범위에서 "50이 SO-100에서 됨" + "우리는 OOD" → 2-3배 추정 = 100-150
- **실제로는 100개에서 먼저 테스트하고 판단해야 한다**

---

## 2. 공간 구역 설계 — "어디에 물체를 놓을 것인가?"

### 2.1 5-Zone 모델 (SmolVLA 공식 패턴)

```
        카메라 (Azure Kinect)
        ┌─────────────────┐
        │     시야 범위     │
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
  LEFT       CENTER        RIGHT
(-30~-40°)   (-5~+5°)    (+30~+40°)
    │            │            │
 LEFT_FAR    (BASE)      RIGHT_FAR
(<-40°)                   (>+40°)
    │            │            │
    └────────────┴────────────┘
           로봇 팔 (RoArm M3)
```

### 2.2 현재 분포 vs 목표 분포

| Zone | Base 각도 | 현재 (74ep) | 목표 (150ep) | 부족분 |
|------|----------|------------|-------------|-------|
| LEFT_FAR | < -40° | 0 | 25 | **25** |
| LEFT | -20° ~ -40° | 0 | 25 | **25** |
| CENTER | -20° ~ +20° | 44 (59.5%) | 30 | 0 (과다) |
| RIGHT | +20° ~ +40° | 11 | 25 | 14 |
| RIGHT_FAR | > +40° | 19 | 25 | 6 |
| **합계** | | **74** | **130-150** | **~70** |

### 2.3 의심 #3: "CENTER가 44개나 있는데 버려야 하나?"

**답변: 버리지 않는다. 하지만 추가 수집의 우선순위가 낮다.**

근거:
- 에피소드 라벨은 학습에 미사용 (MEMORY #14)
- 새 데이터 추가 시 stats.json 변경 → 기존 체크포인트에서 이어 학습 불가 (MEMORY #15)
- 즉, 어차피 smolvla_base부터 재학습이므로 기존 44개 CENTER는 그대로 포함
- CENTER 과다 → action mean이 CENTER로 편향 → LEFT/RIGHT에서 성능 저하
- **해결책: LEFT/RIGHT를 많이 추가하여 분포를 균형화**

---

## 3. 에피소드 품질 기준 — "좋은 데모란 무엇인가?"

### 3.1 7-Phase 프로토콜 (검증 완료)

```
Phase 1: 시작 (Init 위치)           → 0.5초
Phase 2: 접근 + 그리퍼 열기          → 1.0초
Phase 3: Pre-grasp (열린 채 호버링)   → 0.5초
Phase 4: 하강 (열린 채 내려감)        → 1.0초
Phase 5: 잡기 (그리퍼 닫기)           → 0.5초
Phase 6: 들어올리기                   → 1.0초
Phase 7: 복귀 (Init으로)             → 1.0초
                                총: 5-6초
```

### 3.2 품질 체크리스트

| 항목 | 기준 | 근거 |
|------|------|------|
| 그리퍼 최대 개방 | > 40° | collect_data_manual.py 검증 로직 |
| 그리퍼 닫힘 감지 | 개방 → 24-28° (스펀지 접촉) | 배포 성공 시 24-28°가 정상 |
| Z-height at grasp | < 150mm | FK Z 147-156mm에서 성공 |
| Shoulder at grasp | ≥ 50° | DEEP grasp 조건 |
| 에피소드 길이 | 150-300 프레임 (5-10초) | 공식 13초보다 짧지만 pick-only는 5초 적정 |
| 정지 프레임 비율 | < 25% | 현재 33.5% → 개선 필요 |
| 7-Phase 완성도 | 7/7 Phase 포함 | 모든 phase가 있어야 full trajectory 학습 |

### 3.3 의심 #4: "정지 프레임 33%가 정말 문제인가?"

**검토 결과:**

```
30fps에서 5초 에피소드 = 150 프레임
정지 프레임 33% = ~50 프레임이 "아무것도 안 하는" 데이터
→ 모델이 "멈춰있기"를 자주 예측하게 됨
```

반론: SmolVLA는 50-step action chunk로 학습하므로 정지 구간도 trajectory의 일부.
   Init에서 잠깐 멈추는 것 = 안전한 시작 행동. 완전히 나쁘진 않음.

결론:
- 정지 프레임 25% 이하가 이상적
- **하지만 이것 때문에 에피소드를 버리진 않는다**
- 수집 시 "멈추지 말고 바로 움직이기" 습관이 중요

### 3.4 의심 #5: "Hand-guiding의 품질이 Leader-Follower보다 나쁜가?"

| 방법 | 장점 | 단점 |
|------|------|------|
| Hand-guiding (우리) | 장비 1대, 직관적 | 데모 품질 변동, 가림(occlusion) 가능 |
| Leader-Follower | 높은 재현성, 가림 없음 | 장비 2대, 설정 복잡 |

실제 영향:
- MEMORY #23: "Hand-guiding 가림: 2/43만 감지, 물체 중앙은 안 가려짐"
- 가림은 큰 문제 아님 (실험으로 확인됨)
- **데모 품질 변동이 실질적 차이점** → 더 많은 에피소드로 보상

**결론: Hand-guiding으로 충분. 대신 에피소드를 20% 더 수집.**

---

## 4. 수집 세션 계획 — "실제로 어떻게 모을 것인가?"

### 4.1 세션 구조

```
1세션 = 25 에피소드 (약 30분)
  - 5개 zone × 5 reps
  - zone 순서: LEFT_FAR → LEFT → CENTER → RIGHT → RIGHT_FAR
  - 각 zone에서 5회 연속 수집
  - zone 변경 시 물체 위치 이동 + 사진 기록

준비 (5분):
  - 카메라 위치 확인 (고정!)
  - 조명 확인 (일정한 조도)
  - 스펀지 상태 확인
  - robot init 동작 확인

수집 (20분):
  - Zone별 5 에피소드
  - 각 에피소드 사이 reset (I키) + 물체 재배치
  - 에피소드 품질 실시간 확인 (OSD)

검증 (5분):
  - 불량 에피소드 확인 및 재수집
  - 세션 통계 확인
```

### 4.2 목표 수집 일정

| 세션 | 목표 | 누적 | 테스트 |
|------|------|------|--------|
| 기존 | 74ep (CENTER 편향) | 74 | v3 성공 (RIGHT_FAR만) |
| 세션 1 | LEFT_FAR 25ep | 99 | - |
| 세션 2 | LEFT 25ep | 124 | **중간 배포 테스트** |
| 세션 3 | RIGHT + RIGHT_FAR 20ep | 144 | - |
| 세션 4 | 부족 zone 보충 | ~150 | **최종 배포 테스트 (5 zone)** |

### 4.3 의심 #6: "25개씩 4세션이면 되는가, 아니면 한 번에 모아야 하나?"

**분석:**
- 세션 간 시간 경과 → 조명 변화 가능 (자연광 사용 시)
- 카메라 위치 변경 = 모든 데이터 무효 (CLAUDE.md 절대 규칙)
- **인위적 조명(스탠드)을 쓰면 세션 분리 가능**
- **자연광만 쓰면 같은 시간대에 수집 권장**

**결론: 카메라만 고정하면 세션 분리 OK. 조명 일관성 유지.**

---

## 5. 도메인 비교 — "다른 연구자들은 어떻게 하나?"

### 5.1 SmolVLA 공식 vs 우리

| 항목 | SmolVLA 공식 | 우리 v3 | 차이 |
|------|-------------|---------|------|
| 로봇 | SO-100 (in-dist) | RoArm M3 (OOD) | 사전학습에 포함 안 됨 |
| 에피소드 | 50 (5pos×10rep) | 74 (불균등) | 분포 불균형 |
| 카메라 | 2-3대 | 1대 | 시야각 부족 가능 |
| 에피소드 길이 | ~13초 | ~5초 | 짧음 (pick-only라 OK) |
| batch_size | 64 | 64 | 동일 |
| steps | 20K-200K | 50K | 적절 |
| 물체 변형 | 5 위치 | CENTER 편향 | **핵심 차이** |

### 5.2 ACT/ALOHA vs 우리

| 항목 | ALOHA | 우리 |
|------|-------|------|
| 데모 방법 | Leader-Follower (bimanual) | Hand-guiding (single arm) |
| 데모 품질 | 높음 (kinematic 정밀) | 중간 (사람 손 변동) |
| 50 demos 성공률 | 85-95% | 100% (단일 위치) |
| 가림 | 없음 (leader가 별도) | 미미 (2/43) |

**교훈: ALOHA의 "50 demos면 OK"를 Hand-guiding에 그대로 적용하면 안 된다.**
**Hand-guiding은 데모 품질이 낮으므로 60-70 demos = ALOHA의 50에 해당.**

### 5.3 Diffusion Policy vs 우리

| 항목 | Diffusion Policy | 우리 (SmolVLA) |
|------|-----------------|---------------|
| 사전학습 | 없음 (from-scratch) | SmolVLM2 + smolvla_base |
| 필요 에피소드 | 90-284 | 74에서 이미 성공 |
| 이미지 augmentation | random crop, color jitter | 없음 |

**교훈: SmolVLA의 사전학습이 데이터 효율성을 크게 높여줌.**
**그래서 150개면 충분하다고 판단할 수 있는 근거가 됨.**

### 5.4 RT-2/OpenVLA vs 우리 (스케일 비교)

```
RT-2:    130,000+ 데모 → "새 물체 제로샷 가능"
OpenVLA: 970,000+ 에피소드 사전학습 → 25-200 fine-tune
pi0:     68 tasks 사전학습 → 3,000 steps fine-tune
SmolVLA: 11,132 에피소드 사전학습 → 50+ fine-tune (공식)
우리:    74 에피소드 → 성공 (단일 위치)
```

**핵심 인사이트: 사전학습 규모가 클수록 fine-tune 데이터가 적게 필요.**
- RT-2/OpenVLA는 제로샷/few-shot 가능 (거대 사전학습)
- SmolVLA는 중간 규모 → 50+ 에피소드 fine-tune 필요
- 우리가 OOD이므로 → 100-150 에피소드 fine-tune 필요

---

## 6. 핵심 의심과 검증 — "이게 정말 맞는가?"

### 의심 #7: "물체가 바뀌면 처음부터 다시 모아야 하나?"

**현재 답: 그렇다. SmolVLA 450M에서는.**

근거 (이전 브리핑 Q1):
```
텍스트 "Pick up the sponge"
  → SmolLM2 토크나이저 (고정, 학습 안 됨)
  → SmolVLM-2 백본 (고정, 학습 안 됨)
  → Action Expert만 학습됨
```
- VLM이 고정이므로 "sponge" vs "cup" 텍스트 차이를 활용하도록 Action Expert가 학습한 적 없음
- **새 물체 = 새 데이터 수집 + 재학습**

**하지만 희망 있는 시나리오:**
- 여러 물체를 하나의 데이터셋에 합치고 태스크 텍스트를 다르게 하면
- "Pick up the sponge\n" / "Pick up the red cup\n" 등
- Action Expert가 텍스트 차이를 학습할 수 있음
- 물체당 20-50 에피소드 (MEMORY SmolVLA Pretraining Analysis 참조)

### 의심 #8: "Depth 데이터를 왜 안 쓰나?"

현재 상태:
- Azure Kinect가 Depth도 수집함 (NFOV_UNBINNED)
- `depth_YYYY.npy`로 저장됨
- **하지만 SmolVLA는 RGB만 입력으로 사용**

활용 방안:
1. **DepthBio 아이디어** (브리핑 #5): Depth로 접촉 감지 → 그리퍼 타이밍 개선
2. **3DGS 데이터 증강** (RESEARCH_IDEAS Depth-GS-Aug): Depth → 3DGS → novel view
3. **품질 검증**: 수집 시 Depth로 물체 거리 실시간 확인 (이미 구현됨)

**결론: Depth는 현재 학습에는 미사용이지만, 수집 품질 검증 + 미래 연구에 활용.**

### 의심 #9: "이미지 augmentation을 안 해도 되나?"

| Augmentation | 효과 | SmolVLA에서? |
|-------------|------|-------------|
| Random crop/flip | 시야각 변형 | VLM이 고정이므로 pretrained 분포 벗어남 → 위험 |
| Color jitter | 조명 변형 | VLM 고정 → 효과 불명 |
| 물체 위치 변경 | 공간 다양성 | **이것이 SmolVLA의 augmentation** |
| Temporal resampling | 속도 변형 | action mean 안 바뀜 → 무효 |

**결론: SmolVLA에서는 이미지 augmentation 없이 실제 데모 다양성으로 대체.**
**이것은 공식 권장사항과 일치.** (MEMORY SmolVLA SOTA Data Collection 참조)

### 의심 #10: "Closed-loop이 왜 실패하고 Open-loop이 성공하나?"

```
Closed-loop (n=1):
  매 스텝마다 새 이미지 → 새 추론 → 새 액션
  → 그리퍼 닫기 순간에 노이즈 → 24° ↔ 26° 진동 → 잡기 실패

Open-loop (4-chunk):
  50 스텝 동안 trajectory 확정 → 중간에 재추론 안 함
  → 그리퍼가 부드럽게 닫힘 → 성공
```

이것은 **문헌에 잘 안 나오는 우리 프로젝트 고유의 발견**:
- 대부분의 VLA 논문은 closed-loop을 권장
- 우리 경우 bimodal action (그리퍼 open/close)에서 open-loop이 더 좋음
- **논문 아이디어: "When Open-Loop Beats Closed-Loop in VLA"**

---

## 7. 데이터 수집 실행 체크리스트

### 수집 전

- [ ] `conda activate roarm` 환경 활성화
- [ ] 카메라 위치 확인 (이전 세션과 동일? 사진 비교!)
- [ ] 조명 확인 (일관성)
- [ ] 스펀지 상태 확인
- [ ] `python scan_servos.py /dev/ttyUSB0` (모터 상태)
- [ ] `arm.move_init()` 확인
- [ ] collect_data_manual.py 실행, OSD 정상 작동 확인

### 수집 중 (Zone별)

- [ ] 물체를 target zone에 배치 (테이프 마커!)
- [ ] 5회 연속 수집
- [ ] 각 에피소드 후 OSD 품질 확인:
  - Z-height: GREEN (<80mm)?
  - 그리퍼: 열렸다 닫혔나?
  - 길이: 150-300 프레임?
- [ ] 불량 에피소드 즉시 재수집

### 수집 후

- [ ] 에피소드 수 확인
- [ ] zone별 분포 확인
- [ ] `convert_to_lerobot_v3.py` 실행
- [ ] `run_official_train.py` 학습 (batch_size=64, 50K+ steps)
- [ ] 5-zone 배포 테스트

---

## 8. 미래 확장 — "스펀지 이후에는?"

### 8.1 다중 물체 학습 로드맵

```
Phase 1 (현재): 스펀지 1개, 5 zone → 150 에피소드
Phase 2: 스펀지 + 컵, task text 분리 → +100 에피소드 (컵용)
Phase 3: 3개 물체, 다중 태스크 → +150 에피소드
Phase 4: Baby AI(비비) 연동 → 호기심 기반 자동 태스크 생성
```

### 8.2 Cloud VLM + Local VLA 아키텍처

```
[비비 + Gemini Vision API]  ←── 고수준 (1-2Hz)
  "이게 뭐지? 노란 스펀지네. 집어볼까?"
          │
          ↓ 언어 명령 "Pick up the sponge\n"
[SmolVLA + RTX 4090]  ←── 저수준 (10-30Hz)
  관절 각도 예측, 모터 제어
          │
          ↓
    [RoArm M3]
```

- 이것이 NVIDIA GR00T N1의 System 1 + System 2와 같은 패턴
- Gemini Vision이 물체 인식 → SmolVLA가 해당 물체 잡기 실행
- 로컬 대안: PaliGemma 3B INT4 (2.5GB VRAM) → SmolVLA와 동시 실행 가능

### 8.3 CDRL (Concept-Driven Robot Learning) 비전

```
비비의 호기심 → "저 빨간 물체가 궁금해"
      ↓
Gemini Vision → "빨간 컵이야"
      ↓
SmolVLA → "Pick up the red cup\n" 실행
      ↓
성공/실패 → 비비 기억에 저장
      ↓
다음: "다른 물체도 집어볼까?"
```

---

## 9. 핵심 숫자 요약 (Quick Reference)

| 항목 | 값 | 비고 |
|------|-----|------|
| 현재 에피소드 | 74 | CENTER 44, RIGHT_FAR 19 |
| 목표 에피소드 | 150 | 5 zone × 30 |
| 추가 수집 필요 | ~76 | LEFT 50 + RIGHT 보충 |
| 에피소드 길이 | 5-8초 (150-240 프레임) | pick-only 태스크 |
| FPS | 30 | Azure Kinect 고정 |
| 그리퍼 최소 개방 | > 40° | 품질 기준 |
| Z-height at grasp | < 150mm | DEEP 기준 |
| 정지 프레임 | < 25% | 현재 33.5% |
| batch_size | 64 | 공식 권장 |
| 학습 steps | 50K+ | v3 성공 기준 |
| 카메라 | 고정! 절대 이동 금지 | 이동 = 전체 데이터 무효 |
| 배포 모드 | open-loop, 4-chunk | closed-loop은 실패 |

---

## 10. 검증되지 않은 가설 (향후 실험 필요)

1. **LEFT zone에서 RIGHT와 동일한 성공률이 나올까?** → 실험 필요
2. **100 에피소드에서 중간 테스트 → 통과하면 150까지 안 모아도 되나?** → 실험 필요
3. **에피소드 길이를 10초로 늘리면 정지 프레임 비율이 줄까?** → 수집 후 분석
4. **Closed-loop이 더 많은 데이터로 개선되나?** → 150ep 학습 후 재테스트
5. **multi-task (스펀지+컵)에서 각 물체 20개면 충분한가?** → Phase 2에서 검증
6. **PaliGemma 3B를 Gemini 대신 로컬로 돌리면 비비와 작동하나?** → 구현 후 검증

---

## Appendix A: 관련 논문/출처 Reference

| 출처 | 핵심 내용 | 링크/위치 |
|------|----------|----------|
| SmolVLA 공식 docs | 50 eps minimum, 5 positions × 10 reps | `lerobot/docs/source/smolvla.mdx` |
| LeRobot il_robots.mdx | EPISODE_TIME=60s, RESET_TIME=10s | `lerobot/docs/source/il_robots.mdx` |
| ACT (Zhao 2023) | 50 demos, action chunk k=100 | arxiv 2304.13705 |
| Diffusion Policy (Chi 2023) | 90-284 episodes | arxiv 2303.04137 |
| OpenVLA-OFT | 200 episodes, LoRA fine-tune | arxiv 2502.19645 |
| LoRA-VLA | 200 episodes, 8GB VRAM | arxiv 2512.11921 |
| Consistency Matters | 데모 데이터 품질 메트릭 | arxiv 2412.14309 |
| GraspVLA | Billion-frame grasping | arxiv 2505.03233 |
| ICLR 2026 Moritz Reuss | "Data quality: surprisingly few submissions" | 블로그 |

## Appendix B: collect_data_manual.py 핵심 파라미터

```python
# 품질 검증 임계값
GRIPPER_OPEN_THRESHOLD = 40      # 최소 그리퍼 개방 각도
GRIPPER_CLOSE_RANGE = 15         # 개방→닫힘 최소 변화량
SHOULDER_AT_CLOSE = 50           # 닫힘 시 최소 shoulder 각도
Z_AT_CLOSE = 130                 # 닫힘 시 최대 Z-height (mm)
MIN_FRAMES = 90                  # 최소 프레임 (3초)
MAX_FRAMES = 600                 # 최대 프레임 (20초)

# Z-height 분류 (OSD 색상)
DEEP = Z < 80mm                  # GREEN
APPROACH = 80mm ≤ Z < 160mm      # YELLOW
SHALLOW = Z ≥ 160mm              # ORANGE
```
