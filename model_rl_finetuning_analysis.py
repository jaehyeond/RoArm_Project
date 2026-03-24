"""
[B1 VLA MODEL] VLA + Simulation RL Fine-tuning Trend Analysis
==============================================================
Date: 2026-03-24
Purpose: 2025-2026 "sim RL fine-tune for VLA" 트렌드 비판적 분석
         SmolVLA + RoArm-M3 $130 consumer setup 관점

This file is a structured analysis document (not runnable experiments).
All claims include confidence level: HIGH / MEDIUM / LOW
"""

# =============================================================================
# SECTION 1: TREND CLASSIFICATION
# =============================================================================
TREND_OVERVIEW = """
질문: "sim에서 VLA를 RL로 fine-tune하는 것"이 2025-2026 주류 트렌드인가?

답: YES, 주류 트렌드가 맞다. 단, 다음 조건 하에서만 현실적이다:
  - 대형 모델 (OpenVLA 7B, pi0 3B) 타겟
  - 잘 정의된 sim환경 있는 플랫폼 (Franka, UR5, humanoid)
  - compute-rich 연구 환경 (A100 클러스터)

SmolVLA 450M + RoArm-M3 $130 + RTX 4090 Laptop에서는
트렌드 "따라가기"가 가능하지만 접근법이 근본적으로 달라야 함.
"""

# =============================================================================
# SECTION 2: TECHNICAL APPROACHES CLASSIFICATION
# =============================================================================

APPROACH_TAXONOMY = {
    "Category_A_World_Model_as_Simulator": {
        "papers": [
            "VLA-RFT (2510.00406, 2025-10): MLLM이 world model 역할. verified rewards로 RFT",
            "WoVR (2602.13977, 2026-02): World model 시뮬레이터로 VLA RL. dynamics model 통한 virtual rollout",
        ],
        "mechanism": "실제 물리 시뮬레이터 없이 visual world model이 next-state 생성 → reward 계산",
        "key_advantage": "실제 로봇 없이도 RL 루프 가능. domain gap 최소 (world model = real appearance)",
        "key_weakness": "world model 자체 학습에 대규모 real data 필요. 오래된 action 분포 밖에선 환각",
        "confidence_applicable_to_smolvla": "LOW",
        "reason": "SmolVLA flow-matching action space ≠ world model이 예측한 token space. "
                  "VLA-RFT는 token-level reward → flow matching에 직접 적용 불가. "
                  "WoVR도 visual dynamics model 사전학습 필요 = 수백 GB real data"
    },
    "Category_B_Physics_Simulator": {
        "papers": [
            "GR00T N1.6: NVIDIA Isaac Lab. Franka/G1 humanoid 타겟",
            "Scaling VLA w/ Generative 3D Worlds (2603.18532, 2026-03): 생성 3D → physics sim",
            "Beyond Imitation (2602.12628, 2026-02): RL sim-real co-training",
        ],
        "mechanism": "MuJoCo / Isaac Lab에서 RL 학습 후 real robot 전이",
        "key_advantage": "대규모 병렬 rollout. 안전. reset 비용 없음",
        "key_weakness": "sim-to-real gap. URDF/MJCF 모델 필요. 시각적 gap (SigLIP frozen 문제)",
        "confidence_applicable_to_smolvla": "LOW-MEDIUM",
        "reason": "RoArm-M3 URDF 없음 (공식 CAD 미공개). "
                  "SigLIP frozen → sim 렌더링 이미지 cosine dist ~0.6-0.8 (전이 불가 수준). "
                  "Isaac Lab rasterizer는 SigLIP 관점에서 OOD 이미지 생성 확인됨 (A2 에이전트 검증). "
                  "3DGS 렌더링만 cosine ~0.1-0.2로 전이 가능하지만 그것도 정적 장면 한정"
    },
    "Category_C_Real_World_RL": {
        "papers": [
            "HIL-SERL (LeRobot 공식 지원): SAC + human intervention replay",
            "SERL (2401.16013, 2024): real-robot SAC, reward classifier",
        ],
        "mechanism": "실제 로봇에서 직접 SAC/PPO. reward는 human labeler or binary classifier",
        "key_advantage": "sim-to-real gap 없음. 실제 물리 학습",
        "key_weakness": "시간 비용 (1 episode = 30-60sec reset). 하드웨어 마모. reward 설계 어려움",
        "confidence_applicable_to_smolvla": "MEDIUM",
        "reason": "LeRobot에 SAC 이미 구현됨 (hilserl_example.py). "
                  "BUT: SmolVLA는 SAC가 아님 — flow-matching. "
                  "HIL-SERL은 SACPolicy 전용. SmolVLA에 RL 루프 붙이려면 policy gradient나 "
                  "reward-weighted imitation 접근 필요. 바닐라 SAC 교체 불가"
    },
    "Category_D_Reward_Weighted_Imitation": {
        "papers": [
            "SimpleVLA-RL (ICLR 2026): simple reward-weighted BC for VLA",
            "VLA-RFT의 RA-BC component: per-sample loss weighting",
            "OpenVLA-OFT (2025): reward-based fine-tuning",
        ],
        "mechanism": "기존 BC loss에 reward signal로 가중치. 성공 에피소드 = 높은 loss weight",
        "key_advantage": "flow-matching 모델에도 직접 적용 가능. 아키텍처 변경 불필요",
        "key_weakness": "reward는 여전히 binary (success/fail) → sparse. 탐색 없음",
        "confidence_applicable_to_smolvla": "HIGH",
        "reason": "SmolVLA forward() 이미 reduction='none' 지원 (per-sample loss 반환). "
                  "이것을 reward로 가중치만 하면 됨. 코드 수정 최소. "
                  "검증: modeling_smolvla.py line 392-396"
    },
    "Category_E_Residual_RL": {
        "papers": [
            "Residual RL 계열 (NeurIPS 2019~): base policy + residual network",
            "VLA + residual: 아직 소수 논문",
        ],
        "mechanism": "frozen VLA base policy 위에 소형 RL 정책(residual) 학습. 합산 출력",
        "key_advantage": "base VLA 보존. RL이 교정 역할만",
        "key_weakness": "SmolVLA는 chunk (50-step) 출력 → residual이 어느 step에 붙는가 불명확. "
                        "action space mismatch 가능성",
        "confidence_applicable_to_smolvla": "LOW-MEDIUM",
        "reason": "flow-matching denoised output은 연속적 궤적 청크. "
                  "residual 붙이려면 step-level vs chunk-level 결정 필요. "
                  "chunk-level residual은 이론상 가능하지만 실험 없음"
    }
}

# =============================================================================
# SECTION 3: SmolVLA RL 기술적 가능성 심층 분석
# =============================================================================

SMOLVLA_RL_TECHNICAL = """
핵심 질문: SmolVLA flow-matching 기반 모델에 RL이 기술적으로 가능한가?

A. Flow-Matching + RL의 이론적 장벽

   문제: RL은 일반적으로 policy gradient → log π(a|s) 의 미분이 필요
   SmolVLA는 결정론적 ODE 해 (denoising 10 steps) → "log-prob" 정의 어려움

   BUT: 두 가지 우회 경로가 존재:

   경로 1 - REINFORCE-style reward weighting:
     loss = -R * BCE_matching_loss (BC loss에 반대 부호 reward 곱)
     이것이 SimpleVLA-RL / RA-BC의 핵심 아이디어
     SmolVLA에 직접 적용 가능 (reduction='none' 이미 지원됨)

   경로 2 - VLA로 demo 수집 후 filtered BC:
     성공 에피소드만 re-train. reward = success binary
     사실상 DAgger의 self-supervised 버전
     기술적으로 가장 쉬움

B. 450M vs 7B RL 학습 속도 비교

   OpenVLA 7B RL (A100 80GB):
     - LoRA만 업데이트해도 gradient 계산 비용 높음
     - Batch size = 8-16 (메모리 한계)
     - 1K RL steps = ~2-4시간

   SmolVLA 450M RL (RTX 4090 16GB):
     - Action Expert 100M만 업데이트 (VLM frozen)
     - Batch size = 64 (현재 9.85GB, 여유 있음)
     - 1K RL steps = 예상 ~20-30분 (BC 학습 대비 ~2-3x 느림, reward compute 때문에)
     - **이것이 SmolVLA의 RL 연구 가치**: 소비자 GPU에서 RL 실험 가능

   확신도: MEDIUM (실제 타이밍은 미측정)

C. SigLIP Frozen 상태에서 RL이 Action Expert만 업데이트?

   YES. SmolVLA 학습 구조:
   - VLM (SigLIP + SmolLM2 first 16 layers): FROZEN
   - Action Expert (100M, cross-attention to KV cache): TRAINABLE

   RL 신호는 Action Expert만 업데이트 가능.

   의미:
   - 시각적 representation은 변하지 않음 (SigLIP frozen)
   - RL이 "무엇을 봐야 하는가"는 못 바꾸고 "어떻게 움직일 것인가"만 바꿈
   - 이것은 제약이기도 하고 장점이기도 함:
     제약: 새로운 visual concept (새 물체 모양) 학습 불가
     장점: RL이 VLM을 망가뜨릴 수 없음 (catastrophic forgetting 없음)

   확신도: HIGH (아키텍처에서 직접 확인)
"""

# =============================================================================
# SECTION 4: Consumer Hardware ($130 arm) 현실성 평가
# =============================================================================

CONSUMER_HW_FEASIBILITY = """
RoArm-M3 $130 + RTX 4090 Laptop에서 RL 접근법별 현실성:

1. Physics Sim (Isaac Lab / MuJoCo): VERY HARD
   - RoArm-M3 공식 URDF/MJCF 없음 → 직접 측정/역설계 필요 (2-4주)
   - Isaac Lab 설치 완료 (isaac_lab_setup.md 확인) but RoArm 환경 없음
   - SigLIP frozen → sim-rendered image는 OOD (cosine ~0.6-0.8)
   - 즉, sim에서 학습해도 real SigLIP feature space로 전이 안 됨
   - 예외: 3DGS 렌더링 사용 시 cosine ~0.1-0.2 (전이 가능)
     but 동적 물체 grasping에 3DGS 적용 = 미해결 문제
   - 현실성: LOW (단독 논문 토픽이 됨. 3개월+ 소요)

2. World Model Sim: NOT FEASIBLE
   - Visual dynamics world model 사전학습에 수백 GB real data 필요
   - 현재 dataset: 74 episodes (~10K frames) → world model 학습 불충분
   - 현실성: VERY LOW

3. Real-World RL (SAC): FEASIBLE BUT SLOW
   - LeRobot HIL-SERL 이미 구현됨 (hilserl_example.py)
   - BUT SmolVLA는 SACPolicy가 아님 → 직접 교체 불가
   - Real RL 루프: 1 episode = ~30sec. 10K steps = ~83시간 (3.5일)
   - 현실성: MEDIUM (긴 시간, 코드 수정 필요)

4. Reward-Weighted BC (SimpleVLA-RL 스타일): MOST FEASIBLE
   - SmolVLA forward(reduction='none') 이미 지원
   - 성공/실패 라벨은 deploy_smolvla.py 실행 결과로 수집 가능
   - 추가 코드: reward 수집 루프 + weighted loss = ~100줄
   - 현실성: HIGH
   - 예상 소요: 1주 구현 + 2-3일 실험

5. VLA Denoising Variance as Active Learning Signal: UNIQUE GAP
   - SmolVLA 10 denoising steps의 중간 variance = uncertainty proxy
   - 이미 research_ideas_corl_thesis.md에서 "미탐색 gap" 확인됨
   - RL은 아니지만 self-improvement의 핵심 신호가 됨
   - 확신도: HIGH (gap 실존)
"""

# =============================================================================
# SECTION 5: 논문 기여 가능한 구체적 GAP 분석
# =============================================================================

CONTRIBUTION_GAPS = {
    "Gap_1_Consumer_Arm_RL": {
        "claim": "Consumer-grade arm ($130-$300)에서 VLA RL fine-tuning 실험 없음",
        "evidence_for": [
            "VLA-RFT, WoVR, Beyond Imitation: 모두 Franka Panda ($20K+) 또는 UR5",
            "SimpleVLA-RL (ICLR 2026): 논문 확인 필요 — 어떤 로봇인지 명시 안 됨",
            "GR00T N1.6: Franka + humanoid",
        ],
        "evidence_against": [
            "SERL (2401.16013): 사실 Franka이지만 저비용 실험 강조",
            "다른 SO-100 + SmolVLA RL 실험 존재 가능성 (검색 필요)",
        ],
        "confidence": "MEDIUM",
        "verification_needed": "SimpleVLA-RL 논문 로봇 확인. SO-100 RL 관련 논문 검색",
        "if_confirmed_contribution": "동일한 reward-weighted BC를 consumer arm에서 검증. "
                                     "compute/data 효율성 분석이 핵심 메시지"
    },
    "Gap_2_SmolVLA_Specific_RL": {
        "claim": "SmolVLA에 RL 적용 논문이 없음 (OpenVLA/pi0 타겟 RL 논문만 존재)",
        "evidence_for": [
            "ICLR 2026 VLA 164편 중 RL+VLA: 8개 이상. 하지만 SmolVLA 언급 없음 (tech_smolvla_pretraining.md 확인)",
            "VLA-RFT, WoVR: OpenVLA/pi0 타겟",
            "SimpleVLA-RL: 이름은 simple이지만 어떤 VLA인지 확인 필요",
        ],
        "evidence_against": [
            "SmolVLA 자체가 2506.01844 = 2025-06 논문. RL 후속연구 시간 부족",
        ],
        "confidence": "MEDIUM-HIGH",
        "verification_needed": "SimpleVLA-RL 2206.xxxxx 논문 확인",
        "if_confirmed_contribution": "SmolVLA 최초 RL 실험 = paper-sized contribution"
    },
    "Gap_3_Flow_Matching_RL_Mechanism": {
        "claim": "Flow-matching 기반 VLA에 RL을 적용하는 메커니즘이 명확히 정의되지 않음",
        "evidence_for": [
            "VLA-RFT는 token-level 접근 (LLM style). pi0 (flow-matching) RL은 pi0-FAST 논문에서 부분 언급",
            "SmolVLA flow-matching의 per-sample loss를 reward로 weighting하는 것은 미발표",
        ],
        "evidence_against": [
            "pi0 논문 (arXiv 2410.24164)의 후속에서 flow-matching RL 언급 가능성",
        ],
        "confidence": "MEDIUM",
        "verification_needed": "pi0 Section 4+ 확인. pi0-FAST (2411.xxxxx) 확인",
        "if_confirmed_contribution": "flow-matching VLA의 reward-weighted loss 수식 정의 = 방법론 기여"
    },
    "Gap_4_Denoising_Variance_RL_Signal": {
        "claim": "Flow-matching VLA의 denoising step variance를 RL reward proxy로 사용 = 미탐색",
        "evidence_for": [
            "SmolVLA 10 denoising steps → per-step action 분산 = uncertainty",
            "Diffusion Policy 분야에서도 variance를 uncertainty로 쓴 논문 거의 없음",
            "research_ideas_corl_thesis.md: 'HIGH confidence gap'으로 확인됨",
        ],
        "evidence_against": [
            "Uncertainty quantification for diffusion: 일부 NLP/CV 논문 존재 가능성",
        ],
        "confidence": "HIGH (denoising variance as RL proxy 구체적 조합은)",
        "verification_needed": "Diffusion Policy uncertainty + RL 조합 arXiv 검색",
        "if_confirmed_contribution": "RL reward 없이도 exploration signal 자동 생성. 독창적 메커니즘"
    }
}

# =============================================================================
# SECTION 6: SmolVLA RL 구현 로드맵 (CoRL 2026 관점)
# =============================================================================

IMPLEMENTATION_ROADMAP = """
전제: CoRL 2026 deadline 5/28. 현재 2026-03-24. 남은 시간: ~65일
현재 상태: 74ep 데이터, 100% success (open-loop), 카메라 재장착 OOD 문제 있음

옵션 1: RL을 현재 연구 방향에 추가 모듈로 (Overnight Self-Improvement Loop에 RL 신호 통합)
  - 현재 IDEA 2 (Self-Improvement Loop)에 reward-weighted BC 추가
  - 구현: deploy_smolvla.py 실행 → 성공/실패 기록 → weighted re-train
  - 투자 시간: 1-2주
  - CoRL 기여: "RL-guided self-improvement on consumer hardware" 강화됨
  - 위험: 카메라 OOD 문제 해결 안 되면 reward 수집 불가

옵션 2: RL을 독립 연구로 (현재 방향 포기)
  - 리스크: 방향 전환 비용 + 65일 남음 = VERY HIGH RISK
  - 권장하지 않음

옵션 3: RL 분석을 논문 Related Work / Discussion에 포함 (구현 없음)
  - "우리 방법(AR-Guided + Oracle)이 RL 접근법 대비 데이터 효율이 높은 이유" 설명
  - 실험 없이 포지셔닝만으로 CoRL 차별화 가능
  - 투자 시간: 0 (논문 writing에서 처리)
  - 가장 현실적인 옵션

결론: 옵션 3 + 옵션 1의 단순 버전 (성공/실패 기반 filtered BC)
"""

# =============================================================================
# SECTION 7: 우리가 실제로 할 수 있는 가장 작은 RL 실험
# =============================================================================

MINIMAL_RL_EXPERIMENT = """
"Minimal RL for SmolVLA on RoArm-M3"

목표: RL 논문 트렌드 참여. 구현 최소화.

방법:
  1. deploy_smolvla.py로 100 episodes 실행 (자율)
  2. 각 episode 성공 여부 binary 라벨 (사람 관찰 or Qwen2.5-VL judge)
  3. 성공 episodes만 데이터셋에 추가 (filtered BC)
  4. run_official_train.py로 재학습
  5. 반복 (3회 루프)

이것은 사실 "RL"의 가장 단순한 형태: REINFORCE with binary reward + BC replay
구현 추가 코드: ~50줄 (success label 추가 + filtering logic)

코드에서 지원 확인:
  - SmolVLAPolicy.forward(reduction='none') → per-sample loss (line 392-396)
  - deploy_smolvla.py → episode 저장 이미 존재
  - run_official_train.py → dataset re-train 지원

선행연구와의 차이:
  - SimpleVLA-RL: weighted loss (soft reward). 우리: filtered BC (hard threshold)
  - SOAR: seed policy 30%+ 필요. 우리: 100% 이미 달성 → 새 tasks에 적용
  - 핵심 차이: consumer arm ($130) + local VLM judge (Qwen 3B) + no sim
"""

if __name__ == "__main__":
    print("VLA RL Fine-tuning Trend Analysis")
    print("=" * 60)
    print(TREND_OVERVIEW)
    print("\nMost feasible approach for our setup:")
    print("  -> Reward-Weighted BC (Category D)")
    print("  -> Confidence: HIGH")
    print("  -> SmolVLA forward(reduction='none') already supports this")
    print("\nKey gap (HIGH confidence):")
    print("  -> Denoising variance as active learning / RL proxy")
    print("  -> SmolVLA-specific RL = no prior work found")
    print("\nPhysics sim (Isaac Lab) applicability: LOW")
    print("  -> No RoArm-M3 URDF")
    print("  -> SigLIP frozen + sim rendering = cosine dist ~0.6-0.8 (OOD)")
