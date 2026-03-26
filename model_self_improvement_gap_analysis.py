"""
[B1 VLA MODEL] Robot Self-Improvement Research Gap Analysis
===========================================================
Date: 2026-03-25
Purpose: CoRL 2026 방향 탐색 — Robot Self-Improvement / Autonomous Practice
         SmolVLA + RoArm-M3 + consumer-grade setup 기준 실현 가능한 갭 분석

CRITICAL RULE: 모든 갭 주장은 EVIDENCE_FOR / EVIDENCE_AGAINST / CONFIDENCE 명시.
이전 실수(2026-03-10, 2026-03-23) 반복 방지.
"""

# =============================================================================
# SECTION 1: PAPERS ANALYZED (사전학습 논문 목록)
# =============================================================================

PAPERS_TO_ANALYZE = {
    "SOAR": {
        "ref": "Berkeley, CoRL 2024",
        "title": "SOAR: Autonomous Improvement of Instruction Following Skills via Foundation Models",
        "claim_to_verify": "Self-improving with WidowX + CLIP",
        "robot": "WidowX (low-cost but ~$2K+, not $130)",
        "key_mechanism": "Internet video → skill library expansion. CLIP for task proposal + success detection.",
        "relevance_to_us": "Closest analog to our setup. But: WidowX, not $130 arm. CLIP, not frozen SigLIP.",
        "key_difference": "SOAR expands SKILL LIBRARY (new tasks). We target IMPROVEMENT on fixed tasks.",
    },
    "Self_Improving_Robots_2303_01488": {
        "ref": "arXiv 2303.01488",
        "title": "Autonomous Improvement of Instruction Following Skills via Foundation Models",
        "claim_to_verify": "autonomous visuomotor RL",
        "verified_content": "UNKNOWN — need to verify what robot/setup this uses",
        "note": "Title overlap with SOAR possible. Verify if this is the same paper.",
    },
    "VLAC_2509_15937": {
        "ref": "arXiv 2509.15937",
        "title": "Vision-Language-Action-Critic",
        "claim_to_verify": "learns from success AND failure",
        "key_mechanism": "VLM-based reward (critic) to label episodes. BC on success + contrastive on failure.",
        "relevance_to_us": "VLM critic = our Qwen2.5-VL judge idea. This validates the direction.",
        "key_difference": "VLAC uses online VLM critic. We want LOCAL inference (Qwen 3B on RTX 4090).",
        "gap_question": "VLAC on consumer hardware ($130 arm)? Likely uses Franka/xArm. Verify.",
    },
    "OnTheFly_VLA_2601_06748": {
        "ref": "arXiv 2601.06748",
        "title": "On-the-Fly VLA Adaptation",
        "claim_to_verify": "test-time RL for VLA",
        "key_mechanism": "At inference time, adapts VLA policy via RL gradient without offline retraining.",
        "relevance_to_us": "HIGH — if we can do test-time RL, no retraining loop needed.",
        "gap_question": "Does this work with flow-matching VLAs (SmolVLA)? Or LM-action VLAs only?",
        "technical_concern": "Flow-matching ODE → test-time gradient through 10 denoise steps = expensive.",
    },
    "SimpleVLA_RL_2509_09674": {
        "ref": "arXiv 2509.09674",
        "title": "SimpleVLA-RL",
        "claim_to_verify": "RL for VLA training",
        "key_mechanism": "Reward-weighted BC. Binary reward. VLA pretrained → RL fine-tuned.",
        "relevance_to_us": "VERY HIGH — this is literally our proposed approach.",
        "gap_question": "What robot? If not consumer-grade, gap exists. Verify robot hardware.",
        "smolvla_connection": "SmolVLA forward(reduction='none') already supports this mechanism (line 392-396).",
    },
}

# =============================================================================
# SECTION 2: Q1 — VLA Self-Improvement 현황 분석
# =============================================================================

Q1_CURRENT_STATE = """
Q1: VLA self-improvement 분야에서 정확히 무엇이 되어 있는가?

VERIFIED LANDSCAPE (기존 메모리 + source 분석 기반):

[A] 있는 것 (선행연구 존재, 갭 주장 불가):
  - SOAR (CoRL'24): Internet video → skill library. CLIP success detection. WidowX.
  - SimpleVLA-RL (arXiv 2509.09674): reward-weighted BC for VLA. Binary reward.
  - VLAC (arXiv 2509.15937): VLM critic (online) for success/failure labeling.
  - On-the-Fly VLA (arXiv 2601.06748): test-time RL adaptation.
  - OpenVLA-OFT (2025): reward-based fine-tuning for OpenVLA.
  - DAgger variants: interactive imitation learning with human corrections.
  - AutoRT (Google, 2024): 20+ robots self-directed via LLM instructions.

[B] 없거나 약한 것 (갭 후보):
  1. Consumer-grade ($130) arm에서 VLA self-improvement = SOAR은 WidowX ~$2K
  2. Flow-matching VLA (SmolVLA/pi0)에 reward-weighted BC:
     - SimpleVLA-RL의 대상 VLA가 무엇인지 미확인
     - pi0 (flow-matching) RL = "pi0-FAST" 이후 논문 가능성 존재
  3. LOCAL VLM judge (3B 이하, consumer GPU) for success detection:
     - VLAC와 SOAR 모두 클라우드 또는 큰 VLM 사용 가정
     - Qwen2.5-VL 3B on RTX 4090 = 로컬 추론 가능하지만 선행연구 미확인
  4. Denoising variance (flow-matching intermediate steps) as uncertainty proxy:
     - 일반적인 diffusion UQ 논문: 있음 (CV/NLP)
     - 로봇 manipulation 특화 + flow-matching VLA 특화 = 미확인
  5. Fleet self-improvement on consumer hardware (2+ cheap robots parallel):
     - AutoRT (Google) = 20대 로봇 but 비용 불명확
     - 2x RoArm-M3 ($260 total) 병렬 = 미발표 (우리만 보유 가능)
"""

# =============================================================================
# SECTION 3: Q2 — Consumer-Grade Gap 분석
# =============================================================================

Q2_CONSUMER_GAP = """
Q2: $130 로봇에서의 갭이 실제로 존재하는가?

[근거 FOR 갭 존재]:
  - SOAR: WidowX 사용 → robot.com 기준 ~$1,800-$2,500. 우리는 $130 = 10x 저렴
  - SimpleVLA-RL (2509.09674): "VLA" 언급하지만 어떤 robot 미확인 (MEDIUM 신뢰도)
  - ICLR 2026 VLA 164편: RL+VLA 8개+ but 모두 Franka($20K)/UR5($30K)/humanoid 타겟
    (확인: model_rl_finetuning_analysis.py SECTION 4)
  - HIL-SERL (LeRobot): SAC 기반 → SmolVLA에 직접 적용 불가
  - 우리 실증: v1(50ep bad) → 0%, v3(74ep good) → 100% = 데이터 품질이 결정적
    이 실증 자체가 consumer hw에서 VLA 데이터 품질 문제를 검증한 것

[근거 AGAINST 갭 존재]:
  - SO-100 ($269) + SmolVLA 논문 (arXiv 2506.01844): consumer 로봇에서 VLA 학습 이미 발표
    하지만: 이것은 RL/self-improvement가 아닌 supervised BC
  - LeRobot 에코시스템 자체가 consumer-grade 타겟 → SO-100 관련 논문 증가 예상
  - ALOHA 2 ($18K) vs ALOHA Unleashed ($32K) vs SO-100 ($269):
    학습 커뮤니티에서 저비용 로봇 RL 논문은 증가 추세

[결론]:
  - "Consumer-grade arm ($130-$300)에서 VLA RL/self-improvement" = MEDIUM confidence 갭
  - SO-100 + SmolVLA supervised BC는 있지만 RL/self-improvement는 없음 (현재까지)
  - 이 갭은 강력한 포지셔닝이 될 수 있음: "누구나 할 수 있는 VLA 자율개선"
  - CRITICAL: SimpleVLA-RL (2509.09674) 로봇 확인 필수 — 만약 SO-100이면 갭 없음
"""

# =============================================================================
# SECTION 4: Q3 — Flow-Matching Denoising Variance as Uncertainty Signal
# =============================================================================

Q3_DENOISING_VARIANCE = """
Q3: SmolVLA flow-matching denoising variance가 uncertainty signal로 작동하는가?

[기술적 분석 (SOURCE CODE 검증됨)]:

SmolVLA inference loop (modeling_smolvla.py line 826-863):
  x_t = noise (random Gaussian)
  for step in range(10):  # num_steps=10
      v_t = denoise_step(x_t, time=1.0 + step * (-0.1))
      x_t = x_t + (-0.1) * v_t   # Euler integration

  → 10단계 ODE 적분. 각 step에서 x_t는 action trajectory.

[Variance 신호로 쓸 수 있는 이유]:
  1. x_t는 매 inference마다 다른 initial noise에서 시작 → 여러 번 sample하면 분산 계산 가능
  2. 모델이 확신하면 (seen scenario): x_t trajectories가 빠르게 수렴 → 낮은 분산
  3. 모델이 불확신하면 (OOD scenario): x_t가 다양한 mode로 수렴 → 높은 분산
  4. n=5회 sampling, variance = mean([std(x_final)] over 6-dim) → 간단한 계산

[수식]:
  uncertainty_t = (1/D) * sum_d [ std(x_final^(1)_d, ..., x_final^(N)_d) ]
  where D=6 (6-DOF), N=5 (samples), x_final = final denoised action

[선행연구 분석]:
  A) Diffusion Policy uncertainty (일반):
     - Ho et al. 2020 이후 diffusion UQ 연구 있음
     - 하지만: 이미지 생성 모델 중심 (DDPM, DDIM variance 분석)
     - 로봇 manipulation 적용 = 소수
  B) DDPO (Denoising Diffusion Policy Optimization, arXiv 2305.13301):
     - RL for diffusion text-to-image → 로봇에 직접 적용 아님
     - 하지만 flow-matching VLA에 이런 접근 없음
  C) Uncertainty-driven active learning for diffusion policy:
     - "active learning + diffusion policy" 조합 논문: 검색 필요
     - 아직 발견 못함 = MEDIUM confidence 갭
  D) VLAC (2509.15937): VLM critic으로 uncertainty 측정 → denoising variance 아님

[SmolVLA-specific 분석]:
  - 10 denoising steps + KV cache reuse = inference efficient
  - N=5 sampling 추가 비용: 5x inference time (108ms → ~540ms)
  - 하지만 "uncertainty 높음 → 더 데이터 필요" 신호로 쓰면:
    deployment 중 low-confidence 상황을 자동 감지 가능
  - 기존 deploy_smolvla.py에 5-sample wrapper 추가 = ~20줄 코드

[결론]:
  - "Flow-matching denoising ensemble variance as manipulation uncertainty proxy" = HIGH confidence 갭
  - 선행연구: diffusion UQ 있음 (image gen). 로봇 manipulation + flow-matching VLA 특화 = 없음
  - 검증 필요: "diffusion policy uncertainty active learning" 검색 (3개+ 키워드)
  - 우리 차별점: SmolVLA의 frozen SigLIP + 10-step ODE + consumer hardware 조합
"""

# =============================================================================
# SECTION 5: Q4 — Local VLM Judge (Qwen2.5-VL 3B) for Success Detection
# =============================================================================

Q4_LOCAL_VLM_JUDGE = """
Q4: Local VLM judge (Qwen2.5-VL 3B) for success detection — precedent 있는가?

[선행연구 현황]:

A) 있는 것:
  - VLAC (2509.15937): VLM을 critic으로 사용 — 하지만 GPT-4V 또는 large VLM (>7B) 가정
  - SOAR (CoRL'24): Foundation model이 task proposal + success detection — 클라우드 API
  - AutoRT (Google, 2024): PaLM-E (대형 모델) 기반 instruction generation
  - VoxPoser (NeurIPS'24): GPT-4 기반 task planning + reward shaping
  - SayCan (Google, 2022): GPT-3 + affordance → 대형 클라우드 모델

B) 없거나 약한 것:
  - LOCAL (edge-deployable) VLM judge for robot success detection:
    - 3B급 모델을 consumer GPU에서 실행하는 판단 에이전트 = 미발표
    - Qwen2.5-VL 3B RTX 4090에서 4-bit quantized: ~3-4GB VRAM, 추론 ~200-400ms
    - 이것이 robot success 판단에 쓰인 논문 = 미확인
  - VLM judge + flow-matching VLA self-improvement loop:
    - VLAC: VLM critic for VLA, but 어떤 VLA인지 + 로컬 실행 여부 미확인

[기술적 실현 가능성 (RTX 4090 기준)]:
  Scenario A: 순차 실행 (SmolVLA inference → 잠시 후 → Qwen judge)
    - SmolVLA 450M: 9.85GB VRAM (batch=64 학습 시). 추론 시 ~3-5GB
    - Qwen2.5-VL 3B (INT4): ~3-4GB
    - 합산: ~7-9GB → RTX 4090 16.7GB에서 동시 로드 가능
    - 하지만: 학습 중에는 SmolVLA가 9.85GB → Qwen 동시 불가
    - 해결: inference phase에서만 Qwen 로드, 학습 시 언로드

  Scenario B: 별도 판단 루프
    - deploy_smolvla.py: 에피소드 실행 → 결과 이미지 저장
    - Qwen judge: 배치로 이미지 분석 → success/fail 라벨
    - 완전히 비동기 가능 → VRAM 충돌 없음

[결론]:
  - "Local VLM judge for robot success detection" = MEDIUM confidence 갭
  - VLAC/SOAR가 이미 VLM-as-judge를 했지만 → 클라우드/대형 모델 가정
  - Local (3B, edge-deployable) judge = 새로운 기여점
  - 차별화 메시지: "API 비용 없이, 인터넷 없이, 소비자 GPU에서 자율 성공 판단"
  - 검증 필요: "small VLM robot success detection" + "edge VLM reward robot" 검색
"""

# =============================================================================
# SECTION 6: Q5 — Reward-Weighted BC with Flow-Matching VLA
# =============================================================================

Q5_REWARD_WEIGHTED_BC = """
Q5: Flow-matching VLA + reward-weighted BC — any precedent?

[SmolVLA 코드 검증 (HIGH confidence, 직접 소스 확인)]:
  modeling_smolvla.py line 356-401:
    def forward(self, batch, noise=None, time=None, reduction='mean'):
        ...
        if reduction == 'none':
            per_sample_loss = losses.mean(dim=(1, 2))  # shape (B,)
            return per_sample_loss, loss_dict

  → SmolVLA는 이미 per-sample loss를 반환할 수 있음
  → reward = binary success → loss = -reward * per_sample_loss (reward-weighted)
  → 코드 구현: ~50줄 추가

[선행연구 현황]:

A) Flow-matching 구체적으로:
  - pi0 (arXiv 2410.24164): flow-matching VLA. RL 언급? → pi0 paper Section 4에서 확인 필요
  - pi0-FAST (arXiv 2411.xxxxx): 고속 추론 버전. RL variant?
  - SmolVLA (arXiv 2506.01844): forward(reduction='none') 있음. 이것이 RA-BC용으로 설계됨
    CRITICAL: SmolVLA 논문이 이미 RA-BC를 언급할 수 있음 → 논문 Section 확인 필요

B) Reward-weighted BC 일반:
  - SimpleVLA-RL (ICLR 2026, arXiv 2509.09674): 명확한 reward-weighted BC for VLA
    → 어떤 VLA 대상인지 미확인. LM-action VLA일 가능성 있음 (token-level)
  - RA-BC (Reward-Advantaged BC): theoretical component of VLA-RFT (2510.00406)
  - RLHF for LLM: 원류. 로봇에 적용은 별도 문제.

C) Flow-matching VLA 특화 reward-weighted BC:
  - 이것이 KEY QUESTION: SimpleVLA-RL이 flow-matching VLA에 적용된 사례?
  - SmolVLA와 pi0 모두 flow-matching → token-based VLA (OpenVLA, Octo token)와 다름
  - flow-matching의 loss = continuous ODE matching loss (MSE between u_t and v_t)
  - 이것에 scalar reward를 곱하는 것 = 수식적으로 자명하지만
  - "flow-matching per-sample loss × binary reward" = 실험적 검증 논문 미확인
  - 이것이 진짜 갭일 수 있음 (MEDIUM confidence)

[결론]:
  - "Reward-weighted BC for flow-matching VLA" = MEDIUM confidence 갭
  - SimpleVLA-RL이 flow-matching을 대상으로 하는지 확인 필요
  - SmolVLA forward(reduction='none')은 이미 이를 위해 설계되었으나 사용 사례 없음
  - 우리가 처음으로 SmolVLA + reward-weighted BC를 consumer hardware에서 실험하면:
    contribution = "flow-matching VLA RL self-improvement, consumer hardware"
"""

# =============================================================================
# SECTION 7: Q6 — Fleet Learning at Consumer Scale
# =============================================================================

Q6_FLEET_LEARNING = """
Q6: Fleet learning (2+ cheap robots in parallel) — precedent at consumer scale?

[선행연구]:
A) Large-scale fleet:
  - AutoRT (Google, 2024): ~20대 로봇, 고가 + 클라우드 LLM. Consumer scale 아님.
  - DROID (2024): 86 lab setups 분산. 학교/기업 랩 레벨.
  - Open X-Embodiment (2023): 22개 embodiment, 1M episodes.
  - RT-X (2023): Google 내부 다중 로봇.

B) Consumer-scale (2-3 identical cheap robots):
  - SO-100 × 2 병렬 데이터 수집: 있는지 미확인 (최근 ICLR'26 찾아봐야)
  - "Fleet learning" + consumer arm: 검색 결과 없음 (3/23 기준)
  - 가장 비슷한: PIFold(2023), Diffusion IL Fleet 계열 — 이것도 고가 HW

C) 우리 보유 장비:
  - RoArm-M3 × 2 (실사용 가능) + Azure Kinect × 2 (or ZED Mini 1대)
  - 2대 병렬 수집: 같은 task, 다른 object positions
  - 데이터 2배 수집 = 동일 인력 시간 기준
  - Fleet 학습: merge dataset → 공통 policy → 분기 없음 (SmolVLA 단일 모델)

[결론]:
  - "2× consumer arm fleet self-improvement" = MEDIUM confidence 갭
  - 자체적으로 CoRL 논문 될 정도의 기여는 아님
  - AR-Guided + Oracle 방향의 "실험 규모" 강화 요소로 포함 가능
  - "single human + 2 cheap robots = faster data collection" = 표 하나로 처리 가능
"""

# =============================================================================
# SECTION 8: REAL GAP ASSESSMENT (비판적 종합)
# =============================================================================

REAL_GAP_ASSESSMENT = """
비판적 종합: 실제로 CoRL 2026에서 쓸 수 있는 갭은 무엇인가?

[STRONG GAPS (검증 후 주장 가능)]:

GAP A: Consumer-Hardware VLA Self-Improvement Loop
  - 핵심 주장: SOAR/VLAC/SimpleVLA-RL 모두 $2K+ 로봇 or 클라우드 의존
  - 우리: $130 RoArm-M3 + local Qwen2.5-VL 3B + SmolVLA 450M
  - 검증 필요: SimpleVLA-RL 로봇 확인 (CRITICAL)
  - Confidence: MEDIUM → HIGH (SimpleVLA-RL 검증 후)

GAP B: Local VLM Judge for Success Detection (Edge-Deployable)
  - 핵심 주장: 모든 선행연구는 GPT-4V 또는 대형 클라우드 모델 사용
  - 우리: Qwen2.5-VL 3B, 로컬, 4-bit quant, RTX 4090
  - 차별점: API 비용 없음, 인터넷 없음, privacy 보장
  - 검증 필요: "edge VLM reward labeling robot" 검색
  - Confidence: MEDIUM

GAP C: Flow-Matching VLA Denoising Ensemble Variance as Active Learning Signal
  - 핵심 주장: diffusion/flow-matching의 multi-sample variance = uncertainty →
               어디서 더 데이터 필요한지 자동 감지
  - 우리: SmolVLA 10-step ODE, N=5 샘플, variance 계산
  - 검증 필요: "diffusion policy active learning uncertainty" + "flow matching uncertainty robot"
  - Confidence: HIGH (mechanism 새로움) / MEDIUM (선행연구 완전 부재 확인 필요)

[WEAK GAPS (단독으로는 불충분)]:

GAP D: SmolVLA-specific RL
  - SmolVLA 자체가 2025-06 논문 → RL 후속 거의 없을 것
  - BUT: "가장 가벼운 VLA로 RL" = 컴퓨팅 효율 논문 될 수 있음
  - Confidence: MEDIUM

GAP E: Fleet learning at consumer scale
  - 자체 논문으로는 약함, 보조 실험으로 좋음
  - Confidence: MEDIUM

[NOT A GAP (이미 존재함)]:
  - VLA RL fine-tuning: SimpleVLA-RL, OpenVLA-OFT 등
  - VLM-as-judge for robot: VLAC, SOAR
  - Self-improving robot: SOAR, AutoRT
  - Reward-weighted BC: RA-BC, SimpleVLA-RL
  - Data quality filtering: MimicGen, SOAR

[POSITIONING RECOMMENDATION]:
  현재 CoRL 방향 (AR-Guided Collection + Quality Oracle) vs Self-Improvement 비교:

  AR-Guided (현재 방향):
  + 더 명확한 "문제 정의" (수동 수집의 품질 불균일)
  + 실증 데이터 있음 (v1 0% vs v3 100%)
  + SigLIP frozen 제약이 핵심 연구 동기로 작동
  + Phase-Selective augmentation = 논문 가능 기여
  - consumer hardware 포지셔닝이 약함

  Self-Improvement (새 방향):
  + "로봇이 스스로 개선" = CoRL 어필 강함
  + Local VLM judge = 명확한 기술 기여
  + Denoising variance = unique mechanism
  - 더 많은 구현 필요 (60일 남음)
  - SimpleVLA-RL 검증 전까지 갭 불확실

  최적 전략: AR-Guided (메인) + Self-Improvement (Chapter 5, 확장)
  CoRL = AR-Guided Data Collection with Quality Oracle
  Thesis Chapter 5 = Autonomous Self-Improvement with Local VLM Judge
"""

# =============================================================================
# SECTION 9: VERIFICATION CHECKLIST (다음 단계)
# =============================================================================

VERIFICATION_CHECKLIST = {
    "CRITICAL_1": {
        "task": "SimpleVLA-RL (arXiv 2509.09674) 논문 확인",
        "what_to_check": "1) 어떤 로봇? 2) 어떤 VLA? (flow-matching vs token-based) 3) 클라우드 vs 로컬?",
        "why_critical": "이것이 flow-matching + consumer arm이면 GAP A, C 상당히 줄어듦",
        "time_estimate": "30분",
    },
    "CRITICAL_2": {
        "task": "VLAC (arXiv 2509.15937) 세부 확인",
        "what_to_check": "1) VLM critic 크기? (3B 이하?) 2) 로컬 실행 가능? 3) 어떤 로봇?",
        "why_critical": "Local VLM judge 갭 (GAP B) 존재 여부 결정",
        "time_estimate": "30분",
    },
    "CRITICAL_3": {
        "task": "2303.01488 확인",
        "what_to_check": "SOAR과 동일 논문인가? 다른 논문이면 robot/mechanism 확인",
        "why_critical": "선행연구 중복 체크",
        "time_estimate": "10분",
    },
    "MEDIUM_1": {
        "task": "Diffusion Policy + Active Learning + Uncertainty 검색",
        "keywords": [
            "diffusion policy uncertainty active learning",
            "flow matching robot uncertainty",
            "ensemble diffusion policy manipulation",
            "denoising variance robot manipulation",
        ],
        "time_estimate": "1시간",
    },
    "MEDIUM_2": {
        "task": "Edge VLM reward labeling 검색",
        "keywords": [
            "small VLM robot success detection",
            "edge vision language model reward",
            "local VLM reward labeling manipulation",
            "Qwen VLM robot reward",
        ],
        "time_estimate": "45분",
    },
    "MEDIUM_3": {
        "task": "On-the-Fly VLA Adaptation (2601.06748) 확인",
        "what_to_check": "flow-matching 지원? SmolVLA 호환? 로봇/compute 요구사항?",
        "time_estimate": "30분",
    },
}

# =============================================================================
# SECTION 10: SELF-IMPROVEMENT LOOP TECHNICAL DESIGN
# =============================================================================

SELF_IMPROVEMENT_TECHNICAL = """
SmolVLA 기반 Self-Improvement Loop — 기술 설계 (구현 전 검토용)

[LOOP 구조]:
  Phase 1: Autonomous Practice
    - deploy_smolvla.py (자율 실행, n=20-30 episodes)
    - 각 episode = 30sec. 20 episodes = 10분

  Phase 2: Success Detection (Local VLM Judge)
    - 에피소드 마지막 프레임 → Qwen2.5-VL 3B
    - Prompt: "Is the [task] successfully completed in this image? Answer YES/NO."
    - 비용: ~200ms/judgment. 20 judgments = 4초
    - VRAM: SmolVLA 언로드 → Qwen 로드 (순차)

  Phase 3: Denoising Variance Filtering
    - 낮은 성공률 상황에서 uncertainty 높은 상태 식별
    - 방법: deploy 중 각 step에서 N=3 샘플 → variance 계산
    - variance > threshold → 이 상태는 추가 data collection 필요
    - 인간에게: "이 위치에서 더 데모 보여주세요"

  Phase 4: Dataset Update + Retrain
    - 성공 에피소드 → 기존 dataset에 추가
    - reward-weighted BC: success (r=1), fail (r=0)
    - SmolVLA forward(reduction='none') × reward → weighted loss
    - run_official_train.py로 재학습

  Phase 5: 반복 (3 round loop)

[코드 수정 범위]:
  - deploy_smolvla.py: success_label 저장 추가 (~10줄)
  - new file: model_success_detector.py (Qwen2.5-VL judge)
  - new file: model_denoising_uncertainty.py (variance sampling)
  - new file: model_reward_weighted_train.py (weighted loss wrapper)
  - run_official_train.py: reward filtering 옵션 추가

[핵심 위험]:
  1. Qwen2.5-VL 3B 판단 정확도: 로봇 manipulation context에서 얼마나 정확?
     - 선행연구: VoxPoser에서 GPT-4V 사용, 정확도 ~85-90% (가정)
     - Qwen2.5-VL 3B: 훨씬 작음 → 정확도 낮을 수 있음
     - 해결: 신뢰도 threshold 적용 (Qwen이 불확신하면 human에게 위임)
  2. 노이즈 레이블로 재학습 → 성능 저하:
     - 해결: 성공 에피소드만 추가 (실패 데이터는 추가 안 함, filtered BC)
  3. Distribution shift:
     - 점점 bias된 데이터로 학습 → 새로운 OOD 발생
     - 해결: 기존 74ep 데이터 유지, 새 데이터만 추가
"""

# =============================================================================
# SECTION 11: CoRL 2026 FINAL POSITIONING
# =============================================================================

CORL_2026_POSITIONING = """
CoRL 2026 포지셔닝 최종 권장안 (65일 남음, 2026-03-25 기준)

[권장: AR-Guided가 메인, Self-Improvement는 보조]

이유:
  1. Self-Improvement 갭 검증 미완료 (SimpleVLA-RL 확인 전)
  2. 구현 시간: AR-Guided = 3-4주. Self-Improvement = 5-6주 (추가 구현)
  3. AR-Guided의 실증 데이터 (v1 0% vs v3 100%) = 이미 확보
  4. "문제-중심" 구조 = AR-Guided가 더 명확한 문제 정의

[Self-Improvement를 보조로 포함하는 방법]:
  - Main experiment: AR-Guided Collection + Quality Oracle
  - Ablation/Extension: self-improvement loop (without AR guidance) vs with AR guidance
  - Table: [No guidance, quality=random] vs [AR guidance, quality=oracle] vs [AR+self-improvement]
  - "자연스러운 다음 단계" = 데이터 수집 후 자율 개선

[만약 Self-Improvement가 Strong GAP으로 확인되면]:
  - 방향 전환 가능 (3/28 이전 결정해야 함)
  - 2주 이내 prototype 구현 → 결과 보고
  - 결정 기준: SimpleVLA-RL이 consumer arm + flow-matching이 아닌 경우

[논문 제목 후보]:
  Option 1 (AR-Guided 메인):
    "AR-Guided Data Collection and Quality-Driven Adaptation for Consumer Robot VLA"
  Option 2 (Self-Improvement 메인):
    "Autonomous VLA Self-Improvement on Consumer Hardware via Local VLM Reward"
  Option 3 (통합):
    "Closed-Loop VLA Adaptation: From Guided Collection to Autonomous Self-Improvement"
"""

if __name__ == "__main__":
    print("[B1 VLA MODEL] Self-Improvement Gap Analysis")
    print("=" * 60)
    print("\nSTRONG GAPS (HIGH/MEDIUM-HIGH confidence):")
    print("  GAP A: Consumer-grade VLA self-improvement ($130 arm)")
    print("         → verify SimpleVLA-RL robot FIRST")
    print("  GAP B: Local VLM judge (3B, edge) for success detection")
    print("         → SOAR/VLAC use cloud/large models")
    print("  GAP C: Flow-matching denoising variance as uncertainty proxy")
    print("         → SmolVLA 10-step ODE, N=5 sampling = unique mechanism")
    print("\nCRITICAL VERIFICATIONS (do before claiming gap):")
    for k, v in VERIFICATION_CHECKLIST.items():
        if k.startswith("CRITICAL"):
            print(f"  [{k}] {v['task']}")
    print("\nRECOMMENDATION:")
    print("  AR-Guided (main) + Self-Improvement (Chapter 5/ablation)")
    print("  Direction lock: 2026-03-28 (after verification)")
