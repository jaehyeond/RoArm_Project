# VLA 관련 논문 모음

2024-2026 VLA(Vision-Language-Action) 연구 현황 파악을 위한 논문 링크 모음.
연구 아이디어 탐색 과정에서 발견한 논문들을 분류별로 정리.

---

## 1. Safety / Uncertainty in VLA

CC-VLA 아이디어와 직접 관련된 핵심 논문들.

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **Confidence Calibration in VLA** | 2025.07 | Post-hoc confidence 보정만 (행동 변경 안함) → **CC-VLA gap의 근거** | [arxiv 2507.17383](https://arxiv.org/abs/2507.17383) |
| **Evaluating Uncertainty and Quality of VLA** | 2025.07 | 13개 uncertainty metric 제안, 평가 프레임워크 | [arxiv 2507.17049](https://arxiv.org/abs/2507.17049) |
| **SafeVLA** (NeurIPS 2025 Spotlight) | 2025.03 | CMDP로 unsafe 행동 제약, violation 83.58% 감소 | [arxiv 2503.03480](https://arxiv.org/abs/2503.03480) |
| **VLSA / AEGIS** | 2025.12 | Control Barrier Function plug-and-play safety layer | [arxiv 2512.11891](https://arxiv.org/abs/2512.11891) |
| **CompliantVLA** | 2026.01 | Variable Impedance로 접촉 안전 | [arxiv 2601.15541](https://arxiv.org/abs/2601.15541) |
| **VLAC** (Vision-Language-Action-Critic) | 2025.09 | Critic model로 dense reward 생성, real-world RL | [arxiv 2509.15937](https://arxiv.org/abs/2509.15937) |
| **Uncertainty-aware Latent Safety Filters** | 2025.05 | Epistemic uncertainty로 OOD 필터링 | [arxiv 2505.00779](https://arxiv.org/abs/2505.00779) |
| **LIBERO-Plus** | 2025.10 | VLA robustness 벤치마크 | [arxiv 2510.13626](https://arxiv.org/abs/2510.13626) |
| **CRT** (Corruption Restoration Transformer) | 2026.02 | 시각 입력 손상 복원 (VLA 90%→2% 성능하락 해결) | [arxiv 2602.01158](https://arxiv.org/abs/2602.01158) |

---

## 2. Chain-of-Thought / Reasoning in VLA

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **CoT-VLA** (CVPR 2025) | 2025.03 | Visual CoT → future image 예측 후 action | [arxiv 2503.22020](https://arxiv.org/abs/2503.22020) |
| **FlowVLA** | 2025.08 | Motion reasoning + visual chain of thought | [arxiv 2508.18269](https://arxiv.org/abs/2508.18269) |
| **CoA-VLA** (ICCV 2025) | 2025 | Chain-of-Affordance reasoning | [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_CoA-VLA_Improving_Vision-Language-Action_Models_via_Visual-Text_Chain-of-Affordance_ICCV_2025_paper.pdf) |

---

## 3. Future Prediction / Anticipatory VLA

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **UP-VLA** | 2025.01 | Understanding + Future Image Prediction | [arxiv 2501.18867](https://arxiv.org/abs/2501.18867) |
| **HiF-VLA** | 2025.12 | Hindsight + Insight + Foresight, Motion Vector | [arxiv 2512.09928](https://arxiv.org/abs/2512.09928) |

---

## 4. Additional Modality VLA (포화 영역)

| 논문 | 날짜 | 추가 Modality | 링크 |
|------|------|-------------|------|
| **Tactile-VLA** | 2025.07 | 촉각 센서 (haptics) | [arxiv 2507.09160](https://arxiv.org/abs/2507.09160) |
| **Audio-VLA** | 2025.11 | 접촉 소리 (contact audio) | [arxiv 2511.09958](https://arxiv.org/abs/2511.09958) |
| **ForceVLA** | 2025 | 6축 힘센서 (force-aware MoE) | [OpenReview](https://openreview.net/forum?id=2845H8Ua5D) |
| **TA-VLA** (Torque-aware) | 2025.09 | 토크 신호 (decoder에 통합) | [arxiv 2509.07962](https://arxiv.org/abs/2509.07962) |
| **SVA** (Speech-VLA) | 2025 | 음성 명령 (spoken commands) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S003132032501578X) |

---

## 5. Preference / Alignment / Style

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **GRAPE** (ICLR 2025) | 2024.11 | Trajectory-level preference alignment | [arxiv 2411.19309](https://arxiv.org/abs/2411.19309) |

---

## 6. Memory / Cross-Embodiment (포화 영역)

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **MemoryVLA** | 2025 | 메모리 증강 VLA | [검색 필요] |
| **MAP-VLA** | 2025 | Memory-Augmented Planning | [검색 필요] |
| **EchoVLA** | 2025 | Episodic Memory | [검색 필요] |
| **ET-VLA** | 2025 | Cross-embodiment transfer | [검색 필요] |
| **X-VLA** | 2025 | Cross-embodiment | [검색 필요] |

---

## 7. Reversibility / Safe RL (VLA 미적용 - Gap 존재)

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **Google RAE/RAC** | 2023 | Self-Supervised Reversibility-Aware RL | [Google Blog](https://research.google/blog/self-supervised-reversibility-aware-reinforcement-learning/) |
| **ReVLA** (다른 개념) | 2025 | Reversible training (메모리 효율), action reversibility 아님 | [검색 필요] |

---

## 8. Architecture / Efficiency

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **GR00T N1** (NVIDIA) | 2025.03 | System 1 + System 2 dual architecture | [NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-isaac-gr00t-n1-open-foundation-model-for-humanoid-robots/) |
| **Spatial Forcing** | 2025.10 | 3D spatial representation alignment | [arxiv 2510.12276](https://arxiv.org/abs/2510.12276) |
| **BayesianVLA** | 2026.01 | Bayesian decomposition via latent action queries | [arxiv 2601.15197](https://arxiv.org/abs/2601.15197) |
| **Spec-VLA** | 2025 | Speculative decoding for VLA | [EMNLP 2025 PDF](https://aclanthology.org/2025.emnlp-main.1367.pdf) |
| **CEED-VLA** | 2025 | Consistency + early-exit decoding | [검색 필요] |

---

## 9. Contrastive / Reward-Conditioned

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **CLASS** | 2025.08 | Action sequence contrastive learning | [arxiv 2508.01600](https://arxiv.org/abs/2508.01600) |
| **RECAP** | 2025 | Advantage-conditioned VLA (pi*0.6) | [검색 필요] |

---

## 10. Survey / Overview 논문

전체 VLA 분야 파악에 필수적인 서베이 논문들.

| 논문 | 날짜 | 범위 | 링크 |
|------|------|------|------|
| **VLA Models: Concepts, Progress, Applications and Challenges** | 2025.05 | VLA 전반 개념/진행/과제 | [arxiv 2505.04769](https://arxiv.org/abs/2505.04769) |
| **Pure Vision Language Action Models: A Comprehensive Survey** | 2025.09 | Pure VLA 종합 서베이 | [arxiv 2509.19012](https://arxiv.org/abs/2509.19012) |
| **Large VLM-based VLA Models: A Survey** | 2025.08 | 대형 VLM 기반 VLA 서베이 | [arxiv 2508.13073](https://arxiv.org/abs/2508.13073) |
| **VLA Models in Robotic Manipulation: A Systematic Review** | 2025.07 | Manipulation 특화 서베이 | [arxiv 2507.10672](https://arxiv.org/abs/2507.10672) |
| **A Survey on Efficient VLA Models** | 2025.10 | VLA 효율화 서베이 | [arxiv 2510.24795](https://arxiv.org/abs/2510.24795) |
| **Action Tokenization Perspective Survey** | 2025.07 | Action 토큰화 관점 서베이 | [arxiv 2507.01925](https://arxiv.org/abs/2507.01925) |
| **Multimodal Fusion with VLA: Systematic Review** | 2025 | 멀티모달 융합 서베이 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1566253525011248) |
| **Recipe for VLA in Robotic Manipulation** | 2025 | VLA 레시피 서베이 | [TechRxiv PDF](https://www.techrxiv.org/users/956965/articles/1326137/master/file/data/eart2025recipevla/eart2025recipevla.pdf?inline=true) |
| **RL of VLA for Robotic Manipulation Survey** | 2025 | RL + VLA 서베이 | [TechRxiv PDF](https://www.techrxiv.org/users/934012/articles/1366553/master/file/data/TechrxivA_Survey_on_Reinforcement_Learning_of_Vision-Language-Action_Models_for_Robotic_Manipulation/TechrxivA_Survey_on_Reinforcement_Learning_of_Vision-Language-Action_Models_for_Robotic_Manipulation.pdf?inline=true) |

---

## 11. Test-Time Adaptation / Compute Scaling (포화 영역)

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **TT-VLA** | 2026.01 | Test-time RL로 VLA 적응 | [arxiv 2601.06748](https://arxiv.org/abs/2601.06748) |
| **SITCOM** | 2025.10 | Inference-time compute scaling (MPC 영감) | [arxiv 2510.04041](https://arxiv.org/abs/2510.04041) |
| **Dynamic TTC in Control** | 2025 | Difficulty-aware adaptive inference (Easy/Medium/Hard) | [OpenReview](https://openreview.net/pdf?id=oDoPiR8wZJ) |
| **HyperVLA** | 2025.10 | Hypernetwork 기반 few-shot adaptation | [arxiv 2510.04898](https://arxiv.org/abs/2510.04898) |
| **PLD (Self-Improving VLA)** | 2025.11 | Residual RL + distribution-aware data collection | [arxiv 2511.00091](https://arxiv.org/abs/2511.00091) |

---

## 12. In-Context Learning / Few-Shot (활발)

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **MoS-VLA** | 2025.10 | One-shot skill adaptation (gradient-free) | [arxiv 2510.16617](https://arxiv.org/abs/2510.16617) |
| **RoboPrompt** | 2025.03 | ICL로 LLM에서 직접 robot action 예측 | [arxiv 2410.12782](https://arxiv.org/abs/2410.12782) |
| **Interleave-VLA** | 2025.05 | Interleaved image-text instruction | [arxiv 2505.02152](https://arxiv.org/abs/2505.02152) |
| **UniVLA** | 2025.05 | Task-centric latent actions, zero-shot | [arxiv 2505.06111](https://arxiv.org/abs/2505.06111) |
| **DEMONSTRATE** | 2025.07 | Zero-shot via multi-task demo learning | [arxiv 2507.12855](https://arxiv.org/abs/2507.12855) |

---

## 13. Data Quality / Self-Improvement (gap 있음)

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **Consistency Matters** | 2024.12 | Demo data quality metric 정의 | [arxiv 2412.14309](https://arxiv.org/abs/2412.14309) |
| **PLD** | 2025.11 | Residual RL로 failure region 데이터 자동 수집 | [arxiv 2511.00091](https://arxiv.org/abs/2511.00091) |
| **EMMA** | 2025.09 | Challenging sample 동적 reweighting | [arxiv 2509.22407](https://arxiv.org/abs/2509.22407) |
| **RESample** | 2025.10 | Robust data augmentation for manipulation | [arxiv 2510.17640](https://arxiv.org/abs/2510.17640) |

---

## 14. Language-to-Impedance / Compliance (포화)

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **HumanoidVLM** | 2026.01 | VLM으로 impedance parameter 선택 (RAG) | [arxiv 2601.14874](https://arxiv.org/abs/2601.14874) |
| **CompliantVLA** | 2026.01 | Variable impedance for contact | [arxiv 2601.15541](https://arxiv.org/abs/2601.15541) |

---

## 15. ICLR 2026 주요 논문

| 논문 | 핵심 내용 | 링크 |
|------|----------|------|
| **UniVLA** | 통합 multimodal VLA, discrete token | [OpenReview](https://openreview.net/forum?id=PklMD8PwUy) |
| **Unified Diffusion VLA** | Joint discrete diffusion | [OpenReview](https://openreview.net/forum?id=UvQOcw2oCD) |
| **Vlaser** | Synergistic embodied reasoning | [OpenReview](https://openreview.net/forum?id=8xTDnj39Ti) |
| **XR-1** | Cross-embodiment via VQ-VAE | [OpenReview](https://openreview.net/forum?id=XJclc9Eabd) |
| **InstructVLA** | Instruction tuning preservation | [GitHub](https://github.com/InternRobotics/InstructVLA) |
| **ICLR 2026 VLA 분석 블로그** | 164편 트렌드 분석, gap 지적 | [Moritz Reuss Blog](https://mbreuss.github.io/blog_post_iclr_26_vla.html) |

---

## 읽기 우선순위 (3차 업데이트)

### 1순위: 반드시 읽기
1. [ICLR 2026 VLA 분석 블로그](https://mbreuss.github.io/blog_post_iclr_26_vla.html) - 전체 동향 + gap 지적
2. [Confidence Calibration in VLA](https://arxiv.org/abs/2507.17383) - CC-VLA gap의 근거
3. [SafeVLA](https://arxiv.org/abs/2503.03480) - Safety VLA 최강 경쟁자 (NeurIPS Spotlight)
4. [Consistency Matters](https://arxiv.org/abs/2412.14309) - Data quality metric (DQ-VLA gap 근거)

### 2순위: 방향 결정에 도움
5. [VLSA/AEGIS](https://arxiv.org/abs/2512.11891) - Safety constraint layer
6. [PLD (Self-Improving)](https://arxiv.org/abs/2511.00091) - Data quality 자동화 경쟁자
7. [MoS-VLA](https://arxiv.org/abs/2510.16617) - One-shot adaptation
8. [VLA Concepts Survey](https://arxiv.org/abs/2505.04769) - 전체 VLA 파악

### 3순위: 서베이/배경
9. [Pure VLA Survey](https://arxiv.org/abs/2509.19012) - 종합 서베이
10. [VLAC](https://arxiv.org/abs/2509.15937) - Critic 접근
11. [CRT](https://arxiv.org/abs/2602.01158) - Robustness 문제
12. 나머지 modality/CoT 논문들

---

## 16. Manipulation 특화 VLA (4차 탐색에서 추가)

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **Oat-VLA** | 2025 | Object-Agent-centric Tokenization, 2x 빠른 수렴 | [ResearchGate](https://www.researchgate.net/publication/395972544) |
| **DynamicVLA** | 2026.01 | 동적 물체 조작, temporal reasoning (0.4B) | [arxiv 2601.22153](https://arxiv.org/abs/2601.22153) |
| **ObjectVLA** | 2025.02 | End-to-end open-world object manipulation | [arxiv 2502.19250](https://arxiv.org/abs/2502.19250) |
| **GraspVLA** | 2025.05 | Billion-frame grasping dataset, Progressive Action Gen | [arxiv 2505.03233](https://arxiv.org/abs/2505.03233) |
| **VLA-Grasp** | 2025 | Cross-modality fusion for task-oriented grasping | [Springer](https://link.springer.com/article/10.1007/s40747-025-01893-x) |
| **DexGraspVLA** | 2025.02 | Dexterous grasping framework | [HuggingFace](https://huggingface.co/papers?q=DexGraspVLA) |
| **VTLA** | 2025.05 | Vision-Tactile-Language-Action, peg insertion 90%+ | [arxiv 2505.09577](https://arxiv.org/abs/2505.09577) |
| **Shake-VLA** | 2025 | Bimanual manipulation + liquid mixing | [ACM](https://dl.acm.org/doi/10.5555/3721488.3721686) |
| **Long-VLA** | 2025.08 | Long-horizon manipulation, phase-aware masking | [arxiv 2508.19958](https://arxiv.org/abs/2508.19958) |

---

## 17. Spatial / 3D Reasoning VLA

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **SpatialVLA** | 2025.01 | Ego3D Position Encoding, adaptive action grid | [arxiv 2501.15830](https://arxiv.org/abs/2501.15830) |
| **GraphCoT-VLA** | 2025.08 | 3D graph-based spatial-aware reasoning | [arxiv 2508.07650](https://arxiv.org/abs/2508.07650) |
| **Spatial Forcing** | 2025.10 | Implicit spatial representation | [arxiv 2510.12276](https://arxiv.org/abs/2510.12276) |
| **DepthVLA** | 2025 | Depth transformer expert, 78.5% real-world | [HuggingFace](https://huggingface.co/papers?q=DepthVLA) |

---

## 18. Ensemble / Voting / Uncertainty Estimation

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **VOTE** | 2025.07 | Trajectory Ensemble Voting, 39x faster, 46Hz | [arxiv 2507.05116](https://arxiv.org/abs/2507.05116) |
| **PI-VLA** | 2026.01 | AURD: action disagreement + prediction error 모니터링 | [Preprints.org](https://www.preprints.org/manuscript/202601.0682) |
| **Embodied ToT** | 2025.12 | Tree of Thoughts + physics digital twin | [arxiv 2512.08188](https://arxiv.org/abs/2512.08188) |
| **VLA-Reasoner** | 2025 | MCTS for candidate action evaluation | [ICLR Blog](https://mbreuss.github.io/blog_post_iclr_26_vla.html) |

---

## 19. Efficient Fine-tuning / Adaptation

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **OpenVLA-OFT** | 2025.02 | 최적 fine-tuning recipe, 97.1% LIBERO, 26x 빠름 | [arxiv 2502.19645](https://arxiv.org/abs/2502.19645) |
| **LoRA-VLA** | 2025.12 | 8GB VRAM, 200 에피소드, consumer GPU | [arxiv 2512.11921](https://arxiv.org/abs/2512.11921) |
| **VLA-RFT** | 2025 | RL fine-tuning, <400 steps | [OpenReview](https://openreview.net/forum?id=Jaut99EHeu) |
| **VLA-RL** | 2025.05 | Online RL로 VLA 개선 | [arxiv 2505.18719](https://arxiv.org/abs/2505.18719) |
| **Green-VLA** | 2025 | Multi-embodiment, staged training recipe | [연구 페이지](https://mbreuss.github.io/blog_post_iclr_26_vla.html) |
| **Soft Robot VLA** | 2025.10 | VLA를 soft robot에 fine-tuning | [arxiv 2510.17369](https://arxiv.org/abs/2510.17369) |

---

## 20. World Model / Sim2Real VLA

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **WorldVLA** | 2025 | Autoregressive action world model | [연구 페이지](https://embodied-world-models.github.io/) |
| **Sim2Real-VLA** | 2025 | Synthetic-only training, chain-of-affordances | [OpenReview](https://openreview.net/forum?id=H4SyKHjd4c) |
| **Cosmos Policy** | 2025 | NVIDIA video model fine-tuning for control | [NVIDIA](https://nvidianews.nvidia.com) |
| **WALL-A** | 2025 | VLA + world model + causal inference | [X Square Robot](https://www.therobotreport.com/x-square-robot-secures-140m-in-funding-for-ai-foundation-models/) |

---

## 21. Counterfactual / Data Augmentation

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **RoCoDA** | 2025 | Scene-level counterfactual data augmentation | [ResearchGate](https://www.researchgate.net/publication/395230290) |
| **CRED** | 2025 | Counterfactual reasoning for preference learning | [Yi-Shiuan Tung Blog](https://yi-shiuan-tung.github.io/blog/2025/cred/) |
| **Creative Tool Use** | 2025 | Counterfactual tool substitution via 3D editing | [OpenReview](https://openreview.net/forum?id=OvzxjANR83) |
| **FailSafe-VLM** | 2025.10 | 체계적 failure 생성 + recovery action 수집 | [arxiv 2510.01642](https://arxiv.org/abs/2510.01642) |

---

## 22. Action Chunking / Efficiency

| 논문 | 날짜 | 핵심 내용 | 링크 |
|------|------|----------|------|
| **PD-VLA** | 2025.03 | Parallel decoding + action chunking, 2.52x 빠름 | [arxiv 2503.02310](https://arxiv.org/abs/2503.02310) |
| **FAST** | 2025 | Variable-length token compression | [ICLR Blog](https://mbreuss.github.io/blog_post_iclr_26_vla.html) |
| **RTC** | 2025 | Real-time chunking, predict-while-execute | [arxiv 2601.20130](https://arxiv.org/abs/2601.20130) |
| **VLA-Cache** | 2025.02 | Adaptive token caching, ~14Hz | [arxiv 2502.02175](https://arxiv.org/abs/2502.02175) |
| **DeeR-VLA** | 2025 | Early termination based on complexity | [연구 페이지](https://arxiv.org/html/2510.24795v1) |
