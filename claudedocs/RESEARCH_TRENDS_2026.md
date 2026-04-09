# Research Trends 2026 — Domain Reference

> **Purpose**: 유저가 세션마다 던지는 "도메인 소개" 자료를 누적 저장. 파이프라인 코드(`*.py`, `lerobot_dataset_v*`, `outputs/`)와 완전히 분리된 참고자료.
> **Rule**: HARD RULE #4 정신 — 모든 주장은 1차 소스(arXiv ID / DOI / URL) + 검증 상태(✅/⚠️/❌) 명기. 미검증 재인용 금지. 스크린샷·LLM 출력 mislabeling이 발견되면 정정 기록 유지.
> **Scope**: 연구 방향 *결정*은 이 문서에 쓰지 않음. 방향 결정은 `project_corl2026_direction.md` / `feedback_baseline_first.md` / `experiment_log*.md` 에서. 이 문서는 **언제든 참조 가능한 도메인 knowledge base**.

---

## Index

| # | 도메인 | 추가 날짜 | v6 gripper와의 직접 관련성 |
|---|---|---|---|
| 1 | [LLM-as-Planner / Code-as-Policy](#1-llm-as-planner--code-as-policy) | 2026-04-09 | △ (GenCHiP만 부분 접점, 나머지는 abstraction layer 불일치) |
| 2 | *(빈 슬롯 — 다음 도메인)* | — | — |

---

## 1. LLM-as-Planner / Code-as-Policy

### 1.1 개요
"LLM에게 SDK/API/primitive library를 던져서 로봇을 제어한다"는 패러다임. 학계에서는 2022년에 **LLM-as-Planner** 혹은 **Code-as-Policy** 로 정립되었고, 2024-2025년에 contact-rich / hierarchical 방향으로 확장 중. 주요 아이디어: 자연어 명령 → LLM이 구조화된 계획(텍스트 또는 Python 코드) 생성 → perception API + control primitive API 호출 → 실행.

대조: VLA end-to-end(SmolVLA/OpenVLA/π₀) 는 pixel→action 직접 맵. LLM-as-Planner는 pixel→(VLM/perception)→**symbolic/code**→primitive→action으로 **추상 레이어가 명시적**.

### 1.2 Landmark Papers (1차 소스 검증 완료)

#### ① SayCan (Google, 2022) — LLM-as-Planner 원조
- **1차 소스**: arXiv **[2204.01691](https://arxiv.org/abs/2204.01691)** — *"Do As I Can, Not As I Say: Grounding Language in Robotic Affordances"*
- **저자**: Michael Ahn, Anthony Brohan, Noah Brown et al. (Google Robotics / Everyday Robots)
- **핵심 아이디어**: LLM이 multi-step plan을 후보로 내고, **affordance 필터**(학습된 value function)가 "물리적으로 지금 실행 가능한 액션"만 통과. LLM 지식 + 로봇 현실의 grounding gap을 value function으로 메움.
- **검증 상태**: ✅ (arXiv + 다수 언급 확인)
- **v6 관련성**: △ 추상 레이어 불일치. SayCan은 "pick up the sponge"를 고르고 그 아래 primitive는 별도 학습된 behavior policy가 담당. SmolVLA flow matching이 tail을 못 뽑는 우리 low-level 문제를 SayCan은 건드리지 않음.

#### ② Code as Policies (Google, 2022) — 코드-중심의 전환점
- **1차 소스**: arXiv **[2209.07753](https://arxiv.org/abs/2209.07753)** — *"Code as Policies: Language Model Programs for Embodied Control"*
- **저자**: Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, Andy Zeng
- **프로젝트 페이지**: https://code-as-policies.github.io/ (✅ fetch 확인)
- **GitHub**: https://github.com/google-research/google-research/tree/master/code_as_policies
- **핵심 아이디어**: LLM(code-trained)이 perception API(`detect_object`, `get_pose`) + control primitive(`move_to`, `grasp`) 를 호출하는 **Python 코드를 생성**. Few-shot prompting + 3rd-party lib(NumPy, Shapely) 체이닝으로 정량 계산도 가능.
- **벤치마크**: Hierarchical code generation이 HumanEval에서 39.8% 달성 (LLM이 helper function을 재귀적으로 정의).
- **실증 영역**: tabletop manipulation, whiteboard drawing, mobile manipulation.
- **공식 한계**: "long pauses between commands and responses are mostly caused by OpenAI API query times and rate limiting" → 실시간 제어 inner loop에는 LLM을 두지 못함.
- **검증 상태**: ✅ (홈페이지 + arXiv + GitHub 3중 확인)
- **v6 관련성**: △ 같은 추상 레이어 문제. 단 "primitive library 설계" 철학은 §1.6 접점 후보에 활용 가능.

#### ③ ProgPrompt (USC + NVIDIA, 2022) — 프로그램 생성 > 텍스트 플래닝
- **1차 소스**: arXiv **[2209.11302](https://arxiv.org/abs/2209.11302)** — *"ProgPrompt: Generating Situated Robot Task Plans using Large Language Models"*
- **저널 버전**: Autonomous Robots **[10.1007/s10514-023-10135-3](https://link.springer.com/article/10.1007/s10514-023-10135-3)** (2023)
- **저자**: Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, Animesh Garg
- **핵심 아이디어**: 순수 텍스트 플래닝보다 **프로그램 형식(Python-like pseudocode)** 생성이 우수함을 보임. 이유: 프로그램은 available objects/actions를 import 구문 + 함수 시그니처로 **환경에 직접 grounding**시키기 쉬움 → hallucination 감소.
- **검증 상태**: ✅
- **v6 관련성**: △ 같은 layer 문제. 단 "환경 객체를 코드 scope에 명시적으로 노출"하는 아이디어는 primitive library 설계에 차용 가능.

#### ④ GenCHiP (Google DeepMind, 2024) — **contact-rich / high-precision 예외** ⭐
- **1차 소스**: arXiv **[2404.06645](https://arxiv.org/abs/2404.06645)** — *"GenCHiP: Generating Robot Policy Code for High-Precision and Contact-Rich Manipulation Tasks"*
- **저자**: Kaylee Burns, Ajinkya Jain, Keegan Go et al.
- **핵심 아이디어 (abstract 인용)**: *"LLMs have been successful at generating robot policy code, but so far these results have been limited to high-level tasks that do not require precise movement. It is an open question..."* — 즉, 2022년 CaP 라인이 contact-rich에서 실패한다는 점을 직접 문제화. 해결: **적절한 action space(compliant / force-aware primitive)** 를 정의해주면 LLM이 contact-rich 태스크도 코드 생성으로 풀 수 있음을 보임.
- **검증 상태**: ✅
- **v6 관련성**: ⭐ **오늘 5편 중 유일한 직접 접점**. 우리 gripper long-tail 실패는 정확히 "policy가 contact-rich action space의 tail을 학습 못 함" 현상. GenCHiP 처방을 v6에 번역하면: raw joint[5] 각도 직접 예측 대신 `close_until_contact(max_force)` / `grasp_commit(hold_ms=500)` 같은 **compliant primitive**를 VLA action space에 노출하고 policy는 trigger만 학습. tail sampling 부담 제거. → §1.6 후보.

#### ⑤ BrainBody-LLM (NYU Tandon, 2025) — hierarchical 2-LLM + sim feedback
- **1차 소스**: Wiley Advanced Robotics Research **[10.1002/adrr.202500072](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adrr.202500072)**
- **연구실**: NYU Tandon School of Engineering, Prof. Farshad Khorrami
- **GitHub**: https://github.com/llm-brainbodyllm/brainbodyllm (MIT license, created 2025-10-30, stars=2 at time of check)
- **언론**: TechXplore / NYU Tandon newsroom, 2025-11-28; Interesting Engineering 2025-11-30
- **핵심 아이디어**: 2개 LLM 계층 — "Brain LLM"이 고수준 플래닝을 담당, "Body LLM"이 저수준 제어를 담당, **시뮬레이터 에러 피드백 루프**로 Body가 Brain에게 실행 실패를 보고. 인간의 "brain/body coordination" 에서 영감.
- **검증 상태**: ✅ 논문+언론+GitHub 3중 확인. 단 GitHub star 2개 → **재현성은 약할 가능성**. 인용 시 직접 코드 돌려본 사람의 후속 기록 필요.
- **v6 관련성**: △ Sim feedback은 유저 Isaac Lab 포기(2026-03-26 결정, `critical_analysis_isaaclab_vla_scaling_20260326.md`)와 충돌. 단 "brain/body 분리" 메타포는 `project_corl2026_direction.md`의 3-VLA 비교 플랜과 교차 가능: brain=LLM 플래너, body=SmolVLA/Octo/π₀ 중 택일.

### 1.3 Surveys (스크린샷에서 원논문으로 오인됐던 URL 재분류)

#### ⑥ Frontiers 2025 Agentic LLM Robots Survey
- **1차 소스**: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1605405/full
- **제목**: *"Agentic LLM-based robotic systems for real-world applications: a review on their agenticness and ethics"*
- **저자**: Emmanuel K. Raptis, Athanasios Ch. Kapoutsis, Elias B. Kosmatopoulos (CERTH + Democritus University of Thrace), 2025
- **리뷰 범위**: 2022-2025 peer-reviewed 30편. 시뮬레이션-only 제외, 물리 로봇 실증 논문 중심. SayCan을 "foundational architecture"로 인용.
- **기여**: (a) agenticness classification framework (autonomy, goal-directedness, adaptability, decision-making), (b) ethics axes (bias, fairness, robustness, safety, oversight, explainability, compliance).
- **핵심 비판점**: "Most LLMs lack embodied grounding despite multimodal inputs. Real-time responsiveness constraints. Insufficient commonsense about physical constraints."
- **검증 상태**: ✅
- **용도**: Related Work에서 2025 VLA+LLM landscape 인용, ethics 축이 유저 "문제-중심" 프레임과 정합.

#### ⑦ Intelligent Service Robotics 2024 LLM+Robot Integration Survey
- **1차 소스**: https://link.springer.com/article/10.1007/s11370-024-00550-5
- **제목**: *"A survey on integration of large language models with intelligent robots"*
- **저널**: Intelligent Service Robotics, Vol 17, pp 1091–…, 2024-08-13 (Open Access Review)
- **검증 상태**: ✅
- **용도**: 2024년 기준 LLM+robot 통합 전반 landscape. ProgPrompt/CaP/SayCan 모두 포함.

### 1.4 ⚠️ 스크린샷 URL Mislabeling — 정정 기록

유저가 이 세션에서 제공한 자료(2026-04-09)에서 **서베이 논문 URL이 원논문 옆에 잘못 태그**된 사례 2건 발견. HARD RULE #4 준수 위해 기록:

| 스크린샷 표기 | 유저가 준 URL | 실제 논문 | 정정 |
|---|---|---|---|
| SayCan 옆 `Frontiers` 태그 | frobt.2025.1605405 | Raptis et al. 2025 *Agentic LLM robots* **서베이** | SayCan 1차 ≠ 이 Frontiers URL. SayCan 1차는 arXiv 2204.01691. Frontiers URL은 §1.3 ⑥에 survey로 재분류 |
| ProgPrompt 옆 `Springer` 태그 | s11370-024-00550-5 | Intelligent Service Robotics 2024 *LLM+robot integration* **서베이** | ProgPrompt 저널 버전은 다른 DOI(**s10514-023-10135-3**). s11370-024-00550-5는 §1.3 ⑦에 survey로 재분류 |

**교훈**: URL 도메인(frontiersin.org, springer.com)과 논문 이름이 "근처에 같이 나왔다"는 이유로 동일시하면 안 됨. DOI 수준에서 1차 소스를 직접 확인해야 함. 리뷰어가 가장 먼저 잡는 실수 유형.

### 1.5 v6 파이프라인과의 정합성 평가 (비판적)

**핵심 질문**: LLM-as-Planner / Code-as-Policy가 v6 gripper long-tail 실패(2026-04-08 분석: >80° 1.9%, episode-max median 67.4°, 모델 출력 max 72°)를 **직접** 치유하는가?

**결론: 대부분 NO. GenCHiP 한 편만 부분 접점.**

**근거**:
1. SayCan/CaP/ProgPrompt/BrainBody 4편은 전부 **high-level task decomposition** 레이어("집어→옮겨→놓아"). 우리 실패는 **low-level action distribution 병리** — flow matching이 tail을 sampling 못 함.
2. LLM이 "close gripper firmly"라고 말해도 SmolVLA가 출력하는 action[5]=72°라는 **수치 병리 자체는 무효화되지 않음**. LLM wrapper는 SmolVLA 내부 병리에 접근 불가.
3. 즉 LLM-as-Planner는 **SmolVLA 위에 얹는 조정자**이지 **SmolVLA 내부 치료제**가 아님. 잘못된 약방문.

**GenCHiP 예외의 구조**:
- GenCHiP은 "low-level primitive 자체를 force-aware / compliant로 재설계" 후에 LLM이 그 primitive를 호출. 즉 **action space 교체** 접근.
- v6에 번역: action[5]를 joint angle이 아닌 discrete primitive trigger(`close_until_contact`, `release`)로 교체하면 flow matching이 tail 각도 직접 예측하는 부담이 사라짐. 이건 SmolVLA 내부 구조를 건드리는 처방이므로 실제로 유효할 가능성 있음.
- 단 SmolVLA는 6-dim continuous action을 전제로 사전학습 → 6번째 차원을 discrete/primitive로 바꾸려면 **재학습 + custom head** 필요 → HARD RULE #2(커스텀 학습 금지) 와 충돌 여지 → 구현 전에 `lerobot-train` 내에서 가능한 범위로 재설계 필요.

### 1.6 잠재적 연구 접점 (baseline 달성 *이후* 판단)

> **주의**: 이 섹션은 **아이디어 카탈로그**일 뿐, 지금 구현하자는 제안이 아님. `feedback_baseline_first.md`에 따라 Stage 1 (v6 재배포 3-zone 평가) 달성 전엔 연구 방향 확정 금지.

| 접점 | 출발 논문 | v6 연결 | 비용 | HARD RULE 위반 여부 |
|---|---|---|---|---|
| **EAAS — Event-Augmented Action Space** ⭐ (2026-04-09 추가) | GenCHiP action space insight + 유저 v6 3점 ablation | C1 Event Detector (state-derivative, no ML) + C2 Dataset Re-parameterization (원본 불변, `lerobot-train` CLI 그대로) + C3 Online Wrapper (Stage 1 패치 일반화). Cross-VLA (SmolVLA + Octo/π₀) + Cross-task 축. 상세: **[EAAS_PROPOSAL.md](EAAS_PROPOSAL.md)** | 중 (Wk 2-6) | 없음 — HARD RULE #1~#10 전부 정합 검증 완료 |
| **Action primitive wrapping** | GenCHiP | action[5]를 compliant primitive(`close_until_contact`)로 교체 → flow matching tail 문제 해소 | 중 (custom head + 재학습) | #2 위반 가능성 → `lerobot-train` 내 재학습으로 제한 필수 |
| **LLM failure-recovery loop** | BrainBody body→brain feedback | 배포 로그(gripper max<threshold) → LLM이 "retry with firmer grasp" 코드 재생성 | 저 (lerobot 학습 건드리지 않음, deploy 레이어만) | 없음 |
| **Primitive library + SmolVLA trigger** | Code as Policies + SayCan | 저수준 primitive library(`home`, `reach`, `grasp`, `release`) 를 노출, VLA는 trigger 예측, primitive가 contact-rich 실행 | 중 | 없음 (lerobot 학습 외부) |
| **Agenticness/ethics 축 논문 포지셔닝** | Frontiers 2025 survey | 유저 CoRL 논문의 "consumer-scale constraint" 스토리를 agenticness axes로 정량화 | 저 (문서 작업만) | 없음 |

**추가로 검증된 prior art (2026-04-09 EAAS 반증 검색 중 확인)** — EAAS가 이들과 겹치지 않음을 `EAAS_PROPOSAL.md` §3에서 상세 방어:
- arXiv **2602.06512** — *Beyond the Majority: Long-tail Imitation Learning for Robotic Manipulation* (Zhu et al., **ICRA 2026**) — task-budget inter-task long-tail (EAAS = action-value intra-task, 층위 다름)
- arXiv **2505.22159** — *ForceVLA: Force-aware MoE for Contact-rich Manipulation* (Yu et al., **NeurIPS 2025**) — force sensor 전제 (EAAS = sensor 부재 조건)
- arXiv **2602.23648** — *FAVLA: Force-Adaptive Fast-Slow VLA* — force sensor 전제
- arXiv **2507.17294** — *VLA-Touch: Dual-Level Tactile Feedback* (NUS) — tactile sensor 전제
- arXiv **2601.01948** — *Learning Diffusion Policy from Primitive Skills* — global instruction용 primitive, contact event tail 아님
- arXiv **2512.11921** — *Towards Accessible Physical AI: LoRA-Based Fine-Tuning of VLA* — parameter efficiency 축 (EAAS = 데이터 표현 축), 병행 인용
- arXiv **2602.22818** — *LeRobot: An Open-Source Library* — 인프라 인용
- Non-peer-reviewed: Correll Lab Medium 2025-12-15 *"When Fine-Tuning Hurts: Failure Modes on a Low-Cost Robot"* — 커뮤니티 동일 문제 인식 증거

### 1.7 알려진 limits / 리뷰어 반박 포인트

LLM-as-Planner 계열을 논문에 넣을 때 리뷰어가 찌를 자리 (미리 방어선 준비 필요):

1. **LLM latency**: Code-as-Policies 공식 한계. 해결: 플래닝은 task 시작에 1회, 실행 loop는 로컬. BrainBody는 brain 호출 빈도를 낮춤으로써 회피.
2. **Embodied grounding 부족**: 2025 Frontiers 서베이의 핵심 비판. LLM은 질량/마찰/force 같은 physical constraint 상식이 부재 → SayCan의 affordance filter, GenCHiP의 action space 설계가 각각 다른 해법.
3. **재현성**: BrainBody-LLM의 GitHub star 2개 → 실제로 돌아가는지는 내가 코드 돌리기 전엔 단정 못함. 언론 보도(TechXplore/InterestingEngineering)는 peer review를 대체하지 않음.
4. **2022년 프레임의 노후화**: SayCan/CaP는 2022년 작업. 2025-26 주류는 VLA end-to-end. "VLA 시대에도 여전히 LLM-as-Planner가 필요한 이유"를 명시하지 않으면 "왜 step back?" 공격을 받음. 유효 반박: (a) failure recovery, (b) long-horizon reasoning, (c) primitive library 설계로 data efficiency 확보 — 이 중 하나를 명확히 택해야 함.
5. **HumanEval 39.8% 같은 수치는 코드 생성 품질 벤치**: 로봇 성공률과 혼동하면 안 됨. 인용 시 맥락 정확히.

### 1.8 Key References (arXiv ID / DOI 요약)

- arXiv 2204.01691 — SayCan (Ahn et al., Google, 2022)
- arXiv 2209.07753 — Code as Policies (Liang et al., Google, 2022)
- arXiv 2209.11302 / Autonomous Robots 10.1007/s10514-023-10135-3 — ProgPrompt (Singh et al., USC+NVIDIA, 2022/2023)
- arXiv 2404.06645 — GenCHiP (Burns et al., Google DeepMind, 2024) ⭐ 유일한 v6 접점
- Wiley adrr.202500072 — BrainBody-LLM (NYU Tandon Khorrami lab, 2025)
- Frontiers frobt.2025.1605405 — Agentic LLM Robots Survey (Raptis et al., 2025)
- Springer s11370-024-00550-5 — LLM+Intelligent Robots Survey (2024)

---

## 2. *(빈 슬롯 — 유저가 다음 세션에 줄 도메인 자리)*

> 새 도메인 추가 시: §1 구조 그대로 복제 → 1.1 개요 → 1.2 Landmark Papers(1차 소스 검증 필수) → 1.3 Surveys → 1.4 Mislabeling (있다면) → 1.5 v6 정합성 → 1.6 잠재적 접점 → 1.7 limits → 1.8 Key References. Index 테이블 갱신.
