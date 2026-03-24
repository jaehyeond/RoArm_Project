# Physical AI Competitive Landscape: Gemini Robotics and Opportunities for a Master's Student

**C3 (Paper Writing & Positioning) | Date: 2026-03-24**
**Purpose: Strategic intelligence for CoRL 2026 positioning + student career guidance**

---

## Important Caveat (CLAUDE.md 연구 검증 규칙 적용)

This document synthesizes publicly available information as of my knowledge cutoff (August 2025) plus project files documenting searches through March 2026. Gemini Robotics was announced in March 2025 (arXiv:2503.xxxxx). Some specifics below are marked UNVERIFIED where I cannot confirm from primary sources. Overclaim warnings are marked explicitly.

---

## 1. Competitive Landscape: Side-by-Side Comparison

### 1.1 Google DeepMind: Gemini Robotics

| Attribute | Details |
|-----------|---------|
| **Announced** | March 2025 (Google AI blog + DeepMind blog) |
| **Models** | Gemini Robotics (general) + Gemini Robotics-ER (Embodied Reasoning) + Gemini Robotics Edge (on-device) |
| **Backbone** | Gemini 2.0 Flash (multimodal VLM) |
| **Architecture** | Gemini VLM → action decoder (similar to RT-2 paradigm but at Gemini scale) |
| **Model size** | Not disclosed publicly |
| **Key capability** | Natural language instruction following, dexterous manipulation, long-horizon planning |
| **Robot platforms** | Demonstrated on: ALOHA 2 (bimanual), Apptronik Apollo (humanoid), Boston Dynamics Spot, others |
| **Open-source?** | **No. Completely closed.** |
| **API access?** | No public API as of 2026-03. "Trusted testing partners" only |
| **VRAM required** | Unknown (cloud inference implied; Edge version targets on-device but specs unannounced) |
| **Key papers** | "Gemini Robotics: Bringing AI into the Physical World" (DeepMind blog, March 2025); arXiv preprint reportedly available |
| **Real demos?** | Yes — YouTube demos showing folding laundry, pouring, stacking, following spoken instructions |
| **Deployment status** | Research / trusted tester stage. NOT commercially available. |

**What "Gemini teaches robots" actually means technically:**
Gemini Robotics uses Gemini 2.0 Flash as the vision-language backbone (already pretrained on web-scale multimodal data). This backbone is connected to an action prediction head that outputs continuous control signals. The key insight is that Gemini's rich language and vision understanding transfers to robot reasoning: the robot can understand "move the red cup next to the bowl but don't knock over the spoon" as a compositional instruction, rather than needing discrete commands. This is fundamentally the same paradigm as RT-2 (Brohan et al., 2023) but with a much more capable VLM backbone.

The "Embodied Reasoning" (ER) variant specifically extends the model to answer spatial/physical questions: "can this container hold liquid?" or "which object should I pick up first to clear the table?" This is a major differentiator from action-only VLAs.

**Gemini Robotics Edge:** This is the on-device variant intended to run without cloud connectivity. Size and architecture not publicly disclosed. The goal is low-latency control (similar to SmolVLA's design philosophy). This is the variant most relevant to students.

---

### 1.2 NVIDIA: GR00T N1 / N1.5 / N1.6 / N1.7

| Attribute | Details |
|-----------|---------|
| **Architecture** | Dual-system: System 1 (fast, reactive, ~2B params) + System 2 (slow, reasoning, larger VLM) |
| **Current versions** | N1.5 (CoRL 2025), N1.6 (GTC March 2026), N1.7 (NVIDIA press release, 2026) |
| **Open-source?** | **Yes — open weights on HuggingFace** |
| **VRAM required** | ~16 GB for inference (fits RTX 4090); training requires more |
| **Robot targets** | Humanoids primarily: Figure, Agility Robotics, Boston Dynamics Atlas, Fourier, Unitree |
| **Simulation** | Isaac Lab (GPU-parallelized), Cosmos world models for data generation |
| **Key ecosystem** | Cosmos (world model for synthetic data), Isaac (sim), Jetson Thor (edge compute) |
| **Real demos?** | Yes — GTC demos, partner integrations |
| **API access?** | HuggingFace download, NGC for enterprise |

**What's accessible to a student:**
- GR00T N1.5 weights downloadable (Apache/non-commercial license, check specifics)
- Isaac Lab free for academic use
- But: designed for humanoids. Adapting to 6-DOF arm requires significant work.
- Cosmos world models are large (7B/14B) and require significant compute for training

---

### 1.3 Tesla: Optimus

| Attribute | Details |
|-----------|---------|
| **Current version** | Optimus Gen 2 (late 2024), Gen 3 implied |
| **AI stack** | Uses Tesla's existing FSD neural net architecture (HydraNet-style), adapted for humanoid |
| **Open-source?** | **No. Completely closed.** |
| **Papers published?** | None. Zero academic publications. All information from Tesla AI Day demos and X posts. |
| **Real demos?** | Demos at Tesla events, reportedly working in Fremont factory |
| **Deployment status** | UNVERIFIED. Tesla claims 1000+ Optimus units in internal deployment. No independent verification. |
| **Relevance to student** | Near zero for research purposes. No papers, no code, no collaboration path. |

**Warning for positioning:** CLAUDE.md rules apply here. The "verified" claims about Optimus are minimal. Project files note "검증 안됨 (텔레오퍼레이션 이벤트, 논문 없음)". Do not cite Optimus capabilities as established facts in a paper.

---

### 1.4 Figure AI: Figure 02 / Helix VLA

| Attribute | Details |
|-----------|---------|
| **Hardware** | Figure 01, Figure 02, Figure 03 (announced 2026) |
| **AI system** | Helix VLA — proprietary VLA for Figure humanoids |
| **Partnership** | OpenAI partnership for "Figure-GPT" integration (2024), later ended |
| **Open-source?** | **No.** |
| **Papers?** | Limited. "Helix" described in blog posts, not peer-reviewed papers |
| **Deployment** | BMW factory partnership (scope disputed), Figure claims commercial deployment |
| **Valuation** | $39B (reported 2025) — widely questioned by analysts |
| **Relevance to student** | Low. Closed system, no papers, questionable deployment claims. |

**Skeptic note:** Project files state "BMW 파트너십 범위 의문 제기됨" (BMW partnership scope questioned). No independent verification. Not a reliable citation target.

---

### 1.5 Physical Intelligence: pi0 / pi0.5

| Attribute | Details |
|-----------|---------|
| **pi0** | CoRL 2024 paper (Black et al.). 3B params. PaliGemma backbone + flow matching action head. |
| **pi0.5** | arXiv 2504.16054. Added web data fine-tuning, semantic generalization, mobile base. |
| **pi0-FAST** | arXiv 2501.09747. FAST tokenizer for faster inference (10x speedup vs pi0). |
| **Open-source?** | **Yes — pi0 weights on HuggingFace, training code open.** |
| **VRAM required** | 3B params → ~24-48 GB for training. Inference can be squeezed to ~12 GB with tricks. NOT ideal for RTX 4090 16GB. |
| **Real deployment?** | Yes. Confirmed commercial customers (Weave Robotics, others). Only company with verified commercial VLA deployment. |
| **Key differentiator** | Flow matching action head (vs discrete token prediction). Much better for dexterous, continuous manipulation. |
| **Relevance to student** | HIGH for related work. LOW for direct use (VRAM constraints). |

---

### 1.6 HuggingFace: SmolVLA / LeRobot

| Attribute | Details |
|-----------|---------|
| **SmolVLA** | 450M params. SmolVLM-500M backbone + flow matching action expert. |
| **Pretrained on** | community_dataset_v1: 128 datasets, 11,132 episodes, ALL SO-100 (in-distribution) |
| **Open-source?** | **Yes. Fully open-source, Apache 2.0.** |
| **VRAM required** | 2-10 GB depending on batch size. PERFECT for RTX 4090. |
| **Training** | `lerobot-train` CLI, batch_size=64, 50K steps for in-distribution tasks |
| **Real deployment?** | YES — our project: 100% success rate (sponge pick) with 74 episodes, 50K steps |
| **Key differentiator** | Smallest open VLA, consumer-runnable, fully reproducible, community ecosystem |

---

### 1.7 Meta: Robotics?

Meta does not have a publicly announced robotics VLA model as of my knowledge cutoff. Their robotics research is primarily:
- **FAIR robotics** group: dexterous manipulation, touch sensing (ReSkin/GelSight research), locomotion
- **Habitat / AI2-Thor**: simulated environments for embodied navigation
- **No VLA equivalent to pi0/SmolVLA/GR00T**

Meta's closest relevant work: **GROOT** (grounded object-centric representations, different from NVIDIA GR00T) and work on tactile sensing integration.

**WARNING (CLAUDE.md rule):** I cannot confidently state "Meta has no VLA" without 10+ search verification. Project files do not mention a Meta VLA. Use hedged language if citing this.

---

### 1.8 OpenAI: Robotics?

OpenAI's robotics position is complicated:
- Disbanded robotics team around 2021
- Partnership with Figure AI for "Figure-GPT" (ended 2025 reportedly)
- No standalone robotics model as of knowledge cutoff
- 1X Technologies (humanoid company) has received OpenAI investment
- "ChatGPT for robots" style integration (using GPT-4V for task planning) is documented in many academic papers, but not an OpenAI product

**WARNING:** Same verification caveat as Meta. State "no publicly available robotics model" with hedging.

---

## 2. What Can a Student Actually Use?

### 2.1 Open-Source + Consumer GPU Accessible

| Model | VRAM Needed | Can Use? | How? |
|-------|------------|---------|------|
| **SmolVLA (450M)** | 2-16 GB | YES, ideal | `lerobot-train`, HuggingFace |
| **OpenVLA (7B)** | 14-16 GB (inference), 40+ GB (training) | Inference only on 4090; training needs LoRA/quantization | `lerobot` with OpenVLA-OFT |
| **OpenVLA-OFT** | ~14 GB with LoRA + 4-bit | YES with LoRA | OpenVLA-OFT repo |
| **pi0 (3B)** | 12 GB inference, 24-48 GB training | Inference only; training needs gradient checkpointing + 4-bit | Physical Intelligence repo |
| **GR00T N1.5** | ~16 GB inference | Inference only; designed for humanoids | HuggingFace / NGC |
| **Gemini Robotics** | Unknown / cloud | NO (closed, no API) | N/A |
| **Helix (Figure)** | N/A | NO (proprietary) | N/A |
| **Optimus** | N/A | NO (closed) | N/A |

**Practical answer for student:**
1. SmolVLA is the PRIMARY tool — already working at 100% success
2. OpenVLA-OFT is feasible for comparison experiments (fits 4090 with LoRA + 4-bit)
3. pi0 inference-only for qualitative comparison
4. GR00T N1.5 is accessible but not relevant to 6-DOF arm without significant adaptation

---

## 3. Research Opportunities from Gemini Robotics

### 3.1 What Gemini Actually Claims

Based on the DeepMind blog and available descriptions (VERIFY with arXiv preprint before citing):

**Claimed capabilities:**
1. Dexterity: complex bimanual tasks (folding clothes, using tools)
2. Generalizability: follows novel natural language instructions without task-specific fine-tuning
3. Embodiment-agnostic: can control different robot morphologies
4. Embodied reasoning: answers spatial questions, reasons about physics

**What is NOT claimed (important for gap analysis):**
- Performance on low-cost hardware ($100-200 consumer robots)
- Data efficiency (how many demos needed for a new task)
- Reproducibility (no open code, no public benchmarks)
- Comparison with open-source baselines on standardized tasks

### 3.2 Research Gaps Gemini Creates

**Gap 1: The Accessibility Gap (HIGH CONFIDENCE)**
Gemini Robotics is closed and cloud-based. For the millions of researchers and practitioners who cannot access it, the question becomes: "What is the best open, consumer-deployable alternative, and how does it compare?" This creates a genuine need for systematic benchmarking of accessible VLAs.

Opportunity: "How close can open, consumer-hardware VLAs (SmolVLA, OpenVLA) get to closed frontier systems on standardized tasks?"

**Gap 2: The Data Efficiency Gap (HIGH CONFIDENCE, our current direction)**
Gemini Robotics relies on massive pretraining data. It does not publish data efficiency curves ("how many demos for a new task on new hardware"). Open VLAs on OOD hardware face a real data efficiency problem. There is no systematic study of how many demos you need for a $130 OOD robot.

This is directly our Contribution 1 (OOD scaling laws).

**Gap 3: The Embodiment Diversity Gap (MEDIUM CONFIDENCE)**
Gemini demos use high-end hardware (ALOHA 2, Apollo humanoid). The question "does this work on a $130 robot arm with a single consumer camera?" is answered by nobody at the frontier. Our work occupies this niche.

**Gap 4: The Reproducibility Gap (HIGH CONFIDENCE)**
Gemini cannot be reproduced by researchers. This creates structural demand for reproducible baselines. Papers that establish and characterize reproducible VLA baselines on standardized tasks have inherent long-term value.

### 3.3 Can a Student Build ON TOP of Gemini?

Currently: **No.** There is no fine-tuning API, no weights, no SDK.

If Gemini provides an API in the future (possible but unannounced):
- Could use Gemini as a "brain" for high-level planning, SmolVLA for low-level execution (hierarchical approach)
- Could compare Gemini API prompting vs SmolVLA fine-tuning on same tasks
- This is speculative; do not plan CoRL paper around it

**Practical path instead:** Use the contrast with Gemini rhetorically. Frame your paper as: "While closed frontier systems like Gemini Robotics demonstrate impressive capabilities, access is limited. We characterize what is achievable with fully open tools on consumer hardware."

### 3.4 VLA Head-to-Head Comparison Papers

**Do they exist?** There is a small but growing body of comparative work:

| Paper | Comparison |
|-------|-----------|
| "RoboArena" (RSS 2025 workshop area) | Standardized robot benchmark for VLA comparison |
| OpenVLA (CoRL 2024) | Compared vs RT-2, Octo, Diffusion Policy |
| SmolVLA paper (HuggingFace, 2025) | Compared vs OpenVLA, Octo on SIMPLER benchmarks |
| Towards Accessible Physical AI (arXiv:2512.11921) | Compared quantized 3.1B VLA vs baselines |

**Key observation:** Head-to-head comparisons on REAL hardware (not SIMPLER/LIBERO simulation) are rare. Papers using simulated benchmarks dominate. Real-robot comparative data on low-cost hardware is scarce.

**Opportunity:** Even a partial real-robot comparison (SmolVLA vs OpenVLA-OFT on same task with same data) would be publishable because the community lacks it.

---

## 4. Positioning for the Student

### 4.1 Does Gemini Make SmolVLA Work Obsolete?

**No, and here is the argument structure for the paper:**

The "obsolescence" question rests on a false premise: that Gemini is a substitute. It is not, for three reasons:

**Reason 1: Access.** Gemini Robotics is not accessible. A paper that demonstrates what researchers CAN USE (SmolVLA) has immediate practical value. Gemini has zero practical value for 99.9% of researchers right now.

**Reason 2: Research question shift.** Gemini's existence reframes the scientific question from "can VLAs work on robots?" (answered yes) to "what are the minimal requirements for effective VLA deployment?" Our work directly addresses this second question, which Gemini does not.

**Reason 3: Methodological contribution.** Our data quality tools (FK-depth, gripper phase, static frame detection) and scaling law methodology are hardware-agnostic. They are useful regardless of which VLA backbone is used. A paper that says "here is how to measure and improve training data quality for ANY VLA" has value independent of whether Gemini exists.

### 4.2 "David vs Goliath" or "Complementary Research"?

**Recommended framing: "Enabling the Long Tail"**

Neither pure contrast nor pure complement. The narrative:

"Frontier closed systems (Gemini Robotics, pi0.5) demonstrate what is achievable with massive compute and data. But the majority of real-world deployments — research labs, small companies, individual practitioners — cannot access frontier infrastructure. We study what is achievable at the other end of the spectrum: a 450M parameter open model, consumer GPU, and a $130 robot. Our systematic characterization of data efficiency and quality provides a principled foundation for this long tail of deployments, which is where most of the world's robots will actually be trained."

This framing:
- Does not trash Gemini (avoids alienating reviewers who may be from large labs)
- Establishes clear complementarity
- Positions the work as filling a demonstrated gap (accessibility, data efficiency)
- Is honest about our constraints

### 4.3 Revised One-Sentence Pitch (Updated from MARCH_2026_LANDSCAPE_UPDATE.md)

Previous: "We characterize the data-efficiency frontier for adapting a 450M-parameter VLA to an out-of-distribution $130 robot arm on a single consumer GPU."

Updated with Gemini context:
"As frontier VLA systems like Gemini Robotics and pi0.5 demonstrate impressive capabilities at scale, we study the opposite extreme: what data quality and quantity are needed to adapt a 450M open-source VLA to an OOD $130 robot on a single consumer GPU, providing systematic scaling laws and reusable quality metrics for the long tail of practitioners."

### 4.4 Specific Opportunities for a Metaverse/XR-Background Student

The student's Unity/XR expertise creates a unique angle that no pure robotics lab can replicate:

**Angle A: XR-augmented data collection**
Using XR tools (which the student knows deeply) to collect better robot demonstration data. This could be a Section in the paper or a separate submission. No major VLA paper has focused on XR-based data collection quality improvement.

**Angle B: Visualization and Analysis**
Using Unity to build real-time visualization of VLA attention maps, action distributions, or data quality metrics. Not a core paper contribution, but distinguishes demos and could be a workshop paper.

**Angle C: Digital Twin for Data Augmentation**
Using Unity's physics engine to augment real robot data with rendered variations (object texture, lighting, position). This would directly address the data efficiency problem from a different angle than more data collection.

NOTE: These are suggestions for AFTER CoRL 2026 submission or for a thesis chapter. Do NOT add these to the current paper — no experimental data exists yet.

---

## 5. Summary Table: Physical AI Stack Comparison

| System | Size | Open? | RTX 4090? | Real Robot? | Papers? | Student Relevance |
|--------|------|-------|-----------|-------------|---------|-------------------|
| **SmolVLA** | 450M | YES | YES (ideal) | YES (our project) | Yes (HF 2025) | PRIMARY |
| **OpenVLA** | 7B | YES | Inference only | Yes (CoRL 2024) | Yes | COMPARISON |
| **pi0** | 3B | YES | Inference only | Yes (CoRL 2024) | Yes | RELATED WORK |
| **pi0.5** | 3B | YES | Inference only | Yes | Yes | RELATED WORK |
| **GR00T N1.5** | ~2B | Weights only | Inference only | Yes (humanoid) | Partial | RELATED WORK |
| **Gemini Robotics** | ? | NO | NO | Yes (demos) | Blog only | CONTRAST |
| **Figure Helix** | ? | NO | NO | Claimed | No | MENTION ONLY |
| **Optimus** | ? | NO | NO | Unverified | No | SKIP |
| **Octo** | 93M | YES | YES | Yes (RSS 2024) | Yes | RELATED WORK |

---

## 6. Paper-Specific Recommendations

### What to cite about Gemini Robotics

In the Related Work section, cite Gemini Robotics as evidence of the frontier, not as a baseline:

"Recent closed frontier systems (Gemini Robotics [DeepMind 2025], pi0.5 [Black et al. 2025]) demonstrate impressive manipulation capabilities but remain inaccessible for independent research. Our work addresses the complementary question of what is achievable within the open, consumer-hardware setting."

### What NOT to claim regarding Gemini

- Do NOT claim SmolVLA is better than Gemini (no comparison data, would be overclaim)
- Do NOT claim Gemini "cannot" do what we do (we don't know its limitations)
- Do NOT use "despite Gemini's existence" framing (patronizing to reviewers from Google)

### Overclaim flags for this analysis

| Claim | Status | Action needed |
|-------|--------|---------------|
| "Meta has no VLA" | UNCERTAIN | Add hedge: "to our knowledge" |
| "OpenAI has no robotics model" | LIKELY TRUE but verify | Add hedge |
| "Gemini requires cloud inference" | LIKELY TRUE for ER/main; Edge variant unconfirmed | Don't state as fact |
| "Figure BMW deployment scope" | DISPUTED | Don't cite as deployment success |
| "pi0.5 has 3B params" | LIKELY from pi0 architecture, but pi0.5 paper should be verified | Check arXiv:2504.16054 |

---

## 7. Updated Memory Implications

This analysis suggests updating the competitive landscape memory:
1. Gemini Robotics is a contrast point, not a competitor (closed, inaccessible)
2. GR00T N1.5 is a related work (open weights, humanoid-focused, not directly comparable)
3. The "enabling the long tail" framing is stronger than "David vs Goliath"
4. Student's XR background creates post-CoRL research opportunities
