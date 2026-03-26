# VLA Manipulation Baselines: State-of-the-Art (March 2026)

**Date:** 2026-03-25
**Scope:** Grasping, language-conditioned manipulation, sequential tasks, low-cost arms, data efficiency, dual-arm, voice-to-action

---

## 1. Single-Arm Grasping Baselines with VLA

### SmolVLA (450M, HuggingFace)
- **Architecture:** SmolVLM-2 + Flow Matching Action Expert, 10 denoise steps
- **Pretraining data:** 481 community datasets, 22.9K episodes, 10.6M frames
- **Fine-tuning for grasping:**
  - Community report (SO-101, Reddit): ~90+% success rate with cube within training boundary (+/-10cm), LoRA fine-tune, 15K steps, batch_size 8, trained on 3090 in 10 hours. Completely fails outside boundary.
  - Medium report (Henry Hu): Successfully fine-tuned with only 25 demonstrations for pick-and-place (no explicit language instructions in training data).
  - Medium report (Xavier O'Keefe, Correll Lab): 40% grasp success in sim with systematic validation, visual domain gap was the main issue.
  - Medium report (Nikhil Sawane, Correll Lab): ~60 success-only demonstrations, arm kept moving straight down -- failure mode of visuomotor imitation learning on low-cost robot.
- **Demo requirements:** 25-50 demos can produce basic behavior; 50+ recommended for reliable single-position grasping. Multi-position generalization requires position diversity in training data.
- **GPU:** Single A100 ~4hrs for 20K steps. RTX 4090 feasible (batch_size=64 fits in ~10GB VRAM).
- **Key limitation:** Very narrow spatial generalization -- works within training distribution boundary only.

### OpenVLA (7B, Stanford/Berkeley)
- **Architecture:** Prismatic VLM (DINOv2 + SigLIP + Llama 2 7B), discrete action tokens
- **Pretraining data:** 970K real-world trajectories from Open X-Embodiment
- **Success rates:**
  - Outperforms RT-2-X (55B) by 16.5% absolute across 29 tasks
  - LIBERO average: 76.5% (vanilla fine-tuning)
  - Requires ~100 demonstrations for fine-tuning to new domains (per GitHub docs)
- **Key limitation:** Slow inference (autoregressive), single-step actions (no chunking)

### OpenVLA-OFT (7B, Stanford) -- RSS 2025
- **Key improvements:** Parallel decoding, action chunking, continuous actions, L1 loss
- **LIBERO results (modified dataset):**
  - OpenVLA-OFT (Cont-L1): 95.3% average
  - pi0 (fine-tuned): 94.2%
  - OpenVLA baseline: 76.5%
  - Diffusion Policy (scratch): 72.4%
- **LIBERO with additional inputs:** 97.1% (SOTA)
- **Inference speed:** 25-50x faster than vanilla OpenVLA, 109.7 Hz with 8-step chunks
- **ALOHA real-world:** Outperforms pi0, RDT-1B, ACT, Diffusion Policy
  - Fold shorts: 20 demos
  - Fold shirt: 30 demos
  - Scoop X into bowl: 45 demos
  - Put X into pot: 300 demos
- **Language following:** Drops to 33% (random) when FiLM conditioning is removed

### pi0 (~3B, Physical Intelligence) -- RSS 2025
- **Architecture:** VLM + Flow Matching, 50-step action chunks
- **Pretraining:** >10,000 hours of robot data, 7 robot configs, 68 tasks + OXE
- **Zero-shot evaluation (their tasks):** Shirt folding ~95%, bussing easy ~90%, grocery bagging ~85%
- **In-the-wild evaluation (Penn PaL Lab, 300+ trials):**
  - Overall average progress: 42.3%
  - Pick-and-place success: 24%
  - Articulated objects: 28.5%
  - Fabric manipulation: 19.4%
  - Small objects (pineapple, markers): 90% progress
  - Prompt sensitivity: "Close the toilet" = 0% vs "Close the white lid of the toilet" = 100%
- **Fine-tuning data requirements:** 1-5 hours of demos for easy tasks, 5+ hours for medium tasks
- **Key insight:** Zero-shot on familiar task types is strong, but zero-shot on novel environments/objects drops dramatically.
- **Access:** CLOSED model (pi0-FAST DROID partially open)

### GraspVLA (1.8B, PKU) -- CoRL 2025
- **Architecture:** InternLM2 (1.8B) + DINOv2 + SigLIP + flow-matching action expert
- **Pretraining:** SynGrasp-1B (1 billion frames, 10M trajectories, 10,680 objects, 240 categories, entirely synthetic, ~$5K cost)
- **Real-world zero-shot grasping:**
  - Basic: 93.3% (vs pi0 w/o pretrain 80.0%, OpenVLA 20.0%)
  - With lighting changes: 96.6%
  - With distractors: 93.3%
  - Height variation: 90.0%
- **LIBERO zero-shot:**
  - Long: 82.0% (vs pi0 fine-tuned 62.7%, OpenVLA fine-tuned 33.7%)
  - Goal: 91.2% (vs pi0 79.4%, OpenVLA 56.6%)
  - Object: 94.1% (vs pi0 93.8%, OpenVLA 65.4%)
- **Post-training data:** 100 demos for new industrial tasks --> 80-90% success; 10 demos per object for sequential grasping --> 90% success
- **Key innovation:** Proves synthetic-only pretraining can outperform real-data pretrained models for grasping.

### RT-2 (12B/55B, Google DeepMind)
- **Language Table sim:** 90% success rate (vs RT-1 74%, BC-Z 72%)
- **Emergent capabilities:** Novel object generalization, rudimentary reasoning, 3x improvement over RT-1 on emergent tasks
- **Real eval:** 6K+ trials on Google Mobile Manipulator fleet
- **Access:** CLOSED, hardware platform discontinued

### Octo (93M, Berkeley) -- RSS 2024
- **Architecture:** Transformer + Diffusion head (NOT VLM-based)
- **Pretraining:** 800K trajectories from Open X-Embodiment
- **Zero-shot:** 72% average across 3 robot embodiments, 29% higher than RT-1-X
- **Goal-image vs language:** Goal-image conditioning 25% higher success than language conditioning
- **Fine-tuning:** ~100 in-domain demonstrations, <5 hours on NVIDIA A5000
  - Fine-tuned results: 50-100% per domain (avg 72%), outperforms next best baseline by 52%
- **Key limitation:** JAX-based, weaker language understanding than VLM-based models

### GR00T N1.5/N1.6 (3B, NVIDIA) -- Open
- **Fine-tuning:** 20-40 demonstrations sufficient for post-training
- **Post-training:** 10K-30K steps, batch size <=1K
- **LIBERO:** Outperforms baselines (exact numbers behind figures)
- **LeRobot-compatible** data format
- **Access:** Fully open weights/code

---

## 2. Language-Conditioned Grasping

### How language instruction is integrated:

| Model | Language Integration | Mechanism |
|-------|---------------------|-----------|
| **SmolVLA** | Text prompt via VLM | SmolVLM-2 processes text + image jointly, feeds to action expert |
| **OpenVLA/OFT** | Text tokens in VLM | Prismatic VLM encodes instruction, FiLM conditioning for action head |
| **pi0** | VLM dual decoding | Same model outputs text (reasoning) AND motor commands |
| **Octo** | Language OR goal image | Transformer cross-attention on language tokens |
| **RT-2** | Text tokens as actions | Actions encoded as text tokens in VLM vocabulary |
| **GraspVLA** | Progressive Action Generation | VLM predicts 2D bounding box first, then action expert generates grasp |
| **GR00T N1** | Language via VLM | System 2 (VLM) reasons, System 1 (DiT) executes |

### Language following quality:
- **OpenVLA-OFT+ on ALOHA:** Best language grounding among tested methods; 33% (random) when FiLM removed
- **Octo:** Language conditioning significantly weaker than goal-image conditioning (25% gap)
- **pi0:** Strong language following but extremely sensitive to prompt phrasing (0% vs 100% on same task with different wording)
- **RT-2:** Best emergent language understanding (picks "extinct animal" = dinosaur figurine), but closed
- **SmolVLA:** Language instruction given at training time; community data used Qwen2.5-VL-3B for auto-annotation when labels were poor

### Color-conditioned grasping ("pick up the red sponge"):
- All VLM-based VLAs (SmolVLA, OpenVLA, pi0, RT-2, GR00T) support this natively through their VLM backbone
- Octo supports it through language conditioning but with lower accuracy
- **Data requirement for color generalization:** Not well-studied in isolation. The VLM backbone provides semantic understanding of "red," but the action head must be trained on diverse enough data to actually reach/grasp at variable locations. Community reports suggest training data must include the specific object variants to get reliable behavior.
- **GraspVLA:** Progressive Action Generation first predicts a 2D bounding box of the target object, then generates the grasp -- this two-stage approach is naturally suited for language-conditioned grasping.

---

## 3. Sequential/Multi-Step Tasks with VLA

### Current approaches:

**A. Single VLA for long-horizon (end-to-end):**
- **Long-VLA** (CoRL 2025, arXiv 2508.19958): First end-to-end VLA for long-horizon. Uses "phase-aware input masking" to segment subtasks into moving/interaction phases. Proposes L-CALVIN benchmark. "Significantly outperforms prior SOTA" but exact numbers behind figures.
- **LIBERO-Long benchmark:** GraspVLA zero-shot 82.0% vs pi0 fine-tuned 62.7% vs OpenVLA fine-tuned 33.7%
- **Self-Improving VLA (PLD):** Achieves near-saturated 99% on LIBERO (including multi-step suites) through autonomous self-improvement without additional human demos.

**B. Hierarchical/Planning-based:**
- **RT-H** (Google, 2024): Predicts "language motions" as intermediate representation, then actions. Enables language intervention during execution.
- **SayCan** (Google, 2022): LLM plans high-level, RT-1/RT-2 executes low-level. Classic but pre-VLA.
- **Agentic Robot:** Uses Standardized Action Procedures (SAP) with explicit planning and verification. Outperforms OpenVLA by 7.4% on LIBERO-Long.

**C. VLABench** (ICCV 2025): Large-scale benchmark specifically for language-conditioned long-horizon reasoning tasks.

### Task chaining mechanisms:
- Most current VLAs handle multi-step implicitly through the language instruction and long action chunks
- pi0.5 handles multi-step cleaning tasks in real homes by combining text reasoning (high-level plan) with motor commands from the same model
- Long-VLA explicitly segments phases; most others rely on the VLM backbone to maintain task context across steps
- **No widely adopted automatic task segmentation** -- current practice is either a single long instruction or hierarchical planning with a separate LLM

---

## 4. Low-Cost Robot Arms + VLA

### Documented combinations:

| Robot | VLA Model | Demos | Result | Source |
|-------|-----------|-------|--------|--------|
| **SO-101** | SmolVLA (LoRA) | not specified | ~90+% on cube within +/-10cm boundary | Reddit r/robotics |
| **SO-100/SO-101** | SmolVLA | 25+ | Basic pick-and-place working | HuggingFace/LeRobot official |
| **Low-cost arm (Correll Lab)** | SmolVLA | ~60 | Failure: arm moves straight down | Medium (Nikhil Sawane) |
| **Low-cost arm (sim)** | SmolVLA | varied | 40% grasp success in sim | Medium (Xavier O'Keefe) |
| **SO-100** | GR00T N1.5 | 20-40 | Fine-tuning documented | Hackaday.io project |
| **Koch v1.1** | LeRobot-compatible | varies | Community projects active | HuggingFace docs |
| **ALOHA** | OpenVLA-OFT+ | 20-300 | Best among VLAs tested | RSS 2025 |
| **ALOHA** | pi0, RDT-1B, ACT | 20-300 | All evaluated | RSS 2025 |
| **RoArm M3 Pro** | SmolVLA | 50 | Failed: gripper not opening, drift | This project (2026-02-11) |

### Key findings for low-cost arms:
- SmolVLA is the most commonly used VLA on low-cost hardware due to 450M parameter size
- SO-100/SO-101 are the primary community platforms (LeRobot ecosystem)
- Success is highly dependent on data quality and camera setup stability
- **Spatial generalization is the main bottleneck** -- works within training distribution only
- GR00T N1.5 is starting to appear on SO-100 but requires more VRAM
- No published papers specifically studying VLA on RoArm, Koch, or similar ultra-low-cost arms in a rigorous way -- most evidence is community blog posts and Reddit

---

## 5. Data Efficiency

### Demonstrations needed by task complexity:

| Task Type | Demos Needed | Success Rate | Source |
|-----------|-------------|--------------|--------|
| Single-position grasping (fixed object) | 25-50 | 70-90% | SmolVLA community reports |
| Multi-position grasping (+/-10cm) | 50-100 | ~90% within boundary | Reddit SO-101 report |
| LIBERO tasks (10 tasks per suite) | 500 total (50/task) | 72-97% | OpenVLA-OFT LIBERO |
| Real ALOHA simple (fold shorts) | 20 | Reliable | OpenVLA-OFT |
| Real ALOHA medium (scoop into bowl) | 45 | Reliable | OpenVLA-OFT |
| Real ALOHA complex (put X into pot) | 300 | Reliable | OpenVLA-OFT |
| Octo fine-tuning per domain | ~100 | 72% avg | Octo paper |
| GR00T N1.5 post-training | 20-40 | Competitive | NVIDIA docs |
| GraspVLA post-training (new objects) | 10-100 | 80-90% | GraspVLA CoRL 2025 |
| pi0 easy tasks (from pretrain) | 1-5 hours of demos | Good | pi0 paper |
| pi0 medium tasks | 5+ hours | Variable | pi0 paper |
| RDT-1B zero-shot new skills | 1-5 demos | Claimed | RDT-1B paper |
| MoS-VLA (one-shot adaptation) | 1 expert trajectory | 70-100% | arXiv 2510.16617 |

### Color/attribute generalization:
- Not well-isolated in existing studies
- VLM backbone provides semantic grounding ("red," "blue"), but action head generalization depends on positional diversity in training data
- No published study specifically reports "N demos needed for color-conditioned generalization"

### Multi-position generalization:
- SmolVLA on SO-101: works within +/-10cm of training positions, completely fails outside
- This is a common VLA limitation -- the action head overfits to spatial distribution of demonstrations
- GraspVLA's synthetic pretraining may help (extensive spatial randomization), but real-world evaluation limited

---

## 6. Dual-Arm VLA

### Key systems:

**TwinVLA** (ICLR 2026, arXiv 2511.05275)
- Composes two copies of pretrained single-arm VLA into coordinated bimanual system
- 50 demos per task, evaluated on Anubis robot (6-DOF per arm)
- Results vs baselines (real-world fold towel):
  - Low light: TwinVLA 45%, RDT-1B 15%, pi0 40%
  - With distractors: TwinVLA 25%, RDT-1B 15%, pi0 60%
- Training from scratch penalty: -46% (real world)
- Key advantage: Data-efficient -- leverages single-arm pretraining for bimanual

**Bi-VLA** (IEEE SMC 2024, arXiv 2405.06039)
- Integrates vision + language + action for bimanual dexterous manipulation
- Tested on household tasks (salad preparation)
- Uses language comprehension to translate human instructions to executable code
- Earlier work, less quantitative evaluation

**OpenVLA-OFT+ on ALOHA** (RSS 2025)
- Bimanual ALOHA robot evaluation
- Best performance among all tested VLAs for bimanual tasks
- 20-300 demos per task
- Language following maintained

**RDT-1B** (arXiv 2410.07864)
- 1.2B parameter diffusion transformer
- Specifically designed for bimanual manipulation
- Zero-shot generalization with 1-5 demos claimed
- Fine-tuned on 6K+ bimanual episodes

**ODIL** (One-Shot Dual-Arm Imitation Learning, arXiv 2503.06831)
- Learns from single demonstration for bimanual tasks
- Not VLA-based but relevant comparison

### State of dual-arm VLA:
- Still significantly behind single-arm performance
- Most bimanual datasets are small (public data is overwhelmingly single-arm)
- TwinVLA's modular approach (composing two single-arm VLAs) is the most data-efficient path
- pi0/pi0.5 demonstrated real-world bimanual (laundry, etc.) but closed model

---

## 7. Voice-to-Action

### VLAS (ICLR 2025, arXiv 2502.13508)
- **First end-to-end VLA with native speech input**
- Architecture: LLaVA + Vicuna-7B + CLIP (vision) + Whisper (speech)
- CALVIN benchmark with speech instructions:
  - Text: 94.5% (1-step), 56.6% (5-step)
  - Synthesized speech: 94.2% (1-step), 54.6% (5-step)
  - Real speech: 93.6% (1-step), 51.3% (5-step)
- Customization with voice RAG: 86% avg (vs <20% for text-only VLA)
- Training data: SQA (185K samples, 1,152 voices) + CSI (194K audio samples)
- Robots: Franka Panda (sim) + UR5 (real)
- Key innovation: Preserves voiceprint information (speaker identity) for personalized tasks

### SVA (Speech-VLA, ScienceDirect 2025)
- Accepts spoken commands + visual observations through unified embedding space
- Published in Pattern Recognition
- Less detailed evaluation than VLAS

### Alternative: Pipeline approach (ASR + VLA)
- Self-Supervised Voice Denoising (PMC, 2025): SpeechRecognition library converts audio to text, then feeds to VLA
- Most practical deployments use ASR (Whisper/etc.) -> text -> VLA as a pipeline
- Pipeline approach loses non-semantic speech information (emotion, speaker identity, urgency)
- VLAS argues end-to-end is better because it avoids ASR error propagation

### State of voice-to-action:
- VLAS is the primary end-to-end solution (ICLR 2025)
- Speech instructions achieve within ~2-5% of text instruction performance
- Real speech (vs synthesized) adds another ~1-3% degradation
- Voiceprint-based personalization ("grab MY cup") is a unique capability of end-to-end approach
- Most real-world systems still use ASR -> text -> VLA pipeline

---

## Summary Table: VLA Comparison for Basic Manipulation

| Model | Params | Open? | Grasping Success | Demos Needed | Language | Multi-step | Low-cost HW |
|-------|--------|-------|------------------|--------------|----------|------------|-------------|
| SmolVLA | 450M | Yes | 40-90% (varies) | 25-100 | Yes (VLM) | Limited | Excellent |
| OpenVLA | 7B | Yes | ~76% (LIBERO) | ~100 | Yes (VLM) | Limited | Moderate |
| OpenVLA-OFT | 7B | Yes | 95-97% (LIBERO) | 20-300 | Yes (FiLM) | LIBERO-Long OK | Moderate |
| pi0 | ~3B | No | 24-95% (task-dep) | Hours of data | Yes (VLM) | Yes (pi0.5) | Impossible |
| GraspVLA | 1.8B | Yes | 90-97% (real) | 10-100 | Yes (VLM) | LIBERO-Long 82% | Challenging |
| Octo | 93M | Yes | ~72% (fine-tuned) | ~100 | Weak | Limited | Excellent |
| GR00T N1.6 | 3B | Yes | Competitive | 20-40 | Yes (VLM) | Limited | Challenging |
| RT-2 | 12-55B | No | 90% (sim) | Fleet data | Best | Limited | Impossible |

---

## Key Takeaways for RoArm M3 Project

1. **50 demos is marginal** -- community evidence confirms 50 demos can produce basic behavior but spatial generalization is very limited (+/-10cm). 100+ demos with positional diversity is the minimum for reliable multi-position grasping.

2. **SmolVLA is the right choice for RTX 4090** -- only open VLA that trains comfortably on consumer GPU. GraspVLA (1.8B) would be next option with cloud GPU.

3. **Language-conditioned grasping works natively** in all VLM-based VLAs but quality depends on training data diversity and VLM backbone quality.

4. **Sequential tasks are the frontier** -- Long-VLA (CoRL 2025) is the first dedicated solution. Most VLAs handle multi-step through long language instructions or external planning.

5. **GraspVLA's synthetic pretraining** is a potential game-changer for grasping-focused work -- 93% zero-shot real-world grasping with $5K compute budget.

6. **Voice-to-action is solved in principle** (VLAS, ICLR 2025) but pipeline ASR->VLA is more practical for now.

7. **Spatial generalization is THE bottleneck** for all low-cost arm + VLA setups, not model capacity.

---

## Sources

### Papers (verified arXiv IDs)
- SmolVLA: arXiv 2506.01844 (Jun 2025)
- OpenVLA: arXiv 2406.09246 (Jun 2024)
- OpenVLA-OFT: arXiv 2502.19645 (Feb 2025, RSS 2025)
- pi0: arXiv 2410.24164 (Oct 2024, RSS 2025)
- GraspVLA: arXiv 2505.03233 (May 2025, CoRL 2025)
- Octo: arXiv 2405.12213 (May 2024, RSS 2024)
- RT-2: arXiv 2307.15818 (Jul 2023)
- TwinVLA: arXiv 2511.05275 (Nov 2025, ICLR 2026)
- Bi-VLA: arXiv 2405.06039 (May 2024, IEEE SMC 2024)
- VLAS: arXiv 2502.13508 (Feb 2025, ICLR 2025)
- Long-VLA: arXiv 2508.19958 (Aug 2025, CoRL 2025)
- RDT-1B: arXiv 2410.07864 (Oct 2024)
- GR00T N1: arXiv 2503.14734 (Mar 2025)
- MoS-VLA: arXiv 2510.16617

### Community/Web Sources
- Reddit r/robotics: SmolVLA/pi0.5 on SO-101 experience thread
- Medium (Henry Hu): SmolVLA pick-and-place fine-tuning with 25 demos
- Medium (Correll Lab): SmolVLA failure modes on low-cost robot
- Penn PaL Lab: pi0 in-the-wild evaluation (300+ trials)
- OpenVLA-OFT project page: openvla-oft.github.io
- Google DeepMind blog: RT-2 results
