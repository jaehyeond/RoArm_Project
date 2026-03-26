# How Robots Understand and Grasp Objects: An Honest Assessment (March 2026)

> **Purpose:** Critical, evidence-based analysis of the state of robotic grasping and object understanding.
> **Method:** Cross-referenced arXiv papers, NVIDIA blogs, community reports, peer-reviewed venues (RSS, CoRL, ICRA, ICLR, NeurIPS 2024-2026), and this project's own deployment experience.
> **Bias warning:** I explicitly flag where evidence is from marketing/blogs vs peer-reviewed papers.

---

## TL;DR — The Honest Answer

**Is robotic grasping hard?** It depends enormously on what you mean:

| Task | Difficulty in 2026 | Best Success Rate | Caveat |
|------|-------------------|-------------------|--------|
| Pick known object at known position | **Solved** | 99%+ | Industrial arms do this daily |
| Pick unknown rigid object from bin | **Largely solved** | 93% (AnyGrasp) | Parallel gripper, good depth sensor |
| "Pick the red cup" (language-conditioned) | **Works but fragile** | 70-95% | Highly sensitive to prompt phrasing, lighting, scene |
| Pick deformable/transparent object | **Hard** | 60-80% | Still active research |
| Pick novel object in novel scene (zero-shot) | **Partially solved** | 24-93% depending on method | Massive variance; marketing vs reality gap |
| Multi-step manipulation (pick, pour, place) | **Hard** | 42-85% | Long-horizon = compounding errors |
| Dexterous in-hand manipulation | **Very hard** | ~80% sim-to-real on known objects | Allegro/Shadow hand only; fragile transfer |

**The uncomfortable truth:** The field has gotten very good at grasping in controlled settings and very good at publishing impressive numbers. But robust, general-purpose grasping in truly unstructured environments remains unsolved. The gap between lab demos and real deployment is still large.

---

## 1. Do Robots "Understand" What They're Grasping?

### 1.1 What Foundation Models Actually Provide

Foundation models (CLIP, SigLIP, DINOv2) give robots **semantic features**, not **understanding**:

| Model | What It Does Well | What It Does NOT Do |
|-------|------------------|---------------------|
| **CLIP/SigLIP** | Maps images to language-aligned embeddings; "red cup" matches image of red cup | Does NOT understand physical properties (weight, fragility, deformability) |
| **DINOv2** | Rich visual features; better at instance-level differentiation than CLIP | Does NOT provide language grounding; requires separate mapping to actions |
| **Both combined** (OpenVLA, SmolVLA) | Semantic grounding + visual detail | Neither provides grasp affordance or physics understanding |

**Critical distinction:** These models provide **recognition** ("that is a cup"), not **understanding** ("cups are hollow, fragile, should be grasped from the side, and can contain liquid"). The difference matters enormously for manipulation.

**Evidence:**
- DINOv2 can distinguish two different mugs from each other better than CLIP (MEXC analysis, 2024). But neither tells you WHERE to grasp a mug.
- Probing 3D awareness of vision foundation models (arXiv:2404.08636): DINOv2 achieves competitive 3D awareness, but CLIP is "clearly inaccurate for objects" in 3D spatial reasoning.
- OpenVLA uses DINOv2+SigLIP dual encoder precisely because neither alone is sufficient — SigLIP for language grounding, DINOv2 for spatial detail.

### 1.2 How VLAs "Understand" Objects

VLAs do NOT understand objects. They learn **statistical correlations** between visual patterns and motor actions:

**What actually happens inside a VLA when you say "pick up the red cup":**
1. SigLIP/CLIP encodes the image into a feature vector that is close (in embedding space) to the text "red cup"
2. The VLM backbone (Llama, SmolLM, PaliGemma) processes this as a sequence prediction task
3. The action head (flow matching / diffusion) generates motor commands conditioned on the VLM output
4. At NO point does the system reason about "cup-ness," "red-ness," or grasping affordances explicitly

**Evidence of the gap:**
- pi0 evaluation at Penn PaL Lab: "Close the toilet" = 0% success. "Close the white lid of the toilet" = 100%. Same physical action, different prompt. This is pattern matching, not understanding.
- SmolVLA on SO-101 (Reddit): Works within +/-10cm of training positions, completely fails outside. The model memorized spatial correlations, not object understanding.
- RT-2 showed the most "emergent understanding" (picking "an extinct animal" = dinosaur figurine), but this came from the 55B PaLM-E backbone, not from manipulation training.

### 1.3 The GraspVLA Exception

GraspVLA (PKU, CoRL 2025) is the closest to genuine "object understanding" for grasping:

- **Progressive Action Generation**: First predicts a 2D bounding box around the target object, THEN generates grasp actions. This two-stage approach forces the model to explicitly localize the object before acting.
- **SynGrasp-1B**: Pre-trained on 1 billion synthetic frames across 10,680 objects in 240 categories
- **Result**: 93.3% zero-shot real-world grasping, robust to lighting changes (96.6%) and distractors (93.3%)
- **Why it matters**: The explicit localization step is a form of structured reasoning that other VLAs lack

**But even GraspVLA doesn't "understand"** — it has learned extremely robust visual-motor correlations across massive object diversity. It would still fail on a novel object with deceptive visual properties (e.g., a cup-shaped rock that is too heavy to grasp).

---

## 2. Language-Conditioned Grasping: How Well Does "Pick Up the Red Cup" Work?

### 2.1 Success Rates by System

| System | Task | Success Rate | Conditions | Source |
|--------|------|-------------|------------|--------|
| **RT-2 (55B)** | Open-vocabulary manipulation | ~90% (Language Table sim) | Google's fleet, closed | arXiv:2307.15818 |
| **RT-2** | Emergent tasks (novel commands) | 3x over RT-1 baseline | Closed model, proprietary data | Google paper |
| **OpenVLA-OFT** | LIBERO-90 (incl. language) | 95-97% | Fine-tuned, sim benchmark | arXiv:2502.19645, RSS 2025 |
| **OpenVLA-OFT** | ALOHA real-world | Best among tested VLAs | 20-300 demos per task | RSS 2025 |
| **pi0** | Familiar tasks (in-lab) | ~90-95% | Trained distribution | arXiv:2410.24164 |
| **pi0** | In-the-wild (Penn PaL) | **24%** pick-and-place | Novel environment, zero-shot | Penn PaL Lab, 300+ trials |
| **SmolVLA** | Single-position grasp | ~90% within boundary | +/-10cm only, 50+ demos | Reddit community |
| **SmolVLA** | Multi-position grasp | Fails outside training dist. | Narrow spatial generalization | Community reports |
| **GraspVLA** | Zero-shot real grasping | 93.3% | Rigid objects, good lighting | arXiv:2505.03233, CoRL 2025 |
| **Octo (93M)** | Language-conditioned | 25% lower than goal-image | Language is weaker modality | arXiv:2405.12213, RSS 2024 |

### 2.2 The Real Story

**Language conditioning works, but with massive caveats:**

1. **Prompt sensitivity is extreme**: pi0 shows 0% vs 100% on the SAME physical task with different wording. This is not "understanding" — it is pattern matching against training data distribution.

2. **Color/attribute grounding works via VLM backbone**: All VLM-based VLAs (SmolVLA, OpenVLA, pi0, GraspVLA) can ground "red" via their pre-trained vision encoder. But the action head must have been trained on spatially diverse data to actually reach different positions.

3. **No published study isolates color generalization**: There is no paper that specifically reports "N demos needed for color-conditioned generalization." The VLM backbone gives you color semantics for free; what you actually need demos for is spatial diversity.

4. **Open-vocabulary is not the same as open-world**: RT-2 can pick up "an extinct animal," but it cannot pick up an object it has never physically interacted with in a category it has never grasped. The semantic understanding transfers; the motor understanding does not.

### 2.3 Classical Grasp Detection + Language (The Pipeline Approach)

An alternative to end-to-end VLAs: use a foundation model for object detection, then a grasp planner.

**Example pipeline:**
```
Language instruction → CLIP/GroundingDINO (object detection) → AnyGrasp/Contact-GraspNet (grasp pose) → Motion planner → Execute
```

| System | Approach | Success Rate | Source |
|--------|----------|-------------|--------|
| ThinkGrasp (arXiv:2407.11298) | GPT-4V + grasp planner | Improved in clutter | 2024 |
| SegGrasp (arXiv:2410.08901) | SAM + Contact-GraspNet | Better segmentation → better grasp | 2024 |
| Ground4Act (ScienceDirect) | VLM + pushing + grasping | 90%+ in clutter | 2024 |
| GraspGPT (arXiv:2307.13204) | LLM for task-oriented grasping | Task-appropriate grasps | 2023 |

**Critical comparison**: The pipeline approach often achieves comparable or better results than end-to-end VLAs for **grasping specifically**, because:
- Grasp detection (AnyGrasp, Contact-GraspNet) is very mature (93%+ success)
- Foundation models (CLIP, SAM) are very good at localization
- The weakness is in the integration (slow, brittle handoffs) and inability to learn from experience

---

## 3. VLA vs RL for Manipulation — Honest Comparison

### 3.1 What Each Approach Actually Does

| Dimension | VLA (SmolVLA, pi0, OpenVLA) | RL (PPO/SAC in Isaac Lab) |
|-----------|----------------------------|--------------------------|
| **Object understanding** | Implicit via VLM backbone (CLIP/SigLIP/DINOv2 features) | None — operates on state vectors (joint angles, object poses) or raw pixels |
| **How it generates actions** | Predicts actions conditioned on image+text | Optimizes a reward function through trial-and-error |
| **Data source** | Human demonstrations (10-300) | Millions of sim episodes (self-generated) |
| **Generalization** | Can generalize to objects semantically similar to training | Can generalize to poses/dynamics within randomization range |
| **Novel objects** | Partial (VLM backbone recognizes; action head may not adapt) | None without retraining (no object semantics at all) |
| **Language** | Native support | Not possible without separate module |
| **Training cost** | Hours on single GPU (fine-tuning) | Minutes to hours in GPU-parallel sim; months for sim-to-real |
| **Real-world data** | Requires human demos | Requires sim-to-real transfer |

### 3.2 Where Each Actually Works

**VLA wins:**
- Tasks requiring semantic understanding ("pick the edible item")
- Multi-task policies from shared backbone
- Quick adaptation to new tasks with few demos (25-100)
- Language-conditioned behavior

**RL wins:**
- High-speed, precision manipulation (peg insertion: 83-99% sim-to-real, NVIDIA IndustReal)
- Dexterous in-hand manipulation (cube rotation with Allegro hand)
- Locomotion (quadruped, humanoid — universally RL)
- Tasks with clear reward signals and dynamics models
- When you have a good simulator but no human demos

**Neither wins alone:**
- Long-horizon multi-step tasks in novel environments
- Deformable object manipulation
- Truly zero-shot generalization to arbitrary objects

### 3.3 The Honest Numbers

**VLA real-world manipulation:**
| Model | Real Success Rate | Task | Notes |
|-------|------------------|------|-------|
| pi0 (in-lab) | 90-95% | Familiar tasks | Massive pretraining, closed |
| pi0 (in-the-wild) | **24%** pick-and-place | Novel environment | Penn PaL, 300+ trials |
| pi0 (in-the-wild) | **42.3% avg progress** | 15 task types | Not completion, just progress |
| OpenVLA-OFT | 95-97% | LIBERO sim benchmark | Sim, not real |
| SmolVLA | 40-90% | Grasping | Depends on data quality, boundaries |
| GraspVLA | 93.3% | Zero-shot grasping | Best in class, but grasping only |
| This project (SmolVLA) | **0% then 100%** | Sponge grasp | v1 failed, v3 succeeded after fixes |

**RL real-world manipulation (sim-to-real):**
| System | Real Success Rate | Task | Notes |
|--------|------------------|------|-------|
| NVIDIA IndustReal | 83-99% | Peg insertion, gear assembly | 600 trials, zero-shot sim-to-real |
| NVIDIA TacSL | 83-91% | Tactile insertion | With touch feedback |
| NVIDIA DeXtreme | Repeated success | Cube rotation (Allegro) | 32+ hours training in sim |
| NVIDIA sim-to-real humanoid | ~80% | Dexterous grasping novel objects | arXiv:2502.20396 |
| NVIDIA TriFinger | 83% | In-hand manipulation | Domain randomization |
| OpenAI Rubik's Cube | Solved | In-hand cube rotation (Shadow) | Massive DR, 2019, not reproduced |
| SERL (SAC) | >90% | PCB insertion, cable routing | 25-50 min real training, NOT sim-to-real |
| RealTo-sim-to-real | +67% robustness | Tabletop manipulation | Franka, digital twin |

### 3.4 The Hybrid Trend (2025-2026)

The field is converging: **VLA for perception + RL for refinement**.

| Paper | Approach | Improvement | Venue |
|-------|----------|-------------|-------|
| Beyond Imitation (2026) | SFT warm-start → RL in sim | +24% OpenVLA, +20% pi0.5 | arXiv:2602.12628 |
| VLA-RL (2025) | Online RL to improve VLA | Improved over SFT-only | arXiv:2505.18719 |
| PLD (Self-Improving VLA) | Residual RL + data collection | 99% on LIBERO | arXiv:2511.00091 |
| VLAC (Critic) | Critic model for VLA | Dense reward from VLM | arXiv:2509.15937 |
| ReinFlow | RL fine-tuning of flow matching | Improved exploration | 2025 |

**The emerging consensus:** Pre-train VLA on demonstrations, then refine with RL in sim (optionally with real data anchoring). Neither IL alone nor RL alone is optimal.

---

## 4. Isaac Lab RL for Manipulation — What's Actually Solved?

### 4.1 Available Manipulation Environments

From Isaac Lab documentation and NVIDIA blog posts:

| Task | Status | Success Rate (Sim) | Sim-to-Real? |
|------|--------|-------------------|-------------|
| Franka Cabinet (open drawer) | Mature | 150,000+ FPS training | Not tested specifically |
| Franka Cube Lift | Mature | High | Limited validation |
| Pick-and-Place (UR10e, Franka) | Available | ~90% (with tuning) | Partial (TIAGo: 90%, but specific setup) |
| Factory: Peg Insertion | Mature | High, ~1hr training | **Yes: 83-99% over 600 trials (IndustReal)** |
| Factory: Gear Assembly | Mature | High | **Yes: IndustReal** |
| Kuka + Allegro Dexterous | Available (Isaac Lab 2.3) | High in sim | Limited real validation |
| Unitree G1 Pick-Place | New (2025) | Available | Unknown |
| BEHAVIOR benchmark (50 tasks) | Challenge format | Varies widely | Sim-only benchmark |

### 4.2 Critical Assessment of Isaac Lab for Manipulation

**What works well:**
- Industrial assembly tasks (peg insertion, gear meshing) with precise reward engineering
- GPU-parallel training makes RL training fast (minutes instead of days)
- Automatic Domain Randomization (ADR) in Isaac Lab 2.3 helps sim-to-real

**What does NOT work well (or lacks evidence):**
- **Diverse object grasping is NOT solved in Isaac Lab RL**: Most RL environments use single primitive shapes (cubes, cylinders). There is no Isaac Lab environment with hundreds of diverse objects for RL-based grasping.
- **Semantic/language tasks are impossible in pure RL**: RL in Isaac Lab has no mechanism for language conditioning. You would need a separate VLM layer.
- **Sim-to-real for manipulation is fragile**: Success depends heavily on precise sim-real calibration. The 83-99% IndustReal numbers are for **specific industrial parts** with known geometry, NOT general objects.
- **Most impressive NVIDIA numbers are sim-only**: The 150,000 FPS training speed is simulation throughput. Real-world validation at scale is rare.

### 4.3 How RL Handles New Objects It Hasn't Seen

**Short answer: It doesn't, unless specifically designed to.**

- Standard RL trains on specific objects (cube, cylinder, specific peg). Novel objects = retraining.
- **Domain randomization** can create robustness to object SIZE and MASS variation within a range, but not to fundamentally different shapes.
- **DexPoint** (arXiv:2211.09423): Uses point cloud input for RL, enabling generalization to new objects of the SAME CATEGORY. But not cross-category.
- **NVIDIA sim-to-real dexterous** (arXiv:2502.20396): Achieves ~80% success on novel objects — but "novel" means new instances within trained categories, not arbitrary unseen objects.
- **The fundamental limitation**: RL optimizes a reward function. If the reward is "grasp ANY object," you need massive object diversity in sim. This is starting to happen (DexGraspNet: 5,355 objects) but is still far from the diversity of the real world.

---

## 5. Sim-to-Real Gap — Current State of the Art

### 5.1 Is It "Everyone's Problem" or Largely Solved?

**Neither.** The sim-to-real gap is **task-dependent and conditionally solvable**:

| Task Category | Gap Status | Typical Drop | Evidence |
|---------------|-----------|-------------|---------|
| **Locomotion** | **Largely solved** | <5% drop | ANYmal, Unitree, Agility — hundreds of deployed robots |
| **Industrial assembly** (known parts) | **Mostly solved** | 5-17% drop | IndustReal: 83-99% real (NVIDIA, 600 trials) |
| **Simple pick-place** (known objects) | **Mostly solved** | 10-20% drop | Multiple labs, domain randomization works |
| **Dexterous manipulation** (single object) | **Conditionally solved** | 10-30% drop | DeXtreme, DexPoint — specific objects/tasks |
| **Diverse object grasping** | **Partially solved** | 20-40% drop | Needs massive DR + object diversity |
| **Contact-rich tasks** (deformable, liquid) | **NOT solved** | 40-60%+ drop | Physics sim inadequacy |
| **VLA sim-to-real** | **Emerging** | Varies | Frozen vision encoders create visual domain gap |

### 5.2 Domain Randomization Effectiveness (2024-2026)

**Domain randomization works, but it's not magic:**

| Parameter | Easy to Randomize | Hard to Randomize |
|-----------|-------------------|-------------------|
| Visual (textures, lighting) | Yes — standard practice | Camera intrinsics, lens distortion |
| Dynamics (mass, friction) | Yes — ADR in Isaac Lab | Contact dynamics, deformation |
| Object pose | Yes — trivial | Object category/shape distribution |
| Robot parameters | Yes (joint damping, etc.) | Actuator backlash, cable routing |

**Key results:**
- NVIDIA TriFinger sim-to-real: 83% with DR (arXiv:2108.09779)
- DeXtreme: Required 32+ hours of sim training with extensive DR
- IndustReal: 83-99% over 600 trials — but these are **precision industrial parts** with known CAD models
- Reconciling Reality (arXiv:2403.03949): Without DR, BC baseline achieves only 25%. With DR + distractors: 11%. Real-to-sim-to-real approach needed for robustness.

**The honest assessment of DR:**
- DR is necessary but not sufficient for most manipulation tasks
- It helps with appearance variation but struggles with dynamics variation
- The "sim-to-real gap" is increasingly a "physics sim fidelity gap" — visual transfer is nearly solved (especially with 3DGS/Cosmos Transfer), but contact dynamics remain hard
- Newton Physics Engine 1.0 (NVIDIA/DeepMind/Disney, March 2026) may help with deformable/contact-rich tasks

### 5.3 Teacher-Student Distillation

The standard recipe for sim-to-real in 2025-2026:

```
[Privileged Teacher in Sim]     → has ground-truth object pose, contact forces
        ↓ distillation
[Student Policy]                → uses only onboard sensors (RGB, proprioception)
        ↓ deploy
[Real Robot]                    → zero-shot or few-shot adaptation
```

**Results:**
- NVIDIA sim-to-real dexterous (arXiv:2502.20396): Teacher with privileged info → student with vision only → ~80% on novel objects
- PTLD (arXiv:2603.04531): Privileged tactile → latent distillation → dexterous manipulation
- Isaac Lab has built-in student distillation pipeline (docs: Sim-to-Real Policy Transfer)

**Specific success rate drops:**
- Locomotion: sim ~100% → real ~95% (5% drop)
- Industrial insertion: sim ~100% → real 83-99% (1-17% drop)
- Dexterous cube rotation: sim reliable → real "repeated success" (no % given — suspicious)
- Tabletop manipulation: sim ~80-90% → real 60-75% (15-30% drop, from Scaling Sim-to-Real RL, arXiv:2603.18532)
- VLA co-training: sim baseline 9.7% → after scaling 79.8% sim, 75% real (same paper)

---

## 6. Zero-Shot / Open-Vocabulary Grasping Systems

### 6.1 Dedicated Grasp Detection Systems (NOT VLAs)

These are geometry-focused systems that predict WHERE to grasp, not how to manipulate:

| System | Success Rate | Objects | Method | Year |
|--------|-------------|---------|--------|------|
| **AnyGrasp** | **93.3%** bin clearing (300+ unseen objects) | Rigid, any shape | Dense 6-DoF, point cloud + temporal | TRO 2023 |
| **AnyDexGrasp** | **75-95%** (3 hands, 150+ novel objects) | Rigid in clutter | Multi-hand dexterous | 2025 |
| **Contact-GraspNet** | **90%+** (unseen objects, structured clutter) | Rigid | 6-DoF from single depth image | ICRA 2021 |
| **GraspNet-1Billion** | **88%** benchmark | Varied | Dataset + baseline | CVPR 2020 |
| **GraspGen** (2025) | SOTA (surpasses Contact-GraspNet) | Diverse | Diffusion-based 6-DoF | arXiv:2507.13097 |
| **6-DoF VLG** (DexVLG, ICCV 2025) | Improved in clutter | Dexterous | Vision-language guided | ICCV 2025 |

### 6.2 Can They Grasp Objects They've Never Seen?

**Yes, with caveats.**

**AnyGrasp** is the strongest evidence that **geometric grasp detection is largely solved for rigid objects**:
- 93.3% on 300+ unseen objects in bin clearing
- "Comparable with human subjects under controlled conditions"
- Over 900 MPPH (mean picks per hour) on single-arm system
- Works with standard depth cameras

**But "grasp detection" is not "manipulation":**
- AnyGrasp tells you WHERE to grasp. It does not tell you how to approach, what to do after grasping, or how to handle the object.
- It requires a depth sensor with reasonable quality (struggles with transparent/reflective objects)
- It assumes a parallel-jaw gripper. Dexterous hands are much harder.

### 6.3 GraspGPT and Task-Oriented Grasping

GraspGPT (arXiv:2307.13204, 2023) adds **task awareness** to grasping:
- Input: "use the mug to pour water" → system must grasp the handle, not the rim
- Uses LLM (GPT) knowledge about object parts and task requirements
- Maps language to grasp affordances via knowledge graph
- **Not about success rate** — about grasping the RIGHT part for the task

This is genuinely closer to "object understanding" than pure VLAs, because it explicitly reasons about functional parts.

### 6.4 The Full Picture: What's Missing

The fundamental gap in zero-shot grasping (as of March 2026):

1. **Transparent/reflective objects**: Depth sensors fail → 60-80% success at best
2. **Deformable objects**: Grasp planning for cloth, food, cables is unsolved at AnyGrasp-level reliability
3. **Heavy/fragile objects**: No way to infer physical properties from vision alone (force/torque sensing helps but isn't standard)
4. **Cluttered small objects**: Success drops in dense clutter (Contact-GraspNet trained on GraspClutter6D: 68.5% in 15-object scenes)
5. **Post-grasp manipulation**: Grasping ≠ manipulation. Picking up an object is the easy part; using it purposefully is the hard part.

---

## 7. NVIDIA's Actual Manipulation Capabilities

### 7.1 What's Real vs. What's Marketing

| Claim | Evidence Level | Actual Numbers |
|-------|---------------|----------------|
| **IndustReal: zero-shot assembly** | **Peer-reviewed (RSS 2023), 600 trials** | 83-99% success on peg/gear assembly |
| **DeXtreme: cube rotation** | **Published** | "Repeated success after 32hrs training" — no % given |
| **TriFinger manipulation** | **Published, competition** | 83% success with DR |
| **TacSL: tactile insertion** | **Published** | 83-91% success |
| **GR00T-Mimic: 780K trajectories** | Blog post | H100 cluster, humanoid locomotion, NOT manipulation |
| **GR00T N1: LIBERO SOTA** | Paper (arXiv:2503.14734) | Sim benchmark only, no real manipulation numbers |
| **Isaac Lab: 150K FPS** | Documentation | Sim throughput, not real success rate |
| **40% improvement from sim data** | GR00T paper | Humanoid control, baseline unclear |
| **Cosmos Transfer: photorealistic** | Blog + demo | Reduces visual domain gap, no manipulation success rates |
| **Newton 1.0: 475x faster** | GTC 2026 announcement | vs MuJoCo MJX, benchmark tasks |

### 7.2 What's Actually Deployed on Real Robots

**Confirmed real-robot results from NVIDIA research:**
1. IndustReal: Peg insertion + gear assembly on Franka → 83-99% (RSS 2023, 600 trials)
2. DeXtreme: Allegro hand cube rotation → qualitative success
3. TriFinger: In-hand manipulation → 83% (competition setting)
4. TacSL: Tactile insertion → 83-91%
5. Sim-to-Real Dexterous Humanoid: ~80% novel objects (arXiv:2502.20396, 2025)

**NOT confirmed on real robots:**
- GR00T N1 on real manipulation tasks (only sim benchmarks published)
- Large-scale diverse object grasping
- Factory-scale deployment of RL-trained manipulation policies
- Long-horizon multi-step manipulation

### 7.3 Honest Assessment

NVIDIA's manipulation work is **strong in specific industrial tasks** (assembly, insertion) where:
- The environment is well-characterized (known objects, precise CAD models)
- The reward function is clear (did the peg go in?)
- GPU-parallel training gives a decisive speed advantage

NVIDIA is **weaker in** (or lacks published evidence for):
- Diverse, open-world object manipulation
- Language-conditioned manipulation (deferred to GR00T N1, which lacks real-world validation)
- Deformable object handling (Newton 1.0 is promising but unproven for policy learning)

---

## 8. Summary: What's Really Hard and What's Not

### Solved (>90% reliable)
- Bin picking of rigid objects with known geometry (industrial)
- Geometric grasp detection for unseen rigid objects (AnyGrasp: 93%)
- Sim-to-real for locomotion
- Sim-to-real for precision assembly with known parts (IndustReal)
- Single-task imitation learning with 50+ good demos (within training distribution)

### Mostly Working (70-90%, active improvement)
- Language-conditioned grasping of seen object categories
- VLA fine-tuning for specific manipulation tasks
- Dexterous in-hand manipulation (with specific hardware + extensive training)
- Sim-to-real with domain randomization for simple manipulation

### Hard (40-70%, active research frontier)
- Zero-shot manipulation in truly novel environments (pi0 in-the-wild: 42%)
- Language-conditioned manipulation with prompt sensitivity
- Transparent/deformable object grasping
- Multi-step, long-horizon manipulation
- Sim-to-real for contact-rich tasks

### Unsolved (<40% or no reliable method)
- General-purpose "pick up anything anywhere" robot
- Understanding object physics from vision alone
- Dexterous manipulation of arbitrary objects
- Manipulation in truly unstructured environments (homes, outdoors)
- Safe manipulation (knowing when NOT to grasp)

---

## 9. Implications for This Project (RoArm M3 + SmolVLA)

### What the landscape tells us:

1. **SmolVLA is the right choice for our hardware** — it is the only open VLA trainable on RTX 4090. GraspVLA (1.8B) would be better for grasping but needs more VRAM.

2. **Our v3 100% success rate is within expected range** for single-position, well-demonstrated tasks. It is NOT generalizable — this is consistent with all community reports.

3. **Spatial generalization is THE bottleneck**, not model capacity. Every system from SmolVLA to pi0 struggles with this. More positional diversity in training data is the only proven solution.

4. **A hybrid pipeline (AnyGrasp for grasp detection + VLA for high-level) might outperform pure VLA** for grasping tasks. But it requires depth sensing (Azure Kinect provides this) and significantly more engineering.

5. **RL in Isaac Lab could complement, not replace, our VLA approach** — but only if we have a URDF for RoArm M3 and invest in sim environment creation. The payoff would be in precision/robustness, not in semantic understanding.

6. **The "100+ episodes" requirement is confirmed** by the entire field. No VLA achieves reliable multi-position grasping with fewer.

---

## Sources & Confidence Levels

### Peer-Reviewed (HIGH confidence)
- AnyGrasp: TRO 2023, 93.3% success, 300+ objects
- Contact-GraspNet: ICRA 2021, 90%+ success
- IndustReal: RSS 2023, 83-99% over 600 trials
- OpenVLA-OFT: RSS 2025, LIBERO 95-97%
- pi0: RSS 2025
- GraspVLA: CoRL 2025, 93.3% zero-shot
- Octo: RSS 2024
- Sim-and-Real Co-Training: RSS 2025

### Preprints (MEDIUM confidence)
- Penn PaL pi0 evaluation: 42.3% average progress (blog + preprint, 300+ trials)
- Sim-to-real dexterous humanoid: ~80% (arXiv:2502.20396)
- Beyond Imitation (RL-Co): +24% OpenVLA (arXiv:2602.12628)
- GR00T N1: arXiv:2503.14734

### Community Reports (LOW-MEDIUM confidence)
- SmolVLA on SO-101: ~90% within boundary (Reddit)
- SmolVLA on low-cost arm: 40-60% (Medium/Correll Lab)
- This project: 0% → 100% (v1 → v3)

### Marketing/Blog (LOW confidence, treat as claims)
- GR00T-Mimic: 780K trajectories in 11 hours
- Newton 1.0: 475x faster
- Cosmos Transfer: photorealistic sim data
- Tesla Optimus: zero published numbers
