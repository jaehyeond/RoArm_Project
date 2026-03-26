# Visual Prompting for VLA: Complete Landscape Analysis

> B1 VLA Foundation Model Scientist — systematic search
> Date: 2026-03-25
> Queries executed: 21 arXiv searches + direct ID lookups
> Purpose: VERIFY or KILL the claim "nobody has compared digital vs physical visual prompting for VLA"

---

## TL;DR (Read This First)

**The claim is PARTIALLY TRUE but INCORRECTLY FRAMED.**

| Question | Answer | Confidence |
|----------|--------|------------|
| How many digital visual prompting papers for VLA/robot? | 9-11 confirmed | HIGH |
| How many physical visual prompting AS VLA INPUT papers? | 0 confirmed | HIGH |
| Has anyone compared digital vs physical? | 0 papers | HIGH |
| Is "physical visual prompting for VLA" truly 0 papers? | YES, 0 confirmed | HIGH |
| Does pointing/gesture exist as physical prompting? | Laser HRI exists, but NOT as VLA input | MEDIUM |

**Critical nuance**: The correct framing is not "digital vs physical comparison" but rather:
"Physical visual prompting has NEVER been used as input to a learned manipulation policy (VLA/IL/RL).
Digital visual prompting exists in 9+ papers. The physical-as-policy-input gap is the actual gap."

---

## CATEGORY 1: Digital Visual Prompting for VLA (Overlays on Images)

These papers use software-generated overlays on camera images as input to or output of robot policies.

---

### 1.1 TraceVLA (arXiv 2412.10345)
- **Title**: TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies
- **Venue**: ICLR 2025
- **What it does**: Overlays visual traces (trajectory history as colored path drawn on image) onto the current camera frame. The VLA receives the annotated image with trajectory context.
- **Key insight**: Trace overlays help VLA understand motion history and direction. +30% on challenging manipulation.
- **Physical counterpart**: No. Trace is computed from robot joint history, rendered digitally onto image.
- **Relevance**: DIRECT digital visual prompting for VLA. Anchor paper for this direction.

---

### 1.2 AimBot (arXiv 2508.08113)
- **Title**: AimBot: A Simple Auxiliary Visual Cue to Enhance Spatial Awareness of Visuomotor Policies
- **Venue**: CoRL 2025
- **What it does**: Overlays shooting lines and scope reticles (crosshair) onto multi-view RGB images. These geometric visual cues encode spatial information helping the policy understand target position.
- **Key insight**: Simple geometric overlays improve spatial awareness without architecture changes.
- **Physical counterpart**: A laser crosshair projected onto the workspace would be the physical analogue.
- **Relevance**: Most direct predecessor to physical visual prompting. CoRL'26 must differentiate clearly.

---

### 1.3 GENIMA (arXiv 2407.07875)
- **Title**: Generative Image as Action Models
- **Venue**: IROS 2024
- **What it does**: Fine-tunes Stable Diffusion to "draw joint-action targets" as colored dot overlays on RGB images. These annotated images are fed to a controller mapping visual targets to joint positions.
- **Key insight**: Diffusion model as an image-annotation intermediate representation. Visual targets drawn ON the image.
- **Physical counterpart**: Projected colored dots or markers on real workspace would be the physical version.
- **Relevance**: Closest existing work conceptually to "physical projection as policy input." GENIMA draws digitally; nobody projects physically.

---

### 1.4 PIVOT (arXiv 2402.07872)
- **Title**: PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs
- **Venue**: ICML 2024
- **What it does**: Overlays arrows and numbered circles on images in an iterative refinement loop. VLM selects best action from visually-annotated candidates. Enables robot control without fine-tuning.
- **Key insight for SmolVLA**: Frozen VLMs CAN recognize overlaid arrows/circles from internet pretraining. This directly validates that SigLIP will respond to digital overlays.
- **Physical counterpart**: None attempted.

---

### 1.5 MOKA (arXiv 2403.03174)
- **Title**: MOKA: Open-World Robotic Manipulation through Mark-Based Visual Prompting
- **Venue**: RSS 2024
- **What it does**: Uses VLMs to solve robotic manipulation via mark-based visual prompting. Draws marks (circles, arrows) on images to specify keypoint affordances, then extracts coordinates.
- **Key insight**: Mark-based prompting enables open-world generalization without task-specific training.

---

### 1.6 RT-Trajectory (arXiv 2311.01977)
- **Title**: RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches
- **Venue**: arXiv 2023 (Google)
- **What it does**: Uses 2D trajectory sketches drawn on images as intermediate representations for task generalization.

---

### 1.7 RT-Sketch (arXiv 2403.02709)
- **Title**: RT-Sketch: Goal-Conditioned Imitation Learning from Hand-Drawn Sketches
- **Venue**: arXiv 2024 (Google)
- **What it does**: Hand-drawn sketches as goal specifications for imitation learning.

---

### 1.8 RoboPoint (arXiv 2406.10721)
- **Title**: RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics
- **Venue**: arXiv 2024
- **What it does**: VLM fine-tuned to predict 2D action points on images for robot manipulation.

---

### 1.9 RoVI (arXiv 2505.00693)
- **Title**: Robotic Visual Instruction
- **Venue**: arXiv 2025
- **What it does**: New paradigm — "object-centric visual instruction" as alternative to natural language. Uses visual annotations on images to specify tasks spatially. Addresses language ambiguity and verbosity.
- **Key insight**: Explicitly positions visual annotation as a NEW MODALITY replacing language for robots. Most recent paper (2025) and the one that most directly defines the "visual prompting for robot" frame.
- **Physical counterpart**: Not attempted. Pure digital annotation.

---

### 1.10 Set-of-Mark / SoM (arXiv 2310.11441)
- **Title**: Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V
- **Venue**: NeurIPS 2023 (non-robot, origin paper)
- **What it does**: Overlays numbered segmentation masks on images to enhance VLM grounding. The origin of the mark-based visual prompting paradigm adopted by MOKA, PIVOT, etc.

---

### 1.11 RoboVIP (arXiv 2601.05241)
- **Title**: Multi-View Video Generation with Visual Identity Prompting Augments Robot Manipulation
- **Venue**: arXiv 2026
- **What it does**: Uses visual identity prompting in a video generation model to augment robot training data.
- **Relevance**: More data augmentation than visual prompting at inference time.

---

## CATEGORY 2: Physical Visual Prompting — Laser/Projector AS Policy Input

**CONFIRMED COUNT: 0 papers**

After 21 search queries, no paper was found that:
1. Uses a physical projector, laser pointer, LED, or marker on a real workspace
2. AND uses a learned manipulation policy (VLA, imitation learning, RL, behavior cloning)
3. Where the physical cue is visible through the robot's camera and thus serves as policy input

All physical projection papers found fall into:
- Safety systems (Vogel/Fraunhofer, 2011-2017)
- Intent communication for human-robot interaction (Chadalavada 2015/2020)
- Assistive interfaces (Torielli 2025 — laser for wheelchair arm, hardcoded control)
- Industrial programming interfaces (PATI 2019, Andersen 2015)
- Swarm stigmergy (LARS 2411.00007, 2024)

**Closest physical paper (Torielli 2503.15987):**
User points laser at object → wheelchair robotic arm grasps it. BUT the robot uses hardcoded affordance recognition + kinematics. NOT a learned policy. NOT a VLA. This is the direct physical analogue of what we are proposing to combine with VLA.

---

## CATEGORY 3: Comparison of Digital vs Physical Visual Prompting

**CONFIRMED COUNT: 0 papers**

Query results from "comparing physical digital visual prompting robotics" and related queries returned no relevant papers. No systematic comparison exists.

---

## The SigLIP Frozen Encoder Constraint Analysis

SmolVLA uses frozen SigLIP (ViT, pretrained on internet images). Implications for both modalities:

### Digital overlays through SigLIP:
- PIVOT (2402.07872) CONFIRMS frozen VLMs recognize arrows/circles/numbers from pretraining
- SigLIP was trained on internet images containing annotations, pointers, diagrams
- Arrows, circles, crosshairs are IN the pretraining distribution
- Estimated probability SigLIP responds to digital overlays: HIGH

### Physical markers through SigLIP:
- A laser dot, projected circle, or colored tape on a real workspace appears in internet images
- SigLIP has seen laser pointer demo photos, projector setup photos, colored stickers
- Estimated probability SigLIP recognizes physical markers: MEDIUM
- Key uncertainty: will SigLIP ATTEND to a small laser dot vs a prominent object?

### The empirical gap nobody has tested:
Does SigLIP encode a digitally-rendered red circle at position X identically to a physically-projected red circle at position X (captured through camera)?

If cosine_similarity(SigLIP(digital_circle_image), SigLIP(physical_circle_image)) ≈ 1.0,
then digital training data transfers to physical deployment at zero additional cost.

This test can be run in 2 hours using the script in `model_siglip_marker_test.py`.

---

## Gap Verification Summary

### Claim: "Nobody has compared digital vs physical visual prompting for VLA"

**VERDICT: TRUE with HIGH confidence**

Evidence:
- 21 arXiv search queries executed
- Direct ID lookups for 15+ specific papers
- Prior landscape file (PROJECTION_AR_LASER_ROBOT_LANDSCAPE.md) covering 38 papers
- No counter-evidence found for "physical visual prompting + learned policy"

Residual risk (things we cannot fully verify via arXiv):
1. ICRA/CoRL workshop papers not indexed on arXiv (IEEE Xplore manual search recommended)
2. Industry demos without publications (Boston Dynamics, Figure AI, etc.)
3. Robotics lab technical reports / theses

Recommended additional verification:
- Search IEEE Xplore: "laser pointer robot policy" + "projector robot imitation" (2023-2026)
- Check ICRA 2024/2025 workshop proceedings manually

### Accurate framing for the CoRL 2026 paper:

"To our knowledge, while digital visual prompting for robot manipulation has received growing attention — with methods such as TraceVLA (trajectory overlays, ICLR'25), AimBot (geometric reticles, CoRL'25), GENIMA (diffusion-drawn joint targets, IROS'24), PIVOT (iterative arrow prompting, ICML'24), and RoVI (visual instruction modality, 2025) — no prior work has (1) used physical visual cues visible through the robot's camera as input to a learned manipulation policy, or (2) empirically investigated whether frozen vision encoders respond equivalently to digital vs. physical instantiations of the same visual prompt. We address both questions."

---

## Research Design Recommendations

### Fast-path (CoRL 2026 deadline compatible):

**Phase 1 (2 hours, immediate)**: SigLIP marker recognition test
- Script: `model_siglip_marker_test.py`
- Take 10 workspace photos
- Add digital overlays (circle, arrow, crosshair) in software
- Photograph same workspace with physical markers (colored tape, laser pointer)
- Compare SigLIP embeddings: cosine similarity between digital and physical versions
- GATE: If similarity > 0.95, physical prompting is viable with digitally-trained policy

**Phase 2 (1-2 weeks)**: Proof-of-concept training
- Collect 30 episodes with physical colored tape marking the target object
- Train SmolVLA on these episodes
- Test whether model uses the tape cue (test with/without tape)
- Compare performance: language-only vs language+physical-marker

**Phase 3 (if Phase 2 works, 2-4 weeks)**: Full comparison
- 4 conditions: no visual prompt / digital overlay / physical marker / language+digital
- 50 episodes per condition
- Metrics: success rate, data efficiency (episodes to reach threshold)

### Lower-risk path (builds on digital side only):
Apply TraceVLA-style trace overlays to SmolVLA on consumer hardware + RoArm-M3.
Novelty: first to apply visual trace prompting to sub-1B VLA on consumer hardware with 6-DOF non-SO-100 arm.
Risk: incremental. CoRL reviewers may see this as "replication at small scale."

**Recommendation**: Physical visual prompting is the stronger angle. Phase 1 test takes 2 hours.
Run it immediately. If SigLIP similarity > 0.95, go with physical.

---

## Paper Count Final Tally

| Category | Count | Notes |
|----------|-------|-------|
| Digital visual prompting for learned robot policy | 9 confirmed | TraceVLA, AimBot, GENIMA, PIVOT, MOKA, RT-Traj, RT-Sketch, RoboPoint, RoVI |
| Digital visual prompting (origin/non-robot) | 1 | SoM (NeurIPS 2023) |
| Digital visual augmentation (data aug, not prompting) | 2 | ROSIE, RoboVIP |
| Physical laser/projector + hardcoded robot control | 3 | Torielli, Kaiser, LARS |
| Physical laser/projector + learned policy | 0 | THE GAP |
| Comparison of digital vs physical | 0 | THE GAP |
