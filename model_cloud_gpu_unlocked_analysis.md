# Cloud GPU Unlocked: What Changes for This Project
# B1 VLA Foundation Model Scientist Analysis
# Date: 2026-03-24

---

## Framing: What Constraint Actually Changed

**Before**: Train and infer on RTX 4090 Laptop (15.6GB VRAM)
- SmolVLA (450M) full fine-tune: OK (9.85GB at batch=64)
- pi0 (3B) full fine-tune: **impossible** (>40GB)
- OpenVLA (7B) full fine-tune: **impossible** (>80GB)
- OpenVLA-OFT (LoRA): marginal (requires quantization)

**After**: VAST.ai / GCP = A100 (40/80GB) or H100 on-demand
- pi0 (3B) full fine-tune: OK (A100 40GB, batch=32-64)
- OpenVLA (7B) full fine-tune: OK (A100 80GB or 2xA100 40GB)
- Multi-model training comparison: OK
- Inference (deployment): still uses RTX 4090 locally for latency-sensitive testing
  OR cloud inference if open-loop (4-chunk, 200ms latency okay)

**Critical point**: Inference for 4-chunk open-loop deployment
= 50 steps pre-computed = no real-time requirement
= cloud inference is viable for experiments (not for final product)

---

## What Becomes Possible (Categorized by Confidence)

### TIER 1: Directly enables the CoRL paper (HIGH impact, do now)

#### 1A. pi0 Fine-tuning on RoArm-M3
**Why it was impossible before**: pi0 (3B) full fine-tune needs ~40GB VRAM minimum.
**What cloud gives**: A100 40GB = pi0 full fine-tune with batch=32+.

**Research value**: This is the most important unlock.
- pi0 uses flow matching like SmolVLA but at 3B vs 450M
- pi0 was pretrained on 68 robot types (OXE + proprietary) — much richer cross-embodiment
- RoArm-M3 is OOD for both, but pi0's richer pretraining may transfer better
- **Direct comparison on same robot/task/data = controlled ablation**

Concretely: "Does pretraining breadth (pi0, 68 robots) outperform
architecture efficiency (SmolVLA, cleaner flow matching) on extreme OOD embodiment?"

**Feasibility**: pi0 weights are public (pi0-fast, arXiv 2412.06727).
LeRobot 0.4.4 supports pi0-fast natively (same lerobot-train CLI).
Run time: A100 40GB, 200K steps, estimated 8-12 hours.
Cost: VAST.ai A100 ~$1.5-2/hr → $12-24 per training run.

**CAUTION**: pi0 expects specific image normalization and camera configs.
Must verify compatibility with RoArm-M3 LeRobot dataset format before training.
Test: load pi0-fast with our dataset — does lerobot-train accept it without code change?

#### 1B. OpenVLA-OFT Full Fine-tune (not just LoRA)
**Before**: OpenVLA 7B needed LoRA + int8 quantization even to run.
**After**: A100 80GB = OpenVLA full fine-tune, batch=16+.

**Research value**:
- OpenVLA-OFT (arXiv 2501.03257) is the current SOTA for VLA fine-tuning methods
- Their key contribution: "action chunking + parallel decoding" for OpenVLA
- Training on same RoArm data = direct comparison datapoint

**Feasibility**: MEDIUM. OpenVLA-OFT uses a custom training script (not lerobot-train).
Pipeline integration = 2-3 days of work.
Cost: A100 80GB on VAST.ai ~$2.5-3.5/hr → $20-35 per training run.

**Recommendation**: Include in comparison table but deprioritize vs pi0.
If lerobot-train doesn't support OpenVLA natively, write a data adapter instead.

#### 1C. Octo (93M) — Already Possible Locally, But Cloud Enables More
Octo is 93M, runs on RTX 4090. Cloud adds speed (10x batch size).
Lower priority — Octo is older (2023) and less relevant to 2026 reviewers.
Include in ablation table for completeness (small/medium/large VLA sweep).

---

### TIER 2: Enables new research directions (MEDIUM impact, evaluate carefully)

#### 2A. Depth-Enhanced VLA Input (ZED Mini + Azure Kinect RGBD)
**The idea**: Modify VLA to accept RGBD (RGB + depth channel) instead of RGB only.
**What cloud enables**: Larger models with depth encoder can be trained.

**Why this is interesting**:
- Azure Kinect gives 512x512 depth map, aligned with RGB
- ZED Mini on wrist = close-range depth for grasping phase
- Grasping failures in v1/v2 were partly due to depth ambiguity

**Why this is risky**:
- SmolVLA SigLIP is frozen and only processes RGB
- To use depth, you need to modify the VLM input OR add a separate depth encoder
- "Modifying VLM input" = unfreeze SigLIP or add adapter = 3-4x more compute, weeks of work
- Adding separate depth encoder = new architecture, not comparable to baseline VLAs

**Feasibility**: LOW-MEDIUM for CoRL. MEDIUM-HIGH for thesis chapter.
Cloud GPU enables training the architecture, but architecture design is the bottleneck.
3-4 months of work, not 3-4 weeks.

**Better framing for thesis**: "When does depth actually help VLA grasping?"
= systematic study. Advisor's 3DGS expertise fits here.

#### 2B. 3-Camera Training (Matching smolvla_base Pretraining)
**Key finding from architecture analysis**:
smolvla_base was pretrained with 3 cameras (camera1/2/3 on SO-100).
Our single-camera fine-tuning discards 2/3 of pretrained visual capacity.

**What cloud gives**: Training with 3 cameras = 291 token sequences (vs 163 single-cam).
Memory: ~0.8GB extra per camera → 3-cam at batch=64 needs ~11-12GB locally,
but with cloud A100 can use batch=128+ easily.

**Hardware requirement**: We now have Azure Kinect ×3 and ZED Mini ×1.
- Azure Kinect as wrist cam: possible but large and heavy for wrist mounting
- ZED Mini as wrist cam: designed for this use case, stereo 720p
- 3-camera setup: external view (Azure Kinect #1) + overhead (Azure Kinect #2) + wrist (ZED Mini)

**Research value**: This is actually one of the cleanest experiments possible.
- Ablation: 1-cam vs 2-cam vs 3-cam on same task
- Tests whether matching pretraining camera count matters
- SmolVLA paper does not provide this ablation (opportunity!)

**Feasibility**: MEDIUM. Hardware is available. Main work:
1. Mount cameras rigidly (2 hours)
2. Calibrate multi-camera extrinsics (1 day)
3. Update LeRobot data collection to capture 3 streams (1-2 days)
4. Retrain from scratch with 3-cam config (cloud GPU, 1 day)

**CAUTION**: Camera count change = all existing data invalid. Full recollection needed.
This should be decided BEFORE the main data collection push, not after.

#### 2C. pi0 vs SmolVLA: Data Efficiency Scaling Study
**The idea**: How many episodes does each VLA need to reach X% success?
- SmolVLA (450M, SO-100 pretraining): measured = ~74ep for 100%
- pi0 (3B, 68-robot pretraining): unknown for RoArm-M3

**What cloud enables**: Training pi0 at multiple data sizes (10/25/50/100/200ep).
Each training run = 6-8 hours on A100. Full sweep = ~30-40 hours = $45-80 total.

**Research value**: HIGH for the core paper argument.
"Does a better-pretrained larger VLA need fewer real-robot demos?"
This directly answers the "minimum demos" question.

**Connection to current paper direction**: This IS the experiment that makes
"AR-Guided + Quality Oracle" generalizable. If the method works for SmolVLA
AND reduces pi0 data requirements by same factor, the method is model-agnostic.

**Feasibility**: HIGH. Pure training runs, no new hardware/code needed (assuming pi0 is lerobot-train compatible).

---

### TIER 3: Novel directions, higher risk, longer timeline

#### 3A. VLA + Projected Affordances (Robot Shows Plan to Human)
**The idea**: VLA plans trajectory → project trajectory as light on workspace via projector →
human can correct before execution. Bidirectional communication.

**Previously analyzed**: Projector+Unity+SAM2 had 4 critical failures.
BUT the core communication idea is still valid with a different implementation:
- Instead of projector: display the planned trajectory on a monitor in AR overlay
- ZED Mini (wrist) provides real-time 6-DoF pose → Unity AR overlay of planned motion
- Human approves/rejects/corrects before robot executes

**What cloud gives**: Can run larger VLA (pi0) for the "brain" component.
**Hardware fit**: ZED Mini (6-DoF pose), Unity (AR), 3 Kinects (scene understanding).
**Research gap**: "VLA trajectory preview for human correction" — needs verification.
**Feasibility for CoRL**: LOW (2 months is tight for UI + integration + eval).
**Feasibility for thesis**: HIGH. This is the XR+robotics combo the advisor would love.

#### 3B. Cross-Embodiment Transfer Study with 3 Robots
**The idea**: Train pi0 on Robot A (sponge pick) → transfer to Robot B (different task?) →
and C simultaneously.

**Hardware**: 3 RoArm-M3 robots = identical embodiment.
**Problem**: They're the same robot — "cross-embodiment" within identical hardware is not cross-embodiment.
**Revised idea**: 3 robots as data collection accelerators (3x episodes per day),
THEN test single-model multi-task (Task 1, 2, 3 on the SAME model).

**Research value**: Multi-task VLA on consumer hardware — feasible and underexplored.
**Feasibility**: MEDIUM. Requires all 3 setups to be calibrated identically.
Training multi-task is standard (task text prompt distinguishes tasks).

#### 3C. Self-Improving Loop with Cloud Inference as Judge
**Previously**: Qwen2.5-VL 3B judge needed to run locally on RTX 4090 (tight with SmolVLA).
**With cloud**: Run Gemini/GPT-4V via API as judge OR run Qwen2.5-VL on cloud.
Better judge = higher precision → loop is more reliable.

**Also**: VLM judge doesn't need to run during robot execution —
it runs AFTER the episode is recorded. So cloud API latency is fine.

**Feasibility**: HIGH for the judge component. The loop architecture is unchanged.
Main risk (85% judge accuracy threshold) is unchanged by cloud GPU.

---

## Critical Prioritization: What to Actually Do

Given CoRL deadline (5/28, 65 days), the cloud GPU changes the priority order:

### DO IMMEDIATELY (this week)
1. **Test pi0-fast compatibility** with our LeRobot dataset format.
   - Load pi0-fast from HuggingFace, run one training step on our data.
   - If it works: pi0 comparison becomes core paper experiment.
   - If it doesn't: 1-2 days of data adapter code needed.
   - Command: `lerobot-train --policy.path=lerobot/pi0fast --dataset.repo_id=our_data`
   - This is a 2-hour test. Do before committing to comparison experiment.

2. **Decide camera count** before data recollection.
   - Single-cam vs 3-cam changes data collection protocol entirely.
   - 3-cam gives stronger paper (matches pretraining, ablation possible) but needs 1 day of setup.
   - Decision gate: can ZED Mini mount on wrist without obstructing RoArm-M3 range?
     Run `hw_wrist_camera_feasibility.py` result and check.

### DO IN WEEK 1-2
3. **pi0 vs SmolVLA data efficiency experiment** (if compatibility confirmed).
   - Collect 100ep (AR-guided + Oracle) on Task 1 (sponge pick).
   - Train SmolVLA at 25/50/74/100ep + pi0 at same splits.
   - Deploy and measure success rate.
   - This becomes Figure 2 or Table 1 of the paper.

4. **OpenVLA-OFT compatibility check** (2-hour test).
   - Lower priority than pi0, but inclusion makes paper more complete.

### DEFER (after core experiment)
5. Depth-enhanced VLA architecture (thesis chapter, not CoRL).
6. VLA + AR trajectory preview (thesis Chapter 5/6, not CoRL).
7. 3-robot multi-task (add-on if data collection is running in parallel).

---

## Architecture Implications: SmolVLA vs pi0 vs OpenVLA-OFT

### Why SmolVLA is still valuable even with cloud GPU

SmolVLA (450M) remains the **fastest iteration target** even when cloud is available:
- Training: 200K steps = ~4 hours on A100 vs 8-12h for pi0
- Cost: $6-8 vs $12-24 per run
- Local deployment (RTX 4090): SmolVLA full inference, pi0 may need quantization
- **Full fine-tune without LoRA** = cleaner ablations, no LoRA confounds

For the paper, SmolVLA's role shifts from "only option" to "efficient baseline."
pi0 becomes the "stronger baseline that validates our data collection method is model-agnostic."

### VLA Comparison Table for Paper (Proposed)

| Model | Params | Pretraining | Fine-tune method | Data needed (RoArm) | Deploy latency |
|-------|--------|-------------|------------------|---------------------|----------------|
| SmolVLA | 450M | SO-100 only | Full (Action Expert) | 74ep (measured) | ~108ms/step |
| pi0-fast | 3B | OXE + 68 robots | Full (cloud GPU) | TBD | TBD |
| OpenVLA-OFT | 7B | OXE | LoRA (cloud GPU) | TBD | TBD |
| Octo | 93M | OXE | Full (local GPU) | TBD | TBD |

**Key question the table answers**: Does pretraining breadth reduce demo requirements?
SmolVLA is maximally OOD. If it achieves same success as pi0 with same data quality
(AR-guided + Oracle), the data collection method is the bottleneck, not the model.
This is a strong result either way.

### Action Chunking Comparison Opportunity
- SmolVLA: n_action_steps=50, flow matching
- pi0-fast: n_action_steps=50, flow matching (same)
- OpenVLA-OFT: token-by-token decoding with chunking
- The fact that SmolVLA and pi0 use identical action representation
  means cross-model transfer of action chunks is theoretically possible.
  This is a new angle if the data efficiency results don't pan out.

---

## What Cloud GPU Does NOT Change

Be explicit about constraints that remain:

1. **SigLIP frozen in SmolVLA** — cloud GPU cannot unfreeze it without full retraining from scratch on massive data. Not a cloud GPU problem.

2. **RoArm-M3 deployment latency** — inference must run locally or via low-latency API for real-time use. Cloud inference only viable for 4-chunk open-loop (200ms latency okay). For future closed-loop work, local inference remains required.

3. **Camera shift problem** — Azure Kinect was remounted, SSIM=0.49. Existing 74ep data is OOD for any new model trained from scratch. All models need fresh data collection. Cloud GPU doesn't fix this.

4. **Data collection bottleneck** — collecting 100ep takes ~3-4 hours of human labor regardless of cloud GPU. The AR-guided collection method is still the primary bottleneck.

5. **URDF quality for Isaac Lab** — sim-to-real from Isaac Lab still limited by SigLIP cosine distance 0.6-0.8 for rasterized rendering. Cloud GPU doesn't fix the visual domain gap.

---

## Updated Research Positioning (Cloud GPU Incorporated)

### Thesis of the paper (unchanged core, extended scope)

**Problem**: VLA adaptation to new embodiments requires hundreds of manual demonstrations.
Existing solutions (fleet collection, sim-to-real) require large infrastructure.

**Our constraint**: $130 robot, single GPU for deployment, single researcher.

**Contribution 1 (unchanged)**: AR-guided collection enforces coverage → better data with same N.
**Contribution 2 (unchanged)**: Quality Oracle filters bad episodes in real-time.
**Contribution 3 (NEW, enabled by cloud GPU)**: Method is model-agnostic.
Validated on SmolVLA (450M), pi0 (3B), OpenVLA-OFT (7B) — same improvement across all.

This is a significantly stronger paper than SmolVLA-only.
The method claim changes from "works for SmolVLA" to "works for any VLA."

### Positioning vs competitors
- ARMADA (Apple Vision Pro + cloud): requires $3500 headset. We use $130 robot + Unity app.
- XRoboToolkit (ByteDance XR teleop): teleop-based, we use guidance.
- MimicGen (NeurIPS 2024): sim-only, frozen VLM constraint absent. We handle real + frozen VLM.
- AutoRT (Google): fleet of 20 robots. We use 1-3 robots.

---

## Cost Estimate for Cloud GPU (VAST.ai Pricing, 2026)

| Experiment | GPU | Est. hours | Cost |
|------------|-----|-----------|------|
| pi0-fast compat test | A100 40GB | 0.5 | $1 |
| pi0-fast 200K steps | A100 40GB | 10 | $15-20 |
| pi0 data sweep (5 sizes) | A100 40GB | 50 | $75-100 |
| OpenVLA-OFT 100K steps | A100 80GB | 15 | $40-55 |
| Octo comparison | A100 40GB | 4 | $6-8 |
| **Total estimate** | | ~80 hrs | **~$140-190** |

Reasonable research budget. Single month of usage.

---

## For Pipeline-Agent: Training Config Recommendations

If pi0-fast is lerobot-train compatible, recommended config for comparison:

```
Model: lerobot/pi0fast (from HuggingFace)
Steps: 100K (half of SmolVLA 200K — pi0 richer pretraining needs fewer steps)
Batch size: 32-64 (A100 40GB)
LR: 1e-4 with cosine decay (pi0 paper recommendation)
Warmup: 1000 steps
Task prompt: "Pick up the sponge\n" (same as SmolVLA — test newline requirement)
n_action_steps: 50 (match SmolVLA for fair comparison)
```

If pi0 converges faster than SmolVLA on same data, that's a finding.
If it converges slower (richer model needs more data to adapt), also a finding.
Both results are publishable.

---

## Files Created
- `/home/cgxr/Documents/Robotics/RoArm_Project/model_cloud_gpu_unlocked_analysis.md` (this file)
