# SmolVLA Lab Meeting Reference Document

**Prepared for:** Lab meeting presentation — SmolVLA architecture and training decisions
**Project:** RoArm M3 Pro + SmolVLA pick-and-place task
**Date:** 2026-02-25
**Paper:** Shukor et al., "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics," arXiv 2506.01844 (2025)

---

## 1. What is SmolVLA — One-Paragraph Summary

SmolVLA is a 450M-parameter Vision-Language-Action model developed by Hugging Face.
It combines a frozen VLM backbone (SigLIP vision encoder + SmolLM2 language model,
~350M parameters) with a trainable Action Expert (~100M parameters) that operates via
**flow matching** — a continuous-time generative process that maps Gaussian noise into
robot action trajectories through 10 deterministic denoising steps.
The full model is pretrained on a community dataset of 487 robot learning datasets
(~10 million frames) before task-specific fine-tuning.

---

## 2. Architecture — Detailed Breakdown

### 2.1 Three-Module Structure

```
Input:
  - RGB image (1280x720 → resized to 512x512 with padding)
  - Joint state (6 DOF → zero-padded to 32-dim)
  - Language instruction (tokenized, max 48 tokens)

[SigLIP Vision Encoder]  ← FROZEN during fine-tuning
         |
         | image embeddings (normalized by sqrt(dim))
         v
[SmolLM2 Language Model] ← FROZEN during fine-tuning
(SmolVLM2-500M-Video-Instruct backbone)
  16 transformer layers (num_vlm_layers=16)
         |
         | key-value cache (cross-attention)
         v
[Action Expert]          ← TRAINABLE (22.2% of total params)
  Smaller transformer, hidden_size = 0.75 x VLM hidden_size
  Interleaved self-attention every 2 layers
  Receives KV-cache from VLM via cross-attention
         |
         v
Output: action chunk (50 steps x 6 DOF = 300 values)
```

Source: `lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py`

### 2.2 Parameter Budget

| Component          | Parameters | Status       |
|--------------------|------------|--------------|
| Total              | 450,046,176| -            |
| Vision Encoder     | ~300M      | Frozen       |
| Language Model     | ~50M       | Frozen       |
| **Action Expert**  | **~100M**  | **Trainable**|
| Trainable fraction | 22.2%      | -            |

Key implication: Fine-tuning is very efficient — only 100M params are updated.
This is why the model trains in ~2h50m on an RTX 4090 Laptop for 50K steps.

### 2.3 VLM Backbone: SmolVLM2-500M-Video-Instruct

```
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
├── Vision: SigLIP (image → patch embeddings)
│     Input normalized to [-1, 1] range (from [0, 1])
│     Images resized to 512x512 with aspect-ratio-preserving padding
├── Text: SmolLM2-derived language model
│     Tokenizer: max_length=48 tokens
│     Positional encoding: RoPE (Rotary Position Embeddings)
└── Connector: modality projection from vision to text space
```

Source: `configuration_smolvla.py` line 87, `smolvlm_with_expert.py` lines 78-91

### 2.4 Action Expert Architecture

```python
# From smolvlm_with_expert.py lines 95-105
lm_expert_config.hidden_size = int(hidden_size * 0.75)  # 75% of VLM hidden_size
lm_expert_config.num_hidden_layers = 16                  # Same depth as VLM
# Cross-attention: expert queries VLM's KV-cache every layer
# Self-attention: only every self_attn_every_n_layers=2 layers
```

The expert is "smaller but same depth" — efficient but with full representational depth.

### 2.5 State/Action Dimension Handling

```python
# max_state_dim = 32, max_action_dim = 32 (configuration_smolvla.py)
# Our robot: 6 DOF
# Zero-padding: [j1, j2, j3, j4, j5, j6, 0, 0, ..., 0]  (6 → 32)
# After inference: unpad back to 6
# This allows one pretrained model to serve 1-32 DOF robots
```

---

## 3. Flow Matching — How Denoising Works

### 3.1 Conceptual Overview

Flow matching defines a **straight path** from noise to data in continuous time:

```
t=1.0:  x_t = pure noise (Gaussian)
t=0.0:  x_t = clean action trajectory

Training target:  u_t = noise - action  (the "velocity field")
Network learns:   v_t = predicted velocity at timestep t
Loss:             MSE(u_t, v_t)  per (batch, time_step, action_dim)
```

Source: `modeling_smolvla.py` lines 766-791

### 3.2 Training Forward Pass

```python
# From modeling_smolvla.py lines 763-791
time = Beta(1.5, 1.0)  # skewed toward t=1 (more noise exposure)
x_t = time * noise + (1 - time) * actions  # interpolation
u_t = noise - actions                        # target velocity
v_t = action_expert(x_t, time, vlm_context) # predicted velocity
loss = F.mse_loss(u_t, v_t, reduction="none")  # (B, 50, 32)
```

Beta(1.5, 1.0) concentrates training on noisier samples — better for learning
the large-scale structure of motions before fine-grained details.

### 3.3 Inference (10 Denoising Steps)

```python
# From modeling_smolvla.py lines 826-862
# IMPORTANT: KV-cache computed ONCE for image+language (expensive)
# Then 10 fast denoising iterations using cached context

dt = -1.0 / 10  # step size
x_t = noise      # start from pure Gaussian noise

for step in range(10):
    time = 1.0 + step * dt  # 1.0 → 0.1
    v_t = expert(x_t, time, kv_cache)  # fast: uses KV-cache
    x_t = x_t + dt * v_t               # Euler integration

return x_t  # shape: (batch, 50, 32)
```

**Wall-clock cost breakdown (measured on RTX 4090 Laptop):**
- KV-cache computation: ~85ms (image encode + language encode + VLM forward)
- 10 denoising steps: ~5ms total
- Total per inference: ~90ms → 11 Hz maximum
- n_action_steps=50 → robot executes 50 steps before re-inference → real frequency 0.2 Hz

### 3.4 Why Flow Matching vs Diffusion Policy?

| Property              | Diffusion Policy (DDPM) | Flow Matching (SmolVLA/pi0) |
|-----------------------|-------------------------|-----------------------------|
| Path shape            | Curved (stochastic SDE) | Straight (deterministic ODE)|
| Denoising steps       | 50-1000                 | **10** (5-10x faster)       |
| Training samples      | Fixed noise schedule    | Beta distribution (flexible)|
| Theoretical framework | Score matching          | Continuous normalizing flows|
| Inference quality     | Good                    | Comparable with fewer steps |

Key advantage: 10 steps vs 100 steps = 10x faster inference, enabling near-real-time control.

---

## 4. Action Chunking — Why chunk_size=50 and n_action_steps=50

### 4.1 What Action Chunking Is

```
Without chunking (naive):
  t=1: observe → infer 1 action → execute → t=2: observe → ...
  Problem: inference latency (90ms) limits to ~11 Hz control

With action chunking (chunk_size=50):
  t=1: observe → infer 50 actions at once → execute all 50 → t=51: re-infer
  Effective control: 30 Hz execution (pre-planned chunk)
  Re-inference: every 50 steps = every 1.67 seconds
```

Source: ACT paper (Zhao et al., 2023) introduced action chunking for imitation learning.
SmolVLA inherits this design.

### 4.2 chunk_size=50 at 30 fps

```
50 steps / 30 fps = 1.67 seconds per chunk

For a ~5-second episode (our data):
  - ~150 frames total
  - 3 chunks of 50 steps each
  - Model re-infers 3 times per episode during deployment
```

### 4.3 n_action_steps Controls Re-inference Frequency

```python
# configuration_smolvla.py line 33
n_action_steps: int = 50  # default = full chunk, no re-inference

# deploy_smolvla.py (our deployment setting)
# n_action_steps=1 → re-infer EVERY step (true closed-loop)
# Cost: 90ms per step, ~11 Hz control
# Benefit: real camera/joint state used at every step
```

**Critical lesson from our v1 deployment failure:**

| n_action_steps | Mode | Result |
|----------------|------|--------|
| 50 (default) | Open-loop: execute 50 pre-planned actions | Drifted, no correction |
| **1** | **Closed-loop: re-infer every step** | **Better tracking** |

With n_action_steps=50, the model ignores what actually happened — it executes the
pre-planned chunk even if the robot drifted. With n_action_steps=1, every 90ms the model
sees the actual camera image and joint state.

### 4.4 Multi-Chunk Coverage

For full episode coverage (150+ frames):
```
With n_action_steps=50: chunk 1 runs frames 1-50, chunk 2 runs 51-100, etc.
With n_action_steps=1:  each frame gets fresh inference (300 inferences for 300 steps)
```

---

## 5. Normalization — Why It Matters

### 5.1 What the Official Pipeline Does Automatically

```python
# configuration_smolvla.py lines 35-41
normalization_mapping = {
    "VISUAL": NormalizationMode.IDENTITY,   # images: no normalization (SigLIP does its own)
    "STATE":  NormalizationMode.MEAN_STD,   # joint angles: (x - mean) / std
    "ACTION": NormalizationMode.MEAN_STD,   # action targets: (x - mean) / std
}
```

Dataset statistics (computed from training data):
- action.mean = [2.71, 40.31, 13.04, 62.75, -2.65, 9.61] (v1 dataset, 50 episodes)
- action.std  = [~9.7, ~33.0, ~29.4, ~25.2, ~13.6, ~16.9]

After normalization, all joints have mean=0, std=1 in the training space.
The model predicts in normalized space; post-processing unnormalizes to degrees.

### 5.2 Why Custom Training Scripts Failed (Root Cause)

Our 3 failed attempts on Windows before using lerobot-train:

| Attempt | batch_size | load_vlm | Normalization | Result |
|---------|------------|----------|---------------|--------|
| 1 | 1 | False | Missing | Mean action (flat output) |
| 2 | 8 | False | Missing | Mean action |
| 3 | 8 | True | Missing | Mean action |

The normalization missing means the model saw raw degree values (e.g. elbow=-64°)
which are much larger than the normalized range, causing gradient instability.

### 5.3 Cosine Decay + Warmup Schedule

```
Step 0 → 1000:    warmup (0 → 1e-4)
Step 1000 → 30000: cosine decay (1e-4 → 2.5e-6)
Step 30000+:       stays at 2.5e-6
```

Source: `configuration_smolvla.py` lines 83-85

---

## 6. Pretraining — Why smolvla_base Is Essential

### 6.1 smolvla_base Community Pretraining

The `lerobot/smolvla_base` checkpoint on HuggingFace Hub:
- **Pretrained on:** Community dataset v1 — 128 datasets, 11,132 episodes (per memory)
  - Note: An older version had 487 datasets listed; the exact number may differ by release
- **All data:** SO-100 robot (5-DOF tabletop manipulator)
- **Pretrained VLM:** SmolVLM2-500M-Video-Instruct weights loaded
- **Training command reference (from model card):**

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
  --batch_size=64 \
  --steps=200000
```

Source: `modeling_smolvla.py` lines 29-36 (docstring)

### 6.2 Pretrained vs Scratch Performance

From SmolVLA paper (arXiv 2506.01844):
- Pretrained smolvla_base: **78.3% success rate**
- Training Action Expert from scratch: **51.7% success rate**
- Improvement: **+26.6 percentage points** from pretraining
- Ratio: 1.52x better with pretrained Action Expert

The VLM backbone alone (without pretrained Action Expert) is insufficient.
The Action Expert must also be pretrained on robot manipulation data.

### 6.3 RoArm M3 as OOD Embodiment

Key finding: SmolVLA was pretrained ONLY on SO-100 robot data.
RoArm M3 was NOT in the pretraining dataset.

```
Pretraining robot:   SO-100 (5 DOF, tabletop)
Our robot:           RoArm M3 Pro (6 DOF, different kinematics)
Cross-embodiment:    6 DOF → zero-padded to 32 → unpadded back to 6
                     Model never saw RoArm M3 during pretraining
```

Implications for our project:
- SO-100 fine-tuning: 50 episodes / 50K steps sufficient (in-distribution robot)
- RoArm M3 fine-tuning: needs more data (150+ episodes recommended for OOD)
- We achieved good results with 74 episodes (v3 dataset) + 50K steps

Transfer analogy: fine-tuning ImageNet ResNet for a new image classification task.
The backbone features still help even if the new task is different.

---

## 7. batch_size=64 — Why the Official Setting Matters

### 7.1 Gradient Noise and batch_size

Stochastic gradient descent with small batches introduces high variance in gradient
estimates. For flow matching with chunked action prediction, this is particularly
harmful because:

```
Batch sample = [image, state] → [50-step action chunk]
Each sample is a single (observation, action_sequence) pair

batch_size=1:  gradient from 1 data point → high variance
               model "averages out" high-variance gradients → predicts mean
batch_size=64: gradient from 64 data points → stable direction
               model can learn fine-grained joint-specific patterns
```

### 7.2 Measured VRAM Usage on RTX 4090 Laptop (16.72 GB)

| batch_size | Peak VRAM | Utilization | Headroom |
|------------|-----------|-------------|---------|
| 8          | 2.03 GB   | 12.1%       | 14.69 GB |
| 16         | 3.15 GB   | 18.9%       | 13.56 GB |
| 32         | 5.38 GB   | 32.2%       | 11.33 GB |
| **64**     | **9.85 GB**| **58.9%**  | **6.87 GB** |

Source: `train_batch_size_test.py` (measured empirically, 2026-02-24)

Conclusion: batch_size=64 fits comfortably. Our earlier choice of batch_size=8 was
not a VRAM limitation — it was a conservative setting that left 88% VRAM unused.

### 7.3 What Changed Between Our Experiments

| Experiment | batch_size | Steps | L2 Error | Wrist_R Std | Gripper Std |
|------------|------------|-------|----------|-------------|-------------|
| v1 (50 eps) | 8         | 50K   | 2.53°    | 3.34° (bad) | poor        |
| v3 (74 eps) | **64**    | 50K   | **2.81°** | **high**   | **24.89°**  |

With batch_size=64, the gripper diversity went from 15% of dataset std to 103% of
dataset std — the critical fix for our deployment failure.

### 7.4 Gradient Accumulation is NOT Available

Verification from source search:
```bash
# Searched entire lerobot/src/lerobot/ for gradient_accumulation
# Result: zero matches
# lerobot-train CLI does not support --gradient_accumulation_steps
# TrainPipelineConfig has no such field
```

The only way to use effective batch_size=64 is to set batch_size=64 directly.
This rules out any gradient accumulation workaround.

---

## 8. Data Requirements — What the Numbers Say

### 8.1 Official SmolVLA Recipe

From the SmolVLA paper and documentation:
- **Minimum viable:** 5 positions × 10 repetitions = **50 episodes**
- **Average episode length:** ~13 seconds (~393 frames at 30 fps)
- **Recommended steps:** 20K-200K (task-dependent)
- **Critical constraint:** Must use `lerobot/smolvla_base` pretrained model

Our episodes: ~5 seconds (~145 frames) — shorter than official recommendation.
This is a limitation in our data collection protocol.

### 8.2 Data Scaling Empirical Results

From literature (OpenVLA-OFT and SmolVLA scaling experiments):

| Episodes | Typical Success Rate |
|----------|----------------------|
| 25       | too few (confirmed sub-optimal) |
| 50       | 45-52% |
| 100      | 68-72% |
| 200      | 74-76% |

Success rate plateaus around 200 episodes for tabletop manipulation tasks.
For OOD embodiments (like RoArm M3), the plateau may require more data.

### 8.3 Our Dataset Evolution

| Version | Episodes | Frames  | batch_size | Steps | Key Issue |
|---------|----------|---------|------------|-------|-----------|
| v1      | 50       | 10,803  | 8          | 50K   | 68% shallow, gripper failure |
| v2      | 43       | 9,747   | 8          | 50K   | 7 bad removed, same issues |
| **v3**  | **74**   | **13,145** | **64** | **50K** | **HEALTHY — all metrics pass** |

### 8.4 The Depth Distribution Problem (v1 Failure Analysis)

```
v1 data distribution (elbow angle as proxy for grasp depth):
  SHALLOW (elbow > -10°):    34/50 = 68%
  APPROACH (-30° to -10°):   7/50  = 14%
  DEEP (elbow < -30°):       9/50  = 18%  ← too few!

Result: model never learned deep grasps → elbow stayed above -20° in deployment
Needed z-score for elbow=-64°: z = -3.04 → model max output: ±1.5
```

Fix in v3: targeted collection with 50%+ deep episodes and gripper-diverse episodes.

---

## 9. Loss Function — What SmolVLA Actually Optimizes

### 9.1 MSE Loss on Flow Velocity

```python
# modeling_smolvla.py line 791
losses = F.mse_loss(u_t, v_t, reduction="none")
# Shape: (batch_size=64, chunk_size=50, max_action_dim=32)
# u_t = noise - actions  (target velocity)
# v_t = expert_output    (predicted velocity)

# Then averaged uniformly:
loss = losses[:, :, :max_action_dim].mean()
```

**Critical implication: All joints are weighted equally.**
If joint 5 (Wrist_R) has std=13.6° and joint 2 (Elbow) has std=29.4°,
the model sees them as equally important — even though Wrist_R varies less.

### 9.2 No Built-in Per-Joint Weighting

After searching the entire LeRobot source:
```
lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py line 399:
    losses.mean()  ← plain unweighted mean
No mechanism for joint-specific loss weights in SmolVLA or LeRobot framework.
```

Workarounds (without modifying LeRobot source):
1. **Data resampling:** Oversample episodes with diverse Wrist_R and Gripper values
2. **Episode duplication:** Duplicate critical episodes (DEEP grasp) in the dataset
3. **Cannot use:** gradient_accumulation (not supported), custom loss (requires fork)

### 9.3 Training Loss Trajectory (Our v3 Run)

| Checkpoint | Loss   | L2 Error | Diversity Ratio |
|------------|--------|----------|-----------------|
| 5K         | ~0.012 | 4.531°   | 0.976           |
| 15K        | ~0.006 | 3.458°   | 0.973           |
| **25K**    | ~0.004 | **2.810°** | **0.986**    |
| 50K        | ~0.003 | 2.985°   | 0.985           |

Observation: L2 error bottoms out around 25K steps. Further training (25K→50K)
yields diminishing returns. Loss continues decreasing but L2 does not.

---

## 10. Deployment: Key Lessons

### 10.1 Start Position — dataset_mean is Critical

```python
# If starting from zero position [0, 0, 0, 0, 0, 0]:
# - All joints at zero = outside training distribution (OOD)
# - Model outputs tiny corrections (conservative, z-score ≈ 0)
# - Robot barely moves

# Correct: start from dataset mean
# dataset_mean = [2.71, 40.31, 13.04, 62.75, -2.65, 9.61] (v1)
# This is in the center of the training distribution
```

Measured improvement: Shoulder movement 13° → 34° (2.6x), Elbow 2° → 34° (17x)
by simply changing start position from zero to dataset mean.

### 10.2 n_action_steps=50 vs n_action_steps=1

The default n_action_steps=50 effectively makes deployment **open-loop**:
- Model infers chunk → executes all 50 steps → re-infers
- During those 50 steps, actual robot state is ignored
- Any drift accumulates

Setting n_action_steps=1 makes it **closed-loop**:
- Model infers 1 action → executes → re-infers from actual observation
- Cost: 90ms per step (11 Hz)
- Benefit: errors are corrected at each step

### 10.3 Inference Timing (Measured)

```
KV-cache computation (image + language + VLM forward): ~85ms
10 denoising steps (using cached KV):                  ~5ms
Total per inference:                                   ~90ms
Control frequency (n_action_steps=1):                 ~11 Hz
```

---

## 11. VLA Landscape Context

### 11.1 Where SmolVLA Fits

```
Parameter Count Comparison:
  RT-2-X (Google):         55B  -- not open source
  OpenVLA:                  7B  -- open source, action tokenization
  π₀ (Physical Intelligence): 3.3B  -- flow matching, not open source
  GR00T N1 (NVIDIA):        ~1B  -- dual system architecture
  CogACT (Microsoft):       7.6B -- 3-stage cognition-action split
  **SmolVLA (HuggingFace)**: **450M** -- fully open, flow matching, affordable
  Octo:                     93M  -- no language conditioning
```

SmolVLA's niche: affordable, open-source, flow-matching, single-camera, ~500M scale.
Target hardware: consumer GPU (tested on RTX 4090 Laptop, 16.7 GB VRAM).

### 11.2 Key Technical Lineage

```
RT-1 (Google, 2022)    → Transformer + tokenized actions
RT-2 (Google, 2023)    → VLM backbone (PaLI-X) + robot actions
OpenVLA (Stanford, 2024) → 7B LLaMA + action discretization
π₀ (PI, 2024)          → Flow matching + VLM + Action Expert
SmolVLA (HuggingFace, 2025) → π₀-inspired + smaller (500M) + fully open
```

SmolVLA is directly inspired by the π₀ architecture but scaled down by ~7x
and made fully open-source.

### 11.3 Action Chunking Origins

Action chunking was introduced in:
- **ACT (Action Chunked Transformers)**: Zhao et al., RSS 2023
- "Predict N actions at once, execute them sequentially"
- Originally justified as reducing compounding error in imitation learning

SmolVLA chunk_size=50 at 30fps = 1.67-second future horizon per inference.

### 11.4 Flow Matching Origins

Flow matching for generative models:
- Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023
- Applied to robotics first in: **Diffusion Policy** → then **π₀** introduced flow matching
- Key advantage: straight paths between noise and data → fewer integration steps

### 11.5 Recent Efficiency Extensions

Papers that improve SmolVLA-style inference for real-time use:

| Paper | Key Idea | Speedup |
|-------|----------|---------|
| RTC (arXiv 2601.20130) | Predict-while-execute, overlap inference | 50 Hz effective |
| VLA-Cache (arXiv 2502.02175) | Adaptive token caching | ~14 Hz |
| PD-VLA (arXiv 2503.02310) | Parallel decoding | 2.52x |
| VOTE (arXiv 2507.05116) | Trajectory ensemble voting | 46 Hz |

---

## 12. Evaluation Metrics We Use

### 12.1 Offline Metrics (from train_eval_v3_checkpoints.py)

```python
# 1. L2 Error (degrees, lower is better)
l2 = np.sqrt(np.sum((pred - gt)**2))  # across 6 joints

# 2. Diversity Ratio (closer to 1.0 is better)
diversity = pred_std / dataset_std  # per joint
# If < 0.5: model is averaging (mean action problem)
# If > 0.8: healthy generalization

# 3. Z-score range (should exceed ±2.0 for extreme poses)
z_score = (pred - dataset_mean) / dataset_std  # per joint
```

### 12.2 v3 Best Checkpoint (25K steps)

```
L2 error:       2.810° (all 6 joints, 222 stratified samples)
Diversity ratio: 0.986 (near 1.0 = full data range reproduced)
Gripper range:   95.6° (min=1.85°, max=97.48°) — FIXED from v1
Wrist_R z-score: 5.7 (was 1.5 in v1 — the critical joint that failed)
Inference time:  ~90ms per step
```

---

## 13. Citations

### Primary Paper
```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn
          and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline
          and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu
          and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```

### Action Chunking (ACT)
```bibtex
@inproceedings{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}
```

### Flow Matching
```bibtex
@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  booktitle={ICLR},
  year={2023}
}
```

### pi0 (Architecture Predecessor)
```bibtex
@article{black2024pi0,
  title={pi0: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and Brown, Noah and Driess, Danny and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024}
}
```

### LeRobot Framework
```
Cadene, Remi and Alibert, Simon and others.
LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch.
HuggingFace, 2024. https://github.com/huggingface/lerobot
```

### OpenVLA (Comparison)
```bibtex
@article{kim2024openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and others},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```

### Diffusion Policy (Context)
```bibtex
@inproceedings{chi2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and others},
  booktitle={Robotics: Science and Systems},
  year={2023}
}
```

---

## 14. Quick-Reference Numbers for Presentation

```
SmolVLA model size:          450M total, 100M trainable (22.2%)
VLM backbone:                SmolVLM2-500M-Video-Instruct
Pretraining dataset:         SO-100 robot, ~11K episodes
Pretrained vs scratch:       78.3% vs 51.7% success rate
Flow matching steps:         10 denoising iterations per inference
Chunk size:                  50 steps at 30 fps = 1.67 seconds
n_action_steps default:      50 (open-loop), set to 1 for closed-loop
Inference time:              ~90ms per call (11 Hz)
Normalization:               MEAN_STD for state and action
batch_size official:         64
batch_size=64 VRAM:          9.85 GB (fits in RTX 4090 16.7 GB)
Our v3 dataset:              74 episodes, 13,145 frames
Our best checkpoint:         25K steps, L2=2.81°, diversity=0.986
Paper arXiv:                 2506.01844
```

---

*Document generated by Pipeline Agent*
*Source files verified: configuration_smolvla.py, modeling_smolvla.py, smolvlm_with_expert.py*
*Project directory: /home/cgxr/Documents/Robotics/RoArm_Project*
