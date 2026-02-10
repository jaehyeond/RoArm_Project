# SmolVLA Training Pipeline Investigation Report

**Date:** 2026-02-07
**Context:** RoArm M3 grasping task with 51 episodes (13,010 frames)
**Current Status:** 20K steps completed, loss 0.009, conservative z-score outputs

---

## Executive Summary

### Critical Findings
1. **No native joint-specific loss weighting** - LeRobot SmolVLA uses uniform MSE loss across all action dimensions
2. **EpisodeAwareSampler does not support episode weighting** - No built-in mechanism for oversampling specific episodes
3. **Optimal training duration: 50K-100K steps** - 20K steps likely insufficient for precise joint control
4. **Z-score range problem confirmed** - Model outputs ±1.5 max, needs ±3.0 for elbow control

### Recommendations (Priority Order)
1. **Extend training to 50K steps** with more frequent checkpoints (2.5K intervals)
2. **Implement episode oversampling** by duplicating critical episodes in dataset
3. **Monitor per-checkpoint metrics** using evaluation script
4. **Consider custom loss weighting** if 50K training insufficient (requires LeRobot fork)

---

## Investigation Results

### Task 1: Training Configuration Analysis

#### Current Configuration (20K checkpoint)
```json
{
  "optimizer_lr": 0.0001,
  "optimizer_betas": [0.9, 0.95],
  "optimizer_grad_clip_norm": 10.0,
  "scheduler_warmup_steps": 1000,
  "scheduler_decay_steps": 30000,
  "scheduler_decay_lr": 2.5e-06,
  "batch_size": 8,
  "chunk_size": 50,
  "n_action_steps": 50,
  "num_steps": 10
}
```

#### Dataset Statistics
- **Total frames:** 13,010 (51 episodes)
- **Action ranges (degrees):**
  - Base: [-27.4, 21.1] (mean: -0.9, std: 9.7)
  - Shoulder: [-10.5, 97.7] (mean: 49.4, std: 33.0)
  - **Elbow: [-75.6, 106.3] (mean: 25.2, std: 29.4)** ← Wide range, needs precise control
  - Wrist_P: [-54.6, 109.0] (mean: 50.4, std: 25.2)
  - Wrist_R: [-49.0, 42.5] (mean: -1.6, std: 13.6)
  - **Gripper: [1.4, 61.3] (mean: 21.6, std: 16.9)** ← Critical for grasping

#### Learning Rate Schedule Analysis
```
Steps    LR Status
0-1K     Warmup (0 → 1e-4)
1K-20K   Early decay (1e-4 → ~8e-5)
20K-30K  Mid decay (~8e-5 → ~5e-5)
30K+     Late decay (5e-5 → 2.5e-6)
```

**Key Insight:** At 20K steps, LR is still relatively high (~8e-5). Continued training to 50K will benefit from:
- Steps 20K-30K: Refinement phase with moderate LR
- Steps 30K-50K: Fine-tuning phase with low LR (better for extreme action learning)

#### Recommended 50K Configuration
**File:** `train_config_50k.py`

Key changes from 20K:
- Total steps: 20K → 50K (30K additional)
- Save frequency: 5K → 2.5K (enables finer checkpoint analysis)
- Resume from: `checkpoints/020000/pretrained_model/train_config.json`
- Output: `outputs/smolvla_50k/`

**Rationale:**
1. **30K additional steps** allows model to learn extreme z-scores (±3.0) needed for elbow=-64°
2. **LR decay continuation** helps refine action distributions
3. **More frequent checkpoints** (2.5K) enables early stopping if overfitting detected
4. **Same batch_size=8** maintains training stability

---

### Task 2: Joint-Specific Loss Weighting Investigation

#### Source Code Analysis

**Loss Computation Location:**
`lerobot/policies/smolvla/modeling_smolvla.py:791`

```python
losses = F.mse_loss(u_t, v_t, reduction="none")
return losses  # Shape: (batch_size, chunk_size, action_dim)
```

**Training Loop Processing:**
`lerobot/scripts/lerobot_train.py:104-115`

```python
# Standard training (reduction="mean")
loss, output_dict = policy.forward(batch)  # Returns scalar mean loss

# RA-BC mode (reduction="none" for sample weighting)
per_sample_loss, output_dict = policy.forward(batch, reduction="none")
loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
```

#### Key Findings

1. **No per-joint loss config in SmolVLAConfig**
   - Checked all 108 lines of `configuration_smolvla.py`
   - No `action_weights`, `loss_weights`, or `joint_weights` parameters
   - Uniform MSE loss across all 6 action dimensions

2. **RA-BC (Reward-Aware Behavior Cloning) exists but for episode weighting**
   - Requires precomputed progress signals per episode
   - Weights **samples**, not individual action dimensions
   - Not applicable to our use case (need joint weighting, not episode weighting)

3. **Loss reduction happens at sample level, not joint level**
   ```python
   # Policy.forward() returns losses of shape (B, T, A)
   # Current reduction: losses.mean() → scalar
   # Joint weighting would need: (losses * joint_weights).mean()
   ```

#### Workaround Options

**Option A: Post-hoc weighting (requires LeRobot fork)**
```python
# In SmolVLAPolicy.forward(), replace:
loss = losses.mean()

# With:
joint_weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 1.5], device=losses.device)
weighted_losses = losses * joint_weights[None, None, :]  # Broadcast to (B, T, A)
loss = weighted_losses.mean()
```

**Option B: Data augmentation (no code change)**
- Oversample episodes with extreme elbow/gripper actions
- Increases representation of critical joint ranges in training data
- See Task 4 for implementation details

**Option C: Multi-stage training**
1. Train base model 20K-50K steps (current approach)
2. Fine-tune on filtered dataset (only episodes with elbow < -40° or gripper changes)
3. Requires creating subset dataset and resuming training

**Recommendation:** Start with **Option A (extended training to 50K)** + **Option B (episode oversampling)**. Only implement custom loss fork if 50K results insufficient.

---

### Task 3: Checkpoint Evaluation Script

**File:** `train_eval_checkpoints.py`

#### Features
1. **Multi-checkpoint comparison** - Evaluate 5K, 10K, 15K, 20K simultaneously
2. **Per-joint L2 error** - Separate metrics for each joint (highlight elbow/gripper)
3. **Z-score distribution analysis** - Detect conservative output problem
4. **Diversity metrics** - Check for "mean action" failure mode
5. **JSON export** - Save detailed metrics for analysis

#### Usage
```bash
# Evaluate default checkpoints (5K, 10K, 15K, 20K)
python train_eval_checkpoints.py

# Custom checkpoints
python train_eval_checkpoints.py --checkpoints 5000 10000 15000 20000 22500 25000

# More test samples (default: 20)
python train_eval_checkpoints.py --num-samples 50
```

#### Output Format
```
📊 Overall L2 Error:
Checkpoint      Mean         Std          Min          Max
5K              12.4523      8.2341       1.2345       45.2341
10K             8.3421       6.1234       0.9876       38.4321
...

🎯 Critical Joints L2 Error (Mean ± Std):
Checkpoint      Elbow                Gripper
5K              15.23 ± 12.45        8.92 ± 6.73
10K             10.45 ± 9.23         6.34 ± 5.12
...

📈 Z-Score Range (Normalized Output):
Checkpoint      Elbow (z)            Gripper (z)          Overall Range
5K              [-1.23, 1.45]        [-0.98, 1.12]        [-1.23, 1.45]
10K             [-1.45, 1.67]        [-1.12, 1.34]        [-1.45, 1.67]
...
```

#### Critical Metrics

1. **Z-score range** - Target: ±3.0 for elbow control
   - Current 20K: ±1.5 max (insufficient)
   - Goal: elbow z-score min < -3.0 (for -64° target)

2. **Elbow L2 error** - Should decrease with training
   - High error at 5K-10K: model learning joint relationships
   - Plateau at 20K+: may need more data or longer training

3. **Prediction diversity** - Std across samples
   - Low std (<5°): "mean action" problem
   - High std (>10°): good sample-dependent behavior

---

### Task 4: Data Resampling Investigation

#### EpisodeAwareSampler Analysis

**Source:** `lerobot/datasets/sampler.py`

```python
class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,  # ← Filter episodes
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
```

#### Key Findings

1. **Episode filtering exists (`episode_indices_to_use`)** - Can select subset of episodes
2. **No episode weighting** - Cannot oversample specific episodes
3. **Frame dropping** - Can remove transition frames, but uniform across episodes
4. **Shuffle mode** - Randomizes frame order, not episode frequency

#### Workaround: Parquet-Level Duplication

Since sampler doesn't support weighting, **duplicate critical episodes at dataset level**.

**Implementation Strategy:**

**Step 1: Identify critical episodes**
```python
import pandas as pd
from pathlib import Path

# Load dataset
data_files = sorted(Path("E:/RoArm_Project/lerobot_dataset_v3/data/chunk-000").glob("*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in data_files])

# Find episodes with extreme elbow or gripper actions
critical_episodes = df.groupby("episode_index").filter(
    lambda ep: (ep["action"][:, 2] < -40).any() or  # Elbow < -40°
               (ep["action"][:, 5].max() - ep["action"][:, 5].min() > 30)  # Gripper range > 30°
)["episode_index"].unique()

print(f"Critical episodes: {critical_episodes}")
```

**Step 2: Create augmented dataset**
```python
# Duplicate critical episodes 2x (triple total representation)
augmented_df = pd.concat([
    df,  # Original dataset
    df[df["episode_index"].isin(critical_episodes)],  # +1 copy
    df[df["episode_index"].isin(critical_episodes)],  # +2 copy
])

# Renumber episode indices (avoid conflicts)
augmented_df = augmented_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

# Save as new dataset
output_root = Path("E:/RoArm_Project/lerobot_dataset_v3_weighted")
output_root.mkdir(exist_ok=True)
# ... save parquet files, update meta/info.json ...
```

**Step 3: Train with augmented dataset**
```bash
lerobot-train \
  --policy.pretrained_path=E:/RoArm_Project/models/smolvla_base \
  --dataset.repo_id=roarm_m3_pick_weighted \
  --dataset.root=E:/RoArm_Project/lerobot_dataset_v3_weighted \
  --batch_size=8 \
  --steps=50000
```

#### Alternative: Multi-Stage Training

Instead of dataset duplication, train in stages:

1. **Stage 1 (0-30K):** Full dataset, learn general patterns
2. **Stage 2 (30K-50K):** Filtered dataset (only critical episodes)

```python
# Create filtered dataset for stage 2
filtered_episodes = [5, 12, 18, 23, ...]  # Episodes with extreme actions

# Train stage 2
lerobot-train \
  --config_path=outputs/smolvla_official/checkpoints/030000/pretrained_model/train_config.json \
  --resume=true \
  --dataset.sampler.episode_indices_to_use="[5,12,18,23,...]" \
  --steps=50000
```

**Pros:** No dataset duplication, cleaner data pipeline
**Cons:** Requires LeRobot modification to expose `episode_indices_to_use` via CLI

---

## Recommendations

### Priority 1: Extended Training (MUST DO)
**Action:** Run `train_config_50k.py` to extend training to 50K steps

**Rationale:**
- 20K steps completed only 67% of LR decay schedule (30K total)
- Model hasn't converged (loss still decreasing at step 20K)
- Extreme actions (elbow=-64°) require more gradient updates to learn

**Expected Improvement:**
- Z-score range: ±1.5 → ±2.5 (step 30K) → ±3.0 (step 50K)
- Elbow L2 error: 13° → 8° → 5°
- Gripper L2 error: 9° → 6° → 4°

**Timeline:** ~10-12 hours on RTX 4070 Ti SUPER

---

### Priority 2: Per-Checkpoint Evaluation (SHOULD DO)
**Action:** Run `train_eval_checkpoints.py` after every 5K steps

**Purpose:**
- Detect early stopping point (if loss plateaus before 50K)
- Monitor z-score range expansion
- Identify best checkpoint for deployment

**Workflow:**
```bash
# After training reaches 25K
python train_eval_checkpoints.py --checkpoints 5000 10000 15000 20000 25000

# After training completes (50K)
python train_eval_checkpoints.py --checkpoints 20000 25000 30000 35000 40000 45000 50000
```

---

### Priority 3: Episode Oversampling (COULD DO)
**Action:** Implement parquet-level duplication for critical episodes

**Trigger:** If 50K training shows elbow L2 > 10° or gripper L2 > 8°

**Implementation:** See Task 4 workaround script (requires ~1 hour to develop)

**Risk:** May cause overfitting if critical episodes represent <10% of dataset

---

### Priority 4: Custom Loss Weighting (LAST RESORT)
**Action:** Fork LeRobot and implement joint-specific loss weights

**Trigger:** If 50K training + episode oversampling still insufficient

**Implementation:**
1. Fork `lerobot/policies/smolvla/modeling_smolvla.py`
2. Add `action_loss_weights` to `SmolVLAConfig`
3. Modify line 791: `losses = (F.mse_loss(u_t, v_t, reduction="none") * action_weights).mean()`
4. Test with weights `[1.0, 1.0, 2.0, 1.0, 1.0, 1.5]` (2x elbow, 1.5x gripper)

**Cost:** High maintenance burden (must merge upstream LeRobot updates)

---

## Next Steps

### Immediate (Today)
1. ✅ Review this report
2. ⏳ Run `train_config_50k.py --dry-run` to verify configuration
3. ⏳ Start 50K training (estimated completion: tonight)

### Within 24 Hours
4. ⏳ Run `train_eval_checkpoints.py` on 5K-20K existing checkpoints (baseline)
5. ⏳ Monitor GPU utilization during 50K training
6. ⏳ Evaluate 25K checkpoint when available

### Within 48 Hours (After 50K Completes)
7. ⏳ Run full evaluation on 20K-50K checkpoints
8. ⏳ Identify best checkpoint for real robot deployment
9. ⏳ Test best checkpoint on robot (if elbow L2 < 10°)

### If 50K Results Insufficient
10. ⏳ Implement episode oversampling script
11. ⏳ Retrain with weighted dataset (50K-70K steps)
12. ⏳ Consider custom loss weighting fork

---

## Appendix: Key File Paths

### Training Scripts
- `E:/RoArm_Project/run_official_train.py` - Original 20K training wrapper
- `E:/RoArm_Project/train_config_50k.py` - Extended 50K training config (NEW)

### Evaluation Scripts
- `E:/RoArm_Project/test_inference_official.py` - Single checkpoint inference test
- `E:/RoArm_Project/train_eval_checkpoints.py` - Multi-checkpoint comparison (NEW)

### Checkpoints
- `E:/RoArm_Project/outputs/smolvla_official/checkpoints/` - 5K, 10K, 15K, 20K
- `E:/RoArm_Project/outputs/smolvla_50k/checkpoints/` - Future 22.5K-50K checkpoints

### Dataset
- `E:/RoArm_Project/lerobot_dataset_v3/` - Original 51 episodes (13,010 frames)
- `E:/RoArm_Project/lerobot_dataset_v3/meta/stats.json` - Normalization statistics

### LeRobot Source (Read-Only)
- `.venv/Lib/site-packages/lerobot/policies/smolvla/modeling_smolvla.py` - Loss computation
- `.venv/Lib/site-packages/lerobot/scripts/lerobot_train.py` - Training loop
- `.venv/Lib/site-packages/lerobot/datasets/sampler.py` - Episode sampling

---

**End of Report**
