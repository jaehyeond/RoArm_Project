# SmolVLA Training Guide - Quick Reference

## Current Status (2026-02-07)
- ✅ 20K steps completed (loss: 0.009)
- ⚠️ Model outputs conservative z-scores (±1.5 max, need ±3.0)
- 🎯 Goal: Extend to 50K steps with checkpoint evaluation

---

## 1. Extended Training (50K Steps)

### Dry Run (Check Configuration)
```powershell
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\train_config_50k.py --dry-run
```

**Expected Output:**
```
SmolVLA Extended Training Configuration: 50000 steps
Resuming from: E:/RoArm_Project/outputs/smolvla_official/checkpoints/020000/pretrained_model
Additional steps: 30000 (current: 20K → target: 50K)
Output: outputs/smolvla_50k
```

### Start Training (Actual GPU Training)
```powershell
# Default: 50K steps
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\train_config_50k.py

# Or 100K steps
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\train_config_50k.py --steps 100k
```

**Monitor Progress:**
- Loss logged every 100 steps (console output)
- Checkpoints saved every 2500 steps
- Watch GPU utilization: `nvidia-smi -l 1`

**Training Time:**
- 50K steps: ~10-12 hours (RTX 4070 Ti SUPER)
- 100K steps: ~20-24 hours

**Checkpoints Created:**
```
outputs/smolvla_50k/checkpoints/
├── 022500/  (2.5K after resume)
├── 025000/
├── 027500/
├── 030000/
...
├── 050000/
└── last  (text file pointing to latest)
```

---

## 2. Checkpoint Evaluation

### Evaluate Existing Checkpoints (5K-20K)
```powershell
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\train_eval_checkpoints.py
```

**Output:**
```
📊 Overall L2 Error:
Checkpoint      Mean         Std          Min          Max
5K              XX.XXXX      X.XXXX       X.XXXX       XX.XXXX
10K             XX.XXXX      X.XXXX       X.XXXX       XX.XXXX
...

🎯 Critical Joints L2 Error:
Checkpoint      Elbow (°)         Gripper (°)
5K              15.23 ± 12.45     8.92 ± 6.73
...

📈 Z-Score Range:
Checkpoint      Elbow (z)         Gripper (z)       Overall Range
5K              [-1.2, 1.4]       [-1.0, 1.1]       [-1.2, 1.4]
...

✨ Best checkpoint: 20K
⚠️ Z-score range < ±2.5 → Continue training to 50K
```

**Results saved to:** `train_checkpoint_eval_results.json`

### Evaluate Custom Checkpoints
```powershell
# After 50K training completes
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\train_eval_checkpoints.py `
    --checkpoints 20000 25000 30000 35000 40000 45000 50000 `
    --output-dir E:\RoArm_Project\outputs\smolvla_50k `
    --num-samples 30
```

### Key Metrics to Watch

| Metric | Target | Current (20K) | Interpretation |
|--------|--------|---------------|----------------|
| **Z-score range** | ±3.0 | ±1.5 | Need wider range for extreme actions |
| **Elbow L2 (°)** | <10 | ~13 | Precision for grasping (-64° target) |
| **Gripper L2 (°)** | <8 | ~9 | Open/close precision |
| **Pred diversity (°)** | >10 | ~5 | Sample-dependent behavior |

---

## 3. Deployment Decision Tree

### After 50K Training Completes

```
1. Run evaluation on all checkpoints (20K-50K)
   ↓
2. Check best checkpoint metrics
   ↓
3. Decision:
   ├─ Z-score ≥ ±2.5 AND Elbow L2 < 10°
   │  → ✅ DEPLOY to real robot
   │  → Use: deploy_smolvla.py --checkpoint best_checkpoint
   │
   ├─ Z-score ≥ ±2.0 BUT Elbow L2 > 10°
   │  → ⚠️ MARGINAL: Test on robot, may need episode oversampling
   │
   └─ Z-score < ±2.0
      → ❌ INSUFFICIENT: Implement episode oversampling + retrain
```

### Quick Deployment Test (Without Evaluation)
```powershell
# Test latest 50K checkpoint on real robot (5 steps)
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\deploy_smolvla.py `
    --checkpoint E:\RoArm_Project\outputs\smolvla_50k\checkpoints\050000\pretrained_model `
    --num-steps 5
```

---

## 4. Advanced: Episode Oversampling

### When to Use
- ✅ 50K training completed
- ❌ Elbow L2 still > 10° or Gripper L2 > 8°
- 🎯 Goal: Oversample episodes with extreme actions

### Step 1: Identify Critical Episodes
```python
import pandas as pd
from pathlib import Path

data_files = sorted(Path("E:/RoArm_Project/lerobot_dataset_v3/data/chunk-000").glob("*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in data_files])

# Find episodes with extreme elbow (<-40°) or large gripper changes (>30°)
critical_mask = (
    (df["action"].apply(lambda x: x[2]) < -40) |  # Elbow < -40°
    (df.groupby("episode_index")["action"].transform(
        lambda x: x.apply(lambda a: a[5]).max() - x.apply(lambda a: a[5]).min() > 30
    ))
)

critical_episodes = df[critical_mask]["episode_index"].unique()
print(f"Critical episodes ({len(critical_episodes)}/{df['episode_index'].nunique()}):")
print(critical_episodes)
```

### Step 2: Create Weighted Dataset
```python
# Duplicate critical episodes 2x (3x total representation)
augmented_df = pd.concat([
    df,  # Original
    df[df["episode_index"].isin(critical_episodes)],  # +1 copy
    df[df["episode_index"].isin(critical_episodes)],  # +2 copy
]).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

# Save to new directory
output_root = Path("E:/RoArm_Project/lerobot_dataset_v3_weighted")
# ... implement full dataset export ...
```

**Note:** Full implementation script not provided (requires ~1 hour to develop)

---

## 5. Monitoring Training

### GPU Utilization (Windows PowerShell)
```powershell
# Monitor GPU every second
nvidia-smi -l 1

# Watch for:
# - GPU Util: 90-100% (good)
# - GPU Memory: ~16GB/24GB (RTX 4070 Ti SUPER)
# - Temperature: <85°C (safe)
```

### Training Log Analysis
```powershell
# Follow training log in real-time
Get-Content E:\RoArm_Project\outputs\smolvla_50k\train.log -Wait -Tail 50

# Key lines to watch:
# Step 22500: loss=0.008, lr=7.5e-05
# Step 25000: loss=0.007, lr=6.8e-05
# ...
```

### Checkpoint Disk Usage
```powershell
# Each checkpoint: ~2.5GB
# 50K training (12 checkpoints): ~30GB total
Get-ChildItem E:\RoArm_Project\outputs\smolvla_50k\checkpoints -Recurse | Measure-Object -Property Length -Sum
```

---

## 6. Troubleshooting

### Training Crashes with CUDA OOM
```powershell
# Reduce batch size in train_config_50k.py:
# Line 73: "--batch_size=8" → "--batch_size=4"

# Or enable gradient checkpointing (requires LeRobot fork)
```

### Loss Not Decreasing
```powershell
# Check if stuck at local minimum:
python train_eval_checkpoints.py --checkpoints 20000 25000 30000

# If all checkpoints have similar L2 error → need data augmentation
```

### Checkpoint Evaluation Fails
```powershell
# Check checkpoint path:
ls E:\RoArm_Project\outputs\smolvla_50k\checkpoints\050000\pretrained_model

# Required files:
# - config.json
# - model.safetensors
# - policy_preprocessor_step_5_normalizer_processor.safetensors
# - policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

### Training Stuck at Step 20K
```powershell
# Check if resume flag is working:
cat E:\RoArm_Project\outputs\smolvla_50k\checkpoints\020000\pretrained_model\train_config.json

# Verify "resume": true in sys.argv
```

---

## 7. File Organization

### Before 50K Training
```
E:/RoArm_Project/
├── outputs/smolvla_official/
│   └── checkpoints/
│       ├── 005000/
│       ├── 010000/
│       ├── 015000/
│       └── 020000/  ← Resume from here
├── run_official_train.py  (original 20K wrapper)
├── train_config_50k.py    (NEW: 50K config)
└── train_eval_checkpoints.py  (NEW: evaluation)
```

### After 50K Training
```
E:/RoArm_Project/
├── outputs/
│   ├── smolvla_official/  (20K checkpoints)
│   └── smolvla_50k/       (NEW: 22.5K-50K)
│       └── checkpoints/
│           ├── 022500/
│           ├── 025000/
│           ...
│           └── 050000/
├── train_checkpoint_eval_results.json  (NEW: metrics)
└── train_pipeline_investigation_report.md  (NEW: detailed report)
```

---

## 8. Quick Commands Cheatsheet

```powershell
# 1. Start 50K training
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\train_config_50k.py

# 2. Monitor GPU
nvidia-smi -l 1

# 3. Evaluate checkpoints (after training)
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\train_eval_checkpoints.py

# 4. Deploy best checkpoint
E:\RoArm_Project\.venv\Scripts\python.exe E:\RoArm_Project\deploy_smolvla.py --checkpoint <BEST_CKPT_PATH>

# 5. Check disk space
Get-ChildItem E:\RoArm_Project\outputs -Recurse | Measure-Object -Property Length -Sum
```

---

**Last Updated:** 2026-02-07
**Next Update:** After 50K training completes
