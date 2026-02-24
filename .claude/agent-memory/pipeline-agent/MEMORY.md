# Pipeline Agent Memory

## 50K Training Analysis (2026-02-11)

### Model Escaped Mean Action Problem ✅
- L2 error: 2.53° average (excellent)
- Diversity: Overall std=21.55° (dataset: 21.75-29.03°)
- Elbow deep extension works: -63.37° pred vs -65.39° GT

### Critical Issues Found
1. **Wrist_R under-prediction**: Pred std=3.34° vs dataset std=22.14° (15% variance)
   - Hypothesis: MSE loss weights all joints equally, model minimizes by staying near mean
   - Impact: Orientation errors during manipulation
2. **Gripper timing lag**: 2° error at some samples (43.72 vs 45.70)
3. **Overfitting risk**: 37 epochs (50 episodes × 37), loss 0.126 → 0.007 (94% drop)
4. **No validation set**: Test samples from training set, zero OOD confidence

### SmolVLA Loss Investigation
- File: `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`
- Line 791: `F.mse_loss(u_t, v_t, reduction="none")` → (B, T, num_motors)
- Line 399: `losses.mean()` → averages all dimensions equally
- **NO built-in per-joint weighting** in SmolVLA
- Custom weighting possible but violates "no custom training" rule
- **Preferred alternative**: Data resampling (oversample Wrist_R-heavy episodes)

### Deployment Readiness
- **CONDITIONALLY READY** with monitoring
- Start with dry-run, then 10 limited trials
- Monitor: Per-joint z-scores, Wrist_R range, gripper timing
- Abort if: Elbow < -70°, gripper opens during lift, base > 180°

### Next Steps Priority
1. Run checkpoint evaluation (15K, 25K, 35K, 45K, 50K) - find optimal before overfitting
2. Create validation split (10 held-out episodes)
3. Cautious deployment test (5 trials, log failures)
4. Collect 100+ episodes (50 elbow<-30°, 30 wrist_R diverse, 20 rapid gripper)
5. Re-train 100K steps (16 epochs vs current 37, lower overfitting)

### Files Created
- `train_recommendations_50k.md`: Comprehensive analysis (8 sections, appendices)
- Updated `train_eval_checkpoints.py`: Default checkpoints 15K-50K

### Key Lessons
- Loss ↓ ≠ Good model (0.007 loss but Wrist_R std=3.34° is bad)
- Need validation set BEFORE claiming success
- Per-joint analysis critical (overall metrics hide joint-specific failures)
- Flow matching's 10 denoising steps may lag on rapid transitions (gripper)

## Data Collection Protocol (2026-02-23)

### Perfect Episode Structure (7 phases, ~145 frames at 30fps)
1. Start (15f): Closed gripper, arm at init, Z=250-350mm
2. Approach (30f): Gripper OPENS (0°→70°), arm moves toward object
3. Pre-grasp (15f): Gripper FULLY OPEN (60-80°), hovering above, Z=150-200mm
4. Descent (15f): Gripper STILL OPEN, arm lowers to Z=80-120mm (DEEP zone)
5. Grasp (10f): Gripper CLOSES decisively (70°→5°), ALL OTHER JOINTS STATIC
6. Lift (30f): Gripper STAYS CLOSED, arm rises to Z=300mm+
7. Return (30f): Return to init, gripper opens at end

### 5 Position Layout (from base center)
- P1 center: 250mm, 0° (Blue tape, 30 eps)
- P2 left: 250mm, -30° (Yellow tape, 20 eps)
- P3 far: 300mm, 0° (Red tape, 20 eps) — DEEP focus
- P4 right: 250mm, +30° (Green tape, 20 eps)
- P5 near: 200mm, 0° (White tape, 10 eps)

### Gripper Timing (root cause of v1 58% early, 40% late)
- Rule: Open gripper AT START OF APPROACH (not when descending)
- Rule: Close gripper ONLY when Z is stable at 80-120mm
- Rule: Close decisively in 10 frames (0.3s), not slowly
- Rule: Never close while still descending (Z decreasing)

### Target Distribution (100 episodes)
- DEEP (Z < 100mm): 50+ episodes (50%+) — most critical
- Gripper range > 40°: 100% of episodes
- P3 far: 20 episodes (fixes v1 DEEP gap of only 9/50)

### Official SmolVLA Reference
- 50 eps, 5 positions × 10 reps = enough for SO101
- Our target: 100 eps (5 positions × 20 reps) for better generalization
- Full pick-lift-return cycle required (not just pick-and-stop)
- Task text must end with \n (SmolVLANewLineProcessor requirement)

### Protocol File
- `/home/cgxr/Documents/Robotics/RoArm_Project/train_data_collection_protocol.md`

## VRAM Test Results (2026-02-24)

### RTX 4090 Laptop (16.72 GB actual, not 15.6 GB)
| Batch | Peak VRAM | Util% | Headroom |
|-------|-----------|-------|---------|
| 8     | 2.03 GB   | 12.1% | 14.69 GB |
| 16    | 3.15 GB   | 18.9% | 13.56 GB |
| 32    | 5.38 GB   | 32.2% | 11.33 GB |
| 64    | 9.85 GB   | 58.9% | 6.87 GB |

- **batch_size=64 FITS (9.85 GB / 16.72 GB = 58.9%)** — use official config!
- Base VRAM (model load): 0.91 GB
- Per-sample activation: ~140 MB/sample
- Model: 450M total, 99.9M trainable (Action Expert only), 350.2M frozen (VLM)

### gradient_accumulation in lerobot
- lerobot-train does NOT support --gradient_accumulation_steps
- Searched entire lerobot/src/lerobot/: zero matches for 'gradient_accumulation'
- TrainPipelineConfig has no such field
- Workaround: use batch_size=64 directly (it fits!)

### Dummy Batch Shape (for test scripts)
- Image: (B, 3, 480, 640) float32 [0,1] — model resizes to 512x512 internally
- OBS_STATE: (B, 1, 32) float32
- ACTION: (B, 50, 32) float32
- OBS_LANGUAGE_TOKENS: (B, 48) int64
- OBS_LANGUAGE_ATTENTION_MASK: (B, 48) **BOOL** (not int64!)
  - make_att_2d_masks requires bool — int64 raises RuntimeError

### Test Script
- `/home/cgxr/Documents/Robotics/RoArm_Project/train_batch_size_test.py`
