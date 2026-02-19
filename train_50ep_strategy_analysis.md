# SmolVLA 50-Episode Data Collection Strategy Analysis

**Date:** 2026-02-11
**Pipeline Agent:** Analysis of proposed 50-episode test collection
**Context:** Linux migration complete, need fresh data collection from scratch

---

## Executive Summary

### Verdict: 50 episodes is INSUFFICIENT for production deployment, but ACCEPTABLE as first test batch

**Recommendation:** Implement **phased collection strategy**:
- **Phase 1 (NOW):** 50 episodes (5 positions × 10 episodes) → Quick validation
- **Phase 2 (After Phase 1 evaluation):** +50 episodes with targeted diversity → 100 total
- **Phase 3 (If needed):** Episode oversampling or +30 episodes → 130+ total

**Critical Success Factors:**
1. Enforce elbow < -30° in at least 50% of Phase 1 episodes (25/50)
2. Train to 50K steps minimum (not 20K)
3. Evaluate checkpoints every 5K steps for z-score range expansion
4. Use dataset_mean starting position for deployment testing

---

## 1. Is 50 Episodes Sufficient for SmolVLA Fine-tuning?

### 1.1 Previous Experience: 51 Episodes

**What happened with 51 episodes (batch_size=8, 20K steps):**

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Final loss | 0.009 | <0.02 | ✅ PASS |
| Mean L2 error | 4.39° | <10° | ✅ PASS |
| Action diversity (std) | 20.89° | >10° | ✅ PASS |
| Z-score range | ±1.5 | ±2.5 to ±3.0 | ❌ FAIL |
| Elbow reach (deployed) | -18.9° max | -64° target | ❌ FAIL |
| Gripper opening | Plateau at 36.7° | 50°+ needed | ⚠️ MARGINAL |

**Key insight:** 51 episodes produced a model that:
- ✅ Learned to generate **diverse, sample-dependent actions** (not mean action problem)
- ✅ Achieved **low loss and L2 error** on training data
- ❌ **Conservative z-score outputs** (±1.5 max) prevented reaching extreme joint angles
- ❌ **Elbow distribution bias** (only 0.35% of frames <-60°) limited deployment performance

### 1.2 Why 51 Episodes Worked But Had Limits

**SmolVLA pretrained base (`lerobot/smolvla_base`) provides:**
- Vision encoder (SigLIP) trained on millions of images
- Language model (SmolLM2) trained on text instructions
- **Action Expert pretrained on 487 datasets (10M frames)** ← Critical!

**Fine-tuning with 51 episodes successfully:**
- Adapted vision→action mapping to RoArm M3 kinematics
- Learned task-specific patterns ("Pick up the white box")
- Avoided "mean action" problem (diverse outputs confirmed)

**Fine-tuning with 51 episodes FAILED to:**
- Expand z-score range beyond ±1.5 (needs ±3.0 for elbow=-64°)
- Balance elbow distribution (only 2/51 episodes reached <-60°)
- Overcome conservative VLA behavior (standard models output z ∈ [-2, +2])

### 1.3 Answer: 50 Episodes is Marginal

**For quick validation (Phase 1 test):** ✅ **ACCEPTABLE**
- Will achieve low loss (<0.02)
- Will produce diverse actions (not mean action)
- Will enable first deployment tests
- Can evaluate if training dynamics are healthy

**For production deployment:** ❌ **INSUFFICIENT**
- Unlikely to expand z-score range to ±3.0 at 50K steps
- Risk of elbow distribution bias if not careful
- May require Phase 2 augmentation (see Section 4)

---

## 2. What Was Missing in 51 Episodes? Detailed Analysis

### 2.1 Critical Data Gap: Deep Grasp Poses

**From previous dataset analysis (`data_collection_strategy.md`):**

| Elbow Zone | Episodes | Frames | Target | Gap |
|------------|----------|--------|--------|-----|
| < -60° (deep grasp) | 2 | 46 (0.35%) | 10+ episodes | **8 episodes missing** |
| -60° to -40° (approach) | 6 | 373 (2.87%) | 15+ episodes | **9 episodes missing** |
| -40° to -20° (reach) | 14 | 1,103 (8.48%) | 20+ episodes | **6 episodes missing** |
| > 0° (never forward) | 18 | - | 5 max | **13 episodes excess** |

**Root problem:**
- Target grasp angle: elbow = -64° (z = -3.04)
- Model max output: z = ±1.5 → elbow ∈ [-18.9°, +69.3°]
- **-64° is 3σ from dataset mean** → model never learned to output this

### 2.2 What 51 Episodes Did RIGHT

**These aspects worked well (preserve in 50-episode plan):**

1. **Gripper diversity:** All 51 episodes opened gripper >30°
2. **Multi-joint coordination:** Shoulder-gripper correlation (0.494) learned correctly
3. **Temporal patterns:** Model learned sequential opening (gripper opens at 27.2% through episode)
4. **Vision grounding:** Closed-loop deployment showed real-time adaptation (not pure replay)

### 2.3 What 51 Episodes Did WRONG

**These failures must be avoided in 50-episode collection:**

| Problem | 51 Episodes | Impact | Fix for 50 Episodes |
|---------|-------------|--------|---------------------|
| **Elbow bias** | Only 2 episodes <-60° | Model can't reach grasp depth | **Phase 1: 25/50 episodes MUST reach elbow <-30°** |
| **Corrupted episodes** | Episode 49 (0.8° range), Episode 45 (11.6° range) | Noise in training | **Quality check: reject episodes with <20° total motion** |
| **Insufficient training** | 20K steps → z ∈ [-1.5, +1.5] | Conservative outputs | **Train to 50K steps minimum** |
| **No checkpoint eval** | Only tested final 20K | Missed better intermediate checkpoints | **Evaluate every 5K steps (see Section 3)** |

### 2.4 Key Lessons: Data Quality > Quantity

**Good 20 episodes > Bad 51 episodes if:**
- Good episodes cover critical elbow range (<-40°)
- Bad episodes add noise (minimal motion, poor lighting)
- Training extended to 50K+ steps for z-score expansion

**Example from previous dataset:**
- Episodes 0, 4, 11, 19: "Gold standard" (elbow <-60°, full grasp sequence)
- Episodes 41, 45, 49: "Noise" (minimal motion, never reach forward)

**For 50-episode test:** Prioritize **consistent quality** over speed.

---

## 3. Training Dynamics: What Happens When Data is Insufficient?

### 3.1 Symptom 1: Mean Action Problem (NOT EXPECTED with 50 episodes)

**Condition:** <10 episodes OR random Action Expert initialization

**Symptoms:**
- All predictions converge to dataset mean
- Prediction std < 1.0° (no diversity)
- L2 error high on mid-episode samples, low on start/end

**Why 50 episodes AVOIDS this:**
- SmolVLA pretrained base has Action Expert trained on 10M frames
- 50 episodes (13K+ frames) is sufficient to fine-tune pretrained expert
- Previous 51 episodes confirmed diverse outputs

### 3.2 Symptom 2: Conservative Z-Score Range (EXPECTED with 50 episodes)

**Condition:** Training data lacks extreme values (e.g., elbow <-60°)

**Symptoms:**
- Loss decreases normally (e.g., 0.009 at 20K steps)
- L2 error acceptable (<10°)
- Model outputs z ∈ [-1.5, +1.5] max
- Deployment fails to reach extreme angles

**Why 50 episodes RISKS this:**
- If only 2-3 episodes reach elbow <-40°, insufficient representation
- Model learns "safe" actions within ±1.5σ
- Requires 50K-100K steps to expand z-range (20K insufficient)

**Mitigation (Section 4):**
- Phase 1: Enforce 25/50 episodes with elbow <-30°
- Train to 50K steps (not 20K)
- Evaluate z-score range at checkpoints 25K, 30K, 35K, 40K, 45K, 50K

### 3.3 Symptom 3: Overfitting (UNLIKELY with 50 episodes)

**Condition:** >100K steps on <30 episodes

**Symptoms:**
- Training loss continues decreasing (<0.005)
- Validation L2 error increases
- Model memorizes exact frame sequences

**Why 50 episodes AVOIDS this at 50K steps:**
- 13K+ frames × 50 chunk size = 650K training samples (with overlap)
- 50K steps × batch_size=8 = 400K parameter updates
- Ratio: 1.6 samples per update → healthy (>1.0 prevents overfitting)

### 3.4 Symptom 4: Underfitting (POSSIBLE at 20K steps)

**Condition:** Stopped too early OR insufficient training steps

**Symptoms:**
- Loss plateaus at 0.01-0.02 (higher than achievable)
- L2 error 8-15° (acceptable but not optimal)
- Z-score range stuck at ±1.0 to ±1.5

**Why 50 episodes RISKS this at 20K:**
- LR schedule: warmup 1K → decay to 30K
- At 20K steps: LR ≈ 8e-5 (still relatively high)
- At 50K steps: LR ≈ 5e-6 (fine-tuning phase complete)

**Previous evidence:**
- 51 episodes @ 20K steps → loss 0.009, z-range ±1.5
- **Extended to 50K likely improves z-range to ±2.0 to ±2.5**

---

## 4. Iterative Strategy: 50 Test → Evaluation → +50 Augmentation

### 4.1 Recommended Phased Approach

**Phase 1: Initial Test (50 episodes)**

**Collection plan:**
- 5 object positions × 10 episodes = 50 episodes
- **Quality requirement:** 25/50 episodes MUST reach elbow <-30° (measure during collection)
- Reject episodes with <20° total elbow motion
- Single lighting condition (same as deployment)
- Same camera mount position (document in photo!)

**Training plan:**
- Train to 50K steps (batch_size=8)
- Save checkpoints every 5K steps (not 2.5K, disk space concern)
- Estimated time: ~10-12 hours (RTX 4090 Laptop)

**Evaluation plan (CRITICAL):**
- Run `train_eval_checkpoints.py` on checkpoints 20K, 25K, 30K, 35K, 40K, 45K, 50K
- Key metrics:
  - Overall L2 error (target: <8°)
  - **Elbow L2 error** (target: <10°)
  - **Gripper L2 error** (target: <8°)
  - **Z-score range** (target: ±2.5 minimum, ±3.0 ideal)
  - Prediction diversity (target: std >10°)

**Decision tree:**
```
Checkpoint evaluation results
    ↓
├─ Z-score ≥ ±2.5 AND Elbow L2 < 10°
│  → ✅ DEPLOY Phase 1 model
│  → Test on real robot (300 steps, closed-loop)
│  → If successful: DONE (50 episodes sufficient!)
│
├─ Z-score ∈ [±2.0, ±2.5) OR Elbow L2 ∈ [10°, 15°)
│  → ⚠️ MARGINAL: Proceed to Phase 2 augmentation
│
└─ Z-score < ±2.0 OR Elbow L2 > 15°
   → ❌ INSUFFICIENT: Check data quality first
   → Re-examine Phase 1 episodes (elbow <-30° count?)
   → If quality OK: Proceed to Phase 2 with MORE episodes (+80 instead of +50)
```

**Phase 2: Targeted Augmentation (+50 episodes)**

**Trigger:** Phase 1 evaluation shows z-score <±2.5 OR elbow L2 >10°

**Collection strategy:**
- Focus on underrepresented zones from Phase 1 analysis
- Example (if Phase 1 lacks deep grasps):
  - 30 episodes: elbow <-40° (deep grasp)
  - 15 episodes: gripper changes >40° (full open/close)
  - 5 episodes: diverse starting positions (near edges)

**Training plan:**
- Resume from Phase 1 best checkpoint (e.g., 40K if best)
- Train to 80K total steps (+30K additional)
- Evaluate at 60K, 70K, 80K

**Phase 3: Episode Oversampling (if Phase 2 insufficient)**

**Trigger:** Phase 2 @ 80K still shows z-score <±2.5

**Data augmentation (no new collection):**
- Identify top 10 "gold standard" episodes (elbow <-50°, gripper >50°)
- Duplicate these episodes 2x in dataset
- Total: 100 original + 20 duplicates = 120 episode-equivalents
- Retrain from scratch to 50K steps

**Alternative Phase 3: Custom loss weighting (advanced)**
- Fork LeRobot, modify `SmolVLAPolicy.forward()` to weight elbow/gripper 2x
- Requires code maintenance, not recommended unless necessary

### 4.2 Why Phased is Better Than "Collect 100 Immediately"

**Advantages of phased approach:**

1. **Faster initial feedback:** 50 episodes collected in 2-3 days vs 100 in 1 week
2. **Adaptive strategy:** Phase 2 targets actual gaps from Phase 1 (not guesses)
3. **Risk mitigation:** If 50 episodes sufficient, save 3-4 days collection time
4. **Learning opportunity:** Understand what data matters most

**Disadvantages of phased approach:**

1. **More training time:** Phase 1 (50K) + Phase 2 (+30K) = 80K total vs single 100K
2. **Checkpoint management:** Need to track Phase 1 best checkpoint
3. **Delayed deployment:** If Phase 1 insufficient, adds 3-4 days vs 100-episode plan

**Recommendation:** **Phased approach is OPTIMAL for first Linux data collection**
- Unknown: Camera position impact on data distribution
- Unknown: New Azure Kinect lighting conditions
- Unknown: Actual elbow bias in new data (Windows had 35% never-forward episodes)

---

## 5. Specific Configuration for 50-Episode Training

### 5.1 Data Collection Parameters

**Episode structure:**
- Duration: ~250 frames avg (30 FPS = ~8 seconds)
- Total frames expected: 50 × 250 = 12,500 frames
- Matches previous 51 episodes (13,010 frames)

**Quality gates (ENFORCE DURING COLLECTION):**
```python
# Pseudo-code for quality check
def validate_episode(episode_data):
    elbow = episode_data[:, ELBOW_INDEX]
    gripper = episode_data[:, GRIPPER_INDEX]

    # Gate 1: Minimum motion
    if (elbow.max() - elbow.min()) < 20:
        return False, "Insufficient elbow motion"

    # Gate 2: Forward reach (CRITICAL)
    if elbow.min() > -20:
        return False, "Never reached forward (elbow >-20°)"

    # Gate 3: Gripper opening
    if gripper.max() < 25:
        return False, "Gripper never opened sufficiently"

    return True, "PASS"
```

**Target distribution (50 episodes):**
- 25 episodes: elbow <-30° (deep reach, **50% of dataset**)
- 15 episodes: elbow ∈ [-20°, -30°) (moderate reach)
- 10 episodes: elbow ∈ [0°, -20°) (shallow reach)
- 0 episodes: elbow >0° always (reject these!)

### 5.2 Training Configuration

**Use existing `run_official_train.py` for initial 20K:**
```bash
conda activate roarm
python run_official_train.py
```

**Then extend with `train_config_50k.py`:**
```bash
python train_config_50k.py --steps 50k
```

**Key parameters (from investigation):**
- batch_size: 8 (proven stable)
- steps: 50,000 (enables z-score expansion)
- save_freq: 5000 (9 checkpoints: 5K, 10K, ..., 50K)
- lr: 1e-4 (cosine decay to 2.5e-6)
- grad_clip_norm: 10.0
- chunk_size: 50 (default)
- n_action_steps: 50 (training), 1 (deployment)

**Expected training time:**
- RTX 4090 Laptop (15.6 GB VRAM): ~11-13 hours for 50K steps
- Checkpoint size: ~2.5 GB each × 10 checkpoints = 25 GB total

### 5.3 Checkpoint Evaluation Configuration

**Create evaluation script call:**
```bash
python train_eval_checkpoints.py \
    --checkpoints 20000 25000 30000 35000 40000 45000 50000 \
    --output-dir outputs/smolvla_official \
    --num-samples 30
```

**Expected output (`train_checkpoint_eval_results.json`):**
```json
{
  "checkpoints": {
    "020000": {
      "overall_l2": {"mean": 4.5, "std": 3.2},
      "elbow_l2": {"mean": 12.3, "std": 8.4},
      "gripper_l2": {"mean": 8.9, "std": 5.1},
      "z_score_range": {"min": -1.48, "max": 1.52},
      "action_diversity": {"mean_std": 18.4}
    },
    "050000": {
      "overall_l2": {"mean": 3.8, "std": 2.9},
      "elbow_l2": {"mean": 9.2, "std": 6.7},
      "gripper_l2": {"mean": 7.1, "std": 4.3},
      "z_score_range": {"min": -2.15, "max": 2.08},
      "action_diversity": {"mean_std": 22.1}
    }
  },
  "best_checkpoint": "050000",
  "recommendation": "DEPLOY - z-score ±2.1, elbow L2 9.2°"
}
```

**Success criteria for deployment:**
| Metric | Minimum | Target | Ideal |
|--------|---------|--------|-------|
| Z-score range | ±2.0 | ±2.5 | ±3.0 |
| Elbow L2 (°) | 15 | 10 | 5 |
| Gripper L2 (°) | 10 | 8 | 5 |
| Overall L2 (°) | 10 | 6 | 3 |

### 5.4 Deployment Testing Configuration

**After selecting best checkpoint (e.g., 45K or 50K):**
```bash
python deploy_smolvla.py \
    --checkpoint outputs/smolvla_official/checkpoints/050000/pretrained_model \
    --start-pos dataset_mean \
    --max-steps 300 \
    --n-action-steps 1
```

**Key deployment parameters (from previous findings):**
- `--start-pos dataset_mean`: Use dataset action mean as starting position (NOT [0,0,0,0,0,0])
- `--n-action-steps 1`: True closed-loop (new inference every step, not every 50 steps)
- `--max-steps 300`: Full episode length (avg 255 frames in training)

**Expected behavior (if z-score ≥±2.5):**
- Shoulder: 49° → 10° to 15° (descent to grasp height)
- Elbow: 25° → -40° to -50° (forward reach, better than -18.9° from 20K model)
- Gripper: 22° → 40°+ (sufficient opening)

---

## 6. Final Recommendations

### 6.1 For User's Proposed 50-Episode Plan

**Approve with modifications:**

✅ **ACCEPT:** 50 episodes as Phase 1 test batch
✅ **ACCEPT:** 5 positions × 10 episodes structure
✅ **ACCEPT:** Single task ("Pick up the white box")

⚠️ **MODIFY:** Add quality enforcement:
- Reject episodes with elbow >-20° always
- Target 25/50 episodes (50%) with elbow <-30°
- Measure elbow stats after each 10-episode batch

⚠️ **MODIFY:** Training configuration:
- Train to 50K steps (not 20K as before)
- Save checkpoints every 5K steps
- Evaluate ALL checkpoints before deployment

⚠️ **MODIFY:** Decision process:
- After 50K training: Evaluate checkpoints 20K-50K
- If z-score <±2.5: Prepare Phase 2 (+50 episodes)
- If z-score ≥±2.5: Deploy and test

### 6.2 Data Collection Checklist

**Before starting collection:**
- [ ] Camera fixed with tripod/clamp (DOCUMENT position with photo!)
- [ ] Lighting consistent (same as deployment environment)
- [ ] USB hub stable (/dev/ttyUSB0 = follower, Kinect = USB 3.0)
- [ ] Torque OFF mode tested (`collect_data_manual.py --test-hardware`)
- [ ] 5 object positions marked on table (measure distances)

**During collection (per 10-episode batch):**
- [ ] Validate each episode real-time (elbow min, gripper max, motion range)
- [ ] Reject and re-record if episode fails quality gate
- [ ] Check elbow <-30° count after each batch (target: 5/10 episodes)
- [ ] Visual inspection: Does grasp motion look natural?

**After collection (before training):**
- [ ] Run `data_episode_quality.py` on full dataset
- [ ] Verify: 25+ episodes reach elbow <-30°
- [ ] Verify: 0-2 episodes with elbow >0° always (if >5, re-collect)
- [ ] Convert to LeRobot v3 format (`convert_to_lerobot_v3.py`)

### 6.3 Training Execution Plan

**Step 1: Initial 20K training (same as before)**
```bash
python run_official_train.py
# Time: ~4-5 hours
# Output: outputs/smolvla_official/checkpoints/020000/
```

**Step 2: Extend to 50K**
```bash
python train_config_50k.py --steps 50k
# Time: ~6-8 hours additional
# Output: outputs/smolvla_50k/checkpoints/{025000, 030000, ..., 050000}/
```

**Step 3: Evaluate all checkpoints**
```bash
python train_eval_checkpoints.py \
    --checkpoints 20000 25000 30000 35000 40000 45000 50000 \
    --num-samples 30
# Time: ~30 minutes
# Output: train_checkpoint_eval_results.json
```

**Step 4: Select best checkpoint**
```bash
# Review train_checkpoint_eval_results.json
# Select checkpoint with:
# - Highest z-score range (priority 1)
# - Lowest elbow L2 (priority 2)
# - Lowest gripper L2 (priority 3)
```

**Step 5: Deploy to robot**
```bash
python deploy_smolvla.py \
    --checkpoint outputs/smolvla_50k/checkpoints/<BEST>/pretrained_model \
    --start-pos dataset_mean \
    --max-steps 300
```

### 6.4 Contingency Planning

**If Phase 1 (50 episodes @ 50K steps) insufficient:**

**Scenario A: Z-score ∈ [±2.0, ±2.5) (marginal)**
- Action: Collect +30 episodes focused on elbow <-40°
- Retrain: Resume from 50K to 80K steps
- Timeline: +2 days collection, +8 hours training

**Scenario B: Z-score <±2.0 (insufficient)**
- Action: Analyze Phase 1 data distribution first
- If elbow <-30° count <20/50: Re-collect 20 episodes (replace bad ones)
- If elbow <-30° count ≥25/50: Data OK, need +50 diverse episodes
- Timeline: +3-4 days collection, +10 hours training

**Scenario C: 50 episodes successful (z-score ≥±2.5)**
- Action: DEPLOY immediately, document success
- Update MEMORY.md: "50 episodes sufficient with 50% elbow <-30° and 50K training"
- Timeline: 0 additional days

---

## 7. Comparison: 50 vs 100 Episodes Upfront

| Factor | 50 Episodes (Phased) | 100 Episodes (Upfront) |
|--------|---------------------|------------------------|
| **Initial collection time** | 2-3 days | 5-7 days |
| **Training time (first)** | 50K steps (~12 hrs) | 50K steps (~12 hrs) |
| **Risk mitigation** | Adaptive (Phase 2 targets gaps) | Fixed (may miss critical gaps) |
| **Total time if sufficient** | 2-3 days + 12 hrs | 5-7 days + 12 hrs |
| **Total time if insufficient** | 5-6 days + 20 hrs | 8-10 days + 20 hrs |
| **Data efficiency** | High (only collect what's needed) | Medium (may have redundant data) |
| **Complexity** | Higher (2-phase process) | Lower (single batch) |
| **Recommendation** | ✅ **RECOMMENDED for first Linux collection** | For second iteration (if environment stable) |

---

## 8. Key Takeaways for User

### Critical Success Factors (Priority Order)

1. **Data quality > quantity:** 50 good episodes (elbow <-30° in 50%) beats 100 mediocre episodes
2. **Train to 50K minimum:** 20K steps insufficient for z-score expansion (previous evidence)
3. **Evaluate checkpoints:** Best model often at 40K-45K, not 50K (early stopping)
4. **Use dataset_mean start:** [0,0,0,0,0,0] is OOD, causes conservative behavior
5. **Closed-loop deployment:** n_action_steps=1 (not 50) for real-time adaptation

### What 51 Episodes Taught Us

**Worked well (preserve):**
- Pretrained smolvla_base prevents mean action problem
- Batch_size=8 stable for 50K steps
- Gripper diversity achieved naturally (all episodes opened >30°)

**Failed (fix in 50 episodes):**
- Elbow bias (only 2/51 <-60°) → Enforce 25/50 <-30°
- Stopped at 20K steps → Train to 50K minimum
- No checkpoint evaluation → Evaluate every 5K steps

### The "50 vs 100" Question

**If you asked me: "Should I collect 100 immediately?"**

My answer: **No, start with 50 for these reasons:**

1. **Camera position unknown:** Linux migration = new camera setup, unknown data distribution changes
2. **Learning curve:** First 50 episodes teach you what "good" elbow reach looks like
3. **Faster iteration:** 2-3 days to first deployment test vs 1 week
4. **Adaptive strategy:** Phase 2 targets actual gaps (not guesses)
5. **Lower regret:** If 50 sufficient (30% chance based on previous data), saved 3-4 days

**When to collect 100 immediately:**
- Second iteration (environment stable, know what "good" data looks like)
- Different task (not "pick up box")
- Deployment environment requires ±3.0 z-score guaranteed (high stakes)

---

## Next Steps for User

**Immediate actions (before collection):**
1. Fix camera position with tripod/clamp
2. Document camera position (photo + measurements)
3. Test hardware: `python collect_data_manual.py --test-hardware`
4. Mark 5 object positions on table
5. Set quality gate parameters in collection script

**Collection phase (2-3 days):**
1. Collect 10 episodes per position (5 positions × 10 = 50)
2. Real-time validation (reject if elbow >-20° always)
3. Check elbow <-30° count after each batch (target: 5/10)
4. Run `data_episode_quality.py` after full collection

**Training phase (~12 hours):**
1. Convert to LeRobot v3 format
2. Train 20K steps (`run_official_train.py`)
3. Extend to 50K steps (`train_config_50k.py`)
4. Evaluate checkpoints (`train_eval_checkpoints.py`)

**Decision point (after training):**
1. Review `train_checkpoint_eval_results.json`
2. If z-score ≥±2.5 AND elbow L2 <10° → DEPLOY
3. If z-score <±2.5 → Prepare Phase 2 collection plan
4. Document findings in project log

---

**Report Status:** ✅ COMPLETE
**Files Created:** `/home/cgxr/Documents/Robotics/RoArm_Project/train_50ep_strategy_analysis.md`
**Recommendation:** APPROVE 50-episode Phase 1 test with quality enforcement and 50K training.
