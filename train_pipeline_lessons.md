# SmolVLA Training Pipeline: Critical Lessons Learned

**Date:** 2026-02-11
**Context:** Post-migration to Linux, preparing for fresh 50-100 episode data collection
**Source:** Analysis of 51-episode Windows training results + pipeline investigation

---

## Executive Summary: 50 Episodes Strategy

### Verdict
**50 episodes is ACCEPTABLE as Phase 1 test, but likely INSUFFICIENT for production deployment.**

### Critical Success Factors
1. **Data quality:** 50% of episodes (25/50) MUST reach elbow <-30°
2. **Training duration:** 50K steps MINIMUM (not 20K)
3. **Checkpoint evaluation:** Test every 5K steps (20K, 25K, 30K, ..., 50K)
4. **Deployment config:** dataset_mean start + n_action_steps=1 + 300 max steps

### Recommended Approach
**Phased collection strategy:**
- Phase 1: 50 episodes (5 positions × 10) → train 50K → evaluate
- Phase 2: +50 episodes if z-score <±2.5 → train to 80K → evaluate
- Phase 3: Episode oversampling if still insufficient

**Total timeline if Phase 1 sufficient:** 2-3 days collection + 12 hours training
**Total timeline if Phase 2 needed:** 5-6 days collection + 20 hours training

---

## Why 51 Episodes @ 20K Steps Had Limits

### What Worked
| Metric | Result | Evidence |
|--------|--------|----------|
| **Mean action avoided** | ✅ Diverse outputs | Action std 20.89° (not <1°) |
| **Low training loss** | ✅ 0.009 | 99.7% decrease from initial |
| **Good L2 error** | ✅ 4.39° avg | Better than 10° target |
| **Closed-loop adaptation** | ✅ Real-time | 300-step deployment showed plateau, not drift |

### What Failed
| Metric | Result | Root Cause |
|--------|--------|------------|
| **Z-score range** | ❌ ±1.5 max | Need ±3.0 for elbow=-64° |
| **Elbow reach (deployed)** | ❌ -18.9° max | z=-1.5 × 29.38 + 25.19 = -18.9° |
| **Deep grasp representation** | ❌ 0.35% frames <-60° | Only 2/51 episodes reached deep zone |
| **Training duration** | ❌ 20K steps | LR still 8e-5, needs 50K for fine-tuning |

### Key Insight
**SmolVLA pretrained base prevented mean action problem, but could not overcome elbow distribution bias.**
- Action Expert (100M params) pretrained on 10M frames → sufficient for 50 episodes
- But model outputs z ∈ [-1.5, +1.5] when training data lacks extreme values
- Only 2/51 episodes (3.9%) reached elbow <-60° → insufficient for z=-3.04 learning

---

## Data Quality Requirements for 50 Episodes

### Quality Gates (Enforce During Collection)

```python
# Reject episodes that fail ANY of these criteria:
def validate_episode(episode_actions):
    elbow = episode_actions[:, ELBOW_INDEX]
    gripper = episode_actions[:, GRIPPER_INDEX]

    # Gate 1: Minimum motion
    if (elbow.max() - elbow.min()) < 20:
        return False, "Insufficient elbow motion (<20°)"

    # Gate 2: Forward reach (CRITICAL)
    if elbow.min() > -20:
        return False, "Never reached forward (elbow >-20° always)"

    # Gate 3: Gripper opening
    if gripper.max() < 25:
        return False, "Gripper never opened (<25°)"

    return True, "PASS"
```

### Target Distribution for 50 Episodes

| Elbow Zone | Episodes | Percentage | Previous (51 ep) | Gap |
|------------|----------|------------|------------------|-----|
| < -30° (deep reach) | **25** | **50%** | 8 (15.7%) | **+17 episodes** |
| -30° to -20° (moderate) | 15 | 30% | 14 (27.5%) | +1 episode |
| -20° to 0° (shallow) | 10 | 20% | 11 (21.6%) | -1 episode |
| > 0° (never forward) | **0** | **0%** | 18 (35.3%) | **-18 episodes** |

**Critical:** Previous dataset had 18/51 episodes (35%) that never reached forward. This MUST be 0/50 in new dataset.

### Practical Collection Strategy

**Setup:**
- 5 object positions on table (vary X, Y coordinates)
- Single lighting condition (match deployment environment)
- Camera fixed with tripod/clamp (DOCUMENT position!)

**Per-position collection (10 episodes):**
- Record 10 episodes at same object position
- Vary starting position and grasp trajectory (not exact replay)
- Real-time validation: Accept only if elbow <-20° at some frame
- Target: 5/10 episodes reach elbow <-30° (50% success rate)

**After each 10-episode batch:**
- Run quick stats: `min_elbow = min(episode_elbows)`
- Count: How many episodes reached <-30°? (should be ~5)
- If <3/10 reached <-30°: Adjust demonstration technique (reach deeper)

---

## Training Configuration: 50K Steps

### Why 50K (Not 20K)?

**Learning rate schedule (cosine decay):**
| Steps | LR | Phase | Z-score expansion |
|-------|-----|-------|------------------|
| 0-1K | 0 → 1e-4 | Warmup | None |
| 1K-20K | 1e-4 → 8e-5 | Early decay | Limited (±1.0 to ±1.5) |
| **20K-30K** | **8e-5 → 5e-5** | **Refinement** | **±1.5 to ±2.0** (target) |
| **30K-50K** | **5e-5 → 2.5e-6** | **Fine-tuning** | **±2.0 to ±2.5** (ideal) |

**Evidence from previous training:**
- 20K steps: loss 0.009, z-range ±1.5 (insufficient)
- LR at 20K: 8e-5 (still relatively high)
- LR at 50K: ~5e-6 (fine-tuning phase complete)

**Expected improvement:**
- 30K additional steps allow model to explore extreme z-scores
- Lower LR stabilizes training on rare data (elbow <-60° frames)
- **Estimated z-range at 50K: ±2.0 to ±2.5** (based on typical VLA behavior)

### Training Commands

**Step 1: Initial 20K (4-5 hours)**
```bash
conda activate roarm
python run_official_train.py
```
Output: `outputs/smolvla_official/checkpoints/020000/pretrained_model`

**Step 2: Extend to 50K (6-8 hours additional)**
```bash
python train_config_50k.py --steps 50k
```
Output: `outputs/smolvla_50k/checkpoints/{025000, 030000, ..., 050000}/pretrained_model`

**Step 3: Evaluate checkpoints (30 minutes)**
```bash
python train_eval_checkpoints.py \
    --checkpoints 20000 25000 30000 35000 40000 45000 50000 \
    --num-samples 30
```
Output: `train_checkpoint_eval_results.json`

### Checkpoint Evaluation Metrics

**Success criteria for deployment:**
| Metric | Minimum (marginal) | Target | Ideal |
|--------|-------------------|--------|-------|
| **Z-score range** | **±2.0** | **±2.5** | **±3.0** |
| Elbow L2 error (°) | 15 | 10 | 5 |
| Gripper L2 error (°) | 10 | 8 | 5 |
| Overall L2 error (°) | 10 | 6 | 3 |
| Action diversity (std °) | 15 | 20 | 25 |

**Decision tree:**
```
Z-score ≥ ±2.5 AND Elbow L2 < 10°
    → ✅ DEPLOY to real robot (Phase 1 success!)

Z-score ∈ [±2.0, ±2.5) OR Elbow L2 ∈ [10°, 15°)
    → ⚠️ MARGINAL: Test deployment, may need Phase 2

Z-score < ±2.0 OR Elbow L2 > 15°
    → ❌ INSUFFICIENT: Analyze data, proceed to Phase 2
```

---

## Deployment Testing Configuration

### Critical Parameters (From Previous Findings)

**DO NOT use defaults - they cause conservative behavior:**

| Parameter | Default | Required | Why |
|-----------|---------|----------|-----|
| `--start-pos` | [0,0,0,0,0,0] | **dataset_mean** | [0,0,0,0,0,0] is OOD → model outputs cautious actions |
| `--n-action-steps` | 50 | **1** | 50 = open-loop (ignores real observations for 50 steps) |
| `--max-steps` | 100 | **300** | 100 covers only 39% of episode (avg 255 frames) |

**Deployment command:**
```bash
python deploy_smolvla.py \
    --checkpoint outputs/smolvla_50k/checkpoints/<BEST>/pretrained_model \
    --start-pos dataset_mean \
    --n-action-steps 1 \
    --max-steps 300
```

### Expected Behavior by Z-Score Range

**If z-score ≥ ±2.5 (target achieved):**
- Shoulder: 49° → 10° to 15° (descent to grasp height)
- Elbow: 25° → -40° to -50° (forward reach toward object)
- Gripper: 22° → 45°+ (wide opening for grasp)
- Behavior: Smooth approach, plateau at grasp position

**If z-score ∈ [±2.0, ±2.5) (marginal):**
- Shoulder: 49° → 15° to 20°
- Elbow: 25° → -30° to -40° (partial reach)
- Gripper: 22° → 35° to 40° (moderate opening)
- Behavior: Approaches but may not reach deep enough

**If z-score < ±2.0 (insufficient, like 20K model):**
- Shoulder: 49° → 30° to 35°
- Elbow: 25° → -18° max (z=-1.5 limit)
- Gripper: 22° → 35° (opens but plateaus early)
- Behavior: Conservative "air grasping", never touches object

---

## Phase 2: What If 50 Episodes @ 50K Insufficient?

### Diagnosis: Check Data Distribution First

**Before collecting more data, analyze Phase 1:**
```bash
python data_episode_quality.py
python data_distribution_simple.py
```

**Key questions:**
1. How many episodes reached elbow <-30°? (target: 25/50 = 50%)
2. What % of frames have elbow <-40°? (target: >5%, was 2.87% in previous)
3. What % of frames have elbow <-60°? (target: >1%, was 0.35% in previous)

**If Phase 1 data meets targets but z-score still <±2.5:**
→ Data distribution OK, need more training or episode oversampling

**If Phase 1 data FAILS targets (e.g., only 10/50 episodes <-30°):**
→ Data quality issue, re-collect 20 episodes to replace worst ones

### Option A: Collect +50 Targeted Episodes

**When to use:** Phase 1 data meets quality targets but z-score <±2.5

**Collection strategy:**
- 30 episodes: elbow <-40° (deep grasp focus)
- 15 episodes: gripper changes >40° (full open/close)
- 5 episodes: diverse starting positions (near workspace edges)

**Training:**
- Resume from Phase 1 best checkpoint (e.g., 45K)
- Train to 80K total steps (+30K to 35K additional)
- Evaluate at 60K, 70K, 80K

**Expected improvement:**
- Z-score range: ±2.5 to ±3.0 (with 100 total episodes)
- Elbow L2: <8° (better precision)

### Option B: Episode Oversampling (No New Collection)

**When to use:** Phase 1 data has 5-10 "gold standard" episodes (elbow <-50°) but z-score still <±2.5

**Method:**
1. Identify top 10 episodes:
   - Min elbow <-50°
   - Max gripper >50°
   - Elbow range >80°

2. Duplicate these episodes:
   - Copy episode data in dataset
   - Re-index episode_index (50 original + 10 duplicates = 60)
   - Update meta.json episode count

3. Retrain from scratch:
   - Use oversampled dataset
   - Train to 50K steps
   - Overrepresentation of deep grasps → wider z-score range

**Expected improvement:**
- Effective training data: 50 original + 10×3 duplicates = 80 episode-equivalents
- Z-score range: ±2.0 to ±2.5 (moderate improvement)

### Option C: Custom Loss Weighting (Advanced)

**When to use:** Phase 2 (100 episodes) still insufficient, z-score <±2.0

**Requirements:**
- Fork LeRobot repository
- Modify `lerobot/policies/smolvla/modeling_smolvla.py:791`

**Implementation:**
```python
# Original (line 791):
loss = losses.mean()

# Modified (joint-weighted):
joint_weights = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 1.5], device=losses.device)
# Weights: [Base, Shoulder, ELBOW×2, Wrist_P, Wrist_R, GRIPPER×1.5]
weighted_losses = losses * joint_weights[None, None, :]  # Broadcast to (B, T, A)
loss = weighted_losses.mean()
```

**Impact:**
- Elbow errors weighted 2× → model prioritizes elbow precision
- Gripper errors weighted 1.5× → model prioritizes open/close timing

**Risks:**
- Requires maintaining LeRobot fork
- May destabilize training if weights too extreme
- Needs validation on held-out episodes

**Recommendation:** Only use if Phase 2 (100 episodes + 80K steps) fails. Prefer data augmentation over code modification.

---

## Common Failure Modes and Fixes

### Symptom: Mean Action Problem (All Predictions Identical)

**Evidence:**
- Action std <1.0° across test samples
- L2 error high (>40°) on mid-episode samples
- Predictions always near dataset mean

**Root causes:**
1. ❌ Random Action Expert initialization (not loaded from smolvla_base)
2. ❌ <10 episodes (insufficient even with pretrained base)
3. ❌ batch_size=1 (gradient noise overwhelming signal)

**Diagnosis:**
```bash
python test_inference_official.py
# Check output: Action std should be >10° (not <1°)
```

**Fix:**
- Verify `--policy.pretrained_path=lerobot/smolvla_base` in run_official_train.py
- Use batch_size=8 (never 1)
- Collect minimum 20 episodes (50 recommended)

**Note:** With 50 episodes + smolvla_base pretrained, this problem is VERY UNLIKELY.

### Symptom: Conservative Z-Score Range (±1.5 max)

**Evidence:**
- Loss decreases normally (0.009)
- L2 error acceptable (<10°)
- Deployment tests show limited joint motion
- Model output range narrower than dataset range

**Root causes:**
1. ⚠️ Elbow distribution bias (only 2-3 episodes reach <-40°)
2. ⚠️ Insufficient training (stopped at 20K steps)
3. ⚠️ Lack of extreme value representation (P1 percentile = -58.9°)

**Diagnosis:**
```bash
python data_distribution_simple.py
# Check: % of frames with elbow <-40° (target: >5%)
```

**Fix (priority order):**
1. Train to 50K steps (not 20K)
2. Ensure 25/50 episodes reach elbow <-30° during collection
3. If still insufficient: Phase 2 (+50 targeted episodes)

**Note:** This is the MOST LIKELY failure mode for 50-episode Phase 1.

### Symptom: Overfitting (Training Loss <0.005, Deployment Fails)

**Evidence:**
- Training loss very low (<0.005)
- Exact replay of training episodes works
- Any variation from training conditions fails
- Model memorizes frame sequences

**Root causes:**
1. ❌ Too many training steps (>100K on <30 episodes)
2. ❌ Insufficient data diversity (all episodes same lighting/position)

**Diagnosis:**
- Compare training loss curve: Does it continue decreasing after 50K?
- Test deployment with varied lighting: Does performance degrade?

**Fix:**
- Stop training at 50K-80K for 50-100 episodes
- Increase data diversity (5 object positions, not 1)
- Early stopping based on checkpoint evaluation

**Note:** With 50 episodes + 50K steps, this problem is UNLIKELY.

---

## Critical Lessons from Previous Failures

### Custom Training Scripts (3 Attempts, ALL FAILED)

**Attempt 1: batch_size=1, vlm=False**
- Result: Mean action problem (all predictions identical)
- Cause: Gradient noise (single sample per update)

**Attempt 2: batch_size=8, vlm=False**
- Result: Mean action problem (all predictions identical)
- Cause: Action Expert random initialization (not pretrained)

**Attempt 3: batch_size=8, vlm=True (VLM weights only)**
- Result: Mean action problem (all predictions identical)
- Cause: Action Expert STILL random (only VLM loaded, not full smolvla_base)

**Key insight:** Must load FULL `lerobot/smolvla_base` (VLM + Action Expert), not just VLM.

### Official CLI Success (First Try)

**Configuration:**
```bash
lerobot-train \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id=roarm_m3_pick \
  --batch_size=8 \
  --steps=20000
```

**Result:**
- Diverse, sample-dependent actions (std 20.89°)
- Low loss (0.009) and L2 error (4.39°)
- Closed-loop deployment worked (though conservative)

**What official CLI does correctly:**
1. Loads pretrained Action Expert from smolvla_base
2. Applies MEAN_STD normalization automatically
3. Uses cosine LR decay + warmup + gradient clipping
4. Mixed precision training (faster, lower memory)

**Lesson:** NEVER write custom training scripts. Always use `lerobot-train` CLI via `run_official_train.py` wrapper.

---

## Quick Reference: File Responsibilities

### Pipeline Agent Owns (Can Modify)
- ✅ `train_*.py` - Training configs, evaluation scripts
- ✅ `run_official_train.py` - Training wrapper
- ✅ `test_inference_official.py` - Offline evaluation

### Pipeline Agent Can Read (Investigation Only)
- 📖 `lerobot/` - LeRobot source code
- 📖 `outputs/` - Checkpoints (read for evaluation)
- 📖 `lerobot_dataset_v*/` - Datasets (read for stats)

### Pipeline Agent CANNOT Modify
- ❌ `lerobot/` - LeRobot source (unless fork approved by Lead)
- ❌ `collect_data_*.py` - Data collection (Data Agent owns)
- ❌ `deploy_*.py` - Deployment scripts (Deploy Agent owns)
- ❌ Git operations - Commits, pushes (Lead Agent only)

---

## Next Actions for 50-Episode Strategy

**User should:**
1. Fix camera position with tripod/clamp
2. Document camera position (photo + measurements from table)
3. Test hardware: `python collect_data_manual.py --test-hardware`
4. Mark 5 object positions on table (measure distances)

**Data Agent should:**
1. Implement quality gate validation in `collect_data_manual.py`
2. Add elbow <-30° count tracking per batch
3. Auto-reject episodes that fail quality gates

**Pipeline Agent (me) should:**
1. Monitor training progress (already configured in `run_official_train.py`)
2. Run checkpoint evaluation after 50K training completes
3. Analyze results and recommend Phase 2 if needed

**Deploy Agent should:**
1. Prepare deployment script with correct parameters (dataset_mean, n_action_steps=1)
2. Document deployment test results
3. Compare with 20K model baseline (elbow -18.9° max)

---

## Summary: 50-Episode Strategy Approval

**VERDICT:** ✅ **APPROVE with quality enforcement and 50K training**

**Critical requirements:**
1. 25/50 episodes MUST reach elbow <-30° (reject and re-record if not)
2. Train to 50K steps minimum (not 20K)
3. Evaluate checkpoints every 5K steps (20K-50K)
4. Deploy with dataset_mean start + n_action_steps=1

**Expected outcomes:**
- **Best case (30% chance):** Z-score ≥±2.5, elbow L2 <10° → deploy immediately
- **Likely case (50% chance):** Z-score ∈[±2.0, ±2.5) → marginal, test deployment, may need Phase 2
- **Worst case (20% chance):** Z-score <±2.0 → insufficient, proceed to Phase 2 (+50 episodes)

**Timeline:**
- Phase 1 collection: 2-3 days (50 episodes with quality gates)
- Phase 1 training: 12 hours (50K steps)
- Phase 1 evaluation: 30 minutes (checkpoint analysis)
- **Total to first deployment test: 3-4 days**

**Phased approach advantages:**
- Faster initial feedback vs 100-episode upfront (3-4 days vs 1 week)
- Adaptive Phase 2 targeting (based on actual gaps, not guesses)
- Lower regret if 50 sufficient (saves 3-4 days)

**References:**
- Detailed analysis: `/home/cgxr/Documents/Robotics/RoArm_Project/train_50ep_strategy_analysis.md`
- Previous results: `SMOLVLA_TRAINING_RESULTS.md` Section 11-14
- Pipeline investigation: `train_pipeline_investigation_report.md`
- Data strategy: `data_collection_strategy.md` Section 1-3
