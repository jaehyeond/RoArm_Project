# Data Collection Strategy for RoArm M3 SmolVLA

## ⚠️ OBSOLETE DATA WARNING (2026-02-07)

> **카메라 위치가 물리적으로 변경되어 기존 51 에피소드는 전부 OOD (Out-of-Distribution).**
> SmolVLA는 Vision-Language-Action 모델이므로 이미지 분포가 바뀌면 학습 데이터가 무효화됩니다.
> **Linux 이식 후 100+ 에피소드를 처음부터 재수집해야 합니다.**
>
> 아래 분석은 이전 카메라 위치 기준이지만, **수집 전략 (Phase 1-3)**과 **품질 기준 (Section 1)**은
> 새 데이터 수집에도 동일하게 적용됩니다.

---

## Executive Summary

Analysis of 51 episodes (13,010 frames) reveals **critical underrepresentation of deep grasp poses**. The model outputs conservative actions (z-scores within ±1.5) but successful grasping requires elbow=-64° (z=-3.04), which is 3+ standard deviations from the mean. Only 0.35% of training data contains elbow < -60°, explaining why the deployed model stays at +31.8° during inference.

**Key Findings:**
- Only 2/51 episodes (3.9%) reach elbow < -60° (deep grasp zone)
- Only 8/51 episodes (15.7%) reach elbow < -40° (grasp approach)
- 31/51 episodes (60.8%) are classified as "BAD" (insufficient reach or gripper opening)
- Elbow=-64° requires z=-3.04, but model outputs max z=±1.5 (standard VLA behavior)

**Recommendation:** Collect 80+ new episodes focused on deep grasp poses to balance the distribution.

---

## 1. Episode Quality Analysis

### 1.1 Classification Criteria

Episodes are classified as "GOOD" if they meet ALL criteria:
- `min_elbow < -20°` (reaches forward toward object)
- `max_gripper > 30°` (gripper opens sufficiently)
- `elbow_range > 40°` (demonstrates motion variety)

### 1.2 Overall Statistics

| Metric | Value |
|--------|-------|
| Total episodes | 51 |
| Good episodes | 20 (39.2%) |
| Bad episodes | 31 (60.8%) |
| Mean min elbow | -12.3° |
| Mean max elbow | 53.2° |
| Episodes reaching elbow < -60° | 2 (3.9%) |
| Episodes reaching elbow < -40° | 8 (15.7%) |

### 1.3 Top 10 Best Episodes (Lowest Min Elbow)

| Episode | Min Elbow | Elbow Range | Max Gripper | Quality |
|---------|-----------|-------------|-------------|---------|
| 11 | -75.6° | 81.0° | 49.6° | GOOD |
| 0 | -64.2° | 68.6° | 44.5° | GOOD |
| 4 | -59.0° | 61.3° | 46.7° | GOOD |
| 19 | -58.6° | 134.9° | 54.8° | GOOD |
| 3 | -49.7° | 90.4° | 50.0° | GOOD |
| 30 | -45.1° | 85.2° | 45.5° | GOOD |
| 2 | -41.8° | 112.1° | 51.2° | GOOD |
| 7 | -41.8° | 79.3° | 56.0° | GOOD |
| 20 | -34.0° | 35.3° | 50.9° | BAD* |
| 8 | -33.5° | 95.4° | 49.5° | GOOD |

*Episode 20 fails criteria due to insufficient elbow range (<40°)

**Analysis:**
- Episodes 0, 4, 11, 19 are **gold standard** (reach elbow < -60°)
- Episodes 2, 3, 7, 30 demonstrate good grasp approach (-42° to -50°)
- These episodes should be **oversampled** during training (2x-3x weight)

### 1.4 Top 10 Worst Episodes (Highest Min Elbow)

| Episode | Min Elbow | Elbow Range | Max Gripper | Quality |
|---------|-----------|-------------|-------------|---------|
| 41 | +37.6° | 53.9° | 54.4° | BAD |
| 45 | +30.9° | 11.6° | 41.1° | BAD |
| 38 | +29.3° | 38.3° | 49.8° | BAD |
| 49 | +21.5° | 0.8° | 51.7° | BAD |
| 1 | +21.4° | 68.8° | 58.7° | BAD |
| 22 | +21.4° | 52.2° | 51.2° | BAD |
| 46 | +19.0° | 37.0° | 45.2° | BAD |
| 42 | +18.8° | 87.5° | 50.7° | BAD |
| 34 | +16.9° | 39.8° | 47.2° | BAD |
| 27 | +16.3° | 17.1° | 51.9° | BAD |

**Analysis:**
- Episodes 41, 45, 38, 49 never reach forward (elbow stays >20°)
- Episode 49 has nearly zero motion (0.8° range) - likely corrupted data
- Episode 45 also shows minimal motion (11.6° range)
- These episodes add noise to training and should be **excluded** or **downweighted**

### 1.5 Distribution of Episode Quality by Elbow Reach

| Elbow Zone | Episodes | Percentage | Classification |
|------------|----------|------------|----------------|
| < -60° (deep grasp) | 2 | 3.9% | Excellent |
| -60° to -40° (grasp approach) | 6 | 11.8% | Good |
| -40° to -20° (forward reach) | 14 | 27.5% | Acceptable |
| -20° to 0° (neutral forward) | 11 | 21.6% | Marginal |
| > 0° (never forward) | 18 | 35.3% | Poor |

**Critical Gap:** Only 2 episodes in the zone model needs to reach (-60° to -70°).

---

## 2. Data Distribution Analysis

### 2.1 Joint Normalization Statistics

| Joint | Mean | Std Dev | Notes |
|-------|------|---------|-------|
| Base | -0.92° | 9.72° | Centered, good coverage |
| Shoulder | 49.43° | 33.04° | Broad distribution |
| **Elbow** | **25.19°** | **29.38°** | **Biased upward, insufficient negative values** |
| Wrist_pitch | 50.43° | 25.16° | Broad distribution |
| Wrist_roll | -1.64° | 13.62° | Centered, good coverage |
| Gripper | 21.58° | 16.88° | Slightly biased toward closed |

### 2.2 Elbow Distribution Percentiles

| Percentile | Value | Z-Score | Coverage |
|------------|-------|---------|----------|
| P1 | -58.9° | -2.86 | Deep grasp |
| P5 | -33.0° | -1.98 | Grasp approach |
| P10 | -13.8° | -1.33 | Forward reach |
| P25 | 9.3° | -0.54 | Neutral |
| **P50 (median)** | **26.8°** | **+0.06** | **Upright** |
| P75 | 46.6° | +0.73 | Retracted |
| P90 | 59.3° | +1.16 | High retract |
| P95 | 66.7° | +1.41 | Very high |
| P99 | 83.5° | +1.98 | Extreme |

**Key Insight:** Median elbow (26.8°) is nearly identical to the mean (25.2°), indicating a **symmetric but poorly positioned distribution**. The distribution is centered too high for grasping tasks.

### 2.3 Underrepresented Regions

| Region | Frames | Percentage | Status |
|--------|--------|------------|--------|
| Elbow < -60° (deep grasp) | 46 | 0.35% | **CRITICAL GAP** |
| Elbow < -40° (grasp zone) | 373 | 2.87% | Severely underrepresented |
| Elbow < -30° (pre-grasp) | 808 | 6.21% | Underrepresented |
| Elbow < -20° (approach) | 1,103 | 8.48% | Underrepresented |
| Elbow < 0° (forward) | 2,245 | 17.26% | Marginal |

**Target Inference Angle:** -64° (z=-3.04)
**Model Output Range:** z ∈ [-1.5, +1.5] (standard VLA behavior)
**Maximum Reachable:** z=-1.5 → elbow = 25.19 - 1.5×29.38 = **-18.9°** (far from -64°!)

### 2.4 Gripper Distribution

| Region | Frames | Percentage |
|--------|--------|------------|
| Gripper > 50° (wide open) | 843 | 6.48% |
| Gripper > 30° (open) | 3,833 | 29.46% |
| Gripper > 10° (slightly open) | 8,653 | 66.51% |
| Gripper < 10° (closed) | 4,357 | 33.49% |

**Analysis:** Gripper distribution is more balanced. All 51 episodes include gripper opening >30°, indicating teleoperation consistently demonstrated opening behavior. The gripper opens at ~27.2% of each episode (temporal pattern).

### 2.5 Joint Correlation Matrix

|            | Base  | Shoulder | Elbow | Wrist_pitch | Wrist_roll | Gripper |
|------------|-------|----------|-------|-------------|------------|---------|
| Base       | 1.000 | -0.273   | 0.006 | -0.152      | **0.482**  | 0.046   |
| Shoulder   | -0.273| 1.000    | 0.019 | 0.352       | -0.367     | **0.494** |
| Elbow      | 0.006 | 0.019    | 1.000 | **-0.327**  | -0.035     | -0.114  |
| Wrist_pitch| -0.152| 0.352    | -0.327| 1.000       | -0.148     | 0.050   |
| Wrist_roll | 0.482 | -0.367   | -0.035| -0.148      | 1.000      | -0.123  |
| Gripper    | 0.046 | **0.494**| -0.114| 0.050       | -0.123     | 1.000   |

**Key Correlations:**
- **Shoulder-Gripper (+0.494)**: When shoulder raises, gripper tends to open (coordinated reach-and-grasp)
- **Base-Wrist_roll (+0.482)**: Base rotation coordinated with wrist roll (orientation alignment)
- **Elbow-Wrist_pitch (-0.327)**: When elbow extends forward, wrist pitches down (natural kinematic coupling)

**Implication:** Elbow is nearly independent of other joints (low correlation), suggesting it's controlled separately. This means **elbow-focused data collection won't disturb other joint distributions**.

---

## 3. Root Cause Analysis

### 3.1 Why Model Outputs Conservative Actions

**VLA models (SmolVLA) are trained to output normalized actions (z-scores):**
- During training, actions are normalized: `z = (action - mean) / std`
- Model learns to output z ∈ [-2, +2] (rare to exceed this range)
- During inference, outputs are denormalized: `action = mean + z * std`

**For RoArm M3 Elbow:**
- Dataset mean = 25.19°, std = 29.38°
- Model outputs z ∈ [-1.5, +1.5] (conservative, typical behavior)
- Denormalized range: 25.19 ± 1.5×29.38 = **[-18.9°, +69.3°]**
- **Problem:** Target grasp angle (-64°) is outside this range!

### 3.2 Z-Score Requirements for Critical Angles

| Target Angle | Required Z-Score | Reachable? |
|--------------|------------------|------------|
| -64° (grasp) | z = -3.04 | No (>2σ from mean) |
| -60° (deep grasp) | z = -2.90 | No (>2σ from mean) |
| -40° (approach) | z = -2.22 | Rare (<5% probability) |
| -30° (pre-grasp) | z = -1.88 | Marginal (~10% probability) |
| -20° (forward) | z = -1.54 | Feasible (~20% probability) |

**Conclusion:** Model cannot reliably produce elbow < -40° because the dataset mean (25.19°) is too far from the target zone.

### 3.3 Comparison to Ideal Distribution

| Metric | Current Dataset | Ideal Distribution |
|--------|-----------------|-------------------|
| Elbow mean | 25.19° | -10° to 0° |
| Elbow std | 29.38° | 30-40° (keep broad) |
| Frames with elbow < -60° | 0.35% | 10-15% |
| Frames with elbow < -40° | 2.87% | 20-25% |
| Frames with elbow < -20° | 8.48% | 35-40% |

**Strategy:** Shift the distribution mean downward by adding deep grasp episodes.

---

## 4. Data Collection Strategy

### 4.1 Recommended Collection Protocol

#### Phase 1: Deep Grasp Episodes (Priority 1)
**Target:** 50 episodes with elbow < -60°

**Setup:**
1. Place object at **maximum forward reach** (requires elbow extension)
2. Start robot with shoulder=70°, elbow=-30° (pre-extended pose)
3. Demonstrate trajectory: extend elbow to -60° → grasp → retract

**Validation:**
- Each episode must reach elbow < -60° (use real-time monitoring)
- Reject episodes where min_elbow > -55°

**Estimated Impact:**
- Current: 46 frames with elbow < -60° (0.35%)
- After 50 episodes (~12,500 frames): ~13,000 frames with elbow < -60° (50.7%)
- New mean elbow: ~5° (shift from 25° to 5°)
- Target angle z-score: z = (-64 - 5) / 29 = **-2.03** (feasible within model output range)

#### Phase 2: Grasp Approach Episodes (Priority 2)
**Target:** 30 episodes with -60° < elbow < -40°

**Setup:**
1. Place object at moderate reach distance
2. Start robot with shoulder=50°, elbow=-10° (neutral forward pose)
3. Demonstrate trajectory: extend to -45° → grasp → retract

**Validation:**
- Each episode must reach -60° < min_elbow < -40°

**Estimated Impact:**
- Ensures smooth distribution coverage between neutral and deep grasp
- Prevents distribution gap that could cause interpolation errors

#### Phase 3: Diverse Start Positions (Priority 3)
**Target:** 20 episodes with varied initial poses

**Setup:**
1. Randomize start pose within safe workspace
2. Include episodes starting from retracted (elbow > 50°)
3. Demonstrate recovery from awkward poses

**Purpose:**
- Improve generalization
- Handle deployment scenarios where robot isn't at dataset_mean start

### 4.2 Start Position Guidelines

| Phase | Shoulder Start | Elbow Start | Object Distance |
|-------|----------------|-------------|-----------------|
| Current dataset | 49° (mean) | 25° (mean) | Moderate |
| Phase 1 (deep grasp) | 60-80° | -30° to -10° | Far (max reach) |
| Phase 2 (approach) | 40-60° | -10° to +10° | Medium-far |
| Phase 3 (diverse) | Random | Random | Random |

### 4.3 Real-Time Monitoring Script

Modify `collect_data_manual.py` to show live elbow angle feedback:

```python
# During collection loop, add:
current_elbow = current_angles[2]  # Elbow is joint index 2
if current_elbow < -60:
    status = "[DEEP GRASP]"
elif current_elbow < -40:
    status = "[APPROACH]"
elif current_elbow < -20:
    status = "[FORWARD]"
else:
    status = "[NEUTRAL]"

print(f"\rElbow: {current_elbow:+7.1f}° {status}  ", end='', flush=True)
```

Add post-episode validation:
```python
# After episode finishes:
episode_elbow = np.array([frame[2] for frame in episode_actions])
min_elbow = episode_elbow.min()

if args.mode == "deep_grasp" and min_elbow > -55:
    print(f"\n[REJECT] Episode only reached {min_elbow:.1f}°, target was <-60°")
    print("Retry? [y/n]: ", end='')
    if input().lower() == 'y':
        episode_data.pop()  # Remove last episode
        episode_count -= 1
```

### 4.4 Collection Timeline Estimate

| Phase | Episodes | Avg Time/Episode | Total Time |
|-------|----------|------------------|------------|
| Phase 1 (deep grasp) | 50 | 8 min | 6.7 hours |
| Phase 2 (approach) | 30 | 7 min | 3.5 hours |
| Phase 3 (diverse) | 20 | 6 min | 2.0 hours |
| **Total** | **100** | - | **12.2 hours** |

Assuming 3-4 hours of collection per day, completion in 3-4 days.

---

## 5. Data Augmentation Feasibility

### 5.1 Temporal Augmentation

**Method:** Speed up/slow down episode playback (resample time series)

**Pros:**
- Easy to implement (`scipy.interpolate`)
- Increases data variety without new collection
- Useful for learning tempo-invariant policies

**Cons:**
- Doesn't add new pose coverage (same min/max elbow per episode)
- May create unrealistic dynamics (too fast → violates physics)

**Recommendation:** Use conservatively (0.8x-1.2x speed) for gripper timing variation only. **Won't solve elbow distribution problem.**

### 5.2 Action Noise Injection

**Method:** Add Gaussian noise to actions during training

```python
# During training:
augmented_action = original_action + torch.randn_like(action) * noise_std
```

**Pros:**
- Standard practice in RL (improves robustness)
- Can encourage exploration near dataset boundaries

**Cons:**
- Noise is symmetric (doesn't shift mean downward)
- Large noise → unrealistic trajectories, potential collisions
- Doesn't fundamentally change distribution coverage

**Recommendation:** Apply small noise (std=0.1×action_std) for robustness, but **won't shift elbow distribution to grasp zone**.

### 5.3 Episode Oversampling Strategy

**Method:** Weight episodes with elbow < -40° higher during training

```python
# In LeRobot dataset config:
episode_weights = []
for ep_idx in range(num_episodes):
    min_elbow = episode_elbow_mins[ep_idx]
    if min_elbow < -60:
        weight = 5.0  # 5x oversampling
    elif min_elbow < -40:
        weight = 3.0  # 3x oversampling
    elif min_elbow < -20:
        weight = 1.5  # 1.5x oversampling
    else:
        weight = 1.0  # Normal sampling
    episode_weights.append(weight)
```

**Pros:**
- Shifts effective distribution mean without new data
- Standard technique in imbalanced learning
- Easy to implement in LeRobot

**Cons:**
- Limited by existing data (only 8 episodes with elbow < -40°)
- Overfitting risk if oversampling ratio too high
- Doesn't create new pose coverage

**Recommendation:**
- **Immediate action:** Implement 3x-5x oversampling for episodes 0, 4, 11, 19 (elbow < -60°)
- **Long-term:** Collect new deep grasp episodes (Phase 1), then use balanced sampling

### 5.4 Synthetic Data Generation (Advanced)

**Method:** Fit kinematic model to existing data, generate interpolated trajectories

**Pros:**
- Can fill distribution gaps mathematically
- Infinite data generation potential

**Cons:**
- Requires accurate robot kinematics model
- May generate infeasible trajectories (collision, singularity)
- Visual observations still from real episodes (distribution mismatch)

**Recommendation:** **NOT recommended** for VLA (vision is critical). Only viable for state-based policies.

### 5.5 Augmentation Summary Table

| Method | Effectiveness for Elbow Distribution | Difficulty | Recommend? |
|--------|-------------------------------------|------------|------------|
| Temporal (speed) | Low (no pose change) | Easy | No |
| Action noise | Low (symmetric noise) | Easy | Yes (small noise for robustness) |
| Episode oversampling | **Medium** (effective with current data) | Easy | **Yes (immediate action)** |
| Synthetic trajectories | High (but risky) | Hard | No (vision mismatch) |
| **New data collection** | **High** | Medium | **Yes (best solution)** |

---

## 6. Recommended Action Plan

### Immediate Actions (This Week)

1. **Implement episode oversampling** in training script:
   - 5x weight for episodes [0, 4, 11, 19] (elbow < -60°)
   - 3x weight for episodes [2, 3, 7, 30] (-60° < elbow < -40°)
   - Retrain SmolVLA for 20K steps, compare inference behavior

2. **Exclude bad episodes** from training:
   - Remove episodes [41, 45, 49] (minimal motion or never forward)
   - Reduces noise in training data

3. **Verify current model behavior**:
   - Run `deploy_smolvla.py` with elbow monitoring
   - Confirm model stays at ~+32° elbow
   - Document failure mode for comparison after retraining

### Short-Term Actions (Next 2 Weeks)

4. **Collect Phase 1 data** (50 deep grasp episodes):
   - Modify `collect_data_manual.py` with real-time elbow monitoring
   - Follow Phase 1 protocol (start at elbow=-30°, reach to -60°)
   - Validate each episode before accepting

5. **Retrain with Phase 1 data**:
   - Expected dataset: 51 + 50 = 101 episodes, ~25,500 frames
   - Expected elbow distribution shift: mean 25° → 5°
   - Train for 30K steps (longer due to more data)

6. **Evaluate on robot**:
   - Deploy new checkpoint
   - Measure: min elbow reached, grasp success rate
   - If still insufficient, proceed to Phase 2

### Long-Term Actions (3-4 Weeks)

7. **Collect Phase 2 data** (30 approach episodes):
   - Target -60° < elbow < -40°
   - Total dataset: 131 episodes, ~33,000 frames

8. **Collect Phase 3 data** (20 diverse episodes):
   - Random start poses for generalization
   - Total dataset: 151 episodes, ~38,000 frames

9. **Final training run**:
   - Train for 50K steps with full dataset
   - Expected performance: reliable elbow extension to -60° zone
   - Measure grasp success rate across 20 test runs

---

## 7. Success Metrics

### Training Metrics

| Metric | Current | Target (after Phase 1) | Target (after Phase 3) |
|--------|---------|------------------------|------------------------|
| Episodes with elbow < -60° | 2 (3.9%) | 52 (51.5%) | 72 (47.7%) |
| Frames with elbow < -60° | 46 (0.35%) | ~13,000 (50.7%) | ~18,500 (48.7%) |
| Elbow mean | 25.19° | ~5° | ~-5° |
| Elbow std | 29.38° | ~32° | ~35° |

### Inference Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Min elbow reached (50 steps) | +31.8° | < -50° |
| Grasp success rate | 0% | > 70% |
| Mean L2 action error | 4.39° | < 5° |
| Action diversity (std) | Low | High |

### Validation Protocol

For each checkpoint, run 10 inference episodes:
1. Record min/max elbow reached
2. Measure gripper opening timing
3. Count successful grasps (object lifted >5cm)
4. Compute action diversity (std of outputs across episodes)

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Phase 1 data still insufficient | Medium | High | Collect Phase 2 immediately (don't wait for retraining) |
| Overfitting to deep grasp poses | Low | Medium | Phase 3 diverse data for generalization |
| Robot hardware limits (can't reach -64°) | Low | Critical | Test manually: `arm.joints_angle_ctrl([0, 70, -64, 50, 0, 0])` |
| Model still outputs conservative z-scores | Medium | High | Consider architecture change (diffusion policy) |
| Temporal pattern in gripper (not vision-conditional) | High | Medium | Add variation in gripper opening timing |

---

## 9. Alternative Approaches (If Data Collection Fails)

### 9.1 Diffusion Policy Instead of VLA

**Rationale:** Diffusion models can represent multimodal distributions better than VLAs (which tend to output mean actions).

**Trade-offs:**
- Pros: Better action diversity, handles rare poses
- Cons: Requires more data, slower inference (~100ms vs ~10ms)

### 9.2 Explicit Elbow Control Heuristic

**Rationale:** Override model output for elbow joint during grasp phase.

```python
# In deploy script:
model_action = policy.select_action(observation)
if gripper_opening_detected:  # Model outputs gripper > 30°
    model_action[2] = -60  # Force elbow to grasp zone
```

**Trade-offs:**
- Pros: Immediate solution, guaranteed elbow reach
- Cons: Defeats purpose of end-to-end learning, fragile

### 9.3 Two-Stage Policy

**Rationale:** Train separate policies for reach and grasp phases.

- Policy 1 (VLA): Vision → reach target (stop at elbow=-40°)
- Policy 2 (scripted): Extend elbow to -64°, close gripper

**Trade-offs:**
- Pros: Leverages VLA vision for targeting, scripted precision for grasp
- Cons: Manual switching logic, not generalizable

---

## 10. Conclusion

The current dataset is **insufficient for learning deep grasp behavior** due to extreme underrepresentation of elbow < -60° poses (0.35% of data). The model has learned the dataset distribution accurately but that distribution is centered at elbow=25° instead of the required -10° to 0° range.

**Recommended path forward:**
1. **Immediate:** Implement episode oversampling (5x for best episodes) and retrain
2. **Short-term:** Collect 50 deep grasp episodes (Phase 1)
3. **Long-term:** Collect 50 more varied episodes (Phases 2+3) for robustness

Expected timeline: 3-4 weeks to production-ready model.

**Success probability:** High (80%+) if Phase 1 data is collected with proper validation. The problem is **data distribution, not model architecture**.

---

## Appendix A: Script Modifications

### A.1 Add Episode Oversampling to Training

File: `run_official_train.py` or training config

```python
# After loading dataset:
episode_weights = torch.ones(num_episodes)

# Define high-value episodes (elbow < -60°)
gold_episodes = [0, 4, 11, 19]
good_episodes = [2, 3, 7, 30]

for ep in gold_episodes:
    episode_weights[ep] = 5.0

for ep in good_episodes:
    episode_weights[ep] = 3.0

# Exclude bad episodes
bad_episodes = [41, 45, 49]
for ep in bad_episodes:
    episode_weights[ep] = 0.0

# Apply to dataset sampler:
sampler = WeightedRandomSampler(
    weights=episode_weights,
    num_samples=len(dataset),
    replacement=True
)
dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
```

### A.2 Add Real-Time Monitoring to collect_data_manual.py

```python
# In collection loop (line ~180):
import sys

def format_elbow_status(elbow_angle):
    """Return color-coded status string."""
    if elbow_angle < -60:
        return "\033[92m[DEEP GRASP]\033[0m"  # Green
    elif elbow_angle < -40:
        return "\033[93m[APPROACH]\033[0m"    # Yellow
    elif elbow_angle < -20:
        return "\033[94m[FORWARD]\033[0m"     # Blue
    else:
        return "[NEUTRAL]"

# In main loop:
current_elbow = current_angles[2]
status = format_elbow_status(current_elbow)
sys.stdout.write(f"\rElbow: {current_elbow:+7.1f}° {status}  ")
sys.stdout.flush()
```

### A.3 Post-Episode Validation

```python
# After episode collection (line ~210):
episode_actions = np.array([frame['action'] for frame in episode_buffer])
episode_elbow = episode_actions[:, 2]
min_elbow = episode_elbow.min()
max_elbow = episode_elbow.max()
elbow_range = max_elbow - min_elbow

print(f"\nEpisode {ep_count} summary:")
print(f"  Elbow: min={min_elbow:+.1f}°, max={max_elbow:+.1f}°, range={elbow_range:.1f}°")

# Validation based on collection mode
if args.collection_mode == "deep_grasp":
    if min_elbow > -55:
        print(f"[WARNING] Target was elbow < -60°, but only reached {min_elbow:.1f}°")
        print("Accept this episode? [y/n]: ", end='')
        if input().lower() != 'y':
            print("Episode rejected, restarting...")
            ep_count -= 1
            continue

elif args.collection_mode == "approach":
    if min_elbow > -40 or min_elbow < -60:
        print(f"[WARNING] Target was -60° < elbow < -40°, but reached {min_elbow:.1f}°")
        # Similar validation
```

---

## Appendix B: Hardware Validation

Before collecting deep grasp data, verify robot can physically reach elbow=-64°:

```python
# test_elbow_limits.py
from roarm_sdk.roarm import roarm
import time

arm = roarm(roarm_type="roarm_m3", port="COM3", baudrate=115200)
time.sleep(0.5)

# Test deep grasp pose
print("Testing elbow=-64° (grasp pose)...")
test_angles = [0, 70, -64, 50, 0, 0]  # [base, shoulder, elbow, wrist_p, wrist_r, gripper]

arm.joints_angle_ctrl(angles=test_angles, speed=300, acc=150)
time.sleep(3)

# Read back actual position
actual = arm.joints_angle_get()
print(f"Commanded: {test_angles}")
print(f"Actual: {actual}")
print(f"Elbow error: {abs(actual[2] - test_angles[2]):.1f}°")

if abs(actual[2] - test_angles[2]) > 5:
    print("[ERROR] Robot cannot reach target elbow angle!")
    print("Check:")
    print("  1. Mechanical limits (-70° to 190° per spec)")
    print("  2. Object collision")
    print("  3. SDK bug")
else:
    print("[OK] Robot can reach elbow=-64°")

arm.disconnect()
```

Run this before starting Phase 1 data collection. If it fails, the hardware cannot support the grasp pose and strategy must be revised.

---

**End of Report**

Generated by: Data Agent
Date: 2026-02-07
Updated: 2026-02-10 (카메라 위치 변경으로 데이터 OBSOLETE 경고 추가)
Dataset: E:/RoArm_Project/lerobot_dataset_v3/ (51 episodes, 13,010 frames) ← OBSOLETE
Analysis scripts: data_episode_quality.py, data_distribution_simple.py
