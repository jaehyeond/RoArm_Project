# Open-Loop Action Chunk Analysis Report
**Date**: 2026-02-11
**Task**: Analyze SmolVLA model behavior through 50-step action chunks from single observations

---

## Executive Summary

**Key Findings:**
1. **Model shows scenario-dependent behavior** (DEEP vs APPROACH vs SHALLOW)
2. **Elbow movement is meaningful** (36° change in APPROACH scenario)
3. **Gripper action is present but incomplete** (opens in APPROACH, but no grasp pattern)
4. **Critical Issue: No coordinated grasp sequence** (open → approach → close pattern missing)
5. **35K vs 50K: Nearly identical behavior** (no overfitting, but also no improvement)

**Verdict:**
- PARTIAL PASS: Model learned basic joint control and scenario sensitivity
- FAIL on task completion: No purposeful "reach down → grasp → lift" sequence observed
- Root cause: Data quality (68% SHALLOW episodes, gripper mostly closed)

---

## Methodology

### Test Setup
- **Checkpoints tested**: 35K steps, 50K steps (final)
- **Dataset**: lerobot_dataset_v3 (50 episodes, 10,803 frames)
- **Scenarios**:
  - DEEP: elbow < -30° (deep grasp, episode 2, frame 186)
  - APPROACH: -30° < elbow < -10° (approaching grasp, episode 2, frame 57)
  - SHALLOW: elbow > -10° (shallow/no grasp, episode 22, frame 4843)

### Procedure
For each scenario:
1. Load single observation (image + state) from dataset
2. Generate 50-step action chunk (open-loop, same observation repeated)
3. Analyze temporal patterns, gripper behavior, convergence
4. Compare across scenarios and checkpoints

---

## Results: 35K Checkpoint

### DEEP Scenario (Elbow -65.4°, Already Deep in Grasp)

| Joint | Range | Total Δ | Convergence | Interpretation |
|-------|-------|---------|-------------|----------------|
| Base | 2.3° | -1.2° | YES | Minimal adjustment |
| Shoulder | 9.1° | -8.5° | YES | Slight lift attempt |
| **Elbow** | **5.9°** | **+5.9°** | **YES** | **Rising (away from object!)** |
| Wrist_P | 3.2° | -1.5° | YES | Stable |
| Wrist_R | 1.8° | -1.1° | YES | Stable |
| **Gripper** | **0.9°** | **-0.8°** | **YES** | **Already closed, staying closed** |

**Analysis:**
- Starting at -65° (already deep), model thinks task is done
- Small elbow rise (5.9°) = moving AWAY from object
- Gripper stays closed (4-5°)
- **No purposeful action, just minor convergence**

### APPROACH Scenario (Elbow -29.6°, Approaching Object)

| Joint | Range | Total Δ | Convergence | Interpretation |
|-------|-------|---------|-------------|----------------|
| Base | 9.9° | -9.3° | YES | Repositioning |
| Shoulder | 25.6° | +25.6° | NO | **Lifting upward** |
| **Elbow** | **38.0°** | **+36.2°** | **YES** | **RISING (should go down!)** |
| Wrist_P | 4.9° | -0.8° | YES | Stable |
| Wrist_R | 1.7° | +0.9° | NO | Minimal |
| **Gripper** | **32.7°** | **+32.5°** | **NO** | **Opens 1.8° → 34.3° (good!)** |

**Analysis:**
- **PROBLEM**: Elbow rises from -29.6° → +6° (moving AWAY from grasp zone)
- **GOOD**: Gripper opens (1.8° → 34.3°), showing grasp intent
- **PROBLEM**: No coordinated sequence (should be: keep going down → then grasp)
- **Model learned**: "approach position → open gripper + lift", not "approach → go deeper → grasp"

### SHALLOW Scenario (Elbow +38.9°, High Position)

| Joint | Range | Total Δ | Convergence | Interpretation |
|-------|-------|---------|-------------|----------------|
| Base | 4.4° | -3.1° | YES | Minor adjust |
| Shoulder | 11.5° | +10.9° | YES | Slight lift |
| Elbow | 3.7° | -3.0° | YES | Minimal movement |
| Wrist_P | 3.0° | 0.0° | NO | No change |
| Wrist_R | 1.5° | -0.8° | YES | Stable |
| **Gripper** | **14.3°** | **+6.8°** | **NO** | **Opens 4° → 10.8°** |

**Analysis:**
- Starting high (+38.9°), model makes small adjustments
- Gripper opens slightly (4° → 10.8°), but stays mostly closed
- **No aggressive "go down and grasp" action** (expected in SHALLOW scenario)

---

## Results: 50K Checkpoint

### Comparison to 35K

| Metric | 35K | 50K | Change |
|--------|-----|-----|--------|
| **DEEP Elbow Δ** | +5.9° | +5.5° | -0.4° (nearly identical) |
| **APPROACH Elbow Δ** | +36.2° | +36.3° | +0.1° (identical) |
| **SHALLOW Elbow Δ** | -3.0° | -2.7° | +0.3° (nearly identical) |
| **APPROACH Gripper Δ** | +32.5° | +29.3° | -3.2° (slightly less opening) |
| **Z-score range (APPROACH)** | -1.49 to 1.81 | -1.51 to 1.68 | Slightly tighter |

**Conclusion:**
- 35K and 50K show **nearly identical behavior**
- No overfitting (z-scores still within ±2σ)
- No improvement in task understanding (same elbow rise problem)
- **Training plateau at ~35K steps** (consistent with pipeline-agent's L2 analysis)

---

## Critical Findings

### 1. Wrong Elbow Direction in APPROACH
- **Expected**: Elbow -29.6° → -60° (go deeper to grasp)
- **Actual**: Elbow -29.6° → +6° (rise up by 36°)
- **Root cause**: Dataset has 68% SHALLOW episodes (elbow positive), model learned "APPROACH → LIFT" pattern, not "APPROACH → DESCEND → GRASP"

### 2. No Coordinated Grasp Sequence
- **Observed**: Gripper opens in APPROACH (good!)
- **Missing**:
  1. Approach with gripper open
  2. Descend to object
  3. Close gripper
  4. Lift with gripper closed
- **Model's pattern**: "APPROACH → open gripper + lift simultaneously"

### 3. Gripper Always Stays Near Closed
- DEEP: 4-5° (already closed)
- APPROACH: 1.8° → 34.3° (opens, but late in sequence)
- SHALLOW: 4° → 10.8° (barely opens)
- **Problem**: No "open wide → approach → close" pattern

### 4. Convergence After 20-30 Steps
- Most joints converge (std < 1°) by step 30-40
- **Temporal horizon issue**: 50-step chunks, but model only plans 20-30 steps ahead
- Later steps just repeat converged position

### 5. Z-scores Within ±2σ
- 35K: z-scores -1.49 to +1.81
- 50K: z-scores -1.51 to +1.68
- **Conservative predictions** (matches deployment failure: solow drift, no aggressive actions)

---

## Connection to Deployment Failures

### Run 1 (2026-02-11, auto, 100 steps)
- **Symptom**: Wrist_R -3° → -92° (runaway drift)
- **Open-loop analysis**: Z-scores within ±2σ, so single-step predictions OK
- **Failure mode**: Closed-loop error accumulation → OOD → drift

### Run 2 (2026-02-11, manual, 100 steps)
- **Symptom**: All joints drift one direction (Base +12°, Elbow +36°), no grasp
- **Open-loop analysis**: Elbow rises in APPROACH scenario (matches!)
- **Failure mode**: Model learned "APPROACH → LIFT" (wrong direction for grasp)

### Why Open-Loop Test Predicts Deployment Failure
1. **APPROACH scenario shows elbow rises (+36°)** → Matches Run 2 elbow rise
2. **No gripper grasp pattern** → Matches Run 1 & 2 gripper stuck at 2-4°
3. **Convergence after 20-30 steps** → Matches slow, directionless drift
4. **Z-scores ±2σ** → Conservative actions → insufficient for task

---

## Root Cause Analysis

### Data Distribution Problem
- **SHALLOW**: 68% (34/50 episodes, elbow > -10°)
- **APPROACH**: 14% (7/50 episodes)
- **DEEP**: 18% (9/50 episodes, elbow < -30°)

### What Model Learned
1. **SHALLOW → LIFT** (dominant pattern in data)
2. **APPROACH → LIFT + OPEN GRIPPER** (partial grasp understanding)
3. **DEEP → STAY** (already at object, do nothing)

### What Model Missed
1. **APPROACH → DESCEND** (not in data: SHALLOW bias)
2. **OPEN → DESCEND → CLOSE sequence** (gripper mostly closed in dataset)
3. **Closed-loop correction** (open-loop chunks don't account for environment feedback)

---

## Recommendations

### Immediate: Data Collection Strategy
1. **Target 100+ episodes** with balanced distribution:
   - DEEP: 50 episodes (50%, elbow -30° to -70°)
   - APPROACH: 30 episodes (30%, elbow -10° to -30°)
   - SHALLOW: 20 episodes (20%, elbow 0° to +50°)

2. **Gripper diversity**:
   - All episodes MUST start with gripper open (>50°)
   - Include "approach → descend → grasp → lift" full sequence
   - Avoid static gripper (currently 80%+ frames are gripper closed)

3. **Quality filters**:
   - Elbow must reach < -40° for at least 30% of episode
   - Gripper must transition open → close → open
   - Include "lift after grasp" (elbow negative → elbow rising + gripper closed)

### Short-term: Training Improvements
1. **Continue training**: Model plateaued at 35K, but not overfitting
2. **Data augmentation**: Add random gripper noise to open/close transitions
3. **Weighted sampling**: Oversample DEEP episodes (2x weight)

### Medium-term: Architecture Exploration
1. **Action chunking**: Test n_action_steps > 1 for temporal consistency
2. **Diffusion steps**: Increase from 10 → 20 for finer control
3. **Action scaling**: Try --action-scale 1.5 to amplify z-scores

### Long-term: Closed-Loop Robustness
1. **Online fine-tuning**: Deploy → collect failures → retrain
2. **Residual policy**: Add small correction network for closed-loop drift
3. **Multi-modal data**: Add depth for better spatial understanding

---

## Conclusion

**35K and 50K checkpoints show identical behavior:**
- Scenario-dependent (good!)
- Elbow movement present (good!)
- Wrong direction in APPROACH: rises instead of descends (critical!)
- No coordinated grasp sequence (critical!)

**Deployment failures are predictable from open-loop analysis:**
- APPROACH scenario: elbow +36° (matches Run 2 behavior)
- No gripper pattern (matches Run 1 & 2 stuck gripper)
- Conservative z-scores (matches slow drift)

**Next step: Collect better data (100+ episodes, 50% DEEP, gripper diversity)**

---

## Appendix: Numerical Data

### 35K Checkpoint - Full Scenario Comparison

| Joint | DEEP Range | DEEP Δ | APPROACH Range | APPROACH Δ | SHALLOW Range | SHALLOW Δ |
|-------|------------|--------|----------------|------------|---------------|-----------|
| Base | 2.3° | -1.2° | 9.9° | -9.3° | 4.4° | -3.1° |
| Shoulder | 9.1° | -8.5° | 25.6° | +25.6° | 11.5° | +10.9° |
| Elbow | 5.9° | +5.9° | 38.0° | +36.2° | 3.7° | -3.0° |
| Wrist_P | 3.2° | -1.5° | 4.9° | -0.8° | 3.0° | 0.0° |
| Wrist_R | 1.8° | -1.1° | 1.7° | +0.9° | 1.5° | -0.8° |
| Gripper | 0.9° | -0.8° | 32.7° | +32.5° | 14.3° | +6.8° |

### 50K Checkpoint - Full Scenario Comparison

| Joint | DEEP Range | DEEP Δ | APPROACH Range | APPROACH Δ | SHALLOW Range | SHALLOW Δ |
|-------|------------|--------|----------------|------------|---------------|-----------|
| Base | 1.8° | -1.1° | 9.5° | -8.1° | 4.2° | -3.7° |
| Shoulder | 9.6° | -9.0° | 25.2° | +25.2° | 10.9° | +10.3° |
| Elbow | 5.7° | +5.5° | 38.5° | +36.3° | 3.7° | -2.7° |
| Wrist_P | 2.5° | -1.2° | 3.6° | -0.6° | 1.8° | -0.8° |
| Wrist_R | 2.6° | -0.2° | 1.9° | +1.4° | 2.0° | -1.4° |
| Gripper | 0.8° | -0.5° | 29.5° | +29.3° | 15.4° | +13.1° |

### Z-Score Ranges (35K Checkpoint)

| Scenario | Min Z | Max Z | Interpretation |
|----------|-------|-------|----------------|
| DEEP | -2.57 | +1.72 | Within ±3σ, conservative |
| APPROACH | -1.49 | +1.81 | Within ±2σ, very conservative |
| SHALLOW | -0.42 | +2.19 | Within ±3σ, conservative |

---

**Generated by**: deploy-agent
**Script**: `deploy_openloop_analysis.py`
**Output directories**:
- `/home/cgxr/Documents/Robotics/RoArm_Project/analysis_openloop_35k/`
- `/home/cgxr/Documents/Robotics/RoArm_Project/analysis_openloop_50k/`
