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
