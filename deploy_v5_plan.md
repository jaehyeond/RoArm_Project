# deploy_v5_plan.md — SmolVLA v5 Deployment Plan
**Date**: 2026-03-30
**Model**: smolvla_v5_multipos (200K steps, 136 episodes, 5 zones)
**Script**: deploy_smolvla.py

---

## 1. Current deploy_smolvla.py Audit

### 1.1 Feature Checklist

| Feature | Status | Notes |
|---------|--------|-------|
| `--start-pos dataset_mean` | PRESENT | Supported. Default is `dataset_mean`. |
| `--checkpoint` arg | PRESENT | Configurable checkpoint path via CLI. |
| JOINT_LIMITS | PRESENT | List-of-tuples, correct hardware limits. |
| n_action_steps default | 5 | Closed-loop default. Open-loop uses 50-step chunks. |
| Open-loop `--open-loop` | PRESENT | Multi-chunk, re-observes at chunk boundaries. |
| `--n-chunks` | PRESENT | Default 4. |
| CSV logging | PRESENT | `--log-csv` or `--log-csv auto`. |
| EMA smoothing | PRESENT | `--ema-alpha`. Default 1.0 (off). |
| Convergence detection | PRESENT | `--convergence-threshold/window/action`. |
| Workspace safety (Z_FLOOR, DIST_MAX) | PRESENT | Z_FLOOR=-130mm, DIST_MAX=420mm. |
| Distal joint speed cap | PRESENT | JOINT_SPEED_CAPS=[500,500,500,300,300,300]. |

### 1.2 V5-Specific Issues Found

**Issue 1 — CHECKPOINT_PATH default is v3 (line 86)**
- Current default: `outputs/smolvla_v3_sponge/checkpoints/025000/pretrained_model`
- Must always pass `--checkpoint` explicitly for v5 runs.

**Issue 2 — DATASET_MEAN_POS is v3 values (line 91) — BUG FOR V5**
- Current value: `[0, 30, 59, 41, -2, 26]` (v3 mean)
- V5 actual mean from state.mean: `[10, 44, 41, 67, 0, 28]`
- Impact: `--start-pos dataset_mean` will move robot to v3 mean position,
  which is OOD for v5 (especially shoulder 30→44, wrist_pitch 41→67).
- This MUST be patched before using `--start-pos dataset_mean` with v5.
- Safe workaround: use `--start-pos current` after manually positioning arm.

**Issue 3 — Open-loop chunk count default too high for v5**
- V5 episodes: mean 99 frames (3.3s @ 30fps)
- Current default 4 chunks × 50 steps = 200 steps at 10Hz = 20s
- V5 episode equivalent: 2 chunks × 50 steps = 100 steps = 10s
- Recommended: `--n-chunks 2` for v5.

---

## 2. V5 Normalization Statistics (verified from checkpoint)

Source: `outputs/smolvla_v5_multipos/checkpoints/200000/pretrained_model/`
Note: 50K and 200K checkpoints have identical stats (dataset-derived, not step-dependent).

| Stat | Base | Shoulder | Elbow | WristPitch | WristRoll | Gripper |
|------|------|----------|-------|-----------|----------|---------|
| action.mean | 9.95 | 43.94 | 41.31 | 66.57 | 0.21 | 28.25 |
| action.std  | 30.74 | 16.13 | 32.53 | 29.13 | 26.36 | 20.24 |
| state.mean  | 9.93 | 44.10 | 40.94 | 67.18 | 0.20 | 28.08 |
| state.std   | 30.96 | 16.05 | 32.33 | 28.55 | 26.60 | 20.39 |

Gripper success criterion: 15-20° = sponge gripped (compliance prevents full close to 0°).

---

## 3. Deployment Commands

### Phase 0 — Dry-run verification (no robot motion)

```bash
conda activate roarm
cd /home/cgxr/Documents/Robotics/RoArm_Project

python deploy_smolvla.py \
  --checkpoint outputs/smolvla_v5_multipos/checkpoints/050000/pretrained_model \
  --dry-run \
  --start-pos current \
  --max-steps 5 \
  --log-csv auto
```

Expected: model loads, prints v5 normalization stats (~9.95/43.94/41.31/66.57/0.21/28.25),
5 inference steps printed, no robot motion.

### Phase 1 — 50K checkpoint (first real robot test)

Manually position robot to approximately [10, 44, 41, 67, 0, 28] before running.
This avoids the DATASET_MEAN_POS bug.

```bash
python deploy_smolvla.py \
  --checkpoint outputs/smolvla_v5_multipos/checkpoints/050000/pretrained_model \
  --start-pos current \
  --open-loop \
  --n-chunks 2 \
  --hz 10 \
  --log-csv auto \
  --convergence-action warn
```

### Phase 2 — 200K checkpoint (full training)

```bash
python deploy_smolvla.py \
  --checkpoint outputs/smolvla_v5_multipos/checkpoints/200000/pretrained_model \
  --start-pos current \
  --open-loop \
  --n-chunks 2 \
  --hz 10 \
  --log-csv auto \
  --convergence-action warn
```

### Alternative — Closed-loop (if open-loop motion is jerky)

```bash
python deploy_smolvla.py \
  --checkpoint outputs/smolvla_v5_multipos/checkpoints/200000/pretrained_model \
  --start-pos current \
  --n-action-steps 5 \
  --ema-alpha 0.6 \
  --max-steps 150 \
  --hz 10 \
  --log-csv auto \
  --convergence-action warn
```

---

## 4. Required Code Fix Before Real Runs

One change required in `deploy_smolvla.py`:

**Line 91 — Update DATASET_MEAN_POS to v5**
```python
# REPLACE:
DATASET_MEAN_POS = [0, 30, 59, 41, -2, 26]

# WITH:
DATASET_MEAN_POS = [10, 44, 41, 67, 0, 28]  # v5: state.mean rounded
```

Also recommended:
- Line 86: change CHECKPOINT_PATH default to `outputs/smolvla_v5_multipos/checkpoints/last/pretrained_model`
- Lines 88-92: update comments to reference v5 (136 episodes, lerobot_dataset_v5)

---

## 5. Safety Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Z_FLOOR_DEPLOY | -130mm | Table surface protection (measured -120mm) |
| DIST_MAX_DEPLOY | 420mm | Workspace boundary |
| speed | 500 | Default |
| acc | 200 | Default |
| JOINT_SPEED_CAPS distal | 300 | Prevents wrist runaway (v1 lesson) |

All safety parameters are appropriate for v5. No changes needed.

---

## 6. Checkpoint Selection Order

Test in this order:
1. **050000** — verifies basic motion learned. Fast fail signal.
2. **200000** — expect best. Full training.
3. **100000 / 150000** — only if 200K fails in a specific way that suggests overfit.

---

## 7. Success Criteria

| Test | Pass Condition |
|------|---------------|
| Dry-run | Model loads, stats show v5 values, no crash |
| Motion test | All joints move within first 20 steps |
| Approach | FK z < 150mm (arm reaches near table) |
| Grasp | Gripper closes to 15-20° with sponge resistance |
| Zone success | 60%+ across 5 zones (Stage 1 target) |

Gripper note: 15-20° = success with sponge. 0-5° = miss (no contact). Do not confuse.

---

## 8. Known Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| OOD start from wrong DATASET_MEAN_POS | HIGH if not patched | Use `--start-pos current` + manual position |
| Elbow dead zone stutter (42-60°) | MODERATE | Open-loop chunks commit through it |
| Gripper opens mid-air (v3 failure) | LOW — v5 has better depth coverage | Monitor FK z in CSV logs |
| Wrist_R runaway | LOW — speed cap active | JOINT_SPEED_CAPS[4]=300 |

---

## 9. Evaluation Protocol

Per zone (5 zones total), per checkpoint under test:
- Minimum 5 trials, 20 for reliable statistics
- Record: CSV log per run (`--log-csv auto`)
- Log path: `logs/deploy_YYYYMMDD_HHMMSS.csv`
- Success: sponge visually lifted 5cm+ from table
- Key CSV columns to review: `fk_z`, `gripper`, `z_gripper`, `convergence_detected`
