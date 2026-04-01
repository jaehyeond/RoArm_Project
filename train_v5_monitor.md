# SmolVLA V5 Training Monitor Report

**Generated**: 2026-03-27 16:10 (monitoring session)
**Pipeline Agent** — Initial health check, steps 100-1100

---

## 1. Training Status: HEALTHY

No errors, no NaN/Inf values, no training failures. Only one benign warning:
- `RequestsDependencyWarning: urllib3 (2.6.1)` — harmless, unrelated to training

---

## 2. Configuration Verified

From log (confirmed matches `run_official_train.py`):

| Parameter | V5 Value | V3 Value | Note |
|-----------|----------|----------|------|
| `steps` | 200,000 | 50,000 | 4x longer |
| `batch_size` | 64 | 64 | same |
| `num_episodes` | 136 | 74 | 1.84x more |
| `num_frames` | 13,470 | 13,145 | similar total |
| `pretrained_path` | `lerobot/smolvla_base` | same | correct |
| `scheduler_warmup_steps` | 2,000 | 1,000 (default) | improved: 1% of total |
| `scheduler_decay_steps` | 200,000 | 30,000 (default) | FIXED: full-run cosine decay |
| `scheduler_decay_lr` | 2.5e-6 | 2.5e-6 (default) | same |
| `save_freq` | 10,000 | 5,000 | 20 checkpoints total |
| `eval_freq` | 20,000 | 5,000 | 10 eval points |
| `log_freq` | 100 | 100 | same |
| `num_learnable_params` | 99,880,992 (100M) | same | Action Expert only |
| `num_total_params` | 450,046,176 (450M) | same | VLM 350M frozen |

**Key scheduler improvement**: V3 used `decay_steps=30K` default with `steps=50K`. This caused LR to hit minimum by step 30K and stay flat for the last 20K steps (40% of run at minimum LR). V5 sets `decay_steps=200K` to match total steps — smooth cosine decay across the entire training run.

---

## 3. GPU / VRAM Usage

Measured at step ~500:
- **VRAM used**: 11,431 MiB / 16,376 MiB = **69.8%**
- **GPU Utilization**: 91-100%
- **Temperature**: 62-63°C (healthy, well within limit)

Note: VRAM is 11.4 GB vs the 9.85 GB measured during batch-size testing. The difference (~1.5 GB) is due to:
- Two additional `python3` worker processes using 390 MiB each
- Dynamic activation memory during active training

Headroom: ~4.9 GB remaining — no OOM risk.

---

## 4. Throughput Analysis

Measured from steady-state (steps 200-900, 148s per 100 steps):

| Metric | Value |
|--------|-------|
| Steps per second | 0.676 steps/s |
| Seconds per step | ~1.428s (matches `updt_s` in log) |
| Data loading overhead | 0.046-0.071s per step (~4-5%) |

**Time estimates (based on 1.428s/step from step 200 onward):**

| Milestone | Steps | Estimated Time | Absolute ETA |
|-----------|-------|----------------|--------------|
| 1st checkpoint | 10,000 | 3.97 hours | ~19:35 Mar 27 |
| 25% | 50,000 | 19.8 hours | ~11:30 Mar 28 |
| 50% | 100,000 | 39.6 hours | ~07:10 Mar 29 |
| 75% | 150,000 | 59.4 hours | ~02:50 Mar 30 |
| 100% complete | 200,000 | **79.2 hours** | **~22:35 Mar 30** |

Training started: 2026-03-27 15:37:32
**ETA completion: 2026-03-30 ~22:35 (approximately 79 hours total)**

---

## 5. Loss Progression — V5 vs V3 Comparison

### V5 Loss (current run)
| Step | Loss | LR | Gradient Norm | Epoch |
|------|------|----|---------------|-------|
| 100  | 0.138 | 2.6e-06 | 1.300 | 0.48 |
| 200  | 0.078 | 7.6e-06 | 0.418 | 0.95 |
| 300  | 0.061 | 1.3e-05 | 0.354 | 1.43 |
| 400  | 0.052 | 1.8e-05 | 0.378 | 1.90 |
| 500  | 0.048 | 2.3e-05 | 0.386 | 2.38 |
| 600  | 0.044 | 2.8e-05 | 0.420 | 2.85 |
| 700  | 0.041 | 3.3e-05 | 0.426 | 3.33 |
| 800  | 0.039 | 3.8e-05 | 0.394 | 3.80 |
| 900  | 0.037 | 4.3e-05 | 0.434 | 4.28 |
| 1000 | 0.036 | 4.8e-05 | 0.439 | 4.75 |
| 1100 | 0.035 | 5.3e-05 | 0.430 | 5.23 |

### V3 Loss (reference, from `outputs/training_v3.log`)
| Step | Loss | LR | Gradient Norm |
|------|------|----|---------------|
| 100  | 0.117 | 5.1e-06 | 0.820 |
| 200  | 0.078 | 1.5e-05 | 0.410 |
| 300  | 0.062 | 2.5e-05 | 0.439 |
| 400  | 0.054 | 3.5e-05 | 0.471 |
| 500  | 0.048 | 4.5e-05 | 0.464 |
| 600  | 0.045 | 5.5e-05 | 0.463 |
| 700  | 0.043 | 6.5e-05 | 0.487 |
| 800  | 0.042 | 7.5e-05 | 0.480 |
| 900  | 0.042 | 8.5e-05 | 0.488 |
| 1000 | 0.041 | 9.5e-05 | 0.455 |

### Direct Comparison at Step 1000
| Metric | V5 | V3 | Delta |
|--------|----|----|-------|
| Loss | **0.036** | 0.041 | -12% lower |
| LR | 4.8e-05 | 9.5e-05 | V5 slower warmup (expected) |
| Gradient norm | 0.439 | 0.455 | similar |

V5 loss is consistently lower than V3 at matching step counts. Possible reasons:
1. V5 dataset (136 episodes, 5 zones) may have better quality/diversity
2. V5 LR still in warmup (4.8e-5 vs V3 near-peak 9.5e-5) — lower LR = more conservative updates = lower instantaneous loss

This is a positive early signal but not yet conclusive — the key metrics come at 25K+ steps.

---

## 6. Learning Rate Schedule Verification

V5 warmup target: reach peak LR=1e-4 at step 2,000.

Observed progression:
- Step 100: 2.6e-6 (ramping from 0)
- Step 200: 7.6e-6
- Step 300: 1.3e-5
- Step 400: 1.8e-5
- Step 500: 2.3e-5
- Step 1100: 5.3e-5

Pattern: ~5e-6 increase per 100 steps. At step 2000, expected LR ~1e-4. This matches linear warmup of `peak_lr / warmup_steps = 1e-4 / 2000`.

After step 2000: cosine decay from 1e-4 to 2.5e-6 over 200,000 steps.

**V3 comparison note**: V3 had `decay_steps=30K` (default), so LR hit peak faster (step 1000 already at 9.5e-5) and decayed to minimum by step 30K, leaving 30K-50K at flat minimum LR. V5 correctly fixes this with full-run cosine decay.

---

## 7. Dataset: Actual vs Expected

From training log:
- `dataset.num_frames = 13,470`
- `dataset.num_episodes = 136` (planned 150)

**Average frames/episode**: 13,470 / 136 = **99 frames/episode** (significantly less than the 178 planned)

**Epoch recalculation for actual dataset:**
- steps/epoch = 13,470 / 64 = 210.5
- epochs at 200K steps = 200,000 / 210.5 = **950 epochs**

The `run_official_train.py` comment estimated 480 epochs (150 eps × 178 frames). Actual is ~950 epochs because:
1. Only 136 episodes collected (not 150)
2. Average 99 frames/ep (not 178) — shorter episodes in v5 data

**950 epochs vs V3's 243 epochs** is a significant difference. V3 best checkpoint was 25K steps (~120 epochs). V5 will hit 120 epochs at just ~25K steps — same as V3. The question is whether more epochs on the same data drives further improvement or overfitting.

Given V3 showed diminishing returns after 25K steps (L2: 2.81 → 2.99 at 50K), and V5 has 4x more steps on similar data, there is **elevated overfitting risk** in the 50K-200K range. However, the 5-zone diversity (vs V3's single sponge zone) may offset this.

**Recommendation**: Evaluate checkpoints at 10K, 25K, 50K, 100K — don't assume 200K is best.

---

## 8. Common SmolVLA Issues — Status

| Issue | Check | Status |
|-------|-------|--------|
| NaN/Inf loss | grep in log | CLEAR |
| Mean action problem | Loss at step 100 = 0.138, rapidly falling | CLEAR |
| Gradient explosion | grdn norm < 2.0 throughout | CLEAR (max=1.3 at step 100) |
| OOM crash | VRAM 69.8%, no OOM error | CLEAR |
| Pretrained weights not loaded | Loss starts 0.138 (vs scratch ~0.5+) | CLEAR |
| LR schedule misconfiguration | LR ramping correctly per warmup plan | CLEAR |
| Training process died | Step 1100 logged at 16:04 | CLEAR |

---

## 9. First Checkpoint (10K) — What to Check

ETA: ~19:35 on 2026-03-27

```bash
# Verify checkpoint exists
ls outputs/smolvla_v5_multipos/checkpoints/010000/pretrained_model/

# Check loss at 10K from log
grep "step:10K" training_v5.log

# Run offline evaluation (once training is further along)
# Use train_eval_v3_checkpoints.py adapted for v5 dataset path
```

V3 reference: loss at 10K was approximately 0.017-0.020 (estimated from trend). V5 should be at similar or slightly lower loss given its lower early-step loss.

---

## 10. Recommendations

### Checkpoint evaluation plan
Adapt `train_eval_v3_checkpoints.py` for v5 dataset path and evaluate at:
- 10K, 25K, 50K (early assessment — compare to V3 25K best)
- 100K, 150K, 200K (mid-to-late assessment)

Key metrics: L2 error + diversity ratio + gripper range

### Overfitting watch
- V3 best checkpoint was 25K, diminishing returns after
- V5's 950-epoch count means model sees data more often
- Watch for loss divergence between train and eval at 50K+ checkpoints
- **Expected best checkpoint**: 25K-50K range (not 200K)

### Dataset gap (136 vs 150 episodes)
- 136 episodes is 91% of planned 150 — acceptable, not a blocking issue
- The shorter episode length (99 vs 178 frames) means data collection used shorter demos
- Consider whether v5 episodes capture full grasp-lift-return cycles

---

## 11. Summary Table

| Item | Result |
|------|--------|
| Training status | HEALTHY |
| Errors | None (1 benign urllib3 warning) |
| VRAM | 11.4 GB / 16.4 GB (69.8%) — safe |
| GPU temp | 62-63°C — normal |
| GPU utilization | 91-100% |
| Throughput | 0.676 steps/s, 1.428 s/step |
| ETA (200K) | ~2026-03-30 22:35 (~79 hours total) |
| Loss at step 1000 | 0.036 (V3 was 0.041 — 12% better) |
| LR schedule | Correct — linear warmup, cosine decay from step 2K |
| Scheduler fix vs V3 | Confirmed: decay_steps=200K (V3 had 30K default) |
| Dataset actual | 136 eps, 13,470 frames, 99 frames/ep |
| Epoch count | ~950 epochs (V3 was ~243) |
| Main risk | High epoch count — overfitting possible after 50K steps |
| Best checkpoint window | 25K-50K (based on V3 pattern + epoch analysis) |
