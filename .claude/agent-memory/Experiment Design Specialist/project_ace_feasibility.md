---
name: project_ace_feasibility
description: Autonomous Competence Expansion (self-improving loop) — quantitative feasibility analysis for CoRL 2026. 8 questions answered with numbers.
type: project
---

## "Autonomous Competence Expansion" Feasibility (2026-03-25)

File: `experiment_autonomous_competence_expansion.py`

### Q1 VLM Judge Memory
- SmolVLA inference: ~3.3 GB VRAM
- Qwen2.5-VL-3B-int4: ~2.8 GB → total 6.1 GB → FEASIBLE on RTX 4090 (15.6 GB)
- Qwen2.5-VL-7B-int4: ~6.0 GB → total 9.3 GB → FEASIBLE
- Qwen2.5-VL-7B-bf16: 14 GB alone → NOT FEASIBLE simultaneously
- Judge runs AFTER episode ends, not concurrently → timing OK

### Q2 Safety
- Homing routine: move_init() + torque_set(cmd=1) after each episode (already in SDK)
- Object fall detection: Azure Kinect depth slice at target XY (no new hardware)
- Episode timeout: 60 sec hard limit required
- Velocity clipping: (target - current) clipped to ±15° per step (prevents v1 Wrist_R runaway)
- Use SAME open-loop 4-chunk config as v3 (closed-loop not safe for autonomous)

### Q3 Throughput
- Episode time (open-loop 4-chunk + homing + VLM judge): ~20 sec realistic
- Per hour: ~120 episodes (with 15% overhead)
- With foam arena (5% fall rate): ~852 episodes per 8-hour night
- Practical target: 200-400 episodes per overnight run

### Q4 Reward-Weighted BC
- MATHEMATICALLY SOUND for flow-matching (MSE loss supports per-sample weighting)
- SmolVLA forward(reduction='none') directly supported
- CRITICAL: stats.json changes with new data → must retrain from smolvla_base each cycle (Lesson #3)
- Recommended: binary filter (w=1 success, w=0 failure) for simplicity

### Q5 Retraining Time
- A100 80GB (VAST.ai ~$1.20/hr): 50K steps ≈ 1.25 hours
- Overnight loop: collect 4h + upload 15min + train 1.25h + download 5min = ~5.75h total
- 3 cycles cost: ~$4.50 total

### Q6 Flow-Matching Uncertainty
- MC dropout (option A): enable dropout at test time, 3-5 passes, variance = uncertainty proxy
- Denoising trajectory variance (option B): std(x_t) across t=0.1..0.9 per joint = research gap
- Recommend MC dropout for implementation; denoising variance as novel contribution

### Q7 VLM Judge Accuracy
- Dual-signal judge: VLM=YES AND FK_z>120mm AND gripper<30°
- Expected combined precision: >92% (vs ~80% VLM alone)
- VALIDATION REQUIRED: 50-ep ground-truth comparison before overnight run
- Threshold: precision > 0.85 to proceed

### Q8 Statistical Power
- N=71 per condition for 80% power at Δ=0.20 (50%→70%), one-sided α=0.05
- N=95 per condition for 90% power at same effect
- Recommended protocol: 25 positions × 4 trials = 100 trials per condition (covers 90% power)
- McNemar's test (paired) more powerful than independent proportions if same grid used before/after

### Key Risks
1. VLM judge precision < 85% → corrupts loop → validate first (1 day)
2. Object falls without arena → throughput drops to ~10% → foam arena required
3. stats.json must be recomputed per cycle → always train from smolvla_base

**Why:** This feasibility analysis supports Contribution #4 of the CoRL paper ("self-improving loop").
**How to apply:** Before designing the overnight experiment, complete VLM validation (50 episodes, precision check). Use dual-signal judge as default.
