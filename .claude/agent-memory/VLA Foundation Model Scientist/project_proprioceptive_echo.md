---
name: Proprioceptive Echo Failure Mode Analysis (2026-03-31)
description: VGST r=1.000 root cause, literature basis, architectural mechanism, and 5 prioritized fixes for SmolVLA proprioceptive shortcut learning
type: project
---

## Failure
VGST on 120K checkpoint: M1 r=1.000 (pred_base ≈ state_base), M2 image sensitivity=2.10° (FAIL).
Model has learned the proprioceptive echo shortcut: output ≈ current joint state regardless of vision.

**Why:** 3 compounding causes:
1. Data imbalance (LEFT=2ep, CENTER=109/136 = 80%) makes state→action a near-optimal predictor
2. Per-frame shortcut: at 30fps, action[t][0] ≈ state[t] + 0.67° (slow joints), making state≈action for early chunk steps
3. Frozen VLM (train_expert_only=True): state_proj (linear, 100% trained) vs cross-attn to frozen VLM KV (weak visual gradient)

**How to apply:** Before any deployment test, verify M_echo > 5° (mean |pred_base - state_base|) and M_visual > 10° std across zones at frame 0. If failing, do NOT deploy.

## Literature Basis
- **Causal confusion** (de Haan et al., NeurIPS 2019, arXiv:1905.11979): exact phenomenon. State co-varies with action → spurious state shortcut.
- **Shortcut learning** (Geirhos et al., ICLR 2020): distributional bias enables loss minimization without intended features.
- **Image permutation test** (RT-1, 2022): standard diagnostic for vision-blindness. Shuffle images → if <5° change, policy ignores vision.
- Flow matching specific: NOT documented, but mechanically expected. Delta echo: u_t = noise - state satisfies the loss if action ≈ state.

## Architectural Mechanism (confirmed from source)
Source: `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`, line 686
- state → pad(32) → state_proj(32→960 Linear, TRAINED) → 1 token in prefix, att_mask=1
- image → SigLIP(FROZEN) → connector(FROZEN) → 64 tokens, prefix-LM att_mask=0
- Flow matching loss: MSE(noise - action, v_t). If action ≈ state → v_t ≈ noise - state_emb → trainable from state alone.

## Priority Fixes (no source modification needed)
1. **[HIGHEST] Delta actions**: convert action target to (action - state). Echo shortcut impossible: MSE(noise - (state+delta), v_t) cannot be minimized by predicting v_t = noise - state. Implement in convert_to_lerobot_v3.py + deploy_smolvla.py.
2. **[HIGH] Virtual state dropout**: for 30% of episodes, replace state with dataset_mean before saving to parquet. Forces model to use vision for those episodes. Implement in convert_to_lerobot_v3.py.
3. **[HIGH] Short episodes**: 45-50 frames/episode (1.5s at 30fps). Stop before robot reaches target. Eliminates 54/99 = 55% "at-target echo frames." Also doubles episode count for same collection time.
4. **[MEDIUM] M_echo metric**: add to test_inference_official.py. Pass criterion: >5°.
5. **[LAST RESORT] VLM LoRA**: --policy.peft_type=lora. Adds capacity to visual pathway. Risk: complexity, forgetting.

## NOT FIXES
- Balanced data alone (30ep/zone): necessary but insufficient. Per-frame shortcut still valid for 95% of frames.
- Longer training: amplifies the existing shortcut, does not break it.
- Higher batch size: no effect on gradient source distribution.

## Files
Analysis script: `/home/cgxr/Documents/Robotics/RoArm_Project/model_proprioceptive_echo_analysis.py`
