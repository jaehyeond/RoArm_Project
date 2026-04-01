"""
SmolVLA Architecture Critical Analysis
B1 VLA Foundation Model Scientist

Answers to all 5 questions from architecture source code inspection.
Run as: python model_smolvla_architecture_critique.py
"""

# =============================================================================
# SECTION 1: HOW STATE IS INJECTED — EXACT MECHANISM
# =============================================================================

STATE_INJECTION = """
STATE INJECTION MECHANISM (from modeling_smolvla.py lines 571-697)
===================================================================

Architecture diagram from VLAFlowMatching.__init__:

  image(s) → SigLIP (FROZEN) → connector (FROZEN) → N img tokens (prefix, att=0)
  language  → embed_language_tokens (FROZEN VLM embedding) → M lang tokens (prefix, att=0)
  state     → pad_vector(6→32) → state_proj(32→960 Linear, TRAINED) → 1 token (prefix, att=1)
  ─────────────────────────────────────────────────────────────────────────────────
  all concatenated → prefix_embs [B, N+M+1, 960]
                          ↓ VLM forward (frozen cross-attn to prefix KV)
  noise+timestep → action_in_proj → action_time_mlp → chunk_size tokens (suffix)
                          ↓ Action Expert forward (cross-attn to prefix KV)
  action output ← action_out_proj ← expert_hidden_size

KEY FACTS:
1. state_proj is a nn.Linear(32, 960). It is ALWAYS trained (train_state_proj=True).
2. image features come through FROZEN SigLIP + FROZEN connector.
3. The state token is in the PREFIX, with att_mask=1, meaning it CAN attend to
   language/image tokens but image/language tokens CANNOT attend back to state.
4. The Action Expert attends to ALL prefix tokens (image + language + state)
   via its KV cache. State contributes as a single 960-dim embedding.
5. There is NO explicit weighting between visual vs state tokens — both contribute
   to the Action Expert's cross-attention equally in principle.

WHAT THIS MEANS FOR PROPRIOCEPTIVE ECHO:
- state_proj is the ONLY fully-trained linear projection in the entire prefix pathway
- Image features: frozen SigLIP → frozen connector → fixed representation
- The gradient ONLY flows to: state_proj, action_in_proj, action_out_proj,
  action_time_mlp_in, action_time_mlp_out, lm_expert cross-attn k/v projections
- If state → action is a near-valid predictor (as in center-biased data),
  state_proj will be trained to encode state in the KV cache optimally for echo
- Visual features CANNOT be updated (frozen) — they cannot compensate
"""

# =============================================================================
# SECTION 2: FULL FORWARD FLOW
# =============================================================================

FORWARD_FLOW = """
COMPLETE INFORMATION FLOW (Training and Inference)
====================================================

TRAINING (modeling_smolvla.py VLAFlowMatching.forward, line 756):
  1. embed_prefix(images, lang_tokens, state):
     - images: SigLIP(FROZEN) → connector(FROZEN) → N=64 tokens/image at 512×512
     - lang: embedding(FROZEN) → M tokens (≤48)
     - state: pad(6→32) → state_proj(Linear, TRAINED) → 1 token
     - concatenated → prefix_embs [B, N+M+1, 960]

  2. embed_suffix(noisy_actions, timestep):
     - action_in_proj(32→expert_hidden) + sinusoidal timestep embedding
     - action_time_mlp_in + SiLU + action_time_mlp_out
     - → suffix_embs [B, 50, expert_hidden_size]

  3. SmolVLMWithExpertModel.forward([prefix_embs, suffix_embs]):
     - attention_mode="cross_attn" (default from smolvla_base)
     - VLM processes prefix only via self-attention (every 2nd layer)
     - Action Expert attends to VLM KV via cross-attention
     - Result: suffix_out [B, 50, expert_hidden_size]

  4. action_out_proj(suffix_out) → v_t [B, 50, 32]
  5. Loss = MSE(u_t, v_t) where u_t = noise - action

INFERENCE (sample_actions, line 794):
  1. embed_prefix → prefix_embs (same as training)
  2. VLM forward with fill_kv_cache=True → past_key_values cached
     - CRITICAL: state IS part of the prefix at inference too
     - KV cache includes image, language, AND state tokens
  3. 10 denoising steps:
     - Only embed_suffix is re-run per step (noisy action + timestep)
     - VLM is NOT re-run — prefix KV cache reused
     - Action Expert cross-attends to cached KV each step

IMPLICATION: state is embedded once into the KV cache at the start of inference.
The denoising loop cannot "update" its understanding of state mid-inference.
"""

# =============================================================================
# SECTION 3: IS PROPRIOCEPTIVE ECHO EXPECTED OR A BUG?
# =============================================================================

ECHO_ANALYSIS = """
IS PROPRIOCEPTIVE ECHO EXPECTED BEHAVIOR OR A BUG?
====================================================

VERDICT: Expected failure mode under the specific training conditions.
NOT a bug in SmolVLA architecture. It IS a consequence of this project's data.

WHY IT HAPPENS ON RoArm M3 BUT NOT SO100:
───────────────────────────────────────────

SO100 conditions (official recipe):
  - 50 episodes × 5 DISTINCT cube positions × 10 reps each
  - Positions are spatially spread (left/center/right on table)
  - state at t=0 does NOT encode which position (robot starts at same home)
  - action[t][0] ≠ state[t] because base joint must rotate significantly
  - Echo shortcut is NOT valid: state alone cannot predict target base angle
  - state_proj trained to give ZERO signal → model forced to use vision

v5 RoArm M3 conditions (what went wrong):
  - 136 episodes: LEFT=2(1.5%), CENTER=109(80%), RIGHT=25(18%)
  - Dataset_mean ≈ center position (80% weighted average)
  - state at t=0 WAS collected WITHOUT HOME start → state encodes current zone
    (robot was already near task position when recording started)
  - action[t][0] ≈ state[t]+0.67° (30fps, slow joints = per-frame echo valid)
  - state_proj gets LARGE gradient to encode state_base → action_base
  - Visual gradient is near-zero: image never needed to predict base correctly

CRITICAL DIFFERENCE BETWEEN SO100 AND RoArm M3:
  SO100: fresh start → state t=0 uninformative → vision required
  v5 RoArm M3: start ≈ task position → state t=0 encodes zone → vision not needed
  This is the ROOT CAUSE. It is a DATA collection bug, not a model bug.

M2 IMAGE SENSITIVITY (VGST) VALUES:
  v5 120K checkpoint: M2 = 2.10° (FAIL, threshold 5°)
  v3 checkpoint: M2 = 1.73° (also FAIL, but went undetected — no VGST then)
  SO100 official: not reported, but implied >>10° (model passes real evals)

WHY FROZEN VLM MAKES IT WORSE:
  - state_proj: 100% trained gradient
  - SigLIP/connector: 0% trained gradient
  - The only way visual features affect output is through frozen KV cross-attn
  - If state_proj can satisfy loss, no gradient pressure to use visual pathway
  - contrast with pi0 (full VLM fine-tune): visual pathway CAN be trained
    to compete with state pathway → echo less likely

DOES v3 ALSO HAVE ECHO?
  YES. M2=1.73° means v3 also ignores vision nearly entirely.
  The 5/5 (100%) result is explained by: model memorized trajectories for
  the ONE position tested (base≈10°, CENTER zone). False positive.
  This project has never had a genuinely visual model.
"""

# =============================================================================
# SECTION 4: CAN STATE BE ZEROED WITHOUT CODE CHANGES?
# =============================================================================

STATE_ZEROING = """
CAN STATE BE DISABLED/REDUCED WITHOUT MODIFYING SmolVLA SOURCE?
=================================================================

OPTION A: Zero state in dataset at convert time — YES, no source change
  In convert_to_lerobot_v3.py, when writing state observations:
    state = np.zeros(6)  # zero out state
    # or
    state = DATASET_MEAN  # constant → same as no information

  Effect: state_proj receives constant input → trains to output constant bias
  Problem: model still has state pathway, just learns to ignore it
  ACTUAL EFFECT: likely reduces echo but does NOT eliminate it
                 state_proj will still have trainable bias terms

OPTION B: State dropout at convert time (30% of episodes)
  Already recommended as Priority 2 fix in proprioceptive_echo_analysis.py
  Replace state with dataset_mean for 30% of episodes
  Effect: model cannot rely on state for those episodes → must use vision
  Cost: no source code change, just dataset preprocessing
  This is the RECOMMENDED path without source modification

OPTION C: Delta actions (Priority 1 fix)
  Convert action targets to action[t] - state[t]
  After prediction, add back state during deployment
  Effect: echo shortcut becomes impossible (MSE(noise-(state+delta), v_t)
          cannot be satisfied by v_t = noise - state_emb alone)
  Implementation: convert_to_lerobot_v3.py + deploy_smolvla.py
  This is the MOST PRINCIPLED fix without source modification

OPTION D: LoRA on VLM (Priority 5)
  --policy.peft_type=lora targets q/v in lm_expert
  Does NOT unfreeze SigLIP/connector
  Effect: action expert becomes more flexible, may better utilize frozen vision
  This does NOT fix the echo shortcut — just adds capacity

WHAT CANNOT BE DONE WITHOUT SOURCE CHANGES:
  - Directly reduce state_proj gradient weight
  - Make state dropout probabilistic during training (requires custom sampler)
  - Unfreeze SigLIP for training (code change in smolvlm_with_expert.py)
  - Increase visual token count (would need processor changes)

RECOMMENDATION: Implement B + C together in v6 pipeline.
  B (state dropout) + C (delta actions) are additive defenses.
"""

# =============================================================================
# SECTION 5: OFFICIAL RECIPE vs THIS PROJECT — CRITICAL COMPARISON
# =============================================================================

RECIPE_COMPARISON = """
OFFICIAL SMOLVLA RECIPE vs THIS PROJECT
========================================

Parameter        Official Recipe        v5 (FAILED)           v6 (Target)
─────────────    ─────────────────      ─────────────────      ──────────────────
Episodes         50 (5pos × 10rep)      136 (80% center)       50 (5zone × 10rep)
Frames/episode   ~393 (~13 sec)         ~99 (3.3 sec)          ~200 (6.7 sec)
Total frames     ~19,650                ~13,470                ~10,000
Steps            20,000                 200,000 (10× EXCESS)   20,000
batch_size       64                     64                     64
Steps/epoch      19650/64 ≈ 307         13470/64 ≈ 210         10000/64 ≈ 156
Epochs           20K/307 ≈ 65           200K/210 ≈ 952         20K/156 ≈ 128
Start position   HOME (same every ep)   ANYWHERE (no HOME)     HOME enforced
Zone balance     Equal (10/position)    LEFT:2, CTR:109, R:25  Equal (10/zone)
Scheduler        warmup=1K, decay=30K   warmup=1K, decay=30K   same as official
                 (decay>steps = normal) (v5 BUG: wrong path)   (path fixed)

CRITICAL OBSERVATIONS:
1. v5 total frames (13,470) vs official (~19,650): v5 had MORE episodes but
   SHORTER episodes → same total steps but with worse episode diversity
2. 200K steps on 13,470 frames = 952 epochs = SEVERE OVERFITTING SETUP
   Official 20K on 19,650 frames = 65 epochs = reasonable regularization
3. The scheduler decay_steps=30K > training_steps=20K is INTENTIONAL in official
   This means the cosine decay never completes a full cycle → learning rate stays
   near peak_lr for most of training, then gently decays
   This is NOT a bug. v5 had a bug in the PATH to smolvla_base, not the scheduler.

SCHEDULER ANALYSIS (SmolVLAConfig defaults):
  optimizer_lr = 1e-4 (peak)
  scheduler_warmup_steps = 1,000
  scheduler_decay_steps = 30,000
  scheduler_decay_lr = 2.5e-6

  For 20K training: LR rises 0→1e-4 over 1K steps, then
  cosine decays 1e-4→2.5e-6 over 30K steps, but training ends at 20K.
  → At step 20K: LR ≈ 1e-4 * cos(π * (20K-1K)/(30K-1K)) / 2 + offset ≈ 5e-5
  → Still at 50% of peak at training end. NOT fully decayed.

  This design means: use 20K for fast convergence testing, scale to 50K for
  better final performance. Official paper uses batch=64 throughout.

WHAT EXPLAINS v3 FALSE POSITIVE?
  v3: 74ep, 50K steps, batch=64
  Episodes: mix of zones BUT only tested at CENTER (base≈10°)
  50K steps on ~10K frames = ~500 epochs → also overfit
  M2=1.73° → also ignores vision
  Passed because evaluation was at the exact memorized position
  FALSE POSITIVE confirmed.

BOTTOM LINE FOR v6:
  The official recipe works because of DATA STRUCTURE, not hyperparameter magic.
  5pos × 10rep, uniform coverage, HOME start, ~13sec episodes.
  v6 must replicate this EXACTLY. 50ep is not 50ep if zones are unbalanced.
"""

# =============================================================================
# SECTION 6: ARCHITECTURAL CAPACITY — CAN 450M HANDLE MULTI-TASK?
# =============================================================================

MODEL_CAPACITY = """
MODEL CAPACITY ANALYSIS: 450M FOR MULTI-OBJECT/TASK
=====================================================

SmolVLA parameter breakdown:
  VLM (frozen): ~350M (SigLIP ~90M + SmolLM2 ~500M text → subset ~260M)
  Action Expert (trained): ~100M
  state_proj: 32×960 = 30,720 params (negligible)
  action_proj: ~100K params (negligible)
  Effective trainable params: ~100M action expert

CAPACITY QUESTION: Can 100M trained params handle 4-object/5-zone multi-task?
  Reference points:
  - Diffusion Policy (Chi et al., 2023): 300M params, 1 task, 1 robot → works
  - ACT: 83M params, Aloha, 5 tasks → works
  - SmolVLA SO100: 100M trained, 1 task, 5 positions → works

  FOR 4-OBJECT PICK-PLACE (if that were the goal):
  Objects are distinguished by the FROZEN SigLIP features (not trained weights).
  SigLIP at 512×512 input: 64 tokens × 1152-dim per image = rich visual representation.
  The Action Expert does NOT need to classify objects — it just needs to condition
  on the SigLIP embeddings that already encode object identity.

  VERDICT: 100M action expert is SUFFICIENT for multi-object conditioning,
  IF SigLIP distinguishes the objects (which it does — it's pretrained on web data).
  The bottleneck is NOT model capacity. It is training data quality.

CURRENT SCOPE (v6 = pick single sponge, 5 zones):
  This is well within 450M capacity. The v5 failure was 100% data/training,
  not model capacity. Even 10M params would have been sufficient.
"""

# =============================================================================
# MAIN OUTPUT
# =============================================================================

if __name__ == "__main__":
    sections = [
        ("STATE INJECTION MECHANISM", STATE_INJECTION),
        ("COMPLETE FORWARD FLOW", FORWARD_FLOW),
        ("PROPRIOCEPTIVE ECHO ANALYSIS", ECHO_ANALYSIS),
        ("STATE ZEROING OPTIONS", STATE_ZEROING),
        ("OFFICIAL RECIPE COMPARISON", RECIPE_COMPARISON),
        ("MODEL CAPACITY ANALYSIS", MODEL_CAPACITY),
    ]

    for title, content in sections:
        print("=" * 70)
        print(f" {title}")
        print("=" * 70)
        print(content)

    print("\n" + "=" * 70)
    print(" CRITICAL FINDINGS SUMMARY")
    print("=" * 70)
    print("""
KEY FINDINGS (B1 VLA Foundation Model Scientist):

1. STATE INJECTION: state → Linear(32→960) → 1 VLM prefix token.
   This is the ONLY fully trained layer in the prefix pathway.
   Image features come through FROZEN SigLIP. This is the echo source.

2. ECHO IS DATA-CAUSED, NOT ARCHITECTURE-CAUSED:
   SO100 success: HOME start → state at t=0 uninformative → vision required.
   v5 failure: no HOME start → state encodes zone → vision not needed.
   The architecture works correctly on properly collected data.

3. FIXES WITHOUT SOURCE CHANGES (in priority order):
   [1] Delta actions in convert_to_lerobot_v3.py + deploy_smolvla.py
   [2] State dropout (30% of episodes set to dataset_mean) in convert script
   [3] Balanced zones (5 × 10 reps) + HOME start enforcement
   These together should bring M2 > 10°.

4. v3 WAS ALSO ECHO: M2=1.73° confirmed. 5/5 result was trajectory memorization
   at one position. This project has never deployed a genuinely visual model.

5. v5 200K STEPS WAS HARMFUL: 952 epochs on 13K frames = deeply overfit echo.
   Official 65 epochs is the correct regime. Step count does NOT compensate
   for data quality deficiencies — it amplifies existing shortcuts.

6. SCHEDULER: warmup=1K, decay=30K on 20K training is INTENTIONAL. LR ends
   at ~50% of peak. This is the official design. Not a bug to fix.

7. MODEL CAPACITY (450M) IS NOT THE BOTTLENECK for 5-zone single-task.
   Even for 4-object multi-task it would be sufficient. Focus on data.
""")
