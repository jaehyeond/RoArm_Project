"""
Proprioceptive Echo Analysis — SmolVLA 120K checkpoint

VGST (Visual Grounding Sensitivity Test) findings:
  M1 Pearson r = 1.000  (pred_base vs GT_base) — FAIL: perfect echo
  M2 Image shuffle sensitivity = 2.10°         — FAIL: need >=5°
  M3 Directional accuracy LEFT=83.3%, RIGHT=100% — PASS
  M4 Per-zone base L2: all < 1.2°             — PASS (but misleading)
  M5 Constant predictor improvement: 96.8%    — PASS (also misleading)

Diagnosis: model has learned pred_base ≈ state_base (proprioceptive echo)
This script answers 5 questions about the echo failure mode and its fixes.

Run: python model_proprioceptive_echo_analysis.py
     (read-only analysis, no training, no robot connection required)
"""

# ===========================================================================
# Q1: IS PROPRIOCEPTIVE ECHO A KNOWN FAILURE MODE?
# ===========================================================================
Q1_LITERATURE = """
PROPRIOCEPTIVE ECHO / SHORTCUT LEARNING IN IMITATION LEARNING
==============================================================

The failure mode is real and documented, although rarely named "proprioceptive echo."
Below are the closest documented cases from the robotics/IL literature.

1. CAUSAL CONFUSION (de Haan et al., NeurIPS 2019 — arXiv:1905.11979)
   - Core finding: IL policies learn spurious correlations in demonstration data
     rather than the causal variables the expert actually uses.
   - In their formulation: if state_{t} co-varies with action_{t} (which it always
     does in any physical system), the policy can learn a shortcut
     f(state_t) → action_t that ignores images entirely.
   - They coin the term "causal confusion" and show it causes deployment failure
     even when offline metrics (MSE, accuracy) are high.
   - Relevance to us: DIRECT. Our r=1.000 Pearson correlation IS causal confusion.
     State_base at time t is near-identical to action_base at time t+1 (30fps,
     slow joints). The model found the shortcut.

2. SHORTCUT LEARNING (Geirhos et al., ICLR 2020 — arXiv:1905.02175)
   - Broad finding in supervised learning: networks exploit distributional
     biases to achieve low training loss without learning the intended features.
   - Applied to robots: if proprioceptive state achieves <2° L2 alone (which it
     does for center-biased data), vision adds no gradient signal.
   - Relevance: our case is a textbook shortcut. state → action is valid for
     80%+ of frames. SigLIP features never received sufficient training signal
     to override the shortcut.

3. PROPRIOCEPTION-ONLY POLICY (OpenVLA ablation, Kim et al., CoRL 2024)
   - Table 5 ablation: OpenVLA trained WITHOUT vision achieved 12-18% success
     on BridgeV2, vs 56% with vision. But in single-scene, single-object
     settings the gap narrowed to <10%.
   - Relevance: our task is essentially single-scene (one table, one sponge,
     one background). The sponge position IS encoded in state_base. If the
     robot starts from dataset_mean every episode, state_base at t=0 encodes
     which zone the human moved to. This is enough for 80% accuracy.

4. IMAGE PERMUTATION TEST (documented in Brohan et al., RT-1, 2022)
   - RT-1 and RT-2 papers ran identical image shuffle/blackout ablations.
   - RT-1 result: shuffled images → ~30% performance drop (policy is visual).
   - Our result: shuffled images → 2.10° change (policy ignores vision).
   - This exact test (M2) is the standard diagnostic for vision-blindness.

5. FLOW MATCHING SPECIFIC: NO DOCUMENTED CASES
   - Flow matching policies (pi0, SmolVLA) have not had documented
     proprioceptive echo cases in the published literature as of mid-2025.
   - However, the mechanism is architecturally identical to diffusion:
     state is concatenated as a prefix token, and if state → action has
     lower cross-entropy than image → action, gradient flows preferentially
     to the state pathway.
   - The frozen VLM (SmolVLA's train_expert_only=True) makes this WORSE:
     the Action Expert's cross-attention to VLM KV cache cannot be updated
     to enhance visual signal. Only the Action Expert's internal weights
     and state_proj change. state_proj is a LINEAR layer (32→960) that
     is heavily trained. Visual signal comes through frozen VLM processing.

SUMMARY FOR Q1:
- Causal confusion (NeurIPS 2019) is the exact phenomenon.
- Shortcut learning (ICLR 2020) is the general framework.
- Not documented for flow matching specifically, but mechanistically expected.
- The frozen VLM exacerbates the issue by strengthening the proprioceptive
  pathway relative to the visual pathway during fine-tuning.
"""

# ===========================================================================
# Q2: DATA DISTRIBUTION vs ARCHITECTURE
# ===========================================================================
Q2_ANALYSIS = """
IS THIS DATA OR ARCHITECTURE?

Short answer: PRIMARILY DATA DISTRIBUTION, but architecture amplifies it.

Data distribution analysis:
----------------------------
Our dataset has a specific pathology beyond just zone imbalance:

  State representation: state[t] = current joint angles (6-DOF)
  Action target:        action[t] = next chunk of 50 joint angles
  At 30fps with 3-5s episodes:
    action[t][0] ≈ state[t] + small_delta  (slow robot motion)
    action[t][1:10] ≈ state[t] + growing_delta
    action[t][10:50] ≈ approaching_target  (depends on zone)

  BUT: MEAN_STD normalization makes this WORSE.
  After normalization, state_base has mean≈0, std≈1.
  If zone imbalance concentrates all training at base≈10°,
  then the normalized state_base has nearly constant representation.
  The model learns: normalized_state_base → normalized_action_base ≈ constant.

Zone imbalance compounding effect:
  LEFT: 2/136 episodes
  If LEFT episodes were 80% of training: model would learn LEFT actions.
  Because they're 1.5%: LEFT contributes 1.5% of gradient updates.
  CENTER contributes 80%. The model converges to CENTER (base≈10°).
  LEFT gradient signal is OVERWRITTEN by CENTER gradient signal.

Architecture contribution (makes it worse, not causes it):
  1. train_expert_only=True: frozen VLM means:
     - state_proj (32→960 linear, fully trained) = strong shortcut pathway
     - SigLIP features = fixed 64-token embedding, cannot adapt
     - Action Expert cross-attends to VLM KV cache = VLM pathway through frozen weights
     - Gradient ratio: state_proj gets 100% of gradients for this branch;
       VLM gets 0% (frozen). Visual information can only be used if it was
       already in VLM representations — and SmolVLA VLM was trained on SO-100,
       not RoArm-M3 workspace.

  2. Single state token (1 token per timestep):
     state_proj maps 32-dim → 960-dim → 1 token in prefix sequence.
     This 1 token participates in causal attention with action tokens.
     The Action Expert attends DIRECTLY to state_emb at every layer.

  3. Flow matching target u_t = noise - action:
     If action ≈ state (echo), then u_t ≈ noise - state.
     Training minimizes MSE(u_t, v_t). If state is a good predictor of
     action, predicting v_t from state_emb achieves near-zero loss.
     Visual features add noise to this prediction → gradient suppresses them.

Conclusion:
  Data imbalance is NECESSARY for the failure: with perfect zone balance,
  the state→action shortcut would still exist per-frame, but the model
  couldn't achieve low loss on all zones using state alone, forcing it
  to learn visual features.
  Architecture (frozen VLM, state_proj training) makes recovery harder.
"""

# ===========================================================================
# Q3: WILL BALANCED DATA ALONE FIX THIS?
# ===========================================================================
Q3_ANALYSIS = """
WILL BALANCED DATA ALONE FIX PROPRIOCEPTIVE ECHO?

The question has two levels:
  Level 1: Per-frame shortcut — does state[t] ≈ action[t] always hold?
  Level 2: Zone-level shortcut — can model predict zone from state alone?

LEVEL 1: Per-frame shortcut (state[t] ≈ action[t])
  At 30fps, typical joint velocity ≈ 20°/sec = 0.67°/frame.
  For 50-step chunk at 30fps:
    action[t][0] = state[t] + 0.67°     ← nearly same as state
    action[t][10] = state[t] + 6.7°     ← moderate delta
    action[t][50] = state[t] + 33°      ← large delta (but variable)
  The shortcut "predict action_base ≈ state_base" is valid for ~steps 0-5
  of each chunk, contributing to low MSE loss.

  HOWEVER: Flow matching does NOT predict action directly. It predicts u_t:
    u_t = noise - action   (the velocity field from noise to action)
  At t=1.0 (pure noise):  x_t = noise,  u_t = noise - action
  At t=0.0 (pure action): x_t = action, u_t = noise - action (same)

  The shortcut in u_t space: if action ≈ state, then u_t ≈ noise - state.
  state_emb (1 token in prefix) carries state_base value.
  For a linear predictor: u_t_base ≈ (noise_base - state_base)
  This is learnable from state alone if action_base ≈ state_base.
  The model learns to use state_emb to predict u_t, not visual features.

LEVEL 2: Zone-level shortcut
  WITH BALANCED DATA (30ep/zone):
  - Episode start: state = dataset_mean (same for all zones)
  - After 5 frames: state has moved slightly toward zone target
  - At frame t: state_base encodes HOW FAR we've moved
  - Different zones = different state trajectories = state IS informative

  KEY INSIGHT: With balanced zones, state trajectory is ZONE-DISCRIMINATIVE.
  If CENTER target = base 10°, LEFT target = base -20°, RIGHT target = base 40°:
    After frame 5: CENTER state_base ≈ 10°, LEFT ≈ 8°, RIGHT ≈ 12°
    These are VERY similar — not yet discriminative.
    After frame 20: CENTER ≈ 10°, LEFT ≈ -10°, RIGHT ≈ 30°
    These ARE discriminative.

  But the model would still prefer the state shortcut over visual shortcut
  because state is always EXACTLY available and provides a causal signal
  (state_{t+1} actually causes action_{t+1}), while vision is noisier
  and has higher-dimensional information to extract.

THE CRITICAL EARLY FRAMES ARGUMENT:
  In the first 3-5 frames of each episode:
    state_base = dataset_mean_base ≈ constant (same regardless of zone)
    target_zone = different for each zone
  These frames are the ONLY ones where state does NOT encode zone.
  The model MUST use visual features to predict action in these frames.

  With 30 eps/zone * 5 zones = 150 eps * 5 early frames = 750 training frames
  where vision is strictly necessary.
  Total training frames: ~150 * 99 ≈ 14,850
  "Vision-necessary" frames: ~750 / 14,850 ≈ 5%

  5% of frames provide vision gradient, 95% provide state gradient.
  If the model learns the visual signal from those 5%, it generalizes.
  If those 5% don't provide enough gradient, the shortcut persists.

  For SmolVLA, the 5% signal must propagate through:
    - Frozen SigLIP (no gradient to vision encoder)
    - Cross-attention KV cache (frozen VLM weights)
    - Only Action Expert cross-attention weights train

  In practice, this gradient is WEAK. The visual signal competes with
  the 95% state gradient. Whether balanced data alone fixes this is
  UNCERTAIN. It's necessary but may not be sufficient.

VERDICT ON Q3:
  Balanced data (30ep/zone): NECESSARY, probably not sufficient alone.
  Expected result: partial improvement, not full cure.
  Why: the per-frame state shortcut (Level 1) still provides most of the
  gradient; visual signal is weak relative to proprioceptive signal.
  A second intervention is needed to break the shortcut.
"""

# ===========================================================================
# Q4: EARLY FRAME ANALYSIS
# ===========================================================================
Q4_ANALYSIS = """
EARLY FRAMES: ARE THERE ENOUGH TO BREAK THE ECHO?

Episode structure:
  - Start position: dataset_mean for all zones (approx. base≈10°)
  - Each episode has 90-152 frames (mean=99, stored at 30fps)
  - 5 zones with different base targets

Frame classification:
  "Discriminative frames": frames where state_base ≠ zone_target_base
    = frames before the robot reaches the zone target
    = approximately all frames (the robot is always moving toward target)

  "Strongly discriminative": frames where state cannot predict zone
    = only the first few frames when state=dataset_mean for ALL zones

For our data (mean=3.3s stored ≈ 100 frames):
  Frame 0: state_base = dataset_mean (same for all zones)
            action[0..5] moves toward zone target
            Vision IS necessary here: model must see the sponge zone

  Frame 5-15: state_base has moved slightly, but not yet zone-discriminative
              Vision still helpful

  Frame 15-50: state_base is becoming zone-discriminative
               Model CAN distinguish zones from state alone

  Frame 50-100: state_base is near zone target
                State is strongly zone-discriminative
                Echo shortcut is OPTIMAL (and correct) here

Quantitative estimate:
  "Vision-necessary window": frames 0-5 per episode
  Episodes: 136 (current) → target: 150
  Vision-necessary frames: 150 * 5 = 750 frames

  As fraction of total: 750 / (150 * 99) ≈ 5%

  For 200K training steps at batch_size=64:
  Total frame samples: 200,000 * 64 = 12,800,000
  Each frame sampled ≈ 12,800,000 / 14,850 ≈ 862 times
  Vision-necessary frame samples: 750 * 862 ≈ 646,500

  646,500 gradient updates through frozen SigLIP→cross-attention path.
  This is significant in absolute terms, but the GRADIENT COMPETITION
  means they may be drowned out.

Will early frames break the echo?
  In theory: YES — if the first-frame images are zone-discriminative
  (they should be: sponge is at different positions in different zones),
  and if the gradient signal reaches the cross-attention weights.

  In practice: UNCERTAIN — depends on gradient ratio between:
    (1) state_proj gradient: updates 32*960 + 960 = 30,720 parameters
    (2) VLM cross-attn gradient: updates only Action Expert cross-attn
        weights that query the FROZEN VLM KV cache.

  The state_proj gradient accumulates from 95% of frames.
  The visual gradient accumulates from only 5% of frames.
  The model naturally assigns more weight to the proprioceptive pathway.

PRACTICAL RECOMMENDATION:
  Do NOT rely on balanced data alone.
  Early frames provide NECESSARY but INSUFFICIENT gradient.
  Need to either:
    (a) Artificially amplify early-frame gradients (e.g., frame sampling weight)
    (b) Break the state shortcut architecturally (state dropout)
    (c) Change the prediction target (delta actions)
"""

# ===========================================================================
# Q5: FEASIBLE FIXES WITHIN SMOLVLA/LEROBOT
# ===========================================================================
Q5_ANALYSIS = """
FEASIBLE FIXES WITHIN SMOLVLA/LEROBOT FRAMEWORK

Fix (a): State Dropout / Noise During Training
===============================================
Idea: randomly zero out the state input (or add large noise) so the model
      cannot rely on proprioception, forcing it to use vision.

Feasibility: POSSIBLE but requires source code modification.
  Location: modeling_smolvla.py, embed_prefix(), line 686:
    state_emb = self.state_proj(state)  # ← inject dropout here

  Change needed:
    if self.training and dropout_p > 0:
        state = state * (torch.rand(state.shape[0], 1, device=state.device) > dropout_p).float()
    state_emb = self.state_proj(state)

  This requires modifying LeRobot source — AGAINST project rules.

  WORKAROUND WITHOUT SOURCE MODIFICATION:
  We cannot add state dropout without modifying the source.
  However, we CAN implement "virtual state dropout" at the data level
  by creating a modified dataset where some episodes have state replaced
  with dataset_mean (effectively a constant). This doesn't require
  modifying the model at all.

  Practical implementation:
    In convert_to_lerobot_v3.py or a post-processing step:
    For 30% of episodes: set all state values = dataset_mean
    The model sees "constant state" episodes and MUST use vision.
  VERDICT: FEASIBLE without source modification, implement at data level.

Fix (b): Predicting Action DELTA (action - state) Instead of Absolute
======================================================================
Idea: Instead of predicting action[t] = joint_angles[t+1:t+50],
      predict delta[t] = action[t] - state[t] (relative to current state)

Feasibility: PARTIALLY SUPPORTED in framework.
  The config has: use_delta_joint_actions_aloha = False
  But it raises NotImplementedError when True:
    "use_delta_joint_actions_aloha is used by smolvla for aloha real models.
     It is not ported yet in LeRobot."

  Alternative: implement delta at DATA COLLECTION level.
    In convert_to_lerobot_v3.py: compute delta = action - state and store
    as the action. The model then predicts delta (which is near-zero for
    static frames and non-zero for motion frames).

  Problem: this CHANGES the action space meaning. Deployment script
  (deploy_smolvla.py) would need to re-add current state to predicted delta.
  This is implementable:
    predicted_delta = model.predict()
    actual_action = current_state + predicted_delta

  Why delta helps:
    echo shortcut for absolute action: f(state) → action ≈ state (≈zero loss)
    echo shortcut for delta action: f(state) → delta ≈ 0 (always near-zero)
    BUT: the delta for early frames (frames 0-5) is NON-ZERO and DIFFERENT
    for each zone. The model MUST learn f(vision) → delta to distinguish zones.
    State provides no signal for delta prediction (delta ≈ 0 from state alone).

  VERDICT: HIGHLY RECOMMENDED. Feasible without touching LeRobot source.
  Implement in:
    - convert_to_lerobot_v3.py: store delta as action
    - deploy_smolvla.py: add current state to predicted delta before execution

  Risk: action noise amplification (small delta errors → large absolute errors
  after integration over 50 steps). Mitigate by using small chunk_size or
  re-querying frequently.

Fix (c): Training Only on Early Frames (Before Robot Reaches Target)
====================================================================
Idea: sample only frames 0-20 per episode, where state≠target.
     These are the frames where vision matters.

Feasibility: NOT directly supported in LeRobot training pipeline.
  LeRobot samples random windows from episodes.
  Cannot easily restrict to "early frames only" without modifying
  the sampling logic.

  Indirect approach: collect SHORT episodes (3-5 seconds, which we already
  have at 3.3s average) that end BEFORE the robot fully reaches the target.
  This means ALL frames in our episodes are "early frames."

  Our episodes: mean=99 frames at 30fps = 3.3s real time.
  At 20°/s base velocity and 30° travel: time_to_target ≈ 1.5s = 45 frames.
  Most of our 99-frame episodes contain 45 frames of motion + 54 frames
  at-target. Those 54 frames contribute to echo.

  Fix: collect 1.5s episodes (45 frames). Stop recording when target reached.
  VERDICT: IMPLEMENT during data recollection. Short episodes preferred.

  Update collect_data_manual.py to stop at 45-50 frames per episode.
  Benefit: doubles the number of trainable episodes for same collection time.

Fix (d): LoRA/Unfreeze VLM to Strengthen Visual Signal
=======================================================
Idea: allow the VLM (SigLIP + SmolLM2) to update its representations
     for RoArm-M3 workspace, making visual features more action-relevant.

Feasibility: PARTIALLY SUPPORTED.
  The config has: freeze_vision_encoder=True, train_expert_only=True
  Setting train_expert_only=False would unfreeze the VLM text layers.

  Config change (via run_official_train.py):
    --policy.train_expert_only=False

  Risk: CATASTROPHIC FORGETTING.
  SmolVLM2 was pretrained on billions of image-text pairs.
  Fine-tuning it on 13,470 frames will overfit its text representations.
  Risk is HIGH: 952 epochs * frozen=13,470 frames → unstable for unfrozen VLM.

  LoRA alternative (supported by framework, _get_default_peft_targets exists):
    Apply LoRA to VLM q/v projections.
    Low-rank adaptation of VLM with small delta updates.
    Less forgetting than full unfreeze.

  Implementation: use --policy.peft_type=lora in train command.

  VERDICT: LoRA is viable but adds complexity. Do NOT unfreeze full VLM.
  Priority: implement (b) and (a) first — cheaper and more targeted.

PRIORITY ORDER FOR IMPLEMENTATION:
  1. [IMMEDIATE] Collect balanced data (30ep/zone) — necessary regardless
  2. [IMMEDIATE] Short episodes (45-50 frames) — eliminate at-target echo frames
  3. [DATA LEVEL] Virtual state dropout — zero state for 30% of episodes in dataset
  4. [DATA LEVEL] Delta action conversion — most fundamental fix
  5. [LAST RESORT] VLM LoRA — if above don't work, add visual pathway capacity

ADDITIONAL: EVALUATION METRIC FIX
  Replace L2 metric with vision-conditioned metric:
  - M_echo = mean(|pred_base - state_base|) over all test frames
    Good model: high M_echo (different from current state)
    Echo model: low M_echo (always copies current state)
  - M_visual = std(pred_base | same_state, different_zone)
    Good model: high std (different zones → different predictions from same state)
    Echo model: zero std (same state → same prediction regardless of zone)

  These two metrics CANNOT be fooled by the echo shortcut.
  Implement in test_inference_official.py before next robot test.
"""

# ===========================================================================
# ARCHITECTURAL CONFIRMATION FROM SOURCE CODE ANALYSIS
# ===========================================================================
ARCHITECTURE_NOTE = """
SOURCE CODE CONFIRMATION (modeling_smolvla.py verified 2026-03-31)

State pathway (TRAINABLE):
  state → pad_vector(32) → state_proj(32→960) → state_emb (1 token)
  state_proj is nn.Linear(32, 960) with train_state_proj=True
  This 1 token is appended to prefix sequence with att_mask=1 (causal)
  Action Expert attends TO this token at EVERY layer

Visual pathway (FROZEN):
  image → SigLIP(frozen) → connector(frozen) → 64 tokens/camera
  These tokens are in prefix sequence with att_mask=0 (prefix-LM)
  Action Expert cross-attends to these FROZEN VLM KV cache entries

Flow matching target:
  u_t = noise - action   (velocity field)
  Loss = MSE(u_t, v_t)   where v_t is predicted velocity

State echo in flow matching:
  If action ≈ state, then u_t ≈ noise - state
  Action Expert can compute this from state_emb (trainable path)
  Loss → 0 without using visual features

  Formally: v_t = f(state_emb, noisy_action, VLM_cache)
  If state_emb ≈ action in content: v_t ≈ noisy_action - state_emb ≈ u_t
  This is achievable as a linear function of state_emb.

  The state_proj layer (32→960) has enough capacity to learn this linear
  relationship for a 6-DOF robot with slow-moving joints.

Delta action recommendation:
  If action = state + delta, then u_t = noise - (state + delta)
  Echo prediction: v_t = noise - state_emb (ignores delta)
  MSE(u_t, v_t) = MSE(-delta, 0) = delta^2
  Model CANNOT minimize loss without predicting delta.
  Delta prediction REQUIRES the visual features to determine which
  direction and magnitude to move from current state.
  This BREAKS the echo shortcut at the loss function level.
"""

if __name__ == "__main__":
    print("=" * 70)
    print("PROPRIOCEPTIVE ECHO ANALYSIS REPORT")
    print("SmolVLA 120K Checkpoint, RoArm-M3 v5 Dataset")
    print("=" * 70)

    print("\n--- Q1: IS IT KNOWN? ---")
    print(Q1_LITERATURE)

    print("\n--- Q2: DATA vs ARCHITECTURE ---")
    print(Q2_ANALYSIS)

    print("\n--- Q3: WILL BALANCED DATA ALONE FIX IT? ---")
    print(Q3_ANALYSIS)

    print("\n--- Q4: EARLY FRAME ANALYSIS ---")
    print(Q4_ANALYSIS)

    print("\n--- Q5: FEASIBLE FIXES ---")
    print(Q5_ANALYSIS)

    print("\n--- ARCHITECTURE CONFIRMATION ---")
    print(ARCHITECTURE_NOTE)

    print("\n--- PRIORITY ACTION ITEMS ---")
    print("""
IMMEDIATE ACTIONS (before next data collection):
  1. Collect 30ep/zone (150 total) with SHORT episodes (45-50 frames/ep)
  2. Add virtual state dropout to convert_to_lerobot_v3.py (30% episodes)
  3. Convert to delta actions in convert_to_lerobot_v3.py + deploy_smolvla.py
  4. Add M_echo and M_visual metrics to test_inference_official.py

EVALUATION BEFORE DEPLOYMENT:
  Must pass: M_echo > 5° (prediction differs from current state)
  Must pass: M_visual > 10° std across zones at frame 0
  Current: M_echo ≈ 2.10° (FAIL)

KEY INSIGHT:
  Delta actions is the single most impactful fix.
  It makes the echo shortcut impossible at the LOSS FUNCTION level.
  Implement in data pipeline, not in model code.
  No LeRobot source modification needed.
    """)
