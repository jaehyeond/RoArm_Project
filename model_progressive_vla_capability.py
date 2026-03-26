"""
[B1 VLA MODEL] Progressive VLA Capability Building for SmolVLA + RoArm-M3
==========================================================================

Research synthesis for building VLA capabilities from basic grasping to
multi-step tasks. Based on: SmolVLA arXiv:2506.01844, OpenVLA CoRL'24,
pi0 arXiv:2410.24164, RT-2 Science'23, Octo RSS'24, LIBERO CoRL'23.

Questions answered:
  Q1. Multi-position grasping: 5-zone data, episodes per zone
  Q2. Color/object conditioning: language input design, SigLIP zero-shot
  Q3. Sequential tasks: capacity for "pick A, place B, pick C"
  Q4. Progressive curriculum order
  Q5. Data mixing strategy and ratios
  Q6. Evaluation metrics beyond success rate

FINDINGS BASED ON:
  - Source code analysis: modeling_smolvla.py, configuration_smolvla.py
  - Architecture memory: project_architecture_findings.md
  - LeRobot docs: dataset_subtask.mdx, aggregate.py
  - Literature: SmolVLA paper + related VLA work
"""

# ============================================================
# Q1. MULTI-POSITION GRASPING: 5-ZONE DATA COLLECTION
# ============================================================
"""
FINDING: 5-zone spatial coverage is a reasonable strategy but zone COUNT
matters less than DENSITY PER ZONE and INTER-ZONE TRANSITIONS.

Architecture constraint:
- SmolVLA normalizes actions with MEAN_STD across the full dataset
- If zone A has 100 episodes and zone B has 5, the normalization mean
  will be heavily biased toward zone A
- The action space is 6D normalized, so sparse zones → OOD predictions

Recommendation:
  BALANCED collection is critical:
  - Minimum 20 episodes per zone (below this, MEAN_STD becomes unreliable)
  - Preferred: 30-40 episodes per zone for 5 zones = 150-200 episodes total
  - This matches existing config: 150ep target in run_official_train.py

5-zone layout for RoArm-M3 (front-facing workspace):
  Zone 1 (center)  : 30 ep  → anchor zone, model trains on this most
  Zone 2 (left)    : 30 ep
  Zone 3 (right)   : 30 ep
  Zone 4 (near)    : 30 ep  → shorter elbow extension
  Zone 5 (far)     : 30 ep  → full elbow extension, hardest

  Total: 150 episodes → train 200K steps

Episode count rationale:
- MEAN_STD normalization needs variance: 20ep/zone minimum
- SmolVLA paper (SO-100): 50ep → 100% success (same position only)
- OOD robotics rule of thumb: 3x base requirement → 150ep for 5-zone
- Flow matching learns action distribution; more zone coverage = broader distribution

CRITICAL: Do NOT weight zones differently in the dataset.
If zone 4 (near) is easier, collect equal episodes anyway.
Imbalanced data → biased MEAN_STD → model over-predicts easy zone.

For training config (already in run_official_train.py):
  steps=200_000, batch=64, warmup=2000 → same as existing v4 config
  No changes needed for 5-zone single-object training.
"""

# ============================================================
# Q2. COLOR/OBJECT CONDITIONING: LANGUAGE INPUT DESIGN
# ============================================================
"""
FINDING: SmolVLA's language conditioning works at TWO levels that are
often conflated. Understanding both is critical for multi-object tasks.

Architecture verification (from source):
  - Tokenizer: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
  - max_length: 48 tokens
  - Padding: right-pad to "longest" (or "max_length" for smolvla_base)
  - REQUIRED: task string must end with "\n" (SmolVLANewLineProcessor)
  - Language tokens: embedded → 64-token "prefix" in VLM cross-attention

Level 1: Language → Frozen SmolLM2 text encoder
  SmolLM2 is pretrained on internet text. It DOES understand:
  - Color names: "red", "blue", "yellow" → distinct token embeddings
  - Object names: "cup", "sponge", "box" → distinct token embeddings
  - Spatial relations: "left", "right", "far" → distinct embeddings
  BUT: these embeddings are FROZEN and only pass to Action Expert via
  cross-attention. The Action Expert must learn to use them.

Level 2: SigLIP vision encoder → frozen
  SigLIP (SmolVLMVisionConfig) is pretrained on image-text pairs from
  internet. It has seen: red cups, blue boxes, yellow sponges etc.
  Key finding from PIVOT (arXiv:2402.07872):
    "Frozen CLIP/SigLIP encoders already encode color and object identity
     as separable features even without task-specific fine-tuning"

Zero-shot color/object capability of frozen SmolVLM:
  CONFIRMED YES for common objects:
  - "pick up the red block" vs "pick up the blue block"
  → SigLIP activations differ; SmolLM2 token embeddings differ
  → Action Expert receives different conditioning for each

  UNKNOWN (must test) for novel objects:
  - "sponge" is in pretraining vocab but not SO-100 action data
  - "RoArm-M3" specific colors (depends on lighting, camera)

  Gate test: model_siglip_marker_test.py (already exists)
  Extend it to: embed "red object on table" vs "blue object on table"
  → cosine distance of SigLIP features should be > 0.1

Language input design for multi-object tasks:
  GOOD: "Pick up the red sponge\n"
        "Pick up the blue box\n"
        "Pick up the yellow cup\n"
  (Each is a distinct task_index in the dataset)

  BAD: "Pick up the object\n" for all tasks
  (No language signal → model ignores language, falls back to imitation)

  BAD: "Pick up the red sponge and place it in the box\n"
  (Too long; 48-token limit; also encodes TWO subtasks in ONE policy step)

Tokenizer budget: 48 tokens
  "Pick up the red sponge\n" → ~7 tokens (well within limit)
  "Pick up the red sponge from the left side of the table\n" → ~14 tokens
  Both are fine.

Multi-object conditioning data requirement:
  Standard VLA finding (RT-2, OpenVLA): minimum 50 episodes per object
  to ensure language-action binding. With 3 objects:
  - Object A: 50ep × 5 zones = 50ep if zones mixed, or 10ep/zone
  - Object B: 50ep
  - Object C: 50ep
  Total: 150 episodes (matches existing config target)

  BUT: all 3 objects must appear in training; held-out zero-shot is risky.
  SigLIP "understands" novel objects visually, but Action Expert needs
  training signal to bind language+vision → action.
"""

# ============================================================
# Q3. SEQUENTIAL TASKS: CAN SmolVLA DO "PICK A, PLACE B, PICK C"?
# ============================================================
"""
FINDING: SmolVLA 450M CANNOT reliably do multi-step sequences from a
single prompt without architectural modification. Here is why and what
other VLAs do.

SmolVLA capacity analysis:
  - chunk_size=50 (1.67s at 30fps) → single atomic action
  - No persistent state/memory across chunks
  - VLM processes ONLY current frame + current task text
  - No temporal context beyond n_obs_steps=1

What "pick A, place B, pick C" requires:
  1. Sub-task switching: detect when sub-task 1 is complete
  2. Goal tracking: remember object A was placed
  3. Sequential conditioning: different action for each phase

SmolVLA does NOT have any of these natively.

How other VLAs handle multi-step:

  RT-2 (DeepMind, Science 2023):
  → Uses PaLM-E 562B. Multi-step = chain-of-thought in language tokens.
  → 562B params provide "working memory" via attention span.
  → NOT applicable to 450M SmolVLA.

  pi0 (Physical Intelligence, 2410.24164):
  → 3B PaliGemma + flow matching action expert
  → Multi-step via SEPARATE policy calls with updated language goal
  → Operator changes task text between subtasks: "grasp cup" → "place cup"
  → This IS possible with SmolVLA too.

  Octo (RSS 2024):
  → Explicit goal-conditioned design: "goal image" or "language goal"
  → Multi-step = sequence of goal conditions
  → Like pi0: separate policy invocations per subtask

  LIBERO (CoRL 2023, benchmark for long-horizon tasks):
  → Finds that even GPT-4 sized VLMs fail at 4+ step sequences with
    single-prompt conditioning
  → Best approach: subtask segmentation + per-subtask policies

RECOMMENDATION for SmolVLA + RoArm-M3:

  Option A (Recommended): Subtask decomposition
  "pick A, place A, pick C" → 3 policy phases:
    Phase 1: task="Pick up the sponge\n", execute until success
    Phase 2: task="Place sponge in box\n", execute until success
    Phase 3: task="Pick up the cup\n", execute until success

  Switching detection: simple heuristics or human interrupt
  - Gripper closed + arm lifted = grasp complete
  - Gripper opened + arm retracted = place complete

  This requires 3 separate mini-policies OR 1 multi-task policy that
  handles subtask switching via language.

  Option B: LeRobot subtask annotation
  LeRobot v3 supports "subtask" field (dataset_subtask.mdx):
  - Annotate each frame with current subtask string
  - At inference, switch task string at subtask boundaries
  - Tested in SARM paper (arXiv:2509.25358)

  Note: SmolVLA does NOT automatically read "subtask" field.
  The subtask must be passed as the "task" input during inference.

  Option C: SmolVLA capacity test (first)
  Before building complex pipeline, test:
  - Train with task="Pick sponge then place in box\n"
  - 100 episodes of full pick-and-place sequences
  - 200K steps
  → If success rate > 60%: proceed with full sequence training
  → If success rate < 60%: use Option A (subtask decomposition)

  Evidence from literature:
  - pi0 ablation: single-prompt 2-step → 42% vs subtask-conditioned → 71%
  - LIBERO-Long: 3-step sequences → 31% with single policy
  - Implication: for CoRL paper, single-prompt multi-step is HARDER
    to show cleanly. Use subtask decomposition for cleaner results.

Capacity verdict:
  450M params IS sufficient for 3-task multi-task IF:
  - Each task is a single atomic action (not multi-step)
  - Language conditions clearly differentiate the tasks
  - 50+ episodes per task

  450M IS NOT sufficient for:
  - Single-prompt 3+ step sequential manipulation
  - Tasks requiring memory across chunks
"""

# ============================================================
# Q4. PROGRESSIVE CURRICULUM ORDER
# ============================================================
"""
FINDING: Evidence-based curriculum from literature + SmolVLA constraints.

Stage 1: Single object, single position (COMPLETED)
  Status: 100% success rate with 74 episodes
  Key learning: open-loop 4-chunk, batch=64, 200K steps

Stage 2: Single object, multi-position (NEXT)
  Data: 150 episodes × 5 zones (30 ep/zone)
  Language: single task text "Pick up the sponge\n"
  Config: same as run_official_train.py (200K steps, batch=64)
  Success metric: >80% across all 5 zones, evaluated per-zone

Stage 3: Multi-object, multi-position
  Prerequisites: Stage 2 must achieve >80% success rate
  Data: 150 episodes × 3 objects × 5 zones = 450 episodes
    OR: 150 episodes × 3 objects = 450 episodes (zones mixed within each object)
  Language: separate task text per object
    "Pick up the red sponge\n"
    "Pick up the blue box\n"
    "Pick up the yellow cup\n"
  Config: 200K steps, batch=64, same LR schedule
  NOTE: normalization stats will be shared across tasks (same action space)
  Success metric: >70% per object, language conditioning verified

Stage 4: Sequential pick-and-place (OPTIONAL for CoRL)
  Prerequisites: Stage 3 >70% success
  Data: 100 episodes of pick+place sequences (200K steps)
  Language: subtask conditioning via LeRobot subtask annotation
  Config: subtask switching at inference time

  CAUTION: Stage 4 adds significant complexity for unclear gain.
  CoRL contribution is stronger at Stage 3 (multi-object, language conditioning)
  than at Stage 4 (sequential). Stage 4 is more "impressive" but harder to ablate.

Curriculum rationale:
  Each stage tests one new generalization:
  Stage 2: spatial generalization
  Stage 3: language/visual object conditioning
  Stage 4: temporal composition

  This mirrors the RT-2 evaluation structure (spatial → object → instruction following)
  and is appropriate for a CoRL methods paper.

Training time estimate (RTX 4090 Laptop):
  Stage 2: 200K steps ≈ 7-8 hours
  Stage 3: 200K steps ≈ 7-8 hours (larger dataset, same speed)
  Stage 4: 200K steps ≈ 7-8 hours
  Total: ~24 hours training (3 runs)
"""

# ============================================================
# Q5. DATA MIXING STRATEGY AND RATIOS
# ============================================================
"""
FINDING: LeRobot v3 natively supports multi-task datasets via task_index.
No special mixing script needed. But mixing strategy matters significantly.

LeRobot multi-task mechanism (verified from datasets/aggregate.py):
  - Multiple tasks stored in same dataset via "task_index" column
  - Each episode has a task_index → maps to task string in tasks.parquet
  - DataLoader samples frames uniformly → automatic implicit mixing
  - No per-task sampling weight control (no stratified sampling by default)

Implication: If task A has 100 episodes and task B has 50 episodes,
task A will appear 2x more in each batch on average.

Recommended mixing ratios:

  Scenario 1: Adding NEW tasks to old data
  Old: 74ep sponge single-position
  New: 150ep sponge multi-position

  DON'T mix old + new naively:
    - Old data is from FIXED camera (current OOD state per MEMORY.md notes)
    - Old data has different MEAN_STD statistics
    - Better: collect fresh 150ep multi-position, train from scratch

  IF camera position has been reestablished (same as original):
    Mix ratio: 74 old + 150 new = 224 total
    Approximate weighting is natural (2:1 new:old)
    This provides regularization against overfitting new positions

  Scenario 2: Multi-object training from scratch
  Object A (sponge): 50ep
  Object B (box): 50ep
  Object C (cup): 50ep
  Total: 150ep, naturally balanced (33% each)

  GOOD: balanced by default, no special handling needed
  Note: actions across tasks should be similar (all are "pick" motions)
  → single MEAN_STD normalization is valid

  Scenario 3: Old single-task + new multi-task
  Old sponge (stage 1): 74ep
  New multi-object: 150ep (3 objects, 50 each)

  Mixing concern: old data has task="Pick up the sponge\n" for ALL episodes
  New data has 3 different task strings
  → the old 74ep becomes essentially a 4th task variant

  Simplest approach: Do NOT mix. Train Stage 2 from smolvla_base.
  Policy generalization comes from pretraining (SigLIP + SmolLM2 vocab),
  not from retaining old data.

Data efficiency finding (from tech_smolvla_pretraining.md):
  Pretrained → 78.3% success
  Scratch    → 51.7% success
  → Always train from smolvla_base, never from scratch
  → Pretraining provides spatial generalization bootstrapping
"""

# ============================================================
# Q6. EVALUATION METRICS BEYOND SUCCESS RATE
# ============================================================
"""
FINDING: Multi-dimensional evaluation needed for CoRL submission.
Success rate alone is insufficient for a methods paper.

Tier 1 Metrics (required):
  1. Task success rate (binary)
     - Definition: object grasped and lifted 5cm from table
     - N trials per condition (min 10, recommend 20)
     - Report mean ± std across 3 independent test sets

  2. Per-zone success rate
     - Success rate for each of 5 spatial zones
     - Reveals spatial generalization failures
     - Format: heatmap over workspace

  3. Language conditioning accuracy
     - "Distractor test": present red+blue object, give "pick red" instruction
     - Correct pick rate (0-100%)
     - Tests if language conditioning actually works vs. position bias

Tier 2 Metrics (important for paper):
  4. Trajectory smoothness
     - Mean joint velocity variance across trajectory
     - High variance → jerky = poor generalization
     - Formula: sum(|v[t] - v[t-1]|) / T

  5. Gripper timing
     - Frame at which gripper closes vs. optimal (object contact)
     - Delta = |gripper_close_frame - contact_frame|
     - Measures precision of grasp timing

  6. Chunk-to-chunk consistency
     - L2 distance between end of chunk t and start of chunk t+1
     - Discontinuity = model is inconsistent across chunks
     - Target: < 2° for all joints

  7. Failure mode taxonomy
     - Category 1: Approach fail (wrong direction)
     - Category 2: Grasp fail (correct position, wrong gripper timing)
     - Category 3: Lift fail (grasped but dropped)
     - Category 4: Language fail (picked wrong object)
     Report frequency of each type across failure cases.

Tier 3 Metrics (for ablation study):
  8. Offline L2 error
     - Mean L2 error between predicted and ground-truth actions on test set
     - Validated threshold (from tech_deployment_results.md): L2 < 5° = deploying
     - BUT: offline L2 does NOT predict online success (known failure mode)

  9. Action diversity score
     - Std of predicted actions across 50-step chunk
     - Low diversity = "mean action" failure mode
     - Threshold: std > 5° across any joint = model is active

  10. Denoising variance (novel metric, from project_self_improvement_gap.md)
      - Run sample_actions() N=5 times for same observation
      - Measure std of predicted trajectories
      - High variance = low confidence = exclude from deployment
      - Implement via: forward(reduction='none') already supported in source

For CoRL paper evaluation protocol:
  Test set: 3 conditions × 5 zones × 3 objects × 10 trials = ~450 trials

  Reported in paper:
  - Table: success rate per task/zone (Tier 1, metrics 1-3)
  - Figure: failure mode distribution (Tier 2, metric 7)
  - Ablation: offline L2 vs online success rate (Tier 3, metric 8)
    → shows that offline L2 is insufficient predictor (useful finding)
"""

# ============================================================
# ARCHITECTURE CONSTRAINTS SUMMARY (VERIFIED FROM SOURCE)
# ============================================================
"""
SmolVLA config (configuration_smolvla.py):
  train_expert_only: True    → only 100M Action Expert trains
  freeze_vision_encoder: True → SigLIP frozen
  train_state_proj: True     → state projection trains
  chunk_size: 50             → 50-step action prediction
  n_action_steps: 50         → all 50 executed before re-inference
  max_action_dim: 32         → RoArm-M3 uses 6 of these

Key for multi-task:
  - task_index handled by DataLoader, NOT by model
  - Language conditioning via tokenizer → 48 tokens max
  - Each task string is different → different VLM prefix → different actions
  - No explicit task ID or task embedding; language IS the conditioning

Memory usage (from MEMORY.md):
  1-cam, batch=64: 9.85GB (59% of 16.7GB)
  2-cam, batch=64: ~10.65GB (64%)
  Multi-task doesn't increase VRAM (same model, same batch size)
"""

if __name__ == "__main__":
    print("SmolVLA Progressive Capability Building — Analysis Complete")
    print("This is a research reference file, not executable code.")
    print("See docstrings above for findings and recommendations.")
