# Session 2026-06-17 - Cube10cm Top-View Method Pipeline Reframe D254

## Scope

Active branch:

- Professor 10cm / 0.72kg cube top-view visual trajectory dataset branch.

User corrected the framing: the professor is not merely asking to see example
validation data. The professor is asking whether this method/pipeline can be
used: sim generation, camera contract, label validation, dataset formatting,
and learning/evaluation connection.

This session therefore did documentation and framing correction only.

No Isaac Lab render, SmolVLA/VLA fine-tuning, PPO/L2/Large PPO, action-teacher,
RoArm deployment, RunPod runtime, deletion, archive, move, B200/SSH/pull, `.ssh`
copy, or Track A work was run.

## Checks Performed

- Ran `git status --short --untracked-files=all --branch`.
- Re-read `CLAUDE.md` Current-State Protocol references.
- Re-checked `START_HERE.md` D232-D253 current truth.
- Re-checked these repo logs:
  - D232 camera contract / visual dataset direction;
  - D246 Isaac Lab 0-999 render and post-render labels;
  - D247 LeRobot v3 AV1 conversion and metadata;
  - D248 label package and camera-fail audit;
  - D249-D252 freeze, filtered loader, and distribution audit;
  - D253 training-input preflight.

## Correction

Previous D253 wording made the next possible runtime look like a 50-step
SmolVLA training smoke. That was technically available after preflight, but it
was not the right professor-facing framing.

Correct framing:

- D246-D253 prove a dataset method pipeline up to training-input readiness.
- They do not prove model performance.
- They do not show Isaac Lab training.
- They do not mean SmolVLA training is now the core result.
- SmolVLA smoke is only an optional next gate for training-loop connectivity.

## Added Document

Added:

- `claudedocs/cube10cm_top_view_method_pipeline_d254.md`

Purpose:

- explain the full method pipeline in professor-facing terms;
- separate Isaac Lab data generation from LeRobot storage and model training;
- define English terms in Korean;
- state what D246-D253 proved and what they did not prove;
- mark SmolVLA smoke as optional connectivity verification, not the main claim.

## Updated Current Truth

Updated:

- `START_HERE.md`

Key changes:

- `Last updated` now points to D254.
- `Latest Result` is now D254.
- Active next work no longer says "50-step SmolVLA smoke" as the immediate core
  next research step.
- Recommended order is:
  1. use D254 as the professor-facing method/pipeline framing;
  2. only if explicitly approved, run a 50-step SmolVLA smoke to verify
     training-loop connectivity;
  3. after an approved checkpoint exists, build offline evaluation for
     `eval_clean_holdout` and `eval_overshoot_diagnostic`;
  4. keep `quarantine_camera_fail` excluded.

## Current Pipeline State

Completed:

1. Camera contract direction.
2. Isaac Lab 0-999 visual trajectory generation.
3. Post-render numeric label validation.
4. LeRobot v3 AV1+parquet conversion.
5. Train/eval/quarantine split curation.
6. Official LeRobot training-input preflight.

Not completed:

1. SmolVLA training.
2. Offline model evaluation on held-out clean and overshoot diagnostic splits.
3. Real RoArm deployment.
4. 1000/10000 expansion.
5. Raw cleanup/archive beyond already approved prior actions.

## Verdict

`D254_METHOD_PIPELINE_FRAMING_LOCKED_NO_TRAINING`

The branch should now be described as a method-pipeline proof through
training-input readiness, not as a model-training result and not merely as a
visual sample showcase.

## Blocked Until Explicit Approval

- 50-step SmolVLA training smoke.
- 20k SmolVLA candidate training.
- Offline evaluation implementation after a checkpoint exists.
- Any RunPod/H100 job.
- Any extra Isaac Lab render or 1000/10000 expansion.
- Any PPO/L2/Large PPO/action-teacher/RoArm deployment work.
- Any deletion/archive/move/cleanup.
- Any B200/SSH/pull/.ssh copy.
