# Cube10cm Top-View Chunk100 Sampling Contract D236

Status: pre-render sampling contract draft. This file does not authorize
IsaacLab rendering, 0-99 generation, 0-999 generation, deletion/archive/move,
training, action-teacher work, RoArm deployment, SSH/B200, pull, or Track A.

## Why This Exists

The next runtime target is a 0-99 episode chunk, not a full 0-999 dataset. A
100 episode chunk should not be made by simply repeating the D233 five smoke
poses. That would test the camera pipeline but would be a weak trajectory
dataset.

D232 requires future data to be split explicitly into:

- `train_success`
- `eval_boundary`
- `eval_failure`
- `debug_smoke`

Sampling ranges and seeds must be recorded in metadata. D235 fixed the
metadata layout, but it did not define the sampling manifest.

## Current Evidence

D233 smoke camera poses:

- `(0.24, 0.00)`
- `(0.14, -0.10)`
- `(0.14, +0.10)`
- `(0.34, -0.10)`
- `(0.34, +0.10)`

D230 useful-tap evidence:

- fixed xy10 corners clean-passed useful tap under the no-legacy-success
  interpretation;
- randomized xy10 band had stable overshoot failures across seeds;
- one overshoot sample replayed as a fixed pose clean-passed, so the failure is
  randomized-band / trajectory-conditioning, not a durable single fixed pose.

D225-D228 boundary evidence:

- close-x/high-y and transition x-band regions are useful as evaluation
  candidates, but they came from the RL/metric branch and need camera coverage
  and post-render label validation before becoming dataset claims.

## Critical Interpretation

Do not pre-label episodes as final "success" or "failure" solely from their
sampling bucket. Use the bucket as an intended split, then validate labels from
post-render metadata:

- contact seen
- reaction seen
- overshoot seen
- visibility
- reprojection quality
- frame count
- LeRobot decode

For the first 0-99 chunk, the intended split should be conservative:

- enough `debug_smoke` to preserve the D233 camera-contract anchor;
- mostly `train_success_candidate` trajectories in the already visible xy10
  workspace;
- a small `eval_failure_candidate` slice for randomized-band overshoot audit;
- only a small `eval_boundary_candidate` slice until camera coverage and labels
  are checked.

## Proposed 0-99 Candidate Split

This is a draft for the next manifest, not yet rendered:

| Episode IDs | Count | Split candidate | Purpose |
|---|---:|---|---|
| `000-004` | 5 | `debug_smoke` | Replay D233 five camera-contract poses for continuity |
| `005-069` | 65 | `train_success_candidate` | Stratified visible xy10 workspace; final label must be posthoc |
| `070-084` | 15 | `eval_failure_candidate` | Randomized xy10 diagnostic slice for overshoot/robustness labels |
| `085-099` | 15 | `eval_boundary_candidate` | Boundary/transition candidates from prior pose-bin work; camera coverage and labels must be checked |

## Candidate Sampling Rules

`debug_smoke`:

- fixed poses exactly match D233:
  `(0.24,0.00)`, `(0.14,-0.10)`, `(0.14,+0.10)`, `(0.34,-0.10)`,
  `(0.34,+0.10)`.

`train_success_candidate`:

- sample inside the camera-validated xy10 rectangle:
  - `x in [0.14, 0.34]`
  - `y in [-0.10, +0.10]`
- use stratified coverage rather than pure random sampling.
- write the seed and cell id into companion metadata.
- do not call the label final until post-render contact/reaction/no-overshoot
  and visibility gates are computed.

`eval_failure_candidate`:

- sample inside the same xy10 rectangle using randomized seeds.
- purpose is to preserve the D230 randomized-band overshoot diagnostic in the
  visual dataset.
- these episodes are not training-success demonstrations unless posthoc labels
  say they are clean.

`eval_boundary_candidate`:

- use a small count only.
- candidate regions can draw from D225-D228 close-x/high-y and transition x-band
  evidence.
- because some prior boundary evidence uses `y=+0.15`, first verify top-view
  coverage and occlusion before treating these as regular dataset samples.

## Manifest Fields Required Before Render

Each planned episode row should include:

- `episode_index`
- `split_candidate`
- `cube_x_m`
- `cube_y_m`
- `seed`
- `sampling_rule`
- `sampling_cell_id`
- `source_decision`
- `requires_posthoc_label_validation`

After render, companion metadata should add actual labels:

- `contact_seen_any`
- `reaction_seen_any`
- `overshoot_seen_any`
- `full_visibility_frames`
- `partial_visibility_frames`
- `full_occlusion_frames`
- `centroid_error_px_max`
- `label_status`

## Gate Before Writing A Chunk Renderer

Before implementing or running the 0-99 chunk renderer:

1. Confirm this sampling contract or revise it.
2. Generate a manifest from this contract.
3. Confirm local free space remains above the D234 threshold.
4. Confirm the target output root is fresh.
5. Keep the D233 smoke script capped at 1-10 episodes.
6. Build a separate chunk script or explicit chunk mode that consumes the
   manifest and refuses to run without it.

## Still Blocked

- IsaacLab render
- 0-99 chunk generation
- 0-999 / 1000 / 10000 expansion
- deletion/archive/move
- PPO/L2/Large PPO
- SmolVLA/VLA fine-tuning
- action-teacher work
- RoArm deployment
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy
