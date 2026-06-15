# Session 2026-06-13 - Cube10cm Top-View Chunk100 Render + Labels D239

Status: local 0-99 render and post-render numeric label validation complete.
This session did not run LeRobot conversion, companion metadata conversion for
the new chunk, training, RoArm deployment, SSH/B200, pull, deletion, move, or
archive.

## Scope

- Branch scope: professor 10cm / 0.72kg cube top-view visual trajectory dataset.
- Explicit approved runtime: local 0-99 render, then numeric labels per episode.
- Out of scope: PPO/L2/Large PPO, SmolVLA/VLA fine-tuning, action-teacher,
  RoArm deployment, B200/SSH, and Track A.

## Preflight

- `git status --short --untracked-files=all --branch` showed
  `## master...origin/master` before runtime.
- Local disk before the successful rerun was about `32GB` free; after d241 it is
  about `26-27GB` free.
- The default/root attempts were preserved, not deleted:
  - `cube10cm_top_view_visual_chunk100_d235`: earlier interrupted partial,
    about `127MB`.
  - `cube10cm_top_view_visual_chunk100_d239`: sandbox GPU/Vulkan failure,
    manifest-only about `24KB`.
  - `cube10cm_top_view_visual_chunk100_d240`: host GPU render started but was
    stopped after `345` frames / about `88MB` because `quat_rotate` warnings made
    the log unusable.

## Runtime Fix

The host GPU path was required because the sandboxed Isaac/Kit process could not
see the needed CUDA/Vulkan device. After that, the render worked but printed a
large IsaacLab warning stream for deprecated `quat_rotate`.

Applied a small code patch in `roarm_rl/roarm_cube_push_env.py`:

- import `quat_apply` instead of `quat_rotate`;
- replace two `quat_rotate(...)` calls with `quat_apply(...)`.

`python -m py_compile` passed. This is the same quaternion-vector application
semantics through IsaacLab's non-deprecated API; it was needed to keep the 100ep
runtime log usable.

## Successful Render

Successful root:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d241`

Render summary:

- `num_episodes`: `100`
- frames: `19500`
- resolution: `[1280, 720]`
- splits: `debug_smoke=5`, `train_success_candidate=65`,
  `eval_failure_candidate=15`, `eval_boundary_candidate=15`
- elapsed seconds: `4647.953013896942`
- effective render FPS: `4.195395250704307`
- raw PNG total: `5142551626` bytes
- debug PNG MB/episode: `51.42551626`
- projection: 100ep raw PNG about `5.142551626GB`, 1000ep about
  `51.42551626GB`, 10000ep about `514.2551626GB`
- all-frame visibility: `19500/19500` full
- contact-window visibility: `18372/18372` full
- reprojection centroid error median/max:
  `3.0758927127400306px` / `17.06565232897021px`
- contract violations: `[]`

The render process wrote `frames.jsonl`, `render_summary.json`, copied
`episode_manifest.csv`, and produced exactly `19500` PNG files.

## Process Cleanup

Isaac/Kit did not exit cleanly after writing all frames. I verified the frame
count and files first, then killed only leftover cube10cm render PIDs:

- old D232 smoke sanity render PID;
- d240 partial render PID;
- d241 completed render PID.

No files were deleted, moved, or archived. GPU process list after cleanup showed
no remaining cube10cm render process.

## Post-Render Numeric Labels

Added:

`sim_scripts/cube10cm_top_view_postrender_label_validation.py`

The script reads only `frames.jsonl` and writes:

- `postrender_label_validation_d241/episode_labels.csv`
- `postrender_label_validation_d241/episode_labels.json`
- `postrender_label_validation_d241/label_validation_summary.json`

Label policy:

- Sampling buckets are not final labels.
- Episode labels come from actual rendered frame metrics:
  contact seen, reaction seen, overshoot seen, legacy target-band success,
  full visibility, projection-inside count, frame count, and reprojection error.
- Default reprojection max gate is explicit in the output:
  `reprojection_max_gate_px=20.0`.
- Because runtime `tap_overshoot_now` is defined from displacement only
  (`disp_xy >= tap_overshoot_disp_m`), status uses
  `contact_reaction_with_overshoot`, not "overshoot after contact".

Label summary:

- episodes: `100`
- frames: `19500`
- frame count OK: `true`
- camera contract pass: `100/100`
- contact seen: `100/100`
- reaction seen: `100/100`
- missing contact/reaction: `0`
- overshoot seen: `39/100`
- overshoot before contact: `9/100`
- legacy target-band success: `62/100`
- useful clean: `61/100`
- status counts:
  - `clean_useful_tap`: `61`
  - `contact_reaction_with_overshoot`: `39`

Split by status:

- `debug_smoke`: `3` clean, `2` with overshoot
- `train_success_candidate`: `49` clean, `16` with overshoot
- `eval_failure_candidate`: `8` clean, `7` with overshoot
- `eval_boundary_candidate`: `1` clean, `14` with overshoot

Critical interpretation:

- The camera coverage concern for `y=0.15` is closed for this 100ep chunk:
  boundary candidates all rendered with full visibility and projection-inside
  frames, so they are camera-covered.
- The sampling buckets are not equivalent to final labels. In particular,
  `train_success_candidate` still contains `16` overshoot episodes, and
  `eval_failure_candidate` contains `8` clean useful episodes.
- For useful-tap dataset selection, use `label_useful_clean_numeric`, not
  `split_candidate`.

## Still Blocked

- LeRobot AV1 conversion for d241.
- Companion metadata generation for d241.
- LeRobot load/decode, PNG extraction, source PNG vs decoded MP4 pixel diff, and
  row alignment for d241.
- Any 0-999 / 1000 / 10000 expansion.
- Any deletion/move/archive cleanup.
- PPO/L2/Large PPO, SmolVLA/VLA fine-tuning, action-teacher, RoArm deployment.
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy.
