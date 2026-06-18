# Session 2026-06-15 - Cube10cm top-view 0-999 render and labels D246

## Scope

- Active branch: professor 10cm / 0.72kg cube top-view visual trajectory dataset
  camera-contract branch.
- User approved local runtime execution after D245 cleanup.
- This session ran local Isaac render and post-render label validation only.
- No PPO, L2, Large PPO, VLA/SmolVLA fine-tuning, action-teacher, RoArm
  deployment, RunPod runtime, B200/SSH/pull, `.ssh` copy, deletion, move, or
  archive was run.

## Commands

Render:

```bash
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 conda run -n isaaclab --no-capture-output python -u sim_scripts/cube10cm_top_view_visual_manifest_render.py --render-approved --device cuda:0
```

Post-render labels:

```bash
python3 sim_scripts/cube10cm_top_view_postrender_label_validation.py --render-dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242 --out-dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/postrender_label_validation_d246 --expected-episodes 1000 --expected-frames-per-episode 195
```

## Render Result

Output root:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242
```

Primary summary:

- artifact: `cube10cm_top_view_visual_manifest_render_d242`
- runtime: `ISAAC_RENDER_ONLY_MANIFEST_FED_NO_TRAINING`
- camera contract: `cube10cm_top_view_v1_candidate`
- episodes: `1000`
- frames: `195000`
- resolution: `1280x720`
- target fps: `30`
- elapsed: `28349.806646108627s` (about `7.88h`)
- effective captured FPS: `6.878353790351732`
- raw PNG bytes: `51386208295`
- raw PNG cost: `51.386208294999996MB/episode`
- contract violations: `[]`
- manifest bucket counts:
  - `debug_camera_anchor=50`
  - `clean_prior_candidate=650`
  - `transition_mixed_probe=200`
  - `overshoot_eval_candidate=100`

Independent checks:

- `frames.jsonl`: `195000` rows
- `raw_env_render_frames/*.png`: `195000` files
- render root size: about `49G`

## Label Result

Label output root:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/postrender_label_validation_d246
```

Primary summary:

- expected/actual episodes: `1000/1000`
- actual frames: `195000`
- per-episode frame count: all pass (`195` frames/episode)
- camera contract pass: `986/1000`
- label status counts:
  - `clean_useful_tap=819`
  - `contact_reaction_with_overshoot=167`
  - `camera_quality_fail=14`
- missing contact/reaction count: `0`
- raw useful-clean numeric count: `829`
- raw overshoot numeric count: `171`
- legacy target-band count: `817`
- centroid error median over episodes: `3.204588743804298px`
- centroid error max over episodes: `33.66615168193798px`

Critical interpretation:

- `split_candidate` is still an intended sampling bucket, not a final label.
- Use camera-gated `label_status` or numeric labels for filtering.
- Camera-gated usable outcomes are `819` clean useful taps and `167` overshoot
  taps.
- Raw event labels include camera failures: `829` clean and `171` overshoot.
  The 14 camera-quality failures are 10 clean-event and 4 overshoot-event rows.
- 13 camera failures are clean-prior reprojection-gate failures slightly above
  the `20px` gate.
- Episode `721` is the stronger camera-design warning: transition bucket,
  `projection_inside_frames=7/195`, centroid max `33.66615168193798px`.

## Runtime Cleanup

- The renderer printed:

```text
[cube10cm-top-view-manifest-render] done frames=195000 effective_fps=6.878 png_mb_per_ep=51.39
```

- The script then skipped `sim_app.close()` by design because local Kit close can
  hang.
- After artifact verification, the remaining completed process was terminated.
  The log therefore contains a post-completion killed-process line, but the
  render summary, frame counts, and label outputs are complete.
- No active Isaac/Kit/render process remains.

Post-run storage:

```text
df -h . -> 590G total, 528G used, 33G available, 95% used
```

GPU state:

- Render used the local `NVIDIA GeForce RTX 4090 Laptop GPU`.
- During render, VRAM was about 7.6GiB and utilization was usually limited by
  Isaac render, CPU metadata work, and PNG file I/O rather than a custom CUDA
  kernel.
- After cleanup, GPU returned to the small pre-existing process baseline; no
  cube10cm render process remained.

## Decision

`LOCAL_0_999_RENDER_D242_COMPLETE_POSTRENDER_LABELS_D246`

The branch now has a local raw rendered 0-999 corpus and post-render numeric
labels. It is not yet a LeRobot v3 dataset.

Next gated work:

- LeRobot v3 conversion/load validation.
- Companion metadata generation.
- Codec/decode validation.
- PNG extraction proof.
- Source-vs-decoded pixel-diff validation.
- Row alignment checks.

Blocked until explicit approval:

- LeRobot conversion if storage preflight is not acceptable.
- Any deletion/move/archive.
- 1000/10000 expansion beyond this raw 0-999 render.
- PPO/L2/Large PPO.
- VLA/SmolVLA fine-tuning.
- Action-teacher work.
- RoArm deployment.
- RunPod runtime.
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy.
- Track A work.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `sim_scripts/cube10cm_top_view_visual_manifest_render.py`
- `sim_scripts/cube10cm_top_view_postrender_label_validation.py`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/render_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/postrender_label_validation_d246/label_validation_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/postrender_label_validation_d246/episode_labels.csv`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242_stdout.log`
