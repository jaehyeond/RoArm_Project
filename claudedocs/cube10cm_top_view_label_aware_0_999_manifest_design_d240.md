# Cube10cm Top-View Label-Aware 0-999 Manifest Design D240

Status: draft design only. No Isaac render, no 0-999 dataset generation, no
training, no deletion, no move, no archive.

## Scope

This design continues the professor 10cm / 0.72kg cube top-view visual
trajectory dataset branch. It does not use the old v6 dataset schema as the
research target. The existing v6 data was useful only as a codec/backend
reference. The target dataset follows the June 11 professor requirement:
frame-by-frame image-state pairs from IsaacLab top-view cube tap/push
trajectories, stored as LeRobot MP4 + parquet, with PNG used only for
smoke/debug/extraction.

## Terms

- Manifest: a render plan table. In Korean, think of it as an episode list or
  sampling plan. It decides which cube pose, seed, and intended bucket each
  episode should render.
- Sampling bucket: the intended source region for an episode before rendering.
  It is not a success/failure label.
- Post-render label validation: the numeric check after rendering. It reads the
  actual frames and assigns labels from observed contact, reaction, overshoot,
  camera visibility, reprojection, and frame count.
- Camera coverage target: a pose range that must stay visible and inside the
  projected image even if it later becomes an overshoot or negative example.

## D241 Evidence

Successful rendered root:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d241`

Rendered and labeled:

- Episodes: `100`
- Frames: `19500`
- Raw PNG storage: `5142551626` bytes, `51.42551626MB/episode`
- Render elapsed: `4647.953013896942s`
- Effective captured FPS: `4.195395250704307`
- Camera visibility: `19500/19500` full-frame full visibility
- Contact-window visibility: `18372/18372` full visibility
- Reprojection centroid median/max: `3.0758927127400306px` /
  `17.06565232897021px`
- Camera contract violations: `[]`

Post-render labels:

- Camera contract pass: `100/100`
- Contact seen: `100/100`
- Reaction seen: `100/100`
- Missing contact/reaction: `0`
- Useful clean tap: `61/100`
- Overshoot: `39/100`
- Legacy target-band success: `62/100`

Split bucket versus actual label:

| Sampling bucket | Clean | Overshoot | Camera pass |
| --- | ---: | ---: | ---: |
| `debug_smoke` | 3 | 2 | 5 |
| `train_success_candidate` | 49 | 16 | 65 |
| `eval_failure_candidate` | 8 | 7 | 15 |
| `eval_boundary_candidate` | 1 | 14 | 15 |

Y-band evidence:

| Y band | Count | Clean | Overshoot | Interpretation |
| --- | ---: | ---: | ---: | --- |
| low negative | 19 | 15 | 4 | clean-prior region |
| center | 34 | 28 | 6 | clean-prior / mixed region |
| positive mid | 32 | 17 | 15 | transition region |
| high boundary | 15 | 1 | 14 | camera-covered overshoot/eval region |

Critical interpretation:

- The camera coverage concern for the high-boundary samples is not the current
  blocker. D241 shows they are visible and projection-inside.
- The high-boundary samples are mostly overshoot outcomes, so they should not be
  counted as train-success demonstrations.
- `split_candidate` must remain a sampling bucket. Actual dataset filtering must
  use `label_useful_clean_numeric`, `label_overshoot_numeric`, and
  `label_camera_contract_numeric`.
- If the professor needs 1000 clean train-positive demonstrations, a naive
  0-999 render is not enough. D241's observed clean rate is `61%`; a direct
  extension would likely produce far fewer than 1000 clean episodes unless the
  sampling distribution is tightened or the render count is intentionally
  over-generated.

## D241 Format Gate

LeRobot conversion and validation for d241 passed:

- LeRobot root:
  `cube10cm_top_view_visual_chunk100_d241/lerobot_dataset_av1`
- Codec: `av1`
- Pixel format: `yuv420p`
- FPS: `30`
- Frames: `19500`
- Episodes: `100`
- Frame count match: `true`
- Video bytes total: `56604396`
- Video MB/episode: `0.56604396`
- Projected video size: `0.56604396GB/1000ep`,
  `5.6604396GB/10000ep`
- Sampled LeRobot decode avg/max: `0.008618485927581788s` /
  `0.09812450408935547s`
- Sampled PNG-vs-decoded MP4 max mean abs diff: `0.8940353732638889`
- Sampled PNG-vs-decoded MP4 max pixel abs diff: `74`
- Final LeRobot root size: about `56MB`
- Remaining temporary PNG in LeRobot root: `0`

Companion metadata gate passed:

- Per-frame metadata rows: `19500`
- Episode metadata rows: `100`
- LeRobot alignment checked: `true`
- Aligned keys: `index`, `episode_index`, `frame_index`
- LeRobot core columns: `observation.state`, `action`, `timestamp`,
  `frame_index`, `episode_index`, `index`, `task_index`

PNG extraction proof:

- Extracted from d241 AV1 MP4:
  `debug_extract_frames_d241/episode_000099_frame_000050.png`
- Resolution: `1280x720`
- Same-frame source-vs-extracted mean abs diff: `0.7776012731481482`
- Same-frame source-vs-extracted max abs diff: `30`

## Proposed 0-999 Manifest Structure

Every row should include at least:

- `episode_index`
- `intended_sampling_bucket`
- `intended_role`
- `cube_x_m`
- `cube_y_m`
- `seed`
- `sampling_rule`
- `sampling_cell_id`
- `source_decision`
- `requires_posthoc_label_validation=True`
- `expected_postrender_labels`

The manifest should not pre-fill final success/failure labels. The label fields
are written only after the render validation script runs.

## Proposed 1000-Episode Sampling Draft

This is a draft distribution for one 0-999 render chunk. Counts are intended
sampling counts, not final labels.

| Intended bucket | Count | Purpose |
| --- | ---: | --- |
| `clean_prior_candidate` | 650 | Bias toward y low-negative and center regions that produced most clean taps in d241 |
| `transition_mixed_probe` | 200 | Probe positive-mid y values where clean and overshoot are both common |
| `overshoot_eval_candidate` | 100 | Preserve high-boundary and known overshoot cases as evaluation/negative examples |
| `debug_camera_anchor` | 50 | Repeat fixed anchors/corners/edge poses for camera regression and frame-level inspection |

Expected use after render:

- Train-positive pool: only episodes with
  `label_camera_contract_numeric=1` and `label_useful_clean_numeric=1`.
- Overshoot/eval pool: episodes with `label_overshoot_numeric=1`, especially
  camera-passing boundary samples.
- Rejected/quarantine pool: any episode with failed frame count, failed camera
  contract, missing contact/reaction, or reprojection gate failure.

## Gates Before Running 0-999

Required before any 0-999 render:

- Explicit approval for a new 0-999 render.
- Disk/output-root preflight. Do not delete or move existing data by default.
- Decide whether the goal is 1000 rendered episodes or 1000 clean train-positive
  episodes. These are different goals.
- Generate the label-aware manifest only; validate row count, seed uniqueness,
  and all `requires_posthoc_label_validation=True`.
- Keep PNG as debug/extraction only. Raw PNG at scale remains a storage risk.
- Preserve LeRobot AV1 + companion metadata as the primary format unless a
  target training environment fails AV1 decode.

## Still Blocked Without Explicit Approval

- Any 0-999 / 1000 / 10000 Isaac render.
- Any dataset scale-up beyond the already rendered d241 0-99 chunk.
- Any deletion, move, archive, or cleanup.
- PPO, L2, Large PPO, SmolVLA/VLA fine-tuning, action-teacher, RoArm deployment.
- SSH JHPark/B200 reconnect, pull, or `.ssh` copy.
- Track A work.
