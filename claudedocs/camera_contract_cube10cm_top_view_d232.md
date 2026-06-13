# D232 Camera Contract - Cube10cm Top-View Visual Trajectory Dataset

Status: draft for professor confirmation plus local 5-episode smoke result.
No full dataset generation, deletion, PPO, VLA/action-teacher, RoArm deployment,
SSH/B200, or Track A work is implied by this document.

## Professor Confirmation Framing

One-line confirmation:

> We will keep the frame-by-frame image-state pair structure, but store it in the
> industry-standard LeRobot format, MP4 plus parquet. Any arbitrary frame can be
> extracted immediately as PNG when needed.

Korean phrasing for the professor:

> 프레임별 이미지-상태 페어 구조는 그대로 유지하되, 저장은 업계 표준인
> LeRobot 포맷(mp4+parquet)으로 하겠습니다. 임의 frame을 PNG로 즉시 추출
> 가능합니다.

If there is a concrete reason to require all frames as PNGs, such as frame-level
manual review workflow or an external tool that cannot read MP4/parquet, adjust
after hearing that requirement. Until then, use MP4/parquet as the primary
storage and PNG only for smoke/debug/extraction.

## Current Source Truth

- `sim_scripts/kinect_calib.yaml` intrinsics are camera-intrinsic candidates:
  `fx=608.33`, `fy=608.28`, `cx=638.31`, `cy=365.26`, `width=1280`,
  `height=720`.
- The same file's old extrinsics are hand-eye extrinsics and are invalid for
  the new top-view dataset camera pose.
- Initial top-view candidate details live in
  `claudedocs/camera_top_view_extrinsics_candidate_d232.md`.
- Raw visual storage target is Azure-Kinect-compatible `1280x720`; `224x224`
  is only model preprocessing.
- Full 1000/10000 episode generation is blocked until this contract and the
  smoke gates pass.

## Camera Contract Fields

### Physical Mount

- Camera model: Azure Kinect DK.
- Mount method: fixed clamp/tripod/rig, to be physically reproducible.
- Mount reference frame: table/world frame, not a visually pleasing simulator
  viewpoint.
- Table-to-camera height: numeric value required before smoke render.
- Height tolerance: numeric tolerance required, recommended to record measured
  value and allowed remount error.
- Workspace anchoring landmarks: table corners, robot base center, cube nominal
  spawn center, and at least one visual marker/corner set for reprojection.

### Intrinsics

- Raw RGB resolution: `1280x720`.
- Candidate intrinsics: `fx=608.33`, `fy=608.28`, `cx=638.31`, `cy=365.26`.
- FPS target: `30`.
- Depth policy: RGB is required; depth mode may be recorded for real setup
  compatibility, but the first visual dataset contract is RGB-first unless
  explicitly expanded.
- USD camera conversion: reuse the existing intrinsics-to-FOV/focal conversion
  pattern, but do not reuse old hand-eye extrinsics.

### Top-View Extrinsics

- Camera position in world/table frame: numeric XYZ required.
- Camera orientation: roll/pitch/yaw or quaternion required.
- Inverted mount flag: required.
- OpenCV/USD convention: record whether image axes need Y/Z flip, rotation, or
  post-render flip.
- Crop/rotation/flip convention: must be deterministic and written into
  metadata. Do not leave this implicit.

### Workspace Coverage

- Visible table bounds in meters.
- Cube pose sampling bounds for each split.
- Robot/tool visible area and allowed image margin.
- Cube must remain inside the raw frame for all smoke frames except frames
  explicitly labeled as occluded by robot/tool.
- Top-view must expose the tap/push direction and cube displacement, not only
  robot motion.

### Occlusion Metrics

For every rendered frame, classify cube visibility:

- `cube_visible_full`: all projected cube top corners / required mask visible.
- `cube_visible_partial`: cube still localizable but partially covered.
- `cube_occluded_full`: cube cannot be localized from the image.

Smoke pass requires reporting full/partial/full-occlusion rates and a separate
contact-window occlusion rate. Fully occluded contact-critical frames are a fail
unless the professor explicitly accepts them as part of the visual task.

### Dataset Layout

Primary storage:

- LeRobot-style video+parquet.
- Video key: `observation.images.top`.
- Shape: `(720, 1280, 3)` raw RGB.
- Per-frame parquet metadata: state, action/teacher command if present, object
  pose, camera contract id, split, episode id, frame id, timestamp, seed.

Debug/extraction:

- Smoke may preserve source PNGs for visual inspection and codec comparison.
- Scale-up must not store all frames as PNG by default.
- `extract_frames.py` provides direct PNG extraction by `episode_id` and
  `frame_id` from LeRobot MP4/parquet.

### Split Contract

Keep split intent explicit:

- `debug_smoke`: 5-10 episodes only, camera contract validation.
- `train_success`: clean scripted/base tap-push trajectories.
- `eval_boundary`: D225-D228 boundary poses and transition regions.
- `eval_failure`: D230 useful-tap overshoot/failure regions.

Each split must record sampling bounds, seeds, cube mass/size, action source,
camera contract id, and whether the sample is for training or evaluation only.

## Smoke Gate

Run only after explicit approval.

Minimum smoke: 5-10 episodes, `1280x720`, target `30fps`, LeRobot
`observation.images.top` video+parquet output, plus smoke/debug PNGs only where
needed.

Pass/fail report must include:

1. Reprojection sanity: known cube corners/markers projected with selected
   intrinsics/extrinsics and compared to rendered pixels. Report median/max
   pixel error.
2. Occlusion: full/partial/full cube visibility rates, plus contact-window
   occlusion.
3. Render speed: seconds per episode and effective rendered fps.
4. Storage cost: MB per episode and projected GB for 100/1000/10000 episodes.
5. Codec quality: source PNG vs decoded MP4 sampled-frame pixel difference.
   Report max/mean absolute difference and frame count match. Use conservative
   CRF or near-lossless settings if quality is questionable.
6. LeRobot load and dataloader decode: dataset opens with installed LeRobot and
   item access returns `observation.images.top`, `observation.state`, and
   `action` with expected shapes. This must use the training dataloader/backend,
   not only `extract_frames.py` or OpenCV/ffmpeg helper extraction.
   Existing v6 read-only preflight passed locally for AV1 only when writable
   HuggingFace cache paths were set under `/tmp`; repeat on smoke output and
   RunPod before scale-up.
7. Extraction check: `extract_frames.py --episode-id E --frame-id F` writes a
   valid PNG matching the decoded MP4 frame.

Scale-up fails automatically if local disk remains near the D232 audited state
of about 39GB free, unless external/RunPod storage is provisioned.

## Blocked Until Explicit Approval

- Any additional Isaac Sim/render execution.
- Any additional 5-10 episode smoke render.
- Any 100/1000/10000 episode generation.
- Any deletion, archive, or move of existing files.
- PPO, L2/Large PPO, VLA/action-teacher, RoArm deployment.
- SSH/B200 reconnect, pull, or `.ssh` copy.

## Local Smoke Result - 2026-06-12 D233

Approved local smoke produced 5 episodes and 975 frames at `1280x720` using
camera contract `cube10cm_top_view_v1_candidate`.

Source logs:

- render:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/render_summary.json`;
- LeRobot validation:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/lerobot_validation_summary.json`;
- session record:
  `claudedocs/session_20260612_cube10cm_top_view_visual_smoke_lerobot_d233.md`.

Pass facts:

- reprojection centroid error median/max: `3.074639061891291px` /
  `9.956731449704932px`;
- all-frame cube visibility: `975/975` full, `0` partial, `0` full occlusion;
- contact-window visibility: `882/882` full, `0` partial, `0` full occlusion;
- render throughput: `180.79416966438293s` for 975 captured frames,
  `5.392873021347648` captured frames/sec;
- LeRobot conversion/load/decode status: `PASS`;
- video key: `observation.images.top`;
- codec: AV1 via `libsvtav1`, `yuv420p`, `30fps`;
- sampled decoded image/state/action shapes: `[720,1280,3]`, `[6]`, `[6]`;
- source PNG vs decoded MP4 sampled max mean absolute pixel difference:
  `0.8939572482638889`;
- arbitrary PNG extraction from the LeRobot dataset succeeded for episode `3`,
  frame `50`, producing a `1280x720` PNG.

Decision update:

- Camera contract v1 is local-smoke-pass, not final scale-up approval.
- Existing v6 datasets were codec/backend fixtures only; they do not define the
  professor visual trajectory schema.
- PNG remains limited to smoke/debug/extraction. Primary storage remains LeRobot
  MP4+parquet.
- AV1 remains acceptable locally, but must be repeated on RunPod/H100 through
  the training LeRobot dataloader before scale-up there.
- 100 episode chunk still requires professor confirmation, storage decision, and
  explicit approval.

## Professor Packet / RunPod Gate - 2026-06-13 D234

Professor-facing packet prepared:

- `claudedocs/professor_view_format_packet_cube10cm_top_view_d234.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/professor_review_contact_sheet_d234.png`

The packet keeps the D232/D233 framing: frame-by-frame image-state pairs remain,
LeRobot MP4+parquet is primary, and arbitrary PNG extraction is available on
demand. Direct professor response is not yet recorded in repo docs.

RunPod/H100 LeRobot AV1 dataloader gate passed:

- pod id `86qyuxeldab9h4`, H100 80GB;
- full 975-frame decode status `PASS`;
- codec `av1`, pix_fmt `yuv420p`, fps `30`;
- decoded image/state/action shapes `[3,720,1280]`, `[6]`, `[6]`;
- avg/max decode `0.017871856689453125s` /
  `0.10865616798400879s`.

Decision update:

- AV1 is selected for the next 100 episode chunk by current local + RunPod
  evidence.
- 100 episode chunk was not run in D234.
- 1000/10000 generation remains blocked.
