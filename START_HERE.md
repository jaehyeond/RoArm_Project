# START_HERE.md

Last updated: 2026-06-18 KST (D257 current truth: professor branch now has a LeRobot pair, RL transition/reward table, and PPO-compatible state-action teacher checkpoint; no Isaac Lab PPO runtime, render, cleanup, RunPod, B200, or RoArm control has been run.)

## Current Truth

- Active branch: professor 10cm / 0.72kg cube top-view visual trajectory
  dataset camera-contract branch. The earlier tap RL branch is frozen unless
  explicitly resumed.
- Do not mix with Track A grasp/dataset/training work.
- Research objective is useful tap: physical contact/reaction without overshoot. The runtime's legacy `tap_success` still encodes the 6mm target-band until explicitly changed; use D229/D230 useful-tap log fields for the new objective.
- Keep `policy_target_disp_m=0.006` and `tap_target_disp_tolerance_m=0.003` as quality-tier diagnostics, not as the primary "any useful tap" claim.
- Current clean residual action branch is 3D task-space target residual (`candidate8_diffik_target_residual`, `policy_action_space=3`), no gates.
- D224 geometry remains current: hand TCP is already offset from link5; distal collision surface is about 4.46mm beyond hand_tcp. Do not add another TCP offset.
- D231 tool-contact proxy truth:
  - The 10cm cube has a real physics collider; the current dispute is the robot tool-side reward/metric contact proxy.
  - Direct `gripper_link_collision_g2a` is rejected as a replacement: it is a tiny ~4mm collision proxy, not the full fingertip/tool surface.
  - `gripper_link` visual mesh is fuller geometry, but USD config has `collision_from_visuals=false`, so visual geometry is not current physics collision.
  - Current `link5_collision_aabb` remains the runtime proxy and should be understood as fixed-jaw/distal-tool proxy, not bare wrist/TCP.
  - If the proxy changes, use a named `tool_surface_union` metric: fixed jaw/link5 distal surface plus moving gripper full geometry or properly authored collision surface.
- D232 professor visual-dataset direction:
  - Current practical next work is not PPO promotion. It is a camera-calibrated top-view visual trajectory dataset smoke path for 10cm cube tap/push.
  - Raw visual data should be Azure-Kinect-compatible `1280x720`; `224x224` is only a model preprocessing size.
  - `sim_scripts/kinect_calib.yaml` intrinsics may be reused as camera-intrinsic candidates, but its old hand-eye extrinsics are not the new top-view camera pose.
  - The camera contract must define physical mount, height, pose, flip/crop convention, workspace coverage, fps, and self-occlusion metrics before rendering scale-up.
  - Dataset format should be LeRobot-style video+parquet (`observation.images.top`), with PNGs only for smoke/debug.
  - Full 1000/10000 episode generation is blocked until a 5-10 episode smoke proves reprojection, occlusion, codec, fps, render-time, LeRobot load, and disk-cost gates.
  - Current disk is too tight for full rendering: `RoArm_Project` is about 269G and filesystem free space is only about 39G. Clean/archive before scale-up.
- D233 local smoke result:
  - Camera contract `cube10cm_top_view_v1_candidate` completed a 5-episode local
    smoke at `1280x720`: 975 frames, reprojection median/max
    `3.074639061891291px` / `9.956731449704932px`, all-frame visibility
    `975/975` full, contact-window visibility `882/882` full.
  - LeRobot conversion/load/decode passed locally with key
    `observation.images.top`, codec `av1`, `yuv420p`, `30fps`, 975 frame count
    match, sampled decode avg/max `0.016793251037597656s` /
    `0.06672263145446777s`, and sampled PNG-vs-decoded mean abs max
    `0.8939572482638889`.
  - `extract_frames.py` proved arbitrary PNG extraction from MP4+parquet with a
    `1280x720` episode 3 frame 50 PNG.
  - PNG-at-scale is rejected by measurement: debug PNG is
    `52.3344778MB/episode`; LeRobot AV1 video is `0.5964878MB/episode`.
  - Existing v6 data is only a codec/backend fixture, not the professor schema.
  - Local disk after smoke is about `32G` free, worse than D232.
- D234 gate result:
  - Professor-facing view/format packet is prepared at
    `claudedocs/professor_view_format_packet_cube10cm_top_view_d234.md`.
    It includes the agreed wording: LeRobot MP4+parquet primary storage,
    arbitrary PNG extraction on demand. Direct professor response is not yet
    recorded in repo docs.
  - Contact-sheet artifact:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/professor_review_contact_sheet_d234.png`
    (`1326x1442` RGB PNG).
  - Storage/output-root decision for the next 100 episode chunk, if launched
    later: fresh local root
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d235`,
    allowed only as one 100ep chunk with pre-run free space at least about
    `25GB`; 1000/10000 still require external/RunPod storage, no-full-PNG
    retention, or explicit cleanup/archive approval.
  - RunPod/H100 AV1 LeRobot dataloader gate passed on pod
    `86qyuxeldab9h4` (`NVIDIA H100 80GB HBM3`). Full 975-frame decode:
    status `PASS`, codec `av1`, `yuv420p`, `30fps`, image/state/action shapes
    `[3,720,1280]` / `[6]` / `[6]`, avg/max decode
    `0.017871856689453125s` / `0.10865616798400879s`.
  - RunPod pod was stopped after results were copied back. It was not deleted.
    Cost rule: always stop RunPod pods immediately after work; if all outputs
    are copied back and the pod environment is no longer needed, delete/terminate
    the pod too because stopped pods can still accrue volume-storage charges.
- D235 schema/metadata result:
  - Do not stuff all rich metadata into the LeRobot core parquet by default.
    Keep standard LeRobot core fields for loader compatibility and write
    companion metadata keyed by `global_index` plus (`episode_index`,
    `frame_index`).
  - Added `claudedocs/cube10cm_top_view_visual_dataset_schema_d235.md` and
    `sim_scripts/cube10cm_top_view_metadata_companion.py`.
  - Ran the companion builder on the existing D233 5ep smoke only. It wrote
    `metadata_companion_d235/per_frame_metadata.parquet`,
    `episode_metadata.parquet`, `metadata_schema.json`, and
    `metadata_validation_summary.json`.
  - Validation PASS: 975 companion rows / 5 episodes aligned with LeRobot core
    `index`, `episode_index`, and `frame_index`. This was non-render,
    non-training, and did not generate a new dataset chunk.
- D236 sampling-contract result:
  - Do not make the 0-99 chunk by repeating the D233 five smoke poses.
  - Added `claudedocs/cube10cm_top_view_chunk100_sampling_contract_d236.md`.
  - The next renderer should consume an explicit manifest with `split_candidate`,
    `cube_x_m`, `cube_y_m`, `seed`, `sampling_rule`, `sampling_cell_id`,
    `source_decision`, and `requires_posthoc_label_validation`.
  - Proposed draft split: 5 `debug_smoke`, 65 `train_success_candidate`, 15
    `eval_failure_candidate`, 15 `eval_boundary_candidate`. These are intended
    buckets only; final labels must come from post-render contact/reaction/
    overshoot/visibility/reprojection checks.
- D237 manifest result:
  - Added `sim_scripts/cube10cm_top_view_chunk100_manifest.py`.
  - Generated deterministic manifest at
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_chunk100_manifest_d236/episode_manifest.csv`
    plus JSON and summary.
  - Validation PASS: 100 rows, episode ids `0..99`, split counts
    `debug_smoke=5`, `train_success_candidate=65`,
    `eval_failure_candidate=15`, `eval_boundary_candidate=15`, all rows marked
    `requires_posthoc_label_validation=True`.
  - This is still non-render/no-dataset-generation. Boundary rows use `y=0.15`
    candidates and require camera coverage validation before claims.
- D238 renderer-prep result:
  - Added `sim_scripts/cube10cm_top_view_visual_chunk_render.py`.
  - The D233 smoke renderer remains capped at 1-10 episodes.
  - The chunk renderer requires a manifest, copies it into the render root, and
    attaches manifest fields to per-frame metadata.
  - `--validate-only` PASSed without starting IsaacLab: 100 episodes,
    split counts `5/65/15/15`.
  - Actual 0-99 render remains blocked until explicit launch approval and disk/
    output-root preflight.
- D239 chunk100 render + label result:
  - After explicit launch approval, ran the local manifest-fed 0-99 render. The
    successful root is
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d241`.
  - Failed/partial roots were preserved, not deleted:
    `cube10cm_top_view_visual_chunk100_d235` partial, `d239` sandbox GPU/Vulkan
    failure, and `d240` 345-frame warning-spam partial.
  - Successful d241 render: 100 episodes / 19,500 frames at `1280x720`, split
    counts `debug_smoke=5`, `train_success_candidate=65`,
    `eval_failure_candidate=15`, `eval_boundary_candidate=15`.
  - Render metrics: raw PNG total `5142551626` bytes,
    `51.42551626MB/episode`, elapsed `4647.953013896942s`, effective captured
    FPS `4.195395250704307`.
  - Camera metrics: all-frame visibility `19500/19500` full,
    contact-window visibility `18372/18372` full, reprojection centroid
    median/max `3.0758927127400306px` / `17.06565232897021px`, contract
    violations `[]`.
  - Added `sim_scripts/cube10cm_top_view_postrender_label_validation.py` and
    generated d241 labels at
    `postrender_label_validation_d241/episode_labels.csv` plus JSON summary.
  - Label result: camera contract pass `100/100`, contact seen `100/100`,
    reaction seen `100/100`, missing contact/reaction `0`, overshoot seen
    `39/100`, useful clean `61/100`, legacy target-band success `62/100`.
  - Status split: `clean_useful_tap=61`,
    `contact_reaction_with_overshoot=39`. By split: debug `3/2`, train
    candidate `49/16`, eval-failure candidate `8/7`, boundary candidate `1/14`
    clean/overshoot.
  - Critical interpretation: `split_candidate` is a sampling bucket, not a final
    label. Use `label_useful_clean_numeric` / `label_overshoot_numeric` from
    post-render validation for dataset filtering.
  - `roarm_rl/roarm_cube_push_env.py` now uses `quat_apply` instead of deprecated
    `quat_rotate` at the two chunk-render hot paths to prevent unusable warning
    logs. This is an IsaacLab API replacement for the same quaternion-vector
    application semantics.
  - D240 completed d241 LeRobot AV1 conversion and validation: codec `av1`,
    `yuv420p`, `30fps`, 19,500 frame count match, video `56604396` bytes,
    `0.56604396MB/episode`, sampled decode avg/max `0.008618485927581788s` /
    `0.09812450408935547s`, sampled PNG-vs-decoded mean abs max
    `0.8940353732638889`, sampled max pixel abs diff `74`, final LeRobot root
    about `56MB`, temporary dataset PNG count `0`.
  - D240 generated d241 companion metadata under `metadata_companion_d241`:
    19,500 per-frame rows / 100 episodes, row-aligned to LeRobot core by
    `index`, `episode_index`, and `frame_index`.
  - D240 proved PNG extraction from d241 AV1:
    `debug_extract_frames_d241/episode_000099_frame_000050.png`, `1280x720`,
    source-vs-extracted mean abs diff `0.7776012731481482`, max abs diff `30`.
  - Added label-aware 0-999 design:
    `claudedocs/cube10cm_top_view_label_aware_0_999_manifest_design_d240.md`.
    Proposed sampling buckets are intended buckets only; all rows still require
    post-render numeric label validation.
  - D241 added `sim_scripts/cube10cm_top_view_labelaware_manifest_0_999.py` and
    generated a manifest-only 0-999 plan at
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_labelaware_manifest_0_999_d241`.
  - D241 manifest validation PASS: 1000 rows, episode ids `0..999`, seed base
    `2410`, seeds unique, all rows require post-render label validation,
    forbidden final label fields absent, x range `0.09..0.39`, y range
    `-0.1..0.15`.
  - D241 intended bucket counts: `debug_camera_anchor=50`,
    `clean_prior_candidate=650`, `transition_mixed_probe=200`,
    `overshoot_eval_candidate=100`.
  - Critical D241 blocker: this is not a dataset; it is only a render plan.
    Current `sim_scripts/cube10cm_top_view_visual_chunk_render.py` is scoped to
    exactly 100 episodes, so actual 0-999 render requires a separately approved
    renderer update or new renderer.
  - D242 added `sim_scripts/cube10cm_top_view_visual_manifest_render.py`, leaving
    the old 100ep renderer unchanged for reproducibility.
  - D242 validate-only PASS:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_manifest_render_validate_d242.json`.
    It validated 1000 rows / expected episodes 1000, episode range `0..999`,
    bucket counts `50/650/200/100`, all post-render label validation required,
    all camera coverage required, seed unique, robot USD exists, output render
    root absent/empty, `render_approved=false`, and runtime
    `VALIDATE_ONLY_NO_RENDER_NO_DATASET_GENERATION_NO_TRAINING`.
  - Actual 0-999 render remains blocked. The new renderer will not render unless
    `--validate-only` is omitted and `--render-approved` is supplied after
    explicit runtime approval and disk/output-root preflight.
- D243 runtime preflight result:
  - User approved actual runtime direction, but requested expected time/capacity
    verification.
  - No IsaacLab render was started because local disk preflight failed.
  - `df -B1 .` showed total `632825225216`, used `572916858880`,
    available `27687411712`, use `96%`.
  - `/tmp` is the same filesystem and is not a larger output root.
  - D241 measured 100ep render scales to 1000 episodes as: `195000` frames,
    expected render time `46479.530s` (`12.911h`, practically about `13-15h`),
    raw PNG `51425516260` bytes (`51.426GB` decimal), AV1 video
    `566043960` bytes, and minimal raw+video+JSONL+metadata about `52.486GB`.
  - Current renderer writes all captured frames first under
    `raw_env_render_frames/*.png`, so final AV1 compactness does not remove the
    local pre-conversion disk requirement.
  - Decision: `LOCAL_0_999_RUNTIME_NOT_STARTED_DISK_HARD_BLOCK_D243`.
  - The actual render root `cube10cm_top_view_visual_0_999_d242` remains absent.
- D244 cleanup result:
  - User explicitly approved deleting only
    `claudedocs/figures/p6v12_rollout/frames`.
  - Pre-delete manifest saved at
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/p6v12_rollout_frames_cleanup_d243/frames_manifest_predelete.tsv`.
  - Manifest line count `73366`, sha256
    `45a0ece58f4e86cd605b262e4e43bce17b11c1326cb3285dfcccded6ea922e26`.
  - Deleted exactly `claudedocs/figures/p6v12_rollout/frames`.
  - Preserved P6v12 compact evidence: `p6v12_rollout.mp4`,
    `p6v12_trajectory.csv`, `replay`, `replay_old_camera`, and
    `replay_silver_backup`.
  - Post-delete `df -h .`: `590G` total, `501G` used, `60G` available, `90%`
    used. Net available-space increase from D243 baseline:
    `35689971712` bytes, about `35.69GB` decimal.
  - Decision: `P6V12_RAW_FRAMES_CLEANUP_COMPLETE_D244`.
  - This improves storage but does not automatically launch the 0-999 render;
    current free space is close to the D243 minimum requirement and has limited
    safety margin.
- D245 outputs training-state cleanup result:
  - User asked for a critical recheck and storage recovery using the D232 second
    cleanup path.
  - Deleted only `outputs/*/checkpoints/*/training_state` after manifest.
  - Pre-delete manifest directory:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/outputs_training_state_cleanup_d245`.
  - Deleted `58` training-state directories / `290` files; preserved `66`
    `pretrained_model` directories.
  - `training_state` represented SmolVLA exact-resume state
    (optimizer/scheduler/RNG/step), not model weights. `pretrained_model`
    inference/eval/new fine-tuning artifacts remain.
  - Post-delete `df -h .`: `590G` total, `479G` used, `82G` available, `86%`
    used.
  - Net available-space increase from D245 pre-delete:
    `23936794624` bytes, about `23.94GB` decimal. Net increase since D243:
    about `59.61GB` decimal.
  - Decision: `OUTPUTS_TRAINING_STATE_CLEANUP_COMPLETE_D245`.
  - Actual 0-999 render still requires a fresh runtime decision and final
    preflight; this cleanup only improves the storage gate.
- D246 local 0-999 render + post-render label result:
  - After explicit local runtime approval, ran
    `sim_scripts/cube10cm_top_view_visual_manifest_render.py --render-approved --device cuda:0`
    against the D241 0-999 manifest. This was Isaac render/data capture only:
    no PPO, no VLA/SmolVLA fine-tuning, no action-teacher, no RoArm deployment,
    no RunPod, no B200/SSH/pull, and no deletion/move/archive.
  - Render root:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242`.
  - Render summary: 1000 episodes / 195000 frames, `1280x720`, target `30fps`,
    elapsed `28349.806646108627s` (about `7.88h`), effective captured FPS
    `6.878353790351732`, raw PNG total `51386208295` bytes, about
    `51.386208294999996MB/episode`, contract violations `[]`.
  - Manifest buckets were preserved as intended buckets:
    `debug_camera_anchor=50`, `clean_prior_candidate=650`,
    `transition_mixed_probe=200`, `overshoot_eval_candidate=100`.
  - Post-render labels were generated at
    `postrender_label_validation_d246`: expected/actual episodes `1000/1000`,
    frames `195000`, all episodes have `195` frames.
  - Camera gate passed for `986/1000` episodes. Label status counts after camera
    gate: `clean_useful_tap=819`, `contact_reaction_with_overshoot=167`,
    `camera_quality_fail=14`, `missing_contact_or_reaction=0`.
  - Raw event labels before camera filtering: useful-clean numeric `829`,
    overshoot numeric `171`. The difference is the 14 camera-quality-fail
    episodes: 10 clean-event and 4 overshoot-event episodes are excluded by
    camera gate.
  - Critical failure interpretation: 13 camera failures are clean-prior bucket
    reprojection-gate failures just above `20px`; episode `721` is a stronger
    coverage/projection failure (`projection_inside_frames=7/195`) and should be
    treated as a camera-contract design warning before scale-up.
  - Post-render disk state: output root is about `49G`; `df -h .` is `590G`
    total, `528G` used, `33G` available, `95%` used.
  - The renderer printed the done line and wrote `render_summary.json`; local
    Kit close was then cleaned up manually because the script intentionally skips
    `sim_app.close()` to avoid local close hangs. This makes the log show a
    post-completion killed process, not a missing render artifact.
  - Decision: `LOCAL_0_999_RENDER_D242_COMPLETE_POSTRENDER_LABELS_D246`.
  - Next gated step is not training. It is LeRobot v3 conversion/load validation,
    companion metadata generation, video codec/decode checks, PNG extraction
    proof, source-vs-decoded pixel diff, and row alignment. Because only about
    `33G` remains free, conversion or further scale-up requires a fresh storage
    check and explicit approval.
- D247 LeRobot v3 conversion + metadata result:
  - User approved proceeding without cleanup. Raw PNGs were preserved; no deletion,
    move, archive, render, PPO, VLA/SmolVLA fine-tuning, action-teacher, RoArm,
    RunPod, B200/SSH/pull, or Track A was run.
  - Converted existing D246 raw render to
    `cube10cm_top_view_visual_0_999_d242/lerobot_dataset_av1_d247`.
  - Final LeRobot dataset size is about `540M`; companion metadata is about
    `34M`; extracted debug PNG folder is about `104K`.
  - Initial validation in `lerobot` env failed after all `1000` episodes were
    saved because default torchcodec loading failed against local
    `torch==2.10.0+cu128` / missing FFmpeg shared libraries. This is an
    environment/backend failure, not a missing dataset build.
  - Patched `sim_scripts/cube10cm_top_view_smoke_to_lerobot.py` to support
    `--validate-only` and `--video-backend`; revalidated the existing dataset
    with `video_backend=pyav`.
  - LeRobot validation PASS:
    `total_frames=195000`, `total_episodes=1000`, codec `av1`, pix fmt
    `yuv420p`, fps `30`, video bytes `548571183`,
    `0.548571183MB/episode`, sampled decode avg/max
    `0.015330815315246582s` / `0.017406463623046875s`, sampled
    PNG-vs-decoded mean abs max `0.898435691550926`, max abs diff `80`.
  - PyAV video frame-count check PASS: three MP4 files contain
    `67275 + 87945 + 39780 = 195000` frames at `1280x720@30fps`.
  - Companion metadata PASS: `195000` per-frame rows / `1000` episodes aligned
    to LeRobot core parquet by `index`, `episode_index`, and `frame_index`.
  - PNG extraction proof PASS:
    episode `999`, frame `194`, extracted `1280x720` PNG; source-vs-extracted
    mean/max abs diff `0.792978515625` / `31`.
  - Post-conversion disk: `df -h .` remains about `590G` total, `528G` used,
    `32G` available, `95%` used. Raw PNG storage remains the dominant disk cost.
  - Decision: `D247_0_999_LEROBOT_AV1_PYAV_VALIDATION_METADATA_PASS`.
  - Next work is analysis/packaging/filtering of the camera-gated labels or
    storage policy for raw PNG retention. Any training remains separate.
- D248 label package + camera-fail audit result:
  - No render, training, deletion, move, archive, PPO, VLA/SmolVLA fine-tuning,
    action-teacher, RoArm, RunPod, B200/SSH/pull, or Track A was run.
  - Added `sim_scripts/cube10cm_top_view_package_label_splits.py`.
  - Output package:
    `cube10cm_top_view_visual_0_999_d242/label_package_d248`.
  - Packaged all `1000` episodes:
    `train_clean_positive=737`, `eval_clean_holdout=82`,
    `eval_overshoot_diagnostic=167`, `quarantine_camera_fail=14`.
  - Basis: only camera-pass `clean_useful_tap` episodes enter positive BC train;
    10% of clean useful taps are deterministic held-out eval; camera-pass
    overshoot episodes are diagnostic eval, not positive train; camera failures
    are quarantined.
  - Camera-fail audit: 13/14 failures are reprojection-only gate failures
    (`projection_inside_frames=195/195`, full visibility `195/195`, max
    reprojection error just over the `20px` gate); episode `721` is the stronger
    camera coverage failure with `projection_inside_frames=7/195`.
  - Visual audit exists at
    `label_package_d248/camera_fail_contact_sheet.png` (`1780x2688` RGB PNG).
  - Decision: `D248_LABEL_PACKAGE_TRAIN_EVAL_QUARANTINE_CAMERA_FAIL_AUDIT_PASS`.
  - Next valid work is to decide whether to train only on
    `train_clean_positive` locally/RunPod later, or first inspect/fix camera
    contract around the 14 quarantined episodes. Training remains separate.
- D249-D252 dataset freeze + filtered-loader preflight result:
  - No render, training, deletion, move, archive, PPO, VLA/SmolVLA fine-tuning,
    action-teacher, RoArm, RunPod, B200/SSH/pull, or Track A was run.
  - D249 dataset freeze created
    `dataset_freeze_d249` with freeze id
    `cube10cm_top_view_0_999_v0_1_d249`, dataset card, and SHA256 manifest for
    `24` primary files totaling `1089314018` bytes. Raw PNGs remain preserved
    but are not individually SHA256 hashed.
  - D250 filtered views created
    `filtered_views_d250`: 학습용 정상 성공 예시
    `train_clean_positive=143715` frames / `737` episodes, 평가용 정상 보류
    예시 `eval_clean_holdout=15990` frames / `82` episodes, 과하게 민 케이스
    진단용 평가 데이터 `eval_overshoot_diagnostic=32565` frames / `167`
    episodes, 카메라 기준 실패 격리 데이터 `quarantine_camera_fail=2730`
    frames / `14` episodes.
  - D251 filtered dataloader smoke passed locally with `video_backend=pyav`:
    all 4 splits decode through LeRobot with image shape `[3,720,1280]` and
    state/action shape `[6]`; no training was run.
  - D252 split distribution check passed: 학습용 정상 성공 예시는 sampled
    workspace x/y range를 포함하고, 과하게 민 케이스는 high-y/boundary-y에
    몰리며, 카메라 실패는 x 약 `0.14-0.165m` 근처와 episode `721` coverage
    failure가 핵심 경고다.
  - Decision: `D249_D252_DATASET_FREEZE_FILTERED_VIEW_DATALOADER_DISTRIBUTION_PASS`.
  - Next valid work is a training preflight plan/dry-run only. Actual
    SmolVLA/VLA fine-tuning remains blocked until explicit approval.
- D253 training preflight result:
  - No training, render, deletion, move, archive, PPO, action-teacher, RoArm,
    RunPod, B200/SSH/pull, or Track A was run.
  - Added `sim_scripts/cube10cm_top_view_training_preflight.py`.
  - Output:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/training_preflight_d253`.
  - Status `PASS`: official LeRobot dataset factory consumed
    `train_clean_positive` through `dataset.episodes` with `video_backend=pyav`.
  - Selected training data is 학습용 정상 성공 예시
    `train_clean_positive=737` episodes / `143715` frames.
  - First DataLoader batch shape was image `[4,3,720,1280]`, state `[4,6]`,
    action `[4,6]`.
  - Critical limitation: LeRobot train has one dataset input;
    `eval_clean_holdout` and `eval_overshoot_diagnostic` are not automatically
    used by `lerobot-train`. They remain offline evaluation inputs for a later
    script.
  - Proposed commands were written but not executed:
    50-step smoke and 20k candidate. Any actual SmolVLA/VLA fine-tuning remains
    blocked until explicit approval.
- D254 method-pipeline reframe:
  - Added `claudedocs/cube10cm_top_view_method_pipeline_d254.md`.
  - This corrects the framing: the professor-facing result is not "show some
    validation images" and not "we trained SmolVLA." It is a repeatable method
    pipeline:
    1. camera contract;
    2. Isaac Lab 0-999 visual trajectory generation;
    3. post-render contact/reaction/overshoot/camera label validation;
    4. LeRobot MP4+parquet storage;
    5. train/eval/quarantine curation;
    6. official LeRobot training-input preflight.
  - SmolVLA 50-step smoke is now classified as an optional next gate to verify
    training-loop connectivity, not as the core next research claim.
  - Offline held-out evaluation still needs a separate script after an approved
    checkpoint exists.
  - No training, render, deletion, move, archive, RunPod, PPO, action-teacher,
    RoArm, B200/SSH/pull, or Track A work was run for D254.

## Previous Result: D247

- Converted the local D246 0-999 raw render into LeRobot v3 AV1+parquet and
  generated companion metadata.
- Decision:
  - `D247_0_999_LEROBOT_AV1_PYAV_VALIDATION_METADATA_PASS`;
  - LeRobot pyav load/decode validation passed for `195000` frames / `1000`
    episodes;
  - companion metadata row alignment passed;
  - arbitrary PNG extraction passed on episode `999`, frame `194`;
  - local default torchcodec path is currently broken in the `lerobot` env, so use
    `video_backend=pyav` locally unless torchcodec/FFmpeg is repaired;
  - local available space is still only about `32G`.

## Latest Result: D257

- Added and ran `sim_scripts/cube10cm_top_view_train_state_action_teacher.py`.
- Trained a small PPO-compatible state-action teacher from D256
  `ppo_actor_prior_teacher_rows_d256.csv`.
- Scope was CPU supervised teacher fitting only:
  no Isaac Lab PPO runtime, no render, no RunPod, no cleanup, no RoArm control.
- Output root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257`.
- Checkpoint:
  `cube10cm_d257_state_action_teacher_clipped0040.pt`
  (`155965` bytes, sha256
  `f81df20278ec9ceddef141729f717abbba2412a4a2f9f3a366d88b387caa76b8`).
- Metrics:
  - rows total/train/validation: `142978` / `128622` / `14356`;
  - feature count `27`, target count `5`;
  - baseline validation MSE norm `1.047513484954834`;
  - final validation MSE norm `0.10824249684810638`;
  - validation RMSE by joint rad:
    `[0.003628572914749384, 0.004176619462668896, 0.003567545209079981, 0.0034607199486345053, 0.0011242240434512496]`;
  - checkpoint reload/key/shape check passed.
- Critical target decision:
  - D257 trained on targets clipped to `+-0.040rad`, matching current
    `bc_teacher_policy_delta_clip_rad`;
  - train-clean-only raw target clip exceed rate is
    `0.14844941179761922`;
  - if the next session raises the teacher cap, retrain this checkpoint.
- Wrote the next-session PPO smoke command:
  `state_action_teacher_d257/ppo_data_prior_smoke_command_d257.txt`.
- Verdict:
  - `D257_STATE_ACTION_TEACHER_CHECKPOINT_PASS_NO_PPO_RUNTIME`.

## Previous Result: D256

- D256 converted the D247/D248 pair data into `194000` transition rows and
  `142978` PPO teacher-prior rows from `train_clean_positive`.
- D256 output root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/rl_transition_preflight_d256`.
- Verdict:
  `D256_RL_TRANSITION_REWARD_PREFLIGHT_PASS_NO_TRAINING`.

## Previous Result: D255

- Checked the next optional runtime gate after D254: a 50-step SmolVLA
  training-loop smoke using `train_clean_positive`.
- Did not start training. CUDA preflight found:
  - host GPU is visible through `nvidia-smi`;
  - `isaaclab` env can see CUDA and the RTX 4090 Laptop GPU;
  - `lerobot` env has the `lerobot-train` path but reports
    `torch.cuda.is_available() == False`, `device_count == 0`, with
    `Can't initialize NVML`;
  - quick env overrides did not fix `lerobot` CUDA visibility.
- Verdict:
  - `D255_SMOLVLA_SMOKE_BLOCKED_LEROBOT_ENV_CUDA_FALSE_NO_TRAINING`;
  - running smoke now would test a broken Python environment or CPU fallback,
    not the intended LeRobot GPU training path.

## Previous Result: D254

- Reframed the professor branch as a method-pipeline proof rather than a data
  preview or a premature SmolVLA training claim.
- Decision:
  - `D254_METHOD_PIPELINE_FRAMING_LOCKED_NO_TRAINING`;
  - D246-D253 prove the pipeline through data generation, label validation,
    LeRobot storage, split curation, and training-input preflight;
  - SmolVLA smoke is optional training-loop connectivity verification, not the
    core method result;
  - no model-performance claim exists yet.

## Previous Result: D246

- Completed the approved local 0-999 top-view render and post-render numeric
  labeling.
- Decision:
  - `LOCAL_0_999_RENDER_D242_COMPLETE_POSTRENDER_LABELS_D246`;
  - raw render exists, but LeRobot v3 conversion has not been run yet;
  - camera-contract usable subset is `986/1000`;
  - camera-gated labels are `819` clean useful taps and `167` contact/reaction
    with overshoot;
  - local available space is now about `33G`, so further conversion/scale-up must
    be storage-gated.

## Previous Result: D245

- Removed only approved SmolVLA checkpoint `training_state` directories after
  manifest.
- Decision:
  - `OUTPUTS_TRAINING_STATE_CLEANUP_COMPLETE_D245`;
  - `pretrained_model` model artifacts remain preserved;
  - exact optimizer/scheduler/RNG resume from affected old SmolVLA checkpoints is
    intentionally lost;
  - local available space is now about `82G`;
  - 0-999 render is not automatically launched by this cleanup.

## Previous Result: D244

- Removed only the approved P6v12 raw PNG dump after writing a manifest.
- Decision:
  - `P6V12_RAW_FRAMES_CLEANUP_COMPLETE_D244`;
  - `frames` raw/debug dump removed;
  - compact P6v12 lab-meeting evidence remains preserved;
  - local available space is now about `60G`;
  - actual 0-999 render still requires a fresh runtime decision because `60G`
    is close to the raw-PNG-first lower bound and leaves limited margin.

## Previous Result: D243

- Actual local 0-999 runtime was approved but not started after disk/output-root
  preflight.
- Decision:
  - `LOCAL_0_999_RUNTIME_NOT_STARTED_DISK_HARD_BLOCK_D243`;
  - GPU is visible, but disk is the blocker;
  - current renderer needs about `52.49GB` minimum for 1000ep raw-PNG-first
    output plus expected artifacts, before safety margin;
  - local available space is only about `27.69GB`;
  - next safe step is external/output-root storage, an approved cleanup/archive
    plan, or a separately approved chunked/streaming pipeline change.

## Previous Result: D242

- Added and validate-only checked a 0-999-capable manifest renderer.
- Decision:
  - `MANIFEST_RENDERER_0_999_VALIDATE_ONLY_PASS_RENDER_STILL_BLOCKED`;
  - The old 100ep renderer remains unchanged;
  - The new renderer accepts the D241 label-aware 0-999 manifest and writes a
    validation summary without launching IsaacLab;
  - No render output root was created;
  - Next runtime step, only if explicitly approved, is actual 0-999 render with
    disk/output-root preflight and `--render-approved`.

## Previous Result: D241

- Implemented and validated the label-aware 0-999 manifest generator only.
- Decision:
  - `LABEL_AWARE_0_999_MANIFEST_PASS_RENDERER_UPDATE_STILL_BLOCKED`;
  - The generated manifest keeps renderer-required fields and adds intent fields
    without writing final labels;
  - All rows remain marked for post-render numeric label validation;
  - The next non-render step is a 0-999 renderer update/new renderer design with
    validate-only guard;
  - Any actual 0-999 render remains blocked pending explicit runtime approval,
    disk/output-root preflight, and renderer validation.

## Previous Result: D240

- Converted the successful d241 0-99 render to validated LeRobot AV1 and
  generated aligned companion metadata.
- Decision:
  - `D241_LEROBOT_AV1_COMPANION_METADATA_PNG_EXTRACTION_PASS_LABEL_AWARE_0_999_DESIGN_DRAFTED`;
  - AV1 remains acceptable for this branch based on local d241 LeRobot decode and
    the earlier RunPod/H100 smoke decode gate, with H264 kept only as a fallback
    if a later target training environment fails AV1;
  - `split_candidate` remains a sampling bucket, not a final label;
  - If the goal is 1000 clean train-positive demonstrations, a naive 1000 episode
    render is not enough because d241 observed only `61/100` useful clean;
  - Next non-render step is a label-aware 0-999 manifest generator/validation;
    next runtime render remains blocked pending explicit approval and disk/
    output-root preflight.

## Previous Result: D239

- Rendered the first local 0-99 top-view chunk and generated per-episode numeric
  labels from actual post-render metrics.
- Decision:
  - `CHUNK100_RENDER_D241_COMPLETE_POSTRENDER_LABELS_COMPLETE_LEROBOT_NOT_RUN`;
  - The camera coverage gate for the rendered d241 chunk passed at full-frame
    visibility/projection level, including boundary candidates;
  - The sampling buckets cannot be treated as labels because train/eval buckets
    contain both clean and overshoot outcomes;
  - Next gated step, only after explicit approval, is LeRobot AV1 conversion for
    d241 plus companion metadata, LeRobot load/decode, PNG extraction, decoded-vs-
    source pixel diff, and row alignment.

## Previous Result: D238

- Wrote the manifest-fed chunk renderer and validated only the non-render front
  gate.
- Decision:
  - `MANIFEST_FED_CHUNK_RENDERER_VALIDATE_ONLY_PASS_RENDER_NOT_RUN`;
  - The next runtime step is exactly one 0-99 render only after explicit launch
    approval;
  - Follow-on gates after render: LeRobot AV1 conversion, companion metadata,
    LeRobot load/decode, PNG extraction, source-vs-decoded pixel diff, row
    alignment, storage projection, visibility/reprojection.

## Previous Result: D237

- Generated and validated the deterministic 0-99 manifest from the D236 sampling
  contract.
- Decision:
  - `CHUNK100_MANIFEST_PASS_RENDERER_STILL_BLOCKED`;
  - Next non-render implementation step is a manifest-fed chunk renderer that
    refuses to run without this manifest;
  - Next runtime step, only after explicit launch approval, is 0-99 render +
    LeRobot AV1 conversion + companion metadata + gate validation;
  - No Isaac render, 0-99 data generation, or 0-999 expansion has run.

## Previous Result: D236

- User asked to proceed step-by-step with critical rechecking.
- After D235 schema validation, checked D232/D233 sampling evidence and found a
  blocker before renderer work: D232 requires explicit dataset splits and
  recorded sampling ranges/seeds; D233 smoke only covered five camera poses.
- Decision:
  - `CHUNK100_SAMPLING_MANIFEST_REQUIRED_BEFORE_RENDERER_NO_RENDER`;
  - First write/confirm a 0-99 manifest, then build a chunk renderer that refuses
    to run without that manifest;
  - Keep the smoke script capped at 1-10 episodes;
  - No 0-99 or 0-999 render has been run.

## Previous Result: D235

- User asked to proceed step-by-step after clarifying parquet and rich metadata.
- Verified that the existing smoke LeRobot core parquet contains only
  `observation.state`, `action`, and timestamp/index columns while video is stored
  separately as `observation.images.top`.
- Decision:
  - `LEROBOT_CORE_PLUS_COMPANION_METADATA_SCHEMA_PASS_NO_RENDER_NO_100EP`;
  - Keep LeRobot core standard for future training compatibility;
  - Store camera/cube/projection/visibility/contact audit fields in companion
    metadata joined by frame indices;
  - Next valid implementation step is a fresh 0-99 chunk render/convert only
    after explicit launch approval and disk/output-root preflight.

## Previous Result: D234

- User approved proceeding with professor view/format, storage/output-root, and
  RunPod/H100 AV1 dataloader gates.
- No Isaac render, 100 episode chunk, 1000/10000 generation, deletion/archive/
  move, PPO/L2/Large PPO, VLA/action-teacher, RoArm, SSH JHPark/B200, pull,
  `.ssh` copy, or Track A work.
- Created:
  - `claudedocs/professor_view_format_packet_cube10cm_top_view_d234.md`
  - `claudedocs/session_20260613_cube10cm_top_view_professor_storage_runpod_d234.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/professor_review_contact_sheet_d234.png`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runpod_d234/cube10cm_runpod_h100_av1_decode_preflight_d234.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runpod_d234/cube10cm_runpod_h100_av1_decode_preflight_full_d234.json`
- RunPod/H100 result:
  - 50-sample decode PASS: avg/max `0.032439069747924806s` /
    `0.11921572685241699s`;
  - full 975-frame decode PASS: avg/max `0.017871856689453125s` /
    `0.10865616798400879s`;
  - codec/pix_fmt/fps `av1/yuv420p/30`;
  - state/action shape `[6]` / `[6]`.
- Decision:
  - `PROFESSOR_PACKET_READY_STORAGE_ROOT_DECIDED_RUNPOD_H100_AV1_DATALOADER_PASS_100EP_NOT_RUN`;
  - AV1 is selected for the next 100ep chunk by current local + RunPod evidence;
  - 100ep chunk still has not been run and should be launched only by a fresh
    explicit run instruction.

## Previous Result: D233

- One approved local IsaacLab render smoke plus local LeRobot conversion/load
  validation; no deletion, archive, move, PPO/L2/Large PPO, VLA/action-teacher,
  RoArm, SSH/B200, pull, Track A, 100 episode chunk, or 1000/10000 episode
  generation.
- Added:
  - `sim_scripts/cube10cm_top_view_visual_smoke_render.py`
  - `sim_scripts/cube10cm_top_view_smoke_to_lerobot.py`
  - `extract_frames.py`
- Render result:
  - 5 episodes, 975 frames, `1280x720`;
  - camera contract `cube10cm_top_view_v1_candidate`;
  - contract violations `[]`;
  - reprojection centroid median/max `3.074639061891291px` /
    `9.956731449704932px`;
  - all-frame visibility `975/975` full;
  - contact-window visibility `882/882` full;
  - render elapsed `180.79416966438293s`, effective captured fps
    `5.392873021347648`.
- LeRobot result:
  - status `PASS`;
  - video key `observation.images.top`;
  - codec `av1`, pix_fmt `yuv420p`, fps `30`;
  - total frames/episodes `975/5`, frame count match `true`;
  - sampled image/state/action shapes `[720,1280,3]`, `[6]`, `[6]`;
  - sampled decode avg/max `0.016793251037597656s` /
    `0.06672263145446777s`;
  - sampled source PNG vs decoded MP4 mean abs max `0.8939572482638889`.
- Storage result:
  - debug PNG `52.3344778MB/episode`, projected `52.3344778GB` per 1000
    episodes and `523.344778GB` per 10000 episodes;
  - LeRobot AV1 video `0.5964878MB/episode`, projected `0.5964878GB` per 1000
    episodes and `5.964878GB` per 10000 episodes;
  - local disk after smoke about `590G` total / `529G` used / `32G` free.
- Decision:
  - `TOP_VIEW_CAMERA_CONTRACT_V1_LOCAL_SMOKE_PASS_LEROBOT_AV1_LOCAL_PASS_SCALEUP_BLOCKED`;
  - v6 was only a codec/backend fixture, not the professor schema;
  - primary storage remains LeRobot MP4+parquet and PNG remains smoke/debug/
    extraction only;
  - AV1 is locally acceptable through LeRobot, but RunPod/H100 dataloader
    decode/speed must be verified before scale-up there;
  - 100 episode chunk requires professor confirmation, storage/output-root
    decision, and explicit approval.

## Previous Result: D232

- Local audit/documentation decision only; no deletion, no runtime, no Isaac Sim render, no PPO, no dataset generation, no VLA/action-teacher, no RoArm, no SSH/B200, no pull, and no Track A.
- D232 camera contract decision:
  - The top-view sim camera must be chosen from a real Azure Kinect mounting plan, not from a visually pleasing sim viewpoint.
  - Intrinsics from the old Kinect calibration can inform the sim camera; extrinsics must be newly specified for the top-view setup.
  - Required smoke metrics: marker/corner reprojection check, cube self-occlusion frame rates, render seconds per episode, MB per episode, codec decode-vs-source check, and LeRobot load check.
- D232 disk audit:
  - `outputs` is about 96G, mostly repeated SmolVLA checkpoints.
  - Preserve SmolVLA `outputs/` by default. If disk pressure appears, do not
    re-scan randomly or delete arbitrary runs; follow the D232 outputs cleanup
    order only after explicit approval and manifest.
  - First approved outputs cleanup path: remove/archive only
    `outputs/*/checkpoints/*/training_state` after manifest. Estimated reclaim:
    about 25.6GB decimal, while preserving `pretrained_model` inference
    artifacts; resume training state is lost.
  - Larger outputs cleanup path, only if more space is needed: keep one
    representative checkpoint per run
    (`smolvla_official=050000`, `smolvla_v2_cleaned=050000`,
    `smolvla_v3_sponge=050000`, `smolvla_v5_multipos=200000`,
    `smolvla_v6=last`, `smolvla_v6_b200=last`,
    `smolvla_v6_stacking_b200=last`,
    `smolvla_v6_stacking_v2_b200=010000`,
    `smolvla_v6_stacking_v3_b200=020000`). Estimated reclaim: about 90.15GB
    decimal total; old four runs alone about 74.1GB decimal.
  - `claudedocs/figures/p6v12_rollout/frames` is about 34G with 73969 raw frame files and is the highest-value cleanup/archive candidate.
  - `collected_data*`, `b200_backup_*`, and `openvla_oft_b200_pulls` are needed
    data/backups. They are archive/move-only with explicit approval, not cleanup
    or blind-delete candidates.

## Previous Result: D231

- Ran local asset/code audit only; no IsaacLab runtime, GPU, PPO, L2/Large PPO, dataset, VLA/action-teacher, RoArm, SSH/B200, pull, or Track A.
- Added `sim_scripts/cube10cm_tool_contact_proxy_asset_audit.py`.
- D231 audit summary:
  - USD converter config: `collision_from_visuals=False`, `collider_type=convex_hull`.
  - Native bbox sizes:
    `link5_collision=[0.046496,0.035520,0.120635]`,
    `gripper_collision=[0.004,0.004,0.004]`,
    `gripper_visual=[0.077850,0.025240,0.039368]`.
  - q0 link5-frame distal z:
    `link5_collision=0.119885620`,
    `gripper_collision=0.054035007`,
    `gripper_visual=0.119117587`,
    `hand_tcp=0.115428`.
- D231 decision:
  - Option 2 is directionally right only as `tool_surface_union`, not as direct `gripper_link_collision`.
  - Do not silently switch the metric to `gripper_link_collision_g2a`.
  - PPO is not unblocked. Before PPO/constant-baseline runtime, either explicitly accept current `link5_collision_aabb` as the fixed-jaw tool proxy or implement `tool_surface_union` and rerun zero/base metric-equivalence.

## Previous Result: D230

- Ran base-only local RTX4090/IsaacLab diagnostics; no PPO, no L2/Large PPO, no dataset, no VLA/action-teacher, no RoArm, no SSH/B200.
- Important metric correction:
  - `--tap_success_terminate` is legacy target-band termination and can make `*_final` useful metrics look like partial post-reset episodes.
  - For useful-tap discovery, use no legacy success termination; for per-env pose localization, also disable overshoot termination and read `overshoot_seen`.
- Code/audit changes:
  - Added summary final fields: `useful_seen_final`, `contact_reaction_seen_final`, `no_overshoot_seen_final`.
  - Added `--per_env_summary_json` and `--disable_tap_overshoot_terminate` for diagnostics.
  - Fixed tap env termination so `cfg.tap_overshoot_terminate` is actually respected; default remains terminate-on-overshoot unless the diagnostic flag disables it.
- D230 runtime evidence:
  - Fixed corners in xy10 band with no legacy success termination were clean useful-pass:
    `(0.14,-0.10)`, `(0.34,-0.10)`, `(0.14,+0.10)`, `(0.34,+0.10)` all had contact/reaction/useful final `1.0` and overshoot `0.0`.
  - Random xy10 no-termination per-env seeds `1036` and `1037`, n64 each, were stable:
    contact/reaction final `1.0`, useful final `0.828125`, no-overshoot final `0.828125`, overshoot `0.171875` in each seed.
  - Combined per-env: `22/128` overshoot, `106/128` useful-clean.
  - Exact replay of one overshoot sample `(x=0.197617,y=-0.085433)` as a fixed pose was clean-pass, so the current failure is a randomized-band/trajectory condition, not a durable fixed-pose failure.
- D230 interpretation:
  - Base useful-tap failure exists in xy +/-10cm randomization, but the failure mode is overshoot after contact/reaction, not missing contact.
  - The thesis-shaped statement should be: "In a pose perturbation band where the scripted base over-taps some cases, residual RL should recover clean useful tap by reducing overshoot while preserving contact/reaction."
  - This is not RL readiness yet; it only defines the next valid evaluation stage.

## Previous Result: D229

- No GPU runtime, no PPO, no reward/control/action-space change.
- Added logging-only useful-tap metrics:
  - `cube_tap_useful_now_rate`
  - `cube_tap_useful_seen_rate`
  - `cube_tap_contact_reaction_seen_rate`
  - `cube_tap_no_overshoot_seen_rate`
- The smoke summary now reports:
  - `useful_seen_max`
  - `contact_reaction_seen_max`
  - `no_overshoot_seen_min`
- Static checks passed:
  - `python3 -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_tap10cm_ppo_smoke.py`
  - `git diff --check`
- D229 interpretation:
  - The statement "base fails in useful-tap pose regions and RL recovers useful tap" is the right thesis-shaped claim, but it is not proven yet.
  - D225/D228 weak bins are weak under the 6mm target-band, not under useful tap: their base runs already had contact/reaction `1.0` and overshoot `0.0`.
  - Therefore the next evidence step is a base-only pose-binned useful-tap failure sweep using the new metrics, not L2/Large PPO.

## Previous Result: D228

- D228 remains valid only for the 6mm target-band quality objective.
- Narrow x-band `[0.160,0.165]`, y `0.15`, seed1027/1028:
  - base event `0.34375`, target_band `0.046875`, overshoot `0`.
  - best constant target_band `0.09375`, event up to `1.0`.
  - PPO L1 post event `0.234375`, target_band `0.0625`, overshoot `0`.
- Verdict remains `TRANSITION_XBAND_CONSTANT_BASELINE_AND_L1_FAIL_NO_L2`.

## Active Direction

- PPO promotion is frozen while the professor visual-dataset branch is active.
- Current professor branch state is method-pipeline-ready plus data-prior PPO
  input-ready through the D257 state-action teacher checkpoint, not
  training-complete and not model-performance-proven.
- Recommended next work:
  1. inspect D257 outputs:
     `state_action_teacher_metrics_d257.json`,
     `cube10cm_d257_state_action_teacher_clipped0040.pt`, and
     `ppo_data_prior_smoke_command_d257.txt`;
  2. run a tiny Isaac Lab PPO data-prior smoke only with explicit runtime
     approval;
  3. verify checkpoint loading through `bc_teacher_checkpoint_path`,
     nonzero `cube_push_bc_teacher_blend_mean`, and logged
     `cube_push_bc_teacher_imitation_mse`;
  4. stop/cleanup any Isaac/Kit/PPO process and verify GPU release;
  5. do not claim learned policy or RoArm readiness until teacher-off eval passes.
- Do not start actual SmolVLA/VLA fine-tuning, PPO, action-teacher, RoArm
  deployment, RunPod runtime, or raw cleanup without explicit approval.
- Do not generate 1000/10000 additional episodes from this result.
- Do not run the xy10 useful-tap constant/PPO gate until the tool-contact proxy branch is explicitly closed:
  1. Either accept current `link5_collision_aabb` as the fixed-jaw/distal-tool metric for this sim contract; or
  2. Implement a named `tool_surface_union` contact metric and run zero/base metric-equivalence before any PPO.
- Stop treating the 6mm target-band as the primary success gate if the professor/user goal is "make a useful tap."
- Do not claim D225/D228 target-band weak bins prove useful-tap recovery.
- Do not use fixed single-pose corners as the next PPO stage; D230 says corners pass and the useful failure appears in the randomized xy10 band.
- If the RL branch is explicitly resumed later, use D229-D231/D248-D252 as
  guardrails and do not mix claims with the professor visual-dataset branch.

## Must Read First

1. `CLAUDE.md` Current-State Protocol.
2. `START_HERE.md` D257 current truth and Active Direction.
3. `claudedocs/DECISIONS.md` D257, D256, D254, D253, D249-D252, D248,
   D247, D246, D232.
4. `claudedocs/EXPERIMENT_LEDGER.md` latest D257 row.
5. `claudedocs/session_20260618_cube10cm_top_view_state_action_teacher_d257.md`.
6. `claudedocs/session_20260618_cube10cm_top_view_rl_transition_preflight_d256.md`.
7. `claudedocs/cube10cm_top_view_method_pipeline_d254.md`.
8. `claudedocs/session_20260617_cube10cm_top_view_method_pipeline_reframe_d254.md`.
9. `claudedocs/session_20260617_cube10cm_top_view_training_preflight_d253.md`.
10. `claudedocs/session_20260617_cube10cm_top_view_dataset_freeze_filtered_loader_distribution_d249_d252.md`.
11. `claudedocs/session_20260617_cube10cm_top_view_label_package_d248.md`.
12. `claudedocs/session_20260616_cube10cm_top_view_0_999_lerobot_metadata_d247.md`.
13. `claudedocs/session_20260615_cube10cm_top_view_0_999_render_labels_d246.md`.
14. `claudedocs/session_20260612_camera_contract_visual_dataset_disk_audit_d232.md`.
15. D257 output root:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/state_action_teacher_metrics_d257.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/ppo_data_prior_smoke_command_d257.txt`
16. Relevant code:
   - `sim_scripts/cube10cm_top_view_train_state_action_teacher.py`
   - `sim_scripts/cube10cm_top_view_build_rl_transition_dataset.py`
   - `roarm_rl/roarm_cube_push_env.py`
   - `roarm_rl/train_cube_push_ppo.py`
   - `sim_scripts/cube10cm_top_view_training_preflight.py`
   - `sim_scripts/cube10cm_top_view_freeze_dataset.py`
   - `sim_scripts/cube10cm_top_view_build_filtered_views.py`
   - `sim_scripts/cube10cm_top_view_filtered_dataloader_smoke.py`
   - `sim_scripts/cube10cm_top_view_split_distribution_check.py`
   - `sim_scripts/cube10cm_top_view_package_label_splits.py`
   - `sim_scripts/cube10cm_top_view_smoke_to_lerobot.py`
   - `extract_frames.py`

## Hard Blocks

- B200 is expired/disconnected: do not SSH/reconnect/pull/copy `.ssh`.
- Do not use `HANDOFF.md` or `TASKS.md` as current truth.
- Do not revert dirty/untracked/ahead state unless explicitly requested.
- Do not run broad xy L2/Large PPO from D227/D228.
- Do not run PPO on D225/D228 target-band weak bins as if they were useful-tap failures.
- Do not run fixed-corner PPO from D230; corners were clean useful-pass.
- Do not replace `link5_collision_aabb` with direct `gripper_link_collision_g2a`; D231 rejects that as a tiny proxy.
- Do not run L2/Large on xy10 useful tap until same-band constants and a small L1 pass the useful/no-overshoot gate.
- Do not generate a full dataset, start VLA/action-teacher, or deploy to RoArm from these results.
- D234 closed storage/output-root and RunPod/H100 codec gates, but 100ep chunk
  has still not been run. Launch only if the user gives a fresh explicit run
  instruction.
- RunPod cost rule: after any RunPod job, copy results back, stop the pod, and
  prefer delete/terminate if no remote volume/environment needs to be preserved.
- Do not start 1000/10000 episode rendering from D234. Local 1000/10000 scale
  remains blocked.
- Any additional 5-10 episode smoke render still requires explicit approval.
- Do not delete SmolVLA `outputs/` by default. Under disk pressure, use the D232
  outputs cleanup order only after manifest and explicit approval: first
  `training_state` (~25.6GB), then run-specific keep-one pruning (~90.15GB total
  possible).
- Do not delete `collected_data*`, `b200_backup_*`, or
  `openvla_oft_b200_pulls`; they are needed data/backups and are archive/move
  only with explicit approval.
- Do not claim RL readiness.
