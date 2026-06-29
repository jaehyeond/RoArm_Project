# START_HERE.md

Last updated: 2026-06-30 KST (D307 current truth: a default-off non-PPO action governor was added to `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py` after D306. D306 candidate-2 ep561 with `predict_stop`, horizon `0.020s`, speed stop `0.200m/s`, and the true D304 runtime action contract changed the D306 overshoot case from `0.041465m` XY with overshoot to useful `1.0`, overshoot `0.0`, cap `0.0`, max XY `0.004996m`. The same setting over D304 failed6 gave useful `1.0`, overshoot `0.0`, cap `0.0`, mean/max XY `0.002727/0.007170m`, but only `4/6` envs reached `>=1mm`; episodes `991` and `29` stayed at `0.023/0.027mm`. A recorded-target supervised repair reached offline val MSE/cosine `0.030512/0.883410` and checkpoint sha256 `2d2bc75c30c0fb2241bf7a6230cc2513abac6a9a3ccfe5a7fd769479f4a1fa60`, but runtime failed6 collapsed to mean/max XY `0.0000154/0.0000228m`. D307 is partial and no-promotion. Do not run long PPO, tiny PPO trace gate, PPO ladder, partial actor preservation, or real actor update from D307; next work is non-PPO deployable action-space/control repair.)

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

## Latest Result: D307

- D307 purpose:
  - Continue from D306 without PPO.
  - Test whether a displacement/velocity-aware action governor can turn the
    D306 tiny-vs-overshoot bracket into controlled 1mm+ displacement.
  - Check whether recorded-target supervised repair can fix the low-displacement
    failed cases before any PPO gate.
- Code change:
  - Added default-off `action_governor_mode=off|predict_stop|predict_brake` to
    `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`.
  - The governor uses current displacement, cube speed, contact state, and a
    projected displacement horizon. This is a non-PPO diagnostic/prototype only.
- Best D307 governor result:
  - Actor: D306 candidate-2
    `phase_iter2_replay_plus_failed6_lr5e5_ep100/model_actor_d256_replay_batches_d290.pt`.
  - Runtime contract: true D304 contract with `max_joint_delta_per_step_rad=0.04`,
    `contact_joint_delta_scale=0.35`, `fast_cube_joint_delta_scale=0.2`,
    `action_smoothing_alpha=0.25`, `tap_stop_after_disp_m=0.003`, no useful or
    overshoot terminate.
  - ep561 `predict_stop`, horizon `0.020s`, speed stop `0.200m/s`: useful `1.0`,
    overshoot `0.0`, cap `0.0`, max XY `0.0049958923m`.
  - This fixes the D306 ep561 overshoot bracket locally: candidate-2 no-projection
    was `0.041465m` XY with overshoot `1.0`.
- D307 failed6 result:
  - Same governor over D304 failed episodes `561,265,341,991,536,29`:
    useful `1.0`, overshoot `0.0`, cap `0.0`, mean/max XY
    `0.0027265977` / `0.0071698078m`.
  - XY `>=1mm` rate: `0.6666667`; XY `>=3mm` rate: `0.3333333`.
  - Episodes `991` and `29` stayed tiny: `0.023mm` and `0.027mm`.
  - Interpretation: governor solves late-stopping/overshoot in some cases, but
    not action-direction/contact-geometry failures.
- D307 recorded-target repair:
  - Built recorded-target dataset from the D307 failed6 closed-loop states.
  - Fine-tuned from D306 candidate-2 for 80 epochs on D304 failed6 replay plus
    D307 recorded-target data.
  - Checkpoint sha256:
    `2d2bc75c30c0fb2241bf7a6230cc2513abac6a9a3ccfe5a7fd769479f4a1fa60`.
  - Offline val MSE/cosine: `0.0305119734` / `0.8834095001`.
  - Runtime failed6 with the same governor collapsed: useful `1.0`,
    overshoot `0.0`, cap `0.0`, mean/max XY
    `0.0000154146` / `0.0000227652m`, XY `>=1mm` rate `0.0`.
- Decision:
  - No long PPO.
  - No tiny PPO trace gate.
  - No PPO ladder, partial actor preservation, or real actor update.
  - Do not promote D307 as learned policy or RoArm readiness.
  - Next work is non-PPO deployable action-space/control repair: either move a
    default-off displacement/velocity governor into the env and broaden fresh
    reset diagnostics, or change the action representation toward a tool/object
    push primitive instead of brittle scalar joint deltas.
- D307 verdict:
  `D307_ACTION_GOVERNOR_PARTIAL_NO_PPO_PROMOTION`.
- Primary D307 artifacts:
  - `claudedocs/session_20260630_cube10cm_top_view_d307_action_governor.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/failed6_predict_stop_h020_v200/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/recorded_repair_failed6_predict_stop_h020_v200/`

## Previous Result: D306

- D306 purpose:
  - Continue from D305 without PPO.
  - Try phase/displacement-aware supervised action repair before any new PPO
    trace gate.
  - Determine whether the actor can satisfy contact/useful, no overshoot, and
    at least Tier-1 displacement under the true D304 runtime action contract.
- Candidate-1 target rewrite and training:
  - Source data: D305 candidate-1 closed-loop recovery dataset plus D304 failed6
    D256 replay data.
  - Target rewrite: recovery weight `0.65 -> 0.10`, transition steps `40..260`,
    target clip `0.85`, smooth alpha `0.45`.
  - Target clip >=0.99 became `0.0`; late push target-vs-recorded cosine was
    `0.9524435997`.
  - Trained checkpoint sha256:
    `a407729e342197dffd2b6395dafff1a7b6c7cb55d252c013efbfb9817530427c`.
  - Train val MSE/cosine: `0.0334460661` / `0.8521609306`.
  - Offline actor-vs-target diagnostic passed: MSE/cosine
    `0.0311942883` / `0.8587358594`.
- Candidate-1 rollout:
  - Rechecked ep561 under the true D304 action contract:
    `max_joint_delta_per_step_rad=0.04`, `contact_joint_delta_scale=0.35`,
    `fast_cube_joint_delta_scale=0.2`, `action_smoothing_alpha=0.25`,
    `tap_stop_after_disp_m=0.003`.
  - Result: useful `1.0`, overshoot `0.0`, cap `0.0`, but max XY/along only
    `0.000037225m` / `0.000027448m`.
  - Per-step trace found late push underpower: in steps `300..579`, D256
    recorded elbow/wrist_pitch abs mean was `0.9646/0.7303`, while the actor was
    only `0.3992/0.0960`.
- Candidate-2 stronger late-push repair:
  - Collected D306 failed6 closed-loop states under the D304 runtime contract.
  - Target rewrite: recovery weight `0.50 -> 0.00`, transition steps `40..180`,
    target clip `1.0`, smooth alpha `0.80`.
  - Trained checkpoint sha256:
    `8f5d154f9ba76bc467e96f73ed3017e21dd6b8ead265d547c3cadc4ff30844b5`.
  - Train val MSE/cosine: `0.0345205478` / `0.8549892306`.
- Candidate-2 action projection checks on ep561:
  - No projection: useful `1.0`, cap `0.0`, but overshoot `1.0` and max XY
    `0.041465m`.
  - `exec_action_clip_abs=0.50`: useful `1.0`, overshoot `0.0`, cap `0.0`, but
    max XY only `0.0000428m`.
  - `exec_action_clip_abs=0.75`: useful `1.0`, overshoot `0.0`, cap `0.0`, but
    max XY only `0.0000450m`.
  - contact slowdown proxy: useful `1.0`, overshoot `0.0`, cap `0.0`, but max
    XY only `0.0000364m`.
- Interpretation:
  - D306 found a real bracket: safe actions are too weak, unprojected strong
    recorded-like actions overshoot.
  - Simple supervised actor fitting, global action clipping, and contact
    slowdown do not create a controlled 1mm..3mm push.
  - The next issue is a threshold/impulse control problem, not another PPO
    runtime candidate.
- Decision:
  - No long PPO.
  - No tiny PPO trace gate.
  - No PPO ladder, partial actor preservation, or real actor update.
  - Do not promote D306 as learned policy or RoArm readiness.
  - Next work is a non-PPO displacement/velocity-aware action governor or push
    pulse controller that uses current displacement, cube velocity, contact
    state, and pre-overshoot braking before any PPO gate.
- D306 verdict:
  `D306_PHASE_ACTION_REPAIR_BRACKETED_TINY_VS_OVERSHOOT_NO_PPO_PROMOTION`.
- Primary D306 artifacts:
  - `claudedocs/session_20260630_cube10cm_top_view_d306_phase_action_repair.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_c1_replay_plus_phase_lr5e5_ep100/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/fresh_onebin_iter2_d304runtime_ep561/`

## Previous Result: D305

- D305 purpose:
  - Continue from D304 without PPO.
  - Repair the actor/teacher bridge through supervised closed-loop recovery data
    and action-constraint diagnostics on the D304 failed episodes
    `561,265,341,991,536,29`.
  - Decide whether the result is strong enough for another tiny PPO trace gate.
- Candidate-1 actor repair:
  - Source actor: D304 actor-preserved `model_0.pt`.
  - Training data: D304 failed6 D256 recorded replay dataset plus D304 failed6
    closed-loop recovery dataset.
  - Non-PPO supervised fit, `lr=1e-4`, `epochs=80`.
  - Checkpoint sha256:
    `07043ec3d75f70f08dbd827d029578d1c5a1d3be2d1a208035672fcd17b43b1d`.
  - Train summary: final val MSE/cosine `0.1098324433` / `0.8255895972`
    with WARN due inherited recovery-data quality.
  - Offline actor-vs-replay/recovery diagnostic passed: aggregate MSE/cosine
    `0.1072969660` / `0.8226829171`.
- Candidate-1 rollout diagnostics:
  - Fresh default one-bin probes restored contact/reaction/useful max `1.0` and
    overshoot `0.0` on all failed6.
  - Safe-bin passed for ep265, ep341, and ep29.
  - ep561, ep536, and ep991 still failed safe-bin because cap stayed
    `0.333333`.
  - D304-like no-useful-stop + `tap_stop_after_disp_m=0.003` on ep561/265/991
    kept useful `1.0` and overshoot `0.0`, but cap was high
    `0.333333/0.566667/0.666667` and displacement remained tiny
    `~0.013..0.016mm`.
  - Closed-loop recovery after candidate-1 improved useful to `1.0`, overshoot
    `0.0`, and mean/max XY to `0.0011089662m` / `0.0035506743m`, but
    actor-vs-recovery MSE remained high `0.890877` and cap max was
    `0.277778`.
- Candidate-2 and action-clip controls:
  - Candidate-2 trained from candidate-1 on D304 replay plus candidate-1
    recovery data, `lr=5e-5`, `epochs=80`.
  - Candidate-2 checkpoint sha256:
    `fb1fc7137574face1c0f4c55beb16fdaf71012ca98a6c059e39901e32e1fd880`.
  - Candidate-2 reduced recovery MSE to `0.705775` and cap max to `0.111111`,
    but useful dropped to `0.833333` and displacement collapsed to
    `0.0000167m` mean / `0.0000382m` max.
  - `exec_action_clip_abs=0.75` with candidate-1 did not fix cap:
    ep561/265/991 cap `0.333333/0.5/0.666667`.
- Interpretation:
  - D305 did solve the D304 no-contact failure mode in the narrow failed6
    diagnostic set.
  - It did not solve the full promotion problem because displacement and cap
    control remain unstable under the D304 collection contract.
  - Candidate-1 is behaviorally better than candidate-2, but neither is a PPO
    candidate.
- Decision:
  - No long PPO.
  - No tiny PPO trace gate yet.
  - No PPO ladder, partial actor preservation, or real actor update.
  - Do not promote D305 as learned policy or RoArm readiness.
  - Next work is phase/displacement-aware non-PPO action repair: separate
    approach/contact/push phases or add supervised loss terms/constraints for
    cap pressure, action smoothness, and minimum displacement.
- D305 verdict:
  `D305_CLOSED_LOOP_RECOVERY_REPAIR_PARTIAL_CONTACT_RESTORED_NO_PPO_PROMOTION`.
- Primary D305 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d305_closed_loop_recovery_repair.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_iter2_recovery_lr5e5_ep80/`

## Previous Result: D304

- D304 purpose:
  - Run exactly one tiny no-success-terminate actor-preserved PPO collection
    trace gate.
  - Do not claim learning success. Capture true PPO collection-path failed envs
    through `collection_final_env_trace_iter_0.jsonl`.
  - Compare collection-final failures against fresh one-bin probes, D256
    recorded-action replay, offline actor-vs-D256 matching, and closed-loop
    recovery.
- PPO trace gate result:
  - PPO clean exit, `actor_preserve_blend=1.0`.
  - Actor preservation: `max_pre_restore_delta=0.228326231`,
    `max_post_restore_delta=0.000000000`.
  - `model_0.pt` sha256:
    `753df107215e434a421da8eb029f2daf8c028c0f33ab4b4be55d945511e6d971`.
  - Trace JSONL sha256:
    `38bc56857210d25cf46dc17db55f8843b2504afff15a49dcf275c98fe0848291`.
- TensorBoard/collection-final result:
  - Verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`.
  - Contact/reaction `0.84375`, useful `0.8125`, success `0.84375`,
    overshoot `0.03125`.
  - XY >=1mm `0.625`, XY >=3mm `0.5625`, mean/max XY
    `0.0037104846m` / `0.0537344441m`.
  - Gate failed because collection-final contact/reaction and useful are below
    the strict `0.90` promotion threshold.
- Failed collection-final envs:
  - env4 ep561: no contact/useful, no overshoot, XY `0.004266m`,
    stop-after-displacement held.
  - env5 ep265: contact true but useful false, overshoot true, XY
    `0.053734m`.
  - env14 ep341: no contact/useful, no overshoot, tiny XY `0.0000117m`.
  - env15 ep991: no contact/useful, no overshoot, tiny XY `0.0000116m`.
  - env22 ep536: no contact/useful, no overshoot, tiny XY `0.0000116m`.
  - env31 ep29: no contact/useful, no overshoot, tiny XY `0.0000115m`.
- Follow-up diagnostics:
  - Fresh one-bin noise `0.005` probes: ep561 fail, ep265 useful `0.8` with
    overshoot `0.2`, ep341 partial `0.8`, ep991 fail, ep536 pass, ep29 pass.
  - Deterministic fresh probes: ep561, ep265, and ep991 all failed useful `0.0`;
    ep265 and ep991 showed cap pressure `0.833333`.
  - D256 recorded-action replay for failed6 passed: contact/useful `1.0`,
    overshoot `0.0`, mean/max XY `0.0097861877m` / `0.0161318127m`.
  - Offline actor-vs-D256 on failed6 passed: MSE `0.0061870781`, MAE
    `0.0377806723`, cosine `0.9509615898`.
  - Closed-loop recovery on failed6 warned: useful `0.833333`, overshoot `0.0`,
    actor-vs-recovery MSE `1.084940`, recovery clip rate mean/max
    `0.710805` / `0.966667`.
- Interpretation:
  - D304 achieved the diagnostic goal: the true PPO collection path now exports
    failed envs directly.
  - The actor is not simply bad on static D256 rows; D256 replay and offline
    actor-vs-D256 pass.
  - The blocker is closed-loop recovery/stability on some reset states,
    especially ep561/ep265/ep991 style states, not a reason to run longer PPO.
  - `tap_success_terminate=False` remains the correct collection contract for
    this branch, but final coverage is still below promotion.
- Decision:
  - No long PPO.
  - No PPO ladder.
  - No partial actor preservation or real actor update.
  - Do not lower the `0.90` final useful/contact threshold as a promotion
    standard based on D304.
  - Next work is non-PPO closed-loop recovery/action repair: aggregate or train
    on failed-state recovery data, or add a pre-contact projection/constraint,
    then re-run fresh one-bin/direct-reset diagnostics before any new tiny PPO
    trace gate.
- D304 verdict:
  `D304_COLLECTION_TRACE_GATE_FAIL_NO_PROMOTION_CLOSED_LOOP_RECOVERY_REPAIR_NEXT`.
- Primary D304 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d304_collection_trace_gate.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/collection_final_env_trace_iter_0.jsonl`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/tensorboard_scalar_gate_d304_seed29801_trace.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/d304_failed6_d256_replay_dataset.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/offline_actor_vs_d256_failed6/offline_actor_batch_diagnostic_summary_d290.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/closed_loop_recovery_failed6/closed_loop_recovery_summary_d304_closed_loop_recovery_failed6.json`

## Previous Result: D303

- D303 purpose:
  - Cross-check and correct D302 before using it as a repair target.
  - Test whether the D301 hard episodes fail in fresh single-bin processes or
    only after sequential multi-bin reuse inside one Isaac process.
  - Compare D256 recorded-action replay, offline actor-vs-D256 action matching,
    manual closed-loop recovery, and fresh env-hook single-bin probes.
- Results:
  - D256 recorded-action replay for episodes `221,198,13,322,935` passed:
    contact/reaction/useful `1.0`, overshoot `0.0`, max XY
    `0.0018194274744018912m`.
  - Offline D300 actor-vs-D256 hard-episode batch comparison passed:
    MSE `0.007595527917146683`, MAE `0.03969154506921768`, cosine
    `0.9679368734359741`.
  - Manual closed-loop recovery over the same five episodes passed:
    useful `1.0`, overshoot `0.0`, max XY `0.00030884623993188143m`.
  - Fresh single-bin env-hook actor probes passed for ep13, ep322, and ep935
    with `num_envs=5`: useful `1.0`, overshoot `0.0`.
  - Re-running all five bins sequentially inside one Isaac process reproduced
    the later-bin failures: ep13/935 overshoot and ep322 no-useful.
- Interpretation:
  - D302's multi-bin actor/teacher hard-bin failures are superseded as
    sequential-process contamination, matching the D289 warning that multi-batch
    collection inside one Isaac process is unsafe for this probe family.
  - The hard episodes are not proven actor failures when run in fresh
    one-bin/one-process diagnostics.
  - Blind hard-bin actor repair, teacher-KL, or action projection is not the
    next step from D303.
  - The remaining unresolved issue is the true PPO collection path. Future tiny
    PPO gates must use `roarm_rl/train_cube_push_ppo.py`'s new
    `collection_final_env_trace_iter_<N>.jsonl` export so failed envs are
    captured at collection time rather than reconstructed through stale
    posthoc probes.
- Decision:
  - No long PPO.
  - No PPO ladder.
  - No partial actor preservation or real actor update.
  - Do not use sequential multi-bin Isaac probes as promotion/blocker evidence
    unless each bin is run in a fresh process.
  - Next valid runtime, only as a tiny gate, is a no-success-terminate
    actor-preserved PPO collection trace run using the new per-env JSONL export.
- D303 verdict:
  `D303_HARD_BIN_MULTI_PROCESS_REAUDIT_SUPERSEDES_D302_NO_REPAIR_YET`.
- Primary D303 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d303_hard_bin_reaudit.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_hard_episode_d256_replay_dataset.pt`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_offline_actor_vs_d256_hard_episodes/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_closed_loop_recovery_hard_episodes/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_envhook_ep13_n5_actor/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_envhook_ep322_n5_actor/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/d303_envhook_ep935_n5_actor/`

## Previous Result: D302 - superseded by D303 for multi-bin actor/teacher failure claims

> D303 supersedes the D302 interpretation that episodes `13/322/935` are
> standalone hard-bin actor or teacher failures. Those failures reproduce only
> in sequential multi-bin reuse inside one Isaac process. Use fresh-process
> one-bin probes for hard-bin evidence.

- D302 purpose:
  - Continue D301 without PPO training by probing the exact hard D256 episodes
    that failed final coverage: `221,198,13,322,935`.
  - Compare actor-only, D257 teacher-only, and actor plus small action noise on
    the same reset bins before choosing any repair.
  - Add collection-time per-env final trace export to the PPO train script so
    future tiny gates preserve the actual failed env rows.
- Code update:
  - `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py` now supports
    explicit `--episode_range`, direct-reset warmup selection, actor-vs-teacher
    direction metrics, and final face-gap/TCP diagnostics.
  - `roarm_rl/train_cube_push_ppo.py` now writes
    `collection_final_env_trace_iter_<N>.jsonl` after tap10cm collection with
    per-env D256 episode, contact/useful/overshoot, displacement, face gap,
    action magnitude, joint-delta cap, and teacher-blend diagnostics.
- Runtime:
  - Non-PPO hard-bin diagnostics only.
  - D300 seed `29604` actor checkpoint, D256 direct reset, `num_envs=8`,
    `eval_steps=580`, `link5_collision_aabb`, `tap_stop_after_disp_m=0.003`,
    BC teacher feature target `env_target`, and linear phase steps `579`.
  - Ran actor deterministic, D257 teacher-only, and actor with
    `action_noise_std=0.005`.
- Results:
  - Actor deterministic: episodes `221` and `198` passed useful/no-overshoot;
    episode `322` was partial (`0.5` useful, no overshoot); episodes `13` and
    `935` overshot (`26.3mm` and `34.2mm` max XY).
  - Actor plus `0.005` noise was essentially the same: `221/198` pass,
    `322` partial (`0.375` useful), `13/935` overshoot.
  - D257 teacher-only was not a safe repair: `221/198` passed, but
    `13/322/935` overshot (`21.3mm`, `32.9mm`, `33.8mm` max XY).
  - Joint delta cap stayed `0.0` across these hard-bin runs; this is not a
    cap-saturation failure.
- Interpretation:
  - The remaining hard-bin issue is mixed: some edge resets miss contact, while
    other hard resets produce direction/overshoot failures.
  - Blindly adding teacher-KL or teacher blending is unsafe because the D257
    teacher itself overshoots on the same hard-bin states.
  - Small stochastic action noise is not the main root cause.
  - D256 recorded state/action data, not D257 teacher alone, should drive the
    next actor/action repair.
- Decision:
  - No long PPO.
  - No PPO ladder.
  - No partial actor preservation or real actor update yet.
  - Do not relax the AABB contact band to hide failures.
  - Next work is non-PPO actor/action repair: compare hard-bin actor outputs to
    D256 recorded deltas and add either hard-bin supervised warm-start rows or
    a pre-contact action projection/approach constraint. Only after that should
    a tiny PPO gate be rerun, using the new collection-final env trace export.
- D302 verdict:
  `D302_HARD_BIN_ACTOR_TEACHER_DIAGNOSTIC_NO_PPO_NO_TEACHER_KL`.
- Primary D302 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d302_hard_bin_actor_teacher.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/actor_deterministic_seed29604_model/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/teacher_only_d257/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/hard_bin_actor_teacher_d302/actor_noise005_seed29604_model/`

## Previous Result: D301

- D301 purpose:
  - Diagnose D300's failed final envs without running PPO training.
  - Inspect episode index, action magnitude, contact proxy, displacement, and
    overshoot at per-env and per-env-step resolution.
- Code update:
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py` now writes
    expanded final per-env diagnostics through `--out_env_csv`.
  - Added `--out_env_step_csv` for non-PPO per-env step traces.
- Runtime:
  - Non-PPO frozen-checkpoint diagnostics only.
  - D300 `model_0.pt`, `num_envs=32`, `eval_steps=580`, D256 random frame-0
    reset active, `link5_collision_aabb`, `tap_stop_after_disp_m=0.003`,
    `tap_success_terminate=False`, BC teacher off, `action_mode=ppo_stochastic`.
  - The first sandbox attempt failed before env creation because Isaac/PhysX
    could not acquire CUDA. `nvidia-smi` showed the GPU was healthy, so the
    approved Isaac Lab commands were rerun outside the sandbox.
- Results:
  - Seed `29801` saved-checkpoint diagnostic did not reproduce D300's failed
    final envs: final useful/success/overshoot was `1.0/1.0/0.0`, XY `>=1mm`
    `0.53125`, mean/max XY `0.0020347752142697573/0.007077273912727833m`.
    It only failed the strict all-step RSL-like useful mean gate
    (`0.8638469827586207`).
  - Seed `29604` reproduced the final-coverage issue: final
    contact/reaction/useful/success `0.84375`, overshoot `0.0`, XY `>=1mm`
    `0.5`, mean/max XY
    `0.0021026856265962124/0.013731294311583042m`.
  - Seed `29604` failed envs were all `no_contact_seen`: envs
    `2,10,24,25,31`, D256 episodes `221,198,13,322,935`.
  - Failed envs had no overshoot and almost no cube motion
    (`~0.011mm` max XY). They started just outside the AABB face band
    (`2..7mm` outside `±0.010m`) and then moved away while action magnitude
    increased. Joint delta cap stayed `0.0`.
- Interpretation:
  - D301 points to a hard reset/state coverage problem: far cube states with low
    joint-2/joint-3 posture are not reliably closed by the actor.
  - Do not solve this by relaxing the contact band; that would hide the actor
    coverage issue.
  - Useful/success is still not enough for the mining/excavation primitive
    framing because some useful/success envs have tiny displacement. Keep
    displacement-rate gates.
  - D300 seed `29801` failures cannot be exactly recovered from the saved
    checkpoint alone; future PPO gates must export collection-time per-env
    traces directly from `train_cube_push_ppo.py`.
- Decision:
  - No long PPO.
  - No PPO ladder.
  - No partial actor preservation or real actor update.
  - Next work is non-PPO hard-bin repair: isolate the far-cube / low joint-2
    and joint-3 reset bin, run actor-vs-teacher/action-direction diagnostics,
    then add hard-bin supervised warm-start data or a pre-contact action
    projection/approach constraint.
- D301 verdict:
  `D301_FINAL_ENV_DIAGNOSTIC_EDGE_RESET_NO_CONTACT_NO_PPO`.
- Primary D301 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d301_final_env_diagnostic.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/final_env_diagnostic_d301/seed29801/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/final_env_diagnostic_d301/seed29604/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/final_env_diagnostic_d301/seed29604_trace/`

## Previous Result: D300

- D300 purpose:
  - Cross-check D298/D299 against TensorBoard gate semantics.
  - Add collection-final TensorBoard scalars so `0.90+` gates apply to final
    state, not all-step collection averages.
  - Run two explicitly approved tiny no-success-terminate actor-preserved PPO
    re-gates.
  - No long PPO, PPO ladder, partial actor preservation, real actor update,
    render, cleanup, RunPod/B200/SSH, Track A, VLA fine-tuning, or RoArm
    deployment was performed.
- Code update:
  - `roarm_rl/train_cube_push_ppo.py` now writes `CollectionFinal/...`
    TensorBoard scalars after `runner.learn(...)` for tap10cm.
  - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py` now supports
    `--require_collection_final_tap_gate` and final-state thresholds.
  - Final useful gating now uses `CollectionFinal/cube_tap_useful_seen_rate`
    directly, not the max of useful and success.
- Tiny PPO re-gates:
  - Common contract: `num_envs=32`, `max_iterations=1`,
    `num_steps_per_env=580`, D256 random frame-0 reset active,
    `link5_collision_aabb`, `tap_stop_after_disp_m=0.003`,
    `tap_success_terminate=False`, BC teacher off,
    `actor_preserve_blend=1.0`, and `init_noise_std=0.005`.
  - Seed `29801`:
    - actor `max_post_restore_delta=0.000000000`;
    - checkpoint sha256
      `753df107215e434a421da8eb029f2daf8c028c0f33ab4b4be55d945511e6d971`;
    - collection-average useful/overshoot
      `0.7658405303955078/0.0018318966031074524`;
    - collection-final contact/reaction `0.84375`, useful `0.8125`, success
      `0.84375`, overshoot `0.03125`, XY `>=1mm` `0.625`, mean/max XY
      `0.0037104845978319645/0.053734444081783295m`;
    - verdict `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`.
  - Seed `29604`:
    - actor `max_post_restore_delta=0.000000000`;
    - checkpoint sha256
      `39e7080988517cab1ad017d9bc4f3ee69973eac351ae16ce6b583562d68eaf7b`;
    - collection-average useful/overshoot `0.7738685607910156/0.0`;
    - collection-final contact/reaction/useful/success `0.875`, overshoot
      `0.0`, XY `>=1mm` `0.65625`, mean/max XY
      `0.0023780229967087507/0.006142078433185816m`;
    - verdict `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`.
- Interpretation:
  - D300 confirms D299: removing `tap_success_terminate` fixes the major
    overshoot failure mode from D298.
  - The remaining blocker is final-state coverage, not overshoot.
  - Seed `29604` misses the strict `0.90` final useful gate by one env
    (`28/32 = 0.875`), while seed `29801` misses by more and has one overshoot
    env.
  - Lowering the final useful gate to `0.85` would make seed `29604` look
    acceptable, but that weakens the promotion standard. Keep `0.90` for
    promotion unless the user explicitly chooses a weaker exploratory gate.
  - D300 is still not learned-policy success because actor was fully preserved
    and no completed-episode Train reward scalars exist.
- Decision:
  - Do not run long PPO.
  - Do not run a PPO ladder.
  - Do not use `tap_success_terminate=True` for this actor-preserved tap10cm
    collection gate.
  - Do not claim learned-policy success, RoArm readiness, or mining automation
    readiness.
  - Next work is non-PPO final-coverage diagnostic: identify which final envs
    miss contact/useful, then inspect episode index, action magnitude, contact
    proxy, displacement, and overshoot.
- D300 verdict:
  `D300_COLLECTION_FINAL_TENSORBOARD_GATE_FAIL_NO_PROMOTION`.
- Primary D300 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d300_collection_final_gate.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29801_1it/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/`

## Previous Result: D299

- D299 purpose:
  - Diagnose D298's mismatch between PPO collection failure and teacher-off
    direct-reset pass.
  - Test whether `tap_success_terminate=True` caused unsafe episode recycle.
  - Run one no-success-terminate tiny PPO re-gate after the non-PPO diagnostic.
  - No long PPO, PPO ladder, partial actor preservation, render, cleanup,
    RunPod/B200/SSH, Track A, VLA fine-tuning, or RoArm deployment was
    performed.
- Code diagnostic update:
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py` now supports
    `--tap_success_terminate`, `--action_mode inference|ppo_stochastic`,
    RSL-like pre-reset log aggregation, and done/reset tracing.
  - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py` now supports
    `--allow_missing_train_episode_scalars` for no-termination gates.
- Non-PPO collection contract diagnostic:
  - `tap_success_terminate=True` reproduced failure:
    - inference: useful `0.09375`, overshoot `0.875`;
    - PPO-like stochastic: useful `0.0`, overshoot `0.84375`, max XY
      `13.797537803649902m`;
  - stochastic no-termination removed overshoot:
    - seed `29801`: useful `1.0`, overshoot `0.0`, max XY
      `0.007077273912727833m`;
    - seed `29604`: useful `0.84375`, overshoot `0.0`, max XY
      `0.013731294311583042m`.
- Tiny PPO re-gate:
  - Same actor-preserved D298 contract except `tap_success_terminate` was not
    enabled.
  - PPO exited cleanly; `actor_preserve_blend=1.0` restored actor weights
    exactly (`max_post_restore_delta=0.000000000`).
  - Saved checkpoint:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/cube10cm_d299_directreset_actorfreeze_random_stop003_no_success_term_1it/model_0.pt`
    sha256 `753df107215e434a421da8eb029f2daf8c028c0f33ab4b4be55d945511e6d971`.
- TensorBoard gate:
  - Verdict: `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`.
  - Issues: none.
  - Warnings: missing `Train/mean_reward` and `Train/mean_episode_length` were
    allowed for the no-termination gate; raw TCP distance is still high for
    AABB diagnostics.
  - Collection metrics: contact/reaction `0.7676724195480347`, useful
    `0.7658405303955078`, success `0.7676724195480347`, overshoot
    `0.0018318966031074524`.
  - Displacement: max along/XY
    `0.001473818439990282/0.0016653359634801745m`, along/XY `>=1mm`
    `0.40420258045196533/0.4132543206214905`.
  - D256 reset active `1.0`, BC teacher blend `0.0`, joint cap `0.0`.
- Saved-checkpoint teacher-off direct-reset re-eval:
  - Seed `29801` passed: useful `0.96875`, overshoot `0.03125`, mean/max XY
    `0.0040031/0.0626363m`, XY `>=1mm` rate `0.5625`, joint cap max `0.0`.
  - Seed `29604` passed: useful `1.0`, overshoot `0.0`, mean/max XY
    `0.0011871/0.0040057m`, XY `>=1mm` rate `0.375`, joint cap max `0.0`.
- Interpretation:
  - D298's overshoot explosion was caused by success-termination episode
    recycling under collection, not by stochastic policy sampling alone.
  - Removing `tap_success_terminate` fixes the overshoot failure mode.
  - D299 is still not learned-policy success: actor was fully preserved, the
    no-termination TensorBoard gate lacks completed-episode reward scalars, and
    collection useful mean is improved but not `0.90+`.
- Decision:
  - Do not run long PPO.
  - Do not use `tap_success_terminate=True` for this actor-preserved tap10cm
    collection gate.
  - Do not claim RoArm readiness or final learned policy.
  - Next work should be a short controlled follow-up: decide whether to use a
    collection-average useful threshold around `0.65..0.75`, or add a final
    useful/success TensorBoard metric before another tiny no-success-terminate
    gate.
- D299 verdict:
  `D299_NO_SUCCESS_TERMINATE_COLLECTION_OVERSHOOT_FIX_WARN_NO_LEARNED_POLICY`.
- Primary D299 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d299_collection_contract_no_success_terminate.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/collection_contract_d299/`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/ppo_command_d299.txt`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/tensorboard_dashboard_command_d299.txt`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/cube10cm_d299_directreset_actorfreeze_random_stop003_no_success_term_1it/tensorboard_scalar_gate_d299.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/cube10cm_d299_directreset_actorfreeze_random_stop003_no_success_term_1it/teacher_off_direct_seed29801/teacher_off_policy_eval_summary_d299_direct_seed29801.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/cube10cm_d299_directreset_actorfreeze_random_stop003_no_success_term_1it/teacher_off_direct_seed29604/teacher_off_policy_eval_summary_d299_direct_seed29604.json`

## Previous Result: D298

- D298 purpose:
  - Execute the one explicitly approved tiny PPO + TensorBoard gate after D297.
  - Keep actor preservation on and BC teacher off.
  - Recheck the saved checkpoint using corrected direct-reset teacher-off eval.
  - No long PPO, PPO ladder, partial actor preservation, render, cleanup,
    RunPod/B200/SSH, Track A, VLA fine-tuning, or RoArm deployment was
    performed.
- Runtime:
  - PPO exited cleanly with `max_iterations=1`, `num_envs=32`,
    `num_steps_per_env=580`, D256 random reset active, AABB contact proxy,
    `tap_success_terminate`, `tap_stop_after_disp_m=0.003`,
    `bc_teacher_blend=0.0`, and `actor_preserve_blend=1.0`.
  - Saved checkpoint:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it/model_0.pt`
    sha256 `4dcbebbaaafbd50166cd40d2610b903e7209491a542fb8e041dac1cd4b1faf70`.
- TensorBoard gate result:
  - Verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`.
  - Train scalars exist: mean reward `10.783509254455566`, mean episode length
    `64.90697479248047`.
  - Collection metrics failed: contact/reaction `0.7029094696044922`, useful
    `0.04482758790254593`, success `0.0023168104235082865`, overshoot
    `0.7133082151412964`.
  - Displacement existed but was unsafe: max along/XY
    `0.01091606542468071/0.03478653356432915m`, along/XY `>=1mm` rates
    `0.3975215554237366/0.7559267282485962`.
  - D256 reset active `1.0`, BC teacher blend `0.0`, joint cap `0.0`,
    target lead `0.0`, stop-after-displacement hold rate
    `0.04251077398657799`.
- Saved-checkpoint teacher-off direct-reset re-eval:
  - Seed `29801` passed: useful `0.96875`, overshoot `0.03125`, mean/max XY
    `0.0040031/0.0626363m`, XY `>=1mm` rate `0.5625`, joint cap max `0.0`.
  - Seed `29604` passed: useful `1.0`, overshoot `0.0`, mean/max XY
    `0.0011871/0.0040057m`, XY `>=1mm` rate `0.375`, joint cap max `0.0`.
- Interpretation:
  - PPO plumbing, TensorBoard extraction, and saved-checkpoint output work.
  - The collection-time gate failed hard; the saved actor still passes the
    corrected direct-reset eval. This points to collection-time reset/
    termination/episode-recycle contract risk, not a clean learned-policy
    promotion.
- Decision:
  - Do not run long PPO.
  - Do not run another PPO gate immediately.
  - Do not claim learned-policy success or RoArm readiness.
  - Next work is non-PPO collection-time contract diagnostic comparing PPO
    collection versus teacher-off direct-reset, especially
    `tap_success_terminate`, stop-after-displacement hold timing, per-env
    overshoot traces, and reset/contact-cache behavior.
- D298 verdict:
  `D298_TINY_PPO_TENSORBOARD_COLLECTION_FAIL_TEACHER_OFF_DIRECT_PASS_NO_PROMOTION`.
- Primary D298 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d298_tiny_ppo_directreset_gate.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/ppo_command_d298.txt`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/tensorboard_dashboard_command_d298.txt`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it/tensorboard_scalar_gate_d298.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it/teacher_off_direct_seed29801/teacher_off_policy_eval_summary_d298_direct_seed29801.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it/teacher_off_direct_seed29604/teacher_off_policy_eval_summary_d298_direct_seed29604.json`

## Previous Result: D297

- D297 purpose:
  - Re-audit D296 random-reset overshoot before blaming the actor/policy.
  - Preserve episode/frame/action metadata for actor-vs-recorded and
    actor-vs-recovery diagnostics.
  - Check whether D296's random-reset failure was caused by the reset protocol.
  - No PPO training, long PPO, render, cleanup, RunPod/B200/SSH, Track A,
    VLA fine-tuning, or RoArm deployment was performed.
- Code diagnostic update:
  - `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py` now
    records per-env/per-step action traces, reset alignment, recorded actions,
    recovery actions, and episode/frame metadata.
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py` now supports
    `--d256_reset_warmup_mode` and defaults to `direct_reset`; old forced-reset
    modes are explicit (`force_step_zero`, `force_step_policy`).
- Critical finding:
  - D296's overshoot should not be treated as clean policy failure without
    qualification.
  - Exact D296 failing episodes under manual reset passed: useful `1.0`,
    overshoot `0.0`, max XY `0.0032778m`.
  - Env-hook random with forced second reset reproduced failure: useful
    `0.8125`, overshoot `0.15625`, max XY `0.0531604m`.
  - Manual 32-env replay of the same env-hook-selected episodes passed:
    useful `0.96875`, overshoot `0.0`, max XY `0.0167372m`.
  - Env-hook direct reset without the forced second reset passed: useful
    `1.0`, overshoot `0.0`, max XY `0.0034951m`.
  - Reset alignment audit showed reset pose, cube start, arm joint, target, and
    velocities were zero-error before action; step-0 overshoot was tied to the
    forced second reset path/contact cache behavior.
- Corrected teacher-off gate:
  - D295 saved actor, D256 random reset, BC teacher off, AABB contact proxy,
    `tap_stop_after_disp_m=0.003`, direct reset.
  - Seed `29603`: useful `1.0`, overshoot `0.0`, mean/max XY
    `0.0020230/0.0116959m`, XY `>=1mm` rate `0.53125`, XY `>=3mm` rate
    `0.4375`, joint cap max `0.0`.
  - Seed `29604`: useful `1.0`, overshoot `0.0`, mean/max XY
    `0.0011871/0.0040057m`, XY `>=1mm` rate `0.375`, XY `>=3mm` rate
    `0.3125`, joint cap max `0.0`.
- Decision:
  - Do not run long PPO.
  - Do not claim RoArm readiness or final learned policy.
  - D296's old random-reset failure is superseded as a reset-protocol artifact
    for teacher-off evaluation.
  - The next valid runtime is one explicitly approved tiny PPO + TensorBoard
    gate using the corrected direct-reset teacher-off contract, not a PPO
    ladder.
- D297 verdict:
  `D297_TEACHER_OFF_DIRECT_RESET_GATE_PASS_NO_PPO`.
- Primary D297 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d297_teacher_off_reset_protocol.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/teacher_off_direct_seed29603/teacher_off_policy_eval_summary_d297_direct_seed29603.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/teacher_off_direct_seed29604/teacher_off_policy_eval_summary_d297_direct_seed29604.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/random_envhook_direct_seed29604/closed_loop_recovery_summary_d297_random_envhook_direct_seed29604_actor_action_diagnostic.json`

## Previous Result: D296

- Supersession note:
  - D297 supersedes D296's next-work instruction for teacher-off eval. D296 is
    now a historical negative control showing why the reset protocol had to be
    audited, not the current blocker by itself.
- D296 purpose:
  - Diagnose D295 overshoot without running PPO.
  - Keep the D295 saved actor fixed and compare action projection/constraint
    options under teacher-off eval.
  - Check whether linspace reset success is robust to random D256 reset
    sampling.
  - No PPO training, long PPO, render, cleanup, RunPod/B200/SSH, Track A,
    VLA fine-tuning, or RoArm deployment was performed.
- Code diagnostic update:
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py` now supports
    `--tap_stop_after_disp_m`, `--tap_contact_slowdown_use_proxy`,
    `--exec_action_clip_abs`, and `--out_env_csv`.
- Linspace result:
  - raw D295 actor rerun failed: useful `0.8125`, overshoot `0.1875`.
  - `exec_clip_abs=0.5` passed linspace: useful `1.0`, overshoot `0.0`,
    mean/max XY `0.0036601/0.0191714m`, XY `>=1mm` rate `0.4375`.
  - `tap_stop_after_disp_m=0.001` passed linspace: useful `1.0`,
    overshoot `0.0`, mean/max XY `0.0008960/0.0032587m`, XY `>=1mm`
    rate `0.6875`.
  - `tap_stop_after_disp_m=0.003` passed linspace: useful `1.0`,
    overshoot `0.0`, mean/max XY `0.0021452/0.0038778m`, XY `>=1mm`
    rate `0.6875`.
  - useful-stop/zero-action and proxy slowdown removed overshoot but failed
    displacement rate: XY `>=1mm` rate only `0.03125`.
- Random D256 reset result:
  - every tested constraint failed random D256 reset seeds `29603/29604`;
  - `tap_stop_after_disp_m=0.003`: useful `0.75/0.75`,
    overshoot `0.25/0.25`;
  - `exec_clip_abs=0.5`: useful `0.75/0.625`,
    overshoot `0.25/0.3125`;
  - `tap_stop_after_disp_m=0.001`: useful `0.75/0.75`,
    overshoot `0.25/0.25`;
  - `exec_clip_abs=0.25`: useful `0.5/0.25`,
    overshoot `0.25/0.28125`;
  - `exec_clip_abs=0.5 + tap_stop_after_disp_m=0.001`: useful
    `0.75/0.6875`, overshoot `0.25/0.25`.
- Per-env failure audit:
  - `stop_disp003_random_seed29604_envtrace_d296` overshot in `8/32` envs;
  - overshoot D256 episode indices:
    `339, 154, 198, 668, 736, 656, 195, 606`;
  - all eight are original D256 `train_clean_positive` /
    `clean_useful_tap` episodes with camera contract pass;
  - original D256 max XY for those episodes was only
    `0.004109..0.008026m`;
  - actor rollout max XY became `0.02119..0.03467m`, and most failed rows had
    max along displacement `0.0`, indicating off-axis/lateral displacement.
- Interpretation:
  - D256 labels are not the immediate problem for these failures.
  - Linspace reset is too weak as a gate.
  - Magnitude-only constraints and simple displacement-stop constraints do not
    solve random reset closed-loop failure.
  - The current blocker is direction/control under random D256 reset states,
    not just action size or horizon.
- Decision:
  - Do not run long PPO.
  - Do not run another tiny PPO gate from D296 constraints.
  - Do not promote D295/D296 to learned-policy or RoArm readiness.
  - Next work is episode-index-preserving actor/action diagnostics and
    direction-aware actor/action contract repair before PPO.
- D296 verdict:
  `D296_ACTION_CONSTRAINT_LINSPACE_PASS_RANDOM_FAIL_NO_PPO`.
- Primary D296 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d296_overshoot_control_diagnostic.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296/run_overshoot_control_matrix_d296.sh`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296/run_candidate_random_checks_d296.sh`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296/run_conservative_random_checks_d296.sh`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296/stop_disp003_random_seed29604_envtrace_d296/teacher_off_policy_eval_envs_stop_disp003_random_seed29604_envtrace_d296.csv`

## Previous Runtime Result: D295

- D295 purpose:
  - Run the explicitly approved constrained short PPO gate from the D294
    max/mean/rate displacement contract.
  - Keep the D290 replay-batch actor as the actor prior, but preserve it fully
    during PPO collection with `actor_preserve_blend=1.0`.
  - Use a longer collection horizon than D292 so displacement can be measured
    instead of hidden by a 24-step tiny smoke.
  - No long PPO, render, cleanup, RunPod/B200/SSH, Track A, VLA fine-tuning, or
    RoArm deployment was performed.
- Runtime contract:
  - checkpoint:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
  - `max_iterations=1`, `num_envs=32`, `num_steps_per_env=580`,
    `init_noise_std=0.005`, `actor_preserve_blend=1.0`;
  - D256 env reset hook active, frame `0`, `d256_reset_sample_mode=linspace`;
  - `bc_teacher_checkpoint_path=NONE`, `bc_teacher_blend=0.0`,
    `bc_teacher_imitation_reward_scale=0.0`;
  - contact proxy `link5_collision_aabb`;
  - action contract: `action_scale=0.04`,
    `max_joint_delta_per_step_rad=0.04`,
    `joint_target_lead_limit_rad=0.06`,
    `joint_delta_reference=joint_pos`.
- PPO runtime result:
  - exit code `0`;
  - actor preservation restored the actor exactly:
    `max_pre_restore_delta=0.270150483`,
    `max_post_restore_delta=0.000000000`;
  - saved `model_0.pt` sha256:
    `d3073e7446652d6a7c7c6a160c336bfa7cdf8bf04ef988010adf6bd79b322b0a`.
- TensorBoard scalar gate:
  - verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`;
  - hard issues:
    - missing core TensorBoard scalars `Train/mean_reward` and
      `Train/mean_episode_length`;
    - contact/reaction/useful below the D293/D294 `0.90` promotion threshold:
      contact/reaction/success `0.8786637783`, useful `0.8710668087`.
  - key collection metrics:
    - overshoot `0.0075969826`;
    - max displacement along/XY `0.0025365781m / 0.0026646142m`;
    - along/XY `>=1mm` rate `0.3124461174 / 0.3125`;
    - XY `>=3mm` rate `0.2603987157`;
    - D256 reset active `1.0`, BC teacher blend `0.0`, joint cap `0.0`.
- Saved-checkpoint teacher-off eval under the D295 action contract:
  - verdict: `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`;
  - contact/reaction/success `1.0`, useful `0.8125`;
  - overshoot `0.1875`, which fails the `<=0.05` overshoot gate;
  - max displacement along/XY `0.0590043068m / 0.0590082631m`;
  - mean displacement along/XY `0.0074102012m / 0.0106040258m`;
  - along/XY `>=1mm` rate `0.53125 / 0.6875`;
  - joint cap max trace `0.0`, policy action abs mean/max
    `0.3058707444 / 1.0`.
- Interpretation:
  - D295 resolves the D292 tiny-displacement ambiguity: with sufficient horizon,
    the actor can move the cube by meaningful distances.
  - D295 does not prove a learned policy. The actor was fully preserved during
    PPO, and the saved checkpoint still fails teacher-off due to overshoot.
  - The current blocker is not "no displacement"; it is uncontrolled
    displacement and useful consistency under the raw actor/action contract.
- Decision:
  - Do not run long PPO.
  - Do not claim learned-policy success or RoArm readiness.
  - Do not promote to partial actor preservation or real PPO actor updates yet.
  - Next work is a non-PPO overshoot-control diagnostic comparing raw actor eval
    against explicit action projection/constraint options such as
    `tap_stop_after_disp_m`, useful-stop/zero-action safety, proxy contact
    slowdown, and action clipping/projection.
- D295 verdict:
  `D295_RATE_GATED_SHORT_PPO_COLLECTION_PARTIAL_TEACHER_OFF_FAIL_NO_PROMOTION`.
- Primary D295 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_d295_rate_gate_runtime.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/ppo_command_d295.txt`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/tensorboard_dashboard_command_d295.txt`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/tensorboard_scalar_gate_d295.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/teacher_off_eval_model0_d295_contract/teacher_off_policy_eval_summary_d295_model0.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`

## Previous Gate Result: D294

- D294 purpose:
  - Make the D293 displacement contract robust against max-only false positives.
  - Add distribution-level displacement rate metrics before the next PPO runtime.
  - No Isaac Lab runtime, PPO, render, cleanup, RunPod/B200/SSH, Track A, VLA
    fine-tuning, or RoArm deployment was performed.
- Critical reason:
  - Max displacement can be high because one env moved while most envs stayed
    near zero.
  - Mean displacement and `>=1mm` rate must be checked together with max
    displacement.
- D256 train-clean positive rate check:
  - rows: `737`;
  - `max_tap_disp_xy_m >= 0.001`: `733/737`, rate `0.994572592`;
  - `max_tap_disp_xy_m >= 0.003`: `727/737`, rate `0.986431479`;
  - `max_tap_disp_along_m >= 0.001`: `729/737`, rate `0.989145183`;
  - `max_tap_disp_along_m >= 0.003`: `723/737`, rate `0.981004071`.
- Code guardrail added:
  - `roarm_rl/roarm_cube_push_env.py` now logs
    `cube_tap_max_disp_along_ge_1mm_rate`,
    `cube_tap_max_disp_xy_ge_1mm_rate`,
    `cube_tap_max_disp_along_ge_3mm_rate`, and
    `cube_tap_max_disp_xy_ge_3mm_rate`.
  - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py` now supports
    `--min_tap_disp_along_ge_1mm_rate` and
    `--min_tap_disp_xy_ge_1mm_rate`.
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py` now records
    along/XY `>=1mm` and `>=3mm` rates and supports
    `--min_disp_along_ge_1mm_rate` and `--min_disp_xy_ge_1mm_rate`.
- Next gate recommendation:
  - Do not pass from max displacement alone.
  - TensorBoard should use `--require_tap_displacement_gate`,
    `--min_tap_max_disp_along_m 0.001`, and an initial conservative
    `--min_tap_disp_xy_ge_1mm_rate 0.25`.
  - Teacher-off eval should use mean/max displacement thresholds plus initial
    conservative `--min_disp_xy_ge_1mm_rate 0.25`.
  - `0.25` is a first short-gate threshold only. D256 clean data supports much
    higher rates, so later gates should move toward `0.90+` after runtime
    stability is proven.
- Actor-preservation caveat:
  - Full/heavy actor preservation is a plumbing/safety gate, not a learned-policy
    claim.
  - Only after preserved-actor max/mean/rate gates pass should partial
    preservation or real PPO actor-update gates be discussed.
- D294 verdict:
  `D294_DISPLACEMENT_RATE_GATE_ADDED_NO_RUNTIME_NO_LONG_PPO`.
- Primary D294 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_displacement_rate_gate_d294.md`
  - `roarm_rl/roarm_cube_push_env.py`
  - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`

## Previous Contract Result: D293

- D293 purpose:
  - Cross-check the D292 next-step conclusion against the broader
    mining/excavation automation framing.
  - Turn the D292 displacement warning into an explicit next-gate contract.
  - Prevent accidental long PPO promotion from contact/reaction-only results.
  - No Isaac Lab runtime, PPO, render, cleanup, RunPod/B200/SSH, Track A, VLA
    fine-tuning, or RoArm deployment was performed.
- Cross-check result:
  - The previous conclusion remains correct: next work is not long PPO.
  - D292 contact/reaction and saved-checkpoint teacher-off eval are meaningful
    plumbing evidence, but displacement is too small for a policy or automation
    claim.
  - The 10cm cube task should be treated as a tool-object interaction primitive:
    contact, reaction, controlled displacement, no overshoot, and visual
    trajectory capture.
- Hard displacement tiers:
  - Tier 0: contact/reaction only, no overshoot;
  - Tier 1: at least `0.001m` displacement;
  - Tier 2: at least `0.003m` stable displacement;
  - Tier 3: `0.005..0.010m` strong push tier;
  - Fail: `>=0.020m` overshoot.
- Next gate thresholds:
  - useful/contact/reaction `>=0.90`;
  - overshoot `<=0.05`;
  - D256 reset active `1.0`;
  - BC teacher blend `0.0`;
  - joint delta cap below existing ceiling `<=0.25`;
  - TensorBoard tap max displacement along `>=0.001m`;
  - teacher-off mean max displacement along or XY target `>=0.0005m`,
    preferred `>=0.001m`;
  - teacher-off max displacement along or XY `>=0.001m`.
- Code guardrail added:
  - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py` now supports
    `--require_tap_displacement_gate`, which turns small tap displacement into
    an issue instead of a warning.
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py` now supports
    `--min_mean_disp_along_m`, `--min_max_disp_along_m`,
    `--min_mean_disp_xy_m`, and `--min_max_disp_xy_m`.
  - D292 posthoc regate with `--require_tap_displacement_gate` and
    `--min_tap_max_disp_along_m 0.001` now fails as intended:
    `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`.
- Physical spec decision:
  - The current env still uses a 10cm, `0.720kg`, friction `1.5/1.2`,
    restitution `0.0` rigid cube.
  - Treat `0.720kg` as a coherent density-preserving hard/stress tier, not as
    the only real-world nominal if the physical proxy object is lighter.
  - Future sim2real work should measure the real proxy mass and then add mass
    robustness, but mass randomization is not required before the next short PPO
    gate.
- Decision:
  - Do not run long PPO.
  - Do not claim learned-policy success or RoArm readiness.
  - Do not promote D292 from contact/reaction alone.
  - Next runtime, only after explicit approval, is a constrained short PPO gate
    with actor preservation, D256 reset active, BC teacher blend off,
    `link5_collision_aabb`, TensorBoard displacement hard gate, and
    saved-checkpoint teacher-off displacement gate.
- D293 verdict:
  `D293_DISPLACEMENT_HORIZON_CONTRACT_SET_NO_LONG_PPO`.
- Primary D293 artifacts:
  - `claudedocs/session_20260629_cube10cm_top_view_displacement_horizon_contract_d293.md`
  - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
  - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`

## Previous Runtime Result: D292

- D292 purpose:
  - Run the explicitly approved tiny PPO smoke after D291 showed the D290
    same-process reset-bin failure was likely a reused-env diagnostic artifact.
  - Keep the D290 replay-batch actor as the data prior, but do not use the D257
    MLP teacher blend.
  - Check PPO plumbing, TensorBoard scalars, saved checkpoint integrity, and a
    teacher-off frozen eval before any longer PPO.
- PPO smoke configuration:
  - checkpoint:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
  - `max_iterations=1`, `num_envs=32`, `num_steps_per_env=24`,
    `init_noise_std=0.005`, `actor_preserve_blend=1.0`;
  - D256 env reset hook active, `d256_reset_sample_mode=linspace`;
  - `bc_teacher_checkpoint_path=NONE`, `bc_teacher_blend=0.0`,
    `bc_teacher_imitation_reward_scale=0.0`.
- PPO runtime result:
  - exit code `0`;
  - actor preservation restored the actor exactly:
    `max_post_restore_delta=0.000000000`;
  - saved `model_0.pt` sha256:
    `d56065796c2549bfc70c7d2200314118b924580e1d38f19a8265ee2c8aebf271`.
- TensorBoard scalar gate:
  - verdict: `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`;
  - hard issues: `[]`;
  - warnings:
    - raw TCP-cube distance is high for tap/AABB diagnostics:
      `0.09063738584518433`;
    - tap displacement remains tiny:
      `cube_tap_max_disp_along_m=1.3096122529532295e-05`.
  - useful/success/overshoot during the 24-step PPO collection:
    `0.0768229216337204 / 0.0729166716337204 / 0.01302083395421505`;
  - joint delta cap rate and target lead limit were safe:
    `0.0008680556202307343 / 0.0`.
- Saved-checkpoint teacher-off eval:
  - verdict: `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`;
  - useful/success/overshoot: `0.96875 / 0.96875 / 0.0`;
  - D256 reset active and BC teacher blend: `1.0 / 0.0`;
  - joint delta cap max trace: `0.015625`;
  - max displacement along/XY:
    `0.0031909942626953125m / 0.00327563239261508m`;
  - mean displacement along/XY is still small:
    `0.0001180088147521019m / 0.0001241182180820033m`.
- Interpretation:
  - D292 proves the tiny PPO runtime path, actor preservation, TensorBoard
    scalar extraction, and saved-checkpoint teacher-off eval are wired.
  - It does not prove a learned policy. With `actor_preserve_blend=1.0`, the
    actor is intentionally restored after the PPO update.
  - The TensorBoard displacement warning is real. The 24-step collection horizon
    and tap-useful termination make displacement hard to read, but the small
    teacher-off mean displacement means the next gate must separate
    contact/reaction from meaningful push distance.
- Decision:
  - Do not run long PPO.
  - Do not claim learned-policy success or RoArm readiness.
  - Do not go back to the D257 MLP teacher path for this PPO gate.
  - Next valid runtime candidate, only after explicit approval, is a constrained
    short PPO gate that keeps actor preservation and adds a clearer displacement
    or horizon contract.
- D292 verdict:
  `D292_TINY_PPO_ACTORFREEZE_TENSORBOARD_WARN_TEACHER_OFF_PASS_NO_LONG_PPO`.
- Primary D292 artifacts:
  - `claudedocs/session_20260628_cube10cm_top_view_tiny_ppo_freshgate_d292.md`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d292/tap10cm/ppo_replay_actor_freshgate_actorfreeze_1it/cube10cm_d292_replay_actor_freshgate_actorfreeze_1it/tensorboard_scalar_gate_d292.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d292/tap10cm/ppo_replay_actor_freshgate_actorfreeze_1it/cube10cm_d292_replay_actor_freshgate_actorfreeze_1it/teacher_off_eval_model0/teacher_off_policy_eval_summary_d292_model0.json`
  - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d292/tap10cm/ppo_replay_actor_freshgate_actorfreeze_1it/cube10cm_d292_replay_actor_freshgate_actorfreeze_1it/model_0.pt`

## Previous Result: D286

- D286 purpose:
  - Diagnose the D285 collection failure before any more PPO.
  - Bin D256 frame-0 reset rows by episode index and measure frozen actor
    action magnitude, joint-cap pressure, useful/contact signal, and overshoot.
  - This was diagnostic only: no PPO training, no long PPO, no render, no
    cleanup, no RunPod/B200/SSH, and no RoArm deployment.
- Code changes:
  - Added opt-in D256 reset episode filters:
    `d256_reset_episode_min` and `d256_reset_episode_max`.
  - Exposed the same filters through `roarm_rl/train_cube_push_ppo.py`.
  - Added diagnostic script:
    `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`.
- D286 default action-scale result:
  - Actor checkpoint:
    `actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/model_9.pt`.
  - Run class:
    `bin_count=5`, `num_envs=32`, `eval_steps=580`,
    `action_noise_std=0.02`, default `action_scale=0.04`.
  - Corrected step-trace max metrics:
    cap max by bin `0.6302083730697632 / 0.7604166865348816 /
    0.8229166865348816 / 0.703125 / 0.78125`;
    useful max `0.0` across all bins.
  - Interpretation: this is not just one bad reset bin. The D285 actor produces
    saturated actions and no useful signal across the D256 episode-index bins.
- D286 action-scale check:
  - `action_scale=0.01` lowers cap max by bin to
    `0.010416666977107525 / 0.015625 / 0.0052083334885537624 /
    0.0781250074505806 / 0.0833333358168602`.
  - But useful max remains `0.0` across all bins.
  - Interpretation: action-scale reduction alone is not a valid fix. It removes
    cap pressure but also fails to produce useful/contact behavior.
- D286 decision:
  - Do not run long PPO.
  - Do not run another tiny PPO smoke from D285 until the actor/teacher bridge
    or action projection is repaired.
  - Reset-bin filtering alone is not enough.
  - Action-scale reduction alone is not enough.
  - Next work should repair the actor/teacher bridge or add explicit action
    projection/constraint, then rerun teacher-off/bin diagnostics before the
    next TensorBoard-gated PPO smoke.
- D286 verdict:
  `D286_NO_RESET_BIN_OR_ACTION_SCALE_FIX_READY_FOR_PPO`.
- Primary D286 comparison artifact:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_actor_probe_d286_comparison/tap10cm/d256_reset_bin_actor_probe_comparison_d286.md`.

## Previous Result: D285

- D283-D285 purpose:
  - Test whether short PPO can be promoted after adding actor-preservation.
  - Keep all runs short: `10` PPO iterations, `num_envs=32`,
    `num_steps_per_env=24`, D256 frame-0 reset, AABB contact proxy,
    D280 actor warm-start, BC teacher as imitation metric/reward sidecar with
    `bc_teacher_blend=0.0`, and no long PPO.
- D283 actor-preserve095, noise `0.1`:
  - Command class:
    `--actor_preserve_blend 0.95`, `--init_noise_std 0.1`.
  - TensorBoard gate failed:
    `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`;
    joint-cap max `0.6579861640930176`;
    useful last `0.03125`;
    reward `-6.631277561187744 -> 5.1462788581848145`;
    BC MSE last `0.0750335305929184`.
  - Saved `model_9.pt` corrected teacher-off eval passed:
    useful/overshoot/joint-cap `0.71875/0.0/0.2135416716337204`;
    policy action abs max trace `3.6983871459960938`.
  - Saved `model_9.pt` corrected actor-vs-teacher trace passed:
    MSE/cosine `0.05202638357877731/0.6651849150657654`;
    useful/overshoot `0.71875/0.0`.
  - Interpretation: actor-preservation keeps the deterministic saved actor
    usable, but collection-time TensorBoard still blocks promotion.
- D284 actor-preserve095, lower noise `0.02`:
  - Command class:
    `--actor_preserve_blend 0.95`, `--init_noise_std 0.02`.
  - TensorBoard gate still failed:
    joint-cap max `0.6430121660232544`;
    useful last `0.03125`;
    reward `-6.0151824951171875 -> 5.879624843597412`;
    BC MSE last `0.07686792314052582`;
    action abs max last `0.8672488927841187`.
  - Interpretation: lowering exploration noise is not enough.
- D285 actor-freeze, lower noise `0.02`:
  - Command class:
    `--actor_preserve_blend 1.0`, `--init_noise_std 0.02`.
  - Runtime logged exact actor restoration after each update:
    `max_post_restore_delta=0.000000000`.
  - TensorBoard gate still failed:
    joint-cap max `0.6536458730697632`;
    useful last `0.03125`;
    reward `-6.012892246246338 -> 5.879622936248779`;
    BC MSE last `0.06847621500492096`;
    action abs max last `0.8579933643341064`.
  - Interpretation: even a frozen actor fails collection on later D256 reset
    sample regions. The blocker is now reset-sample/state-distribution and
    action-cap pressure, not simply actor update drift.
- D285 decision:
  - Do not run long PPO.
  - Do not promote from D283, D284, or D285.
  - Do not interpret rising reward as policy progress.
  - Next work should be a non-long diagnostic/fix:
    compare D256 reset episode-index bins and actor action/cap statistics,
    restrict or stratify reset samples to proven useful/contact-safe bins, or
    add an explicit action-cap/teacher-KL constraint before another short PPO
    gate.

## Previous Result: D282

- D282 code change:
  - Added PPO internal actor-preservation to `roarm_rl/train_cube_push_ppo.py`
    through `--actor_preserve_blend`.
  - The hook snapshots actor-related checkpoint keys after warm-start/noise
    override and restores a blend after every PPO update:
    `cur = (1 - blend) * cur + blend * ref`.
  - Preserved keys include `actor.*`, `actor_obs_normalizer.*`, `std`, and
    `log_std` if present.
  - Static checks passed:
    `python -m py_compile roarm_rl/train_cube_push_ppo.py` and
    `git diff --check -- roarm_rl/train_cube_push_ppo.py`.
- D282 corrected D281 evaluation protocol:
  - `tap_useful_terminate` is valid for training-runtime diagnostics, but it
    should not be used for saved-checkpoint frozen eval/trace because successful
    episodes can reset and disappear from the final summary.
  - Corrected eval/trace contract:
    `tap10cm + link5_collision_aabb + D256 frame-0 reset +
    episode_length_s=6.0 + tap_stop_after_useful_seen +
    vertical_gate_mode=min_contact`, without `tap_useful_terminate`.
  - Re-evaluated the D281 conservative checkpoint under the corrected contract:
    teacher-off useful/overshoot/joint-cap
    `0.8125/0.0/0.1666666716337204`;
    actor-vs-teacher MSE/cosine
    `0.04292111471295357/0.6536584496498108`;
    trace useful/overshoot `0.8125/0.0`.
  - Therefore the D281 "all PPO update unsafe" wording is too strong. The
    stricter current truth is: 1-iteration conservative PPO can pass corrected
    saved-checkpoint gates, but promotion still requires TensorBoard and
    post-checkpoint checks.
- D282 one-iteration actor-preservation smokes:
  - Actor-freeze (`--actor_preserve_blend 1.0`) logged
    `actor_preserve_after_update blend=1.000000 keys=13
    max_pre_restore_delta=0.016347766 max_post_restore_delta=0.000000000`.
  - Corrected teacher-off eval passed:
    useful/overshoot/joint-cap `0.71875/0.0/0.2135416716337204`.
  - Corrected trace passed:
    MSE/cosine `0.05346343293786049/0.6641471982002258`,
    useful/overshoot `0.71875/0.0`.
  - Actor-preserve095 (`--actor_preserve_blend 0.95`) logged
    `actor_preserve_after_update blend=0.950000 keys=13
    max_pre_restore_delta=0.016167104 max_post_restore_delta=0.000808358`.
  - Corrected teacher-off eval passed:
    useful/overshoot/joint-cap `0.71875/0.0/0.21875`.
  - Corrected trace passed:
    MSE/cosine `0.05328662693500519/0.6633936166763306`,
    useful/overshoot `0.71875/0.0`.
- D282 no-preservation 10-iteration PPO check:
  - Ran a 10-iteration conservative PPO smoke without actor preservation.
  - TensorBoard gate failed:
    verdict `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`;
    issue `joint-delta cap rate too high: max=0.6664496660232544`.
  - Reward rose from `-6.725722312927246` to `5.878491401672363`, but task
    behavior degraded:
    useful last `0.03125`;
    joint cap max `0.6664496660232544`;
    action abs max last `0.9052953720092773`;
    BC imitation MSE last `0.10183489322662354`;
    raw TCP diagnostic last `0.5229541063308716`.
  - Saved `model_9.pt` corrected teacher-off eval failed:
    verdict `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`;
    useful/overshoot/joint-cap `0.65625/0.03125/0.2760416567325592`;
    policy action abs max trace `2.2301270961761475`.
  - Saved `model_9.pt` corrected actor-vs-teacher trace also blocked
    promotion:
    MSE/cosine `0.05501702427864075/0.6031392812728882`;
    useful/overshoot `0.65625/0.03125`;
    joint cap max `0.2760416567325592`.
- D282 decision:
  - Do not run long PPO.
  - Reward alone is not a promotion gate; rising reward can coexist with action
    saturation, joint cap hits, low useful rate, and teacher-prior drift.
  - Next valid runtime is not a long PPO ladder. It is a short PPO variant with
    actor preservation enabled, likely `--actor_preserve_blend 0.95`, followed
    by the same three gates: TensorBoard scalar gate, corrected teacher-off
    frozen eval, and corrected actor-vs-teacher trace.
  - No learned-policy or RoArm-readiness claim exists.

## Previous Result: D280

- D280 actor/teacher bridge work:
  - Added `sim_scripts/cube10cm_top_view_distill_actor_from_teacher.py`.
  - Added PPO warm-start support to `roarm_rl/train_cube_push_ppo.py` via
    `--warm_start_checkpoint_path`.
  - Added tap runtime flags `--tap_success_terminate` and
    `--tap_overshoot_terminate` to the PPO entrypoint.
  - Added diagnostic teacher-off eval knobs:
    `--zero_actions_after_useful_seen`, `--vertical_gate_mode`,
    `--action_scale`, and `--max_joint_delta_per_step_rad`.
- D280 supervised actor distillation:
  - Source actor:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/model_0.pt`.
  - Distilled checkpoint:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/model_actor_distill_d280.pt`.
  - Checkpoint sha256:
    `4c12862320883ebaab14c97043999e235224a5d892916d6a23f16189358639dd`.
  - Samples train/val: `16704` / `1856`.
  - Offline initial val MSE/MAE/cosine:
    `0.38865897059440613` / `0.5184221863746643` /
    `0.32961708307266235`.
  - Offline final val MSE/MAE/cosine:
    `0.01078740879893303` / `0.0625312477350235` /
    `0.9815400838851929`.
  - Verdict:
    `D280_ACTOR_DISTILL_SUPERVISED_FIT_WARN_NEEDS_ROLLOUT_EVAL`, because the
    teacher rollout used for collection still had overshoot `0.21875`.
- D280 rollout checks:
  - D279-style trace of the distilled actor improved actor/teacher alignment:
    MSE/cosine `0.0765833854675293` / `0.8944697976112366`.
  - The same trace still blocked promotion:
    useful `0.59375`, overshoot `0.125`, vertical max
    `0.22511835396289825`, joint cap max `0.7604166865348816`.
  - Default teacher-off frozen eval also failed:
    useful `0.59375`, overshoot `0.125`, joint cap max
    `0.7604166865348816`, last-frame vertical max
    `0.15143540501594543`.
  - `action_scale=0.020` and `action_scale=0.010` probes did not fix the
    behavior; useful fell to `0.5625` / `0.46875`, and overshoot stayed
    `0.125` / rose to `0.15625`.
- D280 stop-after-useful diagnostic:
  - With `--zero_actions_after_useful_seen` and
    `--vertical_gate_mode=min_contact`, teacher-off eval passed:
    verdict `TEACHER_OFF_FROZEN_EVAL_PASS_FOR_NEXT_SHORT_PPO_GATE`;
    useful `0.71875`;
    overshoot `0.0`;
    joint cap max `0.2135416716337204`;
    contact-time vertical gate value `0.0`.
  - Interpretation: the main remaining design issue is post-useful stop or
    termination semantics plus final-frame vertical gate strictness. This is not
    a learned-policy success claim.
- D280 warm-start PPO smoke:
  - Ran exactly one 1-iteration PPO smoke with the D280 actor warm-start,
    D256 reset/AABB, `bc_teacher_blend=0.0`, BC imitation reward scale `0.05`,
    `tap_success_terminate=True`, `tap_overshoot_terminate=True`,
    `--no_init_at_random_ep_len`, `num_envs=32`, and
    `num_steps_per_env=24`.
  - PPO runtime wrote TensorBoard events and `model_0.pt` at:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/ppo_warmstart_smoke/cube10cm_d280_warmstart_success_terminate_smoke/`.
  - TensorBoard showed wiring active:
    D256 reset active `1.0`;
    BC teacher blend `0.0`;
    BC imitation MSE logged `0.46044886112213135`;
    BC teacher action abs mean logged `0.1677628606557846`;
    useful `0.12109375`;
    overshoot `0.0221354179084301`.
  - TensorBoard gate verdict:
    `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`;
    issue `joint-delta cap rate too high: max=0.3993055820465088`;
    warnings: one reward point, raw TCP diagnostic high, and tap max
    displacement small.
  - D279-style trace after this single PPO update worsened:
    MSE/cosine `0.086099773645401` / `0.8869514465332031`;
    useful `0.5`;
    overshoot `0.1875`;
    joint cap max `0.78125`.
- D280 current decision:
  - Runtime wiring is now capable of using a warm-start actor and logging
    TensorBoard/BC/D256 metrics.
  - Promotion still fails. Do not run long PPO or a short PPO ladder.
  - Do not claim learned policy, teacher-off success, or RoArm readiness.
  - Next work should encode stop-after-useful semantics properly in the env
    contract or termination/reward design, then rerun teacher-off eval,
    actor-vs-teacher trace, and only then another tiny PPO/TensorBoard gate.

## Previous Result: D278

- D278 teacher-off frozen eval:
  - Added `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`.
  - Evaluated the D277 PPO actor checkpoint:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/model_0.pt`.
  - Runtime contract:
    - `RoArm-CubeTap10cm-Direct-v0`;
    - `tap_contact_proxy_mode=link5_collision_aabb`;
    - D256 reset hook from `ppo_actor_prior_teacher_rows_d256.csv`;
    - `d256_reset_frame_index=0`;
    - `d256_reset_sample_mode=linspace`;
    - fixed +x;
    - `episode_length_s=6.0`;
    - `eval_steps=580`;
    - `num_envs=32`;
    - `bc_teacher_blend=0.0`;
    - `bc_teacher_imitation_reward_scale=0.0`;
    - no BC teacher checkpoint loaded.
  - Output:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_off_policy_eval_d278/tap10cm/teacher_off_policy_eval_summary_d278.json`.
  - Verdict:
    `TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`.
- D278 key metrics:
  - D256 reset active rate `1.0`;
  - BC teacher blend last `0.0`;
  - contact/reaction/useful seen:
    `0.875` / `0.875` / `0.5625`;
  - success flag `0.875`, but this is not sufficient because overshoot happens
    after or around the useful event;
  - overshoot seen `0.3125`;
  - max displacement along mean/max:
    `0.0024283849634230137` / `0.018782615661621094`;
  - max displacement xy mean/max:
    `0.020250540226697922` / `0.10077980160713196`;
  - min contact vertical offset mean/min/max:
    `0.0` / `0.0` / `0.0`;
  - last contact vertical offset mean/max:
    `0.02129734866321087` / `0.24940747022628784`;
  - raw TCP-threshold contact seen `0.0`;
  - joint-delta cap rate last/max trace:
    `0.1145833432674408` / `0.15625`;
  - raw policy action abs mean/max trace:
    `0.1184795308344323` / `1.3003933429718018`;
  - reward/obs/action finite all `True`.
- Critical interpretation:
  - D278 is stricter than D277 in the right way: it removes the BC teacher
    action blend and tests the frozen actor.
  - The actor is not completely inert, because it produces contact/reaction in
    many D256-reset states.
  - It is not controlled enough for a policy claim: overshoot `0.3125` is far
    above the `0.05` gate, and the last vertical offset max `0.249m` means some
    rollouts leave the intended tool-surface contact geometry.
  - Therefore D277's teacher-on success was still teacher-prior behavior, not
    learned-policy behavior.
- Post-run:
  - `python -m py_compile` passed for the teacher-off eval script.
  - `git diff --check` passed.
  - `ps -C python -C python3` showed no active local Python process.
  - `nvidia-smi` returned to the observed baseline of about `2509MiB` used.
- Verdict:
  `D278_TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`.
- Current next work:
  - Do not run long PPO.
  - Do not run a short PPO ladder from D278.
  - Do not claim learned policy, teacher-off success, or RoArm readiness.
  - Next concrete work is diagnostic, not scale: compare teacher-on D277/D274
    versus teacher-off D278 action traces, especially overshoot cases,
    vertical-offset outliers, policy action saturation above clip, and whether
    the actor learned the teacher direction or merely relied on
    `bc_teacher_blend=1.0`.

## Previous Result: D277

- D273-D274 D256 reset/pose alignment:
  - Added an explicit opt-in PPO/env reset hook:
    - `d256_reset_csv_path`;
    - `d256_reset_frame_index`;
    - `d256_reset_sample_mode`.
  - The hook samples D256 train-clean frame-0 rows and sets the arm joints,
    gripper, cube pose, target pose, and push direction directly at env reset.
  - Default behavior is unchanged when `d256_reset_csv_path` is empty.
  - D274 teacher-only from `env_d256_initial` on `tap10cm`,
    `link5_collision_aabb`, fixed +x, and `env_target`:
    - reset hook active rate `1.0`;
    - initial feature outside train min/max `0.0`;
    - AABB contact/useful/reaction `0.71875` / `0.71875` / `0.71875`;
    - tap overshoot seen `0.03125`;
    - min contact vertical offset mean/min/max `0.0` / `0.0` / `0.0`;
    - max displacement along mean/max
      `0.0014097457751631737` / `0.01252603530883789`;
    - raw delta clip exceed `0.22213362068965517`;
    - action cap rate `0.14152298850574713`.
- D275-D277 tiny PPO data-prior smokes:
  - D275 used D256 reset plus D257 teacher prior but kept rsl_rl's default
    `init_at_random_ep_len=True`. Gate failed because tap overshoot seen rose
    to `0.125`. This shows random episode offset is unsafe for frame-0 reset
    plus `bc_teacher_phase_timing=direct_steps`.
  - D276 added `--no_init_at_random_ep_len`. Overshoot stayed `0.0`, D256 reset
    active and BC teacher blend were both `1.0`, but the 24-step smoke was too
    short to complete an episode and did not emit `Train/mean_reward`.
  - D277 kept `--no_init_at_random_ep_len` and ran one episode-complete tiny
    smoke (`num_envs=32`, `max_iterations=1`, `num_steps_per_env=600`;
    `19,200` total timesteps). This is a TensorBoard/reward visibility smoke,
    not long PPO.
  - D277 TensorBoard gate:
    - verdict `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`;
    - issues `none`;
    - warnings:
      - short run: `Train/mean_reward` has `1` point, promotion gate expects
        at least `10`;
      - raw TCP-cube distance high for tap/AABB diagnostic:
        `0.20408329367637634`.
  - D277 selected metrics:
    - `Train/mean_reward`: `-3957.08154296875`;
    - `Train/mean_episode_length`: `599.0`;
    - `cube_push_d256_reset_active_rate`: `1.0`;
    - `cube_tap_d256_reset_active_rate`: `1.0`;
    - `cube_push_bc_teacher_blend_mean`: `1.0`;
    - `cube_tap_bc_teacher_blend_mean`: `1.0`;
    - `cube_push_bc_teacher_imitation_mse`: `0.66529381275177`;
    - `cube_tap_contact_seen_rate`: `0.6662499904632568`;
    - `cube_tap_useful_seen_rate`: `0.6469791531562805`;
    - `cube_tap_success_rate`: `0.6652604341506958`;
    - `cube_tap_overshoot_seen_rate`: `0.019687499850988388`;
    - `cube_tap_max_disp_along_m`: `0.0018036302644759417`;
    - `cube_tap_contact_vertical_offset_m`: `0.015306632034480572`;
    - `cube_push_joint_delta_cap_rate`: `0.15915799140930176`.
- Code changes in D273-D277:
  - `roarm_rl/roarm_cube_push_env.py` now supports explicit D256 reset
    sampling and logs D256 reset activity to TensorBoard extras.
  - `roarm_rl/train_cube_push_ppo.py` exposes D256 reset flags and
    `--no_init_at_random_ep_len`.
  - `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py` can test the
    env-side D256 reset hook and reports AABB/useful/reaction/vertical/
    overshoot metrics.
  - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py` now supports
    `--expect_d256_reset`.
- Critical interpretation:
  - The D272 default-reset failure cause is now concrete: the teacher was being
    asked to act from states outside the D256 train-clean pose distribution.
  - Matching only TCP is not enough; matching the D256 joint/reset distribution
    is necessary. D256 frame-0 wrist roll and joint posture matter because the
    D257 teacher is a supervised data-prior model, not a robust global policy.
  - D277 is not a learned-policy result. The policy is still teacher-on, with
    `bc_teacher_blend=1.0`, and only one episode-level reward point exists.
  - The gate is intentionally conservative but not too strict for promotion:
    AABB/contact/useful/overshoot/action-cap passed issue checks, while the
    short-run warning correctly blocks promotion to long PPO.
- Post-run:
  - `python -m py_compile` passed for the edited env, PPO entrypoint, teacher
    probe, and TensorBoard gate.
  - `git diff --check` passed.
  - `ps -C python -C python3` showed no active local Python process.
  - `nvidia-smi` showed the observed baseline class state:
    `2509MiB` used, `13436MiB` free on the RTX 4090 Laptop GPU.
- Verdict:
  `D277_D256_RESET_ALIGNED_DATA_PRIOR_TINY_SMOKE_WARN_NO_LONG_PPO`.
- Current next work:
  - Do not run long PPO.
  - Do not claim learned policy, teacher-off success, or RoArm readiness.
  - Next concrete step is teacher-off frozen eval from the D277/D256-reset
    setup: `bc_teacher_blend=0.0`, same `tap10cm + link5_collision_aabb`,
    same D256 reset hook, no random episode offset, TensorBoard gate with
    `--expect_d256_reset`, and strict AABB/useful/overshoot/action-cap checks.
  - Only after teacher-off frozen eval passes should a short controlled PPO
    ladder be considered.

## Previous Result: D272

- D272 corrected tiny PPO smoke:
  - Ran a tiny Isaac Lab PPO smoke on `tap10cm`, fixed +x, and
    `tap_contact_proxy_mode=link5_collision_aabb`.
  - Runtime root:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d272_logs/cube10cm_d272_tap10cm_aabb_bc_metrics_smoke/`.
  - D257 checkpoint loaded through:
    `bc_teacher_checkpoint_path`.
  - TensorBoard now records BC teacher metrics in the tap branch:
    - `cube_push_bc_teacher_blend_mean`: `1.0 -> 1.0`;
    - `cube_tap_bc_teacher_blend_mean`: `1.0 -> 1.0`;
    - `cube_push_bc_teacher_imitation_mse`:
      `0.9571873545646667 -> 0.9571402072906494`;
    - `cube_tap_bc_teacher_imitation_mse`:
      `0.9571873545646667 -> 0.9571402072906494`;
    - `bc_teacher_imitation_penalty`:
      `-4.7859368324279785 -> -4.785701751708984`.
  - Behavior gate still fails:
    - `cube_tap_contact_seen_rate`: `0.0 -> 0.0`;
    - `cube_tap_useful_seen_rate`: `0.0 -> 0.0`;
    - `cube_tap_success_rate`: `0.0 -> 0.0`;
    - `cube_tap_max_disp_along_m` max:
      `0.0005519219557754695`;
    - `cube_tap_contact_vertical_offset_m` last:
      `0.1504015177488327`;
    - `cube_tap_overshoot_seen_rate` max:
      `0.0651041716337204`;
    - `cube_push_joint_delta_cap_rate` max:
      `0.330078125`;
    - `Train/mean_reward`: `-24.44062042236328 -> -70.10071563720703`.
  - TensorBoard gate:
    `tensorboard_scalar_gate_d272.json/md`.
  - Gate verdict:
    `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`.
- Code fixes in D272:
  - `RoArmCubeTap10cmEnv._get_rewards()` now applies
    `bc_teacher_imitation_reward_scale` through `bc_imitation_penalty`.
  - Tap reward logs both push-compatible and tap-specific BC teacher metrics:
    blend, imitation MSE, and teacher action magnitude.
  - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py` now has
    `--env_kind tap10cm` and `--expect_bc_teacher`; tap/AABB uses contact,
    reaction, useful, vertical-offset, displacement, overshoot, action-cap, and
    BC teacher scalars as primary gates. Raw TCP distance is diagnostic only for
    tap/AABB.
- Critical interpretation:
  - D272 proves the teacher prior is wired and observable, not that the policy
    has learned or that teacher-on behavior pushes correctly from default PPO
    resets.
  - This is not “too strict” on the main gates: contact/useful/success are all
    zero while the tool is vertically far from the AABB contact proxy. The TCP
    warning is intentionally weaker because TCP-point distance is not the
    dataset contact contract.
  - The likely blocker is reset/pose-distribution mismatch: D269 teacher-only
    from D256 initial reset reached AABB contact/useful `0.71875`, while D272
    default PPO reset reached `0.0`.
- Post-run:
  - no active `python`/`python3` process was visible via `ps -C`;
  - `nvidia-smi` returned to the observed baseline of about `2509MiB` used;
  - generated files include one TensorBoard event file and `model_0.pt` /
    `model_1.pt`, but these are smoke artifacts, not learned-policy evidence.
- Verdict:
  `D272_TAP10CM_AABB_DATA_PRIOR_WIRING_VISIBLE_BEHAVIOR_FAIL_NO_PPO_PROMOTION`.
- Current next work:
  - Do not run long PPO.
  - Do not claim learned policy, teacher-off success, or RoArm readiness.
  - Next concrete step is D256 pose/reset distribution alignment for the PPO env:
    choose reset states from the D256 train-clean frame-0/state distribution,
    rerun teacher-only/contact probe under the same AABB/useful/TensorBoard
    gates, and only then consider another tiny PPO smoke.

## Previous Result: D270

- D262 visualization:
  - Added `sim_scripts/cube10cm_top_view_d256_feature_distribution_viz.py`.
  - Output:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_feature_distribution_viz_d262/`.
  - Key plot for interpretation:
    `d256_vs_d261_normalized_support_bars.png`.
  - It shows D261 live env ranges outside D256 support for arm joints,
    `tcp_local_z_m`, `target_to_tcp_*`, and `tcp_to_cube_*`.
- D263 D256 initial-pose reset teacher-only probe:
  - Updated `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py` with
    `--reset_pose_source d256_initial`.
  - This injects D256 frame-0 joint/cube/target/push-dir state into the live
    env, then runs D257 teacher-only.
  - Output:
    `teacher_rollout_probe_d263_d256_initial_reset/tap10cm/teacher_rollout_probe_summary_d263_d256_initial_reset.json`.
  - Initial feature support improved:
    `initial_feature_outside_train_minmax_rate=0.0`;
    `initial_feature_outside_train_p01p99_rate=0.19328703703703703`.
  - Teacher-only behavior still failed contact:
    - contact rate `0.0`;
    - min TCP-cube distance mean/min/max
      `0.08348368108272552` / `0.06940185278654099` /
      `0.09543989598751068`;
    - max disp along mean/max
      `0.0014523034915328026` / `0.01252603530883789`;
    - raw delta clip exceed rate `0.20877155172413794`;
    - raw delta max `0.6774565577507019`;
    - action cap rate `0.13050466954022988`.
  - Interpretation:
    D256 pose reset is a real improvement over D261, but not sufficient for
    contact.
- D264 D256 action replay probe:
  - Added `sim_scripts/cube10cm_top_view_d256_action_replay_probe.py`.
  - Teacher and PPO are disabled.
  - It resets from D256 frame-0 state and replays D256
    `state + joint_delta` targets directly in the live 10cm env with
    `hold_steps=3`.
  - Output:
    `d256_action_replay_probe_d264/tap10cm/d256_action_replay_summary_d264.json`.
  - Result:
    - teacher used `False`;
    - contact rate `0.0`;
    - min TCP-cube distance mean/min/max
      `0.07518836855888367` / `0.06179572641849518` /
      `0.09923214465379715`;
    - max disp along mean/max
      `0.006767723709344864` / `0.017127275466918945`;
    - max target jump abs mean/max
      `0.06703907251358032` / `0.09352636337280273`.
  - Interpretation:
    Even direct D256 action replay does not reach the current env contact
    threshold (`0.055m` TCP-cube distance). Therefore the current blocker is not
    just MLP teacher generalization. The D256 visual action/control/contact
    contract does not yet directly reproduce in the current env replay timing

- D265 replay timing sweep:
  - Reused direct D256 action replay with teacher/PPO disabled and varied
    `hold_steps`.
  - Outputs:
    - `d256_action_replay_probe_d265_hold1/tap10cm/d256_action_replay_summary_d264.json`;
    - `d256_action_replay_probe_d265_hold2/tap10cm/d256_action_replay_summary_d264.json`;
    - `d256_action_replay_probe_d265_hold4/tap10cm/d256_action_replay_summary_d264.json`;
    - `d256_action_replay_probe_d265_hold5/tap10cm/d256_action_replay_summary_d264.json`.
  - Contact rate stayed `0.0` for all tested timings.
  - Minimum TCP-cube distance mins:
    - hold 1: `0.06175459548830986`;
    - hold 2: `0.061787448823451996`;
    - hold 3: `0.06179572641849518` from D264;
    - hold 4: `0.06307728588581085`;
    - hold 5: `0.0657862201333046`.
  - Interpretation:
    simple frame-cadence/action-hold tuning does not explain the missing
    contact; slower replay moves farther from the threshold.
- D266 recorded-state sequence probe:
  - Added `sim_scripts/cube10cm_top_view_d256_state_sequence_probe.py`.
  - Teacher, PPO, and learned action replay are disabled.
  - It writes the D256 recorded arm joint state and cube pose into the live
    10cm env, then measures the current contact proxy.
  - Output:
    `d256_state_sequence_probe_d266/tap10cm/d256_state_sequence_summary_d266.json`.
  - Result:
    - contact rate `0.0`;
    - min TCP-cube distance mean/min/max
      `0.07699309289455414` / `0.06270913034677505` /
      `0.1001250371336937`;
    - max disp along mean/max
      `0.0074400329031050205` / `0.018024206161499023`.
  - Interpretation:
    even recorded D256 states do not satisfy the current env's
    `tcp_cube_dist < 0.055m` contact proxy. The next blocker is therefore the
    visual-label/contact-proxy/tool-surface geometry contract, not PPO.
    and contact metric.
- D267-D270 contact-proxy correction:
  - Code review found that the top-view renderer uses
    `train_cube_tap10cm_ppo_smoke._apply_candidate6_contract()`, which sets
    `tap_contact_proxy_mode="link5_collision_aabb"`.
  - Therefore D264-D266 measured the wrong contact gate when they used only
    `_push_terms().tcp_cube_dist < 0.055m`.
  - Updated probes so tap10cm contact uses `_tap_terms().tap_contact_proxy` and
    separately logs the older TCP threshold.
  - D267 recorded-state sequence with `link5_collision_aabb`:
    - output:
      `d256_state_sequence_probe_d267_aabb/tap10cm/d256_state_sequence_summary_d267_aabb.json`;
    - AABB contact rate `1.0`;
    - tap useful rate `1.0`;
    - TCP-threshold contact rate `0.0`;
    - min TCP-cube distance mean/min/max
      `0.07699309289455414` / `0.06270913034677505` /
      `0.1001250371336937`.
  - D267 recorded-state sequence with `tcp_point`:
    - output:
      `d256_state_sequence_probe_d267_tcppoint/tap10cm/d256_state_sequence_summary_d267_tcppoint.json`;
    - contact rate `0.0`;
    - tap useful rate `0.0`;
    - TCP-threshold contact rate `0.0`.
  - D268 direct D256 action replay with `link5_collision_aabb`, `hold_steps=3`:
    - output:
      `d256_action_replay_probe_d268_aabb_hold3/tap10cm/d256_action_replay_summary_d268_aabb_hold3.json`;
    - AABB contact rate `1.0`;
    - tap useful rate `1.0`;
    - TCP-threshold contact rate `0.0`;
    - max disp along mean/max
      `0.006767723709344864` / `0.017127275466918945`.
  - D269 D257 teacher-only from D256 initial reset with
    `link5_collision_aabb`:
    - output:
      `teacher_rollout_probe_d269_aabb_d256_initial/tap10cm/teacher_rollout_probe_summary_d269_aabb_d256_initial.json`;
    - AABB contact rate `0.71875`;
    - tap useful rate `0.71875`;
    - TCP-threshold contact rate `0.0`;
    - max disp along mean/max
      `0.0014523034915328026` / `0.01252603530883789`;
    - raw delta clip exceed rate `0.20877155172413794`;
    - action cap rate `0.13050466954022988`.
  - D270 offline audit over all D256 train-clean teacher rows:
    - output:
      `d256_contact_contract_audit_d270/d256_contact_contract_audit_d270.json`;
    - rows `142978`;
    - `tap_contact_proxy` rate `0.8646784820042245`;
    - `tap_contact_seen` and `tap_reaction_seen` rate
      `0.9137559624557624`;
    - `tcp_sphere_055` rate `0.0`;
    - `tcp_point_face_band` rate `0.0`.
  - Interpretation:
    the previous TCP-only contact failures were overly strict false negatives
    for the top-view dataset contract. The correct next gate is not raw TCP
    contact; it is AABB contact plus reaction/displacement/no-overshoot quality.
    Teacher-only now has partial AABB contact/useful, but it is weaker than
    direct D256 replay on mean displacement, so do not claim learned behavior.
  - Prepared corrected tiny PPO command candidate, not run:
    `cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/ppo_data_prior_smoke_command_d270_corrected_tap10cm_aabb.txt`.
- Post-run:
  - no matching Isaac/PPO/teacher-probe/action-replay/torchrun/rl_games process
    remained;
  - GPU returned to the observed baseline, about `2509MiB` used /
    `13436MiB` free.
- Historical D270 verdict:
  `D270_TCP_CONTACT_GATE_FALSE_NEGATIVE_AABB_CONTRACT_RESTORED_NO_LONG_PPO`.
- Historical D270 implication:
  raw TCP-point contact is too strict for this top-view dataset contract. The
  correct branch gate is AABB/tool-surface contact plus reaction, displacement,
  no-overshoot, action saturation, and TensorBoard task scalars.

## Previous Result: D261

- Added a D256-compatible BC teacher feature target mode:
  `bc_teacher_feature_target_mode`.
  - Default remains `tcp_target` for existing behavior.
  - D257/D256 teacher probes can now set `env_target`, which uses
    `self._target_world` and matches the visual-log
    `target_position_world_m` contract.
- Updated `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py` so the
  current teacher-only probe records `bc_teacher_feature_target_mode` and writes
  D261-tagged artifacts.
- Updated `roarm_rl/train_cube_push_ppo.py` for the next tiny PPO smoke only
  after teacher-only contact passes:
  - `--env_kind {push3cm,tap10cm}`;
  - dynamic `gym.make(env_id, cfg=env_cfg)`;
  - `--fixed_push_dir_x/--fixed_push_dir_y`;
  - `--bc_teacher_feature_target_mode {tcp_target,env_target}`.
- Static validation:
  `python3 -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_push_ppo.py sim_scripts/cube10cm_top_view_teacher_rollout_probe.py`
  passed.
- Ran teacher-only D261 probe, no PPO learning:
  `tap10cm`, fixed +x (`fixed_push_dir_x=1`, `fixed_push_dir_y=0`),
  `bc_teacher_feature_target_mode=env_target`, no IK reset.
  - Output:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx/tap10cm/teacher_rollout_probe_summary_d261_envtarget_posx.json`.
  - Contact rate: `0.0`.
  - Min TCP-cube distance mean/min/max:
    `0.2137620449066162` / `0.144382044672966` /
    `0.29925453662872314`.
  - Max disp along mean/max:
    `1.163780689239502e-05` / `2.765655517578125e-05`.
  - Raw delta clip exceed rate: `0.7170689655172414`.
  - Action cap rate: `0.37896012931034484`.
  - Feature outside D256 train min/max rate: `0.4267700351213282`.
  - Contract improvements:
    `push_dx=1`, `push_dy=0`, `target_local_z_m=0.03788299858570099`
    now match D256 train-clean.
  - Remaining blockers:
    arm joints, TCP height, and target-to-TCP features remain out of D256
    train-clean range.
- Ran teacher-only D261 probe with the same config plus `--ik_endpoint_reset`.
  - Output:
    `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d261_envtarget_posx_ik/tap10cm/teacher_rollout_probe_summary_d261_envtarget_posx_ik.json`.
  - Contact rate: `0.0`.
  - Min TCP-cube distance mean/min/max:
    `0.0902239978313446` / `0.07528560608625412` /
    `0.1327148675918579`.
  - Max disp along mean/max:
    `1.2454720735549927` / `10.891912460327148`.
  - Raw delta clip exceed rate: `0.6805603448275862`.
  - Raw delta max: `264.475830078125`.
  - Action cap rate: `0.3602280890804598`.
  - Interpretation:
    IK reset reduces distance but creates unstable/explosive rollout behavior;
    it is not a PPO promotion fix.
- Post-run:
  - no matching Isaac/PPO/teacher-probe/torchrun/rl_games process remained;
  - GPU returned to the same observed baseline, about `2509MiB` used /
    `13436MiB` free, with existing non-Isaac contexts still present.
- Verdict:
  `D261_FEATURE_TARGET_CONTRACT_PARTIAL_FIX_TEACHER_ONLY_CONTACT_FAIL_NO_PPO_PROMOTION`.
- Current next work:
  - Do not run long PPO.
  - Do not run even tiny PPO yet unless explicitly overriding the failed
    teacher-only gate.
  - Next concrete research task is to align the env reset/action rollout
    distribution with D256 train-clean visual trajectory states, or retrain a
    teacher on env-side rollout features. The key blockers are arm joint range,
    TCP height, target-to-TCP geometry, and raw delta explosion under IK reset.
  - TensorBoard remains mandatory for any later tiny PPO smoke, but there is no
    PPO candidate to dashboard until teacher-only contact is plausible.

## Previous Result: D260

- Added `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`.
- Purpose:
  - read existing TensorBoard event logs;
  - summarize reward, PPO loss, policy noise, task behavior, action saturation,
    and BC teacher scalars;
  - write JSON/markdown evidence before any PPO promotion.
- TensorBoard availability:
  - system Python does not have TensorBoard;
  - `conda run -n isaaclab tensorboard --version` reports `2.20.0`.
- Dashboard command pattern:
  `conda run -n isaaclab tensorboard --logdir <ppo_log_dir> --host 127.0.0.1 --port 6006`.
- Applied the scalar gate to the existing D258 event log:
  `ppo_data_prior_d257_logs/cube10cm_d257_data_prior_smoke2`.
- Output:
  - `tensorboard_scalar_gate_d260.json`;
  - `tensorboard_scalar_gate_d260.md`.
- D258 TensorBoard gate verdict:
  `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`.
- Key failures:
  - no task success/contact signal;
  - `cube_push_low_motion_rate` last `0.9778646230697632`;
  - `cube_push_joint_delta_cap_rate` max `0.7411024570465088`.
- Key warnings:
  - `Train/mean_reward` has only `1` point;
  - `cube_push_tcp_cube_dist_m` last `0.3268700838088989`;
  - `cube_push_disp_along_m` last `0.00015073080430738628`;
  - `cube_push_controlled_rate` last `0.0182291679084301`.
- PPO/policy scalars exist:
  - `Loss/value_function`: `6711.08642578125 -> 6737.7255859375`;
  - `Loss/surrogate`: `-0.011346347630023956 -> -0.012086811475455761`;
  - `Loss/entropy`: `7.177923202514648 -> 7.179165840148926`;
  - `Policy/mean_noise_std`: `0.8005133867263794 -> 0.8006278276443481`.
- Interpretation:
  TensorBoard is now mandatory for PPO smoke/ladder inspection, but D260 does
  not unblock PPO. Applying it to D258 reinforces the D259 conclusion: no long
  PPO before feature-contract correction and teacher-only contact.

## Previous Result: D259

- Added and ran
  `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py` with no PPO
  learning.
- Found and fixed a 10cm env config bug before the valid 10cm probe:
  `RoArmCubeTap10cmEnvCfg` was missing `tap_overshoot_terminate`, while
  `_get_dones()` reads it. The default is now explicitly `False`.
- Probe outputs:
  - D258 env reproduction:
    `teacher_rollout_probe_d259/push3cm`;
  - intended 10cm env:
    `teacher_rollout_probe_d259/tap10cm`;
  - 10cm +x/IK alignment attempt:
    `teacher_rollout_probe_d259_posx_ik/tap10cm`.
- D258 env reproduction uses `RoArm-CubePush-Direct-v0`, cube size `0.03`, not
  the professor 10cm cube. It reached no contact:
  - contact rate `0.0`;
  - min TCP-cube distance mean/min/max
    `0.21149027347564697` / `0.09386380761861801` /
    `0.3410947620868683`;
  - raw delta clip exceed rate `1.0`;
  - action cap rate `0.7770743534482759`;
  - feature outside D256 train min/max rate `0.5803001277139208`.
- Intended 10cm env also reached no contact:
  - contact rate `0.0`;
  - min TCP-cube distance mean/min/max
    `0.18824803829193115` / `0.07045303285121918` /
    `0.3121436536312103`;
  - raw delta clip exceed rate `1.0`;
  - action cap rate `0.7768139367816091`;
  - feature outside D256 train min/max rate `0.593532487228608`.
- 10cm +x/IK alignment attempt fixed `push_dx/push_dy` but was still invalid
  for promotion:
  - contact rate `0.0`;
  - raw delta clip exceed rate `0.9999676724137931`;
  - action cap rate `0.5438308189655172`;
  - cube displacement exploded in some envs, with max disp along
    `11.039312362670898m`.
- Critical mismatches:
  - D258 PPO smoke used the default 3cm `RoArmCubePushEnvCfg` path, while the
    D247-D257 data is professor 10cm cube data.
  - D256 train-clean `push_dx/push_dy` is +x-only (`1.0/0.0`), but env reset
    randomizes push direction unless `fixed_push_dir_x/y` is set.
  - D256 feature `target_position_world_m` is not the same semantic object as
    env-side `_bc_teacher_tcp_target()` used in `_bc_teacher_feature_tensor()`.
    The clearest symptom is 10cm `target_local_z_m`: D256 train is fixed at
    `0.03788299858570099`, while env teacher features are
    `0.0768829956650734..0.09088299423456192`.
  - Default env reset starts near `HOME_RAD` unless `ik_endpoint_reset=True`;
    D256 train-clean starts from visual trajectory poses, so joint features are
    out of distribution.
- Post-run GPU returned to the same observed baseline of about `2509MiB` used /
  `13436MiB` free. Existing non-Isaac Python/Rerun compute contexts remain.
- Verdict:
  `D259_TEACHER_ROLLOUT_PROBE_CONTACT_FAIL_FEATURE_CONTRACT_MISMATCH_NO_LONG_PPO`.

## Previous Result: D258

- Ran the first tiny Isaac Lab PPO data-prior smoke from the D257 teacher
  checkpoint.
- First sandbox attempt failed before a valid smoke:
  - Isaac/PhysX could not see CUDA inside the sandbox;
  - the original D257 command lacked `PYTHONPATH=.` and hit a `roarm_rl` import
    error.
- Reran on host GPU with `PYTHONPATH=.` and the same D257 PPO knobs:
  `num_envs=32`, `max_iterations=2`, `num_steps_per_env=24`,
  `bc_teacher_blend=1.0`, `bc_teacher_imitation_reward_scale=5.0`,
  `bc_teacher_policy_delta_clip_rad=0.04`,
  `bc_teacher_phase_timing=direct_steps`.
- Output root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs/cube10cm_d257_data_prior_smoke2`.
- Tracked summary:
  `ppo_data_prior_smoke_summary_d258.json` and
  `ppo_data_prior_smoke_summary_d258.md`.
- Generated runtime artifacts:
  `events.out.tfevents.*`, `model_0.pt`, and `model_1.pt`; these are now ignored
  as generated runtime artifacts.
- Wiring metrics:
  - checkpoint loaded through `bc_teacher_checkpoint_path`;
  - `cube_push_bc_teacher_blend_mean`: `1.0`, `1.0`;
  - `cube_push_bc_teacher_imitation_mse`: `1.210442`, `1.253437`;
  - `bc_teacher_imitation_penalty`: `-6.052209`, `-6.267184`.
- Behavior metrics:
  - `cube_push_disp_along_m`: `-0.0`, `0.000151`;
  - `cube_push_disp_xy_m`: `0.000008`, `0.000615`;
  - `cube_push_tcp_cube_dist_m`: `0.338168`, `0.32687`;
  - `cube_push_success_rate`: `0.0`, `0.0`;
  - final `Train/mean_reward`: `-392.534027`.
- Post-run check:
  no active Isaac/PPO/torchrun process remained; GPU memory returned to about the
  pre-run baseline.
- Verdict:
  `D258_PPO_DATA_PRIOR_SMOKE_WIRING_PASS_BEHAVIOR_UNPROVEN`.

## Previous Result: D257

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

- Long PPO promotion is frozen while the professor visual-dataset branch is
  active.
- Current professor branch state is method-pipeline-ready plus D256
  transition/reward data, D257 state-action teacher checkpoint, D270 restored
  AABB dataset contact contract, D277 D256-reset-aligned teacher-on tiny PPO
  smoke, D280 supervised actor distillation, D281 env-stop/min-contact PPO
  safety controls, D282 PPO internal actor-preservation, D283-D285 short
  preserved-actor PPO gates, D286-D291 actor/teacher bridge diagnostics, D292
  tiny PPO plumbing/checkpoint pass, D293 displacement/horizon contract, D294
  max/mean/rate displacement gate, D295 constrained short PPO rate-gate
  runtime, D296 non-PPO overshoot-control diagnostic, D297 reset-protocol
  re-audit, D298 tiny PPO TensorBoard collection failure, D299
  no-success-terminate collection overshoot fix, D300 collection-final
  TensorBoard gate failure, D301 final-env diagnostic, D302 hard-bin
  diagnostic later superseded by D303, D303 hard-bin process-contamination
  re-audit, D304 true PPO collection-path JSONL trace gate failure, D305
  non-PPO closed-loop recovery repair partial result, D306 phase-aware
  action repair tiny-vs-overshoot bracket, and D307 non-PPO action-governor
  partial diagnostic. It is still not
  training-complete, learned policy, or RoArm-ready.
- Recommended next work:
  1. inspect D307 first:
     `claudedocs/session_20260630_cube10cm_top_view_d307_action_governor.md`;
  2. treat D307 as partial/no-promotion: `predict_stop` fixed ep561 from
     `41.5mm` overshoot to `4.996mm` no-overshoot and failed6 reached useful
     `1.0`, overshoot `0.0`, cap `0.0`, but only `4/6` envs reached
     `>=1mm`; recorded-target repair improved offline metrics but collapsed
     runtime displacement;
  3. do not run long PPO, a PPO ladder, partial actor preservation, real actor
     updates, or another tiny PPO trace gate from D307;
  4. next work is non-PPO deployable action-space/control repair: either move a
     default-off displacement/velocity governor into the env and broaden fresh
     reset diagnostics, or change the action representation toward a tool/object
     push primitive instead of brittle scalar joint deltas;
  5. inspect D306:
     `claudedocs/session_20260630_cube10cm_top_view_d306_phase_action_repair.md`;
  6. treat D306 as no-promotion: candidate-1 restores useful contact but gives
     only about `0.037mm` XY displacement on ep561; candidate-2 can create
     `41.5mm` displacement but overshoots; global action clips and contact
     slowdown collapse displacement below `0.05mm`;
  7. do not run long PPO, a PPO ladder, partial actor preservation, real actor
     updates, or another tiny PPO trace gate from D306;
  8. inspect D305:
     `claudedocs/session_20260629_cube10cm_top_view_d305_closed_loop_recovery_repair.md`;
  9. treat D305 as partial repair/no-promotion: no-contact was repaired on the
     failed6 diagnostic set, but D304-like displacement stayed tiny and cap
     pressure stayed high;
  10. do not run long PPO, a PPO ladder, partial actor preservation, real actor
     updates, or another tiny PPO trace gate from D305;
  11. inspect D304 as the true PPO collection-path trace source:
     `claudedocs/session_20260629_cube10cm_top_view_d304_collection_trace_gate.md`;
  12. treat D304 as no-promotion: true PPO collection-path trace exists, but the
     collection-final gate failed contact/reaction `0.84375` and useful
     `0.8125` versus the strict `0.90` promotion threshold;
  13. do not lower the `0.90` final useful/contact gate as a promotion standard
     unless the user explicitly chooses a weaker exploratory gate;
  14. inspect D303 as the warning against stale sequential multi-bin probes:
     `claudedocs/session_20260629_cube10cm_top_view_d303_hard_bin_reaudit.md`;
  15. inspect D300 as the final-state TensorBoard scalar gate baseline:
     `claudedocs/session_20260629_cube10cm_top_view_d300_collection_final_gate.md`;
  16. inspect D299 as the success-termination negative-control fix:
     `claudedocs/session_20260629_cube10cm_top_view_d299_collection_contract_no_success_terminate.md`;
  17. do not use `tap_success_terminate=True` for the current actor-preserved
     tap10cm collection gate;
  18. inspect D298 as the negative control:
     `claudedocs/session_20260629_cube10cm_top_view_d298_tiny_ppo_directreset_gate.md`;
  19. inspect D297 next:
     `claudedocs/session_20260629_cube10cm_top_view_d297_teacher_off_reset_protocol.md`;
  20. keep `--d256_reset_warmup_mode direct_reset` as the default teacher-off
     gate. The old forced-step paths are diagnostic only and must not be used as
     promotion gates;
  21. inspect D296 only as the negative control that motivated the reset-protocol
     audit:
     `claudedocs/session_20260629_cube10cm_top_view_d296_overshoot_control_diagnostic.md`;
  22. keep the 10cm cube task framed as a tool-object interaction primitive:
     contact, reaction, controlled displacement, no overshoot, and visual
     trajectory output;
  23. keep `link5_collision_aabb` as the current tap10cm contact proxy unless a
     separately named `tool_surface_union` contract is implemented and validated;
  24. do not claim final policy, learned policy, RoArm readiness, or mining
      automation readiness from D307.
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
2. `START_HERE.md` D307 current truth and Active Direction.
3. `claudedocs/DECISIONS.md` D307, D306, D305, D304, D303, D302, D301, D300, D299, D298, D297, D296, D295, D294, D293, D292, D291, D290, D288, D287, D286, D283-D285,
   D282, D281, D257, D256, D254, D247, D246, D232.
4. `claudedocs/EXPERIMENT_LEDGER.md` latest D307 row.
5. `claudedocs/session_20260630_cube10cm_top_view_d307_action_governor.md`.
6. D307 runtime/diagnostic output roots:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/failed6_predict_stop_h020_v200/`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/recorded_repair_failed6_predict_stop_h020_v200/`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/action_governor_d307/tap10cm/recorded_repair_lr5e5_ep80/`
7. `claudedocs/session_20260630_cube10cm_top_view_d306_phase_action_repair.md`.
8. D306 runtime/diagnostic output roots:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_c1_replay_plus_phase_lr5e5_ep100/`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/phase_iter2_replay_plus_failed6_lr5e5_ep100/`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d306/tap10cm/fresh_onebin_iter2_d304runtime_ep561/`
9. `claudedocs/session_20260629_cube10cm_top_view_d305_closed_loop_recovery_repair.md`.
10. D305 runtime/diagnostic output roots:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_recovery_lr1e4_ep80/`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_recovery_repair_d305/tap10cm/failed6_replay_plus_iter2_recovery_lr5e5_ep80/`
11. `claudedocs/session_20260629_cube10cm_top_view_d304_collection_trace_gate.md`.
12. D304 runtime/diagnostic output roots:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/`
   - `cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/collection_final_env_trace_iter_0.jsonl`
   - `cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/tensorboard_scalar_gate_d304_seed29801_trace.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/fresh_failed_episode_probe_d304/`
13. `claudedocs/session_20260629_cube10cm_top_view_d303_hard_bin_reaudit.md`.
14. `claudedocs/session_20260629_cube10cm_top_view_d300_collection_final_gate.md`.
15. D300 runtime output roots:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29801_1it/`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/`
   - `cube10cm_d300_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29801_1it/tensorboard_scalar_gate_d300_seed29801_finalgate.json`
   - `cube10cm_d300_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/tensorboard_scalar_gate_d300_seed29604_finalgate.json`
16. `claudedocs/session_20260629_cube10cm_top_view_d299_collection_contract_no_success_terminate.md`.
17. D299 diagnostic/runtime output roots:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/collection_contract_d299/`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d299/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_1it/`
   - `ppo_command_d299.txt`
   - `tensorboard_dashboard_command_d299.txt`
   - `cube10cm_d299_directreset_actorfreeze_random_stop003_no_success_term_1it/tensorboard_scalar_gate_d299.json`
   - `cube10cm_d299_directreset_actorfreeze_random_stop003_no_success_term_1it/teacher_off_direct_seed29801/teacher_off_policy_eval_summary_d299_direct_seed29801.json`
   - `cube10cm_d299_directreset_actorfreeze_random_stop003_no_success_term_1it/teacher_off_direct_seed29604/teacher_off_policy_eval_summary_d299_direct_seed29604.json`
18. `claudedocs/session_20260629_cube10cm_top_view_d298_tiny_ppo_directreset_gate.md`.
19. D298 PPO output root:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/`
   - `ppo_command_d298.txt`
   - `tensorboard_dashboard_command_d298.txt`
   - `cube10cm_d298_directreset_actorfreeze_random_stop003_1it/tensorboard_scalar_gate_d298.json`
   - `cube10cm_d298_directreset_actorfreeze_random_stop003_1it/teacher_off_direct_seed29801/teacher_off_policy_eval_summary_d298_direct_seed29801.json`
   - `cube10cm_d298_directreset_actorfreeze_random_stop003_1it/teacher_off_direct_seed29604/teacher_off_policy_eval_summary_d298_direct_seed29604.json`
20. `claudedocs/session_20260629_cube10cm_top_view_d297_teacher_off_reset_protocol.md`.
21. D297 action/reset diagnostic output root:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/action_diagnostic_d297/`
   - `teacher_off_direct_seed29603/teacher_off_policy_eval_summary_d297_direct_seed29603.json`
   - `teacher_off_direct_seed29604/teacher_off_policy_eval_summary_d297_direct_seed29604.json`
   - `random_envhook_seed29604/closed_loop_recovery_summary_d297_random_envhook_seed29604_actor_action_diagnostic.json`
   - `random_envhook_direct_seed29604/closed_loop_recovery_summary_d297_random_envhook_direct_seed29604_actor_action_diagnostic.json`
   - `reset_alignment_envhook_seed29604_vel/reset_alignment_envhook_seed29604_vel_d297.csv`
22. `claudedocs/session_20260629_cube10cm_top_view_d296_overshoot_control_diagnostic.md`.
23. D296 overshoot-control output root:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/overshoot_control_d296/`
   - `run_overshoot_control_matrix_d296.sh`
   - `run_candidate_random_checks_d296.sh`
   - `run_conservative_random_checks_d296.sh`
   - `stop_disp003_random_seed29604_envtrace_d296/teacher_off_policy_eval_envs_stop_disp003_random_seed29604_envtrace_d296.csv`
24. `claudedocs/session_20260629_cube10cm_top_view_d295_rate_gate_runtime.md`.
25. D295 PPO output root:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/`
   - `ppo_command_d295.txt`
   - `tensorboard_dashboard_command_d295.txt`
   - `cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/tensorboard_scalar_gate_d295.json`
   - `cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/teacher_off_eval_model0_d295_contract/teacher_off_policy_eval_summary_d295_model0.json`
   - `cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it/model_0.pt`
26. `claudedocs/session_20260629_cube10cm_top_view_displacement_rate_gate_d294.md`.
27. `claudedocs/session_20260629_cube10cm_top_view_displacement_horizon_contract_d293.md`.
28. D292 PPO output root:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d292/tap10cm/ppo_replay_actor_freshgate_actorfreeze_1it/cube10cm_d292_replay_actor_freshgate_actorfreeze_1it/`
   - `tensorboard_scalar_gate_d292.json`
   - `teacher_off_eval_model0/teacher_off_policy_eval_summary_d292_model0.json`
   - `model_0.pt`
29. D291 fresh-per-bin gate:
   - `claudedocs/session_20260628_cube10cm_top_view_fresh_bin_actor_d291.md`
   - `sim_scripts/cube10cm_top_view_d290_closed_loop_recovery_probe.py`
30. D290 replay-batch actor:
   - `claudedocs/session_20260627_cube10cm_top_view_d256_replay_batch_actor_d290.md`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_d256_replay_batches_d290/tap10cm_ep155/model_actor_d256_replay_batches_d290.pt`
31. Relevant code:
   - `roarm_rl/roarm_cube_push_env.py`
   - `roarm_rl/train_cube_push_ppo.py`
   - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
   - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
   - `sim_scripts/cube10cm_top_view_d256_action_replay_probe.py`
   - `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`
   - `sim_scripts/cube10cm_top_view_train_actor_from_replay_batches.py`
   - `sim_scripts/cube10cm_top_view_train_state_action_teacher.py`

## Archived D286 Must Read Reference

1. `CLAUDE.md` Current-State Protocol.
2. `START_HERE.md` D286 current truth and Active Direction.
3. `claudedocs/DECISIONS.md` D286, D283-D285, D282, D281, D280, D279, D278, D277, D272,
   D267-D270, D265-D266, D262-D264, D261, D260, D259, D258, D257, D256,
   D254, D253, D249-D252, D248, D247, D246, D232.
4. `claudedocs/EXPERIMENT_LEDGER.md` latest D286 row.
5. `claudedocs/session_20260626_cube10cm_top_view_d256_reset_bin_actor_probe_d286.md`.
6. D286 outputs:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_actor_probe_d286_comparison/tap10cm/d256_reset_bin_actor_probe_comparison_d286.md`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_actor_probe_d286_default_steps580_corrected/tap10cm/d256_reset_bin_actor_probe_summary_d286.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_actor_probe_d286_action_scale0010_steps580_corrected/tap10cm/d256_reset_bin_actor_probe_summary_d286.json`
7. `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`.
8. `claudedocs/session_20260625_cube10cm_top_view_actor_preserve_short_gates_d283_d285.md`.
9. D283-D285 short-gate outputs:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke/tensorboard_scalar_gate_d283_preserve095_10.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke/teacher_off_eval_model9_no_useful_term/teacher_off_policy_eval_summary_d283_preserve095_10_model9_no_useful_term.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke/actor_teacher_trace_model9_no_useful_term/actor_teacher_trace_summary_d279.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d284/tap10cm/ppo_preserve095_noise002_10_smoke/cube10cm_d284_preserve095_noise002_10_smoke/tensorboard_scalar_gate_d284_preserve095_noise002_10.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke/tensorboard_scalar_gate_d285_actorfreeze_noise002_10.json`
10. `claudedocs/session_20260625_cube10cm_top_view_actor_preservation_d282.md`.
11. D282 actor-preservation outputs:
   - `roarm_rl/train_cube_push_ppo.py`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_freeze_smoke/cube10cm_d282_warmstart_actor_freeze_smoke/teacher_off_eval_after_actor_freeze_no_useful_term/teacher_off_policy_eval_summary_d282_after_actor_freeze_no_useful_term.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_freeze_smoke/cube10cm_d282_warmstart_actor_freeze_smoke/actor_teacher_trace_after_actor_freeze_no_useful_term/actor_teacher_trace_summary_d279.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_preserve095_smoke/cube10cm_d282_warmstart_actor_preserve095_smoke/teacher_off_eval_after_actor_preserve095_no_useful_term/teacher_off_policy_eval_summary_d282_after_actor_preserve095_no_useful_term.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_preserve095_smoke/cube10cm_d282_warmstart_actor_preserve095_smoke/actor_teacher_trace_after_actor_preserve095_no_useful_term/actor_teacher_trace_summary_d279.json`
12. D282 no-preservation conservative10 outputs:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke/tensorboard_scalar_gate_d282_conservative10.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke/teacher_off_eval_model9_no_useful_term/teacher_off_policy_eval_summary_d282_conservative10_model9_no_useful_term.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke/actor_teacher_trace_model9_no_useful_term/actor_teacher_trace_summary_d279.json`
13. D281 corrected no-useful-terminate re-eval:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke/teacher_off_eval_after_conservative_update_no_useful_term/teacher_off_policy_eval_summary_d281_after_conservative_update_no_useful_term.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke/actor_teacher_trace_after_conservative_update_no_useful_term/actor_teacher_trace_summary_d279.json`
14. `claudedocs/session_20260625_cube10cm_top_view_env_stop_ppo_update_d281.md`.
15. `claudedocs/session_20260625_cube10cm_top_view_actor_distill_d280.md`.
16. `claudedocs/session_20260625_cube10cm_top_view_actor_teacher_trace_d279.md`.
17. `claudedocs/session_20260620_cube10cm_top_view_teacher_off_eval_d278.md`.
18. `claudedocs/session_20260620_cube10cm_top_view_d256_reset_aligned_ppo_d273_d277.md`.
19. D277 TensorBoard gate:
   - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/tensorboard_scalar_gate_d277.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/tensorboard_scalar_gate_d277.md`
20. D274 teacher-only env reset hook probe:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d274_env_d256_reset_metrics/tap10cm/teacher_rollout_probe_summary_d274_env_d256_reset_teacher_only_metrics.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d274_env_d256_reset_metrics/tap10cm/teacher_rollout_probe_summary_d274_env_d256_reset_teacher_only_metrics.md`
21. D275/D276 contrast gates:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d275_logs/cube10cm_d275_tap10cm_aabb_d256reset_bc_smoke/tensorboard_scalar_gate_d275.md`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d276_logs/cube10cm_d276_tap10cm_aabb_d256reset_bc_no_randlen_smoke/tensorboard_scalar_gate_d276.md`
22. `claudedocs/session_20260620_cube10cm_top_view_tap10cm_aabb_ppo_gate_d272.md`.
23. D272 TensorBoard gate:
   - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d272_logs/cube10cm_d272_tap10cm_aabb_bc_metrics_smoke/tensorboard_scalar_gate_d272.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d272_logs/cube10cm_d272_tap10cm_aabb_bc_metrics_smoke/tensorboard_scalar_gate_d272.md`
24. `claudedocs/session_20260620_cube10cm_top_view_contact_proxy_correction_d267_d270.md`.
25. D267-D270 outputs:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_state_sequence_probe_d267_aabb/tap10cm/d256_state_sequence_summary_d267_aabb.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_action_replay_probe_d268_aabb_hold3/tap10cm/d256_action_replay_summary_d268_aabb_hold3.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d269_aabb_d256_initial/tap10cm/teacher_rollout_probe_summary_d269_aabb_d256_initial.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_contact_contract_audit_d270/d256_contact_contract_audit_d270.json`
26. `claudedocs/session_20260620_cube10cm_top_view_d256_pose_reset_replay_d262_d264.md`.
27. `claudedocs/session_20260620_cube10cm_top_view_feature_contract_probe_d261.md`.
28. `claudedocs/session_20260619_cube10cm_top_view_tensorboard_gate_d260.md`.
29. `claudedocs/session_20260619_cube10cm_top_view_teacher_rollout_probe_d259.md`.
30. `claudedocs/session_20260619_cube10cm_top_view_ppo_data_prior_smoke_d258.md`.
31. `claudedocs/session_20260618_cube10cm_top_view_state_action_teacher_d257.md`.
32. `claudedocs/session_20260618_cube10cm_top_view_rl_transition_preflight_d256.md`.
33. `claudedocs/cube10cm_top_view_method_pipeline_d254.md`.
31. `claudedocs/session_20260617_cube10cm_top_view_method_pipeline_reframe_d254.md`.
32. `claudedocs/session_20260617_cube10cm_top_view_training_preflight_d253.md`.
33. `claudedocs/session_20260617_cube10cm_top_view_dataset_freeze_filtered_loader_distribution_d249_d252.md`.
34. `claudedocs/session_20260617_cube10cm_top_view_label_package_d248.md`.
35. `claudedocs/session_20260616_cube10cm_top_view_0_999_lerobot_metadata_d247.md`.
36. `claudedocs/session_20260615_cube10cm_top_view_0_999_render_labels_d246.md`.
37. `claudedocs/session_20260612_camera_contract_visual_dataset_disk_audit_d232.md`.
38. D257 output root:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/state_action_teacher_metrics_d257.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/ppo_data_prior_smoke_command_d257.txt`
39. Relevant code:
   - `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
   - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
   - `sim_scripts/cube10cm_top_view_teacher_rollout_probe.py`
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
- Do not promote D292 to long PPO or policy success: it was
  `actor_preserve_blend=1.0` plumbing/checkpoint validation and its displacement
  was too small under D293/D294.
- Do not promote D295 to learned-policy success, partial actor preservation, or
  long PPO: TensorBoard failed below the `0.90` useful/contact/reaction gate,
  and the old saved-checkpoint teacher-off overshoot must be read through the
  corrected D297 direct-reset re-audit.
- Do not use D296's old forced-second-reset teacher-off path as a policy
  promotion gate. D297 showed the random-reset overshoot was tied to the eval
  reset protocol/contact-cache path, not to the actor action or D256 labels.
- Do not run another cube10cm PPO runtime from D296 action constraints alone:
  linspace D256 reset passed for some constraints, but those constraints were
  superseded by the D297 direct-reset teacher-off re-audit.
- Do not accept linspace D256 reset alone as a promotion gate. Random D256 reset
  sampling is now mandatory before any next tiny PPO + TensorBoard gate.
- Do not treat magnitude-only action clipping or displacement-stop as the fix.
  Under D297 these are safety/gate options to test inside a corrected tiny PPO
  gate, not learned-policy evidence.
- Do not run long PPO, a PPO ladder, partial actor preservation, or real PPO
  actor updates from D297. The next valid runtime is only one explicitly
  approved tiny PPO + TensorBoard gate with direct-reset teacher-off validation.
- Do not promote D298 to PPO success: the one approved tiny PPO runtime exited
  cleanly, but TensorBoard collection failed (`useful=0.0448`,
  `overshoot=0.7133`) even though saved-checkpoint teacher-off direct-reset
  evals passed.
- Do not run another PPO or long PPO from D298 before a non-PPO collection-time
  reset/termination contract diagnostic explains the mismatch between PPO
  collection and teacher-off direct-reset eval.
- Do not treat high D298 displacement alone as progress: max XY reached
  `0.0348m`, but it came with high overshoot and low useful/success rates.
- Do not use `tap_success_terminate=True` for the current actor-preserved
  tap10cm collection gate. D299 traced D298's collection overshoot mode to
  success-termination episode recycle.
- Do not promote D299 to learned-policy success: it fixed collection overshoot
  under no-success-terminate, but used `actor_preserve_blend=1.0`, ended as
  `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`, and lacks completed-episode
  Train reward scalars.
- Do not run long PPO, a PPO ladder, partial actor preservation, or real actor
  updates from D299. Next work is only gate-semantics cleanup or one explicitly
  approved tiny no-success-terminate multi-seed gate.
- Do not promote D300 to learned-policy success: final-state TensorBoard scalars
  work, but both seed `29801` and seed `29604` failed the strict `0.90`
  collection-final contact/useful gate under full actor preservation.
- Do not lower the D300 final useful threshold from `0.90` as a promotion
  standard unless the user explicitly chooses a weaker exploratory gate.
- Do not run long PPO, a PPO ladder, partial actor preservation, or real actor
  updates from D300. Next work is non-PPO final-coverage diagnostic, not another
  training run.
- D303 supersedes D302's sequential multi-bin hard-bin failure interpretation:
  later-bin actor/teacher failures reproduced only under sequential multi-bin
  process reuse. Fresh one-bin/fresh-process probes are required for hard-bin
  evidence.
- D304 ran the valid tiny no-success-terminate actor-preserved PPO collection
  trace gate and captured true failed envs in JSONL, but it failed promotion:
  collection-final contact/reaction `0.84375`, useful `0.8125`, overshoot
  `0.03125`.
- Do not promote D304 to learned-policy success: actor preservation was full,
  the strict `0.90` final contact/useful gate failed, and failed-state
  closed-loop recovery still has high actor-vs-recovery MSE `1.084940`.
- Do not run long PPO, a PPO ladder, partial actor preservation, or real actor
  updates from D304. Next work is non-PPO closed-loop recovery/action repair,
  then fresh one-bin/direct-reset diagnostics before any new tiny PPO trace gate.
- D305 partially repaired D304 no-contact on failed6, but it did not meet the
  displacement/cap contract: candidate-1 D304-like probes kept useful `1.0` and
  overshoot `0.0`, but cap stayed `0.333333..0.666667` and displacement was only
  about `0.013..0.016mm`; candidate-2 reduced MSE/cap but lost useful contact
  and collapsed displacement.
- Do not promote D305 to learned-policy success or PPO readiness. Do not run
  long PPO, a tiny PPO trace gate, PPO ladder, partial actor preservation, or
  real actor updates from D305. Next work is phase/displacement-aware non-PPO
  action repair with explicit cap/smoothness control and minimum displacement
  preservation.
- D306 brackets the D305 action-repair problem: candidate-1 is safe but produces
  only about `0.037mm` XY displacement on ep561, while candidate-2 can produce
  `41.5mm` but overshoots; action clips `0.50/0.75` and contact slowdown avoid
  overshoot but collapse displacement below `0.05mm`.
- Do not promote D306 to learned-policy success or PPO readiness. Do not run
  long PPO, a tiny PPO trace gate, PPO ladder, partial actor preservation, or
  real actor updates from D306. Next work is a non-PPO displacement/velocity
  aware action governor or push pulse controller, not another scalar actor fit.
- D307 partially validates the action-governor direction: D306 candidate-2 ep561
  with `predict_stop h=0.020s v=0.200m/s` reached max XY `4.996mm` with no
  overshoot, but failed6 reached `>=1mm` in only `4/6` envs and recorded-target
  supervised repair collapsed runtime displacement below `0.023mm`.
- Do not promote D307 to learned-policy success or PPO readiness. Do not run
  long PPO, a tiny PPO trace gate, PPO ladder, partial actor preservation, or
  real actor updates from D307. Next work is non-PPO deployable
  action-space/control repair before any PPO gate.
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
