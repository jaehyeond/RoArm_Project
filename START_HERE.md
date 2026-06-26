# START_HERE.md

Last updated: 2026-06-26 KST (D286 current truth: D256 reset episode-bin diagnostics are wired through `d256_reset_episode_min/max` and `sim_scripts/cube10cm_top_view_d256_reset_bin_actor_probe.py`. The D285 frozen actor fails all D256 episode bins at default `action_scale=0.04`: cap max by bin `0.6302/0.7604/0.8229/0.7031/0.78125`, useful max `0.0` across bins. Reducing `action_scale` to `0.01` lowers cap max to `0.0104/0.0156/0.0052/0.0781/0.0833`, but useful max is still `0.0` across bins. Verdict: `D286_NO_RESET_BIN_OR_ACTION_SCALE_FIX_READY_FOR_PPO`. No long PPO, learned-policy claim, RoArm readiness, RunPod/B200 action, or cleanup claim exists.)

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

## Latest Result: D286

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

- PPO promotion is frozen while the professor visual-dataset branch is active.
- Current professor branch state is method-pipeline-ready plus D256
  transition/reward data, D257 state-action teacher checkpoint, D270 restored
  AABB dataset contact contract, D277 D256-reset-aligned teacher-on tiny PPO
  smoke, D280 supervised actor distillation, D281 env-stop/min-contact PPO
  safety controls, D282 PPO internal actor-preservation, D283-D285 short
  preserved-actor PPO gates, and D286 D256 reset-bin actor diagnostics. It is
  still not training-complete, learned policy, or RoArm-ready.
- Recommended next work:
  1. inspect D286 first:
     `claudedocs/session_20260626_cube10cm_top_view_d256_reset_bin_actor_probe_d286.md`;
  2. inspect the D286 comparison artifact:
     `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_reset_bin_actor_probe_d286_comparison/tap10cm/d256_reset_bin_actor_probe_comparison_d286.md`;
  3. treat reset-bin filtering alone and action-scale reduction alone as
     insufficient fixes;
  4. inspect D285/D284/D283 background:
     `claudedocs/session_20260625_cube10cm_top_view_actor_preserve_short_gates_d283_d285.md`;
  5. inspect D285 actor-freeze TensorBoard failure and D284/D283 contrasts:
     D285 proves actor update is not the only blocker, D284 proves lowering
     exploration noise is not enough, and D283 proves saved deterministic
     checkpoints can pass even when collection fails;
  6. inspect D282 actor-preservation wiring and D281 corrected eval protocol;
  7. keep the corrected saved-checkpoint eval/trace protocol:
     use `tap_stop_after_useful_seen` plus `vertical_gate_mode=min_contact`,
     but do not use `tap_useful_terminate` for frozen eval/trace summaries;
  8. do not launch longer PPO from D283/D284/D285/D286;
  9. next valid work is non-PPO fix before PPO scale:
     repair the actor/teacher bridge or add explicit action projection,
     action-cap constraint, or teacher constraint;
  10. only after teacher-off/bin diagnostics, collection TensorBoard, and
     saved-checkpoint gates all pass
     should a longer controlled PPO ladder be considered.
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
20. D272 TensorBoard gate:
   - `sim_scripts/cube10cm_top_view_tensorboard_scalar_gate.py`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d272_logs/cube10cm_d272_tap10cm_aabb_bc_metrics_smoke/tensorboard_scalar_gate_d272.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d272_logs/cube10cm_d272_tap10cm_aabb_bc_metrics_smoke/tensorboard_scalar_gate_d272.md`
21. `claudedocs/session_20260620_cube10cm_top_view_contact_proxy_correction_d267_d270.md`.
22. D267-D270 outputs:
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_state_sequence_probe_d267_aabb/tap10cm/d256_state_sequence_summary_d267_aabb.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_action_replay_probe_d268_aabb_hold3/tap10cm/d256_action_replay_summary_d268_aabb_hold3.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_rollout_probe_d269_aabb_d256_initial/tap10cm/teacher_rollout_probe_summary_d269_aabb_d256_initial.json`
   - `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_contact_contract_audit_d270/d256_contact_contract_audit_d270.json`
23. `claudedocs/session_20260620_cube10cm_top_view_d256_pose_reset_replay_d262_d264.md`.
24. `claudedocs/session_20260620_cube10cm_top_view_feature_contract_probe_d261.md`.
25. `claudedocs/session_20260619_cube10cm_top_view_tensorboard_gate_d260.md`.
26. `claudedocs/session_20260619_cube10cm_top_view_teacher_rollout_probe_d259.md`.
27. `claudedocs/session_20260619_cube10cm_top_view_ppo_data_prior_smoke_d258.md`.
28. `claudedocs/session_20260618_cube10cm_top_view_state_action_teacher_d257.md`.
29. `claudedocs/session_20260618_cube10cm_top_view_rl_transition_preflight_d256.md`.
30. `claudedocs/cube10cm_top_view_method_pipeline_d254.md`.
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
