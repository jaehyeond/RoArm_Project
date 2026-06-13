# START_HERE.md

Last updated: 2026-06-13 KST (D238 current truth: manifest-fed 0-99 chunk renderer written and validate-only gate passed; no Isaac render or 100ep/1000ep generation has been run.)

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

## Latest Result: D238

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
- Next valid work, only with a fresh explicit run instruction, is one 100
  episode visual chunk at
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d235`
  after checking local free space is still at least about `25GB`.
- Do not render 1000 episodes from D234. Local 1000/10000 scale remains blocked
  by storage/pipeline constraints.
- Do not run the xy10 useful-tap constant/PPO gate until the tool-contact proxy branch is explicitly closed:
  1. Either accept current `link5_collision_aabb` as the fixed-jaw/distal-tool metric for this sim contract; or
  2. Implement a named `tool_surface_union` contact metric and run zero/base metric-equivalence before any PPO.
- Stop treating the 6mm target-band as the primary success gate if the professor/user goal is "make a useful tap."
- Do not claim D225/D228 target-band weak bins prove useful-tap recovery.
- Do not use fixed single-pose corners as the next PPO stage; D230 says corners pass and the useful failure appears in the randomized xy10 band.
- After the proxy branch is closed, the next valid runtime, only with explicit approval, is a small xy10 useful-tap L1 screen after constant-residual baselines:
  1. Same-band base: useful final/no-overshoot around `0.828125`, contact/reaction `1.0`.
  2. Constant residual baselines on the same xy10 band, using useful/no-overshoot as the primary metric.
  3. Small residual PPO L1 only if constants do not already solve overshoot.
  4. Promotion requires useful/no-overshoot improvement over same-run base and best constant while preserving contact/reaction `1.0`, finite obs/reward/actions, contract cleanliness, and no L2/Large PPO until that passes.

## Must Read First

1. `CLAUDE.md` Current-State Protocol.
2. `claudedocs/DECISIONS.md` D234, D233, and D232 first; D224, D227, D228, D229,
   D230, D231 only if resuming the RL branch.
3. `claudedocs/EXPERIMENT_LEDGER.md` latest D234 row.
4. `claudedocs/session_20260613_cube10cm_top_view_professor_storage_runpod_d234.md`.
5. `claudedocs/professor_view_format_packet_cube10cm_top_view_d234.md`.
6. `claudedocs/session_20260612_cube10cm_top_view_visual_smoke_lerobot_d233.md`.
7. `claudedocs/session_20260612_camera_contract_visual_dataset_disk_audit_d232.md`.
8. `claudedocs/session_20260611_tool_contact_proxy_asset_audit_d231.md`.
9. `claudedocs/session_20260611_useful_tap_poseband_failure_sweep.md`.
10. `claudedocs/session_20260611_useful_tap_objective_reframe.md`.
11. `claudedocs/session_20260610_transition_xband_constant_baseline.md`.
12. Runtime/audit summaries listed in D228-D234.
13. Relevant code:
   - `roarm_rl/train_cube_tap10cm_ppo_smoke.py`
   - `roarm_rl/roarm_cube_push_env.py`
   - `roarm_rl/roarm_stack_env.py`
   - `sim_scripts/cube10cm_tool_contact_proxy_asset_audit.py`
   - `sim_scripts/cube10cm_top_view_visual_smoke_render.py`
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
