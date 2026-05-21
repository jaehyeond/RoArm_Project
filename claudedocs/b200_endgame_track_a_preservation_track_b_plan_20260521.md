# B200 Endgame: Track A Preservation + Track B Plan

Date: 2026-05-21 KST

Scope:

- Track A P7/Branch B continues as the sim/contact/control investigation.
- Track B CoRL/OpenVLA-OFT/pi0 work remains separate.
- B200 availability is expected to end around 2026-05-22 23:59, so Track A
  evidence must be preserved before large Track B runs or B200 release.

## Track B Plan To Keep Separate

User-stated Track B phases:

1. Phase 1: backup pipeline test.
   Target confirmation, rsync 1GB speed measurement, and Track A `/tmp` log
   preservation plan.
2. Phase 2: B200 env setup.
   `openvla-oft` conda, `flash-attn==2.5.5`, and HARD RULE #15 recovery:
   install LeRobot/OpenVLA dependencies first, then force PyTorch nightly cu128
   and verify `sm_100`.
3. Phase 2 smoke: 1K smoke test.
   `action_dim=6`, image source `top`, loss curve, and time/step measurement.
4. Phase 3: OpenVLA-OFT main finetune.
   Decide 30K-50K from smoke time/step; save checkpoints at
   5K/10K/15K/20K/30K/50K.
5. Phase 4: offline eval plus final backup.
   Compare L2, z-score, diversity by checkpoint; backup codebase, best
   checkpoint, train config, and Track A `/tmp` logs.
6. Phase 5: pi0 RunPod handoff plan.
   After B200 release, use RunPod for LeRobot pi0 50K, expected 12-15h, fitting
   the 2026-05-28 paper deadline.

Do not mix Track B training state into Track A P7/Branch B verdicts.

## Observed Local Backup State

Local path `b200_backup_20260521/` exists and is untracked. During inspection it
contained:

- `env.sh`, 2054 bytes;
- `._speedtest_model.safetensors.MIJ5aq`, an rsync-style temporary file whose
  observed size changed during inspection from about 508MB to about 908MB.

Final check later showed only `env.sh` remaining. Do not interpret the transient
temp file as a completed backup artifact, and do not assume the speed test has
completed until a clean target path, final file, elapsed time, and md5 manifest
are recorded.

## Track A Logs To Preserve

Priority Track A `/tmp` logs:

- `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.out`
- `/tmp/p7_branch_b_cube2cm_runtime_jaw_telemetry_v7_d024_b200.err`
- `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.out`
- `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_b200.err`
- `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_python_direct_fail_b200.err`
- `/tmp/p7_branch_b_cube2cm_soft_contact_material_v7_close26_isaaclab_launcher_fail_b200.err`
- `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.out`
- `/tmp/p7_branch_b_cube2cm_virtual_compression_damping_v7_close26_b200.err`
- `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_urdf_prep_b200.out`
- `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_urdf_prep_b200.err`
- `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.out`
- `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_convert_d024_b200.err`

Priority Track A artifacts:

- `/tmp/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/`
- repo files under `sim_scripts/p7_branch_b_cube2cm_*`
- `START_HERE.md`
- `claudedocs/DECISIONS.md`
- `claudedocs/EXPERIMENT_LEDGER.md`
- `claudedocs/session_20260521_p7_branch_b_compliance_direction_analysis.md`
- `claudedocs/p7_branch_b_cube2cm_failure_mode_register.md`

## Backup Pipeline Guardrails

Before any heavy Track B run:

1. Confirm the backup target path and available storage.
2. Ensure no rsync temp file is actively growing in `b200_backup_20260521/`.
3. Run the 1GB rsync speed test only after target confirmation.
4. Copy Track A `/tmp` logs with md5 manifests.
5. Preserve code and docs in one timestamped backup tree.
6. Do not use `HANDOFF.md` or `TASKS.md`.
7. Do not delete existing dirty/untracked files unless explicitly requested.

## Track A Next Work

Track A does not need another runtime now. The approved virtual runtime already
proved that speed damping can work while active, but failed target-error and
support/damping horizon.

Next Track A work is static/code-first:

- target-error controlled close progression below the fixed 3mm gate;
- support/damping retention beyond step 4;
- no attach, posewrite, constraints, SurfaceGripper, transport/release, or gate
  tuning;
- any future runtime requires separate approval.
