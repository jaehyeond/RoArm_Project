# Session 2026-05-23 — B200 disconnected: local/RunPod continuation plan

## TL;DR

The user reported on 2026-05-23 KST that B200 has ended and now shows
`disconnect`. This confirms the transition from "retiring soon" to "unavailable".

Do not attempt B200 SSH or any B200 recovery-by-access in future sessions unless
the user explicitly provides a new valid allocation. Continue from verified
local backups and local/RunPod compute.

## Current Verified Base

The final B200 backup verification remains:

- `claudedocs/session_20260522_b200_retirement_track_a_b_backup_verified.md`
- `b200_backup_20260522_final/README_BACKUP.md`
- `claudedocs/DECISIONS.md` D087-D088

Key preserved artifacts:

- Track A `/tmp/p7_branch_b_*`: 494 files, path+size hash
  `c308d1a682560cf51136cdd1a018c50ce2e7b488f1a0d4620e31abf7de80cfd4`,
  content aggregate
  `cca0586b77c36ee79532d0640f9a35b2f1056654ab2758f256ea2bc1f149a4ae`.
- Track A B200 `sim_scripts`: 53 files, path+size hash
  `98563bbc3d27426351abd13272a88537009372b2c709b46d2a5021560c5ea23a`,
  content aggregate
  `fefe4c873c1e45ec4cb95226a2c1a0d53860e4eca926c93d3da1b9887c9ca83f`.
- Track B outputs are split across:
  - `b200_backup_20260522_final/outputs`
  - `b200_backup_20260521`
  - `openvla_oft_b200_pulls`
- Complete OpenVLA full checkpoints are in `openvla_oft_b200_pulls`, not in
  `b200_backup_20260522_final/outputs/openvla_oft_v6_b200`.

## What To Do Next

Recommended next session priority:

1. Boot from `CLAUDE.md` and `START_HERE.md`; do not use `HANDOFF.md` or
   `TASKS.md`.
2. Confirm local dirty state only; do not revert existing dirty/untracked files.
3. Do not run any `ssh JHPark` or B200 command.
4. Pick one track explicitly:
   - Track B P5 local deploy continuation if the goal is robot deploy:
     local reboot/CUDA verification, OpenVLA 7B cache fixup, GPU dry-run,
     Kinect dry-run, then user-approved real robot deploy.
   - Track A local/static continuation if the goal is sim contact primitive:
     design active target/support recovery after v6 projected block. No dataset,
     PPO/training, hold-lift, transport/release, constraints, SurfaceGripper, or
     gate tuning until close_26 audit PASS.
   - RunPod continuation if the goal is new training/large compute:
     create a fresh pod, copy only required local artifacts, rebuild env from
     backups, run smoke tests, then train/eval.

## Critical Non-Claims

- Track A is not solved. v6 audit still reports
  `SOFT_CONTACT_RUNTIME_CRITERIA_PASS=NO`.
- Track B P5 real deploy is not complete. B200 backup completion is not a robot
  deployment result.
- `.ssh` secrets are not research data and should not be copied or requested.
