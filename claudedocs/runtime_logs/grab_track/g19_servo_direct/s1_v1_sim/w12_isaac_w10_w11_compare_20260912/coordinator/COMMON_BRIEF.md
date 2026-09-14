# W12 worker common brief — read in full before work

MAIN=/home/cgxr/Documents/Robotics/RoArm_Project. All MAIN files are read-only to workers.
Read MAIN/AGENTS.md, START_HERE.md, claudedocs/DECISIONS_ACTIVE.md, LEDGER_RECENT.md, session_20260912_w11_dt_sensitivity.md, session_20260911_video_w9_w10_review.md and the W9/W10/W11 reports linked below. Your checkout state docs may be stale. No edits to any START_HERE, DECISIONS*, LEDGER*, EXPERIMENT_LEDGER, session_*.md, relay, MEMORY, or AGENTS in either checkout. Root exclusively owns state ledgers. You are not alone; never revert others or overwrite existing outputs.

New variables []: saved-result replay and comparison only. User authorizes actual Isaac scene replay of W10 and W11, not new DEME or PhysX particle solving, training, A/B/C discharge, physical robot queries/drive/PID/torque/camera acquisition. No package install, commit, push, reset or existing-file deletion. Keep isaaclab numpy1.26.0 psutil5.9.8 rerun0.34.1. Only render worker may launch one Isaac app at a time; audit worker CPU-only. No nested agents. Models must remain requested; report mismatch instead of substituting.

## Frozen sources (absolute MAIN prefix)

- Original W9 renderer: MAIN/sim_isaac_render_deme_scoop.py; read only, do not run with defaults (defaults overwrite W9 and use W8). Read hardware-named imports for safe pure geometry; never instantiate serial/cameras.
- W9 report: MAIN/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w9_isaac_render_deme/REPORT_w9.md and gates_w9.json. W9 replayed W8 (154 capture), NOT W10.
- W10 root: MAIN/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/; REPORT_w10.md; input cell_DE_dt2e6_c/.
- W11 root: MAIN/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/; REPORT_w11.md, comparison.json, analysis_w11.py; input cell_dt1e6_seed460/.
- Each cell: scoop_s1_seed460.json, scoop_s1_seed460.npz, timeline_seed460.json, render_timeline_seed460.npz, _obj/. Resolve actual files from source JSON, not stale prereg paths.
- Pile: /home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz
- USD: MAIN/local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd (resolve dependencies read-only).
- Existing W11 full Rerun exporter/validator and verify_rrd_coverage.py can be read/reused as reference, never rerun into original outputs.

W10/W11 each have 64 particle frames but 2774/2771 sync decision rows; do not pretend particle frames cover every dt. Preserve source time seconds, no per-run phase normalization that hides timing differences. For close/reclose event snapshots, annotate nearest saved particle time delta and exact tool event time. Original JSON/NPZ remain scientific authority, Rerun Float32 spatial copies inspection only.

Expectations are NOT proof: W10/W11 capture 541/517; first close both servo_stall; reclose W10 servo_stall vs W11 pinch_guard. Verify raw rather than copying these numbers. W11 did not establish convergence or real-world equivalence.

W9 mapping origin [0.35,0,0.163]m is a project display mapping, not calibrated real surface. Preserve same mapping for both. W9 source converts lip169.6 to lip166.6 and solves IK; checks actual link5/lip/hinge after scene steps. Nominal door_deg differs from actual saved DEME door quaternion: display actual source door geometry if possible and report joint-angle nominal/actual/rendered explicitly. Do not silently move particles with mapped robot to hide IK mismatch. First-close-only K_CAP logic must be extended to reclose/final diagnostics. Source frames must be replayed intact.

Installed stack previously verified Isaac Sim5.1.0.0 / runtime5.1.0-rc.19+release.26219.9c81211b.gl, IsaacLab2.3.0. Reverify installed relevant versions without installing. For API claims use version-matched official docs plus installed source: https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.write_joint_state_to_sim ; installed articulation.py at isaaclab/source/isaaclab/isaaclab/assets/articulation/articulation.py. This writes joint state to simulator: robot kinematic display does not validate actuator feasibility; particle PointInstancer display does not perform Isaac contact simulation.

Output only under OWN_WORKTREE/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/{replay or audit}/. Use new trial subpaths for repair; never overwrite a failed attempt's evidence. Put own runnable verification and GATES/report there, not main state docs. Send acceptance heartbeat first with exact model, cwd, output, task scope. Use the injected Orca Task/Dispatch preamble for heartbeat, ask and worker_done; no ordinary text-only completion. Report absolute output paths, commands, checks, deficiencies and visual observations in Korean. Root will reverify all results.
