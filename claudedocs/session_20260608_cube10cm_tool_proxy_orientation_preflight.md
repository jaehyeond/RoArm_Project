# 2026-06-08 cube10cm tool/contact-proxy orientation preflight

## Scope

- Active branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier.
- Not Track A, not grasp, not dataset generation, not RL, not RoArm deployment.
- No B200/SSH/pull/.ssh work.
- No new IsaacLab/GPU runtime in this preflight.

## Question

The next approved research direction was local-only tool/contact-proxy + orientation-path preflight, with GPU/IsaacLab only if the local evidence required it.

The concrete question:

- Is the teacher proxy the current hand TCP, the gripper collision mesh, link5 collision, or a pose/orientation-aware proxy?
- Does this justify a tiny GPU runtime now?

## Evidence Checked

Code contracts:

- `sim_scripts/cube3cm_push_diffik_probe.py` constructs `DifferentialIKControllerCfg(command_type="position", ik_method="dls")`.
- The same probe computes targets as either `side_center` or `top_margin`.
- It maps a target point to link5 target with a local offset.
- `measured_contact_now` is defined from cube displacement, not a physical contact sensor.
- `roarm_rl/roarm_cube_push_env.py` defines `tcp_cube_dist` as distance between TCP point and cube center.
- Local IsaacLab source supports `DifferentialIKControllerCfg(command_type="pose")`, but the current probe does not use it.

Assets:

- URDF `link5` collision uses `meshes/link5.stl`.
- URDF `gripper_link` collision uses `meshes/gripper_link_collision_g2a.stl`.
- `hand_tcp` is a fixed link from link5 at `xyz="0 0 0.115428"`.

## Added Audit

Added:

- `sim_scripts/cube10cm_tool_contact_proxy_orientation_preflight.py`

Outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tool_contact_proxy_orientation_preflight.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tool_contact_proxy_orientation_preflight_summary.out`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tool_contact_proxy_orientation_preflight_topmargin_seed962.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_tool_contact_proxy_orientation_preflight_topmargin_seed962_summary.out`

## Side-Center seed962 Result

Summary:

- Line 1: local audit only; no GPU/runtime/data/training/robot/SSH.
- Line 2: 1568 trace rows, 16 first-contact envs, source lines 962-1028.
- Line 3: current runtime is `command_type=position`, `ik_method=dls`, `tcp_height_mode=side_center`; hand TCP offset z is `0.115428`; gripper collision bbox is only `0.004m` cube; link5 collision bbox is about `0.0465 x 0.0355 x 0.1206m`.
- Line 4: FK reconstruction is trustworthy: p95 TCP reconstruction error `0.000001505m`.
- Line 5: current hand TCP is not side-center: side-center distance mean `0.056645103m`, z error mean `0.048793540m`, near-10mm rate `0.0`.
- Line 6: gripper collision is not the proxy: side-center distance mean `0.094208332m`, z error mean `0.090346870m`, cube AABB overlap `0.0`.
- Line 7: link5 collision is closer but still upper/offset: side-center distance mean `0.040192105m`, z error mean `0.034543147m`, side-top distance mean `0.025700774m`, overlap `1.0`.
- Line 8: best physical proxy is stable across 16/16 rows: `mesh_mode=link5_collision`, `label_mode=link5_collision:corner_011`, but near10/near20 are both `0.0`.
- Line 9: retargeting that link5 corner would reduce current link5 target error from `0.053769121m` to `0.040192105m` (`0.748512386x`), but requires link5 z target delta `-0.034543147m`.
- Line 10: orientation is not validated: current controller is position-only and trace has no link5 quaternion.
- Line 11: `link5_proxy_candidate_promising=True`, but verdict class is `LINK5_PROXY_REDUCES_TARGET_ERROR_BUT_ORIENTATION_AND_CONTACT_SEMANTICS_UNVALIDATED`.
- Line 12: dataset/RL/RoArm remain blocked; next is local code preflight before GPU.

## Top-Margin Comparison

The same preflight was run on the existing top-margin negative-control trace.

Key result:

- FK reconstruction remains trustworthy.
- Link5 collision remains the best mesh proxy, but proxy target feasibility is worse: current link5 target error `0.011370077m` would become `0.043058712m`, ratio `3.801698970`.
- Verdict is `NO_STABLE_SIDE_CONTACT_PROXY_FROM_EXISTING_TRACE`.

Interpretation:

- Top-margin is still just a tracking shortcut.
- It does not define the selected teacher path.

## Critical Interpretation

`measured_contact_now` should not be read as "the gripper touched the cube." In the probe it is created when cube displacement crosses the threshold. That is valid reaction/event evidence, but not a physical contact-source label.

The physical proxy evidence says:

- Hand TCP is too high/offset.
- Gripper collision is far from the side-center and does not overlap the cube.
- Link5 collision overlaps the cube and is the stable physical proxy, but it is still upper/offset.
- Retargeting `link5_collision:corner_011` may reduce IK target error in the side-center trace, but orientation and semantics are unvalidated.

## Decision

Do not run GPU/IsaacLab yet.

Next step:

1. Local code preflight for a `link5_collision:corner_011` proxy mode.
2. Decide whether it should be position-only proxy targeting or pose-command proxy targeting.
3. Add trace support for the chosen proxy before runtime.
4. Only after that, consider exactly one tiny local IsaacLab runtime.

Dataset/RL/RoArm remain blocked.
