# 2026-05-26 - 3cm cube push/tap rollout probe for professor request

## Scope

User relayed the professor's near-term request: if the endpoint is known, do not
start with grasping. In Isaac Lab, move near a 3cm x 3cm x 3cm cube and hit/push
it, then run thousands or tens of thousands of trials and inspect output values,
code structure, training-result shape, and robot action outputs.

This session therefore created a separate local push/tap rollout probe. It is
not Track A close_26, not grasp, not hold-lift, not dataset generation, not PPO,
and not VLA. Track A remains saved separately: first approved v8 close_26 runtime
failed and the post-fail damping wiring fix is static-ready only.

No B200 SSH/reconnect/pull or `.ssh` copying was used. No RunPod pod was used.
GPU/IsaacLab commands were run locally with escalated Codex execution because the
default sandbox hides `/dev/nvidia*`.

## Code and mechanism

Added:

- `sim_scripts/cube3cm_push_rollout_probe.py`
- md5 `8d329b79106e7ca2c03fa91b7ac87170`

Important source facts:

- Script lines 2-11 state the scope: parallel 3cm cube push/tap rollout, not
  grasp/hold-lift/dataset/PPO/VLA, and cube motion during rollout must come only
  from physics.
- `roarm_rl/roarm_stack_env.py:106-112` defines `action_space = 6` and
  `observation_space = 28`.
- `roarm_rl/roarm_stack_env.py:484-489` clamps actions to `[-1, 1]` and updates
  joint targets as `robot_dof_targets + action_scale * actions`.
- `roarm_rl/roarm_stack_env.py:491-498` applies the robot target through
  `_robot.set_joint_position_target(...)` and would call attach if `_grasped` is
  true.
- `roarm_rl/roarm_stack_env.py:501-529` shows the observation vector contains
  scaled joint positions, joint velocities, cube/object pose, TCP-to-object, and
  object-to-target terms.
- `roarm_rl/roarm_stack_env.py:1216-1236` is the original hidden attach path that
  writes object pose to the sim. The new probe monkeypatches this to a
  counter-only no-op.
- Probe lines 291-305 install the attach/posewrite counters and no-op attach.
- Probe lines 307-319 print the no-grasp/no-attach/no-training metadata and the
  robot action semantics.
- Probe lines 482-511 write the summary metrics and explicitly preserve
  `training=False`, `dataset_generation=False`, `grasp=False`,
  `attach_posewrite=False`, `rollout_object_posewrite=False`, attach-call count,
  and posewrite-call count.

Existing training code structure, not run here:

- `roarm_rl/train_ppo.py:126-135` launches IsaacLab, Gym, roarm env registration,
  RSL-RL wrapper, and `OnPolicyRunner`.
- `roarm_rl/train_ppo.py:137-148` selects Pick or Stack env.
- `roarm_rl/train_ppo.py:228-240` builds PPO config and experiment name.
- `roarm_rl/agents/rsl_rl_ppo_cfg.py:16-42` defines the PPO actor/critic and
  algorithm hyperparameters.

Critical distinction: this session produced scripted rollout statistics, not a
trained policy and not PPO learning curves/checkpoints. A real learning result
for the professor's push/tap task requires a separate no-attach cube-push
DirectRLEnv or equivalent reward task before running PPO.

## Runs

### Smoke: 16 trials

Directory:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_smoke/`

Command shape:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n isaaclab python sim_scripts/cube3cm_push_rollout_probe.py \
  --num_envs 16 --episodes 1 --seed 7 \
  --approach_steps 15 --precontact_steps 8 --push_steps 12 --post_steps 8 \
  --save_arrays \
  --out_dir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_smoke
```

Artifacts:

- `runtime.out` md5 `6f69b62d6116e0b4b97956bd3ceacdad`
- `runtime.err` md5 `30b990c1766da0c11a257fc0bec68526`
- `summary.json` md5 `a8e5554589729d534f762815e5b6f663`
- `per_env.csv` md5 `9a1d44a830a05daf04e84f60d776213f`
- `rollout_arrays.npz` md5 `05d51e997d3be507919cc8e6ee3b2da7`

Result:

- Runtime stdout line 20 confirms local Isaac run, 16 trials, 3cm cube, no grasp,
  no attach posewrite, no rollout object posewrite, no training, and no dataset.
- Runtime stdout line 21 confirms action semantics:
  `robot_dof_targets += action_scale(0.100) * action`, action dim 6,
  action clip `[-1, 1]`, gripper target open 0 rad.
- Runtime stdout line 22 reports `ik_ok_rate=1.0000`,
  `disp_xy_mean_m=0.039310`, `moved_5mm_rate=1.0000`,
  `push_positive_1mm_rate=1.0000`, `action_sat_frac_mean=0.000000`,
  `grasped_marker_rate=0.0000`, `attach_calls=0`,
  `posewrite_calls_during_rollout=0`.

### 1,024 trials

Directory:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_1024/`

Artifacts:

- `runtime.out` md5 `5ee8fd54b2079aca2ed01f7c81dbee81`
- `runtime.err` md5 `30b990c1766da0c11a257fc0bec68526`
- `summary.json` md5 `13ca288f9b9dc351a2df8a8b62e9f272`
- `per_env.csv` md5 `20ba0173ba11660377926a1afea4fee8`

Result:

- Runtime stdout line 20 confirms 1024 local trials, 3cm cube, no grasp/attach/
  posewrite/training/dataset.
- Runtime stdout line 21 confirms action semantics.
- Runtime stdout lines 22-23 report `ik_ok_rate=1.0000`,
  `disp_xy_mean_m=0.031509`, `disp_xy_p95_m=0.090220`,
  `moved_1mm_rate=0.9268`, `moved_5mm_rate=0.8799`,
  `moved_10mm_rate=0.8438`, `push_positive_1mm_rate=0.9033`,
  `action_abs_mean=0.086139`, `action_saturation_frac_mean=0.000000`,
  `grasped_marker_rate=0.0000`, `attach_calls=0`,
  `posewrite_calls_during_rollout=0`.

### 5,120 trials

Directory:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_5120/`

Artifacts:

- `runtime.out` md5 `da0e6b41fe1ecab9410f6efeb251d080`
- `runtime.err` md5 `30b990c1766da0c11a257fc0bec68526`
- `summary.json` md5 `a2115b0331317e175636674f3ae392fd`
- `per_env.csv` md5 `487e4a784cd8b1c5393966dee533255b`

Result:

- Runtime stdout line 20 confirms 1024 envs x 5 episodes = 5120 trials, 3cm cube,
  no grasp/attach/posewrite/training/dataset.
- Runtime stdout line 21 confirms action semantics.
- Runtime stdout lines 22-26 show episode-level repeatability: every episode has
  `ik_ok_rate=1.0000`, zero action saturation, zero grasp marker, zero attach
  calls, and zero rollout posewrite calls.
- Runtime stdout line 27 summary: `total_trials=5120`,
  `disp_xy_mean_m=0.031767`, `disp_xy_p95_m=0.088972`,
  `moved_1mm_rate=0.9242`, `moved_5mm_rate=0.8750`,
  `push_positive_1mm_rate=0.9062`, `action_abs_mean=0.086341`,
  `action_saturation_frac_mean=0.000000`, `grasped_marker_rate=0.0000`,
  `attach_calls=0`, `posewrite_calls_during_rollout=0`.

### 20,480 trials

Directory:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/`

Artifacts:

- `runtime.out` md5 `2aad344f08f95c880e43bc0d7f655998`
- `runtime.err` md5 `30b990c1766da0c11a257fc0bec68526`
- `summary.json` md5 `5c9278450b5531afb7b0ca2a1fed46ee`
- `per_env.csv` md5 `4c2864301bea8e2ae798a8f77adf23ab`
- `rollout_stats_audit.out` md5 `3e0096ba54e7cc0ec0e55b1b26a50b8e`

Result:

- Runtime stdout line 20 confirms 1024 envs x 20 episodes = 20,480 trials,
  local backup USD, 3cm cube, no grasp, no attach posewrite, no rollout object
  posewrite, no training, and no dataset generation.
- Runtime stdout line 21 confirms action semantics:
  normalized 6D joint-delta actions, target update
  `robot_dof_targets += action_scale(0.100) * action`, clip `[-1, 1]`,
  gripper target open 0 rad.
- Runtime stdout lines 22-41 show all 20 episode summaries. Every episode has
  `ik_ok_rate=1.0000`, `action_sat_frac_mean=0.000000`,
  `grasped_marker_rate=0.0000`, `attach_calls=0`,
  and `posewrite_calls_during_rollout=0`.
- Runtime stdout line 42 summary:
  `total_trials=20480`, `ik_ok_rate=1.0000`,
  `disp_xy_mean_m=0.031809`, `disp_xy_p95_m=0.089702`,
  `moved_1mm_rate=0.9267`, `moved_5mm_rate=0.8774`,
  `push_positive_1mm_rate=0.9086`, `action_abs_mean=0.086382`,
  `action_saturation_frac_mean=0.000000`, `grasped_marker_rate=0.0000`,
  `attach_calls=0`, `posewrite_calls_during_rollout=0`.
- `rollout_stats_audit.out` line 1 cross-checks row count: 20,480 CSV rows and
  20,480 summary trials.
- Audit line 2 matches the summary rates:
  `disp_xy_mean_m=0.031809252`, `disp_xy_p95_m=0.089701957`,
  `moved_1mm_rate=0.926660`, `moved_5mm_rate=0.877441`,
  `moved_10mm_rate=0.839795`, `push_positive_1mm_rate=0.908594`.
- Audit line 3 confirms mechanism separation: grasp/training/dataset false,
  attach/object posewrite false, attach calls 0, posewrite calls 0.
- Audit line 4 converts action scale: mean normalized action magnitude
  `0.086382226`; with action scale 0.1 rad this is mean target delta
  `0.008638223rad = 0.494934deg` per control update.
- Audit lines 5-11 provide distribution stats. Important risk flags:
  `disp_xy_max_m=0.521036748`, `max_cube_speed_mps max=4.549609073`,
  `tip_angle_deg max=179.981780282`, and `q_err_max_deg max=30.048652272`.
- Audit lines 12-15 show direction asymmetry:
  `(-1,0)` moved 5mm rate `0.988795`,
  `(1,0)` moved 5mm rate `0.944785`,
  `(0,-1)` moved 5mm rate `0.842324`,
  `(0,1)` moved 5mm rate `0.732799` but higher mean displacement due to
  larger/outlier moves.
- Audit line 16 shows low-motion trials: 1502 / 20480 = `0.073340`.
- Audit lines 17-21 show top outliers. Example:
  per-env CSV line 7865 has `disp_xy_m=0.521036748`,
  `speed_mps=4.549609073`; line 6999 has `disp_xy_m=0.503155234`,
  `speed_mps=3.792765159`. These should be treated as physics/impact outliers,
  not normal controlled pushing.

## Interpretation

What is validated:

- The local IsaacLab scene can run thousands to tens of thousands of parallel
  endpoint-driven cube push/tap trials on the local GPU.
- In this probe, the cube moved only through physics during rollout:
  attach calls were 0 and rollout object posewrite calls were 0.
- The robot action output is available as normalized 6D joint-delta commands.
  The 20,480-trial run averaged about 0.086 normalized action magnitude, or
  about 0.495 degrees of joint target change per control update at
  `action_scale=0.1rad`.
- The main output artifacts are `summary.json`, `per_env.csv`, and optional
  `rollout_arrays.npz` for smaller runs.

What is not validated:

- This is not a grasp, not close_26 PASS, not hold-lift PASS, not Track A dataset
  readiness, and not learned policy performance.
- This is not PPO/VLA training output. There are no policy checkpoints or reward
  curves from this probe.
- Large displacement and high speed outliers exist. Those may be valid impacts
  for a "hit the cube" task, but they should not be interpreted as stable,
  controlled pushing without filtering.

## Next concrete work

1. For the professor's tomorrow briefing: present the 20,480-trial rollout table,
   action semantics, output file formats, and the critical caveat that this is
   scripted physics rollout, not learning.
2. If the professor wants a training result, implement a separate no-attach
   cube-push DirectRLEnv with a reward such as projected cube displacement,
   contact/no-contact classification, action penalty, tip/outlier penalty, and
   no object posewrite during rollout.
3. Before training, add an analysis filter that separates controlled pushes from
   impact outliers: e.g. report robust p50/p90/p95, cap/flag speeds above a
   threshold, and stratify by push direction and cube initial position.
4. Keep Track A v8 post-fix close_26 as a separate line. Do not let this push/tap
   probe be cited as Track A grasp success.
