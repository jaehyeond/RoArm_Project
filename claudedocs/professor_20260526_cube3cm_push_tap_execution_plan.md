# 2026-05-26 Professor Cube 3cm Push/Tap Execution Plan

## Purpose

Professor request: before grasping, if the endpoint is known, move near a
3cm cube in IsaacLab, hit/push it many times, then inspect output values, code
structure, training-result shape, and robot action outputs.

The goal is not to claim success. The goal is to run the research direction in
small falsifiable steps and preserve what happened.

## Verified Step 1 - Scripted Physics Rollout

Evidence:

- `runtime.out:20` confirms local IsaacLab, 1024 envs x 20 episodes, 20,480
  trials, 3cm cube, no grasp, no attach/object posewrite, no training, no
  dataset generation.
- `runtime.out:21` confirms action semantics: 6D normalized joint-delta,
  `robot_dof_targets += action_scale(0.100) * action`, clip `[-1,1]`,
  gripper open.
- `runtime.out:42` reports 20,480 trials, `ik_ok_rate=1.0000`,
  `disp_xy_mean_m=0.031809`, `disp_xy_p95_m=0.089702`,
  `moved_5mm_rate=0.8774`, `push_positive_1mm_rate=0.9086`, zero
  action saturation, zero grasp/attach/posewrite.
- `rollout_stats_audit.out:1-4` cross-checks CSV rows, summary metrics,
  mechanism separation, and action scale.
- `rollout_stats_audit.out:5-21` shows caveats: outliers, direction asymmetry,
  and low-motion cases.

Output files:

- `runtime.out`: run metadata, action semantics, per-episode summary, final
  summary.
- `summary.json`: machine-readable run summary.
- `per_env.csv`: one row per trial with initial cube pose, push direction,
  displacement, speed, tip angle, action statistics, and grasp marker.
- `rollout_stats_audit.out`: independent CSV/summary/mechanism cross-check.
- `controlled_push_filter_audit.out`: posthoc controlled-push vs impact split.
- `controlled_push_filter_summary.json`: machine-readable filtered summary.

## Verified Step 2 - Controlled Push Filter

Evidence:

- `controlled_push_filter_audit.out:1` confirms 20,480 CSV rows and source md5s.
- `controlled_push_filter_audit.out:3-7` cross-checks filter thresholds against
  `rollout_stats_audit.out` p95/p99 values.
- `controlled_push_filter_audit.out:8-12` separates buckets:
  valid basic 100%, low-motion 12.26%, controlled-push 80.75%, impact-outlier
  2.67%, failure-not-controlled 19.25%.
- `controlled_push_filter_audit.out:13` shows outlier removal effect:
  mean displacement `0.031809m -> 0.030130m`, std `0.028320m -> 0.024638m`.
- `controlled_push_filter_audit.out:14-17` shows direction weakness. `(0,1)`
  is weakest: controlled 63.80%, low-motion 26.72%, impact 4.50%.
- `controlled_push_filter_audit.out:35-39` shows the weakest initial-position
  buckets.

Interpretation:

- Physics response exists: the cube moved through contact/physics, not object
  teleportation.
- Stable-looking pushes are a majority but not universal.
- The `(0,1)` direction and high-x / positive-y initial positions need the next
  experiment to be stratified, not averaged away.

## Step 3 - No-Attach Cube-Push RL Stage 0

Reason:

- The existing Pick/Stack PPO path is not valid evidence for this professor
  request because it can use grasp attach / object posewrite.
- A learned result, even a failed smoke result, must come from a separate
  no-attach cube-push task.

New code path:

- `roarm_rl/roarm_cube_push_env.py`: 3cm cube DirectRLEnv-style task using the
  existing robot/action scaffold but overriding attach as no-op.
- `roarm_rl/__init__.py`: registers `RoArm-CubePush-Direct-v0`.
- `roarm_rl/train_cube_push_ppo.py`: separate PPO entry point for this task.
- `sim_scripts/cube3cm_push_rl_stage0_static_audit.py`: local static audit
  before any Isaac/PPO runtime.

Reward shape:

- Positive progress along sampled push direction.
- Positive target score for moving toward cube-start + push direction target.
- Controlled-push bonus using the prior rollout p95/p99 thresholds.
- Penalty for impact outliers, excessive gripper closing, action magnitude, and
  TCP-cube distance.
- Success is a metric latch, not an episode terminator.

Expected training-result shape:

- PPO log directory with scalar logs such as `cube_push_disp_along_m`,
  `cube_push_controlled_rate`, `cube_push_impact_rate`,
  `cube_push_low_motion_rate`, and `cube_push_success_rate`.
- RSL-RL checkpoints, if the smoke run reaches a save interval.
- A post-training rollout/audit must still be run before claiming anything.

## Next Runtime Step After Static PASS

One small PPO smoke was run after local static audit PASS:

```bash
python -m roarm_rl.train_cube_push_ppo \
  --num_envs 256 \
  --max_iterations 10 \
  --experiment_name cube_push_no_attach_smoke_20260526 \
  --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_smoke_logs \
  --robot_usd_path b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd
```

Runtime/audit evidence:

- `ppo_smoke_exit_code.txt:1` records `exit_code:0`.
- `ppo_smoke_stdout.out:3-8` records no-attach cube-push scope, 6D action
  semantics, 256 envs, 10 PPO iterations, local backup USD, and log directory.
- `ppo_smoke_stdout.out:11` confirms CUDA device use.
- `ppo_smoke_audit.out:2` confirms TensorBoard event file plus `model_0.pt`
  and `model_9.pt`.
- `ppo_smoke_audit.out:9` confirms stdout visibly printed iterations 0-4 and
  TensorBoard event scalars cover 10 iterations.
- `ppo_smoke_audit.out:13-17` shows early signal and caveats:
  controlled rate `0.0550 -> 0.3444`, low-motion rate `0.9051 -> 0.3672`,
  success rate only `0.0055 -> 0.0181`, impact rate `0.0000 -> 0.0039`,
  grasp marker stayed `0.0000`.
- `ppo_smoke_audit.out:20-21` explicitly records:
  training loop ran, learned policy success NO, evaluation rollout NO,
  dataset generation NO, Track A grasp success NO, verdict
  `SMOKE_SIGNAL_PRESENT_BUT_WEAK_NO_SUCCESS_CLAIM`.

The smoke verdict should be reported as one of:

- `RUNTIME_BLOCKED`: infrastructure/import/GPU failure.
- `TRAINING_RAN_BUT_NO_LEARNING_CLAIM`: PPO loop executed, but no evaluation yet.
- `SMOKE_SIGNAL_PRESENT`: logs show improving displacement/controlled rate.
- `SMOKE_SIGNAL_ABSENT`: PPO ran, but metrics did not improve.
- Actual current verdict: `SMOKE_SIGNAL_PRESENT_BUT_WEAK_NO_SUCCESS_CLAIM`.

None of these means grasp success, Track A success, dataset readiness, PPO
success at scale, or VLA success.

## Step 4 - Model 9 Policy Rollout Cross-Check

The `model_9.pt` checkpoint was evaluated separately from the training loop:

```bash
python -m roarm_rl.eval_cube_push_policy \
  --checkpoint claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_smoke_logs/cube_push_no_attach_smoke_20260526/model_9.pt \
  --num_envs 256 \
  --num_rollouts 2 \
  --seed 321 \
  --robot_usd_path b200_backup_20260522_final/tmp_p7/p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd \
  --out_csv claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_smoke_eval_model9.csv \
  --summary_json claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_smoke_eval_model9_summary.json
```

Evaluation evidence:

- `ppo_smoke_eval_model9_audit.out:1` confirms exit code 0, 512 CSV rows, and
  summary/CSV md5s.
- `ppo_smoke_eval_model9_summary.json:3-18` reports 512 trials,
  controlled-push rate `0.2754`, mean push-aligned displacement `0.000917m`,
  mean XY displacement `0.014053m`, low-motion rate `0.5020`,
  success-marker rate `0.0391`, and grasped-marker rate `0.0000`.
- `ppo_smoke_eval_model9_audit.out:4` confirms training false, dataset false,
  grasp attach false, rollout object posewrite false, and grasp marker 0.
- `ppo_smoke_eval_model9_audit.out:5` compares training vs eval:
  train controlled last `0.3444` vs eval `0.2754`, train low-motion last
  `0.3672` vs eval `0.5020`, train success last `0.0181` vs eval `0.0391`.
- `ppo_smoke_eval_model9_audit.out:10` verdict:
  `EVAL_RAN_WEAK_POLICY_NO_SUCCESS_CLAIM`.

Interpretation:

- The professor's direction has now been executed through scripted rollout,
  controlled filter, no-attach PPO smoke, and frozen-policy rollout.
- The learned policy signal is weak. It is a real run, not a success result.
- The next research step is not a bigger claim; it is a better reward/curriculum
  or a longer controlled training run followed by the same eval audit.

## Step 5 - Reward/Curriculum Follow-Up

We then tested the professor's endpoint-known idea as a separate no-attach
learning branch:

- `roarm_rl/roarm_cube_push_env.py:42-113` now defines a 3cm cube-push task with
  IK endpoint reset, clean success target tolerance, speed gate, low-motion,
  reverse, target-distance, lateral, overshoot, speed, and impact penalties.
- `roarm_rl/roarm_cube_push_env.py:346-352` defines clean success as controlled,
  non-impact, push-aligned, near target, and below `0.5m/s`.
- `roarm_rl/roarm_cube_push_env.py:378-391` terminates on clean success or
  terminal impact.
- `cube_push_rl_stage0_static_audit.out:14-17` confirms clean success gate,
  impact/far-target termination, overshoot/lateral penalties, and speed guard.

Three 50-iteration PPO variants were run and then checked with frozen 1k eval:

1. IK curriculum:
   `ppo_ik_curriculum_50iter_audit.out:12-13` confirms IK reset active and
   accurate, but `ppo_ik_curriculum_50iter_audit.out:4-8` shows the policy
   became impact-heavy and direction-regressed. Frozen eval verdict:
   `MODEL49_EVAL_RAN_IMPACT_HEAVY_NO_SUCCESS_CLAIM`.
2. Clean reward:
   `ppo_clean_reward_50iter_audit.out:18` shows training-log improvement
   (`disp_xy 0.604205787 -> 0.025637908`, impact `0.351481140 -> 0.003580729`),
   but frozen eval still had impact `0.283282675`:
   `ppo_clean_reward_model49_eval1024_audit.out:3-6,18`.
3. Speed guard:
   `ppo_speed_guard_50iter_stdout.out:5` confirms action scale `0.050`, but
   frozen eval still had impact `0.286984127`:
   `ppo_speed_guard_model49_eval1024_audit.out:3-6,17`.

Current learned-policy interpretation:

- IK endpoint/pre-contact setup works.
- No-attach PPO execution works.
- Reward changes reduce far-fling behavior in training logs.
- Frozen eval still has about `28-29%` high-speed impact, so this is **not** a
  clean learned push success and should not be scaled to 10k/100k yet.

Next valid research step:

- Add velocity-limited or action-smoothed push control, contact-speed curriculum,
  or an IK/scripted teacher warm-start.
- Then run another 50-100 iteration PPO and the same frozen 1k audit.
- Only after 1k eval impact falls below about 5% while controlled/clean success
  remains meaningful should 10k/100k evaluation be run.

## Step 6 - Velocity/Teacher Follow-Up Results

What we ran next:

- Action-smoothed/velocity-limited PPO (V4) and frozen 1k eval.
- Scripted-teacher warm-start PPO (V5) and teacher-off frozen 1k eval.
- Policy-only contact-speed curriculum PPO (V6) and frozen 1k eval.
- Teacher-on IK/scripted diagnostic eval to isolate whether the scripted teacher
  itself is safe enough.

Key evidence:

- V4 frozen eval improved impact versus speed guard but still had impact
  `0.245576787`: `ppo_smooth_limit_model49_eval1024_audit.out:3-17`.
- V5 teacher-assisted training looked good, but teacher-off eval regressed:
  impact `0.257686676`, clean success `0.095168375`, verdict
  `TEACHER_MODEL49_EVAL_TEACHER_OFF_NO_TRANSFER_NO_10K`:
  `ppo_teacher_warmstart_model49_eval1024_audit.out:3-17`.
- V6 contact-speed training reduced training impact to `0.000813802`, but also
  had high low-motion `0.377115905`:
  `ppo_contact_speed_50iter_audit.out:4-27`.
- V6 frozen eval was the best learned-policy impact so far but still failed:
  impact `0.153782895`, clean success `0.110197368`, verdict
  `CONTACT_SPEED_MODEL49_EVAL_IMPROVED_BUT_NO_10K`:
  `ppo_contact_speed_model49_eval1024_audit.out:3-18`.
- Teacher-on diagnostic was not a safe teacher: impact `0.162448980`, clean
  success `0.067755102`, verdict
  `TEACHER_ON_DIAGNOSTIC_UNSAFE_OR_WEAK_NOT_LEARNED_NO_10K`:
  `ppo_contact_speed_teacher_on_eval1024_audit.out:3-18`.

Interpretation for professor briefing:

- We did run the "known endpoint / IK / push near cube" idea in IsaacLab and
  turned it into no-attach PPO variants.
- The strongest current statement is not "success"; it is "we found the failure
  surface." Contact-speed control reduces violent impact, but the learned policy
  becomes weak/low-motion, and the current scripted teacher is itself not clean
  enough to serve as a reliable teacher.
- Therefore the next step is not 10k/100k. It is to redesign the contact teacher
  trajectory or add actual imitation/resume fine-tuning, then repeat 50-100 iter
  plus frozen 1k audit.

## Step 7 - Locked Next Direction: IsaacLab Built-In Differential IK Probe

Clarification:

- The professor's "if the endpoint is known, use IK to go near the cube first"
  should be treated as an IK/task-space instruction, not as an FK-first
  instruction.
- The current RoArm cube-push env uses RoArm-local `ik_dls` to solve endpoint
  targets before writing joint targets into IsaacLab. That is useful IK evidence,
  but it is not the same as using IsaacLab's built-in Differential IK controller.

Locked next action:

1. Add a scoped cube-push probe using IsaacLab's built-in
   `DifferentialIKController` / live Jacobian path.
2. Feed TCP/end-effector targets near the 3cm cube: pre-contact point, contact
   point, short push-through point, and hold/settle.
3. Let IsaacLab compute joint targets from current end-effector pose, Jacobian,
   and current joint state.
4. Keep no grasp, no attach/object posewrite, no dataset generation, and no
   learned-policy success claim.
5. Run a small smoke first, then produce `runtime.out`, `summary.json`,
   `per_env.csv`, and a posthoc audit before considering larger counts.

Reason:

- This matches the professor's wording more literally than the existing
  RoArm-local IK path.
- It tests whether IsaacLab's own task-space IK machinery can make the known
  endpoint push/tap primitive cleaner before returning to PPO/reward scaling.

## Step 8 - Differential IK Probe Result: 1024 Headless Eval Complete

What ran:

- Added `sim_scripts/cube3cm_push_diffik_probe.py`, which uses IsaacLab's
  built-in `DifferentialIKController` with `command_type="position"` and
  `ik_method="dls"`, plus live PhysX Jacobians.
- It sends TCP targets near the 3cm cube, writes robot joint targets only, and
  audits no grasp, no attach/object posewrite, no training, and no dataset.

Key checks:

- The short 16-env smoke was not enough: mechanism PASS but low-motion
  `1.000000000` and final TCP target error `0.161282191m`.
- After increasing approach/horizon and DiffIK joint-step budget, the 16-env
  reach smoke produced controlled `0.937500000`, impact `0`, low-motion
  `0.062500000`.
- Frozen 1024 eval then ran headless in IsaacLab. Audit lines 1-2 confirm
  `controller=IsaacLab_DifferentialIKController`, RoArm-local IK control loop
  false, no training/dataset/grasp/posewrite, auto-reset disabled, and action
  loop bypassed for the scripted DiffIK probe.
- Audit lines 3-5 report controlled `0.892578125`, impact `0.023437500`,
  low-motion `0.136718750`, success marker `0.520507812`,
  `disp_xy_mean_m=0.034856980`, max speed `1.931515932m/s`, final TCP target
  error mean `0.028779610m`, and DiffIK clip rate mean `0.658035710`.
- Posthoc lines 3-7 show direction `(1, 0)` is weakest; posthoc line 8 marks
  initial grid `(1, 1)` as the worst low+impact pocket.

Interpretation:

- This is the first clean local evidence for the professor's literal direction:
  known endpoint/TCP target -> IsaacLab built-in Differential IK -> physical
  cube push/tap statistics.
- It is not a learned PPO/VLA policy and not Track A grasp success.
- The next research step should be direction/position-specific trajectory
  correction before 10k/100k scaling or before treating it as a teacher.
