# 2026-05-26 - cube3cm push RL reward/curriculum follow-up

## Scope

This session continued the professor's separate 3cm cube push/tap branch. It is
not Track A grasp, not Track A close_26, not hold-lift, not dataset generation,
not VLA, and not Track A training readiness.

The question being tested was: if the cube endpoint is known, can IsaacLab use IK
to bring RoArm near a 3cm cube and learn a directional push/tap behavior from
robot actions only?

No B200 SSH/reconnect/pull or `.ssh` copying was used. GPU/IsaacLab commands were
local and escalated because default Codex sandbox hides `/dev/nvidia*`.

## Starting Evidence

- Scripted 20,480-trial rollout remains the baseline:
  `runtime.out:20` confirms no grasp, no attach/object posewrite, no training,
  no dataset.
- `runtime.out:21` defines robot action semantics as normalized 6D joint-delta:
  `robot_dof_targets += action_scale(0.100) * action`, clip `[-1,1]`.
- `runtime.out:42` reports `total_trials=20480`, `ik_ok_rate=1.0000`,
  `disp_xy_mean_m=0.031809`, `disp_xy_p95_m=0.089702`,
  `moved_5mm_rate=0.8774`, `push_positive_1mm_rate=0.9086`, and zero
  grasp/attach/posewrite.
- `controlled_push_filter_audit.out:10-13` split the scripted run:
  controlled-push rate `0.807471`, impact-outlier rate `0.026709`, and
  non-impact mean displacement `0.030130129m`.
- `controlled_push_filter_audit.out:16` showed the weakest direction remained
  `(0,1)` with controlled-push rate `0.637998`.

## Code Changes

Current no-attach RL code md5s:

- `roarm_rl/roarm_cube_push_env.py` md5
  `b44996c396c099847e5196949ed86742`.
- `roarm_rl/train_cube_push_ppo.py` md5
  `cb8c3303ca10bae2299c4e6f561c240d`.
- `roarm_rl/eval_cube_push_policy.py` md5
  `f91c5107503d4e2f4f41cab7f70cb51a`.

Current key env facts:

- `roarm_cube_push_env.py:42-47` defines a separate `RoArmCubePushEnvCfg` with
  `episode_length_s=1.2` and speed-guard `action_scale=0.05`.
- `roarm_cube_push_env.py:83-96` defines a 4cm target push, 3cm success
  displacement, 5cm target tolerance, 0.5m/s success speed gate, and IK endpoint
  reset controls.
- `roarm_cube_push_env.py:98-113` defines push-aligned, low-motion, reverse,
  target-distance, lateral, overshoot, speed, impact, and success reward scales.
- `roarm_cube_push_env.py:346-352` requires success to be controlled, non-impact,
  push-aligned, near target, and below 0.5m/s.
- `roarm_cube_push_env.py:378-391` terminates on clean success or terminal impact.
- `_apply_action` remains robot-target-only and `_update_grasp_attach` remains a
  no-op; see static audit `cube_push_rl_stage0_static_audit.out:7-9`.

Current static audit:

- `cube_push_rl_stage0_static_audit.out:1` md5s match the current env/train code.
- `cube_push_rl_stage0_static_audit.out:14-17` confirms clean success gate,
  impact/far-target termination, overshoot/lateral penalties, and v3 speed guard.
- `cube_push_rl_stage0_static_audit.out:31` reports PASS, local static only, no
  Isaac runtime, no training run, no dataset generation.

## IK Curriculum V1

Run:

- `ppo_ik_curriculum_50iter_exit_code.txt:1` is `exit_code:0`.
- `ppo_ik_curriculum_50iter_audit.out:2-3` confirms event file, `model_49.pt`,
  and 50 TensorBoard iterations.
- `ppo_ik_curriculum_50iter_audit.out:12-13` confirms IK endpoint reset active
  with reset error about `0.75-0.78mm`.

Result:

- `ppo_ik_curriculum_50iter_audit.out:4-8` shows the failure mode:
  final push-aligned displacement `-0.105634235m`, XY displacement
  `0.604205787m`, target distance `0.610159278m`, and impact rate `0.351481140`.
- `ppo_ik_curriculum_50iter_audit.out:18` verdict:
  `IK_CURRICULUM_RAN_IMPACT_HEAVY_REGRESSED_DIRECTION_NO_SUCCESS_CLAIM`.

Frozen eval:

- `ppo_ik_curriculum_model49_eval1024_audit.out:1` confirms exit 0 and 1024 CSV
  rows.
- `ppo_ik_curriculum_model49_eval1024_audit.out:3-6` reports controlled
  `0.405273438`, impact `0.369140625`, success marker `0.424804688`, but clean
  success marker only `0.148437500`.
- `ppo_ik_curriculum_model49_eval1024_audit.out:10-14` shows top displacement
  outliers up to `9.656235695m`.
- Verdict line 18: `MODEL49_EVAL_RAN_IMPACT_HEAVY_NO_SUCCESS_CLAIM`.

Interpretation:

- Professor's IK endpoint idea worked mechanically: IK reset was active and
  accurate.
- The first reward let policy exploit high-impact/large displacement; not learned
  clean push.

## Clean-Reward V2

Changes:

- Added clean success gate, target-distance penalty, lateral/overshoot penalties,
  impact/far-target termination, and lower controlled bonus.

Run:

- `ppo_clean_reward_50iter_audit.out:1-3` confirms exit 0, model_49, and 50
  event iterations.
- `ppo_clean_reward_50iter_audit.out:18` compares v1 to v2 in training logs:
  XY displacement `0.604205787 -> 0.025637908`, target distance
  `0.610159278 -> 0.040735956`, impact `0.351481140 -> 0.003580729`.
- `ppo_clean_reward_50iter_audit.out:21` verdict:
  `CLEAN_REWARD_RAN_SAFER_BUT_WEAK_NEEDS_MODEL49_EVAL`.

Frozen eval:

- `ppo_clean_reward_model49_eval1024_audit.out:1` confirms exit 0 and 1645 CSV
  rows. More than 1024 rows are expected because early terminations produce
  additional reset records during one rollout horizon.
- `ppo_clean_reward_model49_eval1024_audit.out:3-6` reports controlled
  `0.631610942`, mean XY displacement `0.031481969m`, target distance
  `0.036184914m`, success marker `0.326443769`, clean success marker
  `0.326443769`, but impact still `0.283282675`.
- `ppo_clean_reward_model49_eval1024_audit.out:10-14` shows top speed outliers
  up to about `10m/s`.
- Verdict line 18: `CLEAN_MODEL49_EVAL_SAFER_BUT_IMPACT_TOO_HIGH_NO_10K`.

Interpretation:

- V2 fixed the far-fling behavior and produced meaningful push-scale displacement.
- But frozen eval still has too many high-speed impacts, so 10k/100k scaling is
  not justified.

## Speed-Guard V3

Changes:

- Set cube-push env action scale to `0.05`.
- Added speed excess penalty starting at `0.5m/s`.
- Added success speed gate `speed <= 0.5m/s`.

Run:

- `ppo_speed_guard_50iter_stdout.out:5` confirms action scale `0.050`.
- `ppo_speed_guard_50iter_stdout.out:7` confirms speed penalty and success-speed
  gate in the printed curriculum.
- `ppo_speed_guard_50iter_audit.out:7-14` reports final training-log speed
  `0.111070380m/s`, speed-over-0.5 rate `0.060546875`, controlled
  `0.627685547`, impact `0.003255208`, low motion `0.124755859`, success
  `0.004882812`.
- `ppo_speed_guard_50iter_audit.out:22` verdict:
  `SPEED_GUARD_SIGNAL_PRESENT_NEEDS_MODEL49_EVAL`.

Frozen eval:

- `ppo_speed_guard_model49_eval1024_summary.json:2-20` confirms action scale
  `0.05`, IK endpoint reset true, no training/dataset/attach, and 1575 eval rows.
- `ppo_speed_guard_model49_eval1024_audit.out:3-6` reports controlled
  `0.619682540`, impact `0.286984127`, clean success marker `0.323809524`,
  p95 speed `5.578794289m/s`, and p95 displacement `0.095530352m`.
- `ppo_speed_guard_model49_eval1024_audit.out:6` shows impact did not improve
  versus V2: `0.283282675 -> 0.286984127`.
- `ppo_speed_guard_model49_eval1024_audit.out:17` verdict:
  `SPEED_MODEL49_EVAL_NO_IMPACT_IMPROVEMENT_NO_10K`.

Interpretation:

- Lower action scale and speed penalty improved training-log appearance but did
  not solve frozen-eval high-speed impact.
- This is useful negative information: the next fix should target contact
  impulse/approach scheduling, action smoothing, or a velocity-limited controller,
  not simply bigger PPO iteration count.

## Current Answer To Scaling Question

Do not run 10k or 100k learned-policy eval yet.

The professor's "try it and inspect what comes out" requirement has been met at
small scale:

1. scripted physics rollout: 20,480 trials;
2. controlled/outlier posthoc filter;
3. no-attach PPO smoke and eval;
4. IK endpoint reset curriculum;
5. clean-reward curriculum;
6. speed-guard curriculum;
7. frozen 1k eval after each learned-policy variant.

But the current learned-policy gate fails because frozen eval impact remains
about `28-29%`. A reasonable gate before 10k is:

- 1k frozen eval exit 0;
- no attach/object posewrite, grasp marker 0;
- controlled rate >= 0.60;
- clean success marker >= 0.30;
- impact rate <= 0.05;
- no multi-meter or high-speed outlier family.

Current v3 has controlled and clean success around the right range, but impact is
too high. Therefore the next concrete research action is reward/controller v4:
action smoothing or velocity-limited push primitives, contact-speed curriculum,
and only then another 50-100 iteration run plus 1k frozen audit.

## 2026-05-26 Continuation - Smooth, Teacher, Contact-Speed

### V4 action-smoothed / velocity-limited controller

Code state:

- `roarm_rl/roarm_cube_push_env.py:97-105` sets action smoothing, per-step joint
  delta cap, contact slowdown, fast-cube slowdown, lead cap, and scripted teacher
  defaults.
- `roarm_rl/roarm_cube_push_env.py:213-276` applies policy actions through
  teacher blend if enabled, smoothing, joint-delta clamp, contact/fast-cube
  slowdown, lead limiting, and gripper-open hold. `_apply_action` remains
  robot-target-only with no attach.

Result:

- `ppo_smooth_limit_50iter_audit.out:4-24` showed training signal: lower speed
  and impact than the speed-guard run.
- Frozen eval still failed the scale gate. `ppo_smooth_limit_model49_eval1024_audit.out:3-8`
  reports controlled `0.618542109`, impact `0.245576787`, clean success
  `0.197452229`, no attach/dataset/grasp, and verdict
  `SMOOTH_MODEL49_EVAL_IMPROVED_BUT_NO_10K`.

### V5 scripted-teacher warm-start

Code state:

- `roarm_rl/roarm_cube_push_env.py:181-211` computes an IK teacher goal from cube
  position and push direction.
- `roarm_rl/roarm_cube_push_env.py:217-235` blends teacher actions into policy
  actions only when `scripted_teacher_blend > 0`.
- `cube_push_rl_stage0_static_audit.out:19` confirms the teacher warm-start
  wiring exists and is logged.

Result:

- Training looked better: `ppo_teacher_warmstart_50iter_audit.out:9-26` reports
  controlled `0.722493529`, impact `0.001790365`, teacher goal ok `1.0`, and
  verdict `TEACHER_WARMSTART_TRAIN_SIGNAL_PRESENT_NEEDS_TEACHER_OFF_EVAL`.
- But teacher-off frozen eval did not transfer. `ppo_teacher_warmstart_model49_eval1024_audit.out:3-8`
  reports controlled `0.428257687`, impact `0.257686676`, clean success
  `0.095168375`, teacher blend `0.0`, and verdict
  `TEACHER_MODEL49_EVAL_TEACHER_OFF_NO_TRANSFER_NO_10K`.

Interpretation:

- Teacher blending as action assistance is not the same as learned policy
  competence. It may improve training telemetry while leaving the standalone
  actor weak.

### V6 policy-only contact-speed curriculum

Eval entry update:

- `roarm_rl/eval_cube_push_policy.py` md5
  `bec3214d391862e4196d221775f4477d`.
- `cube_push_rl_stage0_static_audit.out:1` records env/train/eval md5s.
- `cube_push_rl_stage0_static_audit.out:22` confirms frozen eval can replay the
  same motion-limit overrides as training.
- `cube_push_rl_stage0_static_audit.out:37` reports static audit PASS.

Training command parameters:

- action scale `0.025`
- max joint delta per step `0.004rad`
- contact slowdown `0.15`
- fast-cube slowdown `0.05`
- joint target lead cap `0.030rad`
- IK precontact clearance `0.020m`
- speed threshold `0.200m/s`
- higher impact/terminal/speed penalties
- teacher blend `0.0`

Training result:

- `ppo_contact_speed_50iter_exit_code.txt:1` exit code 0.
- `ppo_contact_speed_50iter_audit.out:2` confirms `model_49.pt` exists with md5
  `387a7cd95ba218c1045c523ce1c6c89e`.
- `ppo_contact_speed_50iter_audit.out:4-14` reports last training values:
  push-aligned displacement `0.003850930m`, XY displacement `0.013729703m`,
  speed `0.031655323m/s`, speed-over-0.5 rate `0.011067709`, controlled
  `0.508544922`, impact `0.000813802`, low motion `0.377115905`, success
  `0.000325521`.
- `ppo_contact_speed_50iter_audit.out:24-27` shows speed/impact improvement over
  V4 training but high low-motion and verdict
  `CONTACT_SPEED_TRAIN_SIGNAL_PRESENT_NEEDS_MODEL49_EVAL`.

Frozen 1k eval:

- `ppo_contact_speed_model49_eval1024_exit_code.txt:1` exit code 0.
- `ppo_contact_speed_model49_eval1024_audit.out:1` confirms summary/CSV md5s and
  1216 CSV rows.
- `ppo_contact_speed_model49_eval1024_audit.out:3-8` reports action scale
  `0.025`, joint cap `0.004`, teacher blend `0.0`, controlled `0.527138158`,
  impact `0.153782895`, low motion `0.180098684`, clean success `0.110197368`,
  and no attach/dataset/grasp.
- `ppo_contact_speed_model49_eval1024_audit.out:7` compares against V4/V5 eval:
  impact improved from `0.245576787`/`0.257686676` to `0.153782895`, but clean
  success is only `0.110197368`.
- `ppo_contact_speed_model49_eval1024_audit.out:18` verdict:
  `CONTACT_SPEED_MODEL49_EVAL_IMPROVED_BUT_NO_10K`.

Interpretation:

- This is a real improvement on impact but not enough. The policy became safer
  and weaker at the same time: less violent contact, but still too many impact
  cases and too little clean 3cm push success.

### Teacher-on diagnostic

Purpose:

- Separate "the policy failed to imitate" from "the current IK/scripted teacher
  path itself is not clean enough".
- This is not a learned-policy claim.

Result:

- `ppo_contact_speed_teacher_on_eval1024_exit_code.txt:1` exit code 0.
- `ppo_contact_speed_teacher_on_eval1024_audit.out:3-9` reports teacher blend
  `1.0`, horizon fraction `1.0`, goal push `0.055m`, controlled `0.471836735`,
  impact `0.162448980`, low motion `0.341224490`, clean success `0.067755102`,
  no attach/dataset/grasp, and critical interpretation that this is not a frozen
  policy claim.
- `ppo_contact_speed_teacher_on_eval1024_audit.out:18` verdict:
  `TEACHER_ON_DIAGNOSTIC_UNSAFE_OR_WEAK_NOT_LEARNED_NO_10K`.

Interpretation:

- The current IK teacher trajectory is also not a clean teacher under the
  contact-speed limits. It is too slow/weak in many cases and still has impact
  outliers. Therefore the next step should not be to scale teacher-assisted PPO.

## Current Scaling Answer

Do not run 10k/100k learned-policy evaluation yet.

The most recent learned-policy result improves impact from the prior ~25% band
to `15.4%`, but it still misses the `~5%` impact gate and does not maintain
clean success. The teacher-on diagnostic also fails, so the next research step is
to redesign the teacher/contact trajectory or implement true imitation/resume
fine-tuning, then repeat:

1. 50-100 iteration training;
2. frozen teacher-off 1k audit;
3. scale to 10k only if impact is below ~5% and clean/controlled success remains
   meaningful.
