# Session 2026-05-29 - Cube3cm rsl_rl Actor Distillation Gate

## Scope

- Professor cube3cm push/tap branch only.
- Local machine only: no B200, no SSH, no pull, no `.ssh` copy.
- Goal: execute D110's next gate: true supervised actor/normalized-action
  distillation into the rsl_rl actor, then only a tiny teacher-off 128 audit.
- Not Track A grasp, not Track A runtime, not PPO scale-up, not dataset
  generation, not 10k/100k learned-policy robustness.

## Verified Starting Point

- `START_HERE.md:257-277`, `DECISIONS.md:5587-5626`,
  `EXPERIMENT_LEDGER.md:148`, and
  `claudedocs/session_20260529_cube3cm_bc_teacher_bridge_redesign.md:75-132`
  agree that teacher-on direct-step/joint-pos lowx `1.0` recovered the bridge,
  but existing `model_11.pt` and smoke8 `model_7.pt` failed teacher-off 128.
- Rechecked source logs before acting:
  - Teacher-on lowx `1.0` audit:
    `ppo_bc_teacher_warmstart_bridge_directstep_jointpos_lowx100_seed890_eval128_audit.out:1-5`.
  - Teacher-on lowx `1.0` bucket:
    `ppo_bc_teacher_warmstart_bridge_directstep_jointpos_lowx100_seed890_eval128_bucket.out:1-10`.
  - Existing `model_11.pt` teacher-off 128:
    `ppo_bc_teacher_warmstart_newloop_teacheroff_model11_seed891_eval128_audit.out:1-5`
    and bucket `:1-10`.
  - Smoke8 `model_7.pt` train config/MSE:
    `ppo_bc_teacher_newloop_distill_smoke8_seed892_stdout.out:3-30,102,144,349`.
  - Smoke8 `model_7.pt` teacher-off 128:
    `ppo_bc_teacher_newloop_distill_smoke8_seed892_model7_teacheroff_eval128_seed893_audit.out:1-5`
    and bucket `:1-10`.
- Citation caveat preserved: current local
  `ppo_bc_teacher_warmstart_smoke12_seed885_stderr.out:1-6` contains only
  `requests` warnings, not the previously claimed missing `gymnasium` error.

## Code

- Added `roarm_rl/distill_cube_push_actor.py`.
- It launches `RoArm-CubePush-Direct-v0`, collects the BC teacher's normalized
  joint-delta actions from the direct-step/joint-pos bridge, trains the rsl_rl
  actor mean with actor observation normalization, and writes a normal rsl_rl
  checkpoint loadable by `roarm_rl/eval_cube_push_policy.py`.
- Static checks:
  - `python -m py_compile roarm_rl/distill_cube_push_actor.py` PASS.
  - `git diff --check -- roarm_rl/distill_cube_push_actor.py` PASS.

## Actor / Checkpoint Structure Verified

- rsl_rl checkpoint keys are `model_state_dict`, `optimizer_state_dict`, `iter`,
  `infos`.
- Actor structure in the checkpoint is `actor.0/2/4/6` with shapes
  `28 -> 256 -> 128 -> 64 -> 6`; actor and critic observation normalizers are
  stored in `actor_obs_normalizer.*` and `critic_obs_normalizer.*`.
- Source code agrees: `roarm_rl/agents/rsl_rl_ppo_cfg.py:22-29` configures
  actor/critic obs normalization and `[256, 128, 64]` ELU hidden dims.
- Env observation is 28D in `roarm_rl/roarm_stack_env.py:501-529`; BC teacher
  action conversion to normalized 6D action is in
  `roarm_rl/roarm_cube_push_env.py:424-452`.

## Distillation Run

- Run directory:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_bc_teacher_actor_distill_seed894/`.
- Command used local IsaacLab/GPU with `PYTHONPATH` pointing at this repo.
- Scope/semantics:
  - `actor_distill_stdout.out:47` confirms professor cube branch scope,
    training YES, dataset generation NO, Track A runtime NO, grasp attach NO,
    rollout object posewrite NO.
  - `actor_distill_stdout.out:48` confirms normalized joint-delta action semantics
    with `robot_dof_targets = joint_pos + action_scale(0.040) * action`.
  - `actor_distill_stdout.out:49` confirms `num_envs=128`,
    `collect_steps=600`, seed `894`, 6s horizon, `joint_delta_reference=joint_pos`,
    `bc_teacher_phase_timing=direct_steps`, lowx scale `1.0`.
- Output checkpoint:
  - `model_actor_distill.pt`
  - md5 `57811cfb054ca7ac39b134d1d97cd543`
- Metrics:
  - `actor_distill_stdout.out:69` samples `76800`, obs dim `28`, action dim `6`,
    train `69120`, val `7680`.
  - `actor_distill_stdout.out:70` initial train/val MSE
    `0.169973969` / `0.169735238`, final train/val MSE
    `0.000760018` / `0.000794161`.
  - `actor_distill_stdout.out:71` final val MAE `0.015571725`,
    final val max abs `0.743932724`, teacher action abs mean `0.280680388`,
    actor action abs mean `0.281257778`.
  - `actor_distill_stdout.out:72` verdict `PASS_ACTOR_DISTILLATION`, but
    teacher-off rollout was not yet validated at that point.

## Teacher-Off 128 Audit

- Evaluated only 128 envs, seed895, one rollout. No teacher-off 1024 was run.
- Eval stdout confirms:
  - `model_actor_distill_teacheroff_eval128_seed895_stdout.out` scope is
    no-attach cube push eval, training NO, dataset generation NO, grasp attach NO,
    rollout object posewrite NO.
  - Same stdout confirms `bc_teacher_blend=0.0`,
    `bc_teacher_imitation_reward_scale=0.0`, `bc_teacher_checkpoint_path=NONE`,
    `num_envs=128`, `num_rollouts=1`, seed `895`, and checkpoint
    `model_actor_distill.pt`.
- Mechanism audit:
  - `model_actor_distill_teacheroff_eval128_seed895_audit.out:1-5` PASSes
    mechanism only: row count match, learned policy true, no training, no dataset
    generation, no grasp attach, no object posewrite, no grasped marker.
  - Performance failed: controlled `0.101562500`, impact `0.000000000`,
    low-motion `0.929687500`, success `0.031250000`.
- Bucket audit:
  - `model_actor_distill_teacheroff_eval128_seed895_bucket.out:1-10` FAILs
    per-bucket screen.
  - Overall controlled `0.101562500`, impact `0`, low-motion `0.929687500`,
    success `0.031250000`.
  - `(1,0)` low_x success `0`, mid_x success `0`, high_x success
    `0.076923077`.

## Interpretation

- This disproves the narrow hypothesis that simply writing the BC teacher into
  rsl_rl actor weights is enough.
- The actor now imitates one-step teacher actions well under the collected
  teacher-state distribution, but closed-loop teacher-off execution still drifts
  into low-motion.
- The likely remaining issue is state-distribution / action-target dynamics:
  small one-step action error, action smoothing, joint target lead, reset timing,
  and contact dynamics may compound into a policy that mostly fails to push.

## Decision

- Do not call `model_actor_distill.pt` a successful learned PPO/RL policy.
- Do not run teacher-off 1024, PPO scale-up, 10k/100k learned robustness, dataset
  generation, or Track A runtime from this checkpoint.
- Next valid work is closed-loop/action-target analysis of the actor-distilled
  rollout, or a stronger rollout-aware actor initialization, followed by only a
  teacher-off 128 audit before any 1024 audit.
