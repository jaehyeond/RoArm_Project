# 2026-06-20 - cube10cm top-view teacher-off frozen eval D278

## Scope

This session continued the professor 10cm / 0.72kg cube top-view visual
trajectory branch. It did not run long PPO, RunPod, B200, RoArm deployment,
Track A, SmolVLA/VLA fine-tuning, render scale-up, or cleanup.

Goal:

- evaluate the D277 actor checkpoint with the BC teacher action blend disabled;
- keep the same D256 reset and AABB contact contract;
- decide whether a short PPO ladder is justified.

## Setup

Added:

- `sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`

Valid D278 command used host GPU access because sandboxed Isaac/PhysX could not
see CUDA:

- checkpoint:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/model_0.pt`
- env:
  `RoArm-CubeTap10cm-Direct-v0`
- contact proxy:
  `link5_collision_aabb`
- D256 reset CSV:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/rl_transition_preflight_d256/ppo_actor_prior_teacher_rows_d256.csv`
- D256 reset:
  `frame_index_t=0`, `linspace`
- episode:
  `episode_length_s=6.0`, `eval_steps=580`, `num_envs=32`
- teacher:
  `bc_teacher_blend=0.0`, `bc_teacher_imitation_reward_scale=0.0`,
  no BC teacher checkpoint loaded.

Earlier invalid attempts were not counted:

- no `PYTHONPATH=.` caused `roarm_rl` import failure;
- sandboxed Isaac/PhysX could not see CUDA;
- first script default pointed to a non-existent D256 CSV name;
- default tap episode length was `1.2s`, not D277's `6.0s`;
- TensorDict finite-check needed explicit handling.

## Result

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_off_policy_eval_d278/tap10cm/teacher_off_policy_eval_summary_d278.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_off_policy_eval_d278/tap10cm/teacher_off_policy_eval_summary_d278.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/teacher_off_policy_eval_d278/tap10cm/teacher_off_policy_eval_steps_d278.csv`

Verdict:

`TEACHER_OFF_FROZEN_EVAL_FAIL_NO_POLICY_CLAIM`

Key metrics:

- D256 reset active rate: `1.0`
- BC teacher blend mean last: `0.0`
- contact seen: `0.875`
- reaction seen: `0.875`
- useful seen: `0.5625`
- success flag: `0.875`
- overshoot seen: `0.3125`
- max displacement along mean/max:
  `0.0024283849634230137` / `0.018782615661621094`
- max displacement xy mean/max:
  `0.020250540226697922` / `0.10077980160713196`
- min contact vertical offset mean/min/max:
  `0.0` / `0.0` / `0.0`
- last contact vertical offset mean/max:
  `0.02129734866321087` / `0.24940747022628784`
- raw TCP-threshold contact seen: `0.0`
- joint-delta cap rate last/max trace:
  `0.1145833432674408` / `0.15625`
- raw policy action abs mean/max trace:
  `0.1184795308344323` / `1.3003933429718018`
- reward/obs/action finite all: `True`

Issues:

- tap overshoot seen rate too high: `0.3125`
- tap contact vertical offset too high: max `0.24940747022628784`

## Interpretation

D278 is not a total no-motion failure. The frozen actor can still produce AABB
contact/reaction in many D256-reset states.

However, D278 fails the policy gate because it is not controlled:

- overshoot is far above the `0.05` gate;
- some rollouts leave the intended vertical contact geometry;
- raw policy action max exceeds the wrapper clip range before clipping;
- the high success flag is not sufficient because overshoot can occur after or
  around the useful event.

This means D277's teacher-on behavior was still teacher-prior behavior, not
learned-policy evidence.

## Verification

- `python -m py_compile sim_scripts/cube10cm_top_view_teacher_off_policy_eval.py`
  passed.
- `git diff --check` passed.
- D278 valid eval exited cleanly.
- `ps -C python -C python3` showed no active local Python process.
- `nvidia-smi` returned to the observed baseline class state:
  RTX 4090 Laptop GPU, `2509MiB` used, `13436MiB` free.

## Next Step

Do not run long PPO or a short PPO ladder from D278.

Next work should be diagnostic:

1. Compare D274/D277 teacher-on action traces against D278 teacher-off actor
   actions on the same D256 reset states.
2. Isolate overshoot cases and vertical-offset outliers.
3. Check whether the actor learned teacher direction or relied on
   `bc_teacher_blend=1.0`.
4. Consider a controlled teacher-blend/BC-distillation strategy only after the
   action-trace mismatch is quantified.

No learned policy, teacher-off success, or RoArm readiness claim exists.
