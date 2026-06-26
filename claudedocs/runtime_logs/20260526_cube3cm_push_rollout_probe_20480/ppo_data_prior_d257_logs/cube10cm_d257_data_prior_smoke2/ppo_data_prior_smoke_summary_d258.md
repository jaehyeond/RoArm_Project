# D258 PPO Data-Prior Smoke Summary

- verdict: `D258_PPO_DATA_PRIOR_SMOKE_WIRING_PASS_BEHAVIOR_UNPROVEN`
- runtime: host `isaaclab`, `cuda:0`, `num_envs=32`, `max_iterations=2`, `num_steps_per_env=24`
- D257 checkpoint loaded through `bc_teacher_checkpoint_path`: yes
- `cube_push_bc_teacher_blend_mean`: `1.0`, `1.0`
- `cube_push_bc_teacher_imitation_mse`: `1.210442`, `1.253437`
- `bc_teacher_imitation_penalty`: `-6.052209`, `-6.267184`
- final logged mean reward: `-392.534027`
- final logged mean episode length: `42.333332`
- displacement remained near zero: `cube_push_disp_along_m=0.000151`, `cube_push_disp_xy_m=0.000615` at iteration 1
- success remained zero: `cube_push_success_rate=0.0`
- active Isaac/PPO process after run: no
- GPU returned to pre-run baseline memory use: yes

Important interpretation:

- This is a wiring smoke, not a policy-performance result.
- The original D257 command must be run with `PYTHONPATH=.` from the repo root.
- The sandbox attempt failed before a valid smoke because Isaac/PhysX could not see CUDA and `roarm_rl` import failed.
- Do not run a longer PPO from this alone. The next step should be a teacher-only rollout/feature-alignment probe to explain near-zero displacement before any longer PPO.
