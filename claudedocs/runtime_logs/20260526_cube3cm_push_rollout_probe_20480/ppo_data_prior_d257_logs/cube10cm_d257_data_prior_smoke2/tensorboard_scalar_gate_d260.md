# D260 TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs/cube10cm_d257_data_prior_smoke2`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs/cube10cm_d257_data_prior_smoke2 --host 127.0.0.1 --port 6006`

## Issues

- no task success/contact signal in TensorBoard (max success/contact-like scalar=0.0)
- low-motion rate too high: last=0.9778646230697632
- joint-delta cap rate too high: max=0.7411024570465088

## Warnings

- short run: Train/mean_reward has 1 points, promotion gate expects at least 10
- TCP-cube distance remains high: last=0.3268700838088989
- disp_along remains too small: last=0.00015073080430738628
- controlled rate remains low: last=0.0182291679084301

## Selected Scalars

- `Train/mean_reward`: n `1`, first `-392.5340270996094`, last `-392.5340270996094`, min `-392.5340270996094`, max `-392.5340270996094`
- `Train/mean_episode_length`: n `1`, first `42.33333206176758`, last `42.33333206176758`, min `42.33333206176758`, max `42.33333206176758`
- `Loss/value_function`: n `2`, first `6711.08642578125`, last `6737.7255859375`, min `6711.08642578125`, max `6737.7255859375`
- `Loss/surrogate`: n `2`, first `-0.011346347630023956`, last `-0.012086811475455761`, min `-0.012086811475455761`, max `-0.011346347630023956`
- `Loss/entropy`: n `2`, first `7.177923202514648`, last `7.179165840148926`, min `7.177923202514648`, max `7.179165840148926`
- `Loss/learning_rate`: n `2`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `2`, first `0.8005133867263794`, last `0.8006278276443481`, min `0.8005133867263794`, max `0.8006278276443481`
- `Episode/cube_push_disp_along_m`: n `2`, first `-1.603420969331637e-07`, last `0.00015073080430738628`, min `-1.603420969331637e-07`, max `0.00015073080430738628`
- `Episode/cube_push_disp_xy_m`: n `2`, first `7.5530833782977425e-06`, last `0.0006145103834569454`, min `7.5530833782977425e-06`, max `0.0006145103834569454`
- `Episode/cube_push_target_xy_dist_m`: n `2`, first `0.04000016674399376`, last `0.04001577943563461`, min `0.04000016674399376`, max `0.04001577943563461`
- `Episode/cube_push_tcp_cube_dist_m`: n `2`, first `0.33816835284233093`, last `0.3268700838088989`, min `0.3268700838088989`, max `0.33816835284233093`
- `Episode/cube_push_controlled_rate`: n `2`, first `0.0`, last `0.0182291679084301`, min `0.0`, max `0.0182291679084301`
- `Episode/cube_push_impact_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_low_motion_rate`: n `2`, first `1.0`, last `0.9778646230697632`, min `0.9778646230697632`, max `1.0`
- `Episode/cube_push_success_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_joint_delta_abs_mean`: n `2`, first `0.032467830926179886`, last `0.03268555551767349`, min `0.032467830926179886`, max `0.03268555551767349`
- `Episode/cube_push_joint_delta_abs_max`: n `2`, first `0.038961395621299744`, last `0.03922266513109207`, min `0.038961395621299744`, max `0.03922266513109207`
- `Episode/cube_push_joint_delta_cap_rate`: n `2`, first `0.5034722685813904`, last `0.7411024570465088`, min `0.5034722685813904`, max `0.7411024570465088`
- `Episode/cube_push_action_abs_mean`: n `2`, first `0.81169593334198`, last `0.8171388506889343`, min `0.81169593334198`, max `0.8171388506889343`
- `Episode/cube_push_action_abs_max`: n `2`, first `0.9740350246429443`, last `0.980566680431366`, min `0.9740350246429443`, max `0.980566680431366`
- `Episode/cube_push_target_lead_limit_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `2`, first `1.2104418277740479`, last `1.253436803817749`, min `1.2104418277740479`, max `1.253436803817749`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `2`, first `0.81169593334198`, last `0.8171388506889343`, min `0.81169593334198`, max `0.8171388506889343`
