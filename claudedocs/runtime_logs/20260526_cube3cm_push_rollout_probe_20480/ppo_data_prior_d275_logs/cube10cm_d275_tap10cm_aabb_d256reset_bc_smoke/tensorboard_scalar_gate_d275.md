# D275 TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d275_logs/cube10cm_d275_tap10cm_aabb_d256reset_bc_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d275_logs/cube10cm_d275_tap10cm_aabb_d256reset_bc_smoke --host 127.0.0.1 --port 6006`

## Issues

- tap overshoot seen rate too high: max=0.125

## Warnings

- short run: Train/mean_reward has 2 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.0885186716914177
- tap max displacement remains small: max=0.0005887373699806631

## Selected Scalars

- `Train/mean_reward`: n `2`, first `-43.598243713378906`, last `-43.598243713378906`, min `-43.598243713378906`, max `-43.598243713378906`
- `Train/mean_episode_length`: n `2`, first `21.25`, last `21.25`, min `21.25`, max `21.25`
- `Loss/value_function`: n `2`, first `804.9876098632812`, last `73632.7265625`, min `804.9876098632812`, max `73632.7265625`
- `Loss/surrogate`: n `2`, first `-0.017300685867667198`, last `-0.0038089316803961992`, min `-0.017300685867667198`, max `-0.0038089316803961992`
- `Loss/entropy`: n `2`, first `7.182097434997559`, last `7.185716152191162`, min `7.182097434997559`, max `7.185716152191162`
- `Loss/learning_rate`: n `2`, first `9.999999747378752e-06`, last `7.593750342493877e-05`, min `9.999999747378752e-06`, max `7.593750342493877e-05`
- `Policy/mean_noise_std`: n `2`, first `0.8011373281478882`, last `0.8022983074188232`, min `0.8011373281478882`, max `0.8022983074188232`
- `Episode/cube_push_tcp_cube_dist_m`: n `2`, first `0.08980129659175873`, last `0.0885186716914177`, min `0.0885186716914177`, max `0.08980129659175873`
- `Episode/cube_push_joint_delta_abs_mean`: n `2`, first `0.007745746523141861`, last `0.007854601368308067`, min `0.007745746523141861`, max `0.007854601368308067`
- `Episode/cube_push_joint_delta_abs_max`: n `2`, first `0.021927958354353905`, last `0.019679471850395203`, min `0.019679471850395203`, max `0.021927958354353905`
- `Episode/cube_push_joint_delta_cap_rate`: n `2`, first `0.009765625931322575`, last `0.0290798619389534`, min `0.009765625931322575`, max `0.0290798619389534`
- `Episode/cube_push_action_abs_mean`: n `2`, first `0.19364365935325623`, last `0.19636502861976624`, min `0.19364365935325623`, max `0.19636502861976624`
- `Episode/cube_push_action_abs_max`: n `2`, first `0.548198938369751`, last `0.49198684096336365`, min `0.49198684096336365`, max `0.548198938369751`
- `Episode/cube_push_target_lead_limit_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `2`, first `0.5184258222579956`, last `0.5418598055839539`, min `0.5184258222579956`, max `0.5418598055839539`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `2`, first `0.19364365935325623`, last `0.19636502861976624`, min `0.19364365935325623`, max `0.19636502861976624`
- `Episode/cube_push_d256_reset_active_rate`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `2`, first `472.90234375`, last `476.71875`, min `472.90234375`, max `476.71875`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `2`, first `0.5184258222579956`, last `0.5418598055839539`, min `0.5184258222579956`, max `0.5418598055839539`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `2`, first `0.19364365935325623`, last `0.19636502861976624`, min `0.19364365935325623`, max `0.19636502861976624`
- `Episode/cube_tap_d256_reset_active_rate`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `2`, first `472.90234375`, last `476.71875`, min `472.90234375`, max `476.71875`
- `Episode/bc_teacher_imitation_penalty`: n `2`, first `-2.5921289920806885`, last `-2.709298849105835`, min `-2.709298849105835`, max `-2.5921289920806885`
- `Episode/cube_tap_contact_seen_rate`: n `2`, first `0.546875`, last `0.53125`, min `0.53125`, max `0.546875`
- `Episode/cube_tap_contact_proxy_rate`: n `2`, first `0.5403646230697632`, last `0.53125`, min `0.53125`, max `0.5403646230697632`
- `Episode/cube_tap_reaction_seen_rate`: n `2`, first `0.546875`, last `0.53125`, min `0.53125`, max `0.546875`
- `Episode/cube_tap_reaction_signal_now_rate`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `2`, first `0.546875`, last `0.53125`, min `0.53125`, max `0.546875`
- `Episode/cube_tap_reaction_now_rate`: n `2`, first `0.546875`, last `0.53125`, min `0.53125`, max `0.546875`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `2`, first `0.546875`, last `0.53125`, min `0.53125`, max `0.546875`
- `Episode/cube_tap_useful_now_rate`: n `2`, first `0.5364583730697632`, last `0.4986979365348816`, min `0.4986979365348816`, max `0.5364583730697632`
- `Episode/cube_tap_useful_seen_rate`: n `2`, first `0.5364583730697632`, last `0.46875`, min `0.46875`, max `0.5364583730697632`
- `Episode/cube_tap_success_rate`: n `2`, first `0.5364583730697632`, last `0.46875`, min `0.46875`, max `0.5364583730697632`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `2`, first `0.9856771230697632`, last `0.875`, min `0.875`, max `0.9856771230697632`
- `Episode/cube_tap_overshoot_now_rate`: n `2`, first `0.014322916977107525`, last `0.0950520858168602`, min `0.014322916977107525`, max `0.0950520858168602`
- `Episode/cube_tap_overshoot_seen_rate`: n `2`, first `0.014322916977107525`, last `0.125`, min `0.014322916977107525`, max `0.125`
- `Episode/cube_tap_max_disp_along_m`: n `2`, first `0.00016351963859051466`, last `0.0005887373699806631`, min `0.00016351963859051466`, max `0.0005887373699806631`
- `Episode/cube_tap_max_disp_xy_m`: n `2`, first `0.0005594385438598692`, last `0.0035700853914022446`, min `0.0005594385438598692`, max `0.0035700853914022446`
- `Episode/cube_tap_contact_face_gap_m`: n `2`, first `-0.008573184721171856`, last `-0.008954408578574657`, min `-0.008954408578574657`, max `-0.008573184721171856`
- `Episode/cube_tap_contact_lateral_m`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_vertical_offset_m`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
