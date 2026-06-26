# D276 TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d276_logs/cube10cm_d276_tap10cm_aabb_d256reset_bc_no_randlen_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d276_logs/cube10cm_d276_tap10cm_aabb_d256reset_bc_no_randlen_smoke --host 127.0.0.1 --port 6006`

## Issues

- missing core TensorBoard scalars: ['Train/mean_reward', 'Train/mean_episode_length']

## Warnings

- short run: Train/mean_reward has 0 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.09187658876180649
- tap max displacement remains small: max=0.00036540720611810684

## Selected Scalars

- `Loss/value_function`: n `2`, first `2906.629638671875`, last `802.4774169921875`, min `802.4774169921875`, max `2906.629638671875`
- `Loss/surrogate`: n `2`, first `-0.01921144314110279`, last `-0.0013032738352194428`, min `-0.01921144314110279`, max `-0.0013032738352194428`
- `Loss/entropy`: n `2`, first `7.1751179695129395`, last `7.173254013061523`, min `7.173254013061523`, max `7.1751179695129395`
- `Loss/learning_rate`: n `2`, first `9.999999747378752e-06`, last `0.00011390625149942935`, min `9.999999747378752e-06`, max `0.00011390625149942935`
- `Policy/mean_noise_std`: n `2`, first `0.800047755241394`, last `0.7993403673171997`, min `0.7993403673171997`, max `0.800047755241394`
- `Episode/cube_push_tcp_cube_dist_m`: n `2`, first `0.09352301061153412`, last `0.09187658876180649`, min `0.09187658876180649`, max `0.09352301061153412`
- `Episode/cube_push_joint_delta_abs_mean`: n `2`, first `0.00799211859703064`, last `0.004545318894088268`, min `0.004545318894088268`, max `0.00799211859703064`
- `Episode/cube_push_joint_delta_abs_max`: n `2`, first `0.021875901147723198`, last `0.01306622289121151`, min `0.01306622289121151`, max `0.021875901147723198`
- `Episode/cube_push_joint_delta_cap_rate`: n `2`, first `0.0190972238779068`, last `0.0236545167863369`, min `0.0190972238779068`, max `0.0236545167863369`
- `Episode/cube_push_action_abs_mean`: n `2`, first `0.199802964925766`, last `0.11363296210765839`, min `0.11363296210765839`, max `0.199802964925766`
- `Episode/cube_push_action_abs_max`: n `2`, first `0.5468975305557251`, last `0.3266555964946747`, min `0.3266555964946747`, max `0.5468975305557251`
- `Episode/cube_push_target_lead_limit_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `2`, first `0.5771029591560364`, last `0.4828728437423706`, min `0.4828728437423706`, max `0.5771029591560364`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `2`, first `0.199802964925766`, last `0.11363296210765839`, min `0.11363296210765839`, max `0.199802964925766`
- `Episode/cube_push_d256_reset_active_rate`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `2`, first `447.71875`, last `447.71875`, min `447.71875`, max `447.71875`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `2`, first `0.5771029591560364`, last `0.4828728437423706`, min `0.4828728437423706`, max `0.5771029591560364`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `2`, first `0.199802964925766`, last `0.11363296210765839`, min `0.11363296210765839`, max `0.199802964925766`
- `Episode/cube_tap_d256_reset_active_rate`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `2`, first `447.71875`, last `447.71875`, min `447.71875`, max `447.71875`
- `Episode/bc_teacher_imitation_penalty`: n `2`, first `-2.885514736175537`, last `-2.4143643379211426`, min `-2.885514736175537`, max `-2.4143643379211426`
- `Episode/cube_tap_contact_seen_rate`: n `2`, first `0.45703125`, last `0.46875`, min `0.45703125`, max `0.46875`
- `Episode/cube_tap_contact_proxy_rate`: n `2`, first `0.4466145932674408`, last `0.45703125`, min `0.4466145932674408`, max `0.45703125`
- `Episode/cube_tap_reaction_seen_rate`: n `2`, first `0.45703125`, last `0.46875`, min `0.45703125`, max `0.46875`
- `Episode/cube_tap_reaction_signal_now_rate`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `2`, first `0.45703125`, last `0.46875`, min `0.45703125`, max `0.46875`
- `Episode/cube_tap_reaction_now_rate`: n `2`, first `0.45703125`, last `0.46875`, min `0.45703125`, max `0.46875`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `2`, first `0.45703125`, last `0.46875`, min `0.45703125`, max `0.46875`
- `Episode/cube_tap_useful_now_rate`: n `2`, first `0.45703125`, last `0.46875`, min `0.45703125`, max `0.46875`
- `Episode/cube_tap_useful_seen_rate`: n `2`, first `0.45703125`, last `0.46875`, min `0.45703125`, max `0.46875`
- `Episode/cube_tap_success_rate`: n `2`, first `0.39453125`, last `0.42578125`, min `0.39453125`, max `0.42578125`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_overshoot_now_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_overshoot_seen_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_max_disp_along_m`: n `2`, first `0.0001555407652631402`, last `0.00036540720611810684`, min `0.0001555407652631402`, max `0.00036540720611810684`
- `Episode/cube_tap_max_disp_xy_m`: n `2`, first `0.0007376158609986305`, last `0.0009607080137357116`, min `0.0007376158609986305`, max `0.0009607080137357116`
- `Episode/cube_tap_contact_face_gap_m`: n `2`, first `-0.00949584599584341`, last `-0.009016145020723343`, min `-0.00949584599584341`, max `-0.009016145020723343`
- `Episode/cube_tap_contact_lateral_m`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_vertical_offset_m`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
