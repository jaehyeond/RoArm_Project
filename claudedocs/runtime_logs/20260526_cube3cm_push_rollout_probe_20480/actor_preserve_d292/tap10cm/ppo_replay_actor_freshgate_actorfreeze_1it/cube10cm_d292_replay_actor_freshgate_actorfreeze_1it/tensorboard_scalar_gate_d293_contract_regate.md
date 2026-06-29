# D293_CONTRACT_REGATE_D292 TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d292/tap10cm/ppo_replay_actor_freshgate_actorfreeze_1it/cube10cm_d292_replay_actor_freshgate_actorfreeze_1it`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d292/tap10cm/ppo_replay_actor_freshgate_actorfreeze_1it/cube10cm_d292_replay_actor_freshgate_actorfreeze_1it --host 127.0.0.1 --port 6006`

## Issues

- tap max displacement remains small: max=1.3096122529532295e-05

## Warnings

- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.09063738584518433

## Selected Scalars

- `Train/mean_reward`: n `1`, first `-8.49217414855957`, last `-8.49217414855957`, min `-8.49217414855957`, max `-8.49217414855957`
- `Train/mean_episode_length`: n `1`, first `1.6521738767623901`, last `1.6521738767623901`, min `1.6521738767623901`, max `1.6521738767623901`
- `Loss/value_function`: n `1`, first `96.76457977294922`, last `96.76457977294922`, min `96.76457977294922`, max `96.76457977294922`
- `Loss/surrogate`: n `1`, first `0.11207561194896698`, last `0.11207561194896698`, min `0.11207561194896698`, max `0.11207561194896698`
- `Loss/entropy`: n `1`, first `-23.27627182006836`, last `-23.27627182006836`, min `-23.27627182006836`, max `-23.27627182006836`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.004999999422580004`, last `0.004999999422580004`, min `0.004999999422580004`, max `0.004999999422580004`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.09063738584518433`, last `0.09063738584518433`, min `0.09063738584518433`, max `0.09063738584518433`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.004646200221031904`, last `0.004646200221031904`, min `0.004646200221031904`, max `0.004646200221031904`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.01462140865623951`, last `0.01462140865623951`, min `0.01462140865623951`, max `0.01462140865623951`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.0008680556202307343`, last `0.0008680556202307343`, min `0.0008680556202307343`, max `0.0008680556202307343`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.1168801337480545`, last `0.1168801337480545`, min `0.1168801337480545`, max `0.1168801337480545`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.3655352294445038`, last `0.3655352294445038`, min `0.3655352294445038`, max `0.3655352294445038`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `576.6588745117188`, last `576.6588745117188`, min `576.6588745117188`, max `576.6588745117188`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `576.6588745117188`, last `576.6588745117188`, min `576.6588745117188`, max `576.6588745117188`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.08984375`, last `0.08984375`, min `0.08984375`, max `0.08984375`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.08984375`, last `0.08984375`, min `0.08984375`, max `0.08984375`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.08984375`, last `0.08984375`, min `0.08984375`, max `0.08984375`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.08984375`, last `0.08984375`, min `0.08984375`, max `0.08984375`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.08984375`, last `0.08984375`, min `0.08984375`, max `0.08984375`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.08984375`, last `0.08984375`, min `0.08984375`, max `0.08984375`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.0768229216337204`, last `0.0768229216337204`, min `0.0768229216337204`, max `0.0768229216337204`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.0768229216337204`, last `0.0768229216337204`, min `0.0768229216337204`, max `0.0768229216337204`
- `Episode/cube_tap_success_rate`: n `1`, first `0.0729166716337204`, last `0.0729166716337204`, min `0.0729166716337204`, max `0.0729166716337204`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.9869791865348816`, last `0.9869791865348816`, min `0.9869791865348816`, max `0.9869791865348816`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.01302083395421505`, last `0.01302083395421505`, min `0.01302083395421505`, max `0.01302083395421505`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.01302083395421505`, last `0.01302083395421505`, min `0.01302083395421505`, max `0.01302083395421505`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `1.3096122529532295e-05`, last `1.3096122529532295e-05`, min `1.3096122529532295e-05`, max `1.3096122529532295e-05`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.0003893479588441551`, last `0.0003893479588441551`, min `0.0003893479588441551`, max `0.0003893479588441551`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.0167599655687809`, last `-0.0167599655687809`, min `-0.0167599655687809`, max `-0.0167599655687809`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.08984375`, last `0.08984375`, min `0.08984375`, max `0.08984375`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
