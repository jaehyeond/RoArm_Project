# D277 TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke --host 127.0.0.1 --port 6006`

## Issues

- none

## Warnings

- short run: Train/mean_reward has 1 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.20408329367637634

## Selected Scalars

- `Train/mean_reward`: n `1`, first `-3957.08154296875`, last `-3957.08154296875`, min `-3957.08154296875`, max `-3957.08154296875`
- `Train/mean_episode_length`: n `1`, first `599.0`, last `599.0`, min `599.0`, max `599.0`
- `Loss/value_function`: n `1`, first `59124.1484375`, last `59124.1484375`, min `59124.1484375`, max `59124.1484375`
- `Loss/surrogate`: n `1`, first `-0.003391646547242999`, last `-0.003391646547242999`, min `-0.003391646547242999`, max `-0.003391646547242999`
- `Loss/entropy`: n `1`, first `7.181930065155029`, last `7.181930065155029`, min `7.181930065155029`, max `7.181930065155029`
- `Loss/learning_rate`: n `1`, first `0.0003000000142492354`, last `0.0003000000142492354`, min `0.0003000000142492354`, max `0.0003000000142492354`
- `Policy/mean_noise_std`: n `1`, first `0.8021852970123291`, last `0.8021852970123291`, min `0.8021852970123291`, max `0.8021852970123291`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.20408329367637634`, last `0.20408329367637634`, min `0.20408329367637634`, max `0.20408329367637634`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.013785047456622124`, last `0.013785047456622124`, min `0.013785047456622124`, max `0.013785047456622124`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.02828078344464302`, last `0.02828078344464302`, min `0.02828078344464302`, max `0.02828078344464302`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.15915799140930176`, last `0.15915799140930176`, min `0.15915799140930176`, max `0.15915799140930176`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.34462615847587585`, last `0.34462615847587585`, min `0.34462615847587585`, max `0.34462615847587585`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.7070196270942688`, last `0.7070196270942688`, min `0.7070196270942688`, max `0.7070196270942688`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.66529381275177`, last `0.66529381275177`, min `0.66529381275177`, max `0.66529381275177`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.34462615847587585`, last `0.34462615847587585`, min `0.34462615847587585`, max `0.34462615847587585`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `465.8321838378906`, last `465.8321838378906`, min `465.8321838378906`, max `465.8321838378906`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.66529381275177`, last `0.66529381275177`, min `0.66529381275177`, max `0.66529381275177`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.34462615847587585`, last `0.34462615847587585`, min `0.34462615847587585`, max `0.34462615847587585`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `465.8321838378906`, last `465.8321838378906`, min `465.8321838378906`, max `465.8321838378906`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `-3.3264687061309814`, last `-3.3264687061309814`, min `-3.3264687061309814`, max `-3.3264687061309814`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.6662499904632568`, last `0.6662499904632568`, min `0.6662499904632568`, max `0.6662499904632568`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.3722916841506958`, last `0.3722916841506958`, min `0.3722916841506958`, max `0.3722916841506958`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.6662499904632568`, last `0.6662499904632568`, min `0.6662499904632568`, max `0.6662499904632568`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.6662499904632568`, last `0.6662499904632568`, min `0.6662499904632568`, max `0.6662499904632568`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.6662499904632568`, last `0.6662499904632568`, min `0.6662499904632568`, max `0.6662499904632568`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.6662499904632568`, last `0.6662499904632568`, min `0.6662499904632568`, max `0.6662499904632568`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.6549479365348816`, last `0.6549479365348816`, min `0.6549479365348816`, max `0.6549479365348816`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.6469791531562805`, last `0.6469791531562805`, min `0.6469791531562805`, max `0.6469791531562805`
- `Episode/cube_tap_success_rate`: n `1`, first `0.6652604341506958`, last `0.6652604341506958`, min `0.6652604341506958`, max `0.6652604341506958`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.9803125262260437`, last `0.9803125262260437`, min `0.9803125262260437`, max `0.9803125262260437`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.01171875`, last `0.01171875`, min `0.01171875`, max `0.01171875`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.019687499850988388`, last `0.019687499850988388`, min `0.019687499850988388`, max `0.019687499850988388`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `0.0018036302644759417`, last `0.0018036302644759417`, min `0.0018036302644759417`, max `0.0018036302644759417`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.0023345458321273327`, last `0.0023345458321273327`, min `0.0023345458321273327`, max `0.0023345458321273327`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.07378709316253662`, last `-0.07378709316253662`, min `-0.07378709316253662`, max `-0.07378709316253662`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.026030004024505615`, last `0.026030004024505615`, min `0.026030004024505615`, max `0.026030004024505615`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.015306632034480572`, last `0.015306632034480572`, min `0.015306632034480572`, max `0.015306632034480572`
