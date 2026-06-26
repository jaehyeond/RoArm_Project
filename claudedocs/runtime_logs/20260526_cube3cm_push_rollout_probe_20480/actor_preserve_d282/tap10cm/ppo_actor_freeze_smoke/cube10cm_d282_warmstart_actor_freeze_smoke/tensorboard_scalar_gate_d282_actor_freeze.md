# D282_ACTOR_FREEZE_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_freeze_smoke/cube10cm_d282_warmstart_actor_freeze_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_freeze_smoke/cube10cm_d282_warmstart_actor_freeze_smoke --host 127.0.0.1 --port 6006`

## Issues

- none

## Warnings

- short run: Train/mean_reward has 1 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.09298501908779144
- tap max displacement remains small: max=1.3838871382176876e-05

## Selected Scalars

- `Train/mean_reward`: n `1`, first `-7.681184768676758`, last `-7.681184768676758`, min `-7.681184768676758`, max `-7.681184768676758`
- `Train/mean_episode_length`: n `1`, first `1.875`, last `1.875`, min `1.875`, max `1.875`
- `Loss/value_function`: n `1`, first `82.390625`, last `82.390625`, min `82.390625`, max `82.390625`
- `Loss/surrogate`: n `1`, first `0.03922593593597412`, last `0.03922593593597412`, min `0.03922593593597412`, max `0.03922593593597412`
- `Loss/entropy`: n `1`, first `-5.301923751831055`, last `-5.301923751831055`, min `-5.301923751831055`, max `-5.301923751831055`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.10000000149011612`, last `0.10000000149011612`, min `0.10000000149011612`, max `0.10000000149011612`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.09298501908779144`, last `0.09298501908779144`, min `0.09298501908779144`, max `0.09298501908779144`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.003511878428980708`, last `0.003511878428980708`, min `0.003511878428980708`, max `0.003511878428980708`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.008040973916649818`, last `0.008040973916649818`, min `0.008040973916649818`, max `0.008040973916649818`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.1634114682674408`, last `0.1634114682674408`, min `0.1634114682674408`, max `0.1634114682674408`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.16762679815292358`, last `0.16762679815292358`, min `0.16762679815292358`, max `0.16762679815292358`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.382690966129303`, last `0.382690966129303`, min `0.382690966129303`, max `0.382690966129303`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.044948749244213104`, last `0.044948749244213104`, min `0.044948749244213104`, max `0.044948749244213104`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.1557496190071106`, last `0.1557496190071106`, min `0.1557496190071106`, max `0.1557496190071106`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `579.9166870117188`, last `579.9166870117188`, min `579.9166870117188`, max `579.9166870117188`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.044948749244213104`, last `0.044948749244213104`, min `0.044948749244213104`, max `0.044948749244213104`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.1557496190071106`, last `0.1557496190071106`, min `0.1557496190071106`, max `0.1557496190071106`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `579.9166870117188`, last `579.9166870117188`, min `579.9166870117188`, max `579.9166870117188`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `-0.0022474373690783978`, last `-0.0022474373690783978`, min `-0.0022474373690783978`, max `-0.0022474373690783978`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.08203125`, last `0.08203125`, min `0.08203125`, max `0.08203125`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.08203125`, last `0.08203125`, min `0.08203125`, max `0.08203125`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.08203125`, last `0.08203125`, min `0.08203125`, max `0.08203125`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.08203125`, last `0.08203125`, min `0.08203125`, max `0.08203125`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.08203125`, last `0.08203125`, min `0.08203125`, max `0.08203125`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.08203125`, last `0.08203125`, min `0.08203125`, max `0.08203125`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.0729166716337204`, last `0.0729166716337204`, min `0.0729166716337204`, max `0.0729166716337204`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.0729166716337204`, last `0.0729166716337204`, min `0.0729166716337204`, max `0.0729166716337204`
- `Episode/cube_tap_success_rate`: n `1`, first `0.0690104216337204`, last `0.0690104216337204`, min `0.0690104216337204`, max `0.0690104216337204`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.9895833730697632`, last `0.9895833730697632`, min `0.9895833730697632`, max `0.9895833730697632`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.010416666977107525`, last `0.010416666977107525`, min `0.010416666977107525`, max `0.010416666977107525`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.010416666977107525`, last `0.010416666977107525`, min `0.010416666977107525`, max `0.010416666977107525`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `1.3838871382176876e-05`, last `1.3838871382176876e-05`, min `1.3838871382176876e-05`, max `1.3838871382176876e-05`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.0003425767063163221`, last `0.0003425767063163221`, min `0.0003425767063163221`, max `0.0003425767063163221`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.018720410764217377`, last `-0.018720410764217377`, min `-0.018720410764217377`, max `-0.018720410764217377`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.08203125`, last `0.08203125`, min `0.08203125`, max `0.08203125`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
