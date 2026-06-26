# D281_WARMSTART_USEFUL_STOP_NOISE010_RELOADSTD_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_noise010_reloadstd_smoke/cube10cm_d281_warmstart_useful_stop_noise010_reloadstd_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_noise010_reloadstd_smoke/cube10cm_d281_warmstart_useful_stop_noise010_reloadstd_smoke --host 127.0.0.1 --port 6006`

## Issues

- none

## Warnings

- short run: Train/mean_reward has 1 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.09254561364650726
- tap max displacement remains small: max=1.3730334103456698e-05

## Selected Scalars

- `Train/mean_reward`: n `1`, first `-7.812595367431641`, last `-7.812595367431641`, min `-7.812595367431641`, max `-7.812595367431641`
- `Train/mean_episode_length`: n `1`, first `1.875`, last `1.875`, min `1.875`, max `1.875`
- `Loss/value_function`: n `1`, first `83.53681945800781`, last `83.53681945800781`, min `83.53681945800781`, max `83.53681945800781`
- `Loss/surrogate`: n `1`, first `0.049505915492773056`, last `0.049505915492773056`, min `0.049505915492773056`, max `0.049505915492773056`
- `Loss/entropy`: n `1`, first `-5.294706344604492`, last `-5.294706344604492`, min `-5.294706344604492`, max `-5.294706344604492`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.10020245611667633`, last `0.10020245611667633`, min `0.10020245611667633`, max `0.10020245611667633`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.09254561364650726`, last `0.09254561364650726`, min `0.09254561364650726`, max `0.09254561364650726`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.0035780377220362425`, last `0.0035780377220362425`, min `0.0035780377220362425`, max `0.0035780377220362425`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.008108392357826233`, last `0.008108392357826233`, min `0.008108392357826233`, max `0.008108392357826233`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.1571180671453476`, last `0.1571180671453476`, min `0.1571180671453476`, max `0.1571180671453476`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.1674107015132904`, last `0.1674107015132904`, min `0.1674107015132904`, max `0.1674107015132904`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.3821067810058594`, last `0.3821067810058594`, min `0.3821067810058594`, max `0.3821067810058594`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.03936203941702843`, last `0.03936203941702843`, min `0.03936203941702843`, max `0.03936203941702843`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.1477719247341156`, last `0.1477719247341156`, min `0.1477719247341156`, max `0.1477719247341156`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `588.7252807617188`, last `588.7252807617188`, min `588.7252807617188`, max `588.7252807617188`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.03936203941702843`, last `0.03936203941702843`, min `0.03936203941702843`, max `0.03936203941702843`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.1477719247341156`, last `0.1477719247341156`, min `0.1477719247341156`, max `0.1477719247341156`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `588.7252807617188`, last `588.7252807617188`, min `588.7252807617188`, max `588.7252807617188`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `-0.001968102063983679`, last `-0.001968102063983679`, min `-0.001968102063983679`, max `-0.001968102063983679`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.0807291716337204`, last `0.0807291716337204`, min `0.0807291716337204`, max `0.0807291716337204`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.0807291716337204`, last `0.0807291716337204`, min `0.0807291716337204`, max `0.0807291716337204`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.0807291716337204`, last `0.0807291716337204`, min `0.0807291716337204`, max `0.0807291716337204`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.0807291716337204`, last `0.0807291716337204`, min `0.0807291716337204`, max `0.0807291716337204`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.0807291716337204`, last `0.0807291716337204`, min `0.0807291716337204`, max `0.0807291716337204`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.0807291716337204`, last `0.0807291716337204`, min `0.0807291716337204`, max `0.0807291716337204`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.0716145858168602`, last `0.0716145858168602`, min `0.0716145858168602`, max `0.0716145858168602`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.0716145858168602`, last `0.0716145858168602`, min `0.0716145858168602`, max `0.0716145858168602`
- `Episode/cube_tap_success_rate`: n `1`, first `0.0690104216337204`, last `0.0690104216337204`, min `0.0690104216337204`, max `0.0690104216337204`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.98828125`, last `0.98828125`, min `0.98828125`, max `0.98828125`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.01171875`, last `0.01171875`, min `0.01171875`, max `0.01171875`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.01171875`, last `0.01171875`, min `0.01171875`, max `0.01171875`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `1.3730334103456698e-05`, last `1.3730334103456698e-05`, min `1.3730334103456698e-05`, max `1.3730334103456698e-05`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.0003445661859586835`, last `0.0003445661859586835`, min `0.0003445661859586835`, max `0.0003445661859586835`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.01838250458240509`, last `-0.01838250458240509`, min `-0.01838250458240509`, max `-0.01838250458240509`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.0807291716337204`, last `0.0807291716337204`, min `0.0807291716337204`, max `0.0807291716337204`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
