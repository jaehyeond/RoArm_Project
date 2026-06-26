# D282_ACTOR_PRESERVE095_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_preserve095_smoke/cube10cm_d282_warmstart_actor_preserve095_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_actor_preserve095_smoke/cube10cm_d282_warmstart_actor_preserve095_smoke --host 127.0.0.1 --port 6006`

## Issues

- none

## Warnings

- short run: Train/mean_reward has 1 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.09249413013458252
- tap max displacement remains small: max=1.4015435226610862e-05

## Selected Scalars

- `Train/mean_reward`: n `1`, first `-5.474422454833984`, last `-5.474422454833984`, min `-5.474422454833984`, max `-5.474422454833984`
- `Train/mean_episode_length`: n `1`, first `1.8196721076965332`, last `1.8196721076965332`, min `1.8196721076965332`, max `1.8196721076965332`
- `Loss/value_function`: n `1`, first `64.9226303100586`, last `64.9226303100586`, min `64.9226303100586`, max `64.9226303100586`
- `Loss/surrogate`: n `1`, first `0.029469916597008705`, last `0.029469916597008705`, min `0.029469916597008705`, max `0.029469916597008705`
- `Loss/entropy`: n `1`, first `-5.301975250244141`, last `-5.301975250244141`, min `-5.301975250244141`, max `-5.301975250244141`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.09999967366456985`, last `0.09999967366456985`, min `0.09999967366456985`, max `0.09999967366456985`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.09249413013458252`, last `0.09249413013458252`, min `0.09249413013458252`, max `0.09249413013458252`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.003562162397429347`, last `0.003562162397429347`, min `0.003562162397429347`, max `0.003562162397429347`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.008111853152513504`, last `0.008111853152513504`, min `0.008111853152513504`, max `0.008111853152513504`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.157986119389534`, last `0.157986119389534`, min `0.157986119389534`, max `0.157986119389534`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.16717946529388428`, last `0.16717946529388428`, min `0.16717946529388428`, max `0.16717946529388428`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.38311609625816345`, last `0.38311609625816345`, min `0.38311609625816345`, max `0.38311609625816345`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.04562274366617203`, last `0.04562274366617203`, min `0.04562274366617203`, max `0.04562274366617203`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.15915898978710175`, last `0.15915898978710175`, min `0.15915898978710175`, max `0.15915898978710175`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `563.1732177734375`, last `563.1732177734375`, min `563.1732177734375`, max `563.1732177734375`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.04562274366617203`, last `0.04562274366617203`, min `0.04562274366617203`, max `0.04562274366617203`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.15915898978710175`, last `0.15915898978710175`, min `0.15915898978710175`, max `0.15915898978710175`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `563.1732177734375`, last `563.1732177734375`, min `563.1732177734375`, max `563.1732177734375`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `-0.002281137276440859`, last `-0.002281137276440859`, min `-0.002281137276440859`, max `-0.002281137276440859`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.0794270858168602`, last `0.0794270858168602`, min `0.0794270858168602`, max `0.0794270858168602`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.0794270858168602`, last `0.0794270858168602`, min `0.0794270858168602`, max `0.0794270858168602`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.0794270858168602`, last `0.0794270858168602`, min `0.0794270858168602`, max `0.0794270858168602`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.0794270858168602`, last `0.0794270858168602`, min `0.0794270858168602`, max `0.0794270858168602`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.0794270858168602`, last `0.0794270858168602`, min `0.0794270858168602`, max `0.0794270858168602`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.0794270858168602`, last `0.0794270858168602`, min `0.0794270858168602`, max `0.0794270858168602`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.0716145858168602`, last `0.0716145858168602`, min `0.0716145858168602`, max `0.0716145858168602`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.0716145858168602`, last `0.0716145858168602`, min `0.0716145858168602`, max `0.0716145858168602`
- `Episode/cube_tap_success_rate`: n `1`, first `0.0677083358168602`, last `0.0677083358168602`, min `0.0677083358168602`, max `0.0677083358168602`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.9921875`, last `0.9921875`, min `0.9921875`, max `0.9921875`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.0078125`, last `0.0078125`, min `0.0078125`, max `0.0078125`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.0078125`, last `0.0078125`, min `0.0078125`, max `0.0078125`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `1.4015435226610862e-05`, last `1.4015435226610862e-05`, min `1.4015435226610862e-05`, max `1.4015435226610862e-05`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.00027852566563524306`, last `0.00027852566563524306`, min `0.00027852566563524306`, max `0.00027852566563524306`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.018170608207583427`, last `-0.018170608207583427`, min `-0.018170608207583427`, max `-0.018170608207583427`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.0794270858168602`, last `0.0794270858168602`, min `0.0794270858168602`, max `0.0794270858168602`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
