# D284_PRESERVE095_NOISE002_10_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d284/tap10cm/ppo_preserve095_noise002_10_smoke/cube10cm_d284_preserve095_noise002_10_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d284/tap10cm/ppo_preserve095_noise002_10_smoke/cube10cm_d284_preserve095_noise002_10_smoke --host 127.0.0.1 --port 6006`

## Issues

- joint-delta cap rate too high: max=0.6430121660232544

## Warnings

- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.45948922634124756
- tap max displacement remains small: max=1.4015298802405596e-05

## Selected Scalars

- `Train/mean_reward`: n `10`, first `-6.0151824951171875`, last `5.879624843597412`, min `-6.0151824951171875`, max `5.879637718200684`
- `Train/mean_episode_length`: n `10`, first `1.5333333015441895`, last `1.0`, min `1.0`, max `7.050000190734863`
- `Loss/value_function`: n `10`, first `68.01058959960938`, last `1.603988766670227`, min `1.603988766670227`, max `68.01058959960938`
- `Loss/surrogate`: n `10`, first `0.05164673179388046`, last `0.16091398894786835`, min `0.036379776895046234`, max `0.3446614146232605`
- `Loss/entropy`: n `10`, first `-14.958710670471191`, last `-14.958158493041992`, min `-14.958710670471191`, max `-14.957696914672852`
- `Loss/learning_rate`: n `10`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `10`, first `0.01999981515109539`, last `0.020000148564577103`, min `0.01999981515109539`, max `0.02000034786760807`
- `Episode/cube_push_tcp_cube_dist_m`: n `10`, first `0.09290391206741333`, last `0.45948922634124756`, min `0.09290391206741333`, max `0.45948922634124756`
- `Episode/cube_push_joint_delta_abs_mean`: n `10`, first `0.0032422642689198256`, last `0.007132243365049362`, min `0.0032422642689198256`, max `0.007249555550515652`
- `Episode/cube_push_joint_delta_abs_max`: n `10`, first `0.008070521056652069`, last `0.009725581854581833`, min `0.008070521056652069`, max `0.009725581854581833`
- `Episode/cube_push_joint_delta_cap_rate`: n `10`, first `0.1532118171453476`, last `0.6404080390930176`, min `0.1532118171453476`, max `0.6430121660232544`
- `Episode/cube_push_action_abs_mean`: n `10`, first `0.12750154733657837`, last `0.48184072971343994`, min `0.12750154733657837`, max `0.48184072971343994`
- `Episode/cube_push_action_abs_max`: n `10`, first `0.33292075991630554`, last `0.8672488927841187`, min `0.33292075991630554`, max `0.8672488927841187`
- `Episode/cube_push_target_lead_limit_rate`: n `10`, first `0.0`, last `0.0249565988779068`, min `0.0`, max `0.0249565988779068`
- `Episode/cube_push_bc_teacher_blend_mean`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `10`, first `0.0328340046107769`, last `0.07686792314052582`, min `0.028097262606024742`, max `0.07697897404432297`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `10`, first `0.14938709139823914`, last `0.544418454170227`, min `0.14938709139823914`, max `0.544418454170227`
- `Episode/cube_push_d256_reset_active_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `10`, first `557.81640625`, last `683.625`, min `557.81640625`, max `683.625`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `10`, first `0.0328340046107769`, last `0.07686792314052582`, min `0.028097262606024742`, max `0.07697897404432297`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `10`, first `0.14938709139823914`, last `0.544418454170227`, min `0.14938709139823914`, max `0.544418454170227`
- `Episode/cube_tap_d256_reset_active_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `10`, first `557.81640625`, last `683.625`, min `557.81640625`, max `683.625`
- `Episode/bc_teacher_imitation_penalty`: n `10`, first `-0.0016417003935202956`, last `-0.0038433964364230633`, min `-0.0038489485159516335`, max `-0.0014048632001504302`
- `Episode/cube_tap_contact_seen_rate`: n `10`, first `0.078125`, last `0.03125`, min `0.03125`, max `0.078125`
- `Episode/cube_tap_contact_proxy_rate`: n `10`, first `0.078125`, last `0.03125`, min `0.03125`, max `0.078125`
- `Episode/cube_tap_reaction_seen_rate`: n `10`, first `0.078125`, last `0.03125`, min `0.03125`, max `0.078125`
- `Episode/cube_tap_reaction_signal_now_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `10`, first `0.078125`, last `0.03125`, min `0.03125`, max `0.078125`
- `Episode/cube_tap_reaction_now_rate`: n `10`, first `0.078125`, last `0.03125`, min `0.03125`, max `0.078125`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `10`, first `0.078125`, last `0.03125`, min `0.03125`, max `0.078125`
- `Episode/cube_tap_useful_now_rate`: n `10`, first `0.0690104216337204`, last `0.03125`, min `0.03125`, max `0.0690104216337204`
- `Episode/cube_tap_useful_seen_rate`: n `10`, first `0.0690104216337204`, last `0.03125`, min `0.03125`, max `0.0690104216337204`
- `Episode/cube_tap_success_rate`: n `10`, first `0.06640625`, last `0.03125`, min `0.03125`, max `0.06640625`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `10`, first `0.9908854365348816`, last `1.0`, min `0.9908854365348816`, max `1.0`
- `Episode/cube_tap_overshoot_now_rate`: n `10`, first `0.00911458395421505`, last `0.0`, min `0.0`, max `0.00911458395421505`
- `Episode/cube_tap_overshoot_seen_rate`: n `10`, first `0.00911458395421505`, last `0.0`, min `0.0`, max `0.00911458395421505`
- `Episode/cube_tap_max_disp_along_m`: n `10`, first `1.4015298802405596e-05`, last `1.206621527671814e-05`, min `1.2048831194988452e-05`, max `1.4015298802405596e-05`
- `Episode/cube_tap_max_disp_xy_m`: n `10`, first `0.000284951354842633`, last `1.5237928892020136e-05`, min `1.5235533282975666e-05`, max `0.000284951354842633`
- `Episode/cube_tap_contact_face_gap_m`: n `10`, first `-0.018597833812236786`, last `-0.3044513761997223`, min `-0.3044513761997223`, max `-0.018597833812236786`
- `Episode/cube_tap_contact_lateral_m`: n `10`, first `0.0`, last `0.017369244247674942`, min `0.0`, max `0.020486511290073395`
- `Episode/cube_tap_contact_vertical_offset_m`: n `10`, first `0.0`, last `0.03844398260116577`, min `0.0`, max `0.03844398260116577`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `10`, first `0.078125`, last `0.03125`, min `0.03125`, max `0.078125`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
