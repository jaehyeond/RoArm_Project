# CUBE10CM_TOP_VIEW_TENSORBOARD_SCALAR_GATE_D304_SEED29801_TRACE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d304/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it/cube10cm_d304_directreset_actorfreeze_random_stop003_no_success_term_trace_seed29801_1it --host 127.0.0.1 --port 6006`

## Issues

- collection-final contact/reaction below threshold: last=0.84375, threshold=0.9
- collection-final useful below threshold: last=0.8125, threshold=0.9

## Warnings

- missing Train episode scalars allowed for no-termination gate: ['Train/mean_reward', 'Train/mean_episode_length']
- short run: Train/mean_reward has 0 points, promotion gate expects at least 1
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.12879188358783722

## Selected Scalars

- `Loss/value_function`: n `1`, first `16657.455078125`, last `16657.455078125`, min `16657.455078125`, max `16657.455078125`
- `Loss/surrogate`: n `1`, first `0.09777658432722092`, last `0.09777658432722092`, min `0.09777658432722092`, max `0.09777658432722092`
- `Loss/entropy`: n `1`, first `-22.674470901489258`, last `-22.674470901489258`, min `-22.674470901489258`, max `-22.674470901489258`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.004999999422580004`, last `0.004999999422580004`, min `0.004999999422580004`, max `0.004999999422580004`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.12879188358783722`, last `0.12879188358783722`, min `0.12879188358783722`, max `0.12879188358783722`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.008730879984796047`, last `0.008730879984796047`, min `0.008730879984796047`, max `0.008730879984796047`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.020993098616600037`, last `0.020993098616600037`, min `0.020993098616600037`, max `0.020993098616600037`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.22413606941699982`, last `0.22413606941699982`, min `0.22413606941699982`, max `0.22413606941699982`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.533484697341919`, last `0.533484697341919`, min `0.533484697341919`, max `0.533484697341919`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `427.5`, last `427.5`, min `427.5`, max `427.5`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `427.5`, last `427.5`, min `427.5`, max `427.5`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.7676724195480347`, last `0.7676724195480347`, min `0.7676724195480347`, max `0.7676724195480347`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.5615301728248596`, last `0.5615301728248596`, min `0.5615301728248596`, max `0.5615301728248596`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.7676724195480347`, last `0.7676724195480347`, min `0.7676724195480347`, max `0.7676724195480347`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.7676724195480347`, last `0.7676724195480347`, min `0.7676724195480347`, max `0.7676724195480347`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.7676724195480347`, last `0.7676724195480347`, min `0.7676724195480347`, max `0.7676724195480347`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.7676724195480347`, last `0.7676724195480347`, min `0.7676724195480347`, max `0.7676724195480347`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.7658405303955078`, last `0.7658405303955078`, min `0.7658405303955078`, max `0.7658405303955078`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.7658405303955078`, last `0.7658405303955078`, min `0.7658405303955078`, max `0.7658405303955078`
- `Episode/cube_tap_success_rate`: n `1`, first `0.7676724195480347`, last `0.7676724195480347`, min `0.7676724195480347`, max `0.7676724195480347`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.9981681108474731`, last `0.9981681108474731`, min `0.9981681108474731`, max `0.9981681108474731`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.0018318966031074524`, last `0.0018318966031074524`, min `0.0018318966031074524`, max `0.0018318966031074524`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.0018318966031074524`, last `0.0018318966031074524`, min `0.0018318966031074524`, max `0.0018318966031074524`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `0.001473818439990282`, last `0.001473818439990282`, min `0.001473818439990282`, max `0.001473818439990282`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.0016653359634801745`, last `0.0016653359634801745`, min `0.0016653359634801745`, max `0.0016653359634801745`
- `Episode/cube_tap_max_disp_along_ge_1mm_rate`: n `1`, first `0.40420258045196533`, last `0.40420258045196533`, min `0.40420258045196533`, max `0.40420258045196533`
- `Episode/cube_tap_max_disp_xy_ge_1mm_rate`: n `1`, first `0.4132543206214905`, last `0.4132543206214905`, min `0.4132543206214905`, max `0.4132543206214905`
- `Episode/cube_tap_max_disp_along_ge_3mm_rate`: n `1`, first `0.25339439511299133`, last `0.25339439511299133`, min `0.25339439511299133`, max `0.25339439511299133`
- `Episode/cube_tap_max_disp_xy_ge_3mm_rate`: n `1`, first `0.3439655303955078`, last `0.3439655303955078`, min `0.3439655303955078`, max `0.3439655303955078`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.01525175478309393`, last `-0.01525175478309393`, min `-0.01525175478309393`, max `-0.01525175478309393`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.003081616945564747`, last `0.003081616945564747`, min `0.003081616945564747`, max `0.003081616945564747`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0011654960690066218`, last `0.0011654960690066218`, min `0.0011654960690066218`, max `0.0011654960690066218`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.7676724195480347`, last `0.7676724195480347`, min `0.7676724195480347`, max `0.7676724195480347`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_stop_after_disp_hold_rate`: n `1`, first `0.34121766686439514`, last `0.34121766686439514`, min `0.34121766686439514`, max `0.34121766686439514`
- `Episode/cube_tap_stop_after_disp_m`: n `1`, first `0.003000000026077032`, last `0.003000000026077032`, min `0.003000000026077032`, max `0.003000000026077032`
- `CollectionFinal/cube_tap_contact_seen_rate`: n `1`, first `0.84375`, last `0.84375`, min `0.84375`, max `0.84375`
- `CollectionFinal/cube_tap_reaction_seen_rate`: n `1`, first `0.84375`, last `0.84375`, min `0.84375`, max `0.84375`
- `CollectionFinal/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.84375`, last `0.84375`, min `0.84375`, max `0.84375`
- `CollectionFinal/cube_tap_useful_seen_rate`: n `1`, first `0.8125`, last `0.8125`, min `0.8125`, max `0.8125`
- `CollectionFinal/cube_tap_success_rate`: n `1`, first `0.84375`, last `0.84375`, min `0.84375`, max `0.84375`
- `CollectionFinal/cube_tap_overshoot_seen_rate`: n `1`, first `0.03125`, last `0.03125`, min `0.03125`, max `0.03125`
- `CollectionFinal/cube_tap_max_disp_along_m`: n `1`, first `0.0034395549446344376`, last `0.0034395549446344376`, min `0.0034395549446344376`, max `0.0034395549446344376`
- `CollectionFinal/cube_tap_max_disp_xy_m`: n `1`, first `0.0037104845978319645`, last `0.0037104845978319645`, min `0.0037104845978319645`, max `0.0037104845978319645`
- `CollectionFinal/cube_tap_max_disp_along_max_m`: n `1`, first `0.05364418029785156`, last `0.05364418029785156`, min `0.05364418029785156`, max `0.05364418029785156`
- `CollectionFinal/cube_tap_max_disp_xy_max_m`: n `1`, first `0.053734444081783295`, last `0.053734444081783295`, min `0.053734444081783295`, max `0.053734444081783295`
- `CollectionFinal/cube_tap_max_disp_along_ge_1mm_rate`: n `1`, first `0.59375`, last `0.59375`, min `0.59375`, max `0.59375`
- `CollectionFinal/cube_tap_max_disp_xy_ge_1mm_rate`: n `1`, first `0.625`, last `0.625`, min `0.625`, max `0.625`
- `CollectionFinal/cube_tap_max_disp_along_ge_3mm_rate`: n `1`, first `0.4375`, last `0.4375`, min `0.4375`, max `0.4375`
- `CollectionFinal/cube_tap_max_disp_xy_ge_3mm_rate`: n `1`, first `0.5625`, last `0.5625`, min `0.5625`, max `0.5625`
- `CollectionFinal/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `CollectionFinal/cube_push_joint_delta_cap_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
