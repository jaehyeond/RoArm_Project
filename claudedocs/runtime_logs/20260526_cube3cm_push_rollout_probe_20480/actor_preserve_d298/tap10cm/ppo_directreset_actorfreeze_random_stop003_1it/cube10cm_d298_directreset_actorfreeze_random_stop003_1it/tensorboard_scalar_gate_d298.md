# D298_DIRECTRESET_ACTORFREEZE_RANDOM_STOP003 TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d298/tap10cm/ppo_directreset_actorfreeze_random_stop003_1it/cube10cm_d298_directreset_actorfreeze_random_stop003_1it --host 127.0.0.1 --port 6006`

## Issues

- tap contact/reaction signal below threshold in TensorBoard (max=0.7029094696044922, threshold=0.9)
- tap useful/success signal below threshold in TensorBoard (max=0.04482758790254593, threshold=0.9)
- tap overshoot seen rate too high: max=0.7133082151412964

## Warnings

- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.14984621107578278

## Selected Scalars

- `Train/mean_reward`: n `1`, first `10.783509254455566`, last `10.783509254455566`, min `10.783509254455566`, max `10.783509254455566`
- `Train/mean_episode_length`: n `1`, first `64.90697479248047`, last `64.90697479248047`, min `64.90697479248047`, max `64.90697479248047`
- `Loss/value_function`: n `1`, first `28658414.0`, last `28658414.0`, min `28658414.0`, max `28658414.0`
- `Loss/surrogate`: n `1`, first `0.14215880632400513`, last `0.14215880632400513`, min `0.14215880632400513`, max `0.14215880632400513`
- `Loss/entropy`: n `1`, first `-22.670753479003906`, last `-22.670753479003906`, min `-22.670753479003906`, max `-22.670753479003906`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.004999999422580004`, last `0.004999999422580004`, min `0.004999999422580004`, max `0.004999999422580004`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.14984621107578278`, last `0.14984621107578278`, min `0.14984621107578278`, max `0.14984621107578278`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.013113240711390972`, last `0.013113240711390972`, min `0.013113240711390972`, max `0.013113240711390972`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.030380958691239357`, last `0.030380958691239357`, min `0.030380958691239357`, max `0.030380958691239357`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.33964672684669495`, last `0.33964672684669495`, min `0.33964672684669495`, max `0.33964672684669495`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.7767066955566406`, last `0.7767066955566406`, min `0.7767066955566406`, max `0.7767066955566406`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `504.66314697265625`, last `504.66314697265625`, min `504.66314697265625`, max `504.66314697265625`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `504.66314697265625`, last `504.66314697265625`, min `504.66314697265625`, max `504.66314697265625`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.7029094696044922`, last `0.7029094696044922`, min `0.7029094696044922`, max `0.7029094696044922`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.49687498807907104`, last `0.49687498807907104`, min `0.49687498807907104`, max `0.49687498807907104`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.7029094696044922`, last `0.7029094696044922`, min `0.7029094696044922`, max `0.7029094696044922`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.7029094696044922`, last `0.7029094696044922`, min `0.7029094696044922`, max `0.7029094696044922`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.7029094696044922`, last `0.7029094696044922`, min `0.7029094696044922`, max `0.7029094696044922`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.7029094696044922`, last `0.7029094696044922`, min `0.7029094696044922`, max `0.7029094696044922`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.09240301698446274`, last `0.09240301698446274`, min `0.09240301698446274`, max `0.09240301698446274`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.04482758790254593`, last `0.04482758790254593`, min `0.04482758790254593`, max `0.04482758790254593`
- `Episode/cube_tap_success_rate`: n `1`, first `0.0023168104235082865`, last `0.0023168104235082865`, min `0.0023168104235082865`, max `0.0023168104235082865`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.286691814661026`, last `0.286691814661026`, min `0.286691814661026`, max `0.286691814661026`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.665732741355896`, last `0.665732741355896`, min `0.665732741355896`, max `0.665732741355896`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.7133082151412964`, last `0.7133082151412964`, min `0.7133082151412964`, max `0.7133082151412964`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `0.01091606542468071`, last `0.01091606542468071`, min `0.01091606542468071`, max `0.01091606542468071`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.03478653356432915`, last `0.03478653356432915`, min `0.03478653356432915`, max `0.03478653356432915`
- `Episode/cube_tap_max_disp_along_ge_1mm_rate`: n `1`, first `0.3975215554237366`, last `0.3975215554237366`, min `0.3975215554237366`, max `0.3975215554237366`
- `Episode/cube_tap_max_disp_xy_ge_1mm_rate`: n `1`, first `0.7559267282485962`, last `0.7559267282485962`, min `0.7559267282485962`, max `0.7559267282485962`
- `Episode/cube_tap_max_disp_along_ge_3mm_rate`: n `1`, first `0.3674568831920624`, last `0.3674568831920624`, min `0.3674568831920624`, max `0.3674568831920624`
- `Episode/cube_tap_max_disp_xy_ge_3mm_rate`: n `1`, first `0.7559267282485962`, last `0.7559267282485962`, min `0.7559267282485962`, max `0.7559267282485962`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.013615148141980171`, last `-0.013615148141980171`, min `-0.013615148141980171`, max `-0.013615148141980171`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.011377677321434021`, last `0.011377677321434021`, min `0.011377677321434021`, max `0.011377677321434021`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.006965111009776592`, last `0.006965111009776592`, min `0.006965111009776592`, max `0.006965111009776592`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.7029094696044922`, last `0.7029094696044922`, min `0.7029094696044922`, max `0.7029094696044922`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_stop_after_disp_hold_rate`: n `1`, first `0.04251077398657799`, last `0.04251077398657799`, min `0.04251077398657799`, max `0.04251077398657799`
- `Episode/cube_tap_stop_after_disp_m`: n `1`, first `0.003000000026077032`, last `0.003000000026077032`, min `0.003000000026077032`, max `0.003000000026077032`
