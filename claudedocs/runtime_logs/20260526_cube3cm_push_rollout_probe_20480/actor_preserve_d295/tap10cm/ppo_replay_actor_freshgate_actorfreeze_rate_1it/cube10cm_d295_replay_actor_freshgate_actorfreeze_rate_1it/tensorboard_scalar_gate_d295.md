# D295_REPLAY_ACTOR_FRESHGATE_ACTORFREEZE_RATE_1IT TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d295/tap10cm/ppo_replay_actor_freshgate_actorfreeze_rate_1it/cube10cm_d295_replay_actor_freshgate_actorfreeze_rate_1it --host 127.0.0.1 --port 6006`

## Issues

- missing core TensorBoard scalars: ['Train/mean_reward', 'Train/mean_episode_length']
- no tap contact/reaction/useful signal in TensorBoard (max tap contact-like scalar=0.8786637783050537)
- tap useful/success signal remains absent: max=0.8786637783050537

## Warnings

- short run: Train/mean_reward has 0 points, promotion gate expects at least 1
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.12133253365755081

## Selected Scalars

- `Loss/value_function`: n `1`, first `56418.1171875`, last `56418.1171875`, min `56418.1171875`, max `56418.1171875`
- `Loss/surrogate`: n `1`, first `0.17035825550556183`, last `0.17035825550556183`, min `0.17035825550556183`, max `0.17035825550556183`
- `Loss/entropy`: n `1`, first `-22.682758331298828`, last `-22.682758331298828`, min `-22.682758331298828`, max `-22.682758331298828`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.004999999422580004`, last `0.004999999422580004`, min `0.004999999422580004`, max `0.004999999422580004`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.12133253365755081`, last `0.12133253365755081`, min `0.12133253365755081`, max `0.12133253365755081`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.013181592337787151`, last `0.013181592337787151`, min `0.013181592337787151`, max `0.013181592337787151`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.03230322524905205`, last `0.03230322524905205`, min `0.03230322524905205`, max `0.03230322524905205`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.3370344340801239`, last `0.3370344340801239`, min `0.3370344340801239`, max `0.3370344340801239`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.8214231729507446`, last `0.8214231729507446`, min `0.8214231729507446`, max `0.8214231729507446`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `466.34375`, last `466.34375`, min `466.34375`, max `466.34375`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `466.34375`, last `466.34375`, min `466.34375`, max `466.34375`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.8786637783050537`, last `0.8786637783050537`, min `0.8786637783050537`, max `0.8786637783050537`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.596875011920929`, last `0.596875011920929`, min `0.596875011920929`, max `0.596875011920929`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.8786637783050537`, last `0.8786637783050537`, min `0.8786637783050537`, max `0.8786637783050537`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.8786637783050537`, last `0.8786637783050537`, min `0.8786637783050537`, max `0.8786637783050537`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.8786637783050537`, last `0.8786637783050537`, min `0.8786637783050537`, max `0.8786637783050537`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.8786637783050537`, last `0.8786637783050537`, min `0.8786637783050537`, max `0.8786637783050537`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.8710668087005615`, last `0.8710668087005615`, min `0.8710668087005615`, max `0.8710668087005615`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.8710668087005615`, last `0.8710668087005615`, min `0.8710668087005615`, max `0.8710668087005615`
- `Episode/cube_tap_success_rate`: n `1`, first `0.8786637783050537`, last `0.8786637783050537`, min `0.8786637783050537`, max `0.8786637783050537`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.9924030303955078`, last `0.9924030303955078`, min `0.9924030303955078`, max `0.9924030303955078`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.007596982643008232`, last `0.007596982643008232`, min `0.007596982643008232`, max `0.007596982643008232`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.007596982643008232`, last `0.007596982643008232`, min `0.007596982643008232`, max `0.007596982643008232`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `0.0025365781038999557`, last `0.0025365781038999557`, min `0.0025365781038999557`, max `0.0025365781038999557`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.002664614235982299`, last `0.002664614235982299`, min `0.002664614235982299`, max `0.002664614235982299`
- `Episode/cube_tap_max_disp_along_ge_1mm_rate`: n `1`, first `0.31244611740112305`, last `0.31244611740112305`, min `0.31244611740112305`, max `0.31244611740112305`
- `Episode/cube_tap_max_disp_xy_ge_1mm_rate`: n `1`, first `0.3125`, last `0.3125`, min `0.3125`, max `0.3125`
- `Episode/cube_tap_max_disp_along_ge_3mm_rate`: n `1`, first `0.25985991954803467`, last `0.25985991954803467`, min `0.25985991954803467`, max `0.25985991954803467`
- `Episode/cube_tap_max_disp_xy_ge_3mm_rate`: n `1`, first `0.2603987157344818`, last `0.2603987157344818`, min `0.2603987157344818`, max `0.2603987157344818`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.027426114305853844`, last `-0.027426114305853844`, min `-0.027426114305853844`, max `-0.027426114305853844`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.0003184943925589323`, last `0.0003184943925589323`, min `0.0003184943925589323`, max `0.0003184943925589323`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0003620779316406697`, last `0.0003620779316406697`, min `0.0003620779316406697`, max `0.0003620779316406697`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.8786637783050537`, last `0.8786637783050537`, min `0.8786637783050537`, max `0.8786637783050537`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
