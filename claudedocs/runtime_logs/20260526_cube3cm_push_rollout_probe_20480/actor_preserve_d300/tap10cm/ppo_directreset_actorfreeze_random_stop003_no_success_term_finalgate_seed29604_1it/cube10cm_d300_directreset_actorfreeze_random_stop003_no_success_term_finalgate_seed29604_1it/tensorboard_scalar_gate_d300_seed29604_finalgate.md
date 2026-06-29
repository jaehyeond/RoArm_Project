# D300_SEED29604_FINALGATE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/cube10cm_d300_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d300/tap10cm/ppo_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it/cube10cm_d300_directreset_actorfreeze_random_stop003_no_success_term_finalgate_seed29604_1it --host 127.0.0.1 --port 6006`

## Issues

- collection-final contact/reaction below threshold: last=0.875, threshold=0.9
- collection-final useful below threshold: last=0.875, threshold=0.9

## Warnings

- missing Train episode scalars allowed for no-termination gate: ['Train/mean_reward', 'Train/mean_episode_length']
- short run: Train/mean_reward has 0 points, promotion gate expects at least 1
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.1253020167350769

## Selected Scalars

- `Loss/value_function`: n `1`, first `595.8927001953125`, last `595.8927001953125`, min `595.8927001953125`, max `595.8927001953125`
- `Loss/surrogate`: n `1`, first `0.3469434678554535`, last `0.3469434678554535`, min `0.3469434678554535`, max `0.3469434678554535`
- `Loss/entropy`: n `1`, first `-22.775827407836914`, last `-22.775827407836914`, min `-22.775827407836914`, max `-22.775827407836914`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.004999999422580004`, last `0.004999999422580004`, min `0.004999999422580004`, max `0.004999999422580004`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.1253020167350769`, last `0.1253020167350769`, min `0.1253020167350769`, max `0.1253020167350769`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.008671958930790424`, last `0.008671958930790424`, min `0.008671958930790424`, max `0.008671958930790424`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.019837673753499985`, last `0.019837673753499985`, min `0.019837673753499985`, max `0.019837673753499985`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.22207783162593842`, last `0.22207783162593842`, min `0.22207783162593842`, max `0.22207783162593842`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.5029023289680481`, last `0.5029023289680481`, min `0.5029023289680481`, max `0.5029023289680481`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `542.09375`, last `542.09375`, min `542.09375`, max `542.09375`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `542.09375`, last `542.09375`, min `542.09375`, max `542.09375`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.7738685607910156`, last `0.7738685607910156`, min `0.7738685607910156`, max `0.7738685607910156`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.615463376045227`, last `0.615463376045227`, min `0.615463376045227`, max `0.615463376045227`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.7738685607910156`, last `0.7738685607910156`, min `0.7738685607910156`, max `0.7738685607910156`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.7738685607910156`, last `0.7738685607910156`, min `0.7738685607910156`, max `0.7738685607910156`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.7738685607910156`, last `0.7738685607910156`, min `0.7738685607910156`, max `0.7738685607910156`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.7738685607910156`, last `0.7738685607910156`, min `0.7738685607910156`, max `0.7738685607910156`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.7738685607910156`, last `0.7738685607910156`, min `0.7738685607910156`, max `0.7738685607910156`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.7738685607910156`, last `0.7738685607910156`, min `0.7738685607910156`, max `0.7738685607910156`
- `Episode/cube_tap_success_rate`: n `1`, first `0.7738685607910156`, last `0.7738685607910156`, min `0.7738685607910156`, max `0.7738685607910156`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `0.0013290472561493516`, last `0.0013290472561493516`, min `0.0013290472561493516`, max `0.0013290472561493516`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.001594485598616302`, last `0.001594485598616302`, min `0.001594485598616302`, max `0.001594485598616302`
- `Episode/cube_tap_max_disp_along_ge_1mm_rate`: n `1`, first `0.42737069725990295`, last `0.42737069725990295`, min `0.42737069725990295`, max `0.42737069725990295`
- `Episode/cube_tap_max_disp_xy_ge_1mm_rate`: n `1`, first `0.4314655065536499`, last `0.4314655065536499`, min `0.4314655065536499`, max `0.4314655065536499`
- `Episode/cube_tap_max_disp_along_ge_3mm_rate`: n `1`, first `0.27990302443504333`, last `0.27990302443504333`, min `0.27990302443504333`, max `0.27990302443504333`
- `Episode/cube_tap_max_disp_xy_ge_3mm_rate`: n `1`, first `0.379741370677948`, last `0.379741370677948`, min `0.379741370677948`, max `0.379741370677948`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.021800467744469643`, last `-0.021800467744469643`, min `-0.021800467744469643`, max `-0.021800467744469643`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.0026908458676189184`, last `0.0026908458676189184`, min `0.0026908458676189184`, max `0.0026908458676189184`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0007719264831393957`, last `0.0007719264831393957`, min `0.0007719264831393957`, max `0.0007719264831393957`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.7738685607910156`, last `0.7738685607910156`, min `0.7738685607910156`, max `0.7738685607910156`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_stop_after_disp_hold_rate`: n `1`, first `0.3787715435028076`, last `0.3787715435028076`, min `0.3787715435028076`, max `0.3787715435028076`
- `Episode/cube_tap_stop_after_disp_m`: n `1`, first `0.003000000026077032`, last `0.003000000026077032`, min `0.003000000026077032`, max `0.003000000026077032`
- `CollectionFinal/cube_tap_contact_seen_rate`: n `1`, first `0.875`, last `0.875`, min `0.875`, max `0.875`
- `CollectionFinal/cube_tap_reaction_seen_rate`: n `1`, first `0.875`, last `0.875`, min `0.875`, max `0.875`
- `CollectionFinal/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.875`, last `0.875`, min `0.875`, max `0.875`
- `CollectionFinal/cube_tap_useful_seen_rate`: n `1`, first `0.875`, last `0.875`, min `0.875`, max `0.875`
- `CollectionFinal/cube_tap_success_rate`: n `1`, first `0.875`, last `0.875`, min `0.875`, max `0.875`
- `CollectionFinal/cube_tap_overshoot_seen_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `CollectionFinal/cube_tap_max_disp_along_m`: n `1`, first `0.00201477762311697`, last `0.00201477762311697`, min `0.00201477762311697`, max `0.00201477762311697`
- `CollectionFinal/cube_tap_max_disp_xy_m`: n `1`, first `0.0023780229967087507`, last `0.0023780229967087507`, min `0.0023780229967087507`, max `0.0023780229967087507`
- `CollectionFinal/cube_tap_max_disp_along_max_m`: n `1`, first `0.005473017692565918`, last `0.005473017692565918`, min `0.005473017692565918`, max `0.005473017692565918`
- `CollectionFinal/cube_tap_max_disp_xy_max_m`: n `1`, first `0.006142078433185816`, last `0.006142078433185816`, min `0.006142078433185816`, max `0.006142078433185816`
- `CollectionFinal/cube_tap_max_disp_along_ge_1mm_rate`: n `1`, first `0.625`, last `0.625`, min `0.625`, max `0.625`
- `CollectionFinal/cube_tap_max_disp_xy_ge_1mm_rate`: n `1`, first `0.65625`, last `0.65625`, min `0.65625`, max `0.65625`
- `CollectionFinal/cube_tap_max_disp_along_ge_3mm_rate`: n `1`, first `0.4375`, last `0.4375`, min `0.4375`, max `0.4375`
- `CollectionFinal/cube_tap_max_disp_xy_ge_3mm_rate`: n `1`, first `0.5625`, last `0.5625`, min `0.5625`, max `0.5625`
- `CollectionFinal/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `CollectionFinal/cube_push_joint_delta_cap_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
