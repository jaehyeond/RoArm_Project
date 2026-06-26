# D256 Feature Distribution Visualization D262

- source csv: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/rl_transition_preflight_d256/ppo_actor_prior_teacher_rows_d256.csv`
- rows / episodes: `142978` / `737`
- label counts: `{'clean_useful_tap': 142978}`
- joint delta abs > `0.04` rate: `0.14844941179761922`

## Plots

- `workspace_xy`: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_feature_distribution_viz_d262/d256_workspace_xy_distribution.png`
- `arm_joint_state`: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_feature_distribution_viz_d262/d256_arm_joint_state_distribution.png`
- `joint_delta`: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_feature_distribution_viz_d262/d256_joint_delta_distribution.png`
- `relative_geometry`: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_feature_distribution_viz_d262/d256_tcp_target_relative_geometry_distribution.png`
- `d261_overlay`: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_feature_distribution_viz_d262/d256_hist_with_d261_env_range_overlay.png`
- `d261_normalized_support`: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/d256_feature_distribution_viz_d262/d256_vs_d261_normalized_support_bars.png`

## Key Feature Quantiles

| feature | min | p01 | p50 | p99 | max |
|---|---:|---:|---:|---:|---:|
| `cube_local_x_m` | `0.0894518` | `0.090597` | `0.247833` | `0.383802` | `0.402054` |
| `cube_local_y_m` | `-0.10003` | `-0.100009` | `-0.0207382` | `0.149974` | `0.150005` |
| `tcp_local_z_m` | `0.0373942` | `0.0404628` | `0.0500002` | `0.0574649` | `0.0959378` |
| `target_to_tcp_x_m` | `0.0577249` | `0.0657086` | `0.0826081` | `0.10833` | `0.148975` |
| `target_to_tcp_y_m` | `-0.020482` | `-0.0133469` | `2.45906e-05` | `0.0139776` | `0.017242` |
| `target_to_tcp_z_m` | `-0.0580548` | `-0.0195819` | `-0.0121172` | `-0.0025798` | `0.000488773` |
| `arm_joint_0_rad` | `-1.10695` | `-0.956729` | `-0.115826` | `1.38942` | `1.57991` |
| `arm_joint_1_rad` | `0.14346` | `0.181148` | `0.332826` | `0.662846` | `0.716756` |
| `arm_joint_2_rad` | `1.78091` | `1.99427` | `2.32915` | `2.95166` | `2.98401` |
| `arm_joint_3_rad` | `-1.56157` | `-1.33024` | `0.37974` | `0.88395` | `1.16476` |
| `arm_joint_4_rad` | `-0.0191768` | `-0.0120018` | `0.000398823` | `0.220932` | `0.872175` |

## Reading The Overlay

- Blue histogram: D256 train-clean teacher feature distribution.
- Blue band: D256 p01-p99 support.
- Orange band: D261 live env range without IK reset.
- Red band: D261 live env range with IK reset.
- When orange/red bands sit outside the blue mass, the D257 MLP teacher is extrapolating.
