# D281_WARMSTART_CONSERVATIVE_UPDATE_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_WARN_REQUIRES_MANUAL_REVIEW`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_conservative_update_smoke/cube10cm_d281_warmstart_conservative_update_smoke --host 127.0.0.1 --port 6006`

## Issues

- none

## Warnings

- short run: Train/mean_reward has 1 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.09270390123128891
- tap max displacement remains small: max=1.399593566020485e-05

## Selected Scalars

- `Train/mean_reward`: n `1`, first `-7.265285015106201`, last `-7.265285015106201`, min `-7.265285015106201`, max `-7.265285015106201`
- `Train/mean_episode_length`: n `1`, first `2.0`, last `2.0`, min `2.0`, max `2.0`
- `Loss/value_function`: n `1`, first `77.85538482666016`, last `77.85538482666016`, min `77.85538482666016`, max `77.85538482666016`
- `Loss/surrogate`: n `1`, first `0.016735004261136055`, last `0.016735004261136055`, min `0.016735004261136055`, max `0.016735004261136055`
- `Loss/entropy`: n `1`, first `-5.301711082458496`, last `-5.301711082458496`, min `-5.301711082458496`, max `-5.301711082458496`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.10000469535589218`, last `0.10000469535589218`, min `0.10000469535589218`, max `0.10000469535589218`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.09270390123128891`, last `0.09270390123128891`, min `0.09270390123128891`, max `0.09270390123128891`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.003523709252476692`, last `0.003523709252476692`, min `0.003523709252476692`, max `0.003523709252476692`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.00803494080901146`, last `0.00803494080901146`, min `0.00803494080901146`, max `0.00803494080901146`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.1549479365348816`, last `0.1549479365348816`, min `0.1549479365348816`, max `0.1549479365348816`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.1662440001964569`, last `0.1662440001964569`, min `0.1662440001964569`, max `0.1662440001964569`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.3793683648109436`, last `0.3793683648109436`, min `0.3793683648109436`, max `0.3793683648109436`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.04458475112915039`, last `0.04458475112915039`, min `0.04458475112915039`, max `0.04458475112915039`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.15608687698841095`, last `0.15608687698841095`, min `0.15608687698841095`, max `0.15608687698841095`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `569.1771240234375`, last `569.1771240234375`, min `569.1771240234375`, max `569.1771240234375`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.04458475112915039`, last `0.04458475112915039`, min `0.04458475112915039`, max `0.04458475112915039`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.15608687698841095`, last `0.15608687698841095`, min `0.15608687698841095`, max `0.15608687698841095`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `569.1771240234375`, last `569.1771240234375`, min `569.1771240234375`, max `569.1771240234375`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `-0.002229237463325262`, last `-0.002229237463325262`, min `-0.002229237463325262`, max `-0.002229237463325262`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.0833333358168602`, last `0.0833333358168602`, min `0.0833333358168602`, max `0.0833333358168602`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.0833333358168602`, last `0.0833333358168602`, min `0.0833333358168602`, max `0.0833333358168602`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.0833333358168602`, last `0.0833333358168602`, min `0.0833333358168602`, max `0.0833333358168602`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.0833333358168602`, last `0.0833333358168602`, min `0.0833333358168602`, max `0.0833333358168602`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.0833333358168602`, last `0.0833333358168602`, min `0.0833333358168602`, max `0.0833333358168602`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.0833333358168602`, last `0.0833333358168602`, min `0.0833333358168602`, max `0.0833333358168602`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.07421875`, last `0.07421875`, min `0.07421875`, max `0.07421875`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.07421875`, last `0.07421875`, min `0.07421875`, max `0.07421875`
- `Episode/cube_tap_success_rate`: n `1`, first `0.0690104216337204`, last `0.0690104216337204`, min `0.0690104216337204`, max `0.0690104216337204`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.9908854365348816`, last `0.9908854365348816`, min `0.9908854365348816`, max `0.9908854365348816`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.00911458395421505`, last `0.00911458395421505`, min `0.00911458395421505`, max `0.00911458395421505`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.00911458395421505`, last `0.00911458395421505`, min `0.00911458395421505`, max `0.00911458395421505`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `1.399593566020485e-05`, last `1.399593566020485e-05`, min `1.399593566020485e-05`, max `1.399593566020485e-05`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.00033495164825581014`, last `0.00033495164825581014`, min `0.00033495164825581014`, max `0.00033495164825581014`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.018261492252349854`, last `-0.018261492252349854`, min `-0.018261492252349854`, max `-0.018261492252349854`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.0833333358168602`, last `0.0833333358168602`, min `0.0833333358168602`, max `0.0833333358168602`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
