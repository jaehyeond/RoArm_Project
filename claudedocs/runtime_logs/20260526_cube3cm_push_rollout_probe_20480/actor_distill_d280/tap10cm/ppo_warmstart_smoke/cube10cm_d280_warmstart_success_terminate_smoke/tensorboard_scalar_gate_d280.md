# D280_WARMSTART_SUCCESS_TERMINATE_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/ppo_warmstart_smoke/cube10cm_d280_warmstart_success_terminate_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d280/tap10cm/ppo_warmstart_smoke/cube10cm_d280_warmstart_success_terminate_smoke --host 127.0.0.1 --port 6006`

## Issues

- joint-delta cap rate too high: max=0.3993055820465088

## Warnings

- short run: Train/mean_reward has 1 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.09469194710254669
- tap max displacement remains small: max=0.0002397357311565429

## Selected Scalars

- `Train/mean_reward`: n `1`, first `-14.207649230957031`, last `-14.207649230957031`, min `-14.207649230957031`, max `-14.207649230957031`
- `Train/mean_episode_length`: n `1`, first `2.853658437728882`, last `2.853658437728882`, min `2.853658437728882`, max `2.853658437728882`
- `Loss/value_function`: n `1`, first `895.8475341796875`, last `895.8475341796875`, min `895.8475341796875`, max `895.8475341796875`
- `Loss/surrogate`: n `1`, first `-0.0026019741781055927`, last `-0.0026019741781055927`, min `-0.0026019741781055927`, max `-0.0026019741781055927`
- `Loss/entropy`: n `1`, first `7.193819522857666`, last `7.193819522857666`, min `7.193819522857666`, max `7.193819522857666`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.8026366829872131`, last `0.8026366829872131`, min `0.8026366829872131`, max `0.8026366829872131`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.09469194710254669`, last `0.09469194710254669`, min `0.09469194710254669`, max `0.09469194710254669`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.005758419167250395`, last `0.005758419167250395`, min `0.005758419167250395`, max `0.005758419167250395`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.0098577244207263`, last `0.0098577244207263`, min `0.0098577244207263`, max `0.0098577244207263`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.3993055820465088`, last `0.3993055820465088`, min `0.3993055820465088`, max `0.3993055820465088`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.5723713636398315`, last `0.5723713636398315`, min `0.5723713636398315`, max `0.5723713636398315`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.9634577035903931`, last `0.9634577035903931`, min `0.9634577035903931`, max `0.9634577035903931`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.004991319961845875`, last `0.004991319961845875`, min `0.004991319961845875`, max `0.004991319961845875`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.46044886112213135`, last `0.46044886112213135`, min `0.46044886112213135`, max `0.46044886112213135`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.1677628606557846`, last `0.1677628606557846`, min `0.1677628606557846`, max `0.1677628606557846`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `617.10546875`, last `617.10546875`, min `617.10546875`, max `617.10546875`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.46044886112213135`, last `0.46044886112213135`, min `0.46044886112213135`, max `0.46044886112213135`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.1677628606557846`, last `0.1677628606557846`, min `0.1677628606557846`, max `0.1677628606557846`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `617.10546875`, last `617.10546875`, min `617.10546875`, max `617.10546875`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `-0.023022443056106567`, last `-0.023022443056106567`, min `-0.023022443056106567`, max `-0.023022443056106567`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.13671875`, last `0.13671875`, min `0.13671875`, max `0.13671875`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.13671875`, last `0.13671875`, min `0.13671875`, max `0.13671875`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.13671875`, last `0.13671875`, min `0.13671875`, max `0.13671875`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.13671875`, last `0.13671875`, min `0.13671875`, max `0.13671875`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.13671875`, last `0.13671875`, min `0.13671875`, max `0.13671875`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.13671875`, last `0.13671875`, min `0.13671875`, max `0.13671875`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.12109375`, last `0.12109375`, min `0.12109375`, max `0.12109375`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.12109375`, last `0.12109375`, min `0.12109375`, max `0.12109375`
- `Episode/cube_tap_success_rate`: n `1`, first `0.0846354216337204`, last `0.0846354216337204`, min `0.0846354216337204`, max `0.0846354216337204`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.9778646230697632`, last `0.9778646230697632`, min `0.9778646230697632`, max `0.9778646230697632`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.0221354179084301`, last `0.0221354179084301`, min `0.0221354179084301`, max `0.0221354179084301`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.0221354179084301`, last `0.0221354179084301`, min `0.0221354179084301`, max `0.0221354179084301`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `0.0002397357311565429`, last `0.0002397357311565429`, min `0.0002397357311565429`, max `0.0002397357311565429`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.000873955141287297`, last `0.000873955141287297`, min `0.000873955141287297`, max `0.000873955141287297`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.01957070268690586`, last `-0.01957070268690586`, min `-0.01957070268690586`, max `-0.01957070268690586`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
