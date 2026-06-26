# D281_WARMSTART_USEFUL_STOP_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_smoke/cube10cm_d281_warmstart_useful_stop_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_distill_d281/tap10cm/ppo_warmstart_useful_stop_smoke/cube10cm_d281_warmstart_useful_stop_smoke --host 127.0.0.1 --port 6006`

## Issues

- joint-delta cap rate too high: max=0.3923611342906952

## Warnings

- short run: Train/mean_reward has 1 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.09564356505870819
- tap max displacement remains small: max=0.00010366198694100603

## Selected Scalars

- `Train/mean_reward`: n `1`, first `-13.199975967407227`, last `-13.199975967407227`, min `-13.199975967407227`, max `-13.199975967407227`
- `Train/mean_episode_length`: n `1`, first `2.9166667461395264`, last `2.9166667461395264`, min `2.9166667461395264`, max `2.9166667461395264`
- `Loss/value_function`: n `1`, first `146.1429901123047`, last `146.1429901123047`, min `146.1429901123047`, max `146.1429901123047`
- `Loss/surrogate`: n `1`, first `0.009710587561130524`, last `0.009710587561130524`, min `0.009710587561130524`, max `0.009710587561130524`
- `Loss/entropy`: n `1`, first `7.191607475280762`, last `7.191607475280762`, min `7.191607475280762`, max `7.191607475280762`
- `Loss/learning_rate`: n `1`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `1`, first `0.8022546768188477`, last `0.8022546768188477`, min `0.8022546768188477`, max `0.8022546768188477`
- `Episode/cube_push_tcp_cube_dist_m`: n `1`, first `0.09564356505870819`, last `0.09564356505870819`, min `0.09564356505870819`, max `0.09564356505870819`
- `Episode/cube_push_joint_delta_abs_mean`: n `1`, first `0.005678713321685791`, last `0.005678713321685791`, min `0.005678713321685791`, max `0.005678713321685791`
- `Episode/cube_push_joint_delta_abs_max`: n `1`, first `0.009799868799746037`, last `0.009799868799746037`, min `0.009799868799746037`, max `0.009799868799746037`
- `Episode/cube_push_joint_delta_cap_rate`: n `1`, first `0.3923611342906952`, last `0.3923611342906952`, min `0.3923611342906952`, max `0.3923611342906952`
- `Episode/cube_push_action_abs_mean`: n `1`, first `0.5800027251243591`, last `0.5800027251243591`, min `0.5800027251243591`, max `0.5800027251243591`
- `Episode/cube_push_action_abs_max`: n `1`, first `0.9624688029289246`, last `0.9624688029289246`, min `0.9624688029289246`, max `0.9624688029289246`
- `Episode/cube_push_target_lead_limit_rate`: n `1`, first `0.00390625`, last `0.00390625`, min `0.00390625`, max `0.00390625`
- `Episode/cube_push_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `1`, first `0.47823211550712585`, last `0.47823211550712585`, min `0.47823211550712585`, max `0.47823211550712585`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `1`, first `0.1662634015083313`, last `0.1662634015083313`, min `0.1662634015083313`, max `0.1662634015083313`
- `Episode/cube_push_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `1`, first `614.1328125`, last `614.1328125`, min `614.1328125`, max `614.1328125`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `1`, first `0.47823211550712585`, last `0.47823211550712585`, min `0.47823211550712585`, max `0.47823211550712585`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `1`, first `0.1662634015083313`, last `0.1662634015083313`, min `0.1662634015083313`, max `0.1662634015083313`
- `Episode/cube_tap_d256_reset_active_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `1`, first `614.1328125`, last `614.1328125`, min `614.1328125`, max `614.1328125`
- `Episode/bc_teacher_imitation_penalty`: n `1`, first `-0.023911606520414352`, last `-0.023911606520414352`, min `-0.023911606520414352`, max `-0.023911606520414352`
- `Episode/cube_tap_contact_seen_rate`: n `1`, first `0.10546875`, last `0.10546875`, min `0.10546875`, max `0.10546875`
- `Episode/cube_tap_contact_proxy_rate`: n `1`, first `0.10546875`, last `0.10546875`, min `0.10546875`, max `0.10546875`
- `Episode/cube_tap_reaction_seen_rate`: n `1`, first `0.10546875`, last `0.10546875`, min `0.10546875`, max `0.10546875`
- `Episode/cube_tap_reaction_signal_now_rate`: n `1`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `1`, first `0.10546875`, last `0.10546875`, min `0.10546875`, max `0.10546875`
- `Episode/cube_tap_reaction_now_rate`: n `1`, first `0.10546875`, last `0.10546875`, min `0.10546875`, max `0.10546875`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `1`, first `0.10546875`, last `0.10546875`, min `0.10546875`, max `0.10546875`
- `Episode/cube_tap_useful_now_rate`: n `1`, first `0.0950520858168602`, last `0.0950520858168602`, min `0.0950520858168602`, max `0.0950520858168602`
- `Episode/cube_tap_useful_seen_rate`: n `1`, first `0.0950520858168602`, last `0.0950520858168602`, min `0.0950520858168602`, max `0.0950520858168602`
- `Episode/cube_tap_success_rate`: n `1`, first `0.08203125`, last `0.08203125`, min `0.08203125`, max `0.08203125`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `1`, first `0.9856771230697632`, last `0.9856771230697632`, min `0.9856771230697632`, max `0.9856771230697632`
- `Episode/cube_tap_overshoot_now_rate`: n `1`, first `0.014322916977107525`, last `0.014322916977107525`, min `0.014322916977107525`, max `0.014322916977107525`
- `Episode/cube_tap_overshoot_seen_rate`: n `1`, first `0.014322916977107525`, last `0.014322916977107525`, min `0.014322916977107525`, max `0.014322916977107525`
- `Episode/cube_tap_max_disp_along_m`: n `1`, first `0.00010366198694100603`, last `0.00010366198694100603`, min `0.00010366198694100603`, max `0.00010366198694100603`
- `Episode/cube_tap_max_disp_xy_m`: n `1`, first `0.0006184530793689191`, last `0.0006184530793689191`, min `0.0006184530793689191`, max `0.0006184530793689191`
- `Episode/cube_tap_contact_face_gap_m`: n `1`, first `-0.02041502855718136`, last `-0.02041502855718136`, min `-0.02041502855718136`, max `-0.02041502855718136`
- `Episode/cube_tap_contact_lateral_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `1`, first `0.10546875`, last `0.10546875`, min `0.10546875`, max `0.10546875`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `1`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
