# D282_CONSERVATIVE10_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d282/tap10cm/ppo_conservative10_smoke/cube10cm_d282_conservative10_smoke --host 127.0.0.1 --port 6006`

## Issues

- joint-delta cap rate too high: max=0.6664496660232544

## Warnings

- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.5229541063308716
- tap max displacement remains small: max=1.3993121683597565e-05

## Selected Scalars

- `Train/mean_reward`: n `10`, first `-6.725722312927246`, last `5.878491401672363`, min `-6.725722312927246`, max `5.878551006317139`
- `Train/mean_episode_length`: n `10`, first `1.9841269254684448`, last `1.0`, min `1.0`, max `5.71999979019165`
- `Loss/value_function`: n `10`, first `76.38603210449219`, last `1.5773062705993652`, min `1.5773062705993652`, max `76.38603210449219`
- `Loss/surrogate`: n `10`, first `0.021732661873102188`, last `0.0491613894701004`, min `0.00539695518091321`, max `0.0491613894701004`
- `Loss/entropy`: n `10`, first `-5.30189323425293`, last `-5.303498268127441`, min `-5.303498268127441`, max `-5.30189323425293`
- `Loss/learning_rate`: n `10`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `10`, first `0.09999814629554749`, last `0.09997402876615524`, min `0.09997253865003586`, max `0.09999814629554749`
- `Episode/cube_push_tcp_cube_dist_m`: n `10`, first `0.09281744062900543`, last `0.5229541063308716`, min `0.09281744062900543`, max `0.5229541063308716`
- `Episode/cube_push_joint_delta_abs_mean`: n `10`, first `0.0034653693437576294`, last `0.0071069421246647835`, min `0.0034653693437576294`, max `0.007298292592167854`
- `Episode/cube_push_joint_delta_abs_max`: n `10`, first `0.007992837578058243`, last `0.009758410044014454`, min `0.007992837578058243`, max `0.009758410044014454`
- `Episode/cube_push_joint_delta_cap_rate`: n `10`, first `0.15625`, last `0.6393229365348816`, min `0.15625`, max `0.6664496660232544`
- `Episode/cube_push_action_abs_mean`: n `10`, first `0.16475552320480347`, last `0.5155216455459595`, min `0.16475552320480347`, max `0.5155216455459595`
- `Episode/cube_push_action_abs_max`: n `10`, first `0.37411054968833923`, last `0.9052953720092773`, min `0.37411054968833923`, max `0.9052953720092773`
- `Episode/cube_push_target_lead_limit_rate`: n `10`, first `0.0`, last `0.0345052108168602`, min `0.0`, max `0.0345052108168602`
- `Episode/cube_push_bc_teacher_blend_mean`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `10`, first `0.04494510218501091`, last `0.10183489322662354`, min `0.04494510218501091`, max `0.1378413289785385`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `10`, first `0.15592549741268158`, last `0.6010936498641968`, min `0.15592549741268158`, max `0.6010936498641968`
- `Episode/cube_push_d256_reset_active_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `10`, first `581.3034057617188`, last `688.15625`, min `581.3034057617188`, max `688.15625`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `10`, first `0.04494510218501091`, last `0.10183489322662354`, min `0.04494510218501091`, max `0.1378413289785385`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `10`, first `0.15592549741268158`, last `0.6010936498641968`, min `0.15592549741268158`, max `0.6010936498641968`
- `Episode/cube_tap_d256_reset_active_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `10`, first `581.3034057617188`, last `688.15625`, min `581.3034057617188`, max `688.15625`
- `Episode/bc_teacher_imitation_penalty`: n `10`, first `-0.0022472550626844168`, last `-0.005091744940727949`, min `-0.006892066448926926`, max `-0.0022472550626844168`
- `Episode/cube_tap_contact_seen_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_contact_proxy_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_reaction_seen_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_reaction_signal_now_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_reaction_now_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_useful_now_rate`: n `10`, first `0.0716145858168602`, last `0.03125`, min `0.03125`, max `0.0716145858168602`
- `Episode/cube_tap_useful_seen_rate`: n `10`, first `0.0716145858168602`, last `0.03125`, min `0.03125`, max `0.0716145858168602`
- `Episode/cube_tap_success_rate`: n `10`, first `0.0690104216337204`, last `0.03125`, min `0.03125`, max `0.0690104216337204`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `10`, first `0.9895833730697632`, last `1.0`, min `0.9895833730697632`, max `1.0`
- `Episode/cube_tap_overshoot_now_rate`: n `10`, first `0.010416666977107525`, last `0.0`, min `0.0`, max `0.010416666977107525`
- `Episode/cube_tap_overshoot_seen_rate`: n `10`, first `0.010416666977107525`, last `0.0`, min `0.0`, max `0.010416666977107525`
- `Episode/cube_tap_max_disp_along_m`: n `10`, first `1.3993121683597565e-05`, last `1.150183379650116e-05`, min `1.1501251719892025e-05`, max `1.3993121683597565e-05`
- `Episode/cube_tap_max_disp_xy_m`: n `10`, first `0.00031507620587944984`, last `1.4510248547594529e-05`, min `1.4506615116260946e-05`, max `0.00031507620587944984`
- `Episode/cube_tap_contact_face_gap_m`: n `10`, first `-0.018527764827013016`, last `-0.3685229420661926`, min `-0.3685229420661926`, max `-0.018527764827013016`
- `Episode/cube_tap_contact_lateral_m`: n `10`, first `0.0`, last `0.024203382432460785`, min `0.0`, max `0.03835725039243698`
- `Episode/cube_tap_contact_vertical_offset_m`: n `10`, first `0.0`, last `0.04646040126681328`, min `0.0`, max `0.04748968407511711`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
