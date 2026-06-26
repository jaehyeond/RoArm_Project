# D285_ACTORFREEZE_NOISE002_10_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d285/tap10cm/ppo_actorfreeze_noise002_10_smoke/cube10cm_d285_actorfreeze_noise002_10_smoke --host 127.0.0.1 --port 6006`

## Issues

- joint-delta cap rate too high: max=0.6536458730697632

## Warnings

- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.45665818452835083
- tap max displacement remains small: max=1.4040988389751874e-05

## Selected Scalars

- `Train/mean_reward`: n `10`, first `-6.012892246246338`, last `5.879622936248779`, min `-6.012892246246338`, max `5.879622936248779`
- `Train/mean_episode_length`: n `10`, first `1.5499999523162842`, last `1.0`, min `1.0`, max `5.329999923706055`
- `Loss/value_function`: n `10`, first `68.01371765136719`, last `1.6108853816986084`, min `1.6108853816986084`, max `68.01371765136719`
- `Loss/surrogate`: n `10`, first `0.0727522075176239`, last `0.23615744709968567`, min `0.03743408992886543`, max `0.23615744709968567`
- `Loss/entropy`: n `10`, first `-14.957725524902344`, last `-14.957691192626953`, min `-14.958006858825684`, max `-14.95693302154541`
- `Loss/learning_rate`: n `10`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `10`, first `0.019999999552965164`, last `0.019999999552965164`, min `0.019999999552965164`, max `0.019999999552965164`
- `Episode/cube_push_tcp_cube_dist_m`: n `10`, first `0.09288626909255981`, last `0.45665818452835083`, min `0.09288626909255981`, max `0.45665818452835083`
- `Episode/cube_push_joint_delta_abs_mean`: n `10`, first `0.003240172518417239`, last `0.007106025703251362`, min `0.003240172518417239`, max `0.007313653361052275`
- `Episode/cube_push_joint_delta_abs_max`: n `10`, first `0.008082783780992031`, last `0.009725427255034447`, min `0.008082783780992031`, max `0.009725427255034447`
- `Episode/cube_push_joint_delta_cap_rate`: n `10`, first `0.1525607705116272`, last `0.6254340410232544`, min `0.1525607705116272`, max `0.6536458730697632`
- `Episode/cube_push_action_abs_mean`: n `10`, first `0.1283589005470276`, last `0.4767536520957947`, min `0.1283589005470276`, max `0.4824260473251343`
- `Episode/cube_push_action_abs_max`: n `10`, first `0.333892822265625`, last `0.8579933643341064`, min `0.333892822265625`, max `0.8579933643341064`
- `Episode/cube_push_target_lead_limit_rate`: n `10`, first `0.0`, last `0.0190972238779068`, min `0.0`, max `0.0190972238779068`
- `Episode/cube_push_bc_teacher_blend_mean`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `10`, first `0.03232358396053314`, last `0.06847621500492096`, min `0.03232358396053314`, max `0.07570036500692368`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `10`, first `0.14858248829841614`, last `0.5513858795166016`, min `0.14858248829841614`, max `0.5513858795166016`
- `Episode/cube_push_d256_reset_active_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `10`, first `561.9700927734375`, last `660.375`, min `561.9700927734375`, max `660.375`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `10`, first `0.03232358396053314`, last `0.06847621500492096`, min `0.03232358396053314`, max `0.07570036500692368`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `10`, first `0.14858248829841614`, last `0.5513858795166016`, min `0.14858248829841614`, max `0.5513858795166016`
- `Episode/cube_tap_d256_reset_active_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `10`, first `561.9700927734375`, last `660.375`, min `561.9700927734375`, max `660.375`
- `Episode/bc_teacher_imitation_penalty`: n `10`, first `-0.0016161793610081077`, last `-0.0034238104708492756`, min `-0.003785018576309085`, max `-0.0016161793610081077`
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
- `Episode/cube_tap_max_disp_along_m`: n `10`, first `1.4040988389751874e-05`, last `1.2635253369808197e-05`, min `1.2600367881532293e-05`, max `1.4040988389751874e-05`
- `Episode/cube_tap_max_disp_xy_m`: n `10`, first `0.0002849685843102634`, last `1.601023177499883e-05`, min `1.6008640159270726e-05`, max `0.0002849685843102634`
- `Episode/cube_tap_contact_face_gap_m`: n `10`, first `-0.01855669543147087`, last `-0.30435556173324585`, min `-0.30435556173324585`, max `-0.01855669543147087`
- `Episode/cube_tap_contact_lateral_m`: n `10`, first `0.0`, last `0.013352042995393276`, min `0.0`, max `0.020709160715341568`
- `Episode/cube_tap_contact_vertical_offset_m`: n `10`, first `0.0`, last `0.03636502847075462`, min `0.0`, max `0.03636502847075462`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `10`, first `0.078125`, last `0.03125`, min `0.03125`, max `0.078125`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
