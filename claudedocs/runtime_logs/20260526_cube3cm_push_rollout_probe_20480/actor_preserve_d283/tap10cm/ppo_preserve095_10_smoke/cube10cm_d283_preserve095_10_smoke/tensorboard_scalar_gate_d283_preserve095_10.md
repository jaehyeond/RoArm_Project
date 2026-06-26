# D283_PRESERVE095_10_SMOKE TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/actor_preserve_d283/tap10cm/ppo_preserve095_10_smoke/cube10cm_d283_preserve095_10_smoke --host 127.0.0.1 --port 6006`

## Issues

- joint-delta cap rate too high: max=0.6579861640930176

## Warnings

- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.4975029230117798
- tap max displacement remains small: max=1.3879791367799044e-05

## Selected Scalars

- `Train/mean_reward`: n `10`, first `-6.631277561187744`, last `5.1462788581848145`, min `-6.631277561187744`, max `5.229940414428711`
- `Train/mean_episode_length`: n `10`, first `1.8548387289047241`, last `2.059999942779541`, min `1.8548387289047241`, max `6.349999904632568`
- `Loss/value_function`: n `10`, first `72.73954772949219`, last `1.5947767496109009`, min `1.5947767496109009`, max `72.73954772949219`
- `Loss/surrogate`: n `10`, first `0.012254511937499046`, last `0.060594748705625534`, min `0.005711695179343224`, max `0.060594748705625534`
- `Loss/entropy`: n `10`, first `-5.301486492156982`, last `-5.301990509033203`, min `-5.302125930786133`, max `-5.301486492156982`
- `Loss/learning_rate`: n `10`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `10`, first `0.10000063478946686`, last `0.09999976307153702`, min `0.09999953210353851`, max `0.10000063478946686`
- `Episode/cube_push_tcp_cube_dist_m`: n `10`, first `0.09302966296672821`, last `0.4975029230117798`, min `0.09302966296672821`, max `0.4975029230117798`
- `Episode/cube_push_joint_delta_abs_mean`: n `10`, first `0.0035021870862692595`, last `0.007307484280318022`, min `0.0035021870862692595`, max `0.007307484280318022`
- `Episode/cube_push_joint_delta_abs_max`: n `10`, first `0.008132177405059338`, last `0.009745802730321884`, min `0.008132177405059338`, max `0.009745802730321884`
- `Episode/cube_push_joint_delta_cap_rate`: n `10`, first `0.1588541865348816`, last `0.6579861640930176`, min `0.1588541865348816`, max `0.6579861640930176`
- `Episode/cube_push_action_abs_mean`: n `10`, first `0.16388319432735443`, last `0.5162656307220459`, min `0.16388319432735443`, max `0.5162656307220459`
- `Episode/cube_push_action_abs_max`: n `10`, first `0.37137994170188904`, last `0.8954037427902222`, min `0.37137994170188904`, max `0.8954037427902222`
- `Episode/cube_push_target_lead_limit_rate`: n `10`, first `0.0`, last `0.0297309011220932`, min `0.0`, max `0.0297309011220932`
- `Episode/cube_push_bc_teacher_blend_mean`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `10`, first `0.04660369083285332`, last `0.0750335305929184`, min `0.04660369083285332`, max `0.08587181568145752`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `10`, first `0.15462492406368256`, last `0.5728745460510254`, min `0.15462492406368256`, max `0.5728745460510254`
- `Episode/cube_push_d256_reset_active_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_d256_reset_episode_index_mean`: n `10`, first `563.4193115234375`, last `693.75`, min `563.4193115234375`, max `693.75`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `10`, first `0.04660369083285332`, last `0.0750335305929184`, min `0.04660369083285332`, max `0.08587181568145752`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `10`, first `0.15462492406368256`, last `0.5728745460510254`, min `0.15462492406368256`, max `0.5728745460510254`
- `Episode/cube_tap_d256_reset_active_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_d256_reset_episode_index_mean`: n `10`, first `563.4193115234375`, last `693.75`, min `563.4193115234375`, max `693.75`
- `Episode/bc_teacher_imitation_penalty`: n `10`, first `-0.0023301844485104084`, last `-0.00375167652964592`, min `-0.004293590784072876`, max `-0.0023301844485104084`
- `Episode/cube_tap_contact_seen_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_contact_proxy_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_reaction_seen_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_reaction_signal_now_rate`: n `10`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_reaction_now_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_useful_now_rate`: n `10`, first `0.0716145858168602`, last `0.03125`, min `0.03125`, max `0.0716145858168602`
- `Episode/cube_tap_useful_seen_rate`: n `10`, first `0.0716145858168602`, last `0.03125`, min `0.03125`, max `0.0716145858168602`
- `Episode/cube_tap_success_rate`: n `10`, first `0.0677083358168602`, last `0.03125`, min `0.03125`, max `0.0677083358168602`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `10`, first `0.9908854365348816`, last `1.0`, min `0.9908854365348816`, max `1.0`
- `Episode/cube_tap_overshoot_now_rate`: n `10`, first `0.00911458395421505`, last `0.0`, min `0.0`, max `0.00911458395421505`
- `Episode/cube_tap_overshoot_seen_rate`: n `10`, first `0.00911458395421505`, last `0.0`, min `0.0`, max `0.00911458395421505`
- `Episode/cube_tap_max_disp_along_m`: n `10`, first `1.3879791367799044e-05`, last `1.2043553397234064e-05`, min `1.2043553397234064e-05`, max `1.3879791367799044e-05`
- `Episode/cube_tap_max_disp_xy_m`: n `10`, first `0.0003092968836426735`, last `1.5233285921567585e-05`, min `1.5233285921567585e-05`, max `0.0003092968836426735`
- `Episode/cube_tap_contact_face_gap_m`: n `10`, first `-0.018620379269123077`, last `-0.3400287628173828`, min `-0.3400287628173828`, max `-0.018620379269123077`
- `Episode/cube_tap_contact_lateral_m`: n `10`, first `0.0`, last `0.034656066447496414`, min `0.0`, max `0.04125251621007919`
- `Episode/cube_tap_contact_vertical_offset_m`: n `10`, first `0.0`, last `0.047891631722450256`, min `0.0`, max `0.0482008159160614`
- `Episode/cube_tap_min_contact_vertical_offset_m`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_min_contact_vertical_finite_rate`: n `10`, first `0.0807291716337204`, last `0.03125`, min `0.03125`, max `0.0807291716337204`
- `Episode/cube_tap_stop_after_useful_hold_rate`: n `10`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
