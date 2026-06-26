# D260 TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d270_logs/cube10cm_d270_tap10cm_aabb_data_prior_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d270_logs/cube10cm_d270_tap10cm_aabb_data_prior_smoke --host 127.0.0.1 --port 6006`

## Issues

- no task success/contact signal in TensorBoard (max success/contact-like scalar=0.0)
- joint-delta cap rate too high: max=0.4782986640930176

## Warnings

- short run: Train/mean_reward has 1 points, promotion gate expects at least 10
- TCP-cube distance remains high: last=0.26218080520629883

## Selected Scalars

- `Train/mean_reward`: n `1`, first `29.420793533325195`, last `29.420793533325195`, min `29.420793533325195`, max `29.420793533325195`
- `Train/mean_episode_length`: n `1`, first `36.0`, last `36.0`, min `36.0`, max `36.0`
- `Loss/value_function`: n `2`, first `43.50541687011719`, last `14617.35546875`, min `43.50541687011719`, max `14617.35546875`
- `Loss/surrogate`: n `2`, first `-0.011165696196258068`, last `-0.0027191671542823315`, min `-0.011165696196258068`, max `-0.0027191671542823315`
- `Loss/entropy`: n `2`, first `7.175827503204346`, last `7.175779342651367`, min `7.175779342651367`, max `7.175827503204346`
- `Loss/learning_rate`: n `2`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `2`, first `0.8001441955566406`, last `0.800128161907196`, min `0.800128161907196`, max `0.8001441955566406`
- `Episode/cube_push_tcp_cube_dist_m`: n `2`, first `0.2926810383796692`, last `0.26218080520629883`, min `0.26218080520629883`, max `0.2926810383796692`
- `Episode/cube_push_joint_delta_abs_mean`: n `2`, first `0.029553567990660667`, last `0.03026289865374565`, min `0.029553567990660667`, max `0.03026289865374565`
- `Episode/cube_push_joint_delta_abs_max`: n `2`, first `0.037224266678094864`, last `0.03750266134738922`, min `0.037224266678094864`, max `0.03750266134738922`
- `Episode/cube_push_joint_delta_cap_rate`: n `2`, first `0.3183594048023224`, last `0.4782986640930176`, min `0.3183594048023224`, max `0.4782986640930176`
- `Episode/cube_push_action_abs_mean`: n `2`, first `0.7388391494750977`, last `0.756572425365448`, min `0.7388391494750977`, max `0.756572425365448`
- `Episode/cube_push_action_abs_max`: n `2`, first `0.9306066036224365`, last `0.9375664591789246`, min `0.9306066036224365`, max `0.9375664591789246`
- `Episode/cube_push_target_lead_limit_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_seen_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_reaction_seen_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_useful_seen_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_success_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_overshoot_seen_rate`: n `2`, first `0.0`, last `0.03125`, min `0.0`, max `0.03125`
- `Episode/cube_tap_max_disp_along_m`: n `2`, first `1.3772262718703132e-05`, last `0.0006061792373657227`, min `1.3772262718703132e-05`, max `0.0006061792373657227`
- `Episode/cube_tap_max_disp_xy_m`: n `2`, first `1.7419773939764127e-05`, last `0.0007881582714617252`, min `1.7419773939764127e-05`, max `0.0007881582714617252`
- `Episode/cube_tap_contact_proxy_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
