# D272 TensorBoard Scalar Gate

- verdict: `TENSORBOARD_GATE_FAIL_NO_PPO_PROMOTION`
- env kind: `tap10cm`
- log dir: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d272_logs/cube10cm_d272_tap10cm_aabb_bc_metrics_smoke`
- event files: `1`
- dashboard command: `conda run -n isaaclab tensorboard --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d272_logs/cube10cm_d272_tap10cm_aabb_bc_metrics_smoke --host 127.0.0.1 --port 6006`

## Issues

- no tap contact/reaction/useful signal in TensorBoard (max tap contact-like scalar=0.0)
- tap useful/success signal remains absent: max=0.0
- tap overshoot seen rate too high: max=0.0651041716337204
- joint-delta cap rate too high: max=0.330078125

## Warnings

- short run: Train/mean_reward has 2 points, promotion gate expects at least 10
- raw TCP-cube distance is high for tap/AABB diagnostic: last=0.26899513602256775
- tap max displacement remains small: max=0.0005519219557754695
- tap contact vertical offset remains high: last=0.1504015177488327
- mean reward decreased: first=-24.44062042236328 last=-70.10071563720703

## Selected Scalars

- `Train/mean_reward`: n `2`, first `-24.44062042236328`, last `-70.10071563720703`, min `-70.10071563720703`, max `-24.44062042236328`
- `Train/mean_episode_length`: n `2`, first `4.5`, last `17.66666603088379`, min `4.5`, max `17.66666603088379`
- `Loss/value_function`: n `2`, first `33301.984375`, last `62765.28515625`, min `33301.984375`, max `62765.28515625`
- `Loss/surrogate`: n `2`, first `-0.014689930714666843`, last `0.003032295498996973`, min `-0.014689930714666843`, max `0.003032295498996973`
- `Loss/entropy`: n `2`, first `7.1746978759765625`, last `7.174907207489014`, min `7.1746978759765625`, max `7.174907207489014`
- `Loss/learning_rate`: n `2`, first `9.999999747378752e-06`, last `9.999999747378752e-06`, min `9.999999747378752e-06`, max `9.999999747378752e-06`
- `Policy/mean_noise_std`: n `2`, first `0.8000050187110901`, last `0.8000316023826599`, min `0.8000050187110901`, max `0.8000316023826599`
- `Episode/cube_push_tcp_cube_dist_m`: n `2`, first `0.2935640513896942`, last `0.26899513602256775`, min `0.26899513602256775`, max `0.2935640513896942`
- `Episode/cube_push_joint_delta_abs_mean`: n `2`, first `0.028490189462900162`, last `0.02872391790151596`, min `0.028490189462900162`, max `0.02872391790151596`
- `Episode/cube_push_joint_delta_abs_max`: n `2`, first `0.03544992581009865`, last `0.03578248247504234`, min `0.03544992581009865`, max `0.03578248247504234`
- `Episode/cube_push_joint_delta_cap_rate`: n `2`, first `0.203125`, last `0.330078125`, min `0.203125`, max `0.330078125`
- `Episode/cube_push_action_abs_mean`: n `2`, first `0.7122547030448914`, last `0.7180979251861572`, min `0.7122547030448914`, max `0.7180979251861572`
- `Episode/cube_push_action_abs_max`: n `2`, first `0.886247992515564`, last `0.8945619463920593`, min `0.886247992515564`, max `0.8945619463920593`
- `Episode/cube_push_target_lead_limit_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_push_bc_teacher_blend_mean`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_push_bc_teacher_imitation_mse`: n `2`, first `0.9571873545646667`, last `0.9571402072906494`, min `0.9571402072906494`, max `0.9571873545646667`
- `Episode/cube_push_bc_teacher_action_abs_mean`: n `2`, first `0.7122547030448914`, last `0.7180979251861572`, min `0.7122547030448914`, max `0.7180979251861572`
- `Episode/cube_tap_bc_teacher_blend_mean`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_bc_teacher_imitation_mse`: n `2`, first `0.9571873545646667`, last `0.9571402072906494`, min `0.9571402072906494`, max `0.9571873545646667`
- `Episode/cube_tap_bc_teacher_action_abs_mean`: n `2`, first `0.7122547030448914`, last `0.7180979251861572`, min `0.7122547030448914`, max `0.7180979251861572`
- `Episode/bc_teacher_imitation_penalty`: n `2`, first `-4.7859368324279785`, last `-4.785701751708984`, min `-4.7859368324279785`, max `-4.785701751708984`
- `Episode/cube_tap_contact_seen_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_proxy_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_reaction_seen_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_reaction_signal_now_rate`: n `2`, first `1.0`, last `1.0`, min `1.0`, max `1.0`
- `Episode/cube_tap_reaction_contact_context_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_reaction_now_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_contact_reaction_seen_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_useful_now_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_useful_seen_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_success_rate`: n `2`, first `0.0`, last `0.0`, min `0.0`, max `0.0`
- `Episode/cube_tap_no_overshoot_seen_rate`: n `2`, first `0.94921875`, last `0.9348958730697632`, min `0.9348958730697632`, max `0.94921875`
- `Episode/cube_tap_overshoot_now_rate`: n `2`, first `0.05078125`, last `0.0651041716337204`, min `0.05078125`, max `0.0651041716337204`
- `Episode/cube_tap_overshoot_seen_rate`: n `2`, first `0.05078125`, last `0.0651041716337204`, min `0.05078125`, max `0.0651041716337204`
- `Episode/cube_tap_max_disp_along_m`: n `2`, first `0.00045672833221033216`, last `0.0005519219557754695`, min `0.00045672833221033216`, max `0.0005519219557754695`
- `Episode/cube_tap_max_disp_xy_m`: n `2`, first `0.0013907048851251602`, last `0.001989529700949788`, min `0.0013907048851251602`, max `0.001989529700949788`
- `Episode/cube_tap_contact_face_gap_m`: n `2`, first `0.007283593527972698`, last `-0.0012389846378937364`, min `-0.0012389846378937364`, max `0.007283593527972698`
- `Episode/cube_tap_contact_lateral_m`: n `2`, first `0.009999308735132217`, last `0.01822040230035782`, min `0.009999308735132217`, max `0.01822040230035782`
- `Episode/cube_tap_contact_vertical_offset_m`: n `2`, first `0.18639254570007324`, last `0.1504015177488327`, min `0.1504015177488327`, max `0.18639254570007324`
