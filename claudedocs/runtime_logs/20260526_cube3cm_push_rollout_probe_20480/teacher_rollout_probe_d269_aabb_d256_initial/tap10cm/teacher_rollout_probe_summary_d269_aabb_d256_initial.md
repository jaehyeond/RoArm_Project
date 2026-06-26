# D269_AABB_D256_INITIAL Teacher Rollout Probe Summary - tap10cm

- status: `PASS_PROBE_EXECUTED`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- cube size x/z m: `0.1` / `0.1`
- steps/envs: `580` / `32`
- ik endpoint reset: `False`
- reset pose source: `d256_initial`
- initial feature outside train min/max rate: `0.0`
- initial feature outside train p01/p99 rate: `0.19328703703703703`
- fixed push dir x/y: `1.0` / `0.0`
- bc teacher feature target mode: `env_target`
- tap contact proxy mode: `link5_collision_aabb`
- contact rate: `0.71875`
- first contact step min: `0`
- TCP-threshold contact rate: `0.0`
- tap useful rate: `0.71875`
- first alpha > 0 step min: `300`
- first alpha == 1 step min: `519`
- min TCP-cube distance mean/min/max: `0.08348368108272552` / `0.06940185278654099` / `0.09543989598751068`
- max disp along mean/min/max: `0.0014523034915328026` / `9.059906005859375e-06` / `0.01252603530883789`
- raw delta clip exceed rate: `0.20877155172413794`
- action cap rate: `0.13050466954022988`
- feature outside train min/max rate: `0.16936063218390804`
- feature outside train p01/p99 rate: `0.24879869093231163`

Top feature alignment warnings:

- `arm_joint_0_rad` train [`-1.1069484949111938`, `1.5799118280410767`], env [`-1.9324408769607544`, `2.9017906188964844`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_1_rad` train [`0.14346005022525787`, `0.7167561650276184`], env [`-1.5707999467849731`, `1.5708379745483398`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_2_rad` train [`1.7809100151062012`, `2.9840071201324463`], env [`-0.5786268711090088`, `2.9492645263671875`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_3_rad` train [`-1.5615715980529785`, `1.1647570133209229`], env [`-0.40166985988616943`, `1.9200022220611572`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_4_rad` train [`-0.019176799803972244`, `0.872174859046936`], env [`-0.4223622679710388`, `2.4753222465515137`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_z_m` train [`0.03788299858570099`, `0.06352606415748596`], env [`0.03788299858570099`, `0.0680786594748497`], outside_minmax=`True`, outside_p01p99=`True`
- `phase_alpha` train [`0.0`, `0.9982758620689656`], env [`0.0`, `1.0`], outside_minmax=`True`, outside_p01p99=`True`
- `target_to_cube_y_m` train [`-0.009521465748548508`, `0.010963410139083862`], env [`-0.006470918655395508`, `0.03020763397216797`], outside_minmax=`True`, outside_p01p99=`True`

Interpretation: This uses the 10cm CubeTap env geometry that matches the professor dataset object size better than D258's CubePush env. Use contact and feature-alignment metrics here to decide whether the teacher itself can produce plausible contact before any longer PPO. For tap10cm, contact_rate uses tap_contact_proxy_mode, while tcp_threshold_contact_rate reports the older tcp_cube_dist threshold. The BC teacher feature target is env_target, matching the D256 visual-log target_position_world_m feature contract rather than the online TCP waypoint.
