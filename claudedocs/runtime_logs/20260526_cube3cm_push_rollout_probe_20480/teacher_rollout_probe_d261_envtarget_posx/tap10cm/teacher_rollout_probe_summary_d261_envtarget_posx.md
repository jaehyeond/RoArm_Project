# D261_ENVTARGET_POSX Teacher Rollout Probe Summary - tap10cm

- status: `PASS_PROBE_EXECUTED`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- cube size x/z m: `0.1` / `0.1`
- steps/envs: `580` / `32`
- ik endpoint reset: `False`
- fixed push dir x/y: `1.0` / `0.0`
- bc teacher feature target mode: `env_target`
- contact rate: `0.0`
- first contact step min: `-1`
- first alpha > 0 step min: `300`
- first alpha == 1 step min: `519`
- min TCP-cube distance mean/min/max: `0.2137620449066162` / `0.144382044672966` / `0.29925453662872314`
- max disp along mean/min/max: `1.163780689239502e-05` / `9.268522262573242e-06` / `2.765655517578125e-05`
- raw delta clip exceed rate: `0.7170689655172414`
- action cap rate: `0.37896012931034484`
- feature outside train min/max rate: `0.4267700351213282`
- feature outside train p01/p99 rate: `0.47317608556832697`

Top feature alignment warnings:

- `arm_joint_0_rad` train [`-1.1069484949111938`, `1.5799118280410767`], env [`-2.8085362911224365`, `3.1262781620025635`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_1_rad` train [`0.14346005022525787`, `0.7167561650276184`], env [`-1.5707999467849731`, `1.5708187818527222`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_2_rad` train [`1.7809100151062012`, `2.9840071201324463`], env [`-0.998024046421051`, `2.3849706649780273`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_3_rad` train [`-1.5615715980529785`, `1.1647570133209229`], env [`-0.32902300357818604`, `1.9200005531311035`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_4_rad` train [`-0.019176799803972244`, `0.872174859046936`], env [`-0.01717858947813511`, `3.1416003704071045`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_y_m` train [`-0.10003045946359634`, `0.1500047743320465`], env [`-0.12426316738128662`, `0.1227412223815918`], outside_minmax=`True`, outside_p01p99=`True`
- `phase_alpha` train [`0.0`, `0.9982758620689656`], env [`0.0`, `1.0`], outside_minmax=`True`, outside_p01p99=`True`
- `target_local_y_m` train [`-0.10000000149011612`, `0.15000000596046448`], env [`-0.12425628304481506`, `0.1227412223815918`], outside_minmax=`True`, outside_p01p99=`True`

Interpretation: This uses the 10cm CubeTap env geometry that matches the professor dataset object size better than D258's CubePush env. Use contact and feature-alignment metrics here to decide whether the teacher itself can produce plausible contact before any longer PPO. The BC teacher feature target is env_target, matching the D256 visual-log target_position_world_m feature contract rather than the online TCP waypoint.
