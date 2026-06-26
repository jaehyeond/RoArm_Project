# D261_ENVTARGET_POSX_IK Teacher Rollout Probe Summary - tap10cm

- status: `PASS_PROBE_EXECUTED`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- cube size x/z m: `0.1` / `0.1`
- steps/envs: `580` / `32`
- ik endpoint reset: `True`
- fixed push dir x/y: `1.0` / `0.0`
- bc teacher feature target mode: `env_target`
- contact rate: `0.0`
- first contact step min: `-1`
- first alpha > 0 step min: `300`
- first alpha == 1 step min: `519`
- min TCP-cube distance mean/min/max: `0.0902239978313446` / `0.07528560608625412` / `0.1327148675918579`
- max disp along mean/min/max: `1.2454720735549927` / `-0.027089953422546387` / `10.891912460327148`
- raw delta clip exceed rate: `0.6805603448275862`
- action cap rate: `0.3602280890804598`
- feature outside train min/max rate: `0.43064734993614306`
- feature outside train p01/p99 rate: `0.5013669380587484`

Top feature alignment warnings:

- `arm_joint_0_rad` train [`-1.1069484949111938`, `1.5799118280410767`], env [`-3.1415960788726807`, `3.1415884494781494`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_1_rad` train [`0.14346005022525787`, `0.7167561650276184`], env [`-1.5708211660385132`, `1.570817470550537`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_2_rad` train [`1.7809100151062012`, `2.9840071201324463`], env [`-0.9999988675117493`, `2.950005292892456`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_3_rad` train [`-1.5615715980529785`, `1.1647570133209229`], env [`-1.9200013875961304`, `1.9205161333084106`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_4_rad` train [`-0.019176799803972244`, `0.872174859046936`], env [`-3.1416001319885254`, `3.1415998935699463`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_x_m` train [`0.0894518494606018`, `0.4020536541938782`], env [`-2.653024673461914`, `11.11439323425293`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_y_m` train [`-0.10003045946359634`, `0.1500047743320465`], env [`-11.167580604553223`, `10.403951644897461`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_z_m` train [`0.03788299858570099`, `0.06352606415748596`], env [`0.03788299858570099`, `2.098430633544922`], outside_minmax=`True`, outside_p01p99=`True`

Interpretation: This uses the 10cm CubeTap env geometry that matches the professor dataset object size better than D258's CubePush env. Use contact and feature-alignment metrics here to decide whether the teacher itself can produce plausible contact before any longer PPO. The BC teacher feature target is env_target, matching the D256 visual-log target_position_world_m feature contract rather than the online TCP waypoint.
