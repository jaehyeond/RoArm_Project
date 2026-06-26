# D259 Teacher Rollout Probe Summary - tap10cm

- status: `PASS_PROBE_EXECUTED`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- cube size x/z m: `0.1` / `0.1`
- steps/envs: `580` / `32`
- contact rate: `0.0`
- first contact step min: `-1`
- first alpha > 0 step min: `220`
- first alpha == 1 step min: `309`
- min TCP-cube distance mean/min/max: `0.18824803829193115` / `0.07045303285121918` / `0.3121436536312103`
- max disp along mean/min/max: `0.0033230045810341835` / `-2.384185791015625e-05` / `0.0329592227935791`
- raw delta clip exceed rate: `1.0`
- action cap rate: `0.7768139367816091`
- feature outside train min/max rate: `0.593532487228608`
- feature outside train p01/p99 rate: `0.6467712324393359`

Top feature alignment warnings:

- `arm_joint_0_rad` train [`-1.1069484949111938`, `1.5799118280410767`], env [`-3.141587734222412`, `3.1416001319885254`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_1_rad` train [`0.14346005022525787`, `0.7167561650276184`], env [`-1.5708472728729248`, `1.5708147287368774`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_2_rad` train [`1.7809100151062012`, `2.9840071201324463`], env [`-1.0000030994415283`, `2.9495866298675537`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_3_rad` train [`-1.5615715980529785`, `1.1647570133209229`], env [`-0.018116280436515808`, `1.9200005531311035`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_4_rad` train [`-0.019176799803972244`, `0.872174859046936`], env [`-3.890448808670044`, `1.346267819404602`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_y_m` train [`-0.10003045946359634`, `0.1500047743320465`], env [`-0.11206066608428955`, `0.12317907810211182`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_z_m` train [`0.03788299858570099`, `0.06352606415748596`], env [`0.03788299858570099`, `0.06883428245782852`], outside_minmax=`True`, outside_p01p99=`True`
- `gripper_joint_rad` train [`-2.45716469393642e-09`, `4.0746224840404466e-05`], env [`-0.12279818207025528`, `0.20575232803821564`], outside_minmax=`True`, outside_p01p99=`True`

Interpretation: This uses the 10cm CubeTap env geometry that matches the professor dataset object size better than D258's CubePush env. Use contact and feature-alignment metrics here to decide whether the teacher itself can produce plausible contact before any longer PPO.
