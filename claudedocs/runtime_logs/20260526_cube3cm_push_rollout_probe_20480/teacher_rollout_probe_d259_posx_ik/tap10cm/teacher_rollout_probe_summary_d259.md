# D259 Teacher Rollout Probe Summary - tap10cm

- status: `PASS_PROBE_EXECUTED`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- cube size x/z m: `0.1` / `0.1`
- steps/envs: `580` / `32`
- ik endpoint reset: `True`
- fixed push dir x/y: `1.0` / `0.0`
- contact rate: `0.0`
- first contact step min: `-1`
- first alpha > 0 step min: `300`
- first alpha == 1 step min: `519`
- min TCP-cube distance mean/min/max: `0.08880475163459778` / `0.06847080588340759` / `0.13267822563648224`
- max disp along mean/min/max: `1.254022479057312` / `-0.026723504066467285` / `11.039312362670898`
- raw delta clip exceed rate: `0.9999676724137931`
- action cap rate: `0.5438308189655172`
- feature outside train min/max rate: `0.60452187100894`
- feature outside train p01/p99 rate: `0.6507104086845467`

Top feature alignment warnings:

- `arm_joint_0_rad` train [`-1.1069484949111938`, `1.5799118280410767`], env [`-3.1415936946868896`, `0.4700818657875061`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_1_rad` train [`0.14346005022525787`, `0.7167561650276184`], env [`-1.5707892179489136`, `1.570796012878418`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_2_rad` train [`1.7809100151062012`, `2.9840071201324463`], env [`-1.000001072883606`, `2.3949878215789795`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_3_rad` train [`-1.5615715980529785`, `1.1647570133209229`], env [`-1.9200001955032349`, `1.9221649169921875`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_4_rad` train [`-0.019176799803972244`, `0.872174859046936`], env [`-3.1416001319885254`, `0.9614098072052002`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_x_m` train [`0.0894518494606018`, `0.4020536541938782`], env [`-2.546884059906006`, `11.26179313659668`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_y_m` train [`-0.10003045946359634`, `0.1500047743320465`], env [`-11.482237815856934`, `9.843143463134766`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_z_m` train [`0.03788299858570099`, `0.06352606415748596`], env [`0.03788299858570099`, `2.0933594703674316`], outside_minmax=`True`, outside_p01p99=`True`

Interpretation: This uses the 10cm CubeTap env geometry that matches the professor dataset object size better than D258's CubePush env. Use contact and feature-alignment metrics here to decide whether the teacher itself can produce plausible contact before any longer PPO.
