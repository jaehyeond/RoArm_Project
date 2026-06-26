# D259 Teacher Rollout Probe Summary - push3cm

- status: `PASS_PROBE_EXECUTED`
- env id: `RoArm-CubePush-Direct-v0`
- cube size x/z m: `0.03` / `0.03`
- steps/envs: `580` / `32`
- contact rate: `0.0`
- first contact step min: `-1`
- first alpha > 0 step min: `220`
- first alpha == 1 step min: `309`
- min TCP-cube distance mean/min/max: `0.21149027347564697` / `0.09386380761861801` / `0.3410947620868683`
- max disp along mean/min/max: `0.0005538503755815327` / `-4.76837158203125e-06` / `0.01771417260169983`
- raw delta clip exceed rate: `1.0`
- action cap rate: `0.7770743534482759`
- feature outside train min/max rate: `0.5803001277139208`
- feature outside train p01/p99 rate: `0.6528715676883781`

Top feature alignment warnings:

- `arm_joint_0_rad` train [`-1.1069484949111938`, `1.5799118280410767`], env [`-0.019814368337392807`, `3.1416001319885254`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_1_rad` train [`0.14346005022525787`, `0.7167561650276184`], env [`-1.5708472728729248`, `1.570812463760376`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_2_rad` train [`1.7809100151062012`, `2.9840071201324463`], env [`-1.0000020265579224`, `2.9500014781951904`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_3_rad` train [`-1.5615715980529785`, `1.1647570133209229`], env [`-0.018116280436515808`, `1.92000150680542`], outside_minmax=`True`, outside_p01p99=`True`
- `arm_joint_4_rad` train [`-0.019176799803972244`, `0.872174859046936`], env [`-3.1416001319885254`, `0.019731801003217697`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_y_m` train [`-0.10003045946359634`, `0.1500047743320465`], env [`-0.12125790119171143`, `0.11936163902282715`], outside_minmax=`True`, outside_p01p99=`True`
- `cube_local_z_m` train [`0.03788299858570099`, `0.06352606415748596`], env [`0.0028830000665038824`, `0.02598906308412552`], outside_minmax=`True`, outside_p01p99=`True`
- `gripper_joint_rad` train [`-2.45716469393642e-09`, `4.0746224840404466e-05`], env [`-1.8588667272112502e-09`, `0.20997048914432526`], outside_minmax=`True`, outside_p01p99=`True`

Interpretation: This reproduces the D258 env kind: it is a 3cm CubePush env, not the 10cm professor cube env used by the D247-D257 data. Any feature mismatch here can explain weak D258 behavior and should not be treated as teacher failure on the intended 10cm task.
