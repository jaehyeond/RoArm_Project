# D279 Actor-vs-Teacher Trace

- verdict: `D279_ACTOR_TEACHER_TRACE_DIAGNOSTIC_BLOCKS_PPO_PROMOTION`
- diagnostic class: `actor_teacher_mismatch_plus_unsafe_physics`
- actor checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/model_0.pt`
- teacher checkpoint: `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/cube10cm_d257_state_action_teacher_clipped0040.pt`
- env id: `RoArm-CubeTap10cm-Direct-v0`
- steps/envs: `580` / `32`
- D256 reset active rate: `1.0`
- BC blend last: `0.0`
- actor-teacher MSE/MAE/cosine: `0.46601414680480957` / `0.5703011751174927` / `0.07783761620521545`
- actor clipped abs mean/max trace: `0.11846771989146183` / `1.0`
- teacher abs mean/max trace: `0.5554168882908236` / `1.0`
- actor raw clip exceed rate/max: `6.285919727564886e-05` / `0.010416666977107525`
- contact/useful/reaction seen: `0.875` / `0.5625` / `0.875`
- success/overshoot seen: `0.875` / `0.3125`
- max disp along mean/max: `0.0024283849634230137` / `0.018782615661621094`
- max disp xy mean/max: `0.020250540226697922` / `0.10077980160713196`
- max vertical offset mean/max: `0.023817647248506546` / `0.24940747022628784`
- min contact vertical offset mean/min/max: `0.0` / `0.0` / `0.0`
- joint delta cap last/max: `0.1145833432674408` / `0.15625`

## Issues

- actor-teacher action MSE above diagnostic threshold: 0.46601414680480957
- actor-teacher action cosine below diagnostic threshold: 0.07783761620521545
- tap overshoot seen rate too high: 0.3125
- tap vertical offset too high: max=0.24940747022628784

## Groups

- all: count `32`, mse `0.46601414680480957`, actor abs `0.11846773326396942`, teacher abs `0.5554170608520508`, max disp xy `0.020250540226697922`, max vertical `0.023817647248506546`
- overshoot: count `10`, mse `0.3936862051486969`, actor abs `0.13856154680252075`, teacher abs `0.48090705275535583`, max disp xy `0.059471823275089264`, max vertical `0.0`
- no_overshoot: count `22`, mse `0.4988904595375061`, actor abs `0.10933417826890945`, teacher abs `0.5892852544784546`, max disp xy `0.0024226864334195852`, max vertical `0.03464385122060776`
- useful: count `18`, mse `0.5252473950386047`, actor abs `0.10820580273866653`, teacher abs `0.6004774570465088`, max disp xy `0.002957139629870653`, max vertical `0.010126648470759392`
- not_useful: count `14`, mse `0.38985708355903625`, actor abs `0.13166163861751556`, teacher abs `0.49748238921165466`, max disp xy `0.04248492047190666`, max vertical `0.041420359164476395`
- vertical_over_threshold: count `5`, mse `0.4085454046726227`, actor abs `0.11813195794820786`, teacher abs `0.5760436058044434`, max disp xy `0.0005952191422693431`, max vertical `0.14238949120044708`
- vertical_ok: count `27`, mse `0.4766564965248108`, actor abs `0.11852991580963135`, teacher abs `0.5515973567962646`, max disp xy `0.023890415206551552`, max vertical `0.001859900075942278`

## Interpretation

This trace does not promote a policy. It only checks whether the frozen actor matches the D257 teacher sidecar under the same D256 reset/AABB contract used by D277-D278.
AABB/link5 contact is the primary contact proxy; raw TCP distance remains diagnostic only.
