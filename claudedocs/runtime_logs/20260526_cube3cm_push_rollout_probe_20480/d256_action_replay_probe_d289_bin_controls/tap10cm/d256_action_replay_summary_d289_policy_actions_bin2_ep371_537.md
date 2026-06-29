# D264 D256 Action Replay Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- teacher compared: `False`
- execution mode: `policy_actions`
- tap contact proxy mode: `link5_collision_aabb`
- action scale/max delta: `0.04` / `0.04`
- action smoothing/contact scales: `1.0` / `1.0` / `1.0`
- joint delta reference: `joint_pos`
- bc teacher phase timing: `linear_steps`
- steps/envs/hold_steps: `580` / `32` / `3`
- episode filter: `371`..`537`
- selected episode range/count: `371..537` / `32`
- contact rate: `1.0`
- first contact step min: `0`
- TCP-threshold contact rate: `0.0`
- tap useful rate: `1.0`
- min TCP-cube distance mean/min/max: `0.07890918850898743` / `0.06827820837497711` / `0.11469642072916031`
- max disp along mean/min/max: `0.004763957113027573` / `9.313225746154785e-06` / `0.016132831573486328`
- max target jump abs mean/max: `0.20537921786308289` / `0.3477451801300049`

- teacher recorded delta MSE/MAE/cosine mean: `None` / `None` / `None`
- teacher needed delta MSE/MAE/cosine mean: `None` / `None` / `None`
- teacher action abs mean/max: `None` / `None`

Interpretation: This replays D256 state+joint_delta targets directly in the live 10cm env. For tap10cm, contact_rate uses tap_contact_proxy_mode and tcp_threshold_contact_rate reports the older tcp_cube_dist threshold.
