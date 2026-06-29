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
- episode filter: `209`..`370`
- selected episode range/count: `209..370` / `32`
- contact rate: `1.0`
- first contact step min: `0`
- TCP-threshold contact rate: `0.0`
- tap useful rate: `1.0`
- min TCP-cube distance mean/min/max: `0.07742948830127716` / `0.07033134996891022` / `0.10138150304555893`
- max disp along mean/min/max: `0.0045182243920862675` / `1.3589859008789062e-05` / `0.015462875366210938`
- max target jump abs mean/max: `0.24005042016506195` / `0.3856651782989502`

- teacher recorded delta MSE/MAE/cosine mean: `None` / `None` / `None`
- teacher needed delta MSE/MAE/cosine mean: `None` / `None` / `None`
- teacher action abs mean/max: `None` / `None`

Interpretation: This replays D256 state+joint_delta targets directly in the live 10cm env. For tap10cm, contact_rate uses tap_contact_proxy_mode and tcp_threshold_contact_rate reports the older tcp_cube_dist threshold.
