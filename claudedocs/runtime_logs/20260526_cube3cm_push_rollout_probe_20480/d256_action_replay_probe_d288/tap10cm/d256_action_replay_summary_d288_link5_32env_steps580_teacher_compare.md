# D264 D256 Action Replay Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- teacher compared: `True`
- tap contact proxy mode: `link5_collision_aabb`
- joint delta reference: `joint_pos`
- bc teacher phase timing: `linear_steps`
- steps/envs/hold_steps: `580` / `32` / `3`
- selected episode range/count: `1..999` / `32`
- contact rate: `1.0`
- first contact step min: `0`
- TCP-threshold contact rate: `0.0`
- tap useful rate: `1.0`
- min TCP-cube distance mean/min/max: `0.07518836855888367` / `0.06179572641849518` / `0.09923214465379715`
- max disp along mean/min/max: `0.006767723709344864` / `9.298324584960938e-06` / `0.017127275466918945`
- max target jump abs mean/max: `0.06703907251358032` / `0.09352636337280273`

- teacher recorded delta MSE/MAE/cosine mean: `0.00013523723706308327` / `0.005776729541911005` / `0.8618926034148398`
- teacher needed delta MSE/MAE/cosine mean: `0.00012767873194851793` / `0.005706412376482682` / `0.8641119014343311`
- teacher action abs mean/max: `0.23752034487973514` / `1.0`

Interpretation: This replays D256 state+joint_delta targets directly in the live 10cm env. For tap10cm, contact_rate uses tap_contact_proxy_mode and tcp_threshold_contact_rate reports the older tcp_cube_dist threshold.
