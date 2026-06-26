# D266 D256 State Sequence Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- tap contact proxy mode: `link5_collision_aabb`
- steps/envs/hold_steps: `580` / `32` / `1`
- contact rate: `1.0`
- first contact step min: `0`
- TCP-threshold contact rate: `0.0`
- tap useful rate: `1.0`
- min TCP-cube distance mean/min/max: `0.07699309289455414` / `0.06270913034677505` / `0.1001250371336937`
- min contact face-gap abs mean/min/max: `0.0` / `0.0` / `0.0`
- max disp along mean/min/max: `0.0074400329031050205` / `0.0034220367670059204` / `0.018024206161499023`

Interpretation: This writes D256 recorded arm/cube states into the live 10cm env and measures tap contact with the selected tap_contact_proxy_mode, while also logging the older tcp_cube_dist threshold. The D247/D256 visual dataset was rendered through the Candidate6 contract, which uses link5_collision_aabb rather than raw tcp_point.
