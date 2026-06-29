# D264 D256 Action Replay Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- tap contact proxy mode: `link5_collision_aabb`
- steps/envs/hold_steps: `580` / `8` / `3`
- selected episode range/count: `1..999` / `8`
- contact rate: `1.0`
- first contact step min: `0`
- TCP-threshold contact rate: `0.0`
- tap useful rate: `1.0`
- min TCP-cube distance mean/min/max: `0.07507079839706421` / `0.06764444708824158` / `0.09364736080169678`
- max disp along mean/min/max: `0.007791876792907715` / `9.357929229736328e-06` / `0.016317665576934814`
- max target jump abs mean/max: `0.06586561352014542` / `0.07087968289852142`

Interpretation: This replays D256 state+joint_delta targets directly in the live 10cm env. For tap10cm, contact_rate uses tap_contact_proxy_mode and tcp_threshold_contact_rate reports the older tcp_cube_dist threshold.
