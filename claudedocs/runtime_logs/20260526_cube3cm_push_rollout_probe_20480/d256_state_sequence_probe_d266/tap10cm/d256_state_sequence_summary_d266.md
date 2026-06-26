# D266 D256 State Sequence Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- steps/envs/hold_steps: `580` / `32` / `1`
- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max: `0.07699309289455414` / `0.06270913034677505` / `0.1001250371336937`
- max disp along mean/min/max: `0.0074400329031050205` / `0.0034220367670059204` / `0.018024206161499023`

Interpretation: This writes D256 recorded arm/cube states into the live 10cm env and measures the current contact proxy. If this reaches contact while action replay does not, the controller/action replay contract is the main blocker. If it does not, the current contact proxy or tool/cube geometry is not equivalent to the visual-label contract.
