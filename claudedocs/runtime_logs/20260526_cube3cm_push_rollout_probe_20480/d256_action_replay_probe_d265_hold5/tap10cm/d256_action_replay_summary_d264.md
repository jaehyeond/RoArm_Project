# D264 D256 Action Replay Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- steps/envs/hold_steps: `580` / `32` / `5`
- selected episode range/count: `1..999` / `32`
- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max: `0.07529466599225998` / `0.0657862201333046` / `0.09934181720018387`
- max disp along mean/min/max: `0.0033459109254181385` / `9.298324584960938e-06` / `0.008709907531738281`
- max target jump abs mean/max: `0.06678351759910583` / `0.0937492847442627`

Interpretation: This replays D256 state+joint_delta targets directly in the live 10cm env. If this fails contact, the visual-data action/control contract does not directly reproduce in the current env under this replay timing.
