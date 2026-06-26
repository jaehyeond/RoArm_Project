# D264 D256 Action Replay Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- steps/envs/hold_steps: `580` / `32` / `3`
- selected episode range/count: `1..999` / `32`
- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max: `0.07518836855888367` / `0.06179572641849518` / `0.09923214465379715`
- max disp along mean/min/max: `0.006767723709344864` / `9.298324584960938e-06` / `0.017127275466918945`
- max target jump abs mean/max: `0.06703907251358032` / `0.09352636337280273`

Interpretation: This replays D256 state+joint_delta targets directly in the live 10cm env. If this fails contact, the visual-data action/control contract does not directly reproduce in the current env under this replay timing.
