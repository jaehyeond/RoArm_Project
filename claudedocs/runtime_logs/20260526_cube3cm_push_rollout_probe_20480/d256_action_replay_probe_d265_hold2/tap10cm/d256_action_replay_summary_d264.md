# D264 D256 Action Replay Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- steps/envs/hold_steps: `580` / `32` / `2`
- selected episode range/count: `1..999` / `32`
- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max: `0.0751795768737793` / `0.061787448823451996` / `0.09905730187892914`
- max disp along mean/min/max: `0.006762760225683451` / `9.298324584960938e-06` / `0.01702404022216797`
- max target jump abs mean/max: `0.07886089384555817` / `0.09396450966596603`

Interpretation: This replays D256 state+joint_delta targets directly in the live 10cm env. If this fails contact, the visual-data action/control contract does not directly reproduce in the current env under this replay timing.
