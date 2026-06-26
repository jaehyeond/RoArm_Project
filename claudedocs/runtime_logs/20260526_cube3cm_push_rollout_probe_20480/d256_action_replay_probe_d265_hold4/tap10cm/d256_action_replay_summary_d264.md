# D264 D256 Action Replay Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- steps/envs/hold_steps: `580` / `32` / `4`
- selected episode range/count: `1..999` / `32`
- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max: `0.07523223012685776` / `0.06307728588581085` / `0.09924843907356262`
- max disp along mean/min/max: `0.005708895158022642` / `9.298324584960938e-06` / `0.016956567764282227`
- max target jump abs mean/max: `0.06580531597137451` / `0.09397292137145996`

Interpretation: This replays D256 state+joint_delta targets directly in the live 10cm env. If this fails contact, the visual-data action/control contract does not directly reproduce in the current env under this replay timing.
