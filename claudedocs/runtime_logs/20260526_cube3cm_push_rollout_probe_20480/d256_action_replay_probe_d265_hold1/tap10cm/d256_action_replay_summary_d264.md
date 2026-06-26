# D264 D256 Action Replay Probe

- status: `PASS_PROBE_EXECUTED`
- teacher used: `False`
- steps/envs/hold_steps: `580` / `32` / `1`
- selected episode range/count: `1..999` / `32`
- contact rate: `0.0`
- first contact step min: `-1`
- min TCP-cube distance mean/min/max: `0.0751647800207138` / `0.06175459548830986` / `0.09928955137729645`
- max disp along mean/min/max: `0.006831126753240824` / `9.298324584960938e-06` / `0.016594409942626953`
- max target jump abs mean/max: `0.11397312581539154` / `0.16101033985614777`

Interpretation: This replays D256 state+joint_delta targets directly in the live 10cm env. If this fails contact, the visual-data action/control contract does not directly reproduce in the current env under this replay timing.
