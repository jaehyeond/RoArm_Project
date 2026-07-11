# D333 sole-support static retest

Verdict: `D333_G0A_CLEAN_STATIC_BODY_ATTRIBUTION_MIXED_STOP`

| Metric | Result |
|---|---:|
| Stage/support hard contract | `True` |
| Sensor structural hard contract | `True` |
| Baseline support hard gate | `True` |
| First-step object z delta | `0.000003 mm` |
| TapTable Fz last-50 median | `7.063635 N` |
| Tail bottom/table max abs gap | `0.000135 mm` |
| Baseline max XY / tilt | `0.003774 mm / 0.003365 deg` |
| First robot contact step | `0` |
| First link5 contact step | `-1` |
| Object disturbance start step | `0` |
| Final/max XY displacement | `9.298849 / 12.598179 mm` |
| Final/max tilt | `3.881523 / 8.074518 deg` |
| Final TCP / commanded TCP error | `6.673174 / 0.817812 mm` |

a clean final-pose event or object disturbance remains without an immediate sampled link5 event; late or absent link5 attribution cannot support body-specific repair.
