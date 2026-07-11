# D332 canonical static collision discriminator

Verdict: `D332_G0A_CONTACT_WITNESS_INVALID_MIXED`

| Metric | Result |
|---|---:|
| Actual PhysX-cooked hull signed distance | `-6.236272 mm` |
| Actual cooked hull verdict | `OVERLAP` |
| Mathematical full-hull precheck | `-6.363467 mm` |
| Raw STL negative-control distance | `4.273819 mm` |
| Contact sensor hard contract | `True` |
| Support-plane positive control | `False` |
| First robot contact physics step | `0` |
| First link5 contact physics step | `-1` |
| Object disturbance start physics step | `0` |
| Peak object speed | `0.315708 m/s` |
| Final object XY displacement | `10.282285 mm` |
| Final object tilt | `9.235161 deg` |
| Final actual TCP error | `3.413499 mm` |
| Commanded TCP error | `0.817812 mm` |

contact witness did not pass its hard runtime contract/positive control.

This is a canonical final-pose discriminator, not a G0a pass or swept-path result.
