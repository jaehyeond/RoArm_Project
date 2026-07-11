# D332 canonical static collision discriminator

Verdict: `D332_G0A_PRESTEP_MIRROR_HULL_OVERLAP_RUNTIME_GRIPPER_CONTACT_SCENE_CONFOUNDED_MIXED`

| Metric | Result |
|---|---:|
| Default PhysX mirror-recook signed distance | `-6.236272 mm` |
| Mirror-recook verdict | `OVERLAP` |
| Mathematical full-hull precheck | `-6.363467 mm` |
| Raw STL negative-control distance | `4.273819 mm` |
| Contact sensor hard contract | `True` |
| Frozen filtered-support positive control | `False` |
| Posthoc net-reporter diagnostic | `True` |
| Baseline net force (last-50 median) | `7.063201 N` |
| Baseline max XY / tilt | `0.459483 mm / 0.672643 deg` |
| Initial ground penetration | `12.117000 mm` |
| First observed robot-contact post-step row | `0` |
| First observed link5-contact post-step row | `-1` |
| Runtime suspected link | `gripper_link` |
| Suspected-link peak force | `66.866266 N` |
| Object disturbance start physics step | `0` |
| Peak object speed | `0.315708 m/s` |
| Final object XY displacement | `10.282285 mm` |
| Final object tilt | `9.235161 deg` |
| Final actual TCP error | `3.413499 mm` |
| Commanded TCP error | `0.817812 mm` |

the pre-step default mirror recook overlaps, but the preregistered support control failed and the first runtime sample couples ground depenetration with robot contact.

This attempted final-pose discriminator is scene-confounded; it is neither a G0a pass nor a swept-path result.
