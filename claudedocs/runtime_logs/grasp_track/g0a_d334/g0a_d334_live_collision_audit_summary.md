# D334 live collision shape / ownership audit

Verdict: `D334_G0A_ACTUAL_TOOL_OVERLAP_SUPPORTED`

| Gate | Result |
|---|---:|
| Frozen invariant contract | `True` |
| Stage/sensor contracts | `True` / `True` |
| Baseline replay hard gate | `True` |
| Ownership parity (all) | `True` |
| Step-0 replay parity | `True` |
| link5 `node_STL_BINARY_` cook parity (certified) | `True` (`True`) |
| gripper_link `node_STL_BINARY_` cook parity (certified) | `False` (`False`) |

## Signed distances (mm)

| Pose | Body | Rep | Signed dist | State |
|---|---|---|---:|---|
| pose_a_prestep | link5 | cooked | `-6.236686` | overlap |
| pose_a_prestep | link5 | raw | `4.272646` | clear |
| pose_a_prestep | gripper_link | cooked | `-15.386724` | overlap |
| pose_a_prestep | gripper_link | raw | `-5.956677` | overlap |
| pose_b_poststep0 | link5 | cooked | `3.043827` | clear |
| pose_b_poststep0 | link5 | raw | `7.355701` | clear |
| pose_b_poststep0 | gripper_link | cooked | `-5.273719` | overlap |
| pose_b_poststep0 | gripper_link | raw | `-1.721644` | overlap |

the raw tool mesh itself overlaps the cylinder at the frozen pose with the recorded gripper_link attribution; a target-family repair is the candidate.
