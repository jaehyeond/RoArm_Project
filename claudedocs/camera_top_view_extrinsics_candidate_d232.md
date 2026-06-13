# D232 Top-View Camera Extrinsics Candidate

Status: candidate definition for review and reprojection-smoke design. Do not
use for scale-up until reprojection, occlusion, codec, fps/render-time,
LeRobot-load, and disk gates pass.

## Coordinate Assumptions

- World/table frame follows the existing sim convention: `+Z` is up, workspace
  is in table `X/Y`.
- Existing P6v12 table precedent:
  - table center: `(x=0.25, y=0.0)`;
  - table size: `0.90m x 0.70m`;
  - table top: `z=-0.012117m`.
- Cube10cm tap env default cube placement range is broader in code, but recent
  D230 useful-tap region uses the relevant `x=[0.14,0.34]`,
  `y=[-0.10,0.10]` band. The camera should cover the table/workspace, not only
  that narrow band.

## Intrinsics-Derived Coverage

From `sim_scripts/kinect_calib.yaml` candidate intrinsics:

- `fx=608.33`, `fy=608.28`, `width=1280`, `height=720`.
- Horizontal FOV: about `92.907deg`.
- Vertical FOV: about `61.237deg`.

If the camera optical axis is vertical and the image is not cropped, approximate
table-plane coverage is:

| table-to-camera height | coverage width | coverage height |
|---:|---:|---:|
| 0.55m | 1.157m | 0.651m |
| 0.60m | 1.262m | 0.710m |
| 0.65m | 1.368m | 0.769m |
| 0.70m | 1.473m | 0.829m |
| 0.75m | 1.578m | 0.888m |
| 0.80m | 1.683m | 0.947m |

Initial height recommendation: `0.65m` above table top. This gives margin over
the existing `0.90m x 0.70m` table precedent while staying plausibly mountable.

Static no-render projection precheck:

- `python3 sim_scripts/cube10cm_camera_reprojection_contract.py`
- Output JSON:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_camera_reprojection_contract_d232.json`
- Result with candidate V1:
  - table corners all inside, projected `u=[217.2,1059.5]`,
    `v=[37.7,692.8]`, margins left/right/top/bottom
    `[217.2,220.5,37.7,27.2]px`;
  - D230 xy10 cube-top outer corners all inside, projected
    `u=[461.3,793.2]`, `v=[199.4,531.2]`;
  - env-default cube-top outer corners all inside, projected
    `u=[533.2,815.3]`, `v=[171.7,558.8]`.
- Interpretation: 0.65m is a plausible first smoke candidate, but the table
  bottom margin is only about `27px`, so the first rendered marker frame must
  verify orientation/framing before any scale-up.

## Candidate V1

Camera contract id: `cube10cm_top_view_v1_candidate`.

- Camera model: Azure Kinect DK.
- Mount: rigid overhead clamp/tripod, lens facing downward.
- Table-to-camera height: `0.65m` above table top.
- World camera center:
  - target table point: `(0.25, 0.0, -0.012117)`;
  - camera center: `(0.25, 0.0, 0.637883)`.
- Optical axis: vertical, pointing toward world `-Z`.
- Roll/pitch/yaw:
  - conceptual top-down: roll `0deg`, pitch `0deg`, yaw selected so image width
    is aligned with table/workspace `X`;
  - implementation must record exact USD/OpenCV basis, not rely on this prose.
- Image convention:
  - raw `1280x720`;
  - no crop in candidate V1;
  - deterministic flip/rotation must be decided by a rendered axis-marker frame;
  - preferred display convention: image right is world `+X`, image down is world
    `-Y` or explicitly documented alternative.

## Why Not Reuse Old Extrinsics

The old `kinect_calib.yaml` extrinsics are hand-eye calibration from the prior
camera setup. They can provide intrinsics/FOV, but they do not define this
overhead top-view physical mount. Candidate V1 therefore defines a new physical
camera center and optical axis.

## Smoke Must Decide

Before this candidate can become a fixed contract, smoke must report:

- rendered axis-marker orientation and any required image flip/rotation;
- marker/cube-corner reprojection median/max pixel error;
- cube visibility/occlusion rates over contact windows;
- whether 0.65m height leaves enough margin for all intended split bounds;
- whether robot/tool self-occlusion is acceptable from a pure top view.

## Open Risks

- Pure vertical top view may hide tool-cube contact depth cues. If contact
  interpretation is poor, a small pitch tilt may be required, but that must be a
  physical mount decision and not a simulator-only aesthetic change.
- Azure Kinect body/cable clearance may make exactly vertical mounting awkward.
  If the real mount is offset, update the camera center and yaw/roll/pitch before
  rendering scale-up.
- The final flip convention must be validated with an axis marker; guessing from
  code conventions is not enough.
