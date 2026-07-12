# D337 design-time offline scoping record (NOT decision evidence)

2026-07-13. Offline URDF-STL + hppfcl scoping that motivated the D337 case.
Script: `d337_design_scoping_script.py` (run in the isaaclab conda env).

## Pipeline validation vs runtime audits
- Stage gripper collision mesh identity: raw `gripper_link.stl` soup =
  41,094 vtx / 13,698 faces (matches D334 stage extraction exactly).
- Old target (7,11)mm, q5=0, full mesh: offline exact-EPA `-6.460447mm` vs
  D336 runtime exact layer `-6.460556mm` -> 0.0001mm-level agreement.
- link5 offline `+4.274mm` vs runtime `+4.2726mm`.

## q5 convention finding
- URDF gripper limits `[0, 1.571]`; D322 contract: real max opening 88.3deg
  <-> URDF 1.571rad => q5=0 CLOSED, 1.571 OPEN. The D325-family "open
  gripper q5=0" was a convention error.
- q5 sweep at old target (7,11): 0.0 -> -6.460mm overlap; 0.8 -> -1.237mm;
  1.2 -> +5.362mm clear; 1.5413 -> +11.175mm clear; 1.571 -> +11.634mm clear.
- q5=1.571 across anchors (7,11)/(14.6,13.9)/(15.25,9)/(11,11.5)/(0,11)/(3,12):
  gripper +11.44..+12.50mm clear everywhere; alignment gates pass at all
  anchors except (15.25,9.0) (fixed-jaw gap -0.033mm at the t=9 boundary).
- Table clearance at q5 open: gripper min-z ~ +0.073m vs table top -0.0121m
  (~+85mm margin; the jaw swings upward).

## Control-4 anchor
- Expected open-jaw gripper exact clearance at (7,11), q5=1.5413: +11.175mm
  (tolerance +/-0.5mm in the harness control).
- g2a note: URDF collision entry `gripper_link_collision_g2a.stl` is a 36-vtx
  ~4mm box (authored 2026-05-14, one day after the robot USD build) and is
  NOT what the sim collides with; the stage uses the full mesh.
