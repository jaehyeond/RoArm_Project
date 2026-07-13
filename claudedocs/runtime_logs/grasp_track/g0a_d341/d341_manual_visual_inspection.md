# D341 manual Rerun visual inspection

- Screenshot: `d341_d340_cook_observability_rerun_inspection.png`
- sha256: `dc3f1d82e05d324f2fd032a0caca5077a04cbd3dc145f936bf897e0c1c8450ee`
- Registered logical window: `2400x1400`; PNG raster: `4800x2800`
- Inspection method: opened the generated PNG at original detail with the
  local image inspection tool.

Observed:

- All eight independent panel titles are visible: source, live instance,
  prototype, and candidate for link5; the same four for gripper.
- Every spatial panel contains visible geometry. The three x1 variants are not
  overlaid in a single panel.
- The lower metric Dataframe contains `part_idx` and numeric scalar cells.
- The event table contains INFO and WARN rows, including the retained D340 FAIL
  warning and the D341 no-physics/no-asset-mutation stop message.
- The Viewer status/loading notification remains in the upper-right corner,
  but it does not hide a required panel title or the displayed candidate
  geometry.

Boundaries:

- Pixels do not prove bit-exact geometry equality. Original JSON/hash evidence
  remains authoritative.
- A single decision view does not expose every one of 52 parts or 67 events at
  once. The exact archive entity/timeline/component gate certifies full row
  presence; this inspection certifies that the registered groups and diagnostic
  panes are actually viewable.

Manual visual-inspection verdict: `PASS`. This does not change D340's
scientific verdict and does not set `g0a_pass=true`.
