# Attempt3 collider decomposition — presentation inspection

- Purpose: lab-meeting presentation sidecar only; no scientific verdict change.
- Source geometry: preserved D347 PhysX callback faces, interpreted with the
  D348 corrected polygon topology.
- Source pose: D349 exact OPEN zero-step body/object pose.
- Isaac launched: no.
- USD writes / cook requests / physics steps: `0 / 0 / 0`.
- Rerun SDK/CLI: `0.34.1`.

## Presentation PNG selected

- Path: `attempt3_collider_decomposition_rerun_clean.png`
- Raster: `2400 x 1350`, RGB, exact 16:9.
- SHA-256: `ed7c056dbc29941d894960b3bf6bb1e4ef756c70cd597d24de8e10343096c2be`.
- Capture method: the verified RRD was kept in the headless viewer for eight
  seconds, then `rerun.experimental.ViewerClient.save_screenshot` captured the
  settled layout after loading notices disappeared.

Original-resolution visual inspection:

1. The upper-right physical-pose panel is nonempty and shows the D349 OPEN
   target with cool-colored `link5` parts, warm-colored `gripper_link` parts,
   gold fixed-point replacements, and the orange D34 x H90 cylinder.
2. The lower-left `link5` exploded panel is nonempty and makes the independent
   convex pieces visually separable.
3. The lower-right `gripper_link` exploded panel is nonempty and makes the
   independent convex pieces visually separable.
4. The upper-left guide is legible and states `64 + 64 = 128`, one disabled
   legacy collider per body, 13 fixed-point replacements, 115 preserved parts,
   the two D349 clearances, and the zero-step/narrowphase limitation.
5. No Rerun loading notice covers the robot/cylinder geometry or the reading
   guide.

Verdict: suitable for the main lab-meeting collider-structure slide.

An optional `3840 x 2160` clean copy is preserved as
`attempt3_collider_decomposition_rerun_clean_4k.png`, SHA-256
`7faea6d276722c6c54e5be9d4fed5ad3d03333ca3773efe02225c9b3b48c9d3c`.
The `2400 x 1350` copy is preferred for the full-slide layout because its guide
text remains proportionally larger; use the 4K copy when cropping a geometry
panel.

## Preserved first headless capture

`attempt3_collider_decomposition_rerun.png` is preserved but must not be used
in the presentation.  Its RRD/RBL/entity/component/footer contract passed, but
the screenshot was taken during initial viewer loading and most panels were
visually blank.  SHA-256:
`d1beb586352a917ea530d49a28a1b2355ee60a41af24b7197bcec488303b7179`.

`attempt3_collider_decomposition_rerun_presentation.png` is a complete immediate
rerender, but still contains loading notices.  It is preserved as an
intermediate display attempt; prefer the notice-free clean PNG above.

The selected presentation PNG is a second headless render of the same immutable
RRD and embedded blueprint; it does not recompute geometry or advance physics.
The RRD and RBL footer verification both passed independently.

## Scope boundary

The exploded panels move part positions for readability and therefore have no
physical-pose authority.  The upper-right panel preserves the D349 relative
pose.  Neither view is direct PhysX narrowphase, contact, solver, or settle
evidence.  No new experiment was run because this task only derives a
presentation view from already completed evidence and cannot change G0a.
