# D334 collision-table slide

## Purpose

This folder replaces the screenshot-only D334 table with a paper-style slide.
The original editable PPT/HTML/SVG source was not present in the repository or
the user document folders; only `/home/cgxr/Downloads/image.png` was available.

The slide is explicitly scoped to the historical D334 condition:

- jaw convention: `q5=0` = closed (corrected by D337),
- before the later `64+64` collider decomposition,
- pose A = exact-state write before physics,
- pose B = one PhysX step later (`dt=0.005 s`).

The final slide uses beginner-facing labels while keeping the technical terms
as secondary text:

- `세밀한 충돌 표면`: the stored triangle-based source collision mesh,
- `한 덩어리 볼록 껍질`: the single-convex physics proxy,
- state 1: positions written, before collision response,
- state 2: one 5 ms collision calculation later,
- `+`: a gap exists, `0`: touching, `-`: overlap.

## Source values

Canonical source:
`claudedocs/runtime_logs/grasp_track/g0a_d334/d334_signed_distance_matrix.json`.

| State | Body | Raw BVH output (mm) | Mirror-cooked convex output (mm) |
|---|---|---:|---:|
| A, pre-step | `link5` | +4.2726455 | -6.2366860 |
| A, pre-step | `gripper_link` | -5.9566769 | -15.3867239 |
| B, post-step-0 | `link5` | +7.3557009 | +3.0438270 |
| B, post-step-0 | `gripper_link` | -1.7216436 | -5.2737192 |

The slide converts valid displayed distances to cm and rounds to two decimal
places. This preserves the D334 classification resolution because
`0.01 cm = 0.1 mm`, the registered threshold.

## Scientific correction to the old table

For a colliding raw triangle-mesh BVH query, the D334 code states that the BVH
distance result is not a signed penetration depth. Raw overlap was gated by
triangle-level collision plus an EPA contact with depth at least `0.1 mm`.
Therefore, the revised slide reports the two raw `gripper_link` cells as
`중첩*` instead of presenting the old negative BVH scalars as penetration
depths. Positive raw separation values remain valid and are shown numerically.

The mirror-cooked `link5` hull passed live-volume parity (0.0498% difference),
whereas the `gripper_link` hull did not (1.46% > 0.5%). The latter is marked
with `†` and is not used as decision authority.

## Render

The source is `d334_collision_table_academic.html` at a fixed 1920x1080 canvas.
The PNG is rendered headlessly and should be inspected at original resolution
before use.

Final render audit (2026-07-15):

- output: `d334_collision_table_academic.png`, 1920x1080 RGB,
- SHA-256: `ddc9db2795f4d66b2564adf156829e6a143a599ceb72f6bb9fa28ab25e68a183`,
- visual inspection: PASS — no clipping, overlap, missing glyphs, or illegible
  table values at the full slide view,
- numeric audit: PASS — every displayed value was re-read from the canonical
  JSON and independently converted from mm to cm,
- source hygiene: `git diff --check` PASS.
