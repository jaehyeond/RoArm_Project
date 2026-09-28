# RAW_SCHEMA_REQUIRED erratum 04 (DRAFT, pre-registration) — floors are support surfaces; transition boundary frame index

Drafted 2026-09-16 by the main (coordinator) session after the user decision of the same day
("1번 2번 추천대로"). This append-only erratum supersedes only the two phrases named below in
`RAW_SCHEMA_REQUIRED.md`; earlier files remain preserved. It becomes effective for a producer
revision only after the independent auditor registers it in the audit contract folder
(`w13-cycle-audit/.../resume_20260913/audit/RAW_SCHEMA_REQUIRED_ERRATUM_04.md`) and it never
retroactively changes the W13 run_01 verdict under the earlier text (that FAIL stands as history).

## 1. Source and bin floors are support surfaces (supersedes the floor part of :50-52 and :76-77)

- Rationale. The 2.5 mm margin band exists to avoid forcing a label where a particle could be on
  either side of an open boundary (tray side walls above the fill, tray top, bin rim, tool mouth).
  A floor is a solid wall: a particle cannot be below it except by numerical penetration. Under the
  literal "every sphere strictly inside by margin" rule the resting bottom layer (about 5,300 of
  20,000 owners in W13 run_01, 26 %) is permanently `ambiguous`, and a pellet resting on the bin
  floor can never become `receiving_bin`, which makes definite delivery counting impossible.
- New rule (v2). For the **source tray floor** (`source_bounds_m[2][0]`) and the **bin inner floor**
  (`floor_inner_z_m`), whole-sphere containment on the lower z side is
  `min_k(center_z_k - r_k) > floor - margin` (penetration allowance = the frozen 2.5 mm margin,
  reused so that no new number is introduced). Every other face keeps the strict inward margin:
  x/y walls and tray top `> lo + margin` / `< hi - margin`; bin side apothem `< apothem - margin`;
  bin rim `< rim - margin`; tool cavity radius/half-width unchanged.
- The `near_*` (ambiguous) bands are unchanged: any single sphere overlapping the margin-expanded
  region still yields `ambiguous`, priority chain `tool > near_tool > bin > near_bin > source >
  near_source > in_flight > spill` unchanged, `moving_threshold` (>=) and `spill_rest_z_m` unchanged.
- A particle whose lowest sphere bottom is below `floor - margin` is not inside; if any sphere still
  overlaps the expanded region it is `ambiguous`, otherwise the chain proceeds to `in_flight`/`spill`.
- Metadata must declare `classify_floor_rule = "support_surface_v2: min(center_z - r) > floor - margin"`
  and `classify_contract_version = "RAW_SCHEMA_REQUIRED + ERRATUM_04"`.

## 2. Phase transition particle frame index (supersedes the sentence at :42 "Every phase transition sync must have a particle frame")

- Observed producer behaviour (rev28/rev29): `enter()` forces a particle frame **before** the first
  dynamics call of the new phase, i.e. at the last completed sync of the previous phase, row `i-1`,
  which is the physical state at the boundary instant. `transition_sync_index[i]` is the first row
  carrying the new phase code. A frame exactly at `i` exists only when a decision snapshot happens
  to be recorded there (W13 run_01: 7336 only).
- New wording (v2). "For every `transition_sync_index` value `i`, a particle frame row must exist
  with `particle_frame_sync_index == i-1` (boundary state). A frame at `i` is permitted in addition
  (decision rows) but not required." Producer self-verification must implement exactly this check
  (`transitions_have_boundary_frames`), replacing `set(tr) <= set(pf)`.
- No change to saved-frame semantics, ERRATUM_01 (1:1 replay of every saved frame) or ERRATUM_03
  (row identity, nondecreasing sync index).

## 3. Non-claims

- This erratum does not alter physics, thresholds (5 m/s warning, 20 m/s stop, 3 N pinch), or any
  scientific acceptance line. It changes only how derived inventory labels and transition-frame
  checks are computed for revisions that declare it.
- Derived v2 labels for W13 run_01 (rev30) are a re-derivation from unchanged raw coordinates; the
  run_01 raw contract verdict under the pre-erratum text remains FAIL 2/16 as recorded.
