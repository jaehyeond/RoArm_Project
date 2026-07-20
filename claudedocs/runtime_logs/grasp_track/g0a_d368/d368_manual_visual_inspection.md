# D368 manual original-resolution visual inspection

Inspection date: 2026-07-20 KST  
Overall: **FAIL — observability only; numerical measurement verdict preserved**

Both images were opened and inspected at their original decoded resolution:

- `claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_summary_1920x1080.png`
  (`1920x1080`)
- `claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_rerun.png`
  (`4800x2800`, 2x-DPR raster of the registered `2400x1400` window)

## Checklist

- PASS — both PNGs opened at original resolution.
- PASS — all four professor-summary geometry panels are nonblank.
- PASS — all four Rerun spatial views are nonblank.
- PASS — the cyan link5 seed patch and its green certified-carrier state are visible.
- PASS — the cyan moving-inner patch and its green/yellow certified-carrier state are visible.
- PASS — purple outer and yellow dual-carrier semantics agree with the summary legend.
- FAIL — text layout is not fully clean: the summary's upper-row axis labels approach/overlap the
  lower-row titles, and moving-jaw Rerun marker labels overlap one another.
- FAIL — the Rerun `allocation counts and geometry budget` panel displays `Unknown timeline` and no
  metric values. The upper-right notification also reports
  `message proxy server crashed: Operation not permitted (os error 1)`.
- PASS — the summary visibly states `Hull count optimality = NULL`, identifies callback topology as
  authority, and labels the case `OFFLINE, no physics`.

## Interpretation

The geometry itself is present and the cyan/green/yellow/purple allocation distinction is visible.
The Rerun archive also passed automated footer, entity, component, timeline-inventory, RBL, and
headless-render checks. Nevertheless, the rendered viewer selected a nonexistent timeline for the
metric panel, so the preregistered human-facing observability contract fails. This does not alter or
erase the already-written Float64 measurement evidence. No audit rerun or artifact overwrite is
performed.
