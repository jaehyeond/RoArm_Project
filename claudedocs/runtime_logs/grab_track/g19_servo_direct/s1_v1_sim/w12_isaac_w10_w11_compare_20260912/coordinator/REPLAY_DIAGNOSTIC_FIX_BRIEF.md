# W12 acceptance follow-up — one unused diagnostic only

Reuse the current proven Claude Opus5 terminal/worktree. Your main replay return
was received and its media/numeric/RRD work is accepted. No new replay, physics,
training, package work, hardware/camera, state docs, commit/push, or other changes.
You are not alone: root owns main state/coordinator. Do not revert others' work.
You own only your existing replay output directory.

One late coordinator message was not reflected in the completion report:
msg_1369f5b3ef7c. Fix exactly this report-only diagnostic and finish promptly.

1. Read `sim_isaac_replay_w12.py:209`: DOOR_R_MAX_M subtracts TOOL_W (display
   world, already +ORIGIN) from NODES_D (DEME). Its reported7.7822mm/deg is wrong.
   Also hinge angular displacement uses radius perpendicular to the hinge axis,
   not the full Euclidean radius. Actual rendering/node mapping correctly adds
   ORIGIN elsewhere; do NOT modify or rerun frozen renderer or per-run gates.
2. Calculate correct lever from raw source nodes and source hinge/axis in the
   same coordinate frame. Record formula, actual source paths/hashes, original
   invalid fields and correction in NEW `diagnostic_correction_01.json` with a
   small pure read-only checker if needed. Show old expression differs and
   correct source-frame and display-frame calculations agree. No new thresholds.
3. Update only your REPORT to remove wrong7.7822 and derived0.18mm claims or
   clearly mark them invalid with corrected values/JSON reference. This applies
   to table3.2, section3.3 and final section4. Explicitly state original renderer
   metadata is preserved historical evidence and this unused scalar is superseded.
4. Preserve all existing artifacts. Do not repeat the expensive RRD checks/root
   scripts; root already rechecked those. Run the tiny correction checker and
   `verify_w12_replay.py <replay_root> --read-only` with PYTHONDONTWRITEBYTECODE=1.
5. Report exact fixed files + correction values + original renderer/gates hashes
   unchanged, and send worker_done with this NEW task/dispatch from your preamble.

If checking coordinator mail, consuming `check` can replay an unacknowledged
Delivery. Read/process every row, acknowledge the returned deliveryId, then get
the next batch; do not assume another plain check reads newly queued messages.
No additional work is requested beyond this bounded diagnostic erratum.
