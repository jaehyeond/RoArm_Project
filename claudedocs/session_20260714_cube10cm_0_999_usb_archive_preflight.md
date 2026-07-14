# 2026-07-14 Cube10cm 0-999 USB archive preflight

Status: `PREFLIGHT_COMPLETE_COPY_NOT_STARTED`

This was a user-authorized storage/dependency audit. It did not change the
active cylinder grasp G0a case, consume D345, write to USB, move or delete local
data, run Isaac, run training, or control hardware.

## Why no research experiment ran

This session answers a storage-integrity question: which old files are active
dependencies, which are archival source data, and what must be verified before
externalization. Training, physics, or perturbation evaluation could not change
that dependency decision and would violate the rule against validation that
cannot change a decision. The useful failure-capable gates belong to the later
copy operation: manifest generation, tar completion, and source/USB/final hash
equality. None was claimed as executed here.

## Verified chronology

- The 0-999 corpus is genuinely pre-RL. D246/D247 produced `1000` episodes and
  `195000` frames before the first Isaac Lab PPO runtime in D258.
- D249 freezes the compact LeRobot dataset as the primary artifact and calls
  raw PNGs large source/debug frames.
- D319 and D321 use the old corpus as a script-only comparison/control. D321
  also consumes the D256 reset CSV derived from the tree, but creates a separate
  replay-rendered LeRobot corpus.
- The 2026-07-08 lab-meeting pivot freezes the tap track at D321 and explicitly
  retains D256 reset and script 0-999 control as inheritable assets.
- D322-D344 grasp code contains no D242/D247/D256 path dependency.

## Canonical path and exact inventory

Canonical root:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/
```

Measured current payload:

- whole tree: `53025215622` bytes by `du -sb`;
- file payload: `53014066310` bytes, `195267` files;
- raw source: `raw_env_render_frames/`, `195000` contiguous PNG files,
  `51386208295` bytes;
- non-raw/control: `267` files, `1627858015` bytes;
- compact D247: `8` files, `565313872` bytes;
- companion metadata: `4` files, `34868154` bytes;
- symlinks/hardlinks: `0/0`.

Only `11` files in the tree are Git-tracked. The raw frames, compact dataset,
`frames.jsonl`, companion metadata, and D249 freeze directory are ignored, so
the 2026-07-14 Git push is not a dataset backup.

## Dependency verdict

1. Do not move or delete the parent D242 root. Exact paths are embedded in
   twenty-two scripts and many frozen runtime provenance records.
2. Keep compact D247, `frames.jsonl`, manifests, labels, companion metadata,
   D256, and D257 locally. They are small relative to the raw source and retain
   the frozen script-control/reproducibility chain.
3. Copy the complete non-raw tree externally once so the archive is
   self-describing.
4. Copy the raw PNG source in five 200-episode batches.
5. Only the raw PNG payload is a future local cleanup candidate, and only after
   the core plus all five raw batches pass at the final destination and the user
   separately authorizes local deletion.
6. D321 and the earlier D241 100-episode corpus are separate assets and are not
   part of this operation.

## Integrity gap

D249's existing SHA-256 manifest covers only `24` files / `1089314018` bytes.
It intentionally excludes raw PNGs and also misses companion parquet and other
core files. It cannot certify a USB archive. Each later trip must generate a
new ordered per-member path/size/SHA-256 manifest and verify the tar hash on the
USB and again at the final destination.

## Selected transfer plan

Five 200-episode raw batches were selected over four 250-episode batches. Four
fit arithmetically, but five preserve about 4GB headroom per trip under the
currently observed `15333294080` free bytes. The first trip can also carry the
`1627858015`-byte non-raw core.

Exact episode/frame ranges and byte totals are recorded in:

- `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/raw_batches_20260714.tsv`;
- `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/archive_inventory_20260714.json`;
- `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/ARCHIVE_PLAN.md`.

## Current USB gate

The observed `/dev/sda1` is exFAT UUID `B0C1-F936`, mounted at
`/media/cgxr/B0C1-F936`. It has `15333294080` bytes free and contains the
unrelated `RELOC.zip` (`16114254207` bytes), which must be preserved. It is
currently mounted read-only with `errors=remount-ro`, so no write is eligible.

Before each later trip, ask the user to connect/reconnect the USB and authorize
exactly that trip. Recheck UUID, read-write status, free space, and existing
contents before writing. A read-only remount or filesystem repair requires its
own approval. After the user copies a trip to final storage and confirms the
same SHA-256, request separate approval before clearing the shuttle.

No local deletion is authorized by this session.

## Final cross-validation

- Recounted source: raw `195000 / 51386208295` bytes, non-raw
  `267 / 1627858015`, total files `195267 / 53014066310` payload bytes.
- The five TSV rows independently sum to `195000` files and
  `51386208295` bytes; every row matches a fresh filename-derived recount.
- `archive_inventory_20260714.json` parses with `jq` and its counts/bytes match
  the live tree and TSV.
- Current G0a/grasp scripts and D322-D344 grasp sessions return no D242/D247/D256
  path match; the legacy/tap script audit returns exactly `22` direct consumers.
- `git diff --check`: PASS.
- USB files, source dataset files, DECISIONS, and EXPERIMENT_LEDGER were not
  modified.
