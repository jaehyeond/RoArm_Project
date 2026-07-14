# Cube10cm Top-View 0-999 v0.1 USB Archive Plan

Status: `LOCAL_RAW_CLEANUP_COMPLETE_WINDOWS_RAW_ARCHIVE_USER_CONFIRMED`

Date: 2026-07-14 KST

This is an operational storage sidecar. It does not change the active cylinder
grasp G0a case or authorize any research-case advancement. Sections 4-8 retain
the original preflight plan. The actual one-shot raw-only transfer and local
cleanup are recorded in Section 10 and
`raw_local_cleanup_receipt_20260714.json`.

## 1. Canonical identity

Canonical source root:

```text
claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_0_999_d242/
```

Raw source frames:

```text
.../cube10cm_top_view_visual_0_999_d242/raw_env_render_frames/
```

Primary frozen LeRobot dataset:

```text
.../cube10cm_top_view_visual_0_999_d242/lerobot_dataset_av1_d247/
```

Freeze id: `cube10cm_top_view_0_999_v0_1_d249`.

The corpus contains `1000` episodes and `195000` frames at `1280x720`. Each
episode has `195` frames. The raw filenames are contiguous from
`rgb_000000.png` through `rgb_194999.png`.

## 2. Timeline and scientific role

- D246 generated and validated the 0-999 render before any Isaac Lab PPO
  runtime. D247 converted it to AV1 LeRobot video+parquet, also without PPO,
  VLA/SmolVLA training, or an action teacher.
- D249 designates the compact LeRobot dataset and split manifests as the
  primary frozen artifact. It explicitly classifies raw PNGs as large
  source/debug frames and does not hash them individually.
- D319 and D321 later used the old 0-999 labels as the `script-only`
  diversity/control baseline. D321 also used the D256 reset CSV derived from
  this tree, but replay-rendered a separate D321 dataset instead of reusing the
  old raw PNGs or D247 videos.
- The 2026-07-08 grasp pivot freezes the tap track at D321 while retaining
  D256 reset and the script 0-999 control as inheritable assets.
- Current D322-D344 grasp code does not read this D242/D247 tree.

Therefore the raw PNGs are safe to place offline after a verified external
archive exists, but the corpus is not disposable. The compact/control lineage
must remain available.

## 3. Dependency classification

| Asset | Current role | Local policy | External policy |
|---|---|---|---|
| `raw_env_render_frames/` | Lossless source, LeRobot rebuild and source-vs-decoded pixel audit | Copy first; possible later removal only after final archive PASS and separate approval | Required, five raw batches |
| `lerobot_dataset_av1_d247/` | Primary frozen dataset and future script-only control | Keep local | Required in core backup |
| `frames.jsonl`, manifest, D246 labels | Frame lineage and D319/D321 comparison | Keep local | Required in core backup |
| companion metadata, D248-D255 | Dataset recovery, splits, validation, loader preflight | Keep local | Required in core backup |
| D256 transition/prior and D257 teacher | Dormant but direct frozen-tap dependencies | Keep local | Required in core backup |
| D321 dataset | Separate post-meeting script-v2 asset | Unchanged | Not part of this archive |
| D241 100-episode corpus | Independent earlier corpus, no inode/link identity | Unchanged | Not part of this archive |

Do not move or rename the D242 parent root. Multiple scripts and historical
artifacts contain its exact path. Only the raw frame payload is a candidate for
later local space reclamation.

## 4. Exact preflight inventory

- Whole source tree apparent `du -sb`: `53025215622` bytes.
- File payload: `53014066310` bytes in `195267` files.
- Raw PNG payload: `51386208295` bytes in `195000` files.
- Non-raw/control payload: `1627858015` bytes in `267` files.
- Compact LeRobot dataset: `565313872` bytes in `8` files.
- Companion metadata: `34868154` bytes in `4` files.
- Symlinks: `0`.
- Hardlinked files (`nlink > 1`): `0`.
- Empty episode directories: `episodes/episode_000` through
  `episodes/episode_999`.

Existing D249 integrity coverage is insufficient for an external archive. Its
SHA-256 manifest covers `24` files and `1089314018` bytes, but omits all raw
PNGs and also omits companion parquet data and several core split/label files.
New transfer manifests are mandatory.

Machine-readable snapshot:
`archive_inventory_20260714.json`.

## 5. Selected five-trip raw split

The current USB had `15333294080` bytes free. Four 250-episode archives would
fit arithmetically, but five 200-episode batches leave roughly 4GB of headroom
per trip and are safer against filesystem allocation, tar overhead, and free
space drift.

| Trip | Episodes | Global frames / filenames | Files | Payload bytes |
|---|---:|---|---:|---:|
| 1 | 000-199 | `000000-038999` | 39000 | 10454762849 |
| 2 | 200-399 | `039000-077999` | 39000 | 10164485830 |
| 3 | 400-599 | `078000-116999` | 39000 | 10090886778 |
| 4 | 600-799 | `117000-155999` | 39000 | 10140133100 |
| 5 | 800-999 | `156000-194999` | 39000 | 10535939738 |

The complete non-raw/control core (`1627858015` bytes) can accompany Trip 1;
the combined file payload is `12082620864` bytes before tar headers, leaving a
substantial margin under the observed free space.

Exact table: `raw_batches_20260714.tsv`.

## 6. Archive layout and verification contract

Preferred shuttle artifacts are uncompressed, deterministic tar files because
the PNG inputs are already compressed and tens of thousands of loose files are
slow and allocation-heavy on exFAT.

```text
roarm_cube10cm_0_999_v0_1_core.tar
roarm_cube10cm_0_999_v0_1_raw_e000_199.tar
roarm_cube10cm_0_999_v0_1_raw_e200_399.tar
roarm_cube10cm_0_999_v0_1_raw_e400_599.tar
roarm_cube10cm_0_999_v0_1_raw_e600_799.tar
roarm_cube10cm_0_999_v0_1_raw_e800_999.tar
```

Every transfer must produce and retain:

1. an ordered per-member manifest containing relative path, byte count, and
   SHA-256;
2. the manifest's own SHA-256;
3. tar member count and summed source bytes;
4. tar SHA-256 after the USB write and `sync`;
5. the final-destination tar SHA-256 after the user copies it off the shuttle;
6. USB UUID, mount options, free space, source commit, timestamp, and PASS/FAIL
   in a trip receipt.

The tar must preserve paths relative to the D242 root. The core tar includes
all non-raw files and the empty `episodes/episode_000..999` directories. Each
raw tar includes only its registered contiguous filename range beneath
`raw_env_render_frames/`.

## 7. Preflight USB observation and stop condition (historical)

Observed USB on 2026-07-14:

- source: `/dev/sda1`;
- filesystem: exFAT, UUID `B0C1-F936`;
- mount: `/media/cgxr/B0C1-F936`;
- total/free: `31447875584 / 15333294080` bytes;
- existing unrelated file: `RELOC.zip`, `16114254207` bytes;
- mount options include `ro` and `errors=remount-ro`.

The USB is currently not write-eligible. Do not modify `RELOC.zip`, remount,
repair the filesystem, or begin a transfer without a separate user-approved
step. On a later trip, first ask the user to connect/reconnect the USB, then
recheck identity, read-write state, and free space. If it still mounts read-only,
stop and request approval before any unmount/filesystem check or repair.

## 8. Authorization gates

The approvals are intentionally separate:

1. **Trip approval:** permission to write one named core/raw batch to the
   verified USB. Approval for one trip does not authorize later trips.
2. **USB clearing approval:** after the user confirms the same tar hash at the
   final destination, permission to remove only that completed shuttle copy so
   the next trip can start.
3. **Local raw cleanup approval:** only after core plus all five raw archives
   pass at the final destination, permission to remove local raw PNGs. USB
   success alone is insufficient.

The actual workflow later differed from this five-trip proposal. The user
manually copied the complete raw set in one pass to a 465.8GB exFAT device,
Codex verified that intermediate copy byte-for-byte, and the user then confirmed
the final Windows-PC copy. The user separately authorized deletion of exactly
the local 195,000 raw PNG members. The compact/control core was not part of the
observed USB copy and remains local intact; this completion does not claim an
external core backup.

## 9. Future restore rule

For episode `e` and frame-in-episode `f`:

```text
global_frame = e * 195 + f
raw_batch = floor(e / 200) + 1
```

If a future task needs raw source pixels, pause that task and ask the user to
connect the archive containing the corresponding batch. Restore to the exact
canonical `raw_env_render_frames/` path and verify the member manifest before
running a rebuild or pixel-comparison audit.

## 10. Actual completion record

- Final raw archive: another Windows PC, explicitly confirmed complete by the
  user. It was not mounted locally for an independent final-destination hash.
- Verified intermediate path:
  `/media/cgxr/도관목/RoARM/raw_env_render_frames/`.
- Intermediate verification: `195000` files / `51386208295` bytes; relative
  names and sizes exact; all-file SHA-256-list aggregate exact at
  `dfbf466ea9f69906f574ebf93bc051cdd3268d60b6b025aea8cb141eba16a586`.
- Pre-delete manifest:
  `raw_predelete_manifest_20260714.tsv`, `195000` entries,
  `51386208295` registered bytes, SHA-256
  `462418736b7dfe3542138441edec710dbc472da60a61fe065b4b91ff58427750`.
- User deletion authorization was explicit. Deletion used only the manifest
  paths and no recursive directory removal.
- Post-delete: canonical raw directory retained and empty; non-raw/control
  `267` files / `1627858015` bytes retained; D249's `24` registered compact
  files / `1089314018` bytes re-hashed PASS; Git-tracked deletions `0`.
- Disk free changed from `29072855040` bytes immediately before deletion to
  `80836767744` bytes after deletion (`+51763912704` bytes); filesystem use
  changed from `96%` to `87%`.
- Receipt: `raw_local_cleanup_receipt_20260714.json`.
