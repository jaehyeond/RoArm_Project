# Session 2026-07-14 - Cube10cm 0-999 Windows Archive And Local Raw Cleanup

## Scope

- Operational storage/integrity maintenance only.
- No Isaac/PhysX, render, dataset generation, training, PPO/RL, VLA, robot
  hardware, B200, or research-variable work was run.
- The session-progress experiment requirement is not applicable: this operation
  only archives and removes already-frozen raw storage and cannot change a
  scientific verdict.
- User authorization was exact:
  `Windows 최종 보관본 확인 완료. 로컬 raw PNG 195,000개만 삭제 승인.`

## Archive chain

1. Local raw source contained `195000` PNG files / `51386208295` payload bytes.
2. The manually copied intermediate exFAT set at
   `/media/cgxr/도관목/RoARM/raw_env_render_frames/` matched all relative names,
   sizes, and the all-file SHA-256-list aggregate:
   `dfbf466ea9f69906f574ebf93bc051cdd3268d60b6b025aea8cb141eba16a586`.
3. The user moved the archive to another Windows PC and explicitly confirmed
   that final copy complete. The Windows destination was not mounted locally,
   so this is recorded as user-confirmed rather than independently machine-hashed.

## Pre-delete gate

- Raw members: `195000`; other entries in the raw directory: `0`.
- Raw payload: `51386208295` bytes; allocated blocks: `51792117760` bytes.
- Non-raw/control retained baseline: `267` files / `1627858015` bytes.
- Active raw-consuming process count: `0`.
- Pre-delete manifest:
  `claudedocs/dataset_archives/cube10cm_top_view_0_999_v0_1/raw_predelete_manifest_20260714.tsv`.
- Manifest: `195000` entries / `51386208295` registered bytes / `21255018`
  file bytes / SHA-256
  `462418736b7dfe3542138441edec710dbc472da60a61fe065b4b91ff58427750`.
- Reconstructed SHA-256-list aggregate matched the earlier external verification
  exactly: `dfbf466e...16a586`.
- Disk immediately before deletion: `29072855040` bytes free, `96%` used.

## Deletion

- Deleted only paths registered in the pre-delete manifest.
- Used NUL-delimited manifest paths with non-recursive `rm --`.
- `rm -rf` and directory deletion were not used.
- Command exit code: `0`.

## Post-delete verification

- Canonical `raw_env_render_frames/` directory: retained and empty.
- Raw entries / PNGs: `0 / 0`.
- Non-raw/control: unchanged at `267` files / `1627858015` bytes.
- Preserved key sizes:
  - compact D247 LeRobot: `565313872` bytes;
  - D247 companion metadata: `34868154` bytes;
  - D256 transition/prior: `416560229` bytes;
  - D257 teacher: `168801` bytes;
  - `frames.jsonl`: `521683054` bytes.
- D249 registered compact evidence: all `24` files and `1089314018` bytes passed
  SHA-256 and size verification.
- Git-tracked deleted files: `0`; `git diff --check`: PASS.
- Disk after deletion: `80836767744` bytes free, `87%` used.
- Immediate free-space gain: `51763912704` bytes (`51.764GB` decimal).

## Operational consequence

- Compact D247, labels/metadata, D256/D257, and script-control lineage remain
  local and usable.
- A raw archive restore is required only for lossless D247 rebuilding,
  source-vs-decoded pixel comparison, D248 contact-sheet regeneration, or exact
  raw-frame inspection.
- Restore only to the canonical empty raw directory and validate against the
  pre-delete manifest before use.
- Never rerender into the frozen D242 root. Future paired data must use a new
  forward-only root and transient chunk render -> LeRobot validation -> raw
  cleanup.

## State-file decision

- No DECISIONS entry: no durable scientific lesson or failure rule changed.
- No EXPERIMENT_LEDGER row: this was not a research experiment.
- START_HERE receives only an operational storage pointer; the active grasp case
  and its verdict are unchanged.
