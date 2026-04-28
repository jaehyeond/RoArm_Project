"""Merge v6 (real pick) + stacking_v1 (sim stacking) into combined LeRobot v3 dataset.

Uses native lerobot.datasets.aggregate.aggregate_datasets() — mp4 stream-copy concat
(no re-encoding, no double lossy), parallel-variance stats aggregation,
name-based task_index remap.

Output: lerobot_dataset_v6_stacking_v1/
  - 100 episodes (v6 ep 0-49 → out 0-49, stacking ep 0-49 → out 50-99)
  - 11692 frames (v6 6942 + stacking 4750)
  - 2 tasks:
      0 = "Pick up the sponge\\n"                              (v6)
      1 = "Stack the pink sponge at A onto B via Temp buffer"  (stacking)
  - data: chunk-000/file-000.parquet (concat, ~6MB combined)
  - videos: chunk-000/file-000.mp4 (75MB v6 + 24MB stacking = 99MB AV1 stream copy
            < 200MB DEFAULT_VIDEO_FILE_SIZE_IN_MB so single file)

Run (after stacking_v1 build completes):
  conda run -n roarm python sim_scripts/merge_v6_stacking.py
"""
from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V6_ROOT = REPO / "lerobot_dataset_v6"
STACKING_ROOT = REPO / "lerobot_dataset_stacking_v1"
OUT_ROOT = REPO / "lerobot_dataset_v6_stacking_v1"
AGGR_REPO_ID = "roarm_m3_v6_stacking"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    if not V6_ROOT.exists():
        sys.exit(f"ERROR: missing {V6_ROOT}")
    if not STACKING_ROOT.exists():
        sys.exit(f"ERROR: missing {STACKING_ROOT}")

    if OUT_ROOT.exists():
        log(f"Removing existing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    from lerobot.datasets.aggregate import aggregate_datasets

    t0 = time.time()
    log("=== aggregate_datasets ===")
    log(f"  src[0] = {V6_ROOT}    (v6 real pick, 50 ep, 6942 frames)")
    log(f"  src[1] = {STACKING_ROOT}  (sim stacking, 50 ep, 4750 frames)")
    log(f"  dst    = {OUT_ROOT}  (repo_id={AGGR_REPO_ID})")

    aggregate_datasets(
        repo_ids=["local/v6_pick", "local/stacking_sim"],
        aggr_repo_id=AGGR_REPO_ID,
        roots=[V6_ROOT, STACKING_ROOT],
        aggr_root=OUT_ROOT,
    )
    log(f"=== aggregate_datasets DONE in {time.time()-t0:.1f}s ===\n")

    log("=== Validate merged dataset ===")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(AGGR_REPO_ID, root=str(OUT_ROOT))
    log(f"  total_episodes = {ds.meta.total_episodes}")
    log(f"  total_frames   = {ds.meta.total_frames}")
    log(f"  total_tasks    = {len(ds.meta.tasks)}")
    log(f"  fps={ds.meta.fps}, robot={ds.meta.robot_type}")
    log(f"  tasks:\n{ds.meta.tasks}")

    log("\n--- spot-check: ds[0] (first v6 frame) ---")
    item = ds[0]
    log(f"  episode_index={item['episode_index'].item()}  task_index={item['task_index'].item()}")
    log(f"  task='{item['task']}'")
    log(f"  state={[round(x,2) for x in item['observation.state'].tolist()]}")

    log("\n--- spot-check: ds[6941] (last v6 frame) ---")
    item = ds[6941]
    log(f"  episode_index={item['episode_index'].item()}  task_index={item['task_index'].item()}")
    log(f"  task='{item['task']}'")

    log("\n--- spot-check: ds[6942] (first stacking frame) ---")
    item = ds[6942]
    log(f"  episode_index={item['episode_index'].item()}  task_index={item['task_index'].item()}")
    log(f"  task='{item['task']}'")
    log(f"  state={[round(x,2) for x in item['observation.state'].tolist()]}")

    log("\n--- spot-check: ds[-1] (last frame) ---")
    last = ds[ds.meta.total_frames - 1]
    log(f"  idx={ds.meta.total_frames - 1}  episode_index={last['episode_index'].item()}  task_index={last['task_index'].item()}")

    expected = (100, 11692, 2)
    actual = (ds.meta.total_episodes, ds.meta.total_frames, len(ds.meta.tasks))
    assert actual == expected, f"FAIL: expected (eps,frames,tasks)={expected}, got {actual}"
    log("\n=== ALL ASSERTIONS PASS ===")


if __name__ == "__main__":
    main()
