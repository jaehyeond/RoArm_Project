"""Convert stacking sim demos → LeRobot v3 dataset.

V3 (5/03 evening pivot): N=4 # tower edge-stand stacking (47mm tall sponges).
  Inputs:  sim_demos_v3/demo_{seed:04d}_trajectory.csv  (50 × 146 × 6 deg)
           sim_renders_v5/episode_{seed:03d}/frame_{f:04d}.png  (50 × 146 PNGs)
  Output:  lerobot_dataset_stacking_v3/  (LeRobot v3, fps=30, 7300 frames, 50 eps, 1 task)

Task instruction (single, all eps):
  "Stack four pink sponges into a # pattern"

State/action convention:
  observation.state[t] = action[t] = trajectory[t]
  No L-F gap (procedural sim demo, not teleop). SmolVLA learns chunk prediction
  from action[t : t+horizon] context regardless.

Run (after render_stacking_demos_v3.py --all completes):
  conda run -n roarm python sim_scripts/sim_to_lerobot_stacking.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEMOS_DIR = REPO / "sim_demos_v3"
RENDERS_DIR = REPO / "sim_renders_v5"
OUT_DIR = REPO / "lerobot_dataset_stacking_v3"
REPO_ID = "roarm_m3_stacking_sim_v3"
TASK_INSTRUCTION = "Stack four pink sponges into a # pattern"

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
NUM_EPISODES = 50
FPS = 30
WIDTH, HEIGHT = 1280, 720


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demos-dir", type=str, default=str(DEMOS_DIR))
    p.add_argument("--renders-dir", type=str, default=str(RENDERS_DIR))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--overwrite", action="store_true",
                   help="Delete existing out-dir before building")
    p.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    return p.parse_args()


def validate_inputs(demos_dir, renders_dir, num_eps):
    for ep in range(num_eps):
        traj_path = Path(demos_dir) / f"demo_{ep:04d}_trajectory.csv"
        if not traj_path.exists():
            sys.exit(f"ERROR: missing {traj_path}")
        ep_dir = Path(renders_dir) / f"episode_{ep:03d}"
        if not ep_dir.exists():
            sys.exit(f"ERROR: missing render dir {ep_dir}")
        traj = np.loadtxt(traj_path, delimiter=",", skiprows=1)
        T = len(traj)
        n_pngs = len(list(ep_dir.glob("frame_*.png")))
        if n_pngs != T:
            sys.exit(f"ERROR: ep{ep} traj={T} frames but {n_pngs} PNGs in {ep_dir}")
    log(f"Validation PASS: {num_eps} eps, all CSV/PNG counts match")


def build_dataset(demos_dir, renders_dir, out_dir, num_eps, overwrite):
    out = Path(out_dir)
    if out.exists():
        if overwrite:
            log(f"Removing existing {out}")
            shutil.rmtree(out)
        else:
            sys.exit(f"ERROR: {out} exists. Use --overwrite to rebuild.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": (HEIGHT, WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": JOINT_NAMES},
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": {"motors": JOINT_NAMES},
        },
    }

    log(f"Creating LeRobotDataset @ {out}")
    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=str(out),
        features=features,
        robot_type="roarm_m3",
        use_videos=True,
    )

    overall_t0 = time.time()
    for ep in range(num_eps):
        ep_t0 = time.time()
        traj_path = Path(demos_dir) / f"demo_{ep:04d}_trajectory.csv"
        ep_dir = Path(renders_dir) / f"episode_{ep:03d}"
        traj = np.loadtxt(traj_path, delimiter=",", skiprows=1).astype(np.float32)
        T = len(traj)

        for f in range(T):
            img_path = ep_dir / f"frame_{f:04d}.png"
            img = np.array(Image.open(img_path))[:, :, :3]
            ds.add_frame(
                {
                    "observation.images.top": img,
                    "observation.state": traj[f],
                    "action": traj[f],  # action == state (procedural, no L-F gap)
                    "task": TASK_INSTRUCTION,
                }
            )
        ds.save_episode()
        log(f"  ep {ep:02d}: T={T}  {time.time()-ep_t0:.1f}s "
            f"(total {time.time()-overall_t0:.1f}s)")

    log(f"\n=== BUILT in {time.time()-overall_t0:.1f}s ===")
    log(f"  total_episodes={ds.meta.total_episodes}  total_frames={ds.meta.total_frames}")
    return ds


def validate_dataset(out_dir):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(REPO_ID, root=str(out_dir))
    log(f"Loaded LeRobotDataset: n_eps={ds.meta.total_episodes}, n_frames={ds.meta.total_frames}")
    item = ds[0]
    img = item["observation.images.top"]
    state = item["observation.state"]
    action = item["action"]
    log(f"  observation.images.top: shape={tuple(img.shape)} dtype={img.dtype} "
        f"range=[{img.min():.4f}, {img.max():.4f}]")
    log(f"  observation.state: shape={tuple(state.shape)} dtype={state.dtype} "
        f"first={state.tolist()}")
    log(f"  action: shape={tuple(action.shape)} dtype={action.dtype}")
    return ds


def main():
    args = parse_args()
    log(f"Demos: {args.demos_dir}")
    log(f"Renders: {args.renders_dir}")
    log(f"Output: {args.out_dir}")

    log("=== 1. Validate inputs ===")
    validate_inputs(args.demos_dir, args.renders_dir, args.num_episodes)

    log("=== 2. Build dataset ===")
    build_dataset(args.demos_dir, args.renders_dir, args.out_dir,
                  args.num_episodes, args.overwrite)

    log("=== 3. Validate dataset ===")
    validate_dataset(args.out_dir)
    log("DONE.")


if __name__ == "__main__":
    main()
