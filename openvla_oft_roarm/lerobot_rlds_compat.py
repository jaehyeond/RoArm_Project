"""LeRobot v3 → RLDS-compat Dataset adapter for openvla-oft finetune.

Yields per-sample dicts already transformed by RLDSBatchTransform.
Matches the schema RLDSBatchTransform expects:
    rlds_batch = {
        "dataset_name": str,
        "action": np.ndarray(NUM_ACTIONS_CHUNK, ACTION_DIM)  # BOUNDS_Q99 [-1, 1]
        "observation": {
            "image_primary": np.ndarray(1, H, W, 3) uint8,
            "proprio": np.ndarray(PROPRIO_DIM,)  # optional, BOUNDS_Q99 [-1, 1]
        },
        "task": {"language_instruction": bytes},
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class LeRobotV3RLDSCompatDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        batch_transform,
        resize_resolution: Tuple[int, int],
        num_actions_chunk: int = 8,
        use_proprio: bool = False,
        image_aug: bool = False,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.root = Path(root)
        self.batch_transform = batch_transform
        self.resize = tuple(resize_resolution)
        self.num_chunk = int(num_actions_chunk)
        self.use_proprio = bool(use_proprio)
        self.image_aug = bool(image_aug)

        self.lerobot_ds = LeRobotDataset(repo_id=repo_id, root=str(self.root))

        # Load stats.json for BOUNDS_Q99 normalization
        stats_path = self.root / "meta" / "stats.json"
        with stats_path.open() as f:
            stats = json.load(f)
        self._action_q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)
        self._action_q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)
        self._state_q01 = np.asarray(stats["observation.state"]["q01"], dtype=np.float32)
        self._state_q99 = np.asarray(stats["observation.state"]["q99"], dtype=np.float32)
        self._action_dim = int(self._action_q01.shape[0])
        self._proprio_dim = int(self._state_q01.shape[0])

        # Episode boundaries via meta.episodes (LeRobot 0.4.x API)
        ep_meta = self.lerobot_ds.meta.episodes
        self._ep_from = np.asarray(ep_meta["dataset_from_index"], dtype=np.int64)
        self._ep_to = np.asarray(ep_meta["dataset_to_index"], dtype=np.int64)
        self._num_episodes = int(self._ep_from.shape[0])

        # Every frame is a valid starting frame (action chunk is padded by repeating last action)
        self._global_indices = np.arange(len(self.lerobot_ds), dtype=np.int64)

        # Map global frame_idx -> episode_idx (for fast chunk look-up)
        self._frame_to_ep_to = np.zeros(len(self.lerobot_ds), dtype=np.int64)
        for ep_idx in range(self._num_episodes):
            self._frame_to_ep_to[self._ep_from[ep_idx]:self._ep_to[ep_idx]] = self._ep_to[ep_idx]

        # Expose dataset_statistics in openvla-oft format
        self.dataset_statistics = {
            self.repo_id: {
                "action": {
                    "q01": self._action_q01,
                    "q99": self._action_q99,
                    "mean": np.asarray(stats["action"]["mean"], dtype=np.float32),
                    "std": np.asarray(stats["action"]["std"], dtype=np.float32),
                    "min": np.asarray(stats["action"]["min"], dtype=np.float32),
                    "max": np.asarray(stats["action"]["max"], dtype=np.float32),
                    "mask": np.ones(self._action_dim, dtype=bool),
                },
                "num_trajectories": self._num_episodes,
                "num_transitions": len(self.lerobot_ds),
            }
        }
        if self.use_proprio:
            self.dataset_statistics[self.repo_id]["proprio"] = {
                "q01": self._state_q01,
                "q99": self._state_q99,
                "mean": np.asarray(stats["observation.state"]["mean"], dtype=np.float32),
                "std": np.asarray(stats["observation.state"]["std"], dtype=np.float32),
                "min": np.asarray(stats["observation.state"]["min"], dtype=np.float32),
                "max": np.asarray(stats["observation.state"]["max"], dtype=np.float32),
            }

        print(
            f"[LeRobotV3RLDSCompatDataset] repo_id={repo_id} "
            f"len={len(self.lerobot_ds)} num_episodes={self._num_episodes} "
            f"action_dim={self._action_dim} proprio_dim={self._proprio_dim} "
            f"chunk={self.num_chunk} use_proprio={self.use_proprio} resize={self.resize}"
        )

    @staticmethod
    def _bounds_q99(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        out = 2 * (x - q01) / (q99 - q01 + 1e-8) - 1
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.lerobot_ds))

    def _load_chunk_actions(self, global_idx: int, ep_to: int) -> np.ndarray:
        out = np.zeros((self.num_chunk, self._action_dim), dtype=np.float32)
        last = None
        for k in range(self.num_chunk):
            actual_idx = global_idx + k
            if actual_idx < ep_to:
                a = self.lerobot_ds[int(actual_idx)]["action"]
                last = a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
            # else: pad by repeating last action
            out[k] = last
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = int(self._global_indices[idx])
        ep_to = int(self._frame_to_ep_to[global_idx])

        sample0 = self.lerobot_ds[global_idx]

        # Image (3, 720, 1280) float [0,1] -> (H, W, 3) uint8 -> resize
        img_t = sample0["observation.images.top"]
        if torch.is_tensor(img_t):
            img_np = (img_t.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        else:
            img_np = np.asarray(img_t)
        img_pil = Image.fromarray(img_np).resize(self.resize, Image.BILINEAR)
        image_uint8 = np.array(img_pil, dtype=np.uint8)  # (H, W, 3)

        # Action chunk + normalization
        actions_raw = self._load_chunk_actions(global_idx, ep_to)
        actions_norm = self._bounds_q99(actions_raw, self._action_q01, self._action_q99)

        # Task (single task across the dataset, but read per-sample to honor any task variants)
        task_str = str(sample0.get("task", "Pick up the sponge")).rstrip("\n")

        rlds_batch: Dict[str, Any] = {
            "dataset_name": self.repo_id,
            "action": actions_norm,  # (num_chunk, action_dim)
            "observation": {
                "image_primary": image_uint8[None, ...],  # (1, H, W, 3) uint8
            },
            "task": {"language_instruction": task_str.encode("utf-8")},
        }
        if self.use_proprio:
            state_raw = sample0["observation.state"]
            state_np = state_raw.cpu().numpy() if torch.is_tensor(state_raw) else np.asarray(state_raw)
            proprio_norm = self._bounds_q99(state_np.astype(np.float32), self._state_q01, self._state_q99)
            rlds_batch["observation"]["proprio"] = proprio_norm

        return self.batch_transform(rlds_batch)
