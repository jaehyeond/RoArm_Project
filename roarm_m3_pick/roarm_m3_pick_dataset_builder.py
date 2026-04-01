"""RoArm M3 Pick dataset in RLDS format for OpenVLA-OFT fine-tuning.

Converts collected_data_v5/ (Azure Kinect RGB + 6-DOF joint angles) into
a standard TFDS/RLDS dataset that OpenVLA-OFT can load via tfds.builder().

Usage:
    cd roarm_m3_pick
    conda run -n openvla tfds build --overwrite

Data source: ../collected_data_v5/episode_XXXX/
  - rgb_NNNN.jpg: 1280x720 RGB (Azure Kinect)
  - metadata.json: episode metadata + per-frame joint angles

Action representation:
  - action[t] = state[t+1] (absolute joint positions, next-state-as-action)
  - action[-1] = state[-1] (stop action = repeat last state)
  - 6-dim: [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper] in degrees
  - Raw values stored (normalization handled by OpenVLA-OFT at training time)
"""

import json
from pathlib import Path
from typing import Any, Iterator, Tuple

import cv2
import numpy as np
import tensorflow_datasets as tfds

# Path to collected data (relative to this file's parent directory)
_DATA_DIR = Path(__file__).resolve().parent.parent / "collected_data_v5"


class RoarmM3Pick(tfds.core.GeneratorBasedBuilder):
    """RLDS dataset for RoArm M3 single-arm pick task."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release. 138 episodes from collected_data_v5.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(720, 1280, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Azure Kinect RGB 720p.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(6,),
                                        dtype=np.float32,
                                        doc="6 joint angles in degrees: "
                                        "[base, shoulder, elbow, wrist_pitch, wrist_roll, gripper].",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(6,),
                                dtype=np.float32,
                                doc="Absolute joint positions (next-state-as-action). "
                                "6-dim in degrees.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount factor, 1.0 for demonstrations.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="1.0 on final step (success), 0.0 otherwise.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on first step of episode.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of episode.",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step (terminal for demos).",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Natural language task instruction.",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to source episode directory.",
                            ),
                            "episode_id": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="Episode index.",
                            ),
                            "zone": tfds.features.Text(
                                doc="Spatial zone classification.",
                            ),
                            "num_frames": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="Number of frames in episode.",
                            ),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples(_DATA_DIR),
        }

    def _generate_examples(
        self, data_dir: Path
    ) -> Iterator[Tuple[str, Any]]:
        episodes = sorted(data_dir.glob("episode_*"))
        if not episodes:
            raise FileNotFoundError(f"No episodes found in {data_dir}")

        for ep_dir in episodes:
            meta_path = ep_dir / "metadata.json"
            if not meta_path.exists():
                print(f"  SKIP: {ep_dir.name} (no metadata.json)")
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            frames = meta["frames"]
            num_frames = len(frames)

            if num_frames < 2:
                print(f"  SKIP: {ep_dir.name} (too few frames: {num_frames})")
                continue

            # Build steps
            steps = []
            skip_episode = False

            for i, frame in enumerate(frames):
                # Load RGB image
                rgb_path = ep_dir / frame["rgb_path"]
                img = cv2.imread(str(rgb_path))
                if img is None:
                    print(f"  WARNING: Cannot read {rgb_path}, skipping episode")
                    skip_episode = True
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Joint state (6-dim, degrees)
                state = np.array(frame["angles"], dtype=np.float32)

                # Action = next state (absolute joint positions)
                if i < num_frames - 1:
                    action = np.array(frames[i + 1]["angles"], dtype=np.float32)
                else:
                    action = state.copy()  # stop action

                steps.append(
                    {
                        "observation": {
                            "image": img,
                            "state": state,
                        },
                        "action": action,
                        "discount": 1.0,
                        "reward": float(i == num_frames - 1),
                        "is_first": i == 0,
                        "is_last": i == num_frames - 1,
                        "is_terminal": i == num_frames - 1,
                        "language_instruction": "Pick up the sponge",
                    }
                )

            if skip_episode:
                continue

            episode_id = meta.get("episode_id", 0)
            zone = meta.get("zone", "UNKNOWN")

            yield ep_dir.name, {
                "steps": steps,
                "episode_metadata": {
                    "file_path": str(ep_dir),
                    "episode_id": int(episode_id),
                    "zone": zone,
                    "num_frames": num_frames,
                },
            }
