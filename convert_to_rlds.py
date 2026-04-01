"""
collected_data_v5/ → RLDS (TFRecord) 변환 스크립트

OpenVLA-OFT fine-tuning을 위한 RLDS 데이터셋 생성.
conda activate openvla 환경에서 실행 (TF 전용, PyTorch/roarm env와 별개!)

사용법:
    conda run -n openvla python convert_to_rlds.py

입력: collected_data_v5/episode_XXXX/ (rgb_NNNN.jpg + metadata.json)
출력: openvla_dataset_v5/roarm_m3_pick/ (TFRecord shards)

=== 핵심 결정 사항 ===
1. 정규화: BOUNDS (하드웨어 스펙 기준, Q99 아님)
   - 절대 관절 각도이므로 Q99 클리핑하면 유효 위치에 도달 불가
   - ALOHA 선례: 절대 관절 → BOUNDS 사용
2. 액션 정의: action[t] = state[t+1] (next-state as action)
   - LeRobot v3 convert와 동일 로직
   - 마지막 프레임: action = current state (stop action)
3. 이미지: 원본 1280x720 RGB로 저장
   - 학습 시 OpenVLA가 224x224로 자동 리사이즈
4. language_instruction: "Pick up the sponge" (단일 태스크)
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# ─── Constants ───────────────────────────────────────────────────────────────

INPUT_DIR = "collected_data_v5"
OUTPUT_DIR = "openvla_dataset_v5"
DATASET_NAME = "roarm_m3_pick"
LANGUAGE_INSTRUCTION = "Pick up the sponge"

# RoArm M3 하드웨어 관절 범위 (CLAUDE.md 스펙)
JOINT_BOUNDS = np.array([
    [-190.0, 190.0],   # base
    [-110.0, 110.0],   # shoulder
    [-70.0,  190.0],   # elbow (비대칭!)
    [-110.0, 110.0],   # wrist_pitch
    [-190.0, 190.0],   # wrist_roll
    [-10.0,  100.0],   # gripper
], dtype=np.float32)

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]


# ─── Normalization ───────────────────────────────────────────────────────────

def normalize_to_bounds(angles: np.ndarray) -> np.ndarray:
    """관절 각도를 [-1, 1] 범위로 BOUNDS 정규화.

    BOUNDS 방식: (angle - low) / (high - low) * 2 - 1
    - ALOHA 선례: 절대 관절 각도 → BOUNDS (Q99 아님)
    - Q99는 극단 위치를 클리핑하므로 사용 금지
    """
    low = JOINT_BOUNDS[:, 0]
    high = JOINT_BOUNDS[:, 1]
    return 2.0 * (angles - low) / (high - low) - 1.0


def denormalize_from_bounds(normalized: np.ndarray) -> np.ndarray:
    """[-1, 1] → 원래 각도 (추론 시 역변환용)."""
    low = JOINT_BOUNDS[:, 0]
    high = JOINT_BOUNDS[:, 1]
    return (normalized + 1.0) / 2.0 * (high - low) + low


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_episode(episode_dir: Path) -> dict:
    """단일 에피소드 로드. metadata.json + rgb JPG 파일들."""
    meta_path = episode_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    frames = meta["frames"]
    num_frames = len(frames)

    images = []
    states = []
    actions = []

    for i, frame in enumerate(frames):
        # RGB 이미지 로드 (BGR → RGB)
        rgb_path = episode_dir / frame["rgb_path"]
        img = cv2.imread(str(rgb_path))
        if img is None:
            print(f"  WARNING: Cannot read {rgb_path}, skipping frame {i}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

        # 관절 상태 (degrees)
        state = np.array(frame["angles"], dtype=np.float32)
        states.append(state)

        # 액션 = next state (LeRobot v3 변환과 동일 로직)
        if i < num_frames - 1:
            next_state = np.array(frames[i + 1]["angles"], dtype=np.float32)
        else:
            next_state = state.copy()  # 마지막 프레임: stop action
        actions.append(next_state)

    return {
        "images": images,
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "episode_id": meta.get("episode_id", 0),
        "zone": meta.get("zone", "UNKNOWN"),
        "num_frames": len(images),
    }


# ─── TFRecord Writer ─────────────────────────────────────────────────────────

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecord_episode(writer, episode_data: dict, normalize: bool = False):
    """단일 에피소드를 TFRecord에 쓴다.

    RLDS 포맷: 각 에피소드는 steps의 시퀀스.
    각 step은 observation(image, state), action, language_instruction 포함.
    """
    images = episode_data["images"]
    states = episode_data["states"]
    actions = episode_data["actions"]
    num_frames = episode_data["num_frames"]

    # 에피소드 steps 구성
    steps = []
    for i in range(num_frames):
        # 이미지를 JPEG 인코딩
        img_encoded = cv2.imencode(".jpg", cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))[1].tobytes()

        state = states[i]
        action = actions[i]

        # 정규화 (옵션 — RLDS 자체에서는 raw 값 저장, 학습 시 정규화하는 게 표준)
        if normalize:
            state = normalize_to_bounds(state)
            action = normalize_to_bounds(action)

        step_feature = {
            "observation/image_primary": _bytes_feature(img_encoded),
            "observation/image_primary_shape": tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(images[i].shape))
            ),
            "observation/state": _float_feature(state.tolist()),
            "action": _float_feature(action.tolist()),
            "language_instruction": _bytes_feature(LANGUAGE_INSTRUCTION.encode("utf-8")),
            "is_first": _int64_feature(1 if i == 0 else 0),
            "is_last": _int64_feature(1 if i == num_frames - 1 else 0),
            "is_terminal": _int64_feature(1 if i == num_frames - 1 else 0),
        }
        steps.append(tf.train.Example(features=tf.train.Features(feature=step_feature)))

    # 각 step을 개별 Example로 쓴다
    for step in steps:
        writer.write(step.SerializeToString())


# ─── Dataset Statistics ──────────────────────────────────────────────────────

def compute_and_save_statistics(all_states: list, all_actions: list, output_dir: Path):
    """dataset_statistics.json 생성.

    OpenVLA 추론 시 역정규화에 필수!
    BOUNDS 정규화이므로 min/max = 하드웨어 스펙 범위.
    """
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    stats = {
        "action": {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            # BOUNDS 정규화용 하드웨어 범위
            "bounds_low": JOINT_BOUNDS[:, 0].tolist(),
            "bounds_high": JOINT_BOUNDS[:, 1].tolist(),
            "p01": np.percentile(actions, 1, axis=0).tolist(),
            "p99": np.percentile(actions, 99, axis=0).tolist(),
        },
        "proprio": {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
            "bounds_low": JOINT_BOUNDS[:, 0].tolist(),
            "bounds_high": JOINT_BOUNDS[:, 1].tolist(),
            "p01": np.percentile(states, 1, axis=0).tolist(),
            "p99": np.percentile(states, 99, axis=0).tolist(),
        },
        "num_transitions": int(len(actions)),
        "num_trajectories": len(all_states),
        "joint_names": JOINT_NAMES,
        "normalization_type": "BOUNDS",
    }

    stats_path = output_dir / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")
    return stats


# ─── TFDS-compatible Dataset Builder ─────────────────────────────────────────

def create_rlds_dataset(input_dir: str, output_dir: str):
    """collected_data_v5/ → RLDS TFRecord 변환.

    tfds DatasetBuilder를 사용하지 않고 직접 TFRecord를 생성합니다.
    이유: tfds build는 복잡한 설정 필요. OpenVLA의 make_dataset_from_rlds()는
    결국 tfds.load()로 TFRecord를 읽으므로, tfds 표준 구조만 맞추면 됨.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / DATASET_NAME

    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)

    # 에피소드 목록
    episodes = sorted(input_path.glob("episode_*"))
    if not episodes:
        print(f"ERROR: No episodes found in {input_path}")
        sys.exit(1)

    print(f"Found {len(episodes)} episodes in {input_path}")

    # 통계용 수집
    all_states = []
    all_actions = []
    episode_lengths = []

    # TFRecord 쓰기
    tfrecord_path = output_path / "train.tfrecord"
    writer = tf.io.TFRecordWriter(str(tfrecord_path))

    for ep_idx, ep_dir in enumerate(episodes):
        ep_data = load_episode(ep_dir)

        if ep_data["num_frames"] < 2:
            print(f"  SKIP: {ep_dir.name} (too few frames: {ep_data['num_frames']})")
            continue

        # TFRecord에 raw 값 저장 (정규화는 학습 시 수행)
        write_tfrecord_episode(writer, ep_data, normalize=False)

        all_states.append(ep_data["states"])
        all_actions.append(ep_data["actions"])
        episode_lengths.append(ep_data["num_frames"])

        if (ep_idx + 1) % 20 == 0 or ep_idx == len(episodes) - 1:
            print(f"  Processed {ep_idx + 1}/{len(episodes)} episodes")

    writer.close()
    print(f"\nTFRecord saved: {tfrecord_path}")

    # 통계 저장
    stats = compute_and_save_statistics(all_states, all_actions, output_path)

    # 메타데이터 저장
    meta = {
        "dataset_name": DATASET_NAME,
        "num_episodes": len(episode_lengths),
        "num_frames": sum(episode_lengths),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "min_episode_length": int(min(episode_lengths)),
        "max_episode_length": int(max(episode_lengths)),
        "image_shape": [720, 1280, 3],
        "action_dim": 6,
        "state_dim": 6,
        "fps": 30,
        "language_instruction": LANGUAGE_INSTRUCTION,
        "joint_bounds": JOINT_BOUNDS.tolist(),
        "normalization_type": "BOUNDS (hardware spec)",
    }
    meta_path = output_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # 검증 요약
    print(f"\n{'='*60}")
    print(f"RLDS Dataset Created: {output_path}")
    print(f"{'='*60}")
    print(f"  Episodes: {meta['num_episodes']}")
    print(f"  Total frames: {meta['num_frames']}")
    print(f"  Avg episode length: {meta['avg_episode_length']:.1f}")
    print(f"  Action dim: {meta['action_dim']}")
    print(f"  Image shape: {meta['image_shape']}")
    print(f"  Language: '{LANGUAGE_INSTRUCTION}'")
    print(f"  Normalization: BOUNDS (hardware spec)")
    print()
    print("  Action range (raw degrees):")
    for i, name in enumerate(JOINT_NAMES):
        print(f"    {name:12s}: [{stats['action']['min'][i]:8.2f}, {stats['action']['max'][i]:8.2f}]"
              f"  bounds=[{JOINT_BOUNDS[i,0]:7.1f}, {JOINT_BOUNDS[i,1]:7.1f}]")

    # 정규화 검증
    print("\n  Normalization verification (sample frame 0, ep 0):")
    sample = all_actions[0][0]
    norm = normalize_to_bounds(sample)
    denorm = denormalize_from_bounds(norm)
    error = np.abs(denorm - sample).max()
    print(f"    Raw:    {sample}")
    print(f"    Norm:   {norm}")
    print(f"    Denorm: {denorm}")
    print(f"    Max roundtrip error: {error:.6f}")
    if error > 1e-4:
        print("    ⚠ WARNING: Roundtrip error too high!")
    else:
        print("    ✓ Roundtrip OK")

    return output_path


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("OpenVLA RLDS Converter for RoArm M3")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}/{DATASET_NAME}")
    print()

    output = create_rlds_dataset(INPUT_DIR, OUTPUT_DIR)
    print(f"\nDone. Dataset at: {output}")
