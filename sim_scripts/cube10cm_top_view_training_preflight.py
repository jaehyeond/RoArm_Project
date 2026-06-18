#!/usr/bin/env python3
"""Preflight the D247-D252 cube10cm top-view dataset for future LeRobot training.

This script does not train, render, delete, move, or archive anything. It checks
that the approved training split can be consumed as a LeRobot episode-filtered
dataset and writes a proposed command for a later, explicitly approved run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    REPO_ROOT
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
    / "cube10cm_top_view_visual_0_999_d242"
)
LEROBOT_ROOT = RUNTIME_ROOT / "lerobot_dataset_av1_d247"
FREEZE_DIR = RUNTIME_ROOT / "dataset_freeze_d249"
LABEL_DIR = RUNTIME_ROOT / "label_package_d248"
FILTERED_DIR = RUNTIME_ROOT / "filtered_views_d250"
D251_DIR = FILTERED_DIR / "dataloader_smoke_d251"
D252_DIR = RUNTIME_ROOT / "split_distribution_d252"
OUT_DIR = RUNTIME_ROOT / "training_preflight_d253"

REPO_ID = "roarm_cube10cm_top_view_0_999_d247"
OUTPUT_DIR_CANDIDATE = "outputs/smolvla_cube10cm_top_view_d253_candidate"
OUTPUT_DIR_SMOKE = "outputs/smolvla_cube10cm_top_view_d253_smoke"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def read_ids(path: Path) -> list[int]:
    ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(int(line))
    return ids


def write_ids(path: Path, ids: list[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for episode_id in ids:
            f.write(f"{episode_id}\n")


def tensorish_to_py(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(x) for x in shape]


def build_command(episodes: list[int], *, smoke: bool) -> list[str]:
    steps = 50 if smoke else 20_000
    save_freq = 50 if smoke else 5_000
    output_dir = OUTPUT_DIR_SMOKE if smoke else OUTPUT_DIR_CANDIDATE
    command = [
        "HF_HOME=/tmp/roarm_hf_cache",
        "HF_DATASETS_CACHE=/tmp/roarm_hf_datasets_cache",
        "conda",
        "run",
        "-n",
        "lerobot",
        "--no-capture-output",
        "lerobot-train",
        "--policy.type=smolvla",
        "--policy.pretrained_path=lerobot/smolvla_base",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.root={LEROBOT_ROOT}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        "--dataset.video_backend=pyav",
        "--batch_size=64",
        f"--steps={steps}",
        "--eval_freq=0",
        f"--save_freq={save_freq}",
        "--log_freq=10" if smoke else "--log_freq=100",
        f"--output_dir={output_dir}",
        "--num_workers=4",
        "--policy.device=cuda",
        "--wandb.enable=false",
    ]
    if smoke:
        command.append("--save_checkpoint=false")
    return command


def shell_command_line(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def verify_split_integrity(splits: dict[str, list[int]], package: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_counts = package["counts"]["by_subsplit"]
    for name, ids in splits.items():
        if len(ids) != expected_counts[name]:
            errors.append(f"{name} count mismatch: {len(ids)} != {expected_counts[name]}")
        if ids != sorted(ids):
            errors.append(f"{name} ids are not sorted")
        if len(ids) != len(set(ids)):
            errors.append(f"{name} ids contain duplicates")

    all_ids: list[int] = []
    for ids in splits.values():
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        errors.append("split episode ids overlap")
    if sorted(all_ids) != list(range(1000)):
        errors.append("split episode ids do not cover exactly 0..999")
    return errors


def verify_lerobot_factory(train_ids: list[int]) -> dict[str, Any]:
    start = time.perf_counter()
    from lerobot.datasets.factory import make_dataset

    dataset_cfg = SimpleNamespace(
        repo_id=REPO_ID,
        root=str(LEROBOT_ROOT),
        episodes=train_ids,
        image_transforms=SimpleNamespace(enable=False),
        revision=None,
        use_imagenet_stats=False,
        video_backend="pyav",
        streaming=False,
    )
    policy_cfg = SimpleNamespace(
        reward_delta_indices=None,
        action_delta_indices=None,
        observation_delta_indices=None,
    )
    cfg = SimpleNamespace(
        dataset=dataset_cfg,
        policy=policy_cfg,
        tolerance_s=1e-4,
        num_workers=0,
    )
    dataset = make_dataset(cfg)
    factory_s = time.perf_counter() - start

    if dataset.num_episodes != len(train_ids):
        raise RuntimeError(f"LeRobot selected episodes {dataset.num_episodes} != {len(train_ids)}")

    sample_indices = [0, dataset.num_frames // 2, dataset.num_frames - 1]
    samples: list[dict[str, Any]] = []
    for index in sample_indices:
        sample_start = time.perf_counter()
        sample = dataset[index]
        decode_s = time.perf_counter() - sample_start
        samples.append(
            {
                "dataset_index": index,
                "decode_s": decode_s,
                "episode_index": tensorish_to_py(sample.get("episode_index")),
                "frame_index": tensorish_to_py(sample.get("frame_index")),
                "image_shape": shape_of(sample.get("observation.images.top")),
                "state_shape": shape_of(sample.get("observation.state")),
                "action_shape": shape_of(sample.get("action")),
            }
        )

    loader_start = time.perf_counter()
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    loader_s = time.perf_counter() - loader_start

    return {
        "factory_path": "lerobot.datasets.factory.make_dataset",
        "dataset_root": str(LEROBOT_ROOT),
        "repo_id": REPO_ID,
        "video_backend": "pyav",
        "selected_episodes": dataset.num_episodes,
        "selected_frames": dataset.num_frames,
        "factory_init_s": factory_s,
        "samples": samples,
        "dataloader_batch": {
            "batch_size": 4,
            "loader_first_batch_s": loader_s,
            "image_shape": shape_of(batch.get("observation.images.top")),
            "state_shape": shape_of(batch.get("observation.state")),
            "action_shape": shape_of(batch.get("action")),
        },
    }


def write_brief(path: Path, summary: dict[str, Any]) -> None:
    smoke_file = OUT_DIR / "proposed_smolvla_train_smoke_50_steps_d253.txt"
    candidate_file = OUT_DIR / "proposed_smolvla_train_candidate_20000_steps_d253.txt"
    text = f"""# D253 Training Preflight Brief

Status: `{summary['status']}`

This is not a training run. No model checkpoint was created.

## What Was Verified

- `train_clean_positive` means 학습용 정상 성공 예시: camera-pass and clean useful tap episodes only.
- `eval_clean_holdout` means 평가용 정상 보류 예시: clean useful tap episodes held out for later model evaluation.
- `eval_overshoot_diagnostic` means 과하게 민 케이스 진단용 평가 데이터: camera-pass episodes where the cube moved too far, kept out of positive BC training.
- `quarantine_camera_fail` means 카메라 기준 실패 격리 데이터: camera-contract failures excluded from train and eval.
- LeRobot official dataset factory can consume `train_clean_positive` through `dataset.episodes`.
- LeRobot video decoding must use `dataset.video_backend=pyav` on this local environment.

## Critical Notes

- LeRobot `eval_freq` is environment rollout evaluation, not this dataset's held-out split.
- Therefore `eval_clean_holdout` and `eval_overshoot_diagnostic` remain offline evaluation inputs for a later script.
- Actual SmolVLA/VLA fine-tuning still needs explicit approval.
- PPO, action-teacher, RoArm deployment, RunPod runtime, deletion, and extra rendering remain out of scope.

## Proposed Later Smoke Command

Full command is stored in:

`{smoke_file}`

## Proposed Later Candidate Command

Full command is stored in:

`{candidate_file}`
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite D253 outputs if they already exist.")
    parser.add_argument(
        "--skip-lerobot",
        action="store_true",
        help="Skip LeRobot factory import/decode check. Use only if the lerobot env is unavailable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()) and not args.force:
        raise SystemExit(f"{OUT_DIR} already has files; rerun with --force to overwrite D253 preflight outputs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    package = read_json(LABEL_DIR / "split_package_summary.json")
    freeze = read_json(FREEZE_DIR / "dataset_freeze_manifest_d249.json")
    filtered = read_json(FILTERED_DIR / "filtered_views_summary.json")
    dataloader = read_json(D251_DIR / "filtered_dataloader_smoke_summary.json")
    distribution = read_json(D252_DIR / "split_distribution_summary.json")
    info = read_json(LEROBOT_ROOT / "meta" / "info.json")

    splits = {
        "train_clean_positive": read_ids(LABEL_DIR / "train_clean_positive_episode_ids.txt"),
        "eval_clean_holdout": read_ids(LABEL_DIR / "eval_clean_holdout_episode_ids.txt"),
        "eval_overshoot_diagnostic": read_ids(LABEL_DIR / "eval_overshoot_diagnostic_episode_ids.txt"),
        "quarantine_camera_fail": read_ids(LABEL_DIR / "quarantine_camera_fail_episode_ids.txt"),
    }

    errors = verify_split_integrity(splits, package)
    for name, artifact in [
        ("split_package_summary", package),
        ("dataset_freeze_manifest", freeze),
        ("filtered_views_summary", filtered),
        ("filtered_dataloader_smoke", dataloader),
        ("split_distribution_summary", distribution),
    ]:
        if artifact.get("status") != "PASS":
            errors.append(f"{name} status is {artifact.get('status')}, expected PASS")

    if info.get("total_episodes") != 1000:
        errors.append(f"LeRobot total_episodes {info.get('total_episodes')} != 1000")
    if info.get("total_frames") != 195000:
        errors.append(f"LeRobot total_frames {info.get('total_frames')} != 195000")
    if info.get("fps") != 30:
        errors.append(f"LeRobot fps {info.get('fps')} != 30")

    train_ids = splits["train_clean_positive"]
    lerobot_check: dict[str, Any]
    if args.skip_lerobot:
        lerobot_check = {"skipped": True}
    else:
        try:
            lerobot_check = verify_lerobot_factory(train_ids)
        except Exception as exc:
            errors.append(f"LeRobot factory train split check failed: {exc}")
            lerobot_check = {"error": repr(exc)}

    write_ids(OUT_DIR / "train_clean_positive_episode_ids_d253.txt", train_ids)
    write_ids(OUT_DIR / "eval_clean_holdout_episode_ids_d253.txt", splits["eval_clean_holdout"])
    write_ids(OUT_DIR / "eval_overshoot_diagnostic_episode_ids_d253.txt", splits["eval_overshoot_diagnostic"])
    write_ids(OUT_DIR / "quarantine_camera_fail_episode_ids_d253.txt", splits["quarantine_camera_fail"])

    command_smoke = shell_command_line(build_command(train_ids, smoke=True))
    command_candidate = shell_command_line(build_command(train_ids, smoke=False))
    (OUT_DIR / "proposed_smolvla_train_smoke_50_steps_d253.txt").write_text(
        command_smoke + "\n", encoding="utf-8"
    )
    (OUT_DIR / "proposed_smolvla_train_candidate_20000_steps_d253.txt").write_text(
        command_candidate + "\n", encoding="utf-8"
    )

    summary = {
        "artifact": "cube10cm_top_view_training_preflight_d253",
        "status": "PASS" if not errors else "FAIL",
        "runtime": "NO_TRAINING_NO_RENDER_NO_DELETE_TRAINING_PREFLIGHT_ONLY",
        "repo_id": REPO_ID,
        "dataset_root": str(LEROBOT_ROOT),
        "source_artifacts": {
            "d248_label_package": str(LABEL_DIR / "split_package_summary.json"),
            "d249_freeze": str(FREEZE_DIR / "dataset_freeze_manifest_d249.json"),
            "d250_filtered_views": str(FILTERED_DIR / "filtered_views_summary.json"),
            "d251_dataloader_smoke": str(D251_DIR / "filtered_dataloader_smoke_summary.json"),
            "d252_distribution": str(D252_DIR / "split_distribution_summary.json"),
        },
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "selected_train_split": {
            "name": "train_clean_positive",
            "korean_definition": "학습용 정상 성공 예시",
            "episodes": len(train_ids),
            "expected_frames": filtered["views"]["train_clean_positive"]["frames"],
            "selection_mechanism": "LeRobot DatasetConfig.episodes / LeRobotDataset episodes filter",
        },
        "held_out_splits": {
            "eval_clean_holdout": {
                "korean_definition": "평가용 정상 보류 예시",
                "episodes": len(splits["eval_clean_holdout"]),
            },
            "eval_overshoot_diagnostic": {
                "korean_definition": "과하게 민 케이스 진단용 평가 데이터",
                "episodes": len(splits["eval_overshoot_diagnostic"]),
            },
            "quarantine_camera_fail": {
                "korean_definition": "카메라 기준 실패 격리 데이터",
                "episodes": len(splits["quarantine_camera_fail"]),
            },
        },
        "lerobot_factory_check": lerobot_check,
        "proposed_commands": {
            "smoke_50_steps": command_smoke,
            "candidate_20000_steps": command_candidate,
        },
        "blocked_until_explicit_approval": [
            "run the 50-step SmolVLA training smoke",
            "run the 20k-step SmolVLA candidate training",
            "RunPod runtime or H100 jobs",
            "PPO/L2/Large PPO/action-teacher/RoArm deployment",
            "delete, move, or archive files",
            "render additional episodes",
        ],
        "errors": errors,
        "notes": [
            "LeRobot train has one dataset input; eval_clean_holdout is not automatically used by lerobot-train.",
            "Dataset held-out evaluation requires a separate offline evaluation script after an approved checkpoint exists.",
            "Use pyav because local torchcodec/default decode was previously not reliable for AV1.",
        ],
    }
    write_json(OUT_DIR / "training_preflight_summary_d253.json", summary)
    write_brief(OUT_DIR / "training_preflight_brief_d253.md", summary)

    print(json.dumps({"status": summary["status"], "out_dir": str(OUT_DIR)}, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
