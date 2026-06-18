#!/usr/bin/env python3
"""Build RL transition/reward preflight artifacts from the D247 LeRobot pair data.

This is a local data-prep preflight only. It does not train a model, launch
Isaac Lab, render frames, copy raw images, delete files, or control RoArm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    REPO
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
    / "cube10cm_top_view_visual_0_999_d242"
)
FRAMES_JSONL = RUNTIME_ROOT / "frames.jsonl"
SPLIT_MANIFEST = RUNTIME_ROOT / "label_package_d248" / "episode_split_manifest.csv"
LEROBOT_INFO = RUNTIME_ROOT / "lerobot_dataset_av1_d247" / "meta" / "info.json"
DEFAULT_OUT_DIR = RUNTIME_ROOT / "rl_transition_preflight_d256"

FEATURE_COLUMNS = [
    "push_dx",
    "push_dy",
    "phase_alpha",
    "cube_local_x_m",
    "cube_local_y_m",
    "cube_local_z_m",
    "tcp_local_x_m",
    "tcp_local_y_m",
    "tcp_local_z_m",
    "target_local_x_m",
    "target_local_y_m",
    "target_local_z_m",
    "tcp_to_cube_x_m",
    "tcp_to_cube_y_m",
    "tcp_to_cube_z_m",
    "target_to_tcp_x_m",
    "target_to_tcp_y_m",
    "target_to_tcp_z_m",
    "target_to_cube_x_m",
    "target_to_cube_y_m",
    "target_to_cube_z_m",
    "arm_joint_0_rad",
    "arm_joint_1_rad",
    "arm_joint_2_rad",
    "arm_joint_3_rad",
    "arm_joint_4_rad",
    "gripper_joint_rad",
]

TARGET_COLUMNS = [
    "joint_delta_0_rad",
    "joint_delta_1_rad",
    "joint_delta_2_rad",
    "joint_delta_3_rad",
    "joint_delta_4_rad",
]

TRANSITION_COLUMNS = [
    "transition_index",
    "episode_index",
    "frame_index_t",
    "frame_index_tp1",
    "global_index_t",
    "global_index_tp1",
    "timestamp_t_s",
    "timestamp_tp1_s",
    "done",
    "package_split",
    "package_subsplit",
    "rl_role",
    "label_status",
    "camera_contract_pass",
    "use_for_actor_prior",
    "use_for_reward_dataset",
    "reward_sparse_terminal_v0",
    "reward_dense_event_v0",
    "reward_total_v0",
    "reward_version",
    "progress_delta_m",
    "contact_onset",
    "reaction_onset",
    "overshoot_onset",
    "tap_disp_along_t_m",
    "tap_disp_along_tp1_m",
    "tap_disp_xy_t_m",
    "tap_disp_xy_tp1_m",
    "tap_speed_t_mps",
    "tap_speed_tp1_mps",
    "source_png_t",
    "source_png_tp1",
    "lerobot_video_key",
    *FEATURE_COLUMNS,
    *TARGET_COLUMNS,
    "state_0_t",
    "state_1_t",
    "state_2_t",
    "state_3_t",
    "state_4_t",
    "state_5_t",
    "action_joint_target_0_t",
    "action_joint_target_1_t",
    "action_joint_target_2_t",
    "action_joint_target_3_t",
    "action_joint_target_4_t",
    "action_joint_target_5_t",
    "state_0_tp1",
    "state_1_tp1",
    "state_2_tp1",
    "state_3_tp1",
    "state_4_tp1",
    "state_5_tp1",
]

TEACHER_PRIOR_COLUMNS = [
    "transition_index",
    "episode_index",
    "frame_index_t",
    "global_index_t",
    "package_subsplit",
    "label_status",
    *FEATURE_COLUMNS,
    *TARGET_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-jsonl", type=Path, default=FRAMES_JSONL)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--lerobot-info", type=Path, default=LEROBOT_INFO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-scale-m", type=float, default=0.006)
    parser.add_argument("--teacher-delta-clip-rad", type=float, default=0.040)
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_split_manifest(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as fp:
        rows = [dict(row) for row in csv.DictReader(fp)]
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        episode_id = int(row["episode_index"])
        if episode_id in out:
            raise ValueError(f"duplicate episode_index in split manifest: {episode_id}")
        out[episode_id] = row
    return out


def read_frames(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fp:
        for index, line in enumerate(fp):
            if not line.strip():
                continue
            row = json.loads(line)
            row["global_index"] = index
            rows.append(row)
    rows.sort(key=lambda row: (int(row["episode_id"]), int(row["frame_id"])))
    for expected, row in enumerate(rows):
        if int(row["global_index"]) != expected:
            raise ValueError(f"non-contiguous global index at sorted row {expected}")
    return rows


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def vec(row: dict[str, Any], key: str, length: int) -> list[float]:
    value = row.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must be length {length}, got {value!r}")
    return [float(x) for x in value]


def bool_int(value: Any) -> int:
    return 1 if bool(int(float(value))) else 0


def split_role(split_row: dict[str, str]) -> str:
    subsplit = split_row["package_subsplit"]
    if subsplit == "train_clean_positive":
        return "train_positive_actor_prior"
    if subsplit == "eval_clean_holdout":
        return "eval_positive_holdout"
    if subsplit == "eval_overshoot_diagnostic":
        return "eval_negative_overshoot_diagnostic"
    if subsplit == "quarantine_camera_fail":
        return "quarantine_camera_excluded"
    return "unknown"


def terminal_reward(split_row: dict[str, str]) -> float:
    label = split_row["label_status"]
    if label == "clean_useful_tap":
        return 1.0
    if label == "contact_reaction_with_overshoot":
        return -1.0
    if label == "camera_quality_fail":
        return -2.0
    return 0.0


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def feature_values(row: dict[str, Any], split_row: dict[str, str]) -> dict[str, float]:
    cube = vec(row, "cube_position_world_m", 3)
    tcp = vec(row, "tcp_position_world_m", 3)
    target = vec(row, "target_position_world_m", 3)
    push = vec(row, "push_dir_xy", 2)
    state = vec(row, "observation_state", 6)
    last_sim_step = max(1.0, float(split_row.get("last_sim_step", 580) or 580))
    phase_alpha = clamp(as_float(row, "sim_step") / last_sim_step, 0.0, 1.0)

    values = {
        "push_dx": push[0],
        "push_dy": push[1],
        "phase_alpha": phase_alpha,
        "cube_local_x_m": cube[0],
        "cube_local_y_m": cube[1],
        "cube_local_z_m": cube[2],
        "tcp_local_x_m": tcp[0],
        "tcp_local_y_m": tcp[1],
        "tcp_local_z_m": tcp[2],
        "target_local_x_m": target[0],
        "target_local_y_m": target[1],
        "target_local_z_m": target[2],
        "tcp_to_cube_x_m": cube[0] - tcp[0],
        "tcp_to_cube_y_m": cube[1] - tcp[1],
        "tcp_to_cube_z_m": cube[2] - tcp[2],
        "target_to_tcp_x_m": target[0] - tcp[0],
        "target_to_tcp_y_m": target[1] - tcp[1],
        "target_to_tcp_z_m": target[2] - tcp[2],
        "target_to_cube_x_m": target[0] - cube[0],
        "target_to_cube_y_m": target[1] - cube[1],
        "target_to_cube_z_m": target[2] - cube[2],
        "arm_joint_0_rad": state[0],
        "arm_joint_1_rad": state[1],
        "arm_joint_2_rad": state[2],
        "arm_joint_3_rad": state[3],
        "arm_joint_4_rad": state[4],
        "gripper_joint_rad": state[5],
    }
    return values


def make_transition(
    *,
    index: int,
    row: dict[str, Any],
    next_row: dict[str, Any],
    split_row: dict[str, str],
    progress_scale_m: float,
) -> dict[str, Any]:
    state = vec(row, "observation_state", 6)
    next_state = vec(next_row, "observation_state", 6)
    action = vec(row, "action", 6)
    features = feature_values(row, split_row)
    joint_delta = {f"joint_delta_{i}_rad": action[i] - state[i] for i in range(5)}

    contact_onset = int(as_float(next_row, "tap_contact_seen") >= 0.5 and as_float(row, "tap_contact_seen") < 0.5)
    reaction_onset = int(as_float(next_row, "tap_reaction_seen") >= 0.5 and as_float(row, "tap_reaction_seen") < 0.5)
    overshoot_onset = int(as_float(next_row, "tap_overshoot_seen") >= 0.5 and as_float(row, "tap_overshoot_seen") < 0.5)
    progress_delta = as_float(next_row, "tap_disp_along_m") - as_float(row, "tap_disp_along_m")
    progress_reward = 0.05 * clamp(progress_delta / max(progress_scale_m, 1.0e-9), -1.0, 1.0)
    dense_reward = progress_reward + 0.25 * contact_onset + 0.25 * reaction_onset - 0.75 * overshoot_onset
    done = int(int(next_row["frame_id"]) == int(split_row["last_frame_index"]))
    sparse = terminal_reward(split_row) if done else 0.0
    role = split_role(split_row)

    out: dict[str, Any] = {
        "transition_index": index,
        "episode_index": int(row["episode_id"]),
        "frame_index_t": int(row["frame_id"]),
        "frame_index_tp1": int(next_row["frame_id"]),
        "global_index_t": int(row["global_index"]),
        "global_index_tp1": int(next_row["global_index"]),
        "timestamp_t_s": as_float(row, "timestamp_s"),
        "timestamp_tp1_s": as_float(next_row, "timestamp_s"),
        "done": done,
        "package_split": split_row["package_split"],
        "package_subsplit": split_row["package_subsplit"],
        "rl_role": role,
        "label_status": split_row["label_status"],
        "camera_contract_pass": bool_int(split_row["camera_contract_pass"]),
        "use_for_actor_prior": int(role == "train_positive_actor_prior"),
        "use_for_reward_dataset": int(role in {"train_positive_actor_prior", "eval_negative_overshoot_diagnostic"}),
        "reward_sparse_terminal_v0": sparse,
        "reward_dense_event_v0": dense_reward,
        "reward_total_v0": sparse + dense_reward,
        "reward_version": "d256_candidate_v0_not_final",
        "progress_delta_m": progress_delta,
        "contact_onset": contact_onset,
        "reaction_onset": reaction_onset,
        "overshoot_onset": overshoot_onset,
        "tap_disp_along_t_m": as_float(row, "tap_disp_along_m"),
        "tap_disp_along_tp1_m": as_float(next_row, "tap_disp_along_m"),
        "tap_disp_xy_t_m": as_float(row, "tap_disp_xy_m"),
        "tap_disp_xy_tp1_m": as_float(next_row, "tap_disp_xy_m"),
        "tap_speed_t_mps": as_float(row, "tap_speed_mps"),
        "tap_speed_tp1_mps": as_float(next_row, "tap_speed_mps"),
        "source_png_t": row.get("source_png", ""),
        "source_png_tp1": next_row.get("source_png", ""),
        "lerobot_video_key": "observation.images.top",
        **features,
        **joint_delta,
    }
    for i, value in enumerate(state):
        out[f"state_{i}_t"] = value
    for i, value in enumerate(action):
        out[f"action_joint_target_{i}_t"] = value
    for i, value in enumerate(next_state):
        out[f"state_{i}_tp1"] = value
    return out


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return float(ordered[idx])


def main() -> None:
    args = parse_args()
    args.frames_jsonl = args.frames_jsonl.resolve()
    args.split_manifest = args.split_manifest.resolve()
    args.lerobot_info = args.lerobot_info.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists; pass --force to overwrite D256 outputs")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_rows = read_split_manifest(args.split_manifest)
    frames = read_frames(args.frames_jsonl)
    info = read_json(args.lerobot_info)

    if int(info.get("total_frames", -1)) != len(frames):
        raise RuntimeError(f"LeRobot total_frames {info.get('total_frames')} != frames.jsonl {len(frames)}")
    if int(info.get("total_episodes", -1)) != len(split_rows):
        raise RuntimeError(f"LeRobot total_episodes {info.get('total_episodes')} != split rows {len(split_rows)}")

    transitions: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    by_episode = defaultdict(list)
    for row in frames:
        by_episode[int(row["episode_id"])].append(row)

    transition_index = 0
    validation_errors: list[str] = []
    for episode_id in sorted(by_episode):
        if episode_id not in split_rows:
            validation_errors.append(f"missing split row for episode {episode_id}")
            continue
        episode_rows = sorted(by_episode[episode_id], key=lambda row: int(row["frame_id"]))
        split_row = split_rows[episode_id]
        if len(episode_rows) != int(split_row["num_frames"]):
            validation_errors.append(
                f"episode {episode_id} frame count {len(episode_rows)} != manifest {split_row['num_frames']}"
            )
        for left, right in zip(episode_rows[:-1], episode_rows[1:], strict=True):
            tr = make_transition(
                index=transition_index,
                row=left,
                next_row=right,
                split_row=split_row,
                progress_scale_m=float(args.progress_scale_m),
            )
            transitions.append(tr)
            if int(tr["use_for_actor_prior"]) == 1:
                teacher_rows.append({key: tr[key] for key in TEACHER_PRIOR_COLUMNS})
            transition_index += 1

    if validation_errors:
        raise RuntimeError("; ".join(validation_errors[:10]))

    transitions_csv = args.out_dir / "rl_transitions_d256.csv"
    teacher_csv = args.out_dir / "ppo_actor_prior_teacher_rows_d256.csv"
    summary_json = args.out_dir / "rl_transition_preflight_summary_d256.json"
    brief_md = args.out_dir / "rl_transition_preflight_brief_d256.md"

    write_csv(transitions_csv, TRANSITION_COLUMNS, transitions)
    write_csv(teacher_csv, TEACHER_PRIOR_COLUMNS, teacher_rows)

    role_counts = defaultdict(int)
    subsplit_counts = defaultdict(int)
    reward_by_role: dict[str, list[float]] = defaultdict(list)
    max_abs_delta_by_joint = [0.0] * 5
    all_abs_delta: list[float] = []
    clip_exceed_count = 0
    for tr in transitions:
        role = str(tr["rl_role"])
        role_counts[role] += 1
        subsplit_counts[str(tr["package_subsplit"])] += 1
        reward_by_role[role].append(float(tr["reward_total_v0"]))
        for i in range(5):
            value = abs(float(tr[f"joint_delta_{i}_rad"]))
            all_abs_delta.append(value)
            max_abs_delta_by_joint[i] = max(max_abs_delta_by_joint[i], value)
            if value > float(args.teacher_delta_clip_rad) + 1.0e-12:
                clip_exceed_count += 1

    reward_summary = {
        role: {
            "count": len(values),
            "mean": mean(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
        for role, values in sorted(reward_by_role.items())
    }

    summary = {
        "artifact": "cube10cm_top_view_rl_transition_preflight_d256",
        "runtime": "NO_TRAINING_NO_RENDER_NO_DELETE_NO_ROARM_CONTROL",
        "status": "PASS",
        "render_dir": str(RUNTIME_ROOT.relative_to(REPO)),
        "frames_jsonl": str(args.frames_jsonl.relative_to(REPO)),
        "split_manifest": str(args.split_manifest.relative_to(REPO)),
        "lerobot_info": str(args.lerobot_info.relative_to(REPO)),
        "lerobot_feature_keys": list(info.get("features", {}).keys()),
        "lerobot_total_frames": int(info.get("total_frames")),
        "lerobot_total_episodes": int(info.get("total_episodes")),
        "frames_read": len(frames),
        "episodes_read": len(by_episode),
        "transitions": len(transitions),
        "expected_transitions": sum(max(0, len(rows) - 1) for rows in by_episode.values()),
        "transition_csv": str(transitions_csv.relative_to(REPO)),
        "transition_csv_sha256": sha256(transitions_csv),
        "teacher_prior_csv": str(teacher_csv.relative_to(REPO)),
        "teacher_prior_csv_sha256": sha256(teacher_csv),
        "teacher_prior_rows": len(teacher_rows),
        "role_counts": dict(sorted(role_counts.items())),
        "subsplit_transition_counts": dict(sorted(subsplit_counts.items())),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "target_semantics": "joint_delta_i_rad = action_joint_target_i_t - state_i_t for arm joints 0..4",
        "coordinate_assumption": "single-env Isaac world coordinates are used as local coordinates for PPO teacher-prior preflight",
        "reward_version": "d256_candidate_v0_not_final",
        "reward_definition": {
            "terminal_clean_useful_tap": 1.0,
            "terminal_contact_reaction_with_overshoot": -1.0,
            "terminal_camera_quality_fail": -2.0,
            "contact_onset": 0.25,
            "reaction_onset": 0.25,
            "overshoot_onset": -0.75,
            "progress_delta": f"0.05 * clip(delta_disp_along / {float(args.progress_scale_m)}, -1, 1)",
        },
        "reward_summary_by_role": reward_summary,
        "teacher_delta_clip_rad": float(args.teacher_delta_clip_rad),
        "joint_delta_abs_max_by_joint": max_abs_delta_by_joint,
        "joint_delta_abs_p95": percentile(all_abs_delta, 0.95),
        "joint_delta_abs_p99": percentile(all_abs_delta, 0.99),
        "joint_delta_clip_exceed_count": clip_exceed_count,
        "joint_delta_clip_exceed_rate": float(clip_exceed_count / max(1, len(all_abs_delta))),
        "ppo_data_prior_path": {
            "env_supports_bc_teacher_checkpoint_path": True,
            "env_supports_bc_teacher_blend": True,
            "env_supports_bc_teacher_imitation_reward_scale": True,
            "compatible_feature_columns_with_roarm_cube_push_env": True,
            "compatible_target_column_count": len(TARGET_COLUMNS) == 5,
            "next_required_artifact": "train a small state-action teacher checkpoint from teacher_prior_csv, then run Isaac Lab PPO smoke with bc_teacher_imitation_reward_scale > 0",
        },
        "blocked_until_explicit_approval": [
            "teacher checkpoint training",
            "Isaac Lab PPO runtime",
            "RunPod runtime",
            "RoArm deployment/control",
            "additional render/dataset generation",
            "delete/archive/move cleanup",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    brief_md.write_text(
        "\n".join(
            [
                "# D256 RL Transition Preflight Brief",
                "",
                f"- Status: `{summary['status']}`",
                f"- Transitions: `{summary['transitions']}`",
                f"- Teacher-prior rows: `{summary['teacher_prior_rows']}`",
                f"- Transition CSV: `{summary['transition_csv']}`",
                f"- Teacher-prior CSV: `{summary['teacher_prior_csv']}`",
                "- Reward: `d256_candidate_v0_not_final`, not a final training reward.",
                "- No training, render, deletion, RunPod, B200, or RoArm control was run.",
                "",
            ]
        )
    )
    print(
        "[cube10cm-rl-transition-preflight] "
        f"status=PASS transitions={summary['transitions']} teacher_rows={summary['teacher_prior_rows']} "
        f"out={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
