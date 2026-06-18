#!/usr/bin/env python3
"""Train a small PPO-compatible state-action teacher from D256 rows.

This is supervised pretraining for a PPO data prior. It does not launch Isaac
Lab, run PPO, render, delete files, or control RoArm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    REPO
    / "claudedocs"
    / "runtime_logs"
    / "20260526_cube3cm_push_rollout_probe_20480"
    / "cube10cm_top_view_visual_0_999_d242"
)
D256_ROOT = RUNTIME_ROOT / "rl_transition_preflight_d256"
DEFAULT_TEACHER_CSV = D256_ROOT / "ppo_actor_prior_teacher_rows_d256.csv"
DEFAULT_OUT_DIR = RUNTIME_ROOT / "state_action_teacher_d257"

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_model(in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int):
    import torch

    layers: list[torch.nn.Module] = []
    last = int(in_dim)
    for _ in range(int(hidden_layers)):
        layers.append(torch.nn.Linear(last, int(hidden_dim)))
        layers.append(torch.nn.ReLU())
        last = int(hidden_dim)
    layers.append(torch.nn.Linear(last, int(out_dim)))
    return torch.nn.Sequential(*layers)


def quantiles(values, qs: Iterable[float]) -> dict[str, float]:
    import torch

    flat = values.detach().abs().reshape(-1).cpu()
    return {f"p{int(q * 100):02d}": float(torch.quantile(flat, q).item()) for q in qs}


def load_rows(csv_path: Path, target_clip_rad: float):
    import torch

    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    episode_ids: list[int] = []
    subsplits: set[str] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty csv: {csv_path}")
        missing = [c for c in ["episode_index", "package_subsplit", *FEATURE_COLUMNS, *TARGET_COLUMNS] if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing columns in {csv_path}: {missing}")
        for row in reader:
            subsplits.add(str(row["package_subsplit"]))
            x_rows.append([float(row[c]) for c in FEATURE_COLUMNS])
            y_rows.append([float(row[c]) for c in TARGET_COLUMNS])
            episode_ids.append(int(row["episode_index"]))

    if not x_rows:
        raise ValueError(f"no teacher rows found in {csv_path}")
    if subsplits != {"train_clean_positive"}:
        raise ValueError(f"teacher csv must contain only train_clean_positive rows, got {sorted(subsplits)}")

    x = torch.tensor(x_rows, dtype=torch.float32)
    y_raw = torch.tensor(y_rows, dtype=torch.float32)
    if target_clip_rad > 0.0:
        y = torch.clamp(y_raw, -float(target_clip_rad), float(target_clip_rad))
    else:
        y = y_raw.clone()
    episodes = torch.tensor(episode_ids, dtype=torch.long)
    return x, y, y_raw, episodes, sorted(set(episode_ids))


def split_by_episode(episodes, unique_episodes: list[int], val_episode_stride: int):
    import torch

    stride = max(2, int(val_episode_stride))
    val_eps = {ep for idx, ep in enumerate(unique_episodes) if idx % stride == 0}
    if len(val_eps) == len(unique_episodes):
        val_eps = {unique_episodes[-1]}
    val_mask = torch.tensor([int(ep.item()) in val_eps for ep in episodes], dtype=torch.bool)
    train_mask = ~val_mask
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise ValueError("episode split produced an empty train or validation set")
    return train_mask, val_mask, sorted(val_eps)


def write_ppo_smoke_command(out_dir: Path, checkpoint_path: Path) -> Path:
    command_path = out_dir / "ppo_data_prior_smoke_command_d257.txt"
    command = f"""# Proposed next-session smoke only. Not run by this script.
# Keep tiny first; stop/cleanup GPU process after completion.
conda run -n isaaclab python roarm_rl/train_cube_push_ppo.py \\
  --num_envs 32 \\
  --max_iterations 2 \\
  --seed 1257 \\
  --experiment_name cube10cm_d257_data_prior_smoke2 \\
  --logdir claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/ppo_data_prior_d257_logs \\
  --episode_length_s 6.0 \\
  --action_scale 0.04 \\
  --action_smoothing_alpha 1.0 \\
  --max_joint_delta_per_step_rad 0.04 \\
  --contact_joint_delta_scale 1.0 \\
  --fast_cube_joint_delta_scale 1.0 \\
  --joint_target_lead_limit_rad 0.06 \\
  --joint_delta_reference joint_pos \\
  --bc_teacher_checkpoint_path {checkpoint_path.relative_to(REPO)} \\
  --bc_teacher_blend 1.0 \\
  --bc_teacher_imitation_reward_scale 5.0 \\
  --bc_teacher_policy_delta_clip_rad 0.04 \\
  --bc_teacher_policy_delta_scale 1.0 \\
  --bc_teacher_lowx_policy_delta_scale 1.0 \\
  --bc_teacher_highx_policy_delta_scale 0.8 \\
  --bc_teacher_delta_smoothing_alpha 0.85 \\
  --bc_teacher_phase_timing direct_steps \\
  --num_steps_per_env 24 \\
  --save_interval 1
"""
    command_path.write_text(command)
    return command_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_csv", type=Path, default=DEFAULT_TEACHER_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--target_clip_rad", type=float, default=0.04)
    parser.add_argument("--val_episode_stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1257)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    import torch

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    elif args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    teacher_csv = args.teacher_csv
    x, y, y_raw, episodes, unique_episodes = load_rows(teacher_csv, float(args.target_clip_rad))
    train_mask, val_mask, val_episodes = split_by_episode(episodes, unique_episodes, int(args.val_episode_stride))

    x_train = x[train_mask]
    y_train = y[train_mask]
    x_val = x[val_mask]
    y_val = y[val_mask]
    y_raw_train = y_raw[train_mask]
    y_raw_val = y_raw[val_mask]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1.0e-6)

    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    x_val_n = (x_val - x_mean) / x_std
    y_val_n = (y_val - y_mean) / y_std

    model = make_model(len(FEATURE_COLUMNS), len(TARGET_COLUMNS), int(args.hidden_dim), int(args.hidden_layers)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.MSELoss()
    train_ds = torch.utils.data.TensorDataset(x_train_n, y_train_n)
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )

    baseline_val_mse = float(loss_fn(torch.zeros_like(y_val_n), y_val_n).item())
    best_val_mse = float("inf")
    best_state = None
    first_train_mse = None
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log(
        "teacher_train line1 "
        f"rows={int(x.shape[0])} train_rows={int(train_mask.sum())} val_rows={int(val_mask.sum())} "
        f"episodes={len(unique_episodes)} val_episodes={len(val_episodes)} device={device.type}"
    )
    log(
        "teacher_train line2 "
        f"target_clip_rad={float(args.target_clip_rad):.6f} hidden_dim={int(args.hidden_dim)} "
        f"hidden_layers={int(args.hidden_layers)} epochs={int(args.epochs)} batch_size={int(args.batch_size)}"
    )

    x_train_n_dev = x_train_n.to(device)
    y_train_n_dev = y_train_n.to(device)
    x_val_n_dev = x_val_n.to(device)
    y_val_n_dev = y_val_n.to(device)

    for epoch in range(int(args.epochs)):
        model.train()
        loss_sum = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * int(xb.shape[0])
            seen += int(xb.shape[0])
        train_mse = loss_sum / max(1, seen)
        if first_train_mse is None:
            first_train_mse = float(train_mse)
        model.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(model(x_val_n_dev), y_val_n_dev).item())
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == int(args.epochs):
            log(f"teacher_train epoch={epoch + 1:03d} train_mse_norm={train_mse:.9f} val_mse_norm={val_mse:.9f}")

    if best_state is None:
        raise RuntimeError("no best model state captured")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred_n = model(x_train_n_dev).cpu()
        val_pred_n = model(x_val_n_dev).cpu()
    train_pred = train_pred_n * y_std + y_mean
    val_pred = val_pred_n * y_std + y_mean
    train_mse_norm = float(loss_fn(train_pred_n, y_train_n).item())
    val_mse_norm = float(loss_fn(val_pred_n, y_val_n).item())
    train_rmse_rad = torch.sqrt(torch.mean((train_pred - y_train) ** 2, dim=0))
    val_rmse_rad = torch.sqrt(torch.mean((val_pred - y_val) ** 2, dim=0))
    val_mae_rad = torch.mean(torch.abs(val_pred - y_val), dim=0)

    clip = float(args.target_clip_rad)
    raw_abs = y_raw.abs()
    clip_exceed_count = int((raw_abs > clip).sum().item()) if clip > 0.0 else 0
    clip_exceed_rate = float(clip_exceed_count / max(1, raw_abs.numel()))
    raw_quant = quantiles(y_raw, (0.50, 0.95, 0.99))
    clipped_quant = quantiles(y, (0.50, 0.95, 0.99))

    checkpoint_path = out_dir / "cube10cm_d257_state_action_teacher_clipped0040.pt"
    checkpoint = {
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "model_state_dict": best_state,
        "x_mean": x_mean.squeeze(0).cpu(),
        "x_std": x_std.squeeze(0).cpu(),
        "y_mean": y_mean.squeeze(0).cpu(),
        "y_std": y_std.squeeze(0).cpu(),
        "metadata": {
            "artifact": "cube10cm_d257_state_action_teacher",
            "source": "D256 train_clean_positive teacher-prior rows",
            "teacher_csv": str(teacher_csv.relative_to(REPO) if teacher_csv.is_relative_to(REPO) else teacher_csv),
            "teacher_csv_sha256": sha256_file(teacher_csv),
            "target_clip_rad": clip,
            "target_was_clipped_for_training": clip > 0.0,
            "rows_total": int(x.shape[0]),
            "rows_train": int(train_mask.sum().item()),
            "rows_val": int(val_mask.sum().item()),
            "episode_count": len(unique_episodes),
            "val_episode_count": len(val_episodes),
            "seed": int(args.seed),
            "device": device.type,
            "runtime": "NO_ISAAC_NO_PPO_NO_RENDER_NO_DELETE_NO_ROARM_CONTROL",
        },
    }
    torch.save(checkpoint, checkpoint_path)

    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reload_model = make_model(
        len(reloaded["feature_columns"]),
        len(reloaded["target_columns"]),
        int(reloaded["hidden_dim"]),
        int(reloaded["hidden_layers"]),
    )
    reload_model.load_state_dict(reloaded["model_state_dict"])
    reload_model.eval()
    with torch.no_grad():
        reload_out = reload_model((x_val[:8] - reloaded["x_mean"].view(1, -1)) / reloaded["x_std"].view(1, -1))
    reload_ok = tuple(reload_out.shape) == (min(8, int(x_val.shape[0])), len(TARGET_COLUMNS))

    ppo_command_path = write_ppo_smoke_command(out_dir, checkpoint_path)
    metrics_path = out_dir / "state_action_teacher_metrics_d257.json"
    log_path = out_dir / "state_action_teacher_training_log_d257.txt"
    summary_path = out_dir / "state_action_teacher_summary_d257.md"
    elapsed_s = time.time() - started

    checkpoint_sha = sha256_file(checkpoint_path)
    metrics = {
        "artifact": "cube10cm_d257_state_action_teacher",
        "status": "PASS" if reload_ok and val_mse_norm < baseline_val_mse and train_mse_norm < float(first_train_mse) else "CHECK",
        "teacher_csv": str(teacher_csv.relative_to(REPO) if teacher_csv.is_relative_to(REPO) else teacher_csv),
        "teacher_csv_sha256": sha256_file(teacher_csv),
        "checkpoint_path": str(checkpoint_path.relative_to(REPO)),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "metrics_path": str(metrics_path.relative_to(REPO)),
        "ppo_smoke_command_path": str(ppo_command_path.relative_to(REPO)),
        "rows_total": int(x.shape[0]),
        "rows_train": int(train_mask.sum().item()),
        "rows_val": int(val_mask.sum().item()),
        "episode_count": len(unique_episodes),
        "val_episode_count": len(val_episodes),
        "val_episode_stride": int(args.val_episode_stride),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "device": device.type,
        "baseline_val_mse_norm": baseline_val_mse,
        "first_train_mse_norm": float(first_train_mse),
        "final_train_mse_norm": train_mse_norm,
        "best_val_mse_norm": float(best_val_mse),
        "final_val_mse_norm": val_mse_norm,
        "train_rmse_rad_by_joint": [float(v) for v in train_rmse_rad.tolist()],
        "val_rmse_rad_by_joint": [float(v) for v in val_rmse_rad.tolist()],
        "val_mae_rad_by_joint": [float(v) for v in val_mae_rad.tolist()],
        "raw_target_abs_quantiles_rad": raw_quant,
        "clipped_target_abs_quantiles_rad": clipped_quant,
        "raw_target_abs_max_by_joint": [float(v) for v in torch.max(torch.abs(y_raw), dim=0).values.tolist()],
        "raw_target_clip_exceed_count": clip_exceed_count,
        "raw_target_clip_exceed_rate": clip_exceed_rate,
        "target_clip_rad": clip,
        "target_was_clipped_for_training": clip > 0.0,
        "checkpoint_reload_ok": bool(reload_ok),
        "isaac_lab_ppo_runtime_executed": False,
        "teacher_is_final_policy": False,
        "roarm_deployment_ready": False,
        "elapsed_s": elapsed_s,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    log_lines.append(f"teacher_train line3 final_train_mse_norm={train_mse_norm:.9f} final_val_mse_norm={val_mse_norm:.9f}")
    log_lines.append(f"teacher_train line4 checkpoint={checkpoint_path.relative_to(REPO)} sha256={checkpoint_sha}")
    log_lines.append(f"teacher_train line5 status={metrics['status']} reload_ok={reload_ok} elapsed_s={elapsed_s:.2f}")
    log_path.write_text("\n".join(log_lines) + "\n")
    summary_path.write_text(
        "# D257 State-Action Teacher Summary\n\n"
        f"- status: `{metrics['status']}`\n"
        f"- checkpoint: `{checkpoint_path.relative_to(REPO)}`\n"
        f"- checkpoint sha256: `{checkpoint_sha}`\n"
        f"- rows total/train/val: `{metrics['rows_total']}` / `{metrics['rows_train']}` / `{metrics['rows_val']}`\n"
        f"- final train MSE norm: `{train_mse_norm:.9f}`\n"
        f"- final validation MSE norm: `{val_mse_norm:.9f}`\n"
        f"- baseline validation MSE norm: `{baseline_val_mse:.9f}`\n"
        f"- target clip rad: `{clip:.6f}`\n"
        f"- raw target clip exceed rate: `{clip_exceed_rate:.9f}`\n"
        f"- checkpoint reload ok: `{reload_ok}`\n"
        "- Isaac Lab PPO runtime was not executed in this script.\n"
        "- This teacher is a data-prior bridge, not a final learned RoArm policy.\n"
    )

    print(log_lines[-3])
    print(log_lines[-2])
    print(log_lines[-1])
    return 0 if metrics["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
