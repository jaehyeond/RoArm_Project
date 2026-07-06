"""Supervised warm-start of the D277 PPO actor from the D257 teacher sidecar."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)
DEFAULT_D256_CSV = (
    REPO
    / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/"
    "cube10cm_top_view_visual_0_999_d242/rl_transition_preflight_d256/"
    "ppo_actor_prior_teacher_rows_d256.csv"
)
DEFAULT_TEACHER_CHECKPOINT = (
    REPO
    / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/"
    "cube10cm_top_view_visual_0_999_d242/state_action_teacher_d257/"
    "cube10cm_d257_state_action_teacher_clipped0040.pt"
)
DEFAULT_SOURCE_ACTOR_CHECKPOINT = (
    REPO
    / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/"
    "ppo_data_prior_d277_logs/cube10cm_d277_tap10cm_aabb_d256reset_bc_episode_complete_smoke/"
    "model_0.pt"
)
DEFAULT_OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/"
    "actor_distill_d280/tap10cm"
)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def _tensor_mean(x) -> float:
    return float(x.detach().float().mean().cpu().item())


def _tensor_max(x) -> float:
    return float(x.detach().float().max().cpu().item())


def _metrics(torch, pred, target) -> dict[str, float]:
    diff = pred - target
    cosine = torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1.0e-6)
    return {
        "mse": _tensor_mean(torch.mean(diff * diff, dim=-1)),
        "mae": _tensor_mean(torch.mean(torch.abs(diff), dim=-1)),
        "cosine": _tensor_mean(cosine),
        "pred_abs_mean": _tensor_mean(torch.mean(torch.abs(pred), dim=-1)),
        "target_abs_mean": _tensor_mean(torch.mean(torch.abs(target), dim=-1)),
        "pred_abs_max": _tensor_max(torch.abs(pred)),
        "target_abs_max": _tensor_max(torch.abs(target)),
    }


def _per_dim_metrics(torch, pred, target, labels: list[str]) -> list[dict[str, float | int | str]]:
    out = []
    diff = pred - target
    for idx, label in enumerate(labels):
        out.append(
            {
                "dim": idx,
                "label": label,
                "pred_abs_mean": _tensor_mean(torch.abs(pred[:, idx])),
                "target_abs_mean": _tensor_mean(torch.abs(target[:, idx])),
                "abs_gap_mean": _tensor_mean(torch.abs(diff[:, idx])),
                "mse": _tensor_mean(diff[:, idx] * diff[:, idx]),
                "pred_signed_mean": _tensor_mean(pred[:, idx]),
                "target_signed_mean": _tensor_mean(target[:, idx]),
            }
        )
    return out


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D280 Actor Distillation",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- source actor: `{summary['source_actor_checkpoint']}`",
        f"- distilled checkpoint: `{summary['distilled_actor_checkpoint']}`",
        f"- teacher checkpoint: `{summary['teacher_checkpoint']}`",
        f"- samples train/val: `{summary['train_samples']}` / `{summary['val_samples']}`",
        f"- initial val MSE/MAE/cosine: `{summary['initial_val_metrics']['mse']}` / `{summary['initial_val_metrics']['mae']}` / `{summary['initial_val_metrics']['cosine']}`",
        f"- final val MSE/MAE/cosine: `{summary['final_val_metrics']['mse']}` / `{summary['final_val_metrics']['mae']}` / `{summary['final_val_metrics']['cosine']}`",
        f"- teacher rollout contact/useful/reaction: `{summary['teacher_rollout_contact_seen_rate']}` / `{summary['teacher_rollout_useful_seen_rate']}` / `{summary['teacher_rollout_reaction_seen_rate']}`",
        f"- teacher rollout overshoot: `{summary['teacher_rollout_overshoot_seen_rate']}`",
        f"- D256 reset active: `{summary['d256_reset_active_rate']}`",
        "",
        "## Issues",
        "",
    ]
    if summary["issues"]:
        lines.extend(f"- {issue}" for issue in summary["issues"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is supervised actor warm-start, not PPO training. It only attempts to make the rsl_rl actor match the D257 teacher sidecar under the D256 reset/AABB contract.",
            "Promotion still requires teacher-off eval and actor-vs-teacher trace after loading the saved actor checkpoint.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_actor_checkpoint", type=Path, default=DEFAULT_SOURCE_ACTOR_CHECKPOINT)
    parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--collect_steps", type=int, default=580)
    parser.add_argument("--seed", type=int, default=28001)
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--episode_length_s", type=float, default=6.0)
    parser.add_argument("--d256_reset_csv_path", type=Path, default=DEFAULT_D256_CSV)
    parser.add_argument("--d256_reset_frame_index", type=int, default=0)
    parser.add_argument("--d256_reset_sample_mode", choices=("random", "linspace"), default="linspace")
    parser.add_argument("--fixed_push_dir_x", type=float, default=1.0)
    parser.add_argument("--fixed_push_dir_y", type=float, default=0.0)
    parser.add_argument("--tap_contact_proxy_mode", choices=("tcp_point", "link5_collision_aabb"), default="link5_collision_aabb")
    parser.add_argument("--bc_teacher_feature_target_mode", choices=("tcp_target", "env_target"), default="env_target")
    parser.add_argument("--bc_teacher_phase_timing", choices=("episode_scaled", "direct_steps"), default="direct_steps")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--max_val_mse", type=float, default=0.02)
    parser.add_argument("--min_val_cosine", type=float, default=0.90)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact_tag", type=str, default="d280_actor_distill")
    args = parser.parse_args()

    if int(args.collect_steps) <= 0:
        raise ValueError("--collect_steps must be positive")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if not (0.0 < float(args.train_fraction) < 1.0):
        raise ValueError("--train_fraction must be in (0, 1)")
    for path in (args.source_actor_checkpoint, args.teacher_checkpoint, args.d256_reset_csv_path):
        if not path.exists():
            raise FileNotFoundError(path)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import torch
    import torch.nn.functional as F
    import roarm_rl  # noqa: F401 - registers envs
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg

    torch.manual_seed(int(args.seed))

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.fixed_push_dir_x = float(args.fixed_push_dir_x)
    env_cfg.fixed_push_dir_y = float(args.fixed_push_dir_y)
    env_cfg.tap_contact_proxy_mode = str(args.tap_contact_proxy_mode)
    env_cfg.d256_reset_csv_path = str(args.d256_reset_csv_path)
    env_cfg.d256_reset_frame_index = int(args.d256_reset_frame_index)
    env_cfg.d256_reset_sample_mode = str(args.d256_reset_sample_mode)
    env_cfg.bc_teacher_checkpoint_path = str(args.teacher_checkpoint)
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    env_cfg.bc_teacher_feature_target_mode = str(args.bc_teacher_feature_target_mode)
    env_cfg.bc_teacher_phase_timing = str(args.bc_teacher_phase_timing)

    ppo_cfg = RoArmPickPPORunnerCfg()
    ppo_cfg.seed = int(args.seed)

    env_id = "RoArm-CubeTap10cm-Direct-v0"
    print(
        "[actor-distill] scope=cube10cm_top_view_d280_actor_distill "
        f"env_id={env_id} training=PPO_NO supervised=YES num_envs={args.num_envs} "
        f"collect_steps={args.collect_steps}",
        flush=True,
    )

    env = gym.make(env_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    inner = env.unwrapped
    if int(args.collect_steps) >= int(inner.max_episode_length) - 1:
        raise ValueError(
            f"--collect_steps {args.collect_steps} would hit env truncation/reset; "
            f"use <= {int(inner.max_episode_length) - 2}"
        )
    if not getattr(inner, "_bc_teacher_ready", False):
        raise RuntimeError("BC teacher sidecar did not load")

    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir=None, device=inner.device)
    runner.load(str(args.source_actor_checkpoint), load_optimizer=False, map_location=inner.device)
    policy = runner.alg.policy

    inner.episode_length_buf[:] = inner.max_episode_length
    obs = env.get_observations()
    with torch.inference_mode():
        zero_actions = torch.zeros((inner.num_envs, int(inner.cfg.action_space)), device=inner.device)
        obs, _, _, _ = env.step(zero_actions)
    print("[actor-distill] warmup_reset_done", flush=True)

    obs_parts = []
    target_parts = []
    phase_means: list[float] = []
    with torch.inference_mode():
        for _step in range(int(args.collect_steps)):
            actor_obs = policy.get_actor_obs(obs).detach().clone()
            traj = inner._bc_teacher_traj()
            phase_alpha = inner._bc_teacher_phase_alpha(traj)
            teacher_actions = torch.clamp(inner._bc_teacher_actions(), -1.0, 1.0).detach().clone()
            obs_parts.append(actor_obs)
            target_parts.append(teacher_actions)
            phase_means.append(_tensor_mean(phase_alpha))
            obs, _, _, _ = env.step(teacher_actions)

    obs_all = torch.cat(obs_parts, dim=0)
    target_all = torch.cat(target_parts, dim=0)
    sample_count = int(obs_all.shape[0])
    action_labels = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
    if len(action_labels) != int(target_all.shape[-1]):
        action_labels = [f"action_{idx}" for idx in range(int(target_all.shape[-1]))]
    useful_min_disp_m = max(float(getattr(inner.cfg, "tap_useful_min_disp_m", 0.001)), 0.0)
    useful_seen = (
        inner._tap_contact_seen
        & inner._tap_reaction_seen
        & (inner._tap_max_disp_xy >= useful_min_disp_m)
        & ~inner._tap_overshoot_seen
    )
    teacher_rollout_contact = _tensor_mean(inner._tap_contact_seen.float())
    teacher_rollout_reaction = _tensor_mean(inner._tap_reaction_seen.float())
    teacher_rollout_useful = _tensor_mean(useful_seen.float())
    teacher_rollout_overshoot = _tensor_mean(inner._tap_overshoot_seen.float())
    d256_active = _tensor_mean(inner._last_d256_reset_active)

    perm = torch.randperm(sample_count, device=inner.device)
    train_n = int(sample_count * float(args.train_fraction))
    train_idx = perm[:train_n]
    val_idx = perm[train_n:]
    train_obs = obs_all[train_idx]
    train_target = target_all[train_idx]
    val_obs = obs_all[val_idx]
    val_target = target_all[val_idx]

    def actor_forward(raw_obs):
        return policy.actor(policy.actor_obs_normalizer(raw_obs))

    policy.eval()
    with torch.inference_mode():
        initial_train_metrics = _metrics(torch, actor_forward(train_obs), train_target)
        initial_val_pred = actor_forward(val_obs)
        initial_val_metrics = _metrics(torch, initial_val_pred, val_target)
        initial_per_dim = _per_dim_metrics(torch, initial_val_pred, val_target, action_labels)

    optimizer = torch.optim.AdamW(
        policy.actor.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    loss_rows: list[dict[str, float | int]] = []
    policy.actor.train()
    policy.actor_obs_normalizer.eval()
    for epoch in range(int(args.epochs)):
        epoch_perm = train_idx[torch.randperm(train_n, device=inner.device)]
        losses = []
        cosine_losses = []
        for start in range(0, train_n, int(args.batch_size)):
            batch_idx = epoch_perm[start : start + int(args.batch_size)]
            pred = actor_forward(obs_all[batch_idx])
            target = target_all[batch_idx]
            mse = F.mse_loss(pred, target)
            cosine_loss = (1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1.0e-6)).mean()
            loss = mse + 0.02 * cosine_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(mse.detach().cpu().item()))
            cosine_losses.append(float(cosine_loss.detach().cpu().item()))

        policy.actor.eval()
        with torch.inference_mode():
            val_pred = actor_forward(val_obs)
            val_metrics = _metrics(torch, val_pred, val_target)
        policy.actor.train()
        row = {
            "epoch": epoch,
            "train_mse_batch_mean": sum(losses) / len(losses),
            "train_cosine_loss_batch_mean": sum(cosine_losses) / len(cosine_losses),
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_cosine": val_metrics["cosine"],
        }
        loss_rows.append(row)
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == int(args.epochs):
            print(
                "[actor-distill] "
                f"epoch={epoch + 1}/{args.epochs} val_mse={row['val_mse']:.6f} "
                f"val_cosine={row['val_cosine']:.6f}",
                flush=True,
            )

    policy.actor.eval()
    with torch.inference_mode():
        final_train_pred = actor_forward(train_obs)
        final_val_pred = actor_forward(val_obs)
        final_train_metrics = _metrics(torch, final_train_pred, train_target)
        final_val_metrics = _metrics(torch, final_val_pred, val_target)
        final_per_dim = _per_dim_metrics(torch, final_val_pred, val_target, action_labels)

    issues: list[str] = []
    if d256_active < 0.99:
        issues.append(f"D256 reset inactive during collection: {d256_active}")
    if final_val_metrics["mse"] > float(args.max_val_mse):
        issues.append(f"final val MSE above threshold: {final_val_metrics['mse']}")
    if final_val_metrics["cosine"] < float(args.min_val_cosine):
        issues.append(f"final val cosine below threshold: {final_val_metrics['cosine']}")
    if teacher_rollout_overshoot > 0.05:
        issues.append(f"teacher rollout overshoot high during data collection: {teacher_rollout_overshoot}")

    verdict = (
        "D280_ACTOR_DISTILL_SUPERVISED_FIT_PASS_NEEDS_ROLLOUT_EVAL"
        if not issues
        else "D280_ACTOR_DISTILL_SUPERVISED_FIT_WARN_NEEDS_ROLLOUT_EVAL"
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_checkpoint = out_dir / "model_actor_distill_d280.pt"
    out_json = out_dir / "actor_distill_summary_d280.json"
    out_md = out_dir / "actor_distill_summary_d280.md"
    out_csv = out_dir / "actor_distill_losses_d280.csv"

    runner.current_learning_iteration = 0
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "iter": runner.current_learning_iteration,
            "infos": {"artifact_tag": str(args.artifact_tag), "verdict": verdict},
        },
        out_checkpoint,
    )

    with out_csv.open("w", newline="") as f:
        fieldnames = list(loss_rows[0].keys()) if loss_rows else ["epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(loss_rows)

    summary = {
        "artifact_tag": str(args.artifact_tag),
        "verdict": verdict,
        "issues": issues,
        "source_actor_checkpoint": _rel(args.source_actor_checkpoint),
        "distilled_actor_checkpoint": _rel(out_checkpoint),
        "teacher_checkpoint": _rel(args.teacher_checkpoint),
        "env_id": env_id,
        "num_envs": int(args.num_envs),
        "collect_steps": int(args.collect_steps),
        "episode_length_s": float(env_cfg.episode_length_s),
        "max_episode_length": int(inner.max_episode_length),
        "seed": int(args.seed),
        "sample_count": sample_count,
        "train_samples": int(train_obs.shape[0]),
        "val_samples": int(val_obs.shape[0]),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "d256_reset_csv_path": _rel(args.d256_reset_csv_path),
        "d256_reset_frame_index": int(args.d256_reset_frame_index),
        "d256_reset_sample_mode": str(args.d256_reset_sample_mode),
        "tap_contact_proxy_mode": str(args.tap_contact_proxy_mode),
        "bc_teacher_feature_target_mode": str(args.bc_teacher_feature_target_mode),
        "bc_teacher_phase_timing": str(args.bc_teacher_phase_timing),
        "d256_reset_active_rate": d256_active,
        "teacher_rollout_contact_seen_rate": teacher_rollout_contact,
        "teacher_rollout_reaction_seen_rate": teacher_rollout_reaction,
        "teacher_rollout_useful_seen_rate": teacher_rollout_useful,
        "tap_useful_min_disp_m": useful_min_disp_m,
        "teacher_rollout_overshoot_seen_rate": teacher_rollout_overshoot,
        "phase_alpha_mean_first": phase_means[0] if phase_means else None,
        "phase_alpha_mean_last": phase_means[-1] if phase_means else None,
        "initial_train_metrics": initial_train_metrics,
        "initial_val_metrics": initial_val_metrics,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "initial_val_per_dim": initial_per_dim,
        "final_val_per_dim": final_per_dim,
        "out_checkpoint": _rel(out_checkpoint),
        "out_json": _rel(out_json),
        "out_md": _rel(out_md),
        "out_csv": _rel(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_md(out_md, summary)

    print(
        "[actor-distill] SUMMARY "
        f"verdict={verdict} val_mse={final_val_metrics['mse']:.6f} "
        f"val_cosine={final_val_metrics['cosine']:.6f} checkpoint={out_checkpoint}",
        flush=True,
    )

    env.close()
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
