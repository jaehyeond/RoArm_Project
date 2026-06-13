#!/usr/bin/env python3
"""Render a small D232 top-view visual smoke set for the 10cm cube tap task.

This is a camera/data-contract smoke only:
- no PPO training or checkpoint loading
- no B200/SSH access
- no large dataset generation
- no deletion or cleanup

Output is debug PNG plus JSONL metadata. A separate roarm-env converter turns the
debug frames into a LeRobot video+parquet smoke dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_OUT = LOG_DIR / "cube10cm_top_view_visual_smoke_d232"
DEFAULT_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)

CAMERA_PATH = "/World/Cube10cmTopViewCamera"
CAMERA_CONTRACT_ID = "cube10cm_top_view_v1_candidate"
WIDTH = 1280
HEIGHT = 720
FPS = 30
FX = 608.33
FY = 608.28
CX = 638.31
CY = 365.26
TABLE_CENTER_XY = (0.25, 0.0)
TABLE_Z_TOP = -0.012117
CAMERA_HEIGHT_ABOVE_TABLE_M = 0.65
CAMERA_CENTER = (
    TABLE_CENTER_XY[0],
    TABLE_CENTER_XY[1],
    TABLE_Z_TOP + CAMERA_HEIGHT_ABOVE_TABLE_M,
)
CUBE_SIZE_M = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2320)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=580)
    parser.add_argument("--capture-stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--poses",
        default="0.24,0.0;0.14,-0.10;0.14,0.10;0.34,-0.10;0.34,0.10",
        help="semicolon-separated fixed cube xy pairs, e.g. '0.24,0;0.14,-0.1'",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--robot-usd-path", type=Path, default=DEFAULT_USD)
    parser.add_argument(
        "--close-sim-app",
        action="store_true",
        help="Explicitly close Isaac app at the end. Default skips close because local Kit close can hang.",
    )
    return parser.parse_args()


def parse_poses(raw: str, n: int) -> list[tuple[float, float]]:
    poses: list[tuple[float, float]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        x_raw, y_raw = [part.strip() for part in item.split(",", maxsplit=1)]
        poses.append((float(x_raw), float(y_raw)))
    if not poses:
        raise ValueError("--poses must contain at least one x,y pair")
    return [poses[i % len(poses)] for i in range(n)]


def base_contract_args(args: argparse.Namespace, fixed_x: float, fixed_y: float) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        seed=int(args.seed),
        num_envs=1,
        max_iterations=0,
        num_steps_per_env=64,
        eval_steps=int(args.steps_per_episode),
        save_interval=1,
        robot_usd_path=args.robot_usd_path,
        runtime_dir=args.out_dir,
        summary_json=args.out_dir / "unused_summary.json",
        summary_out=args.out_dir / "unused_summary.out",
        experiment_name="cube10cm_top_view_visual_smoke_d232",
        fixed_cube_x_m=float(fixed_x),
        fixed_cube_y_m=float(fixed_y),
        cube_randomization_half_extent_x_m=0.0,
        cube_randomization_half_extent_y_m=0.0,
        policy_target_disp_m=0.006,
        precontact_clearance_m=0.040,
        episode_length_s=6.08,
        step_clip_rad=0.010,
        joint_target_lead_limit_rad=0.060,
        action_scale=0.050,
        rl_action_mode="candidate8_diffik_target_residual",
        candidate6_diffik_push_steps=580,
        candidate6_diffik_residual_scale_rad=0.002,
        candidate6_diffik_lambda=0.010,
        candidate8_diffik_target_residual_forward_m=0.004,
        candidate8_diffik_target_residual_lateral_m=0.012,
        candidate8_diffik_target_residual_height_m=0.004,
        tap_success_terminate=False,
        disable_tap_overshoot_terminate=True,
        candidate6_diffik_no_hold_after_tap_success=False,
        candidate6_diffik_target_base_mode="previous_joint_target",
        candidate6_diffik_target_path_mode="near_face_goal",
        candidate6_diffik_cube_reference_mode="current_pose",
        init_at_random_ep_len=False,
        load_checkpoint=None,
        initial_policy_eval=False,
        constant_policy_action=None,
        ppo_init_noise_std=0.2,
        tap_transient_disp_reward_scale=None,
        tap_overshoot_penalty_scale=None,
        action_penalty_scale=None,
    )


def scalar(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return math.nan


def tensor_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        out = value.tolist()
    else:
        out = list(value)
    return [float(x) for x in out]


def project_top_view(point_xyz: tuple[float, float, float]) -> tuple[float, float]:
    x_w, y_w, z_w = point_xyz
    x_c = x_w - CAMERA_CENTER[0]
    y_c = -(y_w - CAMERA_CENTER[1])
    z_c = CAMERA_CENTER[2] - z_w
    if z_c <= 0.0:
        return (math.nan, math.nan)
    u = FX * x_c / z_c + CX
    v = FY * y_c / z_c + CY
    return (float(u), float(v))


def cube_top_projection(cube_xyz: list[float]) -> dict[str, Any]:
    half = CUBE_SIZE_M * 0.5
    z_top = float(cube_xyz[2]) + half
    corners = [
        (float(cube_xyz[0]) - half, float(cube_xyz[1]) - half, z_top),
        (float(cube_xyz[0]) - half, float(cube_xyz[1]) + half, z_top),
        (float(cube_xyz[0]) + half, float(cube_xyz[1]) - half, z_top),
        (float(cube_xyz[0]) + half, float(cube_xyz[1]) + half, z_top),
    ]
    uv = [project_top_view(xyz) for xyz in corners]
    center_uv = project_top_view((float(cube_xyz[0]), float(cube_xyz[1]), z_top))
    us = [p[0] for p in uv if math.isfinite(p[0])]
    vs = [p[1] for p in uv if math.isfinite(p[1])]
    if not us or not vs:
        return {"uv_corners": uv, "uv_center": center_uv, "bbox": None, "inside": False}
    bbox = [
        max(0, int(math.floor(min(us)))),
        max(0, int(math.floor(min(vs)))),
        min(WIDTH - 1, int(math.ceil(max(us)))),
        min(HEIGHT - 1, int(math.ceil(max(vs)))),
    ]
    inside = all(0.0 <= u < WIDTH and 0.0 <= v < HEIGHT for u, v in uv)
    return {"uv_corners": uv, "uv_center": center_uv, "bbox": bbox, "inside": inside}


def blue_visibility(rgb: Any, projection: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    bbox = projection.get("bbox")
    if bbox is None:
        return {
            "cube_visibility": "cube_occluded_full",
            "blue_coverage": 0.0,
            "blue_pixels": 0,
            "bbox_area": 0,
            "centroid_error_px": math.nan,
        }
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return {
            "cube_visibility": "cube_occluded_full",
            "blue_coverage": 0.0,
            "blue_pixels": 0,
            "bbox_area": 0,
            "centroid_error_px": math.nan,
        }
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1, :3].astype(np.int16)
    red = crop[:, :, 0]
    green = crop[:, :, 1]
    blue = crop[:, :, 2]
    mask = (blue > 120) & (blue > red + 8) & (green > red + 4)
    blue_pixels = int(mask.sum())
    area = int(mask.size)
    coverage = float(blue_pixels / max(1, area))
    if coverage >= 0.25:
        label = "cube_visible_full"
    elif coverage >= 0.03:
        label = "cube_visible_partial"
    else:
        label = "cube_occluded_full"

    centroid_error = math.nan
    if blue_pixels > 0:
        ys, xs = np.nonzero(mask)
        cx_px = float(xs.mean() + x0)
        cy_px = float(ys.mean() + y0)
        u, v = projection["uv_center"]
        if math.isfinite(u) and math.isfinite(v):
            centroid_error = float(math.hypot(cx_px - u, cy_px - v))
    return {
        "cube_visibility": label,
        "blue_coverage": coverage,
        "blue_pixels": blue_pixels,
        "bbox_area": area,
        "centroid_error_px": centroid_error,
    }


def install_camera(stage: Any, width: int, height: int) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    dome_path = "/World/Cube10cmTopViewDomeLight"
    if not stage.GetPrimAtPath(dome_path):
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(2500.0)
        dome.CreateColorAttr(Gf.Vec3f(0.86, 0.86, 0.86))

    aperture_h = 20.955
    hfov = 2.0 * math.atan(float(width) / (2.0 * FX))
    focal = 0.5 * aperture_h / math.tan(hfov * 0.5)
    cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    cam.CreateFocalLengthAttr(float(focal))
    cam.CreateHorizontalApertureAttr(float(aperture_h))
    cam.CreateVerticalApertureAttr(float(aperture_h * float(height) / float(width)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 10.0))
    cam_xf = UsdGeom.Xformable(cam.GetPrim())
    cam_xf.ClearXformOpOrder()
    cam_xf.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_CENTER))


def force_reset(env: Any, inner_env: Any, torch: Any) -> Any:
    inner_env.episode_length_buf[:] = inner_env.max_episode_length
    obs = env.get_observations()
    actions = torch.zeros((inner_env.num_envs, int(inner_env.cfg.action_space)), device=inner_env.device)
    obs, _, _, _ = env.step(actions)
    return obs


def capture_rgb_from_env(env: Any) -> Any:
    import numpy as np

    last = None
    for _ in range(8):
        rgb = env.render()
        last = rgb
        if rgb is not None and getattr(rgb, "ndim", 0) == 3 and int(np.asarray(rgb).size) > 0:
            arr = np.asarray(rgb[:, :, :3], dtype=np.uint8)
            if int(arr.max()) > 0:
                return arr
    if last is None:
        raise RuntimeError("env.render() returned None")
    return np.asarray(last[:, :, :3], dtype=np.uint8)


def frame_metadata(
    *,
    inner_env: Any,
    episode_id: int,
    frame_id: int,
    sim_step: int,
    action3: Any,
    fps: int,
) -> dict[str, Any]:
    inner_env._compute_intermediate_values()
    terms = inner_env._tap_terms()
    cube_xyz = tensor_list(inner_env._sponge_pos_w[0])
    projection = cube_top_projection(cube_xyz)
    return {
        "episode_id": int(episode_id),
        "frame_id": int(frame_id),
        "sim_step": int(sim_step),
        "timestamp_s": float(frame_id) / float(fps),
        "sim_time_s": float(sim_step) * float(inner_env.cfg.episode_length_s) / float(inner_env.max_episode_length),
        "source_png": "",
        "camera_contract_id": CAMERA_CONTRACT_ID,
        "camera_path": CAMERA_PATH,
        "camera_center_world_m": list(CAMERA_CENTER),
        "camera_height_above_table_m": CAMERA_HEIGHT_ABOVE_TABLE_M,
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "image_convention": "image_right_world_pos_x__image_down_world_neg_y",
        "observation_state": tensor_list(inner_env._robot.data.joint_pos[0]),
        "action": tensor_list(inner_env.robot_dof_targets[0]),
        "policy_action3_zero_residual": tensor_list(action3[0]),
        "cube_position_world_m": cube_xyz,
        "cube_quat_wxyz": tensor_list(inner_env._sponge_quat_w[0]),
        "cube_linear_velocity_mps": tensor_list(inner_env._sponge.data.root_lin_vel_w[0]),
        "tcp_position_world_m": tensor_list(inner_env._tcp_pos_w[0]),
        "target_position_world_m": tensor_list(inner_env._target_world[0]),
        "push_dir_xy": tensor_list(inner_env._push_dir_xy[0]),
        "tap_contact_proxy": scalar(terms["tap_contact_proxy"][0]),
        "tap_contact_seen": scalar(inner_env._tap_contact_seen[0]),
        "tap_reaction_seen": scalar(inner_env._tap_reaction_seen[0]),
        "tap_overshoot_seen": scalar(inner_env._tap_overshoot_seen[0]),
        "tap_success_flag": scalar(inner_env._tap_success_flag[0]),
        "tap_disp_along_m": scalar(terms["disp_along"][0]),
        "tap_disp_xy_m": scalar(terms["disp_xy"][0]),
        "tap_speed_mps": scalar(terms["speed"][0]),
        "projection": projection,
    }


def attach_source_pngs_and_visibility(rows: list[dict[str, Any]], frame_paths: list[Path]) -> None:
    import numpy as np
    from PIL import Image

    if len(frame_paths) != len(rows):
        raise RuntimeError(f"frame count mismatch: png={len(frame_paths)} metadata={len(rows)}")
    for row, path in zip(rows, frame_paths, strict=True):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        row["source_png"] = str(path.relative_to(REPO))
        row.update(blue_visibility(rgb, row["projection"]))


def summarize(rows: list[dict[str, Any]], out_dir: Path, elapsed_s: float, poses: list[tuple[float, float]]) -> dict[str, Any]:
    png_bytes = sum((REPO / row["source_png"]).stat().st_size for row in rows)
    episode_ids = sorted({int(row["episode_id"]) for row in rows})
    by_label = {"cube_visible_full": 0, "cube_visible_partial": 0, "cube_occluded_full": 0}
    contact_by_label = {"cube_visible_full": 0, "cube_visible_partial": 0, "cube_occluded_full": 0}
    contact_frames = 0
    centroid_errors = []
    for row in rows:
        by_label[row["cube_visibility"]] += 1
        if row["tap_contact_proxy"] >= 0.5 or row["tap_contact_seen"] >= 0.5:
            contact_frames += 1
            contact_by_label[row["cube_visibility"]] += 1
        err = row["centroid_error_px"]
        if isinstance(err, (int, float)) and math.isfinite(float(err)):
            centroid_errors.append(float(err))

    def pct(count: int, denom: int) -> float:
        return float(count / max(1, denom))

    centroid_errors_sorted = sorted(centroid_errors)
    median_err = (
        centroid_errors_sorted[len(centroid_errors_sorted) // 2]
        if centroid_errors_sorted
        else math.nan
    )
    max_err = max(centroid_errors) if centroid_errors else math.nan
    mb_per_ep = (png_bytes / 1_000_000.0) / max(1, len(episode_ids))
    return {
        "artifact": "cube10cm_top_view_visual_smoke_d232_render",
        "runtime": "ISAAC_RENDER_ONLY_NO_TRAINING_NO_SCALEUP",
        "camera_contract_id": CAMERA_CONTRACT_ID,
        "num_episodes": len(episode_ids),
        "poses_xy_m": [[float(x), float(y)] for x, y in poses],
        "frames": len(rows),
        "resolution": [WIDTH, HEIGHT],
        "target_fps": FPS,
        "elapsed_s": elapsed_s,
        "effective_render_fps": float(len(rows) / max(elapsed_s, 1.0e-9)),
        "png_bytes_total": int(png_bytes),
        "debug_png_mb_per_episode": mb_per_ep,
        "debug_png_projected_gb": {
            "100_ep": mb_per_ep * 100.0 / 1000.0,
            "1000_ep": mb_per_ep * 1000.0 / 1000.0,
            "10000_ep": mb_per_ep * 10000.0 / 1000.0,
        },
        "visibility_counts": by_label,
        "visibility_rates": {key: pct(value, len(rows)) for key, value in by_label.items()},
        "contact_window_frames": contact_frames,
        "contact_visibility_counts": contact_by_label,
        "contact_visibility_rates": {key: pct(value, contact_frames) for key, value in contact_by_label.items()},
        "reprojection_centroid_error_px": {
            "median": median_err,
            "max": max_err,
            "samples": len(centroid_errors),
        },
    }


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    if int(args.num_episodes) < 1 or int(args.num_episodes) > 10:
        raise ValueError("--num-episodes must be in [1, 10] for this smoke script")
    if int(args.capture_stride) < 1:
        raise ValueError("--capture-stride must be >= 1")
    if not args.robot_usd_path.exists():
        raise FileNotFoundError(f"local robot USD missing: {args.robot_usd_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = args.out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    poses = parse_poses(args.poses, int(args.num_episodes))

    print(
        "[cube10cm-top-view-smoke-render] start "
        f"episodes={args.num_episodes} steps={args.steps_per_episode} "
        f"stride={args.capture_stride} out={args.out_dir}",
        flush=True,
    )

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True, device=args.device)
    sim_app = app_launcher.app

    try:
        import gymnasium as gym
        import omni.usd
        import torch
        from PIL import Image

        import roarm_rl  # noqa: F401
        from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
        from roarm_rl.train_cube_tap10cm_ppo_smoke import (
            _apply_candidate6_contract,
            _contract_violations,
        )

        first_args = base_contract_args(args, *poses[0])
        env_cfg = RoArmCubeTap10cmEnvCfg()
        contract = _apply_candidate6_contract(env_cfg, first_args)
        violations = _contract_violations(contract, first_args)
        if violations:
            raise RuntimeError(f"candidate contract violations: {violations}")
        env_cfg.viewer.cam_prim_path = CAMERA_PATH
        env_cfg.viewer.resolution = (int(args.width), int(args.height))

        env = gym.make("RoArm-CubeTap10cm-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
        inner_env = env.unwrapped
        print(
            "[cube10cm-top-view-smoke-render] env_ready "
            f"max_episode_length={inner_env.max_episode_length} action_space={inner_env.cfg.action_space}",
            flush=True,
        )

        stage = omni.usd.get_context().get_stage()
        install_camera(stage, int(args.width), int(args.height))
        raw_frames_dir = args.out_dir / "raw_env_render_frames"
        if raw_frames_dir.exists() and any(raw_frames_dir.glob("*.png")):
            raise FileExistsError(f"{raw_frames_dir} already contains PNG files; choose a new --out-dir")
        raw_frames_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(10):
            sim_app.update()

        rows: list[dict[str, Any]] = []
        overall_t0 = time.time()
        with torch.inference_mode():
            for episode_id, (fixed_x, fixed_y) in enumerate(poses):
                inner_env.cfg.cube_x_min = float(fixed_x)
                inner_env.cfg.cube_x_max = float(fixed_x)
                inner_env.cfg.cube_y_min = float(fixed_y)
                inner_env.cfg.cube_y_max = float(fixed_y)
                obs, _info = env.reset()
                ep_dir = episodes_dir / f"episode_{episode_id:03d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                episode_t0 = time.time()
                frame_id = 0
                action3 = torch.zeros((inner_env.num_envs, int(inner_env.cfg.action_space)), device=inner_env.device)

                rgb = capture_rgb_from_env(env)
                frame_path = raw_frames_dir / f"rgb_{len(rows):06d}.png"
                Image.fromarray(rgb).save(frame_path)
                rows.append(
                    frame_metadata(
                        inner_env=inner_env,
                        episode_id=episode_id,
                        frame_id=frame_id,
                        sim_step=0,
                        action3=action3,
                        fps=int(args.fps),
                    )
                )

                for sim_step in range(1, int(args.steps_per_episode) + 1):
                    action3 = torch.zeros((inner_env.num_envs, int(inner_env.cfg.action_space)), device=inner_env.device)
                    obs, _rewards, _terminated, _truncated, _extras = env.step(action3)
                    if sim_step % int(args.capture_stride) != 0 and sim_step != int(args.steps_per_episode):
                        continue
                    frame_id += 1
                    rgb = capture_rgb_from_env(env)
                    frame_path = raw_frames_dir / f"rgb_{len(rows):06d}.png"
                    Image.fromarray(rgb).save(frame_path)
                    rows.append(
                        frame_metadata(
                            inner_env=inner_env,
                            episode_id=episode_id,
                            frame_id=frame_id,
                            sim_step=sim_step,
                            action3=action3,
                            fps=int(args.fps),
                        )
                    )

                print(
                    "[cube10cm-top-view-smoke-render] episode "
                    f"{episode_id} xy=({fixed_x:.3f},{fixed_y:.3f}) frames={frame_id + 1} "
                    f"elapsed_s={time.time() - episode_t0:.1f}",
                    flush=True,
                )

        elapsed_s = time.time() - overall_t0
        frame_paths = sorted(raw_frames_dir.glob("*.png"))
        attach_source_pngs_and_visibility(rows, frame_paths)
        frames_jsonl = args.out_dir / "frames.jsonl"
        with frames_jsonl.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        summary = summarize(rows, args.out_dir, elapsed_s, poses)
        summary["contract"] = contract
        summary["contract_violations"] = violations
        summary["frames_jsonl"] = str(frames_jsonl.relative_to(REPO))
        summary_path = args.out_dir / "render_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        print(
            "[cube10cm-top-view-smoke-render] done "
            f"frames={summary['frames']} effective_fps={summary['effective_render_fps']:.3f} "
            f"png_mb_per_ep={summary['debug_png_mb_per_episode']:.2f} "
            f"summary={summary_path}",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        if not args.close_sim_app:
            os._exit(0)
        if args.close_sim_app:
            env.close()
    finally:
        if args.close_sim_app:
            sim_app.close()


if __name__ == "__main__":
    main()
