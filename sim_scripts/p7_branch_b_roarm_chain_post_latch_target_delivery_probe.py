#!/usr/bin/env python3
"""Post-latch target-delivery diagnostic for P7 Branch B.

This stays inside the local CLOSE/post-latch executor problem. It instruments
whether joint targets are passed into the RoArm articulation and whether the
same joint nudge is realized before CLOSE versus after CLOSE/latch. It does not
insert constraints, attach SurfaceGripper, execute transport, release, train P7,
or edit env/train/chain defaults.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))

from p7_branch_b_roarm_chain_dynamics_timing_probe import (  # noqa: E402
    HOME_DEG,
    PICK_WRIST_R_DEG,
    SPONGE_CENTER_Z,
    build_command_events,
    fk_tcp,
)
from p7_branch_b_roarm_chain_handoff_micro_motion_probe import (  # noqa: E402
    _fmt_xyz,
    _norm,
    _yes,
)


def _fmt_deg(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in np.asarray(v, dtype=np.float64)) + "]"


def _fmt_rad(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.6f}" for x in np.asarray(v, dtype=np.float64)) + "]"


def _as_np(v) -> np.ndarray:
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy().astype(np.float64)
    return np.asarray(v, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04])
    ap.add_argument("--resample_fraction", type=float, default=0.90)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--home_fk_gate_m", type=float, default=0.003)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--initial_settle_steps", type=int, default=20)
    ap.add_argument("--max_steps_per_event", type=int, default=80)
    ap.add_argument("--delivery_steps", type=int, default=40)
    ap.add_argument("--restore_steps", type=int, default=40)
    ap.add_argument("--joint_nudge_index", type=int, default=1)
    ap.add_argument("--joint_nudge_deg", type=float, default=5.0)
    ap.add_argument("--episode_length_s", type=float, default=10.0)
    ap.add_argument("--log_every_event", type=int, default=8)
    ap.add_argument("--direct_steps", type=int, default=20)
    args = ap.parse_args()

    if args.joint_nudge_index < 0 or args.joint_nudge_index >= 6:
        raise ValueError("joint_nudge_index must be in [0, 5]")

    stream_args = argparse.Namespace(
        sponge_xy=args.sponge_xy,
        place_xyz=[0.280, -0.0435, SPONGE_CENTER_Z],
        resample_fraction=args.resample_fraction,
        max_tcp_step_m=args.max_tcp_step_m,
    )
    events, meta = build_command_events(stream_args)
    close_event = next(event for event in events if event.kind == "CLOSE")
    pre_move_events = [event for event in events if event.kind == "PRE_MOVE"]

    close_q_deg = close_event.q_deg.copy()
    target_q_deg = close_q_deg.copy()
    target_q_deg[args.joint_nudge_index] += args.joint_nudge_deg
    target_q_deg[4] = PICK_WRIST_R_DEG
    target_q_open_deg = target_q_deg.copy()
    target_q_open_deg[5] = 0.0
    target_q_rad_np = np.radians(target_q_deg)
    target_q_open_rad_np = np.radians(target_q_open_deg)
    expected_tcp = fk_tcp(target_q_deg)
    close_expected_tcp = fk_tcp(close_q_deg)
    expected_tcp_delta = _norm(expected_tcp - close_expected_tcp)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    import gymnasium as gym
    import roarm_rl  # noqa: F401 registers env
    import torch
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, SPONGE_CENTER_Z as ENV_SPONGE_CENTER_Z
    from roarm_rl.roarm_stack_env import _quat_rotate

    print("[roarm_chain_target_delivery] post_latch_target_delivery_probe", flush=True)
    print(
        "[roarm_chain_target_delivery] "
        "target_delivery_instrumentation_only=YES constraint_prim_insertion=NO "
        "fixed_dynamic_constraint_integration=NO surface_gripper=NO "
        "surface_gripper_chain_attachment=NO attached_transport=NO transport_target=NO "
        "release_marker=NO p7_training=NO env_default_edits=NO chain_defaults_edits=NO "
        "kinematic_env_latch_only=YES attach_physics_validated=NO "
        "release_physics_validated=NO claim_attach_success=NO",
        flush=True,
    )
    print(
        f"[roarm_chain_target_delivery] gates target_error_gate_m={args.target_error_gate_m:.6f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} home_fk_gate_m={args.home_fk_gate_m:.6f} "
        f"delivery_steps={args.delivery_steps} direct_steps={args.direct_steps} "
        f"joint_nudge_index={args.joint_nudge_index} joint_nudge_deg={args.joint_nudge_deg:.3f} "
        f"resample_fraction={args.resample_fraction:.3f}",
        flush=True,
    )
    print(
        f"[roarm_chain_target_delivery] stream source_events_total={meta['events_total']} "
        f"executed_pre_moves={len(pre_move_events)} close_index={close_event.index} "
        f"move_cmds_executed=0 raw_max_gap_m={meta['raw_max_gap_m']:.6f} "
        f"raw_gap_ok={_yes(meta['raw_max_gap_m'] <= args.max_tcp_step_m)}",
        flush=True,
    )
    print(
        f"[roarm_chain_target_delivery] nudge_plan close_cmd_q_deg={_fmt_deg(close_q_deg)} "
        f"target_q_deg={_fmt_deg(target_q_deg)} delta_q_deg={_fmt_deg(target_q_deg - close_q_deg)} "
        f"target_q_open_deg={_fmt_deg(target_q_open_deg)} "
        f"expected_tcp_delta_m={expected_tcp_delta:.6f} close_expected_tcp={_fmt_xyz(close_expected_tcp)} "
        f"expected_tcp={_fmt_xyz(expected_tcp)}",
        flush=True,
    )

    cfg = RoArmStackEnvCfg()
    cfg.scene.num_envs = 1
    cfg.reward_phase = 6
    cfg.curriculum_pregrasp = False
    cfg.curriculum_pregrasp_hover = False
    cfg.curriculum_attached_transport_release = False
    cfg.curriculum_post_grasp_cap = False
    cfg.curriculum_disable_nearzone_cap = False
    cfg.curriculum_spawn_min_r = 0.0
    cfg.curriculum_spawn_max_r = 0.0
    cfg.episode_length_s = args.episode_length_s
    cfg.attach_quat_mode = "preserve"
    cfg.attach_velocity_mode = "zero"

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    base_env = env.unwrapped
    device = base_env.device
    null_action = torch.zeros((1, 6), device=device, dtype=torch.float32)

    # Marker-only local latch: keep _grasped marker but do not pose-write sponge.
    attach_stats = {"attach_calls": 0}

    def _marker_only_attach() -> None:
        env_ids = torch.where(base_env._grasped)[0]
        if len(env_ids) > 0:
            attach_stats["attach_calls"] += 1

    base_env._update_grasp_attach = _marker_only_attach

    watch = {"active": False, "label": "none", "target": None, "calls": 0, "max_diff": 0.0}
    original_set_joint_position_target = base_env._robot.set_joint_position_target

    def _wrapped_set_joint_position_target(target, *set_args, **set_kwargs):
        if watch["active"] and watch["target"] is not None:
            arr = _as_np(target)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            tgt = watch["target"]
            diff = float(np.max(np.abs(arr[0, : tgt.shape[0]] - tgt)))
            watch["calls"] += 1
            watch["max_diff"] = max(watch["max_diff"], diff)
            if watch["calls"] <= 3:
                print(
                    f"[roarm_chain_target_delivery] set_joint_position_target_call "
                    f"label={watch['label']} call={watch['calls']} max_diff_to_watch_rad={diff:.8f} "
                    f"target_rad={_fmt_rad(arr[0])}",
                    flush=True,
                )
        return original_set_joint_position_target(target, *set_args, **set_kwargs)

    base_env._robot.set_joint_position_target = _wrapped_set_joint_position_target

    env.reset()

    home_rad = torch.tensor(np.radians(HOME_DEG), device=device, dtype=torch.float32).unsqueeze(0)
    base_env._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
    base_env._robot.set_joint_position_target(home_rad)
    base_env.robot_dof_targets[:] = home_rad

    sponge_pose = torch.tensor(
        [[args.sponge_xy[0], args.sponge_xy[1], ENV_SPONGE_CENTER_Z, 1.0, 0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    sponge_pose[:, 0:3] += base_env.scene.env_origins[:1]
    base_env._sponge.write_root_pose_to_sim(sponge_pose)
    base_env._sponge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    base_env._grasped[:] = False
    base_env._was_grasped[:] = False

    total_sim_steps = 0
    episode_done = False
    nan_seen = False
    latch_seen = False
    latch_global_step = -1

    def step_once() -> bool:
        out = env.step(null_action)
        if len(out) == 5:
            _obs, _rew, terminated, truncated, _extras = out
            return bool((terminated | truncated).any().item())
        _obs, _rew, dones, _extras = out
        return bool(dones.any().item())

    def direct_step_once() -> None:
        base_env.scene.write_data_to_sim()
        try:
            base_env.sim.step(render=False)
        except TypeError:
            base_env.sim.step()
        base_env.scene.update(base_env.sim.get_physics_dt())

    def fresh_tcp_local() -> np.ndarray:
        link5_pos = base_env._robot.data.body_pos_w[:1, base_env.link5_idx]
        link5_quat = base_env._robot.data.body_quat_w[:1, base_env.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, base_env._tcp_local.expand(1, 3))
        tcp = link5_pos + tcp_offset_world
        return (tcp[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def current_q_rad() -> np.ndarray:
        return base_env._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)

    def targets_rad() -> np.ndarray:
        return base_env.robot_dof_targets[0].detach().cpu().numpy().astype(np.float64)

    def data_target_snapshot(label: str, target_rad_np: np.ndarray) -> tuple[str, float]:
        names: list[str] = []
        best = float("inf")
        for name in dir(base_env._robot.data):
            lname = name.lower()
            if "target" not in lname:
                continue
            try:
                value = getattr(base_env._robot.data, name)
            except Exception:
                continue
            if not hasattr(value, "shape"):
                continue
            arr = _as_np(value)
            if arr.size < target_rad_np.size:
                continue
            names.append(name)
            flat = arr.reshape(-1, arr.shape[-1])
            if flat.shape[-1] >= target_rad_np.size:
                best = min(best, float(np.max(np.abs(flat[0, : target_rad_np.size] - target_rad_np))))
        best_text = "nan" if not math.isfinite(best) else f"{best:.8f}"
        print(
            f"[roarm_chain_target_delivery] data_target_snapshot label={label} "
            f"target_attrs={names} best_attr_diff_rad={best_text}",
            flush=True,
        )
        return ",".join(names), best

    for _ in range(args.initial_settle_steps):
        step_once()
        total_sim_steps += 1

    home_tcp = fresh_tcp_local()
    home_fk_error = _norm(home_tcp - fk_tcp(HOME_DEG))
    print(
        f"[roarm_chain_target_delivery] initial home_fresh_tcp={_fmt_xyz(home_tcp)} "
        f"home_expected_tcp={_fmt_xyz(fk_tcp(HOME_DEG))} home_fk_error_m={home_fk_error:.6f} "
        f"home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)}",
        flush=True,
    )

    def execute_event(event) -> tuple[bool, int, float]:
        nonlocal total_sim_steps, episode_done, latch_seen, latch_global_step, nan_seen
        target_q = torch.tensor(np.radians(event.q_deg), device=device, dtype=torch.float32).unsqueeze(0)
        settle_count = 0
        final_error = float("inf")
        steps_used = 0
        for step_idx in range(1, args.max_steps_per_event + 1):
            base_env.robot_dof_targets[:] = target_q
            done = step_once()
            total_sim_steps += 1
            episode_done |= done
            steps_used = step_idx
            tcp = fresh_tcp_local()
            final_error = _norm(tcp - event.target_tcp)
            if not np.isfinite(tcp).all() or not math.isfinite(final_error):
                nan_seen = True
            gripper_q = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].detach().cpu().item())
            if bool(base_env._grasped[0].detach().cpu().item()) and not latch_seen:
                latch_seen = True
                latch_global_step = total_sim_steps
            reached = final_error <= args.target_error_gate_m
            if event.kind == "CLOSE":
                reached = reached and gripper_q >= base_env.cfg.grasp_gripper_thresh
            settle_count = settle_count + 1 if reached else 0
            if settle_count >= args.settle_steps or (event.kind == "CLOSE" and latch_seen):
                break
        return final_error <= args.target_error_gate_m, steps_used, final_error

    for event in pre_move_events:
        reached, steps, final_error = execute_event(event)
        if event.index <= 5 or event.index % args.log_every_event == 0 or not reached:
            print(
                f"[roarm_chain_target_delivery] event_index={event.index:03d} event={event.kind} "
                f"phase={event.phase} segment={event.segment} steps={steps} "
                f"final_target_error_m={final_error:.6f} reached={_yes(reached)} "
                f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))}",
                flush=True,
            )

    close_target_rad = torch.tensor(np.radians(close_q_deg), device=device, dtype=torch.float32).unsqueeze(0)
    target_q_rad = torch.tensor(target_q_rad_np, device=device, dtype=torch.float32).unsqueeze(0)
    target_q_open_rad = torch.tensor(target_q_open_rad_np, device=device, dtype=torch.float32).unsqueeze(0)

    def run_delivery(label: str, mode: str, target_rad, target_rad_np_for_metrics: np.ndarray, steps: int):
        nonlocal total_sim_steps, episode_done, nan_seen
        start_tcp = fresh_tcp_local()
        start_q = current_q_rad()
        expected_tcp_for_target = fk_tcp(np.degrees(target_rad_np_for_metrics))
        expected_motion_from_start = _norm(expected_tcp_for_target - start_tcp)
        watch["active"] = True
        watch["label"] = label
        watch["target"] = target_rad_np_for_metrics
        watch["calls"] = 0
        watch["max_diff"] = 0.0
        max_realized_tcp_delta = 0.0
        max_step_tcp_delta = 0.0
        min_joint_error_max_deg = float("inf")
        final_target_tcp_error = float("inf")
        final_joint_error_max_deg = float("inf")
        prev_tcp = start_tcp
        best_attr_diff = float("inf")
        for step_idx in range(1, steps + 1):
            if mode == "env_step":
                base_env.robot_dof_targets[:] = target_rad
                done = step_once()
                total_sim_steps += 1
                episode_done |= done
            elif mode == "direct":
                base_env._robot.set_joint_position_target(target_rad)
                direct_step_once()
                total_sim_steps += 1
                done = False
            else:
                raise ValueError(f"unknown delivery mode {mode}")
            after_tcp = fresh_tcp_local()
            after_q = current_q_rad()
            step_tcp_delta = _norm(after_tcp - prev_tcp)
            realized_tcp_delta = _norm(after_tcp - start_tcp)
            target_tcp_error = _norm(after_tcp - expected_tcp_for_target)
            joint_error_deg = np.degrees(target_rad_np_for_metrics - after_q)
            joint_error_max_deg = float(np.max(np.abs(joint_error_deg)))
            max_realized_tcp_delta = max(max_realized_tcp_delta, realized_tcp_delta)
            max_step_tcp_delta = max(max_step_tcp_delta, step_tcp_delta)
            min_joint_error_max_deg = min(min_joint_error_max_deg, joint_error_max_deg)
            final_target_tcp_error = target_tcp_error
            final_joint_error_max_deg = joint_error_max_deg
            if step_idx in (1, 2, 3, steps):
                _attrs, diff = data_target_snapshot(f"{label}_step{step_idx:03d}", target_rad_np_for_metrics)
                best_attr_diff = min(best_attr_diff, diff)
                print(
                    f"[roarm_chain_target_delivery] delivery_step label={label} mode={mode} "
                    f"step={step_idx:03d} set_calls={watch['calls']} set_max_diff_rad={watch['max_diff']:.8f} "
                    f"robot_dof_target_diff_rad={float(np.max(np.abs(targets_rad() - target_rad_np_for_metrics))):.8f} "
                    f"current_q_deg={_fmt_deg(np.degrees(after_q))} "
                    f"joint_error_max_deg={joint_error_max_deg:.6f} "
                    f"fresh_tcp={_fmt_xyz(after_tcp)} expected_tcp={_fmt_xyz(expected_tcp_for_target)} "
                    f"target_tcp_error_m={target_tcp_error:.6f} "
                    f"realized_tcp_delta_m={realized_tcp_delta:.6f} step_tcp_delta_m={step_tcp_delta:.6f} "
                    f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))} done={_yes(done)}",
                    flush=True,
                )
            if not np.isfinite(after_tcp).all() or not math.isfinite(target_tcp_error):
                nan_seen = True
            prev_tcp = after_tcp
        watch["active"] = False
        joint_error_reduced = final_joint_error_max_deg < max(1.0, 0.5 * args.joint_nudge_deg)
        tcp_target_reduced = final_target_tcp_error < max(args.target_error_gate_m, 0.75 * expected_motion_from_start)
        target_realized = joint_error_reduced and tcp_target_reduced
        incidental_tcp_motion = max_realized_tcp_delta > 0.0005
        print(
            f"[roarm_chain_target_delivery] delivery_result label={label} mode={mode} "
            f"steps={steps} start_q_deg={_fmt_deg(np.degrees(start_q))} "
            f"target_q_deg={_fmt_deg(np.degrees(target_rad_np_for_metrics))} "
            f"expected_motion_from_start_m={expected_motion_from_start:.6f} "
            f"set_calls={watch['calls']} set_target_seen={_yes(watch['calls'] > 0 and watch['max_diff'] <= 1.0e-5)} "
            f"set_max_diff_rad={watch['max_diff']:.8f} "
            f"robot_dof_target_diff_rad={float(np.max(np.abs(targets_rad() - target_rad_np_for_metrics))):.8f} "
            f"best_data_target_attr_diff_rad={'nan' if not math.isfinite(best_attr_diff) else f'{best_attr_diff:.8f}'} "
            f"max_realized_tcp_delta_m={max_realized_tcp_delta:.6f} max_step_tcp_delta_m={max_step_tcp_delta:.6f} "
            f"final_target_tcp_error_m={final_target_tcp_error:.6f} "
            f"min_joint_error_max_deg={min_joint_error_max_deg:.6f} "
            f"final_joint_error_max_deg={final_joint_error_max_deg:.6f} "
            f"incidental_tcp_motion_seen={_yes(incidental_tcp_motion)} "
            f"tcp_target_reduced={_yes(tcp_target_reduced)} "
            f"joint_error_reduced={_yes(joint_error_reduced)} target_realized={_yes(target_realized)} "
            f"grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))}",
            flush=True,
        )
        return {
            "set_seen": watch["calls"] > 0 and watch["max_diff"] <= 1.0e-5,
            "target_realized": target_realized,
            "incidental_tcp_motion": incidental_tcp_motion,
            "joint_error_reduced": joint_error_reduced,
            "tcp_target_reduced": tcp_target_reduced,
            "max_realized_tcp_delta": max_realized_tcp_delta,
            "min_joint_error_max_deg": min_joint_error_max_deg,
        }

    print("[roarm_chain_target_delivery] compare_stage=before_close_open", flush=True)
    before_result = run_delivery(
        "before_close_open_joint_nudge",
        "env_step",
        target_q_open_rad,
        target_q_open_rad_np,
        args.delivery_steps,
    )

    print("[roarm_chain_target_delivery] restore_stage=before_close_open_to_close_pose", flush=True)
    _restore_before = run_delivery(
        "restore_before_close_pose",
        "env_step",
        close_target_rad,
        np.radians(close_q_deg),
        args.restore_steps,
    )

    reached, close_steps, close_error = execute_event(close_event)
    print(
        f"[roarm_chain_target_delivery] event_index={close_event.index:03d} event=CLOSE phase=CLOSE "
        f"segment={close_event.segment} steps={close_steps} final_target_error_m={close_error:.6f} "
        f"reached={_yes(reached)} grasped={_yes(bool(base_env._grasped[0].detach().cpu().item()))} "
        f"latch_global_step={latch_global_step}",
        flush=True,
    )

    print("[roarm_chain_target_delivery] compare_stage=after_close_latched_env_step", flush=True)
    after_env_result = run_delivery(
        "after_close_latched_joint_nudge",
        "env_step",
        target_q_rad,
        target_q_rad_np,
        args.delivery_steps,
    )

    print("[roarm_chain_target_delivery] compare_stage=after_close_latched_direct_set", flush=True)
    after_direct_result = run_delivery(
        "after_close_latched_direct_joint_nudge",
        "direct",
        target_q_rad,
        target_q_rad_np,
        args.direct_steps,
    )

    before_vs_after_split = before_result["target_realized"] and not after_env_result["target_realized"]
    direct_rescues = (not after_env_result["target_realized"]) and after_direct_result["target_realized"]
    preclose_blocker = before_result["set_seen"] and not before_result["target_realized"]
    post_latch_blocker = after_env_result["set_seen"] and not after_env_result["target_realized"]
    general_grasp_pose_blocker = preclose_blocker and post_latch_blocker
    success = (
        home_fk_error <= args.home_fk_gate_m
        and before_result["set_seen"]
        and after_env_result["set_seen"]
        and post_latch_blocker
        and not nan_seen
        and not episode_done
    )
    print(
        f"[roarm_chain_target_delivery] aggregate total_sim_steps={total_sim_steps} "
        f"latch_seen={_yes(latch_seen)} latch_global_step={latch_global_step} "
        f"before_target_realized={_yes(before_result['target_realized'])} "
        f"after_env_target_realized={_yes(after_env_result['target_realized'])} "
        f"after_direct_target_realized={_yes(after_direct_result['target_realized'])} "
        f"preclose_target_delivery_blocker={_yes(preclose_blocker)} "
        f"before_vs_after_split={_yes(before_vs_after_split)} direct_rescues={_yes(direct_rescues)} "
        f"post_latch_target_delivery_blocker={_yes(post_latch_blocker)} "
        f"general_grasp_pose_target_delivery_blocker={_yes(general_grasp_pose_blocker)} "
        f"attach_calls={attach_stats['attach_calls']} action_scale={base_env.cfg.action_scale:.6f} "
        f"null_action_max_abs={float(torch.max(torch.abs(null_action)).item()):.6f}",
        flush=True,
    )
    print(
        f"[roarm_chain_target_delivery] gates home_fk_ok={_yes(home_fk_error <= args.home_fk_gate_m)} "
        f"before_set_seen={_yes(before_result['set_seen'])} "
        f"after_set_seen={_yes(after_env_result['set_seen'])} "
        f"post_latch_blocker_seen={_yes(post_latch_blocker)} "
        f"attach_physics_validated=NO release_physics_validated=NO claim_attach_success=NO "
        f"nan_seen={_yes(nan_seen)} episode_done={_yes(episode_done)}",
        flush=True,
    )
    print(f"[roarm_chain_target_delivery] ROARM_POST_LATCH_TARGET_DELIVERY_SUCCESS={_yes(success)}", flush=True)

    env.close()
    sim_app.close()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
