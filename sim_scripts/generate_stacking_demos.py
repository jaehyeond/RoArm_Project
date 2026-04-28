"""Stacking demo generator (Phase ST-A).

Generates N procedural demonstrations of the 3-step N=2 stacking task:
  Step 1: A_top  → Temp     (disassemble: lift top sponge to buffer)
  Step 2: A_bot  → B        (move bottom to dest)
  Step 3: Temp   → B_top    (stack top onto B)

Each demo: 24 anchor poses × ~5 frames interpolation = ~110 frames @ 30fps.
Output: per-demo joint trajectory CSV + summary stats.

Design notes:
  - Position-only IK with v6-warm-start → lateral grasp (matches v6 distribution).
  - Linear joint-space interpolation between anchors (cubic OK for ST-B).
  - Gripper segment: in-place open/close (3 frames each).
  - Safety clip: TCP z ≤ +260mm (training), enforced at anchor design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roarm_kinematics import fk_tcp, ik_dls, V6WarmStart, JOINT_LIMITS_DEG, clip_joints


# ===============================================================
# Constants — match stacking_scene.py & 4/24 stacking pivot.
# ===============================================================
LAYOUT = {
    "A":    (+0.280,  0.000),
    "B":    (+0.280, +0.130),
    "Temp": (+0.280, -0.110),
}
TABLE_Z = -0.012117  # URDF world m
SPONGE_H = 0.125     # sponge upright height (z dim)
STACK_GAP = 0.002

# Sponge top-face heights (URDF world Z, m).
Z_SINGLE_TOP = TABLE_Z + SPONGE_H            # top of single sponge on table = +0.1129
Z_STACK_TOP  = TABLE_Z + 2 * SPONGE_H + STACK_GAP  # top of 2-stack          = +0.2400

# Approach offsets.
Z_APPROACH = 0.040      # 40mm hover above grasp/release height
Z_TRANSIT  = 0.060      # 60mm above for xy transit (above max)
Z_TRANSIT_MAX = TABLE_Z + 2 * SPONGE_H + STACK_GAP + Z_TRANSIT  # +0.300m

SAFETY_Z_MAX_TRAIN = +0.260  # 5a
SAFETY_Z_MAX_DEPLOY = +0.280  # 5a hard

# Gripper commands (deg). v6 final frame: open ~80, close ~10.
G_OPEN = 60.0
G_CLOSE = 10.0
G_PRECLOSE = 30.0  # approach posture

HOME = np.array([0.0, 0.0, 90.0, 0.0, 0.0, G_PRECLOSE])

# Per-segment frame counts (target ~110 frames per demo).
FRAMES_PER_SEG = {
    "approach_xy":   3,  # HOME → above_src
    "descend_grasp": 4,  # above → at (slow descent)
    "close":         3,  # in-place gripper close
    "lift_grasp":    4,  # at → above (lift with grip)
    "transit_xy":    5,  # above_src → above_dst (xy move at high z)
    "descend_place": 4,  # above → at_dst (slow descent)
    "open":          3,  # in-place gripper open
    "lift_release":  3,  # at_dst → above
}


def grasp_z_for(level: str) -> float:
    """Grasp height (TCP z, world m) — gripper grabs sponge mid-side at half-height down from top.
    For lateral grip, TCP is approx at sponge top edge minus ~30mm (mid-finger).
    """
    if level == "single":
        return Z_SINGLE_TOP - 0.030    # +0.083m  (single sponge mid-side grasp)
    elif level == "stack_top":
        return Z_STACK_TOP - 0.030     # +0.210m  (top of stack, mid)
    else:
        raise ValueError(level)


def place_z_for(level: str) -> float:
    """Release height: where TCP lets go after positioning."""
    if level == "floor":
        return Z_SINGLE_TOP - 0.030    # placing single sponge down
    elif level == "on_top":
        return Z_STACK_TOP - 0.030     # placing on top of bottom sponge
    else:
        raise ValueError(level)


def build_anchors(rng, layout):
    """Build the 24-anchor sequence for one stacking demo.
    Randomization: layout xy ± dxy.

    Returns: list of (tag, target_xyz_world, gripper_cmd) tuples.
    """
    dxy = 0.010  # 10mm
    A    = (layout["A"][0]    + rng.uniform(-dxy, dxy), layout["A"][1]    + rng.uniform(-dxy, dxy))
    B    = (layout["B"][0]    + rng.uniform(-dxy, dxy), layout["B"][1]    + rng.uniform(-dxy, dxy))
    Temp = (layout["Temp"][0] + rng.uniform(-dxy, dxy), layout["Temp"][1] + rng.uniform(-dxy, dxy))

    z_grab_top = grasp_z_for("stack_top")
    z_grab_bot = grasp_z_for("single")
    z_place_floor = place_z_for("floor")
    z_place_top   = place_z_for("on_top")

    z_above_top  = z_grab_top + Z_APPROACH
    z_above_bot  = z_grab_bot + Z_APPROACH
    z_above_floor = z_place_floor + Z_APPROACH
    z_above_ontop = z_place_top + Z_APPROACH
    z_transit_top = min(Z_TRANSIT_MAX, SAFETY_Z_MAX_TRAIN)

    # Each step = 8 anchors.
    def step(src_xy, src_z_at, src_z_above, dst_xy, dst_z_at, dst_z_above, tag):
        return [
            (f"{tag}.above_src",   (src_xy[0], src_xy[1], src_z_above), G_OPEN),
            (f"{tag}.at_src",      (src_xy[0], src_xy[1], src_z_at),    G_OPEN),
            (f"{tag}.close",       (src_xy[0], src_xy[1], src_z_at),    G_CLOSE),
            (f"{tag}.lift",        (src_xy[0], src_xy[1], z_transit_top), G_CLOSE),
            (f"{tag}.transit",     (dst_xy[0], dst_xy[1], z_transit_top), G_CLOSE),
            (f"{tag}.at_dst",      (dst_xy[0], dst_xy[1], dst_z_at),     G_CLOSE),
            (f"{tag}.open",        (dst_xy[0], dst_xy[1], dst_z_at),     G_OPEN),
            (f"{tag}.lift_off",    (dst_xy[0], dst_xy[1], dst_z_above),  G_OPEN),
        ]

    anchors = []
    # Step 1: A_top → Temp (disassemble)
    anchors += step(A,    z_grab_top, z_above_top,  Temp, z_place_floor, z_above_floor, "S1_A2T")
    # Step 2: A_bot → B
    anchors += step(A,    z_grab_bot, z_above_bot,  B,    z_place_floor, z_above_floor, "S2_A2B")
    # Step 3: Temp → B_top (stack)
    anchors += step(Temp, z_grab_bot, z_above_bot,  B,    z_place_top,   z_above_ontop, "S3_T2B")

    return anchors, {"A": A, "B": B, "Temp": Temp}


def solve_anchors(anchors, ws):
    """Solve IK for each anchor, return (solved_states (N,6), tcp_actual (N,3), errs)."""
    states = []
    tcps = []
    errs = []
    fails = []
    q_prev = HOME.copy()  # keep last solved state for continuity warm-start
    for tag, target_xyz, g_cmd in anchors:
        target = np.asarray(target_xyz)
        # Warm-start strategy: prefer continuity (q_prev) if its TCP is close, else v6-nearest
        tcp_prev = fk_tcp(q_prev)
        d_prev = np.linalg.norm(tcp_prev - target)
        if d_prev < 0.150:  # within 15cm: trust continuity
            q0 = q_prev.copy()
        else:
            q0_v6, _ = ws.query(target)
            q0 = q0_v6.copy()
        q, conv, err_mm, _ = ik_dls(target, q0, max_iter=300, tol_mm=1.0, damping=2.0)
        # Override gripper command (joint 5)
        q[5] = g_cmd
        q = clip_joints(q)
        states.append(q)
        tcps.append(fk_tcp(q))
        errs.append(err_mm)
        if not conv:
            fails.append((tag, err_mm))
        q_prev = q.copy()
    return np.array(states), np.array(tcps), np.array(errs), fails


def interpolate_trajectory(anchor_states, frames_per_seg):
    """Linear joint-space interpolation between consecutive anchors.
    anchor_states: (24, 6) anchor joint configs.
    frames_per_seg: list of 24 ints (frames *between* anchor i and i+1, last unused).

    Returns: (T, 6) full joint trajectory.
    """
    traj = []
    n = len(anchor_states)
    for i in range(n - 1):
        n_frames = frames_per_seg[i]
        for k in range(n_frames):
            alpha = k / n_frames
            q = (1.0 - alpha) * anchor_states[i] + alpha * anchor_states[i + 1]
            traj.append(q)
    traj.append(anchor_states[-1])
    return np.array(traj)


def get_seg_frames(anchors):
    """Build per-segment frame count list from anchor tags."""
    counts = []
    for i in range(len(anchors) - 1):
        cur = anchors[i][0]
        nxt = anchors[i + 1][0]
        # Tag format: "S{n}_{tag}.{seg}". Determine segment by next anchor's seg name.
        seg_dst = nxt.split(".")[1]
        # Map by destination segment of the current move
        if seg_dst == "above_src":
            counts.append(FRAMES_PER_SEG["approach_xy"])  # transit between steps
        elif seg_dst == "at_src":
            counts.append(FRAMES_PER_SEG["descend_grasp"])
        elif seg_dst == "close":
            counts.append(FRAMES_PER_SEG["close"])
        elif seg_dst == "lift":
            counts.append(FRAMES_PER_SEG["lift_grasp"])
        elif seg_dst == "transit":
            counts.append(FRAMES_PER_SEG["transit_xy"])
        elif seg_dst == "at_dst":
            counts.append(FRAMES_PER_SEG["descend_place"])
        elif seg_dst == "open":
            counts.append(FRAMES_PER_SEG["open"])
        elif seg_dst == "lift_off":
            counts.append(FRAMES_PER_SEG["lift_release"])
        else:
            counts.append(3)
    return counts


def generate_one_demo(seed, ws):
    """Generate a single demo. Returns dict with traj, states, anchors."""
    rng = np.random.default_rng(seed)
    anchors, layout_used = build_anchors(rng, LAYOUT)
    # Prepend HOME entry, append HOME exit
    anchors_full = [("HOME_start", tuple(fk_tcp(HOME)), G_PRECLOSE)] + anchors + \
                   [("HOME_end", tuple(fk_tcp(HOME)), G_PRECLOSE)]
    states, tcps, errs, fails = solve_anchors(anchors_full, ws)
    # Frame counts: HOME→first anchor (5), inter-step transitions, last→HOME (5)
    seg_frames = [5]  # HOME → S1.above_src
    seg_frames += get_seg_frames(anchors_full[1:-1])  # within main 24
    seg_frames += [5]  # last_anchor → HOME_end
    # interpolate_trajectory expects len(seg_frames) == len(states)-1
    assert len(seg_frames) == len(states) - 1, f"{len(seg_frames)} vs {len(states)-1}"
    traj = interpolate_trajectory(states, seg_frames)

    return {
        "seed": seed,
        "layout_used": layout_used,
        "anchors_tags": [a[0] for a in anchors_full],
        "anchor_states": states,
        "anchor_tcps": tcps,
        "anchor_errs_mm": errs,
        "ik_fails": fails,
        "trajectory": traj,
    }


def summarize_demo(demo):
    traj = demo["trajectory"]
    print(f"  Demo seed={demo['seed']}: T={len(traj)} frames")
    print(f"    IK errors (anchor): mean={demo['anchor_errs_mm'].mean():.2f}mm  "
          f"max={demo['anchor_errs_mm'].max():.2f}mm  fails={len(demo['ik_fails'])}")
    if demo["ik_fails"]:
        for tag, e in demo["ik_fails"][:3]:
            print(f"      FAIL {tag}: err={e:.1f}mm")
    print(f"    State stats:")
    names = ["base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper"]
    for i, n in enumerate(names):
        j = traj[:, i]
        lo, hi = JOINT_LIMITS_DEG[n]
        clip_count = int(((j <= lo + 0.01) | (j >= hi - 0.01)).sum())
        print(f"      {n:10s}: [{j.min():+6.1f}, {j.max():+6.1f}] mean={j.mean():+5.1f}  "
              f"clipped={clip_count}")
    # TCP z range (all frames)
    tcps = np.array([fk_tcp(q) for q in traj])
    z = tcps[:, 2]
    over_train = (z > SAFETY_Z_MAX_TRAIN).sum()
    over_deploy = (z > SAFETY_Z_MAX_DEPLOY).sum()
    print(f"    TCP z range: [{z.min()*1000:+.1f}, {z.max()*1000:+.1f}]mm   "
          f"frames > +260mm (train): {over_train}    > +280mm (deploy): {over_deploy}")
    print(f"    Layout used:")
    for k, v in demo["layout_used"].items():
        print(f"      {k}: ({v[0]*1000:+.1f}, {v[1]*1000:+.1f})mm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="seeds to generate (default: just 1 demo with seed 42 for validation)")
    p.add_argument("--out-dir", type=str, default="sim_demos_v1")
    p.add_argument("--dry-run", action="store_true",
                   help="generate trajectories but don't write files")
    args = p.parse_args()

    print(f"Loading v6 warm-start index ...")
    ws = V6WarmStart()
    print(f"Generating {len(args.seeds)} demos: seeds={args.seeds}")

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for s in args.seeds:
        demo = generate_one_demo(s, ws)
        summarize_demo(demo)
        summary.append({
            "seed": s,
            "n_frames": len(demo["trajectory"]),
            "ik_max_err_mm": float(demo["anchor_errs_mm"].max()),
            "ik_fails": len(demo["ik_fails"]),
        })
        if not args.dry_run:
            np.savetxt(out_dir / f"demo_{s:04d}_trajectory.csv",
                       demo["trajectory"], delimiter=",",
                       header="base,shoulder,elbow,wrist_p,wrist_r,gripper",
                       comments="")
            np.savetxt(out_dir / f"demo_{s:04d}_anchors.csv",
                       demo["anchor_states"], delimiter=",",
                       header="base,shoulder,elbow,wrist_p,wrist_r,gripper",
                       comments="")
    if not args.dry_run:
        with open(out_dir / "summary.json", "w") as f:
            json.dump({"demos": summary}, f, indent=2)
    print(f"\nDone. {'(dry-run)' if args.dry_run else f'Wrote {out_dir}/'}")


if __name__ == "__main__":
    main()
