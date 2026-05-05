"""generate_stacking_demos_v3.py — N=4 well-pattern (#) EDGE-STAND stacking.

V3 = edge-stand sponge orientation (47mm vertical), confirmed 5/03 evening
(HARD RULE #19, #20). V2 lying-flat 가정 폐기.

V3 corrections vs V2:
  - Sponge orientation: EDGE-STAND (47mm tall, 22mm wide on table, 125mm long).
    Footprint per sponge = 125 × 22 mm (was 125 × 47 lying-flat).
  - Z layer heights:
      Z_LAYER1_TOP = TABLE_Z + 0.047 = +0.0349 m world (was +0.010 lying-flat)
      Z_LAYER2_TOP = TABLE_Z + 0.094 = +0.0819 m world
  - TCP grasp z = +0.033 m world (was +0.010). v6 measured grasp z mean=+36.8mm,
    median=+35.9mm, range=[+26.1, +49.4]mm — +33mm sits at v6 ~30th percentile,
    safely in-distribution.
  - TCP place L1 z = +0.033 m world (same height as grasp; sponge sits on table)
  - TCP place L2 z = +0.080 m world (= +33+47, sponge sits on L1 top, same offset)
  - Z_TRANSIT = +0.150 m world (HARD CONFIRMED 5/03 evening). v6 distribution:
    p50=165.8mm, 53.8% of v6 frames > 150mm — well in-distribution. Held sponge
    bottom at transit = +150 - 45 = +105mm world (70mm clearance over L1 top +35mm).
  - HASH1_CENTER = (+0.280, +0.000) m world (was -0.100). Symmetric layout.
  - DY_L1 = +0.0435 (c2c 87mm, inner gap 65mm — 사용자 측정 60-70mm range).
  - DX_L2 = +0.0335 (c2c 67mm, inner gap 45mm — 사용자 측정 40-50mm range).
  - WRIST_P_MIN_TOPDOWN = +75° (was +80°). v6 grasp wrist_p mean=+68.8°,
    range=[+41.7°, +88.8°] — +75° design value with IK margin.
  - SPONGE_WIDTH = 0.022 m (new const, edge-stand width on table).
  - Output dir: sim_demos_v3/

Layout (deterministic destinations, m world):
  HASH1 center = (+0.280, +0.000)
  L1 sp1 = (+0.280, -0.0435)  X-axis edge-stand, Y offset -43.5mm
  L1 sp2 = (+0.280, +0.0435)  X-axis edge-stand, Y offset +43.5mm
  L2 sp3 = (+0.2465, +0.000)  Y-axis edge-stand, X offset -33.5mm (on L1 top)
  L2 sp4 = (+0.3135, +0.000)  Y-axis edge-stand, X offset +33.5mm (on L1 top)
  L1 wrist_r = 0°   (length=X, fingers along Y, close on 22mm Y-width)
  L2 wrist_r = +90° (length=Y, fingers along X, close on 22mm X-width)

Source regions (per seed random uniform; reject body-overlap with # area + min-dist):
  R1 좌하: X∈[+0.150,+0.250], Y∈[-0.220,-0.130]
  R2 좌상: X∈[+0.150,+0.250], Y∈[+0.070,+0.200]
  R3 우하: X∈[+0.330,+0.430], Y∈[-0.220,-0.100]
  R4 우상: X∈[+0.330,+0.430], Y∈[+0.050,+0.200]
  Exclusion (#1 build area + 5mm margin):
    X∈[+0.2125,+0.3475], Y∈[-0.0675,+0.0675]

Anchor 4-tuple: (tag, target_xyz_world, gripper_cmd, wrist_r_deg)
  HOME_start (closed) → S1 (8 anchors) → S2 (8) → S3 (8) → S4 (8) → HOME_end (closed)
  Per step (8 anchors): above_src(open) → at_src(open) → close → lift(transit_z) →
                        transit(rotate wrist_r) → at_dst → open → lift_off
  src_wrist_r = source orientation (random); dst_wrist_r = layer (L1=0°, L2=+90°).

Frame count: 146 per demo (constant; matches v2 baseline).

Outputs per seed:
  sim_demos_v3/demo_{seed:04d}_trajectory.csv  (146 × 6 deg)
  sim_demos_v3/demo_{seed:04d}_anchors.csv     (34 × 6 deg)
  sim_demos_v3/demo_{seed:04d}_layout.json     (sources, orients, dst, anchor_frames)

Run:
  conda run -n roarm python sim_scripts/generate_stacking_demos_v3.py --seeds 0 --dry-run
  conda run -n roarm python sim_scripts/generate_stacking_demos_v3.py --seeds 0 1 2 ... 49
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roarm_kinematics
from roarm_kinematics import fk_tcp, ik_dls, V6WarmStart, JOINT_LIMITS_DEG, clip_joints


# ===============================================================
# Heights (m world)
# ===============================================================
TABLE_Z = -0.012117                   # 4/24 calib RMSE 1.24 mm
SPONGE_HEIGHT_EDGE = 0.047            # edge-stand vertical (was 0.022 lying-flat in v2)
SPONGE_LEN_LONG = 0.125               # length on table (along major axis)
SPONGE_WIDTH = 0.022                  # width on table (transverse, gripper closes on this)

Z_LAYER1_TOP = TABLE_Z + SPONGE_HEIGHT_EDGE             # +0.0349 m
Z_LAYER2_TOP = TABLE_Z + 2 * SPONGE_HEIGHT_EDGE         # +0.0819 m

# TCP grasp/place heights (per v3 design + v6 mean +36.8mm match)
Z_TCP_GRASP_L1 = 0.033                                   # +33 mm world
Z_TCP_PLACE_L1 = 0.033                                   # same as grasp (sponge on table)
Z_TCP_PLACE_L2 = 0.080                                   # +33+47 mm world (sponge on L1)

Z_APPROACH = 0.040                                       # hover above grasp/place
Z_TRANSIT = 0.150                                        # USER CONFIRMED. v6 p50=166mm; 53.8% v6 > 150mm

SAFETY_Z_MAX_TRAIN = 0.155                               # transit + 5mm tolerance
SAFETY_Z_MAX_DEPLOY = 0.180                              # hard ceiling


# ===============================================================
# Layout (m world) — per project_well_pattern_design_v3.md
# ===============================================================
HASH1_CENTER = (+0.280, +0.000)

DY_L1 = +0.0435   # c2c 87mm  = 22mm width + 65mm inner gap  (사용자 측정 60-70mm range)
DX_L2 = +0.0335   # c2c 67mm  = 22mm width + 45mm inner gap  (사용자 측정 40-50mm range)

DST_L1_SP1 = (HASH1_CENTER[0],         HASH1_CENTER[1] - DY_L1)   # (+0.2800, -0.0435)
DST_L1_SP2 = (HASH1_CENTER[0],         HASH1_CENTER[1] + DY_L1)   # (+0.2800, +0.0435)
DST_L2_SP3 = (HASH1_CENTER[0] - DX_L2, HASH1_CENTER[1])           # (+0.2465, +0.0000)
DST_L2_SP4 = (HASH1_CENTER[0] + DX_L2, HASH1_CENTER[1])           # (+0.3135, +0.0000)

# Source regions (per v3 design, m world)
REGIONS = [
    {"name": "R1_좌하", "x": (+0.150, +0.250), "y": (-0.220, -0.130)},
    {"name": "R2_좌상", "x": (+0.150, +0.250), "y": (+0.070, +0.200)},
    {"name": "R3_우하", "x": (+0.330, +0.430), "y": (-0.220, -0.100)},
    {"name": "R4_우상", "x": (+0.330, +0.430), "y": (+0.050, +0.200)},
]
# # build area exclusion (combined L1+L2 footprint + 5mm margin):
#   L1 length-X span: X ∈ [+0.2175, +0.3425] (center ±62.5mm)
#   L2 length-Y span: Y ∈ [-0.0625, +0.0625] (center ±62.5mm)
#   Combined + 5mm margin
EXCLUSION_X = (+0.2125, +0.3475)
EXCLUSION_Y = (-0.0675, +0.0675)
MIN_PAIRWISE_DIST = 0.150        # m

ORIENT_TO_WRIST_R = {"X": 0.0, "Y": +90.0}


# ===============================================================
# Gripper (HARD RULE #19/#20 verified: v6 STATE small=closed, large=open)
# ===============================================================
G_OPEN = +60.0    # jaw ~45mm (≥ 22mm sponge width clearance), v6 ep0 frame 40 state +65.4
G_CLOSE = +5.0    # mech close (jaw ~3mm), grip on 22mm sponge width
G_PRECLOSE = +5.0 # HOME starts CLOSED (matches v6 ep0 frame 0 state +1.5)


# ===============================================================
# wrist_p hard clamp (top-down enforce, per v3 design)
# v6 grasp wrist_p mean +68.8°, range [+41.7°, +88.8°].
# +75° design clamp keeps top-down with IK margin (safer than v2's +80°).
# ===============================================================
WRIST_P_MIN_TOPDOWN = +75.0


HOME = np.array([0.0, 0.0, 90.0, 0.0, 0.0, G_PRECLOSE])


# ===============================================================
# Frame counts per inter-segment (constant; total 146 per demo)
# ===============================================================
SEG_TO_FRAMES = {
    "above_src":  3,     # prev → above_src (start of step)
    "at_src":     4,     # above_src → at_src (descend)
    "close":      3,     # at_src → close (gripper close)
    "lift":       4,     # close → lift (rise to transit_z)
    "transit":    8,     # lift → transit (xy translate + wrist_r rotate)
    "at_dst":     4,     # transit → at_dst (descend straight)
    "open":       3,     # at_dst → open (gripper release)
    "lift_off":   3,     # open → lift_off (rise to next step's above_src z)
}
HOME_BRIDGE_FRAMES = 10  # HOME ↔ first/last anchor (handles 90° wrist_p/wrist_r jump)


# ===============================================================
# Source layout sampling (random per seed)
# ===============================================================
def _sponge_body_overlaps_exclusion(x, y, orient):
    """AABB overlap test: source sponge body (oriented) vs # build area."""
    if orient == "X":
        sp_x_half, sp_y_half = SPONGE_LEN_LONG / 2, SPONGE_WIDTH / 2
    else:
        sp_x_half, sp_y_half = SPONGE_WIDTH / 2, SPONGE_LEN_LONG / 2
    sp_x_min, sp_x_max = x - sp_x_half, x + sp_x_half
    sp_y_min, sp_y_max = y - sp_y_half, y + sp_y_half
    return (sp_x_min < EXCLUSION_X[1] and sp_x_max > EXCLUSION_X[0] and
            sp_y_min < EXCLUSION_Y[1] and sp_y_max > EXCLUSION_Y[0])


def sample_layout(seed, ws=None, max_attempts=200, ik_tol_mm=3.0):
    """Sample 4 sources (one per region) + random orientation per source.

    Returns dict {sources: [(x,y),...], orients: ['X'|'Y',...], attempt: int}.
    If ws (V6WarmStart) provided, also reject layouts where any IK error > ik_tol_mm.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(max_attempts):
        sources, orients = [], []
        for r in REGIONS:
            for _ in range(200):
                x = rng.uniform(*r["x"])
                y = rng.uniform(*r["y"])
                ori = "X" if rng.random() < 0.5 else "Y"
                if not _sponge_body_overlaps_exclusion(x, y, ori):
                    break
            else:
                raise RuntimeError(f"seed={seed}: failed to sample {r['name']} (sponge body overlap)")
            sources.append((x, y))
            orients.append(ori)
        ok_dist = True
        for i in range(4):
            for j in range(i + 1, 4):
                d = np.hypot(sources[i][0] - sources[j][0], sources[i][1] - sources[j][1])
                if d < MIN_PAIRWISE_DIST:
                    ok_dist = False; break
            if not ok_dist:
                break
        if not ok_dist:
            continue

        layout = {"sources": sources, "orients": orients, "attempt": attempt}
        if ws is None:
            return layout
        # IK feasibility check on full anchor set
        anchors = build_anchors(layout)
        home_tcp = tuple(fk_tcp(HOME))
        anchors_full = ([("HOME_start", home_tcp, G_PRECLOSE, 0.0)] +
                        anchors +
                        [("HOME_end",   home_tcp, G_PRECLOSE, 0.0)])
        _, _, errs, _fails = solve_anchors(anchors_full, ws)
        if errs.max() <= ik_tol_mm:
            return layout
    raise RuntimeError(
        f"seed={seed}: failed to sample layout in {max_attempts} attempts "
        f"(pairwise>{MIN_PAIRWISE_DIST*1000:.0f}mm AND IK<{ik_tol_mm}mm)"
    )


# ===============================================================
# Anchor build (32 step anchors = 4 step × 8 anchor)
# ===============================================================
def build_anchors(layout):
    """Build 32 step anchors from layout dict.

    Anchor 4-tuple: (tag, target_xyz_world, gripper_cmd, wrist_r_deg).
    For L2 placements (S3, S4), the descent from transit (+150mm) to at_dst (+80mm)
    is a STRAIGHT VERTICAL drop directly above the L2 destination XY — held sponge
    bottom passes from +105mm to +35mm world over the L2 dst, just settling onto L1
    top. No lateral motion below transit_z within the # build area.
    """
    src = layout["sources"]
    orients = layout["orients"]
    src_wr = [ORIENT_TO_WRIST_R[o] for o in orients]

    z_above_src = Z_TCP_GRASP_L1 + Z_APPROACH    # +0.073 (above source grasp z)
    z_above_dst_l1 = Z_TCP_PLACE_L1 + Z_APPROACH # +0.073 (above L1 place z)
    z_above_dst_l2 = Z_TCP_PLACE_L2 + Z_APPROACH # +0.120 (above L2 place z)

    def step(tag, src_xy, src_wrist_r, dst_xy, dst_wrist_r, dst_z, z_above_dst):
        return [
            (f"{tag}.above_src", (src_xy[0], src_xy[1], z_above_src),    G_OPEN,  src_wrist_r),
            (f"{tag}.at_src",    (src_xy[0], src_xy[1], Z_TCP_GRASP_L1), G_OPEN,  src_wrist_r),
            (f"{tag}.close",     (src_xy[0], src_xy[1], Z_TCP_GRASP_L1), G_CLOSE, src_wrist_r),
            (f"{tag}.lift",      (src_xy[0], src_xy[1], Z_TRANSIT),      G_CLOSE, src_wrist_r),
            (f"{tag}.transit",   (dst_xy[0], dst_xy[1], Z_TRANSIT),      G_CLOSE, dst_wrist_r),
            (f"{tag}.at_dst",    (dst_xy[0], dst_xy[1], dst_z),          G_CLOSE, dst_wrist_r),
            (f"{tag}.open",      (dst_xy[0], dst_xy[1], dst_z),          G_OPEN,  dst_wrist_r),
            (f"{tag}.lift_off",  (dst_xy[0], dst_xy[1], z_above_dst),    G_OPEN,  dst_wrist_r),
        ]

    anchors = []
    # S1, S2 → L1 (X-axis), S3, S4 → L2 (Y-axis)
    anchors += step("S1", src[0], src_wr[0], DST_L1_SP1, ORIENT_TO_WRIST_R["X"],
                    Z_TCP_PLACE_L1, z_above_dst_l1)
    anchors += step("S2", src[1], src_wr[1], DST_L1_SP2, ORIENT_TO_WRIST_R["X"],
                    Z_TCP_PLACE_L1, z_above_dst_l1)
    anchors += step("S3", src[2], src_wr[2], DST_L2_SP3, ORIENT_TO_WRIST_R["Y"],
                    Z_TCP_PLACE_L2, z_above_dst_l2)
    anchors += step("S4", src[3], src_wr[3], DST_L2_SP4, ORIENT_TO_WRIST_R["Y"],
                    Z_TCP_PLACE_L2, z_above_dst_l2)
    return anchors


# ===============================================================
# IK solve with wrist_p clamp + wrist_r override
# ===============================================================
def solve_anchors(anchors_full, ws):
    """Solve IK per anchor. Mutates roarm_kinematics.JOINT_LIMITS_DEG["wrist_p"]
    temporarily so clip_joints inside ik_dls enforces wrist_p ≥ +75° (top-down).
    HARD RULE #5 compliance: try/finally restores limits.
    """
    orig_wp = roarm_kinematics.JOINT_LIMITS_DEG["wrist_p"]
    roarm_kinematics.JOINT_LIMITS_DEG["wrist_p"] = (WRIST_P_MIN_TOPDOWN, +90.0)
    try:
        states, tcps, errs, fails = [], [], [], []
        q_prev = HOME.copy()
        for tag, target_xyz, g_cmd, wrist_r_deg in anchors_full:
            # HOME bridges bypass IK (preserve real-robot start/end pose)
            if tag in ("HOME_start", "HOME_end"):
                q = HOME.copy()
                q[5] = g_cmd
                tcp_actual = fk_tcp(q)
                err_mm = float(np.linalg.norm(tcp_actual - np.asarray(target_xyz)) * 1000.0)
                states.append(q); tcps.append(tcp_actual); errs.append(err_mm)
                q_prev = q.copy()
                continue

            target = np.asarray(target_xyz)
            tcp_prev = fk_tcp(q_prev)
            if np.linalg.norm(tcp_prev - target) < 0.150:
                q0 = q_prev.copy()
            else:
                q0_v6, _ = ws.query(target)
                q0 = q0_v6.copy()
            q0[3] = max(q0[3], WRIST_P_MIN_TOPDOWN)
            q0[4] = wrist_r_deg

            q, _conv, _err, _it = ik_dls(target, q0, max_iter=300, tol_mm=1.0, damping=2.0)
            q[5] = g_cmd
            q[4] = wrist_r_deg
            q = clip_joints(q)

            tcp_actual = fk_tcp(q)
            err_mm = float(np.linalg.norm(tcp_actual - target) * 1000.0)
            states.append(q); tcps.append(tcp_actual); errs.append(err_mm)
            if err_mm > 5.0:
                fails.append((tag, err_mm))
            q_prev = q.copy()
        return np.array(states), np.array(tcps), np.array(errs), fails
    finally:
        roarm_kinematics.JOINT_LIMITS_DEG["wrist_p"] = orig_wp


# ===============================================================
# Trajectory interpolation (linear in joint space)
# ===============================================================
def get_seg_frames(anchors_full):
    """Constant frame counts. HOME bridges = 10, within-step segments = SEG_TO_FRAMES[suffix]."""
    counts = []
    for i in range(len(anchors_full) - 1):
        cur_tag = anchors_full[i][0]
        nxt_tag = anchors_full[i + 1][0]
        if cur_tag == "HOME_start" or nxt_tag == "HOME_end":
            counts.append(HOME_BRIDGE_FRAMES)
        else:
            seg_dst = nxt_tag.split(".")[1]
            counts.append(SEG_TO_FRAMES.get(seg_dst, 3))
    return counts


def interpolate_trajectory(anchor_states, frames_per_seg):
    traj = []
    cum_frames = [0]
    for i in range(len(anchor_states) - 1):
        n_frames = frames_per_seg[i]
        for k in range(n_frames):
            alpha = k / n_frames
            q = (1.0 - alpha) * anchor_states[i] + alpha * anchor_states[i + 1]
            traj.append(q)
        cum_frames.append(cum_frames[-1] + n_frames)
    traj.append(anchor_states[-1])
    return np.array(traj), cum_frames


# ===============================================================
# Per-demo generation
# ===============================================================
def generate_one_demo(seed, ws):
    layout = sample_layout(seed, ws=ws)
    anchors = build_anchors(layout)
    home_tcp = tuple(fk_tcp(HOME))
    anchors_full = ([("HOME_start", home_tcp, G_PRECLOSE, 0.0)] +
                    anchors +
                    [("HOME_end",   home_tcp, G_PRECLOSE, 0.0)])
    states, tcps, errs, fails = solve_anchors(anchors_full, ws)
    seg_frames = get_seg_frames(anchors_full)
    traj, cum_frames = interpolate_trajectory(states, seg_frames)

    anchor_tags = [a[0] for a in anchors_full]
    anchor_frame_map = {tag: cum_frames[i] for i, tag in enumerate(anchor_tags)}

    return {
        "seed": seed,
        "layout": layout,
        "anchor_tags": anchor_tags,
        "anchor_states": states,
        "anchor_tcps": tcps,
        "anchor_errs_mm": errs,
        "ik_fails": fails,
        "trajectory": traj,
        "anchor_frame_map": anchor_frame_map,
        "n_frames": len(traj),
    }


def summarize_demo(demo):
    traj = demo["trajectory"]
    print(f"  Demo seed={demo['seed']}: T={len(traj)} frames, attempt={demo['layout']['attempt']}")
    print(f"    Sources (mm world):")
    for i, (xy, ori) in enumerate(zip(demo['layout']['sources'], demo['layout']['orients']), 1):
        print(f"      S{i} ({ori}): ({xy[0]*1000:+7.1f}, {xy[1]*1000:+7.1f})")
    print(f"    Anchor IK errors: mean={demo['anchor_errs_mm'].mean():.2f}mm  "
          f"max={demo['anchor_errs_mm'].max():.2f}mm  fails(>5mm)={len(demo['ik_fails'])}")
    if demo["ik_fails"]:
        for tag, e in demo["ik_fails"][:5]:
            print(f"      FAIL {tag}: err={e:.1f}mm")
    print(f"    Joint state stats (over interpolated trajectory):")
    names = ["base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper"]
    for i, n in enumerate(names):
        j = traj[:, i]
        lo, hi = JOINT_LIMITS_DEG[n]
        clip_count = int(((j <= lo + 0.01) | (j >= hi - 0.01)).sum())
        print(f"      {n:10s}: [{j.min():+6.1f}, {j.max():+6.1f}] mean={j.mean():+5.1f}  clipped={clip_count}")
    tcps = np.array([fk_tcp(q) for q in traj])
    z = tcps[:, 2]
    over_train = int((z > SAFETY_Z_MAX_TRAIN).sum())
    over_deploy = int((z > SAFETY_Z_MAX_DEPLOY).sum())
    print(f"    TCP z range: [{z.min()*1000:+.1f}, {z.max()*1000:+.1f}]mm  "
          f">+{int(SAFETY_Z_MAX_TRAIN*1000)}mm: {over_train}  >+{int(SAFETY_Z_MAX_DEPLOY*1000)}mm: {over_deploy}")


def save_demo(out_dir, demo):
    seed = demo["seed"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / f"demo_{seed:04d}_trajectory.csv",
               demo["trajectory"], delimiter=",",
               header="base,shoulder,elbow,wrist_p,wrist_r,gripper", comments="")
    np.savetxt(out / f"demo_{seed:04d}_anchors.csv",
               demo["anchor_states"], delimiter=",",
               header="base,shoulder,elbow,wrist_p,wrist_r,gripper", comments="")
    layout = {
        "seed": seed,
        "sources_m": [list(s) for s in demo["layout"]["sources"]],
        "orients": demo["layout"]["orients"],
        "src_wrist_r_deg": [ORIENT_TO_WRIST_R[o] for o in demo["layout"]["orients"]],
        "dst_l1_sp1_m": list(DST_L1_SP1),
        "dst_l1_sp2_m": list(DST_L1_SP2),
        "dst_l2_sp3_m": list(DST_L2_SP3),
        "dst_l2_sp4_m": list(DST_L2_SP4),
        "dst_l1_wrist_r_deg": ORIENT_TO_WRIST_R["X"],
        "dst_l2_wrist_r_deg": ORIENT_TO_WRIST_R["Y"],
        "z_layer1_top_m": Z_LAYER1_TOP,
        "z_layer2_top_m": Z_LAYER2_TOP,
        "z_tcp_grasp_l1_m": Z_TCP_GRASP_L1,
        "z_tcp_place_l2_m": Z_TCP_PLACE_L2,
        "z_transit_m": Z_TRANSIT,
        "n_frames": demo["n_frames"],
        "anchor_tags": demo["anchor_tags"],
        "anchor_frame_map": demo["anchor_frame_map"],
        "g_open": G_OPEN,
        "g_close": G_CLOSE,
        "g_preclose": G_PRECLOSE,
        "table_z_m": TABLE_Z,
        "sponge_height_edge_m": SPONGE_HEIGHT_EDGE,
        "sponge_len_long_m": SPONGE_LEN_LONG,
        "sponge_width_m": SPONGE_WIDTH,
        "wrist_p_min_topdown_deg": WRIST_P_MIN_TOPDOWN,
    }
    with open(out / f"demo_{seed:04d}_layout.json", "w") as f:
        json.dump(layout, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out-dir", type=str, default="sim_demos_v3")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print("Loading v6 warm-start index ...")
    ws = V6WarmStart()
    show = args.seeds[:5] + (["..."] if len(args.seeds) > 5 else [])
    print(f"Generating {len(args.seeds)} demos: seeds={show}")

    out_dir = Path(args.out_dir)
    summary = []
    for s in args.seeds:
        demo = generate_one_demo(s, ws)
        summarize_demo(demo)
        summary.append({
            "seed": s,
            "n_frames": demo["n_frames"],
            "ik_max_err_mm": float(demo["anchor_errs_mm"].max()),
            "ik_fails": len(demo["ik_fails"]),
            "src_orients": demo["layout"]["orients"],
        })
        if not args.dry_run:
            save_demo(out_dir, demo)
    if not args.dry_run:
        with open(out_dir / "summary.json", "w") as f:
            json.dump({
                "n_demos": len(summary),
                "n_frames_per_demo_constant": summary[0]["n_frames"] if summary else None,
                "demos": summary,
                "constants": {
                    "g_open": G_OPEN, "g_close": G_CLOSE, "g_preclose": G_PRECLOSE,
                    "sponge_height_edge_m": SPONGE_HEIGHT_EDGE,
                    "sponge_width_m": SPONGE_WIDTH,
                    "z_tcp_grasp_l1_m": Z_TCP_GRASP_L1,
                    "z_tcp_place_l2_m": Z_TCP_PLACE_L2,
                    "z_transit_m": Z_TRANSIT,
                    "wrist_p_min_topdown_deg": WRIST_P_MIN_TOPDOWN,
                    "hash1_center_m": list(HASH1_CENTER),
                    "dy_l1_m": DY_L1, "dx_l2_m": DX_L2,
                },
            }, f, indent=2)
    print(f"\nDone. {'(dry-run)' if args.dry_run else f'Wrote {out_dir}/'}")


if __name__ == "__main__":
    main()
