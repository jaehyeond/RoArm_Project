#!/usr/bin/env python3
"""G0b T3 — D29xH50 standing cylinder, top-center vertical physical grasp probe (p9).

Case g0b_d420 (T-ladder T3, D419/D420/D421). New authorship from the p7 skeleton
(p7_branch_b_cube2cm_local_grasp_close_sweep_probe.py) with the preregistered
deltas D-1..D-8 of claudedocs/runtime_logs/grasp_track/g0b_d420/t3_conversion_design.md:

  D-1  q5 convention reversal: frozen-track authority (d337:59-60, d409 close sweep)
       is q5 LARGE = OPEN (1.5413 rad = 88.31 deg), decreasing = closing. APPROACH/
       DESCEND run at --descend_open_deg (default = frozen OPEN; attempt3 measured
       that a full-open descend closes past the D29 top without contact, so a
       preregistered partial opening > the marker q5 gate is allowed); LATCH is a
       DESCENDING sweep close_deg[0]=descend_open -> 24 deg (2-deg steps inside
       the d409 first-contact band 41.40..31.65 deg).
  D-2  env _grasped marker is structurally unfireable for H50 top grasp
       (TCP-center distance 0.0255 m > 0.025 m) -> probe-side monkeypatch:
       distance < 0.030 m AND q5 <= 41.40 deg. Marker only gates LATCH; the
       physical grasp evidence remains LIFT follow >= 6 mm.
  D-3  gripper collision body = frozen attempt3 asset REUSE (no re-decomposition,
       D415/D420-R1): ROARM_M3_USD_PATH -> g0a_d344 attempt3 64+64-part USD,
       root+physics layer sha pinned, stage audit (64 enabled part_* + exactly 1
       disabled legacy node_STL_BINARY_* per body) hard-fails before any physics step.
  D-4  spawn = CylinderCfg D29xH50 axis Z, mass 0.02483 kg (measured, HARD RULE #18),
       friction mu_s 0.40 / mu_d 0.30 / rest 0.0 (preregistered assumption leg,
       t3_mass_friction_contract.md — NOT measured; p7's 1.5/1.2 is a banned D362
       lineage transfer). Robot/ground material + combine mode recorded at runtime.
  D-5  ground plane z=0 vs TABLE_Z plan constant -> object settles +12.117 mm up;
       the inherited settled replan re-derives all targets (T2b covers this height).
  D-6  env default USD path is the retired B200 path (HARD RULE #27) -> this probe
       sets ROARM_M3_USD_PATH itself and aborts on any /NHNHOME resolution.
  D-7  D341 Rerun: full executed step timeline (TCP/object pos, object quat, q5,
       gate scalars, marker) on physics_step/sim_time_s timelines + verdict +
       fixed blueprint + contract validation + inspection PNG. Contact-force
       arrows omitted-justified: no gate consumes contact forces.
  D-8  inherited unchanged: APPROACH->DESCEND->LATCH->HOLD->LIFT chain, verdict set,
       physics gates (drift 6mm / speed 0.08 / tilt 12deg / upright 0.95 /
       lift_follow 6mm / target 3mm), marker-only attach (kinematic pin disabled),
       posewrite watch, set_joint_position_target watch. episode_length_s 10 -> 20 s
       in the design doc, superseded to 60 s by the audit MAJOR-c repair (third
       prereg supersession entry alongside D-4 spawn and D-6 asset root).
       Attempt1 keeps the original gates; if a push-grasping funnel trips the drift
       gate, gate revision is a separate preregistered leg.

T2 consequence (D421; scope corrected per audit wf_78b1adfd FATAL-2 / D422 Impl 1):
all waypoint IK uses the vertical tool-axis DLS solver from the p8 T2 probe
(5-task DLS over q0..q3, q4=0). The vertical CONTRACT (pos_err <= 3 mm AND
tool-axis tilt <= 5 deg) applies to the three planned targets (REACH gate), to
the approach ARRIVAL waypoint, and to every descend/lift corridor waypoint.
HOME->approach transit waypoints are gated on POSITION ONLY (p7 ik_dls
acceptance semantics — a vertical tool axis is kinematically infeasible above
z ~0.153 m). Two reverify wf_3cea04db MAJOR repairs harden that transit:
selection uses a tight 0.5 mm position band (the earlier 3 mm min-tilt-in-band
selection commanded 2.52-2.55 mm residuals, leaving ~0.5 mm of the 3 mm reached
gate), and every chain waypoint solve is clamped to a joint trust region around
its seed (--waypoint_max_joint_dev_deg; an unclamped ~46 deg wrist/elbow hop at
wp002 put the velocity-clamped TCP slew at 9.90 mm vs the 10 mm early-kill
gate). The soft vertical bias (w_axis) still blends the posture toward vertical
across the descending chain, so the arrival waypoint reaches the T2-verified
vertical configuration continuously.
Plan-target IK failure -> REACH_FAIL; waypoint execution failure -> APPROACH_FAIL.
Default spawn pose = seed0_S1 (T2/T2b PASS candidate, D421/D422 recommendation).

Audit wf_78b1adfd-20d confirmed-finding repairs (applied in this revision; to be
pinned in t3_prereg.md before the Isaac run — the prereg does not exist yet at
authoring time): FATAL-1 lift follow is measured phase-cumulative (object z vs
lift-phase start), not per-waypoint segment; FATAL-2 transit vertical-gate scope
above; MAJOR close 'reached' is stall-aware (contact stall = |dq5| below
--gripper_stall_rate_deg_per_step for --gripper_stall_min_steps while err > gate
counts as reached, recorded as gripper_stalled); MAJOR the close sweep continues
through the full preregistered band after marker fire (p7 flag restored,
default ON) so close_records covers the band and HOLD no longer jumps 41.4->24;
MAJOR episode_length_s 20 -> 60 s (worst-case budget 3735 control steps —
seed0_S2 settled height, 46 approach wp, preflight v2 measured — < 6000);
MAJOR outputs are --tag parameterized with an existing-artifact abort guard;
MAJOR RERUN_VERSION_MISMATCH exit closes the app, plus a __main__ finally net;
MINOR the /NHNHOME guard now checks the post-import effective
cfg.robot.spawn.usd_path; MINOR empty-results aggregates serialize as null.

Reverify wf_3cea04db-7c2 survivor repairs (this revision): MAJOR transit IK
selection band 0.5 mm + per-waypoint joint trust region (see T2 paragraph) +
the TCP-step runaway gate is phase-scoped (--transit_tcp_step_gate_m 20 mm for
the approach phase only, where the object sits >= 40 mm away and its own
drift/speed/tilt gates are untouched — the trust region caps hop length but
not the instantaneous multi-joint slew speed, so the pessimistic
velocity-clamp bound can legitimately sweep ~10 mm TCP per control step);
MINOR _close_all closes each handle under its own try so sim_app.close() always
runs; MINOR app_id/recording_id derive from --tag (multi-leg viewer sessions
never merge recordings); MINOR _verdict consumes lift path_ok (an IK-truncated
lift cannot report GRASP_PASS); MINOR aggregate lift_follow_delta_m is
NaN-sanitized; MINOR the solver ranking band follows a narrowed
--target_error_gate_m (min with 3.0 mm).

Diagnostic-only Isaac script: no training, no env/chain default edits, no
constraint prims, no SurfaceGripper, no transport target or release, and the
env's hidden kinematic pose-write attach is disabled and watched (never counted
as grasp evidence). New outputs only under
claudedocs/runtime_logs/grasp_track/g0b_d420/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim_scripts"))

from roarm_kinematics import _CHAIN, JOINT_LIMITS_DEG, Tmat, Trot_z  # noqa: E402

LOG = "g0b_t3_grasp"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"

TABLE_Z = -0.012117
HOME_ARM_DEG = np.array([0.0, 0.0, 90.0, 0.0], dtype=np.float64)

# D-1: frozen grasp-track gripper convention (d337:59-60): large = OPEN.
Q5_OPEN_RAD = 1.5413
Q5_OPEN_DEG = math.degrees(Q5_OPEN_RAD)  # 88.3096 deg

# D-2: probe-side grasp marker (env constants 0.025/0.4rad are cube-10.5mm era).
GRASP_MARKER_DIST_M = 0.030
GRASP_MARKER_Q5_MAX_DEG = 41.40

# D-3: frozen attempt3 collision asset (REUSE ONLY — re-decomposition banned, D415).
ATTEMPT3_USD = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3"
    / "roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd"
)
ATTEMPT3_ROOT_SHA256 = "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff"
ATTEMPT3_PHYSICS_LAYER = ATTEMPT3_USD.parent / "configuration/roarm_m3_physics.usd"
ATTEMPT3_PHYSICS_SHA256 = "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503"
BODY_PATHS = {
    "link5": "/World/envs/env_0/Robot/link5",
    "gripper_link": "/World/envs/env_0/Robot/gripper_link",
}
EXPECTED_PART_COUNT = 64
LEGACY_COLLIDER_FRAGMENT = "node_STL_BINARY_"

# p7 candidate spawn poses (p7_branch_b probe constants; T2 PASS set = seed0_S1,
# seed0_S2, R1_center, R2_center — outer poses will fail the vertical plan gate).
SOURCE_REGIONS = (
    (0.150, 0.250, -0.220, -0.130),
    (0.150, 0.250, +0.070, +0.200),
    (0.330, 0.430, -0.220, -0.100),
    (0.330, 0.430, +0.050, +0.200),
)
FOUR_SPONGE_SEED0_SOURCES = {
    "seed0_S1": (+0.21369616873214542, -0.19571919576125169),
    "seed0_S2": (+0.15165276355285290, +0.17572513109603544),
    "seed0_S3": (+0.39066357757671800, -0.13246041268192021),
    "seed0_S4": (+0.42350724237877680, +0.17237803311822986),
}

OUT_DIR = REPO / "claudedocs" / "runtime_logs" / "grasp_track" / "g0b_d420"

# Audit MAJOR-e repair: handles for the __main__ finally net so any exception or
# early return after AppLauncher still closes Kit (no wedged headless process).
_CLEANUP: dict[str, Any] = {"env": None, "sim_app": None}


def _close_all() -> None:
    # Reverify wf_3cea04db MINOR (2 lenses): each handle closes under its own
    # try and is cleared FIRST, so an env.close() exception can neither skip
    # sim_app.close() nor poison the __main__ finally re-entry with a repeated
    # half-torn-down teardown.
    env = _CLEANUP["env"]
    _CLEANUP["env"] = None
    if env is not None:
        try:
            env.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[{LOG}] cleanup_env_error={exc!r}", flush=True)
    sim_app = _CLEANUP["sim_app"]
    _CLEANUP["sim_app"] = None
    if sim_app is not None:
        try:
            sim_app.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[{LOG}] cleanup_sim_app_error={exc!r}", flush=True)

# ---------------------------------------------------------------------------
# Vertical tool-axis IK (copied from the T2 probe
# p8_g0b_t2_cyld29h50_vertical_tool_axis_ik_reachability_probe.py, sha-pinned in
# t2_prereg/t2b_prereg — 5-task DLS over q0..q3 with q4=0; roll about a vertical
# tool axis moves neither position nor axis).
# ---------------------------------------------------------------------------
V6_LIMITS_DEG = {k: JOINT_LIMITS_DEG[k] for k in ("base", "shoulder", "elbow", "wrist_p")}
IK_POS_KEY_MM = 3.0  # solver-internal ranking constant (gates are applied outside)
# Reverify wf_3cea04db MAJOR repair: transit (require_tilt=False) selection uses
# this tight position band so the commanded residual never spends the 3 mm
# reached gate (the 3.0 mm min-tilt-in-band selection commanded 2.52-2.55 mm);
# tilt is still preferred INSIDE the band so the chain keeps verticalizing.
TRANSIT_POS_BAND_MM = 0.5
# The biased solve alone equilibrates at ~1.2-1.5 mm in mid-transit (measured,
# trust-region-size independent), so transit waypoints get a short bias-free
# position polish seeded at the biased solution; it moves <= this many degrees,
# recovering pe to ~0.0x mm without giving back the verticalization.
TRANSIT_POLISH_DEV_DEG = 2.0


def fk_points(q4_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    q = np.radians(np.array([q4_deg[0], q4_deg[1], q4_deg[2], q4_deg[3], 0.0, 0.0]))
    T = np.eye(4)
    origins: list[np.ndarray] = []
    link5 = None
    for name, xyz, rpy, qi in _CHAIN:
        T = T @ Tmat(xyz, rpy)
        if qi is not None:
            T = T @ Trot_z(q[qi])
        origins.append(T[:3, 3].copy())
        if name == "link4_to_link5":
            link5 = T[:3, 3].copy()
    tcp = T[:3, 3].copy()
    return tcp, link5, origins


def axis_tilt(q4_deg: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    tcp, link5, _ = fk_points(q4_deg)
    axis = tcp - link5
    axis = axis / np.linalg.norm(axis)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(axis, [0.0, 0.0, -1.0]))))))
    return tcp, tilt, axis


def clip4(q4_deg: np.ndarray, limits: dict[str, tuple[float, float]]) -> np.ndarray:
    out = q4_deg.copy()
    for i, name in enumerate(("base", "shoulder", "elbow", "wrist_p")):
        lo, hi = limits[name]
        out[i] = max(lo, min(hi, out[i]))
    return out


def _task_error(q4_deg: np.ndarray, target_p: np.ndarray, w_axis: float) -> np.ndarray:
    tcp, _tilt, axis = axis_tilt(q4_deg)
    return np.array(
        [
            target_p[0] - tcp[0],
            target_p[1] - tcp[1],
            target_p[2] - tcp[2],
            -w_axis * axis[0],
            -w_axis * axis[1],
        ],
        dtype=np.float64,
    )


def dls_vertical(
    target_p: np.ndarray,
    seed4_deg: np.ndarray,
    limits: dict[str, tuple[float, float]],
    max_iter: int = 160,
    w_axis: float = 0.03,
    damping: float = 0.002,
    step_clip_deg: float = 4.0,
    eps_deg: float = 0.05,
    pos_band_mm: float = IK_POS_KEY_MM,
    max_dev_from_seed_deg: float | None = None,
) -> tuple[np.ndarray, float, float, int]:
    seed_ref = clip4(np.asarray(seed4_deg, dtype=np.float64).copy(), limits)
    q = seed_ref.copy()
    best = None
    for it in range(max_iter):
        e = _task_error(q, target_p, w_axis)
        tcp, tilt, _axis = axis_tilt(q)
        pos_err_mm = float(np.linalg.norm(target_p - tcp)) * 1000.0
        key = (pos_err_mm > pos_band_mm, tilt if pos_err_mm <= pos_band_mm else 1.0e9, pos_err_mm)
        if best is None or key < best[0]:
            best = (key, q.copy(), pos_err_mm, tilt, it)
        if pos_err_mm < 0.2 and tilt < 0.2:
            break
        J = np.zeros((5, 4), dtype=np.float64)
        for i in range(4):
            qp = q.copy()
            qp[i] += eps_deg
            qm = q.copy()
            qm[i] -= eps_deg
            J[:, i] = (_task_error(qp, target_p, w_axis) - _task_error(qm, target_p, w_axis)) / (2.0 * eps_deg)
        M = J @ J.T + (damping**2) * np.eye(5)
        try:
            dq = -J.T @ np.linalg.solve(M, e)
        except np.linalg.LinAlgError:
            break
        m = float(np.max(np.abs(dq)))
        if m > step_clip_deg:
            dq = dq * (step_clip_deg / m)
        q = clip4(q + dq, limits)
        if max_dev_from_seed_deg is not None:
            # Reverify wf_3cea04db MAJOR repair: joint trust region around the
            # chain seed keeps consecutive waypoint commands continuous (the
            # unclamped solver hopped ~46 deg wrist/elbow at transit wp002,
            # putting the velocity-clamped TCP slew at 9.90 mm vs the 10 mm
            # early-kill gate). Clip stays inside joint limits: seed_ref is
            # limit-clipped and the interval is intersected after clip4.
            q = np.clip(q, seed_ref - max_dev_from_seed_deg, seed_ref + max_dev_from_seed_deg)
    _key, qb, pe, tl, it_used = best
    return qb, pe, tl, it_used


def vertical_seeds(x: float, y: float) -> list[np.ndarray]:
    az = math.degrees(math.atan2(y, x))
    return [
        np.array([az, 0.0, 90.0, 90.0]),
        np.array([az, 45.0, 100.0, 35.0]),
        np.array([az, 20.0, 130.0, 30.0]),
        np.array([az, 60.0, 85.0, 35.0]),
        np.array([1.7730, 35.6563, 111.8334, 9.4908]),
    ]


# ---------------------------------------------------------------------------
# Plan / step result containers (p7 structure + vertical tilt fields)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PosePlan:
    label: str
    center: np.ndarray
    world_grasp: np.ndarray
    approach_tcp: np.ndarray
    descend_tcp: np.ndarray
    lift_tcp: np.ndarray
    q_approach_deg: np.ndarray  # 6-dof, q4=0, q5=OPEN
    q_descend_deg: np.ndarray
    q_lift_deg: np.ndarray
    approach_ik_ok: bool
    descend_ik_ok: bool
    lift_ik_ok: bool
    approach_ik_err_mm: float
    descend_ik_err_mm: float
    lift_ik_err_mm: float
    approach_tilt_deg: float
    descend_tilt_deg: float
    lift_tilt_deg: float


@dataclass
class StepResult:
    label: str
    reached: bool
    steps: int
    final_target_error_m: float
    max_tcp_step_m: float
    max_object_drift_m: float
    max_object_speed_mps: float
    max_tilt_deg: float
    min_upright_z: float
    object_follow_delta_m: float
    grasped_seen: bool
    attach_calls: int
    posewrite_calls: int
    early_kill: bool
    gripper_stalled: bool = False


def _yes(value: bool) -> str:
    return "YES" if value else "NO"


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _fmt_xyz(value: np.ndarray) -> str:
    return f"([{value[0]:+.6f}, {value[1]:+.6f}, {value[2]:+.6f}])"


def _fmt_quat(value: np.ndarray) -> str:
    q = np.asarray(value, dtype=np.float64)
    return f"[w={q[0]:+.6f}, x={q[1]:+.6f}, y={q[2]:+.6f}, z={q[3]:+.6f}]"


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _object_pose_metrics(pos: np.ndarray, quat: np.ndarray, object_size: np.ndarray) -> dict[str, float]:
    # Box-half-extent approximation for oriented_top_z (reporting only);
    # up_z / tilt_deg are exact for the upright cylinder question.
    rot = _quat_wxyz_to_rot(quat)
    half_extents = object_size / 2.0
    oriented_half_height = float(np.dot(np.abs(rot[2, :]), half_extents))
    up_z = float(rot[2, 2])
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    return {
        "up_z": up_z,
        "tilt_deg": tilt_deg,
        "oriented_top_z_m": float(pos[2] + oriented_half_height),
        "center_z_m": float(pos[2]),
    }


def _solve_q_vertical(
    target_tcp: np.ndarray,
    seed4_deg: np.ndarray,
    gripper_deg: float,
    args: argparse.Namespace,
    multi_seed_xy: tuple[float, float] | None = None,
    require_tilt: bool = True,
    max_dev_from_seed_deg: float | None = None,
) -> tuple[np.ndarray, bool, float, float]:
    """Vertical-constrained IK -> (q6_deg with q4=0/q5=gripper, ok, pos_err_mm, tilt_deg).

    require_tilt=False (FATAL-2 repair): transit waypoints accept on position only;
    the solver still soft-biases toward vertical, but tilt does not gate ok.
    Reverify wf_3cea04db MAJOR repairs: transit SELECTION uses the tight
    TRANSIT_POS_BAND_MM band (position budget is not spent chasing an ungated
    tilt), and max_dev_from_seed_deg clamps the solve to a joint trust region
    around the chain seed. The require_tilt band also follows a narrowed
    --target_error_gate_m (reverify MINOR: band/gate mismatch).
    """
    candidates: list[np.ndarray] = [np.asarray(seed4_deg[:4], dtype=np.float64)]
    if multi_seed_xy is not None:
        candidates.extend(vertical_seeds(*multi_seed_xy))
    band_mm = min(IK_POS_KEY_MM, args.target_error_gate_m * 1000.0) if require_tilt else TRANSIT_POS_BAND_MM
    best = None
    for seed in candidates:
        q4, pe, tl, _it = dls_vertical(
            target_tcp,
            seed,
            V6_LIMITS_DEG,
            pos_band_mm=band_mm,
            max_dev_from_seed_deg=max_dev_from_seed_deg,
        )
        if not require_tilt and pe > TRANSIT_POS_BAND_MM:
            # Transit position polish (reverify MAJOR-2): the biased solve
            # equilibrates at ~1.2-1.5 mm here; a short bias-free polish from
            # the biased solution recovers position (strictly non-worsening).
            q4_p, pe_p, tl_p, _it_p = dls_vertical(
                target_tcp,
                q4,
                V6_LIMITS_DEG,
                max_iter=60,
                w_axis=0.0,
                pos_band_mm=band_mm,
                max_dev_from_seed_deg=TRANSIT_POLISH_DEV_DEG,
            )
            if pe_p < pe:
                q4, pe, tl = q4_p, pe_p, tl_p
        key = (pe > band_mm, tl if pe <= band_mm else 1.0e9, pe)
        if best is None or key < best[0]:
            best = (key, q4, pe, tl)
        if pe < 0.3 and tl < 0.5:
            break
    _key, q4, pe, tl = best
    q6 = np.array([q4[0], q4[1], q4[2], q4[3], 0.0, gripper_deg], dtype=np.float64)
    ok = pe <= args.target_error_gate_m * 1000.0 and (tl <= args.plan_tilt_gate_deg if require_tilt else True)
    return q6, bool(ok), float(pe), float(tl)


def _workspace_xy_from_label(label: str) -> tuple[float, float]:
    if label in FOUR_SPONGE_SEED0_SOURCES:
        return FOUR_SPONGE_SEED0_SOURCES[label]
    if label.startswith("R") and "_center" in label:
        region_idx = int(label[1]) - 1
        x_min, x_max, y_min, y_max = SOURCE_REGIONS[region_idx]
        return (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
    raise ValueError(f"unknown pose_label={label!r}; use seed0_S1..seed0_S4 or R1_center..R4_center")


def _build_plan_from_center(args: argparse.Namespace, center: np.ndarray, label: str) -> PosePlan:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    # D419: grasp style fixed to top-center vertical. No alternative grasp points.
    world_grasp = center + np.array([0.0, 0.0, object_size[2] / 2.0], dtype=np.float64)
    approach_tcp = world_grasp + np.array([0.0, 0.0, args.approach_clearance_m], dtype=np.float64)
    descend_tcp = world_grasp + np.array([0.0, 0.0, args.grasp_surface_margin_m], dtype=np.float64)
    lift_tcp = descend_tcp + np.array([0.0, 0.0, args.lift_delta_m], dtype=np.float64)

    seed = HOME_ARM_DEG.copy()
    xy = (float(center[0]), float(center[1]))
    q_approach, approach_ok, approach_err, approach_tilt = _solve_q_vertical(
        approach_tcp, seed, args.descend_open_deg, args, multi_seed_xy=xy
    )
    q_descend, descend_ok, descend_err, descend_tilt = _solve_q_vertical(
        descend_tcp, q_approach, args.descend_open_deg, args
    )
    q_lift, lift_ok, lift_err, lift_tilt = _solve_q_vertical(
        lift_tcp, q_descend, args.close_deg[-1], args
    )
    return PosePlan(
        label=label,
        center=center,
        world_grasp=world_grasp,
        approach_tcp=approach_tcp,
        descend_tcp=descend_tcp,
        lift_tcp=lift_tcp,
        q_approach_deg=q_approach,
        q_descend_deg=q_descend,
        q_lift_deg=q_lift,
        approach_ik_ok=approach_ok,
        descend_ik_ok=descend_ok,
        lift_ik_ok=lift_ok,
        approach_ik_err_mm=approach_err,
        descend_ik_err_mm=descend_err,
        lift_ik_err_mm=lift_err,
        approach_tilt_deg=approach_tilt,
        descend_tilt_deg=descend_tilt,
        lift_tilt_deg=lift_tilt,
    )


def _build_plan(args: argparse.Namespace) -> PosePlan:
    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    if args.object_xy is None:
        x, y = _workspace_xy_from_label(args.pose_label)
        label = args.pose_label
    else:
        x, y = args.object_xy
        label = "custom_xy"
    center = np.array([x, y, TABLE_Z + object_size[2] / 2.0], dtype=np.float64)
    return _build_plan_from_center(args, center, label)


def _verdict(
    plan: PosePlan,
    approach: StepResult | None,
    descend: StepResult | None,
    latch: StepResult | None,
    hold: StepResult | None,
    lift: StepResult | None,
    args: argparse.Namespace,
    lift_path_ok: bool = True,
) -> str:
    if not (plan.approach_ik_ok and plan.descend_ik_ok and plan.lift_ik_ok):
        return "REACH_FAIL"
    if approach is None or not approach.reached or approach.early_kill:
        return "APPROACH_FAIL"
    if descend is None or not descend.reached or descend.early_kill:
        return "APPROACH_FAIL"
    if latch is None or not latch.reached or latch.early_kill or not latch.grasped_seen:
        return "LATCH_FAIL"
    if hold is None or not hold.reached or hold.early_kill:
        return "HOLD_FAIL"
    # Reverify wf_3cea04db MINOR: an IK-truncated lift path (path_ok False with a
    # reached last waypoint) must not report a completed lift.
    if lift is None or not lift.reached or lift.early_kill or not lift_path_ok or lift.object_follow_delta_m < args.min_lift_follow_m:
        return "LIFT_FAIL"
    return "GRASP_PASS"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# D-3 stage audit (copied from cyl34_top_view_d334...:197-239 inventory +
# cyl34_top_view_d349...:921-934 body checks, adapted to the attempt3 layout).
# ---------------------------------------------------------------------------
def _usd_collision_inventory(inner: Any, body_label: str) -> list[dict[str, Any]]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = inner.scene.stage
    body_path = BODY_PATHS[body_label]
    rows: list[dict[str, Any]] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        if path != body_path and not path.startswith(body_path + "/"):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        approximation = None
        api_prim = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            api_prim = prim
        mesh_prims = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
        if api_prim is None:
            for mesh in mesh_prims:
                if mesh.HasAPI(UsdPhysics.MeshCollisionAPI):
                    api_prim = mesh
                    break
        if api_prim is not None:
            approximation = UsdPhysics.MeshCollisionAPI(api_prim).GetApproximationAttr().Get()
        rows.append(
            {
                "path": path,
                "collision_enabled": True if enabled is None else bool(enabled),
                "approximation": None if approximation is None else str(approximation),
            }
        )
    return rows


def _audit_collision_bodies(base_env: Any) -> tuple[bool, dict[str, Any]]:
    body_checks: dict[str, Any] = {}
    for body in BODY_PATHS:
        rows = _usd_collision_inventory(base_env, body)
        enabled = [r for r in rows if r["collision_enabled"]]
        enabled_parts = [r for r in enabled if "part_" in r["path"].rsplit("/", 1)[-1]]
        legacy = [r for r in rows if LEGACY_COLLIDER_FRAGMENT in r["path"]]
        body_checks[body] = {
            "enabled_total": len(enabled),
            "enabled_part_count": len(enabled_parts),
            "part_count_64": len(enabled_parts) == EXPECTED_PART_COUNT,
            "enabled_only_parts": len(enabled) == len(enabled_parts),
            "legacy_rows": len(legacy),
            "disabled_legacy_exact_one": len(legacy) == 1 and legacy[0]["collision_enabled"] is False,
        }
    audit_pass = all(
        row["part_count_64"] and row["enabled_only_parts"] and row["disabled_legacy_exact_one"]
        for row in body_checks.values()
    )
    return audit_pass, body_checks


# ---------------------------------------------------------------------------
# t3_mass_friction_contract.md section 3: runtime material recording duty.
# ---------------------------------------------------------------------------
def _material_report(base_env: Any) -> list[dict[str, Any]]:
    from pxr import Usd, UsdPhysics

    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None
    rows: list[dict[str, Any]] = []
    stage = base_env.scene.stage
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MaterialAPI):
            continue
        mat = UsdPhysics.MaterialAPI(prim)
        row: dict[str, Any] = {
            "path": prim.GetPath().pathString,
            "static_friction": mat.GetStaticFrictionAttr().Get(),
            "dynamic_friction": mat.GetDynamicFrictionAttr().Get(),
            "restitution": mat.GetRestitutionAttr().Get(),
            "friction_combine_mode": None,
            "restitution_combine_mode": None,
        }
        if PhysxSchema is not None and prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
            pmat = PhysxSchema.PhysxMaterialAPI(prim)
            fcm = pmat.GetFrictionCombineModeAttr()
            rcm = pmat.GetRestitutionCombineModeAttr()
            row["friction_combine_mode"] = None if not fcm else fcm.Get()
            row["restitution_combine_mode"] = None if not rcm else rcm.Get()
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="G0b T3 p9 — D29xH50 top-center vertical grasp probe")
    ap.add_argument("--object_size_m", nargs=3, type=float, default=[0.029, 0.029, 0.050])
    ap.add_argument("--object_mass_kg", type=float, default=0.02483)
    ap.add_argument("--static_friction", type=float, default=0.40)
    ap.add_argument("--dynamic_friction", type=float, default=0.30)
    ap.add_argument("--restitution", type=float, default=0.0)
    ap.add_argument("--pose_label", default="seed0_S1")
    ap.add_argument("--object_xy", nargs=2, type=float, default=None)
    ap.add_argument("--approach_clearance_m", type=float, default=0.040)
    ap.add_argument("--grasp_surface_margin_m", type=float, default=0.0005)
    ap.add_argument("--lift_delta_m", type=float, default=0.010)
    ap.add_argument(
        "--close_deg",
        nargs="+",
        type=float,
        default=[Q5_OPEN_DEG, 60.0, 45.0, 41.40, 39.0, 37.0, 35.0, 33.0, 31.65, 28.0, 24.0],
    )
    # Attempt3 (t3_grasp3): attempt2 measured that a FULL-OPEN (88.31 deg)
    # descend to the contact-limited depth (TCP=top+5.5mm; open-jaw structure
    # touches the top at +4.4mm) closes 88.31->24 deg without ever contacting
    # the D29 cylinder — the closing faces pass above the top. A preregistered
    # partial opening lets the deeper-hanging jaw tips straddle the top rim
    # during descend. Must stay above the marker q5 gate (41.40) so the D-2
    # marker still cannot fire before LATCH; default = frozen OPEN (D-1,
    # attempt1/2 semantics unchanged).
    ap.add_argument("--descend_open_deg", type=float, default=Q5_OPEN_DEG)
    # Attempt2 (t3_grasp2): the D-2 marker distance follows the descend depth —
    # attempt1 measured the open-gripper/cylinder-top contact limit at TCP
    # z=top+4.4mm, so deeper grasp margins move the TCP-to-center distance past
    # the 0.030 constant and the marker would be structurally unfireable again.
    # Default keeps the attempt1 preregistered value.
    ap.add_argument("--marker_dist_m", type=float, default=GRASP_MARKER_DIST_M)
    ap.add_argument("--target_error_gate_m", type=float, default=0.003)
    ap.add_argument("--plan_tilt_gate_deg", type=float, default=5.0)
    ap.add_argument("--max_tcp_step_m", type=float, default=0.010)
    # Reverify wf_3cea04db MAJOR repair: per-waypoint IK joint trust region.
    # Without it the min-tilt selection hopped ~46 deg (wrist_p) at transit
    # wp002 and the velocity-clamped slew reached 9.90 mm TCP per control step
    # vs the 10 mm early-kill gate (1-6% margin, seed0_S1 worst).
    ap.add_argument("--waypoint_max_joint_dev_deg", type=float, default=12.0)
    # Same MAJOR, second layer: the trust region caps hop LENGTH but not the
    # instantaneous multi-joint slew speed |J*qdot|, so under the pessimistic
    # velocity-clamp bound (1.8 deg/control step per joint) a legitimate
    # approach-phase reorientation step can still sweep ~10 mm TCP. The
    # runaway gate is therefore phase-scoped: approach (object untouched,
    # >=40 mm away; drift/speed/tilt/upright gates unchanged) uses this
    # relaxed bound, descend/latch/hold/lift keep max_tcp_step_m. The
    # realistic effort-limited slew (2.5 N*m / damping 4 -> ~0.5 rad/s
    # equilibrium, ~6x below the velocity clamp) stays far under both.
    ap.add_argument("--transit_tcp_step_gate_m", type=float, default=0.020)
    ap.add_argument("--command_resample_fraction", type=float, default=0.80)
    ap.add_argument("--substep_steps", type=int, default=60)
    ap.add_argument("--object_drift_gate_m", type=float, default=0.006)
    ap.add_argument("--object_speed_gate_mps", type=float, default=0.080)
    ap.add_argument("--lift_speed_gate_mps", type=float, default=0.250)
    ap.add_argument("--gripper_error_gate_deg", type=float, default=0.75)
    ap.add_argument("--tilt_gate_deg", type=float, default=12.0)
    ap.add_argument("--min_upright_z_gate", type=float, default=0.95)
    ap.add_argument("--min_lift_follow_m", type=float, default=0.006)
    ap.add_argument("--settle_steps", type=int, default=2)
    ap.add_argument("--initial_settle_steps", type=int, default=30)
    ap.add_argument("--close_steps_per_angle", type=int, default=45)
    ap.add_argument("--hold_steps", type=int, default=30)
    # Audit MAJOR-c repair: worst-case budget settle 30 + approach 46wp*60 +
    # descend 5wp*60 + latch 11*45 + hold 30 + lift 2wp*60 = 3735 control steps
    # (seed0_S2 settled height is the measured worst of the four preregistered
    # poses, 3435-3735; preflight v2). 60 s -> 6000 steps keeps truncation
    # structurally unreachable.
    ap.add_argument("--episode_length_s", type=float, default=60.0)
    # Audit MAJOR-a repair: stall-aware close 'reached' (contact stall is the
    # contract-expected physics, t3_conversion_design.md D-3 expectation).
    ap.add_argument("--gripper_stall_rate_deg_per_step", type=float, default=0.02)
    ap.add_argument("--gripper_stall_min_steps", type=int, default=5)
    # Audit MAJOR-b repair: p7 diagnostic flag restored, default ON so the full
    # preregistered close band executes and close_records covers it.
    ap.add_argument(
        "--continue_close_after_grasped_until_angles_done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep sweeping later close_deg values after the marker fires "
        "(early-kill still stops immediately). Default ON (audit MAJOR-b).",
    )
    # Audit MAJOR-d repair: tag-parameterized artifacts + overwrite refusal.
    ap.add_argument("--tag", default="t3_grasp")
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    object_size = np.asarray(args.object_size_m, dtype=np.float64)
    if object_size.shape != (3,) or np.any(object_size <= 0.0):
        raise ValueError("object_size_m must be three positive dimensions")
    if abs(object_size[0] - object_size[1]) > 1.0e-9:
        raise ValueError("cylinder requires object_size_m[0] == object_size_m[1] (diameter)")
    # D-1: descending close sweep, starting at frozen OPEN, ending above the env
    # release threshold (0.4 rad = 22.92 deg) so the marker latch cannot self-release.
    if sorted(args.close_deg, reverse=True) != list(args.close_deg) or len(set(args.close_deg)) != len(args.close_deg):
        raise ValueError("close_deg values must be strictly descending (D-1 reversed convention)")
    if args.descend_open_deg > Q5_OPEN_DEG + 0.02:
        raise ValueError(f"descend_open_deg must not exceed frozen OPEN {Q5_OPEN_DEG:.4f} deg (D-1)")
    if args.descend_open_deg <= GRASP_MARKER_Q5_MAX_DEG:
        raise ValueError(
            f"descend_open_deg must stay above the marker q5 gate {GRASP_MARKER_Q5_MAX_DEG:.2f} deg "
            "(otherwise the D-2 marker could fire before LATCH)"
        )
    if abs(args.close_deg[0] - args.descend_open_deg) > 0.02:
        raise ValueError(
            f"first close_deg must equal descend_open_deg {args.descend_open_deg:.4f} deg "
            "(D-1 continuity: the sweep starts at the descend opening)"
        )
    if args.close_deg[-1] < 23.0:
        raise ValueError("last close_deg must stay >= 23.0 deg (> env 0.4 rad release threshold)")
    if args.approach_clearance_m <= args.grasp_surface_margin_m:
        raise ValueError("approach_clearance_m must be above grasp_surface_margin_m")
    if args.lift_delta_m <= 0.0:
        raise ValueError("lift_delta_m must be positive")
    if args.command_resample_fraction <= 0.0 or args.command_resample_fraction > 1.0:
        raise ValueError("command_resample_fraction must be in (0, 1]")
    if args.gripper_stall_rate_deg_per_step <= 0.0 or args.gripper_stall_min_steps < 1:
        raise ValueError("gripper stall detection constants must be positive")
    if args.waypoint_max_joint_dev_deg <= 0.0:
        raise ValueError("waypoint_max_joint_dev_deg must be positive")
    if args.transit_tcp_step_gate_m < args.max_tcp_step_m:
        raise ValueError("transit_tcp_step_gate_m must be >= max_tcp_step_m (equal disables the relaxation)")
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.tag):
        raise ValueError("--tag must match [A-Za-z0-9_]+")
    if args.marker_dist_m <= object_size[2] / 2.0 + args.grasp_surface_margin_m:
        raise ValueError(
            "marker_dist_m must exceed H/2 + grasp_surface_margin_m "
            "(otherwise the D-2 marker is structurally unfireable at the descend pose)"
        )

    # ---- Audit MAJOR-d guard: tag-derived artifact paths + overwrite refusal ----
    rrd_path = OUT_DIR / f"{args.tag}_timeline.rrd"
    rbl_path = OUT_DIR / f"{args.tag}_timeline.rbl"
    png_path = OUT_DIR / f"{args.tag}_inspection.png"
    validation_path = OUT_DIR / f"{args.tag}_rerun_validation.json"
    results_path = OUT_DIR / f"{args.tag}_results.json"
    csv_path = OUT_DIR / f"{args.tag}_steps.csv"
    existing = [p.name for p in (rrd_path, rbl_path, png_path, validation_path, results_path, csv_path) if p.exists()]
    if existing:
        print(
            f"[{LOG}] ABORT existing artifacts for --tag {args.tag}: {existing} — "
            "pass a fresh --tag (prior-evidence protection, audit MAJOR-d)",
            flush=True,
        )
        return 3

    # ---- D-3 / D-6 guards: frozen USD injection + sha pins, before any Isaac import
    if not ATTEMPT3_USD.exists() or not ATTEMPT3_PHYSICS_LAYER.exists():
        print(f"[{LOG}] G0B_T3_GRASP_VERDICT=USD_GUARD_FAIL missing attempt3 asset", flush=True)
        return 3
    root_sha = _sha256_file(ATTEMPT3_USD)
    physics_sha = _sha256_file(ATTEMPT3_PHYSICS_LAYER)
    if root_sha != ATTEMPT3_ROOT_SHA256 or physics_sha != ATTEMPT3_PHYSICS_SHA256:
        print(
            f"[{LOG}] G0B_T3_GRASP_VERDICT=USD_GUARD_FAIL sha mismatch root={root_sha} physics={physics_sha}",
            flush=True,
        )
        return 3
    # D-6: inject before any roarm_rl import (module reads env var at import time).
    # The effective-path /NHNHOME check runs post-import on cfg.robot.spawn.usd_path
    # (audit MINOR repair: the old same-string check here was vacuous).
    os.environ["ROARM_M3_USD_PATH"] = str(ATTEMPT3_USD)
    print(
        f"[{LOG}] usd_guard PASS path={ATTEMPT3_USD} root_sha={root_sha[:16]} physics_sha={physics_sha[:16]}",
        flush=True,
    )

    plan = _build_plan(args)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app
    _CLEANUP["sim_app"] = sim_app

    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import rerun as rr
    import rerun.blueprint as rrb
    import roarm_rl  # noqa: F401  registers env
    import torch
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg, _quat_rotate

    if str(rr.__version__) != RERUN_VERSION:
        print(f"[{LOG}] G0B_T3_GRASP_VERDICT=RERUN_VERSION_MISMATCH have={rr.__version__}", flush=True)
        _close_all()  # audit MAJOR-e repair: never leave Kit wedged on early exit
        return 3

    print(f"[{LOG}] G0b T3 p9 — D29xH50 top-center vertical physical grasp probe", flush=True)
    print(
        f"[{LOG}] diagnostic_only=YES isaac_run=YES env_default_edits=NO chain_defaults_edits=NO "
        "training=NO constraint_prim_insertion=NO surface_gripper=NO attached_transport=NO "
        "transport_target=NO release_marker=NO hidden_kinematic_posewrite_allowed=NO "
        "q5_convention=LARGE_IS_OPEN(D-1) marker=probe_patched(D-2) collision_asset=attempt3_frozen(D-3)",
        flush=True,
    )
    print(
        f"[{LOG}] object cylinder D={object_size[0] * 1000:.1f}mm H={object_size[2] * 1000:.1f}mm "
        f"mass_kg={args.object_mass_kg:.5f} friction_static={args.static_friction:.2f} "
        f"friction_dynamic={args.dynamic_friction:.2f} restitution={args.restitution:.2f} "
        "friction_provenance=PREREGISTERED_ASSUMPTION_NOT_MEASURED(t3_mass_friction_contract)",
        flush=True,
    )
    print(
        f"[{LOG}] gates target_error_gate_m={args.target_error_gate_m:.6f} plan_tilt_gate_deg={args.plan_tilt_gate_deg:.2f} "
        f"max_tcp_step_m={args.max_tcp_step_m:.6f} object_drift_gate_m={args.object_drift_gate_m:.6f} "
        f"object_speed_gate_mps={args.object_speed_gate_mps:.6f} lift_speed_gate_mps={args.lift_speed_gate_mps:.6f} "
        f"gripper_error_gate_deg={args.gripper_error_gate_deg:.3f} tilt_gate_deg={args.tilt_gate_deg:.2f} "
        f"min_upright_z_gate={args.min_upright_z_gate:.3f} min_lift_follow_m={args.min_lift_follow_m:.6f} "
        f"marker_dist_m={args.marker_dist_m:.3f} marker_q5_max_deg={GRASP_MARKER_Q5_MAX_DEG:.2f} "
        f"close_sweep_deg={','.join(f'{x:.2f}' for x in args.close_deg)} episode_length_s={args.episode_length_s:.1f}",
        flush=True,
    )
    print(
        f"[{LOG}] plan pose={plan.label} center={_fmt_xyz(plan.center)} world_grasp={_fmt_xyz(plan.world_grasp)} "
        f"approach_tcp={_fmt_xyz(plan.approach_tcp)} descend_tcp={_fmt_xyz(plan.descend_tcp)} "
        f"lift_tcp={_fmt_xyz(plan.lift_tcp)} "
        f"ik_ok={_yes(plan.approach_ik_ok and plan.descend_ik_ok and plan.lift_ik_ok)} "
        f"ik_err_mm=({plan.approach_ik_err_mm:.3f},{plan.descend_ik_err_mm:.3f},{plan.lift_ik_err_mm:.3f}) "
        f"ik_tilt_deg=({plan.approach_tilt_deg:.3f},{plan.descend_tilt_deg:.3f},{plan.lift_tilt_deg:.3f}) "
        f"q_descend_deg=([{plan.q_descend_deg[0]:+.3f},{plan.q_descend_deg[1]:+.3f},"
        f"{plan.q_descend_deg[2]:+.3f},{plan.q_descend_deg[3]:+.3f},{plan.q_descend_deg[4]:+.3f},"
        f"{plan.q_descend_deg[5]:+.3f}])",
        flush=True,
    )

    # ---- D-4 env config -------------------------------------------------------
    cfg = RoArmStackEnvCfg()
    # D-6 effective guard (audit MINOR repair): assert the injection actually
    # reached the composed config — catches stale-module / import-order drift.
    effective_usd = str(cfg.robot.spawn.usd_path)
    if effective_usd != str(ATTEMPT3_USD) or "/NHNHOME" in effective_usd:
        print(
            f"[{LOG}] G0B_T3_GRASP_VERDICT=USD_GUARD_FAIL effective usd_path={effective_usd} "
            "(expected attempt3 injection; HARD RULE #27)",
            flush=True,
        )
        _close_all()
        return 3
    print(f"[{LOG}] usd_effective PASS cfg.robot.spawn.usd_path={effective_usd}", flush=True)
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
    cfg.sponge.spawn = sim_utils.CylinderCfg(
        radius=float(object_size[0]) / 2.0,
        height=float(object_size[2]),
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            max_angular_velocity=10.0,
            max_linear_velocity=10.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=args.object_mass_kg),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=args.static_friction,
            dynamic_friction=args.dynamic_friction,
            restitution=args.restitution,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.62, 0.38), metallic=0.0),
    )
    cfg.sponge.init_state.pos = tuple(float(x) for x in plan.center)
    cfg.sponge.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    _CLEANUP["env"] = env
    base_env = env.unwrapped
    device = base_env.device
    null_action = torch.zeros((1, 6), device=device, dtype=torch.float32)
    control_dt = float(cfg.sim.dt * cfg.decimation)

    # ---- D-3 stage audit: hard-fail before any physics step -------------------
    audit_pass, body_checks = _audit_collision_bodies(base_env)
    print(f"[{LOG}] usd_stage_audit pass={_yes(audit_pass)} body_checks={json.dumps(body_checks)}", flush=True)
    if not audit_pass:
        print(f"[{LOG}] G0B_T3_GRASP_VERDICT=USD_AUDIT_FAIL", flush=True)
        _close_all()
        return 3

    # ---- material recording duty (contract section 3) -------------------------
    materials = _material_report(base_env)
    print(
        f"[{LOG}] materials cfg_sim=1.0/1.0/0.0(multiply) cfg_terrain=1.0/1.0/0.0(multiply) "
        f"cfg_object={args.static_friction:.2f}/{args.dynamic_friction:.2f}/{args.restitution:.2f}"
        "(combine_mode_cfg_unspecified) robot_usd=env_asset_default(no_env_override)",
        flush=True,
    )
    for row in materials:
        print(f"[{LOG}] material_prim {json.dumps(row)}", flush=True)

    # ---- D-2 marker monkeypatch + D-8 integrity devices -----------------------
    marker_q5_max_rad = math.radians(GRASP_MARKER_Q5_MAX_DEG)

    def patched_grasp_condition() -> "torch.Tensor":
        d = torch.norm(base_env._sponge_pos_w - base_env._tcp_pos_w, p=2, dim=-1)
        gripper_q = base_env._robot.data.joint_pos[:, base_env.gripper_joint_idx]
        return (d < args.marker_dist_m) & (gripper_q <= marker_q5_max_rad)

    base_env._grasp_condition = patched_grasp_condition

    attach_stats = {"attach_calls": 0, "posewrite_calls": 0}
    original_set_joint_position_target = base_env._robot.set_joint_position_target
    watch = {"active": False, "target": None, "calls": 0, "max_diff": 0.0}

    def marker_only_attach() -> None:
        attach_stats["attach_calls"] += 1

    def watched_set_joint_position_target(target, *a, **kw):
        if watch["active"] and watch["target"] is not None:
            arr = target.detach().cpu().numpy().astype(np.float64)
            watch["calls"] += 1
            watch["max_diff"] = max(watch["max_diff"], float(np.max(np.abs(arr - watch["target"]))))
        return original_set_joint_position_target(target, *a, **kw)

    base_env._update_grasp_attach = marker_only_attach
    base_env._robot.set_joint_position_target = watched_set_joint_position_target

    def step_once() -> bool:
        out = env.step(null_action)
        if len(out) == 5:
            _obs, _rew, terminated, truncated, _extras = out
            return bool((terminated | truncated).any().item())
        _obs, _rew, dones, _extras = out
        return bool(dones.any().item())

    def fresh_tcp_local() -> np.ndarray:
        link5_pos = base_env._robot.data.body_pos_w[:1, base_env.link5_idx]
        link5_quat = base_env._robot.data.body_quat_w[:1, base_env.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, base_env._tcp_local.expand(1, 3))
        tcp = link5_pos + tcp_offset_world
        return (tcp[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def object_local() -> np.ndarray:
        return (base_env._sponge.data.root_pos_w[0] - base_env.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)

    def object_quat() -> np.ndarray:
        return base_env._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)

    def object_vel6() -> np.ndarray:
        return base_env._sponge.data.root_vel_w[0].detach().cpu().numpy().astype(np.float64)

    def gripper_q_deg() -> float:
        return math.degrees(float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].item()))

    def marker_now() -> bool:
        return bool(base_env._grasped[0].detach().cpu().item())

    def write_object_pose() -> None:
        pose = torch.tensor(
            [[plan.center[0], plan.center[1], plan.center[2], 1.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=torch.float32,
        )
        pose[:, 0:3] += base_env.scene.env_origins[:1]
        base_env._sponge.write_root_pose_to_sim(pose)
        base_env._sponge.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        base_env.scene.write_data_to_sim()
        base_env.scene.update(base_env.sim.get_physics_dt())

    # ---- D-7 Rerun: sink attached before the first logged step ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows: list[list[Any]] = []
    rerun_ctx: dict[str, Any] = {"rec": None, "global_step": 0}

    def log_step(
        phase: str,
        label: str,
        tcp: np.ndarray,
        obj: np.ndarray,
        quat: np.ndarray,
        q5_actual_deg: float,
        q5_cmd_deg: float,
        target_error_m: float,
        drift_m: float,
        speed_mps: float,
        tilt_deg_v: float,
        upright_z: float,
        marker: bool,
    ) -> None:
        rec = rerun_ctx["rec"]
        step = rerun_ctx["global_step"]
        rec.reset_time()
        rec.set_time("physics_step", sequence=step)
        rec.set_time("sim_time_s", timestamp=step * control_dt)
        rec.log("world/tcp", rr.Points3D([list(map(float, tcp))], colors=[[60, 170, 255]], radii=0.003))
        rec.log("world/object", rr.Points3D([list(map(float, obj))], colors=[[210, 170, 110]], radii=0.005))
        rec.log("plots/q5_deg", rr.Scalars(float(q5_actual_deg)))
        rec.log("plots/q5_cmd_deg", rr.Scalars(float(q5_cmd_deg)))
        rec.log("plots/target_error_mm", rr.Scalars(float(target_error_m) * 1000.0))
        rec.log("plots/object_drift_mm", rr.Scalars(float(drift_m) * 1000.0))
        rec.log("plots/object_speed_mps", rr.Scalars(float(speed_mps)))
        rec.log("plots/tilt_deg", rr.Scalars(float(tilt_deg_v)))
        rec.log("plots/upright_z", rr.Scalars(float(upright_z)))
        rec.log("plots/marker", rr.Scalars(1.0 if marker else 0.0))
        csv_rows.append(
            [
                step,
                round(step * control_dt, 4),
                phase,
                label,
                *[round(float(v), 6) for v in tcp],
                *[round(float(v), 6) for v in obj],
                *[round(float(v), 6) for v in quat],
                round(q5_actual_deg, 4),
                round(q5_cmd_deg, 4),
                round(target_error_m, 6),
                round(drift_m, 6),
                round(speed_mps, 6),
                round(tilt_deg_v, 4),
                round(upright_z, 6),
                int(marker),
                attach_stats["attach_calls"],
                attach_stats["posewrite_calls"],
            ]
        )
        rerun_ctx["global_step"] = step + 1

    def log_event(rec, text: str) -> None:
        rec.log("events/phase", rr.TextLog(text, level=rr.TextLogLevel.INFO))

    # Reverify wf_3cea04db MINOR: recording identity follows --tag so rrd files
    # from different legs opened in one viewer never merge into one recording.
    app_id = f"roarm_g0b_{args.tag}"
    with rr.RecordingStream(app_id, recording_id=f"g0b_d420_{args.tag}", make_default=False, send_properties=True) as rec:
        rec.save(str(rrd_path), write_footer=True)
        rerun_ctx["rec"] = rec

        env.reset()
        home_deg = np.array([*HOME_ARM_DEG, 0.0, args.descend_open_deg], dtype=np.float64)
        home_rad = torch.tensor(np.radians(home_deg), device=device, dtype=torch.float32).unsqueeze(0)
        base_env._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
        base_env._robot.set_joint_position_target(home_rad)
        base_env.robot_dof_targets[:] = home_rad
        base_env._grasped[:] = False
        base_env._was_grasped[:] = False
        write_object_pose()

        original_write_root_pose_to_sim = base_env._sponge.write_root_pose_to_sim
        posewrite_watch = {"active": False}

        def watched_write_root_pose_to_sim(*a, **kw):
            if posewrite_watch["active"]:
                attach_stats["posewrite_calls"] += 1
            return original_write_root_pose_to_sim(*a, **kw)

        base_env._sponge.write_root_pose_to_sim = watched_write_root_pose_to_sim

        total_sim_steps = 0
        episode_done = False
        nan_seen = False
        log_event(rec, "phase=settle begin")
        for _ in range(args.initial_settle_steps):
            episode_done |= step_once()
            total_sim_steps += 1
            tcp_s = fresh_tcp_local()
            obj_s = object_local()
            quat_s = object_quat()
            m_s = _object_pose_metrics(obj_s, quat_s, object_size)
            log_step(
                "settle", "initial_settle", tcp_s, obj_s, quat_s, gripper_q_deg(), args.descend_open_deg,
                _norm(tcp_s - plan.approach_tcp), _norm(obj_s - plan.center),
                _norm(object_vel6()[:3]), m_s["tilt_deg"], m_s["up_z"], marker_now(),
            )

        initial_object = object_local()
        initial_quat = object_quat()
        initial_metrics = _object_pose_metrics(initial_object, initial_quat, object_size)
        print(
            f"[{LOG}] initial home_tcp={_fmt_xyz(fresh_tcp_local())} object_pos={_fmt_xyz(initial_object)} "
            f"object_quat_wxyz={_fmt_quat(initial_quat)} object_top_z_m={initial_metrics['oriented_top_z_m']:.6f} "
            f"upright_z={initial_metrics['up_z']:.6f} tilt_deg={initial_metrics['tilt_deg']:.3f} "
            f"gripper_q_deg={gripper_q_deg():.3f}",
            flush=True,
        )
        if _norm(initial_object - plan.center) > args.target_error_gate_m:
            old_center = plan.center.copy()
            plan = _build_plan_from_center(args, initial_object.copy(), f"{plan.label}_settled_pose")
            print(
                f"[{LOG}] settled_pose_replan=YES requested_center={_fmt_xyz(old_center)} "
                f"settled_center={_fmt_xyz(initial_object)} settled_top_z_m={initial_metrics['oriented_top_z_m']:.6f} "
                f"updated_world_grasp={_fmt_xyz(plan.world_grasp)} updated_approach_tcp={_fmt_xyz(plan.approach_tcp)} "
                f"updated_descend_tcp={_fmt_xyz(plan.descend_tcp)} updated_lift_tcp={_fmt_xyz(plan.lift_tcp)} "
                f"updated_ik_ok={_yes(plan.approach_ik_ok and plan.descend_ik_ok and plan.lift_ik_ok)} "
                f"updated_ik_err_mm=({plan.approach_ik_err_mm:.3f},{plan.descend_ik_err_mm:.3f},{plan.lift_ik_err_mm:.3f}) "
                f"updated_ik_tilt_deg=({plan.approach_tilt_deg:.3f},{plan.descend_tilt_deg:.3f},{plan.lift_tilt_deg:.3f})",
                flush=True,
            )
        else:
            print(f"[{LOG}] settled_pose_replan=NO", flush=True)
        posewrite_watch["active"] = True

        def run_to_q(label: str, q_deg: np.ndarray, target_tcp: np.ndarray, max_steps: int, phase: str) -> StepResult:
            nonlocal total_sim_steps, episode_done, nan_seen
            target_rad_np = np.radians(q_deg)
            target_rad = torch.tensor(target_rad_np, device=device, dtype=torch.float32).unsqueeze(0)
            start_object = object_local()
            start_lift_ref = start_object.copy()
            prev_tcp = fresh_tcp_local()
            settle_count = 0
            reached = False
            early_kill = False
            steps_used = 0
            final_error = float("inf")
            prev_gq_deg: float | None = None
            stall_count = 0
            gripper_stalled = False
            max_tcp_step = 0.0
            max_drift = 0.0
            max_speed = 0.0
            max_tilt = 0.0
            min_upright = 1.0
            grasped_seen = marker_now()
            attach_start = attach_stats["attach_calls"]
            posewrite_start = attach_stats["posewrite_calls"]
            watch["active"] = True
            watch["target"] = target_rad_np
            watch["calls"] = 0
            watch["max_diff"] = 0.0
            for step_idx in range(1, max_steps + 1):
                base_env.robot_dof_targets[:] = target_rad
                done = step_once()
                total_sim_steps += 1
                steps_used = step_idx
                tcp = fresh_tcp_local()
                obj = object_local()
                quat = object_quat()
                vel = object_vel6()
                metrics = _object_pose_metrics(obj, quat, object_size)
                gq_deg = gripper_q_deg()
                gripper_err_deg = abs(gq_deg - math.degrees(float(target_rad_np[5])))
                # Audit MAJOR-a: contact-stall detection (jaw blocked short of command).
                if prev_gq_deg is not None and abs(gq_deg - prev_gq_deg) < args.gripper_stall_rate_deg_per_step:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_gq_deg = gq_deg
                target_error = _norm(tcp - target_tcp)
                tcp_step = _norm(tcp - prev_tcp)
                drift = _norm(obj - start_object)
                speed = _norm(vel[:3])
                tilt = float(metrics["tilt_deg"])
                upright = float(metrics["up_z"])
                max_tcp_step = max(max_tcp_step, tcp_step)
                max_drift = max(max_drift, drift)
                max_speed = max(max_speed, speed)
                max_tilt = max(max_tilt, tilt)
                min_upright = min(min_upright, upright)
                final_error = target_error
                grasped_seen = grasped_seen or marker_now()
                log_step(
                    phase, label, tcp, obj, quat, gq_deg, math.degrees(float(target_rad_np[5])),
                    target_error, drift, speed, tilt, upright, marker_now(),
                )
                if not np.isfinite(tcp).all() or not np.isfinite(obj).all() or not math.isfinite(target_error):
                    nan_seen = True
                episode_done |= done
                speed_gate = args.lift_speed_gate_mps if phase == "lift" else args.object_speed_gate_mps
                drift_gate = args.object_drift_gate_m
                if phase == "lift":
                    drift_gate = max(args.object_drift_gate_m, args.lift_delta_m + 0.010)
                # Reverify wf_3cea04db MAJOR: approach transit reorientation can
                # legitimately sweep ~10 mm TCP in one control step under the
                # velocity-clamp bound; the runaway gate is phase-scoped there.
                tcp_step_gate = args.transit_tcp_step_gate_m if phase == "approach" else args.max_tcp_step_m
                early_kill = (
                    tcp_step > tcp_step_gate
                    or speed > speed_gate
                    or tilt > args.tilt_gate_deg
                    or upright < args.min_upright_z_gate
                    or drift > drift_gate
                    or done
                    or nan_seen
                )
                reached = target_error <= args.target_error_gate_m
                if phase == "close":
                    gripper_ok = gripper_err_deg <= args.gripper_error_gate_deg
                    if not gripper_ok and stall_count >= args.gripper_stall_min_steps:
                        # Contact stall = contract-expected physics (D-3), counts as
                        # reached; recorded so LATCH attribution stays honest.
                        gripper_ok = True
                        gripper_stalled = True
                    reached = reached and gripper_ok
                settle_count = settle_count + 1 if reached else 0
                if step_idx <= 3 or step_idx == max_steps or reached or early_kill or (args.log_every > 0 and step_idx % args.log_every == 0):
                    print(
                        f"[{LOG}] event label={label} phase={phase} step={step_idx:03d} "
                        f"target_tcp={_fmt_xyz(target_tcp)} fresh_tcp={_fmt_xyz(tcp)} target_error_m={target_error:.6f} "
                        f"tcp_step_m={tcp_step:.6f} object_pos={_fmt_xyz(obj)} object_drift_m={drift:.6f} "
                        f"object_speed_mps={speed:.6f} upright_z={upright:.6f} tilt_deg={tilt:.3f} "
                        f"gripper_q_deg={gq_deg:.3f} gripper_err_deg={gripper_err_deg:.3f} "
                        f"gripper_stalled={_yes(gripper_stalled)} "
                        f"_grasped_marker={_yes(marker_now())} "
                        f"attach_calls_total={attach_stats['attach_calls']} posewrite_calls_total={attach_stats['posewrite_calls']} "
                        f"set_target_seen={_yes(watch['calls'] > 0 and watch['max_diff'] <= 1.0e-5)} "
                        f"set_max_diff_rad={watch['max_diff']:.8f} reached={_yes(reached)} early_kill={_yes(early_kill)}",
                        flush=True,
                    )
                prev_tcp = tcp
                if early_kill or settle_count >= args.settle_steps:
                    break
            watch["active"] = False
            obj_end = object_local()
            return StepResult(
                label=label,
                reached=reached,
                steps=steps_used,
                final_target_error_m=final_error,
                max_tcp_step_m=max_tcp_step,
                max_object_drift_m=max_drift,
                max_object_speed_mps=max_speed,
                max_tilt_deg=max_tilt,
                min_upright_z=min_upright,
                object_follow_delta_m=float(obj_end[2] - start_lift_ref[2]),
                grasped_seen=grasped_seen,
                attach_calls=attach_stats["attach_calls"] - attach_start,
                posewrite_calls=attach_stats["posewrite_calls"] - posewrite_start,
                early_kill=early_kill,
                gripper_stalled=gripper_stalled,
            )

        def resampled_waypoints(start_tcp: np.ndarray, end_tcp: np.ndarray) -> list[np.ndarray]:
            delta = np.asarray(end_tcp, dtype=np.float64) - np.asarray(start_tcp, dtype=np.float64)
            gap = _norm(delta)
            max_cmd_gap = args.max_tcp_step_m * args.command_resample_fraction
            count = max(1, int(math.ceil(gap / max_cmd_gap)))
            return [np.asarray(start_tcp, dtype=np.float64) + delta * (i / count) for i in range(1, count + 1)]

        def run_resampled_path(
            label: str,
            start_tcp: np.ndarray,
            end_tcp: np.ndarray,
            seed_q: np.ndarray,
            gripper_deg: float,
            phase: str,
            max_steps: int,
            vertical_scope: str = "all",
        ) -> tuple[StepResult | None, np.ndarray, bool]:
            # vertical_scope (audit FATAL-2 repair): "all" gates every waypoint on
            # pos AND tilt (descend/lift vertical corridor); "arrival" gates transit
            # waypoints on position only and enforces tilt at the final waypoint
            # (the T2-verified approach arrival).
            if vertical_scope not in ("all", "arrival"):
                raise ValueError(f"vertical_scope must be 'all' or 'arrival', got {vertical_scope!r}")
            q_seed = seed_q.copy()
            final_result: StepResult | None = None
            # Audit FATAL-1 repair: follow is anchored at the PHASE start object z,
            # then overwritten onto each waypoint result (phase-cumulative, not
            # per-waypoint segment).
            phase_start_obj_z = float(object_local()[2])
            waypoints = resampled_waypoints(start_tcp, end_tcp)
            print(
                f"[{LOG}] path_plan label={label} start_tcp={_fmt_xyz(start_tcp)} "
                f"end_tcp={_fmt_xyz(end_tcp)} waypoints={len(waypoints)} "
                f"max_command_gap_m={args.max_tcp_step_m * args.command_resample_fraction:.6f} "
                f"vertical_scope={vertical_scope} phase_start_obj_z={phase_start_obj_z:.6f}",
                flush=True,
            )
            for idx, waypoint in enumerate(waypoints, start=1):
                require_tilt = vertical_scope == "all" or idx == len(waypoints)
                q_step, ik_ok, ik_err_mm, ik_tilt = _solve_q_vertical(
                    waypoint,
                    q_seed,
                    gripper_deg,
                    args,
                    require_tilt=require_tilt,
                    max_dev_from_seed_deg=args.waypoint_max_joint_dev_deg,
                )
                seed_dev_deg = float(np.max(np.abs(np.asarray(q_step[:4]) - np.asarray(q_seed[:4], dtype=np.float64))))
                print(
                    f"[{LOG}] path_waypoint label={label} index={idx:03d}/{len(waypoints):03d} "
                    f"target_tcp={_fmt_xyz(waypoint)} ik_ok={_yes(ik_ok)} ik_err_mm={ik_err_mm:.3f} "
                    f"ik_tilt_deg={ik_tilt:.3f} vertical_gate={_yes(require_tilt)} "
                    f"seed_dev_deg={seed_dev_deg:.3f}",
                    flush=True,
                )
                if not ik_ok:
                    return final_result, q_seed, False
                final_result = run_to_q(f"{label}_wp{idx:03d}", q_step, waypoint, max_steps, phase)
                final_result.object_follow_delta_m = float(object_local()[2] - phase_start_obj_z)
                q_seed = q_step
                if not final_result.reached or final_result.early_kill:
                    return final_result, q_seed, False
            return final_result, q_seed, True

        approach_result: StepResult | None = None
        descend_result: StepResult | None = None
        latch_result: StepResult | None = None
        hold_result: StepResult | None = None
        lift_result: StepResult | None = None
        close_records: list[dict[str, Any]] = []

        current_seed_q = np.array([*HOME_ARM_DEG, 0.0, args.descend_open_deg], dtype=np.float64)
        log_event(rec, "phase=approach begin")
        if plan.approach_ik_ok and plan.descend_ik_ok and plan.lift_ik_ok:
            approach_result, current_seed_q, approach_path_ok = run_resampled_path(
                "approach_open",
                fresh_tcp_local(),
                plan.approach_tcp,
                current_seed_q,
                args.descend_open_deg,
                "approach",
                args.substep_steps,
                vertical_scope="arrival",
            )
        else:
            approach_path_ok = False
        if approach_path_ok and approach_result and approach_result.reached and not approach_result.early_kill:
            log_event(rec, "phase=descend begin")
            descend_result, current_seed_q, descend_path_ok = run_resampled_path(
                "descend_open",
                fresh_tcp_local(),
                plan.descend_tcp,
                current_seed_q,
                args.descend_open_deg,
                "descend",
                args.substep_steps,
                vertical_scope="all",
            )
        else:
            descend_path_ok = False
        q_close = current_seed_q.copy()
        if descend_path_ok and descend_result and descend_result.reached and not descend_result.early_kill:
            log_event(rec, "phase=latch begin (descending q5 sweep, D-1)")
            for close_deg in args.close_deg:
                q_close[5] = close_deg
                result = run_to_q(f"close_{close_deg:.2f}deg", q_close, plan.descend_tcp, args.close_steps_per_angle, "close")
                print(
                    f"[{LOG}] close_result angle_deg={close_deg:.2f} reached={_yes(result.reached)} "
                    f"gripper_stalled={_yes(result.gripper_stalled)} "
                    f"grasped_seen={_yes(result.grasped_seen)} final_target_error_m={result.final_target_error_m:.6f} "
                    f"object_drift_m={result.max_object_drift_m:.6f} object_speed_mps={result.max_object_speed_mps:.6f} "
                    f"tilt_deg={result.max_tilt_deg:.3f} attach_calls={result.attach_calls} posewrite_calls={result.posewrite_calls} "
                    f"early_kill={_yes(result.early_kill)}",
                    flush=True,
                )
                close_records.append(
                    {
                        "angle_deg": close_deg,
                        "reached": result.reached,
                        "gripper_stalled": result.gripper_stalled,
                        "grasped_seen": result.grasped_seen,
                        "early_kill": result.early_kill,
                        "max_object_drift_m": result.max_object_drift_m,
                        "max_tilt_deg": result.max_tilt_deg,
                    }
                )
                latch_result = result
                if result.early_kill:
                    break
                # Audit MAJOR-b repair: default is to keep sweeping the full
                # preregistered band after marker fire (p7 flag, default ON here).
                if result.grasped_seen and not args.continue_close_after_grasped_until_angles_done:
                    break
        if latch_result is not None and close_records:
            # With the full-band sweep (MAJOR-b) the marker may fire at an earlier
            # angle than the last one; "marker seen during LATCH" is the p7-equivalent
            # phase semantics (per-angle truth stays in close_records).
            latch_grasped_any = any(rec["grasped_seen"] for rec in close_records)
            if latch_grasped_any and not latch_result.grasped_seen:
                print(f"[{LOG}] latch_grasped_any=YES (earlier-angle marker fire carried to phase verdict)", flush=True)
            latch_result.grasped_seen = latch_result.grasped_seen or latch_grasped_any
        if latch_result and latch_result.reached and latch_result.grasped_seen and not latch_result.early_kill:
            log_event(rec, "phase=hold begin")
            q_hold = q_close.copy()
            q_hold[5] = args.close_deg[-1]
            hold_result = run_to_q("stationary_hold_closed", q_hold, plan.descend_tcp, args.hold_steps, "hold")
        lift_path_ok = True
        if hold_result and hold_result.reached and not hold_result.early_kill:
            log_event(rec, "phase=lift begin")
            lift_result, current_seed_q, lift_path_ok = run_resampled_path(
                "tiny_lift_closed_10mm",
                fresh_tcp_local(),
                plan.lift_tcp,
                q_hold,
                args.close_deg[-1],
                "lift",
                args.substep_steps,
                vertical_scope="all",
            )

        results = [r for r in [approach_result, descend_result, latch_result, hold_result, lift_result] if r is not None]
        max_target_error = max((r.final_target_error_m for r in results), default=float("inf"))
        max_tcp_step = max((r.max_tcp_step_m for r in results), default=float("inf"))
        max_drift = max((r.max_object_drift_m for r in results), default=float("inf"))
        max_speed = max((r.max_object_speed_mps for r in results), default=float("inf"))
        max_tilt = max((r.max_tilt_deg for r in results), default=float("inf"))
        min_upright = min((r.min_upright_z for r in results), default=float("nan"))
        total_attach_calls = attach_stats["attach_calls"]
        total_posewrite_calls = attach_stats["posewrite_calls"]
        lift_follow = 0.0 if lift_result is None else lift_result.object_follow_delta_m
        hidden_posewrite_ok = total_posewrite_calls == 0
        verdict = _verdict(
            plan, approach_result, descend_result, latch_result, hold_result, lift_result, args,
            lift_path_ok=lift_path_ok,
        )

        print(
            f"[{LOG}] aggregate verdict={verdict} events_done={len(results)}/5 "
            f"max_target_error_m={max_target_error:.6f} max_tcp_step_m={max_tcp_step:.6f} "
            f"max_object_drift_m={max_drift:.6f} max_object_speed_mps={max_speed:.6f} "
            f"max_tilt_deg={max_tilt:.3f} min_upright_z={min_upright:.6f} "
            f"lift_follow_delta_m={lift_follow:.6f} attach_calls={total_attach_calls} "
            f"posewrite_calls={total_posewrite_calls} hidden_kinematic_posewrite_artifact={_yes(not hidden_posewrite_ok)} "
            f"episode_done={_yes(episode_done)} nan_seen={_yes(nan_seen)} total_sim_steps={total_sim_steps}",
            flush=True,
        )
        print(f"[{LOG}] G0B_T3_GRASP_VERDICT={verdict}", flush=True)
        if episode_done:
            # Audit MAJOR-c: make truncation loud — a budget-truncated episode
            # auto-resets the env, so phase verdicts may be budget-attributed.
            print(
                f"[{LOG}] episode_truncated=YES episode_length_s={args.episode_length_s:.1f} "
                "— phase *_FAIL may be budget-, not physics-attributed",
                flush=True,
            )
        log_event(rec, f"verdict={verdict}")

        # ---- static context + summary (D-7) -----------------------------------
        settled_center = object_local()
        cx, cy = float(settled_center[0]), float(settled_center[1])
        top_z = float(plan.world_grasp[2])
        circle_top, circle_bot = [], []
        for k in range(33):
            a = 2.0 * math.pi * k / 32
            px, py = cx + object_size[0] / 2.0 * math.cos(a), cy + object_size[0] / 2.0 * math.sin(a)
            circle_top.append([px, py, top_z])
            circle_bot.append([px, py, top_z - float(object_size[2])])
        rec.log(
            "world/cylinder",
            rr.LineStrips3D([circle_top, circle_bot], colors=[[210, 170, 110], [210, 170, 110]], radii=0.001),
            static=True,
        )
        rec.log(
            "world/targets",
            rr.Points3D(
                [list(map(float, plan.approach_tcp)), list(map(float, plan.descend_tcp)), list(map(float, plan.lift_tcp))],
                colors=[[60, 170, 255], [40, 200, 80], [230, 210, 40]],
                radii=0.004,
                labels=["approach", "descend", "lift"],
            ),
            static=True,
        )
        summary_md = (
            f"# G0b T3 p9 — D29xH50 top-center vertical grasp (case g0b_d420)\n\n"
            f"- verdict: **{verdict}**\n"
            f"- pose: {plan.label} settled_center=({cx:+.4f},{cy:+.4f})\n"
            f"- q5 convention: LARGE=OPEN (D-1); sweep {args.close_deg[0]:.2f}->{args.close_deg[-1]:.2f} deg\n"
            f"- marker (D-2): dist<{args.marker_dist_m} m AND q5<={GRASP_MARKER_Q5_MAX_DEG} deg; "
            f"grasp evidence = LIFT follow >= {args.min_lift_follow_m * 1000:.0f} mm\n"
            f"- collision asset (D-3): attempt3 64+64 frozen, root {ATTEMPT3_ROOT_SHA256[:16]}, "
            f"physics {ATTEMPT3_PHYSICS_SHA256[:16]}, stage audit pass={audit_pass}\n"
            f"- mass/friction (D-4): {args.object_mass_kg:.5f} kg, mu_s {args.static_friction:.2f} / "
            f"mu_d {args.dynamic_friction:.2f} / rest {args.restitution:.2f} (preregistered assumption)\n"
            f"- gates: drift {args.object_drift_gate_m * 1000:.0f}mm / speed {args.object_speed_gate_mps} / "
            f"tilt {args.tilt_gate_deg} deg / upright {args.min_upright_z_gate} / target {args.target_error_gate_m * 1000:.0f}mm\n"
            f"- lift_follow_delta_m: {lift_follow:.6f} / attach_calls {total_attach_calls} / "
            f"posewrite_calls {total_posewrite_calls}\n"
            f"- contact-force arrows omitted-justified: no gate consumes contact forces (D-7)\n"
            f"- audit wf_78b1adfd repairs: cumulative lift follow / transit pos-only gate / "
            f"stall-aware close / full-band sweep / episode {args.episode_length_s:.0f}s / tag={args.tag}\n"
            f"- reverify wf_3cea04db repairs: transit selection band {TRANSIT_POS_BAND_MM}mm + "
            f"joint trust region {args.waypoint_max_joint_dev_deg:.0f}+{TRANSIT_POLISH_DEV_DEG:.0f}deg "
            f"(bias stage + transit polish) / tag-scoped recording / "
            f"lift path_ok in verdict / hardened _close_all\n"
            f"- Float32 spatial copies are inspection evidence only; authority = stdout/JSON/CSV (D341)\n"
        )
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN), static=True)
        rec.log(
            "metadata/materials",
            rr.TextDocument(
                "# Runtime material record (contract section 3)\n\n```json\n"
                + json.dumps(materials, indent=2, default=str)
                + "\n```\n",
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )

        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run", name="1 | T3 verdict + contract"),
                    rrb.Spatial3DView(origin="/", contents=["/world/**"], name="2 | targets vs executed"),
                    rrb.TextLogView(origin="/events/phase", contents="/events/phase/**", name="3 | phase events"),
                    column_shares=[0.30, 0.45, 0.25],
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots", contents=["/plots/q5_deg/**", "/plots/q5_cmd_deg/**"], name="4 | q5 actual vs command (deg)"),
                    rrb.TimeSeriesView(origin="/plots", contents=["/plots/target_error_mm/**", "/plots/object_drift_mm/**"], name="5 | TCP error / object drift (mm)"),
                    rrb.TimeSeriesView(
                        origin="/plots",
                        contents=["/plots/tilt_deg/**", "/plots/upright_z/**", "/plots/object_speed_mps/**", "/plots/marker/**"],
                        name="6 | object stability + marker",
                    ),
                ),
                row_shares=[0.55, 0.45],
            ),
            auto_layout=False,
            auto_views=False,
            collapse_panels=True,
        )
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=30.0)
    blueprint.save(app_id, str(rbl_path))

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "physics_step", "sim_time_s", "phase", "label",
                "tcp_x", "tcp_y", "tcp_z", "obj_x", "obj_y", "obj_z",
                "quat_w", "quat_x", "quat_y", "quat_z",
                "q5_deg", "q5_cmd_deg", "target_error_m", "object_drift_m",
                "object_speed_mps", "tilt_deg", "upright_z", "marker",
                "attach_calls_total", "posewrite_calls_total",
            ]
        )
        w.writerows(csv_rows)

    def _step_result_dict(r: StepResult | None) -> dict[str, Any] | None:
        if r is None:
            return None
        return {
            "label": r.label,
            "reached": r.reached,
            "gripper_stalled": r.gripper_stalled,
            "steps": r.steps,
            "final_target_error_m": r.final_target_error_m,
            "max_tcp_step_m": r.max_tcp_step_m,
            "max_object_drift_m": r.max_object_drift_m,
            "max_object_speed_mps": r.max_object_speed_mps,
            "max_tilt_deg": r.max_tilt_deg,
            "min_upright_z": r.min_upright_z,
            "object_follow_delta_m": r.object_follow_delta_m,
            "grasped_seen": r.grasped_seen,
            "attach_calls": r.attach_calls,
            "posewrite_calls": r.posewrite_calls,
            "early_kill": r.early_kill,
        }

    def _finite_or_none(value: float) -> float | None:
        # Audit MINOR repair: REACH_FAIL leaves results empty -> inf/nan aggregates;
        # RFC 8259 has no Infinity/NaN tokens, so serialize them as null.
        v = float(value)
        return v if math.isfinite(v) else None

    results_doc = {
        "artifact": "G0B_T3_CYLD29H50_TOP_CENTER_VERTICAL_GRASP_V1",
        "case": "g0b_d420",
        "tag": args.tag,
        "verdict": verdict,
        "usd": {
            "path": str(ATTEMPT3_USD),
            "root_sha256": root_sha,
            "physics_sha256": physics_sha,
            "stage_audit_pass": audit_pass,
            "body_checks": body_checks,
        },
        "object": {
            "shape": "cylinder",
            "size_m": [float(v) for v in object_size],
            "mass_kg": args.object_mass_kg,
            "static_friction": args.static_friction,
            "dynamic_friction": args.dynamic_friction,
            "restitution": args.restitution,
            "friction_provenance": "preregistered assumption (t3_mass_friction_contract.md), NOT measured",
        },
        "materials_runtime": materials,
        "plan": {
            "label": plan.label,
            "center": [float(v) for v in plan.center],
            "world_grasp": [float(v) for v in plan.world_grasp],
            "approach_tcp": [float(v) for v in plan.approach_tcp],
            "descend_tcp": [float(v) for v in plan.descend_tcp],
            "lift_tcp": [float(v) for v in plan.lift_tcp],
            "q_approach_deg": [float(v) for v in plan.q_approach_deg],
            "q_descend_deg": [float(v) for v in plan.q_descend_deg],
            "q_lift_deg": [float(v) for v in plan.q_lift_deg],
            "ik_err_mm": [plan.approach_ik_err_mm, plan.descend_ik_err_mm, plan.lift_ik_err_mm],
            "ik_tilt_deg": [plan.approach_tilt_deg, plan.descend_tilt_deg, plan.lift_tilt_deg],
        },
        "gates": {
            "target_error_gate_m": args.target_error_gate_m,
            "plan_tilt_gate_deg": args.plan_tilt_gate_deg,
            "max_tcp_step_m": args.max_tcp_step_m,
            "waypoint_max_joint_dev_deg": args.waypoint_max_joint_dev_deg,
            "transit_polish_dev_deg": TRANSIT_POLISH_DEV_DEG,
            "transit_pos_band_mm": TRANSIT_POS_BAND_MM,
            "transit_tcp_step_gate_m": args.transit_tcp_step_gate_m,
            "object_drift_gate_m": args.object_drift_gate_m,
            "object_speed_gate_mps": args.object_speed_gate_mps,
            "lift_speed_gate_mps": args.lift_speed_gate_mps,
            "gripper_error_gate_deg": args.gripper_error_gate_deg,
            "tilt_gate_deg": args.tilt_gate_deg,
            "min_upright_z_gate": args.min_upright_z_gate,
            "min_lift_follow_m": args.min_lift_follow_m,
            "marker_dist_m": args.marker_dist_m,
            "marker_q5_max_deg": GRASP_MARKER_Q5_MAX_DEG,
            "close_deg": list(args.close_deg),
            "episode_length_s": args.episode_length_s,
            "gripper_stall_rate_deg_per_step": args.gripper_stall_rate_deg_per_step,
            "gripper_stall_min_steps": args.gripper_stall_min_steps,
            "continue_close_after_grasped_until_angles_done": bool(args.continue_close_after_grasped_until_angles_done),
        },
        "phases": {
            "approach": _step_result_dict(approach_result),
            "descend": _step_result_dict(descend_result),
            "latch": _step_result_dict(latch_result),
            "hold": _step_result_dict(hold_result),
            "lift": _step_result_dict(lift_result),
        },
        # Round-3 MINOR repair: _verdict consumes path_ok flags; serializing
        # them keeps the verdict recomputable from this JSON alone (the
        # IK-truncated-lift domain would otherwise recompute GRASP_PASS).
        "path_ok": {
            "approach": bool(approach_path_ok),
            "descend": bool(descend_path_ok),
            "lift": bool(lift_path_ok),
        },
        "close_records": close_records,
        "aggregate": {
            "max_target_error_m": _finite_or_none(max_target_error),
            "max_tcp_step_m": _finite_or_none(max_tcp_step),
            "max_object_drift_m": _finite_or_none(max_drift),
            "max_object_speed_mps": _finite_or_none(max_speed),
            "max_tilt_deg": _finite_or_none(max_tilt),
            "min_upright_z": _finite_or_none(min_upright),
            "lift_follow_delta_m": _finite_or_none(lift_follow),
            "attach_calls": total_attach_calls,
            "posewrite_calls": total_posewrite_calls,
            "hidden_kinematic_posewrite_artifact": not hidden_posewrite_ok,
            "episode_done": episode_done,
            "nan_seen": nan_seen,
            "total_sim_steps": total_sim_steps,
        },
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "rerun_sdk": RERUN_VERSION},
    }
    results_path.write_text(json.dumps(results_doc, indent=2, default=str) + "\n")

    expected_entities = [
        "metadata/run", "metadata/materials", "world/targets", "world/cylinder",
        "world/tcp", "world/object", "plots/q5_deg", "plots/q5_cmd_deg",
        "plots/target_error_mm", "plots/object_drift_mm", "plots/object_speed_mps",
        "plots/tilt_deg", "plots/upright_z", "plots/marker", "events/phase",
    ]
    components = {
        "metadata/run": ["TextDocument:text"],
        "metadata/materials": ["TextDocument:text"],
        "world/targets": ["Points3D:positions", "Points3D:colors", "Points3D:radii", "Points3D:labels"],
        "world/cylinder": ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"],
        "world/tcp": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "world/object": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "plots/q5_deg": ["Scalars:scalars"],
        "plots/q5_cmd_deg": ["Scalars:scalars"],
        "plots/target_error_mm": ["Scalars:scalars"],
        "plots/object_drift_mm": ["Scalars:scalars"],
        "plots/object_speed_mps": ["Scalars:scalars"],
        "plots/tilt_deg": ["Scalars:scalars"],
        "plots/upright_z": ["Scalars:scalars"],
        "plots/marker": ["Scalars:scalars"],
        "events/phase": ["TextLog:text", "TextLog:level"],
    }
    validation = validate_rerun_artifact(
        rrd_path,
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time", "physics_step", "sim_time_s"],
        expected_entity_components=components,
        blueprint_path=rbl_path,
        screenshot_path=png_path,
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=180.0,
    )
    validation_path.write_text(json.dumps(validation, indent=2, default=str) + "\n")
    print(
        f"[{LOG}] rerun_validation pass={validation.get('pass')} errors={validation.get('errors')}",
        flush=True,
    )
    print(
        f"[{LOG}] artifacts rrd={rrd_path.name} sha={_sha256_file(rrd_path)[:16]} "
        f"results={results_path.name} csv={csv_path.name}",
        flush=True,
    )
    _close_all()
    return 0 if verdict == "GRASP_PASS" and validation.get("pass") else 2


if __name__ == "__main__":
    try:
        _code = main()
    finally:
        # Audit MAJOR-e safety net: close Kit even on exceptions/early exits so a
        # headless batch run can never wedge (idempotent — normal paths already
        # cleared the handles via _close_all()).
        try:
            _close_all()
        except Exception as _close_exc:  # noqa: BLE001
            print(f"[{LOG}] cleanup_error={_close_exc!r}", flush=True)
    raise SystemExit(_code)
