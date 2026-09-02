#!/usr/bin/env python3
"""B1: make the grab actually grab, and find out why it did not (track P4e).

The blocker
-----------
Two independent scoop runs captured NOTHING:

* `claudedocs/runtime_logs/scoop_track/s1_closure/scoop_closure.json` (DEME):
  `captured_particles: 0`, `close_peak: 0.0 N`, `descend_peak: 0.0 N`.
* the P4 PBD scoop: 0 particles in the bucket on all 5 seeds.

Reading `sim_deme_scoop.py` explains the DEME zero without any physics failing:

1. `shell_obj(src, side, P["shell_travel_deg"], ...)` bakes each shell **already
   rotated to the fully-open 44.5 deg** and never re-bakes it.  Its own docstring
   says the closing rotation is "approximated by circular translation about the
   pivot" and that "the mesh itself is not rotated".
2. Closure is therefore `SetFamilyPrescribedLinVel` - a pure 29 mm **translation**
   in x, with the shells still splayed 44.5 deg open.  Two plates held open and
   slid sideways cannot close on anything; the lips pass each other.
3. That run also used `SCOOP_NSUB=1500`, a debug subset of the 18,796-pellet pile
   that keeps the smallest \|y\|, i.e. a thin slab.

So the question B1 has to answer is not "which engine" but "does a correctly
closing grab capture material, and at what insert depth".

Why this runs in Newton MPM
---------------------------
N1 (`claudedocs/runtime_logs/newton_mpm/REPORT_N1.md`) measured Newton's implicit
MPM at 1.64x real time versus DEME's 0.043x - about 38x faster per simulated
second - and its piles settle in 1.25 s and stay put.  That makes it the right
place to SWEEP a trajectory.  The winner still has to be confirmed in DEME; this
probe does not replace that.

The decisive cell pair
----------------------
`rot_d18` and `trans_d18` are identical except for how the shells close:
rotation (correct) versus translation at a fixed 44.5 deg open angle (what
`sim_deme_scoop.py` did).  If rotation captures and translation does not, the
DEME zero is explained by the approximation rather than by the tool design.

NOT MEASURED: no pellet has been procured or measured.  The MPM continuum is
calibrated to nothing yet; N1 showed its angle is grid-dependent (5.63 deg over a
4x voxel change).  Captured volumes here are for COMPARING trajectories against
each other, not for predicting a real capture mass.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sim_pellet_model as p3
import sim_pbd_pellet_probe as p4
import sim_newton_mpm_probe as n1

ARTIFACT = "NEWTON_SCOOP_PROBE_B1"
OUT_DIR = Path("claudedocs/runtime_logs/newton_mpm/b1_scoop")
SHELL_L = Path("claudedocs/runtime_logs/scoop_grab_v1/shell_L_ALL.stl")
SHELL_R = Path("claudedocs/runtime_logs/scoop_grab_v1/shell_R_ALL.stl")

NON_CLAIMS = list(n1.NON_CLAIMS) + [
    "TRAJECTORY_ONLY: captured volumes compare trajectories with each other. The MPM continuum is "
    "not calibrated to any measured pellet, so no captured mass here predicts a real capture.",
    "GEOMETRY: the shells are the track-A grab v1 STLs, but the pivot is taken as the "
    "(+/- pivot_gap/2, 0) vertical axis that sim_deme_scoop.py uses. If the real linkage pivots "
    "elsewhere the closing arc differs.",
]


def load_stl_mm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL -> (vertices_m, triangle_indices). Vertices are welded."""
    raw = path.read_bytes()
    n_tri = struct.unpack("<I", raw[80:84])[0]
    rec = np.frombuffer(raw[84 : 84 + n_tri * 50], dtype=np.uint8).reshape(n_tri, 50)
    tri = np.frombuffer(rec[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3).astype(np.float64)
    verts, inv = np.unique(np.round(tri, 6), axis=0, return_inverse=True)
    return verts * 1.0e-3, inv.astype(np.int32)


@dataclass(frozen=True)
class ScoopConfig:
    label: str = "rot_d18"
    # pile (N1 defaults, friction chosen so the heap is ~21 deg = DEME sphere territory)
    friction: float = 0.68
    voxel_size_m: float = 0.005
    n_pellets_equivalent: int = 1500
    column_radius_m: float = 0.030
    settle_s: float = 3.0
    seed: int = 460
    # grab (track A: D462/D463)
    pivot_gap_mm: float = 26.0
    shell_travel_deg: float = 44.5
    lip_pivot_radius_mm: float = 38.332   # design.json derived
    shell_dir: str = ""   # 비우면 원본 STL. 지정하면 shell_{L,R}_wt.stl 을 쓴다
    shell_friction: float = 0.5
    mesh_is_solid: bool = False
    collider_margin_mm: float = 0.0
    # trajectory
    closure: str = "rotate"          # "rotate" (correct) | "translate" (what DEME did)
    insert_depth_mm: float = 18.0
    approach_clearance_mm: float = 10.0
    descend_s: float = 0.5
    close_s: float = 0.7
    lift_s: float = 0.6
    lift_height_mm: float = 60.0
    settle_after_s: float = 0.5
    # numerics / measurement
    fps: float = 60.0
    substeps: int = 1
    max_iterations: int = 250
    half_extent_m: float = 0.12
    profile_cell_mm: float = 2.0
    capture_margin_mm: float = 30.0   # above the pre-scoop apex = definitely carried

    def measurement_config(self) -> p4.PbdConfig:
        return replace(p4.PbdConfig(), profile_cell_mm=self.profile_cell_mm)


def _pile_config(cfg: ScoopConfig) -> n1.MpmConfig:
    return replace(
        n1.MpmConfig(),
        label=cfg.label,
        friction=cfg.friction,
        voxel_size_m=cfg.voxel_size_m,
        n_pellets_equivalent=cfg.n_pellets_equivalent,
        column_radius_m=cfg.column_radius_m,
        seed=cfg.seed,
        half_extent_m=cfg.half_extent_m,
        profile_cell_mm=cfg.profile_cell_mm,
    )


def run_cell(cfg: ScoopConfig, out_json: Path) -> dict[str, Any]:
    import warp as wp
    import newton
    from newton.solvers import SolverImplicitMPM

    t0 = time.time()
    pcfg = _pile_config(cfg)
    pts, spacing, sample_radius = n1._cylinder_particles(pcfg)
    bulk_density = 950.0 * n1.PACKING_FRACTION
    mass = spacing**3 * bulk_density
    # one MPM sample point stands for this many pellets of BULK volume
    pellet_bulk_volume = n1.PELLET_VOLUME_M3 / n1.PACKING_FRACTION
    pellets_per_sample = spacing**3 / pellet_bulk_volume

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    zero = wp.vec3(0.0, 0.0, 0.0)
    for p in pts:
        builder.add_particle(wp.vec3(*p), zero, mass, radius=sample_radius)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

    # --- two shells as KINEMATIC bodies whose origin is their pivot ----------
    gap = cfg.pivot_gap_mm * 1.0e-3
    # HINGE AXIS = Y (horizontal), NOT Z.  scoop_grab_v1_design.py: "arc in the
    # rotation plane, width along the hinge axis", and the 50 mm shell width maps
    # to the STL's Y extent (52.34 mm).  The pivot height follows from
    # design.json `lip_pivot_radius_mm` = 38.332 = hypot(pivot_gap/2, lip_depth).
    r_lip_design = cfg.lip_pivot_radius_mm * 1.0e-3
    shells = {}
    if cfg.shell_dir:
        _sd = Path(cfg.shell_dir)
        _paths = ((-1, _sd / "shell_L_wt.stl"), (+1, _sd / "shell_R_wt.stl"))
    else:
        _paths = ((-1, SHELL_L), (+1, SHELL_R))
    for side, path in _paths:
        v, idx = load_stl_mm(path)
        z_pivot = math.sqrt(max(r_lip_design**2 - (gap / 2.0) ** 2, 0.0)) + float(v[:, 2].min())
        pivot = np.array([side * gap / 2.0, 0.0, z_pivot])
        v_local = v - pivot                       # body origin == pivot
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*pivot), wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
            label=f"shell_{'L' if side < 0 else 'R'}",
        )
        builder.add_shape_mesh(
            body=body,
            mesh=newton.Mesh(v_local, idx, compute_inertia=False, is_solid=cfg.mesh_is_solid),
            cfg=newton.ModelBuilder.ShapeConfig(mu=cfg.shell_friction, density=0.0),
        )
        shells[side] = {"body": body, "pivot": pivot, "v_local": v_local,
                        "z_lo_local": float(v_local[:, 2].min())}

    model = builder.finalize()
    model.set_gravity(wp.vec3(0.0, 0.0, -9.81))
    opts = SolverImplicitMPM.Config()
    opts.voxel_size = cfg.voxel_size_m
    opts.max_iterations = cfg.max_iterations
    opts.collider_velocity_mode = "backward"
    for key in ("friction",):
        getattr(model.mpm, key).fill_(cfg.friction)

    state_0 = model.state()
    state_1 = model.state()
    solver = SolverImplicitMPM(model, config=opts)
    collider_kwargs = {"body_mass": wp.zeros_like(model.body_mass), "body_q": state_0.body_q}
    if cfg.collider_margin_mm > 0.0:
        collider_kwargs["collider_margins"] = [cfg.collider_margin_mm * 1e-3] * 2
    solver.setup_collider(**collider_kwargs)

    frame_dt = 1.0 / cfg.fps
    dt = frame_dt / cfg.substeps
    n_bodies = model.body_count

    def shell_angle(side: int, angle_frac: float) -> float:
        """Lips swing OUTWARD to open: angle_frac 1 = open 44.5 deg, 0 = closed."""
        return -side * math.radians(cfg.shell_travel_deg) * angle_frac

    def lowest_z_local(angle_frac: float) -> float:
        """Lowest point of the posed shells, relative to the body origin height."""
        lo = 1e9
        for side, sh in shells.items():
            a = shell_angle(side, angle_frac)
            c, sn = math.cos(a), math.sin(a)
            vz = -sh["v_local"][:, 0] * sn + sh["v_local"][:, 2] * c
            lo = min(lo, float(vz.min()))
        return lo

    def set_bodies(z_body: float, angle_frac: float, x_shift: float) -> None:
        q = np.zeros((n_bodies, 7), dtype=np.float32)
        for side, sh in shells.items():
            if cfg.closure == "rotate":
                ang = shell_angle(side, angle_frac)
                px = sh["pivot"][0]
            else:
                # the sim_deme_scoop.py approximation: hold fully open, slide in x
                ang = shell_angle(side, 1.0)
                px = sh["pivot"][0] - side * x_shift
            b = sh["body"]
            q[b, 0:3] = (px, sh["pivot"][1], z_body)
            q[b, 3:7] = (0.0, math.sin(ang / 2.0), 0.0, math.cos(ang / 2.0))  # xyzw about Y
        state_0.body_q.assign(q)
        state_1.body_q.assign(q)

    def advance(seconds: float) -> None:
        nonlocal state_0, state_1
        for _ in range(int(round(seconds / frame_dt))):
            for _ in range(cfg.substeps):
                solver.step(state_0, state_1, None, None, dt)
                solver.project_outside(state_1, state_1, dt)
                state_0, state_1 = state_1, state_0

    def positions() -> np.ndarray:
        return state_0.particle_q.numpy().astype(np.float64)

    def measure(q: np.ndarray) -> dict[str, Any]:
        cell = cfg.profile_cell_mm * 1.0e-3
        h, ax, _ = p3.surface_height_field(q, sample_radius, cfg.half_extent_m, cell)
        m = p4.measure_from_height_field(
            h, ax, cfg.measurement_config(), n1.PELLET_EQ_RADIUS_M, cfg.half_extent_m
        )
        m.pop("radial_profile", None)
        return m

    # --- 1. settle the pile with the grab parked high above ------------------
    z_park = 0.35
    set_bodies(z_park, 1.0, 0.0)
    advance(cfg.settle_s)
    q_pile = positions()
    apex = float(q_pile[:, 2].max())
    m_before = measure(q_pile)

    # --- 2. descend (open) ---------------------------------------------------
    z_lo_local = lowest_z_local(1.0)   # shells are OPEN while descending
    z_start = apex + cfg.approach_clearance_mm * 1e-3 - z_lo_local
    z_insert = apex - cfg.insert_depth_mm * 1e-3 - z_lo_local
    n_desc = max(1, int(round(cfg.descend_s / frame_dt)))
    for k in range(n_desc):
        f = (k + 1) / n_desc
        set_bodies(z_start + (z_insert - z_start) * f, 1.0, 0.0)
        advance(frame_dt)

    # --- 3. close ------------------------------------------------------------
    # translation distance is sim_deme_scoop.py's own chord formula
    r_lip = math.hypot(cfg.pivot_gap_mm / 2.0, 36.06) * 1e-3
    a0 = math.atan2(-36.06, cfg.pivot_gap_mm / 2.0)
    dx_close = abs(r_lip * (math.cos(a0 - math.radians(cfg.shell_travel_deg)) - math.cos(a0)))
    n_close = max(1, int(round(cfg.close_s / frame_dt)))
    for k in range(n_close):
        f = (k + 1) / n_close
        set_bodies(z_insert, 1.0 - f, dx_close * f)
        advance(frame_dt)

    # --- 4. lift -------------------------------------------------------------
    n_lift = max(1, int(round(cfg.lift_s / frame_dt)))
    z_top = z_insert + cfg.lift_height_mm * 1e-3
    for k in range(n_lift):
        f = (k + 1) / n_lift
        set_bodies(z_insert + (z_top - z_insert) * f, 0.0, dx_close)
        advance(frame_dt)
    set_bodies(z_top, 0.0, dx_close)
    advance(cfg.settle_after_s)

    q_after = positions()
    carried = q_after[:, 2] > (apex + cfg.capture_margin_mm * 1e-3)
    n_carried = int(carried.sum())
    captured_pellets = n_carried * pellets_per_sample
    captured_volume_m3 = n_carried * spacing**3

    # remaining pile: everything not carried, measured the usual way
    m_after = measure(q_after[~carried]) if (~carried).sum() > 100 else {"error": "pile too small"}

    result = {
        "artifact": ARTIFACT,
        "label": cfg.label,
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "non_claims": NON_CLAIMS,
        "environment": {"newton": "1.5.1", "warp": wp.__version__, "numpy": np.__version__},
        "derived": {
            "n_mpm_particles": int(pts.shape[0]),
            "particle_spacing_m": spacing,
            "pellets_per_sample_point": pellets_per_sample,
            "pile_apex_m": apex,
            "pile_angle_deg": m_before["repose_angle_deg"],
            "z_start_body_m": z_start,
            "z_insert_body_m": z_insert,
            "shell_z_lo_local_open_m": z_lo_local,
            "shell_z_lo_local_closed_m": lowest_z_local(0.0),
            "hinge_axis": "Y (horizontal)",
            "translate_chord_m": dx_close,
            "capture_threshold_z_m": apex + cfg.capture_margin_mm * 1e-3,
        },
        "capture": {
            "closure": cfg.closure,
            "insert_depth_mm": cfg.insert_depth_mm,
            "carried_sample_points": n_carried,
            "captured_pellet_equivalents": captured_pellets,
            "captured_bulk_volume_cm3": captured_volume_m3 * 1e6,
            "gate_pellets": 100.0,
            "gate_pass": bool(captured_pellets >= 100.0),
        },
        "measurement_before": m_before,
        "measurement_remaining": m_after,
        "timing": {"total_wall_s": time.time() - t0},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_json.with_suffix(".npz"),
        positions_pile_m=q_pile.astype(np.float32),
        positions_after_m=q_after.astype(np.float32),
        carried_mask=carried,
        metadata_json=np.array(json.dumps(result, default=str)),
    )
    out_json.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    specs = json.loads(Path(args.cells).read_text())
    out_dir = Path(args.out_dir)
    rc = 0
    for spec in specs:
        cfg = replace(ScoopConfig(), **spec)
        path = out_dir / f"b1_{cfg.label}.json"
        if path.exists() and not args.force:
            print(f"[b1] skip existing {path}", flush=True)
            continue
        print(f"[b1] === {cfg.label} ===", flush=True)
        try:
            r = run_cell(cfg, path)
            c = r["capture"]
            print(
                f"[b1] {cfg.label}: closure={c['closure']:9s} insert={c['insert_depth_mm']:5.1f}mm "
                f"pile={r['derived']['pile_angle_deg']:5.2f}deg -> carried={c['carried_sample_points']:6d} pts "
                f"= {c['captured_pellet_equivalents']:8.1f} pellets "
                f"({c['captured_bulk_volume_cm3']:6.2f} cm3) gate>=100 {'PASS' if c['gate_pass'] else 'FAIL'} "
                f"wall={r['timing']['total_wall_s']:.1f}s",
                flush=True,
            )
        except BaseException as exc:  # noqa: BLE001
            import traceback

            rc = 1
            print(f"[b1] CELL_FAILED {cfg.label}: {exc!r}", flush=True)
            print(traceback.format_exc(), flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"artifact": ARTIFACT, "label": cfg.label, "failed": True,
                 "error": repr(exc), "traceback": traceback.format_exc(),
                 "config": asdict(cfg)}, indent=1))
    print("B1_DONE" if rc == 0 else "B1_PARTIAL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
