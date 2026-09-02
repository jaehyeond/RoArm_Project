#!/usr/bin/env python3
"""N1: does Newton's implicit-MPM granular material behave like a MATERIAL? (track P4d)

Why this exists
---------------
Track P4 killed Isaac PBD for two reasons, both measured:

* the repose angle was a function of the TIME STEP, not of the material
  (8.74 deg at 1/60 s -> 32.27 deg at 1/1920 s, same friction/damping/adhesion), and
* the pile never stopped moving (best case -0.093 deg/s of creep after a 40 s settle).

`SolverImplicitMPM` in `newton` 1.5.1 exposes a Drucker-Prager granular
constitutive model whose `friction` is a MATERIAL parameter, so it should not
have either failure.  N1 asks exactly the two questions PBD failed, plus the one
question a continuum model raises on its own:

  Q1  does the repose angle respond monotonically to `friction`?
  Q2  does the pile settle, and how fast does it creep afterwards?
  Q3  is the angle independent of the time step (<= 2 deg over a 4x change)?

NOT a DEM
---------
The SIGGRAPH 2026 Newton talk lists `Particles - 0-dimensional: DEM, MPM`
(15:46), but `newton/_src/solvers/__init__.py` exports
SolverBase / SolverFeatherstone / SolverImplicitMPM / SolverKamino / SolverMuJoCo /
SolverSemiImplicit / SolverStyle3D / SolverVBD / SolverXPBD - there is no DEM, and
the strings "DEM" / "discrete element" appear nowhere in the wheel.  So this
probe tests a CONTINUUM, not discrete pellets.  That is the standing caveat: our
pellets are 4.19 mm and the scoop is 60 mm, only 14x apart, so a grid cell cannot
simultaneously hold many grains and resolve the tool.  Q1-Q3 are still the right
first questions; the DEME cross-check on scooped volume is a separate step.

Measurement
-----------
Identical to tracks P3 and P4: the height field goes through
`sim_pbd_pellet_probe.measure_from_height_field`, which is P3's
`sidewall_regression` definition using P3's own helper functions.  Nothing about
the definition is re-implemented here, so DEME, PBD and MPM land on one scale.

The height field is built with the MPM particle sampling radius (that is the
material surface), while the toe threshold is taken from the PELLET diameter so
the toe convention matches P3/DEME exactly.  Both radii are recorded.

Interpreter: `~/miniconda3/envs/newton/bin/python` (newton 1.5.1, warp 1.17.0,
numpy 2.4.6).  This env is SEPARATE from `isaaclab` on purpose - newton needs
warp>=1.16 and isaaclab is pinned to warp 1.11.1 with numpy 1.26.0 / psutil
5.9.8 (D326).  Nothing here touches that env.

NOT MEASURED: no pellet has been procured or measured.  `friction` here is a
Drucker-Prager coefficient of the SIMULATED continuum, never a polypropylene
property.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sim_pellet_model as p3
import sim_pbd_pellet_probe as p4

ARTIFACT = "NEWTON_MPM_PROBE_N1"
OUT_DIR = Path("claudedocs/runtime_logs/newton_mpm")

# P3 pellet reference, so the toe convention and the poured volume match DEME.
PELLET_VOLUME_M3 = 38.48e-9
PELLET_EQ_RADIUS_M = 2.0944e-3
PACKING_FRACTION = 0.60

NON_CLAIMS = [
    "MEASURE: no pellet has been procured, calipered or weighed. Every physical value is a "
    "placeholder inherited from track P3.",
    "SOLVER_COEFFICIENT: `friction` is the Drucker-Prager friction coefficient of the simulated "
    "CONTINUUM. It is not a measured polypropylene property.",
    "CONTINUUM: MPM has no individual grains. At our pellet size (4.19 mm) a grid cell that holds "
    "enough grains for a continuum (>=100 grains needs ~20 mm) cannot also resolve a 60 mm scoop "
    "(needs <=10 mm). This probe does not settle that tension; it only asks whether the material "
    "behaves like a material.",
    "PROTOCOL: this is a released-column (lifted-cylinder) repose test, not P3's batch pour. The "
    "two protocols are not interchangeable and absolute angles must not be cross-quoted without "
    "the matched-protocol comparison.",
]


@dataclass(frozen=True)
class MpmConfig:
    label: str = "base"
    # material (Drucker-Prager)
    friction: float = 0.68
    young_modulus: float = 1.0e15
    poisson_ratio: float = 0.3
    damping: float = 0.0
    yield_pressure: float = 1.0e12
    tensile_yield_ratio: float = 0.0
    yield_stress: float = 0.0
    hardening: float = 0.0
    dilatancy: float = 0.0
    viscosity: float = 0.0
    critical_fraction: float = 0.0
    air_drag: float = 0.0
    ground_mu: float = 0.5
    # numerics
    voxel_size_m: float = 0.005
    particles_per_cell_axis: int = 3
    fps: float = 60.0
    substeps: int = 1
    max_iterations: int = 250
    tolerance: float = 1.0e-4
    grid_type: str = "sparse"
    # geometry: match P3's 1500-pellet bulk volume
    n_pellets_equivalent: int = 1500
    column_radius_m: float = 0.030
    seed: int = 460
    # measurement / schedule
    half_extent_m: float = 0.12
    profile_cell_mm: float = 2.0
    settle_probe_s: float = 0.25
    max_settle_s: float = 8.0
    creep_watch_s: float = 6.0
    settle_apex_drift_frac: float = 0.10   # of a pellet diameter, as in P4

    def bulk_volume_m3(self) -> float:
        return self.n_pellets_equivalent * PELLET_VOLUME_M3 / PACKING_FRACTION

    def column_height_m(self) -> float:
        return self.bulk_volume_m3() / (math.pi * self.column_radius_m**2)

    def measurement_config(self) -> p4.PbdConfig:
        return replace(p4.PbdConfig(), profile_cell_mm=self.profile_cell_mm)


def _cylinder_particles(cfg: MpmConfig) -> tuple[np.ndarray, float, float]:
    """A cylinder of MPM sample points sitting on the floor."""
    spacing = cfg.voxel_size_m / cfg.particles_per_cell_axis
    radius = spacing * 0.5
    R, H = cfg.column_radius_m, cfg.column_height_m()
    n_r = int(math.ceil(2.0 * R / spacing))
    n_z = int(math.ceil(H / spacing))
    xs = (np.arange(n_r) + 0.5) * spacing - R
    zs = (np.arange(n_z) + 0.5) * spacing + radius
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    inside = (X**2 + Y**2) <= R * R
    xy = np.stack([X[inside], Y[inside]], axis=1)
    pts = np.empty((xy.shape[0] * n_z, 3), dtype=np.float64)
    for k, z in enumerate(zs):
        s = slice(k * xy.shape[0], (k + 1) * xy.shape[0])
        pts[s, :2] = xy
        pts[s, 2] = z
    rng = np.random.default_rng(cfg.seed)
    pts[:, :2] += rng.uniform(-0.25 * spacing, 0.25 * spacing, size=(pts.shape[0], 2))
    return pts, spacing, radius


def _measure(cfg: MpmConfig, q: np.ndarray, sample_radius: float) -> dict[str, Any]:
    """P3's sidewall_regression, applied to the MPM material surface."""
    cell_m = cfg.profile_cell_mm * 1.0e-3
    height, axis, _ = p3.surface_height_field(q, sample_radius, cfg.half_extent_m, cell_m)
    m = p4.measure_from_height_field(
        height, axis, cfg.measurement_config(), PELLET_EQ_RADIUS_M, cfg.half_extent_m
    )
    m.pop("radial_profile", None)
    return m


def run_cell(cfg: MpmConfig, out_json: Path) -> dict[str, Any]:
    import warp as wp
    import newton
    from newton.solvers import SolverImplicitMPM

    t0 = time.time()
    pts, spacing, sample_radius = _cylinder_particles(cfg)
    bulk_density = 950.0 * PACKING_FRACTION           # continuum density of the BULK
    mass = spacing**3 * bulk_density

    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    zero = wp.vec3(0.0, 0.0, 0.0)
    for p in pts:
        builder.add_particle(wp.vec3(*p), zero, mass, radius=sample_radius)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=cfg.ground_mu))

    model = builder.finalize()
    model.set_gravity(wp.vec3(0.0, 0.0, -9.81))

    opts = SolverImplicitMPM.Config()
    opts.voxel_size = cfg.voxel_size_m
    opts.max_iterations = cfg.max_iterations
    opts.tolerance = cfg.tolerance
    opts.grid_type = cfg.grid_type
    for key in ("friction", "young_modulus", "poisson_ratio", "damping", "yield_pressure",
                "tensile_yield_ratio", "yield_stress", "hardening", "dilatancy",
                "viscosity", "critical_fraction", "air_drag"):
        if hasattr(model.mpm, key):
            getattr(model.mpm, key).fill_(getattr(cfg, key))

    state_0 = model.state()
    state_1 = model.state()
    solver = SolverImplicitMPM(model, config=opts)

    frame_dt = 1.0 / cfg.fps
    dt = frame_dt / cfg.substeps
    init_wall = time.time() - t0

    def positions() -> np.ndarray:
        return state_0.particle_q.numpy().astype(np.float64)

    def advance(seconds: float) -> None:
        nonlocal state_0, state_1
        for _ in range(int(round(seconds / frame_dt))):
            for _ in range(cfg.substeps):
                solver.step(state_0, state_1, None, None, dt)
                solver.project_outside(state_1, state_1, dt)
                state_0, state_1 = state_1, state_0

    diameter_mm = 2.0 * PELLET_EQ_RADIUS_M * 1e3
    thr_apex_mm = cfg.settle_apex_drift_frac * diameter_mm

    history: list[dict[str, Any]] = []
    sim_t = 0.0
    prev_apex = None
    settled_at = None
    quiet_run = 0
    t_dyn = time.time()
    while sim_t < cfg.max_settle_s:
        advance(cfg.settle_probe_s)
        sim_t += cfg.settle_probe_s
        q = positions()
        speed = np.linalg.norm(state_0.particle_qd.numpy().astype(np.float64), axis=1)
        apex = float(q[:, 2].max())
        row = {
            "sim_time_s": sim_t,
            "apex_m": apex,
            "max_radius_m": float(np.hypot(q[:, 0], q[:, 1]).max()),
            "speed_rms_m_s": float(np.sqrt((speed**2).mean())),
            "speed_p99_m_s": float(np.quantile(speed, 0.99)),
        }
        if prev_apex is not None:
            row["apex_drift_mm"] = abs(apex - prev_apex) * 1e3
            quiet = row["apex_drift_mm"] <= thr_apex_mm
            row["quiet"] = bool(quiet)
            quiet_run = quiet_run + 1 if quiet else 0
        history.append(row)
        prev_apex = apex
        if quiet_run >= 4:
            settled_at = sim_t
            break
    settle_wall = time.time() - t_dyn

    q_settled = positions()
    m_settled = _measure(cfg, q_settled, sample_radius)

    # creep watch, exactly as P4: keep watching an unloaded pile
    creep_rows = []
    for _ in range(int(round(cfg.creep_watch_s))):
        advance(1.0)
        sim_t += 1.0
        qq = positions()
        creep_rows.append({"sim_time_s": sim_t, "apex_m": float(qq[:, 2].max())})
    q_after = positions()
    m_after = _measure(cfg, q_after, sample_radius)
    creep = {
        "watch_s": cfg.creep_watch_s,
        "angle_before_deg": m_settled["repose_angle_deg"],
        "angle_after_deg": m_after["repose_angle_deg"],
        "angle_drift_deg": m_after["repose_angle_deg"] - m_settled["repose_angle_deg"],
        "angle_drift_deg_per_s": (
            (m_after["repose_angle_deg"] - m_settled["repose_angle_deg"]) / cfg.creep_watch_s
        ),
        "apex_drift_mm": (q_after[:, 2].max() - q_settled[:, 2].max()) * 1e3,
        "history": creep_rows,
    }

    result = {
        "artifact": ARTIFACT,
        "label": cfg.label,
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "non_claims": NON_CLAIMS,
        "environment": {
            "newton": "1.5.1",
            "warp": wp.__version__,
            "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "derived": {
            "n_mpm_particles": int(pts.shape[0]),
            "particle_spacing_m": spacing,
            "sample_radius_m": sample_radius,
            "column_radius_m": cfg.column_radius_m,
            "column_height_m": cfg.column_height_m(),
            "bulk_volume_m3": cfg.bulk_volume_m3(),
            "bulk_density_kg_m3": bulk_density,
            "frame_dt_s": frame_dt,
            "solver_dt_s": dt,
            "pellets_per_voxel": PACKING_FRACTION / PELLET_VOLUME_M3 * cfg.voxel_size_m**3,
            "atan_friction_deg": math.degrees(math.atan(cfg.friction)),
        },
        "settle": {
            "criterion": (
                f"apex height changes <= {thr_apex_mm:.3f} mm (0.1 pellet diameters) over "
                f"{cfg.settle_probe_s} s, four probes in a row"
            ),
            "settled": settled_at is not None,
            "settled_sim_time_s": settled_at,
            "history": history,
        },
        "creep": creep,
        "measurement_settled": m_settled,
        "measurement_after_creep": m_after,
        "timing": {
            "init_wall_s": init_wall,
            "settle_wall_s": settle_wall,
            "total_wall_s": time.time() - t0,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_json.with_suffix(".npz"),
        positions_settled_m=q_settled.astype(np.float32),
        positions_after_creep_m=q_after.astype(np.float32),
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
        cfg = replace(MpmConfig(), **spec)
        path = out_dir / f"n1_{cfg.label}.json"
        if path.exists() and not args.force:
            print(f"[n1] skip existing {path}", flush=True)
            continue
        print(f"[n1] === {cfg.label} ===", flush=True)
        try:
            r = run_cell(cfg, path)
            m = r["measurement_settled"]
            print(
                f"[n1] {cfg.label}: mu={cfg.friction:.3f} (atan={r['derived']['atan_friction_deg']:.1f}d) "
                f"angle={m['repose_angle_deg']:.2f}d r2={m['fit_r_squared']:.4f} "
                f"pass={m['measurement_pass']} settled={r['settle']['settled']}@"
                f"{r['settle']['settled_sim_time_s']} creep={creepfmt(r)} "
                f"N={r['derived']['n_mpm_particles']} wall={r['timing']['total_wall_s']:.1f}s",
                flush=True,
            )
        except BaseException as exc:  # noqa: BLE001
            import traceback

            rc = 1
            print(f"[n1] CELL_FAILED {cfg.label}: {exc!r}", flush=True)
            print(traceback.format_exc(), flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"artifact": ARTIFACT, "label": cfg.label, "failed": True,
                 "error": repr(exc), "traceback": traceback.format_exc(),
                 "config": asdict(cfg)}, indent=1))
    print("N1_DONE" if rc == 0 else "N1_PARTIAL")
    return rc


def creepfmt(r: dict[str, Any]) -> str:
    c = r["creep"]["angle_drift_deg_per_s"]
    return f"{c:+.4f}d/s"


if __name__ == "__main__":
    raise SystemExit(main())
