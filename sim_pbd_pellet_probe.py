#!/usr/bin/env python3
"""Isaac Sim 5.1 PBD solid-particle feasibility probe for the pellet pile (track P4).

Why this exists
---------------
`claudedocs/DECISIONS.md` D457 ruled "Isaac PBD = dead end", and the decisive
reason recorded there was that particle state cannot be read back out of the GPU
pipeline, so no post-scoop heightmap could be produced.  **That premise is the
wrong test.**  The physical pipeline never reads particle coordinates: it reads a
DEPTH CAMERA and turns the depth frame into a heightmap.  The simulator only has
to be observable the same way.  So this probe measures the pile through a
top-down depth render, exactly as the real rig does, and treats particle
coordinates as an optional cross-check rather than a dependency.

That also removes a sim/real asymmetry: if both sides are observed by a depth
camera, the GP residual carries the physics difference and not the difference
between "exact particle centres" and "what a camera can see".

What is decided here
--------------------
Five gates (G1..G5, see `--gates`) that either keep Isaac PBD as the granular
engine - which would put the pile, the arm and the depth camera in ONE engine -
or send the project back to DEME.  Every gate can fail; `--selftest` proves the
measurement code can emit FAIL by feeding it analytic surfaces with known
answers.

Comparison basis
----------------
The angle of repose uses the SAME definition as track P3
(`sim_pellet_model.measure_repose_angle`, `sidewall_regression`): the helper
functions `radial_profile`, `_interp_radius_at_height` and `_fit_slope_angle`
are IMPORTED from that module, not reimplemented, so the two tracks cannot drift
apart.  Only the source of the height field differs (depth render here, exact
sphere-surface operator there), and every run reports both so the difference is
measured rather than assumed.

The pour protocol is also P3's: `derive_repose_geometry` sizes the domain,
`generate_column` lays out the batches, and each batch is teleported to a
release plane a constant `drop_gap_mm` above the current heap top.  PBD has no
"held fixed" family, so unreleased batches wait as a flat single layer in a
holding pen 5 m away in +x, far outside both the camera frustum and the profile
grid.

NOT MEASURED
------------
No pellet has been procured or measured.  Nothing here may be cited as a
property of real polypropylene.  The 30 deg target is a placeholder chosen by
the coordinator, not a measurement.  PBD friction / damping / adhesion are
dimensionless solver coefficients, NOT the Coulomb mu, rolling resistance or
surface energy of a real material; they are swept, not derived.

Coordinate frame: right-handed, +z up, floor at z=0, heap axis near x=y=0.
SI units (m, m/s, kg, s) except names ending in `_mm` or `_deg`.

Interpreter: `~/miniconda3/envs/isaaclab/bin/python` (isaacsim 5.1.0.0,
isaaclab 2.3.0, warp 1.11.1; newton is NOT installed, so Newton MPM is out of
scope).  The D326 pins `numpy==1.26.0` / `psutil==5.9.8` must hold; this module
installs nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sim_pellet_model as p3

ARTIFACT = "PBD_PELLET_PROBE_V1"
SCHEMA_VERSION = 1
OUT_DIR = Path("claudedocs/runtime_logs/pbd_probe")

NON_CLAIMS = [
    "MEASURE: no pellet has been procured, calipered or weighed; pellet_dia_mm / pellet_len_mm / "
    "density are the same placeholders track P3 uses",
    "MEASURE: the 30 deg repose target is a coordinator-chosen placeholder, not a measured "
    "polypropylene angle",
    "SOLVER_COEFFICIENT: PBD friction / particleFrictionScale / damping / adhesion are "
    "dimensionless PhysX PBD coefficients. They are NOT Coulomb mu, NOT a rolling resistance and "
    "NOT a surface energy, and must never be quoted as measured material properties",
    "GEOMETRY: PBD solid particles are isotropic spheres. There is no clump, no rolling friction "
    "and no torsional friction, so any interlocking that a real extruded cylinder provides is "
    "absent by construction",
    "PROTOCOL: the pour reproduces P3's batch/drop-gap protocol but the engine, contact model and "
    "time step differ, so a PBD angle and a DEME angle are the same MEASUREMENT applied to two "
    "different materials-in-simulation, not a validation of one by the other",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PbdConfig:
    """One PBD cell.  Everything the run depends on lives here and is serialized."""

    # --- shared with P3 so the geometry and the measurement match -----------
    n_particles: int = 10_000
    seed: int = 460
    pellet_dia_mm: float = 3.5
    pellet_len_mm: float = 4.0
    particle_density_kg_m3: float = 950.0
    profile_cell_mm: float = 2.0
    fit_upper_height_fraction: float = 0.80
    fit_lower_height_fraction: float = 0.20
    toe_height_fraction_of_diameter: float = 0.50
    min_fit_r_squared: float = 0.90
    max_quadrant_spread_deg: float = 15.0
    wall_clearance_fraction: float = 0.90
    min_heap_fraction: float = 0.90
    drop_gap_mm: float = 15.0
    n_release_batches: int = 18
    release_quiet_speed_m_s: float = 0.25
    release_interval_s: float = 0.06
    release_timeout_s: float = 0.14

    # --- PBD solver -------------------------------------------------------
    # solidRestOffset is the particle's physical radius.  It is set to the
    # volume-equivalent sphere radius of the nominal pellet cylinder, the same
    # radius P3's `sphere` template uses, so the two tracks pour the same bulk
    # volume.
    # `fluid` is the single attribute that decides whether PhysX treats the set
    # as granular solids or as an SPH fluid.  It is exposed as a config field
    # ONLY so a fluid=True control cell can prove the flag is live: if the two
    # settle to the same heap, the granular result would be an artifact.
    fluid: bool = False
    self_collision: bool = True
    rest_offset_scale: float = 1.0
    # PhysX autocomputation relates the two as solidRestOffset ~ 0.99*0.5*
    # particleContactOffset (schema: solidRestOffset must be < particleContactOffset).
    solid_to_contact_ratio: float = 0.495
    friction: float = 0.60
    particle_friction_scale: float = 1.0
    damping: float = 0.0
    adhesion: float = 0.0
    particle_adhesion_scale: float = 1.0
    adhesion_offset_scale: float = 0.0
    solver_position_iterations: int = 16
    time_steps_per_second: int = 60
    max_velocity_m_s: float = 50.0
    max_depenetration_velocity_m_s: float = 1.0
    enable_ccd: bool = False
    max_neighborhood: int = 96

    # --- pour / settle schedule -------------------------------------------
    max_pour_s: float = 12.0
    settle_probe_interval_s: float = 0.25
    settle_window_probes: int = 4          # 1.0 s of quiet at 0.25 s probes
    # Thresholds are fractions of the PARTICLE DIAMETER, not tuned millimetres:
    # the bulk of the surface must move less than a tenth of a grain per probe,
    # while a single grain is still allowed to roll one full diameter.
    settle_p99_drift_frac_of_diameter: float = 0.10
    settle_max_drift_frac_of_diameter: float = 1.00
    settle_apex_drift_frac_of_diameter: float = 0.10
    settle_toe_drift_frac_of_diameter: float = 0.25
    max_settle_s: float = 20.0
    creep_watch_s: float = 10.0            # G2: keep watching AFTER settling

    # --- depth camera ------------------------------------------------------
    cam_res: int = 1536
    cam_focal_mm: float = 24.0
    cam_aperture_mm: float = 20.955
    cam_fov_margin: float = 1.15           # frustum half-width / half_extent
    depth_render_subframes: int = 1

    # --- scoop (G3/G5) -----------------------------------------------------
    scoop_width_m: float = 0.060
    scoop_depth_m: float = 0.045
    scoop_height_m: float = 0.040
    scoop_wall_m: float = 0.0025
    scoop_start_r_m: float = 0.085
    scoop_plunge_z_m: float = 0.006        # blade tip height above the floor
    scoop_travel_s: float = 1.20
    scoop_lift_s: float = 0.80
    scoop_lift_z_m: float = 0.12
    # After the lift the tool must LEAVE THE FRAME before the pile is measured -
    # exactly as the real rig measures the pile only once the grab has gone to
    # the discharge point.  Without this the lifted scoop, and the material it
    # carries, are rendered as pile height and the "after" volume comes out
    # LARGER than the "before" volume.
    scoop_retreat_x_m: float = 0.35
    scoop_retreat_z_m: float = 0.30
    scoop_retreat_s: float = 0.60

    label: str = "base"

    def p3_config(self) -> "p3.ReposeConfig":
        """The P3 config that owns the geometry and the measurement definition."""
        return p3.ReposeConfig(
            n_particles=self.n_particles,
            shape="sphere",
            seed=self.seed,
            pellet_dia_mm=self.pellet_dia_mm,
            pellet_len_mm=self.pellet_len_mm,
            particle_density_kg_m3=self.particle_density_kg_m3,
            profile_cell_mm=self.profile_cell_mm,
            fit_upper_height_fraction=self.fit_upper_height_fraction,
            fit_lower_height_fraction=self.fit_lower_height_fraction,
            toe_height_fraction_of_diameter=self.toe_height_fraction_of_diameter,
            min_fit_r_squared=self.min_fit_r_squared,
            max_quadrant_spread_deg=self.max_quadrant_spread_deg,
            wall_clearance_fraction=self.wall_clearance_fraction,
            min_heap_fraction=self.min_heap_fraction,
            drop_gap_mm=self.drop_gap_mm,
            n_release_batches=self.n_release_batches,
            release_quiet_speed_m_s=self.release_quiet_speed_m_s,
            release_interval_s=self.release_interval_s,
            release_timeout_s=self.release_timeout_s,
        )


# ---------------------------------------------------------------------------
# Measurement: height field -> repose angle, P3 definition, P3 helpers
# ---------------------------------------------------------------------------


def measure_from_height_field(
    height: np.ndarray,
    axis: np.ndarray,
    cfg: PbdConfig,
    sphere_radius_m: float,
    half_extent_m: float,
) -> dict[str, Any]:
    """`sidewall_regression` on an already-built height field.

    Steps 2-6 of `sim_pellet_model.measure_repose_angle` verbatim, with the
    arithmetic done by that module's own helpers.  Only step 1 differs: the
    height field arrives from a depth render instead of the analytic
    highest-sphere-surface operator.

    The two confinement checks that P3 computes from particle coordinates are
    replaced by their height-field equivalents, because a depth camera has no
    particle coordinates:

    * `wall_clearance_ok` is unchanged (it only ever used the toe radius).
    * `heap_holds_the_bulk` becomes a MATERIAL-VOLUME fraction: the share of
      integrated height that sits inside 1.15 * r_toe, instead of the share of
      particle centres.  A sprayed pour fails both, but the numbers are not
      interchangeable and are labelled differently in the output.
    """
    cell = float(axis[1] - axis[0]) if axis.size > 1 else cfg.profile_cell_mm * 1e-3
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    weight = height.ravel()
    if weight.sum() <= 0.0:
        raise RuntimeError("PBD_MEASURE_FAIL: height field is empty")
    axis_xy = (
        float((grid_x.ravel() * weight).sum() / weight.sum()),
        float((grid_y.ravel() * weight).sum() / weight.sum()),
    )
    profile = p3.radial_profile(height, axis, axis_xy, cell)

    h_apex = float(profile[:, 1].max())
    if h_apex <= 0.0:
        raise RuntimeError("PBD_MEASURE_FAIL: heap has zero height")
    r_upper = p3._interp_radius_at_height(profile, cfg.fit_upper_height_fraction * h_apex)
    r_lower = p3._interp_radius_at_height(profile, cfg.fit_lower_height_fraction * h_apex)
    if r_upper is None or r_lower is None or r_lower <= r_upper:
        raise RuntimeError(
            f"PBD_MEASURE_FAIL: cannot bracket the sidewall band "
            f"(r_upper={r_upper}, r_lower={r_lower}, h_apex={h_apex:.6f})"
        )
    primary_deg, r_squared, n_fit = p3._fit_slope_angle(profile, r_upper, r_lower)

    toe_threshold = cfg.toe_height_fraction_of_diameter * 2.0 * sphere_radius_m
    above_toe = profile[(profile[:, 1] >= toe_threshold) & (profile[:, 3] > 0)]
    r_toe = float(above_toe[:, 0].max()) if above_toe.size else float("nan")
    r_plateau = p3._interp_radius_at_height(profile, 0.95 * h_apex)
    if r_plateau is None:
        r_plateau = 0.0
    secondary_deg = (
        math.degrees(math.atan(h_apex / (r_toe - r_plateau)))
        if math.isfinite(r_toe) and r_toe > r_plateau
        else float("nan")
    )

    xx, yy = np.meshgrid(axis - axis_xy[0], axis - axis_xy[1], indexing="xy")
    theta = np.arctan2(yy, xx)
    rr = np.hypot(xx, yy)
    quadrant_angles: list[float] = []
    for q in range(4):
        lo = -math.pi + q * math.pi / 2.0
        mask = (theta >= lo) & (theta < lo + math.pi / 2.0)
        sub = p3.radial_profile(np.where(mask, height, 0.0), axis, axis_xy, cell)
        bins = np.floor(rr[mask] / cell).astype(np.int64)
        counts = np.bincount(bins, minlength=sub.shape[0])[: sub.shape[0]].astype(np.float64)
        sums = np.bincount(bins, weights=height[mask], minlength=sub.shape[0])[: sub.shape[0]]
        with np.errstate(invalid="ignore", divide="ignore"):
            sub[:, 1] = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
        sub[:, 3] = counts
        q_apex = float(sub[:, 1].max())
        q_up = p3._interp_radius_at_height(sub, cfg.fit_upper_height_fraction * q_apex)
        q_lo = p3._interp_radius_at_height(sub, cfg.fit_lower_height_fraction * q_apex)
        if q_up is None or q_lo is None or q_lo <= q_up:
            quadrant_angles.append(float("nan"))
            continue
        quadrant_angles.append(p3._fit_slope_angle(sub, q_up, q_lo)[0])
    finite_q = [a for a in quadrant_angles if math.isfinite(a)]
    quadrant_spread = max(finite_q) - min(finite_q) if len(finite_q) >= 2 else float("nan")

    gate_radius = cfg.wall_clearance_fraction * half_extent_m
    total_material = float(height.sum())
    inside = float(height[rr <= 1.15 * r_toe].sum()) if math.isfinite(r_toe) else 0.0
    volume_fraction = inside / total_material if total_material > 0 else 0.0
    cell_area = cell * cell
    material_volume_m3 = total_material * cell_area

    # A `sidewall_regression` fit assumes there IS a sidewall.  A pancake - flat
    # to near its rim and then a one-cell cliff - puts the 80%-20% band entirely
    # on the cliff and returns a steep, well-fitted, MEANINGLESS angle.  (Seen
    # directly: claudedocs/runtime_logs/pbd_probe/diagnose_f1_fric5p00.png, flat
    # at 13 mm out to r=115 mm then a 10 mm drop, reported as 37 deg.)  For a
    # cone with a flat top of radius `a` and base `R`, the 95%-apex radius is
    # a + 0.05*(R-a), so even a heavily truncated cone stays below 0.5*R while a
    # pancake sits near R.  This check is what stops a cliff passing as a slope.
    plateau_ratio = (
        float(r_plateau / r_toe) if math.isfinite(r_toe) and r_toe > 0 else float("nan")
    )
    checks = {
        "fit_r_squared_ok": bool(math.isfinite(r_squared) and r_squared >= cfg.min_fit_r_squared),
        "heap_is_conical": bool(math.isfinite(plateau_ratio) and plateau_ratio <= 0.5),
        "wall_clearance_ok": bool(math.isfinite(r_toe) and r_toe <= gate_radius),
        "heap_holds_the_bulk_volume": bool(volume_fraction >= cfg.min_heap_fraction),
        "angle_finite": bool(math.isfinite(primary_deg)),
        "angle_physical": bool(math.isfinite(primary_deg) and 5.0 < primary_deg < 70.0),
        "not_lopsided": bool(
            not math.isfinite(quadrant_spread) or quadrant_spread <= cfg.max_quadrant_spread_deg
        ),
    }
    return {
        "primary_definition": p3.REPOSE_PRIMARY_DEFINITION,
        "primary_definition_detail": p3.REPOSE_PRIMARY_DEFINITION_TEXT,
        "primary_definition_band": [cfg.fit_upper_height_fraction, cfg.fit_lower_height_fraction],
        "repose_angle_deg": float(primary_deg),
        "repose_angle_secondary_deg": float(secondary_deg),
        "fit_r_squared": float(r_squared),
        "fit_band_m": [float(r_upper), float(r_lower)],
        "fit_points": int(n_fit),
        "apex_height_m": h_apex,
        "toe_radius_m": float(r_toe),
        "plateau_radius_m": float(r_plateau),
        "plateau_over_toe_ratio": plateau_ratio,
        "quadrant_angles_deg": [float(a) for a in quadrant_angles],
        "quadrant_spread_deg": float(quadrant_spread),
        "azimuthal_uncertainty_deg": float(0.5 * quadrant_spread),
        "heap_axis_xy_m": [axis_xy[0], axis_xy[1]],
        "wall_gate_radius_m": gate_radius,
        "material_volume_fraction_within_1p15_toe": volume_fraction,
        "integrated_material_volume_m3": material_volume_m3,
        "domain_half_extent_m": half_extent_m,
        "profile_cell_m": cell,
        "radial_profile": profile,
        "checks": checks,
        "measurement_pass": bool(all(checks.values())),
    }


def analytic_pancake_height_field(
    flat_radius_m: float, cliff_width_m: float, thickness_m: float,
    half_extent_m: float, cell_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """A flat cake with a steep rim - the shape PBD actually produced.

    Its 80%-20% band lies entirely on the rim, so the regression returns a steep
    angle with a good r^2.  `--selftest` uses it to prove `heap_is_conical`
    rejects that reading.
    """
    n = int(math.ceil(2.0 * half_extent_m / cell_m))
    axis = (np.arange(n, dtype=np.float64) + 0.5) * cell_m - half_extent_m
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    r = np.hypot(xx, yy)
    h = np.where(
        r <= flat_radius_m,
        thickness_m,
        thickness_m * np.clip(1.0 - (r - flat_radius_m) / cliff_width_m, 0.0, 1.0),
    )
    return h, axis


def analytic_cone_height_field(
    angle_deg: float, apex_m: float, half_extent_m: float, cell_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """A perfect cone on P3's grid.  Used by `--selftest` to prove the fit works."""
    n = int(math.ceil(2.0 * half_extent_m / cell_m))
    axis = (np.arange(n, dtype=np.float64) + 0.5) * cell_m - half_extent_m
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    r = np.hypot(xx, yy)
    slope = math.tan(math.radians(angle_deg))
    return np.maximum(apex_m - slope * r, 0.0), axis


# ---------------------------------------------------------------------------
# Depth -> height field
# ---------------------------------------------------------------------------


def depth_to_height_field(
    depth: np.ndarray,
    cam_z_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    half_extent_m: float,
    cell_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Pinhole-unproject a top-down `distance_to_image_plane` frame onto P3's grid.

    The camera looks straight down (-z world) with its own +x along world +x and
    its own +y along world +y, so image row `v` runs towards world -y.  Cells
    take the MAX z of the points that land in them - the top-of-material
    operator, the same reduction `surface_height_field` performs analytically.
    Cells with no return stay at 0 (bare floor), matching P3.
    """
    d = np.asarray(depth, dtype=np.float64)
    h_px, w_px = d.shape
    v_idx, u_idx = np.nonzero(np.isfinite(d) & (d > 1.0e-4) & (d < 10.0))
    z_cam = d[v_idx, u_idx]
    x_w = (u_idx.astype(np.float64) - cx) * z_cam / fx
    y_w = -(v_idx.astype(np.float64) - cy) * z_cam / fy
    z_w = cam_z_m - z_cam

    n = int(math.ceil(2.0 * half_extent_m / cell_m))
    axis = (np.arange(n, dtype=np.float64) + 0.5) * cell_m - half_extent_m
    col = np.floor((x_w + half_extent_m) / cell_m).astype(np.int64)
    row = np.floor((y_w + half_extent_m) / cell_m).astype(np.int64)
    keep = (col >= 0) & (col < n) & (row >= 0) & (row < n)
    flat = row[keep] * n + col[keep]
    height = np.zeros(n * n, dtype=np.float64)
    np.maximum.at(height, flat, z_w[keep])
    counts = np.bincount(flat, minlength=n * n)
    # A depth camera cannot return a negative height; clamp floor noise to 0 so
    # the bare-floor convention matches P3 exactly.
    height = np.maximum(height, 0.0).reshape(n, n)
    meta = {
        "pixels_total": int(d.size),
        "pixels_valid": int(z_cam.size),
        "pixels_in_grid": int(keep.sum()),
        "grid_cells": int(n * n),
        "cells_with_return": int(np.count_nonzero(counts)),
        "cells_empty": int(n * n - np.count_nonzero(counts)),
        "pixels_per_cell_mean": float(counts[counts > 0].mean()) if counts.any() else 0.0,
        "floor_clamped_cells": int(np.count_nonzero(counts.reshape(n, n) > 0) - np.count_nonzero(height > 0)),
    }
    return height, axis, meta


# ---------------------------------------------------------------------------
# Scene construction (Isaac Sim only - imported lazily inside run_cell)
# ---------------------------------------------------------------------------

PEN_X_M = 5.0  # holding pen for unreleased batches, far outside camera + grid


def _pen_layout(n: int, spacing: float, radius: float) -> np.ndarray:
    """A single flat resting layer for every not-yet-poured particle."""
    side = int(math.ceil(math.sqrt(n)))
    i, j = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    xy = np.stack([i.ravel(), j.ravel()], axis=1)[:n].astype(np.float64) * spacing
    out = np.zeros((n, 3), dtype=np.float64)
    out[:, 0] = PEN_X_M + xy[:, 0]
    out[:, 1] = xy[:, 1] - 0.5 * side * spacing
    out[:, 2] = radius * 1.02
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _env_stamp() -> dict[str, Any]:
    import importlib.metadata as md

    def ver(name: str) -> str | None:
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001
            return None

    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "isaacsim": ver("isaacsim"),
        "isaaclab": ver("isaaclab"),
        "warp_lang": ver("warp-lang"),
        "psutil": ver("psutil"),
        "newton": ver("newton"),
        "d326_pins_ok": np.__version__ == "1.26.0" and ver("psutil") == "5.9.8",
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_cell(cfg: PbdConfig, out_json: Path, *, do_scoop: bool, store_depth: bool) -> dict[str, Any]:
    """Pour, settle, measure, and (optionally) scoop one PBD cell.

    Assumes a live `SimulationApp`; `main` owns its lifetime so a sweep pays the
    ~11 s Kit start-up once.
    """
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
    from omni.physx.scripts import particleUtils, physicsUtils
    from isaacsim.core.api import World

    t_start = time.time()
    p3cfg = cfg.p3_config()
    template = p3.build_template(p3cfg.pellet(), "sphere", size_match="volume")
    geom = p3.derive_repose_geometry(p3cfg, template)
    parked, _quats, batch_index = p3.generate_column(p3cfg, geom)

    radius = float(template.sphere_radius_m) * cfg.rest_offset_scale
    pco = radius / cfg.solid_to_contact_ratio
    half_extent = float(geom["half_extent_m"])
    cell_m = cfg.profile_cell_mm * 1.0e-3
    dt = 1.0 / float(cfg.time_steps_per_second)
    n = int(cfg.n_particles)

    World.clear_instance()
    omni.usd.get_context().new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world.scene.add_default_ground_plane(
        static_friction=cfg.friction, dynamic_friction=cfg.friction, restitution=0.0
    )
    scene_prim = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
    px = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    px.CreateEnableGPUDynamicsAttr().Set(True)
    px.CreateBroadphaseTypeAttr().Set("GPU")
    px.CreateSolverTypeAttr().Set("TGS")
    px.CreateTimeStepsPerSecondAttr().Set(int(cfg.time_steps_per_second))
    UsdLux.DistantLight.Define(stage, Sdf.Path("/World/light")).CreateIntensityAttr(3000.0)

    ps_path = Sdf.Path("/World/particleSystem")
    particleUtils.add_physx_particle_system(
        stage,
        ps_path,
        simulation_owner=scene_prim.GetPath(),
        particle_contact_offset=pco,
        solid_rest_offset=radius,
        contact_offset=pco,
        rest_offset=radius,
        max_velocity=cfg.max_velocity_m_s,
        solver_position_iterations=cfg.solver_position_iterations,
        max_neighborhood=cfg.max_neighborhood,
        max_depenetration_velocity=cfg.max_depenetration_velocity_m_s,
        enable_ccd=cfg.enable_ccd,
    )
    mat_path = Sdf.Path("/World/pbdMaterial")
    particleUtils.add_pbd_particle_material(
        stage,
        mat_path,
        friction=cfg.friction,
        particle_friction_scale=cfg.particle_friction_scale,
        damping=cfg.damping,
        adhesion=cfg.adhesion,
        particle_adhesion_scale=cfg.particle_adhesion_scale,
        adhesion_offset_scale=cfg.adhesion_offset_scale,
        density=cfg.particle_density_kg_m3,
    )
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(ps_path), mat_path)

    pen = _pen_layout(n, 2.4 * radius, radius)
    particle_mass = float(template.mass_kg)
    inst_path = Sdf.Path("/World/particles")
    particleUtils.add_physx_particleset_pointinstancer(
        stage,
        inst_path,
        Vt.Vec3fArray.FromNumpy(pen.astype(np.float32)),
        Vt.Vec3fArray.FromNumpy(np.zeros((n, 3), dtype=np.float32)),
        ps_path,
        self_collision=cfg.self_collision,
        fluid=cfg.fluid,
        particle_group=0,
        particle_mass=particle_mass,
        density=0.0,
    )
    proto = UsdGeom.Sphere(stage.GetPrimAtPath(inst_path.AppendChild("particlePrototype0")))
    proto.CreateRadiusAttr().Set(float(radius))
    proto.CreateExtentAttr().Set([(-radius, -radius, -radius), (radius, radius, radius)])

    # --- top-down depth camera ------------------------------------------------
    cam_half_fov_tan = cfg.cam_aperture_mm / 2.0 / cfg.cam_focal_mm
    cam_z = cfg.cam_fov_margin * half_extent / cam_half_fov_tan
    cam = UsdGeom.Camera.Define(stage, Sdf.Path("/World/depthCam"))
    cam.CreateFocalLengthAttr().Set(cfg.cam_focal_mm)
    cam.CreateHorizontalApertureAttr().Set(cfg.cam_aperture_mm)
    cam.CreateVerticalApertureAttr().Set(cfg.cam_aperture_mm)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, float(cam_z * 4.0)))
    UsdGeom.Xformable(cam.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(cam_z)))
    fx = fy = cfg.cam_res * cfg.cam_focal_mm / cfg.cam_aperture_mm
    cx = cy = cfg.cam_res / 2.0

    # --- scoop (kinematic open box: floor + back + two sides) -----------------
    scoop_root = None
    scoop_op = None
    if do_scoop:
        scoop_root = UsdGeom.Xform.Define(stage, Sdf.Path("/World/scoop"))
        scoop_op = UsdGeom.Xformable(scoop_root.GetPrim()).AddTranslateOp()
        w, dpt, hgt, th = cfg.scoop_width_m, cfg.scoop_depth_m, cfg.scoop_height_m, cfg.scoop_wall_m
        panels = {
            "floor": ((w, dpt, th), (0.0, 0.0, th / 2.0)),
            "back": ((w, th, hgt), (0.0, dpt / 2.0, hgt / 2.0)),
            "left": ((th, dpt, hgt), (-w / 2.0, 0.0, hgt / 2.0)),
            "right": ((th, dpt, hgt), (w / 2.0, 0.0, hgt / 2.0)),
        }
        for name, (scale, offset) in panels.items():
            c = UsdGeom.Cube.Define(stage, Sdf.Path(f"/World/scoop/{name}"))
            c.CreateSizeAttr().Set(1.0)
            xf = UsdGeom.Xformable(c.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*offset))
            xf.AddScaleOp().Set(Gf.Vec3f(*[float(s) for s in scale]))
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            rb = UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
            rb.CreateKinematicEnabledAttr().Set(True)
        scoop_op.Set(Gf.Vec3d(0.0, -cfg.scoop_start_r_m, 2.0))  # parked above everything

    world.reset()
    inst = UsdGeom.PointInstancer(stage.GetPrimAtPath(inst_path))
    # Read the authored USD back so the artifact records what PhysX was actually
    # handed, not what the config asked for.
    usd_authored = {
        "particle_set": {
            a.GetName(): a.Get()
            for a in stage.GetPrimAtPath(inst_path).GetAttributes()
            if a.HasAuthoredValue()
            and (a.GetName().startswith("physxParticle") or a.GetName().startswith("physics:"))
        },
        "particle_system": {
            a.GetName(): a.Get()
            for a in stage.GetPrimAtPath(ps_path).GetAttributes()
            if a.HasAuthoredValue() and not a.GetName().startswith("primvars")
        },
        "pbd_material": {
            a.GetName(): a.Get()
            for a in stage.GetPrimAtPath(mat_path).GetAttributes()
            if a.HasAuthoredValue() and a.GetName().startswith("physxPBDMaterial")
        },
        "material_binding_targets": [
            str(t)
            for t in (
                UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(ps_path))
                .GetDirectBindingRel("physics")
                .GetTargets()
                or []
            )
        ],
        "prototype_radius_m": float(proto.GetRadiusAttr().Get()),
    }
    rp = rep.create.render_product("/World/depthCam", (cfg.cam_res, cfg.cam_res))
    annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    annot.attach(rp)

    def positions() -> np.ndarray:
        return np.array(inst.GetPositionsAttr().Get(), dtype=np.float64)

    def velocities() -> np.ndarray:
        v = inst.GetVelocitiesAttr().Get()
        return np.array(v, dtype=np.float64) if v is not None else np.zeros((n, 3))

    def write_positions(arr: np.ndarray, vel: np.ndarray) -> None:
        inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(arr.astype(np.float32)))
        inst.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(vel.astype(np.float32)))

    stale_frames = {"count": 0, "last": None}

    def depth_frame(require_fresh: bool = False) -> np.ndarray:
        """One depth frame, with an explicit guard against a STALE frame.

        `annot.get_data()` returns an empty 1-D buffer until the render product
        has been pumped, and afterwards it can hand back the PREVIOUS frame if
        the renderer has not finished a new one.  A stale frame reads as "zero
        drift", which would let the settle test pass on a pile that is still
        moving, so it is detected (bitwise identity with the previous frame) and
        pumped again rather than trusted.
        """
        arr = None
        for attempt in range(24):
            world.render()
            candidate = np.asarray(annot.get_data(), dtype=np.float32)
            if candidate.ndim != 2 or candidate.shape != (cfg.cam_res, cfg.cam_res):
                continue
            arr = candidate
            if attempt < 2:
                continue
            if not require_fresh or stale_frames["last"] is None:
                break
            if not np.array_equal(arr, stale_frames["last"]):
                break
        if arr is None:
            raise RuntimeError(
                f"DEPTH_ANNOTATOR_FAIL: no {cfg.cam_res}x{cfg.cam_res} frame after 24 render "
                f"pumps (last shape {np.asarray(annot.get_data()).shape})"
            )
        identical = stale_frames["last"] is not None and np.array_equal(arr, stale_frames["last"])
        if identical:
            stale_frames["count"] += 1
        stale_frames["last"] = arr
        stale_frames["identical"] = bool(identical)
        return arr

    def height_from_depth(require_fresh: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return depth_to_height_field(
            depth_frame(require_fresh), cam_z, fx, fy, cx, cy, half_extent, cell_m
        )

    toe_threshold_m = cfg.toe_height_fraction_of_diameter * 2.0 * radius

    def toe_radius(height: np.ndarray, axis_: np.ndarray) -> float:
        """Outermost annulus whose mean height still carries material."""
        prof = p3.radial_profile(height, axis_, (0.0, 0.0), cell_m)
        above = prof[(prof[:, 1] >= toe_threshold_m) & (prof[:, 3] > 0)]
        return float(above[:, 0].max()) if above.size else float("nan")

    # ---------------- pour -------------------------------------------------
    released = np.zeros(n, dtype=bool)
    batch_index = np.asarray(batch_index)
    release_log: list[dict[str, Any]] = []
    sim_t = 0.0
    steps = 0
    last_release_t = -1e9
    next_batch = 0
    n_batches = int(cfg.n_release_batches)
    pour_timeout = False
    while next_batch < n_batches:
        sel = batch_index == next_batch
        due_quiet = False
        if released.any():
            v = velocities()[released]
            due_quiet = float(np.linalg.norm(v, axis=1).max()) < cfg.release_quiet_speed_m_s
        else:
            due_quiet = True
        elapsed = sim_t - last_release_t
        if elapsed >= cfg.release_interval_s and (due_quiet or elapsed >= cfg.release_timeout_s):
            pos = positions()
            heap_top = float(pos[released, 2].max()) if released.any() else 0.0
            release_z = heap_top + cfg.drop_gap_mm * 1.0e-3
            local = parked[sel].copy()
            local[:, 2] -= local[:, 2].min()
            pos[sel, 0] = parked[sel, 0]
            pos[sel, 1] = parked[sel, 1]
            pos[sel, 2] = release_z + local[:, 2] + radius
            vel = velocities()
            vel[sel] = 0.0
            write_positions(pos, vel)
            released |= sel
            release_log.append(
                {
                    "batch": next_batch,
                    "sim_time_s": sim_t,
                    "n": int(sel.sum()),
                    "heap_top_m": heap_top,
                    "release_z_m": release_z,
                    "triggered_by": "quiet" if due_quiet else "timeout",
                }
            )
            last_release_t = sim_t
            next_batch += 1
            continue
        world.step(render=False)
        sim_t += dt
        steps += 1
        if sim_t > cfg.max_pour_s:
            pour_timeout = True
            break
    pour_end_t = sim_t

    # ---------------- settle (depth-observable criterion) -------------------
    diameter_mm = 2.0 * radius * 1e3
    thr_p99_mm = cfg.settle_p99_drift_frac_of_diameter * diameter_mm
    thr_max_mm = cfg.settle_max_drift_frac_of_diameter * diameter_mm
    thr_apex_mm = cfg.settle_apex_drift_frac_of_diameter * diameter_mm
    thr_toe_mm = cfg.settle_toe_drift_frac_of_diameter * diameter_mm

    probe_steps = max(1, int(round(cfg.settle_probe_interval_s / dt)))
    settle_history: list[dict[str, Any]] = []
    prev_height = None
    prev_toe = None
    quiet_run = 0
    settled_at = None
    axis = None
    while sim_t - pour_end_t < cfg.max_settle_s:
        for _ in range(probe_steps):
            world.step(render=False)
            sim_t += dt
            steps += 1
        height, axis, dmeta = height_from_depth(require_fresh=True)
        v = np.linalg.norm(velocities()[released], axis=1)
        toe = toe_radius(height, axis)
        row: dict[str, Any] = {
            "sim_time_s": sim_t,
            "apex_m": float(height.max()),
            "toe_radius_m": toe,
            "material_sum_m": float(height.sum()),
            "speed_max_m_s": float(v.max()),
            "speed_p99_m_s": float(np.quantile(v, 0.99)),
            "speed_rms_m_s": float(np.sqrt((v**2).mean())),
            "frame_identical_to_previous": bool(stale_frames.get("identical")),
        }
        if prev_height is not None:
            dh = np.abs(height - prev_height)
            row["max_cell_drift_mm"] = float(dh.max() * 1e3)
            row["p99_cell_drift_mm"] = float(np.quantile(dh, 0.99) * 1e3)
            row["apex_drift_mm"] = float(abs(height.max() - prev_height.max()) * 1e3)
            row["toe_drift_mm"] = float(abs(toe - prev_toe) * 1e3) if prev_toe is not None else 0.0
            quiet = (
                not row["frame_identical_to_previous"]
                and row["p99_cell_drift_mm"] <= thr_p99_mm
                and row["max_cell_drift_mm"] <= thr_max_mm
                and row["apex_drift_mm"] <= thr_apex_mm
                and row["toe_drift_mm"] <= thr_toe_mm
            )
            row["quiet"] = bool(quiet)
            quiet_run = quiet_run + 1 if quiet else 0
        settle_history.append(row)
        prev_height = height
        prev_toe = toe
        if quiet_run >= cfg.settle_window_probes:
            settled_at = sim_t
            break
    settle_pass = settled_at is not None
    settle_end_t = sim_t

    height_settled, axis, depth_meta = height_from_depth(require_fresh=True)
    pos_settled = positions()
    in_domain = (np.abs(pos_settled[:, 0]) <= half_extent) & (
        np.abs(pos_settled[:, 1]) <= half_extent
    )
    centres = pos_settled[released & in_domain]

    measure_depth = measure_from_height_field(height_settled, axis, cfg, radius, half_extent)
    # Cross-check against P3's OWN function on the readback coordinates.  If the
    # two agree, the depth path is a valid stand-in for the analytic operator.
    try:
        measure_analytic = p3.measure_repose_angle(centres, radius, p3cfg, geom)
    except Exception as exc:  # noqa: BLE001
        measure_analytic = {"error": repr(exc), "repose_angle_deg": float("nan")}
    height_analytic, ax_a, _ = p3.surface_height_field(centres, radius, half_extent, cell_m)

    # ---------------- G2 creep watch ---------------------------------------
    creep_history: list[dict[str, Any]] = []
    creep_start_height = height_settled
    creep_start_toe = measure_depth["toe_radius_m"]
    creep_probe = max(1, int(round(1.0 / dt)))
    n_creep = int(round(cfg.creep_watch_s))
    for _ in range(n_creep):
        for _ in range(creep_probe):
            world.step(render=False)
            sim_t += dt
            steps += 1
        h, a, _ = height_from_depth(require_fresh=True)
        dh = np.abs(h - creep_start_height)
        creep_history.append(
            {
                "sim_time_s": sim_t,
                "apex_m": float(h.max()),
                "max_cell_drift_mm": float(dh.max() * 1e3),
                "p99_cell_drift_mm": float(np.quantile(dh, 0.99) * 1e3),
                "apex_drift_mm": float((h.max() - creep_start_height.max()) * 1e3),
            }
        )
    height_after_creep, _, _ = height_from_depth(require_fresh=True)
    measure_after_creep = measure_from_height_field(
        height_after_creep, axis, cfg, radius, half_extent
    )
    creep = {
        "watch_s": float(cfg.creep_watch_s),
        "angle_before_deg": measure_depth["repose_angle_deg"],
        "angle_after_deg": measure_after_creep["repose_angle_deg"],
        "angle_drift_deg": float(
            measure_after_creep["repose_angle_deg"] - measure_depth["repose_angle_deg"]
        ),
        "toe_before_m": creep_start_toe,
        "toe_after_m": measure_after_creep["toe_radius_m"],
        "toe_drift_mm": float((measure_after_creep["toe_radius_m"] - creep_start_toe) * 1e3),
        "apex_drift_mm": creep_history[-1]["apex_drift_mm"] if creep_history else 0.0,
        "max_cell_drift_mm": creep_history[-1]["max_cell_drift_mm"] if creep_history else 0.0,
        "history": creep_history,
    }

    result: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "label": cfg.label,
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "non_claims": NON_CLAIMS,
        "environment": _env_stamp(),
        "derived": {
            "sphere_radius_m": radius,
            "solid_rest_offset_m": radius,
            "particle_contact_offset_m": pco,
            "particle_mass_kg": particle_mass,
            "half_extent_m": half_extent,
            "profile_cell_m": cell_m,
            "camera_height_m": cam_z,
            "camera_fx_px": fx,
            "floor_mm_per_pixel": float(cam_z / fx * 1e3),
            "physics_dt_s": dt,
            "template_bounding_diameter_m": float(template.bounding_diameter_m),
            "template_volume_m3": float(template.union_volume_m3),
        },
        "pour": {
            "protocol": "P3 batch pour: teleport each batch to drop_gap above the current heap top",
            "unreleased_particles_wait_in": f"flat holding pen at x={PEN_X_M} m (outside grid+frustum)",
            "batches": release_log,
            "pour_end_sim_s": pour_end_t,
            "pour_timeout": pour_timeout,
            "batches_released": next_batch,
        },
        "settle": {
            "criterion": (
                "depth-observable, thresholds set as fractions of the particle diameter "
                f"({diameter_mm:.3f} mm): per-probe p99 cell drift <= {thr_p99_mm:.3f} mm AND "
                f"max cell drift <= {thr_max_mm:.3f} mm AND apex drift <= {thr_apex_mm:.3f} mm "
                f"AND toe-radius drift <= {thr_toe_mm:.3f} mm, sustained over "
                f"{cfg.settle_window_probes} consecutive {cfg.settle_probe_interval_s} s probes, "
                "each probe using a render-fresh frame"
            ),
            "thresholds_mm": {
                "p99_cell_drift": thr_p99_mm,
                "max_cell_drift": thr_max_mm,
                "apex_drift": thr_apex_mm,
                "toe_drift": thr_toe_mm,
                "particle_diameter": diameter_mm,
            },
            "stale_frames_detected": int(stale_frames["count"]),
            "settled": bool(settle_pass),
            "settled_sim_time_s": settled_at,
            "settle_wall_end_sim_s": settle_end_t,
            "history": settle_history,
            "p3_speed_gate_mm_s": {"max": 60.0, "p99": 3.0, "rms": 1.5},
            "p3_speed_gate_pass": bool(
                settle_history
                and settle_history[-1]["speed_max_m_s"] * 1e3 <= 60.0
                and settle_history[-1]["speed_p99_m_s"] * 1e3 <= 3.0
                and settle_history[-1]["speed_rms_m_s"] * 1e3 <= 1.5
            ),
        },
        "creep": creep,
        "measurement_depth": {k: v for k, v in measure_depth.items() if k != "radial_profile"},
        "measurement_analytic_readback": {
            k: v for k, v in measure_analytic.items() if k != "radial_profile"
        },
        "depth_vs_analytic": {
            "angle_delta_deg": float(
                measure_depth["repose_angle_deg"]
                - float(measure_analytic.get("repose_angle_deg", float("nan")))
            ),
            "apex_delta_mm": float(
                (measure_depth["apex_height_m"]
                 - float(measure_analytic.get("apex_height_m", float("nan")))) * 1e3
            ),
            "height_field_rms_mm": float(
                np.sqrt(((height_settled - height_analytic) ** 2).mean()) * 1e3
            ),
            "height_field_max_abs_mm": float(
                np.abs(height_settled - height_analytic).max() * 1e3
            ),
            "note": (
                "analytic side = sim_pellet_model.measure_repose_angle on the particle "
                "coordinates read back from USD; depth side = the same definition applied to a "
                "top-down depth render. The readback is a CROSS-CHECK only; the depth number is "
                "the reported result because it is the observation the physical rig can make."
            ),
        },
        "depth_meta": depth_meta,
        "usd_authored": usd_authored,
        "particles": {
            "n_total": n,
            "n_released": int(released.sum()),
            "n_in_domain": int((released & in_domain).sum()),
            "n_escaped_domain": int((released & ~in_domain).sum()),
        },
        "timing": {
            "physics_steps": steps,
            "sim_time_s": sim_t,
            "wall_s": time.time() - t_start,
        },
    }

    npz_payload: dict[str, np.ndarray] = {
        "height_depth_m": height_settled.astype(np.float32),
        "height_analytic_m": height_analytic.astype(np.float32),
        "height_after_creep_m": height_after_creep.astype(np.float32),
        "grid_axis_m": axis.astype(np.float64),
        "radial_profile_depth": measure_depth["radial_profile"].astype(np.float64),
        "positions_settled_m": pos_settled.astype(np.float32),
        "released_mask": released,
        "batch_index": batch_index.astype(np.int32),
        "metadata_json": np.array(json.dumps(result, sort_keys=True, default=str)),
    }

    # ---------------- G3/G5 scoop ------------------------------------------
    if do_scoop:
        scoop = run_scoop(
            cfg=cfg,
            world=world,
            scoop_op=scoop_op,
            positions=positions,
            velocities=velocities,
            height_from_depth=height_from_depth,
            height_before=height_settled,
            axis=axis,
            cell_m=cell_m,
            radius=radius,
            dt=dt,
            Gf=Gf,
        )
        result["scoop"] = {k: v for k, v in scoop.items() if not isinstance(v, np.ndarray)}
        npz_payload["height_after_scoop_m"] = scoop["height_after"].astype(np.float32)
        npz_payload["positions_after_scoop_m"] = scoop["positions_after"].astype(np.float32)
        npz_payload["metadata_json"] = np.array(
            json.dumps(result, sort_keys=True, default=str)
        )

    if store_depth:
        npz_payload["depth_frame_m"] = depth_frame().astype(np.float32)

    annot.detach(rp)
    rp.destroy()

    npz_path = out_json.with_suffix(".npz")
    np.savez_compressed(npz_path, **npz_payload)
    result["npz"] = str(npz_path)
    result["npz_sha256"] = _sha256(npz_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    return result


def run_scoop(
    *,
    cfg: PbdConfig,
    world,
    scoop_op,
    positions,
    velocities,
    height_from_depth,
    height_before: np.ndarray,
    axis: np.ndarray,
    cell_m: float,
    radius: float,
    dt: float,
    Gf,
) -> dict[str, Any]:
    """Drive the kinematic scoop through the settled pile.

    Trajectory (fixed, non-learned, exactly as the proposal specifies):
    park -> descend outside the pile -> horizontal sweep towards the axis at a
    constant blade height -> vertical lift.  The gate is stability, not skill.
    """
    pos_before = positions()
    travel_steps = max(1, int(round(cfg.scoop_travel_s / dt)))
    lift_steps = max(1, int(round(cfg.scoop_lift_s / dt)))
    y0 = -cfg.scoop_start_r_m
    y1 = 0.0
    z_blade = cfg.scoop_plunge_z_m

    # descend outside the pile
    descend_steps = max(1, int(round(0.4 / dt)))
    for k in range(descend_steps):
        f = (k + 1) / descend_steps
        scoop_op.Set(Gf.Vec3d(0.0, y0, float(0.25 * (1 - f) + z_blade * f)))
        world.step(render=False)

    sx = cfg.scoop_width_m / 2.0
    sy = cfg.scoop_depth_m / 2.0

    def outside_bucket(p: np.ndarray, y_now: float, z_floor: float) -> np.ndarray:
        """Mask of particles NOT inside the bucket interior.

        The G3 height bound is about material LAUNCHED by the entry, so material
        legitimately riding up inside the bucket must not count against it.
        """
        return ~(
            (np.abs(p[:, 0]) <= sx)
            & (np.abs(p[:, 1] - y_now) <= sy)
            & (p[:, 2] >= z_floor)
            & (p[:, 2] <= z_floor + cfg.scoop_height_m)
        )

    speeds: list[float] = []
    heights: list[float] = []
    radii: list[float] = []
    for k in range(travel_steps):
        f = (k + 1) / travel_steps
        y_now = float(y0 + (y1 - y0) * f)
        scoop_op.Set(Gf.Vec3d(0.0, y_now, float(z_blade)))
        world.step(render=False)
        if k % 5 == 0:
            v = np.linalg.norm(velocities(), axis=1)
            p = positions()
            free = outside_bucket(p, y_now, z_blade) & (p[:, 0] < PEN_X_M / 2)
            speeds.append(float(v[p[:, 0] < PEN_X_M / 2].max()))
            if free.any():
                heights.append(float(p[free, 2].max()))
                radii.append(float(np.hypot(p[free, 0], p[free, 1]).max()))
    for k in range(lift_steps):
        f = (k + 1) / lift_steps
        scoop_op.Set(Gf.Vec3d(0.0, float(y1), float(z_blade + cfg.scoop_lift_z_m * f)))
        world.step(render=False)
        if k % 5 == 0:
            v = np.linalg.norm(velocities(), axis=1)
            p = positions()
            speeds.append(float(v[p[:, 0] < PEN_X_M / 2].max()))

    # count the catch BEFORE the retreat, while the bucket is still over the pile
    pos_lifted = positions()
    z_lo = z_blade + cfg.scoop_lift_z_m
    in_bucket = ~outside_bucket(pos_lifted, y1, z_lo)
    particles_in_bucket = int(in_bucket.sum())

    # retreat: carry the tool and its contents out of the camera frustum, then
    # let the pile stand still before the "after" frame is taken
    retreat_steps = max(1, int(round(cfg.scoop_retreat_s / dt)))
    for k in range(retreat_steps):
        f = (k + 1) / retreat_steps
        scoop_op.Set(
            Gf.Vec3d(
                float(cfg.scoop_retreat_x_m * f),
                float(y1),
                float(z_lo + (cfg.scoop_retreat_z_m - z_lo) * f),
            )
        )
        world.step(render=False)
    for _ in range(max(1, int(round(0.5 / dt)))):
        world.step(render=False)

    pos_after = positions()
    height_after, _, _ = height_from_depth(require_fresh=True)

    cell_area = cell_m * cell_m
    removed = np.clip(height_before - height_after, 0.0, None)
    added = np.clip(height_after - height_before, 0.0, None)
    domain = np.abs(axis).max() + cell_m
    in_flight = pos_after[:, 0] < PEN_X_M / 2
    # readback cross-check of the depth volume: how many particles are still
    # lying inside the measured grid at pile height
    still_in_grid = int(
        (
            (np.abs(pos_after[:, 0]) <= domain)
            & (np.abs(pos_after[:, 1]) <= domain)
            & (pos_after[:, 2] < 0.05)
        ).sum()
    )
    return {
        "trajectory": {
            "descend_to_z_m": z_blade,
            "sweep_y_m": [y0, y1],
            "lift_z_m": cfg.scoop_lift_z_m,
            "travel_s": cfg.scoop_travel_s,
            "lift_s": cfg.scoop_lift_s,
            "retreat_to_xz_m": [cfg.scoop_retreat_x_m, cfg.scoop_retreat_z_m],
            "retreat_s": cfg.scoop_retreat_s,
            "tool_speed_m_s": abs(y1 - y0) / cfg.scoop_travel_s,
        },
        "height_and_radius_metrics_exclude_bucket_interior": True,
        "max_particle_speed_during_entry_m_s": float(max(speeds)) if speeds else 0.0,
        "max_particle_height_during_entry_m": float(max(heights)) if heights else 0.0,
        "max_radial_extent_during_entry_m": float(max(radii)) if radii else 0.0,
        "max_radial_extent_before_m": float(
            np.hypot(pos_before[:, 0], pos_before[:, 1])[pos_before[:, 0] < PEN_X_M / 2].max()
        ),
        "max_radial_extent_after_m": float(
            np.hypot(pos_after[:, 0], pos_after[:, 1])[in_flight].max()
        ),
        "particles_in_bucket": particles_in_bucket,
        "particles_still_in_grid_after": still_in_grid,
        "removed_volume_m3": float(removed.sum() * cell_area),
        "spilled_back_volume_m3": float(added.sum() * cell_area),
        "net_removed_volume_m3": float((removed.sum() - added.sum()) * cell_area),
        "pile_volume_before_m3": float(height_before.sum() * cell_area),
        "pile_volume_after_m3": float(height_after.sum() * cell_area),
        "domain_half_extent_used_m": float(domain),
        "height_after": height_after,
        "positions_after": pos_after,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def selftest() -> int:
    """Prove `measure_from_height_field` recovers a known angle AND can FAIL."""
    ok = True
    lines = []
    for want in (18.0, 25.0, 30.0, 38.0, 45.0):
        h, ax = analytic_cone_height_field(want, 0.05, 0.12, 0.002)
        got = measure_from_height_field(h, ax, PbdConfig(), 2.0944e-3, 0.12)
        err = abs(got["repose_angle_deg"] - want)
        good = err < 0.6
        ok &= good
        lines.append(
            f"  cone {want:5.1f} deg -> {got['repose_angle_deg']:6.2f} deg "
            f"(err {err:.3f}, r2={got['fit_r_squared']:.5f}) {'OK' if good else 'FAIL'}"
        )
    # FAIL-capability: a flat pancake must not pass as a heap
    h, ax = analytic_cone_height_field(2.0, 0.006, 0.12, 0.002)
    flat = measure_from_height_field(h, ax, PbdConfig(), 2.0944e-3, 0.12)
    fails = not flat["measurement_pass"]
    lines.append(
        f"  2 deg pancake -> angle={flat['repose_angle_deg']:.2f} pass={flat['measurement_pass']} "
        f"(expected FAIL) {'OK' if fails else 'FAIL'}"
    )
    ok &= fails
    # FAIL-capability: a dome with no straight sidewall band must raise, not
    # silently return a number.
    dome_raises = False
    try:
        hd, axd = analytic_cone_height_field(2.0, 0.05, 0.12, 0.002)
        measure_from_height_field(hd, axd, PbdConfig(), 2.0944e-3, 0.12)
    except RuntimeError:
        dome_raises = True
    lines.append(
        f"  unbracketable profile -> raised={dome_raises} (expected True) "
        f"{'OK' if dome_raises else 'FAIL'}"
    )
    ok &= dome_raises
    # FAIL-capability: the pancake-with-a-cliff that PBD actually produced must
    # NOT be reported as a steep heap.
    hp, axp = analytic_pancake_height_field(0.115, 0.010, 0.0134, 0.20, 0.002)
    cake = measure_from_height_field(hp, axp, PbdConfig(), 2.0944e-3, 0.20)
    cake_rejected = not cake["checks"]["heap_is_conical"] and not cake["measurement_pass"]
    lines.append(
        f"  pancake+cliff -> angle={cake['repose_angle_deg']:.2f} r2={cake['fit_r_squared']:.3f} "
        f"plateau/toe={cake['plateau_over_toe_ratio']:.3f} conical={cake['checks']['heap_is_conical']} "
        f"(expected rejected) {'OK' if cake_rejected else 'FAIL'}"
    )
    ok &= cake_rejected
    # ...and a real cone must still pass that same check
    hc, axc = analytic_cone_height_field(30.0, 0.05, 0.20, 0.002)
    cone = measure_from_height_field(hc, axc, PbdConfig(), 2.0944e-3, 0.20)
    cone_ok = cone["checks"]["heap_is_conical"]
    lines.append(
        f"  30 deg cone -> plateau/toe={cone['plateau_over_toe_ratio']:.3f} "
        f"conical={cone_ok} (expected True) {'OK' if cone_ok else 'FAIL'}"
    )
    ok &= cone_ok
    # FAIL-capability: a cone whose toe reaches past 0.9*half_extent must trip
    # wall clearance even though its sidewall fit is perfect.
    h, ax = analytic_cone_height_field(25.0, 0.0536, 0.12, 0.002)
    wide = measure_from_height_field(h, ax, PbdConfig(), 2.0944e-3, 0.12)
    trips = not wide["checks"]["wall_clearance_ok"]
    lines.append(
        f"  domain-filling cone -> wall_clearance_ok={wide['checks']['wall_clearance_ok']} "
        f"(expected False) {'OK' if trips else 'FAIL'}"
    )
    ok &= trips
    print("\n".join(lines))
    print("SELFTEST_OK" if ok else "SELFTEST_FAIL")
    return 0 if ok else 1


def describe() -> int:
    cfg = PbdConfig()
    p3cfg = cfg.p3_config()
    tpl = p3.build_template(p3cfg.pellet(), "sphere", size_match="volume")
    geom = p3.derive_repose_geometry(p3cfg, tpl)
    radius = tpl.sphere_radius_m
    print(f"artifact                 {ARTIFACT} v{SCHEMA_VERSION}")
    print(f"repose definition        {p3.REPOSE_PRIMARY_DEFINITION} (imported from sim_pellet_model)")
    print(f"n_particles              {cfg.n_particles}")
    print(f"sphere radius (m)        {radius:.8f}   [volume-matched to the nominal pellet]")
    print(f"solidRestOffset (m)      {radius * cfg.rest_offset_scale:.8f}")
    print(f"particleContactOffset(m) {radius * cfg.rest_offset_scale / cfg.solid_to_contact_ratio:.8f}")
    print(f"particle mass (kg)       {tpl.mass_kg:.10f}")
    print(f"domain half extent (m)   {geom['half_extent_m']:.6f}")
    print(f"profile cell (m)         {cfg.profile_cell_mm * 1e-3:.6f}")
    cam_z = cfg.cam_fov_margin * geom["half_extent_m"] / (cfg.cam_aperture_mm / 2 / cfg.cam_focal_mm)
    fx = cfg.cam_res * cfg.cam_focal_mm / cfg.cam_aperture_mm
    print(f"camera height (m)        {cam_z:.6f}")
    print(f"floor mm / pixel         {cam_z / fx * 1e3:.4f}")
    print(f"physics dt (s)           {1.0 / cfg.time_steps_per_second:.6f}")
    for claim in NON_CLAIMS:
        print(f"NON_CLAIM: {claim}")
    print("PBD_PROBE_FORMAT_OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cells", type=str, help="path to a JSON list of config overrides")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--scoop", action="store_true", help="also run the scoop entry (G3/G5)")
    ap.add_argument("--store-depth", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun cells whose JSON already exists")
    args = ap.parse_args(argv)

    if args.describe:
        return describe()
    if args.selftest:
        return selftest()
    if not args.cells:
        ap.error("one of --describe / --selftest / --cells is required")

    cells = json.loads(Path(args.cells).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for spec in cells:
        cfg = replace(PbdConfig(), **spec)
        path = out_dir / f"cell_{cfg.label}.json"
        if path.exists() and not args.force:
            print(f"[probe] skip existing {path}", flush=True)
            continue
        todo.append((cfg, path))
    if not todo:
        print("PBD_PROBE_NOTHING_TO_DO")
        return 0

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    rc = 0
    try:
        for cfg, path in todo:
            print(f"[probe] === cell {cfg.label} ===", flush=True)
            try:
                res = run_cell(cfg, path, do_scoop=args.scoop, store_depth=args.store_depth)
                m = res["measurement_depth"]
                print(
                    f"[probe] {cfg.label}: angle_depth={m['repose_angle_deg']:.3f} deg "
                    f"r2={m['fit_r_squared']:.4f} pass={m['measurement_pass']} "
                    f"analytic={res['measurement_analytic_readback'].get('repose_angle_deg', float('nan')):.3f} "
                    f"settled={res['settle']['settled']} wall={res['timing']['wall_s']:.1f}s",
                    flush=True,
                )
            except BaseException as exc:  # noqa: BLE001
                import traceback

                rc = 1
                print(f"[probe] CELL_FAILED {cfg.label}: {exc!r}", flush=True)
                print(traceback.format_exc(), flush=True)
                path.write_text(
                    json.dumps(
                        {
                            "artifact": ARTIFACT,
                            "label": cfg.label,
                            "failed": True,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                            "config": asdict(cfg),
                        },
                        indent=1,
                    )
                )
    finally:
        app.close()
    print("PBD_PROBE_DONE" if rc == 0 else "PBD_PROBE_PARTIAL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
