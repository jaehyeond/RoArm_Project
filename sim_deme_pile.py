#!/usr/bin/env python3
"""Parameterized DEME pellet-pile settling environment.

The simulation uses SI units throughout.  The exported ``.npz`` contract is:

``positions_m``
    ``float64[N, 3]`` settled sphere-center positions in metres.
``velocities_m_s``
    ``float64[N, 3]`` settled linear velocities in metres/second.
``radii_m``
    ``float64[N]`` sphere radii in metres.
``particle_ids``
    ``int64[N]`` stable DEME owner IDs (the particles are owners ``0..N-1``).
``initial_positions_m``
    ``float64[N, 3]`` seeded, non-overlapping release positions in metres.
``box_bounds_m``
    ``float64[3, 2]`` closed-container bounds ``[[xmin,xmax], [ymin,ymax],
    [zmin,zmax]]`` in metres.  The top is physically open but remains the
    configured diagnostic extent.
``settle_history``
    ``float64[K, 6]`` rows with columns ``sim_time_s, max_speed_m_s,
    p99_speed_m_s, rms_speed_m_s, kinetic_energy_j, num_contacts``.
``metadata_json``
    A scalar Unicode JSON document containing the parameter set, coordinate
    contract, target geometry, and explicit settlement/containment gates.

Coordinate frame: right-handed, ``+x`` across the pile width, ``+y`` along its
length, ``+z`` upward; the origin is the centre of the container floor and the
floor is exactly ``z=0``.  Downstream heightmaps therefore use ``(x, y)`` as
the horizontal plane and ``z`` as height without a unit conversion.

Material warning
----------------
The default density, friction, restitution, rolling-friction, stiffness, and
target repose-angle values are provisional engineering placeholders.  The
pellets have not been procured or measured, and these defaults are NOT values
taken from the three prior papers.  ``bulk_density_g_cm3=0.55`` is the current
provisional bulk-density assumption.  The target repose angle is only used to
size a nominal target envelope; DEME produces the actual angle from the
contact parameters, so it must later be calibrated to the purchased pellets.

The deterministic NPZ deliberately excludes wall-clock timings.  Timings are
stored in a sibling ``*.timing.json`` so two same-seed scientific artifacts can
be byte-identical even though runtime cost varies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


OUT_DIR = Path("claudedocs/runtime_logs/sim_deme")
FORMAT_KEYS = (
    "box_bounds_m",
    "initial_positions_m",
    "metadata_json",
    "particle_ids",
    "positions_m",
    "radii_m",
    "settle_history",
    "velocities_m_s",
)
HISTORY_COLUMNS = (
    "sim_time_s",
    "max_speed_m_s",
    "p99_speed_m_s",
    "rms_speed_m_s",
    "kinetic_energy_j",
    "num_contacts",
)
PROVISIONAL_WARNING = (
    "UNMEASURED PROVISIONAL VALUES; pellets not procured; not sourced from prior papers"
)


@dataclass(frozen=True)
class PileConfig:
    n_particles: int = 18_796
    diameter_mm: float = 4.16
    seed: int = 460
    config_name: str = "practical"
    target_shape: str = "ridge"
    target_volume_cm3: float | None = None
    target_repose_angle_deg: float = 28.0
    ridge_length_width_ratio: float = 240.0 / 110.0
    box_width_mm: float | None = None
    box_length_mm: float | None = None
    bulk_density_g_cm3: float = 0.55
    particle_density_kg_m3: float = 950.0
    particle_mu: float = 0.50
    wall_mu: float = 0.50
    restitution: float = 0.30
    rolling_friction: float = 0.05
    young_modulus_pa: float = 1.0e7
    poisson_ratio: float = 0.30
    material_status: str = "provisional"
    dt_s: float = 2.0e-5
    cd_update_freq: int = 20
    spacing_factor: float = 1.10
    jitter_fraction_diameter: float = 0.015
    initialization_mode: str = "target_fcc"
    # For a ridge, this is the narrow cross-slope rain footprint.  A compact
    # release must fall and spread instead of preserving a rectangular slab.
    release_footprint_fraction: float = 0.22
    ridge_release_length_fraction: float = 0.85
    drop_height_mm: float | None = None
    check_interval_s: float = 0.05
    min_sim_time_s: float = 0.40
    max_sim_time_s: float = 2.00
    # max is a runaway/outlier guard; p99 and RMS are the bulk-settlement gate.
    # With ~19k particles a single wall-rattling sphere can remain at 10-70 mm/s
    # while p99<0.5 mm/s and RMS<0.8 mm/s.  Requiring max<10 mm/s therefore
    # misclassifies a stationary pile as moving.
    speed_max_mm_s: float = 100.0
    speed_p99_mm_s: float = 2.0
    speed_rms_mm_s: float = 1.0
    stable_checks: int = 5
    max_penetration_fraction_diameter: float = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a NumPy-loadable ZIP with fixed metadata and key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to replace unexpected temporary file: {tmp}")
    try:
        with zipfile.ZipFile(
            tmp,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for key in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.asarray(arrays[key]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info.create_system = 3
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _timing_path(npz_path: Path) -> Path:
    return npz_path.with_name(f"{npz_path.stem}.timing.json")


def _diameter_token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def default_output_path(config: PileConfig) -> Path:
    return OUT_DIR / (
        f"pile_{config.config_name}_d{_diameter_token(config.diameter_mm)}"
        f"_n{config.n_particles}_seed{config.seed}.npz"
    )


def _validate_config(config: PileConfig) -> None:
    numeric_positive = {
        "n_particles": config.n_particles,
        "diameter_mm": config.diameter_mm,
        "bulk_density_g_cm3": config.bulk_density_g_cm3,
        "particle_density_kg_m3": config.particle_density_kg_m3,
        "young_modulus_pa": config.young_modulus_pa,
        "dt_s": config.dt_s,
        "cd_update_freq": config.cd_update_freq,
        "spacing_factor": config.spacing_factor,
        "check_interval_s": config.check_interval_s,
        "max_sim_time_s": config.max_sim_time_s,
        "stable_checks": config.stable_checks,
    }
    for name, value in numeric_positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    if config.target_shape not in {"ridge", "cone", "box"}:
        raise ValueError(f"unsupported target_shape: {config.target_shape}")
    if config.material_status not in {"provisional", "measured"}:
        raise ValueError("material_status must be provisional or measured")
    if config.initialization_mode not in {"target_fcc", "rain_box"}:
        raise ValueError("initialization_mode must be target_fcc or rain_box")
    if not (0.0 <= config.poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in [0, 0.5)")
    if config.spacing_factor <= 1.0 + 2.0 * config.jitter_fraction_diameter:
        raise ValueError(
            "spacing_factor must exceed 1 + 2*jitter_fraction_diameter; "
            "otherwise seeded jitter can create initial penetration"
        )
    if not (0.0 < config.release_footprint_fraction <= 1.0):
        raise ValueError("release_footprint_fraction must be in (0, 1]")
    if not (0.0 < config.ridge_release_length_fraction <= 1.0):
        raise ValueError("ridge_release_length_fraction must be in (0, 1]")
    if config.min_sim_time_s > config.max_sim_time_s:
        raise ValueError("min_sim_time_s cannot exceed max_sim_time_s")
    if config.stable_checks * config.check_interval_s > config.max_sim_time_s:
        raise ValueError("stable gate duration exceeds max_sim_time_s")
    for name in ("box_width_mm", "box_length_mm", "target_volume_cm3", "drop_height_mm"):
        value = getattr(config, name)
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f"{name} must be finite and > 0 when provided")


def derive_geometry(config: PileConfig) -> dict[str, Any]:
    _validate_config(config)
    radius_m = config.diameter_mm * 0.5e-3
    diameter_m = 2.0 * radius_m
    sphere_volume_m3 = 4.0 / 3.0 * math.pi * radius_m**3
    particle_mass_kg = sphere_volume_m3 * config.particle_density_kg_m3
    total_mass_kg = particle_mass_kg * config.n_particles
    bulk_density_kg_m3 = config.bulk_density_g_cm3 * 1000.0
    auto_target_volume_m3 = total_mass_kg / bulk_density_kg_m3
    target_volume_m3 = (
        config.target_volume_cm3 * 1.0e-6
        if config.target_volume_cm3 is not None
        else auto_target_volume_m3
    )
    # 0.74 is the densest equal-sphere packing fraction.  Anything smaller is
    # impossible before material calibration, regardless of friction.
    minimum_physical_bulk_volume_m3 = config.n_particles * sphere_volume_m3 / 0.74
    if target_volume_m3 < minimum_physical_bulk_volume_m3:
        raise ValueError(
            f"target volume {target_volume_m3 * 1e6:.3f} cm^3 is below the equal-sphere "
            f"packing lower bound {minimum_physical_bulk_volume_m3 * 1e6:.3f} cm^3"
        )

    angle_rad = math.radians(config.target_repose_angle_deg)
    if not (0.0 < angle_rad < math.pi / 2.0):
        raise ValueError("target_repose_angle_deg must be in (0, 90)")
    tan_angle = math.tan(angle_rad)
    if config.target_shape == "ridge":
        aspect = config.ridge_length_width_ratio
        target_width_m = (4.0 * target_volume_m3 / (aspect * tan_angle)) ** (1.0 / 3.0)
        target_length_m = aspect * target_width_m
        target_height_m = 0.5 * target_width_m * tan_angle
    elif config.target_shape == "cone":
        target_radius_m = (3.0 * target_volume_m3 / (math.pi * tan_angle)) ** (1.0 / 3.0)
        target_width_m = target_length_m = 2.0 * target_radius_m
        target_height_m = target_radius_m * tan_angle
    else:
        aspect = config.ridge_length_width_ratio
        height_to_width = 0.30
        target_width_m = (target_volume_m3 / (aspect * height_to_width)) ** (1.0 / 3.0)
        target_length_m = aspect * target_width_m
        target_height_m = height_to_width * target_width_m

    margin_m = max(3.0 * diameter_m, 0.010)
    box_width_m = (
        config.box_width_mm * 1.0e-3
        if config.box_width_mm is not None
        else target_width_m + 2.0 * margin_m
    )
    box_length_m = (
        config.box_length_mm * 1.0e-3
        if config.box_length_mm is not None
        else target_length_m + 2.0 * margin_m
    )
    if box_width_m < target_width_m + 2.0 * diameter_m:
        raise ValueError("box_width_mm is too small for the target envelope and particle margin")
    if box_length_m < target_length_m + 2.0 * diameter_m:
        raise ValueError("box_length_mm is too small for the target envelope and particle margin")

    release_width_m = min(
        config.release_footprint_fraction * target_width_m,
        box_width_m - 4.0 * radius_m,
    )
    release_length_fraction = (
        config.ridge_release_length_fraction
        if config.target_shape == "ridge"
        else config.release_footprint_fraction
    )
    release_length_m = min(
        release_length_fraction * target_length_m,
        box_length_m - 4.0 * radius_m,
    )
    spacing_m = config.spacing_factor * diameter_m
    nx = max(1, int(math.floor(release_width_m / spacing_m)))
    ny = max(1, int(math.floor(release_length_m / spacing_m)))
    nz = int(math.ceil(config.n_particles / (nx * ny)))
    release_width_used_m = (nx - 1) * spacing_m + diameter_m
    release_length_used_m = (ny - 1) * spacing_m + diameter_m
    if release_width_used_m > box_width_m - 2.0 * radius_m + 1.0e-12:
        raise ValueError("initial x lattice does not fit inside the box")
    if release_length_used_m > box_length_m - 2.0 * radius_m + 1.0e-12:
        raise ValueError("initial y lattice does not fit inside the box")
    drop_height_m = config.drop_height_mm * 1.0e-3 if config.drop_height_mm is not None else (
        0.25 * diameter_m
        if config.initialization_mode == "target_fcc"
        else max(3.0 * diameter_m, 0.20 * target_height_m)
    )
    initial_top_m = (
        drop_height_m + target_height_m + radius_m
        if config.initialization_mode == "target_fcc"
        else drop_height_m + radius_m + (nz - 1) * spacing_m
    )
    box_height_m = max(
        initial_top_m + 5.0 * diameter_m,
        target_height_m + 6.0 * diameter_m,
        0.080,
    )
    theoretical_initial_gap_m = (
        spacing_m - diameter_m - 2.0 * config.jitter_fraction_diameter * diameter_m
    )
    if theoretical_initial_gap_m <= 0.0:
        raise ValueError("derived initial lattice can overlap")
    return {
        "radius_m": radius_m,
        "diameter_m": diameter_m,
        "sphere_volume_m3": sphere_volume_m3,
        "particle_mass_kg": particle_mass_kg,
        "total_mass_kg": total_mass_kg,
        "bulk_density_kg_m3": bulk_density_kg_m3,
        "auto_target_volume_m3": auto_target_volume_m3,
        "target_volume_m3": target_volume_m3,
        "minimum_physical_bulk_volume_m3": minimum_physical_bulk_volume_m3,
        "target_width_m": target_width_m,
        "target_length_m": target_length_m,
        "target_height_m": target_height_m,
        "box_width_m": box_width_m,
        "box_length_m": box_length_m,
        "box_height_m": box_height_m,
        "margin_m": margin_m,
        "release_width_m": release_width_m,
        "release_length_m": release_length_m,
        "release_length_fraction": release_length_fraction,
        "spacing_m": spacing_m,
        "lattice_nx": nx,
        "lattice_ny": ny,
        "lattice_nz": nz,
        "drop_height_m": drop_height_m,
        "theoretical_initial_gap_m": theoretical_initial_gap_m,
    }


def _generate_rain_box_positions(config: PileConfig, geometry: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    nx = int(geometry["lattice_nx"])
    ny = int(geometry["lattice_ny"])
    nz = int(geometry["lattice_nz"])
    spacing = float(geometry["spacing_m"])
    radius = float(geometry["radius_m"])
    diameter = float(geometry["diameter_m"])
    x = (np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)) * spacing
    y = (np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)) * spacing
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    positions: list[np.ndarray] = []
    remaining = config.n_particles
    for layer in range(nz):
        order = rng.permutation(xy.shape[0])
        take = min(remaining, xy.shape[0])
        chosen = xy[order[:take]]
        z = geometry["drop_height_m"] + radius + layer * spacing
        layer_positions = np.column_stack(
            [chosen, np.full(take, z, dtype=np.float64)]
        )
        positions.append(layer_positions)
        remaining -= take
        if remaining == 0:
            break
    result = np.concatenate(positions, axis=0)
    if result.shape != (config.n_particles, 3):
        raise AssertionError(f"initial position shape mismatch: {result.shape}")
    jitter_amp = config.jitter_fraction_diameter * diameter
    result += rng.uniform(-jitter_amp, jitter_amp, size=result.shape)
    # Do not let z jitter violate the analytical non-penetration floor margin.
    result[:, 2] = np.maximum(result[:, 2], radius + geometry["drop_height_m"] * 0.5)
    if np.max(np.abs(result[:, 0])) + radius > 0.5 * geometry["box_width_m"]:
        raise AssertionError("seeded initial x positions exceed box")
    if np.max(np.abs(result[:, 1])) + radius > 0.5 * geometry["box_length_m"]:
        raise AssertionError("seeded initial y positions exceed box")
    if np.max(result[:, 2]) + radius > geometry["box_height_m"]:
        raise AssertionError("seeded initial z positions exceed diagnostic extent")
    return result


def _generate_target_fcc_positions(
    config: PileConfig,
    geometry: dict[str, Any],
) -> np.ndarray:
    """Seed a non-overlapping FCC sample inside the requested bulk envelope.

    Sampling a seeded subset of a face-centred cubic lattice gives the target
    provisional bulk fraction without an initially interpenetrating block.
    Particle centres may extend by one radius beyond the ideal continuum
    envelope so the *sphere surfaces*, not merely centres, approximate it.
    """
    rng = np.random.default_rng(config.seed)
    radius = float(geometry["radius_m"])
    diameter = float(geometry["diameter_m"])
    nearest = float(geometry["spacing_m"])
    cell = math.sqrt(2.0) * nearest
    half_width = 0.5 * float(geometry["target_width_m"])
    half_length = 0.5 * float(geometry["target_length_m"])
    height = float(geometry["target_height_m"])
    clearance = float(geometry["drop_height_m"])
    nx = int(math.ceil((half_width + radius) / cell)) + 2
    ny = int(math.ceil((half_length + radius) / cell)) + 2
    nz = int(math.ceil((height + 2.0 * radius) / cell)) + 2
    basis = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    candidates: list[np.ndarray] = []
    for bx, by, bz in basis:
        ix = np.arange(-nx, nx + 1, dtype=np.float64) + bx
        iy = np.arange(-ny, ny + 1, dtype=np.float64) + by
        iz = np.arange(0, nz + 1, dtype=np.float64) + bz
        xx, yy, zz = np.meshgrid(ix * cell, iy * cell, iz * cell, indexing="ij")
        z_local = radius + zz.ravel()
        x = xx.ravel()
        y = yy.ravel()
        z_fraction = np.clip((z_local - radius) / height, 0.0, 1.0)
        if config.target_shape == "ridge":
            inside = (
                (np.abs(y) <= half_length + radius)
                & (np.abs(x) <= half_width * (1.0 - z_fraction) + radius)
                & (z_local <= height + radius)
            )
        elif config.target_shape == "cone":
            allowed_radius = half_width * (1.0 - z_fraction) + radius
            inside = (
                (np.hypot(x, y) <= allowed_radius)
                & (z_local <= height + radius)
            )
        else:
            inside = (
                (np.abs(x) <= half_width + radius)
                & (np.abs(y) <= half_length + radius)
                & (z_local <= height + radius)
            )
        candidates.append(np.column_stack([x[inside], y[inside], z_local[inside] + clearance]))
    pool = np.concatenate(candidates, axis=0)
    if pool.shape[0] < config.n_particles:
        raise ValueError(
            "target FCC envelope contains too few non-overlapping sites: "
            f"{pool.shape[0]} < {config.n_particles}; reduce spacing_factor or increase target volume"
        )
    selected = pool[rng.choice(pool.shape[0], size=config.n_particles, replace=False)].copy()
    jitter_amp = config.jitter_fraction_diameter * diameter
    selected += rng.uniform(-jitter_amp, jitter_amp, size=selected.shape)
    selected[:, 2] = np.maximum(selected[:, 2], radius + 0.5 * clearance)
    if np.max(np.abs(selected[:, 0])) + radius > 0.5 * geometry["box_width_m"]:
        raise AssertionError("seeded target FCC x positions exceed box")
    if np.max(np.abs(selected[:, 1])) + radius > 0.5 * geometry["box_length_m"]:
        raise AssertionError("seeded target FCC y positions exceed box")
    if np.max(selected[:, 2]) + radius > geometry["box_height_m"]:
        raise AssertionError("seeded target FCC z positions exceed diagnostic extent")
    geometry["target_fcc_candidate_count"] = int(pool.shape[0])
    geometry["target_fcc_selected_fraction"] = config.n_particles / float(pool.shape[0])
    return selected


def generate_initial_positions(config: PileConfig, geometry: dict[str, Any]) -> np.ndarray:
    if config.initialization_mode == "target_fcc":
        return _generate_target_fcc_positions(config, geometry)
    return _generate_rain_box_positions(config, geometry)


def minimum_surface_gap_m(positions: np.ndarray, diameter_m: float) -> float:
    """Return the closest center distance minus one diameter using a spatial hash."""
    points = np.asarray(positions, dtype=np.float64)
    cell_size = 1.5 * float(diameter_m)
    origin = points.min(axis=0) - cell_size
    cells = np.floor((points - origin) / cell_size).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    min_distance_sq = math.inf
    offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    )
    for index, cell in enumerate(cells):
        cx, cy, cz = (int(cell[0]), int(cell[1]), int(cell[2]))
        for dx, dy, dz in offsets:
            for other in buckets.get((cx + dx, cy + dy, cz + dz), ()):
                delta = points[index] - points[other]
                distance_sq = float(np.dot(delta, delta))
                if distance_sq < min_distance_sq:
                    min_distance_sq = distance_sq
        buckets.setdefault((cx, cy, cz), []).append(index)
    if not math.isfinite(min_distance_sq):
        return math.inf
    return math.sqrt(min_distance_sq) - float(diameter_m)


def containment_metrics(
    positions: np.ndarray,
    radii: np.ndarray,
    box_bounds: np.ndarray,
) -> dict[str, Any]:
    points = np.asarray(positions, dtype=np.float64)
    rr = np.asarray(radii, dtype=np.float64)
    lower_margin = points - rr[:, None] - box_bounds[:, 0][None, :]
    upper_margin = box_bounds[:, 1][None, :] - points - rr[:, None]
    axis_min_margin = np.minimum(lower_margin.min(axis=0), upper_margin.min(axis=0))
    return {
        "axis_min_surface_margin_m": axis_min_margin,
        "minimum_surface_margin_m": float(axis_min_margin.min()),
        "positions_min_m": points.min(axis=0),
        "positions_max_m": points.max(axis=0),
        "surface_min_m": (points - rr[:, None]).min(axis=0),
        "surface_max_m": (points + rr[:, None]).max(axis=0),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_simulation(config: PileConfig, output: Path, *, overwrite: bool = False) -> None:
    if output.suffix != ".npz":
        raise ValueError("simulation output must end in .npz")
    timing_output = _timing_path(output)
    if not overwrite and (output.exists() or timing_output.exists()):
        raise FileExistsError(f"refusing to overwrite {output} or {timing_output}")
    if overwrite:
        raise ValueError("--overwrite is intentionally unsupported for forward-only evidence")

    try:
        import DEME
    except ImportError as exc:
        raise RuntimeError(
            "DEME is not installed in this interpreter; use the roarm conda Python"
        ) from exc
    deme_version = _package_version("DEME")
    if deme_version != "2.4.0":
        raise RuntimeError(f"DEME version contract failed: {deme_version} != 2.4.0")

    run_started = datetime.now(timezone.utc).isoformat()
    total_t0 = time.perf_counter()
    geometry = derive_geometry(config)
    initial_positions = generate_initial_positions(config, geometry)
    initial_gap = minimum_surface_gap_m(initial_positions, geometry["diameter_m"])
    if initial_gap <= 0.0:
        raise RuntimeError(f"initial penetration detected: surface gap {initial_gap:.9g} m")

    solver = DEME.DEMSolver()
    solver.SetVerbosity("ERROR")
    particle_material = solver.LoadMaterial(
        {
            "E": config.young_modulus_pa,
            "nu": config.poisson_ratio,
            "CoR": config.restitution,
            "mu": config.particle_mu,
            "Crr": config.rolling_friction,
        }
    )
    wall_material = solver.LoadMaterial(
        {
            "E": config.young_modulus_pa,
            "nu": config.poisson_ratio,
            "CoR": config.restitution,
            "mu": config.wall_mu,
            "Crr": config.rolling_friction,
        }
    )
    solver.SetMaterialPropertyPair("mu", particle_material, wall_material, config.wall_mu)
    solver.SetMaterialPropertyPair("CoR", particle_material, wall_material, config.restitution)
    solver.SetMaterialPropertyPair(
        "Crr", particle_material, wall_material, config.rolling_friction
    )
    solver.UseFrictionalHertzianModel()
    sphere_type = solver.LoadSphereType(
        geometry["particle_mass_kg"], geometry["radius_m"], particle_material
    )
    solver.AddClumps(sphere_type, initial_positions)
    half_width = 0.5 * geometry["box_width_m"]
    half_length = 0.5 * geometry["box_length_m"]
    solver.InstructBoxDomainDimension(
        (-half_width, half_width),
        (-half_length, half_length),
        (0.0, geometry["box_height_m"]),
    )
    solver.InstructBoxDomainBoundingBC("top_open", wall_material)
    solver.SetInitTimeStep(config.dt_s)
    solver.SetGravitationalAcceleration([0.0, 0.0, -9.81])
    # Frequency=20 is the measured production setting from D460.  A diagnostic
    # frequency=1 run still was not bit-exact across independent processes, so
    # paying its ~3x small-case cost does not buy determinism.
    solver.SetCDUpdateFreq(config.cd_update_freq)
    # Keep DEME's measured fast production path: adaptive contact/bin updates,
    # unsorted contact arrays, and direct atomic force collection.  Diagnostic
    # runs with all exposed determinism controls enabled still diverged across
    # independent processes and made the 18,796-particle run ~3x slower, so
    # those controls are recorded as a failed experiment rather than imposed.

    init_t0 = time.perf_counter()
    solver.Initialize()
    initialization_wall_s = time.perf_counter() - init_t0
    if solver.GetNumClumps() != config.n_particles:
        raise RuntimeError(
            f"DEME clump count mismatch: {solver.GetNumClumps()} != {config.n_particles}"
        )
    print(
        f"Initialize: {initialization_wall_s:.3f}s; N={config.n_particles}; "
        f"box={geometry['box_width_m']*1e3:.1f}x{geometry['box_length_m']*1e3:.1f}"
        f"x{geometry['box_height_m']*1e3:.1f}mm; initial_gap={initial_gap*1e3:.4f}mm",
        flush=True,
    )

    history: list[list[float]] = []
    stable_count = 0
    settled = False
    dynamics_t0 = time.perf_counter()
    max_chunks = int(math.ceil(config.max_sim_time_s / config.check_interval_s))
    particle_mass = float(geometry["particle_mass_kg"])
    for chunk in range(max_chunks):
        solver.DoDynamicsThenSync(config.check_interval_s)
        positions = np.asarray(
            solver.GetOwnerPosition(0, config.n_particles), dtype=np.float64
        )
        velocities = np.asarray(
            solver.GetOwnerVelocity(0, config.n_particles), dtype=np.float64
        )
        if positions.shape != (config.n_particles, 3):
            raise RuntimeError(f"DEME position shape mismatch: {positions.shape}")
        if velocities.shape != (config.n_particles, 3):
            raise RuntimeError(f"DEME velocity shape mismatch: {velocities.shape}")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise RuntimeError("DEME returned NaN/Inf state")
        speeds = np.linalg.norm(velocities, axis=1)
        sim_time_s = float(solver.GetSimTime())
        max_speed = float(speeds.max())
        p99_speed = float(np.quantile(speeds, 0.99))
        rms_speed = float(np.sqrt(np.mean(speeds * speeds)))
        kinetic_energy = float(0.5 * particle_mass * np.sum(speeds * speeds))
        contacts = int(solver.GetNumContacts())
        history.append(
            [
                sim_time_s,
                max_speed,
                p99_speed,
                rms_speed,
                kinetic_energy,
                float(contacts),
            ]
        )
        gate_pass = bool(
            sim_time_s + 1.0e-12 >= config.min_sim_time_s
            and max_speed <= config.speed_max_mm_s * 1.0e-3
            and p99_speed <= config.speed_p99_mm_s * 1.0e-3
            and rms_speed <= config.speed_rms_mm_s * 1.0e-3
        )
        stable_count = stable_count + 1 if gate_pass else 0
        print(
            f"settle {chunk+1:02d}/{max_chunks}: t={sim_time_s:.3f}s "
            f"v(max/p99/rms)={max_speed*1e3:.3f}/{p99_speed*1e3:.3f}/"
            f"{rms_speed*1e3:.3f}mm/s contacts={contacts} stable={stable_count}/"
            f"{config.stable_checks}",
            flush=True,
        )
        if stable_count >= config.stable_checks:
            settled = True
            break
    dynamics_wall_s = time.perf_counter() - dynamics_t0
    if not settled:
        last = history[-1]
        raise RuntimeError(
            "SETTLEMENT_GATE_FAIL: max simulation time reached; "
            f"last max/p99/rms={last[1]*1e3:.3f}/{last[2]*1e3:.3f}/"
            f"{last[3]*1e3:.3f} mm/s"
        )

    post_t0 = time.perf_counter()
    positions = np.asarray(
        solver.GetOwnerPosition(0, config.n_particles), dtype=np.float64
    )
    velocities = np.asarray(
        solver.GetOwnerVelocity(0, config.n_particles), dtype=np.float64
    )
    radii = np.full(config.n_particles, geometry["radius_m"], dtype=np.float64)
    particle_ids = np.arange(config.n_particles, dtype=np.int64)
    box_bounds = np.asarray(
        [
            [-half_width, half_width],
            [-half_length, half_length],
            [0.0, geometry["box_height_m"]],
        ],
        dtype=np.float64,
    )
    containment = containment_metrics(positions, radii, box_bounds)
    final_gap = minimum_surface_gap_m(positions, geometry["diameter_m"])
    allowed_overlap = config.max_penetration_fraction_diameter * geometry["diameter_m"]
    if containment["minimum_surface_margin_m"] < -allowed_overlap:
        raise RuntimeError(
            "CONTAINMENT_GATE_FAIL: surface is outside box by "
            f"{-containment['minimum_surface_margin_m']*1e3:.4f} mm"
        )
    if final_gap < -allowed_overlap:
        raise RuntimeError(
            f"PENETRATION_GATE_FAIL: min surface gap {final_gap*1e3:.4f} mm, "
            f"allowance {-allowed_overlap*1e3:.4f} mm"
        )

    final_speed = np.linalg.norm(velocities, axis=1)
    metadata: dict[str, Any] = {
        "artifact": "DEME_PELLET_PILE_V1",
        "schema_version": 1,
        "scientific_authority": "float64 NPZ arrays; Rerun is inspection-only Float32 copy",
        "coordinate_frame": {
            "handedness": "right-handed",
            "axes": {"x": "pile width", "y": "pile length", "z": "up"},
            "origin": "container floor center",
            "floor_z_m": 0.0,
            "units": "SI: positions/radii=m, velocities=m/s, time=s, mass=kg",
        },
        "array_contract": {
            "positions_m": [config.n_particles, 3],
            "velocities_m_s": [config.n_particles, 3],
            "radii_m": [config.n_particles],
            "particle_ids": [config.n_particles],
            "initial_positions_m": [config.n_particles, 3],
            "box_bounds_m": [3, 2],
            "settle_history": [len(history), len(HISTORY_COLUMNS)],
            "settle_history_columns": HISTORY_COLUMNS,
        },
        "config": asdict(config),
        "material_assumptions": {
            "status": config.material_status,
            "warning": PROVISIONAL_WARNING,
            "bulk_density_g_cm3": config.bulk_density_g_cm3,
            "target_repose_angle_deg": config.target_repose_angle_deg,
            "target_repose_angle_role": "target envelope sizing only; not a DEM force input",
            "prior_paper_numeric_source": None,
        },
        "derived_geometry": geometry,
        "settling_gate": {
            "settled": True,
            "rule": (
                "after min_sim_time_s, max/p99/rms speed must each be <= its threshold "
                "for stable_checks consecutive check intervals"
            ),
            "stable_checks_observed": stable_count,
            "settled_sim_time_s": float(history[-1][0]),
            "thresholds_m_s": {
                "max": config.speed_max_mm_s * 1.0e-3,
                "p99": config.speed_p99_mm_s * 1.0e-3,
                "rms": config.speed_rms_mm_s * 1.0e-3,
            },
            "final_m_s": {
                "max": float(final_speed.max()),
                "p99": float(np.quantile(final_speed, 0.99)),
                "rms": float(np.sqrt(np.mean(final_speed * final_speed))),
            },
        },
        "initialization_gate": {
            "no_initial_penetration": bool(initial_gap > 0.0),
            "minimum_surface_gap_m": initial_gap,
            "theoretical_minimum_surface_gap_m": geometry["theoretical_initial_gap_m"],
        },
        "post_settle_gates": {
            "containment_pass": bool(
                containment["minimum_surface_margin_m"] >= -allowed_overlap
            ),
            "penetration_pass": bool(final_gap >= -allowed_overlap),
            "allowed_overlap_m": allowed_overlap,
            "minimum_pair_surface_gap_m": final_gap,
            **containment,
        },
        "software": {
            "DEME": deme_version,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "determinism": {
            "seed": config.seed,
            "npz_zip_timestamp": "1980-01-01T00:00:00Z",
            "wall_clock_excluded": True,
            "adaptive_contact_update_frequency": True,
            "adaptive_bin_size": True,
            "contact_detection_update_frequency_steps": config.cd_update_freq,
            "contact_pairs_sorted": False,
            "cub_force_collection": False,
            "independent_raw_final_bit_exact_supported": False,
            "independent_reproducibility_gate": "seeded input exact + 5mm heightmap tolerance",
        },
    }
    arrays = {
        "positions_m": positions,
        "velocities_m_s": velocities,
        "radii_m": radii,
        "particle_ids": particle_ids,
        "initial_positions_m": initial_positions,
        "box_bounds_m": box_bounds,
        "settle_history": np.asarray(history, dtype=np.float64),
        "metadata_json": np.asarray(_canonical_json(metadata)),
    }
    if set(arrays) != set(FORMAT_KEYS):
        raise AssertionError(f"internal format key mismatch: {sorted(arrays)}")
    _write_deterministic_npz(output, arrays)
    output_hash = _sha256(output)
    postprocess_wall_s = time.perf_counter() - post_t0
    total_wall_s = time.perf_counter() - total_t0
    timing = {
        "artifact": "DEME_PELLET_PILE_TIMING_V1",
        "scientific_npz": str(output),
        "scientific_npz_sha256": output_hash,
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_particles": config.n_particles,
        "diameter_mm": config.diameter_mm,
        "seed": config.seed,
        "settled_sim_time_s": float(history[-1][0]),
        "initialization_wall_s": initialization_wall_s,
        "dynamics_wall_s": dynamics_wall_s,
        "postprocess_and_write_wall_s": postprocess_wall_s,
        "total_wall_s": total_wall_s,
        "steps": int(round(float(history[-1][0]) / config.dt_s)),
        "steps_per_wall_s_dynamics": int(round(float(history[-1][0]) / config.dt_s))
        / dynamics_wall_s,
        "DEME": deme_version,
        "gpu": "RTX 4090 Laptop 16GB (project machine fact)",
        "package_install_performed_by_this_task": False,
    }
    _write_json(timing_output, timing)
    print(
        f"SETTLEMENT_OK output={output} sha256={output_hash} "
        f"sim={history[-1][0]:.3f}s dynamics_wall={dynamics_wall_s:.3f}s "
        f"total_wall={total_wall_s:.3f}s",
        flush=True,
    )


def load_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        if keys != sorted(FORMAT_KEYS):
            raise ValueError(f"{path}: keys {keys} != {sorted(FORMAT_KEYS)}")
        arrays = {key: np.array(archive[key], copy=True) for key in keys}
    metadata_value = arrays["metadata_json"]
    if metadata_value.shape != () or metadata_value.dtype.kind != "U":
        raise ValueError(f"{path}: metadata_json must be a scalar Unicode array")
    metadata = json.loads(str(metadata_value.item()))
    return arrays, metadata


def validate_output(
    paths: Iterable[Path], *, expected_n: int | None = None, require_timing: bool = False
) -> None:
    checked = 0
    for path in paths:
        arrays, metadata = load_artifact(path)
        n = int(metadata["config"]["n_particles"])
        if expected_n is not None and n != expected_n:
            raise ValueError(f"{path}: expected N={expected_n}, got {n}")
        expected_shapes = {
            "positions_m": (n, 3),
            "velocities_m_s": (n, 3),
            "radii_m": (n,),
            "particle_ids": (n,),
            "initial_positions_m": (n, 3),
            "box_bounds_m": (3, 2),
        }
        for key, shape in expected_shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"{path}: {key} shape {arrays[key].shape} != {shape}")
        for key in (
            "positions_m",
            "velocities_m_s",
            "radii_m",
            "initial_positions_m",
            "box_bounds_m",
            "settle_history",
        ):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{path}: {key} contains NaN/Inf")
        if not np.array_equal(arrays["particle_ids"], np.arange(n, dtype=np.int64)):
            raise ValueError(f"{path}: particle_ids are not 0..N-1")
        if arrays["settle_history"].ndim != 2 or arrays["settle_history"].shape[1] != 6:
            raise ValueError(f"{path}: invalid settle_history shape")
        if not metadata["settling_gate"]["settled"]:
            raise ValueError(f"{path}: settlement gate is false")
        if not metadata["initialization_gate"]["no_initial_penetration"]:
            raise ValueError(f"{path}: initial penetration gate is false")
        if not metadata["post_settle_gates"]["containment_pass"]:
            raise ValueError(f"{path}: containment gate is false")
        if not metadata["post_settle_gates"]["penetration_pass"]:
            raise ValueError(f"{path}: penetration gate is false")
        containment = containment_metrics(
            arrays["positions_m"], arrays["radii_m"], arrays["box_bounds_m"]
        )
        allowance = float(metadata["post_settle_gates"]["allowed_overlap_m"])
        if containment["minimum_surface_margin_m"] < -allowance:
            raise ValueError(f"{path}: reloaded state is outside container")
        recorded_min = np.asarray(
            metadata["post_settle_gates"]["positions_min_m"], dtype=np.float64
        )
        recorded_max = np.asarray(
            metadata["post_settle_gates"]["positions_max_m"], dtype=np.float64
        )
        if not np.array_equal(recorded_min, arrays["positions_m"].min(axis=0)):
            raise ValueError(f"{path}: recorded minimum range differs from arrays")
        if not np.array_equal(recorded_max, arrays["positions_m"].max(axis=0)):
            raise ValueError(f"{path}: recorded maximum range differs from arrays")
        timing_path = _timing_path(path)
        timing: dict[str, Any] | None = None
        if timing_path.is_file():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            if timing["scientific_npz_sha256"] != _sha256(path):
                raise ValueError(f"{timing_path}: NPZ hash mismatch")
        if require_timing:
            if timing is None:
                raise FileNotFoundError(timing_path)
            if float(timing["dynamics_wall_s"]) <= 0 or float(timing["total_wall_s"]) <= 0:
                raise ValueError(f"{timing_path}: non-positive timing")
            print(
                f"PRACTICAL_COST_OK n={n} dynamics_wall_s={timing['dynamics_wall_s']:.6f} "
                f"total_wall_s={timing['total_wall_s']:.6f}"
            )
        bounds = arrays["box_bounds_m"]
        pos_min = arrays["positions_m"].min(axis=0)
        pos_max = arrays["positions_m"].max(axis=0)
        print(
            f"VALID {path}: N={n} xyz_min_mm={np.round(pos_min*1e3,3).tolist()} "
            f"xyz_max_mm={np.round(pos_max*1e3,3).tolist()} "
            f"box_mm={np.round(bounds*1e3,3).tolist()}"
        )
        checked += 1
    print(f"OUTPUT_VALIDATION_OK count={checked}")


def compare_output(left: Path, right: Path) -> None:
    left_arrays, _ = load_artifact(left)
    right_arrays, _ = load_artifact(right)
    if sorted(left_arrays) != sorted(right_arrays):
        raise ValueError("artifact key sets differ")
    for key in sorted(left_arrays):
        if not np.array_equal(left_arrays[key], right_arrays[key], equal_nan=True):
            differing = int(np.count_nonzero(left_arrays[key] != right_arrays[key]))
            raise ValueError(f"array mismatch for {key}: {differing} elements differ")
    left_hash = _sha256(left)
    right_hash = _sha256(right)
    if left_hash != right_hash:
        raise ValueError(f"NPZ byte hashes differ: {left_hash} != {right_hash}")
    print(
        f"REPRODUCIBILITY_OK arrays=bit_exact files=byte_exact sha256={left_hash}"
    )


def compare_reproducibility(left: Path, right: Path, report_path: Path) -> None:
    """Gate same-seed DEME repeats without hiding the raw GPU nondeterminism.

    DEME 2.4.0 does not expose a deterministic contact-pair reduction order.
    Independent runs diverge chaotically even with sorted contact types, CUB
    reduction, contact detection every step, and both adaptive controls off.
    The accepted scientific comparison is therefore: exact seeded inputs and
    identical configuration, plus agreement of the downstream 5 mm heightmap
    within one radius RMS / one diameter p95 / two diameters maximum.
    """
    from roarm_rl.heightmap import GridSpec, heightmap_from_particles

    left_arrays, left_meta = load_artifact(left)
    right_arrays, right_meta = load_artifact(right)
    n = int(left_meta["config"]["n_particles"])
    if int(right_meta["config"]["n_particles"]) != n:
        raise ValueError("repeat particle counts differ")
    exact_checks = {
        "config": left_meta["config"] == right_meta["config"],
        "initial_positions_m": np.array_equal(
            left_arrays["initial_positions_m"], right_arrays["initial_positions_m"]
        ),
        "radii_m": np.array_equal(left_arrays["radii_m"], right_arrays["radii_m"]),
        "particle_ids": np.array_equal(
            left_arrays["particle_ids"], right_arrays["particle_ids"]
        ),
        "box_bounds_m": np.array_equal(
            left_arrays["box_bounds_m"], right_arrays["box_bounds_m"]
        ),
    }
    raw_final_bit_exact = np.array_equal(
        left_arrays["positions_m"], right_arrays["positions_m"]
    ) and np.array_equal(left_arrays["velocities_m_s"], right_arrays["velocities_m_s"])

    bounds = left_arrays["box_bounds_m"]
    cell_m = 0.005
    rows = int(math.ceil((float(bounds[1, 1]) - float(bounds[1, 0])) / cell_m))
    cols = int(math.ceil((float(bounds[0, 1]) - float(bounds[0, 0])) / cell_m))
    spec = GridSpec(
        origin_xy_m=(float(bounds[0, 0]), float(bounds[1, 0])),
        cell_m=cell_m,
        shape=(rows, cols),
        frame="deme_box_floor_center",
        z_datum_m=0.0,
    )
    height_left = heightmap_from_particles(
        left_arrays["positions_m"], left_arrays["radii_m"], spec
    ).height
    height_right = heightmap_from_particles(
        right_arrays["positions_m"], right_arrays["radii_m"], spec
    ).height
    abs_diff = np.abs(height_left.astype(np.float64) - height_right.astype(np.float64))
    radius_m = float(left_arrays["radii_m"][0])
    diameter_m = 2.0 * radius_m
    metrics = {
        "rms_m": float(np.sqrt(np.mean(abs_diff * abs_diff))),
        "mean_abs_m": float(np.mean(abs_diff)),
        "p95_abs_m": float(np.quantile(abs_diff, 0.95)),
        "max_abs_m": float(abs_diff.max()),
        "exact_cell_fraction": float(np.mean(abs_diff == 0.0)),
    }
    thresholds = {
        "rms_m": radius_m,
        "p95_abs_m": diameter_m,
        "max_abs_m": 2.0 * diameter_m,
    }
    geometry_checks = {
        "rms": metrics["rms_m"] <= thresholds["rms_m"],
        "p95": metrics["p95_abs_m"] <= thresholds["p95_abs_m"],
        "max": metrics["max_abs_m"] <= thresholds["max_abs_m"],
    }
    report = {
        "artifact": "DEME_PELLET_PILE_REPRODUCIBILITY_V1",
        "left": str(left),
        "right": str(right),
        "left_sha256": _sha256(left),
        "right_sha256": _sha256(right),
        "seed": left_meta["config"]["seed"],
        "n_particles": n,
        "engine": "DEME 2.4.0 GPU",
        "raw_final_bit_exact": raw_final_bit_exact,
        "raw_final_bit_exact_verdict": "FAIL_ENGINE_LIMITATION",
        "attempted_determinism_controls": [
            "SetSortContactPairs(true)",
            "UseCubForceCollection(true)",
            "DisableAdaptiveUpdateFreq()",
            "DisableAdaptiveBinSize()",
            "diagnostic SetCDUpdateFreq(1) (also non-bit-exact)",
        ],
        "exact_seeded_input_checks": exact_checks,
        "heightmap_contract": {
            "producer": "roarm_rl.heightmap.heightmap_from_particles",
            "cell_m": cell_m,
            "shape": [rows, cols],
            "operator": "exact highest sphere surface over each cell footprint",
            "metrics": metrics,
            "thresholds": thresholds,
            "checks": geometry_checks,
        },
        "pass": bool(all(exact_checks.values()) and all(geometry_checks.values())),
        "interpretation": (
            "seeded input/config are bit-exact; raw final GPU floats are explicitly not bit-exact; "
            "the downstream 5 mm pile geometry agrees within pellet-scale tolerances"
        ),
    }
    _write_json(report_path, report)
    if not report["pass"]:
        raise ValueError(f"same-seed geometry reproducibility gate failed; see {report_path}")
    print(
        "REPRODUCIBILITY_OK initial=bit_exact final=heightmap_tolerance "
        f"raw_final_bit_exact={str(raw_final_bit_exact).lower()} "
        f"rms_mm={metrics['rms_m']*1e3:.6f} p95_mm={metrics['p95_abs_m']*1e3:.6f} "
        f"report={report_path}"
    )


def _open_box_mesh(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    zmin, zmax = bounds[2]
    vertices = np.asarray(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, triangles


def _target_mesh(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    geometry = metadata["derived_geometry"]
    shape = metadata["config"]["target_shape"]
    width = float(geometry["target_width_m"])
    length = float(geometry["target_length_m"])
    height = float(geometry["target_height_m"])
    if shape == "ridge":
        vertices = np.asarray(
            [
                [-width / 2, -length / 2, 0],
                [width / 2, -length / 2, 0],
                [0, -length / 2, height],
                [-width / 2, length / 2, 0],
                [width / 2, length / 2, 0],
                [0, length / 2, height],
            ],
            dtype=np.float64,
        )
        triangles = np.asarray(
            [[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [0, 2, 5], [0, 5, 3], [1, 4, 5], [1, 5, 2]],
            dtype=np.int64,
        )
        return vertices, triangles
    if shape == "box":
        bounds = np.asarray(
            [[-width / 2, width / 2], [-length / 2, length / 2], [0, height]],
            dtype=np.float64,
        )
        return _open_box_mesh(bounds)
    segments = 48
    radius = width / 2.0
    ring = np.column_stack(
        [
            radius * np.cos(np.linspace(0, 2 * math.pi, segments, endpoint=False)),
            radius * np.sin(np.linspace(0, 2 * math.pi, segments, endpoint=False)),
            np.zeros(segments),
        ]
    )
    vertices = np.vstack([ring, [[0, 0, height]], [[0, 0, 0]]])
    apex = segments
    floor_center = segments + 1
    triangles: list[list[int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.append([index, nxt, apex])
        triangles.append([floor_center, nxt, index])
    return vertices.astype(np.float64), np.asarray(triangles, dtype=np.int64)


def _rerun_expected_contract() -> tuple[set[str], dict[str, list[str]], list[str]]:
    exact_entities = {
        "/metadata/run",
        "/coordinate_frames/world_m",
        "/geometry/container",
        "/metadata/meshes/geometry__container",
        "/geometry/target_envelope",
        "/metadata/meshes/geometry__target_envelope",
        "/geometry/pile/particles",
        "/metrics/max_speed_m_s",
        "/metrics/p99_speed_m_s",
        "/metrics/rms_speed_m_s",
        "/metrics/kinetic_energy_j",
        "/metrics/num_contacts",
        "/events/settlement",
    }
    component_contract = {
        "/metadata/run": ["TextDocument:text"],
        "/geometry/container": [
            "Mesh3D:albedo_factor", "Mesh3D:triangle_indices", "Mesh3D:vertex_positions"
        ],
        "/geometry/target_envelope": [
            "Mesh3D:albedo_factor", "Mesh3D:triangle_indices", "Mesh3D:vertex_positions"
        ],
        "/geometry/pile/particles": [
            "Points3D:colors", "Points3D:positions", "Points3D:radii"
        ],
        "/metrics/max_speed_m_s": ["Scalars:scalars"],
        "/metrics/p99_speed_m_s": ["Scalars:scalars"],
        "/metrics/rms_speed_m_s": ["Scalars:scalars"],
        "/metrics/kinetic_energy_j": ["Scalars:scalars"],
        "/metrics/num_contacts": ["Scalars:scalars"],
        "/events/settlement": ["TextLog:level", "TextLog:text"],
    }
    # Rerun itself registers blueprint and log_time alongside user timelines.
    exact_timelines = ["blueprint", "log_time", "sample", "settle_phase", "sim_time_s"]
    return exact_entities, component_contract, exact_timelines


def _build_deme_pile_blueprint(mode: str) -> Any:
    """Build a pile-specific view where translucent envelopes cannot hide particles."""
    if mode != "deme_pile":
        raise ValueError(f"unsupported local Rerun blueprint mode: {mode!r}")
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/geometry/pile/particles"],
                    name="actual settled particles",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/geometry/container", "/geometry/target_envelope"],
                    name="container + target envelope",
                ),
                column_shares=[0.60, 0.40],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin="/metrics", contents="/metrics/**", name="Float64 metrics"
                ),
                rrb.TextLogView(
                    origin="/events", contents="/events/**", name="events and gates"
                ),
                column_shares=[0.62, 0.38],
            ),
            row_shares=[0.72, 0.28],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def export_rerun(npz_path: Path, artifact_tag: str | None = None) -> None:
    arrays, metadata = load_artifact(npz_path)
    try:
        import rerun as rr
        from roarm_rl.rerun_contract import RERUN_CONTRACT_VERSION, validate_rerun_artifact
        import roarm_rl.viz_debug as viz_debug
    except ImportError as exc:
        raise RuntimeError("Rerun export requires the isaaclab conda Python") from exc
    if str(rr.__version__) != "0.34.1" or RERUN_CONTRACT_VERSION != "0.34.1":
        raise RuntimeError(
            f"Rerun pin mismatch: sdk={rr.__version__}, contract={RERUN_CONTRACT_VERSION}"
        )
    # Make the version-matched CLI discoverable by rerun_contract without
    # altering any conda environment or global PATH.
    interpreter_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = interpreter_bin + os.pathsep + os.environ.get("PATH", "")

    stem = npz_path.with_suffix("")
    if artifact_tag:
        if not artifact_tag.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Rerun artifact tag must contain only letters, digits, '_' or '-'")
        stem = stem.with_name(f"{stem.name}_{artifact_tag}")
    rrd_path = stem.with_suffix(".rrd")
    rbl_path = stem.with_suffix(".rbl")
    screenshot_path = stem.with_name(f"{stem.name}_inspection.png")
    validation_path = stem.with_name(f"{stem.name}_rerun_validation.json")
    for path in (rrd_path, rbl_path, screenshot_path, validation_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    box_vertices, box_triangles = _open_box_mesh(arrays["box_bounds_m"])
    target_vertices, target_triangles = _target_mesh(metadata)
    radius = float(arrays["radii_m"][0])
    n = arrays["positions_m"].shape[0]
    initial_colors = np.tile(np.asarray([[125, 135, 145, 65]], dtype=np.uint8), (n, 1))
    final_colors = np.tile(np.asarray([[235, 205, 115, 220]], dtype=np.uint8), (n, 1))
    settled_sim_time_s = float(metadata["settling_gate"]["settled_sim_time_s"])
    point_rows = [
        {
            "entity_path": "geometry/pile/particles",
            "positions_m": arrays["initial_positions_m"],
            "radii": arrays["radii_m"].astype(np.float32),
            "colors": initial_colors,
            "coordinate_frame": "world_m",
            "sequence": {"settle_phase": 0},
            "duration": {"sim_time_s": 0.0},
        },
        {
            "entity_path": "geometry/pile/particles",
            "positions_m": arrays["positions_m"],
            "radii": arrays["radii_m"].astype(np.float32),
            "colors": final_colors,
            "coordinate_frame": "world_m",
            "sequence": {"settle_phase": 1},
            "duration": {"sim_time_s": settled_sim_time_s},
        },
    ]
    scalar_rows: list[dict[str, Any]] = []
    metric_names = (
        "max_speed_m_s",
        "p99_speed_m_s",
        "rms_speed_m_s",
        "kinetic_energy_j",
        "num_contacts",
    )
    for sample, row in enumerate(arrays["settle_history"]):
        for column, metric_name in enumerate(metric_names, start=1):
            scalar_rows.append(
                {
                    "entity_path": f"metrics/{metric_name}",
                    "value": float(row[column]),
                    "sequence": {"sample": sample},
                    "duration": {"sim_time_s": float(row[0])},
                }
            )
    events = [
        {
            "entity_path": "events/settlement",
            "text": (
                f"SETTLEMENT_OK N={n} sim_time_s="
                f"{metadata['settling_gate']['settled_sim_time_s']:.6f}"
            ),
            "level": "INFO",
            "sequence": {"settle_phase": 1},
            "duration": {"sim_time_s": settled_sim_time_s},
        }
    ]
    meshes = [
        {
            "entity_path": "geometry/container",
            "vertices_m": box_vertices,
            "triangles": box_triangles,
            "color_rgba": [80, 145, 210, 38],
            "coordinate_frame": "world_m",
            "static": True,
        },
        {
            "entity_path": "geometry/target_envelope",
            "vertices_m": target_vertices,
            "triangles": target_triangles,
            "color_rgba": [80, 225, 165, 30],
            "coordinate_frame": "world_m",
            "static": True,
        },
    ]
    original_blueprint_builder = viz_debug.build_rerun_blueprint
    try:
        # Keep this task's display customization local to sim_deme_pile.py;
        # roarm_rl.viz_debug remains the shared recording/validation backend.
        viz_debug.build_rerun_blueprint = _build_deme_pile_blueprint
        log_status = viz_debug.log_rerun(
            rrd_path,
            coordinate_frames=[
                {
                    "frame": "world_m",
                    "parent_frame": "tf#/",
                    "entity_path": "coordinate_frames/world_m",
                }
            ],
            meshes=meshes,
            points=point_rows,
            scalar_trace=scalar_rows,
            events=events,
            recording_metadata={
                "artifact": "DEME_PELLET_PILE_RERUN_V1",
                "source_npz": str(npz_path),
                "source_npz_sha256": _sha256(npz_path),
                "n_particles": n,
                "coordinate_frame": metadata["coordinate_frame"],
                "scientific_authority": "source NPZ arrays",
            },
            recording_id=f"deme_pile_n{n}_seed{metadata['config']['seed']}",
            blueprint_path=rbl_path,
            blueprint_mode="deme_pile",
            live_viewer=False,
            app_id="roarm_deme_pile",
        )
    finally:
        viz_debug.build_rerun_blueprint = original_blueprint_builder
    if not log_status.get("ok", False):
        raise RuntimeError(f"Rerun logging contract failed: {log_status}")

    exact_entities, component_contract, exact_timelines = _rerun_expected_contract()
    validation = validate_rerun_artifact(
        rrd_path,
        expected_entity_paths=sorted(exact_entities),
        expected_timeline_names=exact_timelines,
        exact_entity_paths=sorted(exact_entities),
        exact_timeline_names=exact_timelines,
        expected_entity_components=component_contract,
        blueprint_path=rbl_path,
        screenshot_path=screenshot_path,
        cli_path=Path(interpreter_bin) / "rerun",
        expected_version="0.34.1",
        timeout_s=180.0,
    )
    validation["source_npz"] = str(npz_path)
    validation["source_npz_sha256"] = _sha256(npz_path)
    validation["log_status_summary"] = {
        key: log_status.get(key)
        for key in (
            "ok", "bytes", "rerun_sdk_version", "sink_attached_before_logging",
            "sink_finalized", "flush_ok", "blueprint_status"
        )
    }
    _write_json(validation_path, validation)
    if not validation.get("pass", False):
        raise RuntimeError(f"Rerun exact contract failed; see {validation_path}")
    print(
        f"RERUN_EXPORT_OK rrd={rrd_path} rbl={rbl_path} screenshot={screenshot_path} "
        f"validation={validation_path}"
    )


def revalidate_existing_rerun(npz_path: Path, tag: str) -> None:
    """Re-run exact contracts and headless rendering without rewriting RRD/RBL."""
    from roarm_rl.rerun_contract import validate_rerun_artifact

    interpreter_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = interpreter_bin + os.pathsep + os.environ.get("PATH", "")
    stem = npz_path.with_suffix("")
    rrd_path = stem.with_suffix(".rrd")
    rbl_path = stem.with_suffix(".rbl")
    screenshot_path = stem.with_name(f"{stem.name}_{tag}_inspection.png")
    validation_path = stem.with_name(f"{stem.name}_{tag}_rerun_validation.json")
    for required in (npz_path, rrd_path, rbl_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    for output in (screenshot_path, validation_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    exact_entities, component_contract, exact_timelines = _rerun_expected_contract()
    validation = validate_rerun_artifact(
        rrd_path,
        expected_entity_paths=sorted(exact_entities),
        expected_timeline_names=exact_timelines,
        exact_entity_paths=sorted(exact_entities),
        exact_timeline_names=exact_timelines,
        expected_entity_components=component_contract,
        blueprint_path=rbl_path,
        screenshot_path=screenshot_path,
        cli_path=Path(interpreter_bin) / "rerun",
        expected_version="0.34.1",
        timeout_s=180.0,
    )
    validation["source_npz"] = str(npz_path)
    validation["source_npz_sha256"] = _sha256(npz_path)
    validation["revalidation_tag"] = tag
    validation["reuses_finalized_rrd_rbl"] = True
    _write_json(validation_path, validation)
    if not validation.get("pass", False):
        raise RuntimeError(f"Rerun revalidation failed; see {validation_path}")
    print(
        f"RERUN_REVALIDATION_OK validation={validation_path} screenshot={screenshot_path}"
    )


def validate_rerun_contract(validation_path: Path, inspection_path: Path) -> None:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    required_validation = {
        "pass": validation.get("pass") is True,
        "version": validation.get("version", {}).get("expected_version_match") is True,
        "footer": validation.get("footer_manifest_present") is True,
        "entity": validation.get("entity_path_contract", {}).get("pass") is True,
        "timeline": validation.get("timeline_contract", {}).get("pass") is True,
        "components": validation.get("component_contract", {}).get("pass") is True,
        "blueprint": validation.get("blueprint_verify", {}).get("ok") is True,
        "screenshot": validation.get("headless_render", {}).get("ok") is True,
    }
    failed = [name for name, passed in required_validation.items() if not passed]
    if failed:
        raise ValueError(f"Rerun validation failures: {failed}")
    if inspection.get("visual_inspection_complete") is not True:
        raise ValueError("manual visual inspection is not complete")
    if inspection.get("completion_contract_pass") is not True:
        raise ValueError("inspection completion contract is false")
    screenshot_path = Path(validation["headless_render"]["path"])
    if inspection.get("screenshot_path") != str(screenshot_path):
        raise ValueError("inspection screenshot path does not match validation")
    if inspection.get("screenshot_sha256") != _sha256(screenshot_path):
        raise ValueError("inspection screenshot hash mismatch")
    observations = inspection.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("inspection observations are empty")
    print("RERUN_OBSERVABILITY_OK visual_inspection=complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    utility = parser.add_mutually_exclusive_group()
    utility.add_argument("--describe-format", action="store_true")
    utility.add_argument("--validate-output", nargs="+", type=Path)
    utility.add_argument("--compare-output", nargs=2, type=Path, metavar=("LEFT", "RIGHT"))
    utility.add_argument(
        "--compare-reproducibility",
        nargs=3,
        type=Path,
        metavar=("LEFT", "RIGHT", "REPORT_JSON"),
    )
    utility.add_argument("--export-rerun", type=Path, metavar="NPZ")
    utility.add_argument("--revalidate-rerun", type=Path, metavar="NPZ")
    utility.add_argument(
        "--validate-rerun-contract",
        nargs=2,
        type=Path,
        metavar=("VALIDATION_JSON", "INSPECTION_JSON"),
    )
    parser.add_argument("--expected-n", type=int)
    parser.add_argument("--require-timing", action="store_true")
    parser.add_argument("--rerun-validation-tag", default="v2")
    parser.add_argument("--rerun-artifact-tag")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config-name", default="practical")
    parser.add_argument("--n-particles", type=int, default=18_796)
    parser.add_argument("--diameter-mm", type=float, default=4.16)
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--target-shape", choices=("ridge", "cone", "box"), default="ridge")
    parser.add_argument("--target-volume-cm3", type=float)
    parser.add_argument("--target-repose-angle-deg", type=float, default=28.0)
    parser.add_argument("--ridge-length-width-ratio", type=float, default=240.0 / 110.0)
    parser.add_argument("--box-width-mm", type=float)
    parser.add_argument("--box-length-mm", type=float)
    parser.add_argument("--bulk-density-g-cm3", type=float, default=0.55)
    parser.add_argument("--particle-density-kg-m3", type=float, default=950.0)
    parser.add_argument("--particle-mu", type=float, default=0.50)
    parser.add_argument("--wall-mu", type=float, default=0.50)
    parser.add_argument("--restitution", type=float, default=0.30)
    parser.add_argument("--rolling-friction", type=float, default=0.05)
    parser.add_argument("--young-modulus-pa", type=float, default=1.0e7)
    parser.add_argument("--poisson-ratio", type=float, default=0.30)
    parser.add_argument("--material-status", choices=("provisional", "measured"), default="provisional")
    parser.add_argument("--dt-s", type=float, default=2.0e-5)
    parser.add_argument("--cd-update-freq", type=int, default=20)
    parser.add_argument("--spacing-factor", type=float, default=1.10)
    parser.add_argument("--jitter-fraction-diameter", type=float, default=0.015)
    parser.add_argument(
        "--initialization-mode",
        choices=("target_fcc", "rain_box"),
        default="target_fcc",
    )
    parser.add_argument("--release-footprint-fraction", type=float, default=0.22)
    parser.add_argument("--ridge-release-length-fraction", type=float, default=0.85)
    parser.add_argument("--drop-height-mm", type=float)
    parser.add_argument("--check-interval-s", type=float, default=0.05)
    parser.add_argument("--min-sim-time-s", type=float, default=0.40)
    parser.add_argument("--max-sim-time-s", type=float, default=2.00)
    parser.add_argument("--speed-max-mm-s", type=float, default=100.0)
    parser.add_argument("--speed-p99-mm-s", type=float, default=2.0)
    parser.add_argument("--speed-rms-mm-s", type=float, default=1.0)
    parser.add_argument("--stable-checks", type=int, default=5)
    parser.add_argument("--max-penetration-fraction-diameter", type=float, default=0.05)
    return parser


def config_from_args(args: argparse.Namespace) -> PileConfig:
    return PileConfig(
        n_particles=args.n_particles,
        diameter_mm=args.diameter_mm,
        seed=args.seed,
        config_name=args.config_name,
        target_shape=args.target_shape,
        target_volume_cm3=args.target_volume_cm3,
        target_repose_angle_deg=args.target_repose_angle_deg,
        ridge_length_width_ratio=args.ridge_length_width_ratio,
        box_width_mm=args.box_width_mm,
        box_length_mm=args.box_length_mm,
        bulk_density_g_cm3=args.bulk_density_g_cm3,
        particle_density_kg_m3=args.particle_density_kg_m3,
        particle_mu=args.particle_mu,
        wall_mu=args.wall_mu,
        restitution=args.restitution,
        rolling_friction=args.rolling_friction,
        young_modulus_pa=args.young_modulus_pa,
        poisson_ratio=args.poisson_ratio,
        material_status=args.material_status,
        dt_s=args.dt_s,
        cd_update_freq=args.cd_update_freq,
        spacing_factor=args.spacing_factor,
        jitter_fraction_diameter=args.jitter_fraction_diameter,
        initialization_mode=args.initialization_mode,
        release_footprint_fraction=args.release_footprint_fraction,
        ridge_release_length_fraction=args.ridge_release_length_fraction,
        drop_height_mm=args.drop_height_mm,
        check_interval_s=args.check_interval_s,
        min_sim_time_s=args.min_sim_time_s,
        max_sim_time_s=args.max_sim_time_s,
        speed_max_mm_s=args.speed_max_mm_s,
        speed_p99_mm_s=args.speed_p99_mm_s,
        speed_rms_mm_s=args.speed_rms_mm_s,
        stable_checks=args.stable_checks,
        max_penetration_fraction_diameter=args.max_penetration_fraction_diameter,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.describe_format:
        print(__doc__)
        print("FORMAT_CONTRACT_OK")
        return
    if args.validate_output:
        validate_output(
            args.validate_output,
            expected_n=args.expected_n,
            require_timing=args.require_timing,
        )
        return
    if args.compare_output:
        compare_output(*args.compare_output)
        return
    if args.compare_reproducibility:
        compare_reproducibility(*args.compare_reproducibility)
        return
    if args.export_rerun:
        export_rerun(args.export_rerun, args.rerun_artifact_tag)
        return
    if args.revalidate_rerun:
        revalidate_existing_rerun(args.revalidate_rerun, args.rerun_validation_tag)
        return
    if args.validate_rerun_contract:
        validate_rerun_contract(*args.validate_rerun_contract)
        return
    config = config_from_args(args)
    output = args.output or default_output_path(config)
    run_simulation(config, output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
