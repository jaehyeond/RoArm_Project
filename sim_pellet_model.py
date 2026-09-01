#!/usr/bin/env python3
"""Pellet shape model + angle-of-repose calibration harness for DEME 2.4.0.

Why this exists
---------------
``sim_deme_pile.py`` models every pellet as one perfect sphere
(``LoadSphereType`` at ``sim_deme_pile.py:639``).  Real polypropylene pellets
are extrusion-cut cylinders, and a cylinder heap does not have the same angle
of repose as a sphere heap: spheres roll, cylinders interlock.  Because this
project predicts "the shape that remains after a scoop", a wrong heap shape
invalidates the prediction target itself.

This module therefore does three things:

1. Builds multi-sphere **clump** templates (1, 2 or 3 spheres) whose mass and
   inertia tensor are computed **analytically from the actual union-of-spheres
   solid**, not estimated.
2. Runs a standard **poured-heap angle-of-repose** simulation and measures the
   angle with an explicitly stated, reproducible definition.
3. Sweeps ``(shape, mu, Crr)`` and exposes :func:`fit_to_measured_repose`, the
   inverse lookup: give it a measured angle, it returns the parameter
   combinations that produce it.

MEASUREMENT STATUS
------------------
No pellet has been procured or measured as of this file's creation.  Every
physical default below is a placeholder and is tagged ``MEASURE`` in the
emitted JSON ``non_claims`` block:

* ``pellet_dia_mm`` / ``pellet_len_mm`` - nominal pellet cylinder, unmeasured.
* ``particle_density_kg_m3`` - generic PP solid density, unmeasured.
* ``mu`` / ``Crr`` / ``CoR`` - unmeasured; these are exactly the axes the sweep
  exists to calibrate.
* ``young_modulus_pa`` - a NUMERICAL STABILITY value (D464 sec.4: 1e8..1e9
  diverges at this time step, 5e6 is stable).  It is not a measured stiffness.
* ``packing_fraction_for_sizing`` and ``sizing_repose_angle_deg`` - used only to
  pick a domain size big enough that walls do not touch the heap.  They are not
  predictions and are never reported as results.

Nothing in this file may be cited as a measured pellet property.

Coordinate frame
----------------
Right-handed, ``+z`` up, origin at the centre of the container floor, floor
exactly ``z=0``.  SI units throughout (m, m/s, kg, s, rad unless a name ends in
``_deg`` or ``_mm``).

Two-interpreter split (same as ``sim_deme_pile.py``)
---------------------------------------------------
``DEME 2.4.0`` lives only in the ``roarm`` env; ``rerun 0.34.1`` (D326 pin)
lives only in the ``isaaclab`` env.  Simulate with the ``roarm`` interpreter,
export Rerun with the ``isaaclab`` interpreter.
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
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

OUT_DIR = Path("claudedocs/runtime_logs/pellet_model")

ARTIFACT = "PELLET_REPOSE_V1"
SCHEMA_VERSION = 1

# Anything in this list is a placeholder awaiting the physical pellets.
NON_CLAIMS = [
    "MEASURE: pellet_dia_mm / pellet_len_mm are nominal placeholders; no pellet has been measured",
    "MEASURE: particle_density_kg_m3 is a generic PP solid-density placeholder, not weighed",
    "MEASURE: mu / wall_mu / rolling_friction (Crr) / restitution are unmeasured sweep axes",
    "NUMERICAL: young_modulus_pa=5e6 is a time-step stability value from D464 sec.4, not a measured stiffness",
    "SIZING_ONLY: packing_fraction_for_sizing and sizing_repose_angle_deg only size the domain; they are not results",
    "NOT_A_RESULT: every repose angle produced here is the angle of the SIMULATED material at the stated parameters, "
    "not a prediction of the real pellet angle until fit_to_measured_repose is given a measured value",
]

HISTORY_COLUMNS = (
    "sim_time_s",
    "max_speed_m_s",
    "p99_speed_m_s",
    "rms_speed_m_s",
    "kinetic_energy_j",
    "num_contacts",
    "heap_top_z_m",
    "max_radial_extent_m",
)

NPZ_KEYS = (
    "box_bounds_m",
    "clump_positions_m",
    "clump_quaternions_xyzw",
    "contact_forces_n",
    "contact_points_m",
    "initial_positions_m",
    "initial_quaternions_xyzw",
    "metadata_json",
    "particle_ids",
    "radial_profile",
    "settle_history",
    "sphere_offsets_m",
    "sphere_radii_m",
    "state_history",
    "velocities_m_s",
)

RADIAL_PROFILE_COLUMNS = ("radius_m", "mean_height_m", "max_height_m", "cell_count")

# The single canonical statement of what "the repose angle" means here.  Printed
# by --describe-format, stored in every artifact, and quoted back by
# fit_to_measured_repose so a physical measurement can be made the same way.
REPOSE_PRIMARY_DEFINITION = "sidewall_regression"
REPOSE_PRIMARY_DEFINITION_TEXT = (
    "area-weighted least-squares line fitted to the annulus-mean height profile h(r) of the "
    "settled heap, over the band where h falls from 80% to 20% of the apex height; "
    "angle = atan(-slope). The apex plateau and the toe are excluded because that is where "
    "the classical definitions disagree most."
)


# ---------------------------------------------------------------------------
# 1. Exact geometry of a union of equal, axially-aligned, overlapping spheres
# ---------------------------------------------------------------------------
#
# All moments below are GEOMETRIC (unit density).  Multiply by density to get
# mass properties.  Every formula is a closed-form polynomial integral of the
# solid of revolution; nothing is sampled or approximated.
#
# Cap convention: a sphere of radius R centred at the origin, the cap being the
# region u >= a with -R <= a <= R.  The cap's symmetry axis is u.


def _cap_volume(radius: float, a: float) -> float:
    """Volume of the spherical cap ``u >= a``.  Exact."""
    r = float(radius)
    return math.pi * (2.0 * r**3 / 3.0 - r * r * a + a**3 / 3.0)


def _cap_first_moment(radius: float, a: float) -> float:
    """``integral u dV`` over the cap, about the sphere centre.  Exact."""
    r = float(radius)
    return math.pi * (r**4 / 4.0 - r * r * a * a / 2.0 + a**4 / 4.0)


def _cap_second_moment(radius: float, a: float) -> float:
    """``integral u^2 dV`` over the cap, about the sphere centre.  Exact."""
    r = float(radius)
    return math.pi * (2.0 * r**5 / 15.0 - r * r * a**3 / 3.0 + a**5 / 5.0)


def _cap_axial_moment(radius: float, a: float) -> float:
    """Cap moment of inertia about its own symmetry axis ``u``.  Exact."""
    r = float(radius)
    upper = r**5 - 2.0 * r**5 / 3.0 + r**5 / 5.0
    lower = r**4 * a - 2.0 * r * r * a**3 / 3.0 + a**5 / 5.0
    return 0.5 * math.pi * (upper - lower)


def _cap_transverse_moment_about_sphere_centre(radius: float, a: float) -> float:
    """Cap moment about a transverse axis through the SPHERE centre.  Exact.

    A disk slice of radius ``rho`` at station ``u`` contributes
    ``rho^2/4`` (its own diameter) plus ``u^2`` (parallel axis), so the
    transverse moment is ``axial/2 + integral u^2 dV``.
    """
    return 0.5 * _cap_axial_moment(radius, a) + _cap_second_moment(radius, a)


def _lens_moments(radius: float, centre_distance: float) -> dict[str, float]:
    """Geometric moments of the lens shared by two equal overlapping spheres.

    The lens is the intersection of two spheres of radius ``radius`` whose
    centres are ``centre_distance`` apart.  Results are reported about the lens
    centre (the midpoint of the two sphere centres).  Exact: the lens is two
    mirrored caps cut at the mid-plane, i.e. ``a = centre_distance / 2``.
    """
    r = float(radius)
    s = float(centre_distance)
    if not (0.0 < s < 2.0 * r):
        raise ValueError(f"lens requires 0 < centre_distance < 2R, got s={s}, R={r}")
    a = 0.5 * s
    volume_half = _cap_volume(r, a)
    axial_half = _cap_axial_moment(r, a)
    trans_half_about_sphere_centre = _cap_transverse_moment_about_sphere_centre(r, a)
    centroid_u = _cap_first_moment(r, a) / volume_half
    # Shift the half-cap's transverse moment from the sphere centre (u=0) to
    # the lens centre (u=a) through the centroid.
    trans_half_about_centroid = (
        trans_half_about_sphere_centre - volume_half * centroid_u * centroid_u
    )
    trans_half_about_lens = (
        trans_half_about_centroid + volume_half * (a - centroid_u) ** 2
    )
    return {
        "volume": 2.0 * volume_half,
        "axial_moment": 2.0 * axial_half,
        "transverse_moment_about_lens_centre": 2.0 * trans_half_about_lens,
    }


def union_geometric_moments(radius: float, spacing: float, n_spheres: int) -> dict[str, float]:
    """Exact volume and inertia of ``n_spheres`` equal spheres strung on an axis.

    Sphere ``i`` sits at ``x_i = (i - (n-1)/2) * spacing`` with radius
    ``radius``.  Results are about the union's centroid, which is the origin by
    symmetry; ``xx`` is the chain axis, ``yy == zz`` are transverse.

    Validity: only ADJACENT spheres may overlap, otherwise the two-term
    inclusion-exclusion below would double-subtract a triple region.  For
    ``n >= 3`` that requires ``spacing >= radius``; for every ``n`` it requires
    ``spacing < 2*radius`` so the body stays connected.
    """
    r = float(radius)
    n = int(n_spheres)
    if n < 1:
        raise ValueError("n_spheres must be >= 1")
    sphere_volume = 4.0 / 3.0 * math.pi * r**3
    sphere_moment = 8.0 / 15.0 * math.pi * r**5  # (2/5) * V * R^2, any axis
    if n == 1:
        return {
            "volume": sphere_volume,
            "ixx": sphere_moment,
            "iyy": sphere_moment,
            "izz": sphere_moment,
            "length": 2.0 * r,
            "width": 2.0 * r,
            "lens_volume": 0.0,
            "n_lenses": 0,
        }
    s = float(spacing)
    if not (0.0 < s < 2.0 * r):
        raise ValueError(
            f"spacing must satisfy 0 < s < 2R for a connected union, got s={s}, R={r}"
        )
    if n >= 3 and s < r - 1.0e-15:
        raise ValueError(
            f"spacing {s} < radius {r}: non-adjacent spheres would overlap and the "
            "exact two-term inclusion-exclusion would be wrong"
        )
    centres = np.asarray([(i - 0.5 * (n - 1)) * s for i in range(n)], dtype=np.float64)
    lens = _lens_moments(r, s)
    lens_centres = 0.5 * (centres[:-1] + centres[1:])

    volume = n * sphere_volume - (n - 1) * lens["volume"]
    ixx = n * sphere_moment - (n - 1) * lens["axial_moment"]
    iyy = float(
        np.sum(sphere_moment + sphere_volume * centres**2)
        - np.sum(
            lens["transverse_moment_about_lens_centre"] + lens["volume"] * lens_centres**2
        )
    )
    return {
        "volume": float(volume),
        "ixx": float(ixx),
        "iyy": iyy,
        "izz": iyy,
        "length": float((n - 1) * s + 2.0 * r),
        "width": float(2.0 * r),
        "lens_volume": float(lens["volume"]),
        "n_lenses": n - 1,
    }


# ---------------------------------------------------------------------------
# 2. Pellet + template specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PelletSpec:
    """The nominal pellet.  Every field here is a MEASURE placeholder."""

    # MEASURE: extrusion-cut PP pellets are typically 3-5 mm in both dimensions.
    # These two numbers are the first thing to overwrite when the pellets land.
    pellet_dia_mm: float = 3.5
    pellet_len_mm: float = 4.0
    # MEASURE: generic solid PP density.  Weigh a counted sample instead.
    particle_density_kg_m3: float = 950.0

    def cylinder_volume_m3(self) -> float:
        d = self.pellet_dia_mm * 1.0e-3
        length = self.pellet_len_mm * 1.0e-3
        return 0.25 * math.pi * d * d * length

    def mass_kg(self) -> float:
        return self.cylinder_volume_m3() * self.particle_density_kg_m3

    def nominal_aspect(self) -> float:
        return self.pellet_len_mm / self.pellet_dia_mm


# Shape axis of the sweep.  ``aspect`` is union length / union width.
# sphere is the control (the current sim_deme_pile.py model).
# The two clump aspects are PROVISIONAL brackets: once the pellet is measured,
# pass the measured L/D via --clump2-aspect / --clump3-aspect, or call
# template_for_measured_pellet().
DEFAULT_TEMPLATE_ASPECTS: dict[str, float] = {
    "sphere": 1.0,
    "clump2": 1.5,
    "clump3": 2.2,
}
TEMPLATE_SPHERE_COUNTS: dict[str, int] = {"sphere": 1, "clump2": 2, "clump3": 3}
SHAPE_ORDER = ("sphere", "clump2", "clump3")


def aspect_bounds(n_spheres: int) -> tuple[float, float]:
    """Legal aspect range for an ``n``-sphere chain under the exactness rule.

    ``spacing = 2R(aspect-1)/(n-1)``.  Connectivity needs ``spacing < 2R``;
    exact inclusion-exclusion needs ``spacing >= R`` when ``n >= 3``.
    """
    if n_spheres == 1:
        return (1.0, 1.0)
    if n_spheres == 2:
        return (1.0 + 1.0e-9, 2.0 - 1.0e-9)
    lo = 1.0 + 0.5 * (n_spheres - 1)
    hi = 1.0 + (n_spheres - 1) - 1.0e-9
    return (lo, hi)


@dataclass(frozen=True)
class ClumpTemplate:
    """A DEME clump template with mass properties derived from its true solid."""

    name: str
    n_spheres: int
    aspect: float
    size_match: str
    sphere_radius_m: float
    spacing_m: float
    offsets_m: tuple[tuple[float, float, float], ...]
    mass_kg: float
    moi_kg_m2: tuple[float, float, float]
    union_volume_m3: float
    cylinder_volume_m3: float
    union_length_m: float
    union_width_m: float
    bounding_diameter_m: float

    def volume_mismatch_fraction(self) -> float:
        return (self.union_volume_m3 - self.cylinder_volume_m3) / self.cylinder_volume_m3


def build_template(
    pellet: PelletSpec,
    name: str,
    *,
    aspect: float | None = None,
    size_match: str = "volume",
) -> ClumpTemplate:
    """Construct a clump template and compute its mass properties exactly.

    ``size_match``:

    ``volume``
        Solve the sphere radius so the union solid's volume equals the pellet
        cylinder's volume, at the requested aspect ratio.  The heap then has
        the right bulk volume, at the cost of a bounding box that differs from
        the nominal cylinder.  This is the default because every downstream
        quantity in this project (heap shape, scooped volume, heightmap) is
        geometric.
    ``diameter``
        Fix the sphere radius at ``pellet_dia_mm/2`` so the union's width
        matches the pellet diameter exactly.  The union then under-fills the
        cylinder and ``volume_mismatch_fraction`` reports by how much.

    The clump MASS is always the true pellet mass ``rho * V_cylinder`` so all
    shapes are mass-matched.  The MOI is the union solid's inertia scaled to
    that mass, so shape and inertia stay self-consistent with the body DEME
    actually collides.
    """
    if name not in TEMPLATE_SPHERE_COUNTS:
        raise ValueError(f"unknown template {name!r}; expected one of {SHAPE_ORDER}")
    if size_match not in {"volume", "diameter"}:
        raise ValueError("size_match must be 'volume' or 'diameter'")
    n = TEMPLATE_SPHERE_COUNTS[name]
    aspect_value = float(DEFAULT_TEMPLATE_ASPECTS[name] if aspect is None else aspect)
    lo, hi = aspect_bounds(n)
    if n == 1:
        if abs(aspect_value - 1.0) > 1.0e-12:
            raise ValueError("the single-sphere template has aspect 1.0 by construction")
    elif not (lo <= aspect_value <= hi):
        raise ValueError(
            f"{name}: aspect {aspect_value} outside the exact-geometry range "
            f"[{lo:.6f}, {hi:.6f}] for {n} spheres"
        )

    cylinder_volume = pellet.cylinder_volume_m3()
    # spacing = shape_ratio * radius, independent of radius.
    shape_ratio = 0.0 if n == 1 else 2.0 * (aspect_value - 1.0) / (n - 1)

    if size_match == "volume":
        unit = union_geometric_moments(1.0, shape_ratio, n)
        radius = (cylinder_volume / unit["volume"]) ** (1.0 / 3.0)
    else:
        radius = 0.5 * pellet.pellet_dia_mm * 1.0e-3
    spacing = shape_ratio * radius
    moments = union_geometric_moments(radius, spacing, n)

    mass = pellet.mass_kg()
    # Unit-density moments -> mass properties of a body of the given mass.
    scale = mass / moments["volume"]
    moi = (
        scale * moments["ixx"],
        scale * moments["iyy"],
        scale * moments["izz"],
    )
    offsets = tuple(
        ((i - 0.5 * (n - 1)) * spacing, 0.0, 0.0) for i in range(n)
    )
    return ClumpTemplate(
        name=name,
        n_spheres=n,
        aspect=aspect_value,
        size_match=size_match,
        sphere_radius_m=float(radius),
        spacing_m=float(spacing),
        offsets_m=offsets,
        mass_kg=float(mass),
        moi_kg_m2=(float(moi[0]), float(moi[1]), float(moi[2])),
        union_volume_m3=float(moments["volume"]),
        cylinder_volume_m3=float(cylinder_volume),
        union_length_m=float(moments["length"]),
        union_width_m=float(moments["width"]),
        bounding_diameter_m=float(max(moments["length"], moments["width"])),
    )


def template_for_measured_pellet(
    dia_mm: float, len_mm: float, density_kg_m3: float, *, size_match: str = "volume"
) -> ClumpTemplate:
    """Pick the clump family that can represent a MEASURED pellet exactly.

    Call this the moment the pellets are on the bench and calipered.  It picks
    the smallest sphere count whose legal aspect range contains the measured
    ``len/dia`` and returns the corresponding template.
    """
    pellet = PelletSpec(
        pellet_dia_mm=float(dia_mm),
        pellet_len_mm=float(len_mm),
        particle_density_kg_m3=float(density_kg_m3),
    )
    aspect = pellet.nominal_aspect()
    if aspect < 1.0:
        raise ValueError(
            f"measured aspect {aspect:.4f} < 1 (a disc/lens pellet). The axially "
            "strung clump family in this module only represents aspect >= 1; a "
            "disc needs a transverse clump layout, which is not implemented."
        )
    for name in SHAPE_ORDER:
        lo, hi = aspect_bounds(TEMPLATE_SPHERE_COUNTS[name])
        if lo <= aspect <= hi:
            return build_template(pellet, name, aspect=aspect, size_match=size_match)
    raise ValueError(
        f"measured aspect {aspect:.4f} needs more than 3 spheres; extend "
        "TEMPLATE_SPHERE_COUNTS before using it"
    )


# ---------------------------------------------------------------------------
# 3. Repose run configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReposeConfig:
    shape: str = "sphere"
    aspect: float | None = None
    size_match: str = "volume"
    n_particles: int = 1500
    seed: int = 460

    pellet_dia_mm: float = 3.5
    pellet_len_mm: float = 4.0
    particle_density_kg_m3: float = 950.0

    # Sweep axes.  All MEASURE placeholders.
    particle_mu: float = 0.50
    wall_mu: float = 0.50
    rolling_friction: float = 0.05
    restitution: float = 0.30
    # NUMERICAL, not measured: D464 sec.4 (1e8..1e9 diverges at this step).
    young_modulus_pa: float = 5.0e6
    poisson_ratio: float = 0.30

    dt_s: float = 4.0e-5
    # Contact-detection cadence.  DEME lets the dynamics thread drift ahead of
    # the contact-detection thread and inflates the contact margin to compensate,
    # so a LARGER value is not cheaper: at freq 20 the margin balloons to ~138
    # potential contacts per sphere and one clump2 cell took 54.2 s, versus 8.6 s
    # at freq 6.  Measured on clump2 mu=0.25 Crr=0.02 (the worst cell):
    #     freq 20 -> 54.2 s, 14.334 deg, r2=0.9803
    #     freq  6 ->  8.6 s, 15.629 deg, r2=0.9876
    #     freq  2 -> 10.8 s, 15.019 deg, r2=0.9842
    # 6 and 2 agree to 0.6 deg while 20 sits 1.3 deg low, i.e. 20 was also
    # under-resolved, not merely slow.  6 is the converged and cheapest setting.
    cd_update_freq: int = 6
    cd_max_update_freq: int = 6
    error_out_avg_contacts: float = 300.0
    init_bin_num_target: int = 200_000
    # Some agitated low-friction clump cells put DEME into a stall rather than a
    # crash: the process sits at ~2% CPU and ~10% GPU making no progress. It is
    # bounded here so one stalled cell is recorded as a failed cell in minutes
    # instead of consuming the whole sweep.
    cell_timeout_s: float = 300.0
    # D464 sec.4: DoDynamicsThenSync(0.02) core-dumps; 0.004 chunks are stable.
    advance_chunk_s: float = 0.004
    chunks_per_sample: int = 5
    min_sim_time_s: float = 0.30
    max_sim_time_s: float = 7.00
    speed_max_mm_s: float = 60.0
    speed_p99_mm_s: float = 3.0
    speed_rms_mm_s: float = 1.5
    stable_samples: int = 4

    # Release protocol ("virtual funnel"): the particles are seeded as a parked
    # stack of batches held fixed, then poured one batch at a time.  Just before
    # each batch is freed it is teleported to a release plane a constant
    # drop_gap_mm above the CURRENT heap top, so every batch falls the same
    # distance exactly as it would from a funnel held above a growing heap.
    #
    # Dropping the whole column at once instead produces a granular COLUMN
    # COLLAPSE, whose runout slope is not the poured angle of repose; a seeded
    # lattice column can even stay standing as a crystal.  Batch pouring is what
    # makes this a repose measurement.
    column_radius_fraction: float = 0.30
    seed_spacing_factor: float = 1.25
    seed_jitter_fraction: float = 0.35
    drop_gap_mm: float = 15.0
    n_release_batches: int = 18
    min_release_sites: int = 24
    # The pour is event-driven, not clock-driven: a batch is released only once
    # the previous one has actually landed.  A fixed clock either drops the next
    # batch into the airborne previous one, or fires so late that the run is
    # mostly idle waiting.
    release_interval_s: float = 0.06
    release_quiet_speed_m_s: float = 0.25
    release_timeout_s: float = 0.14
    randomize_initial_orientation: bool = True

    # SIZING_ONLY - never reported as a result.
    packing_fraction_for_sizing: float = 0.60
    # Deliberately at the LOW end of the plausible range: the domain must be
    # wide enough for the flattest cell in the sweep (low mu, low Crr, spheres),
    # otherwise that cell fails wall clearance and drops out of the comparison.
    sizing_repose_angle_deg: float = 20.0
    domain_radius_factor: float = 1.70

    # Measurement definition knobs (see measure_repose_angle).
    profile_cell_mm: float = 2.0
    fit_upper_height_fraction: float = 0.80
    fit_lower_height_fraction: float = 0.20
    toe_height_fraction_of_diameter: float = 0.50
    min_fit_r_squared: float = 0.90
    # Quadrant spread is REPORTED AS THE AZIMUTHAL UNCERTAINTY of the angle, not
    # used as a precision gate.  Measured on the clump3 mu=0.45 Crr=0.06 heap
    # (full-heap fit r_squared 0.9973, i.e. an excellent cone): quadrant angles
    # 35.1/32.2/40.8/36.1, spread 8.6 deg.  Re-centring the heap axis on the
    # height-weighted centroid only moved that to 7.7 deg, so the scatter is
    # genuine azimuthal structure of a 1500-rod heap, not a centring artifact.
    # Gating validity on it would reject physically sound measurements, so the
    # threshold below is only a LOPSIDEDNESS check: material genuinely piled to
    # one side reads far above this, ordinary granular scatter does not.
    max_quadrant_spread_deg: float = 15.0
    wall_clearance_fraction: float = 0.90
    min_heap_fraction: float = 0.90

    max_penetration_fraction_diameter: float = 0.08
    store_state_history: bool = True

    def pellet(self) -> PelletSpec:
        return PelletSpec(
            pellet_dia_mm=self.pellet_dia_mm,
            pellet_len_mm=self.pellet_len_mm,
            particle_density_kg_m3=self.particle_density_kg_m3,
        )


def _validate_repose_config(config: ReposeConfig) -> None:
    positive = {
        "n_particles": config.n_particles,
        "pellet_dia_mm": config.pellet_dia_mm,
        "pellet_len_mm": config.pellet_len_mm,
        "particle_density_kg_m3": config.particle_density_kg_m3,
        "young_modulus_pa": config.young_modulus_pa,
        "dt_s": config.dt_s,
        "advance_chunk_s": config.advance_chunk_s,
        "chunks_per_sample": config.chunks_per_sample,
        "max_sim_time_s": config.max_sim_time_s,
        "profile_cell_mm": config.profile_cell_mm,
        "stable_samples": config.stable_samples,
    }
    for key, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{key} must be finite and > 0, got {value}")
    if config.shape not in SHAPE_ORDER:
        raise ValueError(f"shape must be one of {SHAPE_ORDER}")
    if config.particle_mu < 0 or config.rolling_friction < 0:
        raise ValueError("mu and Crr must be >= 0")
    if not (0.0 <= config.poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in [0, 0.5)")
    if not (0.0 < config.column_radius_fraction < 1.0):
        raise ValueError("column_radius_fraction must be in (0, 1)")
    if config.seed_spacing_factor <= 1.0:
        raise ValueError("seed_spacing_factor must exceed 1 or seeding can interpenetrate")
    if not (0.0 <= config.seed_jitter_fraction < 0.5):
        raise ValueError("seed_jitter_fraction must be in [0, 0.5) of the seeded surface gap")
    if config.n_release_batches < 1:
        raise ValueError("n_release_batches must be >= 1")
    if config.n_release_batches > config.n_particles:
        raise ValueError("n_release_batches cannot exceed n_particles")
    if config.release_timeout_s < config.release_interval_s:
        raise ValueError("release_timeout_s must be >= release_interval_s")
    if config.release_quiet_speed_m_s <= 0:
        raise ValueError("release_quiet_speed_m_s must be > 0")
    fall_time = math.sqrt(2.0 * config.drop_gap_mm * 1.0e-3 / 9.81)
    if config.release_interval_s < fall_time:
        raise ValueError(
            f"release_interval_s ({config.release_interval_s}) is shorter than the "
            f"{fall_time:.4f} s free fall over drop_gap_mm; batches would overlap in flight"
        )
    # Worst case the pour takes one timeout per batch; the run still has to have
    # room to settle afterwards.
    worst_pour = (config.n_release_batches - 1) * config.release_timeout_s
    if config.max_sim_time_s < worst_pour + 0.3:
        raise ValueError(
            f"max_sim_time_s ({config.max_sim_time_s}) cannot accommodate a worst-case "
            f"{worst_pour:.3f} s pour plus settling"
        )
    if not (
        0.0
        < config.fit_lower_height_fraction
        < config.fit_upper_height_fraction
        < 1.0
    ):
        raise ValueError("fit height fractions must satisfy 0 < lower < upper < 1")
    if config.min_sim_time_s > config.max_sim_time_s:
        raise ValueError("min_sim_time_s cannot exceed max_sim_time_s")


def derive_repose_geometry(config: ReposeConfig, template: ClumpTemplate) -> dict[str, Any]:
    """Size the domain and the release column.

    Uses ``packing_fraction_for_sizing`` and ``sizing_repose_angle_deg`` ONLY to
    guarantee the walls stay clear of the heap.  A wrong guess makes the box
    slightly too big or too small; the ``wall_clearance`` gate catches the
    too-small case and fails the run rather than reporting a confined angle.
    """
    _validate_repose_config(config)
    bulk_volume = (
        config.n_particles * template.cylinder_volume_m3 / config.packing_fraction_for_sizing
    )
    tan_sizing = math.tan(math.radians(config.sizing_repose_angle_deg))
    predicted_base_radius = (3.0 * bulk_volume / (math.pi * tan_sizing)) ** (1.0 / 3.0)
    predicted_height = predicted_base_radius * tan_sizing

    bound_d = template.bounding_diameter_m
    half_extent = max(
        config.domain_radius_factor * predicted_base_radius,
        predicted_base_radius + 6.0 * bound_d,
    )

    seed_spacing = config.seed_spacing_factor * bound_d
    # The release disc must hold enough lattice sites that a batch is a few thin
    # layers rather than a tall slug.  For an elongated clump the bounding
    # diameter is large, so a disc sized purely as a fraction of the heap radius
    # would only fit a handful of sites.
    min_sites_radius = seed_spacing * math.sqrt(
        config.min_release_sites * 0.866 / math.pi
    )
    column_radius = max(
        config.column_radius_fraction * predicted_base_radius,
        1.5 * bound_d,
        min_sites_radius,
    )
    sites = _hex_disc_sites(column_radius, seed_spacing)
    if sites.shape[0] < 1:
        raise ValueError("release column is too narrow to hold a single particle")

    n_batches = config.n_release_batches
    per_batch = int(math.ceil(config.n_particles / n_batches))
    layers_per_batch = int(math.ceil(per_batch / sites.shape[0]))
    batch_height = (layers_per_batch - 1) * seed_spacing + bound_d
    # Parked batches sit above every reachable heap height, stacked with a full
    # bounding diameter of clearance so the fixed stack can never touch itself.
    park_base_z = predicted_height + config.drop_gap_mm * 1.0e-3 + 2.0 * bound_d
    park_pitch = batch_height + bound_d
    park_top_z = park_base_z + (n_batches - 1) * park_pitch + batch_height
    box_height = park_top_z + 4.0 * bound_d
    return {
        "sizing_bulk_volume_m3": bulk_volume,
        "sizing_predicted_base_radius_m": predicted_base_radius,
        "sizing_predicted_height_m": predicted_height,
        "half_extent_m": half_extent,
        "box_height_m": box_height,
        "column_radius_m": column_radius,
        "seed_spacing_m": seed_spacing,
        "sites_per_layer": int(sites.shape[0]),
        "n_batches": n_batches,
        "particles_per_batch": per_batch,
        "layers_per_batch": layers_per_batch,
        "batch_height_m": batch_height,
        "park_base_z_m": park_base_z,
        "park_pitch_m": park_pitch,
        "park_top_z_m": park_top_z,
        "drop_gap_m": config.drop_gap_mm * 1.0e-3,
        "bounding_diameter_m": bound_d,
    }


def _hex_disc_sites(radius: float, spacing: float) -> np.ndarray:
    """Hexagonal lattice sites whose centres lie within ``radius``."""
    row_step = spacing * math.sqrt(3.0) / 2.0
    n_rows = int(math.floor(radius / row_step)) + 1
    n_cols = int(math.floor(radius / spacing)) + 1
    sites: list[tuple[float, float]] = []
    for j in range(-n_rows, n_rows + 1):
        y = j * row_step
        offset = 0.5 * spacing if j % 2 else 0.0
        for i in range(-n_cols - 1, n_cols + 2):
            x = i * spacing + offset
            if x * x + y * y <= radius * radius:
                sites.append((x, y))
    return np.asarray(sites, dtype=np.float64).reshape(-1, 2)


def generate_column(
    config: ReposeConfig, geometry: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed the parked batch stack.

    Returns ``(positions (N,3), quaternions (N,4) xyzw, batch_index (N,))``.
    Particles are ordered batch-major so each batch occupies a CONTIGUOUS owner
    ID range, which is what ``DEMSolver.SetOwnerPosition`` needs to teleport a
    batch to the release plane in one call.

    Positions are jittered inside the seeded surface gap.  Without jitter the
    hexagonal lattice is a crystal, and a crystal column can stand up instead of
    slumping - which is exactly the failure that motivated batch pouring.
    """
    rng = np.random.default_rng(config.seed)
    sites = _hex_disc_sites(geometry["column_radius_m"], geometry["seed_spacing_m"])
    spacing = float(geometry["seed_spacing_m"])
    n_batches = int(geometry["n_batches"])
    per_batch = int(geometry["particles_per_batch"])
    positions: list[np.ndarray] = []
    batch_index: list[np.ndarray] = []
    remaining = config.n_particles
    for batch in range(n_batches):
        take_batch = min(per_batch, remaining)
        if take_batch <= 0:
            break
        base_z = float(geometry["park_base_z_m"]) + batch * float(geometry["park_pitch_m"])
        left = take_batch
        layer = 0
        while left > 0:
            order = rng.permutation(sites.shape[0])
            take = min(left, sites.shape[0])
            chosen = sites[order[:take]]
            z = base_z + layer * spacing
            positions.append(np.column_stack([chosen, np.full(take, z)]))
            left -= take
            layer += 1
        batch_index.append(np.full(take_batch, batch, dtype=np.int64))
        remaining -= take_batch
    result = np.concatenate(positions, axis=0)
    batches = np.concatenate(batch_index, axis=0)
    if result.shape != (config.n_particles, 3) or batches.shape != (config.n_particles,):
        raise AssertionError(f"seed shape mismatch: {result.shape}, {batches.shape}")

    # Jitter must not consume the whole seeded gap or particles interpenetrate.
    gap = spacing - float(geometry["bounding_diameter_m"])
    amplitude = config.seed_jitter_fraction * gap / math.sqrt(3.0)
    if amplitude > 0.0:
        result = result + rng.uniform(-amplitude, amplitude, size=result.shape)

    if config.randomize_initial_orientation:
        # Uniform random unit quaternions (Shoemake), xyzw ordering.
        u = rng.random((config.n_particles, 3))
        quats = np.column_stack(
            [
                np.sqrt(1.0 - u[:, 0]) * np.sin(2.0 * math.pi * u[:, 1]),
                np.sqrt(1.0 - u[:, 0]) * np.cos(2.0 * math.pi * u[:, 1]),
                np.sqrt(u[:, 0]) * np.sin(2.0 * math.pi * u[:, 2]),
                np.sqrt(u[:, 0]) * np.cos(2.0 * math.pi * u[:, 2]),
            ]
        )
    else:
        quats = np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]]), (config.n_particles, 1))
    return result, quats.astype(np.float64), batches


# ---------------------------------------------------------------------------
# 4. Sphere reconstruction and the repose-angle measurement
# ---------------------------------------------------------------------------


def sphere_centres_world(
    clump_positions_m: np.ndarray,
    clump_quaternions_xyzw: np.ndarray,
    sphere_offsets_m: np.ndarray,
) -> np.ndarray:
    """Expand clump bodies into their constituent sphere centres.

    Returns ``(N * n_spheres, 3)`` ordered clump-major.
    """
    positions = np.asarray(clump_positions_m, dtype=np.float64)
    offsets = np.asarray(sphere_offsets_m, dtype=np.float64)
    rot = _quat_to_matrix(clump_quaternions_xyzw)
    # (N, k, 3) = (N, 3, 3) @ (k, 3)^T
    rotated = np.einsum("nij,kj->nki", rot, offsets)
    return (positions[:, None, :] + rotated).reshape(-1, 3)


def _quat_to_matrix(quats_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quats_xyzw, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"expected (N,4) quaternions, got {q.shape}")
    norms = np.linalg.norm(q, axis=1)
    if not np.all(np.isfinite(norms)) or float(np.abs(norms - 1.0).max()) > 1.0e-3:
        raise ValueError(
            "quaternions are not unit norm "
            f"(max deviation {float(np.abs(norms - 1.0).max()):.6g})"
        )
    q = q / norms[:, None]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[:, 0, 1] = 2.0 * (x * y - z * w)
    m[:, 0, 2] = 2.0 * (x * z + y * w)
    m[:, 1, 0] = 2.0 * (x * y + z * w)
    m[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[:, 1, 2] = 2.0 * (y * z - x * w)
    m[:, 2, 0] = 2.0 * (x * z - y * w)
    m[:, 2, 1] = 2.0 * (y * z + x * w)
    m[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return m


def min_surface_gap_m(centres: np.ndarray, radius: float, *, exclude_stride: int = 1) -> float:
    """Closest surface-to-surface gap between spheres, via a uniform spatial hash.

    ``exclude_stride`` suppresses pairs belonging to the same clump: with
    clump-major ordering, spheres ``i`` and ``j`` share a body when
    ``i // exclude_stride == j // exclude_stride``.  Intra-clump spheres are
    supposed to overlap, so counting them would make the gate meaningless.
    """
    points = np.asarray(centres, dtype=np.float64)
    cell = 2.2 * float(radius)
    origin = points.min(axis=0) - cell
    keys = np.floor((points - origin) / cell).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(index)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    best_sq = math.inf
    for index, key in enumerate(keys):
        cx, cy, cz = int(key[0]), int(key[1]), int(key[2])
        body = index // exclude_stride
        for dx, dy, dz in offsets:
            for other in buckets.get((cx + dx, cy + dy, cz + dz), ()):
                if other >= index or other // exclude_stride == body:
                    continue
                delta = points[index] - points[other]
                dsq = float(delta @ delta)
                if dsq < best_sq:
                    best_sq = dsq
    if not math.isfinite(best_sq):
        return math.inf
    return math.sqrt(best_sq) - 2.0 * float(radius)


def _min_cross_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Smallest centre-to-centre distance between two point sets."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return math.inf
    best = math.inf
    for start in range(0, a.shape[0], 256):
        chunk = a[start : start + 256]
        d = np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def surface_height_field(
    centres: np.ndarray,
    radius: float,
    half_extent: float,
    cell_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-of-material height over a square grid.

    The operator is the exact highest sphere surface above each cell centre:
    ``h(cx, cy) = max_i (z_i + sqrt(r^2 - d_i^2))`` over spheres whose horizontal
    distance ``d_i`` to the cell centre is below ``r``.  Cells with no sphere
    above them are ``0`` (bare floor).  Returns ``(height, x_edges_centres,
    y_centres)`` with ``height[row, col]`` indexed ``row -> +y``, ``col -> +x``.
    """
    r = float(radius)
    n = int(math.ceil(2.0 * half_extent / cell_m))
    axis = (np.arange(n, dtype=np.float64) + 0.5) * cell_m - half_extent
    height = np.zeros((n, n), dtype=np.float64)
    pts = np.asarray(centres, dtype=np.float64)
    reach = int(math.ceil(r / cell_m))
    col_index = np.clip(
        np.floor((pts[:, 0] + half_extent) / cell_m).astype(np.int64), 0, n - 1
    )
    row_index = np.clip(
        np.floor((pts[:, 1] + half_extent) / cell_m).astype(np.int64), 0, n - 1
    )
    for k in range(pts.shape[0]):
        c0, r0 = int(col_index[k]), int(row_index[k])
        cs = slice(max(0, c0 - reach), min(n, c0 + reach + 1))
        rs = slice(max(0, r0 - reach), min(n, r0 + reach + 1))
        dx = axis[cs] - pts[k, 0]
        dy = axis[rs] - pts[k, 1]
        dsq = dy[:, None] ** 2 + dx[None, :] ** 2
        inside = dsq < r * r
        if not inside.any():
            continue
        cap = np.zeros_like(dsq)
        cap[inside] = pts[k, 2] + np.sqrt(r * r - dsq[inside])
        block = height[rs, cs]
        np.maximum(block, np.where(inside, cap, block), out=block)
        height[rs, cs] = block
    return height, axis, axis


def radial_profile(
    height: np.ndarray,
    axis: np.ndarray,
    centre_xy: tuple[float, float],
    cell_m: float,
) -> np.ndarray:
    """Annulus-averaged height profile: rows of ``RADIAL_PROFILE_COLUMNS``."""
    xx, yy = np.meshgrid(axis - centre_xy[0], axis - centre_xy[1], indexing="xy")
    r = np.hypot(xx, yy)
    bins = np.floor(r / cell_m).astype(np.int64)
    n_bins = int(bins.max()) + 1
    counts = np.bincount(bins.ravel(), minlength=n_bins).astype(np.float64)
    sums = np.bincount(bins.ravel(), weights=height.ravel(), minlength=n_bins)
    maxima = np.zeros(n_bins, dtype=np.float64)
    np.maximum.at(maxima, bins.ravel(), height.ravel())
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    radii = (np.arange(n_bins, dtype=np.float64) + 0.5) * cell_m
    return np.column_stack([radii, means, maxima, counts])


def _interp_radius_at_height(profile: np.ndarray, target_h: float) -> float | None:
    """First radius (outward from the apex) where the mean height crosses target."""
    radii = profile[:, 0]
    heights = profile[:, 1]
    peak = int(np.argmax(heights))
    for i in range(peak, len(radii) - 1):
        h0, h1 = heights[i], heights[i + 1]
        if h0 >= target_h > h1:
            if h0 == h1:
                return float(radii[i])
            t = (h0 - target_h) / (h0 - h1)
            return float(radii[i] + t * (radii[i + 1] - radii[i]))
    return None


def _fit_slope_angle(
    profile: np.ndarray, r_lo: float, r_hi: float
) -> tuple[float, float, int]:
    """Area-weighted least-squares line over ``r in [r_lo, r_hi]``.

    Returns ``(angle_deg, r_squared, n_points)``.  ``angle_deg`` is
    ``atan(-slope)``: a heap slopes down with radius, so the fitted slope is
    negative and the repose angle is its magnitude in degrees.
    """
    mask = (profile[:, 0] >= r_lo) & (profile[:, 0] <= r_hi) & (profile[:, 3] > 0)
    x = profile[mask, 0]
    y = profile[mask, 1]
    w = profile[mask, 3]
    if x.size < 3:
        return (float("nan"), float("nan"), int(x.size))
    wsum = w.sum()
    xm = float((w * x).sum() / wsum)
    ym = float((w * y).sum() / wsum)
    sxx = float((w * (x - xm) ** 2).sum())
    sxy = float((w * (x - xm) * (y - ym)).sum())
    if sxx <= 0.0:
        return (float("nan"), float("nan"), int(x.size))
    slope = sxy / sxx
    intercept = ym - slope * xm
    residual = y - (slope * x + intercept)
    ss_res = float((w * residual**2).sum())
    ss_tot = float((w * (y - ym) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (math.degrees(math.atan(-slope)), r_squared, int(x.size))


def measure_repose_angle(
    sphere_centres: np.ndarray,
    sphere_radius_m: float,
    config: ReposeConfig,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Measure the angle of repose of a settled heap.

    MEASUREMENT DEFINITION (this is the number reported as the result)
    ------------------------------------------------------------------
    ``primary`` = ``sidewall_regression``:

    1. Build a top-of-material height field on a square grid of
       ``profile_cell_mm`` using the exact highest-sphere-surface operator.
    2. Take the heap axis as the horizontal centroid of all sphere centres.
    3. Collapse the field into annuli of one cell width, averaging height inside
       each annulus (mean, not max: a single perched pellet must not set the
       profile).
    4. ``h_apex`` = the largest annulus mean height.
    5. Find ``r_upper`` and ``r_lower``, the radii where the annulus mean crosses
       ``0.80 h_apex`` and ``0.20 h_apex``, by linear interpolation.
    6. Fit a straight line to ``h(r)`` over ``[r_upper, r_lower]``, weighted by
       the number of grid cells in each annulus.  The angle is
       ``atan(-slope)`` in degrees.

    The band excludes the flat top (an artifact of a finite-width release
    column) and the toe (where the profile flattens into the floor), which is
    where the two classical definitions disagree most.

    ``secondary`` = ``apex_to_toe``: ``atan(h_apex / (r_toe - r_plateau))``,
    where ``r_plateau`` is the radius at ``0.95 h_apex`` and ``r_toe`` is the
    outermost radius whose annulus mean still exceeds half a sphere diameter.
    Reported for comparison only; it is systematically different from the
    regression angle and the two must never be mixed between runs.

    Quality gates: the regression ``r_squared``, the spread of the same fit
    repeated in four azimuthal quadrants, and the clearance between the heap toe
    and the domain wall.
    """
    centres = np.asarray(sphere_centres, dtype=np.float64)
    cell = config.profile_cell_mm * 1.0e-3
    half_extent = float(geometry["half_extent_m"])
    height, axis, _ = surface_height_field(centres, sphere_radius_m, half_extent, cell)
    # Heap axis = the height-weighted centroid of the top-of-material field, i.e.
    # the cone's centre of mass in plan view.  A plain particle centroid is pulled
    # off-centre by the handful of pellets that skitter away across the floor;
    # those cells carry ~zero height and so cannot bias this estimate.
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    weight = height.ravel()
    if weight.sum() <= 0.0:
        raise RuntimeError("REPOSE_MEASURE_FAIL: heap has zero height")
    axis_xy = (
        float((grid_x.ravel() * weight).sum() / weight.sum()),
        float((grid_y.ravel() * weight).sum() / weight.sum()),
    )
    profile = radial_profile(height, axis, axis_xy, cell)

    h_apex = float(profile[:, 1].max())
    if h_apex <= 0.0:
        raise RuntimeError("REPOSE_MEASURE_FAIL: heap has zero height")
    r_upper = _interp_radius_at_height(profile, config.fit_upper_height_fraction * h_apex)
    r_lower = _interp_radius_at_height(profile, config.fit_lower_height_fraction * h_apex)
    if r_upper is None or r_lower is None or r_lower <= r_upper:
        raise RuntimeError(
            "REPOSE_MEASURE_FAIL: could not bracket the sidewall band "
            f"(r_upper={r_upper}, r_lower={r_lower}, h_apex={h_apex:.6f} m)"
        )
    primary_deg, r_squared, n_fit = _fit_slope_angle(profile, r_upper, r_lower)

    toe_threshold = config.toe_height_fraction_of_diameter * 2.0 * sphere_radius_m
    above_toe = profile[(profile[:, 1] >= toe_threshold) & (profile[:, 3] > 0)]
    r_toe = float(above_toe[:, 0].max()) if above_toe.size else float("nan")
    r_plateau = _interp_radius_at_height(profile, 0.95 * h_apex)
    if r_plateau is None:
        r_plateau = 0.0
    secondary_deg = (
        math.degrees(math.atan(h_apex / (r_toe - r_plateau)))
        if math.isfinite(r_toe) and r_toe > r_plateau
        else float("nan")
    )

    quadrant_angles: list[float] = []
    xx, yy = np.meshgrid(axis - axis_xy[0], axis - axis_xy[1], indexing="xy")
    theta = np.arctan2(yy, xx)
    for q in range(4):
        lo = -math.pi + q * math.pi / 2.0
        hi = lo + math.pi / 2.0
        mask = (theta >= lo) & (theta < hi)
        sub_height = np.where(mask, height, 0.0)
        sub_profile = radial_profile(sub_height, axis, axis_xy, cell)
        # Only the cells inside this quadrant carry information.
        counts = np.bincount(
            np.floor(np.hypot(xx, yy)[mask] / cell).astype(np.int64),
            minlength=sub_profile.shape[0],
        )[: sub_profile.shape[0]].astype(np.float64)
        sums = np.bincount(
            np.floor(np.hypot(xx, yy)[mask] / cell).astype(np.int64),
            weights=height[mask],
            minlength=sub_profile.shape[0],
        )[: sub_profile.shape[0]]
        with np.errstate(invalid="ignore", divide="ignore"):
            sub_profile[:, 1] = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
        sub_profile[:, 3] = counts
        q_apex = float(sub_profile[:, 1].max())
        q_up = _interp_radius_at_height(sub_profile, config.fit_upper_height_fraction * q_apex)
        q_lo = _interp_radius_at_height(sub_profile, config.fit_lower_height_fraction * q_apex)
        if q_up is None or q_lo is None or q_lo <= q_up:
            quadrant_angles.append(float("nan"))
            continue
        quadrant_angles.append(_fit_slope_angle(sub_profile, q_up, q_lo)[0])
    finite_quadrants = [a for a in quadrant_angles if math.isfinite(a)]
    quadrant_spread = (
        max(finite_quadrants) - min(finite_quadrants) if len(finite_quadrants) >= 2 else float("nan")
    )

    # Confinement is a property of the HEAP, so the gate is the toe radius - the
    # edge of the load-bearing body - not the farthest individual pellet.  A few
    # pellets always skitter off across a bare floor (that is real behaviour for
    # low-friction grains, and it is why physical repose rigs use a fixed-diameter
    # base plate that strays fall off).  They sit in near-zero-height annuli well
    # outside the fit band and cannot move the angle.
    #
    # A separate check makes sure they stay a minority: if the "strays" are most
    # of the material then the pour sprayed and there is no heap to measure.
    radii_from_axis = np.hypot(centres[:, 0] - axis_xy[0], centres[:, 1] - axis_xy[1])
    max_radius = float(radii_from_axis.max())
    bulk_radius = float(np.quantile(radii_from_axis, 0.995))
    gate_radius = config.wall_clearance_fraction * half_extent
    wall_clearance_ok = math.isfinite(r_toe) and r_toe <= gate_radius
    escapees = int(np.count_nonzero(radii_from_axis > gate_radius))
    heap_fraction = (
        float(np.mean(radii_from_axis <= 1.15 * r_toe)) if math.isfinite(r_toe) else 0.0
    )

    checks = {
        "fit_r_squared_ok": bool(math.isfinite(r_squared) and r_squared >= config.min_fit_r_squared),
        "quadrant_spread_ok": bool(
            math.isfinite(quadrant_spread) and quadrant_spread <= config.max_quadrant_spread_deg
        ),
        "wall_clearance_ok": bool(wall_clearance_ok),
        "heap_holds_the_bulk": bool(heap_fraction >= config.min_heap_fraction),
        "angle_finite": bool(math.isfinite(primary_deg)),
        "angle_physical": bool(math.isfinite(primary_deg) and 5.0 < primary_deg < 70.0),
        # NOT a precision check - see max_quadrant_spread_deg.  This only catches a
        # heap that is lopsided rather than merely grainy.
        "not_lopsided": bool(
            not math.isfinite(quadrant_spread) or quadrant_spread <= config.max_quadrant_spread_deg
        ),
    }
    return {
        "primary_definition": REPOSE_PRIMARY_DEFINITION,
        "primary_definition_detail": REPOSE_PRIMARY_DEFINITION_TEXT,
        "primary_definition_band": [
            config.fit_upper_height_fraction,
            config.fit_lower_height_fraction,
        ],
        "secondary_definition": "apex_to_toe",
        "secondary_definition_detail": (
            "atan(h_apex / (r_toe - r_plateau)); r_plateau at 0.95*h_apex, r_toe the outermost "
            f"annulus with mean height >= {config.toe_height_fraction_of_diameter:.2f} sphere diameters"
        ),
        "repose_angle_deg": float(primary_deg),
        "repose_angle_secondary_deg": float(secondary_deg),
        "fit_r_squared": float(r_squared),
        "fit_band_m": [float(r_upper), float(r_lower)],
        "fit_points": n_fit,
        "apex_height_m": h_apex,
        "toe_radius_m": float(r_toe),
        "plateau_radius_m": float(r_plateau),
        "quadrant_angles_deg": [float(a) for a in quadrant_angles],
        "quadrant_spread_deg": float(quadrant_spread),
        "azimuthal_uncertainty_deg": float(0.5 * quadrant_spread),
        "quadrant_spread_role": (
            "reported azimuthal uncertainty of the angle at this particle count, "
            "NOT a validity gate; only lopsidedness beyond max_quadrant_spread_deg fails"
        ),
        "heap_axis_xy_m": [axis_xy[0], axis_xy[1]],
        "max_particle_radius_m": max_radius,
        "bulk_particle_radius_p995_m": bulk_radius,
        "wall_gate_radius_m": gate_radius,
        "particles_beyond_gate_radius": escapees,
        "fraction_within_1p15_toe": heap_fraction,
        "domain_half_extent_m": half_extent,
        "profile_cell_m": cell,
        "checks": checks,
        "measurement_pass": bool(all(checks.values())),
    }


# ---------------------------------------------------------------------------
# 5. Serialization helpers (deterministic NPZ, mirrors sim_deme_pile.py)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _finite_only(value: Any) -> Any:
    """Replace NaN/Inf with None so the JSON stays strict and the gap is explicit.

    A measurement that could not be computed (an un-fittable quadrant, a heap
    with no toe) must not silently become a number, and must not silently make
    the whole artifact unwritable either.
    """
    if isinstance(value, dict):
        return {k: _finite_only(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_only(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _finite_only(value),
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
            _finite_only(value),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        raise FileExistsError(f"refusing to replace unexpected temporary file: {tmp}")
    try:
        with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for key in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info.create_system = 3
                z.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_tag(config: ReposeConfig) -> str:
    def token(value: float) -> str:
        return f"{value:g}".replace(".", "p").replace("-", "m")

    return (
        f"repose_{config.shape}_mu{token(config.particle_mu)}"
        f"_crr{token(config.rolling_friction)}_n{config.n_particles}_seed{config.seed}"
    )


def default_output_path(config: ReposeConfig) -> Path:
    return OUT_DIR / f"{run_tag(config)}.npz"


# ---------------------------------------------------------------------------
# 6. The simulation
# ---------------------------------------------------------------------------


def run_repose(config: ReposeConfig, output: Path) -> dict[str, Any]:
    """Pour a heap, settle it, measure its angle, write the artifact."""
    if output.suffix != ".npz":
        raise ValueError("output must end in .npz")
    summary_path = output.with_suffix(".json")
    if output.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {summary_path}")

    try:
        import DEME
    except ImportError as exc:
        raise RuntimeError(
            "DEME is not importable here; run with ~/miniconda3/envs/roarm/bin/python"
        ) from exc
    deme_version = _package_version("DEME")
    if deme_version != "2.4.0":
        raise RuntimeError(f"DEME version contract failed: {deme_version} != 2.4.0")

    run_started = datetime.now(timezone.utc).isoformat()
    wall_t0 = time.perf_counter()
    pellet = config.pellet()
    template = build_template(
        pellet, config.shape, aspect=config.aspect, size_match=config.size_match
    )
    geometry = derive_repose_geometry(config, template)
    positions0, quats0, batch_index = generate_column(config, geometry)

    offsets = np.asarray(template.offsets_m, dtype=np.float64)
    seeded_spheres = sphere_centres_world(positions0, quats0, offsets)
    seed_gap = min_surface_gap_m(
        seeded_spheres, template.sphere_radius_m, exclude_stride=template.n_spheres
    )
    if seed_gap <= 0.0:
        raise RuntimeError(
            f"SEED_PENETRATION_FAIL: seeded surface gap {seed_gap*1e3:.6f} mm <= 0; "
            "raise seed_spacing_factor (D460 sec.S)"
        )

    solver = DEME.DEMSolver()
    solver.SetVerbosity("ERROR")
    material = solver.LoadMaterial(
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
    solver.SetMaterialPropertyPair("mu", material, wall_material, config.wall_mu)
    solver.SetMaterialPropertyPair("CoR", material, wall_material, config.restitution)
    solver.SetMaterialPropertyPair("Crr", material, wall_material, config.rolling_friction)
    solver.UseFrictionalHertzianModel()
    # Must precede Initialize, or GetContactDetailedInfo raises "does not have
    # field" for every channel.  The D341 contract needs contact points and
    # force arrows for a physics/settle verdict.
    solver.SetContactOutputContent(["OWNER", "FORCE", "POINT", "NORMAL"])

    if template.n_spheres == 1:
        clump_type = solver.LoadSphereType(
            template.mass_kg, template.sphere_radius_m, material
        )
    else:
        clump_type = solver.LoadClumpType(
            template.mass_kg,
            list(template.moi_kg_m2),
            [template.sphere_radius_m] * template.n_spheres,
            [list(o) for o in template.offsets_m],
            material,
        )
    clump_type.SetVolume(template.union_volume_m3)

    batch = solver.AddClumps(clump_type, positions0.tolist())
    if template.n_spheres > 1:
        batch.SetOriQ(quats0.tolist())
    # Family i+1 = parked batch i (fixed).  Family 0 = poured, free to move.
    batch.SetFamilies((batch_index + 1).astype(np.int64).tolist())
    for index in range(int(geometry["n_batches"])):
        solver.SetFamilyFixed(index + 1)

    half = float(geometry["half_extent_m"])
    solver.InstructBoxDomainDimension(
        (-half, half), (-half, half), (0.0, float(geometry["box_height_m"]))
    )
    solver.InstructBoxDomainBoundingBC("top_open", wall_material)
    solver.SetInitTimeStep(config.dt_s)
    solver.SetGravitationalAcceleration([0.0, 0.0, -9.81])
    solver.SetCDUpdateFreq(config.cd_update_freq)
    # DEME lets the dynamics thread run ahead of the contact-detection thread and
    # inflates the contact margin to stay safe.  While a heap is still agitated
    # that margin balloons, the potential-contact count per sphere explodes, and
    # contact detection becomes ~70x more expensive - or aborts outright with
    # "On average a sphere has N contacts, more than the max allowance".  Capping
    # the adaptive update frequency bounds the drift, and hence the margin.
    solver.SetCDMaxUpdateFreq(config.cd_max_update_freq)
    solver.SetErrorOutAvgContacts(config.error_out_avg_contacts)
    solver.SetInitBinNumTarget(config.init_bin_num_target)

    init_t0 = time.perf_counter()
    solver.Initialize()
    init_wall_s = time.perf_counter() - init_t0
    n = config.n_particles
    if solver.GetNumClumps() != n:
        raise RuntimeError(f"clump count mismatch: {solver.GetNumClumps()} != {n}")

    # --- Contract checks against DEME's own readback, before any dynamics. ---
    mass_readback = np.asarray(solver.GetOwnerMass(0, n), dtype=np.float64)
    moi_readback = np.asarray(solver.GetOwnerMOI(0, n), dtype=np.float64)
    quat_readback = np.asarray(solver.GetOwnerOriQ(0, n), dtype=np.float64)
    mass_err = float(np.abs(mass_readback - template.mass_kg).max() / template.mass_kg)
    moi_err = float(
        np.abs(moi_readback - np.asarray(template.moi_kg_m2)).max()
        / max(template.moi_kg_m2)
    )
    if mass_err > 1.0e-5 or moi_err > 1.0e-5:
        raise RuntimeError(
            f"MASS_PROPERTY_READBACK_FAIL: mass_rel_err={mass_err:.3e}, moi_rel_err={moi_err:.3e}"
        )
    # If the xyzw quaternion convention were wrong, reconstructing sphere
    # centres from the readback would place spheres in the wrong place and
    # distinct clumps would appear to interpenetrate.
    init_spheres = sphere_centres_world(
        np.asarray(solver.GetOwnerPosition(0, n), dtype=np.float64), quat_readback, offsets
    )
    init_gap = min_surface_gap_m(
        init_spheres, template.sphere_radius_m, exclude_stride=template.n_spheres
    )
    if init_gap <= -1.0e-9:
        raise RuntimeError(
            "QUATERNION_CONVENTION_FAIL: reconstructed spheres overlap at t=0 "
            f"(gap {init_gap*1e3:.6f} mm); xyzw ordering assumption is wrong"
        )

    print(
        f"init {config.shape}: {init_wall_s:.2f}s N={n} spheres={n*template.n_spheres} "
        f"R={template.sphere_radius_m*1e3:.4f}mm box=+-{half*1e3:.1f}mm "
        f"H={geometry['box_height_m']*1e3:.1f}mm seed_gap={seed_gap*1e3:.4f}mm",
        flush=True,
    )

    batch_slices = [
        (int(np.searchsorted(batch_index, b, "left")), int(np.searchsorted(batch_index, b, "right")))
        for b in range(int(geometry["n_batches"]))
    ]
    drop_gap = float(geometry["drop_gap_m"])
    bound_d = float(geometry["bounding_diameter_m"])
    column_radius = float(geometry["column_radius_m"])
    release_log: list[dict[str, float]] = []

    def footprint_mask(pos: np.ndarray, count: int) -> np.ndarray:
        return np.hypot(pos[:count, 0], pos[:count, 1]) <= 1.3 * column_radius

    def heap_top_under_column(pos: np.ndarray, vel: np.ndarray, count: int) -> float:
        """Height of the SETTLED material the next batch will land on.

        Two filters, both load-bearing:

        * only material under the pour footprint, and
        * only material that has come to rest.

        Dropping either one creates a positive feedback loop.  Airborne or
        bouncing pellets sit above the heap, so counting them raises the release
        plane, which makes the next batch fall further, which splashes harder,
        which raises the apparent top again.  A release plane that climbs to
        twice the heap height both sprays material across the domain and keeps
        the heap agitated, and a permanently agitated heap makes DEME's contact
        margin balloon (see SetCDMaxUpdateFreq above).

        The 99.5th percentile rather than the maximum, so one pellet perched on
        the apex cannot drive the plane either.
        """
        if count <= 0:
            return 0.0
        inside = footprint_mask(pos, count)
        settled = np.linalg.norm(vel[:count], axis=1) <= config.release_quiet_speed_m_s
        chosen = inside & settled
        if not chosen.any():
            chosen = inside if inside.any() else settled
        if not chosen.any():
            return 0.0
        return float(np.quantile(pos[:count][chosen, 2], 0.995))

    def pour_zone_is_quiet(pos: np.ndarray, vel: np.ndarray, count: int) -> bool:
        """True once the previous batch has actually landed under the column."""
        if count <= 0:
            return True
        inside = np.hypot(pos[:count, 0], pos[:count, 1]) <= 1.5 * column_radius
        if not inside.any():
            return True
        return bool(
            np.linalg.norm(vel[:count][inside], axis=1).max()
            <= config.release_quiet_speed_m_s
        )

    history: list[list[float]] = []
    state_history: list[np.ndarray] = []
    stable = 0
    settled = False
    next_batch = 0
    sample_interval = config.advance_chunk_s * config.chunks_per_sample
    max_samples = int(math.ceil(config.max_sim_time_s / sample_interval))
    dyn_t0 = time.perf_counter()
    positions = positions0.copy()
    quats = quats0.copy()
    velocities = np.zeros((n, 3))
    last_release_time = -math.inf
    for sample in range(max_samples):
        sim_time = float(solver.GetSimTime())
        # --- pour: release the next batch once the previous one has landed ---
        elapsed = sim_time - last_release_time
        if next_batch < len(batch_slices) and (
            elapsed >= config.release_timeout_s
            or (
                elapsed >= config.release_interval_s
                and pour_zone_is_quiet(positions, velocities, batch_slices[next_batch][0])
            )
        ):
            start, stop = batch_slices[next_batch]
            released = positions[:start] if start else np.zeros((0, 3))
            heap_top = heap_top_under_column(positions, velocities, start)
            release_z = heap_top + drop_gap
            parked = positions[start:stop]
            shifted = parked.copy()
            shifted[:, 2] += release_z - float(parked[:, 2].min())
            clearance = math.inf
            if released.size:
                # A teleport landing on top of existing material injects an
                # overlap the solver then has to explode out of.  A grazing
                # stray is tolerable; a real burial is not.
                clearance = _min_cross_distance(shifted, released)
                if clearance < 0.5 * bound_d:
                    raise RuntimeError(
                        f"RELEASE_OVERLAP_FAIL: batch {next_batch} teleported to z={release_z*1e3:.2f} mm "
                        f"lands {clearance*1e3:.3f} mm from existing material "
                        f"(need >= {0.5*bound_d*1e3:.3f} mm); increase drop_gap_mm"
                    )
            solver.SetOwnerPosition(start, shifted.tolist())
            solver.ChangeFamily(next_batch + 1, 0)
            positions[start:stop] = shifted
            release_log.append(
                {
                    "batch": float(next_batch),
                    "sim_time_s": sim_time,
                    "heap_top_m": heap_top,
                    "release_z_m": release_z,
                    "n_particles": float(stop - start),
                    "clearance_to_existing_m": (
                        None if not math.isfinite(clearance) else float(clearance)
                    ),
                }
            )
            next_batch += 1
            last_release_time = sim_time

        for _ in range(config.chunks_per_sample):
            solver.DoDynamicsThenSync(config.advance_chunk_s)
        positions = np.asarray(solver.GetOwnerPosition(0, n), dtype=np.float64)
        velocities = np.asarray(solver.GetOwnerVelocity(0, n), dtype=np.float64)
        quats = np.asarray(solver.GetOwnerOriQ(0, n), dtype=np.float64)
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise RuntimeError("DEME returned NaN/Inf state")
        sim_time = float(solver.GetSimTime())
        # Statistics over POURED particles only: parked batches sit at exactly
        # zero speed and would otherwise make the heap look settled mid-pour.
        poured = batch_slices[next_batch - 1][1] if next_batch else 0
        speeds = np.linalg.norm(velocities[:poured], axis=1)
        max_speed = float(speeds.max()) if speeds.size else 0.0
        p99 = float(np.quantile(speeds, 0.99)) if speeds.size else 0.0
        rms = float(np.sqrt(np.mean(speeds * speeds))) if speeds.size else 0.0
        radial = np.hypot(positions[:poured, 0], positions[:poured, 1])
        history.append(
            [
                sim_time,
                max_speed,
                p99,
                rms,
                float(0.5 * template.mass_kg * np.sum(speeds * speeds)),
                float(solver.GetNumContacts()),
                float(positions[:poured, 2].max()) if poured else 0.0,
                float(radial.max()) if poured else 0.0,
            ]
        )
        if config.store_state_history:
            state_history.append(np.concatenate([positions, quats], axis=1).copy())
        pour_complete = next_batch >= len(batch_slices)
        gate = bool(
            pour_complete
            and sim_time + 1.0e-12 >= config.min_sim_time_s
            and max_speed <= config.speed_max_mm_s * 1.0e-3
            and p99 <= config.speed_p99_mm_s * 1.0e-3
            and rms <= config.speed_rms_mm_s * 1.0e-3
        )
        stable = stable + 1 if gate else 0
        print(
            f"  settle {sample+1:03d}/{max_samples} t={sim_time:.3f}s poured={poured}/{n} "
            f"v(max/p99/rms)={max_speed*1e3:.2f}/{p99*1e3:.2f}/{rms*1e3:.2f}mm/s "
            f"top={history[-1][6]*1e3:.1f}mm r={history[-1][7]*1e3:.1f}mm "
            f"stable={stable}/{config.stable_samples}",
            flush=True,
        )
        if stable >= config.stable_samples:
            settled = True
            break
    dynamics_wall_s = time.perf_counter() - dyn_t0
    if next_batch < len(batch_slices):
        raise RuntimeError(
            f"POUR_INCOMPLETE: only {next_batch}/{len(batch_slices)} batches released before "
            "max_sim_time_s; raise max_sim_time_s or lower release_interval_s"
        )
    if not settled:
        last = history[-1]
        raise RuntimeError(
            "SETTLEMENT_GATE_FAIL: max sim time reached; last max/p99/rms="
            f"{last[1]*1e3:.3f}/{last[2]*1e3:.3f}/{last[3]*1e3:.3f} mm/s"
        )

    info = solver.GetContactDetailedInfo(0.0)
    contact_points = np.asarray(info.GetPoint(), dtype=np.float64).reshape(-1, 3)
    contact_forces = np.asarray(info.GetForce(), dtype=np.float64).reshape(-1, 3)
    if contact_points.shape[0] == 0:
        raise RuntimeError(
            "CONTACT_EXPORT_FAIL: a settled heap must have contacts; "
            "GetContactDetailedInfo returned none"
        )
    if contact_points.shape != contact_forces.shape:
        raise RuntimeError(
            f"CONTACT_EXPORT_FAIL: point/force shape mismatch "
            f"{contact_points.shape} vs {contact_forces.shape}"
        )

    spheres = sphere_centres_world(positions, quats, offsets)
    final_gap = min_surface_gap_m(
        spheres, template.sphere_radius_m, exclude_stride=template.n_spheres
    )
    allowed_overlap = (
        config.max_penetration_fraction_diameter * 2.0 * template.sphere_radius_m
    )
    penetration_pass = final_gap >= -allowed_overlap

    measurement = measure_repose_angle(spheres, template.sphere_radius_m, config, geometry)
    height, axis, _ = surface_height_field(
        spheres,
        template.sphere_radius_m,
        float(geometry["half_extent_m"]),
        config.profile_cell_mm * 1.0e-3,
    )
    profile = radial_profile(
        height,
        axis,
        (measurement["heap_axis_xy_m"][0], measurement["heap_axis_xy_m"][1]),
        config.profile_cell_mm * 1.0e-3,
    )
    total_wall_s = time.perf_counter() - wall_t0

    metadata: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "scientific_authority": "float64 NPZ arrays; Rerun is an inspection-only Float32 copy",
        "non_claims": NON_CLAIMS,
        "coordinate_frame": {
            "handedness": "right-handed",
            "axes": {"x": "east", "y": "north", "z": "up"},
            "origin": "container floor centre",
            "floor_z_m": 0.0,
            "units": "SI (m, m/s, kg, s); *_mm and *_deg are the only exceptions",
            "quaternion_order": "xyzw (verified: DEME identity readback is [0,0,0,1])",
        },
        "config": asdict(config),
        "pellet": asdict(pellet),
        "template": asdict(template),
        "template_derivation": {
            "mass_source": "rho * cylinder volume (pi/4 * D^2 * L); all shapes are mass-matched",
            "moi_source": (
                "exact closed-form inertia of the union-of-spheres solid "
                "(sphere moments minus pairwise-lens moments via spherical-cap integrals), "
                "scaled from unit density to the pellet mass"
            ),
            "moi_verified_against": "DEME GetOwnerMOI readback",
            "size_match": template.size_match,
            "volume_mismatch_fraction": template.volume_mismatch_fraction(),
            "nominal_pellet_aspect": pellet.nominal_aspect(),
            "aspect_bounds": list(aspect_bounds(template.n_spheres)),
        },
        "derived_geometry": geometry,
        "release_protocol": {
            "method": "batch pour from a constant-height release plane (virtual funnel)",
            "why_not_single_drop": (
                "releasing the whole seeded column at once is a granular column collapse, "
                "whose runout slope is not the poured angle of repose; a seeded hexagonal "
                "column can also stand up as a crystal instead of slumping"
            ),
            "n_batches": int(geometry["n_batches"]),
            "schedule": "event-driven: next batch waits for the pour zone to go quiet",
            "release_interval_s": config.release_interval_s,
            "release_quiet_speed_m_s": config.release_quiet_speed_m_s,
            "release_timeout_s": config.release_timeout_s,
            "drop_gap_m": drop_gap,
            "drop_gap_is_constant_per_batch": True,
            "releases": release_log,
            "pour_duration_s": (
                release_log[-1]["sim_time_s"] - release_log[0]["sim_time_s"]
                if len(release_log) > 1
                else 0.0
            ),
        },
        "settling_gate": {
            "settled": True,
            "rule": (
                "after the pour completes and after min_sim_time_s, max/p99/rms speed of the "
                "POURED particles each below threshold for stable_samples consecutive samples"
            ),
            "settled_sim_time_s": float(history[-1][0]),
            "samples": len(history),
            "thresholds_m_s": {
                "max": config.speed_max_mm_s * 1e-3,
                "p99": config.speed_p99_mm_s * 1e-3,
                "rms": config.speed_rms_mm_s * 1e-3,
            },
        },
        "seed_gate": {
            "no_seed_penetration": bool(seed_gap > 0.0),
            "seed_min_surface_gap_m": seed_gap,
            "initial_readback_min_surface_gap_m": init_gap,
            "mass_readback_rel_err": mass_err,
            "moi_readback_rel_err": moi_err,
        },
        "penetration_gate": {
            "pass": bool(penetration_pass),
            "min_inter_clump_surface_gap_m": final_gap,
            "allowed_overlap_m": allowed_overlap,
            "note": "intra-clump sphere pairs are excluded; they overlap by construction",
        },
        "repose": measurement,
        "settle_history_columns": HISTORY_COLUMNS,
        "radial_profile_columns": RADIAL_PROFILE_COLUMNS,
        "software": {
            "DEME": deme_version,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    }
    if not penetration_pass:
        metadata["penetration_gate"]["failure"] = (
            f"min inter-clump surface gap {final_gap*1e3:.4f} mm exceeds the "
            f"{allowed_overlap*1e3:.4f} mm allowance"
        )

    arrays = {
        "clump_positions_m": positions,
        "clump_quaternions_xyzw": quats,
        "velocities_m_s": velocities,
        "initial_positions_m": positions0,
        "initial_quaternions_xyzw": quats0,
        "sphere_offsets_m": offsets,
        "sphere_radii_m": np.full(template.n_spheres, template.sphere_radius_m),
        "particle_ids": np.arange(n, dtype=np.int64),
        "box_bounds_m": np.asarray(
            [[-half, half], [-half, half], [0.0, float(geometry["box_height_m"])]]
        ),
        "settle_history": np.asarray(history, dtype=np.float64),
        "state_history": (
            np.asarray(state_history, dtype=np.float64)
            if state_history
            else np.zeros((0, n, 7), dtype=np.float64)
        ),
        "radial_profile": profile,
        "contact_points_m": contact_points,
        "contact_forces_n": contact_forces,
        "metadata_json": np.asarray(_canonical_json(metadata)),
    }
    if set(arrays) != set(NPZ_KEYS):
        raise AssertionError(f"format key mismatch: {sorted(set(arrays) ^ set(NPZ_KEYS))}")
    _write_deterministic_npz(output, arrays)

    summary = {
        "artifact": f"{ARTIFACT}_SUMMARY",
        "npz": str(output),
        "npz_sha256": _sha256(output),
        "run_started_utc": run_started,
        "shape": config.shape,
        "aspect": template.aspect,
        "n_spheres": template.n_spheres,
        "n_particles": n,
        "particle_mu": config.particle_mu,
        "rolling_friction": config.rolling_friction,
        "restitution": config.restitution,
        "repose_angle_deg": measurement["repose_angle_deg"],
        "repose_angle_secondary_deg": measurement["repose_angle_secondary_deg"],
        "repose_definition": measurement["primary_definition"],
        "fit_r_squared": measurement["fit_r_squared"],
        "quadrant_spread_deg": measurement["quadrant_spread_deg"],
        "measurement_pass": measurement["measurement_pass"],
        "penetration_pass": bool(penetration_pass),
        "settled_sim_time_s": float(history[-1][0]),
        "initialization_wall_s": init_wall_s,
        "dynamics_wall_s": dynamics_wall_s,
        "total_wall_s": total_wall_s,
        "sphere_count": n * template.n_spheres,
        "non_claims": NON_CLAIMS,
    }
    _write_json(summary_path, summary)
    print(
        f"REPOSE_RUN_OK shape={config.shape} mu={config.particle_mu} Crr={config.rolling_friction} "
        f"angle={measurement['repose_angle_deg']:.3f}deg r2={measurement['fit_r_squared']:.4f} "
        f"pass={measurement['measurement_pass'] and penetration_pass} "
        f"wall={total_wall_s:.1f}s out={output}",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# 7. Sweep and the inverse lookup
# ---------------------------------------------------------------------------


DEFAULT_MU_GRID = (0.25, 0.45, 0.65)
DEFAULT_CRR_GRID = (0.01, 0.05, 0.12)


def run_sweep(
    base: ReposeConfig,
    shapes: Iterable[str],
    mu_grid: Iterable[float],
    crr_grid: Iterable[float],
    sweep_path: Path,
) -> dict[str, Any]:
    """Run the (shape x mu x Crr) grid and write one combined sweep table."""
    if sweep_path.exists():
        raise FileExistsError(f"refusing to overwrite {sweep_path}")
    shapes = list(shapes)
    mu_grid = [float(v) for v in mu_grid]
    crr_grid = [float(v) for v in crr_grid]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()
    for shape in shapes:
        for mu in mu_grid:
            for crr in crr_grid:
                config = replace(
                    base,
                    shape=shape,
                    particle_mu=mu,
                    wall_mu=mu,
                    rolling_friction=crr,
                    store_state_history=False,
                )
                out = default_output_path(config)
                cell = {"shape": shape, "particle_mu": mu, "rolling_friction": crr}
                if out.exists() and out.with_suffix(".json").is_file():
                    summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
                    summary["reused_existing"] = True
                    rows.append(summary)
                    continue
                error = _run_cell_subprocess(base, config, out)
                if error is not None:
                    failures.append({**cell, **error, "npz": str(out)})
                    print(f"SWEEP_CELL_FAIL {shape} mu={mu} Crr={crr}: {error['error']}", flush=True)
                    continue
                summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
                summary["reused_existing"] = False
                rows.append(summary)

    cost = _cost_multipliers(rows)
    trend = _shape_trend(rows)
    document = {
        "artifact": f"{ARTIFACT}_SWEEP",
        "schema_version": SCHEMA_VERSION,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": asdict(base),
        "grid": {"shapes": shapes, "mu": mu_grid, "crr": crr_grid},
        "cells_requested": len(shapes) * len(mu_grid) * len(crr_grid),
        "cells_completed": len(rows),
        "runs": rows,
        "failures": failures,
        "cost_multipliers": cost,
        "shape_trend": trend,
        "repose_definition": "sidewall_regression (see measure_repose_angle docstring)",
        "non_claims": NON_CLAIMS,
    }
    _write_json(sweep_path, document)
    print(
        f"SWEEP_OK cells={len(rows)}/{document['cells_requested']} "
        f"failures={len(failures)} out={sweep_path}",
        flush=True,
    )
    return document


def _run_cell_subprocess(
    base: ReposeConfig, config: ReposeConfig, out: Path
) -> dict[str, Any] | None:
    """Run one sweep cell in its own process; return None on success.

    DEME can abort in native code (a bad contact-detection state segfaults with
    no Python traceback and sometimes no stderr at all).  In-process that kills
    the whole sweep and loses every remaining cell, so each cell gets its own
    process: a crash is recorded as one failed cell and the sweep continues.
    Combined with the reuse-if-present check above, an interrupted sweep resumes
    where it stopped rather than restarting.
    """
    import subprocess

    command = [
        sys.executable, str(Path(__file__).resolve()), "--run",
        "--shape", config.shape,
        "--mu", repr(config.particle_mu),
        "--crr", repr(config.rolling_friction),
        "--output", str(out),
        "--no-state-history",
        "--size-match", base.size_match,
        "--n-particles", str(base.n_particles),
        "--seed", str(base.seed),
        "--pellet-dia-mm", repr(base.pellet_dia_mm),
        "--pellet-len-mm", repr(base.pellet_len_mm),
        "--particle-density-kg-m3", repr(base.particle_density_kg_m3),
        "--restitution", repr(base.restitution),
        "--young-modulus-pa", repr(base.young_modulus_pa),
        "--dt-s", repr(base.dt_s),
        "--cd-update-freq", str(base.cd_update_freq),
        "--cd-max-update-freq", str(base.cd_max_update_freq),
        "--advance-chunk-s", repr(base.advance_chunk_s),
        "--chunks-per-sample", str(base.chunks_per_sample),
        "--min-sim-time-s", repr(base.min_sim_time_s),
        "--max-sim-time-s", repr(base.max_sim_time_s),
        "--profile-cell-mm", repr(base.profile_cell_mm),
        "--column-radius-fraction", repr(base.column_radius_fraction),
        "--n-release-batches", str(base.n_release_batches),
    ]
    if base.aspect is not None:
        command += ["--aspect", repr(base.aspect)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=base.cell_timeout_s)
    except subprocess.TimeoutExpired:
        return {
            "error": f"cell exceeded the {base.cell_timeout_s:.0f} s wall-clock guard (DEME stall: the process sits near-idle rather than computing)",
            "returncode": None,
            "stderr_tail": [],
        }
    if result.returncode == 0 and out.with_suffix(".json").is_file():
        for line in result.stdout.splitlines():
            if line.startswith("REPOSE_RUN_OK"):
                print(line, flush=True)
        return None
    if result.returncode < 0:
        reason = f"killed by signal {-result.returncode} (native DEME abort, no Python traceback)"
    else:
        tail = [ln for ln in result.stderr.strip().splitlines() if ln.strip()]
        reason = tail[-1] if tail else f"exit {result.returncode} with no stderr"
    return {
        "error": reason,
        "returncode": result.returncode,
        "stderr_tail": result.stderr.strip().splitlines()[-6:],
    }


def _cost_multipliers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Wall-clock cost of each shape relative to the single-sphere control."""
    by_shape: dict[str, list[float]] = {}
    for row in rows:
        if row.get("reused_existing") and "dynamics_wall_s" not in row:
            continue
        by_shape.setdefault(row["shape"], []).append(float(row["dynamics_wall_s"]))
    means = {k: float(np.mean(v)) for k, v in by_shape.items() if v}
    baseline = means.get("sphere")
    return {
        "metric": "mean dynamics_wall_s per run across the mu x Crr grid",
        "baseline_shape": "sphere",
        "mean_dynamics_wall_s": means,
        "multiplier_vs_sphere": (
            {k: v / baseline for k, v in means.items()} if baseline else {}
        ),
        "sample_counts": {k: len(v) for k, v in by_shape.items()},
    }


def _shape_trend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the angle actually respond to shape?  A flat response is a model bug."""
    usable = [r for r in rows if r.get("measurement_pass")]
    by_shape: dict[str, list[float]] = {}
    for row in usable:
        by_shape.setdefault(row["shape"], []).append(float(row["repose_angle_deg"]))
    means = {k: float(np.mean(v)) for k, v in by_shape.items() if v}
    ordered = [means[s] for s in SHAPE_ORDER if s in means]
    monotonic = all(b > a for a, b in zip(ordered, ordered[1:])) if len(ordered) > 1 else False
    spread = (max(means.values()) - min(means.values())) if len(means) > 1 else 0.0
    # Paired comparison at identical (mu, Crr) is the honest test: it removes
    # the friction axis from the shape comparison.
    paired: dict[str, list[float]] = {}
    index = {(r["shape"], r["particle_mu"], r["rolling_friction"]): r for r in usable}
    for (shape, mu, crr), row in index.items():
        if shape == "sphere":
            continue
        control = index.get(("sphere", mu, crr))
        if control:
            paired.setdefault(shape, []).append(
                float(row["repose_angle_deg"]) - float(control["repose_angle_deg"])
            )
    return {
        "mean_angle_deg_by_shape": means,
        "shape_order_tested": [s for s in SHAPE_ORDER if s in means],
        "monotonic_increasing": bool(monotonic),
        "angle_spread_deg": float(spread),
        "paired_delta_vs_sphere_deg": {
            k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
            for k, v in paired.items()
        },
        "verdict": (
            "shape_responsive"
            if spread >= 1.0
            else "SHAPE_UNRESPONSIVE - the model does not distinguish shapes; do not calibrate on it"
        ),
    }


def fit_to_measured_repose(
    measured_deg: float,
    tol_deg: float = 1.0,
    sweep_path: Path | None = None,
) -> dict[str, Any]:
    """Inverse lookup: measured angle -> the parameter sets that produce it.

    This is the function to call the moment the PP repose angle is on paper.

    Parameters
    ----------
    measured_deg
        The angle measured on the physical heap.  It MUST have been measured the
        same way the simulation measures it (see the ``measurement_protocol``
        block in the return value), or the comparison is meaningless.
    tol_deg
        Half-width of the acceptance band.  Candidates whose simulated angle is
        within ``tol_deg`` are returned as ``candidates``; everything is also
        returned ranked in ``ranked`` so a near miss is still visible.
    sweep_path
        The sweep JSON written by :func:`run_sweep`.  Defaults to the standard
        location.

    Returns a dict with ``candidates`` (inside tolerance, best first),
    ``ranked`` (all runs by distance), ``bracketed`` (whether the measured value
    lies inside the simulated range at all, per shape) and ``suggested_refinement``
    (for each shape, the mu/Crr interval to sweep next, obtained by linear
    interpolation between the two nearest grid points).
    """
    path = Path(sweep_path) if sweep_path else OUT_DIR / "repose_sweep.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; run --sweep first so there is a table to invert"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    usable = [r for r in document["runs"] if r.get("measurement_pass")]
    if not usable:
        raise ValueError(f"{path} contains no run whose measurement gate passed")

    target = float(measured_deg)
    ranked = sorted(
        (
            {
                "shape": r["shape"],
                "aspect": r["aspect"],
                "particle_mu": r["particle_mu"],
                "rolling_friction": r["rolling_friction"],
                "repose_angle_deg": r["repose_angle_deg"],
                "repose_angle_secondary_deg": r["repose_angle_secondary_deg"],
                "abs_error_deg": abs(float(r["repose_angle_deg"]) - target),
                "fit_r_squared": r["fit_r_squared"],
                "npz": r["npz"],
            }
            for r in usable
        ),
        key=lambda r: r["abs_error_deg"],
    )
    candidates = [r for r in ranked if r["abs_error_deg"] <= float(tol_deg)]

    bracketed: dict[str, Any] = {}
    refinement: dict[str, Any] = {}
    for shape in SHAPE_ORDER:
        rows = [r for r in usable if r["shape"] == shape]
        if not rows:
            continue
        angles = [float(r["repose_angle_deg"]) for r in rows]
        lo, hi = min(angles), max(angles)
        bracketed[shape] = {
            "simulated_range_deg": [lo, hi],
            "measured_inside_range": bool(lo <= target <= hi),
        }
        if not (lo <= target <= hi):
            refinement[shape] = {
                "action": "extend the grid",
                "direction": "increase mu and/or Crr" if target > hi else "decrease mu and/or Crr",
                "reason": f"measured {target:.2f} deg lies outside the simulated [{lo:.2f}, {hi:.2f}]",
            }
            continue
        below = max((r for r in rows if float(r["repose_angle_deg"]) <= target),
                    key=lambda r: float(r["repose_angle_deg"]))
        above = min((r for r in rows if float(r["repose_angle_deg"]) >= target),
                    key=lambda r: float(r["repose_angle_deg"]))
        a_lo, a_hi = float(below["repose_angle_deg"]), float(above["repose_angle_deg"])
        t = 0.0 if a_hi == a_lo else (target - a_lo) / (a_hi - a_lo)
        refinement[shape] = {
            "action": "interpolate then verify",
            "bracket_low": {
                "particle_mu": below["particle_mu"],
                "rolling_friction": below["rolling_friction"],
                "angle_deg": a_lo,
            },
            "bracket_high": {
                "particle_mu": above["particle_mu"],
                "rolling_friction": above["rolling_friction"],
                "angle_deg": a_hi,
            },
            "interpolated_guess": {
                "particle_mu": float(below["particle_mu"]) * (1 - t)
                + float(above["particle_mu"]) * t,
                "rolling_friction": float(below["rolling_friction"]) * (1 - t)
                + float(above["rolling_friction"]) * t,
            },
            "warning": (
                "linear interpolation between two grid points is a starting guess only; "
                "re-run the simulation at the interpolated parameters and re-measure"
            ),
        }

    return {
        "measured_repose_angle_deg": target,
        "tolerance_deg": float(tol_deg),
        "sweep": str(path),
        "sweep_definition": document.get("repose_definition"),
        "measurement_protocol": (
            "The physical measurement must use the SAME definition as the simulation: pour the "
            "pellets from a tube onto a flat plate, photograph the heap side-on, and fit a "
            "straight line to the flank between 80% and 20% of the heap height, excluding the "
            "flat top and the toe. An apex-to-base angle is NOT interchangeable with this; if "
            "that is what was measured, compare against repose_angle_secondary_deg instead."
        ),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "ranked": ranked,
        "bracketed_by_shape": bracketed,
        "suggested_refinement": refinement,
        "non_claims": NON_CLAIMS,
    }


# ---------------------------------------------------------------------------
# 8. Validation entry points (the GATES.md CHECK commands call these)
# ---------------------------------------------------------------------------


def describe_templates(config: ReposeConfig) -> None:
    """Print the three templates and verify their exact mass properties."""
    pellet = config.pellet()
    print(f"pellet (MEASURE placeholders): D={pellet.pellet_dia_mm} mm  L={pellet.pellet_len_mm} mm  "
          f"rho={pellet.particle_density_kg_m3} kg/m^3")
    print(f"  cylinder volume = {pellet.cylinder_volume_m3()*1e9:.4f} mm^3   "
          f"mass = {pellet.mass_kg()*1e6:.6f} mg   nominal L/D = {pellet.nominal_aspect():.4f}")
    for name in SHAPE_ORDER:
        t = build_template(pellet, name, size_match=config.size_match)
        lo, hi = aspect_bounds(t.n_spheres)
        print(
            f"  {name:7s} k={t.n_spheres} aspect={t.aspect:.4f} (legal [{lo:.3f},{hi:.3f}])  "
            f"R={t.sphere_radius_m*1e3:.4f}mm s={t.spacing_m*1e3:.4f}mm  "
            f"LxW={t.union_length_m*1e3:.3f}x{t.union_width_m*1e3:.3f}mm"
        )
        print(
            f"          mass={t.mass_kg*1e6:.6f}mg  "
            f"MOI=({t.moi_kg_m2[0]:.6e}, {t.moi_kg_m2[1]:.6e}, {t.moi_kg_m2[2]:.6e}) kg m^2  "
            f"dV/V={t.volume_mismatch_fraction()*100:+.3f}%"
        )
    _verify_moment_identities()
    print("TEMPLATE_CONTRACT_OK")


def _verify_moment_identities() -> None:
    """Check the exact formulas against cases with independently known answers."""
    r = 0.0031
    # 1. A single sphere must reproduce V = 4/3 pi r^3 and I = 2/5 m r^2.
    single = union_geometric_moments(r, 0.0, 1)
    v_expect = 4.0 / 3.0 * math.pi * r**3
    i_expect = 0.4 * v_expect * r * r
    for key, expect in (("volume", v_expect), ("ixx", i_expect), ("iyy", i_expect)):
        if abs(single[key] - expect) > 1e-12 * max(abs(expect), 1e-30):
            raise AssertionError(f"sphere identity failed for {key}")
    # 2. Two spheres at s -> 2R (just touching) must approach 2 separate spheres.
    touching = union_geometric_moments(r, 2.0 * r * (1 - 1e-9), 2)
    if abs(touching["volume"] - 2.0 * v_expect) > 1e-6 * v_expect:
        raise AssertionError("touching-pair volume does not approach two spheres")
    iyy_expect = 2.0 * (i_expect + v_expect * r * r)
    if abs(touching["iyy"] - iyy_expect) > 1e-5 * iyy_expect:
        raise AssertionError("touching-pair transverse inertia does not match parallel axis")
    # 3. Independent Monte-Carlo cross-check of a genuinely overlapping clump.
    #    Only a cross-check: the analytic value stays authoritative.
    rng = np.random.default_rng(20260901)
    n_spheres, spacing = 3, 1.2 * r
    exact = union_geometric_moments(r, spacing, n_spheres)
    centres = np.asarray([(i - 1) * spacing for i in range(n_spheres)])
    half_x = abs(centres[0]) + r
    samples = rng.uniform(
        [-half_x, -r, -r], [half_x, r, r], size=(4_000_000, 3)
    )
    inside = np.zeros(samples.shape[0], dtype=bool)
    for cx in centres:
        d = samples - np.asarray([cx, 0.0, 0.0])
        inside |= np.einsum("ij,ij->i", d, d) <= r * r
    box_volume = (2 * half_x) * (2 * r) * (2 * r)
    mc_volume = box_volume * inside.mean()
    pts = samples[inside]
    mc_ixx = box_volume / samples.shape[0] * float(np.sum(pts[:, 1] ** 2 + pts[:, 2] ** 2))
    mc_iyy = box_volume / samples.shape[0] * float(np.sum(pts[:, 0] ** 2 + pts[:, 2] ** 2))
    for label, mc, ex in (
        ("volume", mc_volume, exact["volume"]),
        ("ixx", mc_ixx, exact["ixx"]),
        ("iyy", mc_iyy, exact["iyy"]),
    ):
        rel = abs(mc - ex) / ex
        if rel > 5.0e-3:
            raise AssertionError(
                f"Monte-Carlo cross-check disagrees on {label}: exact={ex:.6e} mc={mc:.6e} rel={rel:.2e}"
            )
        print(f"    cross-check {label}: exact={ex:.6e} mc={mc:.6e} rel_diff={rel:.2e}")


def load_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        if keys != sorted(NPZ_KEYS):
            raise ValueError(f"{path}: keys {keys} != {sorted(NPZ_KEYS)}")
        arrays = {key: np.array(archive[key], copy=True) for key in keys}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    return arrays, metadata


def validate_runs(paths: Iterable[Path]) -> None:
    checked = 0
    for path in paths:
        arrays, metadata = load_artifact(path)
        n = int(metadata["config"]["n_particles"])
        k = int(metadata["template"]["n_spheres"])
        shapes = {
            "clump_positions_m": (n, 3),
            "clump_quaternions_xyzw": (n, 4),
            "velocities_m_s": (n, 3),
            "initial_positions_m": (n, 3),
            "initial_quaternions_xyzw": (n, 4),
            "sphere_offsets_m": (k, 3),
            "sphere_radii_m": (k,),
            "particle_ids": (n,),
            "box_bounds_m": (3, 2),
        }
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"{path}: {key} shape {arrays[key].shape} != {shape}")
        for key in shapes:
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"{path}: {key} contains NaN/Inf")
        if arrays["settle_history"].shape[1] != len(HISTORY_COLUMNS):
            raise ValueError(f"{path}: settle_history has wrong column count")
        if not metadata["settling_gate"]["settled"]:
            raise ValueError(f"{path}: settlement gate is false")
        if not metadata["seed_gate"]["no_seed_penetration"]:
            raise ValueError(f"{path}: seed penetration gate is false")
        if not metadata["penetration_gate"]["pass"]:
            raise ValueError(f"{path}: inter-clump penetration gate is false")
        if not metadata["repose"]["measurement_pass"]:
            raise ValueError(
                f"{path}: repose measurement gate is false: {metadata['repose']['checks']}"
            )
        # Recompute the angle from the stored arrays: the metadata must not be
        # the only place the number exists.
        config = ReposeConfig(**metadata["config"])
        spheres = sphere_centres_world(
            arrays["clump_positions_m"],
            arrays["clump_quaternions_xyzw"],
            arrays["sphere_offsets_m"],
        )
        recomputed = measure_repose_angle(
            spheres,
            float(arrays["sphere_radii_m"][0]),
            config,
            metadata["derived_geometry"],
        )
        stored = float(metadata["repose"]["repose_angle_deg"])
        if abs(recomputed["repose_angle_deg"] - stored) > 1.0e-6:
            raise ValueError(
                f"{path}: recomputed angle {recomputed['repose_angle_deg']:.9f} != "
                f"stored {stored:.9f}"
            )
        print(
            f"VALID {path.name}: shape={metadata['config']['shape']} "
            f"mu={metadata['config']['particle_mu']} Crr={metadata['config']['rolling_friction']} "
            f"angle={stored:.3f}deg (recomputed match) r2={metadata['repose']['fit_r_squared']:.4f}"
        )
        checked += 1
    print(f"RUN_VALIDATION_OK count={checked}")


def validate_sweep(path: Path, *, min_cells: int) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["artifact"] != f"{ARTIFACT}_SWEEP":
        raise ValueError(f"{path}: not a sweep document")
    completed = int(document["cells_completed"])
    if completed < min_cells:
        raise ValueError(f"{path}: only {completed} cells completed, need >= {min_cells}")
    trend = document["shape_trend"]
    if trend["verdict"] != "shape_responsive":
        raise ValueError(
            f"{path}: SHAPE_UNRESPONSIVE - angle spread {trend['angle_spread_deg']:.3f} deg "
            "across shapes. The shape model is not doing anything; this is a model failure, "
            "not a pass."
        )
    cost = document["cost_multipliers"]["multiplier_vs_sphere"]
    if "sphere" not in cost:
        raise ValueError(f"{path}: no sphere baseline, cost multipliers are meaningless")
    order = trend["shape_order_tested"]
    means = trend["mean_angle_deg_by_shape"]
    print(
        "SWEEP_VALIDATION_OK cells=%d shapes=%s angles_deg=%s spread=%.3f monotonic=%s"
        % (
            completed,
            order,
            {k: round(means[k], 3) for k in order},
            trend["angle_spread_deg"],
            trend["monotonic_increasing"],
        )
    )
    print("  cost multiplier vs sphere: " + ", ".join(f"{k}={v:.3f}x" for k, v in cost.items()))


def validate_fit(sweep_path: Path, measured_deg: float, tol_deg: float) -> None:
    result = fit_to_measured_repose(measured_deg, tol_deg, sweep_path)
    for row in result["candidates"][:6]:
        print(
            f"  candidate shape={row['shape']} mu={row['particle_mu']} "
            f"Crr={row['rolling_friction']} angle={row['repose_angle_deg']:.3f}deg "
            f"err={row['abs_error_deg']:.3f}deg"
        )
    if not result["candidates"]:
        best = result["ranked"][0]
        print(
            f"  no candidate within {tol_deg} deg; nearest is shape={best['shape']} "
            f"mu={best['particle_mu']} Crr={best['rolling_friction']} "
            f"angle={best['repose_angle_deg']:.3f}deg err={best['abs_error_deg']:.3f}deg"
        )
    for shape, bracket in result["bracketed_by_shape"].items():
        lo, hi = bracket["simulated_range_deg"]
        inside = "BRACKETS it" if bracket["measured_inside_range"] else "cannot reach it"
        print(f"  {shape:7s} spans {lo:6.2f}..{hi:6.2f} deg - {inside}")
    for shape, plan in result["suggested_refinement"].items():
        if plan["action"] == "extend the grid":
            print(f"  {shape:7s} next: {plan['action']} ({plan['direction']})")
        else:
            guess = plan["interpolated_guess"]
            print(
                f"  {shape:7s} next: simulate mu={guess['particle_mu']:.4f} "
                f"Crr={guess['rolling_friction']:.4f} then re-measure "
                f"(interpolated between {plan['bracket_low']['angle_deg']:.2f} and "
                f"{plan['bracket_high']['angle_deg']:.2f} deg)"
            )
    print(
        f"FIT_OK measured={measured_deg:.2f}deg tol={tol_deg:.2f}deg "
        f"candidates={result['n_candidates']}"
    )


# ---------------------------------------------------------------------------
# 9. Rerun export (D341 observability contract; isaaclab interpreter)
# ---------------------------------------------------------------------------


def _rerun_expected_contract() -> tuple[set[str], dict[str, list[str]], list[str]]:
    entities = {
        "/metadata/run",
        "/coordinate_frames/world_m",
        "/geometry/container",
        "/metadata/meshes/geometry__container",
        "/geometry/heap/spheres",
        "/geometry/heap/settled",
        "/geometry/contacts/points",
        "/geometry/contacts/forces",
        "/metrics/max_speed_m_s",
        "/metrics/p99_speed_m_s",
        "/metrics/rms_speed_m_s",
        "/metrics/kinetic_energy_j",
        "/metrics/num_contacts",
        "/metrics/heap_top_z_m",
        "/metrics/max_radial_extent_m",
        "/metrics/repose_angle_deg",
        "/metrics/fit_r_squared",
        "/events/repose",
    }
    components = {
        "/metadata/run": ["TextDocument:text"],
        "/geometry/container": [
            "Mesh3D:albedo_factor",
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        ],
        "/geometry/heap/spheres": ["Points3D:colors", "Points3D:positions", "Points3D:radii"],
        "/geometry/heap/settled": ["Points3D:colors", "Points3D:positions", "Points3D:radii"],
        "/geometry/contacts/points": ["Points3D:colors", "Points3D:positions", "Points3D:radii"],
        "/geometry/contacts/forces": ["Arrows3D:colors", "Arrows3D:origins", "Arrows3D:vectors"],
        "/metrics/max_speed_m_s": ["Scalars:scalars"],
        "/metrics/p99_speed_m_s": ["Scalars:scalars"],
        "/metrics/rms_speed_m_s": ["Scalars:scalars"],
        "/metrics/kinetic_energy_j": ["Scalars:scalars"],
        "/metrics/num_contacts": ["Scalars:scalars"],
        "/metrics/heap_top_z_m": ["Scalars:scalars"],
        "/metrics/max_radial_extent_m": ["Scalars:scalars"],
        "/metrics/repose_angle_deg": ["Scalars:scalars"],
        "/metrics/fit_r_squared": ["Scalars:scalars"],
        "/events/repose": ["TextLog:level", "TextLog:text"],
    }
    timelines = ["blueprint", "log_time", "sample", "sim_time_s"]
    return entities, components, timelines


def _build_repose_blueprint(mode: str) -> Any:
    if mode != "pellet_repose":
        raise ValueError(f"unsupported blueprint mode: {mode!r}")
    import rerun.blueprint as rrb

    # The container mesh is deliberately excluded from the two heap views: the
    # simulation box is ~900 mm tall because the un-poured batches are parked
    # above the heap, so including it makes the camera frame the parking stack
    # and shrink the 30 mm heap to a speck.  It keeps its own view for reference.
    #
    # The metrics are split because num_contacts runs to ~10^5 while the speeds
    # are ~10^-3 m/s; on one shared axis every speed trace is pinned flat at zero.
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/geometry/heap/settled"],
                    name="settled heap - the measured subject",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/geometry/heap/spheres"],
                    name="pour + settle timeline",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/geometry/contacts/points", "/geometry/contacts/forces"],
                    name="contact points and force chains",
                ),
                column_shares=[0.34, 0.33, 0.33],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin="/metrics",
                    contents=[
                        "/metrics/max_speed_m_s",
                        "/metrics/p99_speed_m_s",
                        "/metrics/rms_speed_m_s",
                        "/metrics/heap_top_z_m",
                        "/metrics/max_radial_extent_m",
                    ],
                    name="settling: speeds and heap extent",
                ),
                rrb.TimeSeriesView(
                    origin="/metrics",
                    contents=[
                        "/metrics/num_contacts",
                        "/metrics/kinetic_energy_j",
                        "/metrics/repose_angle_deg",
                        "/metrics/fit_r_squared",
                    ],
                    name="contacts, energy, decision scalars",
                ),
                rrb.TextLogView(origin="/events", contents="/events/**", name="verdict"),
                column_shares=[0.36, 0.36, 0.28],
            ),
            rrb.Spatial3DView(
                origin="/",
                contents=["/geometry/container", "/geometry/heap/settled"],
                name="full simulation domain (incl. parking headroom)",
            ),
            row_shares=[0.50, 0.28, 0.22],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _box_mesh(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    zmin, zmax = bounds[2]
    vertices = np.asarray(
        [
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
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


def export_rerun(npz_path: Path) -> None:
    arrays, metadata = load_artifact(npz_path)
    if arrays["state_history"].shape[0] < 2:
        raise RuntimeError(
            "D341 requires the full executed settling timeline; this artifact was written "
            "with store_state_history=False. Re-run with --store-state-history."
        )
    import rerun as rr
    from roarm_rl.rerun_contract import RERUN_CONTRACT_VERSION, validate_rerun_artifact
    import roarm_rl.viz_debug as viz_debug

    if str(rr.__version__) != "0.34.1" or RERUN_CONTRACT_VERSION != "0.34.1":
        raise RuntimeError(
            f"Rerun pin mismatch (D326): sdk={rr.__version__}, contract={RERUN_CONTRACT_VERSION}"
        )
    interpreter_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = interpreter_bin + os.pathsep + os.environ.get("PATH", "")

    stem = npz_path.with_suffix("")
    rrd_path = stem.with_suffix(".rrd")
    rbl_path = stem.with_suffix(".rbl")
    screenshot_path = stem.with_name(f"{stem.name}_inspection.png")
    validation_path = stem.with_name(f"{stem.name}_rerun_validation.json")
    for path in (rrd_path, rbl_path, screenshot_path, validation_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    offsets = arrays["sphere_offsets_m"]
    radius = float(arrays["sphere_radii_m"][0])
    history = arrays["settle_history"]
    state = arrays["state_history"]
    n_samples = state.shape[0]
    n_particles = state.shape[1]
    k = offsets.shape[0]

    # Only material that has actually been poured belongs in the heap view.  The
    # batches still parked overhead are staging, not the decision subject, and
    # logging them makes the camera frame a 900 mm parking stack instead of a
    # 30 mm heap.
    releases = metadata["release_protocol"]["releases"]
    release_times = np.asarray([float(r["sim_time_s"]) for r in releases])
    release_counts = np.cumsum([int(r["n_particles"]) for r in releases])

    def poured_at(sim_time: float) -> int:
        released = int(np.count_nonzero(release_times <= sim_time + 1e-12))
        return int(release_counts[released - 1]) if released else 0

    def height_colors(centres: np.ndarray) -> np.ndarray:
        z = centres[:, 2]
        span = max(float(z.max() - z.min()), 1e-9)
        t = (z - z.min()) / span
        return np.column_stack(
            [
                (60 + 195 * t).astype(np.uint8),
                (90 + 120 * (1.0 - np.abs(2 * t - 1))).astype(np.uint8),
                (220 - 170 * t).astype(np.uint8),
                np.full(centres.shape[0], 235, dtype=np.uint8),
            ]
        )

    point_rows: list[dict[str, Any]] = []
    for sample in range(n_samples):
        sim_time = float(history[sample, 0])
        poured = max(poured_at(sim_time), 1)
        centres = sphere_centres_world(
            state[sample, :poured, 0:3], state[sample, :poured, 3:7], offsets
        )
        point_rows.append(
            {
                "entity_path": "geometry/heap/spheres",
                "positions_m": centres,
                "radii": np.full(centres.shape[0], radius, dtype=np.float32),
                "colors": height_colors(centres),
                "coordinate_frame": "world_m",
                "sequence": {"sample": sample},
                "duration": {"sim_time_s": sim_time},
            }
        )

    # Static copies of the settled heap and its contacts.  The headless renderer
    # parks the time cursor wherever it likes; a static entity guarantees the
    # screenshot shows the state the verdict was actually read from.
    settled = sphere_centres_world(
        arrays["clump_positions_m"], arrays["clump_quaternions_xyzw"], offsets
    )
    point_rows.append(
        {
            "entity_path": "geometry/heap/settled",
            "positions_m": settled,
            "radii": np.full(settled.shape[0], radius, dtype=np.float32),
            "colors": height_colors(settled),
            "coordinate_frame": "world_m",
            "static": True,
        }
    )

    contacts = arrays["contact_points_m"]
    forces = arrays["contact_forces_n"]
    if contacts.shape[0] == 0:
        raise RuntimeError("source NPZ has no contact points; D341 needs them for a physics verdict")
    magnitude = np.linalg.norm(forces, axis=1)
    scale = magnitude / max(float(magnitude.max()), 1e-12)
    contact_colors = np.column_stack(
        [
            (255 * scale).astype(np.uint8),
            (60 + 60 * (1 - scale)).astype(np.uint8),
            (40 + 180 * (1 - scale)).astype(np.uint8),
            np.full(contacts.shape[0], 220, dtype=np.uint8),
        ]
    )
    point_rows.append(
        {
            "entity_path": "geometry/contacts/points",
            "positions_m": contacts,
            "radii": np.full(contacts.shape[0], 0.25 * radius, dtype=np.float32),
            "colors": contact_colors,
            "coordinate_frame": "world_m",
            "static": True,
        }
    )
    # Force arrows: only the load-bearing contacts.  DEME reports every potential
    # pair, and the zero-force majority would bury the force chains.  The kept
    # count is recorded so this is a stated filter, not a silent truncation.
    bearing = magnitude > 0.0
    arrow_scale_m = 4.0 * radius / max(float(magnitude.max()), 1e-12)
    arrow_rows = [
        {
            "entity_path": "geometry/contacts/forces",
            "origins_m": contacts[bearing],
            "vectors_m": forces[bearing] * arrow_scale_m,
            "colors": contact_colors[bearing],
            "coordinate_frame": "world_m",
            "static": True,
        }
    ]

    scalar_rows: list[dict[str, Any]] = []
    for sample in range(history.shape[0]):
        for column, name in enumerate(HISTORY_COLUMNS[1:], start=1):
            scalar_rows.append(
                {
                    "entity_path": f"metrics/{name}",
                    "value": float(history[sample, column]),
                    "sequence": {"sample": sample},
                    "duration": {"sim_time_s": float(history[sample, 0])},
                }
            )
    repose = metadata["repose"]
    for name, value in (
        ("repose_angle_deg", repose["repose_angle_deg"]),
        ("fit_r_squared", repose["fit_r_squared"]),
    ):
        # A decision scalar is constant in time but must exist on the timeline
        # at both ends so the plot renders and the value is readable.
        for sample in (0, history.shape[0] - 1):
            scalar_rows.append(
                {
                    "entity_path": f"metrics/{name}",
                    "value": float(value),
                    "sequence": {"sample": sample},
                    "duration": {"sim_time_s": float(history[sample, 0])},
                }
            )

    box_vertices, box_triangles = _box_mesh(arrays["box_bounds_m"])
    config = metadata["config"]
    events = [
        {
            "entity_path": "events/repose",
            "text": (
                f"{config['shape']} aspect={metadata['template']['aspect']:.3f} "
                f"mu={config['particle_mu']} Crr={config['rolling_friction']} -> "
                f"repose {repose['repose_angle_deg']:.3f} deg "
                f"({repose['primary_definition']}, r2={repose['fit_r_squared']:.4f}); "
                f"secondary {repose['repose_angle_secondary_deg']:.3f} deg; "
                f"MEASURE-placeholder material, not a measured pellet property"
            ),
            "level": "INFO" if repose["measurement_pass"] else "WARN",
            "sequence": {"sample": history.shape[0] - 1},
            "duration": {"sim_time_s": float(history[-1, 0])},
        }
    ]

    original = viz_debug.build_rerun_blueprint
    try:
        viz_debug.build_rerun_blueprint = _build_repose_blueprint
        status = viz_debug.log_rerun(
            rrd_path,
            coordinate_frames=[
                {
                    "frame": "world_m",
                    "parent_frame": "tf#/",
                    "entity_path": "coordinate_frames/world_m",
                }
            ],
            meshes=[
                {
                    "entity_path": "geometry/container",
                    "vertices_m": box_vertices,
                    "triangles": box_triangles,
                    "color_rgba": [80, 145, 210, 30],
                    "coordinate_frame": "world_m",
                    "static": True,
                }
            ],
            points=point_rows,
            arrows=arrow_rows,
            scalar_trace=scalar_rows,
            events=events,
            recording_metadata={
                "artifact": f"{ARTIFACT}_RERUN",
                "source_npz": str(npz_path),
                "source_npz_sha256": _sha256(npz_path),
                "n_particles": n_particles,
                "n_spheres_per_clump": k,
                "samples_logged": n_samples,
                "coordinate_frame": metadata["coordinate_frame"],
                "scientific_authority": "source NPZ float64 arrays",
                "non_claims": NON_CLAIMS,
            },
            recording_id=f"pellet_repose_{config['shape']}_seed{config['seed']}",
            blueprint_path=rbl_path,
            blueprint_mode="pellet_repose",
            live_viewer=False,
            app_id="roarm_pellet_repose",
        )
    finally:
        viz_debug.build_rerun_blueprint = original
    if not status.get("ok", False):
        raise RuntimeError(f"Rerun logging contract failed: {status}")

    entities, components, timelines = _rerun_expected_contract()
    validation = validate_rerun_artifact(
        rrd_path,
        expected_entity_paths=sorted(entities),
        expected_timeline_names=timelines,
        exact_entity_paths=sorted(entities),
        exact_timeline_names=timelines,
        expected_entity_components=components,
        blueprint_path=rbl_path,
        screenshot_path=screenshot_path,
        cli_path=Path(interpreter_bin) / "rerun",
        expected_version="0.34.1",
        timeout_s=180.0,
    )
    validation["source_npz"] = str(npz_path)
    validation["source_npz_sha256"] = _sha256(npz_path)
    validation["samples_logged"] = n_samples
    validation["log_status_summary"] = {
        key: status.get(key)
        for key in (
            "ok", "bytes", "rerun_sdk_version", "sink_attached_before_logging",
            "sink_finalized", "flush_ok", "blueprint_status",
        )
    }
    _write_json(validation_path, validation)
    if not validation.get("pass", False):
        raise RuntimeError(f"Rerun exact contract failed; see {validation_path}")
    print(
        f"RERUN_EXPORT_OK rrd={rrd_path} rbl={rbl_path} screenshot={screenshot_path} "
        f"validation={validation_path} samples={n_samples}"
    )


def validate_rerun_contract(validation_path: Path, inspection_path: Path) -> None:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    required = {
        "pass": validation.get("pass") is True,
        "version": validation.get("version", {}).get("expected_version_match") is True,
        "footer": validation.get("footer_manifest_present") is True,
        "entity": validation.get("entity_path_contract", {}).get("pass") is True,
        "timeline": validation.get("timeline_contract", {}).get("pass") is True,
        "components": validation.get("component_contract", {}).get("pass") is True,
        "blueprint": validation.get("blueprint_verify", {}).get("ok") is True,
        "screenshot": validation.get("headless_render", {}).get("ok") is True,
    }
    failed = [name for name, ok in required.items() if not ok]
    if failed:
        raise ValueError(f"Rerun validation failures: {failed}")
    if int(validation.get("samples_logged", 0)) < 2:
        raise ValueError("RRD does not contain a multi-sample settling timeline (D341)")
    if inspection.get("visual_inspection_complete") is not True:
        raise ValueError("manual visual inspection is not complete")
    screenshot_path = Path(validation["headless_render"]["path"])
    if inspection.get("screenshot_path") != str(screenshot_path):
        raise ValueError("inspection screenshot path does not match validation")
    if inspection.get("screenshot_sha256") != _sha256(screenshot_path):
        raise ValueError("inspection screenshot hash mismatch")
    observations = inspection.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("inspection observations are empty")
    print("RERUN_OBSERVABILITY_OK visual_inspection=complete")


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe-format", action="store_true")
    mode.add_argument("--describe-templates", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--sweep", action="store_true")
    mode.add_argument("--validate-runs", nargs="+", type=Path)
    mode.add_argument("--validate-sweep", type=Path)
    mode.add_argument("--fit-repose", type=float, metavar="MEASURED_DEG")
    mode.add_argument("--measured-pellet", nargs=3, type=float,
                      metavar=("DIA_MM", "LEN_MM", "DENSITY_KG_M3"))
    mode.add_argument("--export-rerun", type=Path, metavar="NPZ")
    mode.add_argument("--validate-rerun-contract", nargs=2, type=Path,
                      metavar=("VALIDATION_JSON", "INSPECTION_JSON"))

    parser.add_argument("--output", type=Path)
    parser.add_argument("--sweep-output", type=Path)
    parser.add_argument("--min-cells", type=int, default=27)
    parser.add_argument("--tol-deg", type=float, default=1.0)
    parser.add_argument("--shape", choices=SHAPE_ORDER, default="sphere")
    parser.add_argument("--shapes", nargs="+", choices=SHAPE_ORDER, default=list(SHAPE_ORDER))
    parser.add_argument("--aspect", type=float)
    parser.add_argument("--size-match", choices=("volume", "diameter"), default="volume")
    parser.add_argument("--n-particles", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--pellet-dia-mm", type=float, default=3.5)
    parser.add_argument("--pellet-len-mm", type=float, default=4.0)
    parser.add_argument("--particle-density-kg-m3", type=float, default=950.0)
    parser.add_argument("--mu", type=float, default=0.50)
    parser.add_argument("--crr", type=float, default=0.05)
    parser.add_argument("--restitution", type=float, default=0.30)
    parser.add_argument("--young-modulus-pa", type=float, default=5.0e6)
    parser.add_argument("--mu-grid", nargs="+", type=float, default=list(DEFAULT_MU_GRID))
    parser.add_argument("--crr-grid", nargs="+", type=float, default=list(DEFAULT_CRR_GRID))
    parser.add_argument("--dt-s", type=float, default=4.0e-5)
    parser.add_argument("--n-release-batches", type=int, default=18)
    parser.add_argument("--cd-update-freq", type=int, default=6)
    parser.add_argument("--cd-max-update-freq", type=int, default=6)
    parser.add_argument("--advance-chunk-s", type=float, default=0.004)
    parser.add_argument("--chunks-per-sample", type=int, default=5)
    parser.add_argument("--min-sim-time-s", type=float, default=0.30)
    parser.add_argument("--max-sim-time-s", type=float, default=7.00)
    parser.add_argument("--profile-cell-mm", type=float, default=2.0)
    parser.add_argument("--column-radius-fraction", type=float, default=0.30)
    parser.add_argument("--store-state-history", dest="store_state_history",
                        action="store_true", default=True)
    parser.add_argument("--no-state-history", dest="store_state_history", action="store_false")
    return parser


def config_from_args(args: argparse.Namespace) -> ReposeConfig:
    return ReposeConfig(
        shape=args.shape,
        aspect=args.aspect,
        size_match=args.size_match,
        n_particles=args.n_particles,
        seed=args.seed,
        pellet_dia_mm=args.pellet_dia_mm,
        pellet_len_mm=args.pellet_len_mm,
        particle_density_kg_m3=args.particle_density_kg_m3,
        particle_mu=args.mu,
        wall_mu=args.mu,
        rolling_friction=args.crr,
        restitution=args.restitution,
        young_modulus_pa=args.young_modulus_pa,
        dt_s=args.dt_s,
        cd_update_freq=args.cd_update_freq,
        cd_max_update_freq=args.cd_max_update_freq,
        advance_chunk_s=args.advance_chunk_s,
        chunks_per_sample=args.chunks_per_sample,
        min_sim_time_s=args.min_sim_time_s,
        max_sim_time_s=args.max_sim_time_s,
        profile_cell_mm=args.profile_cell_mm,
        column_radius_fraction=args.column_radius_fraction,
        n_release_batches=args.n_release_batches,
        store_state_history=args.store_state_history,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.describe_format:
        print(__doc__)
        print("NPZ keys: " + ", ".join(sorted(NPZ_KEYS)))
        print("settle_history columns: " + ", ".join(HISTORY_COLUMNS))
        print("radial_profile columns: " + ", ".join(RADIAL_PROFILE_COLUMNS))
        print(f"repose definition (primary): {REPOSE_PRIMARY_DEFINITION}")
        print(f"  {REPOSE_PRIMARY_DEFINITION_TEXT}")
        for item in NON_CLAIMS:
            print("NON_CLAIM " + item)
        print("FORMAT_CONTRACT_OK")
        return
    if args.describe_templates:
        describe_templates(config_from_args(args))
        return
    if args.measured_pellet:
        template = template_for_measured_pellet(*args.measured_pellet, size_match=args.size_match)
        print(json.dumps(asdict(template), indent=2, default=_json_default))
        print(
            f"MEASURED_TEMPLATE_OK shape={template.name} aspect={template.aspect:.4f} "
            f"R={template.sphere_radius_m*1e3:.4f}mm mass={template.mass_kg*1e6:.6f}mg"
        )
        return
    if args.validate_runs:
        validate_runs(args.validate_runs)
        return
    if args.validate_sweep:
        validate_sweep(args.validate_sweep, min_cells=args.min_cells)
        return
    if args.fit_repose is not None:
        validate_fit(
            args.sweep_output or OUT_DIR / "repose_sweep.json",
            args.fit_repose,
            args.tol_deg,
        )
        return
    if args.export_rerun:
        export_rerun(args.export_rerun)
        return
    if args.validate_rerun_contract:
        validate_rerun_contract(*args.validate_rerun_contract)
        return
    config = config_from_args(args)
    if args.sweep:
        run_sweep(
            config,
            args.shapes,
            args.mu_grid,
            args.crr_grid,
            args.sweep_output or OUT_DIR / "repose_sweep.json",
        )
        return
    if args.run:
        run_repose(config, args.output or default_output_path(config))
        return
    build_parser().print_help()


if __name__ == "__main__":
    main()
