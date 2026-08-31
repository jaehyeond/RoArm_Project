"""Common heightmap (높이 지도) module — the shared observation input for the
scoop / bulk-material research direction.

Two producers, ONE output contract
----------------------------------
  (A) particle centres (DEM / sim pellets)  -> :func:`heightmap_from_particles`
  (B) depth image (Azure Kinect DK, real)   -> :func:`heightmap_from_depth`

Both return a :class:`Heightmap` with an identical on-disk layout, so the s3
prediction model sees exactly one input format regardless of the source.

OUTPUT CONTRACT (spec_version = "roarm-heightmap-v1")
-----------------------------------------------------
frame
    ``roarm_base`` — the RoArm base frame used everywhere else in this repo
    (Isaac stage metersPerUnit = 1.0, ``sim_scripts/kinect_calib.yaml``
    extrinsics map camera -> this frame via ``p_base = R @ p_cam + t``).
    z is up.
axes / indexing
    ``height[row, col]``.  ``col`` indexes world +x, ``row`` indexes world +y.
    Cell centre:  ``x = origin_x + (col + 0.5) * cell_m``
                  ``y = origin_y + (row + 0.5) * cell_m``
    ``origin_xy_m`` is the LOWER-LEFT CORNER of cell ``[0, 0]`` (not its centre).
    This is the same ``[row=y, col=x]`` convention as the frozen yard_track
    probes (``p26..p31``, ``region_cells()``), so old evidence stays readable.
units / dtype
    ``height``  float32, **metres**, = surface z in the base frame minus
                ``z_datum_m``.  Internals are computed in float64 and cast once
                at the end; float32 quantisation at h ~ 0.05 m is ~4e-9 m and is
                therefore not a measurable error source.
    ``valid``   bool_,  True where at least one surface sample landed in the cell.
    ``counts``  int32,  number of samples that landed in the cell (confidence).
empty cells
    ``valid[r, c] = False`` **and** ``height[r, c] = header["empty_cell"]
    ["height_fill_m"]`` (default 0.0 = the support plane).  ``valid`` is the
    authority; the fill value is only there so the array is NaN-free and can be
    fed to a network directly.
    Path (A) observes every cell (a downward ray always terminates, at worst on
    the floor), so ``valid`` is all-True there; ``counts == 0`` on that path
    means "no sphere over this cell, height = floor_z_m", which is a real
    measurement, not a hole.  Path (B) marks camera shadow / out-of-FOV /
    dropped-return cells invalid, and there ``counts == 0`` does imply
    ``valid == False``.
    ``counts`` is diagnostic (confidence / sample support), never a validity
    test on its own.
measured quantity
    **The highest surface point anywhere inside the cell's square footprint.**
    Both paths implement that same operator, which is what makes them
    interchangeable:
      * (A) computes it in closed form against the spheres (exact), and
      * (B) approximates it by ``agg="max"`` over the depth samples that land
        in the cell (exact in the dense-sampling limit).
    Using the cell *centre* instead of the cell *footprint* is NOT equivalent:
    on a sparse pellet layer a centre sample can drop through an inter-pellet
    gap to the floor while the footprint still contains a pellet crown — a
    difference of the full pile height.  The footprint operator was chosen
    precisely to remove that discontinuity.
aggregation
    ``agg`` is part of the contract and is recorded in the header.  It only
    applies to path (B), where it selects the nearest-rank percentile of the
    in-cell depth samples: ``"max"`` (default, matches (A)), ``"p95"``,
    ``"p90"``, ``"median"``.
    ⚠ See :func:`slope_bias_m` — on a slope, the footprint operator carries a
    deterministic ``+(cell/2)·tanθ`` bias relative to the cell-centre height;
    ``"median"`` drops that bias but no longer measures the contract quantity.
    ⚠ ``"max"`` is also the worst choice under sensor noise: with ~11 samples
    per cell and the Azure Kinect NFOV spec-bound σ = 17 mm, the probe measures
    a +25 mm mean overshoot for ``"max"`` versus 7.3 mm rms for ``"median"``.
    Pick per data source, and always record it in the header.

CELL SIZE — default 5.0 mm, and why
-----------------------------------
yard_track used 10 mm because its objects were 22-34 mm rocks (D453 design
``sim_scripts/p26_y1_testbed_design_author.py``: "최소 물체 폭 22 mm의 절반
이하").  Continuous pellets are 3-5 mm, so that argument does not transfer.

  * lower bound — below one pellet diameter each cell resolves at most a single
    pellet, so the map alternates between crown and inter-pellet valley; the
    per-cell variance is then ~the pellet radius (2.5 mm) of pure aliasing that
    carries no pile-scale information.
  * upper bound — 10 mm is two pellet diameters and averages away the scoop
    trench walls.  Predicting "what shape is left after a scoop" is the whole
    point of the study, so the trench edge must survive the discretisation.
  * real sensing — at the calibrated Kinect standoff (~0.9 m) one colour-aligned
    depth pixel covers ``z/fx = 0.9/608.33 = 1.48 mm``, so a 5 mm cell collects
    ~11 samples; a 3 mm cell would collect ~4 and would not average the ~17 mm
    NFOV random depth error down usefully.
  * model input size — a 150x150 mm source region is 30x30 cells at 5 mm, a
    convenient CNN input; at 3 mm it is 50x50 with mostly sensor noise added.

For a **polydisperse** population there is no single "the" pellet diameter, so
read the two bounds separately: the aliasing floor is set by the *largest*
radius (that is what a cell crown can overshoot the mean surface by), while the
shape-resolution requirement is unchanged.  Size the cell from the characteristic
(median) diameter and check it against ``2 * particle_radius_max_m``, which
every Heightmap header records.

⚠ D453 CARRY-OVER (steep faces)
-------------------------------
D453 (``claudedocs/DECISIONS.md:27985``) found that a raycast heightmap matches
raw-mesh GT bit-exactly on flat cells but that a ~1.2 mm *horizontal*
representation gap (cooked convex hull vs raw mesh) is amplified by ``tan(θ)``
on steep faces — 6.4 mm on a 79.5° cell.  The same amplification law applies
here and is NOT specific to Isaac cooking: **any** horizontal error source
(calibration, cell footprint, point-cloud registration) becomes a vertical
error of ``ε_h · tan(θ)`` on a face of slope θ.  Use :func:`slope_bias_m` and
:func:`lateral_to_vertical_m` when designing a max-style gate, and prefer
slope-aware tolerances over a single global max threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SPEC_VERSION = "roarm-heightmap-v1"
DEFAULT_FRAME = "roarm_base"
DEFAULT_CELL_M = 0.005          # 5 mm — see module docstring for the derivation
DEFAULT_FILL_M = 0.0            # empty-cell height fill = support plane
AGGS = ("max", "p95", "p90", "median")


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridSpec:
    """Axis-aligned XY grid in the robot base frame.

    origin_xy_m is the lower-left CORNER of cell [0, 0]; shape is (rows, cols)
    with rows indexing +y and cols indexing +x.
    """

    origin_xy_m: tuple[float, float]
    cell_m: float = DEFAULT_CELL_M
    shape: tuple[int, int] = (30, 30)
    frame: str = DEFAULT_FRAME
    z_datum_m: float = 0.0

    def __post_init__(self) -> None:
        if self.cell_m <= 0:
            raise ValueError(f"cell_m must be > 0, got {self.cell_m}")
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ValueError(f"shape must be (rows, cols) >= 1, got {self.shape}")

    @classmethod
    def centered(cls, center_xy_m, extent_m: float, cell_m: float = DEFAULT_CELL_M,
                 **kw) -> "GridSpec":
        """Square grid of side ``extent_m`` centred on ``center_xy_m``.

        The side is rounded up to a whole number of cells, so the realised
        extent can exceed ``extent_m`` by at most one cell.
        """
        n = int(np.ceil(extent_m / cell_m - 1e-9))
        half = n * cell_m / 2.0
        return cls(origin_xy_m=(float(center_xy_m[0]) - half,
                                float(center_xy_m[1]) - half),
                   cell_m=float(cell_m), shape=(n, n), **kw)

    @property
    def n_cells(self) -> int:
        return int(self.shape[0] * self.shape[1])

    def cell_centers(self) -> np.ndarray:
        """(rows, cols, 2) float64 world XY of every cell centre."""
        rows, cols = self.shape
        x0, y0 = self.origin_xy_m
        xs = x0 + self.cell_m * (np.arange(cols) + 0.5)
        ys = y0 + self.cell_m * (np.arange(rows) + 0.5)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        return np.stack([gx, gy], axis=-1)

    def bounds_m(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the grid footprint."""
        rows, cols = self.shape
        x0, y0 = self.origin_xy_m
        return (x0, x0 + cols * self.cell_m, y0, y0 + rows * self.cell_m)

    def index_of(self, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World XY -> (row, col, inside).  Out-of-grid entries are clamped and
        flagged False in ``inside``."""
        x0, y0 = self.origin_xy_m
        rows, cols = self.shape
        c = np.floor((np.asarray(x, float) - x0) / self.cell_m).astype(np.int64)
        r = np.floor((np.asarray(y, float) - y0) / self.cell_m).astype(np.int64)
        inside = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
        return np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1), inside

    def to_header(self) -> dict:
        rows, cols = self.shape
        return {
            "frame": self.frame,
            "cell_m": float(self.cell_m),
            "cell_mm": float(self.cell_m * 1000.0),
            "origin_xy_m": [float(self.origin_xy_m[0]), float(self.origin_xy_m[1])],
            "origin_is": "lower-left corner of cell [0, 0]",
            "shape": [int(rows), int(cols)],
            "indexing": "height[row, col]; row -> +y, col -> +x",
            "cell_center_formula": ("x = origin_x + (col + 0.5) * cell_m ; "
                                    "y = origin_y + (row + 0.5) * cell_m"),
            "z_datum_m": float(self.z_datum_m),
        }

    @classmethod
    def from_header(cls, h: dict) -> "GridSpec":
        return cls(origin_xy_m=(float(h["origin_xy_m"][0]), float(h["origin_xy_m"][1])),
                   cell_m=float(h["cell_m"]), shape=(int(h["shape"][0]), int(h["shape"][1])),
                   frame=str(h["frame"]), z_datum_m=float(h.get("z_datum_m", 0.0)))


# --------------------------------------------------------------------------- #
# heightmap
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Heightmap:
    height: np.ndarray                    # (rows, cols) float32, metres
    valid: np.ndarray                     # (rows, cols) bool
    counts: np.ndarray                    # (rows, cols) int32
    spec: GridSpec
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.height.shape != tuple(self.spec.shape):
            raise ValueError(f"height {self.height.shape} != spec {self.spec.shape}")
        for name, arr, dt in (("height", self.height, np.float32),
                              ("valid", self.valid, np.bool_),
                              ("counts", self.counts, np.int32)):
            if arr.dtype != dt:
                raise TypeError(f"{name} must be {np.dtype(dt).name}, got {arr.dtype}")
            if arr.shape != tuple(self.spec.shape):
                raise ValueError(f"{name} shape {arr.shape} != spec {self.spec.shape}")

    # -- contract header ---------------------------------------------------- #
    def header(self) -> dict:
        return {
            "spec_version": SPEC_VERSION,
            **self.spec.to_header(),
            "height_unit": "m",
            "height_dtype": "float32",
            "height_is": ("surface z in `frame` minus z_datum_m "
                          "(z_datum_m = z of the support plane)"),
            "valid_dtype": "bool",
            "counts_dtype": "int32",
            "counts_is": "number of surface samples reduced into the cell",
            "empty_cell": {
                "rule": "valid=False  =>  height = height_fill_m",
                "height_fill_m": float(self.meta.get("height_fill_m", DEFAULT_FILL_M)),
                "consumer_note": ("use `valid`; the fill value is padding, not a "
                                  "measurement.  `counts` is sample support only — "
                                  "on the particle path counts==0 is a valid "
                                  "floor reading, not a hole"),
            },
            **self.meta,
        }

    # -- io ------------------------------------------------------------------ #
    def save(self, npz_path) -> tuple[Path, Path]:
        """Write ``<stem>.npz`` (arrays + embedded header) and ``<stem>.json``
        (header alone, for humans and for grepping the contract)."""
        npz_path = Path(npz_path)
        hdr = self.header()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, height=self.height, valid=self.valid,
                 counts=self.counts, header_json=np.array(json.dumps(hdr,
                                                                     ensure_ascii=False)))
        json_path = npz_path.with_suffix(".json")
        json_path.write_text(json.dumps(hdr, indent=2, ensure_ascii=False) + "\n")
        return npz_path, json_path

    @staticmethod
    def load(npz_path) -> "Heightmap":
        d = np.load(Path(npz_path), allow_pickle=False)
        hdr = json.loads(str(d["header_json"]))
        if hdr.get("spec_version") != SPEC_VERSION:
            raise ValueError(f"spec_version mismatch: {hdr.get('spec_version')!r}")
        fixed = set(GridSpec(origin_xy_m=(0, 0)).to_header()) | {
            "spec_version", "height_unit", "height_dtype", "height_is",
            "valid_dtype", "counts_dtype", "counts_is", "empty_cell"}
        meta = {k: v for k, v in hdr.items() if k not in fixed}
        meta["height_fill_m"] = float(hdr["empty_cell"]["height_fill_m"])
        return Heightmap(height=d["height"].astype(np.float32),
                         valid=d["valid"].astype(np.bool_),
                         counts=d["counts"].astype(np.int32),
                         spec=GridSpec.from_header(hdr), meta=meta)


def _finalize(z: np.ndarray, counts: np.ndarray, spec: GridSpec, meta: dict,
              fill_m: float, valid: np.ndarray | None = None) -> Heightmap:
    """float64 accumulator -> the frozen float32 contract.

    ``valid`` defaults to ``counts > 0``; path (A) overrides it because a cell
    with no sphere above it still yields a real floor reading.
    """
    valid = (counts > 0) if valid is None else np.asarray(valid, dtype=bool)
    z = np.where(valid, z - spec.z_datum_m, fill_m)
    return Heightmap(height=np.ascontiguousarray(z, dtype=np.float32),
                     valid=np.ascontiguousarray(valid, dtype=np.bool_),
                     counts=np.ascontiguousarray(counts, dtype=np.int32),
                     spec=spec, meta={"height_fill_m": float(fill_m), **meta})


def _reduce_cells(flat_idx: np.ndarray, z: np.ndarray, n_cells: int,
                  agg: str) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-rank percentile reduction of z per cell.

    One lexsort handles every ``agg`` uniformly; ``max`` is just q = 1.0.
    Returns (z_per_cell float64, counts int64), both length ``n_cells``.
    """
    if agg not in AGGS:
        raise ValueError(f"agg must be one of {AGGS}, got {agg!r}")
    q = {"max": 1.0, "p95": 0.95, "p90": 0.90, "median": 0.5}[agg]
    counts = np.bincount(flat_idx, minlength=n_cells)
    out = np.zeros(n_cells, dtype=np.float64)
    if flat_idx.size:
        order = np.lexsort((z, flat_idx))
        idx_s, z_s = flat_idx[order], z[order]
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        nz = counts > 0
        # nearest-rank: index round(q * (n - 1)) within each cell's sorted block
        # nearest rank, round-half-UP (np.rint is banker's rounding and would
        # make the median index depend on the parity of the sample count)
        take = starts[nz] + np.floor(q * (counts[nz] - 1) + 0.5).astype(np.int64)
        out[nz] = z_s[take]
        assert idx_s[take[0]] == np.nonzero(nz)[0][0]  # block alignment guard
    return out, counts


# --------------------------------------------------------------------------- #
# (A) particles -> heightmap
# --------------------------------------------------------------------------- #
def heightmap_from_particles(centers_m, radius_m, spec: GridSpec,
                             *, fill_m: float = DEFAULT_FILL_M,
                             floor_z_m: float = 0.0,
                             extra_meta: dict | None = None) -> Heightmap:
    """Exact highest sphere-surface point inside each cell footprint.

    For cell rectangle ``R`` the height is
    ``max over spheres i of  p_z,i + sqrt(r_i^2 - dist(p_xy,i, R)^2)``
    over spheres with ``dist(p_xy,i, R) <= r_i``, where ``dist`` is the distance
    from the sphere centre to the nearest point of ``R``.  Because ``sqrt`` is
    monotonically decreasing in that distance, the nearest-point evaluation is
    the exact footprint maximum — no sub-sampling.  Cells that no sphere
    overhangs get ``floor_z_m`` (the ray reaches the support plane) and stay
    ``valid``.

    This is the particle analogue of the yard_track PhysX ``raycast_closest``
    observation, and unlike that one it is *exact*: DEM spheres have no cooked
    convex approximation, so the D453 cooked-hull horizontal gap does not exist
    on this path.  Steep-face error on a pellet pile is instead bounded by the
    pellet radius (crown-vs-valley), see the probe's slope measurement.

    Parameters
    ----------
    centers_m : (N, 3) array of sphere centres in the base frame, metres.
    radius_m  : scalar (monodisperse) **or** (N,) array of per-sphere radii,
        metres.  A DEME pile carries ``radii_m`` as an (N,) array; pass it
        straight through.  A uniform (N,) array gives bit-identical output to
        the equivalent scalar.

    Notes
    -----
    Each sphere is scattered over a ``+-k`` cell window with ``k`` derived from
    **its own** radius, so a narrow particle in a wide size distribution is not
    scanned over the largest sphere's window.  Spheres are grouped by ``k`` and
    each group runs its own window loop; cost is therefore set by the size
    distribution, not by ``r_max`` alone.
    """
    P = np.asarray(centers_m, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"centers_m must be (N, 3), got {P.shape}")
    n = P.shape[0]

    r_arr = np.asarray(radius_m, dtype=np.float64).ravel()
    if r_arr.size == 1:
        r_arr = np.full(n, float(r_arr[0]), dtype=np.float64)
    elif r_arr.size != n:
        raise ValueError(f"radius_m must be scalar or ({n},), got shape "
                         f"{np.shape(radius_m)}")
    if n and not np.all(np.isfinite(r_arr)):
        raise ValueError("radius_m contains non-finite values")
    if n and (r_arr <= 0).any():
        raise ValueError(f"all radii must be > 0, got min {float(r_arr.min())}")

    rows, cols = spec.shape
    z = np.full((rows, cols), float(floor_z_m), dtype=np.float64)
    counts = np.zeros((rows, cols), dtype=np.int64)

    if n:
        # A sphere can only reach cell rectangles within its own radius: a +-k
        # cell window around the cell it sits in (+1 because r is measured from
        # the rectangle, not from the cell centre).  Group by k so that a
        # polydisperse population costs what its own sizes cost.
        r0, c0, _ = spec.index_of(P[:, 0], P[:, 1])
        x0g, y0g = spec.origin_xy_m
        k_all = np.ceil(r_arr / spec.cell_m).astype(np.int64) + 1
        for k in np.unique(k_all):
            grp = np.nonzero(k_all == k)[0]
            k = int(k)
            for dr in range(-k, k + 1):
                for dc in range(-k, k + 1):
                    ri, ci = r0[grp] + dr, c0[grp] + dc
                    ok = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
                    if not ok.any():
                        continue
                    ri, ci, sel = ri[ok], ci[ok], grp[ok]
                    # nearest point of the cell rectangle to the sphere centre
                    qx = np.clip(P[sel, 0], x0g + ci * spec.cell_m,
                                 x0g + (ci + 1) * spec.cell_m)
                    qy = np.clip(P[sel, 1], y0g + ri * spec.cell_m,
                                 y0g + (ri + 1) * spec.cell_m)
                    dx, dy = P[sel, 0] - qx, P[sel, 1] - qy
                    d2 = dx * dx + dy * dy
                    r2 = r_arr[sel] ** 2
                    hit = d2 <= r2
                    if not hit.any():
                        continue
                    ri, ci, sel = ri[hit], ci[hit], sel[hit]
                    ztop = P[sel, 2] + np.sqrt(np.maximum(r2[hit] - d2[hit], 0.0))
                    np.maximum.at(z, (ri, ci), ztop)
                    np.add.at(counts, (ri, ci), 1)

    uniform = bool(n == 0 or r_arr.min() == r_arr.max())
    meta = {"source": "particles", "agg": "max",
            "agg_rule": "exact closed-form maximum over the cell footprint",
            "n_particles": int(n), "polydisperse": bool(not uniform),
            "particle_radius_min_m": float(r_arr.min()) if n else None,
            "particle_radius_max_m": float(r_arr.max()) if n else None,
            "particle_radius_mean_m": float(r_arr.mean()) if n else None,
            "floor_z_m": float(floor_z_m),
            "note": ("cells no sphere overhangs report floor_z_m and stay valid; "
                     "counts is the number of spheres reaching the cell rectangle"),
            **(extra_meta or {})}
    if uniform and n:
        # kept for monodisperse so existing readers of this key keep working
        meta["particle_radius_m"] = float(r_arr[0])
    # every cell is observed on this path: uncovered cells legitimately see the floor
    return _finalize(z, counts, spec, meta, fill_m,
                     valid=np.ones((rows, cols), dtype=bool))


# --------------------------------------------------------------------------- #
# (B) depth image -> heightmap
# --------------------------------------------------------------------------- #
def deproject_depth(depth_m: np.ndarray, intr: dict, *, stride: int = 1,
                    valid_range_m: tuple[float, float] = (0.30, 2.00)) -> np.ndarray:
    """Depth image -> (M, 3) camera-frame points, metres.

    Pinhole model identical to ``sim_scripts/kinect_handeye_solve.back_project``:
    ``x = (u - cx) * z / fx``, ``y = (v - cy) * z / fy``.  Depth values of 0 or
    NaN (Kinect "no return") and values outside ``valid_range_m`` are dropped.
    ``depth_m`` must already be in metres and aligned to the intrinsics given
    (i.e. a colour-aligned transformed-depth image for the colour intrinsics in
    ``sim_scripts/kinect_calib.yaml``).
    """
    D = np.asarray(depth_m, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError(f"depth_m must be (H, W), got {D.shape}")
    D = D[::stride, ::stride]
    v_idx, u_idx = np.nonzero(np.isfinite(D) & (D >= valid_range_m[0])
                              & (D <= valid_range_m[1]))
    zc = D[v_idx, u_idx]
    u = u_idx.astype(np.float64) * stride
    v = v_idx.astype(np.float64) * stride
    xc = (u - intr["cx"]) * zc / intr["fx"]
    yc = (v - intr["cy"]) * zc / intr["fy"]
    return np.stack([xc, yc, zc], axis=1)


def heightmap_from_points(points_base_m, spec: GridSpec, *, agg: str = "max",
                          z_range_m: tuple[float, float] | None = None,
                          fill_m: float = DEFAULT_FILL_M,
                          extra_meta: dict | None = None) -> Heightmap:
    """Reduce an arbitrary base-frame point cloud into the grid.

    Shared core of path (B).  ``z_range_m`` (in base-frame z, before the datum
    subtraction) rejects floor-penetrating and above-pile outliers.
    """
    P = np.asarray(points_base_m, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {P.shape}")
    rows, cols = spec.shape
    r, c, inside = spec.index_of(P[:, 0], P[:, 1])
    keep = inside
    if z_range_m is not None:
        keep = keep & (P[:, 2] >= z_range_m[0]) & (P[:, 2] <= z_range_m[1])
    flat = (r[keep] * cols + c[keep]).astype(np.int64)
    z_flat, cnt_flat = _reduce_cells(flat, P[keep, 2], spec.n_cells, agg)
    meta = {"source": "points", "agg": agg,
            "agg_rule": "nearest-rank percentile within the cell footprint",
            "n_points_in": int(P.shape[0]), "n_points_used": int(keep.sum()),
            "z_range_m": list(z_range_m) if z_range_m else None,
            **(extra_meta or {})}
    return _finalize(z_flat.reshape(rows, cols), cnt_flat.reshape(rows, cols),
                     spec, meta, fill_m)


def heightmap_from_depth(depth_m, intr: dict, R_cam_to_base, t_cam_to_base,
                         spec: GridSpec, *, agg: str = "max", stride: int = 1,
                         depth_valid_range_m: tuple[float, float] = (0.30, 2.00),
                         z_range_m: tuple[float, float] | None = None,
                         fill_m: float = DEFAULT_FILL_M,
                         extra_meta: dict | None = None) -> Heightmap:
    """Azure Kinect depth frame -> heightmap in the robot base frame.

    ``p_base = R_cam_to_base @ p_cam + t_cam_to_base`` — the exact convention
    solved and stored by ``sim_scripts/kinect_handeye_solve.py`` /
    ``sim_scripts/kinect_calib.yaml`` (RMSE 10.13 mm, 27/31 poses).

    ⚠ that 10.13 mm is a 3-D residual; its horizontal component is amplified to
    ``ε_h · tan(θ)`` of vertical error on a pile face of slope θ (D453 law).
    Do not gate a steep pile with a flat-cell tolerance.
    """
    R = np.asarray(R_cam_to_base, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_cam_to_base, dtype=np.float64).reshape(3)
    pc = deproject_depth(depth_m, intr, stride=stride,
                         valid_range_m=depth_valid_range_m)
    pb = pc @ R.T + t
    meta = {"source": "depth", "camera": "azure_kinect_dk",
            "extrinsics_convention": "p_base = R @ p_cam + t",
            "intrinsics": {k: float(intr[k]) for k in ("fx", "fy", "cx", "cy")},
            "stride": int(stride),
            "depth_valid_range_m": list(depth_valid_range_m),
            **(extra_meta or {})}
    return heightmap_from_points(pb, spec, agg=agg, z_range_m=z_range_m,
                                 fill_m=fill_m, extra_meta=meta)


# --------------------------------------------------------------------------- #
# calibration loading + the D453 slope law
# --------------------------------------------------------------------------- #
def load_kinect_calib(path="sim_scripts/kinect_calib.yaml") -> dict:
    """Read ``kinect_calib.yaml`` -> ``{intrinsics, R, t, rmse_mm}``."""
    import yaml  # optional dependency; only path (B) with a real camera needs it
    d = yaml.safe_load(Path(path).read_text())
    return {"intrinsics": d["intrinsics"],
            "R": np.array(d["extrinsics"]["rotation_matrix"], dtype=np.float64),
            "t": np.array(d["extrinsics"]["translation_m"], dtype=np.float64),
            "rmse_mm": float(d["quality"]["rmse_mm"])}


def lateral_to_vertical_m(lateral_err_m: float, slope_deg) -> np.ndarray:
    """D453 amplification law: a horizontal error becomes ``ε_h · tan(θ)`` of
    vertical error on a face of slope θ (θ measured from horizontal)."""
    th = np.radians(np.minimum(np.asarray(slope_deg, dtype=np.float64), 89.9))
    return float(lateral_err_m) * np.tan(th)


def slope_bias_m(cell_m: float, slope_deg, *, agg: str = "max",
                 diagonal: bool = False) -> np.ndarray:
    """Footprint bias of the contract quantity relative to the cell-centre height.

    The contract measures the highest surface point in the cell, which on a
    plane of slope θ sits at the up-slope cell edge: ``+(cell/2)·tanθ`` when the
    gradient is axis-aligned, and ``+(cell/2)·tanθ·sqrt(2)`` when it points
    along the cell diagonal (the corner is ``cell/2`` further out in each axis,
    but each axis only carries ``tanθ/sqrt(2)`` of the gradient).
    ``agg="median"`` is unbiased on a plane, i.e. it measures the cell-centre
    height instead of the contract quantity.
    """
    half = 0.5 * float(cell_m) * (np.sqrt(2.0) if diagonal else 1.0)
    th = np.radians(np.minimum(np.asarray(slope_deg, dtype=np.float64), 89.9))
    return (half * np.tan(th)) if agg == "max" else np.zeros_like(th)


def slope_aware_tol_m(cell_m: float, lateral_err_m: float, slope_deg,
                      base_tol_m: float = 5.0e-4) -> np.ndarray:
    """Suggested per-cell tolerance = flat-cell tolerance + both tan(θ) terms.

    Use this instead of a single global max threshold; D453's G-hmap FAIL was a
    gate-design artefact of ignoring exactly these two terms.
    """
    return (float(base_tol_m) + slope_bias_m(cell_m, slope_deg, agg="max")
            + lateral_to_vertical_m(lateral_err_m, slope_deg))


def cell_slope_deg(hm: Heightmap) -> np.ndarray:
    """Per-cell surface slope from horizontal, degrees, by central differences.

    Differences the raw ``height`` array, fill values included, so the result is
    only meaningful away from invalid cells — mask with ``hm.valid`` (and its
    neighbours) before using it.  Intended for slope-band reporting, not as a
    gate input.
    """
    h = np.array(hm.height, dtype=np.float64)
    gy, gx = np.gradient(h, hm.spec.cell_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))
