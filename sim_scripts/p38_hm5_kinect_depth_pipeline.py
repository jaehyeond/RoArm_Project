#!/usr/bin/env python3
"""Azure Kinect transformed-depth -> roarm-heightmap-v1 PP-ready pipeline.

This script never imports or controls the RoArm SDK.  It accepts either one
saved colour-aligned/transformed depth frame or one live pyk4a capture, applies
the frozen 2026-08-31 grid, preserves pixel/cell validity masks, and optionally
emits the complete D341 Rerun bundle.  ``--synthetic`` is the deterministic
positive control used while the physical Kinect is disconnected.

PP arrival (new, empty output folder under heightmap_track):

  python sim_scripts/p38_hm5_kinect_depth_pipeline.py --live \
    --output-dir claudedocs/runtime_logs/heightmap_track/pp_pile_<date> --rerun

or, after saving pyk4a ``capture.transformed_depth`` in millimetres:

  python sim_scripts/p38_hm5_kinect_depth_pipeline.py \
    --input-depth transformed_depth.npy --depth-unit mm \
    --output-dir claudedocs/runtime_logs/heightmap_track/pp_pile_<date> --rerun
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.heightmap import (  # noqa: E402
    Heightmap,
    GridSpec,
    SPEC_VERSION,
    deproject_depth,
    heightmap_from_kinect_depth,
    load_kinect_calib,
)


HEIGHTMAP_ROOT = REPO / "claudedocs/runtime_logs/heightmap_track"
DEFAULT_OUT = HEIGHTMAP_ROOT / "hm5_real_depth_pp_ready"
CALIB_PATH = REPO / "sim_scripts/kinect_calib.yaml"

# User-frozen 2026-08-31 contract.  Do not parameterize these from the CLI.
GRID = GridSpec(
    origin_xy_m=(0.125, -0.190),
    cell_m=0.005,
    shape=(76, 38),
    frame="roarm_base",
    z_datum_m=0.0,
)
AGG = "max"
FILL_M = 0.0
DEPTH_RANGE_M = (0.30, 2.00)
PILE_Z_RANGE_M = (-0.020, 0.200)
EDGE_JUMP_M = 0.030
OUTLIER_DELTA_M = 0.015
SEED = 5038


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def _require_heightmap_output(path: Path) -> Path:
    resolved = path.resolve()
    root = HEIGHTMAP_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a new folder below {root}, got {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _refuse_overwrite(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing evidence: " + ", ".join(existing))


def load_saved_depth(path: Path) -> tuple[np.ndarray, str]:
    """Load a 2-D saved transformed-depth frame without guessing geometry."""
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
        key = "npy_array"
    elif path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        candidates = [
            key for key in ("transformed_depth_mm", "depth_mm", "depth_m", "depth")
            if key in data.files
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"NPZ must contain exactly one supported depth key, got {candidates}; "
                "supported=transformed_depth_mm,depth_mm,depth_m,depth"
            )
        key = candidates[0]
        depth = data[key]
    else:
        raise ValueError("saved depth must be .npy or .npz")
    if depth.ndim != 2:
        raise ValueError(f"saved depth must be 2-D, got {depth.shape}")
    return np.asarray(depth), key


def capture_live_transformed_depth(warmup_frames: int = 5) -> np.ndarray:
    """Read one aligned Azure Kinect frame; no robot modules or ports touched."""
    import pyk4a
    from pyk4a import Config, PyK4A

    camera = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    ))
    camera.start()
    try:
        capture = None
        for _ in range(max(1, int(warmup_frames))):
            capture = camera.get_capture(timeout=10_000)
        assert capture is not None
        depth = capture.transformed_depth
        if depth is None or depth.ndim != 2:
            raise RuntimeError("Kinect returned no transformed_depth frame")
        return np.asarray(depth).copy()
    finally:
        camera.stop()


def _cone_surface_pellets() -> np.ndarray:
    """Pellet-scale surface used only to render the disconnected-camera control."""
    cx, cy = 0.220, 0.0
    cone_r, cone_h, pellet_r, spacing = 0.075, 0.070, 0.0025, 0.0055
    points: list[np.ndarray] = []
    ys = np.arange(cy - cone_r, cy + cone_r + spacing / 2, spacing)
    for y in ys:
        half = np.sqrt(max(0.0, cone_r ** 2 - (y - cy) ** 2))
        for x in np.arange(cx - half, cx + half + spacing / 2, spacing):
            radial = np.hypot(x - cx, y - cy)
            if radial > cone_r:
                continue
            slope = cone_h / cone_r
            normal = np.array([0.0, 0.0, 1.0]) if radial < 1e-12 else np.array([
                slope * (x - cx) / radial,
                slope * (y - cy) / radial,
                1.0,
            ])
            normal /= np.linalg.norm(normal)
            surface = np.array([x, y, cone_h * (1.0 - radial / cone_r)])
            points.append(surface + pellet_r * normal)
    return np.asarray(points, dtype=np.float64)


def make_synthetic_depth(calib: dict) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Render the calibrated oblique view and inject known real-depth defects."""
    from sim_scripts.p33_hm1_heightmap_contract_probe import render_sphere_depth

    rng = np.random.default_rng(SEED)
    pellets = _cone_surface_pellets()
    depth_m = render_sphere_depth(
        pellets,
        0.0025,
        calib["intrinsics"],
        calib["R"],
        calib["t"],
        hw=(720, 1280),
        floor_z=0.0,
        noise_sigma_m=0.0,
        rng=rng,
    ).astype(np.float32)
    raw_mm = depth_m * np.float32(1000.0)

    zero_mask = np.zeros(raw_mm.shape, dtype=bool)
    nan_mask = np.zeros(raw_mm.shape, dtype=bool)
    flight_mask = np.zeros(raw_mm.shape, dtype=bool)

    # Known valid pixels near the projected pile centre become explicit zero
    # and NaN no-return controls.  Their numeric content must never become z=0.
    pc = calib["R"].T @ (np.array([0.220, 0.0, 0.035]) - calib["t"])
    uc = int(round(calib["intrinsics"]["cx"] + calib["intrinsics"]["fx"] * pc[0] / pc[2]))
    vc = int(round(calib["intrinsics"]["cy"] + calib["intrinsics"]["fy"] * pc[1] / pc[2]))
    zero_mask[max(0, vc - 7):vc - 2, max(0, uc - 10):uc - 5] = True
    nan_mask[vc + 2:vc + 7, uc + 5:uc + 10] = True
    zero_mask &= raw_mm > 0
    nan_mask &= raw_mm > 0
    raw_mm[zero_mask] = 0.0
    raw_mm[nan_mask] = np.nan

    # Flying-pixel positive control: replace deterministic high-gradient pixels
    # by an in-between ToF return.  The edge guard must reject them.
    left = np.roll(depth_m, 1, axis=1)
    boundary = ((depth_m > 0.0) & (left > 0.0)
                & (np.abs(depth_m - left) > 0.040)
                & ~zero_mask & ~nan_mask)
    boundary[:, 0] = False
    candidates = np.argwhere(boundary)
    if candidates.size:
        chosen = candidates[::max(1, len(candidates) // 256)][:256]
        for v, u in chosen:
            raw_mm[v, u] = np.float32(500.0 * (depth_m[v, u] + left[v, u]))
            flight_mask[v, u] = True
    if int(flight_mask.sum()) < 16:
        # Fallback positive control if this renderer/view lacks enough 40 mm
        # silhouettes: isolated +80 mm ToF spikes are the same rejection class.
        valid_coords = np.argwhere((depth_m > 0.35) & (depth_m < 1.5)
                                   & ~zero_mask & ~nan_mask)
        for v, u in valid_coords[::max(1, len(valid_coords) // 64)][:64]:
            raw_mm[v, u] += 80.0
            flight_mask[v, u] = True

    masks = {
        "synthetic_zero_injected_mask": zero_mask,
        "synthetic_nan_injected_mask": nan_mask,
        "synthetic_flight_injected_mask": flight_mask,
    }
    meta = {
        "seed": SEED,
        "scene": "pellet-scale conical surface plus floor",
        "n_surface_pellets": int(pellets.shape[0]),
        "zero_injected_pixels": int(zero_mask.sum()),
        "nan_injected_pixels": int(nan_mask.sum()),
        "flight_injected_pixels": int(flight_mask.sum()),
        "non_claim": "synthetic positive control; not a real Kinect frame",
    }
    return raw_mm, masks, meta


def points_in_decision_volume(filtered_depth_m: np.ndarray, calib: dict) -> np.ndarray:
    points_cam = deproject_depth(
        filtered_depth_m,
        calib["intrinsics"],
        valid_range_m=DEPTH_RANGE_M,
    )
    points_base = points_cam @ calib["R"].T + calib["t"]
    x0, x1, y0, y1 = GRID.bounds_m()
    keep = (
        (points_base[:, 0] >= x0) & (points_base[:, 0] <= x1)
        & (points_base[:, 1] >= y0) & (points_base[:, 1] <= y1)
        & (points_base[:, 2] >= PILE_Z_RANGE_M[0])
        & (points_base[:, 2] <= PILE_Z_RANGE_M[1])
    )
    return np.ascontiguousarray(points_base[keep], dtype=np.float32)


def save_diagnostic_png(out: Path, raw_depth: np.ndarray, unit: str,
                        masks: dict[str, np.ndarray], hm: Heightmap,
                        points_base: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    depth_m = raw_depth.astype(np.float64) * (0.001 if unit == "mm" else 1.0)
    depth_vis = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, np.nan)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    im = axes[0, 0].imshow(depth_vis, cmap="turbo", vmin=0.45, vmax=1.05)
    axes[0, 0].set_title("raw aligned depth [m]")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)
    axes[0, 1].imshow(masks["valid_mask"], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("pixel valid_mask (white=True)")
    rejected = (masks["edge_rejected_mask"].astype(np.uint8) * 1
                + masks["outlier_rejected_mask"].astype(np.uint8) * 2
                + masks["input_invalid_mask"].astype(np.uint8) * 3)
    axes[0, 2].imshow(rejected, cmap="magma", vmin=0, vmax=3)
    axes[0, 2].set_title("invalid/edge/outlier rejection")
    if points_base.size:
        axes[1, 0].scatter(points_base[::max(1, len(points_base) // 15000), 0],
                           points_base[::max(1, len(points_base) // 15000), 1],
                           s=0.2, c=points_base[::max(1, len(points_base) // 15000), 2],
                           cmap="viridis")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title("roarm_base point cloud (top view)")
    extent = [GRID.bounds_m()[0], GRID.bounds_m()[1],
              GRID.bounds_m()[2], GRID.bounds_m()[3]]
    im = axes[1, 1].imshow(hm.height, origin="lower", extent=extent,
                           cmap="viridis", vmin=0.0, vmax=0.08, aspect="auto")
    axes[1, 1].set_title("heightmap [m], fill shown as 0")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    axes[1, 2].imshow(hm.valid, origin="lower", extent=extent,
                      cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[1, 2].set_title("cell valid mask (black = unseen)")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("HM5 real-depth path: numeric height and observation validity stay separate")
    fig.tight_layout()
    path = out / "hm5_diagnostic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


RERUN_ENTITIES = [
    "input/depth_raw",
    "input/masks",
    "geometry/pointcloud_base",
    "heightmap/valid_cells",
    "heightmap/unseen_cells",
    "frames/camera_origin",
    "frames/camera_axes",
    "frames/grid_origin",
    "frames/grid_axes",
]


def _rerun_component_contract() -> dict[str, list[str]]:
    return {
        "input/depth_raw": ["DepthImage:buffer", "DepthImage:format", "DepthImage:meter"],
        "input/masks": ["Image:buffer", "Image:format"],
        "geometry/pointcloud_base": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "heightmap/valid_cells": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
        "heightmap/unseen_cells": ["Points3D:positions", "Points3D:colors", "Points3D:radii"],
    }


def emit_rerun(out: Path, raw_depth: np.ndarray, unit: str,
               filter_masks: dict[str, np.ndarray], hm: Heightmap,
               points_base: np.ndarray, calib: dict) -> dict:
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if rr.__version__ != "0.34.1":
        raise RuntimeError(f"rerun version must stay 0.34.1, got {rr.__version__}")
    # ``hm5_timeline.rrd`` is retained as the failed first authoring attempt
    # (Grid container was not wrapped in Blueprint).  Forward-only evidence
    # policy forbids deleting/renaming it; c1 is the corrected final bundle.
    rrd = out / "hm5_timeline_c1.rrd"
    rbl = out / "hm5_timeline_c1.rbl"
    screenshot = out / "hm5_rerun_inspection_c1.png"
    report_path = out / "hm5_rerun_validation_c1.json"
    _refuse_overwrite([rrd, rbl, screenshot, report_path])

    app_id = "roarm_hm5_real_depth"
    layout = rrb.Grid(
        rrb.Spatial2DView(origin="/input/depth_raw", contents=["/input/depth_raw"],
                          name="1 | aligned depth"),
        rrb.Spatial2DView(origin="/input/masks", contents=["/input/masks"],
                          name="2 | valid / rejected pixels"),
        rrb.Spatial3DView(origin="/", contents=["/geometry/**", "/frames/**"],
                          name="3 | roarm_base point cloud + calibrated frames"),
        rrb.Spatial3DView(origin="/", contents=["/heightmap/**", "/frames/grid_*"],
                          name="4 | heightmap valid vs unseen cells"),
        grid_columns=2,
        name="HM5 Kinect depth decision bundle",
    )
    bp = rrb.Blueprint(layout, auto_layout=False, auto_views=False,
                       collapse_panels=True)

    depth_m = raw_depth.astype(np.float32) * np.float32(0.001 if unit == "mm" else 1.0)
    depth_log = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0.0).astype(np.float32)
    mask_rgb = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    mask_rgb[filter_masks["valid_mask"]] = [40, 210, 80]
    mask_rgb[filter_masks["input_invalid_mask"]] = [220, 40, 40]
    mask_rgb[filter_masks["edge_rejected_mask"]] = [255, 170, 20]
    mask_rgb[filter_masks["outlier_rejected_mask"]] = [210, 60, 230]

    centers = GRID.cell_centers()
    valid = hm.valid.ravel()
    xyz = np.stack([
        centers[..., 0].ravel(),
        centers[..., 1].ravel(),
        hm.height.astype(np.float64).ravel(),
    ], axis=1)
    invalid_xyz = xyz[~valid].copy()
    if invalid_xyz.size:
        invalid_xyz[:, 2] = GRID.z_datum_m

    camera_origin = np.asarray(calib["t"], dtype=np.float64).reshape(1, 3)
    camera_axes = np.asarray(calib["R"], dtype=np.float64).T * 0.060
    grid_origin = np.array([[GRID.origin_xy_m[0], GRID.origin_xy_m[1], GRID.z_datum_m]])
    grid_axes = np.eye(3, dtype=np.float64) * 0.060
    axis_colors = np.array([[255, 40, 40], [40, 255, 40], [40, 100, 255]], dtype=np.uint8)

    with rr.RecordingStream(app_id, recording_id=out.name, make_default=False,
                            send_properties=True) as rec:
        rec.save(str(rrd), write_footer=True)
        rec.send_blueprint(bp, make_active=True, make_default=True)
        rec.set_time("frame_idx", sequence=0)
        rec.log("input/depth_raw", rr.DepthImage(depth_log, meter=1.0,
                                                  depth_range=[0.30, 1.20]))
        rec.log("input/masks", rr.Image(mask_rgb))
        rec.log("geometry/pointcloud_base", rr.Points3D(
            points_base, colors=[60, 160, 240], radii=0.0010))
        rec.log("heightmap/valid_cells", rr.Points3D(
            xyz[valid], colors=[60, 220, 100], radii=0.0020))
        rec.log("heightmap/unseen_cells", rr.Points3D(
            invalid_xyz, colors=[240, 50, 50], radii=0.0022))
        rec.log("frames/camera_origin", rr.Points3D(
            camera_origin, colors=[255, 255, 255], radii=0.0060))
        rec.log("frames/camera_axes", rr.Arrows3D(
            origins=np.repeat(camera_origin, 3, axis=0), vectors=camera_axes,
            colors=axis_colors, radii=0.0015))
        rec.log("frames/grid_origin", rr.Points3D(
            grid_origin, colors=[255, 255, 255], radii=0.0040))
        rec.log("frames/grid_axes", rr.Arrows3D(
            origins=np.repeat(grid_origin, 3, axis=0), vectors=grid_axes,
            colors=axis_colors, radii=0.0015))
        rec.flush(timeout_sec=60.0)
    bp.save(app_id, str(rbl))

    report = validate_rerun_artifact(
        rrd,
        expected_entity_paths=RERUN_ENTITIES,
        exact_entity_paths=RERUN_ENTITIES,
        expected_timeline_names=["blueprint", "frame_idx", "log_time"],
        exact_timeline_names=["blueprint", "frame_idx", "log_time"],
        expected_entity_components=_rerun_component_contract(),
        blueprint_path=rbl,
        screenshot_path=screenshot,
        expected_version="0.34.1",
        timeout_s=300.0,
        cli_path="/home/cgxr/miniconda3/envs/isaaclab/bin/rerun",
    )
    write_json(report_path, report)
    return report


def process_frame(depth: np.ndarray, unit: str, source_meta: dict,
                  synthetic_masks: dict[str, np.ndarray], out: Path,
                  make_rerun: bool) -> dict:
    calib = load_kinect_calib(CALIB_PATH)
    expected_shape = (int(calib["intrinsics"]["height"]),
                      int(calib["intrinsics"]["width"]))
    if depth.shape != expected_shape:
        raise ValueError(
            f"depth shape {depth.shape} != calibrated colour-aligned shape {expected_shape}; "
            "use pyk4a capture.transformed_depth, not native 640x576 depth"
        )

    target_paths = [
        out / "hm5_input_depth.npz",
        out / "hm5_depth_masks.npz",
        out / "hm5_points_base.npz",
        out / "hm5_heightmap.npz",
        out / "hm5_heightmap.json",
        out / "hm5_results.json",
        out / "hm5_diagnostic.png",
    ]
    _refuse_overwrite(target_paths)

    hm, filtered = heightmap_from_kinect_depth(
        depth,
        calib["intrinsics"],
        calib["R"],
        calib["t"],
        GRID,
        unit=unit,
        depth_valid_range_m=DEPTH_RANGE_M,
        z_range_m=PILE_Z_RANGE_M,
        edge_jump_m=EDGE_JUMP_M,
        outlier_delta_m=OUTLIER_DELTA_M,
        fill_m=FILL_M,
        extra_meta={
            "input_alignment": "pyk4a capture.transformed_depth -> colour intrinsics",
            "calibration_path": str(CALIB_PATH.relative_to(REPO)),
            "calibration_sha256": sha256_file(CALIB_PATH),
            "calibration_rmse_mm": calib["rmse_mm"],
            "fk_rotation_convention": "fixed-axis Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            "source_kind": source_meta["source_kind"],
        },
    )
    points_base = points_in_decision_volume(filtered.depth_m, calib)

    np.savez_compressed(out / "hm5_input_depth.npz", depth=depth,
                        unit=np.array(unit), source_json=np.array(json.dumps(source_meta)))
    mask_payload = {
        "filtered_depth_m": filtered.depth_m,
        "raw_valid_mask": filtered.raw_valid_mask,
        "valid_mask": filtered.valid_mask,
        "input_invalid_mask": filtered.input_invalid_mask,
        "edge_rejected_mask": filtered.edge_rejected_mask,
        "outlier_rejected_mask": filtered.outlier_rejected_mask,
        **synthetic_masks,
    }
    np.savez_compressed(out / "hm5_depth_masks.npz", **mask_payload)
    np.savez_compressed(out / "hm5_points_base.npz", points_base_m=points_base)
    hm.save(out / "hm5_heightmap.npz")
    save_diagnostic_png(out, depth, unit, mask_payload, hm, points_base)

    invalid_cells = ~hm.valid
    measured_zero_cells = hm.valid & np.isclose(hm.height, 0.0, atol=1e-5)
    results = {
        "tool": "p38_hm5_kinect_depth_pipeline",
        "case": out.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "source": source_meta,
        "calibration": {
            "path": str(CALIB_PATH.relative_to(REPO)),
            "sha256": sha256_file(CALIB_PATH),
            "rmse_mm": calib["rmse_mm"],
            "extrinsics_convention": "p_base = R @ p_cam + t",
            "fk_rotation_convention": "fixed-axis Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            "recalibrated": False,
        },
        "contract": {
            "spec_version": SPEC_VERSION,
            "frame": GRID.frame,
            "shape": list(GRID.shape),
            "indexing": "height[row=y, col=x]",
            "cell_m": GRID.cell_m,
            "origin_xy_m": list(GRID.origin_xy_m),
            "agg": AGG,
            "empty_fill_m": FILL_M,
        },
        "pixel_filter": filtered.diagnostics,
        "heightmap_observation": {
            "total_cells": GRID.n_cells,
            "valid_cells": int(hm.valid.sum()),
            "unseen_or_occluded_cells": int(invalid_cells.sum()),
            "valid_fraction": float(hm.valid.mean()),
            "valid_measured_zero_height_cells": int(measured_zero_cells.sum()),
            "invalid_zero_padding_cells": int((invalid_cells & (hm.height == FILL_M)).sum()),
            "invalid_cells_nonfill_count": int((invalid_cells & (hm.height != FILL_M)).sum()),
            "n_points_in_decision_volume": int(points_base.shape[0]),
            "interpretation": ("height==0 alone is ambiguous; hm.valid is authoritative. "
                               "valid=False identifies unseen/out-of-FOV/occluded cells."),
        },
        "synthetic_positive_controls": {
            key: int(mask.sum()) for key, mask in synthetic_masks.items()
        },
        "artifacts": {
            "input_depth": "hm5_input_depth.npz",
            "pixel_masks": "hm5_depth_masks.npz",
            "pointcloud": "hm5_points_base.npz",
            "heightmap": "hm5_heightmap.npz",
            "heightmap_header": "hm5_heightmap.json",
            "diagnostic": "hm5_diagnostic.png",
        },
        "pp_arrival": {
            "live_command": ("/home/cgxr/miniconda3/envs/isaaclab/bin/python "
                             "sim_scripts/p38_hm5_kinect_depth_pipeline.py --live "
                             "--output-dir claudedocs/runtime_logs/heightmap_track/"
                             "pp_pile_<date> --rerun"),
            "saved_frame_command": ("/home/cgxr/miniconda3/envs/isaaclab/bin/python "
                                    "sim_scripts/p38_hm5_kinect_depth_pipeline.py "
                                    "--input-depth <transformed_depth.npy> --depth-unit mm "
                                    "--output-dir claudedocs/runtime_logs/heightmap_track/"
                                    "pp_pile_<date> --rerun"),
            "only_replace": "one 1280x720 colour-aligned transformed-depth frame",
            "camera_must_not_move": True,
            "recalibration_required": False,
            "robot_control": "none",
        },
    }
    write_json(out / "hm5_results.json", results)

    if make_rerun:
        report = emit_rerun(out, depth, unit, mask_payload, hm, points_base, calib)
        results["rerun"] = {
            "validation_path": "hm5_rerun_validation_c1.json",
            "pass": bool(report.get("pass")),
            "rrd": "hm5_timeline_c1.rrd",
            "rbl": "hm5_timeline_c1.rbl",
            "screenshot": "hm5_rerun_inspection_c1.png",
            "failed_attempt_retained": "hm5_timeline.rrd",
            "failed_attempt_reason": "layout Grid lacked Blueprint wrapper before RBL export",
        }
        write_json(out / "hm5_results.json", results)
    return results


def rerun_existing_bundle(out: Path) -> dict:
    """Complete D341 after a non-destructive authoring retry."""
    input_data = np.load(out / "hm5_input_depth.npz", allow_pickle=False)
    depth = input_data["depth"]
    unit = str(input_data["unit"])
    mask_data = np.load(out / "hm5_depth_masks.npz", allow_pickle=False)
    masks = {key: mask_data[key] for key in mask_data.files}
    hm = Heightmap.load(out / "hm5_heightmap.npz")
    points = np.load(out / "hm5_points_base.npz", allow_pickle=False)["points_base_m"]
    c1_rrd = out / "hm5_timeline_c1.rrd"
    if c1_rrd.exists():
        # The c1 recording itself is complete; only its validation component
        # expectation was wrong (Image vs DepthImage namespace).  Preserve the
        # false c1 report and render a fresh c2 screenshot/report forward-only.
        from roarm_rl.rerun_contract import validate_rerun_artifact
        c2_report_path = out / "hm5_rerun_validation_c2.json"
        c2_screenshot = out / "hm5_rerun_inspection_c2.png"
        _refuse_overwrite([c2_report_path, c2_screenshot])
        report = validate_rerun_artifact(
            c1_rrd,
            expected_entity_paths=RERUN_ENTITIES,
            exact_entity_paths=RERUN_ENTITIES,
            expected_timeline_names=["blueprint", "frame_idx", "log_time"],
            exact_timeline_names=["blueprint", "frame_idx", "log_time"],
            expected_entity_components=_rerun_component_contract(),
            blueprint_path=out / "hm5_timeline_c1.rbl",
            screenshot_path=c2_screenshot,
            expected_version="0.34.1",
            timeout_s=300.0,
            cli_path="/home/cgxr/miniconda3/envs/isaaclab/bin/rerun",
        )
        write_json(c2_report_path, report)
    else:
        report = emit_rerun(out, depth, unit, masks, hm, points,
                            load_kinect_calib(CALIB_PATH))
    results_path = out / "hm5_results.json"
    results = json.loads(results_path.read_text())
    results["rerun"] = {
        "validation_path": ("hm5_rerun_validation_c2.json" if c1_rrd.exists()
                            else "hm5_rerun_validation_c1.json"),
        "pass": bool(report.get("pass")),
        "rrd": "hm5_timeline_c1.rrd",
        "rbl": "hm5_timeline_c1.rbl",
        "screenshot": ("hm5_rerun_inspection_c2.png" if c1_rrd.exists()
                       else "hm5_rerun_inspection_c1.png"),
        "failed_attempt_retained": "hm5_timeline.rrd",
        "failed_attempt_reason": "layout Grid lacked Blueprint wrapper before RBL export",
        "failed_validation_retained": ("hm5_rerun_validation_c1.json"
                                       if c1_rrd.exists() else None),
    }
    write_json(results_path, results)
    return results


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_runtime_audit(path: Path) -> None:
    audit = json.loads(path.read_text())
    _assert(audit["pyk4a"]["installed"] is True, "pyk4a missing")
    _assert(audit["pyk4a"]["version"] == "1.5.0", "unexpected pyk4a version")
    _assert(audit["libk4a"]["installed"] is True, "libk4a missing")
    _assert(audit["libk4a"]["version"] == "1.4.2", "unexpected libk4a version")
    _assert(audit["physical_device"]["connected_device_count"] >= 0,
            "device count not recorded")
    _assert(audit["physical_device"]["status"] in {"connected", "not_connected"},
            "device status not explicit")
    print("KINECT_RUNTIME_AUDIT_OK")


def validate_bundle(out: Path) -> None:
    results = json.loads((out / "hm5_results.json").read_text())
    hm = Heightmap.load(out / "hm5_heightmap.npz")
    contract = results["contract"]
    _assert(contract == {
        "spec_version": "roarm-heightmap-v1",
        "frame": "roarm_base",
        "shape": [76, 38],
        "indexing": "height[row=y, col=x]",
        "cell_m": 0.005,
        "origin_xy_m": [0.125, -0.19],
        "agg": "max",
        "empty_fill_m": 0.0,
    }, f"contract drift: {contract}")
    _assert(hm.spec == GRID, "loaded GridSpec differs from frozen GRID")
    _assert(hm.height.dtype == np.float32 and hm.valid.dtype == np.bool_
            and hm.counts.dtype == np.int32, "dtype drift")
    _assert(hm.meta["agg"] == "max", "aggregation drift")
    _assert(results["calibration"]["extrinsics_convention"] == "p_base = R @ p_cam + t",
            "extrinsics convention drift")
    _assert(results["calibration"]["fk_rotation_convention"]
            == "fixed-axis Rz(yaw) @ Ry(pitch) @ Rx(roll)", "FK convention drift")
    _assert(results["calibration"]["sha256"] == sha256_file(CALIB_PATH),
            "calibration file changed")
    print("HM5_BUNDLE_CONTRACT_OK")


def validate_masks(out: Path) -> None:
    results = json.loads((out / "hm5_results.json").read_text())
    masks = np.load(out / "hm5_depth_masks.npz", allow_pickle=False)
    hm = Heightmap.load(out / "hm5_heightmap.npz")
    zero = masks["synthetic_zero_injected_mask"]
    nan = masks["synthetic_nan_injected_mask"]
    flight = masks["synthetic_flight_injected_mask"]
    rejected = masks["edge_rejected_mask"] | masks["outlier_rejected_mask"]
    _assert(int(zero.sum()) > 0 and int(nan.sum()) > 0, "invalid positive controls absent")
    _assert(np.all(masks["input_invalid_mask"][zero | nan]),
            "zero/NaN controls leaked into valid depth")
    _assert(not np.any(masks["valid_mask"][zero | nan]),
            "zero/NaN controls marked valid")
    _assert(int(flight.sum()) >= 16, "flight positive control absent")
    overlap = int((flight & rejected).sum()) / int(flight.sum())
    _assert(overlap >= 0.95, f"only {overlap:.3f} of flight controls rejected")
    _assert(int((~hm.valid).sum()) > 0, "no unseen/occluded cells identified")
    _assert(np.all(hm.height[~hm.valid] == 0.0), "invalid cells not zero padded")
    _assert(int((hm.valid & np.isclose(hm.height, 0.0, atol=1e-5)).sum()) > 0,
            "no measured floor cells for valid-vs-fill distinction")
    _assert(results["heightmap_observation"]["invalid_cells_nonfill_count"] == 0,
            "results disagree with cell padding")
    print("HM5_MASKS_AND_OCCLUSION_OK")


def _validate_rerun_current(out: Path) -> dict:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    return validate_rerun_artifact(
        out / "hm5_timeline_c1.rrd",
        expected_entity_paths=RERUN_ENTITIES,
        exact_entity_paths=RERUN_ENTITIES,
        expected_timeline_names=["blueprint", "frame_idx", "log_time"],
        exact_timeline_names=["blueprint", "frame_idx", "log_time"],
        expected_entity_components=_rerun_component_contract(),
        blueprint_path=out / "hm5_timeline_c1.rbl",
        expected_version="0.34.1",
        timeout_s=300.0,
        cli_path="/home/cgxr/miniconda3/envs/isaaclab/bin/rerun",
    )


def validate_rerun_bundle(out: Path) -> None:
    import rerun as rr
    import psutil
    results = json.loads((out / "hm5_results.json").read_text())
    rerun_paths = results["rerun"]
    prior = json.loads((out / rerun_paths["validation_path"]).read_text())
    inspection = json.loads((out / "hm5_inspection.json").read_text())
    current = _validate_rerun_current(out)
    _assert(rr.__version__ == "0.34.1", "rerun pin drift")
    _assert(np.__version__ == "1.26.0", "Isaac numpy pin drift")
    _assert(psutil.__version__ == "5.9.8", "Isaac psutil pin drift")
    _assert(prior.get("pass") is True and current.get("pass") is True,
            "Rerun validation failed")
    _assert(prior["headless_render"]["ok"] is True, "headless screenshot render failed")
    _assert((out / rerun_paths["screenshot"]).is_file(), "screenshot absent")
    _assert(inspection.get("status") == "complete"
            and len(inspection.get("observations", [])) >= 3,
            "actual visual inspection record incomplete")
    print("HM5_RERUN_OBSERVABILITY_OK visual_inspection=complete")


def validate_pp_readiness(out: Path) -> None:
    results = json.loads((out / "hm5_results.json").read_text())
    pp = results["pp_arrival"]
    source = Path(__file__).read_text()
    _assert(pp["only_replace"] == "one 1280x720 colour-aligned transformed-depth frame",
            "frame-swap contract absent")
    _assert("--live" in pp["live_command"] and "--input-depth" in pp["saved_frame_command"],
            "both capture paths not documented")
    _assert(pp["recalibration_required"] is False and pp["robot_control"] == "none",
            "workflow exceeds authorization")
    forbidden_tokens = (
        "/dev/" + "ttyUSB",
        "torque" + "_set",
        "joints" + "_angle_ctrl",
        "move" + "_init",
    )
    for forbidden in forbidden_tokens:
        _assert(forbidden not in source, f"forbidden robot-control token in script: {forbidden}")
    _assert((out / "README.md").is_file(), "PP instructions missing")
    print("HM5_PP_FRAME_SWAP_READY")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true",
                        help="deterministic disconnected-camera positive control")
    source.add_argument("--input-depth", type=Path,
                        help="saved 1280x720 transformed depth (.npy/.npz)")
    source.add_argument("--live", action="store_true",
                        help="capture one pyk4a transformed_depth frame")
    parser.add_argument("--depth-unit", choices=("mm", "m"), default="mm")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--validate-runtime-audit", type=Path)
    parser.add_argument("--validate-bundle", type=Path)
    parser.add_argument("--validate-masks", type=Path)
    parser.add_argument("--validate-rerun-bundle", type=Path)
    parser.add_argument("--validate-pp-readiness", type=Path)
    parser.add_argument("--rerun-existing", type=Path,
                        help="complete D341 from an already generated numeric bundle")
    args = parser.parse_args()

    validators = [args.validate_runtime_audit, args.validate_bundle,
                  args.validate_masks, args.validate_rerun_bundle,
                  args.validate_pp_readiness]
    if sum(value is not None for value in validators) > 1:
        parser.error("choose only one validation mode")
    if args.validate_runtime_audit is not None:
        validate_runtime_audit(args.validate_runtime_audit)
        return 0
    if args.validate_bundle is not None:
        validate_bundle(args.validate_bundle)
        return 0
    if args.validate_masks is not None:
        validate_masks(args.validate_masks)
        return 0
    if args.validate_rerun_bundle is not None:
        validate_rerun_bundle(args.validate_rerun_bundle)
        return 0
    if args.validate_pp_readiness is not None:
        validate_pp_readiness(args.validate_pp_readiness)
        return 0
    if args.rerun_existing is not None:
        out = _require_heightmap_output(args.rerun_existing)
        results = rerun_existing_bundle(out)
        print(json.dumps({"verdict": "HM5_RERUN_RETRY_COMPLETE",
                          "rerun_pass": results["rerun"]["pass"]}, indent=2))
        return 0
    if not (args.synthetic or args.input_depth is not None or args.live):
        parser.error("choose --synthetic, --input-depth, or --live")

    out = _require_heightmap_output(args.output_dir)
    synthetic_masks: dict[str, np.ndarray] = {}
    if args.synthetic:
        depth, synthetic_masks, synth_meta = make_synthetic_depth(load_kinect_calib(CALIB_PATH))
        source_meta = {"source_kind": "synthetic_positive_control", **synth_meta}
        unit = "mm"
    elif args.input_depth is not None:
        depth, key = load_saved_depth(args.input_depth)
        source_meta = {
            "source_kind": "saved_transformed_depth",
            "path": str(args.input_depth.resolve()),
            "sha256": sha256_file(args.input_depth.resolve()),
            "array_key": key,
        }
        unit = args.depth_unit
    else:
        depth = capture_live_transformed_depth()
        source_meta = {
            "source_kind": "live_pyk4a_transformed_depth",
            "pyk4a_config": "RES_720P + NFOV_UNBINNED + synchronized_images_only",
        }
        unit = "mm"

    results = process_frame(depth, unit, source_meta, synthetic_masks, out, args.rerun)
    print(json.dumps({
        "verdict": "HM5_FRAME_TO_HEIGHTMAP_OK",
        "source": results["source"]["source_kind"],
        "out": str(out.relative_to(REPO)),
        "valid_cells": results["heightmap_observation"]["valid_cells"],
        "unseen_or_occluded_cells": results["heightmap_observation"]["unseen_or_occluded_cells"],
        "rerun_pass": results.get("rerun", {}).get("pass"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
