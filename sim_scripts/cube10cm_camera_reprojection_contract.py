#!/usr/bin/env python3
"""Static pinhole reprojection check for the D232 cube10cm top-view camera.

This script does not launch Isaac Sim, render images, or generate a dataset. It
checks whether the current camera-contract candidate projects table/cube corner
points inside the raw 1280x720 image with usable margins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs" / "runtime_logs" / "20260526_cube3cm_push_rollout_probe_20480"

INTRINSICS = {
    "fx": 608.33,
    "fy": 608.28,
    "cx": 638.31,
    "cy": 365.26,
    "width": 1280,
    "height": 720,
}
TABLE_CENTER = (0.25, 0.0)
TABLE_SIZE = (0.90, 0.70)
TABLE_Z_TOP = -0.012117
CAMERA_HEIGHT_ABOVE_TABLE = 0.65
CAMERA_CENTER = (
    TABLE_CENTER[0],
    TABLE_CENTER[1],
    TABLE_Z_TOP + CAMERA_HEIGHT_ABOVE_TABLE,
)
CUBE_SIZE = 0.10
CUBE_Z_TOP = TABLE_Z_TOP + CUBE_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=LOG_DIR / "cube10cm_camera_reprojection_contract_d232.json",
    )
    return parser.parse_args()


def project_top_view(point_xyz: tuple[float, float, float]) -> dict:
    """Project world XYZ with image right=world +X and image down=world -Y."""
    x_w, y_w, z_w = point_xyz
    x_c = x_w - CAMERA_CENTER[0]
    y_c = -(y_w - CAMERA_CENTER[1])
    z_c = CAMERA_CENTER[2] - z_w
    if z_c <= 0:
        raise ValueError(f"point is behind camera: {point_xyz}")
    u = INTRINSICS["fx"] * x_c / z_c + INTRINSICS["cx"]
    v = INTRINSICS["fy"] * y_c / z_c + INTRINSICS["cy"]
    return {
        "world_xyz_m": [x_w, y_w, z_w],
        "camera_xyz_m": [x_c, y_c, z_c],
        "uv_px": [u, v],
        "inside": 0.0 <= u < INTRINSICS["width"] and 0.0 <= v < INTRINSICS["height"],
    }


def rect_corners(center_xy: tuple[float, float], size_xy: tuple[float, float], z: float) -> dict[str, tuple[float, float, float]]:
    cx, cy = center_xy
    sx, sy = size_xy
    return {
        "xmin_ymin": (cx - sx / 2, cy - sy / 2, z),
        "xmin_ymax": (cx - sx / 2, cy + sy / 2, z),
        "xmax_ymin": (cx + sx / 2, cy - sy / 2, z),
        "xmax_ymax": (cx + sx / 2, cy + sy / 2, z),
    }


def cube_top_corners(cube_xy: tuple[float, float]) -> dict[str, tuple[float, float, float]]:
    return rect_corners(cube_xy, (CUBE_SIZE, CUBE_SIZE), CUBE_Z_TOP)


def summarize_margin(projections: list[dict]) -> dict:
    us = [p["uv_px"][0] for p in projections]
    vs = [p["uv_px"][1] for p in projections]
    return {
        "u_min": min(us),
        "u_max": max(us),
        "v_min": min(vs),
        "v_max": max(vs),
        "left_margin_px": min(us),
        "right_margin_px": INTRINSICS["width"] - max(us),
        "top_margin_px": min(vs),
        "bottom_margin_px": INTRINSICS["height"] - max(vs),
        "all_inside": all(p["inside"] for p in projections),
    }


def main() -> None:
    args = parse_args()

    points: dict[str, dict[str, tuple[float, float, float]]] = {
        "table_corners": rect_corners(TABLE_CENTER, TABLE_SIZE, TABLE_Z_TOP),
        "d230_xy10_workspace_cube_top_outer": {},
        "env_default_workspace_cube_top_outer": {},
    }

    for name, xy in {
        "x014_yneg010": (0.14, -0.10),
        "x014_y010": (0.14, 0.10),
        "x034_yneg010": (0.34, -0.10),
        "x034_y010": (0.34, 0.10),
    }.items():
        for corner_name, xyz in cube_top_corners(xy).items():
            points["d230_xy10_workspace_cube_top_outer"][f"{name}_{corner_name}"] = xyz

    for name, xy in {
        "x0205_yneg0125": (0.205, -0.125),
        "x0205_y0125": (0.205, 0.125),
        "x0360_yneg0125": (0.360, -0.125),
        "x0360_y0125": (0.360, 0.125),
    }.items():
        for corner_name, xyz in cube_top_corners(xy).items():
            points["env_default_workspace_cube_top_outer"][f"{name}_{corner_name}"] = xyz

    result = {
        "artifact": "cube10cm_camera_reprojection_contract_d232",
        "runtime": "NO_RENDER_NO_ISAAC_NO_DATASET",
        "intrinsics": INTRINSICS,
        "camera_contract_id": "cube10cm_top_view_v1_candidate",
        "camera_center_world_m": list(CAMERA_CENTER),
        "table_to_camera_height_m": CAMERA_HEIGHT_ABOVE_TABLE,
        "image_convention": "image_right_world_pos_x__image_down_world_neg_y",
        "sets": {},
    }

    for set_name, set_points in points.items():
        projections = {name: project_top_view(xyz) for name, xyz in set_points.items()}
        result["sets"][set_name] = {
            "points": projections,
            "margin": summarize_margin(list(projections.values())),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")

    print("line1 artifact=cube10cm_camera_reprojection_contract_d232 runtime=NO_RENDER_NO_ISAAC_NO_DATASET")
    print(
        "line2 camera "
        f"contract={result['camera_contract_id']} center={result['camera_center_world_m']} "
        f"height={CAMERA_HEIGHT_ABOVE_TABLE} convention={result['image_convention']}"
    )
    for set_name, data in result["sets"].items():
        margin = data["margin"]
        print(
            f"line3 set={set_name} all_inside={margin['all_inside']} "
            f"u=[{margin['u_min']:.1f},{margin['u_max']:.1f}] "
            f"v=[{margin['v_min']:.1f},{margin['v_max']:.1f}] "
            f"margins_lrtb=[{margin['left_margin_px']:.1f},{margin['right_margin_px']:.1f},"
            f"{margin['top_margin_px']:.1f},{margin['bottom_margin_px']:.1f}]"
        )
    print(f"line4 out_json={args.out_json}")


if __name__ == "__main__":
    main()
