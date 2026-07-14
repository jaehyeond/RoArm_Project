#!/usr/bin/env python3
"""Build a presentation-only Rerun view of the D344 attempt3 colliders.

This script does not launch Isaac, write a USD asset, request a PhysX cook, or
advance physics.  It replays the immutable D347/D348 callback-face geometry at
the exact D349 zero-step body/object poses.  The exploded views are display-only
translations of the same per-part meshes and have no scientific authority.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
D349_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d349"
AUDIT_PATH = D349_DIR / "d349_d348_corrected_live_topology_audit.json"
BINDING_PATH = D349_DIR / "d349_live_topology_runtime_binding.json"
MEASUREMENT_PATH = D349_DIR / "d349_frozen_target_distance_measurement.json"
DEFAULT_OUT = REPO / "claudedocs/lab_meeting/20260715/attempt3_collider_decomposition"

EXPECTED_MEASUREMENT_SHA256 = (
    "5de6d14e37d6b74b202d1bb668120a6bb57221eac24ea5c751457ce9823b6300"
)
RERUN_VERSION = "0.34.1"
APP_ID = "roarm_labmeeting_attempt3_collider_decomposition"
RECORDING_ID = "20260715_attempt3_collider_decomposition_display_only"

BODY_ORDER = ("link5", "gripper_link")
CHANGED_PARTS = {
    "link5": {11, 18, 23, 24, 40, 41, 45, 54},
    "gripper_link": {0, 35, 36, 48, 57},
}


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quat_wxyz_to_rot(quat: list[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError(f"invalid quaternion: {quat}")
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _cylinder_mesh(
    center_m: np.ndarray,
    quat_wxyz: list[float],
    *,
    radius_m: float = 0.017,
    height_m: float = 0.090,
    sides: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    half = 0.5 * height_m
    vertices = []
    for z in (-half, half):
        vertices.extend([[radius_m * math.cos(a), radius_m * math.sin(a), z] for a in angles])
    vertices.extend([[0.0, 0.0, -half], [0.0, 0.0, half]])
    triangles: list[list[int]] = []
    bottom_center = 2 * sides
    top_center = bottom_center + 1
    for index in range(sides):
        nxt = (index + 1) % sides
        triangles.extend(
            [
                [index, nxt, sides + nxt],
                [index, sides + nxt, sides + index],
                [bottom_center, nxt, index],
                [top_center, sides + index, sides + nxt],
            ]
        )
    local = np.asarray(vertices, dtype=np.float64)
    world = (_quat_wxyz_to_rot(quat_wxyz) @ local.T).T + center_m
    return world, np.asarray(triangles, dtype=np.uint32)


def _color(body: str, index: int, *, alpha: int) -> list[int]:
    if index in CHANGED_PARTS[body]:
        return [255, 205, 35, alpha]
    phase = (index * 0.6180339887498949) % 1.0
    if body == "link5":
        hue = 0.48 + 0.25 * phase
    else:
        hue = (0.92 + 0.20 * phase) % 1.0
    saturation = 0.62 + 0.28 * ((index * 7) % 11) / 10.0
    value = 0.72 + 0.25 * ((index * 5) % 13) / 12.0
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return [int(round(255.0 * value)) for value in rgb] + [alpha]


def _camera(
    points: np.ndarray,
    *,
    azimuth_sign: float = -1.0,
    distance_scale: float = 0.50,
) -> rrb.EyeControls3D:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    target = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 0.05)
    position = target + distance_scale * np.asarray([1.45, 1.75 * azimuth_sign, 1.25]) * span
    return rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=position.tolist(),
        look_target=target.tolist(),
        eye_up=[0.0, 0.0, 1.0],
    )


def _mesh_parts(
    audit: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for body in BODY_ORDER:
        body_row = audit["per_body"][body]
        if not body_row["pass"]:
            raise RuntimeError(f"D348 corrected audit is not PASS for {body}")
        rows = sorted(body_row["part_checks"], key=lambda row: row["name"])
        if len(rows) != 64 or [row["name"] for row in rows] != [f"part_{i:03d}" for i in range(64)]:
            raise RuntimeError(f"{body}: expected exact part_000..part_063")
        parsed = []
        for index, row in enumerate(rows):
            if not row["pass"]:
                raise RuntimeError(f"{body}/{row['name']}: corrected audit FAIL")
            canonical = row["channel_consensus"]["instance"]["canonical"]
            vertices = np.asarray(canonical["vertices_m"], dtype=np.float64).reshape(-1, 3)
            triangles = np.asarray(canonical["triangles"], dtype=np.uint32).reshape(-1, 3)
            if vertices.shape[0] != int(canonical["vertex_count"]):
                raise RuntimeError(f"{body}/{row['name']}: vertex count mismatch")
            if triangles.shape[0] != int(canonical["triangle_count"]):
                raise RuntimeError(f"{body}/{row['name']}: triangle count mismatch")
            parsed.append(
                {
                    "body": body,
                    "index": index,
                    "name": row["name"],
                    "vertices_local_m": vertices,
                    "triangles": triangles,
                    "geometry_sha256": canonical["geometry_sha256"],
                }
            )
        result[body] = parsed
    return result


def _write_recording(output_dir: Path) -> dict[str, Any]:
    if str(rr.__version__) != RERUN_VERSION:
        raise RuntimeError(f"rerun-sdk {rr.__version__} != required {RERUN_VERSION}")

    rrd_path = output_dir / "attempt3_collider_decomposition.rrd"
    rbl_path = output_dir / "attempt3_collider_decomposition.rbl"
    screenshot_path = output_dir / "attempt3_collider_decomposition_rerun.png"
    validation_path = output_dir / "attempt3_collider_decomposition_validation.json"
    provenance_path = output_dir / "attempt3_collider_decomposition_provenance.json"
    for path in (rrd_path, rbl_path, screenshot_path, validation_path, provenance_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    measurement_sha = _sha256(MEASUREMENT_PATH)
    if measurement_sha != EXPECTED_MEASUREMENT_SHA256:
        raise RuntimeError("D349 measurement hash does not match the completed evidence")
    audit_sha = _sha256(AUDIT_PATH)
    binding = _json(BINDING_PATH)
    audit = _json(AUDIT_PATH)
    if (
        audit["d348_evidence"]["sha256"] != binding["d348_evidence_sha256"]
        or not audit["d348_evidence"]["pass"]
        or not binding["pass"]
    ):
        raise RuntimeError("D348 evidence/runtime binding contract failed")
    if not audit["pass"]:
        raise RuntimeError("D348 corrected audit is not PASS")
    measurement = _json(MEASUREMENT_PATH)
    parts = _mesh_parts(audit)

    authority = measurement["live_topology_surface_authority"]
    poses = authority["body_poses_w"]
    object_pos = np.asarray(authority["object_pos_w_m"], dtype=np.float64)
    object_quat = authority["object_quat_wxyz"]
    display_origin = object_pos.copy()

    actual_meshes: dict[str, list[dict[str, Any]]] = {body: [] for body in BODY_ORDER}
    actual_points = []
    exploded_meshes: dict[str, list[dict[str, Any]]] = {body: [] for body in BODY_ORDER}
    exploded_points: dict[str, list[np.ndarray]] = {body: [] for body in BODY_ORDER}

    for body in BODY_ORDER:
        pose = poses[body]
        rotation = _quat_wxyz_to_rot(pose["quat_wxyz"])
        translation = np.asarray(pose["pos_m"], dtype=np.float64)
        local_centroids = np.asarray(
            [row["vertices_local_m"].mean(axis=0) for row in parts[body]], dtype=np.float64
        )
        body_center = local_centroids.mean(axis=0)
        for row, centroid in zip(parts[body], local_centroids, strict=True):
            world = (rotation @ row["vertices_local_m"].T).T + translation - display_origin
            actual_meshes[body].append({**row, "vertices_m": world})
            actual_points.append(world)

            relative = centroid - body_center
            norm = float(np.linalg.norm(relative))
            if norm < 1e-12:
                angle = 2.0 * math.pi * row["index"] / 64.0
                direction = np.asarray([math.cos(angle), math.sin(angle), 0.35], dtype=np.float64)
                direction /= np.linalg.norm(direction)
            else:
                direction = relative / norm
            exploded = (
                row["vertices_local_m"]
                - body_center
                + 0.70 * relative
                + 0.0015 * direction
            )
            exploded_meshes[body].append({**row, "vertices_m": exploded})
            exploded_points[body].append(exploded)

    cylinder_vertices, cylinder_triangles = _cylinder_mesh(object_pos, object_quat)
    cylinder_vertices -= display_origin
    actual_points.append(cylinder_vertices)
    actual_cloud = np.concatenate(actual_points, axis=0)
    link5_cloud = np.concatenate(exploded_points["link5"], axis=0)
    gripper_cloud = np.concatenate(exploded_points["gripper_link"], axis=0)

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.TextDocumentView(
                    origin="/metadata/summary",
                    contents=["/metadata/summary"],
                    name="1 | Reading guide / scientific scope",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/scene/**"],
                    name="2 | D349 OPEN target: active live colliders (64 + 64), zero-step",
                    background=[7, 11, 18, 255],
                    line_grid=False,
                    eye_controls=_camera(actual_cloud, azimuth_sign=-1.0, distance_scale=0.43),
                ),
                column_shares=[0.27, 0.73],
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/exploded/link5/**"],
                    name="3 | link5: 64 active convex parts (DISPLAY EXPLODED)",
                    background=[7, 11, 18, 255],
                    line_grid=False,
                    eye_controls=_camera(link5_cloud, azimuth_sign=-1.0, distance_scale=0.42),
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=["/exploded/gripper_link/**"],
                    name="4 | gripper_link: 64 active convex parts (DISPLAY EXPLODED)",
                    background=[7, 11, 18, 255],
                    line_grid=False,
                    eye_controls=_camera(gripper_cloud, azimuth_sign=1.0, distance_scale=0.42),
                ),
                column_shares=[0.50, 0.50],
            ),
            row_shares=[0.60, 0.40],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )

    per_body_gate = measurement["distance_gate"]["per_body"]
    summary = "\n".join(
        [
            "# D344 attempt3 collider decomposition",
            "",
            "- **link5:** 64 active convex parts",
            "- **gripper_link:** 64 active convex parts",
            "- **total:** 128 active parts",
            "- **legacy:** 1 disabled full-mesh collider per body",
            "- **cool colors:** link5 parts",
            "- **warm colors:** gripper_link parts",
            "- **gold:** 13 fixed-point replacements (8 + 5)",
            "- **other 115:** bit-exact preserved",
            "",
            "## Frozen OPEN target",
            "",
            "- q5 = 1.5413 rad",
            "- cylinder = D34 x H90 mm",
            f"- link5 clearance = {per_body_gate['link5']['live_topology_exact_signed_distance_mm']:.4f} mm",
            f"- gripper clearance = {per_body_gate['gripper_link']['live_topology_exact_signed_distance_mm']:.4f} mm",
            "",
            "## Scope",
            "",
            "- source: D347 PhysX callback faces",
            "- topology: D348 corrected polygon faces",
            "- pose: D349 exact zero-step state",
            "- exploded panels offset positions for display only",
            "- **0 physics steps; not direct PhysX narrowphase or settle evidence**",
        ]
    )
    provenance = {
        "artifact": "LABMEETING_ATTEMPT3_COLLIDER_DECOMPOSITION_DISPLAY_V1",
        "scientific_authority": "immutable D347-D349 JSON and hashes, not this display",
        "presentation_only": True,
        "isaac_launched": False,
        "asset_write_count": 0,
        "cook_request_count": 0,
        "physics_step_count": 0,
        "exploded_view_has_physical_authority": False,
        "source_hashes": {
            str(AUDIT_PATH.relative_to(REPO)): audit_sha,
            str(BINDING_PATH.relative_to(REPO)): _sha256(BINDING_PATH),
            str(MEASUREMENT_PATH.relative_to(REPO)): measurement_sha,
        },
        "part_counts": {body: len(parts[body]) for body in BODY_ORDER},
        "fixed_point_replacements": {
            body: [f"part_{index:03d}" for index in sorted(CHANGED_PARTS[body])]
            for body in BODY_ORDER
        },
        "world_display_translation_removed_m": display_origin.tolist(),
        "rerun_sdk_version": str(rr.__version__),
    }

    expected_entities = {"/metadata/summary", "/metadata/provenance"}
    component_contract: dict[str, list[str]] = {
        "metadata/summary": ["TextDocument:media_type", "TextDocument:text"],
        "metadata/provenance": ["TextDocument:text"],
    }

    with rr.RecordingStream(
        APP_ID,
        recording_id=RECORDING_ID,
        make_default=False,
        send_properties=True,
    ) as recording:
        recording.save(str(rrd_path), write_footer=True)
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.log(
            "metadata/summary",
            rr.TextDocument(summary, media_type=rr.MediaType.MARKDOWN),
            static=True,
        )
        recording.log(
            "metadata/provenance",
            rr.TextDocument(json.dumps(provenance, indent=2, sort_keys=True)),
            static=True,
        )

        for body in BODY_ORDER:
            for row in actual_meshes[body]:
                path = f"scene/live/{body}/{row['name']}"
                recording.log(
                    path,
                    rr.Mesh3D(
                        vertex_positions=row["vertices_m"].astype(np.float32),
                        triangle_indices=row["triangles"],
                        albedo_factor=_color(body, row["index"], alpha=238),
                    ),
                    static=True,
                )
                expected_entities.add(f"/{path}")
                component_contract[path] = [
                    "Mesh3D:albedo_factor",
                    "Mesh3D:triangle_indices",
                    "Mesh3D:vertex_positions",
                ]

            for row in exploded_meshes[body]:
                path = f"exploded/{body}/{row['name']}"
                recording.log(
                    path,
                    rr.Mesh3D(
                        vertex_positions=row["vertices_m"].astype(np.float32),
                        triangle_indices=row["triangles"],
                        albedo_factor=_color(body, row["index"], alpha=255),
                    ),
                    static=True,
                )
                expected_entities.add(f"/{path}")
                component_contract[path] = [
                    "Mesh3D:albedo_factor",
                    "Mesh3D:triangle_indices",
                    "Mesh3D:vertex_positions",
                ]

        cylinder_path = "scene/target/cylinder_D34xH90"
        recording.log(
            cylinder_path,
            rr.Mesh3D(
                vertex_positions=cylinder_vertices.astype(np.float32),
                triangle_indices=cylinder_triangles,
                albedo_factor=[255, 145, 28, 132],
            ),
            static=True,
        )
        expected_entities.add(f"/{cylinder_path}")
        component_contract[cylinder_path] = [
            "Mesh3D:albedo_factor",
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        ]

        recording.flush(timeout_sec=30.0)

    blueprint.save(APP_ID, rbl_path)

    from roarm_rl.rerun_contract import validate_rerun_artifact

    validation = validate_rerun_artifact(
        rrd_path,
        expected_entity_paths=sorted(expected_entities),
        exact_entity_paths=sorted(expected_entities),
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=component_contract,
        blueprint_path=rbl_path,
        screenshot_path=screenshot_path,
        screenshot_window_size="2400x1350",
        expected_version=RERUN_VERSION,
        timeout_s=180.0,
    )
    with validation_path.open("x", encoding="utf-8") as stream:
        json.dump(validation, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if not validation["pass"]:
        raise RuntimeError(f"Rerun validation failed: {validation['errors']}")

    provenance.update(
        {
            "output_hashes": {
                str(rrd_path.name): _sha256(rrd_path),
                str(rbl_path.name): _sha256(rbl_path),
                str(screenshot_path.name): _sha256(screenshot_path),
                str(validation_path.name): _sha256(validation_path),
            },
            "screenshot_dimensions_expected": [4800, 2700],
            "rerun_validation_pass": True,
        }
    )
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {
        "output_dir": str(output_dir),
        "rrd": str(rrd_path),
        "rbl": str(rbl_path),
        "screenshot": str(screenshot_path),
        "validation": str(validation_path),
        "provenance": str(provenance_path),
        "validation_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(_write_recording(args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
