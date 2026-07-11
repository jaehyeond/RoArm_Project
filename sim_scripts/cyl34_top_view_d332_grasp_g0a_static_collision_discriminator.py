#!/usr/bin/env python3
"""D332 canonical static collision discriminator for cylinder G0a.

The probe has two intentionally sequential stages:

1. ``offline`` computes raw-mesh and unrestricted mathematical-hull signed
   distance to the D34 x H90 cylinder at one deterministic HOME-seeded IK pose.
2. ``runtime`` recooks an exact live-stage source-mesh mirror with the default
   PhysX convex-hull policy, teleports to the same command, and records a
   cylinder-owned contact witness plus full object motion during settling.

It does not change collision geometry, the G0a target/gates, object physics,
the open gripper command, or the variable ladder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.viz_debug import draw_frames, frame_from_axes, log_rerun, snapshot_frame_plot
from sim_scripts.cube10cm_top_view_d323_grasp_g0a_frame_repair_probe import (
    HOME_DEG,
    _fk_runtime_tcp,
    _quat_wxyz_to_rot,
    _solve_runtime_ik,
)


DEFAULT_OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d332"
DEFAULT_ROBOT_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
DEFAULT_URDF = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
LINK5_MESH = REPO / "local_assets/roarm_m3/urdf/meshes/link5.stl"

ARM_JOINT_NAMES = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
)
GRIPPER_JOINT_NAME = "link5_to_gripper_link"
ALL_JOINT_NAMES = ARM_JOINT_NAMES + (GRIPPER_JOINT_NAME,)

TABLE_Z_M = -0.012117
OBJECT_CENTER_LOCAL_M = np.asarray([0.300, 0.000, TABLE_Z_M + 0.045], dtype=np.float64)
CYLINDER_RADIUS_M = 0.017
CYLINDER_HEIGHT_M = 0.090
OBJECT_MASS_KG = 0.72
STATIC_FRICTION = 1.5
DYNAMIC_FRICTION = 1.2
RADIAL_CENTER_OFFSET_M = 0.007
TANGENT_CENTER_OFFSET_M = 0.011
ADOPTED_TANGENT_SIGN = -1.0
MESH_SCALE_M_PER_UNIT = 0.001

BASELINE_PHYSICS_STEPS = 200
TARGET_SETTLE_PHYSICS_STEPS = 200
PHYSICS_DT_S = 1.0 / 200.0
BASELINE_TAIL_STEPS = 50
SUPPORT_POSITIVE_CONTROL_N = 1.0
ROBOT_FORCE_EVENT_N = 0.1
DISTURBANCE_XY_M = 0.0005
DISTURBANCE_TILT_DEG = 1.0
CONSECUTIVE_EVENT_STEPS = 2
SIGNED_DISTANCE_BORDER_M = 0.0001

FILTER_LABELS = ("support_plane", "link4", "link5", "gripper_link")
FILTER_PATHS = (
    "/World/ground",
    "/World/envs/env_.*/Robot/link4",
    "/World/envs/env_.*/Robot/link5",
    "/World/envs/env_.*/Robot/gripper_link",
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module(name)
        value = getattr(module, "__version__", None)
        return None if value is None else str(value)


def _unit(value: Any, fallback: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def _canonical_contract() -> dict[str, Any]:
    radial = _unit([OBJECT_CENTER_LOCAL_M[0], OBJECT_CENTER_LOCAL_M[1], 0.0])
    tangent = np.asarray([-radial[1], radial[0], 0.0], dtype=np.float64) * ADOPTED_TANGENT_SIGN
    target_tcp = OBJECT_CENTER_LOCAL_M.copy()
    target_tcp -= radial * RADIAL_CENTER_OFFSET_M
    target_tcp -= tangent * TANGENT_CENTER_OFFSET_M
    target_tcp[2] = OBJECT_CENTER_LOCAL_M[2]

    ik = _solve_runtime_ik(
        target_tcp,
        HOME_DEG,
        target_x_axis=None,
        target_z_axis=None,
        max_iter=120,
        pos_tol_mm=1.0,
    )
    if not bool(ik["converged"]):
        raise RuntimeError(f"canonical HOME-seeded IK did not converge: {ik}")
    q_deg = np.asarray(ik["q_deg"], dtype=np.float64)
    q_deg[5] = 0.0
    tcp, link5_pos, link5_rot = _fk_runtime_tcp(q_deg)
    return {
        "object_center_local_m": OBJECT_CENTER_LOCAL_M.tolist(),
        "radial_axis": radial.tolist(),
        "tangent_axis": tangent.tolist(),
        "target_tcp_local_m": target_tcp.tolist(),
        "home_seed_deg": HOME_DEG.tolist(),
        "joint_names": list(ALL_JOINT_NAMES),
        "commanded_joint_deg": q_deg.tolist(),
        "commanded_joint_rad": np.radians(q_deg).tolist(),
        "commanded_tcp_local_m": tcp.tolist(),
        "commanded_link5_pos_local_m": link5_pos.tolist(),
        "commanded_link5_rot_local": link5_rot.tolist(),
        "commanded_tcp_error_mm": float(np.linalg.norm(tcp - target_tcp) * 1000.0),
        "ik": ik,
        "target_formula": "TCP=center-radial*0.007-tangent*0.011; z=center_z",
    }


def _fcl_points(hppfcl: Any, vertices: np.ndarray) -> Any:
    points = hppfcl.StdVec_Vec3f()
    for vertex in np.asarray(vertices, dtype=np.float64):
        points.append(vertex)
    return points


def _fcl_triangles(hppfcl: Any, faces: np.ndarray) -> Any:
    triangles = hppfcl.StdVec_Triangle()
    for face in np.asarray(faces, dtype=np.int64):
        triangles.append(hppfcl.Triangle(int(face[0]), int(face[1]), int(face[2])))
    return triangles


def _build_raw_bvh(hppfcl: Any, vertices: np.ndarray, faces: np.ndarray) -> Any:
    model = hppfcl.BVHModelOBBRSS()
    codes = {
        "begin": int(model.beginModel(int(len(faces)), int(len(vertices)))),
        "vertices": int(model.addVertices(np.asarray(vertices, dtype=np.float64))),
        "triangles": int(model.addTriangles(np.asarray(faces, dtype=np.int64))),
    }
    codes["end"] = int(model.endModel())
    if any(value != 0 for value in codes.values()):
        raise RuntimeError(f"hppfcl raw BVH build failed: {codes}")
    return model


def _fcl_query(hppfcl: Any, geometry: Any, transform: Any, cylinder: Any, cylinder_tf: Any) -> dict[str, Any]:
    distance_request = hppfcl.DistanceRequest(True, 1.0e-9, 1.0e-9)
    distance_request.gjk_tolerance = 1.0e-9
    distance_request.gjk_max_iterations = 1000
    distance_result = hppfcl.DistanceResult()
    signed_distance_m = float(
        hppfcl.distance(geometry, transform, cylinder, cylinder_tf, distance_request, distance_result)
    )

    collision_request = hppfcl.CollisionRequest()
    collision_request.enable_contact = True
    collision_request.num_max_contacts = 16
    collision_result = hppfcl.CollisionResult()
    collision_count = int(
        hppfcl.collide(geometry, transform, cylinder, cylinder_tf, collision_request, collision_result)
    )
    contact = None
    if collision_result.numContacts() > 0:
        item = collision_result.getContact(0)
        contact = {
            "penetration_depth_m": float(item.penetration_depth),
            "normal": np.asarray(item.normal, dtype=np.float64).tolist(),
            "position_m": np.asarray(item.pos, dtype=np.float64).tolist(),
        }
    return {
        "signed_distance_m": signed_distance_m,
        "signed_distance_mm": signed_distance_m * 1000.0,
        "nearest_point_geometry_m": np.asarray(distance_result.getNearestPoint1(), dtype=np.float64).tolist(),
        "nearest_point_cylinder_m": np.asarray(distance_result.getNearestPoint2(), dtype=np.float64).tolist(),
        "distance_normal": np.asarray(distance_result.normal, dtype=np.float64).tolist(),
        "collision_count": collision_count,
        "is_collision": bool(collision_result.isCollision()),
        "contact": contact,
    }


def _signed_distance_verdict(distance_m: float) -> str:
    if distance_m <= -SIGNED_DISTANCE_BORDER_M:
        return "OVERLAP"
    if distance_m >= SIGNED_DISTANCE_BORDER_M:
        return "CLEAR"
    return "BORDERLINE"


def _set_axes_equal(ax: Any, points: np.ndarray, margin_m: float = 0.012) -> None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo)) * 0.5 + margin_m, 0.03)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _write_offline_figure(
    path: Path,
    *,
    hull_vertices_world: np.ndarray,
    hull_faces: np.ndarray,
    canonical: dict[str, Any],
    result: dict[str, Any],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.0, 8.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    triangles = hull_vertices_world[np.asarray(hull_faces, dtype=np.int64)]
    hull_collection = Poly3DCollection(
        triangles,
        facecolors=(0.76, 0.20, 0.18, 0.22),
        edgecolors=(0.52, 0.10, 0.08, 0.28),
        linewidths=0.25,
    )
    ax.add_collection3d(hull_collection)

    center = OBJECT_CENTER_LOCAL_M
    theta = np.linspace(0.0, 2.0 * math.pi, 96)
    z = np.linspace(center[2] - 0.5 * CYLINDER_HEIGHT_M, center[2] + 0.5 * CYLINDER_HEIGHT_M, 12)
    tt, zz = np.meshgrid(theta, z)
    xx = center[0] + CYLINDER_RADIUS_M * np.cos(tt)
    yy = center[1] + CYLINDER_RADIUS_M * np.sin(tt)
    ax.plot_surface(xx, yy, zz, color=(0.90, 0.55, 0.12), alpha=0.33, linewidth=0.0)

    p_hull = np.asarray(result["nearest_point_geometry_m"], dtype=np.float64)
    p_cyl = np.asarray(result["nearest_point_cylinder_m"], dtype=np.float64)
    contact = result.get("contact") or {}
    p_contact = np.asarray(contact.get("position_m", p_cyl), dtype=np.float64)
    ax.scatter(*p_hull, color="#9e1b1b", s=42, label="hull witness")
    ax.scatter(*p_cyl, color="#d47a00", s=42, label="cylinder witness")
    ax.scatter(*p_contact, color="#111111", s=28, marker="x", label="EPA contact")
    ax.plot(*np.vstack([p_hull, p_cyl]).T, color="#111111", linewidth=1.5)

    link5_pos = np.asarray(canonical["commanded_link5_pos_local_m"], dtype=np.float64)
    link5_rot = np.asarray(canonical["commanded_link5_rot_local"], dtype=np.float64)
    for idx, color in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
        delta = link5_rot[:, idx] * 0.025
        ax.quiver(*link5_pos, *delta, color=color, arrow_length_ratio=0.18, linewidth=1.5)
    target_tcp = np.asarray(canonical["target_tcp_local_m"], dtype=np.float64)
    ax.scatter(*target_tcp, color="#005bbb", marker="*", s=90, label="target TCP")
    ax.scatter(*link5_pos, color="#7d1f8a", marker="o", s=32, label="link5 origin")

    ax.set_xlabel("env-local x (m)")
    ax.set_ylabel("env-local y (m)")
    ax.set_zlabel("env-local z (m)")
    ax.set_title("D332 default PhysX mirror recook of live-stage link5 mesh vs D34 x H90 cylinder")
    ax.view_init(elev=22.0, azim=-67.0)
    _set_axes_equal(ax, np.vstack([hull_vertices_world, p_hull, p_cyl, center, target_tcp]))
    ax.legend(loc="upper left", fontsize=8)
    fig.text(
        0.02,
        0.02,
        "signed distance = "
        f"{result['signed_distance_mm']:.6f} mm ({result['verdict']}); "
        f"raw STL control = {result['raw_signed_distance_mm']:.6f} mm\n"
        "Geometry is a default PhysX mirror recook, not a direct live-collider extraction; AABB is not used.\n"
        "Runtime reset places the cylinder bottom 12.117 mm inside the global ground plane.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_canonical_csv(path: Path, canonical: dict[str, Any]) -> None:
    q_deg = canonical["commanded_joint_deg"]
    q_rad = canonical["commanded_joint_rad"]
    row: dict[str, Any] = {"env": 0, "seed_policy": "exact_home_seed_no_jitter"}
    for idx, name in enumerate(ALL_JOINT_NAMES):
        row[f"commanded_{name}_rad"] = q_rad[idx]
        row[f"commanded_{name}_deg"] = q_deg[idx]
    for idx, axis in enumerate("xyz"):
        row[f"target_tcp_{axis}_m"] = canonical["target_tcp_local_m"][idx]
        row[f"commanded_tcp_{axis}_m"] = canonical["commanded_tcp_local_m"][idx]
    row["commanded_tcp_error_mm"] = canonical["commanded_tcp_error_mm"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _run_offline(args: argparse.Namespace) -> dict[str, Any]:
    import hppfcl
    import scipy
    import trimesh
    from scipy.spatial import ConvexHull

    args.out_dir.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_contract()
    mesh = trimesh.load_mesh(LINK5_MESH, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected one Trimesh from {LINK5_MESH}, got {type(mesh)!r}")
    raw_vertices = np.asarray(mesh.vertices, dtype=np.float64) * MESH_SCALE_M_PER_UNIT
    raw_faces = np.asarray(mesh.faces, dtype=np.int64)
    unique_vertices = np.unique(raw_vertices, axis=0)
    scipy_hull = ConvexHull(unique_vertices)

    points = _fcl_points(hppfcl, unique_vertices)
    convex_geometry = hppfcl.Convex.convexHull(points, False, "")
    if convex_geometry is None:
        raise RuntimeError("hppfcl Qhull convexHull returned None")
    raw_geometry = _build_raw_bvh(hppfcl, raw_vertices, raw_faces)
    cylinder = hppfcl.Cylinder(CYLINDER_RADIUS_M, CYLINDER_HEIGHT_M)

    link5_pos = np.asarray(canonical["commanded_link5_pos_local_m"], dtype=np.float64)
    link5_rot = np.asarray(canonical["commanded_link5_rot_local"], dtype=np.float64)
    link5_tf = hppfcl.Transform3f(link5_rot, link5_pos)
    cylinder_tf = hppfcl.Transform3f(np.eye(3, dtype=np.float64), OBJECT_CENTER_LOCAL_M)
    convex_query = _fcl_query(hppfcl, convex_geometry, link5_tf, cylinder, cylinder_tf)
    raw_query = _fcl_query(hppfcl, raw_geometry, link5_tf, cylinder, cylinder_tf)
    convex_query["verdict"] = _signed_distance_verdict(float(convex_query["signed_distance_m"]))
    raw_query["verdict"] = _signed_distance_verdict(float(raw_query["signed_distance_m"]))

    epa_depth_m = None
    if convex_query["contact"] is not None:
        epa_depth_m = float(convex_query["contact"]["penetration_depth_m"])
    sign_consistent = bool(
        (float(convex_query["signed_distance_m"]) < 0.0) == bool(convex_query["is_collision"])
    )
    depth_consistent = bool(
        epa_depth_m is None
        or abs(epa_depth_m + float(convex_query["signed_distance_m"])) <= SIGNED_DISTANCE_BORDER_M
    )
    if not sign_consistent or not depth_consistent:
        raise RuntimeError(
            "hppfcl signed distance/collision disagreement: "
            f"distance={convex_query['signed_distance_m']} contact={convex_query['contact']}"
        )

    hull_vertices_world = (link5_rot @ unique_vertices.T).T + link5_pos
    payload = {
        "artifact": "D332_OFFLINE_GEOMETRY_PRECHECK",
        "new_variable": [],
        "canonical": canonical,
        "geometry_contract": {
            "link5_mesh": _rel(LINK5_MESH),
            "link5_mesh_sha256": _sha256(LINK5_MESH),
            "urdf": _rel(DEFAULT_URDF),
            "urdf_sha256": _sha256(DEFAULT_URDF),
            "robot_usd": _rel(args.robot_usd_path),
            "robot_usd_sha256": _sha256(args.robot_usd_path),
            "urdf_collision_origin": {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]},
            "mesh_scale_m_per_unit": MESH_SCALE_M_PER_UNIT,
            "raw_vertex_count": int(len(raw_vertices)),
            "raw_face_count": int(len(raw_faces)),
            "raw_unique_vertex_count": int(len(unique_vertices)),
            "raw_bounds_m": np.vstack([raw_vertices.min(axis=0), raw_vertices.max(axis=0)]).tolist(),
            "scipy_hull_vertex_count": int(len(scipy_hull.vertices)),
            "scipy_hull_face_count": int(len(scipy_hull.simplices)),
            "hppfcl_hull_point_count": int(convex_geometry.num_points),
            "method": "unrestricted mathematical Qhull + hppfcl GJK/EPA; default PhysX mirror recook deferred to runtime",
            "physx_default_hull_vertex_limit": 64,
            "representation_warning": "mathematical full hull is not a mirror recook or a direct live-collider extraction",
        },
        "cylinder_contract": {
            "center_local_m": OBJECT_CENTER_LOCAL_M.tolist(),
            "radius_m": CYLINDER_RADIUS_M,
            "height_m": CYLINDER_HEIGHT_M,
            "axis": "Z",
        },
        "signed_distance_thresholds_mm": {
            "overlap_at_or_below": -SIGNED_DISTANCE_BORDER_M * 1000.0,
            "clear_at_or_above": SIGNED_DISTANCE_BORDER_M * 1000.0,
        },
        "mathematical_full_hull": convex_query,
        "raw_stl_negative_control": raw_query,
        "cross_checks": {
            "signed_distance_collision_sign_consistent": sign_consistent,
            "gjk_epa_depth_consistent_within_0p1mm": depth_consistent,
            "raw_clear_and_hull_overlap": bool(
                raw_query["verdict"] == "CLEAR" and convex_query["verdict"] == "OVERLAP"
            ),
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
            "hppfcl": getattr(hppfcl, "__version__", None),
        },
        "visualization": {
            "offline_snapshot": None,
            "note": "decision snapshot is written after the default PhysX mirror recook in runtime stage",
        },
        "stop_rule": {
            "default_physx_mirror_recook_required": True,
            "nullspace_scan_required_now": False,
        },
    }
    offline_json = args.out_dir / "d332_offline_geometry_precheck.json"
    canonical_csv = args.out_dir / "d332_canonical_joint_targets.csv"
    _json_dump(offline_json, payload)
    _write_canonical_csv(canonical_csv, canonical)
    payload["artifacts"] = {
        "offline_json": _rel(offline_json),
        "canonical_joint_csv": _rel(canonical_csv),
        "offline_snapshot": None,
    }
    _json_dump(offline_json, payload)
    return payload


def _configure_runtime_env(args: argparse.Namespace) -> Any:
    import isaaclab.sim as sim_utils
    from pxr import PhysxSchema
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnvCfg
    from roarm_rl.roarm_stack_env import TABLE_Z

    if not math.isclose(float(TABLE_Z), TABLE_Z_M, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"TABLE_Z changed: runtime={TABLE_Z} preregistered={TABLE_Z_M}")

    env_cfg = RoArmCubeTap10cmEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args.seed)
    env_cfg.robot.spawn.usd_path = str(args.robot_usd_path)
    env_cfg.robot.spawn.activate_contact_sensors = False
    env_cfg.episode_length_s = 4.0
    env_cfg.cube_x_min = float(OBJECT_CENTER_LOCAL_M[0])
    env_cfg.cube_x_max = float(OBJECT_CENTER_LOCAL_M[0])
    env_cfg.cube_y_min = float(OBJECT_CENTER_LOCAL_M[1])
    env_cfg.cube_y_max = float(OBJECT_CENTER_LOCAL_M[1])
    env_cfg.cube_size_x_m = 2.0 * CYLINDER_RADIUS_M
    env_cfg.cube_size_y_m = 2.0 * CYLINDER_RADIUS_M
    env_cfg.cube_size_z_m = CYLINDER_HEIGHT_M

    old_spawn = env_cfg.sponge.spawn
    cylinder_spawn = sim_utils.CylinderCfg(
        radius=CYLINDER_RADIUS_M,
        height=CYLINDER_HEIGHT_M,
        axis="Z",
        rigid_props=old_spawn.rigid_props,
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS_KG),
        collision_props=old_spawn.collision_props,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=STATIC_FRICTION,
            dynamic_friction=DYNAMIC_FRICTION,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.55, 0.20), metallic=0.0),
        activate_contact_sensors=False,
    )
    base_spawn = cylinder_spawn.func

    def _spawn_cylinder_with_zero_reporter(
        prim_path: str,
        cfg: Any,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        prim = base_spawn(
            prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        sim_utils.activate_contact_sensors(prim.GetPath().pathString, threshold=0.0)
        api = PhysxSchema.PhysxContactReportAPI.Get(prim.GetStage(), prim.GetPath())
        threshold = api.GetThresholdAttr().Get()
        if threshold is None or abs(float(threshold)) > 1.0e-12:
            raise RuntimeError(f"unexpected cylinder reporter threshold after spawn: {threshold}")
        return prim

    cylinder_spawn.func = _spawn_cylinder_with_zero_reporter
    env_cfg.sponge.spawn = cylinder_spawn
    env_cfg.sponge.init_state.pos = tuple(float(v) for v in OBJECT_CENTER_LOCAL_M)
    env_cfg.sponge.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env_cfg.fixed_push_dir_x = 1.0
    env_cfg.fixed_push_dir_y = 0.0
    env_cfg.ik_endpoint_reset = False
    env_cfg.rl_action_mode = "joint_delta"
    env_cfg.bc_teacher_checkpoint_path = ""
    env_cfg.bc_teacher_blend = 0.0
    env_cfg.bc_teacher_imitation_reward_scale = 0.0
    return env_cfg


def _make_runtime_env(args: argparse.Namespace) -> Any:
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from roarm_rl.roarm_cube_push_env import RoArmCubeTap10cmEnv

    class D332CylinderWitnessEnv(RoArmCubeTap10cmEnv):
        def _setup_scene(self) -> None:
            super()._setup_scene()
            sensor_cfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Sponge",
                filter_prim_paths_expr=list(FILTER_PATHS),
                update_period=0.0,
                history_length=1,
                track_pose=True,
                track_contact_points=True,
                max_contact_data_count_per_prim=16,
                force_threshold=0.0,
                debug_vis=False,
            )
            sensor = ContactSensor(sensor_cfg)
            self.scene.sensors["d332_cylinder_contact"] = sensor
            self._d332_contact_sensor = sensor

    env_cfg = _configure_runtime_env(args)
    return D332CylinderWitnessEnv(cfg=env_cfg)


def _joint_target_tensor(inner: Any, q_rad: np.ndarray) -> Any:
    import torch

    target = inner._robot.data.joint_pos.detach().clone()
    for idx, name in enumerate(ALL_JOINT_NAMES):
        ids, names = inner._robot.find_joints(name)
        if len(ids) != 1 or list(names) != [name]:
            raise RuntimeError(f"joint path contract failed for {name}: ids={ids}, names={names}")
        target[:, int(ids[0])] = float(q_rad[idx])
    return target.to(device=inner.device, dtype=torch.float32)


def _write_exact_state(inner: Any, q_rad: np.ndarray, object_local_m: np.ndarray) -> Any:
    import torch

    env_ids = torch.arange(inner.num_envs, device=inner.device, dtype=torch.long)
    target = _joint_target_tensor(inner, q_rad)
    zero_joint_vel = torch.zeros_like(target)
    inner._robot.write_joint_state_to_sim(target, zero_joint_vel, env_ids=env_ids)
    inner._robot.set_joint_position_target(target, env_ids=env_ids)
    inner.robot_dof_targets[env_ids] = target
    inner._external_joint_targets_override = target.detach().clone()

    pose = torch.zeros((inner.num_envs, 7), device=inner.device, dtype=torch.float32)
    pose[:, :3] = inner.scene.env_origins + torch.tensor(
        object_local_m, device=inner.device, dtype=torch.float32
    ).unsqueeze(0)
    pose[:, 3] = 1.0
    velocity = torch.zeros((inner.num_envs, 6), device=inner.device, dtype=torch.float32)
    inner._sponge.write_root_pose_to_sim(pose, env_ids=env_ids)
    inner._sponge.write_root_velocity_to_sim(velocity, env_ids=env_ids)
    inner.scene.write_data_to_sim()
    inner.sim.forward()
    inner.scene.update(dt=0.0)
    inner._compute_intermediate_values()
    inner._d332_contact_sensor.reset(env_ids)
    return target


def _physics_step(inner: Any) -> None:
    inner._sim_step_counter += 1
    inner.scene.write_data_to_sim()
    inner.sim.step(render=False)
    inner.scene.update(dt=inner.physics_dt)
    inner._compute_intermediate_values()


def _resolved_filter_map(sensor: Any) -> tuple[list[str], dict[str, int]]:
    outer = list(sensor.contact_physx_view.filter_paths)
    if len(outer) == 1 and not isinstance(outer[0], (str, bytes)):
        try:
            raw = [str(item) for item in list(outer[0])]
        except TypeError:
            raw = [str(item) for item in outer]
    else:
        raw = [str(item) for item in outer]
    if len(raw) != len(FILTER_LABELS):
        raise RuntimeError(f"expected four resolved filter paths, got outer={outer!r}, flat={raw!r}")
    mapping: dict[str, int] = {}
    for idx, path in enumerate(raw):
        for label, suffix in (
            ("support_plane", "/ground"),
            ("link4", "/Robot/link4"),
            ("link5", "/Robot/link5"),
            ("gripper_link", "/Robot/gripper_link"),
        ):
            if path.endswith(suffix) or suffix in path:
                if label in mapping:
                    raise RuntimeError(f"duplicate resolved filter label {label}: {raw}")
                mapping[label] = idx
    if set(mapping) != set(FILTER_LABELS):
        raise RuntimeError(f"resolved filter path mapping failed: paths={raw}, mapping={mapping}")
    if len(set(mapping.values())) != len(FILTER_LABELS):
        raise RuntimeError(f"resolved filter indices are not one-to-one: paths={raw}, mapping={mapping}")
    return raw, mapping


def _sensor_contract(inner: Any) -> tuple[dict[str, Any], dict[str, int]]:
    from pxr import PhysxSchema

    sensor = inner._d332_contact_sensor
    paths, mapping = _resolved_filter_map(sensor)
    data = sensor.data
    expected_shapes = {
        "net_forces_w": [1, 1, 3],
        "net_forces_w_history": [1, 1, 1, 3],
        "force_matrix_w": [1, 1, 4, 3],
        "force_matrix_w_history": [1, 1, 1, 4, 3],
        "contact_pos_w": [1, 1, 4, 3],
        "pos_w": [1, 1, 3],
        "quat_w": [1, 1, 4],
    }
    actual_shapes: dict[str, list[int] | None] = {}
    for name in expected_shapes:
        value = getattr(data, name)
        actual_shapes[name] = None if value is None else list(value.shape)
    errors = [
        f"{name}: expected {shape}, got {actual_shapes[name]}"
        for name, shape in expected_shapes.items()
        if actual_shapes[name] != shape
    ]

    stage = inner.scene.stage
    sponge_prim = stage.GetPrimAtPath("/World/envs/env_0/Sponge")
    reporter = PhysxSchema.PhysxContactReportAPI.Get(stage, sponge_prim.GetPath())
    reporter_threshold = reporter.GetThresholdAttr().Get()
    rigid = PhysxSchema.PhysxRigidBodyAPI.Get(stage, sponge_prim.GetPath())
    sleep_threshold = rigid.GetSleepThresholdAttr().Get()
    checks = {
        "num_instances": int(sensor.num_instances),
        "num_bodies": int(sensor.num_bodies),
        "body_names": list(sensor.body_names),
        "sensor_count": int(sensor.contact_physx_view.sensor_count),
        "filter_count": int(sensor.contact_physx_view.filter_count),
        "resolved_filter_paths": paths,
        "resolved_filter_index_by_label": mapping,
        "actual_tensor_shapes": actual_shapes,
        "expected_tensor_shapes": expected_shapes,
        "reporter_threshold_n": None if reporter_threshold is None else float(reporter_threshold),
        "rigid_body_sleep_threshold": None if sleep_threshold is None else float(sleep_threshold),
        "instrumentation_side_effect": "activate_contact_sensors also authors rigid-body sleep threshold 0",
    }
    if checks["num_instances"] != 1:
        errors.append(f"num_instances={checks['num_instances']}")
    if checks["num_bodies"] != 1:
        errors.append(f"num_bodies={checks['num_bodies']}")
    if checks["body_names"] != ["Sponge"]:
        errors.append(f"body_names={checks['body_names']}")
    if checks["sensor_count"] != 1:
        errors.append(f"sensor_count={checks['sensor_count']}")
    if checks["filter_count"] != 4:
        errors.append(f"filter_count={checks['filter_count']}")
    if reporter_threshold is None or abs(float(reporter_threshold)) > 1.0e-12:
        errors.append(f"reporter_threshold={reporter_threshold}")
    checks["errors"] = errors
    checks["hard_contract_pass"] = not errors
    return checks, mapping


def _extract_default_physx_mirror_recook(inner: Any, canonical: dict[str, Any]) -> dict[str, Any]:
    import hppfcl
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils
    from scipy.spatial import ConvexHull

    stage = inner.scene.stage
    suffix = "/Robot/link5/collisions/link5/node_STL_BINARY_"
    candidates = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = prim.GetPath().pathString
        if path.startswith("/World/envs/env_0/") and path.endswith(suffix):
            candidates.append(prim)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one env0 link5 collision mesh instance proxy, got {[p.GetPath().pathString for p in candidates]}"
        )
    collision_root_prim = candidates[0]
    mesh_candidates = [prim for prim in Usd.PrimRange(collision_root_prim) if prim.IsA(UsdGeom.Mesh)]
    if len(mesh_candidates) != 1:
        raise RuntimeError(
            "expected one UsdGeomMesh below link5 collision root, got "
            f"{[(p.GetPath().pathString, p.GetTypeName()) for p in mesh_candidates]}"
        )
    collision_prim = mesh_candidates[0]
    api_prim = collision_prim if collision_prim.HasAPI(UsdPhysics.MeshCollisionAPI) else collision_root_prim
    approximation = UsdPhysics.MeshCollisionAPI(api_prim).GetApproximationAttr().Get()
    if str(approximation) != "convexHull":
        raise RuntimeError(f"unexpected link5 collision approximation: {approximation}")
    prototype_mesh_prim = collision_prim.GetPrimInPrototype() if collision_prim.IsInstanceProxy() else collision_prim
    if not prototype_mesh_prim.IsValid() or not prototype_mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(
            "prototype mesh prim is invalid: "
            f"instance={collision_prim.GetPath().pathString}, prototype={prototype_mesh_prim.GetPath().pathString}"
        )

    mesh_geom = UsdGeom.Mesh(collision_prim)
    source_points = list(mesh_geom.GetPointsAttr().Get() or [])
    source_face_counts = [int(value) for value in list(mesh_geom.GetFaceVertexCountsAttr().Get() or [])]
    source_face_indices = [int(value) for value in list(mesh_geom.GetFaceVertexIndicesAttr().Get() or [])]
    if not source_points or not source_face_counts or not source_face_indices:
        raise RuntimeError("live link5 collision mesh has empty topology")
    mesh_l2w = UsdGeom.Xformable(collision_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    link5_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/link5")
    if not link5_prim.IsValid():
        raise RuntimeError("runtime link5 prim is invalid")
    link5_w2l = UsdGeom.Xformable(link5_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    source_link5_vertices = []
    for vertex in source_points:
        world_point = mesh_l2w.Transform(Gf.Vec3d(float(vertex[0]), float(vertex[1]), float(vertex[2])))
        link_point = link5_w2l.Transform(world_point)
        source_link5_vertices.append([float(link_point[0]), float(link_point[1]), float(link_point[2])])
    source_link5_vertices_np = np.asarray(source_link5_vertices, dtype=np.float64)

    # The collision API lives on an instance-proxy Xform while the actual Mesh
    # is its child. PhysX's public cooking request accepts only a UsdGeomMesh
    # carrying the collision API, so mirror the exact stage-extracted physical
    # mesh in meters, cook it synchronously, and remove it before any step.
    mirror_root_path = "/World/D332CookMirror"
    mirror_mesh_path = mirror_root_path + "/link5_collision_mesh"
    if stage.GetPrimAtPath(mirror_root_path).IsValid():
        stage.RemovePrim(mirror_root_path)
    mirror = UsdGeom.Mesh.Define(stage, mirror_mesh_path)
    mirror.CreatePointsAttr([Gf.Vec3f(*[float(value) for value in point]) for point in source_link5_vertices_np])
    mirror.CreateFaceVertexCountsAttr(source_face_counts)
    mirror.CreateFaceVertexIndicesAttr(source_face_indices)
    mirror.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mirror.GetPrim())
    mirror_api = UsdPhysics.MeshCollisionAPI.Apply(mirror.GetPrim())
    mirror_api.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    cooking_prim = mirror.GetPrim()

    holder: dict[str, Any] = {}

    def _on_result(result: Any, convexes: list[Any]) -> None:
        holder["result"] = result
        holder["convexes"] = list(convexes)

    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    prim_id = PhysicsSchemaTools.sdfPathToInt(cooking_prim.GetPath())
    try:
        get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id,
            collision_prim_id=prim_id,
            run_asynchronously=False,
            on_result=_on_result,
        )
    finally:
        stage.RemovePrim(mirror_root_path)
    if holder.get("result") != PhysxCollisionRepresentationResult.RESULT_VALID:
        raise RuntimeError(f"PhysX cooked hull request failed: {holder}")
    convexes = holder.get("convexes", [])
    if len(convexes) != 1:
        raise RuntimeError(f"convexHull approximation returned {len(convexes)} convex parts")
    cooked = convexes[0]

    mesh_vertices = []
    link5_vertices = []
    for vertex in cooked.vertices:
        mesh_point = Gf.Vec3d(float(vertex.x), float(vertex.y), float(vertex.z))
        mesh_vertices.append([float(mesh_point[0]), float(mesh_point[1]), float(mesh_point[2])])
        link5_vertices.append([float(mesh_point[0]), float(mesh_point[1]), float(mesh_point[2])])
    mesh_vertices_np = np.asarray(mesh_vertices, dtype=np.float64)
    link5_vertices_np = np.asarray(link5_vertices, dtype=np.float64)
    if len(link5_vertices_np) < 4:
        raise RuntimeError(f"mirror-recooked hull has too few vertices: {len(link5_vertices_np)}")

    polygon_rows = []
    cooked_indices = [int(index) for index in cooked.indices]
    for polygon in cooked.polygons:
        start = int(polygon.index_base)
        count = int(polygon.num_vertices)
        polygon_rows.append(cooked_indices[start : start + count])

    points = _fcl_points(hppfcl, link5_vertices_np)
    geometry = hppfcl.Convex.convexHull(points, False, "")
    if geometry is None:
        raise RuntimeError("hppfcl could not reconstruct the mirror-recooked convex set")
    cylinder = hppfcl.Cylinder(CYLINDER_RADIUS_M, CYLINDER_HEIGHT_M)
    link5_pos = np.asarray(canonical["commanded_link5_pos_local_m"], dtype=np.float64)
    link5_rot = np.asarray(canonical["commanded_link5_rot_local"], dtype=np.float64)
    query = _fcl_query(
        hppfcl,
        geometry,
        hppfcl.Transform3f(link5_rot, link5_pos),
        cylinder,
        hppfcl.Transform3f(np.eye(3, dtype=np.float64), OBJECT_CENTER_LOCAL_M),
    )
    query["verdict"] = _signed_distance_verdict(float(query["signed_distance_m"]))
    epa_depth_m = None if query["contact"] is None else float(query["contact"]["penetration_depth_m"])
    query["signed_distance_collision_sign_consistent"] = bool(
        (float(query["signed_distance_m"]) < 0.0) == bool(query["is_collision"])
    )
    query["gjk_epa_depth_consistent_within_0p1mm"] = bool(
        epa_depth_m is None
        or abs(epa_depth_m + float(query["signed_distance_m"])) <= SIGNED_DISTANCE_BORDER_M
    )
    if not query["signed_distance_collision_sign_consistent"] or not query[
        "gjk_epa_depth_consistent_within_0p1mm"
    ]:
        raise RuntimeError(f"mirror-recooked hull GJK/EPA disagreement: {query}")

    scipy_hull = ConvexHull(link5_vertices_np)
    hull_vertices_world = (link5_rot @ link5_vertices_np.T).T + link5_pos
    return {
        "source": "omni.physx synchronous cook of an exact non-stepped mirror of the live stage mesh",
        "live_instance_direct_request": "not supported: collision API is on instance-proxy Xform, mesh is child",
        "callback_result": str(holder["result"]),
        "collision_prim_path": collision_prim.GetPath().pathString,
        "collision_prim_is_instance_proxy": bool(collision_prim.IsInstanceProxy()),
        "collision_root_prim_path": collision_root_prim.GetPath().pathString,
        "mesh_collision_api_prim_path": api_prim.GetPath().pathString,
        "prototype_mesh_prim_path": prototype_mesh_prim.GetPath().pathString,
        "cooking_mirror_prim_path": mirror_mesh_path,
        "cooking_mirror_removed_before_physics": bool(not stage.GetPrimAtPath(mirror_root_path).IsValid()),
        "usd_approximation": str(approximation),
        "convex_part_count": len(convexes),
        "cooked_vertex_count": int(len(link5_vertices_np)),
        "cooked_polygon_count": int(len(polygon_rows)),
        "cooked_index_count": int(len(cooked_indices)),
        "cooked_vertices_mesh_local": mesh_vertices_np.tolist(),
        "cooked_vertices_link5_local_m": link5_vertices_np.tolist(),
        "cooked_polygons": polygon_rows,
        "cooked_mesh_local_bounds": np.vstack([mesh_vertices_np.min(axis=0), mesh_vertices_np.max(axis=0)]).tolist(),
        "cooked_link5_local_bounds_m": np.vstack(
            [link5_vertices_np.min(axis=0), link5_vertices_np.max(axis=0)]
        ).tolist(),
        "source_stage_mesh": {
            "vertex_count": int(len(source_link5_vertices_np)),
            "face_count": int(len(source_face_counts)),
            "index_count": int(len(source_face_indices)),
            "link5_local_bounds_m": np.vstack(
                [source_link5_vertices_np.min(axis=0), source_link5_vertices_np.max(axis=0)]
            ).tolist(),
        },
        "query": query,
        "figure_vertices_world": hull_vertices_world,
        "figure_faces": np.asarray(scipy_hull.simplices, dtype=np.int64),
    }


def _contact_state(sensor: Any, mapping: dict[str, int]) -> dict[str, Any]:
    matrix = sensor.data.force_matrix_w[0, 0].detach().cpu().numpy().astype(np.float64)
    points = sensor.data.contact_pos_w[0, 0].detach().cpu().numpy().astype(np.float64)
    out: dict[str, Any] = {}
    for label in FILTER_LABELS:
        idx = int(mapping[label])
        force = matrix[idx]
        point = points[idx]
        out[label] = {
            "filter_index": idx,
            "force_w_n": force.tolist(),
            "force_norm_n": float(np.linalg.norm(force)),
            "contact_point_w_m": point.tolist() if bool(np.all(np.isfinite(point))) else None,
        }
    net = sensor.data.net_forces_w[0, 0].detach().cpu().numpy().astype(np.float64)
    return {"by_filter": out, "net_force_w_n": net.tolist(), "net_force_norm_n": float(np.linalg.norm(net))}


def _pose_frame(
    name: str,
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    label: str,
    role: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rot = _quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64))
    return {
        "name": name,
        "label": label,
        "position": np.asarray(position, dtype=np.float64).tolist(),
        "axes": {
            "x": rot[:, 0].tolist(),
            "y": rot[:, 1].tolist(),
            "z": rot[:, 2].tolist(),
        },
        "role": role,
        "metadata": dict(metadata or {}),
    }


def _state_row(
    inner: Any,
    *,
    phase: str,
    step: int,
    command_target: Any,
    canonical: dict[str, Any],
    object_start_w: np.ndarray,
    contact: dict[str, Any],
) -> dict[str, Any]:
    origin = inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
    object_pos_w = inner._sponge.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    object_quat_w = inner._sponge.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    object_lin_vel_w = inner._sponge.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
    object_ang_vel_w = inner._sponge.data.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float64)
    object_rot = _quat_wxyz_to_rot(object_quat_w)
    tilt_deg = math.degrees(math.acos(float(np.clip(object_rot[2, 2], -1.0, 1.0))))
    disp_w = object_pos_w - object_start_w

    body_pos = inner._robot.data.body_pos_w[0].detach().cpu().numpy().astype(np.float64)
    body_quat = inner._robot.data.body_quat_w[0].detach().cpu().numpy().astype(np.float64)
    link5_pos_local = body_pos[inner.link5_idx] - origin
    link5_quat = body_quat[inner.link5_idx]
    link5_rot = _quat_wxyz_to_rot(link5_quat)
    gripper_pos_local = body_pos[inner.gripper_link_idx] - origin
    gripper_quat = body_quat[inner.gripper_link_idx]
    actual_tcp_local = inner._tcp_pos_w[0].detach().cpu().numpy().astype(np.float64) - origin
    target_tcp = np.asarray(canonical["target_tcp_local_m"], dtype=np.float64)
    commanded_tcp = np.asarray(canonical["commanded_tcp_local_m"], dtype=np.float64)

    actual_joint = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    commanded_joint = command_target[0].detach().cpu().numpy().astype(np.float64)
    actual_by_name = {name: float(actual_joint[idx]) for idx, name in enumerate(inner._robot.joint_names)}
    commanded_by_name = {name: float(commanded_joint[idx]) for idx, name in enumerate(inner._robot.joint_names)}

    robot_items = [contact["by_filter"][label] for label in ("link4", "link5", "gripper_link")]
    robot_force_norms = [float(item["force_norm_n"]) for item in robot_items]
    max_robot_idx = int(np.argmax(robot_force_norms))
    max_robot_label = ("link4", "link5", "gripper_link")[max_robot_idx]
    max_robot_item = contact["by_filter"][max_robot_label]
    witness_world = max_robot_item["contact_point_w_m"]
    witness_source = f"sensor_mean_{max_robot_label}"
    if witness_world is None or float(max_robot_item["force_norm_n"]) <= 0.0:
        witness_local = np.asarray(
            canonical["offline_witness_cylinder_local_m"], dtype=np.float64
        )
        witness_source = "offline_hull_cylinder_witness"
        witness_axis = np.asarray(canonical["tangent_axis"], dtype=np.float64)
    else:
        witness_local = np.asarray(witness_world, dtype=np.float64) - origin
        witness_axis = _unit(max_robot_item["force_w_n"], fallback=tuple(canonical["tangent_axis"]))
    cylinder_z = object_rot[:, 2]
    if abs(float(np.dot(_unit(witness_axis), _unit(cylinder_z)))) > 0.98:
        witness_axis = np.asarray(canonical["tangent_axis"], dtype=np.float64)

    frames = [
        frame_from_axes(
            "d332_target_tcp",
            target_tcp,
            x_axis=canonical["tangent_axis"],
            z_axis=[0.0, 0.0, 1.0],
            role="target",
            label="target TCP",
        ),
        {
            "name": "d332_actual_tcp",
            "label": "actual TCP",
            "position": actual_tcp_local.tolist(),
            "axes": {
                "x": link5_rot[:, 0].tolist(),
                "y": link5_rot[:, 1].tolist(),
                "z": link5_rot[:, 2].tolist(),
            },
            "role": "actual",
        },
        _pose_frame("d332_link5", link5_pos_local, link5_quat, label="link5 body", role="actual"),
        _pose_frame(
            "d332_gripper_link",
            gripper_pos_local,
            gripper_quat,
            label="gripper_link body",
            role="fixed_jaw",
        ),
        _pose_frame(
            "d332_cylinder",
            object_pos_w - origin,
            object_quat_w,
            label="live cylinder D34xH90",
            role="object",
        ),
        frame_from_axes(
            "d332_contact_or_gap_witness",
            witness_local,
            x_axis=witness_axis,
            z_axis=cylinder_z,
            role="cube_face",
            label=f"witness ({witness_source})",
        ),
        frame_from_axes(
            "d332_commanded_tcp",
            commanded_tcp,
            x_axis=canonical["tangent_axis"],
            z_axis=np.asarray(canonical["commanded_link5_rot_local"], dtype=np.float64)[:, 2],
            role="candidate",
            label="commanded FK TCP",
        ),
    ]
    return {
        "step": int(step),
        "phase": phase,
        "physics_time_s": float((step + 1) * PHYSICS_DT_S),
        "object_pos_w_m": object_pos_w.tolist(),
        "object_pos_local_m": (object_pos_w - origin).tolist(),
        "object_quat_wxyz": object_quat_w.tolist(),
        "object_lin_vel_w_mps": object_lin_vel_w.tolist(),
        "object_ang_vel_w_radps": object_ang_vel_w.tolist(),
        "object_speed_mps": float(np.linalg.norm(object_lin_vel_w)),
        "object_ang_speed_radps": float(np.linalg.norm(object_ang_vel_w)),
        "object_tilt_deg": float(tilt_deg),
        "object_disp_w_m": disp_w.tolist(),
        "object_disp_xy_mm": float(np.linalg.norm(disp_w[:2]) * 1000.0),
        "object_z_delta_mm": float(disp_w[2] * 1000.0),
        "target_tcp_local_m": target_tcp.tolist(),
        "actual_tcp_local_m": actual_tcp_local.tolist(),
        "commanded_tcp_local_m": commanded_tcp.tolist(),
        "tcp_error_mm": float(np.linalg.norm(actual_tcp_local - target_tcp) * 1000.0),
        "commanded_tcp_error_mm": float(np.linalg.norm(commanded_tcp - target_tcp) * 1000.0),
        "joint_tracking_error_max_rad": float(np.max(np.abs(actual_joint - commanded_joint))),
        "actual_joint_rad_by_name": actual_by_name,
        "commanded_joint_rad_by_name": commanded_by_name,
        "contact": contact,
        "max_robot_filter": max_robot_label,
        "max_robot_filter_force_n": float(max_robot_item["force_norm_n"]),
        "witness_source": witness_source,
        "frames": frames,
    }


def _flatten_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        key: row[key]
        for key in (
            "step",
            "phase",
            "physics_time_s",
            "object_speed_mps",
            "object_ang_speed_radps",
            "object_tilt_deg",
            "object_disp_xy_mm",
            "object_z_delta_mm",
            "tcp_error_mm",
            "commanded_tcp_error_mm",
            "joint_tracking_error_max_rad",
            "max_robot_filter",
            "max_robot_filter_force_n",
            "witness_source",
        )
    }
    for name, values in (
        ("object_pos_w_m", row["object_pos_w_m"]),
        ("object_pos_local_m", row["object_pos_local_m"]),
        ("object_lin_vel_w_mps", row["object_lin_vel_w_mps"]),
        ("object_ang_vel_w_radps", row["object_ang_vel_w_radps"]),
        ("object_disp_w_m", row["object_disp_w_m"]),
        ("target_tcp_local_m", row["target_tcp_local_m"]),
        ("actual_tcp_local_m", row["actual_tcp_local_m"]),
        ("commanded_tcp_local_m", row["commanded_tcp_local_m"]),
    ):
        for axis, value in zip("xyz", values, strict=True):
            out[f"{name}_{axis}"] = value
    for axis, value in zip(("w", "x", "y", "z"), row["object_quat_wxyz"], strict=True):
        out[f"object_quat_{axis}"] = value
    for prefix in ("actual", "commanded"):
        for name, value in row[f"{prefix}_joint_rad_by_name"].items():
            out[f"{prefix}_{name}_rad"] = value
    out["sensor_net_force_norm_n"] = row["contact"]["net_force_norm_n"]
    for axis, value in zip("xyz", row["contact"]["net_force_w_n"], strict=True):
        out[f"sensor_net_force_{axis}_n"] = value
    for label in FILTER_LABELS:
        item = row["contact"]["by_filter"][label]
        out[f"{label}_force_norm_n"] = item["force_norm_n"]
        for axis, value in zip("xyz", item["force_w_n"], strict=True):
            out[f"{label}_force_{axis}_n"] = value
        point = item["contact_point_w_m"]
        for axis, value in zip("xyz", point if point is not None else [math.nan] * 3, strict=True):
            out[f"{label}_contact_point_{axis}_w_m"] = value
    return out


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = [_flatten_trace_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def _first_consecutive(mask: list[bool], count: int = CONSECUTIVE_EVENT_STEPS) -> int:
    run = 0
    for idx, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= count:
            return idx - count + 1
    return -1


def _trace_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contact_onsets: dict[str, int] = {}
    max_forces: dict[str, float] = {}
    max_force_steps: dict[str, int] = {}
    for label in ("link4", "link5", "gripper_link"):
        values = [float(row["contact"]["by_filter"][label]["force_norm_n"]) for row in rows]
        contact_onsets[label] = _first_consecutive([value >= ROBOT_FORCE_EVENT_N for value in values])
        max_idx = int(np.argmax(values))
        max_forces[label] = float(values[max_idx])
        max_force_steps[label] = max_idx
    valid_contact = [(step, label) for label, step in contact_onsets.items() if step >= 0]
    valid_contact.sort()
    first_contact_step = valid_contact[0][0] if valid_contact else -1
    suspected_link = valid_contact[0][1] if valid_contact else max(max_forces, key=max_forces.get)

    disturbance_mask = [
        float(row["object_disp_xy_mm"]) >= DISTURBANCE_XY_M * 1000.0
        or float(row["object_tilt_deg"]) >= DISTURBANCE_TILT_DEG
        for row in rows
    ]
    disturbance_step = _first_consecutive(disturbance_mask)
    speeds = np.asarray([float(row["object_speed_mps"]) for row in rows], dtype=np.float64)
    angular = np.asarray([float(row["object_ang_speed_radps"]) for row in rows], dtype=np.float64)
    displacements = np.asarray([float(row["object_disp_xy_mm"]) for row in rows], dtype=np.float64)
    tilts = np.asarray([float(row["object_tilt_deg"]) for row in rows], dtype=np.float64)
    tcp_errors = np.asarray([float(row["tcp_error_mm"]) for row in rows], dtype=np.float64)
    joint_errors = np.asarray([float(row["joint_tracking_error_max_rad"]) for row in rows], dtype=np.float64)
    return {
        "trace_physics_steps": len(rows),
        "first_contact_step_by_link": contact_onsets,
        "first_robot_contact_step": first_contact_step,
        "object_disturbance_start_step": disturbance_step,
        "suspected_link": suspected_link,
        "max_force_n_by_link": max_forces,
        "max_force_step_by_link": max_force_steps,
        "peak_object_speed_mps": float(speeds.max()),
        "peak_object_speed_step": int(speeds.argmax()),
        "peak_object_angular_speed_radps": float(angular.max()),
        "peak_object_angular_speed_step": int(angular.argmax()),
        "max_object_disp_xy_mm": float(displacements.max()),
        "final_object_disp_xy_mm": float(displacements[-1]),
        "max_object_tilt_deg": float(tilts.max()),
        "final_object_tilt_deg": float(tilts[-1]),
        "final_object_pos_local_m": rows[-1]["object_pos_local_m"],
        "final_object_quat_wxyz": rows[-1]["object_quat_wxyz"],
        "min_tcp_error_mm": float(tcp_errors.min()),
        "final_tcp_error_mm": float(tcp_errors[-1]),
        "commanded_tcp_error_mm": float(rows[-1]["commanded_tcp_error_mm"]),
        "max_joint_tracking_error_rad": float(joint_errors.max()),
        "final_joint_tracking_error_rad": float(joint_errors[-1]),
    }


def _baseline_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tail = rows[-BASELINE_TAIL_STEPS:]
    support_tail = [float(row["contact"]["by_filter"]["support_plane"]["force_norm_n"]) for row in tail]
    net_tail = [float(row["contact"]["net_force_norm_n"]) for row in tail]
    robot_max = {
        label: max(float(row["contact"]["by_filter"][label]["force_norm_n"]) for row in rows)
        for label in ("link4", "link5", "gripper_link")
    }
    support_median = float(np.median(support_tail))
    net_median = float(np.median(net_tail))
    expected_weight_n = OBJECT_MASS_KG * 9.81
    reporter_diagnostic_positive = bool(net_median > SUPPORT_POSITIVE_CONTROL_N)
    preregistered_support_positive = bool(support_median > SUPPORT_POSITIVE_CONTROL_N)
    robot_quiet = bool(max(robot_max.values()) < ROBOT_FORCE_EVENT_N)
    max_xy_mm = max(float(row["object_disp_xy_mm"]) for row in rows)
    max_tilt_deg = max(float(row["object_tilt_deg"]) for row in rows)
    return {
        "physics_steps": len(rows),
        "positive_control_source": "unfiltered cylinder net force in robot-free baseline",
        "net_force_last50_median_n": net_median,
        "expected_static_weight_n": expected_weight_n,
        "net_force_vs_weight_error_n": net_median - expected_weight_n,
        "support_force_last50_median_n": support_median,
        "support_force_max_n": max(
            float(row["contact"]["by_filter"]["support_plane"]["force_norm_n"]) for row in rows
        ),
        "robot_filter_baseline_max_n": robot_max,
        "net_positive_control_pass": reporter_diagnostic_positive,
        "posthoc_reporter_diagnostic_pass": bool(reporter_diagnostic_positive and robot_quiet),
        "support_positive_control_pass": preregistered_support_positive,
        "preregistered_positive_control_pass": bool(preregistered_support_positive and robot_quiet),
        "support_filtered_channel_available": bool(support_median > 0.0),
        "support_filter_limitation": (
            "the root filter /World/ground returned no usable channel; the exact CollisionPlane prim was not tested"
        ),
        "robot_baseline_quiet_pass": robot_quiet,
        "max_object_disp_xy_mm": max_xy_mm,
        "max_object_tilt_deg": max_tilt_deg,
        "baseline_disturbance_below_diagnostic_thresholds": bool(
            max_xy_mm < DISTURBANCE_XY_M * 1000.0 and max_tilt_deg < DISTURBANCE_TILT_DEG
        ),
        "positive_control_pass": bool(preregistered_support_positive and robot_quiet),
    }


def _frame_with_rotation(
    name: str,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    label: str,
    role: str,
    label_offset: tuple[float, float, float],
) -> dict[str, Any]:
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return frame_from_axes(
        name,
        position,
        x_axis=rot[:, 0],
        z_axis=rot[:, 2],
        role=role,
        label=label,
        metadata={"label_offset": list(label_offset), "show_axis_labels": False},
    )


def _frames_from_flat_trace_row(row: dict[str, str], summary: dict[str, Any]) -> list[dict[str, Any]]:
    from sim_scripts.roarm_kinematics import Tmat, Trot_z

    canonical = summary["offline"]["canonical"]
    actual_q_rad = np.asarray([float(row[f"actual_{name}_rad"]) for name in ALL_JOINT_NAMES], dtype=np.float64)
    _fk_tcp, link5_pos, link5_rot = _fk_runtime_tcp(np.degrees(actual_q_rad))
    actual_tcp = np.asarray([float(row[f"actual_tcp_local_m_{axis}"]) for axis in "xyz"], dtype=np.float64)
    target_tcp = np.asarray(canonical["target_tcp_local_m"], dtype=np.float64)

    gripper_rel = Tmat((0.0, 0.018821, 0.052035), (-1.5708, -1.5708, 0.0)) @ Trot_z(
        float(actual_q_rad[5])
    )
    gripper_pos = link5_pos + link5_rot @ gripper_rel[:3, 3]
    gripper_rot = link5_rot @ gripper_rel[:3, :3]

    object_pos = np.asarray([float(row[f"object_pos_local_m_{axis}"]) for axis in "xyz"], dtype=np.float64)
    object_quat = np.asarray([float(row[f"object_quat_{axis}"]) for axis in ("w", "x", "y", "z")])
    object_rot = _quat_wxyz_to_rot(object_quat)
    suspected = str(summary["runtime"]["target_settle"]["suspected_link"])
    witness = np.asarray(
        [float(row[f"{suspected}_contact_point_{axis}_w_m"]) for axis in "xyz"], dtype=np.float64
    )
    force = np.asarray([float(row[f"{suspected}_force_{axis}_n"]) for axis in "xyz"], dtype=np.float64)
    if not bool(np.all(np.isfinite(witness))):
        witness = np.asarray(
            summary["default_physx_mirror_recook"]["query"]["nearest_point_cylinder_m"], dtype=np.float64
        )
    witness_axis = _unit(force, fallback=tuple(canonical["tangent_axis"]))
    if abs(float(np.dot(witness_axis, object_rot[:, 2]))) > 0.98:
        witness_axis = np.asarray(canonical["tangent_axis"], dtype=np.float64)

    common_meta = {"show_axis_labels": False}
    return [
        frame_from_axes(
            "d332_target_tcp",
            target_tcp,
            x_axis=canonical["tangent_axis"],
            z_axis=[0.0, 0.0, 1.0],
            role="target",
            label="target TCP",
            metadata={**common_meta, "label_offset": [0.010, 0.010, -0.010]},
        ),
        _frame_with_rotation(
            "d332_actual_tcp",
            actual_tcp,
            link5_rot,
            label="actual TCP",
            role="actual",
            label_offset=(0.010, -0.014, -0.004),
        ),
        _frame_with_rotation(
            "d332_link5",
            link5_pos,
            link5_rot,
            label="link5",
            role="actual",
            label_offset=(0.006, 0.004, 0.008),
        ),
        _frame_with_rotation(
            "d332_gripper_link",
            gripper_pos,
            gripper_rot,
            label="gripper_link",
            role="fixed_jaw",
            label_offset=(-0.040, -0.005, 0.006),
        ),
        _frame_with_rotation(
            "d332_cylinder",
            object_pos,
            object_rot,
            label="cylinder",
            role="object",
            label_offset=(0.014, 0.014, 0.012),
        ),
        frame_from_axes(
            "d332_contact_or_gap_witness",
            witness,
            x_axis=witness_axis,
            z_axis=object_rot[:, 2],
            role="cube_face",
            label=f"{suspected} contact",
            metadata={**common_meta, "label_offset": [-0.045, 0.010, 0.014]},
        ),
    ]


def _write_reanalysis_snapshot(
    path: Path,
    row: dict[str, str],
    summary: dict[str, Any],
    *,
    title: str,
) -> None:
    suspected = str(summary["runtime"]["target_settle"]["suspected_link"])
    frames = _frames_from_flat_trace_row(row, summary)
    snapshot_frame_plot(
        path,
        frames,
        title=title,
        axis_length=0.025,
        view=(22.0, -68.0),
        annotations=[
            f"physics step={row['step']}, t={float(row['physics_time_s']):.3f}s",
            f"TCP error={float(row['tcp_error_mm']):.3f}mm; commanded={float(row['commanded_tcp_error_mm']):.3f}mm",
            f"object XY={float(row['object_disp_xy_mm']):.3f}mm; tilt={float(row['object_tilt_deg']):.3f}deg",
            f"object z delta={float(row['object_z_delta_mm']):.3f}mm; net force={float(row['sensor_net_force_norm_n']):.3f}N",
            f"{suspected} normal force={float(row[f'{suspected}_force_norm_n']):.3f}N",
            f"witness=sensor mean contact point ({suspected})",
            "scene confound: reset cylinder-ground penetration=12.117mm",
        ],
    )


def _write_runtime_snapshot(path: Path, row: dict[str, Any], title: str) -> str:
    contact = row["contact"]["by_filter"]
    snapshot_frame_plot(
        path,
        row["frames"],
        title=title,
        axis_length=0.025,
        view=(22.0, -68.0),
        annotations=[
            f"physics step={row['step']}, t={row['physics_time_s']:.3f}s",
            f"TCP error={row['tcp_error_mm']:.3f}mm; commanded={row['commanded_tcp_error_mm']:.3f}mm",
            f"object XY={row['object_disp_xy_mm']:.3f}mm; tilt={row['object_tilt_deg']:.3f}deg",
            "forces N: "
            + ", ".join(f"{label}={contact[label]['force_norm_n']:.3f}" for label in FILTER_LABELS),
            f"witness={row['witness_source']}",
        ],
    )
    return _rel(path)


def _runtime_versions() -> dict[str, Any]:
    import numpy
    import psutil

    versions = {
        "python": sys.version,
        "numpy": numpy.__version__,
        "psutil": psutil.__version__,
    }
    for name in ("isaaclab", "isaacsim", "rerun-sdk", "trimesh", "hppfcl", "scipy"):
        try:
            versions[name] = _package_version(name)
        except Exception as exc:
            versions[name] = {"error": repr(exc)}
    if numpy.__version__ != "1.26.0" or psutil.__version__ != "5.9.8":
        raise RuntimeError(f"Isaac package pins changed: numpy={numpy.__version__}, psutil={psutil.__version__}")
    return versions


def _classify(
    geometry_verdict: str,
    sensor_valid: bool,
    reporter_diagnostic_valid: bool,
    preregistered_support_valid: bool,
    runtime_scene_confounded: bool,
    stats: dict[str, Any],
) -> dict[str, Any]:
    link5_step = int(stats["first_contact_step_by_link"]["link5"])
    robot_step = int(stats["first_robot_contact_step"])
    disturbance_step = int(stats["object_disturbance_start_step"])
    timing_compatible = bool(link5_step >= 0 and disturbance_step >= 0 and link5_step <= disturbance_step + 1)
    robot_timing_compatible = bool(robot_step >= 0 and disturbance_step >= 0 and robot_step <= disturbance_step + 1)
    if not sensor_valid or not reporter_diagnostic_valid:
        verdict = "D332_G0A_CONTACT_WITNESS_INVALID_MIXED"
        interpretation = "contact witness did not pass its hard contract or net-reporter diagnostic"
    elif geometry_verdict == "BORDERLINE":
        verdict = "D332_G0A_CANONICAL_OVERLAP_BORDERLINE"
        interpretation = "canonical signed distance is inside the preregistered borderline band"
    elif geometry_verdict == "OVERLAP" and (not preregistered_support_valid or runtime_scene_confounded):
        verdict = "D332_G0A_PRESTEP_MIRROR_HULL_OVERLAP_RUNTIME_GRIPPER_CONTACT_SCENE_CONFOUNDED_MIXED"
        interpretation = (
            "the pre-step default mirror recook overlaps, but the preregistered support control failed "
            "and the first runtime sample couples ground depenetration with robot contact"
        )
    elif geometry_verdict == "OVERLAP" and timing_compatible:
        verdict = "D332_G0A_LINK5_CONVEX_HULL_BLOCKER_SUPPORTED"
        interpretation = "offline hull overlap and link5-attributed runtime disturbance agree"
    elif geometry_verdict == "OVERLAP" and robot_timing_compatible:
        verdict = "D332_G0A_STATIC_COLLISION_CONFIRMED_LINK_ATTRIBUTION_MIXED"
        interpretation = (
            "default mirror-recooked link5 hull overlaps, while runtime disturbance is timing-compatible "
            f"with {stats['suspected_link']} rather than a sampled link5 force"
        )
    elif geometry_verdict == "OVERLAP":
        verdict = "D332_G0A_OFFLINE_RUNTIME_COLLIDER_MISMATCH"
        interpretation = "offline hull overlap lacks a timing-compatible link5 runtime event"
    elif disturbance_step >= 0 or int(stats["first_robot_contact_step"]) >= 0:
        verdict = "D332_G0A_OTHER_LINK_TRANSFORM_OR_STATIC_CAUSE"
        interpretation = "canonical link5 hull is clear but runtime contact/disturbance exists"
    else:
        verdict = "D332_G0A_FINAL_POSE_OVERLAP_REFUTED"
        interpretation = "only canonical final-pose overlap is refuted; swept path remains open"
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "link5_contact_disturbance_timing_compatible": timing_compatible,
        "any_robot_contact_disturbance_timing_compatible": robot_timing_compatible,
        "link5_contact_step": link5_step,
        "disturbance_step": disturbance_step,
        "preregistered_support_positive_control_pass": bool(preregistered_support_valid),
        "runtime_scene_confounded": bool(runtime_scene_confounded),
        "first_observation_is_post_physics_step": True,
        "contact_onset_left_censored": bool(robot_step == 0),
    }


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    cooked = summary["default_physx_mirror_recook"]["query"]
    full_hull = summary["offline"]["mathematical_full_hull"]
    baseline = summary["runtime"]["baseline"]
    stats = summary["runtime"]["target_settle"]
    classification = summary["classification"]
    lines = [
        "# D332 canonical static collision discriminator",
        "",
        f"Verdict: `{classification['verdict']}`",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Default PhysX mirror-recook signed distance | `{cooked['signed_distance_mm']:.6f} mm` |",
        f"| Mirror-recook verdict | `{cooked['verdict']}` |",
        f"| Mathematical full-hull precheck | `{full_hull['signed_distance_mm']:.6f} mm` |",
        f"| Raw STL negative-control distance | `{summary['offline']['raw_stl_negative_control']['signed_distance_mm']:.6f} mm` |",
        f"| Contact sensor hard contract | `{summary['runtime']['sensor_contract']['hard_contract_pass']}` |",
        f"| Frozen filtered-support positive control | `{baseline['preregistered_positive_control_pass']}` |",
        f"| Posthoc net-reporter diagnostic | `{baseline['posthoc_reporter_diagnostic_pass']}` |",
        f"| Baseline net force (last-50 median) | `{baseline.get('net_force_last50_median_n', 0.0):.6f} N` |",
        f"| Baseline max XY / tilt | `{baseline.get('max_object_disp_xy_mm', 0.0):.6f} mm / {baseline.get('max_object_tilt_deg', 0.0):.6f} deg` |",
        f"| Initial ground penetration | `{summary['runtime']['support_domain_audit'].get('initial_ground_penetration_mm', 0.0):.6f} mm` |",
        f"| First observed robot-contact post-step row | `{stats['first_robot_contact_step']}` |",
        f"| First observed link5-contact post-step row | `{stats['first_contact_step_by_link']['link5']}` |",
        f"| Runtime suspected link | `{stats['suspected_link']}` |",
        f"| Suspected-link peak force | `{stats['max_force_n_by_link'][stats['suspected_link']]:.6f} N` |",
        f"| Object disturbance start physics step | `{stats['object_disturbance_start_step']}` |",
        f"| Peak object speed | `{stats['peak_object_speed_mps']:.6f} m/s` |",
        f"| Final object XY displacement | `{stats['final_object_disp_xy_mm']:.6f} mm` |",
        f"| Final object tilt | `{stats['final_object_tilt_deg']:.6f} deg` |",
        f"| Final actual TCP error | `{stats['final_tcp_error_mm']:.6f} mm` |",
        f"| Commanded TCP error | `{stats['commanded_tcp_error_mm']:.6f} mm` |",
        "",
        classification["interpretation"] + ".",
        "",
        "This attempted final-pose discriminator is scene-confounded; it is neither a G0a pass nor a swept-path result.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cosine_xy(a: list[float], b: list[float]) -> float | None:
    av = np.asarray(a[:2], dtype=np.float64)
    bv = np.asarray(b[:2], dtype=np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1.0e-12:
        return None
    return float(np.dot(av, bv) / denom)


def _run_reanalysis(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.out_dir / "g0a_d332_static_collision_summary.json"
    baseline_path = args.out_dir / "d332_contact_baseline_trace.csv"
    target_path = args.out_dir / "d332_teleport_settle_trace.csv"
    if not summary_path.is_file() or not baseline_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("runtime summary and both trace CSVs are required for reanalysis")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    offline_path = args.out_dir / "d332_offline_geometry_precheck.json"
    if offline_path.is_file():
        summary["offline"] = json.loads(offline_path.read_text(encoding="utf-8"))
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    with target_path.open(newline="", encoding="utf-8") as handle:
        target_rows = list(csv.DictReader(handle))
    if len(baseline_rows) != BASELINE_PHYSICS_STEPS or len(target_rows) != TARGET_SETTLE_PHYSICS_STEPS:
        raise RuntimeError(
            f"unexpected trace lengths: baseline={len(baseline_rows)}, target={len(target_rows)}"
        )

    net_values = [float(row["sensor_net_force_norm_n"]) for row in baseline_rows]
    support_values = [float(row["support_plane_force_norm_n"]) for row in baseline_rows]
    robot_max = {
        label: max(float(row[f"{label}_force_norm_n"]) for row in baseline_rows)
        for label in ("link4", "link5", "gripper_link")
    }
    net_median = float(np.median(net_values[-BASELINE_TAIL_STEPS:]))
    support_median = float(np.median(support_values[-BASELINE_TAIL_STEPS:]))
    max_baseline_xy = max(float(row["object_disp_xy_mm"]) for row in baseline_rows)
    max_baseline_tilt = max(float(row["object_tilt_deg"]) for row in baseline_rows)
    robot_quiet = bool(max(robot_max.values()) < ROBOT_FORCE_EVENT_N)
    expected_weight_n = OBJECT_MASS_KG * 9.81
    reporter_diagnostic_pass = bool(net_median > SUPPORT_POSITIVE_CONTROL_N and robot_quiet)
    preregistered_support_pass = bool(support_median > SUPPORT_POSITIVE_CONTROL_N and robot_quiet)
    baseline = {
        "physics_steps": len(baseline_rows),
        "positive_control_source": "frozen gate: filtered support force; posthoc diagnostic: unfiltered net force",
        "net_force_last50_median_n": net_median,
        "net_force_min_n": min(net_values),
        "net_force_max_n": max(net_values),
        "expected_static_weight_n": expected_weight_n,
        "net_force_vs_weight_error_n": net_median - expected_weight_n,
        "support_force_last50_median_n": support_median,
        "support_force_max_n": max(support_values),
        "support_filtered_channel_available": bool(max(support_values) > 0.0),
        "support_filter_limitation": (
            "the root filter /World/ground returned no usable channel; raw stderr was not retained and "
            "the exact /World/ground/terrain/GroundPlane/CollisionPlane filter was not tested"
        ),
        "support_positive_control_pass": False,
        "net_positive_control_pass": bool(net_median > SUPPORT_POSITIVE_CONTROL_N),
        "posthoc_reporter_diagnostic_pass": reporter_diagnostic_pass,
        "preregistered_positive_control_pass": preregistered_support_pass,
        "robot_filter_baseline_max_n": robot_max,
        "robot_baseline_quiet_pass": robot_quiet,
        "max_object_disp_xy_mm": max_baseline_xy,
        "max_object_tilt_deg": max_baseline_tilt,
        "baseline_disturbance_below_diagnostic_thresholds": bool(
            max_baseline_xy < DISTURBANCE_XY_M * 1000.0 and max_baseline_tilt < DISTURBANCE_TILT_DEG
        ),
        "positive_control_pass": preregistered_support_pass,
    }

    stats = dict(summary["runtime"]["target_settle"])
    contact_step = int(stats["first_robot_contact_step"])
    suspected = str(stats["suspected_link"])
    if contact_step >= 0:
        row = target_rows[contact_step]
        force = [float(row[f"{suspected}_force_{axis}_n"]) for axis in "xyz"]
        displacement = [float(row[f"object_disp_w_m_{axis}"]) for axis in "xyz"]
        velocity = [float(row[f"object_lin_vel_w_mps_{axis}"]) for axis in "xyz"]
        point = [float(row[f"{suspected}_contact_point_{axis}_w_m"]) for axis in "xyz"]
        center = [float(row[f"object_pos_w_m_{axis}"]) for axis in "xyz"]
        stats["first_contact_witness"] = {
            "physics_step": contact_step,
            "suspected_link": suspected,
            "force_w_n": force,
            "force_norm_n": float(row[f"{suspected}_force_norm_n"]),
            "contact_point_w_m": point,
            "contact_point_relative_to_cylinder_center_mm": [
                (point[idx] - center[idx]) * 1000.0 for idx in range(3)
            ],
            "object_displacement_w_m": displacement,
            "object_velocity_w_mps": velocity,
            "force_vs_displacement_xy_cosine": _cosine_xy(force, displacement),
            "force_vs_velocity_xy_cosine": _cosine_xy(force, velocity),
        }
    summary["runtime"]["baseline"] = baseline
    summary["runtime"]["target_settle"] = stats
    summary["runtime"]["support_domain_audit"] = {
        "tap_table_top_z_m": TABLE_Z_M,
        "global_terrain_plane_z_m": 0.0,
        "terrain_above_tap_table_mm": -TABLE_Z_M * 1000.0,
        "initial_cylinder_bottom_z_m": float(OBJECT_CENTER_LOCAL_M[2] - 0.5 * CYLINDER_HEIGHT_M),
        "initial_ground_penetration_mm": float(
            max(0.0, -(OBJECT_CENTER_LOCAL_M[2] - 0.5 * CYLINDER_HEIGHT_M)) * 1000.0
        ),
        "baseline_first_post_step_z_delta_mm": float(baseline_rows[0]["object_z_delta_mm"]),
        "target_first_post_step_z_delta_mm": float(target_rows[0]["object_z_delta_mm"]),
        "baseline_first_post_step_net_force_n": float(baseline_rows[0]["sensor_net_force_norm_n"]),
        "target_first_post_step_net_force_n": float(target_rows[0]["sensor_net_force_norm_n"]),
        "observed_baseline_final_cylinder_center_z_m": float(baseline_rows[-1]["object_pos_local_m_z"]),
        "active_support": "global terrain plane inferred from settled cylinder center z",
        "clean_static_discriminator": False,
        "confound": "each phase starts with 12.117mm ground penetration; first sample couples depenetration and contact",
    }
    summary["runtime"]["sensor_contract"]["robot_filter_attribution_valid"] = False
    summary["runtime"]["sensor_contract"]["gripper_filter_positive_event_valid"] = bool(
        int(stats["first_contact_step_by_link"]["gripper_link"]) >= 0
    )
    summary["runtime"]["sensor_contract"]["support_filter_valid"] = False
    summary["runtime"]["sensor_contract"]["positive_control_resolution"] = (
        "unfiltered net force matching object weight validates the net reporter posthoc; the frozen filtered-support "
        "positive-control gate failed, and link4/link5 zero channels lack independent positive controls"
    )
    mirror = summary.pop("actual_physx_cooked_hull", None)
    if mirror is None:
        mirror = summary["default_physx_mirror_recook"]
    summary["default_physx_mirror_recook"] = mirror
    legacy_snapshot = summary["artifacts"].pop("actual_cooked_hull_snapshot", None)
    if legacy_snapshot is not None:
        summary["artifacts"]["default_physx_mirror_recook_snapshot"] = legacy_snapshot
    mirror["representation_label"] = "default PhysX mirror recook of the exact live-stage source mesh"
    mirror["direct_live_collider_cook_extracted"] = False
    mirror["live_collider_parity_limit"] = (
        "source topology/transform and convexHull approximation match, but rigid-body ownership and all live cook "
        "attributes were not reproduced or verified"
    )
    classification = _classify(
        str(mirror["query"]["verdict"]),
        bool(summary["runtime"]["sensor_contract"]["hard_contract_pass"]),
        reporter_diagnostic_pass,
        preregistered_support_pass,
        True,
        stats,
    )
    summary["classification"] = classification
    summary["verdict"] = classification["verdict"]
    from scipy.spatial import ConvexHull

    canonical = summary["offline"]["canonical"]
    mirror_vertices = np.asarray(mirror["cooked_vertices_link5_local_m"], dtype=np.float64)
    link5_rot = np.asarray(canonical["commanded_link5_rot_local"], dtype=np.float64)
    link5_pos = np.asarray(canonical["commanded_link5_pos_local_m"], dtype=np.float64)
    mirror_vertices_world = (link5_rot @ mirror_vertices.T).T + link5_pos
    mirror_faces = ConvexHull(mirror_vertices).simplices
    mirror_figure_result = dict(mirror["query"])
    mirror_figure_result["raw_signed_distance_mm"] = float(
        summary["offline"]["raw_stl_negative_control"]["signed_distance_mm"]
    )
    offline_png = args.out_dir / "d332_offline_hull_overlap.png"
    _write_offline_figure(
        offline_png,
        hull_vertices_world=mirror_vertices_world,
        hull_faces=mirror_faces,
        canonical=canonical,
        result=mirror_figure_result,
    )
    event_step = int(stats["first_robot_contact_step"])
    if event_step < 0:
        event_step = max(0, int(stats["object_disturbance_start_step"]))
    event_png = args.out_dir / "d332_teleport_first_event.png"
    final_png = args.out_dir / "d332_teleport_final.png"
    _write_reanalysis_snapshot(
        event_png,
        target_rows[event_step],
        summary,
        title=f"D332 first observed post-step event (physics step {event_step})",
    )
    _write_reanalysis_snapshot(
        final_png,
        target_rows[-1],
        summary,
        title=f"D332 canonical teleport final (physics step {len(target_rows) - 1})",
    )
    summary["visualization"]["runtime_snapshots_regenerated_from_trace_csv"] = True
    summary["visualization"]["runtime_snapshot_frame_count"] = 6
    summary["visualization"]["runtime_snapshot_provenance"] = (
        "CSV actual joints/object pose with repo FK reconstruction; original live frames remain in the RRD"
    )
    summary["visualization"]["offline_snapshot_regenerated_from_summary"] = True
    summary["reanalysis"] = {
        "method": "structured CSV reanalysis only; no additional physics run",
        "baseline_trace_sha256": _sha256(baseline_path),
        "target_trace_sha256": _sha256(target_path),
        "offline_snapshot_sha256": _sha256(offline_png),
        "first_event_snapshot_sha256": _sha256(event_png),
        "final_snapshot_sha256": _sha256(final_png),
        "invalid_attempt0": _rel(args.out_dir / "attempt0_invalid_table_filter"),
    }
    reanalysis_path = args.out_dir / "d332_contact_witness_reanalysis.json"
    _json_dump(
        reanalysis_path,
        {
            "artifact": "D332_CONTACT_WITNESS_REANALYSIS",
            "baseline": baseline,
            "target_first_contact_witness": stats.get("first_contact_witness"),
            "support_domain_audit": summary["runtime"]["support_domain_audit"],
            "classification": classification,
            "source_trace_sha256": summary["reanalysis"],
        },
    )
    summary["artifacts"]["contact_witness_reanalysis"] = _rel(reanalysis_path)
    _json_dump(summary_path, summary)
    _write_summary_markdown(args.out_dir / "g0a_d332_static_collision_summary.md", summary)
    return summary


def _run_runtime(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    offline_path = args.out_dir / "d332_offline_geometry_precheck.json"
    if not offline_path.is_file():
        raise FileNotFoundError(f"run --stage offline first: {offline_path}")
    offline = json.loads(offline_path.read_text(encoding="utf-8"))
    canonical = dict(offline["canonical"])
    if offline["geometry_contract"]["link5_mesh_sha256"] != _sha256(LINK5_MESH):
        raise RuntimeError("link5 mesh changed after offline stage")
    recomputed = _canonical_contract()
    if not np.allclose(
        np.asarray(canonical["commanded_joint_rad"]),
        np.asarray(recomputed["commanded_joint_rad"]),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("canonical IK changed after offline stage")
    versions = _runtime_versions()
    inner = _make_runtime_env(args)
    try:
        inner.reset(seed=int(args.seed))
        cooked = _extract_default_physx_mirror_recook(inner, canonical)
        actual_query = cooked["query"]
        canonical["offline_witness_cylinder_local_m"] = actual_query["nearest_point_cylinder_m"]
        offline_png = args.out_dir / "d332_offline_hull_overlap.png"
        figure_result = dict(actual_query)
        figure_result["raw_signed_distance_mm"] = float(
            offline["raw_stl_negative_control"]["signed_distance_mm"]
        )
        _write_offline_figure(
            offline_png,
            hull_vertices_world=np.asarray(cooked.pop("figure_vertices_world"), dtype=np.float64),
            hull_faces=np.asarray(cooked.pop("figure_faces"), dtype=np.int64),
            canonical=canonical,
            result=figure_result,
        )
        sensor_contract, filter_map = _sensor_contract(inner)
        q_home = np.radians(np.asarray(HOME_DEG, dtype=np.float64))
        q_home[5] = 0.0
        home_target = _write_exact_state(inner, q_home, OBJECT_CENTER_LOCAL_M)
        baseline_start_w = (
            inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64) + OBJECT_CENTER_LOCAL_M
        )
        baseline_rows: list[dict[str, Any]] = []
        for step in range(BASELINE_PHYSICS_STEPS):
            _physics_step(inner)
            contact = _contact_state(inner._d332_contact_sensor, filter_map)
            baseline_rows.append(
                _state_row(
                    inner,
                    phase="robot_free_baseline",
                    step=step,
                    command_target=home_target,
                    canonical=canonical,
                    object_start_w=baseline_start_w,
                    contact=contact,
                )
            )
        baseline_stats = _baseline_statistics(baseline_rows)

        q_target = np.asarray(canonical["commanded_joint_rad"], dtype=np.float64)
        command_target = _write_exact_state(inner, q_target, OBJECT_CENTER_LOCAL_M)
        object_start_w = (
            inner.scene.env_origins[0].detach().cpu().numpy().astype(np.float64) + OBJECT_CENTER_LOCAL_M
        )
        target_rows: list[dict[str, Any]] = []
        for step in range(TARGET_SETTLE_PHYSICS_STEPS):
            _physics_step(inner)
            contact = _contact_state(inner._d332_contact_sensor, filter_map)
            target_rows.append(
                _state_row(
                    inner,
                    phase="canonical_target_settle",
                    step=step,
                    command_target=command_target,
                    canonical=canonical,
                    object_start_w=object_start_w,
                    contact=contact,
                )
            )

        stats = _trace_statistics(target_rows)
        classification = _classify(
            str(actual_query["verdict"]),
            bool(sensor_contract["hard_contract_pass"]),
            bool(baseline_stats["posthoc_reporter_diagnostic_pass"]),
            bool(baseline_stats["preregistered_positive_control_pass"]),
            bool(OBJECT_CENTER_LOCAL_M[2] - 0.5 * CYLINDER_HEIGHT_M < 0.0),
            stats,
        )

        onset_candidates = [
            value
            for value in (int(stats["first_robot_contact_step"]), int(stats["object_disturbance_start_step"]))
            if value >= 0
        ]
        event_step = min(onset_candidates) if onset_candidates else 0
        event_png = args.out_dir / "d332_teleport_first_event.png"
        final_png = args.out_dir / "d332_teleport_final.png"
        event_snapshot = _write_runtime_snapshot(
            event_png,
            target_rows[event_step],
            f"D332 first observed post-step event (physics step {event_step})",
        )
        final_snapshot = _write_runtime_snapshot(
            final_png,
            target_rows[-1],
            f"D332 canonical teleport final (physics step {len(target_rows) - 1})",
        )
        marker_status = draw_frames(target_rows[-1]["frames"], prim_path="/World/D332CanonicalFrames")

        baseline_csv = args.out_dir / "d332_contact_baseline_trace.csv"
        target_csv = args.out_dir / "d332_teleport_settle_trace.csv"
        _write_trace_csv(baseline_csv, baseline_rows)
        _write_trace_csv(target_csv, target_rows)

        rrd_path = args.out_dir / "d332_contact_disturbance_trace_v2.rrd"
        rrd_status = log_rerun(
            rrd_path,
            frames=target_rows[-1]["frames"],
            joint_state={
                "label": "d332_canonical_static_collision_discriminator",
                "physics_steps": TARGET_SETTLE_PHYSICS_STEPS,
                "physics_dt_s": PHYSICS_DT_S,
                "object": "cylinder_d34_h90",
            },
            joint_trace=target_rows,
            urdf_path=args.urdf_path,
            live_viewer=False,
            app_id="roarm_g0a_d332_static_collision_discriminator",
        )
        if bool(rrd_status.get("ok")):
            rrd_status["nonzero_file"] = bool(rrd_path.is_file() and rrd_path.stat().st_size > 0)

        summary = {
            "verdict": classification["verdict"],
            "active_case": "G0a cylinder D34xH90 alignment-only static discriminator",
            "new_variable": [],
            "offline": offline,
            "default_physx_mirror_recook": cooked,
            "runtime": {
                "seed": int(args.seed),
                "num_envs": 1,
                "physics_dt_s": float(inner.physics_dt),
                "baseline_physics_steps": BASELINE_PHYSICS_STEPS,
                "target_settle_physics_steps": TARGET_SETTLE_PHYSICS_STEPS,
                "sensor_contract": sensor_contract,
                "baseline": baseline_stats,
                "target_settle": stats,
                "diagnostic_thresholds": {
                    "robot_force_event_n": ROBOT_FORCE_EVENT_N,
                    "object_xy_disturbance_mm": DISTURBANCE_XY_M * 1000.0,
                    "object_tilt_disturbance_deg": DISTURBANCE_TILT_DEG,
                    "consecutive_steps": CONSECUTIVE_EVENT_STEPS,
                },
            },
            "classification": classification,
            "visualization": {
                "snapshots": [
                    _rel(offline_png),
                    event_snapshot,
                    final_snapshot,
                ],
                "snapshot_count": 3,
                "marker_status": marker_status,
                "rrd_status": rrd_status,
            },
            "artifacts": {
                "offline_json": _rel(offline_path),
                "default_physx_mirror_recook_snapshot": _rel(offline_png),
                "canonical_joint_csv": _rel(args.out_dir / "d332_canonical_joint_targets.csv"),
                "baseline_trace_csv": _rel(baseline_csv),
                "target_settle_trace_csv": _rel(target_csv),
                "summary_json": _rel(args.out_dir / "g0a_d332_static_collision_summary.json"),
                "summary_markdown": _rel(args.out_dir / "g0a_d332_static_collision_summary.md"),
                "rrd": _rel(rrd_path),
            },
            "environment": versions,
            "non_goals_respected": [
                "no collision mesh rewrite",
                "no target/gate/offset/standoff tuning",
                "no waypoint or approach path",
                "no gripper close/grasp/lift/G0b",
                "no RL/PPO/randomization/render/video/VLA/RoArm/B200/cube",
            ],
        }
        summary_json = args.out_dir / "g0a_d332_static_collision_summary.json"
        summary_md = args.out_dir / "g0a_d332_static_collision_summary.md"
        _json_dump(summary_json, summary)
        _write_summary_markdown(summary_md, summary)
        return summary
    finally:
        inner.close()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=("offline", "runtime", "reanalyze"), required=True)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robot_usd_path", type=Path, default=DEFAULT_ROBOT_USD)
    parser.add_argument("--urdf_path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=33201)


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--stage", choices=("offline", "runtime", "reanalyze"), required=True)
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.stage == "offline":
        parser = argparse.ArgumentParser(description=__doc__)
        _add_common_args(parser)
        args = parser.parse_args()
        payload = _run_offline(args)
        print(
            "D332 offline precheck: "
            f"full_hull={payload['mathematical_full_hull']['signed_distance_mm']:.6f}mm "
            f"({payload['mathematical_full_hull']['verdict']}), "
            f"raw={payload['raw_stl_negative_control']['signed_distance_mm']:.6f}mm"
        )
        return 0

    if pre_args.stage == "reanalyze":
        parser = argparse.ArgumentParser(description=__doc__)
        _add_common_args(parser)
        args = parser.parse_args()
        summary = _run_reanalysis(args)
        stats = summary["runtime"]["target_settle"]
        print(
            f"{summary['verdict']}: suspected={stats['suspected_link']} "
            f"contact_step={stats['first_robot_contact_step']} "
            f"disturbance_step={stats['object_disturbance_start_step']}",
        )
        return 0

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        try:
            summary = _run_runtime(args)
            stats = summary["runtime"]["target_settle"]
            print(
                f"{summary['verdict']}: contact_step={stats['first_robot_contact_step']} "
                f"disturbance_step={stats['object_disturbance_start_step']} "
                f"final_disp={stats['final_object_disp_xy_mm']:.6f}mm",
                flush=True,
            )
            return 0
        except Exception:
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            return 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
