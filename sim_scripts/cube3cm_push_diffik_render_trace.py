"""Render recorded cube3cm DiffIK traces to frames and MP4.

This is visualization only. It replays joint/cube poses from a trace CSV produced
by cube3cm_push_diffik_probe.py and does not run training, dataset generation, or
new physics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from roarm_kinematics import Tmat, Trot_z


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = REPO / "local_assets/roarm_m3/urdf/meshes"
DEFAULT_LOCAL_USD = (
    REPO
    / "b200_backup_20260522_final/tmp_p7/"
    "p7_branch_b_cube2cm_opposing_jaw_v7_collision_usd_d024/roarm_m3.usd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_csv", required=True)
    parser.add_argument("--mesh_dir", type=str, default=str(DEFAULT_MESH_DIR))
    parser.add_argument("--robot_usd_path", type=str, default=str(DEFAULT_LOCAL_USD))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_mp4", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--env_spacing_m", type=float, default=3.0)
    parser.add_argument("--layout_cols", type=int, default=0)
    parser.add_argument("--tile_spacing_x_m", type=float, default=0.90)
    parser.add_argument("--tile_spacing_y_m", type=float, default=0.72)
    parser.add_argument("--camera_target_push_m", type=float, default=0.025)
    parser.add_argument("--max_frames", type=int, default=0)
    return parser.parse_args()


def load_trace(path: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = 0.0
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"empty trace: {path}")
    return rows


def infer_env_origin(first: dict[str, float], env_spacing_m: float) -> np.ndarray:
    if "env_origin_x_m" in first:
        return np.array(
            [first["env_origin_x_m"], first["env_origin_y_m"], first["env_origin_z_m"]],
            dtype=np.float64,
        )
    return np.array(
        [
            round(first["cube_x_m"] / env_spacing_m) * env_spacing_m,
            round(first["cube_y_m"] / env_spacing_m) * env_spacing_m,
            0.0,
        ],
        dtype=np.float64,
    )


def local_xyz(row: dict[str, float], prefix: str, origin: np.ndarray) -> np.ndarray:
    return np.array([row[f"{prefix}_x_m"], row[f"{prefix}_y_m"], row[f"{prefix}_z_m"]], dtype=np.float64) - origin


def load_binary_stl(path: Path, scale: float = 0.001) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 84:
        raise RuntimeError(f"STL too small: {path}")
    tri_count = int(np.frombuffer(data[80:84], dtype="<u4", count=1)[0])
    expected = 84 + tri_count * 50
    if len(data) != expected:
        raise RuntimeError(f"expected binary STL for visual mesh: {path} bytes={len(data)} expected={expected}")
    raw = np.frombuffer(data, dtype=np.uint8, offset=84).reshape(tri_count, 50)
    floats = raw[:, :48].copy().view("<f4").reshape(tri_count, 12)
    normals = np.repeat(floats[:, 0:3], 3, axis=0).astype(np.float32)
    points = floats[:, 3:12].reshape(tri_count * 3, 3).astype(np.float32) * float(scale)
    counts = np.full(tri_count, 3, dtype=np.int32)
    indices = np.arange(tri_count * 3, dtype=np.int32)
    return points, counts, indices, normals


def roarm_link_transforms(joints_rad: np.ndarray) -> dict[str, np.ndarray]:
    q = np.asarray(joints_rad, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transforms: dict[str, np.ndarray] = {}
    transform = transform @ Tmat([0.0, 0.0, 0.0701], [0.0, 0.0, 0.0])
    transforms["base_link"] = transform.copy()
    transform = transform @ Tmat([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) @ Trot_z(q[0])
    transforms["link1"] = transform.copy()
    transform = transform @ Tmat([0.0, 0.0, 0.05196], [-math.pi / 2, -math.pi / 2, 0.0]) @ Trot_z(q[1])
    transforms["link2"] = transform.copy()
    transform = transform @ Tmat([0.236815, 0.030002, 0.0], [0.0, 0.0, math.pi / 2]) @ Trot_z(q[2])
    transforms["link3"] = transform.copy()
    transform = transform @ Tmat([0.0, -0.144586, 0.0], [0.0, 0.0, 0.0]) @ Trot_z(q[3])
    transforms["link4"] = transform.copy()
    transform = transform @ Tmat([0.015147, -0.053653, 0.0], [math.pi / 2, math.pi / 2, 0.0]) @ Trot_z(q[4])
    transforms["link5"] = transform.copy()
    transform = transform @ Tmat([0.0, 0.018821, 0.052035], [-math.pi / 2, -math.pi / 2, 0.0]) @ Trot_z(q[5])
    transforms["gripper_link"] = transform.copy()
    return transforms


ROARM_VISUAL_MESHES = {
    "base_link": "base_link.stl",
    "link1": "link1.stl",
    "link2": "link2.stl",
    "link3": "link3.stl",
    "link4": "link4.stl",
    "link5": "link5.stl",
    "gripper_link": "gripper_link.stl",
}


def group_frames(rows: list[dict[str, float]]) -> list[tuple[int, dict[int, dict[str, float]]]]:
    grouped: dict[int, dict[int, dict[str, float]]] = {}
    for row in rows:
        frame = int(row["frame"])
        env_id = int(row["env_id"])
        grouped.setdefault(frame, {})[env_id] = row
    return [(frame, grouped[frame]) for frame in sorted(grouped)]


def main() -> int:
    args = parse_args()
    all_rows = load_trace(args.trace_csv)
    env_ids = sorted({int(row["env_id"]) for row in all_rows})
    if not env_ids:
        raise RuntimeError("trace has no env_id rows")

    frames = group_frames(all_rows)
    if args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    first_by_env: dict[int, dict[str, float]] = {}
    for row in all_rows:
        first_by_env.setdefault(int(row["env_id"]), row)

    env_count = len(env_ids)
    cols = int(args.layout_cols) if int(args.layout_cols) > 0 else min(env_count, 4)
    rows_n = int(math.ceil(env_count / cols))
    origins = {env_id: infer_env_origin(first_by_env[env_id], float(args.env_spacing_m)) for env_id in env_ids}
    offsets: dict[int, np.ndarray] = {}
    for idx, env_id in enumerate(env_ids):
        col = idx % cols
        row = idx // cols
        offsets[env_id] = np.array(
            [col * float(args.tile_spacing_x_m), -row * float(args.tile_spacing_y_m), 0.0],
            dtype=np.float64,
        )

    out_dir = Path(args.output_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = Path(args.output_mp4)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": int(args.width), "height": int(args.height)})

    import cv2
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image, ImageDraw
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

    from omni.isaac.core import World

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")

    def make_material(path: str, color: tuple[float, float, float], roughness: float) -> object:
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    robot_mat = make_material("/World/Looks/RobotBlackMat", (0.0, 0.0, 0.0), 0.82)
    table_mat = make_material("/World/Looks/TableGreyMat", (0.50, 0.50, 0.50), 0.88)
    cube_mat = make_material("/World/Looks/CubePinkMat", (1.00, 0.16, 0.52), 0.58)
    tcp_mat = make_material("/World/Looks/TCPBlueMat", (0.18, 0.32, 1.00), 0.50)
    target_mat = make_material("/World/Looks/TargetRedMat", (1.00, 0.10, 0.10), 0.45)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(2600.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    cube_ops: dict[int, tuple[object, object]] = {}
    tcp_ops: dict[int, object] = {}
    target_ops: dict[int, object] = {}
    robot_link_ops: dict[int, dict[str, object]] = {}
    mesh_dir = Path(args.mesh_dir)
    mesh_cache = {
        link_name: load_binary_stl(mesh_dir / mesh_file)
        for link_name, mesh_file in ROARM_VISUAL_MESHES.items()
    }
    mesh_triangle_count = {name: int(data[1].shape[0]) for name, data in mesh_cache.items()}

    for env_id in env_ids:
        offset = offsets[env_id]
        table = UsdGeom.Cube.Define(stage, f"/World/Table_{env_id}")
        table_xf = UsdGeom.Xformable(table.GetPrim())
        table_xf.ClearXformOpOrder()
        table_xf.AddTranslateOp().Set(Gf.Vec3d(*(offset + np.array([0.30, 0.0, -0.017]))))
        table_xf.AddScaleOp().Set(Gf.Vec3f(0.45, 0.32, 0.01))
        UsdShade.MaterialBindingAPI(table.GetPrim()).Bind(table_mat)

        robot_link_ops[env_id] = {}
        for link_name, (points, counts, indices, normals) in mesh_cache.items():
            mesh = UsdGeom.Mesh.Define(stage, f"/World/RoArmMesh_{env_id}_{link_name}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(robot_mat)
            xf = UsdGeom.Xformable(mesh.GetPrim())
            xf.ClearXformOpOrder()
            robot_link_ops[env_id][link_name] = xf.AddTransformOp()

        cube = UsdGeom.Cube.Define(stage, f"/World/Cube_{env_id}")
        cube_xf = UsdGeom.Xformable(cube.GetPrim())
        cube_xf.ClearXformOpOrder()
        cube_translate = cube_xf.AddTranslateOp()
        cube_orient = cube_xf.AddOrientOp()
        first = first_by_env[env_id]
        cube_x = float(first.get("cube_size_x_m", 0.03))
        cube_y = float(first.get("cube_size_y_m", cube_x))
        cube_z = float(first.get("cube_size_z_m", cube_x))
        cube_xf.AddScaleOp().Set(Gf.Vec3f(cube_x * 0.5, cube_y * 0.5, cube_z * 0.5))
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(cube_mat)
        cube_ops[env_id] = (cube_translate, cube_orient)

        for name, radius, mat, registry in [
            ("TCPMarker", 0.007, tcp_mat, tcp_ops),
            ("TargetMarker", 0.006, target_mat, target_ops),
        ]:
            sphere = UsdGeom.Sphere.Define(stage, f"/World/{name}_{env_id}")
            sphere.CreateRadiusAttr(radius)
            xf = UsdGeom.Xformable(sphere.GetPrim())
            xf.ClearXformOpOrder()
            trans = xf.AddTranslateOp()
            UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(mat)
            registry[env_id] = trans

    first_points = [
        local_xyz(first_by_env[env_id], "cube", origins[env_id]) + offsets[env_id]
        for env_id in env_ids
    ]
    first_points_arr = np.vstack(first_points)
    min_xy = first_points_arr[:, :2].min(axis=0)
    max_xy = first_points_arr[:, :2].max(axis=0)
    center_xy = (min_xy + max_xy) * 0.5
    camera_target = np.array([center_xy[0], center_xy[1], 0.065], dtype=np.float64)
    camera_eye = camera_target + np.array(
        [
            max(0.75, 0.38 * cols),
            -max(1.05, 0.62 * rows_n + 0.45),
            max(0.70, 0.48 + 0.12 * cols + 0.12 * rows_n),
        ],
        dtype=np.float64,
    )

    def look_at_matrix(eye: np.ndarray, target: np.ndarray) -> Gf.Matrix4d:
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up_guess)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        mat = np.eye(4, dtype=np.float64)
        mat[:3, 0] = right
        mat[:3, 1] = up
        mat[:3, 2] = -forward
        mat[:3, 3] = eye
        return Gf.Matrix4d(*mat.T.flatten().tolist())

    cam = UsdGeom.Camera.Define(stage, "/World/RenderCam")
    cam.CreateFocalLengthAttr(18.0 if env_count > 1 else 20.0)
    cam.CreateHorizontalApertureAttr(24.0)
    cam.CreateVerticalApertureAttr(24.0 * float(args.height) / float(args.width))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 8.0))
    cam_xf = UsdGeom.Xformable(cam.GetPrim())
    cam_xf.ClearXformOpOrder()
    cam_xf.AddTransformOp().Set(look_at_matrix(camera_eye, camera_target))

    world = World(stage_units_in_meters=1.0)
    world.reset()

    render_product = rep.create.render_product("/World/RenderCam", (int(args.width), int(args.height)))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([render_product])
    for _ in range(10):
        app.update()

    last_rows: dict[int, dict[str, float]] = {env_id: first_by_env[env_id] for env_id in env_ids}

    def set_env_frame(env_id: int, row: dict[str, float]) -> None:
        joints = np.array([row[f"arm_joint_{idx}_rad"] for idx in range(5)] + [row["gripper_joint_rad"]])
        offset = offsets[env_id]
        origin = origins[env_id]
        offset_tf = np.eye(4, dtype=np.float64)
        offset_tf[:3, 3] = offset
        for link_name, link_tf in roarm_link_transforms(joints).items():
            world_tf = offset_tf @ link_tf
            robot_link_ops[env_id][link_name].Set(Gf.Matrix4d(*world_tf.T.flatten().tolist()))
        cube_translate, cube_orient = cube_ops[env_id]
        cube_translate.Set(Gf.Vec3d(*(local_xyz(row, "cube", origin) + offset)))
        cube_orient.Set(Gf.Quatf(row["cube_qw"], row["cube_qx"], row["cube_qy"], row["cube_qz"]))
        tcp_ops[env_id].Set(Gf.Vec3d(*(local_xyz(row, "tcp", origin) + offset)))
        target_ops[env_id].Set(Gf.Vec3d(*(local_xyz(row, "target", origin) + offset)))

    def overlay(rgb: np.ndarray, frame_idx: int, frame_id: int) -> np.ndarray:
        img = Image.fromarray(rgb[:, :, :3].astype(np.uint8))
        draw = ImageDraw.Draw(img)
        height = 104 if env_count > 1 else 88
        draw.rectangle((12, 10, min(int(args.width) - 12, 980), height), fill=(0, 0, 0))
        directions = " ".join(
            f"env{env_id}=({last_rows[env_id]['push_dx']:+.0f},{last_rows[env_id]['push_dy']:+.0f})"
            for env_id in env_ids
        )
        draw.text(
            (24, 20),
            "v3 scripted IsaacLab Differential IK trace replay\n"
            f"frame={frame_idx:03d}/{len(frames)-1:03d} trace_frame={frame_id} envs={env_count} "
            f"| robot=black table=gray cube=pink\n"
            f"{directions} | training=NO dataset=NO",
            fill=(255, 255, 255),
        )
        return np.asarray(img)

    frames_written = 0
    t0 = time.time()
    for out_frame_idx, (frame_id, rows_by_env) in enumerate(frames):
        for env_id, row in rows_by_env.items():
            last_rows[env_id] = row
        for env_id in env_ids:
            set_env_frame(env_id, last_rows[env_id])
        for _ in range(2):
            world.step(render=True)
            app.update()
        rep.orchestrator.step()
        app.update()
        rgb = rgb_annot.get_data()
        if rgb is None or getattr(rgb, "ndim", 0) != 3:
            raise RuntimeError(f"rgb annotator returned {type(rgb)} at frame {out_frame_idx}")
        frame = overlay(rgb, out_frame_idx, frame_id)
        Image.fromarray(frame).save(frame_dir / f"frame_{out_frame_idx:04d}.png")
        frames_written += 1

    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (int(args.width), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open cv2 VideoWriter: {mp4_path}")
    for frame_idx in range(frames_written):
        img = cv2.imread(str(frame_dir / f"frame_{frame_idx:04d}.png"), cv2.IMREAD_COLOR)
        writer.write(img)
    writer.release()

    summary = {
        "trace_csv": str(args.trace_csv),
        "output_dir": str(out_dir),
        "output_mp4": str(mp4_path),
        "frames_written": frames_written,
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "env_ids": env_ids,
        "env_count": env_count,
        "layout_cols": cols,
        "layout_rows": rows_n,
        "env_origin_subtracted_m": {str(env_id): origins[env_id].tolist() for env_id in env_ids},
        "env_offsets_m": {str(env_id): offsets[env_id].tolist() for env_id in env_ids},
        "robot_visual_mode": "black_roarm_urdf_stl_mesh_from_trace_joints",
        "robot_mesh_dir": str(mesh_dir),
        "robot_mesh_triangle_count": mesh_triangle_count,
        "directions": {
            str(env_id): [first_by_env[env_id]["push_dx"], first_by_env[env_id]["push_dy"]]
            for env_id in env_ids
        },
        "cube_size_m": {
            str(env_id): [
                first_by_env[env_id].get("cube_size_x_m", 0.03),
                first_by_env[env_id].get("cube_size_y_m", first_by_env[env_id].get("cube_size_x_m", 0.03)),
                first_by_env[env_id].get("cube_size_z_m", first_by_env[env_id].get("cube_size_x_m", 0.03)),
            ]
            for env_id in env_ids
        },
        "colors": {
            "background": "white",
            "robot": "black",
            "table": "gray",
            "cube": "pink",
            "tcp_marker": "blue",
            "target_marker": "red",
        },
        "camera_eye_m": camera_eye.tolist(),
        "camera_target_m": camera_target.tolist(),
        "elapsed_s": time.time() - t0,
        "training": False,
        "dataset_generation": False,
        "physics_recomputed": False,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "[cube3cm_push_diffik_render_trace] "
        f"frames={frames_written} env_count={env_count} env_ids={env_ids} "
        f"mp4={mp4_path} trace={args.trace_csv} "
        "training=NO dataset_generation=NO physics_recomputed=NO",
        flush=True,
    )
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
