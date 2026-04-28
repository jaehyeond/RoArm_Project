"""Replay v6 parquet trajectories in Isaac Sim with calibrated Kinect pose.

Improvements over replay_trajectory_sim.py (2026-04-13):
  - SeattleLabTable USD (not a blank ground plane).
  - Pink sponge cuboid at per-episode pose from sponge_poses.json.
  - Kinect camera at calibrated extrinsics (kinect_calib.yaml) — no approx pose.
  - Table surface at URDF world Z = -12.12 mm (from estimate_table_plane.py).

Step 5a (single-frame): boot scene, render ep<E> frame<F> PNG to
    sim_renders_v2/ep<EEEE>_frame<FFFF>.png

Step 5b (--all-frames): loop the full episode, render every frame to
    sim_renders_v2/episode_<EEE>/frame_<FFFF>.png  (sim_real_compare.py layout)
and dump per-frame joint tracking RMSE to
    sim_renders_v2/episode_<EEE>/tracking_rmse.json

Run:
    conda run -n isaaclab python sim_scripts/replay_v6_sim.py \
        --episode 0 --frame 0 --output-dir sim_renders_v2
    conda run -n isaaclab python sim_scripts/replay_v6_sim.py \
        --episode 0 --all-frames --output-dir sim_renders_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
URDF_PATH_DEFAULT = "/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf"
PARQUET_DEFAULT = str(REPO / "lerobot_dataset_v6" / "data" / "chunk-000" / "file-000.parquet")
CALIB_DEFAULT = str(REPO / "sim_scripts" / "kinect_calib.yaml")
POSES_DEFAULT = str(REPO / "sim_scripts" / "sponge_poses.json")
TABLE_PLANE_DEFAULT = str(REPO / "sim_scripts" / "table_plane.json")

JOINT_NAMES = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
]

SPONGE_SIZE_M = (0.022, 0.047, 0.125)  # x, y, z when upright (4/27 update: 20mm->22mm, user re-measured)
SPONGE_COLOR = (0.95, 0.45, 0.60)
SPONGE_PRIM_PATH = "/World/Sponge"
TABLE_PRIM_PATH = "/World/Table"
CAM_PRIM_PATH = "/World/SimCam"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0, help="Single-episode mode (legacy)")
    p.add_argument("--episodes", type=str, default=None,
                   help="Comma-separated ep indices, e.g. '1,17,39,40'. Forces --all-frames.")
    p.add_argument("--all-eps", action="store_true",
                   help="Render every episode present in parquet. Forces --all-frames.")
    p.add_argument("--frame", type=int, default=0, help="Single-frame mode: frame index to render")
    p.add_argument("--all-frames", action="store_true", help="(5b) Render all frames in episode")
    p.add_argument("--output-dir", type=str, default=str(REPO / "sim_renders_v2"))
    p.add_argument("--parquet", type=str, default=PARQUET_DEFAULT)
    p.add_argument("--urdf", type=str, default=URDF_PATH_DEFAULT)
    p.add_argument("--calib", type=str, default=CALIB_DEFAULT)
    p.add_argument("--poses", type=str, default=POSES_DEFAULT)
    p.add_argument("--table-plane", type=str, default=TABLE_PLANE_DEFAULT)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--settle-steps", type=int, default=30)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip episodes whose output dir already has tracking_rmse.json")
    return p.parse_args()


def load_calib(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    intr = c["intrinsics"]
    R = np.asarray(c["extrinsics"]["rotation_matrix"], dtype=np.float64)
    t = np.asarray(c["extrinsics"]["translation_m"], dtype=np.float64)
    return intr, R, t


def load_sponge_pose(path, ep_idx):
    with open(path) as f:
        d = json.load(f)
    eps = d["episodes"]
    key = f"{ep_idx:04d}"
    if key not in eps:
        raise KeyError(f"ep {key} missing in {path}")
    return np.asarray(eps[key]["pos_m"]), np.asarray(eps[key]["rot_quat_wxyz"])


def load_table_z(path):
    if not os.path.exists(path):
        print(f"WARN: {path} missing — using z=0 fallback")
        return 0.0
    with open(path) as f:
        d = json.load(f)
    return float(d["table_z_urdf_world_m"])


def fov_x_from_intrinsics(fx, width):
    """Horizontal FOV (deg) from pinhole focal length and image width."""
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * fx))))


def make_camera_xform(R_cv, t):
    """Build 4x4 camera-to-world matrix from Kinect extrinsics (OpenCV) for USD.

    Kinect/OpenCV: camera +X right, +Y down, +Z forward.
    USD/OpenGL: camera +X right, +Y up, -Z forward (camera looks down -Z).
    Rotate basis: R_usd = R_cv @ diag(1, -1, -1).
    """
    flip = np.diag([1.0, -1.0, -1.0])
    R_usd = R_cv @ flip
    M = np.eye(4)
    M[:3, :3] = R_usd
    M[:3, 3] = t
    return M


def main():
    args = parse_args()

    # ------------------------------------------------------------
    # Pre-flight: load all non-sim inputs (fail fast).
    # ------------------------------------------------------------
    for p in (args.parquet, args.urdf, args.calib, args.poses):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing file {p}")

    df = pd.read_parquet(args.parquet)
    all_eps_in_parquet = sorted(df["episode_index"].unique().tolist())

    # Resolve episode list.
    if args.all_eps:
        ep_list = [int(e) for e in all_eps_in_parquet]
        multi_episode = True
        force_all_frames = True
    elif args.episodes is not None:
        ep_list = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
        multi_episode = True
        force_all_frames = True
    else:
        ep_list = [int(args.episode)]
        multi_episode = False
        force_all_frames = False

    do_all_frames = args.all_frames or force_all_frames

    # Validate + preload per-ep states and sponge poses.
    per_ep_states = {}
    per_ep_sponge = {}
    for ep in ep_list:
        m = df["episode_index"].values == ep
        if not m.any():
            sys.exit(f"ERROR: episode {ep} not found in parquet")
        per_ep_states[ep] = np.array(df[m]["observation.state"].tolist())  # (T, 6) deg
        per_ep_sponge[ep] = load_sponge_pose(args.poses, ep)

    # Skip-existing handling (multi-ep only).
    skipped = []
    if args.skip_existing and multi_episode:
        remaining = []
        for ep in ep_list:
            tr_path = os.path.join(args.output_dir, f"episode_{ep:03d}", "tracking_rmse.json")
            if os.path.exists(tr_path):
                skipped.append(ep)
            else:
                remaining.append(ep)
        if skipped:
            print(f"Skip-existing: {len(skipped)} eps already rendered → {skipped[:10]}{'...' if len(skipped)>10 else ''}")
        ep_list = remaining

    if not ep_list:
        print("Nothing to render. Done.")
        return

    # Single-frame mode: validate frame idx vs ep length.
    if not do_all_frames:
        T = len(per_ep_states[ep_list[0]])
        if args.frame >= T:
            sys.exit(f"ERROR: frame {args.frame} >= T={T} for ep {ep_list[0]}")

    print(f"Parquet: {len(all_eps_in_parquet)} eps total. Rendering {len(ep_list)} ep(s): "
          f"{ep_list if len(ep_list)<=10 else str(ep_list[:10])+'...'}")
    if do_all_frames:
        total_frames = sum(len(per_ep_states[e]) for e in ep_list)
        print(f"  mode: all-frames, total {total_frames} frames")
    else:
        print(f"  mode: single-frame {args.frame}")

    intr, R_cv, t_cv = load_calib(args.calib)
    print(f"Calib: t={t_cv.tolist()} fx={intr['fx']:.1f} fy={intr['fy']:.1f}")

    table_z = load_table_z(args.table_plane)
    print(f"Table surface Z (URDF world) = {table_z*1000:+.1f} mm")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Boot Isaac Sim.
    # ------------------------------------------------------------
    print("Booting Isaac Sim (headless)...")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    import omni.kit.commands
    import omni.usd
    from pxr import UsdGeom, UsdLux, Gf, UsdPhysics, Sdf, UsdShade
    import omni.replicator.core as rep

    stage = omni.usd.get_context().get_stage()

    # World root
    UsdGeom.Xform.Define(stage, "/World")

    # Dome light (bright enough for SigLIP-relevant color fidelity).
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(2500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))

    # ------------------------------------------------------------
    # Table USD (SeattleLabTable from Nucleus).
    # ------------------------------------------------------------
    # isaaclab base pkg is not editable-installed — read Nucleus root from carb.
    import carb
    nucleus_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    if not nucleus_root:
        nucleus_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
        print(f"WARN: carb nucleus setting empty, fallback {nucleus_root}")
    table_usd = f"{nucleus_root}/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    print(f"Table USD: {table_usd}")
    table_prim = stage.DefinePrim(TABLE_PRIM_PATH, "Xform")
    table_prim.GetReferences().AddReference(table_usd)
    # SeattleLabTable root origin: table TOP is at local z ≈ 0. Position it so
    # its top matches measured table_z in URDF world.
    table_xform = UsdGeom.Xformable(table_prim)
    table_xform.ClearXformOpOrder()
    # Translate then rotate 90° about +Z (reach env convention).
    op_t = table_xform.AddTranslateOp()
    op_t.Set(Gf.Vec3d(0.55, 0.0, table_z))
    op_r = table_xform.AddRotateZOp()
    op_r.Set(90.0)

    # ------------------------------------------------------------
    # Robot URDF import.
    # ------------------------------------------------------------
    print(f"Importing URDF: {args.urdf}")
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.set_fix_base(True)
    import_config.set_make_default_prim(True)
    import_config.set_distance_scale(1.0)
    import_config.set_create_physics_scene(True)
    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile", urdf_path=args.urdf, import_config=import_config
    )
    if not result:
        app.close()
        sys.exit("ERROR: URDF parse failed")
    omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_path=args.urdf,
        urdf_robot=robot_model,
        import_config=import_config,
        dest_path="",
    )
    for _ in range(10):
        app.update()

    # ------------------------------------------------------------
    # Sponge cuboid (pink). Pose is updated per-episode inside the main loop.
    # ------------------------------------------------------------
    sponge_prim = UsdGeom.Cube.Define(stage, SPONGE_PRIM_PATH)
    # Cube default extent: [-1, +1]. Scale to half-size.
    hx, hy, hz = [s / 2.0 for s in SPONGE_SIZE_M]
    sp_xform = UsdGeom.Xformable(sponge_prim.GetPrim())
    sp_xform.ClearXformOpOrder()
    # Placeholder transform; will be overwritten per-ep. Order: translate → rotate → scale.
    sponge_op_t = sp_xform.AddTranslateOp()
    sponge_op_t.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    sponge_op_r = sp_xform.AddOrientOp()
    sponge_op_r.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    sponge_op_s = sp_xform.AddScaleOp()
    sponge_op_s.Set(Gf.Vec3f(hx, hy, hz))

    # Pink diffuse material.
    mat_path = "/World/Looks/SpongeMat"
    mat_prim = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*SPONGE_COLOR))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat_prim.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(sponge_prim.GetPrim()).Bind(mat_prim)

    for _ in range(5):
        app.update()

    # ------------------------------------------------------------
    # Articulation (after URDF stage settles).
    # ------------------------------------------------------------
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation

    world = World(stage_units_in_meters=1.0)
    robot = world.scene.add(Articulation(prim_path="/roarm_m3", name="roarm_m3"))
    world.reset()
    print(f"Robot DOFs={robot.num_dof}, names={robot.dof_names}")
    for i, name in enumerate(JOINT_NAMES):
        assert robot.dof_names[i] == name, f"DOF mismatch at {i}: want {name} got {robot.dof_names[i]}"

    # ------------------------------------------------------------
    # Camera at calibrated Kinect pose.
    # ------------------------------------------------------------
    cam_prim = UsdGeom.Camera.Define(stage, CAM_PRIM_PATH)
    # Focal length: USD uses mm with a 24mm-ish aperture. We emulate Kinect intrinsics
    # by matching horizontal FOV. Hfov = 2*atan(W / (2*fx)).
    hfov_deg = fov_x_from_intrinsics(intr["fx"], args.width)
    # USD default aperture: 20.955mm (equivalent to 36mm/1.72x crop? actually 20.955mm = full frame /w 1.72x).
    # Use horizontal aperture relation: focal_mm = (0.5 * aperture_mm) / tan(Hfov/2).
    aperture_h_mm = 20.955
    focal_mm = 0.5 * aperture_h_mm / np.tan(np.radians(hfov_deg) / 2.0)
    cam_prim.CreateFocalLengthAttr(float(focal_mm))
    cam_prim.CreateHorizontalApertureAttr(aperture_h_mm)
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))

    # Apply calibrated extrinsics: transform is camera→world (OpenCV→USD basis flip).
    M = make_camera_xform(R_cv, t_cv)
    gf_mat = Gf.Matrix4d()
    for row in range(4):
        for col in range(4):
            gf_mat[row, col] = float(M[col, row])  # transpose into Gf row-major
    cam_xform = UsdGeom.Xformable(cam_prim.GetPrim())
    cam_xform.ClearXformOpOrder()
    cam_xform.AddTransformOp().Set(gf_mat)

    print(f"Camera @ t={t_cv.tolist()} HFOV={hfov_deg:.2f}° focal_mm={focal_mm:.2f}")

    # ------------------------------------------------------------
    # Render product.
    # ------------------------------------------------------------
    rp = rep.create.render_product(CAM_PRIM_PATH, (args.width, args.height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([rp])
    for _ in range(args.settle_steps):
        app.update()

    # ------------------------------------------------------------
    # Render loop over episodes (single Sim boot, per-ep sponge pose update).
    # ------------------------------------------------------------
    from PIL import Image
    import time

    global_summaries = []  # list of per-ep summary dicts
    overall_t0 = time.time()

    for ep_idx, ep in enumerate(ep_list):
        ep_t0 = time.time()
        states_deg = per_ep_states[ep]
        T = len(states_deg)

        if do_all_frames:
            frame_indices = list(range(T))
            frame_dir = os.path.join(args.output_dir, f"episode_{ep:03d}")
            os.makedirs(frame_dir, exist_ok=True)
        else:
            frame_indices = [args.frame]
            frame_dir = args.output_dir

        # Update sponge transform for this episode.
        sp_pos, sp_quat = per_ep_sponge[ep]
        sponge_op_t.Set(Gf.Vec3d(float(sp_pos[0]), float(sp_pos[1]), float(sp_pos[2])))
        sponge_op_r.Set(Gf.Quatf(float(sp_quat[0]), float(sp_quat[1]), float(sp_quat[2]), float(sp_quat[3])))
        # Let the USD change propagate + annotator settle.
        for _ in range(5):
            app.update()

        print(f"\n=== ep {ep} ({ep_idx+1}/{len(ep_list)})  T={T} frames  "
              f"sponge_pos={[round(x,3) for x in sp_pos]} ===")

        tracking_log = []
        for f_idx in frame_indices:
            target_deg = states_deg[f_idx]
            target_rad = np.radians(target_deg)
            robot.set_joint_positions(target_rad)

            for _ in range(3):
                world.step(render=True)
                app.update()
            rep.orchestrator.step()
            app.update()

            rgb = rgb_annotator.get_data()
            if rgb is None or rgb.ndim != 3:
                app.close()
                sys.exit(f"ERROR: RGB annotator returned {type(rgb)} shape={getattr(rgb, 'shape', None)} at ep {ep} frame {f_idx}")

            if do_all_frames:
                out_path = os.path.join(frame_dir, f"frame_{f_idx:04d}.png")
            else:
                out_path = os.path.join(frame_dir, f"ep{ep:04d}_frame{f_idx:04d}.png")
            Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(out_path)

            actual_rad = robot.get_joint_positions()
            err_deg = np.degrees(actual_rad) - target_deg
            rmse = float(np.sqrt(np.mean(err_deg ** 2)))
            tracking_log.append({
                "frame": int(f_idx),
                "target_deg": [float(x) for x in target_deg],
                "actual_deg": [float(x) for x in np.degrees(actual_rad)],
                "err_deg": [float(x) for x in err_deg],
                "rmse_deg": rmse,
                "max_abs_err_deg": float(np.max(np.abs(err_deg))),
            })

            if (not do_all_frames) or f_idx == 0 or f_idx == frame_indices[-1] or (f_idx % 40 == 0):
                print(f"  ep{ep} f{f_idx:4d}: rmse={rmse:.3f}°  max|err|={np.max(np.abs(err_deg)):.3f}°")

        # Aggregate per-ep stats.
        if tracking_log:
            all_rmse = np.array([t["rmse_deg"] for t in tracking_log])
            all_max = np.array([t["max_abs_err_deg"] for t in tracking_log])
            per_joint_abs = np.array([[abs(e) for e in t["err_deg"]] for t in tracking_log])
            ep_summary = {
                "episode": int(ep),
                "num_frames": len(tracking_log),
                "rmse_deg_mean": float(all_rmse.mean()),
                "rmse_deg_max": float(all_rmse.max()),
                "max_abs_err_deg_global": float(all_max.max()),
                "per_joint_abs_err_mean_deg": [float(x) for x in per_joint_abs.mean(axis=0)],
                "per_joint_abs_err_max_deg": [float(x) for x in per_joint_abs.max(axis=0)],
                "joint_names": JOINT_NAMES,
                "elapsed_s": round(time.time() - ep_t0, 2),
            }
            if do_all_frames:
                log_path = os.path.join(frame_dir, "tracking_rmse.json")
                with open(log_path, "w") as fh:
                    json.dump({"summary": ep_summary, "per_frame": tracking_log}, fh, indent=2)
            global_summaries.append(ep_summary)
            print(f"  ep{ep} done  rmse mean={ep_summary['rmse_deg_mean']:.3f}° "
                  f"max={ep_summary['rmse_deg_max']:.3f}°  elapsed={ep_summary['elapsed_s']}s")

    # Overall summary (multi-ep mode).
    if multi_episode and global_summaries:
        overall = {
            "eps_rendered": [s["episode"] for s in global_summaries],
            "eps_skipped": skipped,
            "total_frames": sum(s["num_frames"] for s in global_summaries),
            "rmse_mean_across_eps": float(np.mean([s["rmse_deg_mean"] for s in global_summaries])),
            "rmse_max_across_eps": float(np.max([s["rmse_deg_max"] for s in global_summaries])),
            "max_abs_err_deg_global": float(np.max([s["max_abs_err_deg_global"] for s in global_summaries])),
            "total_elapsed_s": round(time.time() - overall_t0, 2),
        }
        overall_path = os.path.join(args.output_dir, "tracking_overall.json")
        with open(overall_path, "w") as fh:
            json.dump({"overall": overall, "per_episode": global_summaries}, fh, indent=2)
        print(f"\n=== OVERALL ({len(global_summaries)} eps rendered, {overall['total_frames']} frames) ===")
        print(f"  rmse mean across eps = {overall['rmse_mean_across_eps']:.3f}°")
        print(f"  rmse max  across eps = {overall['rmse_max_across_eps']:.3f}°")
        print(f"  global max |err|     = {overall['max_abs_err_deg_global']:.3f}°")
        print(f"  total elapsed        = {overall['total_elapsed_s']}s "
              f"({overall['total_elapsed_s']/max(1,overall['total_frames'])*1000:.0f} ms/frame avg)")
        print(f"  → {overall_path}")

    app.close()
    print("Done.")


if __name__ == "__main__":
    main()
