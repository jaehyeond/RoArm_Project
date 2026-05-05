"""Render stacking sim demos v3 (Phase ST-A Sub-A5/A6).

V3 corrections vs v2 (5/03 evening, HARD RULE #19/#20):
  - SPONGE_SIZE prim: edge-stand (LEN_LONG, WIDTH, HEIGHT_EDGE) = (0.125, 0.022, 0.047)
    was lying-flat (0.125, 0.047, 0.022) in v2.
  - TCP_TO_SPONGE_CENTER_Z = (TABLE_Z + HEIGHT_EDGE/2) − Z_TCP_GRASP_L1
                           = (-0.012117 + 0.0235) − 0.033 = -0.02162 m (held offset)
    was -SPONGE_THICK/2 = -0.011 m in v2.
  - z_floor (L1/source mid-z) = TABLE_Z + 47/2 = +0.01138 m world
    was TABLE_Z + 22/2 = -0.00112 m in v2.
  - z_l2 (L2 mid-z) = TABLE_Z + 1.5 × 47 = +0.05838 m world
    was TABLE_Z + 1.5 × 22 = +0.02088 m in v2.
  - L1 dst rotz=0° (length X) / L2 dst rotz=+90° (length Y) — same as v2.
  - Per-source orientation from layout JSON — same as v2.

Run (4090 only — HARD RULE #17):
    conda run -n isaaclab python sim_scripts/render_stacking_demos_v3.py --seeds 0
    conda run -n isaaclab python sim_scripts/render_stacking_demos_v3.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_scripts"))
from roarm_kinematics import fk_tcp  # noqa: E402
from generate_stacking_demos_v3 import (  # noqa: E402
    SPONGE_HEIGHT_EDGE, SPONGE_LEN_LONG, SPONGE_WIDTH, ORIENT_TO_WRIST_R,
    TABLE_Z, Z_TCP_GRASP_L1,
)

URDF_PATH_DEFAULT = "/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf"
CALIB_DEFAULT = str(REPO / "sim_scripts" / "kinect_calib.yaml")
TABLE_PLANE_DEFAULT = str(REPO / "sim_scripts" / "table_plane.json")
DEMOS_DEFAULT = str(REPO / "sim_demos_v3")
OUT_DEFAULT = str(REPO / "sim_renders_v5")

JOINT_NAMES = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
]

# Edge-stand cube prim: length along local-X (125mm), width along local-Y (22mm),
# height along local-Z (47mm). Per-frame Z rotation switches length-along-world-X
# (rotz=0°, L1) vs length-along-world-Y (rotz=+90°, L2).
SPONGE_SIZE = (SPONGE_LEN_LONG, SPONGE_WIDTH, SPONGE_HEIGHT_EDGE)   # 0.125 × 0.022 × 0.047

SPONGE_COLOR_L1 = (0.95, 0.45, 0.60)   # 1층 light pink
SPONGE_COLOR_L2 = (0.85, 0.30, 0.45)   # 2층 darker pink

# Held offset: sponge center Z relative to TCP at moment of grasp (constant during grip).
TCP_TO_SPONGE_CENTER_Z = (TABLE_Z + SPONGE_HEIGHT_EDGE / 2.0) - Z_TCP_GRASP_L1   # -0.02162 m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Specific seeds to render (default: all in demos_dir)")
    p.add_argument("--all", action="store_true", help="Render all demos found in demos_dir")
    p.add_argument("--demos-dir", type=str, default=DEMOS_DEFAULT)
    p.add_argument("--output-dir", type=str, default=OUT_DEFAULT)
    p.add_argument("--urdf", type=str, default=URDF_PATH_DEFAULT)
    p.add_argument("--calib", type=str, default=CALIB_DEFAULT)
    p.add_argument("--table-plane", type=str, default=TABLE_PLANE_DEFAULT)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--settle-steps", type=int, default=30)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip episodes that already have all PNGs")
    return p.parse_args()


def load_calib(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    intr = c["intrinsics"]
    R = np.asarray(c["extrinsics"]["rotation_matrix"], dtype=np.float64)
    t = np.asarray(c["extrinsics"]["translation_m"], dtype=np.float64)
    return intr, R, t


def load_table_z(path):
    if not os.path.exists(path):
        print(f"WARN: {path} missing — using {TABLE_Z} fallback")
        return TABLE_Z
    with open(path) as f:
        d = json.load(f)
    return float(d["table_z_urdf_world_m"])


def fov_x_from_intrinsics(fx, width):
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * fx))))


def load_layout_json(demos_dir, seed):
    path = Path(demos_dir) / f"demo_{seed:04d}_layout.json"
    if not path.exists():
        sys.exit(f"ERROR: missing {path}. Generate first.")
    with open(path) as f:
        return json.load(f)


def sponge_state(sp_idx, frame, layout, table_z, joints_deg):
    """Return ((x, y, z) world, rotz_deg) for sponge sp_idx at frame.

    sp_idx: 1..4
      1,2 → L1 (length X, dst_rotz=0°)
      3,4 → L2 (length Y, dst_rotz=+90°)
    Sponge state machine:
      frame < S{i}.close  → at source (xy=src, z=z_floor, rotz=src_wrist_r)
      S{i}.close ≤ frame < S{i}.open → held (follows TCP + offset, rotz=current wrist_r)
      frame ≥ S{i}.open  → placed at dst
    """
    z_floor = table_z + SPONGE_HEIGHT_EDGE / 2.0    # source/L1 mid-z (edge-stand)
    z_l2 = table_z + 1.5 * SPONGE_HEIGHT_EDGE       # L2 mid-z (on top of L1)

    src_idx = sp_idx - 1
    src_xy = layout["sources_m"][src_idx]
    src_rotz = layout["src_wrist_r_deg"][src_idx]

    if sp_idx <= 2:
        dst_xy = layout["dst_l1_sp1_m"] if sp_idx == 1 else layout["dst_l1_sp2_m"]
        dst_z = z_floor
        dst_rotz = layout["dst_l1_wrist_r_deg"]
    else:
        dst_xy = layout["dst_l2_sp3_m"] if sp_idx == 3 else layout["dst_l2_sp4_m"]
        dst_z = z_l2
        dst_rotz = layout["dst_l2_wrist_r_deg"]

    f_close = layout["anchor_frame_map"][f"S{sp_idx}.close"]
    f_open = layout["anchor_frame_map"][f"S{sp_idx}.open"]

    if frame < f_close:
        return (src_xy[0], src_xy[1], z_floor), float(src_rotz)
    if frame < f_open:
        tcp = fk_tcp(joints_deg)
        return (float(tcp[0]), float(tcp[1]), float(tcp[2]) + TCP_TO_SPONGE_CENTER_Z), float(joints_deg[4])
    return (dst_xy[0], dst_xy[1], dst_z), float(dst_rotz)


def discover_seeds(demos_dir):
    demos_dir = Path(demos_dir)
    seeds = set()
    for p in demos_dir.glob("demo_*_trajectory.csv"):
        seed = int(p.stem.split("_")[1])
        seeds.add(seed)
    return sorted(seeds)


def main():
    args = parse_args()

    for p in (args.urdf, args.calib, args.demos_dir):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing {p}")

    if args.all or args.seeds is None:
        seeds = discover_seeds(args.demos_dir)
        if not seeds:
            sys.exit(f"ERROR: no demo CSVs in {args.demos_dir}")
    else:
        seeds = args.seeds

    for s in seeds:
        traj_path = Path(args.demos_dir) / f"demo_{s:04d}_trajectory.csv"
        layout_path = Path(args.demos_dir) / f"demo_{s:04d}_layout.json"
        if not traj_path.exists():
            sys.exit(f"ERROR: missing {traj_path}")
        if not layout_path.exists():
            sys.exit(f"ERROR: missing {layout_path}")

    intr, R_cv, t_cv = load_calib(args.calib)
    table_z = load_table_z(args.table_plane)
    print(f"Calib: t={t_cv.tolist()} fx={intr['fx']:.1f}")
    print(f"Table z (URDF world) = {table_z*1000:+.2f} mm")
    print(f"Sponge: edge-stand height={SPONGE_HEIGHT_EDGE*1000:.0f}mm long={SPONGE_LEN_LONG*1000:.0f}mm "
          f"width={SPONGE_WIDTH*1000:.0f}mm")
    print(f"TCP→sponge center Z offset = {TCP_TO_SPONGE_CENTER_Z*1000:+.2f} mm")
    print(f"z_floor (L1/source mid)= {(table_z + SPONGE_HEIGHT_EDGE/2.0)*1000:+.2f}mm  "
          f"z_l2 (mid)= {(table_z + 1.5*SPONGE_HEIGHT_EDGE)*1000:+.2f}mm")
    print(f"Rendering {len(seeds)} demo(s): {seeds[:5]}{'...' if len(seeds)>5 else ''}")

    out_dir_abs = str(Path(args.output_dir).resolve())
    os.makedirs(out_dir_abs, exist_ok=True)

    if args.skip_existing:
        remaining = []
        for s in seeds:
            ep_dir = Path(out_dir_abs) / f"episode_{s:03d}"
            traj_path = Path(args.demos_dir) / f"demo_{s:04d}_trajectory.csv"
            traj = np.loadtxt(traj_path, delimiter=",", skiprows=1)
            T = len(traj)
            if ep_dir.exists() and len(list(ep_dir.glob("frame_*.png"))) >= T:
                print(f"  skip seed={s} (already {T} PNGs)")
            else:
                remaining.append(s)
        seeds = remaining
        if not seeds:
            print("Nothing to render. Done.")
            return

    # ============================================================
    # Boot Isaac Sim
    # ============================================================
    print("\nBooting Isaac Sim (headless)...")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    import omni.kit.commands
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdLux, Gf, UsdShade, Sdf

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(2500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))

    import carb
    nucleus_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    if not nucleus_root:
        nucleus_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    table_usd = f"{nucleus_root}/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    print(f"Table USD: {table_usd}")
    table_prim = stage.DefinePrim("/World/Table", "Xform")
    table_prim.GetReferences().AddReference(table_usd)
    tx = UsdGeom.Xformable(table_prim)
    tx.ClearXformOpOrder()
    tx.AddTranslateOp().Set(Gf.Vec3d(0.55, 0.0, table_z))
    tx.AddRotateZOp().Set(90.0)

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
        app.close(); sys.exit("ERROR: URDF parse failed")
    omni.kit.commands.execute(
        "URDFImportRobot", urdf_path=args.urdf, urdf_robot=robot_model,
        import_config=import_config, dest_path="",
    )
    for _ in range(10):
        app.update()

    # ------------------------------------------------------------
    # Spawn 4 sponge prims (translate + rotateZ + scale; updated per-frame)
    # ------------------------------------------------------------
    def define_sponge(prim_path, color, mat_name):
        cube = UsdGeom.Cube.Define(stage, prim_path)
        hx, hy, hz = SPONGE_SIZE[0] / 2.0, SPONGE_SIZE[1] / 2.0, SPONGE_SIZE[2] / 2.0
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        op_t = xf.AddTranslateOp()
        op_r = xf.AddRotateZOp()
        op_s = xf.AddScaleOp()
        op_t.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        op_r.Set(0.0)
        op_s.Set(Gf.Vec3f(hx, hy, hz))
        mat_path = f"/World/Looks/{mat_name}"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)
        return op_t, op_r

    sponge_ops = {}
    for sp_idx in (1, 2, 3, 4):
        color = SPONGE_COLOR_L1 if sp_idx <= 2 else SPONGE_COLOR_L2
        op_t, op_r = define_sponge(f"/World/Sponge_S{sp_idx}", color, f"MatS{sp_idx}")
        sponge_ops[sp_idx] = (op_t, op_r)

    for _ in range(10):
        app.update()

    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    world = World(stage_units_in_meters=1.0)
    robot = world.scene.add(Articulation(prim_path="/roarm_m3", name="roarm_m3"))
    world.reset()
    print(f"Robot DOFs={robot.num_dof}")
    for i, name in enumerate(JOINT_NAMES):
        assert robot.dof_names[i] == name, f"DOF mismatch at {i}: want {name} got {robot.dof_names[i]}"

    # Camera
    cam_prim = UsdGeom.Camera.Define(stage, "/World/SimCam")
    hfov = fov_x_from_intrinsics(intr["fx"], args.width)
    aperture_h = 20.955
    focal = (aperture_h / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    cam_prim.CreateFocalLengthAttr(float(focal))
    cam_prim.CreateHorizontalApertureAttr(float(aperture_h))
    cam_prim.CreateVerticalApertureAttr(float(aperture_h * args.height / args.width))
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    flip = np.diag([1.0, -1.0, -1.0])
    R_usd = R_cv @ flip
    M = np.eye(4); M[:3, :3] = R_usd; M[:3, 3] = t_cv
    M_gf = Gf.Matrix4d(*M.T.flatten().tolist())
    cam_xf = UsdGeom.Xformable(cam_prim.GetPrim())
    cam_xf.ClearXformOpOrder()
    cam_xf.AddTransformOp().Set(M_gf)
    print(f"Camera HFOV={hfov:.2f}° focal_mm={focal:.2f}")

    rp = rep.create.render_product("/World/SimCam", (args.width, args.height))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([rp])
    for _ in range(args.settle_steps):
        app.update()

    from PIL import Image

    overall_t0 = time.time()
    summaries = []

    for ep_idx, seed in enumerate(seeds):
        ep_t0 = time.time()
        traj_path = Path(args.demos_dir) / f"demo_{seed:04d}_trajectory.csv"
        traj = np.loadtxt(traj_path, delimiter=",", skiprows=1)
        T = len(traj)
        layout = load_layout_json(args.demos_dir, seed)

        ep_dir = Path(out_dir_abs) / f"episode_{seed:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== ep {seed} ({ep_idx+1}/{len(seeds)})  T={T} ===")
        for i, (s, o) in enumerate(zip(layout["sources_m"], layout["orients"]), 1):
            print(f"  src S{i} ({o}): ({s[0]*1000:+.0f}, {s[1]*1000:+.0f})")
        print(f"  hold intervals: " + " ".join(
            f"S{i}=[{layout['anchor_frame_map'][f'S{i}.close']},{layout['anchor_frame_map'][f'S{i}.open']})"
            for i in (1, 2, 3, 4)))

        joints_deg = traj[0]
        for sp_idx in (1, 2, 3, 4):
            (pos, rotz) = sponge_state(sp_idx, 0, layout, table_z, joints_deg)
            sponge_ops[sp_idx][0].Set(Gf.Vec3d(*pos))
            sponge_ops[sp_idx][1].Set(float(rotz))
        for _ in range(5):
            app.update()

        for f in range(T):
            joints_deg = traj[f]
            joints_rad = np.radians(joints_deg)
            robot.set_joint_positions(joints_rad)

            for sp_idx in (1, 2, 3, 4):
                (pos, rotz) = sponge_state(sp_idx, f, layout, table_z, joints_deg)
                sponge_ops[sp_idx][0].Set(Gf.Vec3d(*pos))
                sponge_ops[sp_idx][1].Set(float(rotz))

            for _ in range(3):
                world.step(render=True)
                app.update()
            rep.orchestrator.step()
            app.update()

            rgb = rgb_annot.get_data()
            if rgb is None or rgb.ndim != 3:
                app.close()
                sys.exit(f"ERROR: RGB annotator returned {type(rgb)} at ep {seed} frame {f}")

            out_path = ep_dir / f"frame_{f:04d}.png"
            Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(out_path)

            if f == 0 or f == T - 1 or f % 25 == 0:
                tcp = fk_tcp(joints_deg)
                held = [sp for sp in (1, 2, 3, 4)
                        if layout["anchor_frame_map"][f"S{sp}.close"] <= f
                        < layout["anchor_frame_map"][f"S{sp}.open"]]
                print(f"  ep{seed} f{f:3d}: TCP=({tcp[0]*1000:+.0f},{tcp[1]*1000:+.0f},{tcp[2]*1000:+.0f})mm "
                      f"wrist_r={joints_deg[4]:+.0f}° held={held}")

        elapsed = time.time() - ep_t0
        summaries.append({"seed": seed, "n_frames": T, "elapsed_s": round(elapsed, 1)})
        print(f"  ep{seed} done  {elapsed:.1f}s  ({elapsed/T*1000:.0f} ms/frame)")

    overall_elapsed = time.time() - overall_t0
    total_frames = sum(s["n_frames"] for s in summaries)
    summary_path = Path(out_dir_abs) / "render_summary.json"
    with open(summary_path, "w") as fh:
        json.dump({
            "total_eps": len(summaries),
            "total_frames": total_frames,
            "total_elapsed_s": round(overall_elapsed, 1),
            "ms_per_frame": round(overall_elapsed / max(1, total_frames) * 1000, 1),
            "per_episode": summaries,
        }, fh, indent=2)

    print(f"\n=== OVERALL: {len(summaries)} eps, {total_frames} frames in "
          f"{overall_elapsed:.1f}s ({overall_elapsed/max(1,total_frames)*1000:.0f} ms/frame avg) ===")
    print(f"  → {summary_path}")
    app.close()
    print("Done.")


if __name__ == "__main__":
    main()
