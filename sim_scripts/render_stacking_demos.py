"""Render stacking sim demos (Phase ST-A → ST-B bridge).

Renders all 50 procedural stacking demos from sim_demos_v1/ at the calibrated
Kinect viewpoint. Two sponges are tracked per frame:

  - SpongeA_Bot : starts at A floor    -> grabbed in S2 -> placed at B floor
  - SpongeA_Top : starts on A_Bot stack -> grabbed in S1 -> placed at Temp floor
                                       -> grabbed in S3 -> placed on B_Bot stack

Held intervals are derived from generate_stacking_demos.py FRAMES_PER_SEG (fixed
95-frame structure, 26 anchors). When held, sponge follows TCP with offset
(0, 0, -0.0325 m) so its center sits at lateral-grasp mid-side height.

Run (4090 only — HARD RULE #17):
    conda run -n isaaclab python sim_scripts/render_stacking_demos.py --seeds 0
    conda run -n isaaclab python sim_scripts/render_stacking_demos.py --all
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

URDF_PATH_DEFAULT = "/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf"
CALIB_DEFAULT = str(REPO / "sim_scripts" / "kinect_calib.yaml")
TABLE_PLANE_DEFAULT = str(REPO / "sim_scripts" / "table_plane.json")
DEMOS_DEFAULT = str(REPO / "sim_demos_v1")
OUT_DEFAULT = str(REPO / "sim_renders_v3")

JOINT_NAMES = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
]

# Constants — must match generate_stacking_demos.py + stacking_scene.py.
LAYOUT_NOMINAL = {
    "A":    (+0.280,  0.000),
    "B":    (+0.280, +0.130),
    "Temp": (+0.280, -0.110),
}
DXY = 0.010  # ±10mm xy randomization, applied to A then B then Temp (rng order)

SPONGE_SIZE_M = (0.022, 0.047, 0.125)
SPONGE_COLOR_BOT = (0.95, 0.45, 0.60)
SPONGE_COLOR_TOP = (0.85, 0.30, 0.45)
STACK_GAP = 0.002
TCP_TO_SPONGE_CENTER_Z = -0.0325  # sponge center sits 32.5mm BELOW TCP grasp height

# Anchor frame boundaries (95-frame demo). Derived from FRAMES_PER_SEG sums.
# Verified: HOME_start(0) ... HOME_end(94). 26 anchors total.
ANCHOR_FRAMES = [
    0, 5, 9, 12, 16, 21, 25, 28, 31,    # HOME, S1.{above_src,at_src,close,lift,transit,at_dst,open,lift_off}
    34, 38, 41, 45, 50, 54, 57, 60,     # S2.{above_src,at_src,close,lift,transit,at_dst,open,lift_off}
    63, 67, 70, 74, 79, 83, 86, 89,     # S3.{above_src,at_src,close,lift,transit,at_dst,open,lift_off}
    94,                                  # HOME_end
]
# Held intervals: [close_anchor_frame, open_anchor_frame). At open frame, sponge is dropped.
# S1 close=anchor 3 (frame 12), open=anchor 7 (frame 28) -> top held [12, 28)
# S2 close=anchor 11 (frame 41), open=anchor 15 (frame 57) -> bot held [41, 57)
# S3 close=anchor 19 (frame 70), open=anchor 23 (frame 86) -> top held [70, 86)
HOLD_INTERVALS = {
    "top": [(12, 28), (70, 86)],
    "bot": [(41, 57)],
}

SPONGE_TOP_PRIM = "/World/SpongeA_Top"
SPONGE_BOT_PRIM = "/World/SpongeA_Bot"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Specific seeds to render (default: all 50)")
    p.add_argument("--all", action="store_true", help="Render all 50 demos")
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
        print(f"WARN: {path} missing — using z=0 fallback")
        return 0.0
    with open(path) as f:
        d = json.load(f)
    return float(d["table_z_urdf_world_m"])


def fov_x_from_intrinsics(fx, width):
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * fx))))


def layout_for_seed(seed):
    """Reproduce layout_used from generate_stacking_demos.build_anchors()."""
    rng = np.random.default_rng(seed)
    A    = (LAYOUT_NOMINAL["A"][0]    + rng.uniform(-DXY, DXY),
            LAYOUT_NOMINAL["A"][1]    + rng.uniform(-DXY, DXY))
    B    = (LAYOUT_NOMINAL["B"][0]    + rng.uniform(-DXY, DXY),
            LAYOUT_NOMINAL["B"][1]    + rng.uniform(-DXY, DXY))
    Temp = (LAYOUT_NOMINAL["Temp"][0] + rng.uniform(-DXY, DXY),
            LAYOUT_NOMINAL["Temp"][1] + rng.uniform(-DXY, DXY))
    return {"A": A, "B": B, "Temp": Temp}


def sponge_state_for_frame(f, layout, table_z):
    """Returns dict with (top_pos, top_held, bot_pos, bot_held) for frame f.

    Positions are sponge CENTERS in URDF world meters. When held, position is
    'follow-TCP' tag and caller substitutes TCP+offset.
    """
    h = SPONGE_SIZE_M[2]
    z_bot_center = table_z + h / 2.0
    z_top_center = z_bot_center + h + STACK_GAP
    z_floor_center = table_z + h / 2.0  # single sponge on floor

    A, B, Temp = layout["A"], layout["B"], layout["Temp"]

    # Bot sponge state machine
    if f < 41:  # before S2.close
        bot_pos = (A[0], A[1], z_bot_center)
        bot_held = False
    elif 41 <= f < 57:  # S2 held
        bot_pos = None
        bot_held = True
    else:  # f >= 57, placed at B floor
        bot_pos = (B[0], B[1], z_floor_center)
        bot_held = False

    # Top sponge state machine
    if f < 12:  # before S1.close — on top of bot at A
        top_pos = (A[0], A[1], z_top_center)
        top_held = False
    elif 12 <= f < 28:  # S1 held
        top_pos = None
        top_held = True
    elif 28 <= f < 70:  # placed at Temp floor (S1 done) waiting for S3
        top_pos = (Temp[0], Temp[1], z_floor_center)
        top_held = False
    elif 70 <= f < 86:  # S3 held
        top_pos = None
        top_held = True
    else:  # f >= 86, placed on top of B's bot
        top_pos = (B[0], B[1], z_top_center)
        top_held = False

    return top_pos, top_held, bot_pos, bot_held


def held_sponge_world_pos(joints_deg):
    """TCP+offset so held sponge center sits at TCP_z - 32.5mm (lateral grasp mid-side)."""
    tcp = fk_tcp(joints_deg)
    return (float(tcp[0]), float(tcp[1]), float(tcp[2] + TCP_TO_SPONGE_CENTER_Z))


def main():
    args = parse_args()

    for p in (args.urdf, args.calib, args.demos_dir):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing {p}")

    # Resolve seed list
    if args.all or args.seeds is None:
        seeds = list(range(50))
    else:
        seeds = args.seeds

    # Validate trajectory CSVs exist
    for s in seeds:
        traj_path = Path(args.demos_dir) / f"demo_{s:04d}_trajectory.csv"
        if not traj_path.exists():
            sys.exit(f"ERROR: missing {traj_path}")

    intr, R_cv, t_cv = load_calib(args.calib)
    table_z = load_table_z(args.table_plane)
    print(f"Calib: t={t_cv.tolist()} fx={intr['fx']:.1f}")
    print(f"Table z (URDF world) = {table_z*1000:+.2f} mm")
    print(f"Rendering {len(seeds)} demo(s): {seeds[:5]}{'...' if len(seeds)>5 else ''}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Skip-existing handling
    if args.skip_existing:
        remaining = []
        for s in seeds:
            ep_dir = Path(args.output_dir) / f"episode_{s:03d}"
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

    # Table USD (SeattleLabTable)
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

    # URDF
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
        "URDFImportRobot", urdf_path=args.urdf, urdf_robot=robot_model,
        import_config=import_config, dest_path="",
    )
    for _ in range(10):
        app.update()

    # Sponges (created once, transforms updated per-frame)
    def define_sponge(prim_path, color, mat_name):
        cube = UsdGeom.Cube.Define(stage, prim_path)
        hx, hy, hz = [s / 2.0 for s in SPONGE_SIZE_M]
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        op_t = xf.AddTranslateOp()
        op_t.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        op_r = xf.AddOrientOp()
        op_r.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        op_s = xf.AddScaleOp()
        op_s.Set(Gf.Vec3f(hx, hy, hz))
        # Material
        mat_path = f"/World/Looks/{mat_name}"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)
        return op_t

    op_top_t = define_sponge(SPONGE_TOP_PRIM, SPONGE_COLOR_TOP, "MatTop")
    op_bot_t = define_sponge(SPONGE_BOT_PRIM, SPONGE_COLOR_BOT, "MatBot")

    for _ in range(10):
        app.update()

    # Articulation
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
    M = np.eye(4)
    M[:3, :3] = R_usd
    M[:3, 3] = t_cv
    M_gf = Gf.Matrix4d(*M.T.flatten().tolist())
    cam_xf = UsdGeom.Xformable(cam_prim.GetPrim())
    cam_xf.ClearXformOpOrder()
    cam_xf.AddTransformOp().Set(M_gf)
    print(f"Camera HFOV={hfov:.2f}° focal_mm={focal:.2f}")

    # Render product
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
        traj = np.loadtxt(traj_path, delimiter=",", skiprows=1)  # (T, 6) deg
        T = len(traj)
        layout = layout_for_seed(seed)

        ep_dir = Path(args.output_dir) / f"episode_{seed:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== ep {seed} ({ep_idx+1}/{len(seeds)})  T={T}  "
              f"A=({layout['A'][0]*1000:+.1f},{layout['A'][1]*1000:+.1f}) "
              f"B=({layout['B'][0]*1000:+.1f},{layout['B'][1]*1000:+.1f}) "
              f"Temp=({layout['Temp'][0]*1000:+.1f},{layout['Temp'][1]*1000:+.1f}) ===")

        # Initial sponge poses for frame 0
        top_pos, _, bot_pos, _ = sponge_state_for_frame(0, layout, table_z)
        op_top_t.Set(Gf.Vec3d(*top_pos))
        op_bot_t.Set(Gf.Vec3d(*bot_pos))
        for _ in range(5):
            app.update()

        for f in range(T):
            joints_deg = traj[f]
            joints_rad = np.radians(joints_deg)
            robot.set_joint_positions(joints_rad)

            top_pos, top_held, bot_pos, bot_held = sponge_state_for_frame(f, layout, table_z)
            if top_held:
                top_pos = held_sponge_world_pos(joints_deg)
            if bot_held:
                bot_pos = held_sponge_world_pos(joints_deg)
            op_top_t.Set(Gf.Vec3d(*top_pos))
            op_bot_t.Set(Gf.Vec3d(*bot_pos))

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

            if f == 0 or f == T - 1 or f % 20 == 0:
                tcp = fk_tcp(joints_deg)
                print(f"  ep{seed} f{f:3d}: TCP=({tcp[0]*1000:+.0f},{tcp[1]*1000:+.0f},{tcp[2]*1000:+.0f})mm "
                      f"top_held={top_held} bot_held={bot_held}")

        elapsed = time.time() - ep_t0
        summaries.append({"seed": seed, "n_frames": T, "elapsed_s": round(elapsed, 1)})
        print(f"  ep{seed} done  {elapsed:.1f}s  ({elapsed/T*1000:.0f} ms/frame)")

    overall_elapsed = time.time() - overall_t0
    total_frames = sum(s["n_frames"] for s in summaries)
    summary_path = Path(args.output_dir) / "render_summary.json"
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
