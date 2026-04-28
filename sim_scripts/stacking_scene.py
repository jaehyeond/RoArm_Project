"""Stacking scene: spawn 2 sponges stacked at position A + markers at B and Temp.

Goal (first pass): visual validation of layout decisions from 4/24 stacking pivot:
  A    : (+280, 0) mm world  — initial stack of 2 sponges (start state)
  B    : (+280, +130) mm    — empty target (where 2-stack must end)
  Temp : (+280, -110) mm    — intermediate buffer area (single-sponge holds top of A)

Sponge: 22 × 47 × 125 mm, upright (125 mm along +Z), pink.

Run:
    conda run -n isaaclab python sim_scripts/stacking_scene.py \
        --output sim_renders_v2/stacking_initial.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
URDF_PATH_DEFAULT = "/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf"
CALIB_DEFAULT = str(REPO / "sim_scripts" / "kinect_calib.yaml")
TABLE_PLANE_DEFAULT = str(REPO / "sim_scripts" / "table_plane.json")

JOINT_NAMES = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
]

# ---------------------------------------------------------------
# Sponge spec (4/27 update: 20mm -> 22mm).
# ---------------------------------------------------------------
SPONGE_SIZE_M = (0.022, 0.047, 0.125)  # x, y, z (upright = z vertical)
SPONGE_COLOR = (0.95, 0.45, 0.60)

# ---------------------------------------------------------------
# Layout (URDF world frame, meters). From 4/24 stacking pivot.
# ---------------------------------------------------------------
LAYOUT = {
    "A":    (+0.280,  0.000),
    "B":    (+0.280, +0.130),
    "Temp": (+0.280, -0.110),
}

# Marker visual (semi-transparent ghost cube) for empty positions.
MARKER_COLOR = (0.30, 0.65, 0.85)  # cyan-blue

# Home joint position — RoArm M3 default ready pose.
HOME_DEG = [0.0, 0.0, 90.0, 0.0, 0.0, 30.0]  # cmd 30 = approach-ready gripper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default=str(REPO / "sim_renders_v2" / "stacking_initial.png"))
    p.add_argument("--urdf", type=str, default=URDF_PATH_DEFAULT)
    p.add_argument("--calib", type=str, default=CALIB_DEFAULT)
    p.add_argument("--table-plane", type=str, default=TABLE_PLANE_DEFAULT)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-markers", action="store_true",
                   help="Skip B/Temp ghost markers (cleaner image for SigLIP-style eval)")
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
    import json
    with open(path) as f:
        d = json.load(f)
    return float(d["table_z_urdf_world_m"])


def fov_x_from_intrinsics(fx, width):
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * fx))))


def main():
    args = parse_args()

    for p in (args.urdf, args.calib):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing file {p}")

    intr, R_cv, t_cv = load_calib(args.calib)
    table_z = load_table_z(args.table_plane)
    print(f"Table surface Z (URDF world) = {table_z*1000:+.2f} mm")
    print(f"Calib: t={t_cv.tolist()}  fx={intr['fx']:.1f}")

    # Sponge stacking heights (URDF world Z).
    h = SPONGE_SIZE_M[2]  # 0.125 m
    z_bot_center = table_z + h / 2.0          # bottom sponge center
    z_top_center = table_z + h * 1.5          # top sponge center (stacked)
    z_floor_center = table_z + h / 2.0        # single sponge sitting on table
    print(f"  Stack heights (URDF Z, mm):  bot_center={z_bot_center*1000:.1f}  "
          f"top_center={z_top_center*1000:.1f}")

    # Compute predicted clearances (Layout safety check).
    print("\n--- Layout clearance analysis ---")
    A = np.array(LAYOUT["A"]); B = np.array(LAYOUT["B"]); T = np.array(LAYOUT["Temp"])
    print(f"  A↔B  distance = {np.linalg.norm(B-A)*1000:.1f} mm")
    print(f"  A↔Temp distance = {np.linalg.norm(T-A)*1000:.1f} mm")
    print(f"  B↔Temp distance = {np.linalg.norm(T-B)*1000:.1f} mm")
    sponge_y = SPONGE_SIZE_M[1] * 1000.0  # 47mm
    print(f"  Sponge Y body width = {sponge_y:.1f} mm")
    arm_assembly_w = 80.0  # housing 35.5 + jaw 45 (cmd 60 approach), see tech_gripper_grasp_anchors.md
    print(f"  Arm assembly width @ cmd60 jaw ≈ {arm_assembly_w:.1f} mm")
    edge_gap_AT = np.linalg.norm(T-A)*1000 - sponge_y
    edge_gap_AB = np.linalg.norm(B-A)*1000 - sponge_y
    print(f"  A↔Temp edge-edge gap = {edge_gap_AT:.1f} mm (transport corridor)")
    print(f"  A↔B   edge-edge gap = {edge_gap_AB:.1f} mm")
    print(f"  → Arm assembly 80mm needs ≥80mm corridor.")
    print(f"  → A-Temp gap {edge_gap_AT:.1f}mm vs 80mm: "
          f"{'PASS' if edge_gap_AT >= 80 else 'TIGHT (≤0mm side margin)'}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ------------------------------------------------------------
    # Boot Isaac Sim (headless render).
    # ------------------------------------------------------------
    print("\nBooting Isaac Sim (headless)...")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    import omni.kit.commands
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdLux, Gf, UsdShade, Sdf

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")

    # Light.
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(2500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))

    # ------------------------------------------------------------
    # Table USD.
    # ------------------------------------------------------------
    import carb
    nucleus_root = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    if not nucleus_root:
        nucleus_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    table_usd = f"{nucleus_root}/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    print(f"Table USD: {table_usd}")
    table_prim = stage.DefinePrim("/World/Table", "Xform")
    table_prim.GetReferences().AddReference(table_usd)
    table_xform = UsdGeom.Xformable(table_prim)
    table_xform.ClearXformOpOrder()
    op_t = table_xform.AddTranslateOp(); op_t.Set(Gf.Vec3d(0.55, 0.0, table_z))
    op_r = table_xform.AddRotateZOp(); op_r.Set(90.0)

    # ------------------------------------------------------------
    # Robot URDF.
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
        app.close(); sys.exit("ERROR: URDF parse failed")
    omni.kit.commands.execute(
        "URDFImportRobot", urdf_path=args.urdf, urdf_robot=robot_model,
        import_config=import_config, dest_path="",
    )
    for _ in range(10):
        app.update()

    # ------------------------------------------------------------
    # Spawn 2 sponges stacked at A.
    # ------------------------------------------------------------
    def spawn_sponge(prim_path, pos_xy, z_center, color, mat_name="SpongeMat"):
        """Spawn a sponge cuboid at world (x, y, z_center). Size: SPONGE_SIZE_M."""
        cube = UsdGeom.Cube.Define(stage, prim_path)
        hx, hy, hz = [s / 2.0 for s in SPONGE_SIZE_M]
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(pos_xy[0], pos_xy[1], z_center))
        xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # upright
        xf.AddScaleOp().Set(Gf.Vec3f(hx, hy, hz))
        # Material.
        mat_path = f"/World/Looks/{mat_name}"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)

    print("\nSpawning sponges:")
    A_xy = LAYOUT["A"]
    spawn_sponge("/World/SpongeA_Bot", A_xy, z_bot_center, SPONGE_COLOR, "MatA_Bot")
    print(f"  A_Bot at ({A_xy[0]:.3f}, {A_xy[1]:.3f}, {z_bot_center:.3f})")
    spawn_sponge("/World/SpongeA_Top", A_xy, z_top_center, SPONGE_COLOR, "MatA_Top")
    print(f"  A_Top at ({A_xy[0]:.3f}, {A_xy[1]:.3f}, {z_top_center:.3f})")

    # Markers for B, Temp (low-opacity ghost cubes at floor height).
    if not args.no_markers:
        spawn_sponge("/World/MarkerB", LAYOUT["B"], z_floor_center, MARKER_COLOR, "MatB")
        print(f"  Marker B at ({LAYOUT['B'][0]:.3f}, {LAYOUT['B'][1]:.3f}, "
              f"{z_floor_center:.3f}) [empty target]")
        spawn_sponge("/World/MarkerTemp", LAYOUT["Temp"], z_floor_center, MARKER_COLOR, "MatTemp")
        print(f"  Marker Temp at ({LAYOUT['Temp'][0]:.3f}, {LAYOUT['Temp'][1]:.3f}, "
              f"{z_floor_center:.3f}) [intermediate buffer]")

    for _ in range(5):
        app.update()

    # ------------------------------------------------------------
    # Articulation: set arm to HOME pose (cmd 30 gripper).
    # ------------------------------------------------------------
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation

    world = World(stage_units_in_meters=1.0)
    robot = world.scene.add(Articulation(prim_path="/roarm_m3", name="roarm_m3"))
    world.reset()
    print(f"Robot DOFs={robot.num_dof}  names={robot.dof_names}")
    for i, name in enumerate(JOINT_NAMES):
        assert robot.dof_names[i] == name, f"DOF mismatch at {i}: want {name} got {robot.dof_names[i]}"

    home_rad = np.deg2rad(np.array(HOME_DEG, dtype=np.float32))
    robot.set_joint_positions(home_rad)
    for _ in range(args.width and 30):
        world.step(render=False)

    # ------------------------------------------------------------
    # Camera at calibrated Kinect pose.
    # ------------------------------------------------------------
    cam_prim = UsdGeom.Camera.Define(stage, "/World/SimCam")
    hfov = fov_x_from_intrinsics(intr["fx"], args.width)
    aperture_h = 20.955
    focal = (aperture_h / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    cam_prim.CreateFocalLengthAttr(float(focal))
    cam_prim.CreateHorizontalApertureAttr(float(aperture_h))
    cam_prim.CreateVerticalApertureAttr(float(aperture_h * args.height / args.width))
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    # Pose: OpenCV → USD basis flip (replay_v6_sim convention).
    flip = np.diag([1.0, -1.0, -1.0])
    R_usd = R_cv @ flip
    M = np.eye(4)
    M[:3, :3] = R_usd
    M[:3, 3] = t_cv
    M_gf = Gf.Matrix4d(*M.T.flatten().tolist())
    cam_xf = UsdGeom.Xformable(cam_prim.GetPrim())
    cam_xf.ClearXformOpOrder()
    op_x = cam_xf.AddTransformOp()
    op_x.Set(M_gf)

    # ------------------------------------------------------------
    # Render via replicator.
    # ------------------------------------------------------------
    print(f"\nRendering {args.width}x{args.height} → {args.output}")
    rp = rep.create.render_product("/World/SimCam", (args.width, args.height))
    writer = rep.WriterRegistry.get("BasicWriter")
    out_dir = os.path.dirname(args.output) or "."
    writer.initialize(output_dir=out_dir, rgb=True)
    writer.attach([rp])

    rep.orchestrator.step()
    for _ in range(5):
        app.update()
    rep.orchestrator.wait_until_complete()

    # Replicator writes auto-named files. Find newest png and rename.
    import glob, time, shutil
    time.sleep(0.5)
    pngs = sorted(glob.glob(os.path.join(out_dir, "rgb_*.png")), key=os.path.getmtime)
    if pngs:
        newest = pngs[-1]
        if newest != args.output:
            shutil.move(newest, args.output)
        print(f"  Saved: {args.output}")
    else:
        print(f"  WARN: no rgb_*.png produced in {out_dir}")

    app.close()
    print("Done.")


if __name__ == "__main__":
    main()
