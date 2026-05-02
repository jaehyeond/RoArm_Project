"""Stacking scene v2 (Sub-A4) — spawn 4 lying flat sponges at SOURCES + # destination markers.

V3 corrections (4/30 late-evening):
  - SPONGE_THICK = 0.022 (was 0.020)
  - HOME closed (G_PRECLOSE = +5.0)
  - Sources: random per --seed (4 region partition + min pairwise dist 150mm)
  - Source orientation per source from generate_stacking_demos_v2.sample_layout
  - Destination orientation: L1 = length-X (wrist_r=0°), L2 = length-Y (wrist_r=+90°)

Run (4090 only — HARD RULE #17):
    conda run -n isaaclab python sim_scripts/stacking_scene_v2.py \\
        --seed 0 --output sim_renders_v4/stacking_initial_seed0.png --markers
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_scripts"))
from generate_stacking_demos_v2 import (  # noqa: E402
    sample_layout, ORIENT_TO_WRIST_R,
    HASH1_CENTER, DST_L1_SP1, DST_L1_SP2, DST_L2_SP3, DST_L2_SP4,
    SPONGE_THICK, SPONGE_LEN_LONG, SPONGE_LEN_SHORT, TABLE_Z,
    G_PRECLOSE,
)

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

SIZE_LENGTH_X = (SPONGE_LEN_LONG, SPONGE_LEN_SHORT, SPONGE_THICK)   # 125,47,22
SIZE_LENGTH_Y = (SPONGE_LEN_SHORT, SPONGE_LEN_LONG, SPONGE_THICK)   # 47,125,22

SPONGE_COLOR_L1 = (0.95, 0.45, 0.60)
SPONGE_COLOR_L2 = (0.85, 0.30, 0.45)
SPONGE_COLOR_SRC = (0.95, 0.55, 0.65)
MARKER_COLOR = (0.30, 0.65, 0.85)

HOME_DEG = [0.0, 0.0, 90.0, 0.0, 0.0, G_PRECLOSE]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demos-dir", type=str, default=str(REPO / "sim_demos_v2"),
                   help="If layout JSON exists for seed, load it (consistent with generated demos)")
    p.add_argument("--output", type=str, default=str(REPO / "sim_renders_v4" / "stacking_initial.png"))
    p.add_argument("--urdf", type=str, default=URDF_PATH_DEFAULT)
    p.add_argument("--calib", type=str, default=CALIB_DEFAULT)
    p.add_argument("--table-plane", type=str, default=TABLE_PLANE_DEFAULT)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--markers", action="store_true",
                   help="Show ghost markers at L1/L2 # destinations")
    return p.parse_args()


def load_or_sample_layout(seed, demos_dir):
    layout_json_path = Path(demos_dir) / f"demo_{seed:04d}_layout.json"
    if layout_json_path.exists():
        import json as _json
        with open(layout_json_path) as f:
            d = _json.load(f)
        sources = [(s[0], s[1]) for s in d["sources_m"]]
        orients = d["orients"]
        return {"sources": sources, "orients": orients, "attempt": -1, "source": "json"}
    layout = sample_layout(seed)
    layout["source"] = "sampled"
    return layout


def load_calib(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    intr = c["intrinsics"]
    R = np.asarray(c["extrinsics"]["rotation_matrix"], dtype=np.float64)
    t = np.asarray(c["extrinsics"]["translation_m"], dtype=np.float64)
    return intr, R, t


def load_table_z(path):
    if not os.path.exists(path):
        print(f"WARN: {path} missing — using TABLE_Z fallback")
        return TABLE_Z
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
    layout = load_or_sample_layout(args.seed, args.demos_dir)

    print(f"Seed={args.seed}  source={layout.get('source','?')}  attempt={layout['attempt']}")
    print(f"Table z (URDF world) = {table_z*1000:+.2f} mm")
    print(f"Calib: t={t_cv.tolist()} fx={intr['fx']:.1f}")
    print(f"Sponge: thick={SPONGE_THICK*1000:.0f}mm long={SPONGE_LEN_LONG*1000:.0f}mm "
          f"short={SPONGE_LEN_SHORT*1000:.0f}mm")

    # Sponge centers.
    z_floor_center = table_z + SPONGE_THICK / 2.0
    z_l2_center = table_z + 1.5 * SPONGE_THICK   # top of L1 + half L2
    print(f"Sponge center z (mm): floor={z_floor_center*1000:+.2f}  L2={z_l2_center*1000:+.2f}")

    print("\n--- Layout (mm world) ---")
    for i, (xy, ori) in enumerate(zip(layout["sources"], layout["orients"]), 1):
        wr = ORIENT_TO_WRIST_R[ori]
        print(f"  Source {i} (length {ori}, wrist_r={wr:+.0f}°): "
              f"({xy[0]*1000:+7.1f}, {xy[1]*1000:+7.1f})")
    for tag, xy in (("L1 sp1 (length X)", DST_L1_SP1),
                    ("L1 sp2 (length X)", DST_L1_SP2),
                    ("L2 sp3 (length Y)", DST_L2_SP3),
                    ("L2 sp4 (length Y)", DST_L2_SP4)):
        print(f"  {tag}: ({xy[0]*1000:+7.1f}, {xy[1]*1000:+7.1f})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ------------------------------------------------------------
    # Boot Isaac Sim (headless).
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
    # Spawn 4 sponges at sources (lying flat, per-source orientation).
    # ------------------------------------------------------------
    def spawn_sponge(prim_path, pos_xy, z_center, size_xyz, color, mat_name):
        cube = UsdGeom.Cube.Define(stage, prim_path)
        hx, hy, hz = [s / 2.0 for s in size_xyz]
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(pos_xy[0], pos_xy[1], z_center))
        xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        xf.AddScaleOp().Set(Gf.Vec3f(hx, hy, hz))
        mat_path = f"/World/Looks/{mat_name}"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)

    print("\nSpawning 4 source sponges (lying flat):")
    for i, (xy, ori) in enumerate(zip(layout["sources"], layout["orients"]), 1):
        size = SIZE_LENGTH_X if ori == "X" else SIZE_LENGTH_Y
        spawn_sponge(f"/World/Sponge_S{i}", xy, z_floor_center, size,
                     SPONGE_COLOR_SRC, f"MatS{i}")
        print(f"  Sponge S{i}: ({xy[0]:.3f}, {xy[1]:.3f}, {z_floor_center:.4f}) "
              f"size={tuple(round(s,3) for s in size)} ori={ori}")

    if args.markers:
        print("\nSpawning ghost markers at # destinations:")
        for i, xy in enumerate((DST_L1_SP1, DST_L1_SP2), 1):
            spawn_sponge(f"/World/Marker_L1_sp{i}", xy, z_floor_center,
                         SIZE_LENGTH_X, MARKER_COLOR, f"MatMarkerL1S{i}")
            print(f"  Marker L1 sp{i} (length X): ({xy[0]:.3f}, {xy[1]:.3f}, {z_floor_center:.4f})")
        for i, xy in enumerate((DST_L2_SP3, DST_L2_SP4), 3):
            spawn_sponge(f"/World/Marker_L2_sp{i}", xy, z_l2_center,
                         SIZE_LENGTH_Y, MARKER_COLOR, f"MatMarkerL2S{i}")
            print(f"  Marker L2 sp{i} (length Y): ({xy[0]:.3f}, {xy[1]:.3f}, {z_l2_center:.4f})")

    for _ in range(30):
        app.update()

    # ------------------------------------------------------------
    # Articulation: HOME pose.
    # ------------------------------------------------------------
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation

    world = World(stage_units_in_meters=1.0)
    robot = world.scene.add(Articulation(prim_path="/roarm_m3", name="roarm_m3"))
    world.reset()
    print(f"Robot DOFs={robot.num_dof} names={robot.dof_names}")
    for i, name in enumerate(JOINT_NAMES):
        assert robot.dof_names[i] == name, f"DOF mismatch at {i}: want {name} got {robot.dof_names[i]}"

    home_rad = np.deg2rad(np.array(HOME_DEG, dtype=np.float32))
    robot.set_joint_positions(home_rad)
    for _ in range(30):
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
    flip = np.diag([1.0, -1.0, -1.0])
    R_usd = R_cv @ flip
    M = np.eye(4); M[:3, :3] = R_usd; M[:3, 3] = t_cv
    M_gf = Gf.Matrix4d(*M.T.flatten().tolist())
    cam_xf = UsdGeom.Xformable(cam_prim.GetPrim())
    cam_xf.ClearXformOpOrder()
    cam_xf.AddTransformOp().Set(M_gf)

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
