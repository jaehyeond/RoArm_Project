"""Replay P6v12 policy trajectory to PNG sequence (v3 패턴 차용).

Input:  claudedocs/figures/p6v12_rollout/p6v12_trajectory.csv
        (from scripts/extract_p6v12_trajectory.py — 200 rows × 14 cols)
Output: claudedocs/figures/p6v12_rollout/replay/frame_NNNN.png  (200 frames)
        claudedocs/figures/p6v12_rollout/p6v12_rollout.mp4         (30fps)

색상 (교수님 지시):
  - 로봇      = 검정 (visual mesh material override)
  - 책상      = 흰   (procedural Cube prim + white material)
  - 배경/dome = 회색 (DomeLight color (0.5, 0.5, 0.5))

URDF: /home/cgxr/Documents/Robotics/isaac_roarm_m3/.../roarm_m3.urdf  (v3 동일)

Run:
    conda run -n isaaclab --no-capture-output python -u \
        -m scripts.render_p6v12_trajectory_replay
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
URDF_DEFAULT = "/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf"
CSV_DEFAULT = str(REPO / "claudedocs/figures/p6v12_rollout/p6v12_trajectory.csv")
OUT_DEFAULT = str(REPO / "claudedocs/figures/p6v12_rollout")
CALIB_DEFAULT = str(REPO / "sim_scripts/kinect_calib.yaml")

# Sponge dimensions (HARD RULE #19: edge-stand 47mm)
SPONGE_LEN = 0.125    # X (length)
SPONGE_WIDTH = 0.022  # Y (width)
SPONGE_HEIGHT = 0.047 # Z (height when edge-stand)
SPONGE_COLOR = (0.95, 0.45, 0.60)  # pink (visible against neutral palette)

# Table: large WHITE plate that covers robot base (x=0) + workspace (x=+0.28).
# sim_renders_v2 style — robot base must sit ON the table.
TABLE_CENTER = (0.25, 0.0)   # world XY, centered between robot base & workspace
TABLE_SIZE = (0.90, 0.70)    # X × Y meters — extends from x=-0.20 to +0.70 (covers base)
TABLE_THICKNESS = 0.02
TABLE_Z_TOP = -0.012117       # from table_plane.json
TABLE_COLOR = (0.92, 0.92, 0.92)  # WHITE table (user requested)

# Dome (background): medium-dark gray for distinct contrast vs white table.
# Lower intensity also keeps the dark robot truly black (not lifted by dome flood).
DOME_COLOR = (0.45, 0.45, 0.45)
DOME_INTENSITY = 700.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_DEFAULT)
    p.add_argument("--urdf", default=URDF_DEFAULT)
    p.add_argument("--out_dir", default=OUT_DEFAULT)
    p.add_argument("--calib", default=CALIB_DEFAULT)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--settle_steps", type=int, default=15)
    return p.parse_args()


def load_calib(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    intr = c["intrinsics"]
    R = np.asarray(c["extrinsics"]["rotation_matrix"], dtype=np.float64)
    t = np.asarray(c["extrinsics"]["translation_m"], dtype=np.float64)
    return intr, R, t


def fov_x_from_intrinsics(fx, width):
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * fx))))


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: missing {csv_path}")
    if not Path(args.urdf).exists():
        sys.exit(f"ERROR: missing URDF {args.urdf}")
    if not Path(args.calib).exists():
        sys.exit(f"ERROR: missing calib {args.calib}")

    intr, R_cv, t_cv = load_calib(args.calib)
    print(f"[replay] Kinect calib t={t_cv.tolist()} fx={intr['fx']:.1f}", flush=True)

    print(f"[replay] CSV : {csv_path}", flush=True)
    print(f"[replay] URDF: {args.urdf}", flush=True)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    T = len(data)
    print(f"[replay] {T} frames", flush=True)

    out_dir = Path(args.out_dir).resolve()
    frames_dir = out_dir / "replay"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Boot Isaac Sim (DIRECT — no Isaac Lab wrapper)
    # ============================================================
    print("\n[replay] Booting Isaac Sim (headless)...", flush=True)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    import omni.kit.commands
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdLux, Gf, UsdShade, Sdf

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")

    # === Lighting: bright white dome (sim_renders 배경 = white) ===
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(DOME_INTENSITY)
    dome.CreateColorAttr(Gf.Vec3f(*DOME_COLOR))
    # Distant fill light for shape clarity
    dist = UsdLux.DistantLight.Define(stage, "/World/FillLight")
    dist.CreateIntensityAttr(800.0)
    dist.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    # === Table: medium gray procedural cube (sim_renders style — distinguishable from white dome) ===
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    txt = UsdGeom.Xformable(table.GetPrim())
    txt.ClearXformOpOrder()
    txt.AddTranslateOp().Set(Gf.Vec3d(TABLE_CENTER[0], TABLE_CENTER[1],
                                      TABLE_Z_TOP - TABLE_THICKNESS / 2.0))
    txt.AddScaleOp().Set(Gf.Vec3f(TABLE_SIZE[0] / 2.0,
                                  TABLE_SIZE[1] / 2.0,
                                  TABLE_THICKNESS / 2.0))
    mat_table = UsdShade.Material.Define(stage, "/World/Looks/TableMat")
    sh_table = UsdShade.Shader.Define(stage, "/World/Looks/TableMat/Shader")
    sh_table.CreateIdAttr("UsdPreviewSurface")
    sh_table.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*TABLE_COLOR))
    sh_table.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    sh_table.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat_table.CreateSurfaceOutput().ConnectToSource(sh_table.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(table.GetPrim()).Bind(mat_table)
    print(f"[replay] Table: gray {TABLE_COLOR} at ({TABLE_CENTER}, z={TABLE_Z_TOP*1000:.1f}mm)", flush=True)

    # === Robot: URDF import (v3 패턴) ===
    print(f"[replay] Importing URDF: {args.urdf}", flush=True)
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

    # === Robot color: 검정 override ===
    # Isaac Sim URDF importer puts mesh data in fabric backend (USD에는 Mesh prim 없음).
    # 모든 link visual은 /roarm_m3/Looks/material_silver MDL material에 binding.
    # → existing shader의 diffuse_color_constant를 직접 override (debug_roarm_prim_tree.py 진단).
    silver_shader_prim = stage.GetPrimAtPath("/roarm_m3/Looks/material_silver/Shader")
    if silver_shader_prim.IsValid():
        sh_silver = UsdShade.Shader(silver_shader_prim)
        inp = sh_silver.GetInput("diffuse_color_constant")
        if inp:
            old = inp.Get()
            inp.Set(Gf.Vec3f(0.03, 0.03, 0.03))
            print(f"[replay] Robot: material_silver diffuse_color_constant {old} -> (0.03,0.03,0.03)", flush=True)
        else:
            print(f"[replay] WARN: diffuse_color_constant input not found on silver shader", flush=True)
        # Kill any reflective/metallic shading so robot stays uniformly black.
        for name, val in [("reflection_roughness_constant", 0.95),
                          ("metallic_constant", 0.0),
                          ("specular_level", 0.0),
                          ("diffuse_tint", Gf.Vec3f(0.03, 0.03, 0.03))]:
            inp_extra = sh_silver.GetInput(name)
            if inp_extra:
                try:
                    inp_extra.Set(val)
                except Exception:
                    pass
    else:
        print(f"[replay] WARN: /roarm_m3/Looks/material_silver/Shader not found — robot stays silver", flush=True)

    # === Sponge: pink cube (P6v12 1-sponge) ===
    # Op order: T · Rz · Ry · S  (apply scale → Y-tilt → Z-spin → translate)
    # Rz tracks wrist_roll delta (j4_t - j4_grasp); Ry tracks wrist_pitch delta (j3_t - j3_grasp).
    sponge = UsdGeom.Cube.Define(stage, "/World/Sponge")
    sponge_xf = UsdGeom.Xformable(sponge.GetPrim())
    sponge_xf.ClearXformOpOrder()
    op_t = sponge_xf.AddTranslateOp()
    op_rz = sponge_xf.AddRotateZOp()
    op_ry = sponge_xf.AddRotateYOp()
    op_s = sponge_xf.AddScaleOp()
    op_t.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    op_rz.Set(0.0)
    op_ry.Set(0.0)
    op_s.Set(Gf.Vec3f(SPONGE_LEN / 2.0, SPONGE_WIDTH / 2.0, SPONGE_HEIGHT / 2.0))
    mat_sp = UsdShade.Material.Define(stage, "/World/Looks/SpongeMat")
    sh_sp = UsdShade.Shader.Define(stage, "/World/Looks/SpongeMat/Shader")
    sh_sp.CreateIdAttr("UsdPreviewSurface")
    sh_sp.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*SPONGE_COLOR))
    sh_sp.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    mat_sp.CreateSurfaceOutput().ConnectToSource(sh_sp.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(sponge.GetPrim()).Bind(mat_sp)
    print(f"[replay] Sponge: pink edge-stand 47mm", flush=True)

    # === Articulation handle for joint control ===
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    world = World(stage_units_in_meters=1.0)
    robot = world.scene.add(Articulation(prim_path="/roarm_m3", name="roarm_m3"))
    world.reset()
    print(f"[replay] Robot DOFs={robot.num_dof}, names={robot.dof_names}", flush=True)

    # === Camera: Kinect calib (intrinsics + extrinsics) — matches sim_renders_v2/stacking_initial.png ===
    cam_prim = UsdGeom.Camera.Define(stage, "/World/SimCam")
    hfov = fov_x_from_intrinsics(intr["fx"], args.width)
    aperture_h = 20.955
    focal = (aperture_h / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    cam_prim.CreateFocalLengthAttr(float(focal))
    cam_prim.CreateHorizontalApertureAttr(float(aperture_h))
    cam_prim.CreateVerticalApertureAttr(float(aperture_h * args.height / args.width))
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    # CV (OpenCV) → USD convention: flip Y and Z columns
    flip = np.diag([1.0, -1.0, -1.0])
    R_usd = R_cv @ flip
    M = np.eye(4); M[:3, :3] = R_usd; M[:3, 3] = t_cv
    M_gf = Gf.Matrix4d(*M.T.flatten().tolist())
    cam_xf = UsdGeom.Xformable(cam_prim.GetPrim())
    cam_xf.ClearXformOpOrder()
    cam_xf.AddTransformOp().Set(M_gf)
    print(f"[replay] Camera HFOV={hfov:.2f}° focal_mm={focal:.2f} eye={t_cv.tolist()}", flush=True)

    rp = rep.create.render_product("/World/SimCam", (args.width, args.height))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([rp])
    for _ in range(args.settle_steps):
        app.update()
    print(f"[replay] settled {args.settle_steps} frames", flush=True)

    # ============================================================
    # Pre-pass: locate first grasped frame for delta-rotation anchor.
    # ============================================================
    grasped_flags = data[:, 10] > 0.5
    grasp_idx = np.where(grasped_flags)[0]
    if grasp_idx.size > 0:
        first_grasp = int(grasp_idx[0])
        j4_grasp = float(data[first_grasp, 5])   # wrist_roll  at grasp
        j3_grasp = float(data[first_grasp, 4])   # wrist_pitch at grasp
        print(f"[replay] first_grasp f={first_grasp}  j4_grasp={j4_grasp:+.2f}°  j3_grasp={j3_grasp:+.2f}°", flush=True)
    else:
        first_grasp = T   # never grasped
        j4_grasp = j3_grasp = 0.0
        print(f"[replay] WARN: no grasp frames found — sponge stays at spawn pose", flush=True)

    spawn_xyz = (float(data[0, 7]), float(data[0, 8]), float(data[0, 9]))
    last_held = {"xyz": spawn_xyz, "rz": 0.0, "ry": 0.0}

    # ============================================================
    # Frame-by-frame replay
    # ============================================================
    from PIL import Image
    t0 = time.time()
    for f in range(T):
        row = data[f]
        # CSV: t, j0~5_deg, sponge_x,y,z, grasped, tcp_x,y,z
        joints_deg = row[1:7]
        sponge_xyz_csv = row[7:10]
        grasped = row[10] > 0.5
        joints_rad = np.radians(joints_deg)

        robot.set_joint_positions(joints_rad)

        # Sponge state machine (3-state):
        #   pre-grasp     : spawn pose, rz=ry=0
        #   held          : csv xyz follow (TCP propagated) + delta wrist_roll/pitch
        #   post-grasp    : freeze last held pose (release rare in RL policy)
        if f < first_grasp:
            sp_pos = spawn_xyz
            rz = 0.0
            ry = 0.0
        elif grasped:
            sp_pos = (float(sponge_xyz_csv[0]), float(sponge_xyz_csv[1]), float(sponge_xyz_csv[2]))
            rz = float(joints_deg[4]) - j4_grasp                # wrist_roll delta
            ry = -(float(joints_deg[3]) - j3_grasp)             # wrist_pitch delta (sign TBD via dry-run)
            last_held["xyz"] = sp_pos
            last_held["rz"] = rz
            last_held["ry"] = ry
        else:
            sp_pos = last_held["xyz"]
            rz = last_held["rz"]
            ry = last_held["ry"]

        op_t.Set(Gf.Vec3d(*sp_pos))
        op_rz.Set(rz)
        op_ry.Set(ry)

        for _ in range(2):
            world.step(render=True)
            app.update()
        rep.orchestrator.step()
        app.update()

        rgb = rgb_annot.get_data()
        if rgb is None or rgb.ndim != 3:
            print(f"  WARN ep f{f}: rgb={type(rgb)}", flush=True)
            continue
        Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(frames_dir / f"frame_{f:04d}.png")

        if f < 3 or f % 25 == 0 or f == T - 1:
            tag = "PRE" if f < first_grasp else ("HELD" if grasped else "POST")
            print(f"  f={f:3d} [{tag:4s}]  j3={joints_deg[3]:+.1f}° j4={joints_deg[4]:+.1f}°  rz={rz:+.1f}° ry={ry:+.1f}°  "
                  f"sp=({sp_pos[0]*1000:+.0f},{sp_pos[1]*1000:+.0f},{sp_pos[2]*1000:+.0f})mm", flush=True)

    elapsed = time.time() - t0
    print(f"[replay] {T} frames in {elapsed:.1f}s  ({elapsed/T*1000:.0f} ms/frame)", flush=True)

    # ============================================================
    # ffmpeg encode
    # ============================================================
    import subprocess
    import imageio_ffmpeg
    mp4_path = out_dir / f"{out_dir.name}.mp4"
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    pngs = sorted(frames_dir.glob("frame_*.png"))
    print(f"[replay] encoding {len(pngs)} -> {mp4_path}", flush=True)
    list_path = out_dir / "_concat.txt"
    with open(list_path, "w") as fh:
        for p in pngs:
            fh.write(f"file '{p}'\nduration {1.0/args.fps}\n")
        fh.write(f"file '{pngs[-1]}'\n")
    rc = subprocess.call([
        ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", f"fps={args.fps}",
        "-crf", "23", str(mp4_path),
    ])
    if rc == 0:
        print(f"[replay] MP4 OK: {mp4_path}", flush=True)
    else:
        print(f"[replay] ffmpeg FAIL rc={rc}", flush=True)

    app.close()


if __name__ == "__main__":
    main()
