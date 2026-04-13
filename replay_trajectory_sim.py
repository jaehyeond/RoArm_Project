"""
Replay v6 parquet trajectories in Isaac Sim and save rendered images.

Phase 1 of Isaac Sim pipeline: verify URDF joint replay matches real robot.
Outputs rendered images for SigLIP Sim-Real comparison (sim_real_compare.py).

Uses omni.replicator.core for headless rendering (Camera sensor doesn't work).

Usage:
    conda activate isaaclab
    python replay_trajectory_sim.py --episode 0 --output-dir sim_renders
    python replay_trajectory_sim.py --episode 0 5 --output-dir sim_renders
    python replay_trajectory_sim.py --all --output-dir sim_renders --skip-frames 5
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# Constants
# ============================================================
URDF_PATH_ALT = "/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf"
PARQUET_PATH = str(Path(__file__).parent / "lerobot_dataset_v6" / "data" / "chunk-000" / "file-000.parquet")

JOINT_NAMES = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
]

# Camera position approximating Azure Kinect viewpoint
CAM_EYE = (0.45, 0.15, 0.40)
CAM_TARGET = (0.0, 0.0, 0.10)


def parse_args():
    parser = argparse.ArgumentParser(description="Replay v6 trajectories in Isaac Sim")
    parser.add_argument("--episode", type=int, nargs="+", default=[0],
                        help="Episode index(es) to replay")
    parser.add_argument("--all", action="store_true",
                        help="Replay all episodes")
    parser.add_argument("--output-dir", type=str, default="sim_renders",
                        help="Output directory for rendered images")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Save every N-th frame (1=all, 5=every 5th)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--settle-steps", type=int, default=30,
                        help="app.update() steps to settle scene")
    parser.add_argument("--parquet", type=str, default=PARQUET_PATH)
    parser.add_argument("--urdf", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve URDF
    urdf_path = args.urdf or URDF_PATH_ALT
    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found at {urdf_path}")
        sys.exit(1)

    # Load parquet before Isaac Sim (faster feedback on errors)
    print(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    states_all = np.array(df["observation.state"].tolist())  # (N, 6) degrees
    episodes = df["episode_index"].values

    if args.all:
        ep_list = sorted(df["episode_index"].unique())
    else:
        ep_list = args.episode

    print(f"Episodes: {ep_list}, Total frames: {len(df)}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # Isaac Sim setup
    # ============================================================
    print("Starting Isaac Sim (headless)...")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    import omni.kit.commands
    import omni.usd
    from pxr import UsdGeom, UsdLux, Gf
    import omni.replicator.core as rep

    stage = omni.usd.get_context().get_stage()

    # Dome light
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1500.0)

    # Import URDF
    print(f"Loading URDF: {urdf_path}")
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.set_fix_base(True)
    import_config.set_make_default_prim(True)
    import_config.set_distance_scale(1.0)
    import_config.set_create_physics_scene(True)

    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile", urdf_path=urdf_path, import_config=import_config
    )
    if not result:
        print("ERROR: URDF parse failed")
        app.close()
        sys.exit(1)

    omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_path=urdf_path,
        urdf_robot=robot_model,
        import_config=import_config,
        dest_path="",
    )
    print("URDF imported OK")

    # Let stage settle after URDF import
    for _ in range(10):
        app.update()

    # World + Articulation
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    robot = world.scene.add(
        Articulation(prim_path="/roarm_m3", name="roarm_m3")
    )

    world.reset()
    print(f"DOFs={robot.num_dof}, names={robot.dof_names}")

    for i, name in enumerate(JOINT_NAMES):
        assert robot.dof_names[i] == name, f"DOF mismatch: {name} vs {robot.dof_names[i]}"

    # ============================================================
    # Camera via USD + Replicator
    # ============================================================
    cam_prim = UsdGeom.Camera.Define(stage, "/World/SimCam")
    cam_prim.CreateFocalLengthAttr(24.0)
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    # LookAt transform
    eye = Gf.Vec3d(*CAM_EYE)
    target = Gf.Vec3d(*CAM_TARGET)
    up = Gf.Vec3d(0.0, 0.0, 1.0)

    look_at = Gf.Matrix4d()
    look_at.SetLookAt(eye, target, up)
    cam_xform = UsdGeom.Xformable(cam_prim)
    cam_xform.AddTransformOp().Set(look_at.GetInverse())

    for _ in range(10):
        app.update()

    # Replicator render product + annotator
    render_product = rep.create.render_product("/World/SimCam", (args.width, args.height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])

    # Settle scene
    for _ in range(args.settle_steps):
        app.update()

    print("Camera + Replicator ready")

    # ============================================================
    # Replay loop
    # ============================================================
    from PIL import Image

    metadata = {
        "urdf_path": urdf_path,
        "parquet_path": args.parquet,
        "cam_eye": list(CAM_EYE),
        "cam_target": list(CAM_TARGET),
        "width": args.width,
        "height": args.height,
        "skip_frames": args.skip_frames,
        "episodes": {},
    }

    for ep_idx in ep_list:
        ep_mask = episodes == ep_idx
        ep_states_deg = states_all[ep_mask]
        ep_states_rad = np.radians(ep_states_deg)
        T = len(ep_states_rad)

        ep_dir = os.path.join(args.output_dir, f"episode_{ep_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)

        print(f"\nEpisode {ep_idx}: {T} frames")
        saved_count = 0

        for t in range(T):
            # Set joint positions
            robot.set_joint_positions(ep_states_rad[t])

            # Step physics + render
            world.step(render=True)
            app.update()

            # Save frame
            if t % args.skip_frames == 0:
                rep.orchestrator.step()
                app.update()

                rgb_data = rgb_annotator.get_data()
                if rgb_data is not None and len(rgb_data.shape) == 3:
                    img = Image.fromarray(rgb_data[:, :, :3].astype(np.uint8))
                    frame_path = os.path.join(ep_dir, f"frame_{t:04d}.png")
                    img.save(frame_path)
                    saved_count += 1

                    if saved_count == 1 or saved_count % 20 == 0:
                        actual_pos = robot.get_joint_positions()
                        error_deg = np.abs(np.degrees(actual_pos) - ep_states_deg[t])
                        print(f"  Frame {t}/{T}: max_error={error_deg.max():.2f}°")

        # Final error
        actual_pos = robot.get_joint_positions()
        error_deg = np.abs(np.degrees(actual_pos) - ep_states_deg[-1])

        metadata["episodes"][str(ep_idx)] = {
            "num_frames": T,
            "saved_frames": saved_count,
            "final_error_deg": error_deg.tolist(),
            "max_final_error_deg": float(error_deg.max()),
        }
        print(f"  Episode {ep_idx} done: {saved_count} images, max_error={error_deg.max():.2f}°")

    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata: {meta_path}")

    app.close()
    print("Done!")


if __name__ == "__main__":
    main()
