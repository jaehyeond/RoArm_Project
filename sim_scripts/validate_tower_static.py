"""Validate HARD RULE #20 # tower geometry physical stability in Isaac Sim (Bug #3).

A1 Manipulation specialist 산수 (5/13 evening):
  L2.spot1 CoM X=+0.2465 < L1 support polygon min X=+0.269 (22.5mm 차이)
  L2.spot2 CoM X=+0.3135 > L1 support polygon max X=+0.291 (22.5mm 차이)
  → 각 L2 sponge 단독으로는 CoM이 L1 support 밖 → 놓는 즉시 X 방향 tipping

이 스크립트는 산수 결론을 sim physics로 확인:
  1. 4 sponge edge-stand cuboids 정적 배치 (L1.spot1/spot2 → L2.spot1/spot2)
  2. 10s physics step (no robot, gravity only)
  3. 각 sponge final pos + upright score 측정 → tipping rate 출력

Edge-stand 47mm tall × 22mm wide × 125mm long. HARD RULE #19/#20.

Run (4090, isaaclab env):
    conda run -n isaaclab python sim_scripts/validate_tower_static.py --steps 1000
"""
from __future__ import annotations

import argparse
import math

import torch

# =====================================================================
# Geometry constants (HARD RULE #19/#20)
# =====================================================================
TABLE_Z = -0.012117
SPONGE_HEIGHT_EDGE = 0.047
SPONGE_LEN_LONG = 0.125
SPONGE_WIDTH = 0.022
SPONGE_CENTER_Z_L1 = TABLE_Z + SPONGE_HEIGHT_EDGE / 2.0  # +0.011383
SPONGE_CENTER_Z_L2 = TABLE_Z + SPONGE_HEIGHT_EDGE * 1.5  # +0.058383

LAYOUT_CENTER_X = 0.280
L1_Y_C2C = 0.087
L2_X_C2C = 0.067

# 4 target poses (xyz, quat_wxyz). L1 yaw=0° (long axis Y), L2 yaw=90° (long axis X).
TOWER_TARGETS = [
    # name, pos, quat_wxyz
    ("L1.spot1", (LAYOUT_CENTER_X, -L1_Y_C2C / 2, SPONGE_CENTER_Z_L1), (1.0, 0.0, 0.0, 0.0)),
    ("L1.spot2", (LAYOUT_CENTER_X, +L1_Y_C2C / 2, SPONGE_CENTER_Z_L1), (1.0, 0.0, 0.0, 0.0)),
    ("L2.spot1", (LAYOUT_CENTER_X - L2_X_C2C / 2, 0.0, SPONGE_CENTER_Z_L2),
     (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))),
    ("L2.spot2", (LAYOUT_CENTER_X + L2_X_C2C / 2, 0.0, SPONGE_CENTER_Z_L2),
     (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400, help="Physics steps (dt=1/200) → 2s")
    p.add_argument("--dt", type=float, default=1.0 / 200, help="Sim dt")
    p.add_argument("--upright_thresh", type=float, default=0.90,
                   help="sz_world_z threshold for upright (~25° tipping)")
    p.add_argument("--friction", type=float, default=1.5, help="static friction (match env)")
    p.add_argument("--restitution", type=float, default=0.0, help="match env")
    p.add_argument("--mode", type=str, default="full", choices=["full", "l1_only", "l2_only"],
                   help="full = spawn all 4. l1_only = L1 only. l2_only = sequential L1→L2 isolated")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[validate_tower] launching Isaac Sim (headless, no cameras)...", flush=True)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    sim_app = app_launcher.app

    print(f"[validate_tower] Isaac Sim launched OK", flush=True)

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    # Sim setup
    sim_cfg = SimulationCfg(
        dt=args.dt,
        device="cuda:0",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.6, 0.4], [0.28, 0.0, 0.05])

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    # Light (cosmetic)
    light_cfg = sim_utils.DomeLightCfg(intensity=1000.0)
    light_cfg.func("/World/Light", light_cfg)

    # Spawn 4 sponges via RigidObject (no robot)
    sponges = {}
    for name, pos, quat in TOWER_TARGETS:
        if args.mode == "l1_only" and name.startswith("L2"):
            continue
        if args.mode == "l2_only":
            # 이 mode는 L2만 단독 테스트 (L1 없이 floating → expected fall)
            if name.startswith("L1"):
                continue
        cfg = RigidObjectCfg(
            prim_path=f"/World/Sponges/{name.replace('.', '_')}",
            spawn=sim_utils.CuboidCfg(
                size=(SPONGE_LEN_LONG, SPONGE_WIDTH, SPONGE_HEIGHT_EDGE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=args.friction,
                    dynamic_friction=args.friction * 0.8,
                    restitution=args.restitution,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95 if name.startswith("L1") else 0.85, 0.55, 0.55),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=quat),
        )
        sponges[name] = RigidObject(cfg)

    sim.reset()

    # Initial positions (before step)
    print(f"\n[{args.mode}] === Initial positions ===")
    init_pos = {}
    for name, sp in sponges.items():
        pos = sp.data.root_pos_w[0].cpu().numpy()
        quat = sp.data.root_quat_w[0].cpu().numpy()
        sz = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        init_pos[name] = (pos.copy(), sz.item())
        print(f"  {name}: pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})  sz_world_z={sz:+.3f}")

    # Step physics
    print(f"\nStepping {args.steps} steps (dt={args.dt}) = {args.steps * args.dt:.2f}s simulated...")
    for i in range(args.steps):
        sim.step()
        for sp in sponges.values():
            sp.update(args.dt)

    # Final state
    print(f"\n[{args.mode}] === Final positions after {args.steps * args.dt:.2f}s ===")
    n_tipped = 0
    n_total = len(sponges)
    n_displaced = 0
    for name, sp in sponges.items():
        pos = sp.data.root_pos_w[0].cpu().numpy()
        quat = sp.data.root_quat_w[0].cpu().numpy()
        sz = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        init_p, init_sz = init_pos[name]
        dx = pos[0] - init_p[0]
        dy = pos[1] - init_p[1]
        dz = pos[2] - init_p[2]
        d_xy = (dx ** 2 + dy ** 2) ** 0.5
        tipped = bool(sz < args.upright_thresh)
        displaced = bool(d_xy > 0.010)  # > 10mm shift
        n_tipped += int(tipped)
        n_displaced += int(displaced)
        flags = []
        if tipped:
            flags.append("TIPPED")
        if displaced:
            flags.append("DISPLACED")
        flag_str = " ".join(flags) if flags else "OK"
        print(f"  {name}: pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})  "
              f"sz_world_z={sz:+.3f}  Δxy={d_xy * 1000:+.1f}mm  Δz={dz * 1000:+.1f}mm  [{flag_str}]")

    print(f"\n=== SUMMARY ({args.mode}) ===")
    print(f"  Tipped: {n_tipped} / {n_total} ({100 * n_tipped / max(n_total, 1):.1f}%)")
    print(f"  Displaced (>10mm xy): {n_displaced} / {n_total} ({100 * n_displaced / max(n_total, 1):.1f}%)")

    if args.mode == "full":
        # Predicted: L2 spot1/spot2 tipping (산수 결론)
        l2_names = ["L2.spot1", "L2.spot2"]
        l2_tipped = sum(
            int(1.0 - 2.0 * (sponges[n].data.root_quat_w[0, 1] ** 2 + sponges[n].data.root_quat_w[0, 2] ** 2) < args.upright_thresh)
            for n in l2_names
        )
        print(f"\n  HARD RULE #20 prediction: L2 sponges tip (CoM outside L1 support polygon)")
        print(f"  Result: L2 tipped {l2_tipped}/2")
        if l2_tipped >= 1:
            print(f"  ⚠️  CONFIRMED: # tower geometry is sim-unstable. Sequential placement → cascade failure.")
        else:
            print(f"  ✓ Geometry stable in sim (real-world friction/deformation may explain user's photo).")

    sim_app.close()


if __name__ == "__main__":
    main()
