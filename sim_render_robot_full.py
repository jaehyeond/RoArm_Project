#!/usr/bin/env python3
"""로봇 전체를 화면에 꽉 차게 렌더 (단순·확실). Isaac Sim 5.1 RTX.

자연스러운 자세 + 그랩 개방. 카메라는 로봇 AABB 를 담게 멀리서(거리 1.14m, 검증된 범위).
사용: OMNI_KIT_ACCEPT_EULA=YES python sim_render_robot_full.py <usd> <out.png>
"""
import sys, os, glob, shutil
from isaacsim import SimulationApp
USD = os.path.abspath(sys.argv[1]); OUT = os.path.abspath(sys.argv[2])
sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 1000})

import numpy as np, omni.usd
from pxr import UsdGeom, UsdLux, Gf
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation

world = World(stage_units_in_meters=1.0)
add_reference_to_stage(usd_path=USD, prim_path="/World/roarm")
stage = omni.usd.get_context().get_stage()
world.scene.add_default_ground_plane()
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1300.0)
kl = UsdLux.DistantLight.Define(stage, "/World/Key"); kl.CreateIntensityAttr(2600.0)
UsdGeom.XformCommonAPI(kl).SetRotate(Gf.Vec3f(-45, 30, 0))
world.get_physics_context().set_gravity(0.0)

art = SingleArticulation(prim_path="/World/roarm", name="roarm")
world.reset(); art.initialize()
names = list(art.dof_names); print("DOF:", names)

q = np.zeros(art.num_dof)
POSE = {"link1_to_link2": 0.55, "link2_to_link3": 0.95, "link3_to_link4": 0.15,
        "grab_shell_L_joint": 0.77667, "grab_shell_R_joint": 0.77667}
for nm, v in POSE.items():
    if nm in names: q[names.index(nm)] = v
art.set_joint_positions(q)
try: art.set_joint_position_targets(q)
except Exception: pass
for _ in range(45): world.step(render=True)

# 로봇 전체를 담는 카메라 (AABB center=(0.26,0.03,0.18), diag 0.81 → 거리 1.14m)
cam = rep.create.camera(position=(1.02, -0.73, 0.54), look_at=(0.26, 0.03, 0.18))
rp = rep.create.render_product(cam, (1280, 1000))
d = os.path.join(os.path.dirname(OUT), "_fullshot")
if os.path.isdir(d): shutil.rmtree(d)
os.makedirs(d, exist_ok=True)
w = rep.WriterRegistry.get("BasicWriter"); w.initialize(output_dir=d, rgb=True); w.attach([rp])
for _ in range(12): rep.orchestrator.step()
rep.orchestrator.wait_until_complete()
fs = sorted(glob.glob(os.path.join(d, "rgb_*.png")))
if fs: shutil.move(fs[-1], OUT); print("saved", OUT)
else: print("❌ no png")
sim_app.close(); print("[done]")
