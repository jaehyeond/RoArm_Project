#!/usr/bin/env python3
"""합성 USD(로봇+그랩)를 Isaac Sim 5.1 RTX 렌더 → 그랩 근접, **닫힘/열림 둘 다** PNG.

셸 개폐를 명확히 보이려고 같은 자세·카메라에서 grab_shell 0(닫힘)·0.777(완전개방) 두 장.
⚠️ mimic→normal 변환된 USD 라야 양쪽 셸이 독립 구동된다.
⚠️ D447: close() 는 예외 삼킴 → 호출부가 PNG 검증. 캡처는 annotator 로 픽셀 직접(파일명 의존 X).
사용: OMNI_KIT_ACCEPT_EULA=YES python sim_render_grab_usd.py <usd> <out_prefix>
"""
import sys, os, math
from isaacsim import SimulationApp

USD = os.path.abspath(sys.argv[1])
PREFIX = os.path.abspath(sys.argv[2])          # <prefix>_closed.png / _open.png
sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 800})

import numpy as np
import omni.usd
from pxr import UsdGeom, UsdLux, Gf
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from PIL import Image

world = World(stage_units_in_meters=1.0)
add_reference_to_stage(usd_path=USD, prim_path="/World/roarm")
stage = omni.usd.get_context().get_stage()
world.scene.add_default_ground_plane()
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1500.0)
key = UsdLux.DistantLight.Define(stage, "/World/KeyLight"); key.CreateIntensityAttr(3000.0)
UsdGeom.XformCommonAPI(key).SetRotate(Gf.Vec3f(-35, 20, 0))
world.get_physics_context().set_gravity(0.0)

# 더미(펠릿 파일) 프록시 — 그랩 입 아래 바닥
pile = UsdGeom.Cone.Define(stage, "/World/pile")
pile.CreateHeightAttr(0.06); pile.CreateRadiusAttr(0.09); pile.CreateAxisAttr("Z")
UsdGeom.XformCommonAPI(pile).SetTranslate(Gf.Vec3d(0.25, 0.0, 0.03))
pile.CreateDisplayColorAttr([(0.82, 0.80, 0.74)])

art = SingleArticulation(prim_path="/World/roarm", name="roarm")
world.reset(); art.initialize()
names = list(art.dof_names)
print("DOF:", names)

# 스쿱 자세: 그랩 입이 아래(-Z) → 더미 위로 내려 닫아 퍼냄
base_pose = {"link1_to_link2": 0.39, "link2_to_link3": 1.39, "link3_to_link4": 1.34}

# 카메라: 작동 검증된 3/4 뷰(앞-오른쪽-위). 팔 끝 + 그랩이 함께 보이고 셸 개폐 차이가 드러난다.
import glob, shutil
cam = rep.create.camera(position=(0.78, -0.60, 0.52), look_at=(0.26, 0.0, 0.15))
rp = rep.create.render_product(cam, (1280, 800))

def shot(open_val, out, tag):
    q = np.zeros(art.num_dof)
    for nm, v in base_pose.items():
        if nm in names: q[names.index(nm)] = v
    for nm in ("grab_shell_L_joint", "grab_shell_R_joint"):
        if nm in names: q[names.index(nm)] = open_val
    art.set_joint_positions(q)
    try: art.set_joint_position_targets(q)
    except Exception: pass
    for _ in range(40): world.step(render=True)
    d = os.path.join(os.path.dirname(PREFIX), f"_shot_{tag}")
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    w = rep.WriterRegistry.get("BasicWriter"); w.initialize(output_dir=d, rgb=True)
    w.attach([rp])
    for _ in range(10): rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()
    w.detach()
    fs = sorted(glob.glob(os.path.join(d, "rgb_*.png")))
    if fs:
        shutil.move(fs[-1], out); print(f"[shot] open={open_val} -> {out}")
    else:
        print(f"[shot] ❌ open={open_val} PNG 미생성")

shot(0.0, PREFIX + "_closed.png", "closed")
shot(0.77667, PREFIX + "_open.png", "open")
sim_app.close()
print("[done]")
