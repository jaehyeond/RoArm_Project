#!/usr/bin/env python3
"""스쿱 시퀀스 렌더: 열림→하강→닫힘→인양, 여러 프레임. Isaac Sim 5.1 RTX.

🔴 순정 조(gripper_link)는 **숨긴다** — 커스텀 클램셸 그랩만 명확히 보이게(순정 조는
   실물에선 그랩을 구동하는 내부 인터페이스로 남지만 시각적으로는 클러터).
카메라는 앞에서(그랩 입이 아래·셸이 ±Y 로 벌어지므로 좌우로 벌어지는 게 보인다).
사용: OMNI_KIT_ACCEPT_EULA=YES python sim_render_scoop_sequence.py <usd> <out_prefix>
"""
import sys, os, glob, shutil
from isaacsim import SimulationApp
USD = os.path.abspath(sys.argv[1]); PREFIX = os.path.abspath(sys.argv[2])
sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 900})

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
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1400.0)
kl = UsdLux.DistantLight.Define(stage, "/World/Key"); kl.CreateIntensityAttr(2500.0)
UsdGeom.XformCommonAPI(kl).SetRotate(Gf.Vec3f(-40, 25, 0))
world.get_physics_context().set_gravity(0.0)

# 🔴 순정 조 숨김
g = stage.GetPrimAtPath("/World/roarm/gripper_link")
if g and g.IsValid():
    UsdGeom.Imageable(g).MakeInvisible()
    print("gripper_link hidden:", g.IsValid())

# 더미(펠릿 파일) — 더 크게, 연회색
pile = UsdGeom.Cone.Define(stage, "/World/pile")
pile.CreateHeightAttr(0.08); pile.CreateRadiusAttr(0.11); pile.CreateAxisAttr("Z")
UsdGeom.XformCommonAPI(pile).SetTranslate(Gf.Vec3d(0.25, 0.0, 0.04))
pile.CreateDisplayColorAttr([(0.85, 0.83, 0.77)])

art = SingleArticulation(prim_path="/World/roarm", name="roarm")
world.reset(); art.initialize()
names = list(art.dof_names); print("DOF:", names)

HIGH = {"link1_to_link2": 0.12, "link2_to_link3": 1.18, "link3_to_link4": 1.45}
LOW  = {"link1_to_link2": 0.42, "link2_to_link3": 1.78, "link3_to_link4": 0.81}
OPEN = 0.77667

# 시퀀스: (팔자세, 개폐, 라벨)
frames = [
    (HIGH, OPEN,  "1_open_over"),    # 열림, 더미 위
    (LOW,  OPEN,  "2_open_down"),    # 하강(더미 속)
    (LOW,  0.40,  "3_closing"),      # 닫는 중
    (LOW,  0.0,   "4_closed"),       # 닫힘(퍼냄)
    (HIGH, 0.0,   "5_lift"),         # 인양
]

# 작동 검증된 거리(~0.85m). 가까우면(~0.5m) 프레임을 놓쳤다.
cam = rep.create.camera(position=(0.80, -0.60, 0.55), look_at=(0.25, 0.0, 0.14))
rp = rep.create.render_product(cam, (1280, 900))

def render_to(out):
    d = os.path.join(os.path.dirname(PREFIX), "_seqshot")
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    w = rep.WriterRegistry.get("BasicWriter"); w.initialize(output_dir=d, rgb=True); w.attach([rp])
    for _ in range(10): rep.orchestrator.step()
    rep.orchestrator.wait_until_complete(); w.detach()
    fs = sorted(glob.glob(os.path.join(d, "rgb_*.png")))
    if fs: shutil.move(fs[-1], out); print("  ->", out)
    else: print("  ❌ no png")

for pose, ov, label in frames:
    q = np.zeros(art.num_dof)
    for nm, v in pose.items():
        if nm in names: q[names.index(nm)] = v
    for nm in ("grab_shell_L_joint", "grab_shell_R_joint"):
        if nm in names: q[names.index(nm)] = ov
    art.set_joint_positions(q)
    try: art.set_joint_position_targets(q)
    except Exception: pass
    for _ in range(45): world.step(render=True)
    print(f"[frame {label}] open={ov}")
    render_to(f"{PREFIX}_{label}.png")

sim_app.close(); print("[done]")
