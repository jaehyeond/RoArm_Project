#!/usr/bin/env python3
"""S1 합성 USD(로봇 + 문 + 고정부) 근접 RTX 렌더 — sim_render_grab_closeup.py(D477) 의 S1 판.
관절 = link5_to_gripper_link 하나(문 직결). 자세 2(HOME/스쿱) × 개폐 2(0 / 0.511 rad=29.3°) × 시점 2 = 8장.
BasicWriter 금지·annotator 1장·JSON 선기록·close() 뒤 exit 코드 불신 (D477).
사용: OMNI_KIT_ACCEPT_EULA=YES python sim_render_s1_closeup.py <usd> <out_dir>
"""
import sys, os, json
from isaacsim import SimulationApp
USD = os.path.abspath(sys.argv[1]); OUT = os.path.abspath(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 960})
import numpy as np, omni.usd
from pxr import Usd, UsdGeom, UsdLux, Gf
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from PIL import Image
world = World(stage_units_in_meters=1.0)
add_reference_to_stage(usd_path=USD, prim_path="/World/roarm")
stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(3000.0)
# 🔴 키 라이트 1방향이면 힌지축(link5 Y) 법선 면이 빛을 못 받아 검게 나온다(09-03, 법선 맵으로 확인) → 4방향
for i, rot in enumerate(((-40, 30, 0), (-40, -150, 0), (30, 100, 0), (-30, -80, 0))):
    L = UsdLux.DistantLight.Define(stage, f"/World/KeyLight{i}"); L.CreateIntensityAttr(1800.0); UsdGeom.XformCommonAPI(L).SetRotate(Gf.Vec3f(*rot))
world.get_physics_context().set_gravity(0.0)
# 🔴 IsaacLab 변환 USD 는 collision 조각도 purpose=default·기본(검정) 재질로 들어와 시각 메시 위에 겹쳐 그려진다(09-03 발견).
#    → /collisions 아래 프림 전부 비가시화(물리에는 영향 없음). 벤더 메쉬 뒷면 대비 양면 셰이딩도 켠다.
n_hidden = 0
for prim in list(stage.Traverse()):                       # instanceable 링크 → 해제해야 자식(프록시) 편집 가능
    if prim.IsInstanceable(): prim.SetInstanceable(False)
for prim in list(stage.Traverse()):
    path = prim.GetPath().pathString
    if path.endswith("/collisions") and prim.GetParent().GetPath().pathString.startswith("/World/roarm/"):
        stage.RemovePrim(prim.GetPath()); n_hidden += 1          # 비가시화가 RTX 에 안 먹혀 아예 제거 (렌더 전용 스크립트)
for prim in list(stage.Traverse()):
    if prim.GetTypeName() == "Mesh": UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
print(f"[hide] collision prims hidden: {n_hidden}", flush=True)
art = SingleArticulation(prim_path="/World/roarm", name="roarm"); world.reset(); art.initialize()
for _ in range(5): world.step(render=False)
names = list(art.dof_names); print("DOF:", names, flush=True)
J = "link5_to_gripper_link"; OPEN = 0.511
POSES = {"home": {}, "scoop": {"link1_to_link2": 0.39, "link2_to_link3": 1.39, "link3_to_link4": 1.34}}
LOG = []
def wpos(path):
    m = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(path)); t = m.ExtractTranslation(); return np.array([t[0], t[1], t[2]])
def bowl_world():
    """보울 중심 = link5 프레임 (X 8.1, Y 0, Z 145) mm → 월드."""
    m = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath("/World/roarm/link5"))
    p = m.Transform(Gf.Vec3d(0.0081, 0.0, 0.145)); return np.array([p[0], p[1], p[2]])
def set_pose(pose, jaw):
    q = np.zeros(art.num_dof)
    for nm, v in pose.items():
        for i, dn in enumerate(names):
            if nm in dn: q[i] = v
    if J in names: q[names.index(J)] = jaw
    if art.get_joint_positions() is None:
        world.play(); [world.step(render=False) for _ in range(3)]; art.initialize()
    art.set_joint_positions(q)
    try: art.set_joint_position_targets(q)
    except Exception: pass
    for _ in range(30): world.step(render=True)
    got = art.get_joint_positions()
    rec = {"target_jaw": jaw, "jaw_readback": (round(float(got[names.index(J)]), 4) if got is not None and J in names else None), "arm": ([round(float(v), 3) for v in got[:5]] if got is not None else None)}
    rec["ok"] = rec["jaw_readback"] is not None and abs(rec["jaw_readback"] - jaw) <= 0.05
    LOG.append(rec); json.dump(LOG, open(os.path.join(OUT, "closeup_poses.json"), "w"), indent=2); print("[pose]", rec, flush=True)
def shot(tag, cam_dir, dist=1.0, focal=70.0):
    tgt = bowl_world()
    d = np.asarray(cam_dir, float); d /= np.linalg.norm(d); pos = tgt + d * dist
    cam = rep.create.camera(position=tuple(pos.tolist()), look_at=tuple(tgt.tolist()), focal_length=focal)
    rp = rep.create.render_product(cam, (1280, 960)); ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    for _ in range(6): rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
    img = ann.get_data(); ann.detach(); out = os.path.join(OUT, f"{tag}.png")
    if img is not None and getattr(img, "size", 0) > 0:
        Image.fromarray(np.asarray(img)[..., :3]).save(out); LOG[-1].setdefault("shots", []).append({"tag": tag, "target": np.round(tgt, 4).tolist(), "cam": np.round(pos, 4).tolist()})
        json.dump(LOG, open(os.path.join(OUT, "closeup_poses.json"), "w"), indent=2); print(f"[shot] {tag} -> {out}", flush=True)
    else: print(f"[shot] ❌ {tag}", flush=True)
    try: rp.destroy()
    except Exception: pass
for pname, pose in POSES.items():
    for oname, ov in (("closed", 0.0), ("open", OPEN)):
        set_pose(pose, ov)
        shot(f"{pname}_{oname}_mouth", (0.35, -0.45, 0.82) if pname == "home" else (0.55, -0.6, -0.6), focal=90.0)
        shot(f"{pname}_{oname}_side", (0.25, 0.9, 0.35), focal=90.0)
        if pname == "scoop": shot(f"{pname}_{oname}_front", (0.95, 0.15, -0.3), focal=90.0)
print("[done]", flush=True)
import threading; threading.Timer(60.0, lambda: os._exit(0)).start()   # close() 무한 대기 워치독 (D477)
sim_app.close()
