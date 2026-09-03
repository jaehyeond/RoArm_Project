#!/usr/bin/env python3
"""합성 USD(로봇+g18 그랩) 근접 RTX 렌더 — **그랩 위치 자동 조준 + 장초점** (D474 ⑤ 교훈 반영).

D474: Isaac 카메라는 ~0.85 m 안쪽으로 다가가면 프레임을 놓친다 → 거리 1.0 m 를 유지하고 focal_length 로 확대한다.
조준점 = grab_base 프림의 월드 위치(자세마다 다시 읽음). 순정 가동 조는 **보이게 둔다**(실물 = 순정 조 + 그랩).
자세 2종(HOME 입 위 / 스쿱 입 아래) × 개폐 2종(0 / 0.77667) × 시점 2종(입 정면·측면) = 8장.
⚠️ D447: close() 는 예외 삼킴 → 호출부가 PNG 개수로 판정.
사용: OMNI_KIT_ACCEPT_EULA=YES python sim_render_grab_closeup.py <usd> <out_dir>
"""
import sys, os, math, glob, shutil, json
from isaacsim import SimulationApp

USD = os.path.abspath(sys.argv[1]); OUT = os.path.abspath(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
sim_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 960})

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
# 바닥면 없음: 스쿱 자세는 입이 아래(-Z)라 바닥이 시야를 막는다. 중력 0 이라 물리적으로도 불필요.
UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1200.0)
key = UsdLux.DistantLight.Define(stage, "/World/KeyLight"); key.CreateIntensityAttr(2500.0)
UsdGeom.XformCommonAPI(key).SetRotate(Gf.Vec3f(-40, 30, 0))
world.get_physics_context().set_gravity(0.0)
art = SingleArticulation(prim_path="/World/roarm", name="roarm")
world.reset(); art.initialize()
for _ in range(5): world.step(render=False)          # 물리 뷰 생성 워밍업 ("Simulation View is not created yet" 회피)
names = list(art.dof_names); print("DOF:", names)

POSE_LOG = []
POSES = {"home": {},
         "scoop": {"link1_to_link2": 0.39, "link2_to_link3": 1.39, "link3_to_link4": 1.34}}   # D474 ⑤ 입 아래(-Z) 자세, 벤더 URDF 관절명


def grab_world_pos():
    p = stage.GetPrimAtPath("/World/roarm/grab_base")
    m = omni.usd.get_world_transform_matrix(p)
    t = m.ExtractTranslation(); return np.array([t[0], t[1], t[2]])


def set_pose(pose, open_val):
    q = np.zeros(art.num_dof)
    for nm, v in pose.items():
        for i, dn in enumerate(names):
            if nm in dn: q[i] = v
    for nm in ("grab_shell_L_joint", "grab_shell_R_joint"):
        if nm in names: q[names.index(nm)] = open_val
    if art.get_joint_positions() is None:        # Replicator 캡처 뒤 물리 뷰가 사라진다 → 타임라인 재생 + 재초기화
        world.play()
        for _ in range(3): world.step(render=False)
        art.initialize()
    art.set_joint_positions(q)
    try: art.set_joint_position_targets(q)
    except Exception: pass
    for _ in range(30): world.step(render=True)
    got = art.get_joint_positions()
    if got is None:
        print("[pose] ❌ readback None — 물리 뷰 없음", flush=True); return
    sh = {nm: round(float(got[names.index(nm)]), 4) for nm in ("grab_shell_L_joint", "grab_shell_R_joint") if nm in names}
    rec = {"target_open": open_val, "shell_readback": sh, "arm_readback": [round(float(v), 4) for v in got[:5]],
           "ok": all(abs(v - open_val) <= 0.05 for v in sh.values())}
    POSE_LOG.append(rec)
    json.dump(POSE_LOG, open(os.path.join(OUT, "closeup_poses.json"), "w"), indent=2)   # stdout 버퍼링 유실 대비(09-03)
    print(f"[pose] {rec}", flush=True)


def shot(tag, cam_dir, dist=1.0, focal=70.0):
    tgt = grab_world_pos()
    d = np.asarray(cam_dir, float); d /= np.linalg.norm(d)
    pos = tgt + d * dist
    cam = rep.create.camera(position=tuple(pos.tolist()), look_at=tuple(tgt.tolist()), focal_length=focal)
    rp = rep.create.render_product(cam, (1280, 960))
    # annotator 로 픽셀 직접 (BasicWriter 는 폭주·타임라인 정지 사고, 09-03). 타임라인은 멈추지 않는다.
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    for _ in range(6): rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
    img = ann.get_data(); ann.detach()
    out = os.path.join(OUT, f"{tag}.png")
    if img is not None and getattr(img, "size", 0) > 0:
        Image.fromarray(np.asarray(img)[..., :3]).save(out)
        POSE_LOG[-1].setdefault("shots", []).append({"tag": tag, "target": np.round(tgt, 4).tolist(), "cam": np.round(pos, 4).tolist(), "focal_mm": focal})
        json.dump(POSE_LOG, open(os.path.join(OUT, "closeup_poses.json"), "w"), indent=2)
        print(f"[shot] {tag} target={np.round(tgt,3).tolist()} cam={np.round(pos,3).tolist()} -> {out}", flush=True)
    else:
        print(f"[shot] ❌ {tag} 픽셀 없음")
    try: rp.destroy()
    except Exception: pass


for pname, pose in POSES.items():
    for oname, ov in (("closed", 0.0), ("open", 0.77667)):
        set_pose(pose, ov)
        t = grab_world_pos()
        # 시점: 입 정면(HOME 은 입이 +Z 위 → 위에서, 스쿱은 입이 -Z 아래 → 비스듬히 아래에서 올려봄) + 측면(+Y 쪽 비스듬)
        if pname == "home":
            shot(f"{pname}_{oname}_mouth", (0.35, -0.45, 0.82))
        else:
            shot(f"{pname}_{oname}_mouth", (0.55, -0.6, -0.6))
        shot(f"{pname}_{oname}_side", (0.25, 0.9, 0.35))
print("[done]")
sim_app.close()
