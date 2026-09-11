#!/usr/bin/env python3
"""Isaac Lab — S1 그랩(고정 반쪽 + 서보축 직결 문)으로 구(⌀30 mm) 파지 시퀀스 + 카메라 영상 (D480 후속).

D478/D479 스크립트의 S1 판. 차이:
  · 관절 = link5_to_gripper_link 하나가 문(가동 반쪽 보울)을 직접 돌린다. 표·mimic·셸 관절 없음 → 실물과 같은 구조.
  · 문 서보 토크 상한 = **1.96 N·m** (ST3215-HS 20 kg·cm @12 V, 실물 라벨 09-03). 팔 상한 8.0 은 D478 과 같은 **비물리 데모값**(인용 금지).
  · 보울 중심 = link5 (X 8.1, Y 0, Z 145) mm, 닫힘 립 = 중심 + Z 21.6. 문은 link5 +X 로 열림(29.3° = 0.511 rad 에서 입 58 mm).
  · 하강 높이: 문 립은 닫힘에서 가장 낮고 열리면 올라가므로(반경 115 호) 스윕 최저점 = 립. 중심 높이 = 21.6 + 여유 2 + 처짐 2 mm.
성공 = 마지막 1 초 구 z ≥ 0.08 m 이고 보울 중심과 xy 거리 ≤ 0.03 m.
사용:  --solve-only    /   OMNI_KIT_ACCEPT_EULA=YES python -u sim_isaaclab_grasp_sphere_s1.py --headless --enable_cameras --out <dir>
"""
import argparse, os, sys, json, math, time
import numpy as np


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); return Rz @ Ry @ Rx
def _T(xyz, rpy):
    T = np.eye(4); T[:3, :3] = _rpy(*rpy); T[:3, 3] = xyz; return T
def _Rz(q):
    T = np.eye(4); c, s = math.cos(q), math.sin(q); T[:2, :2] = [[c, -s], [s, c]]; return T

CHAIN = [((0, 0, 0.0701), (0, 0, 0)), ((0, 0, 0), (0, 0, 0)), ((0, 0, 0.051959), (-1.5708, -1.5708, 0)),
         ((0.236815, 0.030002, 0), (0, 0, 1.5708)), ((0, -0.144586, 0), (0, 0, 0)), ((0.015147, -0.053653, 0), (1.5708, 1.5708, 0))]
BOWL_LOCAL = np.array([0.0081, 0.0, 0.145])        # link5 프레임 보울 중심 (m)
R_OUT = 0.0216                                       # 보울 바깥 반경 (립 = 중심 + Z R_OUT)
LIMITS = [(-3.1416, 3.1416), (-1.5708, 1.5708), (-1.0, 2.95), (-1.92, 1.92), (-3.1416, 3.1416)]
DOOR_OPEN = 0.511                                    # rad (29.3°, 입 58 mm)


def fk(q):
    """q(5) → (보울 중심, link5 회전, 닫힘 립, 입 방향 = link5 +Z, 문 열림 방향 = link5 +X)"""
    T = _T(*CHAIN[0])
    for (xyz, rpy), qi in zip(CHAIN[1:], q):
        T = T @ _T(xyz, rpy) @ _Rz(qi)
    R = T[:3, :3]; p5 = T[:3, 3]; c = p5 + R @ BOWL_LOCAL
    return c, R, c + R @ np.array([0, 0, R_OUT]), R @ np.array([0, 0, 1.0]), R @ np.array([1.0, 0, 0])


def solve_pose(target, tol=0.002):
    best = None
    for q1 in np.linspace(*LIMITS[1], 315):
        for q2 in np.linspace(*LIMITS[2], 395):
            q3 = math.pi - q1 - q2
            if not (LIMITS[3][0] <= q3 <= LIMITS[3][1]): continue
            c = fk([0.0, q1, q2, q3, 0.0])[0]; e = np.linalg.norm(c - target)
            if best is None or e < best[0]: best = (e, q1, q2, q3)
    e, q1, q2, q3 = best
    for _ in range(200):
        improved = False
        for i, step in ((0, 0.002), (1, 0.002)):
            for sgn in (+1, -1):
                qq = [q1, q2]; qq[i] += sgn * step; q3n = math.pi - qq[0] - qq[1]
                if not (LIMITS[1][0] <= qq[0] <= LIMITS[1][1] and LIMITS[2][0] <= qq[1] <= LIMITS[2][1] and LIMITS[3][0] <= q3n <= LIMITS[3][1]): continue
                en = np.linalg.norm(fk([0.0, qq[0], qq[1], q3n, 0.0])[0] - target)
                if en < e: e, q1, q2, q3 = en, qq[0], qq[1], q3n; improved = True
        if not improved: break
    return [0.0, q1, q2, q3, 0.0], e


BOTTOM_CLEAR, SAG_MARGIN = 0.002, 0.005   # 1차 런 처짐 3.9 mm(계획 25.8 → 실측 21.9)
def keyframes(x_t, y_t, h_pre=0.10, h_lift=0.18):
    out = {}
    for name, h in (("high", 0.20), ("pre", h_pre), ("down", R_OUT + BOTTOM_CLEAR + SAG_MARGIN), ("lift", h_lift)):
        q, e = solve_pose(np.array([x_t, y_t, h])); c, R, lip, d, ox = fk(q)
        out[name] = {"q": [round(v, 5) for v in q], "err_mm": round(1000 * e, 2), "bowl": np.round(c, 4).tolist(), "lip": np.round(lip, 4).tolist(),
                     "mouth_dir": np.round(d, 3).tolist(), "door_open_dir": np.round(ox, 3).tolist()}
    return out


X_T, Y_T, R_S = 0.25, -0.0081, 0.015     # y = 요 0 에서 보울 중심의 자연 y (link5 X 8.1 → 월드 −Y)
KF = keyframes(X_T, Y_T)
_ox = np.array(KF["down"]["door_open_dir"]); _oxy = _ox[:2] / (np.linalg.norm(_ox[:2]) + 1e-9)
SPH_XY = (X_T + 0.022 * _oxy[0], Y_T + 0.022 * _oxy[1])      # 구를 문 쪽으로 22 mm: 1차 런에서 고정 립(파팅면)이 하강 중 구 위를 눌러 튕겨냄(1.2 m). 고정 립과 7 mm 여유

if "--solve-only" in sys.argv:
    c, R, lip, d, ox = fk([0.0, 0.39, 1.39, 1.34, 0.0]); print("FK scoop pose: bowl", np.round(c, 4).tolist(), "mouth_dir", np.round(d, 3).tolist(), "door_open_dir", np.round(ox, 3).tolist())
    print(json.dumps(KF, indent=1)); print("sphere xy", SPH_XY); sys.exit(0)

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, default="local_assets/roarm_m3/usd_s1/roarm_m3_s1.usd")
parser.add_argument("--out", type=str, default="claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v0/isaaclab_grasp_sphere")
parser.add_argument("--fps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args); simulation_app = app_launcher.app
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from PIL import Image
from scipy.spatial.transform import Rotation

USD = os.path.abspath(args.usd); OUT = os.path.abspath(args.out); FR = os.path.join(OUT, "frames"); os.makedirs(FR, exist_ok=True)
json.dump({"keyframes": KF, "sphere": {"x": SPH_XY[0], "y": SPH_XY[1], "z": R_S, "r": R_S, "mass_kg": 0.01}}, open(os.path.join(OUT, "keyframes.json"), "w"), indent=1)
res = {"ok": False, "usd": USD, "keyframes": KF, "design": "S1 (D480)", "door_effort_limit_Nm": 1.96}

def look_at_quat(pos, tgt):
    f = np.asarray(tgt, float) - np.asarray(pos, float); f /= np.linalg.norm(f); y = np.cross([0, 0, 1.0], f); y /= np.linalg.norm(y); z = np.cross(f, y)
    x, y_, z_, w = Rotation.from_matrix(np.stack([f, y, z], 1)).as_quat(); return (float(w), float(x), float(y_), float(z_))
CAM_POS, CAM_TGT = (0.62, -0.40, 0.22), (X_T, Y_T, 0.07)
ROBOT = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD, articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False), rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5"], stiffness=400.0, damping=40.0, effort_limit_sim=8.0),   # 비물리 데모값 (D478)
               "door": ImplicitActuatorCfg(joint_names_expr=["link5_to_gripper_link"], stiffness=300.0, damping=30.0, effort_limit_sim=1.96)})                  # ST3215-HS 실물 상한
SPHERE = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Sphere",
    spawn=sim_utils.SphereCfg(radius=R_S, rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=0.01), collision_props=sim_utils.CollisionPropertiesCfg(),
                              physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15))),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(SPH_XY[0], SPH_XY[1], R_S)))
CAM = CameraCfg(prim_path="/World/Cam", update_period=0.0, height=600, width=960, data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=28.0, clipping_range=(0.05, 5.0)),
                offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=look_at_quat(CAM_POS, CAM_TGT), convention="world"))

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9)))
    dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
    key = AssetBaseCfg(prim_path="/World/key", spawn=sim_utils.DistantLightCfg(intensity=2500.0), init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.3827, 0.0, 0.0)))
    robot: ArticulationCfg = ROBOT
    sphere: RigidObjectCfg = SPHERE
    cam: CameraCfg = CAM

def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device)); scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0)); sim.reset()
    robot: Articulation = scene["robot"]; sphere: RigidObject = scene["sphere"]; cam: Camera = scene["cam"]
    jn = list(robot.joint_names); bn = list(robot.body_names)
    ARM = [jn.index(n) for n in ("base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5")]
    iS = jn.index("link5_to_gripper_link"); i5 = bn.index("link5"); iJ = bn.index("gripper_link")
    dt = sim.get_physics_dt(); home = [0.0] * 5
    seq = [("home→high", 2.0, home, KF["high"]["q"], 0.0, 0.0), ("high→pre", 1.5, KF["high"]["q"], KF["pre"]["q"], 0.0, 0.0), ("open", 1.0, KF["pre"]["q"], KF["pre"]["q"], 0.0, DOOR_OPEN),
           ("pre→down", 2.0, KF["pre"]["q"], KF["down"]["q"], DOOR_OPEN, DOOR_OPEN), ("close", 2.5, KF["down"]["q"], KF["down"]["q"], DOOR_OPEN, 0.0),
           ("hold", 1.0, KF["down"]["q"], KF["down"]["q"], 0.0, 0.0), ("lift", 2.0, KF["down"]["q"], KF["lift"]["q"], 0.0, 0.0), ("hold2", 1.5, KF["lift"]["q"], KF["lift"]["q"], 0.0, 0.0)]
    tgt = torch.zeros(1, robot.num_joints, device=sim.device); log, frame_i, t, step_i = [], 0, 0.0, 0
    every = max(1, int(round(1.0 / (args.fps * dt))))
    def bowl_w():
        p = robot.data.body_pos_w[0, i5].cpu().numpy(); q = robot.data.body_quat_w[0, i5].cpu().numpy()   # (w,x,y,z)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); return p + R @ BOWL_LOCAL
    for name, dur, qa, qb, sa, sb in seq:
        n = int(round(dur / dt))
        for k in range(n):
            a = 0.5 * (1 - math.cos(math.pi * (k + 1) / n)); q = [(1 - a) * x + a * y for x, y in zip(qa, qb)]
            for j, idx in enumerate(ARM): tgt[0, idx] = q[j]
            s = (1 - a) * sa + a * sb; tgt[0, iS] = s
            robot.set_joint_position_target(tgt); scene.write_data_to_sim(); sim.step(); scene.update(dt); t += dt; step_i += 1
            if step_i % every == 0:
                img = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8); Image.fromarray(img).save(os.path.join(FR, f"f_{frame_i:04d}.png")); frame_i += 1
                qp = robot.data.joint_pos[0]; sp = sphere.data.root_pos_w[0].cpu().numpy(); bw = bowl_w()
                log.append({"t": round(t, 3), "phase": name, "door_tgt": round(s, 4), "door": round(float(qp[iS]), 4), "sphere": [round(float(v), 4) for v in sp],
                            "bowl": [round(float(v), 4) for v in bw], "arm": [round(float(qp[i]), 4) for i in ARM], "door_torque": round(float(robot.data.applied_torque[0, iS]), 3)})
    last = [r for r in log if r["t"] > t - 1.0]
    z_end = float(np.mean([r["sphere"][2] for r in last])); dxy = float(np.mean([math.hypot(r["sphere"][0] - r["bowl"][0], r["sphere"][1] - r["bowl"][1]) for r in last]))
    down = [r for r in log if r["phase"] == "pre→down"]
    res.update({"frames": frame_i, "fps": args.fps, "sim_seconds": round(t, 2), "sphere_z_end_m": round(z_end, 4), "sphere_bowl_xy_dist_end_m": round(dxy, 4),
                "sphere_z_min_m": round(min(r["sphere"][2] for r in log), 4), "sphere_z_max_m": round(max(r["sphere"][2] for r in log), 4),
                "door_final_rad": log[-1]["door"], "door_max_rad": max(r["door"] for r in log), "bowl_z_at_down_end": down[-1]["bowl"][2] if down else None,
                "door_torque_abs_max_Nm": round(max(abs(r["door_torque"]) for r in log), 3), "finite": bool(all(np.isfinite(r["sphere"]).all() for r in log)),
                "joint_names": jn, "body_names": bn, "camera": {"pos": CAM_POS, "target": CAM_TGT}, "sphere_xy": SPH_XY})
    res["ok"] = bool(res["finite"] and z_end >= 0.08 and dxy <= 0.03)
    json.dump(res, open(os.path.join(OUT, "grasp_result.json"), "w"), indent=1); json.dump(log, open(os.path.join(OUT, "grasp_log.json"), "w"))
    print("[grasp]", json.dumps({k: v for k, v in res.items() if k not in ("keyframes", "joint_names", "body_names", "usd")}), flush=True)

try:
    main()
except Exception:
    import traceback; res["error"] = traceback.format_exc(); json.dump(res, open(os.path.join(OUT, "grasp_result.json"), "w"), indent=1); print("[grasp] ERROR", res["error"][-800:], flush=True)
import threading
threading.Thread(target=lambda: (time.sleep(20), os._exit(0)), daemon=True).start()
simulation_app.close()
