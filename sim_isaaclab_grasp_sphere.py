#!/usr/bin/env python3
"""Isaac Lab — g18 그랩으로 단순 구(⌀30 mm) 파지 시퀀스 + 카메라 센서 영상 (D477 후속, 사용자 요청).

무엇을 하나
    로봇(고정 베이스) + g18 그랩 + 바닥 + 구 1개(r 15 mm, 10 g). 키프레임(관절 공간, 오프라인 FK/IK):
    HOME → 접근(입 아래, 구 위) → 셸 개방 → 하강(립이 바닥 2 mm 위) → 셸 폐합(구를 클램셸 안에) → 들어올림 → 유지.
    Isaac Lab `Camera` 센서로 20 fps 프레임 저장 → ffmpeg mp4. 구 높이·셸 각·grab_base 위치를 JSON 으로 기록.
성공 기준 (파일로 판정): 마지막 1 초 구 z ≥ 0.08 m 이고 grab_base 와 xy 거리 ≤ 0.03 m (= 클램셸 안에 들려 있음).
⚠️ 입자 물리 없음. 드라이브 게인은 임의(스모크와 동일 300/30). D447/D477: 결과 JSON 은 종료 전에 쓰고 os._exit 워치독.
사용:  --solve-only                              (FK 검증 + 키프레임 출력, Isaac 불필요)
       OMNI_KIT_ACCEPT_EULA=YES python -u sim_isaaclab_grasp_sphere.py --headless --enable_cameras --out <dir>
"""
import argparse, os, sys, json, math, time

# ── 0. URDF 기하 (벤더 roarm_m3.urdf + 그랩 부착, D476) — FK 를 numpy 로 ──────────────────
import numpy as np


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); return Rz @ Ry @ Rx


def _T(xyz, rpy):
    T = np.eye(4); T[:3, :3] = _rpy(*rpy); T[:3, 3] = xyz; return T


def _Rz(q):
    T = np.eye(4); c, s = math.cos(q), math.sin(q); T[:2, :2] = [[c, -s], [s, c]]; return T


CHAIN = [  # (xyz, rpy) then Rz(q_i)  — URDF 원문 그대로
    ((0, 0, 0.0701), (0, 0, 0)),                    # world_to_base_link (fixed) — q 없음
    ((0, 0, 0), (0, 0, 0)),                          # base_link_to_link1  q0 (yaw)
    ((0, 0, 0.051959), (-1.5708, -1.5708, 0)),       # link1_to_link2      q1
    ((0.236815, 0.030002, 0), (0, 0, 1.5708)),       # link2_to_link3      q2
    ((0, -0.144586, 0), (0, 0, 0)),                  # link3_to_link4      q3
    ((0.015147, -0.053653, 0), (1.5708, 1.5708, 0)), # link4_to_link5      q4 (roll)
]
ATTACH = ((-0.01354, -0.000745, 0.12518), (-1.570796, 0.0, 0.0))   # link5 -> grab_base (피벗선 중심)
LIP_LOCAL = np.array([0.0, -0.03606, 0.0])          # 닫힘 립 접점 (그랩 로컬, 립 깊이 36.06 mm)
LIMITS = [(-3.1416, 3.1416), (-1.5708, 1.5708), (-1.0, 2.95), (-1.92, 1.92), (-3.1416, 3.1416)]


def fk(q):
    """q = 5 팔 관절 → (grab_base 위치, grab_base 회전행렬, 립 위치, 입 방향(로컬 -Y 의 월드 벡터))"""
    T = _T(*CHAIN[0])
    for (xyz, rpy), qi in zip(CHAIN[1:], q):
        T = T @ _T(xyz, rpy) @ _Rz(qi)
    T = T @ _T(*ATTACH)
    R = T[:3, :3]; p = T[:3, 3]
    return p, R, p + R @ LIP_LOCAL, R @ np.array([0, -1.0, 0])


def solve_pose(target_pivot, tol=0.002):
    """입이 아래(-Z)를 보도록 q1+q2+q3 = π 를 걸고 (q1,q2) 격자 → 최근접 → 국소 미세화."""
    best = None
    for q1 in np.linspace(*LIMITS[1], 315):
        for q2 in np.linspace(*LIMITS[2], 395):
            q3 = math.pi - q1 - q2
            if not (LIMITS[3][0] <= q3 <= LIMITS[3][1]):
                continue
            p, R, lip, d = fk([0.0, q1, q2, q3, 0.0])
            e = np.linalg.norm(p - target_pivot)
            if best is None or e < best[0]:
                best = (e, q1, q2, q3)
    e, q1, q2, q3 = best
    for _ in range(200):                                    # 좌표 하강 미세화
        improved = False
        for i, step in ((0, 0.002), (1, 0.002)):
            for sgn in (+1, -1):
                qq = [q1, q2]; qq[i] += sgn * step
                q3n = math.pi - qq[0] - qq[1]
                if not (LIMITS[1][0] <= qq[0] <= LIMITS[1][1] and LIMITS[2][0] <= qq[1] <= LIMITS[2][1] and LIMITS[3][0] <= q3n <= LIMITS[3][1]):
                    continue
                p, _, _, _ = fk([0.0, qq[0], qq[1], q3n, 0.0]); en = np.linalg.norm(p - target_pivot)
                if en < e:
                    e, q1, q2, q3 = en, qq[0], qq[1], q3n; improved = True
        if not improved:
            break
    return [0.0, q1, q2, q3, 0.0], e


SHELL_SWEEP_BOTTOM = 0.03980   # 셸이 닫히는 도중 피벗선 아래 최저점(22° 에서 −39.80 mm, 설계 조각 실측). 립(−36.06)보다 3.7 mm 아래.
SAG_MARGIN = 0.002             # 2차 런 실측: 하강 피벗 계획 0.0381 → 실제 0.0362 (팔 처짐 1.9 mm)


def keyframes(x_t, y_t, r_sphere, bottom_clear=0.002, h_pre=0.10, h_lift=0.18):
    """피벗선 중심 목표 높이: 하강 = 셸 스윕 최저점이 바닥 bottom_clear 위 (+ 처짐 여유), 접근/들어올림 = 고정 높이.
    🔴 2차 런(09-03): 립 기준(0.0381)으로 내렸더니 닫히는 도중 배가 바닥에 3.6 mm 박혀 셸이 안 닫히다 들어올릴 때 닫혔다."""
    out = {}
    for name, h in (("high", 0.22), ("pre", h_pre), ("down", SHELL_SWEEP_BOTTOM + bottom_clear + SAG_MARGIN), ("lift", h_lift)):
        q, e = solve_pose(np.array([x_t, y_t, h]))
        p, R, lip, d = fk(q)
        out[name] = {"q": [round(v, 5) for v in q], "err_mm": round(1000 * e, 2), "pivot": np.round(p, 4).tolist(),
                     "lip": np.round(lip, 4).tolist(), "mouth_dir": np.round(d, 3).tolist()}
    return out


X_T, Y_T, R_S = 0.25, 0.0135, 0.015

if "--solve-only" in sys.argv:
    p, R, lip, d = fk([0.0, 0.39, 1.39, 1.34, 0.0])
    print("FK check scoop pose: grab_base", np.round(p, 4).tolist(), "(closeup 실측 [0.2487, 0.0135, 0.1206])", "mouth_dir", np.round(d, 3).tolist())
    kf = keyframes(X_T, Y_T, R_S)
    print(json.dumps(kf, indent=1)); sys.exit(0)

# ── 1. Isaac Lab ─────────────────────────────────────────────────────────────
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, default="local_assets/roarm_m3/usd/roarm_m3_with_grab.usd")
parser.add_argument("--out", type=str, default="claudedocs/runtime_logs/grab_track/g18_nut_trap/isaaclab_grasp_sphere")
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

USD = os.path.abspath(args.usd); OUT = os.path.abspath(args.out); FR = os.path.join(OUT, "frames")
os.makedirs(FR, exist_ok=True)
KF = keyframes(X_T, Y_T, R_S)
json.dump({"keyframes": KF, "sphere": {"x": X_T, "y": Y_T, "z": R_S, "r": R_S, "mass_kg": 0.01}}, open(os.path.join(OUT, "keyframes.json"), "w"), indent=1)
res = {"ok": False, "usd": USD, "keyframes": KF}


def look_at_quat(pos, tgt):
    f = np.asarray(tgt, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    y = np.cross([0, 0, 1.0], f); y /= np.linalg.norm(y); z = np.cross(f, y)
    R = np.stack([f, y, z], 1)
    from scipy.spatial.transform import Rotation
    x, y_, z_, w = Rotation.from_matrix(R).as_quat(); return (float(w), float(x), float(y_), float(z_))


CAM_POS, CAM_TGT = (0.62, -0.40, 0.22), (X_T, Y_T, 0.07)

ROBOT = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD,
                               articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False),
                               rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}),
    # 🔴 1차 실패 원인(09-03): 액추에이터가 USD maxForce(URDF effort 1.9 N·m = 7.4 V 값)를 상한으로 써서 어깨가
    #    중력에 포화·처짐(목표 z 0.100 → 실제 0.042) → 처진 그랩이 접근 중 구를 밀어냄. 데모용으로 팔 상한을 넉넉히 둔다
    #    (실물 ST3235 12 V = 2.94 N·m; 아래 8.0 은 **비물리 데모값**, 토크 실측·전압 확정 전까지 인용 금지).
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5", "link5_to_gripper_link"],
                                          stiffness=400.0, damping=40.0, effort_limit_sim=8.0),
               "grab": ImplicitActuatorCfg(joint_names_expr=["grab_shell_.*_joint"], stiffness=300.0, damping=30.0, effort_limit_sim=2.94)})
SPHERE = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Sphere",
    spawn=sim_utils.SphereCfg(radius=R_S,
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                              mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                              collision_props=sim_utils.CollisionPropertiesCfg(),
                              physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15))),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(X_T, Y_T, R_S)))
CAM = CameraCfg(prim_path="/World/Cam", update_period=0.0, height=600, width=960, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=28.0, clipping_range=(0.05, 5.0)),
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
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0))
    sim.reset()
    robot: Articulation = scene["robot"]; sphere: RigidObject = scene["sphere"]; cam: Camera = scene["cam"]
    jn = list(robot.joint_names); bn = list(robot.body_names)
    ARM = [jn.index(n) for n in ("base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5")]
    iL, iR = jn.index("grab_shell_L_joint"), jn.index("grab_shell_R_joint"); iGB = bn.index("grab_base")
    dt = sim.get_physics_dt(); OPEN = 0.77667
    # 시퀀스 (초): 구간별 (팔 키프레임 from→to, 셸 from→to)
    home = [0.0] * 5
    seq = [("home→high", 2.0, home, KF["high"]["q"], 0.0, 0.0), ("high→pre", 1.5, KF["high"]["q"], KF["pre"]["q"], 0.0, 0.0),
           ("open", 1.0, KF["pre"]["q"], KF["pre"]["q"], 0.0, OPEN),
           ("pre→down", 2.0, KF["pre"]["q"], KF["down"]["q"], OPEN, OPEN), ("close", 2.5, KF["down"]["q"], KF["down"]["q"], OPEN, 0.0),
           ("hold", 1.0, KF["down"]["q"], KF["down"]["q"], 0.0, 0.0), ("lift", 2.0, KF["down"]["q"], KF["lift"]["q"], 0.0, 0.0),
           ("hold2", 1.5, KF["lift"]["q"], KF["lift"]["q"], 0.0, 0.0)]
    tgt = torch.zeros(1, robot.num_joints, device=sim.device)
    log, frame_i, t = [], 0, 0.0
    every = max(1, int(round(1.0 / (args.fps * dt))))
    step_i = 0
    for name, dur, qa, qb, sa, sb in seq:
        n = int(round(dur / dt))
        for k in range(n):
            a = 0.5 * (1 - math.cos(math.pi * (k + 1) / n))        # 부드러운 보간
            q = [(1 - a) * x + a * y for x, y in zip(qa, qb)]
            for j, idx in enumerate(ARM):
                tgt[0, idx] = q[j]
            s = (1 - a) * sa + a * sb; tgt[0, iL] = s; tgt[0, iR] = s
            robot.set_joint_position_target(tgt); scene.write_data_to_sim()
            sim.step(); scene.update(dt); t += dt; step_i += 1
            if step_i % every == 0:
                rgb = cam.data.output["rgb"][0]
                img = rgb.detach().cpu().numpy()[..., :3].astype(np.uint8)
                Image.fromarray(img).save(os.path.join(FR, f"f_{frame_i:04d}.png")); frame_i += 1
                qp = robot.data.joint_pos[0]; sp = sphere.data.root_pos_w[0]; gb = robot.data.body_pos_w[0, iGB]
                log.append({"t": round(t, 3), "phase": name, "shell_L": round(float(qp[iL]), 4), "shell_R": round(float(qp[iR]), 4),
                            "shell_tgt": round(s, 4), "sphere": [round(float(v), 4) for v in sp], "grab_base": [round(float(v), 4) for v in gb],
                            "arm": [round(float(qp[i]), 4) for i in ARM]})
    last = [r for r in log if r["t"] > t - 1.0]
    z_end = float(np.mean([r["sphere"][2] for r in last])); dxy = float(np.mean([math.hypot(r["sphere"][0] - r["grab_base"][0], r["sphere"][1] - r["grab_base"][1]) for r in last]))
    res.update({"frames": frame_i, "fps": args.fps, "sim_seconds": round(t, 2), "sphere_z_end_m": round(z_end, 4), "sphere_grab_xy_dist_end_m": round(dxy, 4),
                "sphere_z_min_m": round(min(r["sphere"][2] for r in log), 4), "sphere_z_max_m": round(max(r["sphere"][2] for r in log), 4),
                "shell_final": [log[-1]["shell_L"], log[-1]["shell_R"]], "finite": bool(all(np.isfinite(r["sphere"]).all() for r in log)),
                "joint_names": jn, "body_names": bn, "camera": {"pos": CAM_POS, "target": CAM_TGT}})
    res["ok"] = bool(res["finite"] and z_end >= 0.08 and dxy <= 0.03)
    json.dump(res, open(os.path.join(OUT, "grasp_result.json"), "w"), indent=1)
    json.dump(log, open(os.path.join(OUT, "grasp_log.json"), "w"))
    print("[grasp]", json.dumps({k: v for k, v in res.items() if k not in ("keyframes", "joint_names", "body_names", "usd")}), flush=True)


try:
    main()
except Exception:
    import traceback; res["error"] = traceback.format_exc(); json.dump(res, open(os.path.join(OUT, "grasp_result.json"), "w"), indent=1)
    print("[grasp] ERROR", res["error"][-800:], flush=True)
import threading
threading.Thread(target=lambda: (time.sleep(20), os._exit(0)), daemon=True).start()
simulation_app.close()
