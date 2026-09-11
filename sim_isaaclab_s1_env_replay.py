#!/usr/bin/env python3
"""Isaac Lab — 실물 환경 치수(상자·받침·펠릿면·놓기 자리)를 씬에 넣고 09-07 실물 5회 사이클(`hw_s1_manual.py cycle 5`) 관절 명령을 재생. 펠릿 없음 (W2, `s1_v1_sim/w2_env_replay`).

목적: (a) 팔·그랩 vs 상자 간섭 (b) 시뮬 립 vs 실물 FK 립 (c) 영상.
규약(hw_s1_scoop_probe.chain / roarm_kinematics): SDK deg == URDF rad, 부호·오프셋 없음. 립 = link5 (8.1, 0, 166.6) mm.
  jsonl `lip` = lip_world(read) = FK(read) 어깨축 기준 → 세계 z = lip_z + SHOULDER_ABOVE_PLATE + PLATE_Z. 문 = read[5] servo_deg → rad 직접(0 = 닫힘).
환경(바닥 기준 m, 09-07 실측): 베이스판 윗면 0.38 / 상자 안쪽 0.31(x)×0.22(y), 중심 x 0.35, 윗단 0.385, 받침 0.16 / 펠릿면 0.26 / 놓기 립 (0.01, 0.34, 0.25).
가정(실측 아님): 상자 벽·바닥 두께 3 mm, 받침 footprint = 상자 외곽, 펠릿 슬래브·놓기 표식·받침대는 충돌 없음(시각 전용).
게이트(후처리 w2_gates.py 가 산출 JSON 을 읽어 판정): G1 door·grab_fixed·link5 vs 상자(벽4+바닥+받침) 최소거리 > 0 전 구간(표본점 부호거리 + PhysX 접촉력)
  / G2 goto 정착 립 sim vs jsonl 차 중앙값 ≤ 10 mm / G3 5 사이클·NaN 0·관절 편차 / G4 mp4·strip·놓기 프레임.
사용: OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 1500 ~/miniconda3/envs/isaaclab/bin/python -u sim_isaaclab_s1_env_replay.py --headless --enable_cameras \
       --usd local_assets/roarm_m3/usd_s1/roarm_m3_s1.usd --log <manual_*.jsonl> --out <dir> [--fps 10] [--speed 60] [--max-events N]
"""
import argparse, os, sys, json, math, time, hashlib
import numpy as np

# ── 실물 환경 (바닥 기준 m) ──
PLATE_Z = 0.38
SHOULDER_ABOVE_PLATE = 0.0701 + 0.05196     # = hw_s1_scoop_probe.SHOULDER_ABOVE_PLATE
LIP_L5 = np.array([0.0081, 0.0, 0.1666])    # = hw_s1_scoop_probe.LIP_L5[:3]
WRIST_MAX = 90.0
HOME = [0.0, 0.0, 90.0, 0.0, 0.0]
SETTLE_MAX, SETTLE_CHK, SETTLE_DQ = 2.0, 0.4, 0.3   # 정착 상한 s · 위치 안정 판정: 0.4 s 간격 두 번 연속 전 관절 |Δq| < 0.3° (D481 ④ 실물 규약, W2b 09-10 — 속도 기준은 실효 없었음 W2 §6 ⑦)
BOX_CX, HX, HY = 0.35, 0.155, 0.11          # 안쪽 반폭
BOX_TOP, BOX_BOT, PELLET_Z, WALL = 0.385, 0.16, 0.26, 0.003   # WALL = 가정
PLACE_LIP = (0.01, 0.34, 0.25)
MESH_DIR = "local_assets/roarm_m3/urdf/meshes"
GATE_BODIES = ("gripper_link", "grab_fixed", "link5")

# 상자 슬래브 (lo, hi) — 벽4 + 바닥 + 받침. 펠릿 슬래브는 충돌 없음 → 제외
SLABS = {
    "wall_xp": ((BOX_CX + HX, -HY - WALL, BOX_BOT), (BOX_CX + HX + WALL, HY + WALL, BOX_TOP)),
    "wall_xn": ((BOX_CX - HX - WALL, -HY - WALL, BOX_BOT), (BOX_CX - HX, HY + WALL, BOX_TOP)),
    "wall_yp": ((BOX_CX - HX, HY, BOX_BOT), (BOX_CX + HX, HY + WALL, BOX_TOP)),
    "wall_yn": ((BOX_CX - HX, -HY - WALL, BOX_BOT), (BOX_CX + HX, -HY, BOX_TOP)),
    "box_floor": ((BOX_CX - HX, -HY, BOX_BOT), (BOX_CX + HX, HY, BOX_BOT + WALL)),
    "support": ((BOX_CX - HX - WALL, -HY - WALL, 0.0), (BOX_CX + HX + WALL, HY + WALL, BOX_BOT)),
}
SLAB_LO = np.array([v[0] for v in SLABS.values()]); SLAB_HI = np.array([v[1] for v in SLABS.values()])
BOX_ALL_LO, BOX_ALL_HI = SLAB_LO.min(0), SLAB_HI.max(0)


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


def fk_lip_world(q5_deg):
    """SDK deg 5 → 립 세계 좌표(바닥 기준, 로봇 root = PLATE_Z)."""
    T = _T(*CHAIN[0])
    for (xyz, rpy), qd in zip(CHAIN[1:], q5_deg): T = T @ _T(xyz, rpy) @ _Rz(math.radians(qd))
    p = T[:3, 3] + T[:3, :3] @ LIP_L5; return np.array([p[0], p[1], p[2] + PLATE_Z])


def jsonl_lip_world(lip):
    return np.array([lip[0], lip[1], lip[2] + SHOULDER_ABOVE_PLATE + PLATE_Z])


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def build_schedule(events, speed_dps):
    """jsonl 이벤트 → 세그먼트(관절·문 목표, 블렌드 시간). torque/scoop_done/place_done 은 마커."""
    q = list(HOME); door = next((e["read"][5] for e in events if e["ev"] == "goto"), 0.0); seg = []; cyc = 1
    for i, e in enumerate(events):
        if e["ev"] == "goto":
            tgt = list(e["cmd"]); tgt[3] = max(-WRIST_MAX, min(WRIST_MAX, tgt[3]))
            dmax = max(abs(a - b) for a, b in zip(tgt, q))
            seg.append(dict(i=i, kind="goto", name=e["name"], cycle=cyc, q0=q, q1=tgt, d0=door, d1=float(e["read"][5]), dur=min(2.5, max(0.5, dmax / speed_dps)), settle=0.5))
            q, door = tgt, float(e["read"][5])
        elif e["ev"] == "door":
            seg.append(dict(i=i, kind="door", name=f"door{e['target']:.0f}", cycle=cyc, q0=q, q1=q, d0=door, d1=float(e["read"]), dur=0.5, settle=0.4))
            door = float(e["read"])
        else:
            seg.append(dict(i=i, kind=e["ev"], name=e["ev"], cycle=cyc, dur=0.0, settle=0.0))
            if e["ev"] == "place_done": cyc += 1
    return seg


def aabb_sdist(P, lo, hi):
    """점 (N,3) vs AABB: 밖 = 최소거리(+), 안 = −관통깊이."""
    d_out = np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=1)
    inside = np.all((P > lo) & (P < hi), axis=1)
    d_in = np.minimum(P - lo, hi - P).min(axis=1)
    return np.where(inside, -d_in, d_out)


parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="local_assets/roarm_m3/usd_s1/roarm_m3_s1.usd")
parser.add_argument("--log", default="claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/manual_20260907_164725.jsonl")
parser.add_argument("--out", default="claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w2_env_replay")
parser.add_argument("--fps", type=int, default=10); parser.add_argument("--speed", type=float, default=60.0, help="관절 블렌드 속도 deg/s (실물 타이밍 아님)")
parser.add_argument("--max-events", type=int, default=0, help="스모크: 앞 N 이벤트만"); parser.add_argument("--res", default="640x400")
parser.add_argument("--urdf", default=None, help="거리·부록용 URDF (기본: USD 가 usd_s1_v1 이면 roarm_m3_s1_v1.urdf, 아니면 roarm_m3_s1.urdf)")
parser.add_argument("--tcp-mass", type=float, default=None, help="hand_tcp 질량 덮어쓰기 kg (USD 는 무질량 링크에 임포터 기본 1.0 kg — 부록 변형용, 기본 = 건드리지 않음)")
if "--schedule-only" in sys.argv:
    sys.argv.remove("--schedule-only"); a, _ = parser.parse_known_args()
    ev = [json.loads(l) for l in open(a.log)]; S = build_schedule(ev, a.speed)
    print("segments", len(S), "sim s", round(sum(s["dur"] + s["settle"] for s in S), 1))
    err = max(np.linalg.norm(fk_lip_world(e["read"][:5]) - jsonl_lip_world(e["lip"])) for e in ev if e["ev"] == "goto")
    print("FK self-check max mm", round(1000 * err, 4)); sys.exit(0)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args); simulation_app = app_launcher.app
MESH_TAG = "s1_v1" if "usd_s1_v1" in args.usd else "s1"; URDF = args.urdf or f"local_assets/roarm_m3/urdf/roarm_m3_{MESH_TAG}.urdf"   # W2b: v1 USD 면 v1 시각 메시·URDF (문 뺨 2 mm 등 v1 형상)
DIST_BODIES = {"gripper_link": f"{MESH_TAG}_door.stl", "grab_fixed": f"{MESH_TAG}_fixed.stl", "link5": "link5.stl", "link4": "link4.stl", "link3": "link3.stl"}
import torch, trimesh
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

USD = os.path.abspath(args.usd); OUT = os.path.abspath(args.out); FR = os.path.join(OUT, "frames"); os.makedirs(FR, exist_ok=True)
W, H = (int(v) for v in args.res.split("x"))
EVENTS = [json.loads(l) for l in open(args.log)]
if args.max_events: EVENTS = EVENTS[: args.max_events]
SCHED = build_schedule(EVENTS, args.speed)
inputs = {p: sha16(p) for p in [args.log, args.usd, URDF, "hw_s1_scoop_probe.py"] + [os.path.join(MESH_DIR, m) for m in DIST_BODIES.values()]}
res = {"ok": False, "usd": USD, "log": os.path.abspath(args.log), "inputs_sha256_16": inputs, "env": dict(plate_z=PLATE_Z, shoulder_above_plate=SHOULDER_ABOVE_PLATE, box_cx=BOX_CX, box_inner=(2 * HX, 2 * HY), box_top=BOX_TOP, box_bot=BOX_BOT, pellet_z=PELLET_Z, wall_assumed=WALL, place_lip=PLACE_LIP, slabs={k: [list(v[0]), list(v[1])] for k, v in SLABS.items()}),
       "actuators": {"arm": dict(stiffness=800.0, damping=40.0, effort_limit_sim=8.0, note="비물리 데모값(D478 계열)"), "door": dict(stiffness=300.0, damping=30.0, effort_limit_sim=1.96)},
       "settle_rule": f"블렌드 후 최소 0.5 s(goto)/0.4 s(door), 이후 {SETTLE_CHK} s 간격 두 번 연속 전 관절 |Δq| < {SETTLE_DQ}° 면 정착, 상한 {SETTLE_MAX} s (위치 안정, D481 ④)", "mesh_tag": MESH_TAG, "urdf": URDF, "blend_speed_dps": args.speed, "n_events": len(EVENTS), "n_segments": len(SCHED), "fk_selfcheck_max_mm": round(1000 * max([np.linalg.norm(fk_lip_world(e["read"][:5]) - jsonl_lip_world(e["lip"])) for e in EVENTS if e["ev"] == "goto"] or [0.0]), 4)}
json.dump(res, open(os.path.join(OUT, "replay_result.json"), "w"), indent=1)   # 선기록 (D477)

# 표본점: 시각 STL(mm) 정점 + 표면 표본 ~1.5 mm 격자 상당
PTS, RAD = {}, {}
for b, m in DIST_BODIES.items():
    mesh = trimesh.load(os.path.join(MESH_DIR, m), force="mesh"); mesh.apply_scale(0.001)
    n = int(min(20000, max(2000, mesh.area / (0.0015 ** 2)))); sp, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    PTS[b] = np.vstack([mesh.vertices, sp]).astype(np.float64); RAD[b] = float(np.linalg.norm(PTS[b], axis=1).max())
res["dist_points"] = {b: int(len(p)) for b, p in PTS.items()}


def look_at_quat(pos, tgt):
    f = np.asarray(tgt, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.99 else np.array([1.0, 0, 0])
    y = np.cross(up, f); y /= np.linalg.norm(y); z = np.cross(f, y)
    x, y_, z_, w = Rotation.from_matrix(np.stack([f, y, z], 1)).as_quat(); return (float(w), float(x), float(y_), float(z_))
CAM_SIDE = ((0.95, -0.72, 0.78), (0.28, 0.02, 0.33)); CAM_TOP = ((0.22, 0.10, 1.62), (0.22, 0.10, 0.35))   # D474: 카메라-대상 ≥ 0.85 m (측면 1.09 · 위 1.27), 확대는 focal 로
def cam_cfg(path, pos_tgt, focal):
    return CameraCfg(prim_path=path, update_period=0.0, height=H, width=W, data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(0.05, 6.0)),
                     offset=CameraCfg.OffsetCfg(pos=pos_tgt[0], rot=look_at_quat(*pos_tgt), convention="world"))
def static_box(path, lo, hi, color, opacity=1.0, collide=True):
    lo, hi = np.asarray(lo), np.asarray(hi); c = (lo + hi) / 2; s = hi - lo
    return AssetBaseCfg(prim_path=path, spawn=sim_utils.CuboidCfg(size=tuple(float(v) for v in s), collision_props=sim_utils.CollisionPropertiesCfg() if collide else None,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity)), init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(v) for v in c)))

ROBOT = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD, activate_contact_sensors=True, articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False), rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, PLATE_Z), joint_pos={"link2_to_link3": math.radians(HOME[2]), "link5_to_gripper_link": math.radians(SCHED[0]["d0"]) if SCHED and "d0" in SCHED[0] else 0.0}),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5"], stiffness=800.0, damping=40.0, effort_limit_sim=8.0),
               "door": ImplicitActuatorCfg(joint_names_expr=["link5_to_gripper_link"], stiffness=300.0, damping=30.0, effort_limit_sim=1.96)})
WALLC = (0.80, 0.55, 0.25); RIM = (0.45, 0.25, 0.10); RT = 0.006   # 벽색 / 윗단 테두리(6 mm 각, 불투명)+모서리 기둥 색 — 시각 전용, 충돌 없음(벽 AABB 안에 있어 거리 게이트 영향 없음). configclass 안에 상수 두면 asset 으로 오인됨

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9)))
    dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
    key = AssetBaseCfg(prim_path="/World/key", spawn=sim_utils.DistantLightCfg(intensity=2500.0), init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.3827, 0.0, 0.0)))
    robot: ArticulationCfg = ROBOT
    wall_xp = static_box("/World/box/wall_xp", *SLABS["wall_xp"], WALLC, 0.45)
    wall_xn = static_box("/World/box/wall_xn", *SLABS["wall_xn"], WALLC, 0.45)
    wall_yp = static_box("/World/box/wall_yp", *SLABS["wall_yp"], WALLC, 0.45)
    wall_yn = static_box("/World/box/wall_yn", *SLABS["wall_yn"], WALLC, 0.45)
    box_floor = static_box("/World/box/box_floor", *SLABS["box_floor"], WALLC, 1.0)
    support = static_box("/World/box/support", *SLABS["support"], (0.22, 0.22, 0.26), 1.0)
    pellet = static_box("/World/vis/pellet_slab", (BOX_CX - HX, -HY, BOX_BOT + WALL), (BOX_CX + HX, HY, PELLET_Z), (0.85, 0.75, 0.35), 0.25, collide=False)
    place_mark = static_box("/World/vis/place_mark", (PLACE_LIP[0] - 0.04, PLACE_LIP[1] - 0.04, PLACE_LIP[2] - 0.010), (PLACE_LIP[0] + 0.04, PLACE_LIP[1] + 0.04, PLACE_LIP[2]), (0.05, 0.75, 0.15), 1.0, collide=False)   # W2b: 불투명 패드 8×8 cm × 10 mm, 윗면 = 놓기 립 z (렌더에서 보이게; 충돌 없음)
    pedestal = static_box("/World/vis/pedestal", (-0.08, -0.08, 0.0), (0.08, 0.08, PLATE_Z), (0.4, 0.4, 0.42), 1.0, collide=False)
    rim_xp = static_box("/World/vis/rim_xp", (BOX_CX + HX - RT, -HY - WALL, BOX_TOP - RT), (BOX_CX + HX + WALL, HY + WALL, BOX_TOP), RIM, 1.0, collide=False)
    rim_xn = static_box("/World/vis/rim_xn", (BOX_CX - HX - WALL, -HY - WALL, BOX_TOP - RT), (BOX_CX - HX + RT, HY + WALL, BOX_TOP), RIM, 1.0, collide=False)
    rim_yp = static_box("/World/vis/rim_yp", (BOX_CX - HX, HY - RT, BOX_TOP - RT), (BOX_CX + HX, HY + WALL, BOX_TOP), RIM, 1.0, collide=False)
    rim_yn = static_box("/World/vis/rim_yn", (BOX_CX - HX, -HY - WALL, BOX_TOP - RT), (BOX_CX + HX, -HY + RT, BOX_TOP), RIM, 1.0, collide=False)
    post_a = static_box("/World/vis/post_a", (BOX_CX + HX - RT, -HY - WALL, BOX_BOT), (BOX_CX + HX + WALL, -HY + RT, BOX_TOP), RIM, 1.0, collide=False)
    post_b = static_box("/World/vis/post_b", (BOX_CX + HX - RT, HY - RT, BOX_BOT), (BOX_CX + HX + WALL, HY + WALL, BOX_TOP), RIM, 1.0, collide=False)
    post_c = static_box("/World/vis/post_c", (BOX_CX - HX - WALL, -HY - WALL, BOX_BOT), (BOX_CX - HX + RT, -HY + RT, BOX_TOP), RIM, 1.0, collide=False)
    post_d = static_box("/World/vis/post_d", (BOX_CX - HX - WALL, HY - RT, BOX_BOT), (BOX_CX - HX + RT, HY + WALL, BOX_TOP), RIM, 1.0, collide=False)
    cam_side: CameraCfg = cam_cfg("/World/CamSide", CAM_SIDE, 30.0)
    cam_top: CameraCfg = cam_cfg("/World/CamTop", CAM_TOP, 26.0)
    contact: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/(link3|link4|link5|grab_fixed|gripper_link)", update_period=0.0, history_length=0)

try: FONT = ImageFont.load_default(size=16)
except TypeError: FONT = ImageFont.load_default()


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device)); scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0)); sim.reset()
    robot: Articulation = scene["robot"]; cams = [scene["cam_side"], scene["cam_top"]]; cs: ContactSensor = scene["contact"]
    jn = list(robot.joint_names); bn = list(robot.body_names); cbn = list(cs.body_names)
    if args.tcp_mass is not None:
        m = robot.root_physx_view.get_masses().clone(); m[:, bn.index("hand_tcp")] = args.tcp_mass; robot.root_physx_view.set_masses(m, torch.arange(1)); res["tcp_mass_override_kg"] = args.tcp_mass
    ARM = [jn.index(n) for n in ("base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5")]; iD = jn.index("link5_to_gripper_link")
    BI = {b: bn.index(b) for b in DIST_BODIES}; i5 = bn.index("link5"); CI = {b: cbn.index(b) for b in DIST_BODIES if b in cbn}
    res.update({"joint_names": jn, "body_names": bn, "contact_bodies": cbn, "cameras": {"side": CAM_SIDE, "top": CAM_TOP, "res": [W, H]},
                "body_mass_kg": {n: round(float(m), 5) for n, m in zip(bn, robot.data.default_mass[0].cpu().numpy())},
                "physx_dof_gains": {n: [round(float(k), 3), round(float(d), 3), round(float(f), 3)] for n, k, d, f in zip(jn, robot.root_physx_view.get_dof_stiffnesses()[0].cpu().numpy(), robot.root_physx_view.get_dof_dampings()[0].cpu().numpy(), robot.root_physx_view.get_dof_max_forces()[0].cpu().numpy())}})   # PhysX 가 실제로 쓰는 값(질량·강성/감쇠/상한)
    dt = sim.get_physics_dt(); every = max(1, int(round(1.0 / (args.fps * dt))))
    tgt = torch.zeros(1, robot.num_joints, device=sim.device)
    gotos, log, t, step_i, frame_i = [], [], 0.0, 0, 0
    dmin_all = {b: [1e9, None] for b in DIST_BODIES}; fmax_all = {b: [0.0, None] for b in CI}

    def body_pose(b):
        p = robot.data.body_link_pos_w[0, BI[b]].cpu().numpy().astype(np.float64); q = robot.data.body_link_quat_w[0, BI[b]].cpu().numpy()
        return p, Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    def min_dists():
        out = {}
        for b in DIST_BODIES:
            p, R = body_pose(b); bound = float(aabb_sdist(p[None], BOX_ALL_LO, BOX_ALL_HI)[0]) - RAD[b]
            if bound > 0.05: out[b] = (bound, "bound", None); continue
            Pw = PTS[b] @ R.T + p; best = (1e9, None)
            for k, name in enumerate(SLABS):
                d = aabb_sdist(Pw, SLAB_LO[k], SLAB_HI[k]); j = int(np.argmin(d))
                if d[j] < best[0]: best = (float(d[j]), name, Pw[j].tolist())
            out[b] = best
        return out

    def render_frame(seg, dm, door_deg):
        imgs = [Image.fromarray(c.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)) for c in cams]
        canvas = Image.new("RGB", (2 * W, H)); canvas.paste(imgs[0], (0, 0)); canvas.paste(imgs[1], (W, 0))
        gm = min(dm[b][0] for b in GATE_BODIES)
        ImageDraw.Draw(canvas).text((6, 4), f"c{seg['cycle']} {seg['name']}  t={t:.1f}s  door={door_deg:.1f}deg  dmin(grab,link5)={1000*gm:.0f}mm", fill=(255, 255, 0), font=FONT)
        canvas.save(os.path.join(FR, f"f_{frame_i:05d}.jpg"), quality=88)
        return [float(np.asarray(im, dtype=np.float32).std()) for im in imgs]

    for si, seg in enumerate(SCHED):
        if seg["dur"] == 0.0:
            log.append({"t": round(t, 3), "seg": si, "marker": seg["kind"], "cycle": seg["cycle"]}); continue
        n_b, n_s, n_smax, n_chk = int(round(seg["dur"] / dt)), int(round(seg["settle"] / dt)), int(round(SETTLE_MAX / dt)), int(round(SETTLE_CHK / dt)); q0, q1 = np.array(seg["q0"]), np.array(seg["q1"]); k = -1
        q_chk, n_stable, dq_last = None, 0, None
        while True:
            k += 1
            if k >= n_b and (k - n_b) % n_chk == 0:                     # 위치 안정 판정: SETTLE_CHK 간격으로 전 관절 |Δq| 검사, 두 번 연속 < SETTLE_DQ 면 정착 (D481 ④)
                q_now = np.degrees(robot.data.joint_pos[0].cpu().numpy())
                if q_chk is not None: dq_last = float(np.abs(q_now - q_chk).max()); n_stable = n_stable + 1 if dq_last < SETTLE_DQ else 0
                q_chk = q_now
            if k >= n_b + n_s and (k >= n_b + n_smax or n_stable >= 2): break   # 정착 = 위치 안정 두 번 연속, 상한 SETTLE_MAX
            a = 0.5 * (1 - math.cos(math.pi * min(1.0, (k + 1) / n_b))); q = (1 - a) * q0 + a * q1; d = (1 - a) * seg["d0"] + a * seg["d1"]
            for j, idx in enumerate(ARM): tgt[0, idx] = math.radians(q[j])
            tgt[0, iD] = math.radians(d)
            robot.set_joint_position_target(tgt); scene.write_data_to_sim()
            is_frame = (step_i % every == 0); sim.step(render=is_frame); scene.update(dt); t += dt; step_i += 1
            dm = min_dists(); qp = robot.data.joint_pos[0].cpu().numpy(); f = cs.data.net_forces_w[0].cpu().numpy()
            for b, v in dm.items():
                if v[0] < dmin_all[b][0]: dmin_all[b] = [v[0], dict(t=round(t, 3), seg=si, name=seg["name"], cycle=seg["cycle"], slab=v[1], point=v[2], frame=frame_i)]
            for b, ci in CI.items():
                fn = float(np.linalg.norm(f[ci]))
                if fn > fmax_all[b][0]: fmax_all[b] = [fn, dict(t=round(t, 3), seg=si, name=seg["name"], cycle=seg["cycle"])]
            if is_frame:
                stds = render_frame(seg, dm, math.degrees(float(qp[iD]))); frame_i += 1
            if is_frame or step_i % 4 == 0:
                log.append({"t": round(t, 3), "seg": si, "name": seg["name"], "cycle": seg["cycle"], "q_tgt": [round(float(v), 3) for v in q], "q_sim": [round(math.degrees(float(qp[i])), 3) for i in ARM],
                            "door_tgt": round(float(d), 3), "door_sim": round(math.degrees(float(qp[iD])), 3), "dmin_mm": {b: round(1000 * v[0], 2) for b, v in dm.items()},
                            "contact_N": {b: round(float(np.linalg.norm(f[ci])), 4) for b, ci in CI.items()}, "tau": [round(float(robot.data.applied_torque[0, i]), 4) for i in ARM] + [round(float(robot.data.applied_torque[0, iD]), 4)],
                            **({"frame": frame_i - 1, "img_std": [round(s, 1) for s in stds]} if is_frame else {})})
        # 정착 후 기록
        qp = robot.data.joint_pos[0].cpu().numpy(); p5, R5 = body_pose("link5"); lip_sim = p5 + R5 @ LIP_L5; e = EVENTS[seg["i"]]; dm = min_dists()
        rec = {"seg": si, "i_event": seg["i"], "kind": seg["kind"], "name": seg["name"], "cycle": seg["cycle"], "t": round(t, 3), "frame": frame_i - 1, "settle_s": round((k - n_b) * dt, 3), "vel_max_dps": round(math.degrees(float(robot.data.joint_vel[0].abs().max())), 3), "settle_dq_last_deg": dq_last, "settle_n_stable": n_stable,
               "q_cmd": [round(float(v), 3) for v in q1], "q_sim": [round(math.degrees(float(qp[i])), 3) for i in ARM], "door_tgt": round(seg["d1"], 3), "door_sim": round(math.degrees(float(qp[iD])), 3),
               "lip_sim": [round(float(v), 5) for v in lip_sim], "lip_fk_cmd": [round(float(v), 5) for v in fk_lip_world(q1)], "dmin_mm": {b: round(1000 * v[0], 2) for b, v in dm.items()},
               "tau": [round(float(robot.data.applied_torque[0, i]), 4) for i in ARM] + [round(float(robot.data.applied_torque[0, iD]), 4)]}
        if seg["kind"] == "goto":
            lr = jsonl_lip_world(e["lip"]); rec.update({"q_read": [round(float(v), 3) for v in e["read"][:5]], "lip_real": [round(float(v), 5) for v in lr], "lip_fk_read": [round(float(v), 5) for v in fk_lip_world(e["read"][:5])],
                                                       "d_sim_real_mm": round(1000 * float(np.linalg.norm(lip_sim - lr)), 3), "d_sim_fkcmd_mm": round(1000 * float(np.linalg.norm(lip_sim - fk_lip_world(q1))), 3),
                                                       "d_real_fkcmd_mm": round(1000 * float(np.linalg.norm(lr - fk_lip_world(q1))), 3), "loads_real": e.get("loads"), "dev_real_deg": e.get("dev")})
        gotos.append(rec)
        if si % 10 == 0: print(f"[replay] seg {si}/{len(SCHED)} t={t:.1f}s frames={frame_i} {seg['name']}", flush=True)
    n_goto = sum(1 for r in gotos if r["kind"] == "goto"); dsr = [r["d_sim_real_mm"] for r in gotos if r["kind"] == "goto"]
    res.update({"frames": frame_i, "fps": args.fps, "sim_seconds": round(t, 2), "n_goto": n_goto, "n_door": sum(1 for r in gotos if r["kind"] == "door"),
                "n_scoop_done": sum(1 for r in log if r.get("marker") == "scoop_done"), "n_place_done": sum(1 for r in log if r.get("marker") == "place_done"),
                "dmin_overall_mm": {b: [round(1000 * v[0], 3), v[1]] for b, v in dmin_all.items()}, "contact_max_N": {b: [round(v[0], 4), v[1]] for b, v in fmax_all.items()},
                "d_sim_real_mm": {"median": round(float(np.median(dsr)), 3), "max": round(float(np.max(dsr)), 3), "mean": round(float(np.mean(dsr)), 3)} if dsr else None,
                "finite": bool(all(np.isfinite(r["lip_sim"]).all() and np.isfinite(r["q_sim"]).all() for r in gotos))})
    res["ok"] = bool(res["finite"] and frame_i > 0)
    json.dump(res, open(os.path.join(OUT, "replay_result.json"), "w"), indent=1); json.dump(gotos, open(os.path.join(OUT, "settle_records.json"), "w"), indent=1); json.dump(log, open(os.path.join(OUT, "replay_log.json"), "w"))
    print("[replay]", json.dumps({k: v for k, v in res.items() if k in ("ok", "frames", "sim_seconds", "n_goto", "n_door", "n_scoop_done", "n_place_done", "dmin_overall_mm", "contact_max_N", "d_sim_real_mm", "fk_selfcheck_max_mm")}, ensure_ascii=False), flush=True)


try:
    main()
except Exception:
    import traceback; res["error"] = traceback.format_exc(); json.dump(res, open(os.path.join(OUT, "replay_result.json"), "w"), indent=1); print("[replay] ERROR", res["error"][-1500:], flush=True)
import threading
threading.Thread(target=lambda: (time.sleep(20), os._exit(0)), daemon=True).start()
simulation_app.close()
