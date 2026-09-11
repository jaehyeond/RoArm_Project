#!/usr/bin/env python3
"""W9 — Isaac Sim 5.1 렌더: W8 DEME 퍼내기 결과(`render_timeline_cell1.npz`) 를 S1 v1 로봇 USD(W1b) 와 합쳐 "퍼내는 장면" 영상.

입력: W8 렌더 타임라인(클럼프 중심·자세 [T,N], 툴 = 두꺼워진 립점(link5 (8.1,0,169.6) mm) 세계 위치, 문 관절각) ·
      렌즈 클럼프 템플릿(더미 npz `clump_template_json`: 구 7개 반경·상대위치) · 로봇 USD `local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd` ·
      환경 상수 = `sim_isaaclab_s1_env_replay.py`(W2/W2b) 의 베이스판 0.38 · 상자 0.31×0.22(중심 x 0.35, 윗단 0.385, 바닥 0.16) · 받침 0.16 을 복사(스크립트 import 는 AppLauncher 부작용이라 불가).
방법: 입자 = 물리 없는 UsdGeomPointInstancer 3개(나머지·딸려간·포획, 프로토타입 = 구 7개 합집합 메시 1개씩), 프레임마다 positions/orientations 갱신(키네마틱, PhysX 없음).
      로봇 = W8 툴 포즈(립 169.6) → 립 166.6 목표로 환산(툴 수직이라 +z 3 mm) → `hw_s1_manual._grid`(= solve_fast 의 툴 수직 격자 해, 어깨 ≤ URDF 상한 90°) +
      `hw_s1_scoop_probe.lip_fw` 로 반경·베이스 요 보정 → 5관절, 문 관절 = door_deg. 프레임마다 write_joint_state_to_sim(중력 끔, 상자 충돌 없음 → 사실상 키네마틱).
      립 오차 = 시뮬 link5 포즈로 계산한 169.6 립 vs W8 tool_pos(세계). 두 모듈은 읽기 전용 import(수정 0).
좌표: DEME 세계(더미 npz 프레임 = 상자 바닥 중심, z 위) → 로봇 세계 = + (BOX_CX, 0, --deme-origin-z). 축 정렬 근거 = 실물 09-07 scoop 자세 FK 의 link5 회전 == W8 R_W(열 = link5 축).
      기본 --deme-origin-z 0.163 = 상자 안쪽 바닥(0.16 + 벽 3 mm 가정). 실물 펠릿면 0.26 에 더미 표면(≈0.040)을 맞추려면 0.2205 + --pellet-slab.
카메라 2대(측면 45°·위 nadir), 대상 거리 1.05 m(D474 AABB·D480 근접 RTX 검정면 회피, 확대는 focal 로), Isaac Lab Camera(annotator) 로 저장 — BasicWriter 금지.
게이트: G1 프레임 수 = 타임라인 길이 / G2 빈(검정) 프레임 0 / G3 립 오차 ≤ 5 mm 전 프레임 / G4 포획 클럼프가 보울 공동 안(W8 in_cav 식을 시뮬 link5 포즈로 재적용 + 스크린샷 육안).
사용: OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 1500 ~/miniconda3/envs/isaaclab/bin/python -u sim_isaac_render_deme_scoop.py --headless --enable_cameras \
        [--out DIR] [--deme-origin-z 0.163] [--pellet-slab] [--frame-stride N] [--res 1024x640]
      python3 sim_isaac_render_deme_scoop.py --ik-only          (Isaac 없이 IK·좌표·sha 사전 검사, gates_w9.json 선기록)
"""
import argparse, os, sys, json, math, time, hashlib, subprocess
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sim_scripts"))
import hw_s1_scoop_probe as S   # 읽기 전용: lip_fw / chain / SHOULDER_ABOVE_PLATE / LIP_L5 / WRIST_MAX
import hw_s1_manual as M        # 읽기 전용: _grid (solve_fast 의 격자 해)

S1_SIM = "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim"
W8_DIR = f"{S1_SIM}/w8_deme_scoop_lens"
PILE_NPZ = "/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz"
# ── 실물 환경 (sim_isaaclab_s1_env_replay.py W2/W2b 상수 복사, 바닥 기준 m) ──
PLATE_Z = 0.38; BOX_CX, HX, HY = 0.35, 0.155, 0.11
BOX_TOP, BOX_BOT, PELLET_Z, WALL = 0.385, 0.16, 0.26, 0.003          # WALL = 가정(실측 아님)
SLABS = {
    "wall_xp": ((BOX_CX + HX, -HY - WALL, BOX_BOT), (BOX_CX + HX + WALL, HY + WALL, BOX_TOP)),
    "wall_xn": ((BOX_CX - HX - WALL, -HY - WALL, BOX_BOT), (BOX_CX - HX, HY + WALL, BOX_TOP)),
    "wall_yp": ((BOX_CX - HX, HY, BOX_BOT), (BOX_CX + HX, HY + WALL, BOX_TOP)),
    "wall_yn": ((BOX_CX - HX, -HY - WALL, BOX_BOT), (BOX_CX + HX, -HY, BOX_TOP)),
    "box_floor": ((BOX_CX - HX, -HY, BOX_BOT), (BOX_CX + HX, HY, BOX_BOT + WALL)),
    "support": ((BOX_CX - HX - WALL, -HY - WALL, 0.0), (BOX_CX + HX + WALL, HY + WALL, BOX_BOT)),
}
PHASES = {0: "settle", 1: "descend", 2: "close", 3: "lift", 4: "reclose"}
URDF_SHOULDER_MAX = 90.0        # roarm_m3_s1_v1.urdf link1_to_link2 limit ±1.5708 rad → IK 격자 상한(PhysX 클램프 회피)
ARM_JOINTS = ["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5"]; DOOR_JOINT = "link5_to_gripper_link"
BOWL_R_IN, CHEEK_HALF_Y = 0.020, 0.0182    # sim_deme_scoop_s1.py P["bowl_r_in_mm"] 20 · P["cheek_half_y_mm"] 18.2 (in_cav 식)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def solve_frame(lip_w, sh_max=URDF_SHOULDER_MAX):
    """립(166.6) 세계 목표 → 5관절 deg. 툴 수직 격자(hw_s1_manual._grid, 2° → 0.125°) + 반경·베이스 요 보정 3회. 반환 (q5, lip_fk_w, err_m, vert_cos) 또는 None."""
    x_t, y_t, z_t = (float(v) for v in lip_w); r_t = math.hypot(x_t, y_t); z_fw = z_t - PLATE_Z - S.SHOULDER_ABOVE_PLATE
    r_cmd, q, l0 = r_t, None, None
    for _ in range(3):
        b = M._grid(r_cmd, z_fw, (-30.0, sh_max), (-10.0, 150.0), 2.0, S.WRIST_MAX, 0.995)
        if b is None: return None
        b2 = M._grid(r_cmd, z_fw, (b[1][1] - 2.0, min(sh_max, b[1][1] + 2.0)), (b[1][2] - 2.0, b[1][2] + 2.0), 0.125, S.WRIST_MAX, 0.995)
        if b2 is not None and b2[0] < b[0]: b = b2
        q = list(b[1]); l0, _ = S.lip_fw(q); r_cmd += r_t - math.hypot(l0[0], l0[1])
    q[0] = math.degrees(math.atan2(y_t, x_t) - math.atan2(l0[1], l0[0]))
    # 손목 롤(link4_to_link5 = 툴축 = 수직) 로 베이스 요를 상쇄 → link5 회전 == R_W (W8 툴 자세). 립은 툴축에서 8.1 mm 라 이동 < 0.2 mm → 요 1회 재보정
    q[4] = min((+q[0], -q[0]), key=lambda r: rot_angle_deg(link5_T_world(q[:4] + [r])[:3, :3], R_W))
    l0, _ = S.lip_fw(q); q[0] += math.degrees(math.atan2(y_t, x_t) - math.atan2(l0[1], l0[0]))
    l, zax = S.lip_fw(q); lw = np.array([l[0], l[1], l[2] + S.SHOULDER_ABOVE_PLATE + PLATE_Z])
    return q, lw, float(np.linalg.norm(lw - np.array([x_t, y_t, z_t]))), float(-zax[2])


def link5_T_world(q5):
    """chain() 은 로봇 root 기준 → 세계 = + (0,0,PLATE_Z)."""
    T = S.chain(q5)["link4_to_link5"].copy(); T[2, 3] += PLATE_Z; return T


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0; return math.degrees(math.acos(max(-1.0, min(1.0, c))))


parser = argparse.ArgumentParser()
parser.add_argument("--timeline", default=f"{W8_DIR}/render_timeline_cell1.npz"); parser.add_argument("--pile", default=PILE_NPZ)
parser.add_argument("--usd", default="local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd"); parser.add_argument("--out", default=f"{S1_SIM}/w9_isaac_render_deme")
parser.add_argument("--deme-origin-z", type=float, default=BOX_BOT + WALL, help="DEME 원점(상자 바닥 중심) 의 로봇 세계 z (기본 0.163 = 상자 안쪽 바닥)")
parser.add_argument("--pellet-slab", action="store_true", help="상자 안쪽 바닥 ~ DEME 원점 사이를 W2 펠릿 슬래브 시각(충돌 없음)으로 채움")
parser.add_argument("--frame-stride", type=int, default=1, help="스모크: 프레임 N 개마다 1개만 렌더 (1 = 전부)")
parser.add_argument("--res", default="1024x640"); parser.add_argument("--fps", type=int, default=10); parser.add_argument("--warmup", type=int, default=4)
parser.add_argument("--grab-opacity", type=float, default=1.0, help="그랩 부품(grab_fixed·gripper_link) 시각 재질 불투명도. 1.0 = 원래 재질(영상용). <1.0 = 이 RTX 설정(Isaac Sim 5.1 headless, fractionalCutoutOpacity 유무 무관)에서는 반투명이 아니라 부품이 통째로 사라짐 → x-ray 진단 렌더용(보울 안 포획 펠릿 확인)")
parser.add_argument("--ik-only", action="store_true")
args, _ = parser.parse_known_args()

OUT = os.path.abspath(args.out); FR = os.path.join(OUT, "frames"); os.makedirs(FR, exist_ok=True)
TL = np.load(args.timeline, allow_pickle=True); META = json.loads(str(TL["metadata_json"]))
PILE = np.load(args.pile, allow_pickle=True); TPL = json.loads(str(PILE["clump_template_json"]))
OFFS = np.array(TPL["offsets_m"], float); RADII = np.array(TPL["sphere_radii_m"], float)
R_W = np.array(META["link5_to_world_R_W_columns_are_link5_axes"], float)
L169 = np.array(META["lip_l5_mm"], float) / 1000.0; L166 = np.array(S.LIP_L5[:3], float); H5 = np.array(META["hinge_l5_mm"], float) / 1000.0; C5 = np.array(META["bowl_center_l5_mm"], float) / 1000.0
ORIGIN = np.array([BOX_CX, 0.0, args.deme_origin_z])
N_ALL = int(len(TL["t_s"])); IDX = list(range(0, N_ALL, max(1, args.frame_stride)))
if IDX[-1] != N_ALL - 1: IDX.append(N_ALL - 1)
TOOL_W = TL["tool_pos_m"].astype(float) + ORIGIN                  # 립 169.6 세계
LIP166_W = TOOL_W + (R_W @ (L166 - L169))                          # 166.6 립 목표 (= +z 3 mm)
DOOR_W = TL["door_pos_m"].astype(float) + ORIGIN                   # 힌지 세계
CAP = np.array(TL["captured_ids"], int); CAR = np.setdiff1d(np.array(TL["carried_ids"], int), CAP)
N_CL = int(TL["clump_pos_m"].shape[1]); REST = np.setdiff1d(np.arange(N_CL), np.concatenate([CAP, CAR]))
PH = [int(v) for v in TL["phase"]]; K_CAP = max(i for i in range(N_ALL) if PH[i] == 2)   # 포획 순간 = 폐합 마지막 프레임
assert bool(np.allclose(TL["tool_quat_xyzw"], [0, 0, 0, 1], atol=1e-6)), "tool_quat 가 단위가 아님 — 툴 수직 가정 위반"

# ── IK (Isaac 불필요) ──
t0 = time.time(); IK, cache = [], {}
for i in range(N_ALL):
    key = tuple(np.round(LIP166_W[i], 5))
    if key not in cache: cache[key] = solve_frame(LIP166_W[i])
    sol = cache[key]
    if sol is None: raise RuntimeError(f"IK 실패 frame {i} lip166 {LIP166_W[i]}")
    q, lw, err, vc = sol; T5 = link5_T_world(q); lip169_fk = T5[:3, :3] @ L169 + T5[:3, 3]
    IK.append(dict(frame=i, t_s=round(float(TL["t_s"][i]), 4), phase=PHASES.get(PH[i], str(PH[i])), q5_deg=[round(float(v), 4) for v in q], door_deg=round(float(TL["door_deg"][i]), 4),
                   lip166_target_w=[round(float(v), 5) for v in LIP166_W[i]], lip166_fk_w=[round(float(v), 5) for v in lw], ik_err_mm=round(1000 * err, 4), vert_cos=round(vc, 6),
                   lip169_fk_vs_tool_mm=round(1000 * float(np.linalg.norm(lip169_fk - TOOL_W[i])), 4), R_angle_vs_RW_deg=round(rot_angle_deg(T5[:3, :3], R_W), 4)))
IK_SUMMARY = dict(n=N_ALL, distinct_targets=len(cache), max_ik_err_mm=max(r["ik_err_mm"] for r in IK), max_fk169_vs_tool_mm=max(r["lip169_fk_vs_tool_mm"] for r in IK),
                  max_R_angle_deg=max(r["R_angle_vs_RW_deg"] for r in IK), min_vert_cos=min(r["vert_cos"] for r in IK), q_range_deg={n: [min(r["q5_deg"][j] for r in IK), max(r["q5_deg"][j] for r in IK)] for j, n in enumerate(ARM_JOINTS)},
                  door_range_deg=[min(r["door_deg"] for r in IK), max(r["door_deg"] for r in IK)], ik_seconds=round(time.time() - t0, 2))
INPUTS = {p: sha256(p) for p in [args.timeline, args.pile, args.usd] + [os.path.join(os.path.dirname(args.usd), "configuration", f) for f in sorted(os.listdir(os.path.join(os.path.dirname(args.usd), "configuration")))]
          + ["hw_s1_scoop_probe.py", "hw_s1_manual.py", "sim_scripts/roarm_kinematics.py", "sim_isaaclab_s1_env_replay.py", os.path.abspath(__file__), f"{W8_DIR}/gates_w8.json", f"{W8_DIR}/params_w8F_cell_c.json"] if os.path.exists(p)}
RES = {"artifact": "W9_ISAAC_RENDER_DEME_SCOOP_V1", "ok": False, "stage": "ik", "out": OUT, "inputs_sha256": INPUTS, "timeline_meta": {k: META[k] for k in ("artifact", "cell_tag", "params_file", "pile_sha16", "dt_s", "world_frame", "lip_l5_mm", "hinge_l5_mm", "bowl_center_l5_mm", "q_open_joint_deg", "servo_zero_offset_deg")},
       "frame_contract": {"n_timeline": N_ALL, "render_idx": IDX, "frame_stride": args.frame_stride, "k_capture": K_CAP, "n_captured": int(len(CAP)), "n_carried_not_captured": int(len(CAR)), "n_rest": int(len(REST)), "n_clumps": N_CL},
       "mapping": {"deme_origin_world": ORIGIN.tolist(), "axes": "DEME xyz == 로봇 세계 xyz (실물 09-07 scoop FK link5 R == R_W)", "pellet_slab_visual": bool(args.pellet_slab), "pile_surface_world_z_approx": round(float(args.deme_origin_z + np.percentile(TL["clump_pos_m"][0, :, 2], 99.5)), 4), "real_pellet_surface_z": PELLET_Z,
                   "lip_target": "W8 tool_pos(169.6) + R_W·(L166 − L169) = +z 3 mm → IK 는 hw_s1_scoop_probe.LIP_L5(166.6) 기준"},
       "env": dict(plate_z=PLATE_Z, box_cx=BOX_CX, box_inner=[2 * HX, 2 * HY], box_top=BOX_TOP, box_bot=BOX_BOT, wall_assumed=WALL, pellet_z_real=PELLET_Z, slabs={k: [list(v[0]), list(v[1])] for k, v in SLABS.items()}, collide="none (시각 전용)"),
       "ik": IK_SUMMARY, "ik_frames": IK, "cavity_rule": {"bowl_r_in_m": BOWL_R_IN, "cheek_half_y_m": CHEEK_HALF_Y, "source": "sim_deme_scoop_s1.py in_cav (link5 프레임, hypot(x−Cx, z−Cz) < r_in & |y| < cheek)"}}
json.dump(RES, open(os.path.join(OUT, "gates_w9.json"), "w"), indent=1, ensure_ascii=False)    # 선기록 (D477)
print("[w9] IK", json.dumps(IK_SUMMARY, ensure_ascii=False), flush=True)
if args.ik_only: sys.exit(0)

# ── Isaac ──
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser); args = parser.parse_args(); app_launcher = AppLauncher(args); simulation_app = app_launcher.app
import torch, trimesh
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sim.spawners.materials import spawn_preview_surface
from isaaclab.sim.utils import bind_visual_material
from isaaclab.utils import configclass
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, Gf, Vt, Sdf
import omni.usd
import isaacsim
USD = os.path.abspath(args.usd); W, H = (int(v) for v in args.res.split("x"))
RES["versions"] = {"isaacsim": open(os.path.join(os.path.dirname(isaacsim.__file__), "VERSION")).read().strip() if os.path.exists(os.path.join(os.path.dirname(isaacsim.__file__), "VERSION")) else "?", "isaaclab": "2.3.0 (pip show)", "torch": torch.__version__, "trimesh": trimesh.__version__}


def look_at_quat(pos, tgt):
    f = np.asarray(tgt, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.99 else np.array([1.0, 0, 0])
    y = np.cross(up, f); y /= np.linalg.norm(y); z = np.cross(f, y)
    x, y_, z_, w = Rotation.from_matrix(np.stack([f, y, z], 1)).as_quat(); return (float(w), float(x), float(y_), float(z_))
TARGET = np.array([BOX_CX, 0.0, args.deme_origin_z + 0.045]); CAM_DIST = 1.05                      # 대상 = 더미 표면·보울 근처, 거리 ≥ 1.0 m
CAM_SIDE = (tuple(float(v) for v in TARGET + CAM_DIST * np.array([0.5, -0.5, math.sqrt(0.5)])), tuple(float(v) for v in TARGET))   # 고도 45°, 방위 −45°(+x,−y 사분면 = 문 쪽)
CAM_TOP = (tuple(float(v) for v in TARGET + np.array([0.0, 0.0, CAM_DIST])), tuple(float(v) for v in TARGET))
FOCAL = {"side": 40.0, "top": 40.0}                          # 확대는 focal 로(거리 1.05 m 유지)
def cam_cfg(path, pos_tgt, focal):
    return CameraCfg(prim_path=path, update_period=0.0, height=H, width=W, data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(0.05, 6.0)),
                     offset=CameraCfg.OffsetCfg(pos=pos_tgt[0], rot=look_at_quat(*pos_tgt), convention="world"))
def static_box(path, lo, hi, color, opacity=1.0):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float); c = (lo + hi) / 2; s = hi - lo
    return AssetBaseCfg(prim_path=path, spawn=sim_utils.CuboidCfg(size=tuple(float(v) for v in s), collision_props=None, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity)), init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(v) for v in c)))

ROBOT = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD, articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False), rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, PLATE_Z), joint_pos={**{n: math.radians(IK[IDX[0]]["q5_deg"][j]) for j, n in enumerate(ARM_JOINTS)}, DOOR_JOINT: math.radians(IK[IDX[0]]["door_deg"])}),
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=ARM_JOINTS, stiffness=2000.0, damping=100.0, effort_limit_sim=50.0), "door": ImplicitActuatorCfg(joint_names_expr=[DOOR_JOINT], stiffness=500.0, damping=30.0, effort_limit_sim=10.0)})
WALLC = (0.80, 0.55, 0.25); RIM = (0.45, 0.25, 0.10); RT = 0.006
SLAB_TOP = args.deme_origin_z if args.pellet_slab else None

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
    key = AssetBaseCfg(prim_path="/World/key", spawn=sim_utils.DistantLightCfg(intensity=2500.0), init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.3827, 0.0, 0.0)))
    robot: ArticulationCfg = ROBOT
    wall_xp = static_box("/World/box/wall_xp", *SLABS["wall_xp"], WALLC, 0.45)
    wall_xn = static_box("/World/box/wall_xn", *SLABS["wall_xn"], WALLC, 0.45)
    wall_yp = static_box("/World/box/wall_yp", *SLABS["wall_yp"], WALLC, 0.45)
    wall_yn = static_box("/World/box/wall_yn", *SLABS["wall_yn"], WALLC, 0.45)
    box_floor = static_box("/World/box/box_floor", *SLABS["box_floor"], WALLC, 1.0)
    support = static_box("/World/box/support", *SLABS["support"], (0.22, 0.22, 0.26), 1.0)
    pedestal = static_box("/World/vis/pedestal", (-0.08, -0.08, 0.0), (0.08, 0.08, PLATE_Z), (0.4, 0.4, 0.42), 1.0)
    rim_xp = static_box("/World/vis/rim_xp", (BOX_CX + HX - RT, -HY - WALL, BOX_TOP - RT), (BOX_CX + HX + WALL, HY + WALL, BOX_TOP), RIM, 1.0)
    rim_xn = static_box("/World/vis/rim_xn", (BOX_CX - HX - WALL, -HY - WALL, BOX_TOP - RT), (BOX_CX - HX + RT, HY + WALL, BOX_TOP), RIM, 1.0)
    rim_yp = static_box("/World/vis/rim_yp", (BOX_CX - HX, HY - RT, BOX_TOP - RT), (BOX_CX + HX, HY + WALL, BOX_TOP), RIM, 1.0)
    rim_yn = static_box("/World/vis/rim_yn", (BOX_CX - HX, -HY - WALL, BOX_TOP - RT), (BOX_CX + HX, -HY + RT, BOX_TOP), RIM, 1.0)
    post_a = static_box("/World/vis/post_a", (BOX_CX + HX - RT, -HY - WALL, BOX_BOT), (BOX_CX + HX + WALL, -HY + RT, BOX_TOP), RIM, 1.0)
    post_b = static_box("/World/vis/post_b", (BOX_CX + HX - RT, HY - RT, BOX_BOT), (BOX_CX + HX + WALL, HY + WALL, BOX_TOP), RIM, 1.0)
    post_c = static_box("/World/vis/post_c", (BOX_CX - HX - WALL, -HY - WALL, BOX_BOT), (BOX_CX - HX + RT, -HY + RT, BOX_TOP), RIM, 1.0)
    post_d = static_box("/World/vis/post_d", (BOX_CX - HX - WALL, HY - RT, BOX_BOT), (BOX_CX - HX + RT, HY + WALL, BOX_TOP), RIM, 1.0)
    cam_side: CameraCfg = cam_cfg("/World/CamSide", CAM_SIDE, FOCAL["side"])
    cam_top: CameraCfg = cam_cfg("/World/CamTop", CAM_TOP, FOCAL["top"])
    if SLAB_TOP is not None and SLAB_TOP > BOX_BOT + WALL + 0.002:   # 옵션 b: 미교란 하층 펠릿 시각(충돌 없음) — 클래스 본문 안이어야 configclass 필드가 됨
        pellet_slab = static_box("/World/vis/pellet_slab", (BOX_CX - HX, -HY, BOX_BOT + WALL), (BOX_CX + HX, HY, SLAB_TOP), (0.85, 0.75, 0.35), 1.0)

try: FONT = ImageFont.load_default(size=18)
except TypeError: FONT = ImageFont.load_default()
GRAB_LABEL = "opaque" if args.grab_opacity >= 1.0 else "hidden (x-ray, opacity %.2f)" % args.grab_opacity
COLORS = {"rest": (0.92, 0.86, 0.70), "carried": (0.70, 0.20, 0.85), "captured": (0.95, 0.45, 0.10)}   # 보라 = 딸려갔지만 미포획(문 파랑과 구분), 주황 = 포획


def lens_proto(stage, path, color):
    parts = [trimesh.creation.icosphere(subdivisions=1, radius=float(r)).apply_translation(o) for o, r in zip(OFFS, RADII)]
    m = trimesh.util.concatenate(parts); mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(m.vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * int(len(m.faces)))); mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(v) for v in m.faces.reshape(-1)]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(m.vertex_normals, dtype=np.float32))); mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none); b = m.bounds
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in b[0]]), Gf.Vec3f(*[float(v) for v in b[1]])]))
    mat = path.rsplit("/", 1)[0] + "_mat"; spawn_preview_surface(mat, sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6)); bind_visual_material(path, mat)
    return dict(n_vertices=int(len(m.vertices)), n_faces=int(len(m.faces)), extent_m=b.tolist())


def make_instancer(stage, name, n, color):
    path = f"/World/pellets/{name}"; pi = UsdGeom.PointInstancer.Define(stage, path); info = lens_proto(stage, path + "/proto", color)
    pi.CreatePrototypesRel().SetTargets([Sdf.Path(path + "/proto")]); pi.CreateProtoIndicesAttr(Vt.IntArray([0] * int(n)))
    pi.CreatePositionsAttr(); pi.CreateOrientationsAttr(); pi.CreateExtentAttr(); return pi, info


def set_instancer(pi, pos, quat_xyzw):
    pi.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pos, dtype=np.float32)))
    pi.GetOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(float(q[3]), float(q[0]), float(q[1]), float(q[2])) for q in quat_xyzw]))
    lo, hi = pos.min(0) - 0.005, pos.max(0) + 0.005; pi.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in lo]), Gf.Vec3f(*[float(v) for v in hi])]))


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device)); scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    stage = omni.usd.get_context().get_stage(); INST = {}
    for name, ids in (("rest", REST), ("carried", CAR), ("captured", CAP)):
        if len(ids) == 0: continue
        pi, info = make_instancer(stage, name, len(ids), COLORS[name]); INST[name] = (pi, ids); RES.setdefault("instancers", {})[name] = dict(n=int(len(ids)), path=f"/World/pellets/{name}", **info)
    def set_pile(i):
        P = TL["clump_pos_m"][i].astype(np.float64) + ORIGIN; Q = TL["clump_quat_xyzw"][i]
        for name, (pi, ids) in INST.items(): set_instancer(pi, P[ids], Q[ids])
    if args.grab_opacity < 1.0:   # 닫힌 보울 안의 포획 펠릿을 보이게. 전부 렌더 시작 전(sim.reset 전) 세션 레이어 편집 — USD 파일 무수정. 효과는 스크린샷으로 확인
        ov = RES["grab_visual_override"] = {"opacity": args.grab_opacity, "deinstanced": [], "bound": [], "method": "visuals instanceable 해제 + PreviewSurface(opacity) 바인딩(벽과 같은 재질 경로). OmniPBR opacity 편집은 부품이 완전 소실(cutout)되어 폐기(run3)"}
        for link, col in (("grab_fixed", (0.25, 0.70, 0.35)), ("gripper_link", (0.20, 0.45, 0.85))):     # 색 = USD 원래 OmniPBR diffuse 와 동일
            vp = f"/World/envs/env_0/Robot/{link}/visuals"; prim = stage.GetPrimAtPath(vp)
            if prim and prim.IsInstanceable(): prim.SetInstanceable(False); ov["deinstanced"].append(vp)     # ① instanceable 참조 해제 → 바인딩 가능
            mat = f"/World/Looks/grab_{link}"; spawn_preview_surface(mat, sim_utils.PreviewSurfaceCfg(diffuse_color=col, roughness=0.5, opacity=args.grab_opacity))
            try: bind_visual_material(vp, mat); ov["bound"].append(vp)
            except Exception as e: ov.setdefault("bind_errors", []).append(f"{vp}: {e}")
    set_pile(IDX[0]); sim.reset()
    robot: Articulation = scene["robot"]; cams = {"side": scene["cam_side"], "top": scene["cam_top"]}
    jn = list(robot.joint_names); bn = list(robot.body_names); ARM = [jn.index(n) for n in ARM_JOINTS]; iD = jn.index(DOOR_JOINT); i5 = bn.index("link5")
    RES.update({"joint_names": jn, "body_names": bn, "cameras": {"side": {"pos": CAM_SIDE[0], "target": CAM_SIDE[1], "focal_mm": FOCAL["side"], "dist_m": round(float(np.linalg.norm(np.array(CAM_SIDE[0]) - np.array(CAM_SIDE[1]))), 4)},
                                                                 "top": {"pos": CAM_TOP[0], "target": CAM_TOP[1], "focal_mm": FOCAL["top"], "dist_m": round(float(np.linalg.norm(np.array(CAM_TOP[0]) - np.array(CAM_TOP[1]))), 4)}, "res": [W, H]},
                "physx_joint_limits_deg": {n: [round(math.degrees(float(lo)), 2), round(math.degrees(float(hi)), 2)] for n, (lo, hi) in zip(jn, robot.data.joint_pos_limits[0].cpu().numpy())}})
    dt = sim.get_physics_dt(); RES["stage"] = "render"
    def set_pose(i):
        q = torch.zeros(1, robot.num_joints, device=sim.device)
        for j, idx in enumerate(ARM): q[0, idx] = math.radians(IK[i]["q5_deg"][j])
        q[0, iD] = math.radians(IK[i]["door_deg"])
        robot.write_joint_state_to_sim(q, torch.zeros_like(q)); robot.set_joint_position_target(q); scene.write_data_to_sim()
    def body_pose(b):
        p = robot.data.body_link_pos_w[0, bn.index(b)].cpu().numpy().astype(np.float64); q = robot.data.body_link_quat_w[0, bn.index(b)].cpu().numpy()
        return p, Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    APERTURE_H = 20.955                                   # PinholeCameraCfg 기본 horizontal_aperture (mm)
    def cam_basis(pos, tgt):
        f = np.asarray(tgt, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
        up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.99 else np.array([1.0, 0, 0]); y = np.cross(up, f); y /= np.linalg.norm(y); z = np.cross(f, y)
        return np.asarray(pos, float), np.stack([f, y, z], 1)  # 열 = 카메라 축(전방, 좌, 상) — look_at_quat 과 동일 구성
    CAMB = {"side": cam_basis(*CAM_SIDE), "top": cam_basis(*CAM_TOP)}
    def project(name, Pw):
        """해석적 핀홀 투영(카메라 포즈 = 내가 지정한 look_at, fx = W·f/aperture). 센서 버퍼(quat_w_ros) 는 0 으로 와서 쓰지 않음 — intrinsic 은 교차확인용 기록만."""
        c, Rc = CAMB[name]; pc = Rc.T @ (np.asarray(Pw, float) - c)
        if pc[0] <= 1e-6: return None
        fx = W * FOCAL[name] / APERTURE_H; return (float(W / 2 + fx * (-pc[1]) / pc[0]), float(H / 2 + fx * (-pc[2]) / pc[0]))
    try: RES["camera_intrinsics_sensor"] = {n: cams[n].data.intrinsic_matrices[0].cpu().numpy().round(3).tolist() for n in cams}
    except Exception as e: RES["camera_intrinsics_sensor"] = f"unavailable: {e}"
    RES["camera_intrinsics_analytic_fx"] = {n: round(W * FOCAL[n] / APERTURE_H, 3) for n in cams}
    set_pose(IDX[0])
    for _ in range(args.warmup): sim.step(render=True); scene.update(dt)
    frames, t_start = [], time.time(); K_NEAR = min(IDX, key=lambda i: abs(i - K_CAP))   # 포획 정지 프레임 = 렌더된 프레임 중 폐합 완료에 가장 가까운 것(stride 스모크 대비)
    for k, i in enumerate(IDX):
        set_pile(i); set_pose(i); sim.step(render=True); sim.render(); scene.update(dt)
        qp = robot.data.joint_pos[0].cpu().numpy(); p5, R5 = body_pose("link5"); lip169 = p5 + R5 @ L169; hinge = p5 + R5 @ H5; bowl_c = p5 + R5 @ C5
        err = float(np.linalg.norm(lip169 - TOOL_W[i])); ang = rot_angle_deg(R5, R_W); herr = float(np.linalg.norm(hinge - DOOR_W[i]))
        Pc = TL["clump_pos_m"][i][CAP].astype(np.float64) + ORIGIN; pl5 = (R5.T @ (Pc - p5).T).T
        in_cav = (np.hypot(pl5[:, 0] - C5[0], pl5[:, 2] - C5[2]) < BOWL_R_IN) & (np.abs(pl5[:, 1]) < CHEEK_HALF_Y)
        imgs, stds, means, marks = {}, {}, {}, {}
        for name, cam in cams.items():
            a = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8); imgs[name] = Image.fromarray(a)
            stds[name] = round(float(a.astype(np.float32).std()), 2); means[name] = round(float(a.astype(np.float32).mean()), 2)
            marks[name] = {"tool_w8": project(name, TOOL_W[i]), "lip_sim": project(name, lip169), "bowl_sim": project(name, bowl_c)}
        canvas = Image.new("RGB", (2 * W, H)); canvas.paste(imgs["side"], (0, 0)); canvas.paste(imgs["top"], (W, 0)); d = ImageDraw.Draw(canvas)
        th = np.linspace(0.0, 2 * np.pi, 49)
        rings = [np.stack([C5[0] + BOWL_R_IN * np.cos(th), np.full_like(th, yy), C5[2] + BOWL_R_IN * np.sin(th)], 1) @ R5.T + p5 for yy in (-CHEEK_HALF_Y, CHEEK_HALF_Y)]   # 보울 공동 윤곽(link5 x–z 원, 뺨 두 면) → 세계
        for name, ox in (("side", 0), ("top", W)):
            for ring in rings:
                pts = [project(name, q) for q in ring]
                if all(pts): d.line([(ox + u, v) for u, v in pts], fill=(0, 255, 210), width=1)
            for j in range(0, 48, 12):
                a, b = project(name, rings[0][j]), project(name, rings[1][j])
                if a and b: d.line([(ox + a[0], a[1]), (ox + b[0], b[1])], fill=(0, 255, 210), width=1)
            m = marks[name]
            if m["tool_w8"]: u, v = m["tool_w8"]; d.line([(ox + u - 9, v), (ox + u + 9, v)], fill=(255, 230, 0), width=2); d.line([(ox + u, v - 9), (ox + u, v + 9)], fill=(255, 230, 0), width=2)
            if m["lip_sim"]: u, v = m["lip_sim"]; d.ellipse([ox + u - 6, v - 6, ox + u + 6, v + 6], outline=(0, 230, 255), width=2)
        d.text((6, 4), f"W9 f{k:02d}/{len(IDX)} (tl {i:02d}/{N_ALL})  t={TL['t_s'][i]:.2f}s  {IK[i]['phase']}  door={math.degrees(float(qp[iD])):.1f}deg  lip err={1000*err:.2f}mm  R err={ang:.2f}deg  captured in bowl {int(in_cav.sum())}/{len(CAP)}", fill=(255, 255, 0), font=FONT)
        d.text((6, H - 22), f"+ = W8 tool(lip 169.6)   o = sim lip   pellets: beige rest / purple carried / orange captured   grab parts {GRAB_LABEL}   cyan wire = bowl cavity (r 20 mm, cheeks ±18.2 mm) from sim link5", fill=(255, 255, 255), font=FONT)
        fn = os.path.join(FR, f"f_{k:05d}.png"); canvas.save(fn)
        if i == K_NEAR:
            imgs["side"].save(os.path.join(OUT, "capture_side_raw.png")); imgs["top"].save(os.path.join(OUT, "capture_top_raw.png"))
        if i == IDX[-1]:
            imgs["side"].save(os.path.join(OUT, "final_side_raw.png")); imgs["top"].save(os.path.join(OUT, "final_top_raw.png"))
        frames.append({"k": k, "i": i, "t_s": round(float(TL["t_s"][i]), 4), "phase": IK[i]["phase"], "q_sim_deg": [round(math.degrees(float(qp[j])), 3) for j in ARM], "door_sim_deg": round(math.degrees(float(qp[iD])), 3), "door_tl_deg": IK[i]["door_deg"],
                       "lip169_sim_w": [round(float(v), 5) for v in lip169], "tool_w8_w": [round(float(v), 5) for v in TOOL_W[i]], "lip_err_mm": round(1000 * err, 3), "R_err_deg": round(ang, 3), "hinge_err_mm": round(1000 * herr, 3),
                       "captured_in_cavity": int(in_cav.sum()), "captured_total": int(len(CAP)), "bowl_center_sim_w": [round(float(v), 5) for v in bowl_c], "img_std": stds, "img_mean": means, "marks_px": marks, "file": os.path.basename(fn)})
        print(f"[w9] frame {k}/{len(IDX)} tl {i} {IK[i]['phase']} lip_err {1000*err:.2f} mm R {ang:.2f} deg in_cav {int(in_cav.sum())}/{len(CAP)} std {stds}", flush=True)
    RES["frames"] = frames; RES["render_seconds"] = round(time.time() - t_start, 1); RES["stage"] = "media"
    json.dump(RES, open(os.path.join(OUT, "gates_w9.json"), "w"), indent=1, ensure_ascii=False)
    # ── 매체: mp4 · 12장 strip · 포획 정지 프레임(측면 + 보울 중심 3× 인셋) ──
    mp4 = os.path.join(OUT, "render_w9.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", os.path.join(FR, "f_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4], check=True)
    sel = sorted(set(int(round(v)) for v in np.linspace(0, len(frames) - 1, 12))); tw, th = (2 * W) // 3, H // 3
    strip = Image.new("RGB", (4 * tw, 3 * th), (20, 20, 20))
    for n, k in enumerate(sel):
        im = Image.open(os.path.join(FR, frames[k]["file"])).resize((tw, th), Image.LANCZOS); strip.paste(im, ((n % 4) * tw, (n // 4) * th))
        ImageDraw.Draw(strip).text(((n % 4) * tw + 4, (n // 4) * th + th - 20), f"#{n+1} f{k} {frames[k]['phase']} t={frames[k]['t_s']:.2f}s", fill=(255, 255, 0), font=FONT)
    strip.save(os.path.join(OUT, "keyframe_strip_w9.png"))
    def make_still(i_sel, raw_name, out_name, label):
        fc = next(f for f in frames if f["i"] == i_sel); side = Image.open(os.path.join(OUT, raw_name)); still = Image.open(os.path.join(FR, fc["file"])).copy(); bm = fc["marks_px"]["side"]["bowl_sim"]
        if bm:
            cw, ch = 200, 125; u, v = int(bm[0]), int(bm[1]); box = (max(0, u - cw // 2), max(0, v - ch // 2), min(W, u + cw // 2), min(H, v + ch // 2))
            inset = side.crop(box).resize((cw * 3, ch * 3), Image.LANCZOS); still.paste(inset, (2 * W - cw * 3 - 8, H - ch * 3 - 8)); ImageDraw.Draw(still).rectangle([2 * W - cw * 3 - 8, H - ch * 3 - 8, 2 * W - 8, H - 8], outline=(255, 255, 0), width=2)
            ImageDraw.Draw(still).text((2 * W - cw * 3 - 4, H - ch * 3 - 30), f"inset 3x: side cam around sim bowl center ({label} tl {i_sel}, door {fc['door_sim_deg']:.1f} deg, in bowl {fc['captured_in_cavity']}/{fc['captured_total']})", fill=(255, 255, 0), font=FONT)
        still.save(os.path.join(OUT, out_name)); return fc
    fc = make_still(K_NEAR, "capture_side_raw.png", "capture_frame_w9.png", "capture frame"); make_still(IDX[-1], "final_side_raw.png", "final_frame_w9.png", "final frame")
    # ── 게이트 ──
    G = {}
    G["G1_frame_count"] = {"n_rendered": len(frames), "n_timeline": N_ALL, "pass": bool(len(frames) == N_ALL), "note": "frame_stride>1 스모크면 FAIL 이 정상"}
    G["G2_no_blank_frames"] = {"min_std": {c: min(f["img_std"][c] for f in frames) for c in cams}, "min_mean": {c: min(f["img_mean"][c] for f in frames) for c in cams}, "threshold_std": 3.0, "pass": bool(all(f["img_std"][c] >= 3.0 for f in frames for c in cams))}
    G["G3_lip_error"] = {"max_mm": max(f["lip_err_mm"] for f in frames), "mean_mm": round(float(np.mean([f["lip_err_mm"] for f in frames])), 3), "limit_mm": 5.0, "max_R_err_deg": max(f["R_err_deg"] for f in frames), "max_hinge_err_mm": max(f["hinge_err_mm"] for f in frames), "pass": bool(max(f["lip_err_mm"] for f in frames) <= 5.0)}
    fin = frames[-1]; frac_final = fin["captured_in_cavity"] / fin["captured_total"]
    G["G4_captured_in_bowl_numeric"] = {"rule": "W8 captured_ids 는 최종 프레임 보울 공동 판정(in_cav: 보울 반경 20 mm·뺨 반폭 18.2 mm, link5 프레임). 같은 식을 시뮬 link5 포즈(렌더된 로봇)로 최종 프레임에 재적용 — 렌더 자세가 W8 툴과 같으면 154/154 재현", "k_capture": K_CAP, "k_capture_rendered": K_NEAR,
                                        "final_frame": f"{fin['captured_in_cavity']}/{fin['captured_total']}", "final_fraction": round(frac_final, 4), "threshold_fraction": 0.95, "pass": bool(frac_final >= 0.95),
                                        "trajectory_after_close_info": {str(f["i"]): f["captured_in_cavity"] for f in frames if f["i"] >= K_CAP}, "trajectory_note": "폐합 직후에는 일부가 공동 경계 밖(문 틈·립 근처)에 있다가 리프트 중 안으로 모임 — W8 정의상 포획은 최종 프레임 기준이므로 게이트 아님(정보)"}
    G["G4_captured_in_bowl_visual"] = {"screenshot": os.path.join(OUT, "capture_frame_w9.png"), "raw_side": os.path.join(OUT, "capture_side_raw.png"), "status": "PENDING_HUMAN_LOOK", "pass": None}
    G["media"] = {"mp4": mp4, "mp4_bytes": os.path.getsize(mp4), "fps": args.fps, "strip": os.path.join(OUT, "keyframe_strip_w9.png"), "strip_frames": sel, "capture_frame": os.path.join(OUT, "capture_frame_w9.png"), "final_frame": os.path.join(OUT, "final_frame_w9.png"), "frames_dir": FR}
    RES["gates"] = G; RES["all_pass_numeric"] = bool(all(G[k]["pass"] for k in ("G1_frame_count", "G2_no_blank_frames", "G3_lip_error", "G4_captured_in_bowl_numeric"))); RES["ok"] = True; RES["stage"] = "done"
    print("[w9]", json.dumps({k: G[k] for k in G if k != "media"}, ensure_ascii=False), flush=True)


try:
    main()
except Exception:
    import traceback; RES["error"] = traceback.format_exc(); print("[w9] ERROR", RES["error"][-2000:], flush=True)
json.dump(RES, open(os.path.join(OUT, "gates_w9.json"), "w"), indent=1, ensure_ascii=False)
import threading
threading.Thread(target=lambda: (time.sleep(20), os._exit(0)), daemon=True).start()
simulation_app.close()
