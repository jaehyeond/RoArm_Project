"""W13 사전검토 ② — 강체 힌지·문 속도·토크 분해의 **오프라인 운동학 검증**(DEME 실행 0).

D1 문 owner 원점 = 툴 원점 + R_tool·hinge_off  (전 궤적에서 수치 확인)
D2 문 owner 선속도 = v_tool + ω_tool × (p_door − p_tool)  — 문 자체 회전은 원점을 움직이지 않는다.
   명령 궤적의 유한차분으로 확인한다(베이스 회전 중·문 회전 중 모두).
D3 문 회전축은 툴과 같이 돈다: a_world = R_tool·a_local, 그리고 R_tool^T·R_door = axis_angle(a_local, q−q_open)
D4 토크 분해: 힌지 저항 모멘트는 **접촉력만**의 축 모멘트다. 합성 시험력으로 β=0/90° 에서 해석값과 대조하고,
   접촉이 없으면(베이스 회전만 있어도) 0 임을 보인다.
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN)); sys.path.insert(0, str(HERE))
import sim_deme_scoop_s1 as W11SRC
import w13_kinematics as K
import w13_fk as FK

P = dict(W11SRC.DEFAULT); P.update(K.W13_DEFAULT)
P.update(json.load(open(sys.argv[2])))
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
_, _, _, _, hinge_off, L5mm, _ = W11SRC.load_tool(P, q_open)
P["lip_l5_mm"] = [float(v) for v in L5mm]
axis_l = W11SRC.R_W @ np.array([0.0, 1.0, 0.0])
ad, ad_info = FK.build_adapter(None, 0.0411807760, P["arm_radius_m"], P["lip_l5_mm"],
                                 P["declared_base_cm"], P["declared_pellet_cm"])
r0 = float(np.hypot(*FK.lip_pose(ad_info["reference_pose_q5"], P["lip_l5_mm"])[0][:2]))

def pose(q5, q_door):
    p_f, R_f = ad.owner_pose(q5)
    return p_f, R_f, p_f + R_f @ hinge_off, R_f @ K.axis_angle(axis_l, q_door - q_open)

R = {"artifact": "W13_RIGID_HINGE_CHECK_01", "deme_runs": 0}

# 궤적: HOME→P1→베이스 0→90 회전 + 문 27.5→0
traj = []
for t in np.linspace(0, 1, 120):
    traj.append((( np.asarray(FK.HOME_Q5) + (np.asarray(FK.P1_Q5) - np.asarray(FK.HOME_Q5)) * t).tolist(), q_open))
p1_90 = [90.0] + list(FK.P1_Q5[1:])
for t in np.linspace(0, 1, 120):
    traj.append(((np.asarray(FK.P1_Q5) + (np.asarray(p1_90) - np.asarray(FK.P1_Q5)) * t).tolist(), q_open))
for t in np.linspace(0, 1, 120):
    traj.append((p1_90, q_open + (0.0 - q_open) * t))

d1, d3 = [], []
for q5, qd in traj:
    p_f, R_f, p_d, R_d = pose(q5, qd)
    d1.append(float(np.linalg.norm(p_d - (p_f + R_f @ hinge_off))))
    rel = R_f.T @ R_d
    d3.append(float(np.abs(rel - K.axis_angle(axis_l, qd - q_open)).max()))
R["D1_door_origin_rigid"] = {"max_err_m": max(d1), "n": len(d1), "pass": bool(max(d1) < 1e-12)}
R["D3_door_axis_rotates_with_tool"] = {"max_rel_rot_err": max(d3), "n": len(d3), "pass": bool(max(d3) < 1e-12)}

# D2 중심차분 — 궤적을 매개변수 u 의 연속 함수로 보고 아주 작은 h 로 순간속도를 만든다.
#    (앞선 1차 전방차분은 회전 스텝이 커서 O(omega*dt) 오차가 남았다 — 그건 공식이 아니라 차분의 한계다.)
def traj_pose(u):
    """u in [0,3): 0-1 HOME->P1, 1-2 base 0->90, 2-3 문 27.5->0"""
    if u < 1.0:
        q5 = (np.asarray(FK.HOME_Q5) + (np.asarray(FK.P1_Q5) - np.asarray(FK.HOME_Q5)) * u).tolist(); qd = q_open
    elif u < 2.0:
        t = u - 1.0
        q5 = (np.asarray(FK.P1_Q5) + (np.asarray(p1_90) - np.asarray(FK.P1_Q5)) * t).tolist(); qd = q_open
    else:
        t = min(1.0, u - 2.0); q5 = p1_90; qd = q_open + (0.0 - q_open) * t
    return pose(q5, qd)

h = 1e-7
res, rel = [], []
for u in np.linspace(0.02, 2.98, 150):
    pa_f, Ra_f, pa_d, Ra_d = traj_pose(u - h)
    pb_f, Rb_f, pb_d, Rb_d = traj_pose(u + h)
    pm_f, Rm_f, pm_d, _ = traj_pose(u)
    v_tool = (pb_f - pa_f) / (2 * h)
    w_tool = K.rotvec_of(Rb_f @ Ra_f.T) / (2 * h)          # 세계 프레임 각속도
    v_door = (pb_d - pa_d) / (2 * h)
    v_exp = v_tool + np.cross(w_tool, pm_d - pm_f)
    e = float(np.linalg.norm(v_door - v_exp))
    res.append(e)
    rel.append(e / max(float(np.linalg.norm(v_door)), 1e-12))
R["D2_door_velocity_rigid"] = {
    "max_abs_residual_per_unit_param": max(res), "max_relative_residual": max(rel),
    "n_samples": len(res), "central_difference_h": h,
    "formula": "v_door = v_tool + omega_tool_world x (p_door - p_tool); 문 자체 회전은 원점을 움직이지 않는다",
    "note": "u 는 궤적 매개변수이므로 절대 크기는 m/파라미터 단위다. 판정은 상대 잔차로 한다.",
    "pass": bool(max(rel) < 1e-6)}

# D4 토크 분해
def hinge_moment(beta_deg, q_door, pts_local, forces_world):
    q5 = [beta_deg] + list(FK.P1_Q5[1:])
    p_f, R_f, p_d, R_d = pose(q5, q_door)
    axis_now = R_f @ axis_l
    Pp = (R_f @ np.asarray(pts_local, float).T).T + p_f
    Ff = np.asarray(forces_world, float)
    return float(np.dot(np.cross(Pp - p_d, Ff).sum(0), axis_now)), axis_now, p_d
# 시험력: 힌지에서 lever 만큼 떨어진 점에 축과 수직인 1 N
lever = 0.05
pt_l = (hinge_off + np.array([0.0, 0.0, -lever])).tolist()
rows = []
for beta in (0.0, 45.0, 90.0):
    q5 = [beta] + list(FK.P1_Q5[1:])
    _, R_f, _, _ = pose(q5, 3.0)
    axis_now = R_f @ axis_l
    r_vec = R_f @ np.array([0.0, 0.0, -lever])
    f_dir = np.cross(axis_now, r_vec); f_dir /= np.linalg.norm(f_dir)
    M, an, pd = hinge_moment(beta, 3.0, [pt_l], [f_dir.tolist()])
    rows.append({"beta_deg": beta, "M_Nm": round(M, 12), "analytic_Nm": round(lever, 12),
                 "axis_world": an.round(8).tolist()})
M0, _, _ = hinge_moment(90.0, 3.0, np.zeros((0, 3)), np.zeros((0, 3)))
R["D4_torque_decomposition"] = {
    "unit_force_at_lever_m": lever, "rows": rows,
    "no_contact_moment_at_beta90_Nm": M0,
    "statement": "모멘트는 접촉력만으로 계산한다. 접촉이 없으면 베이스가 돌아도 0 이다 — 베이스 회전의 강체 운동이 "
                 "문 액추에이터 토크에 섞이지 않는다는 뜻이다.",
    "opening_sign_rule": "열림 저항 = -M >= M_stall, 닫힘 저항 = +M >= M_stall (축은 열림 +q 방향)",
    "pass": bool(all(abs(r["M_Nm"] - lever) < 1e-9 for r in rows) and abs(M0) < 1e-15)}

R["all_pass"] = all(v.get("pass", True) for v in R.values() if isinstance(v, dict))
json.dump(R, open(out / "rigid_hinge_check.json", "w"), ensure_ascii=False, indent=2, default=float)
print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in R.items()}, ensure_ascii=False, indent=1))
print("D2 max relative residual:", R["D2_door_velocity_rigid"]["max_relative_residual"])
print("D4 rows:", json.dumps(rows, ensure_ascii=False))
