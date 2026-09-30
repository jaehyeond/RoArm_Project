"""rev36-chain (W26): 셀 위치 입력 — 명령점(상자 좌표) → 립 자리·툴 회전 + 벽 여유 검사. 순수 CPU.

규약 (실물과 같게)
    실물은 `goto_xyz` 로 명령한다: 베이스 = atan2(y, x), 나머지 관절 = 평면 수직해(`w13_fk.solve_xyz`).
    rev34 의 취점 자리도 같은 규약이다 — 상자 중심을 명령했을 때 **FK 립**이 떨어지는 상자 좌표
    (`w13_fk.w25_scoop_site`). 이 모듈은 그 명령점을 상자 안 임의 (x, y) 로 일반화한다.
    · 립 자리 = 명령점의 로봇 좌표로 `solve_xyz` 를 풀고 FK 립을 상자 좌표로 옮긴 점(xy).
    · 툴 회전 = rev34 취점 회전(R_box_robot)에 베이스 회전 Rz(b) 를 곱한 것.
      수직 툴은 베이스만 돌면 같은 자세가 z 축으로 돈다. 명령점 = 상자 중심이면 b = 0 이라 rev34 와 **같은 행렬**.
      FK 로 구한 실제 owner 회전과의 차이는 `R_fk_minus_ideal_max` 로 보고한다(판정 아님).
    · 명령점 = 상자 중심이면 `w25_scoop_site` 와 같은 립 자리를 내야 한다(`self_check_center`).

손목 회전(2026-09-30 사용자 결정 "손목 회전 허용")
    roll_deg = 손목 롤 관절(q5[4], 한계 ±90°). FK 실측: 툴 yaw(상자 좌표) = 기준 yaw + 베이스 − 롤, 립은 롤 축 둘레
    반경 8.1 mm 원 위를 돈다. 툴 회전 = R_box_robot · Rz(베이스 − 롤). roll 0 = 위 규약 그대로(rev34 와 동일).

벽 여유 (fail-closed)
    셀 동안 툴은 이 자리에서 수직으로만 움직인다. 고정부 셸과 문 셸(닫힘·열림 두 각)의 모든 정점 xy 가
    트레이 안쪽 경계 안에 `margin` 이상 떨어져 있어야 한다. 아니면 물리 전에 멈춘다.
    이 검사는 **기하 선행 조건**이고, 벽 근처 퍼내기의 물리를 판정하지 않는다.
"""
import math

import numpy as np

import w13_fk as FK
import w13_kinematics as K


def Rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ik(P, lip_l5_owner_mm, cmd_box_xy_m):
    conv, R, anchor = FK.w25_frame(P)
    if R is None or anchor != "declared_box_center":
        raise SystemExit("w26_cell_cmd_box_xy_m 는 rev34 상자 규약(A/B + declared_box_center)에서만 쓴다")
    cmd = np.array([float(cmd_box_xy_m[0]), float(cmd_box_xy_m[1]), 0.0])
    sol_ref, p_lip_ref, _, pellet_robot_z = FK._w25_reference(P, lip_l5_owner_mm)
    t_xy = FK._w25_t_xy(P, anchor, p_lip_ref)
    ad0 = FK.Adapter([t_xy[0], t_xy[1], 0.0], lip_l5_owner_mm, R)
    cmd_robot = R @ cmd + np.array([t_xy[0], t_xy[1], 0.0])
    sol = FK.solve_xyz(float(cmd_robot[0]), float(cmd_robot[1]), pellet_robot_z, lip_l5_owner_mm)
    if sol is None:
        raise SystemExit(f"셀 위치 IK 실패(도달 불가): 명령 상자 xy {cmd[:2].tolist()} m")
    return R, ad0, cmd, cmd_robot, sol


def _pose(R, ad0, cmd, cmd_robot, sol, roll_deg):
    q5 = list(sol["q5"]); q5[4] = float(roll_deg)
    viol = FK.in_limits(q5)
    if viol:
        raise SystemExit(f"셀 위치 관절 제한 위반: {viol} (명령 상자 xy {cmd[:2].tolist()} m, 롤 {roll_deg}°)")
    p_lip_w, R_fk = ad0.owner_pose(q5)
    base_deg = float(q5[0])
    R_cell = R.T @ Rz(base_deg - float(roll_deg))
    info = {"cmd_box_xy_m": cmd[:2].tolist(), "cmd_robot_xy_m": cmd_robot[:2].tolist(),
            "base_deg": base_deg, "roll_deg": float(roll_deg), "q5_deg": [float(v) for v in q5],
            "tool_yaw_box_deg": float(math.degrees(math.atan2(R_cell[1, 0], R_cell[0, 0]))),
            "ik_err_m": sol.get("err_m"), "tilt_deg": sol.get("tilt_deg"),
            "lip_box_xy_m": [float(p_lip_w[0]), float(p_lip_w[1])],
            "lip_minus_cmd_mm": [float((p_lip_w[0] - cmd[0]) * 1000), float((p_lip_w[1] - cmd[1]) * 1000)],
            "R_cell_owner_box": R_cell.tolist(),
            "R_fk_minus_ideal_max": float(np.abs(R_fk - R_cell).max()),
            "rule": "goto_xyz 규약(베이스 atan2·평면 수직해)으로 명령점을 풀어 FK 립 xy 를 셀 자리로, "
                    "rev34 취점 회전에 Rz(베이스 − 손목 롤)을 곱한 것을 툴 회전으로 쓴다(롤 0 = rev34)"}
    return (float(p_lip_w[0]), float(p_lip_w[1])), R_cell, info


def cell_site(P, lip_l5_owner_mm, cmd_box_xy_m, roll_deg=0.0):
    return _pose(*_ik(P, lip_l5_owner_mm, cmd_box_xy_m), roll_deg)


def cell_site_rolls(P, lip_l5_owner_mm, cmd_box_xy_m, rolls_deg):
    """IK 는 한 번, 롤별 자세는 FK 만 — 롤 탐색용. 반환 {roll: (site, R_cell, info) 또는 SystemExit 메시지}."""
    ik = _ik(P, lip_l5_owner_mm, cmd_box_xy_m)
    out = {}
    for r in rolls_deg:
        try:
            out[float(r)] = _pose(*ik, r)
        except SystemExit as e:
            out[float(r)] = str(e)
    return out


def self_check_center(P, lip_l5_owner_mm):
    """명령점 = 상자 중심 → rev34 `w25_scoop_site` 와 같은 자리·같은 회전인지(binary64)."""
    (x, y), R_cell, info = cell_site(P, lip_l5_owner_mm, (0.0, 0.0))
    site_ref, _ = FK.w25_scoop_site(P, lip_l5_owner_mm)
    _, R, _ = FK.w25_frame(P)
    return {"site_equal": [x, y] == list(site_ref), "R_equal": bool(np.array_equal(R_cell, R.T)),
            "site_cell": [x, y], "site_rev34": list(site_ref), "base_deg": info["base_deg"]}


def wall_clearance(fixed_v, door_v, hinge_off, axis_w, q_open, q_list_deg, site_xy, z_list_m, R_cell, box):
    """셸 정점 xy 의 트레이 안쪽 네 벽까지 최소 여유(m). 음수 = 벽을 뚫음."""
    pts = []
    for z in z_list_m:
        p = np.array([site_xy[0], site_xy[1], z], float)
        pts.append((R_cell @ np.asarray(fixed_v, float).T).T + p)
        for q in q_list_deg:
            Rd = R_cell @ K.axis_angle(axis_w, q - q_open)
            pts.append((Rd @ np.asarray(door_v, float).T).T + p + R_cell @ hinge_off)
    X = np.vstack(pts)
    m = {"x_lo": float(X[:, 0].min() - box[0, 0]), "x_hi": float(box[0, 1] - X[:, 0].max()),
         "y_lo": float(X[:, 1].min() - box[1, 0]), "y_hi": float(box[1, 1] - X[:, 1].max())}
    return min(m.values()), m
