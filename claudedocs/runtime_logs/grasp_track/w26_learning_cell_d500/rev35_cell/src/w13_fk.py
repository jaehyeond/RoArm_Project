"""W13 오프라인 FK/IK — 실물 RoArm-M3 관절에서 S1 립 포즈를 내고, 전 사이클 경로의 로봇 실현 가능성을 증명한다.

하드웨어 스크립트를 **import 하지 않는다**(계약). 아래 상수는 저장된 소스를 **텍스트로 읽어 옮겨 적은 것**이며
출처를 줄 단위로 밝힌다. 값이 바뀌면 `verify_source_constants()` 가 원문을 다시 파싱해 불일치를 잡는다.

출처 (모두 메인 repo, 읽기 전용)
    `sim_scripts/roarm_kinematics.py:18-27`   URDF 관절 체인 `_CHAIN` (roarm_m3.urdf 에서 추출, 4/28)
    `sim_scripts/roarm_kinematics.py:29-36`   `JOINT_LIMITS_DEG` (v6 분포 기반 클립)
    `sim_scripts/roarm_kinematics.py:38-58`   `rpy_R` / `Tmat` / `Trot_z` 규약
    `hw_s1_scoop_probe.py:17-23`              `SHOULDER_ABOVE_PLATE`, `LIP_L5`, `WRIST_MAX`, `HOME`, `P1`
    `hw_s1_manual.py:31-47`                   툴 수직 격자 IK(`_grid`/`solve_fast`) 규약
    `hw_s1_manual.py:182-201`                 실물 `place()` 경로(P1 후퇴 → 베이스 회전 → 뻗기 → 하강 → 개폐 → 역순)

⚠️ 이 모듈은 **로봇 하드웨어를 건드리지 않는다.** 순수 수치 FK/IK 이며 시리얼·SDK 를 쓰지 않는다.
"""
import json
import math
import re
from pathlib import Path

import numpy as np

MAIN_REPO = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
SRC_KIN = MAIN_REPO / "sim_scripts/roarm_kinematics.py"
SRC_PROBE = MAIN_REPO / "hw_s1_scoop_probe.py"

PI = math.pi
# roarm_kinematics.py:18-27 — (name, xyz, rpy, joint_index)
CHAIN = [
    ("world_to_base", [0.0, 0.0, 0.0701], [0.0, 0.0, 0.0], None),
    ("base_to_link1", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0),
    ("link1_to_link2", [0.0, 0.0, 0.05196], [-PI / 2, -PI / 2, 0.0], 1),
    ("link2_to_link3", [0.236815, 0.030002, 0.0], [0.0, 0.0, PI / 2], 2),
    ("link3_to_link4", [0.0, -0.144586, 0.0], [0.0, 0.0, 0.0], 3),
    ("link4_to_link5", [0.015147, -0.053653, 0.0], [PI / 2, PI / 2, 0.0], 4),
    ("link5_to_tcp", [0.0, 0.0, 0.115428], [PI / 2, -PI / 2, 0.0], None),
]
# roarm_kinematics.py:29-36
JOINT_LIMITS_DEG = {"base": (-90.0, 90.0), "shoulder": (-30.0, 75.0), "elbow": (5.0, 135.0),
                    "wrist_p": (-30.0, 90.0), "wrist_r": (-90.0, 90.0), "gripper": (-10.0, 100.0)}
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_p", "wrist_r"]
# hw_s1_scoop_probe.py:17-23
SHOULDER_ABOVE_PLATE = 0.0701 + 0.05196
LIP_L5_PHYSICAL_MM = [8.1, 0.0, 166.6]      # 실물 S1 립 (FK 기준)
WRIST_MAX = 90.0
HOME_Q5 = [0.0, 0.0, 90.0, 0.0, 0.0]
P1_Q5 = [0.0, 0.7, 91.3, 88.0, 0.0]


def verify_source_constants():
    """원문 텍스트를 다시 파싱해 위 상수가 실제 소스와 같은지 확인한다(하드코딩 표류 방지)."""
    out = {"sources": {str(SRC_KIN): SRC_KIN.exists(), str(SRC_PROBE): SRC_PROBE.exists()}}
    kin = SRC_KIN.read_text()
    probe = SRC_PROBE.read_text()
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+", kin[kin.index("_CHAIN = ["):kin.index("# v6-derived")])]
    mine = [v for _, xyz, _, _ in CHAIN for v in xyz if isinstance(v, float) and v not in (0.0,)]
    out["chain_numeric_subset_present"] = all(any(abs(m - n) < 1e-12 for n in nums) for m in mine)
    out["joint_limits_match"] = all(f'"{k}"' in kin.replace("'", '"') or f"'{k}'" in kin for k in JOINT_LIMITS_DEG)
    for k, (lo, hi) in JOINT_LIMITS_DEG.items():
        m = re.search(rf'"{k}":\s*\(([-\d.+]+),\s*([-\d.+]+)\)', kin.replace("'", '"'))
        if m:
            out[f"limit_{k}"] = bool(abs(float(m.group(1)) - lo) < 1e-9 and abs(float(m.group(2)) - hi) < 1e-9)
    m = re.search(r"LIP_L5\s*=\s*np\.array\(\[([\d.]+),\s*([\d.]+),\s*([\d.]+)", probe)
    out["lip_l5_match"] = bool(m and abs(float(m.group(1)) * 1000 - LIP_L5_PHYSICAL_MM[0]) < 1e-6
                               and abs(float(m.group(3)) * 1000 - LIP_L5_PHYSICAL_MM[2]) < 1e-6)
    m = re.search(r"HOME\s*=\s*\[([^\]]+)\]", probe)
    out["home_match"] = bool(m and [float(v) for v in m.group(1).split(",")] == HOME_Q5)
    m = re.search(r"P1\s*=\s*\[([^\]]+)\]", probe)
    out["p1_match"] = bool(m and [float(v) for v in m.group(1).split(",")] == P1_Q5)
    m = re.search(r"SHOULDER_ABOVE_PLATE\s*=\s*([\d.]+)\s*\+\s*([\d.]+)", probe)
    out["shoulder_above_plate_match"] = bool(m and abs(float(m.group(1)) + float(m.group(2)) - SHOULDER_ABOVE_PLATE) < 1e-12)
    out["pass"] = all(v is True for k, v in out.items() if k not in ("sources",)) and all(out["sources"].values())
    return out


def rpy_R(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    return Rz @ Ry @ Rx


def _T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_R(*rpy)
    T[:3, 3] = xyz
    return T


def _Tz(q_rad):
    T = np.eye(4)
    c, s = math.cos(q_rad), math.sin(q_rad)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def link5_T(q5):
    """관절각(°, 5개) → link5 프레임의 4x4 변환(URDF world = 베이스판 기준)."""
    q = np.radians(list(q5) + [0.0])
    T = np.eye(4)
    for _, xyz, rpy, qi in CHAIN:
        T = T @ _T(xyz, rpy)
        if qi is not None:
            T = T @ _Tz(q[qi])
        if _ == "link4_to_link5":
            return T
    raise RuntimeError("link4_to_link5 없음")


def lip_pose(q5, lip_l5_mm=None):
    """반환 (p_lip_robot_m, R_link5_robot). 로봇 프레임 = 어깨축 원점(z 를 SHOULDER_ABOVE_PLATE 만큼 내린 것),
    x 앞·y 왼쪽·z 위. `hw_s1_scoop_probe.lip_fw` 와 같은 정의."""
    T = link5_T(q5)
    L = np.array(list(lip_l5_mm or LIP_L5_PHYSICAL_MM), float) / 1000.0
    p = T @ np.array([L[0], L[1], L[2], 1.0])
    return np.array([p[0], p[1], p[2] - SHOULDER_ABOVE_PLATE]), T[:3, :3]


def in_limits(q5):
    bad = []
    for i, n in enumerate(JOINT_NAMES):
        lo, hi = JOINT_LIMITS_DEG[n]
        if not (lo - 1e-9 <= q5[i] <= hi + 1e-9):
            bad.append({"joint": n, "value": float(q5[i]), "limit": [lo, hi]})
    if abs(q5[3]) > WRIST_MAX + 1e-9:
        bad.append({"joint": "wrist_p_firmware_clamp", "value": float(q5[3]), "limit": [-WRIST_MAX, WRIST_MAX]})
    return bad


def _grid(r_m, z_m, sh_rng, el_rng, step, tilt_ok, wrist_fixed=None):
    """`hw_s1_manual.py:31-41` `_grid` 를 그대로 옮겨 적은 것. 반환 (err_m, q5, verticality) 또는 None."""
    best = None
    for sh in np.arange(sh_rng[0], sh_rng[1] + 1e-9, step):
        for el in np.arange(el_rng[0], el_rng[1] + 1e-9, step):
            wp = wrist_fixed if wrist_fixed is not None else 88.0 + (0.7 - sh) + (91.32 - el)
            if abs(wp) > WRIST_MAX:
                continue
            q = [0.0, float(sh), float(el), float(wp), 0.0]
            l, R = lip_pose(q, _grid.lip_l5_mm)
            if -R[2, 2] < tilt_ok:
                continue
            e = float(math.hypot(l[0] - r_m, l[2] - z_m))
            if best is None or e < best[0]:
                best = (e, q, float(-R[2, 2]))
    return best


_grid.lip_l5_mm = None


def solve_fast(r_m, z_m, lip_l5_mm=None):
    """`hw_s1_manual.py:44-52` `solve_fast` 규약을 그대로 옮겨 적은 것.
    ① 툴 수직(2° 격자 → 0.25° 정밀) ② 오차 > 10 mm 면 손목 90 고정·기울임 ≤ 14°(cos ≥ 0.97) 허용.
    반환 dict(err_m, q5, verticality, branch, limits_violations) 또는 None."""
    _grid.lip_l5_mm = lip_l5_mm
    b = _grid(r_m, z_m, (-30, 110), (-10, 150), 2.0, 0.995)
    branch = "vertical"
    if b is not None:
        b2 = _grid(r_m, z_m, (b[1][1] - 2, b[1][1] + 2), (b[1][2] - 2, b[1][2] + 2), 0.25, 0.995)
        if b2 is not None and b2[0] < b[0]:
            b = b2
        b3 = _grid(r_m, z_m, (b[1][1] - 0.25, b[1][1] + 0.25), (b[1][2] - 0.25, b[1][2] + 0.25), 0.02, 0.995)
        if b3 is not None and b3[0] < b[0]:
            b = b3
    if b is None or b[0] > 0.01:
        t = _grid(r_m, z_m, (20, 110), (-10, 130), 1.0, 0.97, wrist_fixed=WRIST_MAX)
        if t is not None and (b is None or t[0] < b[0]):
            t2 = _grid(r_m, z_m, (t[1][1] - 1, t[1][1] + 1), (t[1][2] - 1, t[1][2] + 1), 0.05, 0.97, wrist_fixed=WRIST_MAX)
            b, branch = (t2 if (t2 is not None and t2[0] < t[0]) else t), "wrist90_tilt"
    _grid.lip_l5_mm = None
    if b is None:
        return None
    return {"err_m": b[0], "q5": b[1], "verticality": b[2], "branch": branch,
            "tilt_deg": math.degrees(math.acos(min(1.0, b[2]))), "limits_violations": in_limits(b[1])}


def solve_vertical(r_m, z_m, lip_l5_mm=None, **kw):
    """수직 분기만(어댑터 기준 자세용)."""
    _grid.lip_l5_mm = lip_l5_mm
    b = _grid(r_m, z_m, (-30, 110), (-10, 150), 1.0, 0.995)
    if b is not None:
        for step, half in ((0.1, 1.0), (0.01, 0.1)):
            b2 = _grid(r_m, z_m, (b[1][1] - half, b[1][1] + half), (b[1][2] - half, b[1][2] + half), step, 0.995)
            if b2 is not None and b2[0] < b[0]:
                b = b2
    _grid.lip_l5_mm = None
    if b is None:
        return None
    return {"err_m": b[0], "q5": b[1], "verticality": b[2], "branch": "vertical",
            "tilt_deg": math.degrees(math.acos(min(1.0, b[2]))), "limits_violations": in_limits(b[1])}


def solve_xyz(x_m, y_m, z_m, lip_l5_mm=None):
    """실물 `goto_xyz` 규약: 베이스 = atan2(y,x), 나머지는 평면 내 수직해."""
    b = math.degrees(math.atan2(y_m, x_m))
    if abs(b) > 90.0 + 1e-9:
        return None
    sol = solve_vertical(math.hypot(x_m, y_m), z_m, lip_l5_mm)
    if sol is None:
        return None
    q = list(sol["q5"])
    q[0] = b
    sol = dict(sol, q5=q, limits_violations=in_limits(q))
    return sol


class Adapter:
    """로봇 프레임 ↔ DEME 세계 프레임. **축은 평행**(아래 FK 증거)이고 원점만 평행이동한다.

    증거: 툴 수직 자세에서 FK 가 주는 link5 축의 로봇 좌표는
        link5 X → (0,−1,0), link5 Y → (−1,0,0), link5 Z → (0,0,−1)
    이며, 이는 동결 W11 소스의 `R_W` 열벡터와 **성분까지 같다**. 따라서 DEME 세계축 = 로봇 세계축이고
    회전 성분이 없다. (W13 이전에는 이것이 '선언'이었고 부호가 미확정이었다 — FK 로 확정했다.)

    원점: DEME 원점(더미 상자 바닥 중심) = 로봇 좌표 `t_robot`. 취점 자세의 립을 기준점으로 잡는다.
    """

    def __init__(self, t_robot_m, lip_l5_owner_mm, R_robot_box=None):
        self.t = np.asarray(t_robot_m, float)
        self.lip_l5_owner_mm = list(lip_l5_owner_mm)
        # rev34: None = rev32(축 평행, 원점만 이동). 행렬이면 p_robot = R @ p_box + t (열 = 상자축의 로봇 좌표).
        self.R = None if R_robot_box is None else np.asarray(R_robot_box, float)

    def to_world(self, p_robot):
        if self.R is None:
            return np.asarray(p_robot, float) - self.t
        return self.R.T @ (np.asarray(p_robot, float) - self.t)

    def to_robot(self, p_world):
        if self.R is None:
            return np.asarray(p_world, float) + self.t
        return self.R @ np.asarray(p_world, float) + self.t

    def owner_pose(self, q5):
        """관절각 → (DEME 세계 립-owner 위치, 툴 owner 회전행렬).

        owner 립은 **두꺼워진 충돌 셸의 립**(link5 z=169.6 mm)이고 실물 FK 립은 166.6 mm 다.
        그 3 mm 차이는 여기서 link5 프레임 안에서 정확히 흡수한다(고정 어댑터).
        owner 회전 = R_link5_robot · R_W^{-1} — 즉 W11 초기 자세에서 단위행렬이 되도록 잡은 것.
        """
        p_owner_robot, R_l5 = lip_pose(q5, self.lip_l5_owner_mm)
        if self.R is None:
            return self.to_world(p_owner_robot), R_l5 @ R_W_FROZEN.T
        return self.to_world(p_owner_robot), self.R.T @ R_l5 @ R_W_FROZEN.T


R_W_FROZEN = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)   # sim_deme_scoop_s1.R_W (동결)


def build_adapter(box_bounds_m, pile_surface_z_world_m, arm_box_x_m, lip_l5_owner_mm,
                  declared_base_cm=38.0, declared_pellet_cm=26.0):
    """취점 자세(베이스 0, 립을 상자 중심·펠릿면 높이)를 기준으로 원점 평행이동을 고정한다."""
    # 기준 자세: 실물이 쓰는 (반경 arm_box_x, 펠릿면 높이) — 높이는 로봇 좌표에서 자유롭게 잡을 수 있으므로
    # z 정렬은 "DEME 펠릿면 ↔ 로봇 펠릿면"으로 둔다. 로봇 펠릿면 z 는 실물 설정(바닥+26 cm)을 쓴다.
    # ⚠️ base_cm/pellet_cm 은 argparse 기본값이 **아니다**(required=True). hw_s1_manual.py:3 사용 예시 숫자를
    #    W13 이 declared_not_measured 픽스처로 채택한 것이다.
    floor_robot_z = -(declared_base_cm / 100.0 + SHOULDER_ABOVE_PLATE)
    pellet_robot_z = floor_robot_z + declared_pellet_cm / 100.0
    sol = solve_vertical(arm_box_x_m, pellet_robot_z, lip_l5_owner_mm)
    if sol is None:
        raise RuntimeError("취점 기준 자세 IK 실패")
    p_lip_robot, _ = lip_pose(sol["q5"], lip_l5_owner_mm)
    # DEME 취점 자리(x_s, y_s) = (0,0) 에 **실제 립이 정확히 오도록** 원점을 맞춘다.
    # (실물 상자 중심으로 립을 "명령"하면 툴의 8.1 mm 옆 오프셋 때문에 립이 중심에서 벗어난다.
    #  동결 W11 의 취점 자리를 그대로 보존하려면 상자를 립에 맞춰 놓은 것으로 본다 — 실현 가능한 배치다.)
    t = np.array([p_lip_robot[0], p_lip_robot[1], pellet_robot_z - pile_surface_z_world_m], float)
    return Adapter(t, lip_l5_owner_mm), {"reference_pose_q5": sol["q5"], "reference_err_m": sol["err_m"],
                                         "floor_robot_z_m": floor_robot_z, "pellet_robot_z_m": pellet_robot_z,
                                         "declared_base_cm": declared_base_cm, "declared_pellet_cm": declared_pellet_cm,
                                         "declared_not_measured": True,
                                         "provenance": "hw_s1_manual.py:3 CLI usage example — not an argparse default, not measured",
                                         "t_robot_m": t.tolist(),
                                         "note": "DEME 원점(상자 바닥 중심)의 로봇 좌표. 축은 평행(회전 없음)."}


# ── rev34 (W25-A) 실물 정렬 배치 선언 ───────────────────────────────────────────
# 키가 없으면 전부 rev32 동작이다(기존 params 파일 호환). 값의 뜻은 params_w25.json 의 _note 참조.
# 규약 A/B 정의 = W23 F REPORT.md:53 (x_box = 로봇 −y, y_box = 로봇 +x → R_robot_box = Rz(−90°); B 는 부호 반대).
BOX_FRAME_CONVENTIONS = {
    "rev32_parallel": None,                                              # rev32: 상자축 = 로봇축
    "A": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),  # Rz(−90°)
    "B": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),  # Rz(+90°)
}
BOX_ANCHORS = ("rev32_lip_at_box_origin", "declared_box_center")


def w25_frame(P):
    conv = P.get("box_frame_convention", "rev32_parallel")
    anchor = P.get("w25_box_anchor", "rev32_lip_at_box_origin")
    if conv not in BOX_FRAME_CONVENTIONS:
        raise SystemExit(f"box_frame_convention 은 {list(BOX_FRAME_CONVENTIONS)} 중 하나여야 한다: {conv!r}")
    if anchor not in BOX_ANCHORS:
        raise SystemExit(f"w25_box_anchor 는 {list(BOX_ANCHORS)} 중 하나여야 한다: {anchor!r}")
    return conv, BOX_FRAME_CONVENTIONS[conv], anchor


def _w25_reference(P, lip_l5_owner_mm):
    floor_robot_z = -(P["declared_base_cm"] / 100.0 + SHOULDER_ABOVE_PLATE)
    pellet_robot_z = floor_robot_z + P["declared_pellet_cm"] / 100.0
    sol = solve_vertical(P["arm_radius_m"], pellet_robot_z, lip_l5_owner_mm)
    if sol is None:
        raise RuntimeError("취점 기준 자세 IK 실패")
    p_lip_robot, _ = lip_pose(sol["q5"], lip_l5_owner_mm)
    return sol, p_lip_robot, floor_robot_z, pellet_robot_z


def _w25_t_xy(P, anchor, p_lip_robot):
    if anchor == "rev32_lip_at_box_origin":
        return [float(p_lip_robot[0]), float(p_lip_robot[1])]
    return [float(v) for v in P["w25_box_center_robot_xy_m"]]


def w25_scoop_site(P, lip_l5_owner_mm):
    """취점 자리(상자 좌표 xy, m). rev32 = (0,0). declared_box_center 면 실물 `above`/`scoop` 처럼
    상자 중심(로봇 (r,0))을 명령했을 때 **실제 FK 립**이 떨어지는 상자 좌표다(립 y −8.1 mm 옆 오프셋 포함)."""
    conv, R, anchor = w25_frame(P)
    if R is None and anchor == "rev32_lip_at_box_origin":
        return (0.0, 0.0), {"convention": conv, "anchor": anchor, "site_box_xy_m": [0.0, 0.0],
                            "rule": "rev32: 립을 상자 원점에 둔 배치", "rev32_path": True}
    sol, p_lip, _, _ = _w25_reference(P, lip_l5_owner_mm)
    t_xy = _w25_t_xy(P, anchor, p_lip)
    ad = Adapter([t_xy[0], t_xy[1], 0.0], lip_l5_owner_mm, R)
    s = ad.to_world(p_lip)
    site = (float(s[0]), float(s[1]))
    return site, {"convention": conv, "anchor": anchor, "site_box_xy_m": list(site),
                  "reference_lip_robot_m": [float(v) for v in p_lip], "box_center_robot_xy_m": t_xy,
                  "reference_pose_q5": sol["q5"],
                  "rule": "상자 중심(로봇 좌표)을 명령한 기준 자세의 FK 립을 상자 좌표로 옮긴 점",
                  "rev32_path": False}


def build_adapter_w25(box_bounds_m, pile_surface_z_world_m, P, lip_l5_owner_mm):
    """rev34 어댑터. 반환 (ad, ad_info, w25_info). rev32 설정이면 `build_adapter` 를 **그대로** 부른다
    (ad_info 키·값 불변). 회전이 있으면 p_robot = R·p_box + t, owner 회전 = Rᵀ·R_l5·R_Wᵀ."""
    conv, R, anchor = w25_frame(P)
    if R is None and anchor == "rev32_lip_at_box_origin":
        ad, info = build_adapter(box_bounds_m, pile_surface_z_world_m, P["arm_radius_m"], lip_l5_owner_mm,
                                 P["declared_base_cm"], P["declared_pellet_cm"])
        return ad, info, {"box_frame_convention": conv, "box_anchor": anchor,
                          "R_robot_box": np.eye(3).tolist(), "R_box_robot": np.eye(3).tolist(),
                          "R_scoop_owner_box": np.eye(3).tolist(), "rev32_path": True}
    sol, p_lip, floor_robot_z, pellet_robot_z = _w25_reference(P, lip_l5_owner_mm)
    t_xy = _w25_t_xy(P, anchor, p_lip)
    t = np.array([t_xy[0], t_xy[1], pellet_robot_z - pile_surface_z_world_m], float)
    ad = Adapter(t, lip_l5_owner_mm, R)
    Rm = np.eye(3) if R is None else R
    info = {"reference_pose_q5": sol["q5"], "reference_err_m": sol["err_m"],
            "floor_robot_z_m": floor_robot_z, "pellet_robot_z_m": pellet_robot_z,
            "declared_base_cm": P["declared_base_cm"], "declared_pellet_cm": P["declared_pellet_cm"],
            "declared_not_measured": True,
            "provenance": P.get("declared_placement_provenance"),
            "t_robot_m": t.tolist(),
            "note": "DEME 원점(상자 안쪽 바닥 중심)의 로봇 좌표. p_robot = R_robot_box·p_box + t_robot."}
    return ad, info, {"box_frame_convention": conv, "box_anchor": anchor,
                      "R_robot_box": Rm.tolist(), "R_box_robot": Rm.T.tolist(),
                      "R_scoop_owner_box": Rm.T.tolist(),
                      "R_scoop_rule": "취점 구간(w11 모드) 툴 owner 회전 = R_box_robot (rev32 의 단위행렬을 상자 좌표로 옮긴 것)",
                      "box_center_robot_xy_m": t_xy, "reference_lip_robot_m": [float(v) for v in p_lip],
                      "box_floor_robot_z_m": float(t[2]),
                      "box_floor_above_floor_cm": round((float(t[2]) - floor_robot_z) * 100.0, 4),
                      "rev32_path": False}


def w25_tray_bounds(box_npz, P, sphere_xyz=None, sphere_r=None, fail_closed=True):
    """트레이 안쪽 경계(상자 좌표, m). rev32 = 더미 npz box_bounds_m 그대로.
    declared = params 의 안쪽 치수(x = 긴 변)와 높이. npz 발자국이 선언과 tol 밖이면, 또는 알이 선언 트레이
    밖에 있으면 fail-closed(다른 상자용 더미를 조용히 잘라 쓰지 않는다)."""
    b0 = np.asarray(box_npz, float)
    src = P.get("tray_inner_source", "npz_box_bounds")
    if src == "npz_box_bounds":
        return b0, {"tray_inner_source": src, "box_bounds_m": b0.tolist()}
    # W25-A 후속: "declared..." 로 시작하는 값은 선언 모드(뒤 문자열은 출처 메모로 그대로 기록).
    if not str(src).startswith("declared"):
        raise SystemExit(f"tray_inner_source 는 npz_box_bounds|declared…: {src!r}")
    Lx, Ly = [float(v) / 1000.0 for v in P["tray_inner_mm"]]
    H = float(P["tray_inner_height_mm"]) / 1000.0
    b = np.array([[-Lx / 2, Lx / 2], [-Ly / 2, Ly / 2], [float(b0[2, 0]), float(b0[2, 0]) + H]], float)
    tol = float(P.get("tray_match_tol_mm", 0.5)) / 1000.0
    mism = float(np.abs(b[:2] - b0[:2]).max())
    info = {"tray_inner_source": src, "declared_inner_mm": [Lx * 1000, Ly * 1000, H * 1000],
            "npz_box_bounds_m": b0.tolist(), "box_bounds_m": b.tolist(),
            "npz_vs_declared_xy_max_abs_mm": round(mism * 1000, 6), "tol_mm": tol * 1000}
    bad = []
    if mism > tol:
        bad.append(f"더미 npz 발자국이 선언 트레이와 {mism*1000:.3f} mm 다르다(> {tol*1000} mm)")
    if sphere_xyz is not None:
        # W25-A 후속: 정착 더미는 벽에 µm 급으로 닿는다(20k 더미 +0.61 µm, npz post_settle_gates 허용 겹침 0.225 mm).
        # 발자국 대조와 같은 tol 을 알 봉쇄에도 준다 — tol 보다 크게 벽 밖이면 여전히 거부.
        S, r = np.asarray(sphere_xyz, float), np.asarray(sphere_r, float)
        out_xy = int((((S[:, 0] - r) < b[0, 0] - tol) | ((S[:, 0] + r) > b[0, 1] + tol) |
                      ((S[:, 1] - r) < b[1, 0] - tol) | ((S[:, 1] + r) > b[1, 1] + tol)).sum())
        out_top = int(((S[:, 2] + r) > b[2, 1] + tol).sum())
        info["max_sphere_overhang_mm"] = round(float(max(
            (b[0, 0] - (S[:, 0] - r)).max(), ((S[:, 0] + r) - b[0, 1]).max(),
            (b[1, 0] - (S[:, 1] - r)).max(), ((S[:, 1] + r) - b[1, 1]).max())) * 1000, 6)
        info.update(n_spheres_outside_declared_xy=out_xy, n_spheres_above_declared_top=out_top)
        if out_xy or out_top:
            bad.append(f"선언 트레이 밖 구 {out_xy}개(xy)·윗단 위 {out_top}개")
    info["mismatch_reasons"] = bad
    if bad and fail_closed:
        raise SystemExit("tray_inner_source=declared 거부: " + " / ".join(bad))
    return b, info


# ── 전 사이클 관절 웨이포인트 (실물 hw_s1_manual place()/scoop() 경로) ──────────
def build_waypoints(ad, r0_robot_m, z_travel_w, z_lip0_w, z_surface5_w, z_release_w, place_base_deg):
    """실물 `scoop()`+`place()` 경로를 (베이스각, 세계 높이) 웨이포인트로 푼다.

    반경은 기준 자세의 로봇 평면 반경 `r0` 로 고정한다 → 베이스각 0 에서 립이 DEME z 축 위를 정확히
    수직으로 움직이므로 동결 W11 의 취점 자리(0,0)와 순수 수직 하강이 그대로 보존된다.
    베이스각만 바뀌면 립은 베이스축 둘레 원호를 그린다.
    """
    wps = []

    def add_joint(name, q5):
        p, R = ad.owner_pose(q5)
        wps.append({"name": name, "kind": "joint", "q5": [float(v) for v in q5], "lip_world_m": p.tolist(),
                    "R_owner": R, "ik": {"err_m": 0.0, "branch": "stored_joint_pose", "tilt_deg": None,
                                         "limits_violations": in_limits(q5)}})

    def add_ik(name, base_deg, z_w):
        z_r = float(np.asarray(ad.to_robot([0.0, 0.0, z_w]))[2])
        sol = solve_fast(r0_robot_m, z_r, ad.lip_l5_owner_mm)
        if sol is None:
            wps.append({"name": name, "kind": "ik", "q5": None, "lip_world_m": None, "R_owner": None,
                        "ik": {"err_m": None, "branch": "unreachable", "tilt_deg": None,
                               "limits_violations": [{"joint": "unreachable", "value": None, "limit": None}],
                               "target_world_z_m": z_w, "base_deg": base_deg}})
            return
        q = list(sol["q5"])
        q[0] = float(base_deg)
        p, R = ad.owner_pose(q)
        wps.append({"name": name, "kind": "ik", "q5": q, "lip_world_m": p.tolist(), "R_owner": R,
                    "ik": dict(sol, q5=q, limits_violations=in_limits(q), target_world_z_m=z_w, base_deg=base_deg)})

    def p1_at(base_deg):
        return [float(base_deg)] + list(P1_Q5[1:])

    p_home, _ = ad.owner_pose(HOME_Q5)
    base_axis_w = ad.to_world([0.0, 0.0, 0.0])[:2]
    add_joint("initial_home", HOME_Q5)
    add_joint("p1_tool_vertical", P1_Q5)
    add_ik("above_pile_travel", 0.0, z_travel_w)
    add_ik("surface_plus_50mm", 0.0, z_surface5_w)
    add_ik("approach_gap", 0.0, z_lip0_w)
    add_ik("post_lift_travel", 0.0, z_travel_w)
    add_joint("place_retract_base0", p1_at(0.0))
    add_joint("place_retract_base90", p1_at(place_base_deg))
    add_ik("place_extend_travel", place_base_deg, z_travel_w)
    add_ik("place_target", place_base_deg, z_release_w)
    add_ik("place_up_travel", place_base_deg, z_travel_w)
    add_joint("return_retract_base90", p1_at(place_base_deg))
    add_joint("return_retract_base0", p1_at(0.0))
    add_joint("return_home", HOME_Q5)
    by = {w["name"]: w for w in wps}
    return wps, {"home_lip_world_m": p_home.tolist(), "base_axis_world_xy_m": base_axis_w.tolist(),
                 "place_lip_world_m": by["place_target"]["lip_world_m"],
                 "r0_robot_m": r0_robot_m}


def waypoint_report(wps):
    return [{"name": w["name"], "kind": w["kind"], "q5": w["q5"],
             "lip_world_mm": None if w["lip_world_m"] is None else [round(v * 1000, 3) for v in w["lip_world_m"]],
             "ik_err_mm": None if w["ik"]["err_m"] is None else round(w["ik"]["err_m"] * 1000, 4),
             "verticality": w["ik"].get("verticality"), "branch": w["ik"].get("branch"),
             "tilt_deg": None if w["ik"].get("tilt_deg") is None else round(w["ik"]["tilt_deg"], 4),
             "limit_violations": w["ik"]["limits_violations"]} for w in wps]
