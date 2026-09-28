"""W13 전체 사이클 기구학·고정물 — DEME 없이 순수 기하. 사전검토(preflight)와 본 실행이 같은 이 파일을 쓴다.

무엇을 정의하는가
    ① 로봇 베이스 ↔ DEME 세계 프레임 사상(선언된 규약; 아래 §좌표 규약 참조)
    ② 더미 상자 트레이(유한 높이 벽) 메시와 수신 용기(컵) 메시 — 둘 다 **선언된 시뮬 고정물**
    ③ HOME→접근→하강→폐합→상승→재폐합→들어올림→운반→내림→배출→닫기→복귀→HOME 목표 포즈 일정
    ④ 목표 포즈에서 툴 셸이 트레이/용기와 겹치는지 보는 기하 간섭 검사(DEME 는 메시-메시 접촉을 하지 않는다)

좌표 규약 (⚠️ 선언이며 실기 FK 재확인이 아니다)
    DEME 세계 = 더미 npz 프레임(상자 바닥 중심 원점, z 위, m). 동결 W11 소스의 R_W 열벡터가
    link5 축을 세계로 옮긴다: link5 X → 세계 −Y, link5 Y → 세계 −X, link5 Z → 세계 −Z.
    link5 X 를 팔의 반경 바깥 방향으로 읽으면 베이스각 0 에서 반경 바깥 = 세계 −Y 이므로
    로봇 베이스는 더미 중심에서 세계 +Y 로 arm_radius 만큼 떨어진 (0, +R) 에 있다.
    베이스각 β 는 세계 +Z 둘레 오른손 회전이며 립 = B + R·(sin β, −cos β, 0).
    β=+90° 에서 립 = (R, R) — 실물 `hw_s1_manual.py place()` 기본 `--place-deg 90`(+y 90° 자리)에 해당한다.
    ⚠️ link5 X 의 부호(반경 바깥 vs 안쪽)는 실기 FK 로 재확인하지 않았다. 반대 부호면 배출 자리가
       세계 (−R, R) 로 거울상이 될 뿐 물리·질량수지는 바뀌지 않는다. 본 실행은 위 규약을 고정해 쓴다.

고정물 (⚠️ 실측 아님 — 선언된 시뮬 픽스처)
    트레이 = 더미 npz `box_bounds_m` 의 x·y 안쪽면을 그대로 쓰고 윗단은 같은 npz 의 z 상한(0.136633 m).
        W11 은 이 자리를 무한 해석 평면(도메인 BC)으로 막았다. W13 은 툴이 상자 밖으로 나가야 하므로
        도메인을 넓히고 같은 자리를 **유한 높이 메시 벽**으로 바꾼다(§DEVIATION-1).
    수신 용기 = 열린 원통(내경·깊이·벽두께는 params). 실물 컵은 빈 무게 9.66 g 만 기록돼 있고
        치수 실측이 repo 증거에 없다 → **치수는 측정값이 아니라 선언된 픽스처**다.
"""
import json
import math

import numpy as np
import trimesh

# ── 상수(전부 params 로 덮어쓸 수 있다) ────────────────────────────────────────
W13_DEFAULT = {
    # 프레임 — 출처를 두 부류로 분리한다(감사 지적: 38/26 은 argparse 기본값이 아니다)
    "arm_radius_m": 0.35,                 # hw_s1_manual.py:269 argparse **기본값** --box-x-cm 35.0
    "place_base_deg": 90.0,               # hw_s1_manual.py:269 argparse **기본값** --place-deg 90.0
    "travel_cm": 45.0,                    # hw_s1_manual.py:269 argparse **기본값** --travel-cm 45.0
    # ↓ 아래 둘은 argparse 에서 required=True 라 기본값이 없다. hw_s1_manual.py:3 의 **CLI 사용 예시** 숫자를
    #   W13 이 declared_not_measured 픽스처로 채택한 것이다. 실측값이 아니다.
    "declared_base_cm": 38.0,
    "declared_pellet_cm": 26.0,
    "declared_placement_provenance": "hw_s1_manual.py:3 CLI usage example (NOT argparse default, NOT measured)",
    # 트레이(더미 상자) — x/y 안쪽면과 윗단은 더미 npz 에서 읽고 두께만 여기서 준다
    "tray_wall_t_mm": 5.0,
    # 수신 용기(선언 픽스처 — 실측 아님)
    "bin_inner_r_mm": 40.0,
    "bin_inner_h_mm": 70.0,
    "bin_wall_t_mm": 3.0,
    "bin_floor_z_m": 0.0,                 # 도메인 바닥 평면(= 더미 상자 바닥)과 같은 지지면
    "bin_n_theta": 48,
    # 운반/배출 일정
    "transport_speed_mm_s": 150.0,        # 기존 sim lift_mm_s 와 같은 값(새 속도 상수를 만들지 않는다)
    "release_clearance_mm": 20.0,         # 배출 때 립이 용기 테두리 위로 뜨는 높이
    "discharge_open_servo_deg": 30.0,     # 실물 place(): door(30, tor=200)
    "discharge_hold_s": 1.5,              # 실물 place(): time.sleep(1.5)
    "home_hold_s": 0.004,                 # initial_home = 정확히 1 sync (W11 과 같은 누적 경과시간 유지)
    "final_home_hold_s": 0.1,             # 복귀 뒤 HOME 정지 유지(정착·동일 포즈 확인)
    # 저장 축
    "particle_frame_dt_s": 0.1,           # 희소 입자 프레임 간격(+ 모든 phase 전환·결정 시점 강제)
    "domain_pad_m": 0.06,
    # 재고 분류
    "classify_margin_mm": 2.5,            # 경계 밴드(알 외접 반지름 2.25 mm 보다 크게) → ambiguous
    "spill_rest_z_m": 0.02,               # 지지면 위에서 쉬는 것으로 볼 z 상한
    # 벽 회귀 진단(근접/관통/접촉을 따로 센다)
    "wall_near_tol_mm": 1.0,
    # 정착 관측창 — **감사가 결과를 보기 전에 고정한 operational criterion**. 물리적 rest truth 아님.
    "settlement_window_s": 0.25,
    "settlement_frame_dt_s": 0.05,
    "settle_speed_max_m_s": 0.005,
    "settle_move_max_m": 0.001,
    "definite_geom_eps_mm": 0.2,          # oriented 7-구 definite 판정의 양의 여유
}

# 동결 W11 소스의 link5 → 세계 회전(열 = link5 축). 여기서 다시 정의하지 않고 소스에서 받아 검산한다.
R_W_EXPECTED = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)


def rot_z(deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], float)


def axis_angle(axis, deg):
    """단위축 둘레 오른손 회전행렬(로드리게스)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], float)
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def mat_to_quat_xyzw(R):
    """회전행렬 → xyzw 쿼터니언(DEME OriQ 규약)."""
    m = np.asarray(R, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], float)
    return q / np.linalg.norm(q)


def quat_xyzw_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], float)


def rotvec_of(R):
    """회전행렬 → 회전벡터(rad, 크기 = 각). 서보 각속도 계산용."""
    m = np.asarray(R, float)
    c = min(1.0, max(-1.0, (np.trace(m) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-12:
        return np.zeros(3)
    if math.pi - th < 1e-6:                       # 180° 근방 — 본 일정에서는 나오지 않지만 안전하게
        A = (m + np.eye(3)) / 2.0
        v = np.sqrt(np.clip(np.diag(A), 0, None))
        k = int(np.argmax(v))
        v = v / max(v[k], 1e-12) * v[k]
        s = np.sign([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
        s[s == 0] = 1.0
        return th * v * s / max(np.linalg.norm(v), 1e-12)
    w = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]], float) / (2 * math.sin(th))
    return th * w


# ── 고정물 메시 ──────────────────────────────────────────────────────────────
def tray_mesh(box_bounds_m, wall_t_m):
    """더미 상자 트레이 = 유한 높이 벽 4장(안쪽면 = box_bounds x/y, 윗단 = box_bounds z 상한, 바닥 없음).

    바닥은 W11 과 같이 도메인 경계 BC 평면(z = box_bounds[2,0])이 맡는다. 각 벽은 바깥으로만 두께를
    붙인 직육면체라 안·바깥 면이 모두 있어 구가 벽을 통과해도 되돌려 준다(W3 얇은 벽 발산 교훈).
    네 벽은 서로 겹치지 않게 잘라 붙여 전체가 닫힌 사각 테두리(watertight 성분 4개)가 되게 한다 —
    겹치면 signed_distance 간섭 검사가 거짓 양성을 낸다.
    """
    b = np.asarray(box_bounds_m, float)
    x0, x1 = float(b[0, 0]), float(b[0, 1])
    y0, y1 = float(b[1, 0]), float(b[1, 1])
    z0, z1 = float(b[2, 0]), float(b[2, 1])
    t = float(wall_t_m)
    boxes = [
        (x0 - t, x0, y0 - t, y1 + t, z0, z1),     # -x 벽(모서리 포함)
        (x1, x1 + t, y0 - t, y1 + t, z0, z1),     # +x 벽(모서리 포함)
        (x0, x1, y0 - t, y0, z0, z1),             # -y 벽(모서리 제외 — 겹침 금지)
        (x0, x1, y1, y1 + t, z0, z1),             # +y 벽(모서리 제외)
    ]
    parts = []
    for (a0, a1, c0, c1, d0, d1) in boxes:
        m = trimesh.creation.box(extents=(a1 - a0, c1 - c0, d1 - d0))
        m.apply_translation([(a0 + a1) / 2, (c0 + c1) / 2, (d0 + d1) / 2])
        parts.append(m)
    return trimesh.util.concatenate(parts)


def bin_mesh(P, center_xy):
    """수신 용기 = 밑면 원판 + 그 위에 얹힌 고리벽. 두 성분 모두 watertight 이고 서로 겹치지 않는다."""
    r_in = P["bin_inner_r_mm"] / 1000.0
    r_out = r_in + P["bin_wall_t_mm"] / 1000.0
    h = P["bin_inner_h_mm"] / 1000.0
    t = P["bin_wall_t_mm"] / 1000.0
    z0 = float(P["bin_floor_z_m"])
    n = int(P["bin_n_theta"])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    floor = trimesh.creation.cylinder(radius=r_out, height=t, sections=n)
    floor.apply_translation([cx, cy, z0 + t / 2.0])
    wall = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=h, sections=n)
    wall.apply_translation([cx, cy, z0 + t + h / 2.0])
    m = trimesh.util.concatenate([floor, wall])
    return m, {"inner_r_m": r_in, "outer_r_m": r_out, "floor_inner_z_m": z0 + t, "rim_z_m": z0 + t + h,
               "center_xy_m": [cx, cy], "wall_t_m": t, "n_theta": n,
               "inner_volume_cm3": round(math.pi * r_in ** 2 * h * 1e6, 3),
               "watertight": bool(m.is_watertight), "n_tri": int(len(m.faces)),
               "components": [{"name": "floor_disc", "watertight": bool(floor.is_watertight)},
                              {"name": "wall_ring", "watertight": bool(wall.is_watertight)}]}


# ── 프레임 ───────────────────────────────────────────────────────────────────
class Frames:
    """로봇 베이스 ↔ DEME 세계. 더미 중심(세계 원점)이 베이스각 0 의 립 자리다."""

    def __init__(self, P):
        self.R = float(P["arm_radius_m"])
        self.base_xy = np.array([0.0, self.R], float)      # §좌표 규약

    def lip_xy(self, beta_deg):
        b = math.radians(beta_deg)
        return self.base_xy + self.R * np.array([math.sin(b), -math.cos(b)], float)

    def tool_R(self, beta_deg):
        return rot_z(beta_deg)


# ── 일정(목표 포즈) ──────────────────────────────────────────────────────────
PHASE_IDS = {"settle": 0, "approach": 1, "descend": 2, "close": 3, "lift": 4, "reclose": 5,
             "raise": 6, "carry": 7, "lower": 8, "discharge_open": 9, "discharge_hold": 10,
             "discharge_close": 11, "raise2": 12, "return": 13, "park": 14}


def build_schedule(P, z_lip0_m, z_carry_m, z_release_m):
    """각 phase 의 명목 지속시간·속도. 상태 의존(하강 힘정지·문 정지·재폐합)은 실행 중에 결정된다."""
    v_t = P["transport_speed_mm_s"] / 1000.0
    f = Frames(P)
    beta_end = float(P["place_base_deg"])
    omega = v_t / f.R                                  # rad/s — 립 접선속도를 운반속도로 맞춘다
    return {
        "transport_speed_m_s": v_t,
        "carry_omega_rad_s": omega,
        "carry_omega_deg_s": math.degrees(omega),
        "beta_end_deg": beta_end,
        "z_carry_m": z_carry_m,
        "z_release_m": z_release_m,
        "z_lip0_m": z_lip0_m,
        "nominal_s": {
            "approach": abs(z_carry_m - z_lip0_m) / v_t,
            # "raise" 는 재폐합 종료 립 z 가 실행 중 결정되므로 여기서 명목값을 만들지 않는다
            "carry": math.radians(beta_end) / omega,
            "lower": abs(z_carry_m - z_release_m) / v_t,
            "discharge_open": (P["discharge_open_servo_deg"] - P["servo_zero_offset_deg"]) / P["close_deg_s"],
            "discharge_hold": P["discharge_hold_s"],
            "discharge_close": (P["discharge_open_servo_deg"] - P["servo_zero_offset_deg"]) / P["close_deg_s"],
            "raise2": abs(z_carry_m - z_release_m) / v_t,
            "return": math.radians(beta_end) / omega,
        },
    }


# ── 간섭 검사 ────────────────────────────────────────────────────────────────
def tool_world_vertices(base_v, p_owner, R_owner):
    return (np.asarray(R_owner, float) @ np.asarray(base_v, float).T).T + np.asarray(p_owner, float)


def clearance_report(fixed_v, door_v, poses, obstacles):
    """샘플된 목표 포즈마다 툴 정점과 장애물 메시의 최소 거리(부호 없음)와 내부 침투 여부.

    DEME 는 메시-메시 접촉을 하지 않으므로 툴이 트레이/용기를 뚫고 지나가도 솔버는 막지 않는다.
    따라서 목표 경로가 기하적으로 비어 있는지 여기서 따로 증명해야 한다.
    obstacles = {이름: trimesh}
    반환: 장애물별 최소거리(mm)·해당 phase/포즈 index·침투 정점 수.
    """
    out = {}
    for name, mesh in obstacles.items():
        q = trimesh.proximity.ProximityQuery(mesh)
        worst = {"min_dist_mm": None, "at": None, "n_inside": 0, "inside_at": [],
                 "obstacle_watertight": bool(mesh.is_watertight), "obstacle_n_tri": int(len(mesh.faces)),
                 "signed_distance_valid": bool(mesh.is_watertight)}
        for k, (phase, p_f, R_f, p_d, R_d) in enumerate(poses):
            V = np.vstack([tool_world_vertices(fixed_v, p_f, R_f), tool_world_vertices(door_v, p_d, R_d)])
            d = q.signed_distance(V)               # >0 = 메시 내부
            dmin = float(np.abs(d).min())
            n_in = int((d > 0).sum())
            if worst["min_dist_mm"] is None or dmin * 1000 < worst["min_dist_mm"]:
                worst["min_dist_mm"] = round(dmin * 1000, 4)
                worst["at"] = {"index": k, "phase": phase, "lip_mm": (np.asarray(p_f) * 1000).round(2).tolist()}
            if n_in:
                worst["n_inside"] += n_in
                worst["inside_at"].append({"index": k, "phase": phase, "n": n_in})
        out[name] = worst
    return out


def dump_fixture_json(path, P, box_bounds_m, bin_info, tray, schedule, frames, extra=None):
    d = {
        "artifact": "W13_FIXTURE_V1",
        "declared_not_measured": [
            "수신 용기 치수(내경·깊이·벽두께·자리 높이)는 선언된 시뮬 픽스처다. 실물 컵은 빈 무게 9.66 g 만 기록돼 있고 치수 실측 증거가 repo 에 없다.",
            "로봇 베이스 방향(link5 X 부호)은 동결 R_W 해석에 따른 선언이며 실기 FK 로 재확인하지 않았다.",
            "트레이 벽은 W11 의 무한 해석 평면을 대신하는 유한 높이 메시다(DEVIATION-1).",
        ],
        "world_frame": "DEME 세계 = 더미 npz 프레임(상자 바닥 중심 원점, z 위, m)",
        "arm_radius_m": frames.R,
        "base_world_xy_m": frames.base_xy.tolist(),
        "beta_convention": "세계 +Z 둘레 오른손 회전, 립 = base + R*(sin b, -cos b, 0); b=0 이 더미 중심",
        "box_bounds_m": np.asarray(box_bounds_m, float).tolist(),
        "tray": {"wall_t_m": P["tray_wall_t_mm"] / 1000.0, "n_tri": int(len(tray.faces)),
                 "watertight": bool(tray.is_watertight), "bounds_m": tray.bounds.tolist(),
                 "top_z_m": float(np.asarray(box_bounds_m, float)[2, 1])},
        "bin": bin_info,
        "schedule": schedule,
        "classification_thresholds": {"rest_threshold_rule": "g * dt_sync_s (실행 시 계산)",
                                     "classify_margin_mm": P.get("classify_margin_mm"),
                                     "spill_rest_z_m": P["spill_rest_z_m"]},
    }
    if extra:
        d.update(extra)
    with open(path, "w") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=2)
    return d
