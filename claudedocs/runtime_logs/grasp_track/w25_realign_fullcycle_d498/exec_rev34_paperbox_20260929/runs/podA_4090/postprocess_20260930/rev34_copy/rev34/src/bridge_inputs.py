"""W13 resume-bridge 인증 입력·명령열 재현·fail-closed 게이트 V4 (순수 CPU · DEME/하드웨어 접근 0).

무엇을 하는가
    ① 동결 fixture 기하를 **볼록 셀 AABB** 로 만든다(w13_kinematics.tray_mesh/bin_mesh 와 같은 수식).
    ② `align_move`/`joint_move` 가 낼 **정확한 per-sync 명령 목표열**을 재현한다. 고정부 목표와
       문 목표를 **컨트롤러가 실제로 쓰는 그 수식**(`pose_of`)으로 같이 낸다.
       ⚠️ 이건 *명령 유도* 재현이지 *실행 운동*의 강체 힌지 가정이 아니다 — 실행 운동 상한에서는
          두 몸체의 **실측 포즈를 각각 따로** 쓴다(`msg_2e5ccf261e8f` D7).
    ③ 수치 허용치는 `bridge_numerics` 의 설치본 근거 4항(ideal motion / position lattice /
       tracker read / quaternion rotation)을 **따로** 계산해 더한다. placeholder 없음.
       `l`·`voxelSize` 가 없으면 추정하지 않고 fail-closed (SOURCE_EVIDENCE §1 = preflight NO-GO).
    ④ `gated_bridge()` — PASS 가 아니면 **bridge 의 첫 DoDynamicsThenSync 전에** 예외로 중단한다.

주장하지 않는 것
    고정 fixture 와의 기하 충돌 없음만 인증한다. 접촉 성공·서보 실현성·물리 동등성·수렴의 증거가 아니다.
    용기 셀은 48각형의 축정렬 AABB 겉包이라 보수적이다.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

import bridge_clearance_guard as G
import bridge_numerics as BN
import w13_fk as FK
import w13_kinematics as K

# ── 동결 임계 (criteria.json 과 같은 값. 여기가 코드 정본) ──────────────────────
NUMERIC_EPSILON_M = 1e-6        # float64 기하 누적 여유 (수치 허용치 4항과는 별개의 기하 epsilon)
ORTHONORMAL_TOL = 1e-9          # 회전행렬 직교성/det 허용 오차
UNIT_NORM_TOL = 1e-6            # 트래커 쿼터니언 단위 노름 허용 오차 (ERRATUM_02 E5 ①)
QMIN = BN.QMIN_DEFAULT          # 0.5 (ERRATUM_02 E5 ①)
HALF_ULP = False                # 변환 모드 미고정 → 보수적 full ULP (SOURCE_EVIDENCE §2)
MAX_ALIGN_ROT_DEG = 170.0
PLAN_TOL_M = 1e-9
PLAN_TOL_RAD = 1e-9


class ClearanceUncertified(Exception):
    """bridge 사전 인증 실패 = 계획된 중단. 물리 발산이 아니다."""

    def __init__(self, certificate):
        self.certificate = certificate
        super().__init__(f"CLEARANCE_UNCERTIFIED: {certificate.get('abort_detail')}")


class PrecheckFailed(Exception):
    """per-sync precheck 실패 — sim 의 step() 이 DoDynamicsThenSync **전에** 올린다."""

    def __init__(self, row):
        self.row = row
        super().__init__(f"per-sync precheck failed at sync {row.get('sync_ordinal')}: "
                         f"sep={row.get('separation_ok')} plan={row.get('plan_match_ok')} "
                         f"boot={row.get('bootstrap_ok')} err={row.get('input_error')}")


# ── ① fixture 볼록 셀 ────────────────────────────────────────────────────────
def tray_cells(box_bounds_m, wall_t_mm):
    """w13_kinematics.tray_mesh 와 **같은 6-튜플 수식**. 네 벽 전부 축정렬이라 AABB 가 정확하다."""
    b = np.asarray(box_bounds_m, float)
    x0, x1 = float(b[0, 0]), float(b[0, 1])
    y0, y1 = float(b[1, 0]), float(b[1, 1])
    z0, z1 = float(b[2, 0]), float(b[2, 1])
    t = float(wall_t_mm) / 1000.0
    spec = [("tray_wall_minus_x", x0 - t, x0, y0 - t, y1 + t, z0, z1),
            ("tray_wall_plus_x", x1, x1 + t, y0 - t, y1 + t, z0, z1),
            ("tray_wall_minus_y", x0, x1, y0 - t, y0, z0, z1),
            ("tray_wall_plus_y", x0, x1, y1, y1 + t, z0, z1)]
    return [{"name": n, "min_m": [a0, c0, d0], "max_m": [a1, c1, d1], "exact_aabb": True}
            for (n, a0, a1, c0, c1, d0, d1) in spec]


def bin_cells(P, center_xy):
    """용기 = 밑면 원판 + 고리벽. 48각형을 **축정렬 AABB 로 겉包**(보수적, 외접 r_out 기준)."""
    r_out = (P["bin_inner_r_mm"] + P["bin_wall_t_mm"]) / 1000.0
    h = P["bin_inner_h_mm"] / 1000.0
    t = P["bin_wall_t_mm"] / 1000.0
    z0 = float(P["bin_floor_z_m"])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    return [{"name": "bin_floor_disc", "min_m": [cx - r_out, cy - r_out, z0],
             "max_m": [cx + r_out, cy + r_out, z0 + t], "exact_aabb": False},
            {"name": "bin_wall_ring", "min_m": [cx - r_out, cy - r_out, z0 + t],
             "max_m": [cx + r_out, cy + r_out, z0 + t + h], "exact_aabb": False}]


def obstacle_cells(box_bounds_m, P, bin_center_xy):
    return tray_cells(box_bounds_m, P["tray_wall_t_mm"]) + bin_cells(P, bin_center_xy)


# ── 관절 반경 상한 (교차검증 전용 — 판정식에 없다, D10) ──────────────────────
def joint_point_radius_bounds(lip_l5_owner_mm, fixed_v_m, door_v_m, hinge_off_m):
    tr = [{"name": nm, "norm_m": float(np.linalg.norm(np.asarray(xyz, float))), "joint_index": qi}
          for nm, xyz, _rpy, qi in FK.CHAIN]
    idx_of = {e["joint_index"]: k for k, e in enumerate(tr) if e["joint_index"] is not None}
    k_last = max(idx_of.values())
    L5 = float(np.linalg.norm(np.asarray(lip_l5_owner_mm, float) / 1000.0))
    r_fixed = float(np.linalg.norm(np.asarray(fixed_v_m, float), axis=1).max())
    r_door = float(np.linalg.norm(np.asarray(door_v_m, float), axis=1).max()
                   + np.linalg.norm(np.asarray(hinge_off_m, float)))
    r_tool = max(r_fixed, r_door)
    rho, detail = [], []
    for j in range(len(FK.JOINT_NAMES)):
        down = sum(tr[k]["norm_m"] for k in range(idx_of[j] + 1, k_last + 1))
        rho.append(down + L5 + r_tool)
        detail.append({"joint": FK.JOINT_NAMES[j], "downstream_link_sum_m": round(down, 9),
                       "lip_in_link5_m": round(L5, 9), "max_tool_vertex_radius_m": round(r_tool, 9),
                       "rho_m": round(down + L5 + r_tool, 9)})
    return rho, {"per_joint": detail, "chain_translations": tr,
                 "role": "cross-check only — not used in the PASS/FAIL inequality (D10)"}


def applicable_joint_limits():
    """v6 분포 클립 한계 ∩ wrist_p 펌웨어 ±90°(D481). 5 팔관절만."""
    out, detail = [], []
    for n in FK.JOINT_NAMES:
        lo, hi = FK.JOINT_LIMITS_DEG[n]
        src = "roarm_kinematics.JOINT_LIMITS_DEG"
        if n == "wrist_p":
            lo2, hi2 = max(lo, -FK.WRIST_MAX), min(hi, FK.WRIST_MAX)
            if (lo2, hi2) != (lo, hi):
                src += f" ∩ wrist_p firmware clamp ±{FK.WRIST_MAX:.1f}"
            lo, hi = lo2, hi2
        out.append([float(lo), float(hi)])
        detail.append({"joint": n, "limit_deg": [float(lo), float(hi)], "origin": src})
    return out, detail


def build_static_inputs(*, box_bounds_m, P, bin_center_xy, fixed_v_m, door_v_m, hinge_off_m,
                        lip_l5_owner_mm, door_axis_owner, door_q_open_deg, timestep_s,
                        transport_speed_m_s, close_deg_s, dts_s, domain_x, domain_y, domain_z,
                        length_unit_l_m=None, voxel_size_m=None, domain_max_coord_m=None,
                        numeric_evidence=None):
    """실행 중 변하지 않는 인증 입력 + 해시.

    `door_v_m` 은 **힌지 기준 문 자체 정점**(dv)이다. 문은 자기 owner 원점(힌지)을 갖는 별도 추적
    강체이므로 국소 반경도 그 원점 기준으로 잰다(D7).
    `length_unit_l_m`/`voxel_size_m` 은 설치본 초기화값이며, 없으면 위치 격자 항이 증명 불가라
    인증이 fail-closed 된다(SOURCE_EVIDENCE §1). **추정하지 않는다.**
    """
    cells = obstacle_cells(box_bounds_m, P, bin_center_xy)
    rho, rho_detail = joint_point_radius_bounds(lip_l5_owner_mm, fixed_v_m, door_v_m, hinge_off_m)
    lims, lim_detail = applicable_joint_limits()
    fv = np.ascontiguousarray(np.asarray(fixed_v_m, float))
    dv = np.ascontiguousarray(np.asarray(door_v_m, float))
    # ⚠️ 트래커 readback 항의 좌표 크기는 **사용자 도메인 최대값이 아니라** 설치본이 쓰는 정수 세계의
    #    절대 좌표 상한이어야 한다(DEME_LATTICE_RECONSTRUCTION.md: 1.0154914259910583 m vs 0.5881 m).
    #    검증된 수치 증거가 주어지면 그 값을 쓰고, 없으면 사용자 도메인 최대값으로 **보수적이지 않게**
    #    떨어질 수 있으므로 그 사실을 provenance 에 남긴다.
    user_dom_max = float(max(abs(v) for v in list(domain_x) + list(domain_y) + list(domain_z)))
    dom_max = float(domain_max_coord_m) if domain_max_coord_m is not None else user_dom_max
    N, acc = BN.internal_steps(dts_s, timestep_s)
    return {
        "obstacle_cells": cells, "joint_limits_deg": lims, "joint_point_radius_bounds_m": rho,
        "fixed_vertices_local_m": fv, "door_vertices_local_m": dv,
        "door_hinge_offset_owner_m": np.asarray(hinge_off_m, float),
        "door_axis_owner": np.asarray(door_axis_owner, float),
        "door_q_open_deg": float(door_q_open_deg),
        "timestep_s": float(timestep_s), "dts_s": float(dts_s),
        "internal_steps_N": N, "accumulator_at_stop_s": acc,
        "call_elapsed_upper_bound_s": BN.call_elapsed_upper_bound_s(dts_s, timestep_s),
        "length_unit_l_m": length_unit_l_m, "voxel_size_m": voxel_size_m,
        "domain_max_coord_m": dom_max,
        "domain_max_coord_source": ("installed-binary static reconstruction (verified reusable)"
                                     if domain_max_coord_m is not None
                                     else "user domain max (NOT the engine integer-world bound)"),
        "user_domain_max_coord_m": user_dom_max,
        "numeric_evidence": numeric_evidence,
        "domain_x_m": list(domain_x), "domain_y_m": list(domain_y), "domain_z_m": list(domain_z),
        "qmin": QMIN, "half_ulp": HALF_ULP, "unit_norm_tol": UNIT_NORM_TOL,
        "numeric_epsilon_m": NUMERIC_EPSILON_M, "orthonormal_tol": ORTHONORMAL_TOL,
        "max_align_rotation_deg": MAX_ALIGN_ROT_DEG,
        "plan_tol_m": PLAN_TOL_M, "plan_tol_rad": PLAN_TOL_RAD,
        "transport_speed_m_s": float(transport_speed_m_s), "close_deg_s": float(close_deg_s),
        "numeric_receipt": BN.pinned_receipt(
            requested_duration_s=dts_s, timestep_s=timestep_s, l_m=length_unit_l_m,
            voxel_size_m=voxel_size_m, domain_max_coord_m=dom_max,
            r_max_local_m_by_body={"fixed": float(np.linalg.norm(fv, axis=1).max()),
                                   "door": float(np.linalg.norm(dv, axis=1).max())},
            qmin=QMIN, half_ulp=HALF_ULP),
        "provenance": {
            "cells": "w13_kinematics.tray_mesh/bin_mesh 와 같은 수식. 트레이 4장 정확 AABB, 용기 2성분 보수 겉包.",
            "joint_radius": rho_detail, "joint_limits": lim_detail,
            "numeric_evidence": BN.EVIDENCE,
            "withdrawn_thresholds": (
                "고정 0.5 mm 추종 여유와 k_overrun_steps=2(관측 +1e-06 s 의 2배)는 둘 다 철회됐고 "
                "코드에 없다(msg_c0a8c4367a15). 관측 +1.000000e-06 s 는 구 rev10 의 round(...,9) "
                "십진 양자화(±1 ns)를 포함하므로 정확한 내부 duration 이 아니다(ERRATUM_03)."),
            "door_body_independence": (
                "실행 운동 상한에서 문 포즈를 고정부에서 유도하지 않는다. 두 트래커의 실측 포즈를 각각 쓴다. "
                "명령 **목표** 유도(pose_of)는 컨트롤러가 실제로 그렇게 계산하므로 그대로 재현한다."),
            "vertex_sha256": {"fixed": hashlib.sha256(fv.tobytes()).hexdigest(),
                              "door": hashlib.sha256(dv.tobytes()).hexdigest()},
            "n_fixed_vertices": int(len(fv)), "n_door_vertices": int(len(dv))},
    }


# ── ② 명령 목표열 재현 (align_move / joint_move 와 같은 수식) ─────────────────
def _door_target(p_f, R_f, hinge_off, axis_w, q_deg, q_open_deg):
    """컨트롤러 `pose_of` 의 문 **목표** 유도. 실행 운동 가정이 아니라 명령 재현이다."""
    R_rel = K.axis_angle(axis_w, float(q_deg) - float(q_open_deg))
    return np.asarray(p_f, float) + np.asarray(R_f, float) @ np.asarray(hinge_off, float), \
        np.asarray(R_f, float) @ R_rel


def plan_align_targets(static, p_from, R_from, p_to, R_to, door_q_deg):
    """`sim_w13_full_cycle.align_move` 와 **같은 n·같은 보간**. 고정부/문 목표를 함께 낸다."""
    v_t = static["transport_speed_m_s"]
    dts = static["dts_s"]
    p_fr = np.asarray(p_from, float)
    R_fr = np.asarray(R_from, float)
    rv = K.rotvec_of(R_fr.T @ np.asarray(R_to, float))
    ang = float(np.degrees(np.linalg.norm(rv)))
    dist = float(np.linalg.norm(np.asarray(p_to, float) - p_fr))
    n = max(1, int(math.ceil(max(dist / (v_t * dts), ang / (static["close_deg_s"] * dts)))))
    out = []
    for i in range(1, n + 1):
        f = i / n
        p = p_fr + (np.asarray(p_to, float) - p_fr) * f
        R = (R_fr @ K.axis_angle(rv / max(np.linalg.norm(rv), 1e-12), ang * f)
             if np.linalg.norm(rv) > 1e-12 else R_fr)
        pd, Rd = _door_target(p, R, static["door_hinge_offset_owner_m"], static["door_axis_owner"],
                              door_q_deg, static["door_q_open_deg"])
        out.append({"segment": "align_sync_targets", "i": i, "n": n, "dts_s": dts,
                    "fixed": {"pos_m": p, "R": R}, "door": {"pos_m": pd, "R": Rd}})
    return out


def plan_joint_targets(static, q_from, q_to, owner_pose_fn, door_q_deg):
    """`sim_w13_full_cycle.joint_move` 와 **같은 n·같은 관절 직선 보간**."""
    v_t = static["transport_speed_m_s"]
    dts = static["dts_s"]
    q0 = np.asarray(q_from, float)
    q1 = np.asarray(q_to, float)
    p0, _ = owner_pose_fn(q0.tolist())
    p1, _ = owner_pose_fn(q1.tolist())
    n = max(1, int(math.ceil(float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)))
                             / (v_t * dts))))
    out = []
    for i in range(1, n + 1):
        q = q0 + (q1 - q0) * (i / n)
        p, R = owner_pose_fn(q.tolist())
        pd, Rd = _door_target(p, R, static["door_hinge_offset_owner_m"], static["door_axis_owner"],
                              door_q_deg, static["door_q_open_deg"])
        out.append({"segment": "joint_sync_targets", "i": i, "n": n, "dts_s": dts,
                    "q_deg": q, "fixed": {"pos_m": np.asarray(p, float), "R": np.asarray(R, float)},
                    "door": {"pos_m": pd, "R": Rd}})
    return out


def _boot_inputs(poses_actual, targets, dts_s):
    """ERRATUM_02 E5 입력: 트래커 쿼터니언(xyzw) + 명령 국소 각속도 (몸체별, servo 와 같은 식)."""
    out = {}
    for b in G.BODIES:
        pa, Ra = poses_actual[b]
        pt, Rt = targets[b]
        Ra = np.asarray(Ra, float)
        omega = K.rotvec_of(Ra.T @ np.asarray(Rt, float)) / float(dts_s)   # servo 와 동일(국소 프레임)
        out[b] = {"quat_xyzw": K.mat_to_quat_xyzw(Ra), "omega_rad_s": omega}
    return out


# ── ④ 인증 + per-sync precheck + fail-closed 게이트 ─────────────────────────
def _thin(cert):
    out = {k: v for k, v in cert.items() if k not in ("intervals", "start_ball_trace")}
    out["n_intervals_recorded"] = len(cert.get("intervals", []))
    return out


class SyncPrecheck:
    """bridge 두 구간 동안 **매 sync 의 DoDynamicsThenSync 직전** fail-closed 검사기."""

    def __init__(self, static, planned):
        self.s = static
        self.planned = planned
        self.k = 0
        self.rows = []
        self.worst_slack_m = math.inf
        self.max_plan_dev_m = 0.0

    def __call__(self, dts_s, actual_fixed, actual_door, target_fixed, target_door):
        if self.k >= len(self.planned):
            return {"ok": False, "input_error": f"sync {self.k + 1} exceeds planned {len(self.planned)}"}
        T = self.planned[self.k]
        actual = {"fixed": actual_fixed, "door": actual_door}
        target = {"fixed": target_fixed, "door": target_door}
        row = G.precheck_sync(
            actual_poses={b: {"pos_m": actual[b][0], "R": actual[b][1]} for b in G.BODIES},
            target_poses={b: {"pos_m": target[b][0], "R": target[b][1]} for b in G.BODIES},
            dts_s=dts_s,
            planned_target_poses={b: {"pos_m": T[b]["pos_m"], "R": T[b]["R"]} for b in G.BODIES},
            plan_tol_m=self.s["plan_tol_m"], plan_tol_rad=self.s["plan_tol_rad"],
            fixed_vertices_local_m=self.s["fixed_vertices_local_m"],
            door_vertices_local_m=self.s["door_vertices_local_m"],
            obstacle_cells=self.s["obstacle_cells"], timestep_s=self.s["timestep_s"],
            numeric_epsilon_m=self.s["numeric_epsilon_m"],
            length_unit_l_m=self.s["length_unit_l_m"], voxel_size_m=self.s["voxel_size_m"],
            domain_max_coord_m=self.s["domain_max_coord_m"], qmin=self.s["qmin"],
            half_ulp=self.s["half_ulp"],
            bootstrap=_boot_inputs(actual, target, dts_s),
            orthonormal_tol=self.s["orthonormal_tol"], unit_norm_tol=self.s["unit_norm_tol"])
        row["sync_ordinal"] = self.k + 1
        row["planned_segment_i_n"] = [str(T["segment"]), int(T["i"]), int(T["n"])]
        self.k += 1
        self.rows.append(row)
        if row.get("worst"):
            self.worst_slack_m = min(self.worst_slack_m, float(row["worst"]["slack_m"]))
        for b in G.BODIES:
            self.max_plan_dev_m = max(self.max_plan_dev_m,
                                      float((row.get("plan") or {}).get(b, {}).get("pos_dev_m") or 0.0))
        return row

    def summary(self):
        return {"n_syncs_checked": self.k, "n_planned_targets": len(self.planned),
                "all_planned_consumed": bool(self.k == len(self.planned)),
                "all_ok": bool(self.rows) and all(r.get("ok") for r in self.rows),
                "worst_slack_m": None if self.worst_slack_m == math.inf else self.worst_slack_m,
                "max_plan_target_deviation_m": self.max_plan_dev_m,
                "n_failed": sum(0 if r.get("ok") else 1 for r in self.rows)}


def certify_bridge(static, *, actual_fixed, actual_door, commanded_fixed, align_target_pose,
                   q_res_deg, q_post_lift_deg, owner_pose_fn, door_q_deg_actual,
                   door_q_deg_commanded, z_reached_m, ik_resume, sim_t_s, sync_index, wall_s):
    """정확한 명령 목표열을 재현해 전체 bridge 를 인증한다. 반환값이 bridge 결정의 원시 정본."""
    align = plan_align_targets(static, commanded_fixed[0], commanded_fixed[1],
                               align_target_pose[0], align_target_pose[1], door_q_deg_commanded)
    joints = plan_joint_targets(static, q_res_deg, q_post_lift_deg, owner_pose_fn,
                                door_q_deg_commanded)
    planned = list(align) + list(joints)
    actual = {"fixed": actual_fixed, "door": actual_door}
    q_samples = [list(q_res_deg)] + [list(t["q_deg"]) for t in joints]
    cert = G.certify_resume_bridge(
        actual_poses={b: {"pos_m": actual[b][0], "R": actual[b][1]} for b in G.BODIES},
        target_sequence=planned, q_res_deg=list(q_res_deg), q_samples_deg=q_samples,
        fixed_vertices_local_m=static["fixed_vertices_local_m"],
        door_vertices_local_m=static["door_vertices_local_m"],
        obstacle_cells=static["obstacle_cells"], joint_limits_deg=static["joint_limits_deg"],
        joint_point_radius_bounds_m=static["joint_point_radius_bounds_m"],
        timestep_s=static["timestep_s"], numeric_epsilon_m=static["numeric_epsilon_m"],
        length_unit_l_m=static["length_unit_l_m"], voxel_size_m=static["voxel_size_m"],
        domain_max_coord_m=static["domain_max_coord_m"], qmin=static["qmin"],
        half_ulp=static["half_ulp"],
        bootstrap=_boot_inputs(actual, {b: (planned[0][b]["pos_m"], planned[0][b]["R"])
                                        for b in G.BODIES}, static["dts_s"]),
        orthonormal_tol=static["orthonormal_tol"], unit_norm_tol=static["unit_norm_tol"],
        max_align_rotation_deg=static["max_align_rotation_deg"])
    out = {
        "artifact": "W13R_BRIDGE_CLEARANCE_DECISION_V4",
        "decision_point": "resume_fk_here() 직후 · align_move('transport', ...) 이전 · bridge 첫 DoDynamicsThenSync 이전",
        "sim_t_s": float(sim_t_s), "sync_index": int(sync_index), "wall_s": float(wall_s),
        "z_reached_m": None if z_reached_m is None else float(z_reached_m), "ik_resume": ik_resume,
        "q_res_deg": [float(v) for v in q_res_deg], "q_post_lift_deg": [float(v) for v in q_post_lift_deg],
        "door_q_deg_actual": float(door_q_deg_actual),
        "door_q_deg_commanded": float(door_q_deg_commanded),
        "door_vs_fixed_residual_at_decision": {
            "pos_m": float(np.linalg.norm(np.asarray(actual_door[0], float)
                                          - (np.asarray(actual_fixed[0], float)
                                             + np.asarray(actual_fixed[1], float)
                                             @ static["door_hinge_offset_owner_m"]))),
            "note": "보고용 관측. 실행 운동 상한은 두 몸체를 독립으로 보므로 이 값에 의존하지 않는다."},
        "n_align_sync_targets": len(align), "n_joint_sync_targets": len(joints),
        "planned_certificate": _thin(cert),
        "static_inputs": {k: static[k] for k in (
            "obstacle_cells", "joint_limits_deg", "joint_point_radius_bounds_m", "door_q_open_deg",
            "timestep_s", "dts_s", "internal_steps_N", "call_elapsed_upper_bound_s",
            "length_unit_l_m", "voxel_size_m", "domain_max_coord_m", "qmin", "half_ulp",
            "unit_norm_tol", "numeric_epsilon_m", "orthonormal_tol", "transport_speed_m_s",
            "close_deg_s")},
        "numeric_receipt": static["numeric_receipt"],
        "static_provenance": static["provenance"],
        "pass": bool(cert.get("pass")), "verdict": cert.get("verdict"),
        "non_claims": [
            "고정 fixture 와의 기하 충돌 없음만 인증한다. 접촉 성공·서보 실현성 증거가 아니다.",
            "용기 셀은 48각형 축정렬 AABB 겉包이라 보수적이다.",
            "수치 허용치 네 항은 전부 설치본 고정 입력에서 계산했고 관측 여유에 맞추지 않았다.",
            "사후 잔차 측정은 진단이며 pre-step 안전 증명이 아니다.",
        ],
    }
    if not out["pass"]:
        out["abort_detail"] = {"stage": "planned_certificate", "input_error": cert.get("input_error"),
                               "n_separation_failures_total": cert.get("n_separation_failures_total"),
                               "limit_failures": cert.get("limit_failures"),
                               "bootstrap": cert.get("bootstrap"),
                               "joint_radius_cross_check": cert.get("joint_radius_cross_check"),
                               "global_worst": cert.get("global_worst")}
    out["_cert_full"] = cert
    out["_planned"] = planned
    out["_precheck"] = SyncPrecheck(static, planned)
    return out


def gated_bridge(*, static, cert_kwargs, physics_step_counter, install_step_guard, bridge_moves, record):
    """인증 → 기록 → (PASS 면) per-sync precheck 를 걸고 bridge 실행. FAIL 이면 물리 step 0 으로 중단."""
    steps_before = int(physics_step_counter())
    cert = certify_bridge(static, **cert_kwargs)
    steps_after = int(physics_step_counter())
    cert["physics_steps_before_certify"] = steps_before
    cert["physics_steps_after_certify"] = steps_after
    cert["certify_consumed_zero_physics"] = bool(steps_after == steps_before)
    record(cert)
    if not cert["pass"]:
        cert["physics_steps_at_abort"] = steps_after
        cert["aborted_before_any_bridge_physics"] = bool(steps_after == steps_before)
        raise ClearanceUncertified(cert)
    pre = cert["_precheck"]
    install_step_guard(pre)
    try:
        for mv in bridge_moves:
            mv()
    except PrecheckFailed as exc:
        cert["pass"] = False
        cert["verdict"] = G.UNCERTIFIED
        cert["abort_detail"] = {"stage": "per_sync_precheck", "failed_row": exc.row,
                                "summary": pre.summary()}
        raise ClearanceUncertified(cert) from exc
    finally:
        install_step_guard(None)
        cert["precheck_summary"] = pre.summary()
        cert["physics_steps_after_bridge"] = int(physics_step_counter())
    if not cert["precheck_summary"]["all_ok"] or not cert["precheck_summary"]["all_planned_consumed"]:
        cert["pass"] = False
        cert["verdict"] = G.UNCERTIFIED
        cert["abort_detail"] = {"stage": "per_sync_precheck_summary", "summary": cert["precheck_summary"]}
        raise ClearanceUncertified(cert)
    return cert


def dump_json_safe(cert):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    return conv(cert)


INTERVAL_COLUMNS = ["interval_index", "body_code", "gap_m", "motion_bound_m", "start_ball_m",
                    "numeric_epsilon_m", "slack_m", "chord_m", "theta_rad", "part_radius_m",
                    "ideal_motion_m", "position_lattice_m", "tracker_read_m",
                    "quaternion_rotation_m", "dts_s"]
BODY_CODE = {"fixed": 0.0, "door": 1.0}


def intervals_to_arrays(cert):
    """계획 인증 구간을 NPZ 배열로 펴 낸다(감사가 독립 재계산할 수 있게). 몸체별 한 행."""
    rows, names = [], []
    c = cert.get("_cert_full") or {}
    for r in c.get("intervals", []):
        for body, v in r["parts"].items():
            t = v.get("allowance_terms") or {}
            rows.append([float(r["index"]), BODY_CODE.get(body, -1.0),
                         float(v.get("gap_m", np.nan)), float(v.get("motion_bound_m", np.nan)),
                         float(v.get("start_ball_m", np.nan)), float(v.get("numeric_epsilon_m", np.nan)),
                         float(v.get("slack_m", np.nan)), float(v.get("chord_m", np.nan)),
                         float(v.get("theta_rad", np.nan)), float(v.get("part_radius_m", np.nan)),
                         float(t.get("ideal_motion_m", np.nan)),
                         float(t.get("position_lattice_m", np.nan)),
                         float(t.get("tracker_read_m", np.nan)),
                         float(t.get("quaternion_rotation_m", np.nan)),
                         float(r.get("dts_s", np.nan))])
            names.append(f"{r['segment']}|{body}|{v.get('cell')}|{v.get('axis')}")
    return (np.asarray(rows, np.float64) if rows else np.zeros((0, len(INTERVAL_COLUMNS)), np.float64),
            np.asarray(names) if names else np.zeros((0,), dtype="<U1"))


PRECHECK_COLUMNS = ["sync_ordinal", "ok", "worst_gap_m", "worst_motion_bound_m", "worst_slack_m",
                    "plan_pos_dev_fixed_m", "plan_pos_dev_door_m", "plan_rot_dev_fixed_rad",
                    "plan_rot_dev_door_rad", "qnorm_fixed", "qnorm_door", "omega_norm_fixed_rad_s",
                    "omega_norm_door_rad_s", "omega_bound_rad_s", "dts_s"]


def precheck_to_arrays(cert):
    pre = cert.get("_precheck")
    rows = getattr(pre, "rows", []) if pre is not None else []
    out = []
    for r in rows:
        w = r.get("worst") or {}
        pl = r.get("plan") or {}
        bo = r.get("bootstrap") or {}
        out.append([float(r.get("sync_ordinal", np.nan)), 1.0 if r.get("ok") else 0.0,
                    float(w.get("gap_m", np.nan)), float(w.get("motion_bound_m", np.nan)),
                    float(w.get("slack_m", np.nan)),
                    float(pl.get("fixed", {}).get("pos_dev_m", np.nan)),
                    float(pl.get("door", {}).get("pos_dev_m", np.nan)),
                    float(pl.get("fixed", {}).get("rot_dev_rad", np.nan)),
                    float(pl.get("door", {}).get("rot_dev_rad", np.nan)),
                    float(bo.get("fixed", {}).get("quat_norm", np.nan)),
                    float(bo.get("door", {}).get("quat_norm", np.nan)),
                    float(bo.get("fixed", {}).get("omega_norm_rad_s", np.nan)),
                    float(bo.get("door", {}).get("omega_norm_rad_s", np.nan)),
                    float(bo.get("fixed", {}).get("omega_bound_rad_s", np.nan)),
                    float(r.get("dts_s", np.nan))])
    return np.asarray(out, np.float64) if out else np.zeros((0, len(PRECHECK_COLUMNS)), np.float64)


def planned_targets_to_arrays(cert):
    """계획한 per-sync 명령 목표열(고정부·문)을 NPZ 배열로 낸다."""
    seq = cert.get("_planned") or []
    if not seq:
        z4 = np.zeros((0, 4), np.float64)
        return (np.zeros((0, 3), np.float64), z4, np.zeros((0, 3), np.float64), z4,
                np.zeros((0,), dtype="<U1"))
    fp = np.asarray([np.asarray(t["fixed"]["pos_m"], float) for t in seq], np.float64)
    fq = np.asarray([K.mat_to_quat_xyzw(np.asarray(t["fixed"]["R"], float)) for t in seq], np.float64)
    dp = np.asarray([np.asarray(t["door"]["pos_m"], float) for t in seq], np.float64)
    dq = np.asarray([K.mat_to_quat_xyzw(np.asarray(t["door"]["R"], float)) for t in seq], np.float64)
    lab = np.asarray([str(t["segment"]) for t in seq])
    return fp, fq, dp, dq, lab
