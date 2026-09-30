"""W13 resume bridge — production fail-closed clearance certifier V4 (순수 CPU).

solver·하드웨어·파일·시계·전역상태 접근 0. 입력 배열과 주입된 순수 FK 함수만 쓴다.
모든 구간에서 **엄격한 world-axis 분리 여유 > (시작 불확실성 + 설치본 근거 허용치 합) + epsilon**
일 때만 인증한다. 결론이 안 나면 PASS 로 추정하지 않고 `CLEARANCE_UNCERTIFIED` 로 멈춘다.

────────────────────────────────────────────────────────────────────────────────
계보
    원본  = audit `runtime_bridge_guard_01/bridge_clearance_guard.py`
            sha256 `127a017c40a68c9888cc6d33335697e1b7ceb0d6ead44bd97a81d50bf1107b66` (실측 일치).
            바이트 보존 사본 = 같은 폴더 `bridge_clearance_guard_audit_original.py`.
    후보1 = V3 (설치본 duration 상한까지 반영). 보존 =
            `candidates/candidate_01_v3_preCorrection/` (+ SHA256SUMS.txt). 수정하지 않는다.
    현재  = V4. 코디네이터 `msg_2e5ccf261e8f`·`msg_14774d23bc62` 의 정정을 반영.

델타 (원본 대비 누적. 판정 의미 = 엄격 부등식·fail-closed·INCONCLUSIVE 중단 은 불변)
    D1 경로 모형이 실제 명령열이 아니었다 → `align_move`/`joint_move` 의 **정확한 per-sync 목표열**을 인증.
    D2 문 변위 상한이 linked-door **원점 현**이라 일반 호에 부족(감사 확인) → 아래 D7 로 대체·확장.
    D3 추종 잔차의 경험적 가정(0.5 mm, 그리고 k_overrun=2) **전부 철회**. 설치본 유도 상한만 쓴다.
    D4 회전행렬 검증 없음(비정규·반사 통과) → `_rotmat()` 이 직교성·det>0 강제.
    D5 θ≈π geodesic 축 부호 모호 → 인증 거부.
    D6 `separation_failures` 무제한 → 기록 상한 + **총 개수로 판정**.
    D7 **강체 힌지 가정 제거**(`msg_2e5ccf261e8f`). 실행 운동 상한에서 문 포즈를 고정부에서 유도하지 않는다.
       고정부와 문은 **각자의 실측 포즈·각자의 명령 목표·각자의 v/omega·각자의 국소 반경**으로
       독립 인증한다. servo() 가 두 트래커에 따로 명령하므로 0 이 아닌 door-vs-fixed 잔차가 그대로 덮인다.
    D8 **수치 허용치 placeholder 제거**. `bridge_numerics` 의 설치본 근거 4항
       (ideal motion / position lattice / tracker read / quaternion rotation)을 **따로** 계산해 더한다.
       `l`·`voxelSize` 가 없으면 추정하지 않고 fail-closed(= preflight NO-GO, SOURCE_EVIDENCE §1).
    D9 **bootstrap 검사**(ERRATUM_02 E5): 첫 bridge 물리 호출 전에 트래커 쿼터니언 유한·노름>=qmin·
       단위 허용오차 내, 명령 `||omega|| <= pi/D` 유한을 강제한다.
    D10 관절 반경 상한을 **판정식에서 제거**(감사 지적: 검증 없이 신뢰하던 값). 연속 sync 목표 사이에서
       owner 는 규정속도로 직선+등속회전하므로 chord+θ 가 정확한 모형이다. rho 는 교차검증용으로만 남긴다.

주장하지 않는 것
    고정 fixture 와의 기하 충돌 없음만 인증한다. 입자/툴 접촉 성공·서보 실현성·물리 동등성·
    dt 수렴·목표 도달·배출 성공의 증거가 아니다. AABB 분리축은 충분조건이라 보수적이고,
    FAIL 은 "충돌 증명" 이 아니라 "증명 실패" 다. 사후 잔차 측정은 **진단**이며 pre-step 안전 증명이 아니다.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

import bridge_numerics as BN

ARTIFACT = "W13R_RUNTIME_BRIDGE_CLEARANCE_CERTIFICATE_V4"
UNCERTIFIED = "CLEARANCE_UNCERTIFIED"
CERTIFIED = "CLEARANCE_CERTIFIED"
BODIES = ("fixed", "door")


# ── primitive (원본 유지 + D4) ───────────────────────────────────────────────
def _arr(x, shape, name):
    a = np.asarray(x, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {a.shape}")
    return a


def _rotmat(R, name, tol):
    """D4: 진짜 회전행렬만 통과. 비정규·반사(det<0)는 모든 변위 상한을 깨뜨린다."""
    M = _arr(R, (3, 3), name)
    err = float(np.abs(M.T @ M - np.eye(3)).max())
    det = float(np.linalg.det(M))
    if err > float(tol):
        raise ValueError(f"{name} is not orthonormal: |RtR-I|inf={err:.3e} > {float(tol):.3e}")
    if det <= 0.0:
        raise ValueError(f"{name} is a reflection or singular: det={det:.6e} (must be > 0)")
    if abs(det - 1.0) > float(tol):
        raise ValueError(f"{name} is not a proper rotation: det={det:.12f}")
    return M


def _rotvec(R):
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-14:
        return np.zeros(3)
    if math.pi - th < 1e-7:
        vals, vecs = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = vecs[:, int(np.argmax(vals))]
        return axis / np.linalg.norm(axis) * th
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return axis * th


def _world(local_vertices, p, R):
    return (R @ local_vertices.T).T + p


def _axis_gap(vertices, lo, hi):
    """세계 축 분리 여유의 최대값과 그 축. 음수면 그 축으로는 분리를 증명하지 못한다."""
    vlo, vhi = vertices.min(0), vertices.max(0)
    gaps = np.maximum(lo - vhi, vlo - hi)
    k = int(np.argmax(gaps))
    return float(gaps[k]), k


def _cells(obstacle_cells):
    out = []
    for i, c in enumerate(obstacle_cells):
        if not isinstance(c, Mapping):
            raise ValueError(f"cell {i} must be a mapping")
        lo = _arr(c.get("min_m"), (3,), f"cell[{i}].min_m")
        hi = _arr(c.get("max_m"), (3,), f"cell[{i}].max_m")
        if np.any(hi <= lo):
            raise ValueError(f"cell {i} has non-positive extent")
        out.append((str(c.get("name", f"cell_{i}")), lo, hi))
    if not out:
        raise ValueError("at least one obstacle cell is required")
    return out


def _limit_failures(q_samples, limits):
    """관절공간 직선 경로 전 구간 — 샘플 전체의 min/max 를 본다(끝점만 보지 않는다)."""
    Q = np.asarray(q_samples, float)
    if Q.ndim != 2 or Q.shape[1] != len(limits):
        return [{"reason": "limit_count_mismatch",
                 "expected": int(Q.shape[1]) if Q.ndim == 2 else None, "actual": len(limits)}]
    bad = []
    for j, lim in enumerate(limits):
        lo, hi = float(lim[0]), float(lim[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            bad.append({"joint": j, "reason": "invalid_limit", "limit_deg": [lo, hi]})
            continue
        vmin, vmax = float(Q[:, j].min()), float(Q[:, j].max())
        if vmin < lo or vmax > hi:
            bad.append({"joint": j, "reason": "out_of_limit", "range_deg": [vmin, vmax],
                        "limit_deg": [lo, hi]})
    return bad


# ── 몸체 하나 · 구간 하나 ────────────────────────────────────────────────────
class _BodyGeom:
    """한 추적 강체의 국소 정점과 고정 반경. **다른 몸체에서 유도하지 않는다**(D7)."""

    def __init__(self, name, local_vertices_m):
        v = np.asarray(local_vertices_m, float)
        if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or len(v) < 3:
            raise ValueError(f"{name} local vertices must be finite Nx3 with >= 3 rows")
        self.name = str(name)
        self.v = v
        self.r_max = float(np.linalg.norm(v, axis=1).max())   # 자기 owner 원점 기준 최대 국소 반경


class _Certifier:
    """구간별 분리 증명. 모든 수치는 float64. 허용치는 몸체마다 따로 계산한다."""

    def __init__(self, *, cells, bodies, eps, max_failures, num_cfg):
        self.cells = cells
        self.bodies = bodies                    # {"fixed": _BodyGeom, "door": _BodyGeom}
        self.eps = float(eps)
        self.max_failures = int(max_failures)
        self.num = dict(num_cfg)                # D/h/N/l/domain_max/qmin/half_ulp
        self.intervals = []
        self.failures = []
        self.n_failures = 0

    def allowance(self, body, chord_m, theta_rad, delta_p=None, rotvec=None):
        return BN.allowance_terms(
            chord_m=chord_m, theta_rad=theta_rad, r_max_local_m=self.bodies[body].r_max,
            requested_duration_s=self.num["D_s"], timestep_s=self.num["h_s"],
            n_steps=self.num["N"], l_m=self.num["l_m"], voxel_size_m=self.num["voxel_size_m"],
            domain_max_coord_m=self.num["domain_max_coord_m"], qmin=self.num["qmin"],
            half_ulp=self.num["half_ulp"], delta_p_m=delta_p, rotvec_rad=rotvec,
            position_mode=self.num["position_mode"])

    def interval(self, label, index, poses_actual_or_start, poses_target, start_ball_m, extra=None):
        """poses_* = {"fixed": (p,R), "door": (p,R)}. 두 몸체를 **독립**으로 본다(D7).

        start_ball_m = {"fixed": r, "door": r} — 시작 포즈 불확실 구 반경(실측 시작이면 0).
        """
        row = {"segment": label, "index": int(index), "parts": {}}
        if extra:
            row.update(extra)
        out_ball = {}
        for b in BODIES:
            p0, R0 = poses_actual_or_start[b]
            p1, R1 = poses_target[b]
            # ERRATUM_04 E8: servo 가 실제로 만드는 **delta_p / rotvec** 를 그대로 넘겨
            # binary32 명령 벡터로 상한을 계산한다(크기만 쓰지 않는다).
            dp = np.asarray(p1, float) - np.asarray(p0, float)
            rv = _rotvec(np.asarray(R0, float).T @ np.asarray(R1, float))
            chord = float(np.linalg.norm(dp))
            theta = float(np.linalg.norm(rv))
            terms = self.allowance(b, chord, theta, delta_p=dp, rotvec=rv)
            sb = float(start_ball_m.get(b, 0.0))
            bound = sb + terms["total_m"]
            vv = _world(self.bodies[b].v, np.asarray(p0, float), np.asarray(R0, float))
            best = {"slack_m": math.inf}
            for cname, lo, hi in self.cells:
                gap, ax = _axis_gap(vv, lo, hi)
                slack = gap - bound - self.eps
                if slack < best["slack_m"]:
                    best = {"cell": cname, "axis": "xyz"[ax], "gap_m": gap, "motion_bound_m": bound,
                            "start_ball_m": sb, "part_radius_m": self.bodies[b].r_max,
                            "numeric_epsilon_m": self.eps, "slack_m": slack}
                if slack <= 0:
                    self.n_failures += 1
                    if len(self.failures) < self.max_failures:
                        self.failures.append({"segment": label, "index": int(index), "part": b,
                                              "cell": cname, "axis_best": "xyz"[ax], "gap_m": gap,
                                              "motion_bound_m": bound, "slack_m": slack})
            row["parts"][b] = dict(best, chord_m=chord, theta_rad=theta, allowance_terms=terms)
            # 다음 구간의 시작 불확실 구 = 이 구간의 허용치 합(실측은 목표에서 그만큼 안에 있다).
            out_ball[b] = terms["total_m"]
        self.intervals.append(row)
        return out_ball

    def worst(self):
        return min((dict({k2: v2 for k2, v2 in v.items() if k2 != "allowance_terms"},
                         segment=row["segment"], index=row["index"], part=k)
                    for row in self.intervals for k, v in row["parts"].items()),
                   key=lambda x: x["slack_m"], default=None)


def _num_cfg(D_s, h_s, l_m, domain_max_coord_m, qmin, half_ulp, voxel_size_m=None,
             position_mode="voxel_fallback"):
    N, acc = BN.internal_steps(D_s, h_s)
    cfg = {"D_s": float(D_s), "h_s": float(h_s), "N": N, "accumulator_at_stop_s": acc,
           "l_m": l_m, "voxel_size_m": voxel_size_m, "domain_max_coord_m": domain_max_coord_m,
           "qmin": float(qmin), "half_ulp": bool(half_ulp), "position_mode": position_mode,
           "call_elapsed_upper_bound_s": BN.call_elapsed_upper_bound_s(D_s, h_s)}
    if l_m is not None and voxel_size_m is not None and domain_max_coord_m is not None:
        # ERRATUM_04 E9: 합계만 남기지 말고 **네 구성량을 원시로** 보존한다.
        cfg["position_lattice_constituents"] = BN.position_lattice_constituents(
            N, l_m, voxel_size_m, domain_max_coord_m)
    return cfg


def _poses(d, name, tol):
    return {b: (_arr(d[b]["pos_m"], (3,), f"{name}.{b}.pos_m"),
                _rotmat(d[b]["R"], f"{name}.{b}.R", tol)) for b in BODIES}


# ── 공개 API ① 전체 계획 bridge 인증 ────────────────────────────────────────
def certify_resume_bridge(
    *,
    actual_poses: Mapping,                   # {"fixed": {pos_m,R}, "door": {pos_m,R}} 실측
    target_sequence: Sequence[Mapping],      # [{"fixed":{pos_m,R}, "door":{pos_m,R}, "dts_s":, ...}]
    q_res_deg: Sequence[float],
    q_samples_deg: Sequence[Sequence[float]],
    fixed_vertices_local_m,
    door_vertices_local_m,
    obstacle_cells: Sequence[Mapping],
    joint_limits_deg: Sequence[Sequence[float]],
    joint_point_radius_bounds_m: Sequence[float],
    timestep_s: float,
    numeric_epsilon_m: float,
    length_unit_l_m=None,
    voxel_size_m=None,
    domain_max_coord_m=None,
    qmin: float = BN.QMIN_DEFAULT,
    half_ulp: bool = False,
    position_mode: str = "voxel_fallback",
    bootstrap: Mapping = None,
    orthonormal_tol: float = 1e-9,
    unit_norm_tol: float = 1e-6,
    max_align_rotation_deg: float = 170.0,
    max_failures_recorded: int = 200,
):
    """실측 포즈 → 정확한 per-sync 명령 목표열 전 구간을 fail-closed 로 인증한다(두 몸체 독립)."""
    try:
        eps = float(numeric_epsilon_m)
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("numeric_epsilon_m must be finite and non-negative")
        if not target_sequence:
            raise ValueError("target_sequence must be non-empty")
        lims = [[float(a), float(b)] for a, b in joint_limits_deg]
        q_res = _arr(q_res_deg, (len(lims),), "q_res_deg")
        radii = _arr(joint_point_radius_bounds_m, q_res.shape, "joint_point_radius_bounds_m")
        if np.any(radii <= 0):
            raise ValueError("joint radius bounds must be positive")
        cells = _cells(obstacle_cells)
        bodies = {"fixed": _BodyGeom("fixed", fixed_vertices_local_m),
                  "door": _BodyGeom("door", door_vertices_local_m)}
        A = _poses(actual_poses, "actual_poses", orthonormal_tol)

        # D9 bootstrap — 첫 물리 호출 전에 반드시 통과 (ERRATUM_02 E5)
        boot = dict(bootstrap or {})
        D0 = float(target_sequence[0]["dts_s"])
        boot_res = {}
        for b in BODIES:
            need = boot.get(b)
            if need is None:
                raise ValueError(f"bootstrap inputs for body '{b}' are required "
                                 "(tracker quaternion xyzw and commanded local omega)")
            boot_res[b] = BN.bootstrap_checks(
                quat_xyzw=need["quat_xyzw"], omega_rad_s=need["omega_rad_s"],
                requested_duration_s=D0, qmin=qmin, unit_norm_tol=unit_norm_tol)
        if not all(v["pass"] for v in boot_res.values()):
            raise ValueError(f"bootstrap validation failed (ERRATUM_02 E5): "
                             f"{ {b: v for b, v in boot_res.items() if not v['pass']} }")

        num = _num_cfg(D0, timestep_s, length_unit_l_m, domain_max_coord_m, qmin, half_ulp,
                       voxel_size_m=voxel_size_m, position_mode=position_mode)
        seq = []
        for i, t in enumerate(target_sequence):
            dts = float(t["dts_s"])
            if not (math.isfinite(dts) and dts > 0):
                raise ValueError(f"target_sequence[{i}].dts_s must be finite positive")
            if abs(dts - D0) > 0:
                raise ValueError("all bridge sync targets must share one frozen dts_s; "
                                 f"got {dts} vs {D0} at index {i}")
            seq.append((str(t.get("segment", "bridge")), i + 1,
                        _poses(t, f"target_sequence[{i}]", orthonormal_tol), dts,
                        t.get("q_deg")))

        # D5 정렬 geodesic 이 π 근방이면 축 부호가 모호 → 거부(고정부 기준)
        rot_total = float(math.degrees(np.linalg.norm(
            _rotvec(A["fixed"][1].T @ seq[-1][2]["fixed"][1]))))
        rot_first = float(math.degrees(np.linalg.norm(
            _rotvec(A["fixed"][1].T @ seq[0][2]["fixed"][1]))))
        if max(rot_total, rot_first) > float(max_align_rotation_deg):
            raise ValueError(f"align rotation {max(rot_total, rot_first):.6f} deg exceeds "
                             f"max_align_rotation_deg {float(max_align_rotation_deg)} "
                             "(geodesic axis sign is ambiguous near pi; refusing to assume a path)")

        C = _Certifier(cells=cells, bodies=bodies, eps=eps, max_failures=max_failures_recorded,
                       num_cfg=num)
        start = A
        ball = {b: 0.0 for b in BODIES}          # 구간 1 은 실측에서 시작 → 불확실 구 0
        ball_trace = []
        for lbl, idx, tgt, dts, q in seq:
            extra = {"dts_s": dts}
            if q is not None:
                extra["q_deg"] = [float(v) for v in q]
            ball = C.interval(lbl, idx, start, tgt, ball, extra=extra)
            ball_trace.append({"segment": lbl, "index": int(idx),
                               "start_ball_out_m": {b: ball[b] for b in BODIES}})
            start = tgt

        # D10 rho 교차검증(판정에 쓰지 않는다)
        xcheck = []
        qs = np.asarray(q_samples_deg, float) if q_samples_deg is not None else None
        if qs is not None and len(qs) >= 2:
            for k in range(1, len(qs)):
                jb = float(np.dot(radii, np.abs(np.radians(qs[k] - qs[k - 1]))))
                rows = [r for r in C.intervals if r.get("q_deg") is not None]
                if k - 1 < len(rows):
                    pr = rows[k - 1]["parts"]
                    lhs = max(pr[b]["chord_m"] + bodies[b].r_max * pr[b]["theta_rad"] for b in BODIES)
                    xcheck.append({"index": k, "chord_plus_r_theta_m": lhs,
                                   "joint_radius_bound_m": jb,
                                   "rho_is_upper_bound": bool(lhs <= jb + 1e-12)})
        rho_bad = [r for r in xcheck if not r["rho_is_upper_bound"]]
        lim_fail = _limit_failures(q_samples_deg if q_samples_deg is not None else [q_res.tolist()], lims)
        ok = (C.n_failures == 0) and not lim_fail and not rho_bad
        return {
            "artifact": ARTIFACT, "verdict": CERTIFIED if ok else UNCERTIFIED, "pass": bool(ok),
            "rule": ("PASS only if, for EACH tracked body independently (fixed and door, no rigid-hinge "
                     "derivation), every convex-cell AABB has a world-axis separating gap strictly greater "
                     "than start_ball + ideal_motion + position_lattice + tracker_read + "
                     "quaternion_rotation + epsilon, at every exact per-sync command target, with all "
                     "planned joint samples inside applicable limits and bootstrap validation passed."),
            "bodies": {b: {"n_local_vertices": int(len(bodies[b].v)), "r_max_local_m": bodies[b].r_max}
                       for b in BODIES},
            "bootstrap": boot_res,
            "numeric_inputs": num,
            "numeric_evidence": BN.EVIDENCE,
            "inputs": {"q_res_deg": q_res.tolist(), "n_sync_targets": len(seq), "n_cells": len(cells),
                       "numeric_epsilon_m": eps, "orthonormal_tol": float(orthonormal_tol),
                       "unit_norm_tol": float(unit_norm_tol),
                       "max_align_rotation_deg": float(max_align_rotation_deg),
                       "align_rotation_total_deg": rot_total,
                       "joint_point_radius_bounds_m": radii.tolist(), "joint_limits_deg": lims,
                       "voxel_size_m": voxel_size_m, "length_unit_l_m": length_unit_l_m,
                       "domain_max_coord_m": domain_max_coord_m, "qmin": float(qmin),
                       "tracker_ulp_mode": "half_ulp" if half_ulp else "full_ulp"},
            "n_intervals": len(C.intervals), "global_worst": C.worst(),
            "intervals": C.intervals, "start_ball_trace": ball_trace,
            "joint_radius_cross_check": {"n": len(xcheck), "violations": rho_bad,
                                         "role": "cross-check only, not in the PASS/FAIL inequality"},
            "limit_failures": lim_fail,
            "separation_failures": C.failures, "n_separation_failures_total": C.n_failures,
            "separation_failures_truncated": bool(C.n_failures > len(C.failures)),
            "failure_class": None if ok else UNCERTIFIED,
        }
    except Exception as exc:                      # 결측·형식오류·비유한·비회전·고정입력 누락 = fail-closed
        return {"artifact": ARTIFACT, "verdict": UNCERTIFIED, "pass": False,
                "failure_class": UNCERTIFIED, "input_error": f"{type(exc).__name__}: {exc}"}


# ── 공개 API ② per-sync fail-before-DoDynamics precheck ──────────────────────
def precheck_sync(*, actual_poses, target_poses, dts_s, planned_target_poses,
                  plan_tol_m, plan_tol_rad, fixed_vertices_local_m, door_vertices_local_m,
                  obstacle_cells, timestep_s, numeric_epsilon_m, length_unit_l_m=None,
                  voxel_size_m=None, domain_max_coord_m=None, qmin=BN.QMIN_DEFAULT, half_ulp=False,
                  position_mode="voxel_fallback",
                  bootstrap=None, orthonormal_tol=1e-9, unit_norm_tol=1e-6):
    """한 sync 를 **DoDynamicsThenSync 전에** 검사한다. 실측 포즈 + 실제 명령 목표만 쓴다.

    시작이 실측이라 불확실 구 0 — 이 구간이 pre-step 안전 증명의 본체다.
    두 몸체(fixed/door)를 **독립**으로 본다. 강체 힌지 가정 없음(D7).
    """
    try:
        eps = float(numeric_epsilon_m)
        cells = _cells(obstacle_cells)
        bodies = {"fixed": _BodyGeom("fixed", fixed_vertices_local_m),
                  "door": _BodyGeom("door", door_vertices_local_m)}
        A = _poses(actual_poses, "actual", orthonormal_tol)
        T = _poses(target_poses, "target", orthonormal_tol)
        P = _poses(planned_target_poses, "planned", orthonormal_tol)
        num = _num_cfg(float(dts_s), timestep_s, length_unit_l_m, domain_max_coord_m, qmin, half_ulp,
                       voxel_size_m=voxel_size_m, position_mode=position_mode)

        plan = {}
        for b in BODIES:
            dp = float(np.linalg.norm(T[b][0] - P[b][0]))
            dr = float(np.linalg.norm(_rotvec(P[b][1].T @ T[b][1])))
            plan[b] = {"pos_dev_m": dp, "rot_dev_rad": dr,
                       "ok": bool(dp <= float(plan_tol_m) and dr <= float(plan_tol_rad))}
        boot_res = {}
        for b in BODIES:
            need = (bootstrap or {}).get(b)
            if need is None:
                raise ValueError(f"per-sync bootstrap inputs for body '{b}' are required")
            boot_res[b] = BN.bootstrap_checks(
                quat_xyzw=need["quat_xyzw"], omega_rad_s=need["omega_rad_s"],
                requested_duration_s=float(dts_s), qmin=qmin, unit_norm_tol=unit_norm_tol)

        C = _Certifier(cells=cells, bodies=bodies, eps=eps, max_failures=32, num_cfg=num)
        C.interval("precheck_actual_to_target", 0, A, T, {b: 0.0 for b in BODIES},
                   extra={"dts_s": float(dts_s)})
        sep_ok = C.n_failures == 0
        plan_ok = all(v["ok"] for v in plan.values())
        boot_ok = all(v["pass"] for v in boot_res.values())
        return {"ok": bool(sep_ok and plan_ok and boot_ok), "separation_ok": sep_ok,
                "plan_match_ok": plan_ok, "bootstrap_ok": boot_ok,
                "plan": plan, "bootstrap": boot_res, "numeric_inputs": num,
                "worst": C.worst(), "n_separation_failures": C.n_failures,
                "failures": C.failures, "dts_s": float(dts_s),
                "interval": C.intervals[0] if C.intervals else None,
                "role_note": ("separation_ok is the PRE-STEP safety proof; any post-step residual "
                              "measurement elsewhere is DIAGNOSTIC only (msg_c0a8c4367a15).")}
    except Exception as exc:                                         # noqa: BLE001
        return {"ok": False, "input_error": f"{type(exc).__name__}: {exc}",
                "separation_ok": False, "plan_match_ok": False, "bootstrap_ok": False}
