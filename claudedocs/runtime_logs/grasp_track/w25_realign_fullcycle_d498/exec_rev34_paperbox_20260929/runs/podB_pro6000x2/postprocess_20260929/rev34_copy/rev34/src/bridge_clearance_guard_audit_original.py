"""Pure CPU fail-closed clearance certificate for the W13 resume bridge.

The module has no solver, hardware, file, clock, or global-state access.  It
returns CLEARANCE_CERTIFIED only when every continuous interval has a world-axis
separating plane whose gap exceeds an analytic motion bound.  Inconclusive is a
hard stop, not an inferred collision.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np


def _arr(x, shape, name):
    a = np.asarray(x, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {a.shape}")
    return a


def _rotvec(R):
    R = _arr(R, (3, 3), "rotation")
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


def _axis_angle(axis, radians):
    axis = _arr(axis, (3,), "axis")
    n = float(np.linalg.norm(axis))
    if n <= 0:
        raise ValueError("axis norm must be positive")
    x, y, z = axis / n
    c, s, C = math.cos(radians), math.sin(radians), 1.0 - math.cos(radians)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def _interp_R(R0, R1, f):
    rv = _rotvec(_arr(R0, (3, 3), "R0").T @ _arr(R1, (3, 3), "R1"))
    th = float(np.linalg.norm(rv))
    return R0 if th < 1e-14 else R0 @ _axis_angle(rv / th, th * f)


def _pose(d, name):
    if not isinstance(d, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return _arr(d.get("pos_m"), (3,), f"{name}.pos_m"), _arr(d.get("R"), (3, 3), f"{name}.R")


def _world(local_vertices, pose):
    p, R = pose
    return (R @ local_vertices.T).T + p


def _axis_gap(vertices, lo, hi):
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


def _limits(q0, q1, limits):
    failures = []
    if len(limits) != len(q0):
        return [{"reason": "limit_count_mismatch", "expected": len(q0), "actual": len(limits)}]
    for i, (v0, v1, lim) in enumerate(zip(q0, q1, limits)):
        lo, hi = float(lim[0]), float(lim[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            failures.append({"joint": i, "reason": "invalid_limit", "limit_deg": [lo, hi]})
        elif min(v0, v1) < lo or max(v0, v1) > hi:
            failures.append({"joint": i, "reason": "out_of_limit", "range_deg": [float(min(v0, v1)), float(max(v0, v1))],
                             "limit_deg": [lo, hi]})
    return failures


def certify_resume_bridge(
    *,
    current_fixed_pose: Mapping,
    current_door_pose: Mapping,
    q_res_deg: Sequence[float],
    q_post_lift_deg: Sequence[float],
    owner_pose_fn: Callable[[Sequence[float]], tuple[Sequence[float], Sequence[Sequence[float]]]],
    fixed_vertices_local_m: Sequence[Sequence[float]],
    door_vertices_local_m: Sequence[Sequence[float]],
    door_hinge_offset_owner_m: Sequence[float],
    door_axis_owner: Sequence[float],
    door_q_deg: float,
    door_q_open_deg: float,
    obstacle_cells: Sequence[Mapping],
    joint_limits_deg: Sequence[Sequence[float]],
    joint_point_radius_bounds_m: Sequence[float],
    numeric_epsilon_m: float = 1e-9,
    n_align: int = 64,
    n_joint: int = 256,
):
    """Certify actual-pose→q_res align and q_res→post-lift joint path.

    `owner_pose_fn(q)` must be a pure FK function returning `(position_m, R)`.
    `joint_point_radius_bounds_m[j]` is a conservative bound from joint j's
    axis to every downstream fixed/door point for the declared bridge.
    """
    try:
        if not callable(owner_pose_fn):
            raise ValueError("owner_pose_fn must be callable")
        if n_align < 1 or n_joint < 1:
            raise ValueError("subdivision counts must be positive")
        eps = float(numeric_epsilon_m)
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("numeric_epsilon_m must be finite and non-negative")
        f_local = np.asarray(fixed_vertices_local_m, float)
        d_local = np.asarray(door_vertices_local_m, float)
        if f_local.ndim != 2 or f_local.shape[1] != 3 or not np.isfinite(f_local).all() or len(f_local) < 3:
            raise ValueError("fixed vertices must be finite Nx3")
        if d_local.ndim != 2 or d_local.shape[1] != 3 or not np.isfinite(d_local).all() or len(d_local) < 3:
            raise ValueError("door vertices must be finite Nx3")
        hinge = _arr(door_hinge_offset_owner_m, (3,), "door_hinge_offset_owner_m")
        axis = _arr(door_axis_owner, (3,), "door_axis_owner")
        if np.linalg.norm(axis) <= 0:
            raise ValueError("door axis norm must be positive")
        q0 = _arr(q_res_deg, (len(joint_limits_deg),), "q_res_deg")
        q1 = _arr(q_post_lift_deg, q0.shape, "q_post_lift_deg")
        radii = _arr(joint_point_radius_bounds_m, q0.shape, "joint_point_radius_bounds_m")
        if np.any(radii <= 0):
            raise ValueError("joint radius bounds must be positive")
        cells = _cells(obstacle_cells)
        pf0, Rf0 = _pose(current_fixed_pose, "current_fixed_pose")
        pd0, Rd0 = _pose(current_door_pose, "current_door_pose")
        pres, Rres = owner_pose_fn(q0.tolist())
        pres, Rres = _arr(pres, (3,), "p_res"), _arr(Rres, (3, 3), "R_res")
        ppost, Rpost = owner_pose_fn(q1.tolist())
        ppost, Rpost = _arr(ppost, (3,), "p_post"), _arr(Rpost, (3, 3), "R_post")
        Rrel = _axis_angle(axis, math.radians(float(door_q_deg) - float(door_q_open_deg)))

        def linked_door(p, R):
            return p + R @ hinge, R @ Rrel

        pdres, Rdres = linked_door(pres, Rres)
        intervals = []
        failures = []

        def certify_interval(label, index, poses_start, bounds):
            row = {"segment": label, "index": index, "parts": {}}
            for part, local, pose_start, bound in (("fixed", f_local, poses_start[0], bounds[0]),
                                                   ("door", d_local, poses_start[1], bounds[1])):
                vv = _world(local, pose_start)
                best = {"slack_m": math.inf}
                for cname, lo, hi in cells:
                    gap, ax = _axis_gap(vv, lo, hi)
                    slack = gap - bound - eps
                    if slack < best["slack_m"]:
                        best = {"cell": cname, "axis": "xyz"[ax], "gap_m": gap,
                                "motion_bound_m": bound, "numeric_epsilon_m": eps,
                                "slack_m": slack}
                    if slack <= 0:
                        failures.append({"segment": label, "index": index, "part": part,
                                         "cell": cname, "axis_best": "xyz"[ax], "gap_m": gap,
                                         "motion_bound_m": bound, "slack_m": slack})
                row["parts"][part] = best
            intervals.append(row)

        # First interval begins at actual tracked poses.  Following align target
        # poses use the exact linked rigid-door target used by the controller.
        target1_f = (pf0 + (pres-pf0)/n_align, _interp_R(Rf0, Rres, 1.0/n_align))
        target1_d = linked_door(*target1_f)
        bf = float(np.linalg.norm(target1_f[0]-pf0) + np.linalg.norm(f_local, axis=1).max() *
                   np.linalg.norm(_rotvec(Rf0.T @ target1_f[1])))
        bd = float(np.linalg.norm(target1_d[0]-pd0) + np.linalg.norm(d_local, axis=1).max() *
                   np.linalg.norm(_rotvec(Rd0.T @ target1_d[1])))
        certify_interval("actual_to_resume_align", 0, ((pf0, Rf0), (pd0, Rd0)), (bf, bd))
        for i in range(1, n_align):
            f = i/n_align
            fn = (i+1)/n_align
            ps = pf0 + (pres-pf0)*f
            Rs = _interp_R(Rf0, Rres, f)
            pn = pf0 + (pres-pf0)*fn
            Rn = _interp_R(Rf0, Rres, fn)
            ds, dn = linked_door(ps, Rs), linked_door(pn, Rn)
            bf = float(np.linalg.norm(pn-ps) + np.linalg.norm(f_local, axis=1).max() *
                       np.linalg.norm(_rotvec(Rs.T @ Rn)))
            bd = float(np.linalg.norm(dn[0]-ds[0]) + np.linalg.norm(d_local, axis=1).max() *
                       np.linalg.norm(_rotvec(ds[1].T @ dn[1])))
            certify_interval("actual_to_resume_align", i, ((ps, Rs), ds), (bf, bd))

        dq = np.radians(q1-q0) / n_joint
        bj = float(np.dot(radii, np.abs(dq)))
        for i in range(n_joint):
            q = q0 + (q1-q0)*(i/n_joint)
            p, R = owner_pose_fn(q.tolist())
            p, R = _arr(p, (3,), "joint_path_pos"), _arr(R, (3, 3), "joint_path_R")
            certify_interval("resume_to_post_lift_joint", i, ((p, R), linked_door(p, R)), (bj, bj))

        limit_failures = _limits(q0, q1, joint_limits_deg)
        global_worst = min((dict(v, segment=r["segment"], index=r["index"], part=k)
                            for r in intervals for k, v in r["parts"].items()),
                           key=lambda x: x["slack_m"])
        certified = not failures and not limit_failures
        return {
            "artifact": "W13_RUNTIME_BRIDGE_CLEARANCE_CERTIFICATE_V1",
            "verdict": "CLEARANCE_CERTIFIED" if certified else "CLEARANCE_UNCERTIFIED",
            "pass": certified,
            "inputs": {"q_res_deg": q0.tolist(), "q_post_lift_deg": q1.tolist(),
                       "door_q_deg": float(door_q_deg), "door_q_open_deg": float(door_q_open_deg),
                       "n_cells": len(cells), "n_align": n_align, "n_joint": n_joint,
                       "numeric_epsilon_m": eps, "joint_point_radius_bounds_m": radii.tolist()},
            "resolved_targets": {"resume_pos_m": pres.tolist(), "resume_R": Rres.tolist(),
                                 "post_lift_pos_m": ppost.tolist(), "post_lift_R": Rpost.tolist()},
            "n_intervals": len(intervals), "global_worst": global_worst,
            "intervals": intervals, "limit_failures": limit_failures,
            "separation_failures": failures,
            "failure_class": None if certified else "CLEARANCE_UNCERTIFIED",
            "rule": "PASS only if every convex-cell AABB has a world-axis separating gap strictly greater than analytic motion bound + epsilon."
        }
    except Exception as exc:  # malformed/missing/nonfinite input is fail-closed
        return {"artifact": "W13_RUNTIME_BRIDGE_CLEARANCE_CERTIFICATE_V1",
                "verdict": "CLEARANCE_UNCERTIFIED", "pass": False,
                "failure_class": "CLEARANCE_UNCERTIFIED",
                "input_error": f"{type(exc).__name__}: {exc}"}
