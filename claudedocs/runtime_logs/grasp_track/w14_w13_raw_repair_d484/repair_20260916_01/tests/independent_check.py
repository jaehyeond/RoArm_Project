"""독립 검사기 — 생산 모듈(inventory_geometry / sim_deme_scoop_s1.expand_spheres / scipy / w13_kinematics) 을
import 하지 않는다. 같은 규약(RAW_SCHEMA_REQUIRED.md + 메타데이터 선언)을 **다른 식**으로 구현한다:
  · 회전: Hamilton 곱 sandwich q⊗v⊗q* (생산=scipy 회전행렬, 감사=cross-product 식)
  · in_* containment: clump 축정렬 경계상자(AABB) 의 min/max 로 판정
  · near_* 밴드: 구 하나씩 확장 상자와의 겹침 검사
  · phase-only 전환: itertools.groupby 누적 길이
"""
import itertools
import math

import numpy as np

LABELS = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]
L = {n: i for i, n in enumerate(LABELS)}


def qmul(a, b):
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def rotate_hamilton(q_xyzw, v):
    """q (n,4) · v (k,3) → (n,k,3) = q ⊗ (v,0) ⊗ q*  (단위화 후)."""
    q = np.asarray(q_xyzw, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    n, k = len(q), len(v)
    qb = np.repeat(q[:, None, :], k, axis=1)
    vq = np.concatenate([np.broadcast_to(np.asarray(v, float)[None], (n, k, 3)), np.zeros((n, k, 1))], -1)
    qc = qb * np.array([-1.0, -1.0, -1.0, 1.0])
    return qmul(qmul(qb, vq), qc)[..., :3]


def rot_matrix(q_xyzw):
    """R 의 열 = 회전된 기저벡터 (Hamilton 곱으로 계산)."""
    return rotate_hamilton(np.asarray(q_xyzw, float)[None], np.eye(3))[0].T


def spheres_world(pos, quat, offsets):
    return np.asarray(pos, float)[:, None, :] + rotate_hamilton(quat, offsets)


def phase_only_transitions_groupby(codes):
    starts, cursor = [], 0
    for _, grp in itertools.groupby(np.asarray(codes).astype(int).tolist()):
        if cursor > 0:
            starts.append(cursor)
        cursor += len(list(grp))
    return starts


def classify_independent(pos, quat, vel, tool_pos, tool_quat, bin_pos, bin_quat, offsets, radii, meta, bin_fix):
    S = spheres_world(pos, quat, offsets)                       # (n,k,3)
    r = np.asarray(radii, float)[None, :]                       # (1,k)
    n = len(S)
    m = float(meta["classify_margin_m"])
    assert meta["moving_threshold_operator"].startswith(">=")
    moving = np.linalg.norm(np.asarray(vel, float), axis=1) >= float(meta["moving_threshold_m_s"])
    lo, hi = S - r[..., None], S + r[..., None]                 # per-sphere per-axis extents (n,k,3)

    # tool cavity — 메타데이터 방정식 그대로: p_cavity = A @ R(q_tool)^T @ (p - p_tool) + origin
    tc = meta["tool_cavity"]
    A = np.asarray(tc["tool_to_cavity_R"], float)
    Rt = rot_matrix(tool_quat)
    pc = np.einsum("ij,nkj->nki", A @ Rt.T, S - np.asarray(tool_pos, float)) + np.asarray(tc["cavity_origin_l5_m"], float)
    cx, cz = tc["cavity_center_xz_l5_m"]
    radial = np.hypot(pc[..., 0] - cx, pc[..., 2] - cz)
    ay = np.abs(pc[..., 1])
    R_c, hy = float(tc["radius_m"]), float(tc["half_y_m"])
    in_tool = ((radial + r < R_c - m) & (ay + r < hy - m)).all(1)
    near_tool = ((radial - r < R_c + m) & (ay - r < hy + m)).any(1)

    # receiving bin — 정 n각 프리즘(apothem 선언값 사용, semantics 검사)
    assert bin_fix["radius_semantics"] == "circumradius"
    n_side = int(bin_fix["n_theta"])
    apo = float(bin_fix["apothem_m"])
    assert abs(apo - float(bin_fix["circumradius_m"]) * math.cos(math.pi / n_side)) < 1e-15
    Rb = rot_matrix(bin_quat)
    bl = np.einsum("ij,nkj->nki", Rb.T, S - np.asarray(bin_pos, float))
    ang = (np.arange(n_side) + 0.5) * (2 * math.pi / n_side)
    nrm = np.stack([np.cos(ang), np.sin(ang)], 1)
    d = np.einsum("nki,fi->nkf", bl[..., :2], nrm).max(2)
    zf, zr = float(bin_fix["floor_inner_z_m"]), float(bin_fix["rim_z_m"])
    in_bin = ((d + r < apo - m) & (bl[..., 2] - r > zf + m) & (bl[..., 2] + r < zr - m)).all(1)
    near_bin = ((d - r < apo + m) & (bl[..., 2] + r > zf - m) & (bl[..., 2] - r < zr + m)).any(1)

    # source box — clump AABB 로 strict whole-sphere containment
    box = np.asarray(meta["source_bounds_m"], float)            # (3,2)
    aabb_lo, aabb_hi = lo.min(1), hi.max(1)                     # (n,3)
    in_src = (aabb_lo > box[:, 0] + m).all(1) & (aabb_hi < box[:, 1] - m).all(1)
    near_src = ((hi > box[:, 0] - m).all(2) & (lo < box[:, 1] + m).all(2)).any(1)

    code = np.full(n, L["ambiguous"], np.int8)
    left = np.ones(n, bool)
    for mask, rest_label, move_label in ((in_tool, "tool_residual", "tool_residual"),
                                         (near_tool, "ambiguous", "ambiguous"),
                                         (in_bin, "receiving_bin", "in_flight"),
                                         (near_bin, "ambiguous", "ambiguous"),
                                         (in_src, "source", "in_flight"),
                                         (near_src, "ambiguous", "ambiguous")):
        sel = left & mask
        code[sel & ~moving] = L[rest_label]
        code[sel & moving] = L[move_label]
        left &= ~mask
    code[left & moving] = L["in_flight"]
    left &= ~moving
    below = aabb_hi[:, 2] <= float(meta["spill_rest_z_m"])
    code[left & below] = L["spill"]
    code[left & ~below] = L["ambiguous"]
    return code
