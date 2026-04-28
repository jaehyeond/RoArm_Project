"""RoArm M3 kinematics — FK + numerical IK (DLS) with v6 warm-start.

URDF source-of-truth: roarm_m3.urdf joint chain.
SDK joint convention == URDF convention (validated 4/24 sim_v1 RMSE 0.43°).
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from numpy import cos, sin, pi


# ---------------------------------------------------------------
# URDF chain (from roarm_m3.urdf, extracted 4/28).
# ---------------------------------------------------------------
_CHAIN = [
    ("world_to_base",   [0,        0,        0.0701],   [0,       0,        0],       None),
    ("base_to_link1",   [0,        0,        0],        [0,       0,        0],       0),
    ("link1_to_link2",  [0,        0,        0.05196],  [-pi/2,   -pi/2,    0],       1),
    ("link2_to_link3",  [0.236815, 0.030002, 0],        [0,       0,        pi/2],    2),
    ("link3_to_link4",  [0,        -0.144586, 0],       [0,       0,        0],       3),
    ("link4_to_link5",  [0.015147, -0.053653, 0],       [pi/2,    pi/2,     0],       4),
    ("link5_to_tcp",    [0,        0,        0.115428], [pi/2,    -pi/2,    0],       None),
]

# v6-derived joint limits (clip to keep IK in training distribution)
JOINT_LIMITS_DEG = {
    "base":     (-90.0, +90.0),    # v6 [-49, +76]
    "shoulder": (-30.0, +75.0),    # v6 [-17, +68]
    "elbow":    (+5.0, +135.0),    # v6 [+9, +126]  ← elbow up region
    "wrist_p":  (-30.0, +90.0),    # v6 [-25, +90] — clipped to v6 max for distribution match
    "wrist_r":  (-90.0, +90.0),    # v6 [-60, +84]
    "gripper":  (-10.0, +100.0),
}


def rpy_R(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
    Ry = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
    Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def Tmat(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_R(*rpy)
    T[:3, 3] = xyz
    return T


def Trot_z(q_rad):
    T = np.eye(4)
    c, s = cos(q_rad), sin(q_rad)
    T[0, 0] = c; T[0, 1] = -s
    T[1, 0] = s; T[1, 1] = c
    return T


def fk_full(joints_deg):
    """Returns (tcp_xyz, R_tcp) where R_tcp is 3x3 rotation in URDF world."""
    q = np.radians(joints_deg)
    T = np.eye(4)
    for name, xyz, rpy, qi in _CHAIN:
        T = T @ Tmat(xyz, rpy)
        if qi is not None:
            T = T @ Trot_z(q[qi])
    return T[:3, 3], T[:3, :3]


def fk_tcp(joints_deg):
    p, _ = fk_full(joints_deg)
    return p


def jacobian_numerical(joints_deg, eps_deg=0.01):
    """3x5 position Jacobian (columns: base, shoulder, elbow, wrist_p, wrist_r).
    Gripper (joint 5) excluded — does not affect TCP."""
    q = np.asarray(joints_deg, dtype=np.float64).copy()
    p0 = fk_tcp(q)
    J = np.zeros((3, 5))
    for i in range(5):
        qp = q.copy(); qp[i] += eps_deg
        qm = q.copy(); qm[i] -= eps_deg
        J[:, i] = (fk_tcp(qp) - fk_tcp(qm)) / (2.0 * eps_deg)
    return J


def clip_joints(q_deg):
    out = np.asarray(q_deg, dtype=np.float64).copy()
    names = ["base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper"]
    for i, name in enumerate(names):
        lo, hi = JOINT_LIMITS_DEG[name]
        out[i] = max(lo, min(hi, out[i]))
    return out


def ik_dls(
    target_xyz,
    q0_deg,
    max_iter=200,
    tol_mm=1.0,
    damping=2.0,
    step_clip_deg=5.0,
    verbose=False,
):
    """Damped Least Squares IK. Position-only (3 DoF target, 5 DoF arm).
    Returns (q_deg, converged, final_err_mm, n_iter).

    damping (deg-scale, since J is in mm/deg): higher = more stable near singular.
    step_clip_deg: max joint step per iter (anti-overshoot).
    """
    q = np.asarray(q0_deg, dtype=np.float64).copy()
    target = np.asarray(target_xyz, dtype=np.float64)

    for it in range(max_iter):
        p = fk_tcp(q)
        err = target - p
        err_mm = np.linalg.norm(err) * 1000.0
        if verbose and it % 10 == 0:
            print(f"    iter{it:3d}: err={err_mm:.2f}mm  q={[f'{x:+.1f}' for x in q[:5]]}")
        if err_mm < tol_mm:
            return q, True, err_mm, it
        J = jacobian_numerical(q)  # 3x5, units: m/deg
        # DLS in deg-units. Note J units are m per deg, so damping λ is m/deg-scale.
        # err is in m; target err 1mm => deg-scale step ~1mm/(L*pi/180) ~ small.
        lam = damping * 0.001  # tune: target update of ~1mm => deg
        M = J @ J.T + (lam ** 2) * np.eye(3)
        try:
            dq_5 = J.T @ np.linalg.solve(M, err)  # (5,) deg
        except np.linalg.LinAlgError:
            return q, False, err_mm, it
        # Step clip
        step_norm = np.max(np.abs(dq_5))
        if step_norm > step_clip_deg:
            dq_5 = dq_5 * (step_clip_deg / step_norm)
        q[:5] = q[:5] + dq_5
        q = clip_joints(q)

    return q, False, err_mm, max_iter


# ---------------------------------------------------------------
# v6 warm-start: nearest TCP in v6 dataset → use its state as q0.
# ---------------------------------------------------------------
class V6WarmStart:
    def __init__(self, parquet_path="lerobot_dataset_v6/data/chunk-000/file-000.parquet"):
        df = pd.read_parquet(parquet_path)
        self.states = np.stack(df["observation.state"].values).astype(np.float64)  # (N, 6)
        self.tcps = np.array([fk_tcp(s) for s in self.states])  # (N, 3)
        print(f"V6WarmStart: indexed {len(self.states)} frames, TCP z range "
              f"[{self.tcps[:,2].min()*1000:+.1f}, {self.tcps[:,2].max()*1000:+.1f}] mm")

    def query(self, target_xyz, k=1):
        """Returns the k-th nearest v6 state (deg, shape (6,)) by TCP L2."""
        d = np.linalg.norm(self.tcps - np.asarray(target_xyz), axis=1)
        idx = np.argsort(d)[:k]
        return self.states[idx[0]].copy(), d[idx[0]] * 1000.0  # state, dist_mm


if __name__ == "__main__":
    # Self-test: HOME pose
    home = [0.0, 0.0, 90.0, 0.0, 0.0, 30.0]
    p, R = fk_full(home)
    print(f"HOME TCP = ({p[0]*1000:+.1f}, {p[1]*1000:+.1f}, {p[2]*1000:+.1f}) mm")
    J = jacobian_numerical(home)
    print(f"Jacobian @ HOME (m/deg):\n{J}")
    print(f"Singular values: {np.linalg.svd(J, compute_uv=False)}")
