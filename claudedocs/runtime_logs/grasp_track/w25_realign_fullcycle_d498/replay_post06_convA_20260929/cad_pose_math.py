"""S1 v1 CAD 그리퍼(link5 mm STL) → DEME 세계(m) 배치 수학. 순수 numpy (Isaac·pxr·DEME 0).

좌표 기준 (전부 원자료에서 읽는다 — 하드코딩 0)
    link5 프레임 mm : S1 v1 STL(`fixed_ALL.stl`, `door_ALL.stl`)과 충돌 셸 `half_bowl()` 의 원래 좌표.
    owner 프레임    : DEME 메시 owner. 굽기 규약(`sim_deme_scoop_s1.py:170-193 load_tool`)
                      고정  local_F = R_W @ (v_l5 − L5) / 1000          (L5 = 두꺼워진 립, 결과 JSON
                            frames.lip_collision_owner_l5_mm = (8.1, 0, 169.6))
                      문    local_D = R_W @ (roty(q_open) @ (v_l5 − H5)) / 1000
                            (H5 = params.hinge_l5_mm = (0, 18.821, 52.035), q_open =
                            trajectory.q_open_joint_deg = 27.5 — 문 셸은 q_open 자세로 구워졌다)
                      R_W  = frames.R_W_frozen_columns_are_link5_axes (link5 → 세계, 열 = link5 축)
    DEME 세계 m     : 엔진 보고 노드 = p_owner + R(q_owner) @ local  (`GetMeshNodesGlobal`).
                      p/q = NPZ tool_pos_m/tool_quat_xyzw(고정 owner), door_pos_m/door_quat_xyzw(문 owner).
    표시 좌표       : 재생 스크립트의 `deme_to_disp`(원점 평행이동만)가 담당 — 여기서는 다루지 않는다.

따라서 CAD 정점도 **같은 식**으로 놓는다(STL 은 link5 mm 이므로 충돌 셸과 같은 입력 프레임):
    world_F = tool_pos + R(tool_quat) @ R_W @ (v_fixed_l5 − L5) / 1000
    world_D = door_pos + R(door_quat) @ R_W @ roty(q_open) @ (v_door_l5_closed − H5) / 1000
문 CAD 는 `door_ALL.stl`(link5 mm, 닫힘 q=0) 을 쓴다. `door_ALL_jawframe.stl`(gripper_link mm) 은
v_l5 = R_G2L5 @ v_g + H5 로 같은 형상임을 `cad_pose_check.py` 가 확인한다.
"""
import math

import numpy as np


def quat_xyzw_to_mat(q):
    """(…,4) xyzw → (…,3,3). 정규화한다(저장값은 float32)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), float)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def roty(deg):
    """link5 +Y 축 회전(`sim_deme_scoop_s1.py:110-113` 와 같은 식). +q = 문 열림."""
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def owner_local_fixed(v_l5_mm, lip_owner_l5_mm, R_W):
    """link5 mm → 고정 owner 로컬 m."""
    return (np.asarray(R_W, float) @ (np.asarray(v_l5_mm, float) - np.asarray(lip_owner_l5_mm, float)).T).T / 1000.0


def owner_local_door(v_l5_closed_mm, hinge_l5_mm, q_open_deg, R_W):
    """link5 mm(문 닫힘 q=0) → 문 owner 로컬 m (q_open 으로 구운 규약)."""
    d = np.asarray(v_l5_closed_mm, float) - np.asarray(hinge_l5_mm, float)
    return (np.asarray(R_W, float) @ roty(q_open_deg) @ d.T).T / 1000.0


def place(local_m, pos_m, quat_xyzw):
    """owner 로컬 (N,3) → 세계. pos (3,) 또는 (S,3), quat (4,) 또는 (S,4). 반환 (N,3) 또는 (S,N,3)."""
    R = quat_xyzw_to_mat(quat_xyzw)
    p = np.asarray(pos_m, float)
    L = np.asarray(local_m, float)
    if R.ndim == 2:
        return L @ R.T + p
    return np.einsum("sij,nj->sni", R, L) + p[:, None, :]


def frame_params(res):
    """결과 JSON(w13_cycle_seed*.json) 에서 배치 상수를 읽는다(하드코딩 금지)."""
    fr, P, tr = res["frames"], res["params"], res["trajectory"]
    return {"R_W": np.asarray(fr["R_W_frozen_columns_are_link5_axes"], float),
            "lip_owner_l5_mm": np.asarray(fr["lip_collision_owner_l5_mm"], float),
            "hinge_l5_mm": np.asarray(P["hinge_l5_mm"], float),
            "q_open_deg": float(tr["q_open_joint_deg"])}


def cad_world(fixed_l5_mm, door_l5_mm, fp, tool_pos, tool_quat, door_pos, door_quat):
    """S1 v1 CAD 정점을 세계(DEME m)로. fp = frame_params(res)."""
    lf = owner_local_fixed(fixed_l5_mm, fp["lip_owner_l5_mm"], fp["R_W"])
    ld = owner_local_door(door_l5_mm, fp["hinge_l5_mm"], fp["q_open_deg"], fp["R_W"])
    return place(lf, tool_pos, tool_quat), place(ld, door_pos, door_quat)


def aabb_corners(pts):
    """(N,3) → 로컬 AABB 8 모서리 (8,3). 강체 변환 뒤에도 원 점들을 포함한다(보수적 프레이밍용)."""
    p = np.asarray(pts, float)
    lo, hi = p.min(0), p.max(0)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def load_s1_cad(cad_dir):
    """S1 v1 CAD(link5 mm) 두 부품을 읽는다. 위상 그대로(process=False) · sha256 기록."""
    import hashlib
    from pathlib import Path
    import trimesh
    out = {}
    for key, nm in (("fixed", "fixed_ALL.stl"), ("door", "door_ALL.stl")):
        p = Path(cad_dir) / nm
        m = trimesh.load(p, process=False)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        out[key] = {"path": str(p), "sha256": h, "vertices_l5_mm": np.asarray(m.vertices, float),
                    "faces": np.asarray(m.faces, int)}
    return out
