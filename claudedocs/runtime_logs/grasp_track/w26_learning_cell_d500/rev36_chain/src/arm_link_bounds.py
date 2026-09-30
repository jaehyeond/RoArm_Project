"""팔 링크 **실제 외곽**의 보수적 local bounds — URDF/STL 정적 geometry 에서 유도. 순수 CPU.

왜 필요한가
----------
`renderer_readiness_07_actual` 육안 검수(root `ROOT_READINESS_INSPECTION_02.json`)에서
**frame3 측면 상단 · frames3/4/5 상면 하단의 팔 링크가 잘렸다**. 그때 카메라 경계 집합은
어깨→립 **직선 구간 5점 샘플**뿐이어서 실제 링크를 포함하지 않았다(내 `claim_scope` 가 이미
그 한계를 밝혔고, 그래서 `fits=true` 로 그 관측을 반박할 수 없다).

root `msg_a0752618dd6c` ②: **관절 6원점만으로 mesh 외곽이라 주장하지 말 것**,
**임의 반경 상수를 더하지 말 것**, 기존 URDF/메시/설치 USD 의 **정적 geometry 에서 보수적
local bounds 를 유도**하고 **출처를 기록**한 뒤 FK 로 변환할 것.

무엇을 하나
----------
설치된 URDF 를 파싱해 링크별 **visual mesh** 를 찾고, 그 STL 정점에서 **local AABB** 를 구한다.
URDF 의 `scale` 과 `<origin xyz rpy>` 를 실제로 적용한다. 반환값에는 경로·SHA256·삼각형 수·
AABB 가 함께 들어가 **출처가 영수증에 남는다**. 상수를 지어내지 않는다.

보수성
-----
AABB 는 볼록 껍질보다 **크거나 같다** → 프레이밍 입력으로서 보수적이다.
다만 AABB 는 mesh 자체가 아니므로 "실제 mesh 외곽을 정확히 담았다"가 아니라
**"mesh 를 포함하는 축정렬 상자를 담았다"** 로만 주장한다.

하지 않는 것
-----------
물리·충돌·경로·관절한계에 관여하지 않는다. 표시 프레이밍 입력 전용이다.
GPU·Isaac·USD 런타임을 쓰지 않는다(정적 파일 읽기만).
"""
import hashlib
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# 설치된 정적 자산. 렌더러가 쓰는 USD 와 같은 s1_v1 계열이다.
DEFAULT_URDF = Path("/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/"
                    "urdf/roarm_m3_s1_v1.urdf")
# FK CHAIN 프레임 ↔ URDF 링크. CHAIN 항목을 적용한 뒤의 누적 변환이 그 링크의 프레임이다.
# (URDF joint origin 이 `w13_fk.CHAIN` 수치와 일치함을 `verify_chain_matches_urdf` 가 확인한다.)
CHAIN_FRAME_TO_LINK = {
    "world_to_base": "base_link",
    "base_to_link1": "link1",
    "link1_to_link2": "link2",
    "link2_to_link3": "link3",
    "link3_to_link4": "link4",
    "link4_to_link5": "link5",
}


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rpy(rpy):
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def stl_vertex_bounds(path):
    """binary STL 정점의 min/max 와 삼각형 수. ASCII STL 이면 명시적으로 거절한다."""
    b = Path(path).read_bytes()
    if len(b) < 84:
        raise ValueError(f"STL 이 너무 짧다: {path}")
    if b[:5] == b"solid" and b"facet normal" in b[:4096]:
        raise ValueError(f"ASCII STL 은 이 경로에서 지원하지 않는다(정확한 파서를 쓸 것): {path}")
    n = struct.unpack("<I", b[80:84])[0]
    need = 84 + n * 50
    if len(b) < need:
        raise ValueError(f"STL 길이 불일치: {path} 기대 {need} 실제 {len(b)}")
    a = np.frombuffer(b, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
    tri = a[:, 12:48].copy().view(np.float32).reshape(n, 3, 3).astype(np.float64)
    v = tri.reshape(-1, 3)
    return v.min(0), v.max(0), int(n)


def load_link_local_bounds(urdf_path=DEFAULT_URDF, links=None):
    """링크별 **local AABB**(URDF scale·origin 적용) + 출처. GPU 접촉 0.

    반환 {link: {"lo_m","hi_m","corners_m"(8x3),"mesh","mesh_sha256","n_tris","scale","origin_xyz",
                 "origin_rpy"}}
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 가 없다(정적 자산 읽기 실패): {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    want = set(links) if links else set(CHAIN_FRAME_TO_LINK.values())
    out = {}
    for ln in root.findall("link"):
        name = ln.get("name")
        if name not in want:
            continue
        vis = ln.find("visual")
        mesh = ln.find("visual/geometry/mesh")
        if mesh is None:
            raise ValueError(f"링크 {name} 에 visual mesh 가 없다 — 상수로 대체하지 않는다")
        f = (urdf_path.parent / mesh.get("filename")).resolve()
        if not f.exists():
            f = (urdf_path.parent / "meshes" / Path(mesh.get("filename")).name).resolve()
        if not f.exists():
            raise FileNotFoundError(f"링크 {name} 의 mesh 파일을 찾지 못했다: {mesh.get('filename')}")
        lo, hi, ntri = stl_vertex_bounds(f)
        sc = np.array([float(x) for x in (mesh.get("scale") or "1 1 1").split()], float)
        o = vis.find("origin")
        oxyz = np.array([float(x) for x in (o.get("xyz") if o is not None and o.get("xyz")
                                            else "0 0 0").split()], float)
        orpy = [float(x) for x in (o.get("rpy") if o is not None and o.get("rpy")
                                   else "0 0 0").split()]
        # AABB 8 모서리를 만든 뒤 scale → visual origin(R,t) 을 적용하고 다시 AABB 를 취한다.
        c = np.array([[lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1],
                       lo[2] if i & 4 else hi[2]] for i in range(8)], float) * sc
        c = (_rpy(orpy) @ c.T).T + oxyz
        out[name] = {"lo_m": c.min(0).tolist(), "hi_m": c.max(0).tolist(),
                     "corners_m": c.tolist(), "mesh": str(f), "mesh_sha256": _sha(f),
                     "n_tris": ntri, "scale": sc.tolist(),
                     "origin_xyz": oxyz.tolist(), "origin_rpy": orpy}
    missing = sorted(want - set(out))
    if missing:
        raise ValueError(f"URDF 에서 찾지 못한 링크: {missing} — 임의 상수로 대체하지 않는다")
    return out


def verify_chain_matches_urdf(chain, urdf_path=DEFAULT_URDF, tol=1e-5):
    """`w13_fk.CHAIN` 의 joint origin 이 URDF 와 같은지 확인한다(프레임↔링크 매핑의 근거)."""
    root = ET.parse(Path(urdf_path)).getroot()
    ju = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        ju[j.get("name")] = np.array([float(x) for x in ((o.get("xyz") if o is not None
                                                          else None) or "0 0 0").split()], float)
    alias = {"world_to_base": "world_to_base_link", "base_to_link1": "base_link_to_link1",
             "link1_to_link2": "link1_to_link2", "link2_to_link3": "link2_to_link3",
             "link3_to_link4": "link3_to_link4", "link4_to_link5": "link4_to_link5",
             "link5_to_tcp": "link5_to_hand_tcp"}
    rows, ok = [], True
    for nm, xyz, _rp, _qi in chain:
        u = ju.get(alias.get(nm, nm))
        d = float(np.abs(np.asarray(xyz, float) - u).max()) if u is not None else float("inf")
        ok = ok and d <= tol
        rows.append({"chain": nm, "urdf_joint": alias.get(nm, nm),
                     "chain_xyz": list(map(float, xyz)),
                     "urdf_xyz": (u.tolist() if u is not None else None),
                     "max_abs_diff_m": d})
    return {"all_match": bool(ok), "tolerance_m": tol, "rows": rows,
            "urdf": str(urdf_path), "urdf_sha256": _sha(urdf_path)}


def urdf_display_chain(urdf_path=DEFAULT_URDF):
    """표시 경계 전용 **exact XML** 관절 체인. 물리/FK 경로는 건드리지 않는다.

    감사 `msg_8a86483ae5bb` / root `msg_f1bc52ebbed4`: `w13_fk.CHAIN` 은 반올림 상수
    (`0.05196`, `PI/2`)를 쓰는데 원 URDF 는 `0.051959`, `1.5708` 이다
    (각각 1e-6 m · 3.673205e-06 rad 차이). **표시 경계용으로는 exact XML 이 더 적합**하다.
    여기서 만드는 것은 `T_parent * T_origin * R_axis(q)` 뿐이며 FK/물리/관절한계에 관여하지 않는다.

    반환 [(joint_name, child_link, xyz(3,), rpy(3,), axis(3,)|None)] — base→link5 순서.
    """
    root = ET.parse(Path(urdf_path)).getroot()
    by_parent = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        ax = j.find("axis")
        by_parent.setdefault(j.find("parent").get("link"), []).append({
            "name": j.get("name"), "child": j.find("child").get("link"),
            "xyz": np.array([float(x) for x in ((o.get("xyz") if o is not None else None)
                                                or "0 0 0").split()], float),
            "rpy": np.array([float(x) for x in ((o.get("rpy") if o is not None else None)
                                                or "0 0 0").split()], float),
            "axis": (np.array([float(x) for x in ax.get("xyz").split()], float)
                     if (ax is not None and j.get("type") in ("revolute", "continuous")) else None)})
    order, cur = [], "world"
    wanted = ["base_link", "link1", "link2", "link3", "link4", "link5"]
    for target in wanted:
        nxt = None
        for j in by_parent.get(cur, []):
            if j["child"] == target:
                nxt = j
                break
        if nxt is None:
            raise ValueError(f"URDF 에서 {cur} -> {target} 관절을 찾지 못했다")
        order.append((nxt["name"], nxt["child"], nxt["xyz"], nxt["rpy"], nxt["axis"]))
        cur = target
    return order


def _axis_R(axis, q_rad):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c, s = np.cos(q_rad), np.sin(q_rad)
    K_ = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + s * K_ + (1.0 - c) * (K_ @ K_)


def link_world_corners_exact_urdf(q5_deg, link_bounds, shoulder_above_plate,
                                  display_chain=None, urdf_path=DEFAULT_URDF):
    """**exact URDF XML** 로 각 링크 local AABB 8 모서리를 로봇 프레임으로 옮긴다(표시 전용).

    `T = T_parent @ T_origin(xyz,rpy) @ R_axis(q)` 를 URDF 값 그대로 쓴다.
    관절각은 `q5_deg` 순서(base, shoulder, elbow, wrist_p, wrist_r)를 회전 관절에 차례로 준다.
    """
    ch = display_chain if display_chain is not None else urdf_display_chain(urdf_path)
    q = list(np.radians(list(q5_deg)))
    T = np.eye(4)
    pts, per, qi = [], {}, 0
    for _nm, child, xyz, rpy, axis in ch:
        To = np.eye(4)
        To[:3, :3] = _rpy(rpy)
        To[:3, 3] = xyz
        T = T @ To
        if axis is not None:
            Ra = np.eye(4)
            Ra[:3, :3] = _axis_R(axis, q[qi] if qi < len(q) else 0.0)
            qi += 1
            T = T @ Ra
        if child in link_bounds:
            c = np.asarray(link_bounds[child]["corners_m"], float)
            w = (T[:3, :3] @ c.T).T + T[:3, 3]
            w[:, 2] -= shoulder_above_plate
            per[child] = w.tolist()
            pts.append(w)
    return (np.concatenate(pts, 0) if pts else np.zeros((0, 3))), per


def display_asset_hashes(urdf_path=DEFAULT_URDF, link_bounds=None):
    """표시 경계에 **실제로 쓴** 외부 파일의 full SHA256. candidate/런처 manifest 에 pin 한다."""
    lb = link_bounds if link_bounds is not None else load_link_local_bounds(urdf_path)
    out = {str(Path(urdf_path)): _sha(urdf_path)}
    for v in lb.values():
        out[v["mesh"]] = v["mesh_sha256"]
    return out


def link_world_corners(chain, q5_deg, link_bounds, t_fn, tz_fn, shoulder_above_plate):
    """FK 로 각 링크 local AABB 8 모서리를 **로봇 프레임**으로 옮긴다.

    `t_fn`/`tz_fn` 은 `w13_fk._T`/`w13_fk._Tz` 를 그대로 받는다(별도 FK 재구현 금지).
    반환 (pts(N,3), per_link{link: 8x3}).
    """
    q = np.radians(list(q5_deg) + [0.0])
    T = np.eye(4)
    pts, per = [], {}
    for nm, xyz, rpy, qi in chain:
        T = T @ t_fn(xyz, rpy)
        if qi is not None:
            T = T @ tz_fn(q[qi])
        link = CHAIN_FRAME_TO_LINK.get(nm)
        if link and link in link_bounds:
            c = np.asarray(link_bounds[link]["corners_m"], float)
            w = (T[:3, :3] @ c.T).T + T[:3, 3]
            w[:, 2] -= shoulder_above_plate          # 로봇 프레임 = 어깨축 원점
            per[link] = w.tolist()
            pts.append(w)
        if nm == "link4_to_link5":
            break
    return (np.concatenate(pts, 0) if pts else np.zeros((0, 3))), per


PROVENANCE_NOTE = (
    "링크 외곽은 설치된 URDF 의 visual mesh STL 정점에서 유도한 **local AABB** 이고, "
    "표시 경계 변환은 **exact URDF XML** 의 joint xyz/rpy/axis 로 만든 T_parent*T_origin*R_axis(q) 다"
    "(FK.CHAIN 의 반올림 상수를 쓰지 않는다. 물리/FK 경로는 변경하지 않는다). "
    "임의 반경 상수를 더하지 않았다. AABB 는 mesh 를 포함하므로 프레이밍 입력으로 보수적이지만 "
    "mesh 자체는 아니다 — '축정렬 상자를 담았다'로만 주장하고 '실제 mesh 외곽을 정확히 담았다'로 "
    "주장하지 않는다. GPU·Isaac·USD 런타임 미사용(정적 파일 읽기만).")
