"""D409 design-input: anchor-gate measurements (offline, Isaac 0, physics 0).

Measures, from frozen repo evidence only:
  M1  A64 source identity: d339 cold1 canonical geometry vs d348 attempt2
      instance geometry, per-part vertex-SET comparison (order-insensitive).
  M2  Pure-FK reproduction of the D349 frozen live pose, with two constant
      series: URDF literals (design authority) and pi/2-symbol variant
      (negative-control discriminability probe).
  M3  hppfcl min-distance reproduction of the D349 live-topology values
      (link5 4.272736580324082mm / gripper 11.340262326338637mm) using
      d348 instance geometry under (a) stored live poses, (b) FK-literal
      poses, (c) FK-pi/2 poses.  Old cylinder (0.017, 0.090) at the stored
      object pose is used as CALIBRATION ONLY (query-pipeline check; no
      D362-era physics result is transferred to the real cylinder).

Determinism: canonical payload (sorted keys, repr floats) must be
byte-identical across two independent process runs.
"""
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = "/home/cgxr/Documents/Robotics/RoArm_Project"
URDF = f"{REPO}/local_assets/roarm_m3/urdf/roarm_m3.urdf"
D348 = f"{REPO}/claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_callback_topology_volume_evidence.json"
D339_G = f"{REPO}/claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/d339_gripper_link_cold1_canonical_geometry.json"
D339_L = f"{REPO}/claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/d339_link5_cold1_canonical_geometry.json"
D349 = f"{REPO}/claudedocs/runtime_logs/grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json"

ARM_JOINTS = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
]
GRIPPER_JOINT = "link5_to_gripper_link"
TCP_JOINT = "link5_to_hand_tcp"
ROOT_JOINT = "world_to_base_link"


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rpy(r, p, y):
    return _rz(y) @ _ry(p) @ _rx(r)


def _tf(rot, pos):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = np.asarray(pos, dtype=np.float64)
    return out


def _axis_rot(axis, q):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(q), math.sin(q)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _parse_urdf():
    tree = ET.parse(URDF)
    joints = {}
    for j in tree.getroot().iter("joint"):
        name = j.get("name")
        origin = j.find("origin")
        xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
        axis_el = j.find("axis")
        axis = [float(v) for v in axis_el.get("xyz").split()] if axis_el is not None else [0.0, 0.0, 1.0]
        joints[name] = {"type": j.get("type"), "xyz": xyz, "rpy": rpy, "axis": axis}
    return joints


def _pi2_variant(joints):
    """Reproduce the roarm_kinematics._CHAIN constant series: rpy 1.5708
    literals -> sign-matched pi/2 symbols; link1_to_link2 origin z -> 0.05196."""
    out = {}
    for name, spec in joints.items():
        rpy = [
            math.copysign(math.pi / 2.0, v) if abs(abs(v) - 1.5708) < 1e-9 else v
            for v in spec["rpy"]
        ]
        xyz = list(spec["xyz"])
        if name == "link1_to_link2":
            xyz = [xyz[0], xyz[1], 0.05196]
        out[name] = {"type": spec["type"], "xyz": xyz, "rpy": rpy, "axis": list(spec["axis"])}
    return out


def _fk(joints, q):
    t = _tf(_rpy(*joints[ROOT_JOINT]["rpy"]), joints[ROOT_JOINT]["xyz"])
    for name, qi in zip(ARM_JOINTS, q[:5]):
        spec = joints[name]
        t = t @ _tf(_rpy(*spec["rpy"]), spec["xyz"]) @ _tf(_axis_rot(spec["axis"], qi), [0, 0, 0])
    link5 = t
    g = joints[GRIPPER_JOINT]
    gripper = link5 @ _tf(_rpy(*g["rpy"]), g["xyz"]) @ _tf(_axis_rot(g["axis"], q[5]), [0, 0, 0])
    tc = joints[TCP_JOINT]
    tcp = link5 @ _tf(_rpy(*tc["rpy"]), tc["xyz"])
    return link5, gripper, tcp


def _quat_wxyz_to_rot(quaternion):
    w, x, y, z = [float(v) for v in quaternion]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_angle_deg(r1, r2):
    c = (np.trace(r1.T @ r2) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _pose_delta(tf_mat, pos_stored, quat_stored):
    pos_err_mm = float(np.linalg.norm(tf_mat[:3, 3] - np.asarray(pos_stored)) * 1000.0)
    ang_deg = _rot_angle_deg(tf_mat[:3, :3], _quat_wxyz_to_rot(quat_stored))
    return {"pos_err_mm": repr(pos_err_mm), "rot_err_deg": repr(ang_deg)}


def _min_distance_mm(hppfcl, parts, body_tf, cylinder, cylinder_tf):
    rows = []
    for part in parts:
        model = hppfcl.BVHModelOBBRSS()
        vertices = np.asarray(part["vertices_m"], dtype=np.float64)
        triangles = np.asarray(part["topology_triangles"], dtype=np.int64)
        codes = [
            int(model.beginModel(len(triangles), len(vertices))),
            int(model.addVertices(vertices)),
            int(model.addTriangles(triangles)),
            int(model.endModel()),
        ]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"BVH build failed: {codes}")
        req = hppfcl.DistanceRequest(True, 1.0e-9, 1.0e-9)
        req.gjk_tolerance = 1.0e-9
        req.gjk_max_iterations = 1000
        res = hppfcl.DistanceResult()
        dist_m = float(hppfcl.distance(model, body_tf, cylinder, cylinder_tf, req, res))
        creq = hppfcl.CollisionRequest()
        creq.enable_contact = True
        creq.num_max_contacts = 256
        cres = hppfcl.CollisionResult()
        hppfcl.collide(model, body_tf, cylinder, cylinder_tf, creq, cres)
        contacts = [abs(float(cres.getContact(i).penetration_depth)) for i in range(cres.numContacts())]
        collision = bool(cres.isCollision())
        exact_mm = -max(contacts) * 1000.0 if collision and contacts else dist_m * 1000.0
        rows.append({"name": part["name"], "exact_mm": exact_mm, "collision": collision})
    collisions = [r for r in rows if r["collision"]]
    pool = collisions if collisions else rows
    sel = min(pool, key=lambda r: r["exact_mm"])
    return sel["exact_mm"], sel["name"], len(collisions)


def main():
    import hppfcl

    joints_lit = _parse_urdf()
    joints_pi2 = _pi2_variant(joints_lit)

    d349 = json.loads(open(D349, "rb").read())
    q = [float(v) for v in d349["target_state_guard"]["commanded_joint_rad_float32"]]
    pose = d349["distance_gate"]["authoritative_pose_streams"]["raw_first"]
    per_body = d349["distance_gate"]["per_body"]
    align = d349["frozen_candidate_alignment"]

    d348 = json.loads(open(D348, "rb").read())
    parts_by_body = {}
    for row in d348["rows"]:
        parts_by_body.setdefault(row["body"], []).append(
            {"name": row["name"], "vertices_m": row["instance"]["vertices_m"],
             "topology_triangles": row["instance"]["topology_triangles"],
             "payload_sha256": row["instance"]["payload_sha256"]}
        )
    for body in parts_by_body:
        parts_by_body[body].sort(key=lambda p: p["name"])

    # --- M1: source identity d339 vs d348 (vertex sets, order-insensitive) ---
    m1 = {}
    for body, path in (("gripper_link", D339_G), ("link5", D339_L)):
        d339 = json.loads(open(path, "rb").read())
        inst = {p["name"]: p for p in parts_by_body[body]}
        diffs = []
        set_equal = 0
        for p in d339["parts"]:
            s339 = set(map(tuple, p["vertices_m"]))
            s348 = set(map(tuple, inst[p["name"]]["vertices_m"]))
            if s339 == s348:
                set_equal += 1
            else:
                extra_339 = sorted(s339 - s348)
                missing_348 = sorted(s348 - s339)
                h339 = max(min(math.dist(a, b) for b in s348) for a in s339) * 1000.0
                h348 = max(min(math.dist(a, b) for b in s339) for a in s348) * 1000.0
                diffs.append({
                    "part": p["name"],
                    "v339": len(s339), "v348": len(s348),
                    "extra_in_d339": len(extra_339), "missing_from_d339": len(missing_348),
                    "max_excess_d339_vertex_to_d348_set_mm": repr(h339),
                    "max_d348_vertex_to_d339_set_mm": repr(h348),
                })
        m1[body] = {"parts": len(d339["parts"]), "vertex_set_equal": set_equal, "differing": diffs}

    # --- M2: FK pose reproduction, both constant series ---
    m2 = {}
    for label, joints in (("urdf_literal", joints_lit), ("pi2_symbol_variant", joints_pi2)):
        link5, gripper, tcp = _fk(joints, q)
        m2[label] = {
            "link5_vs_stored": _pose_delta(link5, pose["body_poses_w"]["link5"]["pos_m"],
                                           pose["body_poses_w"]["link5"]["quat_wxyz"]),
            "gripper_vs_stored": _pose_delta(gripper, pose["body_poses_w"]["gripper_link"]["pos_m"],
                                             pose["body_poses_w"]["gripper_link"]["quat_wxyz"]),
            "tcp_vs_actual_mm": repr(float(np.linalg.norm(
                tcp[:3, 3] - np.asarray([align["actual_tcp_x_m"], align["actual_tcp_y_m"], align["actual_tcp_z_m"]])) * 1000.0)),
            "tcp_vs_commanded_mm": repr(float(np.linalg.norm(
                tcp[:3, 3] - np.asarray([align["commanded_tcp_x_m"], align["commanded_tcp_y_m"], align["commanded_tcp_z_m"]])) * 1000.0)),
        }

    # --- M3: hppfcl min-distance reproduction (calibration on OLD cylinder) ---
    cylinder = hppfcl.Cylinder(0.017, 0.090)
    cylinder_tf = hppfcl.Transform3f(
        _quat_wxyz_to_rot(pose["object_quat_wxyz"]), np.asarray(pose["object_pos_w_m"], dtype=np.float64))
    stored_tf = {
        body: hppfcl.Transform3f(
            _quat_wxyz_to_rot(pose["body_poses_w"][body]["quat_wxyz"]),
            np.asarray(pose["body_poses_w"][body]["pos_m"], dtype=np.float64))
        for body in ("link5", "gripper_link")
    }
    fk_lit = _fk(joints_lit, q)
    fk_pi2 = _fk(joints_pi2, q)
    fk_tf = {
        "urdf_literal": {"link5": fk_lit[0], "gripper_link": fk_lit[1]},
        "pi2_symbol_variant": {"link5": fk_pi2[0], "gripper_link": fk_pi2[1]},
    }
    m3 = {}
    for mode in ("stored_pose", "urdf_literal", "pi2_symbol_variant"):
        m3[mode] = {}
        for body in ("link5", "gripper_link"):
            if mode == "stored_pose":
                tfb = stored_tf[body]
            else:
                mat = fk_tf[mode][body]
                tfb = hppfcl.Transform3f(mat[:3, :3], mat[:3, 3])
            value, sel, ncol = _min_distance_mm(hppfcl, parts_by_body[body], tfb, cylinder, cylinder_tf)
            ref = float(per_body[body]["live_topology_exact_signed_distance_mm"])
            m3[mode][body] = {
                "min_exact_signed_distance_mm": repr(value),
                "selected_part": sel,
                "collision_parts": ncol,
                "d349_live_topology_reference_mm": repr(ref),
                "abs_delta_mm": repr(abs(value - ref)),
            }

    payload = {
        "artifact": "d409_anchor_gate_measurement",
        "semantics": (
            "offline anchor-gate measurement for D409 static prep; old cylinder (0.017,0.090) at the "
            "stored D349 object pose is calibration-only for the query pipeline; no D362-era physics "
            "result is transferred to the real D29xH50 cylinder (D379)."
        ),
        "inputs_sha256": {
            "urdf": _sha(URDF), "d348_evidence": _sha(D348), "d349_measurement": _sha(D349),
            "d339_gripper_canonical": _sha(D339_G), "d339_link5_canonical": _sha(D339_L),
        },
        "frozen_joint_rad_float32": [repr(v) for v in q],
        "m1_a64_source_identity": m1,
        "m2_fk_pose_reproduction": m2,
        "m3_hppfcl_distance_reproduction": m3,
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "isaac_executed": 0,
        "physics_steps": 0,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
