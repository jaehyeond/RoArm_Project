"""D409 static-prep measurements S1/S2/S3 (offline, Isaac 0, physics 0).

Obligations from design-confirmation v2 (session doc section 4):
  S1  d348 internal integrity (P1): per-part D409-canonical geometry hash
      over (name, vertices_m, topology_triangles) for all 128 parts
      (64 link5 + 64 gripper_link), pinned as the runtime re-verification
      reference. Stored payload_sha256/witness_sha256 recorded verbatim
      for lineage; uniqueness and 64+64 counts checked. The d348 file-level
      sha256 is pinned separately in prereg input_hashes.
  S2  D368 mask <-> d348 binding (P1): every mask part name (link5_fixed 4,
      gripper_inner 17, gripper_outer 16) must exist in the matching d348
      body row set; outer == inner - {part_035}; part_035/part_048 present
      in gripper rows; per-part vertex counts recorded.
  S3  Query-throughput benchmark persistence (W-OPS4): BVH build of all
      128 parts + per-part distance queries against the real analytic
      cylinder hppfcl.Cylinder(0.0145, 0.050) at the confirmed placement,
      body poses = URDF-literal FK at the D349 frozen joints. Distances are
      deterministic and enter the canonical payload; wall-clock timings are
      NON-deterministic and are written to --timing-out (excluded from the
      bit-exact determinism check by design).

stdout = canonical deterministic payload JSON only (sort_keys, repr floats).
"""
import argparse
import hashlib
import json
import math
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

REPO = "/home/cgxr/Documents/Robotics/RoArm_Project"
URDF = f"{REPO}/local_assets/roarm_m3/urdf/roarm_m3.urdf"
D348 = f"{REPO}/claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_callback_topology_volume_evidence.json"
D368 = f"{REPO}/claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_evidence.json"
D349 = f"{REPO}/claudedocs/runtime_logs/grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json"

LINK5_FIXED = ["part_027", "part_029", "part_030", "part_031"]

# Confirmed design v2 placement (section 4.2 W-SCI3 operation-sequence pin).
CYL_RADIUS = 0.0145
CYL_HEIGHT = 0.050
CYL_X = 0.30000001192092896
TABLE_Z = 0.03288299962878227 - 0.045
Z_CENTER = TABLE_Z + 0.025

ARM_JOINTS = [
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
]
GRIPPER_JOINT = "link5_to_gripper_link"
ROOT_JOINT = "world_to_base_link"

BENCH_ROUNDS = 100  # 128 parts x 100 rounds = 12,800 timed queries


def _sha_file(path):
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
        joints[name] = {"xyz": xyz, "rpy": rpy, "axis": axis}
    return joints


def _fk(joints, q):
    t = _tf(_rpy(*joints[ROOT_JOINT]["rpy"]), joints[ROOT_JOINT]["xyz"])
    for name, qi in zip(ARM_JOINTS, q[:5]):
        spec = joints[name]
        t = t @ _tf(_rpy(*spec["rpy"]), spec["xyz"]) @ _tf(_axis_rot(spec["axis"], qi), [0, 0, 0])
    link5 = t
    g = joints[GRIPPER_JOINT]
    gripper = link5 @ _tf(_rpy(*g["rpy"]), g["xyz"]) @ _tf(_axis_rot(g["axis"], q[5]), [0, 0, 0])
    return link5, gripper


def _canonical_part_hash(name, vertices_m, topology_triangles):
    blob = json.dumps(
        {"name": name,
         "vertices_m": [[repr(float(c)) for c in v] for v in vertices_m],
         "topology_triangles": [[int(i) for i in t] for t in topology_triangles]},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-out", required=True)
    args = parser.parse_args()

    import hppfcl

    d348 = json.loads(open(D348, "rb").read())
    d368 = json.loads(open(D368, "rb").read())
    d349 = json.loads(open(D349, "rb").read())

    rows_by_body = {"link5": {}, "gripper_link": {}}
    for row in d348["rows"]:
        rows_by_body[row["body"]][row["name"]] = row

    # --- S1: per-part D409-canonical geometry hash pin -----------------
    s1 = {}
    for body in ("link5", "gripper_link"):
        parts = rows_by_body[body]
        canon = {}
        stored_payload = {}
        stored_witness = {}
        for name in sorted(parts):
            inst = parts[name]["instance"]
            canon[name] = _canonical_part_hash(name, inst["vertices_m"], inst["topology_triangles"])
            stored_payload[name] = inst["payload_sha256"]
            stored_witness[name] = inst["witness_sha256"]
        s1[body] = {
            "part_count": len(parts),
            "d409_canonical_geometry_sha256": canon,
            "stored_payload_sha256": stored_payload,
            "stored_witness_sha256": stored_witness,
            "canonical_hash_unique": len(set(canon.values())) == len(canon),
        }
    s1["counts_pass"] = s1["link5"]["part_count"] == 64 and s1["gripper_link"]["part_count"] == 64

    # --- S2: D368 mask <-> d348 binding ---------------------------------
    def _mask_names(alloc_key):
        node = d368["patch_allocation"][alloc_key]

        def find(o):
            if isinstance(o, dict):
                if "part_names" in o:
                    return o["part_names"]
                for v in o.values():
                    r = find(v)
                    if r:
                        return r
            return None

        return find(node)

    inner = _mask_names("gripper_inner")
    outer = _mask_names("gripper_outer")
    fixed = _mask_names("link5_fixed")
    s2 = {
        "link5_fixed": fixed,
        "gripper_inner_count": len(inner),
        "gripper_outer_count": len(outer),
        "link5_fixed_equals_design": fixed == LINK5_FIXED,
        "fixed_all_in_d348_link5": all(n in rows_by_body["link5"] for n in fixed),
        "inner_all_in_d348_gripper": all(n in rows_by_body["gripper_link"] for n in inner),
        "outer_all_in_d348_gripper": all(n in rows_by_body["gripper_link"] for n in outer),
        "outer_equals_inner_minus_part035": sorted(outer) == sorted(set(inner) - {"part_035"}),
        "part_035_in_gripper_rows": "part_035" in rows_by_body["gripper_link"],
        "part_048_in_gripper_rows": "part_048" in rows_by_body["gripper_link"],
        "mask_part_vertex_counts": {
            n: rows_by_body["gripper_link"][n]["instance"]["vertex_count"] for n in sorted(inner)
        } | {n: rows_by_body["link5"][n]["instance"]["vertex_count"] for n in fixed},
    }
    s2["pass"] = all(
        s2[k] for k in (
            "link5_fixed_equals_design", "fixed_all_in_d348_link5",
            "inner_all_in_d348_gripper", "outer_all_in_d348_gripper",
            "outer_equals_inner_minus_part035",
            "part_035_in_gripper_rows", "part_048_in_gripper_rows",
        )
    ) and s2["gripper_inner_count"] == 17 and s2["gripper_outer_count"] == 16

    # --- S3: throughput benchmark (deterministic distances + timing) ----
    joints = _parse_urdf()
    q = [float(v) for v in d349["target_state_guard"]["commanded_joint_rad_float32"]]
    link5_tf_mat, gripper_tf_mat = _fk(joints, q)
    body_tf = {
        "link5": hppfcl.Transform3f(link5_tf_mat[:3, :3], link5_tf_mat[:3, 3]),
        "gripper_link": hppfcl.Transform3f(gripper_tf_mat[:3, :3], gripper_tf_mat[:3, 3]),
    }
    cylinder = hppfcl.Cylinder(CYL_RADIUS, CYL_HEIGHT)
    cyl_tf = hppfcl.Transform3f(np.eye(3), np.array([CYL_X, 0.0, Z_CENTER], dtype=np.float64))

    build_t0 = time.perf_counter()
    models = []
    for body in ("link5", "gripper_link"):
        for name in sorted(rows_by_body[body]):
            inst = rows_by_body[body][name]["instance"]
            model = hppfcl.BVHModelOBBRSS()
            vertices = np.asarray(inst["vertices_m"], dtype=np.float64)
            triangles = np.asarray(inst["topology_triangles"], dtype=np.int64)
            codes = [
                int(model.beginModel(len(triangles), len(vertices))),
                int(model.addVertices(vertices)),
                int(model.addTriangles(triangles)),
                int(model.endModel()),
            ]
            if any(code != 0 for code in codes):
                raise RuntimeError(f"BVH build failed for {body}/{name}: {codes}")
            models.append((body, name, model))
    build_elapsed = time.perf_counter() - build_t0

    distances = {}
    query_t0 = time.perf_counter()
    n_queries = 0
    for _ in range(BENCH_ROUNDS):
        for body, name, model in models:
            req = hppfcl.DistanceRequest(True, 1.0e-9, 1.0e-9)
            req.gjk_tolerance = 1.0e-9
            req.gjk_max_iterations = 1000
            res = hppfcl.DistanceResult()
            dist = float(hppfcl.distance(model, body_tf[body], cylinder, cyl_tf, req, res))
            distances[f"{body}/{name}"] = repr(dist)
            n_queries += 1
    query_elapsed = time.perf_counter() - query_t0
    us_per_query = query_elapsed / n_queries * 1e6

    dist_blob = json.dumps(distances, sort_keys=True, separators=(",", ":")).encode()
    s3 = {
        "config": {
            "cylinder": ["hppfcl.Cylinder", repr(CYL_RADIUS), repr(CYL_HEIGHT)],
            "cylinder_pos": [repr(CYL_X), repr(0.0), repr(Z_CENTER)],
            "table_z": repr(TABLE_Z),
            "body_pose_source": "URDF-literal FK at D349 frozen float32 joints",
            "bvh": "BVHModelOBBRSS(topology_triangles)",
            "distance_request": ["DistanceRequest(True,1e-9,1e-9)", "gjk 1e-9/1000"],
            "rounds": BENCH_ROUNDS,
            "parts": 128,
        },
        "n_queries": n_queries,
        "per_part_distance_sha256": hashlib.sha256(dist_blob).hexdigest(),
        "min_distance_link5_mm": repr(
            min(float(v) for k, v in distances.items() if k.startswith("link5/")) * 1000.0
        ),
        "min_distance_gripper_mm": repr(
            min(float(v) for k, v in distances.items() if k.startswith("gripper_link/")) * 1000.0
        ),
    }

    timing = {
        "bvh_build_128_parts_s": build_elapsed,
        "query_elapsed_s": query_elapsed,
        "n_queries": n_queries,
        "us_per_query": us_per_query,
        "extrapolation_s": {
            "2.7M": us_per_query * 2.7e6 / 1e6,
            "4.5M": us_per_query * 4.5e6 / 1e6,
        },
        "registered_budget_check_4p5M_lt_7200s": us_per_query * 4.5e6 / 1e6 < 7200.0,
    }
    with open(args.timing_out, "w") as f:
        json.dump(timing, f, indent=1)

    payload = {
        "artifact": "d409_static_prep_s1s2s3",
        "semantics": (
            "static-prep measurements for confirmed design v2 (session doc section 4): "
            "S1 d348 per-part D409-canonical geometry hash pin, S2 D368 mask binding to d348 rows, "
            "S3 deterministic distance set for the throughput benchmark (timings in --timing-out, "
            "excluded from determinism by design). Real cylinder analytic primitive only (D379); "
            "no Isaac, no physics step, no frozen-file writes."
        ),
        "inputs_sha256": {"d348": _sha_file(D348), "d368": _sha_file(D368),
                          "d349": _sha_file(D349), "urdf": _sha_file(URDF)},
        "s1_d348_integrity": s1,
        "s2_mask_binding": s2,
        "s3_benchmark_deterministic": s3,
        "isaac_executed": 0,
        "physics_steps": 0,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
