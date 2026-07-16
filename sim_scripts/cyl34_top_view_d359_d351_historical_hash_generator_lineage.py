#!/usr/bin/env python3
"""D359 read-only recovery of the D351 historical patch-hash generator lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil


REPO = Path(__file__).resolve().parents[1]
CASE = "g0a_d359"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d359"
PREREG_PATH = OUT_DIR / "d359_preregistration.json"
INVOCATION_PATH = OUT_DIR / "d359_invocation.json"
PHASE_PATH = OUT_DIR / "d359_phase_markers.jsonl"
EVIDENCE_PATH = OUT_DIR / "d359_historical_hash_provenance_evidence.json"
REPORT_PATH = OUT_DIR / "d359_historical_hash_provenance_report.md"
COMPLETION_PATH = OUT_DIR / "d359_completion_summary.json"

HARNESS = Path(__file__).resolve()
SESSION_DOC = REPO / "claudedocs/session_20260716_grasp_g0a_d359_d351_historical_hash_generator_lineage.md"
START_HERE = REPO / "START_HERE.md"
D351_HARNESS = REPO / "sim_scripts/cyl34_top_view_d351_zero_step_closure_geometry.py"
D358_HARNESS = REPO / "sim_scripts/cyl34_top_view_d358_moving_jaw_patch_hash_provenance_retry.py"
D358_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d358/d358_patch_hash_provenance_evidence.json"
D358_COMPLETION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d358/d358_completion_summary.json"
D344_STAGE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/"
    "roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd"
)
LOCAL_STAGE = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
TRANSCRIPT = Path(
    "/home/cgxr/.codex/sessions/2026/07/15/"
    "rollout-2026-07-15T15-08-12-019f6463-d763-7761-bb62-68bfe4f993f2.jsonl"
)

EXPECTED_HEAD = "d4671d4bdefa4f6e5ef1f2f28b8e318c100b7cb5"
EXPECTED_FIRST_COMMIT = "c2cfa5f41d4c15fec15330cfad38b9b14e4c4f61"
EXPECTED_TRANSCRIPT_SHA256 = "75f9f04762a99dd0a551d1455b6c2c5d0244c8d5453a54084c34f046fcc78ffa"
EXPECTED_D351_HARNESS_SHA256 = "3c4501885af7590f5883b36666c984ce88728a24d40451ea0a600660a386107d"
EXPECTED_D358_EVIDENCE_SHA256 = "6c19cf6c3cd99b9567db65bf065afcb95872c4cfa6940c6584a97717638af3ff"
EXPECTED_D358_COMPLETION_SHA256 = "9ea631942cab32708cbc2f58e2b8351ad03dd2f45ff8c6f699caa44079e875f7"
EXPECTED_STAGE_ROOT_SHA256 = "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff"

REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
PXR_ROOT = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/"
    "extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
REGISTERED_PYTHONPATH = str(PXR_ROOT)
REGISTERED_LD_LIBRARY_PATH = (
    "/home/cgxr/miniconda3/envs/isaaclab/lib:"
    + str(PXR_ROOT / "bin")
)
AUDIT_TIMEOUT_SECONDS = 180

MESH_PATH = (
    "/roarm_m3/gripper_link/collisions/gripper_link/"
    "node_STL_BINARY_/mesh"
)
INNER_FACE_IDS = np.arange(672, 1165, dtype=np.int64)
OUTER_FACE_IDS = np.arange(13205, 13698, dtype=np.int64)

EXPECTED = {
    "inner_vertex": "13c65ee478a2668896ec2a8f1e237a9ba7b7e6e0ef40ab08cb350087d3a74d55",
    "outer_vertex": "0d9f7f856eb66d5f749303aa7f4bac8138a595d228dff8424221a6b0b732772a",
    "inner_triangle": "5644e9a66386d68945d340a46cfa9e1507b6dd55cf0b721823ef6afb079b9e17",
    "outer_triangle": "5644e9a66386d68945d340a46cfa9e1507b6dd55cf0b721823ef6afb079b9e17",
    "inner_patch": "c927e8c628073f9f1d8fc0250d8190a71bb2b0701b97b41d7f8069b216c3531b",
    "outer_patch": "9b430c7d7e8c389eb648726014d61169aa671ec910f94a782084b467e96d6486",
    "inner_paired_xz": "917b7154601d23984db01ebbd4adccdb272771920f225d1a021540b4b03bcaf9",
    "outer_paired_xz": "917b7154601d23984db01ebbd4adccdb272771920f225d1a021540b4b03bcaf9",
}

EXPECTED_STREAMS = {
    "points_f32_mm": "b89c67e99bd253ae710e6b0a2fcacd0b27263d6ede29fe6f6334ed70247895ed",
    "face_counts_i64": "f17eac58b9b109f98f7a69efcc3b1e64b632d805ccca8cc8883cf0349e07cb6c",
    "face_indices_i64": "205a08458b895d96c6eb9593d1f04a8815629f7f972a889cce683b86955f2545",
}

FORBIDDEN_MODULE_PREFIXES = (
    "isaacsim",
    "omni",
    "carb",
    "warp",
    "torch",
    "physx",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_x(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, value.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _run(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        list(args), cwd=REPO, text=True, capture_output=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed {args}: rc={result.returncode} stderr={result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git_status_filtered() -> list[str]:
    rows = _run("git", "status", "--porcelain=v1", "-uall").splitlines()
    prefix = "claudedocs/runtime_logs/grasp_track/g0a_d359/"
    return sorted(row for row in rows if prefix not in row)


def _sidecar_inventory() -> list[dict[str, Any]]:
    if not D334_SIDECAR.exists():
        return []
    rows = []
    for path in sorted(item for item in D334_SIDECAR.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "size": path.stat().st_size,
                "sha256": _sha_file(path),
            }
        )
    return rows


def _forbidden_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MODULE_PREFIXES)
    )


def _pxr_preflight() -> dict[str, Any]:
    before = _forbidden_modules()
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Smoke")
    result = {
        "python": str(Path(sys.executable).resolve()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "openusd_version": list(Usd.GetVersion()),
        "numpy_version": np.__version__,
        "psutil_version": psutil.__version__,
        "module_origins": {
            "Gf": str(Path(Gf.__file__).resolve()),
            "Usd": str(Path(Usd.__file__).resolve()),
            "UsdGeom": str(Path(UsdGeom.__file__).resolve()),
        },
        "in_memory_stage_valid": bool(stage.GetPrimAtPath("/Smoke").IsValid()),
        "forbidden_before": before,
        "forbidden_after": _forbidden_modules(),
    }
    result["checks"] = {
        "python_exact": result["python"] == str(Path(REGISTERED_PYTHON).resolve()),
        "pythonpath_exact": result["pythonpath"] == REGISTERED_PYTHONPATH,
        "ld_library_path_exact": result["ld_library_path"] == REGISTERED_LD_LIBRARY_PATH,
        "openusd_0_24_5": result["openusd_version"] == [0, 24, 5],
        "numpy_1_26_0": result["numpy_version"] == "1.26.0",
        "psutil_5_9_8": result["psutil_version"] == "5.9.8",
        "module_origins_registered": all(
            str(PXR_ROOT.resolve()) in origin for origin in result["module_origins"].values()
        ),
        "in_memory_stage_valid": result["in_memory_stage_valid"],
        "forbidden_modules_absent": not result["forbidden_before"] and not result["forbidden_after"],
    }
    result["pass"] = all(result["checks"].values())
    return result


def _input_hashes() -> dict[str, str]:
    return {
        "harness": _sha_file(HARNESS),
        "session": _sha_file(SESSION_DOC),
        "start_here": _sha_file(START_HERE),
        "d351_harness": _sha_file(D351_HARNESS),
        "d358_harness": _sha_file(D358_HARNESS),
        "d358_evidence": _sha_file(D358_EVIDENCE),
        "d358_completion": _sha_file(D358_COMPLETION),
        "transcript": _sha_file(TRANSCRIPT),
        "d344_stage_root": _sha_file(D344_STAGE),
        "local_stage_root": _sha_file(LOCAL_STAGE),
    }


def _prepare() -> None:
    if OUT_DIR.exists():
        raise RuntimeError(f"forward-only output already exists: {OUT_DIR}")
    head = _run("git", "rev-parse", "HEAD")
    origin = _run("git", "rev-parse", "origin/master")
    hashes = _input_hashes()
    pxr = _pxr_preflight()
    checks = {
        "head_exact": head == EXPECTED_HEAD,
        "origin_exact": origin == EXPECTED_HEAD,
        "d351_harness_exact": hashes["d351_harness"] == EXPECTED_D351_HARNESS_SHA256,
        "d358_evidence_exact": hashes["d358_evidence"] == EXPECTED_D358_EVIDENCE_SHA256,
        "d358_completion_exact": hashes["d358_completion"] == EXPECTED_D358_COMPLETION_SHA256,
        "transcript_exact": hashes["transcript"] == EXPECTED_TRANSCRIPT_SHA256,
        "d344_root_exact": hashes["d344_stage_root"] == EXPECTED_STAGE_ROOT_SHA256,
        "local_root_exact": hashes["local_stage_root"] == EXPECTED_STAGE_ROOT_SHA256,
        "sidecar_present": bool(_sidecar_inventory()),
        "pxr_preflight_pass": pxr["pass"],
    }
    prereg = {
        "artifact": "D359_PREREGISTRATION_V1",
        "case": CASE,
        "head": head,
        "origin_master": origin,
        "new_variables": [
            "historical_generator_transcript_lineage",
            "historical_source_point_id_remap_replay",
        ],
        "new_physical_variables": [],
        "input_hashes": hashes,
        "git_status_filtered": _git_status_filtered(),
        "sidecar_inventory": _sidecar_inventory(),
        "pxr_preflight": pxr,
        "expected_historical_bundle": EXPECTED,
        "registered_matrix": {
            "sources": ["d344_attempt3_composed", "local_composed"],
            "remaps": ["original_point_id_ascending", "coordinate_lexicographic_unique"],
            "independent_replay": "python_tuple_dict_struct_pack",
            "negative_controls": ["reverse_original_point_id", "coordinate_remap_must_not_equal_historical_six"],
        },
        "single_invocation": {
            "audit_count": 1,
            "timeout_seconds": AUDIT_TIMEOUT_SECONDS,
            "no_retry_or_overwrite": True,
        },
        "rerun_omission": "pure text/file/hash/schema audit; no spatial or temporal judgment",
        "scope_guards": {
            "simulation_app_kit_isaac_gpu": 0,
            "q5_science_or_state_write": 0,
            "physics_steps": 0,
            "distance_contact_cap_rim_queries": 0,
            "asset_gate_hash_target_ik_path_changes": 0,
            "d334_sidecar_writes": 0,
            "commit_push": 0,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    if not prereg["pass"]:
        raise RuntimeError(f"D359 prepare failed: {checks}")
    print(json.dumps({"stage": "prepare", "pass": True, "path": str(PREREG_PATH.relative_to(REPO))}))


def _phase(start_ns: int, name: str, details: dict[str, Any] | None = None) -> None:
    row = {
        "seq": sum(1 for _ in PHASE_PATH.open("r", encoding="utf-8")) if PHASE_PATH.exists() else 0,
        "phase": name,
        "elapsed_seconds": (time.monotonic_ns() - start_ns) / 1e9,
        "details": details or {},
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _text_from_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(_text_from_output(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return "\n".join(_text_from_output(item) for item in value.values())
    return ""


def _transcript_lineage() -> dict[str, Any]:
    meta = None
    call = None
    output = None
    narration = None
    hash_occurrence_lines = []
    with TRANSCRIPT.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            row = json.loads(raw)
            payload = row.get("payload", {})
            if row.get("type") == "session_meta" and payload.get("id") == "019f6463-d763-7761-bb62-68bfe4f993f2":
                meta = {"line": line_number, "timestamp": row.get("timestamp"), "payload": payload}
            if EXPECTED["inner_vertex"] in raw:
                hash_occurrence_lines.append({"line": line_number, "timestamp": row.get("timestamp"), "type": row.get("type")})
            if payload.get("type") == "custom_tool_call" and payload.get("name") == "exec":
                command = payload.get("input", "")
                if (
                    "g0a_d344/collision_asset/attempt3/roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd" in command
                    and "old=np.unique(f[ids].reshape(-1))" in command
                    and "u=p[old]" in command
                ):
                    call = {
                        "line": line_number,
                        "timestamp": row.get("timestamp"),
                        "call_id": payload.get("call_id"),
                        "input_sha256": _sha_bytes(command.encode("utf-8")),
                        "input_length": len(command),
                        "uses_d344_attempt3_composed_stage": True,
                        "uses_original_point_id_ascending": True,
                        "uses_coordinate_row_unique": "return_inverse=True" in command,
                    }
            if payload.get("type") == "custom_tool_call_output" and call and payload.get("call_id") == call["call_id"]:
                body = _text_from_output(payload.get("output"))
                output = {
                    "line": line_number,
                    "timestamp": row.get("timestamp"),
                    "call_id": payload.get("call_id"),
                    "output_sha256": _sha_bytes(body.encode("utf-8")),
                    "contains_expected": {key: value in body for key, value in EXPECTED.items()},
                    "contains_all_expected": all(value in body for value in EXPECTED.values()),
                }
            if payload.get("type") == "agent_message":
                message = payload.get("message", "")
                if EXPECTED["inner_vertex"] in message and "local_assets/roarm_m3/usd/roarm_m3.usd" in message:
                    narration = {
                        "line": line_number,
                        "timestamp": row.get("timestamp"),
                        "message_sha256": _sha_bytes(message.encode("utf-8")),
                        "claims_local_asset_path": True,
                        "claims_coordinate_lexicographic_unique": "return_inverse=True" in message,
                    }
    source = ((meta or {}).get("payload") or {}).get("source", {})
    subagent = source.get("subagent", {}).get("thread_spawn", {}) if isinstance(source, dict) else {}
    checks = {
        "session_meta_found": meta is not None,
        "subagent_path_d351_patch_design": subagent.get("agent_path") == "/root/d351_patch_design",
        "actual_call_found": call is not None,
        "bound_output_found": output is not None,
        "actual_output_contains_all_expected": bool(output and output["contains_all_expected"]),
        "later_narration_found": narration is not None,
        "actual_call_and_later_narration_semantics_differ": bool(
            call
            and narration
            and call["uses_d344_attempt3_composed_stage"]
            and call["uses_original_point_id_ascending"]
            and narration["claims_local_asset_path"]
            and narration["claims_coordinate_lexicographic_unique"]
        ),
    }
    return {
        "transcript_path": str(TRANSCRIPT),
        "transcript_sha256": _sha_file(TRANSCRIPT),
        "session_meta": None
        if meta is None
        else {
            "line": meta["line"],
            "timestamp": meta["timestamp"],
            "session_id": meta["payload"].get("id"),
            "forked_from_id": meta["payload"].get("forked_from_id"),
            "agent_path": subagent.get("agent_path"),
            "agent_nickname": subagent.get("agent_nickname"),
            "git_commit_at_session_start": (meta["payload"].get("git") or {}).get("commit_hash"),
        },
        "actual_generator_call": call,
        "actual_generator_output": output,
        "later_narration": narration,
        "hash_occurrence_lines": hash_occurrence_lines,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _git_lineage() -> dict[str, Any]:
    rows = {}
    first_commits = []
    for field, value in EXPECTED.items():
        history = _run(
            "git",
            "log",
            "--all",
            "--reverse",
            "--format=%H|%aI|%an|%s",
            f"-S{value}",
            "--",
            ".",
        ).splitlines()
        first = history[0] if history else None
        commit = first.split("|", 1)[0] if first else None
        rows[field] = {"first": first, "history_count": len(history), "history": history}
        first_commits.append(commit)
    parent = _run("git", "rev-parse", EXPECTED_FIRST_COMMIT + "^")
    parent_has_path = subprocess.run(
        ["git", "cat-file", "-e", f"{parent}:sim_scripts/cyl34_top_view_d351_zero_step_closure_geometry.py"],
        cwd=REPO,
        capture_output=True,
        check=False,
    ).returncode == 0
    committed_text = _run(
        "git", "show", f"{EXPECTED_FIRST_COMMIT}:sim_scripts/cyl34_top_view_d351_zero_step_closure_geometry.py"
    )
    checks = {
        "all_fields_have_history": all(rows[field]["first"] for field in rows),
        "all_first_introduced_same_commit": set(first_commits) == {EXPECTED_FIRST_COMMIT},
        "parent_lacks_d351_harness": not parent_has_path,
        "first_commit_file_contains_all_expected": all(value in committed_text for value in EXPECTED.values()),
    }
    return {
        "fields": rows,
        "first_commit": EXPECTED_FIRST_COMMIT,
        "parent_commit": parent,
        "parent_has_d351_harness": parent_has_path,
        "committed_d351_blob_sha256": _sha_bytes(committed_text.encode("utf-8")),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _layer_inventory(stage: Any) -> list[dict[str, Any]]:
    rows = []
    for layer in stage.GetUsedLayers():
        identifier = str(layer.identifier)
        path = Path(identifier)
        row = {"identifier": identifier, "real_path": str(path.resolve()) if path.exists() else None}
        if path.exists() and path.is_file():
            row.update({"size": path.stat().st_size, "sha256": _sha_file(path)})
        rows.append(row)
    return sorted(rows, key=lambda item: item["identifier"])


def _load_stage(path: Path) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open {path}")
    prim = stage.GetPrimAtPath(MESH_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"missing mesh {MESH_PATH} in {path}")
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray([[float(v[0]), float(v[1]), float(v[2])] for v in mesh.GetPointsAttr().Get()], dtype="<f4")
    counts = np.asarray([int(value) for value in mesh.GetFaceVertexCountsAttr().Get()], dtype="<i8")
    indices = np.asarray([int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()], dtype="<i8")
    streams = {
        "points_f32_mm": _sha_bytes(np.ascontiguousarray(points, dtype="<f4").tobytes()),
        "face_counts_i64": _sha_bytes(np.ascontiguousarray(counts, dtype="<i8").tobytes()),
        "face_indices_i64": _sha_bytes(np.ascontiguousarray(indices, dtype="<i8").tobytes()),
    }
    return {
        "stage_path": str(path.relative_to(REPO)),
        "root_sha256": _sha_file(path),
        "used_layers": _layer_inventory(stage),
        "points": points,
        "counts": counts,
        "indices": indices,
        "stream_hashes": streams,
        "point_count": len(points),
        "face_count": len(counts),
        "all_triangles": bool(np.all(counts == 3)),
    }


def _blob(array: Any, dtype: str) -> bytes:
    return np.ascontiguousarray(array, dtype=dtype).tobytes(order="C")


def _patch(points: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, remap: str) -> dict[str, Any]:
    ids = np.sort(np.asarray(face_ids, dtype="<i8"))
    selected = np.asarray(faces[ids], dtype=np.int64)
    if remap == "original_point_id_ascending":
        old = np.unique(selected.reshape(-1))
        lookup = {int(value): index for index, value in enumerate(old.tolist())}
        vertices = np.asarray(points[old], dtype="<f4")
        triangles = np.asarray([[lookup[int(value)] for value in row] for row in selected], dtype="<i8")
        source_point_ids = old
    elif remap == "coordinate_lexicographic_unique":
        triangle_points = points[selected]
        vertices, inverse = np.unique(triangle_points.reshape(-1, 3), axis=0, return_inverse=True)
        vertices = np.asarray(vertices, dtype="<f4")
        triangles = np.asarray(inverse.reshape(-1, 3), dtype="<i8")
        source_point_ids = None
    elif remap == "reverse_original_point_id":
        old = np.unique(selected.reshape(-1))[::-1]
        lookup = {int(value): index for index, value in enumerate(old.tolist())}
        vertices = np.asarray(points[old], dtype="<f4")
        triangles = np.asarray([[lookup[int(value)] for value in row] for row in selected], dtype="<i8")
        source_point_ids = old
    else:
        raise ValueError(remap)
    face_blob = _blob(ids, "<i8")
    vertex_blob = _blob(vertices, "<f4")
    triangle_blob = _blob(triangles, "<i8")
    xz = np.asarray(np.unique(vertices[:, [0, 2]], axis=0), dtype="<f4")
    return {
        "face_sha256": _sha_bytes(face_blob),
        "vertex_sha256": _sha_bytes(vertex_blob),
        "triangle_sha256": _sha_bytes(triangle_blob),
        "patch_sha256": _sha_bytes(face_blob + vertex_blob + triangle_blob),
        "paired_xz_sha256": _sha_bytes(_blob(xz, "<f4")),
        "face_count": len(ids),
        "vertex_count": len(vertices),
        "source_point_id_count": None if source_point_ids is None else len(source_point_ids),
        "blob_lengths": {
            "face": len(face_blob),
            "vertex": len(vertex_blob),
            "triangle": len(triangle_blob),
            "patch": len(face_blob + vertex_blob + triangle_blob),
        },
    }


def _bundle(points: np.ndarray, indices: np.ndarray, remap: str) -> dict[str, Any]:
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    inner = _patch(points, faces, INNER_FACE_IDS, remap)
    outer = _patch(points, faces, OUTER_FACE_IDS, remap)
    bundle = {
        "inner_vertex": inner["vertex_sha256"],
        "outer_vertex": outer["vertex_sha256"],
        "inner_triangle": inner["triangle_sha256"],
        "outer_triangle": outer["triangle_sha256"],
        "inner_patch": inner["patch_sha256"],
        "outer_patch": outer["patch_sha256"],
        "inner_paired_xz": inner["paired_xz_sha256"],
        "outer_paired_xz": outer["paired_xz_sha256"],
    }
    return {
        "remap": remap,
        "bundle": bundle,
        "matches_historical_by_field": {key: bundle[key] == EXPECTED[key] for key in EXPECTED},
        "historical_match_count": sum(bundle[key] == EXPECTED[key] for key in EXPECTED),
        "inner": inner,
        "outer": outer,
    }


def _pure_python_patch(points: np.ndarray, faces: np.ndarray, face_ids: np.ndarray) -> dict[str, str]:
    ids = sorted(int(value) for value in face_ids)
    selected = [[int(value) for value in faces[face_id]] for face_id in ids]
    used = sorted({value for row in selected for value in row})
    lookup = {value: index for index, value in enumerate(used)}
    vertices = [tuple(float(component) for component in points[value]) for value in used]
    triangles = [[lookup[value] for value in row] for row in selected]
    face_blob = b"".join(struct.pack("<q", value) for value in ids)
    vertex_blob = b"".join(struct.pack("<fff", *row) for row in vertices)
    triangle_blob = b"".join(struct.pack("<qqq", *row) for row in triangles)
    xz = sorted(set((row[0], row[2]) for row in vertices))
    xz_blob = b"".join(struct.pack("<ff", *row) for row in xz)
    return {
        "vertex_sha256": _sha_bytes(vertex_blob),
        "triangle_sha256": _sha_bytes(triangle_blob),
        "patch_sha256": _sha_bytes(face_blob + vertex_blob + triangle_blob),
        "paired_xz_sha256": _sha_bytes(xz_blob),
    }


def _pure_python_bundle(points: np.ndarray, indices: np.ndarray) -> dict[str, str]:
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    inner = _pure_python_patch(points, faces, INNER_FACE_IDS)
    outer = _pure_python_patch(points, faces, OUTER_FACE_IDS)
    return {
        "inner_vertex": inner["vertex_sha256"],
        "outer_vertex": outer["vertex_sha256"],
        "inner_triangle": inner["triangle_sha256"],
        "outer_triangle": outer["triangle_sha256"],
        "inner_patch": inner["patch_sha256"],
        "outer_patch": outer["patch_sha256"],
        "inner_paired_xz": inner["paired_xz_sha256"],
        "outer_paired_xz": outer["paired_xz_sha256"],
    }


def _matrix() -> dict[str, Any]:
    sources = {
        "d344_attempt3_composed": _load_stage(D344_STAGE),
        "local_composed": _load_stage(LOCAL_STAGE),
    }
    rows = {}
    for source_name, source in sources.items():
        rows[source_name] = {}
        for remap in ("original_point_id_ascending", "coordinate_lexicographic_unique"):
            rows[source_name][remap] = _bundle(source["points"], source["indices"], remap)
        rows[source_name]["reverse_original_point_id"] = _bundle(
            source["points"], source["indices"], "reverse_original_point_id"
        )
    d358 = _json(D358_EVIDENCE)
    current_bundle = d358["authored_current_recipe"]["bundle"]
    pure = _pure_python_bundle(sources["d344_attempt3_composed"]["points"], sources["d344_attempt3_composed"]["indices"])
    source_public = {}
    for name, source in sources.items():
        source_public[name] = {key: value for key, value in source.items() if key not in {"points", "counts", "indices"}}
    checks = {
        "both_streams_match_frozen": all(source["stream_hashes"] == EXPECTED_STREAMS for source in sources.values()),
        "source_streams_identical": sources["d344_attempt3_composed"]["stream_hashes"] == sources["local_composed"]["stream_hashes"],
        "d344_historical_point_id_8_of_8": rows["d344_attempt3_composed"]["original_point_id_ascending"]["bundle"] == EXPECTED,
        "local_historical_point_id_8_of_8": rows["local_composed"]["original_point_id_ascending"]["bundle"] == EXPECTED,
        "d344_coordinate_matches_d358_current_8_of_8": rows["d344_attempt3_composed"]["coordinate_lexicographic_unique"]["bundle"] == current_bundle,
        "local_coordinate_matches_d358_current_8_of_8": rows["local_composed"]["coordinate_lexicographic_unique"]["bundle"] == current_bundle,
        "coordinate_remap_matches_only_paired_xz_historical_2_of_8": rows["local_composed"]["coordinate_lexicographic_unique"]["historical_match_count"] == 2,
        "reverse_point_id_rejected": rows["local_composed"]["reverse_original_point_id"]["bundle"] != EXPECTED,
        "independent_struct_pack_matches_historical": pure == EXPECTED,
    }
    return {
        "sources": source_public,
        "rows": rows,
        "d358_current_authored_bundle": current_bundle,
        "independent_struct_pack_bundle": pure,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _d358_omission_audit() -> dict[str, Any]:
    text = D358_HARNESS.read_text(encoding="utf-8")
    d351_text = D351_HARNESS.read_text(encoding="utf-8")
    snippets = {
        "d358_coordinate_tuple_flatten": "flat.append(vertex)" in text,
        "d358_vertex_axes_coordinate_only": (
            'recipe["vertex_order"] == "lexicographic_unique"' in text
            and "seen: dict[tuple[float, float, float], int]" in text
        ),
        "d358_original_point_id_recipe_absent": "original_point_id_ascending" not in text,
        "d351_runtime_validator_coordinate_unique": (
            "triangle_points.reshape(-1, 3), axis=0, return_inverse=True" in d351_text
        ),
        "d351_expected_constants_present": all(value in d351_text for value in EXPECTED.values()),
    }
    return {
        "d358_harness_sha256": _sha_file(D358_HARNESS),
        "d351_harness_sha256": _sha_file(D351_HARNESS),
        "checks": snippets,
        "pass": all(snippets.values()),
        "explanation": (
            "D358 varied ordering of coordinate tuples after source point IDs had already been discarded; "
            "the historical generator keyed the remap by ascending original point ID."
        ),
    }


def _report(evidence: dict[str, Any]) -> str:
    matrix = evidence["replay_matrix"]
    point_row = matrix["rows"]["d344_attempt3_composed"]["original_point_id_ascending"]
    coord_row = matrix["rows"]["local_composed"]["coordinate_lexicographic_unique"]
    return f"""# D359 historical hash provenance report

Verdict: `{evidence['verdict']}`

## 초보자용 핵심

과거 여섯 해시는 형상이 달라서 생긴 값이 아니었다. 같은 483개 patch 정점을
`원본 point ID 순서`로 나열한 one-off generator가 만든 값이었다. D351의 실제
validator와 D358 search는 좌표값을 기준으로 정점을 다시 나열했다. 정점 목록의
순서가 달라지면 같은 형상도 byte stream과 SHA-256이 달라진다.

- Historical original-point-ID replay: `{point_row['historical_match_count']}/8`
- Later coordinate-row replay: `{coord_row['historical_match_count']}/8`
- Independent tuple/dict/struct replay: `{matrix['checks']['independent_struct_pack_matches_historical']}`
- First Git introduction: `{evidence['git_lineage']['first_commit']}`
- Actual generator transcript line/output: `{evidence['transcript_lineage']['actual_generator_call']['line']}` / `{evidence['transcript_lineage']['actual_generator_output']['line']}`

This closes the historical generator lineage. It does not rewrite D351/D354/D358,
does not run Isaac or physics, and does not decide contact or grasp.
"""


def _audit() -> None:
    if not PREREG_PATH.exists():
        raise RuntimeError("D359 preregistration is missing")
    if any(path.exists() for path in [INVOCATION_PATH, PHASE_PATH, EVIDENCE_PATH, REPORT_PATH, COMPLETION_PATH]):
        raise RuntimeError("D359 audit output already exists; no retry or overwrite")
    prereg = _json(PREREG_PATH)
    current_hashes = _input_hashes()
    prechecks = {
        "prereg_pass": prereg.get("pass") is True,
        "head_exact": _run("git", "rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_exact": _run("git", "rev-parse", "origin/master") == EXPECTED_HEAD,
        "harness_unchanged": current_hashes["harness"] == prereg["input_hashes"]["harness"],
        "session_unchanged": current_hashes["session"] == prereg["input_hashes"]["session"],
        "start_here_unchanged": current_hashes["start_here"] == prereg["input_hashes"]["start_here"],
        "all_frozen_inputs_unchanged": all(
            current_hashes[key] == value for key, value in prereg["input_hashes"].items()
        ),
        "git_status_filtered_unchanged": _git_status_filtered() == prereg["git_status_filtered"],
        "sidecar_unchanged_before": _sidecar_inventory() == prereg["sidecar_inventory"],
        "pxr_preflight_pass": _pxr_preflight()["pass"],
    }
    if not all(prechecks.values()):
        raise RuntimeError(f"D359 audit precheck failed: {prechecks}")

    start_ns = time.monotonic_ns()
    _write_json_x(
        INVOCATION_PATH,
        {
            "artifact": "D359_SOLE_AUDIT_INVOCATION_V1",
            "case": CASE,
            "pid": os.getpid(),
            "monotonic_start_ns": start_ns,
            "count": 1,
            "no_retry": True,
            "prechecks": prechecks,
        },
    )
    _phase(start_ns, "audit_started")

    def _timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"D359 exceeded {AUDIT_TIMEOUT_SECONDS}s")

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(AUDIT_TIMEOUT_SECONDS)
    try:
        transcript = _transcript_lineage()
        _phase(start_ns, "transcript_generator_bound", {"pass": transcript["pass"]})
        git_lineage = _git_lineage()
        _phase(start_ns, "git_first_introduction_bound", {"pass": git_lineage["pass"]})
        matrix = _matrix()
        _phase(
            start_ns,
            "source_remap_matrix_replayed",
            {
                "pass": matrix["pass"],
                "historical_match_count": matrix["rows"]["d344_attempt3_composed"]["original_point_id_ascending"]["historical_match_count"],
                "coordinate_match_count": matrix["rows"]["local_composed"]["coordinate_lexicographic_unique"]["historical_match_count"],
            },
        )
        omission = _d358_omission_audit()
        _phase(start_ns, "d358_recipe_axis_gap_audited", {"pass": omission["pass"]})
        final_checks = {
            "transcript_lineage_pass": transcript["pass"],
            "git_lineage_pass": git_lineage["pass"],
            "source_remap_replay_pass": matrix["pass"],
            "d358_omission_audit_pass": omission["pass"],
            "sidecar_unchanged_after": _sidecar_inventory() == prereg["sidecar_inventory"],
            "forbidden_modules_absent": not _forbidden_modules(),
        }
        if all(final_checks.values()):
            verdict = "D359_D351_HASH_PROVENANCE_RECOVERED"
        elif transcript["pass"] and git_lineage["pass"]:
            verdict = "D359_D351_HASH_PROVENANCE_PARTIAL_FAIL_STOP"
        else:
            verdict = "D359_D351_HASH_PROVENANCE_UNRESOLVED_FAIL_STOP"
        evidence = {
            "artifact": "D359_D351_HISTORICAL_HASH_PROVENANCE_EVIDENCE_V1",
            "case": CASE,
            "verdict": verdict,
            "transcript_lineage": transcript,
            "git_lineage": git_lineage,
            "replay_matrix": matrix,
            "d358_omitted_axis_audit": omission,
            "root_cause": {
                "actual_generator_source": "D344 attempt3 composed stage",
                "actual_generator_remap": "ascending original USD point ID",
                "later_validator_remap": "coordinate-row lexicographic unique",
                "discriminating_variable": "vertex remap key/order, not geometry or metric unit",
                "why_paired_xz_still_matched": (
                    "paired-XZ applies a second coordinate sort, erasing the prior vertex-list ordering difference"
                ),
                "why_d358_missed_it": omission["explanation"],
            },
            "interpretation_boundary": {
                "d351_d354_d358_immutable": True,
                "expected_hash_replacement_performed": False,
                "gate_relaxation_performed": False,
                "isaac_or_physics_run": False,
                "contact_grasp_target_ik_decision": None,
                "g0a_pass": False,
            },
            "scope_counts": {
                "simulation_app_kit_isaac_gpu": 0,
                "q5_science_or_state_write": 0,
                "physics_steps": 0,
                "distance_contact_cap_rim_queries": 0,
                "asset_gate_hash_target_ik_path_changes": 0,
                "d334_sidecar_writes": 0,
                "commit_push": 0,
            },
            "checks": final_checks,
            "pass": verdict == "D359_D351_HASH_PROVENANCE_RECOVERED",
            "elapsed_seconds": (time.monotonic_ns() - start_ns) / 1e9,
        }
        _write_json_x(EVIDENCE_PATH, evidence)
        _phase(start_ns, "authoritative_evidence_written", {"verdict": verdict})
        _write_text_x(REPORT_PATH, _report(evidence))
        _phase(start_ns, "report_written")
        precompletion_inventory = sorted(path.name for path in OUT_DIR.iterdir() if path.is_file())
        expected_precompletion = sorted(
            [PREREG_PATH.name, INVOCATION_PATH.name, PHASE_PATH.name, EVIDENCE_PATH.name, REPORT_PATH.name]
        )
        completion_checks = {
            "evidence_pass": evidence["pass"],
            "precompletion_inventory_exact": precompletion_inventory == expected_precompletion,
            "sidecar_unchanged": _sidecar_inventory() == prereg["sidecar_inventory"],
            "single_invocation_count": _json(INVOCATION_PATH)["count"] == 1,
        }
        completion = {
            "artifact": "D359_COMPLETION_SUMMARY_V1",
            "case": CASE,
            "verdict": verdict,
            "operational_pass": True,
            "provenance_recovered": evidence["pass"],
            "g0a_pass": False,
            "precompletion_inventory": precompletion_inventory,
            "evidence_sha256": _sha_file(EVIDENCE_PATH),
            "report_sha256": _sha_file(REPORT_PATH),
            "phase_markers_sha256_before_completion": _sha_file(PHASE_PATH),
            "checks": completion_checks,
            "pass": all(completion_checks.values()),
            "elapsed_seconds": (time.monotonic_ns() - start_ns) / 1e9,
        }
        if not completion["pass"]:
            raise RuntimeError(f"D359 completion gate failed: {completion_checks}")
        _phase(start_ns, "completion_ready", {"verdict": verdict})
        completion["phase_markers_sha256"] = _sha_file(PHASE_PATH)
        _write_json_x(COMPLETION_PATH, completion)
    finally:
        signal.alarm(0)

    print(
        json.dumps(
            {
                "stage": "audit",
                "pass": True,
                "verdict": _json(COMPLETION_PATH)["verdict"],
                "path": str(COMPLETION_PATH.relative_to(REPO)),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "audit"], required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        _prepare()
    else:
        _audit()


if __name__ == "__main__":
    main()
