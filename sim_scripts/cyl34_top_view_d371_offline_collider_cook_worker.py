#!/usr/bin/env python3
"""D371 cook-only worker for the offline collider comparison.

The worker launches one headless Isaac application solely to call the frozen
D339 callback-first PhysX cooking routine.  It does not construct a simulation
world, advance time, sample joints, query contacts, or write any USD asset.

Four candidates are cooked twice on independent in-memory stages:

* R64: the frozen raw body mesh with ``maxConvexHulls=64``;
* R32: the same raw body mesh with ``maxConvexHulls=32``;
* C1: all non-retained current D348 callback parts concatenated and cooked with
  ``maxConvexHulls=1``;
* C2: the corresponding, less aggressive remainder cooked with the same
  single-hull cap.

For C1/C2, retained parts are inventory references only.  This worker never
rewrites them or combines them into an asset.  Original callback witnesses are
the downstream geometry authority; the D339 Qhull canonical files are retained
only as repeatability diagnostics.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import numpy as np


REPO = Path(__file__).resolve().parents[1]
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

BODIES = ("link5", "gripper_link")
CANDIDATES = ("R64", "R32", "C1", "C2")

BASE_PARAMS: dict[str, Any] = {
    "hull_vertex_limit": 64,
    "voxel_resolution": 1_000_000,
    "error_percentage": 1.0,
    "min_thickness_m": 0.0001,
    "shrink_wrap": True,
}
CANDIDATE_MAX_HULLS = {"R64": 64, "R32": 32, "C1": 1, "C2": 1}

RETAINED_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "C1": {
        "link5": tuple(f"part_{index:03d}" for index in (27, 29, 30, 31)),
        "gripper_link": tuple(
            f"part_{index:03d}"
            for index in (30, 35, 42, 45, 46, 47, 48, 50, 51, 53, 56, 58, 59, 60, 61, 62, 63)
        ),
    },
    "C2": {
        "link5": tuple(
            f"part_{index:03d}"
            for index in (13, 22, 23, 25, 26, 27, 28, 29, 30, 31)
        ),
        "gripper_link": tuple(
            f"part_{index:03d}"
            for index in (
                29,
                30,
                35,
                39,
                42,
                44,
                45,
                46,
                47,
                48,
                50,
                51,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
            )
        ),
    },
}

SUMMARY_NAME = "d371_cook_worker_summary.json"
PRECLOSE_NAME = "d371_preclose_sentinel.json"
CLAIM_NAME = "d371_cook_worker_claim.json"


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_sanitized(text: str) -> str:
    return text.replace(str(REPO.resolve()) + os.sep, "")


def _array_blob(array: Any, dtype: str) -> bytes:
    return np.ascontiguousarray(np.asarray(array, dtype=dtype)).tobytes(order="C")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return _repo_rel(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _params(candidate: str) -> dict[str, Any]:
    return {**BASE_PARAMS, "max_convex_hulls": CANDIDATE_MAX_HULLS[candidate]}


def _empty_cook_record(candidate: str, body: str) -> dict[str, Any]:
    source_role = (
        "raw_full_mesh"
        if candidate in ("R64", "R32")
        else "current_d348_nonretained_callback_triangle_concatenation"
    )
    row: dict[str, Any] = {
        "source_role": source_role,
        "params": _params(candidate),
        "cold1": {
            "callback_witness_path": None,
            "canonical_geometry_path": None,
            "hard_pass": False,
        },
        "cold2": {
            "callback_witness_path": None,
            "canonical_geometry_path": None,
            "hard_pass": False,
        },
        "reproducibility": {"pass": False, "status": "not_run"},
        "pass": False,
    }
    if candidate in ("C1", "C2"):
        row.update(
            {
                "retained_names": list(RETAINED_NAMES[candidate][body]),
                "remainder_source": None,
            }
        )
    else:
        row["source"] = None
    return row


def _initial_summary(out_dir: Path) -> dict[str, Any]:
    return {
        "artifact": "D371_OFFLINE_COLLIDER_COOK_WORKER_SUMMARY_V1",
        "case": "g0a_d371",
        "worker_invocation_count": 1,
        "controlled_app_launcher_instances": 1,
        "controlled_physx_cook_requests": 0,
        "controlled_in_memory_cook_stages": 0,
        "controlled_simulation_context_constructions": 0,
        "controlled_resets": 0,
        "controlled_environment_resets": 0,
        "controlled_physics_steps": 0,
        "controlled_timeline_requests": 0,
        "controlled_q5_samples": 0,
        "controlled_contact_queries": 0,
        "controlled_live_contact_queries": 0,
        "controlled_cylinder_pose_writes": 0,
        "controlled_target_ik_path_changes": 0,
        "controlled_material_mass_actuator_physics_changes": 0,
        "controlled_usd_asset_writes": 0,
        "controlled_canonical_or_live_asset_writes": 0,
        "app_launches": 1,
        "headless": True,
        "out_dir": _repo_rel(out_dir),
        "geometry_authority": (
            "D339 callback witness vertices/indices/polygons; canonical Qhull JSON is "
            "repeatability diagnostic only"
        ),
        "inputs": {},
        "parameter_guards": [],
        "stage_cache_lifecycle": [],
        "cooks": {
            candidate: {
                body: _empty_cook_record(candidate, body) for body in BODIES
            }
            for candidate in CANDIDATES
        },
        "preclose_sentinel_path": _repo_rel(out_dir / PRECLOSE_NAME),
        "exception": None,
        "checks": {},
        "pass": False,
    }


def _source_summary(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    vertices_f64 = np.asarray(vertices, dtype="<f8")
    triangles_i64 = np.asarray(triangles, dtype="<i8")
    checks = {
        "vertices_n_by_3_finite": bool(
            vertices_f64.ndim == 2
            and vertices_f64.shape[1:] == (3,)
            and len(vertices_f64) >= 4
            and np.all(np.isfinite(vertices_f64))
        ),
        "triangles_n_by_3_nonempty": bool(
            triangles_i64.ndim == 2
            and triangles_i64.shape[1:] == (3,)
            and len(triangles_i64) >= 1
        ),
        "triangle_indices_in_range": bool(
            triangles_i64.size
            and int(triangles_i64.min()) >= 0
            and int(triangles_i64.max()) < len(vertices_f64)
        ),
    }
    return {
        "vertex_count": int(len(vertices_f64)),
        "triangle_count": int(len(triangles_i64)),
        "vertex_stream_dtype": "little-endian-float64",
        "triangle_stream_dtype": "little-endian-int64",
        "vertex_stream_sha256": _sha256_bytes(_array_blob(vertices_f64, "<f8")),
        "triangle_stream_sha256": _sha256_bytes(_array_blob(triangles_i64, "<i8")),
        "bounds_m": [vertices_f64.min(axis=0).tolist(), vertices_f64.max(axis=0).tolist()],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _concatenate_nonretained_parts(
    parts: list[dict[str, Any]], retained_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    retained_set = set(retained_names)
    available_names = [str(part["name"]) for part in parts]
    nonretained = [part for part in parts if str(part["name"]) not in retained_set]
    nonretained_names = [str(part["name"]) for part in nonretained]

    vertex_blocks: list[np.ndarray] = []
    triangle_blocks: list[np.ndarray] = []
    provenance_rows: list[dict[str, Any]] = []
    vertex_offset = 0
    for part in nonretained:
        vertices = np.asarray(part["vertices"], dtype="<f8")
        triangles = np.asarray(part["triangles"], dtype="<i8")
        vertex_blocks.append(vertices)
        triangle_blocks.append(triangles + vertex_offset)
        provenance_rows.append(
            {
                "name": str(part["name"]),
                "vertex_count": int(len(vertices)),
                "triangle_count": int(len(triangles)),
                "payload_sha256": str(part["payload_sha256"]),
                "callback_vertices_sha256": str(part["callback_vertices_sha256"]),
                "callback_topology_sha256": str(part["callback_topology_sha256"]),
                "witness_path": str(part["witness_path"]),
                "witness_sha256": str(part["witness_sha256"]),
            }
        )
        vertex_offset += len(vertices)

    if not vertex_blocks or not triangle_blocks:
        raise RuntimeError("D371 remainder cannot be empty")
    vertices = np.ascontiguousarray(np.concatenate(vertex_blocks, axis=0), dtype="<f8")
    triangles = np.ascontiguousarray(np.concatenate(triangle_blocks, axis=0), dtype="<i8")
    source = _source_summary(vertices, triangles)
    provenance_blob = json.dumps(
        provenance_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    retained_exact = set(retained_names).issubset(set(available_names))
    source.update(
        {
            "retained_names": list(retained_names),
            "retained_count": len(retained_names),
            "nonretained_names": nonretained_names,
            "nonretained_count": len(nonretained_names),
            "input_part_count": len(parts),
            "part_concatenation_order": nonretained_names,
            "part_provenance_sha256": _sha256_bytes(provenance_blob),
            "part_provenance": provenance_rows,
            "concatenation_policy": (
                "part-name ascending; callback vertices unchanged; callback polygon fan "
                "triangles unchanged except cumulative vertex-index offset; no weld/sort/Qhull"
            ),
        }
    )
    source["checks"].update(
        {
            "retained_names_exist": retained_exact,
            "retained_and_nonretained_partition_all_parts": bool(
                len(retained_names) + len(nonretained_names) == len(parts)
                and not (retained_set & set(nonretained_names))
                and retained_set | set(nonretained_names) == set(available_names)
            ),
            "concatenation_order_exact": nonretained_names == sorted(nonretained_names),
            "vertex_count_is_sum": int(len(vertices))
            == sum(row["vertex_count"] for row in provenance_rows),
            "triangle_count_is_sum": int(len(triangles))
            == sum(row["triangle_count"] for row in provenance_rows),
            "all_provenance_witness_paths_repo_relative": all(
                not Path(row["witness_path"]).is_absolute()
                and (REPO / row["witness_path"]).resolve().is_relative_to(REPO.resolve())
                for row in provenance_rows
            ),
        }
    )
    source["pass"] = all(source["checks"].values())
    return vertices, triangles, source


@contextlib.contextmanager
def _temporary_d339_params(
    d339: Any, params: dict[str, Any], guard_rows: list[dict[str, Any]], tag: str
) -> Iterator[None]:
    original = dict(d339.DECOMPOSITION_PARAMS)
    row: dict[str, Any] = {
        "tag": tag,
        "before": original,
        "requested": dict(params),
        "active_exact": False,
        "restored_exact": False,
    }
    d339.DECOMPOSITION_PARAMS.clear()
    d339.DECOMPOSITION_PARAMS.update(params)
    row["active"] = dict(d339.DECOMPOSITION_PARAMS)
    row["active_exact"] = row["active"] == params
    try:
        if not row["active_exact"]:
            raise RuntimeError(f"D371 parameter activation mismatch for {tag}")
        yield
    finally:
        d339.DECOMPOSITION_PARAMS.clear()
        d339.DECOMPOSITION_PARAMS.update(original)
        row["after"] = dict(d339.DECOMPOSITION_PARAMS)
        row["restored_exact"] = row["after"] == original
        row["pass"] = bool(row["active_exact"] and row["restored_exact"])
        guard_rows.append(row)


def _cold_public(cook: dict[str, Any]) -> dict[str, Any]:
    callback_path = cook.get("callback_witness_path")
    canonical_path = cook.get("canonical_geometry_path")

    def artifact_fields(value: Any) -> dict[str, Any]:
        if not value:
            return {"path": None, "exists": False, "sha256": None}
        path = REPO / str(value)
        return {
            "path": str(value),
            "exists": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
        }

    callback_artifact = artifact_fields(callback_path)
    canonical_artifact = artifact_fields(canonical_path)
    return {
        "callback_witness_path": callback_artifact["path"],
        "callback_witness_exists": callback_artifact["exists"],
        "callback_witness_sha256": callback_artifact["sha256"],
        "canonical_geometry_path": canonical_artifact["path"],
        "canonical_geometry_exists": canonical_artifact["exists"],
        "canonical_geometry_sha256": canonical_artifact["sha256"],
        "hard_pass": bool(cook.get("hard_pass")),
        "result": cook.get("result"),
        "stage_id": int(cook.get("stage_id", -1)),
        "stage_identifier": str(cook.get("stage_identifier", "")),
        "returned_part_count": len(cook.get("parts", [])),
        "source_payload": cook.get("source_payload"),
        "parameter_readback": cook.get("parameter_readback"),
        "parameter_readback_checks": cook.get("parameter_readback_checks"),
        "callback_checks": cook.get("callback_checks"),
        "checks": cook.get("checks"),
    }


def _stage_lifecycle(retained: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from pxr import Usd, UsdUtils

    cache = UsdUtils.StageCache.Get()
    rows: list[dict[str, Any]] = []
    for item in retained:
        stage = item["stage"]
        current_id = cache.GetId(stage)
        current_value = int(current_id.ToLongInt()) if current_id.IsValid() else -1
        found = (
            cache.Find(Usd.StageCache.Id.FromLongInt(item["recorded_stage_id"]))
            if item["recorded_stage_id"] >= 0
            else None
        )
        found_identifier = (
            str(found.GetRootLayer().identifier) if found is not None else None
        )
        checks = {
            "current_id_valid": bool(current_id.IsValid()),
            "current_id_matches_recorded": current_value == item["recorded_stage_id"],
            "recorded_id_resolves": found is not None,
            "resolved_identifier_matches_recorded": (
                found_identifier == item["recorded_identifier"]
            ),
            "retained_stage_identifier_unchanged": (
                str(stage.GetRootLayer().identifier) == item["recorded_identifier"]
            ),
        }
        rows.append(
            {
                "tag": item["tag"],
                "recorded_stage_id": item["recorded_stage_id"],
                "current_stage_id": current_value,
                "recorded_identifier": item["recorded_identifier"],
                "resolved_identifier": found_identifier,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    return rows


def _retain_stage(
    retained: list[dict[str, Any]], cook: dict[str, Any], tag: str
) -> None:
    retained.append(
        {
            "tag": tag,
            "stage": cook["_stage_guard"],
            "recorded_stage_id": int(cook["stage_id"]),
            "recorded_identifier": str(cook["stage_identifier"]),
        }
    )


def _cook_pair(
    *,
    d339: Any,
    out_dir: Path,
    candidate: str,
    body: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    summary_row: dict[str, Any],
    worker_summary: dict[str, Any],
    parameter_guards: list[dict[str, Any]],
    retained_stages: list[dict[str, Any]],
) -> None:
    candidate_lower = candidate.lower()
    body_tag = body
    params = _params(candidate)
    first_tag = f"d371_{candidate_lower}_{body_tag}_cold1"
    second_tag = f"d371_{candidate_lower}_{body_tag}_cold2"
    with _temporary_d339_params(
        d339, params, parameter_guards, f"{candidate}/{body}"
    ):
        worker_summary["controlled_physx_cook_requests"] += 1
        first = d339._cold_cook_decomposition(
            vertices,
            triangles,
            first_tag,
            out_dir / f"{first_tag}_callback_witness.json",
            out_dir / f"{first_tag}_canonical_geometry.json",
        )
        worker_summary["controlled_in_memory_cook_stages"] += 1
        _retain_stage(retained_stages, first, first_tag)
        worker_summary["controlled_physx_cook_requests"] += 1
        second = d339._cold_cook_decomposition(
            vertices,
            triangles,
            second_tag,
            out_dir / f"{second_tag}_callback_witness.json",
            out_dir / f"{second_tag}_canonical_geometry.json",
        )
        worker_summary["controlled_in_memory_cook_stages"] += 1
        _retain_stage(retained_stages, second, second_tag)
        reproducibility = d339._compare_cold_cooks(first, second)

    summary_row["cold1"] = _cold_public(first)
    summary_row["cold2"] = _cold_public(second)
    summary_row["reproducibility"] = reproducibility
    source = summary_row.get("source") or summary_row.get("remainder_source")
    expected_source_payload = {
        "vertex_stream_sha256": source["vertex_stream_sha256"],
        "triangle_stream_sha256": source["triangle_stream_sha256"],
        "vertex_count": source["vertex_count"],
        "triangle_count": source["triangle_count"],
    }
    one_hull_required = candidate in ("C1", "C2")
    one_hull_observed = bool(
        len(first["parts"]) == 1 and len(second["parts"]) == 1
    )
    summary_row["checks"] = {
        "source_pass": bool(
            source["pass"]
        ),
        "cold_source_payloads_exact": bool(
            first["source_payload"] == expected_source_payload
            and second["source_payload"] == expected_source_payload
        ),
        "cold_parameter_readback_contract": bool(
            all(first["parameter_readback_checks"].values())
            and all(second["parameter_readback_checks"].values())
            and first["parameter_readback"] == second["parameter_readback"]
        ),
        "cold_artifacts_exist": bool(
            summary_row["cold1"]["callback_witness_exists"]
            and summary_row["cold1"]["canonical_geometry_exists"]
            and summary_row["cold2"]["callback_witness_exists"]
            and summary_row["cold2"]["canonical_geometry_exists"]
        ),
        "cold_artifact_paths_repo_relative": all(
            isinstance(path, str)
            and not Path(path).is_absolute()
            and (REPO / path).resolve().is_relative_to(REPO.resolve())
            for path in (
                summary_row["cold1"]["callback_witness_path"],
                summary_row["cold1"]["canonical_geometry_path"],
                summary_row["cold2"]["callback_witness_path"],
                summary_row["cold2"]["canonical_geometry_path"],
            )
        ),
        "cold1_hard_pass": bool(first["hard_pass"]),
        "cold2_hard_pass": bool(second["hard_pass"]),
        "cold_cook_reproducibility_pass": bool(reproducibility["pass"]),
        "single_remainder_hull_if_required": (
            one_hull_observed if one_hull_required else True
        ),
    }
    summary_row["pass"] = all(summary_row["checks"].values())


def _execute(
    out_dir: Path, app: Any, retained_stages: list[dict[str, Any]]
) -> bool:
    from sim_scripts import (
        cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair as d339,
    )
    from sim_scripts import (
        cyl34_top_view_d368_current_64cap_semantic_allocation_audit as d368,
    )

    summary = _initial_summary(out_dir)
    original_d339_params = dict(d339.DECOMPOSITION_PARAMS)
    expected_d339_base = {**BASE_PARAMS, "max_convex_hulls": 64}
    try:
        raw = d368._load_raw_meshes()
        current_parts, current_inventory = d368._load_current_parts()
        raw_checks: dict[str, bool] = {}
        for body in BODIES:
            actual = raw[body]["stream_summary"]
            expected = d368.RAW_STREAM_EXPECTED[body]
            raw_checks[body] = all(actual.get(key) == value for key, value in expected.items())
        summary["inputs"] = {
            "raw_source": {
                "authoring_usd_path": _repo_rel(d368.AUTHORING_USD),
                "authoring_usd_sha256": _sha256_file(d368.AUTHORING_USD),
                "stream_summaries": {
                    body: raw[body]["stream_summary"] for body in BODIES
                },
                "expected_streams_exact": raw_checks,
                "pass": all(raw_checks.values()),
            },
            "current_d348_parts": {
                "evidence_path": _repo_rel(d368.D348_EVIDENCE),
                "evidence_sha256": _sha256_file(d368.D348_EVIDENCE),
                "inventory": current_inventory,
                "pass": bool(current_inventory["pass"]),
            },
            "d339_reused_functions": [
                "_cold_cook_decomposition",
                "_compare_cold_cooks",
            ],
            "d339_original_params": original_d339_params,
            "d339_expected_frozen_base_params": expected_d339_base,
            "d339_original_params_exact": original_d339_params == expected_d339_base,
        }
        if not all(raw_checks.values()):
            raise RuntimeError("D371 raw D368 stream contract failed")
        if not current_inventory["pass"]:
            raise RuntimeError("D371 current D348 callback-part inventory failed")
        if original_d339_params != expected_d339_base:
            raise RuntimeError("D371 imported D339 parameter baseline changed")

        source_arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            candidate: {} for candidate in CANDIDATES
        }
        for body in BODIES:
            raw_vertices = np.ascontiguousarray(raw[body]["vertices_m"], dtype="<f8")
            raw_triangles = np.ascontiguousarray(raw[body]["triangles"], dtype="<i8")
            for candidate in ("R64", "R32"):
                source = _source_summary(raw_vertices, raw_triangles)
                source_arrays[candidate][body] = (raw_vertices, raw_triangles)
                summary["cooks"][candidate][body]["source"] = source

            for candidate in ("C1", "C2"):
                remainder_vertices, remainder_triangles, remainder_source = (
                    _concatenate_nonretained_parts(
                        current_parts[body], RETAINED_NAMES[candidate][body]
                    )
                )
                source_arrays[candidate][body] = (
                    remainder_vertices,
                    remainder_triangles,
                )
                summary["cooks"][candidate][body]["remainder_source"] = remainder_source

        for candidate in CANDIDATES:
            for body in BODIES:
                vertices, triangles = source_arrays[candidate][body]
                _cook_pair(
                    d339=d339,
                    out_dir=out_dir,
                    candidate=candidate,
                    body=body,
                    vertices=vertices,
                    triangles=triangles,
                    summary_row=summary["cooks"][candidate][body],
                    worker_summary=summary,
                    parameter_guards=summary["parameter_guards"],
                    retained_stages=retained_stages,
                )
    except Exception as error:  # one actual worker invocation; never retry in-process
        summary["exception"] = {
            "type": type(error).__name__,
            "message": _repo_sanitized(str(error)),
            "traceback": _repo_sanitized(traceback.format_exc()),
        }
    finally:
        try:
            summary["stage_cache_lifecycle"] = _stage_lifecycle(retained_stages)
        except Exception as error:
            summary["stage_cache_lifecycle"] = []
            lifecycle_error = {
                "type": type(error).__name__,
                "message": _repo_sanitized(str(error)),
                "traceback": _repo_sanitized(traceback.format_exc()),
            }
            if summary["exception"] is None:
                summary["exception"] = lifecycle_error
            else:
                summary["stage_cache_lifecycle_exception"] = lifecycle_error
        stage_ids = [row["recorded_stage_id"] for row in summary["stage_cache_lifecycle"]]
        stage_identifiers = [
            row["recorded_identifier"] for row in summary["stage_cache_lifecycle"]
        ]
        all_cook_rows = [
            summary["cooks"][candidate][body]
            for candidate in CANDIDATES
            for body in BODIES
        ]
        summary["checks"] = {
            "input_contracts_pass": bool(
                summary["inputs"]
                and summary["inputs"].get("raw_source", {}).get("pass")
                and summary["inputs"].get("current_d348_parts", {}).get("pass")
                and summary["inputs"].get("d339_original_params_exact")
            ),
            "eight_candidate_body_pairs_present": len(all_cook_rows) == 8,
            "all_eight_candidate_body_pairs_pass": bool(
                len(all_cook_rows) == 8 and all(row["pass"] for row in all_cook_rows)
            ),
            "r64_r32_raw_sources_bit_exact_per_body": all(
                summary["cooks"]["R64"][body]["source"]
                == summary["cooks"]["R32"][body]["source"]
                for body in BODIES
            ),
            "r64_r32_only_max_convex_hulls_differs": all(
                {
                    key: value
                    for key, value in summary["cooks"]["R64"][body]["params"].items()
                    if key != "max_convex_hulls"
                }
                == {
                    key: value
                    for key, value in summary["cooks"]["R32"][body]["params"].items()
                    if key != "max_convex_hulls"
                }
                and summary["cooks"]["R64"][body]["params"]["max_convex_hulls"] == 64
                and summary["cooks"]["R32"][body]["params"]["max_convex_hulls"] == 32
                for body in BODIES
            ),
            "sixteen_independent_stage_guards_retained": len(retained_stages) == 16,
            "sixteen_registered_cook_requests_performed": (
                summary["controlled_physx_cook_requests"] == 16
            ),
            "sixteen_in_memory_cook_stages_constructed": (
                summary["controlled_in_memory_cook_stages"] == 16
            ),
            "sixteen_unique_valid_stage_ids": bool(
                len(stage_ids) == 16
                and all(value >= 0 for value in stage_ids)
                and len(set(stage_ids)) == 16
            ),
            "sixteen_unique_stage_identifiers": bool(
                len(stage_identifiers) == 16
                and len(set(stage_identifiers)) == 16
            ),
            "all_stage_cache_mappings_live_before_close": bool(
                len(summary["stage_cache_lifecycle"]) == 16
                and all(row["pass"] for row in summary["stage_cache_lifecycle"])
            ),
            "eight_parameter_scopes_applied_and_restored": bool(
                len(summary["parameter_guards"]) == 8
                and all(row["pass"] for row in summary["parameter_guards"])
            ),
            "d339_global_params_restored": dict(d339.DECOMPOSITION_PARAMS)
            == original_d339_params,
            "no_worker_exception": summary["exception"] is None,
            "controlled_scope_counters_zero": all(
                summary[key] == 0
                for key in (
                    "controlled_simulation_context_constructions",
                    "controlled_resets",
                    "controlled_environment_resets",
                    "controlled_physics_steps",
                    "controlled_timeline_requests",
                    "controlled_q5_samples",
                    "controlled_contact_queries",
                    "controlled_live_contact_queries",
                    "controlled_cylinder_pose_writes",
                    "controlled_target_ik_path_changes",
                    "controlled_material_mass_actuator_physics_changes",
                    "controlled_usd_asset_writes",
                    "controlled_canonical_or_live_asset_writes",
                )
            ),
        }
        summary["pass"] = all(summary["checks"].values())
        summary_path = out_dir / SUMMARY_NAME
        _write_json_exclusive(summary_path, summary)
        try:
            app_running = bool(app.is_running())
            app_running_error = None
        except Exception as error:  # sentinel still proves ordering if the query is unavailable
            app_running = None
            app_running_error = f"{type(error).__name__}: {error}"
        sentinel = {
            "artifact": "D371_PRECLOSE_SENTINEL_V1",
            "case": "g0a_d371",
            "worker_invocation_count": 1,
            "summary_path": _repo_rel(summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "summary_pass": bool(summary["pass"]),
            "app_close_called_before_sentinel": False,
            "app_running_before_close": app_running,
            "app_running_query_error": (
                _repo_sanitized(app_running_error) if app_running_error else None
            ),
            "retained_stage_guard_count": len(retained_stages),
            "stage_cache_lifecycle_pass_before_close": bool(
                summary["stage_cache_lifecycle"]
                and all(row["pass"] for row in summary["stage_cache_lifecycle"])
            ),
            "controlled_physics_steps": 0,
            "controlled_q5_samples": 0,
            "controlled_contact_queries": 0,
            "controlled_usd_asset_writes": 0,
            "preclose_sentinel_written": True,
            "pass": bool(
                summary["pass"]
                and app_running is True
                and app_running_error is None
            ),
        }
        _write_json_exclusive(out_dir / PRECLOSE_NAME, sentinel)
    return bool(sentinel["pass"])


def _exclusive_out_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(REPO.resolve())
    except ValueError as error:
        raise RuntimeError("D371 --out-dir must be inside the repository") from error
    if not relative.parts:
        raise RuntimeError("D371 --out-dir cannot be the repository root")
    resolved.mkdir(parents=True, exist_ok=True)
    worker_owned = [
        resolved / CLAIM_NAME,
        resolved / SUMMARY_NAME,
        resolved / PRECLOSE_NAME,
    ]
    worker_owned.extend(resolved.glob("d371_*_cold*_*.json"))
    existing = sorted(_repo_rel(item) for item in worker_owned if item.exists())
    if existing:
        raise RuntimeError(
            "D371 worker-owned evidence already exists; refusing overwrite: "
            + ", ".join(existing)
        )
    return resolved


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False

    try:
        out_dir = _exclusive_out_dir(args.out_dir)
    except Exception as error:
        print(f"D371 forward-only output refusal: {type(error).__name__}: {error}", flush=True)
        return 73

    claim_path = out_dir / CLAIM_NAME
    try:
        _write_json_exclusive(
            claim_path,
            {
                "artifact": "D371_COOK_WORKER_EXCLUSIVE_CLAIM_V1",
                "case": "g0a_d371",
                "pid": os.getpid(),
                "monotonic_ns": time.monotonic_ns(),
                "single_worker_claimed": True,
            },
        )
    except FileExistsError:
        print("D371 worker claim already exists; refusing concurrent/repeated execution", flush=True)
        return 73

    launcher = None
    retained_stages: list[dict[str, Any]] = []
    try:
        launcher = AppLauncher(args)
        passed = _execute(out_dir, launcher.app, retained_stages)
        print(
            json.dumps(
                {
                    "artifact": "D371_COOK_WORKER_EXIT",
                    "summary": _repo_rel(out_dir / SUMMARY_NAME),
                    "preclose_sentinel": _repo_rel(out_dir / PRECLOSE_NAME),
                    "pass": passed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if passed else 1
    except Exception:  # launch/summary-write failures remain visible to the controller log
        traceback.print_exc()
        return 1
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
