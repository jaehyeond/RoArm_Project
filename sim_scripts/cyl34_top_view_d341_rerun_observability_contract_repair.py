#!/usr/bin/env python3
"""D341: reactive Rerun observability completion-contract repair.

This case is read-only with respect to D340.  It rebuilds no collision asset,
runs no Isaac simulation/physics, and changes no scientific gate.  It records
the already-captured D340 geometry in a new, complete RRD and perturbs only a
copy of that new RRD to prove the footer validator can fail.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roarm_rl.rerun_contract import RERUN_CONTRACT_VERSION, sha256_file, validate_rerun_artifact
from roarm_rl.viz_debug import log_rerun


OUT_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d341"
PREREG_PATH = OUT_DIR / "d341_preregistration.json"
D340_DIR = REPO_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d340"
D340_CANDIDATES = D340_DIR / "d340_capture_fixed_point_candidates.json"
D340_SUMMARY = D340_DIR / "d340_capture_summary.json"
D340_POSTRUN = D340_DIR / "d340_capture_postrun_root_cause_audit.json"
D340_RRD = D340_DIR / "d340_capture_trace.rrd"
D340_SESSION = REPO_ROOT / "claudedocs/session_20260713_grasp_g0a_d340_fixed_point_live_authoring_repair.md"
D341_SESSION = REPO_ROOT / "claudedocs/session_20260713_grasp_g0a_d341_rerun_observability_contract_repair.md"
START_HERE = REPO_ROOT / "START_HERE.md"

GOOD_RRD = OUT_DIR / "d341_d340_cook_observability.rrd"
GOOD_RBL = OUT_DIR / "d341_d340_cook_observability.rbl"
SCREENSHOT = OUT_DIR / "d341_d340_cook_observability_rerun_inspection.png"
NEGATIVE_RRD = OUT_DIR / "d341_footer_truncation_negative_control.rrd"
AUTOMATED_SUMMARY = OUT_DIR / "d341_rerun_observability_automated_summary.json"
AUTOMATED_REPORT = OUT_DIR / "d341_rerun_observability_automated_report.md"

EXPECTED_D340_RRD_SHA256 = "8eb3d6130330334b9d6b457468cd4bb59097114c693cb7caa2e33a8f5993fe47"
INCORRECT_D340_SESSION_RRD_SHA256 = "8eb3d613033034b9d6b457468cd4bb59097114c693cb7caa2e33a8f5993fe47"
NEW_VARIABLES = ["rerun_observability_completion_contract"]

SOURCE_COLORS = {
    "source": [135, 135, 135, 70],
    "instance": [35, 120, 255, 85],
    "prototype": [235, 65, 200, 85],
    "candidate": [35, 205, 90, 115],
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": _relative(path),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _inventory_digest(rows: list[dict[str, Any]]) -> str:
    import hashlib

    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _d340_immutability_report(
    before_inventory: list[dict[str, Any]],
    before_digest: str,
    session_hash_before: str,
) -> dict[str, Any]:
    after_inventory = _inventory(D340_DIR)
    after_digest = _inventory_digest(after_inventory)
    session_hash_after = sha256_file(D340_SESSION)
    return {
        "before_digest": before_digest,
        "after_digest": after_digest,
        "file_count_before": len(before_inventory),
        "file_count_after": len(after_inventory),
        "exact_rows_equal": before_inventory == after_inventory,
        "d340_session_sha256_before": session_hash_before,
        "d340_session_sha256_after": session_hash_after,
        "d340_session_unchanged": session_hash_before == session_hash_after,
        "pass": (
            before_inventory == after_inventory
            and before_digest == after_digest
            and session_hash_before == session_hash_after
        ),
    }


def _self_hashes() -> dict[str, str]:
    paths = {
        "d341_harness": Path(__file__).resolve(),
        "viz_debug": REPO_ROOT / "roarm_rl/viz_debug.py",
        "rerun_contract": REPO_ROOT / "roarm_rl/rerun_contract.py",
        "rerun_contract_tests": REPO_ROOT / "tests/test_viz_debug_rerun_contract.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _build_observability_rows(
    candidate_manifest: dict[str, Any],
    postrun: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    meshes: list[dict[str, Any]] = []
    scalars: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    expected_entities: list[str] = []
    postrun_by_key = {
        (row["body"], row["name"]): row
        for row in postrun["per_part"]
    }
    event_idx = 0
    events.append(
        {
            "entity_path": "events/run",
            "text": "D341 read-only posthoc visualization reconstruction from immutable D340 evidence",
            "level": "INFO",
            "sequence": {"event_idx": event_idx, "part_idx": 0},
        }
    )
    event_idx += 1

    for part_idx, row in enumerate(candidate_manifest["parts"]):
        body = str(row["body"])
        name = str(row["name"])
        fixed = row["fixed_point"]
        geometries = {
            "source": fixed["authored_x0"],
            "instance": fixed["channel_consensus"]["instance"]["canonical"],
            "prototype": fixed["channel_consensus"]["prototype"]["canonical"],
            "candidate": fixed["candidate_x1"],
        }
        for source_kind, geometry in geometries.items():
            current_event_idx = event_idx
            entity_path = f"cook/{source_kind}/{body}/parts/{name}"
            meshes.append(
                {
                    "entity_path": entity_path,
                    "vertices_m": geometry["vertices_m"],
                    "triangles": geometry["triangles"],
                    "body": body,
                    "part": name,
                    "source_kind": source_kind,
                    "coordinate_frame": f"{body}_body_local",
                    "geometry_sha256": geometry["geometry_sha256"],
                    "vertex_stream_sha256": geometry["vertex_stream_sha256"],
                    "topology_sha256": geometry["topology_sha256"],
                    "color_rgba": SOURCE_COLORS[source_kind],
                }
            )
            expected_entities.append(entity_path)
            events.append(
                {
                    "entity_path": "events/cook",
                    "text": (
                        f"{body}/{name} {source_kind}: vertices={geometry['vertex_count']} "
                        f"triangles={geometry['triangle_count']} hash={geometry['geometry_sha256']}"
                    ),
                    "level": "INFO",
                    "sequence": {"event_idx": current_event_idx, "part_idx": part_idx},
                }
            )
            scalars.append(
                {
                    "entity_path": f"metrics/{body}/{name}/{source_kind}_vertex_count",
                    "value": float(geometry["vertex_count"]),
                    "sequence": {"event_idx": current_event_idx, "part_idx": part_idx},
                }
            )
            event_idx += 1

        consensus = fixed["channel_consensus"]
        audit_row = postrun_by_key[(body, name)]
        metric_values = {
            "instance_prototype_coordinate_delta_m": consensus["coordinate_max_abs_delta_m"],
            "live_x1_containment_in_x0_m": fixed["live_x1_containment_in_x0_m"],
            "float32_roundtrip_surface_delta_m": fixed["float32_roundtrip_surface_delta_m"],
            "prim_to_body_transform_delta_m": audit_row["transform_delta"],
            "bounds_delta_m": audit_row["bounds_delta_m"],
            "volume_relative_delta": audit_row["volume_relative_delta"],
        }
        for metric_name, value in metric_values.items():
            scalars.append(
                {
                    "entity_path": f"metrics/{body}/{name}/{metric_name}",
                    "value": float(value),
                    "sequence": {"event_idx": event_idx, "part_idx": part_idx},
                }
            )
        gate_path = f"gate/{body}/{name}/authored_hash_matches_d339_manifest"
        scalars.append(
            {
                "entity_path": gate_path,
                "value": float(bool(row["checks"]["authored_hash_matches_d339_manifest"])),
                "sequence": {"event_idx": event_idx, "part_idx": part_idx},
            }
        )
        expected_entities.append(gate_path)
        events.append(
            {
                "entity_path": "events/gate",
                "text": (
                    f"{body}/{name}: D340 authored-stream hash predicate FALSE; "
                    "D340 FAIL verdict retained"
                ),
                "level": "WARN",
                "sequence": {"event_idx": event_idx, "part_idx": part_idx},
            }
        )
        event_idx += 1

    events.append(
        {
            "entity_path": "events/stop",
            "text": (
                "D341 observability repair only: no attempt3, no validation, no collision asset mutation, "
                "no physics, D340 verdict unchanged"
            ),
            "level": "WARN",
            "sequence": {"event_idx": event_idx, "part_idx": len(candidate_manifest["parts"]) - 1},
        }
    )
    expected_entities.extend(["events/run", "events/cook", "events/gate", "events/stop"])
    return meshes, scalars, events, expected_entities


def _exact_rrd_contract(
    meshes: list[dict[str, Any]],
    scalars: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    entities = {
        "metadata/run",
        "coordinate_frames/link5_body_local",
        "coordinate_frames/gripper_link_body_local",
    }
    components: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "coordinate_frames/link5_body_local": [
            "Transform3D:child_frame",
            "Transform3D:parent_frame",
            "Transform3D:quaternion",
            "Transform3D:translation",
        ],
        "coordinate_frames/gripper_link_body_local": [
            "Transform3D:child_frame",
            "Transform3D:parent_frame",
            "Transform3D:quaternion",
            "Transform3D:translation",
        ],
    }
    for row in meshes:
        path = str(row["entity_path"])
        metadata_path = f"metadata/meshes/{path.replace('/', '__')}"
        entities.update({path, metadata_path})
        components[path] = [
            "CoordinateFrame:frame",
            "Mesh3D:albedo_factor",
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        ]
        components[metadata_path] = ["TextDocument:text"]
    for row in scalars:
        path = str(row["entity_path"])
        entities.add(path)
        components[path] = ["Scalars:scalars"]
    for row in events:
        path = str(row["entity_path"])
        entities.add(path)
        components[path] = ["TextLog:level", "TextLog:text"]
    return sorted(entities), components


def _rrd_contract_digest(
    exact_entities: list[str],
    expected_components: dict[str, list[str]],
) -> str:
    import hashlib

    payload = {
        "exact_non_system_entity_paths": exact_entities,
        "exact_timeline_names": ["blueprint", "event_idx", "log_time", "part_idx"],
        "required_components_by_path": expected_components,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# D341 Rerun observability automated report",
        "",
        f"- Automated verdict: `{summary['automated_verdict']}`",
        f"- New variable: `{NEW_VARIABLES[0]}` (measurement-only)",
        f"- Good RRD footer/entity/timeline gate: `{summary['good_artifact']['validation']['pass']}`",
        f"- Geometry entities: `{summary['scientific_subject']['mesh_entities']}`",
        f"- Negative footer perturbation rejected: `{summary['negative_control']['pass']}`",
        f"- D340 inventory unchanged: `{summary['d340_immutability']['pass']}`",
        f"- Physics steps: `{summary['scope_guards']['controlled_physics_steps']}`",
        "",
        "The generated screenshot proves headless renderability only. A separate",
        "human/agent inspection report is required before the D341 completion",
        "contract may be called fully PASS.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if not PREREG_PATH.is_file():
        raise RuntimeError(f"missing preregistration: {PREREG_PATH}")
    unexpected = sorted(path.name for path in OUT_DIR.iterdir() if path != PREREG_PATH)
    if unexpected:
        raise RuntimeError(f"forward-only output folder is not pristine: {unexpected}")

    prereg = _json(PREREG_PATH)
    before_inventory = _inventory(D340_DIR)
    before_digest = _inventory_digest(before_inventory)
    d340_session_hash_before = sha256_file(D340_SESSION)
    runtime_hashes = _self_hashes()
    prereg_checks = {
        "artifact": prereg.get("artifact") == "D341_RERUN_OBSERVABILITY_PREREGISTRATION_V1",
        "new_variables": prereg.get("new_variables") == NEW_VARIABLES,
        "git_head": prereg.get("git_head") == _git_head(),
        "d340_inventory_digest": prereg.get("d340_inventory_digest") == before_digest,
        "d340_session_hash": prereg.get("d340_session_sha256") == d340_session_hash_before,
        "d341_session_hash": prereg.get("d341_session_sha256") == sha256_file(D341_SESSION),
        "start_here_hash": prereg.get("start_here_sha256") == sha256_file(START_HERE),
        "source_hashes": prereg.get("source_hashes") == runtime_hashes,
        "rerun_version": str(rr.__version__) == RERUN_CONTRACT_VERSION == prereg.get("rerun_version"),
        "numpy_pin": str(np.__version__) == "1.26.0",
        "psutil_pin": str(psutil.__version__) == "5.9.8",
        "d340_rrd_hash": sha256_file(D340_RRD) == EXPECTED_D340_RRD_SHA256,
        "attempt3_absent": not (D340_DIR / "collision_asset/attempt3").exists(),
    }
    if not all(prereg_checks.values()):
        raise RuntimeError(f"D341 preregistration gate failed: {prereg_checks}")

    candidate_manifest = _json(D340_CANDIDATES)
    d340_summary = _json(D340_SUMMARY)
    postrun = _json(D340_POSTRUN)
    meshes, scalars, events, expected_entities = _build_observability_rows(candidate_manifest, postrun)
    exact_entities, expected_components = _exact_rrd_contract(meshes, scalars, events)
    if len(meshes) != 52:
        raise RuntimeError(f"expected 52 scientific geometry entities, got {len(meshes)}")
    prereg_checks.update(
        {
            "registered_subject_counts": prereg.get("scientific_subject_counts")
            == {
                "source_parts": len(candidate_manifest["parts"]),
                "geometry_variants_per_part": 4,
                "mesh_entities": len(meshes),
                "scalar_rows": len(scalars),
                "event_rows": len(events),
                "exact_non_system_entities": len(exact_entities),
            },
            "exact_rrd_contract_digest": prereg.get("rrd_contract_sha256")
            == _rrd_contract_digest(exact_entities, expected_components),
        }
    )
    if not all(prereg_checks.values()):
        raise RuntimeError(f"D341 exact RRD preregistration gate failed: {prereg_checks}")

    legacy_validation = validate_rerun_artifact(D340_RRD)
    session_text = D340_SESSION.read_text(encoding="utf-8")
    documentation_correction = {
        "actual_rrd_sha256": sha256_file(D340_RRD),
        "canonical_summary_rrd_sha256": d340_summary["artifact_sha256"]["d340_capture_trace.rrd"],
        "actual_and_summary_match": (
            sha256_file(D340_RRD) == d340_summary["artifact_sha256"]["d340_capture_trace.rrd"]
        ),
        "incorrect_session_hash_literal_present": INCORRECT_D340_SESSION_RRD_SHA256 in session_text,
        "incorrect_session_hash_length": len(INCORRECT_D340_SESSION_RRD_SHA256),
        "actual_hash_length": len(EXPECTED_D340_RRD_SHA256),
        "d340_rrd_default_footer_verify_pass": legacy_validation["pass"],
        "interpretation": (
            "D340 generated a decodable non-empty RRD, but did not finalize a footer and did not "
            "record its 13-part scientific subject. D340's scientific FAIL verdict is unchanged."
        ),
    }

    recording_metadata = {
        "case": "g0a_d341",
        "purpose": "reactive Rerun observability completion-contract repair",
        "git_head": _git_head(),
        "new_variables": NEW_VARIABLES,
        "source_case": "g0a_d340",
        "source_verdict": "D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP",
        "source_candidate_sha256": sha256_file(D340_CANDIDATES),
        "source_summary_sha256": sha256_file(D340_SUMMARY),
        "source_postrun_sha256": sha256_file(D340_POSTRUN),
        "source_rrd_sha256": EXPECTED_D340_RRD_SHA256,
        "coordinate_contract": (
            "authoritative hashes use original arrays; Rerun Mesh3D is a Float32 body-local display copy"
        ),
        "q5_convention": "0=CLOSED; 1.5413rad=OPEN",
        "physics": "forbidden / 0 steps",
        "collision_asset_mutation": "forbidden / none",
    }
    log_status = log_rerun(
        GOOD_RRD,
        coordinate_frames=[
            {
                "frame": "link5_body_local",
                "parent_frame": "tf#/",
                "entity_path": "coordinate_frames/link5_body_local",
            },
            {
                "frame": "gripper_link_body_local",
                "parent_frame": "tf#/",
                "entity_path": "coordinate_frames/gripper_link_body_local",
            },
        ],
        meshes=meshes,
        scalar_trace=scalars,
        events=events,
        recording_metadata=recording_metadata,
        recording_id="g0a_d341_d340_readonly_observability",
        blueprint_path=GOOD_RBL,
        blueprint_mode="collision_gate",
        live_viewer=False,
        app_id="roarm_g0a_collision_gate",
    )
    if not log_status.get("ok", False):
        d340_immutability = _d340_immutability_report(
            before_inventory,
            before_digest,
            d340_session_hash_before,
        )
        failure_summary = {
            "artifact": "D341_RERUN_OBSERVABILITY_AUTOMATED_SUMMARY_V1",
            "automated_verdict": "D341_RERUN_OBSERVABILITY_AUTOMATED_FAIL_STOP",
            "automated_pass": False,
            "completion_contract_pass": False,
            "manual_visual_inspection_pending": False,
            "new_variables": NEW_VARIABLES,
            "preregistration_checks": prereg_checks,
            "documentation_correction": documentation_correction,
            "legacy_d340_rrd_validation": legacy_validation,
            "scientific_subject": {
                "source_parts": len(candidate_manifest["parts"]),
                "geometry_variants_per_part": 4,
                "mesh_entities_registered": len(meshes),
                "logging_completed": False,
            },
            "good_artifact": {"log_status": log_status, "validation": {"attempted": False, "pass": False}},
            "negative_control": {"attempted": False, "pass": False},
            "d340_immutability": d340_immutability,
            "scope_guards": {
                "controlled_physics_steps": 0,
                "simulation_started": False,
                "collision_asset_writes": [],
                "attempt3_absent": not (D340_DIR / "collision_asset/attempt3").exists(),
                "parameters_increased": [],
                "thresholds_relaxed": [],
                "d340_verdict_unchanged": "D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP",
                "g0a_pass": False,
                "ladder_promoted": False,
            },
            "failure_reason": "Rerun recording/finalization/archive contract failed before render validation",
        }
        _write_json(AUTOMATED_SUMMARY, failure_summary)
        _write_text(
            AUTOMATED_REPORT,
            "# D341 Rerun observability automated report\n\n"
            "- Automated verdict: `D341_RERUN_OBSERVABILITY_AUTOMATED_FAIL_STOP`\n"
            f"- D340 immutability after failure: `{d340_immutability['pass']}`\n"
            "- Dependent render and negative-control stages were not run.\n",
        )
        print(json.dumps({
            "automated_verdict": failure_summary["automated_verdict"],
            "d340_immutable": d340_immutability["pass"],
            "logging_error": log_status.get("error"),
        }, sort_keys=True))
        return 2

    validation = validate_rerun_artifact(
        GOOD_RRD,
        expected_entity_paths=expected_entities,
        expected_timeline_names=["event_idx", "part_idx"],
        exact_entity_paths=exact_entities,
        exact_timeline_names=["blueprint", "event_idx", "log_time", "part_idx"],
        expected_entity_components=expected_components,
        blueprint_path=GOOD_RBL,
        screenshot_path=SCREENSHOT,
    )

    good_bytes = GOOD_RRD.read_bytes()
    truncate_bytes = min(4096, max(1, len(good_bytes) // 10))
    NEGATIVE_RRD.write_bytes(good_bytes[:-truncate_bytes])
    negative_validation = validate_rerun_artifact(NEGATIVE_RRD)
    negative_pass = bool(
        not negative_validation.get("pass", False)
        and negative_validation.get("footer_manifest_present") is False
        and "RRD footer verification failed" in negative_validation.get("errors", [])
    )

    d340_immutability = _d340_immutability_report(
        before_inventory,
        before_digest,
        d340_session_hash_before,
    )
    automated_pass = bool(
        all(prereg_checks.values())
        and not legacy_validation.get("pass", True)
        and documentation_correction["actual_and_summary_match"]
        and documentation_correction["incorrect_session_hash_literal_present"]
        and log_status.get("ok", False)
        and validation.get("pass", False)
        and negative_pass
        and d340_immutability["pass"]
        and len(meshes) == 52
        and not (D340_DIR / "collision_asset/attempt3").exists()
    )
    automated_verdict = (
        "D341_RERUN_OBSERVABILITY_AUTOMATED_PASS_MANUAL_INSPECTION_PENDING"
        if automated_pass
        else "D341_RERUN_OBSERVABILITY_AUTOMATED_FAIL_STOP"
    )
    summary = {
        "artifact": "D341_RERUN_OBSERVABILITY_AUTOMATED_SUMMARY_V1",
        "automated_verdict": automated_verdict,
        "automated_pass": automated_pass,
        "completion_contract_pass": False,
        "manual_visual_inspection_pending": True,
        "new_variables": NEW_VARIABLES,
        "preregistration_checks": prereg_checks,
        "versions": {
            "rerun_sdk": str(rr.__version__),
            "numpy": str(np.__version__),
            "psutil": str(psutil.__version__),
        },
        "documentation_correction": documentation_correction,
        "legacy_d340_rrd_validation": legacy_validation,
        "scientific_subject": {
            "source_parts": len(candidate_manifest["parts"]),
            "geometry_variants_per_part": 4,
            "mesh_entities": len(meshes),
            "scalar_rows": len(scalars),
            "event_rows": len(events),
            "expected_geometry_entity_paths": [row["entity_path"] for row in meshes],
        },
        "good_artifact": {
            "rrd": _relative(GOOD_RRD),
            "rbl": _relative(GOOD_RBL),
            "headless_screenshot": _relative(SCREENSHOT),
            "log_status": log_status,
            "validation": validation,
        },
        "negative_control": {
            "kind": "truncate finalized D341 RRD copy",
            "truncated_bytes": truncate_bytes,
            "path": _relative(NEGATIVE_RRD),
            "validation": negative_validation,
            "pass": negative_pass,
        },
        "d340_immutability": d340_immutability,
        "scope_guards": {
            "controlled_physics_steps": 0,
            "simulation_started": False,
            "collision_asset_writes": [],
            "attempt3_absent": not (D340_DIR / "collision_asset/attempt3").exists(),
            "parameters_increased": [],
            "thresholds_relaxed": [],
            "d340_verdict_unchanged": "D340_G0A_FIXED_POINT_CAPTURE_CONTRACT_FAIL_STOP",
            "g0a_pass": False,
            "ladder_promoted": False,
        },
        "next_gate": (
            "Inspect the generated headless screenshot and write a separate observation report. "
            "Only then may D341 completion_contract_pass become true."
        ),
    }
    _write_json(AUTOMATED_SUMMARY, summary)
    _write_text(AUTOMATED_REPORT, _markdown(summary))
    print(json.dumps({
        "automated_verdict": automated_verdict,
        "rrd": _relative(GOOD_RRD),
        "screenshot": _relative(SCREENSHOT),
        "negative_control_pass": negative_pass,
        "d340_immutable": d340_immutability["pass"],
    }, sort_keys=True))
    return 0 if automated_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
