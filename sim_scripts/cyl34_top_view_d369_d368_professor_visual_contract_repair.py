#!/usr/bin/env python3
"""D369: repair D368's professor-facing visual contract without science/runtime work.

The only frozen domain inputs are the immutable D368 evidence JSON and RRD.  The JSON supplies
already committed facts; the RRD supplies already logged display geometry.  No collider geometry
is decoded or regenerated here, and no Isaac/Kit/PhysX module is imported.
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve()
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d369"
SOURCE_EVIDENCE = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_evidence.json"
)
SOURCE_RRD = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation.rrd"
RERUN_CONTRACT_HELPER = REPO / "roarm_rl/rerun_contract.py"

PREREG_PATH = OUT_DIR / "d369_preregistration.json"
PHASE_PATH = OUT_DIR / "d369_phase_markers.jsonl"
BASE_COPY_PATH = OUT_DIR / "d369_d368_base_bitexact_copy.rrd"
RECORDING_ONLY_PATH = OUT_DIR / "d369_d368_recording_only_display_copy.rrd"
OVERLAY_RRD_PATH = OUT_DIR / "d369_static_text_overlay.rrd"
RBL_PATH = OUT_DIR / "d369_professor_visual_contract.rbl"
PRESENTATION_RRD_PATH = OUT_DIR / "d369_professor_visual_contract.rrd"
INVOCATION_PATH = OUT_DIR / "d369_render_invocation.json"
RECEIPT_PATH = OUT_DIR / "d369_render_receipt.json"
VALIDATION_PATH = OUT_DIR / "d369_rerun_validation.json"
RERUN_PNG_PATH = OUT_DIR / "d369_professor_visual_contract_rerun.png"
SUMMARY_PNG_PATH = OUT_DIR / "d369_professor_board_1920x1080.png"
AUTOMATED_PATH = OUT_DIR / "d369_automated_summary.json"
REPORT_PATH = OUT_DIR / "d369_automated_report.md"
MANUAL_JSON_PATH = OUT_DIR / "d369_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d369_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d369_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d369_runtime_exception.json"

EXPECTED_HEAD = "7c4819632bb193c8fd552372c919f8a107675b41"
EXPECTED_EVIDENCE_SHA256 = "be2a422b0c74e4781b76a640c5312070b84876b1cb9e661d47e705ccdf789cf5"
EXPECTED_EVIDENCE_BYTES = 953696
EXPECTED_RRD_SHA256 = "f66a9fe41c625e3460b341eef2bfb0e107fbccdca4bf012c28b77e694efb5af0"
EXPECTED_RRD_BYTES = 1339534
EXPECTED_MEASUREMENT_VERDICT = "D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_MEASURED_NO_PHYSICS"
EXPECTED_RERUN_VERSION = "0.34.1"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
APP_ID = "roarm_g0a_d368_semantic_allocation"
RECORDING_ID = "g0a_d368_current_64cap_semantic_allocation"
WINDOW_SIZE = "1920x1080"
RENDER_TIMEOUT_SECONDS = 180.0
HOST_RENDER_ENV = "D369_HOST_RENDER_APPROVED"

NEW_VARIABLES = [
    "timeline_free_static_metric_overlay",
    "label_suppressed_professor_layout",
]

SCOPE_GUARDS = {
    "collider_regeneration_or_recook": 0,
    "isaac_or_kit_or_physx": 0,
    "simulation_app": 0,
    "q5_science_or_target": 0,
    "physics_steps": 0,
    "contact_queries": 0,
    "target_ik_path_changes": 0,
    "usd_or_asset_writes": 0,
    "material_mass_actuator_physics_changes": 0,
    "warp_or_cuda_compute": 0,
    "nvidia_smi": 0,
    "rerun_display_render_allowed": 1,
}

FORBIDDEN_MODULE_PREFIXES = (
    "omni",
    "isaacsim",
    "isaaclab",
    "pxr",
    "warp",
    "torch",
    "trimesh",
    "scipy",
)

ALLOWED_DIRTY_PREFIXES = (
    "AGENTS.md",
    "START_HERE.md",
    "claudedocs/DECISIONS.md",
    "claudedocs/EXPERIMENT_LEDGER.md",
    "claudedocs/runtime_logs/grasp_track/g0a_d368/",
    "claudedocs/runtime_logs/grasp_track/g0a_d369/",
    "claudedocs/session_20260720_grasp_g0a_d368_",
    "claudedocs/session_20260720_grasp_g0a_d369_",
    "roarm_rl/viz_debug.py",
    "sim_scripts/cyl34_top_view_d368_",
    "sim_scripts/cyl34_top_view_d369_",
)

MANUAL_CHECK_KEYS = [
    "opened_both_pngs_original_resolution",
    "rerun_four_spatial_views_nonblank",
    "timeline_free_static_metric_cards_legible",
    "no_unknown_timeline_or_empty_metric_panel",
    "no_in_scene_label_overlap",
    "no_error_notification_or_operation_not_permitted_toast",
    "summary_exact_1920x1080",
    "summary_text_legible_no_overlap",
    "colors_and_counts_consistent",
    "null_and_scope_boundary_visible",
    "display_only_authority_visible",
]

EXPECTED_PHASE_SEQUENCE = [
    "render_started",
    "host_loopback_bind_capability_pass",
    "frozen_evidence_fields_copied",
    "d368_rrd_bitexact_copy_complete",
    "static_text_overlay_and_blueprint_finalized",
    "d368_recording_only_display_copy_finalized",
    "single_presentation_archive_finalized",
    "pre_render_artifact_contract_pass",
    "one_shot_headless_render_invoked",
    "one_shot_headless_render_returned",
    "professor_board_written",
    "automated_artifacts_finalized",
]

EXPECTED_NEGATIVE_CHECK_KEYS = {
    "baseline_accepts",
    "evidence_hash_tamper_rejects",
    "rrd_hash_tamper_rejects",
    "g0a_boundary_flip_rejects",
    "null_boundary_flip_rejects",
    "dataframe_view_substitution_rejects",
    "label_path_inclusion_rejects",
    "second_render_rejects",
    "compressed_fact_card_rejects",
    "bbox_overlap_injection_rejects",
}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _read_phase_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with PHASE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("D369 phase row must be a JSON object")
                rows.append(value)
    return rows


def _write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_text_x(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _copy_file_x(source: Path, destination: Path) -> None:
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def _run(command: list[str], *, timeout: float = 60.0) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_seconds": time.monotonic() - started,
        "ok": completed.returncode == 0,
    }


def _loopback_bind_probe() -> dict[str, Any]:
    """Prove the actual worker can bind a local port before consuming the one render."""
    started = time.monotonic()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        handle.listen(1)
        address, port = handle.getsockname()
    return {
        "attempt_count": 1,
        "bind_succeeded": True,
        "listen_succeeded": True,
        "address": address,
        "ephemeral_port": port,
        "elapsed_seconds": time.monotonic() - started,
        "pass": address == "127.0.0.1" and isinstance(port, int) and port > 0,
        "purpose": (
            "non-render host bind+listen capability check before the one Rerun Viewer render "
            "invocation"
        ),
    }


def _git(*args: str) -> str:
    result = _run(["git", *args])
    if not result["ok"]:
        raise RuntimeError(f"git {' '.join(args)} failed: {result['stderr']}")
    return str(result["stdout"]).strip()


def _git_status_paths() -> list[str]:
    rows = []
    result = _run(["git", "status", "--short"])
    if not result["ok"]:
        raise RuntimeError(f"git status --short failed: {result['stderr']}")
    for line in str(result["stdout"]).splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        rows.append(path)
    return rows


def _dirty_scope_pass(paths: list[str]) -> bool:
    return all(any(path == prefix or path.startswith(prefix) for prefix in ALLOWED_DIRTY_PREFIXES) for path in paths)


def _forbidden_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MODULE_PREFIXES)
    )


def _phase(name: str, **fields: Any) -> None:
    row = {"phase": name, "monotonic_seconds": time.monotonic(), **fields}
    with PHASE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _source_manifest() -> dict[str, Any]:
    return {
        _rel(SOURCE_EVIDENCE): {
            "bytes": SOURCE_EVIDENCE.stat().st_size,
            "sha256": _sha(SOURCE_EVIDENCE),
        },
        _rel(SOURCE_RRD): {
            "bytes": SOURCE_RRD.stat().st_size,
            "sha256": _sha(SOURCE_RRD),
        },
    }


def _source_manifest_exact(manifest: dict[str, Any]) -> bool:
    return manifest == {
        _rel(SOURCE_EVIDENCE): {
            "bytes": EXPECTED_EVIDENCE_BYTES,
            "sha256": EXPECTED_EVIDENCE_SHA256,
        },
        _rel(SOURCE_RRD): {
            "bytes": EXPECTED_RRD_BYTES,
            "sha256": EXPECTED_RRD_SHA256,
        },
    }


def _blueprint_activation_ids(text: str) -> list[str]:
    return [row["blueprint_id"] for row in _blueprint_activation_commands(text)]


def _blueprint_activation_commands(text: str) -> list[dict[str, Any]]:
    rows = re.findall(
        r'BlueprintActivationCommand\(StoreId\(Blueprint,\s*"([^"]+)",\s*"([^"]+)"\),'
        r"\s*make_active:\s*(true|false),\s*make_default:\s*(true|false)\)",
        text,
        flags=re.DOTALL,
    )
    return [
        {
            "application_id": application_id,
            "blueprint_id": blueprint_id,
            "make_active": make_active == "true",
            "make_default": make_default == "true",
        }
        for application_id, blueprint_id, make_active, make_default in rows
    ]


def _strict_index_names(stats_text: str) -> set[str]:
    match = re.search(
        r"Num chunks per index\n-+\n(?P<body>.*?)(?=\nNum chunks per |\nSize \(|\Z)",
        stats_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("Rerun stats did not contain a Num chunks per index section")
    return {
        line.split(":", 1)[0].strip()
        for line in match.group("body").splitlines()
        if ":" in line
    }


def _extract_facts(evidence: dict[str, Any]) -> dict[str, Any]:
    certified = evidence["certified_callback_surfaces"]
    classes = evidence["association_classification_counts"]
    distinction = evidence["nvidia_official_source_contract"]["distinction"]
    boundary = evidence["interpretation_boundary"]
    return {
        "case": evidence["case"],
        "artifact": evidence["artifact"],
        "measurement_pass": evidence["measurement_pass"],
        "measurement_verdict": evidence["verdict"],
        "part_counts": evidence["callback_inventory"]["part_counts"],
        "total_parts": evidence["callback_inventory"]["total_parts"],
        "fixed": {
            "face_count": certified["link5_fixed"]["face_count"],
            "part_count": certified["link5_fixed"]["part_count"],
        },
        "moving_inner": {
            "face_count": certified["gripper_inner"]["face_count"],
            "part_count": certified["gripper_inner"]["part_count"],
        },
        "moving_outer": {
            "face_count": certified["gripper_outer"]["face_count"],
            "part_count": certified["gripper_outer"]["part_count"],
        },
        "classification_counts": classes,
        "negative_controls": {
            "passed": evidence["negative_controls"]["passed"],
            "total": evidence["negative_controls"]["total"],
            "pass": evidence["negative_controls"]["pass"],
        },
        "max_convex_hulls": {
            "project_input": distinction["project_authored_candidate"]["maxConvexHulls"],
            "schema_default": distinction["schema_default"]["maxConvexHulls"],
            "ui_range": distinction["ui_authoring_range"]["maxConvexHulls"],
            "optimality_claim": distinction["optimality_claim"],
        },
        "hull_vertex_limit": {
            "project_input": distinction["project_authored_candidate"]["hullVertexLimit"],
            "schema_default": distinction["schema_default"]["hullVertexLimit"],
            "ui_range": distinction["ui_authoring_range"]["hullVertexLimit"],
        },
        "installed_schema_extension": evidence["nvidia_official_source_contract"]
        ["installed_physx_schema_extension"],
        "installed_physx_sdk_semver": evidence["nvidia_official_source_contract"]
        ["installed_physx_sdk_semver"],
        "interpretation_boundary": {
            "current_64cap_optimal": boundary["current_64cap_optimal"],
            "physics_equivalence": boundary["physics_equivalence"],
            "collider_count_tipping_causality": boundary["collider_count_tipping_causality"],
            "actual_gpu_contact_execution": boundary["actual_gpu_contact_execution"],
            "grasp_feasibility": boundary["grasp_feasibility"],
            "g0a_pass": boundary["g0a_pass"],
        },
        "scope_guards": evidence["scope_guards"],
    }


def _facts_exact(facts: dict[str, Any]) -> dict[str, bool]:
    boundary = facts["interpretation_boundary"]
    classes = facts["classification_counts"]
    return {
        "case_exact": facts["case"] == "g0a_d368",
        "artifact_exact": facts["artifact"]
        == "D368_CURRENT_64CAP_SEMANTIC_ALLOCATION_EVIDENCE_V1",
        "measurement_pass": facts["measurement_pass"] is True,
        "measurement_verdict_exact": facts["measurement_verdict"] == EXPECTED_MEASUREMENT_VERDICT,
        "part_counts_64_plus_64": facts["part_counts"] == {"gripper_link": 64, "link5": 64}
        and facts["total_parts"] == 128,
        "fixed_12_faces_4_parts": facts["fixed"]["face_count"] == 12
        and facts["fixed"]["part_count"] == 4,
        "moving_inner_40_faces_17_parts": facts["moving_inner"]["face_count"] == 40
        and facts["moving_inner"]["part_count"] == 17,
        "moving_outer_36_faces_16_parts": facts["moving_outer"]["face_count"] == 36
        and facts["moving_outer"]["part_count"] == 16,
        "classification_exact": classes
        == {
            "gripper_link": {
                "certified:inner_contact_patch": 1,
                "mixed_certified:inner_contact_patch+outer_negative_patch": 16,
                "no_certified_contact_face": 47,
            },
            "link5": {
                "certified:seed_contact_plane_patch": 4,
                "no_certified_contact_face": 60,
            },
        },
        "moving_overlap_relationship_exact": facts["moving_outer"]["part_count"]
        == classes["gripper_link"]["mixed_certified:inner_contact_patch+outer_negative_patch"]
        and facts["moving_inner"]["part_count"]
        == classes["gripper_link"]["mixed_certified:inner_contact_patch+outer_negative_patch"]
        + classes["gripper_link"]["certified:inner_contact_patch"]
        and sum(classes["gripper_link"].values()) == facts["part_counts"]["gripper_link"],
        "negative_controls_8_of_8": facts["negative_controls"]
        == {"passed": 8, "total": 8, "pass": True},
        "max_convex_hulls_distinction_exact": facts["max_convex_hulls"]
        == {
            "project_input": 64,
            "schema_default": 32,
            "ui_range": [1, 2048],
            "optimality_claim": None,
        },
        "hull_vertex_limit_distinction_exact": facts["hull_vertex_limit"]
        == {"project_input": 64, "schema_default": 64, "ui_range": [8, 64]},
        "installed_schema_extension_exact": facts["installed_schema_extension"]
        == "107.3.26+107.3.3",
        "installed_physx_sdk_semver_null": facts["installed_physx_sdk_semver"] is None,
        "science_boundary_null_and_false": all(
            boundary[key] is None
            for key in (
                "current_64cap_optimal",
                "physics_equivalence",
                "collider_count_tipping_causality",
                "actual_gpu_contact_execution",
                "grasp_feasibility",
            )
        )
        and boundary["g0a_pass"] is False,
        "d368_scope_runtime_zero": all(
            value == 0
            for key, value in facts["scope_guards"].items()
            if key != "rerun_display_render_allowed"
        )
        and facts["scope_guards"].get("rerun_display_render_allowed") == 1,
    }


def _font_paths() -> dict[str, str]:
    return {
        "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    }


def _professor_rows(facts: dict[str, Any]) -> list[tuple[str, str]]:
    classes = facts["classification_counts"]
    part_counts = facts["part_counts"]
    max_hulls = facts["max_convex_hulls"]
    vertex_limit = facts["hull_vertex_limit"]
    boundary = facts["interpretation_boundary"]
    negative = facts["negative_controls"]
    null_text = lambda value: "NULL" if value is None else str(value)
    return [
        ("section", f"CURRENT {max_hulls['project_input']}-CAP REFERENCE"),
        ("body", f"Project maxConvexHulls input: {max_hulls['project_input']}"),
        (
            "body",
            f"Observed callback output: link5 {part_counts['link5']} + "
            f"moving jaw {part_counts['gripper_link']}",
        ),
        (
            "body",
            f"Schema default: {max_hulls['schema_default']} | UI authoring range: "
            f"{max_hulls['ui_range'][0]}..{max_hulls['ui_range'][1]}",
        ),
        (
            "body",
            f"Separate hullVertexLimit: {vertex_limit['project_input']} "
            f"(schema {vertex_limit['schema_default']} | UI "
            f"{vertex_limit['ui_range'][0]}..{vertex_limit['ui_range'][1]})",
        ),
        ("body", f"Installed schema extension: {facts['installed_schema_extension']}"),
        ("section", "CERTIFIED SEMANTIC-PATCH CARRIERS"),
        (
            "body",
            f"link5 fixed-jaw seed patch: {facts['fixed']['part_count']}/"
            f"{part_counts['link5']} parts ({facts['fixed']['face_count']} faces)",
        ),
        (
            "body",
            f"moving inner contact patch: {facts['moving_inner']['part_count']}/"
            f"{part_counts['gripper_link']} parts "
            f"({facts['moving_inner']['face_count']} faces)",
        ),
        (
            "body",
            f"moving outer negative patch: {facts['moving_outer']['part_count']}/"
            f"{part_counts['gripper_link']} parts "
            f"({facts['moving_outer']['face_count']} faces; all are dual)",
        ),
        (
            "body",
            "Do not add inner+outer: outer "
            f"{classes['gripper_link']['mixed_certified:inner_contact_patch+outer_negative_patch']} "
            "are already in inner ("
            f"{classes['gripper_link']['mixed_certified:inner_contact_patch+outer_negative_patch']} "
            "dual + "
            f"{classes['gripper_link']['certified:inner_contact_patch']} inner-only = "
            f"{facts['moving_inner']['part_count']} unique); no certified patch face: "
            f"{classes['gripper_link']['no_certified_contact_face']}",
        ),
        ("section", "MEASUREMENT BOUNDARY"),
        (
            "body",
            f"Frozen D368 controls: {negative['passed']}/{negative['total']} "
            f"{'PASS' if negative['pass'] else 'FAIL'}",
        ),
        (
            "body",
            f"{max_hulls['project_input']} optimal: "
            f"{null_text(boundary['current_64cap_optimal'])} | physics equivalence: "
            f"{null_text(boundary['physics_equivalence'])}",
        ),
        (
            "body",
            f"Toppling cause: {null_text(boundary['collider_count_tipping_causality'])} | "
            f"grasp feasibility: {null_text(boundary['grasp_feasibility'])}",
        ),
        (
            "body",
            "Actual GPU contact execution: "
            f"{null_text(boundary['actual_gpu_contact_execution'])} (not instrumented)",
        ),
        ("body", f"G0a: {str(boundary['g0a_pass']).lower()}"),
        (
            "body",
            "D369 runtime: Isaac / PhysX / q5 / steps = "
            f"{SCOPE_GUARDS['isaac_or_kit_or_physx']} / "
            f"{SCOPE_GUARDS['isaac_or_kit_or_physx']} / "
            f"{SCOPE_GUARDS['q5_science_or_target']} / {SCOPE_GUARDS['physics_steps']}",
        ),
    ]


def _wrap_text(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _bbox_intersections(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intersections: list[dict[str, Any]] = []
    for i, first in enumerate(items):
        for j in range(i + 1, len(items)):
            second = items[j]
            x0 = max(first["bbox"][0], second["bbox"][0])
            y0 = max(first["bbox"][1], second["bbox"][1])
            x1 = min(first["bbox"][2], second["bbox"][2])
            y1 = min(first["bbox"][3], second["bbox"][3])
            if x1 > x0 and y1 > y0:
                intersections.append({"i": i, "j": j, "area_px2": (x1 - x0) * (y1 - y0)})
    return intersections


def _plan_fact_card(
    facts: dict[str, Any], *, width: int, height: int, force_overlap: bool = False
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    fonts = _font_paths()
    section_font = ImageFont.truetype(fonts["bold"], 20)
    body_font = ImageFont.truetype(fonts["regular"], 17)
    scratch = Image.new("RGB", (max(width, 1), max(height, 1)), "white")
    draw = ImageDraw.Draw(scratch)
    x = 18
    y = 18
    max_width = width - 36
    items: list[dict[str, Any]] = []
    for style, text in _professor_rows(facts):
        font = section_font if style == "section" else body_font
        if style == "section" and items:
            y += 10
        for line in _wrap_text(draw, text, font, max_width):
            bbox = draw.textbbox((x, y), line, font=font)
            items.append({"style": style, "text": line, "bbox": list(map(int, bbox))})
            y = int(bbox[3]) + (7 if style == "section" else 5)
    if force_overlap and len(items) >= 2:
        first = items[0]["bbox"]
        second = items[1]["bbox"]
        height_second = second[3] - second[1]
        second[1] = first[1]
        second[3] = first[1] + height_second
    intersections = _bbox_intersections(items)
    within = all(
        row["bbox"][0] >= 12
        and row["bbox"][1] >= 12
        and row["bbox"][2] <= width - 12
        and row["bbox"][3] <= height - 12
        for row in items
    )
    return {
        "width": width,
        "height": height,
        "items": items,
        "text_bbox_within_zone": within,
        "text_intersection_count": len(intersections),
        "intersections": intersections,
        "y_end": y,
        "pass": within and not intersections and y <= height - 12,
    }


def _view_contract(*, view_kind: str, includes_label_paths: bool) -> bool:
    return view_kind == "TextDocumentView" and not includes_label_paths


def _candidate_contract(
    *,
    evidence_sha: str,
    rrd_sha: str,
    facts: dict[str, Any],
    view_kind: str,
    includes_label_paths: bool,
    render_count: int,
    card_width: int,
    force_overlap: bool,
) -> dict[str, Any]:
    layout = _plan_fact_card(facts, width=card_width, height=790, force_overlap=force_overlap)
    checks = {
        "evidence_hash_exact": evidence_sha == EXPECTED_EVIDENCE_SHA256,
        "rrd_hash_exact": rrd_sha == EXPECTED_RRD_SHA256,
        "facts_exact": all(_facts_exact(facts).values()),
        "timeline_free_and_label_suppressed": _view_contract(
            view_kind=view_kind, includes_label_paths=includes_label_paths
        ),
        "one_render_no_retry": render_count == 1,
        "fact_card_layout": layout["pass"],
    }
    return {"checks": checks, "layout": layout, "pass": all(checks.values())}


def _negative_controls(facts: dict[str, Any]) -> dict[str, Any]:
    baseline_args = {
        "evidence_sha": EXPECTED_EVIDENCE_SHA256,
        "rrd_sha": EXPECTED_RRD_SHA256,
        "facts": facts,
        "view_kind": "TextDocumentView",
        "includes_label_paths": False,
        "render_count": 1,
        "card_width": 540,
        "force_overlap": False,
    }
    baseline = _candidate_contract(**baseline_args)
    g0a_tampered = copy.deepcopy(facts)
    g0a_tampered["interpretation_boundary"]["g0a_pass"] = True
    null_tampered = copy.deepcopy(facts)
    null_tampered["interpretation_boundary"]["current_64cap_optimal"] = True
    rows = {
        "evidence_hash_tamper": _candidate_contract(
            **{**baseline_args, "evidence_sha": "0" * 64}
        ),
        "rrd_hash_tamper": _candidate_contract(**{**baseline_args, "rrd_sha": "f" * 64}),
        "g0a_boundary_flip": _candidate_contract(**{**baseline_args, "facts": g0a_tampered}),
        "null_boundary_flip": _candidate_contract(**{**baseline_args, "facts": null_tampered}),
        "dataframe_view_substitution": _candidate_contract(
            **{**baseline_args, "view_kind": "DataframeView"}
        ),
        "label_path_inclusion": _candidate_contract(
            **{**baseline_args, "includes_label_paths": True}
        ),
        "second_render": _candidate_contract(**{**baseline_args, "render_count": 2}),
        "compressed_fact_card": _candidate_contract(**{**baseline_args, "card_width": 250}),
        "bbox_overlap_injection": _candidate_contract(
            **{**baseline_args, "force_overlap": True}
        ),
    }
    def only_check_fails(row: dict[str, Any], key: str) -> bool:
        return (
            row["pass"] is False
            and row["checks"].get(key) is False
            and all(value is True for name, value in row["checks"].items() if name != key)
        )

    checks = {
        "baseline_accepts": baseline["pass"] is True and all(baseline["checks"].values()),
        "evidence_hash_tamper_rejects": only_check_fails(
            rows["evidence_hash_tamper"], "evidence_hash_exact"
        ),
        "rrd_hash_tamper_rejects": only_check_fails(
            rows["rrd_hash_tamper"], "rrd_hash_exact"
        ),
        "g0a_boundary_flip_rejects": only_check_fails(
            rows["g0a_boundary_flip"], "facts_exact"
        )
        and g0a_tampered["interpretation_boundary"]["g0a_pass"] is True,
        "null_boundary_flip_rejects": only_check_fails(
            rows["null_boundary_flip"], "facts_exact"
        )
        and null_tampered["interpretation_boundary"]["current_64cap_optimal"] is True,
        "dataframe_view_substitution_rejects": only_check_fails(
            rows["dataframe_view_substitution"], "timeline_free_and_label_suppressed"
        ),
        "label_path_inclusion_rejects": only_check_fails(
            rows["label_path_inclusion"], "timeline_free_and_label_suppressed"
        ),
        "second_render_rejects": only_check_fails(
            rows["second_render"], "one_render_no_retry"
        ),
        "compressed_fact_card_rejects": only_check_fails(
            rows["compressed_fact_card"], "fact_card_layout"
        )
        and rows["compressed_fact_card"]["layout"]["text_bbox_within_zone"] is False
        and rows["compressed_fact_card"]["layout"]["text_intersection_count"] == 0,
        "bbox_overlap_injection_rejects": only_check_fails(
            rows["bbox_overlap_injection"], "fact_card_layout"
        )
        and rows["bbox_overlap_injection"]["layout"]["text_bbox_within_zone"] is True
        and rows["bbox_overlap_injection"]["layout"]["text_intersection_count"] > 0,
    }
    return {
        "baseline": baseline,
        "perturbations": rows,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "pass": all(checks.values()),
    }


def _blueprint_spec() -> dict[str, Any]:
    return {
        "spatial_view_count": 4,
        "text_document_view_count": 2,
        "dataframe_view_count": 0,
        "time_series_view_count": 0,
        "label_entity_paths_included": 0,
        "left_geometry_share": 0.72,
        "right_fact_share": 0.28,
        "source_geometry": (
            "RrdReader recording-store projection, unordered-equivalent to bit-exact D368 RRD copy"
        ),
        "metric_source": "static TextDocument overlay copied from frozen D368 JSON fields",
    }


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    def spatial(name: str, contents: list[str], position: tuple[float, float, float], target: tuple[float, float, float]) -> Any:
        return rrb.Spatial3DView(
            origin="/",
            contents=contents,
            name=name,
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=position,
                look_target=target,
                eye_up=(0.0, 0.0, 1.0),
            ),
            spatial_information=rrb.SpatialInformation(
                target_frame="tf#/", show_axes=True, show_bounding_box=False
            ),
        )

    link5_full = ["/semantic/source/link5/**", "/semantic/collider/link5/**"]
    link5_zoom = [
        "/semantic/source/link5/seed_contact_plane_patch",
        "/semantic/collider/link5/certified_seed_patch_carrier/**",
    ]
    moving_full = ["/semantic/source/gripper_link/**", "/semantic/collider/gripper_link/**"]
    moving_zoom = [
        "/semantic/source/gripper_link/inner_contact_patch",
        "/semantic/source/gripper_link/outer_negative_patch",
        "/semantic/collider/gripper_link/certified_inner_patch_carrier/**",
        "/semantic/collider/gripper_link/dual_inner_outer_patch_carrier/**",
        "/semantic/collider/gripper_link/outer_negative_patch_carrier/**",
    ]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(
                    spatial("link5 | all 64", link5_full, (0.18, -0.22, 0.16), (-0.005, 0.0, 0.065)),
                    spatial("link5 | fixed patch", link5_zoom, (0.08, -0.10, 0.14), (-0.010, 0.0, 0.100)),
                    column_shares=[0.5, 0.5],
                ),
                rrb.Horizontal(
                    spatial("moving jaw | all 64", moving_full, (0.15, -0.18, 0.08), (0.030, 0.0, -0.018)),
                    spatial("moving jaw | contact patches", moving_zoom, (0.10, -0.10, 0.04), (0.045, -0.006, -0.020)),
                    column_shares=[0.5, 0.5],
                ),
                row_shares=[0.5, 0.5],
            ),
            rrb.Vertical(
                rrb.TextDocumentView(
                    origin="/presentation/d369/allocation",
                    contents="/presentation/d369/allocation",
                    name="Frozen D368 allocation",
                ),
                rrb.TextDocumentView(
                    origin="/presentation/d369/scope_and_64_basis",
                    contents="/presentation/d369/scope_and_64_basis",
                    name="What 64 means / does not mean",
                ),
                row_shares=[0.47, 0.53],
            ),
            column_shares=[0.72, 0.28],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _overlay_text(facts: dict[str, Any]) -> tuple[str, str]:
    classes = facts["classification_counts"]["gripper_link"]
    part_counts = facts["part_counts"]
    max_hulls = facts["max_convex_hulls"]
    vertex_limit = facts["hull_vertex_limit"]
    negative = facts["negative_controls"]
    boundary = facts["interpretation_boundary"]
    null_text = lambda value: "NULL" if value is None else str(value)
    allocation = "\n".join(
        [
            "# Frozen D368 counts",
            "",
            f"Current output: **link5 {part_counts['link5']} + "
            f"moving jaw {part_counts['gripper_link']}**",
            "",
            f"- link5 fixed-jaw seed patch: **{facts['fixed']['part_count']}/"
            f"{part_counts['link5']} parts** ({facts['fixed']['face_count']} faces)",
            f"- moving inner contact patch: **{facts['moving_inner']['part_count']}/"
            f"{part_counts['gripper_link']} parts** ({facts['moving_inner']['face_count']} faces)",
            f"- moving outer negative patch: **{facts['moving_outer']['part_count']}/"
            f"{part_counts['gripper_link']} parts** ({facts['moving_outer']['face_count']} faces)",
            "",
            "Do not add inner+outer:",
            f"**all {classes['mixed_certified:inner_contact_patch+outer_negative_patch']} outer "
            f"carriers are already in the {facts['moving_inner']['part_count']} inner carriers**",
            f"({classes['mixed_certified:inner_contact_patch+outer_negative_patch']} dual + "
            f"{classes['certified:inner_contact_patch']} inner-only = "
            f"{facts['moving_inner']['part_count']} unique; "
            f"no certified patch face {classes['no_certified_contact_face']})",
            "",
            f"Negative controls: **{negative['passed']}/{negative['total']} "
            f"{'PASS' if negative['pass'] else 'FAIL'}**",
        ]
    )
    scope = "\n".join(
        [
            "# What 64 means",
            "",
            f"- Project `maxConvexHulls`: **{max_hulls['project_input']}**",
            f"- Observed callback output: **{part_counts['link5']} + "
            f"{part_counts['gripper_link']}**",
            f"- Installed schema default: **{max_hulls['schema_default']}**",
            f"- UI authoring range: **{max_hulls['ui_range'][0]}..{max_hulls['ui_range'][1]}**",
            f"- Separate `hullVertexLimit`: **{vertex_limit['project_input']}** "
            f"(schema {vertex_limit['schema_default']}; UI "
            f"{vertex_limit['ui_range'][0]}..{vertex_limit['ui_range'][1]})",
            f"- Schema extension: `{facts['installed_schema_extension']}`",
            "",
            f"**{max_hulls['project_input']} optimal: "
            f"{null_text(boundary['current_64cap_optimal'])}**",
            "",
            f"Physics equivalence: **{null_text(boundary['physics_equivalence'])}**",
            f"Toppling cause: **{null_text(boundary['collider_count_tipping_causality'])}**",
            f"Actual GPU contact execution: **{null_text(boundary['actual_gpu_contact_execution'])}**",
            f"Grasp feasibility: **{null_text(boundary['grasp_feasibility'])}**",
            "",
            "D369 Isaac / PhysX / q5 / steps: "
            f"**{SCOPE_GUARDS['isaac_or_kit_or_physx']} / "
            f"{SCOPE_GUARDS['isaac_or_kit_or_physx']} / "
            f"{SCOPE_GUARDS['q5_science_or_target']} / {SCOPE_GUARDS['physics_steps']}**",
            "",
            f"G0a: **{str(boundary['g0a_pass']).lower()}**",
            "",
            "CYAN raw patch | GREEN certified | YELLOW dual | BLUE other | PURPLE outer",
            "",
            "Display only. Frozen D368 JSON remains authority.",
        ]
    )
    return allocation, scope


def _write_overlay_and_blueprint(facts: dict[str, Any], *, staging_prefix: Path) -> dict[str, Any]:
    import rerun as rr

    if str(rr.__version__) != EXPECTED_RERUN_VERSION:
        raise RuntimeError(f"rerun SDK mismatch: {rr.__version__}")
    if OVERLAY_RRD_PATH.exists() or RBL_PATH.exists():
        raise FileExistsError("D369 overlay/RBL already exists")
    staging_overlay = staging_prefix.with_name(staging_prefix.name + "_overlay.rrd")
    staging_rbl = staging_prefix.with_name(staging_prefix.name + "_layout.rbl")
    if staging_overlay.exists() or staging_rbl.exists():
        raise FileExistsError("D369 PID-scoped staging overlay/RBL already exists")
    blueprint = _build_blueprint()
    allocation, scope = _overlay_text(facts)
    with rr.RecordingStream(
        APP_ID,
        recording_id=RECORDING_ID,
        make_default=False,
        send_properties=True,
    ) as recording:
        recording.save(str(staging_overlay), write_footer=True)
        recording.log(
            "presentation/d369/allocation",
            rr.TextDocument(allocation, media_type="text/markdown"),
            static=True,
        )
        recording.log(
            "presentation/d369/scope_and_64_basis",
            rr.TextDocument(scope, media_type="text/markdown"),
            static=True,
        )
        recording.flush(timeout_sec=30.0)
    blueprint.save(APP_ID, staging_rbl)
    _copy_file_x(staging_overlay, OVERLAY_RRD_PATH)
    _copy_file_x(staging_rbl, RBL_PATH)
    return {
        "rerun_sdk_version": str(rr.__version__),
        "application_id": APP_ID,
        "recording_id": RECORDING_ID,
        "entities": [
            "presentation/d369/allocation",
            "presentation/d369/scope_and_64_basis",
        ],
        "blueprint_spec": _blueprint_spec(),
        "overlay_rrd_embeds_blueprint": False,
        "staging_paths": [str(staging_overlay), str(staging_rbl)],
    }


def _build_recording_only_display_copy(*, staging_prefix: Path) -> dict[str, Any]:
    from rerun.experimental import RrdReader

    staging_recording = staging_prefix.with_name(staging_prefix.name + "_recording_only.rrd")
    if staging_recording.exists() or RECORDING_ONLY_PATH.exists():
        raise FileExistsError("D369 recording-only display copy already exists")
    reader = RrdReader(BASE_COPY_PATH)
    recordings = reader.recordings()
    blueprints = reader.blueprints()
    if len(recordings) != 1 or len(blueprints) != 1:
        raise RuntimeError(
            f"frozen D368 RRD store inventory changed: recordings={recordings}, blueprints={blueprints}"
        )
    recording = recordings[0]
    if (
        recording.kind != "recording"
        or recording.application_id != APP_ID
        or recording.recording_id != RECORDING_ID
    ):
        raise RuntimeError(f"frozen D368 recording store identity changed: {recording}")
    reader.stream(store=recording).write_rrd(
        staging_recording,
        application_id=APP_ID,
        recording_id=RECORDING_ID,
    )
    if not staging_recording.is_file():
        raise RuntimeError("RrdReader did not write the recording-only display copy")
    _copy_file_x(staging_recording, RECORDING_ONLY_PATH)
    compare_command = [
        str(RERUN_CLI),
        "rrd",
        "compare",
        "--unordered",
        str(BASE_COPY_PATH),
        str(RECORDING_ONLY_PATH),
    ]
    compare = _run(compare_command, timeout=90.0)
    if not compare["ok"]:
        raise RuntimeError(f"D368 recording-data semantic comparison failed: {compare}")
    return {
        "role": (
            "recording-store-only display projection; old blueprint store excluded; "
            "no collider/science recomputation"
        ),
        "source_store_inventory": {
            "recording_count": len(recordings),
            "blueprint_count": len(blueprints),
            "recording": {
                "kind": recording.kind,
                "application_id": recording.application_id,
                "recording_id": recording.recording_id,
            },
        },
        "compare_command": compare_command,
        "compare": compare,
        "staging_path": str(staging_recording),
        **_file_record(RECORDING_ONLY_PATH),
    }


def _build_single_presentation_archive(*, staging_prefix: Path) -> dict[str, Any]:
    staging_presentation = staging_prefix.with_name(staging_prefix.name + "_presentation.rrd")
    if staging_presentation.exists() or PRESENTATION_RRD_PATH.exists():
        raise FileExistsError("D369 presentation archive already exists")
    merge_command = [
        str(RERUN_CLI),
        "rrd",
        "merge",
        "-o",
        str(staging_presentation),
        str(RECORDING_ONLY_PATH),
        str(OVERLAY_RRD_PATH),
        str(RBL_PATH),
    ]
    merge = _run(merge_command)
    if not (merge["ok"] and staging_presentation.is_file()):
        raise RuntimeError(f"D369 RRD presentation merge failed: {merge}")
    _copy_file_x(staging_presentation, PRESENTATION_RRD_PATH)
    return {
        "role": "display-message packaging only; no geometry/collider/science recomputation",
        "merge_command": merge_command,
        "merge": merge,
        "staging_path": str(staging_presentation),
        "path": _rel(PRESENTATION_RRD_PATH),
        "bytes": PRESENTATION_RRD_PATH.stat().st_size,
        "sha256": _sha(PRESENTATION_RRD_PATH),
    }


def _copy_base_rrd() -> dict[str, Any]:
    if BASE_COPY_PATH.exists():
        raise FileExistsError(BASE_COPY_PATH)
    with SOURCE_RRD.open("rb") as source, BASE_COPY_PATH.open("xb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    return {
        "path": _rel(BASE_COPY_PATH),
        "bytes": BASE_COPY_PATH.stat().st_size,
        "sha256": _sha(BASE_COPY_PATH),
        "bitexact_to_source": BASE_COPY_PATH.stat().st_size == EXPECTED_RRD_BYTES
        and _sha(BASE_COPY_PATH) == EXPECTED_RRD_SHA256,
    }


def _png_dimensions(path: Path) -> list[int]:
    from PIL import Image

    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def _spatial_panel_diagnostics(path: Path) -> dict[str, Any]:
    from PIL import Image, ImageStat

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        top = max(1, int(round(height * 0.022)))
        left_width = int(round(width * 0.72))
        middle_x = left_width // 2
        middle_y = top + (height - top) // 2
        boxes = {
            "link5_full": (0, top, middle_x, middle_y),
            "link5_zoom": (middle_x, top, left_width, middle_y),
            "moving_full": (0, middle_y, middle_x, height),
            "moving_zoom": (middle_x, middle_y, left_width, height),
        }
        rows = {}
        required_colors = {
            "link5_full": ["cyan", "green", "blue"],
            "link5_zoom": ["cyan", "green"],
            "moving_full": ["cyan", "yellow", "purple", "blue"],
            "moving_zoom": ["cyan", "yellow", "purple"],
        }
        for name, box in boxes.items():
            crop = rgb.crop(box)
            stat = ImageStat.Stat(crop)
            mean_variance = sum(stat.var) / 3.0
            extrema = crop.getextrema()
            inner = crop.crop(
                (
                    int(crop.width * 0.04),
                    int(crop.height * 0.07),
                    int(crop.width * 0.96),
                    int(crop.height * 0.96),
                )
            )
            counts = {key: 0 for key in ("cyan", "green", "yellow", "purple", "blue")}
            stride = max(1, min(inner.width, inner.height) // 260)
            for y in range(0, inner.height, stride):
                for x in range(0, inner.width, stride):
                    red, green, blue = inner.getpixel((x, y))
                    hue, saturation, value = colorsys.rgb_to_hsv(
                        red / 255.0, green / 255.0, blue / 255.0
                    )
                    if saturation < 0.45 or value < 0.35:
                        continue
                    if 0.47 <= hue < 0.57:
                        counts["cyan"] += 1
                    if 0.25 <= hue < 0.44:
                        counts["green"] += 1
                    if 0.10 <= hue < 0.20:
                        counts["yellow"] += 1
                    if 0.72 <= hue < 0.90:
                        counts["purple"] += 1
                    if 0.57 <= hue < 0.72:
                        counts["blue"] += 1
            color_checks = {color: counts[color] >= 25 for color in required_colors[name]}
            rows[name] = {
                "box": list(box),
                "mean_channel_variance": mean_variance,
                "nonblank": mean_variance > 20.0 and any(high > low for low, high in extrema),
                "semantic_color_sample_stride": stride,
                "semantic_color_pixels": counts,
                "required_semantic_colors": required_colors[name],
                "semantic_color_checks": color_checks,
                "semantic_content_signature_pass": all(color_checks.values()),
            }
        return {
            "image_dimensions": [width, height],
            "requested_logical_window": WINDOW_SIZE,
            "allowed_dimensions": [[1920, 1080], [3840, 2160], [4800, 2800]],
            "dimensions_allowed": [width, height]
            in ([1920, 1080], [3840, 2160], [4800, 2800]),
            "panels": rows,
            "all_four_nonblank": all(row["nonblank"] for row in rows.values()),
            "all_four_semantic_content_signatures_pass": all(
                row["semantic_content_signature_pass"] for row in rows.values()
            ),
        }


def _render_professor_board(facts: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if SUMMARY_PNG_PATH.exists():
        raise FileExistsError(SUMMARY_PNG_PATH)
    fonts = _font_paths()
    title_font = ImageFont.truetype(fonts["bold"], 32)
    subtitle_font = ImageFont.truetype(fonts["regular"], 17)
    section_font = ImageFont.truetype(fonts["bold"], 20)
    body_font = ImageFont.truetype(fonts["regular"], 17)
    small_font = ImageFont.truetype(fonts["regular"], 15)

    canvas = Image.new("RGB", (1920, 1080), "#f4f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1920, 88), fill="#15202b")
    title = (
        f"D369 | Current {facts['max_convex_hulls']['project_input']}-cap reference: "
        "certified semantic-patch carrier allocation"
    )
    subtitle = "Frozen D368 evidence replay | OFFLINE display repair | no Isaac, PhysX, q5, or physics step"
    text_items: list[dict[str, Any]] = []

    def add_text(x: int, y: int, text: str, font: Any, fill: str, zone: str) -> list[int]:
        bbox = list(map(int, draw.textbbox((x, y), text, font=font)))
        draw.text((x, y), text, font=font, fill=fill)
        text_items.append({"text": text, "bbox": bbox, "zone": zone})
        return bbox

    add_text(34, 14, title, title_font, "white", "header")
    add_text(36, 57, subtitle, subtitle_font, "#dce6ef", "header")

    geometry_zone = (34, 108, 1300, 930)
    card_zone = (1324, 108, 1890, 930)
    draw.rounded_rectangle(geometry_zone, radius=12, fill="#111820", outline="#627180", width=2)
    draw.rounded_rectangle(card_zone, radius=12, fill="white", outline="#a8b2bd", width=2)

    with Image.open(RERUN_PNG_PATH) as image:
        rgb = image.convert("RGB")
        top = max(1, int(round(rgb.height * 0.022)))
        left_width = int(round(rgb.width * 0.72))
        spatial_group = rgb.crop((0, top, left_width, rgb.height))
        fitted = ImageOps.contain(spatial_group, (1242, 798), method=Image.Resampling.LANCZOS)
        paste_x = geometry_zone[0] + (geometry_zone[2] - geometry_zone[0] - fitted.width) // 2
        paste_y = geometry_zone[1] + (geometry_zone[3] - geometry_zone[1] - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))

    plan = _plan_fact_card(facts, width=540, height=790)
    for item in plan["items"]:
        style = item["style"]
        font = section_font if style == "section" else body_font
        local = item["bbox"]
        x = card_zone[0] + 13 + local[0]
        y = card_zone[1] + 3 + local[1]
        add_text(x, y, item["text"], font, "#17212b", "fact_card")

    footer_top = 956
    draw.line((34, footer_top, 1890, footer_top), fill="#84919d", width=2)
    footer1 = "CYAN raw patch | GREEN certified | YELLOW dual inner+outer | BLUE other | PURPLE outer patch"
    boundary = facts["interpretation_boundary"]
    null_text = lambda value: "NULL" if value is None else str(value)
    unresolved_values = " / ".join(
        null_text(boundary[key])
        for key in (
            "current_64cap_optimal",
            "physics_equivalence",
            "collider_count_tipping_causality",
            "actual_gpu_contact_execution",
            "grasp_feasibility",
        )
    )
    footer2 = (
        f"{facts['max_convex_hulls']['project_input']} optimal / physics-equivalent / "
        "toppling-cause / GPU-contact / grasp = "
        f"{unresolved_values}. G0a = {str(boundary['g0a_pass']).lower()}."
    )
    footer3 = "Authority: immutable D368 JSON + RRD. This board is display evidence only."
    add_text(44, 970, footer1, small_font, "#28343f", "footer")
    add_text(44, 1001, footer2, small_font, "#28343f", "footer")
    add_text(44, 1032, footer3, small_font, "#56636f", "footer")

    zone_bounds = {
        "header": (20, 8, 1900, 84),
        "fact_card": (card_zone[0] + 8, card_zone[1] + 8, card_zone[2] - 8, card_zone[3] - 8),
        "footer": (30, 962, 1900, 1070),
    }
    containment = {}
    for index, item in enumerate(text_items):
        zone = zone_bounds[item["zone"]]
        box = item["bbox"]
        containment[str(index)] = (
            box[0] >= zone[0]
            and box[1] >= zone[1]
            and box[2] <= zone[2]
            and box[3] <= zone[3]
        )
    overlaps = []
    for i, first in enumerate(text_items):
        for j in range(i + 1, len(text_items)):
            second = text_items[j]
            if first["zone"] != second["zone"]:
                continue
            x0 = max(first["bbox"][0], second["bbox"][0])
            y0 = max(first["bbox"][1], second["bbox"][1])
            x1 = min(first["bbox"][2], second["bbox"][2])
            y1 = min(first["bbox"][3], second["bbox"][3])
            if x1 > x0 and y1 > y0:
                overlaps.append({"i": i, "j": j, "area_px2": (x1 - x0) * (y1 - y0)})

    encoded = io.BytesIO()
    canvas.save(encoded, format="PNG")
    with SUMMARY_PNG_PATH.open("xb") as handle:
        handle.write(encoded.getvalue())
    dimensions = _png_dimensions(SUMMARY_PNG_PATH)
    checks = {
        "exact_1920x1080": dimensions == [1920, 1080],
        "fact_card_plan_pass": plan["pass"],
        "all_text_within_registered_zones": all(containment.values()),
        "drawn_text_bbox_overlap_zero": len(overlaps) == 0,
        "geometry_and_fact_zones_disjoint": geometry_zone[2] < card_zone[0],
        "footer_gap_positive": geometry_zone[3] < footer_top and card_zone[3] < footer_top,
    }
    return {
        "path": _rel(SUMMARY_PNG_PATH),
        "bytes": SUMMARY_PNG_PATH.stat().st_size,
        "sha256": _sha(SUMMARY_PNG_PATH),
        "dimensions": dimensions,
        "zones": {
            "geometry": list(geometry_zone),
            "fact_card": list(card_zone),
            "footer_top": footer_top,
        },
        "text_items": text_items,
        "containment": containment,
        "overlaps": overlaps,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _prepare() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(OUT_DIR.iterdir()):
        raise RuntimeError("D369 output must be empty before prepare")
    source_manifest = _source_manifest()
    evidence = _read_json(SOURCE_EVIDENCE)
    facts = _extract_facts(evidence)
    fact_checks = _facts_exact(facts)
    negative = _negative_controls(facts)
    cli_version = _run([str(RERUN_CLI), "--version"])
    dirty_paths = _git_status_paths()
    checks = {
        "output_empty_before_preregistration": True,
        "head_equals_origin_master": _git("rev-parse", "HEAD") == _git("rev-parse", "origin/master"),
        "head_expected": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "dirty_scope_allowed": _dirty_scope_pass(dirty_paths),
        "source_allowlist_exactly_two": set(source_manifest)
        == {_rel(SOURCE_EVIDENCE), _rel(SOURCE_RRD)},
        "source_hashes_and_bytes_exact": _source_manifest_exact(source_manifest),
        "frozen_facts_exact": all(fact_checks.values()),
        "new_variable_count_two": len(NEW_VARIABLES) == 2,
        "scope_guards_exact": SCOPE_GUARDS["rerun_display_render_allowed"] == 1
        and all(value == 0 for key, value in SCOPE_GUARDS.items() if key != "rerun_display_render_allowed"),
        "negative_controls_exact_10_of_10": negative["pass"]
        and negative["passed"] == 10
        and negative["total"] == 10
        and set(negative["checks"]) == EXPECTED_NEGATIVE_CHECK_KEYS,
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "rerun_cli_version_exact": cli_version["ok"]
        and EXPECTED_RERUN_VERSION in f"{cli_version['stdout']}\n{cli_version['stderr']}",
        "isaac_compatible_pins_unchanged": importlib.metadata.version("numpy") == "1.26.0"
        and importlib.metadata.version("psutil") == "5.9.8",
        "forbidden_runtime_modules_absent": not _forbidden_modules(),
    }
    prereg = {
        "artifact": "D369_PREREGISTRATION_V1",
        "case": "g0a_d369",
        "head": EXPECTED_HEAD,
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": [],
        "frozen_domain_input_allowlist": sorted(source_manifest),
        "source_manifest": source_manifest,
        "facts": facts,
        "fact_checks": fact_checks,
        "root_cause": {
            "classification": "display-contract mismatch, not Isaac/PhysX/geometry failure",
            "d368_metric_storage": "static Scalars without recording time",
            "d368_view": "timeline-dependent DataframeView without a valid recording timeline",
            "repair": "static TextDocumentView overlay; no artificial science timeline",
        },
        "blueprint_spec": _blueprint_spec(),
        "render_contract": {
            "invocation_count": 1,
            "automatic_retry_count": 0,
            "execution": "host-side first attempt; do not consume a sandbox render first",
            "required_caller_environment": {HOST_RENDER_ENV: "1"},
            "pre_render_host_capability_probe": "one local-loopback ephemeral socket bind+listen",
            "d368_bitexact_preservation_copy": _rel(BASE_COPY_PATH),
            "recording_only_projection": {
                "path": _rel(RECORDING_ONLY_PATH),
                "method": "Rerun 0.34.1 RrdReader selects the sole recording store",
                "equivalence_gate": "rerun rrd compare --unordered against bit-exact D368 copy",
                "expected_non_system_entity_count": 284,
            },
            "presentation_build_inputs": [
                _rel(RECORDING_ONLY_PATH),
                _rel(OVERLAY_RRD_PATH),
                _rel(RBL_PATH),
            ],
            "presentation_packaging": (
                "RrdReader excludes the inherited blueprint store; rrd merge adds exactly one "
                "new RBL activation; recording data must compare equal; display messages only"
            ),
            "render_input": [_rel(PRESENTATION_RRD_PATH)],
            "window_size_logical": WINDOW_SIZE,
            "allowed_raw_png_dimensions": [[1920, 1080], [3840, 2160], [4800, 2800]],
            "professor_board_dimensions": [1920, 1080],
        },
        "negative_controls": negative,
        "scope_guards": SCOPE_GUARDS,
        "forbidden_module_prefixes": FORBIDDEN_MODULE_PREFIXES,
        "dirty_paths_at_prepare": dirty_paths,
        "harness_sha256": _sha(HARNESS),
        "dynamic_source_manifest": {
            _rel(HARNESS): _sha(HARNESS),
            _rel(RERUN_CONTRACT_HELPER): _sha(RERUN_CONTRACT_HELPER),
        },
        "rerun_cli_version": cli_version,
        "checks": checks,
        "pass": all(checks.values()),
        "decision_rule": {
            "pass": "D369_D368_PROFESSOR_VISUAL_CONTRACT_REPAIRED_OBSERVABILITY_ONLY",
            "fail": "D369_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP",
            "d368_measurement_verdict_preserved": EXPECTED_MEASUREMENT_VERDICT,
        },
        "science_boundary": facts["interpretation_boundary"],
        "next_science_requires_new_approval": True,
    }
    _write_json_x(PREREG_PATH, prereg)
    if not prereg["pass"]:
        raise RuntimeError(f"D369 preregistration failed: {checks}")


def _render() -> None:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D369 prepare must pass before render")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D369 preregistration is not PASS")
    if os.environ.get(HOST_RENDER_ENV) != "1":
        raise RuntimeError(
            f"host-render capability gate not present: set {HOST_RENDER_ENV}=1 only on the approved host call"
        )
    if {path.name for path in OUT_DIR.iterdir()} != {PREREG_PATH.name}:
        raise RuntimeError("forward-only pre-render inventory mismatch")
    if _git("rev-parse", "HEAD") != EXPECTED_HEAD or _git("rev-parse", "origin/master") != EXPECTED_HEAD:
        raise RuntimeError("Git base changed after D369 preregistration")
    current_dynamic_sources = {
        _rel(HARNESS): _sha(HARNESS),
        _rel(RERUN_CONTRACT_HELPER): _sha(RERUN_CONTRACT_HELPER),
    }
    if current_dynamic_sources != prereg["dynamic_source_manifest"]:
        raise RuntimeError("D369 harness/Rerun validator changed after preregistration")
    if _source_manifest() != prereg["source_manifest"]:
        raise RuntimeError("immutable D368 evidence/RRD changed after preregistration")
    if _forbidden_modules():
        raise RuntimeError(f"forbidden runtime modules imported before render: {_forbidden_modules()}")

    PHASE_PATH.touch(exist_ok=False)
    _phase("render_started")
    started = time.monotonic()
    try:
        host_capability_probe = _loopback_bind_probe()
        if host_capability_probe["pass"] is not True:
            raise RuntimeError(f"host capability probe failed: {host_capability_probe}")
        _phase("host_loopback_bind_capability_pass")

        evidence = _read_json(SOURCE_EVIDENCE)
        facts = _extract_facts(evidence)
        if not all(_facts_exact(facts).values()):
            raise RuntimeError("frozen facts no longer satisfy preregistered contract")
        _phase("frozen_evidence_fields_copied", fact_count=len(facts))

        base_copy = _copy_base_rrd()
        if not base_copy["bitexact_to_source"]:
            raise RuntimeError("D368 RRD output copy is not bit-exact")
        _phase("d368_rrd_bitexact_copy_complete", sha256=base_copy["sha256"])

        staging_prefix = Path(f"/tmp/roarm_d369_{os.getpid()}")
        overlay = _write_overlay_and_blueprint(facts, staging_prefix=staging_prefix)
        _phase("static_text_overlay_and_blueprint_finalized")

        recording_only = _build_recording_only_display_copy(staging_prefix=staging_prefix)
        _phase(
            "d368_recording_only_display_copy_finalized",
            sha256=recording_only["sha256"],
        )

        presentation = _build_single_presentation_archive(staging_prefix=staging_prefix)
        _phase("single_presentation_archive_finalized", sha256=presentation["sha256"])

        from roarm_rl.rerun_contract import (
            _entity_paths,
            _is_system_entity,
            validate_rerun_artifact,
        )

        cli_version = _run([str(RERUN_CLI), "--version"])
        base_verify = _run(
            [str(RERUN_CLI), "rrd", "verify", "--check-footers", "true", str(BASE_COPY_PATH)]
        )
        overlay_verify = _run(
            [str(RERUN_CLI), "rrd", "verify", "--check-footers", "true", str(OVERLAY_RRD_PATH)]
        )
        recording_only_verify = _run(
            [
                str(RERUN_CLI),
                "rrd",
                "verify",
                "--check-footers",
                "true",
                str(RECORDING_ONLY_PATH),
            ]
        )
        rbl_verify = _run(
            [str(RERUN_CLI), "rrd", "verify", "--check-footers", "true", str(RBL_PATH)]
        )
        presentation_verify = _run(
            [
                str(RERUN_CLI),
                "rrd",
                "verify",
                "--check-footers",
                "true",
                str(PRESENTATION_RRD_PATH),
            ]
        )
        base_stats = _run([str(RERUN_CLI), "rrd", "stats", str(BASE_COPY_PATH)])
        recording_only_stats = _run(
            [str(RERUN_CLI), "rrd", "stats", str(RECORDING_ONLY_PATH)]
        )
        overlay_stats = _run([str(RERUN_CLI), "rrd", "stats", str(OVERLAY_RRD_PATH)])
        presentation_stats = _run(
            [str(RERUN_CLI), "rrd", "stats", str(PRESENTATION_RRD_PATH)]
        )
        overlay_print = _run([str(RERUN_CLI), "rrd", "print", "-v", str(OVERLAY_RRD_PATH)])
        recording_only_print = _run(
            [str(RERUN_CLI), "rrd", "print", "-vvv", str(RECORDING_ONLY_PATH)]
        )
        rbl_print = _run([str(RERUN_CLI), "rrd", "print", "-vvv", str(RBL_PATH)])
        rbl_text = f"{rbl_print['stdout']}\n{rbl_print['stderr']}"
        presentation_print = _run(
            [str(RERUN_CLI), "rrd", "print", "-vvv", str(PRESENTATION_RRD_PATH)],
            timeout=90.0,
        )
        presentation_text = f"{presentation_print['stdout']}\n{presentation_print['stderr']}"
        overlay_text = f"{overlay_print['stdout']}\n{overlay_print['stderr']}"
        recording_only_text = (
            f"{recording_only_print['stdout']}\n{recording_only_print['stderr']}"
        )
        rbl_activation_ids = _blueprint_activation_ids(rbl_text)
        presentation_activation_ids = _blueprint_activation_ids(presentation_text)
        rbl_activation_commands = _blueprint_activation_commands(rbl_text)
        presentation_activation_commands = _blueprint_activation_commands(presentation_text)
        base_user_entities = {
            path
            for path in _entity_paths(f"{base_stats['stdout']}\n{base_stats['stderr']}")
            if not _is_system_entity(path)
        }
        expected_overlay_entities = {f"/{path}" for path in overlay["entities"]}
        expected_presentation_entities = base_user_entities | expected_overlay_entities
        recording_only_validation = validate_rerun_artifact(
            RECORDING_ONLY_PATH,
            expected_entity_paths=sorted(base_user_entities),
            exact_entity_paths=sorted(base_user_entities),
            cli_path=RERUN_CLI,
            expected_version=EXPECTED_RERUN_VERSION,
            timeout_s=90.0,
        )
        overlay_validation = validate_rerun_artifact(
            OVERLAY_RRD_PATH,
            expected_entity_paths=sorted(expected_overlay_entities),
            exact_entity_paths=sorted(expected_overlay_entities),
            expected_entity_components={
                path: ["TextDocument:text", "TextDocument:media_type"]
                for path in sorted(expected_overlay_entities)
            },
            cli_path=RERUN_CLI,
            expected_version=EXPECTED_RERUN_VERSION,
            timeout_s=90.0,
        )
        presentation_validation = validate_rerun_artifact(
            PRESENTATION_RRD_PATH,
            expected_entity_paths=sorted(expected_presentation_entities),
            expected_timeline_names=["blueprint", "log_time"],
            exact_entity_paths=sorted(expected_presentation_entities),
            exact_timeline_names=["blueprint", "log_time"],
            expected_entity_components={
                path: ["TextDocument:text", "TextDocument:media_type"]
                for path in sorted(expected_overlay_entities)
            },
            cli_path=RERUN_CLI,
            expected_version=EXPECTED_RERUN_VERSION,
            timeout_s=90.0,
        )
        pre_render_validation = {
            "cli_version": cli_version,
            "base_verify": base_verify,
            "recording_only_verify": recording_only_verify,
            "overlay_verify": overlay_verify,
            "rbl_verify": rbl_verify,
            "presentation_verify": presentation_verify,
            "base_stats": base_stats,
            "recording_only_stats": recording_only_stats,
            "overlay_stats": overlay_stats,
            "presentation_stats": presentation_stats,
            "recording_only_validation": recording_only_validation,
            "overlay_validation": overlay_validation,
            "presentation_validation": presentation_validation,
            "rbl_print": rbl_print,
            "presentation_print": presentation_print,
            "presentation_packaging": presentation,
            "rbl_activation_ids": rbl_activation_ids,
            "presentation_activation_ids": presentation_activation_ids,
            "rbl_activation_commands": rbl_activation_commands,
            "presentation_activation_commands": presentation_activation_commands,
            "checks": {
                "version_exact": cli_version["ok"]
                and EXPECTED_RERUN_VERSION in f"{cli_version['stdout']}\n{cli_version['stderr']}",
                "base_footer_verify": base_verify["ok"],
                "base_stats_and_user_entity_count_exact": base_stats["ok"]
                and len(base_user_entities)
                == prereg["render_contract"]["recording_only_projection"][
                    "expected_non_system_entity_count"
                ],
                "recording_only_footer_verify": recording_only_verify["ok"],
                "overlay_footer_verify": overlay_verify["ok"],
                "rbl_footer_verify": rbl_verify["ok"],
                "presentation_footer_verify": presentation_verify["ok"],
                "recording_only_exact_entity_contract": recording_only_validation["pass"],
                "overlay_exact_entity_component_static_contract": overlay_validation["pass"],
                "presentation_exact_entity_component_timeline_contract": presentation_validation[
                    "pass"
                ],
                "overlay_app_and_recording_id_exact": APP_ID in overlay_text
                and RECORDING_ID in overlay_text,
                "recording_only_app_and_recording_id_exact": APP_ID in recording_only_text
                and RECORDING_ID in recording_only_text,
                "recording_only_has_zero_blueprint_activation": not _blueprint_activation_ids(
                    recording_only_text
                ),
                "overlay_has_zero_blueprint_activation": not _blueprint_activation_ids(overlay_text),
                "recording_only_and_overlay_have_no_indexes": recording_only_stats["ok"]
                and overlay_stats["ok"]
                and _strict_index_names(
                    f"{recording_only_stats['stdout']}\n{recording_only_stats['stderr']}"
                )
                == set()
                and _strict_index_names(
                    f"{overlay_stats['stdout']}\n{overlay_stats['stderr']}"
                )
                == set(),
                "presentation_indexes_exact": presentation_stats["ok"]
                and _strict_index_names(
                    f"{presentation_stats['stdout']}\n{presentation_stats['stderr']}"
                )
                == {"blueprint", "log_time"},
                "serialized_rbl_six_views_exact": rbl_print["ok"]
                and rbl_text.count("[3D]") == 4
                and rbl_text.count("[TextDocument]") == 2
                and rbl_text.count("[Dataframe]") == 0
                and rbl_text.count("[TimeSeries]") == 0
                and rbl_text.count("[TextLog]") == 0,
                "serialized_rbl_text_queries_present": "/presentation/d369/allocation" in rbl_text
                and "/presentation/d369/scope_and_64_basis" in rbl_text,
                "serialized_rbl_timeline_dependent_views_absent": "Dataframe" not in rbl_text
                and "TimeSeries" not in rbl_text
                and "DataframeQuery" not in rbl_text,
                "serialized_rbl_label_paths_absent": "/semantic/anchors" not in rbl_text
                and "/semantic/normals" not in rbl_text,
                "rbl_has_exactly_one_activation": len(rbl_activation_ids) == 1,
                "presentation_has_exactly_one_new_rbl_activation": len(
                    presentation_activation_ids
                )
                == 1
                and presentation_activation_ids == rbl_activation_ids,
                "activation_full_tuple_exact": rbl_activation_commands
                == presentation_activation_commands
                == [
                    {
                        "application_id": APP_ID,
                        "blueprint_id": rbl_activation_ids[0] if rbl_activation_ids else None,
                        "make_active": True,
                        "make_default": True,
                    }
                ],
                "recording_projection_compare_and_merge_pass": recording_only["compare"]["ok"]
                and presentation["merge"]["ok"],
            },
        }
        pre_render_validation["pass"] = all(pre_render_validation["checks"].values())
        if not pre_render_validation["pass"]:
            raise RuntimeError(f"pre-render Rerun contract failed: {pre_render_validation['checks']}")
        _phase("pre_render_artifact_contract_pass")

        staging_render_png = staging_prefix.with_name(staging_prefix.name + "_rerun.png")
        if staging_render_png.exists():
            raise FileExistsError(staging_render_png)
        render_command = [
            str(RERUN_CLI),
            "--headless",
            "--hide-welcome-screen",
            "--port",
            "auto",
            "--window-size",
            WINDOW_SIZE,
            "--screenshot-to",
            str(staging_render_png),
            str(PRESENTATION_RRD_PATH),
        ]
        invocation = {
            "artifact": "D369_ONE_SHOT_RENDER_INVOCATION_V1",
            "render_invocation_count": 1,
            "automatic_retry_count": 0,
            "no_retry": True,
            "command": render_command,
            "input_order": [_rel(PRESENTATION_RRD_PATH)],
            "presentation_build_inputs": [
                _rel(RECORDING_ONLY_PATH),
                _rel(OVERLAY_RRD_PATH),
                _rel(RBL_PATH),
            ],
            "d368_bitexact_preservation_copy": _rel(BASE_COPY_PATH),
            "preregistration_sha256": _sha(PREREG_PATH),
            "host_render_gate": {HOST_RENDER_ENV: os.environ.get(HOST_RENDER_ENV)},
            "host_capability_probe": host_capability_probe,
            "staging_screenshot_path": str(staging_render_png),
            "final_screenshot_path": _rel(RERUN_PNG_PATH),
            "harness_sha256": _sha(HARNESS),
            "source_manifest": _source_manifest(),
            "scope_guards": SCOPE_GUARDS,
        }
        _write_json_x(INVOCATION_PATH, invocation)
        _phase("one_shot_headless_render_invoked")
        render_result = _run(render_command, timeout=RENDER_TIMEOUT_SECONDS)
        if render_result["ok"] and staging_render_png.is_file():
            _copy_file_x(staging_render_png, RERUN_PNG_PATH)
        receipt = {
            "artifact": "D369_ONE_SHOT_RENDER_RECEIPT_V1",
            **render_result,
            "staging_screenshot_exists": staging_render_png.is_file(),
            "screenshot_exists": RERUN_PNG_PATH.is_file(),
            "screenshot_bytes": RERUN_PNG_PATH.stat().st_size if RERUN_PNG_PATH.is_file() else 0,
            "screenshot_sha256": _sha(RERUN_PNG_PATH) if RERUN_PNG_PATH.is_file() else None,
            "unknown_timeline_console_text_absent": "Unknown timeline"
            not in f"{render_result['stdout']}\n{render_result['stderr']}",
            "message_proxy_operation_not_permitted_pair_absent": not (
                "message proxy server crashed" in f"{render_result['stdout']}\n{render_result['stderr']}"
                and "Operation not permitted" in f"{render_result['stdout']}\n{render_result['stderr']}"
            ),
        }
        _write_json_x(RECEIPT_PATH, receipt)
        if not (render_result["ok"] and receipt["screenshot_exists"]):
            raise RuntimeError(f"one-shot D369 render failed: {receipt}")
        _phase("one_shot_headless_render_returned", returncode=render_result["returncode"])

        panel_diagnostics = _spatial_panel_diagnostics(RERUN_PNG_PATH)
        summary = _render_professor_board(facts)
        _phase("professor_board_written", sha256=summary["sha256"])

        source_after = _source_manifest()
        post_modules = _forbidden_modules()
        automated_checks = {
            "pre_render_contract_pass": pre_render_validation["pass"],
            "render_returncode_zero": render_result["ok"],
            "host_loopback_bind_capability_pass": host_capability_probe["pass"] is True
            and host_capability_probe["attempt_count"] == 1
            and host_capability_probe["bind_succeeded"] is True
            and host_capability_probe["listen_succeeded"] is True,
            "render_invocation_exactly_one": invocation["render_invocation_count"] == 1,
            "automatic_retry_zero": invocation["automatic_retry_count"] == 0,
            "render_input_order_exact": invocation["input_order"]
            == [_rel(PRESENTATION_RRD_PATH)],
            "unknown_timeline_console_signature_absent": receipt[
                "unknown_timeline_console_text_absent"
            ],
            "message_proxy_operation_not_permitted_pair_absent": receipt[
                "message_proxy_operation_not_permitted_pair_absent"
            ],
            "raw_png_dimensions_allowed": panel_diagnostics["dimensions_allowed"],
            "raw_four_spatial_panels_nonblank": panel_diagnostics["all_four_nonblank"],
            "raw_four_semantic_content_signatures_pass": panel_diagnostics[
                "all_four_semantic_content_signatures_pass"
            ],
            "summary_layout_pass": summary["pass"],
            "source_inputs_unchanged": source_after == prereg["source_manifest"],
            "base_copy_bitexact": base_copy["bitexact_to_source"],
            "recording_projection_semantically_equal": recording_only["compare"]["ok"],
            "forbidden_runtime_modules_absent_after_render": not post_modules,
            "scope_guards_preserved": SCOPE_GUARDS == prereg["scope_guards"],
        }
        automated = {
            "artifact": "D369_AUTOMATED_VISUAL_CONTRACT_SUMMARY_V1",
            "case": "g0a_d369",
            "elapsed_seconds": time.monotonic() - started,
            "source_manifest_before": prereg["source_manifest"],
            "source_manifest_after": source_after,
            "base_copy": base_copy,
            "recording_only": recording_only,
            "overlay": {
                **overlay,
                "rrd": _file_record(OVERLAY_RRD_PATH),
                "rbl": _file_record(RBL_PATH),
            },
            "presentation": presentation,
            "presentation_rrd": _file_record(PRESENTATION_RRD_PATH),
            "pre_render_validation": pre_render_validation,
            "render_invocation": _file_record(INVOCATION_PATH),
            "render_receipt": _file_record(RECEIPT_PATH),
            "rerun_png": {
                **_file_record(RERUN_PNG_PATH),
                "dimensions": _png_dimensions(RERUN_PNG_PATH),
            },
            "spatial_panel_diagnostics": panel_diagnostics,
            "professor_board": summary,
            "facts": facts,
            "checks": automated_checks,
            "pass": all(automated_checks.values()),
            "manual_original_resolution_inspection_required": True,
            "science_boundary": facts["interpretation_boundary"],
            "scope_guards": SCOPE_GUARDS,
        }
        _write_json_x(AUTOMATED_PATH, automated)
        report = "\n".join(
            [
                "# D369 automated visual-contract report",
                "",
                f"- automated pass: `{automated['pass']}`",
                f"- one render / retry: `1 / 0`",
                f"- source RRD copy bit-exact: `{base_copy['bitexact_to_source']}`",
                f"- four spatial panels nonblank: `{panel_diagnostics['all_four_nonblank']}`",
                "- four spatial semantic signatures: "
                f"`{panel_diagnostics['all_four_semantic_content_signatures_pass']}`",
                f"- exact 1920x1080 board layout pass: `{summary['pass']}`",
                f"- forbidden runtime modules: `{post_modules}`",
                "- scientific verdict: preserved from immutable D368; not recomputed",
                "- manual original-resolution inspection: required before finalize",
                "",
            ]
        )
        _write_text_x(REPORT_PATH, report)
        validation_files = {
            _rel(path): _file_record(path)
            for path in (
                PREREG_PATH,
                BASE_COPY_PATH,
                RECORDING_ONLY_PATH,
                OVERLAY_RRD_PATH,
                RBL_PATH,
                PRESENTATION_RRD_PATH,
                INVOCATION_PATH,
                RECEIPT_PATH,
                RERUN_PNG_PATH,
                SUMMARY_PNG_PATH,
                AUTOMATED_PATH,
                REPORT_PATH,
            )
        }
        validation_checks = {
            "automated_pass": automated["pass"] is True,
            "pre_render_pass": pre_render_validation["pass"] is True,
            "presentation_record_matches_packaging": validation_files[
                _rel(PRESENTATION_RRD_PATH)
            ]["sha256"]
            == presentation["sha256"]
            and validation_files[_rel(PRESENTATION_RRD_PATH)]["bytes"]
            == presentation["bytes"],
            "base_copy_bitexact": base_copy["bitexact_to_source"] is True,
            "recording_projection_record_matches": validation_files[
                _rel(RECORDING_ONLY_PATH)
            ]["sha256"]
            == recording_only["sha256"]
            and validation_files[_rel(RECORDING_ONLY_PATH)]["bytes"]
            == recording_only["bytes"]
            and recording_only["compare"]["ok"],
            "source_inputs_unchanged": source_after == prereg["source_manifest"],
            "blueprint_spec_exact": overlay["blueprint_spec"] == _blueprint_spec(),
        }
        validation = {
            "artifact": "D369_RERUN_VALIDATION_V1",
            "case": "g0a_d369",
            "files": validation_files,
            "pre_render": pre_render_validation,
            "blueprint_spec": _blueprint_spec(),
            "checks": validation_checks,
            "pass": all(validation_checks.values()),
        }
        _write_json_x(VALIDATION_PATH, validation)
        _phase("automated_artifacts_finalized", automated_pass=automated["pass"])
        if not (automated["pass"] and validation["pass"]):
            raise RuntimeError(
                f"D369 automated/validation gate failed: {automated_checks} / {validation_checks}"
            )
    except Exception as exc:
        if not EXCEPTION_PATH.exists():
            _write_json_x(
                EXCEPTION_PATH,
                {
                    "artifact": "D369_RUNTIME_EXCEPTION_V1",
                    "case": "g0a_d369",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "render_invocation_exists": INVOCATION_PATH.is_file(),
                    "render_retry_forbidden": True,
                    "scope_guards": SCOPE_GUARDS,
                },
            )
        raise


def _finalize() -> None:
    if COMPLETION_PATH.exists():
        raise FileExistsError(COMPLETION_PATH)
    prereg = _read_json(PREREG_PATH)
    automated = _read_json(AUTOMATED_PATH)
    validation = _read_json(VALIDATION_PATH)
    manual = _read_json(MANUAL_JSON_PATH)
    invocation = _read_json(INVOCATION_PATH)
    receipt = _read_json(RECEIPT_PATH)
    manual_md = MANUAL_MD_PATH.read_text(encoding="utf-8")
    expected_inventory = {
        PREREG_PATH.name,
        PHASE_PATH.name,
        BASE_COPY_PATH.name,
        RECORDING_ONLY_PATH.name,
        OVERLAY_RRD_PATH.name,
        RBL_PATH.name,
        PRESENTATION_RRD_PATH.name,
        INVOCATION_PATH.name,
        RECEIPT_PATH.name,
        VALIDATION_PATH.name,
        RERUN_PNG_PATH.name,
        SUMMARY_PNG_PATH.name,
        AUTOMATED_PATH.name,
        REPORT_PATH.name,
        MANUAL_JSON_PATH.name,
        MANUAL_MD_PATH.name,
    }
    current_inventory = {path.name for path in OUT_DIR.iterdir()}
    manual_checks = manual.get("checks", {})
    expected_manual_paths = {_rel(RERUN_PNG_PATH), _rel(SUMMARY_PNG_PATH)}
    manual_files = manual.get("files", [])
    observed_manual_paths = {row.get("path") for row in manual_files}
    phase_rows = _read_phase_rows()
    phase_names = [row.get("phase") for row in phase_rows]
    phase_times = [row.get("monotonic_seconds") for row in phase_rows]
    current_dynamic_sources = {
        _rel(HARNESS): _sha(HARNESS),
        _rel(RERUN_CONTRACT_HELPER): _sha(RERUN_CONTRACT_HELPER),
    }
    expected_render_command = [
        str(RERUN_CLI),
        "--headless",
        "--hide-welcome-screen",
        "--port",
        "auto",
        "--window-size",
        WINDOW_SIZE,
        "--screenshot-to",
        invocation.get("staging_screenshot_path"),
        str(PRESENTATION_RRD_PATH),
    ]
    expected_compare_command = [
        str(RERUN_CLI),
        "rrd",
        "compare",
        "--unordered",
        str(BASE_COPY_PATH),
        str(RECORDING_ONLY_PATH),
    ]
    expected_merge_command = [
        str(RERUN_CLI),
        "rrd",
        "merge",
        "-o",
        automated.get("presentation", {}).get("staging_path"),
        str(RECORDING_ONLY_PATH),
        str(OVERLAY_RRD_PATH),
        str(RBL_PATH),
    ]

    def record_matches(record: Any, path: Path) -> bool:
        return (
            isinstance(record, dict)
            and record.get("path") == _rel(path)
            and record.get("bytes") == path.stat().st_size
            and record.get("sha256") == _sha(path)
        )

    validation_files = validation.get("files", {})
    validation_records_match = isinstance(validation_files, dict) and all(
        key == record.get("path") and record_matches(record, REPO / key)
        for key, record in validation_files.items()
    )
    expected_validation_paths = {
        _rel(path)
        for path in (
            PREREG_PATH,
            BASE_COPY_PATH,
            RECORDING_ONLY_PATH,
            OVERLAY_RRD_PATH,
            RBL_PATH,
            PRESENTATION_RRD_PATH,
            INVOCATION_PATH,
            RECEIPT_PATH,
            RERUN_PNG_PATH,
            SUMMARY_PNG_PATH,
            AUTOMATED_PATH,
            REPORT_PATH,
        )
    }
    manual_records_match = len(manual_files) == 2 and all(
        record_matches(row, REPO / row["path"])
        and row.get("dimensions") == _png_dimensions(REPO / row["path"])
        for row in manual_files
    )
    automated_records_match = (
        automated.get("base_copy", {}).get("path") == _rel(BASE_COPY_PATH)
        and automated.get("base_copy", {}).get("bytes") == BASE_COPY_PATH.stat().st_size
        and automated.get("base_copy", {}).get("sha256") == _sha(BASE_COPY_PATH)
        and record_matches(automated.get("recording_only"), RECORDING_ONLY_PATH)
        and record_matches(automated.get("overlay", {}).get("rrd"), OVERLAY_RRD_PATH)
        and record_matches(automated.get("overlay", {}).get("rbl"), RBL_PATH)
        and record_matches(automated.get("presentation_rrd"), PRESENTATION_RRD_PATH)
        and record_matches(automated.get("render_invocation"), INVOCATION_PATH)
        and record_matches(automated.get("render_receipt"), RECEIPT_PATH)
        and record_matches(automated.get("rerun_png"), RERUN_PNG_PATH)
        and record_matches(automated.get("professor_board"), SUMMARY_PNG_PATH)
    )
    checks = {
        "precompletion_inventory_exact": current_inventory == expected_inventory,
        "preregistration_pass": prereg.get("pass") is True,
        "preregistration_artifact_case_and_inner_checks_exact": prereg.get("artifact")
        == "D369_PREREGISTRATION_V1"
        and prereg.get("case") == "g0a_d369"
        and isinstance(prereg.get("checks"), dict)
        and bool(prereg["checks"])
        and all(value is True for value in prereg["checks"].values())
        and isinstance(prereg.get("fact_checks"), dict)
        and all(value is True for value in prereg["fact_checks"].values())
        and prereg.get("negative_controls", {}).get("passed") == 10
        and prereg.get("negative_controls", {}).get("total") == 10
        and prereg.get("negative_controls", {}).get("pass") is True
        and set(prereg.get("negative_controls", {}).get("checks", {}))
        == EXPECTED_NEGATIVE_CHECK_KEYS,
        "automated_pass": automated.get("pass") is True,
        "validation_pass": validation.get("pass") is True,
        "automated_and_pre_render_checks_all_pass": isinstance(
            automated.get("checks"), dict
        )
        and bool(automated["checks"])
        and all(value is True for value in automated["checks"].values())
        and automated.get("pre_render_validation", {}).get("pass") is True
        and all(
            value is True
            for value in automated["pre_render_validation"].get("checks", {}).values()
        ),
        "validation_checks_all_pass": isinstance(validation.get("checks"), dict)
        and bool(validation["checks"])
        and all(value is True for value in validation["checks"].values()),
        "validation_pre_render_matches_automated": validation.get("pre_render")
        == automated.get("pre_render_validation"),
        "git_head_origin_and_preregistered_head_exact": _git("rev-parse", "HEAD")
        == EXPECTED_HEAD
        == _git("rev-parse", "origin/master")
        == prereg.get("head"),
        "dynamic_source_hashes_unchanged": current_dynamic_sources
        == prereg.get("dynamic_source_manifest"),
        "dirty_scope_allowed": _dirty_scope_pass(_git_status_paths()),
        "preregistration_sha_bound_to_invocation": invocation.get(
            "preregistration_sha256"
        )
        == _sha(PREREG_PATH),
        "render_command_exact_and_receipt_bound": invocation.get("command")
        == expected_render_command
        == receipt.get("command"),
        "recording_projection_compare_command_and_result_exact": automated.get(
            "recording_only", {}
        ).get("compare_command")
        == expected_compare_command
        == automated.get("recording_only", {}).get("compare", {}).get("command")
        and automated.get("recording_only", {}).get("compare", {}).get("returncode") == 0
        and automated.get("recording_only", {}).get("compare", {}).get("ok") is True,
        "presentation_merge_command_and_result_exact": automated.get("presentation", {}).get(
            "merge_command"
        )
        == expected_merge_command
        == automated.get("presentation", {}).get("merge", {}).get("command")
        and automated.get("presentation", {}).get("merge", {}).get("returncode") == 0
        and automated.get("presentation", {}).get("merge", {}).get("ok") is True,
        "host_render_gate_and_capability_exact": invocation.get("host_render_gate")
        == {HOST_RENDER_ENV: "1"}
        and invocation.get("host_capability_probe", {}).get("attempt_count") == 1
        and invocation.get("host_capability_probe", {}).get("bind_succeeded") is True
        and invocation.get("host_capability_probe", {}).get("listen_succeeded") is True
        and invocation.get("host_capability_probe", {}).get("pass") is True,
        "manual_check_keys_exact": sorted(manual_checks) == sorted(MANUAL_CHECK_KEYS),
        "manual_artifact_and_case_exact": manual.get("artifact")
        == "D369_MANUAL_VISUAL_INSPECTION_V1"
        and manual.get("case") == "g0a_d369",
        "manual_checks_all_pass": all(manual_checks.get(key) is True for key in MANUAL_CHECK_KEYS),
        "manual_pass": manual.get("pass") is True,
        "manual_exact_two_unique_png_paths": len(manual_files) == 2
        and observed_manual_paths == expected_manual_paths,
        "manual_png_hash_bytes_dimensions_exact": manual_records_match,
        "manual_png_records_match_automated": {
            row["path"]: row["sha256"] for row in manual_files
        }
        == {
            automated["rerun_png"]["path"]: automated["rerun_png"]["sha256"],
            automated["professor_board"]["path"]: automated["professor_board"]["sha256"],
        },
        "manual_markdown_nonempty_and_bound_to_both_pngs": len(manual_md.strip()) >= 200
        and _rel(RERUN_PNG_PATH) in manual_md
        and _sha(RERUN_PNG_PATH) in manual_md
        and _rel(SUMMARY_PNG_PATH) in manual_md
        and _sha(SUMMARY_PNG_PATH) in manual_md,
        "source_inputs_unchanged": _source_manifest() == prereg["source_manifest"],
        "base_copy_bitexact": _sha(BASE_COPY_PATH) == EXPECTED_RRD_SHA256
        and BASE_COPY_PATH.stat().st_size == EXPECTED_RRD_BYTES,
        "automated_file_records_current": automated_records_match,
        "validation_inventory_exact_and_current": set(validation_files)
        == expected_validation_paths
        and validation_records_match,
        "facts_and_science_boundary_preserved": automated.get("facts") == prereg.get("facts")
        and automated.get("science_boundary") == prereg.get("science_boundary")
        and all(_facts_exact(automated["facts"]).values()),
        "one_render_no_retry": invocation.get("render_invocation_count") == 1
        and invocation.get("automatic_retry_count") == 0
        and invocation.get("no_retry") is True,
        "render_returncode_and_error_signatures_clean": receipt.get("returncode") == 0
        and receipt.get("unknown_timeline_console_text_absent") is True
        and receipt.get("message_proxy_operation_not_permitted_pair_absent") is True,
        "receipt_png_record_current": receipt.get("screenshot_exists") is True
        and receipt.get("screenshot_bytes") == RERUN_PNG_PATH.stat().st_size
        and receipt.get("screenshot_sha256") == _sha(RERUN_PNG_PATH),
        "phase_sequence_exact_and_forward": phase_names == EXPECTED_PHASE_SEQUENCE
        and all(isinstance(value, (int, float)) for value in phase_times)
        and all(first <= second for first, second in zip(phase_times, phase_times[1:])),
        "forbidden_runtime_modules_absent": not _forbidden_modules(),
        "summary_exact_1920x1080": _png_dimensions(SUMMARY_PNG_PATH) == [1920, 1080],
        "scope_guards_exact": prereg.get("scope_guards")
        == automated.get("scope_guards")
        == SCOPE_GUARDS,
    }
    visualization_pass = all(checks.values())
    artifacts = {}
    for name in sorted(expected_inventory):
        path = OUT_DIR / name
        artifacts[_rel(path)] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    completion = {
        "artifact": "D369_COMPLETION_SUMMARY_V1",
        "case": "g0a_d369",
        "checks": checks,
        "visualization_pass": visualization_pass,
        "pass": visualization_pass,
        "verdict": (
            "D369_D368_PROFESSOR_VISUAL_CONTRACT_REPAIRED_OBSERVABILITY_ONLY"
            if visualization_pass
            else "D369_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"
        ),
        "d368_measurement_pass_preserved": prereg["facts"]["measurement_pass"],
        "d368_measurement_verdict_preserved": prereg["facts"]["measurement_verdict"],
        "d368_measurement_recomputed": False,
        "science_boundary": prereg["science_boundary"],
        "scope_guards": SCOPE_GUARDS,
        "render_invocation_count": invocation["render_invocation_count"],
        "automatic_retry_count": invocation["automatic_retry_count"],
        "artifacts": artifacts,
        "source_manifest_after": _source_manifest(),
        "next_science_or_collider_candidate_requires_new_approval": True,
        "commit_or_push_performed": False,
        "git_base_unchanged": checks["git_head_origin_and_preregistered_head_exact"],
    }
    _write_json_x(COMPLETION_PATH, completion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "render", "finalize"), required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        _prepare()
    elif args.stage == "render":
        _render()
    else:
        _finalize()


if __name__ == "__main__":
    main()
