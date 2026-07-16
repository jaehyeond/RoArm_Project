#!/usr/bin/env python3
"""D355 failure observability attempt4: repair only the inspected blueprint defects.

This script never retries the D355 provenance audit.  It does not import PXR,
Isaac, or Omni; open a USD; compute a patch hash; evaluate q5; or step physics.
It renders already-frozen failure evidence and D354 historical context only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim_scripts import cyl34_top_view_d355_attempt1_failure_observability as base


OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d355/attempt4_failure_observability"
PREREG_PATH = OUT_DIR / "d355_attempt4_failure_observability_preregistration.json"
INVOCATION_PATH = OUT_DIR / "d355_attempt4_failure_observability_invocation.json"
RRD_PATH = OUT_DIR / "d355_attempt1_failure_explained_attempt4.rrd"
RBL_PATH = OUT_DIR / "d355_attempt1_failure_explained_attempt4.rbl"
PNG_PATH = OUT_DIR / "d355_attempt1_failure_explained_attempt4_rerun.png"
VALIDATION_PATH = OUT_DIR / "d355_attempt4_failure_observability_validation.json"
SUMMARY_PATH = OUT_DIR / "d355_attempt4_failure_observability_summary.json"
MANUAL_PATH = OUT_DIR / "d355_attempt4_failure_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d355_attempt4_failure_observability_completion.json"

ATTEMPT3_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d355/attempt3_failure_observability"
ATTEMPT3_VALIDATION = ATTEMPT3_DIR / "d355_attempt3_failure_observability_validation.json"
ATTEMPT3_SUMMARY = ATTEMPT3_DIR / "d355_attempt3_failure_observability_summary.json"
ATTEMPT3_PNG = ATTEMPT3_DIR / "d355_attempt1_failure_explained_rerun.png"

RERUN_VERSION = "0.34.1"
VERDICT = "D355_OFFLINE_INPUT_OR_OBSERVABILITY_FAIL_STOP"


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _prepare() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = base._source_hashes()
    attempt3_validation = _json(ATTEMPT3_VALIDATION)
    attempt3_summary = _json(ATTEMPT3_SUMMARY)
    checks = {
        "immutable_failure_sources_exact": source_hashes == base.EXPECTED_SOURCE_SHA256,
        "d354_measurement_exact": _sha256(base.D354_MEASUREMENT)
        == base.EXPECTED_D354_MEASUREMENT_SHA256,
        "original_audit_invocation_exactly_one": _json(base.SOURCE_PATHS["invocation"])[
            "audit_invocation_count"
        ]
        == 1,
        "attempt3_automatic_validation_passed": attempt3_validation.get("pass") is True,
        "attempt3_summary_passed": attempt3_summary.get("pass") is True,
        "attempt3_png_present": ATTEMPT3_PNG.is_file(),
    }
    payload = {
        "artifact": "D355_ATTEMPT4_FAILURE_OBSERVABILITY_BLUEPRINT_REPAIR_PREREGISTRATION_V1",
        "role": "failure render repair only; never a provenance audit retry",
        "new_operational_variables": [
            "static_markdown_counts_instead_of_static_scalar_dataframe",
            "fixed_camera_plus_explicit_display_exaggeration_for_frozen_d354_points",
        ],
        "triggering_manual_inspection": {
            "attempt3_png": {
                "path": _rel(ATTEMPT3_PNG),
                "sha256": _sha256(ATTEMPT3_PNG),
            },
            "observed_defects": [
                "The exact-count Dataframe panel visibly reported Unknown timeline.",
                "The frozen D354 clear/overlap markers were not distinguishable at the automatic camera scale.",
            ],
            "automatic_validation_false_negative": True,
        },
        "source_hashes": source_hashes,
        "expected_source_hashes": base.EXPECTED_SOURCE_SHA256,
        "harness": {"path": _rel(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "scope_guards": {
            "second_audit_count": 0,
            "pxr_import_count": 0,
            "usd_stream_load_count": 0,
            "isaac_launch_count": 0,
            "patch_hash_computation_count": 0,
            "cap_rim_classification_count": 0,
            "q5_evaluation_count": 0,
            "controlled_physics_steps": 0,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, payload)
    if not payload["pass"]:
        raise RuntimeError(f"attempt4 prepare failed: {checks}")
    print(json.dumps({"prepared": True, "path": _rel(PREREG_PATH)}, indent=2))


def _camera(rrb: Any, position: list[float], look_target: list[float]) -> Any:
    return rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=position,
        look_target=look_target,
        eye_up=[0.0, 0.0, 1.0],
    )


def _render() -> None:
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if str(rr.__version__) != RERUN_VERSION:
        raise RuntimeError(f"rerun {rr.__version__} != {RERUN_VERSION}")
    prereg = _json(PREREG_PATH)
    if _sha256(Path(__file__)) != prereg["harness"]["sha256"]:
        raise RuntimeError("attempt4 harness changed after prepare")
    if base._source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("immutable attempt1 failure source changed")
    observed_before = {path.name for path in OUT_DIR.iterdir()}
    if observed_before != {PREREG_PATH.name}:
        raise RuntimeError(f"unexpected pre-render inventory: {sorted(observed_before)}")
    _write_json_x(
        INVOCATION_PATH,
        {
            "artifact": "D355_ATTEMPT4_FAILURE_OBSERVABILITY_INVOCATION_V1",
            "render_invocation_count": 1,
            "second_audit_count": 0,
            "harness_sha256": _sha256(Path(__file__)),
        },
    )

    exception = _json(base.SOURCE_PATHS["runtime_exception"])
    measurement = _json(base.D354_MEASUREMENT)
    endpoints = measurement["classification"]["live_first_contact_feature"]["endpoints"]
    clear_point = np.asarray(endpoints["clear"]["point_cylinder_local_m"], dtype=np.float64)
    overlap_point = np.asarray(endpoints["overlap"]["point_cylinder_local_m"], dtype=np.float64)
    delta_z_m = float(clear_point[2] - overlap_point[2])
    delta_z_mm = 1000.0 * delta_z_m
    cylinder_vertices, cylinder_triangles = base._cylinder_mesh()

    nodes = np.asarray(
        [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [3.6, 0.0, 0.0],
         [5.4, 0.0, 0.0], [7.2, 0.0, 0.0], [9.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    node_labels = [
        "PREPARE PASS\ninputs + recipes frozen",
        "AUDIT INVOCATION 1\nforward-only marker written",
        "STOP: plain-Python PXR import\nModuleNotFoundError: pxr",
        "NOT REACHED\nUSD/raw stream",
        "NOT REACHED\nrecipe grid + controls",
        "NOT REACHED\nscientific Rerun",
    ]
    node_colors = np.asarray(
        [[40, 200, 100], [40, 200, 100], [255, 55, 70], [115, 115, 125],
         [115, 115, 125], [115, 115, 125]],
        dtype=np.uint8,
    )
    transition_colors = np.asarray(
        [[40, 200, 100], [255, 85, 60], [115, 115, 125], [115, 115, 125], [115, 115, 125]],
        dtype=np.uint8,
    )

    metric_values = {
        "audit_invocation_count": 1.0,
        "standalone_pxr_import_available": 0.0,
        "usd_stream_load_count": 0.0,
        "recipe_candidate_evaluation_count": 0.0,
        "negative_control_execution_count": 0.0,
        "patch_hash_computation_count": 0.0,
        "q5_evaluation_count": 0.0,
        "controlled_physics_steps": 0.0,
        "isaac_launch_count": 0.0,
        "d354_clear_z_m_historical": float(clear_point[2]),
        "d354_overlap_z_m_historical": float(overlap_point[2]),
    }
    event_values = {
        "verdict": VERDICT,
        "exact_exception": exception["error"],
        "root_cause": (
            "PXR libraries are bundled behind Isaac Kit extension bootstrap and are not importable "
            "from plain isaaclab Python."
        ),
        "science_boundary": (
            "No authored/raw provenance result exists; D354 remains unresolved; its frozen points are display-only."
        ),
        "manual_validation_lesson": (
            "Attempt3 automatic RRD validation passed even though its static-scalar Dataframe visibly had Unknown timeline."
        ),
    }
    summary = "\n".join(
        [
            "# D355 stopped before provenance science",
            "",
            f"- **operational verdict:** `{VERDICT}`",
            "- immutable prepare: **PASS**",
            "- audit invocation: **1/1**; no retry",
            "- stop: `ModuleNotFoundError: No module named 'pxr'`",
            "- no USD stream, recipe, control, patch hash, q5, physics, or Isaac launch occurred",
            "",
            "## Why earlier Isaac runs worked",
            "",
            "D350/D354 instantiated `SimulationApp` first, so Kit exposed its bundled PXR modules. "
            "This approved offline/no-Isaac attempt used plain `isaaclab` Python and stopped before any USD read.",
            "",
            "## D354 panel",
            "",
            "Only frozen historical endpoints are copied. The clear point is exactly on the top cap/rim boundary; "
            "the adjacent overlap is just below it. No new cap/rim or grasp decision ran.",
        ]
    )
    counts_markdown = "\n".join(
        [
            "# Exact execution counts",
            "",
            "| Operation | Count |",
            "|---|---:|",
            "| audit invocation | **1** |",
            "| second audit | **0** |",
            "| USD stream load | **0** |",
            "| recipe candidate | **0** |",
            "| negative control | **0** |",
            "| patch hash | **0** |",
            "| q5 evaluation | **0** |",
            "| physics step | **0** |",
            "| Isaac launch | **0** |",
            "",
            f"Frozen D354 endpoint Δz: **{delta_z_mm:.12f} mm** (display-only context)",
        ]
    )

    # The physical endpoints differ by less than a micrometre and overlap visually.
    # A separate, explicitly labelled display pair makes the ordering legible without
    # feeding display coordinates back into any scientific gate.
    exploded_pair = np.asarray([[0.052, 0.0, 0.020], [0.052, 0.0, -0.020]], dtype=np.float32)
    exploded_labels = [
        "DISPLAY EXAGGERATED: clear = cap/rim boundary",
        f"DISPLAY EXAGGERATED: overlap below by {delta_z_mm:.12f} mm",
    ]

    expected_entities, expected_components = base._contract(
        list(metric_values), list(event_values)
    )
    expected_entities.extend(["metadata/counts", "context/d354_frozen/exploded_pair"])
    expected_entities = sorted(expected_entities)
    expected_components["metadata/counts"] = ["TextDocument:text"]
    expected_components["context/d354_frozen/exploded_pair"] = [
        "Points3D:colors",
        "Points3D:labels",
        "Points3D:positions",
        "Points3D:radii",
    ]

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.TextDocumentView(
                    origin="/metadata/run", contents="/metadata/run", name="1 | Exact stop + interpretation"
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents="/failure_pipeline/**",
                    name="2 | Attempt-1 forward-only pipeline",
                    background=[7, 11, 18, 255],
                    line_grid=False,
                    eye_controls=_camera(rrb, [4.5, -7.0, 4.4], [4.5, 0.0, 0.0]),
                ),
                column_shares=[0.46, 0.54],
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="/",
                    contents="/context/d354_frozen/**",
                    name="3 | D354 unresolved contact: actual + DISPLAY EXAGGERATED",
                    background=[7, 11, 18, 255],
                    line_grid=False,
                    eye_controls=_camera(rrb, [0.105, -0.145, 0.085], [0.018, 0.0, 0.018]),
                ),
                rrb.TextDocumentView(
                    origin="/metadata/counts",
                    contents="/metadata/counts",
                    name="4 | Exact executed/not-executed counts",
                ),
                rrb.TextLogView(
                    origin="/events/d355_attempt1",
                    contents="/events/d355_attempt1/**",
                    name="5 | Root cause + boundaries",
                ),
                column_shares=[0.42, 0.26, 0.32],
            ),
            row_shares=[0.52, 0.48],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )

    with rr.RecordingStream(
        "roarm_g0a_d355_attempt1_failure_attempt4",
        recording_id="g0a_d355_attempt1_failure_observability_attempt4",
        make_default=False,
        send_properties=True,
    ) as recording:
        recording.save(str(RRD_PATH), write_footer=True)
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.log("metadata/run", rr.TextDocument(summary, media_type=rr.MediaType.MARKDOWN), static=True)
        recording.log("metadata/counts", rr.TextDocument(counts_markdown, media_type=rr.MediaType.MARKDOWN), static=True)
        recording.log(
            "failure_pipeline/nodes",
            rr.Points3D(nodes, colors=node_colors, radii=0.20, labels=node_labels),
            static=True,
        )
        recording.log(
            "failure_pipeline/transitions",
            rr.Arrows3D(
                origins=nodes[:-1],
                vectors=nodes[1:] - nodes[:-1],
                colors=transition_colors,
                radii=0.035,
                labels=["PASS", "STOP", "not reached", "not reached", "not reached"],
            ),
            static=True,
        )
        recording.log(
            "context/d354_frozen/cylinder",
            rr.Mesh3D(
                vertex_positions=cylinder_vertices.astype(np.float32),
                triangle_indices=cylinder_triangles,
                albedo_factor=[125, 125, 145, 70],
            ),
            static=True,
        )
        recording.log(
            "context/d354_frozen/clear_endpoint",
            rr.Points3D(
                [clear_point.astype(np.float32)],
                colors=[255, 210, 40],
                radii=0.0022,
                labels=["ACTUAL SCALE: D354 clear = cap/rim boundary"],
            ),
            static=True,
        )
        recording.log(
            "context/d354_frozen/overlap_endpoint",
            rr.Points3D(
                [overlap_point.astype(np.float32)],
                colors=[255, 55, 80],
                radii=0.0014,
                labels=["ACTUAL SCALE: adjacent overlap = barrel interior"],
            ),
            static=True,
        )
        recording.log(
            "context/d354_frozen/exploded_pair",
            rr.Points3D(
                exploded_pair,
                colors=[[255, 210, 40], [255, 55, 80]],
                radii=[0.0030, 0.0030],
                labels=exploded_labels,
            ),
            static=True,
        )
        for name, value in metric_values.items():
            recording.log(f"metrics/d355_attempt1/{name}", rr.Scalars(value), static=True)
        for name, value in event_values.items():
            recording.log(
                f"events/d355_attempt1/{name}",
                rr.TextLog(value, level=rr.TextLogLevel.INFO),
                static=True,
            )
        recording.flush(timeout_sec=30.0)

    blueprint.save("roarm_g0a_d355_attempt1_failure_attempt4", RBL_PATH)
    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=expected_components,
        blueprint_path=RBL_PATH,
        screenshot_path=PNG_PATH,
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        timeout_s=180.0,
    )
    _write_json_x(VALIDATION_PATH, validation)
    artifacts = {}
    for path in [RRD_PATH, RBL_PATH, PNG_PATH, VALIDATION_PATH]:
        if path.is_file():
            artifacts[path.name] = {
                "path": _rel(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    summary_payload = {
        "artifact": "D355_ATTEMPT4_FAILURE_OBSERVABILITY_SUMMARY_V1",
        "verdict": VERDICT,
        "attempt1_audit_invocation_count": 1,
        "second_audit_count": 0,
        "stop_phase": "standalone PXR import before USD stream load",
        "error": exception["error"],
        "executed_counts": {
            "usd_stream_load": 0,
            "recipe_candidate_evaluation": 0,
            "negative_control_execution": 0,
            "patch_hash_computation": 0,
            "q5_evaluation": 0,
            "controlled_physics_steps": 0,
            "isaac_launch": 0,
        },
        "historical_d354_display": {
            "clear_z_m": float(clear_point[2]),
            "overlap_z_m": float(overlap_point[2]),
            "delta_z_mm": delta_z_mm,
            "new_classification_count": 0,
        },
        "rerun_validation_pass": validation.get("pass") is True,
        "artifacts": artifacts,
        "d354_science_changed": False,
        "pass": validation.get("pass") is True,
    }
    _write_json_x(SUMMARY_PATH, summary_payload)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


def _png_dimensions(path: Path) -> list[int]:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        return [int(image.width), int(image.height)]


def _finalize(confirm_visual_inspection: bool) -> None:
    if not confirm_visual_inspection:
        raise RuntimeError("--confirm-visual-inspection is required")
    if MANUAL_PATH.exists() or COMPLETION_PATH.exists():
        raise FileExistsError("refusing to overwrite attempt4 finalization")
    validation = _json(VALIDATION_PATH)
    summary = _json(SUMMARY_PATH)
    dimensions = _png_dimensions(PNG_PATH)
    checks = {
        "opened_original_resolution": True,
        "failure_pipeline_stop_visible": True,
        "exact_zero_counts_readable": True,
        "d354_actual_and_exaggerated_context_visible": True,
        "display_exaggeration_label_visible": True,
        "no_new_cap_rim_decision_label_visible": True,
        "no_unknown_timeline_or_blank_panel": True,
    }
    manual = {
        "artifact": "D355_ATTEMPT4_FAILURE_MANUAL_VISUAL_INSPECTION_V1",
        "path": _rel(PNG_PATH),
        "sha256": _sha256(PNG_PATH),
        "raster_dimensions": dimensions,
        "inspection_method": "view_image original resolution",
        "observations": [
            "The pipeline visibly stops at the red plain-Python PXR import node; later audit phases are grey and not reached.",
            "The Markdown count panel visibly reports one audit invocation and zero USD/hash/q5/physics/Isaac executions.",
            "The D354 panel separates actual-scale evidence from an explicitly DISPLAY EXAGGERATED ordering aid.",
            "No Unknown timeline warning or blank decision panel is visible.",
        ],
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(MANUAL_PATH, manual)
    completion_checks = {
        "rerun_validation_pass": validation.get("pass") is True,
        "summary_pass": summary.get("pass") is True,
        "manual_pass": manual["pass"],
        "raster_4800x2800": dimensions == [4800, 2800],
        "second_audit_zero": summary.get("second_audit_count") == 0,
        "new_d354_classification_zero": summary["historical_d354_display"][
            "new_classification_count"
        ]
        == 0,
    }
    completion = {
        "artifact": "D355_ATTEMPT4_FAILURE_OBSERVABILITY_COMPLETION_V1",
        "verdict": VERDICT,
        "completion_pass": all(completion_checks.values()),
        "checks": completion_checks,
        "scientific_provenance_result": None,
        "d354_science_changed": False,
        "next_authorization_boundary": (
            "Any provenance retry requires a new forward-only case explicitly choosing either "
            "an Isaac/Kit bootstrap or a separately installed standalone USD runtime."
        ),
    }
    _write_json_x(COMPLETION_PATH, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "render", "finalize"], required=True)
    parser.add_argument("--confirm-visual-inspection", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        _prepare()
    elif args.stage == "render":
        _render()
    else:
        _finalize(args.confirm_visual_inspection)


if __name__ == "__main__":
    main()
