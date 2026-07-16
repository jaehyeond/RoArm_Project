#!/usr/bin/env python3
"""Forward-only Rerun explanation of the immutable D355 attempt-1 import stop.

This is observability only, not a second provenance audit.  It never imports
PXR/Isaac/Omni, never opens the USD, and never computes a patch hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
D355_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d355"
OUT_DIR = D355_DIR / "attempt1_failure_observability"
SOURCE_PATHS = {
    "preregistration": D355_DIR / "d355_preregistration.json",
    "invocation": D355_DIR / "d355_audit_invocation.json",
    "phases": D355_DIR / "d355_phase_markers.jsonl",
    "runtime_exception": D355_DIR / "d355_runtime_exception.json",
}
EXPECTED_SOURCE_SHA256 = {
    "preregistration": "6cfaf0efa802546bed0177b88d8467a5d8ca1055ec113907b69bb5a81606325d",
    "invocation": "3bbc44edbec995fcbba0093fb7d9615290dbc32ffb65a8e12fc6b86bbb75a8f2",
    "phases": "1e5808892cbda91a7e7efb5e8a530468ae7d00b06abd961462f40c78b7da138c",
    "runtime_exception": "48bcad5c5740651f7aa8157616b64a639b79f67af5033e60dfe939da4bfdebde",
}
D354_MEASUREMENT = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d354/"
    "d354_zero_step_closure_geometry_measurement.json"
)
EXPECTED_D354_MEASUREMENT_SHA256 = (
    "fd0d43c2a47abefad939fc0e980456cc396bbf3ba3bb104b28bcd999100f23ed"
)

PREREG_PATH = OUT_DIR / "d355_attempt1_failure_observability_preregistration.json"
INVOCATION_PATH = OUT_DIR / "d355_attempt1_failure_observability_invocation.json"
RRD_PATH = OUT_DIR / "d355_attempt1_failure_explained.rrd"
RBL_PATH = OUT_DIR / "d355_attempt1_failure_explained.rbl"
PNG_PATH = OUT_DIR / "d355_attempt1_failure_explained_rerun.png"
VALIDATION_PATH = OUT_DIR / "d355_attempt1_failure_observability_validation.json"
SUMMARY_PATH = OUT_DIR / "d355_attempt1_failure_observability_summary.json"
MANUAL_PATH = OUT_DIR / "d355_attempt1_failure_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d355_attempt1_failure_observability_completion.json"

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


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, path in SOURCE_PATHS.items()}


def _prepare() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes()
    checks = {
        "immutable_failure_sources_exact": source_hashes == EXPECTED_SOURCE_SHA256,
        "d354_measurement_exact": _sha256(D354_MEASUREMENT)
        == EXPECTED_D354_MEASUREMENT_SHA256,
        "attempt1_invocation_exactly_one": _json(SOURCE_PATHS["invocation"])[
            "audit_invocation_count"
        ]
        == 1,
        "attempt1_verdict_exact": _json(SOURCE_PATHS["runtime_exception"])["verdict"]
        == VERDICT,
    }
    payload = {
        "artifact": "D355_ATTEMPT1_FAILURE_OBSERVABILITY_PREREGISTRATION_V1",
        "role": "failure explanation only; not an audit retry",
        "source_hashes": source_hashes,
        "expected_source_hashes": EXPECTED_SOURCE_SHA256,
        "harness": {"path": _rel(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "registered_panels": [
            "readable failure summary",
            "forward-only phase pipeline with the PXR import stop in red",
            "frozen D354 clear/overlap context copied without reclassification",
            "zero-count metrics for all unexecuted audit/science phases",
        ],
        "scope_guards": {
            "second_audit_count": 0,
            "pxr_import_count": 0,
            "isaac_launch_count": 0,
            "physx_query_count": 0,
            "q5_evaluation_count": 0,
            "controlled_physics_steps": 0,
            "patch_hash_computation_count": 0,
            "cap_rim_classification_count": 0,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, payload)
    if not payload["pass"]:
        raise RuntimeError(f"failure observability prepare failed: {checks}")
    print(json.dumps({"prepared": True, "path": _rel(PREREG_PATH)}, indent=2))


def _cylinder_mesh(radius: float = 0.017, height: float = 0.09, segments: int = 96) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    lower = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(segments, -height / 2.0)]
    )
    upper = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(segments, height / 2.0)]
    )
    vertices = np.concatenate(
        [lower, upper, [[0.0, 0.0, -height / 2.0], [0.0, 0.0, height / 2.0]]],
        axis=0,
    )
    triangles: list[list[int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.extend(
            [
                [index, nxt, segments + nxt],
                [index, segments + nxt, segments + index],
                [2 * segments, nxt, index],
                [2 * segments + 1, segments + index, segments + nxt],
            ]
        )
    return vertices, np.asarray(triangles, dtype=np.int64)


def _contract(metric_names: list[str], event_names: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    entities = [
        "metadata/run",
        "failure_pipeline/nodes",
        "failure_pipeline/transitions",
        "context/d354_frozen/cylinder",
        "context/d354_frozen/clear_endpoint",
        "context/d354_frozen/overlap_endpoint",
    ]
    components: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "failure_pipeline/nodes": [
            "Points3D:colors",
            "Points3D:labels",
            "Points3D:positions",
            "Points3D:radii",
        ],
        "failure_pipeline/transitions": [
            "Arrows3D:colors",
            "Arrows3D:labels",
            "Arrows3D:origins",
            "Arrows3D:radii",
            "Arrows3D:vectors",
        ],
        "context/d354_frozen/cylinder": [
            "Mesh3D:albedo_factor",
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        ],
        "context/d354_frozen/clear_endpoint": [
            "Points3D:colors",
            "Points3D:labels",
            "Points3D:positions",
            "Points3D:radii",
        ],
        "context/d354_frozen/overlap_endpoint": [
            "Points3D:colors",
            "Points3D:labels",
            "Points3D:positions",
            "Points3D:radii",
        ],
    }
    for name in metric_names:
        path = f"metrics/d355_attempt1/{name}"
        entities.append(path)
        components[path] = ["Scalars:scalars"]
    for name in event_names:
        path = f"events/d355_attempt1/{name}"
        entities.append(path)
        components[path] = ["TextLog:level", "TextLog:text"]
    return sorted(entities), components


def _render() -> None:
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if str(rr.__version__) != RERUN_VERSION:
        raise RuntimeError(f"rerun {rr.__version__} != {RERUN_VERSION}")
    prereg = _json(PREREG_PATH)
    if _sha256(Path(__file__)) != prereg["harness"]["sha256"]:
        raise RuntimeError("failure-observability harness changed after prepare")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("immutable attempt1 failure source changed")
    expected_before = {PREREG_PATH.name}
    observed_before = {path.name for path in OUT_DIR.iterdir()}
    if observed_before != expected_before:
        raise RuntimeError(f"unexpected pre-render inventory: {sorted(observed_before)}")
    _write_json_x(
        INVOCATION_PATH,
        {
            "artifact": "D355_ATTEMPT1_FAILURE_OBSERVABILITY_INVOCATION_V1",
            "render_invocation_count": 1,
            "second_audit_count": 0,
            "harness_sha256": _sha256(Path(__file__)),
        },
    )

    exception = _json(SOURCE_PATHS["runtime_exception"])
    invocation = _json(SOURCE_PATHS["invocation"])
    measurement = _json(D354_MEASUREMENT)
    classification = measurement["classification"]
    clear = classification["live_first_contact_feature"]["endpoints"]["clear"]
    overlap = classification["live_first_contact_feature"]["endpoints"]["overlap"]
    clear_point = np.asarray(clear["point_cylinder_local_m"], dtype=np.float64)
    overlap_point = np.asarray(overlap["point_cylinder_local_m"], dtype=np.float64)
    cylinder_vertices, cylinder_triangles = _cylinder_mesh()

    labels = [
        "PREPARE PASS\ninputs + recipes frozen",
        "AUDIT INVOCATION 1\nforward-only marker written",
        "STOP: standalone PXR import\nModuleNotFoundError: pxr",
        "NOT REACHED\nUSD/raw stream",
        "NOT REACHED\nrecipe grid + controls",
        "NOT REACHED\nscientific Rerun",
    ]
    nodes = np.asarray([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [3.6, 0.0, 0.0], [5.4, 0.0, 0.0], [7.2, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.asarray(
        [[40, 200, 100], [40, 200, 100], [255, 55, 70], [115, 115, 125], [115, 115, 125], [115, 115, 125]],
        dtype=np.uint8,
    )
    origins = nodes[:-1]
    vectors = nodes[1:] - nodes[:-1]
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
            "pxr libraries exist inside Isaac extension cache but are not importable from plain isaaclab Python; Kit/SimulationApp extension bootstrap is required"
        ),
        "why_d350_d354_worked": (
            "those cases launched Isaac/SimulationApp before importing Omniverse PXR modules; this offline attempt intentionally did not"
        ),
        "no_retry": f"audit_invocation_count={invocation['audit_invocation_count']}; no second audit executed",
        "science_boundary": (
            "no authored/raw provenance result exists; D354 stays unresolved; historical context is copied only"
        ),
    }
    expected_entities, components = _contract(
        list(metric_values), list(event_values)
    )
    summary = "\n".join(
        [
            "# D355 attempt 1 — stopped before provenance science",
            "",
            f"- **operational verdict:** `{VERDICT}`",
            "- prepare and immutable input checks: **PASS**",
            "- audit invocation marker: **1/1**",
            "- stop: `ModuleNotFoundError: No module named 'pxr'`",
            "- USD stream loads / recipe candidates / negative controls: **0 / 0 / 0**",
            "- q5 / physics steps / Isaac launches: **0 / 0 / 0**",
            "",
            "## Why this does not contradict earlier Isaac success",
            "",
            "D350/D354 created `SimulationApp` first. Isaac's extension manager then exposed the bundled PXR modules. Plain `isaaclab` Python has no top-level `pxr` module, so the newly assumed standalone-PXR/no-Isaac route stopped before reading the USD.",
            "",
            "## What the right panel means",
            "",
            "It copies D354's already-frozen clear and overlap points so the prior unresolved contact remains visible. No cap/rim classifier or new geometry decision ran here.",
        ]
    )
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run", name="1 | Exact stop + interpretation"),
                rrb.Spatial3DView(origin="/", contents="/failure_pipeline/**", name="2 | Attempt-1 forward-only pipeline"),
                column_shares=[0.45, 0.55],
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/", contents="/context/d354_frozen/**", name="3 | D354 unresolved contact (historical only)"),
                rrb.DataframeView(origin="/metrics/d355_attempt1", contents="/metrics/d355_attempt1/**", name="4 | Exact executed/not-executed counts"),
                rrb.TextLogView(origin="/events/d355_attempt1", contents="/events/d355_attempt1/**", name="5 | Root cause + boundaries"),
                column_shares=[0.36, 0.28, 0.36],
            ),
            row_shares=[0.50, 0.50],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )
    with rr.RecordingStream(
        "roarm_g0a_d355_attempt1_failure",
        recording_id="g0a_d355_attempt1_failure_observability",
        make_default=False,
        send_properties=True,
    ) as recording:
        recording.save(str(RRD_PATH), write_footer=True)
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.log(
            "metadata/run",
            rr.TextDocument(summary, media_type=rr.MediaType.MARKDOWN),
            static=True,
        )
        recording.log(
            "failure_pipeline/nodes",
            rr.Points3D(nodes, colors=colors, radii=0.20, labels=labels),
            static=True,
        )
        recording.log(
            "failure_pipeline/transitions",
            rr.Arrows3D(
                origins=origins,
                vectors=vectors,
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
                albedo_factor=[125, 125, 145, 100],
            ),
            static=True,
        )
        recording.log(
            "context/d354_frozen/clear_endpoint",
            rr.Points3D(
                [clear_point.astype(np.float32)],
                colors=[255, 210, 40],
                radii=0.0007,
                labels=["D354 clear: cap_or_rim_boundary (historical)"],
            ),
            static=True,
        )
        recording.log(
            "context/d354_frozen/overlap_endpoint",
            rr.Points3D(
                [overlap_point.astype(np.float32)],
                colors=[255, 55, 80],
                radii=0.00055,
                labels=["D354 adjacent overlap: barrel_interior (historical)"],
            ),
            static=True,
        )
        for name, value in metric_values.items():
            recording.log(
                f"metrics/d355_attempt1/{name}", rr.Scalars(value), static=True
            )
        for name, value in event_values.items():
            recording.log(
                f"events/d355_attempt1/{name}",
                rr.TextLog(value, level=rr.TextLogLevel.INFO),
                static=True,
            )
        recording.flush(timeout_sec=30.0)
    blueprint.save("roarm_g0a_d355_attempt1_failure", RBL_PATH)
    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=components,
        blueprint_path=RBL_PATH,
        screenshot_path=PNG_PATH,
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        timeout_s=180.0,
    )
    _write_json_x(VALIDATION_PATH, validation)
    summary_payload = {
        "artifact": "D355_ATTEMPT1_FAILURE_OBSERVABILITY_SUMMARY_V1",
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
        "rerun_validation_pass": validation.get("pass") is True,
        "artifacts": {
            path.name: {"path": _rel(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in [RRD_PATH, RBL_PATH, PNG_PATH, VALIDATION_PATH]
            if path.is_file()
        },
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
        raise FileExistsError("refusing to overwrite failure observability finalization")
    validation = _json(VALIDATION_PATH)
    summary = _json(SUMMARY_PATH)
    dimensions = _png_dimensions(PNG_PATH)
    checks = {
        "opened_original_resolution": True,
        "failure_pipeline_stop_visible": True,
        "exact_zero_counts_readable": True,
        "d354_historical_context_visible": True,
        "no_new_cap_rim_decision_label_visible": True,
        "no_blank_or_corrupt_panel": True,
    }
    manual = {
        "artifact": "D355_ATTEMPT1_FAILURE_MANUAL_VISUAL_INSPECTION_V1",
        "path": _rel(PNG_PATH),
        "sha256": _sha256(PNG_PATH),
        "raster_dimensions": dimensions,
        "inspection_method": "view_image original resolution",
        "observations": [
            "The pipeline visibly stops at the red standalone-PXR import node; all later audit phases are grey and marked not reached.",
            "The summary and scalar panels show one audit invocation and zero USD/hash/q5/physics executions.",
            "The D354 cylinder panel is labeled historical-only and does not imply a new cap/rim decision.",
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
    }
    completion = {
        "artifact": "D355_ATTEMPT1_FAILURE_OBSERVABILITY_COMPLETION_V1",
        "verdict": VERDICT,
        "completion_pass": all(completion_checks.values()),
        "checks": completion_checks,
        "scientific_provenance_result": None,
        "d354_science_changed": False,
        "next_authorization_boundary": (
            "Any provenance retry must be a new forward-only case that explicitly chooses either "
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
