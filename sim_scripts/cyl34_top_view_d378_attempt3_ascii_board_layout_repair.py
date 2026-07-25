#!/usr/bin/env python3
"""D378 attempt3: offline-only ASCII board layout repair.

The single D378 authority audit already completed in attempt2 and is immutable.
This repair reads that evidence and its passed Rerun artifacts, regenerates only
the exact 1920x1080 explanation board, and performs no authority recomputation,
Isaac/PhysX work, Rerun Viewer invocation, or robot/cylinder science.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

CASE_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d378"
ATTEMPT1 = (
    CASE_ROOT
    / "attempt1_ephemeral_identifier_provenance_and_workload_authority_repair"
)
ATTEMPT2 = CASE_ROOT / "attempt2_preregistration_status_order_repair"
OUT_DIR = CASE_ROOT / "attempt3_ascii_board_layout_repair"

PREREG_PATH = OUT_DIR / "d378_attempt3_preregistration.json"
PHASE_PATH = OUT_DIR / "d378_attempt3_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d378_attempt3_invocation.json"
REPAIR_EVIDENCE_PATH = OUT_DIR / "d378_attempt3_visual_repair_evidence.json"
BOARD_PATH = OUT_DIR / "d378_corrected_workload_authority_repaired_1920x1080.png"
LAYOUT_PATH = OUT_DIR / "d378_attempt3_layout_validation.json"
MANUAL_TEMPLATE_PATH = OUT_DIR / "d378_attempt3_manual_visual_inspection_template.json"
MANUAL_PATH = OUT_DIR / "d378_attempt3_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d378_final_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d378_attempt3_runtime_exception.json"

HARNESS = Path(__file__).resolve()
ORIGINAL_D378_HARNESS = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d378_d377_ephemeral_identifier_provenance_and_workload_authority_repair.py"
)
START_HERE = REPO / "START_HERE.md"

D375_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d375/"
    "attempt2_external_gpu_attestation_repair"
)
D377_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d377/"
    "attempt1_stagecache_erase_before_close_localization"
)
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"

A2_PREREG = ATTEMPT2 / "d378_preregistration.json"
A2_INVOCATION = ATTEMPT2 / "d378_offline_invocation.json"
A2_EVIDENCE = ATTEMPT2 / "d378_workload_authority_repair_evidence.json"
A2_AUTOMATED = ATTEMPT2 / "d378_automated_summary.json"
A2_MANUAL = ATTEMPT2 / "d378_manual_visual_inspection.json"
A2_COMPLETION = ATTEMPT2 / "d378_completion_summary.json"
A2_BOARD = ATTEMPT2 / "d378_corrected_workload_authority_1920x1080.png"
A2_RRD = ATTEMPT2 / "d378_workload_authority_repair.rrd"
A2_RBL = ATTEMPT2 / "d378_workload_authority_repair.rbl"
A2_RERUN_VALIDATION = ATTEMPT2 / "d378_rerun_validation.json"
A2_RERUN_PNG = ATTEMPT2 / "d378_rerun_inspection.png"

EXPECTED_HEAD = "2acb5b99567946d343e95e61087357193da0826c"
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
EXPECTED_AUTHORITY_SHA = (
    "28aadb5ff26270039df58f7cd06080bf7afcdec001402e886a6edf1483fdfe31"
)
EXPECTED_A2_HASHES = {
    A2_EVIDENCE: "e9c3d1cadf9cc9516d0d08792a44b6d824fea7ac8cd0849dffc9a25f3bafda88",
    A2_AUTOMATED: "9a17fdcdf8d74ca3da9288af55163aa2af4ef7e6668fc0edee461e5f41800544",
    A2_MANUAL: "02bd77b0ae90d9dd18dbfe6feaaf7c707a567bb9d12228703347cfb2f6dd7b77",
    A2_COMPLETION: "d44b5581576cfddfba20cf6fa01c5a8ee79cfd5dea3613d5ee338cbee8fb0cd0",
    A2_BOARD: "2e94b4a3a3cd670491ebcd42fd0567667237b5c21ec8806bdb6bb8aed33851e2",
    A2_RRD: "6e605b48e88e6aa0dce4b264f2db193b3dbfd9ff702895ea401a57e6763344e8",
    A2_RBL: "08e72cbccf3eacc304fb941cec98838ab7de5660232f94378593714d26406e49",
    A2_RERUN_VALIDATION: "e583a9121ce9fd734ca0d877d3db0226ff2abbe8029f4abe3886824eb5a4022f",
    A2_RERUN_PNG: "267735d18052a01d9e514bc3510441c390f17711efc05ca2cfefaf16ec38333b",
}
REPAIR_CHANGE = "ascii_board_outcome_box_vertical_layout_only"
VERDICT_PASS = "D378_EPHEMERAL_IDENTIFIER_PROVENANCE_AND_WORKLOAD_AUTHORITY_REPAIR_PASS"
VERDICT_FAIL = "D378_ATTEMPT3_ASCII_BOARD_LAYOUT_REPAIR_FAIL_STOP"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase(name: str, **fields: Any) -> None:
    ordinal = 1
    if PHASE_PATH.is_file():
        ordinal = len(
            [line for line in PHASE_PATH.read_text(encoding="utf-8").splitlines() if line]
        ) + 1
    row = {
        "ordinal": ordinal,
        "phase": name,
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _status_paths() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--short", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(record[3:] for record in output.split("\0") if record)


def _inventory(root: Path) -> dict[str, Any]:
    rows = [
        {
            "path": _rel(path),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {
        "root": _rel(root),
        "file_count": len(rows),
        "files": rows,
        "inventory_sha256": _canonical_sha(rows),
    }


def _source_hashes() -> dict[str, str]:
    return {
        "attempt3_harness": _sha(HARNESS),
        "frozen_attempt2_harness": _sha(ORIGINAL_D378_HARNESS),
    }


def _forbidden_modules_loaded() -> list[str]:
    roots = ("omni", "isaacsim", "isaaclab", "warp", "pxr", "rerun")
    return sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in roots)
    )


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        dimensions = [int(image.width), int(image.height)]
        mode = image.mode
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "dimensions": dimensions,
        "mode": mode,
        "exact_1920x1080": dimensions == [1920, 1080],
    }


def _assert_ascii(value: Any) -> None:
    if isinstance(value, str):
        if any(ord(char) > 127 for char in value):
            raise ValueError(f"non-ASCII display text: {value!r}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_ascii(key)
            _assert_ascii(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_ascii(child)


def _rect_normalized(fig: Any, artist: Any) -> list[float]:
    renderer = fig.canvas.get_renderer()
    bbox = artist.get_window_extent(renderer=renderer)
    width, height = fig.canvas.get_width_height()
    return [
        float(bbox.x0 / width),
        float(bbox.y0 / height),
        float(bbox.x1 / width),
        float(bbox.y1 / height),
    ]


def _inside(inner: list[float], outer: list[float], margin: float = 0.006) -> bool:
    return bool(
        inner[0] >= outer[0] + margin
        and inner[1] >= outer[1] + margin
        and inner[2] <= outer[2] - margin
        and inner[3] <= outer[3] - margin
    )


def _intersects(a: list[float], b: list[float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _render_repaired_board(evidence: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    corrected = evidence["corrected_authority"]
    outcomes = evidence["paired_outcomes"]
    texts = {
        "title": "D378 | Repairing the D377 workload comparator",
        "subtitle": (
            "Offline only | immutable D375 + D377 evidence | "
            "Isaac 0 | PhysX 0 | q5 0 | physics 0 | contact 0"
        ),
        "raw_title": "1. Frozen D377 result",
        "raw_body": (
            "V1 selected digests differed.\n"
            "D375: ec930163...b0b7b\n"
            "D377: 75850473...13c81\n"
            "Formal D377 FAIL_STOP remains frozen."
        ),
        "cause_title": "2. Exact false differences",
        "cause_body": (
            "34 callback witness SHA values:\n"
            "only runtime object addresses differed.\n"
            "34 prototype diagnostics:\n"
            "only generated __Prototype_N differed."
        ),
        "repair_title": "3. Preregistered V2 authority",
        "repair_body": (
            "Keep raw files and provenance.\n"
            "Normalize only exact runtime diagnostics.\n"
            "Keep vertices, indices, polygons, paths,\n"
            "owners, Float32 hashes, mass and volume."
        ),
        "digest_title": "Corrected authoritative equality",
        "digest_body": (
            f"D375 workload: {corrected['D375_corrected_workload_sha256'][:16]}...\n"
            f"D377 workload: {corrected['D377_corrected_workload_sha256'][:16]}...\n"
            f"Normalized witnesses: {corrected['D375_normalized_witness_sha256'][:16]}...\n"
            f"Normalized properties: {corrected['D375_normalized_property_sha256'][:16]}..."
        ),
        "payload_title": "Meaningful payload stayed exact",
        "payload_body": (
            "Callbacks: 34 / 34\n"
            "Vertices: 314 | indices: 1016\n"
            "Original polygons: 262\n"
            "Negative controls: 11/11 PASS"
        ),
        "outcome_title": "Observed terminal outcomes",
        "outcome_body": (
            f"D375: no explicit Erase | timeout | return {outcomes['D375_returncode']}\n"
            f"D377: one Erase | clean exit in {outcomes['D377_elapsed_s']:.3f} s | "
            f"return {outcomes['D377_returncode']}\n"
            "Conditional trigger support for this pair: PASS\n"
            "Universal necessity and exact native root cause: NOT PROVEN"
        ),
        "footer": (
            "D378 verdict: corrected workload authority PASS | "
            "D377 artifact not rewritten | full P34 identity, cylinder physics, "
            "closure and grasp remain NULL | g0a_pass=false"
        ),
    }
    _assert_ascii(texts)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#F7F9FC")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.axis("off")
    artists: dict[str, Any] = {}
    containers: dict[str, list[float]] = {}
    artists["title"] = fig.text(
        0.5, 0.958, texts["title"], ha="center", va="center",
        fontsize=23, fontweight="bold", color="#14213D",
    )
    artists["subtitle"] = fig.text(
        0.5, 0.92, texts["subtitle"], ha="center", va="center",
        fontsize=12, color="#4B5563",
    )

    def box(
        key: str,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        fill: str,
        edge: str,
        *,
        title_size: float = 13.2,
        body_size: float = 11.1,
        title_drop: float = 0.045,
        body_drop: float = 0.09,
    ) -> None:
        canvas.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.010,rounding_size=0.012",
                linewidth=2, edgecolor=edge, facecolor=fill,
            )
        )
        containers[key] = [x, y, x + w, y + h]
        artists[f"{key}_title"] = fig.text(
            x + 0.02, y + h - title_drop, title,
            ha="left", va="center", fontsize=title_size,
            fontweight="bold", color=edge,
        )
        artists[f"{key}_body"] = fig.text(
            x + 0.02, y + h - body_drop, body,
            ha="left", va="top", fontsize=body_size,
            color="#1F2937", linespacing=1.38,
        )

    box("raw", 0.045, 0.64, 0.27, 0.21, texts["raw_title"], texts["raw_body"], "#FDECEC", "#9B1C1C")
    box("cause", 0.365, 0.64, 0.27, 0.21, texts["cause_title"], texts["cause_body"], "#FFF4D8", "#9A6700")
    box("repair", 0.685, 0.64, 0.27, 0.21, texts["repair_title"], texts["repair_body"], "#E8F3FF", "#1D4E89")
    box("digest", 0.045, 0.39, 0.43, 0.17, texts["digest_title"], texts["digest_body"], "#E8F7EE", "#176B3A", body_size=10.8, body_drop=0.078)
    box("payload", 0.525, 0.39, 0.43, 0.17, texts["payload_title"], texts["payload_body"], "#E8F7EE", "#176B3A", body_size=10.8, body_drop=0.078)
    box("outcome", 0.045, 0.16, 0.91, 0.15, texts["outcome_title"], texts["outcome_body"], "#EEF2FF", "#3B4CCA", body_size=10.5, body_drop=0.074)
    artists["footer"] = fig.text(
        0.5, 0.065, texts["footer"], ha="center", va="center",
        fontsize=11.7, fontweight="bold", color="#7F1D1D",
    )

    fig.canvas.draw()
    text_rects = {key: _rect_normalized(fig, artist) for key, artist in artists.items()}
    containment = {}
    for key in containers:
        containment[f"{key}_title_inside"] = _inside(
            text_rects[f"{key}_title"], containers[key]
        )
        containment[f"{key}_body_inside"] = _inside(
            text_rects[f"{key}_body"], containers[key]
        )
    overlap_checks = {
        "footer_clear_of_outcome_box": text_rects["footer"][3]
        < containers["outcome"][1] - 0.02,
        "outcome_body_clear_of_footer": not _intersects(
            text_rects["outcome_body"], text_rects["footer"]
        ),
        "top_row_clear_of_middle_row": all(
            containers[key][1] > containers[mid][3] + 0.03
            for key in ("raw", "cause", "repair")
            for mid in ("digest", "payload")
        ),
        "middle_row_clear_of_outcome": all(
            containers[key][1] > containers["outcome"][3] + 0.03
            for key in ("digest", "payload")
        ),
    }
    layout = {
        "artifact": "D378_ATTEMPT3_ASCII_BOARD_LAYOUT_VALIDATION_V1",
        "registered_change": REPAIR_CHANGE,
        "containers_normalized": containers,
        "text_bboxes_normalized": text_rects,
        "containment_checks": containment,
        "overlap_checks": overlap_checks,
        "pass": all(containment.values()) and all(overlap_checks.values()),
    }
    fig.savefig(BOARD_PATH, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    board = _png_info(BOARD_PATH)
    layout["board"] = board
    layout["pass"] = bool(layout["pass"] and board["exact_1920x1080"])
    return board, layout


def _prepare() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only attempt3 exists: {_rel(OUT_DIR)}")
    OUT_DIR.mkdir(parents=True)
    _phase("prepare_start")
    allowed_dirty = sorted(
        [
            "START_HERE.md",
            _rel(CASE_ROOT) + "/",
            "roarm_rl/viz_debug.py",
            _rel(ORIGINAL_D378_HARNESS),
            _rel(HARNESS),
        ]
    )
    a2_evidence = _read_json(A2_EVIDENCE)
    a2_automated = _read_json(A2_AUTOMATED)
    a2_manual = _read_json(A2_MANUAL)
    a2_completion = _read_json(A2_COMPLETION)
    checks = {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_master_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "dirty_paths_exactly_approved_D378": _status_paths() == allowed_dirty,
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "repo_root_sys_path_zero": sys.path[0] == str(REPO),
        "forbidden_runtime_modules_absent": not _forbidden_modules_loaded(),
        "known_attempt2_hashes_exact": all(
            path.is_file() and _sha(path) == expected
            for path, expected in EXPECTED_A2_HASHES.items()
        ),
        "attempt2_authority_evidence_pass": a2_evidence["pass"] is True,
        "attempt2_corrected_digest_exact": a2_evidence["corrected_authority"][
            "D375_corrected_workload_sha256"
        ]
        == a2_evidence["corrected_authority"]["D377_corrected_workload_sha256"]
        == EXPECTED_AUTHORITY_SHA,
        "attempt2_automated_pass": a2_automated["pass"] is True,
        "attempt2_rerun_strict_pass": a2_automated["rerun"][
            "strict_validation_pass"
        ]
        is True,
        "attempt2_manual_failed_board_only": a2_manual["pass"] is False
        and a2_manual["checks"]["board_text_legible_no_overlap_or_clipping"]
        is False
        and all(
            value is True
            for key, value in a2_manual["checks"].items()
            if key != "board_text_legible_no_overlap_or_clipping"
        ),
        "attempt2_completion_failed_only_manual": a2_completion["pass"] is False
        and sorted(
            key for key, value in a2_completion["checks"].items() if not value
        )
        == [
            "manual_check_keys_exact_and_all_true",
            "manual_original_resolution_inspection_pass",
        ],
        "start_here_attempt3_registered": _rel(OUT_DIR)
        in START_HERE.read_text(encoding="utf-8"),
    }
    prereg = {
        "artifact": "D378_ATTEMPT3_PREREGISTRATION_V1",
        "case": "g0a_d378",
        "attempt": OUT_DIR.name,
        "what_and_why": (
            "Repair only the vertically overflowing outcome text in the "
            "attempt2 ASCII board after manual inspection failed."
        ),
        "new_case_variables": [],
        "reactive_repair_of_registered_variable": REPAIR_CHANGE,
        "authority_recomputation": 0,
        "rerun_recording_or_viewer_invocations": 0,
        "scope_counters": {
            "board_regenerations": 1,
            "offline_authority_audits": 0,
            "automatic_retries": 0,
            "isaac_launches": 0,
            "physx_calls": 0,
            "usd_writes": 0,
            "collider_regenerations": 0,
            "physics_steps": 0,
            "q5_commands": 0,
            "q5_samples": 0,
            "contact_queries": 0,
            "cylinder_writes": 0,
            "target_ik_path_changes": 0,
        },
        "source_hashes": _source_hashes(),
        "attempt1_inventory_before": _inventory(ATTEMPT1),
        "attempt2_inventory_before": _inventory(ATTEMPT2),
        "D375_inventory_before": _inventory(D375_DIR),
        "D377_inventory_before": _inventory(D377_DIR),
        "D334_sidecar_before": _inventory(D334_SIDECAR),
        "allowed_dirty_paths": allowed_dirty,
        "environment": {
            "python": sys.executable,
            "matplotlib": importlib.metadata.version("matplotlib"),
            "forbidden_modules_loaded": _forbidden_modules_loaded(),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase(
        "preregistration_frozen",
        preregistration_sha256=_sha(PREREG_PATH),
        passed=prereg["pass"],
    )
    if not prereg["pass"]:
        raise RuntimeError(f"D378 attempt3 preregistration failed: {checks}")


def _run() -> None:
    if not PREREG_PATH.is_file():
        raise RuntimeError("attempt3 preregistration missing")
    if REPAIR_EVIDENCE_PATH.exists():
        raise FileExistsError("attempt3 repair already ran")
    prereg = _read_json(PREREG_PATH)
    if prereg["pass"] is not True:
        raise RuntimeError("attempt3 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("attempt3 source drift")
    for root, expected in (
        (ATTEMPT1, prereg["attempt1_inventory_before"]),
        (ATTEMPT2, prereg["attempt2_inventory_before"]),
        (D375_DIR, prereg["D375_inventory_before"]),
        (D377_DIR, prereg["D377_inventory_before"]),
        (D334_SIDECAR, prereg["D334_sidecar_before"]),
    ):
        if _inventory(root) != expected:
            raise RuntimeError(f"immutable inventory drift: {_rel(root)}")
    if _forbidden_modules_loaded():
        raise RuntimeError("forbidden runtime module loaded")
    _write_json_x(
        INVOCATION_PATH,
        {
            "artifact": "D378_ATTEMPT3_SINGLE_BOARD_REPAIR_INVOCATION_V1",
            "argv": sys.argv,
            "pid": os.getpid(),
            "board_regenerations": 1,
            "authority_audits": 0,
            "rerun_viewer_invocations": 0,
            "automatic_retries": 0,
            "preregistration_sha256": _sha(PREREG_PATH),
        },
    )
    _phase("single_board_layout_repair_start")
    evidence = _read_json(A2_EVIDENCE)
    board, layout = _render_repaired_board(evidence)
    _write_json_x(LAYOUT_PATH, layout)
    checks = {
        "attempt2_authority_sha_immutable": _sha(A2_EVIDENCE)
        == EXPECTED_A2_HASHES[A2_EVIDENCE],
        "corrected_digest_exact": evidence["corrected_authority"][
            "D375_corrected_workload_sha256"
        ]
        == evidence["corrected_authority"]["D377_corrected_workload_sha256"]
        == EXPECTED_AUTHORITY_SHA,
        "board_exact_1920x1080": board["exact_1920x1080"],
        "programmatic_layout_gate_pass": layout["pass"] is True,
        "all_text_contained": all(layout["containment_checks"].values()),
        "all_registered_overlap_checks_pass": all(
            layout["overlap_checks"].values()
        ),
        "attempt1_immutable": _inventory(ATTEMPT1)
        == prereg["attempt1_inventory_before"],
        "attempt2_immutable": _inventory(ATTEMPT2)
        == prereg["attempt2_inventory_before"],
        "D375_immutable": _inventory(D375_DIR) == prereg["D375_inventory_before"],
        "D377_immutable": _inventory(D377_DIR) == prereg["D377_inventory_before"],
        "D334_sidecar_immutable": _inventory(D334_SIDECAR)
        == prereg["D334_sidecar_before"],
        "forbidden_runtime_modules_absent": not _forbidden_modules_loaded(),
    }
    repair = {
        "artifact": "D378_ATTEMPT3_VISUAL_REPAIR_EVIDENCE_V1",
        "case": "g0a_d378",
        "attempt": OUT_DIR.name,
        "registered_change": REPAIR_CHANGE,
        "new_case_variables": [],
        "authority_evidence": {
            "path": _rel(A2_EVIDENCE),
            "sha256": _sha(A2_EVIDENCE),
            "verdict": evidence["verdict"],
        },
        "attempt2_frozen_completion": {
            "path": _rel(A2_COMPLETION),
            "sha256": _sha(A2_COMPLETION),
            "pass": False,
        },
        "repaired_board": board,
        "layout_validation": {
            "path": _rel(LAYOUT_PATH),
            "sha256": _sha(LAYOUT_PATH),
            "pass": layout["pass"],
        },
        "reused_passed_rerun": {
            "rrd": {
                "path": _rel(A2_RRD),
                "sha256": _sha(A2_RRD),
                "bytes": A2_RRD.stat().st_size,
            },
            "rbl": {
                "path": _rel(A2_RBL),
                "sha256": _sha(A2_RBL),
                "bytes": A2_RBL.stat().st_size,
            },
            "validation": {
                "path": _rel(A2_RERUN_VALIDATION),
                "sha256": _sha(A2_RERUN_VALIDATION),
            },
            "inspection_png": _png_info(A2_RERUN_PNG),
            "new_viewer_invocations": 0,
        },
        "scope_counters": prereg["scope_counters"],
        "remaining_nulls": evidence["remaining_nulls"],
        "g0a_pass": False,
        "checks": checks,
        "pass": all(checks.values()),
        "verdict": VERDICT_PASS if all(checks.values()) else VERDICT_FAIL,
        "next_authorization_boundary": evidence["next_authorization_boundary"],
    }
    _write_json_x(REPAIR_EVIDENCE_PATH, repair)
    _write_json_x(
        MANUAL_TEMPLATE_PATH,
        {
            "artifact": "D378_ATTEMPT3_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
            "expected_sha256": {
                "repaired_board": board["sha256"],
                "reused_rerun_inspection": _sha(A2_RERUN_PNG),
            },
            "required_checks": {
                "repaired_board_text_legible": False,
                "all_text_inside_registered_containers": False,
                "no_text_overlap_or_clipping": False,
                "raw_fail_and_corrected_pass_separated": False,
                "D377_frozen_verdict_and_remaining_nulls_visible": False,
                "reused_rerun_ascii_rows_still_readable": False,
            },
            "pass": False,
        },
    )
    _phase(
        "board_layout_repair_complete_awaiting_manual_inspection",
        repair_evidence_sha256=_sha(REPAIR_EVIDENCE_PATH),
        board_sha256=board["sha256"],
    )
    if not repair["pass"]:
        raise RuntimeError(f"D378 attempt3 repair failed: {checks}")


def _finalize() -> None:
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        REPAIR_EVIDENCE_PATH,
        BOARD_PATH,
        LAYOUT_PATH,
        MANUAL_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"attempt3 finalize prerequisite missing: {_rel(path)}")
    if COMPLETION_PATH.exists():
        raise FileExistsError("D378 final completion already exists")
    _phase("finalize_start")
    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    repair = _read_json(REPAIR_EVIDENCE_PATH)
    layout = _read_json(LAYOUT_PATH)
    manual = _read_json(MANUAL_PATH)
    required_manual_checks = {
        "repaired_board_text_legible",
        "all_text_inside_registered_containers",
        "no_text_overlap_or_clipping",
        "raw_fail_and_corrected_pass_separated",
        "D377_frozen_verdict_and_remaining_nulls_visible",
        "reused_rerun_ascii_rows_still_readable",
    }
    expected_manual_hashes = {
        "repaired_board": repair["repaired_board"]["sha256"],
        "reused_rerun_inspection": repair["reused_passed_rerun"][
            "inspection_png"
        ]["sha256"],
    }
    checks = {
        "preregistration_pass": prereg["pass"] is True,
        "preregistration_sha_bound": _sha(PREREG_PATH)
        == invocation["preregistration_sha256"],
        "repair_evidence_pass": repair["pass"] is True,
        "repair_evidence_current_sha_bound": _sha(REPAIR_EVIDENCE_PATH)
        == manual["repair_evidence_sha256"],
        "layout_validation_pass_and_sha_bound": layout["pass"] is True
        and _sha(LAYOUT_PATH) == repair["layout_validation"]["sha256"],
        "repaired_board_current_exact": _png_info(BOARD_PATH)
        == repair["repaired_board"],
        "manual_artifact_exact": manual["artifact"]
        == "D378_ATTEMPT3_MANUAL_VISUAL_INSPECTION_V1",
        "manual_hashes_exact": manual["inspected_sha256"]
        == expected_manual_hashes,
        "manual_check_keys_exact_and_true": set(manual["checks"])
        == required_manual_checks
        and all(manual["checks"].get(key) is True for key in required_manual_checks),
        "manual_pass": manual["pass"] is True,
        "attempt1_immutable": _inventory(ATTEMPT1)
        == prereg["attempt1_inventory_before"],
        "attempt2_immutable": _inventory(ATTEMPT2)
        == prereg["attempt2_inventory_before"],
        "D375_immutable": _inventory(D375_DIR) == prereg["D375_inventory_before"],
        "D377_immutable": _inventory(D377_DIR) == prereg["D377_inventory_before"],
        "D334_sidecar_immutable": _inventory(D334_SIDECAR)
        == prereg["D334_sidecar_before"],
        "authority_not_recomputed": prereg["authority_recomputation"] == 0,
        "new_rerun_viewer_invocations_zero": prereg[
            "rerun_recording_or_viewer_invocations"
        ]
        == 0,
        "remaining_nulls_preserved": all(
            value is None for value in repair["remaining_nulls"].values()
        ),
        "g0a_false": repair["g0a_pass"] is False,
        "forbidden_runtime_modules_absent": not _forbidden_modules_loaded(),
    }
    completion = {
        "artifact": "D378_FINAL_COMPLETION_SUMMARY_V1",
        "case": "g0a_d378",
        "attempt": OUT_DIR.name,
        "new_case_variables": [],
        "reactive_repair": REPAIR_CHANGE,
        "attempt1_preregistration_fail_stop_preserved": {
            "inventory_sha256": prereg["attempt1_inventory_before"][
                "inventory_sha256"
            ]
        },
        "attempt2_authority_and_visual_failure_preserved": {
            "authority_evidence": {
                "path": _rel(A2_EVIDENCE),
                "sha256": _sha(A2_EVIDENCE),
                "pass": True,
            },
            "completion": {
                "path": _rel(A2_COMPLETION),
                "sha256": _sha(A2_COMPLETION),
                "pass": False,
            },
        },
        "attempt3_repair": {
            "preregistration": {
                "path": _rel(PREREG_PATH),
                "sha256": _sha(PREREG_PATH),
            },
            "invocation": {
                "path": _rel(INVOCATION_PATH),
                "sha256": _sha(INVOCATION_PATH),
            },
            "evidence": {
                "path": _rel(REPAIR_EVIDENCE_PATH),
                "sha256": _sha(REPAIR_EVIDENCE_PATH),
            },
            "layout_validation": {
                "path": _rel(LAYOUT_PATH),
                "sha256": _sha(LAYOUT_PATH),
            },
            "manual_inspection": {
                "path": _rel(MANUAL_PATH),
                "sha256": _sha(MANUAL_PATH),
            },
        },
        "corrected_authority": _read_json(A2_EVIDENCE)["corrected_authority"],
        "board": repair["repaired_board"],
        "rrd": repair["reused_passed_rerun"]["rrd"],
        "rbl": repair["reused_passed_rerun"]["rbl"],
        "rerun_validation": repair["reused_passed_rerun"]["validation"],
        "rerun_inspection": repair["reused_passed_rerun"]["inspection_png"],
        "scope_counters": prereg["scope_counters"],
        "remaining_nulls": repair["remaining_nulls"],
        "g0a_pass": False,
        "checks": checks,
        "pass": all(checks.values()),
        "verdict": VERDICT_PASS if all(checks.values()) else VERDICT_FAIL,
        "next_authorization_boundary": repair["next_authorization_boundary"],
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase(
        "finalize_complete",
        completion_sha256=_sha(COMPLETION_PATH),
        verdict=completion["verdict"],
    )
    if not completion["pass"]:
        raise RuntimeError(f"D378 attempt3 completion failed: {checks}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            _prepare()
        elif args.stage == "run":
            _run()
        else:
            _finalize()
        return 0
    except Exception as exc:
        payload = {
            "artifact": "D378_ATTEMPT3_RUNTIME_EXCEPTION_V1",
            "stage": args.stage,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "verdict": VERDICT_FAIL,
        }
        try:
            if OUT_DIR.exists() and not EXCEPTION_PATH.exists():
                _write_json_x(EXCEPTION_PATH, payload)
            if OUT_DIR.exists():
                _phase(
                    "exception",
                    stage=args.stage,
                    exception_type=type(exc).__name__,
                )
        except Exception:
            pass
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
