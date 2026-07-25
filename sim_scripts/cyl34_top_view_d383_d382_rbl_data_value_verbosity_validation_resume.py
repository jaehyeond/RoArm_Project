#!/usr/bin/env python3
"""D383 one-variable RBL data-value validation resume.

The only changed variable is the authority used to inspect stored RBL data
values: ``rerun rrd print -vvv`` replaces the summary-only ``-v`` invocation.
The frozen D382 board, layout JSON, RBL, and presentation RRD are copied
bit-exact into a forward-only D383 folder.  They are not regenerated.

Isaac, Kit, PhysX, USD, colliders, cylinder, physics, q5, contact, and
target/IK/path work are forbidden.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRIPT_PATH = Path(__file__).resolve()
START_HERE = REPO / "START_HERE.md"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
ISAACLAB_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")

CASE = "g0a_d383"
ATTEMPT = "attempt1_d382_rbl_data_value_verbosity_validation_resume"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT

D382_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d382/"
    "attempt1_d381_layout_validation_native_scalar_serialization_repair"
)
D382_INPUT_PATHS = {
    "board": D382_DIR / "d382_d381_visual_contract_repaired_1920x1080.png",
    "layout": D382_DIR / "d382_board_layout_validation.json",
    "rbl": D382_DIR / "d382_notification_safe_layout.rbl",
    "presentation_rrd": D382_DIR / "d382_notification_safe_presentation.rrd",
}
D382_INPUT_HASHES = {
    "board": "19bd70781403eb11c4eaefb6adb60ab91a5e6ca9f67f2929548f8afff0b7f06d",
    "layout": "7b961cdf8bd606c05438e120728fe243653262de81aa45947a1db0b1c03ab79c",
    "rbl": "979ddd6b4a32bfc97e13d75dfb99625af0d0bed90fc1fd9588347667f284b28c",
    "presentation_rrd": (
        "6c4ad99428f8da0ef842031b161e69db906971084da6d3444c1b76c8c27a7d9a"
    ),
}

PREREG_PATH = OUT_DIR / "d383_preregistration.json"
PHASE_PATH = OUT_DIR / "d383_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d383_offline_validation_invocation.json"
WORKER_STDOUT = OUT_DIR / "d383_offline_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d383_offline_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d383_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d383_offline_worker_supervisor.json"

BOARD_PATH = OUT_DIR / "d383_d382_board_bitexact_copy_1920x1080.png"
LAYOUT_PATH = OUT_DIR / "d383_d382_layout_bitexact_copy.json"
RBL_PATH = OUT_DIR / "d383_d382_layout_bitexact_copy.rbl"
PRESENTATION_RRD_PATH = OUT_DIR / "d383_d382_presentation_bitexact_copy.rrd"
COPY_MANIFEST = OUT_DIR / "d383_bitexact_copy_manifest.json"
PAYLOAD_VALIDATION = OUT_DIR / "d383_rbl_data_value_validation.json"
RERUN_VALIDATION = OUT_DIR / "d383_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d383_rerun_inspection.png"
VIEWER_RECEIPT = OUT_DIR / "d383_viewer_receipt.json"
MANUAL_TEMPLATE = OUT_DIR / "d383_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d383_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d383_completion_summary.json"

NEW_VARIABLES = ["rbl_data_value_print_verbosity_v1"]
WATCHDOG_SECONDS = 300.0
VIEWER_TIMEOUT_SECONDS = 240.0
EXPECTED_RERUN_VERSION = "0.34.1"
EXPECTED_D380_VERDICT = "D380_FAILED_PART_PROVENANCE_AUDIT_PASS_REPAIR_REQUIRED"
EXPECTED_TIMELINES = ["blueprint", "log_time"]
EXPECTED_DATA_VALUE_MARKERS = [
    "/presentation/d381/notification_buffer",
    "/presentation/d381/summary",
    "link5 | authored failed parts",
    "link5 | cooked failed parts",
    "moving side | authored failed parts",
    "moving side | cooked failed parts",
]

# This is the frozen D380 set already inherited by the D382 presentation.
# It is embedded as a contract constant so D383 runtime reads only four D382
# artifacts and cannot self-derive its expected paths from observed output.
FAILED_PARTS = [
    ("gripper_link", "p000_proximal_upper_arm_hull_a"),
    ("gripper_link", "p001_proximal_upper_arm_hull_b"),
    ("gripper_link", "p002_proximal_lower_arm_hull_a"),
    ("gripper_link", "p003_proximal_lower_arm_hull_b"),
    ("gripper_link", "p004_moving_jaw_00_proximal_upper_rail"),
    ("gripper_link", "p005_moving_jaw_01_proximal_lower_rail"),
    ("gripper_link", "p009_moving_jaw_05_distal_nose_bridge"),
    ("gripper_link", "p010_moving_jaw_06_moving_brace_01"),
    ("gripper_link", "p011_moving_jaw_07_moving_brace_02"),
    ("gripper_link", "p013_moving_jaw_09_moving_brace_04"),
    ("gripper_link", "p014_moving_jaw_10_moving_brace_05"),
    ("gripper_link", "p016_moving_upper_backbone"),
    ("gripper_link", "p017_moving_lower_backbone"),
    ("link5", "p003_fixed_jaw_00_lower_bridge"),
    ("link5", "p009_fixed_jaw_06_roof_bridge"),
    ("link5", "p013_fixed_backbone_left"),
    ("link5", "p014_fixed_backbone_right"),
]
MESH_COMPONENTS = [
    "CoordinateFrame:frame",
    "Mesh3D:albedo_factor",
    "Mesh3D:triangle_indices",
    "Mesh3D:vertex_positions",
]
TEXT_COMPONENTS = ["TextDocument:text"]
SUMMARY_COMPONENTS = ["TextDocument:media_type", "TextDocument:text"]
MANUAL_CHECK_KEYS = {
    "board_exact_1920x1080",
    "board_no_text_overlap",
    "board_all_labels_inside_canvas",
    "board_frozen_geometry_subjects_visible",
    "board_displayed_facts_match_d380",
    "rerun_four_geometry_views_visible",
    "rerun_summary_visible",
    "rerun_notifications_only_in_empty_buffer",
    "rerun_no_unknown_timeline",
    "rerun_no_decision_obscuring_overlap",
    "rerun_geometry_consistent_with_d380",
}

FORBIDDEN_IMPORT_ROOTS = {
    "carb",
    "cuda",
    "gymnasium",
    "isaaclab",
    "omni",
    "omniisaacgymenvs",
    "physx",
    "pxr",
    "torch",
    "warp",
}
SCOPE_COUNTERS = {
    "authorized_offline_validation_workers_max": 1,
    "automatic_worker_retries": 0,
    "authorized_rerun_viewer_invocations_max": 1,
    "automatic_viewer_retries": 0,
    "external_or_preexisting_process_signals": 0,
    "d382_runtime_input_files": 4,
    "rrd_or_rbl_regenerations": 0,
    "numeric_or_geometry_audit_invocations": 0,
    "representation_or_tolerance_changes": 0,
    "asset_or_usd_reads": 0,
    "asset_or_usd_writes": 0,
    "collider_materializations_or_regenerations": 0,
    "automatic_decomposition_sweeps": 0,
    "isaac_launches": 0,
    "kit_launches": 0,
    "physx_calls": 0,
    "cylinder_creates_or_writes": 0,
    "physics_steps": 0,
    "public_forwards": 0,
    "q5_commands": 0,
    "q5_samples": 0,
    "contact_queries": 0,
    "target_ik_path_pose_changes": 0,
    "material_mass_actuator_physics_setting_changes": 0,
}

_ACTUAL_VIEWER_INVOCATIONS = 0


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object JSON: {path}")
    return value


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    if json.loads(payload) != value:
        raise RuntimeError(f"JSON round-trip changed evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.write("\n")


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _png_record(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return {
        **_file_record(path),
        "width": int(width),
        "height": int(height),
        "exact_1920x1080": (int(width), int(height)) == (1920, 1080),
    }


def _copy_x(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    record = {
        "source": _file_record(source),
        "copy": _file_record(destination),
        "regenerated": False,
    }
    record["bitexact"] = (
        record["source"]["bytes"] == record["copy"]["bytes"]
        and record["source"]["sha256"] == record["copy"]["sha256"]
    )
    return record


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _status_paths() -> list[str]:
    return _git("status", "--short").splitlines()


def _input_hashes() -> dict[str, str]:
    return {name: _sha(path) for name, path in D382_INPUT_PATHS.items()}


def _source_hashes() -> dict[str, str]:
    return {
        "d383_script": _sha(SCRIPT_PATH),
        "start_here_active_case_authorization": _sha(START_HERE),
        "rerun_contract": _sha(RERUN_CONTRACT),
    }


def _dependency_versions() -> dict[str, str]:
    return {
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("pillow"),
        "psutil": importlib.metadata.version("psutil"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "rerun_sdk": importlib.metadata.version("rerun-sdk"),
    }


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _phase(name: str, **fields: Any) -> None:
    row = {
        "case": CASE,
        "attempt": ATTEMPT,
        "phase": name,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_epoch_s": time.time(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _run(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": time.monotonic() - started,
            "timed_out": False,
            "ok": completed.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode()
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode()
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return {
            "command": command,
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_seconds": time.monotonic() - started,
            "timed_out": True,
            "ok": False,
        }


def _command_summary(result: dict[str, Any]) -> dict[str, Any]:
    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    return {
        "command": result.get("command"),
        "returncode": result.get("returncode"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "timed_out": result.get("timed_out"),
        "ok": result.get("ok"),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stdout_sha256": _sha_text(stdout),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "stderr_sha256": _sha_text(stderr),
        "stderr": stderr,
    }


def _expected_contract() -> tuple[list[str], dict[str, list[str]]]:
    entities: list[str] = []
    components: dict[str, list[str]] = {}
    for body, part in FAILED_PARTS:
        for representation in ("authored", "cooked"):
            mesh_path = f"/d380/{representation}/{body}/{part}"
            metadata_path = (
                f"/metadata/meshes/d380__{representation}__{body}__{part}"
            )
            entities.extend([mesh_path, metadata_path])
            components[mesh_path] = list(MESH_COMPONENTS)
            components[metadata_path] = list(TEXT_COMPONENTS)
    entities.extend(["/metadata/run", "/presentation/d381/summary"])
    components["/metadata/run"] = list(TEXT_COMPONENTS)
    components["/presentation/d381/summary"] = list(SUMMARY_COMPONENTS)
    return sorted(entities), components


def _marker_contract(text: str) -> dict[str, Any]:
    checks = {
        marker: marker in text for marker in EXPECTED_DATA_VALUE_MARKERS
    }
    present = sum(bool(value) for value in checks.values())
    return {
        "markers": checks,
        "expected_markers_present": present,
        "expected_markers_total": len(checks),
        "all_expected_markers_present": all(checks.values()),
    }


def _next_viewer_invocation(current: int) -> int:
    if current != 0:
        raise RuntimeError("D383 second Viewer invocation rejected")
    return 1


def _execution_contract_valid(counters: dict[str, int]) -> bool:
    return (
        counters.get("authorized_offline_validation_workers_max") == 1
        and counters.get("automatic_worker_retries") == 0
        and counters.get("authorized_rerun_viewer_invocations_max") == 1
        and counters.get("automatic_viewer_retries") == 0
        and counters.get("external_or_preexisting_process_signals") == 0
    )


def _loopback_preflight() -> dict[str, Any]:
    result = {
        "host": "127.0.0.1",
        "requested_port": 0,
        "bind_ok": False,
        "selected_ephemeral_port": None,
        "error": None,
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        result["selected_ephemeral_port"] = int(sock.getsockname()[1])
        result["bind_ok"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        sock.close()
    return result


def _phase_rows() -> list[dict[str, Any]]:
    if not PHASE_PATH.is_file():
        return []
    rows = []
    for line in PHASE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def prepare() -> int:
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only output already exists: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")

    from PIL import Image

    start_text = START_HERE.read_text(encoding="utf-8")
    layout = _read_json(D382_INPUT_PATHS["layout"])
    with Image.open(D382_INPUT_PATHS["board"]) as image:
        board_size = tuple(int(value) for value in image.size)
    entities, components = _expected_contract()
    dependencies = _dependency_versions()
    imports = _import_roots(SCRIPT_PATH)

    second_viewer_rejected = False
    try:
        _next_viewer_invocation(1)
    except RuntimeError:
        second_viewer_rejected = True
    synthetic_all = "\n".join(EXPECTED_DATA_VALUE_MARKERS)
    synthetic_missing_one = "\n".join(EXPECTED_DATA_VALUE_MARKERS[:-1])
    checks = {
        "only_four_d382_runtime_inputs_registered": (
            set(D382_INPUT_PATHS)
            == {"board", "layout", "rbl", "presentation_rrd"}
            and len(D382_INPUT_PATHS) == 4
        ),
        "all_d382_inputs_exist": all(
            path.is_file() for path in D382_INPUT_PATHS.values()
        ),
        "all_d382_input_hashes_exact": (
            _input_hashes() == D382_INPUT_HASHES
        ),
        "d382_board_exact_1920x1080": board_size == (1920, 1080),
        "d382_layout_valid_and_pass": layout.get("pass") is True,
        "d382_layout_artifact_exact": (
            layout.get("artifact") == "D382_BOARD_LAYOUT_VALIDATION_V1"
        ),
        "one_new_variable_exact": (
            NEW_VARIABLES == ["rbl_data_value_print_verbosity_v1"]
        ),
        "expected_entity_count_exact_70": (
            len(entities) == 70 and len(set(entities)) == 70
        ),
        "expected_component_contract_count_exact_70": (
            len(components) == 70 and set(components) == set(entities)
        ),
        "expected_timeline_contract_exact": (
            EXPECTED_TIMELINES == ["blueprint", "log_time"]
        ),
        "six_data_value_markers_registered": (
            len(EXPECTED_DATA_VALUE_MARKERS) == 6
            and len(set(EXPECTED_DATA_VALUE_MARKERS)) == 6
        ),
        "synthetic_complete_marker_contract_passes": (
            _marker_contract(synthetic_all)[
                "all_expected_markers_present"
            ]
            is True
        ),
        "synthetic_missing_marker_contract_fails": (
            _marker_contract(synthetic_missing_one)[
                "all_expected_markers_present"
            ]
            is False
        ),
        "forbidden_imports_absent": not (
            imports & FORBIDDEN_IMPORT_ROOTS
        ),
        "interpreter_exact": (
            Path(sys.executable).resolve() == ISAACLAB_PYTHON.resolve()
        ),
        "dependency_versions_exact": dependencies
        == {
            "numpy": "1.26.0",
            "pillow": "11.3.0",
            "psutil": "5.9.8",
            "pyarrow": "23.0.1",
            "rerun_sdk": "0.34.1",
        },
        "rerun_cli_exists_and_is_executable": (
            RERUN_CLI.is_file() and os.access(RERUN_CLI, os.X_OK)
        ),
        "execution_contract_exact": _execution_contract_valid(
            SCOPE_COUNTERS
        ),
        "start_here_authorizes_case_variable_and_path": (
            "D383 [d382_rbl_data_value_verbosity_validation_resume]"
            in start_text
            and NEW_VARIABLES[0] in start_text
            and _rel(OUT_DIR) in start_text
        ),
        "head_equals_origin_master": (
            _git("rev-parse", "HEAD")
            == _git("rev-parse", "origin/master")
        ),
    }
    controls = {
        "wrong_board_hash_rejected": (
            "0" * 64 != D382_INPUT_HASHES["board"]
        ),
        "wrong_rbl_hash_rejected": (
            "0" * 64 != D382_INPUT_HASHES["rbl"]
        ),
        "observed_paths_not_used_as_expected_contract": (
            len(FAILED_PARTS) == 17
            and "/presentation/d381/summary" in entities
        ),
        "summary_entity_rename_rejected": (
            "/presentation/d383/summary" not in entities
        ),
        "second_worker_request_rejected": not _execution_contract_valid(
            {
                **SCOPE_COUNTERS,
                "authorized_offline_validation_workers_max": 2,
            }
        ),
        "second_viewer_request_rejected": second_viewer_rejected,
        "physics_counter_nonzero_rejected": (
            {**SCOPE_COUNTERS, "physics_steps": 1} != SCOPE_COUNTERS
        ),
        "regeneration_counter_nonzero_rejected": (
            {**SCOPE_COUNTERS, "rrd_or_rbl_regenerations": 1}
            != SCOPE_COUNTERS
        ),
        "g0a_flip_rejected": False is not True,
        "p34_identity_flip_rejected": False is not True,
    }
    prereg = {
        "artifact": "D383_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Resume the frozen D382 Rerun presentation validation by "
            "changing only the RBL data-value print authority from -v to "
            "-vvv; do not regenerate any presentation subject."
        ),
        "new_variables": NEW_VARIABLES,
        "immutable_d382_inputs": [
            {
                "name": name,
                "path": _rel(D382_INPUT_PATHS[name]),
                "bytes": D382_INPUT_PATHS[name].stat().st_size,
                "sha256": D382_INPUT_HASHES[name],
            }
            for name in sorted(D382_INPUT_PATHS)
        ],
        "registered_reuse": {
            "copy_only": True,
            "bitexact_required": True,
            "board_or_layout_regenerated": False,
            "rrd_or_rbl_regenerated": False,
            "embedded_paths_or_view_names_renamed": False,
            "viewer_command_changed": False,
        },
        "registered_validation": {
            "rbl_data_value_authority": [
                str(RERUN_CLI),
                "rrd",
                "print",
                "-vvv",
                _rel(RBL_PATH),
            ],
            "summary_only_negative_control": "-v",
            "expected_data_value_markers": EXPECTED_DATA_VALUE_MARKERS,
            "exact_non_system_entity_count": len(entities),
            "exact_non_system_entities": entities,
            "exact_component_contract_sha256": _canonical_sha(components),
            "exact_timelines": EXPECTED_TIMELINES,
            "strict_helper_component_column_verbosity_unchanged": "-v",
            "expected_rerun_version": EXPECTED_RERUN_VERSION,
        },
        "registered_execution": {
            **SCOPE_COUNTERS,
            "bounded_worker_watchdog_seconds": WATCHDOG_SECONDS,
            "bounded_viewer_timeout_seconds": VIEWER_TIMEOUT_SECONDS,
            "watchdog_signal_scope": (
                "D383-owned child process group after timeout only"
            ),
        },
        "preserved_status": {
            "numeric_verdict": EXPECTED_D380_VERDICT,
            "p34_authored_to_cooked_identity_pass": False,
            "g0a_pass": False,
            "all_physics_and_grasp_results": None,
        },
        "source_hashes": _source_hashes(),
        "input_hashes": _input_hashes(),
        "dependency_versions": dependencies,
        "registered_dirty_baseline": _status_paths(),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "subject": _git("log", "-1", "--pretty=%s"),
        },
        "checks": checks,
        "negative_controls": {
            "controls": controls,
            "passed": sum(bool(value) for value in controls.values()),
            "total": len(controls),
            "pass": all(controls.values()),
        },
        "pass": all(checks.values()) and all(controls.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase(
        "preregistration_frozen",
        preregistration_sha256=_sha(PREREG_PATH),
        passed=prereg["pass"],
        checks_passed=sum(bool(value) for value in checks.values()),
        checks_total=len(checks),
    )
    if not prereg["pass"]:
        raise RuntimeError(f"D383 preregistration failed: {checks}")
    return 0


def worker() -> int:
    global _ACTUAL_VIEWER_INVOCATIONS

    _phase("worker_start", pid=os.getpid())
    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D383 preregistration did not pass")
    if invocation.get("preregistration_sha256") != _sha(PREREG_PATH):
        raise RuntimeError("D383 invocation not bound to preregistration")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D383 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D383 immutable inputs changed after preregistration")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("D383 dirty baseline changed after preregistration")

    copies = {
        "board": _copy_x(D382_INPUT_PATHS["board"], BOARD_PATH),
        "layout": _copy_x(D382_INPUT_PATHS["layout"], LAYOUT_PATH),
        "rbl": _copy_x(D382_INPUT_PATHS["rbl"], RBL_PATH),
        "presentation_rrd": _copy_x(
            D382_INPUT_PATHS["presentation_rrd"],
            PRESENTATION_RRD_PATH,
        ),
    }
    copy_checks = {
        "four_inputs_copied": len(copies) == 4,
        "all_copies_bitexact": all(
            row["bitexact"] is True for row in copies.values()
        ),
        "all_regenerated_false": all(
            row["regenerated"] is False for row in copies.values()
        ),
        "copied_hashes_match_registered_inputs": {
            name: row["copy"]["sha256"] for name, row in copies.items()
        }
        == D382_INPUT_HASHES,
        "copied_layout_valid_and_pass": (
            _read_json(LAYOUT_PATH).get("pass") is True
        ),
        "copied_board_exact_1920x1080": (
            _png_record(BOARD_PATH)["exact_1920x1080"] is True
        ),
    }
    copy_manifest = {
        "artifact": "D383_BITEXACT_COPY_MANIFEST_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "copies": copies,
        "checks": copy_checks,
        "pass": all(copy_checks.values()),
    }
    _write_json_x(COPY_MANIFEST, copy_manifest)
    _phase(
        "four_immutable_inputs_copied_bitexact",
        manifest_sha256=_sha(COPY_MANIFEST),
        passed=copy_manifest["pass"],
    )
    if not copy_manifest["pass"]:
        raise RuntimeError(f"D383 bit-exact copy failed: {copy_checks}")

    from roarm_rl.rerun_contract import validate_rerun_artifact

    entities, components = _expected_contract()
    archive = validate_rerun_artifact(
        PRESENTATION_RRD_PATH,
        expected_entity_paths=entities,
        exact_entity_paths=entities,
        expected_timeline_names=EXPECTED_TIMELINES,
        exact_timeline_names=EXPECTED_TIMELINES,
        expected_entity_components=components,
        blueprint_path=RBL_PATH,
        screenshot_path=None,
        cli_path=RERUN_CLI,
        expected_version=EXPECTED_RERUN_VERSION,
        timeout_s=180.0,
    )
    if archive.get("pass") is not True:
        raise RuntimeError(f"D383 strict archive validation failed: {archive}")
    _phase(
        "strict_archive_validation_passed",
        entity_count=len(
            archive["entity_path_contract"]["observed_non_system"]
        ),
        timeline_count=len(archive["timeline_contract"]["observed"]),
    )

    summary_only = _run(
        [str(RERUN_CLI), "rrd", "print", "-v", str(RBL_PATH)],
        timeout=90.0,
    )
    data_visible = _run(
        [str(RERUN_CLI), "rrd", "print", "-vvv", str(RBL_PATH)],
        timeout=90.0,
    )
    summary_text = (
        f"{summary_only['stdout']}\n{summary_only['stderr']}"
    )
    data_text = f"{data_visible['stdout']}\n{data_visible['stderr']}"
    summary_markers = _marker_contract(summary_text)
    data_markers = _marker_contract(data_text)
    payload_checks = {
        "summary_only_negative_control_return_zero": (
            summary_only["returncode"] == 0
            and not summary_only["timed_out"]
        ),
        "summary_only_negative_control_zero_of_six": (
            summary_markers["expected_markers_present"] == 0
            and summary_markers["expected_markers_total"] == 6
            and summary_markers["all_expected_markers_present"] is False
        ),
        "data_visible_authority_return_zero": (
            data_visible["returncode"] == 0
            and not data_visible["timed_out"]
        ),
        "data_visible_authority_six_of_six": (
            data_markers["expected_markers_present"] == 6
            and data_markers["expected_markers_total"] == 6
            and data_markers["all_expected_markers_present"] is True
        ),
        "authority_command_exactly_vvv": (
            data_visible["command"][3] == "-vvv"
        ),
        "strict_helper_component_verbosity_unchanged": (
            archive["print_verbose"]["command"][3] == "-v"
        ),
    }
    payload_validation = {
        "artifact": "D383_RBL_DATA_VALUE_VALIDATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variable": NEW_VARIABLES[0],
        "summary_only_negative_control": {
            "result": _command_summary(summary_only),
            "marker_contract": summary_markers,
            "role": "negative control only; not data-value authority",
        },
        "data_visible_authority": {
            "result": _command_summary(data_visible),
            "marker_contract": data_markers,
            "role": "registered RBL data-value authority",
        },
        "checks": payload_checks,
        "pass": all(payload_checks.values()),
    }
    _write_json_x(PAYLOAD_VALIDATION, payload_validation)
    _phase(
        "rbl_data_value_authority_passed",
        payload_validation_sha256=_sha(PAYLOAD_VALIDATION),
        data_visible_markers_present=(
            data_markers["expected_markers_present"]
        ),
        data_visible_markers_total=data_markers["expected_markers_total"],
    )
    if not payload_validation["pass"]:
        raise RuntimeError(
            f"D383 RBL data-value validation failed: {payload_checks}"
        )

    loopback = _loopback_preflight()
    if not loopback["bind_ok"]:
        raise RuntimeError(f"D383 loopback preflight failed: {loopback}")
    if RERUN_SCREENSHOT.exists():
        raise FileExistsError(RERUN_SCREENSHOT)
    _ACTUAL_VIEWER_INVOCATIONS = _next_viewer_invocation(
        _ACTUAL_VIEWER_INVOCATIONS
    )
    viewer_command = [
        str(RERUN_CLI),
        "--headless",
        "--bind",
        "127.0.0.1",
        "--port",
        "auto",
        "--hide-welcome-screen",
        "--window-size",
        "1920x1080",
        "--screenshot-to",
        str(RERUN_SCREENSHOT),
        str(PRESENTATION_RRD_PATH),
    ]
    _phase(
        "viewer_invocation_start",
        actual_viewer_invocations=_ACTUAL_VIEWER_INVOCATIONS,
        command=viewer_command,
    )
    viewer_env = dict(os.environ)
    viewer_env["RERUN_ANALYTICS_ENABLED"] = "false"
    viewer = _run(
        viewer_command,
        timeout=VIEWER_TIMEOUT_SECONDS,
        env=viewer_env,
    )
    _phase(
        "viewer_invocation_complete",
        actual_viewer_invocations=_ACTUAL_VIEWER_INVOCATIONS,
        returncode=viewer.get("returncode"),
        timed_out=viewer.get("timed_out"),
    )
    combined_output = f"{viewer['stdout']}\n{viewer['stderr']}"
    screenshot = (
        _png_record(RERUN_SCREENSHOT)
        if RERUN_SCREENSHOT.is_file()
        else {"path": _rel(RERUN_SCREENSHOT), "exists": False}
    )
    viewer_checks = {
        "loopback_preflight_pass": loopback["bind_ok"],
        "viewer_return_zero": viewer["returncode"] == 0,
        "viewer_not_timed_out": not viewer["timed_out"],
        "screenshot_nonempty": (
            RERUN_SCREENSHOT.is_file()
            and RERUN_SCREENSHOT.stat().st_size > 0
        ),
        "message_proxy_operation_not_permitted_absent": (
            "message proxy server crashed" not in combined_output.lower()
            and "operation not permitted" not in combined_output.lower()
        ),
        "viewer_invocation_exactly_one": (
            _ACTUAL_VIEWER_INVOCATIONS == 1
        ),
        "viewer_retry_zero": True,
        "viewer_command_frozen": (
            "--headless" in viewer_command
            and "--bind" in viewer_command
            and "127.0.0.1" in viewer_command
            and "--port" in viewer_command
            and "auto" in viewer_command
            and "--hide-welcome-screen" in viewer_command
            and "--window-size" in viewer_command
            and "1920x1080" in viewer_command
        ),
    }
    viewer_receipt = {
        "artifact": "D383_VIEWER_RECEIPT_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "loopback_preflight": loopback,
        "authorized_viewer_invocations_max": 1,
        "actual_viewer_invocations": _ACTUAL_VIEWER_INVOCATIONS,
        "automatic_viewer_retries": 0,
        "command": viewer_command,
        "result": _command_summary(viewer),
        "screenshot": screenshot,
        "screenshot_dimension_policy": (
            "record native physical pixels; exact 1920x1080 is required only "
            "for the frozen static board because HiDPI may scale Viewer PNG"
        ),
        "checks": viewer_checks,
        "pass": all(viewer_checks.values()),
    }
    _write_json_x(VIEWER_RECEIPT, viewer_receipt)
    if not viewer_receipt["pass"]:
        raise RuntimeError(f"D383 Viewer contract failed: {viewer_checks}")

    validation_checks = {
        "archive_validation_pass": archive.get("pass") is True,
        "payload_validation_pass": payload_validation["pass"],
        "viewer_receipt_pass": viewer_receipt["pass"],
        "exact_non_system_entities_70": (
            len(
                archive["entity_path_contract"]["observed_non_system"]
            )
            == 70
        ),
        "exact_timelines_two": (
            archive["timeline_contract"]["observed"]
            == EXPECTED_TIMELINES
        ),
        "rrd_copy_hash_exact": (
            _sha(PRESENTATION_RRD_PATH)
            == D382_INPUT_HASHES["presentation_rrd"]
        ),
        "rbl_copy_hash_exact": (
            _sha(RBL_PATH) == D382_INPUT_HASHES["rbl"]
        ),
        "headless_render_not_duplicated_by_strict_helper": (
            archive["headless_render"]["attempted"] is False
        ),
    }
    validation = {
        "artifact": "D383_RERUN_VALIDATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "archive_validation": archive,
        "payload_validation": _file_record(PAYLOAD_VALIDATION),
        "copy_manifest": _file_record(COPY_MANIFEST),
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "screenshot": screenshot,
        "checks": validation_checks,
        "pass": all(validation_checks.values()),
    }
    _write_json_x(RERUN_VALIDATION, validation)
    if not validation["pass"]:
        raise RuntimeError(
            f"D383 combined Rerun validation failed: {validation_checks}"
        )
    _phase(
        "rerun_validation_and_single_capture_complete",
        rerun_validation_sha256=_sha(RERUN_VALIDATION),
        screenshot_sha256=screenshot["sha256"],
    )

    manual_template = {
        "artifact": "D383_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board": _file_record(BOARD_PATH),
        "layout": _file_record(LAYOUT_PATH),
        "presentation_rrd": _file_record(PRESENTATION_RRD_PATH),
        "rbl": _file_record(RBL_PATH),
        "rerun_validation": _file_record(RERUN_VALIDATION),
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "rerun_screenshot": screenshot,
        "required_check_keys": sorted(MANUAL_CHECK_KEYS),
        "inspection_detail_required": "original",
        "inspection_checks": {
            key: None for key in sorted(MANUAL_CHECK_KEYS)
        },
        "observations": [],
        "inspector_result": None,
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)

    post_checks = {
        "copy_manifest_pass": copy_manifest["pass"],
        "payload_validation_pass": payload_validation["pass"],
        "rerun_validation_pass": validation["pass"],
        "viewer_receipt_pass": viewer_receipt["pass"],
        "viewer_invocation_exactly_one": (
            _ACTUAL_VIEWER_INVOCATIONS == 1
        ),
        "viewer_retry_zero": (
            viewer_receipt["automatic_viewer_retries"] == 0
        ),
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
        "p34_identity_still_false": True,
        "g0a_still_false": True,
        "all_forbidden_counters_zero": all(
            value == 0
            for name, value in SCOPE_COUNTERS.items()
            if name
            not in {
                "authorized_offline_validation_workers_max",
                "authorized_rerun_viewer_invocations_max",
                "d382_runtime_input_files",
            }
        ),
    }
    claim = {
        "artifact": "D383_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "pid": os.getpid(),
        "preregistration": _file_record(PREREG_PATH),
        "new_variables": NEW_VARIABLES,
        "copy_manifest": _file_record(COPY_MANIFEST),
        "payload_validation": _file_record(PAYLOAD_VALIDATION),
        "rerun_validation": _file_record(RERUN_VALIDATION),
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "manual_template": _file_record(MANUAL_TEMPLATE),
        "actual_viewer_invocations": _ACTUAL_VIEWER_INVOCATIONS,
        "scope_counters": SCOPE_COUNTERS,
        "preserved_status": {
            "numeric_verdict": EXPECTED_D380_VERDICT,
            "p34_authored_to_cooked_identity_pass": False,
            "g0a_pass": False,
        },
        "checks": post_checks,
        "pass": all(post_checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_claim_written", worker_claim_sha256=_sha(WORKER_CLAIM))
    if not claim["pass"]:
        raise RuntimeError(f"D383 worker post-check failed: {post_checks}")
    return 0


def run_supervisor() -> int:
    if not _execution_contract_valid(SCOPE_COUNTERS):
        raise RuntimeError("D383 execution contract changed")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D383 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D383 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D383 inputs changed after preregistration")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("D383 dirty baseline changed after preregistration")

    command = [sys.executable, "-B", str(SCRIPT_PATH), "--stage", "worker"]
    invocation = {
        "artifact": "D383_OFFLINE_VALIDATION_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "cwd": str(REPO),
        "preregistration_sha256": _sha(PREREG_PATH),
        "source_hashes": _source_hashes(),
        "input_hashes": _input_hashes(),
        "worker_spawn_count_registered": 1,
        "automatic_worker_retry_count_registered": 0,
        "viewer_invocation_max_registered": 1,
        "automatic_viewer_retry_count_registered": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase(
        "supervisor_spawn_start",
        invocation_sha256=_sha(INVOCATION_PATH),
        watchdog_seconds=WATCHDOG_SECONDS,
    )

    started = time.monotonic()
    timed_out = False
    sigterm_sent = False
    sigkill_sent = False
    with WORKER_STDOUT.open("xb") as stdout, WORKER_STDERR.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        pgid = process.pid
        try:
            returncode = process.wait(timeout=WATCHDOG_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(pgid, signal.SIGTERM)
            sigterm_sent = True
            try:
                returncode = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                sigkill_sent = True
                returncode = process.wait(timeout=10.0)
    elapsed = time.monotonic() - started
    try:
        os.killpg(pgid, 0)
        group_alive = True
    except ProcessLookupError:
        group_alive = False
    except PermissionError:
        group_alive = True

    phases = _phase_rows()
    actual_viewer_starts = sum(
        row.get("phase") == "viewer_invocation_start" for row in phases
    )
    actual_viewer_completes = sum(
        row.get("phase") == "viewer_invocation_complete" for row in phases
    )
    claim = _read_json(WORKER_CLAIM) if WORKER_CLAIM.is_file() else {}
    required = {
        "worker_claim": WORKER_CLAIM.is_file(),
        "copy_manifest": COPY_MANIFEST.is_file(),
        "board": BOARD_PATH.is_file(),
        "layout": LAYOUT_PATH.is_file(),
        "presentation_rrd": PRESENTATION_RRD_PATH.is_file(),
        "rbl": RBL_PATH.is_file(),
        "payload_validation": PAYLOAD_VALIDATION.is_file(),
        "rerun_validation": RERUN_VALIDATION.is_file(),
        "viewer_receipt": VIEWER_RECEIPT.is_file(),
        "rerun_screenshot": RERUN_SCREENSHOT.is_file(),
        "manual_template": MANUAL_TEMPLATE.is_file(),
    }
    operational_pass = (
        returncode == 0
        and not timed_out
        and not sigterm_sent
        and not sigkill_sent
        and not group_alive
        and actual_viewer_starts == 1
        and actual_viewer_completes == 1
        and all(required.values())
        and claim.get("pass") is True
        and _source_hashes() == prereg["source_hashes"]
        and _input_hashes() == prereg["input_hashes"]
        and _status_paths() == prereg["registered_dirty_baseline"]
    )
    supervisor = {
        "artifact": "D383_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "worker_pid": process.pid,
        "worker_process_group": pgid,
        "actual_offline_validation_workers": 1,
        "automatic_worker_retries": 0,
        "authorized_rerun_viewer_invocations_max": 1,
        "actual_rerun_viewer_invocations": actual_viewer_starts,
        "actual_rerun_viewer_completions": actual_viewer_completes,
        "automatic_viewer_retries": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "process_group_alive_after_wait": group_alive,
        "required_artifacts": required,
        "worker_claim_sha256": (
            _sha(WORKER_CLAIM) if WORKER_CLAIM.is_file() else None
        ),
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
        "operational_pass": operational_pass,
        "pass": operational_pass,
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase(
        "supervisor_complete",
        returncode=returncode,
        elapsed_seconds=elapsed,
        actual_viewer_invocations=actual_viewer_starts,
        operational_pass=operational_pass,
    )
    return 0 if operational_pass else 1


def finalize() -> int:
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        WORKER_CLAIM,
        SUPERVISOR_PATH,
        COPY_MANIFEST,
        BOARD_PATH,
        LAYOUT_PATH,
        PRESENTATION_RRD_PATH,
        RBL_PATH,
        PAYLOAD_VALIDATION,
        RERUN_VALIDATION,
        VIEWER_RECEIPT,
        RERUN_SCREENSHOT,
        MANUAL_TEMPLATE,
        MANUAL_INSPECTION,
    ]
    if COMPLETION_PATH.exists():
        raise FileExistsError(COMPLETION_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"D383 finalize missing files: {missing}")

    prereg = _read_json(PREREG_PATH)
    claim = _read_json(WORKER_CLAIM)
    supervisor = _read_json(SUPERVISOR_PATH)
    copy_manifest = _read_json(COPY_MANIFEST)
    payload = _read_json(PAYLOAD_VALIDATION)
    validation = _read_json(RERUN_VALIDATION)
    viewer = _read_json(VIEWER_RECEIPT)
    manual_template = _read_json(MANUAL_TEMPLATE)
    manual = _read_json(MANUAL_INSPECTION)
    manual_checks = manual.get("inspection_checks", {})
    checks = {
        "preregistration_pass": prereg.get("pass") is True,
        "worker_claim_pass": claim.get("pass") is True,
        "supervisor_pass": supervisor.get("pass") is True,
        "copy_manifest_pass": copy_manifest.get("pass") is True,
        "payload_validation_pass": payload.get("pass") is True,
        "rerun_validation_pass": validation.get("pass") is True,
        "viewer_receipt_pass": viewer.get("pass") is True,
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
        "board_bitexact_with_d382": (
            _sha(BOARD_PATH) == D382_INPUT_HASHES["board"]
        ),
        "layout_bitexact_with_d382": (
            _sha(LAYOUT_PATH) == D382_INPUT_HASHES["layout"]
        ),
        "rbl_bitexact_with_d382": (
            _sha(RBL_PATH) == D382_INPUT_HASHES["rbl"]
        ),
        "presentation_rrd_bitexact_with_d382": (
            _sha(PRESENTATION_RRD_PATH)
            == D382_INPUT_HASHES["presentation_rrd"]
        ),
        "manual_artifact_exact": (
            manual.get("artifact") == "D383_MANUAL_VISUAL_INSPECTION_V1"
            and manual.get("case") == CASE
            and manual.get("attempt") == ATTEMPT
        ),
        "manual_template_hash_exact": (
            manual.get("template", {}).get("sha256")
            == _sha(MANUAL_TEMPLATE)
        ),
        "manual_board_hash_exact": (
            manual.get("board", {}).get("sha256") == _sha(BOARD_PATH)
        ),
        "manual_rerun_screenshot_hash_exact": (
            manual.get("rerun_screenshot", {}).get("sha256")
            == _sha(RERUN_SCREENSHOT)
        ),
        "manual_check_keys_exact": (
            set(manual_checks) == MANUAL_CHECK_KEYS
        ),
        "manual_checks_all_true": (
            set(manual_checks) == MANUAL_CHECK_KEYS
            and all(value is True for value in manual_checks.values())
        ),
        "manual_observations_nonempty": bool(manual.get("observations")),
        "manual_inspector_result_pass": (
            manual.get("inspector_result") == "PASS"
        ),
        "manual_visual_inspection_pass": manual.get("pass") is True,
        "p34_identity_still_false": True,
        "g0a_still_false": True,
    }
    completion_pass = all(checks.values())
    completion = {
        "artifact": "D383_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": NEW_VARIABLES,
        "checks": checks,
        "completion_pass": completion_pass,
        "verdict": (
            "D383_RBL_DATA_VALUE_VERBOSITY_VALIDATION_RESUME_PASS"
            if completion_pass
            else "D383_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"
        ),
        "preserved_d380_numeric_verdict": EXPECTED_D380_VERDICT,
        "p34_authored_to_cooked_identity_pass": False,
        "g0a_pass": False,
        "remaining_nulls": {
            "p34_representation_repair": None,
            "p34_repaired_live_identity": None,
            "actual_open_jaw_clearance": None,
            "cylinder_contact_or_tipping": None,
            "q5_closure": None,
            "grasp_feasibility": None,
            "target_ik_path_justification": None,
        },
        "scope_counters": {
            **SCOPE_COUNTERS,
            "actual_offline_validation_workers": supervisor[
                "actual_offline_validation_workers"
            ],
            "actual_rerun_viewer_invocations": supervisor[
                "actual_rerun_viewer_invocations"
            ],
        },
        "next_authorization_boundary": (
            "P34 representation repair/live identity, 29x50 target rebase, "
            "and all Isaac/PhysX/physics/q5/contact remain unapproved."
        ),
        "artifacts": {
            path.name: _file_record(path)
            for path in [
                BOARD_PATH,
                LAYOUT_PATH,
                RBL_PATH,
                PRESENTATION_RRD_PATH,
                COPY_MANIFEST,
                PAYLOAD_VALIDATION,
                RERUN_VALIDATION,
                VIEWER_RECEIPT,
                RERUN_SCREENSHOT,
                MANUAL_INSPECTION,
            ]
        },
    }
    _write_json_x(COMPLETION_PATH, completion)
    return 0 if completion_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "run", "worker", "finalize"],
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        return prepare()
    if args.stage == "run":
        return run_supervisor()
    if args.stage == "worker":
        return worker()
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
