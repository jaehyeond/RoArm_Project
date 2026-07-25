#!/usr/bin/env python3
"""D382 forward-only serialization repair for the frozen D381 presentation.

Only two variables are introduced:
1) recursive conversion of NumPy/Matplotlib scalars to JSON-native values;
2) complete JSON serialization before exclusive file creation.

The frozen D381 board/presentation implementation is imported read-only and
executed into a new D382 output path.  Isaac, Kit, PhysX, USD, colliders,
cylinder, physics, q5, contact, and target/IK/path work are forbidden.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRIPT_PATH = Path(__file__).resolve()
BASE_SCRIPT = (
    REPO
    / "sim_scripts/cyl34_top_view_d381_d380_visual_contract_repair.py"
)
START_HERE = REPO / "START_HERE.md"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
ISAACLAB_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")

_BASE_SPEC = importlib.util.spec_from_file_location(
    "frozen_d381_visual_contract",
    BASE_SCRIPT,
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError("cannot load frozen D381 implementation")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)
_ORIGINAL_BASE_RUN = base._run
_ACTUAL_VIEWER_INVOCATIONS = 0

CASE = "g0a_d382"
ATTEMPT = "attempt1_d381_layout_validation_native_scalar_serialization_repair"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT

PREREG_PATH = OUT_DIR / "d382_preregistration.json"
PHASE_PATH = OUT_DIR / "d382_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d382_offline_presentation_invocation.json"
WORKER_STDOUT = OUT_DIR / "d382_offline_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d382_offline_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d382_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d382_offline_worker_supervisor.json"

BOARD_PATH = OUT_DIR / "d382_d381_visual_contract_repaired_1920x1080.png"
LAYOUT_VALIDATION = OUT_DIR / "d382_board_layout_validation.json"
BASE_COPY_PATH = OUT_DIR / "d382_d380_source_bitexact_copy.rrd"
RECORDING_ONLY_PATH = OUT_DIR / "d382_d380_recording_only.rrd"
OVERLAY_RRD_PATH = OUT_DIR / "d382_presentation_overlay.rrd"
RBL_PATH = OUT_DIR / "d382_notification_safe_layout.rbl"
PRESENTATION_RRD_PATH = OUT_DIR / "d382_notification_safe_presentation.rrd"
RECORDING_EQUIVALENCE = OUT_DIR / "d382_recording_equivalence.json"
RERUN_VALIDATION = OUT_DIR / "d382_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d382_rerun_inspection.png"
VIEWER_RECEIPT = OUT_DIR / "d382_viewer_receipt.json"
MANUAL_TEMPLATE = OUT_DIR / "d382_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d382_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d382_completion_summary.json"
NEGATIVE_PARTIAL_PATH = (
    OUT_DIR / "d382_negative_unserializable_must_not_exist.json"
)
NEGATIVE_NAN_PATH = OUT_DIR / "d382_negative_nan_must_not_exist.json"

D381_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d381/"
    "attempt1_d380_visual_contract_repair"
)
D381_INPUT_PATHS = {
    "d381_script": BASE_SCRIPT,
    "d381_preregistration": D381_DIR / "d381_preregistration.json",
    "d381_invocation": D381_DIR / "d381_offline_presentation_invocation.json",
    "d381_stderr": D381_DIR / "d381_offline_worker_stderr.log",
    "d381_supervisor": D381_DIR / "d381_offline_worker_supervisor.json",
    "d381_board": (
        D381_DIR / "d381_d380_visual_contract_repaired_1920x1080.png"
    ),
    "d381_truncated_layout": D381_DIR / "d381_board_layout_validation.json",
    "d381_partial_inspection": (
        D381_DIR / "d381_partial_board_visual_inspection.json"
    ),
    "d381_phase_markers": D381_DIR / "d381_phase_markers.jsonl",
    "d381_fail_attestation": D381_DIR / "d381_fail_stop_attestation.json",
}
D381_INPUT_HASHES = {
    "d381_script": (
        "b58a60ad8f0d8873973c9171f03cb7bd75401c205db074a49a9a498d98adbd2d"
    ),
    "d381_preregistration": (
        "fdffbe48ebd6af275ca534acf197952d3f8430287a20bbdf06af7d596512cc69"
    ),
    "d381_invocation": (
        "a53254cd290b8d74d34d8cbfff88954abbf975e192d5b1edbde459d9913473ae"
    ),
    "d381_stderr": (
        "2218c2837dee983d451f0b38e6e0c1b398bf3679daf66051918bd5cb3dcbfeec"
    ),
    "d381_supervisor": (
        "e2930719620aba6d67fd929c74f150d48ab4c7562ad0e27c65964721c399298a"
    ),
    "d381_board": (
        "19bd70781403eb11c4eaefb6adb60ab91a5e6ca9f67f2929548f8afff0b7f06d"
    ),
    "d381_truncated_layout": (
        "bd8140b23bb4794eef67f98eb84aac670f7669e7d5d0d51be1f48a0f61942dba"
    ),
    "d381_partial_inspection": (
        "58639f2cf3a76bc40ad6921479b32d1645abcf8498fdaba636bc1b13995ba0e1"
    ),
    "d381_phase_markers": (
        "9f86e314d4b9fe352c72b68614d84f56f837b59bcc85eea8d85e5fa22c3ce361"
    ),
    "d381_fail_attestation": (
        "e62f6aba1340bfe54d87638564d103e18248f4e61d033cc5711026ba66939b0b"
    ),
}

NEW_VARIABLES = [
    "json_native_recursive_scalar_normalization_v1",
    "serialize_before_exclusive_create_v1",
]
WATCHDOG_SECONDS = 300.0
VIEWER_TIMEOUT_SECONDS = 240.0
EXPECTED_D381_BOARD_SHA256 = D381_INPUT_HASHES["d381_board"]

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
    "actual_offline_presentation_workers": 1,
    "automatic_worker_retries": 0,
    "rerun_viewer_invocations": 1,
    "automatic_viewer_retries": 0,
    "external_or_preexisting_process_signals": 0,
    "d379_reads": 0,
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


def _execution_contract_valid(counters: dict[str, int]) -> bool:
    """Enforce the user-authorized worker/Viewer limits for this case."""
    return (
        counters.get("actual_offline_presentation_workers") == 1
        and counters.get("automatic_worker_retries") == 0
        and counters.get("rerun_viewer_invocations") == 1
        and counters.get("automatic_viewer_retries") == 0
        and counters.get("external_or_preexisting_process_signals") == 0
    )


def _next_viewer_invocation(current_count: int) -> int:
    if current_count != 0:
        raise RuntimeError("D382 second Viewer invocation rejected")
    return 1


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload, normalized = _serialize_json(value)
    del payload
    compact = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(compact).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object JSON: {path}")
    return value


def _json_native(value: Any) -> Any:
    """Convert NumPy scalars inside JSON containers to JSON-native values."""
    import numpy as np

    if isinstance(value, np.ndarray):
        raise TypeError("numpy.ndarray is outside scalar-normalization scope")
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON evidence dictionary keys must be strings")
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def _serialize_json(value: Any) -> tuple[str, Any]:
    """Finish normalization and JSON encoding before a destination is opened."""
    normalized = _json_native(value)
    if isinstance(normalized, dict):
        artifact = normalized.get("artifact")
        if isinstance(artifact, str) and artifact.startswith("D381_"):
            normalized = dict(normalized)
            normalized["inherited_d381_artifact"] = artifact
            normalized["artifact"] = "D382_" + artifact[len("D381_") :]
    payload = json.dumps(
        normalized,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    if json.loads(payload) != normalized:
        raise RuntimeError("JSON serialization round-trip changed evidence")
    return payload + "\n", normalized


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    """Serialize completely, then exclusively create and write the file."""
    payload, _ = _serialize_json(value)
    payload_bytes = payload.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload_bytes)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _status_paths() -> list[str]:
    return _git("status", "--short").splitlines()


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _source_hashes() -> dict[str, str]:
    return {
        "d382_script": _sha(SCRIPT_PATH),
        "frozen_d381_script": _sha(BASE_SCRIPT),
        "start_here_active_case_authorization": _sha(START_HERE),
        "rerun_contract": _sha(RERUN_CONTRACT),
    }


def _input_hashes() -> dict[str, str]:
    values = {
        f"d380::{name}": digest
        for name, digest in base._input_hashes().items()
    }
    values.update(
        {
            f"d381::{name}": _sha(path)
            for name, path in D381_INPUT_PATHS.items()
        }
    )
    return values


def _expected_input_hashes() -> dict[str, str]:
    values = {
        f"d380::{name}": digest
        for name, digest in base.D380_INPUT_HASHES.items()
    }
    values.update(
        {
            f"d381::{name}": digest
            for name, digest in D381_INPUT_HASHES.items()
        }
    )
    return values


def _dependency_versions() -> dict[str, str]:
    return {
        "matplotlib": importlib.metadata.version("matplotlib"),
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("pillow"),
        "psutil": importlib.metadata.version("psutil"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "rerun_sdk": importlib.metadata.version("rerun-sdk"),
    }


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
                _json_native(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _guarded_base_run(
    command: list[str],
    *,
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Allow the inherited headless Viewer command to start at most once."""
    global _ACTUAL_VIEWER_INVOCATIONS

    is_viewer = "--headless" in command and "--screenshot-to" in command
    if is_viewer:
        _ACTUAL_VIEWER_INVOCATIONS = _next_viewer_invocation(
            _ACTUAL_VIEWER_INVOCATIONS
        )
        _phase(
            "viewer_invocation_start",
            actual_viewer_invocations=_ACTUAL_VIEWER_INVOCATIONS,
            command=command,
        )
    result = _ORIGINAL_BASE_RUN(command, timeout=timeout, env=env)
    if is_viewer:
        _phase(
            "viewer_invocation_complete",
            actual_viewer_invocations=_ACTUAL_VIEWER_INVOCATIONS,
            returncode=result.get("returncode"),
            timed_out=result.get("timed_out"),
        )
    return result


def _configure_frozen_base() -> None:
    """Redirect the frozen D381 implementation into the D382 output path."""
    overrides = {
        "CASE": CASE,
        "ATTEMPT": ATTEMPT,
        "OUT_DIR": OUT_DIR,
        "SCRIPT_PATH": SCRIPT_PATH,
        "START_HERE": START_HERE,
        "PREREG_PATH": PREREG_PATH,
        "PHASE_PATH": PHASE_PATH,
        "INVOCATION_PATH": INVOCATION_PATH,
        "WORKER_STDOUT": WORKER_STDOUT,
        "WORKER_STDERR": WORKER_STDERR,
        "WORKER_CLAIM": WORKER_CLAIM,
        "SUPERVISOR_PATH": SUPERVISOR_PATH,
        "BOARD_PATH": BOARD_PATH,
        "LAYOUT_VALIDATION": LAYOUT_VALIDATION,
        "BASE_COPY_PATH": BASE_COPY_PATH,
        "RECORDING_ONLY_PATH": RECORDING_ONLY_PATH,
        "OVERLAY_RRD_PATH": OVERLAY_RRD_PATH,
        "RBL_PATH": RBL_PATH,
        "PRESENTATION_RRD_PATH": PRESENTATION_RRD_PATH,
        "RECORDING_EQUIVALENCE": RECORDING_EQUIVALENCE,
        "RERUN_VALIDATION": RERUN_VALIDATION,
        "RERUN_SCREENSHOT": RERUN_SCREENSHOT,
        "VIEWER_RECEIPT": VIEWER_RECEIPT,
        "MANUAL_TEMPLATE": MANUAL_TEMPLATE,
        "MANUAL_INSPECTION": MANUAL_INSPECTION,
        "COMPLETION_PATH": COMPLETION_PATH,
        "NEW_VARIABLES": NEW_VARIABLES,
        "WATCHDOG_SECONDS": WATCHDOG_SECONDS,
        "VIEWER_TIMEOUT_SECONDS": VIEWER_TIMEOUT_SECONDS,
        "SCOPE_COUNTERS": SCOPE_COUNTERS,
        "_write_json_x": _write_json_x,
        "_canonical_sha": _canonical_sha,
        "_run": _guarded_base_run,
    }
    for name, value in overrides.items():
        setattr(base, name, value)


_configure_frozen_base()


def prepare() -> int:
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only output already exists: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")

    import numpy as np

    start_text = START_HERE.read_text(encoding="utf-8")
    dependencies = _dependency_versions()
    d381_failure = _read_json(D381_INPUT_PATHS["d381_fail_attestation"])
    d381_supervisor = _read_json(D381_INPUT_PATHS["d381_supervisor"])
    d381_partial = _read_json(D381_INPUT_PATHS["d381_partial_inspection"])
    d380_facts = base._extract_facts(base._read_json(base.D380_EVIDENCE))

    truncated_rejected = False
    try:
        json.loads(
            D381_INPUT_PATHS["d381_truncated_layout"].read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError:
        truncated_rejected = True

    nested_numpy = {
        "bool": np.bool_(True),
        "float": np.float64(1.25),
        "int": np.int64(7),
        "tuple": (np.float32(2.5), np.float32(3.5)),
        "nested": [np.bool_(False), {"value": np.int32(9)}],
    }
    normalized = _json_native(nested_numpy)
    normalized_payload, normalized_roundtrip = _serialize_json(nested_numpy)

    negative_exception = None
    try:
        _write_json_x(NEGATIVE_PARTIAL_PATH, {"bad": object()})
    except TypeError as exc:
        negative_exception = repr(exc)
    partial_guard_pass = (
        negative_exception is not None and not NEGATIVE_PARTIAL_PATH.exists()
    )
    nan_exception = None
    try:
        _write_json_x(
            NEGATIVE_NAN_PATH,
            {"bad": np.float64(float("nan"))},
        )
    except ValueError as exc:
        nan_exception = repr(exc)
    nan_guard_pass = (
        nan_exception is not None and not NEGATIVE_NAN_PATH.exists()
    )
    raw_numpy_bool_json_rejected = False
    try:
        json.dumps(np.bool_(True))
    except TypeError:
        raw_numpy_bool_json_rejected = True
    ndarray_rejected = False
    try:
        _json_native(np.asarray([np.float32(1.0)]))
    except TypeError:
        ndarray_rejected = True
    nonstring_key_rejected = False
    try:
        _json_native({1: np.bool_(True)})
    except TypeError:
        nonstring_key_rejected = True
    second_viewer_rejected = False
    try:
        _next_viewer_invocation(1)
    except RuntimeError:
        second_viewer_rejected = True
    second_worker_rejected = not _execution_contract_valid(
        {
            **SCOPE_COUNTERS,
            "actual_offline_presentation_workers": 2,
        }
    )

    wrapper_imports = _import_roots(SCRIPT_PATH)
    base_imports = _import_roots(BASE_SCRIPT)
    checks = {
        "all_d380_and_d381_inputs_exist": (
            all(path.is_file() for path in base.D380_INPUT_PATHS.values())
            and all(path.is_file() for path in D381_INPUT_PATHS.values())
        ),
        "all_input_hashes_exact": (
            _input_hashes() == _expected_input_hashes()
        ),
        "d381_failure_verdict_exact": (
            d381_failure.get("operational_verdict")
            == "D381_BOARD_VALIDATION_JSON_SERIALIZATION_FAIL_STOP"
            and d381_failure.get("completion_pass") is False
        ),
        "d381_worker_one_retry_zero": (
            d381_supervisor.get("actual_offline_presentation_workers") == 1
            and d381_supervisor.get("automatic_worker_retries") == 0
            and d381_supervisor.get("returncode") == 1
        ),
        "d381_viewer_actual_zero": (
            d381_failure.get("execution", {}).get(
                "actual_rerun_viewer_invocations"
            )
            == 0
        ),
        "d381_partial_inspection_failed": (
            d381_partial.get("pass") is False
        ),
        "d381_board_hash_exact": (
            _sha(D381_INPUT_PATHS["d381_board"])
            == EXPECTED_D381_BOARD_SHA256
        ),
        "d381_truncated_json_rejected": truncated_rejected,
        "d381_truncated_json_exact_144_bytes": (
            D381_INPUT_PATHS["d381_truncated_layout"].stat().st_size == 144
        ),
        "d380_facts_exact": d380_facts == base.EXPECTED_FACTS,
        "recursive_numpy_scalars_are_json_native": (
            type(normalized["bool"]) is bool
            and type(normalized["float"]) is float
            and type(normalized["int"]) is int
            and all(type(value) is float for value in normalized["tuple"])
            and type(normalized["nested"][0]) is bool
            and type(normalized["nested"][1]["value"]) is int
        ),
        "normalized_roundtrip_exact": (
            json.loads(normalized_payload) == normalized_roundtrip == normalized
        ),
        "serialize_failure_precedes_file_creation": partial_guard_pass,
        "negative_partial_path_absent": not NEGATIVE_PARTIAL_PATH.exists(),
        "nan_failure_precedes_file_creation": nan_guard_pass,
        "negative_nan_path_absent": not NEGATIVE_NAN_PATH.exists(),
        "wrapper_forbidden_imports_absent": not (
            wrapper_imports & FORBIDDEN_IMPORT_ROOTS
        ),
        "frozen_base_forbidden_imports_absent": not (
            base_imports & FORBIDDEN_IMPORT_ROOTS
        ),
        "interpreter_exact": (
            Path(sys.executable).resolve() == ISAACLAB_PYTHON.resolve()
        ),
        "dependency_versions_exact": dependencies
        == {
            "matplotlib": "3.10.3",
            "numpy": "1.26.0",
            "pillow": "11.3.0",
            "psutil": "5.9.8",
            "pyarrow": "23.0.1",
            "rerun_sdk": "0.34.1",
        },
        "rerun_cli_exists_and_is_executable": (
            base.RERUN_CLI.is_file()
            and os.access(base.RERUN_CLI, os.X_OK)
        ),
        "required_fonts_exist": (
            base.FONT_REGULAR.is_file() and base.FONT_BOLD.is_file()
        ),
        "execution_contract_exact": _execution_contract_valid(SCOPE_COUNTERS),
        "start_here_authorizes_exact_case_variables_and_path": (
            "D382 [d381_layout_validation_native_scalar_serialization_repair]"
            in start_text
            and _rel(OUT_DIR) in start_text
            and all(variable in start_text for variable in NEW_VARIABLES)
        ),
        "head_equals_origin_master": (
            _git("rev-parse", "HEAD")
            == _git("rev-parse", "origin/master")
        ),
    }
    controls = {
        "wrong_d381_board_hash_rejected": (
            "0" * 64 != EXPECTED_D381_BOARD_SHA256
        ),
        "numpy_bool_is_not_builtin_bool_before_normalization": (
            type(np.bool_(True)) is not bool
        ),
        "numpy_bool_becomes_builtin_bool": (
            type(_json_native(np.bool_(True))) is bool
        ),
        "numpy_float_becomes_builtin_float": (
            type(_json_native(np.float64(1.0))) is float
        ),
        "numpy_int_becomes_builtin_int": (
            type(_json_native(np.int64(1))) is int
        ),
        "raw_numpy_bool_is_rejected_by_json_encoder": (
            raw_numpy_bool_json_rejected
        ),
        "numpy_array_outside_scalar_scope_rejected": ndarray_rejected,
        "nonstring_dictionary_key_rejected": nonstring_key_rejected,
        "unserializable_object_rejected_without_partial_file": (
            partial_guard_pass
        ),
        "nan_rejected_without_partial_file": nan_guard_pass,
        "second_worker_request_rejected": (
            second_worker_rejected
        ),
        "second_viewer_request_rejected": (
            second_viewer_rejected
        ),
        "g0a_flip_rejected": (
            {**d380_facts, "g0a_pass": True} != base.EXPECTED_FACTS
        ),
        "p34_identity_flip_rejected": (
            {**d380_facts, "p34_identity_pass": True}
            != base.EXPECTED_FACTS
        ),
        "physics_counter_nonzero_rejected": (
            {**SCOPE_COUNTERS, "physics_steps": 1} != SCOPE_COUNTERS
        ),
    }
    prereg = {
        "artifact": "D382_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Repair only D381 layout-validation JSON serialization while "
            "preserving the frozen board and presentation contract."
        ),
        "new_variables": NEW_VARIABLES,
        "immutable_inputs": {
            "d380": [
                {
                    "name": name,
                    "path": _rel(base.D380_INPUT_PATHS[name]),
                    "sha256": base.D380_INPUT_HASHES[name],
                }
                for name in sorted(base.D380_INPUT_PATHS)
            ],
            "d381": [
                {
                    "name": name,
                    "path": _rel(D381_INPUT_PATHS[name]),
                    "sha256": D381_INPUT_HASHES[name],
                }
                for name in sorted(D381_INPUT_PATHS)
            ],
        },
        "frozen_d380_facts": d380_facts,
        "frozen_d380_facts_sha256": _canonical_sha(d380_facts),
        "frozen_d381_board_sha256": EXPECTED_D381_BOARD_SHA256,
        "registered_repairs": {
            "json_native_recursive_scalar_normalization": (
                "dict/list/tuple recursion plus numpy.generic.item(); "
                "numpy.ndarray remains rejected"
            ),
            "serialize_before_exclusive_create": (
                "normalization, json.dumps, and json.loads verification finish "
                "before path.open('xb')"
            ),
        },
        "registered_unchanged": {
            "d381_board_pixels_bitexact": True,
            "d381_board_layout_and_text": True,
            "d381_rerun_projection_and_blueprint": True,
            "d380_numeric_verdict": base.EXPECTED_FACTS["verdict"],
            "p34_identity_pass": False,
            "g0a_pass": False,
        },
        "case_namespace_scaffolding": {
            "top_level_d381_artifact_label_rewritten_to_d382": True,
            "original_label_preserved_as_inherited_d381_artifact": True,
            "scientific_or_presentation_variable": False,
        },
        "registered_execution": {
            **SCOPE_COUNTERS,
            "bounded_worker_watchdog_seconds": WATCHDOG_SECONDS,
            "bounded_viewer_timeout_seconds": VIEWER_TIMEOUT_SECONDS,
            "watchdog_signal_scope": (
                "D382-owned child process group after timeout only"
            ),
        },
        "normalization_preflight": {
            "normalized_value": normalized,
            "serialized_sha256": hashlib.sha256(
                normalized_payload.encode("utf-8")
            ).hexdigest(),
            "unserializable_exception": negative_exception,
            "negative_partial_path": _rel(NEGATIVE_PARTIAL_PATH),
            "negative_partial_path_exists": NEGATIVE_PARTIAL_PATH.exists(),
            "nan_exception": nan_exception,
            "negative_nan_path": _rel(NEGATIVE_NAN_PATH),
            "negative_nan_path_exists": NEGATIVE_NAN_PATH.exists(),
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
        raise RuntimeError(f"D382 preregistration failed: {checks}")
    return 0


def worker() -> int:
    _phase("worker_start", pid=os.getpid())
    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D382 preregistration did not pass")
    if invocation.get("preregistration_sha256") != _sha(PREREG_PATH):
        raise RuntimeError("D382 invocation not bound to preregistration")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D382 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D382 inputs changed after preregistration")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("D382 dirty baseline changed after preregistration")

    evidence = base._read_json(base.D380_EVIDENCE)
    facts = base._extract_facts(evidence)
    if facts != prereg["frozen_d380_facts"] or facts != base.EXPECTED_FACTS:
        raise RuntimeError("D382 frozen D380 facts changed")

    board = base._render_board(facts)
    layout = _read_json(LAYOUT_VALIDATION)
    board_bitexact = (
        board["sha256"] == EXPECTED_D381_BOARD_SHA256
        and _sha(BOARD_PATH) == _sha(D381_INPUT_PATHS["d381_board"])
    )
    if not board_bitexact:
        raise RuntimeError(
            f"D382 board pixels changed: {board['sha256']} "
            f"!= {EXPECTED_D381_BOARD_SHA256}"
        )
    _phase(
        "board_layout_serialization_repaired",
        board_sha256=board["sha256"],
        layout_validation_sha256=_sha(LAYOUT_VALIDATION),
        board_bitexact_with_d381=True,
    )

    presentation = base._build_presentation(facts)
    _phase(
        "presentation_archive_finalized",
        presentation_rrd_sha256=presentation["presentation_rrd"]["sha256"],
    )
    validation = base._validate_and_capture(
        base._read_json(base.D380_RERUN_VALIDATION),
        presentation,
    )
    _phase(
        "single_viewer_capture_complete",
        screenshot_sha256=validation["screenshot"]["sha256"],
    )

    manual_template = {
        "artifact": "D382_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "frozen_d380_facts_sha256": _canonical_sha(facts),
        "frozen_d381_board_sha256": EXPECTED_D381_BOARD_SHA256,
        "board": _file_record(BOARD_PATH),
        "layout_validation": _file_record(LAYOUT_VALIDATION),
        "presentation_rrd": _file_record(PRESENTATION_RRD_PATH),
        "rbl": _file_record(RBL_PATH),
        "rerun_validation": _file_record(RERUN_VALIDATION),
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "rerun_screenshot": validation["screenshot"],
        "required_check_keys": sorted(base.MANUAL_CHECK_KEYS),
        "inspection_checks": {
            key: None for key in sorted(base.MANUAL_CHECK_KEYS)
        },
        "observations": [],
        "inspector_result": None,
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)

    post_checks = {
        "board_bitexact_with_d381": board_bitexact,
        "layout_validation_valid_json": isinstance(layout, dict),
        "layout_validation_pass": layout.get("pass") is True,
        "layout_artifact_is_d382": (
            layout.get("artifact") == "D382_BOARD_LAYOUT_VALIDATION_V1"
        ),
        "recording_equivalence_pass": (
            _read_json(RECORDING_EQUIVALENCE).get("pass") is True
        ),
        "rerun_validation_pass": validation.get("pass") is True,
        "viewer_invocation_exactly_one": (
            _read_json(VIEWER_RECEIPT).get("viewer_invocations") == 1
            and _ACTUAL_VIEWER_INVOCATIONS == 1
        ),
        "viewer_retry_zero": (
            _read_json(VIEWER_RECEIPT).get("automatic_viewer_retries") == 0
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
        "frozen_facts_still_exact": facts == base.EXPECTED_FACTS,
        "p34_identity_still_false": facts["p34_identity_pass"] is False,
        "g0a_still_false": facts["g0a_pass"] is False,
        "all_forbidden_counters_zero": all(
            value == 0
            for name, value in SCOPE_COUNTERS.items()
            if name
            not in {
                "actual_offline_presentation_workers",
                "rerun_viewer_invocations",
            }
        ),
    }
    claim = {
        "artifact": "D382_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "pid": os.getpid(),
        "preregistration": _file_record(PREREG_PATH),
        "frozen_d380_facts": facts,
        "frozen_d380_facts_sha256": _canonical_sha(facts),
        "serialization_repair": {
            "new_variables": NEW_VARIABLES,
            "layout_validation": _file_record(LAYOUT_VALIDATION),
            "layout_validation_valid_json": True,
            "board_bitexact_with_d381": board_bitexact,
        },
        "board": board,
        "presentation": presentation,
        "rerun_validation": _file_record(RERUN_VALIDATION),
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "manual_template": _file_record(MANUAL_TEMPLATE),
        "scope_counters": SCOPE_COUNTERS,
        "checks": post_checks,
        "pass": all(post_checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_claim_written", worker_claim_sha256=_sha(WORKER_CLAIM))
    if not claim["pass"]:
        raise RuntimeError(f"D382 worker post-check failed: {post_checks}")
    return 0


def run_supervisor() -> int:
    if not _execution_contract_valid(SCOPE_COUNTERS):
        raise RuntimeError("D382 execution contract changed")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D382 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D382 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D382 inputs changed after preregistration")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("D382 dirty baseline changed after preregistration")

    command = [sys.executable, "-B", str(SCRIPT_PATH), "--stage", "worker"]
    invocation = {
        "artifact": "D382_OFFLINE_PRESENTATION_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "cwd": str(REPO),
        "preregistration_sha256": _sha(PREREG_PATH),
        "source_hashes": _source_hashes(),
        "input_hashes": _input_hashes(),
        "worker_spawn_count_registered": 1,
        "automatic_worker_retry_count_registered": 0,
        "rerun_viewer_count_registered": 1,
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

    claim = _read_json(WORKER_CLAIM) if WORKER_CLAIM.is_file() else {}
    required = {
        "worker_claim": WORKER_CLAIM.is_file(),
        "board": BOARD_PATH.is_file(),
        "layout_validation": LAYOUT_VALIDATION.is_file(),
        "recording_equivalence": RECORDING_EQUIVALENCE.is_file(),
        "presentation_rrd": PRESENTATION_RRD_PATH.is_file(),
        "rbl": RBL_PATH.is_file(),
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
        and all(required.values())
        and claim.get("pass") is True
        and _source_hashes() == prereg["source_hashes"]
        and _input_hashes() == prereg["input_hashes"]
        and _status_paths() == prereg["registered_dirty_baseline"]
    )
    supervisor = {
        "artifact": "D382_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "worker_pid": process.pid,
        "worker_process_group": pgid,
        "actual_offline_presentation_workers": 1,
        "automatic_worker_retries": 0,
        "registered_rerun_viewer_invocations": 1,
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
        operational_pass=operational_pass,
    )
    return 0 if operational_pass else 1


def finalize() -> int:
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        WORKER_CLAIM,
        SUPERVISOR_PATH,
        BOARD_PATH,
        LAYOUT_VALIDATION,
        RECORDING_EQUIVALENCE,
        PRESENTATION_RRD_PATH,
        RBL_PATH,
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
        raise RuntimeError(f"D382 finalize missing files: {missing}")

    prereg = _read_json(PREREG_PATH)
    claim = _read_json(WORKER_CLAIM)
    supervisor = _read_json(SUPERVISOR_PATH)
    manual_template = _read_json(MANUAL_TEMPLATE)
    manual = _read_json(MANUAL_INSPECTION)
    facts = base._extract_facts(base._read_json(base.D380_EVIDENCE))
    manual_checks = manual.get("inspection_checks", {})
    checks = {
        "preregistration_pass": prereg.get("pass") is True,
        "worker_claim_pass": claim.get("pass") is True,
        "supervisor_pass": supervisor.get("pass") is True,
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
        "frozen_facts_exact": facts == base.EXPECTED_FACTS,
        "board_bitexact_with_d381": (
            _sha(BOARD_PATH) == EXPECTED_D381_BOARD_SHA256
        ),
        "layout_validation_valid_and_pass": (
            _read_json(LAYOUT_VALIDATION).get("pass") is True
        ),
        "recording_equivalence_pass": (
            _read_json(RECORDING_EQUIVALENCE).get("pass") is True
        ),
        "rerun_validation_pass": (
            _read_json(RERUN_VALIDATION).get("pass") is True
        ),
        "viewer_receipt_pass": (
            _read_json(VIEWER_RECEIPT).get("pass") is True
        ),
        "manual_artifact_exact": (
            manual.get("artifact") == "D382_MANUAL_VISUAL_INSPECTION_V1"
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
            set(manual_checks) == base.MANUAL_CHECK_KEYS
        ),
        "manual_checks_all_true": (
            set(manual_checks) == base.MANUAL_CHECK_KEYS
            and all(value is True for value in manual_checks.values())
        ),
        "manual_observations_nonempty": bool(manual.get("observations")),
        "manual_inspector_result_pass": (
            manual.get("inspector_result") == "PASS"
        ),
        "manual_visual_inspection_pass": manual.get("pass") is True,
        "template_frozen_facts_exact": (
            manual_template.get("frozen_d380_facts_sha256")
            == _canonical_sha(facts)
        ),
        "p34_identity_still_false": facts["p34_identity_pass"] is False,
        "g0a_still_false": facts["g0a_pass"] is False,
    }
    completion_pass = all(checks.values())
    completion = {
        "artifact": "D382_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "completion_pass": completion_pass,
        "verdict": (
            "D382_D381_LAYOUT_VALIDATION_NATIVE_SCALAR_SERIALIZATION_REPAIR_PASS"
            if completion_pass
            else "D382_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"
        ),
        "preserved_d380_numeric_verdict": facts["verdict"],
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
        "scope_counters": SCOPE_COUNTERS,
        "next_authorization_boundary": (
            "P34 representation repair/live identity, 29x50 target rebase, "
            "and all Isaac/PhysX/physics/q5/contact remain unapproved."
        ),
        "artifacts": {
            path.name: _file_record(path)
            for path in [
                BOARD_PATH,
                LAYOUT_VALIDATION,
                RECORDING_EQUIVALENCE,
                PRESENTATION_RRD_PATH,
                RBL_PATH,
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
