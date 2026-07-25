#!/usr/bin/env python3
"""D379 offline-only resume of the full P34 live-identity classifier.

The script never launches Isaac, Kit, PhysX, Warp, Hydra, Fabric, or USD.  It
reads the immutable D377 clean-exit worker evidence, the D378 corrected
workload authority, the frozen D372 P34 geometry, the D373 callback witnesses,
and the D343 typed-Float32 authority.  D375's frozen classifier functions and
thresholds are reused without a new callback/property request.

Stages are forward-only:

* ``prepare`` freezes the one classifier-resume variable and all input hashes;
* ``audit`` performs exactly one offline classification with no retry and
  emits the evidence, exact 1920x1080 board, save-only RRD/RBL, and manual
  inspection template;
* ``finalize`` binds the manual inspection and all artifact hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
while str(REPO) in sys.path:
    sys.path.remove(str(REPO))
sys.path.insert(0, str(REPO))

CASE_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d379"
ATTEMPT = "attempt2_d372_measurement_field_repair"
OUT_DIR = CASE_ROOT / ATTEMPT
PREREG_PATH = OUT_DIR / "d379_preregistration.json"
PHASE_PATH = OUT_DIR / "d379_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d379_offline_audit_invocation.json"
CLAIM_PATH = OUT_DIR / "d379_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d379_offline_worker_supervisor.json"
STDOUT_PATH = OUT_DIR / "d379_offline_worker_stdout.log"
STDERR_PATH = OUT_DIR / "d379_offline_worker_stderr.log"
EVIDENCE_PATH = OUT_DIR / "d379_p34_full_live_identity_evidence.json"
BOARD_PATH = OUT_DIR / "d379_p34_full_live_identity_1920x1080.png"
RRD_PATH = OUT_DIR / "d379_p34_full_live_identity.rrd"
RBL_PATH = OUT_DIR / "d379_p34_full_live_identity.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d379_rerun_validation.json"
RERUN_PNG_PATH = OUT_DIR / "d379_rerun_inspection.png"
AUTOMATED_PATH = OUT_DIR / "d379_automated_summary.json"
MANUAL_TEMPLATE_PATH = OUT_DIR / "d379_manual_visual_inspection_template.json"
MANUAL_PATH = OUT_DIR / "d379_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d379_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d379_runtime_exception.json"
ATTEMPT1_DIR = (
    CASE_ROOT / "attempt1_p34_full_live_identity_classifier_resume"
)
ATTEMPT1_PREREG = ATTEMPT1_DIR / "d379_preregistration.json"
ATTEMPT1_EXCEPTION = ATTEMPT1_DIR / "d379_runtime_exception.json"
ATTEMPT1_PHASE = ATTEMPT1_DIR / "d379_phase_markers.jsonl"

HARNESS = Path(__file__).resolve()
START_HERE = REPO / "START_HERE.md"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
D375_CLASSIFIER = (
    REPO / "sim_scripts/cyl34_top_view_d375_p34_live_asset_identity_contract_repair.py"
)
D373_CONTROLLER = (
    REPO / "sim_scripts/cyl34_top_view_d373_p34_live_asset_identity_preflight.py"
)

D377_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d377/"
    "attempt1_stagecache_erase_before_close_localization"
)
D377_RAW = D377_DIR / "d377_worker_raw_summary.json"
D377_PRECLOSE = D377_DIR / "d377_worker_preclose_sentinel.json"
D377_SUPERVISOR = D377_DIR / "d377_worker_supervisor.json"
D377_PREREG = D377_DIR / "d377_preregistration.json"
D377_WITNESSES = D377_DIR / "callback_witnesses"

D378_EVIDENCE = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d378/"
    "attempt2_preregistration_status_order_repair/"
    "d378_workload_authority_repair_evidence.json"
)

D375_PREREG = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d375/"
    "attempt2_external_gpu_attestation_repair/d375_preregistration.json"
)
D372_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d372/"
    "attempt2_external_schema_path_repair"
)
D372_GEOMETRY = D372_DIR / "d372_professor_semantic_candidate_geometry.json"
D372_EVIDENCE = D372_DIR / "d372_professor_semantic_candidate_evidence.json"
D372_COMPLETION = D372_DIR / "d372_completion_summary.json"
D373_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d373/"
    "attempt1_p34_live_asset_identity_preflight"
)
D373_RAW = D373_DIR / "d373_worker_raw_summary.json"
D373_WITNESSES = D373_DIR / "callback_witnesses"
D343_SUMMARY = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d343/"
    "d343_usd_typed_float_readback_summary.json"
)
D343_EVIDENCE = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d343/"
    "d343_usd_typed_float_readback_evidence.json"
)
D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"

EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = ["p34_full_live_identity_classifier_resume_v1"]
EXPECTED_COUNTS = {"link5": 16, "gripper_link": 18}
EXPECTED_PROPERTY_ROWS = {"link5": 17, "gripper_link": 19}
SURFACE_TOL_M = 0.0001
BOUNDS_TOL_M = 0.0001
AUTHORED_CALLBACK_VOLUME_REL_TOL = 0.005
CALLBACK_PROPERTY_VOLUME_REL_TOL = 0.05
AUTHORED_MASS_STATE_ATOL = 1.0e-12
PROPERTY_MASS_STATE_ATOL = 1.0e-9
POLYGON_PLANE_RESIDUAL_TOL_M = 1.0e-5
OFFLINE_WORKER_TIMEOUT_S = 360.0
OFFLINE_WORKER_TERM_GRACE_S = 10.0

EXPECTED_TYPED_FLOAT = {
    "expected_bits_hex": "0x38d1b717",
    "expected_type_name": "float",
    "expected_value_m": 0.00009999999747378752,
    "decimal_0_0001_comparator_forbidden": True,
    "d343_retest_count": 0,
}
EXPECTED_CORRECTED_WORKLOAD_SHA = (
    "28aadb5ff26270039df58f7cd06080bf7afcdec001402e886a6edf1483fdfe31"
)
EXPECTED_NORMALIZED_WITNESS_SHA = (
    "0a56d7900470f6f75d5f63ac415d7d0f4cca5c5d941951280387ae2378abfe8c"
)
EXPECTED_NORMALIZED_PROPERTY_SHA = (
    "4710c18232e2d2259c569d01b6326bbea20b36507e5aeb9a85fbe15ca94f7c1f"
)
EXPECTED_CALLBACK_COUNTS = {
    "callbacks": 34,
    "vertices": 314,
    "indices": 1016,
    "original_polygons": 262,
}

EXPECTED_FIXED_HASHES = {
    ATTEMPT1_PREREG: "7d02329010f3622fc8b6e738d7ca5d255e070957329a1a48ec5ef6df3f909ba9",
    ATTEMPT1_EXCEPTION: "5c8c5778bbfb4341d7d5f03c6c685e31ca9686f31142a3142f23e6597eef665a",
    ATTEMPT1_PHASE: "7659744965f56be5d3193e0bc1037ed0f98f513557a0c7bbb10c389a4d87d227",
    D377_RAW: "f14d2cf38cffc03a3121719a4dac0a62d612b46926a5ff6afcc10cd143717fb1",
    D377_PRECLOSE: "4a312e20b4444f84897864e013b1f4eb74ce57aeb52377bfaa0ee9e3d068cc89",
    D377_SUPERVISOR: "1c0a4754da7fa0bae748e6c1095a1c39982ab50cef6202c6918f742fb635ce49",
    D377_PREREG: "01cc3f55e2bb5e7718b5eed210501a37622bc731dbe9e4bc623f97e91f885d87",
    D378_EVIDENCE: "e9c3d1cadf9cc9516d0d08792a44b6d824fea7ac8cd0849dffc9a25f3bafda88",
    D375_PREREG: "4048cc8201029e4f4d196fe6f68e1f0fdfe90907627b20edeb57ca9a6709744b",
    D372_GEOMETRY: "12fd1f32c35dfb9ae36cbbb412f6a51536aa1cc07c2dc17d05a5d189f3ee83e4",
    D372_EVIDENCE: "d68f658089aaf838ff454e9d0b301ec3f602785a3a730b3c329aa7785010e984",
    D372_COMPLETION: "57f3ed8fe6f057d059980a78bb51be8e881d8300297a4f41def6ddf94ad0cf43",
    D373_RAW: "dd57da307acf6134487bcd1dfa4a847fd41f24832177421f6291c45b06091373",
    D343_SUMMARY: "880601aac768df38675603828258850aea796b6436a299c46f8cc489ed8b00da",
    D343_EVIDENCE: "95bb4e3787d300071f1bac22037814b732781cd72a69a0334a34a05a50ac920b",
    D375_CLASSIFIER: "70cb1bf9b0d518117fe90c848a9cefbdb95c8a0cabd4f3fa0fc5a37aae29c5e1",
    D373_CONTROLLER: "df5e404133ff22846cec469ccfdb969946288a72d6d040a3564a9bc3ed8ba2af",
}
EXPECTED_INVENTORY_SHA = {
    ATTEMPT1_DIR: "69254eb48e505d030f59178b3e9e0ae687bd282271a85db13d085cca539513cd",
    D377_WITNESSES: "afe7535f91c38f02f25c768f154afa361606c4a95019089f38eda93b939c4815",
    D373_WITNESSES: "e2333e7d50b4bdebf147ce94da62f05773783e856ed1528f1acfa2c3f54ce1ac",
    D334_SIDECAR: "86c3a8f58b0866458910d2cab13da69f04c2dba5ddfc430a8b648367d759fef2",
}
EXPECTED_INVENTORY_COUNTS = {
    ATTEMPT1_DIR: 3,
    D377_WITNESSES: 34,
    D373_WITNESSES: 68,
    D334_SIDECAR: 3,
}

STRICT_ZERO_SCOPE = [
    "isaac_launches",
    "kit_launches",
    "physx_calls",
    "usd_reads_or_writes",
    "collider_materializations_or_regenerations",
    "automatic_decomposition_sweeps",
    "physics_steps",
    "public_forwards",
    "q5_commands",
    "q5_samples",
    "contact_queries",
    "cylinder_creates_or_writes",
    "target_ik_path_pose_changes",
    "material_mass_actuator_physics_setting_changes",
]
EXPECTED_D378_BASELINE_DIRTY_PATHS = {
    "START_HERE.md",
    "claudedocs/DECISIONS.md",
    "claudedocs/EXPERIMENT_LEDGER.md",
    "claudedocs/runtime_logs/grasp_track/g0a_d378/",
    (
        "claudedocs/session_20260724_grasp_g0a_d378_"
        "ephemeral_identifier_provenance_and_workload_authority_repair.md"
    ),
    "roarm_rl/viz_debug.py",
    "sim_scripts/cyl34_top_view_d378_attempt3_ascii_board_layout_repair.py",
    (
        "sim_scripts/cyl34_top_view_d378_d377_ephemeral_identifier_"
        "provenance_and_workload_authority_repair.py"
    ),
}
SOURCE_RUNTIME_ZERO_KEYS = [
    "automatic_retries",
    "derivative_asset_materializations",
    "usd_stage_file_writes",
    "simulation_context_constructions",
    "resets",
    "timeline_play_requests",
    "timeline_commit_requests",
    "physics_steps",
    "public_forwards",
    "q5_commands",
    "q5_samples",
    "contact_queries",
    "cylinder_creates_or_writes",
    "target_ik_path_pose_changes",
    "automatic_convex_decomposition_sweeps",
    "material_mass_actuator_physics_setting_changes",
    "isaac_hydra_renders",
    "simulation_app_update_pumps",
    "d343_contract_retests",
]

VERDICT_PASS = "D379_P34_FULL_LIVE_IDENTITY_CLASSIFIER_RESUME_PASS_NO_NEW_ISAAC"
VERDICT_FAIL = "D379_P34_FULL_LIVE_IDENTITY_CLASSIFIER_RESUME_FAIL_STOP"
OBSERVABILITY_FAIL = "D379_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"

ROLE_COLORS_HEX = {
    "structural_body": "#0072B2",
    "connector_support": "#E69F00",
    "fixed_jaw": "#F0E442",
    "fixed_jaw_backbone": "#D97800",
    "moving_support": "#CC79A7",
    "moving_jaw": "#009E73",
    "moving_jaw_backbone": "#00695C",
}
ROLE_COLORS_RGBA = {
    "structural_body": [0, 114, 178, 185],
    "connector_support": [230, 159, 0, 185],
    "fixed_jaw": [240, 228, 66, 210],
    "fixed_jaw_backbone": [217, 120, 0, 205],
    "moving_support": [204, 121, 167, 185],
    "moving_jaw": [0, 158, 115, 210],
    "moving_jaw_backbone": [0, 105, 92, 205],
}

OFFICIAL_SOURCES = [
    {
        "title": "Omni Physics 107.3 - Rigid Bodies",
        "url": (
            "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/"
            "dev_guide/rigid_bodies_articulations/rigid_bodies.html"
        ),
        "applicable_version": "installed omni.physx 107.3.26",
        "use": "one rigid body may own multiple child colliders",
    },
    {
        "title": "Omni Physics 107.3 - Query The Mass and Volume",
        "url": (
            "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/"
            "dev_guide/mass_inertia_queries.html"
        ),
        "applicable_version": "installed omni.physx 107.3.26",
        "use": "VALID-only frozen property-query authority",
    },
    {
        "title": "Omni Physics 107.3 - Colliders",
        "url": (
            "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/"
            "dev_guide/rigid_bodies_articulations/collision.html"
        ),
        "applicable_version": "installed omni.physx 107.3.26",
        "use": "convex collider and compound-collider semantics",
    },
]


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
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return _rel(value)
    raise TypeError(type(value).__name__)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _phase(name: str, **fields: Any) -> None:
    ordinal = 1
    if PHASE_PATH.is_file():
        ordinal = (
            sum(
                1
                for line in PHASE_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            + 1
        )
    row = {
        "ordinal": ordinal,
        "phase": name,
        "pid": os.getpid(),
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                row, sort_keys=True, ensure_ascii=False, default=_json_default
            )
            + "\n"
        )
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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _forbidden_modules_loaded() -> list[str]:
    roots = ("omni", "isaacsim", "isaaclab", "warp", "pxr")
    return sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in roots)
    )


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


def _input_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in EXPECTED_FIXED_HASHES}


def _source_hashes() -> dict[str, str]:
    return {
        "harness": _sha(HARNESS),
        "viz_debug": _sha(VIZ_DEBUG),
        "rerun_contract": _sha(RERUN_CONTRACT),
        "frozen_d375_classifier": _sha(D375_CLASSIFIER),
        "frozen_d373_controller": _sha(D373_CONTROLLER),
    }


def _registered_dirty_baseline_snapshot() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for relative_path in sorted(EXPECTED_D378_BASELINE_DIRTY_PATHS):
        path = REPO / relative_path
        if path.is_file():
            rows[relative_path] = {
                "kind": "file",
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        elif path.is_dir():
            rows[relative_path] = {
                "kind": "directory",
                "inventory": _inventory(path),
            }
        else:
            rows[relative_path] = {"kind": "missing"}
    return {
        "entries": rows,
        "canonical_sha256": _canonical_sha(rows),
        "all_present": all(
            row["kind"] in {"file", "directory"} for row in rows.values()
        ),
    }


def _dependency_preflight() -> dict[str, Any]:
    modules = {
        "PIL": "Pillow",
        "matplotlib": "matplotlib",
        "trimesh": "trimesh",
        "rerun": "rerun-sdk",
    }
    rows = {
        module: {
            "module_spec_present": importlib.util.find_spec(module) is not None,
            "distribution": distribution,
            "version": _package_version(distribution),
        }
        for module, distribution in modules.items()
    }
    return {
        "modules": rows,
        "pass": all(
            row["module_spec_present"] and row["version"] is not None
            for row in rows.values()
        ),
    }


def _load_d375_classifier() -> Any:
    spec = importlib.util.spec_from_file_location(
        "d375_frozen_classifier_for_d379", D375_CLASSIFIER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen D375 classifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registered_contract() -> dict[str, Any]:
    return {
        "direct_authored_parts": EXPECTED_COUNTS,
        "proxy_aware_live_parts": EXPECTED_COUNTS,
        "property_rows_including_disabled_legacy": EXPECTED_PROPERTY_ROWS,
        "callback_channel": "one actual live composed path per P34 part; 34 total",
        "frozen_d373_callback": "immutable comparison channel only; no new request",
        "owner_flags": (
            "Robot/link5/gripper_link all non-instance and non-proxy before attach"
        ),
        "result_authority": (
            "rigid body and every collider must be VALID with exact paths"
        ),
        "surface_tolerance_m": SURFACE_TOL_M,
        "bounds_tolerance_m": BOUNDS_TOL_M,
        "authored_callback_topology_volume_relative": (
            AUTHORED_CALLBACK_VOLUME_REL_TOL
        ),
        "callback_property_volume_relative": CALLBACK_PROPERTY_VOLUME_REL_TOL,
        "mass_com_inertia_axes_atol": PROPERTY_MASS_STATE_ATOL,
        "volume_semantics": (
            "D348 callback original polygon topology; no new Qhull"
        ),
    }


def _d378_authority_exact(value: dict[str, Any]) -> bool:
    corrected = value.get("corrected_authority", {})
    return bool(
        value.get("pass") is True
        and value.get("verdict")
        == "D378_EPHEMERAL_IDENTIFIER_PROVENANCE_AND_WORKLOAD_AUTHORITY_REPAIR_PASS"
        and corrected.get("D375_corrected_workload_sha256")
        == corrected.get("D377_corrected_workload_sha256")
        == EXPECTED_CORRECTED_WORKLOAD_SHA
        and corrected.get("D375_normalized_witness_sha256")
        == corrected.get("D377_normalized_witness_sha256")
        == EXPECTED_NORMALIZED_WITNESS_SHA
        and corrected.get("D375_normalized_property_sha256")
        == corrected.get("D377_normalized_property_sha256")
        == EXPECTED_NORMALIZED_PROPERTY_SHA
        and corrected.get("D375_callback_counts")
        == corrected.get("D377_callback_counts")
        == EXPECTED_CALLBACK_COUNTS
        and corrected.get("selected_corrected_diff_count") == 0
        and corrected.get("normalized_witness_pair_diff_count") == 0
        and corrected.get("normalized_property_diff_count") == 0
    )


def _prepare_negative_controls(classifier: Any) -> dict[str, Any]:
    inherited = classifier._prepare_negative_controls()
    expected = {
        "instanceable_owner_is_rejected",
        "missing_live_path_breaks_bijection",
        "error_parsing_is_not_valid_measurement",
        "mutated_preclose_sha_breaks_authority",
    }
    return {
        "source": "frozen D375 prepare controls",
        "controls": inherited["controls"],
        "exact_key_set": set(inherited["controls"]) == expected,
        "pass": inherited.get("pass") is True
        and set(inherited["controls"]) == expected
        and all(value is True for value in inherited["controls"].values()),
    }


def _d377_authority(
    raw: dict[str, Any],
    preclose: dict[str, Any],
    supervisor: dict[str, Any],
    d378: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "raw_hash_exact": _sha(D377_RAW) == EXPECTED_FIXED_HASHES[D377_RAW],
        "preclose_hash_exact": _sha(D377_PRECLOSE)
        == EXPECTED_FIXED_HASHES[D377_PRECLOSE],
        "supervisor_hash_exact": _sha(D377_SUPERVISOR)
        == EXPECTED_FIXED_HASHES[D377_SUPERVISOR],
        "returncode_zero": supervisor.get("returncode") == 0,
        "not_timed_out": supervisor.get("timed_out") is False,
        "supervisor_operational_pass": supervisor.get("operational_pass") is True,
        "supervisor_hash_authority_pass": supervisor.get("hash_authority_pass")
        is True,
        "supervisor_effective_preanalysis_pass": supervisor.get(
            "effective_preanalysis_pass"
        )
        is True,
        "supervisor_pass": supervisor.get("pass") is True,
        "worker_spawn_exactly_one": supervisor.get("worker_spawn_count") == 1,
        "automatic_retry_zero": supervisor.get("automatic_retry_count") == 0,
        "no_sigterm_or_sigkill": supervisor.get("sigterm_sent") is False
        and supervisor.get("sigkill_sent") is False,
        "no_process_group_residue": supervisor.get("process_group_residue") == [],
        "no_worker_gpu_residue": supervisor.get("worker_gpu_residue") == [],
        "required_artifacts_all_present": all(
            supervisor.get("required_artifacts", {}).values()
        ),
        "raw_worker_protocol_pass": raw.get("worker_protocol_pass") is True,
        "preclose_worker_protocol_pass": preclose.get("worker_protocol_pass") is True,
        "preclose_safe_to_close": preclose.get("safe_to_close_app") is True,
        "preclose_summary_path_exact": preclose.get("summary_path") == _rel(D377_RAW),
        "preclose_summary_sha_exact": preclose.get("summary_sha256")
        == _sha(D377_RAW),
        "preclose_counter_copy_exact": preclose.get("counters")
        == raw.get("counters"),
        "preclose_timeline_copy_exact": preclose.get("timeline_after")
        == raw.get("timeline_after"),
        "d378_corrected_workload_authority_exact": _d378_authority_exact(d378),
    }
    return {"checks": checks, "pass": all(checks.values())}


def _typed_contract_audit(
    raw: dict[str, Any], inherited: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for row in raw.get("authored_readback", {}).get("rows", []):
        typed = row.get("typed_min_thickness", {})
        checks = {
            "row_pass": row.get("pass") is True,
            "typed_pass": typed.get("pass") is True,
            "type_name_exact": typed.get("type_name")
            == inherited["expected_type_name"],
            "bits_hex_exact": typed.get("bits_hex")
            == inherited["expected_bits_hex"],
            "inherited_bits_hex_exact": typed.get(
                "inherited_d343_expected_bits_hex"
            )
            == inherited["expected_bits_hex"],
            "value_exact": typed.get("value_m")
            == inherited["expected_value_m"],
        }
        rows.append(
            {
                "body": row.get("body"),
                "prim_name": row.get("prim_name"),
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    counts = Counter(row["body"] for row in rows)
    checks = {
        "d343_summary_pass": _read_json(D343_SUMMARY).get("pass") is True,
        "d343_evidence_pass": _read_json(D343_EVIDENCE).get("pass") is True,
        "inherited_contract_exact": inherited == EXPECTED_TYPED_FLOAT,
        "retest_count_zero": raw["authored_readback"].get(
            "typed_float32_contract_retests"
        )
        == 0,
        "inheritance_flag_true": raw["authored_readback"].get(
            "d343_contract_inherited"
        )
        is True,
        "rows_exact_16_18": dict(counts) == EXPECTED_COUNTS,
        "all_34_rows_exact": len(rows) == 34
        and all(row["pass"] for row in rows),
    }
    return {"rows": rows, "checks": checks, "pass": all(checks.values())}


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


def _representation_maps(
    classifier: Any,
    geometry: dict[str, Any],
    raw: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    return classifier._representation_maps(geometry, raw, rows)


def _render_board(
    classifier: Any,
    geometry: dict[str, Any],
    raw: dict[str, Any],
    rows: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    reps = _representation_maps(classifier, geometry, raw, rows)
    channels = {
        "source": "D372 Float64 source",
        "authored": "USD Float32 readback",
        "frozen_callback": "D373 frozen callback",
        "live_callback": "D377 clean-run callback",
    }
    row_by_key = {(row["body"], row["prim_name"]): row for row in rows}

    def add_mesh(
        axis: Any,
        vertices: np.ndarray,
        triangles: np.ndarray,
        role: str,
    ) -> None:
        axis.add_collection3d(
            Poly3DCollection(
                vertices[triangles] * 1000.0,
                facecolor=ROLE_COLORS_HEX[role],
                edgecolor="#202020",
                linewidth=0.14,
                alpha=0.76,
            )
        )

    def frame(axis: Any, vertices: np.ndarray, body: str) -> None:
        millimeters = vertices * 1000.0
        lower, upper = millimeters.min(axis=0), millimeters.max(axis=0)
        center = (lower + upper) * 0.5
        radius = max(float((upper - lower).max()) * 0.58, 1.0)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=18, azim=-58 if body == "link5" else -72)
        axis.set_proj_type("ortho")
        axis.set_axis_off()

    verdict_word = "PASS" if evidence["identity_pass"] else "FAIL"
    verdict_color = "#176B3A" if evidence["identity_pass"] else "#9B1C1C"
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#FBFCFE")
    for row_index, body in enumerate(("link5", "gripper_link")):
        keys = sorted(key for key in reps if key[0] == body)
        for column_index, channel in enumerate(channels):
            axis = fig.add_subplot(
                2, 4, row_index * 4 + column_index + 1, projection="3d"
            )
            all_vertices = []
            for key in keys:
                vertices, triangles = reps[key][channel]
                add_mesh(axis, vertices, triangles, row_by_key[key]["role"])
                all_vertices.append(vertices)
            frame(axis, np.vstack(all_vertices), body)
            axis.set_title(
                f"{body} ({EXPECTED_COUNTS[body]} parts)\n{channels[channel]}",
                fontproperties=bold,
                fontsize=10.0,
                pad=0.5,
            )
    fig.suptitle(
        "D379 | Offline resume of the full P34 live-identity classifier",
        fontproperties=bold,
        fontsize=20,
        y=0.976,
        color="#14213D",
    )
    fig.text(
        0.5,
        0.944,
        (
            "Immutable D377 callbacks + D378 corrected workload authority | "
            "no new Isaac, PhysX, USD, physics, q5, contact, cylinder, or IK"
        ),
        ha="center",
        va="center",
        fontproperties=regular,
        fontsize=10.5,
        color="#4B5563",
    )
    maxima = evidence["maxima"]
    fig.text(
        0.5,
        0.065,
        (
            f"Identity {verdict_word} | callback {evidence['counts']['callback_pass']}/34 | "
            f"D373 payload exact {evidence['counts']['frozen_payload_exact']}/34 | "
            f"property rows 17+19 | typed Float32 34/34"
        ),
        ha="center",
        va="center",
        fontproperties=bold,
        fontsize=11.5,
        color=verdict_color,
    )
    fig.text(
        0.5,
        0.032,
        (
            f"max surface {maxima['surface_symmetric_mm']:.6f} mm "
            f"(limit 0.100000) | max bounds {maxima['bounds_max_abs_mm']:.6f} mm "
            f"(limit 0.100000) | original JSON and callback arrays remain authority | "
            "g0a_pass=false"
        ),
        ha="center",
        va="center",
        fontproperties=regular,
        fontsize=9.7,
        color="#374151",
    )
    fig.subplots_adjust(
        left=0.006,
        right=0.994,
        top=0.910,
        bottom=0.105,
        wspace=0.005,
        hspace=0.015,
    )
    fig.savefig(BOARD_PATH, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"D379 board dimension failure: {info}")
    return info


def _build_d379_blueprint() -> Any:
    import rerun.blueprint as rrb

    def representation_row(body: str) -> Any:
        return rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/",
                contents=f"/d379/source/{body}/**",
                name=f"{body} | D372 Float64 source",
            ),
            rrb.Spatial3DView(
                origin="/",
                contents=f"/d379/authored/{body}/**",
                name=f"{body} | USD Float32 readback",
            ),
            rrb.Spatial3DView(
                origin="/",
                contents=f"/d379/frozen_callback/{body}/**",
                name=f"{body} | D373 frozen callback",
            ),
            rrb.Spatial3DView(
                origin="/",
                contents=f"/d379/live_callback/{body}/**",
                name=f"{body} | D377 clean-run callback",
            ),
            column_shares=[0.25, 0.25, 0.25, 0.25],
        )

    return rrb.Blueprint(
        rrb.Vertical(
            representation_row("link5"),
            representation_row("gripper_link"),
            rrb.Horizontal(
                rrb.DataframeView(
                    origin="/metrics/d379",
                    contents="/metrics/d379/**",
                    name="D379 inherited classifier metrics",
                ),
                rrb.Tabs(
                    rrb.TextLogView(
                        origin="/events/d379_summary",
                        contents="/events/d379_summary",
                        name="D379 verdict and frozen scope",
                    ),
                    rrb.DataframeView(
                        origin="/gate/d379",
                        contents="/gate/d379/**",
                        name="D379 gate state",
                    ),
                    active_tab=0,
                ),
                column_shares=[0.58, 0.42],
            ),
            row_shares=[0.35, 0.35, 0.30],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _write_rerun(
    classifier: Any,
    geometry: dict[str, Any],
    raw: dict[str, Any],
    rows: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    import roarm_rl.viz_debug as viz_debug

    reps = _representation_maps(classifier, geometry, raw, rows)
    row_by_key = {(row["body"], row["prim_name"]): row for row in rows}
    meshes = []
    expected_entities = {"metadata/run", "events/d379_summary"}
    component_contract: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "events/d379_summary": ["TextLog:text", "TextLog:level"],
    }
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    for key in sorted(reps):
        body, prim_name = key
        role = row_by_key[key]["role"]
        for channel, (vertices, triangles) in reps[key].items():
            entity_path = f"d379/{channel}/{body}/{prim_name}"
            metadata_path = f"metadata/meshes/{entity_path.replace('/', '__')}"
            meshes.append(
                {
                    "entity_path": entity_path,
                    "coordinate_frame": "tf#/",
                    "vertices_m": vertices,
                    "triangles": triangles,
                    "color_rgba": ROLE_COLORS_RGBA[role],
                    "static": True,
                    "body": body,
                    "prim_name": prim_name,
                    "role": role,
                    "representation": channel,
                    "display_role": "Float32 inspection copy only",
                    "source_case": "D379 offline classifier over immutable D377",
                }
            )
            expected_entities.update({entity_path, metadata_path})
            component_contract[entity_path] = mesh_components
            component_contract[metadata_path] = ["TextDocument:text"]
    scalars = [
        {"entity_path": "metrics/d379/part_count", "value": 34, "static": True},
        {
            "entity_path": "metrics/d379/callback_parts_pass",
            "value": evidence["counts"]["callback_pass"],
            "static": True,
        },
        {
            "entity_path": "metrics/d379/frozen_payload_exact",
            "value": evidence["counts"]["frozen_payload_exact"],
            "static": True,
        },
        {
            "entity_path": "metrics/d379/max_surface_delta_mm",
            "value": evidence["maxima"]["surface_symmetric_mm"],
            "static": True,
        },
        {
            "entity_path": "metrics/d379/max_bounds_delta_mm",
            "value": evidence["maxima"]["bounds_max_abs_mm"],
            "static": True,
        },
        {"entity_path": "metrics/d379/new_isaac_runs", "value": 0, "static": True},
        {"entity_path": "metrics/d379/physics_steps", "value": 0, "static": True},
        {"entity_path": "metrics/d379/q5_samples", "value": 0, "static": True},
        {
            "entity_path": "gate/d379/identity_pass",
            "value": 1 if evidence["identity_pass"] else 0,
            "static": True,
        },
        {"entity_path": "gate/d379/g0a_pass", "value": 0, "static": True},
    ]
    for scalar in scalars:
        expected_entities.add(scalar["entity_path"])
        component_contract[scalar["entity_path"]] = ["Scalars:scalars"]
    event = {
        "entity_path": "events/d379_summary",
        "text": (
            f"D379 identity {'PASS' if evidence['identity_pass'] else 'FAIL'}: "
            "offline classification of immutable D377 callbacks; link5=16, "
            "gripper_link=18. No new Isaac/PhysX/USD run. Physics, q5, contact, "
            "cylinder, target/IK/path, grasp, and g0a promotion remain out of scope."
        ),
        "level": "INFO" if evidence["identity_pass"] else "ERROR",
        "static": True,
    }

    original_builder = viz_debug.build_rerun_blueprint

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d379_p34_identity":
            return _build_d379_blueprint()
        return original_builder(mode)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    try:
        saved = viz_debug.log_rerun(
            RRD_PATH,
            meshes=meshes,
            scalar_trace=scalars,
            events=[event],
            recording_metadata={
                "case": "g0a_d379",
                "attempt": ATTEMPT,
                "verdict": evidence["verdict"],
                "evidence_sha256": _sha(EVIDENCE_PATH),
                "source_worker": "immutable D377; no new NVIDIA runtime",
                "offline_audit_invocations": 1,
                "isaac_launches": 0,
                "physx_calls": 0,
                "physics_steps": 0,
                "q5_samples": 0,
                "contact_queries": 0,
                "g0a_pass": False,
                "display_role": (
                    "inspection only; original JSON/callback arrays are authority"
                ),
            },
            recording_id="g0a_d379_p34_full_live_identity_classifier_resume",
            blueprint_path=RBL_PATH,
            blueprint_mode="d379_p34_identity",
            live_viewer=False,
            app_id="roarm_g0a_d379_p34_full_live_identity_classifier_resume",
        )
    finally:
        viz_debug.build_rerun_blueprint = original_builder
        os.environ["PATH"] = old_path
    if not saved.get("ok"):
        raise RuntimeError(f"D379 save-only Rerun failed: {saved}")

    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected_entities),
        exact_entity_paths=sorted(expected_entities),
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=component_contract,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG_PATH,
        screenshot_window_size="1920x1080",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=240.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, validation)
    headless = dict(validation.get("headless_render") or {})
    return {
        "save_only": saved,
        "strict_validation_pass": validation.get("pass") is True,
        "headless_viewer_invocations": (
            1 if headless.get("attempted") is True else 0
        ),
        "headless_viewer_returncode": headless.get("returncode"),
        "rrd": {
            "path": _rel(RRD_PATH),
            "bytes": RRD_PATH.stat().st_size,
            "sha256": _sha(RRD_PATH),
        },
        "rbl": {
            "path": _rel(RBL_PATH),
            "bytes": RBL_PATH.stat().st_size,
            "sha256": _sha(RBL_PATH),
        },
        "validation": {
            "path": _rel(RERUN_VALIDATION_PATH),
            "bytes": RERUN_VALIDATION_PATH.stat().st_size,
            "sha256": _sha(RERUN_VALIDATION_PATH),
        },
        "screenshot": (
            _png_info(RERUN_PNG_PATH)
            if RERUN_PNG_PATH.is_file()
            else {"path": _rel(RERUN_PNG_PATH), "exists": False}
        ),
    }


def prepare() -> int:
    if OUT_DIR.exists():
        raise FileExistsError(
            f"forward-only D379 attempt exists: {_rel(OUT_DIR)}"
        )
    OUT_DIR.mkdir(parents=True)
    _phase("prepare_start")

    expected_dirty = EXPECTED_D378_BASELINE_DIRTY_PATHS | {
        _rel(HARNESS),
        _rel(CASE_ROOT) + "/",
    }
    observed_dirty = set(_status_paths())
    dirty_baseline_snapshot = _registered_dirty_baseline_snapshot()
    dependency_preflight = _dependency_preflight()
    fixed_hash_checks = {
        _rel(path): path.is_file() and _sha(path) == expected
        for path, expected in EXPECTED_FIXED_HASHES.items()
    }
    inventories = {
        _rel(path): _inventory(path) for path in EXPECTED_INVENTORY_SHA
    }
    inventory_checks = {
        _rel(path): (
            inventories[_rel(path)]["inventory_sha256"]
            == EXPECTED_INVENTORY_SHA[path]
            and inventories[_rel(path)]["file_count"]
            == EXPECTED_INVENTORY_COUNTS[path]
        )
        for path in EXPECTED_INVENTORY_SHA
    }
    classifier = _load_d375_classifier()
    inherited_prereg = _read_json(D375_PREREG)
    d378 = _read_json(D378_EVIDENCE)
    attempt1_prereg = _read_json(ATTEMPT1_PREREG)
    attempt1_exception = _read_json(ATTEMPT1_EXCEPTION)
    prepare_negatives = _prepare_negative_controls(classifier)
    checks = {
        "head_equals_origin_master": _git("rev-parse", "HEAD")
        == _git("rev-parse", "origin/master"),
        "dirty_paths_exactly_registered_D378_plus_D379": observed_dirty
        == expected_dirty,
        "d379_output_root_is_forward_only_dirty_path": _rel(CASE_ROOT) + "/"
        in observed_dirty,
        "python_exact": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "rerun_sdk_exact": _package_version("rerun-sdk") == "0.34.1",
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "fonts_exist": FONT_REGULAR.is_file() and FONT_BOLD.is_file(),
        "offline_dependencies_present": dependency_preflight["pass"],
        "repo_root_sys_path_zero": sys.path[0] == str(REPO),
        "forbidden_NVIDIA_modules_absent": not _forbidden_modules_loaded(),
        "registered_dirty_baseline_all_present": dirty_baseline_snapshot[
            "all_present"
        ],
        "fixed_input_hashes_exact": all(fixed_hash_checks.values()),
        "immutable_inventories_exact": all(inventory_checks.values()),
        "d372_measurement_pass_and_verdict_exact": (
            _read_json(D372_EVIDENCE).get("measurement_pass") is True
            and _read_json(D372_EVIDENCE).get("verdict")
            == (
                "D372_PROFESSOR_SEMANTIC_COMPOUND_CANDIDATE_"
                "OFFLINE_PASS_NO_PHYSICS"
            )
        ),
        "d372_completion_pass": _read_json(D372_COMPLETION).get("pass") is True,
        "d343_summary_and_evidence_pass": _read_json(D343_SUMMARY).get("pass")
        is True
        and _read_json(D343_EVIDENCE).get("pass") is True,
        "d375_registered_identity_contract_exact": inherited_prereg.get(
            "registered_identity_contract"
        )
        == _registered_contract(),
        "d343_typed_contract_exact": inherited_prereg.get(
            "typed_float32_authority_inherited"
        )
        == EXPECTED_TYPED_FLOAT,
        "d378_corrected_authority_exact": _d378_authority_exact(d378),
        "attempt1_preregistration_failure_exact": (
            attempt1_prereg.get("pass") is False
            and attempt1_prereg.get("checks", {}).get(
                "d372_evidence_pass"
            )
            is False
            and sum(
                value is False
                for value in attempt1_prereg.get("checks", {}).values()
            )
            == 1
            and attempt1_exception.get("stage") == "prepare"
            and attempt1_exception.get("verdict") == OBSERVABILITY_FAIL
            and attempt1_exception.get("identity_pass") is None
        ),
        "prepare_negative_controls_pass": prepare_negatives["pass"],
        "start_here_attempt_path_registered": (
            ATTEMPT in START_HERE.read_text(encoding="utf-8")
        ),
    }
    prereg = {
        "artifact": "D379_PREREGISTRATION_V1",
        "case": "g0a_d379",
        "attempt": ATTEMPT,
        "what_and_why": (
            "Resume the frozen D375 full P34 classifier offline over the "
            "immutable D377 clean-run evidence after D378 repaired workload "
            "authority, without any new NVIDIA runtime or grasp science."
        ),
        "new_variables": NEW_VARIABLES,
        "forward_only_output": _rel(OUT_DIR),
        "registered_identity_contract": _registered_contract(),
        "typed_float32_authority_inherited": EXPECTED_TYPED_FLOAT,
        "thresholds": {
            "surface_tolerance_m": SURFACE_TOL_M,
            "bounds_tolerance_m": BOUNDS_TOL_M,
            "authored_callback_topology_volume_relative": (
                AUTHORED_CALLBACK_VOLUME_REL_TOL
            ),
            "callback_property_volume_relative": (
                CALLBACK_PROPERTY_VOLUME_REL_TOL
            ),
            "authored_mass_state_atol": AUTHORED_MASS_STATE_ATOL,
            "property_mass_state_atol": PROPERTY_MASS_STATE_ATOL,
            "polygon_plane_residual_tolerance_m": (
                POLYGON_PLANE_RESIDUAL_TOL_M
            ),
        },
        "authorities": {
            "d377_clean_raw_preclose_supervisor": True,
            "d378_corrected_termination_workload": True,
            "d372_float64_geometry": True,
            "d373_frozen_instance_witnesses": True,
            "d343_typed_float32_contract_retest_count": 0,
            "d375_classifier_code_and_thresholds": True,
        },
        "forward_only_attempt_history": {
            "attempt1": {
                "outcome": "preregistration_schema_field_false_negative",
                "classifier_invocations": 0,
                "rerun_viewer_invocations": 0,
                "preregistration": {
                    "path": _rel(ATTEMPT1_PREREG),
                    "sha256": _sha(ATTEMPT1_PREREG),
                },
                "exception": {
                    "path": _rel(ATTEMPT1_EXCEPTION),
                    "sha256": _sha(ATTEMPT1_EXCEPTION),
                },
            }
        },
        "run_contract": {
            "offline_audit_invocations": 1,
            "actual_offline_worker_invocations": 1,
            "automatic_retries": 0,
            "isaac_or_physx_workers": 0,
            "bounded_wall_clock_watchdog_s": OFFLINE_WORKER_TIMEOUT_S,
            "term_grace_s": OFFLINE_WORKER_TERM_GRACE_S,
            "rerun_save_only": 1,
            "headless_rerun_viewer_max": 1,
            "strict_zero_scope": STRICT_ZERO_SCOPE,
        },
        "prepare_negative_controls": prepare_negatives,
        "failure_capable_post_controls": [
            "delete one callback polygon must break closed topology",
            "perturb mass beyond inherited 1e-9 guard must be detected",
        ],
        "expected_fixed_hashes": {
            _rel(path): value for path, value in EXPECTED_FIXED_HASHES.items()
        },
        "input_hashes": _input_hashes(),
        "source_hashes": _source_hashes(),
        "registered_dirty_baseline_snapshot": dirty_baseline_snapshot,
        "inventories_before": inventories,
        "expected_inventory_checks": inventory_checks,
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "status_paths": sorted(observed_dirty),
            "expected_dirty_paths": sorted(expected_dirty),
            "inherited_dirty_scope": (
                "approved D378 completion/state evidence; preserved without "
                "modification by D379"
            ),
        },
        "environment": {
            "python": sys.executable,
            "numpy": _package_version("numpy"),
            "trimesh": _package_version("trimesh"),
            "rerun_sdk": _package_version("rerun-sdk"),
            "rerun_cli": str(RERUN_CLI),
            "dependency_preflight": dependency_preflight,
            "forbidden_modules_loaded": _forbidden_modules_loaded(),
        },
        "official_sources": OFFICIAL_SOURCES,
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
        raise RuntimeError(f"D379 preregistration failed: {checks}")
    return 0


def _audit_worker() -> int:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D379 preregistration missing")
    if not INVOCATION_PATH.is_file():
        raise RuntimeError("D379 controller invocation claim missing")
    if CLAIM_PATH.exists() or EVIDENCE_PATH.exists():
        raise FileExistsError("D379 offline audit was already attempted")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D379 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D379 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D379 immutable input hash drift")
    if (
        _registered_dirty_baseline_snapshot()
        != prereg["registered_dirty_baseline_snapshot"]
    ):
        raise RuntimeError("D379 registered dirty baseline drift")
    for relative_path, before in prereg["inventories_before"].items():
        if _inventory(REPO / relative_path) != before:
            raise RuntimeError(f"D379 immutable inventory drift: {relative_path}")
    if _forbidden_modules_loaded():
        raise RuntimeError("forbidden NVIDIA module loaded before D379 audit")

    invocation = _read_json(INVOCATION_PATH)
    _write_json_x(
        CLAIM_PATH,
        {
            "artifact": "D379_SINGLE_OFFLINE_WORKER_CLAIM_V1",
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "process_group_id": os.getpgrp(),
            "offline_audit_invocations": 1,
            "actual_offline_worker_invocations": 1,
            "automatic_retries": 0,
            "isaac_or_physx_worker_invocations": 0,
            "preregistration_sha256": _sha(PREREG_PATH),
            "invocation_sha256": _sha(INVOCATION_PATH),
            "worker_argv_exact": invocation.get("worker_argv") == sys.argv,
            "new_process_group_exact": os.getpgrp() == os.getpid(),
            "pass": invocation.get("worker_argv") == sys.argv
            and os.getpgrp() == os.getpid()
            and invocation.get("preregistration_sha256")
            == _sha(PREREG_PATH),
        },
    )
    _phase(
        "offline_worker_claimed",
        invocation_sha256=_sha(INVOCATION_PATH),
        claim_sha256=_sha(CLAIM_PATH),
    )

    classifier = _load_d375_classifier()
    raw = _read_json(D377_RAW)
    preclose = _read_json(D377_PRECLOSE)
    supervisor = _read_json(D377_SUPERVISOR)
    d378 = _read_json(D378_EVIDENCE)
    geometry = _read_json(D372_GEOMETRY)
    d375_prereg = _read_json(D375_PREREG)

    authority = _d377_authority(raw, preclose, supervisor, d378)
    typed = _typed_contract_audit(
        raw, d375_prereg["typed_float32_authority_inherited"]
    )
    property_audit = classifier._property_audit(raw)
    mass_audit = classifier.D373._mass_live_audit(raw)
    rows, callback_aggregate = classifier._analyze_callbacks(
        geometry, raw, property_audit
    )
    post_negatives = classifier._post_negative_controls(rows, mass_audit)
    all_negatives = {
        **prereg["prepare_negative_controls"]["controls"],
        **post_negatives["controls"],
    }
    _phase(
        "inherited_failure_capable_controls_complete",
        passed=sum(value is True for value in all_negatives.values()),
        total=len(all_negatives),
    )

    maxima = {
        "surface_symmetric_mm": max(
            row["surface"]["symmetric_m"] * 1000.0 for row in rows
        ),
        "bounds_max_abs_mm": max(
            row["bounds_max_abs_delta_m"] * 1000.0 for row in rows
        ),
        "authored_callback_volume_relative": max(
            row["authored_callback_volume_relative_delta"] for row in rows
        ),
        "callback_property_volume_relative": max(
            row["callback_property_volume_relative_delta"] for row in rows
        ),
        "mass_com_inertia_axes_abs_delta": max(
            value
            for body in mass_audit["bodies"].values()
            for value in body["max_abs_deltas"].values()
        ),
    }
    source_counters = raw["counters"]
    canonical_checks = raw["canonical_outside_collision_subtree_diff"]["checks"]
    canonical_required = all(
        canonical_checks.get(key) is True
        for key in (
            "outside_registered_subtrees_bit_exact",
            "no_runtime_address_patterns",
            "no_unsupported_types",
            "variant_contains_no_a64",
        )
    )
    current_scope = {
        "offline_audit_invocations": 1,
        "actual_offline_worker_invocations": 1,
        "automatic_retries": 0,
        **{key: 0 for key in STRICT_ZERO_SCOPE},
    }
    input_inventory_checks = {
        relative_path: _inventory(REPO / relative_path) == before
        for relative_path, before in prereg["inventories_before"].items()
    }
    checks = {
        "preregistration_pass": prereg["pass"] is True,
        "d377_hash_bound_clean_exit_authority": authority["pass"],
        "d378_corrected_workload_authority": _d378_authority_exact(d378),
        "worker_protocol_pass": raw["worker_protocol_pass"] is True,
        "direct_authored_readback_pass": raw["authored_readback"]["pass"] is True,
        "d343_typed_float32_inherited_without_retest": typed["pass"],
        "owner_structure_noninstance_pass": raw["owner_structure"]["pass"] is True,
        "live_inventory_path_owner_binding_pass": raw["live_inventory"]["pass"]
        is True,
        "property_binding_pass": property_audit["pass"],
        "callback_original_polygon_identity_pass": callback_aggregate["pass"],
        "mass_com_inertia_axes_invariant": mass_audit["pass"],
        "inherited_negative_controls_exact_6_pass": len(all_negatives) == 6
        and all(value is True for value in all_negatives.values()),
        "outside_collision_subtrees_preserved": canonical_required,
        "frozen_asset_was_immutable_in_source_run": raw["asset_reuse"][
            "immutable_before_after"
        ]
        is True
        and raw["asset_reuse"]["materialization_count"] == 0
        and raw["asset_reuse"]["usd_stage_file_write_count"] == 0,
        "source_worker_expected_live_counts": (
            source_counters["simulation_app_launches"] == 1
            and source_counters["worker_invocations"] == 1
            and source_counters["physx_callback_requests"] == 34
            and source_counters["physx_property_queries"] == 2
            and source_counters["physx_stage_attaches"] == 1
            and source_counters["physx_stage_detaches"] == 1
            and source_counters["stagecache_erase_calls"] == 1
        ),
        "source_worker_no_science_counters": all(
            source_counters[key] == 0 for key in SOURCE_RUNTIME_ZERO_KEYS
        ),
        "source_timeline_stopped_at_time_zero": raw["timeline_before"]
        == raw["timeline_after"]
        and raw["timeline_before"]["is_stopped"] is True
        and raw["timeline_before"]["is_playing"] is False
        and raw["timeline_before"]["current_time_s"] == 0.0,
        "current_D379_strict_zero_scope": all(
            current_scope[key] == 0 for key in STRICT_ZERO_SCOPE
        ),
        "immutable_input_hashes_still_exact": _input_hashes()
        == prereg["input_hashes"],
        "registered_dirty_baseline_still_exact": (
            _registered_dirty_baseline_snapshot()
            == prereg["registered_dirty_baseline_snapshot"]
        ),
        "immutable_input_inventories_still_exact": all(
            input_inventory_checks.values()
        ),
        "forbidden_NVIDIA_modules_absent": not _forbidden_modules_loaded(),
    }
    identity_pass = all(checks.values())
    evidence = {
        "artifact": "D379_P34_FULL_LIVE_IDENTITY_CLASSIFIER_RESUME_EVIDENCE_V1",
        "case": "g0a_d379",
        "attempt": ATTEMPT,
        "what_and_why": prereg["what_and_why"],
        "new_variables": NEW_VARIABLES,
        "verdict": VERDICT_PASS if identity_pass else VERDICT_FAIL,
        "identity_pass": identity_pass,
        "g0a_pass": False,
        "inputs": {
            "preregistration": {
                "path": _rel(PREREG_PATH),
                "sha256": _sha(PREREG_PATH),
            },
            "invocation": {
                "path": _rel(INVOCATION_PATH),
                "sha256": _sha(INVOCATION_PATH),
            },
            "worker_claim": {
                "path": _rel(CLAIM_PATH),
                "sha256": _sha(CLAIM_PATH),
            },
            "d377_raw": {"path": _rel(D377_RAW), "sha256": _sha(D377_RAW)},
            "d377_preclose": {
                "path": _rel(D377_PRECLOSE),
                "sha256": _sha(D377_PRECLOSE),
            },
            "d377_supervisor": {
                "path": _rel(D377_SUPERVISOR),
                "sha256": _sha(D377_SUPERVISOR),
            },
            "d378_corrected_authority": {
                "path": _rel(D378_EVIDENCE),
                "sha256": _sha(D378_EVIDENCE),
            },
            "d372_geometry": {
                "path": _rel(D372_GEOMETRY),
                "sha256": _sha(D372_GEOMETRY),
            },
            "d373_witness_inventory": _inventory(D373_WITNESSES),
            "d343_summary": {
                "path": _rel(D343_SUMMARY),
                "sha256": _sha(D343_SUMMARY),
            },
        },
        "registered_identity_contract": prereg["registered_identity_contract"],
        "thresholds": prereg["thresholds"],
        "d377_authority": authority,
        "d378_corrected_authority": {
            "pass": _d378_authority_exact(d378),
            "corrected_authority": d378["corrected_authority"],
            "D377_frozen_verdict_preserved": d378["interpretation"][
                "D377_formal_verdict_preserved"
            ],
        },
        "counts": {
            "direct_authored": sum(
                row["pass"] for row in raw["authored_readback"]["rows"]
            ),
            "live_inventory": sum(
                row["pass"] for row in raw["live_inventory"]["rows"]
            ),
            "callback_pass": callback_aggregate["counts"]["parts_pass"],
            "frozen_payload_exact": callback_aggregate["counts"][
                "frozen_payload_exact"
            ],
            "property_rows": {
                body: len(property_audit["bodies"][body]["p34_rows"])
                + len(property_audit["bodies"][body]["legacy_rows"])
                for body in EXPECTED_COUNTS
            },
            "typed_float32_rows": sum(row["pass"] for row in typed["rows"]),
            "source_actual_worker": source_counters["worker_invocations"],
            "current_offline_audit": 1,
            "current_actual_worker": 1,
            "current_automatic_retry": 0,
        },
        "source_worker_counters": source_counters,
        "current_scope_counters": current_scope,
        "timeline_before": raw["timeline_before"],
        "timeline_after": raw["timeline_after"],
        "owner_structure": raw["owner_structure"],
        "authored_readback": raw["authored_readback"],
        "typed_float32_audit": typed,
        "live_inventory": raw["live_inventory"],
        "property_audit": property_audit,
        "mass_audit": mass_audit,
        "callback_rows": rows,
        "callback_aggregate": callback_aggregate,
        "maxima": maxima,
        "negative_controls": {
            "prepare": prereg["prepare_negative_controls"],
            "post_callback": post_negatives,
            "all_controls": all_negatives,
            "pass_count": sum(value is True for value in all_negatives.values()),
            "expected_count": 6,
            "pass": len(all_negatives) == 6
            and all(value is True for value in all_negatives.values()),
        },
        "canonical_outside_collision_subtree_diff": raw[
            "canonical_outside_collision_subtree_diff"
        ],
        "input_inventory_checks": input_inventory_checks,
        "checks": checks,
        "pass": identity_pass,
        "remaining_nulls": {
            "A64_P34_physics_equivalence": None,
            "29x50_target_rebase": None,
            "center_height_radial_wrist_repair": None,
            "HOME_to_pregrasp_path": None,
            "q5_closure": None,
            "contact_or_tipping": None,
            "grasp_feasibility": None,
        },
        "official_sources": OFFICIAL_SOURCES,
        "next_authorization_boundary": (
            "Only after D379 completion may the 29x50 target geometry rebase be "
            "considered under a separate explicit approval. No physics or q5 "
            "science is authorized by D379."
        ),
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase(
        "offline_identity_evidence_committed",
        evidence_sha256=_sha(EVIDENCE_PATH),
        identity_pass=identity_pass,
    )

    board = _render_board(classifier, geometry, raw, rows, evidence)
    _phase("exact_1920x1080_board_complete", board_sha256=board["sha256"])
    rerun = _write_rerun(classifier, geometry, raw, rows, evidence)
    _phase(
        "save_only_rerun_and_strict_validation_complete",
        strict_validation_pass=rerun["strict_validation_pass"],
    )
    after_forbidden = _forbidden_modules_loaded()
    automated_checks = {
        "identity_evidence_generated": EVIDENCE_PATH.is_file(),
        "board_exact_1920x1080": board["exact_1920x1080"],
        "rerun_save_only_ok": rerun["save_only"].get("ok") is True,
        "rerun_strict_validation_pass": rerun["strict_validation_pass"],
        "headless_viewer_exactly_one": rerun["headless_viewer_invocations"] == 1,
        "headless_viewer_return_zero": rerun["headless_viewer_returncode"] == 0,
        "rerun_screenshot_exists": RERUN_PNG_PATH.is_file()
        and RERUN_PNG_PATH.stat().st_size > 0,
        "forbidden_NVIDIA_modules_absent_after_visualization": not after_forbidden,
        "immutable_input_hashes_after_visualization": _input_hashes()
        == prereg["input_hashes"],
        "registered_dirty_baseline_after_visualization": (
            _registered_dirty_baseline_snapshot()
            == prereg["registered_dirty_baseline_snapshot"]
        ),
        "immutable_input_inventories_after_visualization": all(
            _inventory(REPO / relative_path) == before
            for relative_path, before in prereg["inventories_before"].items()
        ),
        "current_scope_zero_preserved": all(
            current_scope[key] == 0 for key in STRICT_ZERO_SCOPE
        ),
        "remaining_science_nulls_preserved": all(
            value is None for value in evidence["remaining_nulls"].values()
        ),
        "g0a_false": evidence["g0a_pass"] is False,
    }
    automated = {
        "artifact": "D379_AUTOMATED_SUMMARY_V1",
        "invocation": {
            "path": _rel(INVOCATION_PATH),
            "sha256": _sha(INVOCATION_PATH),
        },
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "board": board,
        "rerun": rerun,
        "checks": automated_checks,
        "observability_pass": all(automated_checks.values()),
        "identity_pass": identity_pass,
        "audit_integrity_pass_pending_manual": all(
            automated_checks.values()
        ),
        "manual_visual_inspection": "pending",
        "g0a_pass": False,
    }
    _write_json_x(AUTOMATED_PATH, automated)
    _write_json_x(
        MANUAL_TEMPLATE_PATH,
        {
            "artifact": (
                "D379_MANUAL_ORIGINAL_RESOLUTION_VISUAL_INSPECTION_TEMPLATE_V1"
            ),
            "expected_sha256": {
                "board": board["sha256"],
                "rerun_inspection": rerun["screenshot"].get("sha256"),
            },
            "required_checks": {
                "board_exact_size_and_four_columns_legible": False,
                "source_authored_d373_d377_channels_distinguishable": False,
                "metrics_and_thresholds_legible": False,
                "verdict_matches_evidence": False,
                "no_text_overlap_or_clipping": False,
                "scope_boundary_and_g0a_false_visible": False,
                "rerun_required_rows_visible": False,
            },
            "pass": False,
        },
    )
    _phase(
        "audit_complete_awaiting_manual_inspection",
        automated_summary_sha256=_sha(AUTOMATED_PATH),
        identity_pass=identity_pass,
        observability_pass=automated["observability_pass"],
    )
    return 0 if automated["audit_integrity_pass_pending_manual"] else 1


def audit() -> int:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D379 preregistration missing")
    for path in (
        INVOCATION_PATH,
        CLAIM_PATH,
        SUPERVISOR_PATH,
        STDOUT_PATH,
        STDERR_PATH,
        EVIDENCE_PATH,
    ):
        if path.exists():
            raise FileExistsError(
                f"D379 one-shot offline worker path already claimed: {_rel(path)}"
            )
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D379 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D379 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D379 immutable input hash drift")
    if (
        _registered_dirty_baseline_snapshot()
        != prereg["registered_dirty_baseline_snapshot"]
    ):
        raise RuntimeError("D379 registered dirty baseline drift")

    worker_argv = [str(HARNESS), "--stage", "audit-worker"]
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        *worker_argv,
    ]
    invocation = {
        "artifact": "D379_SINGLE_OFFLINE_WORKER_INVOCATION_V1",
        "command": command,
        "worker_argv": worker_argv,
        "cwd": str(REPO),
        "controller_pid": os.getpid(),
        "controller_sha256": _sha(HARNESS),
        "preregistration_sha256": _sha(PREREG_PATH),
        "actual_offline_worker_invocations": 1,
        "automatic_retries": 0,
        "isaac_or_physx_worker_invocations": 0,
        "bounded_wall_clock_watchdog_s": OFFLINE_WORKER_TIMEOUT_S,
        "term_grace_s": OFFLINE_WORKER_TERM_GRACE_S,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase(
        "offline_worker_spawn_requested",
        invocation_sha256=_sha(INVOCATION_PATH),
        watchdog_s=OFFLINE_WORKER_TIMEOUT_S,
    )

    started = time.monotonic()
    timed_out = False
    sigterm_sent = False
    sigkill_sent = False
    with STDOUT_PATH.open("x", encoding="utf-8") as stdout_stream, STDERR_PATH.open(
        "x", encoding="utf-8"
    ) as stderr_stream:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            start_new_session=True,
        )
        try:
            process.wait(timeout=OFFLINE_WORKER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
                sigterm_sent = True
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=OFFLINE_WORKER_TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    sigkill_sent = True
                except ProcessLookupError:
                    pass
                process.wait()
    elapsed_s = time.monotonic() - started

    process_group_alive_after_wait = False
    try:
        os.killpg(process.pid, 0)
        process_group_alive_after_wait = True
    except ProcessLookupError:
        pass
    except PermissionError:
        process_group_alive_after_wait = True

    required_artifacts = {
        "claim": CLAIM_PATH.is_file(),
        "evidence": EVIDENCE_PATH.is_file(),
        "board": BOARD_PATH.is_file(),
        "rrd": RRD_PATH.is_file(),
        "rbl": RBL_PATH.is_file(),
        "rerun_validation": RERUN_VALIDATION_PATH.is_file(),
        "rerun_screenshot": RERUN_PNG_PATH.is_file(),
        "automated_summary": AUTOMATED_PATH.is_file(),
        "manual_template": MANUAL_TEMPLATE_PATH.is_file(),
    }
    claim = _read_json(CLAIM_PATH) if CLAIM_PATH.is_file() else {}
    automated = _read_json(AUTOMATED_PATH) if AUTOMATED_PATH.is_file() else {}
    source_hashes_still_exact = _source_hashes() == prereg["source_hashes"]
    input_hashes_still_exact = _input_hashes() == prereg["input_hashes"]
    registered_dirty_baseline_still_exact = (
        _registered_dirty_baseline_snapshot()
        == prereg["registered_dirty_baseline_snapshot"]
    )
    operational_pass = bool(
        process.returncode == 0
        and not timed_out
        and not process_group_alive_after_wait
        and all(required_artifacts.values())
        and claim.get("pass") is True
        and automated.get("observability_pass") is True
        and source_hashes_still_exact
        and input_hashes_still_exact
        and registered_dirty_baseline_still_exact
    )
    supervisor = {
        "artifact": "D379_OFFLINE_WORKER_SUPERVISOR_V1",
        "invocation": {
            "path": _rel(INVOCATION_PATH),
            "sha256": _sha(INVOCATION_PATH),
        },
        "claim": (
            {"path": _rel(CLAIM_PATH), "sha256": _sha(CLAIM_PATH)}
            if CLAIM_PATH.is_file()
            else {"path": _rel(CLAIM_PATH), "exists": False}
        ),
        "stdout": {
            "path": _rel(STDOUT_PATH),
            "bytes": STDOUT_PATH.stat().st_size,
            "sha256": _sha(STDOUT_PATH),
        },
        "stderr": {
            "path": _rel(STDERR_PATH),
            "bytes": STDERR_PATH.stat().st_size,
            "sha256": _sha(STDERR_PATH),
        },
        "worker_pid": process.pid,
        "returncode": process.returncode,
        "elapsed_s": elapsed_s,
        "timed_out": timed_out,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "process_group_alive_after_wait": process_group_alive_after_wait,
        "actual_offline_worker_invocations": 1,
        "automatic_retries": 0,
        "isaac_or_physx_worker_invocations": 0,
        "required_artifacts": required_artifacts,
        "worker_identity_pass": automated.get("identity_pass"),
        "worker_observability_pass": automated.get("observability_pass"),
        "source_hashes_still_exact": source_hashes_still_exact,
        "input_hashes_still_exact": input_hashes_still_exact,
        "registered_dirty_baseline_still_exact": (
            registered_dirty_baseline_still_exact
        ),
        "operational_pass": operational_pass,
        "pass": operational_pass,
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase(
        "offline_worker_supervised",
        supervisor_sha256=_sha(SUPERVISOR_PATH),
        returncode=process.returncode,
        timed_out=timed_out,
        operational_pass=operational_pass,
    )
    return 0 if operational_pass else 1


def finalize() -> int:
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        CLAIM_PATH,
        SUPERVISOR_PATH,
        STDOUT_PATH,
        STDERR_PATH,
        EVIDENCE_PATH,
        BOARD_PATH,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION_PATH,
        RERUN_PNG_PATH,
        AUTOMATED_PATH,
        MANUAL_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"D379 finalize prerequisite missing: {_rel(path)}")
    if COMPLETION_PATH.exists():
        raise FileExistsError("D379 completion already exists")
    _phase("finalize_start")

    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    claim = _read_json(CLAIM_PATH)
    supervisor = _read_json(SUPERVISOR_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_PATH)
    required_manual_checks = {
        "board_exact_size_and_four_columns_legible",
        "source_authored_d373_d377_channels_distinguishable",
        "metrics_and_thresholds_legible",
        "verdict_matches_evidence",
        "no_text_overlap_or_clipping",
        "scope_boundary_and_g0a_false_visible",
        "rerun_required_rows_visible",
    }
    expected_manual_hashes = {
        "board": automated["board"]["sha256"],
        "rerun_inspection": automated["rerun"]["screenshot"]["sha256"],
    }
    current_board = _png_info(BOARD_PATH)
    current_rerun_png = _png_info(RERUN_PNG_PATH)
    checks = {
        "preregistration_pass": prereg["pass"] is True,
        "preregistration_hash_bound_to_invocation": _sha(PREREG_PATH)
        == invocation["preregistration_sha256"],
        "single_worker_claim_pass": claim.get("pass") is True
        and claim.get("invocation_sha256") == _sha(INVOCATION_PATH),
        "bounded_supervisor_pass": supervisor.get("pass") is True
        and supervisor.get("returncode") == 0
        and supervisor.get("timed_out") is False
        and supervisor.get("sigterm_sent") is False
        and supervisor.get("sigkill_sent") is False
        and supervisor.get("process_group_alive_after_wait") is False
        and supervisor.get("actual_offline_worker_invocations") == 1
        and supervisor.get("automatic_retries") == 0,
        "invocation_hash_unchanged": _sha(INVOCATION_PATH)
        == automated["invocation"]["sha256"],
        "identity_result_is_boolean": isinstance(
            evidence.get("identity_pass"), bool
        ),
        "identity_verdict_matches_result": evidence.get("verdict")
        == (
            VERDICT_PASS
            if evidence.get("identity_pass") is True
            else VERDICT_FAIL
        ),
        "identity_evidence_hash_unchanged": _sha(EVIDENCE_PATH)
        == automated["evidence"]["sha256"],
        "automated_observability_pass": automated["observability_pass"] is True,
        "automated_identity_matches_evidence": automated["identity_pass"]
        is evidence["identity_pass"],
        "manual_artifact_exact": manual.get("artifact")
        == "D379_MANUAL_ORIGINAL_RESOLUTION_VISUAL_INSPECTION_V1",
        "manual_pass": manual.get("pass") is True,
        "manual_hashes_exact": manual.get("inspected_sha256")
        == expected_manual_hashes,
        "manual_check_keys_exact_and_all_true": set(
            manual.get("checks", {})
        )
        == required_manual_checks
        and all(
            manual["checks"].get(key) is True for key in required_manual_checks
        ),
        "board_current_hash_size_dimensions_exact": current_board
        == automated["board"],
        "rrd_current_hash_size_exact": RRD_PATH.stat().st_size
        == automated["rerun"]["rrd"]["bytes"]
        and _sha(RRD_PATH) == automated["rerun"]["rrd"]["sha256"],
        "rbl_current_hash_size_exact": RBL_PATH.stat().st_size
        == automated["rerun"]["rbl"]["bytes"]
        and _sha(RBL_PATH) == automated["rerun"]["rbl"]["sha256"],
        "rerun_validation_current_hash_size_exact": (
            RERUN_VALIDATION_PATH.stat().st_size
            == automated["rerun"]["validation"]["bytes"]
            and _sha(RERUN_VALIDATION_PATH)
            == automated["rerun"]["validation"]["sha256"]
        ),
        "rerun_png_current_hash_size_dimensions_exact": current_rerun_png
        == automated["rerun"]["screenshot"],
        "source_hashes_still_exact": _source_hashes()
        == prereg["source_hashes"],
        "input_hashes_still_exact": _input_hashes() == prereg["input_hashes"],
        "registered_dirty_baseline_still_exact": (
            _registered_dirty_baseline_snapshot()
            == prereg["registered_dirty_baseline_snapshot"]
        ),
        "input_inventories_still_exact": all(
            _inventory(REPO / relative_path) == before
            for relative_path, before in prereg["inventories_before"].items()
        ),
        "strict_zero_scope_preserved": all(
            evidence["current_scope_counters"][key] == 0
            for key in STRICT_ZERO_SCOPE
        ),
        "remaining_science_nulls_preserved": all(
            value is None for value in evidence["remaining_nulls"].values()
        ),
        "g0a_false": evidence["g0a_pass"] is False
        and automated["g0a_pass"] is False,
        "forbidden_NVIDIA_modules_absent": not _forbidden_modules_loaded(),
    }
    completion_integrity_pass = all(checks.values())
    case_pass = completion_integrity_pass and evidence["identity_pass"] is True
    verdict = (
        OBSERVABILITY_FAIL
        if not completion_integrity_pass
        else evidence["verdict"]
    )
    completion = {
        "artifact": "D379_COMPLETION_SUMMARY_V1",
        "case": "g0a_d379",
        "attempt": ATTEMPT,
        "new_variables": NEW_VARIABLES,
        "preregistration": {
            "path": _rel(PREREG_PATH),
            "sha256": _sha(PREREG_PATH),
        },
        "invocation": {
            "path": _rel(INVOCATION_PATH),
            "sha256": _sha(INVOCATION_PATH),
        },
        "worker_claim": {
            "path": _rel(CLAIM_PATH),
            "sha256": _sha(CLAIM_PATH),
        },
        "worker_supervisor": {
            "path": _rel(SUPERVISOR_PATH),
            "sha256": _sha(SUPERVISOR_PATH),
        },
        "worker_stdout": {
            "path": _rel(STDOUT_PATH),
            "sha256": _sha(STDOUT_PATH),
        },
        "worker_stderr": {
            "path": _rel(STDERR_PATH),
            "sha256": _sha(STDERR_PATH),
        },
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "automated_summary": {
            "path": _rel(AUTOMATED_PATH),
            "sha256": _sha(AUTOMATED_PATH),
        },
        "manual_inspection": {
            "path": _rel(MANUAL_PATH),
            "sha256": _sha(MANUAL_PATH),
        },
        "board": automated["board"],
        "rrd": automated["rerun"]["rrd"],
        "rbl": automated["rerun"]["rbl"],
        "rerun_validation": automated["rerun"]["validation"],
        "rerun_inspection": automated["rerun"]["screenshot"],
        "identity_pass": evidence["identity_pass"],
        "completion_integrity_pass": completion_integrity_pass,
        "g0a_pass": False,
        "counts": evidence["counts"],
        "maxima": evidence["maxima"],
        "current_scope_counters": evidence["current_scope_counters"],
        "remaining_nulls": evidence["remaining_nulls"],
        "checks": checks,
        "pass": case_pass,
        "verdict": verdict,
        "next_authorization_boundary": evidence["next_authorization_boundary"],
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase(
        "finalize_complete",
        completion_sha256=_sha(COMPLETION_PATH),
        verdict=completion["verdict"],
    )
    return 0 if completion_integrity_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "audit", "audit-worker", "finalize"),
    )
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            return prepare()
        if args.stage == "audit":
            return audit()
        if args.stage == "audit-worker":
            return _audit_worker()
        return finalize()
    except Exception as exc:
        frozen_identity = None
        if EVIDENCE_PATH.is_file():
            try:
                frozen_identity = _read_json(EVIDENCE_PATH).get("identity_pass")
            except Exception:
                frozen_identity = None
        payload = {
            "artifact": "D379_RUNTIME_EXCEPTION_V1",
            "stage": args.stage,
            "exception_type": type(exc).__name__,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "verdict": OBSERVABILITY_FAIL,
            "identity_pass": None,
            "frozen_identity_result_if_available": frozen_identity,
            "g0a_pass": False,
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
