#!/usr/bin/env python3
"""D351 zero-step moving-jaw closure-geometry discriminator.

The frozen D350 q0-q4/object state is retained while q5 is written directly
from OPEN to CLOSED.  Raw authored surfaces are queried first, followed by the
D348 callback-topology surface proxy.  No physics step, target/IK change, asset
write, cook, property query, settle, trial, or promotion is licensed.
"""
from __future__ import annotations

import argparse
import colorsys
import copy
import csv
import hashlib
import io
import json
import math
import os
import secrets
import struct
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rerun as rr
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from roarm_rl.rerun_contract import (  # noqa: E402
    RERUN_CONTRACT_VERSION,
    sha256_file,
    validate_rerun_artifact,
)
from roarm_rl.viz_debug import log_rerun  # noqa: E402
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d332_grasp_g0a_static_collision_discriminator as d332,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d333_grasp_g0a_sole_support_static_retest as d333,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d334_grasp_g0a_live_collision_shape_ownership_audit as d334,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d339_grasp_g0a_cook_witness_contract_repair as d339,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d349_grasp_g0a_frozen_open_jaw_target_live_distance_gate as d349,
)
from sim_scripts import (  # noqa: E402
    cyl34_top_view_d350_fixed_jaw_geometry_viewer as d350,
)


CASE = "g0a_d351"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d351"
PREREG_PATH = OUT_DIR / "d351_preregistration.json"
PARAMETER_PATH = OUT_DIR / "d351_parameter_freeze_audit.json"
PREFLIGHT_PATH = OUT_DIR / "d351_validate_preflight.json"
LIVE_BINDING_PATH = OUT_DIR / "d351_live_topology_runtime_binding.json"
MOVING_BINDING_PATH = OUT_DIR / "d351_moving_jaw_surface_binding.json"
MEASUREMENT_PATH = OUT_DIR / "d351_zero_step_closure_geometry_measurement.json"
SWEEP_CSV_PATH = OUT_DIR / "d351_q5_closure_sweep.csv"
OVERLAY_PATH = OUT_DIR / "d351_viewer_overlay_contract.json"
CAPTURE_PATH = OUT_DIR / "d351_viewer_capture_contract.json"
RRD_PATH = OUT_DIR / "d351_zero_step_closure_geometry.rrd"
RBL_PATH = OUT_DIR / "d351_zero_step_closure_geometry.rbl"
RERUN_PNG_PATH = OUT_DIR / "d351_zero_step_closure_geometry_rerun.png"
RERUN_VALIDATION_PATH = OUT_DIR / "d351_rerun_validation.json"
AUTOMATED_PATH = OUT_DIR / "d351_automated_summary.json"
AUTOMATED_MD_PATH = OUT_DIR / "d351_automated_report.md"
RUNTIME_EXCEPTION_PATH = OUT_DIR / "d351_runtime_exception.json"
MANUAL_PATH = OUT_DIR / "d351_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d351_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d351_completion_summary.json"
COMPLETION_MD_PATH = OUT_DIR / "d351_completion_report.md"

VIEWER_PNGS = {
    "open_physx": OUT_DIR / "d351_open_actual_physx_colliders.png",
    "decision_physx": OUT_DIR / "d351_decision_or_open_fallback_actual_physx_colliders.png",
    "decision_colored": OUT_DIR / "d351_decision_or_open_fallback_colored_64plus64.png",
    "decision_side": OUT_DIR / "d351_decision_or_open_fallback_side_geometry.png",
}

SESSION_DOC = REPO / "claudedocs/session_20260715_grasp_g0a_d351_zero_step_closure_geometry.md"
START_HERE = REPO / "START_HERE.md"
HARNESS = Path(__file__).resolve()
URDF_PATH = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
AUTHORING_ROBOT_USD = REPO / "local_assets/roarm_m3/usd/roarm_m3.usd"

D334_SUMMARY = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d334/g0a_d334_live_collision_audit_summary.json"
D348_EVIDENCE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_callback_topology_volume_evidence.json"
D349_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json"
D350_BINDING = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_semantic_binding.json"
D350_MEASUREMENT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350/d350_fixed_jaw_geometry_measurement.json"
D350_COMPLETION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d350/attempt2_observability_repair/d350_completion_summary.json"
VARIANT_ROBOT_USD = d349.VARIANT_ROBOT_USD
VARIANT_PHYSICS_USD = d349.VARIANT_PHYSICS_USD

EXPECTED_HEAD = "cfd9e7501df89724c3cc2b1038fda05ce0d88e2f"
REGISTERED_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
NEW_VARIABLES = [
    "moving_jaw_actual_contact_surface_binding",
    "frozen_pose_q5_closure_sweep",
]
NEW_PHYSICAL_VARIABLES: list[str] = []
SEED = 33201
Q_FROZEN_F32 = np.asarray(
    [
        0.03750238195061684,
        0.542945146560669,
        1.9687392711639404,
        0.18299327790737152,
        0.0,
        1.5413000583648682,
    ],
    dtype=np.float32,
)
OBJECT_POS_F32 = np.asarray(
    [0.30000001192092896, 0.0, 0.03288299962878227], dtype=np.float32
)
OBJECT_QUAT_F32 = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
Q5_OPEN_F32 = np.float32(1.5413000583648682)
Q5_CLOSED_F32 = np.float32(0.0)
GRID_COUNT = 33
CONTACT_Q5_WIDTH_RAD = 1.0e-6
MAX_RECURSION_DEPTH = 32
# The following are numerical-resolution/integrity controls, not new grasp,
# alignment, or pose-success tolerances.  The only representation-fidelity
# gate remains the inherited D349/D350 0.5 mm gate below.
RUNTIME_KINEMATIC_TOL_M = 1.0e-6
BINDING_RESIDUAL_MAX_M = 1.0e-5
CLEAR_GATE_MM = 0.1
FIDELITY_TOL_MM = 0.5
ANCHOR_TOL_MM = 1.0e-6
CLOSED_ANCHOR_TOL_MM = 0.05
TABLE_CLEAR_GATE_MM = 0.0
VIEWER_RASTER_SIZE = [1280, 720]
RERUN_RASTER_SIZE = [4800, 2800]
MOVING_INNER_PATCH_FACE_IDS = np.arange(672, 1165, dtype=np.int64)
MOVING_OUTER_PATCH_FACE_IDS = np.arange(13205, 13698, dtype=np.int64)
MOVING_INNER_PATCH_FACE_ID_SHA256 = (
    "6d97da67c58f38152e97da74e544cdd179fb5352cdc06ec223b33353f187b0d6"
)
MOVING_OUTER_PATCH_FACE_ID_SHA256 = (
    "35b3eda85e97375771e540d5e7b1131b5acf3df0952ee6e394059c70b79d1a60"
)
MOVING_INNER_PATCH_Y_M = -0.005983415603637695
MOVING_OUTER_PATCH_Y_M = -0.0044834156036376955
EXPECTED_LIVE_INNER_FACE_COUNT = 40
EXPECTED_LIVE_INNER_PART_COUNT = 17
EXPECTED_LIVE_GRIPPER_TRIANGLE_COUNT = 832
EXPECTED_LIVE_INNER_FACE_KEY_SHA256 = (
    "5bb7ad8a21826cb0709da55f85b0e3772114a782e1263483c180963aa9eccab5"
)
EXPECTED_AUTHORED_STREAM_HASHES = {
    "points_f32_mm": "b89c67e99bd253ae710e6b0a2fcacd0b27263d6ede29fe6f6334ed70247895ed",
    "face_counts_i64": "f17eac58b9b109f98f7a69efcc3b1e64b632d805ccca8cc8883cf0349e07cb6c",
    "face_indices_i64": "205a08458b895d96c6eb9593d1f04a8815629f7f972a889cce683b86955f2545",
}
EXPECTED_AUTHORED_PATCH_HASHES = {
    "inner_vertex": "13c65ee478a2668896ec2a8f1e237a9ba7b7e6e0ef40ab08cb350087d3a74d55",
    "outer_vertex": "0d9f7f856eb66d5f749303aa7f4bac8138a595d228dff8424221a6b0b732772a",
    "triangle": "5644e9a66386d68945d340a46cfa9e1507b6dd55cf0b721823ef6afb079b9e17",
    "inner_patch": "c927e8c628073f9f1d8fc0250d8190a71bb2b0701b97b41d7f8069b216c3531b",
    "outer_patch": "9b430c7d7e8c389eb648726014d61169aa671ec910f94a782084b467e96d6486",
    "paired_xz": "917b7154601d23984db01ebbd4adccdb272771920f225d1a021540b4b03bcaf9",
}
GUIDE_ROOT = "/World/D351ViewerGuides"
PHYSX_COLLIDER_SETTING = "/persistent/physics/visualizationDisplayColliders"
GUIDE_PURPOSE_SETTING = "/persistent/app/hydra/displayPurpose/guide"

VERDICT_INPUT = "D351_FROZEN_INPUT_OR_ZERO_STEP_CONTRACT_FAIL_STOP"
VERDICT_ORDER = "D351_CONTACT_ORDER_UNRESOLVED_FAIL_STOP"
VERDICT_BINDING = "D351_MOVING_JAW_SURFACE_BINDING_FAIL_STOP"
VERDICT_ELIGIBLE = "D351_CURRENT_PREGRASP_BARREL_CLOSURE_ELIGIBLE"
VERDICT_REPAIR = "D351_CURRENT_POSE_CLOSURE_GEOMETRY_REPAIR_RECOMMENDED"
VERDICT_VISUAL = "D351_VIEWER_OR_RERUN_CONTRACT_FAIL_STOP"
VERDICT_PENDING_SUFFIX = "_MANUAL_PENDING"

EXPECTED_INPUT_HASHES = {
    "d334_summary": "2ff44744df99c7a99d168cdd62a4f9186a5bbad6d673205282abb62b71097b26",
    "d348_evidence": "83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6",
    "d349_measurement": "5de6d14e37d6b74b202d1bb668120a6bb57221eac24ea5c751457ce9823b6300",
    "d350_binding": "1ec1c309461357eeae89204fa55a498b64d2d216708ab6e6c7dfdd3d0b878c12",
    "d350_measurement": "4fe91e4cd37f5b0f064c7e9c91480881973ca51e651132af2c8feb57750e8446",
    "d350_completion": "7866886a49ecfca1c16bd1283c89e920613a4c25581dadf5ebaa195e1303cedb",
    "variant_robot_usd": "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    "authored_robot_usd": "a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff",
    "variant_physics_usd": "043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503",
    "urdf": "64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2",
    "d332": "3ab551232b9c3e2a3886578e5f4baa4589d578567758a351203c2260a1428ad4",
    "d333": "e582f274fca44093b0e1367555459f22428c809792b6cfc3a9a336369dac68b7",
    "d334": "19d2f333c2aeec89282d230324b965e6f5af7e6d05648a858c5637fd24adf735",
    "d339": "fd307cb573699f8a08df1ab580789188774158877b8abf0a05cc4c60ef6562d6",
    "d349": "33a9743337fa269b71e4da3ccccfabc1d746ee29e1582a3d0f8c4764f42d68b9",
    "d350": "99a9b558754c9c4ebf83b265e4bcc70744e1981786066d1343c96cd046d4c538",
    "viz_debug": "622b7197afe8cdeb1bb5411f2a961aa9a9a5c58aaf248417fc145639374577c5",
    "rerun_contract": "90559c931bc753be97def463841d41426a2f1bd8e5ddd15a2a2ab08fb54a2e60",
}
PREEXISTING_USER_UNTRACKED_HASHES = {
    "claudedocs/lab_meeting/20260715/d334_collision_table/README.md": (
        "0b60b1216166f2a5f5728eb2c0148369805d80a1c6a17f80ebd5b93830fc25ef"
    ),
    "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.html": (
        "ab7130edf838f997f4d1e5e8bbedefc1cad56a7088d51acc2e91a44fb50a9f18"
    ),
    "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.png": (
        "1ac92fe89fcee76e0a420ec7a6f3a4ec2dfcaf2084ad201544baca39d24a1d47"
    ),
}
PREEXISTING_USER_GIT_ROLES = {
    "claudedocs/lab_meeting/20260715/d334_collision_table/README.md": "untracked",
    "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.html": "untracked",
    "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.png": "ignored",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> dict[str, str]:
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    fields = raw.decode("utf-8", errors="surrogateescape").split("\0")
    result: dict[str, str] = {}
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        status = field[:2]
        path = field[3:]
        if status[0] in "RC" and index < len(fields):
            path = fields[index]
            index += 1
        result[path] = status
    return result


def _allowed_status() -> set[str]:
    return {
        _rel(START_HERE),
        _rel(SESSION_DOC),
        _rel(HARNESS),
        *PREEXISTING_USER_UNTRACKED_HASHES,
    }


def _status_scope_pass(status: dict[str, str]) -> bool:
    prefix = _rel(OUT_DIR) + "/"
    return all(path in _allowed_status() or path.startswith(prefix) for path in status)


def _preexisting_user_untracked_contract() -> dict[str, Any]:
    status = _git_status()
    rows = {}
    for relative, expected_hash in PREEXISTING_USER_UNTRACKED_HASHES.items():
        role = PREEXISTING_USER_GIT_ROLES[relative]
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative],
            cwd=REPO,
            check=False,
        ).returncode == 0
        rows[relative] = {
            "git_role": role,
            "status": status.get(relative),
            "git_ignored": ignored,
            "git_role_exact": bool(
                (role == "untracked" and status.get(relative) == "??" and not ignored)
                or (role == "ignored" and status.get(relative) is None and ignored)
            ),
            "exists": (REPO / relative).is_file(),
            "sha256": _sha(REPO / relative) if (REPO / relative).is_file() else None,
            "expected_sha256": expected_hash,
        }
    checks = {
        "exact_git_roles": all(row["git_role_exact"] for row in rows.values()),
        "exact_hashes_unchanged": all(
            row["exists"] and row["sha256"] == row["expected_sha256"]
            for row in rows.values()
        ),
    }
    return {
        "role": "preexisting user untracked/ignored files; read-only and not scientific inputs",
        "rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _input_paths() -> dict[str, Path]:
    return {
        "d334_summary": D334_SUMMARY,
        "d348_evidence": D348_EVIDENCE,
        "d349_measurement": D349_MEASUREMENT,
        "d350_binding": D350_BINDING,
        "d350_measurement": D350_MEASUREMENT,
        "d350_completion": D350_COMPLETION,
        "variant_robot_usd": VARIANT_ROBOT_USD,
        "authored_robot_usd": AUTHORING_ROBOT_USD,
        "variant_physics_usd": VARIANT_PHYSICS_USD,
        "urdf": URDF_PATH,
        "d332": Path(d332.__file__).resolve(),
        "d333": Path(d333.__file__).resolve(),
        "d334": Path(d334.__file__).resolve(),
        "d339": Path(d339.__file__).resolve(),
        "d349": Path(d349.__file__).resolve(),
        "d350": Path(d350.__file__).resolve(),
        "viz_debug": REPO / "roarm_rl/viz_debug.py",
        "rerun_contract": REPO / "roarm_rl/rerun_contract.py",
    }


def _input_hashes() -> dict[str, str]:
    return {name: _sha(path) for name, path in _input_paths().items()}


def _png_dimensions(path: Path) -> str | None:
    if not path.is_file():
        return None
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return f"{width}x{height}"


def _decode_png(path: Path, *, expected_size: list[int]) -> dict[str, Any]:
    verify_ok = False
    load_ok = False
    mode = None
    size = None
    error = None
    try:
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        verify_ok = True
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            mode = image.mode
            size = list(image.size)
        load_ok = True
    except Exception as exc:  # recorded as a hard artifact failure
        error = f"{type(exc).__name__}: {exc}"
    return {
        "verify_ok": verify_ok,
        "load_ok": load_ok,
        "mode": mode,
        "size": size,
        "expected_mode": "RGBA",
        "expected_size": expected_size,
        "error": error,
        "pass": bool(
            verify_ok and load_ok and mode == "RGBA" and size == expected_size
        ),
    }


def _sample_png(path: Path) -> dict[str, Any]:
    return {
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
        "sha256": _sha(path) if path.is_file() else None,
    }


def _stable_png(path: Path, *, expected_size: list[int]) -> dict[str, Any]:
    samples = [_sample_png(path)]
    time.sleep(1.0)
    samples.append(_sample_png(path))
    time.sleep(1.0)
    samples.append(_sample_png(path))
    decode = _decode_png(path, expected_size=expected_size)
    post_decode = _sample_png(path)
    stable = samples[0] == samples[1] == samples[2] == post_decode
    return {
        "path": _rel(path),
        "samples": samples,
        "post_decode_sample": post_decode,
        "decode": decode,
        "dimensions": _png_dimensions(path),
        "stable": stable,
        "pass": bool(
            stable
            and decode["pass"]
            and samples[-1]["bytes"] is not None
            and int(samples[-1]["bytes"]) > 0
        ),
    }


def _environment_contract() -> dict[str, Any]:
    import numpy

    checks = {
        "python_exact": str(Path(sys.executable).resolve())
        == str(Path(REGISTERED_PYTHON).resolve()),
        "numpy_1_26_0": numpy.__version__ == "1.26.0",
        "psutil_5_9_8": psutil.__version__ == "5.9.8",
        "rerun_0_34_1": rr.__version__ == RERUN_CONTRACT_VERSION == "0.34.1",
    }
    return {
        "python": str(Path(sys.executable).resolve()),
        "numpy": numpy.__version__,
        "psutil": psutil.__version__,
        "rerun": rr.__version__,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _scope_guards() -> dict[str, Any]:
    return {
        "asset_write": False,
        "decomposition_change": False,
        "fresh_cook_callback_or_property_query": False,
        "q0_q4_change": False,
        "object_runtime_pose_change": False,
        "target_ik_or_path_change": False,
        "tolerance_change": False,
        "material_mass_actuator_physics_change": False,
        "controlled_physics_steps": 0,
        "settle": False,
        "ten_trial": False,
        "g0b": False,
        "rl_or_ppo": False,
        "ladder_promotion": False,
        "g0a_pass": False,
    }


def _runtime_outputs() -> list[Path]:
    return [
        PREFLIGHT_PATH,
        LIVE_BINDING_PATH,
        MOVING_BINDING_PATH,
        MEASUREMENT_PATH,
        SWEEP_CSV_PATH,
        OVERLAY_PATH,
        CAPTURE_PATH,
        RRD_PATH,
        RBL_PATH,
        RERUN_PNG_PATH,
        RERUN_VALIDATION_PATH,
        AUTOMATED_PATH,
        AUTOMATED_MD_PATH,
        RUNTIME_EXCEPTION_PATH,
        MANUAL_PATH,
        MANUAL_MD_PATH,
        COMPLETION_PATH,
        COMPLETION_MD_PATH,
        *VIEWER_PNGS.values(),
    ]


def _parameter_audit() -> dict[str, Any]:
    d350_measurement = _json(D350_MEASUREMENT)
    checks = {
        "new_variables_exact_two": NEW_VARIABLES
        == [
            "moving_jaw_actual_contact_surface_binding",
            "frozen_pose_q5_closure_sweep",
        ],
        "new_physical_variables_zero": NEW_PHYSICAL_VARIABLES == [],
        "q5_open_exact": Q5_OPEN_F32.tobytes() == Q_FROZEN_F32[5].tobytes(),
        "q5_closed_zero": Q5_CLOSED_F32.tobytes() == np.float32(0.0).tobytes(),
        "grid_count_33": GRID_COUNT == 33,
        "contact_width_1e_6": CONTACT_Q5_WIDTH_RAD == 1.0e-6,
        "clear_gate_inherited": CLEAR_GATE_MM == 0.1,
        "fidelity_gate_inherited": FIDELITY_TOL_MM == 0.5,
        "table_gate_is_strict_zero_not_new_margin": TABLE_CLEAR_GATE_MM == 0.0,
        "moving_inner_face_ids_exact": hashlib.sha256(
            MOVING_INNER_PATCH_FACE_IDS.astype("<i8", copy=False).tobytes()
        ).hexdigest()
        == MOVING_INNER_PATCH_FACE_ID_SHA256,
        "moving_outer_face_ids_exact": hashlib.sha256(
            MOVING_OUTER_PATCH_FACE_IDS.astype("<i8", copy=False).tobytes()
        ).hexdigest()
        == MOVING_OUTER_PATCH_FACE_ID_SHA256,
        "live_inner_face_count_frozen_40": EXPECTED_LIVE_INNER_FACE_COUNT == 40,
        "live_inner_part_count_frozen_17": EXPECTED_LIVE_INNER_PART_COUNT == 17,
        "live_gripper_triangle_count_frozen_832": (
            EXPECTED_LIVE_GRIPPER_TRIANGLE_COUNT == 832
        ),
        "live_inner_face_key_hash_frozen": (
            EXPECTED_LIVE_INNER_FACE_KEY_SHA256
            == "5bb7ad8a21826cb0709da55f85b0e3772114a782e1263483c180963aa9eccab5"
        ),
        "d350_fixed_digest_exact": d350_measurement["fixed_jaw_component"][
            "component_digest"
        ]
        == "8f64ddb03308521ce905d0714def9b72e1e69871d2f9f13ea3bd2a3f07559a4d",
        "scope_all_zero_false": not any(
            bool(value) for key, value in _scope_guards().items() if key != "controlled_physics_steps"
        ),
        "controlled_steps_zero": _scope_guards()["controlled_physics_steps"] == 0,
    }
    return {
        "artifact": "D351_PARAMETER_FREEZE_AUDIT_V1",
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "sampling": {
            "anchor_grid": "float32 linspace(1.5413000583648682,0,33)",
            "interval_bound": "2*Rmax*sin(abs(delta_q)/2)",
            "contact_bracket_width_rad": CONTACT_Q5_WIDTH_RAD,
            "max_recursion_depth": MAX_RECURSION_DEPTH,
        },
        "surface_identity": {
            "inner_face_range_inclusive": [672, 1164],
            "outer_negative_control_face_range_inclusive": [13205, 13697],
            "classification": "frozen authored face identity plus exact closing-motion sign",
            "overlap_authority": (
                "all non-saturated hpp-fcl Contact.b1 triangle ids; live contacts "
                "projected to raw under inherited 0.5mm"
            ),
            "competitor_exclusion": (
                "raw and D348 callback-topology live non-inner complements plus "
                "their inner surfaces versus analytic zero-height cylinder cap "
                "disks must exceed each representation's bracket motion bound"
            ),
            "negative_endpoint_policy": (
                "non-inner/non-barrel endpoint evidence is FAIL_STOP unless a future "
                "separate case preregisters a symmetric first-feature certificate"
            ),
            "new_distance_or_angle_success_tolerance": None,
        },
        "numerical_controls_not_scientific_success_tolerances": {
            "contact_bracket_width_rad": CONTACT_Q5_WIDTH_RAD,
            "max_recursion_depth": MAX_RECURSION_DEPTH,
            "runtime_axis_pivot_integrity_m": RUNTIME_KINEMATIC_TOL_M,
            "d350_inherited_binding_residual_m": BINDING_RESIDUAL_MAX_M,
            "d350_inherited_face_tie_m": d350.BINDING_FACE_TIE_M,
            "closed_d337_anchor_diagnostic_only_mm": CLOSED_ANCHOR_TOL_MM,
        },
        "scope_guards": _scope_guards(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_prepare(_args: argparse.Namespace) -> int:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise RuntimeError(f"forward-only D351 output already nonempty: {OUT_DIR}")
    if _git_head() != EXPECTED_HEAD:
        raise RuntimeError(f"D351 base HEAD drift: {_git_head()} != {EXPECTED_HEAD}")
    status = _git_status()
    inputs = _input_hashes()
    environment = _environment_contract()
    parameter = _parameter_audit()
    preexisting_user_files = _preexisting_user_untracked_contract()
    prechecks = {
        "git_scope_only_d351": _status_scope_pass(status),
        "preexisting_user_files_read_only_exact": preexisting_user_files["pass"],
        "input_hashes_exact": inputs == EXPECTED_INPUT_HASHES,
        "environment": environment["pass"],
        "parameter_freeze": parameter["pass"],
        "session_doc_exists": SESSION_DOC.is_file(),
        "start_here_active_d351": "zero_step_moving_jaw_closure_geometry_discriminator"
        in START_HERE.read_text(encoding="utf-8"),
    }
    if not all(prechecks.values()):
        raise RuntimeError(f"D351 prepare STOP: {prechecks}")
    _write_json(PARAMETER_PATH, parameter)
    prereg = {
        "artifact": "D351_PREREGISTRATION_V1",
        "case": CASE,
        "git_head": _git_head(),
        "git_status": status,
        "prepare_process_identity": {"pid": os.getpid(), "nonce": _args.process_nonce},
        "harness_sha256": _sha(HARNESS),
        "state_hashes": {
            "start_here": _sha(START_HERE),
            "session_doc": _sha(SESSION_DOC),
        },
        "parameter_audit_sha256": _sha(PARAMETER_PATH),
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "question": "frozen-pose actual moving-jaw first-contact feature and certified pre-contact q5 corridor",
        "live_semantics": "D348 callback-topology triangle-surface proxy, not direct PhysX narrowphase",
        "positive_semantics": "pre-grasp barrel-closure eligibility only; not grasp/force-closure/G0a",
        "input_hashes": inputs,
        "source_inventories": d349._source_inventories(),
        "environment": environment,
        "parameter_freeze": parameter,
        "preexisting_user_files": preexisting_user_files,
        "prechecks": prechecks,
        "pass": all(prechecks.values()),
    }
    _write_json(PREREG_PATH, prereg)
    print(json.dumps({"stage": "prepare", "pass": prereg["pass"]}, sort_keys=True))
    return 0


def _validate_preflight(args: argparse.Namespace) -> bool:
    import torch

    prereg = _json(PREREG_PATH)
    status = _git_status()
    preexisting_user_files = _preexisting_user_untracked_contract()
    checks = {
        "prereg_pass": prereg.get("pass") is True,
        "fresh_process_pid": prereg.get("prepare_process_identity", {}).get("pid")
        != os.getpid(),
        "fresh_process_nonce": prereg.get("prepare_process_identity", {}).get("nonce")
        != args.process_nonce,
        "git_head_exact": _git_head() == EXPECTED_HEAD == prereg.get("git_head"),
        "git_scope_only_d351": _status_scope_pass(status),
        "preexisting_user_files_read_only_exact": bool(
            preexisting_user_files["pass"]
            and preexisting_user_files == prereg.get("preexisting_user_files")
        ),
        "input_hashes_exact": _input_hashes()
        == EXPECTED_INPUT_HASHES
        == prereg.get("input_hashes"),
        "environment": _environment_contract()["pass"],
        "parameter_freeze": _parameter_audit()["pass"],
        "parameter_hash_exact": _sha(PARAMETER_PATH)
        == prereg.get("parameter_audit_sha256"),
        "harness_hash_exact": _sha(HARNESS) == prereg.get("harness_sha256"),
        "state_hashes_exact": {
            "start_here": _sha(START_HERE),
            "session_doc": _sha(SESSION_DOC),
        }
        == prereg.get("state_hashes"),
        "headless_false": args.headless is False,
        "livestream_zero": int(args.livestream) == 0,
        "viewer_hold_positive": float(args.viewer_hold_seconds) > 0.0,
        "display_present": bool(os.environ.get("DISPLAY")),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_4090": bool(
            torch.cuda.is_available()
            and "4090" in torch.cuda.get_device_name(0)
        ),
        "app_args_exact": all(d350._app_arg_checks(args).values()),
        "runtime_outputs_absent": all(not path.exists() for path in _runtime_outputs()),
    }
    report = {
        "artifact": "D351_VALIDATE_PREFLIGHT_V1",
        "validate_process_identity": {"pid": os.getpid(), "nonce": args.process_nonce},
        "preregistration_sha256": _sha(PREREG_PATH),
        "parameter_audit_sha256": _sha(PARAMETER_PATH),
        "harness_sha256": _sha(HARNESS),
        "environment": {
            "display": os.environ.get("DISPLAY"),
            "cuda": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "app_arg_checks": d350._app_arg_checks(args),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json(PREFLIGHT_PATH, report)
    return report["pass"]


def _quat_to_rot(quat_wxyz: Any) -> np.ndarray:
    return d332._quat_wxyz_to_rot(np.asarray(quat_wxyz, dtype=np.float64))


def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _joint_spec() -> dict[str, Any]:
    root = ET.parse(URDF_PATH).getroot()
    joint = next(
        item
        for item in root.findall("joint")
        if item.attrib.get("name") == "link5_to_gripper_link"
    )
    origin = joint.find("origin")
    axis = joint.find("axis")
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    axis_local = np.fromstring(axis.attrib.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
    axis_local /= float(np.linalg.norm(axis_local))
    return {
        "joint": joint.attrib.get("name"),
        "type": joint.attrib.get("type"),
        "parent": joint.find("parent").attrib["link"],
        "child": joint.find("child").attrib["link"],
        "origin_xyz_parent_m": xyz,
        "origin_rpy_parent_rad": rpy,
        "axis_joint": axis_local,
    }


def _runtime_joint_geometry(inner: Any) -> dict[str, Any]:
    spec = _joint_spec()
    link_pos, link_quat = d334._body_pose_w(inner, "link5")
    grip_pos, grip_quat = d334._body_pose_w(inner, "gripper_link")
    link_rot = _quat_to_rot(link_quat)
    grip_rot = _quat_to_rot(grip_quat)
    joint_rot_world = link_rot @ _rpy_to_rot(spec["origin_rpy_parent_rad"])
    origin_world = link_pos + link_rot @ spec["origin_xyz_parent_m"]
    axis_world = joint_rot_world @ spec["axis_joint"]
    axis_world /= float(np.linalg.norm(axis_world))
    child_axis_world = grip_rot[:, 2]
    checks = {
        "joint_revolute": spec["type"] == "revolute",
        "parent_link5": spec["parent"] == "link5",
        "child_gripper_link": spec["child"] == "gripper_link",
        "origin_matches_runtime_child_le_1um": float(
            np.linalg.norm(origin_world - grip_pos)
        )
        <= RUNTIME_KINEMATIC_TOL_M,
        "axis_matches_runtime_child_z_le_1um": float(
            np.linalg.norm(axis_world - child_axis_world)
        )
        <= RUNTIME_KINEMATIC_TOL_M,
    }
    return {
        "joint": spec["joint"],
        "origin_world_m": origin_world.tolist(),
        "axis_world": axis_world.tolist(),
        "gripper_body_position_world_m": grip_pos.tolist(),
        "gripper_body_quaternion_wxyz": grip_quat.tolist(),
        "origin_runtime_residual_m": float(np.linalg.norm(origin_world - grip_pos)),
        "axis_runtime_residual": float(np.linalg.norm(axis_world - child_axis_world)),
        "numerical_integrity_tolerance_m": RUNTIME_KINEMATIC_TOL_M,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _q5_key(value: float) -> str:
    return np.float32(value).tobytes().hex()


def _raw_parts(raw_shapes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {body: [] for body in d334.BODY_LABELS}
    for shape in raw_shapes:
        result[shape["body"]].append(
            {
                "body": shape["body"],
                "name": "retained_raw_full_mesh",
                "path": shape["collider_path"],
                "_geometry_raw": shape["_geom_raw"],
            }
        )
    return result


def _by_body(distance_set: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["body"]: row for row in distance_set["queries"]}


def _compact_query(row: dict[str, Any]) -> dict[str, Any]:
    selected = None
    for part in row.get("parts", []):
        if part.get("path") == row.get("witness_part_path"):
            selected = part
            break
    return {
        "body": row["body"],
        "representation": row["representation"],
        "part_count": int(row["part_count"]),
        "is_collision": bool(row["is_collision"]),
        "exact_signed_distance_mm": row["exact_signed_distance_mm"],
        "exact_consistent": bool(row["exact_consistent"]),
        "epa_cap_saturated_any": bool(row["epa_cap_saturated_any"]),
        "overlap_state": row["overlap_state"],
        "witness_kind": row["witness_kind"],
        "witness_endpoint_geometry_m": row["witness_endpoint_0_m"],
        "witness_endpoint_cylinder_m": row["witness_endpoint_1_m"],
        "witness_part_path": row["witness_part_path"],
        "selected_part": None
        if selected is None
        else {
            "path": selected["path"],
            "exact_signed_distance_mm": selected["exact_signed_distance_mm"],
            "exact_consistent": selected["exact_consistent"],
            "is_collision": selected["is_collision"],
            "witness_kind": selected["witness_kind"],
            "epa_contact_count": selected["epa_contact_count"],
            "epa_cap_saturated": selected["epa_cap_saturated"],
            "epa_selected_contact": selected["epa_selected_contact"],
        },
    }


def _simulation_clock(inner: Any) -> dict[str, Any]:
    sim_time = getattr(inner.sim, "current_time", None)
    step_index = getattr(inner.sim, "current_time_step_index", None)
    return {
        "current_time": None if sim_time is None else float(sim_time),
        "current_time_step_index": None if step_index is None else int(step_index),
    }


def _state_guard(
    inner: Any,
    q5: float,
    counter_before: int,
    time_before: float,
    timeline: Any,
    simulation_clock_before: dict[str, Any],
) -> dict[str, Any]:
    actual = inner._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    expected = Q_FROZEN_F32.copy()
    expected[5] = np.float32(q5)
    obj_pos, obj_quat = d334._object_pose_w(inner)
    simulation_clock_after = _simulation_clock(inner)
    checks = {
        "joint_float32_exact": np.array_equal(actual, expected),
        "q0_q4_float32_exact": np.array_equal(actual[:5], Q_FROZEN_F32[:5]),
        "q5_float32_exact": actual[5].tobytes() == np.float32(q5).tobytes(),
        "object_position_float32_exact": np.array_equal(
            obj_pos.astype(np.float32), OBJECT_POS_F32
        ),
        "object_quaternion_float32_exact": np.array_equal(
            obj_quat.astype(np.float32), OBJECT_QUAT_F32
        ),
        "counter_zero_unchanged": counter_before == int(inner._sim_step_counter) == 0,
        "timeline_paused": not timeline.is_playing(),
        "timeline_time_unchanged": float(timeline.get_current_time()) == time_before,
        "simulation_context_clock_available": all(
            value is not None for value in simulation_clock_after.values()
        ),
        "simulation_context_clock_unchanged": simulation_clock_after
        == simulation_clock_before,
    }
    return {
        "expected_joint_rad_float32": expected.tolist(),
        "actual_joint_rad_float32": actual.tolist(),
        "object_position_m": obj_pos.tolist(),
        "object_quaternion_wxyz": obj_quat.tolist(),
        "simulation_context_clock_before": simulation_clock_before,
        "simulation_context_clock_after": simulation_clock_after,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _feature_from_cylinder_witness(point: Any, center: Any) -> dict[str, Any]:
    p = np.asarray(point, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    local = p - c
    radial = float(np.linalg.norm(local[:2]))
    abs_z = abs(float(local[2]))
    side_residual = abs(radial - d332.CYLINDER_RADIUS_M)
    cap_residual = abs(abs_z - 0.5 * d332.CYLINDER_HEIGHT_M)
    bottom = float(c[2] - 0.5 * d332.CYLINDER_HEIGHT_M)
    top = float(c[2] + 0.5 * d332.CYLINDER_HEIGHT_M)
    if not np.isfinite(p).all():
        feature = "unresolved_nonfinite"
    elif bottom < float(p[2]) < top:
        feature = "barrel_interior"
    else:
        feature = "cap_or_rim_boundary"
    return {
        "feature": feature,
        "point_world_m": p.tolist(),
        "point_cylinder_local_m": local.tolist(),
        "radial_m": radial,
        "abs_height_m": abs_z,
        "barrel_surface_residual_mm": side_residual * 1000.0,
        "cap_surface_residual_mm": cap_residual * 1000.0,
        "strict_bottom_world_z_m": bottom,
        "strict_top_world_z_m": top,
        "strictly_between_cap_planes": bottom < float(p[2]) < top,
        "classification_rule": "strict z order only; no new geometric success tolerance",
        "radial_and_cap_residuals_verdict_authority": False,
    }


def _raw_table_and_adjacent_diagnostics(
    inner: Any,
    raw_shapes: list[dict[str, Any]],
    topology_parts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    import hppfcl

    by_body = {row["body"]: row for row in raw_shapes}
    gripper = by_body["gripper_link"]
    link5 = by_body["link5"]
    gripper_pos, gripper_quat = d334._body_pose_w(inner, "gripper_link")
    link5_pos, link5_quat = d334._body_pose_w(inner, "link5")
    gripper_rot = _quat_to_rot(gripper_quat)
    world_vertices = (
        gripper_rot @ np.asarray(gripper["_raw_verts"], dtype=np.float64).T
    ).T + gripper_pos
    raw_min_z = float(np.min(world_vertices[:, 2]))
    live_world_vertices = np.vstack(
        [
            _world_vertices(part["_vertices"], gripper_pos, gripper_rot)
            for part in topology_parts["gripper_link"]
        ]
    )
    live_min_z = float(np.min(live_world_vertices[:, 2]))
    raw_table_clearance_mm = (raw_min_z - d332.TABLE_Z_M) * 1000.0
    live_table_clearance_mm = (live_min_z - d332.TABLE_Z_M) * 1000.0
    conservative_table_clearance_mm = min(
        raw_table_clearance_mm, live_table_clearance_mm
    )
    gripper_tf = hppfcl.Transform3f(gripper_rot, gripper_pos)
    link5_tf = hppfcl.Transform3f(_quat_to_rot(link5_quat), link5_pos)
    adjacent = d332._fcl_query(
        hppfcl,
        gripper["_geom_raw"],
        gripper_tf,
        link5["_geom_raw"],
        link5_tf,
    )
    return {
        "gripper_raw_min_world_z_m": raw_min_z,
        "gripper_live_proxy_min_world_z_m": live_min_z,
        "table_top_z_m": d332.TABLE_Z_M,
        "raw_gripper_table_clearance_mm": raw_table_clearance_mm,
        "live_gripper_table_clearance_mm": live_table_clearance_mm,
        "gripper_table_clearance_mm": conservative_table_clearance_mm,
        "table_clearance_semantics": (
            "minimum of retained raw mesh and 64-part callback-topology live proxy"
        ),
        "raw_table_strictly_clear": raw_table_clearance_mm > TABLE_CLEAR_GATE_MM,
        "live_table_strictly_clear": live_table_clearance_mm > TABLE_CLEAR_GATE_MM,
        "table_strictly_clear": conservative_table_clearance_mm
        > TABLE_CLEAR_GATE_MM,
        "adjacent_link5_gripper_diagnostic_only": {
            "signed_distance_mm": float(adjacent["signed_distance_mm"]),
            "is_collision": bool(adjacent["is_collision"]),
            "nearest_point_gripper_m": adjacent["nearest_point_geometry_m"],
            "nearest_point_link5_m": adjacent["nearest_point_cylinder_m"],
            "verdict_authority": False,
            "reason": "adjacent parent-child authored assembly; self-collision is disabled",
        },
    }


def _evaluate_q5(
    inner: Any,
    timeline: Any,
    q5: float,
    raw_parts: dict[str, list[dict[str, Any]]],
    topology_parts: dict[str, list[dict[str, Any]]],
    raw_shapes: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    evaluation_order: list[str],
) -> dict[str, Any]:
    q5_f32 = float(np.float32(q5))
    key = _q5_key(q5_f32)
    if key in cache:
        return cache[key]
    if timeline.is_playing():
        timeline.pause()
    counter_before = int(inner._sim_step_counter)
    time_before = float(timeline.get_current_time())
    simulation_clock_before = _simulation_clock(inner)
    q = Q_FROZEN_F32.copy()
    q[5] = np.float32(q5_f32)
    d332._write_exact_state(inner, q.astype(np.float64), OBJECT_POS_F32.astype(np.float64))
    guard = _state_guard(
        inner,
        q5_f32,
        counter_before,
        time_before,
        timeline,
        simulation_clock_before,
    )
    raw_set = d349._union_distances(
        inner,
        raw_parts,
        "_geometry_raw",
        "d351_retained_raw_full_mesh",
    )
    live_set = d349._union_distances(
        inner,
        topology_parts,
        "_geometry_topology_surface_authority",
        "d351_callback_topology_surface_proxy",
    )
    raw = {body: _compact_query(row) for body, row in _by_body(raw_set).items()}
    live = {body: _compact_query(row) for body, row in _by_body(live_set).items()}
    diagnostics = _raw_table_and_adjacent_diagnostics(
        inner, raw_shapes, topology_parts
    )
    joint_geometry = _runtime_joint_geometry(inner)
    obj_pos, _ = d334._object_pose_w(inner)
    for representation in (raw, live):
        representation["gripper_link"]["cylinder_feature"] = _feature_from_cylinder_witness(
            representation["gripper_link"]["witness_endpoint_cylinder_m"], obj_pos
        )
    raw_value = raw["gripper_link"]["exact_signed_distance_mm"]
    live_value = live["gripper_link"]["exact_signed_distance_mm"]
    finite_pair = bool(
        raw_value is not None
        and live_value is not None
        and math.isfinite(float(raw_value))
        and math.isfinite(float(live_value))
    )
    row = {
        "evaluation_index": len(evaluation_order),
        "q5_float32_rad": q5_f32,
        "q5_float32_bits_hex": key,
        "query_order": ["raw_authored_full_mesh", "live_callback_topology_surface_proxy"],
        "state_guard": guard,
        "raw": raw,
        "live": live,
        "raw_live_gripper_absolute_delta_mm": None
        if not finite_pair
        else abs(float(raw_value) - float(live_value)),
        "diagnostics": diagnostics,
        "runtime_joint_geometry": joint_geometry,
        "simulation_counter": int(inner._sim_step_counter),
    }
    cache[key] = row
    evaluation_order.append(key)
    return row


def _distance(row: dict[str, Any], representation: str) -> float | None:
    value = row[representation]["gripper_link"]["exact_signed_distance_mm"]
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _endpoint_contract(
    row: dict[str, Any], representation: str
) -> dict[str, Any]:
    query = row[representation]["gripper_link"]
    distance = _distance(row, representation)
    selected = query.get("selected_part")
    witnesses = np.asarray(
        [
            query.get("witness_endpoint_geometry_m"),
            query.get("witness_endpoint_cylinder_m"),
        ],
        dtype=np.float64,
    )
    common = bool(
        distance is not None
        and query.get("exact_consistent") is True
        and query.get("epa_cap_saturated_any") is False
        and witnesses.shape == (2, 3)
        and np.isfinite(witnesses).all()
        and selected is not None
        and selected.get("exact_consistent") is True
        and selected.get("epa_cap_saturated") is False
    )
    clear = bool(
        common
        and distance is not None
        and distance > 0.0
        and query.get("is_collision") is False
        and query.get("witness_kind") == "clear_separation"
        and selected.get("is_collision") is False
        and selected.get("witness_kind") == "clear_separation"
    )
    epa = None if selected is None else selected.get("epa_selected_contact")
    overlap = bool(
        common
        and distance is not None
        and distance <= 0.0
        and query.get("is_collision") is True
        and query.get("overlap_state") == "overlap"
        and query.get("witness_kind") == "epa_penetration"
        and selected.get("is_collision") is True
        and selected.get("witness_kind") == "epa_penetration"
        and int(selected.get("epa_contact_count", 0)) > 0
        and isinstance(epa, dict)
        and epa.get("finite") is True
    )
    return {
        "distance_mm": distance,
        "common_exact_contract": common,
        "valid_clear": clear,
        "valid_overlap": overlap,
        "valid_for_signed_state": clear or overlap,
    }


def _interval_bound_mm(radius_m: float, q_hi: float, q_lo: float) -> float:
    return 2.0 * radius_m * math.sin(abs(q_hi - q_lo) / 2.0) * 1000.0


def _certify_first_contact(
    representation: str,
    q_grid: list[float],
    radius_m: float,
    evaluate: Any,
) -> dict[str, Any]:
    certified: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    bracket: dict[str, Any] | None = None

    def traverse(q_hi: float, q_lo: float, depth: int) -> bool:
        nonlocal bracket
        hi = evaluate(q_hi)
        lo = evaluate(q_lo)
        d_hi = _distance(hi, representation)
        d_lo = _distance(lo, representation)
        hi_contract = _endpoint_contract(hi, representation)
        lo_contract = _endpoint_contract(lo, representation)
        if (
            d_hi is None
            or d_lo is None
            or not hi_contract["valid_for_signed_state"]
            or not lo_contract["valid_for_signed_state"]
        ):
            unresolved.append(
                {
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                    "depth": depth,
                    "hi_endpoint_contract": hi_contract,
                    "lo_endpoint_contract": lo_contract,
                    "reason": "invalid_exact_or_epa_endpoint_contract",
                }
            )
            return True
        width = q_hi - q_lo
        bound = _interval_bound_mm(radius_m, q_hi, q_lo)
        if (
            hi_contract["valid_clear"]
            and lo_contract["valid_clear"]
            and min(d_hi, d_lo) > bound
        ):
            certified.append(
                {
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                    "distance_hi_mm": d_hi,
                    "distance_lo_mm": d_lo,
                    "hausdorff_bound_mm": bound,
                    "clear_margin_mm": min(d_hi, d_lo) - bound,
                    "depth": depth,
                }
            )
            return False
        if width <= CONTACT_Q5_WIDTH_RAD or depth >= MAX_RECURSION_DEPTH:
            if hi_contract["valid_clear"] and lo_contract["valid_overlap"]:
                bracket = {
                    "q_clear_float32_rad": q_hi,
                    "q_overlap_float32_rad": q_lo,
                    "q_width_rad": width,
                    "distance_clear_mm": d_hi,
                    "distance_overlap_mm": d_lo,
                    "depth": depth,
                    "numeric_width_target_rad": CONTACT_Q5_WIDTH_RAD,
                    "clear_endpoint_contract": hi_contract,
                    "overlap_endpoint_contract": lo_contract,
                }
            else:
                unresolved.append(
                    {
                        "q_hi": q_hi,
                        "q_lo": q_lo,
                        "distance_hi_mm": d_hi,
                        "distance_lo_mm": d_lo,
                        "hausdorff_bound_mm": bound,
                        "depth": depth,
                        "reason": "terminal_interval_not_certified_or_bracketed",
                    }
                )
            return True
        mid = float(np.float32((q_hi + q_lo) * 0.5))
        if not (q_lo < mid < q_hi):
            unresolved.append(
                {
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                    "distance_hi_mm": d_hi,
                    "distance_lo_mm": d_lo,
                    "depth": depth,
                    "reason": "float32_midpoint_stagnation",
                }
            )
            return True
        if traverse(q_hi, mid, depth + 1):
            return True
        return traverse(mid, q_lo, depth + 1)

    open_row = evaluate(q_grid[0])
    open_distance = _distance(open_row, representation)
    open_contract = _endpoint_contract(open_row, representation)
    if open_distance is None or not open_contract["valid_clear"]:
        unresolved.append(
            {
                "q": q_grid[0],
                "distance_mm": open_distance,
                "endpoint_contract": open_contract,
                "reason": "open_endpoint_not_valid_exact_clear",
            }
        )
    else:
        for q_hi, q_lo in zip(q_grid[:-1], q_grid[1:], strict=True):
            if traverse(q_hi, q_lo, 0):
                break
    return {
        "representation": representation,
        "rotation_radius_m": radius_m,
        "bound_formula": "2*Rmax*sin(abs(delta_q)/2)",
        "certified_clear_intervals": certified,
        "first_contact_bracket": bracket,
        "unresolved_intervals": unresolved,
        "contact_order_certified": bool(bracket is not None and not unresolved),
    }


def _certify_precontact_table_clearance(
    q_grid: list[float],
    q_stop: float,
    radius_m: float,
    evaluate: Any,
) -> dict[str, Any]:
    certified: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    def clearance(row: dict[str, Any]) -> float | None:
        value = row["diagnostics"].get("gripper_table_clearance_mm")
        if value is None or not math.isfinite(float(value)):
            return None
        return float(value)

    def traverse(q_hi: float, q_lo: float, depth: int) -> None:
        hi = evaluate(q_hi)
        lo = evaluate(q_lo)
        c_hi = clearance(hi)
        c_lo = clearance(lo)
        if c_hi is None or c_lo is None:
            unresolved.append(
                {"q_hi": q_hi, "q_lo": q_lo, "depth": depth, "reason": "nonfinite"}
            )
            return
        bound = _interval_bound_mm(radius_m, q_hi, q_lo)
        margin = min(c_hi, c_lo) - bound
        if margin > TABLE_CLEAR_GATE_MM:
            certified.append(
                {
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                    "clearance_hi_mm": c_hi,
                    "clearance_lo_mm": c_lo,
                    "hausdorff_bound_mm": bound,
                    "strict_clear_margin_mm": margin,
                    "depth": depth,
                }
            )
            return
        width = q_hi - q_lo
        if (
            c_hi <= TABLE_CLEAR_GATE_MM
            or c_lo <= TABLE_CLEAR_GATE_MM
            or width <= CONTACT_Q5_WIDTH_RAD
            or depth >= MAX_RECURSION_DEPTH
        ):
            unresolved.append(
                {
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                    "clearance_hi_mm": c_hi,
                    "clearance_lo_mm": c_lo,
                    "hausdorff_bound_mm": bound,
                    "depth": depth,
                    "reason": "table_overlap_or_interval_not_strictly_certified",
                }
            )
            return
        mid = float(np.float32((q_hi + q_lo) * 0.5))
        if not (q_lo < mid < q_hi):
            unresolved.append(
                {
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                    "depth": depth,
                    "reason": "float32_midpoint_stagnation",
                }
            )
            return
        traverse(q_hi, mid, depth + 1)
        traverse(mid, q_lo, depth + 1)

    anchors = [float(q) for q in q_grid if float(q) > float(q_stop)]
    anchors.append(float(np.float32(q_stop)))
    anchors = sorted({_q5_key(q): float(np.float32(q)) for q in anchors}.values(), reverse=True)
    for q_hi, q_lo in zip(anchors[:-1], anchors[1:], strict=True):
        traverse(q_hi, q_lo, 0)
    endpoint_rows = [evaluate(anchors[0]), evaluate(anchors[-1])] if anchors else []
    endpoint_strict = bool(
        endpoint_rows
        and all(
            clearance(row) is not None
            and float(clearance(row)) > TABLE_CLEAR_GATE_MM
            for row in endpoint_rows
        )
    )
    return {
        "q_open_float32_rad": anchors[0] if anchors else None,
        "q_stop_float32_rad": anchors[-1] if anchors else None,
        "rotation_radius_m": radius_m,
        "bound_formula": "2*Rmax*sin(abs(delta_q)/2)",
        "scientific_clearance_threshold_mm": TABLE_CLEAR_GATE_MM,
        "threshold_semantics": "strictly positive, not a new margin tolerance",
        "certified_intervals": certified,
        "unresolved_intervals": unresolved,
        "endpoint_strictly_clear": endpoint_strict,
        "precontact_table_clearance_certified": bool(endpoint_strict and not unresolved),
    }


def _bind_component_edge(
    raw_vertices: np.ndarray,
    raw_triangles: np.ndarray,
    seed_local: np.ndarray,
    *,
    reverse_order: bool,
) -> dict[str, Any]:
    vertices = np.asarray(raw_vertices, dtype=np.float64)
    triangles = np.asarray(raw_triangles, dtype=np.int64)
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    welded = inverse[triangles]
    face_count = len(triangles)
    parent = np.arange(face_count, dtype=np.int64)
    rank = np.zeros(face_count, dtype=np.int8)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    edge_owner: dict[tuple[int, int], int] = {}
    order = range(face_count - 1, -1, -1) if reverse_order else range(face_count)
    for face_idx in order:
        ids = [int(value) for value in welded[face_idx]]
        for left, right in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            edge = tuple(sorted((left, right)))
            if edge in edge_owner:
                union(face_idx, edge_owner[edge])
            else:
                edge_owner[edge] = face_idx

    distances = np.empty(face_count, dtype=np.float64)
    closest = np.empty((face_count, 3), dtype=np.float64)
    for face_idx, ids in enumerate(triangles):
        a, b, c = vertices[ids]
        point = d350._closest_point_triangle(seed_local, a, b, c)
        closest[face_idx] = point
        distances[face_idx] = float(np.linalg.norm(seed_local - point))
    minimum = float(np.min(distances))
    tied = np.flatnonzero(distances <= minimum + d350.BINDING_FACE_TIE_M)
    roots = sorted({find(int(face_idx)) for face_idx in tied})
    selected_root = roots[0] if len(roots) == 1 else -1
    component_faces = np.asarray(
        sorted(face_idx for face_idx in range(face_count) if find(face_idx) == selected_root),
        dtype=np.int64,
    )
    used = (
        np.unique(welded[component_faces].reshape(-1))
        if len(component_faces)
        else np.asarray([], dtype=np.int64)
    )
    remap = {int(old): new for new, old in enumerate(used.tolist())}
    component_vertices = (
        unique[used] if len(used) else np.zeros((0, 3), dtype=np.float64)
    )
    component_triangles = (
        np.asarray(
            [[remap[int(value)] for value in welded[idx]] for idx in component_faces],
            dtype=np.int64,
        )
        if len(component_faces)
        else np.zeros((0, 3), dtype=np.int64)
    )
    seed_face = int(tied[0]) if len(tied) else -1
    return {
        "minimum_distance_m": minimum,
        "tied_face_indices": tied.tolist(),
        "tied_component_roots": roots,
        "seed_face_index": seed_face,
        "seed_face_closest_local_m": closest[seed_face].tolist() if seed_face >= 0 else None,
        "component_face_indices": component_faces,
        "component_vertices": component_vertices,
        "component_triangles": component_triangles,
        "component_digest": d350._component_digest(
            component_vertices, component_triangles, component_faces
        ),
        "component_count": int(len({find(index) for index in range(face_count)})),
    }


def _public_binding(row: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(row["component_vertices"], dtype=np.float64)
    triangles = np.asarray(row["component_triangles"], dtype=np.int64)
    return {
        "minimum_distance_m": row["minimum_distance_m"],
        "tied_face_indices": row["tied_face_indices"],
        "tied_component_roots": row["tied_component_roots"],
        "seed_face_index": row["seed_face_index"],
        "seed_face_closest_local_m": row["seed_face_closest_local_m"],
        "component_face_count": int(len(triangles)),
        "component_unique_vertex_count": int(len(vertices)),
        "component_digest": row["component_digest"],
        "component_face_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(row["component_face_indices"], dtype="<i8").tobytes()
        ).hexdigest(),
        "component_bounds_local_m": None
        if not len(vertices)
        else [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "component_count": row["component_count"],
    }


def _moving_patch_definition(shape: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(shape["_raw_verts"], dtype=np.float64)
    triangles = np.asarray(shape["_triangles"], dtype=np.int64)
    inner_faces = MOVING_INNER_PATCH_FACE_IDS
    outer_faces = MOVING_OUTER_PATCH_FACE_IDS

    def patch(face_ids: np.ndarray) -> dict[str, Any]:
        tri = vertices[triangles[face_ids]]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(lengths[:, None], 1.0e-30)
        unique_vertex_ids = np.unique(triangles[face_ids].reshape(-1))
        patch_vertices = np.unique(vertices[unique_vertex_ids], axis=0)
        return {
            "face_count": int(len(face_ids)),
            "unique_vertex_count": int(len(patch_vertices)),
            "face_id_min": int(face_ids.min()),
            "face_id_max": int(face_ids.max()),
            "face_id_sha256": hashlib.sha256(
                np.ascontiguousarray(face_ids.astype("<i8")).tobytes()
            ).hexdigest(),
            "bounds_local_m": [
                patch_vertices.min(axis=0).tolist(),
                patch_vertices.max(axis=0).tolist(),
            ],
            "normal_min": normals.min(axis=0).tolist(),
            "normal_max": normals.max(axis=0).tolist(),
            "max_normal_length_error": float(np.max(np.abs(lengths / lengths - 1.0))),
            "vertices": patch_vertices,
            "normals": normals,
        }

    inner = patch(inner_faces)
    outer = patch(outer_faces)
    inner_xz = np.unique(inner["vertices"][:, [0, 2]], axis=0)
    outer_xz = np.unique(outer["vertices"][:, [0, 2]], axis=0)
    checks = {
        "owner_gripper_link": shape["body"] == "gripper_link",
        "source_triangle_count_13698": len(triangles) == 13698,
        "inner_face_ids_frozen": inner["face_id_sha256"]
        == MOVING_INNER_PATCH_FACE_ID_SHA256,
        "outer_face_ids_frozen": outer["face_id_sha256"]
        == MOVING_OUTER_PATCH_FACE_ID_SHA256,
        "inner_493_faces_483_vertices": inner["face_count"] == 493
        and inner["unique_vertex_count"] == 483,
        "outer_493_faces_483_vertices": outer["face_count"] == 493
        and outer["unique_vertex_count"] == 483,
        "inner_plane_y_exact_source": bool(
            np.max(np.abs(inner["vertices"][:, 1] - MOVING_INNER_PATCH_Y_M))
            <= d350.BINDING_FACE_TIE_M
        ),
        "outer_plane_y_exact_source": bool(
            np.max(np.abs(outer["vertices"][:, 1] - MOVING_OUTER_PATCH_Y_M))
            <= d350.BINDING_FACE_TIE_M
        ),
        "inner_normal_minus_local_y": bool(
            np.max(
                np.abs(inner["normals"] - np.asarray([0.0, -1.0, 0.0]))
            )
            <= d350.BINDING_FACE_TIE_M
        ),
        "outer_normal_plus_local_y": bool(
            np.max(
                np.abs(outer["normals"] - np.asarray([0.0, 1.0, 0.0]))
            )
            <= d350.BINDING_FACE_TIE_M
        ),
        "paired_xz_vertex_stream_exact": np.array_equal(inner_xz, outer_xz),
        "inner_outer_are_distinct_face_sets": not np.intersect1d(
            inner_faces, outer_faces
        ).size,
    }
    return {
        "semantics": (
            "source-hash-frozen distal paired planar patches; minus-local-y patch "
            "is the closing-facing inner patch and plus-local-y is its outer negative control"
        ),
        "derivation_authority": (
            "frozen raw source face order under D334/D339 source parity; independent "
            "of the D351 first-contact witness"
        ),
        "inner": {
            key: value
            for key, value in inner.items()
            if key not in {"vertices", "normals"}
        },
        "outer": {
            key: value
            for key, value in outer.items()
            if key not in {"vertices", "normals"}
        },
        "paired_xz_vertex_sha256": hashlib.sha256(
            np.ascontiguousarray(inner_xz.astype("<f8")).tobytes()
        ).hexdigest(),
        "plane_separation_mm": (
            MOVING_OUTER_PATCH_Y_M - MOVING_INNER_PATCH_Y_M
        )
        * 1000.0,
        "numerical_definition_tolerance_m": d350.BINDING_FACE_TIE_M,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _authored_moving_patch_identity(shape: dict[str, Any]) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(AUTHORING_ROBOT_USD), load=Usd.Stage.LoadAll)
    if stage is None:
        return {"error": "failed to open authored robot USD", "pass": False}
    mesh_path = (
        "/roarm_m3/gripper_link/collisions/gripper_link/"
        "node_STL_BINARY_/mesh"
    )
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim.IsValid():
        return {"error": f"missing authored mesh {mesh_path}", "pass": False}
    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(
        [[float(p[0]), float(p[1]), float(p[2])] for p in mesh.GetPointsAttr().Get()],
        dtype="<f4",
    )
    counts = np.asarray(
        [int(value) for value in mesh.GetFaceVertexCountsAttr().Get()], dtype="<i8"
    )
    indices = np.asarray(
        [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()], dtype="<i8"
    )

    def blob(array: Any, dtype: str) -> bytes:
        return np.ascontiguousarray(array, dtype=dtype).tobytes(order="C")

    stream_hashes = {
        "points_f32_mm": hashlib.sha256(blob(points, "<f4")).hexdigest(),
        "face_counts_i64": hashlib.sha256(blob(counts, "<i8")).hexdigest(),
        "face_indices_i64": hashlib.sha256(blob(indices, "<i8")).hexdigest(),
    }
    source_faces = indices.reshape(-1, 3)

    def canonical_patch(face_ids: np.ndarray) -> dict[str, Any]:
        ids = np.sort(np.asarray(face_ids, dtype="<i8"))
        triangle_points = points[source_faces[ids]]
        unique_vertices, inverse = np.unique(
            triangle_points.reshape(-1, 3), axis=0, return_inverse=True
        )
        unique_vertices = np.asarray(unique_vertices, dtype="<f4")
        remapped = np.asarray(inverse.reshape(-1, 3), dtype="<i8")
        face_blob = blob(ids, "<i8")
        vertex_blob = blob(unique_vertices, "<f4")
        triangle_blob = blob(remapped, "<i8")
        return {
            "face_ids": ids,
            "vertices": unique_vertices,
            "triangles": remapped,
            "face_id_sha256": hashlib.sha256(face_blob).hexdigest(),
            "vertex_sha256": hashlib.sha256(vertex_blob).hexdigest(),
            "triangle_sha256": hashlib.sha256(triangle_blob).hexdigest(),
            "patch_digest": hashlib.sha256(
                face_blob + vertex_blob + triangle_blob
            ).hexdigest(),
        }

    inner = canonical_patch(MOVING_INNER_PATCH_FACE_IDS)
    outer = canonical_patch(MOVING_OUTER_PATCH_FACE_IDS)
    inner_xz = np.unique(inner["vertices"][:, [0, 2]], axis=0).astype("<f4")
    outer_xz = np.unique(outer["vertices"][:, [0, 2]], axis=0).astype("<f4")
    paired_xz_hash = hashlib.sha256(blob(inner_xz, "<f4")).hexdigest()
    inner_velocity = np.column_stack(
        [
            inner["vertices"][:, 1],
            -inner["vertices"][:, 0],
            np.zeros(len(inner["vertices"]), dtype=np.float32),
        ]
    )
    outer_velocity = np.column_stack(
        [
            outer["vertices"][:, 1],
            -outer["vertices"][:, 0],
            np.zeros(len(outer["vertices"]), dtype=np.float32),
        ]
    )
    inner_motion_dot = inner_velocity @ np.asarray([0.0, -1.0, 0.0])
    outer_motion_dot = outer_velocity @ np.asarray([0.0, 1.0, 0.0])
    runtime_vertices_m = np.asarray(shape["_raw_verts"], dtype=np.float64)
    runtime_triangles = np.asarray(shape["_triangles"], dtype=np.int64)
    runtime_recovered_f32_mm = np.asarray(runtime_vertices_m * 1000.0, dtype="<f4")
    checks = {
        "authored_stream_hashes_exact": stream_hashes
        == EXPECTED_AUTHORED_STREAM_HASHES,
        "all_authored_faces_triangles": bool(np.all(counts == 3)),
        "inner_face_id_hash_exact": inner["face_id_sha256"]
        == MOVING_INNER_PATCH_FACE_ID_SHA256,
        "outer_face_id_hash_exact": outer["face_id_sha256"]
        == MOVING_OUTER_PATCH_FACE_ID_SHA256,
        "inner_vertex_hash_exact": inner["vertex_sha256"]
        == EXPECTED_AUTHORED_PATCH_HASHES["inner_vertex"],
        "outer_vertex_hash_exact": outer["vertex_sha256"]
        == EXPECTED_AUTHORED_PATCH_HASHES["outer_vertex"],
        "both_triangle_hash_exact": inner["triangle_sha256"]
        == outer["triangle_sha256"]
        == EXPECTED_AUTHORED_PATCH_HASHES["triangle"],
        "inner_patch_digest_exact": inner["patch_digest"]
        == EXPECTED_AUTHORED_PATCH_HASHES["inner_patch"],
        "outer_patch_digest_exact": outer["patch_digest"]
        == EXPECTED_AUTHORED_PATCH_HASHES["outer_patch"],
        "paired_xz_exact_and_hash": np.array_equal(inner_xz, outer_xz)
        and paired_xz_hash == EXPECTED_AUTHORED_PATCH_HASHES["paired_xz"],
        "inner_closing_motion_dot_strictly_positive": bool(
            np.all(inner_motion_dot > 0.0)
        ),
        "outer_closing_motion_dot_strictly_negative": bool(
            np.all(outer_motion_dot < 0.0)
        ),
        "runtime_face_order_maps_authored_exact": np.array_equal(
            runtime_triangles, source_faces
        ),
        "runtime_body_points_recover_authored_f32_mm_exact": np.array_equal(
            runtime_recovered_f32_mm, points
        ),
    }
    return {
        "mesh_path": mesh_path,
        "authored_coordinate_units": "millimeters",
        "stream_hashes": stream_hashes,
        "expected_stream_hashes": EXPECTED_AUTHORED_STREAM_HASHES,
        "inner": {
            "face_id_sha256": inner["face_id_sha256"],
            "vertex_sha256": inner["vertex_sha256"],
            "triangle_sha256": inner["triangle_sha256"],
            "patch_digest": inner["patch_digest"],
            "face_count": int(len(inner["face_ids"])),
            "vertex_count": int(len(inner["vertices"])),
            "closing_motion_dot_range_mm_per_rad": [
                float(np.min(inner_motion_dot)),
                float(np.max(inner_motion_dot)),
            ],
        },
        "outer_negative_control": {
            "face_id_sha256": outer["face_id_sha256"],
            "vertex_sha256": outer["vertex_sha256"],
            "triangle_sha256": outer["triangle_sha256"],
            "patch_digest": outer["patch_digest"],
            "face_count": int(len(outer["face_ids"])),
            "vertex_count": int(len(outer["vertices"])),
            "closing_motion_dot_range_mm_per_rad": [
                float(np.min(outer_motion_dot)),
                float(np.max(outer_motion_dot)),
            ],
        },
        "paired_xz_sha256": paired_xz_hash,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _extract_face_patch(
    vertices: Any, triangles: Any, face_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)[face_ids]
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return verts[used], remap[faces]


def _classify_moving_patch_faces(face_ids: Any) -> dict[str, Any]:
    tied = np.asarray(face_ids, dtype=np.int64)
    all_inner = bool(
        len(tied) and np.all(np.isin(tied, MOVING_INNER_PATCH_FACE_IDS))
    )
    all_outer = bool(
        len(tied) and np.all(np.isin(tied, MOVING_OUTER_PATCH_FACE_IDS))
    )
    any_inner = bool(np.any(np.isin(tied, MOVING_INNER_PATCH_FACE_IDS)))
    any_outer = bool(np.any(np.isin(tied, MOVING_OUTER_PATCH_FACE_IDS)))
    if not len(tied):
        classification = "unresolved_empty_face_set_fail_stop"
    elif all_inner:
        classification = "frozen_distal_inner_minus_local_y"
    elif all_outer:
        classification = "paired_outer_plus_local_y_negative_control"
    elif any_inner or any_outer:
        classification = "ambiguous_patch_boundary_fail_stop"
    else:
        classification = "housing_or_nonpad_first"
    return {
        "classification": classification,
        "tied_face_indices": tied.tolist(),
        "all_tied_faces_on_inner_patch": all_inner,
        "all_tied_faces_on_outer_patch": all_outer,
        "any_tied_face_on_inner_patch": any_inner,
        "any_tied_face_on_outer_patch": any_outer,
        "identity_unambiguous": classification
        not in {
            "ambiguous_patch_boundary_fail_stop",
            "unresolved_empty_face_set_fail_stop",
        },
    }


def _surface_face_measurement(
    vertices: Any,
    triangles: Any,
    body_pos: np.ndarray,
    body_rot: np.ndarray,
    seed_world: np.ndarray,
    toward_world: np.ndarray,
) -> dict[str, Any]:
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    seed_local = body_rot.T @ (seed_world - body_pos)
    distances = np.empty(len(faces), dtype=np.float64)
    closest = np.empty((len(faces), 3), dtype=np.float64)
    for face_index, ids in enumerate(faces):
        a, b, c = verts[ids]
        point = d350._closest_point_triangle(seed_local, a, b, c)
        closest[face_index] = point
        distances[face_index] = float(np.linalg.norm(seed_local - point))
    minimum = float(np.min(distances))
    tied = np.flatnonzero(distances <= minimum + d350.BINDING_FACE_TIE_M)
    face_index = int(tied[0])
    triangle = verts[faces[face_index]]
    normal_local = np.cross(
        triangle[1] - triangle[0], triangle[2] - triangle[0]
    )
    normal_length = float(np.linalg.norm(normal_local))
    normal_local /= max(normal_length, 1.0e-30)
    normal_world = body_rot @ normal_local
    toward = toward_world - seed_world
    if float(np.dot(normal_world, toward)) < 0.0:
        normal_world = -normal_world
        normal_local = -normal_local
    return {
        "seed_world_m": seed_world.tolist(),
        "seed_local_m": seed_local.tolist(),
        "minimum_surface_residual_m": minimum,
        "tied_face_indices": tied.tolist(),
        "selected_face_index": face_index,
        "selected_face_triangle_local_m": triangle.tolist(),
        "oriented_normal_local": normal_local.tolist(),
        "oriented_normal_world": normal_world.tolist(),
        "normal_length_before_normalization": normal_length,
        "checks": {
            "residual_le_0p01mm": minimum <= BINDING_RESIDUAL_MAX_M,
            "tied_faces_nonempty": len(tied) > 0,
            "normal_nondegenerate": normal_length > 0.0,
            "normal_finite": bool(np.isfinite(normal_world).all()),
        },
    }


def _set_state_only(inner: Any, timeline: Any, q5: float) -> dict[str, Any]:
    if timeline.is_playing():
        timeline.pause()
    counter = int(inner._sim_step_counter)
    time_value = float(timeline.get_current_time())
    simulation_clock = _simulation_clock(inner)
    q = Q_FROZEN_F32.copy()
    q[5] = np.float32(q5)
    d332._write_exact_state(inner, q.astype(np.float64), OBJECT_POS_F32.astype(np.float64))
    guard = _state_guard(
        inner, q5, counter, time_value, timeline, simulation_clock
    )
    if not guard["pass"]:
        raise RuntimeError(
            f"D351 zero-step direct-write guard failed at q5={q5}: "
            f"{guard['checks']}"
        )
    return guard


def _unflipped_triangle_normal_world(
    triangle_local: Any, body_rot: np.ndarray
) -> np.ndarray:
    triangle = np.asarray(triangle_local, dtype=np.float64)
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    length = float(np.linalg.norm(normal))
    if not math.isfinite(length) or length <= 0.0:
        return np.asarray([math.nan, math.nan, math.nan], dtype=np.float64)
    return np.asarray(body_rot, dtype=np.float64) @ (normal / length)


def _overlap_contact_surface_audit(
    inner: Any,
    parts: list[dict[str, Any]],
    geometry_key: str,
    raw_shape: dict[str, Any],
    representation: str,
) -> dict[str, Any]:
    import hppfcl

    body_pos, body_quat = d334._body_pose_w(inner, "gripper_link")
    body_rot = _quat_to_rot(body_quat)
    body_tf = hppfcl.Transform3f(body_rot, body_pos)
    object_pos, object_quat = d334._object_pose_w(inner)
    cylinder = hppfcl.Cylinder(d332.CYLINDER_RADIUS_M, d332.CYLINDER_HEIGHT_M)
    cylinder_tf = hppfcl.Transform3f(_quat_to_rot(object_quat), object_pos)
    frozen_inner_normal_world = body_rot @ np.asarray([0.0, -1.0, 0.0])
    rows: list[dict[str, Any]] = []
    saturated_paths: list[str] = []
    colliding_paths: list[str] = []
    for part in parts:
        request = hppfcl.CollisionRequest()
        request.enable_contact = True
        request.num_max_contacts = int(d349.d336.EPA_MAX_CONTACTS)
        result = hppfcl.CollisionResult()
        hppfcl.collide(
            part[geometry_key], body_tf, cylinder, cylinder_tf, request, result
        )
        if not result.isCollision():
            continue
        path = str(part["path"])
        colliding_paths.append(path)
        if int(result.numContacts()) >= int(d349.d336.EPA_MAX_CONTACTS):
            saturated_paths.append(path)
        triangles = np.asarray(part["_triangles"], dtype=np.int64)
        vertices = np.asarray(part["_vertices"], dtype=np.float64)
        for contact_index in range(int(result.numContacts())):
            contact = result.getContact(contact_index)
            face_index = int(contact.b1)
            position = np.asarray(contact.pos, dtype=np.float64)
            api_normal = np.asarray(contact.normal, dtype=np.float64)
            api_normal_norm = float(np.linalg.norm(api_normal))
            api_normal_unit = (
                api_normal / api_normal_norm
                if math.isfinite(api_normal_norm) and api_normal_norm > 0.0
                else api_normal
            )
            depth_m = abs(float(contact.penetration_depth))
            endpoint_cylinder = position + api_normal_unit * depth_m
            cylinder_feature = _feature_from_cylinder_witness(
                endpoint_cylinder, object_pos
            )
            face_valid = 0 <= face_index < len(triangles)
            triangle_local = (
                vertices[triangles[face_index]]
                if face_valid
                else np.full((3, 3), math.nan, dtype=np.float64)
            )
            unflipped_normal_world = _unflipped_triangle_normal_world(
                triangle_local, body_rot
            )
            if representation == "raw":
                raw_mapping = {
                    **_classify_moving_patch_faces([face_index] if face_valid else []),
                    "minimum_surface_residual_m": 0.0 if face_valid else None,
                    "mapping_method": "hpp-fcl Contact.b1 direct authored triangle id",
                }
            else:
                projected = _surface_face_measurement(
                    raw_shape["_raw_verts"],
                    raw_shape["_triangles"],
                    body_pos,
                    body_rot,
                    position,
                    endpoint_cylinder,
                )
                raw_mapping = {
                    **_classify_moving_patch_faces(projected["tied_face_indices"]),
                    "minimum_surface_residual_m": projected[
                        "minimum_surface_residual_m"
                    ],
                    "selected_raw_face_index": projected["selected_face_index"],
                    "mapping_method": (
                        "world Contact.pos projected to same-q5 full authored raw mesh"
                    ),
                }
            rows.append(
                {
                    "part_path": path,
                    "contact_index": contact_index,
                    "hppfcl_b1_triangle_index": face_index,
                    "hppfcl_b2_primitive_index": int(contact.b2),
                    "face_index_valid": face_valid,
                    "depth_m": depth_m,
                    "position_world_m": position.tolist(),
                    "api_normal_o1_to_o2": api_normal_unit.tolist(),
                    "api_normal_length_before_normalization": api_normal_norm,
                    "cylinder_endpoint_world_m": endpoint_cylinder.tolist(),
                    "cylinder_feature": cylinder_feature,
                    "unflipped_triangle_normal_world": unflipped_normal_world.tolist(),
                    "unflipped_normal_dot_frozen_inner": float(
                        np.dot(unflipped_normal_world, frozen_inner_normal_world)
                    ),
                    "raw_surface_mapping": raw_mapping,
                    "finite": bool(
                        math.isfinite(depth_m)
                        and np.isfinite(position).all()
                        and np.isfinite(api_normal_unit).all()
                        and api_normal_norm > 0.0
                        and np.isfinite(unflipped_normal_world).all()
                    ),
                }
            )
    classifications = {
        row["raw_surface_mapping"]["classification"] for row in rows
    }
    unambiguous = bool(
        rows
        and len(classifications) == 1
        and all(
            row["raw_surface_mapping"]["identity_unambiguous"] for row in rows
        )
    )
    classification = next(iter(classifications)) if unambiguous else None
    contact_part_paths = sorted({row["part_path"] for row in rows})
    colliding_part_paths = sorted(set(colliding_paths))
    cylinder_features = {
        row["cylinder_feature"]["feature"] for row in rows
    }
    cylinder_feature_consensus = bool(rows and len(cylinder_features) == 1)
    checks = {
        "colliding_part_and_contact_nonempty": bool(colliding_paths and rows),
        "colliding_part_paths_equal_contact_row_paths": (
            colliding_part_paths == contact_part_paths
        ),
        "no_contact_cap_saturation": not saturated_paths,
        "all_contact_rows_finite": bool(rows) and all(row["finite"] for row in rows),
        "all_b1_triangle_indices_valid": bool(rows)
        and all(row["face_index_valid"] for row in rows),
        "all_live_to_raw_residuals_le_inherited_0p5mm": representation == "raw"
        or (
            bool(rows)
            and all(
                row["raw_surface_mapping"]["minimum_surface_residual_m"]
                <= FIDELITY_TOL_MM / 1000.0
                for row in rows
            )
        ),
        "raw_surface_classification_consensus": unambiguous,
        "all_contact_cylinder_features_consensus": cylinder_feature_consensus,
    }
    return {
        "representation": representation,
        "hppfcl_contact_semantics": (
            "b1 is object1 BVH triangle id; pos is world contact position; "
            "normal points object1 to object2"
        ),
        "colliding_part_paths": colliding_part_paths,
        "contact_row_part_paths": contact_part_paths,
        "saturated_part_paths": saturated_paths,
        "contacts": rows,
        "raw_surface_consensus_classification": classification,
        "cylinder_feature_consensus": (
            next(iter(cylinder_features)) if cylinder_feature_consensus else None
        ),
        "all_contact_cylinder_features": sorted(cylinder_features),
        "all_contacts_on_barrel_interior": bool(
            cylinder_features == {"barrel_interior"}
        ),
        "all_inner_unflipped_normals_same_hemisphere": bool(
            classification == "frozen_distal_inner_minus_local_y"
            and all(row["unflipped_normal_dot_frozen_inner"] > 0.0 for row in rows)
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _triangle_soup_bvh(hppfcl: Any, triangle_rows: list[np.ndarray]) -> Any:
    soup = np.asarray(triangle_rows, dtype=np.float64)
    vertices = soup.reshape(-1, 3)
    triangles = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return d332._build_raw_bvh(hppfcl, vertices, triangles)


def _minimum_distance_to_triangle_patch(
    point: np.ndarray, patch_triangles: np.ndarray
) -> float:
    distances = [
        float(
            np.linalg.norm(
                np.asarray(point, dtype=np.float64)
                - d350._closest_point_triangle(
                    np.asarray(point, dtype=np.float64), triangle[0], triangle[1], triangle[2]
                )
            )
        )
        for triangle in np.asarray(patch_triangles, dtype=np.float64)
    ]
    return min(distances) if distances else math.inf


def _live_inner_complement_partition(
    hppfcl: Any,
    topology_parts: list[dict[str, Any]],
    raw_shape: dict[str, Any],
) -> tuple[Any | None, Any | None, float, dict[str, Any]]:
    raw_vertices = np.asarray(raw_shape["_raw_verts"], dtype=np.float64)
    raw_triangles = np.asarray(raw_shape["_triangles"], dtype=np.int64)
    raw_inner_triangles = raw_vertices[
        raw_triangles[MOVING_INNER_PATCH_FACE_IDS]
    ]
    inner_rows: list[np.ndarray] = []
    complement_rows: list[np.ndarray] = []
    inner_keys: list[str] = []
    inner_parts: set[str] = set()
    source_vertex_points: list[np.ndarray] = []
    interior_diagnostic_points: list[np.ndarray] = []
    total_face_count = 0
    all_normals_finite_nondegenerate = True
    all_vertices: list[np.ndarray] = []
    expected_inner_normal = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    for part in topology_parts:
        vertices = np.asarray(part["_vertices"], dtype=np.float64)
        triangles = np.asarray(part["_triangles"], dtype=np.int64)
        all_vertices.append(vertices)
        for face_index, vertex_ids in enumerate(triangles):
            triangle = vertices[vertex_ids]
            total_face_count += 1
            normal = np.cross(
                triangle[1] - triangle[0], triangle[2] - triangle[0]
            )
            length = float(np.linalg.norm(normal))
            finite_nondegenerate = bool(
                math.isfinite(length) and length > 0.0 and np.isfinite(normal).all()
            )
            all_normals_finite_nondegenerate &= finite_nondegenerate
            normal_unit = normal / length if finite_nondegenerate else normal
            plane_match = bool(
                np.max(np.abs(triangle[:, 1] - MOVING_INNER_PATCH_Y_M))
                <= d350.BINDING_FACE_TIE_M
            )
            normal_match = bool(
                finite_nondegenerate
                and np.array_equal(normal_unit, expected_inner_normal)
            )
            if plane_match and normal_match:
                inner_rows.append(triangle)
                key = f"{part['name']}:{face_index}"
                inner_keys.append(key)
                inner_parts.add(str(part["name"]))
                source_vertex_points.extend(
                    [triangle[0], triangle[1], triangle[2]]
                )
                interior_diagnostic_points.extend(
                    [
                        0.5 * (triangle[0] + triangle[1]),
                        0.5 * (triangle[1] + triangle[2]),
                        0.5 * (triangle[2] + triangle[0]),
                        np.mean(triangle, axis=0),
                    ]
                )
            else:
                # Boundary or unresolved faces stay in the competitor set.  This
                # is deliberately conservative for a positive first-contact claim.
                complement_rows.append(triangle)
    key_hash = hashlib.sha256("\n".join(sorted(inner_keys)).encode()).hexdigest()
    unique_source_vertices = (
        np.unique(np.asarray(source_vertex_points, dtype=np.float64), axis=0)
        if source_vertex_points
        else np.zeros((0, 3), dtype=np.float64)
    )
    unique_interior_diagnostics = (
        np.unique(np.asarray(interior_diagnostic_points, dtype=np.float64), axis=0)
        if interior_diagnostic_points
        else np.zeros((0, 3), dtype=np.float64)
    )
    source_vertex_residuals = [
        _minimum_distance_to_triangle_patch(point, raw_inner_triangles)
        for point in unique_source_vertices
    ]
    interior_diagnostic_residuals = [
        _minimum_distance_to_triangle_patch(point, raw_inner_triangles)
        for point in unique_interior_diagnostics
    ]
    max_source_vertex_residual_m = (
        max(source_vertex_residuals) if source_vertex_residuals else math.inf
    )
    max_interior_diagnostic_residual_m = (
        max(interior_diagnostic_residuals)
        if interior_diagnostic_residuals
        else math.inf
    )
    inner_geometry = (
        _triangle_soup_bvh(hppfcl, inner_rows) if inner_rows else None
    )
    complement_geometry = (
        _triangle_soup_bvh(hppfcl, complement_rows)
        if complement_rows
        else None
    )
    radius_m = (
        float(
            np.max(
                np.linalg.norm(np.vstack(all_vertices)[:, :2], axis=1)
            )
        )
        if all_vertices
        else math.inf
    )
    checks = {
        "source_part_count_64": len(topology_parts) == 64,
        "source_triangle_count_832": total_face_count
        == EXPECTED_LIVE_GRIPPER_TRIANGLE_COUNT,
        "all_source_triangles_finite_nondegenerate": all_normals_finite_nondegenerate,
        "inner_face_count_frozen_40": len(inner_rows)
        == EXPECTED_LIVE_INNER_FACE_COUNT,
        "inner_part_count_frozen_17": len(inner_parts)
        == EXPECTED_LIVE_INNER_PART_COUNT,
        "inner_face_key_hash_exact": key_hash
        == EXPECTED_LIVE_INNER_FACE_KEY_SHA256,
        "inner_and_complement_disjoint_exhaustive": (
            len(inner_rows) + len(complement_rows) == total_face_count
            and bool(inner_rows)
            and bool(complement_rows)
        ),
        "inner_source_vertices_map_to_authored_patch_within_inherited_0p5mm": bool(
            source_vertex_residuals
            and all(math.isfinite(value) for value in source_vertex_residuals)
            and max_source_vertex_residual_m <= FIDELITY_TOL_MM / 1000.0
        ),
        "inner_and_complement_bvhs_nonnull": (
            inner_geometry is not None and complement_geometry is not None
        ),
        "finite_rotation_radius": math.isfinite(radius_m) and radius_m > 0.0,
    }
    report = {
        "artifact": "D351_LIVE_INNER_COMPLEMENT_PARTITION_V1",
        "authority": (
            "D348 callback-topology triangles in gripper body-local coordinates; "
            "no regenerated convex hull"
        ),
        "inner_rule": (
            "all triangle vertices on frozen authored inner y-plane within the "
            "existing 1nm face-tie integrity control, exact unflipped -localY "
            "normal, frozen face-key hash, and source-vertex projection to the authored "
            "inner patch within the inherited 0.5mm representation gate"
        ),
        "convex_face_interior_semantics": (
            "edge-midpoint and centroid residuals are diagnostic-only because the "
            "actual callback collider may bridge authored concavities; those actual "
            "coplanar -localY faces remain the live-inner surface queried below"
        ),
        "boundary_rule": "every non-matching or unresolved triangle is competitor",
        "inner_face_count": len(inner_rows),
        "complement_face_count": len(complement_rows),
        "source_face_count": total_face_count,
        "inner_part_names": sorted(inner_parts),
        "inner_face_keys": sorted(inner_keys),
        "inner_face_key_sha256": key_hash,
        "source_vertex_count": len(unique_source_vertices),
        "maximum_source_vertex_to_authored_inner_patch_residual_m": (
            max_source_vertex_residual_m
        ),
        "interior_diagnostic_point_count": len(unique_interior_diagnostics),
        "maximum_interior_diagnostic_to_authored_inner_patch_residual_m": (
            max_interior_diagnostic_residual_m
        ),
        "interior_diagnostic_verdict_authority": False,
        "rotation_radius_m": radius_m,
        "checks": checks,
        "pass": all(checks.values()),
    }
    return inner_geometry, complement_geometry, radius_m, report


def _competitor_exclusion_query(
    inner: Any,
    inner_geometry: Any,
    complement_geometry: Any,
    radius_m: float,
    q_clear: float,
    q_overlap: float,
    representation: str,
    partition: dict[str, Any],
) -> dict[str, Any]:
    import hppfcl

    body_pos, body_quat = d334._body_pose_w(inner, "gripper_link")
    body_tf = hppfcl.Transform3f(_quat_to_rot(body_quat), body_pos)
    object_pos, object_quat = d334._object_pose_w(inner)
    object_rot = _quat_to_rot(object_quat)
    cylinder = hppfcl.Cylinder(d332.CYLINDER_RADIUS_M, d332.CYLINDER_HEIGHT_M)
    cylinder_tf = hppfcl.Transform3f(object_rot, object_pos)
    complement_query = d332._fcl_query(
        hppfcl, complement_geometry, body_tf, cylinder, cylinder_tf
    )
    cap_queries: dict[str, Any] = {}
    for name, sign in (("bottom", -1.0), ("top", 1.0)):
        cap = hppfcl.Cylinder(d332.CYLINDER_RADIUS_M, 0.0)
        cap_center = object_pos + object_rot @ np.asarray(
            [0.0, 0.0, sign * 0.5 * d332.CYLINDER_HEIGHT_M]
        )
        cap_tf = hppfcl.Transform3f(object_rot, cap_center)
        cap_queries[name] = d332._fcl_query(
            hppfcl, inner_geometry, body_tf, cap, cap_tf
        )
    bracket_motion_bound_mm = _interval_bound_mm(
        radius_m, q_clear, q_overlap
    )

    def separated_beyond_bound(query: dict[str, Any]) -> bool:
        value = query.get("signed_distance_mm")
        return bool(
            value is not None
            and math.isfinite(float(value))
            and not query.get("is_collision")
            and float(value) > bracket_motion_bound_mm
        )

    def public_query(query: dict[str, Any]) -> dict[str, Any]:
        return {
            "signed_distance_mm": query.get("signed_distance_mm"),
            "is_collision": query.get("is_collision"),
            "nearest_point_geometry_m": query.get("nearest_point_geometry_m"),
            "nearest_point_competitor_m": query.get("nearest_point_cylinder_m"),
        }

    checks = {
        "inner_complement_partition_pass": partition["pass"],
        "complement_strictly_farther_than_bracket_motion": separated_beyond_bound(
            complement_query
        ),
        "bottom_cap_disk_strictly_farther_than_bracket_motion": separated_beyond_bound(
            cap_queries["bottom"]
        ),
        "top_cap_disk_strictly_farther_than_bracket_motion": separated_beyond_bound(
            cap_queries["top"]
        ),
    }
    return {
        "representation": representation,
        "q_clear_float32_rad": q_clear,
        "q_overlap_float32_rad": q_overlap,
        "bracket_motion_bound_mm": bracket_motion_bound_mm,
        "rotation_radius_m": radius_m,
        "surface_partition": partition,
        "complement_query": public_query(complement_query),
        "cap_disk_queries": {
            name: public_query(query) for name, query in cap_queries.items()
        },
        "cap_disk_semantics": (
            "hpp-fcl analytic Cylinder(radius=frozen radius,height=0) planar disk; "
            "measurement-only competitor geometry"
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _raw_first_contact_competitor_exclusion(
    inner: Any,
    raw_shape: dict[str, Any],
    q_clear: float,
    q_overlap: float,
) -> dict[str, Any]:
    import hppfcl

    vertices = np.asarray(raw_shape["_raw_verts"], dtype=np.float64)
    triangles = np.asarray(raw_shape["_triangles"], dtype=np.int64)
    complement_ids = np.setdiff1d(
        np.arange(len(triangles), dtype=np.int64),
        MOVING_INNER_PATCH_FACE_IDS,
        assume_unique=True,
    )
    inner_geometry = d332._build_raw_bvh(
        hppfcl, vertices, triangles[MOVING_INNER_PATCH_FACE_IDS]
    )
    complement_geometry = d332._build_raw_bvh(
        hppfcl, vertices, triangles[complement_ids]
    )
    partition = {
        "artifact": "D351_RAW_INNER_COMPLEMENT_PARTITION_V1",
        "authority": "frozen authored face IDs on retained full raw mesh",
        "inner_face_count": int(len(MOVING_INNER_PATCH_FACE_IDS)),
        "complement_face_count": int(len(complement_ids)),
        "checks": {
            "inner_face_ids_frozen": hashlib.sha256(
                np.ascontiguousarray(
                    MOVING_INNER_PATCH_FACE_IDS.astype("<i8")
                ).tobytes()
            ).hexdigest()
            == MOVING_INNER_PATCH_FACE_ID_SHA256,
            "partition_disjoint_exhaustive": (
                len(MOVING_INNER_PATCH_FACE_IDS) + len(complement_ids)
                == len(triangles)
            ),
        },
    }
    partition["pass"] = all(partition["checks"].values())
    radius_m = float(np.max(np.linalg.norm(vertices[:, :2], axis=1)))
    return _competitor_exclusion_query(
        inner,
        inner_geometry,
        complement_geometry,
        radius_m,
        q_clear,
        q_overlap,
        "raw",
        partition,
    )


def _live_first_contact_competitor_exclusion(
    inner: Any,
    topology_parts: list[dict[str, Any]],
    raw_shape: dict[str, Any],
    q_clear: float,
    q_overlap: float,
) -> dict[str, Any]:
    import hppfcl

    inner_geometry, complement_geometry, radius_m, partition = (
        _live_inner_complement_partition(hppfcl, topology_parts, raw_shape)
    )
    if inner_geometry is None or complement_geometry is None:
        return {
            "representation": "live",
            "q_clear_float32_rad": q_clear,
            "q_overlap_float32_rad": q_overlap,
            "surface_partition": partition,
            "checks": {"inner_complement_partition_pass": False},
            "pass": False,
        }
    return _competitor_exclusion_query(
        inner,
        inner_geometry,
        complement_geometry,
        radius_m,
        q_clear,
        q_overlap,
        "live_callback_topology_surface_proxy",
        partition,
    )


def _bind_moving_surface(
    inner: Any,
    timeline: Any,
    raw_shapes: list[dict[str, Any]],
    topology_parts: dict[str, list[dict[str, Any]]],
    raw_contact_clear: dict[str, Any],
    raw_contact_overlap: dict[str, Any],
    live_contact_clear: dict[str, Any],
    live_contact_overlap: dict[str, Any],
    open_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    q_clear = float(raw_contact_clear["q5_float32_rad"])
    guard = _set_state_only(inner, timeline, q_clear)
    shape = next(row for row in raw_shapes if row["body"] == "gripper_link")
    gripper_pos, gripper_quat = d334._body_pose_w(inner, "gripper_link")
    gripper_rot = _quat_to_rot(gripper_quat)
    seed_world = np.asarray(
        raw_contact_clear["raw"]["gripper_link"]["witness_endpoint_geometry_m"],
        dtype=np.float64,
    )
    seed_local = gripper_rot.T @ (seed_world - gripper_pos)
    vertex_first = d350._bind_component(
        shape["_raw_verts"], shape["_triangles"], seed_local, reverse_order=False
    )
    vertex_repeat = d350._bind_component(
        shape["_raw_verts"], shape["_triangles"], seed_local, reverse_order=True
    )
    edge_first = _bind_component_edge(
        shape["_raw_verts"], shape["_triangles"], seed_local, reverse_order=False
    )
    edge_repeat = _bind_component_edge(
        shape["_raw_verts"], shape["_triangles"], seed_local, reverse_order=True
    )

    _set_state_only(inner, timeline, float(Q5_OPEN_F32))
    open_pos, open_quat = d334._body_pose_w(inner, "gripper_link")
    open_seed_world = np.asarray(
        open_row["raw"]["gripper_link"]["witness_endpoint_geometry_m"], dtype=np.float64
    )
    open_seed_local = _quat_to_rot(open_quat).T @ (open_seed_world - open_pos)
    open_binding = d350._bind_component(
        shape["_raw_verts"], shape["_triangles"], open_seed_local, reverse_order=False
    )
    _set_state_only(inner, timeline, q_clear)

    vertex_faces = np.asarray(vertex_first["component_face_indices"], dtype=np.int64)
    edge_faces = np.asarray(edge_first["component_face_indices"], dtype=np.int64)
    seed_face = int(vertex_first["seed_face_index"])
    triangle_local = np.asarray(shape["_raw_verts"], dtype=np.float64)[
        np.asarray(shape["_triangles"], dtype=np.int64)[seed_face]
    ]
    normal_local = np.cross(
        triangle_local[1] - triangle_local[0], triangle_local[2] - triangle_local[0]
    )
    normal_local /= max(float(np.linalg.norm(normal_local)), 1.0e-15)
    normal_world = gripper_rot @ normal_local
    cylinder_witness = np.asarray(
        raw_contact_clear["raw"]["gripper_link"]["witness_endpoint_cylinder_m"],
        dtype=np.float64,
    )
    gap = cylinder_witness - seed_world
    if float(np.dot(normal_world, gap)) < 0.0:
        normal_world = -normal_world
        normal_local = -normal_local
    joint = d350._joint_semantics()
    runtime_joint = _runtime_joint_geometry(inner)
    patch_definition = _moving_patch_definition(shape)
    authored_patch_identity = _authored_moving_patch_identity(shape)
    raw_clear_identity = _classify_moving_patch_faces(
        vertex_first["tied_face_indices"]
    )
    raw_parts_for_contact = [
        {
            "path": shape["collider_path"],
            "_geometry_raw": shape["_geom_raw"],
            "_vertices": shape["_raw_verts"],
            "_triangles": shape["_triangles"],
        }
    ]
    raw_q_overlap = float(raw_contact_overlap["q5_float32_rad"])
    raw_overlap_guard = _set_state_only(inner, timeline, raw_q_overlap)
    raw_overlap_contacts = _overlap_contact_surface_audit(
        inner,
        raw_parts_for_contact,
        "_geometry_raw",
        shape,
        "raw",
    )
    _set_state_only(inner, timeline, q_clear)
    raw_competitor_exclusion = _raw_first_contact_competitor_exclusion(
        inner, shape, q_clear, raw_q_overlap
    )

    fixed_measurement = _json(D350_MEASUREMENT)
    fixed_seed = np.asarray(
        fixed_measurement["actual_surface"]["current_raw_witness_world_m"],
        dtype=np.float64,
    )
    fixed_normal = np.asarray(
        fixed_measurement["actual_surface"]["oriented_surface_normal_world"],
        dtype=np.float64,
    )
    fixed_cylinder_witness = np.asarray(
        fixed_measurement["actual_surface"]["cylinder_witness_world_m"],
        dtype=np.float64,
    )
    center = OBJECT_POS_F32.astype(np.float64)
    joint_origin = np.asarray(runtime_joint["origin_world_m"], dtype=np.float64)
    joint_axis = np.asarray(runtime_joint["axis_world"], dtype=np.float64)
    close_velocity_per_positive_rad = -np.cross(joint_axis, seed_world - joint_origin)
    chord = fixed_seed - seed_world
    chord_xy = chord[:2]
    chord_xy_norm_sq = float(np.dot(chord_xy, chord_xy))
    center_projection_t = (
        None
        if chord_xy_norm_sq <= 0.0
        else float(np.dot(center[:2] - seed_world[:2], chord_xy) / chord_xy_norm_sq)
    )
    center_projection_xy = (
        None
        if center_projection_t is None
        else seed_world[:2] + center_projection_t * chord_xy
    )
    chord_centerline_miss_mm = (
        None
        if center_projection_xy is None
        else float(np.linalg.norm(center[:2] - center_projection_xy) * 1000.0)
    )
    fixed_moving_normal_dot = float(np.dot(fixed_normal, normal_world))
    close_toward_fixed_dot = float(
        np.dot(close_velocity_per_positive_rad, fixed_seed - seed_world)
    )
    close_along_moving_normal_dot = float(
        np.dot(close_velocity_per_positive_rad, normal_world)
    )
    opposite_center_sides_dot_m2 = float(
        np.dot(seed_world[:2] - center[:2], fixed_seed[:2] - center[:2])
    )
    fixed_normal_toward_moving_dot = float(
        np.dot(fixed_normal, seed_world - fixed_seed)
    )
    moving_normal_toward_fixed_dot = float(
        np.dot(normal_world, fixed_seed - seed_world)
    )
    moving_cylinder_radial = cylinder_witness[:2] - center[:2]
    fixed_cylinder_radial = fixed_cylinder_witness[:2] - center[:2]
    cylinder_support_opposition_dot_m2 = float(
        np.dot(moving_cylinder_radial, fixed_cylinder_radial)
    )
    support_midpoint = 0.5 * (cylinder_witness + fixed_cylinder_witness)
    pinch_checks = {
        "moving_and_fixed_inward_normals_opposed": fixed_moving_normal_dot < 0.0,
        "fixed_normal_faces_moving_surface": fixed_normal_toward_moving_dot > 0.0,
        "moving_normal_faces_fixed_surface": moving_normal_toward_fixed_dot > 0.0,
        "q5_decrease_moves_contact_toward_fixed_surface": close_toward_fixed_dot > 0.0,
        "q5_decrease_moves_along_moving_inward_normal": close_along_moving_normal_dot > 0.0,
        "cylinder_center_projection_inside_contact_chord": center_projection_t is not None
        and 0.0 < center_projection_t < 1.0,
        "jaw_surface_points_on_opposite_xy_sides_of_center": opposite_center_sides_dot_m2
        < 0.0,
        "cylinder_support_witnesses_on_opposite_xy_halfplanes": cylinder_support_opposition_dot_m2
        < 0.0,
    }
    live_q_clear = float(live_contact_clear["q5_float32_rad"])
    live_guard = _set_state_only(inner, timeline, live_q_clear)
    live_pos, live_quat = d334._body_pose_w(inner, "gripper_link")
    live_rot = _quat_to_rot(live_quat)
    live_query = live_contact_clear["live"]["gripper_link"]
    live_part_path = live_query["witness_part_path"]
    live_part = next(
        (
            part
            for part in topology_parts["gripper_link"]
            if part["path"] == live_part_path
        ),
        None,
    )
    if live_part is None:
        live_surface = {
            "part_path": live_part_path,
            "error": "selected live witness part was not found",
            "checks": {"selected_part_found": False},
            "pass": False,
        }
        live_normal_world = None
        live_unflipped_normal_dot = None
        live_clear_raw_mapping = {
            **_classify_moving_patch_faces([]),
            "minimum_surface_residual_m": None,
        }
    else:
        live_seed_world = np.asarray(
            live_query["witness_endpoint_geometry_m"], dtype=np.float64
        )
        live_cylinder_world = np.asarray(
            live_query["witness_endpoint_cylinder_m"], dtype=np.float64
        )
        live_surface = _surface_face_measurement(
            live_part["_vertices"],
            live_part["_triangles"],
            live_pos,
            live_rot,
            live_seed_world,
            live_cylinder_world,
        )
        live_surface["part_path"] = live_part_path
        live_surface["checks"]["selected_part_found"] = True
        live_surface["pass"] = all(live_surface["checks"].values())
        live_normal_world = np.asarray(
            live_surface["oriented_normal_world"], dtype=np.float64
        )
        live_unflipped_normal = _unflipped_triangle_normal_world(
            live_surface["selected_face_triangle_local_m"], live_rot
        )
        live_unflipped_normal_dot = float(
            np.dot(
                live_unflipped_normal,
                live_rot @ np.asarray([0.0, -1.0, 0.0]),
            )
        )
        projected = _surface_face_measurement(
            shape["_raw_verts"],
            shape["_triangles"],
            live_pos,
            live_rot,
            live_seed_world,
            live_cylinder_world,
        )
        live_clear_raw_mapping = {
            **_classify_moving_patch_faces(projected["tied_face_indices"]),
            "minimum_surface_residual_m": projected["minimum_surface_residual_m"],
            "selected_raw_face_index": projected["selected_face_index"],
            "inherited_0p5mm_fidelity_pass": projected[
                "minimum_surface_residual_m"
            ]
            <= FIDELITY_TOL_MM / 1000.0,
        }

    live_q_overlap = float(live_contact_overlap["q5_float32_rad"])
    live_overlap_guard = _set_state_only(inner, timeline, live_q_overlap)
    live_overlap_contacts = _overlap_contact_surface_audit(
        inner,
        topology_parts["gripper_link"],
        "_geometry_topology_surface_authority",
        shape,
        "live",
    )
    _set_state_only(inner, timeline, live_q_clear)
    live_competitor_exclusion = _live_first_contact_competitor_exclusion(
        inner,
        topology_parts["gripper_link"],
        shape,
        live_q_clear,
        live_q_overlap,
    )
    _set_state_only(inner, timeline, q_clear)
    raw_live_normal_dot = (
        None
        if live_normal_world is None
        else float(np.dot(normal_world, live_normal_world))
    )
    pinch_checks["raw_live_inward_normals_same_hemisphere"] = bool(
        raw_live_normal_dot is not None and raw_live_normal_dot > 0.0
    )
    endpoint_classifications = {
        "raw_clear": raw_clear_identity["classification"],
        "raw_overlap_all_contacts": raw_overlap_contacts[
            "raw_surface_consensus_classification"
        ],
        "live_clear_projected_to_raw": live_clear_raw_mapping["classification"],
        "live_overlap_all_contacts_projected_to_raw": live_overlap_contacts[
            "raw_surface_consensus_classification"
        ],
    }
    endpoint_values = list(endpoint_classifications.values())
    contact_identity_unambiguous = bool(
        all(value is not None for value in endpoint_values)
        and len(set(endpoint_values)) == 1
        and raw_clear_identity["identity_unambiguous"]
        and live_clear_raw_mapping["identity_unambiguous"]
        and raw_overlap_contacts["pass"]
        and live_overlap_contacts["pass"]
    )
    contact_patch = endpoint_values[0] if contact_identity_unambiguous else None
    inner_contact = contact_patch == "frozen_distal_inner_minus_local_y"
    endpoint_feature_names = {
        raw_contact_clear["raw"]["gripper_link"]["cylinder_feature"]["feature"],
        raw_contact_overlap["raw"]["gripper_link"]["cylinder_feature"]["feature"],
        live_contact_clear["live"]["gripper_link"]["cylinder_feature"]["feature"],
        live_contact_overlap["live"]["gripper_link"]["cylinder_feature"]["feature"],
    }
    all_overlap_contact_feature_names = set(
        raw_overlap_contacts["all_contact_cylinder_features"]
    ) | set(live_overlap_contacts["all_contact_cylinder_features"])
    all_first_contact_feature_names = (
        endpoint_feature_names | all_overlap_contact_feature_names
    )
    barrel_eligibility_candidate = bool(
        inner_contact
        and all_first_contact_feature_names == {"barrel_interior"}
        and raw_overlap_contacts["all_contacts_on_barrel_interior"]
        and live_overlap_contacts["all_contacts_on_barrel_interior"]
    )
    live_inner_normal_contract = bool(
        inner_contact
        and (
            live_unflipped_normal_dot is not None
            and live_unflipped_normal_dot > 0.0
            and live_overlap_contacts[
                "all_inner_unflipped_normals_same_hemisphere"
            ]
        )
    )
    competitor_contract = bool(
        barrel_eligibility_candidate
        and raw_competitor_exclusion["pass"]
        and live_competitor_exclusion["pass"]
    )
    pinch_checks.update(
        {
            "first_contact_all_endpoints_on_frozen_inner_patch": inner_contact,
            "paired_outer_patch_negative_control_rejected": contact_patch
            != "paired_outer_plus_local_y_negative_control",
            "live_unflipped_inner_normals_same_hemisphere": bool(
                inner_contact and live_inner_normal_contract
            ),
            "noninner_and_cap_competitors_excluded_over_full_bracket": bool(
                competitor_contract
            ),
        }
    )
    checks = {
        "q_clear_state_guard": guard["pass"],
        "seed_residual_le_0p01mm": vertex_first["minimum_distance_m"]
        <= BINDING_RESIDUAL_MAX_M,
        "vertex_tied_faces_one_component": len(vertex_first["tied_component_roots"]) == 1,
        "edge_tied_faces_one_component": len(edge_first["tied_component_roots"]) == 1,
        "vertex_repeat_digest_exact": vertex_first["component_digest"]
        == vertex_repeat["component_digest"],
        "edge_repeat_digest_exact": edge_first["component_digest"]
        == edge_repeat["component_digest"],
        "vertex_edge_face_set_exact": np.array_equal(vertex_faces, edge_faces),
        "component_nonempty": len(vertex_first["component_vertices"]) >= 4
        and len(vertex_first["component_triangles"]) >= 4,
        "owner_gripper_link": shape["body"] == "gripper_link"
        and shape["owner_body_path"] == d334.BODY_PATHS["gripper_link"],
        "q5_child_gripper_link": joint.get("child") == "gripper_link",
        "link5_parent_negative_control_rejected": joint.get("parent") == "link5",
        "surface_normal_finite": bool(np.isfinite(normal_world).all()),
        "frozen_inner_outer_patch_definition_exact": patch_definition["pass"],
        "authored_inner_outer_patch_identity_exact": authored_patch_identity["pass"],
        "runtime_joint_axis_and_pivot_contract": runtime_joint["pass"],
        "raw_overlap_state_guard": raw_overlap_guard["pass"],
        "raw_overlap_all_contacts_audited": raw_overlap_contacts["pass"],
        "live_clear_state_guard": live_guard["pass"],
        "live_selected_surface_measured": live_surface["pass"],
        "live_clear_maps_to_raw_within_inherited_0p5mm": bool(
            live_clear_raw_mapping.get("minimum_surface_residual_m") is not None
            and live_clear_raw_mapping["minimum_surface_residual_m"]
            <= FIDELITY_TOL_MM / 1000.0
        ),
        "live_overlap_state_guard": live_overlap_guard["pass"],
        "live_overlap_all_contacts_audited": live_overlap_contacts["pass"],
        "raw_live_clear_overlap_surface_identity_consensus": contact_identity_unambiguous,
        "inner_contact_live_unflipped_normal_contract": live_inner_normal_contract,
        "positive_inner_barrel_first_contact_competitor_exclusion_contract": (
            competitor_contract
        ),
    }
    binding = {
        "artifact": "D351_MOVING_JAW_ACTUAL_CONTACT_SURFACE_BINDING_V1",
        "case": CASE,
        "semantics": (
            "exact authored moving-surface identity across raw/live clear and all "
            "non-saturated overlap contacts, plus bracket competitor exclusion"
        ),
        "q5_clear_float32_rad": q_clear,
        "q5_overlap_float32_rad": raw_q_overlap,
        "seed_world_m": seed_world.tolist(),
        "seed_local_m": seed_local.tolist(),
        "cylinder_witness_world_m": cylinder_witness.tolist(),
        "vertex_connected": _public_binding(vertex_first),
        "edge_connected": _public_binding(edge_first),
        "open_nearest_component_crosscheck": {
            "authority": False,
            "seed_world_m": open_seed_world.tolist(),
            "component_digest": open_binding["component_digest"],
            "same_as_first_contact_component": open_binding["component_digest"]
            == vertex_first["component_digest"],
        },
        "seed_triangle": {
            "face_index": seed_face,
            "triangle_local_m": triangle_local.tolist(),
            "oriented_normal_local": normal_local.tolist(),
            "oriented_normal_world": normal_world.tolist(),
            "normal_to_gap_angle_deg": d350._angle_deg(normal_world, gap),
        },
        "contact_patch_classification": {
            "classification": contact_patch,
            "endpoint_classifications": endpoint_classifications,
            "identity_unambiguous": contact_identity_unambiguous,
            "raw_clear": raw_clear_identity,
            "raw_overlap_all_contacts": raw_overlap_contacts,
            "live_clear_projected_to_raw": live_clear_raw_mapping,
            "live_overlap_all_contacts_projected_to_raw": live_overlap_contacts,
            "scientific_role": (
                "only a positive inner-patch plus barrel certificate can reach the "
                "geometry decision; non-inner/non-barrel evidence lacks a symmetric "
                "first-feature exclusion certificate and is therefore FAIL_STOP"
            ),
        },
        "first_contact_competitor_exclusion": {
            "raw": raw_competitor_exclusion,
            "live": live_competitor_exclusion,
            "positive_certificate_required": True,
            "negative_endpoint_policy": (
                "without a symmetric first-feature exclusion certificate, a "
                "non-inner or non-barrel endpoint is FAIL_STOP, never REPAIR"
            ),
            "pass": competitor_contract,
        },
        "barrel_eligibility_candidate_before_competitor_gate": barrel_eligibility_candidate,
        "endpoint_cylinder_feature_names": sorted(endpoint_feature_names),
        "all_overlap_contact_cylinder_feature_names": sorted(
            all_overlap_contact_feature_names
        ),
        "all_first_contact_cylinder_feature_names": sorted(
            all_first_contact_feature_names
        ),
        "frozen_patch_definition": patch_definition,
        "authored_patch_identity": authored_patch_identity,
        "pinch_facing_geometry": {
            "fixed_surface_world_m": fixed_seed.tolist(),
            "moving_surface_world_m": seed_world.tolist(),
            "fixed_inward_normal_world": fixed_normal.tolist(),
            "moving_inward_normal_world": normal_world.tolist(),
            "q5_decrease_velocity_world_m_per_rad": close_velocity_per_positive_rad.tolist(),
            "fixed_moving_normal_dot": fixed_moving_normal_dot,
            "fixed_normal_toward_moving_dot_m": fixed_normal_toward_moving_dot,
            "moving_normal_toward_fixed_dot_m": moving_normal_toward_fixed_dot,
            "raw_live_inward_normal_dot": raw_live_normal_dot,
            "close_velocity_toward_fixed_dot_m2_per_rad": close_toward_fixed_dot,
            "close_velocity_along_moving_normal_dot_m_per_rad": close_along_moving_normal_dot,
            "contact_chord_world_m": chord.tolist(),
            "cylinder_center_projection_parameter": center_projection_t,
            "chord_to_cylinder_axis_xy_miss_mm": chord_centerline_miss_mm,
            "fixed_moving_height_difference_mm": float(
                abs(fixed_seed[2] - seed_world[2]) * 1000.0
            ),
            "opposite_center_sides_xy_dot_m2": opposite_center_sides_dot_m2,
            "fixed_cylinder_witness_world_m": fixed_cylinder_witness.tolist(),
            "moving_cylinder_witness_world_m": cylinder_witness.tolist(),
            "cylinder_support_opposition_xy_dot_m2": cylinder_support_opposition_dot_m2,
            "cylinder_support_midpoint_world_m": support_midpoint.tolist(),
            "cylinder_support_midpoint_axis_xy_residual_mm": float(
                np.linalg.norm(support_midpoint[:2] - center[:2]) * 1000.0
            ),
            "fixed_cylinder_witness_height_from_center_mm": float(
                (fixed_cylinder_witness[2] - center[2]) * 1000.0
            ),
            "moving_cylinder_witness_height_from_center_mm": float(
                (cylinder_witness[2] - center[2]) * 1000.0
            ),
            "cylinder_witness_height_delta_mm": float(
                (cylinder_witness[2] - fixed_cylinder_witness[2]) * 1000.0
            ),
            "threshold_semantics": "only exact sign/order predicates gate; angle, height, and axis-miss values are measurement-only",
            "checks": pinch_checks,
            "pass": all(pinch_checks.values()),
        },
        "live_contact_surface_clear": {
            "selected_callback_surface": live_surface,
            "raw_surface_mapping": live_clear_raw_mapping,
            "unflipped_normal_dot_frozen_inner": live_unflipped_normal_dot,
        },
        "runtime_joint_geometry": runtime_joint,
        "joint_semantics": joint,
        "checks": checks,
        "pass": all(checks.values()),
    }
    inner_patch_vertices, inner_patch_triangles = _extract_face_patch(
        shape["_raw_verts"], shape["_triangles"], MOVING_INNER_PATCH_FACE_IDS
    )
    display = {
        "component_vertices": np.asarray(vertex_first["component_vertices"], dtype=np.float64),
        "component_triangles": np.asarray(vertex_first["component_triangles"], dtype=np.int64),
        "gripper_pos": gripper_pos,
        "gripper_rot": gripper_rot,
        "seed_world": seed_world,
        "cylinder_witness": cylinder_witness,
        "normal_world": normal_world,
        "fixed_seed_world": fixed_seed,
        "fixed_normal_world": fixed_normal,
        "joint_origin_world": joint_origin,
        "joint_axis_world": joint_axis,
        "close_velocity_world": close_velocity_per_positive_rad,
        "contact_patch_classification": contact_patch,
        "inner_patch_vertices": inner_patch_vertices,
        "inner_patch_triangles": inner_patch_triangles,
    }
    return binding, display


def _fallback_moving_display(
    inner: Any,
    timeline: Any,
    raw_shapes: list[dict[str, Any]],
    open_row: dict[str, Any],
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    guard = _set_state_only(inner, timeline, float(Q5_OPEN_F32))
    shape = next(row for row in raw_shapes if row["body"] == "gripper_link")
    pos, quat = d334._body_pose_w(inner, "gripper_link")
    rot = _quat_to_rot(quat)
    seed = np.asarray(
        open_row["raw"]["gripper_link"]["witness_endpoint_geometry_m"],
        dtype=np.float64,
    )
    cylinder = np.asarray(
        open_row["raw"]["gripper_link"]["witness_endpoint_cylinder_m"],
        dtype=np.float64,
    )
    surface = _surface_face_measurement(
        shape["_raw_verts"],
        shape["_triangles"],
        pos,
        rot,
        seed,
        cylinder,
    )
    fixed_measurement = _json(D350_MEASUREMENT)
    fixed_seed = np.asarray(
        fixed_measurement["actual_surface"]["current_raw_witness_world_m"],
        dtype=np.float64,
    )
    fixed_normal = np.asarray(
        fixed_measurement["actual_surface"]["oriented_surface_normal_world"],
        dtype=np.float64,
    )
    joint = _runtime_joint_geometry(inner)
    inner_patch_vertices, inner_patch_triangles = _extract_face_patch(
        shape["_raw_verts"], shape["_triangles"], MOVING_INNER_PATCH_FACE_IDS
    )
    display = {
        "component_vertices": np.asarray(shape["_raw_verts"], dtype=np.float64),
        "component_triangles": np.asarray(shape["_triangles"], dtype=np.int64),
        "gripper_pos": pos,
        "gripper_rot": rot,
        "seed_world": seed,
        "cylinder_witness": cylinder,
        "normal_world": np.asarray(surface["oriented_normal_world"], dtype=np.float64),
        "fixed_seed_world": fixed_seed,
        "fixed_normal_world": fixed_normal,
        "joint_origin_world": np.asarray(joint["origin_world_m"], dtype=np.float64),
        "joint_axis_world": np.asarray(joint["axis_world"], dtype=np.float64),
        "close_velocity_world": np.zeros(3, dtype=np.float64),
        "contact_patch_classification": "unresolved_open_fallback",
        "inner_patch_vertices": inner_patch_vertices,
        "inner_patch_triangles": inner_patch_triangles,
    }
    binding = {
        "artifact": "D351_MOVING_JAW_BINDING_NOT_RUN_WITH_VISUAL_FALLBACK",
        "reason": reason,
        "fallback_semantics": "OPEN full raw gripper mesh only; no first-contact binding authority",
        "fallback_state_guard": guard,
        "fallback_surface": surface,
        "pass": False,
    }
    return binding, display


def _fixed_component_digest_reproduction(
    inner: Any, timeline: Any, raw_shapes: list[dict[str, Any]]
) -> dict[str, Any]:
    _set_state_only(inner, timeline, float(Q5_OPEN_F32))
    shape = next(row for row in raw_shapes if row["body"] == "link5")
    measurement = _json(D350_MEASUREMENT)
    seed_world = np.asarray(
        measurement["actual_surface"]["historical_seed_world_m"], dtype=np.float64
    )
    pos, quat = d334._body_pose_w(inner, "link5")
    seed_local = _quat_to_rot(quat).T @ (seed_world - pos)
    bound = d350._bind_component(
        shape["_raw_verts"], shape["_triangles"], seed_local, reverse_order=False
    )
    expected = measurement["fixed_jaw_component"]["component_digest"]
    return {
        "expected_digest": expected,
        "observed_digest": bound["component_digest"],
        "seed_residual_m": bound["minimum_distance_m"],
        "pass": bool(
            bound["component_digest"] == expected
            and bound["minimum_distance_m"] <= BINDING_RESIDUAL_MAX_M
        ),
    }


def _write_sweep_csv(rows: list[dict[str, Any]]) -> None:
    columns = [
        "evaluation_index",
        "q5_float32_rad",
        "raw_gripper_mm",
        "live_gripper_mm",
        "raw_live_delta_mm",
        "raw_feature",
        "live_feature",
        "raw_collision",
        "live_collision",
        "raw_link5_mm",
        "live_link5_mm",
        "table_clearance_mm",
        "counter",
        "state_guard_pass",
    ]
    if SWEEP_CSV_PATH.exists():
        raise RuntimeError(f"refusing to overwrite {SWEEP_CSV_PATH}")
    SWEEP_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SWEEP_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "evaluation_index": row["evaluation_index"],
                    "q5_float32_rad": row["q5_float32_rad"],
                    "raw_gripper_mm": row["raw"]["gripper_link"]["exact_signed_distance_mm"],
                    "live_gripper_mm": row["live"]["gripper_link"]["exact_signed_distance_mm"],
                    "raw_live_delta_mm": row["raw_live_gripper_absolute_delta_mm"],
                    "raw_feature": row["raw"]["gripper_link"]["cylinder_feature"]["feature"],
                    "live_feature": row["live"]["gripper_link"]["cylinder_feature"]["feature"],
                    "raw_collision": row["raw"]["gripper_link"]["is_collision"],
                    "live_collision": row["live"]["gripper_link"]["is_collision"],
                    "raw_link5_mm": row["raw"]["link5"]["exact_signed_distance_mm"],
                    "live_link5_mm": row["live"]["link5"]["exact_signed_distance_mm"],
                    "table_clearance_mm": row["diagnostics"]["gripper_table_clearance_mm"],
                    "counter": row["simulation_counter"],
                    "state_guard_pass": row["state_guard"]["pass"],
                }
            )


def _classify_measurement(
    rows: list[dict[str, Any]],
    raw_order: dict[str, Any],
    live_order: dict[str, Any],
    table_corridor: dict[str, Any],
    binding: dict[str, Any],
    fixed_reproduction: dict[str, Any],
    raw_radius_m: float,
    live_radius_m: float,
) -> dict[str, Any]:
    historical = _json(D350_MEASUREMENT)["distances"]
    open_row = max(rows, key=lambda row: row["q5_float32_rad"])
    closed_row = min(rows, key=lambda row: row["q5_float32_rad"])
    anchor_checks: dict[str, bool] = {}
    for body in d334.BODY_LABELS:
        anchor_checks[f"open_{body}_raw_exact"] = abs(
            float(open_row["raw"][body]["exact_signed_distance_mm"])
            - float(historical[body]["raw_signed_distance_mm"])
        ) <= ANCHOR_TOL_MM
        anchor_checks[f"open_{body}_live_exact"] = abs(
            float(open_row["live"][body]["exact_signed_distance_mm"])
            - float(historical[body]["live_signed_distance_mm"])
        ) <= ANCHOR_TOL_MM
    closed_raw = closed_row["raw"]["gripper_link"]["exact_signed_distance_mm"]
    anchor_checks["closed_raw_matches_d337"] = bool(
        closed_raw is not None
        and abs(float(closed_raw) - (-6.460556421875954)) <= CLOSED_ANCHOR_TOL_MM
    )
    raw_bracket = raw_order.get("first_contact_bracket")
    live_bracket = live_order.get("first_contact_bracket")
    raw_clear = None if raw_bracket is None else next(
        row for row in rows if _q5_key(row["q5_float32_rad"]) == _q5_key(raw_bracket["q_clear_float32_rad"])
    )
    raw_overlap = None if raw_bracket is None else next(
        row for row in rows if _q5_key(row["q5_float32_rad"]) == _q5_key(raw_bracket["q_overlap_float32_rad"])
    )
    live_clear = None if live_bracket is None else next(
        row for row in rows if _q5_key(row["q5_float32_rad"]) == _q5_key(live_bracket["q_clear_float32_rad"])
    )
    live_overlap = None if live_bracket is None else next(
        row for row in rows if _q5_key(row["q5_float32_rad"]) == _q5_key(live_bracket["q_overlap_float32_rad"])
    )
    raw_features = {
        "clear": None if raw_clear is None else raw_clear["raw"]["gripper_link"]["cylinder_feature"],
        "overlap": None if raw_overlap is None else raw_overlap["raw"]["gripper_link"]["cylinder_feature"],
    }
    live_features = {
        "clear": None if live_clear is None else live_clear["live"]["gripper_link"]["cylinder_feature"],
        "overlap": None if live_overlap is None else live_overlap["live"]["gripper_link"]["cylinder_feature"],
    }
    raw_feature_consensus = bool(
        all(value is not None for value in raw_features.values())
        and len({value["feature"] for value in raw_features.values()}) == 1
    )
    live_feature_consensus = bool(
        all(value is not None for value in live_features.values())
        and len({value["feature"] for value in live_features.values()}) == 1
    )
    raw_link_values = [float(row["raw"]["link5"]["exact_signed_distance_mm"]) for row in rows]
    live_link_values = [float(row["live"]["link5"]["exact_signed_distance_mm"]) for row in rows]
    both_clear_deltas = [
        float(row["raw_live_gripper_absolute_delta_mm"])
        for row in rows
        if row["raw_live_gripper_absolute_delta_mm"] is not None
        and not row["raw"]["gripper_link"]["is_collision"]
        and not row["live"]["gripper_link"]["is_collision"]
    ]
    row_by_key = {_q5_key(row["q5_float32_rad"]): row for row in rows}
    contact_keys: set[str] = set()
    for bracket in (raw_bracket, live_bracket):
        if bracket is not None:
            contact_keys.update(
                {
                    _q5_key(bracket["q_clear_float32_rad"]),
                    _q5_key(bracket["q_overlap_float32_rad"]),
                }
            )
    contact_rows = [row_by_key[key] for key in sorted(contact_keys) if key in row_by_key]
    contact_endpoint_contracts = [
        {
            "q5_float32_rad": row["q5_float32_rad"],
            "raw": _endpoint_contract(row, "raw"),
            "live": _endpoint_contract(row, "live"),
            "raw_live_absolute_delta_mm": row["raw_live_gripper_absolute_delta_mm"],
        }
        for row in contact_rows
    ]
    contact_endpoint_deltas = [
        float(row["raw_live_gripper_absolute_delta_mm"])
        for row in contact_rows
        if row["raw_live_gripper_absolute_delta_mm"] is not None
    ]
    contact_midpoint_delta_rad = (
        None
        if raw_bracket is None or live_bracket is None
        else abs(
            0.5
            * (
                raw_bracket["q_clear_float32_rad"]
                + raw_bracket["q_overlap_float32_rad"]
            )
            - 0.5
            * (
                live_bracket["q_clear_float32_rad"]
                + live_bracket["q_overlap_float32_rad"]
            )
        )
    )
    contact_worst_case_delta_rad = (
        None
        if raw_bracket is None or live_bracket is None
        else max(
            abs(
                raw_bracket["q_clear_float32_rad"]
                - live_bracket["q_overlap_float32_rad"]
            ),
            abs(
                raw_bracket["q_overlap_float32_rad"]
                - live_bracket["q_clear_float32_rad"]
            ),
        )
    )
    contact_surface_travel_delta_mm = (
        None
        if contact_worst_case_delta_rad is None
        else _interval_bound_mm(
            max(raw_radius_m, live_radius_m),
            contact_worst_case_delta_rad,
            0.0,
        )
    )
    full_sweep_table_min = min(
        float(row["diagnostics"]["gripper_table_clearance_mm"]) for row in rows
    )
    q_table_stop = table_corridor.get("q_stop_float32_rad")
    precontact_table_rows = [
        row
        for row in rows
        if q_table_stop is not None
        and float(row["q5_float32_rad"]) >= float(q_table_stop)
    ]
    table_min = (
        min(
            float(row["diagnostics"]["gripper_table_clearance_mm"])
            for row in precontact_table_rows
        )
        if precontact_table_rows
        else full_sweep_table_min
    )
    binding_pinch = binding.get("pinch_facing_geometry", {})
    contact_patch = binding.get("contact_patch_classification", {}).get("classification")
    open_rot = _quat_to_rot(
        open_row["runtime_joint_geometry"]["gripper_body_quaternion_wxyz"]
    )
    closed_rot = _quat_to_rot(
        closed_row["runtime_joint_geometry"]["gripper_body_quaternion_wxyz"]
    )
    delta_q = float(
        closed_row["q5_float32_rad"] - open_row["q5_float32_rad"]
    )
    expected_relative = np.asarray(
        [
            [math.cos(delta_q), -math.sin(delta_q), 0.0],
            [math.sin(delta_q), math.cos(delta_q), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    observed_relative = open_rot.T @ closed_rot
    q5_rotation_contract = {
        "delta_q_rad": delta_q,
        "max_rotation_matrix_absolute_error": float(
            np.max(np.abs(observed_relative - expected_relative))
        ),
        "pivot_position_delta_m": float(
            np.linalg.norm(
                np.asarray(
                    closed_row["runtime_joint_geometry"]["origin_world_m"],
                    dtype=np.float64,
                )
                - np.asarray(
                    open_row["runtime_joint_geometry"]["origin_world_m"],
                    dtype=np.float64,
                )
            )
        ),
    }
    q5_rotation_contract["pass"] = bool(
        q5_rotation_contract["max_rotation_matrix_absolute_error"]
        <= RUNTIME_KINEMATIC_TOL_M
        and q5_rotation_contract["pivot_position_delta_m"]
        <= RUNTIME_KINEMATIC_TOL_M
        and delta_q < 0.0
    )
    contract_checks = {
        **{
            key: value
            for key, value in anchor_checks.items()
            if key != "closed_raw_matches_d337"
        },
        "all_state_guards": all(row["state_guard"]["pass"] for row in rows),
        "all_counters_zero": all(row["simulation_counter"] == 0 for row in rows),
        "open_raw_live_clearance_meets_inherited_0p1mm": all(
            float(open_row[representation][body]["exact_signed_distance_mm"])
            >= CLEAR_GATE_MM
            for representation in ("raw", "live")
            for body in d334.BODY_LABELS
        ),
        "raw_contact_order_certified": raw_order["contact_order_certified"],
        "live_contact_order_certified": live_order["contact_order_certified"],
        "moving_surface_binding": binding["pass"],
        "moving_contact_surface_identity_unambiguous": binding.get(
            "contact_patch_classification", {}
        ).get("identity_unambiguous")
        is True,
        "fixed_component_digest_reproduced": fixed_reproduction["pass"],
        "all_runtime_joint_axis_pivot_contracts": all(
            row["runtime_joint_geometry"]["pass"] for row in rows
        ),
        "runtime_q5_decrease_rotation_sign_and_pivot_contract": q5_rotation_contract[
            "pass"
        ],
        "raw_clear_overlap_cylinder_feature_consensus": raw_feature_consensus,
        "live_clear_overlap_cylinder_feature_consensus": live_feature_consensus,
        "link5_raw_q5_invariant": max(raw_link_values) - min(raw_link_values) <= ANCHOR_TOL_MM,
        "link5_live_q5_invariant": max(live_link_values) - min(live_link_values) <= ANCHOR_TOL_MM,
        "contact_common_rows_all_exact_valid_signed_states": bool(
            contact_endpoint_contracts
        )
        and all(
            item[representation]["valid_for_signed_state"]
            for item in contact_endpoint_contracts
            for representation in ("raw", "live")
        ),
        "raw_live_contact_endpoint_signed_distance_agreement_le_0p5mm": bool(
            contact_endpoint_deltas
        )
        and max(contact_endpoint_deltas) <= FIDELITY_TOL_MM,
        "raw_live_first_contact_surface_travel_delta_le_0p5mm": bool(
            contact_surface_travel_delta_mm is not None
            and contact_surface_travel_delta_mm <= FIDELITY_TOL_MM
        ),
    }
    contract_pass = all(contract_checks.values())
    geometry_checks = {
        "raw_clear_overlap_contact_feature_barrel_interior": raw_feature_consensus
        and raw_features["clear"]["feature"] == "barrel_interior",
        "live_clear_overlap_contact_feature_barrel_interior": live_feature_consensus
        and live_features["clear"]["feature"] == "barrel_interior",
        "raw_first_contact_on_frozen_distal_inner_patch": contact_patch
        == "frozen_distal_inner_minus_local_y",
        "pinch_facing_exact_sign_and_order_contract": binding_pinch.get("pass") is True,
        "precontact_table_strict_clearance_continuously_certified": table_corridor.get(
            "precontact_table_clearance_certified"
        )
        is True,
    }
    if not contract_pass:
        verdict = VERDICT_ORDER if (
            not raw_order["contact_order_certified"]
            or not live_order["contact_order_certified"]
            or not raw_feature_consensus
            or not live_feature_consensus
        ) else (VERDICT_BINDING if not binding["pass"] else VERDICT_INPUT)
    elif all(geometry_checks.values()):
        verdict = VERDICT_ELIGIBLE
    else:
        verdict = VERDICT_REPAIR
    return {
        "anchor_checks": anchor_checks,
        "contract_checks": contract_checks,
        "contract_pass": contract_pass,
        "geometry_checks": geometry_checks,
        "raw_first_contact_feature": {
            "endpoints": raw_features,
            "consensus": raw_feature_consensus,
        },
        "live_first_contact_feature": {
            "endpoints": live_features,
            "consensus": live_feature_consensus,
        },
        "moving_contact_patch_classification": contact_patch,
        "pinch_facing_geometry": binding_pinch,
        "raw_live_contact_q5_delta_rad": contact_midpoint_delta_rad,
        "raw_live_contact_q5_worst_case_delta_rad": contact_worst_case_delta_rad,
        "raw_live_contact_surface_travel_delta_mm": contact_surface_travel_delta_mm,
        "contact_common_endpoint_comparisons": contact_endpoint_contracts,
        "max_raw_live_contact_endpoint_delta_mm": max(contact_endpoint_deltas)
        if contact_endpoint_deltas
        else None,
        "max_raw_live_precontact_delta_mm": max(both_clear_deltas) if both_clear_deltas else None,
        "min_gripper_table_clearance_mm": table_min,
        "full_sweep_min_gripper_table_clearance_diagnostic_mm": full_sweep_table_min,
        "precontact_table_corridor": table_corridor,
        "runtime_q5_rotation_contract": q5_rotation_contract,
        "closed_raw_d337_anchor_diagnostic_only": {
            "observed_mm": closed_raw,
            "reference_mm": -6.460556421875954,
            "absolute_delta_mm": None
            if closed_raw is None
            else abs(float(closed_raw) - (-6.460556421875954)),
            "within_0p05mm": anchor_checks["closed_raw_matches_d337"],
            "verdict_authority": False,
        },
        "raw_link5_q5_range_mm": max(raw_link_values) - min(raw_link_values),
        "live_link5_q5_range_mm": max(live_link_values) - min(live_link_values),
        "scientific_verdict": verdict,
        "verdict_semantics": "zero-step current pre-grasp closure geometry only; not simultaneous two-jaw contact, force closure, grasp, settle, or G0a",
    }


def _cylinder_mesh(segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    radius = d332.CYLINDER_RADIUS_M
    half = 0.5 * d332.CYLINDER_HEIGHT_M
    vertices: list[list[float]] = []
    for z_value in (-half, half):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append(
                [radius * math.cos(angle), radius * math.sin(angle), z_value]
            )
    vertices.extend([[0.0, 0.0, -half], [0.0, 0.0, half]])
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
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def _palette(body: str, index: int) -> list[int]:
    permuted = ((index * 37) % 64) / 63.0
    hue = 0.48 + 0.22 * permuted if body == "link5" else (0.99 + 0.16 * permuted) % 1.0
    saturation = 0.58 + 0.34 * (((index * 17) % 7) / 6.0)
    value = 0.72 + 0.26 * (((index * 29) % 5) / 4.0)
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return [int(round(channel * 255)) for channel in rgb] + [185]


def _world_vertices(vertices: Any, pos: Any, rot: Any) -> np.ndarray:
    return (
        np.asarray(rot, dtype=np.float64)
        @ np.asarray(vertices, dtype=np.float64).T
    ).T + np.asarray(pos, dtype=np.float64)


def _create_viewer_guides(
    inner: Any,
    topology_parts: dict[str, list[dict[str, Any]]],
    moving_display: dict[str, Any],
    raw_clear_row: dict[str, Any],
    live_clear_row: dict[str, Any],
) -> dict[str, Any]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = inner.scene.stage
    session = stage.GetSessionLayer()
    root_layer = stage.GetRootLayer()
    created: list[str] = []

    def colorize(gprim: Any, rgba: list[int]) -> None:
        gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
            [Gf.Vec3f(*(value / 255.0 for value in rgba[:3]))]
        )
        gprim.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set(
            [rgba[3] / 255.0]
        )

    def mesh_at(path: str, vertices: np.ndarray, triangles: np.ndarray, rgba: list[int]) -> None:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(
            [Gf.Vec3f(*[float(value) for value in row]) for row in vertices]
        )
        mesh.CreateFaceVertexCountsAttr([3] * int(len(triangles)))
        mesh.CreateFaceVertexIndicesAttr([int(value) for value in triangles.reshape(-1)])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
        colorize(mesh, rgba)
        created.append(path)

    def line(path: str, points: list[np.ndarray], color: tuple[float, float, float], width: float) -> None:
        curve = UsdGeom.BasisCurves.Define(stage, path)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateBasisAttr(UsdGeom.Tokens.bezier)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr(
            [Gf.Vec3f(*[float(value) for value in point]) for point in points]
        )
        curve.CreateWidthsAttr([width])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreatePurposeAttr(UsdGeom.Tokens.guide)
        curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
        created.append(path)

    def sphere(path: str, center: np.ndarray, radius: float, color: tuple[float, float, float]) -> None:
        item = UsdGeom.Sphere.Define(stage, path)
        item.CreateRadiusAttr(radius)
        item.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in center]))
        item.CreatePurposeAttr(UsdGeom.Tokens.guide)
        item.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
        created.append(path)

    with Usd.EditContext(stage, session):
        root = UsdGeom.Xform.Define(stage, GUIDE_ROOT)
        root.CreatePurposeAttr(UsdGeom.Tokens.guide)
        for body in d334.BODY_LABELS:
            pos, quat = d334._body_pose_w(inner, body)
            rot = _quat_to_rot(quat)
            for index, part in enumerate(topology_parts[body]):
                mesh_at(
                    f"{GUIDE_ROOT}/live_parts/{body}/part_{index:03d}",
                    _world_vertices(part["_vertices"], pos, rot),
                    np.asarray(part["_triangles"], dtype=np.int64),
                    _palette(body, index),
                )
        moving_world = _world_vertices(
            moving_display["component_vertices"],
            moving_display["gripper_pos"],
            moving_display["gripper_rot"],
        )
        mesh_at(
            f"{GUIDE_ROOT}/moving_jaw/raw_decision_component",
            moving_world,
            moving_display["component_triangles"],
            [60, 255, 120, 125],
        )
        mesh_at(
            f"{GUIDE_ROOT}/moving_jaw/frozen_distal_inner_patch",
            _world_vertices(
                moving_display["inner_patch_vertices"],
                moving_display["gripper_pos"],
                moving_display["gripper_rot"],
            ),
            moving_display["inner_patch_triangles"],
            [255, 45, 45, 235],
        )
        cyl_vertices, cyl_triangles = _cylinder_mesh()
        center = OBJECT_POS_F32.astype(np.float64)
        mesh_at(
            f"{GUIDE_ROOT}/target/cylinder",
            cyl_vertices + center,
            cyl_triangles,
            [255, 190, 20, 100],
        )
        raw_geom = np.asarray(
            raw_clear_row["raw"]["gripper_link"]["witness_endpoint_geometry_m"],
            dtype=np.float64,
        )
        raw_cyl = np.asarray(
            raw_clear_row["raw"]["gripper_link"]["witness_endpoint_cylinder_m"],
            dtype=np.float64,
        )
        live_geom = np.asarray(
            live_clear_row["live"]["gripper_link"]["witness_endpoint_geometry_m"],
            dtype=np.float64,
        )
        live_cyl = np.asarray(
            live_clear_row["live"]["gripper_link"]["witness_endpoint_cylinder_m"],
            dtype=np.float64,
        )
        line(f"{GUIDE_ROOT}/witness/raw", [raw_geom, raw_cyl], (1.0, 1.0, 0.0), 0.003)
        line(f"{GUIDE_ROOT}/witness/live", [live_geom, live_cyl], (1.0, 0.0, 1.0), 0.0024)
        sphere(f"{GUIDE_ROOT}/points/cylinder_center", center, 0.0035, (1.0, 0.8, 0.0))
        sphere(f"{GUIDE_ROOT}/points/raw_jaw", raw_geom, 0.0025, (0.2, 1.0, 0.2))
        sphere(f"{GUIDE_ROOT}/points/raw_cylinder", raw_cyl, 0.0025, (1.0, 1.0, 0.0))
        sphere(f"{GUIDE_ROOT}/points/live_jaw", live_geom, 0.0021, (1.0, 0.0, 1.0))
        fixed_seed = np.asarray(moving_display["fixed_seed_world"], dtype=np.float64)
        fixed_normal = np.asarray(moving_display["fixed_normal_world"], dtype=np.float64)
        moving_normal = np.asarray(moving_display["normal_world"], dtype=np.float64)
        joint_origin = np.asarray(moving_display["joint_origin_world"], dtype=np.float64)
        joint_axis = np.asarray(moving_display["joint_axis_world"], dtype=np.float64)
        line(
            f"{GUIDE_ROOT}/closure/fixed_to_moving_decision_chord",
            [raw_geom, fixed_seed],
            (0.15, 0.95, 1.0),
            0.0028,
        )
        line(
            f"{GUIDE_ROOT}/closure/moving_inward_normal",
            [raw_geom, raw_geom + moving_normal * 0.025],
            (0.2, 1.0, 0.25),
            0.0025,
        )
        line(
            f"{GUIDE_ROOT}/closure/fixed_inward_normal",
            [fixed_seed, fixed_seed + fixed_normal * 0.025],
            (0.1, 0.85, 1.0),
            0.0025,
        )
        line(
            f"{GUIDE_ROOT}/closure/q5_runtime_axis",
            [joint_origin - joint_axis * 0.035, joint_origin + joint_axis * 0.035],
            (1.0, 0.35, 0.05),
            0.0020,
        )
        line(
            f"{GUIDE_ROOT}/closure/cylinder_axis",
            [
                center - np.asarray([0.0, 0.0, d332.CYLINDER_HEIGHT_M * 0.75]),
                center + np.asarray([0.0, 0.0, d332.CYLINDER_HEIGHT_M * 0.75]),
            ],
            (1.0, 0.75, 0.05),
            0.0020,
        )
        sphere(f"{GUIDE_ROOT}/points/fixed_jaw", fixed_seed, 0.0025, (0.0, 0.8, 1.0))

    all_paths = [GUIDE_ROOT, *created]
    no_physics: dict[str, bool] = {}
    for path in created:
        prim = stage.GetPrimAtPath(path)
        no_physics[path] = bool(
            prim.IsValid()
            and not prim.HasAPI(UsdPhysics.CollisionAPI)
            and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and not prim.HasAPI(UsdPhysics.MassAPI)
        )
    checks = {
        "live_parts_64_plus_64": all(len(topology_parts[body]) == 64 for body in d334.BODY_LABELS),
        "all_guides_session_only": all(session.GetPrimAtPath(path) is not None for path in all_paths),
        "no_guides_in_root_layer": all(root_layer.GetPrimAtPath(path) is None for path in all_paths),
        "all_display_prims_without_physics_api": all(no_physics.values()),
    }
    return {
        "artifact": "D351_VIEWER_OVERLAY_CONTRACT_V1",
        "guide_root": GUIDE_ROOT,
        "created_prim_count": len(created),
        "created_paths": created,
        "actual_physx_collider_role": "separate debug-render capture",
        "colored_mesh_role": "display-only callback-topology copy",
        "checks": checks,
        "pass": all(checks.values()),
    }


def _set_guide_visibility(inner: Any, visible: bool) -> None:
    from pxr import Usd, UsdGeom

    stage = inner.scene.stage
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(GUIDE_ROOT))
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def _guide_layer_contract(inner: Any) -> dict[str, Any]:
    stage = inner.scene.stage
    session = stage.GetSessionLayer()
    root = stage.GetRootLayer()
    return {
        "guide_root_in_session_layer": session.GetPrimAtPath(GUIDE_ROOT) is not None,
        "guide_root_absent_from_root_layer": root.GetPrimAtPath(GUIDE_ROOT) is None,
    }


def _pump_frames(simulation_app: Any, inner: Any, timeline: Any, count: int) -> int:
    interventions = 0
    for _ in range(count):
        inner.sim.set_setting("/app/player/playSimulations", False)
        if timeline.is_playing():
            timeline.pause()
            interventions += 1
        simulation_app.update()
        inner.sim.set_setting("/app/player/playSimulations", False)
        if timeline.is_playing():
            timeline.pause()
            interventions += 1
    return interventions


def _capture_viewport(
    path: Path, simulation_app: Any, inner: Any, timeline: Any
) -> dict[str, Any]:
    import omni.kit.viewport.utility as viewport_utility

    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        return {"ok": False, "error": "no active viewport"}
    capture = viewport_utility.capture_viewport_to_file(viewport, str(path))
    task = simulation_app.run_coroutine(
        capture.wait_for_result(completion_frames=5), run_until_complete=False
    )
    deadline = time.monotonic() + 30.0
    interventions = 0
    while not task.done() and time.monotonic() < deadline and simulation_app.is_running():
        interventions += _pump_frames(simulation_app, inner, timeline, 1)
    if not task.done():
        task.cancel()
        return {"ok": False, "error": "viewport capture timeout"}
    result = task.result()
    _pump_frames(simulation_app, inner, timeline, 3)
    return {
        "capture_result": bool(result),
        "path": _rel(path),
        "exists_before_app_close_informational": path.is_file(),
        "timeline_interventions": interventions,
        "postclose_validation_pending": True,
        "ok": bool(result),
    }


def _run_viewer(
    args: argparse.Namespace,
    simulation_app: Any,
    inner: Any,
    timeline: Any,
    q_clear: float,
    display_pose_role: str,
) -> dict[str, Any]:
    import carb

    settings = carb.settings.get_settings()
    previous_physx = settings.get(PHYSX_COLLIDER_SETTING)
    previous_guide = settings.get(GUIDE_PURPOSE_SETTING)
    counter_before = int(inner._sim_step_counter)
    time_before = float(timeline.get_current_time())
    simulation_clock_before = _simulation_clock(inner)
    captures: dict[str, Any] = {}
    interventions = 0
    hold_start: float | None = None
    hold_updates = 0
    app_alive_after_hold = False
    try:
        settings.set(GUIDE_PURPOSE_SETTING, True)
        settings.set(PHYSX_COLLIDER_SETTING, 2)
        _set_guide_visibility(inner, False)
        open_guard = _set_state_only(inner, timeline, float(Q5_OPEN_F32))
        inner.sim.set_camera_view([0.49, -0.32, 0.28], [0.285, 0.0, 0.055])
        interventions += _pump_frames(simulation_app, inner, timeline, 12)
        captures["open_physx"] = _capture_viewport(
            VIEWER_PNGS["open_physx"], simulation_app, inner, timeline
        )

        decision_guard = _set_state_only(inner, timeline, q_clear)
        inner.sim.set_camera_view([0.49, -0.32, 0.28], [0.285, 0.0, 0.055])
        interventions += _pump_frames(simulation_app, inner, timeline, 12)
        captures["decision_physx"] = _capture_viewport(
            VIEWER_PNGS["decision_physx"], simulation_app, inner, timeline
        )

        settings.set(PHYSX_COLLIDER_SETTING, 0)
        _set_guide_visibility(inner, True)
        inner.sim.set_camera_view([0.49, -0.32, 0.28], [0.285, 0.0, 0.055])
        interventions += _pump_frames(simulation_app, inner, timeline, 10)
        captures["decision_colored"] = _capture_viewport(
            VIEWER_PNGS["decision_colored"], simulation_app, inner, timeline
        )
        inner.sim.set_camera_view([0.285, -0.42, 0.09], [0.285, 0.0, 0.055])
        interventions += _pump_frames(simulation_app, inner, timeline, 10)
        captures["decision_side"] = _capture_viewport(
            VIEWER_PNGS["decision_side"], simulation_app, inner, timeline
        )
        inner.sim.set_camera_view([0.49, -0.32, 0.28], [0.285, 0.0, 0.055])
        interventions += _pump_frames(simulation_app, inner, timeline, 8)

        hold_start = time.monotonic()
        while (
            simulation_app.is_running()
            and time.monotonic() - hold_start < float(args.viewer_hold_seconds)
        ):
            interventions += _pump_frames(simulation_app, inner, timeline, 1)
            hold_updates += 1
        app_alive_after_hold = bool(simulation_app.is_running())
        final_guard = _state_guard(
            inner,
            q_clear,
            counter_before,
            time_before,
            timeline,
            simulation_clock_before,
        )
    finally:
        if previous_physx is None:
            settings.destroy_item(PHYSX_COLLIDER_SETTING)
        else:
            settings.set(PHYSX_COLLIDER_SETTING, previous_physx)
        if previous_guide is None:
            settings.destroy_item(GUIDE_PURPOSE_SETTING)
        else:
            settings.set(GUIDE_PURPOSE_SETTING, previous_guide)
        _set_guide_visibility(inner, True)
    hold_actual_seconds = (
        0.0 if hold_start is None else time.monotonic() - hold_start
    )
    restored_physx = settings.get(PHYSX_COLLIDER_SETTING)
    restored_guide = settings.get(GUIDE_PURPOSE_SETTING)
    layer_contract = _guide_layer_contract(inner)
    checks = {
        "all_capture_tokens": len(captures) == len(VIEWER_PNGS)
        and all(row.get("ok") for row in captures.values()),
        "open_guard": open_guard["pass"],
        "decision_guard": decision_guard["pass"],
        "final_decision_guard": final_guard["pass"],
        "counter_zero_unchanged": counter_before == int(inner._sim_step_counter) == 0,
        "timeline_paused": not timeline.is_playing(),
        "timeline_time_unchanged": float(timeline.get_current_time()) == time_before,
        "simulation_context_clock_unchanged": _simulation_clock(inner)
        == simulation_clock_before,
        "requested_positive_viewer_hold": float(args.viewer_hold_seconds) > 0.0,
        "viewer_hold_duration_satisfied": hold_actual_seconds
        >= float(args.viewer_hold_seconds),
        "viewer_hold_updated_ui": hold_updates > 0,
        "viewer_app_alive_after_hold": app_alive_after_hold,
        "persistent_physx_setting_restored_exact": restored_physx == previous_physx,
        "persistent_guide_setting_restored_exact": restored_guide == previous_guide,
        "guide_root_still_session_only_after_viewer": all(layer_contract.values()),
    }
    return {
        "artifact": "D351_VIEWER_CAPTURE_CONTRACT_V1",
        "captures": captures,
        "q5_open_rad": float(Q5_OPEN_F32),
        "q5_decision_or_fallback_rad": q_clear,
        "display_pose_role": display_pose_role,
        "interactive_hold": {
            "requested_seconds": float(args.viewer_hold_seconds),
            "actual_seconds": hold_actual_seconds,
            "ui_updates": hold_updates,
            "app_alive_after_hold": app_alive_after_hold,
        },
        "setting_restore": {
            "previous_physx": previous_physx,
            "observed_physx": restored_physx,
            "previous_guide": previous_guide,
            "observed_guide": restored_guide,
        },
        "post_viewer_layer_contract": layer_contract,
        "timeline_interventions": interventions,
        "postclose_png_validation_pending": True,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _rrd_contract() -> tuple[list[str], dict[str, list[str]]]:
    entities = {"metadata/run", "coordinate_frames/world"}
    components: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "coordinate_frames/world": [
            "Transform3D:child_frame",
            "Transform3D:parent_frame",
            "Transform3D:quaternion",
            "Transform3D:translation",
        ],
    }
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    mesh_paths = [
        *(
            f"geometry/live_parts/{body}/part_{index:03d}"
            for body in d334.BODY_LABELS
            for index in range(64)
        ),
        "geometry/moving_jaw/raw_decision_component",
        "geometry/moving_jaw/frozen_distal_inner_patch",
        "geometry/target/cylinder",
    ]
    for path in mesh_paths:
        metadata = f"metadata/meshes/{path.replace('/', '__')}"
        entities.update({path, metadata})
        components[path] = mesh_components
        components[metadata] = ["TextDocument:text"]
    for name in ("raw_witness", "live_witness"):
        path = f"geometry/dynamic_points/{name}"
        entities.add(path)
        components[path] = [
            "CoordinateFrame:frame",
            "Points3D:colors",
            "Points3D:labels",
            "Points3D:positions",
            "Points3D:radii",
        ]
    for name in ("gripper_live_surface", "moving_inner_patch"):
        path = f"geometry/dynamic_points/{name}"
        entities.add(path)
        components[path] = [
            "CoordinateFrame:frame",
            "Points3D:colors",
            "Points3D:positions",
            "Points3D:radii",
        ]
    path = "geometry/dynamic_points/gripper_origin"
    entities.add(path)
    components[path] = [
        "CoordinateFrame:frame",
        "Points3D:colors",
        "Points3D:labels",
        "Points3D:positions",
        "Points3D:radii",
    ]
    for name in ("raw_gap", "live_gap", "q5_axis", "q5_close_velocity"):
        path = f"geometry/dynamic_arrows/{name}"
        entities.add(path)
        components[path] = [
            "Arrows3D:colors",
            "Arrows3D:labels",
            "Arrows3D:origins",
            "Arrows3D:radii",
            "Arrows3D:vectors",
            "CoordinateFrame:frame",
        ]
    for name in (
        "q5_rad",
        "raw_gripper_distance_mm",
        "live_gripper_distance_mm",
        "raw_live_delta_mm",
        "table_clearance_mm",
    ):
        path = f"metrics/d351/{name}"
        entities.add(path)
        components[path] = ["Scalars:scalars"]
    entities.add("events/d351/closure_sample")
    components["events/d351/closure_sample"] = ["TextLog:level", "TextLog:text"]
    return sorted(entities), components


def _rrd_persisted_step_contract(
    path: Path, expected_step_count: int
) -> dict[str, Any]:
    import rerun_bindings as rb

    required: dict[str, str] = {
        "geometry/dynamic_points/gripper_live_surface": "Points3D:positions",
        "geometry/dynamic_points/moving_inner_patch": "Points3D:positions",
        "geometry/dynamic_points/gripper_origin": "Points3D:positions",
        "geometry/dynamic_points/raw_witness": "Points3D:positions",
        "geometry/dynamic_points/live_witness": "Points3D:positions",
        "geometry/dynamic_arrows/raw_gap": "Arrows3D:vectors",
        "geometry/dynamic_arrows/live_gap": "Arrows3D:vectors",
        "geometry/dynamic_arrows/q5_axis": "Arrows3D:vectors",
        "geometry/dynamic_arrows/q5_close_velocity": "Arrows3D:vectors",
        "metrics/d351/q5_rad": "Scalars:scalars",
        "metrics/d351/raw_gripper_distance_mm": "Scalars:scalars",
        "metrics/d351/live_gripper_distance_mm": "Scalars:scalars",
        "metrics/d351/raw_live_delta_mm": "Scalars:scalars",
        "metrics/d351/table_clearance_mm": "Scalars:scalars",
        "events/d351/closure_sample": "TextLog:text",
    }
    expected_steps = list(range(expected_step_count))
    chunks = rb.RrdReaderInternal(str(path)).stream().to_chunks()
    rows: dict[str, Any] = {}
    for entity_path, component in required.items():
        observed_steps: list[int] = []
        matched_chunks = 0
        for chunk in chunks:
            if str(chunk.entity_path).lstrip("/") != entity_path:
                continue
            payload = chunk.to_record_batch().to_pydict()
            if component not in payload or "step" not in payload:
                continue
            matched_chunks += 1
            observed_steps.extend(int(value) for value in payload["step"])
        checks = {
            "step_set_exact": sorted(set(observed_steps)) == expected_steps,
            "one_persisted_row_per_step": sorted(observed_steps) == expected_steps,
            "component_and_step_co_resident": matched_chunks > 0,
        }
        rows[entity_path] = {
            "required_component": component,
            "matched_chunk_count": matched_chunks,
            "observed_steps": observed_steps,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {
        "reader": "rerun_bindings.RrdReaderInternal.stream().to_chunks()",
        "recording_chunk_count": len(chunks),
        "expected_steps": expected_steps,
        "entities": rows,
        "pass": all(row["pass"] for row in rows.values()),
    }


def _build_rerun_rows(
    inner: Any,
    topology_parts: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    moving_display: dict[str, Any],
) -> tuple[list[dict[str, Any]], ...]:
    coordinate_frames = [
        {
            "frame": "world",
            "parent_frame": "tf#/",
            "entity_path": "coordinate_frames/world",
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    ]
    meshes: list[dict[str, Any]] = []
    mesh_paths: list[str] = []
    for body in d334.BODY_LABELS:
        pos, quat = d334._body_pose_w(inner, body)
        rot = _quat_to_rot(quat)
        for index, part in enumerate(topology_parts[body]):
            path = f"geometry/live_parts/{body}/part_{index:03d}"
            mesh_paths.append(path)
            meshes.append(
                {
                    "entity_path": path,
                    "coordinate_frame": "world",
                    "vertices_m": _world_vertices(part["_vertices"], pos, rot),
                    "triangles": part["_triangles"],
                    "color_rgba": _palette(body, index),
                    "static": True,
                    "part_idx": index,
                    "body": body,
                }
            )
    moving_path = "geometry/moving_jaw/raw_decision_component"
    mesh_paths.append(moving_path)
    meshes.append(
        {
            "entity_path": moving_path,
            "coordinate_frame": "world",
            "vertices_m": _world_vertices(
                moving_display["component_vertices"],
                moving_display["gripper_pos"],
                moving_display["gripper_rot"],
            ),
            "triangles": moving_display["component_triangles"],
            "color_rgba": [60, 255, 120, 125],
            "static": True,
        }
    )
    inner_patch_path = "geometry/moving_jaw/frozen_distal_inner_patch"
    mesh_paths.append(inner_patch_path)
    meshes.append(
        {
            "entity_path": inner_patch_path,
            "coordinate_frame": "world",
            "vertices_m": _world_vertices(
                moving_display["inner_patch_vertices"],
                moving_display["gripper_pos"],
                moving_display["gripper_rot"],
            ),
            "triangles": moving_display["inner_patch_triangles"],
            "color_rgba": [255, 45, 45, 235],
            "static": True,
        }
    )
    cyl_vertices, cyl_triangles = _cylinder_mesh()
    cylinder_path = "geometry/target/cylinder"
    mesh_paths.append(cylinder_path)
    meshes.append(
        {
            "entity_path": cylinder_path,
            "coordinate_frame": "world",
            "vertices_m": cyl_vertices + OBJECT_POS_F32.astype(np.float64),
            "triangles": cyl_triangles,
            "color_rgba": [255, 190, 20, 100],
            "static": True,
        }
    )
    points: list[dict[str, Any]] = []
    arrows: list[dict[str, Any]] = []
    scalars: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    live_gripper_local_vertices = np.vstack(
        [
            np.asarray(part["_vertices"], dtype=np.float64)
            for part in topology_parts["gripper_link"]
        ]
    )
    inner_patch_local_vertices = np.asarray(
        moving_display["inner_patch_vertices"], dtype=np.float64
    )
    for step, row in enumerate(rows):
        sequence = {"step": step}
        joint_geometry = row["runtime_joint_geometry"]
        grip_pos = np.asarray(
            joint_geometry["gripper_body_position_world_m"], dtype=np.float64
        )
        grip_rot = _quat_to_rot(
            joint_geometry["gripper_body_quaternion_wxyz"]
        )
        joint_origin = np.asarray(joint_geometry["origin_world_m"], dtype=np.float64)
        joint_axis = np.asarray(joint_geometry["axis_world"], dtype=np.float64)
        dynamic_live_world = _world_vertices(
            live_gripper_local_vertices, grip_pos, grip_rot
        )
        dynamic_patch_world = _world_vertices(
            inner_patch_local_vertices, grip_pos, grip_rot
        )
        patch_centroid = dynamic_patch_world.mean(axis=0)
        close_velocity = -np.cross(joint_axis, patch_centroid - joint_origin)
        points.extend(
            [
                {
                    "entity_path": "geometry/dynamic_points/gripper_live_surface",
                    "coordinate_frame": "world",
                    "positions_m": dynamic_live_world,
                    "radii": [0.00030],
                    "colors": [[25, 255, 90]],
                    "sequence": sequence,
                },
                {
                    "entity_path": "geometry/dynamic_points/moving_inner_patch",
                    "coordinate_frame": "world",
                    "positions_m": dynamic_patch_world,
                    "radii": [0.00055],
                    "colors": [[255, 45, 45]],
                    "sequence": sequence,
                },
                {
                    "entity_path": "geometry/dynamic_points/gripper_origin",
                    "coordinate_frame": "world",
                    "positions_m": [joint_origin.tolist()],
                    "radii": [0.0020],
                    "colors": [[255, 120, 10]],
                    "labels": ["q5 runtime pivot"],
                    "sequence": sequence,
                },
            ]
        )
        arrows.extend(
            [
                {
                    "entity_path": "geometry/dynamic_arrows/q5_axis",
                    "coordinate_frame": "world",
                    "origins_m": [joint_origin.tolist()],
                    "vectors_m": [(joint_axis * 0.035).tolist()],
                    "radii": [0.0010],
                    "colors": [[255, 120, 10]],
                    "labels": ["q5 +axis"],
                    "sequence": sequence,
                },
                {
                    "entity_path": "geometry/dynamic_arrows/q5_close_velocity",
                    "coordinate_frame": "world",
                    "origins_m": [patch_centroid.tolist()],
                    "vectors_m": [(close_velocity * 0.25).tolist()],
                    "radii": [0.0010],
                    "colors": [[255, 80, 40]],
                    "labels": ["q5 decrease direction (0.25 rad scale)"],
                    "sequence": sequence,
                },
            ]
        )
        for representation, color in (("raw", [255, 235, 20]), ("live", [255, 20, 230])):
            query = row[representation]["gripper_link"]
            geom = np.asarray(query["witness_endpoint_geometry_m"], dtype=np.float64)
            cyl = np.asarray(query["witness_endpoint_cylinder_m"], dtype=np.float64)
            points.append(
                {
                    "entity_path": f"geometry/dynamic_points/{representation}_witness",
                    "coordinate_frame": "world",
                    "positions_m": [geom.tolist(), cyl.tolist()],
                    "radii": [0.0020, 0.0020],
                    "colors": [color, color],
                    "labels": [f"{representation}_jaw", f"{representation}_cylinder"],
                    "sequence": sequence,
                }
            )
            arrows.append(
                {
                    "entity_path": f"geometry/dynamic_arrows/{representation}_gap",
                    "coordinate_frame": "world",
                    "origins_m": [geom.tolist()],
                    "vectors_m": [(cyl - geom).tolist()],
                    "radii": [0.0012],
                    "colors": [color],
                    "labels": [f"{representation}_gap"],
                    "sequence": sequence,
                }
            )
        values = {
            "q5_rad": row["q5_float32_rad"],
            "raw_gripper_distance_mm": row["raw"]["gripper_link"]["exact_signed_distance_mm"],
            "live_gripper_distance_mm": row["live"]["gripper_link"]["exact_signed_distance_mm"],
            "raw_live_delta_mm": row["raw_live_gripper_absolute_delta_mm"],
            "table_clearance_mm": row["diagnostics"]["gripper_table_clearance_mm"],
        }
        for name, value in values.items():
            if value is None or not math.isfinite(float(value)):
                continue
            scalars.append(
                {
                    "entity_path": f"metrics/d351/{name}",
                    "value": float(value),
                    "sequence": sequence,
                }
            )
        events.append(
            {
                "entity_path": "events/d351/closure_sample",
                "text": (
                    f"idx={step};q5={row['q5_float32_rad']:.9f};"
                    f"raw={row['raw']['gripper_link']['exact_signed_distance_mm']};"
                    f"live={row['live']['gripper_link']['exact_signed_distance_mm']};"
                    f"raw_feature={row['raw']['gripper_link']['cylinder_feature']['feature']};"
                    f"live_feature={row['live']['gripper_link']['cylinder_feature']['feature']};steps=0"
                ),
                "level": "INFO",
                "sequence": sequence,
            }
        )
    return coordinate_frames, meshes, points, arrows, scalars, events, mesh_paths


def _run_rerun(
    inner: Any,
    topology_parts: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    moving_display: dict[str, Any],
    display_pose_role: str,
) -> dict[str, Any]:
    coordinate_frames, meshes, points, arrows, scalars, events, mesh_paths = _build_rerun_rows(
        inner, topology_parts, rows, moving_display
    )
    expected_steps = set(range(len(rows)))

    def step_coverage(collection: list[dict[str, Any]], path: str) -> bool:
        return {
            int(item["sequence"]["step"])
            for item in collection
            if item.get("entity_path") == path
        } == expected_steps

    temporal_checks = {
        "dynamic_live_surface_every_step": step_coverage(
            points, "geometry/dynamic_points/gripper_live_surface"
        ),
        "dynamic_inner_patch_every_step": step_coverage(
            points, "geometry/dynamic_points/moving_inner_patch"
        ),
        "dynamic_raw_witness_every_step": step_coverage(
            points, "geometry/dynamic_points/raw_witness"
        ),
        "dynamic_live_witness_every_step": step_coverage(
            points, "geometry/dynamic_points/live_witness"
        ),
        "dynamic_q5_axis_every_step": step_coverage(
            arrows, "geometry/dynamic_arrows/q5_axis"
        ),
        "dynamic_close_velocity_every_step": step_coverage(
            arrows, "geometry/dynamic_arrows/q5_close_velocity"
        ),
        "all_five_scalars_every_step": all(
            step_coverage(scalars, f"metrics/d351/{name}")
            for name in (
                "q5_rad",
                "raw_gripper_distance_mm",
                "live_gripper_distance_mm",
                "raw_live_delta_mm",
                "table_clearance_mm",
            )
        ),
        "event_every_step": step_coverage(events, "events/d351/closure_sample"),
        "static_mesh_paths_unique_131": len(mesh_paths) == len(set(mesh_paths)) == 131,
        "dense_live_point_count_constant_nonzero": len(
            {
                len(item["positions_m"])
                for item in points
                if item.get("entity_path")
                == "geometry/dynamic_points/gripper_live_surface"
            }
        )
        == 1
        and any(
            len(item["positions_m"]) > 0
            for item in points
            if item.get("entity_path")
            == "geometry/dynamic_points/gripper_live_surface"
        ),
    }
    temporal_contract_pass = all(temporal_checks.values())
    status = log_rerun(
        RRD_PATH,
        coordinate_frames=coordinate_frames,
        meshes=meshes,
        points=points,
        arrows=arrows,
        scalar_trace=scalars,
        events=events,
        recording_metadata={
            "case": CASE,
            "purpose": "zero-step moving-jaw closure geometry timeline",
            "git_head": _git_head(),
            "new_variables": NEW_VARIABLES,
            "target": "D350 q0-q4/object frozen; q5 sampled OPEN to CLOSED",
            "physics": "forbidden; controlled steps 0",
            "live_semantics": "callback-topology surface proxy, not direct PhysX narrowphase",
            "scientific_authority": "Float64 JSON/CSV and immutable input hashes",
            "viewer_role": "Float32 one-way observability copy",
            "static_display_pose_role": display_pose_role,
            "spatial_timeline": (
                "64+64 Mesh3D is static at the raw last-clear decision pose or "
                "at the explicitly labeled OPEN fallback pose; "
                "dense live gripper surface, frozen inner patch, runtime pivot/axis, "
                "and raw/live witnesses move at every q5 step"
            ),
        },
        recording_id="g0a_d351_zero_step_closure_geometry",
        blueprint_path=RBL_PATH,
        blueprint_mode="robot_geometry",
        live_viewer=False,
        app_id="roarm_g0a_d351_closure_geometry",
    )
    entities, components = _rrd_contract()
    validation = (
        validate_rerun_artifact(
            RRD_PATH,
            expected_entity_paths=[
                "geometry/live_parts/link5/part_000",
                "geometry/live_parts/gripper_link/part_063",
                "geometry/moving_jaw/raw_decision_component",
                "geometry/moving_jaw/frozen_distal_inner_patch",
                "geometry/dynamic_points/gripper_live_surface",
                "geometry/dynamic_points/moving_inner_patch",
                "geometry/dynamic_points/raw_witness",
                "metrics/d351/q5_rad",
                "events/d351/closure_sample",
            ],
            expected_timeline_names=["step"],
            exact_entity_paths=entities,
            exact_timeline_names=["blueprint", "log_time", "step"],
            expected_entity_components=components,
            blueprint_path=RBL_PATH,
            screenshot_path=RERUN_PNG_PATH,
        )
        if status.get("ok")
        else {"pass": False, "errors": ["log_rerun failed"]}
    )
    persisted_steps = (
        _rrd_persisted_step_contract(RRD_PATH, len(rows))
        if status.get("ok") and RRD_PATH.is_file()
        else {"pass": False, "error": "RRD was not finalized"}
    )
    report = {
        "artifact": "D351_RERUN_MACHINE_VALIDATION_V1",
        "log_status": status,
        "archive_validation": validation,
        "persisted_step_contract": persisted_steps,
        "exact_contract": {
            "entity_count": len(entities),
            "entity_paths_sha256": hashlib.sha256(
                json.dumps(entities, separators=(",", ":")).encode()
            ).hexdigest(),
            "timeline_names": ["blueprint", "log_time", "step"],
            "required_component_path_count": len(components),
            "mesh_count": len(meshes),
            "dynamic_sample_count": len(rows),
            "point_rows": len(points),
            "arrow_rows": len(arrows),
            "scalar_rows": len(scalars),
            "event_rows": len(events),
            "temporal_checks": temporal_checks,
            "temporal_contract_pass": temporal_contract_pass,
        },
        "pass": bool(
            temporal_contract_pass
            and status.get("ok")
            and validation.get("pass")
            and persisted_steps.get("pass")
        ),
    }
    _write_json(RERUN_VALIDATION_PATH, report)
    return report


def _run_validate(
    args: argparse.Namespace,
    simulation_app: Any,
    launcher_report: dict[str, Any],
) -> int:
    import omni.timeline

    prereg = _json(PREREG_PATH)
    source_before = d349._source_inventories()
    input_before = _input_hashes()
    args.robot_usd_path = VARIANT_ROBOT_USD
    inner = d333._make_runtime_env(args)
    timeline = omni.timeline.get_timeline_interface()
    try:
        inner.reset(seed=SEED)
        inner.sim.set_setting("/app/player/playSimulations", False)
        if timeline.is_playing():
            timeline.pause()
        counter_after_reset = int(inner._sim_step_counter)
        corrected = d349._corrected_live_audit()
        topology_parts, live_binding = d349._build_live_topology_parts(inner)
        _write_json(LIVE_BINDING_PATH, live_binding)
        raw_shapes, raw_contract = d339._build_retained_raw_shapes(
            inner, _json(D334_SUMMARY)
        )
        prerequisites = {
            "counter_after_reset_zero": counter_after_reset == 0,
            "timeline_paused": not timeline.is_playing(),
            "corrected_d348_128_of_128": corrected["pass"],
            "live_binding_64_plus_64": live_binding["pass"],
            "raw_source_contract": raw_contract["pass"],
            "launcher": launcher_report["pass"],
        }
        if not all(prerequisites.values()):
            raise RuntimeError(f"D351 runtime prerequisites STOP: {prerequisites}")

        raw_part_map = _raw_parts(raw_shapes)
        cache: dict[str, dict[str, Any]] = {}
        evaluation_order: list[str] = []

        def evaluate(q5: float) -> dict[str, Any]:
            return _evaluate_q5(
                inner,
                timeline,
                q5,
                raw_part_map,
                topology_parts,
                raw_shapes,
                cache,
                evaluation_order,
            )

        q_grid = [
            float(value)
            for value in np.linspace(
                Q5_OPEN_F32, Q5_CLOSED_F32, GRID_COUNT, dtype=np.float32
            )
        ]
        for q5 in q_grid:
            evaluate(q5)

        raw_gripper = next(row for row in raw_shapes if row["body"] == "gripper_link")
        raw_radius = float(
            np.max(np.linalg.norm(np.asarray(raw_gripper["_raw_verts"])[:, :2], axis=1))
        )
        live_radius = float(
            max(
                np.max(np.linalg.norm(np.asarray(part["_vertices"])[:, :2], axis=1))
                for part in topology_parts["gripper_link"]
            )
        )
        raw_order = _certify_first_contact("raw", q_grid, raw_radius, evaluate)
        live_order = _certify_first_contact("live", q_grid, live_radius, evaluate)
        q_table_stop = (
            min(
                float(raw_order["first_contact_bracket"]["q_clear_float32_rad"]),
                float(live_order["first_contact_bracket"]["q_clear_float32_rad"]),
            )
            if raw_order["first_contact_bracket"] is not None
            and live_order["first_contact_bracket"] is not None
            else float(Q5_CLOSED_F32)
        )
        table_corridor = _certify_precontact_table_clearance(
            q_grid, q_table_stop, max(raw_radius, live_radius), evaluate
        )
        table_corridor["representation_semantics"] = (
            "per-row minimum raw/live gripper clearance certified with the "
            "maximum raw/live q5 rotation radius"
        )
        rows = sorted(cache.values(), key=lambda row: row["q5_float32_rad"], reverse=True)

        if (
            raw_order["first_contact_bracket"] is None
            or live_order["first_contact_bracket"] is None
        ):
            moving_binding, moving_display = _fallback_moving_display(
                inner,
                timeline,
                raw_shapes,
                cache[_q5_key(float(Q5_OPEN_F32))],
                "raw or live first-contact bracket unresolved",
            )
        else:
            raw_clear_key = _q5_key(raw_order["first_contact_bracket"]["q_clear_float32_rad"])
            raw_overlap_key = _q5_key(
                raw_order["first_contact_bracket"]["q_overlap_float32_rad"]
            )
            live_clear_key = _q5_key(live_order["first_contact_bracket"]["q_clear_float32_rad"])
            live_overlap_key = _q5_key(
                live_order["first_contact_bracket"]["q_overlap_float32_rad"]
            )
            raw_clear_row = cache[raw_clear_key]
            raw_overlap_row = cache[raw_overlap_key]
            live_clear_row = cache[live_clear_key]
            live_overlap_row = cache[live_overlap_key]
            moving_binding, moving_display = _bind_moving_surface(
                inner,
                timeline,
                raw_shapes,
                topology_parts,
                raw_clear_row,
                raw_overlap_row,
                live_clear_row,
                live_overlap_row,
                cache[_q5_key(float(Q5_OPEN_F32))],
            )
        _write_json(MOVING_BINDING_PATH, moving_binding)
        fixed_reproduction = _fixed_component_digest_reproduction(inner, timeline, raw_shapes)

        repeat_cache: dict[str, dict[str, Any]] = {}
        repeat_order: list[str] = []
        repeat_closed = _evaluate_q5(
            inner,
            timeline,
            float(Q5_CLOSED_F32),
            raw_part_map,
            topology_parts,
            raw_shapes,
            repeat_cache,
            repeat_order,
        )
        repeat_open = _evaluate_q5(
            inner,
            timeline,
            float(Q5_OPEN_F32),
            raw_part_map,
            topology_parts,
            raw_shapes,
            repeat_cache,
            repeat_order,
        )
        endpoint_requery = {
            "closed_to_open_order": repeat_order,
            "closed_state_guard_pass": repeat_closed["state_guard"]["pass"],
            "open_state_guard_pass": repeat_open["state_guard"]["pass"],
            "open_raw_exact": repeat_open["raw"]["gripper_link"]
            == cache[_q5_key(float(Q5_OPEN_F32))]["raw"]["gripper_link"],
            "open_live_exact": repeat_open["live"]["gripper_link"]
            == cache[_q5_key(float(Q5_OPEN_F32))]["live"]["gripper_link"],
            "closed_raw_exact": repeat_closed["raw"]["gripper_link"]
            == cache[_q5_key(float(Q5_CLOSED_F32))]["raw"]["gripper_link"],
            "closed_live_exact": repeat_closed["live"]["gripper_link"]
            == cache[_q5_key(float(Q5_CLOSED_F32))]["live"]["gripper_link"],
        }
        endpoint_requery["pass"] = all(
            bool(value)
            for key, value in endpoint_requery.items()
            if key != "closed_to_open_order"
        )

        classification = _classify_measurement(
            rows,
            raw_order,
            live_order,
            table_corridor,
            moving_binding,
            fixed_reproduction,
            raw_radius,
            live_radius,
        )
        classification["contract_checks"]["endpoint_reverse_requery_exact"] = endpoint_requery[
            "pass"
        ]
        classification["contract_pass"] = bool(
            classification["contract_pass"] and endpoint_requery["pass"]
        )
        if not endpoint_requery["pass"]:
            classification["scientific_verdict"] = VERDICT_INPUT

        _write_sweep_csv(rows)
        measurement = {
            "artifact": "D351_ZERO_STEP_CLOSURE_GEOMETRY_MEASUREMENT_V1",
            "case": CASE,
            "new_variables": NEW_VARIABLES,
            "new_physical_variables": NEW_PHYSICAL_VARIABLES,
            "query_semantics": {
                "raw": "retained authored full triangle mesh",
                "live": "D348 callback-topology triangle-surface proxy; not direct PhysX narrowphase",
                "query_order": ["raw", "live"],
            },
            "runtime_prerequisites": prerequisites,
            "q_grid_float32_rad": q_grid,
            "rotation_radius": {
                "raw_gripper_m": raw_radius,
                "live_gripper_proxy_m": live_radius,
                "axis": "gripper_link body-local +z through origin (URDF q5 axis)",
            },
            "raw_contact_order": raw_order,
            "live_contact_order": live_order,
            "precontact_table_clearance_corridor": table_corridor,
            "live_topology_runtime_binding_sha256": _sha(LIVE_BINDING_PATH),
            "moving_surface_binding_sha256": _sha(MOVING_BINDING_PATH),
            "fixed_component_digest_reproduction": fixed_reproduction,
            "endpoint_reverse_requery": endpoint_requery,
            "classification": classification,
            "executed_rows": rows,
            "execution_count": len(rows),
            "scope_guards": _scope_guards(),
            "controlled_physics_steps": 0,
            "g0a_pass": False,
        }
        _write_json(MEASUREMENT_PATH, measurement)

        if moving_binding.get("artifact") == "D351_MOVING_JAW_BINDING_NOT_RUN_WITH_VISUAL_FALLBACK":
            raw_clear_row = cache[_q5_key(float(Q5_OPEN_F32))]
            q_display = float(Q5_OPEN_F32)
            display_pose_role = "open_fallback_no_resolved_contact_bracket"
        else:
            # A resolved raw bracket means ``moving_display`` was measured at
            # its last-clear pose even when a later identity/sign contract
            # rejects scientific eligibility.  Keep failure evidence in that
            # same pose instead of mixing an OPEN robot with pre-contact guides.
            q_display = float(raw_order["first_contact_bracket"]["q_clear_float32_rad"])
            raw_clear_row = cache[_q5_key(q_display)]
            display_pose_role = "resolved_raw_last_clear_decision_pose"
        _set_state_only(inner, timeline, q_display)
        overlay = _create_viewer_guides(
            inner,
            topology_parts,
            moving_display,
            raw_clear_row,
            raw_clear_row,
        )
        _write_json(OVERLAY_PATH, overlay)
        rerun = _run_rerun(
            inner,
            topology_parts,
            rows,
            moving_display,
            display_pose_role,
        )
        viewer = _run_viewer(
            args,
            simulation_app,
            inner,
            timeline,
            q_display,
            display_pose_role,
        )
        _write_json(CAPTURE_PATH, viewer)

        source_after = d349._source_inventories()
        input_after = _input_hashes()
        immutability = {
            "source_inventories_exact": source_before
            == source_after
            == prereg.get("source_inventories", source_before),
            "input_hashes_exact": input_before
            == input_after
            == EXPECTED_INPUT_HASHES,
            "git_scope_only_d351": _status_scope_pass(_git_status()),
            "preexisting_user_files_read_only_exact": (
                _preexisting_user_untracked_contract()
                == prereg.get("preexisting_user_files")
            ),
            "asset_write": False,
        }
        immutability["pass"] = all(immutability.values())
        scientific_result_recorded = classification["scientific_verdict"] in (
            VERDICT_INPUT,
            VERDICT_ORDER,
            VERDICT_BINDING,
            VERDICT_ELIGIBLE,
            VERDICT_REPAIR,
        )
        observability_pass = bool(
            overlay["pass"]
            and rerun["pass"]
            and viewer["pass"]
            and launcher_report["pass"]
            and immutability["pass"]
            and int(inner._sim_step_counter) == 0
        )
        automated_pass = bool(scientific_result_recorded and observability_pass)
        summary = {
            "artifact": "D351_AUTOMATED_SUMMARY_V1",
            "case": CASE,
            "scientific_verdict": classification["scientific_verdict"],
            "automated_verdict": (
                classification["scientific_verdict"] + VERDICT_PENDING_SUFFIX
                if automated_pass
                else VERDICT_VISUAL
            ),
            "automated_pass": automated_pass,
            "scientific_contract_pass": classification["contract_pass"],
            "scientific_result_recorded": scientific_result_recorded,
            "observability_pass": observability_pass,
            "new_variables": NEW_VARIABLES,
            "new_physical_variables": NEW_PHYSICAL_VARIABLES,
            "launcher": launcher_report,
            "classification": classification,
            "execution_count": len(rows),
            "raw_contact_bracket": raw_order["first_contact_bracket"],
            "live_contact_bracket": live_order["first_contact_bracket"],
            "moving_surface_binding_pass": moving_binding["pass"],
            "fixed_component_digest_reproduction_pass": fixed_reproduction["pass"],
            "overlay_pass": overlay["pass"],
            "rerun_pass": rerun["pass"],
            "viewer_capture_tokens_pass": viewer["pass"],
            "postclose_png_validation_pending": True,
            "immutability": immutability,
            "controlled_physics_steps": 0,
            "target_ik_path_changed": False,
            "settle_executed": False,
            "ten_trial_run": False,
            "g0b_run": False,
            "rl_or_ppo_run": False,
            "ladder_promoted": False,
            "g0a_pass": False,
            "manual_visual_inspection_pending": True,
            "evidence_hashes": {
                "preregistration": _sha(PREREG_PATH),
                "parameter_freeze": _sha(PARAMETER_PATH),
                "validate_preflight": _sha(PREFLIGHT_PATH),
                "live_binding": _sha(LIVE_BINDING_PATH),
                "measurement": _sha(MEASUREMENT_PATH),
                "moving_binding": _sha(MOVING_BINDING_PATH),
                "sweep_csv": _sha(SWEEP_CSV_PATH),
                "overlay": _sha(OVERLAY_PATH),
                "capture": _sha(CAPTURE_PATH),
                "rerun_validation": _sha(RERUN_VALIDATION_PATH),
                "rrd": _sha(RRD_PATH),
                "rbl": _sha(RBL_PATH),
                "rerun_screenshot": _sha(RERUN_PNG_PATH),
            },
            "harness_sha256": _sha(HARNESS),
            "state_hashes": {
                "start_here": _sha(START_HERE),
                "session_doc": _sha(SESSION_DOC),
            },
        }
        _write_json(AUTOMATED_PATH, summary)
        _write_text(
            AUTOMATED_MD_PATH,
            "\n".join(
                [
                    "# D351 automated result",
                    "",
                    f"- scientific verdict: `{summary['scientific_verdict']}`",
                    f"- automated pass: `{automated_pass}`",
                    f"- executed zero-step q5 samples: `{len(rows)}`",
                    f"- raw first-contact bracket: `{raw_order['first_contact_bracket']}`",
                    f"- live first-contact bracket: `{live_order['first_contact_bracket']}`",
                    f"- raw/live features: `{classification['raw_first_contact_feature']}` / `{classification['live_first_contact_feature']}`",
                    f"- minimum table clearance: `{classification['min_gripper_table_clearance_mm']}` mm",
                    "- controlled physics steps: `0`",
                    "- target/IK/path change: `false`",
                    "- g0a_pass: `false`",
                ]
            )
            + "\n",
        )
        print(
            json.dumps(
                {
                    "stage": "validate",
                    "automated_pass": automated_pass,
                    "scientific_verdict": summary["scientific_verdict"],
                    "execution_count": len(rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if automated_pass else 2
    finally:
        inner.close()


def _manual_checks(manual: dict[str, Any]) -> dict[str, bool]:
    expected = {**VIEWER_PNGS, "rerun": RERUN_PNG_PATH}
    checks: dict[str, bool] = {
        "artifact_exact": manual.get("artifact") == "D351_MANUAL_VISUAL_INSPECTION_V1",
        "case_exact": manual.get("case") == CASE,
        "date_exact": manual.get("date") == "2026-07-15 KST",
        "method_original_resolution": manual.get("method")
        == "view_image original_resolution",
        "image_set_exact": set(manual.get("images", {})) == set(expected),
        "manual_pass_declared": manual.get("manual_pass") is True,
        "scientific_override_false": manual.get("scientific_override") is False,
        "bounded_interpretation_nonempty": bool(manual.get("bounded_interpretation")),
        "manual_markdown_path": manual.get("manual_markdown", {}).get("path")
        == _rel(MANUAL_MD_PATH),
        "manual_markdown_sha": MANUAL_MD_PATH.is_file()
        and manual.get("manual_markdown", {}).get("sha256") == _sha(MANUAL_MD_PATH),
    }
    for name, path in expected.items():
        row = manual.get("images", {}).get(name, {})
        checks[f"{name}_path"] = row.get("path") == _rel(path)
        checks[f"{name}_sha"] = path.is_file() and row.get("sha256") == _sha(path)
        checks[f"{name}_bytes"] = path.is_file() and row.get("bytes") == path.stat().st_size
        checks[f"{name}_dimensions"] = row.get("raster_dimensions") == _png_dimensions(path)
        checks[f"{name}_observation_nonempty"] = bool(row.get("observation"))
    declared = manual.get("checks", {})
    for name in (
        "actual_isaac_open_and_decision_or_fallback_pose_visible",
        "actual_physx_colliders_visible",
        "colored_link5_64_and_gripper_64_distinguishable",
        "inner_patch_fixed_moving_chord_and_cylinder_feature_visible",
        "rerun_full_q5_dynamic_live_surface_patch_and_witness_timeline_visible",
        "no_obvious_empty_or_corrupt_panel",
    ):
        checks[name] = declared.get(name) is True
    return checks


def _run_finalize(_args: argparse.Namespace) -> int:
    automated = _json(AUTOMATED_PATH)
    manual = _json(MANUAL_PATH)
    prereg = _json(PREREG_PATH)
    preflight = _json(PREFLIGHT_PATH)
    capture_contract = _json(CAPTURE_PATH)
    manual_checks = _manual_checks(manual)
    validate_pid = preflight.get("validate_process_identity", {}).get("pid")
    process_absent_before_sampling = bool(
        isinstance(validate_pid, int) and not psutil.pid_exists(validate_pid)
    )
    pngs = {
        name: _stable_png(path, expected_size=VIEWER_RASTER_SIZE)
        for name, path in VIEWER_PNGS.items()
    }
    pngs["rerun"] = _stable_png(
        RERUN_PNG_PATH, expected_size=RERUN_RASTER_SIZE
    )
    process_absent_after_sampling = bool(
        isinstance(validate_pid, int) and not psutil.pid_exists(validate_pid)
    )
    viewer_png_hashes = [
        pngs[name]["samples"][-1]["sha256"] for name in VIEWER_PNGS
    ]
    display_pose_role = capture_contract.get("display_pose_role")
    if display_pose_role == "resolved_raw_last_clear_decision_pose":
        viewer_png_distinct_contract = bool(
            all(value is not None for value in viewer_png_hashes)
            and len(set(viewer_png_hashes)) == len(viewer_png_hashes)
        )
    else:
        by_name = {
            name: pngs[name]["samples"][-1]["sha256"] for name in VIEWER_PNGS
        }
        viewer_png_distinct_contract = bool(
            display_pose_role == "open_fallback_no_resolved_contact_bracket"
            and all(value is not None for value in by_name.values())
            and len(set(by_name.values())) >= 3
            and by_name["decision_colored"] != by_name["decision_side"]
            and by_name["open_physx"]
            not in {by_name["decision_colored"], by_name["decision_side"]}
            and by_name["decision_physx"]
            not in {by_name["decision_colored"], by_name["decision_side"]}
        )
    postclose_png_pass = bool(
        process_absent_before_sampling
        and process_absent_after_sampling
        and all(row["pass"] for row in pngs.values())
        and viewer_png_distinct_contract
    )
    evidence_paths = {
        "preregistration": PREREG_PATH,
        "parameter_freeze": PARAMETER_PATH,
        "validate_preflight": PREFLIGHT_PATH,
        "live_binding": LIVE_BINDING_PATH,
        "measurement": MEASUREMENT_PATH,
        "moving_binding": MOVING_BINDING_PATH,
        "sweep_csv": SWEEP_CSV_PATH,
        "overlay": OVERLAY_PATH,
        "capture": CAPTURE_PATH,
        "rerun_validation": RERUN_VALIDATION_PATH,
        "rrd": RRD_PATH,
        "rbl": RBL_PATH,
        "rerun_screenshot": RERUN_PNG_PATH,
    }
    evidence_files_exist = all(path.is_file() for path in evidence_paths.values())
    observed_evidence = (
        {name: _sha(path) for name, path in evidence_paths.items()}
        if evidence_files_exist
        else {}
    )
    evidence_exact = bool(
        evidence_files_exist and automated.get("evidence_hashes") == observed_evidence
    )
    state_hashes = {
        "start_here": _sha(START_HERE),
        "session_doc": _sha(SESSION_DOC),
    }
    artifact_binding_checks = {
        "git_head_exact": _git_head() == EXPECTED_HEAD == prereg.get("git_head"),
        "git_scope_only_d351": _status_scope_pass(_git_status()),
        "preexisting_user_files_read_only_exact": (
            _preexisting_user_untracked_contract()
            == prereg.get("preexisting_user_files")
        ),
        "input_hashes_exact": _input_hashes()
        == EXPECTED_INPUT_HASHES
        == prereg.get("input_hashes"),
        "source_inventories_exact": d349._source_inventories()
        == prereg.get("source_inventories"),
        "harness_hash_exact": _sha(HARNESS)
        == prereg.get("harness_sha256")
        == automated.get("harness_sha256"),
        "state_hashes_exact": state_hashes
        == prereg.get("state_hashes")
        == automated.get("state_hashes"),
        "parameter_hash_exact": _sha(PARAMETER_PATH)
        == prereg.get("parameter_audit_sha256")
        == preflight.get("parameter_audit_sha256"),
        "prereg_hash_exact": _sha(PREREG_PATH)
        == preflight.get("preregistration_sha256"),
        "preflight_harness_hash_exact": _sha(HARNESS)
        == preflight.get("harness_sha256"),
        "environment_exact": _environment_contract()["pass"],
        "manual_json_exists": MANUAL_PATH.is_file(),
        "manual_markdown_exists": MANUAL_MD_PATH.is_file(),
        "evidence_hashes_exact": evidence_exact,
    }
    artifact_binding_pass = all(artifact_binding_checks.values())
    rerun_validation = _json(RERUN_VALIDATION_PATH)
    completion_pass = bool(
        automated["automated_pass"]
        and postclose_png_pass
        and all(manual_checks.values())
        and artifact_binding_pass
        and rerun_validation["pass"]
    )
    final_verdict = automated["scientific_verdict"] if completion_pass else VERDICT_VISUAL
    completion = {
        "artifact": "D351_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "completion_pass": completion_pass,
        "final_verdict": final_verdict,
        "scientific_verdict": automated["scientific_verdict"],
        "postclose_png_validation": {
            "validate_process_pid": validate_pid,
            "process_absent_before_sampling": process_absent_before_sampling,
            "process_absent_after_sampling": process_absent_after_sampling,
            "rows": pngs,
            "display_pose_role": display_pose_role,
            "viewer_png_distinct_contract": viewer_png_distinct_contract,
            "pass": postclose_png_pass,
        },
        "manual_visual_inspection": {
            "path": _rel(MANUAL_PATH),
            "sha256": _sha(MANUAL_PATH),
            "markdown_path": _rel(MANUAL_MD_PATH),
            "markdown_sha256": _sha(MANUAL_MD_PATH),
            "checks": manual_checks,
            "pass": all(manual_checks.values()),
        },
        "evidence_hashes_exact": evidence_exact,
        "observed_evidence_hashes": observed_evidence,
        "artifact_binding": {
            "checks": artifact_binding_checks,
            "pass": artifact_binding_pass,
        },
        "rerun_pass": rerun_validation["pass"],
        "new_variables": NEW_VARIABLES,
        "new_physical_variables": NEW_PHYSICAL_VARIABLES,
        "controlled_physics_steps": 0,
        "target_ik_path_changed": False,
        "settle_authorized": False,
        "settle_executed": False,
        "ten_trial_run": False,
        "g0b_run": False,
        "rl_or_ppo_run": False,
        "ladder_promoted": False,
        "g0a_pass": False,
        "commit_or_push_performed": False,
    }
    _write_json(COMPLETION_PATH, completion)
    classification = automated["classification"]
    _write_text(
        COMPLETION_MD_PATH,
        "\n".join(
            [
                "# D351 completion",
                "",
                f"- completion pass: `{completion_pass}`",
                f"- final verdict: `{final_verdict}`",
                f"- raw first-contact: `{classification['raw_first_contact_feature']}`",
                f"- live first-contact: `{classification['live_first_contact_feature']}`",
                f"- minimum table clearance: `{classification['min_gripper_table_clearance_mm']}` mm",
                "- physics/settle/trials/RL/G0a: `0/false`",
                "- target/IK/path changed: `false`",
            ]
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "stage": "finalize",
                "completion_pass": completion_pass,
                "final_verdict": final_verdict,
            },
            sort_keys=True,
        )
    )
    return 0 if completion_pass else 2


def _parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "validate", "finalize"), required=True)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--urdf_path", type=Path, default=d333.DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--viewer_hold_seconds", type=float, default=120.0)
    if stage == "validate":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    stage_probe = argparse.ArgumentParser(add_help=False)
    stage_probe.add_argument("--stage", choices=("prepare", "validate", "finalize"), required=True)
    stage_args, _ = stage_probe.parse_known_args()
    args = _parser(stage_args.stage).parse_args()
    if Path(args.out_dir).resolve() != OUT_DIR.resolve():
        raise RuntimeError("D351 output path is fixed and forward-only")
    if Path(args.urdf_path).resolve() != Path(d333.DEFAULT_URDF).resolve():
        raise RuntimeError("D351 URDF path drift")
    if int(args.seed) != SEED:
        raise RuntimeError("D351 seed drift")
    if not 0.0 < float(args.viewer_hold_seconds) <= 600.0:
        raise RuntimeError("D351 Viewer hold must be >0 and <=600 seconds")
    args.process_nonce = secrets.token_hex(16)
    if args.stage == "prepare":
        return _run_prepare(args)
    if args.stage == "finalize":
        return _run_finalize(args)

    args.headless = False
    args.livestream = 0
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False
    if hasattr(args, "xr"):
        args.xr = False
    args.device = "cuda:0"
    if not _validate_preflight(args):
        return 2

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(copy.deepcopy(args))
    simulation_app = launcher.app
    launcher_report = d350._resolved_gui_launcher(launcher)
    try:
        if not launcher_report["pass"]:
            raise RuntimeError(f"D351 GUI launcher contract failed: {launcher_report}")
        return _run_validate(args, simulation_app, launcher_report)
    except Exception as error:
        if not RUNTIME_EXCEPTION_PATH.exists():
            _write_json(
                RUNTIME_EXCEPTION_PATH,
                {
                    "artifact": "D351_RUNTIME_EXCEPTION_STOP",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "controlled_physics_steps": None,
                    "g0a_pass": False,
                },
            )
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
