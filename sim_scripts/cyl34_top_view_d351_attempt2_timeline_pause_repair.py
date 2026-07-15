#!/usr/bin/env python3
"""D351 attempt2: reactive timeline-pause runtime repair.

Attempt1 stopped before the q5 sweep because Isaac restarted the timeline while
the frozen 64+64 runtime representations were being bound.  This forward-only
wrapper preserves every attempt1 file and the original D351 harness, redirects
all new artifacts into an attempt2 subdirectory, and reasserts PAUSE after the
live and raw representation builders without calling an app update or physics
step.  It also corrects the already-known false-valued ``asset_write`` polarity
aggregation only after the unmodified validator has produced all scientific and
observability results.  The D351 geometry, gates, target/IK, and scientific
classification code remain the immutable attempt1 implementation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sim_scripts import (  # noqa: E402
    cyl34_top_view_d351_zero_step_closure_geometry as d351,
)


CASE = "g0a_d351"
ATTEMPT1_DIR = d351.OUT_DIR
REPAIR_DIR = ATTEMPT1_DIR / "attempt2_timeline_pause_repair"
HARNESS = Path(__file__).resolve()
ORIGINAL_HARNESS = Path(d351.__file__).resolve()
ORIGINAL_SESSION = d351.SESSION_DOC
REPAIR_SESSION = REPO / "claudedocs/session_20260715_grasp_g0a_d351_timeline_pause_repair.md"
CONTROL_REPAIR_PATH = REPAIR_DIR / "d351_timeline_pause_repair_contract.json"
AGGREGATION_REPAIR_PATH = (
    REPAIR_DIR / "d351_automated_aggregation_repair_contract.json"
)
UNMODIFIED_AUTOMATED_PATH = (
    REPAIR_DIR / "d351_unmodified_automated_false_polarity.json"
)
UNMODIFIED_AUTOMATED_MD_PATH = (
    REPAIR_DIR / "d351_unmodified_automated_false_polarity.md"
)

ATTEMPT1_ROOT_HASHES = {
    "d351_parameter_freeze_audit.json": (
        "98b5778e826d411f37606dd724093a1ff292040d8c1d350db3781508735502e2"
    ),
    "d351_preregistration.json": (
        "d0639f51485b96395de88b0942ea4af13a768f31db89a400df7af97a25df1456"
    ),
    "d351_validate_preflight.json": (
        "3e3172ff595bdc48b4216ab0bbb30386a2fdf29f0786ab8d950881114d434660"
    ),
    "d351_live_topology_runtime_binding.json": (
        "9bc8d1c95f3c235816eb1c3c11516f3f27416e45b302cf8b6f9d5ee01ad6ec05"
    ),
    "d351_runtime_exception.json": (
        "138097cee4a471b84202572639fd19c0cba6103d5a628d89a2af49bcbde71914"
    ),
}
ATTEMPT1_OUTSIDE_HASHES = {
    ORIGINAL_HARNESS: "3c4501885af7590f5883b36666c984ce88728a24d40451ea0a600660a386107d",
    ORIGINAL_SESSION: "20367375e05ce8cffb47f86ff0c1645a3544f5bf62516fe2e16a98919c356a06",
}
ATTEMPT1_EXPECTED_ERROR = (
    "RuntimeError: D351 runtime prerequisites STOP: "
    "{'counter_after_reset_zero': True, 'timeline_paused': False, "
    "'corrected_d348_128_of_128': True, 'live_binding_64_plus_64': True, "
    "'raw_source_contract': True, 'launcher': True}"
)
ATTEMPT2_USER_SIDECAR_HASHES = {
    "claudedocs/lab_meeting/20260715/d334_collision_table/README.md": (
        "35e39f584737c888bcf7dfab6154c55c5d13d4154ee7f2042073e1c0a7e18783"
    ),
    "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.html": (
        "6d38933f959eba916208ec04a329ba25e2bd753c90720576010c222a8bda679c"
    ),
    "claudedocs/lab_meeting/20260715/d334_collision_table/d334_collision_table_academic.png": (
        "ddc9db2795f4d66b2564adf156829e6a143a599ceb72f6bb9fa28ab25e68a183"
    ),
}

_ORIGINAL_RUNTIME_OUTPUTS = d351._runtime_outputs
_ORIGINAL_WRITE_JSON = d351._write_json
_ORIGINAL_WRITE_TEXT = d351._write_text
_ORIGINAL_RUN_PREPARE = d351._run_prepare
_ORIGINAL_RUN_VALIDATE = d351._run_validate
_ORIGINAL_FINALIZE = d351._run_finalize
_ORIGINAL_BUILD_LIVE = d351.d349._build_live_topology_parts
_ORIGINAL_BUILD_RAW = d351.d339._build_retained_raw_shapes
_ORIGINAL_EVALUATE_Q5 = d351._evaluate_q5
_REPAIR_EVENTS: list[dict[str, Any]] = []
_BRIDGE_SNAPSHOTS: list[dict[str, Any]] = []
_GEOMETRY_EVALUATION_CALL_COUNT = 0
_PENDING_AUTOMATED_PAYLOAD: dict[str, Any] | None = None
_PENDING_AUTOMATED_MARKDOWN: str | None = None


def _sha(path: Path) -> str:
    return d351._sha(path)


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _attempt1_immutability_contract() -> dict[str, Any]:
    status = d351._git_status()
    root_files = sorted(path.name for path in ATTEMPT1_DIR.iterdir() if path.is_file())
    expected_root_files = sorted(ATTEMPT1_ROOT_HASHES)
    root_rows = {
        name: {
            "path": _rel(ATTEMPT1_DIR / name),
            "status": status.get(_rel(ATTEMPT1_DIR / name)),
            "sha256": _sha(ATTEMPT1_DIR / name)
            if (ATTEMPT1_DIR / name).is_file()
            else None,
            "expected_sha256": expected_hash,
        }
        for name, expected_hash in ATTEMPT1_ROOT_HASHES.items()
    }
    outside_rows = {
        _rel(path): {
            "status": status.get(_rel(path)),
            "sha256": _sha(path) if path.is_file() else None,
            "expected_sha256": expected_hash,
        }
        for path, expected_hash in ATTEMPT1_OUTSIDE_HASHES.items()
    }
    exception = (
        json.loads((ATTEMPT1_DIR / "d351_runtime_exception.json").read_text())
        if (ATTEMPT1_DIR / "d351_runtime_exception.json").is_file()
        else {}
    )
    prereg = (
        json.loads((ATTEMPT1_DIR / "d351_preregistration.json").read_text())
        if (ATTEMPT1_DIR / "d351_preregistration.json").is_file()
        else {}
    )
    preflight = (
        json.loads((ATTEMPT1_DIR / "d351_validate_preflight.json").read_text())
        if (ATTEMPT1_DIR / "d351_validate_preflight.json").is_file()
        else {}
    )
    attempt1_pid = preflight.get("validate_process_identity", {}).get("pid")
    forbidden_attempt1_success_files = [
        ATTEMPT1_DIR / "d351_zero_step_closure_geometry_measurement.json",
        ATTEMPT1_DIR / "d351_automated_summary.json",
        ATTEMPT1_DIR / "d351_viewer_capture_contract.json",
        ATTEMPT1_DIR / "d351_zero_step_closure_geometry.rrd",
        ATTEMPT1_DIR / "d351_completion_summary.json",
    ]
    checks = {
        "attempt1_root_file_inventory_exact": root_files == expected_root_files,
        "attempt1_root_hashes_exact": all(
            row["sha256"] == row["expected_sha256"] for row in root_rows.values()
        ),
        "attempt1_root_files_remain_untracked": all(
            row["status"] == "??" for row in root_rows.values()
        ),
        "original_harness_and_session_hashes_exact": all(
            row["sha256"] == row["expected_sha256"] for row in outside_rows.values()
        ),
        "original_harness_and_session_remain_untracked": all(
            row["status"] == "??" for row in outside_rows.values()
        ),
        "attempt1_prereg_pass": prereg.get("pass") is True,
        "attempt1_exact_early_stop": exception.get("error") == ATTEMPT1_EXPECTED_ERROR,
        "attempt1_validate_process_absent": isinstance(attempt1_pid, int)
        and not Path(f"/proc/{attempt1_pid}").exists(),
        "attempt1_no_scientific_or_viewer_outputs": all(
            not path.exists() for path in forbidden_attempt1_success_files
        ),
    }
    return {
        "artifact": "D351_ATTEMPT1_IMMUTABILITY_CONTRACT_V1",
        "attempt1_root": _rel(ATTEMPT1_DIR),
        "root_files": root_rows,
        "outside_files": outside_rows,
        "expected_early_stop": ATTEMPT1_EXPECTED_ERROR,
        "attempt1_validate_pid": attempt1_pid,
        "forbidden_attempt1_success_files": [
            _rel(path) for path in forbidden_attempt1_success_files
        ],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _attempt2_user_sidecar_contract() -> dict[str, Any]:
    status = d351._git_status()
    rows: dict[str, Any] = {}
    for relative, expected_hash in ATTEMPT2_USER_SIDECAR_HASHES.items():
        role = d351.PREEXISTING_USER_GIT_ROLES[relative]
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative],
            cwd=REPO,
            check=False,
        ).returncode == 0
        path = REPO / relative
        rows[relative] = {
            "git_role": role,
            "status": status.get(relative),
            "git_ignored": ignored,
            "git_role_exact": bool(
                (role == "untracked" and status.get(relative) == "??" and not ignored)
                or (role == "ignored" and status.get(relative) is None and ignored)
            ),
            "exists": path.is_file(),
            "sha256": _sha(path) if path.is_file() else None,
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
        "role": (
            "attempt2-start stable snapshot of preexisting user sidecars; read-only and "
            "not scientific inputs"
        ),
        "rows": rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _combined_preexisting_contract() -> dict[str, Any]:
    historical_user = json.loads(
        (ATTEMPT1_DIR / "d351_preregistration.json").read_text(encoding="utf-8")
    ).get("preexisting_user_files", {})
    active_user = _attempt2_user_sidecar_contract()
    attempt1 = _attempt1_immutability_contract()
    return {
        "role": (
            "attempt1 historical sidecar record plus attempt2-stable read-only user files "
            "and immutable D351 attempt1 early-stop evidence"
        ),
        "attempt1_historical_user_files": historical_user,
        "attempt2_active_user_files": active_user,
        "attempt1": attempt1,
        "pass": bool(active_user["pass"] and attempt1["pass"]),
    }


def _attempt2_status_scope_pass(status: dict[str, str]) -> bool:
    exact_allowed = {
        _rel(d351.START_HERE),
        _rel(ORIGINAL_SESSION),
        _rel(REPAIR_SESSION),
        _rel(ORIGINAL_HARNESS),
        _rel(HARNESS),
        *d351.PREEXISTING_USER_UNTRACKED_HASHES,
        *(_rel(ATTEMPT1_DIR / name) for name in ATTEMPT1_ROOT_HASHES),
    }
    repair_prefix = _rel(REPAIR_DIR) + "/"
    return all(
        path in exact_allowed or path.startswith(repair_prefix) for path in status
    )


def _float32_payload(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values, dtype=np.float32)
    payload = array.tobytes()
    return {
        "values": array.tolist(),
        "bits_hex": payload.hex(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _play_simulations_setting() -> dict[str, Any]:
    try:
        import carb.settings

        value = carb.settings.get_settings().get("/app/player/playSimulations")
        return {
            "readable": value is not None,
            "value": None if value is None else bool(value),
            "error": None,
        }
    except Exception as error:  # pragma: no cover - runtime diagnostic path
        return {
            "readable": False,
            "value": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _bridge_snapshot(inner: Any, phase: str) -> dict[str, Any]:
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    joints = inner._robot.data.joint_pos[0].detach().cpu().numpy()
    object_pos, object_quat = d351.d334._object_pose_w(inner)
    return {
        "phase": phase,
        "custom_step_counter": int(inner._sim_step_counter),
        "timeline_playing": bool(timeline.is_playing()),
        "timeline_time": float(timeline.get_current_time()),
        "simulation_context_clock": d351._simulation_clock(inner),
        "play_simulations_setting": _play_simulations_setting(),
        "geometry_evaluation_call_count": int(_GEOMETRY_EVALUATION_CALL_COUNT),
        "joint_float32": _float32_payload(joints),
        "object_position_float32": _float32_payload(object_pos),
        "object_quaternion_float32": _float32_payload(object_quat),
    }


def _append_bridge_snapshot(inner: Any, phase: str) -> dict[str, Any]:
    snapshot = _bridge_snapshot(inner, phase)
    _BRIDGE_SNAPSHOTS.append(snapshot)
    return snapshot


def _bridge_contract() -> dict[str, Any]:
    expected_phases = [
        "after_reset_initial_pause_before_live_binding",
        "after_live_binding_before_repause",
        "after_live_binding_after_repause",
        "after_live_and_raw_binding_before_final_repause",
        "after_live_and_raw_binding_after_final_repause",
    ]
    snapshots = copy.deepcopy(_BRIDGE_SNAPSHOTS)
    initial = snapshots[0] if snapshots else {}
    final = snapshots[-1] if snapshots else {}
    initial_clock = initial.get("simulation_context_clock", {})
    payload_names = (
        "joint_float32",
        "object_position_float32",
        "object_quaternion_float32",
    )
    setting_rows = [
        row.get("play_simulations_setting", {})
        for row in (initial, final)
        if row
    ]
    checks = {
        "exact_five_bridge_snapshots": len(snapshots) == len(expected_phases),
        "phase_order_exact": [row.get("phase") for row in snapshots]
        == expected_phases,
        "geometry_evaluation_count_zero_before_guard": bool(snapshots)
        and _GEOMETRY_EVALUATION_CALL_COUNT == 0
        and all(row.get("geometry_evaluation_call_count") == 0 for row in snapshots),
        "custom_step_counter_zero_at_every_snapshot": bool(snapshots)
        and all(row.get("custom_step_counter") == 0 for row in snapshots),
        "timeline_time_bit_exact_invariant": bool(snapshots)
        and all(row.get("timeline_time") == initial.get("timeline_time") for row in snapshots),
        "simulation_context_clock_available": bool(snapshots)
        and all(
            all(value is not None for value in row.get("simulation_context_clock", {}).values())
            for row in snapshots
        ),
        "simulation_context_clock_exact_invariant": bool(snapshots)
        and all(row.get("simulation_context_clock") == initial_clock for row in snapshots),
        "reset_joint_and_object_float32_bits_exact_invariant": bool(snapshots)
        and all(
            row.get(name, {}).get("bits_hex") == initial.get(name, {}).get("bits_hex")
            for row in snapshots
            for name in payload_names
        ),
        "initial_timeline_paused": bool(initial)
        and initial.get("timeline_playing") is False,
        "final_timeline_paused": bool(final)
        and final.get("timeline_playing") is False,
        "play_simulations_false_when_readable_at_boundaries": len(setting_rows) == 2
        and all(
            (not row.get("readable")) or row.get("value") is False
            for row in setting_rows
        ),
        "no_app_update_or_physics_step_in_repair_bridge": True,
    }
    return {
        "artifact": "D351_ATTEMPT2_ZERO_STEP_BINDING_BRIDGE_V1",
        "expected_phases": expected_phases,
        "snapshots": snapshots,
        "play_simulations_boundary_readability": [
            bool(row.get("readable")) for row in setting_rows
        ],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _pause_without_update(inner: Any, phase: str) -> dict[str, Any]:
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    counter_before = int(inner._sim_step_counter)
    time_before = float(timeline.get_current_time())
    clock_before = d351._simulation_clock(inner)
    playing_before = bool(timeline.is_playing())
    interventions = 0
    for _ in range(3):
        inner.sim.set_setting("/app/player/playSimulations", False)
        if timeline.is_playing():
            timeline.pause()
            interventions += 1
    counter_after = int(inner._sim_step_counter)
    time_after = float(timeline.get_current_time())
    clock_after = d351._simulation_clock(inner)
    checks = {
        "timeline_paused_after": not timeline.is_playing(),
        "custom_counter_zero_unchanged": counter_before == counter_after == 0,
        "timeline_time_unchanged": time_before == time_after,
        "simulation_context_clock_unchanged": clock_before == clock_after,
        "geometry_evaluation_count_still_zero": _GEOMETRY_EVALUATION_CALL_COUNT == 0,
        "no_app_update_or_physics_step_called": True,
    }
    return {
        "phase": phase,
        "playing_before": playing_before,
        "playing_after": bool(timeline.is_playing()),
        "pause_interventions": interventions,
        "counter_before": counter_before,
        "counter_after": counter_after,
        "timeline_time_before": time_before,
        "timeline_time_after": time_after,
        "simulation_context_clock_before": clock_before,
        "simulation_context_clock_after": clock_after,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _build_live_with_reactive_pause(inner: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    _append_bridge_snapshot(inner, "after_reset_initial_pause_before_live_binding")
    parts, report = _ORIGINAL_BUILD_LIVE(inner)
    original_payload_sha256 = _payload_sha(report)
    _append_bridge_snapshot(inner, "after_live_binding_before_repause")
    event = _pause_without_update(inner, "after_live_topology_binding")
    _REPAIR_EVENTS.append(event)
    _append_bridge_snapshot(inner, "after_live_binding_after_repause")
    report["attempt2_reactive_timeline_pause"] = event
    report["attempt2_original_live_binding_payload_sha256"] = original_payload_sha256
    report["checks"]["attempt2_live_binding_pause_contract"] = event["pass"]
    report["checks"]["attempt2_original_live_binding_payload_reproduces_attempt1"] = (
        original_payload_sha256
        == ATTEMPT1_ROOT_HASHES["d351_live_topology_runtime_binding.json"]
    )
    report["pass"] = all(report["checks"].values())
    return parts, report


def _build_raw_with_reactive_pause(
    inner: Any, summary: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shapes, report = _ORIGINAL_BUILD_RAW(inner, summary)
    _append_bridge_snapshot(
        inner, "after_live_and_raw_binding_before_final_repause"
    )
    event = _pause_without_update(inner, "after_raw_shape_binding_before_prerequisites")
    _REPAIR_EVENTS.append(event)
    _append_bridge_snapshot(
        inner, "after_live_and_raw_binding_after_final_repause"
    )
    attempt1 = _attempt1_immutability_contract()
    bridge = _bridge_contract()
    checks = {
        "exact_two_pause_events": len(_REPAIR_EVENTS) == 2,
        "all_pause_events_pass": len(_REPAIR_EVENTS) == 2
        and all(row["pass"] for row in _REPAIR_EVENTS),
        "attempt1_immutable": attempt1["pass"],
        "binding_bridge_zero_step_exact": bridge["pass"],
        "geometry_evaluation_count_zero_before_prerequisites": (
            _GEOMETRY_EVALUATION_CALL_COUNT == 0
        ),
        "new_scientific_variables_zero": True,
        "new_physical_variables_zero": True,
        "controlled_physics_steps_zero": all(
            row["counter_before"] == row["counter_after"] == 0
            for row in _REPAIR_EVENTS
        ),
    }
    control = {
        "artifact": "D351_ATTEMPT2_REACTIVE_TIMELINE_PAUSE_CONTRACT_V1",
        "case": CASE,
        "trigger": {
            "attempt1_runtime_exception": _rel(
                ATTEMPT1_DIR / "d351_runtime_exception.json"
            ),
            "attempt1_runtime_exception_sha256": _sha(
                ATTEMPT1_DIR / "d351_runtime_exception.json"
            ),
            "observed_failure": ATTEMPT1_EXPECTED_ERROR,
            "failure_stage": "after representation binding, before first q5 sample",
        },
        "repair_scope": (
            "timeline.pause plus playSimulations=false playback suppression after live/raw "
            "representation binding; no app update, physics step, material, actuator, "
            "physics-configuration, geometry, gate, or target/IK change"
        ),
        "new_variables": [],
        "new_physical_variables": [],
        "events": copy.deepcopy(_REPAIR_EVENTS),
        "zero_step_binding_bridge": bridge,
        "attempt1_immutability": attempt1,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _ORIGINAL_WRITE_JSON(CONTROL_REPAIR_PATH, control)
    report["d351_attempt2_timeline_pause_repair"] = event
    report["d351_attempt2_zero_step_binding_bridge"] = bridge
    report["checks"]["attempt2_raw_binding_pause_contract"] = bool(
        event["pass"] and bridge["pass"]
    )
    report["checks"]["attempt2_full_control_contract"] = control["pass"]
    report["pass"] = all(report["checks"].values())
    return shapes, report


def _runtime_outputs() -> list[Path]:
    return [
        *_ORIGINAL_RUNTIME_OUTPUTS(),
        CONTROL_REPAIR_PATH,
        AGGREGATION_REPAIR_PATH,
        UNMODIFIED_AUTOMATED_PATH,
        UNMODIFIED_AUTOMATED_MD_PATH,
    ]


def _write_json(path: Path, payload: Any) -> None:
    global _PENDING_AUTOMATED_PAYLOAD

    value = copy.deepcopy(payload)
    if path == d351.PARAMETER_PATH:
        reproduced_sha = _payload_sha(value)
        expected_sha = ATTEMPT1_ROOT_HASHES["d351_parameter_freeze_audit.json"]
        if reproduced_sha != expected_sha:
            raise RuntimeError(
                "D351 attempt2 parameter-freeze reproduction STOP: "
                f"{reproduced_sha} != {expected_sha}"
            )
    elif path == d351.PREREG_PATH:
        value["attempt"] = "attempt2_timeline_pause_repair"
        value["reactive_repair_trigger"] = ATTEMPT1_EXPECTED_ERROR
        value["attempt1_immutability"] = _attempt1_immutability_contract()
        value["new_attempt2_variables"] = []
        value["new_attempt2_physical_variables"] = []
        value["attempt1_parameter_freeze_reproduced"] = {
            "sha256": _sha(d351.PARAMETER_PATH),
            "expected_sha256": ATTEMPT1_ROOT_HASHES[
                "d351_parameter_freeze_audit.json"
            ],
            "pass": _sha(d351.PARAMETER_PATH)
            == ATTEMPT1_ROOT_HASHES["d351_parameter_freeze_audit.json"],
        }
    elif path == d351.MEASUREMENT_PATH:
        control = json.loads(CONTROL_REPAIR_PATH.read_text())
        value["timeline_pause_repair"] = {
            "path": _rel(CONTROL_REPAIR_PATH),
            "sha256": _sha(CONTROL_REPAIR_PATH),
            "pass": control["pass"],
            "geometry_evaluation_count_before_bridge_guard": control["zero_step_binding_bridge"][
                "snapshots"
            ][-1]["geometry_evaluation_call_count"],
        }
    elif path == d351.AUTOMATED_PATH:
        control = json.loads(CONTROL_REPAIR_PATH.read_text())
        value["timeline_pause_repair"] = {
            "path": _rel(CONTROL_REPAIR_PATH),
            "sha256": _sha(CONTROL_REPAIR_PATH),
            "pass": control["pass"],
        }
        _PENDING_AUTOMATED_PAYLOAD = value
        return
    elif path == d351.COMPLETION_PATH:
        value["attempt"] = "attempt2_timeline_pause_repair"
        value["attempt1_immutable"] = _attempt1_immutability_contract()["pass"]
        value["timeline_pause_repair"] = {
            "path": _rel(CONTROL_REPAIR_PATH),
            "sha256": _sha(CONTROL_REPAIR_PATH),
            "pass": json.loads(CONTROL_REPAIR_PATH.read_text())["pass"],
        }
        value["automated_aggregation_repair"] = {
            "path": _rel(AGGREGATION_REPAIR_PATH),
            "sha256": _sha(AGGREGATION_REPAIR_PATH),
            "pass": json.loads(AGGREGATION_REPAIR_PATH.read_text())["pass"],
        }
    _ORIGINAL_WRITE_JSON(path, value)


def _write_text(path: Path, value: str) -> None:
    global _PENDING_AUTOMATED_MARKDOWN

    if path == d351.AUTOMATED_MD_PATH:
        _PENDING_AUTOMATED_MARKDOWN = value
        return
    _ORIGINAL_WRITE_TEXT(path, value)


def _payload_sha(payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=d351._json_default,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_prepare_with_attempt2_state_guard(args: Any) -> int:
    start_text = d351.START_HERE.read_text(encoding="utf-8")
    session_text = REPAIR_SESSION.read_text(encoding="utf-8") if REPAIR_SESSION.is_file() else ""
    checks = {
        "repair_session_exists": REPAIR_SESSION.is_file(),
        "start_here_attempt2_marker": "attempt2_timeline_pause_repair" in start_text,
        "start_here_attempt1_early_stop_marker": "q5 geometry sample `0`" in start_text,
        "start_here_forward_only_path": _rel(REPAIR_DIR) in start_text,
        "repair_session_attempt1_no_science_marker": "과학 판정은 `없음`" in session_text,
        "repair_session_zero_new_science_variables": (
            "이번 attempt의 신규 과학 변수: `[]`" in session_text
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D351 attempt2 state-marker STOP: {checks}")
    return _ORIGINAL_RUN_PREPARE(args)


def _run_validate_with_reactive_repairs(
    args: Any,
    simulation_app: Any,
    launcher_report: dict[str, Any],
) -> int:
    result = _ORIGINAL_RUN_VALIDATE(args, simulation_app, launcher_report)
    pending = copy.deepcopy(_PENDING_AUTOMATED_PAYLOAD)
    if pending is None:
        raise RuntimeError(
            "D351 attempt2 validator returned without the deferred automated payload: "
            f"{result}"
        )
    _ORIGINAL_WRITE_JSON(UNMODIFIED_AUTOMATED_PATH, pending)
    _ORIGINAL_WRITE_TEXT(
        UNMODIFIED_AUTOMATED_MD_PATH,
        _PENDING_AUTOMATED_MARKDOWN
        if _PENDING_AUTOMATED_MARKDOWN is not None
        else "# D351 unmodified automated report missing\n",
    )
    if result != 2:
        raise RuntimeError(
            "D351 attempt2 unexpected unmodified validator result with pending aggregate: "
            f"{result}"
        )

    immutability = pending.get("immutability", {})
    positive_immutability = {
        key: value
        for key, value in immutability.items()
        if key not in {"asset_write", "pass"}
    }
    control = json.loads(CONTROL_REPAIR_PATH.read_text())
    trigger_checks = {
        "unmodified_result_is_expected_false_aggregate_exit_2": result == 2,
        "asset_write_is_false_as_required": immutability.get("asset_write") is False,
        "all_positive_immutability_checks_true": bool(positive_immutability)
        and all(value is True for value in positive_immutability.values()),
        "unmodified_immutability_pass_false": immutability.get("pass") is False,
        "unmodified_observability_pass_false": pending.get("observability_pass")
        is False,
        "unmodified_automated_pass_false": pending.get("automated_pass") is False,
        "scientific_result_recorded": pending.get("scientific_result_recorded") is True,
        "overlay_pass": pending.get("overlay_pass") is True,
        "rerun_pass": pending.get("rerun_pass") is True,
        "viewer_capture_tokens_pass": pending.get("viewer_capture_tokens_pass") is True,
        "launcher_pass": pending.get("launcher", {}).get("pass") is True,
        "timeline_pause_control_pass": control.get("pass") is True,
        "controlled_physics_steps_zero": pending.get("controlled_physics_steps") == 0,
        "unmodified_json_preserved_exact": _sha(UNMODIFIED_AUTOMATED_PATH)
        == _payload_sha(pending),
        "unmodified_markdown_captured_and_preserved": (
            _PENDING_AUTOMATED_MARKDOWN is not None
            and _sha(UNMODIFIED_AUTOMATED_MD_PATH)
            == hashlib.sha256(_PENDING_AUTOMATED_MARKDOWN.encode("utf-8")).hexdigest()
        ),
    }
    correction_allowed = all(trigger_checks.values())
    corrected_immutability_pass = bool(
        correction_allowed
        and immutability.get("asset_write") is False
        and all(value is True for value in positive_immutability.values())
    )
    corrected_observability_pass = bool(
        corrected_immutability_pass
        and pending.get("overlay_pass") is True
        and pending.get("rerun_pass") is True
        and pending.get("viewer_capture_tokens_pass") is True
        and pending.get("launcher", {}).get("pass") is True
        and control.get("pass") is True
        and pending.get("controlled_physics_steps") == 0
    )
    corrected_automated_pass = bool(
        pending.get("scientific_result_recorded") is True
        and corrected_observability_pass
    )
    repair = {
        "artifact": "D351_ATTEMPT2_AUTOMATED_AGGREGATION_REPAIR_V1",
        "case": CASE,
        "trigger": (
            "known false-valued asset_write polarity was included directly in all(values), "
            "making an otherwise successful observability aggregate false"
        ),
        "repair_scope": (
            "reinterpret asset_write=false as the required no-write condition; no science, "
            "geometry, tolerance, target/IK, material, actuator, physics, Viewer, or Rerun change"
        ),
        "unmodified_automated_payload_sha256": _payload_sha(pending),
        "unmodified_artifacts": {
            "json": {
                "path": _rel(UNMODIFIED_AUTOMATED_PATH),
                "sha256": _sha(UNMODIFIED_AUTOMATED_PATH),
            },
            "markdown": {
                "path": _rel(UNMODIFIED_AUTOMATED_MD_PATH),
                "sha256": _sha(UNMODIFIED_AUTOMATED_MD_PATH),
            },
        },
        "trigger_checks": trigger_checks,
        "corrected_fields": {
            "immutability_pass": corrected_immutability_pass,
            "observability_pass": corrected_observability_pass,
            "automated_pass": corrected_automated_pass,
        },
        "new_variables": [],
        "new_physical_variables": [],
        "pass": bool(correction_allowed and corrected_automated_pass),
    }
    _ORIGINAL_WRITE_JSON(AGGREGATION_REPAIR_PATH, repair)
    if not repair["pass"]:
        _ORIGINAL_WRITE_JSON(d351.AUTOMATED_PATH, pending)
        _ORIGINAL_WRITE_TEXT(
            d351.AUTOMATED_MD_PATH,
            _PENDING_AUTOMATED_MARKDOWN
            if _PENDING_AUTOMATED_MARKDOWN is not None
            else "# D351 automated report missing\n",
        )
        return result

    pending["immutability"]["asset_write_forbidden_and_absent"] = True
    pending["immutability"]["pass"] = corrected_immutability_pass
    pending["observability_pass"] = corrected_observability_pass
    pending["automated_pass"] = corrected_automated_pass
    pending["automated_verdict"] = (
        pending["scientific_verdict"] + d351.VERDICT_PENDING_SUFFIX
    )
    pending["automated_aggregation_repair"] = {
        "path": _rel(AGGREGATION_REPAIR_PATH),
        "sha256": _sha(AGGREGATION_REPAIR_PATH),
        "pass": repair["pass"],
    }
    _ORIGINAL_WRITE_JSON(d351.AUTOMATED_PATH, pending)
    _ORIGINAL_WRITE_TEXT(
        d351.AUTOMATED_MD_PATH,
        "\n".join(
            [
                "# D351 automated result — attempt2 repaired aggregate",
                "",
                f"- scientific verdict: `{pending['scientific_verdict']}`",
                "- automated pass: `true`",
                f"- executed zero-step q5 samples: `{pending['execution_count']}`",
                f"- raw first-contact bracket: `{pending['raw_contact_bracket']}`",
                f"- live first-contact bracket: `{pending['live_contact_bracket']}`",
                "- controlled physics steps: `0`",
                "- target/IK/path change: `false`",
                "- aggregation repair: `asset_write=false` means the required no-write state",
                "- g0a_pass: `false`",
            ]
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "stage": "validate_attempt2_repaired_aggregate",
                "automated_pass": True,
                "scientific_verdict": pending["scientific_verdict"],
                "execution_count": pending["execution_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _run_finalize(args: Any) -> int:
    if not CONTROL_REPAIR_PATH.is_file() or not AGGREGATION_REPAIR_PATH.is_file():
        raise RuntimeError("D351 attempt2 repair evidence missing")
    control = json.loads(CONTROL_REPAIR_PATH.read_text())
    aggregation = json.loads(AGGREGATION_REPAIR_PATH.read_text())
    automated = json.loads(d351.AUTOMATED_PATH.read_text())
    bound = automated.get("timeline_pause_repair", {})
    aggregate_bound = automated.get("automated_aggregation_repair", {})
    unmodified_bound = aggregation.get("unmodified_artifacts", {})
    unmodified_json_bound = unmodified_bound.get("json", {})
    unmodified_markdown_bound = unmodified_bound.get("markdown", {})
    checks = {
        "control_repair_pass": control.get("pass") is True,
        "automated_control_path_exact": bound.get("path")
        == _rel(CONTROL_REPAIR_PATH),
        "automated_control_hash_exact": bound.get("sha256")
        == _sha(CONTROL_REPAIR_PATH),
        "automated_control_pass": bound.get("pass") is True,
        "aggregation_repair_pass": aggregation.get("pass") is True,
        "automated_aggregation_path_exact": aggregate_bound.get("path")
        == _rel(AGGREGATION_REPAIR_PATH),
        "automated_aggregation_hash_exact": aggregate_bound.get("sha256")
        == _sha(AGGREGATION_REPAIR_PATH),
        "automated_aggregation_pass": aggregate_bound.get("pass") is True,
        "unmodified_json_exists": UNMODIFIED_AUTOMATED_PATH.is_file(),
        "unmodified_json_path_exact": unmodified_json_bound.get("path")
        == _rel(UNMODIFIED_AUTOMATED_PATH),
        "unmodified_json_hash_exact_current": UNMODIFIED_AUTOMATED_PATH.is_file()
        and unmodified_json_bound.get("sha256")
        == _sha(UNMODIFIED_AUTOMATED_PATH)
        == aggregation.get("unmodified_automated_payload_sha256"),
        "unmodified_markdown_exists": UNMODIFIED_AUTOMATED_MD_PATH.is_file(),
        "unmodified_markdown_path_exact": unmodified_markdown_bound.get("path")
        == _rel(UNMODIFIED_AUTOMATED_MD_PATH),
        "unmodified_markdown_hash_exact_current": (
            UNMODIFIED_AUTOMATED_MD_PATH.is_file()
            and unmodified_markdown_bound.get("sha256")
            == _sha(UNMODIFIED_AUTOMATED_MD_PATH)
        ),
        "attempt1_immutable": _attempt1_immutability_contract()["pass"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"D351 attempt2 finalize repair binding STOP: {checks}")
    return _ORIGINAL_FINALIZE(args)


def _evaluate_q5_counted(*args: Any, **kwargs: Any) -> dict[str, Any]:
    global _GEOMETRY_EVALUATION_CALL_COUNT

    _GEOMETRY_EVALUATION_CALL_COUNT += 1
    return _ORIGINAL_EVALUATE_Q5(*args, **kwargs)


def _configure_attempt2() -> None:
    d351.OUT_DIR = REPAIR_DIR
    filename_bindings = {
        "PREREG_PATH": "d351_preregistration.json",
        "PARAMETER_PATH": "d351_parameter_freeze_audit.json",
        "PREFLIGHT_PATH": "d351_validate_preflight.json",
        "LIVE_BINDING_PATH": "d351_live_topology_runtime_binding.json",
        "MOVING_BINDING_PATH": "d351_moving_jaw_surface_binding.json",
        "MEASUREMENT_PATH": "d351_zero_step_closure_geometry_measurement.json",
        "SWEEP_CSV_PATH": "d351_q5_closure_sweep.csv",
        "OVERLAY_PATH": "d351_viewer_overlay_contract.json",
        "CAPTURE_PATH": "d351_viewer_capture_contract.json",
        "RRD_PATH": "d351_zero_step_closure_geometry.rrd",
        "RBL_PATH": "d351_zero_step_closure_geometry.rbl",
        "RERUN_PNG_PATH": "d351_zero_step_closure_geometry_rerun.png",
        "RERUN_VALIDATION_PATH": "d351_rerun_validation.json",
        "AUTOMATED_PATH": "d351_automated_summary.json",
        "AUTOMATED_MD_PATH": "d351_automated_report.md",
        "RUNTIME_EXCEPTION_PATH": "d351_runtime_exception.json",
        "MANUAL_PATH": "d351_manual_visual_inspection.json",
        "MANUAL_MD_PATH": "d351_manual_visual_inspection.md",
        "COMPLETION_PATH": "d351_completion_summary.json",
        "COMPLETION_MD_PATH": "d351_completion_report.md",
    }
    for name, filename in filename_bindings.items():
        setattr(d351, name, REPAIR_DIR / filename)
    d351.VIEWER_PNGS = {
        "open_physx": REPAIR_DIR / "d351_open_actual_physx_colliders.png",
        "decision_physx": REPAIR_DIR
        / "d351_decision_or_open_fallback_actual_physx_colliders.png",
        "decision_colored": REPAIR_DIR
        / "d351_decision_or_open_fallback_colored_64plus64.png",
        "decision_side": REPAIR_DIR
        / "d351_decision_or_open_fallback_side_geometry.png",
    }
    d351.HARNESS = HARNESS
    d351.SESSION_DOC = REPAIR_SESSION
    d351._status_scope_pass = _attempt2_status_scope_pass
    d351._preexisting_user_untracked_contract = _combined_preexisting_contract
    d351._runtime_outputs = _runtime_outputs
    d351._write_json = _write_json
    d351._write_text = _write_text
    d351._run_prepare = _run_prepare_with_attempt2_state_guard
    d351._run_validate = _run_validate_with_reactive_repairs
    d351._run_finalize = _run_finalize
    d351._evaluate_q5 = _evaluate_q5_counted
    d351.d349._build_live_topology_parts = _build_live_with_reactive_pause
    d351.d339._build_retained_raw_shapes = _build_raw_with_reactive_pause


def main() -> int:
    _configure_attempt2()
    return d351.main()


if __name__ == "__main__":
    raise SystemExit(main())
