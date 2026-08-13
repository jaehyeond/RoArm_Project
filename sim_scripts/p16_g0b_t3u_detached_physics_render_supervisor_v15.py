#!/usr/bin/env python3
"""Forward-only detached supervisor for p16 physics -> trace-render.

This process must itself be launched by ``nohup setsid`` from a host-authorized
``require_escalated`` command, never from Codex's ``bwrap --die-with-parent`` sandbox.
It performs exactly one
physics attempt and, only after raw-zero plus a complete preclose semantic gate, exactly
one render attempt.  Render success likewise requires raw-zero plus independently
recomputed post-hoc artifacts.  Raw ``waitpid`` remains lifecycle authority; every
timeout/signal, PID/SID/PGID, GPU inventory, and bound file hash is retained for p16's
terminal attestor.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
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

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"
P16_PATH = REPO / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v12.py"
ISAAC_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
CANDIDATES_PATH = CASE_DIR / "t3s_side_sdg2_candidates.json"
CANDIDATES_SHA256 = "67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384"
PHYSICS_TIMEOUT_S = 7200.0
RENDER_TIMEOUT_S = 7200.0
KILL_AFTER_S = 20.0
QUIET_REAP_GRACE_S = 5.0
BOUNDED_REAP_S = 20.0
SEMANTIC_GATE_FAILURE_EXIT_STATUS = 125
WALL_TIME_ABS_TOLERANCE_S = 1.0e-3
FFPROBE_PATH = Path("/home/cgxr/.local/bin/ffprobe")
FFMPEG_PATH = Path("/home/cgxr/.local/bin/ffmpeg")
RENDER_DECODE_TIMEOUT_S = 120.0

_ACTIVE_CHILD: dict[str, Any] = {}
_LAST_CHILD_OUTCOME: dict[str, Any] | None = None
_SUPERVISOR_SIGNAL: dict[str, Any] | None = None
_SUPERVISOR_CLEANUP_ACTIONS: list[dict[str, Any]] = []


class SupervisorTerminationRequested(RuntimeError):
    pass


class HostExecutionBoundaryError(RuntimeError):
    pass

CORE_SUFFIXES = (
    "results.json", "plan.json", "trace.npz", "timeline.rrd", "timeline.rbl",
    "rerun_validation.json", "decision_snapshot.png", "inspection.png",
    "rgb_frames_manifest.json", "side_grasp.mp4", "script.py.txt", "argv.txt",
    "phase.jsonl", "render_phase.jsonl", "preclose_sentinel.json",
    "terminal_attestation.json",
    "manual_visual_inspection.json", "failure.json", "render_failure.json",
    "exit_status.txt",
)
EXTERNAL_SUFFIXES = (
    "stdout.log", "supervisor_pid.txt", "physics_python_pid.txt",
    "render_python_pid.txt", "pgid.txt", "supervisor_contract.json",
    "supervisor_outcome.json", "nvidia_smi_before.csv",
    "nvidia_smi_supervisor_end.csv", "nvidia_smi_after.csv",
    "supervisor_failure.json",
)

PHYSICS_PRECLOSE_REQUIRED_SUFFIXES = (
    "results.json", "plan.json", "trace.npz", "timeline.rrd", "timeline.rbl",
    "rerun_validation.json", "decision_snapshot.png", "inspection.png",
    "script.py.txt", "argv.txt", "phase.jsonl", "preclose_sentinel.json",
)
PHYSICS_PRECLOSE_REQUIRED_PHASES = (
    "run_claim", "results_durable", "preclose_sentinel_durable",
    "simulation_app_close_start",
)
RENDER_REQUIRED_SUFFIXES = (
    "results.json", "plan.json", "trace.npz", "rgb_frames_manifest.json",
    "side_grasp.mp4", "script.py.txt", "phase.jsonl", "render_phase.jsonl",
    "preclose_sentinel.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes_x(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_x(path: Path, payload: Any) -> None:
    write_bytes_x(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _finite_time_unix(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _strict_int(value: Any) -> bool:
    return type(value) is int


FORBIDDEN_LAUNCH_ANCESTOR_TOKENS = (
    "bwrap", "--die-with-parent", "codex-linux-sandbox",
)


def _proc_parent_pid(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = text.rfind(")")
    if close < 0:
        raise RuntimeError(f"HOST_BOUNDARY_PROC_STAT_SHAPE_FAIL pid={pid}")
    fields = text[close + 2 :].split()
    if len(fields) < 2:
        raise RuntimeError(f"HOST_BOUNDARY_PROC_STAT_FIELDS_FAIL pid={pid}")
    return int(fields[1])


def _proc_argv(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [
        token.decode("utf-8", errors="replace")
        for token in raw.split(b"\0")
        if token
    ]


def _assert_host_execution_boundary() -> None:
    """Fail before lifecycle/science writes inside a die-with-parent sandbox.

    In the managed sandbox, bubblewrap is PID 1 in the child PID namespace; on a
    host-authorized launch PID 1 is the host init.  We also walk the visible parent
    chain so the guard remains effective without a PID namespace.  This is a launch
    safety check, not scientific evidence.
    """
    rows: list[tuple[int, list[str]]] = []
    pid = os.getpid()
    seen: set[int] = set()
    for _ in range(64):
        if pid <= 0 or pid in seen:
            break
        seen.add(pid)
        argv = _proc_argv(pid)
        rows.append((pid, argv))
        if pid == 1:
            break
        pid = _proc_parent_pid(pid)
    if not rows or rows[-1][0] != 1:
        raise HostExecutionBoundaryError(
            f"HOST_EXECUTION_BOUNDARY_ANCESTRY_INCOMPLETE rows={rows!r}"
        )
    forbidden: list[dict[str, Any]] = []
    for row_pid, argv in rows:
        executable = Path(argv[0]).name if argv else ""
        if executable in {"bwrap", "codex-linux-sandbox"}:
            forbidden.append({"pid": row_pid, "token": executable})
        if "--die-with-parent" in argv:
            forbidden.append({"pid": row_pid, "token": "--die-with-parent"})
    if forbidden:
        raise HostExecutionBoundaryError(
            "HOST_EXECUTION_REQUIRED__BWRAP_DIE_WITH_PARENT_FORBIDDEN "
            f"matches={forbidden!r}"
        )


def _self_pid_namespace_evidence() -> dict[str, Any]:
    """Read only the supervisor's two equivalent procfs namespace aliases.

    This deliberately makes no claim about PID 1's namespace.  The sandbox rejection
    authority is the PID-1 command line plus the complete visible ancestor walk in
    ``_assert_host_execution_boundary``.
    """
    own_pid = os.getpid()
    self_path = "/proc/self/ns/pid"
    own_path = f"/proc/{own_pid}/ns/pid"
    self_readlink = os.readlink(self_path)
    own_readlink = os.readlink(own_path)
    self_stat = os.stat(self_path)
    own_stat = os.stat(own_path)
    self_device = int(self_stat.st_dev)
    own_device = int(own_stat.st_dev)
    self_inode = int(self_stat.st_ino)
    own_inode = int(own_stat.st_ino)
    consistent = bool(
        own_pid > 1
        and self_device > 0
        and own_device > 0
        and self_device == own_device
        and self_inode > 0
        and own_inode > 0
        and self_inode == own_inode
        and self_readlink == own_readlink
        and self_readlink == f"pid:[{self_inode}]"
    )
    evidence = {
        "supervisor_pid": own_pid,
        "self_pid_namespace_path": self_path,
        "own_pid_namespace_path": own_path,
        "self_pid_namespace_readlink": self_readlink,
        "own_pid_namespace_readlink": own_readlink,
        "self_pid_namespace_device": self_device,
        "own_pid_namespace_device": own_device,
        "self_pid_namespace_inode": self_inode,
        "own_pid_namespace_inode": own_inode,
        "supervisor_self_namespace_consistent": consistent,
        "pid_namespace_evidence_scope": (
            "supervisor_self_and_own_pid_alias_only__not_pid1_namespace_comparison"
        ),
        "namespace_consistency_is_not_pid1_or_host_proof": True,
    }
    if not consistent:
        raise HostExecutionBoundaryError(
            f"SUPERVISOR_SELF_PID_NAMESPACE_INCONSISTENT {evidence!r}"
        )
    return evidence


def _host_launch_context() -> dict[str, Any]:
    """Capture accessible host context after the ancestor guard passed."""
    raw_pid1 = Path("/proc/1/cmdline").read_bytes()
    cmdline = raw_pid1.replace(b"\0", b" ").decode("utf-8", errors="replace")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    namespace = _self_pid_namespace_evidence()
    forbidden = [
        token for token in FORBIDDEN_LAUNCH_ANCESTOR_TOKENS if token in cmdline
    ]
    context = {
        "artifact": "T3U_HOST_LAUNCH_CONTEXT_V2",
        "authorization_boundary": "require_escalated_exec_command",
        "pid1_cmdline": cmdline,
        "pid1_cmdline_hex": raw_pid1.hex(),
        "pid1_cmdline_sha256": hashlib.sha256(raw_pid1).hexdigest(),
        **namespace,
        "sandbox_rejection_authority": (
            "pid1_cmdline_plus_complete_visible_ancestry_forbidden_token_gate"
        ),
        "boot_id": boot_id,
        "forbidden_tokens": list(FORBIDDEN_LAUNCH_ANCESTOR_TOKENS),
        "forbidden_matches": forbidden,
        "pass": bool(
            raw_pid1
            and not forbidden
            and namespace["supervisor_self_namespace_consistent"] is True
            and len(boot_id) == 36
        ),
    }
    if context["pass"] is not True:
        raise HostExecutionBoundaryError(f"HOST_LAUNCH_CONTEXT_FAIL {context!r}")
    return context


def _strict_float(value: Any) -> bool:
    return type(value) is float and math.isfinite(value)


def _physics_phase_semantics(
    profile: str,
    phase_rows: Any,
    results: dict[str, Any],
    sentinel: dict[str, Any],
    *,
    source_sha256: str,
    prereg_sha256: str,
    results_sha256: str,
    sentinel_sha256: str,
) -> dict[str, Any]:
    """Validate the immutable physics phase ledger without trusting row booleans."""
    expected_names = list(PHYSICS_PRECLOSE_REQUIRED_PHASES)
    if not isinstance(phase_rows, list):
        phase_rows = []
    names = [row.get("phase") if isinstance(row, dict) else None for row in phase_rows]
    sequence_exact = names in (
        expected_names,
        [*expected_names, "simulation_app_close_returned"],
    )
    rows_by_name = {
        row.get("phase"): row
        for row in phase_rows
        if isinstance(row, dict) and isinstance(row.get("phase"), str)
    }
    times = [row.get("time_unix") for row in phase_rows if isinstance(row, dict)]
    times_exact = bool(
        len(times) == len(phase_rows)
        and all(_finite_time_unix(value) for value in times)
        and all(float(a) <= float(b) for a, b in zip(times, times[1:]))
    )
    run_claim = rows_by_name.get("run_claim", {})
    results_row = rows_by_name.get("results_durable", {})
    sentinel_row = rows_by_name.get("preclose_sentinel_durable", {})
    close_start = rows_by_name.get("simulation_app_close_start", {})
    close_returned = rows_by_name.get("simulation_app_close_returned")
    row_checks = {
        "run_claim_exact": bool(
            isinstance(run_claim, dict)
            and set(run_claim)
            == {
                "time_unix", "phase", "profile", "source_sha256",
                "prereg_sha256", "candidates_sha256",
            }
            and run_claim.get("profile") == profile
            and run_claim.get("source_sha256") == source_sha256
            and run_claim.get("prereg_sha256") == prereg_sha256
            and run_claim.get("candidates_sha256") == CANDIDATES_SHA256
        ),
        "results_durable_exact": bool(
            isinstance(results_row, dict)
            and set(results_row)
            == {"time_unix", "phase", "results_sha256", "internal_verdict"}
            and results_row.get("results_sha256") == results_sha256
            and results_row.get("internal_verdict") == results.get("internal_verdict")
        ),
        "preclose_sentinel_durable_exact": bool(
            isinstance(sentinel_row, dict)
            and set(sentinel_row)
            == {"time_unix", "phase", "sentinel_sha256"}
            and sentinel_row.get("sentinel_sha256") == sentinel_sha256
        ),
        "simulation_app_close_start_exact": bool(
            isinstance(close_start, dict)
            and set(close_start)
            == {
                "time_unix", "phase", "sentinel_sha256",
                "failure_marker_exists",
            }
            and close_start.get("sentinel_sha256") == sentinel_sha256
            and close_start.get("failure_marker_exists") is False
        ),
        "optional_simulation_app_close_returned_exact": bool(
            close_returned is None
            or (
                isinstance(close_returned, dict)
                and set(close_returned)
                == {"time_unix", "phase", "sentinel_sha256"}
                and close_returned.get("sentinel_sha256") == sentinel_sha256
            )
        ),
        "sentinel_prereg_and_source_exact": bool(
            sentinel.get("source_sha256") == source_sha256
            and sentinel.get("prereg_sha256") == prereg_sha256
        ),
    }
    checks = {
        "phase_sequence_exact": sequence_exact,
        "phase_times_finite_nondecreasing": times_exact,
        **row_checks,
    }
    return {
        "phase_names": names,
        "phase_times_unix": times,
        "checks": checks,
        "pass": bool(checks and all(checks.values())),
    }


def _physics_preclose_semantic_gate(
    profile: str,
    prefix: str,
    paths: dict[str, Path],
    physics: dict[str, Any],
    stdout_path: Path,
) -> dict[str, Any]:
    """Distinguish Kit's terminal raw-zero from a completed p16 preclose.

    ``SimulationApp.close`` may terminate the interpreter with raw status zero
    even after p16 has raised and durably written ``failure.json``.  Rendering is
    therefore admitted only by raw child success *and* independently bound
    preclose artifacts.  This gate never creates or repairs a physics artifact.
    """
    raw_child_success = bool(
        physics.get("wifexited") is True
        and _strict_int(physics.get("raw_wait_status"))
        and physics.get("raw_wait_status") == 0
        and _strict_int(physics.get("exit_code"))
        and physics.get("exit_code") == 0
        and _strict_int(physics.get("normalized_returncode"))
        and physics.get("normalized_returncode") == 0
        and physics.get("timed_out") is False
        and physics.get("signal_actions") == []
        and physics.get("group_reaped") is True
    )
    failure_path = paths["failure.json"]
    required_rows: dict[str, Any] = {}
    for suffix in PHYSICS_PRECLOSE_REQUIRED_SUFFIXES:
        path = paths[suffix]
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        required_rows[suffix] = {
            "path": str(path.relative_to(REPO)),
            "exists": exists,
            "size_bytes": size,
            "sha256": sha256_file(path) if exists else None,
            "nonempty": bool(exists and int(size or 0) > 0),
        }
    missing_or_empty = [
        suffix for suffix, row in required_rows.items() if not row["nonempty"]
    ]
    semantic_checks: dict[str, bool] = {
        "all_required_preclose_files_nonempty": not missing_or_empty,
        "failure_marker_absent": not failure_path.exists(),
    }
    stdout = stdout_path.read_text(errors="replace") if stdout_path.is_file() else ""
    preclose_lines = [
        line for line in stdout.splitlines()
        if line.startswith("[p16_t3u_side] PRECLOSE ")
    ]
    expected_denominator = 5 if profile == "side_preflight13" else 10
    parsed: dict[str, Any] = {}
    parse_error: str | None = None
    if not missing_or_empty:
        try:
            p16 = _load_p16_semantic_validator()
            results = json.loads(paths["results.json"].read_text())
            plan = json.loads(paths["plan.json"].read_text())
            result_semantic_checks = p16.validate_result_semantics(
                profile, paths, results, plan
            )
            result_semantic_keys = set(p16.RESULT_SEMANTIC_CHECK_KEYS)
            sentinel = json.loads(paths["preclose_sentinel.json"].read_text())
            rerun_validation = json.loads(paths["rerun_validation.json"].read_text())
            physics_argv = paths["argv.txt"].read_text().splitlines()
            phase_rows = [
                json.loads(line)
                for line in paths["phase.jsonl"].read_text().splitlines()
                if line.strip()
            ]
            phase_names = [row.get("phase") for row in phase_rows]
            expected_argv = [
                str(P16_PATH), "--run_label", profile,
                "--candidates_sha256", CANDIDATES_SHA256,
            ]
            prereg_path = (
                p16.PREFLIGHT_PREREG
                if profile == p16.PREFLIGHT_PROFILE else p16.CANONICAL_PREREG
            )
            prereg_sha256 = sha256_file(prereg_path)
            prereg_expected_sha256 = (
                p16.PREFLIGHT_PREREG_SHA256
                if profile == p16.PREFLIGHT_PROFILE
                else p16.CANONICAL_PREREG_SHA256
            )
            results_sha256 = sha256_file(paths["results.json"])
            sentinel_sha256 = sha256_file(paths["preclose_sentinel.json"])
            phase_semantics = _physics_phase_semantics(
                profile,
                phase_rows,
                results,
                sentinel,
                source_sha256=sha256_file(P16_PATH),
                prereg_sha256=prereg_sha256,
                results_sha256=results_sha256,
                sentinel_sha256=sentinel_sha256,
            )
            success_values = results.get("metrics", {}).get("success", [])
            success_shape_exact = bool(
                isinstance(success_values, list)
                and len(success_values) == expected_denominator
                and all(isinstance(value, bool) for value in success_values)
            )
            success_numerator = (
                sum(value is True for value in success_values)
                if success_shape_exact else -1
            )
            expected_preclose_line = (
                f"[p16_t3u_side] PRECLOSE profile={profile} "
                f"verdict={results.get('scientific_verdict_preclose_candidate')} "
                f"success={success_numerator}/{expected_denominator}"
            )
            preclose_hashes = results.get("artifact_hashes_preclose", {})
            semantic_checks.update(
                {
                    "results_profile_exact": results.get("profile") == profile,
                    "plan_profile_exact": plan.get("profile") == profile,
                    "selected_prereg_matches_p16_pin": (
                        prereg_sha256 == prereg_expected_sha256
                    ),
                    "physics_argv_exact": physics_argv == expected_argv,
                    "frozen_source_exact": bool(
                        sha256_file(paths["script.py.txt"]) == sha256_file(P16_PATH)
                        == results.get("provenance", {}).get("source_sha256")
                    ),
                    "sentinel_identity_exact": bool(
                        sentinel.get("tag") == prefix
                        and sentinel.get("source_sha256") == sha256_file(P16_PATH)
                        and sentinel.get("p15_sha256") == CANDIDATES_SHA256
                    ),
                    "sentinel_result_trace_rerun_hashes_exact": bool(
                        sentinel.get("results_sha256") == sha256_file(paths["results.json"])
                        and sentinel.get("trace_sha256") == sha256_file(paths["trace.npz"])
                        and sentinel.get("rerun_validation_sha256")
                        == sha256_file(paths["rerun_validation.json"])
                    ),
                    "preclose_artifact_hashes_exact": bool(
                        all(
                            preclose_hashes.get(name) == sha256_file(paths[name])
                            for name in (
                                "plan.json", "trace.npz", "timeline.rrd",
                                "timeline.rbl", "rerun_validation.json",
                                "decision_snapshot.png", "inspection.png",
                            )
                        )
                    ),
                    "rerun_validation_pass": rerun_validation.get("pass") is True,
                    "phase_ledger_exact_schema_values_and_time": bool(
                        phase_semantics.get("pass") is True
                    ),
                    "stdout_exact_result_bound_preclose_record": bool(
                        success_shape_exact
                        and len(preclose_lines) == 1
                        and preclose_lines[0] == expected_preclose_line
                        and stdout.splitlines().count(expected_preclose_line) == 1
                        and stdout.count("[p16_t3u_side] PRECLOSE") == 1
                    ),
                    "result_plan_active_counts_exact": bool(
                        results.get("plan_counts", {}).get("feasible")
                        == expected_denominator
                        and plan.get("n_feasible") == expected_denominator
                        and plan.get("n_feasible_after_static_clearance")
                        == expected_denominator
                        and sum(
                            row.get("feasible") is True
                            for row in plan.get("trials", [])
                            if isinstance(row, dict)
                        ) == expected_denominator
                    ),
                    "pinned_result_semantic_validator_exact_all_true": bool(
                        _strict_result_semantic_checks(
                            p16, result_semantic_checks
                        )
                    ),
                }
            )
            parsed = {
                "phase_names": phase_names,
                "phase_semantics": phase_semantics,
                "results_sha256": results_sha256,
                "sentinel_sha256": sentinel_sha256,
                "trace_sha256": sha256_file(paths["trace.npz"]),
                "rerun_validation_sha256": sha256_file(paths["rerun_validation.json"]),
                "source_sha256": sha256_file(P16_PATH),
                "prereg_sha256": prereg_sha256,
                "prereg_expected_sha256": prereg_expected_sha256,
                "expected_stdout_preclose_line": expected_preclose_line,
                "stdout_preclose_line": (
                    preclose_lines[0] if len(preclose_lines) == 1 else None
                ),
                "stdout_preclose_count": len(preclose_lines),
                "result_semantic_check_keys": sorted(result_semantic_keys),
                "result_semantic_checks": result_semantic_checks,
            }
        except BaseException as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    semantic_checks["preclose_documents_parse_without_error"] = bool(
        not missing_or_empty and parse_error is None
    )
    passed = bool(raw_child_success and all(semantic_checks.values()))
    return {
        "artifact": "T3U_PHYSICS_PRECLOSE_SEMANTIC_GATE_V1",
        "profile": profile,
        "raw_child_success": raw_child_success,
        "failure_marker": {
            "path": str(failure_path.relative_to(REPO)),
            "exists": failure_path.exists(),
            "sha256": sha256_file(failure_path) if failure_path.is_file() else None,
        },
        "required_files": required_rows,
        "missing_or_empty": missing_or_empty,
        "required_phase_sequence": list(PHYSICS_PRECLOSE_REQUIRED_PHASES),
        "semantic_checks": semantic_checks,
        "parsed_bindings": parsed,
        "parse_error": parse_error,
        "pass": passed,
    }


def _load_p16_semantic_validator() -> Any:
    """Load the exact p16 file under a path-and-content-unique cache key."""
    resolved = P16_PATH.resolve()
    source_sha256 = sha256_file(resolved)
    path_sha256 = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    module_name = (
        "p16_t3u_supervisor_semantic_validator_"
        f"{path_sha256}_{source_sha256}"
    )
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if not (
            isinstance(existing_file, str)
            and Path(existing_file).resolve() == resolved
            and sha256_file(Path(existing_file)) == source_sha256
        ):
            raise RuntimeError(
                "SUPERVISOR_P16_VALIDATOR_CACHE_IDENTITY_FAIL "
                f"name={module_name} file={existing_file!r}"
            )
        return existing
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"SUPERVISOR_P16_VALIDATOR_IMPORT_SPEC_FAIL {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
    module_file = getattr(module, "__file__", None)
    if not (
        isinstance(module_file, str)
        and Path(module_file).resolve() == resolved
        and sha256_file(Path(module_file)) == source_sha256
    ):
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise RuntimeError(
            "SUPERVISOR_P16_VALIDATOR_LOADED_IDENTITY_FAIL "
            f"name={module_name} file={module_file!r}"
        )
    return module


def _strict_result_semantic_checks(p16: Any, checks: Any) -> bool:
    """Require the pinned p16 validator's complete, exact, true bool map."""
    expected = getattr(p16, "RESULT_SEMANTIC_CHECK_KEYS", None)
    return bool(
        isinstance(expected, frozenset)
        and all(isinstance(key, str) and key for key in expected)
        and isinstance(checks, dict)
        and set(checks) == expected
        and all(type(value) is bool and value is True for value in checks.values())
    )


def _independent_render_decode(
    frame_dir: Path,
    expected_names: list[str],
    mp4_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    """Actually decode every PNG and every MP4 frame outside the render child."""
    from PIL import Image

    png_errors: list[str] = []
    png_decoded = 0
    for name in expected_names:
        path = frame_dir / name
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB":
                    raise RuntimeError(
                        f"format_mode={image.format}/{image.mode}"
                    )
                if image.size != (width, height):
                    raise RuntimeError(f"size={image.size}")
            png_decoded += 1
        except BaseException as exc:
            if len(png_errors) < 10:
                png_errors.append(f"{name}:{type(exc).__name__}:{exc}")

    mp4_metadata_size: list[int] | None = None
    mp4_metadata_fps: float | None = None
    mp4_decoded = 0
    mp4_frame_byte_lengths_exact = True
    mp4_error: str | None = None
    try:
        if _SUPERVISOR_SIGNAL is not None:
            raise SupervisorTerminationRequested(
                f"supervisor received {_SUPERVISOR_SIGNAL['signal_name']} before decode"
            )
        probe = subprocess.run(
            [
                str(FFPROBE_PATH), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
                "-of", "json", str(mp4_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=RENDER_DECODE_TIMEOUT_S,
        )
        if _SUPERVISOR_SIGNAL is not None:
            raise SupervisorTerminationRequested(
                f"supervisor received {_SUPERVISOR_SIGNAL['signal_name']} after probe"
            )
        if probe.returncode != 0:
            raise RuntimeError(f"ffprobe_rc={probe.returncode}:{probe.stderr}")
        probe_doc = json.loads(probe.stdout)
        streams = probe_doc.get("streams", [])
        if not (isinstance(streams, list) and len(streams) == 1):
            raise RuntimeError(f"ffprobe_streams={streams!r}")
        stream = streams[0]
        if not isinstance(stream, dict):
            raise RuntimeError("ffprobe_stream_not_object")
        mp4_metadata_size = [int(stream["width"]), int(stream["height"])]
        rate_parts = str(stream["avg_frame_rate"]).split("/")
        if len(rate_parts) != 2 or float(rate_parts[1]) == 0.0:
            raise RuntimeError(f"ffprobe_bad_avg_frame_rate={rate_parts!r}")
        mp4_metadata_fps = float(rate_parts[0]) / float(rate_parts[1])

        decoded = subprocess.run(
            [
                str(FFMPEG_PATH), "-v", "error", "-i", str(mp4_path),
                "-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "framehash",
                "-hash", "sha256", "pipe:1",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=RENDER_DECODE_TIMEOUT_S,
        )
        if _SUPERVISOR_SIGNAL is not None:
            raise SupervisorTerminationRequested(
                f"supervisor received {_SUPERVISOR_SIGNAL['signal_name']} after decode"
            )
        if decoded.returncode != 0:
            raise RuntimeError(f"ffmpeg_rc={decoded.returncode}:{decoded.stderr}")
        expected_frame_bytes = width * height * 3
        framehash_rows = [
            line for line in decoded.stdout.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for line in framehash_rows:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                raise RuntimeError(f"framehash_bad_row={line!r}")
            mp4_decoded += 1
            if int(fields[4]) != expected_frame_bytes:
                mp4_frame_byte_lengths_exact = False
        declared_frames = stream.get("nb_frames")
        if declared_frames not in (None, "N/A") and int(declared_frames) != mp4_decoded:
            raise RuntimeError(
                f"ffprobe_decode_frame_count_mismatch={declared_frames}/{mp4_decoded}"
            )
    except subprocess.TimeoutExpired as exc:
        mp4_error = (
            f"TimeoutExpired: bounded decoder exceeded "
            f"{RENDER_DECODE_TIMEOUT_S:.1f}s cmd={exc.cmd!r}"
        )
    except BaseException as exc:
        mp4_error = f"{type(exc).__name__}: {exc}"
    passed = bool(
        png_decoded == len(expected_names)
        and not png_errors
        and mp4_error is None
        and mp4_decoded == len(expected_names)
        and mp4_metadata_size == [width, height]
        and mp4_metadata_fps is not None
        and abs(mp4_metadata_fps - fps) < 1.0e-9
        and mp4_frame_byte_lengths_exact
    )
    return {
        "artifact": "T3U_RENDER_INDEPENDENT_DECODE_V1",
        "png_expected": len(expected_names),
        "png_decoded": png_decoded,
        "png_errors": png_errors,
        "png_format": "PNG",
        "png_mode": "RGB",
        "resolution": [width, height],
        "mp4_decoded_frames": mp4_decoded,
        "mp4_metadata_size": mp4_metadata_size,
        "mp4_metadata_fps": mp4_metadata_fps,
        "mp4_frame_byte_lengths_exact": mp4_frame_byte_lengths_exact,
        "mp4_error": mp4_error,
        "pass": passed,
    }


def _render_posthoc_semantic_gate(
    profile: str,
    prefix: str,
    paths: dict[str, Path],
    render: dict[str, Any] | None,
    stdout_path: Path,
) -> dict[str, Any]:
    """Independently admit a render only after all durable semantics exist.

    Kit can turn a pending Python exception into process status zero while
    ``SimulationApp.close`` runs.  Therefore raw wait status is necessary but
    never sufficient: the exact 234-frame manifest, MP4, zero-physics clocks,
    dependency three-way binding, durable phase, and completion line are all
    recomputed from files by p16's pure validator.
    """
    raw_child_success = bool(
        isinstance(render, dict)
        and render.get("wifexited") is True
        and _strict_int(render.get("raw_wait_status"))
        and render.get("raw_wait_status") == 0
        and _strict_int(render.get("exit_code"))
        and render.get("exit_code") == 0
        and _strict_int(render.get("normalized_returncode"))
        and render.get("normalized_returncode") == 0
        and render.get("timed_out") is False
        and render.get("signal_actions") == []
        and render.get("group_reaped") is True
    )
    physics_failure_path = paths["failure.json"]
    render_failure_path = paths["render_failure.json"]
    required_rows: dict[str, Any] = {}
    for suffix in RENDER_REQUIRED_SUFFIXES:
        path = paths[suffix]
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        required_rows[suffix] = {
            "path": str(path.relative_to(REPO)),
            "exists": exists,
            "size_bytes": size,
            "sha256": sha256_file(path) if exists else None,
            "nonempty": bool(exists and int(size or 0) > 0),
        }
    missing_or_empty = [
        suffix for suffix, row in required_rows.items() if not row["nonempty"]
    ]
    semantic_checks: dict[str, bool] = {
        "all_required_render_files_nonempty": not missing_or_empty,
        "physics_failure_marker_absent": not physics_failure_path.exists(),
        "render_failure_marker_absent": not render_failure_path.exists(),
        "stdout_file_nonempty": bool(
            stdout_path.is_file() and stdout_path.stat().st_size > 0
        ),
    }
    validator_checks: dict[str, bool] = {}
    parsed: dict[str, Any] = {}
    parse_error: str | None = None
    frame_inventory: dict[str, Any] = {
        "directory": str((CASE_DIR / f"{prefix}_rgb_frames").relative_to(REPO)),
        "expected_count": None,
        "actual_names": [],
        "expected_names": [],
        "all_entries_are_files": False,
        "exact": False,
    }
    independent_decode: dict[str, Any] = {
        "artifact": "T3U_RENDER_INDEPENDENT_DECODE_V1",
        "pass": False,
        "not_run_reason": "render_required_files_missing_or_parse_not_reached",
    }
    if not missing_or_empty:
        try:
            p16 = _load_p16_semantic_validator()
            results = json.loads(paths["results.json"].read_text())
            plan = json.loads(paths["plan.json"].read_text())
            manifest = json.loads(paths["rgb_frames_manifest.json"].read_text())
            sentinel = json.loads(paths["preclose_sentinel.json"].read_text())
            phase_rows = [
                json.loads(line)
                for line in paths["phase.jsonl"].read_text().splitlines()
                if line.strip()
            ]
            phase_names = [row.get("phase") for row in phase_rows]
            render_phase_rows = [
                json.loads(line)
                for line in paths["render_phase.jsonl"].read_text().splitlines()
                if line.strip()
            ]
            expected_physics_phase_names = [
                list(PHYSICS_PRECLOSE_REQUIRED_PHASES),
                list(PHYSICS_PRECLOSE_REQUIRED_PHASES)
                + ["simulation_app_close_returned"],
            ]
            render_phase = render_phase_rows[0] if len(render_phase_rows) == 1 else {}
            prereg_path = (
                p16.PREFLIGHT_PREREG
                if profile == p16.PREFLIGHT_PROFILE else p16.CANONICAL_PREREG
            )
            physics_phase_semantics = _physics_phase_semantics(
                profile,
                phase_rows,
                results,
                sentinel,
                source_sha256=sha256_file(P16_PATH),
                prereg_sha256=sha256_file(prereg_path),
                results_sha256=sha256_file(paths["results.json"]),
                sentinel_sha256=sha256_file(paths["preclose_sentinel.json"]),
            )
            physics_phase_last_time = (
                float(physics_phase_semantics["phase_times_unix"][-1])
                if physics_phase_semantics.get("phase_times_unix") else math.nan
            )
            expected_frame_count = int(p16.TOTAL_STEPS // p16.VIDEO_STEP_STRIDE)
            expected_names = [f"frame_{index:04d}.png" for index in range(expected_frame_count)]
            frame_dir = CASE_DIR / f"{prefix}_rgb_frames"
            frame_entries = list(frame_dir.iterdir()) if frame_dir.is_dir() else []
            actual_names = sorted(path.name for path in frame_entries)
            all_entries_are_files = bool(
                frame_entries and all(path.is_file() for path in frame_entries)
            )
            frame_inventory = {
                "directory": str(frame_dir.relative_to(REPO)),
                "expected_count": expected_frame_count,
                "actual_names": actual_names,
                "expected_names": expected_names,
                "all_entries_are_files": all_entries_are_files,
                "exact": bool(
                    all_entries_are_files and actual_names == expected_names
                ),
            }
            validator_checks = p16.validate_render_manifest_semantics(
                profile, paths, manifest, results, plan
            )
            independent_decode = _independent_render_decode(
                frame_dir,
                expected_names,
                paths["side_grasp.mp4"],
                width=int(p16.VIDEO_WIDTH),
                height=int(p16.VIDEO_HEIGHT),
                fps=float(p16.VIDEO_FPS),
            )
            independent_decode_schema_exact = bool(
                set(independent_decode)
                == {
                    "artifact", "png_expected", "png_decoded", "png_errors",
                    "png_format", "png_mode", "resolution",
                    "mp4_decoded_frames", "mp4_metadata_size",
                    "mp4_metadata_fps", "mp4_frame_byte_lengths_exact",
                    "mp4_error", "pass",
                }
                and independent_decode.get("artifact")
                == "T3U_RENDER_INDEPENDENT_DECODE_V1"
                and _strict_int(independent_decode.get("png_expected"))
                and independent_decode.get("png_expected") == expected_frame_count
                and _strict_int(independent_decode.get("png_decoded"))
                and independent_decode.get("png_decoded") == expected_frame_count
                and independent_decode.get("png_errors") == []
                and independent_decode.get("png_format") == "PNG"
                and independent_decode.get("png_mode") == "RGB"
                and isinstance(independent_decode.get("resolution"), list)
                and len(independent_decode["resolution"]) == 2
                and all(_strict_int(value) for value in independent_decode["resolution"])
                and independent_decode["resolution"] == [p16.VIDEO_WIDTH, p16.VIDEO_HEIGHT]
                and _strict_int(independent_decode.get("mp4_decoded_frames"))
                and independent_decode.get("mp4_decoded_frames") == expected_frame_count
                and isinstance(independent_decode.get("mp4_metadata_size"), list)
                and len(independent_decode["mp4_metadata_size"]) == 2
                and all(
                    _strict_int(value)
                    for value in independent_decode["mp4_metadata_size"]
                )
                and independent_decode["mp4_metadata_size"]
                == [p16.VIDEO_WIDTH, p16.VIDEO_HEIGHT]
                and _strict_float(independent_decode.get("mp4_metadata_fps"))
                and abs(
                    independent_decode["mp4_metadata_fps"] - float(p16.VIDEO_FPS)
                ) < 1.0e-9
                and independent_decode.get("mp4_frame_byte_lengths_exact") is True
                and independent_decode.get("mp4_error") is None
                and independent_decode.get("pass") is True
            )
            stdout = stdout_path.read_text(errors="replace")
            manifest_frames = manifest.get("frames", [])
            manifest_frame_paths = [
                row.get("path") for row in manifest_frames if isinstance(row, dict)
            ] if isinstance(manifest_frames, list) else []
            semantic_checks.update(
                {
                    "p16_render_semantic_validator_all_true": bool(
                        validator_checks and all(validator_checks.values())
                    ),
                    "all_png_and_mp4_frames_independently_decoded": (
                        independent_decode_schema_exact
                    ),
                    "frame_directory_exact_234_unique_pngs": bool(
                        frame_inventory["exact"]
                        and expected_frame_count == 234
                        and len(set(manifest_frame_paths)) == expected_frame_count
                    ),
                    "render_phase_exact_once_and_last": bool(
                        phase_names in expected_physics_phase_names
                        and physics_phase_semantics.get("pass") is True
                        and len(render_phase_rows) == 1
                        and render_phase.get("phase") == "render_trace_durable"
                    ),
                    "render_phase_manifest_mp4_hashes_exact": bool(
                        set(render_phase)
                        == {
                            "time_unix", "phase", "manifest_sha256", "mp4_sha256",
                            "observed_physics_step_events",
                            "observed_simulation_manager_step_delta",
                        }
                        and render_phase.get("manifest_sha256")
                        == sha256_file(paths["rgb_frames_manifest.json"])
                        and render_phase.get("mp4_sha256")
                        == sha256_file(paths["side_grasp.mp4"])
                        and _strict_int(
                            render_phase.get("observed_physics_step_events")
                        )
                        and render_phase.get("observed_physics_step_events") == 0
                        and _strict_int(
                            render_phase.get("observed_simulation_manager_step_delta")
                        )
                        and render_phase.get("observed_simulation_manager_step_delta") == 0
                        and _finite_time_unix(render_phase.get("time_unix"))
                        and float(render_phase["time_unix"])
                        >= physics_phase_last_time
                    ),
                    "stdout_render_completion_exact_line_once": bool(
                        stdout.splitlines().count(
                            f"[p16_t3u_side] RENDER_TRACE_COMPLETE "
                            f"profile={profile} frames={expected_frame_count} "
                            f"mp4={paths['side_grasp.mp4']}"
                        ) == 1
                        and stdout.count("[p16_t3u_side] RENDER_TRACE_COMPLETE") == 1
                    ),
                    "stdout_has_no_terminal_failure_tokens": not any(
                        token in stdout
                        for token in (
                            "Traceback (most recent call last)",
                            "G0_ARTIFACT_EXISTS_ABORT", "RuntimeError:",
                            "Segmentation fault", "core dumped",
                        )
                    ),
                    "source_frozen_manifest_binding_exact": bool(
                        sha256_file(paths["script.py.txt"]) == sha256_file(P16_PATH)
                        == manifest.get("executed_source_sha256")
                        == results.get("provenance", {}).get("source_sha256")
                    ),
                }
            )
            parsed = {
                "physics_phase_names": phase_names,
                "physics_phase_semantics": physics_phase_semantics,
                "render_phase_names": [row.get("phase") for row in render_phase_rows],
                "physics_phase_sha256": sha256_file(paths["phase.jsonl"]),
                "render_phase_sha256": sha256_file(paths["render_phase.jsonl"]),
                "manifest_sha256": sha256_file(paths["rgb_frames_manifest.json"]),
                "mp4_sha256": sha256_file(paths["side_grasp.mp4"]),
                "source_sha256": sha256_file(P16_PATH),
                "stdout_render_completion_count": stdout.count(
                    "[p16_t3u_side] RENDER_TRACE_COMPLETE"
                ),
            }
        except BaseException as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    semantic_checks["render_documents_parse_without_error"] = bool(
        not missing_or_empty and parse_error is None
    )
    passed = bool(raw_child_success and all(semantic_checks.values()))
    return {
        "artifact": "T3U_RENDER_POSTHOC_SEMANTIC_GATE_V1",
        "profile": profile,
        "raw_child_success": raw_child_success,
        "failure_markers": {
            "physics": {
                "path": str(physics_failure_path.relative_to(REPO)),
                "exists": physics_failure_path.exists(),
                "sha256": (
                    sha256_file(physics_failure_path)
                    if physics_failure_path.is_file() else None
                ),
            },
            "render": {
                "path": str(render_failure_path.relative_to(REPO)),
                "exists": render_failure_path.exists(),
                "sha256": (
                    sha256_file(render_failure_path)
                    if render_failure_path.is_file() else None
                ),
            },
        },
        "required_files": required_rows,
        "missing_or_empty": missing_or_empty,
        "frame_inventory": frame_inventory,
        "independent_decode": independent_decode,
        "semantic_checks": semantic_checks,
        "p16_validator_checks": validator_checks,
        "parsed_bindings": parsed,
        "parse_error": parse_error,
        "pass": passed,
    }


def _pgid_members(pgid: int) -> list[int]:
    members: list[int] = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text()
            closing = raw.rfind(")")
            fields = raw[closing + 2 :].split()
            if closing > 0 and len(fields) >= 3 and int(fields[2]) == pgid:
                members.append(int(stat_path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return sorted(members)


def _gpu_inventory() -> bytes:
    query = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        timeout=15.0,
    )
    if query.returncode != 0:
        raise RuntimeError(
            f"SUPERVISOR_NVIDIA_SMI_FAIL rc={query.returncode} stderr={query.stderr!r}"
        )
    return query.stdout


def _gpu_pids(payload: bytes) -> list[int]:
    result: list[int] = []
    for line in payload.decode("utf-8", errors="replace").splitlines():
        token = line.split(",", 1)[0].strip()
        if token:
            result.append(int(token))
    return sorted(set(result))


def _decode_wait_status(raw_status: int) -> dict[str, Any]:
    exited = os.WIFEXITED(raw_status)
    signaled = os.WIFSIGNALED(raw_status)
    exit_code = os.WEXITSTATUS(raw_status) if exited else None
    signal_number = os.WTERMSIG(raw_status) if signaled else None
    return {
        "raw_wait_status": int(raw_status),
        "wifexited": bool(exited),
        "exit_code": exit_code,
        "wifsignaled": bool(signaled),
        "signal_number": signal_number,
        "signal_name": (
            signal.Signals(signal_number).name if signal_number is not None else None
        ),
        "core_dumped": bool(os.WCOREDUMP(raw_status)) if signaled else False,
        "normalized_returncode": (
            int(exit_code) if exit_code is not None else 128 + int(signal_number or 0)
        ),
    }


def _elapsed_matches_wall_clock(
    start_time_unix: float, end_time_unix: float, elapsed_seconds: float
) -> bool:
    """Bind monotonic elapsed evidence to the independent wall-clock interval."""
    return bool(
        type(start_time_unix) is float
        and type(end_time_unix) is float
        and type(elapsed_seconds) is float
        and math.isfinite(start_time_unix)
        and math.isfinite(end_time_unix)
        and math.isfinite(elapsed_seconds)
        and start_time_unix > 0.0
        and end_time_unix >= start_time_unix
        and elapsed_seconds >= 0.0
        and abs(elapsed_seconds - (end_time_unix - start_time_unix))
        <= WALL_TIME_ABS_TOLERANCE_S
    )


def _signal_group(pgid: int, sig: signal.Signals) -> dict[str, Any]:
    before = _pgid_members(pgid)
    sent = False
    error: str | None = None
    if before:
        try:
            os.killpg(pgid, int(sig))
            sent = True
        except ProcessLookupError:
            pass
        except BaseException as exc:  # recorded and made terminal-fatal
            error = f"{type(exc).__name__}: {exc}"
    return {
        "signal": sig.name,
        "time_unix": time.time(),
        "members_before": before,
        "sent": sent,
        "error": error,
    }


def _wait_group_empty(pgid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pgid_members(pgid):
            return True
        time.sleep(0.05)
    return not _pgid_members(pgid)


def _bounded_waitpid(pid: int, timeout_s: float) -> tuple[int | None, str | None]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError as exc:
            return None, f"ChildProcessError: {exc}"
        if waited_pid == pid:
            return int(status), None
        time.sleep(0.05)
    return None, "bounded_waitpid_deadline_exceeded"


def _signal_handler(signum: int, _frame: Any) -> None:
    global _SUPERVISOR_SIGNAL
    if _SUPERVISOR_SIGNAL is None:
        _SUPERVISOR_SIGNAL = {
            "signal_number": int(signum),
            "signal_name": signal.Signals(signum).name,
            "time_unix": time.time(),
        }


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


def _prepare_child_parent_death(expected_parent_pid: int) -> None:
    """Make parent death terminate the pre-exec child instead of setting a flag.

    ``fork`` inherits the supervisor's Python TERM/INT handlers.  Those handlers
    are correct in the supervisor, but would swallow ``PDEATHSIG=SIGTERM`` in the
    child during the fork-to-exec window.  Restore terminating dispositions and
    clear the inherited signal mask before arming PDEATHSIG, then close the
    standard prctl race by comparing against the parent PID captured pre-fork.
    """
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, [])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), int(signal.SIGTERM))
        os._exit(125)  # unreachable with SIG_DFL/unblocked; explicit fail-safe


def _terminate_and_reap_child(
    pid: int,
    pgid: int,
    reason: str,
    actions: list[dict[str, Any]],
) -> int:
    term = _signal_group(pgid, signal.SIGTERM)
    term["reason"] = reason
    actions.append(term)
    raw_status, wait_error = _bounded_waitpid(pid, KILL_AFTER_S)
    if raw_status is None:
        kill = _signal_group(pgid, signal.SIGKILL)
        kill["reason"] = f"{reason}:term_grace_expired"
        actions.append(kill)
        raw_status, wait_error = _bounded_waitpid(pid, BOUNDED_REAP_S)
    if raw_status is None:
        raise RuntimeError(
            "BOUNDED_CHILD_REAP_FAIL "
            f"pid={pid} pgid={pgid} reason={reason} wait_error={wait_error} "
            f"members={_pgid_members(pgid)}"
        )
    return raw_status


def _cleanup_active_child(reason: str) -> dict[str, Any] | None:
    if not _ACTIVE_CHILD:
        return None
    snapshot = dict(_ACTIVE_CHILD)
    actions: list[dict[str, Any]] = []
    raw_status: int | None = None
    error: str | None = None
    try:
        raw_status = _terminate_and_reap_child(
            int(snapshot["pid"]), int(snapshot["pgid"]), reason, actions
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    if not _wait_group_empty(int(snapshot["pgid"]), QUIET_REAP_GRACE_S):
        term = _signal_group(int(snapshot["pgid"]), signal.SIGTERM)
        term["reason"] = f"{reason}:descendant_cleanup"
        actions.append(term)
        if not _wait_group_empty(int(snapshot["pgid"]), KILL_AFTER_S):
            kill = _signal_group(int(snapshot["pgid"]), signal.SIGKILL)
            kill["reason"] = f"{reason}:descendant_kill"
            actions.append(kill)
            _wait_group_empty(int(snapshot["pgid"]), BOUNDED_REAP_S)
    event = {
        "reason": reason,
        "active_child": snapshot,
        "signal_actions": actions,
        "raw_wait_status": raw_status,
        "decoded_wait_status": (
            None if raw_status is None else _decode_wait_status(raw_status)
        ),
        "members_after_cleanup": _pgid_members(int(snapshot["pgid"])),
        "error": error,
        "time_unix": time.time(),
    }
    _SUPERVISOR_CLEANUP_ACTIONS.append(event)
    _ACTIVE_CHILD.clear()
    return event


def run_child(
    label: str,
    command: list[str],
    timeout_s: float,
    combined_stdout_fd: int,
    pid_path: Path,
) -> dict[str, Any]:
    global _LAST_CHILD_OUTCOME
    start_unix = time.time()
    start_monotonic = time.monotonic()
    child_tty = {
        "stdin": False,
        "stdout": bool(os.isatty(combined_stdout_fd)),
        "stderr": bool(os.isatty(combined_stdout_fd)),
    }
    expected_parent_pid = os.getpid()
    pid = os.fork()
    if pid == 0:
        try:
            _prepare_child_parent_death(expected_parent_pid)
            os.setpgid(0, 0)
            os.chdir(REPO)
            devnull_fd = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull_fd, 0)
            os.dup2(combined_stdout_fd, 1)
            os.dup2(combined_stdout_fd, 2)
            if devnull_fd > 2:
                os.close(devnull_fd)
            os.execve(command[0], command, dict(os.environ))
        except BaseException:
            traceback.print_exc()
            os._exit(126)

    try:
        os.setpgid(pid, pid)
    except (PermissionError, ProcessLookupError):
        pass
    child_pgid = os.getpgid(pid)
    child_sid = os.getsid(pid)
    _ACTIVE_CHILD.update(
        {"label": label, "pid": pid, "pgid": child_pgid, "sid": child_sid}
    )
    write_bytes_x(pid_path, f"{pid}\n".encode("ascii"))
    actions: list[dict[str, Any]] = []
    timed_out = False
    raw_status: int | None = None
    while raw_status is None:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            raw_status = status
            break
        if _SUPERVISOR_SIGNAL is not None:
            raw_status = _terminate_and_reap_child(
                pid,
                child_pgid,
                f"supervisor_{_SUPERVISOR_SIGNAL['signal_name']}",
                actions,
            )
            break
        if time.monotonic() - start_monotonic >= timeout_s:
            timed_out = True
            raw_status = _terminate_and_reap_child(
                pid, child_pgid, "child_timeout", actions
            )
            break
        time.sleep(0.05)

    decoded = _decode_wait_status(int(raw_status))
    if not _wait_group_empty(child_pgid, QUIET_REAP_GRACE_S):
        quiet_term = _signal_group(child_pgid, signal.SIGTERM)
        quiet_term["reason"] = "post_child_descendant_cleanup"
        actions.append(quiet_term)
        if not _wait_group_empty(child_pgid, KILL_AFTER_S):
            quiet_kill = _signal_group(child_pgid, signal.SIGKILL)
            quiet_kill["reason"] = "post_child_descendant_kill"
            actions.append(quiet_kill)
            _wait_group_empty(child_pgid, KILL_AFTER_S)
    end_unix = time.time()
    outcome = {
        "label": label,
        "attempt_index": 0,
        "attempt_count": 1,
        "command": command,
        "pid": pid,
        "pgid": child_pgid,
        "sid": child_sid,
        "tty": child_tty,
        "start_time_unix": start_unix,
        "end_time_unix": end_unix,
        "elapsed_seconds": time.monotonic() - start_monotonic,
        "timeout_seconds": timeout_s,
        "timed_out": timed_out,
        "supervisor_signal": _SUPERVISOR_SIGNAL,
        "signal_actions": actions,
        "group_members_after_reap": _pgid_members(child_pgid),
        "group_reaped": not _pgid_members(child_pgid),
        **decoded,
    }
    if not _elapsed_matches_wall_clock(
        outcome["start_time_unix"],
        outcome["end_time_unix"],
        outcome["elapsed_seconds"],
    ):
        raise RuntimeError(f"CHILD_WALL_TIME_CONSISTENCY_FAIL label={label}")
    _LAST_CHILD_OUTCOME = outcome
    if outcome["group_reaped"]:
        _ACTIVE_CHILD.clear()
    return outcome


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=("side_preflight13", "side_phys1"))
    parser.add_argument("--candidates_sha256", required=True)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if sys.argv[1:] != [
        "--profile", args.profile, "--candidates_sha256", args.candidates_sha256
    ]:
        raise RuntimeError(f"SUPERVISOR_ARGV_DRIFT {sys.argv}")
    if args.candidates_sha256 != CANDIDATES_SHA256:
        raise RuntimeError("SUPERVISOR_CANDIDATES_LAUNCH_PIN_MISMATCH")
    if sha256_file(CANDIDATES_PATH) != CANDIDATES_SHA256:
        raise RuntimeError("SUPERVISOR_CANDIDATES_FILE_PIN_MISMATCH")
    supervisor_pid = os.getpid()
    supervisor_pgid = os.getpgrp()
    supervisor_sid = os.getsid(0)
    tty = {name: os.isatty(fd) for name, fd in (("stdin", 0), ("stdout", 1), ("stderr", 2))}
    if supervisor_pid != supervisor_pgid or supervisor_pid != supervisor_sid or any(tty.values()):
        raise RuntimeError(
            "SUPERVISOR_MUST_BE_NO_TTY_SETSID_SESSION_LEADER "
            f"pid={supervisor_pid} pgid={supervisor_pgid} sid={supervisor_sid} tty={tty}"
        )
    _assert_host_execution_boundary()
    host_launch_context = _host_launch_context()
    _install_signal_handlers()

    prefix = f"t3u_{args.profile}"
    paths = {suffix: CASE_DIR / f"{prefix}_{suffix}" for suffix in (*CORE_SUFFIXES, *EXTERNAL_SUFFIXES)}
    frame_dir = CASE_DIR / f"{prefix}_rgb_frames"
    launcher_log = CASE_DIR / f"{prefix}_supervisor_launcher.log"
    # The launcher has already opened launcher_log with shell noclobber.  Every
    # other lifecycle or scientific target must still be absent here.
    present = [str(path.relative_to(REPO)) for path in [*paths.values(), frame_dir] if path.exists()]
    if present:
        raise RuntimeError(f"SUPERVISOR_G0_TARGET_EXISTS {present}")
    if not launcher_log.is_file():
        raise RuntimeError(f"SUPERVISOR_LAUNCHER_LOG_MISSING {launcher_log}")

    contract = {
        "artifact": "T3U_DETACHED_PHYSICS_THEN_RENDER_SUPERVISOR_V15",
        "automatic_retry_count": 0,
        "detached": True,
        "physics_timeout_seconds": int(PHYSICS_TIMEOUT_S),
        "render_timeout_seconds": int(RENDER_TIMEOUT_S),
        "term_signal": "TERM",
        "kill_after_seconds": int(KILL_AFTER_S),
        "wall_time_end_minus_start_abs_tolerance_seconds": (
            WALL_TIME_ABS_TOLERANCE_S
        ),
        "physics_then_render_only_on_raw_zero_and_preclose_semantic_gate": True,
        "physics_semantic_gate_artifact": "T3U_PHYSICS_PRECLOSE_SEMANTIC_GATE_V1",
        "render_success_requires_raw_zero_and_posthoc_semantic_gate": True,
        "render_semantic_gate_artifact": "T3U_RENDER_POSTHOC_SEMANTIC_GATE_V1",
        "semantic_gate_failure_exit_status": SEMANTIC_GATE_FAILURE_EXIT_STATUS,
        "completed_preclose_semantic_rejection_terminal_branch": True,
        "completed_preclose_semantic_rejection_terminal_artifact": (
            "T3U_EXTERNAL_TERMINAL_COMPLETED_PRECLOSE_SEMANTIC_REJECTION_"
            "ATTESTATION_V1"
        ),
        "raw_waitpid_status_authority": True,
        "bounded_waitpid_only": True,
        "supervisor_signal_cleanup": (
            "SIGTERM_SIGINT__active_child_pgid_TERM_20s_then_KILL_20s"
        ),
        "child_parent_death_signal": "SIGTERM",
        "child_preexec_signal_state": (
            "SIGTERM_SIGINT_SIGHUP_SIG_DFL__empty_mask__expected_parent_pid_recheck"
        ),
        "host_launch_boundary": (
            "require_escalated_exec_command__outside_bwrap_die_with_parent"
        ),
        "forbidden_sandbox_ancestor_gate": True,
        "host_launch_context": host_launch_context,
    }
    write_json_x(paths["supervisor_contract.json"], contract)
    write_bytes_x(paths["supervisor_pid.txt"], f"{supervisor_pid}\n".encode("ascii"))
    write_bytes_x(paths["pgid.txt"], f"{supervisor_pgid}\n".encode("ascii"))
    gpu_before = _gpu_inventory()
    write_bytes_x(paths["nvidia_smi_before.csv"], gpu_before)
    stdout_fd = os.open(paths["stdout.log"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    start_unix = time.time()
    start_monotonic = time.monotonic()
    physics: dict[str, Any] | None = None
    render: dict[str, Any] | None = None
    physics_artifact_gate: dict[str, Any] | None = None
    render_artifact_gate: dict[str, Any] | None = None
    try:
        physics_command = [
            str(ISAAC_PYTHON), str(P16_PATH), "--run_label", args.profile,
            "--candidates_sha256", CANDIDATES_SHA256,
        ]
        physics = run_child(
            "physics", physics_command, PHYSICS_TIMEOUT_S, stdout_fd,
            paths["physics_python_pid.txt"],
        )
        if _SUPERVISOR_SIGNAL is not None:
            raise SupervisorTerminationRequested(
                f"supervisor received {_SUPERVISOR_SIGNAL['signal_name']}"
            )
        os.fsync(stdout_fd)
        physics_artifact_gate = _physics_preclose_semantic_gate(
            args.profile, prefix, paths, physics, paths["stdout.log"]
        )
        if _SUPERVISOR_SIGNAL is not None:
            raise SupervisorTerminationRequested(
                f"supervisor received {_SUPERVISOR_SIGNAL['signal_name']} after physics gate"
            )
        physics_success = bool(physics_artifact_gate["pass"])
        if physics_success:
            render_command = [
                str(ISAAC_PYTHON), str(P16_PATH), "--render_trace", args.profile,
            ]
            render = run_child(
                "render", render_command, RENDER_TIMEOUT_S, stdout_fd,
                paths["render_python_pid.txt"],
            )
            if _SUPERVISOR_SIGNAL is not None:
                raise SupervisorTerminationRequested(
                    f"supervisor received {_SUPERVISOR_SIGNAL['signal_name']}"
                )
        os.fsync(stdout_fd)
    finally:
        _cleanup_active_child("main_finally")
        os.close(stdout_fd)

    if physics is None or physics_artifact_gate is None:
        raise RuntimeError("PHYSICS_CHILD_OUTCOME_MISSING")

    if render is not None:
        render_artifact_gate = _render_posthoc_semantic_gate(
            args.profile, prefix, paths, render, paths["stdout.log"]
        )
        if _SUPERVISOR_SIGNAL is not None:
            raise SupervisorTerminationRequested(
                f"supervisor received {_SUPERVISOR_SIGNAL['signal_name']} after render gate"
            )
    render_success = bool(
        render_artifact_gate is not None and render_artifact_gate["pass"]
    )
    combined_status = (
        0 if physics_success and render_success
        else (
            int(physics["normalized_returncode"])
            if int(physics["normalized_returncode"]) != 0
            else SEMANTIC_GATE_FAILURE_EXIT_STATUS
        )
        if not physics_success
        else (
            int(render["normalized_returncode"])
            if render is not None and int(render["normalized_returncode"]) != 0
            else SEMANTIC_GATE_FAILURE_EXIT_STATUS
        )
    )
    write_bytes_x(paths["exit_status.txt"], f"{combined_status}\n".encode("ascii"))
    gpu_end = _gpu_inventory()
    write_bytes_x(paths["nvidia_smi_supervisor_end.csv"], gpu_end)
    before_pids = _gpu_pids(gpu_before)
    end_pids = _gpu_pids(gpu_end)
    supervisor_group_members = _pgid_members(supervisor_pgid)
    bindings = {
        name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if path.is_file() and name != "supervisor_outcome.json"
    }
    bindings["supervisor_launcher.log"] = {
        "path": str(launcher_log.relative_to(REPO)),
        "sha256": sha256_file(launcher_log),
    }
    outcome_end_time_unix = time.time()
    outcome_elapsed_seconds = time.monotonic() - start_monotonic
    if not _elapsed_matches_wall_clock(
        start_unix, outcome_end_time_unix, outcome_elapsed_seconds
    ):
        raise RuntimeError("SUPERVISOR_WALL_TIME_CONSISTENCY_FAIL")
    outcome = {
        "artifact": "T3U_DETACHED_SUPERVISOR_OUTCOME_V15",
        "profile": args.profile,
        "argv": list(sys.argv),
        "supervisor_source_sha256": sha256_file(Path(__file__)),
        "p16_source_sha256": sha256_file(P16_PATH),
        "candidates_sha256": sha256_file(CANDIDATES_PATH),
        "start_time_unix": start_unix,
        "end_time_unix": outcome_end_time_unix,
        "elapsed_seconds": outcome_elapsed_seconds,
        "supervisor": {
            "pid": supervisor_pid,
            "pgid": supervisor_pgid,
            "sid": supervisor_sid,
            "tty": tty,
            "group_members_before_exit": supervisor_group_members,
            "self_only_before_exit": supervisor_group_members == [supervisor_pid],
            "signal_received": _SUPERVISOR_SIGNAL,
            "cleanup_actions": list(_SUPERVISOR_CLEANUP_ACTIONS),
            "active_child_at_exit": dict(_ACTIVE_CHILD) if _ACTIVE_CHILD else None,
        },
        "attempts": {
            "physics": 1,
            "render": 1 if render is not None else 0,
            "automatic_retry_count": 0,
        },
        "physics": physics,
        "physics_artifact_gate": physics_artifact_gate,
        "render": render,
        "render_artifact_gate": render_artifact_gate,
        "render_started_iff_physics_success": bool(
            (render is not None) == physics_success
        ),
        "combined_exit_status": combined_status,
        "gpu": {
            "before_pids": before_pids,
            "supervisor_end_pids": end_pids,
            "fresh_pid_delta": sorted(set(end_pids) - set(before_pids)),
            "no_fresh_pid_delta": not (set(end_pids) - set(before_pids)),
        },
        "bindings": bindings,
        "contract": contract,
        "host_launch_context": host_launch_context,
        "pass": bool(
            combined_status == 0
            and physics_success and render_success
            and render is not None
            and (render is not None) == physics_success
            and supervisor_group_members == [supervisor_pid]
            and not (set(end_pids) - set(before_pids))
            and _SUPERVISOR_SIGNAL is None
            and not _SUPERVISOR_CLEANUP_ACTIONS
            and not _ACTIVE_CHILD
        ),
    }
    write_json_x(paths["supervisor_outcome.json"], outcome)
    return combined_status


if __name__ == "__main__":
    try:
        _exit_code = main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            raise
        if isinstance(exc, HostExecutionBoundaryError):
            # The host-boundary gate runs before internal tag paths are resolved or
            # written.  Preserve that property; the externally opened launcher log is
            # the only possible evidence when a caller bypasses the preregistered
            # pre-redirection shell guard.
            raise
        # stdout/stderr are the forward-only launcher diagnostic if the canonical
        # outcome could not be completed.  Best-effort supervisor_failure never
        # overwrites an existing tag artifact.
        try:
            profile = next(
                (sys.argv[i + 1] for i, token in enumerate(sys.argv[:-1]) if token == "--profile"),
                "unknown",
            )
            failure_path = CASE_DIR / f"t3u_{profile}_supervisor_failure.json"
            if not failure_path.exists():
                write_json_x(
                    failure_path,
                    {
                        "artifact": "T3U_DETACHED_SUPERVISOR_FAILURE_V1",
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                        "argv": list(sys.argv),
                        "time_unix": time.time(),
                        "supervisor_signal": _SUPERVISOR_SIGNAL,
                        "active_child_after_finally": dict(_ACTIVE_CHILD),
                        "last_child_outcome": _LAST_CHILD_OUTCOME,
                        "cleanup_actions": list(_SUPERVISOR_CLEANUP_ACTIONS),
                    },
                )
        finally:
            raise
    raise SystemExit(_exit_code)
