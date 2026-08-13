#!/usr/bin/env python3
"""Host-visible, one-attempt supervisor for the P13 RTX replay repair."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"
PREFIX = "t3u_side_render1"
RENDER_SOURCE = REPO / "sim_scripts/p16_g0b_t3u_side_preflight13_rtx_render_repair_v1.py"
RENDER_SOURCE_SHA256 = "7dbc54821e789550f20e10526f7f2d278043378db95b447d0fc2c8d0e727130f"
PREREG_PATH = CASE_DIR / f"{PREFIX}_prereg.md"
PREREG_SHA256 = "59c67cf1e2b3f0bf1071bac0ea7607ced69dcbaae2477d83d525381cda39455e"
ISAAC_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
RETIRED_P13_HOST_PIDS = (2985672, 2988728)
LAUNCH_DEADLINE_S = 120.0
NO_PROGRESS_DEADLINE_S = 90.0
GLOBAL_DEADLINE_S = 900.0
KILL_AFTER_S = 20.0
POLL_S = 0.25
EXPECTED_FRAMES = 234
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 20.0
RESERVED_FAILURE_STATUS = 125

CHILD_OUTPUTS = {
    "input_gate.json": CASE_DIR / f"{PREFIX}_input_gate.json",
    "phase.jsonl": CASE_DIR / f"{PREFIX}_phase.jsonl",
    "script.py.txt": CASE_DIR / f"{PREFIX}_script.py.txt",
    "argv.txt": CASE_DIR / f"{PREFIX}_argv.txt",
    "rgb_frames_manifest.json": CASE_DIR / f"{PREFIX}_rgb_frames_manifest.json",
    "side_grasp.mp4": CASE_DIR / f"{PREFIX}_side_grasp.mp4",
    "failure.json": CASE_DIR / f"{PREFIX}_failure.json",
}
FRAME_DIR = CASE_DIR / f"{PREFIX}_rgb_frames"
SUPERVISOR_OUTPUTS = {
    "supervisor_pid.txt": CASE_DIR / f"{PREFIX}_supervisor_pid.txt",
    "render_python_pid.txt": CASE_DIR / f"{PREFIX}_render_python_pid.txt",
    "pgid.txt": CASE_DIR / f"{PREFIX}_pgid.txt",
    "stdout.log": CASE_DIR / f"{PREFIX}_stdout.log",
    "nvidia_smi_before.csv": CASE_DIR / f"{PREFIX}_nvidia_smi_before.csv",
    "nvidia_smi_after.csv": CASE_DIR / f"{PREFIX}_nvidia_smi_after.csv",
    "supervisor_outcome.json": CASE_DIR / f"{PREFIX}_supervisor_outcome.json",
    "supervisor_failure.json": CASE_DIR / f"{PREFIX}_supervisor_failure.json",
    "exit_status.txt": CASE_DIR / f"{PREFIX}_exit_status.txt",
}

_SUPERVISOR_SIGNAL: dict[str, Any] | None = None


class HostExecutionBoundaryError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_x(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_x(path: Path, value: Any) -> None:
    write_bytes_x(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _proc_parent_pid(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text()
    closing = raw.rfind(")")
    if closing < 0:
        raise HostExecutionBoundaryError(f"PROC_STAT_SHAPE_FAIL pid={pid}")
    fields = raw[closing + 2 :].split()
    return int(fields[1])


def _proc_argv(pid: int) -> list[str]:
    return [
        token.decode("utf-8", errors="replace")
        for token in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if token
    ]


def host_launch_context() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pid = os.getpid()
    seen: set[int] = set()
    for _ in range(64):
        if pid <= 0 or pid in seen:
            break
        seen.add(pid)
        argv = _proc_argv(pid)
        rows.append({"pid": pid, "argv": argv})
        if pid == 1:
            break
        pid = _proc_parent_pid(pid)
    if not rows or rows[-1]["pid"] != 1:
        raise HostExecutionBoundaryError("HOST_ANCESTRY_INCOMPLETE")
    forbidden: list[dict[str, Any]] = []
    for row in rows:
        argv = row["argv"]
        executable = Path(argv[0]).name if argv else ""
        if executable in {"bwrap", "codex-linux-sandbox"}:
            forbidden.append({"pid": row["pid"], "token": executable})
        if "--die-with-parent" in argv:
            forbidden.append({"pid": row["pid"], "token": "--die-with-parent"})
    if forbidden:
        raise HostExecutionBoundaryError(f"HOST_EXECUTION_REQUIRED {forbidden}")
    raw_pid1 = Path("/proc/1/cmdline").read_bytes()
    return {
        "artifact": "T3U_RENDER1_HOST_LAUNCH_CONTEXT_V1",
        "ancestor_rows": rows,
        "pid1_cmdline_hex": raw_pid1.hex(),
        "pid1_cmdline_sha256": hashlib.sha256(raw_pid1).hexdigest(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "forbidden_matches": forbidden,
        "pass": bool(raw_pid1 and not forbidden),
    }


def output_g0() -> None:
    paths = [*CHILD_OUTPUTS.values(), *SUPERVISOR_OUTPUTS.values()]
    existing = [str(path.relative_to(REPO)) for path in paths if path.exists()]
    if FRAME_DIR.exists():
        existing.append(str(FRAME_DIR.relative_to(REPO)))
    if existing:
        raise RuntimeError(f"RENDER1_SUPERVISOR_G0_OUTPUT_EXISTS {existing}")


def gpu_inventory() -> bytes:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        timeout=15.0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"NVIDIA_SMI_FAIL {result.stderr!r}")
    return result.stdout


def gpu_pids(payload: bytes) -> set[int]:
    result: set[int] = set()
    for line in payload.decode("utf-8", errors="replace").splitlines():
        token = line.split(",", 1)[0].strip()
        if token:
            result.add(int(token))
    return result


def pgid_members(pgid: int) -> list[int]:
    members: list[int] = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = path.read_text()
            closing = raw.rfind(")")
            fields = raw[closing + 2 :].split()
            if closing > 0 and len(fields) >= 3 and int(fields[2]) == pgid:
                members.append(int(path.parent.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return sorted(members)


def decode_wait_status(raw: int) -> dict[str, Any]:
    exited = os.WIFEXITED(raw)
    signaled = os.WIFSIGNALED(raw)
    signal_number = os.WTERMSIG(raw) if signaled else None
    exit_code = os.WEXITSTATUS(raw) if exited else None
    return {
        "raw_wait_status": int(raw),
        "wifexited": bool(exited),
        "exit_code": exit_code,
        "wifsignaled": bool(signaled),
        "signal_number": signal_number,
        "signal_name": signal.Signals(signal_number).name
        if signal_number is not None
        else None,
        "normalized_returncode": int(exit_code)
        if exit_code is not None
        else 128 + int(signal_number or 0),
    }


def signal_handler(signum: int, _frame: Any) -> None:
    global _SUPERVISOR_SIGNAL
    if _SUPERVISOR_SIGNAL is None:
        _SUPERVISOR_SIGNAL = {
            "signal_number": int(signum),
            "signal_name": signal.Signals(signum).name,
            "time_unix": time.time(),
        }


def child_setup(expected_parent: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, [])
    os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))
    if os.getppid() != expected_parent:
        os.kill(os.getpid(), int(signal.SIGTERM))
        os._exit(RESERVED_FAILURE_STATUS)


def read_phase_rows() -> list[dict[str, Any]]:
    path = CHILD_OUTPUTS["phase.jsonl"]
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def terminate_and_reap(pid: int, reason: str, actions: list[dict[str, Any]]) -> int:
    before = pgid_members(pid)
    try:
        os.killpg(pid, signal.SIGTERM)
        sent = True
        error = None
    except ProcessLookupError:
        sent = False
        error = None
    except BaseException as exc:
        sent = False
        error = f"{type(exc).__name__}: {exc}"
    actions.append(
        {
            "signal": "SIGTERM",
            "reason": reason,
            "time_unix": time.time(),
            "members_before": before,
            "sent": sent,
            "error": error,
        }
    )
    deadline = time.monotonic() + KILL_AFTER_S
    while time.monotonic() < deadline:
        waited, raw = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return int(raw)
        time.sleep(0.05)
    before = pgid_members(pid)
    try:
        os.killpg(pid, signal.SIGKILL)
        sent = True
        error = None
    except ProcessLookupError:
        sent = False
        error = None
    except BaseException as exc:
        sent = False
        error = f"{type(exc).__name__}: {exc}"
    actions.append(
        {
            "signal": "SIGKILL",
            "reason": reason,
            "time_unix": time.time(),
            "members_before": before,
            "sent": sent,
            "error": error,
        }
    )
    return int(os.waitpid(pid, 0)[1])


def independent_artifact_gate() -> dict[str, Any]:
    from PIL import Image
    import imageio_ffmpeg

    manifest_path = CHILD_OUTPUTS["rgb_frames_manifest.json"]
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    expected_names = [f"frame_{index:04d}.png" for index in range(EXPECTED_FRAMES)]
    actual_names = sorted(path.name for path in FRAME_DIR.iterdir()) if FRAME_DIR.is_dir() else []
    png_errors: list[str] = []
    png_decoded = 0
    for name in expected_names:
        path = FRAME_DIR / name
        try:
            with Image.open(path) as image:
                image.load()
                if (
                    image.format != "PNG"
                    or image.mode != "RGB"
                    or image.size != (VIDEO_WIDTH, VIDEO_HEIGHT)
                ):
                    raise RuntimeError(
                        f"format={image.format} mode={image.mode} size={image.size}"
                    )
            png_decoded += 1
        except BaseException as exc:
            if len(png_errors) < 10:
                png_errors.append(f"{name}:{type(exc).__name__}:{exc}")

    mp4_path = CHILD_OUTPUTS["side_grasp.mp4"]
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    mp4_decoded = 0
    mp4_error: str | None = None
    if mp4_path.is_file():
        try:
            decoded = subprocess.run(
                [
                    str(ffmpeg),
                    "-v",
                    "error",
                    "-i",
                    str(mp4_path),
                    "-map",
                    "0:v:0",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "framehash",
                    "-hash",
                    "sha256",
                    "pipe:1",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=180.0,
            )
            if decoded.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg_rc={decoded.returncode} stderr={decoded.stderr}"
                )
            mp4_decoded = sum(
                1
                for line in decoded.stdout.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        except BaseException as exc:
            mp4_error = f"{type(exc).__name__}: {exc}"
    else:
        mp4_error = "mp4_missing"

    phase_rows = read_phase_rows()
    checks = {
        "child_failure_absent": not CHILD_OUTPUTS["failure.json"].exists(),
        "manifest_nonempty_and_pass": bool(
            manifest_path.is_file()
            and manifest_path.stat().st_size > 0
            and manifest.get("pass") is True
        ),
        "manifest_non_scientific_exact": bool(
            manifest.get("scientific_authoritative") is False
            and manifest.get("render_is_posthoc_observability_only") is True
            and manifest.get("does_not_replace_or_complete_p13_terminal_attestation")
            is True
            and manifest.get("p13_observed_success_count") == 0
            and manifest.get("p13_observed_trial_count") == 5
            and manifest.get("p13_observed_verdict")
            == "NO_BILATERAL_SIDE_CONTACT"
        ),
        "frame_inventory_exact": actual_names == expected_names,
        "all_pngs_full_decode": png_decoded == EXPECTED_FRAMES and not png_errors,
        "mp4_full_decode_exact": mp4_decoded == EXPECTED_FRAMES and mp4_error is None,
        "manifest_mp4_hash_exact": bool(
            mp4_path.is_file()
            and manifest.get("mp4_sha256") == sha256_file(mp4_path)
            and manifest.get("frame_count") == EXPECTED_FRAMES
            and len(manifest.get("frames", [])) == EXPECTED_FRAMES
        ),
        "manifest_all_end_checks_true": bool(
            isinstance(manifest.get("end_checks"), dict)
            and manifest["end_checks"]
            and all(value is True for value in manifest["end_checks"].values())
        ),
        "phase_terminal_exact": bool(
            phase_rows and phase_rows[-1].get("phase") == "render_complete_durable"
        ),
        "frozen_source_snapshot_exact": bool(
            CHILD_OUTPUTS["script.py.txt"].is_file()
            and sha256_file(CHILD_OUTPUTS["script.py.txt"])
            == sha256_file(RENDER_SOURCE)
            == RENDER_SOURCE_SHA256
        ),
        "child_argv_exact": bool(
            CHILD_OUTPUTS["argv.txt"].is_file()
            and CHILD_OUTPUTS["argv.txt"].read_text().splitlines()
            == [str(RENDER_SOURCE), "--run"]
        ),
        "input_gate_pass": bool(
            CHILD_OUTPUTS["input_gate.json"].is_file()
            and json.loads(CHILD_OUTPUTS["input_gate.json"].read_text()).get("pass")
            is True
        ),
    }
    return {
        "artifact": "T3U_RENDER1_INDEPENDENT_ARTIFACT_GATE_V1",
        "checks": checks,
        "png_decoded": png_decoded,
        "png_errors": png_errors,
        "mp4_decoded": mp4_decoded,
        "mp4_error": mp4_error,
        "actual_frame_names": actual_names,
        "pass": bool(checks and all(checks.values())),
    }


def run() -> int:
    output_g0()
    if sha256_file(RENDER_SOURCE) != RENDER_SOURCE_SHA256:
        raise RuntimeError("RENDER1_SOURCE_PIN_DRIFT")
    if sha256_file(PREREG_PATH) != PREREG_SHA256:
        raise RuntimeError("RENDER1_PREREG_PIN_DRIFT")
    live_retired = [pid for pid in RETIRED_P13_HOST_PIDS if Path(f"/proc/{pid}").exists()]
    if live_retired:
        raise RuntimeError(f"ORIGINAL_P13_HOST_PIDS_STILL_LIVE {live_retired}")
    host_context = host_launch_context()
    if host_context.get("pass") is not True:
        raise RuntimeError("HOST_CONTEXT_FAIL")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    supervisor_pid = os.getpid()
    write_bytes_x(
        SUPERVISOR_OUTPUTS["supervisor_pid.txt"], f"{supervisor_pid}\n".encode()
    )
    gpu_before = gpu_inventory()
    write_bytes_x(SUPERVISOR_OUTPUTS["nvidia_smi_before.csv"], gpu_before)
    stdout_handle = SUPERVISOR_OUTPUTS["stdout.log"].open("xb", buffering=0)
    start_wall = time.time()
    start_monotonic = time.monotonic()
    command = [ISAAC_PYTHON, "-B", str(RENDER_SOURCE), "--run"]
    pid = os.fork()
    if pid == 0:
        try:
            child_setup(supervisor_pid)
            os.dup2(stdout_handle.fileno(), 1)
            os.dup2(stdout_handle.fileno(), 2)
            stdout_handle.close()
            os.execv(ISAAC_PYTHON, command)
        except BaseException:
            traceback.print_exc()
            os._exit(RESERVED_FAILURE_STATUS)
    stdout_handle.close()
    write_bytes_x(SUPERVISOR_OUTPUTS["render_python_pid.txt"], f"{pid}\n".encode())
    write_bytes_x(SUPERVISOR_OUTPUTS["pgid.txt"], f"{pid}\n".encode())

    raw_status: int | None = None
    timeout_reason: str | None = None
    capture_started_observed = False
    capture_started_monotonic: float | None = None
    last_phase_size = -1
    last_progress_monotonic = start_monotonic
    signal_actions: list[dict[str, Any]] = []
    while raw_status is None:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            raw_status = int(status)
            break
        now = time.monotonic()
        rows = read_phase_rows()
        phase_path = CHILD_OUTPUTS["phase.jsonl"]
        phase_size = phase_path.stat().st_size if phase_path.is_file() else -1
        if phase_size != last_phase_size:
            last_phase_size = phase_size
            last_progress_monotonic = now
        if not capture_started_observed and any(
            row.get("phase") == "capture_started" for row in rows
        ):
            capture_started_observed = True
            capture_started_monotonic = now
            last_progress_monotonic = now
        if _SUPERVISOR_SIGNAL is not None:
            timeout_reason = f"supervisor_{_SUPERVISOR_SIGNAL['signal_name']}"
        elif now - start_monotonic > GLOBAL_DEADLINE_S:
            timeout_reason = "global_deadline_900s"
        elif (
            not capture_started_observed
            and now - start_monotonic > LAUNCH_DEADLINE_S
        ):
            timeout_reason = "capture_start_deadline_120s"
        elif (
            capture_started_observed
            and now - last_progress_monotonic > NO_PROGRESS_DEADLINE_S
        ):
            timeout_reason = "post_capture_no_progress_deadline_90s"
        if timeout_reason is not None:
            raw_status = terminate_and_reap(pid, timeout_reason, signal_actions)
            break
        time.sleep(POLL_S)

    end_wall = time.time()
    elapsed = time.monotonic() - start_monotonic
    lifecycle = decode_wait_status(raw_status)
    group_members_after_reap = pgid_members(pid)
    artifact_gate = independent_artifact_gate()

    before_pids = gpu_pids(gpu_before)
    gpu_after = b""
    after_pids: set[int] = set()
    gpu_cleanup_deadline = time.monotonic() + 20.0
    while True:
        gpu_after = gpu_inventory()
        after_pids = gpu_pids(gpu_after)
        if not (after_pids - before_pids) or time.monotonic() >= gpu_cleanup_deadline:
            break
        time.sleep(0.5)
    write_bytes_x(SUPERVISOR_OUTPUTS["nvidia_smi_after.csv"], gpu_after)
    lifecycle_pass = bool(
        lifecycle["wifexited"] is True
        and lifecycle["exit_code"] == 0
        and timeout_reason is None
        and not signal_actions
        and not group_members_after_reap
    )
    gpu_pass = not (after_pids - before_pids)
    pass_all = bool(lifecycle_pass and artifact_gate["pass"] and gpu_pass)
    bindings = {
        name: {
            "path": str(path.relative_to(REPO)),
            "sha256": sha256_file(path),
        }
        for name, path in {**CHILD_OUTPUTS, **SUPERVISOR_OUTPUTS}.items()
        if path.is_file()
        and name not in {"supervisor_outcome.json", "exit_status.txt"}
    }
    if FRAME_DIR.is_dir():
        bindings["rgb_frames_directory"] = {
            "path": str(FRAME_DIR.relative_to(REPO)),
            "file_count": len(list(FRAME_DIR.iterdir())),
        }
    outcome = {
        "artifact": "T3U_P13_RTX_RENDER_REPAIR_SUPERVISOR_OUTCOME_V1",
        "scientific_authoritative": False,
        "render_is_posthoc_observability_only": True,
        "does_not_replace_or_complete_p13_terminal_attestation": True,
        "argv": list(sys.argv),
        "command": command,
        "supervisor_source_sha256": sha256_file(Path(__file__)),
        "render_source_sha256": sha256_file(RENDER_SOURCE),
        "prereg_sha256": sha256_file(PREREG_PATH),
        "host_launch_context": host_context,
        "original_p13_host_pids_required_absent": list(RETIRED_P13_HOST_PIDS),
        "start_time_unix": start_wall,
        "end_time_unix": end_wall,
        "elapsed_seconds": elapsed,
        "supervisor_pid": supervisor_pid,
        "render_pid": pid,
        "render_pgid": pid,
        "capture_started_observed": capture_started_observed,
        "capture_started_elapsed_seconds": (
            capture_started_monotonic - start_monotonic
            if capture_started_monotonic is not None
            else None
        ),
        "timeout_reason": timeout_reason,
        "signal_actions": signal_actions,
        "child_lifecycle": {
            **lifecycle,
            "group_members_after_reap": group_members_after_reap,
            "pass": lifecycle_pass,
        },
        "artifact_gate": artifact_gate,
        "gpu": {
            "before_pids": sorted(before_pids),
            "after_pids": sorted(after_pids),
            "fresh_after_pids": sorted(after_pids - before_pids),
            "pass": gpu_pass,
        },
        "contract": {
            "attempt_count": 1,
            "automatic_retry_count": 0,
            "capture_start_deadline_seconds": LAUNCH_DEADLINE_S,
            "post_capture_no_progress_deadline_seconds": NO_PROGRESS_DEADLINE_S,
            "global_deadline_seconds": GLOBAL_DEADLINE_S,
            "term_then_kill_after_seconds": KILL_AFTER_S,
            "child_new_session_and_pgid": True,
            "child_parent_death_signal": "SIGTERM",
            "host_boundary_forbids_bwrap_die_with_parent": True,
            "raw_waitpid_status_authority": True,
            "full_independent_png_and_mp4_decode": True,
            "no_fresh_gpu_pid": True,
        },
        "bindings": bindings,
        "pass": pass_all,
    }
    write_json_x(SUPERVISOR_OUTPUTS["supervisor_outcome.json"], outcome)
    write_bytes_x(
        SUPERVISOR_OUTPUTS["exit_status.txt"],
        ("0\n" if pass_all else f"{RESERVED_FAILURE_STATUS}\n").encode(),
    )
    return 0 if pass_all else RESERVED_FAILURE_STATUS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run or sys.argv[1:] != ["--run"]:
        raise RuntimeError("RENDER1_SUPERVISOR_EXACT_ARGV_REQUIRED --run")
    try:
        return run()
    except BaseException as exc:
        failure = {
            "artifact": "T3U_P13_RTX_RENDER_REPAIR_SUPERVISOR_FAILURE_V1",
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "source_sha256": sha256_file(Path(__file__)),
        }
        path = SUPERVISOR_OUTPUTS["supervisor_failure.json"]
        if not path.exists():
            write_json_x(path, failure)
        status_path = SUPERVISOR_OUTPUTS["exit_status.txt"]
        if not status_path.exists():
            write_bytes_x(status_path, f"{RESERVED_FAILURE_STATUS}\n".encode())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
