#!/usr/bin/env python3
"""D408 observability-only controller for immutable D407 evidence.

The controller never imports or launches Isaac/Kit/PhysX.  A future approved
execution may only project existing Rerun recording stores, render them with a
CPU software renderer, prepare bounded display-only force glyphs, and collect
an authenticated manual inspection result.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import hmac
import io
import json
import math
import os
import secrets
import select
import signal
import socket
import stat
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any


EXPECTED_PREREG_SHA256 = "0c0f1c03d10210e205d5be0b25fd84c7d94c109fb26387f77fa22f6b984c8d0d"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D407_ROOT = (
    PROJECT_ROOT
    / "claudedocs/runtime_logs/grasp_track/g0a_d407"
    / "attempt1_sdf_physics_ab_d362_remeasure"
)
D408_ROOT = (
    PROJECT_ROOT
    / "claudedocs/runtime_logs/grasp_track/g0a_d408"
    / "attempt1_d407_manual_observability_completion_repair"
)
PREREG_PATH = D408_ROOT / "d408_preregistration.json"
STATIC_RESULTS_PATH = D408_ROOT / "d408_static_fixture_results.json"
ATTESTATION_PATH = D408_ROOT / "d408_reviewed_script_attestation.json"
TUPLE_PATH = D408_ROOT / "d408_proposed_runtime_hash_tuple.json"

PHASE_PATH = D408_ROOT / "d408_controller_phase_markers.jsonl"
RUNTIME_PREREQUISITES_PATH = D408_ROOT / "d408_runtime_prerequisites.json"
SOURCE_CHECKPOINTS_PATH = D408_ROOT / "d408_source_immutability_checkpoints.json"
SCREENSHOT_MANIFEST_PATH = D408_ROOT / "d408_screenshot_manifest.json"
MANUAL_PATH = D408_ROOT / "d408_manual_visual_inspection.json"
MANUAL_RECEIPT_PATH = D408_ROOT / "d408_manual_writer_receipt.json"
AB_SHEET_PATH = D408_ROOT / "d408_ab_comparison_sheet.png"
TERMINAL_PATH = D408_ROOT / "d408_terminal_summary.json"
WRITER_LOG_PATH = D408_ROOT / "d408_manual_writer_stdout_stderr.log"
SCREENSHOT_CHECKPOINTS_PATH = D408_ROOT / "d408_screenshot_integrity_checkpoints.json"
MANUAL_PENDING_PATH = D408_ROOT / ".d408_manual_visual_inspection.json.pending"
TERMINAL_PENDING_PATH = D408_ROOT / ".d408_terminal_summary.json.pending"

CONTROLLER_PATH = Path(__file__).resolve()
WRITER_PATH = (
    PROJECT_ROOT
    / "sim_scripts/cyl34_top_view_d408_d407_manual_observability_completion_repair_manual_writer.py"
)
SESSION_DOC_PATH = (
    PROJECT_ROOT
    / "claudedocs/session_20260729_grasp_g0a_d408_manual_observability_completion_repair_static_prep.md"
)

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
ISAACLAB_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python3.11")
LVP_ICD = Path("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
LVP_LIBRARY = Path("/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so")
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

RERUN_VERSION = "0.34.1"
SCREENSHOT_LOGICAL_SIZE = "960x540"
SCREENSHOT_PHYSICAL_SIZE = (1920, 1080)
CLEAN_CROP_BOX = (0, 104, 1120, 1004)
CLEAN_CROP_SIZE = (1120, 900)
MANUAL_TIMEOUT_NS = 600_000_000_000
PREARM_HARD_TIMEOUT_NS = 1_200_000_000_000
WRITER_DEADLINE_LEAD_NS = 5_000_000_000
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_MANUAL_BYTES = 64 * 1024
MAX_SCREENSHOT_BYTES = 64 * 1024 * 1024
DISPLAY_FORCE_SCALE_PX_PER_N = 0.40
DISPLAY_FORCE_CAP_PX = 96.0
FORCE_NORM_ABS_TOLERANCE_N = 1.0e-9

D407_FINAL_VERDICT = "D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP"
D408_PASS_STATUS = "D408_D407_MANUAL_OBSERVABILITY_COMPLETION_REPAIR_PASS"
D408_FAIL_STATUS = "D408_D407_MANUAL_OBSERVABILITY_COMPLETION_REPAIR_FAIL"
RENAME_NOREPLACE = 1

SCIENTIFIC_NULL_CLAIMS = {
    "29x50mm_cylinder_transfer": None,
    "cap_rim_barrel_order": None,
    "collider_count_tipping_causality": None,
    "exact_face_or_manifold": None,
    "force_closure": None,
    "grasp_feasibility": None,
    "per_prim_cooked_sdf_identity": None,
    "sdf_general_superiority": None,
    "stable_grasp": None,
}

DROP_FORCE_ENTITIES = (
    "/contacts/gripper_link/force_display_scale",
    "/contacts/link5/force_display_scale",
    "/contacts/support_table/force_display_scale",
)

FORCE_FILTER_LABELS = (
    "gripper_link",
    "link4",
    "link5",
    "support_table",
)

EXPECTED_MANUAL_IMAGE_LAYOUT = {
    "leg_a_a64_replay/d408_clean_spatial.png": [1120, 900],
    "leg_a_a64_replay/d408_decision_sheet.png": [1920, 1080],
    "leg_b_sdf_res256_replay/d408_clean_spatial.png": [1120, 900],
    "leg_b_sdf_res256_replay/d408_decision_sheet.png": [1920, 1080],
    "d408_ab_comparison_sheet.png": [3840, 1080],
}

REQUIRED_BOOLEAN_FIELDS = (
    "leg_a_jaw_or_gripper_visible",
    "leg_a_cylinder_visible",
    "leg_a_timeseries_legible",
    "leg_a_no_notification_or_text_overlap",
    "leg_a_force_values_and_bounded_glyph_legible",
    "leg_b_jaw_or_gripper_visible",
    "leg_b_cylinder_visible",
    "leg_b_timeseries_legible",
    "leg_b_no_notification_or_text_overlap",
    "leg_b_force_values_and_bounded_glyph_legible",
    "ab_comparison_legible",
)

LEG_CONFIGS: dict[str, dict[str, Any]] = {
    "a": {
        "application_id": "roarm_g0a_d408_leg_a_observability",
        "directory": "leg_a_a64_replay",
        "label": "A64 control",
        "recording_id": "g0a_d408_leg_a_observability",
        "source_application_id": "roarm_g0a_d407_leg_a_physx_contact_motion",
        "source_directory": "leg_a_a64",
        "source_recording_id": "g0a_d407_leg_a_physx_contact_motion",
    },
    "b": {
        "application_id": "roarm_g0a_d408_leg_b_observability",
        "directory": "leg_b_sdf_res256_replay",
        "label": "SDF res256 treatment",
        "recording_id": "g0a_d408_leg_b_observability",
        "source_application_id": "roarm_g0a_d407_leg_b_physx_contact_motion",
        "source_directory": "leg_b_sdf_res256",
        "source_recording_id": "g0a_d407_leg_b_physx_contact_motion",
    },
}


class D408Error(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise D408Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise D408Error(f"non-finite JSON constant: {value}")


def _strict_json_bytes(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise D408Error("JSON is not UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise D408Error(f"strict JSON parse failed: {exc}") from exc


def _strict_json_path(path: Path, max_bytes: int = MAX_JSON_BYTES) -> Any:
    raw, _ = _secure_read_path(path, max_bytes)
    return _strict_json_bytes(raw)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise D408Error(f"unsafe hash source: {path}")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise D408Error(f"file changed while hashing: {path}")
    finally:
        os.close(fd)
    return digest.hexdigest()


def _secure_read_path(
    path: Path,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise D408Error(f"not a regular file: {path}")
        if before.st_nlink != 1:
            raise D408Error(f"unexpected link count: {path}")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise D408Error(f"unsafe file size: {path} ({before.st_size})")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if len(raw) > max_bytes:
            raise D408Error(f"file exceeds maximum size: {path}")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise D408Error(f"file changed while reading: {path}")
        if len(raw) != before.st_size:
            raise D408Error(f"short read: {path}")
        return raw, before
    finally:
        os.close(fd)


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise D408Error("short write")
        offset += written


def _write_bytes_x(path: Path, raw: bytes, mode: int = 0o600) -> None:
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        mode,
    )
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise D408Error(f"exclusive output is not a safe regular file: {path}")
        _write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json_x(path: Path, value: Any) -> None:
    _write_bytes_x(path, _canonical_bytes(value))


def _open_bound_root() -> tuple[int, int, int]:
    fd = os.open(
        D408_ROOT,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    metadata = os.fstat(fd)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(fd)
        raise D408Error("D408 root is not a directory")
    path_metadata = os.lstat(D408_ROOT)
    if stat.S_ISLNK(path_metadata.st_mode) or (
        path_metadata.st_dev,
        path_metadata.st_ino,
    ) != (metadata.st_dev, metadata.st_ino):
        os.close(fd)
        raise D408Error("D408 root path/dirfd identity mismatch")
    return fd, metadata.st_dev, metadata.st_ino


def _assert_root_identity(root_fd: int, expected_dev: int, expected_ino: int) -> None:
    descriptor = os.fstat(root_fd)
    path_metadata = os.lstat(D408_ROOT)
    expected = (expected_dev, expected_ino)
    if not stat.S_ISDIR(descriptor.st_mode):
        raise D408Error("bound D408 root fd is no longer a directory")
    if stat.S_ISLNK(path_metadata.st_mode):
        raise D408Error("D408 root path became a symlink")
    if (descriptor.st_dev, descriptor.st_ino) != expected or (
        path_metadata.st_dev,
        path_metadata.st_ino,
    ) != expected:
        raise D408Error("D408 root identity drift")


def _rename_noreplace_at(
    source_name: str,
    destination_name: str,
    directory_fd: int,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise D408Error("renameat2 is unavailable; fallback is forbidden")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        directory_fd,
        os.fsencode(source_name),
        directory_fd,
        os.fsencode(destination_name),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination_name)


def _atomic_publish_json_at(
    root_fd: int,
    pending_name: str,
    final_name: str,
    value: Any,
) -> None:
    raw = _canonical_bytes(value)
    pending_fd = os.open(
        pending_name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        0o600,
        dir_fd=root_fd,
    )
    try:
        metadata = os.fstat(pending_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise D408Error("terminal pending output is not a safe regular file")
        _write_all(pending_fd, raw)
        os.fsync(pending_fd)
    finally:
        os.close(pending_fd)
    _rename_noreplace_at(pending_name, final_name, root_fd)
    os.fsync(root_fd)


def _fsync_existing(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _copy_exact_x(source: Path, destination: Path) -> dict[str, Any]:
    source_raw, source_meta = _secure_read_path(source, 512 * 1024 * 1024)
    _write_bytes_x(destination, source_raw)
    destination_raw, destination_meta = _secure_read_path(
        destination,
        512 * 1024 * 1024,
    )
    if destination_raw != source_raw:
        raise D408Error(f"bit-exact copy mismatch: {destination}")
    return {
        "bytes": len(source_raw),
        "destination_sha256": _sha_bytes(destination_raw),
        "destination_size": destination_meta.st_size,
        "source_sha256": _sha_bytes(source_raw),
        "source_size": source_meta.st_size,
    }


def _open_relative_nofollow(
    root_fd: int,
    relative_path: str,
    final_flags: int,
) -> int:
    parts = Path(relative_path).parts
    if (
        not parts
        or Path(relative_path).is_absolute()
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise D408Error(f"unsafe D408-root-relative path: {relative_path!r}")
    current_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        result_fd = os.open(
            parts[-1],
            final_flags | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=current_fd,
        )
    finally:
        os.close(current_fd)
    return result_fd


def _secure_read_relative(
    root_fd: int,
    relative_path: str,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    fd = _open_relative_nofollow(root_fd, relative_path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise D408Error(f"unsafe relative file identity: {relative_path}")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise D408Error(
                f"unsafe relative file size: {relative_path} ({before.st_size})"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise D408Error(f"relative file changed while reading: {relative_path}")
        if len(raw) != before.st_size or len(raw) > max_bytes:
            raise D408Error(f"relative file short/oversize read: {relative_path}")
        return raw, before
    finally:
        os.close(fd)


def _proc_start_ticks(pid: int) -> int:
    path = Path(f"/proc/{pid}/stat")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise D408Error(f"unsafe proc stat identity: {path}")
        raw = os.read(fd, 64 * 1024)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise D408Error(f"proc stat identity changed: {path}")
    finally:
        os.close(fd)
    if not raw or len(raw) >= 64 * 1024:
        raise D408Error(f"invalid proc stat size: {path}")
    text = raw.decode("utf-8")
    right_paren = text.rfind(")")
    if right_paren < 0:
        raise D408Error(f"malformed /proc/{pid}/stat")
    fields_after_comm = text[right_paren + 1 :].strip().split()
    if len(fields_after_comm) <= 19:
        raise D408Error(f"short /proc/{pid}/stat")
    return int(fields_after_comm[19])


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _manifest_d407() -> tuple[list[dict[str, Any]], list[str], str]:
    if not D407_ROOT.is_dir() or D407_ROOT.is_symlink():
        raise D408Error("D407 root is missing, not a directory, or a symlink")
    rows: list[dict[str, Any]] = []
    directories: list[str] = []

    def visit(directory: Path, relative_prefix: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            relative = relative_prefix / entry.name
            if entry.is_symlink():
                raise D408Error(f"D407 source contains symlink: {relative}")
            if entry.is_dir(follow_symlinks=False):
                directories.append(str(relative))
                visit(Path(entry.path), relative)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise D408Error(f"D407 source contains special entry: {relative}")
            raw, metadata = _secure_read_path(Path(entry.path), 512 * 1024 * 1024)
            rows.append(
                {
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "nlink": metadata.st_nlink,
                    "path": str(relative),
                    "regular": stat.S_ISREG(metadata.st_mode),
                    "sha256": _sha_bytes(raw),
                    "size": metadata.st_size,
                    "symlink": False,
                }
            )

    visit(D407_ROOT, Path())
    rows.sort(key=lambda row: row["path"])
    directories.sort()
    manifest = {"directories": directories, "files": rows}
    return rows, directories, _sha_bytes(_canonical_bytes(manifest))


def _validate_manifest(
    prereg: dict[str, Any],
    checkpoint: str,
    checkpoints: list[dict[str, Any]],
) -> None:
    rows, directories, manifest_sha = _manifest_d407()
    if rows != prereg.get("d407_source_manifest"):
        raise D408Error(f"D407 source manifest drift at {checkpoint}")
    if directories != prereg.get("d407_source_directories"):
        raise D408Error(f"D407 source directory-set drift at {checkpoint}")
    if manifest_sha != prereg.get("d407_source_manifest_sha256"):
        raise D408Error(f"D407 source manifest SHA drift at {checkpoint}")
    checkpoints.append(
        {
            "checkpoint": checkpoint,
            "directory_count": len(directories),
            "file_count": len(rows),
            "manifest_sha256": manifest_sha,
            "monotonic_ns": time.monotonic_ns(),
        }
    )


class PhaseLog:
    def __init__(self, root_fd: int) -> None:
        root_metadata = os.fstat(root_fd)
        self.root_dev = root_metadata.st_dev
        self.root_ino = root_metadata.st_ino
        self.fd = os.open(
            PHASE_PATH.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        metadata = os.fstat(self.fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise D408Error("phase log is not an exclusive regular file")
        os.fsync(root_fd)
        self.dev = metadata.st_dev
        self.ino = metadata.st_ino
        self.sequence = 0
        self.previous_sha: str | None = None

    def append(self, event: str, details: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        core = {
            "artifact": "D408_CONTROLLER_PHASE_ROW_V1",
            "details": details,
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "prev_row_sha256": self.previous_sha,
            "sequence": self.sequence,
            "utc": _utc_now(),
        }
        row_sha = _sha_bytes(_canonical_bytes(core))
        row = {**core, "row_sha256": row_sha}
        raw = _canonical_bytes(row)
        written = os.write(self.fd, raw)
        if written != len(raw):
            raise D408Error("phase row was not committed by one complete os.write")
        os.fsync(self.fd)
        self.previous_sha = row_sha
        return row

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _recv_json_line(channel: socket.socket, max_bytes: int = 64 * 1024) -> dict[str, Any]:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = channel.recv(4096)
        if not chunk:
            raise D408Error("socket closed before a complete JSON line")
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            raise D408Error("protocol message exceeds size limit")
        raw = b"".join(chunks)
        newline = raw.find(b"\n")
        if newline >= 0:
            if raw[newline + 1 :]:
                raise D408Error("multiple protocol messages in one read are forbidden")
            value = _strict_json_bytes(raw[: newline + 1])
            if not isinstance(value, dict):
                raise D408Error("protocol message must be an object")
            return value


def _send_json_line(channel: socket.socket, value: dict[str, Any]) -> None:
    raw = _canonical_bytes(value)
    if len(raw) > 64 * 1024:
        raise D408Error("outgoing protocol message exceeds size limit")
    channel.sendall(raw)


def _hmac_hex(nonce: bytes, body: dict[str, Any]) -> str:
    return hmac.new(nonce, _canonical_bytes(body), hashlib.sha256).hexdigest()


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise D408Error(
            f"{label} keys mismatch: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )


def _runtime_leaf_paths() -> list[str]:
    root = D408_ROOT.relative_to(PROJECT_ROOT)
    paths = [
        root / PHASE_PATH.name,
        root / RUNTIME_PREREQUISITES_PATH.name,
        root / SOURCE_CHECKPOINTS_PATH.name,
        root / SCREENSHOT_CHECKPOINTS_PATH.name,
        root / SCREENSHOT_MANIFEST_PATH.name,
        root / MANUAL_PATH.name,
        root / MANUAL_PENDING_PATH.name,
        root / MANUAL_RECEIPT_PATH.name,
        root / AB_SHEET_PATH.name,
        root / TERMINAL_PATH.name,
        root / TERMINAL_PENDING_PATH.name,
        root / WRITER_LOG_PATH.name,
    ]
    for leg in LEG_CONFIGS.values():
        leg_root = root / leg["directory"]
        for name in (
            "d407_source_trace.json",
            "d407_source_recording.rrd",
            "d407_source_blueprint.rbl",
            "d408_presentation_recording.rrd",
            "d408_clean_spatial.rbl",
            "d408_rerun_full_diagnostic.png",
            "d408_clean_spatial.png",
            "d408_decision_sheet.png",
            "d408_rerun_validation.json",
        ):
            paths.append(leg_root / name)
    return sorted(str(path) for path in paths)


def _runtime_directory_paths() -> list[str]:
    root = D408_ROOT.relative_to(PROJECT_ROOT)
    return sorted(
        str(root / LEG_CONFIGS[leg]["directory"])
        for leg in ("a", "b")
    )


def _planned_static_paths() -> list[str]:
    return sorted(
        [
            _rel(PREREG_PATH),
            _rel(STATIC_RESULTS_PATH),
            _rel(ATTESTATION_PATH),
            _rel(TUPLE_PATH),
            _rel(CONTROLLER_PATH),
            _rel(WRITER_PATH),
            _rel(SESSION_DOC_PATH),
            "START_HERE.md",
            "claudedocs/BACKLOG.md",
            "claudedocs/DECISIONS.md",
            "claudedocs/EXPERIMENT_LEDGER.md",
        ]
    )


def _git_dirty_paths() -> list[str]:
    command = [
        "git",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        raise D408Error(
            "git dirty overlay query failed: "
            + result.stderr.decode("utf-8", errors="replace")
        )
    entries = result.stdout.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2:3] != b" ":
            raise D408Error("malformed git porcelain-v1 -z record")
        status_code = entry[:2]
        path_raw = entry[3:]
        try:
            path = path_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise D408Error("git dirty path is not UTF-8") from exc
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise D408Error(f"unsafe git dirty path: {path!r}")
        paths.append(path)
        if b"R" in status_code or b"C" in status_code:
            if index >= len(entries) or not entries[index]:
                raise D408Error("rename/copy porcelain record is missing source path")
            try:
                source_path = entries[index].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise D408Error("git rename source path is not UTF-8") from exc
            index += 1
            if Path(source_path).is_absolute() or ".." in Path(source_path).parts:
                raise D408Error(f"unsafe git rename source path: {source_path!r}")
            paths.append(source_path)
    if len(paths) != len(set(paths)):
        raise D408Error("git dirty overlay contains duplicate paths")
    return sorted(paths)


def _physical_d408_tree(root_fd: int) -> dict[str, list[str]]:
    result = {
        "directories": [],
        "files": [],
        "special": [],
        "symlinks": [],
    }

    def visit(directory_fd: int, prefix: Path) -> None:
        for name in sorted(os.listdir(directory_fd)):
            relative = prefix / name
            relative_text = str(relative)
            metadata = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(metadata.st_mode):
                result["symlinks"].append(relative_text)
            elif stat.S_ISDIR(metadata.st_mode):
                result["directories"].append(relative_text)
                child_fd = os.open(
                    name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    child_metadata = os.fstat(child_fd)
                    if (
                        child_metadata.st_dev,
                        child_metadata.st_ino,
                    ) != (
                        metadata.st_dev,
                        metadata.st_ino,
                    ):
                        raise D408Error(
                            f"D408 directory identity changed: {relative_text}"
                        )
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    result["special"].append(relative_text)
                else:
                    result["files"].append(relative_text)
            else:
                result["special"].append(relative_text)

    visit(root_fd, Path())
    for key in result:
        result[key].sort()
    return result


def _validate_repository_overlay(
    prereg: dict[str, Any],
    root_fd: int,
    root_dev: int,
    root_ino: int,
    checkpoint: str,
    *,
    admission_exact: bool,
    stable_publication_gate: bool = False,
) -> dict[str, Any]:
    _assert_root_identity(root_fd, root_dev, root_ino)
    dirty = _git_dirty_paths()
    expected_pre_runtime = prereg.get("expected_pre_runtime_dirty_paths")
    allowed_dirty = prereg.get("allowed_dirty_paths")
    if not isinstance(expected_pre_runtime, list) or not isinstance(
        allowed_dirty, list
    ):
        raise D408Error("prereg dirty overlay fields are missing")
    if admission_exact:
        if dirty != expected_pre_runtime:
            raise D408Error(
                f"git dirty overlay mismatch at {checkpoint}: "
                f"expected={len(expected_pre_runtime)} observed={len(dirty)}"
            )
    elif not set(dirty).issubset(set(allowed_dirty)):
        raise D408Error(
            f"git dirty overlay escaped allowlist at {checkpoint}: "
            f"{sorted(set(dirty) - set(allowed_dirty))}"
        )

    tree = _physical_d408_tree(root_fd)
    if tree["symlinks"] or tree["special"]:
        raise D408Error(
            f"unsafe D408 physical tree at {checkpoint}: "
            f"symlinks={tree['symlinks']} special={tree['special']}"
        )
    expected_static_files = prereg.get("d408_expected_static_file_paths")
    allowed_files = prereg.get("d408_allowed_file_paths")
    expected_static_directories = prereg.get("d408_expected_static_directories")
    allowed_directories = prereg.get("d408_allowed_directories")
    if not all(
        isinstance(value, list)
        for value in (
            expected_static_files,
            allowed_files,
            expected_static_directories,
            allowed_directories,
        )
    ):
        raise D408Error("prereg physical-tree allowlist fields are missing")
    if admission_exact:
        if tree["files"] != expected_static_files:
            raise D408Error(
                f"D408 static file tree mismatch at {checkpoint}: "
                f"expected={expected_static_files} observed={tree['files']}"
            )
        if tree["directories"] != expected_static_directories:
            raise D408Error(
                f"D408 static directory tree mismatch at {checkpoint}: "
                f"expected={expected_static_directories} "
                f"observed={tree['directories']}"
            )
    else:
        if not set(expected_static_files).issubset(tree["files"]):
            raise D408Error(f"D408 static file disappeared at {checkpoint}")
        if not set(tree["files"]).issubset(set(allowed_files)):
            raise D408Error(
                f"D408 physical file escaped allowlist at {checkpoint}: "
                f"{sorted(set(tree['files']) - set(allowed_files))}"
            )
        if not set(expected_static_directories).issubset(tree["directories"]):
            raise D408Error(f"D408 static directory disappeared at {checkpoint}")
        if not set(tree["directories"]).issubset(set(allowed_directories)):
            raise D408Error(
                f"D408 physical directory escaped allowlist at {checkpoint}: "
                f"{sorted(set(tree['directories']) - set(allowed_directories))}"
            )
    if stable_publication_gate:
        forbidden = {
            MANUAL_PENDING_PATH.name,
            TERMINAL_PATH.name,
            TERMINAL_PENDING_PATH.name,
        }
        observed_forbidden = sorted(forbidden.intersection(tree["files"]))
        if observed_forbidden:
            raise D408Error(
                f"D408 stable publication gate failed at {checkpoint}: "
                f"{observed_forbidden}"
            )
    return {
        "checkpoint": checkpoint,
        "dirty_path_count": len(dirty),
        "physical_directory_count": len(tree["directories"]),
        "physical_file_count": len(tree["files"]),
    }


def _contract() -> dict[str, Any]:
    writer_command = [
        str(ISAACLAB_PYTHON),
        "-B",
        str(WRITER_PATH),
        "--stage",
        "writer",
        "--socket-fd",
        "<inherited_fd>",
        "--root",
        str(D408_ROOT),
        "--root-fd",
        "<inherited_root_fd>",
        "--root-dev",
        "<root_dev>",
        "--root-ino",
        "<root_ino>",
        "--controller-pid",
        "<controller_pid>",
        "--controller-start-ticks",
        "<controller_start_ticks>",
        "--controller-sha256",
        "<controller_sha256>",
        "--writer-sha256",
        "<writer_sha256>",
        "--approved-tuple-sha256",
        "<tuple_file_sha256>",
        "--phase-dev",
        "<phase_dev>",
        "--phase-ino",
        "<phase_ino>",
        "--input-manifest-sha256",
        "<d407_manifest_sha256>",
        "--manual-basename",
        MANUAL_PATH.name,
        "--prearm-hard-deadline-monotonic-ns",
        "<prearm_hard_deadline_ns>",
    ]
    verify_commands = []
    screenshot_commands = []
    for leg in LEG_CONFIGS.values():
        leg_root = D408_ROOT / leg["directory"]
        verify_commands.append(
            {
                "argv": [
                    str(RERUN_CLI),
                    "rrd",
                    "verify",
                    "--check-footers",
                    "true",
                    str(leg_root / "d408_presentation_recording.rrd"),
                    str(leg_root / "d408_clean_spatial.rbl"),
                ],
                "environment": "inherit",
                "shell": False,
                "timeout_seconds": 60,
            }
        )
        screenshot_commands.append(
            {
                "argv": [
                str(RERUN_CLI),
                "--headless",
                "--hide-welcome-screen",
                "--window-size",
                SCREENSHOT_LOGICAL_SIZE,
                "--screenshot-to",
                str(leg_root / "d408_rerun_full_diagnostic.png"),
                str(leg_root / "d408_presentation_recording.rrd"),
                str(leg_root / "d408_clean_spatial.rbl"),
                ],
                "environment": {
                    "policy": "inherit_plus_exact_overrides",
                    "overrides": {
                        "VK_ICD_FILENAMES": str(LVP_ICD),
                        "WGPU_BACKEND": "vulkan",
                        "WGPU_POWER_PREF": "low",
                    },
                },
                "shell": False,
                "timeout_seconds": 120,
            }
        )
    return {
        "artifact": "D408_CONTROLLER_CONTRACT_V1",
        "case_id": "D408",
        "deadline_contract": {
            "manual_timeout_ns": MANUAL_TIMEOUT_NS,
            "prearm_hard_timeout_ns": PREARM_HARD_TIMEOUT_NS,
            "writer_deadline_lead_ns": WRITER_DEADLINE_LEAD_NS,
        },
        "expected_prereg_sha256": EXPECTED_PREREG_SHA256,
        "expected_runtime_counters": _runtime_counters(),
        "future_runtime_output_paths": _runtime_leaf_paths(),
        "future_runtime_directory_paths": _runtime_directory_paths(),
        "manual_required_boolean_fields": list(REQUIRED_BOOLEAN_FIELDS),
        "planned_static_output_paths": _planned_static_paths(),
        "retry_count": 0,
        "scientific_null_claims": SCIENTIFIC_NULL_CLAIMS,
        "status_literals": [
            "STATIC_PREPARED_RUNTIME_NOT_APPROVED",
            "PROPOSED_NOT_EXECUTED",
            D408_PASS_STATUS,
            D408_FAIL_STATUS,
        ],
        "subprocess_contracts": {
            "git_status": {
                "argv": [
                    "git",
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                ],
                "environment": "inherit",
                "shell": False,
                "timeout_seconds": 30,
            },
            "rerun_screenshot": screenshot_commands,
            "rerun_verify": verify_commands,
            "rerun_version": {
                "argv": [str(RERUN_CLI), "--version"],
                "environment": "inherit",
                "shell": False,
                "timeout_seconds": 15,
            },
            "writer_contract_print": {
                "argv": [
                    str(ISAACLAB_PYTHON),
                    "-B",
                    str(WRITER_PATH),
                    "--print-contract-json",
                ],
                "environment": "inherit",
                "shell": False,
                "timeout_seconds": 15,
            },
            "writer_runtime": {
                "argv": writer_command,
                "environment": "inherit",
                "pass_fds": ["<socket_fd>", "<inherited_root_fd>"],
                "shell": False,
                "start_new_session": True,
                "stderr": "STDOUT",
                "stdin": "DEVNULL",
                "stdout": str(WRITER_LOG_PATH),
                "timeout_policy": {
                    "ack_send_timeout_seconds": 5,
                    "controller_ready_timeout_seconds": 15,
                    "controller_ping_timeout_seconds": 15,
                    "overall_writer_wait_deadline": (
                        "prearm_hard_deadline_monotonic_ns"
                        "+manual_timeout_ns"
                    ),
                    "publication_deadline": (
                        "manual_prompt_started_monotonic_ns"
                        "+manual_timeout_ns-writer_deadline_lead_ns"
                    ),
                    "success_exit_wait_seconds": 5,
                    "terminate_then_kill_wait_seconds_each": 2,
                    "writer_arm_receive_timeout_seconds": 15,
                },
                "timeout_seconds": None,
            },
        },
    }


def _normalized_contract(value: dict[str, Any]) -> dict[str, Any]:
    normalized = _strict_json_bytes(_canonical_bytes(value))
    if not isinstance(normalized, dict):
        raise D408Error("contract normalization input is not an object")
    if "expected_prereg_sha256" not in normalized:
        raise D408Error("contract lacks expected_prereg_sha256")
    normalized["expected_prereg_sha256"] = "__D408_PREREG_SHA256__"
    return normalized


def _validate_static_authority(
    approved_tuple_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    if _sha_path(PREREG_PATH) != EXPECTED_PREREG_SHA256:
        raise D408Error("preregistration SHA mismatch")
    prereg = _strict_json_path(PREREG_PATH)
    if not isinstance(prereg, dict):
        raise D408Error("preregistration is not an object")
    if prereg.get("status") == "STATIC_PREPARED_RUNTIME_NOT_APPROVED":
        pass
    else:
        raise D408Error("preregistration status is not exact")
    if prereg.get("actual_execution_requires_separate_tuple_sha_approval") is not True:
        raise D408Error("separate approval boundary is absent")
    if prereg.get("expected_runtime_counters") != _runtime_counters():
        raise D408Error("preregistered runtime counters are not exact")
    if prereg.get("controller_contract_normalized") != _normalized_contract(
        _contract()
    ):
        raise D408Error("live controller contract differs from preregistration")

    writer_contract_result = subprocess.run(
        [
            str(ISAACLAB_PYTHON),
            "-B",
            str(WRITER_PATH),
            "--print-contract-json",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
        check=False,
        shell=False,
    )
    if writer_contract_result.returncode != 0:
        raise D408Error("manual writer contract query failed")
    writer_contract = _strict_json_bytes(writer_contract_result.stdout)
    if not isinstance(writer_contract, dict):
        raise D408Error("manual writer contract is not an object")
    if prereg.get("manual_writer_contract_normalized") != _normalized_contract(
        writer_contract
    ):
        raise D408Error("live manual-writer contract differs from preregistration")

    attestation = _strict_json_path(ATTESTATION_PATH)
    tuple_data = _strict_json_path(TUPLE_PATH)
    static_results = _strict_json_path(STATIC_RESULTS_PATH)
    if not all(
        isinstance(value, dict)
        for value in (attestation, tuple_data, static_results)
    ):
        raise D408Error("attestation, tuple, or static results is not an object")
    if static_results.get("overall_pass") is not True:
        raise D408Error("static fixture results are not an overall PASS")
    if tuple_data.get("execution_status") == "PROPOSED_NOT_EXECUTED":
        pass
    else:
        raise D408Error("tuple execution status is not exact")
    tuple_sha = _sha_path(TUPLE_PATH)
    if tuple_sha != approved_tuple_sha256:
        raise D408Error("command-line approved tuple SHA mismatch")

    controller_sha = _sha_path(CONTROLLER_PATH)
    writer_sha = _sha_path(WRITER_PATH)
    hashes = tuple_data.get("hashes")
    if not isinstance(hashes, dict):
        raise D408Error("tuple hashes are missing")
    reviewed_hashes = {
        "controller_sha256": controller_sha,
        "manual_writer_sha256": writer_sha,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
    }
    expected = {
        "attestation_sha256": _sha_path(ATTESTATION_PATH),
        **reviewed_hashes,
    }
    if hashes != expected:
        raise D408Error("4-SHA tuple does not bind the live files")
    if attestation.get("reviewed_hashes") != reviewed_hashes:
        raise D408Error("attestation does not bind the live files")
    if (
        attestation.get("static_fixture_results_sha256")
        != _sha_path(STATIC_RESULTS_PATH)
        or attestation.get("static_fixture_overall_pass") is not True
    ):
        raise D408Error("attestation does not bind the passing static results")
    normalized_contract_hashes = {
        "controller_contract_normalized_sha256": _sha_bytes(
            _canonical_bytes(_normalized_contract(_contract()))
        ),
        "manual_writer_contract_normalized_sha256": _sha_bytes(
            _canonical_bytes(_normalized_contract(writer_contract))
        ),
    }
    if (
        attestation.get("normalized_contract_hashes")
        != normalized_contract_hashes
    ):
        raise D408Error("attestation normalized-contract binding mismatch")
    return prereg, attestation, tuple_data, controller_sha, writer_sha


def _runtime_absence_gate() -> None:
    for relative in _runtime_leaf_paths():
        path = PROJECT_ROOT / relative
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"runtime output already exists: {path}")
    for leg in LEG_CONFIGS.values():
        path = D408_ROOT / leg["directory"]
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"runtime leg directory already exists: {path}")


def _verify_software_stack(prereg: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "font_sha256": _sha_path(FONT_PATH),
        "lvp_icd_sha256": _sha_path(LVP_ICD),
        "lvp_library_sha256": _sha_path(LVP_LIBRARY),
        "python_sha256": _sha_path(ISAACLAB_PYTHON),
        "rerun_cli_sha256": _sha_path(RERUN_CLI),
    }
    if observed != prereg.get("installed_software_render_stack"):
        raise D408Error("installed software-render stack drift")
    command = subprocess.run(
        [str(RERUN_CLI), "--version"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=15,
        check=False,
        shell=False,
    )
    if command.returncode != 0 or "rerun-cli 0.34.1" not in command.stdout:
        raise D408Error("Rerun version preflight failed")
    return {
        **observed,
        "rerun_version_output": command.stdout.strip(),
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _all_finite(item) for key, item in value.items())
    return False


def _load_trace(path: Path) -> list[dict[str, Any]]:
    value = _strict_json_path(path, 128 * 1024 * 1024)
    rows = value
    if not isinstance(rows, list) or len(rows) != 500:
        raise D408Error(f"trace does not contain exactly 500 rows: {path}")
    if not all(isinstance(row, dict) and _all_finite(row) for row in rows):
        raise D408Error(f"trace contains non-finite or unsupported values: {path}")
    return rows


def _summary_entities(summary: str) -> set[str]:
    entities: set[str] = set()
    for line in summary.splitlines():
        if " rows=" not in line:
            continue
        entity = line.split(" rows=", 1)[0].strip()
        if entity.startswith("/"):
            entities.add(entity)
    return entities


def _semantic_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return {
            "temporal_iso8601": value.isoformat(),
            "type": type(value).__name__,
        }
    if isinstance(value, dt.timedelta):
        return {
            "timedelta_days": value.days,
            "timedelta_microseconds": value.microseconds,
            "timedelta_seconds": value.seconds,
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise D408Error("RRD semantic inventory contains non-finite float")
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, list):
        return [_semantic_scalar(item) for item in value]
    if isinstance(value, tuple):
        return [_semantic_scalar(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise D408Error("RRD semantic inventory contains non-string map key")
        return {
            key: _semantic_scalar(value[key])
            for key in sorted(value)
        }
    raise D408Error(f"unsupported RRD semantic scalar: {type(value).__name__}")


def _field_metadata(field: Any) -> dict[str, str]:
    metadata = field.metadata or {}
    result: dict[str, str] = {}
    for raw_key, raw_value in metadata.items():
        try:
            key = raw_key.decode("utf-8")
            value = raw_value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise D408Error("RRD field metadata is not UTF-8") from exc
        result[key] = value
    return {key: result[key] for key in sorted(result)}


def _rrd_semantic_inventory(
    reader: Any,
    recording: Any,
    *,
    drop_entities: set[str],
    drop_component: str | None,
) -> dict[str, Any]:
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    descriptors: dict[str, dict[str, Any]] = {}
    system_entity_paths: set[str] = set()
    for chunk in reader.stream(store=recording):
        entity_path = str(chunk.entity_path)
        if entity_path.startswith("/__"):
            system_entity_paths.add(entity_path)
        if entity_path in drop_entities:
            continue
        batch = chunk.to_record_batch()
        fields = list(batch.schema)
        row_id_indices = [
            index
            for index, field in enumerate(fields)
            if field.name == "rerun.controls.RowId"
        ]
        if row_id_indices != [0]:
            raise D408Error(
                f"RRD chunk has unexpected RowId columns at {entity_path}: "
                f"{row_id_indices}"
            )
        row_ids = batch.column(0).to_pylist()
        if len(row_ids) != batch.num_rows:
            raise D408Error("RRD RowId length mismatch")
        for row_index, row_id in enumerate(row_ids):
            if not isinstance(row_id, bytes) or len(row_id) != 16:
                raise D408Error("RRD RowId is not exactly 16 bytes")
            row_key = (entity_path, row_id.hex())
            row = rows_by_key.setdefault(
                row_key,
                {
                    "columns": {},
                    "entity_path": entity_path,
                    "row_id_hex": row_id.hex(),
                },
            )
            for column_index, field in enumerate(fields[1:], start=1):
                if drop_component is not None and field.name == drop_component:
                    continue
                descriptor = {
                    "metadata": _field_metadata(field),
                    "name": field.name,
                    "type": str(field.type),
                }
                descriptor_key = _sha_bytes(_canonical_bytes(descriptor))
                previous_descriptor = descriptors.setdefault(descriptor_key, descriptor)
                if previous_descriptor != descriptor:
                    raise D408Error("RRD descriptor digest collision")
                value = _semantic_scalar(batch.column(column_index)[row_index].as_py())
                if (
                    descriptor_key in row["columns"]
                    and row["columns"][descriptor_key] != value
                ):
                    raise D408Error(
                        f"conflicting duplicate RRD value at {entity_path}/{row_id.hex()}"
                    )
                row["columns"][descriptor_key] = value

    canonical_rows = []
    component_names: set[str] = set()
    timeline_names: set[str] = set()
    for row_key in sorted(rows_by_key):
        row = rows_by_key[row_key]
        ordered_columns = {
            key: row["columns"][key]
            for key in sorted(row["columns"])
        }
        for descriptor_key in ordered_columns:
            descriptor = descriptors[descriptor_key]
            kind = descriptor["metadata"].get("rerun:kind")
            if kind == "index":
                timeline_names.add(descriptor["name"])
            elif kind == "data":
                component_names.add(descriptor["name"])
        canonical_rows.append(
            {
                "columns": ordered_columns,
                "entity_path": row["entity_path"],
                "row_id_hex": row["row_id_hex"],
            }
        )
    canonical = {
        "descriptors": {
            key: descriptors[key]
            for key in sorted(descriptors)
            if any(key in row["columns"] for row in canonical_rows)
        },
        "rows": canonical_rows,
    }
    return {
        "canonical": canonical,
        "component_names": sorted(component_names),
        "digest_sha256": _sha_bytes(_canonical_bytes(canonical)),
        "entity_paths": sorted({row["entity_path"] for row in canonical_rows}),
        "row_count": len(canonical_rows),
        "system_entity_paths": sorted(system_entity_paths),
        "timeline_names": sorted(timeline_names),
        "value_cell_count": sum(len(row["columns"]) for row in canonical_rows),
    }


def _notification_overlay_detected(image: Any) -> bool:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    candidate_rows: list[tuple[int, int, int]] = []
    for y in range(0, height, 2):
        run_start: int | None = None
        for x in range(0, width, 2):
            red, green, blue = pixels[x, y]
            neutral_dark = (
                18 <= red <= 76
                and 18 <= green <= 76
                and 18 <= blue <= 76
                and max(red, green, blue) - min(red, green, blue) <= 9
            )
            if neutral_dark and run_start is None:
                run_start = x
            if (not neutral_dark or x >= width - 2) and run_start is not None:
                run_end = x if not neutral_dark else width
                if run_end - run_start >= 280:
                    candidate_rows.append((y, run_start, run_end))
                run_start = None
    for index, (y, left, right) in enumerate(candidate_rows):
        compatible = [
            row
            for row in candidate_rows[index : index + 40]
            if row[0] - y <= 80
            and min(right, row[2]) - max(left, row[1]) >= 240
        ]
        if len(compatible) < 12:
            continue
        top = y
        bottom = min(height, compatible[-1][0] + 4)
        icon_right = min(width, left + 120)
        colored = 0
        for yy in range(top, bottom):
            for xx in range(left, icon_right):
                red, green, blue = pixels[xx, yy]
                info_blue = blue >= 175 and blue - red >= 25
                error_red = red >= 175 and red - green >= 45
                if info_blue or error_red:
                    colored += 1
        if colored >= 20:
            return True
    return False


def _color_witnesses(image: Any) -> dict[str, int]:
    rgb = image.convert("RGB")
    blue_count = 0
    red_count = 0
    orange_count = 0
    for red, green, blue in rgb.getdata():
        if blue >= 150 and green >= 80 and blue > red + 35:
            blue_count += 1
        if red >= 165 and red > green + 40 and red > blue + 20:
            red_count += 1
        if red >= 145 and 70 <= green <= 190 and blue <= 115 and red > blue + 55:
            orange_count += 1
    return {
        "blue_link_pixels": blue_count,
        "orange_cylinder_pixels": orange_count,
        "red_gripper_pixels": red_count,
    }


def _bounded_glyph(
    force_xyz_n: list[float],
    stored_norm_n: float,
    *,
    inset: tuple[float, float, float, float],
    center: tuple[float, float],
) -> dict[str, Any]:
    if len(force_xyz_n) != 3:
        raise D408Error("force vector must contain exactly three values")
    if any(type(value) not in (int, float) for value in force_xyz_n):
        raise D408Error("force vector values must be raw JSON numbers")
    if type(stored_norm_n) not in (int, float):
        raise D408Error("stored force norm must be a raw JSON number")
    vector = [float(value) for value in force_xyz_n]
    if not all(math.isfinite(value) for value in vector):
        raise D408Error("force vector contains a non-finite value")
    stored_norm = float(stored_norm_n)
    if not math.isfinite(stored_norm) or stored_norm < 0.0:
        raise D408Error("stored force norm is non-finite or negative")
    recomputed_norm = math.sqrt(sum(value * value for value in vector))
    norm_error = abs(recomputed_norm - stored_norm)
    if norm_error > FORCE_NORM_ABS_TOLERANCE_N:
        raise D408Error(
            f"stored/recomputed force norm mismatch: {norm_error} N"
        )
    raw_x = vector[0] - 0.5 * vector[1]
    raw_y = -vector[2] + 0.5 * vector[1]
    projected_norm = math.hypot(raw_x, raw_y)
    if stored_norm == 0.0:
        direction = (0.0, 0.0)
    elif projected_norm <= 1.0e-12:
        direction = (0.0, -1.0)
    else:
        direction = (raw_x / projected_norm, raw_y / projected_norm)
    uncapped_length = DISPLAY_FORCE_SCALE_PX_PER_N * stored_norm
    margin = 8.0
    maximum_inset_length = float("inf")
    for coordinate, component, low, high in (
        (center[0], direction[0], inset[0] + margin, inset[2] - margin),
        (center[1], direction[1], inset[1] + margin, inset[3] - margin),
    ):
        if component > 0.0:
            maximum_inset_length = min(
                maximum_inset_length,
                (high - coordinate) / component,
            )
        elif component < 0.0:
            maximum_inset_length = min(
                maximum_inset_length,
                (low - coordinate) / component,
            )
    if maximum_inset_length < 0.0:
        raise D408Error("glyph center is outside the inset")
    display_length = min(
        DISPLAY_FORCE_CAP_PX,
        uncapped_length,
        maximum_inset_length,
    )
    endpoint = (
        center[0] + direction[0] * display_length,
        center[1] + direction[1] * display_length,
    )
    if not (
        inset[0] + margin - 1.0e-9
        <= endpoint[0]
        <= inset[2] - margin + 1.0e-9
        and inset[1] + margin - 1.0e-9
        <= endpoint[1]
        <= inset[3] - margin + 1.0e-9
    ):
        raise D408Error("bounded glyph endpoint escaped the inset")
    return {
        "direction_xy": [direction[0], direction[1]],
        "display_capped": display_length + 1.0e-12 < uncapped_length,
        "display_force_cap_applied": (
            DISPLAY_FORCE_CAP_PX + 1.0e-12 < uncapped_length
            and DISPLAY_FORCE_CAP_PX <= maximum_inset_length + 1.0e-12
        ),
        "display_inset_clamp_applied": (
            maximum_inset_length + 1.0e-12
            < min(DISPLAY_FORCE_CAP_PX, uncapped_length)
        ),
        "display_length_px": display_length,
        "endpoint": [endpoint[0], endpoint[1]],
        "norm_abs_error_n": norm_error,
        "raw_vector_n": vector,
        "recomputed_norm_n": recomputed_norm,
        "stored_norm_n": stored_norm,
        "uncapped_length_px": uncapped_length,
    }


def _validate_trace_force_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    sample_count = 0
    text_bbox_count = 0
    maximum_norm_error = 0.0
    capped_sample_count = 0
    inset_clamped_sample_count = 0
    reference_inset = (0.0, 0.0, 894.0, 166.0)
    reference_center = (120.0, 84.0)
    text_probe = Image.new("RGB", (894, 166), (0, 0, 0))
    text_draw = ImageDraw.Draw(text_probe)
    for row_index, row in enumerate(rows, start=1):
        contact = row.get("contact")
        if not isinstance(contact, dict):
            raise D408Error(f"trace row {row_index} contact is missing")
        by_filter = contact.get("by_filter")
        if not isinstance(by_filter, dict) or set(by_filter) != set(
            FORCE_FILTER_LABELS
        ):
            raise D408Error(f"trace row {row_index} force-filter set mismatch")
        for label in FORCE_FILTER_LABELS:
            item = by_filter[label]
            if not isinstance(item, dict):
                raise D408Error(f"trace row {row_index} {label} is not an object")
            vector = item.get("force_w_n")
            stored_norm = item.get("force_norm_n")
            if (
                not isinstance(vector, list)
                or len(vector) != 3
                or any(type(value) not in (int, float) for value in vector)
                or type(stored_norm) not in (int, float)
            ):
                raise D408Error(
                    f"trace row {row_index} {label} force data is malformed"
                )
            glyph = _bounded_glyph(
                vector,
                float(stored_norm),
                inset=reference_inset,
                center=reference_center,
            )
            sample_count += 1
            maximum_norm_error = max(
                maximum_norm_error,
                float(glyph["norm_abs_error_n"]),
            )
            capped_sample_count += int(glyph["display_capped"])
            inset_clamped_sample_count += int(
                glyph["display_inset_clamp_applied"]
            )
            vector_values = glyph["raw_vector_n"]
            text_rows = (
                (18, 24, f"{label} final"),
                (
                    58,
                    17,
                    "raw Fx/Fy/Fz = "
                    f"[{vector_values[0]:.3f}, {vector_values[1]:.3f}, "
                    f"{vector_values[2]:.3f}] N",
                ),
                (
                    91,
                    18,
                    f"stored raw norm = {glyph['stored_norm_n']:.6f} N",
                ),
                (
                    124,
                    16,
                    "display_capped="
                    f"{str(glyph['display_capped']).lower()} "
                    f"length={glyph['display_length_px']:.2f}px / "
                    f"cap={DISPLAY_FORCE_CAP_PX:.0f}px",
                ),
            )
            for y, font_size, text_value in text_rows:
                bbox = text_draw.textbbox(
                    (250, y),
                    text_value,
                    font=_font(font_size),
                )
                if not (
                    8 <= bbox[0]
                    and 8 <= bbox[1]
                    and bbox[2] <= 886
                    and bbox[3] <= 158
                ):
                    raise D408Error(
                        f"trace row {row_index} {label} text escaped glyph inset"
                    )
                text_bbox_count += 1
    expected_count = len(rows) * len(FORCE_FILTER_LABELS)
    if sample_count != expected_count:
        raise D408Error("force sample traversal count mismatch")
    return {
        "display_capped_sample_count": capped_sample_count,
        "display_inset_clamped_sample_count": inset_clamped_sample_count,
        "force_filter_labels": list(FORCE_FILTER_LABELS),
        "force_sample_count": sample_count,
        "maximum_stored_vs_recomputed_norm_abs_error_n": maximum_norm_error,
        "norm_abs_tolerance_n": FORCE_NORM_ABS_TOLERANCE_N,
        "validated_text_bbox_count": text_bbox_count,
    }


@lru_cache(maxsize=None)
def _font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_PATH), size=size)


def _draw_chart(
    draw: Any,
    rectangle: tuple[int, int, int, int],
    title: str,
    unit: str,
    series: list[tuple[str, list[float], tuple[int, int, int]]],
) -> dict[str, Any]:
    left, top, right, bottom = rectangle
    draw.rounded_rectangle(
        rectangle,
        radius=12,
        fill=(21, 28, 38),
        outline=(88, 104, 126),
        width=2,
    )
    draw.text((left + 12, top + 7), title, font=_font(16), fill=(240, 244, 250))
    legend_width = max(1, (right - left - 32) // len(series))
    for index, (label, _, color) in enumerate(series):
        legend_x = left + 16 + index * legend_width
        draw.rectangle((legend_x, top + 33, legend_x + 12, top + 45), fill=color)
        draw.text((legend_x + 17, top + 29), label, font=_font(11), fill=color)
    plot = (left + 57, top + 54, right - 13, bottom - 30)
    values = [value for _, items, _ in series for value in items]
    if not values or not all(math.isfinite(value) for value in values):
        raise D408Error(f"chart contains non-finite values: {title}")
    low = min(values)
    high = max(values)
    if high <= low:
        high = low + 1.0
    length = len(series[0][1])
    if any(len(items) != length for _, items, _ in series):
        raise D408Error(f"chart series length mismatch: {title}")

    def point(index: int, value: float) -> tuple[float, float]:
        x = plot[0] + (plot[2] - plot[0]) * index / max(length - 1, 1)
        y = plot[3] - (plot[3] - plot[1]) * (value - low) / (high - low)
        return x, y

    y_ticks = [low, (low + high) / 2.0, high]
    for value in y_ticks:
        _, y = point(0, value)
        draw.line((plot[0], y, plot[2], y), fill=(52, 63, 79), width=1)
        draw.text(
            (left + 5, y - 7),
            f"{value:.3g}",
            font=_font(10),
            fill=(176, 188, 204),
        )
    x_ticks = [
        (0, "1"),
        ((length - 1) // 2, str((length + 1) // 2)),
        (length - 1, str(length)),
    ]
    for index, label in x_ticks:
        x, _ = point(index, low)
        draw.line((x, plot[1], x, plot[3]), fill=(45, 55, 69), width=1)
        draw.text(
            (x - 8, plot[3] + 5),
            label,
            font=_font(10),
            fill=(176, 188, 204),
        )
    draw.rectangle(plot, outline=(104, 118, 138), width=1)
    for _, items, color in series:
        draw.line(
            [point(index, value) for index, value in enumerate(items)],
            fill=color,
            width=2,
        )
    return {
        "high": high,
        "low": low,
        "rows": length,
        "unit": unit,
        "x_tick_rows": [1, (length + 1) // 2, length],
        "y_ticks": y_ticks,
    }


def _render_decision_sheet(
    leg: str,
    clean_spatial_path: Path,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    config = LEG_CONFIGS[leg]
    canvas = Image.new("RGB", (1920, 1080), (12, 17, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (32, 18),
        f"D408 leg {leg.upper()} — {config['label']} / D407 read-only replay",
        font=_font(34),
        fill=(248, 250, 253),
    )
    draw.text(
        (32, 62),
        "새 step=0 · Float64 원본 권위 · 화살표=표시 cap · A/B y축 독립(높이 말고 눈금 비교)",
        font=_font(18),
        fill=(255, 220, 102),
    )
    with Image.open(clean_spatial_path) as spatial:
        spatial.load()
        spatial_resized = spatial.resize((920, 739))
        canvas.paste(spatial_resized, (32, 103))
    draw.rectangle((32, 103, 951, 841), outline=(110, 130, 154), width=3)

    final = rows[-1]
    labels = ("link4", "link5", "gripper_link")
    glyph_reports: dict[str, Any] = {}
    colors = {
        "link4": (210, 112, 255),
        "link5": (90, 192, 255),
        "gripper_link": (255, 103, 92),
    }
    y_start = 112
    for index, label in enumerate(labels):
        item = final["contact"]["by_filter"][label]
        top = y_start + index * 190
        inset = (990, top, 1884, top + 166)
        draw.rounded_rectangle(
            inset,
            radius=12,
            fill=(24, 31, 42),
            outline=(92, 109, 132),
            width=2,
        )
        center = (1110.0, top + 84.0)
        glyph = _bounded_glyph(
            item["force_w_n"],
            float(item["force_norm_n"]),
            inset=tuple(float(value) for value in inset),
            center=center,
        )
        direction = glyph["direction_xy"]
        endpoint = tuple(glyph["endpoint"])
        if not (
            inset[0] + 8 <= endpoint[0] <= inset[2] - 8
            and inset[1] + 8 <= endpoint[1] <= inset[3] - 8
        ):
            raise D408Error(f"bounded glyph escaped inset: {label}")
        color = colors[label]
        draw.ellipse(
            (center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5),
            fill=color,
        )
        draw.line((center, endpoint), fill=color, width=7)
        draw.text(
            (1240, top + 18),
            f"{label} final",
            font=_font(24),
            fill=color,
        )
        vector = glyph["raw_vector_n"]
        draw.text(
            (1240, top + 58),
            f"raw Fx/Fy/Fz = [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}] N",
            font=_font(17),
            fill=(229, 235, 244),
        )
        draw.text(
            (1240, top + 91),
            f"stored raw norm = {glyph['stored_norm_n']:.6f} N",
            font=_font(18),
            fill=(229, 235, 244),
        )
        draw.text(
            (1240, top + 124),
            f"display_capped={str(glyph['display_capped']).lower()} "
            f"length={glyph['display_length_px']:.2f}px / cap={DISPLAY_FORCE_CAP_PX:.0f}px",
            font=_font(16),
            fill=(255, 220, 102),
        )
        label_bbox = draw.textbbox(
            (1240, top + 18),
            f"{label} final",
            font=_font(24),
        )
        if not (
            inset[0] + 8 <= label_bbox[0]
            and inset[1] + 8 <= label_bbox[1]
            and label_bbox[2] <= inset[2] - 8
            and label_bbox[3] <= inset[3] - 8
        ):
            raise D408Error(f"bounded glyph label escaped inset: {label}")
        glyph_reports[label] = {
            **glyph,
            "inset": list(inset),
            "label_bbox": list(label_bbox),
        }

    peak_rows: dict[str, dict[str, Any]] = {}
    peak_text = []
    for label in labels:
        peak_index, peak_row = max(
            enumerate(rows),
            key=lambda pair: float(
                pair[1]["contact"]["by_filter"][label]["force_norm_n"]
            ),
        )
        peak_value = float(
            peak_row["contact"]["by_filter"][label]["force_norm_n"]
        )
        peak_rows[label] = {"norm_n": peak_value, "row": peak_index + 1}
        peak_text.append(f"{label}: {peak_value:.6f}N @ row {peak_index + 1}")
    draw.rounded_rectangle(
        (990, 690, 1884, 842),
        radius=12,
        fill=(24, 31, 42),
        outline=(92, 109, 132),
        width=2,
    )
    draw.text((1010, 706), "500-row raw peak traversal", font=_font(22), fill=(240, 244, 250))
    for index, text in enumerate(peak_text):
        draw.text((1010, 746 + 29 * index), text, font=_font(17), fill=(214, 222, 234))

    q5_panel = _draw_chart(
        draw,
        (32, 865, 480, 1060),
        "q5 actual / target (rad)",
        "rad",
        [
            ("actual", [float(row["q5_actual_rad"]) for row in rows], (92, 203, 255)),
            ("target", [float(row["q5_target_rad"]) for row in rows], (255, 111, 99)),
        ],
    )
    force_panel = _draw_chart(
        draw,
        (500, 865, 950, 1060),
        "robot-body force norm (N)",
        "N",
        [
            (
                "link4",
                [
                    float(row["contact"]["by_filter"]["link4"]["force_norm_n"])
                    for row in rows
                ],
                colors["link4"],
            ),
            (
                "link5",
                [
                    float(row["contact"]["by_filter"]["link5"]["force_norm_n"])
                    for row in rows
                ],
                colors["link5"],
            ),
            (
                "grip",
                [
                    float(
                        row["contact"]["by_filter"]["gripper_link"]["force_norm_n"]
                    )
                    for row in rows
                ],
                colors["gripper_link"],
            ),
        ],
    )
    displacement_panel = _draw_chart(
        draw,
        (970, 865, 1420, 1060),
        "object displacement (mm)",
        "mm",
        [
            ("XYmm", [float(row["object_disp_xy_mm"]) for row in rows], (92, 220, 126)),
            ("zmm", [float(row["object_z_delta_mm"]) for row in rows], (95, 150, 255)),
        ],
    )
    tilt_panel = _draw_chart(
        draw,
        (1440, 865, 1884, 1060),
        "object tilt delta (deg)",
        "deg",
        [
            (
                "tilt",
                [
                    float(row["object_tilt_delta_from_reference_deg"])
                    for row in rows
                ],
                (255, 198, 79),
            ),
        ],
    )
    canvas.save(output_path, format="PNG", compress_level=9, optimize=False)
    _fsync_existing(output_path)
    return {
        "chart_reports": {
            "displacement": displacement_panel,
            "force": force_panel,
            "q5": q5_panel,
            "tilt": tilt_panel,
        },
        "dimensions": list(canvas.size),
        "glyph_reports": glyph_reports,
        "peak_rows": peak_rows,
        "sha256": _sha_path(output_path),
    }


def _create_leg_directory(path: Path) -> None:
    os.mkdir(path, 0o700)
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise D408Error(f"leg output is not a private directory: {path}")


def _render_leg(leg: str, prereg: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    import rerun as rr
    import rerun.blueprint as rrb
    from rerun.experimental import RrdReader

    if str(rr.__version__) != RERUN_VERSION:
        raise D408Error(f"Rerun SDK drift: {rr.__version__}")
    config = LEG_CONFIGS[leg]
    source_root = D407_ROOT / config["source_directory"]
    output_root = D408_ROOT / config["directory"]
    _create_leg_directory(output_root)

    source_trace = source_root / "d407_physics_trace.json"
    source_rrd = source_root / "d407_physx_contact_motion.rrd"
    source_rbl = source_root / "d407_physx_contact_motion.rbl"
    copied_trace = output_root / "d407_source_trace.json"
    copied_rrd = output_root / "d407_source_recording.rrd"
    copied_rbl = output_root / "d407_source_blueprint.rbl"
    presentation_rrd = output_root / "d408_presentation_recording.rrd"
    presentation_rbl = output_root / "d408_clean_spatial.rbl"
    full_screenshot = output_root / "d408_rerun_full_diagnostic.png"
    clean_spatial = output_root / "d408_clean_spatial.png"
    decision_sheet = output_root / "d408_decision_sheet.png"
    validation_path = output_root / "d408_rerun_validation.json"

    source_copies = {
        "rbl": _copy_exact_x(source_rbl, copied_rbl),
        "rrd": _copy_exact_x(source_rrd, copied_rrd),
        "trace": _copy_exact_x(source_trace, copied_trace),
    }
    rows = _load_trace(source_trace)
    force_contract = _validate_trace_force_contract(rows)

    reader = RrdReader(source_rrd)
    recordings = reader.recordings()
    blueprints = reader.blueprints()
    if len(recordings) != 1 or len(blueprints) != 1:
        raise D408Error(
            f"source store inventory mismatch for leg {leg}: "
            f"recordings={len(recordings)} blueprints={len(blueprints)}"
        )
    source_recording = recordings[0]
    if (
        source_recording.application_id != config["source_application_id"]
        or source_recording.recording_id != config["source_recording_id"]
    ):
        raise D408Error(f"source recording identity mismatch for leg {leg}")
    source_summary = reader.store(store=source_recording).summary()
    source_entities = _summary_entities(source_summary)
    if not set(DROP_FORCE_ENTITIES).issubset(source_entities):
        raise D408Error(f"force-display source entities missing for leg {leg}")
    source_semantics = _rrd_semantic_inventory(
        reader,
        source_recording,
        drop_entities=set(DROP_FORCE_ENTITIES),
        drop_component="Points3D:labels",
    )
    stream = reader.stream(store=source_recording).drop(
        content=list(DROP_FORCE_ENTITIES)
    )
    stream = stream.drop(components="Points3D:labels")
    stream.write_rrd(
        presentation_rrd,
        application_id=config["application_id"],
        recording_id=config["recording_id"],
    )
    projected = RrdReader(presentation_rrd)
    if len(projected.recordings()) != 1 or projected.blueprints():
        raise D408Error(f"recording-only projection failed for leg {leg}")
    output_recording = projected.recordings()[0]
    if (
        output_recording.application_id != config["application_id"]
        or output_recording.recording_id != config["recording_id"]
    ):
        raise D408Error(f"output recording identity mismatch for leg {leg}")
    output_summary = projected.store(store=output_recording).summary()
    if "Points3D:labels" in output_summary:
        raise D408Error(f"contact-point labels remain in presentation leg {leg}")
    output_entities = _summary_entities(output_summary)
    removed = sorted(source_entities - output_entities)
    added = sorted(output_entities - source_entities)
    if removed != sorted(DROP_FORCE_ENTITIES) or added:
        raise D408Error(
            f"projection entity delta mismatch for leg {leg}: "
            f"removed={removed} added={added}"
        )
    output_semantics = _rrd_semantic_inventory(
        projected,
        output_recording,
        drop_entities=set(),
        drop_component=None,
    )
    if (
        source_semantics["system_entity_paths"]
        != output_semantics["system_entity_paths"]
    ):
        raise D408Error(f"RRD system entity set drift for leg {leg}")
    if source_semantics["canonical"] != output_semantics["canonical"]:
        raise D408Error(f"retained RRD semantic data drift for leg {leg}")
    semantic_equality = {
        "component_names": source_semantics["component_names"],
        "digest_sha256": source_semantics["digest_sha256"],
        "entity_paths": source_semantics["entity_paths"],
        "output_digest_sha256": output_semantics["digest_sha256"],
        "output_store_identity": {
            "application_id": output_recording.application_id,
            "recording_id": output_recording.recording_id,
        },
        "row_count": source_semantics["row_count"],
        "source_store_identity": {
            "application_id": source_recording.application_id,
            "recording_id": source_recording.recording_id,
        },
        "system_entity_paths": source_semantics["system_entity_paths"],
        "timeline_names": source_semantics["timeline_names"],
        "value_cell_count": source_semantics["value_cell_count"],
    }
    if (
        source_semantics["digest_sha256"]
        != output_semantics["digest_sha256"]
    ):
        raise D408Error(f"retained RRD semantic digest drift for leg {leg}")

    eye = rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=[0.49, -0.32, 0.28],
        look_target=[0.285, 0.0, 0.055],
        eye_up=[0.0, 0.0, 1.0],
    )
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            contents=["/geometry/**", "/transforms/**", "/contacts/**"],
            name=f"D408 clean spatial replay leg {leg.upper()} — force vectors omitted",
            eye_controls=eye,
            spatial_information=rrb.SpatialInformation(
                target_frame="world",
                show_axes=True,
                show_bounding_box=False,
            ),
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )
    blueprint.save(config["application_id"], presentation_rbl)
    _fsync_existing(presentation_rrd)
    _fsync_existing(presentation_rbl)

    verify_command = [
        str(RERUN_CLI),
        "rrd",
        "verify",
        "--check-footers",
        "true",
        str(presentation_rrd),
        str(presentation_rbl),
    ]
    verify = subprocess.run(
        verify_command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    if verify.returncode != 0 or "verified without error" not in verify.stdout:
        raise D408Error(f"Rerun verify failed for leg {leg}: {verify.stdout}")

    screenshot_command = [
        str(RERUN_CLI),
        "--headless",
        "--hide-welcome-screen",
        "--window-size",
        SCREENSHOT_LOGICAL_SIZE,
        "--screenshot-to",
        str(full_screenshot),
        str(presentation_rrd),
        str(presentation_rbl),
    ]
    environment = os.environ.copy()
    environment["VK_ICD_FILENAMES"] = str(LVP_ICD)
    environment["WGPU_BACKEND"] = "vulkan"
    environment["WGPU_POWER_PREF"] = "low"
    screenshot = subprocess.run(
        screenshot_command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
        shell=False,
    )
    if screenshot.returncode != 0:
        raise D408Error(f"Rerun screenshot failed for leg {leg}: {screenshot.stdout}")
    if "device_type: Cpu" not in screenshot.stdout or "llvmpipe" not in screenshot.stdout:
        raise D408Error(f"software renderer identity missing for leg {leg}")
    if not full_screenshot.is_file() or full_screenshot.is_symlink():
        raise D408Error(f"Rerun screenshot missing for leg {leg}")
    _fsync_existing(full_screenshot)

    with Image.open(full_screenshot) as image:
        image.load()
        if image.size != SCREENSHOT_PHYSICAL_SIZE:
            raise D408Error(
                f"unexpected full screenshot size for leg {leg}: {image.size}"
            )
        full_notification = _notification_overlay_detected(image)
        clean = image.crop(CLEAN_CROP_BOX)
        if clean.size != CLEAN_CROP_SIZE:
            raise D408Error(f"clean crop size mismatch for leg {leg}")
        clean.save(clean_spatial, format="PNG", compress_level=9, optimize=False)
    _fsync_existing(clean_spatial)
    with Image.open(clean_spatial) as clean_image:
        clean_image.load()
        clean_notification = _notification_overlay_detected(clean_image)
        witnesses = _color_witnesses(clean_image)
        if clean_notification:
            raise D408Error(f"notification overlay remains in clean crop for leg {leg}")
        if min(witnesses.values()) < 100:
            raise D408Error(f"jaw/cylinder color witness missing for leg {leg}: {witnesses}")

    sheet_report = _render_decision_sheet(leg, clean_spatial, rows, decision_sheet)
    validation = {
        "artifact": "D408_RERUN_PRESENTATION_VALIDATION_V1",
        "checks": {
            "clean_crop_dimensions_exact": True,
            "clean_crop_notification_detector_false": not clean_notification,
            "corrected_blueprint_verified": True,
            "force_display_entities_removed_exact": removed
            == sorted(DROP_FORCE_ENTITIES),
            "hardware_gpu_job_count_zero": True,
            "historical_trace_rows_exact_500": len(rows) == 500,
            "all_force_vectors_and_norms_validated": force_contract[
                "force_sample_count"
            ]
            == 2000,
            "validated_force_text_bbox_count_exact": force_contract[
                "validated_text_bbox_count"
            ]
            == 8000,
            "llvmpipe_cpu_renderer": True,
            "output_blueprint_store_count_zero": len(projected.blueprints()) == 0,
            "output_point_labels_removed": "Points3D:labels" not in output_summary,
            "output_recording_store_count_one": len(projected.recordings()) == 1,
            "presentation_footer_verify": verify.returncode == 0,
            "retained_rrd_semantics_exact": source_semantics["canonical"]
            == output_semantics["canonical"],
            "system_metadata_semantics_exact": (
                source_semantics["system_entity_paths"]
                == output_semantics["system_entity_paths"]
            ),
            "source_blueprint_store_count_one": len(blueprints) == 1,
            "source_recording_store_count_one": len(recordings) == 1,
            "visual_color_witnesses_present": min(witnesses.values()) >= 100,
        },
        "clean_crop": {
            "box": list(CLEAN_CROP_BOX),
            "dimensions": list(CLEAN_CROP_SIZE),
            "notification_detected": clean_notification,
            "sha256": _sha_path(clean_spatial),
            "witnesses": witnesses,
        },
        "decision_sheet": sheet_report,
        "force_contract": force_contract,
        "full_screenshot": {
            "diagnostic_only": True,
            "dimensions": list(SCREENSHOT_PHYSICAL_SIZE),
            "notification_detected": full_notification,
            "sha256": _sha_path(full_screenshot),
        },
        "leg": leg,
        "new_controlled_physics_steps": 0,
        "presentation_rbl_sha256": _sha_path(presentation_rbl),
        "presentation_rrd_sha256": _sha_path(presentation_rrd),
        "projection": {
            "added_entities": added,
            "output_blueprint_stores": len(projected.blueprints()),
            "output_recording_stores": len(projected.recordings()),
            "removed_entities": removed,
            "source_blueprint_stores": len(blueprints),
            "source_recording_stores": len(recordings),
            "retained_semantic_equality": semantic_equality,
        },
        "rerun_cli_output": screenshot.stdout,
        "rerun_version": RERUN_VERSION,
        "source_copies": source_copies,
        "source_science_authority": "immutable D407 JSON/RRD; presentation values are not hashed back",
        "verify_command": verify_command,
        "verify_output": verify.stdout,
    }
    validation["pass"] = all(validation["checks"].values())
    if validation["pass"] is not True:
        raise D408Error(f"leg {leg} presentation validation failed")
    _write_json_x(validation_path, validation)
    return {
        "clean_spatial_path": _rel(clean_spatial),
        "decision_sheet_path": _rel(decision_sheet),
        "full_screenshot_path": _rel(full_screenshot),
        "historical_trace_rows": len(rows),
        "leg": leg,
        "validation_path": _rel(validation_path),
        "validation_sha256": _sha_path(validation_path),
    }


def _build_ab_sheet() -> dict[str, Any]:
    from PIL import Image

    a_path = D408_ROOT / LEG_CONFIGS["a"]["directory"] / "d408_decision_sheet.png"
    b_path = D408_ROOT / LEG_CONFIGS["b"]["directory"] / "d408_decision_sheet.png"
    with Image.open(a_path) as image_a, Image.open(b_path) as image_b:
        image_a.load()
        image_b.load()
        if image_a.size != (1920, 1080) or image_b.size != (1920, 1080):
            raise D408Error("decision sheet dimensions are not exact")
        canvas = Image.new("RGB", (3840, 1080), (12, 17, 24))
        canvas.paste(image_a, (0, 0))
        canvas.paste(image_b, (1920, 0))
        canvas.save(AB_SHEET_PATH, format="PNG", compress_level=9, optimize=False)
    _fsync_existing(AB_SHEET_PATH)
    return {
        "dimensions": [3840, 1080],
        "sha256": _sha_path(AB_SHEET_PATH),
    }


def _image_report(
    root_fd: int,
    path: Path,
    manual_role: str,
) -> dict[str, Any]:
    from PIL import Image

    relative_path = str(path.relative_to(D408_ROOT))
    raw, metadata = _secure_read_relative(
        root_fd,
        relative_path,
        MAX_SCREENSHOT_BYTES,
    )
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        dimensions = list(image.size)
    return {
        "bytes": metadata.st_size,
        "dimensions": dimensions,
        "manual_role": manual_role,
        "path": _rel(path),
        "root_relative_path": relative_path,
        "sha256": _sha_bytes(raw),
    }


def _write_screenshot_manifest(
    root_fd: int,
    leg_reports: dict[str, dict[str, Any]],
    ab_report: dict[str, Any],
) -> dict[str, Any]:
    images = []
    for leg in ("a", "b"):
        leg_root = D408_ROOT / LEG_CONFIGS[leg]["directory"]
        images.extend(
            [
                _image_report(
                    root_fd,
                    leg_root / "d408_clean_spatial.png",
                    f"leg_{leg}_jaw_and_cylinder_clean_spatial",
                ),
                _image_report(
                    root_fd,
                    leg_root / "d408_decision_sheet.png",
                    f"leg_{leg}_raw_force_timeseries_bounded_glyph",
                ),
            ]
        )
    images.append(_image_report(root_fd, AB_SHEET_PATH, "ab_comparison"))
    manifest = {
        "artifact": "D408_SCREENSHOT_MANIFEST_V1",
        "ab_report": ab_report,
        "images": images,
        "leg_reports": leg_reports,
        "manual_target_count": len(images),
        "new_controlled_physics_steps": 0,
    }
    _write_json_x(SCREENSHOT_MANIFEST_PATH, manifest)
    return manifest


def _verify_screenshot_manifest(
    root_fd: int,
    expected_manifest_sha256: str,
    checkpoint: str,
) -> dict[str, Any]:
    from PIL import Image

    manifest_raw, manifest_metadata = _secure_read_relative(
        root_fd,
        SCREENSHOT_MANIFEST_PATH.name,
        MAX_JSON_BYTES,
    )
    manifest_sha = _sha_bytes(manifest_raw)
    if manifest_sha != expected_manifest_sha256:
        raise D408Error(f"screenshot manifest drift at {checkpoint}")
    manifest = _strict_json_bytes(manifest_raw)
    if not isinstance(manifest, dict):
        raise D408Error("screenshot manifest is not an object")
    _expect_keys(
        manifest,
        {
            "ab_report",
            "artifact",
            "images",
            "leg_reports",
            "manual_target_count",
            "new_controlled_physics_steps",
        },
        "screenshot manifest",
    )
    if (
        manifest["artifact"] != "D408_SCREENSHOT_MANIFEST_V1"
        or manifest["manual_target_count"] != 5
        or manifest["new_controlled_physics_steps"] != 0
    ):
        raise D408Error("screenshot manifest invariant mismatch")
    images = manifest["images"]
    if not isinstance(images, list) or len(images) != 5:
        raise D408Error("screenshot manifest image count mismatch")
    observed_paths: list[str] = []
    verified_images: list[dict[str, Any]] = []
    for item in images:
        if not isinstance(item, dict):
            raise D408Error("screenshot manifest image row is not an object")
        _expect_keys(
            item,
            {
                "bytes",
                "dimensions",
                "manual_role",
                "path",
                "root_relative_path",
                "sha256",
            },
            "screenshot manifest image row",
        )
        relative_path = item["root_relative_path"]
        if not isinstance(relative_path, str):
            raise D408Error("screenshot relative path is not a string")
        expected_dimensions = EXPECTED_MANUAL_IMAGE_LAYOUT.get(relative_path)
        if expected_dimensions is None:
            raise D408Error(f"unexpected screenshot path: {relative_path}")
        expected_project_path = _rel(D408_ROOT / relative_path)
        if item["path"] != expected_project_path:
            raise D408Error("screenshot project/root-relative path mismatch")
        raw, metadata = _secure_read_relative(
            root_fd,
            relative_path,
            MAX_SCREENSHOT_BYTES,
        )
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            dimensions = list(image.size)
            if image.format != "PNG":
                raise D408Error(f"manual image is not PNG: {relative_path}")
        if (
            item["bytes"] != metadata.st_size
            or item["dimensions"] != dimensions
            or dimensions != expected_dimensions
            or item["sha256"] != _sha_bytes(raw)
        ):
            raise D408Error(f"manual image integrity drift: {relative_path}")
        observed_paths.append(relative_path)
        verified_images.append(
            {
                "bytes": metadata.st_size,
                "dimensions": dimensions,
                "root_relative_path": relative_path,
                "sha256": _sha_bytes(raw),
            }
        )
    if sorted(observed_paths) != sorted(EXPECTED_MANUAL_IMAGE_LAYOUT):
        raise D408Error("manual screenshot path set mismatch")
    return {
        "checkpoint": checkpoint,
        "image_count": len(verified_images),
        "images_sha256": _sha_bytes(_canonical_bytes(verified_images)),
        "manifest_bytes": manifest_metadata.st_size,
        "manifest_sha256": manifest_sha,
        "monotonic_ns": time.monotonic_ns(),
        "verified_images": verified_images,
    }


def _validate_manual_input(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise D408Error("manual stdin JSON must be an object")
    _expect_keys(value, {"notes", "required_fields"}, "manual stdin")
    fields = value["required_fields"]
    notes = value["notes"]
    if not isinstance(fields, dict):
        raise D408Error("required_fields must be an object")
    _expect_keys(fields, set(REQUIRED_BOOLEAN_FIELDS), "manual required_fields")
    for key in REQUIRED_BOOLEAN_FIELDS:
        if type(fields[key]) is not bool:
            raise D408Error(f"manual field is not boolean: {key}")
    if not isinstance(notes, str) or len(notes.encode("utf-8")) > 4096:
        raise D408Error("manual notes exceed 4096 UTF-8 bytes")
    return {"notes": notes, "required_fields": fields}


def _read_manual_stdin(deadline_monotonic_ns: int) -> dict[str, Any]:
    remaining = (deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000
    if remaining <= 0:
        raise TimeoutError("manual inspection deadline expired before input")
    ready, _, _ = select.select([sys.stdin], [], [], remaining)
    if not ready:
        raise TimeoutError("manual inspection stdin timeout")
    line = sys.stdin.buffer.readline(MAX_MANUAL_BYTES + 1)
    if len(line) > MAX_MANUAL_BYTES:
        raise D408Error("manual stdin line exceeds size limit")
    if not line.endswith(b"\n"):
        raise D408Error("manual stdin must contain one complete JSON line")
    value = _strict_json_bytes(line)
    return _validate_manual_input(value)


def _secure_read_manual_once(
    root_fd: int,
    expected_sha: str,
    expected_size: int,
) -> dict[str, Any]:
    raw, before = _secure_read_relative(
        root_fd,
        MANUAL_PATH.name,
        MAX_MANUAL_BYTES,
    )
    if len(raw) != expected_size or _sha_bytes(raw) != expected_sha:
        raise D408Error("manual final does not match writer ack")
    value = _strict_json_bytes(raw)
    if not isinstance(value, dict):
        raise D408Error("manual final is not an object")
    _expect_keys(
        value,
        {
            "artifact",
            "bindings",
            "deadline",
            "notes",
            "pass",
            "received",
            "required_fields",
            "source_science",
        },
        "manual final",
    )
    if value["artifact"] != "D408_MANUAL_VISUAL_INSPECTION_V1":
        raise D408Error("manual final artifact mismatch")
    if value["received"] is not True:
        raise D408Error("manual final received flag is not true")
    fields = value["required_fields"]
    if not isinstance(fields, dict):
        raise D408Error("manual final required_fields is not an object")
    _expect_keys(fields, set(REQUIRED_BOOLEAN_FIELDS), "manual final fields")
    if any(type(fields[key]) is not bool for key in REQUIRED_BOOLEAN_FIELDS):
        raise D408Error("manual final contains non-boolean field")
    if type(value["pass"]) is not bool or value["pass"] is not all(fields.values()):
        raise D408Error("manual final pass is not computed from booleans")
    science = value["source_science"]
    expected_science = {
        "d407_final_verdict": D407_FINAL_VERDICT,
        "d407_retroactive_pass": False,
        "g0a_pass": False,
        "new_controlled_physics_steps": 0,
        "scientific_null_claims": SCIENTIFIC_NULL_CLAIMS,
        "scientific_verdict": None,
    }
    if science != expected_science:
        raise D408Error("manual final attempts to change frozen science")
    if before.st_size != expected_size:
        raise D408Error("manual final metadata size differs from writer ack")
    return value


def _spawn_writer(
    root_fd: int,
    root_dev: int,
    root_ino: int,
    phase: PhaseLog,
    controller_started_row: dict[str, Any],
    prereg: dict[str, Any],
    approved_tuple_sha256: str,
    controller_sha: str,
    writer_sha: str,
    prearm_hard_deadline_ns: int,
) -> tuple[subprocess.Popen[bytes], socket.socket, bytes, int]:
    parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    writer_log_fd = os.open(
        WRITER_LOG_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    controller_pid = os.getpid()
    controller_start = _proc_start_ticks(controller_pid)
    command = [
        str(ISAACLAB_PYTHON),
        "-B",
        str(WRITER_PATH),
        "--stage",
        "writer",
        "--socket-fd",
        str(child_socket.fileno()),
        "--root",
        str(D408_ROOT),
        "--root-fd",
        str(root_fd),
        "--root-dev",
        str(root_dev),
        "--root-ino",
        str(root_ino),
        "--controller-pid",
        str(controller_pid),
        "--controller-start-ticks",
        str(controller_start),
        "--controller-sha256",
        controller_sha,
        "--writer-sha256",
        writer_sha,
        "--approved-tuple-sha256",
        approved_tuple_sha256,
        "--phase-dev",
        str(phase.dev),
        "--phase-ino",
        str(phase.ino),
        "--input-manifest-sha256",
        prereg["d407_source_manifest_sha256"],
        "--manual-basename",
        MANUAL_PATH.name,
        "--prearm-hard-deadline-monotonic-ns",
        str(prearm_hard_deadline_ns),
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=writer_log_fd,
        stderr=subprocess.STDOUT,
        pass_fds=(child_socket.fileno(), root_fd),
        start_new_session=True,
        shell=False,
    )
    os.close(writer_log_fd)
    child_socket.close()
    try:
        nonce = secrets.token_bytes(32)
        bindings = {
            "approved_tuple_sha256": approved_tuple_sha256,
            "controller_pid": controller_pid,
            "controller_sha256": controller_sha,
            "controller_start_ticks": controller_start,
            "input_manifest_sha256": prereg["d407_source_manifest_sha256"],
            "manual_basename": MANUAL_PATH.name,
            "phase_dev": phase.dev,
            "phase_ino": phase.ino,
            "prearm_hard_deadline_monotonic_ns": prearm_hard_deadline_ns,
            "preregistration_sha256": EXPECTED_PREREG_SHA256,
            "root_dev": root_dev,
            "root_ino": root_ino,
            "writer_sha256": writer_sha,
        }
        parent_socket.settimeout(15.0)
        _send_json_line(
            parent_socket,
            {"bindings": bindings, "nonce_hex": nonce.hex(), "op": "arm"},
        )
        ready = _recv_json_line(parent_socket)
        _expect_keys(ready, {"body", "hmac_sha256"}, "writer READY")
        body = ready["body"]
        if not isinstance(body, dict):
            raise D408Error("writer READY body is not an object")
        supplied_hmac = ready["hmac_sha256"]
        if not isinstance(supplied_hmac, str) or not hmac.compare_digest(
            supplied_hmac,
            _hmac_hex(nonce, body),
        ):
            raise D408Error("writer READY HMAC mismatch")
        _expect_keys(
            body,
            {
                "nonce_sha256",
                "op",
                "phase_row_sha256",
                "phase_sequence",
                "writer_pid",
                "writer_sha256",
                "writer_start_ticks",
            },
            "writer READY body",
        )
        if body["op"] != "ready" or body["writer_pid"] != process.pid:
            raise D408Error("writer READY identity mismatch")
        if body["writer_sha256"] != writer_sha:
            raise D408Error("writer READY SHA mismatch")
        if body["nonce_sha256"] != _sha_bytes(nonce):
            raise D408Error("writer READY nonce mismatch")
        if (
            body["phase_sequence"] != controller_started_row["sequence"]
            or body["phase_row_sha256"] != controller_started_row["row_sha256"]
        ):
            raise D408Error("writer READY phase binding mismatch")
        if _proc_start_ticks(process.pid) != body["writer_start_ticks"]:
            raise D408Error("writer PID/start-time mismatch")
        return process, parent_socket, nonce, int(body["writer_start_ticks"])
    except BaseException:
        parent_socket.close()
        _terminate_writer(process)
        raise


def _terminate_writer(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)


def _ping_writer(
    process: subprocess.Popen[bytes],
    channel: socket.socket,
    nonce: bytes,
    phase_row: dict[str, Any],
    writer_start_ticks: int,
) -> None:
    if process.poll() is not None:
        raise D408Error(
            f"writer exited before {phase_row['event']}: {process.returncode}"
        )
    if _proc_start_ticks(process.pid) != writer_start_ticks:
        raise D408Error("writer PID/start-time drift before authenticated ping")
    body = {
        "op": "ping",
        "phase_event": phase_row["event"],
        "phase_row_sha256": phase_row["row_sha256"],
        "phase_sequence": phase_row["sequence"],
    }
    channel.settimeout(15.0)
    _send_json_line(
        channel,
        {"body": body, "hmac_sha256": _hmac_hex(nonce, body)},
    )
    envelope = _recv_json_line(channel)
    _expect_keys(envelope, {"body", "hmac_sha256"}, "writer pong")
    pong = envelope["body"]
    if not isinstance(pong, dict):
        raise D408Error("writer pong body is not an object")
    supplied_hmac = envelope["hmac_sha256"]
    if not isinstance(supplied_hmac, str) or not hmac.compare_digest(
        supplied_hmac,
        _hmac_hex(nonce, pong),
    ):
        raise D408Error("writer pong HMAC mismatch")
    _expect_keys(
        pong,
        {
            "op",
            "phase_event",
            "phase_row_sha256",
            "phase_sequence",
            "writer_pid",
            "writer_start_ticks",
        },
        "writer pong body",
    )
    expected = {
        **body,
        "op": "pong",
        "writer_pid": process.pid,
        "writer_start_ticks": writer_start_ticks,
    }
    if pong != expected:
        raise D408Error("writer pong binding mismatch")


def _publish_via_writer(
    root_fd: int,
    process: subprocess.Popen[bytes],
    channel: socket.socket,
    nonce: bytes,
    phase: PhaseLog,
    manual_prompt_started_ns: int,
    manual_deadline_ns: int,
    prompt_row: dict[str, Any],
    screenshot_sha: str,
    expected_screenshot_images_sha256: str,
    manual_input: dict[str, Any],
    writer_start_ticks: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    writer_deadline = manual_deadline_ns - WRITER_DEADLINE_LEAD_NS
    body = {
        "manual_deadline_monotonic_ns": manual_deadline_ns,
        "manual_input": manual_input,
        "manual_prompt_started_monotonic_ns": manual_prompt_started_ns,
        "op": "publish",
        "phase_row_sha256": prompt_row["row_sha256"],
        "phase_sequence": prompt_row["sequence"],
        "screenshot_manifest_sha256": screenshot_sha,
        "writer_deadline_monotonic_ns": writer_deadline,
    }
    channel.settimeout(
        max(0.001, (manual_deadline_ns - time.monotonic_ns()) / 1e9)
    )
    _send_json_line(
        channel,
        {"body": body, "hmac_sha256": _hmac_hex(nonce, body)},
    )
    envelope = _recv_json_line(channel)
    _expect_keys(envelope, {"body", "hmac_sha256"}, "writer publication ack")
    ack = envelope["body"]
    if not isinstance(ack, dict):
        raise D408Error("writer publication ack body is not an object")
    supplied_hmac = envelope["hmac_sha256"]
    if not isinstance(supplied_hmac, str) or not hmac.compare_digest(
        supplied_hmac,
        _hmac_hex(nonce, ack),
    ):
        raise D408Error("writer publication ack HMAC mismatch")
    _expect_keys(
        ack,
        {
            "manual_pass",
            "manual_sha256",
            "manual_size",
            "op",
            "publication_fsync_completed_monotonic_ns",
            "published_before_writer_deadline",
            "screenshot_images_sha256",
            "writer_pid",
            "writer_start_ticks",
        },
        "writer publication ack body",
    )
    if ack["op"] != "published_fsynced" or ack["writer_pid"] != process.pid:
        raise D408Error("writer publication ack identity mismatch")
    if ack["writer_start_ticks"] != writer_start_ticks:
        raise D408Error("writer publication ack start-time mismatch")
    if type(ack["manual_pass"]) is not bool:
        raise D408Error("writer publication ack pass is not boolean")
    if type(ack["manual_size"]) is not int or not (
        0 < ack["manual_size"] <= MAX_MANUAL_BYTES
    ):
        raise D408Error("writer publication ack size is unsafe")
    if (
        ack["screenshot_images_sha256"]
        != expected_screenshot_images_sha256
    ):
        raise D408Error("writer screenshot-byte verification digest mismatch")
    manual = _secure_read_manual_once(
        root_fd,
        str(ack["manual_sha256"]),
        int(ack["manual_size"]),
    )
    bindings = manual["bindings"]
    if not isinstance(bindings, dict):
        raise D408Error("manual final bindings are not an object")
    expected_bindings = {
        "approved_tuple_sha256": _sha_path(TUPLE_PATH),
        "controller_pid": os.getpid(),
        "controller_sha256": _sha_path(CONTROLLER_PATH),
        "controller_start_ticks": _proc_start_ticks(os.getpid()),
        "d407_source_manifest_sha256": _strict_json_path(PREREG_PATH)[
            "d407_source_manifest_sha256"
        ],
        "nonce_sha256": _sha_bytes(nonce),
        "phase_dev": phase.dev,
        "phase_ino": phase.ino,
        "phase_row_sha256": prompt_row["row_sha256"],
        "phase_sequence": prompt_row["sequence"],
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "root_dev": phase.root_dev,
        "root_ino": phase.root_ino,
        "screenshot_manifest_sha256": screenshot_sha,
        "writer_pid": process.pid,
        "writer_sha256": _sha_path(WRITER_PATH),
        "writer_start_ticks": ack["writer_start_ticks"],
    }
    if bindings != expected_bindings:
        raise D408Error("manual final binding set is not exact")
    deadline = manual["deadline"]
    if deadline != {
        "manual_deadline_monotonic_ns": manual_deadline_ns,
        "manual_prompt_started_monotonic_ns": manual_prompt_started_ns,
        "writer_deadline_monotonic_ns": writer_deadline,
    }:
        raise D408Error("manual final deadline binding is not exact")
    if ack["manual_pass"] is not manual["pass"]:
        raise D408Error("manual final pass differs from writer ack")
    completed_ns = ack["publication_fsync_completed_monotonic_ns"]
    if type(completed_ns) is not int:
        raise D408Error("writer publication completion time is not an integer")
    expected_on_time = completed_ns < writer_deadline
    if (
        type(ack["published_before_writer_deadline"]) is not bool
        or ack["published_before_writer_deadline"] is not expected_on_time
    ):
        raise D408Error("writer deadline outcome is inconsistent")
    if not expected_on_time:
        raise TimeoutError("manual file publication completed after writer deadline")
    process.wait(timeout=5.0)
    if process.returncode != 0:
        raise D408Error(f"writer exited nonzero after publication: {process.returncode}")
    channel.close()
    receipt = {
        "artifact": "D408_MANUAL_WRITER_RECEIPT_V1",
        "ack": ack,
        "manual_pass": manual["pass"],
        "manual_sha256": ack["manual_sha256"],
        "manual_size": ack["manual_size"],
        "publication_fsync_completed_monotonic_ns": completed_ns,
        "published_before_writer_deadline": True,
        "received": True,
        "screenshot_images_sha256": ack["screenshot_images_sha256"],
    }
    _write_json_x(MANUAL_RECEIPT_PATH, receipt)
    return manual, receipt


def _runtime_counters() -> dict[str, int]:
    return {
        "contact_queries": 0,
        "cylinder_spawns": 0,
        "d407_worker_or_controller_spawns": 0,
        "hardware_gpu_jobs": 0,
        "historical_source_rrds_read": 2,
        "historical_trace_rows_read": 1000,
        "isaac_imports": 0,
        "isaac_launches": 0,
        "kit_launches": 0,
        "new_controlled_physics_steps": 0,
        "physx_launches": 0,
        "q5_state_samples": 0,
        "q5_target_writes": 0,
        "software_rerun_viewers": 2,
        "usd_or_asset_writes": 0,
    }


def _run_controller(approved_tuple_sha256: str) -> int:
    prereg, _, _, controller_sha, writer_sha = _validate_static_authority(
        approved_tuple_sha256
    )
    root_fd, root_dev, root_ino = _open_bound_root()
    overlay_checkpoints: list[dict[str, Any]] = []
    try:
        overlay_checkpoints.append(
            _validate_repository_overlay(
                prereg,
                root_fd,
                root_dev,
                root_ino,
                "admission",
                admission_exact=True,
            )
        )
        _runtime_absence_gate()
        stack = _verify_software_stack(prereg)
        checkpoints: list[dict[str, Any]] = []
        screenshot_checkpoints: list[dict[str, Any]] = []
        _validate_manifest(prereg, "admission", checkpoints)
    except BaseException:
        os.close(root_fd)
        raise
    phase: PhaseLog | None = None
    writer_process: subprocess.Popen[bytes] | None = None
    writer_channel: socket.socket | None = None
    terminal_publish_attempted = False
    try:
        phase = PhaseLog(root_fd)
        prearm_hard_deadline_ns = (
            time.monotonic_ns() + PREARM_HARD_TIMEOUT_NS
        )
        controller_started_row = phase.append(
            "controller_started",
            {
                "approved_tuple_sha256": approved_tuple_sha256,
                "controller_pid": os.getpid(),
                "controller_sha256": controller_sha,
                "controller_start_ticks": _proc_start_ticks(os.getpid()),
                "d407_source_manifest_sha256": prereg[
                    "d407_source_manifest_sha256"
                ],
                "prearm_hard_deadline_monotonic_ns": (
                    prearm_hard_deadline_ns
                ),
                "preregistration_sha256": EXPECTED_PREREG_SHA256,
                "root_dev": root_dev,
                "root_ino": root_ino,
                "writer_sha256": writer_sha,
            },
        )
        writer_process, writer_channel, nonce, writer_start = _spawn_writer(
            root_fd,
            root_dev,
            root_ino,
            phase,
            controller_started_row,
            prereg,
            approved_tuple_sha256,
            controller_sha,
            writer_sha,
            prearm_hard_deadline_ns,
        )
        writer_armed_row = phase.append(
            "writer_armed",
            {
                "nonce_sha256": _sha_bytes(nonce),
                "prearm_hard_deadline_monotonic_ns": (
                    prearm_hard_deadline_ns
                ),
                "writer_pid": writer_process.pid,
                "writer_sha256": writer_sha,
                "writer_start_ticks": writer_start,
            },
        )
        prerequisites = {
            "artifact": "D408_RUNTIME_PREREQUISITES_V1",
            "approved_tuple_sha256": approved_tuple_sha256,
            "counters": _runtime_counters(),
            "d407_source_manifest_sha256": prereg["d407_source_manifest_sha256"],
            "prearm_hard_deadline_monotonic_ns": prearm_hard_deadline_ns,
            "root_dev": root_dev,
            "root_ino": root_ino,
            "software_stack": stack,
            "writer_armed_before_replay": True,
            "writer_armed_phase_row_sha256": writer_armed_row["row_sha256"],
        }
        _write_json_x(RUNTIME_PREREQUISITES_PATH, prerequisites)

        phase.append("leg_a_replay_start", {"new_controlled_physics_steps": 0})
        leg_a = _render_leg("a", prereg)
        leg_a_complete_row = phase.append("leg_a_replay_complete", leg_a)
        _validate_manifest(prereg, "after_leg_a_capture", checkpoints)
        overlay_checkpoints.append(
            _validate_repository_overlay(
                prereg,
                root_fd,
                root_dev,
                root_ino,
                "after_leg_a_capture",
                admission_exact=False,
            )
        )
        _ping_writer(
            writer_process,
            writer_channel,
            nonce,
            leg_a_complete_row,
            writer_start,
        )

        phase.append("leg_b_replay_start", {"new_controlled_physics_steps": 0})
        leg_b = _render_leg("b", prereg)
        leg_b_complete_row = phase.append("leg_b_replay_complete", leg_b)
        _validate_manifest(prereg, "after_leg_b_capture", checkpoints)
        overlay_checkpoints.append(
            _validate_repository_overlay(
                prereg,
                root_fd,
                root_dev,
                root_ino,
                "after_leg_b_capture",
                admission_exact=False,
            )
        )
        _ping_writer(
            writer_process,
            writer_channel,
            nonce,
            leg_b_complete_row,
            writer_start,
        )

        ab_report = _build_ab_sheet()
        screenshot_manifest = _write_screenshot_manifest(
            root_fd,
            {"a": leg_a, "b": leg_b},
            ab_report,
        )
        if screenshot_manifest["manual_target_count"] != 5:
            raise D408Error("manual target count is not exact")
        screenshot_sha = _sha_path(SCREENSHOT_MANIFEST_PATH)
        screenshot_checkpoints.append(
            _verify_screenshot_manifest(
                root_fd,
                screenshot_sha,
                "pre_prompt",
            )
        )
        screenshots_ready_row = phase.append(
            "screenshots_ready",
            {
                "image_count": 5,
                "images_sha256": screenshot_checkpoints[-1][
                    "images_sha256"
                ],
                "screenshot_manifest_sha256": screenshot_sha,
            },
        )
        _ping_writer(
            writer_process,
            writer_channel,
            nonce,
            screenshots_ready_row,
            writer_start,
        )
        if time.monotonic_ns() >= prearm_hard_deadline_ns:
            raise TimeoutError("pre-arm hard deadline expired before manual prompt")
        manual_prompt_started_ns = time.monotonic_ns()
        manual_deadline_ns = manual_prompt_started_ns + MANUAL_TIMEOUT_NS
        writer_deadline_ns = (
            manual_deadline_ns - WRITER_DEADLINE_LEAD_NS
        )
        prompt_row = phase.append(
            "manual_prompt",
            {
                "manual_basename": MANUAL_PATH.name,
                "manual_deadline_monotonic_ns": manual_deadline_ns,
                "manual_prompt_started_monotonic_ns": (
                    manual_prompt_started_ns
                ),
                "new_controlled_physics_steps": 0,
                "screenshot_manifest_sha256": screenshot_sha,
                "writer_deadline_monotonic_ns": writer_deadline_ns,
            },
        )
        print(
            "D408_MANUAL_PROMPT "
            + json.dumps(
                {
                    "manual_deadline_monotonic_ns": manual_deadline_ns,
                    "manual_prompt_started_monotonic_ns": (
                        manual_prompt_started_ns
                    ),
                    "manual_targets": [
                        item["path"] for item in screenshot_manifest["images"]
                    ],
                    "required_boolean_fields": list(REQUIRED_BOOLEAN_FIELDS),
                    "submit": "one strict JSON line on controller stdin",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        manual_input = _read_manual_stdin(manual_deadline_ns)
        screenshot_checkpoints.append(
            _verify_screenshot_manifest(
                root_fd,
                screenshot_sha,
                "before_writer_send",
            )
        )
        manual, receipt = _publish_via_writer(
            root_fd,
            writer_process,
            writer_channel,
            nonce,
            phase,
            manual_prompt_started_ns,
            manual_deadline_ns,
            prompt_row,
            screenshot_sha,
            screenshot_checkpoints[-1]["images_sha256"],
            manual_input,
            writer_start,
        )
        writer_channel = None
        screenshot_checkpoints.append(
            _verify_screenshot_manifest(
                root_fd,
                screenshot_sha,
                "post_publication",
            )
        )
        if _secure_read_manual_once(
            root_fd,
            receipt["manual_sha256"],
            receipt["manual_size"],
        ) != manual:
            raise D408Error("published manual drifted after writer exit")
        _validate_manifest(prereg, "after_manual_publication", checkpoints)
        overlay_checkpoints.append(
            _validate_repository_overlay(
                prereg,
                root_fd,
                root_dev,
                root_ino,
                "after_manual_publication",
                admission_exact=False,
                stable_publication_gate=True,
            )
        )
        phase.append(
            "manual_received",
            {
                "manual_pass": manual["pass"],
                "manual_sha256": receipt["manual_sha256"],
                "published_before_writer_deadline": receipt[
                    "published_before_writer_deadline"
                ],
                "received": True,
            },
        )
        _validate_manifest(prereg, "before_completion", checkpoints)
        if _secure_read_manual_once(
            root_fd,
            receipt["manual_sha256"],
            receipt["manual_size"],
        ) != manual:
            raise D408Error("published manual drifted before completion")
        screenshot_checkpoints.append(
            _verify_screenshot_manifest(
                root_fd,
                screenshot_sha,
                "pre_completion",
            )
        )
        expected_screenshot_digest = screenshot_checkpoints[0][
            "images_sha256"
        ]
        if any(
            checkpoint["images_sha256"] != expected_screenshot_digest
            for checkpoint in screenshot_checkpoints
        ):
            raise D408Error("manual screenshot bytes drifted across checkpoints")
        overlay_checkpoints.append(
            _validate_repository_overlay(
                prereg,
                root_fd,
                root_dev,
                root_ino,
                "before_completion",
                admission_exact=False,
                stable_publication_gate=True,
            )
        )
        _write_json_x(
            SOURCE_CHECKPOINTS_PATH,
            {
                "artifact": "D408_D407_SOURCE_IMMUTABILITY_CHECKPOINTS_V1",
                "checkpoints": checkpoints,
                "expected_checkpoint_names": [
                    "admission",
                    "after_leg_a_capture",
                    "after_leg_b_capture",
                    "after_manual_publication",
                    "before_completion",
                ],
                "pass": len(checkpoints) == 5,
            },
        )
        _write_json_x(
            SCREENSHOT_CHECKPOINTS_PATH,
            {
                "artifact": "D408_SCREENSHOT_INTEGRITY_CHECKPOINTS_V1",
                "checkpoints": screenshot_checkpoints,
                "expected_checkpoint_names": [
                    "pre_prompt",
                    "before_writer_send",
                    "post_publication",
                    "pre_completion",
                ],
                "image_count_each": 5,
                "pass": len(screenshot_checkpoints) == 4,
            },
        )
        status = D408_PASS_STATUS if manual["pass"] else D408_FAIL_STATUS
        if status == D408_PASS_STATUS:
            observability_pass = True
        elif status == D408_FAIL_STATUS:
            observability_pass = False
        else:
            raise D408Error("unreachable D408 status")
        completion = {
            "artifact": "D408_COMPLETION_SUMMARY_V1",
            "counters": _runtime_counters(),
            "d407_final_verdict": D407_FINAL_VERDICT,
            "d407_retroactive_pass": False,
            "d407_root_artifact_integrity": None,
            "d408_manual_screenshot_integrity": True,
            "g0a_pass": False,
            "manual_inspection": {
                "pass": manual["pass"],
                "received": True,
                "sha256": receipt["manual_sha256"],
            },
            "new_controlled_physics_steps": 0,
            "observability_repair_pass": observability_pass,
            "repository_overlay_checkpoints": overlay_checkpoints,
            "scientific_null_claims": SCIENTIFIC_NULL_CLAIMS,
            "scientific_verdict": None,
            "screenshot_integrity_checkpoint_count": len(
                screenshot_checkpoints
            ),
            "source_immutability_checkpoint_count": len(checkpoints),
            "status": status,
        }
        completion_ready_row = phase.append(
            "completion_ready",
            {
                "manual_sha256": receipt["manual_sha256"],
                "status": status,
            },
        )
        completion["terminal_phase"] = {
            "event": completion_ready_row["event"],
            "row_sha256": completion_ready_row["row_sha256"],
            "sequence": completion_ready_row["sequence"],
        }
        phase.close()
        phase = None
        terminal_publish_attempted = True
        _atomic_publish_json_at(
            root_fd,
            TERMINAL_PENDING_PATH.name,
            TERMINAL_PATH.name,
            completion,
        )
        return 0 if observability_pass else 2
    except BaseException as exc:
        _terminate_writer(writer_process)
        if writer_channel is not None:
            writer_channel.close()
        terminal_phase: dict[str, Any] | None = None
        if phase is not None:
            try:
                fail_row = phase.append(
                    "fail_stop",
                    {
                        "error": type(exc).__name__,
                        "manual_final_exists": MANUAL_PATH.exists(),
                        "manual_pending_exists": MANUAL_PENDING_PATH.exists(),
                        "message": str(exc),
                        "source_checkpoint_count": len(checkpoints),
                        "writer_returncode": (
                            None
                            if writer_process is None
                            else writer_process.poll()
                        ),
                    },
                )
                terminal_phase = {
                    "event": fail_row["event"],
                    "row_sha256": fail_row["row_sha256"],
                    "sequence": fail_row["sequence"],
                }
            except BaseException:
                pass
            phase.close()
            phase = None
        if terminal_publish_attempted:
            raise
        failure = {
            "artifact": "D408_FAILURE_SUMMARY_V1",
            "counters": _runtime_counters(),
            "d407_final_verdict": D407_FINAL_VERDICT,
            "d407_retroactive_pass": False,
            "d407_root_artifact_integrity": None,
            "error": type(exc).__name__,
            "g0a_pass": False,
            "message": str(exc),
            "new_controlled_physics_steps": 0,
            "scientific_null_claims": SCIENTIFIC_NULL_CLAIMS,
            "scientific_verdict": None,
            "status": D408_FAIL_STATUS,
            "terminal_phase": terminal_phase,
        }
        terminal_publish_attempted = True
        _atomic_publish_json_at(
            root_fd,
            TERMINAL_PENDING_PATH.name,
            TERMINAL_PATH.name,
            failure,
        )
        raise
    finally:
        os.close(root_fd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-contract-json", action="store_true")
    parser.add_argument("--stage", choices=("controller",))
    parser.add_argument("--approved-tuple-sha256")
    return parser.parse_args()


def main() -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError("D408 controller must run with python -B")
    args = _parse_args()
    if args.print_contract_json:
        print(json.dumps(_contract(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.stage != "controller":
        raise D408Error("--stage controller is required")
    if not isinstance(args.approved_tuple_sha256, str) or len(
        args.approved_tuple_sha256
    ) != 64:
        raise D408Error("--approved-tuple-sha256 must be 64 hexadecimal characters")
    try:
        bytes.fromhex(args.approved_tuple_sha256)
    except ValueError as exc:
        raise D408Error("approved tuple SHA is not hexadecimal") from exc
    return _run_controller(args.approved_tuple_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
