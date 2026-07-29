#!/usr/bin/env python3
"""D408 tuple-bound pre-armed manual inspection writer.

This process never imports or launches Isaac/Kit/PhysX.  It is spawned once,
before any replay, and receives its secret nonce and the eventual inspection
booleans only through an inherited Unix socket.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import hmac
import io
import json
import os
import socket
import stat
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_PREREG_SHA256 = "0c0f1c03d10210e205d5be0b25fd84c7d94c109fb26387f77fa22f6b984c8d0d"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D408_ROOT = (
    PROJECT_ROOT
    / "claudedocs/runtime_logs/grasp_track/g0a_d408"
    / "attempt1_d407_manual_observability_completion_repair"
)
PREREG_PATH = D408_ROOT / "d408_preregistration.json"
TUPLE_PATH = D408_ROOT / "d408_proposed_runtime_hash_tuple.json"
PHASE_PATH = D408_ROOT / "d408_controller_phase_markers.jsonl"
SCREENSHOT_MANIFEST_PATH = D408_ROOT / "d408_screenshot_manifest.json"
MANUAL_BASENAME = "d408_manual_visual_inspection.json"
MANUAL_PENDING_BASENAME = ".d408_manual_visual_inspection.json.pending"
MANUAL_PATH = D408_ROOT / MANUAL_BASENAME
CONTROLLER_PATH = (
    PROJECT_ROOT
    / "sim_scripts/cyl34_top_view_d408_d407_manual_observability_completion_repair_controller.py"
)
WRITER_PATH = Path(__file__).resolve()

MAX_PROTOCOL_BYTES = 64 * 1024
MAX_MANUAL_BYTES = 64 * 1024
MAX_SCREENSHOT_BYTES = 64 * 1024 * 1024
MAX_SCREENSHOT_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_PHASE_BYTES = 1024 * 1024
MAX_NOTES_UTF8_BYTES = 4096
MANUAL_TIMEOUT_NS = 600_000_000_000
WRITER_DEADLINE_LEAD_NS = 5_000_000_000
RENAME_NOREPLACE = 1

D407_FINAL_VERDICT = "D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP"
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

EXPECTED_MANUAL_IMAGE_LAYOUT = {
    "leg_a_a64_replay/d408_clean_spatial.png": [1120, 900],
    "leg_a_a64_replay/d408_decision_sheet.png": [1920, 1080],
    "leg_b_sdf_res256_replay/d408_clean_spatial.png": [1120, 900],
    "leg_b_sdf_res256_replay/d408_decision_sheet.png": [1920, 1080],
    "d408_ab_comparison_sheet.png": [3840, 1080],
}

LEG_DIRECTORIES = {
    "a": "leg_a_a64_replay",
    "b": "leg_b_sdf_res256_replay",
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


class ProtocolError(RuntimeError):
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


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON constant: {value}")


def _strict_json_bytes(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("JSON is not UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProtocolError(f"strict JSON parse failed: {exc}") from exc


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProtocolError(f"unsafe hash source: {path}")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
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
            raise ProtocolError(f"file changed while hashing: {path}")
    finally:
        os.close(fd)
    return digest.hexdigest()


def _secure_read(path: Path, max_bytes: int) -> tuple[bytes, os.stat_result]:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ProtocolError(f"not a regular file: {path}")
        if before.st_nlink != 1:
            raise ProtocolError(f"unexpected link count: {path}")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise ProtocolError(f"unsafe file size: {path} ({before.st_size})")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise ProtocolError(f"file exceeds maximum size: {path}")
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
        if identity_before != identity_after or len(raw) != before.st_size:
            raise ProtocolError(f"file changed while reading: {path}")
        return raw, before
    finally:
        os.close(fd)


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
        raise ProtocolError(f"unsafe D408-root-relative path: {relative_path!r}")
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
            raise ProtocolError(f"unsafe relative file: {relative_path}")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise ProtocolError(f"unsafe relative file size: {relative_path}")
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
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or len(raw) != before.st_size:
            raise ProtocolError(f"relative file changed while reading: {relative_path}")
        if len(raw) > max_bytes:
            raise ProtocolError(f"relative file exceeds limit: {relative_path}")
        return raw, before
    finally:
        os.close(fd)


def _verify_root_binding(args: argparse.Namespace) -> int:
    root_fd = args.root_fd
    metadata = os.fstat(root_fd)
    path_metadata = os.lstat(D408_ROOT)
    expected = (args.root_dev, args.root_ino)
    if args.root != str(D408_ROOT):
        raise ProtocolError("D408 root argument mismatch")
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(path_metadata.st_mode):
        raise ProtocolError("D408 root is not a bound real directory")
    if (metadata.st_dev, metadata.st_ino) != expected or (
        path_metadata.st_dev,
        path_metadata.st_ino,
    ) != expected:
        raise ProtocolError("D408 root path/dirfd identity mismatch")
    return root_fd


def _proc_start_ticks(pid: int) -> int:
    path = Path(f"/proc/{pid}/stat")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProtocolError(f"unsafe proc stat identity: {path}")
        raw = os.read(fd, 64 * 1024)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ProtocolError(f"proc stat identity changed: {path}")
    finally:
        os.close(fd)
    if not raw or len(raw) >= 64 * 1024:
        raise ProtocolError(f"invalid proc stat size: {path}")
    text = raw.decode("utf-8")
    right_paren = text.rfind(")")
    if right_paren < 0:
        raise ProtocolError(f"malformed /proc/{pid}/stat")
    fields_after_comm = text[right_paren + 1 :].strip().split()
    if len(fields_after_comm) <= 19:
        raise ProtocolError(f"short /proc/{pid}/stat")
    return int(fields_after_comm[19])


def _recv_json_line(channel: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = channel.recv(4096)
        if not chunk:
            raise ProtocolError("socket closed before a complete JSON line")
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_PROTOCOL_BYTES:
            raise ProtocolError("protocol message exceeds size limit")
        raw = b"".join(chunks)
        newline = raw.find(b"\n")
        if newline >= 0:
            if raw[newline + 1 :]:
                raise ProtocolError("multiple protocol messages in one read are forbidden")
            value = _strict_json_bytes(raw[: newline + 1])
            if not isinstance(value, dict):
                raise ProtocolError("protocol message must be an object")
            return value


def _send_json_line(channel: socket.socket, value: dict[str, Any]) -> None:
    raw = _canonical_bytes(value)
    if len(raw) > MAX_PROTOCOL_BYTES:
        raise ProtocolError("outgoing protocol message exceeds size limit")
    channel.sendall(raw)


def _hmac_hex(nonce: bytes, body: dict[str, Any]) -> str:
    return hmac.new(nonce, _canonical_bytes(body), hashlib.sha256).hexdigest()


def _expect_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ProtocolError(
            f"{label} keys mismatch: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _read_phase_chain(
    root_fd: int,
    expected_dev: int,
    expected_ino: int,
) -> list[dict[str, Any]]:
    raw, metadata = _secure_read_relative(
        root_fd,
        PHASE_PATH.name,
        MAX_PHASE_BYTES,
    )
    if (metadata.st_dev, metadata.st_ino) != (expected_dev, expected_ino):
        raise ProtocolError("phase log inode binding mismatch")
    rows: list[dict[str, Any]] = []
    previous_sha: str | None = None
    for index, line in enumerate(raw.splitlines(), start=1):
        row = _strict_json_bytes(line + b"\n")
        if not isinstance(row, dict):
            raise ProtocolError("phase row is not an object")
        _expect_exact_keys(
            row,
            {
                "artifact",
                "details",
                "event",
                "monotonic_ns",
                "prev_row_sha256",
                "row_sha256",
                "sequence",
                "utc",
            },
            "phase row",
        )
        if row["artifact"] != "D408_CONTROLLER_PHASE_ROW_V1":
            raise ProtocolError("phase artifact mismatch")
        if type(row["sequence"]) is not int or row["sequence"] != index:
            raise ProtocolError("phase sequence is not exact")
        if row["prev_row_sha256"] != previous_sha:
            raise ProtocolError("phase previous-row SHA mismatch")
        core = dict(row)
        row_sha = core.pop("row_sha256")
        if not isinstance(row_sha, str) or row_sha != _sha_bytes(_canonical_bytes(core)):
            raise ProtocolError("phase row SHA mismatch")
        previous_sha = row_sha
        rows.append(row)
    if not rows:
        raise ProtocolError("phase log is empty")
    return rows


def _validate_static_bindings(
    args: argparse.Namespace,
    root_fd: int,
) -> dict[str, Any]:
    if args.manual_basename != MANUAL_BASENAME:
        raise ProtocolError("manual basename mismatch")
    if os.getppid() != args.controller_pid:
        raise ProtocolError("writer parent PID is not the bound controller")
    if args.writer_sha256 != _sha_path(WRITER_PATH):
        raise ProtocolError("writer SHA binding mismatch")
    if args.controller_sha256 != _sha_path(CONTROLLER_PATH):
        raise ProtocolError("controller SHA binding mismatch")
    if _proc_start_ticks(args.controller_pid) != args.controller_start_ticks:
        raise ProtocolError("controller PID/start-time binding mismatch")
    prereg_raw, _ = _secure_read_relative(
        root_fd,
        PREREG_PATH.name,
        4 * 1024 * 1024,
    )
    if _sha_bytes(prereg_raw) != EXPECTED_PREREG_SHA256:
        raise ProtocolError("preregistration SHA mismatch")
    tuple_raw, _ = _secure_read_relative(
        root_fd,
        TUPLE_PATH.name,
        1024 * 1024,
    )
    if _sha_bytes(tuple_raw) != args.approved_tuple_sha256:
        raise ProtocolError("approved tuple-file SHA mismatch")

    prereg = _strict_json_bytes(prereg_raw)
    if not isinstance(prereg, dict):
        raise ProtocolError("preregistration is not an object")
    if prereg.get("d407_source_manifest_sha256") != args.input_manifest_sha256:
        raise ProtocolError("D407 input-manifest binding mismatch")
    if prereg.get("actual_execution_requires_separate_tuple_sha_approval") is not True:
        raise ProtocolError("prereg approval boundary is not exact")

    tuple_data = _strict_json_bytes(tuple_raw)
    if not isinstance(tuple_data, dict):
        raise ProtocolError("tuple is not an object")
    hashes = tuple_data.get("hashes")
    if not isinstance(hashes, dict):
        raise ProtocolError("tuple hashes are missing")
    expected_hashes = {
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "controller_sha256": args.controller_sha256,
        "manual_writer_sha256": args.writer_sha256,
    }
    for key, expected in expected_hashes.items():
        if hashes.get(key) != expected:
            raise ProtocolError(f"tuple {key} mismatch")
    return prereg


def _validate_manual_input(value: Any) -> tuple[dict[str, bool], str]:
    if not isinstance(value, dict):
        raise ProtocolError("manual input must be an object")
    _expect_exact_keys(value, {"required_fields", "notes"}, "manual input")
    fields = value["required_fields"]
    notes = value["notes"]
    if not isinstance(fields, dict):
        raise ProtocolError("required_fields must be an object")
    _expect_exact_keys(fields, set(REQUIRED_BOOLEAN_FIELDS), "required_fields")
    normalized: dict[str, bool] = {}
    for key in REQUIRED_BOOLEAN_FIELDS:
        if type(fields[key]) is not bool:
            raise ProtocolError(f"manual field is not boolean: {key}")
        normalized[key] = fields[key]
    if (
        not isinstance(notes, str)
        or len(notes.encode("utf-8")) > MAX_NOTES_UTF8_BYTES
    ):
        raise ProtocolError("notes must be a UTF-8 string of at most 4096 bytes")
    return normalized, notes


def _verify_screenshot_manifest(
    root_fd: int,
    expected_sha256: str,
) -> dict[str, Any]:
    from PIL import Image

    manifest_raw, manifest_metadata = _secure_read_relative(
        root_fd,
        SCREENSHOT_MANIFEST_PATH.name,
        MAX_SCREENSHOT_MANIFEST_BYTES,
    )
    if _sha_bytes(manifest_raw) != expected_sha256:
        raise ProtocolError("screenshot manifest SHA mismatch")
    manifest = _strict_json_bytes(manifest_raw)
    if not isinstance(manifest, dict):
        raise ProtocolError("screenshot manifest is not an object")
    _expect_exact_keys(
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
        raise ProtocolError("screenshot manifest invariant mismatch")
    ab_report = manifest["ab_report"]
    if not isinstance(ab_report, dict):
        raise ProtocolError("screenshot manifest ab_report is not an object")
    _expect_exact_keys(ab_report, {"dimensions", "sha256"}, "ab_report")
    if (
        ab_report["dimensions"] != [3840, 1080]
        or not isinstance(ab_report["sha256"], str)
        or len(ab_report["sha256"]) != 64
    ):
        raise ProtocolError("screenshot manifest ab_report is malformed")
    leg_reports = manifest["leg_reports"]
    if not isinstance(leg_reports, dict):
        raise ProtocolError("screenshot manifest leg_reports is not an object")
    _expect_exact_keys(leg_reports, set(LEG_DIRECTORIES), "leg_reports")
    for leg, directory in LEG_DIRECTORIES.items():
        report = leg_reports[leg]
        if not isinstance(report, dict):
            raise ProtocolError(f"leg {leg} report is not an object")
        _expect_exact_keys(
            report,
            {
                "clean_spatial_path",
                "decision_sheet_path",
                "full_screenshot_path",
                "historical_trace_rows",
                "leg",
                "validation_path",
                "validation_sha256",
            },
            f"leg {leg} report",
        )
        expected_prefix = str(
            (D408_ROOT / directory).relative_to(PROJECT_ROOT)
        )
        expected_paths = {
            "clean_spatial_path": f"{expected_prefix}/d408_clean_spatial.png",
            "decision_sheet_path": f"{expected_prefix}/d408_decision_sheet.png",
            "full_screenshot_path": (
                f"{expected_prefix}/d408_rerun_full_diagnostic.png"
            ),
            "validation_path": f"{expected_prefix}/d408_rerun_validation.json",
        }
        if any(report[key] != value for key, value in expected_paths.items()):
            raise ProtocolError(f"leg {leg} report path binding mismatch")
        if report["leg"] != leg or report["historical_trace_rows"] != 500:
            raise ProtocolError(f"leg {leg} report invariant mismatch")
        validation_sha = report["validation_sha256"]
        if not isinstance(validation_sha, str) or len(validation_sha) != 64:
            raise ProtocolError(f"leg {leg} validation SHA is malformed")
    images = manifest["images"]
    if not isinstance(images, list) or len(images) != 5:
        raise ProtocolError("screenshot manifest image count mismatch")
    verified_images: list[dict[str, Any]] = []
    verified_by_path: dict[str, dict[str, Any]] = {}
    observed_paths: list[str] = []
    for item in images:
        if not isinstance(item, dict):
            raise ProtocolError("screenshot manifest image row is not an object")
        _expect_exact_keys(
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
            raise ProtocolError("screenshot relative path is not a string")
        expected_dimensions = EXPECTED_MANUAL_IMAGE_LAYOUT.get(relative_path)
        if expected_dimensions is None:
            raise ProtocolError(f"unexpected screenshot path: {relative_path}")
        expected_project_path = str(
            (D408_ROOT / relative_path).relative_to(PROJECT_ROOT)
        )
        if item["path"] != expected_project_path:
            raise ProtocolError("screenshot path binding mismatch")
        raw, metadata = _secure_read_relative(
            root_fd,
            relative_path,
            MAX_SCREENSHOT_BYTES,
        )
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            dimensions = list(image.size)
            if image.format != "PNG":
                raise ProtocolError(f"manual image is not PNG: {relative_path}")
        image_sha = _sha_bytes(raw)
        if (
            item["bytes"] != metadata.st_size
            or item["dimensions"] != dimensions
            or dimensions != expected_dimensions
            or item["sha256"] != image_sha
        ):
            raise ProtocolError(f"manual image integrity drift: {relative_path}")
        observed_paths.append(relative_path)
        verified_images.append(
            {
                "bytes": metadata.st_size,
                "dimensions": dimensions,
                "root_relative_path": relative_path,
                "sha256": image_sha,
            }
        )
        verified_by_path[relative_path] = verified_images[-1]
    if sorted(observed_paths) != sorted(EXPECTED_MANUAL_IMAGE_LAYOUT):
        raise ProtocolError("manual screenshot path set mismatch")
    if (
        verified_by_path["d408_ab_comparison_sheet.png"]["sha256"]
        != ab_report["sha256"]
    ):
        raise ProtocolError("A/B report SHA does not bind the comparison PNG")
    for leg, directory in LEG_DIRECTORIES.items():
        report = leg_reports[leg]
        for report_key, basename in (
            ("clean_spatial_path", "d408_clean_spatial.png"),
            ("decision_sheet_path", "d408_decision_sheet.png"),
        ):
            relative_path = f"{directory}/{basename}"
            expected_project_path = str(
                (D408_ROOT / relative_path).relative_to(PROJECT_ROOT)
            )
            if (
                report[report_key] != expected_project_path
                or verified_by_path[relative_path]["sha256"]
                != next(
                    item["sha256"]
                    for item in images
                    if item["root_relative_path"] == relative_path
                )
            ):
                raise ProtocolError(
                    f"leg {leg} report/image binding mismatch: {report_key}"
                )
    return {
        "image_count": len(verified_images),
        "images_sha256": _sha_bytes(_canonical_bytes(verified_images)),
        "manifest_bytes": manifest_metadata.st_size,
        "manifest_sha256": expected_sha256,
        "verified_images": verified_images,
    }


def _rename_noreplace(
    source_name: str,
    destination_name: str,
    directory_fd: int,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ProtocolError("renameat2 is unavailable; fallback is forbidden")
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


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise ProtocolError("short write to manual temp file")
        offset += written


def _atomic_publish_manual(
    root_fd: int,
    document: dict[str, Any],
) -> tuple[str, int, int]:
    raw = _canonical_bytes(document)
    if len(raw) > MAX_MANUAL_BYTES:
        raise ProtocolError("manual document exceeds maximum size")
    temp_fd = -1
    try:
        temp_fd = os.open(
            MANUAL_PENDING_BASENAME,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        metadata = os.fstat(temp_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProtocolError("manual temp file is not an exclusive regular file")
        _write_all(temp_fd, raw)
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = -1
        _rename_noreplace(MANUAL_PENDING_BASENAME, MANUAL_BASENAME, root_fd)
        os.fsync(root_fd)
        completed_monotonic_ns = time.monotonic_ns()
        final_fd = os.open(
            MANUAL_BASENAME,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            final_meta = os.fstat(final_fd)
            if not stat.S_ISREG(final_meta.st_mode) or final_meta.st_nlink != 1:
                raise ProtocolError("published manual file identity is unsafe")
            final_raw = b""
            while True:
                chunk = os.read(final_fd, 65536)
                if not chunk:
                    break
                final_raw += chunk
        finally:
            os.close(final_fd)
        if final_raw != raw:
            raise ProtocolError("published manual bytes differ from source bytes")
        return _sha_bytes(raw), len(raw), completed_monotonic_ns
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)


def _contract() -> dict[str, Any]:
    return {
        "artifact": "D408_MANUAL_WRITER_CONTRACT_V1",
        "expected_prereg_sha256": EXPECTED_PREREG_SHA256,
        "forbidden_runtime_prefixes": [
            "isaaclab",
            "isaacsim",
            "omni",
            "pxr",
            "carb",
            "warp",
        ],
        "manual_basename": MANUAL_BASENAME,
        "manual_output_path": str(MANUAL_PATH.relative_to(PROJECT_ROOT)),
        "manual_pending_basename": MANUAL_PENDING_BASENAME,
        "manual_timeout_ns": MANUAL_TIMEOUT_NS,
        "protocol_schemas": {
            "authenticated_envelope_keys": ["body", "hmac_sha256"],
            "ack_body_keys": sorted(
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
                }
            ),
            "arm_binding_keys": sorted(
                {
                    "approved_tuple_sha256",
                    "controller_pid",
                    "controller_sha256",
                    "controller_start_ticks",
                    "input_manifest_sha256",
                    "manual_basename",
                    "phase_dev",
                    "phase_ino",
                    "prearm_hard_deadline_monotonic_ns",
                    "preregistration_sha256",
                    "root_dev",
                    "root_ino",
                    "writer_sha256",
                }
            ),
            "arm_message_keys": ["bindings", "nonce_hex", "op"],
            "manual_binding_keys": sorted(
                {
                    "approved_tuple_sha256",
                    "controller_pid",
                    "controller_sha256",
                    "controller_start_ticks",
                    "d407_source_manifest_sha256",
                    "nonce_sha256",
                    "phase_dev",
                    "phase_ino",
                    "phase_row_sha256",
                    "phase_sequence",
                    "preregistration_sha256",
                    "root_dev",
                    "root_ino",
                    "screenshot_manifest_sha256",
                    "writer_pid",
                    "writer_sha256",
                    "writer_start_ticks",
                }
            ),
            "manual_deadline_keys": sorted(
                {
                    "manual_deadline_monotonic_ns",
                    "manual_prompt_started_monotonic_ns",
                    "writer_deadline_monotonic_ns",
                }
            ),
            "manual_input_keys": ["notes", "required_fields"],
            "limits_bytes": {
                "manual_document": MAX_MANUAL_BYTES,
                "manual_notes_utf8": MAX_NOTES_UTF8_BYTES,
                "phase_log": MAX_PHASE_BYTES,
                "protocol_message": MAX_PROTOCOL_BYTES,
                "screenshot_image": MAX_SCREENSHOT_BYTES,
                "screenshot_manifest": MAX_SCREENSHOT_MANIFEST_BYTES,
            },
            "manual_output_keys": sorted(
                {
                    "artifact",
                    "bindings",
                    "deadline",
                    "notes",
                    "pass",
                    "received",
                    "required_fields",
                    "source_science",
                }
            ),
            "ping_body_keys": sorted(
                {
                    "op",
                    "phase_event",
                    "phase_row_sha256",
                    "phase_sequence",
                }
            ),
            "pong_body_keys": sorted(
                {
                    "op",
                    "phase_event",
                    "phase_row_sha256",
                    "phase_sequence",
                    "writer_pid",
                    "writer_start_ticks",
                }
            ),
            "publish_body_keys": sorted(
                {
                    "manual_deadline_monotonic_ns",
                    "manual_input",
                    "manual_prompt_started_monotonic_ns",
                    "op",
                    "phase_row_sha256",
                    "phase_sequence",
                    "screenshot_manifest_sha256",
                    "writer_deadline_monotonic_ns",
                }
            ),
            "ready_body_keys": sorted(
                {
                    "nonce_sha256",
                    "op",
                    "phase_row_sha256",
                    "phase_sequence",
                    "writer_pid",
                    "writer_sha256",
                    "writer_start_ticks",
                }
            ),
            "phase_row_keys": sorted(
                {
                    "artifact",
                    "details",
                    "event",
                    "monotonic_ns",
                    "prev_row_sha256",
                    "row_sha256",
                    "sequence",
                    "utc",
                }
            ),
            "protocol_line_framing": {
                "canonical_json_trailing_lf": True,
                "exactly_one_message_per_read": True,
            },
            "required_fields_keys": list(REQUIRED_BOOLEAN_FIELDS),
            "screenshot_image_row_keys": sorted(
                {
                    "bytes",
                    "dimensions",
                    "manual_role",
                    "path",
                    "root_relative_path",
                    "sha256",
                }
            ),
            "screenshot_ab_report_keys": ["dimensions", "sha256"],
            "screenshot_leg_report_keys": sorted(
                {
                    "clean_spatial_path",
                    "decision_sheet_path",
                    "full_screenshot_path",
                    "historical_trace_rows",
                    "leg",
                    "validation_path",
                    "validation_sha256",
                }
            ),
            "screenshot_leg_report_names": sorted(LEG_DIRECTORIES),
            "screenshot_manifest_keys": sorted(
                {
                    "ab_report",
                    "artifact",
                    "images",
                    "leg_reports",
                    "manual_target_count",
                    "new_controlled_physics_steps",
                }
            ),
            "source_science_keys": sorted(
                {
                    "d407_final_verdict",
                    "d407_retroactive_pass",
                    "g0a_pass",
                    "new_controlled_physics_steps",
                    "scientific_null_claims",
                    "scientific_verdict",
                }
            ),
            "source_science": {
                "d407_final_verdict": D407_FINAL_VERDICT,
                "d407_retroactive_pass": False,
                "g0a_pass": False,
                "new_controlled_physics_steps": 0,
                "scientific_null_claims": SCIENTIFIC_NULL_CLAIMS,
                "scientific_verdict": None,
            },
        },
        "publication": [
            "inherited_bound_root_dirfd",
            "fixed_pending_openat_O_EXCL_O_NOFOLLOW",
            "file_fsync",
            "renameat2_RENAME_NOREPLACE",
            "directory_fsync",
            "PUBLISHED_FSYNCED_ack",
        ],
        "required_boolean_fields": list(REQUIRED_BOOLEAN_FIELDS),
        "retry_count": 0,
        "screenshot_layout": EXPECTED_MANUAL_IMAGE_LAYOUT,
        "writer_deadline_lead_ns": WRITER_DEADLINE_LEAD_NS,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-contract-json", action="store_true")
    parser.add_argument("--stage", choices=("writer",))
    parser.add_argument("--socket-fd", type=int)
    parser.add_argument("--root")
    parser.add_argument("--root-fd", type=int)
    parser.add_argument("--root-dev", type=int)
    parser.add_argument("--root-ino", type=int)
    parser.add_argument("--controller-pid", type=int)
    parser.add_argument("--controller-start-ticks", type=int)
    parser.add_argument("--controller-sha256")
    parser.add_argument("--writer-sha256")
    parser.add_argument("--approved-tuple-sha256")
    parser.add_argument("--phase-dev", type=int)
    parser.add_argument("--phase-ino", type=int)
    parser.add_argument("--input-manifest-sha256")
    parser.add_argument("--manual-basename")
    parser.add_argument("--prearm-hard-deadline-monotonic-ns", type=int)
    return parser.parse_args()


def _require_writer_args(args: argparse.Namespace) -> None:
    required = (
        "socket_fd",
        "root",
        "root_fd",
        "root_dev",
        "root_ino",
        "controller_pid",
        "controller_start_ticks",
        "controller_sha256",
        "writer_sha256",
        "approved_tuple_sha256",
        "phase_dev",
        "phase_ino",
        "input_manifest_sha256",
        "manual_basename",
        "prearm_hard_deadline_monotonic_ns",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ProtocolError(f"missing writer arguments: {missing}")


def _run_writer(args: argparse.Namespace) -> int:
    _require_writer_args(args)
    if args.prearm_hard_deadline_monotonic_ns <= time.monotonic_ns():
        raise ProtocolError("pre-arm hard deadline is already stale")
    root_fd = _verify_root_binding(args)
    try:
        return _run_writer_bound(args, root_fd)
    finally:
        os.close(root_fd)


def _run_writer_bound(args: argparse.Namespace, root_fd: int) -> int:
    _validate_static_bindings(args, root_fd)
    for basename in (MANUAL_BASENAME, MANUAL_PENDING_BASENAME):
        try:
            os.stat(basename, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise FileExistsError(D408_ROOT / basename)

    channel = socket.socket(fileno=args.socket_fd)
    channel.settimeout(15.0)
    hello = _recv_json_line(channel)
    _expect_exact_keys(hello, {"bindings", "nonce_hex", "op"}, "arm message")
    if hello["op"] != "arm":
        raise ProtocolError("first operation is not arm")
    bindings = hello["bindings"]
    if not isinstance(bindings, dict):
        raise ProtocolError("arm bindings are not an object")
    expected_bindings = {
        "approved_tuple_sha256": args.approved_tuple_sha256,
        "controller_pid": args.controller_pid,
        "controller_sha256": args.controller_sha256,
        "controller_start_ticks": args.controller_start_ticks,
        "input_manifest_sha256": args.input_manifest_sha256,
        "manual_basename": args.manual_basename,
        "phase_dev": args.phase_dev,
        "phase_ino": args.phase_ino,
        "prearm_hard_deadline_monotonic_ns": (
            args.prearm_hard_deadline_monotonic_ns
        ),
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "root_dev": args.root_dev,
        "root_ino": args.root_ino,
        "writer_sha256": args.writer_sha256,
    }
    if bindings != expected_bindings:
        raise ProtocolError("arm binding payload mismatch")
    nonce_hex = hello["nonce_hex"]
    if not isinstance(nonce_hex, str) or len(nonce_hex) != 64:
        raise ProtocolError("nonce must be a 256-bit hex string")
    try:
        nonce = bytes.fromhex(nonce_hex)
    except ValueError as exc:
        raise ProtocolError("nonce is not valid hexadecimal") from exc
    if len(nonce) != 32:
        raise ProtocolError("nonce is not 256 bits")
    nonce_sha256 = _sha_bytes(nonce)

    initial_rows = _read_phase_chain(root_fd, args.phase_dev, args.phase_ino)
    initial_row = initial_rows[-1]
    expected_initial_details = {
        "approved_tuple_sha256": args.approved_tuple_sha256,
        "controller_pid": args.controller_pid,
        "controller_sha256": args.controller_sha256,
        "controller_start_ticks": args.controller_start_ticks,
        "d407_source_manifest_sha256": args.input_manifest_sha256,
        "prearm_hard_deadline_monotonic_ns": (
            args.prearm_hard_deadline_monotonic_ns
        ),
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "root_dev": args.root_dev,
        "root_ino": args.root_ino,
        "writer_sha256": args.writer_sha256,
    }
    if (
        initial_row["event"] != "controller_started"
        or initial_row["details"] != expected_initial_details
    ):
        raise ProtocolError("writer was not armed at the controller_started phase")
    writer_start_ticks = _proc_start_ticks(os.getpid())
    ready_body = {
        "nonce_sha256": nonce_sha256,
        "op": "ready",
        "phase_row_sha256": initial_row["row_sha256"],
        "phase_sequence": initial_row["sequence"],
        "writer_pid": os.getpid(),
        "writer_sha256": args.writer_sha256,
        "writer_start_ticks": writer_start_ticks,
    }
    _send_json_line(
        channel,
        {"body": ready_body, "hmac_sha256": _hmac_hex(nonce, ready_body)},
    )

    maximum_wait_deadline = (
        args.prearm_hard_deadline_monotonic_ns + MANUAL_TIMEOUT_NS
    )
    body: dict[str, Any] | None = None
    while body is None:
        remaining_s = (
            maximum_wait_deadline - time.monotonic_ns()
        ) / 1_000_000_000
        if remaining_s <= 0.0:
            raise TimeoutError("writer overall wait deadline expired")
        channel.settimeout(remaining_s)
        envelope = _recv_json_line(channel)
        _expect_exact_keys(
            envelope,
            {"body", "hmac_sha256"},
            "authenticated controller envelope",
        )
        candidate = envelope["body"]
        if not isinstance(candidate, dict):
            raise ProtocolError("authenticated body is not an object")
        supplied_hmac = envelope["hmac_sha256"]
        if not isinstance(supplied_hmac, str) or not hmac.compare_digest(
            supplied_hmac,
            _hmac_hex(nonce, candidate),
        ):
            raise ProtocolError("controller envelope HMAC mismatch")
        operation = candidate.get("op")
        if operation == "ping":
            _expect_exact_keys(
                candidate,
                {
                    "op",
                    "phase_event",
                    "phase_row_sha256",
                    "phase_sequence",
                },
                "ping body",
            )
            ping_rows = _read_phase_chain(
                root_fd,
                args.phase_dev,
                args.phase_ino,
            )
            latest_ping_row = ping_rows[-1]
            if (
                candidate["phase_event"] != latest_ping_row["event"]
                or candidate["phase_row_sha256"]
                != latest_ping_row["row_sha256"]
                or candidate["phase_sequence"] != latest_ping_row["sequence"]
            ):
                raise ProtocolError("ping phase binding mismatch")
            pong_body = {
                "op": "pong",
                "phase_event": latest_ping_row["event"],
                "phase_row_sha256": latest_ping_row["row_sha256"],
                "phase_sequence": latest_ping_row["sequence"],
                "writer_pid": os.getpid(),
                "writer_start_ticks": writer_start_ticks,
            }
            _send_json_line(
                channel,
                {
                    "body": pong_body,
                    "hmac_sha256": _hmac_hex(nonce, pong_body),
                },
            )
            continue
        if operation != "publish":
            raise ProtocolError(
                f"unsupported authenticated operation: {operation!r}"
            )
        body = candidate

    _expect_exact_keys(
        body,
        {
            "manual_deadline_monotonic_ns",
            "manual_input",
            "manual_prompt_started_monotonic_ns",
            "op",
            "phase_row_sha256",
            "phase_sequence",
            "screenshot_manifest_sha256",
            "writer_deadline_monotonic_ns",
        },
        "publish body",
    )
    prompt_started = body["manual_prompt_started_monotonic_ns"]
    manual_deadline = body["manual_deadline_monotonic_ns"]
    writer_deadline = body["writer_deadline_monotonic_ns"]
    if any(
        type(value) is not int
        for value in (prompt_started, manual_deadline, writer_deadline)
    ):
        raise ProtocolError("manual deadline fields are not exact integers")
    if prompt_started > args.prearm_hard_deadline_monotonic_ns:
        raise TimeoutError("manual prompt began after the pre-arm hard deadline")
    if manual_deadline != prompt_started + MANUAL_TIMEOUT_NS:
        raise ProtocolError("manual deadline is not prompt+600 seconds")
    if writer_deadline != manual_deadline - WRITER_DEADLINE_LEAD_NS:
        raise ProtocolError("writer deadline binding mismatch")
    if time.monotonic_ns() >= writer_deadline:
        raise TimeoutError("writer publication deadline expired")

    rows = _read_phase_chain(root_fd, args.phase_dev, args.phase_ino)
    latest = rows[-1]
    screenshot_sha = body["screenshot_manifest_sha256"]
    if not isinstance(screenshot_sha, str) or len(screenshot_sha) != 64:
        raise ProtocolError("screenshot manifest SHA is malformed")
    expected_prompt_details = {
        "manual_basename": MANUAL_BASENAME,
        "manual_deadline_monotonic_ns": manual_deadline,
        "manual_prompt_started_monotonic_ns": prompt_started,
        "new_controlled_physics_steps": 0,
        "screenshot_manifest_sha256": screenshot_sha,
        "writer_deadline_monotonic_ns": writer_deadline,
    }
    if (
        latest["event"] != "manual_prompt"
        or latest["details"] != expected_prompt_details
    ):
        raise ProtocolError("latest phase is not the exact manual_prompt")
    if (
        body["phase_sequence"] != latest["sequence"]
        or body["phase_row_sha256"] != latest["row_sha256"]
    ):
        raise ProtocolError("publish phase binding mismatch")
    screenshot_verification = _verify_screenshot_manifest(root_fd, screenshot_sha)

    fields, notes = _validate_manual_input(body["manual_input"])
    manual_pass = all(fields.values())
    document = {
        "artifact": "D408_MANUAL_VISUAL_INSPECTION_V1",
        "bindings": {
            "approved_tuple_sha256": args.approved_tuple_sha256,
            "controller_pid": args.controller_pid,
            "controller_sha256": args.controller_sha256,
            "controller_start_ticks": args.controller_start_ticks,
            "d407_source_manifest_sha256": args.input_manifest_sha256,
            "nonce_sha256": nonce_sha256,
            "phase_dev": args.phase_dev,
            "phase_ino": args.phase_ino,
            "phase_row_sha256": latest["row_sha256"],
            "phase_sequence": latest["sequence"],
            "preregistration_sha256": EXPECTED_PREREG_SHA256,
            "root_dev": args.root_dev,
            "root_ino": args.root_ino,
            "screenshot_manifest_sha256": screenshot_sha,
            "writer_pid": os.getpid(),
            "writer_sha256": args.writer_sha256,
            "writer_start_ticks": writer_start_ticks,
        },
        "deadline": {
            "manual_deadline_monotonic_ns": manual_deadline,
            "manual_prompt_started_monotonic_ns": prompt_started,
            "writer_deadline_monotonic_ns": writer_deadline,
        },
        "notes": notes,
        "pass": manual_pass,
        "received": True,
        "required_fields": fields,
        "source_science": {
            "d407_final_verdict": D407_FINAL_VERDICT,
            "d407_retroactive_pass": False,
            "g0a_pass": False,
            "new_controlled_physics_steps": 0,
            "scientific_null_claims": SCIENTIFIC_NULL_CLAIMS,
            "scientific_verdict": None,
        },
    }
    manual_sha, manual_size, completed_ns = _atomic_publish_manual(
        root_fd,
        document,
    )
    published_before_deadline = completed_ns < writer_deadline
    ack_body = {
        "manual_pass": manual_pass,
        "manual_sha256": manual_sha,
        "manual_size": manual_size,
        "op": "published_fsynced",
        "publication_fsync_completed_monotonic_ns": completed_ns,
        "published_before_writer_deadline": published_before_deadline,
        "screenshot_images_sha256": screenshot_verification["images_sha256"],
        "writer_pid": os.getpid(),
        "writer_start_ticks": writer_start_ticks,
    }
    channel.settimeout(5.0)
    _send_json_line(
        channel,
        {"body": ack_body, "hmac_sha256": _hmac_hex(nonce, ack_body)},
    )
    channel.close()
    return 0


def main() -> int:
    if not sys.dont_write_bytecode:
        raise RuntimeError("D408 writer must run with python -B")
    args = _parse_args()
    if args.print_contract_json:
        print(json.dumps(_contract(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.stage != "writer":
        raise ProtocolError("--stage writer is required")
    return _run_writer(args)


if __name__ == "__main__":
    raise SystemExit(main())
