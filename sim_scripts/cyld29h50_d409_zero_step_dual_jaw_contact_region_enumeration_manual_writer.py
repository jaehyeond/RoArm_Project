#!/usr/bin/env python3
"""D409 tuple-bound pre-armed manual inspection writer (attempt-level, once).

D408 precedent port (source:
sim_scripts/cyl34_top_view_d408_d407_manual_observability_completion_repair_manual_writer.py)
adapted to the D409 zero-step dual-jaw contact-region enumeration case.
Governing spec = claudedocs/session_20260803_grasp_g0a_d409_sweep_recovery_design_v1.md
SS2 (design v1) + SS4 (confirmed delta v2; SS4 overrides SS2 on conflict),
in particular SS4.2 W-OPS3 (field-level manual-writer requirement list) and
SS4.1 P4-7 (observability authored in a separate phase after run1/run2
bit-exact canonical promotion; manual inspection once per attempt).

This process never imports or launches Isaac/Kit/PhysX/Warp/CUDA/USD or any
robot HW/serial stack (an in-process import scope guard rejects the import
attempt itself).  It is spawned once by the controller BEFORE any runtime
enumeration work (pre-arm), and receives its secret nonce and the eventual
inspection booleans only through an inherited Unix socket.  Creating or
repairing this writer during a run is impossible by construction: the
pre-arm hard deadline, the controller_started phase binding, and the
writer/controller/worker/prereg/tuple SHA bindings are all validated before
the writer will publish anything.

Inspection subjects (attempt-level, exactly once, after canonical promotion):
  - d409_region_map_screenshot.png  (1920x1080 headless RRD decision shot)
  - d409_decision_sheet.png         (1920x1080 composed decision sheet)

The 11 required boolean fields are false-publishable: an all-false inspection
still publishes an honest manual result document (pass = AND of all fields).

SPEC AMBIGUITIES RESOLVED
  1. Prereg SHA pin timing: D408 hard-pinned EXPECTED_PREREG_SHA256 in the
     writer source, but the D409 harness is authored before the D409 prereg
     exists.  Resolution: the pin moves to the --preregistration-sha256 CLI
     binding, cross-checked against (i) the actual prereg file bytes and
     (ii) the user-approved tuple file's hashes.preregistration_sha256.
     Trust anchor = --approved-tuple-sha256 (user-approved tuple SHA), which
     must match the tuple file bytes; the tuple pins prereg + all three
     harness file SHAs.  No self-referential source edit is ever needed.
  2. 11-field schema substitution: the spec pins the D408 11-field structure
     substituted with D409 decision subjects (grid admission/region/anchor
     pose visibility, no overlap, numeric axes) but does not letter-pin the
     field names.  Resolution: the 11 names in REQUIRED_BOOLEAN_FIELDS,
     preserving the D408 count and false-publishable semantics.
  3. Screenshot set: D408 inspected 5 images across two legs; D409 is a
     single-leg, single-attempt case.  Resolution: exactly 2 images (region
     map RRD screenshot + decision sheet), both pinned 1920x1080 per SS2.12
     (D404~D408 repair pattern: "1920x1080 exact, ppp 2.0").  The D408
     1120x900 clean-spatial and 3840x1080 A/B sheet have no D409 analog.
  4. Attempt-root artifact basenames (d409_preregistration.json,
     d409_proposed_runtime_hash_tuple.json, d409_controller_phase_markers.jsonl,
     d409_screenshot_manifest.json, d409_manual_visual_inspection.json,
     d409_region_map.rrd / _blueprint.rbl / d409_rerun_validation.json and
     the two PNGs) are not letter-pinned in the spec.  Resolution: D371/D408
     lineage naming, all at the attempt root
     claudedocs/runtime_logs/grasp_track/g0a_d409/attempt1_zero_step_dual_jaw_contact_region_enumeration/.
     run1/ and run2/ are worker-owned; this writer never opens them.
  5. source_science block: D408 embedded the frozen D407 replay verdict; the
     D409 manual inspection precedes any D409 scientific verdict.
     Resolution: source_science carries the invariant prior state (D407
     FAIL-STOP unchanged, g0a_pass=false, new_controlled_physics_steps=0),
     the SS2.14 null claims, and the P3/W-FRZ1 interpretation-boundary
     statements; scientific_verdict is fixed None.  Manual pass is a
     presentation-layer (observability) verdict only and never overrides or
     asserts any scientific gate.
  6. PNG verification dependency: the D408 writer used PIL, which is not in
     the D409 allowed-import list (frozen JSON read + hppfcl/numpy/trimesh +
     rerun-sdk + rerun CLI subprocess).  Resolution: stdlib-only PNG
     verification (8-byte signature + IHDR length/type + big-endian
     width/height + exact IEND trailer), preserving format and dimension
     checks without any third-party import.  DISCLOSED REDUCTION vs D408
     (review writer-6/R18): no IDAT decode and no chunk-CRC validation —
     a mid-file-corrupted PNG with intact signature/IHDR/IEND passes this
     writer.  Compensating controls: byte-exact sha binding, the
     controller's full decode + non-blank gate on the screenshot, and the
     human inspecting the actual files.
  7. Prereg cross-checks: beyond the byte SHA, the writer verifies
     case == "g0a_d409", the approval boundary flag
     actual_execution_requires_separate_tuple_sha_approval is True, and the
     determinism_run_contract pins taken verbatim from SS4.1 P4-3
     (worker_invocations_total 2, automatic_retries 0,
     per_run_out_dirs ["run1","run2"], run2_precondition
     "run1_preclose_pass").  Field names are the P4-3 literals.
  8. Import scope guard: the interface pin requires the import attempt
     itself to be detected and rejected; the D408 writer only declared
     forbidden prefixes contractually.  Resolution: a sys.meta_path finder
     installed at module import time rejects forbidden root modules.
  9. Canonical-output binding time: at arm time run1/run2 canonical outputs
     do not exist, so canonical_evidence_sha256 / region_map_csv_sha256 are
     bound at publish time through the screenshot manifest (whose SHA is
     bound into the manual_prompt phase row and the publish body), and are
     copied into the manual document bindings for direct audit.  The
     manifest must assert determinism_bitexact_pass true and
     rrd_verify_pass true (P4-7: manual phase exists only after bit-exact
     promotion; W-LES2/D341: footer-verified RRD).
 10. worker_sha256 identity: the D409 harness is 3 files (W-OPS5); the D408
     tuple bound only prereg+controller+writer.  Resolution: the tuple must
     also carry worker_sha256 and the writer re-hashes the worker source
     file against it (and against the --worker-sha256 CLI binding).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import hmac
import json
import os
import socket
import stat
import sys
import time
from pathlib import Path
from typing import Any


FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "carb",
        "cuda",
        "cupy",
        "isaac",
        "isaacgym",
        "isaaclab",
        "isaacsim",
        "kit",
        "lerobot",
        "omni",
        "physx",
        "physxcooking",
        "pxr",
        "torch",
        "pyk4a",
        "pycuda",
        "roarm_sdk",
        "serial",
        "usd",
        "usdrt",
        "warp",
    }
)


class _ScopeGuardFinder:
    """Rejects forbidden runtime imports at the import attempt itself."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None):
        root = fullname.split(".", 1)[0].lower()
        if root in FORBIDDEN_IMPORT_ROOTS:
            raise ImportError(f"D409 manual-writer scope guard forbids import: {fullname}")
        return None


sys.meta_path.insert(0, _ScopeGuardFinder())


PROJECT_ROOT = Path(__file__).resolve().parents[1]
D409_ROOT = (
    PROJECT_ROOT
    / "claudedocs/runtime_logs/grasp_track/g0a_d409"
    / "attempt1_zero_step_dual_jaw_contact_region_enumeration"
)
ATTEMPT_RELATIVE_PREFIX = str(D409_ROOT.relative_to(PROJECT_ROOT))
PREREG_PATH = D409_ROOT / "d409_preregistration.json"
TUPLE_PATH = D409_ROOT / "d409_proposed_runtime_hash_tuple.json"
PHASE_PATH = D409_ROOT / "d409_controller_phase_markers.jsonl"
SCREENSHOT_MANIFEST_PATH = D409_ROOT / "d409_screenshot_manifest.json"
MANUAL_BASENAME = "d409_manual_visual_inspection.json"
MANUAL_PENDING_BASENAME = ".d409_manual_visual_inspection.json.pending"
MANUAL_PATH = D409_ROOT / MANUAL_BASENAME
CONTROLLER_PATH = (
    PROJECT_ROOT
    / "sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_controller.py"
)
WORKER_PATH = (
    PROJECT_ROOT
    / "sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_worker.py"
)
WRITER_PATH = Path(__file__).resolve()

MAX_PROTOCOL_BYTES = 64 * 1024
MAX_MANUAL_BYTES = 64 * 1024
MAX_SCREENSHOT_BYTES = 64 * 1024 * 1024
MAX_SCREENSHOT_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_PREREG_BYTES = 4 * 1024 * 1024
MAX_TUPLE_BYTES = 1024 * 1024
MAX_PHASE_BYTES = 1024 * 1024
MAX_NOTES_UTF8_BYTES = 4096
MANUAL_TIMEOUT_NS = 600_000_000_000
WRITER_DEADLINE_LEAD_NS = 5_000_000_000
RENAME_NOREPLACE = 1

PHASE_ROW_ARTIFACT = "D409_CONTROLLER_PHASE_ROW_V1"
SCREENSHOT_MANIFEST_ARTIFACT = "D409_SCREENSHOT_MANIFEST_V1"
MANUAL_ARTIFACT = "D409_MANUAL_VISUAL_INSPECTION_V1"
CONTRACT_ARTIFACT = "D409_MANUAL_WRITER_CONTRACT_V1"

D407_FINAL_VERDICT = "D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP"
SCIENTIFIC_NULL_CLAIMS = {
    "cap_rim_barrel_dynamic_contact_order": None,
    "force_closure": None,
    "grasp_feasibility": None,
    "grasp_success": None,
    "other_cylinder_or_placement_transfer": None,
    "pushover_absence_guarantee": None,
    "sdf_general_superiority": None,
    "stable_grasp": None,
}
INTERPRETATION_BOUNDARY = {
    "a_and_b_excludes_d362_pushover_pose": False,
    "geometry_only_label_training_promotion_allowed": False,
    "part_level_mask_resolves_face_level_inner_outer": False,
    "zero_step_replaces_closure_dynamics": False,
}
SOURCE_SCIENCE = {
    "d407_final_verdict": D407_FINAL_VERDICT,
    "d409_scope": "contact_region_map_and_ordering_constraint_scoring_only",
    "g0a_pass": False,
    "interpretation_boundary": INTERPRETATION_BOUNDARY,
    "new_controlled_physics_steps": 0,
    "scientific_null_claims": SCIENTIFIC_NULL_CLAIMS,
    "scientific_verdict": None,
}

REGION_MAP_SCREENSHOT_BASENAME = "d409_region_map_screenshot.png"
DECISION_SHEET_BASENAME = "d409_decision_sheet.png"
EXPECTED_MANUAL_IMAGE_LAYOUT = {
    REGION_MAP_SCREENSHOT_BASENAME: [1920, 1080],
    DECISION_SHEET_BASENAME: [1920, 1080],
}
EXPECTED_MANUAL_IMAGE_ROLES = {
    REGION_MAP_SCREENSHOT_BASENAME: "region_map_screenshot",
    DECISION_SHEET_BASENAME: "decision_sheet",
}
EXPECTED_RRD_REPORT_PATHS = {
    "rbl_path": f"{ATTEMPT_RELATIVE_PREFIX}/d409_region_map_blueprint.rbl",
    "rrd_path": f"{ATTEMPT_RELATIVE_PREFIX}/d409_region_map.rrd",
    "validation_path": f"{ATTEMPT_RELATIVE_PREFIX}/d409_rerun_validation.json",
}

REQUIRED_BOOLEAN_FIELDS = (
    "grid_all_cells_admission_layer_visible",
    "region_components_distinguishable",
    "region_representative_cells_marked",
    "anchor_pose_both_jaws_visible",
    "cylinder_d29h50_visible",
    "q5_star_or_witness_markers_visible",
    "decision_sheet_numeric_axes_legible",
    "decision_sheet_scores_and_flags_legible",
    "no_notification_or_text_overlap",
    "no_error_banner_visible",
    "region_map_and_decision_sheet_consistent",
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND_TRAILER = b"\x00\x00\x00\x00IEND\xaeB`\x82"
SHA256_HEX_ALPHABET = frozenset("0123456789abcdef")


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


def _require_sha256_hex(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value) <= SHA256_HEX_ALPHABET
    ):
        raise ProtocolError(f"{label} is not a lowercase sha256 hex digest")
    return value


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
        raise ProtocolError(f"unsafe D409-root-relative path: {relative_path!r}")
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
    path_metadata = os.lstat(D409_ROOT)
    expected = (args.root_dev, args.root_ino)
    if args.root != str(D409_ROOT):
        raise ProtocolError("D409 root argument mismatch")
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(path_metadata.st_mode):
        raise ProtocolError("D409 root is not a bound real directory")
    if (metadata.st_dev, metadata.st_ino) != expected or (
        path_metadata.st_dev,
        path_metadata.st_ino,
    ) != expected:
        raise ProtocolError("D409 root path/dirfd identity mismatch")
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
        if row["artifact"] != PHASE_ROW_ARTIFACT:
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
    if args.worker_sha256 != _sha_path(WORKER_PATH):
        raise ProtocolError("worker SHA binding mismatch")
    if _proc_start_ticks(args.controller_pid) != args.controller_start_ticks:
        raise ProtocolError("controller PID/start-time binding mismatch")
    _require_sha256_hex(args.preregistration_sha256, "preregistration SHA argument")
    _require_sha256_hex(args.approved_tuple_sha256, "approved tuple SHA argument")
    prereg_raw, _ = _secure_read_relative(
        root_fd,
        PREREG_PATH.name,
        MAX_PREREG_BYTES,
    )
    if _sha_bytes(prereg_raw) != args.preregistration_sha256:
        raise ProtocolError("preregistration SHA mismatch")
    tuple_raw, _ = _secure_read_relative(
        root_fd,
        TUPLE_PATH.name,
        MAX_TUPLE_BYTES,
    )
    if _sha_bytes(tuple_raw) != args.approved_tuple_sha256:
        raise ProtocolError("approved tuple-file SHA mismatch")

    prereg = _strict_json_bytes(prereg_raw)
    if not isinstance(prereg, dict):
        raise ProtocolError("preregistration is not an object")
    if prereg.get("case") != "g0a_d409":
        raise ProtocolError("preregistration case binding mismatch")
    if prereg.get("actual_execution_requires_separate_tuple_sha_approval") is not True:
        raise ProtocolError("prereg approval boundary is not exact")
    run_contract = prereg.get("determinism_run_contract")
    if not isinstance(run_contract, dict):
        raise ProtocolError("prereg determinism_run_contract is missing")
    if (
        run_contract.get("worker_invocations_total") != 2
        or run_contract.get("automatic_retries") != 0
        or run_contract.get("per_run_out_dirs") != ["run1", "run2"]
        or run_contract.get("run2_precondition") != "run1_preclose_pass"
    ):
        raise ProtocolError("prereg determinism_run_contract pins mismatch")

    tuple_data = _strict_json_bytes(tuple_raw)
    if not isinstance(tuple_data, dict):
        raise ProtocolError("tuple is not an object")
    hashes = tuple_data.get("hashes")
    if not isinstance(hashes, dict):
        raise ProtocolError("tuple hashes are missing")
    expected_hashes = {
        "controller_sha256": args.controller_sha256,
        "manual_writer_sha256": args.writer_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "worker_sha256": args.worker_sha256,
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


def _png_dimensions(raw: bytes, label: str) -> list[int]:
    if len(raw) < 45 or not raw.startswith(PNG_SIGNATURE):
        raise ProtocolError(f"manual image is not PNG: {label}")
    if raw[8:12] != b"\x00\x00\x00\x0d" or raw[12:16] != b"IHDR":
        raise ProtocolError(f"PNG IHDR chunk is not exact: {label}")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width <= 0 or height <= 0:
        raise ProtocolError(f"PNG dimensions are invalid: {label}")
    if raw[-12:] != PNG_IEND_TRAILER:
        raise ProtocolError(f"PNG IEND trailer is not exact: {label}")
    return [width, height]


def _verify_screenshot_manifest(
    root_fd: int,
    expected_sha256: str,
) -> dict[str, Any]:
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
            "artifact",
            "canonical_evidence_sha256",
            "determinism_bitexact_pass",
            "images",
            "manual_target_count",
            "new_controlled_physics_steps",
            "region_map_csv_sha256",
            "rrd_report",
        },
        "screenshot manifest",
    )
    if (
        manifest["artifact"] != SCREENSHOT_MANIFEST_ARTIFACT
        or manifest["manual_target_count"] != 2
        or manifest["new_controlled_physics_steps"] != 0
    ):
        raise ProtocolError("screenshot manifest invariant mismatch")
    if manifest["determinism_bitexact_pass"] is not True:
        raise ProtocolError(
            "manual inspection requires run1/run2 bit-exact canonical promotion"
        )
    canonical_evidence_sha256 = _require_sha256_hex(
        manifest["canonical_evidence_sha256"],
        "manifest canonical_evidence_sha256",
    )
    region_map_csv_sha256 = _require_sha256_hex(
        manifest["region_map_csv_sha256"],
        "manifest region_map_csv_sha256",
    )
    rrd_report = manifest["rrd_report"]
    if not isinstance(rrd_report, dict):
        raise ProtocolError("screenshot manifest rrd_report is not an object")
    _expect_exact_keys(
        rrd_report,
        {
            "rbl_path",
            "rbl_sha256",
            "rrd_path",
            "rrd_sha256",
            "rrd_verify_pass",
            "validation_path",
            "validation_sha256",
        },
        "rrd_report",
    )
    if rrd_report["rrd_verify_pass"] is not True:
        raise ProtocolError("rrd_report rrd_verify_pass is not exactly true")
    if any(
        rrd_report[key] != value for key, value in EXPECTED_RRD_REPORT_PATHS.items()
    ):
        raise ProtocolError("rrd_report path binding mismatch")
    for key in ("rbl_sha256", "rrd_sha256", "validation_sha256"):
        _require_sha256_hex(rrd_report[key], f"rrd_report {key}")
    # WOBS-W2 repair R15: re-read and re-hash the three rrd_report files
    # byte-exactly (previously only format-checked) so stale or wrong SHAs
    # recorded by the controller cannot be published unchallenged.
    for path_key, sha_key in (
        ("rbl_path", "rbl_sha256"),
        ("rrd_path", "rrd_sha256"),
        ("validation_path", "validation_sha256"),
    ):
        report_relative = rrd_report[path_key]
        if not report_relative.startswith(f"{ATTEMPT_RELATIVE_PREFIX}/"):
            raise ProtocolError(f"rrd_report path outside attempt root: {report_relative}")
        report_raw, _report_metadata = _secure_read_relative(
            root_fd,
            report_relative[len(ATTEMPT_RELATIVE_PREFIX) + 1 :],
            MAX_SCREENSHOT_BYTES,
        )
        if _sha_bytes(report_raw) != rrd_report[sha_key]:
            raise ProtocolError(f"rrd_report {sha_key} does not match file bytes")
    images = manifest["images"]
    if not isinstance(images, list) or len(images) != 2:
        raise ProtocolError("screenshot manifest image count mismatch")
    verified_images: list[dict[str, Any]] = []
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
        if item["manual_role"] != EXPECTED_MANUAL_IMAGE_ROLES[relative_path]:
            raise ProtocolError(f"screenshot manual_role mismatch: {relative_path}")
        expected_project_path = f"{ATTEMPT_RELATIVE_PREFIX}/{relative_path}"
        if item["path"] != expected_project_path:
            raise ProtocolError("screenshot path binding mismatch")
        raw, metadata = _secure_read_relative(
            root_fd,
            relative_path,
            MAX_SCREENSHOT_BYTES,
        )
        dimensions = _png_dimensions(raw, relative_path)
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
                "manual_role": item["manual_role"],
                "root_relative_path": relative_path,
                "sha256": image_sha,
            }
        )
    if sorted(observed_paths) != sorted(EXPECTED_MANUAL_IMAGE_LAYOUT):
        raise ProtocolError("manual screenshot path set mismatch")
    return {
        "canonical_evidence_sha256": canonical_evidence_sha256,
        "image_count": len(verified_images),
        "images_sha256": _sha_bytes(_canonical_bytes(verified_images)),
        "manifest_bytes": manifest_metadata.st_size,
        "manifest_sha256": expected_sha256,
        "region_map_csv_sha256": region_map_csv_sha256,
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
        "artifact": CONTRACT_ARTIFACT,
        "expected_paths": {
            "controller": str(CONTROLLER_PATH.relative_to(PROJECT_ROOT)),
            "manual_output": str(MANUAL_PATH.relative_to(PROJECT_ROOT)),
            "phase_log": str(PHASE_PATH.relative_to(PROJECT_ROOT)),
            "preregistration": str(PREREG_PATH.relative_to(PROJECT_ROOT)),
            "root": ATTEMPT_RELATIVE_PREFIX,
            "screenshot_manifest": str(
                SCREENSHOT_MANIFEST_PATH.relative_to(PROJECT_ROOT)
            ),
            "tuple": str(TUPLE_PATH.relative_to(PROJECT_ROOT)),
            "worker": str(WORKER_PATH.relative_to(PROJECT_ROOT)),
            "writer": "sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_manual_writer.py",
        },
        "forbidden_import_roots": sorted(FORBIDDEN_IMPORT_ROOTS),
        "manual_basename": MANUAL_BASENAME,
        "manual_output_path": str(MANUAL_PATH.relative_to(PROJECT_ROOT)),
        "manual_pending_basename": MANUAL_PENDING_BASENAME,
        "manual_timeout_ns": MANUAL_TIMEOUT_NS,
        "prereg_static_checks": {
            "actual_execution_requires_separate_tuple_sha_approval": True,
            "case": "g0a_d409",
            "determinism_run_contract": {
                "automatic_retries": 0,
                "per_run_out_dirs": ["run1", "run2"],
                "run2_precondition": "run1_preclose_pass",
                "worker_invocations_total": 2,
            },
        },
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
                    "manual_basename",
                    "phase_dev",
                    "phase_ino",
                    "prearm_hard_deadline_monotonic_ns",
                    "preregistration_sha256",
                    "root_dev",
                    "root_ino",
                    "worker_sha256",
                    "writer_sha256",
                }
            ),
            "arm_message_keys": ["bindings", "nonce_hex", "op"],
            "manual_binding_keys": sorted(
                {
                    "approved_tuple_sha256",
                    "canonical_evidence_sha256",
                    "controller_pid",
                    "controller_sha256",
                    "controller_start_ticks",
                    "nonce_sha256",
                    "phase_dev",
                    "phase_ino",
                    "phase_row_sha256",
                    "phase_sequence",
                    "preregistration_sha256",
                    "region_map_csv_sha256",
                    "root_dev",
                    "root_ino",
                    "screenshot_manifest_sha256",
                    "worker_sha256",
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
                "preregistration": MAX_PREREG_BYTES,
                "protocol_message": MAX_PROTOCOL_BYTES,
                "screenshot_image": MAX_SCREENSHOT_BYTES,
                "screenshot_manifest": MAX_SCREENSHOT_MANIFEST_BYTES,
                "tuple": MAX_TUPLE_BYTES,
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
            "screenshot_image_roles": dict(EXPECTED_MANUAL_IMAGE_ROLES),
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
            "screenshot_manifest_keys": sorted(
                {
                    "artifact",
                    "canonical_evidence_sha256",
                    "determinism_bitexact_pass",
                    "images",
                    "manual_target_count",
                    "new_controlled_physics_steps",
                    "region_map_csv_sha256",
                    "rrd_report",
                }
            ),
            "screenshot_rrd_report_keys": sorted(
                {
                    "rbl_path",
                    "rbl_sha256",
                    "rrd_path",
                    "rrd_sha256",
                    "rrd_verify_pass",
                    "validation_path",
                    "validation_sha256",
                }
            ),
            "screenshot_rrd_report_paths": dict(EXPECTED_RRD_REPORT_PATHS),
            "source_science_keys": sorted(SOURCE_SCIENCE),
            "source_science": SOURCE_SCIENCE,
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
        "screenshot_layout": {
            key: list(value) for key, value in EXPECTED_MANUAL_IMAGE_LAYOUT.items()
        },
        "tuple_hash_keys": sorted(
            {
                "controller_sha256",
                "manual_writer_sha256",
                "preregistration_sha256",
                "worker_sha256",
            }
        ),
        "w_ops3_reject_surfaces": {
            "common_monotonic_deadline": (
                "manual_deadline==prompt+600s; writer_deadline==manual-5s; "
                "stale pre-arm hard deadline and expired writer deadline both raise"
            ),
            "controller_writer_identity": (
                "ppid==controller_pid; controller/worker/writer source SHA re-hash; "
                "controller /proc start-ticks binding"
            ),
            "eleven_field_false_publishable_schema": (
                "exact 11 boolean fields; pass=AND; all-false still publishes"
            ),
            "exclusive_create_no_replace_fsync": (
                "O_EXCL pending file; renameat2 RENAME_NOREPLACE; file+dir fsync; "
                "byte-exact readback; pre-existing manual/pending file raises"
            ),
            "no_writer_creation_or_repair_during_run": (
                "arm only at controller_started phase row; pre-arm hard deadline "
                "must be in the future; writer source SHA bound into tuple"
            ),
            "nonce_hmac_envelope": (
                "256-bit nonce via inherited socket; HMAC-SHA256 over canonical "
                "body; constant-time compare; unauthenticated ops rejected"
            ),
            "source_screenshot_manifest": (
                "manifest SHA + exact schema; PNG signature/IHDR/IEND + dims + "
                "bytes + sha per image; canonical evidence and region CSV SHA "
                "binding; determinism_bitexact_pass and rrd_verify_pass exact true"
            ),
            "tuple_sha_binding": (
                "tuple file bytes must hash to the user-approved SHA; tuple pins "
                "prereg/controller/worker/manual_writer SHAs"
            ),
            "worst_case_traversal_budget": (
                "every read bounded by an explicit byte limit; O_NOFOLLOW "
                "openat-only traversal; nlink==1 regular-file identity; "
                "identity re-stat after read"
            ),
        },
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
    parser.add_argument("--worker-sha256")
    parser.add_argument("--writer-sha256")
    parser.add_argument("--approved-tuple-sha256")
    parser.add_argument("--preregistration-sha256")
    parser.add_argument("--phase-dev", type=int)
    parser.add_argument("--phase-ino", type=int)
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
        "worker_sha256",
        "writer_sha256",
        "approved_tuple_sha256",
        "preregistration_sha256",
        "phase_dev",
        "phase_ino",
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
        raise FileExistsError(D409_ROOT / basename)

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
        "manual_basename": args.manual_basename,
        "phase_dev": args.phase_dev,
        "phase_ino": args.phase_ino,
        "prearm_hard_deadline_monotonic_ns": (
            args.prearm_hard_deadline_monotonic_ns
        ),
        "preregistration_sha256": args.preregistration_sha256,
        "root_dev": args.root_dev,
        "root_ino": args.root_ino,
        "worker_sha256": args.worker_sha256,
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
        "prearm_hard_deadline_monotonic_ns": (
            args.prearm_hard_deadline_monotonic_ns
        ),
        "preregistration_sha256": args.preregistration_sha256,
        "root_dev": args.root_dev,
        "root_ino": args.root_ino,
        "worker_sha256": args.worker_sha256,
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
    screenshot_sha = _require_sha256_hex(
        body["screenshot_manifest_sha256"],
        "publish screenshot manifest SHA",
    )
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
        "artifact": MANUAL_ARTIFACT,
        "bindings": {
            "approved_tuple_sha256": args.approved_tuple_sha256,
            "canonical_evidence_sha256": (
                screenshot_verification["canonical_evidence_sha256"]
            ),
            "controller_pid": args.controller_pid,
            "controller_sha256": args.controller_sha256,
            "controller_start_ticks": args.controller_start_ticks,
            "nonce_sha256": nonce_sha256,
            "phase_dev": args.phase_dev,
            "phase_ino": args.phase_ino,
            "phase_row_sha256": latest["row_sha256"],
            "phase_sequence": latest["sequence"],
            "preregistration_sha256": args.preregistration_sha256,
            "region_map_csv_sha256": (
                screenshot_verification["region_map_csv_sha256"]
            ),
            "root_dev": args.root_dev,
            "root_ino": args.root_ino,
            "screenshot_manifest_sha256": screenshot_sha,
            "worker_sha256": args.worker_sha256,
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
        "source_science": SOURCE_SCIENCE,
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
        raise RuntimeError("D409 writer must run with python -B")
    args = _parse_args()
    if args.print_contract_json:
        print(json.dumps(_contract(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.stage != "writer":
        raise ProtocolError("--stage writer is required")
    return _run_writer(args)


if __name__ == "__main__":
    raise SystemExit(main())
