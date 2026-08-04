#!/usr/bin/env python3
"""D409 zero-step dual-jaw contact-region enumeration — CONTROLLER (cyld29h50).

Supervisor of the 3-file D409 harness (W-OPS3: controller / worker /
manual_writer).  Confirmed spec = claudedocs/
session_20260803_grasp_g0a_d409_sweep_recovery_design_v1.md section 2
(design v1) as amended by section 4 (confirmed delta v2; section 4 wins on
conflict).  Offline only: no Isaac, no kit, no physx, no warp, no cuda/gpu,
no AppLauncher, no cook, no USD, no robot HW/serial — a scope guard rejects
the import attempt itself.  Allowed: frozen JSON reads, hppfcl/numpy
offline queries, rerun-sdk (save-only RRD authoring), the absolute-path
rerun CLI subprocess (verify / RBL / screenshot only), and writes under
claudedocs/runtime_logs/grasp_track/g0a_d409/.

MODES
  --mode static-prep
      (1) derives the worker/writer interfaces from their SOURCE files
          (AST constants + --print-contract-json; D405/D406/D407 lesson:
          consumer-derived, never hand-authored),
      (2) builds and exclusively creates d409_preregistration.json
          (D371 lineage fields + D407 lineage allowed_dirty derivation +
          determinism_run_contract/determinism_manifest (P4) + registered
          budget citing d409_static_prep_s1s2s3.json + anchor-gate
          ANY-reject pin + the full section 4.3 delta),
      (3) runs the static fixtures: prepare-time negative controls 1-5
          (section 2.11 confirmed), audit_registered negative controls 1-4,
          positive controls ((7,11) anchor reproduction + grid key), and
          the W-LES4 equivalence fixture (d335 stored-row replay),
      (4) writes d409_static_fixture_results.json,
          d409_reviewed_script_attestation.json and
          d409_proposed_runtime_hash_tuple.json (exclusive create — the
          D407 contract forbids candidate overwrite), prints the tuple
          sha256 and STOPS.  Runtime is a separate user approval.
  --mode runtime --approved-tuple-sha256 <sha>
      Admission requires the tuple file to exist and hash to the cited
      user-approved sha, the tuple to pin the current prereg/controller/
      worker/manual_writer bytes, nlink==1 on every bound file, git dirty
      set within the preregistered allowlist, and NO pre-existing runtime
      artifact (crash resume is fail-closed).  Flow: phase log
      (controller_started) -> manual-writer PRE-ARM (before any
      enumeration; D407-R1/D408) -> pre-run inventory -> run1 -> preclose
      verification -> run2 (only after run1 preclose PASS) -> byte compare
      of the determinism manifest members -> canonical promotion record +
      verdict sha256 publication (W-LES3 measurement-before-presentation)
      -> observability phase (RRD with /enum/grid full 1,239-cell layer,
      embedded blueprint + RBL export, rerun CLI footer verify, headless
      1920x1080 screenshot at ppp 2.0, self-composed decision sheet) ->
      screenshot manifest -> manual_prompt phase row -> stdin manual input
      -> authenticated publish through the pre-armed writer -> completion
      summary with the P4-6 audit checks (per-run invocations == 1, total
      == 2, pre-run inventory reflecting the run1/run2 structure).

INTERFACES CONSUMED (read from the sibling files, not re-authored)
  worker  : CLI --out-dir/--prereg; exit 0 pass / 73 claim-preexist /
            other nonzero fail; outputs d409_worker_claim.json,
            d409_enumeration_evidence.json (authority),
            d409_region_map.csv, d409_worker_summary.json,
            d409_worker_preclose_sentinel.json,
            d409_worker_phase_markers.jsonl; prereg keys
            worker_admission_rows (exactly 2 rows) + input_hashes
            cross-check; preclose status literal derived from the worker
            source (see AMBIGUITY C2).
  writer  : --print-contract-json contract (arm/ready/ping/publish
            envelope schemas, 11 boolean fields, screenshot manifest
            schema, prereg static checks, deadlines) + AST-derived
            artifact literals; spawned once, pre-armed, socketpair +
            inherited root dirfd, HMAC-SHA256 envelopes.

SPEC AMBIGUITIES RESOLVED (auditable decisions; spec did not fix these)
  C1.  worker_admission_rows content: the worker pins only the row schema
       ({row_id, path, sha256} x2).  Chosen rows = (a) the worker source
       file itself (code-identity admission: the running worker re-hashes
       the bytes the prereg registered) and (b)
       g0a_d409/design_inputs/d409_static_prep_s1s2s3.json (the registered
       -budget basis file, W-OPS4).
  C2.  "status literals from worker-source ast.Eq" (D405/D407): the
       worker's preclose status literal PRECLOSE_PASS appears only as a
       dict-literal value for key "status" (no ast.Eq node); derivation is
       therefore extended to ast.Compare(Eq|NotEq) string comparators PLUS
       dict-literal "status" values.  The dict-literal path is enforced by
       uniqueness (two or more distinct literals raise) and existence — it
       has NO imported-module-constant cross-check because PRECLOSE_PASS
       exists only inline in the sentinel dict (C2 wording corrected per
       review R17).  No consumption literal in this controller is
       hand-authored.
  C3.  Negative control 2 (section 4.1 P1 re-anchor "d348 payload_sha256
       FAIL"): the stored payload_sha256 preimage is opaque (recorded
       verbatim by S1; not recomputable from vertices/triangles — probed).
       Executable control = (a) 1-bit vertex-stream tamper -> D409-
       canonical per-part geometry hash mismatch vs the S1 pin (the
       registered runtime re-verification basis, section 5.1 S1) and (b)
       1-byte file tamper -> d348 file-level sha256 FAIL vs the worker
       pin.  This matches P1's substituted static checks (i)/(ii).
  C4.  Negative control 5 "empty grid -> admission FAIL" maps to the
       frozen worker's static-pin gate (_verify_static_pins raises when
       the module grid is emptied): the worker builds its grid from module
       constants, so an empty grid is a worker-contract admission failure,
       not a per-pose row outcome.
  C5.  Equivalence fixture (W-LES4): the d335 CSV does not export the
       live-written alignment scalar, so live_exact_written_tcp_error_le_
       5mm is not reconstructible; the replay covers the remaining 7
       legacy checks + the raw clear gate, compares check-by-check, and
       also compares the reconstructed AND against the stored
       legacy_alignment_pass / raw_tool_clear_pass columns (mismatch =
       fixture FAIL, fail-closed).  Tolerance pins: numeric recompute
       1e-9 mm, anti-retreat epsilon 1e-12 (d335 literal), booleans exact.
  C6.  Decision sheet composition: D408 used PIL; PIL is outside the D409
       allowed-import set (frozen JSON + hppfcl/numpy/trimesh + rerun-sdk
       + rerun CLI).  The decision sheet is composed with numpy + stdlib
       zlib/struct PNG encoding and an embedded 5x7 bitmap font, 1920x1080
       exact.  The region-map screenshot remains a rerun CLI headless
       screenshot (D404~D408 repair pattern; logical 960x540 at ppp 2.0
       -> physical 1920x1080).
  C7.  "error banner 0": no spec-pinned automated detector exists; the
       controller verifies decodability + non-blank content of the
       screenshot and delegates the authoritative no_error_banner_visible
       judgment to the manual writer's 11-field schema (as in D408).
  C8.  The writer requires prereg AND tuple to exist at arm time
       (_validate_static_bindings), so the runtime mode hard-requires the
       full static-prep artifact set plus the user-cited tuple sha before
       the writer is spawned.  This is the concrete form of "admission
       requires tuple existence + sha match".
  C9.  RRD leaf naming under the pinned section 2.12 roots:
       /enum/instance/rep_<rho>_<tau>/{link5,gripper_link} for region
       representatives, /enum/instance/anchor_frozen/... for the D349
       frozen anchor pose, /enum/candidate/witness/<tag> and
       /enum/candidate/region_boundary.  The full 1,239-cell layers are
       /enum/grid/admission and /enum/grid/full_pass (W-LES2); the
       representative subset is capped at RRD_MAX_REPRESENTATIVES for the
       detail views only — the evidence JSON remains the authority.
  C10. Tuple file schema: the writer requires hashes{controller_sha256,
       manual_writer_sha256, preregistration_sha256, worker_sha256}; the
       D407 lineage fields are adapted around that (artifact, case,
       created_utc, approval_boundary, attestation_sha256, hashes).  The
       user approves the FILE's sha256.
  C11. Pre-arm hard deadline budget: the spec pins only the per-run
       timeout (7,200 s).  Chosen prearm budget = 2 x 7,200 s (runs) +
       1,800 s (observability) + 900 s (admission/manifest margin) =
       17,100 s (sum corrected per OPS-W1/R10; the formula is the
       authority); the writer's own overall wait adds +600 s.
  C12. Recording finalization (D341 "context exit or disconnect"):
       flush(timeout_sec=30.0) + RecordingStream.disconnect() (rerun-sdk
       0.34.1 signature is keyword-only timeout_sec — OPS-B2/WOBS-B1
       repair R3; D407 precedent form), then the RrdReader + rerun CLI
       footer verify gate runs on the closed file.
  C13. The controller-owned per-run artifacts (worker stdout/stderr logs,
       supervisor JSON) live inside run1//run2/ next to the worker-owned
       artifacts; the worker's fail-closed pre-existence check covers only
       its own six names, so these do not collide.  stdout and stderr are
       captured to SEPARATE files (their sha256s are recorded; both are
       excluded from the determinism byte compare by P4-4).

NULL CLAIMS (section 2.14 + P3 + W-FRZ1): stable grasp, force closure,
grasp feasibility, grasp success, push-over-absence guarantee,
contact-order dynamics, SDF superiority, transfer to other cylinders or
placements are all null.  A-and-B does NOT exclude the D362 push-over pose
(d_fix 4.2727 mm is inside the band); part-level masks cannot distinguish
inner/outer faces (outer 16 subset of inner 17, difference part_035
alone).  Geometry-only labels must not be promoted to standalone training.
g0a_pass=false unchanged; D399 remains reserved for D398-F1.  Completion
of this harness is an observability/enumeration outcome, never a grasp
verdict.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import hmac
import importlib.metadata
import importlib.util
import json
import math
import os
import secrets
import select
import socket
import stat
import struct
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Scope guard — installed before any third-party import (section 2.10 +
# W-OPS2: Isaac/kit/physx/warp/cuda/gpu/AppLauncher/cook/USD/HW/serial all
# zero; rerun-sdk import IS allowed for save-only RRD authoring).
# ---------------------------------------------------------------------------
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
        "pycuda",
        "pyk4a",
        "roarm_sdk",
        "serial",
        "torch",
        "usd",
        "usdrt",
        "warp",
    }
)
_SCOPE_GUARD_VIOLATIONS: list[str] = []


class _ScopeGuardFinder:
    """sys.meta_path finder that refuses forbidden module roots outright."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        root = fullname.split(".")[0].lower()
        if root in FORBIDDEN_IMPORT_ROOTS:
            _SCOPE_GUARD_VIOLATIONS.append(fullname)
            raise ImportError(
                f"D409 scope guard: forbidden import '{fullname}' (offline-only controller)"
            )
        return None


sys.meta_path.insert(0, _ScopeGuardFinder())

import numpy as np  # noqa: E402  (allowed; imported after guard install)

# ---------------------------------------------------------------------------
# Paths (forward-only folders; all outputs under g0a_d409/ — D322 rule).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = Path(__file__).resolve()
WORKER_PATH = (
    PROJECT_ROOT
    / "sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_worker.py"
)
WRITER_PATH = (
    PROJECT_ROOT
    / "sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_manual_writer.py"
)
SESSION_DOC_REL = "claudedocs/session_20260803_grasp_g0a_d409_sweep_recovery_design_v1.md"
CASE_ROOT = PROJECT_ROOT / "claudedocs/runtime_logs/grasp_track/g0a_d409"
ATTEMPT_ROOT = CASE_ROOT / "attempt1_zero_step_dual_jaw_contact_region_enumeration"
RUN_DIR_NAMES = ("run1", "run2")

PREREG_PATH = ATTEMPT_ROOT / "d409_preregistration.json"
STATIC_RESULTS_PATH = ATTEMPT_ROOT / "d409_static_fixture_results.json"
ATTESTATION_PATH = ATTEMPT_ROOT / "d409_reviewed_script_attestation.json"
TUPLE_PATH = ATTEMPT_ROOT / "d409_proposed_runtime_hash_tuple.json"
PHASE_PATH = ATTEMPT_ROOT / "d409_controller_phase_markers.jsonl"
PRERUN_INVENTORY_PATH = ATTEMPT_ROOT / "d409_prerun_inventory.json"
CANONICAL_PROMOTION_PATH = ATTEMPT_ROOT / "d409_canonical_promotion.json"
RRD_PATH = ATTEMPT_ROOT / "d409_region_map.rrd"
RBL_PATH = ATTEMPT_ROOT / "d409_region_map_blueprint.rbl"
RERUN_VALIDATION_PATH = ATTEMPT_ROOT / "d409_rerun_validation.json"
REGION_SCREENSHOT_PATH = ATTEMPT_ROOT / "d409_region_map_screenshot.png"
DECISION_SHEET_PATH = ATTEMPT_ROOT / "d409_decision_sheet.png"
SCREENSHOT_MANIFEST_PATH = ATTEMPT_ROOT / "d409_screenshot_manifest.json"
MANUAL_RECEIPT_PATH = ATTEMPT_ROOT / "d409_manual_writer_receipt.json"
COMPLETION_PATH = ATTEMPT_ROOT / "d409_completion_summary.json"
WRITER_STDOUT_LOG = ATTEMPT_ROOT / "d409_manual_writer_stdout.log"
WRITER_STDERR_LOG = ATTEMPT_ROOT / "d409_manual_writer_stderr.log"
WORKER_STDOUT_BASENAME = "d409_worker_stdout.log"
WORKER_STDERR_BASENAME = "d409_worker_stderr.log"
WORKER_SUPERVISOR_BASENAME = "d409_worker_supervisor.json"

STATIC_ARTIFACT_PATHS = (
    PREREG_PATH,
    STATIC_RESULTS_PATH,
    ATTESTATION_PATH,
    TUPLE_PATH,
)
RUNTIME_CONTROLLER_ATTEMPT_PATHS = (
    PHASE_PATH,
    PRERUN_INVENTORY_PATH,
    CANONICAL_PROMOTION_PATH,
    RRD_PATH,
    RBL_PATH,
    RERUN_VALIDATION_PATH,
    REGION_SCREENSHOT_PATH,
    DECISION_SHEET_PATH,
    SCREENSHOT_MANIFEST_PATH,
    MANUAL_RECEIPT_PATH,
    COMPLETION_PATH,
    WRITER_STDOUT_LOG,
    WRITER_STDERR_LOG,
)
RUN_CONTROLLER_BASENAMES = (
    WORKER_STDOUT_BASENAME,
    WORKER_STDERR_BASENAME,
    WORKER_SUPERVISOR_BASENAME,
)

# Frozen inputs and design-input pins (section 2.3 confirmed + section 4.2).
D348_REL = "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_callback_topology_volume_evidence.json"
D368_REL = "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_evidence.json"
D349_REL = "claudedocs/runtime_logs/grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json"
URDF_REL = "local_assets/roarm_m3/urdf/roarm_m3.urdf"
D371_EVIDENCE_REL = "claudedocs/runtime_logs/grasp_track/g0a_d371/d371_offline_collider_comparison_evidence.json"
FK_REDERIVATION_REL = "claudedocs/runtime_logs/grasp_track/g0a_d409/design_inputs/d409_fk_tcp_scalar_rederivation.json"
STATIC_PREP_REL = "claudedocs/runtime_logs/grasp_track/g0a_d409/design_inputs/d409_static_prep_s1s2s3.json"
D339_GRIPPER_REL = "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/d339_gripper_link_cold1_canonical_geometry.json"
D339_LINK5_REL = "claudedocs/runtime_logs/grasp_track/g0a_d339/collision_asset/attempt2/d339_link5_cold1_canonical_geometry.json"
D335_SCAN_CSV_REL = "claudedocs/runtime_logs/grasp_track/g0a_d335/d335_candidate_scan.csv"

# Session-doc sha pins recomputed at build time (mismatch = fail-closed).
SPEC_INPUT_SHA256 = {
    "d371_offline_collider_comparison_evidence": (
        "e300063d37de44d895da3b96ea6ac95c0d108d217f6f74c458bab218d7bccdf5"
    ),
    "d409_fk_tcp_scalar_rederivation": (
        "c0b13007d36de91b6aa8f1190d6d14f8e45e39564325292a5b85d51a0655d5aa"
    ),
    "d409_static_prep_s1s2s3": (
        "f2aaadd13e6822ceebd6a5d565010c0f45c201d08d35b830d582e5ac36dfd63d"
    ),
    "d339_gripper_cold1_canonical_lineage_only": (
        "dc258b27cdef5d29e23f1b5ef3041c3afb26f50d8c8ad9b222002532e95f2e5e"
    ),
    "d339_link5_cold1_canonical_lineage_only": (
        "c45bd056b3487f92bc724474dbf850ea6da309fea90c4e0a90879ada7ba2b655"
    ),
}
STATIC_PREP_PAYLOAD_SHA256 = (
    "43a46e0552e7c23de936b6da59eeb9771b675651c0edbb301d0ed8e5d575b124"
)

# Environment pins (interface pin + D326).
ISAACLAB_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
LVP_ICD = Path("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
EXPECTED_PACKAGE_VERSIONS = {
    "numpy": "1.26.0",
    "psutil": "5.9.8",
    "hpp-fcl": "2.4.4",
    "scipy": "1.15.3",
    "trimesh": "4.5.1",
    "rerun-sdk": "0.34.1",
}
EXPECTED_RERUN_CLI_PREFIX = "rerun-cli 0.34.1"

# Timing contract (C11).
RUN_TIMEOUT_S = 7_200
OBSERVABILITY_BUDGET_NS = 1_800_000_000_000
ADMISSION_MARGIN_NS = 900_000_000_000
PREARM_RUNTIME_BUDGET_NS = 2 * RUN_TIMEOUT_S * 1_000_000_000 + OBSERVABILITY_BUDGET_NS + ADMISSION_MARGIN_NS
MANUAL_STDIN_SAFETY_LEAD_NS = 10_000_000_000
MAX_MANUAL_STDIN_BYTES = 64 * 1024

# Observability constants (D404~D408 repair pattern; W-LES2).
RRD_APPLICATION_ID = "cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration"
RRD_RECORDING_ID = "d409-attempt1-canonical"
SCREENSHOT_LOGICAL_SIZE = "960x540"  # ppp 2.0 -> physical 1920x1080
SCREENSHOT_PHYSICAL_SIZE = (1920, 1080)
RRD_MAX_REPRESENTATIVES = 8
RRD_TIMELINE = "q5_arc"

# Equivalence fixture pins (C5; W-LES4).
EQUIV_NUMERIC_TOL_MM = 1.0e-9
EQUIV_ANTI_RETREAT_EPS_MM = 1.0e-12
D335_CYLINDER_RADIUS_MM = 17.0
D335_LEGACY_GAP_BAND_MM = (0.0, 5.0)
D335_TOP_RULE_MM = 15.0

# Score labels (W-SCI2; registered verbatim in the prereg).
SCORE_LABELS = {
    "proximity_regime_7p881mm": (
        "D330 proximity-regime upper bound - n=5 proximity-cluster sample max, "
        "3D z-inclusive, single target, D34xH90-era historical proxy"
    ),
    "historical_proxy_36p033mm": (
        "historical execution-error proxy (D34xH90-era, single target, "
        "non-replica mean) - label only, NOT a pass/fail gate"
    ),
    "stall_regime_70_81mm": "unreachable within the offset domain (max 18.5mm)",
    "xy_projected_cluster_1p7_7p0mm": (
        "xy-only projected proximity cluster - dimension-matched comparison (diagnostic)"
    ),
}

PREREG_ARTIFACT = "D409_PREREGISTRATION_V1"
STATIC_RESULTS_ARTIFACT = "D409_STATIC_FIXTURE_RESULTS_V1"
ATTESTATION_ARTIFACT = "D409_REVIEWED_SCRIPT_ATTESTATION_V1"
TUPLE_ARTIFACT = "D409_PROPOSED_RUNTIME_HASH_TUPLE_V1"
PROMOTION_ARTIFACT = "D409_CANONICAL_PROMOTION_V1"
RERUN_VALIDATION_ARTIFACT = "D409_RERUN_VALIDATION_V1"
COMPLETION_ARTIFACT = "D409_COMPLETION_SUMMARY_V1"
RECEIPT_ARTIFACT = "D409_MANUAL_WRITER_RECEIPT_V1"
INVENTORY_ARTIFACT = "D409_PRERUN_INVENTORY_V1"

STATIC_PASS_STATUS = "D409_G0A_ZERO_STEP_STATIC_PREP_PASS_STOP"
STATIC_FAIL_STATUS = "D409_G0A_ZERO_STEP_STATIC_PREP_FAIL_STOP"
RUNTIME_COMPLETE_STATUS = (
    "D409_G0A_ZERO_STEP_DUAL_JAW_CONTACT_REGION_ENUMERATION_RUNTIME_COMPLETE_STOP"
)
RUNTIME_FAIL_STATUS = (
    "D409_G0A_ZERO_STEP_DUAL_JAW_CONTACT_REGION_ENUMERATION_RUNTIME_FAIL_STOP"
)


class D409Error(RuntimeError):
    """Controller contract violation (fail-closed)."""


_UNRESOLVED = object()


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------

def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    """Writer-compatible canonical framing (sort_keys, compact, LF)."""
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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise D409Error(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _strict_json_bytes(raw: bytes) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda name: (_ for _ in ()).throw(
                D409Error(f"non-finite JSON constant: {name}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise D409Error(f"strict JSON parse failed: {exc}") from exc


def _write_json_x(path: Path, value: dict[str, Any]) -> str:
    """Exclusive-create JSON write with fsync (candidate overwrite forbidden)."""
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
    )
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise D409Error(f"short write: {path}")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    return _sha_bytes(raw)


def _write_bytes_x(path: Path, raw: bytes) -> str:
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
    )
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise D409Error(f"short write: {path}")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    return _sha_bytes(raw)


def _fsync_existing(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_regular_nlink1(path: Path, label: str) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise D409Error(f"{label} is not a regular nlink==1 file: {path}")


def _require_hex64(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value) <= set("0123456789abcdef")
    ):
        raise D409Error(f"{label} is not a lowercase sha256 hex digest")
    return value


def _proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_bytes()
    text = raw.decode("utf-8")
    right_paren = text.rfind(")")
    if right_paren < 0:
        raise D409Error(f"malformed /proc/{pid}/stat")
    fields_after_comm = text[right_paren + 1 :].strip().split()
    if len(fields_after_comm) <= 19:
        raise D409Error(f"short /proc/{pid}/stat")
    return int(fields_after_comm[19])


def _git(*args: str) -> str:
    command = ["git", "-C", str(PROJECT_ROOT), *args]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        raise D409Error(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _git_head() -> str:
    return _git("rev-parse", "HEAD").strip()


def _git_dirty_paths() -> list[str]:
    paths: list[str] = []
    for line in _git("status", "--porcelain").splitlines():
        if len(line) >= 4:
            entry = line[3:]
            if " -> " in entry:
                entry = entry.split(" -> ", 1)[1]
            paths.append(entry.strip().strip('"'))
    return sorted(set(paths))


# ---------------------------------------------------------------------------
# Consumer derivation — worker (AST + module import cross-check; D405/D407).
# ---------------------------------------------------------------------------

def _resolve_literal(node: ast.AST, consts: dict[str, Any]) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        pass
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        items = []
        for element in node.elts:
            value = _resolve_literal(element, consts)
            if value is _UNRESOLVED:
                return _UNRESOLVED
            items.append(value)
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "frozenset" and len(node.args) == 1:
            inner = _resolve_literal(node.args[0], consts)
            if inner is not _UNRESOLVED:
                try:
                    return frozenset(inner)
                except TypeError:
                    return _UNRESOLVED
    return _UNRESOLVED


def _ast_module_constants(source_path: Path) -> dict[str, Any]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    consts: dict[str, Any] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            value = _resolve_literal(node.value, consts)
            if value is not _UNRESOLVED:
                consts[node.targets[0].id] = value
    return consts


def _ast_compare_string_literals(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            for candidate in (node.left, comparator):
                if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                    found.add(candidate.value)
    return sorted(found)


def _ast_dict_string_values_for_key(source_path: Path, key: str) -> list[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key_node, value_node in zip(node.keys, node.values):
            if (
                isinstance(key_node, ast.Constant)
                and key_node.value == key
                and isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, str)
            ):
                found.add(value_node.value)
    return sorted(found)


def _ast_cli_option_strings(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                options.add(argument.value)
    return sorted(options)


_WORKER_MODULE_CACHE: Any = None


def _load_worker_module() -> Any:
    global _WORKER_MODULE_CACHE
    if _WORKER_MODULE_CACHE is None:
        spec = importlib.util.spec_from_file_location(
            "cyld29h50_d409_worker_frozen", WORKER_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _WORKER_MODULE_CACHE = module
    return _WORKER_MODULE_CACHE


WORKER_INTERFACE_CONSTANT_NAMES = (
    "CLAIM_NAME",
    "EVIDENCE_NAME",
    "REGION_CSV_NAME",
    "SUMMARY_NAME",
    "PRECLOSE_NAME",
    "PHASE_NAME",
    "EXIT_PASS",
    "EXIT_CONTRACT_FAIL",
    "EXIT_CLAIM_PREEXIST",
    "VERDICT_COMPLETE",
    "VERDICT_CONTRACT_FAIL",
    "DETERMINISM_BYTE_COMPARE_MEMBERS",
    "MAX_QUERIES_PER_POSE",
    "MAX_QUERIES_PER_RUN",
    "CERT_TRAVERSAL_NEW_EVAL_CAP",
    "CERT_NEIGHBORHOOD_RAD",
    "CSV_COLUMNS",
    "PINNED_INPUT_SHA256",
    "OUT_DIR_REQUIRED_FRAGMENT",
    "EXPECTED_POSE_COUNT",
    "GRID_STEP_UM",
    "RADIAL_MIN_UM",
    "RADIAL_MAX_UM",
    "TANGENT_MIN_UM",
    "TANGENT_MAX_UM",
    "POSITIVE_CONTROL_KEY_UM",
    "ANCHOR_GATE_MM",
    "ANCHOR_REF_LINK5_MM_REPR",
    "ANCHOR_REF_GRIPPER_MM_REPR",
    "BISECT_BRACKET_RAD",
    "BISECT_MAX_ITER",
    "ARC_ANCHOR_COUNT",
    "Q5_OPEN_RAD",
    "CYL_RADIUS_M",
    "CYL_HEIGHT_M",
    "CYL_X_M",
    "CLEAR_GATE_MM",
    "FIXED_JAW_BAND_MM",
    "TCP_GATE_MM",
    "JAW_TANGENT_GATE_DEG",
    "RIM_PROXIMITY_BAND_MM",
    "PINCH_CORE_NAMES",
    "PINCH_DIAGNOSTIC_NAMES",
    "LINK5_FIXED_EXPECTED",
    "GRIPPER_INNER_EXPECTED_COUNT",
    "GRIPPER_OUTER_EXPECTED_COUNT",
    "OUTER_DIFF_PART",
    "TABLE_Z_PIN_REPR",
    "Z_CENTER_PIN_REPR",
    "ANTI_RETREAT_NUMERATOR_UM",
)


def _derive_worker_interface() -> dict[str, Any]:
    """AST-derive the worker interface, cross-checked vs the imported module.

    D405/D407 lesson (C2): every literal this controller later compares
    against worker behavior comes from the worker SOURCE, never from
    hand-authoring in this file.
    """
    ast_constants = _ast_module_constants(WORKER_PATH)
    module = _load_worker_module()
    derived: dict[str, Any] = {}
    mismatches: list[str] = []
    for name in WORKER_INTERFACE_CONSTANT_NAMES:
        if name not in ast_constants:
            mismatches.append(f"missing-from-ast:{name}")
            continue
        ast_value = ast_constants[name]
        module_value = getattr(module, name, _UNRESOLVED)
        normalized_module = (
            tuple(module_value) if isinstance(module_value, list) else module_value
        )
        normalized_ast = tuple(ast_value) if isinstance(ast_value, list) else ast_value
        if normalized_ast != normalized_module:
            mismatches.append(f"ast-vs-module:{name}")
        derived[name] = ast_value
    if mismatches:
        raise D409Error(f"worker interface derivation mismatch: {mismatches}")
    compare_literals = _ast_compare_string_literals(WORKER_PATH)
    status_literals = _ast_dict_string_values_for_key(WORKER_PATH, "status")
    if "PASS" not in compare_literals:
        raise D409Error(
            f"worker admission-row status literal 'PASS' not found in ast.Eq/NotEq: {compare_literals}"
        )
    if len(status_literals) != 1:
        raise D409Error(
            f"worker preclose status literal derivation ambiguous: {status_literals}"
        )
    cli_options = _ast_cli_option_strings(WORKER_PATH)
    for required_option in ("--out-dir", "--prereg"):
        if required_option not in cli_options:
            raise D409Error(f"worker CLI option missing: {required_option}")
    derived["ADMISSION_ROW_STATUS_LITERAL"] = "PASS"
    derived["PRECLOSE_STATUS_LITERAL"] = status_literals[0]
    derived["COMPARE_STRING_LITERALS"] = compare_literals
    derived["CLI_OPTIONS"] = cli_options
    derived["WORKER_SOURCE_SHA256"] = _sha_path(WORKER_PATH)
    worker_artifact_basenames = tuple(
        ast_constants[name]
        for name in (
            "CLAIM_NAME",
            "EVIDENCE_NAME",
            "REGION_CSV_NAME",
            "SUMMARY_NAME",
            "PRECLOSE_NAME",
            "PHASE_NAME",
        )
    )
    derived["WORKER_ARTIFACT_BASENAMES"] = worker_artifact_basenames
    return derived


def _worker_registered_metric_names() -> dict[str, list[str]]:
    """Consumer-derive the aggregate-count and region-entry metric names
    from the worker source (OPS-B1 repair R2; D405 no-hand-authoring)."""
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
    counts_keys: list[str] | None = None
    region_entry_keys: list[str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run":
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Assign)
                    and any(getattr(target, "id", None) == "counts" for target in sub.targets)
                    and isinstance(sub.value, ast.Dict)
                ):
                    counts_keys = [key.value for key in sub.value.keys]
        if isinstance(node, ast.FunctionDef) and node.name == "_region_analysis":
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "append"
                    and getattr(sub.func.value, "id", None) == "region_entries"
                    and sub.args
                    and isinstance(sub.args[0], ast.Dict)
                ):
                    region_entry_keys = [key.value for key in sub.args[0].keys]
    if not counts_keys or not region_entry_keys:
        raise D409Error("registered-metrics derivation from worker source failed (R2)")
    return {"counts": counts_keys, "region_entry": region_entry_keys}


# ---------------------------------------------------------------------------
# Consumer derivation — manual writer (contract JSON + AST literals).
# ---------------------------------------------------------------------------

_WRITER_CONTRACT_CACHE: dict[str, Any] | None = None


def _writer_contract() -> dict[str, Any]:
    global _WRITER_CONTRACT_CACHE
    if _WRITER_CONTRACT_CACHE is not None:
        return _WRITER_CONTRACT_CACHE
    result = subprocess.run(
        [str(ISAACLAB_PYTHON), "-B", str(WRITER_PATH), "--print-contract-json"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        raise D409Error(f"writer contract fetch failed: {result.stderr.strip()[:500]}")
    contract = _strict_json_bytes(result.stdout.encode("utf-8"))
    if not isinstance(contract, dict):
        raise D409Error("writer contract is not an object")
    writer_ast = _ast_module_constants(WRITER_PATH)
    for name in ("PHASE_ROW_ARTIFACT", "SCREENSHOT_MANIFEST_ARTIFACT", "MANUAL_ARTIFACT"):
        if name not in writer_ast or not isinstance(writer_ast[name], str):
            raise D409Error(f"writer AST literal missing: {name}")
    contract["_ast"] = {
        "PHASE_ROW_ARTIFACT": writer_ast["PHASE_ROW_ARTIFACT"],
        "SCREENSHOT_MANIFEST_ARTIFACT": writer_ast["SCREENSHOT_MANIFEST_ARTIFACT"],
        "MANUAL_ARTIFACT": writer_ast["MANUAL_ARTIFACT"],
        "MANUAL_TIMEOUT_NS": writer_ast["MANUAL_TIMEOUT_NS"],
        "WRITER_DEADLINE_LEAD_NS": writer_ast["WRITER_DEADLINE_LEAD_NS"],
    }
    if contract.get("artifact") != "D409_MANUAL_WRITER_CONTRACT_V1":
        raise D409Error(f"unexpected writer contract artifact: {contract.get('artifact')}")
    if contract.get("manual_basename") != "d409_manual_visual_inspection.json":
        raise D409Error("writer manual basename drift")
    expected_paths = contract.get("expected_paths", {})
    for key, expected in (
        ("controller", _rel(CONTROLLER_PATH)),
        ("worker", _rel(WORKER_PATH)),
        ("preregistration", _rel(PREREG_PATH)),
        ("tuple", _rel(TUPLE_PATH)),
        ("phase_log", _rel(PHASE_PATH)),
        ("screenshot_manifest", _rel(SCREENSHOT_MANIFEST_PATH)),
        ("root", _rel(ATTEMPT_ROOT)),
    ):
        if expected_paths.get(key) != expected:
            raise D409Error(
                f"writer expected path mismatch [{key}]: {expected_paths.get(key)} != {expected}"
            )
    _WRITER_CONTRACT_CACHE = contract
    return contract


# ---------------------------------------------------------------------------
# Environment gate (D326 pins; interpreter identity; rerun CLI identity).
# ---------------------------------------------------------------------------

def _environment_gate() -> dict[str, Any]:
    if not sys.dont_write_bytecode:
        raise D409Error("controller must run with python -B")
    interpreter_real = os.path.realpath(sys.executable)
    pinned_real = os.path.realpath(str(ISAACLAB_PYTHON))
    packages = {}
    for package, expected in EXPECTED_PACKAGE_VERSIONS.items():
        observed = importlib.metadata.version(package)
        packages[package] = {"expected": expected, "observed": observed}
        if observed != expected:
            raise D409Error(f"package pin violated: {package} {observed} != {expected}")
    if np.__version__ != EXPECTED_PACKAGE_VERSIONS["numpy"]:
        raise D409Error(f"imported numpy drift: {np.__version__}")
    cli = subprocess.run(
        [str(RERUN_CLI), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )
    if cli.returncode != 0 or not cli.stdout.startswith(EXPECTED_RERUN_CLI_PREFIX):
        raise D409Error(f"rerun CLI identity mismatch: {cli.stdout.splitlines()[:1]}")
    gate = {
        "interpreter": sys.executable,
        "interpreter_realpath": interpreter_real,
        "interpreter_pin_realpath": pinned_real,
        "interpreter_matches_pin": interpreter_real == pinned_real,
        "python_version": sys.version.split()[0],
        "packages": packages,
        "rerun_cli_path": str(RERUN_CLI),
        "rerun_cli_version_line": cli.stdout.splitlines()[0],
        "rerun_cli_sha256": _sha_path(RERUN_CLI),
        "scope_guard_violations": list(_SCOPE_GUARD_VIOLATIONS),
    }
    if not gate["interpreter_matches_pin"]:
        raise D409Error(
            f"controller interpreter is not the pinned isaaclab python: {interpreter_real}"
        )
    if _SCOPE_GUARD_VIOLATIONS:
        raise D409Error(f"scope guard violations recorded: {_SCOPE_GUARD_VIOLATIONS}")
    return gate
# ---------------------------------------------------------------------------
# Preregistration builder (D371 lineage + D407 lineage + section 4.3 delta).
# ---------------------------------------------------------------------------

def _future_runtime_paths(worker_iface: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Programmatic future-leaf derivation (D405/D407): worker leaves from the
    worker-source *_NAME constants, writer leaves from the writer contract,
    controller leaves from this module's path constants."""
    leaves: set[str] = set()
    for run_name in RUN_DIR_NAMES:
        run_rel = f"{_rel(ATTEMPT_ROOT)}/{run_name}"
        for basename in worker_iface["WORKER_ARTIFACT_BASENAMES"]:
            leaves.add(f"{run_rel}/{basename}")
        for basename in RUN_CONTROLLER_BASENAMES:
            leaves.add(f"{run_rel}/{basename}")
    for path in RUNTIME_CONTROLLER_ATTEMPT_PATHS:
        leaves.add(_rel(path))
    leaves.add(contract["expected_paths"]["manual_output"])
    leaves.add(f"{_rel(ATTEMPT_ROOT)}/{contract['manual_pending_basename']}")
    for value in contract["protocol_schemas"]["screenshot_rrd_report_paths"].values():
        leaves.add(value)
    for basename in contract["screenshot_layout"]:
        leaves.add(f"{_rel(ATTEMPT_ROOT)}/{basename}")
    directory_sentinels = [
        f"{_rel(CASE_ROOT)}/",
        f"{_rel(ATTEMPT_ROOT)}/",
        *(f"{_rel(ATTEMPT_ROOT)}/{run_name}/" for run_name in RUN_DIR_NAMES),
    ]
    return {
        "future_runtime_leaves": sorted(leaves),
        "directory_sentinels": directory_sentinels,
        "derivation": (
            "programmatic union: worker-source *_NAME constants (ast) x run1/run2 + "
            "manual-writer --print-contract-json expected paths + controller module "
            "path constants (D405 consumer-derivation / D407 allowed-dirty lesson)"
        ),
    }


def _load_static_prep() -> dict[str, Any]:
    path = PROJECT_ROOT / STATIC_PREP_REL
    raw = path.read_bytes()
    observed = _sha_bytes(raw)
    if observed != SPEC_INPUT_SHA256["d409_static_prep_s1s2s3"]:
        raise D409Error(f"static-prep artifact sha drift: {observed}")
    wrapper = json.loads(raw)
    payload_sha = wrapper["determinism_check"]["payload_sha256_run1"]
    if (
        wrapper["determinism_check"]["bit_exact"] is not True
        or payload_sha != STATIC_PREP_PAYLOAD_SHA256
        or wrapper["determinism_check"]["payload_sha256_run2"] != STATIC_PREP_PAYLOAD_SHA256
    ):
        raise D409Error("static-prep determinism check drift")
    return wrapper


def _build_prereg(
    worker_iface: dict[str, Any],
    contract: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    module = _load_worker_module()
    static_prep = _load_static_prep()
    registered_metric_names = _worker_registered_metric_names()
    input_hashes = {
        "d348_callback_topology_volume_evidence": _sha_path(PROJECT_ROOT / D348_REL),
        "d368_semantic_allocation_evidence": _sha_path(PROJECT_ROOT / D368_REL),
        "d349_frozen_target_distance_measurement": _sha_path(PROJECT_ROOT / D349_REL),
        "urdf_roarm_m3": _sha_path(PROJECT_ROOT / URDF_REL),
        "d371_offline_collider_comparison_evidence": _sha_path(PROJECT_ROOT / D371_EVIDENCE_REL),
        "d409_fk_tcp_scalar_rederivation": _sha_path(PROJECT_ROOT / FK_REDERIVATION_REL),
        "d409_static_prep_s1s2s3": _sha_path(PROJECT_ROOT / STATIC_PREP_REL),
        "d339_gripper_cold1_canonical_lineage_only": _sha_path(PROJECT_ROOT / D339_GRIPPER_REL),
        "d339_link5_cold1_canonical_lineage_only": _sha_path(PROJECT_ROOT / D339_LINK5_REL),
    }
    worker_pins = worker_iface["PINNED_INPUT_SHA256"]
    tag_map = {
        "d348": "d348_callback_topology_volume_evidence",
        "d368": "d368_semantic_allocation_evidence",
        "d349": "d349_frozen_target_distance_measurement",
        "urdf": "urdf_roarm_m3",
    }
    for tag, key in tag_map.items():
        if input_hashes[key] != worker_pins[tag]:
            raise D409Error(f"input hash three-way mismatch [{tag}]: {input_hashes[key]}")
    for key, expected in SPEC_INPUT_SHA256.items():
        if input_hashes[key] != expected:
            raise D409Error(f"session-doc sha pin drift [{key}]: {input_hashes[key]}")
    dirty_live = _git_dirty_paths()
    overlay = _future_runtime_paths(worker_iface, contract)
    allowed_dirty = sorted(
        set(dirty_live)
        | {_rel(path) for path in STATIC_ARTIFACT_PATHS}
        | set(overlay["future_runtime_leaves"])
        | set(overlay["directory_sentinels"])
    )
    registered_worker_command_template = [
        str(ISAACLAB_PYTHON),
        "-B",
        str(WORKER_PATH),
        "--out-dir",
        "<absolute run dir>",
        "--prereg",
        str(PREREG_PATH),
    ]
    per_run_commands = {
        run_name: [
            str(ISAACLAB_PYTHON),
            "-B",
            str(WORKER_PATH),
            "--out-dir",
            str(ATTEMPT_ROOT / run_name),
            "--prereg",
            str(PREREG_PATH),
        ]
        for run_name in RUN_DIR_NAMES
    }
    timing_runs = [static_prep["timing_run1"], static_prep["timing_run2"]]
    prereg = {
        "artifact": PREREG_ARTIFACT,
        "case": "g0a_d409",
        "created_utc": _utc_now(),
        "session_spec": {
            "doc": SESSION_DOC_REL,
            "sections": "section 2 (design v1) + section 4 (confirmed delta v2; section 4 wins)",
        },
        "new_variables": [
            "real cylinder geometry D29xH50 (r=0.0145m, H=0.050m; user-measured "
            "2026-08-03, HARD RULE #18; analytic hppfcl.Cylinder per D379)"
        ],
        "head_pin": {"git_head": _git_head(), "git_dirty_live": dirty_live},
        "input_hashes": input_hashes,
        "a64_authority": {
            "authority": D348_REL,
            "query_geometry": "rows[].instance.vertices_m + rows[].instance.topology_triangles (callback original)",
            "per_part_integrity": "instance.payload_sha256 recorded verbatim; runtime re-verification basis = D409-canonical per-part geometry hash (S1)",
            "d339_status": "historical cook witness only - demoted, never queried (P1)",
        },
        "worker_admission_rows": [
            {
                "row_id": "worker_source",
                "path": str(WORKER_PATH),
                "sha256": worker_iface["WORKER_SOURCE_SHA256"],
            },
            {
                "row_id": "static_prep_s1s2s3",
                "path": str(PROJECT_ROOT / STATIC_PREP_REL),
                "sha256": input_hashes["d409_static_prep_s1s2s3"],
            },
        ],
        "environment": environment,
        "candidate_contract": {
            "harness_file_count": 3,
            "controller": {"path": _rel(CONTROLLER_PATH), "sha256": _sha_path(CONTROLLER_PATH)},
            "worker": {"path": _rel(WORKER_PATH), "sha256": worker_iface["WORKER_SOURCE_SHA256"]},
            "manual_writer": {"path": _rel(WRITER_PATH), "sha256": _sha_path(WRITER_PATH)},
        },
        "registered_worker_command": {
            "template": registered_worker_command_template,
            "per_run": per_run_commands,
            "note": "per-run argv differs only in the --out-dir argument (P4-1)",
        },
        "worker_interface_derived": {
            "artifact_basenames": list(worker_iface["WORKER_ARTIFACT_BASENAMES"]),
            "exit_codes": {
                "pass": worker_iface["EXIT_PASS"],
                "claim_preexist": worker_iface["EXIT_CLAIM_PREEXIST"],
                "contract_fail": worker_iface["EXIT_CONTRACT_FAIL"],
            },
            "verdict_complete": worker_iface["VERDICT_COMPLETE"],
            "preclose_status_literal": worker_iface["PRECLOSE_STATUS_LITERAL"],
            "admission_row_status_literal": worker_iface["ADMISSION_ROW_STATUS_LITERAL"],
            "derivation": "worker source AST (constants, ast.Eq/NotEq strings, dict 'status' values) cross-checked vs imported module (C2)",
        },
        "determinism_run_contract": {
            "worker_invocations_total": 2,
            "per_run_out_dirs": ["run1", "run2"],
            "automatic_retries": 0,
            "run2_precondition": "run1_preclose_pass",
            "byte_compare_manifest": list(worker_iface["DETERMINISM_BYTE_COMPARE_MEMBERS"]),
            "run_timeout_s": RUN_TIMEOUT_S,
            "run2_failure_semantics": "run2 failure or byte mismatch = attempt fail-closed, consumed (D408-R1)",
        },
        "determinism_manifest": {
            "byte_compare_members": list(worker_iface["DETERMINISM_BYTE_COMPARE_MEMBERS"]),
            "excluded": sorted(
                {
                    worker_iface["CLAIM_NAME"],
                    worker_iface["SUMMARY_NAME"],
                    worker_iface["PRECLOSE_NAME"],
                    worker_iface["PHASE_NAME"],
                    WORKER_STDOUT_BASENAME,
                    WORKER_STDERR_BASENAME,
                    WORKER_SUPERVISOR_BASENAME,
                    RRD_PATH.name,
                    RBL_PATH.name,
                    REGION_SCREENSHOT_PATH.name,
                    DECISION_SHEET_PATH.name,
                    contract["manual_basename"],
                    PHASE_PATH.name,
                }
            ),
            "note": "canonical evidence JSON + region CSV bytes only (P4-4; OPS1-W1 merged)",
        },
        "registered_budget": {
            "max_queries_per_pose": worker_iface["MAX_QUERIES_PER_POSE"],
            "max_queries_per_run": worker_iface["MAX_QUERIES_PER_RUN"],
            "basis_file": STATIC_PREP_REL,
            "basis_file_sha256": input_hashes["d409_static_prep_s1s2s3"],
            "basis_payload_sha256": STATIC_PREP_PAYLOAD_SHA256,
            "basis_us_per_query": [run["us_per_query"] for run in timing_runs],
            "basis_extrapolation_4p5M_s": [run["extrapolation_s"]["4.5M"] for run in timing_runs],
            "basis_budget_check_true_both_runs": all(
                run["registered_budget_check_4p5M_lt_7200s"] for run in timing_runs
            ),
            "timeout_s": RUN_TIMEOUT_S,
            "note": "budget cited from the persisted static-prep file only (W-OPS4); session-doc prose numbers are demoted to reference",
            "r1_certification_amendment": {
                "reason": (
                    "SCI-B1 repair R1 adds the ordered chord-bound certification "
                    "traversal (memoized midpoint evaluations, deterministic cap); "
                    "registered budgets amended from 3,600/4.5M to the worker pins "
                    "above (worst pose = 2,176 + cap x 64; disposition-recorded)"
                ),
                "cert_new_eval_cap": worker_iface["CERT_TRAVERSAL_NEW_EVAL_CAP"],
                "extrapolation_7p0M_s": [
                    run["extrapolation_s"]["4.5M"] * (7.0 / 4.5) for run in timing_runs
                ],
                "check_7p0M_lt_7200s_both_runs": all(
                    run["extrapolation_s"]["4.5M"] * (7.0 / 4.5) < RUN_TIMEOUT_S
                    for run in timing_runs
                ),
            },
        },
        "registered_metrics": {
            "derivation": (
                "consumer-derived (D405; OPS-B1 repair R2): per-pose metrics = "
                "worker CSV_COLUMNS verbatim (AST constant cross-checked vs the "
                "imported module); aggregate counts and region-entry metric names "
                "are AST-derived from the worker's counts/region_entries dict "
                "literals; region scoring thresholds/labels are registered under "
                "gates.region_scoring"
            ),
            "per_pose_csv_columns": list(worker_iface["CSV_COLUMNS"]),
            "aggregate_counts": registered_metric_names["counts"],
            "region_entry_metrics": registered_metric_names["region_entry"],
            "region_scoring_reference": "gates.region_scoring",
        },
        "anchor_gate": {
            "threshold_mm": worker_iface["ANCHOR_GATE_MM"],
            "policy": "4-channel ANY-reject: {link5 FK pos err, gripper FK pos err, link5 dist delta, gripper dist delta}; any channel > threshold -> reject",
            "reference_link5_mm_repr": worker_iface["ANCHOR_REF_LINK5_MM_REPR"],
            "reference_gripper_mm_repr": worker_iface["ANCHOR_REF_GRIPPER_MM_REPR"],
            "distance_channel_discrimination": "pi/2 gripper dist delta 0.0001777mm is below threshold - distance-channel discrimination is link5 only (section 4.2 pin)",
            "calibration_semantics": "old cylinder (0.017,0.090) at the stored D349 object pose is query-pipeline calibration only; no D362-era physics transfer (D379)",
            "rot_err": "diagnostic record only",
        },
        "geometry": {
            "cylinder_model": "hppfcl.Cylinder (analytic primitive, D379)",
            "radius_m_repr": repr(module.CYL_RADIUS_M),
            "height_m_repr": repr(module.CYL_HEIGHT_M),
            "x_m_repr": repr(module.CYL_X_M),
            "table_z_m_repr": worker_iface["TABLE_Z_PIN_REPR"],
            "z_center_m_repr": worker_iface["Z_CENTER_PIN_REPR"],
            "precision_note": "x = float32(0.3) as float64 literal; table_z/z_center = float64 operation sequence, no cast (W-SCI3)",
        },
        "grid": {
            "radial_um": {
                "min": worker_iface["RADIAL_MIN_UM"],
                "max": worker_iface["RADIAL_MAX_UM"],
                "step": worker_iface["GRID_STEP_UM"],
            },
            "tangent_um": {
                "min": worker_iface["TANGENT_MIN_UM"],
                "max": worker_iface["TANGENT_MAX_UM"],
                "step": worker_iface["GRID_STEP_UM"],
            },
            "pose_count": worker_iface["EXPECTED_POSE_COUNT"],
            "positive_control_key_um": list(worker_iface["POSITIVE_CONTROL_KEY_UM"]),
            "tau_derivation": "d335 formula with radius substitution (inherited): [R-8mm, R-8mm+5mm], R=14.5mm, 8mm=FIXED_JAW_FACE_LOCAL_M frozen literal (d323:38) - P2 honest label",
            "edge_touch_rule": "an admitted 4-connected component touching a rho/tau domain-edge cell gets rho_R flagged domain-censored (report-only, never gated); 'exhaustive' means exhaustive within the declared domain (P2)",
            "anti_retreat": "14.5mm - rho >= 0 (P2 rebase of 17mm - r)",
        },
        "gates": {
            "admission_checks": [
                "ik_converged",
                "commanded_tcp_error_le_5mm",
                "jaw_tangent_le_15deg",
                "link5_all64_noninterpenetration_ge_0p1mm",
                "gripper_open_all64_noninterpenetration_ge_0p1mm",
                "anti_retreat_14p5mm_minus_rho_nonnegative",
            ],
            "fixed_jaw_band_mm": list(worker_iface["FIXED_JAW_BAND_MM"]),
            "fixed_jaw_band_reuse_disclosure": (
                "5.0mm upper bound REUSES the old D330 planar-proxy gate constant; D330 "
                "metric = tangent-projected planar gap 0-5mm vs new metric = hppfcl 3D min "
                "over the link5 4-mask (~1.7mm apart at (7,11)-class poses); planar-gap "
                "equivalent recorded per pose as a diagnostic column (P3/W-SCI1)"
            ),
            "b_checks_core": [
                "crossing_exists_in_open_interval",
                "first_crossing_order_certified",
                "first_contact_part_in_inner17",
                "competitor_exclusion",
                "cylinder_witness_barrel_interior_strict",
            ],
            "order_certification": {
                "rule": (
                    "R1 (SCI-B1): D351 traverse semantics; certification criterion "
                    "= max(d_hi, d_lo) > 2*Rmax*sin(|dq|/2) (declared A12 exclusion "
                    "criterion, sharp sound form); terminal-width clear-clear "
                    "intervals accepted only within the neighborhood cap above the "
                    "bracket with per-part exclusion; fail-closed otherwise; "
                    "precision disclosure (SCI-R1-W1): raw GJK distances "
                    "(tolerance 1e-9 m), no additional numerical allowance — a "
                    "certification margin below that tolerance would be unsound "
                    "(observed margins are orders of magnitude larger)"
                ),
                "new_eval_cap": worker_iface["CERT_TRAVERSAL_NEW_EVAL_CAP"],
                "neighborhood_cap_rad": worker_iface["CERT_NEIGHBORHOOD_RAD"],
            },
            "pinch_core_names": list(worker_iface["PINCH_CORE_NAMES"]),
            "pinch_core_selection_rationale": "opposition 1 + geometric placement 2 + closing direction 1 (W-SCI4); d351 formula-structure reuse, NOT a reuse of its pass=false result",
            "pinch_diagnostic_names": list(worker_iface["PINCH_DIAGNOSTIC_NAMES"]),
            "barrel_cap_classifier": {
                "strict": True,
                "rule": "strict z order only; no new geometric success tolerance (D354 durable rule); R/H rebased 0.0145/0.050",
            },
            "q5_bisection": {
                "bracket_rad": worker_iface["BISECT_BRACKET_RAD"],
                "max_iter": worker_iface["BISECT_MAX_ITER"],
                "note": "numerical-resolution control, not a science tolerance (D351 lineage)",
            },
            "top_15mm_rule": "demoted to non-gate diagnostic; witness top margins recorded per pose; rim-proximity cell fraction reported (band 7.5mm H50-proportional, NOT a gate) (W-LES1)",
            "region_scoring": {
                "region_definition": "4-connected components of admitted cells; representative = deepest interior cell (ties: lexicographic key)",
                "proximity_regime_mm": 7.881,
                "historical_proxy_mm": 36.033,
                "historical_proxy": {"standalone_gate": None},
                "stall_regime_mm": [70.0, 81.0],
                "labels": SCORE_LABELS,
                "no_standalone_36p033_gate": True,
                "offset_space_limitation": "offset-space distance is a TCP displacement proxy (z excluded)",
            },
        },
        "kinematics": {
            "fk_constant_series": "URDF XML literals only; pi/2-symbol chain (roarm_kinematics._CHAIN) banned (section 2.4)",
            "ik_form": "d323 position-only 5-DOF DLS, HOME seed, max_iter 120, pos_tol 1mm, step clip 4deg, v6 soft-limit clip; deterministic, zero randomness",
            "fk_accuracy_statement": "stated as measured residuals (anchor-gate channels), never bit-exact",
        },
        "negative_controls": {
            "prepare_time": [
                {"id": "N1", "control": "cylinder radius tampered to 0.017", "expected": "real-geometry pin check FAIL"},
                {"id": "N2", "control": "1-bit vertex-stream tamper + 1-byte file tamper", "expected": "D409-canonical per-part hash FAIL vs S1 pin + d348 file sha FAIL (C3 re-anchor of P1)"},
                {"id": "N3", "control": "inner mask replaced by outer 16", "expected": "mask name-set/count FAIL; decisive difference = part_035 (W-FRZ1 documented)"},
                {"id": "N4", "control": "FK constants replaced by pi/2-symbol series", "expected": "anchor gate ANY-reject fires (discrimination measured; if it does not fire the control set must be redesigned)"},
                {"id": "N5", "control": "empty grid (0 poses)", "expected": "worker static-pin admission FAIL (C4)"},
            ],
            "audit_registered": [
                {"id": "A1", "control": "tolerance introduced into barrel/cap classifier registration", "expected": "strict-contract violation reject"},
                {"id": "A2", "control": "bisection bracket > 1e-6 rad", "expected": "reject"},
                {"id": "A3", "control": "36.033 mean promoted to standalone gate", "expected": "scoring-contract violation reject"},
                {"id": "A4", "control": "isaac-family import attempt", "expected": "scope guard rejects the import itself"},
            ],
        },
        "positive_controls": [
            {"id": "P1", "control": "frozen (7,11) anchor reproduction (section 2.9-1/2)", "expected": "4 channels <= 0.0005mm"},
            {"id": "P2", "control": "grid key (7000,11000) inclusion", "expected": "present; meaning differs from the frozen candidate (new center, real cylinder)"},
        ],
        "equivalence_fixture": {
            "source_csv": D335_SCAN_CSV_REL,
            "selection": "coarse rows at deterministic quantile indices {0, n/4, n/2, 3n/4, n-1} plus the (7.0, 11.0) row",
            "tolerances": {
                "numeric_recompute_mm": EQUIV_NUMERIC_TOL_MM,
                "anti_retreat_eps_mm": EQUIV_ANTI_RETREAT_EPS_MM,
                "booleans": "exact",
            },
            "not_reconstructible": "live_exact_written_tcp_error_le_5mm (live-written alignment scalar not exported by the d335 CSV) - disclosed (C5)",
        },
        "visual_contract": {
            "rrd": _rel(RRD_PATH),
            "rbl": _rel(RBL_PATH),
            "validation": _rel(RERUN_VALIDATION_PATH),
            "region_map_screenshot": _rel(REGION_SCREENSHOT_PATH),
            "decision_sheet": _rel(DECISION_SHEET_PATH),
            "screenshot_physical": list(SCREENSHOT_PHYSICAL_SIZE),
            "screenshot_logical": SCREENSHOT_LOGICAL_SIZE,
            "pixels_per_point": 2.0,
            "rerun_cli_exact_version": EXPECTED_RERUN_CLI_PREFIX,
            "entities_required": [
                "/metadata/run",
                "/enum/prototype/cylinder",
                "/enum/grid/admission",
                "/enum/grid/full_pass",
                "/enum/source/<body>/part_NNN (128)",
                "/enum/instance/anchor_frozen/*",
                "/enum/instance/rep_<rho>_<tau>/* (region representatives, capped detail subset)",
                "/enum/candidate/**",
            ],
            "required_components": {
                "/enum/grid/*": ["Points2D:positions", "Points2D:colors"],
                "/enum/source/**": ["Mesh3D:vertex_positions", "Mesh3D:triangle_indices"],
                "/enum/instance/**": ["Mesh3D:vertex_positions", "Mesh3D:triangle_indices"],
                "/metadata/run": ["TextDocument:text"],
            },
            "timeline": RRD_TIMELINE,
            "grid_full_layer": "all 1,239 cells logged (W-LES2); Float64 authority = evidence JSON; RRD Float32 copies are inspection-only, never hashed into a gate",
            "blueprint": "embedded (save default_blueprint) + standalone .rbl export; rerun CLI 'rrd verify --check-footers true' on both",
            "phase_order": "canonical evidence + verdict sha256 published BEFORE any presentation artifact (W-LES3), enforced by phase markers",
            "error_banner": "automated: decodability + non-blank check; authoritative judgment = manual 11-field schema (C7)",
        },
        "manual_contract": {
            "required_boolean_fields": contract["required_boolean_fields"],
            "screenshot_layout": contract["screenshot_layout"],
            "manual_timeout_ns": contract["manual_timeout_ns"],
            "writer_deadline_lead_ns": contract["writer_deadline_lead_ns"],
            "inspections_per_attempt": 1,
            "false_publishable": True,
            "w_ops3_reject_surfaces": contract["w_ops3_reject_surfaces"],
        },
        "scope_guards": {
            "forbidden_import_roots": sorted(FORBIDDEN_IMPORT_ROOTS),
            "isaac": 0,
            "kit": 0,
            "physx": 0,
            "physx_cook_callbacks": 0,
            "AppLauncher": 0,
            "SimulationContext": 0,
            "physics_steps": 0,
            "warp": 0,
            "cuda": 0,
            "gpu_compute": 0,
            "usd_read_write": 0,
            "asset_write": 0,
            "robot_hw_serial": 0,
            "lerobot": 0,
            "new_package_install": 0,
            "allowed": [
                "frozen JSON evidence reads",
                "hppfcl/numpy/trimesh offline queries",
                "rerun-sdk import (save-only RRD authoring) (W-OPS2)",
                f"absolute-path rerun CLI subprocess ({RERUN_CLI}) for verify/RBL/screenshot only (W-OPS2)",
                "writes under claudedocs/runtime_logs/grasp_track/g0a_d409/ only",
            ],
        },
        "runtime_overlay_contract": {
            "allowed_dirty_paths": allowed_dirty,
            **_future_runtime_paths(worker_iface, contract),
        },
        "interpretation_boundary": [
            "stable grasp / force closure / grasp feasibility / grasp success / push-over-absence guarantee / contact-order dynamics / SDF superiority / transfer to other cylinders or placements: all null claims (section 2.14)",
            "A-and-B does NOT exclude the D362 push-over pose (d_fix 4.2727mm - inner-4-mask min measured); the ordering constraint is a pure geometric descriptor and its push-over screening power is unverified (null) (P3)",
            "part-level masks cannot distinguish face-level inner/outer (outer 16 subset of inner 17, shared carrier 16; difference = part_035 alone); inner-17 membership is a necessary-condition judgment (W-FRZ1)",
            "geometry-only labels must not be promoted to standalone training (direction decision 2)",
            "mass 24.83g and friction are unused by this harness",
            "g0a_pass=false unchanged; D399 remains reserved for D398-F1",
        ],
        "actual_execution_requires_separate_tuple_sha_approval": True,
    }
    return prereg


# ---------------------------------------------------------------------------
# Registered-contract auditor (used on the real prereg AND tampered copies).
# ---------------------------------------------------------------------------

def _audit_registered(prereg: dict[str, Any]) -> None:
    gates = prereg["gates"]
    classifier = gates["barrel_cap_classifier"]
    if classifier.get("strict") is not True:
        raise D409Error("audit: barrel/cap classifier is not strict")
    for key, value in classifier.items():
        if "tolerance" in key.lower():
            raise D409Error(f"audit: barrel/cap tolerance registration forbidden: {key}={value}")
    if not isinstance(classifier.get("rule"), str) or "no new geometric success tolerance" not in classifier["rule"]:
        raise D409Error("audit: barrel/cap strict rule text missing")
    bracket = float(gates["q5_bisection"]["bracket_rad"])
    if bracket > 1.0e-6:
        raise D409Error(f"audit: bisection bracket exceeds 1e-6 rad: {bracket}")
    scoring = gates["region_scoring"]
    if scoring.get("no_standalone_36p033_gate") is not True:
        raise D409Error("audit: 36.033 standalone-gate flag violated")
    if scoring.get("historical_proxy", {}).get("standalone_gate") not in (None, False):
        raise D409Error("audit: 36.033 promoted to a standalone gate")
    forbidden = set(prereg["scope_guards"]["forbidden_import_roots"])
    for required_root in (
        "isaac",
        "isaacgym",
        "isaacsim",
        "omni",
        "pxr",
        "carb",
        "kit",
        "physx",
        "physxcooking",
        "warp",
        "cuda",
        "cupy",
        "torch",
        "serial",
    ):
        if required_root not in forbidden:
            raise D409Error(f"audit: scope guard missing forbidden root: {required_root}")
    # OPS-B1 repair R2: registered_metrics is a D371-lineage REQUIRED field.
    metrics = prereg.get("registered_metrics")
    if not isinstance(metrics, dict):
        raise D409Error("audit: registered_metrics missing (D371 lineage field)")
    for metrics_key in ("per_pose_csv_columns", "aggregate_counts", "region_entry_metrics"):
        value = metrics.get(metrics_key)
        if not isinstance(value, list) or not value:
            raise D409Error(f"audit: registered_metrics.{metrics_key} missing or empty")
    contract = prereg["determinism_run_contract"]
    if (
        contract.get("worker_invocations_total") != 2
        or contract.get("automatic_retries") != 0
        or contract.get("per_run_out_dirs") != ["run1", "run2"]
        or contract.get("run2_precondition") != "run1_preclose_pass"
    ):
        raise D409Error("audit: determinism_run_contract pins drifted")
    if prereg.get("actual_execution_requires_separate_tuple_sha_approval") is not True:
        raise D409Error("audit: approval boundary flag missing")


# ---------------------------------------------------------------------------
# Static fixtures (section 2.11 confirmed + W-LES4).
# ---------------------------------------------------------------------------

def _pi2_variant_joints(joints: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """roarm_kinematics._CHAIN constant series (anchor-tool port): rpy 1.5708
    literals -> sign-matched pi/2 symbols; link1_to_link2 origin z -> 0.05196."""
    out: dict[str, dict[str, Any]] = {}
    for name, spec in joints.items():
        rpy = [
            math.copysign(math.pi / 2.0, value) if abs(abs(value) - 1.5708) < 1e-9 else value
            for value in spec["rpy"]
        ]
        xyz = list(spec["xyz"])
        if name == "link1_to_link2":
            xyz = [xyz[0], xyz[1], 0.05196]
        out[name] = {
            "type": spec["type"],
            "xyz": xyz,
            "rpy": rpy,
            "axis": list(spec["axis"]),
            "limits_rad": spec["limits_rad"],
        }
    return out


def _fixture_prepare_negative_controls(
    module: Any, hppfcl: Any, frozen: dict[str, Any], prereg: dict[str, Any], static_prep: dict[str, Any]
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # N1 — cylinder radius tamper -> real-geometry pin check FAIL.
    geometry = prereg["geometry"]

    def _geometry_pin_check(radius_m: float, height_m: float) -> bool:
        return (
            repr(radius_m) == geometry["radius_m_repr"]
            and repr(height_m) == geometry["height_m_repr"]
        )

    n1_pass_positive = _geometry_pin_check(module.CYL_RADIUS_M, module.CYL_HEIGHT_M)
    n1_fail_tampered = not _geometry_pin_check(0.017, module.CYL_HEIGHT_M)
    results["N1_radius_tamper"] = {
        "positive_untampered_pass": n1_pass_positive,
        "tampered_0p017_rejected": n1_fail_tampered,
        "runtime_enforcement_note": (
            "this fixture demonstrates the pin discriminator locally; the "
            "RUNTIME enforcement paths are (a) the worker-source sha binding "
            "(admission row + tuple) and (b) run_runtime's geometry repr "
            "re-check vs prereg (OPS-W4 repair R12)"
        ),
        "pass": bool(n1_pass_positive and n1_fail_tampered),
    }

    # N2 — 1-bit vertex-stream tamper -> D409-canonical hash FAIL vs S1 pin;
    # 1-byte file tamper -> d348 file-level sha FAIL (C3).
    d348_raw = (PROJECT_ROOT / D348_REL).read_bytes()
    d348 = json.loads(d348_raw)
    row = next(r for r in d348["rows"] if r["body"] == "gripper_link")
    name = row["name"]
    s1_pin = static_prep["payload"]["s1_d348_integrity"]["gripper_link"][
        "d409_canonical_geometry_sha256"
    ][name]
    untampered_hash = module._canonical_part_hash(
        name, row["instance"]["vertices_m"], row["instance"]["topology_triangles"]
    )
    tampered_vertices = [list(vertex) for vertex in row["instance"]["vertices_m"]]
    packed = bytearray(struct.pack("<d", float(tampered_vertices[0][0])))
    packed[0] ^= 0x01
    tampered_vertices[0][0] = struct.unpack("<d", bytes(packed))[0]
    tampered_hash = module._canonical_part_hash(
        name, tampered_vertices, row["instance"]["topology_triangles"]
    )
    flipped_file = bytearray(d348_raw)
    flipped_file[100] ^= 0x01
    file_pin = prereg["input_hashes"]["d348_callback_topology_volume_evidence"]
    results["N2_payload_tamper"] = {
        "part": name,
        "untampered_matches_s1_pin": untampered_hash == s1_pin,
        "vertex_bitflip_hash_rejected": tampered_hash != s1_pin,
        "file_byteflip_sha_rejected": _sha_bytes(bytes(flipped_file)) != file_pin,
        "payload_sha256_note": "stored payload_sha256 preimage is opaque; recorded verbatim; executable basis = D409-canonical hash (S1) + file sha (C3)",
        "pass": bool(
            untampered_hash == s1_pin
            and tampered_hash != s1_pin
            and _sha_bytes(bytes(flipped_file)) != file_pin
        ),
    }

    # N3 — inner mask replaced by outer 16 -> name-set/count FAIL.
    inner = frozen["masks"]["gripper_inner"]
    outer = frozen["masks"]["gripper_outer"]
    tampered_count_check = len(outer) == module.GRIPPER_INNER_EXPECTED_COUNT
    tampered_nameset_check = sorted(outer) == sorted(inner)
    difference = sorted(set(inner) - set(outer))
    results["N3_mask_swap"] = {
        "tampered_inner_count_check_failed": not tampered_count_check,
        "tampered_name_set_check_failed": not tampered_nameset_check,
        "decisive_difference": difference,
        "decisive_difference_is_part_035_only": difference == [module.OUTER_DIFF_PART],
        "name_set_sha_inner": _sha_bytes(_canonical_bytes(sorted(inner))),
        "name_set_sha_outer": _sha_bytes(_canonical_bytes(sorted(outer))),
        "pass": bool(
            not tampered_count_check
            and not tampered_nameset_check
            and difference == [module.OUTER_DIFF_PART]
        ),
    }

    # N4 — pi/2 FK substitution -> anchor gate ANY-reject must FIRE.
    tampered_frozen = dict(frozen)
    tampered_frozen["joints"] = _pi2_variant_joints(frozen["joints"])
    fired = False
    fire_message = None
    try:
        module._anchor_gate(hppfcl, tampered_frozen, {"pose": 0, "total": 0})
    except module._ContractFail as exc:
        fired = "anchor gate ANY-reject fired" in str(exc)
        fire_message = str(exc)[:300]
    results["N4_pi2_fk_substitution"] = {
        "anchor_gate_fired": fired,
        "message": fire_message,
        "pass": bool(fired),
    }
    if not fired:
        raise D409Error(
            "N4: pi/2 substitution did NOT fire the anchor gate - discrimination lost, "
            "control set must be redesigned (section 2.11-4)"
        )

    # N5 — empty grid -> worker static-pin admission FAIL (C4).
    saved_radials = module.RADIALS_UM
    saved_tangents = module.TANGENTS_UM
    empty_rejected = False
    try:
        module.RADIALS_UM = ()
        module.TANGENTS_UM = ()
        try:
            module._verify_static_pins()
        except module._ContractFail:
            empty_rejected = True
    finally:
        module.RADIALS_UM = saved_radials
        module.TANGENTS_UM = saved_tangents
    positive_pins = module._verify_static_pins()
    results["N5_empty_grid"] = {
        "empty_grid_rejected": empty_rejected,
        "restored_pins_pass": bool(positive_pins),
        "pass": bool(empty_rejected and positive_pins),
    }
    return results


def _fixture_audit_negative_controls(prereg: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    _audit_registered(prereg)
    results["untampered_prereg_audit"] = {"pass": True}

    def _tampered_copy() -> dict[str, Any]:
        return json.loads(json.dumps(prereg))

    # A1 — tolerance in barrel/cap registration.
    copy_a1 = _tampered_copy()
    copy_a1["gates"]["barrel_cap_classifier"]["tolerance_mm"] = 0.5
    rejected_a1 = False
    try:
        _audit_registered(copy_a1)
    except D409Error:
        rejected_a1 = True
    results["A1_barrel_cap_tolerance"] = {"rejected": rejected_a1, "pass": rejected_a1}

    # A2 — bisection bracket loosened.
    copy_a2 = _tampered_copy()
    copy_a2["gates"]["q5_bisection"]["bracket_rad"] = 1.0e-5
    rejected_a2 = False
    try:
        _audit_registered(copy_a2)
    except D409Error:
        rejected_a2 = True
    results["A2_bisect_bracket"] = {"rejected": rejected_a2, "pass": rejected_a2}

    # A3 — 36.033 standalone gate.
    copy_a3 = _tampered_copy()
    copy_a3["gates"]["region_scoring"]["historical_proxy"]["standalone_gate"] = True
    rejected_a3 = False
    try:
        _audit_registered(copy_a3)
    except D409Error:
        rejected_a3 = True
    results["A3_36p033_standalone_gate"] = {"rejected": rejected_a3, "pass": rejected_a3}

    # A4 — live isaac-family import attempt -> scope guard rejects.
    rejected_a4 = False
    try:
        importlib.import_module("omni")
    except ImportError:
        rejected_a4 = True
    registered_a4 = "omni" in set(prereg["scope_guards"]["forbidden_import_roots"])
    results["A4_isaac_import"] = {
        "live_import_rejected": rejected_a4,
        "registered_in_forbidden_list": registered_a4,
        "pass": bool(rejected_a4 and registered_a4),
    }
    if not all(entry["pass"] for entry in results.values()):
        raise D409Error(f"audit_registered negative controls failed: {results}")
    return results


def _fixture_positive_controls(
    module: Any, hppfcl: Any, frozen: dict[str, Any]
) -> dict[str, Any]:
    budget = {"pose": 0, "total": 0}
    gate = module._anchor_gate(hppfcl, frozen, budget)
    grid_key_present = (
        module.POSITIVE_CONTROL_KEY_UM[0] in module.RADIALS_UM
        and module.POSITIVE_CONTROL_KEY_UM[1] in module.TANGENTS_UM
    )
    results = {
        "P1_anchor_reproduction": {
            "channels_mm": gate["channels"],
            "threshold_mm": gate["threshold_mm"],
            "queries_used": budget["total"],
            "pass": bool(gate["pass"]),
        },
        "P2_grid_key": {
            "key_um": list(module.POSITIVE_CONTROL_KEY_UM),
            "present": grid_key_present,
            "meaning_note": "meaning differs from the frozen candidate (new center, real cylinder)",
            "pass": bool(grid_key_present),
        },
    }
    if not all(entry["pass"] for entry in results.values()):
        raise D409Error(f"positive controls failed: {results}")
    return results


def _fixture_equivalence_d335(module: Any) -> dict[str, Any]:
    """W-LES4: replay the redefined admission formulas on stored d335 rows and
    require check-level agreement with the original legacy_checks (C5)."""
    csv_path = PROJECT_ROOT / D335_SCAN_CSV_REL
    csv_sha = _sha_path(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stage"] == "coarse"]
    rows.sort(key=lambda row: (float(row["radial_offset_mm"]), float(row["tangent_offset_mm"])))
    count = len(rows)
    if count == 0:
        raise D409Error("d335 coarse rows missing")
    indices = sorted({0, count // 4, count // 2, (3 * count) // 4, count - 1})
    selected = [rows[index] for index in indices]
    target = next(
        (
            row
            for row in rows
            if float(row["radial_offset_mm"]) == 7.0 and float(row["tangent_offset_mm"]) == 11.0
        ),
        None,
    )
    if target is None:
        raise D409Error("d335 (7.0, 11.0) coarse row not found")
    if target not in selected:
        selected.append(target)

    replayed = []
    all_pass = True
    for row in selected:
        radial = float(row["radial_offset_mm"])
        tangent = float(row["tangent_offset_mm"])
        tcp = float(row["commanded_tcp_error_mm"])
        jaw = float(row["jaw_tangent_error_deg"])
        gap = float(row["fixed_jaw_face_gap_mm"])
        penetration = float(row["fixed_jaw_penetration_mm"])
        below_top = float(row["contact_point_below_top_mm"])
        tip = float(row["radial_tip_past_near_face_mm"])
        link5_raw = float(row["link5_raw_signed_distance_mm"])
        gripper_raw = float(row["gripper_link_raw_signed_distance_mm"])
        stored_legacy = row["legacy_alignment_pass"] == "True"
        stored_raw_clear = row["raw_tool_clear_pass"] == "True"
        stored_ik = row["ik_converged"] == "True"

        reconstructed = {
            "ik_converged": stored_ik,
            "commanded_tcp_error_le_5mm": tcp <= 5.0,
            "jaw_tangent_le_15deg": jaw <= 15.0,
            "fixed_jaw_gap_0_to_5mm": D335_LEGACY_GAP_BAND_MM[0] <= gap <= D335_LEGACY_GAP_BAND_MM[1],
            "no_fixed_jaw_proxy_penetration": penetration <= EQUIV_ANTI_RETREAT_EPS_MM,
            "contact_at_least_15mm_below_top": below_top >= D335_TOP_RULE_MM,
            "anti_retreat_nonnegative": tip >= -EQUIV_ANTI_RETREAT_EPS_MM,
        }
        redefined = {
            "ik_converged": stored_ik,
            "commanded_tcp_error_le_5mm": tcp <= module.TCP_GATE_MM,
            "jaw_tangent_le_15deg": jaw <= module.JAW_TANGENT_GATE_DEG,
            "fixed_jaw_gap_0_to_5mm": 0.0 <= gap <= module.FIXED_JAW_BAND_MM[1],
            "no_fixed_jaw_proxy_penetration": penetration <= EQUIV_ANTI_RETREAT_EPS_MM,
            "contact_at_least_15mm_below_top": below_top >= D335_TOP_RULE_MM,
            "anti_retreat_nonnegative": (D335_CYLINDER_RADIUS_MM - radial) >= -EQUIV_ANTI_RETREAT_EPS_MM,
        }
        anti_retreat_recompute_delta = abs((D335_CYLINDER_RADIUS_MM - radial) - tip)
        agreements = {key: reconstructed[key] == redefined[key] for key in reconstructed}
        clear_replay = (link5_raw >= module.CLEAR_GATE_MM) and (gripper_raw >= module.CLEAR_GATE_MM)
        and_of_seven = all(reconstructed.values())
        row_pass = (
            all(agreements.values())
            and anti_retreat_recompute_delta <= EQUIV_NUMERIC_TOL_MM
            and clear_replay == stored_raw_clear
            and and_of_seven == stored_legacy
        )
        all_pass = all_pass and row_pass
        replayed.append(
            {
                "key_mm": [radial, tangent],
                "reconstructed": reconstructed,
                "redefined_replay": redefined,
                "check_agreements": agreements,
                "anti_retreat_recompute_delta_mm": anti_retreat_recompute_delta,
                "clear_gate_replay": clear_replay,
                "stored_raw_tool_clear_pass": stored_raw_clear,
                "and_of_seven_reconstructed": and_of_seven,
                "stored_legacy_alignment_pass": stored_legacy,
                "row_pass": row_pass,
            }
        )
    result = {
        "source_csv": D335_SCAN_CSV_REL,
        "source_csv_sha256": csv_sha,
        "coarse_row_count": count,
        "selected_keys_mm": [entry["key_mm"] for entry in replayed],
        "not_reconstructible_disclosed": "live_exact_written_tcp_error_le_5mm (C5)",
        "tolerances": {
            "numeric_recompute_mm": EQUIV_NUMERIC_TOL_MM,
            "anti_retreat_eps_mm": EQUIV_ANTI_RETREAT_EPS_MM,
        },
        "rows": replayed,
        "pass": bool(all_pass),
    }
    if not all_pass:
        raise D409Error("W-LES4 equivalence fixture failed (check-level disagreement)")
    return result


# ---------------------------------------------------------------------------
# Static-prep mode.
# ---------------------------------------------------------------------------

_WRITER_MODULE_CACHE_FIXTURE: Any = None


def _load_writer_module() -> Any:
    """Import the manual writer for in-process fixture exercise (G1/R4).
    The writer has a __main__ guard; import only defines functions and adds
    a second scope-guard finder instance (same forbidden roots)."""
    global _WRITER_MODULE_CACHE_FIXTURE
    if _WRITER_MODULE_CACHE_FIXTURE is not None:
        return _WRITER_MODULE_CACHE_FIXTURE
    spec = importlib.util.spec_from_file_location("d409_manual_writer_fixture", WRITER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["d409_manual_writer_fixture"] = module
    spec.loader.exec_module(module)
    _WRITER_MODULE_CACHE_FIXTURE = module
    return module


def _fixture_w_ops3_reject_surfaces(contract: dict[str, Any]) -> dict[str, Any]:
    """G1 repair R4: per-surface tamper fixtures for the 9 W-OPS3 reject
    surfaces.  Authority list = the writer contract's w_ops3_reject_surfaces
    enumeration (consumer-derived; OPS lens instruction).  Surfaces whose
    semantics are filesystem/runtime-bound are verified at the AST level
    (enforcing call presence); the method is disclosed per surface."""
    writer_module = _load_writer_module()
    writer_source = WRITER_PATH.read_text(encoding="utf-8")
    surfaces = contract["w_ops3_reject_surfaces"]
    results: dict[str, Any] = {}

    def _raises_protocol(callable_fn: Any) -> bool:
        try:
            callable_fn()
        except writer_module.ProtocolError:
            return True
        except Exception:
            return False
        return False

    # 1. nonce_hmac_envelope [behavioral]: tampered body -> different HMAC;
    #    constant-time compare in source.
    nonce = b"\x01" * 32
    mac_ok = writer_module._hmac_hex(nonce, {"op": "ping", "value": 1})
    mac_tampered = writer_module._hmac_hex(nonce, {"op": "ping", "value": 2})
    results["nonce_hmac_envelope"] = {
        "method": "behavioral",
        "tampered_body_changes_mac": mac_ok != mac_tampered,
        "constant_time_compare_in_source": "hmac.compare_digest" in writer_source,
        "pass": bool(mac_ok != mac_tampered and "hmac.compare_digest" in writer_source),
    }

    # 2. eleven_field_false_publishable_schema [behavioral].
    all_false = {name: False for name in writer_module.REQUIRED_BOOLEAN_FIELDS}
    normalized, _notes = writer_module._validate_manual_input(
        {"required_fields": dict(all_false), "notes": "fixture"}
    )
    missing_one = dict(all_false)
    missing_one.pop(next(iter(missing_one)))
    extra_field = dict(all_false)
    extra_field["fixture_extra_field"] = True
    non_bool = dict(all_false)
    non_bool[next(iter(non_bool))] = "true"
    results["eleven_field_false_publishable_schema"] = {
        "method": "behavioral",
        "field_count": len(writer_module.REQUIRED_BOOLEAN_FIELDS),
        "all_false_publishable": normalized == all_false,
        "missing_field_rejected": _raises_protocol(
            lambda: writer_module._validate_manual_input(
                {"required_fields": missing_one, "notes": ""}
            )
        ),
        "extra_field_rejected": _raises_protocol(
            lambda: writer_module._validate_manual_input(
                {"required_fields": extra_field, "notes": ""}
            )
        ),
        "non_boolean_rejected": _raises_protocol(
            lambda: writer_module._validate_manual_input(
                {"required_fields": non_bool, "notes": ""}
            )
        ),
        "pass": bool(
            len(writer_module.REQUIRED_BOOLEAN_FIELDS) == 11
            and normalized == all_false
            and _raises_protocol(
                lambda: writer_module._validate_manual_input(
                    {"required_fields": missing_one, "notes": ""}
                )
            )
            and _raises_protocol(
                lambda: writer_module._validate_manual_input(
                    {"required_fields": extra_field, "notes": ""}
                )
            )
            and _raises_protocol(
                lambda: writer_module._validate_manual_input(
                    {"required_fields": non_bool, "notes": ""}
                )
            )
        ),
    }

    # 3. source_screenshot_manifest [behavioral-partial]: PNG tamper +
    #    exact-key schema tamper; sha re-hash loop presence (R15) in source.
    bad_png_rejected = _raises_protocol(
        lambda: writer_module._png_dimensions(b"NOTAPNG" + b"\x00" * 64, "fixture")
    )
    bad_keys_rejected = _raises_protocol(
        lambda: writer_module._expect_exact_keys(
            {"unexpected": 1}, {"rbl_path", "rrd_path"}, "fixture"
        )
    )
    results["source_screenshot_manifest"] = {
        "method": "behavioral+ast",
        "corrupt_png_rejected": bad_png_rejected,
        "wrong_schema_rejected": bad_keys_rejected,
        "rrd_report_rehash_in_source": "does not match file bytes" in writer_source,
        "pass": bool(
            bad_png_rejected
            and bad_keys_rejected
            and "does not match file bytes" in writer_source
        ),
    }

    # 4. worst_case_traversal_budget [behavioral-partial]: byte-limit
    #    enforcement on a real read + O_NOFOLLOW in the traversal helper.
    budget_root_fd = os.open(ATTEMPT_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        byte_limit_rejected = _raises_protocol(
            lambda: writer_module._secure_read_relative(
                budget_root_fd, PREREG_PATH.name, 16
            )
        )
    finally:
        os.close(budget_root_fd)
    results["worst_case_traversal_budget"] = {
        "method": "behavioral+ast",
        "byte_limit_rejected": byte_limit_rejected,
        "o_nofollow_in_source": "O_NOFOLLOW" in writer_source,
        "pass": bool(byte_limit_rejected and "O_NOFOLLOW" in writer_source),
    }

    # 5-9. AST-enforcement surfaces (filesystem/runtime-bound semantics).
    # OPS2-W1 repair: real AST-node inspection of the ENFORCING constructs.
    # Contract strings, docstrings and constant Assign statements cannot
    # satisfy these predicates — deleting an enforcement block removes its
    # node shape and fails the surface (unlike the earlier substring
    # needles, which stayed satisfied via descriptive text).
    writer_ast = ast.parse(writer_source)
    raising_ifs = [
        node
        for node in ast.walk(writer_ast)
        if isinstance(node, ast.If)
        and any(isinstance(sub, ast.Raise) for sub in ast.walk(node))
    ]

    def _test_compares(if_node: ast.If) -> list[ast.Compare]:
        return [n for n in ast.walk(if_node.test) if isinstance(n, ast.Compare)]

    def _sides(compare: ast.Compare) -> list[ast.expr]:
        return [compare.left, *compare.comparators]

    def _any_raising_if_compare(predicate: Any) -> bool:
        return any(
            predicate(cmp) for if_node in raising_ifs for cmp in _test_compares(if_node)
        )

    def _is_call_attr(node: ast.expr, owner: str, attr: str) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == owner
        )

    protocol_raise_args = [
        node.exc.args[0]
        for node in ast.walk(writer_ast)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "ProtocolError"
        and node.exc.args
    ]

    ast_surface_checks: dict[str, dict[str, bool]] = {
        "controller_writer_identity": {
            "getppid_neq_controller_pid_guard": _any_raising_if_compare(
                lambda cmp: any(_is_call_attr(s, "os", "getppid") for s in _sides(cmp))
                and any(isinstance(op, ast.NotEq) for op in cmp.ops)
            ),
            "proc_start_ticks_neq_guard": _any_raising_if_compare(
                lambda cmp: any(
                    isinstance(s, ast.Call)
                    and isinstance(s.func, ast.Name)
                    and s.func.id == "_proc_start_ticks"
                    for s in _sides(cmp)
                )
                and any(isinstance(op, ast.NotEq) for op in cmp.ops)
            ),
        },
        "tuple_sha_binding": {
            "tuple_file_sha_mismatch_raise": any(
                isinstance(arg, ast.Constant)
                and arg.value == "approved tuple-file SHA mismatch"
                for arg in protocol_raise_args
            ),
            "tuple_key_mismatch_fstring_raise": any(
                isinstance(arg, ast.JoinedStr)
                and "".join(
                    part.value
                    for part in arg.values
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                )
                == "tuple  mismatch"
                for arg in protocol_raise_args
            ),
        },
        "common_monotonic_deadline": {
            "monotonic_ns_deadline_guard": _any_raising_if_compare(
                lambda cmp: any(
                    _is_call_attr(s, "time", "monotonic_ns") for s in _sides(cmp)
                )
            ),
            "writer_deadline_lead_arithmetic_guard": _any_raising_if_compare(
                lambda cmp: any(
                    isinstance(s, ast.BinOp)
                    and isinstance(s.op, ast.Sub)
                    and isinstance(s.right, ast.Name)
                    and s.right.id == "WRITER_DEADLINE_LEAD_NS"
                    for s in _sides(cmp)
                )
            ),
        },
        "exclusive_create_no_replace_fsync": {
            "rename_noreplace_call_argument": any(
                isinstance(node, ast.Call)
                and any(
                    isinstance(arg, ast.Name) and arg.id == "RENAME_NOREPLACE"
                    for arg in node.args
                )
                for node in ast.walk(writer_ast)
            ),
            "os_fsync_file_and_directory": (
                sum(
                    1
                    for node in ast.walk(writer_ast)
                    if _is_call_attr(node, "os", "fsync")
                )
                >= 2
            ),
            "os_o_excl_load": any(
                isinstance(node, ast.Attribute)
                and node.attr == "O_EXCL"
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                for node in ast.walk(writer_ast)
            ),
        },
        "no_writer_creation_or_repair_during_run": {
            "controller_started_arm_compare_guard": _any_raising_if_compare(
                lambda cmp: any(
                    isinstance(s, ast.Constant) and s.value == "controller_started"
                    for s in _sides(cmp)
                )
            ),
            "writer_source_sha_selfcheck_guard": _any_raising_if_compare(
                lambda cmp: any(
                    isinstance(s, ast.Call)
                    and isinstance(s.func, ast.Name)
                    and s.func.id == "_sha_path"
                    and any(
                        isinstance(a, ast.Name) and a.id == "WRITER_PATH"
                        for a in s.args
                    )
                    for s in _sides(cmp)
                )
            ),
        },
    }
    for surface_name, checks in ast_surface_checks.items():
        results[surface_name] = {
            "method": "ast",
            "enforcing_ast_nodes": dict(checks),
            "pass": all(checks.values()),
        }

    surface_names = sorted(surfaces.keys())
    covered = sorted(results.keys())
    results["_coverage"] = {
        "contract_surfaces": surface_names,
        "fixture_surfaces": covered,
        "all_contract_surfaces_covered": set(surface_names) <= set(covered),
        "pass": set(surface_names) <= set(covered),
    }
    return results


def run_static_prep() -> int:
    environment = _environment_gate()
    worker_iface = _derive_worker_interface()
    contract = _writer_contract()
    ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in STATIC_ARTIFACT_PATHS:
        if path.exists():
            raise D409Error(
                f"static artifact already exists (candidate overwrite forbidden, D407): {path}"
            )

    prereg = _build_prereg(worker_iface, contract, environment)
    _audit_registered(prereg)
    prereg_sha = _write_json_x(PREREG_PATH, prereg)

    import hppfcl  # allowed offline query library (after scope guard)

    module = _load_worker_module()
    static_prep = _load_static_prep()
    frozen = module._load_frozen_inputs(hppfcl)
    # G1/R4 + NOTE4 fail-closed gate, OPS2-W2 repair: each fixture group is
    # try-wrapped so an INTERNAL fixture raise still leaves a
    # STATIC_FAIL_STATUS results file (evidence preserved) before the error
    # propagates; any failure stops BEFORE the attestation/tuple are
    # authored (the attempt namespace is consumed by the already-written
    # prereg/static results, but no approvable tuple exists).
    fixture_specs: list[tuple[str, Any]] = [
        (
            "prepare_negative",
            lambda: _fixture_prepare_negative_controls(
                module, hppfcl, frozen, prereg, static_prep
            ),
        ),
        ("audit_negative", lambda: _fixture_audit_negative_controls(prereg)),
        ("positive", lambda: _fixture_positive_controls(module, hppfcl, frozen)),
        ("equivalence_d335", lambda: _fixture_equivalence_d335(module)),
        ("w_ops3_reject", lambda: _fixture_w_ops3_reject_surfaces(contract)),
    ]
    fixtures: dict[str, Any] = {}
    fixture_raise: dict[str, str] | None = None
    fixture_exc: Exception | None = None
    for group_name, thunk in fixture_specs:
        try:
            fixtures[group_name] = thunk()
        except Exception as exc:
            fixture_raise = {
                "group": group_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            fixture_exc = exc
            break
    fixture_failures = []
    for group_name, group in fixtures.items():
        if group_name == "equivalence_d335":
            if group.get("pass") is not True:
                fixture_failures.append(group_name)
            continue
        for entry_name, entry in group.items():
            if isinstance(entry, dict) and entry.get("pass") is not True:
                fixture_failures.append(f"{group_name}.{entry_name}")
    if fixture_raise is not None:
        fixture_failures.append(
            f"{fixture_raise['group']}.raised:{fixture_raise['error_type']}"
        )
    all_fixtures_pass = not fixture_failures
    static_results = {
        "artifact": STATIC_RESULTS_ARTIFACT,
        "case": "g0a_d409",
        "created_utc": _utc_now(),
        "preregistration_sha256": prereg_sha,
        "environment": environment,
        "fixtures": fixtures,
        "fixture_failures": fixture_failures,
        "fixture_raise": fixture_raise,
        "isaac_executed": 0,
        "physics_steps": 0,
        "status": STATIC_PASS_STATUS if all_fixtures_pass else STATIC_FAIL_STATUS,
    }
    static_sha = _write_json_x(STATIC_RESULTS_PATH, static_results)
    if fixture_exc is not None:
        raise D409Error(
            f"static fixture raised (fail-closed, STATIC_FAIL recorded): {fixture_raise}"
        ) from fixture_exc
    if not all_fixtures_pass:
        raise D409Error(
            f"static fixtures failed (fail-closed, no attestation/tuple): {fixture_failures}"
        )

    controller_sha = _sha_path(CONTROLLER_PATH)
    worker_sha = worker_iface["WORKER_SOURCE_SHA256"]
    writer_sha = _sha_path(WRITER_PATH)
    for path, label in (
        (CONTROLLER_PATH, "controller"),
        (WORKER_PATH, "worker"),
        (WRITER_PATH, "manual_writer"),
        (PREREG_PATH, "preregistration"),
        (STATIC_RESULTS_PATH, "static_results"),
    ):
        _require_regular_nlink1(path, label)
    attestation = {
        "artifact": ATTESTATION_ARTIFACT,
        "case": "g0a_d409",
        "created_utc": _utc_now(),
        "controller_script_path_and_sha256": [_rel(CONTROLLER_PATH), controller_sha],
        "worker_script_path_and_sha256": [_rel(WORKER_PATH), worker_sha],
        "manual_writer_script_path_and_sha256": [_rel(WRITER_PATH), writer_sha],
        "preregistration_sha256": prereg_sha,
        "static_fixture_results_sha256": static_sha,
        "prepare_negative_controls_pass": all(
            entry["pass"] for entry in static_results["fixtures"]["prepare_negative"].values()
        ),
        "audit_negative_controls_pass": all(
            entry["pass"] for entry in static_results["fixtures"]["audit_negative"].values()
        ),
        "positive_controls_pass": all(
            entry["pass"] for entry in static_results["fixtures"]["positive"].values()
        ),
        "equivalence_fixture_pass": static_results["fixtures"]["equivalence_d335"]["pass"],
        "w_ops3_reject_fixture_pass": all(
            entry["pass"]
            for entry in static_results["fixtures"]["w_ops3_reject"].values()
            if isinstance(entry, dict)
        ),
        "nlink1_verified": True,
        "implementation_static_attestation_pass": True,
        "runtime_boundary": (
            "attempt1 runtime requires the user to cite the tuple sha256 in an explicit "
            "approval; static prep changes nothing scientific (g0a_pass=false unchanged)"
        ),
    }
    attestation_sha = _write_json_x(ATTESTATION_PATH, attestation)

    tuple_document = {
        "artifact": TUPLE_ARTIFACT,
        "case": "g0a_d409",
        "created_utc": _utc_now(),
        "approval_boundary": "actual_execution_requires_separate_tuple_sha_approval",
        "attestation_sha256": attestation_sha,
        "hashes": {
            "preregistration_sha256": prereg_sha,
            "controller_sha256": controller_sha,
            "worker_sha256": worker_sha,
            "manual_writer_sha256": writer_sha,
        },
    }
    _write_json_x(TUPLE_PATH, tuple_document)
    tuple_sha = _sha_path(TUPLE_PATH)

    print(f"D409 static prep status {STATIC_PASS_STATUS}")
    print(f"D409 preregistration sha256 {prereg_sha}")
    print(f"D409 static fixture results sha256 {static_sha}")
    print(f"D409 attestation sha256 {attestation_sha}")
    print(f"D409 proposed runtime hash tuple sha256 {tuple_sha}")
    print(
        "D409 STOP: runtime attempt1 requires explicit user approval citing the tuple "
        "sha256 above; then run --mode runtime --approved-tuple-sha256 <sha>."
    )
    return 0
# ---------------------------------------------------------------------------
# Phase log (hash-chained; schema consumed by the writer's _read_phase_chain).
# ---------------------------------------------------------------------------

class PhaseLog:
    def __init__(self, root_fd: int, artifact: str) -> None:
        self.artifact = artifact
        self.fd = os.open(
            PHASE_PATH.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        metadata = os.fstat(self.fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise D409Error("phase log is not an exclusive regular file")
        os.fsync(root_fd)
        self.dev = metadata.st_dev
        self.ino = metadata.st_ino
        self.sequence = 0
        self.previous_sha: str | None = None
        self.latest_row: dict[str, Any] | None = None

    def append(self, event: str, details: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        core = {
            "artifact": self.artifact,
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
            raise D409Error("phase row was not committed by one complete os.write")
        os.fsync(self.fd)
        self.previous_sha = row_sha
        self.latest_row = row
        return row

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


# ---------------------------------------------------------------------------
# Manual-writer orchestration (arm / ping / publish; writer-source schemas).
# ---------------------------------------------------------------------------

def _recv_json_line(channel: socket.socket, max_bytes: int = 64 * 1024) -> dict[str, Any]:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = channel.recv(4096)
        if not chunk:
            raise D409Error("writer socket closed before a complete JSON line")
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            raise D409Error("writer protocol message exceeds size limit")
        raw = b"".join(chunks)
        newline = raw.find(b"\n")
        if newline >= 0:
            if raw[newline + 1 :]:
                raise D409Error("multiple writer protocol messages in one read")
            value = _strict_json_bytes(raw[: newline + 1])
            if not isinstance(value, dict):
                raise D409Error("writer protocol message is not an object")
            return value


def _send_json_line(channel: socket.socket, value: dict[str, Any]) -> None:
    raw = _canonical_bytes(value)
    if len(raw) > 64 * 1024:
        raise D409Error("outgoing writer protocol message exceeds size limit")
    channel.sendall(raw)


def _hmac_hex(nonce: bytes, body: dict[str, Any]) -> str:
    return hmac.new(nonce, _canonical_bytes(body), hashlib.sha256).hexdigest()


def _expect_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise D409Error(
            f"{label} keys mismatch: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )


def _recv_authenticated(channel: socket.socket, nonce: bytes) -> dict[str, Any]:
    envelope = _recv_json_line(channel)
    _expect_keys(envelope, {"body", "hmac_sha256"}, "writer envelope")
    body = envelope["body"]
    if not isinstance(body, dict):
        raise D409Error("writer envelope body is not an object")
    supplied = envelope["hmac_sha256"]
    if not isinstance(supplied, str) or not hmac.compare_digest(
        supplied, _hmac_hex(nonce, body)
    ):
        raise D409Error("writer envelope HMAC mismatch")
    return body


def _spawn_writer(
    root_fd: int,
    root_dev: int,
    root_ino: int,
    phase: PhaseLog,
    controller_started_row: dict[str, Any],
    approved_tuple_sha256: str,
    prereg_sha: str,
    controller_sha: str,
    worker_sha: str,
    writer_sha: str,
    prearm_hard_deadline_ns: int,
    contract: dict[str, Any],
) -> tuple[subprocess.Popen[bytes], socket.socket, bytes, int]:
    parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    stdout_fd = os.open(
        WRITER_STDOUT_LOG,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    stderr_fd = os.open(
        WRITER_STDERR_LOG,
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
        str(ATTEMPT_ROOT),
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
        "--worker-sha256",
        worker_sha,
        "--writer-sha256",
        writer_sha,
        "--approved-tuple-sha256",
        approved_tuple_sha256,
        "--preregistration-sha256",
        prereg_sha,
        "--phase-dev",
        str(phase.dev),
        "--phase-ino",
        str(phase.ino),
        "--manual-basename",
        contract["manual_basename"],
        "--prearm-hard-deadline-monotonic-ns",
        str(prearm_hard_deadline_ns),
    ]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stdout_fd,
        stderr=stderr_fd,
        pass_fds=(child_socket.fileno(), root_fd),
        start_new_session=True,
        shell=False,
    )
    os.close(stdout_fd)
    os.close(stderr_fd)
    child_socket.close()
    nonce = secrets.token_bytes(32)
    bindings = {
        "approved_tuple_sha256": approved_tuple_sha256,
        "controller_pid": controller_pid,
        "controller_sha256": controller_sha,
        "controller_start_ticks": controller_start,
        "manual_basename": contract["manual_basename"],
        "phase_dev": phase.dev,
        "phase_ino": phase.ino,
        "prearm_hard_deadline_monotonic_ns": prearm_hard_deadline_ns,
        "preregistration_sha256": prereg_sha,
        "root_dev": root_dev,
        "root_ino": root_ino,
        "worker_sha256": worker_sha,
        "writer_sha256": writer_sha,
    }
    parent_socket.settimeout(15.0)
    _send_json_line(parent_socket, {"bindings": bindings, "nonce_hex": nonce.hex(), "op": "arm"})
    body = _recv_authenticated(parent_socket, nonce)
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
        raise D409Error("writer READY identity mismatch")
    if body["writer_sha256"] != writer_sha:
        raise D409Error("writer READY SHA mismatch")
    if body["nonce_sha256"] != _sha_bytes(nonce):
        raise D409Error("writer READY nonce mismatch")
    if (
        body["phase_sequence"] != controller_started_row["sequence"]
        or body["phase_row_sha256"] != controller_started_row["row_sha256"]
    ):
        raise D409Error("writer READY phase binding mismatch")
    writer_start_ticks = int(body["writer_start_ticks"])
    if _proc_start_ticks(process.pid) != writer_start_ticks:
        raise D409Error("writer READY start-ticks binding mismatch")
    return process, parent_socket, nonce, writer_start_ticks


def _ping_writer(
    channel: socket.socket,
    nonce: bytes,
    process: subprocess.Popen[bytes],
    writer_start_ticks: int,
    latest_row: dict[str, Any],
) -> None:
    if process.poll() is not None:
        raise D409Error(f"manual writer exited early: rc={process.returncode}")
    if _proc_start_ticks(process.pid) != writer_start_ticks:
        raise D409Error("manual writer identity drift at ping")
    body = {
        "op": "ping",
        "phase_event": latest_row["event"],
        "phase_row_sha256": latest_row["row_sha256"],
        "phase_sequence": latest_row["sequence"],
    }
    channel.settimeout(30.0)
    _send_json_line(channel, {"body": body, "hmac_sha256": _hmac_hex(nonce, body)})
    pong = _recv_authenticated(channel, nonce)
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
        "writer PONG body",
    )
    if (
        pong["op"] != "pong"
        or pong["phase_row_sha256"] != latest_row["row_sha256"]
        or pong["phase_sequence"] != latest_row["sequence"]
        or pong["writer_pid"] != process.pid
        or pong["writer_start_ticks"] != writer_start_ticks
    ):
        raise D409Error("writer PONG binding mismatch")


def _read_manual_stdin(deadline_monotonic_ns: int, contract: dict[str, Any]) -> dict[str, Any]:
    remaining = (deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000
    if remaining <= 0:
        raise D409Error("manual inspection deadline expired before input")
    ready, _, _ = select.select([sys.stdin], [], [], remaining)
    if not ready:
        raise D409Error("manual inspection stdin timeout (fail-closed, no publish)")
    line = sys.stdin.buffer.readline(MAX_MANUAL_STDIN_BYTES + 1)
    if len(line) > MAX_MANUAL_STDIN_BYTES:
        raise D409Error("manual stdin line exceeds size limit")
    if not line.endswith(b"\n"):
        raise D409Error("manual stdin must contain one complete JSON line")
    value = _strict_json_bytes(line)
    if not isinstance(value, dict):
        raise D409Error("manual input must be an object")
    _expect_keys(value, {"required_fields", "notes"}, "manual input")
    fields = value["required_fields"]
    if not isinstance(fields, dict):
        raise D409Error("required_fields must be an object")
    _expect_keys(fields, set(contract["required_boolean_fields"]), "required_fields")
    for key in contract["required_boolean_fields"]:
        if type(fields[key]) is not bool:
            raise D409Error(f"manual field is not boolean: {key}")
    notes = value["notes"]
    if not isinstance(notes, str) or len(notes.encode("utf-8")) > 4096:
        raise D409Error("notes must be a UTF-8 string of at most 4096 bytes")
    return value


# ---------------------------------------------------------------------------
# Worker run supervision (P4: dual run, retry 0, fail-closed).
# ---------------------------------------------------------------------------

def _run_worker_once(
    run_name: str,
    worker_iface: dict[str, Any],
    prereg: dict[str, Any],
    invocation_index: int,
) -> dict[str, Any]:
    run_dir = ATTEMPT_ROOT / run_name
    if run_dir.exists():
        raise D409Error(f"run directory pre-exists (crash resume fail-closed): {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)
    registered_argv = prereg["registered_worker_command"]["per_run"][run_name]
    argv = [
        str(ISAACLAB_PYTHON),
        "-B",
        str(WORKER_PATH),
        "--out-dir",
        str(run_dir),
        "--prereg",
        str(PREREG_PATH),
    ]
    if argv != registered_argv:
        raise D409Error(f"worker argv differs from registered command: {argv}")
    stdout_path = run_dir / WORKER_STDOUT_BASENAME
    stderr_path = run_dir / WORKER_STDERR_BASENAME
    stdout_fd = os.open(
        stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
    )
    stderr_fd = os.open(
        stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
    )
    started_utc = _utc_now()
    started_monotonic = time.monotonic_ns()
    process = subprocess.Popen(
        argv,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stdout_fd,
        stderr=stderr_fd,
        start_new_session=True,
        shell=False,
    )
    os.close(stdout_fd)
    os.close(stderr_fd)
    try:
        returncode = process.wait(timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=60)
        raise D409Error(f"worker {run_name} exceeded timeout {RUN_TIMEOUT_S}s (fail-closed)")
    elapsed_s = (time.monotonic_ns() - started_monotonic) / 1_000_000_000
    if returncode == worker_iface["EXIT_CLAIM_PREEXIST"]:
        raise D409Error(f"worker {run_name} refused: claim pre-exists (exit 73)")
    if returncode != worker_iface["EXIT_PASS"]:
        raise D409Error(f"worker {run_name} contract fail: exit {returncode}")
    for basename in worker_iface["WORKER_ARTIFACT_BASENAMES"]:
        if not (run_dir / basename).is_file():
            raise D409Error(f"worker {run_name} artifact missing: {basename}")
    sentinel = _strict_json_bytes((run_dir / worker_iface["PRECLOSE_NAME"]).read_bytes())
    if sentinel.get("status") != worker_iface["PRECLOSE_STATUS_LITERAL"]:
        raise D409Error(
            f"worker {run_name} preclose status != {worker_iface['PRECLOSE_STATUS_LITERAL']}: "
            f"{sentinel.get('status')}"
        )
    if sentinel.get("verdict") != worker_iface["VERDICT_COMPLETE"]:
        raise D409Error(f"worker {run_name} verdict mismatch: {sentinel.get('verdict')}")
    evidence_sha = _sha_path(run_dir / worker_iface["EVIDENCE_NAME"])
    csv_sha = _sha_path(run_dir / worker_iface["REGION_CSV_NAME"])
    summary_sha = _sha_path(run_dir / worker_iface["SUMMARY_NAME"])
    for key, observed in (
        ("evidence_sha256", evidence_sha),
        ("region_csv_sha256", csv_sha),
        ("summary_sha256", summary_sha),
    ):
        if sentinel.get(key) != observed:
            raise D409Error(f"worker {run_name} sentinel {key} mismatch: {sentinel.get(key)}")
    supervisor = {
        "artifact": "D409_WORKER_SUPERVISOR_V1",
        "run": run_name,
        "invocation_index": invocation_index,
        "run_invocation_count": 1,
        "automatic_retries": 0,
        "argv": argv,
        "returncode": returncode,
        "started_utc": started_utc,
        "elapsed_s": elapsed_s,
        "timeout_s": RUN_TIMEOUT_S,
        "stdout_sha256": _sha_path(stdout_path),
        "stderr_sha256": _sha_path(stderr_path),
        "evidence_sha256": evidence_sha,
        "region_csv_sha256": csv_sha,
        "summary_sha256": summary_sha,
        "preclose_status": sentinel["status"],
        "verdict": sentinel["verdict"],
    }
    _write_json_x(run_dir / WORKER_SUPERVISOR_BASENAME, supervisor)
    return supervisor


def _byte_compare_and_promote(worker_iface: dict[str, Any]) -> dict[str, Any]:
    members = list(worker_iface["DETERMINISM_BYTE_COMPARE_MEMBERS"])
    comparison = {}
    bit_exact = True
    for member in members:
        run1_bytes = (ATTEMPT_ROOT / "run1" / member).read_bytes()
        run2_bytes = (ATTEMPT_ROOT / "run2" / member).read_bytes()
        equal = run1_bytes == run2_bytes
        comparison[member] = {
            "run1_sha256": _sha_bytes(run1_bytes),
            "run2_sha256": _sha_bytes(run2_bytes),
            "bit_exact": equal,
        }
        bit_exact = bit_exact and equal
    if not bit_exact:
        raise D409Error(f"determinism byte compare FAILED (attempt consumed): {comparison}")
    run1_summary = _strict_json_bytes(
        (ATTEMPT_ROOT / "run1" / worker_iface["SUMMARY_NAME"]).read_bytes()
    )
    promotion = {
        "artifact": PROMOTION_ARTIFACT,
        "case": "g0a_d409",
        "canonical_run": "run1",
        "bit_exact": True,
        "byte_compare_members": comparison,
        "verdict": run1_summary["verdict"],
        "canonical_evidence_sha256": comparison[worker_iface["EVIDENCE_NAME"]]["run1_sha256"],
        "region_map_csv_sha256": comparison[worker_iface["REGION_CSV_NAME"]]["run1_sha256"],
        "counts": run1_summary.get("counts"),
        "total_queries": run1_summary.get("total_queries"),
        "published_utc": _utc_now(),
        "phase_order_note": "published BEFORE any presentation artifact (W-LES3)",
    }
    _write_json_x(CANONICAL_PROMOTION_PATH, promotion)
    print(f"D409 canonical evidence sha256 {promotion['canonical_evidence_sha256']}")
    print(f"D409 region csv sha256 {promotion['region_map_csv_sha256']}")
    print(f"D409 verdict {promotion['verdict']}")
    return promotion


# ---------------------------------------------------------------------------
# PNG encode/decode (stdlib zlib/struct; C6/C7) + 5x7 bitmap font.
# ---------------------------------------------------------------------------

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _png_encode_rgb(array: np.ndarray) -> bytes:
    height, width, channels = array.shape
    if channels != 3 or array.dtype != np.uint8:
        raise D409Error("PNG encoder expects HxWx3 uint8")
    raw = b"".join(b"\x00" + array[y].tobytes() for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )


def _png_dimensions(raw: bytes, label: str) -> tuple[int, int]:
    if len(raw) < 45 or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise D409Error(f"not a PNG: {label}")
    if raw[8:12] != b"\x00\x00\x00\x0d" or raw[12:16] != b"IHDR":
        raise D409Error(f"PNG IHDR is not exact: {label}")
    if raw[-12:] != b"\x00\x00\x00\x00IEND\xaeB`\x82":
        raise D409Error(f"PNG IEND trailer is not exact: {label}")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return width, height


def _png_decode_rgb(raw: bytes, label: str) -> np.ndarray:
    """Minimal decoder for 8-bit truecolor (2) / truecolor+alpha (6), no
    interlace — sufficient for the rerun CLI screenshot (C7)."""
    width, height = _png_dimensions(raw, label)
    depth, color_type, interlace = raw[24], raw[25], raw[28]
    if depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise D409Error(f"unsupported PNG format for {label}: depth={depth} color={color_type}")
    channels = 3 if color_type == 2 else 4
    idat = b""
    offset = 8
    while offset < len(raw):
        length = int.from_bytes(raw[offset : offset + 4], "big")
        tag = raw[offset + 4 : offset + 8]
        if tag == b"IDAT":
            idat += raw[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if tag == b"IEND":
            break
    stream = zlib.decompress(idat)
    stride = width * channels
    if len(stream) != height * (stride + 1):
        raise D409Error(f"PNG stream length mismatch for {label}")
    out = np.zeros((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    position = 0
    for y in range(height):
        filter_type = stream[position]
        position += 1
        line = np.frombuffer(stream[position : position + stride], dtype=np.uint8).astype(np.int32)
        position += stride
        recon = np.zeros(stride, dtype=np.int32)
        prev_line = previous.astype(np.int32)
        if filter_type == 0:
            recon = line
        elif filter_type == 2:
            recon = (line + prev_line) % 256
        else:
            for x in range(stride):
                left = recon[x - channels] if x >= channels else 0
                up = prev_line[x]
                up_left = prev_line[x - channels] if x >= channels else 0
                if filter_type == 1:
                    predictor = left
                elif filter_type == 3:
                    predictor = (left + up) // 2
                elif filter_type == 4:
                    p = left + up - up_left
                    pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                    predictor = left if (pa <= pb and pa <= pc) else (up if pb <= pc else up_left)
                else:
                    raise D409Error(f"unsupported PNG filter {filter_type} for {label}")
                recon[x] = (line[x] + predictor) % 256
        out[y] = recon.astype(np.uint8)
        previous = out[y]
    pixels = out.reshape(height, width, channels)
    return pixels[:, :, :3].copy()


FONT_5X7: dict[str, tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    "A": (0x0E, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    ".": (0, 0, 0, 0, 0, 0x0C, 0x0C),
    ",": (0, 0, 0, 0, 0x0C, 0x04, 0x08),
    ":": (0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0),
    ";": (0, 0x0C, 0x0C, 0, 0x0C, 0x04, 0x08),
    "-": (0, 0, 0, 0x1F, 0, 0, 0),
    "_": (0, 0, 0, 0, 0, 0, 0x1F),
    "/": (0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10),
    "(": (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "[": (0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E),
    "]": (0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E),
    "=": (0, 0x1F, 0, 0x1F, 0, 0, 0),
    "+": (0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0),
    "%": (0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13),
    "#": (0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A),
    "<": (0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02),
    ">": (0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08),
    "|": (0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "*": (0, 0x04, 0x15, 0x0E, 0x15, 0x04, 0),
    "!": (0x04, 0x04, 0x04, 0x04, 0x04, 0, 0x04),
    "?": (0x0E, 0x11, 0x01, 0x02, 0x04, 0, 0x04),
    "'": (0x04, 0x04, 0x08, 0, 0, 0, 0),
    "&": (0x08, 0x14, 0x14, 0x08, 0x15, 0x12, 0x0D),
}


def _draw_text(
    canvas: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 2
) -> None:
    cursor = x
    for character in text.upper():
        glyph = FONT_5X7.get(character, FONT_5X7["?"])
        for row_index, row_bits in enumerate(glyph):
            for column_index in range(5):
                if row_bits & (1 << (4 - column_index)):
                    y0 = y + row_index * scale
                    x0 = cursor + column_index * scale
                    canvas[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += 6 * scale


def _fill_rect(
    canvas: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int]
) -> None:
    canvas[max(0, y) : y + height, max(0, x) : x + width] = color


def _render_decision_sheet(
    evidence: dict[str, Any], promotion: dict[str, Any], worker_iface: dict[str, Any]
) -> dict[str, Any]:
    canvas = np.full((1080, 1920, 3), (13, 17, 23), dtype=np.uint8)
    radials = list(range(worker_iface["RADIAL_MIN_UM"], worker_iface["RADIAL_MAX_UM"] + 1, worker_iface["GRID_STEP_UM"]))
    tangents = list(range(worker_iface["TANGENT_MIN_UM"], worker_iface["TANGENT_MAX_UM"] + 1, worker_iface["GRID_STEP_UM"]))
    rows_by_key = {tuple(row["key_um"]): row for row in evidence["poses"]}
    admission = evidence["region_analysis"]["admission"]
    representatives = {
        tuple(entry["representative_um"]) for entry in admission["regions"]
    }
    grid_x, grid_y, cell_w, cell_h = 90, 110, 38, 15
    for row_index, rho in enumerate(radials):
        for column_index, tau in enumerate(tangents):
            row = rows_by_key[(rho, tau)]
            if row["full_pass"]:
                color = (96, 224, 134)
            elif row["order_constraint_ab_pass"] and row["admission"]["pass"]:
                color = (70, 170, 210)
            elif row["admission"]["pass"]:
                color = (46, 132, 84)
            else:
                color = (44, 50, 58)
            x0 = grid_x + column_index * cell_w
            y0 = grid_y + row_index * cell_h
            _fill_rect(canvas, x0, y0, cell_w - 2, cell_h - 2, color)
            if (rho, tau) in representatives:
                _fill_rect(canvas, x0, y0, cell_w - 2, 3, (240, 160, 60))
            if (rho, tau) == tuple(worker_iface["POSITIVE_CONTROL_KEY_UM"]):
                _fill_rect(canvas, x0, y0 + cell_h - 5, cell_w - 2, 3, (240, 240, 240))
    # WOBS-W3 repair R16: tick labels + legend at scale 2 (12-14 px glyphs)
    # for human legibility; spacing margins verified (tau pitch 4*cell_w,
    # rho left margin, legend line length < canvas width).
    for column_index, tau in enumerate(tangents):
        if column_index % 4 == 0 or column_index == len(tangents) - 1:
            _draw_text(canvas, grid_x + column_index * cell_w, grid_y + len(radials) * cell_h + 8, f"{tau}", (170, 180, 190), 2)
    for row_index, rho in enumerate(radials):
        if row_index % 8 == 0 or row_index == len(radials) - 1:
            _draw_text(canvas, 8, grid_y + row_index * cell_h + 3, f"{rho:>5}", (170, 180, 190), 2)
    _draw_text(canvas, grid_x, grid_y - 44, "RHO(UM) DOWN / TAU(UM) RIGHT - ADMISSION+ORDER+FULL LAYERS", (200, 210, 220), 2)
    _draw_text(canvas, grid_x, grid_y + len(radials) * cell_h + 30, "GRAY=NOT ADMITTED GREEN=ADMITTED BLUE=+ORDER(A&B) BRIGHT=+PINCH ORANGE=REP WHITE=(7000,11000)", (170, 180, 190), 2)

    text_x, line_y, step = 950, 44, 24
    counts = evidence["counts"]

    def line(text: str, color: tuple[int, int, int] = (208, 216, 224), scale: int = 2) -> None:
        nonlocal line_y
        _draw_text(canvas, text_x, line_y, text[:79], color, scale)
        line_y += step

    line("D409 ZERO-STEP DUAL-JAW CONTACT REGION ENUMERATION", (240, 240, 240))
    line("DECISION SHEET - CASE G0A_D409 / CYLD29H50 (D29XH50 REAL CYLINDER)")
    line("SPEC: SESSION 20260803 3RD DOC SECTION 2 + SECTION 4 (S4 WINS)")
    line("")
    line(f"VERDICT: {promotion['verdict'][:60]}", (120, 200, 255))
    line(f"EVIDENCE SHA256: {promotion['canonical_evidence_sha256'][:32]}")
    line(f"                 {promotion['canonical_evidence_sha256'][32:]}")
    line(f"REGION CSV SHA256: {promotion['region_map_csv_sha256'][:32]}")
    line(f"CANONICAL RUN: RUN1 (RUN1/RUN2 BIT-EXACT PASS)")
    line("")
    line(
        f"POSES {counts['poses']}  IK {counts['ik_converged']}  ADMIT {counts['admission_pass']}"
        f"  A {counts['a_band_pass']}  B {counts['b_pass']}  A&B {counts['order_ab_pass']}"
    )
    line(
        f"PINCH {counts['pinch_core_pass']}  FULL {counts['full_pass']}"
        f"  REGIONS {counts['admission_regions']} (CENSORED {counts['admission_regions_censored']})"
    )
    line(f"TOTAL QUERIES {evidence['query_budget_observed']['total_queries']}"
         f" (BUDGET {worker_iface['MAX_QUERIES_PER_RUN']})")
    line("")
    line("ADMISSION REGIONS: ID CELLS RHO_R(MM) CENSORED REP(RHO,TAU UM)", (230, 200, 120))
    for entry in admission["regions"][:12]:
        line(
            f"{entry['region_id']} {entry['cell_count']:>5} {float(entry['rho_R_mm']):>8.3f} "
            f"{'YES' if entry['domain_censored'] else 'NO ':>3} "
            f"({entry['representative_um'][0]},{entry['representative_um'][1]})"
        )
    if len(admission["regions"]) > 12:
        line(f"... {len(admission['regions']) - 12} MORE REGIONS (SEE EVIDENCE JSON)")
    if not admission["regions"]:
        line("(NO ADMITTED REGIONS)")
    line("")
    gate = evidence["anchor_gate"]["channels"]
    line("ANCHOR GATE (ANY>0.0005MM REJECT):", (230, 200, 120))
    line(f" L5 FK {float(gate['link5_fk_pos_err_mm']):.6f}  GR FK {float(gate['gripper_fk_pos_err_mm']):.6f}")
    line(f" L5 D  {float(gate['link5_dist_delta_mm']):.6f}  GR D  {float(gate['gripper_dist_delta_mm']):.6f}")
    line("")
    line("RHO_R VS 7.881MM = D330 PROXIMITY PROXY (D34XH90-ERA LABEL)")
    line("36.033MM = HISTORICAL PROXY LABEL ONLY - NO STANDALONE GATE")
    line("STALL REGIME 70-81MM UNREACHABLE IN DOMAIN (MAX 18.5MM)")
    line("")
    line("NULL: STABLE GRASP/FORCE CLOSURE/FEASIBILITY/SUCCESS/PUSH-OVER", (255, 160, 130))
    line("A&B DOES NOT EXCLUDE D362 PUSH-OVER POSE (D_FIX 4.2727MM)", (255, 160, 130))
    line("G0A_PASS=FALSE UNCHANGED - GEOMETRY LABELS: NO TRAINING PROMOTION", (255, 160, 130))
    line("DISPLAY ROUNDING ONLY - AUTHORITY = EVIDENCE JSON (FLOAT64)", (150, 158, 166), 1)

    raw = _png_encode_rgb(canvas)
    _write_bytes_x(DECISION_SHEET_PATH, raw)
    width, height = _png_dimensions(raw, "decision_sheet")
    if (width, height) != SCREENSHOT_PHYSICAL_SIZE:
        raise D409Error(f"decision sheet dimensions drift: {(width, height)}")
    return {
        "path": _rel(DECISION_SHEET_PATH),
        "dimensions": [width, height],
        "sha256": _sha_bytes(raw),
        "composer": "numpy + stdlib zlib/struct PNG + embedded 5x7 font (C6; PIL forbidden)",
    }


# ---------------------------------------------------------------------------
# Observability phase (RRD + blueprint + verify + screenshot; D341/W-LES2).
# ---------------------------------------------------------------------------

def _cylinder_mesh(radius: float, height: float, center: tuple[float, float, float], segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = np.stack([np.cos(angles) * radius, np.sin(angles) * radius], axis=1)
    bottom = np.concatenate([ring, np.full((segments, 1), -height / 2.0)], axis=1)
    top = np.concatenate([ring, np.full((segments, 1), height / 2.0)], axis=1)
    vertices = np.concatenate(
        [bottom, top, [[0.0, 0.0, -height / 2.0]], [[0.0, 0.0, height / 2.0]]], axis=0
    )
    triangles = []
    for index in range(segments):
        next_index = (index + 1) % segments
        triangles.append([index, next_index, segments + index])
        triangles.append([next_index, segments + next_index, segments + index])
        triangles.append([2 * segments, next_index, index])
        triangles.append([2 * segments + 1, segments + index, segments + next_index])
    vertices = vertices + np.asarray(center, dtype=np.float64)
    return vertices, np.asarray(triangles, dtype=np.int64)


def _combined_body_mesh(parts: list[dict[str, Any]], body_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertex_blocks = []
    triangle_blocks = []
    offset = 0
    rotation = body_mat[:3, :3]
    translation = body_mat[:3, 3]
    for part in parts:
        world = part["vertices"] @ rotation.T + translation
        vertex_blocks.append(world)
        triangle_blocks.append(part["triangles"] + offset)
        offset += len(world)
    return np.concatenate(vertex_blocks, axis=0), np.concatenate(triangle_blocks, axis=0)


def _author_observability(
    worker_iface: dict[str, Any], contract: dict[str, Any], promotion: dict[str, Any]
) -> dict[str, Any]:
    import hppfcl  # noqa: F401  (worker loader dependency)
    import rerun as rr
    import rerun.blueprint as rrb

    if str(rr.__version__) != EXPECTED_PACKAGE_VERSIONS["rerun-sdk"]:
        raise D409Error(f"rerun SDK drift: {rr.__version__}")
    module = _load_worker_module()
    frozen = module._load_frozen_inputs(hppfcl)
    joints = frozen["joints"]
    evidence = json.loads((ATTEMPT_ROOT / "run1" / worker_iface["EVIDENCE_NAME"]).read_bytes())
    rows_by_key = {tuple(row["key_um"]): row for row in evidence["poses"]}
    admission = evidence["region_analysis"]["admission"]
    admitted = {
        tuple(int(part) for part in key.split(":")) for key in admission["region_of"]
    }

    representative_keys = [
        tuple(entry["representative_um"]) for entry in admission["regions"]
    ][:RRD_MAX_REPRESENTATIVES]
    instance_tags: dict[str, tuple[int, int] | None] = {
        f"rep_{key[0]}_{key[1]}": key for key in representative_keys
    }
    instance_tags["anchor_frozen"] = None

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin="/enum/grid",
                name="D409 grid - all 1,239 cells (admission/order/full layers)",
            ),
            rrb.Spatial3DView(
                origin="/",
                contents=[
                    "/enum/prototype/**",
                    "/enum/instance/**",
                    "/enum/candidate/**",
                ],
                name="D409 representative poses + witnesses (q5_arc)",
            ),
            rrb.TextDocumentView(origin="/metadata/run", name="D409 run metadata"),
            column_shares=[3.0, 4.0, 3.0],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )
    recording = rr.RecordingStream(
        application_id=RRD_APPLICATION_ID, recording_id=RRD_RECORDING_ID
    )
    # File sink attached (with the embedded blueprint) BEFORE the first user
    # log (D341); finalized by flush + disconnect (C12).
    recording.save(str(RRD_PATH), default_blueprint=blueprint)
    counts = evidence["counts"]
    metadata_text = "\n".join(
        [
            "# D409 zero-step dual-jaw contact-region enumeration (canonical run1)",
            f"- verdict: {promotion['verdict']}",
            f"- canonical evidence sha256: {promotion['canonical_evidence_sha256']}",
            f"- region csv sha256: {promotion['region_map_csv_sha256']}",
            f"- poses {counts['poses']} / admission {counts['admission_pass']} / "
            f"order A&B {counts['order_ab_pass']} / full {counts['full_pass']}",
            f"- admission regions {counts['admission_regions']} "
            f"(domain-censored {counts['admission_regions_censored']})",
            f"- total queries {evidence['query_budget_observed']['total_queries']}",
            "- Float64 authority = evidence JSON; RRD Float32 copies are inspection-only",
            "- g0a_pass=false unchanged; A&B does not exclude the D362 push-over pose",
        ]
    )
    rr.log(
        "/metadata/run",
        rr.TextDocument(metadata_text, media_type="text/markdown"),
        static=True,
        recording=recording,
    )
    cylinder_vertices, cylinder_triangles = _cylinder_mesh(
        module.CYL_RADIUS_M, module.CYL_HEIGHT_M, tuple(module.CYL_CENTER_M)
    )
    rr.log(
        "/enum/prototype/cylinder",
        rr.Mesh3D(
            vertex_positions=cylinder_vertices,
            triangle_indices=cylinder_triangles,
            albedo_factor=[235, 210, 90, 255],
        ),
        static=True,
        recording=recording,
    )
    for body in ("link5", "gripper_link"):
        for part in frozen["parts_by_body"][body]:
            rr.log(
                f"/enum/source/{body}/{part['name']}",
                rr.Mesh3D(
                    vertex_positions=part["vertices"], triangle_indices=part["triangles"]
                ),
                static=True,
                recording=recording,
            )

    grid_positions = []
    grid_admission_colors = []
    grid_full_colors = []
    for rho in range(worker_iface["RADIAL_MIN_UM"], worker_iface["RADIAL_MAX_UM"] + 1, worker_iface["GRID_STEP_UM"]):
        for tau in range(worker_iface["TANGENT_MIN_UM"], worker_iface["TANGENT_MAX_UM"] + 1, worker_iface["GRID_STEP_UM"]):
            row = rows_by_key[(rho, tau)]
            grid_positions.append([tau / 1000.0, rho / 1000.0])
            if row["admission"]["pass"]:
                grid_admission_colors.append(
                    [70, 170, 210, 255] if row["order_constraint_ab_pass"] else [46, 132, 84, 255]
                )
            else:
                grid_admission_colors.append([64, 70, 78, 255])
            grid_full_colors.append(
                [96, 224, 134, 255] if row["full_pass"] else [0, 0, 0, 0]
            )
    rr.log(
        "/enum/grid/admission",
        rr.Points2D(positions=grid_positions, colors=grid_admission_colors, radii=0.09),
        static=True,
        recording=recording,
    )
    rr.log(
        "/enum/grid/full_pass",
        rr.Points2D(positions=grid_positions, colors=grid_full_colors, radii=0.05),
        static=True,
        recording=recording,
    )
    representative_positions = [
        [key[1] / 1000.0, key[0] / 1000.0] for key in representative_keys
    ]
    if representative_positions:
        rr.log(
            "/enum/grid/representatives",
            rr.Points2D(
                positions=representative_positions,
                colors=[[240, 160, 60, 255]] * len(representative_positions),
                radii=0.12,
            ),
            static=True,
            recording=recording,
        )
    positive_key = worker_iface["POSITIVE_CONTROL_KEY_UM"]
    rr.log(
        "/enum/grid/positive_control",
        rr.Points2D(
            positions=[[positive_key[1] / 1000.0, positive_key[0] / 1000.0]],
            colors=[[245, 245, 245, 255]],
            radii=0.12,
        ),
        static=True,
        recording=recording,
    )
    boundary_positions = []
    step = worker_iface["GRID_STEP_UM"]
    for rho, tau in sorted(admitted):
        neighbors = ((rho - step, tau), (rho + step, tau), (rho, tau - step), (rho, tau + step))
        if any(neighbor not in admitted for neighbor in neighbors):
            boundary_positions.append([tau / 1000.0, rho / 1000.0])
    if boundary_positions:
        rr.log(
            "/enum/candidate/region_boundary",
            rr.Points2D(
                positions=boundary_positions,
                colors=[[230, 220, 120, 255]] * len(boundary_positions),
                radii=0.045,
            ),
            static=True,
            recording=recording,
        )

    logged_candidate_entities = ["/enum/candidate/region_boundary"] if boundary_positions else []
    d349_q = [
        float(value)
        for value in frozen["d349"]["target_state_guard"]["commanded_joint_rad_float32"]
    ]
    anchors_f32 = np.linspace(
        np.float32(module.Q5_OPEN_RAD), np.float32(0.0), module.ARC_ANCHOR_COUNT, dtype=np.float32
    )
    logged_instance_entities: list[str] = []
    for tag, key in sorted(instance_tags.items()):
        if key is None:
            q_arm_rad = np.asarray(d349_q[:5], dtype=np.float64)
            arc_values = [float(anchor) for anchor in anchors_f32]
            witness_points = None
        else:
            row = rows_by_key[key]
            q_arm_rad = np.radians(np.asarray(row["commanded_joint_deg"][:5], dtype=np.float64))
            arc_values = [entry["q5_rad"] for entry in row["arc_sweep"]["anchors"]]
            crossing = row["first_crossing"]
            witness_points = None
            if crossing.get("found") and crossing.get("endpoint_contract_valid"):
                # Canonical evidence serializes floats as repr strings; the
                # RRD copy is Float32 inspection-only (D341).
                witness_points = np.asarray(
                    [
                        crossing["clear_endpoint"]["witness_geometry_m"],
                        crossing["clear_endpoint"]["witness_cylinder_m"],
                        row["link5"]["fixed4_witness_geometry_m"],
                        row["link5"]["fixed4_witness_cylinder_m"],
                    ],
                    dtype=np.float64,
                )
        link5_mat, gripper_pre, _tcp = module._fk_frames(joints, q_arm_rad)
        link5_vertices, link5_triangles = _combined_body_mesh(
            frozen["parts_by_body"]["link5"], link5_mat
        )
        rr.log(
            f"/enum/instance/{tag}/link5",
            rr.Mesh3D(vertex_positions=link5_vertices, triangle_indices=link5_triangles),
            static=True,
            recording=recording,
        )
        logged_instance_entities.append(f"/enum/instance/{tag}/link5")
        for anchor_index, q5_value in enumerate(arc_values):
            rr.set_time(RRD_TIMELINE, sequence=anchor_index, recording=recording)
            gripper_mat = module._gripper_mat(joints, gripper_pre, float(q5_value))
            gripper_vertices, gripper_triangles = _combined_body_mesh(
                frozen["parts_by_body"]["gripper_link"], gripper_mat
            )
            rr.log(
                f"/enum/instance/{tag}/gripper_link",
                rr.Mesh3D(
                    vertex_positions=gripper_vertices, triangle_indices=gripper_triangles
                ),
                recording=recording,
            )
            rr.log(
                f"/enum/instance/{tag}/q5_rad",
                rr.Scalars(float(q5_value)),
                recording=recording,
            )
        logged_instance_entities.append(f"/enum/instance/{tag}/gripper_link")
        logged_instance_entities.append(f"/enum/instance/{tag}/q5_rad")
        if witness_points is not None:
            rr.set_time(RRD_TIMELINE, sequence=len(arc_values) - 1, recording=recording)
            rr.log(
                f"/enum/candidate/witness/{tag}",
                rr.Points3D(
                    positions=witness_points,
                    colors=[
                        [255, 120, 120, 255],
                        [255, 220, 120, 255],
                        [120, 200, 255, 255],
                        [160, 255, 160, 255],
                    ],
                    radii=0.0018,
                ),
                recording=recording,
            )
            logged_candidate_entities.append(f"/enum/candidate/witness/{tag}")
    recording.flush(timeout_sec=30.0)
    recording.disconnect()
    blueprint.save(RRD_APPLICATION_ID, str(RBL_PATH))
    _fsync_existing(RRD_PATH)
    _fsync_existing(RBL_PATH)
    return {
        "evidence": evidence,
        "instance_entities": sorted(set(logged_instance_entities)),
        "candidate_entities": sorted(set(logged_candidate_entities)),
        "representative_keys": [list(key) for key in representative_keys],
    }


def _validate_observability(
    worker_iface: dict[str, Any], authored: dict[str, Any], promotion: dict[str, Any]
) -> dict[str, Any]:
    from rerun.experimental import RrdReader

    reader = RrdReader(str(RRD_PATH))
    recordings = reader.recordings()
    blueprints = reader.blueprints()
    if len(recordings) != 1 or len(blueprints) != 1:
        raise D409Error(
            f"RRD store inventory mismatch: recordings={len(recordings)} blueprints={len(blueprints)}"
        )
    recording_info = recordings[0]
    if (
        recording_info.application_id != RRD_APPLICATION_ID
        or recording_info.recording_id != RRD_RECORDING_ID
    ):
        raise D409Error("RRD recording identity mismatch")
    entity_components: dict[str, set[str]] = {}
    timeline_names: set[str] = set()
    for chunk in reader.stream(store=recording_info):
        entity_path = str(chunk.entity_path)
        if entity_path.startswith("/__"):
            continue
        batch = chunk.to_record_batch()
        components = entity_components.setdefault(entity_path, set())
        for field in list(batch.schema):
            metadata = {
                (key.decode() if isinstance(key, bytes) else key): (
                    value.decode() if isinstance(value, bytes) else value
                )
                for key, value in (field.metadata or {}).items()
            }
            kind = metadata.get("rerun:kind")
            if kind == "index":
                timeline_names.add(field.name)
            elif kind == "data":
                components.add(field.name)
    expected_entities = {
        "/metadata/run",
        "/enum/prototype/cylinder",
        "/enum/grid/admission",
        "/enum/grid/full_pass",
        "/enum/grid/positive_control",
    }
    source_entities = {
        path for path in entity_components if path.startswith("/enum/source/")
    }
    checks = {
        "recording_store_count_one": len(recordings) == 1,
        "embedded_blueprint_store_count_one": len(blueprints) == 1,
        "required_static_entities_present": expected_entities <= set(entity_components),
        "source_part_entity_count_128": len(source_entities) == 128,
        "instance_entities_present": set(authored["instance_entities"]) <= set(entity_components),
        "candidate_entities_present": set(authored["candidate_entities"]) <= set(entity_components),
        "timeline_q5_arc_present": RRD_TIMELINE in timeline_names,
        "grid_points2d_components": {
            "Points2D:positions",
            "Points2D:colors",
        } <= entity_components.get("/enum/grid/admission", set()),
        "source_mesh_components": all(
            {"Mesh3D:vertex_positions", "Mesh3D:triangle_indices"} <= components
            for path, components in entity_components.items()
            if path.startswith("/enum/source/")
        ),
        "metadata_text_component": any(
            name.startswith("TextDocument:")
            for name in entity_components.get("/metadata/run", set())
        ),
    }
    if not all(checks.values()):
        raise D409Error(f"RRD entity/timeline/component contract failed: {checks}")

    verify_command = [
        str(RERUN_CLI),
        "rrd",
        "verify",
        "--check-footers",
        "true",
        str(RRD_PATH),
        str(RBL_PATH),
    ]
    verify = subprocess.run(
        verify_command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
        shell=False,
    )
    if verify.returncode != 0 or "verified without error" not in verify.stdout:
        raise D409Error(f"rerun rrd verify failed: {verify.stdout[:500]}")

    screenshot_command = [
        str(RERUN_CLI),
        "--headless",
        "--hide-welcome-screen",
        "--window-size",
        SCREENSHOT_LOGICAL_SIZE,
        "--screenshot-to",
        str(REGION_SCREENSHOT_PATH),
        str(RRD_PATH),
        str(RBL_PATH),
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
        timeout=300,
        check=False,
        shell=False,
    )
    if screenshot.returncode != 0:
        raise D409Error(f"rerun screenshot failed: {screenshot.stdout[:500]}")
    if "device_type: Cpu" not in screenshot.stdout or "llvmpipe" not in screenshot.stdout:
        raise D409Error("software (llvmpipe CPU) renderer identity missing in screenshot run")
    screenshot_raw = REGION_SCREENSHOT_PATH.read_bytes()
    dimensions = _png_dimensions(screenshot_raw, "region_map_screenshot")
    if dimensions != SCREENSHOT_PHYSICAL_SIZE:
        raise D409Error(f"screenshot physical size mismatch: {dimensions}")
    pixels = _png_decode_rgb(screenshot_raw, "region_map_screenshot")
    pixel_std = float(pixels.astype(np.float64).std())
    if pixel_std <= 1.0:
        raise D409Error(f"screenshot appears blank (std={pixel_std})")
    _fsync_existing(REGION_SCREENSHOT_PATH)

    sheet_report = _render_decision_sheet(authored["evidence"], promotion, worker_iface)
    validation = {
        "artifact": RERUN_VALIDATION_ARTIFACT,
        "case": "g0a_d409",
        "checks": checks,
        "entity_count_nonsystem": len(entity_components),
        "timelines": sorted(timeline_names),
        "instance_entities": authored["instance_entities"],
        "candidate_entities": authored["candidate_entities"],
        "representative_keys_um": authored["representative_keys"],
        "rrd_sha256": _sha_path(RRD_PATH),
        "rbl_sha256": _sha_path(RBL_PATH),
        "rrd_verify_command": verify_command,
        "rrd_verify_pass": True,
        "rrd_verify_output": verify.stdout.strip()[-2000:],
        "screenshot_command": screenshot_command,
        "screenshot_renderer_cpu_llvmpipe": True,
        "screenshot_dimensions": list(dimensions),
        "screenshot_pixel_std": pixel_std,
        "screenshot_sha256": _sha_bytes(screenshot_raw),
        "error_banner_note": (
            "automated check = decodability + non-blank; authoritative judgment = "
            "manual 11-field no_error_banner_visible (C7)"
        ),
        "decision_sheet": sheet_report,
        "float_authority_note": (
            "Float64 authority = canonical evidence JSON; RRD Float32 copies are "
            "inspection-only and never hashed into a scientific gate (D341)"
        ),
        "pass": True,
    }
    _write_json_x(RERUN_VALIDATION_PATH, validation)
    return validation


def _write_screenshot_manifest(
    contract: dict[str, Any], promotion: dict[str, Any]
) -> str:
    roles = contract["protocol_schemas"]["screenshot_image_roles"]
    layout = contract["screenshot_layout"]
    images = []
    for basename in sorted(layout):
        path = ATTEMPT_ROOT / basename
        raw = path.read_bytes()
        dimensions = _png_dimensions(raw, basename)
        if list(dimensions) != list(layout[basename]):
            raise D409Error(f"screenshot layout mismatch for {basename}: {dimensions}")
        images.append(
            {
                "bytes": len(raw),
                "dimensions": list(dimensions),
                "manual_role": roles[basename],
                "path": f"{_rel(ATTEMPT_ROOT)}/{basename}",
                "root_relative_path": basename,
                "sha256": _sha_bytes(raw),
            }
        )
    rrd_paths = contract["protocol_schemas"]["screenshot_rrd_report_paths"]
    manifest = {
        "artifact": contract["_ast"]["SCREENSHOT_MANIFEST_ARTIFACT"],
        "canonical_evidence_sha256": promotion["canonical_evidence_sha256"],
        "determinism_bitexact_pass": True,
        "images": images,
        "manual_target_count": 2,
        "new_controlled_physics_steps": 0,
        "region_map_csv_sha256": promotion["region_map_csv_sha256"],
        "rrd_report": {
            "rbl_path": rrd_paths["rbl_path"],
            "rbl_sha256": _sha_path(RBL_PATH),
            "rrd_path": rrd_paths["rrd_path"],
            "rrd_sha256": _sha_path(RRD_PATH),
            "rrd_verify_pass": True,
            "validation_path": rrd_paths["validation_path"],
            "validation_sha256": _sha_path(RERUN_VALIDATION_PATH),
        },
    }
    return _write_json_x(SCREENSHOT_MANIFEST_PATH, manifest)


# ---------------------------------------------------------------------------
# Runtime admission + inventory (P4-6 audit lineage).
# ---------------------------------------------------------------------------

def _validate_approval_tuple(approved_sha256: str) -> dict[str, Any]:
    _require_hex64(approved_sha256, "--approved-tuple-sha256")
    for path, label in (
        (PREREG_PATH, "preregistration"),
        (STATIC_RESULTS_PATH, "static fixture results"),
        (ATTESTATION_PATH, "attestation"),
        (TUPLE_PATH, "tuple"),
    ):
        if not path.is_file():
            raise D409Error(f"runtime admission: {label} missing: {path}")
        _require_regular_nlink1(path, label)
    tuple_raw = TUPLE_PATH.read_bytes()
    tuple_sha = _sha_bytes(tuple_raw)
    if tuple_sha != approved_sha256:
        raise D409Error(
            f"user-approved tuple sha mismatch: cited {approved_sha256} != file {tuple_sha}"
        )
    tuple_document = _strict_json_bytes(tuple_raw)
    hashes = tuple_document.get("hashes")
    if not isinstance(hashes, dict):
        raise D409Error("tuple hashes missing")
    observed = {
        "preregistration_sha256": _sha_path(PREREG_PATH),
        "controller_sha256": _sha_path(CONTROLLER_PATH),
        "worker_sha256": _sha_path(WORKER_PATH),
        "manual_writer_sha256": _sha_path(WRITER_PATH),
    }
    for key, value in observed.items():
        if hashes.get(key) != value:
            raise D409Error(f"tuple {key} does not match current file bytes")
    if tuple_document.get("attestation_sha256") != _sha_path(ATTESTATION_PATH):
        raise D409Error("tuple attestation_sha256 does not match current attestation bytes")
    # OPS-W3 repair R12: re-bind the static fixture results bytes through
    # the attestation (existence + nlink1 alone would admit a swapped file).
    attestation_document = _strict_json_bytes(ATTESTATION_PATH.read_bytes())
    if attestation_document.get("static_fixture_results_sha256") != _sha_path(STATIC_RESULTS_PATH):
        raise D409Error(
            "attestation static_fixture_results_sha256 does not match current "
            "static fixture results bytes"
        )
    for source in (CONTROLLER_PATH, WORKER_PATH, WRITER_PATH):
        _require_regular_nlink1(source, "harness source")
    return {
        "approved_tuple_sha256": approved_sha256,
        "tuple_sha256": tuple_sha,
        "hashes": observed,
        "pass": True,
    }


def _runtime_admission(prereg: dict[str, Any]) -> dict[str, Any]:
    _audit_registered(prereg)
    overlay = prereg["runtime_overlay_contract"]
    allowed = set(overlay["allowed_dirty_paths"])
    dirty_now = _git_dirty_paths()
    unexpected = sorted(set(dirty_now) - allowed)
    recorded_live = set(prereg["head_pin"]["git_dirty_live"])
    missing_live = sorted(recorded_live - set(dirty_now))
    head_now = _git_head()
    admission = {
        "git_head_registered": prereg["head_pin"]["git_head"],
        "git_head_now": head_now,
        "git_head_unchanged": head_now == prereg["head_pin"]["git_head"],
        "dirty_now": dirty_now,
        "dirty_subset_of_allowlist": not unexpected,
        "unexpected_dirty": unexpected,
        "live_dirty_exactness_missing": missing_live,
    }
    if unexpected:
        raise D409Error(f"runtime admission: unexpected dirty paths: {unexpected}")
    if missing_live:
        raise D409Error(
            f"runtime admission: live dirty paths recorded at prereg vanished (exactness): {missing_live}"
        )
    if not admission["git_head_unchanged"]:
        raise D409Error(
            f"runtime admission: git HEAD moved since prereg: {head_now}"
        )
    # Crash resume fail-closed: no runtime artifact may pre-exist.
    preexisting: list[str] = []
    allowed_existing = {path.name for path in STATIC_ARTIFACT_PATHS}
    for entry in sorted(ATTEMPT_ROOT.iterdir()):
        if entry.name not in allowed_existing:
            preexisting.append(entry.name)
    if preexisting:
        raise D409Error(
            f"runtime admission: pre-existing runtime artifacts (crash resume fail-closed): {preexisting}"
        )
    admission["prerun_attempt_root_entries"] = sorted(
        entry.name for entry in ATTEMPT_ROOT.iterdir()
    )
    admission["pass"] = True
    return admission


def _write_prerun_inventory(admission: dict[str, Any], tuple_gate: dict[str, Any]) -> str:
    inventory = {
        "artifact": INVENTORY_ARTIFACT,
        "case": "g0a_d409",
        "created_utc": _utc_now(),
        "attempt_root": _rel(ATTEMPT_ROOT),
        "entries": admission["prerun_attempt_root_entries"],
        "run_structure": {
            run_name: {"exists": (ATTEMPT_ROOT / run_name).exists()} for run_name in RUN_DIR_NAMES
        },
        "run_dirs_absent_before_run1": not any(
            (ATTEMPT_ROOT / run_name).exists() for run_name in RUN_DIR_NAMES
        ),
        "static_artifact_sha256": {
            path.name: _sha_path(path) for path in STATIC_ARTIFACT_PATHS
        },
        "tuple_gate": tuple_gate,
        "admission": admission,
        "note": "pre-run inventory reflects the run1/run2 dual-run structure (P4-6)",
    }
    return _write_json_x(PRERUN_INVENTORY_PATH, inventory)


# ---------------------------------------------------------------------------
# Runtime mode.
# ---------------------------------------------------------------------------

def run_runtime(approved_tuple_sha256: str) -> int:
    environment = _environment_gate()
    # WOBS-W1 repair R14: the user-approval tuple gate re-hashes all current
    # harness bytes and runs FIRST, before any worker-module import or
    # writer-contract subprocess executes unapproved bytes in/for this
    # process.  It needs no derived interfaces.
    tuple_gate = _validate_approval_tuple(approved_tuple_sha256)
    worker_iface = _derive_worker_interface()
    contract = _writer_contract()
    prereg_raw = PREREG_PATH.read_bytes()
    prereg_sha = _sha_bytes(prereg_raw)
    prereg = _strict_json_bytes(prereg_raw)
    if prereg.get("case") != "g0a_d409":
        raise D409Error("prereg case mismatch")
    if prereg["candidate_contract"]["worker"]["sha256"] != worker_iface["WORKER_SOURCE_SHA256"]:
        raise D409Error("prereg worker candidate sha drift")
    # OPS-W4 repair R12 (strong form): re-compare the imported worker
    # module's real-geometry pins against the prereg registration, so the
    # geometry pin is enforced on the runtime path itself (not only via the
    # worker-source sha binding demonstrated by fixture N1).
    module = _load_worker_module()
    if (
        repr(module.CYL_RADIUS_M) != prereg["geometry"]["radius_m_repr"]
        or repr(module.CYL_HEIGHT_M) != prereg["geometry"]["height_m_repr"]
        or repr(module.CYL_X_M) != prereg["geometry"]["x_m_repr"]
        or repr(module.TABLE_Z_M) != prereg["geometry"]["table_z_m_repr"]
        or repr(module.Z_CENTER_M) != prereg["geometry"]["z_center_m_repr"]
    ):
        raise D409Error("runtime geometry pin mismatch vs prereg (R12/OPS2-W3)")
    admission = _runtime_admission(prereg)

    controller_sha = tuple_gate["hashes"]["controller_sha256"]
    worker_sha = tuple_gate["hashes"]["worker_sha256"]
    writer_sha = tuple_gate["hashes"]["manual_writer_sha256"]

    root_fd = os.open(ATTEMPT_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    root_stat = os.fstat(root_fd)
    prearm_hard_deadline_ns = time.monotonic_ns() + PREARM_RUNTIME_BUDGET_NS
    phase = PhaseLog(root_fd, contract["_ast"]["PHASE_ROW_ARTIFACT"])
    writer_process: subprocess.Popen[bytes] | None = None
    writer_socket: socket.socket | None = None
    try:
        controller_started_row = phase.append(
            "controller_started",
            {
                "approved_tuple_sha256": approved_tuple_sha256,
                "controller_pid": os.getpid(),
                "controller_sha256": controller_sha,
                "controller_start_ticks": _proc_start_ticks(os.getpid()),
                "prearm_hard_deadline_monotonic_ns": prearm_hard_deadline_ns,
                "preregistration_sha256": prereg_sha,
                "root_dev": root_stat.st_dev,
                "root_ino": root_stat.st_ino,
                "worker_sha256": worker_sha,
                "writer_sha256": writer_sha,
            },
        )
        writer_process, writer_socket, nonce, writer_start_ticks = _spawn_writer(
            root_fd,
            root_stat.st_dev,
            root_stat.st_ino,
            phase,
            controller_started_row,
            approved_tuple_sha256,
            prereg_sha,
            controller_sha,
            worker_sha,
            writer_sha,
            prearm_hard_deadline_ns,
            contract,
        )
        inventory_sha = _write_prerun_inventory(admission, tuple_gate)
        phase.append("prerun_inventory", {"inventory_sha256": inventory_sha})

        phase.append("run1_started", {"registered_command": "prereg registered_worker_command.per_run.run1"})
        supervisor_run1 = _run_worker_once("run1", worker_iface, prereg, 1)
        row = phase.append(
            "run1_preclose_verified",
            {
                "evidence_sha256": supervisor_run1["evidence_sha256"],
                "preclose_status": supervisor_run1["preclose_status"],
            },
        )
        _ping_writer(writer_socket, nonce, writer_process, writer_start_ticks, row)

        phase.append("run2_started", {"precondition": "run1_preclose_pass"})
        supervisor_run2 = _run_worker_once("run2", worker_iface, prereg, 2)
        phase.append(
            "run2_preclose_verified",
            {
                "evidence_sha256": supervisor_run2["evidence_sha256"],
                "preclose_status": supervisor_run2["preclose_status"],
            },
        )

        promotion = _byte_compare_and_promote(worker_iface)
        row = phase.append(
            "canonical_promotion",
            {
                "canonical_run": "run1",
                "canonical_evidence_sha256": promotion["canonical_evidence_sha256"],
                "region_map_csv_sha256": promotion["region_map_csv_sha256"],
                "verdict": promotion["verdict"],
            },
        )
        _ping_writer(writer_socket, nonce, writer_process, writer_start_ticks, row)

        authored = _author_observability(worker_iface, contract, promotion)
        validation = _validate_observability(worker_iface, authored, promotion)
        phase.append(
            "observability_validated",
            {
                "rrd_sha256": validation["rrd_sha256"],
                "rbl_sha256": validation["rbl_sha256"],
                "screenshot_sha256": validation["screenshot_sha256"],
            },
        )
        manifest_sha = _write_screenshot_manifest(contract, promotion)
        row = phase.append("screenshot_manifest", {"screenshot_manifest_sha256": manifest_sha})
        _ping_writer(writer_socket, nonce, writer_process, writer_start_ticks, row)

        prompt_started = time.monotonic_ns()
        if prompt_started > prearm_hard_deadline_ns:
            raise D409Error("manual prompt would start after the pre-arm hard deadline")
        manual_deadline = prompt_started + contract["manual_timeout_ns"]
        writer_deadline = manual_deadline - contract["writer_deadline_lead_ns"]
        manual_prompt_row = phase.append(
            "manual_prompt",
            {
                "manual_basename": contract["manual_basename"],
                "manual_deadline_monotonic_ns": manual_deadline,
                "manual_prompt_started_monotonic_ns": prompt_started,
                "new_controlled_physics_steps": 0,
                "screenshot_manifest_sha256": manifest_sha,
                "writer_deadline_monotonic_ns": writer_deadline,
            },
        )
        print("D409 MANUAL INSPECTION (once per attempt, 600s):")
        print(f"  inspect {_rel(REGION_SCREENSHOT_PATH)}")
        print(f"  inspect {_rel(DECISION_SHEET_PATH)}")
        template = {
            "required_fields": {name: False for name in contract["required_boolean_fields"]},
            "notes": "",
        }
        print("  reply with ONE JSON line:")
        print(f"  {json.dumps(template, sort_keys=True)}")
        manual_input = _read_manual_stdin(
            writer_deadline - MANUAL_STDIN_SAFETY_LEAD_NS, contract
        )
        publish_body = {
            "manual_deadline_monotonic_ns": manual_deadline,
            "manual_input": manual_input,
            "manual_prompt_started_monotonic_ns": prompt_started,
            "op": "publish",
            "phase_row_sha256": manual_prompt_row["row_sha256"],
            "phase_sequence": manual_prompt_row["sequence"],
            "screenshot_manifest_sha256": manifest_sha,
            "writer_deadline_monotonic_ns": writer_deadline,
        }
        writer_socket.settimeout(
            max(0.001, (manual_deadline - time.monotonic_ns()) / 1_000_000_000)
        )
        _send_json_line(
            writer_socket, {"body": publish_body, "hmac_sha256": _hmac_hex(nonce, publish_body)}
        )
        ack = _recv_authenticated(writer_socket, nonce)
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
            "writer ACK body",
        )
        if (
            ack["op"] != "published_fsynced"
            or ack["writer_pid"] != writer_process.pid
            or ack["writer_start_ticks"] != writer_start_ticks
            or ack["published_before_writer_deadline"] is not True
        ):
            raise D409Error("writer ACK binding mismatch")
        manual_path = ATTEMPT_ROOT / contract["manual_basename"]
        manual_raw = manual_path.read_bytes()
        if len(manual_raw) != ack["manual_size"] or _sha_bytes(manual_raw) != ack["manual_sha256"]:
            raise D409Error("published manual file does not match writer ACK")
        manual_document = _strict_json_bytes(manual_raw)
        if manual_document.get("artifact") != contract["_ast"]["MANUAL_ARTIFACT"]:
            raise D409Error("manual document artifact mismatch")
        if manual_document.get("received") is not True:
            raise D409Error("manual document received flag is not true")
        fields = manual_document["required_fields"]
        manual_pass = all(fields[name] for name in contract["required_boolean_fields"])
        if manual_document.get("pass") is not manual_pass or ack["manual_pass"] is not manual_pass:
            raise D409Error("manual pass is not the AND of the 11 fields")
        science = manual_document.get("source_science", {})
        if science.get("g0a_pass") is not False or science.get("scientific_verdict") is not None:
            raise D409Error("manual document attempts to change frozen science")
        writer_rc = writer_process.wait(timeout=60)
        if writer_rc != 0:
            raise D409Error(f"manual writer exit code {writer_rc}")
        receipt = {
            "artifact": RECEIPT_ARTIFACT,
            "case": "g0a_d409",
            "ack": ack,
            "manual_sha256": ack["manual_sha256"],
            "manual_pass": manual_pass,
            "writer_returncode": writer_rc,
            "created_utc": _utc_now(),
        }
        receipt_sha = _write_json_x(MANUAL_RECEIPT_PATH, receipt)
        phase.append(
            "manual_published",
            {"manual_sha256": ack["manual_sha256"], "receipt_sha256": receipt_sha},
        )

        dirty_final = _git_dirty_paths()
        allowed = set(prereg["runtime_overlay_contract"]["allowed_dirty_paths"])
        tuple_recheck_pass = True
        try:
            _validate_approval_tuple(approved_tuple_sha256)
        except D409Error:
            tuple_recheck_pass = False
        # OPS-W2 repair R11: replace the two constant-True self-attestations
        # with re-read evidence — the persisted prerun inventory and an
        # independent phase-row count from the on-disk hash-chained log.
        inventory_document = _strict_json_bytes(PRERUN_INVENTORY_PATH.read_bytes())
        phase_event_counts: dict[str, int] = {}
        for phase_line in PHASE_PATH.read_bytes().splitlines():
            if not phase_line.strip():
                continue
            phase_event = _strict_json_bytes(phase_line).get("event")
            phase_event_counts[phase_event] = phase_event_counts.get(phase_event, 0) + 1
        manual_path = ATTEMPT_ROOT / contract["manual_basename"]
        manual_pending_path = ATTEMPT_ROOT / contract["manual_pending_basename"]
        audit_checks = {
            "run1_invocation_count_eq_1": supervisor_run1["run_invocation_count"] == 1,
            "run2_invocation_count_eq_1": supervisor_run2["run_invocation_count"] == 1,
            "worker_invocations_total_eq_2": (
                supervisor_run1["invocation_index"] == 1
                and supervisor_run2["invocation_index"] == 2
            ),
            "worker_invocations_total_eq_2_phase_rows": (
                phase_event_counts.get("run1_started", 0) == 1
                and phase_event_counts.get("run2_started", 0) == 1
            ),
            "automatic_retries_zero": (
                supervisor_run1["automatic_retries"] == 0
                and supervisor_run2["automatic_retries"] == 0
            ),
            "prerun_inventory_run_structure": (
                inventory_document.get("artifact") == INVENTORY_ARTIFACT
                and inventory_document.get("run_dirs_absent_before_run1") is True
            ),
            "final_dirty_subset_of_allowlist": set(dirty_final) <= allowed,
            "approval_tuple_and_bound_files_unchanged": tuple_recheck_pass,
            "manual_inspections_this_attempt_eq_1": (
                phase_event_counts.get("manual_published", 0) == 1
                and manual_path.is_file()
                and not manual_pending_path.exists()
            ),
        }
        if not all(audit_checks.values()):
            raise D409Error(f"completion audit checks failed: {audit_checks}")
        completion = {
            "artifact": COMPLETION_ARTIFACT,
            "case": "g0a_d409",
            "status": RUNTIME_COMPLETE_STATUS,
            "created_utc": _utc_now(),
            "environment": environment,
            "tuple_gate": tuple_gate,
            "supervisors": {"run1": supervisor_run1, "run2": supervisor_run2},
            "canonical_promotion_sha256": _sha_path(CANONICAL_PROMOTION_PATH),
            "rerun_validation_sha256": _sha_path(RERUN_VALIDATION_PATH),
            "screenshot_manifest_sha256": manifest_sha,
            "manual_receipt_sha256": receipt_sha,
            "manual_pass": manual_pass,
            "audit_checks": audit_checks,
            "final_dirty": dirty_final,
            "interpretation_boundary": prereg["interpretation_boundary"],
            "verdict": promotion["verdict"],
        }
        completion_sha = _write_json_x(COMPLETION_PATH, completion)
        phase.append(
            "runtime_complete",
            {"completion_sha256": completion_sha, "status": RUNTIME_COMPLETE_STATUS},
        )
        print(f"D409 runtime status {RUNTIME_COMPLETE_STATUS}")
        print(f"D409 manual pass {manual_pass}")
        print(f"D409 completion sha256 {completion_sha}")
        return 0
    except BaseException as error:
        try:
            phase.append("runtime_fail", {"error": f"{type(error).__name__}: {error}"[:800]})
        except Exception:
            pass
        if writer_process is not None and writer_process.poll() is None:
            writer_process.kill()
        print(f"D409 runtime status {RUNTIME_FAIL_STATUS}")
        raise
    finally:
        if writer_socket is not None:
            writer_socket.close()
        phase.close()
        os.close(root_fd)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=("static-prep", "runtime"))
    parser.add_argument(
        "--approved-tuple-sha256",
        default=None,
        help="user-approved tuple sha256 (required for --mode runtime)",
    )
    arguments = parser.parse_args()
    try:
        if arguments.mode == "static-prep":
            if arguments.approved_tuple_sha256 is not None:
                raise D409Error("--approved-tuple-sha256 is a runtime-only argument")
            return run_static_prep()
        if arguments.approved_tuple_sha256 is None:
            raise D409Error(
                "runtime requires --approved-tuple-sha256 (user approval citing the tuple sha)"
            )
        return run_runtime(arguments.approved_tuple_sha256)
    except D409Error as error:
        print(f"D409 controller FAIL-CLOSED: {error}")
        if arguments.mode == "static-prep":
            print(f"D409 static prep status {STATIC_FAIL_STATUS}")
        return 1
    except Exception:
        traceback.print_exc()
        print("D409 controller FAIL-CLOSED: unhandled exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())
