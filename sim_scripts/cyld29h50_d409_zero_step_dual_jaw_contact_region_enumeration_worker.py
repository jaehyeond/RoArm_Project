#!/usr/bin/env python3
"""D409 zero-step dual-jaw contact-region enumeration — WORKER (cyld29h50).

Confirmed spec = claudedocs/session_20260803_grasp_g0a_d409_sweep_recovery_design_v1.md
section 2 (design v1) as amended by section 4 (confirmed delta v2; section 4
wins on conflict).  Offline only: no Isaac, no kit, no physx, no warp, no
cuda/gpu, no AppLauncher, no cook, no USD, no robot HW/serial — a scope guard
rejects the import attempt itself.  Allowed inputs: frozen JSON evidence reads
plus hppfcl/numpy queries.  Allowed writes: the run output directory under
claudedocs/runtime_logs/grasp_track/g0a_d409/ only.

WORKER RESPONSIBILITIES (this file)
  1.  Atomic exclusive claim (open('x'); pre-existing claim -> print + exit 73).
  2.  Prereg admission 2-row replay (sha256 + status; D405 lesson).
  3.  Frozen-input load + sha256 verification (d348/d368/d349/urdf, section 2.3
      confirmed table incl. P1 authority correction d339 -> d348).
  4.  URDF-literal FK (gripper joint included) + d323-form IK (HOME seed,
      position-only 5-DOF DLS, max_iter 120, pos_tol 1 mm, step clip 4 deg,
      v6 soft-limit clip; FK constants = URDF XML literals, pi/2-symbol chain
      banned).
  5.  Anchor gate (section 2.9 + section 4.2 pin): 4-channel ANY-reject at
      0.0005 mm {link5 FK pos err, gripper FK pos err, link5 dist delta,
      gripper dist delta}; old cylinder (0.017, 0.090) at the stored D349
      object pose is query-pipeline CALIBRATION ONLY.
  6.  Exhaustive enumeration of 1,239 poses: rho {0..14,500 step 250} um x
      tau {6,500..11,500 step 250} um (P2 rebase), integer-um keys, every
      pose gets a result row (silent cap 0).
  7.  Per pose: admission checks (section 2.8, FK-equivalent redefinition of
      d335 legacy_checks), link5 queried once (q5-invariant), q5 arc sweep
      (float32 linspace OPEN->0, 33 anchors, chord upper bound
      2*Rmax*sin(|dq|/2) recorded), ordered chord-bound certification
      traversal + first-crossing q5* (D351 _certify_first_contact
      semantics; bracket <= 1e-6 rad, max depth 32, endpoint validity
      contract, order_certified fail-closed gate — SCI-B1 repair R1),
      (A) fixed-jaw band,
      (B) first-crossing inner-17 + competitor exclusion + strict
      barrel_interior (R/H rebased 0.0145/0.050), pinch 4-core gates +
      remaining diagnostics, planar-gap and top-margin diagnostic columns.
  8.  4-connected region decomposition + deepest-cell representative +
      rho_R + domain-censored flag + triple-labelled comparison
      (7.881 mm / 36.033 mm labels / stall-regime unreachability note).
  9.  Canonical evidence JSON + region CSV (deterministic bytes; zero
      randomness), verdict sha256 publication BEFORE any presentation
      artifact (W-LES3 measurement-before-presentation; this worker writes
      no RRD — observability is a separate post-dual-run phase, P4-7).
  10. Preclose sentinel (summary sha included) + progress phase markers.

INTERFACE (controller binds to this contract)
  CLI     : --out-dir <absolute run dir> --prereg <absolute prereg path>
  exit 0  : contract-complete enumeration (scientific outcome is NOT an exit
            condition; zero admitted poses still exits 0)
  exit 73 : claim file already exists (refusal, nothing else written)
  exit !=0: contract fail (fail-closed; crash resume is fail-closed — any
            other pre-existing worker-owned output also fails with exit 1)
  outputs (in --out-dir):
    d409_worker_claim.json                (exclusive create, first artifact)
    d409_enumeration_evidence.json        (AUTHORITY, canonical bytes,
                                           determinism byte-compare member)
    d409_region_map.csv                   (canonical bytes, byte-compare member)
    d409_worker_summary.json              (verdict + artifact sha256s)
    d409_worker_preclose_sentinel.json    (summary sha; last artifact)
    d409_worker_phase_markers.jsonl       (progress; EXCLUDED from byte compare)
  prereg contract consumed by this worker:
    top-level key "worker_admission_rows": exactly 2 rows, each
    {"row_id": str, "path": absolute path str, "sha256": hex str}; the worker
    recomputes each file's sha256 and reports per-row PASS/FAIL (any FAIL ->
    contract fail).  If prereg carries "input_hashes", any entry whose key
    contains d348/d368/d349/urdf must match this worker's pinned sha256.

SPEC AMBIGUITIES RESOLVED (auditable decisions; spec did not fix these)
  A1. q5* is the CLEAR-side endpoint of the final bisection bracket; the
      first-contact part is the argmin part at that clear endpoint.  A
      consistency flag against the overlap-endpoint deepest collider is
      recorded (they coincide for bracket width <= 1e-6 rad).
  A2. Competitor exclusion is evaluated at the OVERLAP endpoint (colliding
      set must be a subset of inner-17 AND every non-inner gripper part must
      be strictly clear > 0).  Evaluating it at the clear endpoint would be
      vacuous (all 64 parts are clear there by definition).
  A3. Section 2.8 "admitted pose" region layer = admission_pass cells
      (literal reading).  A second diagnostic layer over full-pass cells
      (admission AND A-and-B AND pinch-core) is emitted with identical
      machinery; the admission layer is the section-2.8 authority.
  A4. Region depth metric: Euclidean mm distance in the (rho, tau) offset
      plane between cell centers; the domain exterior is treated as
      non-admitted (virtual cells one step beyond each edge); a region
      touching a domain-edge cell gets domain_censored=true (rho_R still
      reported, never gated) per the P2 edge-touch rule.
  A5. Prereg admission replay = the 2-row contract documented in INTERFACE
      above ("2행" fixed as exactly two registered {path, sha256} rows).
  A6. Anchor-gate reference distances are read from d349
      per_body[*].live_topology_exact_signed_distance_mm and cross-checked
      against the pinned literals 4.272736580324082 / 11.340262326338637 mm
      (repr equality).  The d371 evidence file is not loaded: it is not in
      the 4-file frozen-load contract.  Division of authority (SCI-W2/R7):
      d371 sha256 enforcement is CONTROLLER-side (prereg build three-way
      check); this worker's reference-value authority is the d349
      repr-equality fail-stop above, which encodes the same numbers the
      d371 evidence stores.  The worker's prereg input_hashes crosscheck
      covers only the d348/d368/d349/urdf tags by design.
  A7. Signed-distance query invokes collide/EPA refinement only when GJK
      distance <= 0 (identical output to the unconditional-collide anchor
      tool for clear configurations, so anchor-gate fidelity is unaffected;
      one query = one part-vs-cylinder evaluation for budget accounting).
  A8. Pinch-predicate inward normals are reconstructed offline as the
      nearest-triangle face normal of the witness part, oriented toward the
      cylinder witness (d351 face-normal semantics; the witness-gap direction
      is not used because it degenerates as distance -> 0).
  A9. Pre-existing claim -> exit 73; any other pre-existing worker-owned
      output -> exit 1 (fail-closed crash-resume rule, D408-R1 culture).
  A10. Pinch predicates are computed only when a valid first crossing
      exists; otherwise they are null and pinch_core_pass=false for scoring.
  A11. Per-anchor evidence granularity is a min/argmin/collision-count
      summary (full per-part rows for every anchor would be ~2.6M rows).
      Decision points (q5=OPEN and the final bracket endpoints) likewise
      serialize SUMMARIES ONLY (min/argmin, witness points, colliding-part
      list) — full 64-part signed-distance arrays are NOT serialized
      anywhere in the evidence (SCI-W1/R6 correction; the evidence
      contract of section 2.13 is satisfied by these summaries).
  A12. Arc-sweep transient-contact exclusion (SCI-B1 repair R1): adjacent
      clear-clear anchor pairs are certified contact-free by the declared
      exclusion criterion max(d_hi, d_lo) > 2*Rmax*sin(|dq|/2) (sound:
      d(q) >= d_endpoint - bound(width); sharp form of the A12 bookkeeping
      criterion — deliberate deviation from D351's stricter min() variant,
      which cannot terminate at sub-bound approach slopes); a pair that
      fails the criterion is recursively subdivided in q5 order (D351
      _certify_first_contact traverse semantics; memoized evaluations; new
      midpoint evaluations capped at CERT_TRAVERSAL_NEW_EVAL_CAP).
      Terminal-width (<= 1e-6 rad) clear-clear intervals that the bound
      cannot certify (sub-bound approach slope) are accepted only within
      CERT_NEIGHBORHOOD_RAD above the final bracket AND only when per-part
      exclusion leaves the bracket's first-contact part as the sole
      possible toucher; the resulting q5* resolution is disclosed in the
      evidence (q5_star_resolution_rad).  The
      published first crossing carries
      order_certified; any certification failure (cap exhaustion,
      terminal non-certifiable interval, midpoint stagnation, invalid
      endpoint contract) is fail-closed: order_certified=false and the
      gated b_check first_crossing_order_certified fails.  On
      certification failure the anchor-granularity bracket (no refinement
      queries; refinement="anchor_granularity_uncertified") is still
      published as an UNCERTIFIED diagnostic crossing (evidence
      continuity).  The legacy per-pose unexcluded-pair count remains a
      recorded diagnostic column.
  A13. The worker enumerates the full arc sweep for every pose (admission
      failures included) so the evidence is uniformly exhaustive; admission
      status only affects scoring, never row existence.

NULL CLAIMS (section 2.14 + P3 + W-FRZ1): stable grasp, force closure,
grasp feasibility, grasp success, push-over-absence guarantee, contact-order
dynamics, SDF superiority, transfer to other cylinders/placements are all
null.  A-and-B does NOT exclude the D362 push-over pose (d_fix 4.2727 mm is
inside the band); the ordering constraint is a pure geometric descriptor and
its push-over screening power is unverified.  Part-level masks cannot
distinguish inner/outer faces (outer 16 is a subset of inner 17; difference
= part_035 alone); inner-17 membership is a necessary condition only.
Geometry-only labels must not be promoted to standalone training.  Mass and
friction are unused.  g0a_pass=false is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Scope guard — installed before any third-party import.  Rejects the import
# attempt itself (section 2.10: Isaac/kit/physx/warp/cuda/gpu/AppLauncher/
# cook/USD/HW/serial all zero; rerun-sdk allowed by W-OPS2 but this worker
# does not import it — observability is a separate post-dual-run phase).
# ---------------------------------------------------------------------------
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "isaac",
        "isaacgym",
        "isaacsim",
        "isaaclab",
        "omni",
        "pxr",
        "usd",
        "usdrt",
        "carb",
        "kit",
        "physx",
        "physxcooking",
        "warp",
        "cuda",
        "pycuda",
        "cupy",
        "torch",
        "serial",
        "lerobot",
        "roarm_sdk",
        "pyk4a",
    }
)  # 21-root union, unified across controller/worker/writer (G3 repair R5)
_SCOPE_GUARD_VIOLATIONS: list[str] = []


class _ScopeGuardFinder:
    """sys.meta_path finder that refuses forbidden module roots outright."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        root = fullname.split(".")[0].lower()
        if root in FORBIDDEN_IMPORT_ROOTS:
            _SCOPE_GUARD_VIOLATIONS.append(fullname)
            raise ImportError(
                f"D409 scope guard: forbidden import '{fullname}' (offline-only worker)"
            )
        return None


sys.meta_path.insert(0, _ScopeGuardFinder())

import numpy as np  # noqa: E402  (allowed; imported after guard install)

# ---------------------------------------------------------------------------
# Frozen inputs (section 2.3 confirmed table; P1 authority = d348).
# ---------------------------------------------------------------------------
REPO = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
D348_PATH = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d348/attempt2/d348_callback_topology_volume_evidence.json"
D368_PATH = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d368/d368_semantic_allocation_evidence.json"
D349_PATH = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d349/d349_frozen_target_distance_measurement.json"
URDF_PATH = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
PINNED_INPUT_SHA256 = {
    "d348": "83b8c7b16181d0f5c545cfbeaa992c8ebfd69e2310dd33bce2a64234a1deaab6",
    "d368": "be2a422b0c74e4781b76a640c5312070b84876b1cb9e661d47e705ccdf789cf5",
    "d349": "5de6d14e37d6b74b202d1bb668120a6bb57221eac24ea5c751457ce9823b6300",
    "urdf": "64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2",
}
OUT_DIR_REQUIRED_FRAGMENT = "claudedocs/runtime_logs/grasp_track/g0a_d409/"

EXPECTED_NUMPY_VERSION = "1.26.0"
EXPECTED_HPPFCL_VERSION = "2.4.4"
EXPECTED_PYTHON_VERSION = "3.11.14"  # FRZ-W2 repair R9
EXPECTED_INTERPRETER = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"  # FRZ-W2 repair R9

# ---------------------------------------------------------------------------
# Real cylinder (new variable, D379 primitive rule) + confirmed placement
# (W-SCI3 operation-sequence pin; decimal literals below are re-derived and
# asserted at runtime against the pinned reprs).
# ---------------------------------------------------------------------------
CYL_RADIUS_M = 0.0145
CYL_HEIGHT_M = 0.050
CYL_X_M = 0.30000001192092896  # float64(float32(0.3)) — frozen lineage convention
TABLE_Z_M = 0.03288299962878227 - 0.045  # operation sequence (W-SCI3)
Z_CENTER_M = TABLE_Z_M + 0.025  # operation sequence (W-SCI3)
TABLE_Z_PIN_REPR = "-0.012117000371217726"
Z_CENTER_PIN_REPR = "0.012882999628782275"
CYL_CENTER_M = (CYL_X_M, 0.0, Z_CENTER_M)
CYL_TOP_Z_M = Z_CENTER_M + 0.5 * CYL_HEIGHT_M

# ---------------------------------------------------------------------------
# Grid (section 2.5 + P2 tau rebase): 59 x 21 = 1,239 poses, integer-um keys.
# tau derivation label: d335 formula with radius substitution (inherited),
# [R-8mm, R-8mm+5mm], R=14.5mm, jaw constant 8mm = FIXED_JAW_FACE_LOCAL_M
# frozen literal (d323:38).
# ---------------------------------------------------------------------------
GRID_STEP_UM = 250
RADIAL_MIN_UM = 0
RADIAL_MAX_UM = 14_500
TANGENT_MIN_UM = 6_500
TANGENT_MAX_UM = 11_500
RADIALS_UM = tuple(range(RADIAL_MIN_UM, RADIAL_MAX_UM + 1, GRID_STEP_UM))
TANGENTS_UM = tuple(range(TANGENT_MIN_UM, TANGENT_MAX_UM + 1, GRID_STEP_UM))
EXPECTED_POSE_COUNT = 1_239
ADOPTED_TANGENT_SIGN = -1.0
POSITIVE_CONTROL_KEY_UM = (7_000, 11_000)
ANTI_RETREAT_NUMERATOR_UM = 14_500  # "14.5mm - rho >= 0" (P2 rebase of 17mm-r)

# ---------------------------------------------------------------------------
# q5 closure arc (section 2.6; D351/D354 lineage).
# ---------------------------------------------------------------------------
Q5_OPEN_RAD = 1.5413000583648682
ARC_ANCHOR_COUNT = 33
BISECT_BRACKET_RAD = 1.0e-6
BISECT_MAX_ITER = 32  # numerical-resolution control, not a science tolerance

# ---------------------------------------------------------------------------
# Gates and bands (section 2.7 / 2.8 / 2.9 confirmed).
# ---------------------------------------------------------------------------
CLEAR_GATE_MM = 0.1  # D339/D349 non-interpenetration floor
FIXED_JAW_BAND_MM = (0.1, 5.0)  # (A); 5.0 = reused D330 planar-proxy constant (disclosed)
FIXED_PROX_BANDS_MM = (1.0, 5.0)
TCP_GATE_MM = 5.0
JAW_TANGENT_GATE_DEG = 15.0
ANCHOR_GATE_MM = 0.0005  # 4-channel ANY-reject (section 4.2 pin)
ANCHOR_REF_LINK5_MM_REPR = "4.272736580324082"
ANCHOR_REF_GRIPPER_MM_REPR = "11.340262326338637"
OLD_CYL_RADIUS_M = 0.017  # calibration-only (anchor gate), no D362 physics transfer
OLD_CYL_HEIGHT_M = 0.090
RIM_PROXIMITY_BAND_MM = 7.5  # W-LES1 reference band (H50-proportional), NOT a gate

# ---------------------------------------------------------------------------
# Region scoring labels (section 2.8 + W-SCI2; no single-threshold gate).
# ---------------------------------------------------------------------------
SCORE_PROXIMITY_MM = 7.881
SCORE_PROXIMITY_LABEL = (
    "D330 proximity-regime upper bound - n=5 proximity-cluster sample max, "
    "3D z-inclusive, single target, D34xH90-era historical proxy"
)
SCORE_HISTORICAL_MM = 36.033
SCORE_HISTORICAL_LABEL = (
    "historical execution-error proxy (D34xH90-era, single target, "
    "non-replica mean) - label only, NOT a pass/fail gate"
)
SCORE_STALL_REGIME_MM = (70.0, 81.0)
SCORE_XY_CLUSTER_MM = (1.7, 7.0)
SCORE_XY_CLUSTER_LABEL = (
    "xy-only projected proximity cluster - dimension-matched comparison (diagnostic)"
)
OFFSET_SPACE_LIMITATION = "offset-space distance is a TCP displacement proxy (z excluded)"

# ---------------------------------------------------------------------------
# Registered budget (W-OPS4, amended by SCI-B1 repair R1): one query = one
# part-vs-cylinder evaluation.  R1 adds the ordered chord-bound certification
# traversal (memoized; new midpoint evaluations capped at
# CERT_TRAVERSAL_NEW_EVAL_CAP x 64 queries).  Worst case per pose =
# 64 (link5) + 33x64 (anchors) + 48x64 (traversal) = 5,248 <= 5,400; worst
# run = 128 (anchor gate) + 1,239x5,248 = 6,502,400 <= 7,000,000.  Basis
# throughput (static-prep S3, 7.54-8.71 us/query) puts 7.0M at ~53-61 s,
# far below the 7,200 s timeout.
# ---------------------------------------------------------------------------
MAX_QUERIES_PER_POSE = 5_400
MAX_QUERIES_PER_RUN = 7_000_000
CERT_TRAVERSAL_NEW_EVAL_CAP = 48  # deterministic; exhaustion -> order_certified=False (fail-closed)
# Sub-resolution acceptance neighborhood (R1): a terminal-width clear-clear
# interval that the chord bound cannot certify (approach slope below the
# bound coefficient 2*Rmax mm/rad) is acceptable ONLY within this distance
# above the final bracket AND only if per-part exclusion leaves the
# bracket's first-contact part as the sole possible toucher.  Chord
# displacement at this cap = 2*Rmax*sin(3.2e-5) ~ 4.3e-3 mm, far below
# every gate in play (CLEAR_GATE 0.1 mm); disclosed in the evidence.
CERT_NEIGHBORHOOD_RAD = 6.4e-5  # = 64 * BISECT_BRACKET_RAD

# ---------------------------------------------------------------------------
# Kinematics contract (section 2.4): URDF-literal chain only.
# ---------------------------------------------------------------------------
ROOT_JOINT = "world_to_base_link"
ARM_JOINTS = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
)
GRIPPER_JOINT = "link5_to_gripper_link"
TCP_JOINT = "link5_to_hand_tcp"
HOME_DEG = (0.0, 0.0, 90.0, 0.0, 0.0, 0.0)
IK_MAX_ITER = 120
IK_POS_TOL_MM = 1.0
IK_STEP_CLIP_DEG = 4.0
IK_DAMPING = 0.002
IK_JACOBIAN_EPS_DEG = 0.01
# v6 soft limits (roarm_kinematics.JOINT_LIMITS_DEG verbatim; reachability
# authority per section 2.4 — URDF hard limits are diagnostic-only).
V6_JOINT_LIMITS_DEG = (
    ("base", -90.0, 90.0),
    ("shoulder", -30.0, 75.0),
    ("elbow", 5.0, 135.0),
    ("wrist_p", -30.0, 90.0),
    ("wrist_r", -90.0, 90.0),
    ("gripper", -10.0, 100.0),
)
FIXED_JAW_FACE_LOCAL_M = (-0.008, 0.0, 0.0)  # d323:38 frozen literal

# ---------------------------------------------------------------------------
# Masks (D368 single source; W-FRZ1 part/face limitation disclosed above).
# ---------------------------------------------------------------------------
LINK5_FIXED_EXPECTED = ("part_027", "part_029", "part_030", "part_031")
GRIPPER_INNER_EXPECTED_COUNT = 17
GRIPPER_OUTER_EXPECTED_COUNT = 16
OUTER_DIFF_PART = "part_035"

# ---------------------------------------------------------------------------
# Pinch predicates (W-SCI4 exact-name pin, d351:2568-2580 lineage).  Core
# selection rationale: opposition 1 + geometric placement 2 + closing
# direction 1.  Inputs substituted with FK + hppfcl witnesses (A8).
# ---------------------------------------------------------------------------
PINCH_CORE_NAMES = (
    "moving_and_fixed_inward_normals_opposed",
    "jaw_surface_points_on_opposite_xy_sides_of_center",
    "cylinder_center_projection_inside_contact_chord",
    "q5_decrease_moves_contact_toward_fixed_surface",
)
PINCH_DIAGNOSTIC_NAMES = (
    "fixed_normal_faces_moving_surface",
    "moving_normal_faces_fixed_surface",
    "q5_decrease_moves_along_moving_inward_normal",
    "cylinder_support_witnesses_on_opposite_xy_halfplanes",
)

# ---------------------------------------------------------------------------
# Worker-owned artifacts + exit codes + verdicts.
# ---------------------------------------------------------------------------
CLAIM_NAME = "d409_worker_claim.json"
EVIDENCE_NAME = "d409_enumeration_evidence.json"
REGION_CSV_NAME = "d409_region_map.csv"
SUMMARY_NAME = "d409_worker_summary.json"
PRECLOSE_NAME = "d409_worker_preclose_sentinel.json"
PHASE_NAME = "d409_worker_phase_markers.jsonl"
DETERMINISM_BYTE_COMPARE_MEMBERS = (EVIDENCE_NAME, REGION_CSV_NAME)

EXIT_PASS = 0
EXIT_CONTRACT_FAIL = 1
EXIT_CLAIM_PREEXIST = 73

VERDICT_COMPLETE = "D409_G0A_ZERO_STEP_DUAL_JAW_CONTACT_REGION_ENUMERATION_COMPLETE_STOP"
VERDICT_CONTRACT_FAIL = "D409_G0A_ZERO_STEP_WORKER_CONTRACT_FAIL_STOP"

CSV_COLUMNS = (
    "rho_um",
    "tau_um",
    "ik_converged",
    "ik_iterations",
    "commanded_tcp_error_mm",
    "jaw_tangent_error_deg",
    "urdf_hard_limit_violation",
    "planar_gap_d330_equiv_mm",
    "anti_retreat_margin_mm",
    "link5_min_mm",
    "link5_min_part",
    "link5_fixed4_min_mm",
    "link5_fixed4_min_part",
    "link5_witness_top_margin_mm",
    "gripper_open_min_mm",
    "gripper_open_min_part",
    "admission_pass",
    "admission_fail_reasons",
    "a_band_pass",
    "a_le_1mm",
    "a_le_5mm",
    "q5_star_rad",
    "q5_star_bracket_width_rad",
    "first_contact_part",
    "first_contact_in_inner17",
    "competitor_exclusion_pass",
    "barrel_interior_strict",
    "b_pass",
    "order_ab_pass",
    "pinch_core_pass",
    "full_pass",
    "moving_witness_top_margin_mm",
    "rim_proximal_lt_7p5mm",
    "transient_unexcluded_pairs",
    "first_crossing_order_certified",
    "cert_unresolved_intervals",
    "cert_new_evals",
    "admission_region_id",
    "admission_region_representative",
    "admission_depth_mm",
    "full_region_id",
    "full_depth_mm",
    "queries_used",
)


class _ContractFail(RuntimeError):
    """Raised for any worker contract violation (exit 1, fail-closed)."""


# ---------------------------------------------------------------------------
# Small helpers (canonical serialization + hashing + file discipline).
# ---------------------------------------------------------------------------

def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical(value: Any) -> Any:
    """Recursively convert floats to repr strings for canonical JSON bytes."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, np.floating):
        return repr(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        _canonical(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _write_new_bytes(path: Path, data: bytes) -> None:
    """Exclusive-create write with fsync (no replace, fail on pre-existence)."""
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class _PhaseLog:
    """Progress phase markers (excluded from the determinism byte compare)."""

    def __init__(self, path: Path) -> None:
        self._handle = open(path, "x", encoding="utf-8")

    def mark(self, phase: str, status: str, detail: str = "") -> None:
        row = {"phase": phase, "status": status, "detail": detail, "wall_time_s": time.time()}
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


# ---------------------------------------------------------------------------
# URDF-literal kinematics (section 2.4; pi/2-symbol chain banned).
# ---------------------------------------------------------------------------

def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def _tf(rot: np.ndarray, pos: Any) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = np.asarray(pos, dtype=np.float64)
    return out


def _axis_rot(axis: Any, q: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(q), math.sin(q)
    big_c = 1.0 - c
    return np.array(
        [
            [c + x * x * big_c, x * y * big_c - z * s, x * z * big_c + y * s],
            [y * x * big_c + z * s, c + y * y * big_c, y * z * big_c - x * s],
            [z * x * big_c - y * s, z * y * big_c + x * s, c + z * z * big_c],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_rot(quat: Any) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_angle_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    c = (float(np.trace(r1.T @ r2)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _parse_urdf_literal(urdf_path: Path) -> dict[str, dict[str, Any]]:
    tree = ET.parse(str(urdf_path))
    joints: dict[str, dict[str, Any]] = {}
    for joint in tree.getroot().iter("joint"):
        name = joint.get("name")
        origin = joint.find("origin")
        xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
        rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
        axis_el = joint.find("axis")
        axis = [float(v) for v in axis_el.get("xyz").split()] if axis_el is not None else [0.0, 0.0, 1.0]
        limit_el = joint.find("limit")
        limits = (
            [float(limit_el.get("lower")), float(limit_el.get("upper"))]
            if limit_el is not None and limit_el.get("lower") is not None
            else None
        )
        joints[name] = {
            "type": joint.get("type"),
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "limits_rad": limits,
        }
    for required in (ROOT_JOINT, *ARM_JOINTS, GRIPPER_JOINT, TCP_JOINT):
        if required not in joints:
            raise _ContractFail(f"URDF joint missing: {required}")
    return joints


def _fk_frames(joints: dict[str, dict[str, Any]], q_arm_rad: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (link5_mat, gripper_prerotation_mat, tcp_pos).

    gripper(q5) = gripper_prerotation_mat @ Rot(gripper_axis, q5); the
    gripper joint frame origin/axis live in gripper_prerotation_mat.
    """
    spec = joints[ROOT_JOINT]
    t = _tf(_rpy(*spec["rpy"]), spec["xyz"])
    for name, qi in zip(ARM_JOINTS, q_arm_rad):
        spec = joints[name]
        t = t @ _tf(_rpy(*spec["rpy"]), spec["xyz"]) @ _tf(_axis_rot(spec["axis"], float(qi)), [0, 0, 0])
    link5 = t
    g = joints[GRIPPER_JOINT]
    gripper_pre = link5 @ _tf(_rpy(*g["rpy"]), g["xyz"])
    tc = joints[TCP_JOINT]
    tcp = link5 @ _tf(_rpy(*tc["rpy"]), tc["xyz"])
    return link5, gripper_pre, tcp[:3, 3].copy()


def _gripper_mat(joints: dict[str, dict[str, Any]], gripper_pre: np.ndarray, q5_rad: float) -> np.ndarray:
    return gripper_pre @ _tf(_axis_rot(joints[GRIPPER_JOINT]["axis"], q5_rad), [0, 0, 0])


def _clip_v6(q_deg: np.ndarray) -> np.ndarray:
    out = np.asarray(q_deg, dtype=np.float64).copy()
    for idx, (_name, lo, hi) in enumerate(V6_JOINT_LIMITS_DEG[: out.shape[0]]):
        out[idx] = max(lo, min(hi, out[idx]))
    return out


def _fk_tcp_from_deg(joints: dict[str, dict[str, Any]], q_arm_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_rad = np.radians(np.asarray(q_arm_deg, dtype=np.float64))
    link5, _gripper_pre, tcp = _fk_frames(joints, q_rad)
    return tcp, link5


def _ik_jacobian(joints: dict[str, dict[str, Any]], q_arm_deg: np.ndarray, target_tcp: np.ndarray) -> np.ndarray:
    jac = np.zeros((3, 5), dtype=np.float64)
    for idx in range(5):
        qp = q_arm_deg.copy()
        qm = q_arm_deg.copy()
        qp[idx] += IK_JACOBIAN_EPS_DEG
        qm[idx] -= IK_JACOBIAN_EPS_DEG
        rp = target_tcp - _fk_tcp_from_deg(joints, qp)[0]
        rm = target_tcp - _fk_tcp_from_deg(joints, qm)[0]
        jac[:, idx] = (rp - rm) / (2.0 * IK_JACOBIAN_EPS_DEG)
    return jac


def _solve_ik(joints: dict[str, dict[str, Any]], target_tcp: np.ndarray) -> dict[str, Any]:
    """d323-form position-only 5-DOF DLS (HOME seed) with URDF-literal FK."""
    q_arm = np.asarray(HOME_DEG[:5], dtype=np.float64).copy()
    for it in range(IK_MAX_ITER + 1):
        tcp, _link5 = _fk_tcp_from_deg(joints, q_arm)
        residual = target_tcp - tcp
        pos_err_mm = float(np.linalg.norm(residual) * 1000.0)
        if pos_err_mm <= IK_POS_TOL_MM or it >= IK_MAX_ITER:
            return {
                "q_arm_deg": [float(v) for v in q_arm.tolist()],
                "converged": bool(pos_err_mm <= IK_POS_TOL_MM),
                "iterations": int(it),
                "pos_err_mm": pos_err_mm,
            }
        jac = _ik_jacobian(joints, q_arm, target_tcp)
        mat = jac @ jac.T + (IK_DAMPING ** 2) * np.eye(3, dtype=np.float64)
        try:
            # residual = target - actual; jac is d(residual)/dq, so DLS step
            # is delta = -jac^T (jac jac^T + lambda^2 I)^-1 residual (d323).
            delta = -jac.T @ np.linalg.solve(mat, residual)
        except np.linalg.LinAlgError:
            break
        max_abs = float(np.max(np.abs(delta)))
        if max_abs > IK_STEP_CLIP_DEG:
            delta *= IK_STEP_CLIP_DEG / max_abs
        q_arm = _clip_v6(q_arm + delta)[:5]
    tcp, _link5 = _fk_tcp_from_deg(joints, q_arm)
    return {
        "q_arm_deg": [float(v) for v in q_arm.tolist()],
        "converged": False,
        "iterations": int(IK_MAX_ITER),
        "pos_err_mm": float(np.linalg.norm(target_tcp - tcp) * 1000.0),
    }


def _horizontal_axis_error_deg(axis: np.ndarray, target_axis: np.ndarray) -> float:
    """d330 FK-equivalent jaw-tangent formula: xy-projection angle."""
    axis_h = np.array([float(axis[0]), float(axis[1])], dtype=np.float64)
    target_h = np.array([float(target_axis[0]), float(target_axis[1])], dtype=np.float64)
    axis_n = float(np.linalg.norm(axis_h))
    target_n = float(np.linalg.norm(target_h))
    if axis_n <= 1.0e-9 or target_n <= 1.0e-9:
        return 180.0
    dot = float(np.dot(axis_h / axis_n, target_h / target_n))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


# ---------------------------------------------------------------------------
# hppfcl query layer (anchor-tool parameter lineage; d349 signed convention).
# ---------------------------------------------------------------------------

def _build_bvh(hppfcl: Any, vertices: np.ndarray, triangles: np.ndarray) -> Any:
    model = hppfcl.BVHModelOBBRSS()
    codes = [
        int(model.beginModel(len(triangles), len(vertices))),
        int(model.addVertices(vertices)),
        int(model.addTriangles(triangles)),
        int(model.endModel()),
    ]
    if any(code != 0 for code in codes):
        raise _ContractFail(f"BVH build failed: {codes}")
    return model


def _signed_query(
    hppfcl: Any, model: Any, body_tf: Any, cylinder: Any, cylinder_tf: Any, budget: dict[str, int]
) -> dict[str, Any]:
    """One part-vs-cylinder evaluation (budget unit).  GJK distance first;
    collide/EPA refinement only on distance <= 0 (A7); signed distance =
    negative max EPA depth on overlap (d349 convention)."""
    budget["pose"] += 1
    budget["total"] += 1
    request = hppfcl.DistanceRequest(True, 1.0e-9, 1.0e-9)
    request.gjk_tolerance = 1.0e-9
    request.gjk_max_iterations = 1000
    result = hppfcl.DistanceResult()
    dist_m = float(hppfcl.distance(model, body_tf, cylinder, cylinder_tf, request, result))
    p_geom = np.asarray(result.getNearestPoint1(), dtype=np.float64)
    p_cyl = np.asarray(result.getNearestPoint2(), dtype=np.float64)
    is_collision = False
    epa_finite = None
    signed_m = dist_m
    if dist_m <= 0.0:
        creq = hppfcl.CollisionRequest()
        creq.enable_contact = True
        creq.num_max_contacts = 256
        cres = hppfcl.CollisionResult()
        hppfcl.collide(model, body_tf, cylinder, cylinder_tf, creq, cres)
        is_collision = bool(cres.isCollision())
        depths = [abs(float(cres.getContact(i).penetration_depth)) for i in range(cres.numContacts())]
        if is_collision and depths:
            epa_finite = bool(all(math.isfinite(d) for d in depths))
            signed_m = -max(depths)
        else:
            epa_finite = False
    return {
        "signed_mm": signed_m * 1000.0,
        "p_geom": p_geom,
        "p_cyl": p_cyl,
        "is_collision": is_collision,
        "epa_finite": epa_finite,
        "witness_separation_m": float(np.linalg.norm(p_geom - p_cyl)),
    }


def _eval_body(
    hppfcl: Any,
    parts: list[dict[str, Any]],
    body_mat: np.ndarray,
    cylinder: Any,
    cylinder_tf: Any,
    budget: dict[str, int],
) -> dict[str, Any]:
    """Query all parts of one body; keep per-part signed values + argmin query."""
    body_tf = hppfcl.Transform3f(body_mat[:3, :3], body_mat[:3, 3])
    per: list[tuple[str, float]] = []
    queries: dict[str, dict[str, Any]] = {}
    colliding: list[str] = []
    min_part = None
    min_query = None
    for part in parts:
        query = _signed_query(hppfcl, part["model"], body_tf, cylinder, cylinder_tf, budget)
        per.append((part["name"], float(query["signed_mm"])))
        queries[part["name"]] = query
        if query["is_collision"]:
            colliding.append(part["name"])
        if min_query is None or float(query["signed_mm"]) < float(min_query["signed_mm"]):
            min_part = part["name"]
            min_query = query
    return {
        "per": per,
        "queries": queries,
        "min_mm": float(min_query["signed_mm"]),
        "min_part": min_part,
        "min_query": min_query,
        "collision_count": len(colliding),
        "colliding_parts": colliding,
    }


# ---------------------------------------------------------------------------
# Witness-surface helpers (A8: nearest-face normal, d351 semantics).
# ---------------------------------------------------------------------------

def _point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(p - a))
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(p - b))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(p - (a + v * ab)))
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(p - c))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(p - (a + w * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(p - (b + w * (c - b))))
    total = va + vb + vc
    if abs(total) < 1.0e-30:
        return float(min(np.linalg.norm(p - a), np.linalg.norm(p - b), np.linalg.norm(p - c)))
    denom = 1.0 / total
    v = vb * denom
    w = vc * denom
    return float(np.linalg.norm(p - (a + ab * v + ac * w)))


def _nearest_face_oriented_normal(
    part: dict[str, Any], body_mat: np.ndarray, witness_world: np.ndarray, toward_world: np.ndarray
) -> dict[str, Any]:
    rot = body_mat[:3, :3]
    pos = body_mat[:3, 3]
    local_p = rot.T @ (witness_world - pos)
    vertices = part["vertices"]
    best_dist = None
    best_idx = -1
    for idx, tri in enumerate(part["triangles"]):
        d = _point_triangle_distance(local_p, vertices[tri[0]], vertices[tri[1]], vertices[tri[2]])
        if best_dist is None or d < best_dist:
            best_dist = d
            best_idx = idx
    tri = part["triangles"][best_idx]
    a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
    n_local = np.cross(b - a, c - a)
    n_local = n_local / max(float(np.linalg.norm(n_local)), 1.0e-15)
    n_world = rot @ n_local
    gap = toward_world - witness_world
    flipped = bool(float(np.dot(n_world, gap)) < 0.0)
    if flipped:
        n_world = -n_world
    return {
        "normal_world": n_world,
        "face_index": int(best_idx),
        "face_distance_m": float(best_dist),
        "oriented_flip_applied": flipped,
    }


def _feature_from_cylinder_witness(point: np.ndarray) -> dict[str, Any]:
    """d351 barrel/cap classifier, R/H rebased to the real cylinder; strict
    inequalities only, no new tolerance (D354 durable rule)."""
    p = np.asarray(point, dtype=np.float64)
    c = np.asarray(CYL_CENTER_M, dtype=np.float64)
    local = p - c
    radial = float(np.linalg.norm(local[:2]))
    abs_z = abs(float(local[2]))
    bottom = float(c[2] - 0.5 * CYL_HEIGHT_M)
    top = float(c[2] + 0.5 * CYL_HEIGHT_M)
    if not np.isfinite(p).all():
        feature = "unresolved_nonfinite"
    elif bottom < float(p[2]) < top:
        feature = "barrel_interior"
    else:
        feature = "cap_or_rim_boundary"
    return {
        "feature": feature,
        "point_world_m": p.tolist(),
        "radial_m": radial,
        "abs_height_m": abs_z,
        "barrel_surface_residual_mm": abs(radial - CYL_RADIUS_M) * 1000.0,
        "cap_surface_residual_mm": abs(abs_z - 0.5 * CYL_HEIGHT_M) * 1000.0,
        "strictly_between_cap_planes": bottom < float(p[2]) < top,
        "classification_rule": "strict z order only; no new geometric success tolerance",
    }


def _pinch_predicates(
    moving_seed: np.ndarray,
    moving_cyl_witness: np.ndarray,
    moving_normal: np.ndarray,
    fixed_seed: np.ndarray,
    fixed_cyl_witness: np.ndarray,
    fixed_normal: np.ndarray,
    joint_origin: np.ndarray,
    joint_axis: np.ndarray,
) -> dict[str, Any]:
    """d351:2546-2580 formulas; inputs = FK + hppfcl witnesses (A8)."""
    center = np.asarray(CYL_CENTER_M, dtype=np.float64)
    close_velocity = -np.cross(joint_axis, moving_seed - joint_origin)
    chord = fixed_seed - moving_seed
    chord_xy = chord[:2]
    chord_xy_norm_sq = float(np.dot(chord_xy, chord_xy))
    center_projection_t = (
        None
        if chord_xy_norm_sq <= 0.0
        else float(np.dot(center[:2] - moving_seed[:2], chord_xy) / chord_xy_norm_sq)
    )
    core = {
        "moving_and_fixed_inward_normals_opposed": float(np.dot(fixed_normal, moving_normal)) < 0.0,
        "jaw_surface_points_on_opposite_xy_sides_of_center": float(
            np.dot(moving_seed[:2] - center[:2], fixed_seed[:2] - center[:2])
        )
        < 0.0,
        "cylinder_center_projection_inside_contact_chord": center_projection_t is not None
        and 0.0 < center_projection_t < 1.0,
        "q5_decrease_moves_contact_toward_fixed_surface": float(
            np.dot(close_velocity, fixed_seed - moving_seed)
        )
        > 0.0,
    }
    diagnostics = {
        "fixed_normal_faces_moving_surface": float(np.dot(fixed_normal, moving_seed - fixed_seed)) > 0.0,
        "moving_normal_faces_fixed_surface": float(np.dot(moving_normal, fixed_seed - moving_seed)) > 0.0,
        "q5_decrease_moves_along_moving_inward_normal": float(np.dot(close_velocity, moving_normal)) > 0.0,
        "cylinder_support_witnesses_on_opposite_xy_halfplanes": float(
            np.dot(moving_cyl_witness[:2] - center[:2], fixed_cyl_witness[:2] - center[:2])
        )
        < 0.0,
    }
    assert tuple(core) == PINCH_CORE_NAMES
    assert tuple(diagnostics) == PINCH_DIAGNOSTIC_NAMES
    return {
        "core": core,
        "core_pass": all(core.values()),
        "diagnostics": diagnostics,
        "inputs": {
            "moving_seed_world_m": moving_seed.tolist(),
            "moving_cylinder_witness_world_m": moving_cyl_witness.tolist(),
            "moving_inward_normal_world": moving_normal.tolist(),
            "fixed_seed_world_m": fixed_seed.tolist(),
            "fixed_cylinder_witness_world_m": fixed_cyl_witness.tolist(),
            "fixed_inward_normal_world": fixed_normal.tolist(),
            "close_velocity_per_positive_rad": close_velocity.tolist(),
            "center_projection_t": center_projection_t,
        },
        "core_selection_rationale": "opposition 1 + geometric placement 2 + closing direction 1 (W-SCI4)",
    }


# ---------------------------------------------------------------------------
# Frozen input loading (P2 of the run flow).
# ---------------------------------------------------------------------------

def _canonical_part_hash(name: str, vertices_m: Any, topology_triangles: Any) -> str:
    """D409-canonical per-part geometry hash — identical definition to
    d409_static_prep_s1s2s3_tool.py (runtime re-verification reference)."""
    blob = json.dumps(
        {
            "name": name,
            "vertices_m": [[repr(float(c)) for c in v] for v in vertices_m],
            "topology_triangles": [[int(i) for i in t] for t in topology_triangles],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def _mask_names(d368: dict[str, Any], alloc_key: str) -> list[str]:
    node = d368["patch_allocation"][alloc_key]

    def find(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "part_names" in obj:
                return obj["part_names"]
            for value in obj.values():
                found = find(value)
                if found:
                    return found
        return None

    names = find(node)
    if not names:
        raise _ContractFail(f"D368 mask '{alloc_key}' part_names not found")
    return list(names)


def _load_frozen_inputs(hppfcl: Any) -> dict[str, Any]:
    observed = {
        "d348": _sha_file(D348_PATH),
        "d368": _sha_file(D368_PATH),
        "d349": _sha_file(D349_PATH),
        "urdf": _sha_file(URDF_PATH),
    }
    for tag, pin in PINNED_INPUT_SHA256.items():
        if observed[tag] != pin:
            raise _ContractFail(f"frozen input sha mismatch [{tag}]: {observed[tag]} != {pin}")

    d348 = json.loads(D348_PATH.read_bytes())
    d368 = json.loads(D368_PATH.read_bytes())
    d349 = json.loads(D349_PATH.read_bytes())
    joints = _parse_urdf_literal(URDF_PATH)

    parts_by_body: dict[str, list[dict[str, Any]]] = {"link5": [], "gripper_link": []}
    canonical_hashes: dict[str, dict[str, str]] = {"link5": {}, "gripper_link": {}}
    stored_payload: dict[str, dict[str, str]] = {"link5": {}, "gripper_link": {}}
    rows_by_body: dict[str, dict[str, Any]] = {"link5": {}, "gripper_link": {}}
    for row in d348["rows"]:
        rows_by_body[row["body"]][row["name"]] = row
    for body in ("link5", "gripper_link"):
        for name in sorted(rows_by_body[body]):
            inst = rows_by_body[body][name]["instance"]
            vertices = np.asarray(inst["vertices_m"], dtype=np.float64)
            triangles = np.asarray(inst["topology_triangles"], dtype=np.int64)
            parts_by_body[body].append(
                {
                    "name": name,
                    "vertices": vertices,
                    "triangles": triangles,
                    "model": _build_bvh(hppfcl, vertices, triangles),
                }
            )
            canonical_hashes[body][name] = _canonical_part_hash(
                name, inst["vertices_m"], inst["topology_triangles"]
            )
            stored_payload[body][name] = inst["payload_sha256"]
    counts_pass = len(parts_by_body["link5"]) == 64 and len(parts_by_body["gripper_link"]) == 64
    all_hashes = list(canonical_hashes["link5"].values()) + list(canonical_hashes["gripper_link"].values())
    uniqueness_pass = len(set(all_hashes)) == len(all_hashes)
    if not counts_pass:
        raise _ContractFail("d348 part counts != 64+64")
    if not uniqueness_pass:
        raise _ContractFail("d348 canonical per-part hash uniqueness violated")

    fixed = _mask_names(d368, "link5_fixed")
    inner = _mask_names(d368, "gripper_inner")
    outer = _mask_names(d368, "gripper_outer")
    mask_checks = {
        "link5_fixed_equals_design": tuple(fixed) == LINK5_FIXED_EXPECTED,
        "inner_count_17": len(inner) == GRIPPER_INNER_EXPECTED_COUNT,
        "outer_count_16": len(outer) == GRIPPER_OUTER_EXPECTED_COUNT,
        "outer_equals_inner_minus_part035": sorted(outer) == sorted(set(inner) - {OUTER_DIFF_PART}),
        "fixed_all_in_d348_link5": all(n in rows_by_body["link5"] for n in fixed),
        "inner_all_in_d348_gripper": all(n in rows_by_body["gripper_link"] for n in inner),
    }
    if not all(mask_checks.values()):
        raise _ContractFail(f"D368 mask binding failed: {mask_checks}")

    return {
        "input_sha256": observed,
        "d349": d349,
        "joints": joints,
        "parts_by_body": parts_by_body,
        "canonical_hashes": canonical_hashes,
        "stored_payload_sha256": stored_payload,
        "masks": {"link5_fixed": fixed, "gripper_inner": inner, "gripper_outer": outer},
        "mask_checks": mask_checks,
    }


# ---------------------------------------------------------------------------
# Anchor gate (section 2.9 confirmed; 4-channel ANY-reject 0.0005 mm).
# ---------------------------------------------------------------------------

def _anchor_gate(hppfcl: Any, frozen: dict[str, Any], budget: dict[str, int]) -> dict[str, Any]:
    d349 = frozen["d349"]
    joints = frozen["joints"]
    q_frozen = [float(v) for v in d349["target_state_guard"]["commanded_joint_rad_float32"]]
    raw_first = d349["distance_gate"]["authoritative_pose_streams"]["raw_first"]
    per_body = d349["distance_gate"]["per_body"]

    ref_link5 = float(per_body["link5"]["live_topology_exact_signed_distance_mm"])
    ref_gripper = float(per_body["gripper_link"]["live_topology_exact_signed_distance_mm"])
    if repr(ref_link5) != ANCHOR_REF_LINK5_MM_REPR or repr(ref_gripper) != ANCHOR_REF_GRIPPER_MM_REPR:
        raise _ContractFail(
            f"d349 anchor reference distances differ from pinned literals: "
            f"{repr(ref_link5)}, {repr(ref_gripper)}"
        )
    if repr(float(q_frozen[5])) != repr(Q5_OPEN_RAD):
        raise _ContractFail(f"frozen q5 != OPEN literal: {q_frozen[5]!r}")

    link5_mat, gripper_pre, _tcp = _fk_frames(joints, q_frozen[:5])
    gripper_mat = _gripper_mat(joints, gripper_pre, q_frozen[5])
    fk_pose_err = {}
    for body, mat in (("link5", link5_mat), ("gripper_link", gripper_mat)):
        stored = raw_first["body_poses_w"][body]
        pos_err_mm = float(
            np.linalg.norm(mat[:3, 3] - np.asarray(stored["pos_m"], dtype=np.float64)) * 1000.0
        )
        rot_err_deg = _rot_angle_deg(mat[:3, :3], _quat_wxyz_to_rot(stored["quat_wxyz"]))
        fk_pose_err[body] = {"pos_err_mm": pos_err_mm, "rot_err_deg_diagnostic": rot_err_deg}

    old_cylinder = hppfcl.Cylinder(OLD_CYL_RADIUS_M, OLD_CYL_HEIGHT_M)
    old_cyl_tf = hppfcl.Transform3f(
        _quat_wxyz_to_rot(raw_first["object_quat_wxyz"]),
        np.asarray(raw_first["object_pos_w_m"], dtype=np.float64),
    )
    dist_delta = {}
    for body, mat, ref in (("link5", link5_mat, ref_link5), ("gripper_link", gripper_mat, ref_gripper)):
        evaluation = _eval_body(hppfcl, frozen["parts_by_body"][body], mat, old_cylinder, old_cyl_tf, budget)
        dist_delta[body] = {
            "observed_min_mm": evaluation["min_mm"],
            "observed_min_part": evaluation["min_part"],
            "reference_mm": ref,
            "abs_delta_mm": abs(evaluation["min_mm"] - ref),
        }

    channels = {
        "link5_fk_pos_err_mm": fk_pose_err["link5"]["pos_err_mm"],
        "gripper_fk_pos_err_mm": fk_pose_err["gripper_link"]["pos_err_mm"],
        "link5_dist_delta_mm": dist_delta["link5"]["abs_delta_mm"],
        "gripper_dist_delta_mm": dist_delta["gripper_link"]["abs_delta_mm"],
    }
    gate_pass = all(value <= ANCHOR_GATE_MM for value in channels.values())
    gate = {
        "threshold_mm": ANCHOR_GATE_MM,
        "policy": "ANY channel > threshold -> reject",
        "channels": channels,
        "fk_pose_reproduction": fk_pose_err,
        "distance_reproduction": dist_delta,
        "calibration_semantics": (
            "old cylinder (0.017,0.090) at the stored D349 object pose is "
            "query-pipeline calibration only; no D362-era physics result is "
            "transferred to the real D29xH50 cylinder (D379)."
        ),
        "distance_channel_discrimination_note": (
            "pi/2 gripper dist delta 0.0001777mm is below threshold - the "
            "distance-channel discrimination is link5 only (session doc 4.2 pin)."
        ),
        "pass": gate_pass,
    }
    if not gate_pass:
        raise _ContractFail(f"anchor gate ANY-reject fired: {channels}")
    return gate


# ---------------------------------------------------------------------------
# Pose enumeration.
# ---------------------------------------------------------------------------

def _target_geometry(rho_um: int, tau_um: int) -> dict[str, Any]:
    center = np.asarray(CYL_CENTER_M, dtype=np.float64)
    radial_raw = np.array([center[0], center[1], 0.0], dtype=np.float64)
    radial = radial_raw / float(np.linalg.norm(radial_raw[:2]))
    tangent = np.array([-radial[1], radial[0], 0.0], dtype=np.float64) * ADOPTED_TANGENT_SIGN
    target = center.copy()
    target -= radial * (float(rho_um) * 1.0e-6)
    target -= tangent * (float(tau_um) * 1.0e-6)
    target[2] = Z_CENTER_M
    return {"radial": radial, "tangent": tangent, "target_tcp": target}


def _urdf_hard_limit_violation(joints: dict[str, dict[str, Any]], q_arm_rad: np.ndarray) -> bool:
    for name, q in zip(ARM_JOINTS, q_arm_rad):
        limits = joints[name]["limits_rad"]
        if limits is not None and not (limits[0] <= float(q) <= limits[1]):
            return True
    return False


def _evaluate_pose(
    hppfcl: Any,
    frozen: dict[str, Any],
    cylinder: Any,
    cylinder_tf: Any,
    anchors_f32: np.ndarray,
    rmax_m: float,
    rho_um: int,
    tau_um: int,
    budget: dict[str, int],
) -> dict[str, Any]:
    joints = frozen["joints"]
    fixed_mask = set(frozen["masks"]["link5_fixed"])
    inner_mask = set(frozen["masks"]["gripper_inner"])
    budget["pose"] = 0

    geometry = _target_geometry(rho_um, tau_um)
    target_tcp = geometry["target_tcp"]
    tangent = geometry["tangent"]
    ik = _solve_ik(joints, target_tcp)
    q_arm_deg = np.asarray(ik["q_arm_deg"], dtype=np.float64)
    q_arm_rad = np.radians(q_arm_deg)
    link5_mat, gripper_pre, tcp_pos = _fk_frames(joints, q_arm_rad)
    commanded_tcp_error_mm = float(np.linalg.norm(tcp_pos - target_tcp) * 1000.0)
    jaw_tangent_error_deg = _horizontal_axis_error_deg(link5_mat[:3, 0], tangent)
    hard_limit_violation = _urdf_hard_limit_violation(joints, q_arm_rad)

    # D330 planar-gap equivalent (diagnostic column, P3/W-SCI1; NOT the gate).
    fixed_jaw_face = tcp_pos + link5_mat[:3, :3] @ np.asarray(FIXED_JAW_FACE_LOCAL_M, dtype=np.float64)
    center = np.asarray(CYL_CENTER_M, dtype=np.float64)
    side_plane = center - tangent * CYL_RADIUS_M
    planar_gap_mm = float(np.dot(side_plane[:2] - fixed_jaw_face[:2], tangent[:2]) * 1000.0)

    anti_retreat_margin_mm = float(ANTI_RETREAT_NUMERATOR_UM - rho_um) / 1000.0

    # link5 (fixed jaw): q5-invariant -> exactly one all-64 evaluation per pose.
    link5_eval = _eval_body(hppfcl, frozen["parts_by_body"]["link5"], link5_mat, cylinder, cylinder_tf, budget)
    fixed_pairs = [(name, value) for name, value in link5_eval["per"] if name in fixed_mask]
    fixed4_min_part, fixed4_min_mm = min(fixed_pairs, key=lambda item: (item[1], item[0]))
    fixed4_query = link5_eval["queries"][fixed4_min_part]
    link5_top_margin_mm = float((CYL_TOP_Z_M - float(link5_eval["min_query"]["p_cyl"][2])) * 1000.0)

    # Arc sweep (moving jaw): 33 float32 anchors OPEN -> 0.
    anchor_rows: list[dict[str, Any]] = []
    anchor_evals: list[dict[str, Any]] = []
    for anchor in anchors_f32:
        q5 = float(anchor)
        gmat = _gripper_mat(joints, gripper_pre, q5)
        evaluation = _eval_body(
            hppfcl, frozen["parts_by_body"]["gripper_link"], gmat, cylinder, cylinder_tf, budget
        )
        anchor_evals.append(evaluation)
        anchor_rows.append(
            {
                "q5_rad": q5,
                "min_mm": evaluation["min_mm"],
                "min_part": evaluation["min_part"],
                "collision_parts": evaluation["collision_count"],
            }
        )
    open_eval = anchor_evals[0]

    chord_bounds_mm = []
    transient_unexcluded = 0
    for idx in range(len(anchor_rows) - 1):
        dq = abs(anchor_rows[idx]["q5_rad"] - anchor_rows[idx + 1]["q5_rad"])
        bound_mm = 2.0 * rmax_m * math.sin(0.5 * dq) * 1000.0
        chord_bounds_mm.append(bound_mm)
        d_i = anchor_rows[idx]["min_mm"]
        d_j = anchor_rows[idx + 1]["min_mm"]
        if d_i > 0.0 and d_j > 0.0 and max(d_i, d_j) <= bound_mm:
            transient_unexcluded += 1

    # First crossing + ordered chord-bound certification (A1/A2; SCI-B1
    # repair R1 — D351 _certify_first_contact traverse semantics, memoized
    # evaluations, deterministic new-evaluation cap, fail-closed).
    overlap_idx = None
    for idx, row in enumerate(anchor_rows):
        if row["min_mm"] <= 0.0:
            overlap_idx = idx
            break

    eval_memo: dict[float, dict[str, Any]] = {}
    for row, evaluation in zip(anchor_rows, anchor_evals):
        eval_memo[float(row["q5_rad"])] = evaluation
    cert_state = {"new_evals": 0}

    def _memo_eval(q5_value: float) -> dict[str, Any] | None:
        cached = eval_memo.get(q5_value)
        if cached is not None:
            return cached
        if cert_state["new_evals"] >= CERT_TRAVERSAL_NEW_EVAL_CAP:
            return None
        cert_state["new_evals"] += 1
        evaluation = _eval_body(
            hppfcl,
            frozen["parts_by_body"]["gripper_link"],
            _gripper_mat(joints, gripper_pre, q5_value),
            cylinder,
            cylinder_tf,
            budget,
        )
        eval_memo[q5_value] = evaluation
        return evaluation

    def _clear_valid(evaluation: dict[str, Any]) -> bool:
        return (
            evaluation["min_mm"] > 0.0
            and float(evaluation["min_query"]["witness_separation_m"]) > 0.0
        )

    def _overlap_valid(evaluation: dict[str, Any]) -> bool:
        return evaluation["min_mm"] <= 0.0 and evaluation["min_query"]["epa_finite"] is True

    certified_intervals = 0
    cert_unresolved: list[dict[str, Any]] = []
    sub_resolution: list[dict[str, Any]] = []
    cert_bracket: dict[str, Any] | None = None

    def _traverse(q_hi: float, q_lo: float, depth: int) -> bool:
        """Ordered certification; True = stop (bracket found or hard-unresolved)."""
        nonlocal certified_intervals, cert_bracket
        hi_e = _memo_eval(q_hi)
        lo_e = _memo_eval(q_lo)
        if hi_e is None or lo_e is None:
            cert_unresolved.append(
                {
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                    "depth": depth,
                    "reason": "certification_eval_cap_exhausted",
                }
            )
            return True
        width = q_hi - q_lo
        bound_mm = 2.0 * rmax_m * math.sin(0.5 * width) * 1000.0
        # Certification criterion = the declared A12 exclusion criterion
        # (section 2.6): contact inside a clear-clear interval is excluded
        # iff EITHER endpoint distance exceeds the interval chord bound,
        # because d(q) >= d_endpoint - bound(width) (1-Lipschitz w.r.t. the
        # chord displacement).  max() is the sharp sound form; D351 used the
        # stricter min() variant — deliberate, disposition-recorded
        # deviation (R1): min() cannot terminate when the approach slope is
        # below the bound coefficient 2*Rmax (mm/rad).
        if (
            _clear_valid(hi_e)
            and _clear_valid(lo_e)
            and max(hi_e["min_mm"], lo_e["min_mm"]) > bound_mm
        ):
            certified_intervals += 1
            return False
        if width <= BISECT_BRACKET_RAD or depth >= BISECT_MAX_ITER:
            if _clear_valid(hi_e) and _overlap_valid(lo_e):
                cert_bracket = {"q_hi": q_hi, "q_lo": q_lo, "depth": depth}
                return True
            if (
                width <= BISECT_BRACKET_RAD
                and _clear_valid(hi_e)
                and _clear_valid(lo_e)
            ):
                # Sub-resolution clear-clear interval the chord bound cannot
                # certify (approach slope below the bound coefficient).
                # Record and CONTINUE the ordered traversal; acceptance is
                # adjudicated after the traversal (bracket adjacency +
                # per-part exclusion — R1 resolution rule).
                sub_resolution.append(
                    {
                        "q_hi": q_hi,
                        "q_lo": q_lo,
                        "depth": depth,
                        "max_d_mm": max(hi_e["min_mm"], lo_e["min_mm"]),
                        "bound_mm": bound_mm,
                    }
                )
                return False
            cert_unresolved.append(
                {
                    "q_hi": q_hi,
                    "q_lo": q_lo,
                    "depth": depth,
                    "reason": "terminal_interval_not_certified_or_bracketed",
                }
            )
            return True
        mid = 0.5 * (q_hi + q_lo)
        if not (q_lo < mid < q_hi):
            cert_unresolved.append(
                {"q_hi": q_hi, "q_lo": q_lo, "depth": depth, "reason": "midpoint_stagnation"}
            )
            return True
        if _traverse(q_hi, mid, depth + 1):
            return True
        return _traverse(mid, q_lo, depth + 1)

    open_anchor_overlap = anchor_rows[0]["min_mm"] <= 0.0
    if not open_anchor_overlap:
        if not _clear_valid(anchor_evals[0]):
            cert_unresolved.append(
                {
                    "q": float(anchor_rows[0]["q5_rad"]),
                    "reason": "open_endpoint_not_valid_clear",
                }
            )
        else:
            for idx in range(len(anchor_rows) - 1):
                if _traverse(
                    float(anchor_rows[idx]["q5_rad"]),
                    float(anchor_rows[idx + 1]["q5_rad"]),
                    0,
                ):
                    break

    # Sub-resolution adjudication (R1 resolution rule): accepted iff a
    # bracket exists, the interval lies within CERT_NEIGHBORHOOD_RAD above
    # the bracket clear endpoint, and per-part exclusion leaves the
    # bracket's first-contact part as the only possible toucher inside it.
    sub_res_accepted: list[dict[str, Any]] = []
    sub_res_rejected: list[dict[str, Any]] = []
    for interval in sub_resolution:
        accepted = False
        if cert_bracket is not None:
            bracket_hi = float(cert_bracket["q_hi"])
            adjacent = (
                interval["q_lo"] >= bracket_hi - 1.0e-12
                and (interval["q_hi"] - bracket_hi) <= CERT_NEIGHBORHOOD_RAD
            )
            if adjacent:
                hi_e = eval_memo[interval["q_hi"]]
                lo_e = eval_memo[interval["q_lo"]]
                first_part = eval_memo[bracket_hi]["min_part"]
                per_hi = dict(hi_e["per"])
                per_lo = dict(lo_e["per"])
                accepted = all(
                    max(per_hi[name], per_lo[name]) > interval["bound_mm"]
                    for name in per_hi
                    if name != first_part
                )
        entry = dict(interval)
        entry["accepted"] = accepted
        (sub_res_accepted if accepted else sub_res_rejected).append(entry)

    order_certified = bool(
        (
            cert_bracket is not None
            and not cert_unresolved
            and not sub_res_rejected
        )
        or (
            overlap_idx is None
            and cert_bracket is None
            and not cert_unresolved
            and not sub_resolution
            and not open_anchor_overlap
        )
    )
    q5_star_resolution_rad = None
    if cert_bracket is not None:
        span_above = max(
            (entry["q_hi"] - float(cert_bracket["q_hi"]) for entry in sub_res_accepted),
            default=0.0,
        )
        q5_star_resolution_rad = float(
            (float(cert_bracket["q_hi"]) - float(cert_bracket["q_lo"])) + max(0.0, span_above)
        )
    order_certification = {
        "rule": (
            "D351 _certify_first_contact traverse semantics (R1): every "
            "clear-clear interval preceding the first crossing must be "
            "chord-bound certified or recursively subdivided; certification "
            "criterion = max(d_hi, d_lo) > 2*Rmax*sin(|dq|/2) (the declared "
            "A12 exclusion criterion, sharp and sound: d(q) >= d_endpoint - "
            "bound(width); deliberate deviation from D351's stricter min() "
            "variant which cannot terminate at sub-bound approach slopes). "
            "Terminal-width clear-clear intervals the bound cannot certify "
            "are accepted ONLY inside CERT_NEIGHBORHOOD_RAD above the "
            "bracket with per-part exclusion of every part except the "
            "first-contact part (q5* resolution disclosed).  Hard-unresolved "
            "intervals, rejected sub-resolution intervals, or "
            "evaluation-cap exhaustion are fail-closed "
            "(order_certified=false).  Precision disclosure (SCI-R1-W1): "
            "all distance comparisons use raw hppfcl GJK values "
            "(DistanceRequest/GJK tolerance 1e-9 m = 1e-6 mm) with no "
            "additional numerical-error allowance, so a certification whose "
            "margin is below that tolerance would be theoretically unsound "
            "(observed margins are orders of magnitude larger)"
        ),
        "certified": order_certified,
        "certified_intervals": int(certified_intervals),
        "unresolved": cert_unresolved,
        "sub_resolution_accepted": sub_res_accepted,
        "sub_resolution_rejected": sub_res_rejected,
        "neighborhood_cap_rad": CERT_NEIGHBORHOOD_RAD,
        "q5_star_resolution_rad": q5_star_resolution_rad,
        "new_evals": int(cert_state["new_evals"]),
        "new_eval_cap": CERT_TRAVERSAL_NEW_EVAL_CAP,
    }

    def _crossing_from_bracket(
        hi: float,
        lo: float,
        hi_eval: dict[str, Any],
        lo_eval: dict[str, Any],
        iterations: int,
        refinement: str,
    ) -> dict[str, Any]:
        endpoint_valid = _clear_valid(hi_eval) and _overlap_valid(lo_eval)
        width = hi - lo
        first_contact_part = hi_eval["min_part"]
        non_inner_min = min(
            (value for name, value in lo_eval["per"] if name not in inner_mask), default=None
        )
        return {
            "found": True,
            "refinement": refinement,
            "endpoint_contract_valid": bool(endpoint_valid),
            "bisect_iterations": int(iterations),
            "bracket_lo_rad": lo,
            "bracket_hi_rad": hi,
            "width_rad": width,
            "width_le_contract": bool(width <= BISECT_BRACKET_RAD),
            "q5_star_rad": hi,
            "q5_star_in_open_interval": bool(0.0 < hi < Q5_OPEN_RAD),
            "first_contact_part": first_contact_part,
            "first_contact_part_consistent_with_overlap_deepest": bool(
                first_contact_part == lo_eval["min_part"]
            ),
            "clear_endpoint": {
                "min_mm": hi_eval["min_mm"],
                "part": hi_eval["min_part"],
                "witness_geometry_m": hi_eval["min_query"]["p_geom"].tolist(),
                "witness_cylinder_m": hi_eval["min_query"]["p_cyl"].tolist(),
            },
            "overlap_endpoint": {
                "min_mm": lo_eval["min_mm"],
                "deepest_part": lo_eval["min_part"],
                "colliding_parts": list(lo_eval["colliding_parts"]),
                "colliding_all_inner": bool(set(lo_eval["colliding_parts"]) <= inner_mask),
                "non_inner_min_mm": non_inner_min,
                "non_inner_all_clear": bool(non_inner_min is not None and non_inner_min > 0.0),
            },
        }

    crossing: dict[str, Any]
    if open_anchor_overlap:
        crossing = {"found": False, "reason": "open_anchor_overlap"}
    elif cert_bracket is not None:
        crossing = _crossing_from_bracket(
            float(cert_bracket["q_hi"]),
            float(cert_bracket["q_lo"]),
            eval_memo[float(cert_bracket["q_hi"])],
            eval_memo[float(cert_bracket["q_lo"])],
            int(cert_bracket["depth"]),
            "certified_traversal_bracket",
        )
    elif overlap_idx is not None:
        # Certification stopped unresolved but an anchor-granularity overlap
        # exists: publish the UNCERTIFIED anchor bracket as a diagnostic
        # crossing (A12/R1 evidence continuity; no refinement queries —
        # order_certified=false already fails (B) fail-closed).
        crossing = _crossing_from_bracket(
            float(anchor_rows[overlap_idx - 1]["q5_rad"]),
            float(anchor_rows[overlap_idx]["q5_rad"]),
            anchor_evals[overlap_idx - 1],
            anchor_evals[overlap_idx],
            0,
            "anchor_granularity_uncertified",
        )
    else:
        crossing = {"found": False, "reason": "no_crossing_through_closed"}

    # Admission (section 2.8 confirmed; FK-equivalent redefinition).
    admission_checks = {
        "ik_converged": bool(ik["converged"]),
        "commanded_tcp_error_le_5mm": commanded_tcp_error_mm <= TCP_GATE_MM,
        "jaw_tangent_le_15deg": jaw_tangent_error_deg <= JAW_TANGENT_GATE_DEG,
        "link5_all64_noninterpenetration_ge_0p1mm": link5_eval["min_mm"] >= CLEAR_GATE_MM,
        "gripper_open_all64_noninterpenetration_ge_0p1mm": open_eval["min_mm"] >= CLEAR_GATE_MM,
        "anti_retreat_14p5mm_minus_rho_nonnegative": (ANTI_RETREAT_NUMERATOR_UM - rho_um) >= 0,
    }
    admission_pass = all(admission_checks.values())
    admission_fail_reasons = sorted(name for name, ok in admission_checks.items() if not ok)

    # (A) fixed-jaw band over the link5 4-mask.
    a_band_pass = FIXED_JAW_BAND_MM[0] <= fixed4_min_mm <= FIXED_JAW_BAND_MM[1]
    a_le_1mm = fixed4_min_mm <= FIXED_PROX_BANDS_MM[0]
    a_le_5mm = fixed4_min_mm <= FIXED_PROX_BANDS_MM[1]

    # (B) moving-jaw first crossing.
    feature = None
    if crossing["found"] and crossing["endpoint_contract_valid"]:
        feature = _feature_from_cylinder_witness(
            np.asarray(crossing["clear_endpoint"]["witness_cylinder_m"], dtype=np.float64)
        )
    b_checks = {
        "crossing_exists_in_open_interval": bool(
            crossing["found"]
            and crossing.get("endpoint_contract_valid", False)
            and crossing.get("q5_star_in_open_interval", False)
            and crossing.get("width_le_contract", False)
        ),
        "first_crossing_order_certified": bool(
            crossing["found"] and order_certification["certified"]
        ),
        "first_contact_part_in_inner17": bool(
            crossing.get("first_contact_part") in inner_mask if crossing["found"] else False
        ),
        "competitor_exclusion": bool(
            crossing["found"]
            and crossing.get("overlap_endpoint", {}).get("colliding_all_inner", False)
            and crossing.get("overlap_endpoint", {}).get("non_inner_all_clear", False)
        ),
        "cylinder_witness_barrel_interior_strict": bool(
            feature is not None and feature["feature"] == "barrel_interior"
        ),
    }
    b_pass = all(b_checks.values())
    order_ab_pass = bool(a_band_pass and b_pass)

    # Pinch predicates (A10: only with a valid crossing).
    pinch = None
    moving_top_margin_mm = None
    if crossing["found"] and crossing["endpoint_contract_valid"]:
        q5_star = float(crossing["q5_star_rad"])
        gmat_star = _gripper_mat(joints, gripper_pre, q5_star)
        moving_seed = np.asarray(crossing["clear_endpoint"]["witness_geometry_m"], dtype=np.float64)
        moving_cyl = np.asarray(crossing["clear_endpoint"]["witness_cylinder_m"], dtype=np.float64)
        moving_top_margin_mm = float((CYL_TOP_Z_M - float(moving_cyl[2])) * 1000.0)
        moving_part = next(
            part
            for part in frozen["parts_by_body"]["gripper_link"]
            if part["name"] == crossing["first_contact_part"]
        )
        moving_face = _nearest_face_oriented_normal(moving_part, gmat_star, moving_seed, moving_cyl)
        fixed_seed = np.asarray(fixed4_query["p_geom"], dtype=np.float64)
        fixed_cyl = np.asarray(fixed4_query["p_cyl"], dtype=np.float64)
        fixed_part = next(
            part for part in frozen["parts_by_body"]["link5"] if part["name"] == fixed4_min_part
        )
        fixed_face = _nearest_face_oriented_normal(fixed_part, link5_mat, fixed_seed, fixed_cyl)
        axis_local = np.asarray(joints[GRIPPER_JOINT]["axis"], dtype=np.float64)
        axis_local = axis_local / float(np.linalg.norm(axis_local))
        joint_axis_world = gripper_pre[:3, :3] @ axis_local
        pinch = _pinch_predicates(
            moving_seed,
            moving_cyl,
            np.asarray(moving_face["normal_world"], dtype=np.float64),
            fixed_seed,
            fixed_cyl,
            np.asarray(fixed_face["normal_world"], dtype=np.float64),
            gripper_pre[:3, 3].copy(),
            joint_axis_world,
        )
        pinch["moving_face"] = {
            "face_index": moving_face["face_index"],
            "face_distance_m": moving_face["face_distance_m"],
            "oriented_flip_applied": moving_face["oriented_flip_applied"],
        }
        pinch["fixed_face"] = {
            "part": fixed4_min_part,
            "face_index": fixed_face["face_index"],
            "face_distance_m": fixed_face["face_distance_m"],
            "oriented_flip_applied": fixed_face["oriented_flip_applied"],
        }
    pinch_core_pass = bool(pinch is not None and pinch["core_pass"])
    full_pass = bool(admission_pass and order_ab_pass and pinch_core_pass)
    rim_proximal = (
        None if moving_top_margin_mm is None else bool(moving_top_margin_mm < RIM_PROXIMITY_BAND_MM)
    )

    if budget["pose"] > MAX_QUERIES_PER_POSE:
        raise _ContractFail(
            f"registered per-pose query budget exceeded at ({rho_um},{tau_um}): "
            f"{budget['pose']} > {MAX_QUERIES_PER_POSE}"
        )

    return {
        "key_um": [int(rho_um), int(tau_um)],
        "target_tcp_m": target_tcp.tolist(),
        "ik": {
            "converged": bool(ik["converged"]),
            "iterations": int(ik["iterations"]),
            "pos_err_mm": float(ik["pos_err_mm"]),
        },
        "commanded_joint_deg": [float(v) for v in q_arm_deg.tolist()] + [math.degrees(Q5_OPEN_RAD)],
        "commanded_tcp_m": tcp_pos.tolist(),
        "commanded_tcp_error_mm": commanded_tcp_error_mm,
        "jaw_tangent_error_deg": jaw_tangent_error_deg,
        "urdf_hard_limit_violation_diagnostic": hard_limit_violation,
        "planar_gap_d330_equiv_mm": planar_gap_mm,
        "anti_retreat_margin_mm": anti_retreat_margin_mm,
        "link5": {
            "min_mm": link5_eval["min_mm"],
            "min_part": link5_eval["min_part"],
            "witness_geometry_m": link5_eval["min_query"]["p_geom"].tolist(),
            "witness_cylinder_m": link5_eval["min_query"]["p_cyl"].tolist(),
            "witness_top_margin_mm": link5_top_margin_mm,
            "collision_count": int(link5_eval["collision_count"]),
            "fixed4_min_mm": fixed4_min_mm,
            "fixed4_min_part": fixed4_min_part,
            "fixed4_witness_geometry_m": fixed4_query["p_geom"].tolist(),
            "fixed4_witness_cylinder_m": fixed4_query["p_cyl"].tolist(),
        },
        "gripper_open": {
            "min_mm": open_eval["min_mm"],
            "min_part": open_eval["min_part"],
            "collision_count": int(open_eval["collision_count"]),
        },
        "admission": {
            "checks": admission_checks,
            "pass": admission_pass,
            "fail_reasons": admission_fail_reasons,
        },
        "a_fixed_jaw": {
            "d_fix_mm": fixed4_min_mm,
            "band_mm": list(FIXED_JAW_BAND_MM),
            "band_pass": a_band_pass,
            "le_1mm": a_le_1mm,
            "le_5mm": a_le_5mm,
        },
        "arc_sweep": {
            "anchor_count": len(anchor_rows),
            "anchors": anchor_rows,
            "chord_bound_max_mm": max(chord_bounds_mm),
            "transient_unexcluded_pairs": int(transient_unexcluded),
        },
        "first_crossing": crossing,
        "order_certification": order_certification,
        "b_moving_jaw": {
            "checks": b_checks,
            "pass": b_pass,
            "cylinder_witness_feature": feature,
        },
        "order_constraint_ab_pass": order_ab_pass,
        "pinch": pinch,
        "pinch_core_pass": pinch_core_pass,
        "moving_witness_top_margin_mm": moving_top_margin_mm,
        "rim_proximal_lt_7p5mm": rim_proximal,
        "full_pass": full_pass,
        "queries_used": int(budget["pose"]),
    }


# ---------------------------------------------------------------------------
# Region analysis (section 2.8 + P2 edge-touch; A3/A4 resolutions).
# ---------------------------------------------------------------------------

def _region_analysis(rows: dict[tuple[int, int], dict[str, Any]], pass_key: str, layer: str) -> dict[str, Any]:
    cells = sorted(key for key, row in rows.items() if row[pass_key])
    cell_set = set(cells)
    all_keys = [(r, t) for r in RADIALS_UM for t in TANGENTS_UM]
    non_cells = [key for key in all_keys if key not in cell_set]

    # 4-connected components (stack-based DFS in sorted order for
    # deterministic ids; membership is traversal-order independent — R19).
    region_of: dict[tuple[int, int], str] = {}
    regions: list[dict[str, Any]] = []
    for seed in cells:
        if seed in region_of:
            continue
        region_id = f"{layer[:3]}_{len(regions) + 1:03d}"
        stack = [seed]
        members: list[tuple[int, int]] = []
        region_of[seed] = region_id
        while stack:
            current = stack.pop()
            members.append(current)
            r, t = current
            for nb in ((r - GRID_STEP_UM, t), (r + GRID_STEP_UM, t), (r, t - GRID_STEP_UM), (r, t + GRID_STEP_UM)):
                if nb in cell_set and nb not in region_of:
                    region_of[nb] = region_id
                    stack.append(nb)
        members.sort()
        regions.append({"region_id": region_id, "members": members})

    # Depth per admitted cell (A4): Euclidean um distance to the nearest
    # non-admitted domain cell or to the domain exterior (virtual cells one
    # step beyond each edge), reported in mm.
    depth_of: dict[tuple[int, int], float] = {}
    for r, t in cells:
        exterior_um = min(
            r - RADIAL_MIN_UM + GRID_STEP_UM,
            RADIAL_MAX_UM - r + GRID_STEP_UM,
            t - TANGENT_MIN_UM + GRID_STEP_UM,
            TANGENT_MAX_UM - t + GRID_STEP_UM,
        )
        best = float(exterior_um)
        for rn, tn in non_cells:
            d = math.hypot(r - rn, t - tn)
            if d < best:
                best = d
        depth_of[(r, t)] = best / 1000.0

    region_entries = []
    for region in regions:
        members = region["members"]
        edge_cells = [
            (r, t)
            for r, t in members
            if r in (RADIAL_MIN_UM, RADIAL_MAX_UM) or t in (TANGENT_MIN_UM, TANGENT_MAX_UM)
        ]
        ranked = sorted(members, key=lambda key: (-depth_of[key], key[0], key[1]))
        representative = ranked[0]
        rho_r_mm = depth_of[representative]
        margins = [
            rows[key]["moving_witness_top_margin_mm"]
            for key in members
            if rows[key]["moving_witness_top_margin_mm"] is not None
        ]
        rim_count = sum(1 for margin in margins if margin < RIM_PROXIMITY_BAND_MM)
        region_entries.append(
            {
                "region_id": region["region_id"],
                "layer": layer,
                "cell_count": len(members),
                "cells_um": [[r, t] for r, t in members],
                "edge_cell_count": len(edge_cells),
                "domain_censored": bool(edge_cells),
                "representative_um": [representative[0], representative[1]],
                "rho_R_mm": rho_r_mm,
                "depth_metric": (
                    "Euclidean mm in (rho,tau) offset plane between cell centers; "
                    "domain exterior counted as non-admitted; z excluded"
                ),
                "rim_proximity_cell_fraction": (rim_count / len(members)) if members else None,
                "rim_proximity_margin_null_count": len(members) - len(margins),
                "score_triple": {
                    "proximity_regime": {
                        "threshold_mm": SCORE_PROXIMITY_MM,
                        "label": SCORE_PROXIMITY_LABEL,
                        "rho_R_exceeds": bool(rho_r_mm > SCORE_PROXIMITY_MM),
                    },
                    "historical_proxy": {
                        "threshold_mm": SCORE_HISTORICAL_MM,
                        "label": SCORE_HISTORICAL_LABEL,
                        "rho_R_exceeds": bool(rho_r_mm > SCORE_HISTORICAL_MM),
                        "standalone_gate": None,
                    },
                    "stall_regime": {
                        "range_mm": list(SCORE_STALL_REGIME_MM),
                        "note": (
                            "unreachable: max offset-space radius within the domain "
                            "is sqrt(14.5^2+11.5^2)=18.5mm < 70mm"
                        ),
                    },
                    "xy_projected_cluster_diagnostic": {
                        "range_mm": list(SCORE_XY_CLUSTER_MM),
                        "label": SCORE_XY_CLUSTER_LABEL,
                    },
                    "limitation": OFFSET_SPACE_LIMITATION,
                },
            }
        )
    return {
        "layer": layer,
        "pass_key": pass_key,
        "cell_count": len(cells),
        "region_count": len(regions),
        "regions": region_entries,
        "region_of": {f"{r}:{t}": region_of[(r, t)] for r, t in cells},
        "depth_of": {f"{r}:{t}": depth_of[(r, t)] for r, t in cells},
    }


# ---------------------------------------------------------------------------
# CSV serialization (canonical bytes; determinism byte-compare member).
# ---------------------------------------------------------------------------

def _region_csv_bytes(
    rows: dict[tuple[int, int], dict[str, Any]],
    admission_layer: dict[str, Any],
    full_layer: dict[str, Any],
) -> bytes:
    adm_region_of = admission_layer["region_of"]
    adm_depth_of = admission_layer["depth_of"]
    adm_rep = {
        entry["region_id"]: f"{entry['representative_um'][0]}:{entry['representative_um'][1]}"
        for entry in admission_layer["regions"]
    }
    full_region_of = full_layer["region_of"]
    full_depth_of = full_layer["depth_of"]
    lines = [",".join(CSV_COLUMNS)]
    for key in sorted(rows):
        row = rows[key]
        tag = f"{key[0]}:{key[1]}"
        crossing = row["first_crossing"]
        adm_region = adm_region_of.get(tag)
        values = (
            key[0],
            key[1],
            row["ik"]["converged"],
            row["ik"]["iterations"],
            row["commanded_tcp_error_mm"],
            row["jaw_tangent_error_deg"],
            row["urdf_hard_limit_violation_diagnostic"],
            row["planar_gap_d330_equiv_mm"],
            row["anti_retreat_margin_mm"],
            row["link5"]["min_mm"],
            row["link5"]["min_part"],
            row["link5"]["fixed4_min_mm"],
            row["link5"]["fixed4_min_part"],
            row["link5"]["witness_top_margin_mm"],
            row["gripper_open"]["min_mm"],
            row["gripper_open"]["min_part"],
            row["admission"]["pass"],
            "|".join(row["admission"]["fail_reasons"]),
            row["a_fixed_jaw"]["band_pass"],
            row["a_fixed_jaw"]["le_1mm"],
            row["a_fixed_jaw"]["le_5mm"],
            crossing.get("q5_star_rad") if crossing["found"] else None,
            crossing.get("width_rad") if crossing["found"] else None,
            crossing.get("first_contact_part") if crossing["found"] else None,
            row["b_moving_jaw"]["checks"]["first_contact_part_in_inner17"],
            row["b_moving_jaw"]["checks"]["competitor_exclusion"],
            row["b_moving_jaw"]["checks"]["cylinder_witness_barrel_interior_strict"],
            row["b_moving_jaw"]["pass"],
            row["order_constraint_ab_pass"],
            row["pinch_core_pass"],
            row["full_pass"],
            row["moving_witness_top_margin_mm"],
            row["rim_proximal_lt_7p5mm"],
            row["arc_sweep"]["transient_unexcluded_pairs"],
            row["order_certification"]["certified"],
            len(row["order_certification"]["unresolved"])
            + len(row["order_certification"]["sub_resolution_rejected"]),
            row["order_certification"]["new_evals"],
            adm_region,
            adm_rep.get(adm_region) if adm_region else None,
            adm_depth_of.get(tag),
            full_region_of.get(tag),
            full_depth_of.get(tag),
            row["queries_used"],
        )
        lines.append(",".join(_csv_cell(value) for value in values))
    return ("\n".join(lines) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Prereg admission replay (A5; D405 lesson).
# ---------------------------------------------------------------------------

def _prereg_admission_replay(prereg_path: Path) -> dict[str, Any]:
    if not prereg_path.is_file():
        raise _ContractFail(f"prereg not found: {prereg_path}")
    prereg_bytes = prereg_path.read_bytes()
    prereg_sha = _sha_bytes(prereg_bytes)
    prereg = json.loads(prereg_bytes)
    rows = prereg.get("worker_admission_rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise _ContractFail(
            "prereg worker_admission_rows must be exactly 2 rows "
            f"(got {type(rows).__name__ if rows is not None else 'missing'})"
        )
    replayed = []
    for row in rows:
        path = Path(row["path"])
        registered = str(row["sha256"])
        recomputed = _sha_file(path) if path.is_file() else None
        status = "PASS" if recomputed == registered else "FAIL"
        replayed.append(
            {
                "row_id": str(row.get("row_id")),
                "path": str(path),
                "registered_sha256": registered,
                "recomputed_sha256": recomputed,
                "status": status,
            }
        )
    if any(row["status"] != "PASS" for row in replayed):
        raise _ContractFail(f"prereg admission replay FAIL: {replayed}")
    hash_crosscheck = []
    input_hashes = prereg.get("input_hashes")
    if isinstance(input_hashes, dict):
        for key, value in sorted(input_hashes.items()):
            for tag, pin in PINNED_INPUT_SHA256.items():
                if tag in key:
                    match = value == pin
                    hash_crosscheck.append({"prereg_key": key, "tag": tag, "match": match})
                    if not match:
                        raise _ContractFail(f"prereg input_hashes[{key}] != worker pin for {tag}")
    return {
        "prereg_path": str(prereg_path),
        "prereg_sha256": prereg_sha,
        "rows": replayed,
        "input_hash_crosscheck": hash_crosscheck,
        "pass": True,
    }


# ---------------------------------------------------------------------------
# Static pins + environment checks.
# ---------------------------------------------------------------------------

def _verify_static_pins() -> dict[str, Any]:
    checks = {
        "table_z_operation_sequence_pin": repr(TABLE_Z_M) == TABLE_Z_PIN_REPR,
        "z_center_operation_sequence_pin": repr(Z_CENTER_M) == Z_CENTER_PIN_REPR,
        "cyl_x_is_float32_of_0p3": CYL_X_M == float(np.float32(0.3)),
        "grid_radial_count_59": len(RADIALS_UM) == 59,
        "grid_tangent_count_21": len(TANGENTS_UM) == 21,
        "grid_pose_count_1239": len(RADIALS_UM) * len(TANGENTS_UM) == EXPECTED_POSE_COUNT,
        "positive_control_key_in_grid": (
            POSITIVE_CONTROL_KEY_UM[0] in RADIALS_UM and POSITIVE_CONTROL_KEY_UM[1] in TANGENTS_UM
        ),
        "tau_derivation": (
            "d335 formula with radius substitution (inherited): "
            "[R-8mm, R-8mm+5mm], R=14.5mm, 8mm=FIXED_JAW_FACE_LOCAL_M (d323:38)"
        ),
    }
    if not all(value is True for key, value in checks.items() if isinstance(value, bool)):
        raise _ContractFail(f"static pin verification failed: {checks}")
    return checks


def _verify_environment(hppfcl: Any) -> dict[str, Any]:
    preloaded = sorted(
        {name.split(".")[0].lower() for name in sys.modules} & FORBIDDEN_IMPORT_ROOTS
    )
    if preloaded:
        raise _ContractFail(f"forbidden modules already loaded: {preloaded}")
    env = {
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "hppfcl_version": getattr(hppfcl, "__version__", None),
        "worker_source_sha256": _sha_file(Path(__file__).resolve()),
        "scope_guard_installed": True,
        "scope_guard_violations": list(_SCOPE_GUARD_VIOLATIONS),
        "not_imported_by_worker": ["scipy", "trimesh", "psutil", "rerun"],
    }
    if env["numpy_version"] != EXPECTED_NUMPY_VERSION:
        raise _ContractFail(f"numpy pin violated: {env['numpy_version']} != {EXPECTED_NUMPY_VERSION}")
    if env["hppfcl_version"] != EXPECTED_HPPFCL_VERSION:
        raise _ContractFail(f"hppfcl pin violated: {env['hppfcl_version']} != {EXPECTED_HPPFCL_VERSION}")
    # FRZ-W2 repair R9: enforce interpreter identity + python version in the
    # worker itself (defense in depth; the controller gate also enforces).
    if os.path.realpath(sys.executable) != os.path.realpath(EXPECTED_INTERPRETER):
        raise _ContractFail(
            f"interpreter pin violated: {sys.executable} != {EXPECTED_INTERPRETER}"
        )
    if env["python_version"] != EXPECTED_PYTHON_VERSION:
        raise _ContractFail(
            f"python version pin violated: {env['python_version']} != {EXPECTED_PYTHON_VERSION}"
        )
    if _SCOPE_GUARD_VIOLATIONS:
        raise _ContractFail(f"scope guard violations recorded: {_SCOPE_GUARD_VIOLATIONS}")
    return env


# ---------------------------------------------------------------------------
# Main flow.
# ---------------------------------------------------------------------------

def _run(out_dir: Path, prereg_path: Path) -> int:
    started = time.time()
    resolved_path = out_dir.resolve()
    resolved = str(resolved_path)
    # Repo-anchored containment (FRZ-W1 repair R8): substring matching is
    # bypassable by any foreign path embedding the fragment; anchor to the
    # actual frozen output root instead.
    allowed_out_root = (REPO / "claudedocs/runtime_logs/grasp_track/g0a_d409").resolve()
    if not resolved_path.is_relative_to(allowed_out_root):
        print(f"D409 worker: out-dir outside g0a_d409 runtime tree: {resolved}")
        return EXIT_CONTRACT_FAIL
    out_dir.mkdir(parents=True, exist_ok=True)

    claim_path = out_dir / CLAIM_NAME
    try:
        claim_handle = open(claim_path, "x", encoding="utf-8")
    except FileExistsError:
        print(f"D409 worker: claim already exists, refusing: {claim_path}")
        return EXIT_CLAIM_PREEXIST
    claim = {
        "artifact": "d409_worker_claim",
        "case": "g0a_d409",
        "run_dir": resolved,
        "prereg_path": str(prereg_path),
        "argv": list(sys.argv),
        "interpreter": sys.executable,
        "pid": os.getpid(),
        "wall_time_s": started,
        "exit_contract": {"pass": EXIT_PASS, "claim_preexist": EXIT_CLAIM_PREEXIST, "fail": "other nonzero"},
    }
    claim_handle.write(json.dumps(claim, indent=2, sort_keys=True) + "\n")
    claim_handle.flush()
    os.fsync(claim_handle.fileno())
    claim_handle.close()

    for name in (EVIDENCE_NAME, REGION_CSV_NAME, SUMMARY_NAME, PRECLOSE_NAME, PHASE_NAME):
        if (out_dir / name).exists():
            print(f"D409 worker: pre-existing worker-owned output, fail-closed: {out_dir / name}")
            return EXIT_CONTRACT_FAIL

    phases = _PhaseLog(out_dir / PHASE_NAME)
    phases.mark("P0_claim", "PASS", str(claim_path))
    try:
        phases.mark("P1_prereg_admission_replay", "BEGIN")
        prereg_replay = _prereg_admission_replay(prereg_path)
        phases.mark("P1_prereg_admission_replay", "PASS", prereg_replay["prereg_sha256"])

        phases.mark("P2_frozen_input_load", "BEGIN")
        import hppfcl  # allowed offline query library (after scope guard)

        static_pins = _verify_static_pins()
        environment = _verify_environment(hppfcl)
        frozen = _load_frozen_inputs(hppfcl)
        phases.mark("P2_frozen_input_load", "PASS")

        budget = {"pose": 0, "total": 0}
        phases.mark("P3_anchor_gate", "BEGIN")
        anchor_gate = _anchor_gate(hppfcl, frozen, budget)
        phases.mark("P3_anchor_gate", "PASS")

        phases.mark("P4_enumeration", "BEGIN")
        cylinder = hppfcl.Cylinder(CYL_RADIUS_M, CYL_HEIGHT_M)
        cylinder_tf = hppfcl.Transform3f(
            np.eye(3, dtype=np.float64), np.asarray(CYL_CENTER_M, dtype=np.float64)
        )
        anchors_f32 = np.linspace(
            np.float32(Q5_OPEN_RAD), np.float32(0.0), ARC_ANCHOR_COUNT, dtype=np.float32
        )
        rmax_m = max(
            float(np.max(np.linalg.norm(part["vertices"][:, :2], axis=1)))
            for part in frozen["parts_by_body"]["gripper_link"]
        )
        rows: dict[tuple[int, int], dict[str, Any]] = {}
        completed = 0
        for rho_um in RADIALS_UM:
            for tau_um in TANGENTS_UM:
                rows[(rho_um, tau_um)] = _evaluate_pose(
                    hppfcl, frozen, cylinder, cylinder_tf, anchors_f32, rmax_m, rho_um, tau_um, budget
                )
                completed += 1
                if budget["total"] > MAX_QUERIES_PER_RUN:
                    raise _ContractFail(
                        f"registered per-run query budget exceeded: {budget['total']} > {MAX_QUERIES_PER_RUN}"
                    )
                if completed % 100 == 0 or completed == EXPECTED_POSE_COUNT:
                    print(f"D409 pose {completed}/{EXPECTED_POSE_COUNT}", flush=True)
        if len(rows) != EXPECTED_POSE_COUNT:
            raise _ContractFail(f"pose row count {len(rows)} != {EXPECTED_POSE_COUNT} (silent cap 0 violated)")
        phases.mark("P4_enumeration", "PASS", f"queries={budget['total']}")

        phases.mark("P5_canonical_evidence", "BEGIN")
        # Region layers consume a flat pass flag; add it transiently so the
        # serialized science rows keep the nested admission structure only.
        for row in rows.values():
            row["admission_pass_flat"] = row["admission"]["pass"]
        admission_layer = _region_analysis(rows, "admission_pass_flat", "admission")
        full_layer = _region_analysis(rows, "full_pass", "full_pass")
        for row in rows.values():
            del row["admission_pass_flat"]

        counts = {
            "poses": len(rows),
            "ik_converged": sum(1 for row in rows.values() if row["ik"]["converged"]),
            "admission_pass": sum(1 for row in rows.values() if row["admission"]["pass"]),
            "a_band_pass": sum(1 for row in rows.values() if row["a_fixed_jaw"]["band_pass"]),
            "b_pass": sum(1 for row in rows.values() if row["b_moving_jaw"]["pass"]),
            "order_ab_pass": sum(1 for row in rows.values() if row["order_constraint_ab_pass"]),
            "pinch_core_pass": sum(1 for row in rows.values() if row["pinch_core_pass"]),
            "full_pass": sum(1 for row in rows.values() if row["full_pass"]),
            "order_certified": sum(
                1 for row in rows.values() if row["order_certification"]["certified"]
            ),
            "admission_regions": admission_layer["region_count"],
            "admission_regions_censored": sum(
                1 for entry in admission_layer["regions"] if entry["domain_censored"]
            ),
            "full_pass_regions": full_layer["region_count"],
        }
        evidence = {
            "artifact": "d409_enumeration_evidence",
            "case": "g0a_d409",
            "verdict": VERDICT_COMPLETE,
            "semantics": (
                "zero-step offline dual-jaw contact-region exhaustive enumeration for the real "
                "cylinder D29xH50 (session doc section 2 as amended by section 4); judgment upper "
                "bound = contact-region map + ordering-constraint scoring; no physics step, no Isaac."
            ),
            "interpretation_boundary": [
                "stable grasp / force closure / grasp feasibility / grasp success / push-over-absence "
                "guarantee / contact-order dynamics / SDF superiority / transfer to other cylinders or "
                "placements: all null claims",
                "A-and-B does NOT exclude the D362 push-over pose (d_fix 4.2727mm, inner-4-mask min "
                "measured); the ordering constraint is a pure geometric descriptor and its push-over "
                "screening power is unverified (null)",
                "part-level masks cannot distinguish inner/outer faces (outer 16 subset of inner 17, "
                "difference part_035 alone); inner-17 membership is a necessary-condition judgment only",
                "geometry-only labels must not be promoted to standalone training (direction decision 2)",
                "mass 24.83g and friction are unused by this harness",
                "g0a_pass=false unchanged; enumeration completion is not a grasp verdict",
            ],
            "config": {
                "cylinder": {
                    "model": "hppfcl.Cylinder",
                    "radius_m": CYL_RADIUS_M,
                    "height_m": CYL_HEIGHT_M,
                    "pos_m": list(CYL_CENTER_M),
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "table_z_m": TABLE_Z_M,
                    "precision_note": (
                        "x = float32(0.3) as float64 literal; z_center = float64 operation sequence "
                        "(no cast; hppfcl Transform float64) - W-SCI3"
                    ),
                },
                "grid": {
                    "radial_um": {"min": RADIAL_MIN_UM, "max": RADIAL_MAX_UM, "step": GRID_STEP_UM},
                    "tangent_um": {"min": TANGENT_MIN_UM, "max": TANGENT_MAX_UM, "step": GRID_STEP_UM},
                    "tangent_sign": ADOPTED_TANGENT_SIGN,
                    "pose_count": EXPECTED_POSE_COUNT,
                    "tau_derivation_label": static_pins["tau_derivation"],
                    "positive_control_key_um": list(POSITIVE_CONTROL_KEY_UM),
                    "positive_control_note": (
                        "(7000,11000) retained in the grid; meaning differs from the frozen candidate "
                        "(new center/real cylinder)"
                    ),
                },
                "q5_arc": {
                    "open_rad": Q5_OPEN_RAD,
                    "anchor_count": ARC_ANCHOR_COUNT,
                    "anchor_dtype": "float32 linspace OPEN->0",
                    "bisect_bracket_rad": BISECT_BRACKET_RAD,
                    "bisect_max_iter": BISECT_MAX_ITER,
                    "bisect_note": "numerical-resolution control, not a science tolerance (D351 lineage)",
                },
                "gates": {
                    "clear_gate_mm": CLEAR_GATE_MM,
                    "fixed_jaw_band_mm": list(FIXED_JAW_BAND_MM),
                    "fixed_jaw_band_reuse_disclosure": (
                        "5.0mm upper bound is a REUSE of the old D330 planar-proxy gate constant; "
                        "D330 metric = tangent-projected planar gap 0-5mm vs new metric = hppfcl 3D "
                        "min over the link5 4-mask (~1.7mm apart at (7,11)-class poses); planar-gap "
                        "equivalent recorded as a diagnostic column for every pose (P3/W-SCI1)"
                    ),
                    "tcp_gate_mm": TCP_GATE_MM,
                    "jaw_tangent_gate_deg": JAW_TANGENT_GATE_DEG,
                    "top_15mm_rule": "demoted to non-gate diagnostic (top margins recorded per pose)",
                    "rim_proximity_band_mm": RIM_PROXIMITY_BAND_MM,
                },
                "kinematics": {
                    "fk_constant_series": "URDF XML literals only (pi/2-symbol chain banned)",
                    "ik": {
                        "form": "d323 position-only 5-DOF DLS",
                        "seed_deg": list(HOME_DEG),
                        "max_iter": IK_MAX_ITER,
                        "pos_tol_mm": IK_POS_TOL_MM,
                        "step_clip_deg": IK_STEP_CLIP_DEG,
                        "damping": IK_DAMPING,
                        "jacobian_eps_deg": IK_JACOBIAN_EPS_DEG,
                        "soft_limits_deg": [[name, lo, hi] for name, lo, hi in V6_JOINT_LIMITS_DEG],
                        "urdf_hard_limits": "diagnostic only",
                        "randomness": "none (deterministic DLS)",
                    },
                    "fixed_jaw_face_local_m": list(FIXED_JAW_FACE_LOCAL_M),
                    "fk_accuracy_statement": (
                        "FK reproduction is stated as measured residuals (anchor gate channels), "
                        "never as bit-exact"
                    ),
                },
                "hppfcl": {
                    "bvh": "BVHModelOBBRSS(callback topology_triangles)",
                    "distance_request": ["DistanceRequest(True,1e-9,1e-9)", "gjk 1e-9/1000"],
                    "collision_request": "enable_contact, num_max_contacts 256 (on overlap only, A7)",
                    "signed_distance": "overlap -> negative max EPA depth (d349 convention)",
                },
                "pinch": {
                    "core_names": list(PINCH_CORE_NAMES),
                    "diagnostic_names": list(PINCH_DIAGNOSTIC_NAMES),
                    "core_selection_rationale": (
                        "opposition 1 + geometric placement 2 + closing direction 1 (W-SCI4); "
                        "formula structure reuse from d351 - NOT a reuse of its pass=false result"
                    ),
                },
                "budget": {
                    "max_queries_per_pose": MAX_QUERIES_PER_POSE,
                    "max_queries_per_run": MAX_QUERIES_PER_RUN,
                    "unit": "one part-vs-cylinder evaluation (A7)",
                },
                "determinism_byte_compare_members": list(DETERMINISM_BYTE_COMPARE_MEMBERS),
            },
            "static_pins": static_pins,
            "environment": environment,
            "input_sha256": frozen["input_sha256"],
            "prereg_admission_replay": prereg_replay,
            "masks": {
                "link5_fixed": frozen["masks"]["link5_fixed"],
                "gripper_inner": sorted(frozen["masks"]["gripper_inner"]),
                "gripper_outer": sorted(frozen["masks"]["gripper_outer"]),
                "checks": frozen["mask_checks"],
                "face_limitation": (
                    "part-level mask cannot distinguish inner/outer faces; outer 16 subset of "
                    "inner 17, difference = part_035 (W-FRZ1)"
                ),
            },
            "d348_integrity": {
                "canonical_per_part_sha256": frozen["canonical_hashes"],
                "stored_payload_sha256": frozen["stored_payload_sha256"],
                "counts": {"link5": 64, "gripper_link": 64},
                "hash_definition": "D409-canonical (name, repr vertices_m, topology_triangles) - S1 tool identical",
            },
            "anchor_gate": anchor_gate,
            "arc_geometry": {
                "gripper_rmax_about_joint_axis_m": rmax_m,
                "chord_bound_formula": "2*Rmax*sin(|dq|/2)",
            },
            "poses": [rows[key] for key in sorted(rows)],
            "region_analysis": {
                "admission_layer_authority_note": (
                    "section 2.8 'admitted pose' regions = admission_pass cells (A3); full_pass "
                    "layer is diagnostic"
                ),
                "admission": admission_layer,
                "full_pass": full_layer,
            },
            "counts": counts,
            "query_budget_observed": {
                "total_queries": budget["total"],
                "within_registered_budget": budget["total"] <= MAX_QUERIES_PER_RUN,
            },
            "randomness_sources": [],
        }
        evidence_bytes = _canonical_json_bytes(evidence)
        _write_new_bytes(out_dir / EVIDENCE_NAME, evidence_bytes)
        csv_bytes = _region_csv_bytes(rows, admission_layer, full_layer)
        _write_new_bytes(out_dir / REGION_CSV_NAME, csv_bytes)
        evidence_sha = _sha_bytes(evidence_bytes)
        csv_sha = _sha_bytes(csv_bytes)
        # W-LES3: canonical evidence + verdict sha256 published BEFORE any
        # presentation artifact (this worker never writes RRD/RBL/screenshot).
        print(f"D409 evidence sha256 {evidence_sha}")
        print(f"D409 region csv sha256 {csv_sha}")
        print(f"D409 verdict {VERDICT_COMPLETE}")
        phases.mark("P5_canonical_evidence", "PASS", evidence_sha)

        phases.mark("P6_summary", "BEGIN")
        summary = {
            "artifact": "d409_worker_summary",
            "case": "g0a_d409",
            "verdict": VERDICT_COMPLETE,
            "evidence_sha256": evidence_sha,
            "region_csv_sha256": csv_sha,
            "counts": counts,
            "total_queries": budget["total"],
            "elapsed_s": time.time() - started,
            "determinism_byte_compare_members": list(DETERMINISM_BYTE_COMPARE_MEMBERS),
            "phase_order_contract": "measurement-before-presentation (W-LES3); worker writes no RRD",
        }
        summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _write_new_bytes(out_dir / SUMMARY_NAME, summary_bytes)
        summary_sha = _sha_bytes(summary_bytes)
        phases.mark("P6_summary", "PASS", summary_sha)

        phases.mark("P7_preclose_sentinel", "BEGIN")
        sentinel = {
            "artifact": "d409_worker_preclose_sentinel",
            "case": "g0a_d409",
            "status": "PRECLOSE_PASS",
            "verdict": VERDICT_COMPLETE,
            "summary_sha256": summary_sha,
            "evidence_sha256": evidence_sha,
            "region_csv_sha256": csv_sha,
        }
        _write_new_bytes(
            out_dir / PRECLOSE_NAME, (json.dumps(sentinel, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        phases.mark("P7_preclose_sentinel", "PASS")
        phases.close()
        return EXIT_PASS
    except _ContractFail as exc:
        print(f"D409 worker CONTRACT FAIL: {exc}")
        print(f"D409 verdict {VERDICT_CONTRACT_FAIL}")
        phases.mark("CONTRACT_FAIL", "FAIL", str(exc))
        phases.close()
        return EXIT_CONTRACT_FAIL
    except Exception:
        traceback.print_exc()
        print(f"D409 verdict {VERDICT_CONTRACT_FAIL}")
        phases.mark("UNHANDLED_EXCEPTION", "FAIL", traceback.format_exc(limit=3))
        phases.close()
        return EXIT_CONTRACT_FAIL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, help="absolute run dir (run1/ or run2/)")
    parser.add_argument("--prereg", required=True, help="absolute prereg path")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    prereg_path = Path(args.prereg)
    if not out_dir.is_absolute() or not prereg_path.is_absolute():
        print("D409 worker: --out-dir and --prereg must be absolute paths")
        return EXIT_CONTRACT_FAIL
    return _run(out_dir, prereg_path)


if __name__ == "__main__":
    sys.exit(main())
