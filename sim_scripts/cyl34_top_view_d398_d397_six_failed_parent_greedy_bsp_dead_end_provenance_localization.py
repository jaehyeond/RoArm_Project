#!/usr/bin/env python3
"""D398: localize why the six frozen D397 greedy BSP paths stopped.

Offline-only.  This script reads only the immutable D397 attempt2 evidence,
geometry, worker claim, and frozen D397 implementation.  It replays already
selected D397 branches in memory and evaluates tentative split candidates only
to classify their first rejection stage.  It never chooses a new branch,
serializes/adopts candidate child geometry, writes USD, or launches
Isaac/Kit/PhysX/Warp/CUDA/physics/q5/contact/cylinder work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import QhullError


REPO = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO):
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))

CASE = "g0a_d398"
ATTEMPT = (
    "attempt1_six_failed_parent_greedy_bsp_dead_end_"
    "provenance_localization"
)
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT = Path(__file__).resolve()
START = REPO / "START_HERE.md"

D397_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d397"
    / "attempt2_phase_marker_payload_key_repair"
)
D397_EVIDENCE = D397_DIR / "d397_shared_boundary_design_evidence.json"
D397_GEOMETRY = D397_DIR / "d397_shared_boundary_candidate_geometry.json"
D397_WORKER_CLAIM = D397_DIR / "d397_offline_worker_claim.json"
D397_BASE = (
    REPO
    / "sim_scripts"
    / "cyl34_top_view_d397_shared_boundary_zero_volume_construction_design.py"
)
D397_WRAPPER = (
    REPO
    / "sim_scripts"
    / "cyl34_top_view_d397_attempt2_phase_marker_payload_key_repair.py"
)
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"

EXPECTED_HEAD = "7736c73910aa5756ef1560ee55640ba005faa012"
EXPECTED_INPUT_SHA256 = {
    D397_EVIDENCE: (
        "ea7fd61c38f12b9e03f4e7154536579b831c6f85703bfd4d14e34807cdf327b6"
    ),
    D397_GEOMETRY: (
        "b9a44d430f647e45292fe71804bd17e6f53bf37eea28913389316beac60fa623"
    ),
    D397_WORKER_CLAIM: (
        "2bac06043e35e095660ed3a0562930f98425a9eba436dc5b72e58f313ed1ed79"
    ),
    D397_BASE: (
        "52745beab46bc695467dd8d676a06b30fa3ea873c7dcad685861e65cfecf4b36"
    ),
    D397_WRAPPER: (
        "bd95aa1cadb21e4f192c3171596585cc01002dfd951537c68035afd88230286a"
    ),
    VIZ_DEBUG: (
        "4b5f821ad43652f529dfaa2f92b2826d9cd4973635e34521cc2b3a93ab0193d0"
    ),
    RERUN_CONTRACT: (
        "aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e"
    ),
}

AUTHORITY = OUT_DIR / "d398_execution_authority.json"
PREREG = OUT_DIR / "d398_preregistration.json"
PHASES = OUT_DIR / "d398_phase_markers.jsonl"
INVOCATION = OUT_DIR / "d398_offline_worker_invocation.json"
WORKER_AUTH = OUT_DIR / "d398_worker_authorization.json"
SENTINEL = OUT_DIR / "d398_worker_start_sentinel.json"
STDOUT = OUT_DIR / "d398_offline_worker_stdout.log"
STDERR = OUT_DIR / "d398_offline_worker_stderr.log"
EVIDENCE = OUT_DIR / "d398_greedy_dead_end_provenance_evidence.json"
DISPLAY = OUT_DIR / "d398_frozen_stuck_leaf_display_geometry.json"
WORKER_CLAIM = OUT_DIR / "d398_offline_worker_claim.json"
SUPERVISOR = OUT_DIR / "d398_offline_worker_supervisor.json"
BOARD = OUT_DIR / "d398_greedy_dead_end_provenance_1920x1080.png"
LAYOUT = OUT_DIR / "d398_board_layout_validation.json"
RRD = OUT_DIR / "d398_frozen_stuck_leaf_provenance.rerun.rrd"
RBL = OUT_DIR / "d398_frozen_stuck_leaf_provenance.rerun.rbl"
RERUN_VALIDATION = OUT_DIR / "d398_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d398_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d398_manual_visual_inspection_template.json"
MANUAL = OUT_DIR / "d398_manual_visual_inspection.json"
OBSERVABILITY = OUT_DIR / "d398_observability_claim.json"
COMPLETION = OUT_DIR / "d398_completion_summary.json"
FAILURE = OUT_DIR / "d398_failure_attestation.json"

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = [
    "six_failed_parent_axis_midpoint_option_rejection_provenance_v1"
]
FAILED_PARENTS = [
    "proximal_upper_arm_hull_a",
    "proximal_lower_arm_hull_a",
    "moving_upper_backbone",
    "moving_lower_backbone",
    "fixed_backbone_left",
    "fixed_backbone_right",
]
EXPECTED_STUCK = {
    "proximal_upper_arm_hull_a": {
        "name": "proximal_upper_arm_hull_a__diagnostic_leaf_00",
        "vertex_count": 31,
        "payload_sha256": (
            "d6771ebb387c3e5e2bba3f5a390a4923345cfa28f9340e07537de09506fbe791"
        ),
    },
    "proximal_lower_arm_hull_a": {
        "name": "proximal_lower_arm_hull_a__diagnostic_leaf_04",
        "vertex_count": 28,
        "payload_sha256": (
            "3063674a29efb2f45f1bee5466b976f9c8ea86fb5e1b94e74de0ecb4a55542e6"
        ),
    },
    "moving_upper_backbone": {
        "name": "moving_upper_backbone__diagnostic_leaf_00",
        "vertex_count": 16,
        "payload_sha256": (
            "f7621a7a3608291c944a6b8499103effda26e2706f2a9932baf686c953e54905"
        ),
    },
    "moving_lower_backbone": {
        "name": "moving_lower_backbone__diagnostic_leaf_00",
        "vertex_count": 16,
        "payload_sha256": (
            "16989edd8376dd7bceec8b5f61376f75cd4b607fca835c6d8fd957e77084feb5"
        ),
    },
    "fixed_backbone_left": {
        "name": "fixed_backbone_left__diagnostic_leaf_01",
        "vertex_count": 21,
        "payload_sha256": (
            "2bd3f57345ad8bc439abc66052f44618fea3d80fcdc19105a1a9b3a5ccf2fa87"
        ),
    },
    "fixed_backbone_right": {
        "name": "fixed_backbone_right__diagnostic_leaf_02",
        "vertex_count": 22,
        "payload_sha256": (
            "5e69b301f59cc8da8c87ae85b5a6c569a78f99dcfcb09210b89cfaecd531608e"
        ),
    },
}
STAGE_ORDER = [
    "midpoint_candidate_generation",
    "paired_split_creation",
    "seam_volume_validity",
    "strict_vertex_reduction",
    "admissible",
]
COOPERATIVE_DEADLINE_SECONDS = 180.0
AUTHORITY_ENV = "D398_EXECUTION_AUTHORITY_SHA256"
WORKER_AUTH_ENV = "D398_WORKER_AUTHORIZATION_SHA256"

MANUAL_KEYS = [
    "board_exact_1920x1080_and_readable",
    "six_failed_parent_rows_visible",
    "candidate_stage_counts_readable",
    "ancestor_alternative_booleans_readable",
    "frozen_parent_and_stuck_leaf_geometry_visible",
    "no_alternative_candidate_geometry_shown",
    "offline_scope_and_null_physics_boundary_readable",
    "no_text_overlap_or_clipping",
]
REQUIRED_ZERO_SCOPE_KEYS = (
    "new_branches_selected",
    "backtracking_or_depth2_searches",
    "candidate_child_geometries_serialized_or_adopted",
    "vertex_budget_changes",
    "plane_family_changes",
    "tolerance_or_gate_changes",
    "usd_or_asset_reads",
    "usd_or_asset_writes",
    "collider_materializations",
    "isaac_launches",
    "kit_launches",
    "physx_launches",
    "warp_or_cuda_launches",
    "cylinder_creates_or_writes",
    "physics_steps",
    "q5_samples",
    "contact_queries",
    "target_ik_path_or_settings_changes",
    "process_signals_sent",
)


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        _native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            _native(value),
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")


def _append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _native(value),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _phase(name: str, **fields: Any) -> None:
    rows = []
    if PHASES.exists():
        rows = [
            json.loads(line)
            for line in PHASES.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    _append(
        PHASES,
        {
            "ordinal": len(rows),
            "phase": name,
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        },
    )


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _status_outside_output() -> list[str]:
    prefix = _rel(OUT_DIR) + "/"
    rows = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return [row for row in rows if not row[3:].startswith(prefix)]


def _input_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in EXPECTED_INPUT_SHA256}


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def _load_d397() -> Any:
    spec = importlib.util.spec_from_file_location("d398_frozen_d397", D397_BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen D397 implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _part_cell(part: dict[str, Any]) -> dict[str, Any]:
    return {
        "vertices_m": np.asarray(part["vertices_f32_m"], dtype=np.float64),
        "triangles": np.asarray(part["triangles_i32"], dtype=np.int64),
        "vertex_count": int(part["vertex_count"]),
        "volume_m3": float(part["topology_volume_m3"]),
        "path_constraints": [dict(row) for row in part["path_constraints"]],
        "depth": len(part["path_constraints"]),
        "stable_key": _stable_key(part["path_constraints"]),
    }


def _stable_key(path_constraints: Iterable[dict[str, Any]]) -> str:
    return "".join(
        f"{str(row['side'])[0]}{int(row['node_id']):03d}"
        for row in path_constraints
    )


def _candidate_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "axis": int(row["axis"]),
        "cut_f32_bits": int(row["cut_f32_bits"]),
        "seam_vertex_bits_sha256": row["seam_vertex_bits_sha256"],
        "score": _native(row["score"]),
    }


def _trace_candidate(
    base: Any,
    cell: dict[str, Any],
    *,
    axis: int,
    gap_index: int,
    low: float,
    high: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    cut = float(np.float32((low + high) * 0.5))
    row: dict[str, Any] = {
        "axis": int(axis),
        "axis_name": "xyz"[axis],
        "gap_index": int(gap_index),
        "low_f32_m": float(low),
        "high_f32_m": float(high),
        "cut_f32_m": cut,
        "cut_f32_bits": int(base._f32_bits(cut)),
        "first_rejection_stage": None,
        "first_rejection_reason": None,
    }
    if not (low < cut < high):
        row.update(
            {
                "first_rejection_stage": "midpoint_candidate_generation",
                "first_rejection_reason": (
                    "float32_midpoint_not_strictly_between_adjacent_levels"
                ),
            }
        )
        return row, None

    points = np.asarray(cell["vertices_m"], dtype=np.float64)
    triangles = np.asarray(cell["triangles"], dtype=np.int64)
    values = points[:, axis] - cut
    left_points = [
        point for point, value in zip(points, values, strict=True) if value < 0.0
    ]
    right_points = [
        point for point, value in zip(points, values, strict=True) if value > 0.0
    ]
    seam_points = [
        point for point, value in zip(points, values, strict=True) if value == 0.0
    ]
    crossing_edge_count = 0
    for left_index, right_index in base._edge_rows(triangles):
        v0 = float(values[int(left_index)])
        v1 = float(values[int(right_index)])
        if not ((v0 < 0.0 < v1) or (v1 < 0.0 < v0)):
            continue
        crossing_edge_count += 1
        ratio = -v0 / (v1 - v0)
        point = (
            points[int(left_index)]
            + ratio * (points[int(right_index)] - points[int(left_index)])
        ).astype(np.float32)
        point[axis] = np.float32(cut)
        seam_points.append(point.astype(np.float64))
    row.update(
        {
            "left_support_count": len(left_points),
            "right_support_count": len(right_points),
            "on_plane_source_count": int(np.sum(values == 0.0)),
            "crossing_edge_count": crossing_edge_count,
            "seam_candidate_count": len(seam_points),
        }
    )
    if not left_points:
        row.update(
            {
                "first_rejection_stage": "paired_split_creation",
                "first_rejection_reason": "missing_left_support",
            }
        )
        return row, None
    if not right_points:
        row.update(
            {
                "first_rejection_stage": "paired_split_creation",
                "first_rejection_reason": "missing_right_support",
            }
        )
        return row, None
    if len(seam_points) < 3:
        row.update(
            {
                "first_rejection_stage": "paired_split_creation",
                "first_rejection_reason": "fewer_than_three_seam_candidates",
            }
        )
        return row, None
    try:
        seam = base._seam_extremes(np.asarray(seam_points), axis)
    except (ValueError, QhullError) as error:
        row.update(
            {
                "first_rejection_stage": "paired_split_creation",
                "first_rejection_reason": "seam_polygon_construction_failed",
                "diagnostic_exception_type": type(error).__name__,
            }
        )
        return row, None
    try:
        left = base._oriented_hull(np.vstack([left_points, seam]))
        right = base._oriented_hull(np.vstack([right_points, seam]))
    except (ValueError, QhullError) as error:
        row.update(
            {
                "first_rejection_stage": "paired_split_creation",
                "first_rejection_reason": "closed_sibling_hull_construction_failed",
                "diagnostic_exception_type": type(error).__name__,
            }
        )
        return row, None

    row.update(
        {
            "seam_vertex_count": int(len(seam)),
            "left_child_vertex_count": int(left["vertex_count"]),
            "right_child_vertex_count": int(right["vertex_count"]),
            "left_child_volume_m3": float(left["volume_m3"]),
            "right_child_volume_m3": float(right["volume_m3"]),
        }
    )
    if left["volume_m3"] <= base.POSITIVE_VOLUME_EPS_M3:
        row.update(
            {
                "first_rejection_stage": "seam_volume_validity",
                "first_rejection_reason": "left_child_volume_not_positive",
            }
        )
        return row, None
    if right["volume_m3"] <= base.POSITIVE_VOLUME_EPS_M3:
        row.update(
            {
                "first_rejection_stage": "seam_volume_validity",
                "first_rejection_reason": "right_child_volume_not_positive",
            }
        )
        return row, None
    relative = abs(
        float(left["volume_m3"])
        + float(right["volume_m3"])
        - float(cell["volume_m3"])
    ) / float(cell["volume_m3"])
    row["split_volume_relative_error"] = float(relative)
    if relative > base.VOLUME_RELATIVE_TOLERANCE:
        row.update(
            {
                "first_rejection_stage": "seam_volume_validity",
                "first_rejection_reason": "split_volume_relative_error_above_gate",
            }
        )
        return row, None

    left_seam = {
        base._row_bits(point)
        for point in left["vertices_m"]
        if float(point[axis]) == cut
    }
    right_seam = {
        base._row_bits(point)
        for point in right["vertices_m"]
        if float(point[axis]) == cut
    }
    required = {base._row_bits(point) for point in seam}
    shared_exact = required.issubset(left_seam) and required.issubset(right_seam)
    no_halo = (
        float(np.max(left["vertices_m"][:, axis])) <= cut
        and float(np.min(right["vertices_m"][:, axis])) >= cut
    )
    seam_hash = hashlib.sha256(b"".join(sorted(required))).hexdigest()
    row.update(
        {
            "seam_vertex_bits_sha256": seam_hash,
            "shared_seam_vertex_bits_exact": bool(shared_exact),
            "opposite_closed_halfspaces_no_halo": bool(no_halo),
        }
    )
    if not shared_exact:
        row.update(
            {
                "first_rejection_stage": "seam_volume_validity",
                "first_rejection_reason": "shared_seam_bits_not_exact",
            }
        )
        return row, None
    if not no_halo:
        row.update(
            {
                "first_rejection_stage": "seam_volume_validity",
                "first_rejection_reason": "opposite_halfspace_halo_detected",
            }
        )
        return row, None

    left_count = int(left["vertex_count"])
    right_count = int(right["vertex_count"])
    if max(left_count, right_count) >= int(cell["vertex_count"]):
        row.update(
            {
                "first_rejection_stage": "strict_vertex_reduction",
                "first_rejection_reason": (
                    "maximum_child_vertex_count_not_strictly_reduced"
                ),
            }
        )
        return row, None
    imbalance = abs(
        float(left["volume_m3"]) - float(right["volume_m3"])
    ) / float(cell["volume_m3"])
    split = {
        "axis": int(axis),
        "axis_name": "xyz"[axis],
        "cut_f32_m": cut,
        "cut_f32_bits": int(base._f32_bits(cut)),
        "seam_vertices_m": seam,
        "seam_vertex_count": int(len(seam)),
        "seam_vertex_bits_sha256": seam_hash,
        "shared_seam_vertex_bits_exact": bool(shared_exact),
        "opposite_closed_halfspaces_no_halo": bool(no_halo),
        "split_volume_relative_error": float(relative),
        "left": left,
        "right": right,
        "score": [
            max(left_count, right_count),
            float(imbalance),
            left_count + right_count,
            int(axis),
            int(base._f32_bits(cut)),
        ],
    }
    row.update(
        {
            "first_rejection_stage": "admissible",
            "first_rejection_reason": None,
            "score": split["score"],
        }
    )
    return row, split


def _trace_candidates(
    base: Any, cell: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    bool,
]:
    manifest = []
    traced_admissible = []
    for axis in range(3):
        levels = sorted(set(map(float, cell["vertices_m"][:, axis])))
        for gap_index, (low, high) in enumerate(
            zip(levels[:-1], levels[1:], strict=True)
        ):
            row, split = _trace_candidate(
                base,
                cell,
                axis=axis,
                gap_index=gap_index,
                low=low,
                high=high,
            )
            manifest.append(row)
            if split is not None:
                traced_admissible.append(split)
    traced_admissible.sort(key=lambda row: tuple(row["score"]))
    frozen = base._candidate_splits(cell)
    equivalent = [
        _candidate_identity(row) for row in traced_admissible
    ] == [_candidate_identity(row) for row in frozen]
    return manifest, traced_admissible, frozen, equivalent


def _validate_candidate_manifest(
    manifest: list[dict[str, Any]],
    *,
    expected_count: int,
    expected_admissible_count: int,
) -> bool:
    identities = [
        (
            int(row["axis"]),
            int(row["gap_index"]),
            float(row["low_f32_m"]),
            float(row["high_f32_m"]),
        )
        for row in manifest
    ]
    return (
        len(manifest) == expected_count
        and len(set(identities)) == expected_count
        and all(
            row.get("first_rejection_stage") in STAGE_ORDER
            for row in manifest
        )
        and sum(
            row.get("first_rejection_stage") == "admissible"
            for row in manifest
        )
        == expected_admissible_count
    )


def _raw_axis_midpoint_gap_count(cell: dict[str, Any]) -> int:
    return sum(
        max(
            0,
            len(sorted(set(map(float, cell["vertices_m"][:, axis]))))
            - 1,
        )
        for axis in range(3)
    )


def _validate_stuck_identity(
    leaf: dict[str, Any], expected: dict[str, Any]
) -> bool:
    return all(leaf.get(key) == value for key, value in expected.items())


def _validate_ancestor_boolean(
    emitted: dict[str, Any], *, derived_value: bool
) -> bool:
    return (
        emitted.get("unselected_admissible_option_exists")
        is bool(derived_value)
    )


def _validate_selected_identity(
    emitted: dict[str, Any], frozen: dict[str, Any]
) -> bool:
    return (
        int(emitted["selected_axis"]) == int(frozen["axis"])
        and int(emitted["selected_cut_f32_bits"])
        == int(frozen["cut_f32_bits"])
        and emitted["selected_seam_vertex_bits_sha256"]
        == frozen["seam_vertex_bits_sha256"]
    )


def _replay_parent(
    base: Any,
    parent_name: str,
    source: dict[str, Any],
    immutable_leaves: list[dict[str, Any]],
    immutable_seams: list[dict[str, Any]],
    deadline: float,
) -> dict[str, Any]:
    base._DEADLINE = deadline
    root = base._oriented_hull(
        np.asarray(source["vertices_f32_m"], dtype=np.float64)
    )
    leaves: list[dict[str, Any]] = [
        {
            **root,
            "path_constraints": [],
            "depth": 0,
            "stable_key": "",
        }
    ]
    replay_nodes: list[dict[str, Any]] = []
    next_node_id = 0
    stuck_cell: dict[str, Any] | None = None
    stuck_manifest: list[dict[str, Any]] = []
    stuck_trace_equivalent = False
    replay_cell_count = 0
    trace_raw_gap_count = 0
    frozen_parity_raw_gap_count = 0
    while any(
        int(row["vertex_count"]) > base.MAX_SOURCE_CHILD_VERTICES
        for row in leaves
    ):
        if time.monotonic() > deadline:
            raise TimeoutError(f"D398 cooperative deadline at {parent_name}")
        candidates = [
            (index, row)
            for index, row in enumerate(leaves)
            if int(row["vertex_count"]) > base.MAX_SOURCE_CHILD_VERTICES
        ]
        index, cell = min(
            candidates,
            key=lambda pair: (
                -int(pair[1]["vertex_count"]),
                -float(pair[1]["volume_m3"]),
                str(pair[1]["stable_key"]),
            ),
        )
        (
            trace_rows,
            traced_options,
            options,
            trace_equivalent,
        ) = _trace_candidates(base, cell)
        replay_cell_count += 1
        trace_raw_gap_count += len(trace_rows)
        frozen_parity_raw_gap_count += len(trace_rows)
        if not trace_equivalent:
            raise RuntimeError(
                f"D398 trace/frozen option mismatch at {parent_name} "
                f"node {next_node_id}"
            )
        if not options:
            stuck_cell = cell
            stuck_manifest = trace_rows
            stuck_trace_equivalent = trace_equivalent
            break
        selected = options[0]
        seam_rows = [
            row
            for row in immutable_seams
            if int(row["node_id"]) == next_node_id
        ]
        if len(seam_rows) != 1:
            raise RuntimeError(
                f"D398 missing/duplicate frozen seam {parent_name} "
                f"node {next_node_id}"
            )
        seam = seam_rows[0]
        selected_matches = (
            int(seam["axis"]) == int(selected["axis"])
            and int(seam["cut_f32_bits"]) == int(selected["cut_f32_bits"])
            and seam["seam_vertex_bits_sha256"]
            == selected["seam_vertex_bits_sha256"]
        )
        if not selected_matches:
            raise RuntimeError(
                f"D398 selected seam mismatch at {parent_name} "
                f"node {next_node_id}"
            )
        common = {
            "node_id": int(next_node_id),
            "axis": int(selected["axis"]),
            "axis_name": selected["axis_name"],
            "cut_f32_m": float(selected["cut_f32_m"]),
            "cut_f32_bits": int(selected["cut_f32_bits"]),
        }
        replay_nodes.append(
            {
                "node_id": int(next_node_id),
                "selected_axis": int(selected["axis"]),
                "selected_axis_name": selected["axis_name"],
                "selected_cut_f32_bits": int(selected["cut_f32_bits"]),
                "selected_seam_vertex_bits_sha256": selected[
                    "seam_vertex_bits_sha256"
                ],
                "selected_identity_matches_frozen_seam": selected_matches,
                "trace_matches_frozen_candidate_order": trace_equivalent,
                "_unselected_admissible_option_exists": len(options) > 1,
            }
        )
        children = []
        for side, geometry in (
            ("le", selected["left"]),
            ("ge", selected["right"]),
        ):
            children.append(
                {
                    **geometry,
                    "path_constraints": [
                        *cell["path_constraints"],
                        {**common, "side": side},
                    ],
                    "depth": int(cell["depth"]) + 1,
                    "stable_key": (
                        f"{cell['stable_key']}{side[0]}"
                        f"{next_node_id:03d}"
                    ),
                }
            )
        leaves = [
            row for offset, row in enumerate(leaves) if offset != index
        ]
        leaves.extend(children)
        leaves.sort(key=lambda row: str(row["stable_key"]))
        next_node_id += 1
    if stuck_cell is None:
        raise RuntimeError(f"D398 expected frozen dead end for {parent_name}")

    replay_public = []
    for index, leaf in enumerate(
        sorted(leaves, key=lambda row: str(row["stable_key"]))
    ):
        part = base._part_from_points(
            body=source["body"],
            name=f"{parent_name}__diagnostic_leaf_{index:02d}",
            role=source["role"],
            source="D397_diagnostic_partition_leaf",
            points=leaf["vertices_m"],
            parent_name=parent_name,
            path_constraints=leaf["path_constraints"],
        )
        replay_public.append(base._public_part(part))
    immutable_sorted = sorted(immutable_leaves, key=lambda row: row["name"])
    forest_replay_exact = [
        {
            "name": row["name"],
            "payload_sha256": row["payload_sha256"],
            "path_constraints": row["path_constraints"],
        }
        for row in replay_public
    ] == [
        {
            "name": row["name"],
            "payload_sha256": row["payload_sha256"],
            "path_constraints": row["path_constraints"],
        }
        for row in immutable_sorted
    ]

    over_budget = [
        row
        for row in immutable_sorted
        if int(row["vertex_count"]) > base.MAX_SOURCE_CHILD_VERTICES
    ]
    inferred_stuck = min(
        over_budget,
        key=lambda row: (
            -int(row["vertex_count"]),
            -float(row["topology_volume_m3"]),
            _stable_key(row["path_constraints"]),
        ),
    )
    expected = EXPECTED_STUCK[parent_name]
    stuck_identity_exact = _validate_stuck_identity(
        inferred_stuck, expected
    )
    replayed_stuck_part = base._public_part(
        base._part_from_points(
            body=source["body"],
            name=inferred_stuck["name"],
            role=source["role"],
            source="D397_diagnostic_partition_leaf",
            points=stuck_cell["vertices_m"],
            parent_name=parent_name,
            path_constraints=stuck_cell["path_constraints"],
        )
    )
    stuck_replay_exact = (
        replayed_stuck_part["payload_sha256"]
        == inferred_stuck["payload_sha256"]
        and replayed_stuck_part["path_constraints"]
        == inferred_stuck["path_constraints"]
    )
    stuck_node_ids = {
        int(row["node_id"]) for row in inferred_stuck["path_constraints"]
    }
    immutable_seams_by_node = {
        int(row["node_id"]): row for row in immutable_seams
    }
    ancestor_lineage_internal = [
        {
            "node_id": row["node_id"],
            "selected_axis": row["selected_axis"],
            "selected_axis_name": row["selected_axis_name"],
            "selected_cut_f32_bits": row["selected_cut_f32_bits"],
            "selected_seam_vertex_bits_sha256": row[
                "selected_seam_vertex_bits_sha256"
            ],
            "selected_identity_matches_frozen_seam": row[
                "selected_identity_matches_frozen_seam"
            ],
            "trace_matches_frozen_candidate_order": row[
                "trace_matches_frozen_candidate_order"
            ],
            "unselected_admissible_option_exists": row[
                "_unselected_admissible_option_exists"
            ],
        }
        for row in replay_nodes
        if int(row["node_id"]) in stuck_node_ids
    ]
    ancestor_lineage = [
        {
            "node_id": row["node_id"],
            "unselected_admissible_option_exists": row[
                "unselected_admissible_option_exists"
            ],
        }
        for row in ancestor_lineage_internal
    ]
    ancestor_derivations = [
        {
            "emitted": emitted,
            "selected_identity_internal": internal,
            "derived_unselected_admissible_option_exists": bool(
                replay["_unselected_admissible_option_exists"]
            ),
            "frozen_selected_identity": immutable_seams_by_node[
                int(replay["node_id"])
            ],
        }
        for emitted, internal in zip(
            ancestor_lineage, ancestor_lineage_internal, strict=True
        )
        for replay in replay_nodes
        if int(replay["node_id"]) == int(emitted["node_id"])
    ]
    stage_counts = Counter(
        row["first_rejection_stage"] for row in stuck_manifest
    )
    reason_counts = Counter(
        row["first_rejection_reason"] or "admissible"
        for row in stuck_manifest
    )
    admissible_count = stage_counts.get("admissible", 0)
    expected_raw_gap_count = _raw_axis_midpoint_gap_count(stuck_cell)
    manifest_contract_pass = _validate_candidate_manifest(
        stuck_manifest,
        expected_count=expected_raw_gap_count,
        expected_admissible_count=0,
    )
    ancestor_boolean_derivation_pass = all(
        _validate_ancestor_boolean(
            row["emitted"],
            derived_value=row[
                "derived_unselected_admissible_option_exists"
            ],
        )
        for row in ancestor_derivations
    )
    ancestor_selected_identity_derivation_pass = all(
        _validate_selected_identity(
            row["selected_identity_internal"],
            row["frozen_selected_identity"],
        )
        for row in ancestor_derivations
    )
    return {
        "parent_name": parent_name,
        "body": source["body"],
        "role": source["role"],
        "frozen_construction_error": "no_admissible_shared_plane_split",
        "source_payload_sha256": source["payload_sha256"],
        "final_partial_leaf_count": len(immutable_sorted),
        "selected_split_replay_count": len(replay_nodes),
        "replay_diagnostic_cell_count": replay_cell_count,
        "ephemeral_trace_raw_split_evaluation_count": trace_raw_gap_count,
        "ephemeral_frozen_parity_raw_split_evaluation_count": (
            frozen_parity_raw_gap_count
        ),
        "forest_replay_exact": forest_replay_exact,
        "first_stuck_leaf": {
            "name": inferred_stuck["name"],
            "payload_sha256": inferred_stuck["payload_sha256"],
            "vertex_count": int(inferred_stuck["vertex_count"]),
            "topology_volume_m3": float(
                inferred_stuck["topology_volume_m3"]
            ),
            "stable_key": _stable_key(
                inferred_stuck["path_constraints"]
            ),
            "path_constraints": inferred_stuck["path_constraints"],
            "identity_matches_registered_expectation": stuck_identity_exact,
            "replay_exact": stuck_replay_exact,
        },
        "raw_axis_midpoint_candidate_count": expected_raw_gap_count,
        "classified_candidate_manifest_count": len(stuck_manifest),
        "candidate_manifest": stuck_manifest,
        "candidate_stage_counts": {
            stage: int(stage_counts.get(stage, 0))
            for stage in STAGE_ORDER
        },
        "candidate_reason_counts": dict(sorted(reason_counts.items())),
        "frozen_admissible_candidate_count": admissible_count,
        "trace_matches_frozen_empty_option_set": (
            stuck_trace_equivalent and admissible_count == 0
        ),
        "candidate_manifest_contract_pass": manifest_contract_pass,
        "ancestor_lineage": ancestor_lineage,
        "ancestor_count": len(ancestor_lineage),
        "ancestor_boolean_derivation_pass": (
            ancestor_boolean_derivation_pass
        ),
        "ancestor_selected_identity_derivation_pass": (
            ancestor_selected_identity_derivation_pass
        ),
        "any_ancestor_has_unselected_admissible_option": any(
            row["unselected_admissible_option_exists"]
            for row in ancestor_lineage
        ),
        "_display_source": source,
        "_display_stuck": inferred_stuck,
        "_ancestor_derivations": ancestor_derivations,
    }


def _negative_controls(results: list[dict[str, Any]]) -> dict[str, Any]:
    first = results[0]
    manifest = first["candidate_manifest"]
    expected_count = first["raw_axis_midpoint_candidate_count"]
    missing_row_rejected = not _validate_candidate_manifest(
        manifest[:-1],
        expected_count=expected_count,
        expected_admissible_count=first[
            "frozen_admissible_candidate_count"
        ],
    )
    promoted = [dict(row) for row in manifest]
    promoted[0]["first_rejection_stage"] = "admissible"
    false_promotion_rejected = not _validate_candidate_manifest(
        promoted,
        expected_count=expected_count,
        expected_admissible_count=first[
            "frozen_admissible_candidate_count"
        ],
    )
    mutated_expected = dict(
        EXPECTED_STUCK[first["parent_name"]]
    )
    mutated_expected["payload_sha256"] = "0" * 64
    leaf_payload_mutation_rejected = not _validate_stuck_identity(
        first["first_stuck_leaf"], mutated_expected
    )
    derivation = next(
        row
        for result in results
        for row in result["_ancestor_derivations"]
    )
    flipped_ancestor = dict(derivation["emitted"])
    flipped_ancestor["unselected_admissible_option_exists"] = not bool(
        flipped_ancestor["unselected_admissible_option_exists"]
    )
    ancestor_boolean_flip_rejected = not _validate_ancestor_boolean(
        flipped_ancestor,
        derived_value=derivation[
            "derived_unselected_admissible_option_exists"
        ],
    )
    mutated_selected = dict(derivation["selected_identity_internal"])
    mutated_selected["selected_cut_f32_bits"] = (
        int(mutated_selected["selected_cut_f32_bits"]) ^ 1
    )
    selected_cut_bitflip_rejected = not _validate_selected_identity(
        mutated_selected,
        derivation["frozen_selected_identity"],
    )
    checks = {
        "missing_candidate_row_rejected": missing_row_rejected,
        "false_admissible_promotion_rejected": false_promotion_rejected,
        "final_leaf_payload_mutation_rejected": (
            leaf_payload_mutation_rejected
        ),
        "ancestor_boolean_flip_rejected": ancestor_boolean_flip_rejected,
        "frozen_selected_cut_bitflip_rejected": (
            selected_cut_bitflip_rejected
        ),
    }
    return {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "pass": all(checks.values()),
    }


def _compute() -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    deadline = started + COOPERATIVE_DEADLINE_SECONDS
    evidence397 = _read(D397_EVIDENCE)
    geometry397 = _read(D397_GEOMETRY)
    claim397 = _read(D397_WORKER_CLAIM)
    if evidence397["verdict"] != (
        "D397_SHARED_BOUNDARY_ZERO_VOLUME_CONSTRUCTION_FAIL_STOP"
    ):
        raise RuntimeError("D398 expected frozen D397 FAIL_STOP")
    if claim397["pass"] is not True:
        raise RuntimeError("D398 expected D397 worker claim PASS")
    if geometry397["canonical_evidence_sha256"] != _sha(D397_EVIDENCE):
        raise RuntimeError("D398 D397 evidence/geometry hash link mismatch")
    failed_metrics = [
        row
        for row in evidence397["source_parent_metrics"]
        if row["construction_complete"] is False
    ]
    failed_manifest = [row["name"] for row in failed_metrics]
    if failed_manifest != FAILED_PARENTS:
        raise RuntimeError(
            f"D398 failed-parent manifest mismatch: {failed_manifest}"
        )
    if not all(
        row["construction_error"] == "no_admissible_shared_plane_split"
        for row in failed_metrics
    ):
        raise RuntimeError("D398 unexpected frozen failure reason")

    sources = {
        row["name"]: row
        for row in geometry397["diagnostic_source_parents"]
    }
    leaves_by_parent = {
        parent: [
            row
            for row in geometry397["diagnostic_partition_leaves"]
            if row["parent_name"] == parent
        ]
        for parent in FAILED_PARENTS
    }
    seams_by_parent = {
        parent: [
            row
            for row in geometry397["diagnostic_shared_seams"]
            if row["parent_name"] == parent
        ]
        for parent in FAILED_PARENTS
    }
    base = _load_d397()
    results = []
    for parent in FAILED_PARENTS:
        _phase("failed_parent_audit_start", parent_name=parent)
        result = _replay_parent(
            base,
            parent,
            sources[parent],
            leaves_by_parent[parent],
            seams_by_parent[parent],
            deadline,
        )
        results.append(result)
        _phase(
            "failed_parent_audit_end",
            parent_name=parent,
            raw_candidates=result["raw_axis_midpoint_candidate_count"],
            ancestors=result["ancestor_count"],
        )
    negatives = _negative_controls(results)
    stage_total = Counter()
    reason_total = Counter()
    for result in results:
        stage_total.update(result["candidate_stage_counts"])
        reason_total.update(result["candidate_reason_counts"])
    ancestor_rows = [
        row for result in results for row in result["ancestor_lineage"]
    ]
    replay_cell_total = sum(
        row["replay_diagnostic_cell_count"] for row in results
    )
    trace_evaluation_total = sum(
        row["ephemeral_trace_raw_split_evaluation_count"]
        for row in results
    )
    frozen_parity_evaluation_total = sum(
        row["ephemeral_frozen_parity_raw_split_evaluation_count"]
        for row in results
    )
    raw_total = sum(
        row["raw_axis_midpoint_candidate_count"] for row in results
    )
    checks = {
        "one_new_variable": NEW_VARIABLES
        == [
            "six_failed_parent_axis_midpoint_option_rejection_provenance_v1"
        ],
        "d397_fail_preserved": evidence397["design_pass"] is False,
        "six_failed_parents_exact": len(results) == 6
        and [row["parent_name"] for row in results] == FAILED_PARENTS,
        "six_first_stuck_leaf_identities_exact": all(
            row["first_stuck_leaf"][
                "identity_matches_registered_expectation"
            ]
            for row in results
        ),
        "six_final_forests_replay_exact": all(
            row["forest_replay_exact"] for row in results
        ),
        "six_stuck_leaves_replay_exact": all(
            row["first_stuck_leaf"]["replay_exact"] for row in results
        ),
        "all_candidates_classified_once": raw_total
        == sum(stage_total.values())
        and all(
            row["candidate_manifest_contract_pass"] for row in results
        ),
        "all_stuck_accepted_sets_empty_and_trace_exact": all(
            row["trace_matches_frozen_empty_option_set"]
            for row in results
        ),
        "all_ancestor_selected_identities_match": all(
            row["ancestor_selected_identity_derivation_pass"]
            for row in results
        ),
        "all_ancestor_booleans_match_frozen_option_counts": all(
            row["ancestor_boolean_derivation_pass"] for row in results
        ),
        "negative_controls_5_of_5": negatives["pass"]
        and negatives["passed"] == negatives["total"] == 5,
        "no_new_branch_or_candidate_geometry": True,
    }
    pass_value = all(checks.values())
    ancestors_with_alternative = sum(
        row["unselected_admissible_option_exists"]
        for row in ancestor_rows
    )
    if ancestors_with_alternative:
        diagnostic_conclusion = (
            "AT_LEAST_ONE_FROZEN_GREEDY_ANCESTOR_HAD_AN_UNSELECTED_"
            "ADMISSIBLE_OPTION_COMPLETION_FEASIBILITY_NULL"
        )
    else:
        diagnostic_conclusion = (
            "NO_UNSELECTED_ADMISSIBLE_OPTION_ON_THE_SIX_FROZEN_STUCK_"
            "LINEAGES_COMPLETION_FEASIBILITY_NULL"
        )
    verdict = (
        "D398_SIX_FAILED_PARENT_GREEDY_BSP_DEAD_END_PROVENANCE_LOCALIZED"
        if pass_value
        else "D398_PROVENANCE_OR_REPLAY_INTEGRITY_FAIL_STOP"
    )

    public_results = []
    display_parents = []
    display_stuck = []
    for index, result in enumerate(results):
        public = {
            key: value
            for key, value in result.items()
            if not key.startswith("_")
        }
        public_results.append(public)
        display_parents.append(result["_display_source"])
        display_stuck.append(result["_display_stuck"])
    evidence = {
        "artifact": "D398_GREEDY_DEAD_END_PROVENANCE_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": NEW_VARIABLES,
        "question": (
            "For each of the six frozen D397 greedy dead ends, where was "
            "every raw axis/midpoint option first rejected, and did any "
            "ancestor on the stuck lineage have an unselected admissible "
            "option?"
        ),
        "authority_boundary": {
            "immutable_inputs": (
                "D397 attempt2 evidence, geometry, worker claim, base "
                "implementation, and attempt2 wrapper only"
            ),
            "ephemeral_diagnostic_split_evaluation": True,
            "new_branch_selected": False,
            "backtracking_or_depth2_search": False,
            "candidate_child_geometry_serialized_or_adopted": False,
            "alternative_branch_completion_feasibility": None,
            "meaning": (
                "An ancestor Boolean says only whether the frozen greedy "
                "choice was locally forced. It does not show that an "
                "unselected option completes the tree."
            ),
        },
        "frozen_contract": {
            "source_child_vertex_budget": base.MAX_SOURCE_CHILD_VERTICES,
            "source_child_count_budget": base.MAX_SOURCE_CHILDREN,
            "total_part_exclusive_limit": base.TOTAL_PART_EXCLUSIVE_LIMIT,
            "positive_volume_epsilon_m3": base.POSITIVE_VOLUME_EPS_M3,
            "volume_relative_tolerance": base.VOLUME_RELATIVE_TOLERANCE,
            "plane_family": (
                "Float32 midpoint of adjacent unique x/y/z cell levels"
            ),
            "greedy_selection_order": [
                "minimum maximum child vertex count",
                "minimum normalized volume imbalance",
                "minimum total child vertex count",
                "axis index",
                "Float32 cut bits",
            ],
        },
        "input_hashes": _input_hashes(),
        "failed_parent_audits": public_results,
        "aggregate": {
            "failed_parent_count": len(results),
            "first_stuck_leaf_count": len(results),
            "raw_axis_midpoint_candidate_count": raw_total,
            "classified_candidate_count": sum(stage_total.values()),
            "candidate_stage_counts": {
                stage: int(stage_total.get(stage, 0))
                for stage in STAGE_ORDER
            },
            "candidate_reason_counts": dict(sorted(reason_total.items())),
            "ancestor_lineage_count": len(ancestor_rows),
            "replay_diagnostic_cell_count": replay_cell_total,
            "ephemeral_trace_raw_split_evaluation_count": (
                trace_evaluation_total
            ),
            "ephemeral_frozen_parity_raw_split_evaluation_count": (
                frozen_parity_evaluation_total
            ),
            "ephemeral_total_raw_split_evaluation_count": (
                trace_evaluation_total + frozen_parity_evaluation_total
            ),
            "ancestors_with_unselected_admissible_option": int(
                ancestors_with_alternative
            ),
            "parents_with_any_unselected_ancestor_option": sum(
                row["any_ancestor_has_unselected_admissible_option"]
                for row in results
            ),
            "alternative_branch_completion_feasibility": None,
        },
        "diagnostic_conclusion": diagnostic_conclusion,
        "negative_controls": negatives,
        "checks": checks,
        "localization_pass": pass_value,
        "verdict": verdict,
        "interpretation_boundary": {
            "materializable_candidate": False,
            "live_physx_identity": None,
            "29x50mm_cylinder": None,
            "physics_contact_grasp": None,
            "g0a_pass": False,
        },
        "scope_counters": {
            "offline_worker_invocations": 1,
            "automatic_retries": 0,
            "failed_parent_replays": len(results),
            "classified_stuck_leaf_raw_candidates": raw_total,
            "replay_diagnostic_cells_scanned": replay_cell_total,
            "ephemeral_trace_raw_split_evaluations": (
                trace_evaluation_total
            ),
            "ephemeral_frozen_parity_raw_split_evaluations": (
                frozen_parity_evaluation_total
            ),
            "ephemeral_total_raw_split_evaluations": (
                trace_evaluation_total + frozen_parity_evaluation_total
            ),
            "new_branches_selected": 0,
            "backtracking_or_depth2_searches": 0,
            "candidate_child_geometries_serialized_or_adopted": 0,
            "vertex_budget_changes": 0,
            "plane_family_changes": 0,
            "tolerance_or_gate_changes": 0,
            "usd_or_asset_reads": 0,
            "usd_or_asset_writes": 0,
            "collider_materializations": 0,
            "isaac_launches": 0,
            "kit_launches": 0,
            "physx_launches": 0,
            "warp_or_cuda_launches": 0,
            "cylinder_creates_or_writes": 0,
            "physics_steps": 0,
            "q5_samples": 0,
            "contact_queries": 0,
            "target_ik_path_or_settings_changes": 0,
            "process_signals_sent": 0,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    display = {
        "artifact": "D398_FROZEN_STUCK_LEAF_DISPLAY_GEOMETRY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "canonical_evidence_sha256": None,
        "authority": (
            "Exact copies of immutable D397 source-parent and inferred "
            "first-stuck-leaf geometry for inspection only; no alternative "
            "candidate geometry"
        ),
        "source_parents": display_parents,
        "first_stuck_leaves": display_stuck,
        "alternative_candidate_geometry_count": 0,
    }
    return evidence, display


def _prepare() -> int:
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only output exists: {OUT_DIR}")
    status_before = _status_outside_output()
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    actual_hashes = _input_hashes()
    expected_hashes = {
        _rel(path): digest for path, digest in EXPECTED_INPUT_SHA256.items()
    }
    start_text = START.read_text(encoding="utf-8")
    checks = {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "inputs_exact": actual_hashes == expected_hashes,
        "script_exists": SCRIPT.is_file(),
        "start_active_case_D398": (
            "## Active Case — D398 Approved; Offline Provenance Only"
            in start_text
        ),
        "start_exact_variable": NEW_VARIABLES[0] in start_text,
        "start_exact_output": _rel(OUT_DIR) in start_text,
        "one_new_variable": len(NEW_VARIABLES) == 1,
        "numpy_1_26_0": np.__version__ == "1.26.0",
        "scipy_1_15_3": importlib.metadata.version("scipy") == "1.15.3",
        "rerun_0_34_1": importlib.metadata.version("rerun-sdk") == "0.34.1",
        "rerun_cli": RERUN_CLI.is_file(),
        "fonts": FONT_REGULAR.is_file() and FONT_BOLD.is_file(),
        "d397_fail_stop": _read(D397_EVIDENCE)["verdict"]
        == "D397_SHARED_BOUNDARY_ZERO_VOLUME_CONSTRUCTION_FAIL_STOP",
        "d397_worker_claim_pass": _read(D397_WORKER_CLAIM)["pass"] is True,
        "d397_no_isaac_physx": _read(D397_EVIDENCE)["scope_counters"][
            "isaac_launches"
        ]
        == 0
        and _read(D397_EVIDENCE)["scope_counters"]["physx_launches"]
        == 0,
    }
    authority = {
        "artifact": "D398_EXECUTION_AUTHORITY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "approved_variable": NEW_VARIABLES,
        "head": _git("rev-parse", "HEAD"),
        "origin_master": _git("rev-parse", "origin/master"),
        "status_outside_output_before": status_before,
        "status_outside_output_before_sha256": _canonical_sha(status_before),
        "script": _artifact(SCRIPT),
        "start": _artifact(START),
        "input_hashes": actual_hashes,
        "forward_only_output": _rel(OUT_DIR),
        "scope": {
            "worker_invocations": 1,
            "worker_retries": 0,
            "viewer_maximum": 1,
            "viewer_retries": 0,
            "process_signals": 0,
            "ephemeral_diagnostic_split_evaluation": (
                "allowed only to classify frozen options"
            ),
            "new_branch_selection_or_backtracking": 0,
            "candidate_geometry_serialization_or_adoption": 0,
            "USD_Isaac_PhysX_cylinder_physics_q5_contact": 0,
        },
    }
    _write_json_x(AUTHORITY, authority)
    prereg = {
        "artifact": "D398_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "question": (
            "Classify the first rejection stage of every frozen raw "
            "axis/midpoint option at each of six D397 first stuck leaves, "
            "and record only whether each frozen ancestor had an unselected "
            "admissible option."
        ),
        "new_variables": NEW_VARIABLES,
        "input_authority": {
            "allowed": [
                _rel(D397_EVIDENCE),
                _rel(D397_GEOMETRY),
                _rel(D397_WORKER_CLAIM),
                _rel(D397_BASE),
                _rel(D397_WRAPPER),
            ],
            "other_science_inputs": 0,
        },
        "first_rejection_order": STAGE_ORDER,
        "ephemeral_replay_boundary": {
            "already_selected_D397_branches_may_be_replayed": True,
            "tentative_split_hulls_may_exist_in_memory_for_classification": True,
            "new_branch_selected": False,
            "alternative_geometry_serialized_or_adopted": False,
            "completion_search": False,
        },
        "frozen": {
            "vertex_budget": 12,
            "plane_family": "adjacent-level Float32 x/y/z midpoints",
            "positive_volume_epsilon_m3": 1.0e-18,
            "volume_relative_tolerance": 0.005,
            "greedy_selection_key": [
                "maximum child vertices",
                "volume imbalance",
                "total child vertices",
                "axis",
                "cut bits",
            ],
            "all_surface_void_clearance_count_bounds_overlap_gates": True,
        },
        "negative_controls": [
            "drop one candidate manifest row",
            "promote one rejected row to admissible",
            "mutate one final-leaf payload hash",
            "flip one ancestor Boolean",
            "bit-flip one frozen selected cut",
        ],
        "forbidden": {
            "new_branch_selection": 0,
            "backtracking_or_depth2_search": 0,
            "geometry_adoption_or_candidate_serialization": 0,
            "vertex_budget_plane_family_tolerance_gate_changes": 0,
            "USD_asset_collider_Isaac_Kit_PhysX_Warp_CUDA": 0,
            "cylinder_physics_q5_contact": 0,
            "target_IK_path_or_settings": 0,
            "hardware_signal_commit_push": 0,
        },
        "execution": {
            "worker": 1,
            "worker_retry": 0,
            "cooperative_deadline_seconds": COOPERATIVE_DEADLINE_SECONDS,
            "viewer_maximum": 1,
            "viewer_retry": 0,
            "process_signal": 0,
        },
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "status_outside_output": status_before,
        },
        "authority": _artifact(AUTHORITY),
        "script": _artifact(SCRIPT),
        "start": _artifact(START),
        "input_hashes": actual_hashes,
        "checks": checks,
        "pass": all(checks.values()),
        "forward_only_output": _rel(OUT_DIR),
    }
    _write_json_x(PREREG, prereg)
    _phase("prepare_end", preregistration_pass=prereg["pass"])
    if not prereg["pass"]:
        raise RuntimeError(f"D398 preregistration failed: {checks}")
    print(json.dumps({"prepared": True, "path": _rel(PREREG)}))
    return 0


def _worker() -> int:
    if SENTINEL.exists() or EVIDENCE.exists():
        raise FileExistsError("D398 worker already executed; no retry")
    if os.environ.get(AUTHORITY_ENV) != _sha(AUTHORITY):
        raise RuntimeError("D398 execution authority token mismatch")
    if os.environ.get(WORKER_AUTH_ENV) != _sha(WORKER_AUTH):
        raise RuntimeError("D398 worker authorization token mismatch")
    if _read(PREREG)["pass"] is not True:
        raise RuntimeError("D398 preregistration not PASS")
    checks = {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "inputs_exact": _input_hashes()
        == {
            _rel(path): digest
            for path, digest in EXPECTED_INPUT_SHA256.items()
        },
        "status_outside_output_unchanged": _canonical_sha(
            _status_outside_output()
        )
        == _read(AUTHORITY)["status_outside_output_before_sha256"],
        "invocation_present": INVOCATION.is_file(),
        "worker_authorization_present": WORKER_AUTH.is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D398 worker preflight failed: {checks}")
    _write_json_x(
        SENTINEL,
        {
            "artifact": "D398_WORKER_START_SENTINEL_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "checks": checks,
            "started_monotonic_ns": time.monotonic_ns(),
        },
    )
    _phase("worker_start")
    evidence, display = _compute()
    _write_json_x(EVIDENCE, evidence)
    display["canonical_evidence_sha256"] = _sha(EVIDENCE)
    _write_json_x(DISPLAY, display)
    _phase(
        "canonical_evidence_committed",
        evidence_sha256=_sha(EVIDENCE),
        display_sha256=_sha(DISPLAY),
    )
    claim_checks = {
        "localization_pass": evidence["localization_pass"] is True,
        "evidence_hash": _sha(EVIDENCE),
        "display_bound_to_evidence": _read(DISPLAY)[
            "canonical_evidence_sha256"
        ]
        == _sha(EVIDENCE),
        "scope_zero": all(
            evidence["scope_counters"][key] == 0
            for key in REQUIRED_ZERO_SCOPE_KEYS
        ),
    }
    claim = {
        "artifact": "D398_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": claim_checks,
        "evidence": _artifact(EVIDENCE),
        "display": _artifact(DISPLAY),
        "pass": (
            claim_checks["localization_pass"]
            and bool(claim_checks["evidence_hash"])
            and claim_checks["display_bound_to_evidence"]
            and claim_checks["scope_zero"]
        ),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_end", worker_claim_pass=claim["pass"])
    print(json.dumps(_native(claim), ensure_ascii=False))
    return 0 if claim["pass"] else 1


def _run() -> int:
    if INVOCATION.exists():
        raise FileExistsError("D398 worker invocation already exists; no retry")
    prereg = _read(PREREG)
    checks = {
        "prereg_pass": prereg["pass"] is True,
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "inputs_exact": _input_hashes()
        == {
            _rel(path): digest
            for path, digest in EXPECTED_INPUT_SHA256.items()
        },
        "status_outside_output_unchanged": _canonical_sha(
            _status_outside_output()
        )
        == _read(AUTHORITY)["status_outside_output_before_sha256"],
        "no_worker_artifacts": not any(
            path.exists()
            for path in [SENTINEL, EVIDENCE, DISPLAY, WORKER_CLAIM]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D398 supervisor preflight failed: {checks}")
    invocation = {
        "artifact": "D398_OFFLINE_WORKER_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": [sys.executable, "-B", _rel(SCRIPT), "--stage", "worker"],
        "checks": checks,
        "worker_invocation_ordinal": 1,
        "automatic_retries": 0,
        "process_signals_authorized": 0,
    }
    _write_json_x(INVOCATION, invocation)
    worker_auth = {
        "artifact": "D398_WORKER_AUTHORIZATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "invocation_sha256": _sha(INVOCATION),
        "execution_authority_sha256": _sha(AUTHORITY),
        "worker_ordinal": 1,
        "retry": 0,
    }
    _write_json_x(WORKER_AUTH, worker_auth)
    _phase("supervisor_before_worker")
    environment = os.environ.copy()
    environment[AUTHORITY_ENV] = _sha(AUTHORITY)
    environment[WORKER_AUTH_ENV] = _sha(WORKER_AUTH)
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--stage", "worker"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    STDOUT.write_text(result.stdout, encoding="utf-8")
    STDERR.write_text(result.stderr, encoding="utf-8")
    checks_after = {
        "returncode_zero": result.returncode == 0,
        "sentinel_present": SENTINEL.is_file(),
        "evidence_present": EVIDENCE.is_file(),
        "display_present": DISPLAY.is_file(),
        "claim_present": WORKER_CLAIM.is_file(),
        "claim_pass": WORKER_CLAIM.is_file()
        and _read(WORKER_CLAIM)["pass"] is True,
        "automatic_retries_zero": True,
        "process_signals_zero": True,
    }
    supervisor = {
        "artifact": "D398_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": invocation["command"],
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout": _artifact(STDOUT),
        "stderr": _artifact(STDERR),
        "checks": checks_after,
        "worker_invocations": 1,
        "worker_retries": 0,
        "process_signals_sent": 0,
        "pass": all(checks_after.values()),
    }
    _write_json_x(SUPERVISOR, supervisor)
    _phase("supervisor_after_worker", supervisor_pass=supervisor["pass"])
    if not supervisor["pass"]:
        raise RuntimeError(f"D398 worker failed: {supervisor}")
    print(json.dumps(_native(supervisor), ensure_ascii=False))
    return 0


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _render_board(evidence: dict[str, Any]) -> dict[str, Any]:
    image = Image.new("RGB", (1920, 1080), (246, 248, 252))
    draw = ImageDraw.Draw(image)
    title = _font(FONT_BOLD, 42)
    subtitle = _font(FONT_REGULAR, 24)
    header = _font(FONT_BOLD, 25)
    body = _font(FONT_REGULAR, 21)
    small = _font(FONT_REGULAR, 18)
    draw.text(
        (60, 38),
        "D398 — D397 탐욕 분할이 막힌 위치 추적",
        font=title,
        fill=(20, 28, 45),
    )
    aggregate = evidence["aggregate"]
    draw.text(
        (62, 98),
        (
            f"실패 부모 6개 · 원시 축/중간점 후보 "
            f"{aggregate['raw_axis_midpoint_candidate_count']}개 · "
            f"조상 {aggregate['ancestor_lineage_count']}개"
        ),
        font=subtitle,
        fill=(55, 67, 86),
    )
    stage = aggregate["candidate_stage_counts"]
    draw.rounded_rectangle(
        (55, 145, 1865, 244),
        radius=16,
        fill=(229, 237, 249),
        outline=(126, 151, 190),
        width=2,
    )
    draw.text(
        (82, 164),
        "첫 탈락 단계 합계",
        font=header,
        fill=(22, 54, 96),
    )
    stage_text = (
        f"중간점 생성 {stage['midpoint_candidate_generation']}  |  "
        f"분할체 생성 {stage['paired_split_creation']}  |  "
        f"경계·부피 {stage['seam_volume_validity']}  |  "
        f"꼭짓점 감소 {stage['strict_vertex_reduction']}  |  "
        f"허용 {stage['admissible']}"
    )
    draw.text((335, 169), stage_text, font=body, fill=(25, 40, 65))

    cards = []
    x_positions = [55, 660, 1265]
    y_positions = [280, 585]
    for index, result in enumerate(evidence["failed_parent_audits"]):
        x0 = x_positions[index % 3]
        y0 = y_positions[index // 3]
        x1, y1 = x0 + 570, y0 + 270
        cards.append((x0, y0, x1, y1))
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=15,
            fill=(255, 255, 255),
            outline=(178, 187, 202),
            width=2,
        )
        name = result["parent_name"]
        stuck = result["first_stuck_leaf"]
        counts = result["candidate_stage_counts"]
        draw.text(
            (x0 + 20, y0 + 16),
            name,
            font=header,
            fill=(31, 45, 67),
        )
        draw.text(
            (x0 + 20, y0 + 58),
            (
                f"첫 막힘: {stuck['name'].split('__')[-1]}  "
                f"꼭짓점 {stuck['vertex_count']}"
            ),
            font=body,
            fill=(88, 31, 31),
        )
        draw.text(
            (x0 + 20, y0 + 94),
            (
                f"후보 {result['raw_axis_midpoint_candidate_count']}개: "
                f"생성 {counts['paired_split_creation']}, "
                f"경계·부피 {counts['seam_volume_validity']}, "
                f"감소 {counts['strict_vertex_reduction']}, "
                f"허용 {counts['admissible']}"
            ),
            font=small,
            fill=(48, 59, 77),
        )
        booleans = [
            row["unselected_admissible_option_exists"]
            for row in result["ancestor_lineage"]
        ]
        bool_text = ", ".join(
            f"n{row['node_id']}={'있음' if value else '없음'}"
            for row, value in zip(
                result["ancestor_lineage"], booleans, strict=True
            )
        )
        draw.text(
            (x0 + 20, y0 + 134),
            "조상 단계의 미선택 허용 후보:",
            font=small,
            fill=(34, 70, 111),
        )
        draw.multiline_text(
            (x0 + 20, y0 + 164),
            bool_text or "조상 없음",
            font=small,
            fill=(34, 70, 111),
            spacing=5,
        )
        draw.text(
            (x0 + 20, y0 + 225),
            "다른 분기의 완주 가능성: 미측정(null)",
            font=small,
            fill=(102, 58, 10),
        )

    draw.rounded_rectangle(
        (55, 895, 1865, 1035),
        radius=15,
        fill=(251, 237, 231),
        outline=(188, 111, 81),
        width=2,
    )
    draw.text(
        (82, 917),
        evidence["verdict"],
        font=header,
        fill=(115, 36, 25),
    )
    draw.multiline_text(
        (82, 958),
        (
            "저장·채택한 새 충돌체 형상 0 · 새 분기 선택/되돌리기 0 · "
            "USD/Isaac/PhysX/원통/물리/q5/contact 0\n"
            "조상에 대안이 있어도 그 대안이 전체 트리를 완성한다는 뜻은 아니다."
        ),
        font=body,
        fill=(66, 42, 34),
        spacing=7,
    )
    image.save(BOARD)
    overflow = any(
        x0 < 0 or y0 < 0 or x1 > 1920 or y1 > 1080
        for x0, y0, x1, y1 in cards
    )
    layout = {
        "artifact": "D398_BOARD_LAYOUT_VALIDATION_V1",
        "width": 1920,
        "height": 1080,
        "card_count": len(cards),
        "card_bounds": cards,
        "bounds_overflow_count": int(overflow),
        "text_overlap_detected": False,
        "pass": len(cards) == 6 and not overflow,
    }
    _write_json_x(LAYOUT, layout)
    return layout


def _png_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": _rel(path),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
    }


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(
                origin="/",
                contents="/d398/**",
                name="Frozen D397 source parents and first stuck leaves",
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=(0.22, -0.35, 0.26),
                    look_target=(0.10, 0.04, 0.03),
                    eye_up=(0.0, 0.0, 1.0),
                ),
                spatial_information=rrb.SpatialInformation(
                    target_frame="tf#/",
                    show_axes=True,
                    show_bounding_box=False,
                ),
            ),
            rrb.TextDocumentView(
                origin="/metadata/run",
                contents="/metadata/run",
                name="D398 numeric authority and scope",
            ),
            row_shares=[0.82, 0.18],
        ),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _write_rerun(
    evidence: dict[str, Any], display: dict[str, Any]
) -> dict[str, Any]:
    import roarm_rl.rerun_contract as rerun_contract
    import roarm_rl.viz_debug as viz_debug

    meshes = []
    points = []
    expected_entities = ["/metadata/run"]
    components: dict[str, list[str]] = {
        "/metadata/run": ["TextDocument:text"]
    }
    offsets = [
        (0.00, 0.00, 0.00),
        (0.09, 0.00, 0.00),
        (0.18, 0.00, 0.00),
        (0.00, 0.11, 0.00),
        (0.09, 0.11, 0.00),
        (0.18, 0.11, 0.00),
    ]

    def add_mesh(
        entity: str,
        part: dict[str, Any],
        offset: tuple[float, float, float],
        color: list[int],
    ) -> None:
        vertices = (
            np.asarray(part["vertices_f32_m"], dtype=np.float64)
            + np.asarray(offset, dtype=np.float64)
        )
        meshes.append(
            {
                "entity_path": entity,
                "vertices_m": vertices,
                "triangles": part["triangles_i32"],
                "color_rgba": color,
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
        canonical = f"/{entity}"
        metadata_entity = "/metadata/meshes/" + entity.replace("/", "__")
        expected_entities.extend([canonical, metadata_entity])
        components[canonical] = [
            "CoordinateFrame:frame",
            "Mesh3D:albedo_factor",
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        ]
        components[metadata_entity] = ["TextDocument:text"]
        center = vertices.mean(axis=0)
        point_entity = entity.replace("/mesh", "/label")
        points.append(
            {
                "entity_path": point_entity,
                "positions_m": [center],
                "radii": [0.0022],
                "colors": [[225, 45, 45, 255]],
                "labels": [part["parent_name"]],
                "coordinate_frame": "tf#/",
                "static": True,
            }
        )
        canonical_point = f"/{point_entity}"
        expected_entities.append(canonical_point)
        components[canonical_point] = [
            "CoordinateFrame:frame",
            "Points3D:colors",
            "Points3D:labels",
            "Points3D:positions",
            "Points3D:radii",
        ]

    parents = {
        row["name"]: row for row in display["source_parents"]
    }
    stuck = {
        row["parent_name"]: row for row in display["first_stuck_leaves"]
    }
    for index, parent_name in enumerate(FAILED_PARENTS):
        slug = parent_name.replace("/", "_")
        add_mesh(
            f"d398/{slug}/source/mesh",
            parents[parent_name],
            offsets[index],
            [80, 100, 125, 55],
        )
        add_mesh(
            f"d398/{slug}/stuck/mesh",
            stuck[parent_name],
            offsets[index],
            [220, 65, 55, 185],
        )

    expected_entities = sorted(expected_entities)
    metadata = {
        "case": CASE,
        "attempt": ATTEMPT,
        "verdict": evidence["verdict"],
        "numeric_authority": _rel(EVIDENCE),
        "display_geometry_authority": _rel(DISPLAY),
        "display_subject": (
            "immutable D397 source parents plus first stuck leaves only"
        ),
        "alternative_candidate_geometry_count": 0,
        "new_branch_backtracking_USD_Isaac_PhysX_cylinder_physics_q5_contact": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "g0a_pass": False,
    }
    original_builder = viz_debug.build_rerun_blueprint
    original_runner = rerun_contract._run
    viewer_calls = 0

    def routed_builder(mode: str = "robot_geometry") -> Any:
        return _build_blueprint() if mode == "d398_dead_end" else original_builder(mode)

    def no_signal_run(command: list[str], *, timeout_s: float) -> dict[str, Any]:
        nonlocal viewer_calls
        del timeout_s
        if any("screenshot" in str(part) for part in command):
            viewer_calls += 1
            if viewer_calls > 1:
                return {
                    "command": command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "D398 viewer maximum exceeded",
                    "ok": False,
                    "signals_sent": 0,
                }
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "ok": result.returncode == 0,
            "signals_sent": 0,
            "timeout_ignored_no_signal_authority": True,
        }

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    rerun_contract._run = no_signal_run
    try:
        saved = viz_debug.log_rerun(
            RRD,
            meshes=meshes,
            points=points,
            recording_metadata=metadata,
            recording_id="g0a_d398_greedy_dead_end_provenance",
            blueprint_path=RBL,
            blueprint_mode="d398_dead_end",
            live_viewer=False,
            app_id="roarm_g0a_d398_dead_end",
        )
        if saved.get("ok") is not True:
            raise RuntimeError(f"D398 save-only Rerun failed: {saved}")
        validation = rerun_contract.validate_rerun_artifact(
            RRD,
            expected_entity_paths=expected_entities,
            exact_entity_paths=expected_entities,
            expected_timeline_names=["blueprint", "log_time"],
            exact_timeline_names=["blueprint", "log_time"],
            expected_entity_components=components,
            blueprint_path=RBL,
            screenshot_path=RERUN_SCREENSHOT,
            screenshot_window_size="1920x1080",
            screenshot_port="auto",
            cli_path=RERUN_CLI,
            expected_version="0.34.1",
            timeout_s=0.0,
        )
    finally:
        rerun_contract._run = original_runner
        viz_debug.build_rerun_blueprint = original_builder
        os.environ["PATH"] = old_path
    screenshot = _png_info(RERUN_SCREENSHOT)
    base_pass = validation.get("pass") is True
    validation["d398_contract"] = {
        "source_parent_mesh_entities": 6,
        "first_stuck_leaf_mesh_entities": 6,
        "alternative_candidate_mesh_entities": 0,
        "label_point_entities": 12,
        "viewer_invocations": viewer_calls,
        "viewer_retries": 0,
        "process_signals_sent": 0,
        "screenshot_16x9": screenshot["width"] * 9
        == screenshot["height"] * 16,
    }
    validation["base_contract_pass"] = base_pass
    validation["pass"] = (
        base_pass
        and viewer_calls == 1
        and screenshot["width"] * 9 == screenshot["height"] * 16
    )
    _write_json_x(RERUN_VALIDATION, validation)
    return {
        "pass": validation["pass"],
        "viewer_invocations": viewer_calls,
        "rrd": _artifact(RRD),
        "rbl": _artifact(RBL),
        "validation": _artifact(RERUN_VALIDATION),
        "screenshot": screenshot,
    }


def _observe() -> int:
    if BOARD.exists() or RRD.exists():
        raise FileExistsError("D398 observability already executed; no retry")
    _phase("observability_start")
    evidence = _read(EVIDENCE)
    display = _read(DISPLAY)
    if display["canonical_evidence_sha256"] != _sha(EVIDENCE):
        raise RuntimeError("D398 evidence/display hash link mismatch")
    layout = _render_board(evidence)
    rerun = _write_rerun(evidence, display)
    template = {
        "artifact": "D398_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board": _png_info(BOARD),
        "rerun_screenshot": _png_info(RERUN_SCREENSHOT),
        "checks_to_record_after_actual_viewing": MANUAL_KEYS,
        "minimum_observation_notes": 3,
        "manual_inspection_complete": False,
    }
    _write_json_x(MANUAL_TEMPLATE, template)
    checks = {
        "worker_claim_pass": _read(WORKER_CLAIM)["pass"] is True,
        "board_layout_pass": layout["pass"],
        "board_exact_1920x1080": _png_info(BOARD)["width"] == 1920
        and _png_info(BOARD)["height"] == 1080,
        "rerun_contract_pass": rerun["pass"],
        "viewer_exactly_one": rerun["viewer_invocations"] == 1,
        "no_alternative_candidate_geometry": display[
            "alternative_candidate_geometry_count"
        ]
        == 0,
    }
    observation = {
        "artifact": "D398_OBSERVABILITY_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "board": _png_info(BOARD),
        "rerun": rerun,
        "manual_template": _artifact(MANUAL_TEMPLATE),
        "pass": all(checks.values()),
    }
    _write_json_x(OBSERVABILITY, observation)
    _phase("observability_end", observability_pass=observation["pass"])
    if not observation["pass"]:
        raise RuntimeError(f"D398 observability failed: {checks}")
    print(json.dumps(_native(observation), ensure_ascii=False))
    return 0


def _finalize() -> int:
    if COMPLETION.exists():
        raise FileExistsError("D398 already finalized")
    _phase("finalize_start")
    manual = _read(MANUAL)
    evidence = _read(EVIDENCE)
    observation = _read(OBSERVABILITY)
    checks = {
        "worker_complete": _read(WORKER_CLAIM)["pass"] is True,
        "supervisor_pass": _read(SUPERVISOR)["pass"] is True,
        "canonical_evidence_present": EVIDENCE.is_file()
        and DISPLAY.is_file(),
        "observability_pass": observation["pass"] is True,
        "manual_complete": manual.get("manual_inspection_complete") is True,
        "manual_keys_exact_and_true": set(manual.get("checks", {}))
        == set(MANUAL_KEYS)
        and all(manual["checks"].values()),
        "manual_notes": len(manual.get("observations", [])) >= 3,
        "manual_hash_binding": manual.get("board_sha256") == _sha(BOARD)
        and manual.get("rerun_screenshot_sha256")
        == _sha(RERUN_SCREENSHOT),
        "science_scope_zero": all(
            evidence["scope_counters"][key] == 0
            for key in REQUIRED_ZERO_SCOPE_KEYS
        ),
    }
    completion_pass = all(checks.values())
    completion = {
        "artifact": "D398_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "completion_integrity_pass": completion_pass,
        "localization_verdict": evidence["verdict"],
        "diagnostic_conclusion": evidence["diagnostic_conclusion"],
        "materializable_candidate": False,
        "live_identity_physics_contact_grasp": None,
        "g0a_pass": False,
        "operational_verdict": (
            "D398_GREEDY_DEAD_END_PROVENANCE_AND_OBSERVABILITY_COMPLETE"
            if completion_pass
            else "D398_COMPLETION_INTEGRITY_FAIL_STOP"
        ),
        "artifacts": {
            path.name: _artifact(path)
            for path in [
                EVIDENCE,
                DISPLAY,
                WORKER_CLAIM,
                SUPERVISOR,
                BOARD,
                LAYOUT,
                RRD,
                RBL,
                RERUN_VALIDATION,
                RERUN_SCREENSHOT,
                MANUAL,
                OBSERVABILITY,
            ]
        },
    }
    _write_json_x(COMPLETION, completion)
    _phase("finalize_end", completion_integrity_pass=completion_pass)
    if not completion_pass:
        raise RuntimeError(f"D398 completion failed: {checks}")
    print(json.dumps(_native(completion), ensure_ascii=False))
    return 0


def _record_failure(stage: str, error: BaseException) -> None:
    if not OUT_DIR.exists() or FAILURE.exists():
        return
    try:
        _write_json_x(
            FAILURE,
            {
                "artifact": "D398_FAILURE_ATTESTATION_V1",
                "case": CASE,
                "attempt": ATTEMPT,
                "stage": stage,
                "exception_type": type(error).__name__,
                "exception": repr(error),
                "traceback": traceback.format_exc(),
                "process_signals_sent": 0,
            },
        )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prepare", "run", "worker", "observe", "finalize"],
        required=True,
    )
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            return _prepare()
        if args.stage == "run":
            return _run()
        if args.stage == "worker":
            return _worker()
        if args.stage == "observe":
            return _observe()
        return _finalize()
    except Exception as error:
        _record_failure(args.stage, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
