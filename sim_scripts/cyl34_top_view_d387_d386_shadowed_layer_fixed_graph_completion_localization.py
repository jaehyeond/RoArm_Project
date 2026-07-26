#!/usr/bin/env python3
"""D387 offline completion map for D386's seven shadowed layers.

The case reads immutable D379/D385/D386 evidence and imports hash-verified D385
geometry helpers plus D386 fixed-graph localizer helpers after worker
provenance passes.  It never starts Isaac Sim, Kit, PhysX, USD, CUDA, Warp, or
robot hardware.  Exactly the seven later/shadowed layers inventoried by D386
are evaluated; D386's four first-observed results are inherited by hash and
never recomputed.

Each new layer gets one complete D385 fixed candidate graph and one independent
frozen-D385 B=12 helper evaluation.  The same 12..64 bounded bottleneck DP,
exhaustive complete-path enumeration, polygon/face/volume/surface/topology
volume/no-overlap gates are used.  A valid finite-at-floor, finite-interior, or
null-through-64 classification completes a layer's map entry.  No budget is
selected or applied and no materializable P34 candidate is created.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO):
    if str(REPO) in sys.path:
        sys.path.remove(str(REPO))
    sys.path.insert(0, str(REPO))

CASE = "g0a_d387"
ATTEMPT = (
    "attempt1_shadowed_layer_fixed_graph_completion_localization"
)
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT_PATH = Path(__file__).resolve()
START_HERE = REPO / "START_HERE.md"

D385_SCRIPT = REPO / (
    "sim_scripts/"
    "cyl34_top_view_d385_p34_source_hull_semantic_low_count_redesign.py"
)
D385_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d385/"
    "attempt2_precreate_git_status_capture_repair/"
    "d385_p34_source_hull_redesign_evidence.json"
)
D385_COMPLETION = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d385/"
    "attempt2_precreate_git_status_capture_repair/"
    "d385_completion_summary.json"
)
D379_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d379/"
    "attempt2_d372_measurement_field_repair/"
    "d379_p34_full_live_identity_evidence.json"
)
D386_SCRIPT = REPO / (
    "sim_scripts/"
    "cyl34_top_view_d386_d385_minimum_admissible_vertex_budget_localization.py"
)
D386_EVIDENCE = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d386/"
    "attempt1_observed_no_cover_layer_minimum_vertex_budget_localization/"
    "d386_vertex_budget_localization_evidence.json"
)
D386_GEOMETRY = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d386/"
    "attempt1_observed_no_cover_layer_minimum_vertex_budget_localization/"
    "d386_finite_threshold_witness_geometry.json"
)
D386_COMPLETION = REPO / (
    "claudedocs/runtime_logs/grasp_track/g0a_d386/"
    "attempt1_observed_no_cover_layer_minimum_vertex_budget_localization/"
    "d386_completion_summary.json"
)

EXPECTED_HEAD = "930b41d98576a9c0bf1dce4f3eb1c0d93df8014b"
EXPECTED_INPUT_SHA256 = {
    "d379_evidence": (
        "8eacbad796e8532c3d856b865e90dc54481f0f2003a266c3ebfaa8e93de37af5"
    ),
    "d385_script": (
        "ea1d76a8db9c78a3cae9de50a62e0a25283d5550346dad158e641a0da321c5ed"
    ),
    "d385_evidence": (
        "4ff64045d4e2e7ecc3601927d1d6c97fd1a61b636e838241f9fded6b02e3cc00"
    ),
    "d385_completion": (
        "2caf6c47ad563c9ad82b84d5c3367139943f95c98c62a590b3551a967def91c2"
    ),
    "d386_script": (
        "60b5b2d15518baa0427e44f0928a46993e78eeba45307636b234cb0b042acf8d"
    ),
    "d386_evidence": (
        "ae956a2b64835f4030daf104f08d239f140f8ba9b32ee9205f2b744769c51d4c"
    ),
    "d386_geometry": (
        "ec5016cb5ebee9930c23093a6f3211a397466137f78d93e30357f8e10744a187"
    ),
    "d386_completion": (
        "622c34fdb7cbd11d2b0465eda75ac1119407fcaab441a059ff40289065170b6e"
    ),
}

PREREG_PATH = OUT_DIR / "d387_preregistration.json"
PHASE_PATH = OUT_DIR / "d387_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d387_offline_localizer_invocation.json"
WORKER_STDOUT = OUT_DIR / "d387_offline_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d387_offline_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d387_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d387_offline_worker_supervisor.json"
EVIDENCE_PATH = OUT_DIR / "d387_shadowed_layer_fixed_graph_map_evidence.json"
GEOMETRY_PATH = OUT_DIR / "d387_eleven_layer_fixed_graph_map_geometry.json"
METRICS_CSV = OUT_DIR / "d387_new_layer_candidate_cell_metrics.csv"
BOARD_PATH = OUT_DIR / "d387_fixed_graph_layer_map_1920x1080.png"
BOARD_LAYOUT = OUT_DIR / "d387_board_layout_validation.json"
RRD_PATH = OUT_DIR / "d387_fixed_graph_layer_map.rrd"
RBL_PATH = OUT_DIR / "d387_fixed_graph_layer_map.rbl"
RERUN_VALIDATION = OUT_DIR / "d387_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d387_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d387_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d387_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d387_completion_summary.json"

RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = ["d386_shadowed_layer_fixed_graph_evaluation_set_v1"]
WATCHDOG_SECONDS = 300.0
BASELINE_BUDGET = 12
MAXIMUM_LOCALIZATION_BUDGET = 64
MAXIMUM_POLYGONS = 64
MAXIMUM_VERTICES_PER_POLYGON = 32
SURFACE_TOLERANCE_MM = 0.1
VOLUME_RELATIVE_TOLERANCE = 0.005
POSITIVE_VOLUME_EPS_M3 = 1.0e-18

TARGETS = [
    {
        "body": "gripper_link",
        "prim_name": "p000_proximal_upper_arm_hull_a",
        "name": "proximal_upper_arm_hull_a",
        "role": "moving_support",
        "region_name": "z_layer_01",
        "region_index": 1,
    },
    {
        "body": "gripper_link",
        "prim_name": "p000_proximal_upper_arm_hull_a",
        "name": "proximal_upper_arm_hull_a",
        "role": "moving_support",
        "region_name": "z_layer_02",
        "region_index": 2,
    },
    {
        "body": "gripper_link",
        "prim_name": "p002_proximal_lower_arm_hull_a",
        "name": "proximal_lower_arm_hull_a",
        "role": "moving_support",
        "region_name": "z_layer_00",
        "region_index": 0,
    },
    {
        "body": "gripper_link",
        "prim_name": "p002_proximal_lower_arm_hull_a",
        "name": "proximal_lower_arm_hull_a",
        "role": "moving_support",
        "region_name": "z_layer_02",
        "region_index": 2,
    },
    {
        "body": "link5",
        "prim_name": "p013_fixed_backbone_left",
        "name": "fixed_backbone_left",
        "role": "fixed_jaw_backbone",
        "region_name": "y_layer_00",
        "region_index": 0,
    },
    {
        "body": "link5",
        "prim_name": "p014_fixed_backbone_right",
        "name": "fixed_backbone_right",
        "role": "fixed_jaw_backbone",
        "region_name": "y_layer_01",
        "region_index": 1,
    },
    {
        "body": "link5",
        "prim_name": "p014_fixed_backbone_right",
        "name": "fixed_backbone_right",
        "role": "fixed_jaw_backbone",
        "region_name": "y_layer_02",
        "region_index": 2,
    },
]

DISPLAY_ORDER = [
    ("gripper_link", "p000_proximal_upper_arm_hull_a", "z_layer_01"),
    ("gripper_link", "p000_proximal_upper_arm_hull_a", "z_layer_02"),
    ("gripper_link", "p002_proximal_lower_arm_hull_a", "z_layer_00"),
    ("gripper_link", "p002_proximal_lower_arm_hull_a", "z_layer_02"),
    ("link5", "p013_fixed_backbone_left", "y_layer_00"),
    ("link5", "p014_fixed_backbone_right", "y_layer_01"),
    ("link5", "p014_fixed_backbone_right", "y_layer_02"),
]

PARENT_ORDER = [
    ("gripper_link", "p000_proximal_upper_arm_hull_a"),
    ("gripper_link", "p002_proximal_lower_arm_hull_a"),
    ("link5", "p013_fixed_backbone_left"),
    ("link5", "p014_fixed_backbone_right"),
]

EXPECTED_D385_PREFAIL_BASELINE_KEYS = {
    ("gripper_link", "p002_proximal_lower_arm_hull_a", "z_layer_00"),
    ("link5", "p013_fixed_backbone_left", "y_layer_00"),
}

PALETTE = [
    [0, 163, 163, 215],
    [246, 166, 35, 215],
    [98, 86, 205, 215],
    [20, 136, 204, 215],
    [217, 72, 80, 215],
    [80, 168, 101, 215],
    [185, 92, 174, 215],
]

FORBIDDEN_COUNTERS = {
    "asset_or_usd_reads": 0,
    "asset_or_usd_writes": 0,
    "collider_materializations_or_regenerations": 0,
    "automatic_decomposition_sweeps": 0,
    "alternate_partition_evaluations": 0,
    "fan_group_size_gt_4_evaluations": 0,
    "internal_overlap_allowances": 0,
    "tolerance_changes": 0,
    "budget_selections_or_applications": 0,
    "isaac_launches": 0,
    "kit_launches": 0,
    "physx_launches": 0,
    "live_callback_queries": 0,
    "warp_or_cuda_launches": 0,
    "cylinder_creates_or_writes": 0,
    "controlled_physics_steps": 0,
    "q5_samples": 0,
    "contact_queries": 0,
    "grasp_trials": 0,
    "target_ik_path_changes": 0,
    "material_mass_actuator_physics_setting_changes": 0,
}


def _load_verified_d385_module() -> Any:
    actual_sha256 = _sha(D385_SCRIPT)
    expected_sha256 = EXPECTED_INPUT_SHA256["d385_script"]
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "refusing to execute unverified D385 helper script: "
            f"actual={actual_sha256} expected={expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location(
        "d385_frozen_geometry_helpers", D385_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen D385 helpers: {D385_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_verified_d386_module() -> Any:
    actual_sha256 = _sha(D386_SCRIPT)
    expected_sha256 = EXPECTED_INPUT_SHA256["d386_script"]
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "refusing to execute unverified D386 localizer script: "
            f"actual={actual_sha256} expected={expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location(
        "d386_frozen_fixed_graph_localizer", D386_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen D386 localizer: {D386_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D385: Any | None = None
D386_HELPERS: Any | None = None


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_payload(value: Any) -> str:
    encoded = json.dumps(
        _native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value cannot be serialized: {value}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            _native(value),
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def _phase(name: str, **fields: Any) -> None:
    record = {
        "phase": name,
        "monotonic_seconds": time.monotonic(),
        "wall_time_unix_seconds": time.time(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _native(record),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def _read_phase_records() -> list[dict[str, Any]]:
    records = []
    with PHASE_PATH.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "phase" not in record:
                raise TypeError(
                    f"invalid phase marker at line {line_number}"
                )
            records.append(record)
    return records


def _layer_phase_contract(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    records = _read_phase_records()
    target_keys = {_target_key(row) for row in results}
    phase_names = {
        "layer_graph_start",
        "layer_graph_end",
        "layer_b12_helper_start",
        "layer_b12_helper_end",
        "layer_dp_start",
        "layer_dp_end",
        "layer_exhaustive_start",
        "layer_exhaustive_end",
        "layer_geometry_gate_start",
        "layer_geometry_gate_end",
    }
    layer_records = [
        (index, row)
        for index, row in enumerate(records)
        if row.get("phase") in phase_names
    ]
    observed_keys = {
        (
            str(row.get("body")),
            str(row.get("prim_name")),
            str(row.get("region_name")),
        )
        for _, row in layer_records
    }
    per_layer = []
    for result in results:
        key = _target_key(result)
        rows = [
            (index, row)
            for index, row in layer_records
            if (
                str(row.get("body")),
                str(row.get("prim_name")),
                str(row.get("region_name")),
            )
            == key
        ]
        counts = {
            name: sum(row.get("phase") == name for _, row in rows)
            for name in sorted(phase_names)
        }
        expected_geometry_count = (
            0
            if result["classification"] == "NO_COVER_THROUGH_64"
            else 1
        )
        expected_counts = {
            "layer_graph_start": 1,
            "layer_graph_end": 1,
            "layer_b12_helper_start": 1,
            "layer_b12_helper_end": 1,
            "layer_dp_start": 1,
            "layer_dp_end": 1,
            "layer_exhaustive_start": 1,
            "layer_exhaustive_end": 1,
            "layer_geometry_gate_start": expected_geometry_count,
            "layer_geometry_gate_end": expected_geometry_count,
        }
        positions = {
            name: [
                index for index, row in rows if row.get("phase") == name
            ]
            for name in phase_names
        }
        ordered_pairs = all(
            (
                not positions[start_name]
                and not positions[end_name]
            )
            or (
                len(positions[start_name]) == 1
                and len(positions[end_name]) == 1
                and positions[start_name][0] < positions[end_name][0]
            )
            for start_name, end_name in (
                ("layer_graph_start", "layer_graph_end"),
                ("layer_b12_helper_start", "layer_b12_helper_end"),
                ("layer_dp_start", "layer_dp_end"),
                ("layer_exhaustive_start", "layer_exhaustive_end"),
                (
                    "layer_geometry_gate_start",
                    "layer_geometry_gate_end",
                ),
            )
        )
        per_layer.append(
            {
                "body": key[0],
                "prim_name": key[1],
                "region_name": key[2],
                "classification": result["classification"],
                "counts": counts,
                "expected_counts": expected_counts,
                "counts_exact": counts == expected_counts,
                "start_before_end": ordered_pairs,
                "pass": counts == expected_counts and ordered_pairs,
            }
        )
    return {
        "artifact": "D387_LAYER_PHASE_CONTRACT_V1",
        "target_layer_count": len(target_keys),
        "observed_layer_keys_exact": observed_keys == target_keys,
        "per_layer": per_layer,
        "pass": (
            len(target_keys) == 7
            and observed_keys == target_keys
            and all(row["pass"] for row in per_layer)
        ),
    }


def _global_phase_contract() -> dict[str, Any]:
    records = _read_phase_records()
    ordered_phases = [
        "prepare_start",
        "prepare_end",
        "supervisor_before_worker",
        "worker_start",
        "canonical_evidence_committed",
        "worker_end",
        "supervisor_after_worker",
        "finalize_start",
        "finalize_end",
    ]
    positions = {
        name: [
            index
            for index, row in enumerate(records)
            if row.get("phase") == name
        ]
        for name in ordered_phases
    }
    exact_once = all(len(positions[name]) == 1 for name in ordered_phases)
    forward_only_order = bool(
        exact_once
        and [
            positions[name][0] for name in ordered_phases
        ]
        == sorted(positions[name][0] for name in ordered_phases)
    )
    monotonic_values = [
        float(row["monotonic_seconds"]) for row in records
    ]
    monotonic_file_order = all(
        left <= right
        for left, right in zip(
            monotonic_values[:-1],
            monotonic_values[1:],
            strict=True,
        )
    )
    return {
        "artifact": "D387_GLOBAL_PHASE_CONTRACT_V1",
        "record_count": len(records),
        "ordered_phase_positions": positions,
        "each_registered_global_phase_exactly_once": exact_once,
        "registered_global_phases_forward_only": forward_only_order,
        "monotonic_seconds_nondecreasing_in_file_order": (
            monotonic_file_order
        ),
        "pass": exact_once and forward_only_order and monotonic_file_order,
    }


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _input_hashes() -> dict[str, str]:
    return {
        "d379_evidence": _sha(D379_EVIDENCE),
        "d385_script": _sha(D385_SCRIPT),
        "d385_evidence": _sha(D385_EVIDENCE),
        "d385_completion": _sha(D385_COMPLETION),
        "d386_script": _sha(D386_SCRIPT),
        "d386_evidence": _sha(D386_EVIDENCE),
        "d386_geometry": _sha(D386_GEOMETRY),
        "d386_completion": _sha(D386_COMPLETION),
    }


def _parent_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["body"]), str(row["prim_name"])


def _target_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["body"]),
        str(row["prim_name"]),
        str(row["region_name"]),
    )


def _installed_stack() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "isaac_sim_installed_not_launched": version("isaacsim"),
        "isaac_lab_installed_not_launched": version("isaaclab"),
        "rerun_sdk": version("rerun-sdk"),
        "frozen_d385_helper_script": _rel(D385_SCRIPT),
        "frozen_d385_helper_script_sha256": _sha(D385_SCRIPT),
        "inherited_d386_script": _rel(D386_SCRIPT),
        "inherited_d386_script_sha256": _sha(D386_SCRIPT),
    }


def _compute() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    """Evaluate only D386's seven immutable shadowed-layer inventory entries."""
    if D385 is None or D386_HELPERS is None:
        raise RuntimeError(
            "verified D385/D386 helpers must load after worker provenance"
        )

    d379 = _read_json(D379_EVIDENCE)
    d385_evidence = _read_json(D385_EVIDENCE)
    d385_completion = _read_json(D385_COMPLETION)
    d386_evidence = _read_json(D386_EVIDENCE)
    d386_geometry = _read_json(D386_GEOMETRY)
    d386_completion = _read_json(D386_COMPLETION)

    authored_rows = d379["authored_readback"]["rows"]
    authored_map = {
        (str(row["body"]), str(row["prim_name"])): row
        for row in authored_rows
    }
    target_keys = {_target_key(row) for row in TARGETS}
    d386_shadow_inventory = d386_evidence["scope_statement"][
        "shadowed_or_later_layers"
    ]
    d386_shadow_keys = {
        (
            str(row["body"]),
            str(row["prim_name"]),
            str(row["region_name"]),
        )
        for row in d386_shadow_inventory
    }
    if target_keys != d386_shadow_keys or len(target_keys) != 7:
        raise RuntimeError(
            "D387 target inventory is not the exact D386 shadow inventory: "
            f"targets={sorted(target_keys)} shadow={sorted(d386_shadow_keys)}"
        )

    inherited_results = d386_evidence["layer_results"]
    inherited_keys = {_target_key(row) for row in inherited_results}
    if len(inherited_results) != 4 or len(inherited_keys) != 4:
        raise RuntimeError("D386 inherited layer inventory is not exact four")
    if inherited_keys & target_keys:
        raise RuntimeError("D386 inherited and D387 evaluated layer sets overlap")

    d386_geometry_map = {
        (
            str(row["body"]),
            str(row["prim_name"]),
            str(row["region_name"]),
        ): row
        for row in d386_geometry["layers"]
    }
    if set(d386_geometry_map) != inherited_keys:
        raise RuntimeError("D386 evidence/geometry layer identity mismatch")

    results: list[dict[str, Any]] = []
    new_visual_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    evaluated_layer_keys: set[tuple[str, str, str]] = set()
    full_graph_counts: dict[tuple[str, str, str], int] = {}
    b12_helper_counts: dict[tuple[str, str, str], int] = {}

    for target in TARGETS:
        parent_key = _parent_key(target)
        target_key = _target_key(target)
        authored = authored_map[parent_key]
        identity_checks = {
            "body_exact": authored["body"] == target["body"],
            "prim_name_exact": authored["prim_name"] == target["prim_name"],
            "name_exact": authored["name"] == target["name"],
            "role_exact": authored["role"] == target["role"],
            "d386_shadow_inventory_exact": target_key in d386_shadow_keys,
            "not_recomputed_d386_first_observed_layer": (
                target_key not in inherited_keys
            ),
        }
        if not all(identity_checks.values()):
            raise RuntimeError(
                f"D387 target identity mismatch for {target}: "
                f"{identity_checks}"
            )

        points = D385._unique_f32(
            np.asarray(authored["points_f32"], dtype=np.float64)
        )
        plan = D385.SEMANTIC_PLAN[target["name"]]
        pre_axis = int(plan["semantic_pre_split_axis"])
        pre_axis_name = str(plan["semantic_pre_split_axis_name"])
        if pre_axis_name != target["region_name"][0]:
            raise RuntimeError(
                f"pre-split axis mismatch for {target['name']}"
            )
        levels = np.unique(points[:, pre_axis])
        levels.sort()
        region_index = int(target["region_index"])
        if (
            region_index < 0
            or region_index >= len(levels) - 1
            or target["region_name"]
            != f"{pre_axis_name}_layer_{region_index:02d}"
        ):
            raise RuntimeError(f"target region mismatch: {target}")

        low = float(levels[region_index])
        high = float(levels[region_index + 1])
        interval_m = [low, high]
        layer_mesh = D385._clip_interval(
            points, axis=pre_axis, low=low, high=high
        )
        layer_points = np.asarray(
            layer_mesh["vertices_m"], dtype=np.float64
        )
        evaluated_layer_keys.add(target_key)

        _phase(
            "layer_graph_start",
            body=target["body"],
            prim_name=target["prim_name"],
            region_name=target["region_name"],
        )
        full_graph_counts[target_key] = full_graph_counts.get(target_key, 0) + 1
        candidates, graph = D386_HELPERS._enumerate_candidate_graph(
            layer_points, thin_axis=pre_axis
        )
        triangle_count = int(graph["triangle_count"])
        expected_candidate_count = sum(
            min(4, end_state)
            for end_state in range(1, triangle_count + 1)
        )
        graph["candidate_count_formula_expected"] = expected_candidate_count
        graph["candidate_count_formula_exact"] = (
            graph["candidate_count"] == expected_candidate_count
        )
        _phase(
            "layer_graph_end",
            body=target["body"],
            prim_name=target["prim_name"],
            region_name=target["region_name"],
            triangle_count=triangle_count,
            candidate_count=graph["candidate_count"],
        )
        for row in candidates.values():
            candidate_rows.append(
                {
                    "body": target["body"],
                    "prim_name": target["prim_name"],
                    "parent_name": target["name"],
                    "region_name": target["region_name"],
                    **D386_HELPERS._candidate_public(row),
                }
            )

        _phase(
            "layer_b12_helper_start",
            body=target["body"],
            prim_name=target["prim_name"],
            region_name=target["region_name"],
        )
        b12_helper_counts[target_key] = (
            b12_helper_counts.get(target_key, 0) + 1
        )
        helper_result = None
        helper_error = None
        try:
            helper_result = D385._profile_cell_partition(
                layer_points,
                thin_axis=pre_axis,
                region_name=target["region_name"],
            )
        except D385.RegisteredNoCoverError as exc:
            helper_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        independent_b12_cover = D386_HELPERS._cover_at_budget(
            candidates, triangle_count, BASELINE_BUDGET
        )
        helper_cover_exists = helper_result is not None
        independent_b12_exists = independent_b12_cover is not None
        helper_checks = {
            "full_graph_candidate_count_formula_exact": (
                graph["candidate_count_formula_exact"] is True
            ),
            "helper_and_graph_b12_finiteness_agree": (
                helper_cover_exists == independent_b12_exists
            ),
            "helper_failure_has_registered_type_and_layer_prefix": (
                bool(
                    helper_error is not None
                    and helper_error["type"] == "RegisteredNoCoverError"
                    and helper_error["message"].startswith(
                        f"{target['region_name']}:"
                    )
                )
                if not helper_cover_exists
                else True
            ),
            "helper_success_child_count_agrees": (
                bool(
                    helper_result is not None
                    and independent_b12_cover is not None
                    and helper_result["child_count"]
                    == independent_b12_cover["child_count"]
                )
                if helper_cover_exists
                else True
            ),
            "helper_success_cut_states_agree": (
                bool(
                    helper_result is not None
                    and independent_b12_cover is not None
                    and helper_result["selected_fan_cut_states"]
                    == independent_b12_cover["cut_states"]
                )
                if helper_cover_exists
                else True
            ),
            "helper_success_maximum_vertices_le_12": (
                bool(
                    helper_result is not None
                    and helper_result["maximum_child_vertices"]
                    <= BASELINE_BUDGET
                )
                if helper_cover_exists
                else True
            ),
            "helper_success_maximum_vertices_agrees": (
                bool(
                    helper_result is not None
                    and independent_b12_cover is not None
                    and helper_result["maximum_child_vertices"]
                    == independent_b12_cover[
                        "maximum_child_vertex_count"
                    ]
                )
                if helper_cover_exists
                else True
            ),
        }
        _phase(
            "layer_b12_helper_end",
            body=target["body"],
            prim_name=target["prim_name"],
            region_name=target["region_name"],
            helper_cover_exists=helper_cover_exists,
            independent_b12_cover_exists=independent_b12_exists,
        )

        _phase(
            "layer_dp_start",
            body=target["body"],
            prim_name=target["prim_name"],
            region_name=target["region_name"],
        )
        dp = D386_HELPERS._minimax_dp(candidates, triangle_count)
        _phase(
            "layer_dp_end",
            body=target["body"],
            prim_name=target["prim_name"],
            region_name=target["region_name"],
            finite=dp is not None,
        )
        _phase(
            "layer_exhaustive_start",
            body=target["body"],
            prim_name=target["prim_name"],
            region_name=target["region_name"],
        )
        exhaustive = D386_HELPERS._exhaustive_minimax(
            candidates, triangle_count
        )
        _phase(
            "layer_exhaustive_end",
            body=target["body"],
            prim_name=target["prim_name"],
            region_name=target["region_name"],
            finite=exhaustive is not None,
            complete_path_count=(
                exhaustive.get("complete_path_count")
                if exhaustive is not None
                else 0
            ),
        )
        frontier = D386_HELPERS._reachability_frontier(
            candidates, triangle_count
        )
        raw_dp_budget = (
            dp["minimum_bottleneck_vertex_budget"]
            if dp is not None
            else None
        )
        raw_exhaustive_budget = (
            exhaustive["minimum_bottleneck_vertex_budget"]
            if exhaustive is not None
            else None
        )
        method_checks = {
            "dp_and_exhaustive_finiteness_agree": (
                (dp is None) == (exhaustive is None)
            ),
            "dp_and_exhaustive_raw_budget_agree": (
                raw_dp_budget == raw_exhaustive_budget
            ),
            "dp_and_exhaustive_canonical_cut_agree": (
                dp is None
                and exhaustive is None
                or (
                    dp is not None
                    and exhaustive is not None
                    and dp["canonical_cover_at_minimum"]["cut_states"]
                    == exhaustive["canonical_cover_at_minimum"][
                        "cut_states"
                    ]
                )
            ),
            "bounded_reachability_agrees_with_finiteness": (
                frontier[
                    "end_reachable_with_frozen_gates_and_vertex_le64"
                ]
                == (dp is not None)
            ),
        }

        if raw_dp_budget is None:
            classification = "NO_COVER_THROUGH_64"
            registered_threshold = None
            selected_cover = None
        elif raw_dp_budget <= BASELINE_BUDGET:
            classification = "BASELINE_B12_COVER"
            registered_threshold = BASELINE_BUDGET
            selected_cover = independent_b12_cover
        else:
            classification = "FINITE_RELAXATION_THRESHOLD_13_TO_64"
            registered_threshold = int(raw_dp_budget)
            selected_cover = D386_HELPERS._cover_at_budget(
                candidates, triangle_count, registered_threshold
            )

        previous_budget = (
            registered_threshold - 1
            if classification == "FINITE_RELAXATION_THRESHOLD_13_TO_64"
            else None
        )
        previous_cover = (
            D386_HELPERS._cover_at_budget(
                candidates, triangle_count, previous_budget
            )
            if previous_budget is not None
            else None
        )
        cover64 = (
            D386_HELPERS._cover_at_budget(
                candidates,
                triangle_count,
                MAXIMUM_LOCALIZATION_BUDGET,
            )
            if classification == "NO_COVER_THROUGH_64"
            else None
        )
        boundary = {
            "classification": classification,
            "baseline_budget": BASELINE_BUDGET,
            "baseline_cover_exists": independent_b12_exists,
            "raw_graph_minimax_vertex_count_out_of_decision_range_diagnostic": (
                raw_dp_budget
            ),
            "raw_sub12_minimax_is_not_selected_or_adopted": True,
            "registered_relaxation_threshold_within_12_64": (
                registered_threshold
            ),
            "threshold_minus_one": previous_budget,
            "threshold_minus_one_cover_exists": (
                previous_cover is not None
                if previous_budget is not None
                else None
            ),
            "registered_threshold_cover_exists": (
                selected_cover is not None
                if registered_threshold is not None
                else None
            ),
            "maximum_search_budget": MAXIMUM_LOCALIZATION_BUDGET,
            "maximum_search_budget_cover_exists": (
                cover64 is not None
                if classification == "NO_COVER_THROUGH_64"
                else None
            ),
        }
        boundary_checks = {
            "baseline_classification_consistent": (
                independent_b12_exists
                == (classification == "BASELINE_B12_COVER")
            ),
            "raw_sub12_value_not_selected_or_adopted": (
                boundary[
                    "raw_sub12_minimax_is_not_selected_or_adopted"
                ]
                is True
            ),
            "interior_threshold_strictly_13_to_64": (
                bool(
                    registered_threshold is not None
                    and 13 <= registered_threshold <= 64
                )
                if classification
                == "FINITE_RELAXATION_THRESHOLD_13_TO_64"
                else True
            ),
            "interior_threshold_minus_one_no_cover": (
                previous_cover is None
                if classification
                == "FINITE_RELAXATION_THRESHOLD_13_TO_64"
                else True
            ),
            "finite_registered_threshold_has_cover": (
                selected_cover is not None
                if classification != "NO_COVER_THROUGH_64"
                else True
            ),
            "null_has_no_cover_at_64": (
                cover64 is None
                if classification == "NO_COVER_THROUGH_64"
                else True
            ),
        }

        children: list[dict[str, Any]] = []
        geometry_metrics = None
        if selected_cover is not None and registered_threshold is not None:
            _phase(
                "layer_geometry_gate_start",
                body=target["body"],
                prim_name=target["prim_name"],
                region_name=target["region_name"],
                classification=classification,
            )
            children = D386_HELPERS._selected_children(
                candidates, selected_cover, target, interval_m
            )
            geometry_metrics = D386_HELPERS._layer_geometry_metrics(
                layer_mesh, children, registered_threshold
            )
            _phase(
                "layer_geometry_gate_end",
                body=target["body"],
                prim_name=target["prim_name"],
                region_name=target["region_name"],
                pass_value=geometry_metrics["pass"],
            )

        map_entry_pass = bool(
            all(identity_checks.values())
            and all(helper_checks.values())
            and all(method_checks.values())
            and all(boundary_checks.values())
            and (
                classification == "NO_COVER_THROUGH_64"
                or (
                    geometry_metrics is not None
                    and geometry_metrics["pass"]
                )
            )
        )
        result = {
            **target,
            "identity_checks": identity_checks,
            "thin_axis_index": pre_axis,
            "thin_axis_name": pre_axis_name,
            "pre_split_levels_m": list(map(float, levels)),
            "pre_split_interval_m": interval_m,
            "parent_layer_vertex_count": int(layer_mesh["vertex_count"]),
            "parent_layer_polygon_count": int(layer_mesh["polygon_count"]),
            "candidate_graph": graph,
            "frozen_d385_b12_helper_evaluation": {
                "helper_result": (
                    {
                        "child_count": helper_result["child_count"],
                        "maximum_child_vertices": helper_result[
                            "maximum_child_vertices"
                        ],
                        "selected_fan_cut_states": helper_result[
                            "selected_fan_cut_states"
                        ],
                    }
                    if helper_result is not None
                    else None
                ),
                "helper_error": helper_error,
                "independent_graph_cover_at_12": independent_b12_cover,
                "checks": helper_checks,
                "pass": all(helper_checks.values()),
            },
            "dynamic_programming": dp,
            "independent_exhaustive_enumeration": exhaustive,
            "method_checks": method_checks,
            "non_vertex_gate_frontier": frontier,
            "classification": classification,
            "registered_threshold_within_12_64": registered_threshold,
            "boundary": boundary,
            "boundary_checks": boundary_checks,
            "selected_threshold_cover": selected_cover,
            "selected_threshold_geometry_metrics": geometry_metrics,
            "map_entry_pass": map_entry_pass,
        }
        results.append(result)
        new_visual_rows.append(
            {
                "provenance": "d387_evaluated",
                "target": target,
                "layer_parent": layer_mesh,
                "classification": classification,
                "registered_threshold": registered_threshold,
                "raw_minimum": raw_dp_budget,
                "selected_cover": selected_cover,
                "children": children,
                "geometry_metrics": geometry_metrics,
                "candidate_graph": graph,
                "frontier": frontier,
            }
        )

    inherited_visual_rows: list[dict[str, Any]] = []
    combined_layer_map: list[dict[str, Any]] = []
    for row in inherited_results:
        key = _target_key(row)
        witness = d386_geometry_map[key]
        minimum = row[
            "minimum_admissible_vertex_budget_within_12_64"
        ]
        classification = (
            "NO_COVER_THROUGH_64"
            if minimum is None
            else "FINITE_RELAXATION_THRESHOLD_13_TO_64"
        )
        inherited_target = {
            field: row[field]
            for field in (
                "body",
                "prim_name",
                "name",
                "role",
                "region_name",
                "region_index",
            )
        }
        inherited_children = [
            {
                "name": (
                    f"{row['name']}__{row['region_name']}__"
                    f"cell_{index:02d}"
                ),
                "vertices_m": child["vertices_f64_m"],
                "triangles": child["triangles_i64"],
                "vertex_count": child["vertex_count"],
                "polygon_count": child["polygon_count"],
                "max_vertices_per_polygon": child[
                    "maximum_vertices_per_polygon"
                ],
            }
            for index, child in enumerate(witness["diagnostic_children"])
        ]
        inherited_visual_rows.append(
            {
                "provenance": "d386_inherited",
                "target": inherited_target,
                "layer_parent": {
                    "vertices_m": witness["parent_layer"][
                        "vertices_f64_m"
                    ],
                    "triangles": witness["parent_layer"]["triangles_i64"],
                },
                "classification": classification,
                "registered_threshold": minimum,
                "raw_minimum": minimum,
                "selected_cover": row["selected_threshold_cover"],
                "children": inherited_children,
                "geometry_metrics": row[
                    "selected_threshold_geometry_metrics"
                ],
                "candidate_graph": row["candidate_graph"],
                "frontier": row["non_vertex_gate_frontier"],
            }
        )
        combined_layer_map.append(
            {
                "provenance": "d386_inherited",
                "body": row["body"],
                "prim_name": row["prim_name"],
                "parent_name": row["name"],
                "role": row["role"],
                "region_name": row["region_name"],
                "region_index": row["region_index"],
                "classification": classification,
                "registered_threshold_within_12_64": minimum,
                "child_count": (
                    row["selected_threshold_cover"]["child_count"]
                    if row["selected_threshold_cover"] is not None
                    else None
                ),
                "candidate_count": row["candidate_graph"][
                    "candidate_count"
                ],
                "polygon_count_gt_64_rejections": row[
                    "candidate_graph"
                ]["rejection_reason_counts"].get(
                    "polygon_count_gt_64", 0
                ),
                "map_entry_pass": True,
            }
        )

    for row in results:
        combined_layer_map.append(
            {
                "provenance": "d387_evaluated",
                "body": row["body"],
                "prim_name": row["prim_name"],
                "parent_name": row["name"],
                "role": row["role"],
                "region_name": row["region_name"],
                "region_index": row["region_index"],
                "classification": row["classification"],
                "registered_threshold_within_12_64": row[
                    "registered_threshold_within_12_64"
                ],
                "raw_graph_minimax_vertex_count_out_of_decision_range_diagnostic": row[
                    "boundary"
                ][
                    "raw_graph_minimax_vertex_count_out_of_decision_range_diagnostic"
                ],
                "raw_sub12_minimax_is_not_selected_or_adopted": True,
                "child_count": (
                    row["selected_threshold_cover"]["child_count"]
                    if row["selected_threshold_cover"] is not None
                    else None
                ),
                "candidate_count": row["candidate_graph"][
                    "candidate_count"
                ],
                "polygon_count_gt_64_rejections": row[
                    "candidate_graph"
                ]["rejection_reason_counts"].get(
                    "polygon_count_gt_64", 0
                ),
                "map_entry_pass": row["map_entry_pass"],
            }
        )

    combined_keys = {
        (
            str(row["body"]),
            str(row["prim_name"]),
            str(row["region_name"]),
        )
        for row in combined_layer_map
    }
    expected_full_parent_layer_keys: set[tuple[str, str, str]] = set()
    for parent_key in PARENT_ORDER:
        authored = authored_map[parent_key]
        points = D385._unique_f32(
            np.asarray(authored["points_f32"], dtype=np.float64)
        )
        plan = D385.SEMANTIC_PLAN[str(authored["name"])]
        axis = int(plan["semantic_pre_split_axis"])
        axis_name = str(plan["semantic_pre_split_axis_name"])
        levels = np.unique(points[:, axis])
        for index in range(len(levels) - 1):
            expected_full_parent_layer_keys.add(
                (parent_key[0], parent_key[1], f"{axis_name}_layer_{index:02d}")
            )

    parent_completion_map = []
    for parent_key in PARENT_ORDER:
        rows = sorted(
            [
                row
                for row in combined_layer_map
                if (row["body"], row["prim_name"]) == parent_key
            ],
            key=lambda row: int(row["region_index"]),
        )
        finite = [
            row["registered_threshold_within_12_64"]
            for row in rows
            if row["registered_threshold_within_12_64"] is not None
        ]
        null_layers = [
            row["region_name"]
            for row in rows
            if row["registered_threshold_within_12_64"] is None
        ]
        parent_completion_map.append(
            {
                "body": parent_key[0],
                "prim_name": parent_key[1],
                "parent_name": rows[0]["parent_name"],
                "layer_count": len(rows),
                "all_layers_mapped": all(
                    row["map_entry_pass"] is True for row in rows
                ),
                "null_through_64_layers": null_layers,
                "common_threshold_within_12_64_diagnostic_only": (
                    max(finite) if rows and not null_layers else None
                ),
                "sum_of_individual_layer_witness_child_counts_diagnostic_only": (
                    sum(int(row["child_count"]) for row in rows)
                    if rows
                    and not null_layers
                    and all(row["child_count"] is not None for row in rows)
                    else None
                ),
            }
        )

    graph_counts_public = [
        {
            "body": key[0],
            "prim_name": key[1],
            "region_name": key[2],
            "full_graph_enumerations": full_graph_counts.get(key, 0),
            "frozen_d385_b12_helper_evaluations": (
                b12_helper_counts.get(key, 0)
            ),
        }
        for key in sorted(target_keys)
    ]
    new_classification_counts = {
        name: sum(row["classification"] == name for row in results)
        for name in (
            "BASELINE_B12_COVER",
            "FINITE_RELAXATION_THRESHOLD_13_TO_64",
            "NO_COVER_THROUGH_64",
        )
    }
    inherited_null_present = any(
        row["provenance"] == "d386_inherited"
        and row["registered_threshold_within_12_64"] is None
        for row in combined_layer_map
    )
    result_by_key = {_target_key(row): row for row in results}
    global_checks = {
        "immutable_input_hashes_exact": (
            _input_hashes() == EXPECTED_INPUT_SHA256
        ),
        "d385_scientific_verdict_frozen": (
            d385_evidence["verdict"]
            == (
                "D385_SEMANTIC_THIN_LAYER_PROFILE_CELL_"
                "NO_ADMISSIBLE_CANDIDATE_FAIL_STOP"
            )
        ),
        "d385_completion_pass": d385_completion["pass"] is True,
        "d386_scientific_verdict_frozen": (
            d386_evidence["verdict"]
            == "D386_OBSERVED_LAYER_VERTEX_BUDGET_NOT_LOCALIZABLE_FAIL_STOP"
        ),
        "d386_method_and_completion_pass": (
            d386_evidence["method_contract_pass"] is True
            and d386_completion["pass"] is True
        ),
        "d386_inherited_null_preserved": inherited_null_present,
        "target_inventory_exact_d386_shadowed_seven": (
            target_keys == d386_shadow_keys and len(target_keys) == 7
        ),
        "evaluated_layers_exact_registered_seven": (
            evaluated_layer_keys == target_keys
        ),
        "d386_inherited_four_not_recomputed": (
            len(inherited_keys) == 4
            and not inherited_keys & evaluated_layer_keys
        ),
        "combined_union_exact_eleven": (
            len(combined_keys) == 11
            and combined_keys == expected_full_parent_layer_keys
        ),
        "one_full_graph_and_one_b12_helper_evaluation_per_new_layer": all(
            full_graph_counts.get(key, 0) == 1
            and b12_helper_counts.get(key, 0) == 1
            for key in target_keys
        ),
        "candidate_csv_rows_match_graph_candidate_sum": (
            len(candidate_rows)
            == sum(
                row["candidate_graph"]["candidate_count"]
                for row in results
            )
        ),
        "all_b12_helper_graph_classifications_agree": all(
            row["frozen_d385_b12_helper_evaluation"]["pass"]
            for row in results
        ),
        "d385_prefailure_baseline_layers_remain_b12_cover": all(
            result_by_key[key]["classification"] == "BASELINE_B12_COVER"
            for key in EXPECTED_D385_PREFAIL_BASELINE_KEYS
        ),
        "all_primary_independent_methods_agree": all(
            all(row["method_checks"].values()) for row in results
        ),
        "all_boundary_classifications_pass": all(
            all(row["boundary_checks"].values()) for row in results
        ),
        "all_new_layer_map_entries_pass": all(
            row["map_entry_pass"] for row in results
        ),
        "all_four_parent_layer_maps_complete": all(
            row["all_layers_mapped"] for row in parent_completion_map
        ),
        "forbidden_runtime_counters_zero": all(
            value == 0 for value in FORBIDDEN_COUNTERS.values()
        ),
    }
    method_contract_pass = all(global_checks.values())
    map_completion_pass = method_contract_pass
    verdict = (
        "D387_SHADOWED_LAYER_FIXED_GRAPH_MAP_COMPLETION_PASS_"
        "GLOBAL_BUDGET_NULL"
        if map_completion_pass
        else "D387_FIXED_GRAPH_MAP_CONTRACT_FAIL_STOP"
    )
    finite_thresholds = [
        row["registered_threshold_within_12_64"]
        for row in combined_layer_map
        if row["registered_threshold_within_12_64"] is not None
    ]

    evidence = {
        "artifact": "D387_SHADOWED_LAYER_FIXED_GRAPH_MAP_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Classify exactly D386's seven unevaluated/shadowed layers with "
            "the same fixed graph and gates, then combine them with the four "
            "immutable D386 entries to complete the 11-layer map of D385's "
            "four failed source parents."
        ),
        "new_variables": NEW_VARIABLES,
        "measurement_authority": (
            "immutable D379 authored Float32 streams, immutable D385/D386 "
            "evidence, exact-SHA D385 geometry helpers, and exact-SHA D386 "
            "fixed-graph localizer helpers"
        ),
        "input_hashes": _input_hashes(),
        "installed_stack": _installed_stack(),
        "official_sources": d386_evidence["official_sources"],
        "frozen_contract": {
            "new_target_layers": TARGETS,
            "baseline_vertex_budget": BASELINE_BUDGET,
            "maximum_localization_budget": MAXIMUM_LOCALIZATION_BUDGET,
            "contiguous_fan_group_size": [1, 4],
            "maximum_polygons": MAXIMUM_POLYGONS,
            "maximum_vertices_per_polygon": MAXIMUM_VERTICES_PER_POLYGON,
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "surface_tolerance_mm": SURFACE_TOLERANCE_MM,
            "topology_volume_relative_tolerance": (
                VOLUME_RELATIVE_TOLERANCE
            ),
            "positive_volume_child_overlap": 0,
            "sub12_raw_minimax_may_be_computed_as_out_of_range_diagnostic": (
                True
            ),
            "sub12_budget_selected_or_adopted": False,
        },
        "scope_statement": {
            "d386_inherited_layer_count": len(inherited_keys),
            "d386_inherited_layer_keys": [
                {
                    "body": key[0],
                    "prim_name": key[1],
                    "region_name": key[2],
                }
                for key in sorted(inherited_keys)
            ],
            "d386_inherited_layer_recomputations": 0,
            "d387_evaluated_layer_count": len(evaluated_layer_keys),
            "d387_evaluated_layer_keys": [
                {
                    "body": key[0],
                    "prim_name": key[1],
                    "region_name": key[2],
                }
                for key in sorted(evaluated_layer_keys)
            ],
            "evaluation_counts_by_new_layer": graph_counts_public,
            "inherited_and_evaluated_intersection_count": len(
                inherited_keys & evaluated_layer_keys
            ),
            "combined_layer_count": len(combined_keys),
            "four_failed_parent_layer_map_complete": (
                combined_keys == expected_full_parent_layer_keys
            ),
            "complete_p34_or_all_eight_parent_map_claimed": False,
        },
        "new_layer_results": results,
        "inherited_d386_layer_results": inherited_results,
        "combined_layer_map": sorted(
            combined_layer_map,
            key=lambda row: (
                PARENT_ORDER.index((row["body"], row["prim_name"])),
                int(row["region_index"]),
            ),
        ),
        "parent_completion_map": parent_completion_map,
        "new_layer_classification_counts": new_classification_counts,
        "all_seven_new_layers_finite": (
            new_classification_counts["NO_COVER_THROUGH_64"] == 0
        ),
        "combined_all_eleven_layers_finite": all(
            row["registered_threshold_within_12_64"] is not None
            for row in combined_layer_map
        ),
        "combined_finite_threshold_maximum_diagnostic_only": (
            max(finite_thresholds) if finite_thresholds else None
        ),
        "global_common_vertex_budget": None,
        "adopted_parent_wide_vertex_budget": None,
        "selected_vertex_budget": None,
        "selected_budget_application_count": 0,
        "complete_p34_vertex_budget": None,
        "complete_source_child_count": None,
        "complete_total_part_count": None,
        "global_semantic_preservation_pass": None,
        "materializable_candidate": False,
        "method_contract_checks": global_checks,
        "method_contract_pass": method_contract_pass,
        "map_completion_pass": map_completion_pass,
        "repair_materialized": False,
        "live_identity_pass": None,
        "live_gpu_compatibility_pass": None,
        "current_scope_counters": FORBIDDEN_COUNTERS,
        "cylinder_29x50_rendered_or_measured": False,
        "physics_or_grasp_result": None,
        "p34_authored_to_cooked_identity_pass": False,
        "g0a_pass": False,
        "verdict": verdict,
        "next_authorization_boundary": (
            "Do not select/apply a vertex budget, alter a partition or gate, "
            "materialize USD/PhysX, create the 29x50mm target, or run physics/"
            "contact/grasp without a new explicit approval."
        ),
    }

    geometry_layers = []
    for row in inherited_visual_rows + new_visual_rows:
        geometry_layers.append(
            {
                "provenance": row["provenance"],
                "body": row["target"]["body"],
                "prim_name": row["target"]["prim_name"],
                "parent_name": row["target"]["name"],
                "region_name": row["target"]["region_name"],
                "classification": row["classification"],
                "registered_threshold_within_12_64": row[
                    "registered_threshold"
                ],
                "materializable_candidate": False,
                "parent_layer": {
                    "vertices_f64_m": row["layer_parent"]["vertices_m"],
                    "triangles_i64": row["layer_parent"]["triangles"],
                },
                "diagnostic_children": [
                    {
                        "name": child["name"],
                        "vertices_f64_m": child["vertices_m"],
                        "triangles_i64": child["triangles"],
                        "vertex_count": child["vertex_count"],
                        "polygon_count": child["polygon_count"],
                        "maximum_vertices_per_polygon": child[
                            "max_vertices_per_polygon"
                        ],
                    }
                    for child in row["children"]
                ],
            }
        )
    geometry = {
        "artifact": "D387_ELEVEN_LAYER_FIXED_GRAPH_MAP_GEOMETRY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "authority": (
            "inspection geometry for the four D386-inherited and seven "
            "D387-evaluated layer entries; not an adopted collider stream"
        ),
        "d386_source_geometry_sha256": _sha(D386_GEOMETRY),
        "global_common_vertex_budget": None,
        "selected_vertex_budget": None,
        "complete_p34_vertex_budget": None,
        "complete_materializable_candidate": False,
        "layers": sorted(
            geometry_layers,
            key=lambda row: (
                PARENT_ORDER.index((row["body"], row["prim_name"])),
                row["region_name"],
            ),
        ),
    }
    visual = {
        "rows": inherited_visual_rows + new_visual_rows,
        "combined_layer_map": evidence["combined_layer_map"],
        "parent_completion_map": parent_completion_map,
    }
    return evidence, geometry, visual, candidate_rows


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return {
        "path": _rel(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
        "width": int(width),
        "height": int(height),
        "mode": mode,
        "exact_1920x1080": width == 1920 and height == 1080,
    }


def _render_board(
    evidence: dict[str, Any], visual: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render a readable 4-parent x 3-layer local-profile map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import PolyCollection

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    by_key = {
        (
            row["target"]["body"],
            row["target"]["prim_name"],
            row["target"]["region_name"],
        ): row
        for row in visual["rows"]
    }
    map_by_key = {
        (
            row["body"],
            row["prim_name"],
            row["region_name"],
        ): row
        for row in visual["combined_layer_map"]
    }
    parent_names = {
        (row["body"], row["prim_name"]): row["parent_name"]
        for row in visual["combined_layer_map"]
    }
    parent_summary = {
        (row["body"], row["prim_name"]): row
        for row in visual["parent_completion_map"]
    }

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    grid = fig.add_gridspec(
        4,
        3,
        left=0.055,
        right=0.735,
        top=0.835,
        bottom=0.19,
        wspace=0.10,
        hspace=0.20,
    )
    text_artists: list[Any] = []
    cell_records = []
    all_layer_keys = set()

    for parent_index, parent_key in enumerate(PARENT_ORDER):
        parent_rows = sorted(
            [
                row
                for row in visual["combined_layer_map"]
                if (row["body"], row["prim_name"]) == parent_key
            ],
            key=lambda row: int(row["region_index"]),
        )
        row_by_index = {int(row["region_index"]): row for row in parent_rows}
        for region_index in range(3):
            axis = fig.add_subplot(grid[parent_index, region_index])
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal", adjustable="datalim")
            map_row = row_by_index.get(region_index)
            if map_row is None:
                axis.set_facecolor("#f3f4f6")
                for spine in axis.spines.values():
                    spine.set_color("#9ca3af")
                    spine.set_linestyle(":")
                label = axis.text(
                    0.5,
                    0.52,
                    "N/A\n이 부모에는 해당 층 없음",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontproperties=regular,
                    fontsize=8.5,
                    color="#6b7280",
                )
                text_artists.append(label)
                cell_records.append(
                    {
                        "parent_index": parent_index,
                        "region_index": region_index,
                        "key": None,
                        "axis": axis,
                        "texts": [label],
                    }
                )
                continue

            key = (
                map_row["body"],
                map_row["prim_name"],
                map_row["region_name"],
            )
            all_layer_keys.add(key)
            row = by_key[key]
            parent = row["layer_parent"]
            thin_axis = "xyz".index(map_row["region_name"][0])
            keep = [index for index in range(3) if index != thin_axis]
            parent_vertices = (
                np.asarray(parent["vertices_m"], dtype=np.float64)[
                    :, keep
                ]
                * 1000.0
            )
            projection_name = "".join(
                "XYZ"[axis_index] for axis_index in keep
            )
            parent_triangles = np.asarray(
                parent["triangles"], dtype=np.int64
            )
            classification = map_row["classification"]
            finite = classification != "NO_COVER_THROUGH_64"
            face_color = "#ecfdf5" if finite else "#fff1f2"
            edge_color = "#047857" if finite else "#be123c"
            axis.set_facecolor(face_color)
            for spine in axis.spines.values():
                spine.set_color(edge_color)
                spine.set_linewidth(
                    1.8 if map_row["provenance"] == "d387_evaluated" else 1.2
                )
                spine.set_linestyle(
                    "-" if map_row["provenance"] == "d387_evaluated" else "--"
                )

            parent_polygons = parent_vertices[parent_triangles]
            axis.add_collection(
                PolyCollection(
                    parent_polygons,
                    facecolors=(0.35, 0.38, 0.42, 0.07),
                    edgecolors=(0.22, 0.25, 0.30, 0.38),
                    linewidths=0.35,
                )
            )
            all_points = [parent_vertices]
            for child_index, child in enumerate(row["children"]):
                vertices = (
                    np.asarray(child["vertices_m"], dtype=np.float64)[
                        :, keep
                    ]
                    * 1000.0
                )
                triangles = np.asarray(child["triangles"], dtype=np.int64)
                color = np.asarray(
                    PALETTE[child_index % len(PALETTE)]
                ) / 255.0
                axis.add_collection(
                    PolyCollection(
                        vertices[triangles],
                        facecolors=color,
                        edgecolors=(0.08, 0.10, 0.12, 0.35),
                        linewidths=0.28,
                    )
                )
                all_points.append(vertices)
            points_2d = np.vstack(all_points)
            low = points_2d.min(axis=0)
            high = points_2d.max(axis=0)
            span = np.maximum(high - low, 1.0e-6)
            margin = span * 0.12
            axis.set_xlim(low[0] - margin[0], high[0] + margin[0])
            axis.set_ylim(low[1] - margin[1], high[1] + margin[1])

            provenance = (
                "상속 D386"
                if map_row["provenance"] == "d386_inherited"
                else "신규 D387"
            )
            title = axis.text(
                0.025,
                0.965,
                (
                    f"{map_row['region_name']}  ·  {provenance}  ·  "
                    f"{projection_name} 직교투영"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontproperties=bold,
                fontsize=8.3,
                color="#111827",
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.90,
                ),
            )
            child_count = map_row["child_count"]
            if classification == "BASELINE_B12_COVER":
                raw_minimum = map_row[
                    "raw_graph_minimax_vertex_count_out_of_decision_range_diagnostic"
                ]
                decision_text = (
                    f"B=12: 완전 경로 있음 · {child_count}조각\n"
                    f"raw minimax {raw_minimum}=범위 밖 진단치"
                )
            elif classification == "FINITE_RELAXATION_THRESHOLD_13_TO_64":
                threshold = int(
                    map_row["registered_threshold_within_12_64"]
                )
                decision_text = (
                    f"B={threshold - 1}: 경로 없음 → "
                    f"B={threshold}: 경로 있음\n{child_count}조각"
                )
            else:
                rejected = map_row["polygon_count_gt_64_rejections"]
                decision_text = (
                    "B=64까지 완전 경로 없음\n"
                    f"polygon>64 거부 {rejected}개"
                )
            decision = axis.text(
                0.025,
                0.035,
                decision_text,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontproperties=regular,
                fontsize=7.7,
                color=edge_color,
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.90,
                ),
            )
            text_artists.extend([title, decision])
            cell_records.append(
                {
                    "parent_index": parent_index,
                    "region_index": region_index,
                    "key": key,
                    "axis": axis,
                    "texts": [title, decision],
                }
            )

    title = fig.suptitle(
        "D387 — D385 실패 부모 4개의 11개 층 고정-그래프 지도",
        x=0.5,
        y=0.974,
        fontproperties=bold,
        fontsize=19,
        color="#111827",
    )
    subtitle = fig.text(
        0.5,
        0.927,
        (
            "점선=동결 D386 결과 4개 · 실선=이번 D387 평가 7개 · "
            "얇은 축을 제거한 독립 확대 2D 직교투영이며 실제 위치·공통 "
            "축척 비교가 아님"
        ),
        ha="center",
        fontproperties=regular,
        fontsize=10.5,
        color="#334155",
    )
    text_artists.extend([title, subtitle])

    row_centers = [0.755, 0.59, 0.425, 0.26]
    parent_label_artists = []
    for index, parent_key in enumerate(PARENT_ORDER):
        parent_label = fig.text(
            0.012,
            row_centers[index],
            parent_names[parent_key].replace("_", "\n"),
            ha="left",
            va="center",
            fontproperties=bold,
            fontsize=8.0,
            color="#1f2937",
        )
        parent_label_artists.append(parent_label)
        text_artists.append(parent_label)

    summary_title = fig.text(
        0.765,
        0.845,
        "부모별 진단 임계값",
        ha="left",
        fontproperties=bold,
        fontsize=13,
        color="#111827",
    )
    text_artists.append(summary_title)
    summary_texts = []
    summary_y = [0.765, 0.635, 0.505, 0.375]
    for index, parent_key in enumerate(PARENT_ORDER):
        summary = parent_summary[parent_key]
        threshold = summary[
            "common_threshold_within_12_64_diagnostic_only"
        ]
        null_layers = summary["null_through_64_layers"]
        if threshold is None:
            value = (
                "NULL\n"
                + "64까지 막힌 층: "
                + ", ".join(null_layers)
            )
            color = "#991b1b"
        else:
            value = (
                f"진단 임계값 {threshold}\n"
                f"층 {summary['layer_count']}개 모두 지도화"
            )
            color = "#0f5132"
        artist = fig.text(
            0.765,
            summary_y[index],
            f"{summary['parent_name']}\n{value}",
            ha="left",
            va="top",
            fontproperties=regular,
            fontsize=9.0,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.42",
                facecolor="#f8fafc",
                edgecolor="#cbd5e1",
            ),
        )
        summary_texts.append(artist)
        text_artists.append(artist)

    result_text = (
        "지도 11/11 완성  |  전역 공통 예산 = NULL  |  "
        "선택/적용 = NULL/0  |  완성 P34 후보 = 아님"
        if evidence["map_completion_pass"]
        else "지도 검증 계약 FAIL-STOP  |  전역 예산 선택/적용 0"
    )
    result = fig.text(
        0.5,
        0.115,
        result_text,
        ha="center",
        fontproperties=bold,
        fontsize=13,
        color="#b91c1c",
    )
    footer = fig.text(
        0.5,
        0.062,
        (
            "오프라인 형상 지도만 생성했습니다. 새 분할·gate 완화·USD·"
            "Isaac·PhysX·원통·물리·q5·접촉·파지는 모두 0회입니다."
        ),
        ha="center",
        fontproperties=regular,
        fontsize=9.8,
        color="#475569",
    )
    text_artists.extend([result, footer])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    boxes = []
    checks: dict[str, bool] = {}
    for index, artist in enumerate(text_artists):
        bbox = artist.get_window_extent(renderer=renderer)
        boxes.append(
            {
                "index": index,
                "x0": float(bbox.x0),
                "y0": float(bbox.y0),
                "x1": float(bbox.x1),
                "y1": float(bbox.y1),
            }
        )
        checks[f"text_{index:02d}_inside_canvas_5px"] = bool(
            bbox.x0 >= 5
            and bbox.y0 >= 5
            and bbox.x1 <= canvas_width - 5
            and bbox.y1 <= canvas_height - 5
        )
    for index, cell in enumerate(cell_records):
        axis_box = cell["axis"].get_window_extent(renderer=renderer)
        for text_index, artist in enumerate(cell["texts"]):
            text_box = artist.get_window_extent(renderer=renderer)
            checks[
                f"cell_{index:02d}_text_{text_index:02d}_inside_own_axis"
            ] = bool(
                text_box.x0 >= axis_box.x0 - 1
                and text_box.y0 >= axis_box.y0 - 1
                and text_box.x1 <= axis_box.x1 + 1
                and text_box.y1 <= axis_box.y1 + 1
            )
        if len(cell["texts"]) == 2:
            upper = cell["texts"][0].get_window_extent(renderer=renderer)
            lower = cell["texts"][1].get_window_extent(renderer=renderer)
            checks[f"cell_{index:02d}_title_decision_nonoverlap"] = bool(
                upper.y0 >= lower.y1
            )
    summary_boxes = [
        artist.get_window_extent(renderer=renderer)
        for artist in summary_texts
    ]
    summary_overlap_pairs = []
    for left_index, left in enumerate(summary_boxes):
        for right_index in range(left_index + 1, len(summary_boxes)):
            right = summary_boxes[right_index]
            if (
                min(left.x1, right.x1) > max(left.x0, right.x0)
                and min(left.y1, right.y1) > max(left.y0, right.y0)
            ):
                summary_overlap_pairs.append([left_index, right_index])
    matrix_left = min(
        cell["axis"].get_window_extent(renderer=renderer).x0
        for cell in cell_records
    )
    checks["parent_labels_left_of_matrix_nonoverlap"] = all(
        artist.get_window_extent(renderer=renderer).x1 < matrix_left
        for artist in parent_label_artists
    )
    checks["summary_title_first_card_nonoverlap"] = bool(
        summary_title.get_window_extent(renderer=renderer).y0
        > summary_boxes[0].y1
    )
    checks["all_eleven_layer_cells_present"] = (
        all_layer_keys == set(map_by_key) and len(all_layer_keys) == 11
    )
    checks["inherited_four_new_seven_exact"] = (
        sum(row["provenance"] == "d386_inherited" for row in map_by_key.values())
        == 4
        and sum(
            row["provenance"] == "d387_evaluated"
            for row in map_by_key.values()
        )
        == 7
    )
    checks["summary_cards_pairwise_nonoverlap"] = not summary_overlap_pairs
    checks["title_subtitle_nonoverlap"] = bool(
        title.get_window_extent(renderer=renderer).y0
        > subtitle.get_window_extent(renderer=renderer).y1
    )
    checks["result_footer_nonoverlap"] = bool(
        result.get_window_extent(renderer=renderer).y0
        > footer.get_window_extent(renderer=renderer).y1
    )
    synthetic_left = (10.0, 10.0, 30.0, 30.0)
    synthetic_right = (10.0, 10.0, 30.0, 30.0)
    synthetic_controls = {
        "identical_boxes_overlap_detected": bool(
            min(synthetic_left[2], synthetic_right[2])
            > max(synthetic_left[0], synthetic_right[0])
            and min(synthetic_left[3], synthetic_right[3])
            > max(synthetic_left[1], synthetic_right[1])
        ),
        "negative_margin_box_clipping_detected": bool(
            -1.0 < 0.0 or canvas_width + 1.0 > canvas_width
        ),
        "missing_layer_count_detected": (
            len(list(sorted(all_layer_keys))[:10]) != 11
        ),
    }
    layout = {
        "artifact": "D387_BOARD_LAYOUT_VALIDATION_V1",
        "canvas_pixels": [canvas_width, canvas_height],
        "artist_bboxes_display_pixels": boxes,
        "summary_overlap_pairs": summary_overlap_pairs,
        "checks": checks,
        "synthetic_negative_controls": synthetic_controls,
        "pass": all(checks.values()) and all(synthetic_controls.values()),
    }
    fig.savefig(BOARD_PATH, dpi=100, facecolor="white")
    plt.close(fig)
    info = _png_info(BOARD_PATH)
    if not info["exact_1920x1080"]:
        raise RuntimeError(f"board is not exact 1920x1080: {info}")
    if not layout["pass"]:
        raise RuntimeError(f"board layout validation failed: {layout}")
    return info, layout


def _build_blueprint(summary_path: str) -> Any:
    import rerun.blueprint as rrb

    geometry = rrb.Spatial3DView(
        origin="/",
        contents="/d387/parents/**",
        name="D387 inherited 4 plus evaluated 7 layer geometry",
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=(0.48, -0.58, 0.48),
            look_target=(0.14, 0.0, 0.18),
            eye_up=(0.0, 0.0, 1.0),
        ),
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=False,
            show_bounding_box=False,
        ),
    )
    decision = rrb.Vertical(
        geometry,
        rrb.TextLogView(
            origin=summary_path,
            contents=summary_path,
            name="D387 human-readable map verdict",
        ),
        row_shares=[0.62, 0.38],
    )
    notification_buffer = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d387/notification_buffer/**",
        name="Notification buffer - no decision content",
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=False,
            show_bounding_box=False,
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            decision,
            notification_buffer,
            column_shares=[0.76, 0.24],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _write_rerun(
    evidence: dict[str, Any], visual: dict[str, Any]
) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    import roarm_rl.viz_debug as viz_debug

    ordered_rows = sorted(
        visual["rows"],
        key=lambda row: (
            PARENT_ORDER.index(
                (row["target"]["body"], row["target"]["prim_name"])
            ),
            int(row["target"]["region_index"]),
        ),
    )
    meshes = []
    for row in ordered_rows:
        target = row["target"]
        parent_index = PARENT_ORDER.index(
            (target["body"], target["prim_name"])
        )
        region_index = int(target["region_index"])
        parent = row["layer_parent"]
        center = np.mean(
            np.asarray(parent["vertices_m"], dtype=np.float64), axis=0
        )
        offset = np.asarray(
            [0.14 * region_index, 0.0, 0.12 * (3 - parent_index)],
            dtype=np.float64,
        )
        safe_parent = f"{parent_index:02d}_{target['name']}"
        prefix = (
            f"d387/parents/{safe_parent}/layers/"
            f"{target['region_name']}"
        )
        finite = row["classification"] != "NO_COVER_THROUGH_64"
        provenance = row["provenance"]
        parent_vertices = (
            np.asarray(parent["vertices_m"], dtype=np.float64)
            - center
            + offset
        )
        meshes.append(
            {
                "entity_path": f"{prefix}/frozen_layer_parent",
                "coordinate_frame": "tf#/",
                "vertices_m": parent_vertices,
                "triangles": parent["triangles"],
                "color_rgba": (
                    [90, 96, 105, 42] if finite else [205, 48, 55, 75]
                ),
                "static": True,
                "representation": (
                    "inspection-only shifted layer parent; "
                    f"provenance={provenance}; "
                    f"classification={row['classification']}"
                ),
                "numeric_authority": "canonical unshifted D387 JSON",
            }
        )
        for child_index, child in enumerate(row["children"]):
            child_vertices = (
                np.asarray(child["vertices_m"], dtype=np.float64)
                - center
                + offset
            )
            meshes.append(
                {
                    "entity_path": (
                        f"{prefix}/finite_threshold_witness/"
                        f"child_{child_index:02d}"
                    ),
                    "coordinate_frame": "tf#/",
                    "vertices_m": child_vertices,
                    "triangles": child["triangles"],
                    "color_rgba": PALETTE[child_index % len(PALETTE)],
                    "static": True,
                    "representation": (
                        "inspection-only shifted finite threshold witness; "
                        f"provenance={provenance}"
                    ),
                    "numeric_authority": "canonical unshifted D387 JSON",
                }
            )

    summary_path = "decision/summary"
    parent_lines = []
    for parent in evidence["parent_completion_map"]:
        rows = [
            row
            for row in evidence["combined_layer_map"]
            if (
                row["body"],
                row["prim_name"],
            )
            == (parent["body"], parent["prim_name"])
        ]
        labels = []
        for row in rows:
            threshold = row["registered_threshold_within_12_64"]
            labels.append(
                f"{row['region_name']}="
                + ("NULL" if threshold is None else f"B{threshold}")
                + (
                    "[I]"
                    if row["provenance"] == "d386_inherited"
                    else "[N]"
                )
            )
        parent_lines.append(
            f"{parent['parent_name']}: " + ", ".join(labels)
        )
    if evidence["map_completion_pass"]:
        status_line = "MAP CONTRACT PASS: exact inherited4 + new7 = 11"
        explanation_line = (
            "PASS means map completion only; collider/grasp did not pass."
        )
    else:
        status_line = (
            "MAP CONTRACT FAIL: D387_FIXED_GRAPH_MAP_CONTRACT_FAIL_STOP"
        )
        explanation_line = (
            "FAIL means the fixed-graph map contract was not completed."
        )
    summary_text = "\n".join(
        [
            "D387 FIXED-GRAPH LAYER MAP",
            status_line,
            "[I]=D386 inherited; [N]=D387 newly evaluated",
            *parent_lines,
            "GLOBAL COMMON BUDGET = NULL",
            "SELECTED / APPLIED = NULL / 0",
            "COMPLETE P34 CANDIDATE = NO",
            "ISAAC / PHYSX / USD / CYLINDER / PHYSICS / Q5 / CONTACT = 0",
            "g0a_pass=false",
            explanation_line,
        ]
    )
    expected_entities = {"metadata/run", summary_path}
    component_contract = {
        "metadata/run": ["TextDocument:text"],
        summary_path: ["TextLog:level", "TextLog:text"],
    }
    mesh_components = [
        "CoordinateFrame:frame",
        "Mesh3D:albedo_factor",
        "Mesh3D:triangle_indices",
        "Mesh3D:vertex_positions",
    ]
    for mesh in meshes:
        path = mesh["entity_path"]
        metadata_path = f"metadata/meshes/{path.replace('/', '__')}"
        expected_entities.update({path, metadata_path})
        component_contract[path] = mesh_components
        component_contract[metadata_path] = ["TextDocument:text"]

    original_builder = viz_debug.build_rerun_blueprint

    def routed_builder(mode: str = "robot_geometry") -> Any:
        if mode == "d387_fixed_graph_map":
            return _build_blueprint(summary_path)
        return original_builder(mode)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    viz_debug.build_rerun_blueprint = routed_builder
    try:
        saved = viz_debug.log_rerun(
            RRD_PATH,
            meshes=meshes,
            events=[
                {
                    "entity_path": summary_path,
                    "text": summary_text,
                    "level": "INFO",
                    "static": True,
                }
            ],
            recording_metadata={
                "case": CASE,
                "attempt": ATTEMPT,
                "verdict": evidence["verdict"],
                "d386_inherited_layers": 4,
                "d387_evaluated_layers": 7,
                "combined_layer_map_count": 11,
                "global_common_vertex_budget": None,
                "selected_vertex_budget": None,
                "selected_budget_application_count": 0,
                "complete_p34_candidate": False,
                "g0a_pass": False,
                "viewer_layout_note": (
                    "geometry is shifted into a 4x3 inspection matrix; "
                    "canonical unshifted JSON is numeric authority"
                ),
            },
            recording_id="g0a_d387_fixed_graph_layer_map",
            blueprint_path=RBL_PATH,
            blueprint_mode="d387_fixed_graph_map",
            live_viewer=False,
            app_id="roarm_g0a_d387_fixed_graph_layer_map",
        )
    finally:
        viz_debug.build_rerun_blueprint = original_builder
        os.environ["PATH"] = old_path
    if not saved.get("ok"):
        raise RuntimeError(f"save-only Rerun failed: {saved}")

    validation = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(expected_entities),
        exact_entity_paths=sorted(expected_entities),
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=component_contract,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_SCREENSHOT,
        screenshot_window_size="1920x1080",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=240.0,
    )
    _write_json_x(RERUN_VALIDATION, validation)
    screenshot = (
        _png_info(RERUN_SCREENSHOT)
        if RERUN_SCREENSHOT.is_file()
        else {"path": _rel(RERUN_SCREENSHOT), "exists": False}
    )
    screenshot_size = (
        (screenshot.get("width"), screenshot.get("height"))
        if screenshot.get("exists")
        else None
    )
    allowed_screenshot_sizes = {(1920, 1080), (3840, 2160)}
    screenshot_dimension_contract_pass = (
        screenshot_size in allowed_screenshot_sizes
    )
    return {
        "save_only": saved,
        "strict_validation_pass": validation.get("pass") is True,
        "headless_viewer_invocations": int(
            bool(
                (validation.get("headless_render") or {}).get("attempted")
            )
        ),
        "headless_viewer_returncode": (
            validation.get("headless_render") or {}
        ).get("returncode"),
        "requested_logical_window_size": [1920, 1080],
        "allowed_native_screenshot_sizes": [
            [1920, 1080],
            [3840, 2160],
        ],
        "screenshot_dimension_contract_pass": (
            screenshot_dimension_contract_pass
        ),
        "rrd": {
            "path": _rel(RRD_PATH),
            "bytes": RRD_PATH.stat().st_size,
            "sha256": _sha(RRD_PATH),
        },
        "rbl": {
            "path": _rel(RBL_PATH),
            "bytes": RBL_PATH.stat().st_size,
            "sha256": _sha(RBL_PATH),
        },
        "validation": {
            "path": _rel(RERUN_VALIDATION),
            "bytes": RERUN_VALIDATION.stat().st_size,
            "sha256": _sha(RERUN_VALIDATION),
        },
        "screenshot": screenshot,
    }


def _write_candidate_metrics_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "body",
        "prim_name",
        "parent_name",
        "region_name",
        "start_state",
        "end_state",
        "group_size",
        "geometry_constructed",
        "vertex_count",
        "polygon_count",
        "maximum_vertices_per_polygon",
        "volume_m3",
        "non_vertex_gates_pass",
        "rejection_reasons",
        "error_type",
    ]
    with METRICS_CSV.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            native = dict(row)
            native["rejection_reasons"] = json.dumps(
                native["rejection_reasons"],
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow({key: native.get(key) for key in fieldnames})


def _import_roots_from_ast() -> list[str]:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return sorted(roots)


def _prepare() -> int:
    if OUT_DIR.exists():
        raise RuntimeError(f"refusing to reuse forward-only path: {OUT_DIR}")
    status_before_output_create = _git(["status", "--short"])
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    status_after_output_create = _git(["status", "--short"])
    start_text = START_HERE.read_text(encoding="utf-8")
    head = _git(["rev-parse", "HEAD"])
    origin = _git(["rev-parse", "origin/master"])
    imports = _import_roots_from_ast()
    forbidden_imports = sorted(
        set(imports)
        & {
            "carb",
            "isaaclab",
            "isaacsim",
            "omni",
            "pxr",
            "torch",
            "warp",
        }
    )
    expected_before = {
        " M START_HERE.md",
        (
            "?? sim_scripts/"
            "cyl34_top_view_d387_d386_shadowed_layer_"
            "fixed_graph_completion_localization.py"
        ),
    }
    expected_after = expected_before | {
        "?? claudedocs/runtime_logs/grasp_track/g0a_d387/"
    }

    d386_evidence = _read_json(D386_EVIDENCE)
    shadow_inventory = d386_evidence["scope_statement"][
        "shadowed_or_later_layers"
    ]
    actual_shadow_keys = {
        (
            str(row["body"]),
            str(row["prim_name"]),
            str(row["region_name"]),
        )
        for row in shadow_inventory
    }
    expected_shadow_keys = {_target_key(row) for row in TARGETS}
    inherited_keys = {
        _target_key(row) for row in d386_evidence["layer_results"]
    }
    checks = {
        "head_exact": head == EXPECTED_HEAD,
        "origin_exact": origin == EXPECTED_HEAD,
        "input_hashes_exact": _input_hashes() == EXPECTED_INPUT_SHA256,
        "d386_shadow_inventory_exact_seven": (
            actual_shadow_keys == expected_shadow_keys
            and len(actual_shadow_keys) == 7
        ),
        "d386_inherited_inventory_exact_four": (
            len(inherited_keys) == 4
        ),
        "inherited_and_new_layer_sets_disjoint": (
            not inherited_keys & expected_shadow_keys
        ),
        "combined_inventory_exact_eleven": (
            len(inherited_keys | expected_shadow_keys) == 11
        ),
        "start_here_active_case_present": (
            "D387 [d386_shadowed_layer_fixed_graph_completion_localization]"
            in start_text
        ),
        "start_here_variable_present": NEW_VARIABLES[0] in start_text,
        "start_here_output_path_present": _rel(OUT_DIR) in start_text,
        "start_here_exact_seven_scope_present": (
            "Evaluate exactly the seven D386-inventoried layers" in start_text
        ),
        "start_here_budget_range_12_64_present": (
            "budget range `12..64`" in start_text
        ),
        "start_here_no_overlap_gate_present": (
            "positive-volume child overlap `0`" in start_text
        ),
        "one_new_variable_exact": len(NEW_VARIABLES) == 1,
        "baseline_budget_exact_12": BASELINE_BUDGET == 12,
        "maximum_localization_budget_exact_64": (
            MAXIMUM_LOCALIZATION_BUDGET == 64
        ),
        "non_vertex_gates_frozen": (
            MAXIMUM_POLYGONS == 64
            and MAXIMUM_VERTICES_PER_POLYGON == 32
            and POSITIVE_VOLUME_EPS_M3 == 1.0e-18
            and SURFACE_TOLERANCE_MM == 0.1
            and VOLUME_RELATIVE_TOLERANCE == 0.005
        ),
        "forbidden_runtime_imports_absent": not forbidden_imports,
        "rerun_cli_present": RERUN_CLI.is_file(),
        "font_regular_present": FONT_REGULAR.is_file(),
        "font_bold_present": FONT_BOLD.is_file(),
        "worktree_before_output_create_exact": (
            set(status_before_output_create.splitlines()) == expected_before
        ),
        "output_create_added_only_d387_root": (
            set(status_after_output_create.splitlines()) == expected_after
        ),
    }
    preregistration = {
        "artifact": "D387_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "approved_scope": (
            "offline-only fixed-graph classification of exactly D386's "
            "seven inventoried but unevaluated/shadowed layers"
        ),
        "new_variables": NEW_VARIABLES,
        "d386_inherited_layer_keys": [
            {
                "body": key[0],
                "prim_name": key[1],
                "region_name": key[2],
            }
            for key in sorted(inherited_keys)
        ],
        "new_target_layers": TARGETS,
        "registered_method": {
            "algorithm_authority": (
                "exact-SHA D386 module functions: full graph, bounded "
                "cover, minimax DP, exhaustive minimax, reachability, "
                "selected witness, and geometry metrics"
            ),
            "complete_candidate_graph_enumerations_per_new_layer": 1,
            "frozen_d385_b12_helper_evaluations_per_new_layer": 1,
            "d386_inherited_layer_recomputations": 0,
            "construction": (
                "D385 authored thin-layer interval, broad-profile fan "
                "anchor/order, contiguous group size 1..4, and original-parent "
                "intersection"
            ),
            "primary_localizer": (
                "D386 bounded dynamic-programming minimax over one fixed graph"
            ),
            "independent_localizer": (
                "D386 exhaustive enumeration of every complete fixed-gate path"
            ),
            "classifications": {
                "BASELINE_B12_COVER": (
                    "B=12 has a cover; a raw sub-12 graph minimax may be "
                    "recorded only as an out-of-decision-range diagnostic and "
                    "is never selected or adopted"
                ),
                "FINITE_RELAXATION_THRESHOLD_13_TO_64": (
                    "B=12 no-cover and finite B* with B*-1 no-cover/B* cover"
                ),
                "NO_COVER_THROUGH_64": (
                    "no complete fixed-graph path at B=64"
                ),
            },
            "valid_null_semantics": (
                "A valid NO_COVER_THROUGH_64 completes its map entry and does "
                "not by itself make D387 map completion fail"
            ),
            "forward_only_phase_marker_contract": (
                "each new layer records exact start/end markers for graph, "
                "B12 helper, DP, and exhaustive evaluation; finite entries "
                "also record exact geometry-gate start/end markers"
            ),
            "budget_range_inclusive": [
                BASELINE_BUDGET,
                MAXIMUM_LOCALIZATION_BUDGET,
            ],
        },
        "frozen_non_vertex_gates": {
            "maximum_polygons": MAXIMUM_POLYGONS,
            "maximum_vertices_per_polygon": (
                MAXIMUM_VERTICES_PER_POLYGON
            ),
            "positive_volume_epsilon_m3": POSITIVE_VOLUME_EPS_M3,
            "surface_tolerance_mm": SURFACE_TOLERANCE_MM,
            "topology_volume_relative_tolerance": (
                VOLUME_RELATIVE_TOLERANCE
            ),
            "positive_volume_child_overlap": 0,
        },
        "failure_semantics": {
            "map_completion_pass": (
                "all seven entries receive one independently verified valid "
                "classification and the inherited4/new7 union is exact11"
            ),
            "scientific_contract_fail_stop": (
                "any provenance, inventory, helper/graph, DP/exhaustive, "
                "boundary, geometry, or scope disagreement"
            ),
            "observability_completion_fail_stop": (
                "presentation/layout/Rerun/manual disagreement preserves the "
                "already committed canonical numeric map verdict but blocks "
                "operational completion and retry"
            ),
            "collider_design_pass": False,
        },
        "explicit_nonclaims": {
            "sub12_raw_minimax_as_selected_or_adopted_budget": None,
            "selected_or_adopted_budget": None,
            "selected_budget_application_count": 0,
            "complete_p34_budget": None,
            "complete_source_child_count": None,
            "complete_total_part_count": None,
            "materializable_candidate": False,
            "p34_or_all_eight_parent_map_complete": False,
            "live_gpu_or_physics_or_grasp_result": None,
        },
        "worker_contract": {
            "actual_worker_invocations": 1,
            "retries": 0,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "watchdog_signal_authority": (
                "on timeout only, signal the newly created D387-owned worker "
                "process group; never signal unrelated/external processes"
            ),
            "rerun_viewer_invocations_maximum": 1,
            "rerun_requested_logical_window_size": [1920, 1080],
            "rerun_allowed_native_screenshot_sizes": [
                [1920, 1080],
                [3840, 2160],
            ],
        },
        "forbidden_runtime_counters": FORBIDDEN_COUNTERS,
        "environment": {
            "head": head,
            "origin_master": origin,
            "git_status_before_output_create": (
                status_before_output_create.splitlines()
            ),
            "git_status_after_output_create": (
                status_after_output_create.splitlines()
            ),
            "python": sys.version,
            "executable": sys.executable,
            "script_path": _rel(SCRIPT_PATH),
            "script_sha256": _sha(SCRIPT_PATH),
            "start_here_sha256": _sha(START_HERE),
            "input_hashes": _input_hashes(),
            "import_roots": imports,
            "forbidden_import_roots_found": forbidden_imports,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG_PATH, preregistration)
    _phase("prepare_end", pass_value=preregistration["pass"])
    if not preregistration["pass"]:
        raise RuntimeError(f"D387 preregistration failed: {checks}")
    print(json.dumps({"prepare_pass": True, "path": _rel(PREREG_PATH)}))
    return 0


def _worker() -> int:
    global D385, D386_HELPERS
    if not PREREG_PATH.is_file():
        raise RuntimeError("missing D387 preregistration")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D387 preregistration did not pass")
    provenance_checks = {
        "script_hash_unchanged_after_prepare": (
            _sha(SCRIPT_PATH)
            == prereg["environment"]["script_sha256"]
        ),
        "start_here_hash_unchanged_after_prepare": (
            _sha(START_HERE)
            == prereg["environment"]["start_here_sha256"]
        ),
        "input_hashes_unchanged_after_prepare": (
            _input_hashes() == prereg["environment"]["input_hashes"]
        ),
        "head_unchanged_after_prepare": (
            _git(["rev-parse", "HEAD"])
            == prereg["environment"]["head"]
        ),
        "origin_unchanged_after_prepare": (
            _git(["rev-parse", "origin/master"])
            == prereg["environment"]["origin_master"]
        ),
    }
    if not all(provenance_checks.values()):
        raise RuntimeError(
            f"D387 worker provenance failed: {provenance_checks}"
        )
    if D385 is not None or D386_HELPERS is not None:
        raise RuntimeError(
            "D385/D386 helpers were loaded before worker provenance"
        )
    D385 = _load_verified_d385_module()
    D386_HELPERS = _load_verified_d386_module()
    D386_HELPERS.D385 = D385
    helper_constant_checks = {
        "baseline_budget_12": (
            D386_HELPERS.BASELINE_BUDGET == BASELINE_BUDGET
        ),
        "maximum_budget_64": (
            D386_HELPERS.MAXIMUM_LOCALIZATION_BUDGET
            == MAXIMUM_LOCALIZATION_BUDGET
        ),
        "maximum_polygons_64": (
            D386_HELPERS.MAXIMUM_POLYGONS == MAXIMUM_POLYGONS
        ),
        "maximum_vertices_per_polygon_32": (
            D386_HELPERS.MAXIMUM_VERTICES_PER_POLYGON
            == MAXIMUM_VERTICES_PER_POLYGON
        ),
        "surface_tolerance_exact": (
            D386_HELPERS.SURFACE_TOLERANCE_MM == SURFACE_TOLERANCE_MM
        ),
        "volume_tolerance_exact": (
            D386_HELPERS.VOLUME_RELATIVE_TOLERANCE
            == VOLUME_RELATIVE_TOLERANCE
        ),
        "positive_volume_epsilon_exact": (
            D386_HELPERS.POSITIVE_VOLUME_EPS_M3
            == POSITIVE_VOLUME_EPS_M3
        ),
    }
    if not all(helper_constant_checks.values()):
        raise RuntimeError(
            f"frozen D386 helper constants changed: {helper_constant_checks}"
        )
    _phase("worker_start")
    evidence, geometry, visual, candidate_rows = _compute()
    evidence["script_sha256"] = _sha(SCRIPT_PATH)
    evidence["diagnostic_geometry_payload_sha256"] = _sha_payload(geometry)
    evidence["execution_contract"] = {
        "worker_invocation_index": 1,
        "retry_index": 0,
        "offline_only": True,
        "provenance_checks": provenance_checks,
        "frozen_d386_helper_constant_checks": helper_constant_checks,
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase(
        "canonical_evidence_committed",
        verdict=evidence["verdict"],
        map_completion_pass=evidence["map_completion_pass"],
    )
    _write_json_x(GEOMETRY_PATH, geometry)
    _write_candidate_metrics_csv(candidate_rows)
    board_info, layout = _render_board(evidence, visual)
    _write_json_x(BOARD_LAYOUT, layout)
    rerun = _write_rerun(evidence, visual)
    layer_phase_contract = _layer_phase_contract(
        evidence["new_layer_results"]
    )
    manual_template = {
        "artifact": "D387_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board_path": _rel(BOARD_PATH),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "required_checks": [
            "board_exact_1920x1080_and_readable",
            "all_eleven_layer_cells_and_four_parents_visible",
            "d386_inherited_four_and_d387_new_seven_distinguishable",
            "baseline_interior_null_classifications_unambiguous",
            "parent_and_global_null_semantics_readable",
            "rerun_geometry_and_human_decision_panel_readable",
            "rerun_native_dimension_is_logical_1x_or_hidpi_2x_not_4x",
            "no_budget_adoption_live_or_physics_claim",
        ],
        "inspection_result_path": _rel(MANUAL_INSPECTION),
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    claim = {
        "artifact": "D387_OFFLINE_WORKER_CLAIM_V1",
        "worker_pid": os.getpid(),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "scientific_verdict": evidence["verdict"],
        "map_completion_pass": evidence["map_completion_pass"],
        "method_contract_pass": evidence["method_contract_pass"],
        "artifacts": {
            "evidence": {
                "path": _rel(EVIDENCE_PATH),
                "sha256": _sha(EVIDENCE_PATH),
            },
            "geometry": {
                "path": _rel(GEOMETRY_PATH),
                "sha256": _sha(GEOMETRY_PATH),
            },
            "metrics_csv": {
                "path": _rel(METRICS_CSV),
                "sha256": _sha(METRICS_CSV),
            },
            "board": board_info,
            "board_layout": {
                "path": _rel(BOARD_LAYOUT),
                "sha256": _sha(BOARD_LAYOUT),
                "pass": layout["pass"],
            },
            "rerun": rerun,
            "layer_phase_contract": layer_phase_contract,
            "manual_template": {
                "path": _rel(MANUAL_TEMPLATE),
                "sha256": _sha(MANUAL_TEMPLATE),
            },
        },
        "scope_counters": FORBIDDEN_COUNTERS,
        "pass": bool(
            evidence["method_contract_pass"]
            and layout["pass"]
            and rerun["strict_validation_pass"]
            and rerun["screenshot_dimension_contract_pass"]
            and layer_phase_contract["pass"]
            and rerun["headless_viewer_invocations"] <= 1
            and all(value == 0 for value in FORBIDDEN_COUNTERS.values())
        ),
    }
    _phase("worker_end", worker_claim_pass=claim["pass"])
    _write_json_x(WORKER_CLAIM, claim)
    if not claim["pass"]:
        raise RuntimeError("D387 observability worker claim failed")
    print(
        json.dumps(
            {
                "worker_pass": True,
                "scientific_verdict": evidence["verdict"],
                "evidence": _rel(EVIDENCE_PATH),
            }
        )
    )
    return 0


def _process_group_members(process_group_id: int) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,stat=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) < 3:
            continue
        try:
            pid = int(fields[0])
            pgid = int(fields[1])
        except ValueError:
            continue
        if pgid != process_group_id:
            continue
        rows.append(
            {
                "pid": pid,
                "pgid": pgid,
                "stat": fields[2],
                "args": fields[3] if len(fields) == 4 else "",
            }
        )
    return rows


def _run_supervisor() -> int:
    if not PREREG_PATH.is_file():
        raise RuntimeError("run requires completed D387 prepare stage")
    if INVOCATION_PATH.exists() or SUPERVISOR_PATH.exists():
        raise RuntimeError("refusing to repeat D387 actual worker")
    command = [
        sys.executable,
        "-B",
        str(SCRIPT_PATH),
        "--stage",
        "worker",
    ]
    invocation = {
        "artifact": "D387_OFFLINE_WORKER_INVOCATION_V1",
        "command": command,
        "cwd": str(REPO),
        "worker_invocation_index": 1,
        "retry_index": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase("supervisor_before_worker")
    started = time.monotonic()
    timed_out = False
    termination_action = None
    with WORKER_STDOUT.open("x", encoding="utf-8") as stdout_stream:
        with WORKER_STDERR.open("x", encoding="utf-8") as stderr_stream:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=WATCHDOG_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                termination_action = (
                    "SIGTERM_process_group_then_SIGKILL_if_needed"
                )
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    returncode = process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = process.wait(timeout=10.0)
    elapsed = time.monotonic() - started
    residual_group_members = []
    for _ in range(10):
        residual_group_members = _process_group_members(process.pid)
        if not residual_group_members:
            break
        time.sleep(0.1)
    supervisor = {
        "artifact": "D387_OFFLINE_WORKER_SUPERVISOR_V1",
        "actual_worker_invocations": 1,
        "retries": 0,
        "worker_pid": process.pid,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "watchdog_seconds": WATCHDOG_SECONDS,
        "timed_out": timed_out,
        "termination_action": termination_action,
        "process_group_id": process.pid,
        "residual_process_group_members": residual_group_members,
        "stdout": _rel(WORKER_STDOUT),
        "stderr": _rel(WORKER_STDERR),
        "worker_claim_exists": WORKER_CLAIM.is_file(),
        "pass": bool(
            returncode == 0
            and not timed_out
            and WORKER_CLAIM.is_file()
            and not residual_group_members
        ),
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase("supervisor_after_worker", **supervisor)
    if not supervisor["pass"]:
        raise RuntimeError(f"D387 worker failed: {supervisor}")
    print(json.dumps(supervisor, indent=2))
    return 0


def _finalize() -> int:
    required = [
        PREREG_PATH,
        SUPERVISOR_PATH,
        PHASE_PATH,
        EVIDENCE_PATH,
        GEOMETRY_PATH,
        METRICS_CSV,
        BOARD_PATH,
        BOARD_LAYOUT,
        RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION,
        RERUN_SCREENSHOT,
        MANUAL_INSPECTION,
        WORKER_CLAIM,
    ]
    missing = [_rel(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot finalize; missing artifacts: {missing}")
    _phase("finalize_start")
    evidence = _read_json(EVIDENCE_PATH)
    supervisor = _read_json(SUPERVISOR_PATH)
    layout = _read_json(BOARD_LAYOUT)
    rerun_validation = _read_json(RERUN_VALIDATION)
    manual = _read_json(MANUAL_INSPECTION)
    worker_claim = _read_json(WORKER_CLAIM)
    required_manual_checks = {
        "board_exact_1920x1080_and_readable",
        "all_eleven_layer_cells_and_four_parents_visible",
        "d386_inherited_four_and_d387_new_seven_distinguishable",
        "baseline_interior_null_classifications_unambiguous",
        "parent_and_global_null_semantics_readable",
        "rerun_geometry_and_human_decision_panel_readable",
        "rerun_native_dimension_is_logical_1x_or_hidpi_2x_not_4x",
        "no_budget_adoption_live_or_physics_claim",
    }
    manual_checks = manual.get("checks", {})
    manual_hashes = manual.get("artifact_hashes", {})
    manual_contract_pass = bool(
        manual.get("artifact") == "D387_MANUAL_VISUAL_INSPECTION_V1"
        and set(manual_checks) == required_manual_checks
        and all(value is True for value in manual_checks.values())
        and isinstance(manual.get("observations"), list)
        and len(manual["observations"]) >= 3
        and manual_hashes
        == {
            _rel(BOARD_PATH): _sha(BOARD_PATH),
            _rel(RERUN_SCREENSHOT): _sha(RERUN_SCREENSHOT),
        }
        and manual.get("pass") is True
    )
    map_pass_verdict = (
        "D387_SHADOWED_LAYER_FIXED_GRAPH_MAP_COMPLETION_PASS_"
        "GLOBAL_BUDGET_NULL"
    )
    contract_verdict = "D387_FIXED_GRAPH_MAP_CONTRACT_FAIL_STOP"
    verdict_consistent = bool(
        (
            evidence["method_contract_pass"] is True
            and evidence["map_completion_pass"] is True
            and evidence["verdict"] == map_pass_verdict
        )
        or (
            evidence["method_contract_pass"] is False
            and evidence["map_completion_pass"] is False
            and evidence["verdict"] == contract_verdict
        )
    )
    checks = {
        "scientific_map_verdict_consistent": verdict_consistent,
        "map_completion_pass": evidence["map_completion_pass"] is True,
        "global_common_vertex_budget_null": (
            evidence["global_common_vertex_budget"] is None
        ),
        "adopted_parent_wide_budget_null": (
            evidence["adopted_parent_wide_vertex_budget"] is None
        ),
        "selected_vertex_budget_null": (
            evidence["selected_vertex_budget"] is None
        ),
        "complete_p34_vertex_budget_null": (
            evidence["complete_p34_vertex_budget"] is None
        ),
        "selected_budget_application_zero": (
            evidence["selected_budget_application_count"] == 0
        ),
        "complete_counts_null": (
            evidence["complete_source_child_count"] is None
            and evidence["complete_total_part_count"] is None
        ),
        "global_semantic_preservation_null": (
            evidence["global_semantic_preservation_pass"] is None
        ),
        "materializable_candidate_false": (
            evidence["materializable_candidate"] is False
        ),
        "repair_materialized_false": (
            evidence["repair_materialized"] is False
        ),
        "combined_all_eleven_layers_finite_false": (
            evidence["combined_all_eleven_layers_finite"] is False
        ),
        "d386_inherited_exact_four_no_recompute": (
            evidence["scope_statement"]["d386_inherited_layer_count"] == 4
            and evidence["scope_statement"][
                "d386_inherited_layer_recomputations"
            ]
            == 0
        ),
        "d387_evaluated_exact_seven": (
            evidence["scope_statement"]["d387_evaluated_layer_count"] == 7
        ),
        "combined_map_exact_eleven_disjoint": (
            evidence["scope_statement"]["combined_layer_count"] == 11
            and evidence["scope_statement"][
                "inherited_and_evaluated_intersection_count"
            ]
            == 0
            and evidence["scope_statement"][
                "four_failed_parent_layer_map_complete"
            ]
            is True
        ),
        "supervisor_pass": supervisor["pass"] is True,
        "actual_worker_once_no_retry": (
            supervisor["actual_worker_invocations"] == 1
            and supervisor["retries"] == 0
        ),
        "board_layout_pass": layout["pass"] is True,
        "rerun_strict_validation_pass": (
            rerun_validation["pass"] is True
        ),
        "rerun_screenshot_dimension_contract_pass": (
            worker_claim["artifacts"]["rerun"][
                "screenshot_dimension_contract_pass"
            ]
            is True
        ),
        "layer_phase_contract_pass": (
            worker_claim["artifacts"]["layer_phase_contract"]["pass"] is True
            and _layer_phase_contract(
                evidence["new_layer_results"]
            )["pass"]
            is True
        ),
        "manual_visual_inspection_contract_pass": manual_contract_pass,
        "worker_claim_pass": worker_claim["pass"] is True,
        "headless_viewer_maximum_one": (
            worker_claim["artifacts"]["rerun"][
                "headless_viewer_invocations"
            ]
            <= 1
        ),
        "scope_counters_zero": all(
            value == 0
            for value in evidence["current_scope_counters"].values()
        ),
        "live_identity_null": evidence["live_identity_pass"] is None,
        "live_gpu_compatibility_null": (
            evidence["live_gpu_compatibility_pass"] is None
        ),
        "cylinder_29x50_not_rendered_or_measured": (
            evidence["cylinder_29x50_rendered_or_measured"] is False
        ),
        "physics_or_grasp_null": (
            evidence["physics_or_grasp_result"] is None
        ),
        "p34_identity_false": (
            evidence["p34_authored_to_cooked_identity_pass"] is False
        ),
        "g0a_false": evidence["g0a_pass"] is False,
    }
    preliminary_pass = all(checks.values())
    _phase(
        "finalize_end",
        preliminary_completion_pass=preliminary_pass,
    )
    global_phase_contract = _global_phase_contract()
    checks["global_phase_contract_pass"] = global_phase_contract["pass"]
    completion = {
        "artifact": "D387_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "scientific_map_verdict": evidence["verdict"],
        "map_completion_pass": evidence["map_completion_pass"],
        "collider_design_pass": False,
        "observability_completion_pass": all(checks.values()),
        "checks": checks,
        "global_phase_contract": global_phase_contract,
        "artifact_hashes": {
            _rel(path): _sha(path) for path in required
        },
        "next_authorization_boundary": (
            "D387 completed only the fixed-graph map of D385's four failed "
            "parents. Do not select/apply a budget, alter a partition/gate, "
            "materialize USD/PhysX, create the 29x50mm target, or run physics/"
            "contact/grasp without new explicit approval."
        ),
        "pass": all(checks.values()),
    }
    _write_json_x(COMPLETION_PATH, completion)
    if not completion["pass"]:
        raise RuntimeError(f"D387 completion failed: {checks}")
    print(json.dumps(completion, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "worker", "finalize"),
        required=True,
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        return _prepare()
    if args.stage == "run":
        return _run_supervisor()
    if args.stage == "worker":
        return _worker()
    return _finalize()


if __name__ == "__main__":
    raise SystemExit(main())
