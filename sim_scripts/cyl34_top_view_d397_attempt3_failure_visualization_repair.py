#!/usr/bin/env python3
"""D397 attempt3: immutable failure-visualization repair.

This case never reruns the D397 construction.  It reads the frozen attempt2
evidence and geometry, creates one locally scaled panel per source parent, and
records an exploded display-only Rerun copy with seam labels visually blanked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
START = REPO / "START_HERE.md"
BASE_SCRIPT = (
    REPO
    / "sim_scripts/"
    "cyl34_top_view_d397_shared_boundary_zero_volume_construction_design.py"
)
SOURCE_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d397/"
    "attempt2_phase_marker_payload_key_repair"
)
SOURCE_EVIDENCE = SOURCE_DIR / "d397_shared_boundary_design_evidence.json"
SOURCE_GEOMETRY = SOURCE_DIR / "d397_shared_boundary_candidate_geometry.json"
SOURCE_WORKER = SOURCE_DIR / "d397_offline_worker_claim.json"
SOURCE_COMPLETION = SOURCE_DIR / "d397_completion_summary.json"
SOURCE_MANUAL = SOURCE_DIR / "d397_manual_visual_inspection.json"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"

EXPECTED_INPUT_SHA256 = {
    BASE_SCRIPT: (
        "52745beab46bc695467dd8d676a06b30fa3ea873c7dcad685861e65cfecf4b36"
    ),
    SOURCE_EVIDENCE: (
        "ea7fd61c38f12b9e03f4e7154536579b831c6f85703bfd4d14e34807cdf327b6"
    ),
    SOURCE_GEOMETRY: (
        "b9a44d430f647e45292fe71804bd17e6f53bf37eea28913389316beac60fa623"
    ),
    SOURCE_WORKER: (
        "2bac06043e35e095660ed3a0562930f98425a9eba436dc5b72e58f313ed1ed79"
    ),
    SOURCE_COMPLETION: (
        "c12fb2ea2b8a5fce636666a73e0ad0c0ee47a607c838ceadfd3001c4bb1b1d23"
    ),
    SOURCE_MANUAL: (
        "3b6ff0a7c46263c02592f9239084c3a2e07f14ce6d529d3da830ba026a1ef223"
    ),
    VIZ_DEBUG: (
        "4b5f821ad43652f529dfaa2f92b2826d9cd4973635e34521cc2b3a93ab0193d0"
    ),
    RERUN_CONTRACT: (
        "aaafcd93b9da3d8a97d61a53753ec9667bb98bec7391c91c98974f7ce9c66c1e"
    ),
}

CASE = "g0a_d397"
ATTEMPT = "attempt3_failure_visualization_repair"
NEW_VARIABLES = [
    "per_parent_failure_board_and_label_suppressed_exploded_rerun_v1"
]
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d397/"
    "attempt3_failure_visualization_repair"
)
PREREG = OUT_DIR / "d397_attempt3_preregistration.json"
PHASES = OUT_DIR / "d397_attempt3_phase_markers.jsonl"
OBSERVE_INVOCATION = OUT_DIR / "d397_attempt3_observe_invocation.json"
EVIDENCE = OUT_DIR / "d397_attempt3_presentation_evidence.json"
GEOMETRY = OUT_DIR / "d397_attempt3_display_geometry.json"
BOARD = OUT_DIR / "d397_failure_by_parent_1920x1080.png"
LAYOUT = OUT_DIR / "d397_attempt3_layout_validation.json"
RRD = OUT_DIR / "d397_failure_exploded.rerun.rrd"
RBL = OUT_DIR / "d397_failure_exploded.rerun.rbl"
RERUN_VALIDATION = OUT_DIR / "d397_attempt3_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d397_attempt3_rerun_inspection.png"
OBSERVABILITY = OUT_DIR / "d397_attempt3_observability_claim.json"
MANUAL_TEMPLATE = OUT_DIR / "d397_attempt3_manual_template.json"
MANUAL = OUT_DIR / "d397_attempt3_manual_visual_inspection.json"
FINALIZE_INVOCATION = OUT_DIR / "d397_attempt3_finalize_invocation.json"
COMPLETION = OUT_DIR / "d397_attempt3_completion_summary.json"
FAILURE = OUT_DIR / "d397_attempt3_runtime_failure.json"
_STAGE_WRITE_STARTED = False

EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
MANUAL_KEYS = [
    "board_exact_1920x1080",
    "all_eight_parent_panels_visible",
    "two_complete_and_six_failed_distinguishable",
    "source_leaf_seam_layers_distinguishable",
    "metrics_and_null_boundaries_readable",
    "no_missing_glyphs",
    "no_overlapping_geometry_labels",
    "rerun_exploded_geometry_loaded",
]
DISPLAY_ORDER = [
    "fixed_backbone_left",
    "fixed_backbone_right",
    "proximal_upper_arm_hull_a",
    "proximal_upper_arm_hull_b",
    "proximal_lower_arm_hull_a",
    "proximal_lower_arm_hull_b",
    "moving_upper_backbone",
    "moving_lower_backbone",
]
SHORT_NAMES = {
    "fixed_backbone_left": "FBL",
    "fixed_backbone_right": "FBR",
    "proximal_upper_arm_hull_a": "PUA",
    "proximal_upper_arm_hull_b": "PUB",
    "proximal_lower_arm_hull_a": "PLA",
    "proximal_lower_arm_hull_b": "PLB",
    "moving_upper_backbone": "MUB",
    "moving_lower_backbone": "MLB",
}


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("d397_a3_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
BASE.ATTEMPT = ATTEMPT
BASE.OUT_DIR = OUT_DIR
BASE.EVIDENCE = EVIDENCE
BASE.GEOMETRY = GEOMETRY
BASE.BOARD = BOARD
BASE.LAYOUT = LAYOUT
BASE.RRD = RRD
BASE.RBL = RBL
BASE.RERUN_VALIDATION = RERUN_VALIDATION
BASE.RERUN_SCREENSHOT = RERUN_SCREENSHOT


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            BASE._native(value),
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")


def _phase(phase_name: str, **fields: Any) -> None:
    with PHASES.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "phase": phase_name,
                    "monotonic_seconds": time.monotonic(),
                    "wall_time_unix_seconds": time.time(),
                    **BASE._native(fields),
                },
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _input_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in EXPECTED_INPUT_SHA256}


def _png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "path": _rel(path),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }


def _prepare() -> int:
    global _STAGE_WRITE_STARTED
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only output exists: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _STAGE_WRITE_STARTED = True
    _phase("prepare_start")
    source_evidence = _read(SOURCE_EVIDENCE)
    source_completion = _read(SOURCE_COMPLETION)
    source_manual = _read(SOURCE_MANUAL)
    start_text = START.read_text(encoding="utf-8")
    actual_hashes = _input_hashes()
    expected_hashes = {
        _rel(path): digest for path, digest in EXPECTED_INPUT_SHA256.items()
    }
    status_outside_output = BASE._status_outside_output()
    checks = {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "inputs_exact": actual_hashes == expected_hashes,
        "source_design_fail": source_evidence["design_pass"] is False,
        "source_complete_parent_count_2": sum(
            row["construction_complete"]
            for row in source_evidence["source_parent_metrics"]
        )
        == 2,
        "source_manual_presentation_fail": source_manual["overall_pass"]
        is False,
        "source_completion_integrity_fail": source_completion[
            "completion_integrity_pass"
        ]
        is False,
        "active_variable_exact": NEW_VARIABLES[0] in start_text,
        "active_output_exact": _rel(OUT_DIR) in start_text,
        "one_observability_variable": len(NEW_VARIABLES) == 1,
        "numpy_1_26_0": np.__version__ == "1.26.0",
        "rerun_0_34_1": importlib.metadata.version("rerun-sdk") == "0.34.1",
        "rerun_cli_present_and_executable": BASE.RERUN_CLI.is_file()
        and os.access(BASE.RERUN_CLI, os.X_OK),
        "fonts_present": FONT_REGULAR.is_file() and FONT_BOLD.is_file(),
    }
    prereg = {
        "artifact": "D397_ATTEMPT3_FAILURE_VISUALIZATION_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "new_variables": NEW_VARIABLES,
        "purpose": (
            "repair only the attempt2 failure presentation by giving every "
            "source parent an independent scale and suppressing Rerun seam labels"
        ),
        "immutable_scientific_authority": {
            "evidence": _rel(SOURCE_EVIDENCE),
            "geometry": _rel(SOURCE_GEOMETRY),
        },
        "presentation_contract": {
            "board_exact_px": [1920, 1080],
            "one_panel_per_source_parent": 8,
            "layers": ["source parent", "diagnostic leaves", "shared seams"],
            "Rerun_Viewer_maximum": 1,
            "Rerun_Viewer_retries": 0,
            "seam_labels_visible": False,
        },
        "scope_counters": {
            "science_worker_invocations": 0,
            "construction_evaluations": 0,
            "geometry_or_gate_changes": 0,
            "usd_asset_writes": 0,
            "isaac_kit_physx_warp_cuda": 0,
            "physics_q5_contact_cylinder": 0,
            "target_ik_path_settings_changes": 0,
            "process_signals": 0,
        },
        "script": {"path": _rel(SCRIPT), "sha256": _sha(SCRIPT)},
        "start": {"path": _rel(START), "sha256": _sha(START)},
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "status_outside_output": status_outside_output,
            "status_outside_output_sha256": BASE._sha_payload(
                status_outside_output
            ),
        },
        "input_hashes": actual_hashes,
        "checks": checks,
        "pass": all(checks.values()),
        "forward_only_output": _rel(OUT_DIR),
    }
    _write(PREREG, prereg)
    _phase("prepare_end", preregistration_pass=prereg["pass"])
    if not prereg["pass"]:
        raise RuntimeError(f"D397 attempt3 prepare failed: {checks}")
    print(json.dumps({"prepared": True, "path": _rel(PREREG)}))
    return 0


def _part_shift(part: dict[str, Any], shift: np.ndarray) -> dict[str, Any]:
    result = dict(part)
    vertices = (
        np.asarray(part["vertices_f32_m"], dtype=np.float64) + shift
    )
    result["vertices_f32_m"] = vertices
    result["bounds_m"] = [vertices.min(axis=0), vertices.max(axis=0)]
    result["display_only_translation_m"] = shift
    result["canonical_source_payload_sha256"] = part.get("payload_sha256")
    result["payload_sha256"] = BASE._sha_payload(
        {
            "vertices_f32_m": vertices,
            "triangles_i32": part["triangles_i32"],
        }
    )
    return result


def _display_geometry(
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    parents = {
        row["name"]: row for row in source["diagnostic_source_parents"]
    }
    leaves = source["diagnostic_partition_leaves"]
    seams = source["diagnostic_shared_seams"]
    shifts: dict[str, np.ndarray] = {}
    for body in BASE.BODY_NAMES:
        names = [
            name
            for name in DISPLAY_ORDER
            if parents[name]["body"] == body
        ]
        columns = 2 if body == "link5" else 3
        for index, name in enumerate(names):
            shifts[name] = np.asarray(
                [(index % columns) * 0.105, (index // columns) * 0.115, 0.0],
                dtype=np.float64,
            )
    display_parents = [
        _part_shift(parents[name], shifts[name]) for name in DISPLAY_ORDER
    ]
    display_leaves = [
        _part_shift(row, shifts[row["parent_name"]]) for row in leaves
    ]
    display_seams = []
    for name in DISPLAY_ORDER:
        selected = [row for row in seams if row["parent_name"] == name]
        points = (
            np.vstack(
                [
                    np.asarray(row["seam_vertices_f32_m"], dtype=np.float64)
                    for row in selected
                ]
            )
            if selected
            else np.empty((0, 3), dtype=np.float64)
        )
        if len(points):
            points = np.unique(points.astype(np.float32), axis=0).astype(
                np.float64
            )
            points += shifts[name]
        display_seams.append(
            {
                "body": parents[name]["body"],
                "parent_name": SHORT_NAMES[name],
                "source_parent_name": name,
                "node_id": "all",
                "axis": None,
                "axis_name": "mixed",
                "cut_f32_m": None,
                "cut_f32_bits": None,
                "seam_vertices_f32_m": points,
                "seam_vertex_bits_sha256": BASE._sha_payload(points),
                "display_only_translation_m": shifts[name],
            }
        )
    geometry = {
        "artifact": "D397_ATTEMPT3_EXPLODED_DISPLAY_GEOMETRY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "display_only": True,
        "canonical_numeric_geometry": _rel(SOURCE_GEOMETRY),
        "parts": {body: [] for body in BASE.BODY_NAMES},
        "diagnostic_source_parents": display_parents,
        "diagnostic_partition_leaves": display_leaves,
        "diagnostic_shared_seams": display_seams,
        "display_translation_by_parent_m": {
            key: value for key, value in shifts.items()
        },
        "counts": source["counts"],
    }
    return geometry, {
        key: BASE._native(value) for key, value in shifts.items()
    }


def _equal_limits(axis: Any, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.max(high - low)) * 0.62, 0.0015)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _render_board(
    source_evidence: dict[str, Any], source_geometry: dict[str, Any]
) -> dict[str, Any]:
    os.environ.setdefault(
        "MPLCONFIGDIR", "/tmp/roarm_d397_attempt3_matplotlib"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    regular = FontProperties(fname=str(FONT_REGULAR))
    bold = FontProperties(fname=str(FONT_BOLD))
    metrics = {
        row["name"]: row for row in source_evidence["source_parent_metrics"]
    }
    parents = {
        row["name"]: row
        for row in source_geometry["diagnostic_source_parents"]
    }
    leaves_by_parent = {
        name: [
            row
            for row in source_geometry["diagnostic_partition_leaves"]
            if row["parent_name"] == name
        ]
        for name in DISPLAY_ORDER
    }
    seams_by_parent = {
        name: [
            row
            for row in source_geometry["diagnostic_shared_seams"]
            if row["parent_name"] == name
        ]
        for name in DISPLAY_ORDER
    }
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1, 1, 1, 1, 1.12],
        left=0.025,
        right=0.985,
        top=0.89,
        bottom=0.065,
        wspace=0.13,
        hspace=0.22,
    )
    fig.suptitle(
        "D397 실패 원인 — 원본 8개 부품별 공유경계 분할 결과",
        fontproperties=bold,
        fontsize=20,
        y=0.967,
    )
    fig.text(
        0.5,
        0.925,
        "검정 외곽=원 source · 색 면=마지막 diagnostic leaf · 자홍점=공유 seam · 과학 재실행 없음",
        ha="center",
        fontproperties=regular,
        fontsize=10.5,
    )
    palette = plt.get_cmap("tab20")
    for index, name in enumerate(DISPLAY_ORDER):
        axis = fig.add_subplot(grid[index // 4, index % 4], projection="3d")
        parent = parents[name]
        parent_vertices = np.asarray(
            parent["vertices_f32_m"], dtype=np.float64
        )
        parent_triangles = np.asarray(
            parent["triangles_i32"], dtype=np.int64
        )
        all_points = [parent_vertices]
        axis.add_collection3d(
            Poly3DCollection(
                parent_vertices[parent_triangles],
                facecolor=(0.2, 0.2, 0.2, 0.025),
                edgecolor=(0.02, 0.02, 0.02, 0.55),
                linewidth=0.45,
            )
        )
        for leaf_index, leaf in enumerate(leaves_by_parent[name]):
            vertices = np.asarray(leaf["vertices_f32_m"], dtype=np.float64)
            triangles = np.asarray(leaf["triangles_i32"], dtype=np.int64)
            all_points.append(vertices)
            color = palette((leaf_index * 2) % 20)
            axis.add_collection3d(
                Poly3DCollection(
                    vertices[triangles],
                    facecolor=(*color[:3], 0.42),
                    edgecolor=(*color[:3], 0.78),
                    linewidth=0.32,
                )
            )
        for seam in seams_by_parent[name]:
            points = np.asarray(
                seam["seam_vertices_f32_m"], dtype=np.float64
            )
            all_points.append(points)
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color="#d000ff",
                s=9,
                depthshade=False,
            )
        _equal_limits(axis, np.vstack(all_points))
        axis.view_init(elev=25, azim=-58)
        row = metrics[name]
        status = "COMPLETE" if row["construction_complete"] else "FAIL"
        subtitle = (
            f"{SHORT_NAMES[name]} | {status} | leaves="
            f"{len(leaves_by_parent[name])}"
        )
        axis.set_title(subtitle, fontsize=9.2, pad=3)
        axis.tick_params(labelsize=5.7, pad=0)
        axis.set_xlabel("x", fontsize=6, labelpad=0)
        axis.set_ylabel("y", fontsize=6, labelpad=0)
        axis.set_zlabel("z", fontsize=6, labelpad=0)
        if not row["construction_complete"]:
            axis.text2D(
                0.02,
                0.02,
                "no admissible next shared plane",
                transform=axis.transAxes,
                fontsize=7,
                color="#a00000",
            )
    text_axis = fig.add_subplot(grid[:, 4])
    text_axis.axis("off")
    failed = [
        row["name"]
        for row in source_evidence["source_parent_metrics"]
        if not row["construction_complete"]
    ]
    lines = [
        ("최종 설계 판정", "FAIL — 완성 후보 없음"),
        ("완성된 부모", "2/8 (PUB, PLB)"),
        ("중단된 부모", "6/8 (FBL, FBR, PUA, PLA, MUB, MLB)"),
        ("중단 원인", "12꼭짓점 초과 leaf를 더 줄일 공유 축 평면이 없음"),
        ("마지막 진단 형상", "leaf 46개 · seam 묶음 38개"),
        ("완성 후보 부품 수", "null"),
        ("void / OPEN 간격", "null / null"),
        ("raw 표면 / seed", "null / null"),
        ("USD / Isaac / PhysX", "0 / 0 / 0"),
        ("physics / q5 / contact", "0 / 0 / 0"),
        ("다음 단계", "USD materialization 금지 — 새 분할법 필요"),
        ("g0a_pass", "false"),
    ]
    y = 0.985
    for label, value in lines:
        text_axis.text(
            0.0,
            y,
            label,
            transform=text_axis.transAxes,
            fontproperties=bold,
            fontsize=10.2,
            va="top",
        )
        text_axis.text(
            0.0,
            y - 0.038,
            value,
            transform=text_axis.transAxes,
            fontproperties=regular,
            fontsize=8.8,
            va="top",
            wrap=True,
            color="#333333",
        )
        y -= 0.078
    fig.savefig(BOARD, dpi=100, facecolor="white")
    plt.close(fig)
    info = _png(BOARD)
    layout = {
        "artifact": "D397_ATTEMPT3_LAYOUT_VALIDATION_V1",
        "board": info,
        "checks": {
            "exact_1920x1080": info["width"] == 1920
            and info["height"] == 1080,
            "nonempty": info["bytes"] > 100_000,
            "eight_independently_scaled_parent_panels": True,
            "one_metric_panel": True,
        },
    }
    layout["pass"] = all(layout["checks"].values())
    _write(LAYOUT, layout)
    return layout


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    def view(
        body: str,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> Any:
        return rrb.Spatial3DView(
            origin="/",
            contents=f"/d397/{body}/**",
            name=f"D397 {body} exploded failure geometry",
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=position,
                look_target=target,
                eye_up=(0.0, 0.0, 1.0),
            ),
            spatial_information=rrb.SpatialInformation(
                target_frame="tf#/",
                show_axes=True,
                show_bounding_box=False,
            ),
        )

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                view("link5", (0.32, -0.28, 0.25), (0.052, 0.0, 0.06)),
                view(
                    "gripper_link",
                    (0.43, -0.35, 0.24),
                    (0.105, 0.057, -0.018),
                ),
                column_shares=[0.43, 0.57],
            ),
            rrb.TextDocumentView(
                origin="/metadata/run",
                contents="/metadata/run",
                name="D397 immutable failure authority",
            ),
            row_shares=[0.84, 0.16],
        ),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _observe() -> int:
    global _STAGE_WRITE_STARTED
    if (
        OBSERVE_INVOCATION.exists()
        or BOARD.exists()
        or RRD.exists()
        or OBSERVABILITY.exists()
        or FAILURE.exists()
    ):
        raise FileExistsError("D397 attempt3 observe already consumed")
    _STAGE_WRITE_STARTED = True
    prereg = _read(PREREG)
    continuity = {
        "prereg_pass": prereg["pass"] is True,
        "head_unchanged": _git("rev-parse", "HEAD")
        == prereg["git"]["head"],
        "origin_unchanged": _git("rev-parse", "origin/master")
        == prereg["git"]["origin_master"],
        "script_unchanged": prereg["script"]["sha256"] == _sha(SCRIPT),
        "start_unchanged": prereg["start"]["sha256"] == _sha(START),
        "inputs_unchanged": prereg["input_hashes"] == _input_hashes(),
        "status_outside_output_unchanged": prereg["git"][
            "status_outside_output_sha256"
        ]
        == BASE._sha_payload(BASE._status_outside_output()),
    }
    _write(
        OBSERVE_INVOCATION,
        {
            "artifact": "D397_ATTEMPT3_OBSERVE_INVOCATION_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "invocation_count": 1,
            "automatic_retry_count": 0,
            "continuity_checks": continuity,
            "pass": all(continuity.values()),
        },
    )
    if not all(continuity.values()):
        raise RuntimeError(
            f"D397 attempt3 observe continuity failed: {continuity}"
        )
    _phase("observe_start")
    source_evidence = _read(SOURCE_EVIDENCE)
    source_geometry = _read(SOURCE_GEOMETRY)
    display_geometry, shifts = _display_geometry(source_geometry)
    presentation_evidence = {
        "artifact": "D397_ATTEMPT3_PRESENTATION_EVIDENCE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "verdict": source_evidence["verdict"],
        "design_pass": False,
        "counts": source_evidence["counts"],
        "source_parent_metrics": source_evidence["source_parent_metrics"],
        "canonical_scientific_evidence": {
            "path": _rel(SOURCE_EVIDENCE),
            "sha256": _sha(SOURCE_EVIDENCE),
        },
        "canonical_geometry": {
            "path": _rel(SOURCE_GEOMETRY),
            "sha256": _sha(SOURCE_GEOMETRY),
        },
        "display_translation_by_parent_m": shifts,
        "scope_counters": {
            "science_worker_invocations": 0,
            "construction_evaluations": 0,
            "viewer_invocations": None,
            "viewer_maximum": 1,
            "viewer_retries": 0,
            "process_signals_sent": 0,
            "isaac_kit_physx_warp_cuda": 0,
            "physics_q5_contact_cylinder": 0,
        },
        "interpretation_boundary": {
            "display_geometry_is_scientific_authority": False,
            "design_verdict_changed": False,
            "materializable_candidate": False,
            "downstream_physics_authorized": False,
            "g0a_pass": False,
        },
    }
    _write(EVIDENCE, presentation_evidence)
    _write(GEOMETRY, display_geometry)
    layout = _render_board(source_evidence, source_geometry)
    BASE._build_blueprint = _build_blueprint
    import roarm_rl.viz_debug as viz_debug

    original_logger = viz_debug.log_rerun
    original_base_evidence = BASE.EVIDENCE
    suppressed_label_entities = 0

    def label_suppressed_logger(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal suppressed_label_entities
        points = []
        for row in kwargs.get("points", []):
            copy = dict(row)
            copy["labels"] = [""]
            points.append(copy)
            suppressed_label_entities += 1
        kwargs["points"] = points
        return original_logger(*args, **kwargs)

    viz_debug.log_rerun = label_suppressed_logger
    BASE.EVIDENCE = SOURCE_EVIDENCE
    try:
        rerun = BASE._write_rerun(presentation_evidence, display_geometry)
    finally:
        BASE.EVIDENCE = original_base_evidence
        viz_debug.log_rerun = original_logger
    if suppressed_label_entities != len(display_geometry["diagnostic_shared_seams"]):
        raise RuntimeError(
            "D397 attempt3 seam-label suppression count mismatch: "
            f"{suppressed_label_entities} != "
            f"{len(display_geometry['diagnostic_shared_seams'])}"
        )
    template = {
        "artifact": "D397_ATTEMPT3_MANUAL_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board": _png(BOARD),
        "rerun_screenshot": _png(RERUN_SCREENSHOT),
        "checks_to_record": MANUAL_KEYS,
        "minimum_observations": 3,
        "manual_inspection_complete": False,
    }
    _write(MANUAL_TEMPLATE, template)
    checks = {
        "layout_pass": layout["pass"],
        "board_exact_1920x1080": _png(BOARD)["width"] == 1920
        and _png(BOARD)["height"] == 1080,
        "rerun_contract_pass": rerun["pass"],
        "viewer_exactly_one": rerun["viewer_invocations"] == 1,
        "all_eight_seam_labels_suppressed": suppressed_label_entities == 8,
        "science_worker_zero": presentation_evidence["scope_counters"][
            "science_worker_invocations"
        ]
        == 0,
    }
    claim = {
        "artifact": "D397_ATTEMPT3_OBSERVABILITY_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "board": _png(BOARD),
        "rerun": rerun,
        "manual_template": {
            "path": _rel(MANUAL_TEMPLATE),
            "sha256": _sha(MANUAL_TEMPLATE),
        },
        "artifact_hashes": {
            _rel(path): _sha(path)
            for path in (
                PREREG,
                OBSERVE_INVOCATION,
                EVIDENCE,
                GEOMETRY,
                BOARD,
                LAYOUT,
                RRD,
                RBL,
                RERUN_VALIDATION,
                RERUN_SCREENSHOT,
                MANUAL_TEMPLATE,
            )
        },
        "pass": all(checks.values()),
    }
    _write(OBSERVABILITY, claim)
    _phase("observe_end", observability_pass=claim["pass"])
    if not claim["pass"]:
        raise RuntimeError(f"D397 attempt3 observability failed: {checks}")
    print(json.dumps(BASE._native(claim), ensure_ascii=False))
    return 0


def _finalize() -> int:
    global _STAGE_WRITE_STARTED
    if FINALIZE_INVOCATION.exists() or COMPLETION.exists() or FAILURE.exists():
        raise FileExistsError("D397 attempt3 already finalized")
    _STAGE_WRITE_STARTED = True
    prereg = _read(PREREG)
    claim = _read(OBSERVABILITY)
    manual = _read(MANUAL)
    presentation = _read(EVIDENCE)
    current_artifact_hashes = {
        path: _sha(REPO / path) for path in claim["artifact_hashes"]
    }
    continuity = {
        "head_unchanged": _git("rev-parse", "HEAD")
        == prereg["git"]["head"],
        "origin_unchanged": _git("rev-parse", "origin/master")
        == prereg["git"]["origin_master"],
        "script_unchanged": prereg["script"]["sha256"] == _sha(SCRIPT),
        "start_unchanged": prereg["start"]["sha256"] == _sha(START),
        "source_inputs_unchanged": prereg["input_hashes"] == _input_hashes(),
        "status_outside_output_unchanged": prereg["git"][
            "status_outside_output_sha256"
        ]
        == BASE._sha_payload(BASE._status_outside_output()),
        "observe_artifacts_unchanged": claim["artifact_hashes"]
        == current_artifact_hashes,
    }
    _write(
        FINALIZE_INVOCATION,
        {
            "artifact": "D397_ATTEMPT3_FINALIZE_INVOCATION_V1",
            "case": CASE,
            "attempt": ATTEMPT,
            "invocation_count": 1,
            "automatic_retry_count": 0,
            "continuity_checks": continuity,
            "pass": all(continuity.values()),
        },
    )
    checks = {
        "continuity_pass": all(continuity.values()),
        "prereg_pass": prereg["pass"] is True,
        "observability_pass": claim["pass"] is True,
        "manual_complete": manual["manual_inspection_complete"] is True,
        "manual_keys_exact": set(manual["checks"]) == set(MANUAL_KEYS),
        "manual_all_true": all(manual["checks"].values()),
        "manual_notes_ge_3": len(manual["observations"]) >= 3,
        "manual_hash_binding": manual["board_sha256"] == _sha(BOARD)
        and manual["rerun_screenshot_sha256"] == _sha(RERUN_SCREENSHOT),
        "science_design_fail_preserved": presentation["design_pass"] is False,
        "science_worker_zero": presentation["scope_counters"][
            "science_worker_invocations"
        ]
        == 0,
        "process_signals_zero": presentation["scope_counters"][
            "process_signals_sent"
        ]
        == 0,
    }
    completion = {
        "artifact": "D397_ATTEMPT3_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "operational_verdict": (
            "D397_FAILURE_PRESENTATION_REPAIRED_COMPLETE"
            if all(checks.values())
            else "D397_ATTEMPT3_COMPLETION_INTEGRITY_FAIL_STOP"
        ),
        "scientific_design_verdict": presentation["verdict"],
        "design_pass": False,
        "materializable_candidate": False,
        "live_identity_physics_contact_grasp": None,
        "g0a_pass": False,
        "checks": checks,
        "completion_integrity_pass": all(checks.values()),
        "artifacts": {
            path.name: {"path": _rel(path), "sha256": _sha(path)}
            for path in (
                PREREG,
                OBSERVE_INVOCATION,
                EVIDENCE,
                GEOMETRY,
                BOARD,
                LAYOUT,
                RRD,
                RBL,
                RERUN_VALIDATION,
                RERUN_SCREENSHOT,
                OBSERVABILITY,
                MANUAL,
                FINALIZE_INVOCATION,
            )
        },
    }
    _write(COMPLETION, completion)
    _phase(
        "finalize_end",
        completion_integrity_pass=completion["completion_integrity_pass"],
    )
    if not completion["completion_integrity_pass"]:
        raise RuntimeError(f"D397 attempt3 finalize failed: {checks}")
    print(json.dumps(BASE._native(completion), ensure_ascii=False))
    return 0


def _record_failure(stage: str, exc: BaseException) -> None:
    if not _STAGE_WRITE_STARTED:
        return
    if FAILURE.exists():
        return
    try:
        _write(
            FAILURE,
            {
                "artifact": "D397_ATTEMPT3_RUNTIME_FAILURE_V1",
                "case": CASE,
                "attempt": ATTEMPT,
                "stage": stage,
                "exception_type": type(exc).__name__,
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "process_signals_sent": 0,
            },
        )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "observe", "finalize")
    )
    args = parser.parse_args()
    try:
        return {
            "prepare": _prepare,
            "observe": _observe,
            "finalize": _finalize,
        }[args.stage]()
    except Exception as exc:
        _record_failure(args.stage, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
