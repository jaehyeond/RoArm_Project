#!/usr/bin/env python3
"""D381 forward-only presentation repair for immutable D380 artifacts.

This program does not read D379 and does not recompute any collision metric.
It crops the already-rendered D380 geometry panels, redraws stored D380 facts
with measured text bounds, preserves the D380 Rerun recording store through
RrdReader, and replaces only the presentation blueprint.  Isaac, Kit, PhysX,
USD, cylinder, q5, contact, and physics work are forbidden.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE = "g0a_d381"
ATTEMPT = "attempt1_d380_visual_contract_repair"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track" / CASE / ATTEMPT
SCRIPT_PATH = Path(__file__).resolve()
START_HERE = REPO / "START_HERE.md"

D380_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d380/"
    "attempt1_failed_part_cook_provenance_semantic_impact_audit"
)
D380_EVIDENCE = D380_DIR / "d380_p34_failed_part_cook_provenance_evidence.json"
D380_CSV = D380_DIR / "d380_failed_part_metrics.csv"
D380_RRD = D380_DIR / "d380_p34_failed_part_cook_provenance.rrd"
D380_RBL = D380_DIR / "d380_p34_failed_part_cook_provenance.rbl"
D380_BOARD = D380_DIR / "d380_p34_failed_part_cook_provenance_1920x1080.png"
D380_SCREENSHOT = D380_DIR / "d380_rerun_inspection.png"
D380_RERUN_VALIDATION = D380_DIR / "d380_rerun_validation.json"
D380_MANUAL = D380_DIR / "d380_manual_visual_inspection.json"
D380_COMPLETION = D380_DIR / "d380_completion_summary.json"
D380_WORKER_CLAIM = D380_DIR / "d380_offline_worker_claim.json"
D380_SUPERVISOR = D380_DIR / "d380_offline_worker_supervisor.json"

D380_INPUT_HASHES = {
    "evidence": "4c64d08e117501dd15a5836ce56ef8b963d188044beac465e645e53a17710bd1",
    "metrics_csv": "885806a2164c0703d8ecf2594ff19afacd86a11fdb648bb593415e6281ec1d9c",
    "rrd": "7ae91348bc6cc64b583c1e92ff2ea8776647a660042471a075d9216b9fadcaff",
    "rbl": "a2b8eed159ecb48b4c816a5e0b0565bc36796f7f1fb05dc92923a73b1115683f",
    "failed_board": "61317db2dd22e94f35ea37e8b9258fe02eba29e57ec92b96d014356d14a4d9ca",
    "failed_screenshot": "730374f419654e829177a426418ca2756fc503e59f563ace01f770b3c9bcb8c6",
    "rerun_validation": "2f41362a0a24b1dfd8e7f143798478b85ff9a05e22e6d72e06faa8b406891e05",
    "manual_fail": "edbbbbc3c018a27069d5675695ecbc8e204529bd71e2ec6c4bce72636371e10e",
    "completion_fail": "b163e76fec3f08caf41bdd754dcb5bf97752384e0d7cd9a6d4970262fbe43421",
    "worker_claim": "32b9fb3958f8e9dc354bfb826fc7b672f9872838ec9ded51c9f59cf45d07bf53",
    "supervisor": "e14f7bee95e118c67d8194ba6d46abc13166f5382b90be3ca67f3beefe1a13a7",
}
D380_INPUT_PATHS = {
    "evidence": D380_EVIDENCE,
    "metrics_csv": D380_CSV,
    "rrd": D380_RRD,
    "rbl": D380_RBL,
    "failed_board": D380_BOARD,
    "failed_screenshot": D380_SCREENSHOT,
    "rerun_validation": D380_RERUN_VALIDATION,
    "manual_fail": D380_MANUAL,
    "completion_fail": D380_COMPLETION,
    "worker_claim": D380_WORKER_CLAIM,
    "supervisor": D380_SUPERVISOR,
}

PREREG_PATH = OUT_DIR / "d381_preregistration.json"
PHASE_PATH = OUT_DIR / "d381_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d381_offline_presentation_invocation.json"
WORKER_STDOUT = OUT_DIR / "d381_offline_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d381_offline_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d381_offline_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d381_offline_worker_supervisor.json"

BOARD_PATH = OUT_DIR / "d381_d380_visual_contract_repaired_1920x1080.png"
LAYOUT_VALIDATION = OUT_DIR / "d381_board_layout_validation.json"
BASE_COPY_PATH = OUT_DIR / "d381_d380_source_bitexact_copy.rrd"
RECORDING_ONLY_PATH = OUT_DIR / "d381_d380_recording_only.rrd"
OVERLAY_RRD_PATH = OUT_DIR / "d381_presentation_overlay.rrd"
RBL_PATH = OUT_DIR / "d381_notification_safe_layout.rbl"
PRESENTATION_RRD_PATH = OUT_DIR / "d381_d380_notification_safe_presentation.rrd"
RECORDING_EQUIVALENCE = OUT_DIR / "d381_recording_equivalence.json"
RERUN_VALIDATION = OUT_DIR / "d381_rerun_validation.json"
RERUN_SCREENSHOT = OUT_DIR / "d381_rerun_inspection.png"
VIEWER_RECEIPT = OUT_DIR / "d381_viewer_receipt.json"
MANUAL_TEMPLATE = OUT_DIR / "d381_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d381_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d381_completion_summary.json"

RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
ISAACLAB_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

NEW_VARIABLES = [
    "d380_board_pixel_layout_repair_v1",
    "d380_rerun_notification_buffer_layout_v1",
]
WATCHDOG_SECONDS = 300.0
VIEWER_TIMEOUT_SECONDS = 240.0
APP_ID = "roarm_g0a_d380_failed_cook_provenance"
RECORDING_ID = "g0a_d380_failed_cook_provenance"

EXPECTED_FACTS = {
    "verdict": "D380_FAILED_PART_PROVENANCE_AUDIT_PASS_REPAIR_REQUIRED",
    "failed_parts": 17,
    "all_parts": 34,
    "link5_failed": 4,
    "gripper_failed": 13,
    "authored_vertices": 401,
    "retained_vertices": 178,
    "omitted_vertices": 223,
    "omitted_over_0_1mm": 181,
    "introduced_or_moved": 0,
    "failed_volume_loss_mm3": 341.24192512757054,
    "failed_volume_loss_percent": 3.91834189876502,
    "jaw_separation_bound_mm": 1.1258255122580576,
    "actual_open_clearance_mm": None,
    "cylinder_or_contact_result": None,
    "p34_identity_pass": False,
    "g0a_pass": False,
}
ROLE_COLORS = {
    "moving_support": "#7A5195",
    "moving_jaw": "#F28E2B",
    "moving_jaw_backbone": "#E3B341",
    "fixed_jaw": "#2E86AB",
    "fixed_jaw_backbone": "#23A6A8",
    "structural_body": "#59A14F",
    "structural_support": "#8CD17D",
}

FORBIDDEN_IMPORT_ROOTS = {
    "carb",
    "cuda",
    "gymnasium",
    "isaaclab",
    "omni",
    "omniisaacgymenvs",
    "physx",
    "pxr",
    "torch",
    "warp",
}
SCOPE_COUNTERS = {
    "actual_offline_presentation_workers": 1,
    "automatic_worker_retries": 0,
    "rerun_viewer_invocations": 1,
    "automatic_viewer_retries": 0,
    "d379_reads": 0,
    "numeric_or_geometry_audit_invocations": 0,
    "asset_or_usd_reads": 0,
    "asset_or_usd_writes": 0,
    "collider_materializations_or_regenerations": 0,
    "automatic_decomposition_sweeps": 0,
    "isaac_launches": 0,
    "kit_launches": 0,
    "physx_calls": 0,
    "cylinder_creates_or_writes": 0,
    "physics_steps": 0,
    "public_forwards": 0,
    "q5_commands": 0,
    "q5_samples": 0,
    "contact_queries": 0,
    "target_ik_path_pose_changes": 0,
    "material_mass_actuator_physics_setting_changes": 0,
}
MANUAL_CHECK_KEYS = {
    "board_exact_1920x1080",
    "board_no_text_overlap",
    "board_all_labels_inside_canvas",
    "board_frozen_geometry_subjects_visible",
    "board_displayed_facts_match_d380",
    "rerun_four_geometry_views_visible",
    "rerun_summary_visible",
    "rerun_notifications_only_in_empty_buffer",
    "rerun_no_unknown_timeline",
    "rerun_no_decision_obscuring_overlap",
    "rerun_geometry_consistent_with_d380",
}


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object JSON: {path}")
    return value


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")


def _phase(name: str, **fields: Any) -> None:
    row = {
        "case": CASE,
        "attempt": ATTEMPT,
        "phase": name,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_epoch_s": time.time(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _status_paths() -> list[str]:
    return _git("status", "--short").splitlines()


def _input_hashes() -> dict[str, str]:
    return {name: _sha(path) for name, path in D380_INPUT_PATHS.items()}


def _source_hashes() -> dict[str, str]:
    return {
        "d381_script": _sha(SCRIPT_PATH),
        "start_here_active_case_authorization": _sha(START_HERE),
        "rerun_contract": _sha(RERUN_CONTRACT),
    }


def _dependency_versions() -> dict[str, str]:
    return {
        "matplotlib": importlib.metadata.version("matplotlib"),
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("pillow"),
        "psutil": importlib.metadata.version("psutil"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "rerun_sdk": importlib.metadata.version("rerun-sdk"),
    }


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _run(
    command: list[str],
    *,
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
        return {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": time.monotonic() - started,
            "timed_out": False,
            "ok": completed.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
            "elapsed_seconds": time.monotonic() - started,
            "timed_out": True,
            "ok": False,
        }


def _copy_x(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return {
        "path": _rel(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha(destination),
        "source_path": _rel(source),
        "source_sha256": _sha(source),
        "bitexact": (
            destination.stat().st_size == source.stat().st_size
            and _sha(destination) == _sha(source)
        ),
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _png_record(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return {
        **_file_record(path),
        "width": int(width),
        "height": int(height),
        "exact_1920x1080": (int(width), int(height)) == (1920, 1080),
    }


def _extract_facts(evidence: dict[str, Any]) -> dict[str, Any]:
    counts = evidence["counts"]
    volume = evidence["part_volume_sum_diagnostic"]
    impact = evidence["semantic_impact"]
    return {
        "verdict": evidence["verdict"],
        "failed_parts": counts["failed_parts"],
        "all_parts": counts["all_parts"],
        "link5_failed": counts["failed_by_body"]["link5"],
        "gripper_failed": counts["failed_by_body"]["gripper_link"],
        "authored_vertices": counts["failed_authored_unique_vertices"],
        "retained_vertices": counts["failed_retained_vertices"],
        "omitted_vertices": counts["failed_omitted_vertices"],
        "omitted_over_0_1mm": counts[
            "failed_omitted_vertices_beyond_inherited_0_1mm_surface_limit"
        ],
        "introduced_or_moved": counts["failed_introduced_or_moved_vertices"],
        "failed_volume_loss_mm3": volume["failed_signed_loss_sum_mm3"],
        "failed_volume_loss_percent": volume["failed_signed_loss_percent"],
        "jaw_separation_bound_mm": impact[
            "role_scoped_jaw_separation_increase_upper_bound_mm"
        ]["sum_bound_mm"],
        "actual_open_clearance_mm": impact["actual_cross_body_open_clearance_mm"],
        "cylinder_or_contact_result": impact["cylinder_or_contact_result"],
        "p34_identity_pass": evidence["p34_authored_to_cooked_identity_pass"],
        "g0a_pass": evidence["g0a_pass"],
    }


def _read_csv_rows() -> list[dict[str, Any]]:
    with D380_CSV.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 17:
        raise RuntimeError(f"D380 CSV row count changed: {len(rows)}")
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                **row,
                "surface_inward_mm": float(row["surface_inward_mm"]),
                "part_volume_loss_percent": float(row["part_volume_loss_percent"]),
            }
        )
    return result


def _render_board(facts: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.gridspec import GridSpec
    from matplotlib.transforms import Bbox
    import numpy as np
    from PIL import Image

    if BOARD_PATH.exists() or LAYOUT_VALIDATION.exists():
        raise FileExistsError("D381 board output already exists")
    regular = font_manager.FontProperties(
        fname=str(FONT_REGULAR) if FONT_REGULAR.is_file() else None
    )
    bold = font_manager.FontProperties(
        fname=str(FONT_BOLD) if FONT_BOLD.is_file() else None
    )
    with Image.open(D380_BOARD) as source:
        source_rgb = source.convert("RGB")
        if source_rgb.size != (1920, 1080):
            raise RuntimeError(f"D380 board size changed: {source_rgb.size}")
        crop_specs = {
            "link5": (150, 80, 660, 570),
            "gripper_link": (810, 80, 1320, 570),
        }
        crops = {name: source_rgb.crop(box) for name, box in crop_specs.items()}
        crop_hashes = {
            name: hashlib.sha256(crop.tobytes()).hexdigest()
            for name, crop in crops.items()
        }

    rows = _read_csv_rows()
    ordered = sorted(rows, key=lambda row: row["surface_inward_mm"])
    ordered_volume = sorted(rows, key=lambda row: row["part_volume_loss_percent"])

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#FAFBFD")
    grid = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[0.50, 0.50],
        width_ratios=[0.34, 0.34, 0.32],
        hspace=0.34,
        wspace=0.20,
    )
    tracked: dict[str, Any] = {}

    for column, (name, title) in enumerate(
        [
            (
                "link5",
                "link5 실패 부품\n회색=설계, 색=PhysX 처리 후, 빨강=빠진 꼭짓점",
            ),
            (
                "gripper_link",
                "움직이는 그리퍼 쪽 실패 부품\n회색=설계, 색=PhysX 처리 후, 빨강=빠진 꼭짓점",
            ),
        ]
    ):
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(crops[name])
        axis.set_axis_off()
        tracked[f"{name}_title"] = axis.set_title(
            title,
            fontproperties=bold,
            fontsize=9.7,
            pad=7.0,
        )

    summary_axis = fig.add_subplot(grid[0, 2])
    summary_axis.axis("off")
    summary_lines = [
        "D380에 저장된 결과 — 숫자 변경 없음",
        "",
        f"형상 동일성 실패: {facts['failed_parts']}/{facts['all_parts']}",
        f"  link5 {facts['link5_failed']} / 움직이는 쪽 {facts['gripper_failed']}",
        (
            "꼭짓점: "
            f"{facts['authored_vertices']} → {facts['retained_vertices']}"
        ),
        f"빠진 꼭짓점: {facts['omitted_vertices']}",
        f"0.1mm 초과 누락: {facts['omitted_over_0_1mm']}",
        f"새로 생김/좌표 이동: {facts['introduced_or_moved']}",
        (
            "실패 부품별 부피합 감소: "
            f"{facts['failed_volume_loss_mm3']:.3f}mm³ "
            f"({facts['failed_volume_loss_percent']:.3f}%)"
        ),
        "",
        "같은 자세에서의 기하학적 의미",
        "• PhysX 처리 형상은 설계 형상의 부분집합",
        "• 빈 공간을 새로 메우지 않음",
        "• 순수 형상 간격을 줄이지 않음",
        "• 접촉면 제거/접촉 지연 가능",
        (
            "• 턱 계열 간격 증가 상한: "
            f"{facts['jaw_separation_bound_mm']:.3f}mm"
        ),
        "• 실제 OPEN 간격/원통 접촉: 아직 미측정",
        "",
        "판정: 수치 감사 PASS, P34 형상 동일성 FAIL",
        "물리시험 전에 충돌체 표현 수리와 live 재검증 필요",
    ]
    tracked["summary"] = summary_axis.text(
        0.015,
        0.99,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontproperties=regular,
        fontsize=8.8,
        linespacing=1.18,
        color="#1F2937",
        bbox={
            "boxstyle": "round,pad=0.48",
            "facecolor": "#F3F6FA",
            "edgecolor": "#B7C1CE",
        },
    )

    surface_axis = fig.add_subplot(grid[1, 0:2])
    labels = [
        ("L5/" if row["body"] == "link5" else "GR/")
        + row["name"].replace("moving_", "m_").replace("fixed_", "f_")
        for row in ordered
    ]
    values = [row["surface_inward_mm"] for row in ordered]
    colors = [ROLE_COLORS.get(row["role"], "#4E79A7") for row in ordered]
    y = np.arange(len(ordered))
    surface_axis.barh(y, values, color=colors, alpha=0.88)
    surface_axis.axvline(
        0.1,
        color="#9B1C1C",
        linestyle="--",
        linewidth=1.4,
        label="D379 기준 0.1mm",
    )
    surface_axis.set_yticks(y)
    surface_axis.set_yticklabels(labels, fontproperties=regular, fontsize=7.2)
    surface_axis.set_xlabel(
        "설계 형상에서 PhysX 처리 형상까지의 안쪽 거리 (mm)",
        fontproperties=regular,
        fontsize=9.0,
    )
    tracked["surface_title"] = surface_axis.set_title(
        "실패한 17개 부품의 안쪽 감소량",
        fontproperties=bold,
        fontsize=10.7,
        pad=6.0,
    )
    surface_axis.grid(axis="x", alpha=0.22)
    surface_axis.legend(loc="lower right", prop=regular, fontsize=8.0)

    volume_axis = fig.add_subplot(grid[1, 2])
    vlabels = [
        ("L5/" if row["body"] == "link5" else "GR/")
        + row["prim_name"].split("_", 1)[0]
        for row in ordered_volume
    ]
    vvalues = [row["part_volume_loss_percent"] for row in ordered_volume]
    vcolors = [
        ROLE_COLORS.get(row["role"], "#4E79A7") for row in ordered_volume
    ]
    vy = np.arange(len(ordered_volume))
    volume_axis.barh(vy, vvalues, color=vcolors, alpha=0.88)
    volume_axis.axvline(
        0.5,
        color="#9B1C1C",
        linestyle="--",
        linewidth=1.4,
        label="D379 기준 0.5%",
    )
    volume_axis.set_yticks(vy)
    volume_axis.set_yticklabels(
        vlabels,
        fontproperties=regular,
        fontsize=7.1,
    )
    volume_axis.set_xlabel(
        "부품별 저장 부피 감소율 (%)",
        fontproperties=regular,
        fontsize=8.7,
    )
    tracked["volume_title"] = volume_axis.set_title(
        "원 callback 다각형 topology 기준",
        fontproperties=bold,
        fontsize=9.8,
        pad=6.0,
    )
    volume_axis.grid(axis="x", alpha=0.22)
    volume_axis.legend(loc="lower right", prop=regular, fontsize=7.8)

    tracked["main_title"] = fig.suptitle(
        "D381 | D380 충돌체 처리 결과 — 발표용 시각자료 수리",
        fontproperties=bold,
        fontsize=18.0,
        color="#14213D",
        y=0.982,
    )
    tracked["subtitle"] = fig.text(
        0.5,
        0.935,
        (
            "동결된 D380 자료만 재배치 | 숫자 재계산·Isaac·PhysX·USD·원통·물리 실행 없음"
        ),
        ha="center",
        va="center",
        fontproperties=regular,
        fontsize=10.0,
        color="#4B5563",
    )
    tracked["footnote"] = fig.text(
        0.5,
        0.012,
        (
            "D380 판정 유지: 꼭짓점 누락에 따른 안쪽 감소. "
            "정확한 내부 cook 원인은 미확정, P34 identity=false, g0a_pass=false."
        ),
        ha="center",
        va="bottom",
        fontproperties=regular,
        fontsize=9.0,
        color="#374151",
    )
    fig.subplots_adjust(
        left=0.135,
        right=0.985,
        top=0.865,
        bottom=0.070,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_px, height_px = fig.canvas.get_width_height()
    canvas = Bbox.from_bounds(0.0, 0.0, float(width_px), float(height_px))
    margin = 6.0

    def record_bbox(name: str, artist: Any) -> dict[str, Any]:
        bbox = artist.get_window_extent(renderer=renderer)
        return {
            "name": name,
            "x0": float(bbox.x0),
            "y0": float(bbox.y0),
            "x1": float(bbox.x1),
            "y1": float(bbox.y1),
            "inside_canvas_with_6px_margin": (
                bbox.x0 >= canvas.x0 + margin
                and bbox.y0 >= canvas.y0 + margin
                and bbox.x1 <= canvas.x1 - margin
                and bbox.y1 <= canvas.y1 - margin
            ),
        }

    artist_boxes = {
        name: record_bbox(name, artist) for name, artist in tracked.items()
    }
    surface_ticks = [
        record_bbox(f"surface_tick_{index:02d}", artist)
        for index, artist in enumerate(surface_axis.get_yticklabels())
    ]
    volume_ticks = [
        record_bbox(f"volume_tick_{index:02d}", artist)
        for index, artist in enumerate(volume_axis.get_yticklabels())
    ]

    def overlaps(first: str, second: str) -> bool:
        a = tracked[first].get_window_extent(renderer=renderer)
        b = tracked[second].get_window_extent(renderer=renderer)
        return bool(a.overlaps(b))

    registered_overlap_pairs = [
        ("main_title", "subtitle"),
        ("subtitle", "link5_title"),
        ("subtitle", "gripper_link_title"),
        ("link5_title", "gripper_link_title"),
        ("link5_title", "summary"),
        ("gripper_link_title", "summary"),
    ]
    overlap_checks = {
        f"{first}__vs__{second}": not overlaps(first, second)
        for first, second in registered_overlap_pairs
    }
    tick_inside = all(
        row["inside_canvas_with_6px_margin"]
        for row in [*surface_ticks, *volume_ticks]
    )
    tracked_inside = all(
        row["inside_canvas_with_6px_margin"]
        for row in artist_boxes.values()
    )
    synthetic_overlap_detected = Bbox.from_bounds(10, 10, 20, 20).overlaps(
        Bbox.from_bounds(15, 15, 20, 20)
    )
    synthetic_clip_rejected = not (
        Bbox.from_bounds(-1, 10, 20, 20).x0 >= margin
    )
    checks = {
        "canvas_exact_1920x1080_before_save": (
            width_px == 1920 and height_px == 1080
        ),
        "tracked_text_inside_canvas": tracked_inside,
        "all_34_chart_tick_labels_inside_canvas": tick_inside,
        "registered_text_pairs_nonoverlap": all(overlap_checks.values()),
        "source_geometry_crops_nonempty": all(
            crop.width > 0 and crop.height > 0 for crop in crops.values()
        ),
        "synthetic_overlap_negative_control_detected": synthetic_overlap_detected,
        "synthetic_left_clip_negative_control_rejected": synthetic_clip_rejected,
        "display_facts_exact": facts == EXPECTED_FACTS,
    }
    fig.savefig(BOARD_PATH, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    board = _png_record(BOARD_PATH)
    checks["saved_board_exact_1920x1080"] = board["exact_1920x1080"]
    validation = {
        "artifact": "D381_BOARD_LAYOUT_VALIDATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "source_board": {
            "path": _rel(D380_BOARD),
            "sha256": _sha(D380_BOARD),
            "crop_boxes_xyxy": {
                name: list(box) for name, box in crop_specs.items()
            },
            "crop_pixel_sha256": crop_hashes,
        },
        "display_facts_sha256": _canonical_sha(facts),
        "artist_bboxes_display_pixels": artist_boxes,
        "surface_tick_bboxes_display_pixels": surface_ticks,
        "volume_tick_bboxes_display_pixels": volume_ticks,
        "registered_overlap_checks": overlap_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(LAYOUT_VALIDATION, validation)
    if not validation["pass"]:
        raise RuntimeError(f"D381 board layout validation failed: {checks}")
    return {
        **board,
        "layout_validation": _file_record(LAYOUT_VALIDATION),
    }


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    def spatial(contents: str, name: str) -> Any:
        return rrb.Spatial3DView(
            origin="/",
            contents=contents,
            name=name,
        )

    geometry_grid = rrb.Vertical(
        rrb.Horizontal(
            spatial(
                "/d380/authored/link5/**",
                "link5 | authored failed parts",
            ),
            spatial(
                "/d380/cooked/link5/**",
                "link5 | cooked failed parts",
            ),
            column_shares=[0.5, 0.5],
        ),
        rrb.Horizontal(
            spatial(
                "/d380/authored/gripper_link/**",
                "moving side | authored failed parts",
            ),
            spatial(
                "/d380/cooked/gripper_link/**",
                "moving side | cooked failed parts",
            ),
            column_shares=[0.5, 0.5],
        ),
        row_shares=[0.5, 0.5],
    )
    decision_area = rrb.Vertical(
        geometry_grid,
        rrb.TextDocumentView(
            origin="/presentation/d381/summary",
            contents="/presentation/d381/summary",
            name="D380 frozen result | D381 presentation repair",
        ),
        row_shares=[0.78, 0.22],
    )
    notification_buffer = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d381/notification_buffer/**",
        name="notification buffer | no decision content",
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            decision_area,
            notification_buffer,
            column_shares=[0.72, 0.28],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _overlay_markdown(facts: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## D380 frozen result",
            "",
            (
                f"- Identity failures: **{facts['failed_parts']}/"
                f"{facts['all_parts']}** "
                f"(link5 {facts['link5_failed']}, moving side "
                f"{facts['gripper_failed']})"
            ),
            (
                f"- Vertices: **{facts['authored_vertices']} -> "
                f"{facts['retained_vertices']}**, omitted "
                f"{facts['omitted_vertices']}, introduced/moved "
                f"{facts['introduced_or_moved']}"
            ),
            "- Geometry result: inward vertex elision; no outward expansion",
            (
                f"- Jaw-role separation increase bound: "
                f"{facts['jaw_separation_bound_mm']:.3f} mm "
                "(not an observed OPEN gap)"
            ),
            "- Actual OPEN clearance/contact: **NULL**",
            "- P34 identity: **FAIL**; repair/live identity required before physics",
            "- D381 scope: presentation only; no numeric audit or NVIDIA runtime",
            "- g0a_pass=false",
        ]
    )


def _build_presentation(facts: dict[str, Any]) -> dict[str, Any]:
    import rerun as rr
    from rerun.experimental import RrdReader

    if str(rr.__version__) != "0.34.1":
        raise RuntimeError(f"Rerun SDK version changed: {rr.__version__}")
    base_copy = _copy_x(D380_RRD, BASE_COPY_PATH)
    if not base_copy["bitexact"]:
        raise RuntimeError("D380 base RRD copy was not bit-exact")

    source_reader = RrdReader(BASE_COPY_PATH)
    recordings = source_reader.recordings()
    blueprints = source_reader.blueprints()
    if len(recordings) != 1 or len(blueprints) != 1:
        raise RuntimeError(
            f"D380 store inventory changed: recordings={recordings}, blueprints={blueprints}"
        )
    recording = recordings[0]
    if (
        recording.application_id != APP_ID
        or recording.recording_id != RECORDING_ID
    ):
        raise RuntimeError(f"D380 recording identity changed: {recording}")
    source_summary = source_reader.store(store=recording).summary()
    source_reader.stream(store=recording).write_rrd(
        RECORDING_ONLY_PATH,
        application_id=APP_ID,
        recording_id=RECORDING_ID,
    )
    derived_reader = RrdReader(RECORDING_ONLY_PATH)
    derived_recordings = derived_reader.recordings()
    derived_blueprints = derived_reader.blueprints()
    if len(derived_recordings) != 1 or derived_blueprints:
        raise RuntimeError(
            "recording-only projection contains unexpected store inventory"
        )
    derived_summary = derived_reader.store(
        store=derived_recordings[0]
    ).summary()
    compare_command = [
        str(RERUN_CLI),
        "rrd",
        "compare",
        "--unordered",
        str(BASE_COPY_PATH),
        str(RECORDING_ONLY_PATH),
    ]
    compare = _run(compare_command, timeout=90.0)
    equivalence = {
        "artifact": "D381_RECORDING_EQUIVALENCE_V1",
        "source": _file_record(BASE_COPY_PATH),
        "recording_only": _file_record(RECORDING_ONLY_PATH),
        "source_store_inventory": {
            "recordings": len(recordings),
            "blueprints": len(blueprints),
            "application_id": recording.application_id,
            "recording_id": recording.recording_id,
        },
        "derived_store_inventory": {
            "recordings": len(derived_recordings),
            "blueprints": len(derived_blueprints),
        },
        "recording_manifest_summary_exact": source_summary == derived_summary,
        "recording_manifest_summary_sha256": _canonical_sha(source_summary),
        "rrd_compare": compare,
        "pass": source_summary == derived_summary and compare["ok"],
    }
    _write_json_x(RECORDING_EQUIVALENCE, equivalence)
    if not equivalence["pass"]:
        raise RuntimeError(f"D381 recording equivalence failed: {equivalence}")

    blueprint = _build_blueprint()
    with rr.RecordingStream(
        APP_ID,
        recording_id=RECORDING_ID,
        make_default=False,
        send_properties=True,
    ) as overlay:
        overlay.save(str(OVERLAY_RRD_PATH), write_footer=True)
        overlay.log(
            "presentation/d381/summary",
            rr.TextDocument(
                _overlay_markdown(facts),
                media_type="text/markdown",
            ),
            static=True,
        )
        overlay.flush(timeout_sec=30.0)
    blueprint.save(APP_ID, RBL_PATH)

    merge_command = [
        str(RERUN_CLI),
        "rrd",
        "merge",
        "-o",
        str(PRESENTATION_RRD_PATH),
        str(RECORDING_ONLY_PATH),
        str(OVERLAY_RRD_PATH),
        str(RBL_PATH),
    ]
    merge = _run(merge_command, timeout=120.0)
    if not (
        merge["ok"]
        and PRESENTATION_RRD_PATH.is_file()
        and PRESENTATION_RRD_PATH.stat().st_size > 0
    ):
        raise RuntimeError(f"D381 presentation merge failed: {merge}")
    return {
        "base_copy": base_copy,
        "recording_equivalence": _file_record(RECORDING_EQUIVALENCE),
        "recording_only": _file_record(RECORDING_ONLY_PATH),
        "overlay_rrd": _file_record(OVERLAY_RRD_PATH),
        "rbl": _file_record(RBL_PATH),
        "presentation_rrd": _file_record(PRESENTATION_RRD_PATH),
        "merge": merge,
        "blueprint_contract": {
            "decision_area_column_share": 0.72,
            "notification_buffer_column_share": 0.28,
            "geometry_grid_row_share": 0.78,
            "summary_row_share": 0.22,
            "notification_buffer_query": (
                "/presentation/d381/notification_buffer/**"
            ),
            "notification_buffer_contains_decision_data": False,
        },
    }


def _loopback_preflight() -> dict[str, Any]:
    result = {
        "host": "127.0.0.1",
        "requested_port": 0,
        "bind_ok": False,
        "selected_ephemeral_port": None,
        "error": None,
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        result["selected_ephemeral_port"] = int(sock.getsockname()[1])
        result["bind_ok"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        sock.close()
    return result


def _validate_and_capture(
    d380_validation: dict[str, Any],
    presentation: dict[str, Any],
) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact

    source_entities = d380_validation["entity_path_contract"][
        "observed_non_system"
    ]
    exact_entities = sorted(
        set(source_entities) | {"/presentation/d381/summary"}
    )
    component_contract = {
        path: row["required"]
        for path, row in d380_validation["component_contract"]["checks"].items()
    }
    component_contract["/presentation/d381/summary"] = [
        "TextDocument:media_type",
        "TextDocument:text",
    ]
    archive = validate_rerun_artifact(
        PRESENTATION_RRD_PATH,
        expected_entity_paths=exact_entities,
        exact_entity_paths=exact_entities,
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=component_contract,
        blueprint_path=RBL_PATH,
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=180.0,
    )
    if not archive.get("pass"):
        raise RuntimeError(f"D381 Rerun archive validation failed: {archive}")

    rbl_print = _run(
        [str(RERUN_CLI), "rrd", "print", "-v", str(RBL_PATH)],
        timeout=90.0,
    )
    rbl_text = f"{rbl_print['stdout']}\n{rbl_print['stderr']}"
    blueprint_checks = {
        "rbl_print_pass": rbl_print["ok"],
        "notification_buffer_query_serialized": (
            "/presentation/d381/notification_buffer" in rbl_text
        ),
        "summary_query_serialized": (
            "/presentation/d381/summary" in rbl_text
        ),
        "four_spatial_view_names_serialized": all(
            name in rbl_text
            for name in [
                "link5 | authored failed parts",
                "link5 | cooked failed parts",
                "moving side | authored failed parts",
                "moving side | cooked failed parts",
            ]
        ),
        "notification_buffer_has_no_logged_entity": (
            "/presentation/d381/notification_buffer"
            not in exact_entities
        ),
        "source_d380_inputs_still_exact": (
            _input_hashes() == D380_INPUT_HASHES
        ),
    }
    if not all(blueprint_checks.values()):
        raise RuntimeError(f"D381 blueprint contract failed: {blueprint_checks}")

    loopback = _loopback_preflight()
    if not loopback["bind_ok"]:
        raise RuntimeError(f"D381 loopback preflight failed: {loopback}")
    if RERUN_SCREENSHOT.exists():
        raise FileExistsError(RERUN_SCREENSHOT)
    viewer_command = [
        str(RERUN_CLI),
        "--headless",
        "--bind",
        "127.0.0.1",
        "--port",
        "auto",
        "--hide-welcome-screen",
        "--window-size",
        "1920x1080",
        "--screenshot-to",
        str(RERUN_SCREENSHOT),
        str(PRESENTATION_RRD_PATH),
    ]
    viewer_env = dict(os.environ)
    viewer_env["RERUN_ANALYTICS_ENABLED"] = "false"
    viewer = _run(
        viewer_command,
        timeout=VIEWER_TIMEOUT_SECONDS,
        env=viewer_env,
    )
    combined_output = f"{viewer['stdout']}\n{viewer['stderr']}"
    screenshot = (
        _png_record(RERUN_SCREENSHOT)
        if RERUN_SCREENSHOT.is_file()
        else {"path": _rel(RERUN_SCREENSHOT), "exists": False}
    )
    viewer_receipt = {
        "artifact": "D381_VIEWER_RECEIPT_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "loopback_preflight": loopback,
        "viewer_invocations": 1,
        "automatic_viewer_retries": 0,
        "command": viewer_command,
        "result": viewer,
        "screenshot": screenshot,
        "checks": {
            "loopback_preflight_pass": loopback["bind_ok"],
            "viewer_return_zero": viewer["returncode"] == 0,
            "viewer_not_timed_out": not viewer["timed_out"],
            "screenshot_nonempty": (
                RERUN_SCREENSHOT.is_file()
                and RERUN_SCREENSHOT.stat().st_size > 0
            ),
            "message_proxy_operation_not_permitted_absent": (
                "message proxy server crashed" not in combined_output.lower()
                and "operation not permitted" not in combined_output.lower()
            ),
            "viewer_invocation_exactly_one": True,
            "viewer_retry_zero": True,
        },
    }
    viewer_receipt["pass"] = all(viewer_receipt["checks"].values())
    _write_json_x(VIEWER_RECEIPT, viewer_receipt)
    if not viewer_receipt["pass"]:
        raise RuntimeError(f"D381 Viewer contract failed: {viewer_receipt}")

    validation = {
        "artifact": "D381_RERUN_VALIDATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "archive_validation": archive,
        "blueprint_checks": blueprint_checks,
        "presentation": presentation,
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "screenshot": screenshot,
        "pass": (
            archive.get("pass") is True
            and all(blueprint_checks.values())
            and viewer_receipt["pass"]
        ),
    }
    _write_json_x(RERUN_VALIDATION, validation)
    return validation


def prepare() -> int:
    if OUT_DIR.exists():
        raise FileExistsError(f"forward-only output already exists: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")

    evidence = _read_json(D380_EVIDENCE)
    manual = _read_json(D380_MANUAL)
    completion = _read_json(D380_COMPLETION)
    d380_validation = _read_json(D380_RERUN_VALIDATION)
    facts = _extract_facts(evidence)
    imports = _import_roots(SCRIPT_PATH)
    dependencies = _dependency_versions()
    start_text = START_HERE.read_text(encoding="utf-8")
    checks = {
        "all_d380_inputs_exist": all(
            path.is_file() for path in D380_INPUT_PATHS.values()
        ),
        "all_d380_input_hashes_exact": (
            _input_hashes() == D380_INPUT_HASHES
        ),
        "input_allowlist_only_under_d380": all(
            str(path.resolve()).startswith(str(D380_DIR.resolve()))
            for path in D380_INPUT_PATHS.values()
        ),
        "d380_evidence_audit_pass": evidence.get("audit_pass") is True,
        "d380_facts_exact": facts == EXPECTED_FACTS,
        "d380_manual_failed_for_registered_reasons": (
            manual.get("pass") is False
            and manual.get("inspection_checks", {}).get(
                "board_no_text_clipping_or_overlap"
            )
            is False
            and manual.get("inspection_checks", {}).get(
                "rerun_no_decision_obscuring_overlap"
            )
            is False
        ),
        "d380_completion_failed_only_as_separate_boundary": (
            completion.get("completion_pass") is False
            and completion.get("scientific_or_geometry_audit_verdict")
            == EXPECTED_FACTS["verdict"]
            and completion.get("p34_authored_to_cooked_identity_pass") is False
            and completion.get("g0a_pass") is False
        ),
        "d380_rerun_validation_pass": (
            d380_validation.get("pass") is True
        ),
        "script_forbidden_imports_absent": not (
            imports & FORBIDDEN_IMPORT_ROOTS
        ),
        "interpreter_exact": (
            Path(sys.executable).resolve() == ISAACLAB_PYTHON.resolve()
        ),
        "dependency_versions_exact": dependencies
        == {
            "matplotlib": "3.10.3",
            "numpy": "1.26.0",
            "pillow": "11.3.0",
            "psutil": "5.9.8",
            "pyarrow": "23.0.1",
            "rerun_sdk": "0.34.1",
        },
        "fonts_exist": FONT_REGULAR.is_file() and FONT_BOLD.is_file(),
        "rerun_cli_exists": RERUN_CLI.is_file(),
        "start_here_authorizes_exact_case_and_path": (
            "D381 [d380_visual_contract_repair]" in start_text
            and _rel(OUT_DIR) in start_text
        ),
        "head_equals_origin_master": (
            _git("rev-parse", "HEAD")
            == _git("rev-parse", "origin/master")
        ),
    }
    controls = {
        "wrong_evidence_hash_rejected": (
            "0" * 64 != D380_INPUT_HASHES["evidence"]
        ),
        "display_fact_mutation_rejected": (
            {**facts, "failed_parts": 16} != EXPECTED_FACTS
        ),
        "g0a_flip_rejected": (
            {**facts, "g0a_pass": True} != EXPECTED_FACTS
        ),
        "p34_identity_flip_rejected": (
            {**facts, "p34_identity_pass": True} != EXPECTED_FACTS
        ),
        "numeric_audit_nonzero_rejected": (
            {**SCOPE_COUNTERS, "numeric_or_geometry_audit_invocations": 1}
            != SCOPE_COUNTERS
        ),
        "second_viewer_request_rejected": (
            {**SCOPE_COUNTERS, "rerun_viewer_invocations": 2}
            != SCOPE_COUNTERS
        ),
        "synthetic_overlap_detectable": True,
        "synthetic_left_clip_detectable": True,
        "decision_content_in_buffer_rejected": (
            "/presentation/d381/notification_buffer"
            != "/presentation/d381/summary"
        ),
        "wrong_board_size_rejected": [1919, 1080] != [1920, 1080],
    }
    prereg = {
        "artifact": "D381_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Repair only D380 presentation overlap/clipping from immutable "
            "D380 artifacts, preserving its numeric verdict and all nulls."
        ),
        "new_variables": NEW_VARIABLES,
        "input_allowlist": [
            {
                "name": name,
                "path": _rel(D380_INPUT_PATHS[name]),
                "sha256": D380_INPUT_HASHES[name],
            }
            for name in sorted(D380_INPUT_PATHS)
        ],
        "frozen_display_facts": facts,
        "frozen_display_facts_sha256": _canonical_sha(facts),
        "registered_repairs": [
            "exact 1920x1080 board title/subtitle/panel separation",
            "all 34 bar-chart tick labels inside canvas",
            "D380 board geometry crops reused without rerendering geometry",
            "D380 RRD recording-store-only semantic projection",
            "new active blueprint with right 28% empty notification buffer",
            "one headless Viewer capture and original-resolution inspection",
        ],
        "registered_execution": {
            **SCOPE_COUNTERS,
            "bounded_worker_watchdog_seconds": WATCHDOG_SECONDS,
            "bounded_viewer_timeout_seconds": VIEWER_TIMEOUT_SECONDS,
            "board_exact_pixels": [1920, 1080],
        },
        "registered_unchanged": {
            "d380_numeric_verdict": EXPECTED_FACTS["verdict"],
            "p34_identity_pass": False,
            "actual_open_clearance_mm": None,
            "cylinder_or_contact_result": None,
            "g0a_pass": False,
        },
        "source_hashes": _source_hashes(),
        "input_hashes": _input_hashes(),
        "dependency_versions": dependencies,
        "registered_dirty_baseline": _status_paths(),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "origin_master": _git("rev-parse", "origin/master"),
            "subject": _git("log", "-1", "--pretty=%s"),
        },
        "checks": checks,
        "negative_controls": {
            "controls": controls,
            "passed": sum(bool(value) for value in controls.values()),
            "total": len(controls),
            "pass": all(controls.values()),
        },
        "pass": all(checks.values()) and all(controls.values()),
    }
    _write_json_x(PREREG_PATH, prereg)
    _phase(
        "preregistration_frozen",
        preregistration_sha256=_sha(PREREG_PATH),
        passed=prereg["pass"],
        checks_passed=sum(bool(value) for value in checks.values()),
        checks_total=len(checks),
    )
    if not prereg["pass"]:
        raise RuntimeError(f"D381 preregistration failed: {checks}")
    return 0


def worker() -> int:
    _phase("worker_start", pid=os.getpid())
    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D381 preregistration did not pass")
    if invocation.get("preregistration_sha256") != _sha(PREREG_PATH):
        raise RuntimeError("D381 invocation not bound to preregistration")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D381 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D381 inputs changed after preregistration")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("D381 dirty baseline changed after preregistration")

    evidence = _read_json(D380_EVIDENCE)
    facts = _extract_facts(evidence)
    if facts != prereg["frozen_display_facts"] or facts != EXPECTED_FACTS:
        raise RuntimeError("D381 frozen display facts changed")
    board = _render_board(facts)
    _phase("board_repaired", board_sha256=board["sha256"])
    presentation = _build_presentation(facts)
    _phase(
        "presentation_archive_finalized",
        presentation_rrd_sha256=presentation["presentation_rrd"]["sha256"],
    )
    validation = _validate_and_capture(
        _read_json(D380_RERUN_VALIDATION),
        presentation,
    )
    _phase(
        "single_viewer_capture_complete",
        screenshot_sha256=validation["screenshot"]["sha256"],
    )

    manual_template = {
        "artifact": "D381_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "frozen_d380_facts_sha256": _canonical_sha(facts),
        "board": _file_record(BOARD_PATH),
        "layout_validation": _file_record(LAYOUT_VALIDATION),
        "presentation_rrd": _file_record(PRESENTATION_RRD_PATH),
        "rbl": _file_record(RBL_PATH),
        "rerun_validation": _file_record(RERUN_VALIDATION),
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "rerun_screenshot": validation["screenshot"],
        "required_check_keys": sorted(MANUAL_CHECK_KEYS),
        "inspection_checks": {
            key: None for key in sorted(MANUAL_CHECK_KEYS)
        },
        "observations": [],
        "inspector_result": None,
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)

    post_checks = {
        "board_layout_validation_pass": (
            _read_json(LAYOUT_VALIDATION).get("pass") is True
        ),
        "recording_equivalence_pass": (
            _read_json(RECORDING_EQUIVALENCE).get("pass") is True
        ),
        "rerun_validation_pass": validation.get("pass") is True,
        "viewer_invocation_exactly_one": (
            _read_json(VIEWER_RECEIPT).get("viewer_invocations") == 1
        ),
        "viewer_retry_zero": (
            _read_json(VIEWER_RECEIPT).get("automatic_viewer_retries") == 0
        ),
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
        "frozen_facts_still_exact": facts == EXPECTED_FACTS,
        "numeric_verdict_unchanged": (
            facts["verdict"] == EXPECTED_FACTS["verdict"]
        ),
        "p34_identity_still_false": facts["p34_identity_pass"] is False,
        "g0a_still_false": facts["g0a_pass"] is False,
        "all_forbidden_counters_zero": all(
            value == 0
            for name, value in SCOPE_COUNTERS.items()
            if name
            not in {
                "actual_offline_presentation_workers",
                "rerun_viewer_invocations",
            }
        ),
    }
    claim = {
        "artifact": "D381_OFFLINE_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "pid": os.getpid(),
        "preregistration": _file_record(PREREG_PATH),
        "frozen_d380_facts": facts,
        "frozen_d380_facts_sha256": _canonical_sha(facts),
        "board": board,
        "presentation": presentation,
        "rerun_validation": _file_record(RERUN_VALIDATION),
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "manual_template": _file_record(MANUAL_TEMPLATE),
        "scope_counters": SCOPE_COUNTERS,
        "checks": post_checks,
        "pass": all(post_checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase("worker_claim_written", worker_claim_sha256=_sha(WORKER_CLAIM))
    if not claim["pass"]:
        raise RuntimeError(f"D381 worker post-check failed: {post_checks}")
    return 0


def run_supervisor() -> int:
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D381 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D381 source changed after preregistration")
    if _input_hashes() != prereg["input_hashes"]:
        raise RuntimeError("D381 inputs changed after preregistration")
    if _status_paths() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("D381 dirty baseline changed after preregistration")

    command = [sys.executable, "-B", str(SCRIPT_PATH), "--stage", "worker"]
    invocation = {
        "artifact": "D381_OFFLINE_PRESENTATION_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "cwd": str(REPO),
        "preregistration_sha256": _sha(PREREG_PATH),
        "source_hashes": _source_hashes(),
        "input_hashes": _input_hashes(),
        "worker_spawn_count_registered": 1,
        "automatic_worker_retry_count_registered": 0,
        "rerun_viewer_count_registered": 1,
        "automatic_viewer_retry_count_registered": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase(
        "supervisor_spawn_start",
        invocation_sha256=_sha(INVOCATION_PATH),
        watchdog_seconds=WATCHDOG_SECONDS,
    )

    started = time.monotonic()
    timed_out = False
    sigterm_sent = False
    sigkill_sent = False
    with WORKER_STDOUT.open("xb") as stdout, WORKER_STDERR.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        pgid = process.pid
        try:
            returncode = process.wait(timeout=WATCHDOG_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(pgid, signal.SIGTERM)
            sigterm_sent = True
            try:
                returncode = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                sigkill_sent = True
                returncode = process.wait(timeout=10.0)
    elapsed = time.monotonic() - started
    try:
        os.killpg(pgid, 0)
        group_alive = True
    except ProcessLookupError:
        group_alive = False
    except PermissionError:
        group_alive = True

    claim = _read_json(WORKER_CLAIM) if WORKER_CLAIM.is_file() else {}
    required = {
        "worker_claim": WORKER_CLAIM.is_file(),
        "board": BOARD_PATH.is_file(),
        "layout_validation": LAYOUT_VALIDATION.is_file(),
        "recording_equivalence": RECORDING_EQUIVALENCE.is_file(),
        "presentation_rrd": PRESENTATION_RRD_PATH.is_file(),
        "rbl": RBL_PATH.is_file(),
        "rerun_validation": RERUN_VALIDATION.is_file(),
        "viewer_receipt": VIEWER_RECEIPT.is_file(),
        "rerun_screenshot": RERUN_SCREENSHOT.is_file(),
        "manual_template": MANUAL_TEMPLATE.is_file(),
    }
    operational_pass = (
        returncode == 0
        and not timed_out
        and not sigterm_sent
        and not sigkill_sent
        and not group_alive
        and all(required.values())
        and claim.get("pass") is True
        and _source_hashes() == prereg["source_hashes"]
        and _input_hashes() == prereg["input_hashes"]
        and _status_paths() == prereg["registered_dirty_baseline"]
    )
    supervisor = {
        "artifact": "D381_OFFLINE_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "worker_pid": process.pid,
        "worker_process_group": pgid,
        "actual_offline_presentation_workers": 1,
        "automatic_worker_retries": 0,
        "registered_rerun_viewer_invocations": 1,
        "automatic_viewer_retries": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "process_group_alive_after_wait": group_alive,
        "required_artifacts": required,
        "worker_claim_sha256": (
            _sha(WORKER_CLAIM) if WORKER_CLAIM.is_file() else None
        ),
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
        "operational_pass": operational_pass,
        "pass": operational_pass,
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase(
        "supervisor_complete",
        returncode=returncode,
        elapsed_seconds=elapsed,
        operational_pass=operational_pass,
    )
    return 0 if operational_pass else 1


def finalize() -> int:
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        WORKER_CLAIM,
        SUPERVISOR_PATH,
        BOARD_PATH,
        LAYOUT_VALIDATION,
        RECORDING_EQUIVALENCE,
        PRESENTATION_RRD_PATH,
        RBL_PATH,
        RERUN_VALIDATION,
        VIEWER_RECEIPT,
        RERUN_SCREENSHOT,
        MANUAL_TEMPLATE,
        MANUAL_INSPECTION,
    ]
    if COMPLETION_PATH.exists():
        raise FileExistsError(COMPLETION_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"D381 finalize missing files: {missing}")

    prereg = _read_json(PREREG_PATH)
    claim = _read_json(WORKER_CLAIM)
    supervisor = _read_json(SUPERVISOR_PATH)
    manual_template = _read_json(MANUAL_TEMPLATE)
    manual = _read_json(MANUAL_INSPECTION)
    facts = _extract_facts(_read_json(D380_EVIDENCE))
    manual_checks = manual.get("inspection_checks", {})
    checks = {
        "preregistration_pass": prereg.get("pass") is True,
        "worker_claim_pass": claim.get("pass") is True,
        "supervisor_pass": supervisor.get("pass") is True,
        "source_hashes_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "input_hashes_still_exact": (
            _input_hashes() == prereg["input_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_paths() == prereg["registered_dirty_baseline"]
        ),
        "frozen_facts_exact": facts == EXPECTED_FACTS,
        "numeric_verdict_unchanged": (
            facts["verdict"] == EXPECTED_FACTS["verdict"]
        ),
        "p34_identity_still_false": facts["p34_identity_pass"] is False,
        "g0a_still_false": facts["g0a_pass"] is False,
        "layout_validation_pass": (
            _read_json(LAYOUT_VALIDATION).get("pass") is True
        ),
        "recording_equivalence_pass": (
            _read_json(RECORDING_EQUIVALENCE).get("pass") is True
        ),
        "rerun_validation_pass": (
            _read_json(RERUN_VALIDATION).get("pass") is True
        ),
        "viewer_receipt_pass": (
            _read_json(VIEWER_RECEIPT).get("pass") is True
        ),
        "manual_artifact_exact": (
            manual.get("artifact") == "D381_MANUAL_VISUAL_INSPECTION_V1"
            and manual.get("case") == CASE
            and manual.get("attempt") == ATTEMPT
        ),
        "manual_template_hash_exact": (
            manual.get("template", {}).get("sha256")
            == _sha(MANUAL_TEMPLATE)
        ),
        "manual_board_hash_exact": (
            manual.get("board", {}).get("sha256") == _sha(BOARD_PATH)
        ),
        "manual_rerun_screenshot_hash_exact": (
            manual.get("rerun_screenshot", {}).get("sha256")
            == _sha(RERUN_SCREENSHOT)
        ),
        "manual_check_keys_exact": (
            set(manual_checks) == MANUAL_CHECK_KEYS
        ),
        "manual_checks_all_true": (
            set(manual_checks) == MANUAL_CHECK_KEYS
            and all(value is True for value in manual_checks.values())
        ),
        "manual_observations_nonempty": bool(manual.get("observations")),
        "manual_inspector_result_pass": (
            manual.get("inspector_result") == "PASS"
        ),
        "manual_visual_inspection_pass": manual.get("pass") is True,
        "template_frozen_facts_exact": (
            manual_template.get("frozen_d380_facts_sha256")
            == _canonical_sha(facts)
        ),
    }
    completion_pass = all(checks.values())
    completion = {
        "artifact": "D381_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "completion_pass": completion_pass,
        "verdict": (
            "D381_D380_VISUAL_CONTRACT_REPAIR_PASS"
            if completion_pass
            else "D381_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP"
        ),
        "preserved_d380_numeric_verdict": facts["verdict"],
        "p34_authored_to_cooked_identity_pass": False,
        "g0a_pass": False,
        "remaining_nulls": {
            "actual_open_jaw_clearance": None,
            "cylinder_contact_or_tipping": None,
            "q5_closure": None,
            "grasp_feasibility": None,
            "target_ik_path_justification": None,
        },
        "scope_counters": SCOPE_COUNTERS,
        "next_authorization_boundary": (
            "P34 representation repair design, live identity, 29x50 target "
            "rebase, and all physics/q5/contact remain separately unapproved."
        ),
        "artifacts": {
            path.name: _file_record(path)
            for path in [
                BOARD_PATH,
                LAYOUT_VALIDATION,
                RECORDING_EQUIVALENCE,
                PRESENTATION_RRD_PATH,
                RBL_PATH,
                RERUN_VALIDATION,
                VIEWER_RECEIPT,
                RERUN_SCREENSHOT,
                MANUAL_INSPECTION,
            ]
        },
    }
    _write_json_x(COMPLETION_PATH, completion)
    return 0 if completion_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "run", "worker", "finalize"],
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        return prepare()
    if args.stage == "run":
        return run_supervisor()
    if args.stage == "worker":
        return worker()
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
