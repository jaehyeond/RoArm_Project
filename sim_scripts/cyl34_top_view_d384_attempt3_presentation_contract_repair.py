#!/usr/bin/env python3
"""Forward-only D384 attempt3 presentation-only repair.

Attempt2 produced valid canonical design evidence, but manual inspection found
two presentation defects:

* the fixed 1920x1080 board's title/subtitle overlapped upper panel labels;
* Rerun notifications covered the decision panel and the geometry was
  over-zoomed.

This script reads only immutable attempt2 artifacts.  It does not recompute the
D384 design, author an asset, or launch Isaac/Kit/PhysX.  It projects the
existing RRD recording without its old blueprint, adds a presentation-only
summary and a fixed blueprint, and renders one headless Viewer screenshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CASE = "g0a_d384"
ATTEMPT = "attempt3_presentation_contract_repair"
OUT_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d384"
    / ATTEMPT
)
SCRIPT_PATH = Path(__file__).resolve()

SOURCE_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d384"
    / "attempt2_callback_vertex_count_field_preflight_repair"
)
SOURCE_EVIDENCE = (
    SOURCE_DIR / "d384_p34_representation_repair_design_evidence.json"
)
SOURCE_BOARD = (
    SOURCE_DIR / "d384_p34_representation_repair_design_1920x1080.png"
)
SOURCE_RRD = SOURCE_DIR / "d384_p34_representation_repair_design.rrd"
SOURCE_VALIDATION = SOURCE_DIR / "d384_rerun_validation.json"
SOURCE_MANUAL = SOURCE_DIR / "d384_manual_visual_inspection.json"
SOURCE_COMPLETION = SOURCE_DIR / "d384_completion_summary.json"

EXPECTED_SOURCE_HASHES = {
    "evidence": (
        "16ed5696d7198913367806e3ee13cf17a2b3f83c0c28d139115aa1d51c40822f"
    ),
    "board": (
        "300b7b9b8772edb42c5dd224e4beeaf337215abf0af00fe08a7bb2826972bf22"
    ),
    "rrd": (
        "a357fd47b429d7ca9097c2cd1439836b060ce7d43a564fa2328c5a285f9c6008"
    ),
    "validation": (
        "f1539fbec612baa5be964193308ca6e0dc539f838fb9ec10a304f8f05422ec2f"
    ),
    "manual": (
        "03b1aab0ff9829f154434931293b226c2b73f39f9a3bced6ed92de674cbbc189"
    ),
    "completion": (
        "7dea2155de054080ec7faa2b4f177a103ebd996c4f9ab875090431e794ef937c"
    ),
}

NEW_VARIABLES = [
    "board_reserved_header_layout_v1",
    "rerun_explicit_camera_notification_buffer_v1",
]
WATCHDOG_SECONDS = 300.0
VIEWER_TIMEOUT_SECONDS = 180.0
EXPECTED_RERUN_VERSION = "0.34.1"
RERUN_CLI = (
    REPO.parents[2]
    / "miniconda3/envs/isaaclab/bin/rerun"
)

PREREG_PATH = OUT_DIR / "d384_attempt3_preregistration.json"
PHASE_PATH = OUT_DIR / "d384_attempt3_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d384_attempt3_worker_invocation.json"
WORKER_STDOUT = OUT_DIR / "d384_attempt3_worker_stdout.log"
WORKER_STDERR = OUT_DIR / "d384_attempt3_worker_stderr.log"
WORKER_CLAIM = OUT_DIR / "d384_attempt3_worker_claim.json"
SUPERVISOR_PATH = OUT_DIR / "d384_attempt3_worker_supervisor.json"
BOARD_PATH = OUT_DIR / "d384_attempt3_presentation_board_1920x1080.png"
BOARD_LAYOUT = OUT_DIR / "d384_attempt3_board_layout_validation.json"
RECORDING_ONLY = OUT_DIR / "d384_attempt3_recording_only.rrd"
OVERLAY_RRD = OUT_DIR / "d384_attempt3_summary_overlay.rrd"
RBL_PATH = OUT_DIR / "d384_attempt3_presentation.rbl"
PRESENTATION_RRD = OUT_DIR / "d384_attempt3_presentation.rrd"
RECORDING_EQUIVALENCE = (
    OUT_DIR / "d384_attempt3_recording_equivalence.json"
)
RERUN_VALIDATION = OUT_DIR / "d384_attempt3_rerun_validation.json"
VIEWER_RECEIPT = OUT_DIR / "d384_attempt3_viewer_receipt.json"
RERUN_SCREENSHOT = OUT_DIR / "d384_attempt3_rerun_inspection.png"
MANUAL_TEMPLATE = OUT_DIR / "d384_attempt3_manual_visual_inspection_template.json"
MANUAL_INSPECTION = OUT_DIR / "d384_attempt3_manual_visual_inspection.json"
COMPLETION_PATH = OUT_DIR / "d384_attempt3_completion_summary.json"


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object JSON: {path}")
    return value


def _write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _phase(name: str, **fields: Any) -> None:
    row = {
        "phase": name,
        "monotonic_seconds": time.monotonic(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        )


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
        "width": width,
        "height": height,
        "exact_1920x1080": width == 1920 and height == 1080,
    }


def _input_paths() -> dict[str, Path]:
    return {
        "evidence": SOURCE_EVIDENCE,
        "board": SOURCE_BOARD,
        "rrd": SOURCE_RRD,
        "validation": SOURCE_VALIDATION,
        "manual": SOURCE_MANUAL,
        "completion": SOURCE_COMPLETION,
    }


def _input_hashes() -> dict[str, str]:
    return {name: _sha(path) for name, path in _input_paths().items()}


def _source_hashes() -> dict[str, str]:
    return {"attempt3_script": _sha(SCRIPT_PATH)}


def _status_lines() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.splitlines()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _run(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
            "elapsed_seconds": time.monotonic() - start,
            "ok": result.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            ),
            "stderr": (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            ),
            "timed_out": True,
            "elapsed_seconds": time.monotonic() - start,
            "ok": False,
        }


def _summary_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": result["command"],
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        "elapsed_seconds": result["elapsed_seconds"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "ok": result["ok"],
    }


def _verify_frozen_truth() -> dict[str, Any]:
    evidence = _read_json(SOURCE_EVIDENCE)
    validation = _read_json(SOURCE_VALIDATION)
    manual = _read_json(SOURCE_MANUAL)
    completion = _read_json(SOURCE_COMPLETION)
    candidates = evidence["repair_candidates"]
    checks = {
        "source_hashes_exact": _input_hashes() == EXPECTED_SOURCE_HASHES,
        "canonical_evidence_design_audit_pass": (
            evidence.get("design_audit_pass") is True
        ),
        "canonical_no_candidate_fail_stop": (
            evidence.get("verdict")
            == "D384_REPRESENTATION_REPAIR_DESIGN_NO_ADMISSIBLE_LOW_COUNT_CANDIDATE_FAIL_STOP"
            and evidence.get("repair_design_pass") is False
            and evidence.get("admissible_low_count_candidate_found") is False
        ),
        "canonical_counts_268_and_558": (
            candidates["registered_recursive_partition"][
                "total_collider_parts"
            ]
            == 268
            and candidates["exact_tetra_upper_bound"][
                "total_collider_parts"
            ]
            == 558
        ),
        "canonical_direct_path_capability_null": (
            candidates["direct_polygon_bridge_reserve"][
                "total_collider_parts_if_supported"
            ]
            == 34
            and candidates["direct_polygon_bridge_reserve"][
                "public_usd_selector_found"
            ]
            is False
            and candidates["direct_polygon_bridge_reserve"][
                "live_runtime_capability"
            ]
            is None
        ),
        "source_rerun_archive_pass": validation.get("pass") is True,
        "source_manual_visual_fail": (
            manual.get("pass") is False
            and manual["inspection_checks"][
                "board_no_text_clipping_or_overlap"
            ]
            is False
            and manual["inspection_checks"][
                "rerun_no_decision_obscuring_overlap"
            ]
            is False
        ),
        "source_completion_integrity_fail_only": (
            completion.get("operational_verdict")
            == "D384_COMPLETION_INTEGRITY_FAIL_STOP"
            and completion.get("design_verdict") == evidence.get("verdict")
        ),
        "frozen_runtime_truth": (
            evidence.get("repair_materialized") is False
            and evidence.get("live_identity_pass") is None
            and evidence.get("p34_authored_to_cooked_identity_pass") is False
            and evidence.get("g0a_pass") is False
        ),
    }
    return {
        "artifact": "D384_ATTEMPT3_FROZEN_TRUTH_PREFLIGHT_V1",
        "checks": checks,
        "pass": all(checks.values()),
    }


def prepare() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    frozen = _verify_frozen_truth()
    head = _git_value("rev-parse", "HEAD")
    origin = _git_value("rev-parse", "origin/master")
    prereg = {
        "artifact": "D384_ATTEMPT3_PRESENTATION_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "what_and_why": (
            "Repair only the two presentation defects observed in the "
            "attempt2 manual inspection while preserving its canonical "
            "design evidence bit-for-bit."
        ),
        "new_variables": NEW_VARIABLES,
        "frozen_inputs": {
            name: {
                "path": _rel(path),
                "sha256": _sha(path),
            }
            for name, path in _input_paths().items()
        },
        "source_hashes": _source_hashes(),
        "registered_dirty_baseline": _status_lines(),
        "git": {
            "head": head,
            "origin_master": origin,
            "head_equals_origin_master": head == origin,
        },
        "execution_contract": {
            "actual_offline_workers": 1,
            "automatic_worker_retries": 0,
            "headless_viewer_invocations_max": 1,
            "automatic_viewer_retries": 0,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "presentation_only": True,
        },
        "scope_counters_registered_zero": {
            "asset_or_usd_materialization": 0,
            "isaac_or_kit_launch": 0,
            "physx_or_fabric_or_hydra_runtime": 0,
            "collider_design_recomputation": 0,
            "cylinder_creation_or_render": 0,
            "physics_steps": 0,
            "q5_samples": 0,
            "contact_samples": 0,
            "target_ik_path_changes": 0,
        },
        "frozen_truth": frozen,
    }
    checks = {
        "new_variables_exactly_two": len(NEW_VARIABLES) == 2,
        "frozen_truth_pass": frozen["pass"],
        "source_hashes_exact": _input_hashes() == EXPECTED_SOURCE_HASHES,
        "head_matches_origin_master": head == origin,
        "worker_one_retry_zero": (
            prereg["execution_contract"]["actual_offline_workers"] == 1
            and prereg["execution_contract"]["automatic_worker_retries"] == 0
        ),
        "viewer_max_one_retry_zero": (
            prereg["execution_contract"][
                "headless_viewer_invocations_max"
            ]
            == 1
            and prereg["execution_contract"][
                "automatic_viewer_retries"
            ]
            == 0
        ),
        "all_forbidden_scope_counters_zero": all(
            value == 0
            for value in prereg[
                "scope_counters_registered_zero"
            ].values()
        ),
        "rerun_cli_exists": RERUN_CLI.is_file(),
    }
    prereg["checks"] = checks
    prereg["pass"] = all(checks.values())
    _write_json_x(PREREG_PATH, prereg)
    _phase(
        "preregistration_frozen",
        preregistration_sha256=_sha(PREREG_PATH),
        checks_passed=sum(checks.values()),
        checks_total=len(checks),
        passed=prereg["pass"],
    )
    if not prereg["pass"]:
        raise RuntimeError(f"attempt3 preregistration failed: {checks}")
    return 0


def _render_board(evidence: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from PIL import Image

    regular = FontProperties(
        fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    bold = FontProperties(
        fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    )
    with Image.open(SOURCE_BOARD) as source:
        source_rgb = source.convert("RGB")
        profile_crop = source_rgb.crop((174, 86, 643, 555))
        source_crop = source_rgb.crop((829, 86, 1298, 555))

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#F8FAFC")
    grid = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.985,
        bottom=0.074,
        top=0.855,
        width_ratios=[1.0, 1.0, 0.62],
        height_ratios=[1.05, 0.95],
        wspace=0.22,
        hspace=0.27,
    )

    profile_axis = fig.add_subplot(grid[0, 0])
    profile_axis.imshow(profile_crop)
    profile_axis.axis("off")
    profile_axis.set_title(
        "프로필 프리즘 증거: 원본 1개 → 정확한 삼각 프리즘 5개",
        fontproperties=bold,
        fontsize=11.0,
        pad=9,
    )

    source_axis = fig.add_subplot(grid[0, 1])
    source_axis.imshow(source_crop)
    source_axis.axis("off")
    source_axis.set_title(
        "가장 나쁜 3차원 hull 증거: 사면체 74개 → 최대 8꼭짓점 셀 34개",
        fontproperties=bold,
        fontsize=11.0,
        pad=9,
    )

    candidates = evidence["repair_candidates"]
    decision_axis = fig.add_subplot(grid[0, 2])
    decision_axis.axis("off")
    decision_lines = [
        "설계 판정",
        "",
        "후보 R0 — 점+면 직접 입력",
        "• 이론상 전체 34개",
        "• 입력조건 17/17 통과",
        "• 공개 USD 선택기 없음",
        "• 새 C++/비공개 연결층 필요",
        "",
        "후보 R1 — 작성 형상 정확 분할",
        "• 프로필 자식 46개",
        "• 3차원 자식 205개",
        "• 전체 268개",
        "• 128개 미만 조건 실패 → 기각",
        "",
        "후보 R2 — 정확 사면체 상한",
        "• 전체 558개",
        "• 128개 미만 조건 실패 → 기각",
        "",
        "실제 asset은 만들지 않았습니다.",
        "P34 live identity는 여전히 미통과입니다.",
    ]
    decision_text = decision_axis.text(
        0.03,
        0.97,
        "\n".join(decision_lines),
        va="top",
        ha="left",
        fontproperties=regular,
        fontsize=9.7,
        linespacing=1.24,
        color="#172033",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#EFF6FF",
            "edgecolor": "#60A5FA",
        },
    )

    rows = sorted(
        evidence["failed_parts"],
        key=lambda item: item["surface_inward_mm"],
    )
    bar_axis = fig.add_subplot(grid[1, 0:2])
    labels = [
        ("L5/" if row["body"] == "link5" else "GR/")
        + row["prim_name"].split("_", 1)[0]
        + (
            "/P"
            if row["semantic_class"] == "manual_profile_prism"
            else "/H"
        )
        for row in rows
    ]
    values = [row["surface_inward_mm"] for row in rows]
    colors = [
        "#F28E2B"
        if row["semantic_class"] == "manual_profile_prism"
        else "#4E79A7"
        for row in rows
    ]
    positions = list(range(len(rows)))
    bar_axis.barh(positions, values, color=colors, alpha=0.92)
    bar_axis.axvline(
        0.1,
        color="#991B1B",
        linestyle="--",
        linewidth=1.5,
        label="동결 허용선 0.1 mm",
    )
    bar_axis.set_yticks(positions)
    bar_axis.set_yticklabels(
        labels,
        fontproperties=regular,
        fontsize=7.7,
    )
    bar_axis.set_xlabel(
        "작성 형상에서 cooked 형상 안쪽까지 사라진 거리 (mm)",
        fontproperties=regular,
        fontsize=9.5,
    )
    bar_axis.set_title(
        "D380에서 확인된 실패 17개: 프로필 9개(주황) + 3차원 hull 8개(파랑)",
        fontproperties=bold,
        fontsize=10.8,
    )
    bar_axis.grid(axis="x", alpha=0.20)
    legend = bar_axis.legend(
        loc="lower right",
        prop=regular,
        fontsize=8.5,
    )

    gate_axis = fig.add_subplot(grid[1, 2])
    gate_axis.axis("off")
    gate_lines = [
        "동결한 검사 기준",
        "",
        "• 표면 오차: 0.1 mm",
        "• 부피 오차: 0.5%",
        "• 음성 대조군: 8/8 통과",
        "• 금지한 실행 항목: 모두 0",
        "",
        "이번 PASS의 뜻",
        "계산 과정과 증거 계보가",
        "감사 가능하다는 뜻입니다.",
        "",
        "PhysX가 형상을 보존한다는",
        "뜻은 아닙니다.",
        "",
        "결론",
        f"R1 전체 {candidates['registered_recursive_partition']['total_collider_parts']}개",
        f"R2 전체 {candidates['exact_tetra_upper_bound']['total_collider_parts']}개",
        "낮은 개수의 허용 후보 없음",
        "g0a_pass=false",
    ]
    gate_text = gate_axis.text(
        0.04,
        0.96,
        "\n".join(gate_lines),
        va="top",
        ha="left",
        fontproperties=regular,
        fontsize=10.0,
        linespacing=1.28,
        color="#1F2937",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F0FDF4",
            "edgecolor": "#4ADE80",
        },
    )

    title = fig.text(
        0.5,
        0.974,
        "D384 | P34 실패 부품의 최소변경 표현 수리 설계",
        ha="center",
        va="top",
        fontproperties=bold,
        fontsize=20.0,
        color="#0F172A",
    )
    subtitle = fig.text(
        0.5,
        0.932,
        (
            "D379/D380 동결 증거만 사용 · Isaac/PhysX/USD/원통/물리/q5/접촉 "
            "실행 0 · 실제 충돌체 생성 0"
        ),
        ha="center",
        va="top",
        fontproperties=regular,
        fontsize=10.5,
        color="#475569",
    )
    footer = fig.text(
        0.5,
        0.020,
        (
            "정확 분할은 268개 또는 558개로 늘어 128개 미만 조건을 통과하지 "
            "못했습니다. 34개 직접 경로는 공개 USD 연결 수단이 없어 예비안입니다."
        ),
        ha="center",
        va="bottom",
        fontproperties=regular,
        fontsize=9.5,
        color="#334155",
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_box = title.get_window_extent(renderer)
    subtitle_box = subtitle.get_window_extent(renderer)
    panel_top_px = max(
        profile_axis.get_position().y1,
        source_axis.get_position().y1,
        decision_axis.get_position().y1,
    ) * fig.bbox.height
    decision_box = decision_text.get_window_extent(renderer)
    decision_axes_box = decision_axis.get_window_extent(renderer)
    gate_box = gate_text.get_window_extent(renderer)
    gate_axes_box = gate_axis.get_window_extent(renderer)
    footer_box = footer.get_window_extent(renderer)
    bar_axes_box = bar_axis.get_window_extent(renderer)
    layout_checks = {
        "title_subtitle_do_not_overlap": not title_box.overlaps(
            subtitle_box
        ),
        "subtitle_above_panels": subtitle_box.y0 > panel_top_px,
        "decision_text_inside_panel": (
            decision_axes_box.contains(*decision_box.get_points()[0])
            and decision_axes_box.contains(*decision_box.get_points()[1])
        ),
        "gate_text_inside_panel": (
            gate_axes_box.contains(*gate_box.get_points()[0])
            and gate_axes_box.contains(*gate_box.get_points()[1])
        ),
        "footer_below_bar_panel": footer_box.y1 < bar_axes_box.y0,
        "legend_present": legend is not None,
    }
    if not all(layout_checks.values()):
        plt.close(fig)
        raise RuntimeError(f"attempt3 board layout failed: {layout_checks}")
    fig.savefig(
        BOARD_PATH,
        dpi=100,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    board = _png_record(BOARD_PATH)
    layout = {
        "artifact": "D384_ATTEMPT3_BOARD_LAYOUT_VALIDATION_V1",
        "source_board": _file_record(SOURCE_BOARD),
        "board": board,
        "checks": {
            **layout_checks,
            "exact_1920x1080": board["exact_1920x1080"],
        },
        "pass": all(layout_checks.values())
        and board["exact_1920x1080"],
    }
    _write_json_x(BOARD_LAYOUT, layout)
    return {
        **board,
        "layout_validation": _file_record(BOARD_LAYOUT),
    }


def _build_blueprint() -> Any:
    import rerun.blueprint as rrb

    def spatial(
        contents: str,
        name: str,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> Any:
        return rrb.Spatial3DView(
            origin="/",
            contents=contents,
            name=name,
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=position,
                look_target=target,
                eye_up=(0.0, 0.0, 1.0),
            ),
            spatial_information=rrb.SpatialInformation(
                target_frame="tf#/",
                show_axes=False,
                show_bounding_box=False,
            ),
        )

    geometry = rrb.Horizontal(
        spatial(
            "/d384/profile/**",
            "프로필 프리즘 | 원본과 정확 분할 자식",
            (0.085, -0.075, 0.030),
            (0.0328, -0.0052, -0.0319),
        ),
        spatial(
            "/d384/source/**",
            "3차원 hull | 재귀 분할 증거",
            (0.105, -0.140, 0.145),
            (-0.0174, 0.0160, 0.0618),
        ),
        column_shares=[0.5, 0.5],
    )
    decision_area = rrb.Vertical(
        geometry,
        rrb.TextDocumentView(
            origin="/presentation/d384_attempt3/summary",
            contents="/presentation/d384_attempt3/summary",
            name="D384 동결 계산 결과와 다음 경계",
        ),
        row_shares=[0.72, 0.28],
    )
    notification_buffer = rrb.Spatial3DView(
        origin="/",
        contents="/presentation/d384_attempt3/notification_buffer/**",
        name="알림 전용 여백 | 판정 내용 없음",
        spatial_information=rrb.SpatialInformation(
            target_frame="tf#/",
            show_axes=False,
            show_bounding_box=False,
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            decision_area,
            notification_buffer,
            column_shares=[0.74, 0.26],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def _overlay_markdown(evidence: dict[str, Any]) -> str:
    candidates = evidence["repair_candidates"]
    return "\n".join(
        [
            "## D384 동결 계산 결과",
            "",
            "- 실패 부품: **17/34** (프로필 프리즘 9, 3차원 hull 8)",
            (
                "- 정확 작성형상 분할: **"
                f"{candidates['registered_recursive_partition']['total_collider_parts']}"
                "개** → 128개 미만 조건 실패"
            ),
            (
                "- 정확 사면체 상한: **"
                f"{candidates['exact_tetra_upper_bound']['total_collider_parts']}"
                "개** → 128개 미만 조건 실패"
            ),
            (
                "- 점+면 직접 입력: 이론상 **34개**, 공개 USD 선택기 없음, "
                "runtime capability는 **NULL**"
            ),
            "- 실제 asset/Isaac/PhysX/원통/물리/q5/접촉 실행: **0**",
            "- P34 authored→cooked identity: **FAIL**",
            "- g0a_pass=false",
            "",
            (
                "**다음 별도 승인 경계:** 8개 source hull의 의미 기반 "
                "저개수 재설계. 실제 asset materialization은 아직 금지."
            ),
        ]
    )


def _loopback_preflight() -> dict[str, Any]:
    start = time.monotonic()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            address = sock.getsockname()
        return {
            "address": address[0],
            "ephemeral_port": address[1],
            "bind_ok": True,
            "elapsed_seconds": time.monotonic() - start,
        }
    except OSError as exc:
        return {
            "address": "127.0.0.1",
            "ephemeral_port": None,
            "bind_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - start,
        }


def _build_presentation(evidence: dict[str, Any]) -> dict[str, Any]:
    import rerun as rr
    from rerun.experimental import RrdReader
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if str(rr.__version__) != EXPECTED_RERUN_VERSION:
        raise RuntimeError(f"Rerun SDK drift: {rr.__version__}")
    source_validation = _read_json(SOURCE_VALIDATION)
    source_reader = RrdReader(SOURCE_RRD)
    recordings = source_reader.recordings()
    blueprints = source_reader.blueprints()
    if len(recordings) != 1 or len(blueprints) != 1:
        raise RuntimeError(
            "source RRD inventory drifted: "
            f"recordings={recordings}, blueprints={blueprints}"
        )
    recording = recordings[0]
    if (
        recording.application_id != "roarm_g0a_d384_repair_design"
        or recording.recording_id != "g0a_d384_repair_design"
    ):
        raise RuntimeError(f"source recording identity drifted: {recording}")
    source_summary = source_reader.store(store=recording).summary()
    source_reader.stream(store=recording).write_rrd(
        RECORDING_ONLY,
        application_id=recording.application_id,
        recording_id=recording.recording_id,
    )
    derived_reader = RrdReader(RECORDING_ONLY)
    derived_recordings = derived_reader.recordings()
    derived_blueprints = derived_reader.blueprints()
    if len(derived_recordings) != 1 or derived_blueprints:
        raise RuntimeError(
            "recording-only projection has unexpected store inventory"
        )
    derived_summary = derived_reader.store(
        store=derived_recordings[0]
    ).summary()
    compare = _run(
        [
            str(RERUN_CLI),
            "rrd",
            "compare",
            "--unordered",
            str(SOURCE_RRD),
            str(RECORDING_ONLY),
        ],
        timeout=90.0,
    )
    equivalence_checks = {
        "source_one_recording_one_blueprint": (
            len(recordings) == 1 and len(blueprints) == 1
        ),
        "derived_one_recording_zero_blueprints": (
            len(derived_recordings) == 1 and not derived_blueprints
        ),
        "recording_summary_exact": source_summary == derived_summary,
        "rrd_compare_return_zero": compare["ok"],
    }
    equivalence = {
        "artifact": "D384_ATTEMPT3_RECORDING_EQUIVALENCE_V1",
        "source": _file_record(SOURCE_RRD),
        "recording_only": _file_record(RECORDING_ONLY),
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
        "source_summary_sha256": _canonical_sha(source_summary),
        "derived_summary_sha256": _canonical_sha(derived_summary),
        "rrd_compare": _summary_result(compare),
        "checks": equivalence_checks,
        "pass": all(equivalence_checks.values()),
    }
    _write_json_x(RECORDING_EQUIVALENCE, equivalence)
    if not equivalence["pass"]:
        raise RuntimeError(
            f"recording projection failed: {equivalence_checks}"
        )

    with rr.RecordingStream(
        recording.application_id,
        recording_id=recording.recording_id,
        make_default=False,
        send_properties=True,
    ) as overlay:
        overlay.save(str(OVERLAY_RRD), write_footer=True)
        overlay.log(
            "presentation/d384_attempt3/summary",
            rr.TextDocument(
                _overlay_markdown(evidence),
                media_type="text/markdown",
            ),
            static=True,
        )
        overlay.flush(timeout_sec=30.0)
    blueprint = _build_blueprint()
    blueprint.save(recording.application_id, RBL_PATH)
    merge = _run(
        [
            str(RERUN_CLI),
            "rrd",
            "merge",
            "-o",
            str(PRESENTATION_RRD),
            str(RECORDING_ONLY),
            str(OVERLAY_RRD),
            str(RBL_PATH),
        ],
        timeout=120.0,
    )
    if not (
        merge["ok"]
        and PRESENTATION_RRD.is_file()
        and PRESENTATION_RRD.stat().st_size > 0
    ):
        raise RuntimeError(f"presentation merge failed: {merge}")

    expected_entities = list(
        source_validation["expected_entity_paths"]
    ) + ["/presentation/d384_attempt3/summary"]
    expected_components = dict(
        source_validation["expected_entity_components"]
    )
    expected_components[
        "/presentation/d384_attempt3/summary"
    ] = ["TextDocument:text"]
    archive = validate_rerun_artifact(
        PRESENTATION_RRD,
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=expected_components,
        blueprint_path=RBL_PATH,
        screenshot_path=None,
        cli_path=RERUN_CLI,
        expected_version=EXPECTED_RERUN_VERSION,
        timeout_s=180.0,
    )
    if archive.get("pass") is not True:
        raise RuntimeError(f"strict Rerun archive validation failed: {archive}")

    loopback = _loopback_preflight()
    if not loopback["bind_ok"]:
        raise RuntimeError(f"loopback preflight failed: {loopback}")
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
        str(PRESENTATION_RRD),
    ]
    viewer_env = dict(os.environ)
    viewer_env["RERUN_ANALYTICS_ENABLED"] = "false"
    viewer = _run(
        viewer_command,
        timeout=VIEWER_TIMEOUT_SECONDS,
        env=viewer_env,
    )
    screenshot = (
        _png_record(RERUN_SCREENSHOT)
        if RERUN_SCREENSHOT.is_file()
        else {"path": _rel(RERUN_SCREENSHOT), "exists": False}
    )
    combined_output = (
        f"{viewer.get('stdout', '')}\n{viewer.get('stderr', '')}"
    ).lower()
    viewer_checks = {
        "loopback_preflight_pass": loopback["bind_ok"],
        "viewer_return_zero": viewer["returncode"] == 0,
        "viewer_not_timed_out": viewer["timed_out"] is False,
        "screenshot_nonempty": (
            RERUN_SCREENSHOT.is_file()
            and RERUN_SCREENSHOT.stat().st_size > 0
        ),
        "message_proxy_crash_absent": (
            "message proxy server crashed" not in combined_output
        ),
        "operation_not_permitted_absent": (
            "operation not permitted" not in combined_output
        ),
        "viewer_invocations_exactly_one": True,
        "automatic_viewer_retries_zero": True,
        "viewer_command_has_loopback_bind": (
            "--bind" in viewer_command
            and "127.0.0.1" in viewer_command
            and "--port" in viewer_command
            and "auto" in viewer_command
        ),
    }
    receipt = {
        "artifact": "D384_ATTEMPT3_VIEWER_RECEIPT_V1",
        "loopback_preflight": loopback,
        "authorized_viewer_invocations_max": 1,
        "actual_viewer_invocations": 1,
        "automatic_viewer_retries": 0,
        "command": viewer_command,
        "result": _summary_result(viewer),
        "screenshot": screenshot,
        "screenshot_dimension_policy": (
            "The Viewer command uses a 1920x1080 logical window. Native "
            "HiDPI physical pixels are recorded without resampling; only the "
            "static board is required to be exactly 1920x1080 pixels."
        ),
        "checks": viewer_checks,
        "pass": all(viewer_checks.values()),
    }
    _write_json_x(VIEWER_RECEIPT, receipt)
    if not receipt["pass"]:
        raise RuntimeError(f"Viewer contract failed: {viewer_checks}")

    validation_checks = {
        "recording_equivalence_pass": equivalence["pass"],
        "strict_archive_validation_pass": archive.get("pass") is True,
        "viewer_receipt_pass": receipt["pass"],
        "headless_render_not_duplicated_by_strict_helper": (
            archive["headless_render"]["attempted"] is False
        ),
        "canonical_source_rrd_unchanged": (
            _sha(SOURCE_RRD) == EXPECTED_SOURCE_HASHES["rrd"]
        ),
    }
    validation = {
        "artifact": "D384_ATTEMPT3_RERUN_VALIDATION_V1",
        "recording_equivalence": _file_record(RECORDING_EQUIVALENCE),
        "archive_validation": archive,
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "presentation_rrd": _file_record(PRESENTATION_RRD),
        "rbl": _file_record(RBL_PATH),
        "screenshot": screenshot,
        "checks": validation_checks,
        "pass": all(validation_checks.values()),
    }
    _write_json_x(RERUN_VALIDATION, validation)
    if not validation["pass"]:
        raise RuntimeError(
            f"combined Rerun validation failed: {validation_checks}"
        )
    return validation


def worker() -> int:
    _phase("worker_start", pid=os.getpid())
    prereg = _read_json(PREREG_PATH)
    invocation = _read_json(INVOCATION_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("preregistration not passed")
    if invocation.get("preregistration_sha256") != _sha(PREREG_PATH):
        raise RuntimeError("invocation/preregistration hash mismatch")
    if _input_hashes() != {
        name: row["sha256"]
        for name, row in prereg["frozen_inputs"].items()
    }:
        raise RuntimeError("attempt3 frozen input changed")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("attempt3 script changed after preregistration")
    if _status_lines() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("dirty baseline changed after preregistration")

    evidence = _read_json(SOURCE_EVIDENCE)
    board = _render_board(evidence)
    _phase(
        "fixed_board_written",
        board_sha256=board["sha256"],
        exact_1920x1080=board["exact_1920x1080"],
    )
    rerun = _build_presentation(evidence)
    _phase(
        "fixed_rerun_presentation_written",
        rrd_sha256=rerun["presentation_rrd"]["sha256"],
        screenshot_sha256=rerun["screenshot"]["sha256"],
    )
    manual_template = {
        "artifact": "D384_ATTEMPT3_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "canonical_evidence": _file_record(SOURCE_EVIDENCE),
        "board": board,
        "presentation_rrd": _file_record(PRESENTATION_RRD),
        "rbl": _file_record(RBL_PATH),
        "rerun_validation": _file_record(RERUN_VALIDATION),
        "viewer_receipt": _file_record(VIEWER_RECEIPT),
        "rerun_screenshot": rerun["screenshot"],
        "required_check_keys": [
            "board_exact_1920x1080_and_legible",
            "board_no_text_clipping_or_overlap",
            "board_counts_match_canonical_json",
            "profile_parent_and_children_whole_shape_visible",
            "source_hull_whole_shape_visible",
            "decision_summary_visible",
            "notifications_confined_to_empty_buffer",
            "rerun_no_unknown_timeline",
            "rerun_no_decision_obscuring_overlap",
        ],
        "inspection_detail_required": "original",
        "inspection_checks": {},
        "observations": [],
        "inspector_result": None,
        "pass": None,
    }
    _write_json_x(MANUAL_TEMPLATE, manual_template)
    checks = {
        "canonical_evidence_hash_unchanged": (
            _sha(SOURCE_EVIDENCE) == EXPECTED_SOURCE_HASHES["evidence"]
        ),
        "board_exact_1920x1080": board["exact_1920x1080"],
        "board_layout_validation_pass": (
            _read_json(BOARD_LAYOUT)["pass"] is True
        ),
        "rerun_validation_pass": rerun["pass"] is True,
        "viewer_exactly_one_retry_zero": (
            _read_json(VIEWER_RECEIPT)[
                "actual_viewer_invocations"
            ]
            == 1
            and _read_json(VIEWER_RECEIPT)[
                "automatic_viewer_retries"
            ]
            == 0
        ),
        "frozen_design_verdict_preserved": (
            evidence["verdict"]
            == "D384_REPRESENTATION_REPAIR_DESIGN_NO_ADMISSIBLE_LOW_COUNT_CANDIDATE_FAIL_STOP"
        ),
        "no_asset_or_live_claim": (
            evidence["repair_materialized"] is False
            and evidence["live_identity_pass"] is None
            and evidence["g0a_pass"] is False
        ),
        "frozen_inputs_still_exact": _input_hashes()
        == {
            name: row["sha256"]
            for name, row in prereg["frozen_inputs"].items()
        },
        "source_hash_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_lines() == prereg["registered_dirty_baseline"]
        ),
    }
    claim = {
        "artifact": "D384_ATTEMPT3_PRESENTATION_WORKER_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "canonical_design_verdict": evidence["verdict"],
        "presentation_repair_only": True,
        "scope_counters_actual": {
            "asset_or_usd_materialization": 0,
            "isaac_or_kit_launch": 0,
            "physx_or_fabric_or_hydra_runtime": 0,
            "collider_design_recomputation": 0,
            "cylinder_creation_or_render": 0,
            "physics_steps": 0,
            "q5_samples": 0,
            "contact_samples": 0,
            "target_ik_path_changes": 0,
            "headless_rerun_viewer_invocations": 1,
        },
        "board": board,
        "rerun_validation": _file_record(RERUN_VALIDATION),
        "manual_template": _file_record(MANUAL_TEMPLATE),
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(WORKER_CLAIM, claim)
    _phase(
        "worker_claim_written",
        worker_claim_sha256=_sha(WORKER_CLAIM),
        passed=claim["pass"],
    )
    if not claim["pass"]:
        raise RuntimeError(f"attempt3 worker checks failed: {checks}")
    return 0


def run_supervisor() -> int:
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("preregistration not passed")
    if _input_hashes() != {
        name: row["sha256"]
        for name, row in prereg["frozen_inputs"].items()
    }:
        raise RuntimeError("attempt3 frozen input changed")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("attempt3 script changed")
    if _status_lines() != prereg["registered_dirty_baseline"]:
        raise RuntimeError("dirty baseline changed")
    command = [
        sys.executable,
        "-B",
        str(SCRIPT_PATH),
        "--stage",
        "worker",
    ]
    invocation = {
        "artifact": "D384_ATTEMPT3_WORKER_INVOCATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "cwd": str(REPO),
        "preregistration_sha256": _sha(PREREG_PATH),
        "worker_spawn_count_registered": 1,
        "automatic_retry_count_registered": 0,
        "watchdog_seconds": WATCHDOG_SECONDS,
    }
    _write_json_x(INVOCATION_PATH, invocation)
    _phase(
        "supervisor_spawn_start",
        invocation_sha256=_sha(INVOCATION_PATH),
        watchdog_seconds=WATCHDOG_SECONDS,
    )
    start = time.monotonic()
    timed_out = False
    sigterm_sent = False
    sigkill_sent = False
    with WORKER_STDOUT.open("xb") as stdout, WORKER_STDERR.open(
        "xb"
    ) as stderr:
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
    elapsed = time.monotonic() - start
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
        "board_layout": BOARD_LAYOUT.is_file(),
        "recording_only": RECORDING_ONLY.is_file(),
        "overlay_rrd": OVERLAY_RRD.is_file(),
        "rbl": RBL_PATH.is_file(),
        "presentation_rrd": PRESENTATION_RRD.is_file(),
        "recording_equivalence": RECORDING_EQUIVALENCE.is_file(),
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
        and _input_hashes()
        == {
            name: row["sha256"]
            for name, row in prereg["frozen_inputs"].items()
        }
        and _source_hashes() == prereg["source_hashes"]
        and _status_lines() == prereg["registered_dirty_baseline"]
    )
    supervisor = {
        "artifact": "D384_ATTEMPT3_WORKER_SUPERVISOR_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "command": command,
        "worker_pid": process.pid,
        "actual_offline_worker_invocations": 1,
        "automatic_retries": 0,
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
        "stdout": _file_record(WORKER_STDOUT),
        "stderr": _file_record(WORKER_STDERR),
        "pass": operational_pass,
    }
    _write_json_x(SUPERVISOR_PATH, supervisor)
    _phase(
        "supervisor_complete",
        supervisor_sha256=_sha(SUPERVISOR_PATH),
        returncode=returncode,
        elapsed_seconds=elapsed,
        passed=operational_pass,
    )
    if not operational_pass:
        raise RuntimeError(f"attempt3 supervisor failed: {supervisor}")
    return 0


def finalize() -> int:
    required = [
        PREREG_PATH,
        INVOCATION_PATH,
        WORKER_CLAIM,
        SUPERVISOR_PATH,
        BOARD_PATH,
        BOARD_LAYOUT,
        PRESENTATION_RRD,
        RBL_PATH,
        RERUN_VALIDATION,
        VIEWER_RECEIPT,
        RERUN_SCREENSHOT,
        MANUAL_TEMPLATE,
        MANUAL_INSPECTION,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"attempt3 finalize missing: {missing}")
    prereg = _read_json(PREREG_PATH)
    claim = _read_json(WORKER_CLAIM)
    supervisor = _read_json(SUPERVISOR_PATH)
    manual = _read_json(MANUAL_INSPECTION)
    template = _read_json(MANUAL_TEMPLATE)
    checks_map = manual.get("inspection_checks") or {}
    required_manual = set(template["required_check_keys"])
    evidence = _read_json(SOURCE_EVIDENCE)
    checks = {
        "preregistration_pass": prereg.get("pass") is True,
        "worker_claim_pass": claim.get("pass") is True,
        "supervisor_pass": supervisor.get("pass") is True,
        "worker_one_retry_zero": (
            supervisor.get("actual_offline_worker_invocations") == 1
            and supervisor.get("automatic_retries") == 0
        ),
        "no_timeout_signal_or_residue": (
            supervisor.get("timed_out") is False
            and supervisor.get("sigterm_sent") is False
            and supervisor.get("sigkill_sent") is False
            and supervisor.get("process_group_alive_after_wait") is False
        ),
        "manual_keys_exact": set(checks_map) == required_manual,
        "manual_checks_all_true": all(
            checks_map.get(name) is True for name in required_manual
        ),
        "manual_pass": manual.get("pass") is True,
        "manual_board_hash_exact": (
            manual.get("board", {}).get("sha256") == _sha(BOARD_PATH)
        ),
        "manual_rerun_hash_exact": (
            manual.get("rerun_screenshot", {}).get("sha256")
            == _sha(RERUN_SCREENSHOT)
        ),
        "canonical_design_verdict_unchanged": (
            evidence["verdict"]
            == "D384_REPRESENTATION_REPAIR_DESIGN_NO_ADMISSIBLE_LOW_COUNT_CANDIDATE_FAIL_STOP"
        ),
        "runtime_truth_unchanged": (
            evidence["repair_materialized"] is False
            and evidence["live_identity_pass"] is None
            and evidence["p34_authored_to_cooked_identity_pass"] is False
            and evidence["g0a_pass"] is False
        ),
        "input_hashes_still_exact": _input_hashes()
        == {
            name: row["sha256"]
            for name, row in prereg["frozen_inputs"].items()
        },
        "source_hash_still_exact": (
            _source_hashes() == prereg["source_hashes"]
        ),
        "dirty_baseline_still_exact": (
            _status_lines() == prereg["registered_dirty_baseline"]
        ),
    }
    completion = {
        "artifact": "D384_ATTEMPT3_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "checks": checks,
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "canonical_design_verdict": evidence["verdict"],
        "presentation_verdict": (
            "D384_PRESENTATION_CONTRACT_REPAIRED_PASS"
            if all(checks.values())
            else "D384_ATTEMPT3_PRESENTATION_INTEGRITY_FAIL_STOP"
        ),
        "remaining_truth": {
            "admissible_low_count_candidate_found": False,
            "repair_materialized": False,
            "live_identity_pass": None,
            "p34_identity_pass": False,
            "cylinder_29x50_rendered_or_measured": False,
            "g0a_pass": False,
        },
        "artifacts": {
            "canonical_evidence": _file_record(SOURCE_EVIDENCE),
            "board": _png_record(BOARD_PATH),
            "presentation_rrd": _file_record(PRESENTATION_RRD),
            "rbl": _file_record(RBL_PATH),
            "rerun_screenshot": _png_record(RERUN_SCREENSHOT),
            "manual_inspection": _file_record(MANUAL_INSPECTION),
            "supervisor": _file_record(SUPERVISOR_PATH),
        },
        "next_authorization_boundary": (
            "A separate approved offline case may redesign the eight source "
            "hulls with semantic primitive/low-count parts. Asset "
            "materialization, Isaac/live identity, and cylinder physics "
            "remain unapproved."
        ),
        "pass": all(checks.values()),
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase(
        "completion_written",
        completion_sha256=_sha(COMPLETION_PATH),
        passed=completion["pass"],
    )
    if not completion["pass"]:
        raise RuntimeError(f"attempt3 completion failed: {checks}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "run", "worker", "finalize"),
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
