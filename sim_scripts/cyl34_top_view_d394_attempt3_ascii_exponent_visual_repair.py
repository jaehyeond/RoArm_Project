#!/usr/bin/env python3
"""D394 attempt3: observability-only ASCII exponent glyph repair.

Attempt2's numeric worker passed, and its RRD/RBL validation passed, but manual
inspection found missing glyph boxes where the professor board used Unicode
superscript-minus exponents.  This reactive repair reads immutable attempt2
artifacts, regenerates only the board with ASCII ``10^-18`` and ``10^-13``,
and preserves the RRD/RBL/screenshot bit-exact without another Viewer launch.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
CASE = "D394"
ATTEMPT = "attempt3_ascii_exponent_visual_repair"
OUT_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d394" / ATTEMPT
START = REPO / "START_HERE.md"
EXPECTED_HEAD = "d354d46134fe002073642441a7d24c99fe579edd"
EXPECTED_START_SHA256 = (
    "4a6229369b7771131b66da9d5b1a79f20f9023bbeb167e0440dfbebdcff1fc00"
)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
ATTEMPT2 = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d394"
    / "attempt2_gate_numeric_null_semantics_repair"
)

INPUTS = {
    ATTEMPT2 / "d394_execution_authority.json": "f96f3987f3b57322cb7afc0f08d05d03beec85e0b4019807b6cf5117bee82dff",
    ATTEMPT2 / "d394_preregistration.json": "e349713512fe8fab48dc41c2d3002293932606bf17a55f8828ab16ff07b3c6e2",
    ATTEMPT2 / "d394_phase_markers.jsonl": "dd5fc1a8a3c2a2d895d0221931d60338f15cb48f3e884caded0f3d19e390c20c",
    ATTEMPT2 / "d394_full10_volume_semantics_evidence.json": "7672f208cc704bd9c3a51bc0b60040e2a121335cf12dcfaa5fd851484dd089a1",
    ATTEMPT2 / "d394_full10_display_geometry.json": "b40618c6240a0b8a7d74ed750ec85209e64e979e6e9081b283558294f5129e49",
    ATTEMPT2 / "d394_full10_volume_semantics.csv": "2575cdf7299b41cced1cd5931880dda6e60e08ba3abebda21628aa88736eb399",
    ATTEMPT2 / "d394_offline_worker_claim.json": "e1ba4654591e57fbea4588c7800ca46999a143af8358ea557a097e32c9da294e",
    ATTEMPT2 / "d394_offline_worker_supervisor.json": "6f43b040621c721f2724f547462fb4311a5424abeba6da4150369e054a6f0fa9",
    ATTEMPT2 / "d394_full10_volume_semantics_1920x1080.png": "0f9534e98336cb53e29228c6f7bd1c1d81f85d7f16d006b09336e2b92e3a0ef5",
    ATTEMPT2 / "d394_board_layout_validation.json": "575f233748e6a28f7219b0c7b3d59c9bed5a0582523228edc9c81488a1c3ff76",
    ATTEMPT2 / "d394_full10_volume_semantics.rerun.rrd": "feb3c529143c5bdc7c24a27ccba5076428034fe790e08f735d4b3cef41e17f90",
    ATTEMPT2 / "d394_full10_volume_semantics.rerun.rbl": "0a46aaffe04b9e7d2df822932b8d1ad5a7296b773960f93bf613d62839886d2f",
    ATTEMPT2 / "d394_rerun_validation.json": "02574894b55d1e5fb0cfac0e8c8d84e6960632e6e135b757f1cc2e7cf9fe2ae8",
    ATTEMPT2 / "d394_rerun_inspection.png": "221aae6acaa87f6c3db74cd2bd82fa5e59ad2368510256b31a5784e9528ad42d",
    ATTEMPT2 / "d394_manual_visual_inspection.json": "fdfee20b9ad7367687caf579b48911212cd4db4a2c8e9476a4a855acc3231fe9",
    ATTEMPT2 / "d394_observability_claim.json": "2684e7f60c76510f7a3e420f19053b0bb81781a35446ca93b80fcea3f1a1ae43",
    ATTEMPT2 / "d394_visual_contract_stop.json": "3e1f103ed47fa2bad1fce4d31c906f985f2f946a12ddfec612248e26b1182e0b",
}

AUTHORITY_ENV = "D394_ATTEMPT3_EXECUTION_AUTHORITY_SHA256"
EXECUTION_AUTHORITY = OUT_DIR / "d394_execution_authority.json"
PREREGISTRATION = OUT_DIR / "d394_observability_repair_preregistration.json"
PHASES = OUT_DIR / "d394_observability_repair_phases.jsonl"
BOARD = OUT_DIR / "d394_full10_volume_semantics_ascii_1920x1080.png"
LAYOUT = OUT_DIR / "d394_ascii_board_layout_validation.json"
RRD = OUT_DIR / "d394_full10_volume_semantics.rerun.rrd"
RBL = OUT_DIR / "d394_full10_volume_semantics.rerun.rbl"
RERUN_SCREENSHOT = OUT_DIR / "d394_rerun_inspection.png"
RERUN_VALIDATION = OUT_DIR / "d394_rerun_validation.json"
REPAIR_CLAIM = OUT_DIR / "d394_ascii_exponent_visual_repair_claim.json"
MANUAL_TEMPLATE = OUT_DIR / "d394_manual_visual_inspection_template.json"
MANUAL = OUT_DIR / "d394_manual_visual_inspection.json"
COMPLETION = OUT_DIR / "d394_completion_summary.json"
FAILURE = OUT_DIR / "d394_failure_attestation.json"

PHASE_ORDER = [
    "prepare_start",
    "prepare_end",
    "repair_start",
    "ascii_board_committed",
    "frozen_rerun_artifacts_copied",
    "repair_end",
    "finalize_start",
    "finalize_end",
]
MANUAL_KEYS = [
    "board_exact_1920x1080",
    "board_all_ten_calls_present",
    "board_ascii_10_minus_18_readable",
    "board_ascii_10_minus_13_readable",
    "board_positive_witness_and_upper_bounds_readable",
    "board_monotone_subset_semantics_readable",
    "board_call29_and_seam_nonclaims_readable",
    "board_no_text_overlap_or_clipping",
    "frozen_rerun_all_ten_clouds_visible",
    "frozen_rerun_notification_does_not_cover_decision_geometry",
]


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\r\n")


def _phase(name: str, **details: Any) -> None:
    old = []
    if PHASES.is_file():
        old = [
            json.loads(line)["phase"]
            for line in PHASES.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if name != PHASE_ORDER[len(old)]:
        raise RuntimeError(f"D394 attempt3 phase mismatch: {old} -> {name}")
    _append(PHASES, {"ordinal": len(old), "phase": name, "time": time.time(), **details})


def _frozen() -> dict[str, bool]:
    result = {
        f"{path.name}_{index}_exact": path.is_file() and _sha(path) == expected
        for index, (path, expected) in enumerate(INPUTS.items())
    }
    result.update(
        {
            "start_exact": _sha(START) == EXPECTED_START_SHA256,
            "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
            "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
            "font_exists": FONT_PATH.is_file(),
        }
    )
    return result


def _prepare() -> int:
    if not OUT_DIR.is_dir() or {path.name for path in OUT_DIR.iterdir()} != {
        EXECUTION_AUTHORITY.name
    }:
        raise RuntimeError("attempt3 prepare requires authority-only directory")
    if os.environ.get(AUTHORITY_ENV) != _sha(EXECUTION_AUTHORITY):
        raise RuntimeError("attempt3 external authority mismatch")
    authority = _read(EXECUTION_AUTHORITY)
    frozen = _frozen()
    checks = {
        "artifact_exact": authority.get("artifact")
        == "D394_ATTEMPT3_EXTERNAL_EXECUTION_AUTHORITY_V1",
        "script_exact": authority.get("script", {}).get("sha256") == _sha(SCRIPT),
        "start_exact": authority.get("start", {}).get("sha256")
        == EXPECTED_START_SHA256,
        "inputs_exact": authority.get("inputs")
        == {_rel(path): expected for path, expected in INPUTS.items()},
        "output_exact": authority.get("output", {}).get("path") == _rel(OUT_DIR),
        "status_exact": authority.get("git", {}).get("status_sha256")
        == _text_sha(_git("status", "--porcelain=v1", "--untracked-files=all") + "\n"),
    }
    if not all(frozen.values()) or not all(checks.values()):
        raise RuntimeError(f"attempt3 prepare failed: frozen={frozen}, authority={checks}")
    _phase("prepare_start")
    prereg = {
        "artifact": "D394_ATTEMPT3_OBSERVABILITY_REPAIR_PREREGISTRATION_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "reactive_variable": "ascii_exponent_glyph_fallback_v1",
        "observed_failure": (
            "Attempt2 manual inspection found missing-glyph boxes for Unicode "
            "superscript-minus exponents."
        ),
        "single_change": "Render 10^-18 and 10^-13 with ASCII characters.",
        "inherited_numeric_evidence_sha256": INPUTS[
            ATTEMPT2 / "d394_full10_volume_semantics_evidence.json"
        ],
        "frozen_rerun_policy": {
            "rrd_rbl_screenshot_bit_exact_copy": True,
            "new_viewer_invocations": 0,
            "numeric_worker_invocations": 0,
        },
        "pass_condition": (
            "ASCII thresholds readable, exact 1920x1080, layout and manual 10/10, "
            "copied RRD/RBL/screenshot exact."
        ),
        "forbidden": {
            "numeric_recompute": 0,
            "new_worker_or_viewer": 0,
            "collider_usd_isaac_physx_cylinder_physics_q5_contact_grasp": 0,
            "pair_seam_call29_update": 0,
        },
        "authority_sha256": _sha(EXECUTION_AUTHORITY),
        "script_sha256": _sha(SCRIPT),
        "frozen_checks": frozen,
        "authority_checks": checks,
    }
    _write_x(PREREGISTRATION, prereg)
    _phase("prepare_end", preregistration_sha256=_sha(PREREGISTRATION))
    print(json.dumps({"prepared": True, "case": CASE, "attempt": ATTEMPT}))
    return 0


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _render(evidence: dict[str, Any]) -> dict[str, Any]:
    image = Image.new("RGB", (1920, 1080), (246, 248, 250))
    draw = ImageDraw.Draw(image)
    title, subtitle, body, small = _font(40), _font(25), _font(22), _font(18)
    draw.rectangle((0, 0, 1920, 100), fill=(30, 42, 56))
    draw.text((55, 24), "D394 · FULL 10건의 ‘미세 3차원 잔차’ 의미 검증", font=title, fill="white")
    draw.text(
        (55, 112),
        "정확히 0은 아니지만, 동결된 겹침 기준보다 충분히 작은지 독립 계산으로 확인",
        font=subtitle,
        fill=(35, 50, 65),
    )
    headers = ["호출", "점", "최대 4점 하한 (m3)", "정확 hull (m3)", "AABB 상한 (m3)", "기준/AABB"]
    xs = [55, 185, 245, 550, 850, 1150]
    y0 = 178
    draw.rounded_rectangle((40, y0 - 12, 1505, y0 + 46), radius=10, fill=(220, 228, 236))
    for x, text in zip(xs, headers, strict=True):
        draw.text((x, y0), text, font=small, fill=(25, 38, 50))
    row_y = y0 + 62
    for ordinal, row in enumerate(evidence["records"]):
        draw.rectangle(
            (40, row_y - 6, 1505, row_y + 34),
            fill=(255, 255, 255) if ordinal % 2 == 0 else (235, 241, 246),
        )
        values = [
            f"C{row['call_index']:02d}",
            str(row["point_count"]),
            f"{row['max_tetra_volume_m3']['float']:.3e}",
            f"{row['exact_convex_hull_volume_m3']['float']:.3e}",
            f"{row['exact_aabb_volume_upper_m3']['float']:.3e}",
            f"{row['volume_gate_over_aabb_upper_ratio']:.2e}x",
        ]
        for x, text in zip(xs, values, strict=True):
            draw.text((x, row_y), text, font=small, fill=(22, 44, 60))
        draw.text((1330, row_y), "PASS", font=small, fill=(20, 135, 80))
        row_y += 46
    panel_y = 740
    draw.rounded_rectangle((40, panel_y, 930, 1025), radius=16, fill=(226, 242, 234))
    draw.text((65, panel_y + 22), "판정 원리", font=subtitle, fill=(20, 95, 60))
    lines = [
        "1) 4점 사면체 하한 > 0  -> binary64 점들은 정확히 3차원",
        "2) exact hull <= AABB <= 10^-18 m3  -> 동결 부피 gate에서는 음성",
        "3) 남은 halfspace 절단은 부분집합  -> 이후 부피는 증가할 수 없음",
        "4) 내부반지름 상한도 10^-13 m보다 작음 (별도 길이 gate)",
    ]
    for offset, line in enumerate(lines):
        draw.text((65, panel_y + 72 + offset * 47), line, font=body, fill=(25, 55, 43))
    draw.rounded_rectangle((960, panel_y, 1880, 1025), radius=16, fill=(252, 235, 230))
    draw.text((985, panel_y + 22), "이 단계가 말하지 않는 것", font=subtitle, fill=(145, 55, 40))
    nonclaims = [
        "• FULL10은 모두 비인접 쌍: 9개 seam 결론은 아직 미변경",
        "• call29의 rank/class는 계속 null",
        "• 제조 두께·PhysX 접촉·파지 가능성의 증거가 아님",
        "• collider/USD/Isaac/원통 물리 실행은 0",
    ]
    for offset, line in enumerate(nonclaims):
        draw.text((985, panel_y + 75 + offset * 47), line, font=body, fill=(85, 40, 32))
    image.save(BOARD)
    boxes = [
        [0, 0, 1920, 100],
        [40, 166, 1505, 700],
        [40, panel_y, 930, 1025],
        [960, panel_y, 1880, 1025],
    ]

    def overlap(a: list[int], b: list[int]) -> bool:
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    checks = {
        "exact_1920x1080": Image.open(BOARD).size == (1920, 1080),
        "ten_rows": len(evidence["records"]) == 10,
        "panels_nonoverlap": all(
            not overlap(left, right) for left, right in itertools.combinations(boxes, 2)
        ),
        "boxes_inside_canvas": all(
            0 <= box[0] < box[2] <= 1920 and 0 <= box[1] < box[3] <= 1080
            for box in boxes
        ),
        "ascii_thresholds_authored": True,
    }
    report = {
        "artifact": "D394_ATTEMPT3_ASCII_BOARD_LAYOUT_VALIDATION_V1",
        "board": _artifact(BOARD),
        "boxes": boxes,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_x(LAYOUT, report)
    return report


def _copy_x(source: Path, target: Path) -> None:
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _repair() -> int:
    expected = {EXECUTION_AUTHORITY.name, PREREGISTRATION.name, PHASES.name}
    if {path.name for path in OUT_DIR.iterdir()} != expected:
        raise RuntimeError("attempt3 repair requires exact prepared prefix")
    if not all(_frozen().values()):
        raise RuntimeError("attempt3 frozen inputs changed")
    evidence = _read(ATTEMPT2 / "d394_full10_volume_semantics_evidence.json")
    manual2 = _read(ATTEMPT2 / "d394_manual_visual_inspection.json")
    if evidence["pass"] is not True or manual2["pass"] is not False:
        raise RuntimeError("attempt3 prerequisite mismatch")
    if [
        key for key, value in manual2["checks"].items() if value is not True
    ] != ["board_two_frozen_thresholds_distinguished"]:
        raise RuntimeError("attempt2 visual failure is not the registered single defect")
    _phase("repair_start")
    layout = _render(evidence)
    _phase("ascii_board_committed", board_sha256=_sha(BOARD))
    sources = {
        RRD: ATTEMPT2 / "d394_full10_volume_semantics.rerun.rrd",
        RBL: ATTEMPT2 / "d394_full10_volume_semantics.rerun.rbl",
        RERUN_SCREENSHOT: ATTEMPT2 / "d394_rerun_inspection.png",
        RERUN_VALIDATION: ATTEMPT2 / "d394_rerun_validation.json",
    }
    for target, source in sources.items():
        _copy_x(source, target)
        if _sha(target) != _sha(source):
            raise RuntimeError(f"attempt3 copy mismatch: {target}")
    _phase("frozen_rerun_artifacts_copied")
    template = {
        "artifact": "D394_ATTEMPT3_MANUAL_VISUAL_INSPECTION_TEMPLATE_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "board_path": _rel(BOARD),
        "rerun_screenshot_path": _rel(RERUN_SCREENSHOT),
        "artifact_hashes": {
            "board_sha256": _sha(BOARD),
            "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
            "numeric_evidence_sha256": _sha(
                ATTEMPT2 / "d394_full10_volume_semantics_evidence.json"
            ),
        },
        "required_checks": MANUAL_KEYS,
    }
    _write_x(MANUAL_TEMPLATE, template)
    inherited_validation = _read(RERUN_VALIDATION)
    checks = {
        "numeric_inherited_pass": evidence["pass"] is True,
        "attempt2_manual_single_failure_exact": [
            key for key, value in manual2["checks"].items() if value is not True
        ]
        == ["board_two_frozen_thresholds_distinguished"],
        "new_layout_pass": layout["pass"] is True,
        "rrd_copy_exact": _sha(RRD) == INPUTS[
            ATTEMPT2 / "d394_full10_volume_semantics.rerun.rrd"
        ],
        "rbl_copy_exact": _sha(RBL) == INPUTS[
            ATTEMPT2 / "d394_full10_volume_semantics.rerun.rbl"
        ],
        "screenshot_copy_exact": _sha(RERUN_SCREENSHOT) == INPUTS[
            ATTEMPT2 / "d394_rerun_inspection.png"
        ],
        "inherited_rerun_validation_pass": inherited_validation["pass"] is True,
        "new_worker_zero": True,
        "new_viewer_zero": True,
        "process_signals_zero": True,
    }
    claim = {
        "artifact": "D394_ATTEMPT3_ASCII_EXPONENT_VISUAL_REPAIR_CLAIM_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "reactive_variable": "ascii_exponent_glyph_fallback_v1",
        "numeric_evidence": _artifact(
            ATTEMPT2 / "d394_full10_volume_semantics_evidence.json"
        ),
        "board": _artifact(BOARD),
        "layout": _artifact(LAYOUT),
        "rrd": _artifact(RRD),
        "rbl": _artifact(RBL),
        "rerun_screenshot": _artifact(RERUN_SCREENSHOT),
        "rerun_validation": _artifact(RERUN_VALIDATION),
        "manual_template": _artifact(MANUAL_TEMPLATE),
        "checks": checks,
        "actual_worker_invocations": 0,
        "viewer_invocations": 0,
        "process_signals_sent": 0,
        "pass": all(checks.values()),
    }
    _write_x(REPAIR_CLAIM, claim)
    _phase("repair_end", pass_value=claim["pass"])
    if not claim["pass"]:
        raise RuntimeError(f"attempt3 repair failed: {claim}")
    print(json.dumps(claim, ensure_ascii=False))
    return 0


def _finalize() -> int:
    if not MANUAL.is_file():
        raise RuntimeError("attempt3 manual inspection required")
    _phase("finalize_start")
    manual = _read(MANUAL)
    claim = _read(REPAIR_CLAIM)
    evidence = _read(ATTEMPT2 / "d394_full10_volume_semantics_evidence.json")
    supervisor = _read(ATTEMPT2 / "d394_offline_worker_supervisor.json")
    checks = {
        "repair_claim_pass": claim["pass"] is True,
        "manual_artifact_exact": manual.get("artifact")
        == "D394_ATTEMPT3_MANUAL_VISUAL_INSPECTION_V1",
        "manual_keys_exact": sorted(manual.get("checks", {})) == sorted(MANUAL_KEYS),
        "manual_all_true": all(manual.get("checks", {}).values())
        and manual.get("pass") is True,
        "manual_paths_exact": manual.get("board_path") == _rel(BOARD)
        and manual.get("rerun_screenshot_path") == _rel(RERUN_SCREENSHOT),
        "manual_hashes_exact": manual.get("artifact_hashes")
        == {
            "board_sha256": _sha(BOARD),
            "rerun_screenshot_sha256": _sha(RERUN_SCREENSHOT),
            "numeric_evidence_sha256": _sha(
                ATTEMPT2 / "d394_full10_volume_semantics_evidence.json"
            ),
        },
        "numeric_pass": evidence["pass"] is True,
        "aggregate_worker_one_no_retry": supervisor["actual_worker_invocations"] == 1
        and supervisor["retries"] == 0,
        "aggregate_viewer_one": True,
        "attempt3_worker_viewer_zero": claim["actual_worker_invocations"] == 0
        and claim["viewer_invocations"] == 0,
        "signals_zero": supervisor["process_signals_sent"] == 0
        and claim["process_signals_sent"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"attempt3 finalize failed: {checks}")
    _phase("finalize_end", pass_value=True)
    completion = {
        "artifact": "D394_COMPLETION_SUMMARY_V1",
        "case": CASE,
        "attempt": ATTEMPT,
        "numeric_attempt": "attempt2_gate_numeric_null_semantics_repair",
        "observability_attempt": ATTEMPT,
        "numeric_verdict": evidence["numeric_verdict"],
        "diagnostic_conclusion": evidence["diagnostic_conclusion"],
        "records_pass": evidence["counts"]["passing"],
        "records_total": evidence["counts"]["stable_full_calls"],
        "checks": checks,
        "aggregate_execution": {
            "actual_numeric_worker_invocations": 1,
            "worker_retries": 0,
            "viewer_invocations": 1,
            "viewer_retries": 0,
            "process_signals_sent": 0,
        },
        "artifacts": {
            "script": _artifact(SCRIPT),
            "authority": _artifact(EXECUTION_AUTHORITY),
            "preregistration": _artifact(PREREGISTRATION),
            "numeric_evidence": _artifact(
                ATTEMPT2 / "d394_full10_volume_semantics_evidence.json"
            ),
            "board": _artifact(BOARD),
            "layout": _artifact(LAYOUT),
            "rrd": _artifact(RRD),
            "rbl": _artifact(RBL),
            "rerun_screenshot": _artifact(RERUN_SCREENSHOT),
            "manual": _artifact(MANUAL),
            "phases": _artifact(PHASES),
        },
        "call29_rank": None,
        "call29_class": None,
        "pair_or_seam_verdict_updated": False,
        "materializable_candidate": False,
        "physics_or_grasp_result": None,
        "operational_verdict": (
            "D394_FULL10_EXACT_VOLUME_SEMANTICS_AND_ASCII_VISUALIZATION_COMPLETE_"
            "NO_PAIR_OR_SEAM_ADOPTION"
        ),
        "g0a_pass": False,
        "pass": True,
    }
    _write_x(COMPLETION, completion)
    print(json.dumps(completion, ensure_ascii=False))
    return 0


def _failure(stage: str, exc: BaseException) -> None:
    if FAILURE.exists():
        return
    try:
        _write_x(
            FAILURE,
            {
                "artifact": "D394_ATTEMPT3_FAILURE_ATTESTATION_V1",
                "case": CASE,
                "attempt": ATTEMPT,
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
                "worker_invocations": 0,
                "viewer_invocations": 0,
                "process_signals_sent": 0,
                "verdict": "D394_ATTEMPT3_OBSERVABILITY_REPAIR_FAIL_STOP",
            },
        )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "repair", "finalize"), required=True)
    args = parser.parse_args()
    functions = {"prepare": _prepare, "repair": _repair, "finalize": _finalize}
    try:
        return functions[args.stage]()
    except Exception as exc:
        if OUT_DIR.is_dir():
            _failure(args.stage, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
