#!/usr/bin/env python3
"""D371 observability-only repair over the immutable comparison RRD.

This forward-only repair changes exactly one variable: the Rerun CLI is resolved
by its installed absolute path instead of ambient PATH lookup.  It never
regenerates collider geometry or the RRD, launches Isaac/PhysX, advances physics,
queries live contact, or touches q5/target/IK/path/assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASE = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d371"
OUT = BASE / "visual_repair_attempt1"
PREREG = OUT / "d371_visual_repair_preregistration.json"
VALIDATION = OUT / "d371_rerun_absolute_cli_validation.json"
SCREENSHOT = OUT / "d371_collider_comparison_rerun_absolute_cli.png"
REPORT = OUT / "d371_professor_comparison_report_repaired.md"
RECEIPT = OUT / "d371_visual_repair_receipt.json"
MANUAL_JSON = OUT / "d371_visual_repair_manual_inspection.json"
MANUAL_MD = OUT / "d371_visual_repair_manual_inspection.md"
COMPLETION = OUT / "d371_visual_repair_completion.json"
EXCEPTION = OUT / "d371_visual_repair_exception.json"

RRD = BASE / "d371_collider_comparison.rrd"
RBL = BASE / "d371_collider_comparison.rbl"
EVIDENCE = BASE / "d371_offline_collider_comparison_evidence.json"
ORIGINAL_EXCEPTION = BASE / "d371_runtime_exception.json"
CAP_BOARD = BASE / "d371_cap_comparison_1920x1080.png"
SEMANTIC_BOARD = BASE / "d371_semantic_comparison_1920x1080.png"
CONTACT_BOARD = BASE / "d371_contact_detail_1920x1080.png"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
AGENTS = REPO / "AGENTS.md"
START_HERE = REPO / "START_HERE.md"
HARNESS = Path(__file__).resolve()
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
EXPECTED_HEAD = "4a1120b801e808071583136e78954c78ca941dc8"
EXPECTED_VERSION = "0.34.1"

FAMILY_KEYS = {
    "A": "current64",
    "R64": "raw64",
    "R32": "raw32",
    "C1": "semantic_c1",
    "C2": "semantic_c2",
}
MANUAL_CHECKS = {
    "all_four_pngs_opened_at_original_resolution",
    "three_professor_boards_are_exact_1920x1080",
    "rerun_screenshot_is_exact_3840x2160",
    "candidate_columns_and_actual_counts_are_legible",
    "source_contact_patch_cyan_and_retained_parts_yellow_are_distinguishable",
    "no_unknown_timeline_or_empty_metric_panel",
    "no_notification_obscures_candidate_geometry",
    "Korean_text_has_no_overlap_or_clipping",
    "offline_no_physics_no_grasp_boundary_is_visible",
    "original_failed_run_artifacts_remain_immutable",
}


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_x(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def _write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _source_paths() -> list[Path]:
    return [
        AGENTS,
        START_HERE,
        RRD,
        RBL,
        EVIDENCE,
        ORIGINAL_EXCEPTION,
        CAP_BOARD,
        SEMANTIC_BOARD,
        CONTACT_BOARD,
        RERUN_CONTRACT,
    ]


def _source_hashes() -> dict[str, str]:
    return {_rel(path): _sha(path) for path in _source_paths()}


def _contracts(evidence: dict[str, Any]) -> tuple[list[str], dict[str, list[str]], int]:
    mesh_paths: list[str] = []
    for family, key in FAMILY_KEYS.items():
        for body in ("link5", "gripper_link"):
            for part in evidence["candidate_parts"][family][body]:
                mesh_paths.append(f"compare/{key}/{body}/{part['name']}")
            labels = ("fixed",) if body == "link5" else ("inner", "outer")
            for label in labels:
                mesh_paths.append(f"compare/contact_patch/{key}/{body}/{label}")
    entities: list[str] = []
    components: dict[str, list[str]] = {}
    for path in mesh_paths:
        metadata = f"metadata/meshes/{path.replace('/', '__')}"
        entities.extend([path, metadata])
        components[path] = [
            "CoordinateFrame:frame",
            "Mesh3D:albedo_factor",
            "Mesh3D:triangle_indices",
            "Mesh3D:vertex_positions",
        ]
        components[metadata] = ["TextDocument:text"]
    entities.append("metadata/run")
    components["metadata/run"] = ["TextDocument:text"]
    return sorted(entities), components, len(mesh_paths)


def _png_dimensions(path: Path) -> list[int]:
    from PIL import Image

    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def _prepare() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("D371 visual repair output is not empty")
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = _read(EVIDENCE)
    original_error = _read(ORIGINAL_EXCEPTION)
    entities, components, mesh_count = _contracts(evidence)
    version = subprocess.run(
        [str(RERUN_CLI), "--version"], capture_output=True, text=True, check=False
    )
    checks = {
        "head_exact": _git("rev-parse", "HEAD") == EXPECTED_HEAD,
        "origin_exact": _git("rev-parse", "origin/master") == EXPECTED_HEAD,
        "all_sources_exist": all(path.is_file() for path in _source_paths()),
        "measurement_evidence_pass": evidence.get("measurement_pass") is True,
        "original_failure_is_only_observability_after_measurement": (
            original_error.get("measurement_evidence_valid") is True
            and "rerun CLI not found on PATH" in original_error.get("error", "")
        ),
        "absolute_cli_exists": RERUN_CLI.is_file(),
        "absolute_cli_version_exact": version.returncode == 0
        and EXPECTED_VERSION in f"{version.stdout}\n{version.stderr}",
        "expected_mesh_count_393": mesh_count == 393,
        "expected_non_system_entity_count_787": len(entities) == 787,
        "component_contract_count_787": len(components) == 787,
        "output_empty_before_prereg": not any(OUT.iterdir()),
    }
    payload = {
        "artifact": "D371_RERUN_ABSOLUTE_CLI_VISUAL_REPAIR_PREREGISTRATION_V1",
        "case": "g0a_d371_visual_repair_attempt1",
        "reactive_failure": "ambient PATH lookup could not find installed Rerun CLI",
        "new_variable": "rerun_cli_resolution=installed_absolute_path",
        "absolute_cli": str(RERUN_CLI),
        "expected_version": EXPECTED_VERSION,
        "source_hashes": _source_hashes(),
        "harness_sha256": _sha(HARNESS),
        "expected_entity_count": len(entities),
        "expected_component_contract_count": len(components),
        "expected_mesh_count": mesh_count,
        "execution_contract": {
            "validation_calls": 1,
            "viewer_invocations": 1,
            "automatic_retries": 0,
            "RRD_regeneration": 0,
            "collider_cook_requests": 0,
            "Isaac_or_PhysX_launches": 0,
            "physics_steps": 0,
            "q5_samples": 0,
            "live_contact_queries": 0,
            "target_IK_path_or_asset_changes": 0,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(PREREG, payload)
    if not payload["pass"]:
        raise RuntimeError(f"D371 visual repair prepare failed: {checks}")


def _write_report(evidence: dict[str, Any]) -> None:
    inventories = evidence["candidate_inventories"]
    clearance = evidence["frozen_open_clearance"]["candidates"]
    occupancy = evidence["occupancy_vs_A"]
    pareto = evidence["pareto"]
    rows = [
        "# D371 교수님용 오프라인 충돌체 비교",
        "",
        "물리 스텝·q5 구동·실제 접촉·파지 시험은 모두 0회입니다.",
        "",
        "| 후보 | link5 | 움직이는 턱 | 합계 | 열린 간격 최소(mm) | A 밖 돌출(mm³, 근사) | A 누락(mm³, 근사) | 원본 외피→후보 최대오차(mm) | 오프라인 적격 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for family in FAMILY_KEYS:
        link = inventories[family]["link5"]["part_count"]
        grip = inventories[family]["gripper_link"]["part_count"]
        open_min = min(
            clearance[family]["link5"]["exact_signed_distance_mm"],
            clearance[family]["gripper_link"]["exact_signed_distance_mm"],
        )
        ghost = sum(
            occupancy[body]["candidates"][family]["ghost_volume_mm3_approx"]
            for body in ("link5", "gripper_link")
        )
        missing = sum(
            occupancy[body]["candidates"][family]["undercoverage_volume_mm3_approx"]
            for body in ("link5", "gripper_link")
        )
        surface = pareto["objectives"][family]["max_raw_surface_to_candidate_error_mm"]
        eligible = "예" if pareto["offline_task_gate_eligible"][family] else "아니오"
        rows.append(
            f"| {family} | {link} | {grip} | {link + grip} | {open_min:.6f} | "
            f"{ghost:.1f} | {missing:.1f} | {surface:.6f} | {eligible} |"
        )
    rows.extend(
        [
            "",
            f"- 파레토 비지배 후보: {', '.join(pareto['nondominated_candidates'])}",
            "- `R64 ↔ R32`만 동일 원본·동일 설정에서 최대 hull 상한 하나만 바꾼 비교입니다.",
            "- C1/C2는 몸통을 한 덩어리로 줄이는 아이디어를 빠르게 탈락시킬 수 있는 시제품이지 최적값이 아닙니다.",
            "- 충돌체 수는 계산량의 대리값이며 실제 GPU 속도나 파지 성공을 뜻하지 않습니다.",
            "- 실제 원통 물리시험은 이 결과 보고 뒤 별도 승인 사항입니다.",
        ]
    )
    _write_text_x(REPORT, "\n".join(rows) + "\n")


def _run() -> None:
    if not PREREG.is_file() or _read(PREREG).get("pass") is not True:
        raise RuntimeError("D371 visual repair preregistration is not PASS")
    prereg = _read(PREREG)
    if _sha(HARNESS) != prereg["harness_sha256"] or _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D371 visual repair source/harness drift")
    if {path.name for path in OUT.iterdir()} != {PREREG.name}:
        raise RuntimeError("D371 visual repair pre-run inventory is not prereg-only")

    from roarm_rl.rerun_contract import validate_rerun_artifact

    evidence = _read(EVIDENCE)
    entities, components, mesh_count = _contracts(evidence)
    validation = validate_rerun_artifact(
        RRD,
        expected_entity_paths=entities,
        exact_entity_paths=entities,
        expected_timeline_names=["blueprint", "log_time"],
        exact_timeline_names=["blueprint", "log_time"],
        expected_entity_components=components,
        blueprint_path=RBL,
        screenshot_path=SCREENSHOT,
        screenshot_window_size="3840x2160",
        cli_path=RERUN_CLI,
        expected_version=EXPECTED_VERSION,
        timeout_s=300.0,
    )
    _write_json_x(VALIDATION, validation)
    _write_report(evidence)
    checks = {
        "absolute_cli_validation_pass": validation.get("pass") is True,
        "RRD_sha256_unchanged": _sha(RRD) == prereg["source_hashes"][_rel(RRD)],
        "RBL_sha256_unchanged": _sha(RBL) == prereg["source_hashes"][_rel(RBL)],
        "three_boards_unchanged": all(
            _sha(path) == prereg["source_hashes"][_rel(path)]
            for path in (CAP_BOARD, SEMANTIC_BOARD, CONTACT_BOARD)
        ),
        "entity_count_exact": len(entities) == prereg["expected_entity_count"],
        "mesh_count_exact": mesh_count == prereg["expected_mesh_count"],
        "screenshot_exact_3840x2160": SCREENSHOT.is_file()
        and _png_dimensions(SCREENSHOT) == [3840, 2160],
        "original_exception_preserved": _sha(ORIGINAL_EXCEPTION)
        == prereg["source_hashes"][_rel(ORIGINAL_EXCEPTION)],
    }
    receipt = {
        "artifact": "D371_RERUN_ABSOLUTE_CLI_VISUAL_REPAIR_RECEIPT_V1",
        "case": "g0a_d371_visual_repair_attempt1",
        "new_variable": prereg["new_variable"],
        "validation_path": _rel(VALIDATION),
        "validation_sha256": _sha(VALIDATION),
        "screenshot_path": _rel(SCREENSHOT),
        "screenshot_sha256": _sha(SCREENSHOT) if SCREENSHOT.is_file() else None,
        "report_path": _rel(REPORT),
        "report_sha256": _sha(REPORT),
        "viewer_invocations": 1,
        "automatic_retries": 0,
        "RRD_regeneration": 0,
        "collider_cook_requests": 0,
        "Isaac_or_PhysX_launches": 0,
        "physics_steps": 0,
        "q5_samples": 0,
        "live_contact_queries": 0,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json_x(RECEIPT, receipt)
    if not receipt["pass"]:
        raise RuntimeError(f"D371 visual repair validation failed: {checks}")


def _finalize() -> None:
    for path in (PREREG, VALIDATION, SCREENSHOT, REPORT, RECEIPT, MANUAL_JSON, MANUAL_MD):
        if not path.is_file():
            raise FileNotFoundError(path)
    prereg = _read(PREREG)
    receipt = _read(RECEIPT)
    manual = _read(MANUAL_JSON)
    expected_images = {_rel(path) for path in (CAP_BOARD, SEMANTIC_BOARD, CONTACT_BOARD, SCREENSHOT)}
    observed = {row.get("path") for row in manual.get("files", [])}
    file_checks = []
    for row in manual.get("files", []):
        path = REPO / str(row.get("path"))
        file_checks.append(
            path.is_file()
            and _sha(path) == row.get("sha256")
            and path.stat().st_size == row.get("bytes")
            and _png_dimensions(path) == row.get("dimensions")
            and bool(row.get("observations"))
        )
    checks = {
        "receipt_pass": receipt.get("pass") is True,
        "source_hashes_still_exact": _source_hashes() == prereg["source_hashes"],
        "manual_inspection_performed": manual.get("inspection_performed") is True,
        "manual_check_keys_exact": set(manual.get("checks", {})) == MANUAL_CHECKS,
        "manual_checks_all_true": set(manual.get("checks", {})) == MANUAL_CHECKS
        and all(manual["checks"].values()),
        "manual_four_exact_image_paths": observed == expected_images
        and len(manual.get("files", [])) == 4,
        "manual_file_hash_size_dimension_observation_exact": len(file_checks) == 4
        and all(file_checks),
        "manual_markdown_exact": manual.get("markdown_path") == _rel(MANUAL_MD)
        and manual.get("markdown_sha256") == _sha(MANUAL_MD),
        "measurement_verdict_preserved": _read(EVIDENCE).get("verdict")
        == "D371_OFFLINE_COLLIDER_PARETO_MEASURED_NO_PHYSICS",
    }
    payload = {
        "artifact": "D371_RERUN_ABSOLUTE_CLI_VISUAL_REPAIR_COMPLETION_V1",
        "case": "g0a_d371_visual_repair_attempt1",
        "original_integrated_run_verdict": _read(ORIGINAL_EXCEPTION)["verdict"],
        "measurement_verdict": _read(EVIDENCE)["verdict"],
        "visual_repair_verdict": (
            "D371_RERUN_ABSOLUTE_CLI_VISUAL_REPAIR_PASS"
            if all(checks.values())
            else "D371_RERUN_ABSOLUTE_CLI_VISUAL_REPAIR_FAIL"
        ),
        "checks": checks,
        "pass": all(checks.values()),
        "collider_cook_requests": 0,
        "physics_steps": 0,
        "q5_samples": 0,
        "actual_grasp_or_tipping_verdict": None,
        "g0a_pass": False,
        "next_physics_requires_new_user_approval": True,
    }
    _write_json_x(COMPLETION, payload)
    if not payload["pass"]:
        raise RuntimeError(f"D371 visual repair finalize failed: {checks}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    try:
        {"prepare": _prepare, "run": _run, "finalize": _finalize}[args.stage]()
        print(json.dumps({"stage": args.stage, "pass": True, "output": _rel(OUT)}, ensure_ascii=False))
        return 0
    except Exception as error:
        if args.stage == "run" and OUT.is_dir() and not EXCEPTION.exists():
            _write_json_x(
                EXCEPTION,
                {
                    "artifact": "D371_RERUN_ABSOLUTE_CLI_VISUAL_REPAIR_EXCEPTION_V1",
                    "stage": args.stage,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "automatic_retries": 0,
                },
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
