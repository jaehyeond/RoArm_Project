#!/usr/bin/env python3
"""D376: offline provenance and visualization of the frozen D375 terminal stall.

This case reads immutable D351/D367/D373/D375 evidence, the D375 Kit log, and
installed lifecycle source.  It never imports or launches Isaac, Kit, PhysX,
Warp, Hydra, or Fabric.  It does not write USD, step physics, command q5, query
contacts, or decide P34 live identity / grasp feasibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CASE_ROOT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d376"
OUT_DIR = CASE_ROOT / "attempt1_d375_terminal_close_provenance_and_failure_visualization"
PREREG_PATH = OUT_DIR / "d376_preregistration.json"
PHASE_PATH = OUT_DIR / "d376_phase_markers.jsonl"
INVOCATION_PATH = OUT_DIR / "d376_invocation.json"
EVIDENCE_PATH = OUT_DIR / "d376_terminal_close_provenance_evidence.json"
OFFICIAL_ATTESTATION_PATH = OUT_DIR / "d376_nvidia_official_source_attestation.json"
KIT_SNAPSHOT_PATH = OUT_DIR / "d376_frozen_d375_kit_log.txt"
TERMINAL_PNG = OUT_DIR / "d376_d375_terminal_close_timeline_1920x1080.png"
CLASS_PNG = OUT_DIR / "d376_isaac_failure_classification_1920x1080.png"
RRD_PATH = OUT_DIR / "d376_terminal_close_provenance.rrd"
RBL_PATH = OUT_DIR / "d376_terminal_close_provenance.rbl"
RERUN_VALIDATION_PATH = OUT_DIR / "d376_rerun_validation.json"
RERUN_PNG = OUT_DIR / "d376_rerun_inspection.png"
AUTOMATED_PATH = OUT_DIR / "d376_automated_summary.json"
MANUAL_JSON_PATH = OUT_DIR / "d376_manual_visual_inspection.json"
MANUAL_MD_PATH = OUT_DIR / "d376_manual_visual_inspection.md"
COMPLETION_PATH = OUT_DIR / "d376_completion_summary.json"
EXCEPTION_PATH = OUT_DIR / "d376_runtime_exception.json"

D375_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d375/attempt2_external_gpu_attestation_repair"
D375_RAW = D375_DIR / "d375_worker_raw_summary.json"
D375_PRECLOSE = D375_DIR / "d375_worker_preclose_sentinel.json"
D375_SUPERVISOR = D375_DIR / "d375_worker_supervisor.json"
D375_FAIL = D375_DIR / "d375_fail_stop_attestation.json"
D375_PHASES = D375_DIR / "d375_phase_markers.jsonl"
D375_STDOUT = D375_DIR / "d375_worker_stdout.log"
D375_STDERR = D375_DIR / "d375_worker_stderr.log"
D375_PREREG = D375_DIR / "d375_preregistration.json"
D375_GPU = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d375/d375_external_gpu_attestation.json"
D375_CONTROLLER = REPO / "sim_scripts/cyl34_top_view_d375_p34_live_asset_identity_contract_repair.py"
D375_WORKER = REPO / "sim_scripts/cyl34_top_view_d375_p34_live_asset_identity_contract_repair_worker.py"

D351_EXCEPTION = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d351/d351_runtime_exception.json"
D351_BINDING = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d351/d351_live_topology_runtime_binding.json"
D351_AUDIT = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d351/attempt2_timeline_pause_repair/d351_external_termination_audit.json"
D367_SUPERVISOR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d367/d367_supervisor_summary.json"
D367_POSTRUN = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d367/d367_postrun_classification_audit.json"
D373_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0a_d373/attempt1_p34_live_asset_identity_preflight"
D373_RAW = D373_DIR / "d373_worker_raw_summary.json"
D373_SUPERVISOR = D373_DIR / "d373_worker_supervisor.json"
D373_PHASES = D373_DIR / "d373_phase_markers.jsonl"
D373_WORKER = REPO / "sim_scripts/cyl34_top_view_d373_p34_live_asset_worker.py"

D334_SIDECAR = REPO / "claudedocs/lab_meeting/20260715/d334_collision_table"
VIZ_DEBUG = REPO / "roarm_rl/viz_debug.py"
RERUN_CONTRACT = REPO / "roarm_rl/rerun_contract.py"
HARNESS = Path(__file__).resolve()

ISAAC_ROOT = Path("/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim")
D375_KIT_LOG = ISAAC_ROOT / "kit/logs/Kit/Isaac-Sim/5.1/kit_20260722_105312.log"
D373_KIT_LOG = ISAAC_ROOT / "kit/logs/Kit/Isaac-Sim/5.1/kit_20260722_003619.log"
SIM_APP_SOURCE = ISAAC_ROOT / "exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py"
SIM_APP_EXT_CONFIG = ISAAC_ROOT / "exts/isaacsim.simulation_app/config/extension.toml"
PHYSX_UTILS = ISAAC_ROOT / "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/omni/physx/scripts/utils.py"
PHYSX_EXT_CONFIG = ISAAC_ROOT / "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/config/extension.toml"
SIM_APP_CHANGELOG = ISAAC_ROOT / "exts/isaacsim.simulation_app/docs/CHANGELOG.md"

EXPECTED_HEAD = "e30f7f99d44252f509e383627738f3ad7967ea93"
EXPECTED_PYTHON = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

EXPECTED_D375_HASHES = {
    D375_PREREG: "4048cc8201029e4f4d196fe6f68e1f0fdfe90907627b20edeb57ca9a6709744b",
    D375_RAW: "74f959b765860d06ca1d892823d47dc395cad3aea92d0250e21ff706263fc21e",
    D375_PRECLOSE: "1352d49f63b1ba58c75c1e5ad4d0bcb2d000510f1fc060938d672c53288d5203",
    D375_SUPERVISOR: "69f5f8ec5760e7804f3d076c377fc0ea597bde902f3d8ec7d941f36208f4f51c",
    D375_FAIL: "c3fb645ae9ca918e433bdf1734561504aab01a63d97d50086393c16b5d6f8fc7",
    D375_PHASES: "892251db018ac5a4968552a52672e784667cc8c72ffd9206166ea6c41cecd0be",
    D375_STDOUT: "26b86cbe4efd6db50e4242f01116839464bb21ecd56452d90e878a737bcdd51b",
    D375_STDERR: "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
}
EXPECTED_KIT_SHA256 = "6522efde45e776fabf3186ddf362d509a6a3b04f999adc5024c28f41dce1ccc9"
EXPECTED_SIM_APP_SHA256 = "7cbaa6f00e935a6f14bf1c28ec0db089fd924e931f3b0deee07a822f9b7d0090"

NEW_VARIABLES = [
    "d375_terminal_close_provenance_contract_v1",
    "d375_terminal_failure_visualization_projection_v1",
]
VERDICT_PASS = "D376_D375_TERMINAL_CLOSE_PROVENANCE_AND_FAILURE_VISUALIZATION_PASS"
VERDICT_FAIL = "D376_OFFLINE_PROVENANCE_OR_OBSERVABILITY_FAIL_STOP"

OFFICIAL_SOURCES = [
    {
        "title": "Isaac Sim 5.1.0 isaacsim.simulation_app API",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html",
        "applicable_version": "installed Isaac Sim 5.1.0 / extension 2.12.2",
        "claim": "close(skip_cleanup=False) is graceful cleanup; skip_cleanup=True is immediate exit; fast_shutdown defaults true",
    },
    {
        "title": "Isaac Sim 5.1.0 Release Notes",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html",
        "applicable_version": "installed Isaac Sim 5.1.0",
        "claim": "records stage-close and exit-hang fixes but does not identify this workload's cause",
    },
    {
        "title": "Isaac Sim 6.0.0 Release Notes",
        "url": "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html",
        "applicable_version": "later version; mechanism evidence only, not installed behavior",
        "claim": "replaced shutdown_and_release_framework with app.shutdown to avoid main-thread GIL/carb.tasking teardown deadlock (5948099)",
    },
]


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO.resolve()))
    except ValueError:
        return str(resolved)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json_x(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_bytes_x(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _phase(name: str, **fields: Any) -> None:
    payload = {
        "ordinal": sum(1 for _ in PHASE_PATH.open("r", encoding="utf-8")) + 1 if PHASE_PATH.exists() else 1,
        "phase": name,
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        **fields,
    }
    with PHASE_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _inventory(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha(path)})
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {"root": _rel(root), "file_count": len(rows), "files": rows, "inventory_sha256": hashlib.sha256(canonical).hexdigest()}


def _source_hashes() -> dict[str, str]:
    return {"harness": _sha(HARNESS), "viz_debug": _sha(VIZ_DEBUG), "rerun_contract": _sha(RERUN_CONTRACT)}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _matching_lines(path: Path, needles: list[str]) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if any(needle in line for needle in needles):
            rows.append({"line": number, "text": line})
    return rows


def _effective_worker_pass(supervisor: dict[str, Any], raw: dict[str, Any], preclose: dict[str, Any]) -> bool:
    return bool(
        supervisor.get("returncode") == 0
        and not supervisor.get("timed_out")
        and not supervisor.get("sigterm_sent")
        and not supervisor.get("sigkill_sent")
        and raw.get("worker_protocol_pass") is True
        and preclose.get("worker_protocol_pass") is True
        and preclose.get("summary_sha256") == _sha(D375_RAW)
    )


def _negative_controls() -> dict[str, Any]:
    supervisor = _read_json(D375_SUPERVISOR)
    raw = _read_json(D375_RAW)
    preclose = _read_json(D375_PRECLOSE)
    d367 = _read_json(D367_SUPERVISOR)
    gpu = _read_json(D375_GPU)
    spoof_return = dict(supervisor, returncode=0)
    controls = {
        "raw_pass_cannot_override_terminal_timeout": _effective_worker_pass(supervisor, raw, preclose) is False,
        "spoofed_return_zero_cannot_override_timeout_and_signals": _effective_worker_pass(spoof_return, raw, preclose) is False,
        "D367_clean_exit_must_not_be_called_a_shutdown_hang": bool(d367["worker_exit_code"] == 0 and d367["watchdog_reason"] is None and not any(d367["termination"].values())),
        "available_gpu_must_not_be_relabeled_as_oom": bool(gpu["parsed"]["memory_free_mib"] == 15465 and gpu["compute_process_observation"]["isaac_process_detected"] is False),
        "later_6_0_fix_is_mechanism_evidence_not_exact_thread_dump": True,
        "missing_stagecache_erase_is_hypothesis_not_proven_cause": True,
    }
    return {"controls": controls, "passed": sum(bool(value) for value in controls.values()), "total": len(controls), "pass": all(controls.values())}


def _prepare() -> None:
    if CASE_ROOT.exists():
        raise FileExistsError(f"forward-only D376 path already exists: {CASE_ROOT}")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"wrong Python: {sys.executable}")
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    _phase("prepare_start")
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/master")
    d375_hash_checks = {_rel(path): _sha(path) == expected for path, expected in EXPECTED_D375_HASHES.items()}
    d375_inventory = _inventory(D375_DIR)
    d334_inventory = _inventory(D334_SIDECAR)
    selected_inputs = [
        D351_EXCEPTION, D351_BINDING, D351_AUDIT, D367_SUPERVISOR, D367_POSTRUN,
        D373_RAW, D373_SUPERVISOR, D373_PHASES, D373_WORKER, D375_GPU, D375_CONTROLLER, D375_WORKER,
        D375_KIT_LOG, D373_KIT_LOG, SIM_APP_SOURCE, SIM_APP_EXT_CONFIG, PHYSX_UTILS, PHYSX_EXT_CONFIG, SIM_APP_CHANGELOG,
    ]
    selected_hashes = {_rel(path): _sha(path) for path in selected_inputs}
    forbidden_loaded = sorted(
        name for name in sys.modules
        if name == "omni" or name.startswith(("omni.", "isaacsim", "isaaclab", "warp", "pxr"))
    )
    negative = _negative_controls()
    checks = {
        "head_exact": head == EXPECTED_HEAD,
        "origin_exact": origin == EXPECTED_HEAD,
        "head_equals_origin": head == origin,
        "new_variable_count_1_or_2": 1 <= len(NEW_VARIABLES) <= 2,
        "D375_frozen_hashes_exact": all(d375_hash_checks.values()),
        "D375_kit_log_exact": _sha(D375_KIT_LOG) == EXPECTED_KIT_SHA256,
        "installed_simulation_app_source_exact": _sha(SIM_APP_SOURCE) == EXPECTED_SIM_APP_SHA256,
        "negative_controls_6_of_6": negative["pass"] and negative["passed"] == 6,
        "rerun_sdk_0_34_1": _package_version("rerun-sdk") == "0.34.1",
        "isaac_sim_5_1_0_0": _package_version("isaacsim") == "5.1.0.0",
        "isaac_lab_2_3_0": _package_version("isaaclab") == "2.3.0",
        "simulation_app_extension_2_12_2": 'version = "2.12.2"' in SIM_APP_EXT_CONFIG.read_text(encoding="utf-8"),
        "omni_physx_107_3_26": 'version = "107.3.26"' in PHYSX_EXT_CONFIG.read_text(encoding="utf-8"),
        "numpy_1_26_0": np.__version__ == "1.26.0",
        "psutil_5_9_8": _package_version("psutil") == "5.9.8",
        "rerun_cli_absolute_exists": RERUN_CLI.is_file(),
        "fonts_exist": FONT_REGULAR.is_file() and FONT_BOLD.is_file(),
        "forbidden_runtime_modules_not_loaded": not forbidden_loaded,
    }
    prereg = {
        "artifact": "D376_PREREGISTRATION_V1",
        "case": "g0a_d376",
        "attempt": OUT_DIR.name,
        "status": "PREREGISTERED_NOT_RUN",
        "new_variables": NEW_VARIABLES,
        "scope": {
            "immutable_D375_read_only": True,
            "isaac_launches": 0,
            "physx_calls": 0,
            "usd_writes": 0,
            "physics_steps": 0,
            "q5_commands_or_samples": 0,
            "contact_queries": 0,
            "cylinder_writes": 0,
            "target_ik_path_changes": 0,
            "collider_regeneration": 0,
            "automatic_decomposition_sweeps": 0,
            "rerun_viewer_screenshot_invocations_max": 1,
        },
        "registered_questions": [
            "Which D375 phases completed before the process stopped exiting?",
            "Is this the same failure class as D351, D367, or D373?",
            "What does installed 5.1 source prove, and what remains only a hypothesis?",
        ],
        "registered_authority": {
            "D375_execution": "hash-bound raw/preclose/supervisor/phase/stdout plus exact Kit log",
            "installed_lifecycle": "exact installed SimulationApp 2.12.2 source and PhysX 107.3.26 helper",
            "later_mechanism": "NVIDIA Isaac Sim 6.0 release note; version-mismatched mechanism evidence only",
            "visualization": "inspection projection only; never overrides raw JSON/log authority",
        },
        "registered_nulls": [
            "exact_native_thread_or_plugin_blocker",
            "bug_5948099_exact_identity",
            "StageCache_Erase_single_variable_causality",
            "full_authored_callback_classifier_result",
            "physics_equivalence_or_tipping_causality",
            "grasp_feasibility",
            "D375_shutdown_interval_runtime_gpu_telemetry",
        ],
        "official_sources": OFFICIAL_SOURCES,
        "negative_controls": negative,
        "expected_outputs": [
            _rel(EVIDENCE_PATH), _rel(OFFICIAL_ATTESTATION_PATH), _rel(KIT_SNAPSHOT_PATH), _rel(TERMINAL_PNG), _rel(CLASS_PNG),
            _rel(RRD_PATH), _rel(RBL_PATH), _rel(RERUN_VALIDATION_PATH), _rel(RERUN_PNG),
            _rel(AUTOMATED_PATH), _rel(MANUAL_JSON_PATH), _rel(MANUAL_MD_PATH), _rel(COMPLETION_PATH),
        ],
        "git": {"head": head, "origin_master": origin, "status_short": _git("status", "--short")},
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "psutil": _package_version("psutil"),
            "rerun_sdk": _package_version("rerun-sdk"),
            "isaac_sim": _package_version("isaacsim"),
            "isaac_lab": _package_version("isaaclab"),
            "simulation_app_extension": "2.12.2",
            "omni_physx": "107.3.26",
            "forbidden_loaded_modules": forbidden_loaded,
        },
        "D375_hash_checks": d375_hash_checks,
        "D375_inventory_before": d375_inventory,
        "D334_sidecar_before": d334_inventory,
        "selected_input_hashes": selected_hashes,
        "source_hashes": _source_hashes(),
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not prereg["pass"]:
        raise RuntimeError(f"D376 preregistration failed: {checks}")
    _write_json_x(PREREG_PATH, prereg)
    _phase("preregistration_frozen", passed=True, preregistration_sha256=_sha(PREREG_PATH))


def _png_info(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        dimensions = [int(image.width), int(image.height)]
        mode = image.mode
    return {
        "path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha(path),
        "dimensions": dimensions, "mode": mode, "exact_1920x1080": dimensions == [1920, 1080],
    }


def _render_boards(evidence: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch

    regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
    bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#F7F8FA")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    fig.text(0.5, 0.955, "D376 · D375는 측정이 아니라 ‘종료’에서 멈췄다", ha="center", va="center", fontproperties=bold, fontsize=24, color="#14213D")
    fig.text(0.5, 0.918, "immutable D375 JSON + 실제 Kit 로그 + 설치 Isaac Sim 5.1 소스 · Isaac/PhysX 재실행 0회", ha="center", fontproperties=regular, fontsize=12.5, color="#4B5563")

    stages = [
        (0.055, "앱 시작", "cuda:0\nheadless", "#DDECF8", "#1769AA"),
        (0.225, "측정 완료", "callback 34/34\nproperty 17+19", "#E5F5EC", "#147D5B"),
        (0.395, "정리 완료", "PhysX detach 반환\nraw/preclose 해시", "#E5F5EC", "#147D5B"),
        (0.565, "현재 stage 닫힘", "Kit 5.523초\n여기까지 정상", "#FFF4D8", "#B7791F"),
        (0.735, "종료 경계 미완료", "프레임워크 해제\n이후 종료 없음", "#FDE8E8", "#B42318"),
        (0.885, "외부 종료", "900초 TERM\n+20초 KILL", "#FDE8E8", "#B42318"),
    ]
    for idx, (x, title, body, fill, edge) in enumerate(stages):
        if idx < len(stages) - 1:
            ax.annotate("", xy=(stages[idx + 1][0] - 0.035, 0.682), xytext=(x + 0.07, 0.682), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#64748B", lw=2.0))
        ax.add_patch(FancyBboxPatch((x - 0.047, 0.59), 0.125, 0.185, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1.5, edgecolor=edge, facecolor=fill, transform=ax.transAxes))
        fig.text(x + 0.015, 0.735, title, ha="center", va="center", fontproperties=bold, fontsize=12.5, color=edge)
        fig.text(x + 0.015, 0.655, body, ha="center", va="center", fontproperties=regular, fontsize=10.7, color="#1F2937", linespacing=1.45)

    cards = [
        (0.055, 0.18, 0.275, 0.31, "증명된 사실", "• 측정·콜백·물성 조회는 완료\n• PhysX detach와 current stage close 완료\n• 마지막 Kit 로그는 framework release 직전\n• watchdog 900초, SIGTERM, 20초, SIGKILL\n• GPU OOM·PhysX 오류 로그는 없음", "#E8F1FB", "#1769AA"),
        (0.363, 0.18, 0.275, 0.31, "가장 강한 원인 후보", "NVIDIA 6.0은 바로 같은 종료 함수를\napp.shutdown()으로 교체했습니다. 이유는\nmain thread GIL ↔ carb.tasking worker의\n종료 교착(5948099)입니다.\nD375 경계와 정확히 맞지만 thread dump가\n없어 ‘동일 버그 확정’은 아닙니다.", "#FFF0E6", "#C65D00"),
        (0.671, 0.18, 0.275, 0.31, "별도 검증이 필요한 촉발 조건", "D375는 유효한 StageCache ID를 확보한 뒤\nPhysX detach만 하고 Erase하지 않았습니다.\nNVIDIA PhysX helper는 detach→Erase입니다.\nD373도 Erase 없이 정상 종료했으므로 단독 원인은 아닙니다.\nVALID articulation 객체와 함께 촉발했는지는 아직 null이며\nP34 물리·파지 결과도 여전히 null입니다.", "#F3ECFA", "#71429B"),
    ]
    for x, y, w, h, title, body, fill, edge in cards:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.014,rounding_size=0.018", linewidth=1.4, edgecolor=edge, facecolor=fill, transform=ax.transAxes))
        fig.text(x + 0.018, y + h - 0.05, title, ha="left", va="center", fontproperties=bold, fontsize=14, color=edge)
        fig.text(x + 0.018, y + h - 0.105, body, ha="left", va="top", fontproperties=regular, fontsize=10.8, color="#1F2937", linespacing=1.45)
    fig.text(0.5, 0.075, "판정: D375 full identity PASS 아님 · g0a_pass=false · 다음 live 실험은 별도 승인과 단일 변수 필요", ha="center", va="center", fontproperties=bold, fontsize=13, color="#9B1C1C")
    fig.savefig(TERMINAL_PNG, dpi=120, facecolor=fig.get_facecolor()); plt.close(fig)

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="#FFFFFF")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    fig.text(0.5, 0.95, "‘Isaac Sim이 계속 실패한다’가 아닌 이유 — 서로 다른 4개 실패 분류", ha="center", va="center", fontproperties=bold, fontsize=22, color="#14213D")
    fig.text(0.5, 0.91, "같은 제품 이름 아래에서도 멈춘 단계와 종료 결과가 다르면 같은 원인으로 묶으면 안 됩니다.", ha="center", va="center", fontproperties=regular, fontsize=12.5, color="#4B5563")
    rows = evidence["cross_case_classification"]
    y_values = [0.72, 0.53, 0.34, 0.15]
    colors = [("#E8F1FB", "#1769AA"), ("#FFF4D8", "#B7791F"), ("#E9F7F0", "#147D5B"), ("#FDE8E8", "#B42318")]
    for row, y, (fill, edge) in zip(rows, y_values, colors):
        ax.add_patch(FancyBboxPatch((0.055, y), 0.89, 0.14, boxstyle="round,pad=0.012,rounding_size=0.015", linewidth=1.4, edgecolor=edge, facecolor=fill, transform=ax.transAxes))
        fig.text(0.075, y + 0.102, row["case"], ha="left", va="center", fontproperties=bold, fontsize=14, color=edge)
        fig.text(0.245, y + 0.102, row["plain_class"], ha="left", va="center", fontproperties=bold, fontsize=12.2, color="#1F2937")
        fig.text(0.075, y + 0.048, row["evidence_summary"], ha="left", va="center", fontproperties=regular, fontsize=10.6, color="#374151")
    fig.text(0.5, 0.055, "D375만 실제 종료 미완료 · D367은 정상 종료 · D373은 우리 정체성 검사 계약 실패 · D351의 과거 장기 실행 원인은 미확정", ha="center", va="center", fontproperties=bold, fontsize=12, color="#7F1D1D")
    fig.savefig(CLASS_PNG, dpi=120, facecolor=fig.get_facecolor()); plt.close(fig)

    infos = {"terminal_timeline": _png_info(TERMINAL_PNG), "cross_case_classification": _png_info(CLASS_PNG)}
    if not all(info["exact_1920x1080"] for info in infos.values()):
        raise RuntimeError(f"exact board dimension failure: {infos}")
    return infos


def _write_rerun(evidence: dict[str, Any]) -> dict[str, Any]:
    from roarm_rl.rerun_contract import validate_rerun_artifact
    from roarm_rl.viz_debug import log_rerun

    timeline = [
        (0.0, 1.0, "worker start | cuda:0 | headless"),
        (4.58, 1.0, "SimulationApp launch complete"),
        (5.50, 1.0, "34 callbacks + 2 property queries + detach + preclose complete"),
        (5.523, 0.0, "framework-release log reached | native/teardown boundary begins"),
        (900.0, -1.0, "watchdog expired | SIGTERM sent"),
        (920.0, -1.0, "+20 s | SIGKILL | return -9"),
    ]
    scalars = [
        {"entity_path": "metrics/d376/terminal_phase_code__1_done_0_unknown_minus1_forced", "value": state, "timestamp": {"d375_elapsed_s": elapsed}}
        for elapsed, state, _ in timeline
    ]
    events = [
        {"entity_path": "events/d376/timeline", "text": text, "level": "INFO" if state > 0 else "WARN", "timestamp": {"d375_elapsed_s": elapsed}}
        for elapsed, state, text in timeline
    ]
    events.extend([
        {"entity_path": "events/d376/classification", "text": "Only D375 has supervisor-proven terminal non-exit.", "level": "INFO", "static": True},
        {"entity_path": "events/d376/hypothesis", "text": "Candidate: 5.1 framework-release GIL/tasking deadlock; exact thread state unproven.", "level": "WARN", "static": True},
        {"entity_path": "events/d376/boundary", "text": "Code: 1=done, 0=unknown boundary, -1=forced stop. Physics/grasp remain null.", "level": "WARN", "static": True},
    ])
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{RERUN_CLI.parent}:{old_path}"
    try:
        result = log_rerun(
            RRD_PATH,
            scalar_trace=scalars,
            events=events,
            recording_metadata={
                "case": "g0a_d376", "verdict": evidence["verdict"],
                "source": "immutable D375 JSON/log plus installed lifecycle source",
                "isaac_launches": 0, "physx_calls": 0, "physics_steps": 0,
                "q5_samples": 0, "contact_queries": 0, "g0a_pass": False,
                "display_role": "inspection only",
            },
            recording_id="g0a_d376_d375_terminal_close_provenance",
            blueprint_path=RBL_PATH,
            blueprint_mode="d376_terminal_close_provenance",
            live_viewer=False,
            app_id="roarm_g0a_d376_terminal_close",
        )
    finally:
        os.environ["PATH"] = old_path
    if not result.get("ok"):
        raise RuntimeError(f"Rerun save-only recording failed: {result}")
    exact_entities = {
        "metadata/run", "metrics/d376/terminal_phase_code__1_done_0_unknown_minus1_forced", "events/d376/timeline",
        "events/d376/classification", "events/d376/hypothesis", "events/d376/boundary",
    }
    components = {
        "metadata/run": ["TextDocument:text"],
        "metrics/d376/terminal_phase_code__1_done_0_unknown_minus1_forced": ["Scalars:scalars"],
        "events/d376/timeline": ["TextLog:text", "TextLog:level"],
        "events/d376/classification": ["TextLog:text", "TextLog:level"],
        "events/d376/hypothesis": ["TextLog:text", "TextLog:level"],
        "events/d376/boundary": ["TextLog:text", "TextLog:level"],
    }
    strict = validate_rerun_artifact(
        RRD_PATH,
        expected_entity_paths=sorted(exact_entities),
        exact_entity_paths=sorted(exact_entities),
        expected_timeline_names=["blueprint", "d375_elapsed_s", "log_time"],
        exact_timeline_names=["blueprint", "d375_elapsed_s", "log_time"],
        expected_entity_components=components,
        blueprint_path=RBL_PATH,
        screenshot_path=RERUN_PNG,
        screenshot_window_size="1920x1080",
        screenshot_port="auto",
        cli_path=RERUN_CLI,
        expected_version="0.34.1",
        timeout_s=240.0,
    )
    _write_json_x(RERUN_VALIDATION_PATH, strict)
    headless = dict(strict.get("headless_render") or {})
    return {
        "save_only_log": result,
        "strict_validation_pass": strict.get("pass") is True,
        "rrd": {"path": _rel(RRD_PATH), "bytes": RRD_PATH.stat().st_size, "sha256": _sha(RRD_PATH)},
        "rbl": {"path": _rel(RBL_PATH), "bytes": RBL_PATH.stat().st_size, "sha256": _sha(RBL_PATH)},
        "headless_viewer_invocations": 1 if headless.get("attempted") is True else 0,
        "headless_viewer_returncode": headless.get("returncode"),
        "requested_logical_window_size": "1920x1080",
        "physical_raster_note": "HiDPI may produce a 3840x2160 PNG; exact 1920x1080 authority belongs to the Matplotlib boards",
        "screenshot": _png_info(RERUN_PNG) if RERUN_PNG.is_file() else {"path": _rel(RERUN_PNG), "exists": False},
    }


def _run() -> None:
    if not PREREG_PATH.is_file():
        raise RuntimeError("D376 preregistration is missing")
    prereg = _read_json(PREREG_PATH)
    if prereg.get("pass") is not True:
        raise RuntimeError("D376 preregistration did not pass")
    if _source_hashes() != prereg["source_hashes"]:
        raise RuntimeError("D376 source changed after preregistration")
    if _inventory(D375_DIR) != prereg["D375_inventory_before"]:
        raise RuntimeError("immutable D375 inventory changed before run")
    if _inventory(D334_SIDECAR) != prereg["D334_sidecar_before"]:
        raise RuntimeError("user-owned D334 sidecar changed before run")
    for raw_path, expected in prereg["selected_input_hashes"].items():
        if _sha(Path(raw_path) if Path(raw_path).is_absolute() else REPO / raw_path) != expected:
            raise RuntimeError(f"selected input changed: {raw_path}")
    _write_json_x(INVOCATION_PATH, {
        "artifact": "D376_OFFLINE_INVOCATION_V1", "argv": sys.argv, "pid": os.getpid(),
        "python": sys.executable, "cwd": str(Path.cwd()), "offline_process_invocations": 1,
        "automatic_retries": 0, "isaac_or_physx_worker_invocations": 0, "rerun_viewer_max": 1,
        "preregistration_sha256": _sha(PREREG_PATH),
    })
    _phase("offline_audit_start", invocation_sha256=_sha(INVOCATION_PATH))

    raw = _read_json(D375_RAW)
    preclose = _read_json(D375_PRECLOSE)
    supervisor = _read_json(D375_SUPERVISOR)
    fail = _read_json(D375_FAIL)
    gpu = _read_json(D375_GPU)
    phases = _read_jsonl(D375_PHASES)
    phase_by_name = {row["phase"]: row for row in phases}
    callback_rows = [row for row in phases if row["phase"] == "callback_progress"]
    property_end = [row for row in phases if row["phase"] == "property_query_end"]
    cleanup_gap_s = (phase_by_name["supervisor_worker_exit"]["monotonic_ns"] - phase_by_name["worker_cleanup_end"]["monotonic_ns"]) / 1e9
    kit_text = D375_KIT_LOG.read_text(encoding="utf-8", errors="replace")
    kit_lines = _matching_lines(D375_KIT_LOG, [
        "App Name:", "fastShutdown=True", "Using CUDA device ordinal 0", "SimulationApp.close:", "Simulation App Shutting Down",
    ])
    sim_lines = _matching_lines(SIM_APP_SOURCE, [
        "def close(", "if skip_cleanup:", "self.context.close_stage()", "shutdown_and_release_framework()",
    ])
    physx_lines = _matching_lines(PHYSX_UTILS, ["def new_memory_stage", "cache.Insert(stage)", "def release_memory_stage", "detach_stage()", "cache.Erase(stage)"])
    worker_lines = _matching_lines(D375_WORKER, ["UsdUtils.StageCache.Get()", "cache.Insert(stage)", "detach_stage()", "cache.Erase(stage)"])
    controller_lines = _matching_lines(D375_CONTROLLER, ["process.communicate(timeout=20.0)", "process.terminate()", "process.kill()"])
    d373_worker_lines = _matching_lines(D373_WORKER, ["UsdUtils.StageCache.Get()", "cache.Insert(stage)", "detach_stage()", "cache.Erase(stage)"])
    changelog_lines = _matching_lines(SIM_APP_CHANGELOG, ["Fix hang on shutdown", "Fix for hang on exit"])

    d351_exception = _read_json(D351_EXCEPTION)
    d351_binding = _read_json(D351_BINDING)
    d351_audit = _read_json(D351_AUDIT)
    d367 = _read_json(D367_SUPERVISOR)
    d373 = _read_json(D373_SUPERVISOR)
    cross_case = [
        {
            "case": "D351",
            "plain_class": "두 현상: timeline_paused 선행조건 실패 + 원인 미확정 장기 실행",
            "evidence_summary": f"attempt1은 Isaac/128 binding 성공 뒤 timeline_paused=false; attempt2는 {d351_audit['process']['kit_reported_shutdown_elapsed_s']:.3f}초 뒤 승인 SIGTERM·exit {d351_audit['process']['launcher_session_exit_code']}, 내부 원인 null.",
            "same_as_D375": False,
        },
        {
            "case": "D367",
            "plain_class": "정상 종료를 우리 post-close 표식 계약이 FAIL로 잘못 분류",
            "evidence_summary": f"worker exit {d367['worker_exit_code']}, {d367['elapsed_seconds']:.6f}초, watchdog·SIGTERM·SIGKILL·잔류 프로세스 없음.",
            "same_as_D375": False,
        },
        {
            "case": "D373",
            "plain_class": "Isaac 종료는 정상; identity 감사 코드·instance-proxy 구조가 실패",
            "evidence_summary": f"worker exit {d373['returncode']}, {d373['elapsed_s']:.6f}초. instance-proxy ERROR_PARSING stage라 D375의 non-instance VALID-property stage와 상태가 다름.",
            "same_as_D375": False,
        },
        {
            "case": "D375",
            "plain_class": "측정 완료 뒤 terminal framework-release 경계에서 실제 non-exit",
            "evidence_summary": f"raw/preclose PASS 뒤 {supervisor['timeout_s']:.0f}초 timeout→TERM→KILL, 총 {supervisor['elapsed_s']:.6f}초, return {supervisor['returncode']}.",
            "same_as_D375": True,
        },
    ]
    checks = {
        "raw_and_preclose_protocol_true": raw["worker_protocol_pass"] is True and preclose["worker_protocol_pass"] is True,
        "raw_preclose_hash_exact": preclose["summary_sha256"] == _sha(D375_RAW),
        "callback_34_of_34": len(callback_rows) == 34 and all(row["passed"] for row in callback_rows),
        "property_17_plus_19_valid": sorted(row["collider_count"] for row in property_end) == [17, 19] and all(row["passed"] for row in property_end),
        "cleanup_end_reached": phase_by_name["worker_cleanup_end"]["worker_protocol_pass"] is True,
        "timeout_term_kill_return_minus9": supervisor["timed_out"] is True and supervisor["sigterm_sent"] is True and supervisor["sigkill_sent"] is True and supervisor["returncode"] == -9,
        "kit_reaches_stage_close_and_framework_release": any("Stage closed" in row["text"] for row in kit_lines) and any("releasing framework" in row["text"] for row in kit_lines),
        "kit_log_ends_at_release_boundary": kit_text.rstrip().endswith("SimulationApp.close: shutting down app and releasing framework"),
        "installed_source_calls_shutdown_and_release": any("shutdown_and_release_framework()" in row["text"] for row in sim_lines),
        "skip_cleanup_and_graceful_use_same_installed_native_call": sum("shutdown_and_release_framework()" in row["text"] for row in sim_lines) == 2,
        "fast_shutdown_was_already_true": any("fastShutdown=True" in row["text"] for row in kit_lines),
        "startup_gpu_capacity_healthy_and_no_kit_oom_evidence": gpu["parsed"]["memory_free_mib"] == 15465 and gpu["compute_process_observation"]["isaac_process_detected"] is False and "out of memory" not in kit_text.lower(),
        "D367_clean_exit_not_hang": d367["worker_exit_code"] == 0 and d367["watchdog_reason"] is None and not any(d367["termination"].values()),
        "D373_clean_exit_not_hang": d373["returncode"] == 0 and d373["timed_out"] is False,
        "D351_attempt1_binding_success_timeline_only_false": d351_binding["pass"] is True and "'timeline_paused': False" in d351_exception["error"] and "'live_binding_64_plus_64': True" in d351_exception["error"],
        "D351_attempt2_root_cause_unlocalized": d351_audit["runtime_observations_diagnostic_only"]["root_cause_localized"] is False,
        "D351_attempt2_external_termination_exact": d351_audit["process"]["kit_reported_shutdown_elapsed_s"] == 3693.302 and d351_audit["process"]["termination_user_approved"] is True and d351_audit["process"]["launcher_session_exit_code"] == 0,
        "physics_q5_contact_zero": raw["counters"]["physics_steps"] == 0 and raw["counters"]["q5_samples"] == 0 and raw["counters"]["contact_queries"] == 0,
        "D375_effective_identity_pass_false": fail["identity_pass"] is False,
        "D375_controller_term_grace_and_kill_order_bound": sum("process.communicate(timeout=20.0)" in row["text"] for row in controller_lines) >= 1 and any("process.terminate()" in row["text"] for row in controller_lines) and any("process.kill()" in row["text"] for row in controller_lines),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D376 provenance checks failed: {checks}")

    _write_bytes_x(KIT_SNAPSHOT_PATH, D375_KIT_LOG.read_bytes())
    official_attestation = {
        "artifact": "D376_NVIDIA_OFFICIAL_SOURCE_ATTESTATION_V1",
        "accessed_date": "2026-07-22",
        "installed_stack": {
            "isaac_sim": _package_version("isaacsim"),
            "isaac_lab": _package_version("isaaclab"),
            "simulation_app_extension": "2.12.2",
            "kit": "107.3.3",
            "kernel_carbonite": "206.6",
            "omni_physx_schema": "107.3.26",
            "gpu": gpu["parsed"]["name"],
            "driver": gpu["parsed"]["driver_version"],
            "compute_capability": gpu["parsed"]["compute_capability"],
        },
        "sources": [
            {
                **OFFICIAL_SOURCES[0],
                "web_line_reference": "713-736 and 851-892",
                "verified_interpretation": "5.1 documents graceful versus immediate close and fast_shutdown=True; installed 5.1 still routes both close branches through shutdown_and_release_framework.",
            },
            {
                **OFFICIAL_SOURCES[1],
                "web_line_reference": "1262-1285",
                "verified_interpretation": "5.1 records stage-close and exit-hang fixes; D375 proves stage close passed, so those notes do not identify the later framework-release stall.",
            },
            {
                **OFFICIAL_SOURCES[2],
                "web_line_reference": "2974-2998, especially 2992",
                "verified_interpretation": "6.0 later replaced the exact framework-release call to avoid a documented GIL/carb.tasking teardown deadlock; this strongly supports a mechanism but is not an exact D375 thread dump.",
            },
        ],
        "installed_5_1_changelog": {
            "path": _rel(SIM_APP_CHANGELOG),
            "sha256": _sha(SIM_APP_CHANGELOG),
            "matching_lines": changelog_lines,
        },
        "version_separation_preserved": True,
        "installed_5_1_local_source_pass": len(changelog_lines) >= 2,
        "external_web_claim_machine_frozen": False,
        "external_web_claim_review": "manually verified from NVIDIA official pages at the registered URL/line references; final briefing must cite the live URL",
        "pass": len(changelog_lines) >= 2,
    }
    if not official_attestation["pass"]:
        raise RuntimeError("official lifecycle source attestation failed")
    _write_json_x(OFFICIAL_ATTESTATION_PATH, official_attestation)
    hypotheses = {
        "proven_boundary": {
            "classification": "terminal framework-release / process-exit boundary",
            "confidence": "proven by external non-exit plus exact last Kit log and installed call order",
            "not_proven_below_native_call": True,
        },
        "strongest_mechanism_candidate": {
            "classification": "main-thread GIL versus carb.tasking worker teardown deadlock in shutdown_and_release_framework",
            "support": "NVIDIA Isaac Sim 6.0 release note bug 5948099 replaces the exact call used by installed 5.1",
            "version_mismatch": "6.0 mechanism evidence; D375 ran 5.1",
            "exact_identity": None,
            "reason_null": "no D375 thread dump or native stack",
        },
        "workload_trigger_candidate": {
            "classification": "valid custom in-memory articulation stage remained in StageCache after PhysX detach",
            "support": "D375 worker Insert+detach with no Erase; installed NVIDIA PhysX helper defines detach+StageCache.Erase",
            "not_sufficient_alone": "D373 also omitted Erase but exited 0; D373's articulation parse failed whereas D375 created valid PhysX objects",
            "single_variable_causality": None,
        },
        "rejected_or_unsupported": [
            "GPU/VRAM shortage as established cause", "Warp/SM tuning failure", "P34 geometry failure", "callback count overload",
            "PhysX detach failure", "current stage close failure", "all Isaac 5.1 runs always hang",
        ],
    }
    nulls = {
        "exact_native_thread_or_plugin_blocker": None,
        "bug_5948099_exact_identity": None,
        "StageCache_Erase_single_variable_causality": None,
        "full_authored_callback_classifier_result": None,
        "physics_equivalence_or_tipping_causality": None,
        "grasp_feasibility": None,
        "D375_shutdown_interval_runtime_gpu_telemetry": None,
    }
    evidence = {
        "artifact": "D376_D375_TERMINAL_CLOSE_PROVENANCE_EVIDENCE_V1",
        "case": "g0a_d376", "attempt": OUT_DIR.name, "new_variables": NEW_VARIABLES,
        "what_and_why": "Separate D375's successful live acquisition from its terminal non-exit, and distinguish it from earlier unrelated Isaac-labelled failures without rerunning Isaac.",
        "D375_verdict_preserved": fail["verdict"],
        "D375_effective_identity_pass_preserved": fail["identity_pass"],
        "D375_program_order": {
            "callback_rows_passed": len(callback_rows),
            "property_collider_counts": {row["body"]: row["collider_count"] for row in property_end},
            "cleanup_end_to_supervisor_exit_s": cleanup_gap_s,
            "supervisor_elapsed_s": supervisor["elapsed_s"],
            "timeout_s": supervisor["timeout_s"],
            "supervisor_overrun_after_timeout_s": supervisor["elapsed_s"] - supervisor["timeout_s"] if supervisor["sigkill_sent"] else None,
            "configured_term_grace_s": 20.0,
            "returncode": supervisor["returncode"],
            "kit_last_lines": kit_lines[-7:],
        },
        "installed_source_provenance": {
            "simulation_app": {"path": _rel(SIM_APP_SOURCE), "sha256": _sha(SIM_APP_SOURCE), "matching_lines": sim_lines},
            "physx_utils": {"path": _rel(PHYSX_UTILS), "sha256": _sha(PHYSX_UTILS), "matching_lines": physx_lines},
            "D375_worker_stage_lifecycle": {"path": _rel(D375_WORKER), "sha256": _sha(D375_WORKER), "matching_lines": worker_lines, "erase_match_count": sum("cache.Erase(stage)" in row["text"] for row in worker_lines)},
            "D375_controller_termination_order": {"path": _rel(D375_CONTROLLER), "sha256": _sha(D375_CONTROLLER), "matching_lines": controller_lines},
            "D373_worker_stage_lifecycle_control": {"path": _rel(D373_WORKER), "sha256": _sha(D373_WORKER), "matching_lines": d373_worker_lines, "erase_match_count": sum("cache.Erase(stage)" in row["text"] for row in d373_worker_lines), "process_exit_zero": d373["returncode"] == 0},
            "kit_log": {"path": _rel(D375_KIT_LOG), "sha256": _sha(D375_KIT_LOG), "snapshot_path": _rel(KIT_SNAPSHOT_PATH), "snapshot_sha256": _sha(KIT_SNAPSHOT_PATH)},
        },
        "official_sources": OFFICIAL_SOURCES,
        "official_source_attestation": {"path": _rel(OFFICIAL_ATTESTATION_PATH), "sha256": _sha(OFFICIAL_ATTESTATION_PATH)},
        "hypotheses": hypotheses,
        "cross_case_classification": cross_case,
        "scope_counters": {
            "offline_audit_invocations": 1, "automatic_retries": 0, "isaac_launches": 0,
            "physx_calls": 0, "usd_writes": 0, "physics_steps": 0, "q5_commands": 0,
            "q5_samples": 0, "contact_queries": 0, "cylinder_writes": 0,
            "target_ik_path_changes": 0, "collider_regenerations": 0,
            "automatic_decomposition_sweeps": 0,
        },
        "scientific_or_causal_nulls": nulls,
        "g0a_pass": False,
        "checks": checks,
        "pass": all(checks.values()),
        "verdict": VERDICT_PASS if all(checks.values()) else VERDICT_FAIL,
        "next_authorization_boundary": "A separate live control case must test exactly one lifecycle variable before full P34 identity classification or any A64/P34 cylinder physics comparison.",
    }
    _write_json_x(EVIDENCE_PATH, evidence)
    _phase("authoritative_offline_evidence_committed", evidence_sha256=_sha(EVIDENCE_PATH))

    boards = _render_boards(evidence)
    _phase("exact_1920x1080_boards_complete", board_count=len(boards))
    rerun = _write_rerun(evidence)
    _phase("save_only_rerun_and_single_headless_capture_complete", strict_validation_pass=rerun["strict_validation_pass"])

    d375_after = _inventory(D375_DIR)
    d334_after = _inventory(D334_SIDECAR)
    automated_checks = {
        "evidence_pass": evidence["pass"],
        "boards_exact_1920x1080": all(row["exact_1920x1080"] for row in boards.values()),
        "kit_snapshot_hash_exact": _sha(KIT_SNAPSHOT_PATH) == EXPECTED_KIT_SHA256,
        "official_source_attestation_installed_5_1_pass": _read_json(OFFICIAL_ATTESTATION_PATH)["installed_5_1_local_source_pass"] is True,
        "rerun_save_only_ok": rerun["save_only_log"]["ok"] is True,
        "rerun_strict_validation_pass": rerun["strict_validation_pass"],
        "rerun_viewer_exactly_one": rerun["headless_viewer_invocations"] == 1,
        "rerun_viewer_return_zero": rerun["headless_viewer_returncode"] == 0,
        "rerun_screenshot_exists": RERUN_PNG.is_file() and RERUN_PNG.stat().st_size > 0,
        "D375_immutable": d375_after == prereg["D375_inventory_before"],
        "D334_sidecar_immutable": d334_after == prereg["D334_sidecar_before"],
        "nulls_preserved": all(value is None for value in nulls.values()),
        "no_live_or_physics_scope": all(evidence["scope_counters"][key] == 0 for key in (
            "isaac_launches", "physx_calls", "usd_writes", "physics_steps", "q5_commands", "q5_samples",
            "contact_queries", "cylinder_writes", "target_ik_path_changes", "collider_regenerations", "automatic_decomposition_sweeps",
        )),
    }
    automated = {
        "artifact": "D376_AUTOMATED_SUMMARY_V1",
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "boards": boards, "rerun": rerun,
        "D375_inventory_after": d375_after, "D334_sidecar_after": d334_after,
        "manual_visual_inspection": "pending", "completion_contract_pass": False,
        "checks": automated_checks, "pass": all(automated_checks.values()),
        "status": "AWAITING_MANUAL_ORIGINAL_RESOLUTION_INSPECTION",
        "scientific_verdict": None, "g0a_pass": False,
    }
    _write_json_x(AUTOMATED_PATH, automated)
    if not automated["pass"]:
        raise RuntimeError(f"D376 automated contract failed: {automated_checks}")
    _phase("run_complete_awaiting_manual_inspection", automated_summary_sha256=_sha(AUTOMATED_PATH))


def _finalize() -> None:
    for path in (PREREG_PATH, EVIDENCE_PATH, OFFICIAL_ATTESTATION_PATH, AUTOMATED_PATH, MANUAL_JSON_PATH, MANUAL_MD_PATH):
        if not path.is_file():
            raise RuntimeError(f"finalize prerequisite missing: {path}")
    if COMPLETION_PATH.exists():
        raise FileExistsError(f"forward-only completion already exists: {COMPLETION_PATH}")
    _phase("finalize_start")
    prereg = _read_json(PREREG_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    automated = _read_json(AUTOMATED_PATH)
    manual = _read_json(MANUAL_JSON_PATH)
    expected_hashes = {key: value["sha256"] for key, value in automated["boards"].items()}
    expected_hashes["rerun_inspection"] = automated["rerun"]["screenshot"]["sha256"]
    checks = {
        "preregistration_pass": prereg["pass"] is True,
        "offline_evidence_pass": evidence["pass"] is True,
        "automated_summary_pass": automated["pass"] is True,
        "manual_original_resolution_inspection_pass": manual.get("pass") is True,
        "manual_hashes_exact": manual.get("inspected_sha256") == expected_hashes,
        "D375_immutable": _inventory(D375_DIR) == prereg["D375_inventory_before"],
        "D334_sidecar_immutable": _inventory(D334_SIDECAR) == prereg["D334_sidecar_before"],
        "scientific_nulls_preserved": all(value is None for value in evidence["scientific_or_causal_nulls"].values()),
        "g0a_false": evidence["g0a_pass"] is False and automated["g0a_pass"] is False,
    }
    completion = {
        "artifact": "D376_COMPLETION_SUMMARY_V1", "case": "g0a_d376", "attempt": OUT_DIR.name,
        "new_variables": NEW_VARIABLES,
        "preregistration": {"path": _rel(PREREG_PATH), "sha256": _sha(PREREG_PATH)},
        "evidence": {"path": _rel(EVIDENCE_PATH), "sha256": _sha(EVIDENCE_PATH)},
        "official_source_attestation": {"path": _rel(OFFICIAL_ATTESTATION_PATH), "sha256": _sha(OFFICIAL_ATTESTATION_PATH)},
        "automated_summary": {"path": _rel(AUTOMATED_PATH), "sha256": _sha(AUTOMATED_PATH)},
        "manual_inspection": {"path": _rel(MANUAL_JSON_PATH), "sha256": _sha(MANUAL_JSON_PATH), "report": _rel(MANUAL_MD_PATH)},
        "visual_artifacts": {**automated["boards"], "rerun_inspection": automated["rerun"]["screenshot"]},
        "rrd": automated["rerun"]["rrd"], "rbl": automated["rerun"]["rbl"],
        "scope_counters": evidence["scope_counters"],
        "D375_verdict_preserved": evidence["D375_verdict_preserved"],
        "scientific_or_causal_nulls": evidence["scientific_or_causal_nulls"],
        "g0a_pass": False, "checks": checks, "pass": all(checks.values()),
        "verdict": VERDICT_PASS if all(checks.values()) else VERDICT_FAIL,
        "next_authorization_boundary": evidence["next_authorization_boundary"],
    }
    _write_json_x(COMPLETION_PATH, completion)
    _phase("finalize_complete", completion_sha256=_sha(COMPLETION_PATH), verdict=completion["verdict"])
    if not completion["pass"]:
        raise RuntimeError(f"D376 completion contract failed: {checks}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            _prepare()
        elif args.stage == "run":
            _run()
        else:
            _finalize()
        return 0
    except Exception as exc:
        payload = {
            "artifact": "D376_RUNTIME_EXCEPTION_V1", "stage": args.stage,
            "exception_type": type(exc).__name__, "exception": repr(exc),
            "traceback": traceback.format_exc(), "verdict": VERDICT_FAIL,
        }
        try:
            if OUT_DIR.exists() and not EXCEPTION_PATH.exists():
                _write_json_x(EXCEPTION_PATH, payload)
            if OUT_DIR.exists():
                _phase("exception", stage=args.stage, exception_type=type(exc).__name__)
        except Exception:
            pass
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
