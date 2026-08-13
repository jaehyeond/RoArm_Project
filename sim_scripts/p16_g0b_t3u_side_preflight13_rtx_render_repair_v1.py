#!/usr/bin/env python3
"""Forward-only RTX replay repair for the frozen P13 physics trace.

This program never creates or advances a physics scene.  It loads the immutable
``t3u_side_preflight13`` trace, authors its recorded body/object poses into a
render-only USD stage, and writes new ``t3u_side_render1_*`` attachments.  The
output is observability evidence only and cannot change the P13 result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"
PREFIX = "t3u_side_render1"
SOURCE_PATH = Path(__file__).resolve()
P16_PATH = (
    REPO
    / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v12.py"
)
P16_SHA256 = "f0c88e73a2ebf0c20e48ecbfa5bf672bc11d5adab523ca1c9817d96e1a511999"
PREREG_PATH = CASE_DIR / f"{PREFIX}_prereg.md"
PREREG_SHA256 = "59c67cf1e2b3f0bf1071bac0ea7607ced69dcbaae2477d83d525381cda39455e"
P13_PREFIX = "t3u_side_preflight13"
P13_PROFILE = "side_preflight13"
P13_INPUT_SHA256 = {
    "script.py.txt": P16_SHA256,
    "prereg.md": "ed3d3f5afd9a2ff5f01f341322367a6f68b168ca3760d67a7465b9e95d8864da",
    "results.json": "8324ed7a9682ccb297985dd733c9e91c480bed9ce65bb02672d5b40226eea6d5",
    "plan.json": "d7fcfb47c26c38f4817ce7630671d915e0d77a4b3bcc1f2d7df40fd816f94f66",
    "trace.npz": "ee67d3516a1c7871e5f48d455b420c3f5985ae889bceb097536904548e8134ee",
    "timeline.rrd": "3235ae954121a9218252785e254e037460176c3699b1e86d9c044ea2187a5601",
    "timeline.rbl": "2c0ae6c1672720486105924fa55f60ec820596051a480208cd4d7a4ff363f34d",
    "rerun_validation.json": "526bd29020577b483e21a3b8686ba07eb170a5aff1ba0169ad4b7e52f41ec1e4",
    "decision_snapshot.png": "f0ffd1607061c4dfdc60388f22760299771225fa0950bf8c24d36b29999542bf",
    "inspection.png": "2f0e164424dea7b5d28382675736129b0c0840032865c7f68c4530ee0b9d0aa2",
    "preclose_sentinel.json": "228ae61d6d2df02fdac10a065729caa30cc2bf5a54c33642da755ebb77f2d401",
}

VIDEO_FPS = 20
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_STEP_STRIDE = 10
TOTAL_STEPS = 2340
EXPECTED_FRAMES = 234
WARMUP_CAPTURES = 6
MOVING_BODIES = (
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "gripper_link",
)
ATTEMPT3_ROOT_PATH = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3"
    / "roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd"
)

OUTPUTS = {
    "input_gate.json": CASE_DIR / f"{PREFIX}_input_gate.json",
    "phase.jsonl": CASE_DIR / f"{PREFIX}_phase.jsonl",
    "script.py.txt": CASE_DIR / f"{PREFIX}_script.py.txt",
    "argv.txt": CASE_DIR / f"{PREFIX}_argv.txt",
    "rgb_frames_manifest.json": CASE_DIR / f"{PREFIX}_rgb_frames_manifest.json",
    "side_grasp.mp4": CASE_DIR / f"{PREFIX}_side_grasp.mp4",
    "failure.json": CASE_DIR / f"{PREFIX}_failure.json",
}
FRAME_DIR = CASE_DIR / f"{PREFIX}_rgb_frames"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_x(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_x(path: Path, value: Any) -> None:
    write_bytes_x(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def append_phase(phase: str, **values: Any) -> None:
    row = {"time_unix": time.time(), "phase": phase, **values}
    path = OUTPUTS["phase.jsonl"]
    mode = "x" if not path.exists() else "a"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"[t3u_render1] {phase} {values}", flush=True)


def p13_path(suffix: str) -> Path:
    return CASE_DIR / f"{P13_PREFIX}_{suffix}"


def load_p16() -> Any:
    if sha256_file(P16_PATH) != P16_SHA256:
        raise RuntimeError("P16_SOURCE_PIN_DRIFT")
    path_hash = hashlib.sha256(str(P16_PATH.resolve()).encode("utf-8")).hexdigest()
    module_name = f"p16_render1_{path_hash}_{P16_SHA256}"
    spec = importlib.util.spec_from_file_location(module_name, P16_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("P16_IMPORT_SPEC_FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    loaded = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded != P16_PATH.resolve() or sha256_file(loaded) != P16_SHA256:
        raise RuntimeError("P16_LOADED_IDENTITY_FAIL")
    return module


def child_output_g0() -> None:
    existing = [str(path.relative_to(REPO)) for path in OUTPUTS.values() if path.exists()]
    if FRAME_DIR.exists():
        existing.append(str(FRAME_DIR.relative_to(REPO)))
    if existing:
        raise RuntimeError(f"RENDER1_G0_OUTPUT_EXISTS {existing}")


def input_gate() -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Path], dict[str, str], dict[str, Any]]:
    if sha256_file(PREREG_PATH) != PREREG_SHA256:
        raise RuntimeError("RENDER1_PREREG_PIN_DRIFT")
    current_input_hashes = {
        suffix: sha256_file(p13_path(suffix)) for suffix in P13_INPUT_SHA256
    }
    if current_input_hashes != P13_INPUT_SHA256:
        raise RuntimeError(
            f"RENDER1_P13_INPUT_PIN_DRIFT expected={P13_INPUT_SHA256} "
            f"actual={current_input_hashes}"
        )
    p16 = load_p16()
    paths = p16.run_paths(P13_PREFIX)
    results = json.loads(paths["results.json"].read_text())
    plan = json.loads(paths["plan.json"].read_text())
    sentinel = json.loads(paths["preclose_sentinel.json"].read_text())
    rerun = json.loads(paths["rerun_validation.json"].read_text())
    with np.load(paths["trace.npz"], allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}

    semantic_checks = p16.validate_result_semantics(
        P13_PROFILE, paths, results, plan
    )
    semantic_keys = getattr(p16, "RESULT_SEMANTIC_CHECK_KEYS", None)
    semantic_pass = bool(
        isinstance(semantic_keys, frozenset)
        and isinstance(semantic_checks, dict)
        and set(semantic_checks) == semantic_keys
        and all(type(value) is bool and value is True for value in semantic_checks.values())
    )
    dependency_paths, dependency_hashes = p16.render_dependency_snapshot(P13_PROFILE)
    provenance = results.get("provenance", {})
    dependency_binding_pass = bool(
        dependency_hashes
        == provenance.get("dependency_hashes_at_start")
        == provenance.get("dependency_hashes_at_finalize")
        and provenance.get("dependency_hashes_equal") is True
        and provenance.get("source_stable") is True
    )
    binding = results.get("representative_binding", {})
    exact_binding = {
        "selected_before_physics": True,
        "trial_id": "c05_o00",
        "candidate_id": "side_sdg_005_raw_025092",
        "candidate_index": 5,
        "pinch_offset_index": 0,
        "environment_slot": 0,
    }
    expected_classifications = [
        {"trial_id": "c05_o00", "label": "premature_jaw_contact"},
        {"trial_id": "c05_o01", "label": "no_bilateral_close"},
        {"trial_id": "c05_o02", "label": "no_bilateral_close"},
        {"trial_id": "c05_o03", "label": "no_bilateral_close"},
        {"trial_id": "c05_o04", "label": "no_bilateral_close"},
    ]
    trace_binding_pass = bool(
        trace.get("physics_step", np.empty(0)).shape == (TOTAL_STEPS,)
        and np.array_equal(
            trace["physics_step"], np.arange(1, TOTAL_STEPS + 1, dtype=np.int64)
        )
        and int(trace["representative_environment_slot"]) == 0
        and str(trace["trial_id"][0]) == "c05_o00"
        and results.get("artifact_hashes_preclose", {}).get("trace.npz")
        == P13_INPUT_SHA256["trace.npz"]
    )
    scientific_status_pass = bool(
        results.get("profile") == P13_PROFILE
        and results.get("scientific_authoritative") is False
        and results.get("runtime_instrumentation_pass") is True
        and results.get("internal_verdict")
        == "INSTRUMENTATION_PREFLIGHT_PASS_PENDING_RENDER_TERMINAL_AND_MANUAL_VISUAL"
        and results.get("metrics", {}).get("measurement_valid") == [True] * 5
        and results.get("metrics", {}).get("success") == [False] * 5
        and results.get("classifications") == expected_classifications
        and results.get("classification_summary", {}).get("selected_verdict")
        == "NO_BILATERAL_SIDE_CONTACT"
        and binding == exact_binding
        and rerun.get("pass") is True
        and results.get("rerun", {}).get("technical_pass") is True
        and sentinel.get("results_sha256") == P13_INPUT_SHA256["results.json"]
        and sentinel.get("trace_sha256") == P13_INPUT_SHA256["trace.npz"]
        and sentinel.get("rerun_validation_sha256")
        == P13_INPUT_SHA256["rerun_validation.json"]
    )
    checks = {
        "p13_input_hashes_exact": current_input_hashes == P13_INPUT_SHA256,
        "p16_unique_path_sha_module_identity_exact": True,
        "p16_result_semantic_complete_map_all_true": semantic_pass,
        "p16_full_render_dependency_snapshot_equals_physics_start_finalize": (
            dependency_binding_pass
        ),
        "p13_trace_representative_and_cadence_exact": trace_binding_pass,
        "p13_valid_preflight_zero_success_status_exact": scientific_status_pass,
        "new_render_cannot_change_or_complete_p13_scientific_authority": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"RENDER1_INPUT_GATE_FAIL {checks}")
    gate = {
        "artifact": "T3U_P13_RTX_RENDER_REPAIR_INPUT_GATE_V1",
        "scientific_authoritative": False,
        "render_is_posthoc_observability_only": True,
        "does_not_replace_p13_terminal_attestation": True,
        "source_sha256": sha256_file(SOURCE_PATH),
        "prereg_sha256": sha256_file(PREREG_PATH),
        "p13_input_paths": {
            suffix: str(p13_path(suffix).relative_to(REPO))
            for suffix in P13_INPUT_SHA256
        },
        "p13_input_sha256": current_input_hashes,
        "semantic_checks": semantic_checks,
        "dependency_paths": {
            name: str(path.relative_to(REPO))
            for name, path in dependency_paths.items()
        },
        "dependency_sha256": dependency_hashes,
        "representative_binding": binding,
        "observed_preflight_verdict": "NO_BILATERAL_SIDE_CONTACT",
        "observed_success_count": 0,
        "observed_trial_count": 5,
        "checks": checks,
        "pass": True,
    }
    return p16, results, plan, trace, dependency_paths, dependency_hashes, gate


def matrix_from_pose(p16: Any, pos: Any, quat_wxyz: Any) -> np.ndarray:
    return np.asarray(p16._matrix_from_pose(pos, quat_wxyz), dtype=np.float64)


def run() -> int:
    child_output_g0()
    source_start = SOURCE_PATH.read_bytes()
    source_sha = hashlib.sha256(source_start).hexdigest()
    simulation_app = None
    try:
        (
            p16,
            results,
            _plan,
            trace,
            dependency_paths,
            dependency_hashes_start,
            gate,
        ) = input_gate()
        write_json_x(OUTPUTS["input_gate.json"], gate)
        write_bytes_x(OUTPUTS["script.py.txt"], source_start)
        write_bytes_x(
            OUTPUTS["argv.txt"],
            ("\n".join([str(SOURCE_PATH), *sys.argv[1:]]) + "\n").encode("utf-8"),
        )
        append_phase(
            "input_gate_durable",
            input_gate_sha256=sha256_file(OUTPUTS["input_gate.json"]),
            source_sha256=source_sha,
            prereg_sha256=PREREG_SHA256,
        )
        representative_slot = 0
        frame_indices = np.arange(
            VIDEO_STEP_STRIDE - 1,
            TOTAL_STEPS,
            VIDEO_STEP_STRIDE,
            dtype=np.int64,
        )
        if (
            len(frame_indices) != EXPECTED_FRAMES
            or int(trace["physics_step"][frame_indices[0]]) != 10
            or int(trace["physics_step"][frame_indices[-1]]) != TOTAL_STEPS
        ):
            raise RuntimeError("RENDER1_FRAME_CADENCE_DRIFT")

        FRAME_DIR.mkdir(parents=False, exist_ok=False)
        append_phase("frame_directory_durable", path=str(FRAME_DIR.relative_to(REPO)))

        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=True)
        simulation_app = launcher.app
        import carb
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from PIL import Image, ImageDraw
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        append_phase("kit_started")
        context = omni.usd.get_context()
        context.new_stage()
        rep.orchestrator.set_capture_on_play(False)
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.commit()
        settings = carb.settings.get_settings()
        capture_setting_path = "/omni/replicator/captureOnPlay"
        if timeline.is_playing() or settings.get_as_bool(capture_setting_path):
            raise RuntimeError("RENDER1_STOPPED_CAPTURE_SETTING_FAIL")

        physics_step_observation = {"event_count": 0, "dt_sum_s": 0.0}

        def on_physics_step(dt: float) -> None:
            physics_step_observation["event_count"] += 1
            physics_step_observation["dt_sum_s"] += float(dt)

        physics_subscription = (
            omni.physx.get_physx_interface().subscribe_physics_step_events(
                on_physics_step
            )
        )

        def clock_snapshot() -> dict[str, Any]:
            return {
                "timeline_is_playing": bool(timeline.is_playing()),
                "timeline_time_s": float(timeline.get_current_time()),
                "capture_on_play": bool(settings.get_as_bool(capture_setting_path)),
                "simulation_manager_num_physics_steps": int(
                    SimulationManager.get_num_physics_steps()
                ),
                "simulation_manager_time_s": float(
                    SimulationManager.get_simulation_time()
                ),
                "physics_step_event_count": int(
                    physics_step_observation["event_count"]
                ),
                "physics_step_event_dt_sum_s": float(
                    physics_step_observation["dt_sum_s"]
                ),
            }

        clock_initial = clock_snapshot()
        clock_audits: list[dict[str, Any]] = []

        def gate_clock(label: str, before: dict[str, Any]) -> dict[str, Any]:
            after = clock_snapshot()
            passed = bool(
                before == clock_initial
                and after == clock_initial
                and after["timeline_is_playing"] is False
                and after["capture_on_play"] is False
            )
            row = {"label": label, "before": before, "after": after, "pass": passed}
            clock_audits.append(row)
            if not passed:
                raise RuntimeError(f"RENDER1_CLOCK_OR_CAPTURE_DRIFT {row}")
            return after

        def setup_update(label: str) -> None:
            before = clock_snapshot()
            simulation_app.update()
            gate_clock(f"setup_app_update:{label}", before)

        for index in range(3):
            setup_update(f"new_stage_settle:{index}")
        stage = context.get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        robot_root = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
        robot_root.GetReferences().AddReference(str(ATTEMPT3_ROOT_PATH))
        asset_load_start = time.monotonic()
        asset_load_updates = 0
        while True:
            loading_status = context.get_stage_loading_status()
            pending_assets = int(loading_status[2])
            if asset_load_updates >= 6 and pending_assets == 0:
                break
            if time.monotonic() - asset_load_start > 120.0:
                raise RuntimeError(
                    "RENDER1_ATTEMPT3_STAGE_LOAD_TIMEOUT_120S "
                    f"status={loading_status} updates={asset_load_updates}"
                )
            setup_update(f"robot_reference_load:{asset_load_updates}")
            asset_load_updates += 1
            if asset_load_updates % 30 == 0:
                append_phase(
                    "asset_load_progress",
                    audited_updates=asset_load_updates,
                    pending_assets=pending_assets,
                )
        append_phase(
            "asset_load_complete",
            audited_updates=asset_load_updates,
            elapsed_seconds=time.monotonic() - asset_load_start,
            final_status=list(context.get_stage_loading_status()),
        )

        body_ops: dict[str, Any] = {}
        for body in MOVING_BODIES:
            prim = stage.GetPrimAtPath(f"/World/Robot/{body}")
            if not prim.IsValid():
                raise RuntimeError(f"RENDER1_ROBOT_BODY_MISSING {body}")
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            body_ops[body] = xform.AddTransformOp()

        def material(path: str, color: tuple[float, float, float], roughness: float) -> Any:
            mat = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                float(roughness)
            )
            mat.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            return mat

        support_mat = material("/World/Looks/Support", (0.34, 0.38, 0.42), 0.88)
        object_mat = material("/World/Looks/Object", (0.86, 0.50, 0.12), 0.55)
        support = UsdGeom.Cube.Define(stage, "/World/Support")
        support.CreateSizeAttr(1.0)
        support_xf = UsdGeom.Xformable(support.GetPrim())
        support_xf.ClearXformOpOrder()
        support_xf.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, -0.005))
        support_xf.AddScaleOp().Set(Gf.Vec3f(0.70, 0.55, 0.01))
        UsdShade.MaterialBindingAPI(support.GetPrim()).Bind(support_mat)

        cylinder = UsdGeom.Cylinder.Define(stage, "/World/Object")
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateRadiusAttr(0.0145)
        cylinder.CreateHeightAttr(0.050)
        cylinder_xf = UsdGeom.Xformable(cylinder.GetPrim())
        cylinder_xf.ClearXformOpOrder()
        cylinder_op = cylinder_xf.AddTransformOp()
        UsdShade.MaterialBindingAPI(cylinder.GetPrim()).Bind(object_mat)

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(2200.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        def look_at_matrix(eye: np.ndarray, target: np.ndarray) -> Any:
            forward = target - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 0] = right
            matrix[:3, 1] = up
            matrix[:3, 2] = -forward
            matrix[:3, 3] = eye
            return Gf.Matrix4d(*matrix.T.flatten().tolist())

        camera = UsdGeom.Camera.Define(stage, "/World/RenderCam")
        camera.CreateFocalLengthAttr(22.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(24.0 * VIDEO_HEIGHT / VIDEO_WIDTH)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 5.0))
        camera_xf = UsdGeom.Xformable(camera.GetPrim())
        camera_xf.ClearXformOpOrder()
        camera_xf.AddTransformOp().Set(
            look_at_matrix(
                np.asarray([0.72, -0.48, 0.36], dtype=np.float64),
                np.asarray([0.28, 0.08, 0.11], dtype=np.float64),
            )
        )
        render_product = rep.create.render_product(
            "/World/RenderCam", (VIDEO_WIDTH, VIDEO_HEIGHT)
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])

        def physics_scene_paths() -> list[str]:
            return sorted(
                prim.GetPath().pathString
                for prim in stage.Traverse()
                if prim.GetTypeName() == "PhysicsScene"
            )

        physics_scenes_before = physics_scene_paths()
        if physics_scenes_before:
            raise RuntimeError(
                f"RENDER1_PHYSICS_SCENE_PRESENT {physics_scenes_before}"
            )
        if clock_snapshot() != clock_initial:
            raise RuntimeError("RENDER1_SETUP_CLOCK_DRIFT")

        def set_trace_frame(step_index: int) -> None:
            for body_index, body in enumerate(MOVING_BODIES):
                matrix = matrix_from_pose(
                    p16,
                    trace["moving_body_pos_m"][step_index, representative_slot, body_index],
                    trace["moving_body_quat_wxyz"][step_index, representative_slot, body_index],
                )
                body_ops[body].Set(Gf.Matrix4d(*matrix.T.flatten().tolist()))
            object_matrix = matrix_from_pose(
                p16,
                trace["object_pos_m"][step_index, representative_slot],
                trace["object_quat_wxyz"][step_index, representative_slot],
            )
            cylinder_op.Set(Gf.Matrix4d(*object_matrix.T.flatten().tolist()))

        def op_matrix(op: Any) -> np.ndarray:
            value = op.Get()
            if value is None:
                raise RuntimeError("RENDER1_XFORM_OP_VALUE_MISSING")
            return np.asarray(value, dtype=np.float64).T

        def frame_fidelity(step_index: int) -> dict[str, Any]:
            body_errors: dict[str, float] = {}
            for body_index, body in enumerate(MOVING_BODIES):
                expected = matrix_from_pose(
                    p16,
                    trace["moving_body_pos_m"][step_index, representative_slot, body_index],
                    trace["moving_body_quat_wxyz"][step_index, representative_slot, body_index],
                )
                body_errors[body] = float(
                    np.max(np.abs(op_matrix(body_ops[body]) - expected))
                )
            expected_object = matrix_from_pose(
                p16,
                trace["object_pos_m"][step_index, representative_slot],
                trace["object_quat_wxyz"][step_index, representative_slot],
            )
            object_error = float(
                np.max(np.abs(op_matrix(cylinder_op) - expected_object))
            )
            passed = bool(
                body_errors
                and np.isfinite(list(body_errors.values())).all()
                and math.isfinite(object_error)
                and max(body_errors.values()) <= 1.0e-12
                and object_error <= 1.0e-12
            )
            row = {
                "moving_body_transform_max_abs": body_errors,
                "object_transform_max_abs": object_error,
                "gate_max_abs": 1.0e-12,
                "pass": passed,
            }
            if not passed:
                raise RuntimeError(f"RENDER1_TRACE_STATE_FIDELITY_FAIL {row}")
            return row

        def capture(
            step_index: int, label: str
        ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
            set_trace_frame(step_index)
            fidelity_pre = frame_fidelity(step_index)
            before = clock_snapshot()
            rep.orchestrator.step(
                rt_subframes=1,
                pause_timeline=True,
                delta_time=0.0,
                wait_for_render=True,
            )
            rgba = rgb_annotator.get_data()
            clock_after = gate_clock(f"rep.orchestrator.step:{label}", before)
            fidelity_post = frame_fidelity(step_index)
            if rgba is None or getattr(rgba, "ndim", 0) != 3:
                raise RuntimeError(f"RENDER1_RGB_DATA_INVALID {label}")
            array = np.asarray(rgba)
            if array.shape[0:2] != (VIDEO_HEIGHT, VIDEO_WIDTH) or array.shape[2] < 3:
                raise RuntimeError(f"RENDER1_RGB_SHAPE_INVALID {array.shape}")
            return (
                np.asarray(array[:, :, :3], dtype=np.uint8),
                fidelity_pre,
                fidelity_post,
                clock_after,
            )

        append_phase(
            "capture_started",
            warmup_captures=WARMUP_CAPTURES,
            written_captures=EXPECTED_FRAMES,
            api=(
                "rep.orchestrator.step(rt_subframes=1,pause_timeline=True,"
                "delta_time=0.0,wait_for_render=True)"
            ),
        )
        first_trace_index = int(frame_indices[0])
        warmup_rows: list[dict[str, Any]] = []
        for warmup_index in range(WARMUP_CAPTURES):
            warmup_rgb, warmup_pre, warmup_post, warmup_clock = capture(
                first_trace_index, f"warmup:{warmup_index}"
            )
            if warmup_rgb.shape != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
                raise RuntimeError("RENDER1_WARMUP_RGB_SHAPE_DRIFT")
            warmup_rows.append(
                {
                    "warmup_index": warmup_index,
                    "source_trace_index": first_trace_index,
                    "physics_step": int(trace["physics_step"][first_trace_index]),
                    "clock_after_capture": warmup_clock,
                    "state_fidelity_pre": warmup_pre,
                    "state_fidelity_post": warmup_post,
                    "written_to_png": False,
                }
            )
            append_phase(
                "capture_warmup_progress",
                completed=warmup_index + 1,
                total=WARMUP_CAPTURES,
            )

        frame_rows: list[dict[str, Any]] = []
        phase_names = tuple(p16.PHASE_STEPS)
        for output_index, step_index_raw in enumerate(frame_indices):
            step_index = int(step_index_raw)
            rgb, fidelity_pre, fidelity_post, capture_clock = capture(
                step_index, f"frame:{output_index}"
            )
            image = Image.fromarray(rgb, mode="RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 10, 775, 76), fill=(0, 0, 0))
            phase_id = int(trace["phase_id"][step_index])
            physics_step = int(trace["physics_step"][step_index])
            draw.text(
                (24, 18),
                "p16 P13 fixed-base side-midpoint trace replay (posthoc only)\n"
                f"trial=c05_o00 | phase={phase_names[phase_id]} | "
                f"physics_step={physics_step}/{TOTAL_STEPS} | P13 grasp success=0/5",
                fill=(255, 255, 255),
            )
            frame_path = FRAME_DIR / f"frame_{output_index:04d}.png"
            with frame_path.open("xb") as handle:
                image.save(handle, format="PNG")
                handle.flush()
                os.fsync(handle.fileno())
            with Image.open(frame_path) as decoded_png:
                decoded_png.load()
                png_pass = bool(
                    decoded_png.format == "PNG"
                    and decoded_png.mode == "RGB"
                    and decoded_png.size == (VIDEO_WIDTH, VIDEO_HEIGHT)
                )
            if not png_pass:
                raise RuntimeError(f"RENDER1_PNG_DECODE_FAIL {frame_path}")
            frame_rows.append(
                {
                    "frame_index": output_index,
                    "source_trace_index": step_index,
                    "physics_step": physics_step,
                    "sim_time_s": float(trace["sim_time_s"][step_index]),
                    "phase_id": phase_id,
                    "phase": phase_names[phase_id],
                    "path": str(frame_path.relative_to(REPO)),
                    "sha256": sha256_file(frame_path),
                    "bytes": frame_path.stat().st_size,
                    "png_full_decode_pass": png_pass,
                    "clock_after_capture": capture_clock,
                    "state_fidelity_pre": fidelity_pre,
                    "state_fidelity_post": fidelity_post,
                }
            )
            if (output_index + 1) % 10 == 0 or output_index + 1 == EXPECTED_FRAMES:
                append_phase(
                    "capture_frame_progress",
                    completed=output_index + 1,
                    total=EXPECTED_FRAMES,
                    last_physics_step=physics_step,
                    last_frame_sha256=frame_rows[-1]["sha256"],
                )

        rgb_annotator.detach([render_product])
        render_product.destroy()
        import imageio_ffmpeg

        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-framerate",
            str(VIDEO_FPS),
            "-start_number",
            "0",
            "-i",
            str(FRAME_DIR / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(VIDEO_FPS),
            str(OUTPUTS["side_grasp.mp4"]),
        ]
        encoded = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=180.0
        )
        if encoded.returncode != 0 or not OUTPUTS["side_grasp.mp4"].is_file():
            raise RuntimeError(
                f"RENDER1_FFMPEG_ENCODE_FAIL rc={encoded.returncode} "
                f"stderr={encoded.stderr}"
            )
        reader = imageio_ffmpeg.read_frames(
            str(OUTPUTS["side_grasp.mp4"]), pix_fmt="rgb24"
        )
        decode_metadata = next(reader)
        decoded_frames = 0
        try:
            for decoded in reader:
                if len(decoded) != VIDEO_WIDTH * VIDEO_HEIGHT * 3:
                    raise RuntimeError("RENDER1_MP4_FRAME_BYTE_LENGTH_DRIFT")
                decoded_frames += 1
        finally:
            reader.close()
        decode_pass = bool(
            decoded_frames == EXPECTED_FRAMES
            and tuple(decode_metadata.get("size", ())) == (VIDEO_WIDTH, VIDEO_HEIGHT)
            and abs(float(decode_metadata.get("fps", 0.0)) - VIDEO_FPS) < 1.0e-9
        )
        if not decode_pass:
            raise RuntimeError(
                f"RENDER1_FULL_DECODE_FAIL frames={decoded_frames} "
                f"metadata={decode_metadata}"
            )

        dependency_paths_end, dependency_hashes_end = p16.render_dependency_snapshot(
            P13_PROFILE
        )
        p13_input_hashes_end = {
            suffix: sha256_file(p13_path(suffix)) for suffix in P13_INPUT_SHA256
        }
        clock_final = clock_snapshot()
        physics_scenes_end = physics_scene_paths()
        end_checks = {
            "source_unchanged": SOURCE_PATH.read_bytes() == source_start,
            "prereg_unchanged": sha256_file(PREREG_PATH) == PREREG_SHA256,
            "p13_input_hashes_unchanged": p13_input_hashes_end == P13_INPUT_SHA256,
            "full_render_dependency_paths_unchanged": dependency_paths_end
            == dependency_paths,
            "full_render_dependency_hashes_unchanged": dependency_hashes_end
            == dependency_hashes_start,
            "dependency_still_equals_p13_physics_start_finalize": bool(
                dependency_hashes_end
                == results.get("provenance", {}).get("dependency_hashes_at_start")
                == results.get("provenance", {}).get("dependency_hashes_at_finalize")
            ),
            "exact_234_frames": len(frame_rows) == EXPECTED_FRAMES,
            "all_pngs_decoded": all(row["png_full_decode_pass"] for row in frame_rows),
            "mp4_full_decode_exact": decode_pass,
            "capture_on_play_stayed_false": bool(
                clock_audits
                and all(
                    row["before"]["capture_on_play"] is False
                    and row["after"]["capture_on_play"] is False
                    for row in clock_audits
                )
                and clock_final["capture_on_play"] is False
            ),
            "timeline_and_physics_clocks_unchanged": bool(
                clock_final == clock_initial
                and all(row["pass"] for row in clock_audits)
            ),
            "no_physics_scene": not physics_scenes_before and not physics_scenes_end,
            "no_physics_callback": physics_step_observation["event_count"] == 0
            and physics_step_observation["dt_sum_s"] == 0.0,
            "all_trace_transform_fidelity_exact": all(
                row["state_fidelity_pre"]["pass"]
                and row["state_fidelity_post"]["pass"]
                for row in [*warmup_rows, *frame_rows]
            ),
            "exact_six_unwritten_warmups": bool(
                len(warmup_rows) == WARMUP_CAPTURES
                and all(row["written_to_png"] is False for row in warmup_rows)
            ),
        }
        if not all(end_checks.values()):
            raise RuntimeError(f"RENDER1_END_GATE_FAIL {end_checks}")
        manifest = {
            "artifact": "T3U_P13_ISOLATED_RTX_TRACE_REPLAY_REPAIR_V1",
            "scientific_authoritative": False,
            "render_is_posthoc_observability_only": True,
            "does_not_replace_or_complete_p13_terminal_attestation": True,
            "p13_observed_verdict": "NO_BILATERAL_SIDE_CONTACT",
            "p13_observed_success_count": 0,
            "p13_observed_trial_count": 5,
            "source_sha256": source_sha,
            "prereg_sha256": PREREG_SHA256,
            "input_gate_sha256": sha256_file(OUTPUTS["input_gate.json"]),
            "source_trace_path": str(p13_path("trace.npz").relative_to(REPO)),
            "source_trace_sha256": P13_INPUT_SHA256["trace.npz"],
            "representative_binding": results["representative_binding"],
            "renderer": {
                "installed_contract": "Isaac_Sim_5.1__Kit_107.3__omni.replicator.core_1.12.27",
                "capture_on_play": False,
                "capture_api": (
                    "rep.orchestrator.step(rt_subframes=1,pause_timeline=True,"
                    "delta_time=0.0,wait_for_render=True)"
                ),
                "per_frame_app_update_after_step": False,
                "annotator_get_data_immediate_after_step": True,
                "warmup_capture_count_not_written": WARMUP_CAPTURES,
                "warmup_captures": warmup_rows,
                "written_capture_count": EXPECTED_FRAMES,
                "clock_initial": clock_initial,
                "clock_final": clock_final,
                "clock_audits": clock_audits,
                "physics_scene_paths_before_capture": physics_scenes_before,
                "physics_scene_paths_end": physics_scenes_end,
                "physics_step_event_count": physics_step_observation["event_count"],
                "physics_step_event_dt_sum_s": physics_step_observation["dt_sum_s"],
                "explicit_physics_api_calls": [],
            },
            "cadence": {
                "source_physics_hz": 200,
                "video_fps": VIDEO_FPS,
                "physics_step_stride": VIDEO_STEP_STRIDE,
                "first_physics_step": 10,
                "last_physics_step": TOTAL_STEPS,
            },
            "resolution": [VIDEO_WIDTH, VIDEO_HEIGHT],
            "frame_count": len(frame_rows),
            "frames": frame_rows,
            "first_frame_sha256": frame_rows[0]["sha256"],
            "last_frame_sha256": frame_rows[-1]["sha256"],
            "mp4_path": str(OUTPUTS["side_grasp.mp4"].relative_to(REPO)),
            "mp4_sha256": sha256_file(OUTPUTS["side_grasp.mp4"]),
            "mp4_bytes": OUTPUTS["side_grasp.mp4"].stat().st_size,
            "ffmpeg_command": command,
            "decode": {
                "metadata": decode_metadata,
                "decoded_frame_count": decoded_frames,
                "full_decode_pass": decode_pass,
            },
            "p13_input_sha256_at_start": P13_INPUT_SHA256,
            "p13_input_sha256_at_end": p13_input_hashes_end,
            "full_render_dependency_sha256_at_start": dependency_hashes_start,
            "full_render_dependency_sha256_at_end": dependency_hashes_end,
            "end_checks": end_checks,
            "pass": True,
        }
        write_json_x(OUTPUTS["rgb_frames_manifest.json"], manifest)
        append_phase(
            "render_complete_durable",
            manifest_sha256=sha256_file(OUTPUTS["rgb_frames_manifest.json"]),
            mp4_sha256=manifest["mp4_sha256"],
            frame_count=EXPECTED_FRAMES,
        )
        physics_subscription = None
        print(
            f"[t3u_render1] COMPLETE frames={EXPECTED_FRAMES} "
            f"mp4={OUTPUTS['side_grasp.mp4']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        failure = {
            "artifact": "T3U_P13_RTX_RENDER_REPAIR_FAILURE_V1",
            "scientific_authoritative": False,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "source_sha256": source_sha,
            "prereg_sha256": sha256_file(PREREG_PATH)
            if PREREG_PATH.is_file()
            else None,
        }
        if not OUTPUTS["failure.json"].exists():
            write_json_x(OUTPUTS["failure.json"], failure)
        if OUTPUTS["phase.jsonl"].exists():
            append_phase(
                "render_failure_durable",
                failure_sha256=sha256_file(OUTPUTS["failure.json"]),
                error_type=type(exc).__name__,
                message=str(exc),
            )
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run or sys.argv[1:] != ["--run"]:
        raise RuntimeError("RENDER1_EXACT_ARGV_REQUIRED --run")
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
