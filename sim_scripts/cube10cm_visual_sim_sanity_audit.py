"""Audit visual/sim sanity evidence for the 10cm cube seed962 tap trace.

This consolidates the local trace storyboard, failed live record-video attempt,
and successful trace replay render. It performs no IsaacLab runtime, GPU render,
dataset generation, training, robot control, SSH, or trace mutation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_TRACE_CSV = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus_visual_env0_seed962_trace.csv"
DEFAULT_STORYBOARD_JSON = LOG_DIR / "cube10cm_visual_sanity_trace_storyboard.json"
DEFAULT_RENDER_SUMMARY_JSON = LOG_DIR / "diffik_probe_cube10cm_m072_render_replay_env0_seed962_summary.json"
DEFAULT_MP4_PROBE_OUT = LOG_DIR / "diffik_probe_cube10cm_m072_render_replay_env0_seed962_mp4_probe.out"
DEFAULT_REACTION_GATE_JSON = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_reaction_gate_audit.json"
)
DEFAULT_RECORD_VIDEO_DIR = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus_visual_seed962_video"
DEFAULT_RECORD_VIDEO_SUMMARY = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus_visual1env_seed962_summary.json"
DEFAULT_RECORD_VIDEO_CSV = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus_visual1env_seed962.csv"
DEFAULT_RECORD_VIDEO_TRACE = LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus_visual1env_seed962_trace.csv"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_visual_sim_sanity_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_visual_sim_sanity_audit_summary.out"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mp4_probe(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value in {"True", "False"}:
            result[key] = value == "True"
        else:
            result[key] = _float(value, value)
    return result


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _record_video_file_count(video_dir: Path) -> int:
    if not video_dir.exists():
        return 0
    return sum(1 for path in video_dir.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_csv", type=Path, default=DEFAULT_TRACE_CSV)
    parser.add_argument("--storyboard_json", type=Path, default=DEFAULT_STORYBOARD_JSON)
    parser.add_argument("--render_summary_json", type=Path, default=DEFAULT_RENDER_SUMMARY_JSON)
    parser.add_argument("--mp4_probe_out", type=Path, default=DEFAULT_MP4_PROBE_OUT)
    parser.add_argument("--reaction_gate_json", type=Path, default=DEFAULT_REACTION_GATE_JSON)
    parser.add_argument("--record_video_dir", type=Path, default=DEFAULT_RECORD_VIDEO_DIR)
    parser.add_argument("--record_video_summary", type=Path, default=DEFAULT_RECORD_VIDEO_SUMMARY)
    parser.add_argument("--record_video_csv", type=Path, default=DEFAULT_RECORD_VIDEO_CSV)
    parser.add_argument("--record_video_trace", type=Path, default=DEFAULT_RECORD_VIDEO_TRACE)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    trace_rows = _read_rows(args.trace_csv)
    storyboard = _load_json(args.storyboard_json)
    render_summary = _load_json(args.render_summary_json)
    mp4_probe = _load_mp4_probe(args.mp4_probe_out)
    reaction_gate = _load_json(args.reaction_gate_json)

    contact_rows = [row for row in trace_rows if _int(row.get("measured_contact_now")) == 1]
    contact_row = contact_rows[0] if contact_rows else trace_rows[-1]
    contact_step = _int(contact_row.get("step"), -1)
    contact_frame = _int(contact_row.get("frame"), -1)
    tcp_z = _float(contact_row.get("tcp_z_m"))
    target_z = _float(contact_row.get("target_z_m"))
    z_err = tcp_z - target_z
    tcp_target_err_before = _float(contact_row.get("tcp_target_err_before_m"))
    tcp_target_err_after = _float(contact_row.get("tcp_target_err_after_m"))

    record_video = {
        "attempted": True,
        "file_count": _record_video_file_count(args.record_video_dir),
        "summary_exists": args.record_video_summary.exists(),
        "csv_exists": args.record_video_csv.exists(),
        "trace_exists": args.record_video_trace.exists(),
    }
    record_video["status"] = (
        "FAILED_NO_FRAMES_OR_SUMMARY"
        if record_video["file_count"] == 0
        and not record_video["summary_exists"]
        and not record_video["csv_exists"]
        and not record_video["trace_exists"]
        else "HAS_PARTIAL_ARTIFACTS_INSPECT_MANUALLY"
    )

    render_ok = (
        bool(mp4_probe.get("opened"))
        and bool(mp4_probe.get("first_frame_ok"))
        and _int(mp4_probe.get("frame_count")) == _int(render_summary.get("frames_written"))
        and bool(render_summary.get("physics_recomputed")) is False
        and bool(render_summary.get("training")) is False
        and bool(render_summary.get("dataset_generation")) is False
    )
    cube_sizes = render_summary.get("cube_size_m", {})
    cube_size_env0 = cube_sizes.get("0", []) if isinstance(cube_sizes, dict) else []
    cube_size_ok = cube_size_env0 == [0.1, 0.1, 0.1]
    visual_contact_evidence = bool(reaction_gate.get("reaction_gate_pass")) and render_ok and cube_size_ok
    clean_tap_visual_verified = (
        visual_contact_evidence
        and abs(z_err) <= 0.015
        and tcp_target_err_before <= 0.03
        and _float(contact_row.get("clip_any")) < 0.5
    )

    result = {
        "artifact_type": "cube10cm_visual_sim_sanity_audit_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_audit_only": True,
        "no_gpu_runtime_dataset_training_robot_ssh": True,
        "trace_rows": len(trace_rows),
        "storyboard": {
            "trace_storyboard_ready": True,
            "actual_render_video_run": storyboard.get("actual_render_video_run"),
            "sanity_checks": storyboard.get("sanity_checks"),
        },
        "record_video_attempt": record_video,
        "replay_render": {
            "status": "PASS" if render_ok else "FAIL",
            "frames_written": render_summary.get("frames_written"),
            "mp4_opened": mp4_probe.get("opened"),
            "mp4_frame_count": mp4_probe.get("frame_count"),
            "first_frame_ok": mp4_probe.get("first_frame_ok"),
            "size_bytes": mp4_probe.get("size_bytes"),
            "cube_size_env0_m": cube_size_env0,
            "cube_size_ok_10cm": cube_size_ok,
            "physics_recomputed": render_summary.get("physics_recomputed"),
            "training": render_summary.get("training"),
            "dataset_generation": render_summary.get("dataset_generation"),
        },
        "contact_frame_metrics": {
            "env_id": 0,
            "contact_frame": contact_frame,
            "contact_step": contact_step,
            "tcp_z_m": tcp_z,
            "target_z_m": target_z,
            "tcp_minus_target_z_m": z_err,
            "tcp_target_err_before_m": tcp_target_err_before,
            "tcp_target_err_after_m": tcp_target_err_after,
            "disp_along_push_m": _float(contact_row.get("disp_along_push_m")),
            "cube_speed_mps": _float(contact_row.get("cube_speed_mps")),
            "tip_angle_deg": _float(contact_row.get("tip_angle_deg")),
            "clip_any": _int(contact_row.get("clip_any")),
            "clip_max_joint_name": contact_row.get("clip_max_joint_name", ""),
        },
        "visual_contact_evidence": visual_contact_evidence,
        "clean_tap_visual_verified": clean_tap_visual_verified,
        "dataset_rl_roarm_unblocked": False,
        "verdict": (
            "VISUAL_CONTACT_REPLAY_PASS_BUT_CLEAN_TAP_NOT_VERIFIED"
            if visual_contact_evidence and not clean_tap_visual_verified
            else "VISUAL_REPLAY_BLOCKED_OR_INCOMPLETE"
        ),
    }

    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_summary.write_text(
        "\n".join(
            [
                "line1 artifact=cube10cm_visual_sim_sanity_audit_v1 local_audit_only=YES "
                "gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
                f"line2 trace_storyboard ready=YES rows={len(trace_rows)} "
                f"all_yplus={storyboard.get('sanity_checks', {}).get('all_yplus')} "
                f"contact_all={storyboard.get('sanity_checks', {}).get('contact_seen_all_envs')} "
                f"reaction_all={storyboard.get('sanity_checks', {}).get('reaction_event_all_envs')} "
                f"actual_render_video_run={storyboard.get('actual_render_video_run')}",
                f"line3 record_video_attempt status={record_video['status']} "
                f"file_count={record_video['file_count']} summary_exists={record_video['summary_exists']} "
                f"csv_exists={record_video['csv_exists']} trace_exists={record_video['trace_exists']}",
                f"line4 replay_render status={'PASS' if render_ok else 'FAIL'} frames={render_summary.get('frames_written')} "
                f"mp4_opened={mp4_probe.get('opened')} mp4_frames={mp4_probe.get('frame_count')} "
                f"first_frame_ok={mp4_probe.get('first_frame_ok')} cube_size_env0={cube_size_env0} "
                f"physics_recomputed={render_summary.get('physics_recomputed')} "
                f"training={render_summary.get('training')} dataset_generation={render_summary.get('dataset_generation')}",
                f"line5 contact_frame env=0 frame={contact_frame} step={contact_step} "
                f"tcp_z={tcp_z:.9f} target_z={target_z:.9f} tcp_minus_target_z={z_err:.9f} "
                f"tcp_target_err_before={tcp_target_err_before:.9f} tcp_target_err_after={tcp_target_err_after:.9f} "
                f"disp_along_push={_float(contact_row.get('disp_along_push_m')):.9f} "
                f"cube_speed={_float(contact_row.get('cube_speed_mps')):.9f} "
                f"tip_deg={_float(contact_row.get('tip_angle_deg')):.9f} "
                f"clip_any={_int(contact_row.get('clip_any'))} "
                f"clip_joint={contact_row.get('clip_max_joint_name', '')}",
                f"line6 verdict visual_contact_replay_pass={visual_contact_evidence} "
                f"clean_tap_visual_verified={clean_tap_visual_verified} "
                "reason=10cm_replay_visible_but_contact_frame_has_large_vertical_target_error_and_clip",
                "line7 pipeline dataset_rl_roarm_unblocked=NO action_teacher_dataset_unblocked=NO "
                "next=fix_or_retest_teacher_contact_geometry_before_dataset_rl_robot",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
