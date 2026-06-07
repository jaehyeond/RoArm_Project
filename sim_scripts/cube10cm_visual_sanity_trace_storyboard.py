"""Local visual sanity storyboard for the 10cm cube tap/reaction trace.

This script converts an existing IsaacLab trace into a human-inspectable
HTML/SVG storyboard. It does not run IsaacLab, use GPU, generate a dataset,
train, control a robot, SSH, or mutate source traces.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_TRACE_CSV = (
    LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_trace.csv"
)
DEFAULT_ROLLOUT_CSV = (
    LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962.csv"
)
DEFAULT_SUMMARY_JSON = (
    LOG_DIR / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_summary.json"
)
DEFAULT_REACTION_GATE_JSON = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_xnegheight050_pre020_seed962_reaction_gate_audit.json"
)
DEFAULT_REVALIDATION_JSON = LOG_DIR / "cube10cm_teacher_quality_revalidation_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_visual_sanity_trace_storyboard.json"
DEFAULT_OUT_HTML = LOG_DIR / "cube10cm_visual_sanity_trace_storyboard.html"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_visual_sanity_trace_storyboard_summary.out"


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _scale_points(
    rows: list[dict[str, str]],
    *,
    x_key: str,
    y_key: str,
    width: int,
    height: int,
    margin: int,
) -> tuple[dict[str, float], list[tuple[float, float]]]:
    xs = [_float(row.get(x_key)) for row in rows]
    ys = [_float(row.get(y_key)) for row in rows]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    pad_x = max(0.025, (xmax - xmin) * 0.2)
    pad_y = max(0.025, (ymax - ymin) * 0.2)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y
    sx = (width - 2 * margin) / max(1e-9, xmax - xmin)
    sy = (height - 2 * margin) / max(1e-9, ymax - ymin)
    scale = min(sx, sy)

    def project(x: float, y: float) -> tuple[float, float]:
        px = margin + (x - xmin) * scale
        py = margin + (ymax - y) * scale
        return px, py

    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "scale": scale}, [
        project(_float(row.get(x_key)), _float(row.get(y_key))) for row in rows
    ]


def _topdown_svg(env: dict[str, Any], rows: list[dict[str, str]]) -> str:
    width = 520
    height = 420
    margin = 38
    frame, cube_path = _scale_points(
        rows,
        x_key="cube_x_m",
        y_key="cube_y_m",
        width=width,
        height=height,
        margin=margin,
    )

    def project(x: float, y: float) -> tuple[float, float]:
        px = margin + (x - frame["xmin"]) * frame["scale"]
        py = margin + (frame["ymax"] - y) * frame["scale"]
        return px, py

    tcp_path = [project(_float(row.get("tcp_x_m")), _float(row.get("tcp_y_m"))) for row in rows]
    target_path = [project(_float(row.get("target_x_m")), _float(row.get("target_y_m"))) for row in rows]
    cube_size_px = max(8.0, _float(rows[0].get("cube_size_x_m"), 0.1) * frame["scale"])
    start_x, start_y = project(_float(rows[0].get("cube_x_m")), _float(rows[0].get("cube_y_m")))
    end_x, end_y = project(_float(rows[-1].get("cube_x_m")), _float(rows[-1].get("cube_y_m")))

    contact_step = int(env["first_contact_step"])
    contact_rows = [row for row in rows if _int(row.get("step")) >= contact_step] if contact_step >= 0 else []
    contact_x, contact_y = (None, None)
    if contact_rows:
        contact_x, contact_y = project(_float(contact_rows[0].get("cube_x_m")), _float(contact_rows[0].get("cube_y_m")))

    return f"""
<svg viewBox="0 0 {width} {height}" class="plot" role="img" aria-label="topdown env {env['env_id']}">
  <rect width="{width}" height="{height}" fill="#fbfbf7"/>
  <text x="18" y="24" class="caption">env {env['env_id']} top-down: cube/TCP/target path, y+ is up</text>
  <line x1="38" y1="{height-34}" x2="94" y2="{height-34}" stroke="#111" stroke-width="1"/>
  <line x1="38" y1="{height-34}" x2="38" y2="{height-90}" stroke="#111" stroke-width="1"/>
  <text x="98" y="{height-30}" class="axis">x+</text>
  <text x="26" y="{height-94}" class="axis">y+</text>
  <polyline points="{_polyline(target_path)}" fill="none" stroke="#8a8a8a" stroke-width="1.5" stroke-dasharray="4 4"/>
  <polyline points="{_polyline(tcp_path)}" fill="none" stroke="#d1495b" stroke-width="2.2"/>
  <polyline points="{_polyline(cube_path)}" fill="none" stroke="#00798c" stroke-width="2.4"/>
  <rect x="{start_x - cube_size_px / 2:.2f}" y="{start_y - cube_size_px / 2:.2f}" width="{cube_size_px:.2f}" height="{cube_size_px:.2f}" fill="none" stroke="#00798c" stroke-width="1.4"/>
  <rect x="{end_x - cube_size_px / 2:.2f}" y="{end_y - cube_size_px / 2:.2f}" width="{cube_size_px:.2f}" height="{cube_size_px:.2f}" fill="rgba(0,121,140,0.12)" stroke="#00798c" stroke-width="1.4"/>
  {f'<circle cx="{contact_x:.2f}" cy="{contact_y:.2f}" r="5" fill="#edae49"/>' if contact_x is not None else ''}
  <g class="legend">
    <text x="18" y="{height-18}">blue=cube, red=TCP, gray=target, yellow=first measured contact</text>
  </g>
</svg>
"""


def _side_svg(env: dict[str, Any], rows: list[dict[str, str]]) -> str:
    width = 520
    height = 260
    margin = 34
    side_rows: list[dict[str, str]] = []
    for row in rows:
        side_rows.append({"h": row.get("cube_y_m", "0"), "z": row.get("cube_z_m", "0")})
        side_rows.append({"h": row.get("tcp_y_m", "0"), "z": row.get("tcp_z_m", "0")})
        side_rows.append({"h": row.get("target_y_m", "0"), "z": row.get("target_z_m", "0")})
    frame, cube_path = _scale_points(side_rows, x_key="h", y_key="z", width=width, height=height, margin=margin)

    def project(h: float, z: float) -> tuple[float, float]:
        px = margin + (h - frame["xmin"]) * frame["scale"]
        py = margin + (frame["ymax"] - z) * frame["scale"]
        return px, py

    tcp_path = [project(_float(row.get("tcp_y_m")), _float(row.get("tcp_z_m"))) for row in rows]
    target_path = [project(_float(row.get("target_y_m")), _float(row.get("target_z_m"))) for row in rows]
    return f"""
<svg viewBox="0 0 {width} {height}" class="plot" role="img" aria-label="side env {env['env_id']}">
  <rect width="{width}" height="{height}" fill="#fbfbf7"/>
  <text x="18" y="24" class="caption">env {env['env_id']} side view: y/z height sanity</text>
  <polyline points="{_polyline(target_path)}" fill="none" stroke="#8a8a8a" stroke-width="1.5" stroke-dasharray="4 4"/>
  <polyline points="{_polyline(tcp_path)}" fill="none" stroke="#d1495b" stroke-width="2.2"/>
  <polyline points="{_polyline(cube_path)}" fill="none" stroke="#00798c" stroke-width="2.4"/>
  <text x="18" y="{height-18}" class="legend">blue=cube center z, red=TCP z, gray=target z</text>
</svg>
"""


def _env_metrics(rows: list[dict[str, str]], rollout: dict[str, str] | None) -> dict[str, Any]:
    first = rows[0]
    push_dx = _float(first.get("push_dx"))
    push_dy = _float(first.get("push_dy"))
    start_x = _float(first.get("cube_x_m"))
    start_y = _float(first.get("cube_y_m"))
    start_z = _float(first.get("cube_z_m"))
    first_along_1mm_step = -1
    first_speed_step = -1
    first_z_delta_step = -1
    for row in rows:
        disp = _float(row.get("disp_along_push_m"))
        speed = _float(row.get("cube_speed_mps"))
        z_delta = abs(_float(row.get("cube_z_m")) - start_z)
        step = _int(row.get("step"), -1)
        if first_along_1mm_step < 0 and disp >= 0.001:
            first_along_1mm_step = step
        if first_speed_step < 0 and speed >= 0.02:
            first_speed_step = step
        if first_z_delta_step < 0 and z_delta >= 0.002:
            first_z_delta_step = step
    first_candidates = [s for s in (first_along_1mm_step, first_speed_step, first_z_delta_step) if s >= 0]
    reaction_step = min(first_candidates) if first_candidates else -1

    first_contact = _int((rollout or {}).get("first_contact_step"), -1)
    first_stop = _int((rollout or {}).get("first_stop_step"), -1)
    max_disp = max(_float(row.get("disp_along_push_m")) for row in rows)
    final_disp = _float(rows[-1].get("disp_along_push_m"))
    max_z_delta = max(abs(_float(row.get("cube_z_m")) - start_z) for row in rows)
    max_tip = max(abs(_float(row.get("tip_angle_deg"))) for row in rows)
    max_speed = max(abs(_float(row.get("cube_speed_mps"))) for row in rows)
    clip_rate = _mean([1.0 if _float(row.get("clip_any")) >= 0.5 else 0.0 for row in rows])
    final_x = _float(rows[-1].get("cube_x_m"))
    final_y = _float(rows[-1].get("cube_y_m"))

    return {
        "env_id": _int(first.get("env_id")),
        "rows": len(rows),
        "push_dx": push_dx,
        "push_dy": push_dy,
        "cube_start_xy_m": [start_x, start_y],
        "cube_final_xy_m": [final_x, final_y],
        "cube_delta_xy_m": [final_x - start_x, final_y - start_y],
        "first_reaction_step": reaction_step,
        "first_along_1mm_step": first_along_1mm_step,
        "first_speed_step": first_speed_step,
        "first_z_delta_step": first_z_delta_step,
        "first_contact_step": first_contact,
        "first_stop_step": first_stop,
        "max_disp_along_push_m": max_disp,
        "final_disp_along_push_m": final_disp,
        "max_cube_z_delta_m": max_z_delta,
        "max_tip_angle_deg": max_tip,
        "max_cube_speed_mps": max_speed,
        "clip_any_rate_trace": clip_rate,
        "measured_contact_seen": bool(_int((rollout or {}).get("measured_contact_seen"), 0)),
        "contact_stop_seen": bool(_int((rollout or {}).get("contact_stop_seen"), 0)),
        "overshoot_seen": bool(_int((rollout or {}).get("contact_overshoot_seen"), 0)),
        "reaction_event": bool(_int((rollout or {}).get("reaction_event"), 0)),
        "controlled_push": bool(_int((rollout or {}).get("controlled_push"), 0)),
    }


def _html_doc(artifact: dict[str, Any], rows_by_env: dict[int, list[dict[str, str]]]) -> str:
    envs = artifact["envs"]
    focus_env = artifact["focus_env"]
    style = """
body { font-family: Inter, Arial, sans-serif; margin: 24px; color: #1e1f1c; background: #f6f3ed; }
h1, h2 { margin: 0 0 10px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
.panel { background: #fffdf8; border: 1px solid #d8d2c4; border-radius: 8px; padding: 14px; }
.plot { width: 100%; height: auto; border: 1px solid #d8d2c4; border-radius: 6px; }
.caption { font-size: 13px; font-weight: 700; }
.axis, .legend { font-size: 11px; fill: #333; }
table { width: 100%; border-collapse: collapse; font-size: 13px; background: #fffdf8; }
th, td { border-bottom: 1px solid #ddd6c9; padding: 6px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.warn { color: #8a4b08; font-weight: 700; }
.ok { color: #05605e; font-weight: 700; }
"""
    rows_html = "\n".join(
        "<tr>"
        f"<td>{env['env_id']}</td>"
        f"<td>{env['first_reaction_step']}</td>"
        f"<td>{env['first_along_1mm_step']}</td>"
        f"<td>{env['first_contact_step']}</td>"
        f"<td>{env['first_stop_step']}</td>"
        f"<td>{env['max_disp_along_push_m']:.6f}</td>"
        f"<td>{env['final_disp_along_push_m']:.6f}</td>"
        f"<td>{env['max_cube_z_delta_m']:.6f}</td>"
        f"<td>{env['max_tip_angle_deg']:.3f}</td>"
        f"<td>{env['clip_any_rate_trace']:.3f}</td>"
        f"<td>{'yes' if env['controlled_push'] else 'no'}</td>"
        "</tr>"
        for env in envs
    )
    small_multiples = "\n".join(
        f"<div class='panel'>{_topdown_svg(env, rows_by_env[int(env['env_id'])])}</div>" for env in envs
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>cube10cm seed962 visual sanity trace storyboard</title>
<style>{style}</style>
</head>
<body>
<h1>cube10cm seed962 visual sanity trace storyboard</h1>
<p><span class="ok">Local trace replay only.</span> No IsaacLab runtime, GPU, dataset generation, training, robot control, SSH, or source trace mutation. Actual rendered video is still <span class="warn">not run</span>.</p>
<p>Branch: {html.escape(str(artifact['branch']))}. Objective: reaction/contact tap first; final relocation is secondary.</p>
<div class="grid">
  <div class="panel">{_topdown_svg(focus_env, rows_by_env[int(focus_env['env_id'])])}</div>
  <div class="panel">{_side_svg(focus_env, rows_by_env[int(focus_env['env_id'])])}</div>
</div>
<h2>Env Metrics</h2>
<table>
<thead><tr><th>env</th><th>any react</th><th>along 1mm</th><th>contact step</th><th>stop step</th><th>max disp m</th><th>final disp m</th><th>max z m</th><th>tip deg</th><th>clip</th><th>controlled</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<h2>All Env Top-Down Small Multiples</h2>
<div class="grid">{small_multiples}</div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_csv", type=Path, default=DEFAULT_TRACE_CSV)
    parser.add_argument("--rollout_csv", type=Path, default=DEFAULT_ROLLOUT_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--reaction_gate_json", type=Path, default=DEFAULT_REACTION_GATE_JSON)
    parser.add_argument("--revalidation_json", type=Path, default=DEFAULT_REVALIDATION_JSON)
    parser.add_argument("--focus_env_id", type=int, default=0)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_html", type=Path, default=DEFAULT_OUT_HTML)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    trace_rows = _read_csv(args.trace_csv)
    rollout_rows = _read_csv(args.rollout_csv)
    summary = _load_json(args.summary_json)
    reaction_gate = _load_json(args.reaction_gate_json)
    revalidation = _load_json(args.revalidation_json)
    seed = summary.get("seed")
    if seed is None:
        match = re.search(r"seed(\d+)", args.trace_csv.name)
        seed = int(match.group(1)) if match else None

    rows_by_env: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        rows_by_env[_int(row.get("env_id"))].append(row)
    rows_by_env = {env: sorted(rows, key=lambda row: _int(row.get("step"))) for env, rows in rows_by_env.items()}

    rollout_by_env = {_int(row.get("env_id")): row for row in rollout_rows}
    envs = [
        _env_metrics(rows_by_env[env_id], rollout_by_env.get(env_id)) for env_id in sorted(rows_by_env)
    ]
    focus_env = next((env for env in envs if env["env_id"] == int(args.focus_env_id)), envs[0])

    all_yplus = all(abs(env["push_dx"]) < 1e-9 and env["push_dy"] > 0.5 for env in envs)
    contact_ok = all(env["measured_contact_seen"] for env in envs)
    reaction_ok = all(env["reaction_event"] for env in envs)
    no_overshoot = not any(env["overshoot_seen"] for env in envs)
    visual_motion_consistent = all(env["max_disp_along_push_m"] >= 0.001 for env in envs)
    clip_saturated = all(env["clip_any_rate_trace"] >= 0.99 for env in envs)

    artifact = {
        "artifact_type": "cube10cm_visual_sanity_trace_storyboard_v1",
        "branch": "professor_cube10cm_tap_reaction",
        "local_trace_visual_only": True,
        "no_gpu_isaaclab_runtime_dataset_training_robot_ssh": True,
        "actual_render_video_run": False,
        "inputs": {
            "trace_csv": str(args.trace_csv.resolve()),
            "rollout_csv": str(args.rollout_csv.resolve()),
            "summary_json": str(args.summary_json.resolve()),
            "reaction_gate_json": str(args.reaction_gate_json.resolve()),
            "revalidation_json": str(args.revalidation_json.resolve()),
        },
        "source_runtime": {
            "seed": seed,
            "num_envs": summary.get("num_envs"),
            "fixed_cube_x_m": summary.get("fixed_cube_x_m"),
            "fixed_cube_y_m": summary.get("fixed_cube_y_m"),
            "fixed_push_dir": summary.get("fixed_push_dir"),
            "precontact_clearance_m": summary.get("precontact_clearance_m"),
            "base_lateral_offset_m": summary.get("base_lateral_offset_m"),
            "record_video": summary.get("record_video"),
        },
        "reaction_gate": {
            "reaction_gate_pass": reaction_gate.get("reaction_gate_pass"),
            "reaction_event_rate": reaction_gate.get("reaction_event_rate"),
            "contact_evidence_rate": reaction_gate.get("contact_evidence_rate"),
            "no_posewrite": reaction_gate.get("no_posewrite"),
            "overshoot_rate": reaction_gate.get("overshoot_rate"),
            "teacher_quality_ready": reaction_gate.get("teacher_quality_ready"),
        },
        "teacher_quality_visual_context": {
            "best_policy": revalidation.get("best_policy", {}).get("policy"),
            "best_policy_relative_window": revalidation.get("best_policy", {}).get("relative_window"),
            "best_policy_tiers": revalidation.get("best_policy", {}).get("quality_tier_counts"),
            "best_policy_clip_any_rate_mean": revalidation.get("best_policy", {}).get("accepted_clip_any_rate_mean"),
            "best_policy_follow_p95_to_cap_p95": revalidation.get("best_policy", {}).get(
                "accepted_follow_p95_to_cap_p95"
            ),
            "strict_clean_count": revalidation.get("best_policy", {}).get("strict_clean_count"),
        },
        "sanity_checks": {
            "all_yplus": all_yplus,
            "contact_seen_all_envs": contact_ok,
            "reaction_event_all_envs": reaction_ok,
            "no_overshoot_all_envs": no_overshoot,
            "max_1mm_motion_all_envs": visual_motion_consistent,
            "clip_saturated_all_envs": clip_saturated,
            "needs_actual_render_video": True,
        },
        "envs": envs,
        "focus_env": focus_env,
        "outputs": {
            "html": str(args.out_html.resolve()),
            "json": str(args.out_json.resolve()),
            "summary": str(args.out_summary.resolve()),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_html.write_text(_html_doc(artifact, rows_by_env), encoding="utf-8")
    args.out_summary.write_text(
        "\n".join(
            [
                "line1 artifact=cube10cm_visual_sanity_trace_storyboard_v1 local_trace_visual_only=YES "
                "gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
                f"line2 source trace_rows={len(trace_rows)} envs={len(envs)} seed={seed} "
                f"fixed_cube=({summary.get('fixed_cube_x_m')},{summary.get('fixed_cube_y_m')}) "
                f"push_dir={summary.get('fixed_push_dir')} precontact={summary.get('precontact_clearance_m')}",
                f"line3 reaction_gate pass={reaction_gate.get('reaction_gate_pass')} "
                f"reaction={reaction_gate.get('reaction_event_rate')} contact={reaction_gate.get('contact_evidence_rate')} "
                f"no_posewrite={reaction_gate.get('no_posewrite')} overshoot={reaction_gate.get('overshoot_rate')} "
                f"teacher_quality_ready={reaction_gate.get('teacher_quality_ready')}",
                f"line4 teacher_context best_policy={revalidation.get('best_policy', {}).get('policy')} "
                f"rel={revalidation.get('best_policy', {}).get('relative_window')} "
                f"tiers={revalidation.get('best_policy', {}).get('quality_tier_counts')} "
                f"clip_mean={revalidation.get('best_policy', {}).get('accepted_clip_any_rate_mean')} "
                f"follow_p95_to_cap={revalidation.get('best_policy', {}).get('accepted_follow_p95_to_cap_p95')}",
                f"line5 focus_env env={focus_env['env_id']} first_any_reaction_step={focus_env['first_reaction_step']} "
                f"first_along_1mm_step={focus_env['first_along_1mm_step']} "
                f"first_speed_step={focus_env['first_speed_step']} first_z_delta_step={focus_env['first_z_delta_step']} "
                f"first_contact_step={focus_env['first_contact_step']} first_stop_step={focus_env['first_stop_step']} "
                f"max_disp={focus_env['max_disp_along_push_m']:.9f} final_disp={focus_env['final_disp_along_push_m']:.9f} "
                f"max_z_delta={focus_env['max_cube_z_delta_m']:.9f} tip_deg={focus_env['max_tip_angle_deg']:.9f} "
                f"clip={focus_env['clip_any_rate_trace']:.9f}",
                f"line6 visual_sanity all_yplus={all_yplus} contact_all={contact_ok} reaction_all={reaction_ok} "
                f"no_overshoot_all={no_overshoot} max1mm_all={visual_motion_consistent} "
                f"clip_saturated_all={clip_saturated} actual_render_video=NOT_RUN",
                f"line7 outputs html={args.out_html} json={args.out_json}",
                "line8 verdict trace_storyboard_ready=YES next=one_tiny_local_record_video_run_only_after_explicit_approval",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
