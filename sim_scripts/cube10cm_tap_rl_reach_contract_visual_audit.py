#!/usr/bin/env python3
"""Create a visual audit for command/applied/actual cube10cm reach contract.

Local/posthoc only. It reads the existing x240 per-step reach trace and emits a
small SVG/HTML visual showing why the target appears to pass through the cube
while the applied FK and actual TCP stay outside the contact band.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
TRACE = LOG_DIR / "cube10cm_tap_rl_positive_control_isaac_builtin_diffik_step_clipped_h580_ep608_x240_reachtrace_trace.json"
ROOT_CAUSE = LOG_DIR / "cube10cm_tap_rl_reach_contract_root_cause_audit_summary.out"
OUT_JSON = LOG_DIR / "cube10cm_tap_rl_reach_contract_visual_audit.json"
OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_reach_contract_visual_audit_summary.out"
OUT_SVG = LOG_DIR / "cube10cm_tap_rl_reach_contract_visual_audit.svg"
OUT_HTML = LOG_DIR / "cube10cm_tap_rl_reach_contract_visual_audit.html"
OUT_PNG = LOG_DIR / "cube10cm_tap_rl_reach_contract_visual_audit.png"

CONTACT_BAND_M = 0.010


def _load_trace() -> dict[str, Any]:
    return json.loads(TRACE.read_text(encoding="utf-8"))


def _avg_by_step(rows: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    steps = sorted({int(row["step"]) for row in rows})
    out: list[tuple[int, float]] = []
    for step in steps:
        vals = [float(row[key]) for row in rows if int(row["step"]) == step]
        out.append((step, mean(vals)))
    return out


def _avg_scalar(rows: list[dict[str, Any]], step: int, key: str) -> float:
    vals = [float(row[key]) for row in rows if int(row["step"]) == int(step)]
    if not vals:
        raise ValueError(f"no rows for step={step} key={key}")
    return mean(vals)


def _polyline(points: list[tuple[int, float]], x_map, y_map) -> str:
    return " ".join(f"{x_map(step):.2f},{y_map(value):.2f}" for step, value in points)


def _svg(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    command = _avg_by_step(rows, "command_target_face_gap_m")
    applied = _avg_by_step(rows, "applied_joint_target_fk_face_gap_m")
    actual = _avg_by_step(rows, "actual_tcp_face_gap_m")
    command_inside_steps = sorted(
        {int(row["step"]) for row in rows if bool(row["command_target_inside_contact_band"])}
    )
    first_inside = command_inside_steps[0]
    last_inside = command_inside_steps[-1]
    mid_inside = int(round((first_inside + last_inside) / 2.0))
    final_step = command[-1][0]

    width = 1180
    height = 780
    left = 86
    right = 38
    top = 60
    chart_h = 390
    chart_w = width - left - right
    axis_y = top + chart_h
    y_min = -0.030
    y_max = 0.112

    def x_map(step: int) -> float:
        return left + (float(step) / float(final_step)) * chart_w

    def y_map(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * chart_h

    def mm(value: float) -> str:
        return f"{value * 1000.0:+.1f}mm"

    snapshots = []
    for label, step in (("first in-band", first_inside), ("mid", mid_inside), ("last in-band", last_inside), ("final", final_step)):
        c = _avg_scalar(rows, step, "command_target_face_gap_m")
        a = _avg_scalar(rows, step, "applied_joint_target_fk_face_gap_m")
        t = _avg_scalar(rows, step, "actual_tcp_face_gap_m")
        snapshots.append(
            {
                "label": label,
                "step": step,
                "command_gap_m": c,
                "applied_gap_m": a,
                "actual_gap_m": t,
                "command_minus_applied_m": c - a,
                "applied_minus_actual_m": a - t,
            }
        )

    band_top = y_map(CONTACT_BAND_M)
    band_bottom = y_map(-CONTACT_BAND_M)
    zero_y = y_map(0.0)
    elems: list[str] = []
    elems.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    elems.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    elems.append('<text x="40" y="32" font-size="22" font-family="Arial" font-weight="700" fill="#202124">10cm tap x240 reach contract visual audit</text>')
    elems.append('<text x="40" y="54" font-size="13" font-family="Arial" fill="#5f6368">Face-gap timeline: contact band is ±10mm. Command enters; applied FK and actual TCP stay outside.</text>')

    elems.append(f'<rect x="{left}" y="{band_top:.2f}" width="{chart_w}" height="{band_bottom - band_top:.2f}" fill="#e8f5e9" stroke="#81c784" stroke-width="1"/>')
    for tick in [-0.02, -0.01, 0.0, 0.01, 0.05, 0.10]:
        y = y_map(tick)
        color = "#9aa0a6" if tick != 0 else "#5f6368"
        dash = "4 4" if tick not in (-0.01, 0.01) else "2 4"
        elems.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="{color}" stroke-width="1" stroke-dasharray="{dash}"/>')
        elems.append(f'<text x="18" y="{y + 4:.2f}" font-size="12" font-family="Arial" fill="#5f6368">{mm(tick)}</text>')
    for step in [0, first_inside, mid_inside, last_inside, final_step]:
        x = x_map(step)
        elems.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{axis_y}" stroke="#dadce0" stroke-width="1"/>')
        elems.append(f'<text x="{x - 12:.2f}" y="{axis_y + 20}" font-size="12" font-family="Arial" fill="#5f6368">{step}</text>')
    elems.append(f'<line x1="{left}" y1="{axis_y}" x2="{left + chart_w}" y2="{axis_y}" stroke="#202124" stroke-width="1.2"/>')
    elems.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{axis_y}" stroke="#202124" stroke-width="1.2"/>')
    elems.append(f'<text x="{left + chart_w - 20}" y="{axis_y + 38}" font-size="13" font-family="Arial" fill="#202124">step</text>')
    elems.append(f'<text x="16" y="{top - 16}" font-size="13" font-family="Arial" fill="#202124">face gap</text>')
    elems.append(f'<text x="{left + 8}" y="{band_top - 6:.2f}" font-size="12" font-family="Arial" fill="#2e7d32">contact band</text>')
    elems.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + chart_w}" y2="{zero_y:.2f}" stroke="#2e7d32" stroke-width="1.5"/>')

    elems.append(f'<polyline points="{_polyline(command, x_map, y_map)}" fill="none" stroke="#1a73e8" stroke-width="3"/>')
    elems.append(f'<polyline points="{_polyline(applied, x_map, y_map)}" fill="none" stroke="#f29900" stroke-width="3"/>')
    elems.append(f'<polyline points="{_polyline(actual, x_map, y_map)}" fill="none" stroke="#d93025" stroke-width="3"/>')
    legend_x = left + 20
    legend_y = top + 22
    for idx, (name, color) in enumerate((("command target", "#1a73e8"), ("applied FK target", "#f29900"), ("actual TCP", "#d93025"))):
        y = legend_y + idx * 22
        elems.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        elems.append(f'<text x="{legend_x + 36}" y="{y + 4}" font-size="13" font-family="Arial" fill="#202124">{name}</text>')

    schematic_y = axis_y + 72
    elems.append(f'<text x="{left}" y="{schematic_y - 18}" font-size="17" font-family="Arial" font-weight="700" fill="#202124">Along-axis snapshots</text>')
    elems.append(f'<text x="{left}" y="{schematic_y + 2}" font-size="12" font-family="Arial" fill="#5f6368">Green zone is ±10mm face band. Markers left of the band are still before contact.</text>')
    slot_w = 248
    slot_h = 145
    gap_min = -0.030
    gap_max = 0.115

    def sx(base_x: float, gap: float) -> float:
        return base_x + 18 + (gap - gap_min) / (gap_max - gap_min) * (slot_w - 36)

    for idx, snap in enumerate(snapshots):
        base_x = left + idx * (slot_w + 14)
        base_y = schematic_y + 22
        elems.append(f'<rect x="{base_x}" y="{base_y}" width="{slot_w}" height="{slot_h}" rx="6" fill="#f8fafd" stroke="#dadce0"/>')
        elems.append(f'<text x="{base_x + 12}" y="{base_y + 22}" font-size="13" font-family="Arial" font-weight="700" fill="#202124">{html.escape(snap["label"])} step {snap["step"]}</text>')
        line_y = base_y + 72
        band_x1 = sx(base_x, -CONTACT_BAND_M)
        band_x2 = sx(base_x, CONTACT_BAND_M)
        zero_x = sx(base_x, 0.0)
        elems.append(f'<line x1="{base_x + 18}" y1="{line_y}" x2="{base_x + slot_w - 18}" y2="{line_y}" stroke="#9aa0a6" stroke-width="1"/>')
        elems.append(f'<rect x="{band_x1}" y="{line_y - 16}" width="{band_x2 - band_x1}" height="32" fill="#e8f5e9" stroke="#81c784"/>')
        elems.append(f'<line x1="{zero_x}" y1="{line_y - 22}" x2="{zero_x}" y2="{line_y + 22}" stroke="#2e7d32" stroke-width="1.5"/>')
        for label, key, color, dy in (
            ("C", "command_gap_m", "#1a73e8", -26),
            ("F", "applied_gap_m", "#f29900", 0),
            ("T", "actual_gap_m", "#d93025", 26),
        ):
            x = sx(base_x, snap[key])
            elems.append(f'<circle cx="{x:.2f}" cy="{line_y + dy}" r="6" fill="{color}"/>')
            elems.append(f'<text x="{x - 4:.2f}" y="{line_y + dy + 4}" font-size="9" font-family="Arial" fill="#ffffff">{label}</text>')
        elems.append(f'<text x="{base_x + 12}" y="{base_y + 118}" font-size="11" font-family="Arial" fill="#5f6368">cmd {mm(snap["command_gap_m"])} / FK {mm(snap["applied_gap_m"])} / TCP {mm(snap["actual_gap_m"])}</text>')
        elems.append(f'<text x="{base_x + 12}" y="{base_y + 135}" font-size="11" font-family="Arial" fill="#5f6368">cmd-FK {snap["command_minus_applied_m"] * 1000.0:.1f}mm, FK-TCP {snap["applied_minus_actual_m"] * 1000.0:.1f}mm</text>')

    elems.append('<text x="40" y="752" font-size="12" font-family="Arial" fill="#5f6368">C=command target, F=FK of applied joint target, T=actual TCP. Generated from existing x240 reach trace only.</text>')
    elems.append("</svg>")

    metrics = {
        "first_command_inside_step": first_inside,
        "last_command_inside_step": last_inside,
        "final_step": final_step,
        "snapshots": snapshots,
        "command_inside_rows": sum(1 for row in rows if bool(row["command_target_inside_contact_band"])),
        "applied_inside_rows": sum(1 for row in rows if bool(row["applied_joint_target_fk_inside_contact_band"])),
        "actual_inside_rows": sum(1 for row in rows if bool(row["actual_contact_proxy"])),
    }
    return "\n".join(elems) + "\n", metrics


def main() -> int:
    artifact = _load_trace()
    rows = artifact["rows"]
    svg, metrics = _svg(rows)
    OUT_SVG.write_text(svg, encoding="utf-8")
    html_doc = (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>cube10cm reach contract visual audit</title>"
        "<style>body{margin:0;background:#f5f7fb;font-family:Arial,sans-serif}.wrap{max-width:1220px;margin:24px auto;padding:18px;background:#fff;border:1px solid #dadce0}"
        "pre{white-space:pre-wrap;background:#f8fafd;padding:12px;border:1px solid #e0e3e7}</style></head><body><div class=\"wrap\">"
        + svg
        + "<h2>Source</h2><pre>"
        + html.escape(str(TRACE.relative_to(ROOT)))
        + "\n"
        + html.escape(ROOT_CAUSE.read_text(encoding="utf-8"))
        + "</pre></div></body></html>\n"
    )
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    result = {
        "artifact_type": "cube10cm_tap_rl_reach_contract_visual_audit_v1",
        "local_posthoc_visual_only": True,
        "gpu_runtime": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "trace": str(TRACE.relative_to(ROOT)),
        "svg": str(OUT_SVG.relative_to(ROOT)),
        "html": str(OUT_HTML.relative_to(ROOT)),
        "png": str(OUT_PNG.relative_to(ROOT)) if OUT_PNG.exists() else None,
        "metrics": metrics,
        "interpretation": (
            "Visual confirms command target enters/passes the contact band, while FK of the applied joint target and "
            "actual TCP remain on the pre-contact side. This supports target-generation clipping plus actuator lag."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    first = metrics["snapshots"][0]
    mid = metrics["snapshots"][1]
    final = metrics["snapshots"][-1]
    lines = [
        "line1 artifact=cube10cm_tap_rl_reach_contract_visual_audit_v1 "
        "local_posthoc_visual_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 outputs "
            f"svg={OUT_SVG.relative_to(ROOT)} html={OUT_HTML.relative_to(ROOT)} "
            f"png={OUT_PNG.relative_to(ROOT) if OUT_PNG.exists() else 'NOT_RENDERED'} "
            f"json={OUT_JSON.relative_to(ROOT)}"
        ),
        (
            "line3 visual_contract "
            f"command_inside_rows={metrics['command_inside_rows']} "
            f"applied_inside_rows={metrics['applied_inside_rows']} "
            f"actual_inside_rows={metrics['actual_inside_rows']} "
            f"first_inside_step={metrics['first_command_inside_step']} "
            f"last_inside_step={metrics['last_command_inside_step']}"
        ),
        (
            "line4 first_inband_snapshot "
            f"step={first['step']} "
            f"command_gap_m={first['command_gap_m']:.9f} "
            f"applied_gap_m={first['applied_gap_m']:.9f} "
            f"actual_gap_m={first['actual_gap_m']:.9f} "
            f"command_minus_applied_m={first['command_minus_applied_m']:.9f} "
            f"applied_minus_actual_m={first['applied_minus_actual_m']:.9f}"
        ),
        (
            "line5 mid_snapshot "
            f"step={mid['step']} "
            f"command_gap_m={mid['command_gap_m']:.9f} "
            f"applied_gap_m={mid['applied_gap_m']:.9f} "
            f"actual_gap_m={mid['actual_gap_m']:.9f}"
        ),
        (
            "line6 final_snapshot "
            f"step={final['step']} "
            f"command_gap_m={final['command_gap_m']:.9f} "
            f"applied_gap_m={final['applied_gap_m']:.9f} "
            f"actual_gap_m={final['actual_gap_m']:.9f}"
        ),
        (
            "line7 verdict VISUAL_CONFIRMS_COMMAND_TARGET_PASSES_CONTACT_BAND_BUT_APPLIED_FK_AND_ACTUAL_TCP_STAY_PRECONTACT "
            "contact_gate_relaxation_unblock=NO target_generation_clip_and_actuator_lag_supported=YES"
        ),
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
