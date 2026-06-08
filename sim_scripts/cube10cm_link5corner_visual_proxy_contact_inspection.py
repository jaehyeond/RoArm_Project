"""Local visual proxy-contact inspection for the cube10cm link5-corner runtime.

This script reads an existing trace only. It does not run IsaacLab, use GPU,
generate data, train, control a robot, SSH, or mutate the source trace.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_TRACE_CSV = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_trace.csv"
)
DEFAULT_ROLLOUT_CSV = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962.csv"
)
DEFAULT_SUMMARY_JSON = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_summary.json"
)
DEFAULT_REACTION_GATE_JSON = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_reaction_gate_audit.json"
)
DEFAULT_TRACE_DIAG_JSON = (
    LOG_DIR
    / "diffik_probe_cube10cm_m072_fixed_yplus16_goodxy_latneg020_link5corner_position_seed962_trace_diagnostic_summary.json"
)
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_link5corner_visual_proxy_contact_inspection.json"
DEFAULT_OUT_HTML = LOG_DIR / "cube10cm_link5corner_visual_proxy_contact_inspection.html"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_link5corner_visual_proxy_contact_inspection_summary.out"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _rate(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _vec(row: dict[str, str], prefix: str) -> list[float]:
    return [_f(row.get(f"{prefix}_x_m")), _f(row.get(f"{prefix}_y_m")), _f(row.get(f"{prefix}_z_m"))]


def _xyz(row: dict[str, str], x_key: str, y_key: str, z_key: str) -> list[float]:
    return [_f(row.get(x_key)), _f(row.get(y_key)), _f(row.get(z_key))]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[idx] - b[idx] for idx in range(3)]


def _dot2(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _norm3(a: list[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _push_unit(row: dict[str, str]) -> list[float]:
    dx = _f(row.get("push_dx"))
    dy = _f(row.get("push_dy"))
    n = math.sqrt(dx * dx + dy * dy)
    if n < 1.0e-9:
        return [0.0, 1.0]
    return [dx / n, dy / n]


def _lateral_unit(push: list[float]) -> list[float]:
    return [push[1], -push[0]]


def _row_metrics(row: dict[str, str]) -> dict[str, Any]:
    push = _push_unit(row)
    lateral = _lateral_unit(push)
    cube = _vec(row, "cube")
    proxy = _xyz(row, "tool_proxy_x_after_m", "tool_proxy_y_after_m", "tool_proxy_z_after_m")
    target = _vec(row, "tool_contact_target")
    tcp = _xyz(row, "tcp_x_after_m", "tcp_y_after_m", "tcp_z_after_m") if "tcp_x_after_m" in row else _vec(row, "tcp")
    size = [
        _f(row.get("cube_size_x_m"), 0.1),
        _f(row.get("cube_size_y_m"), 0.1),
        _f(row.get("cube_size_z_m"), 0.1),
    ]
    half_along = abs(push[0]) * size[0] / 2.0 + abs(push[1]) * size[1] / 2.0
    half_lat = abs(lateral[0]) * size[0] / 2.0 + abs(lateral[1]) * size[1] / 2.0
    side_face = [
        cube[0] - push[0] * half_along,
        cube[1] - push[1] * half_along,
        cube[2],
    ]

    def along_gap(point: list[float]) -> float:
        return _dot2([point[0] - side_face[0], point[1] - side_face[1]], push)

    def lateral_offset(point: list[float]) -> float:
        return _dot2([point[0] - cube[0], point[1] - cube[1]], lateral)

    target_err = _norm3(_sub(proxy, target))
    return {
        "env_id": _i(row.get("env_id")),
        "step": _i(row.get("step")),
        "frame": _i(row.get("frame")),
        "phase_alpha": _f(row.get("phase_alpha")),
        "cube": cube,
        "proxy": proxy,
        "target": target,
        "tcp": tcp,
        "cube_size": size,
        "half_along_m": half_along,
        "half_lateral_m": half_lat,
        "push_unit": push,
        "lateral_unit": lateral,
        "proxy_gap_to_live_side_face_m": along_gap(proxy),
        "target_gap_to_live_side_face_m": along_gap(target),
        "tcp_gap_to_live_side_face_m": along_gap(tcp),
        "proxy_lateral_from_cube_center_m": lateral_offset(proxy),
        "target_lateral_from_cube_center_m": lateral_offset(target),
        "tcp_lateral_from_cube_center_m": lateral_offset(tcp),
        "proxy_minus_target_lateral_m": _dot2([proxy[0] - target[0], proxy[1] - target[1]], lateral),
        "proxy_minus_live_cube_center_z_m": proxy[2] - cube[2],
        "target_minus_live_cube_center_z_m": target[2] - cube[2],
        "tcp_minus_live_cube_center_z_m": tcp[2] - cube[2],
        "proxy_below_live_cube_top_m": cube[2] + size[2] / 2.0 - proxy[2],
        "target_below_live_cube_top_m": cube[2] + size[2] / 2.0 - target[2],
        "tcp_below_live_cube_top_m": cube[2] + size[2] / 2.0 - tcp[2],
        "proxy_target_err_m": _f(row.get("tool_proxy_target_err_after_m"), target_err),
        "proxy_target_z_err_m": _f(row.get("tool_proxy_target_z_err_after_m"), proxy[2] - target[2]),
        "tcp_target_err_m": _f(row.get("tcp_target_err_after_m")),
        "disp_along_push_m": _f(row.get("disp_along_push_m")),
        "cube_speed_mps": _f(row.get("cube_speed_mps")),
        "tip_angle_deg": _f(row.get("tip_angle_deg")),
        "clip_any": bool(_i(row.get("clip_any"))),
        "clip_max_joint_name": row.get("clip_max_joint_name", ""),
        "measured_contact_now": bool(_i(row.get("measured_contact_now"))),
        "measured_contact_seen": bool(_i(row.get("measured_contact_seen"))),
        "contact_stop_seen": bool(_i(row.get("contact_stop_seen"))),
    }


def _event_row(rows: list[dict[str, str]], step: int) -> dict[str, str]:
    if step < 0:
        return rows[0]
    return next((row for row in rows if _i(row.get("step")) >= step), rows[-1])


def _env_detail(env_id: int, rows: list[dict[str, str]], rollout: dict[str, str] | None) -> dict[str, Any]:
    rollout = rollout or {}
    contact_step = _i(rollout.get("first_contact_step"), -1)
    stop_step = _i(rollout.get("first_stop_step"), -1)
    contact_row = _event_row(rows, contact_step)
    stop_row = _event_row(rows, stop_step)
    maxdisp_row = max(rows, key=lambda row: _f(row.get("disp_along_push_m")))
    final_row = rows[-1]
    contact = _row_metrics(contact_row)
    stop = _row_metrics(stop_row)
    maxdisp = _row_metrics(maxdisp_row)
    final = _row_metrics(final_row)
    return {
        "env_id": env_id,
        "trace_rows": len(rows),
        "first_contact_step_rollout": contact_step,
        "first_stop_step_rollout": stop_step,
        "contact_trace_step": contact["step"],
        "stop_trace_step": stop["step"],
        "maxdisp_trace_step": maxdisp["step"],
        "contact_stop_same_rollout_step": contact_step >= 0 and contact_step == stop_step,
        "measured_contact_seen": bool(_i(rollout.get("measured_contact_seen"))),
        "contact_overshoot_seen": bool(_i(rollout.get("contact_overshoot_seen"))),
        "reaction_event": bool(_i(rollout.get("reaction_event"))),
        "controlled_push": bool(_i(rollout.get("controlled_push"))),
        "low_motion": bool(_i(rollout.get("low_motion"))),
        "max_disp_along_push_rollout_m": _f(rollout.get("max_disp_along_push_m")),
        "max_cube_speed_rollout_mps": _f(rollout.get("max_cube_speed_mps")),
        "contact": contact,
        "stop": stop,
        "maxdisp": maxdisp,
        "final": final,
    }


def _project(points: list[tuple[float, float]], width: int, height: int, margin: int):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    pad_x = max(0.005, (xmax - xmin) * 0.12)
    pad_y = max(0.005, (ymax - ymin) * 0.12)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y
    scale = min((width - 2 * margin) / max(xmax - xmin, 1.0e-9), (height - 2 * margin) / max(ymax - ymin, 1.0e-9))

    def fn(x: float, y: float) -> tuple[float, float]:
        return margin + (x - xmin) * scale, margin + (ymax - y) * scale

    return fn


def _circle(label: str, x: float, y: float, color: str, radius: int = 5) -> str:
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>'
        f'<text x="{x + 7:.2f}" y="{y - 6:.2f}" class="tiny">{html.escape(label)}</text>'
    )


def _local_topdown_svg(detail: dict[str, Any]) -> str:
    width, height, margin = 560, 420, 42
    event_names = ["contact", "maxdisp", "final"]
    event_points: list[tuple[float, float]] = []
    for name in event_names:
        m = detail[name]
        event_points.extend(
            [
                (m["proxy_lateral_from_cube_center_m"], m["proxy_gap_to_live_side_face_m"]),
                (m["target_lateral_from_cube_center_m"], m["target_gap_to_live_side_face_m"]),
                (m["tcp_lateral_from_cube_center_m"], m["tcp_gap_to_live_side_face_m"]),
            ]
        )
    event_points.extend([(-0.06, -0.02), (0.06, 0.11)])
    project = _project(event_points, width, height, margin)

    def p(lat: float, along: float) -> tuple[float, float]:
        return project(lat, along)

    x0, y0 = p(-0.05, 0.0)
    x1, y1 = p(0.05, 0.10)
    proxy_path = " ".join(
        f"{p(detail[name]['proxy_lateral_from_cube_center_m'], detail[name]['proxy_gap_to_live_side_face_m'])[0]:.2f},"
        f"{p(detail[name]['proxy_lateral_from_cube_center_m'], detail[name]['proxy_gap_to_live_side_face_m'])[1]:.2f}"
        for name in event_names
    )
    target_path = " ".join(
        f"{p(detail[name]['target_lateral_from_cube_center_m'], detail[name]['target_gap_to_live_side_face_m'])[0]:.2f},"
        f"{p(detail[name]['target_lateral_from_cube_center_m'], detail[name]['target_gap_to_live_side_face_m'])[1]:.2f}"
        for name in event_names
    )
    contact = detail["contact"]
    maxdisp = detail["maxdisp"]
    proxy_x, proxy_y = p(contact["proxy_lateral_from_cube_center_m"], contact["proxy_gap_to_live_side_face_m"])
    target_x, target_y = p(contact["target_lateral_from_cube_center_m"], contact["target_gap_to_live_side_face_m"])
    tcp_x, tcp_y = p(contact["tcp_lateral_from_cube_center_m"], contact["tcp_gap_to_live_side_face_m"])
    max_x, max_y = p(maxdisp["proxy_lateral_from_cube_center_m"], maxdisp["proxy_gap_to_live_side_face_m"])
    return f"""
<svg viewBox="0 0 {width} {height}" class="plot" role="img" aria-label="local topdown proxy contact">
  <rect width="{width}" height="{height}" fill="#fffdf8"/>
  <text x="18" y="26" class="caption">env {detail['env_id']} local top-down: lateral vs push-depth</text>
  <rect x="{x0:.2f}" y="{y1:.2f}" width="{x1 - x0:.2f}" height="{y0 - y1:.2f}" fill="#f9d5e5" fill-opacity="0.35" stroke="#222" stroke-width="1.4"/>
  <line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y0:.2f}" stroke="#111" stroke-width="2"/>
  <text x="{x0:.2f}" y="{y0 + 16:.2f}" class="tiny">live approach face</text>
  <polyline points="{target_path}" fill="none" stroke="#d1495b" stroke-width="1.8" stroke-dasharray="4 4"/>
  <polyline points="{proxy_path}" fill="none" stroke="#f2a541" stroke-width="2.4"/>
  {_circle('proxy contact', proxy_x, proxy_y, '#f2a541', 6)}
  {_circle('target contact', target_x, target_y, '#d1495b', 5)}
  {_circle('TCP contact', tcp_x, tcp_y, '#3366cc', 5)}
  {_circle('proxy maxdisp', max_x, max_y, '#7f4acb', 5)}
  <text x="18" y="{height - 26}" class="legend">cube rectangle: lateral +/-50mm, depth 0..100mm. Negative depth means outside/grazing on approach side.</text>
</svg>
"""


def _local_side_svg(detail: dict[str, Any]) -> str:
    width, height, margin = 560, 360, 42
    event_names = ["contact", "maxdisp", "final"]
    points: list[tuple[float, float]] = [(-0.025, -0.065), (0.11, 0.065)]
    for name in event_names:
        m = detail[name]
        points.extend(
            [
                (m["proxy_gap_to_live_side_face_m"], m["proxy_minus_live_cube_center_z_m"]),
                (m["target_gap_to_live_side_face_m"], m["target_minus_live_cube_center_z_m"]),
                (m["tcp_gap_to_live_side_face_m"], m["tcp_minus_live_cube_center_z_m"]),
            ]
        )
    project = _project(points, width, height, margin)

    def p(along: float, z: float) -> tuple[float, float]:
        return project(along, z)

    x0, y_top = p(0.0, 0.05)
    x1, y_bot = p(0.10, -0.05)
    cx0, cy0 = p(0.0, 0.0)
    contact = detail["contact"]
    proxy_x, proxy_y = p(contact["proxy_gap_to_live_side_face_m"], contact["proxy_minus_live_cube_center_z_m"])
    target_x, target_y = p(contact["target_gap_to_live_side_face_m"], contact["target_minus_live_cube_center_z_m"])
    tcp_x, tcp_y = p(contact["tcp_gap_to_live_side_face_m"], contact["tcp_minus_live_cube_center_z_m"])
    maxdisp = detail["maxdisp"]
    max_x, max_y = p(maxdisp["proxy_gap_to_live_side_face_m"], maxdisp["proxy_minus_live_cube_center_z_m"])
    return f"""
<svg viewBox="0 0 {width} {height}" class="plot" role="img" aria-label="local side proxy contact">
  <rect width="{width}" height="{height}" fill="#fffdf8"/>
  <text x="18" y="26" class="caption">env {detail['env_id']} side view: push-depth vs z relative to live cube center</text>
  <rect x="{x0:.2f}" y="{y_top:.2f}" width="{x1 - x0:.2f}" height="{y_bot - y_top:.2f}" fill="#f9d5e5" fill-opacity="0.35" stroke="#222" stroke-width="1.4"/>
  <line x1="{x0:.2f}" y1="{cy0:.2f}" x2="{x1:.2f}" y2="{cy0:.2f}" stroke="#777" stroke-width="1" stroke-dasharray="4 4"/>
  <line x1="{x0:.2f}" y1="{y_top:.2f}" x2="{x1:.2f}" y2="{y_top:.2f}" stroke="#777" stroke-width="1" stroke-dasharray="2 4"/>
  {_circle('proxy contact', proxy_x, proxy_y, '#f2a541', 6)}
  {_circle('target contact', target_x, target_y, '#d1495b', 5)}
  {_circle('TCP contact', tcp_x, tcp_y, '#3366cc', 5)}
  {_circle('proxy maxdisp', max_x, max_y, '#7f4acb', 5)}
  <text x="18" y="{height - 26}" class="legend">z=0 is cube center. Top face is +50mm. Proxy at z~0 means side-center height, not top contact.</text>
</svg>
"""


def _timeline_svg(detail: dict[str, Any], rows: list[dict[str, str]]) -> str:
    width, height, margin = 760, 300, 42
    metrics = [_row_metrics(row) for row in rows]
    xs = [m["step"] for m in metrics]
    ys = [m["disp_along_push_m"] * 1000.0 for m in metrics]
    ys += [m["proxy_gap_to_live_side_face_m"] * 1000.0 for m in metrics]
    ys += [m["target_gap_to_live_side_face_m"] * 1000.0 for m in metrics]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys + [-8.0]), max(ys + [3.0])
    pad_y = max(1.0, (ymax - ymin) * 0.12)
    ymin -= pad_y
    ymax += pad_y

    def p(step: float, value: float) -> tuple[float, float]:
        x = margin + (step - xmin) * (width - 2 * margin) / max(xmax - xmin, 1.0e-9)
        y = margin + (ymax - value) * (height - 2 * margin) / max(ymax - ymin, 1.0e-9)
        return x, y

    def line(values: list[float]) -> str:
        return " ".join(f"{p(metrics[idx]['step'], values[idx])[0]:.2f},{p(metrics[idx]['step'], values[idx])[1]:.2f}" for idx in range(len(metrics)))

    disp_line = line([m["disp_along_push_m"] * 1000.0 for m in metrics])
    proxy_gap_line = line([m["proxy_gap_to_live_side_face_m"] * 1000.0 for m in metrics])
    target_gap_line = line([m["target_gap_to_live_side_face_m"] * 1000.0 for m in metrics])
    zero_y = p(xmin, 0.0)[1]
    contact_x = p(detail["contact_trace_step"], 0.0)[0]
    stop_x = p(detail["stop_trace_step"], 0.0)[0]
    return f"""
<svg viewBox="0 0 {width} {height}" class="plot" role="img" aria-label="timeline">
  <rect width="{width}" height="{height}" fill="#fffdf8"/>
  <text x="18" y="26" class="caption">env {detail['env_id']} timeline: displacement and face gaps</text>
  <line x1="{margin}" y1="{zero_y:.2f}" x2="{width - margin}" y2="{zero_y:.2f}" stroke="#555" stroke-width="1"/>
  <line x1="{contact_x:.2f}" y1="{margin}" x2="{contact_x:.2f}" y2="{height - margin}" stroke="#edae49" stroke-width="1.6"/>
  <line x1="{stop_x:.2f}" y1="{margin}" x2="{stop_x:.2f}" y2="{height - margin}" stroke="#222" stroke-width="1" stroke-dasharray="4 4"/>
  <polyline points="{disp_line}" fill="none" stroke="#00798c" stroke-width="2.2"/>
  <polyline points="{proxy_gap_line}" fill="none" stroke="#f2a541" stroke-width="2.0"/>
  <polyline points="{target_gap_line}" fill="none" stroke="#d1495b" stroke-width="1.8" stroke-dasharray="4 4"/>
  <text x="18" y="{height - 28}" class="legend">teal=disp mm, orange=proxy gap to live face mm, red=target gap mm. Negative gap is outside approach face.</text>
</svg>
"""


def _html_doc(artifact: dict[str, Any], rows_by_env: dict[int, list[dict[str, str]]]) -> str:
    focus = artifact["focus_env"]
    rows = rows_by_env[int(focus["env_id"])]
    env_rows = "\n".join(
        "<tr>"
        f"<td>{env['env_id']}</td>"
        f"<td>{env['contact_trace_step']}</td>"
        f"<td>{env['contact']['proxy_target_err_m'] * 1000.0:.2f}</td>"
        f"<td>{env['contact']['proxy_gap_to_live_side_face_m'] * 1000.0:.2f}</td>"
        f"<td>{env['contact']['target_gap_to_live_side_face_m'] * 1000.0:.2f}</td>"
        f"<td>{env['contact']['proxy_minus_live_cube_center_z_m'] * 1000.0:.2f}</td>"
        f"<td>{env['contact']['proxy_below_live_cube_top_m'] * 1000.0:.2f}</td>"
        f"<td>{env['contact']['tcp_minus_live_cube_center_z_m'] * 1000.0:.2f}</td>"
        f"<td>{env['max_disp_along_push_rollout_m'] * 1000.0:.2f}</td>"
        f"<td>{'yes' if env['contact_stop_same_rollout_step'] else 'no'}</td>"
        f"<td>{'yes' if env['controlled_push'] else 'no'}</td>"
        "</tr>"
        for env in artifact["envs"]
    )
    style = """
body { font-family: Inter, Arial, sans-serif; margin: 24px; color: #20211e; background: #f6f3ed; }
h1, h2 { margin: 0 0 10px; }
p { max-width: 1060px; line-height: 1.45; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; }
.panel { background: #fffdf8; border: 1px solid #d8d2c4; border-radius: 8px; padding: 14px; }
.plot { width: 100%; height: auto; border: 1px solid #d8d2c4; border-radius: 6px; }
.caption { font-size: 13px; font-weight: 700; fill: #20211e; }
.legend, .tiny { font-size: 11px; fill: #333; }
table { width: 100%; border-collapse: collapse; font-size: 13px; background: #fffdf8; }
th, td { border-bottom: 1px solid #ddd6c9; padding: 6px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.ok { color: #05605e; font-weight: 700; }
.warn { color: #8a4b08; font-weight: 700; }
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>cube10cm link5-corner visual proxy-contact inspection</title>
<style>{style}</style>
</head>
<body>
<h1>cube10cm link5-corner visual proxy-contact inspection</h1>
<p><span class="ok">Local trace inspection only.</span> No IsaacLab runtime, GPU, dataset generation, training, robot control, SSH, or trace mutation. Objective remains tap/reaction first; final relocation is secondary.</p>
<p>Critical read: the link5 corner proxy is evaluated against the live cube side face. Negative face gap means the point is still outside/grazing the approach side; z near 0 means side-center height; z near +50mm means top contact.</p>
<div class="grid">
  <div class="panel">{_local_topdown_svg(focus)}</div>
  <div class="panel">{_local_side_svg(focus)}</div>
</div>
<div class="panel">{_timeline_svg(focus, rows)}</div>
<h2>Per-Env Contact Rows</h2>
<table>
<thead><tr><th>env</th><th>contact step</th><th>proxy-target err mm</th><th>proxy face gap mm</th><th>target face gap mm</th><th>proxy z-center mm</th><th>proxy below top mm</th><th>TCP z-center mm</th><th>max disp mm</th><th>stop same</th><th>controlled</th></tr></thead>
<tbody>{env_rows}</tbody>
</table>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_csv", type=Path, default=DEFAULT_TRACE_CSV)
    parser.add_argument("--rollout_csv", type=Path, default=DEFAULT_ROLLOUT_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--reaction_gate_json", type=Path, default=DEFAULT_REACTION_GATE_JSON)
    parser.add_argument("--trace_diag_json", type=Path, default=DEFAULT_TRACE_DIAG_JSON)
    parser.add_argument("--focus_env_id", type=int, default=0)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_html", type=Path, default=DEFAULT_OUT_HTML)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    trace_rows = _read_csv(args.trace_csv)
    rollout_rows = _read_csv(args.rollout_csv)
    summary = _load_json(args.summary_json)
    reaction_gate = _load_json(args.reaction_gate_json)
    trace_diag = _load_json(args.trace_diag_json)
    if not trace_rows:
        raise RuntimeError(f"empty trace: {args.trace_csv}")

    rows_by_env_raw: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        rows_by_env_raw[_i(row.get("env_id"))].append(row)
    rows_by_env = {
        env_id: sorted(rows, key=lambda item: _i(item.get("step"))) for env_id, rows in rows_by_env_raw.items()
    }
    rollout_by_env = {_i(row.get("env_id")): row for row in rollout_rows}
    envs = [_env_detail(env_id, rows_by_env[env_id], rollout_by_env.get(env_id)) for env_id in sorted(rows_by_env)]
    focus_env = next((env for env in envs if env["env_id"] == args.focus_env_id), envs[0])

    contacts = [env["contact"] for env in envs]
    maxdisps = [env["maxdisp"] for env in envs]
    contract_ok = (
        all(row.get("tool_contact_proxy_mode") == "link5_collision_corner_011" for row in trace_rows)
        and all(row.get("diffik_command_type") == "position" for row in trace_rows)
    )
    contact_stop_same_rate = _rate([env["contact_stop_same_rollout_step"] for env in envs])
    proxy_side_center_z_rate = _rate([abs(m["proxy_minus_live_cube_center_z_m"]) <= 0.005 for m in contacts])
    proxy_not_top_rate = _rate([m["proxy_below_live_cube_top_m"] >= 0.030 for m in contacts])
    tcp_near_top_rate = _rate([abs(m["tcp_below_live_cube_top_m"]) <= 0.010 for m in contacts])
    proxy_outside_face_rate = _rate([m["proxy_gap_to_live_side_face_m"] < -0.001 for m in contacts])
    target_outside_face_rate = _rate([m["target_gap_to_live_side_face_m"] < -0.001 for m in contacts])
    proxy_target_5mm_rate = _rate([m["proxy_target_err_m"] <= 0.005 for m in contacts])
    proxy_target_3mm_rate = _rate([m["proxy_target_err_m"] <= 0.003 for m in contacts])
    maxdisp_proxy_outside_face_rate = _rate([m["proxy_gap_to_live_side_face_m"] < -0.001 for m in maxdisps])

    side_center_proxy_visual_verified = bool(proxy_side_center_z_rate >= 0.95 and proxy_not_top_rate >= 0.95)
    top_contact_rejected_for_proxy = bool(proxy_not_top_rate >= 0.95)
    grazing_or_outside_face = bool(proxy_outside_face_rate >= 0.75 and maxdisp_proxy_outside_face_rate >= 0.75)
    early_freeze = bool(contact_stop_same_rate >= 0.95)
    weak_tap_visual_mechanism_supported = bool(
        side_center_proxy_visual_verified and grazing_or_outside_face and early_freeze
    )
    clean_tap_strength_visual_verified = bool(
        side_center_proxy_visual_verified
        and not grazing_or_outside_face
        and _f(summary.get("max_disp_along_push_mean_m")) >= 0.002
        and _f(summary.get("low_motion_rate")) < 0.5
    )

    artifact = {
        "artifact_type": "cube10cm_link5corner_visual_proxy_contact_inspection_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_trace_visual_only": True,
        "no_gpu_isaaclab_runtime_dataset_training_robot_ssh": True,
        "inputs": {
            "trace_csv": str(args.trace_csv.resolve()),
            "rollout_csv": str(args.rollout_csv.resolve()),
            "summary_json": str(args.summary_json.resolve()),
            "reaction_gate_json": str(args.reaction_gate_json.resolve()),
            "trace_diag_json": str(args.trace_diag_json.resolve()),
        },
        "runtime_contract": {
            "contract_ok": contract_ok,
            "tool_contact_proxy_mode": trace_rows[0].get("tool_contact_proxy_mode"),
            "tool_proxy_label": trace_rows[0].get("tool_proxy_label"),
            "diffik_command_type": trace_rows[0].get("diffik_command_type"),
            "trace_rows": len(trace_rows),
            "env_count": len(envs),
            "seed": summary.get("seed"),
            "reaction_gate_pass": reaction_gate.get("reaction_gate_pass"),
            "contact_evidence_rate": reaction_gate.get("contact_evidence_rate"),
            "overshoot_rate": reaction_gate.get("overshoot_rate"),
            "no_posewrite": reaction_gate.get("no_posewrite"),
        },
        "summary_context": {
            "max_disp_along_push_mean_m": summary.get("max_disp_along_push_mean_m"),
            "max_cube_speed_mean_mps": summary.get("max_cube_speed_mean_mps"),
            "low_motion_rate": summary.get("low_motion_rate"),
            "diffik_clip_rate_mean": summary.get("diffik_clip_rate_mean"),
            "final_tool_proxy_target_err_mean_m": summary.get("final_tool_proxy_target_err_mean_m"),
            "trace_clip_any_rate": trace_diag.get("clip_any_rate"),
            "trace_likely_modes": trace_diag.get("likely_modes"),
        },
        "contact_visual_metrics": {
            "proxy_target_err_m": _stats([m["proxy_target_err_m"] for m in contacts]),
            "proxy_target_z_err_m": _stats([m["proxy_target_z_err_m"] for m in contacts]),
            "proxy_gap_to_live_side_face_m": _stats([m["proxy_gap_to_live_side_face_m"] for m in contacts]),
            "target_gap_to_live_side_face_m": _stats([m["target_gap_to_live_side_face_m"] for m in contacts]),
            "proxy_minus_live_cube_center_z_m": _stats([m["proxy_minus_live_cube_center_z_m"] for m in contacts]),
            "target_minus_live_cube_center_z_m": _stats([m["target_minus_live_cube_center_z_m"] for m in contacts]),
            "tcp_minus_live_cube_center_z_m": _stats([m["tcp_minus_live_cube_center_z_m"] for m in contacts]),
            "proxy_below_live_cube_top_m": _stats([m["proxy_below_live_cube_top_m"] for m in contacts]),
            "tcp_below_live_cube_top_m": _stats([m["tcp_below_live_cube_top_m"] for m in contacts]),
            "proxy_lateral_from_cube_center_m": _stats([m["proxy_lateral_from_cube_center_m"] for m in contacts]),
            "target_lateral_from_cube_center_m": _stats([m["target_lateral_from_cube_center_m"] for m in contacts]),
            "proxy_minus_target_lateral_m": _stats([m["proxy_minus_target_lateral_m"] for m in contacts]),
            "proxy_side_center_z_near_5mm_rate": proxy_side_center_z_rate,
            "proxy_not_top_rate": proxy_not_top_rate,
            "tcp_near_top_10mm_rate": tcp_near_top_rate,
            "proxy_outside_live_face_rate": proxy_outside_face_rate,
            "target_outside_live_face_rate": target_outside_face_rate,
            "proxy_target_err_le_5mm_rate": proxy_target_5mm_rate,
            "proxy_target_err_le_3mm_rate": proxy_target_3mm_rate,
            "contact_stop_same_as_contact_rate": contact_stop_same_rate,
        },
        "maxdisp_visual_metrics": {
            "proxy_gap_to_live_side_face_m": _stats([m["proxy_gap_to_live_side_face_m"] for m in maxdisps]),
            "target_gap_to_live_side_face_m": _stats([m["target_gap_to_live_side_face_m"] for m in maxdisps]),
            "proxy_minus_live_cube_center_z_m": _stats([m["proxy_minus_live_cube_center_z_m"] for m in maxdisps]),
            "proxy_below_live_cube_top_m": _stats([m["proxy_below_live_cube_top_m"] for m in maxdisps]),
            "proxy_outside_live_face_rate": maxdisp_proxy_outside_face_rate,
        },
        "verdict": {
            "reaction_contact_no_posewrite_no_overshoot_pass": bool(reaction_gate.get("reaction_gate_pass")),
            "side_center_proxy_visual_verified": side_center_proxy_visual_verified,
            "top_contact_rejected_for_link5_proxy": top_contact_rejected_for_proxy,
            "grazing_or_outside_face_supported": grazing_or_outside_face,
            "early_freeze_supported": early_freeze,
            "weak_tap_visual_mechanism_supported": weak_tap_visual_mechanism_supported,
            "clean_tap_strength_visual_verified": clean_tap_strength_visual_verified,
            "clean_diffik_teacher_ready": False,
            "dataset_rl_roarm_unblocked": False,
            "next": (
                "stop_geometry_tuning_if_1mm_tap_is_enough; "
                "if_2_3mm_required_design_one_strength_preserving_contact_stop_or_through_variant_locally_first"
            ),
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
    metrics = artifact["contact_visual_metrics"]
    max_metrics = artifact["maxdisp_visual_metrics"]
    verdict = artifact["verdict"]
    args.out_summary.write_text(
        "\n".join(
            [
                "line1 artifact=cube10cm_link5corner_visual_proxy_contact_inspection_v1 "
                "local_trace_visual_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
                f"line2 contract_ok={contract_ok} trace_rows={len(trace_rows)} envs={len(envs)} "
                f"proxy_mode={trace_rows[0].get('tool_contact_proxy_mode')} "
                f"proxy_label={trace_rows[0].get('tool_proxy_label')} command_type={trace_rows[0].get('diffik_command_type')} "
                f"reaction_gate_pass={reaction_gate.get('reaction_gate_pass')}",
                "line3 contact_proxy_tracking "
                f"proxy_target_err_mean={metrics['proxy_target_err_m']['mean']:.9f} "
                f"proxy_target_z_err_mean={metrics['proxy_target_z_err_m']['mean']:.9f} "
                f"proxy_target_err_le_3mm_rate={proxy_target_3mm_rate:.9f} "
                f"proxy_target_err_le_5mm_rate={proxy_target_5mm_rate:.9f}",
                "line4 contact_height_semantics "
                f"proxy_minus_cube_center_z_mean={metrics['proxy_minus_live_cube_center_z_m']['mean']:.9f} "
                f"proxy_below_cube_top_mean={metrics['proxy_below_live_cube_top_m']['mean']:.9f} "
                f"proxy_side_center_z_near_5mm_rate={proxy_side_center_z_rate:.9f} "
                f"proxy_not_top_rate={proxy_not_top_rate:.9f} "
                f"tcp_minus_cube_center_z_mean={metrics['tcp_minus_live_cube_center_z_m']['mean']:.9f} "
                f"tcp_near_top_10mm_rate={tcp_near_top_rate:.9f}",
                "line5 contact_face_placement "
                f"proxy_gap_to_live_side_face_mean={metrics['proxy_gap_to_live_side_face_m']['mean']:.9f} "
                f"target_gap_to_live_side_face_mean={metrics['target_gap_to_live_side_face_m']['mean']:.9f} "
                f"proxy_outside_live_face_rate={proxy_outside_face_rate:.9f} "
                f"target_outside_live_face_rate={target_outside_face_rate:.9f} "
                f"proxy_lateral_from_center_mean={metrics['proxy_lateral_from_cube_center_m']['mean']:.9f} "
                f"target_lateral_from_center_mean={metrics['target_lateral_from_cube_center_m']['mean']:.9f}",
                "line6 maxdisp_and_freeze "
                f"maxdisp_proxy_gap_mean={max_metrics['proxy_gap_to_live_side_face_m']['mean']:.9f} "
                f"maxdisp_target_gap_mean={max_metrics['target_gap_to_live_side_face_m']['mean']:.9f} "
                f"maxdisp_proxy_outside_live_face_rate={maxdisp_proxy_outside_face_rate:.9f} "
                f"contact_stop_same_as_contact_rate={contact_stop_same_rate:.9f} "
                f"summary_max_disp_mean={_f(summary.get('max_disp_along_push_mean_m')):.9f} "
                f"summary_speed_mean={_f(summary.get('max_cube_speed_mean_mps')):.9f} "
                f"summary_low_motion_rate={_f(summary.get('low_motion_rate')):.9f}",
                "line7 verdict "
                f"primary_gate_pass={verdict['reaction_contact_no_posewrite_no_overshoot_pass']} "
                f"side_center_proxy_visual_verified={verdict['side_center_proxy_visual_verified']} "
                f"top_contact_rejected_for_link5_proxy={verdict['top_contact_rejected_for_link5_proxy']} "
                f"grazing_or_outside_face_supported={verdict['grazing_or_outside_face_supported']} "
                f"early_freeze_supported={verdict['early_freeze_supported']} "
                f"weak_tap_visual_mechanism_supported={verdict['weak_tap_visual_mechanism_supported']} "
                f"clean_tap_strength_visual_verified={verdict['clean_tap_strength_visual_verified']} "
                f"dataset_rl_roarm_unblocked={verdict['dataset_rl_roarm_unblocked']}",
                f"line8 outputs html={args.out_html} json={args.out_json}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
