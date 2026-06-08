"""Visual contact-frame audit for the 10cm tap positive-control failure.

This reads existing local JSON logs only. It does not launch IsaacLab, run GPU
physics, build datasets, train, control a robot, SSH, pull, or touch B200.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_POSITIVE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_sanity.json"
DEFAULT_FAILURE_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_failure_audit.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_visual_contact_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_positive_control_visual_contact_audit_summary.out"
DEFAULT_OUT_HTML = LOG_DIR / "cube10cm_tap_rl_positive_control_visual_contact_audit.html"
DEFAULT_OUT_SVG = LOG_DIR / "cube10cm_tap_rl_positive_control_visual_contact_audit.svg"
DEFAULT_OUT_PNG = LOG_DIR / "cube10cm_tap_rl_positive_control_visual_contact_audit.png"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(data.get(key, default))


def _tf(value: bool) -> str:
    return "True" if value else "False"


def _mm(value_m: float) -> float:
    return value_m * 1000.0


def _svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _draw_svg(result: dict[str, Any], out_svg: Path) -> None:
    cube_half = float(result["cube_half_along_m"])
    band = float(result["contact_band_m"])
    initial_along = float(result["initial_along_m"])
    final_along = float(result["final_along_m"])
    lateral = float(result["final_lateral_m"])
    vertical = float(result["final_vertical_offset_m"])

    x_min_m = -0.090
    x_max_m = 0.065
    width = 1120
    height = 610
    margin_l = 96
    margin_r = 70
    axis_y = 315

    def sx(x_m: float) -> float:
        return margin_l + (x_m - x_min_m) / (x_max_m - x_min_m) * (width - margin_l - margin_r)

    cube_l = sx(-cube_half)
    cube_r = sx(cube_half)
    band_l = sx(-cube_half - band)
    band_r = sx(-cube_half + band)
    init_x = sx(initial_along)
    final_x = sx(final_along)
    face_x = sx(-cube_half)

    ticks = [
        -0.080,
        -0.070,
        -0.060,
        -0.050,
        -0.040,
        -0.030,
        -0.020,
        -0.010,
        0.0,
        0.010,
        0.020,
        0.030,
        0.040,
        0.050,
    ]
    tick_parts = []
    for t in ticks:
        tx = sx(t)
        tick_parts.append(
            f'<line x1="{tx:.1f}" y1="{axis_y + 72}" x2="{tx:.1f}" y2="{axis_y + 84}" stroke="#64748b"/>'
        )
        tick_parts.append(
            f'<text x="{tx:.1f}" y="{axis_y + 104}" text-anchor="middle" class="tick">{_mm(t):.0f}</text>'
        )

    outside_arrow_y = axis_y - 120
    shortfall = float(result["final_shortfall_to_contact_band_m"])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px Arial, sans-serif; fill: #0f172a; }}
    .sub {{ font: 15px Arial, sans-serif; fill: #334155; }}
    .label {{ font: 14px Arial, sans-serif; fill: #0f172a; }}
    .tick {{ font: 12px Arial, sans-serif; fill: #475569; }}
    .small {{ font: 12px Arial, sans-serif; fill: #334155; }}
    .warn {{ font: 700 14px Arial, sans-serif; fill: #991b1b; }}
    .ok {{ font: 700 14px Arial, sans-serif; fill: #166534; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>
  <text x="56" y="52" class="title">10cm tap positive-control contact-frame audit</text>
  <text x="56" y="82" class="sub">Reconstructed from existing reset/final logs only; no per-step trace and no new GPU runtime.</text>

  <rect x="{cube_l:.1f}" y="{axis_y - 58}" width="{cube_r - cube_l:.1f}" height="116" rx="4" fill="#d9f99d" stroke="#3f6212" stroke-width="2"/>
  <text x="{(cube_l + cube_r) / 2:.1f}" y="{axis_y + 5}" text-anchor="middle" class="label">10cm cube</text>

  <rect x="{band_l:.1f}" y="{axis_y - 82}" width="{band_r - band_l:.1f}" height="164" fill="#bbf7d0" opacity="0.74" stroke="#16a34a" stroke-width="2"/>
  <text x="{(band_l + band_r) / 2:.1f}" y="{axis_y - 96}" text-anchor="middle" class="ok">contact band: face_gap [-10,+10] mm</text>

  <line x1="{face_x:.1f}" y1="{axis_y - 100}" x2="{face_x:.1f}" y2="{axis_y + 100}" stroke="#0f172a" stroke-width="2.5"/>
  <text x="{face_x:.1f}" y="{axis_y + 124}" text-anchor="middle" class="label">live approach face</text>

  <line x1="{margin_l}" y1="{axis_y + 78}" x2="{width - margin_r}" y2="{axis_y + 78}" stroke="#94a3b8"/>
  {''.join(tick_parts)}
  <text x="{width - 56}" y="{axis_y + 104}" text-anchor="end" class="tick">along position vs cube center (mm)</text>

  <line x1="{init_x:.1f}" y1="{axis_y - 58}" x2="{init_x:.1f}" y2="{axis_y + 58}" stroke="#2563eb" stroke-width="4"/>
  <circle cx="{init_x:.1f}" cy="{axis_y - 72}" r="8" fill="#2563eb"/>
  <text x="{init_x:.1f}" y="{axis_y - 92}" text-anchor="middle" class="label">initial TCP</text>
  <text x="{init_x:.1f}" y="{axis_y - 112}" text-anchor="middle" class="small">face_gap {_mm(float(result['initial_face_gap_m'])):.3f} mm</text>

  <line x1="{final_x:.1f}" y1="{axis_y - 58}" x2="{final_x:.1f}" y2="{axis_y + 58}" stroke="#dc2626" stroke-width="4"/>
  <circle cx="{final_x:.1f}" cy="{axis_y - 146}" r="8" fill="#dc2626"/>
  <text x="{final_x:.1f}" y="{axis_y - 168}" text-anchor="middle" class="warn">final TCP outside band</text>
  <text x="{final_x:.1f}" y="{axis_y - 188}" text-anchor="middle" class="small">face_gap {_mm(float(result['final_face_gap_m'])):.3f} mm</text>

  <line x1="{final_x:.1f}" y1="{outside_arrow_y}" x2="{band_l:.1f}" y2="{outside_arrow_y}" stroke="#991b1b" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="{(final_x + band_l) / 2:.1f}" y="{outside_arrow_y - 10}" text-anchor="middle" class="warn">shortfall {_mm(shortfall):.3f} mm</text>

  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 z" fill="#991b1b"/>
    </marker>
  </defs>

  <rect x="58" y="430" width="1004" height="126" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="84" y="462" class="label">Axis diagnosis</text>
  <text x="84" y="492" class="sub">along gap blocker = {_tf(bool(result['along_gap_blocker']))}; lateral_ok = {_tf(bool(result['lateral_ok']))} (final {_mm(lateral):.3f} mm); vertical_ok = {_tf(bool(result['vertical_ok']))} (final {_mm(vertical):.3f} mm)</text>
  <text x="84" y="522" class="sub">gap_delta = {_mm(float(result['gap_delta_m'])):.3f} mm, cube_disp_along = {_mm(float(result['cube_disp_along_m'])):.3f} mm, raw reaction without contact context = {_tf(bool(result['raw_reaction_without_context']))}</text>
  <text x="84" y="548" class="warn">Verdict: contact stayed 0 because the controller did not close the live face gap; this is not a lateral or height failure.</text>
</svg>
"""
    out_svg.write_text(svg, encoding="utf-8")


def _draw_png(result: dict[str, Any], out_png: Path) -> None:
    cube_half = float(result["cube_half_along_m"])
    band = float(result["contact_band_m"])
    initial_along = float(result["initial_along_m"])
    final_along = float(result["final_along_m"])
    lateral = float(result["final_lateral_m"])
    vertical = float(result["final_vertical_offset_m"])

    width, height = 1400, 760
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 15)
        font_warn = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except OSError:
        font_title = font_label = font_small = font_warn = font

    x_min_m, x_max_m = -0.090, 0.065
    margin_l, margin_r, axis_y = 120, 88, 390

    def sx(x_m: float) -> float:
        return margin_l + (x_m - x_min_m) / (x_max_m - x_min_m) * (width - margin_l - margin_r)

    cube_l, cube_r = sx(-cube_half), sx(cube_half)
    band_l, band_r = sx(-cube_half - band), sx(-cube_half + band)
    face_x = sx(-cube_half)
    init_x, final_x = sx(initial_along), sx(final_along)

    draw.text((70, 55), "10cm tap positive-control contact-frame audit", fill="#0f172a", font=font_title)
    draw.text(
        (70, 92),
        "Reconstructed from reset/final logs only; no per-step trace and no new GPU runtime.",
        fill="#334155",
        font=font_label,
    )

    draw.rectangle((band_l, axis_y - 105, band_r, axis_y + 105), fill="#bbf7d0", outline="#16a34a", width=3)
    draw.text((band_l - 30, axis_y - 137), "contact band: face_gap [-10,+10] mm", fill="#166534", font=font_warn)
    draw.rectangle((cube_l, axis_y - 72, cube_r, axis_y + 72), fill="#d9f99d", outline="#3f6212", width=3)
    draw.text(((cube_l + cube_r) / 2 - 48, axis_y - 10), "10cm cube", fill="#0f172a", font=font_label)
    draw.line((face_x, axis_y - 128, face_x, axis_y + 128), fill="#0f172a", width=4)
    draw.text((face_x - 70, axis_y + 146), "live approach face", fill="#0f172a", font=font_small)

    draw.line((margin_l, axis_y + 98, width - margin_r, axis_y + 98), fill="#94a3b8", width=2)
    for tick in [
        -0.080,
        -0.070,
        -0.060,
        -0.050,
        -0.040,
        -0.030,
        -0.020,
        -0.010,
        0.0,
        0.010,
        0.020,
        0.030,
        0.040,
        0.050,
    ]:
        tx = sx(tick)
        draw.line((tx, axis_y + 94, tx, axis_y + 112), fill="#64748b", width=2)
        draw.text((tx - 16, axis_y + 122), f"{_mm(tick):.0f}", fill="#475569", font=font_small)
    draw.text((width - 360, axis_y + 150), "along position vs cube center (mm)", fill="#475569", font=font_small)

    draw.line((init_x, axis_y - 72, init_x, axis_y + 72), fill="#2563eb", width=5)
    draw.ellipse((init_x - 9, axis_y - 100, init_x + 9, axis_y - 82), fill="#2563eb")
    draw.text((init_x - 57, axis_y - 132), "initial TCP", fill="#0f172a", font=font_label)
    draw.text(
        (init_x - 76, axis_y - 158),
        f"face_gap {_mm(float(result['initial_face_gap_m'])):.3f} mm",
        fill="#334155",
        font=font_small,
    )

    draw.line((final_x, axis_y - 72, final_x, axis_y + 72), fill="#dc2626", width=5)
    draw.ellipse((final_x - 9, axis_y - 182, final_x + 9, axis_y - 164), fill="#dc2626")
    draw.text((final_x - 96, axis_y - 221), "final TCP outside band", fill="#991b1b", font=font_warn)
    draw.text(
        (final_x - 78, axis_y - 247),
        f"face_gap {_mm(float(result['final_face_gap_m'])):.3f} mm",
        fill="#334155",
        font=font_small,
    )

    shortfall_y = axis_y - 146
    draw.line((final_x, shortfall_y, band_l, shortfall_y), fill="#991b1b", width=3)
    draw.polygon([(band_l, shortfall_y), (band_l - 12, shortfall_y - 7), (band_l - 12, shortfall_y + 7)], fill="#991b1b")
    draw.text(
        ((final_x + band_l) / 2 - 64, shortfall_y - 28),
        f"shortfall {_mm(float(result['final_shortfall_to_contact_band_m'])):.3f} mm",
        fill="#991b1b",
        font=font_warn,
    )

    panel = (70, 555, width - 70, 704)
    draw.rounded_rectangle(panel, radius=8, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((100, 582), "Axis diagnosis", fill="#0f172a", font=font_label)
    draw.text(
        (100, 617),
        (
            f"along_gap_blocker={_tf(bool(result['along_gap_blocker']))}; "
            f"lateral_ok={_tf(bool(result['lateral_ok']))} (final {_mm(lateral):.3f} mm); "
            f"vertical_ok={_tf(bool(result['vertical_ok']))} (final {_mm(vertical):.3f} mm)"
        ),
        fill="#334155",
        font=font_small,
    )
    draw.text(
        (100, 648),
        (
            f"gap_delta={_mm(float(result['gap_delta_m'])):.3f} mm, "
            f"cube_disp_along={_mm(float(result['cube_disp_along_m'])):.3f} mm, "
            f"raw reaction without contact context={_tf(bool(result['raw_reaction_without_context']))}"
        ),
        fill="#334155",
        font=font_small,
    )
    draw.text(
        (100, 677),
        "Verdict: contact stayed 0 because the controller did not close the live face gap; not lateral or height.",
        fill="#991b1b",
        font=font_warn,
    )
    image.save(out_png)


def _write_html(result: dict[str, Any], out_html: Path, svg_name: str, png_name: str) -> None:
    rows = []
    for key in [
        "initial_face_gap_m",
        "final_face_gap_m",
        "final_shortfall_to_contact_band_m",
        "initial_along_m",
        "final_along_m",
        "gap_delta_m",
        "cube_disp_along_m",
        "final_lateral_m",
        "final_vertical_offset_m",
        "contact_seen",
        "reaction_signal",
        "reaction_context",
        "tap_success",
    ]:
        value = result.get(key)
        rows.append(f"<tr><th>{_svg_escape(key)}</th><td>{_svg_escape(value)}</td></tr>")
    body = "\n".join(rows)
    out_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>10cm tap positive-control visual contact audit</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; color: #0f172a; background: #f8fafc; }}
    main {{ max-width: 1180px; }}
    img {{ max-width: 100%; border: 1px solid #cbd5e1; background: white; }}
    table {{ border-collapse: collapse; margin-top: 18px; background: white; }}
    th, td {{ padding: 8px 12px; border: 1px solid #cbd5e1; text-align: left; }}
    th {{ background: #e2e8f0; }}
    code {{ background: #e2e8f0; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
<main>
  <h1>10cm tap positive-control visual contact audit</h1>
  <p>This is a reconstructed initial/final contact-frame inspection from existing local logs only. It is not a per-step video trace and it did not launch a new GPU runtime.</p>
  <p>SVG: <code>{_svg_escape(svg_name)}</code>; PNG: <code>{_svg_escape(png_name)}</code></p>
  <img src="{_svg_escape(svg_name)}" alt="contact frame visual audit">
  <table>
    <tbody>
{body}
    </tbody>
  </table>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_json", type=Path, default=DEFAULT_POSITIVE_JSON)
    parser.add_argument("--failure_json", type=Path, default=DEFAULT_FAILURE_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--out_html", type=Path, default=DEFAULT_OUT_HTML)
    parser.add_argument("--out_svg", type=Path, default=DEFAULT_OUT_SVG)
    parser.add_argument("--out_png", type=Path, default=DEFAULT_OUT_PNG)
    args = parser.parse_args()

    positive = _load(args.positive_json)
    failure = _load(args.failure_json)
    reset = positive.get("reset_metrics", {})
    log = positive.get("last_log", {})

    cube_size = float(positive.get("cube_size_m", 0.100))
    cube_half = cube_size * 0.5
    contact_band = float(failure.get("contact_band_m", 0.010))
    lateral_limit = cube_half + 0.015
    vertical_limit = cube_half + 0.020

    initial_face_gap = _f(reset, "initial_face_gap_m")
    final_face_gap = _f(log, "cube_tap_contact_face_gap_m")
    initial_along = initial_face_gap - cube_half
    final_along = final_face_gap - cube_half
    gap_delta = final_face_gap - initial_face_gap
    shortfall = max(0.0, -contact_band - final_face_gap)

    lateral = _f(log, "cube_tap_contact_lateral_m")
    vertical = _f(log, "cube_tap_contact_vertical_offset_m")
    contact_seen = _f(log, "cube_tap_contact_seen_rate")
    reaction_signal = _f(log, "cube_tap_reaction_signal_now_rate")
    reaction_context = _f(log, "cube_tap_reaction_contact_context_rate")
    reaction_seen = _f(log, "cube_tap_reaction_seen_rate")
    tap_success = _f(log, "cube_tap_success_rate")
    overshoot = _f(log, "cube_tap_overshoot_seen_rate")
    cube_disp_along = _f(log, "cube_tap_disp_along_m")

    positive_runtime_valid = bool(failure.get("positive_runtime_valid")) and positive.get("device") == "cuda:0"
    lateral_ok = lateral <= lateral_limit
    vertical_ok = vertical <= vertical_limit
    along_gap_blocker = final_face_gap < -contact_band or final_face_gap > contact_band
    raw_reaction_without_context = reaction_signal > 0.0 and reaction_context == 0.0 and reaction_seen == 0.0
    wrapper_blocked_false_positive = raw_reaction_without_context and tap_success == 0.0
    contact_zero_explained = (
        positive_runtime_valid
        and contact_seen == 0.0
        and along_gap_blocker
        and lateral_ok
        and vertical_ok
        and wrapper_blocked_false_positive
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_positive_control_visual_contact_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_visual_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "visual_limitation": "reset_and_final_scalar_reconstruction_only_no_per_step_trace",
        "positive_runtime_valid": positive_runtime_valid,
        "positive_status": positive.get("status"),
        "cube_size_m": cube_size,
        "cube_half_along_m": cube_half,
        "contact_band_m": contact_band,
        "contact_band_along_min_m": -cube_half - contact_band,
        "contact_band_along_max_m": -cube_half + contact_band,
        "lateral_limit_m": lateral_limit,
        "vertical_limit_m": vertical_limit,
        "initial_face_gap_m": initial_face_gap,
        "final_face_gap_m": final_face_gap,
        "initial_along_m": initial_along,
        "final_along_m": final_along,
        "gap_delta_m": gap_delta,
        "final_shortfall_to_contact_band_m": shortfall,
        "cube_disp_along_m": cube_disp_along,
        "gap_delta_plus_cube_disp_abs_m": abs(gap_delta + cube_disp_along),
        "final_lateral_m": lateral,
        "final_vertical_offset_m": vertical,
        "lateral_ok": lateral_ok,
        "vertical_ok": vertical_ok,
        "along_gap_blocker": along_gap_blocker,
        "contact_seen": contact_seen,
        "reaction_signal": reaction_signal,
        "reaction_context": reaction_context,
        "reaction_seen": reaction_seen,
        "tap_success": tap_success,
        "overshoot_seen": overshoot,
        "raw_reaction_without_context": raw_reaction_without_context,
        "wrapper_blocked_false_positive": wrapper_blocked_false_positive,
        "contact_zero_explained": contact_zero_explained,
        "verdict": "controller_did_not_close_live_face_gap_not_lateral_or_height",
        "still_blocked": {
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "action_teacher_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next": {
            "allowed": "one_revised_external_closed_loop_positive_control_runtime_only_after_explicit_approval",
            "not_allowed": "ppo_large_dataset_action_teacher_roarm",
        },
        "outputs": {
            "html": str(args.out_html),
            "svg": str(args.out_svg),
            "png": str(args.out_png),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _draw_svg(result, args.out_svg)
    _draw_png(result, args.out_png)
    _write_html(result, args.out_html, args.out_svg.name, args.out_png.name)

    lines = [
        "line1 artifact=cube10cm_tap_rl_positive_control_visual_contact_audit_v1 "
        "local_visual_audit_only=YES gpu_runtime_launched_by_this_audit=NO "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 contract "
            f"positive_runtime_valid={positive_runtime_valid} status={positive.get('status')} "
            f"visual_limitation={result['visual_limitation']} cube_half_m={cube_half:.9f} "
            f"contact_band_m=[{-contact_band:.9f},{contact_band:.9f}]"
        ),
        (
            "line3 reconstructed_contact_frame "
            f"initial_along_m={initial_along:.9f} final_along_m={final_along:.9f} "
            f"initial_face_gap_m={initial_face_gap:.9f} final_face_gap_m={final_face_gap:.9f} "
            f"final_shortfall_to_band_m={shortfall:.9f}"
        ),
        (
            "line4 axis_diagnosis "
            f"lateral_ok={lateral_ok} vertical_ok={vertical_ok} "
            f"along_gap_blocker={along_gap_blocker} final_lateral_m={lateral:.9f} "
            f"final_vertical_offset_m={vertical:.9f} gap_delta_m={gap_delta:.9f} "
            f"cube_disp_along_m={cube_disp_along:.9f} "
            f"gap_delta_plus_cube_disp_abs_m={result['gap_delta_plus_cube_disp_abs_m']:.9f}"
        ),
        (
            "line5 reaction_guard "
            f"contact_seen={contact_seen} reaction_signal={reaction_signal} "
            f"reaction_context={reaction_context} tap_success={tap_success} "
            f"wrapper_blocked_false_positive={wrapper_blocked_false_positive}"
        ),
        (
            "line6 verdict "
            f"contact_zero_explained={contact_zero_explained} "
            "reason=controller_did_not_close_live_face_gap_not_lateral_or_height "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED action_teacher=BLOCKED roarm=BLOCKED"
        ),
        (
            "line7 outputs "
            f"html={args.out_html} svg={args.out_svg} png={args.out_png}"
        ),
        (
            "line8 next "
            "allowed=one_revised_external_closed_loop_positive_control_runtime_only_after_explicit_approval "
            "not_allowed=ppo_large_dataset_action_teacher_roarm"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if contact_zero_explained else 2


if __name__ == "__main__":
    raise SystemExit(main())
