"""Audit the direct-IK-apply positive-control result for the 10cm tap wrapper.

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
DIRECT_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_direct_ik_apply_sanity.json"
CAP040_JSON = LOG_DIR / "cube10cm_tap_rl_positive_control_external_closed_loop_cap040_sanity.json"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_result_audit.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_result_audit_summary.out"
DEFAULT_OUT_HTML = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_visual_contact_audit.html"
DEFAULT_OUT_SVG = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_visual_contact_audit.svg"
DEFAULT_OUT_PNG = LOG_DIR / "cube10cm_tap_rl_direct_ik_apply_visual_contact_audit.png"
PROFESSOR_PHYSICAL_DISP_EVIDENCE_M = 0.0005
PROFESSOR_PHYSICAL_SPEED_EVIDENCE_MPS = 0.005


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(data: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not data:
        return default
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _trace(run: dict[str, Any], key: str, field: str, default: float = 0.0) -> float:
    return _f(run.get("log_trace_stats", {}).get(key, {}), field, default)


def _mm(value_m: float) -> float:
    return value_m * 1000.0


def _tf(value: bool) -> str:
    return "True" if value else "False"


def _svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _draw_svg(result: dict[str, Any], out_svg: Path) -> None:
    cube_half = float(result["cube_half_along_m"])
    band = float(result["contact_band_m"])
    initial_along = float(result["initial_along_m"])
    best_along = float(result["best_along_m"])
    final_along = float(result["final_along_m"])

    x_min_m = -0.090
    x_max_m = 0.065
    width = 1180
    height = 650
    margin_l = 100
    margin_r = 72
    axis_y = 330

    def sx(x_m: float) -> float:
        return margin_l + (x_m - x_min_m) / (x_max_m - x_min_m) * (width - margin_l - margin_r)

    cube_l = sx(-cube_half)
    cube_r = sx(cube_half)
    band_l = sx(-cube_half - band)
    band_r = sx(-cube_half + band)
    face_x = sx(-cube_half)
    init_x = sx(initial_along)
    best_x = sx(best_along)
    final_x = sx(final_along)
    shortfall = float(result["best_shortfall_to_contact_band_m"])

    tick_parts = []
    for tick in [-0.080, -0.070, -0.060, -0.050, -0.040, -0.030, -0.020, -0.010, 0.0, 0.010, 0.020, 0.030, 0.040, 0.050]:
        tx = sx(tick)
        tick_parts.append(f'<line x1="{tx:.1f}" y1="{axis_y + 80}" x2="{tx:.1f}" y2="{axis_y + 92}" stroke="#64748b"/>')
        tick_parts.append(f'<text x="{tx:.1f}" y="{axis_y + 112}" text-anchor="middle" class="tick">{_mm(tick):.0f}</text>')

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
  <text x="58" y="52" class="title">10cm tap direct-IK-apply contact-frame audit</text>
  <text x="58" y="82" class="sub">Reconstructed from direct runtime scalar + trace aggregates; no new GPU runtime in this audit.</text>

  <rect x="{band_l:.1f}" y="{axis_y - 86}" width="{band_r - band_l:.1f}" height="172" fill="#bbf7d0" opacity="0.72" stroke="#16a34a" stroke-width="2"/>
  <text x="{(band_l + band_r) / 2:.1f}" y="{axis_y - 102}" text-anchor="middle" class="ok">contact band: face_gap [-10,+10] mm</text>
  <rect x="{cube_l:.1f}" y="{axis_y - 58}" width="{cube_r - cube_l:.1f}" height="116" rx="4" fill="#d9f99d" stroke="#3f6212" stroke-width="2"/>
  <text x="{(cube_l + cube_r) / 2:.1f}" y="{axis_y + 5}" text-anchor="middle" class="label">10cm cube</text>
  <line x1="{face_x:.1f}" y1="{axis_y - 104}" x2="{face_x:.1f}" y2="{axis_y + 104}" stroke="#0f172a" stroke-width="2.5"/>
  <text x="{face_x:.1f}" y="{axis_y + 132}" text-anchor="middle" class="label">live approach face</text>

  <line x1="{margin_l}" y1="{axis_y + 86}" x2="{width - margin_r}" y2="{axis_y + 86}" stroke="#94a3b8"/>
  {''.join(tick_parts)}
  <text x="{width - 58}" y="{axis_y + 112}" text-anchor="end" class="tick">along position vs cube center (mm)</text>

  <line x1="{init_x:.1f}" y1="{axis_y - 58}" x2="{init_x:.1f}" y2="{axis_y + 58}" stroke="#2563eb" stroke-width="4"/>
  <text x="{init_x:.1f}" y="{axis_y - 118}" text-anchor="middle" class="small">initial gap {_mm(float(result['initial_face_gap_m'])):.3f} mm</text>
  <circle cx="{init_x:.1f}" cy="{axis_y - 82}" r="7" fill="#2563eb"/>

  <line x1="{best_x:.1f}" y1="{axis_y - 72}" x2="{best_x:.1f}" y2="{axis_y + 72}" stroke="#7c3aed" stroke-width="4"/>
  <text x="{best_x:.1f}" y="{axis_y - 170}" text-anchor="middle" class="warn">best still outside</text>
  <text x="{best_x:.1f}" y="{axis_y - 148}" text-anchor="middle" class="small">face_gap {_mm(float(result['best_face_gap_m'])):.3f} mm</text>
  <circle cx="{best_x:.1f}" cy="{axis_y - 104}" r="7" fill="#7c3aed"/>

  <line x1="{final_x:.1f}" y1="{axis_y - 58}" x2="{final_x:.1f}" y2="{axis_y + 58}" stroke="#dc2626" stroke-width="4"/>
  <text x="{final_x:.1f}" y="{axis_y - 216}" text-anchor="middle" class="warn">final outside band</text>
  <text x="{final_x:.1f}" y="{axis_y - 194}" text-anchor="middle" class="small">face_gap {_mm(float(result['final_face_gap_m'])):.3f} mm</text>
  <circle cx="{final_x:.1f}" cy="{axis_y - 128}" r="7" fill="#dc2626"/>

  <line x1="{best_x:.1f}" y1="{axis_y - 134}" x2="{band_l:.1f}" y2="{axis_y - 134}" stroke="#991b1b" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="{(best_x + band_l) / 2:.1f}" y="{axis_y - 144}" text-anchor="middle" class="warn">best shortfall {_mm(shortfall):.3f} mm</text>

  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 z" fill="#991b1b"/>
    </marker>
  </defs>

  <rect x="60" y="462" width="1058" height="130" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="86" y="496" class="label">Diagnosis</text>
  <text x="86" y="526" class="sub">direct apply active = {_tf(bool(result['direct_apply_active']))}; RL action path bypassed = {_tf(bool(result['rl_action_path_bypassed']))}; contact = {result['contact_seen']}</text>
  <text x="86" y="554" class="sub">lateral_ok = {_tf(bool(result['lateral_ok']))} ({_mm(float(result['final_lateral_m'])):.3f} mm); vertical_ok = {_tf(bool(result['vertical_ok']))} ({_mm(float(result['final_vertical_offset_m'])):.3f} mm); face_gap_near_band = {_tf(bool(result['face_gap_near_band']))}</text>
  <text x="86" y="582" class="warn">Verdict: professor physical reaction evidence is separate from RL contact-gated readiness.</text>
</svg>
"""
    out_svg.write_text(svg, encoding="utf-8")


def _draw_png(result: dict[str, Any], out_png: Path) -> None:
    cube_half = float(result["cube_half_along_m"])
    band = float(result["contact_band_m"])
    initial_along = float(result["initial_along_m"])
    best_along = float(result["best_along_m"])
    final_along = float(result["final_along_m"])
    width, height = 1500, 800
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    try:
        title = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        label = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 15)
        warn = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except OSError:
        title = label = small = warn = font

    x_min_m, x_max_m = -0.090, 0.065
    margin_l, margin_r, axis_y = 128, 90, 405

    def sx(x_m: float) -> float:
        return margin_l + (x_m - x_min_m) / (x_max_m - x_min_m) * (width - margin_l - margin_r)

    cube_l, cube_r = sx(-cube_half), sx(cube_half)
    band_l, band_r = sx(-cube_half - band), sx(-cube_half + band)
    face_x = sx(-cube_half)
    init_x, best_x, final_x = sx(initial_along), sx(best_along), sx(final_along)

    draw.text((72, 56), "10cm tap direct-IK-apply contact-frame audit", fill="#0f172a", font=title)
    draw.text((72, 94), "Trace-aggregate reconstruction; this audit does not launch another GPU run.", fill="#334155", font=label)
    draw.rectangle((band_l, axis_y - 110, band_r, axis_y + 110), fill="#bbf7d0", outline="#16a34a", width=3)
    draw.rectangle((cube_l, axis_y - 74, cube_r, axis_y + 74), fill="#d9f99d", outline="#3f6212", width=3)
    draw.line((face_x, axis_y - 134, face_x, axis_y + 134), fill="#0f172a", width=4)
    draw.text((band_l - 24, axis_y - 148), "contact band: face_gap [-10,+10] mm", fill="#166534", font=warn)
    draw.text(((cube_l + cube_r) / 2 - 50, axis_y - 11), "10cm cube", fill="#0f172a", font=label)

    draw.line((margin_l, axis_y + 100, width - margin_r, axis_y + 100), fill="#94a3b8", width=2)
    for tick in [-0.080, -0.070, -0.060, -0.050, -0.040, -0.030, -0.020, -0.010, 0.0, 0.010, 0.020, 0.030, 0.040, 0.050]:
        tx = sx(tick)
        draw.line((tx, axis_y + 96, tx, axis_y + 114), fill="#64748b", width=2)
        draw.text((tx - 17, axis_y + 126), f"{_mm(tick):.0f}", fill="#475569", font=small)

    draw.line((init_x, axis_y - 76, init_x, axis_y + 76), fill="#2563eb", width=5)
    draw.ellipse((init_x - 9, axis_y - 106, init_x + 9, axis_y - 88), fill="#2563eb")
    draw.text((init_x - 82, axis_y - 158), f"initial gap {_mm(float(result['initial_face_gap_m'])):.3f} mm", fill="#334155", font=small)

    draw.line((best_x, axis_y - 86, best_x, axis_y + 86), fill="#7c3aed", width=5)
    draw.ellipse((best_x - 9, axis_y - 132, best_x + 9, axis_y - 114), fill="#7c3aed")
    draw.text((best_x - 72, axis_y - 205), "best still outside", fill="#991b1b", font=warn)
    draw.text((best_x - 86, axis_y - 178), f"face_gap {_mm(float(result['best_face_gap_m'])):.3f} mm", fill="#334155", font=small)

    draw.line((final_x, axis_y - 76, final_x, axis_y + 76), fill="#dc2626", width=5)
    draw.ellipse((final_x - 9, axis_y - 152, final_x + 9, axis_y - 134), fill="#dc2626")
    draw.text((final_x - 80, axis_y - 248), "final outside", fill="#991b1b", font=warn)
    draw.text((final_x - 86, axis_y - 222), f"face_gap {_mm(float(result['final_face_gap_m'])):.3f} mm", fill="#334155", font=small)

    draw.rectangle((72, 590, 1428, 724), fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((102, 620), "Diagnosis", fill="#0f172a", font=label)
    draw.text((102, 652), f"direct apply active: {_tf(bool(result['direct_apply_active']))}; RL action path bypassed: {_tf(bool(result['rl_action_path_bypassed']))}; contact: {result['contact_seen']}", fill="#334155", font=small)
    draw.text((102, 680), f"best shortfall: {_mm(float(result['best_shortfall_to_contact_band_m'])):.3f} mm; lateral/vertical OK: {_tf(bool(result['lateral_ok']))}/{_tf(bool(result['vertical_ok']))}", fill="#334155", font=small)
    draw.text((102, 708), "Verdict: physical reaction evidence is separate from RL contact-gated readiness.", fill="#991b1b", font=warn)
    image.save(out_png)


def _write_html(result: dict[str, Any], out_html: Path, svg_name: str, png_name: str) -> None:
    body = json.dumps(result, indent=2, sort_keys=True)
    out_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>10cm tap direct-IK visual contact audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #0f172a; background: #f8fafc; }}
    img {{ max-width: 100%; border: 1px solid #cbd5e1; background: white; }}
    pre {{ padding: 16px; background: white; border: 1px solid #cbd5e1; overflow: auto; }}
  </style>
</head>
<body>
  <h1>10cm tap direct-IK visual contact audit</h1>
  <p>Local posthoc artifact only. No GPU runtime, dataset, training, robot control, SSH, B200, or Track A is launched by this audit.</p>
  <p><a href="{_svg_escape(svg_name)}">SVG</a> | <a href="{_svg_escape(png_name)}">PNG</a></p>
  <img src="{_svg_escape(svg_name)}" alt="direct IK contact-frame audit"/>
  <h2>Result JSON</h2>
  <pre>{_svg_escape(body)}</pre>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_json", type=Path, default=DIRECT_JSON)
    parser.add_argument("--cap040_json", type=Path, default=CAP040_JSON)
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--out_html", type=Path, default=DEFAULT_OUT_HTML)
    parser.add_argument("--out_svg", type=Path, default=DEFAULT_OUT_SVG)
    parser.add_argument("--out_png", type=Path, default=DEFAULT_OUT_PNG)
    args = parser.parse_args()

    direct = _load(args.direct_json)
    cap040 = _load(args.cap040_json)
    reset = direct.get("reset_metrics", {})
    controller = direct.get("controller_metrics", {})
    log = direct.get("last_log", {})

    cube_size = float(direct.get("cube_size_m", 0.100))
    cube_half = cube_size * 0.5
    contact_band = 0.010
    lateral_limit = cube_half + 0.015
    vertical_limit = cube_half + 0.020

    initial_gap = _f(reset, "initial_face_gap_m")
    best_gap = _trace(direct, "cube_tap_contact_face_gap_m", "max", _f(log, "cube_tap_contact_face_gap_m"))
    worst_gap = _trace(direct, "cube_tap_contact_face_gap_m", "min", _f(log, "cube_tap_contact_face_gap_m"))
    final_gap = _trace(direct, "cube_tap_contact_face_gap_m", "final", _f(log, "cube_tap_contact_face_gap_m"))
    best_shortfall = _trace(direct, "cube_tap_contact_band_shortfall_m", "min", max(0.0, -contact_band - best_gap))
    final_shortfall = _trace(direct, "cube_tap_contact_band_shortfall_m", "final", max(0.0, -contact_band - final_gap))
    cap040_best_shortfall = _trace(cap040, "cube_tap_contact_band_shortfall_m", "min")
    cap040_best_gap = _trace(cap040, "cube_tap_contact_face_gap_m", "max")

    contact_seen = _f(log, "cube_tap_contact_seen_rate")
    reaction_signal = _f(log, "cube_tap_reaction_signal_now_rate")
    reaction_context = _f(log, "cube_tap_reaction_contact_context_rate")
    reaction_seen = _f(log, "cube_tap_reaction_seen_rate")
    tap_success = _f(log, "cube_tap_success_rate")
    overshoot = _f(log, "cube_tap_overshoot_seen_rate")
    lateral = _f(log, "cube_tap_contact_lateral_m")
    vertical = _f(log, "cube_tap_contact_vertical_offset_m")

    action_abs_max_trace = _trace(direct, "cube_push_action_abs_max", "max")
    joint_delta_abs_max_trace = _trace(direct, "cube_push_joint_delta_abs_max", "max")
    target_lead_abs_max_trace = _trace(direct, "cube_push_target_lead_abs_max", "max")
    cap_rate_trace = _trace(direct, "cube_push_joint_delta_cap_rate", "max")
    lead_limit_trace = _trace(direct, "cube_push_target_lead_limit_rate", "max")

    direct_runtime_valid = (
        direct.get("gpu_runtime") == "YES_LOCAL_TINY_ISAACLAB_POSITIVE_CONTROL"
        and direct.get("device") == "cuda:0"
        and direct.get("controller_mode") == "external_closed_loop_direct_apply"
        and direct.get("direct_ik_joint_target_apply") is True
        and direct.get("dataset_generation") is False
        and direct.get("training") is False
        and direct.get("robot_control") is False
        and direct.get("ssh") is False
        and direct.get("b200") is False
        and direct.get("track_a") is False
    )
    direct_apply_active = direct.get("direct_ik_joint_target_apply") is True and _f(controller, "closed_loop_ik_ok_rate") == 1.0
    rl_action_path_bypassed = action_abs_max_trace == 0.0 and cap_rate_trace == 0.0 and lead_limit_trace == 0.0
    lateral_ok = lateral <= lateral_limit
    vertical_ok = vertical <= vertical_limit
    face_gap_moved_toward_band = best_gap > initial_gap
    face_gap_near_band = best_shortfall <= 0.002
    along_gap_blocker = best_shortfall > 0.0 and final_shortfall > 0.0
    direct_ik_apply_pass = contact_seen > 0.0 and reaction_context > 0.0 and reaction_seen > 0.0 and tap_success > 0.0 and overshoot == 0.0
    max_disp_along = _f(log, "cube_tap_max_disp_along_m")
    max_speed = _f(log, "cube_tap_max_speed_mps")
    professor_physical_reaction_evidence = (
        (max_disp_along >= PROFESSOR_PHYSICAL_DISP_EVIDENCE_M or max_speed >= PROFESSOR_PHYSICAL_SPEED_EVIDENCE_MPS)
        and overshoot == 0.0
    )
    wrapper_only_explanation_falsified = (
        direct_runtime_valid and direct_apply_active and rl_action_path_bypassed and not direct_ik_apply_pass and along_gap_blocker
    )

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_direct_ik_apply_result_audit_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_posthoc_audit_only": True,
        "gpu_runtime_launched_by_this_audit": False,
        "dataset_generation": False,
        "training": False,
        "robot_control": False,
        "ssh": False,
        "b200": False,
        "track_a": False,
        "direct_runtime_valid": direct_runtime_valid,
        "status": direct.get("status"),
        "device": direct.get("device"),
        "controller_mode": direct.get("controller_mode"),
        "direct_apply_active": direct_apply_active,
        "rl_action_path_bypassed": rl_action_path_bypassed,
        "closed_loop_ik_ok_rate": _f(controller, "closed_loop_ik_ok_rate"),
        "closed_loop_ik_err_mm_mean": _f(controller, "closed_loop_ik_err_mm_mean"),
        "initial_face_gap_m": initial_gap,
        "best_face_gap_m": best_gap,
        "worst_face_gap_m": worst_gap,
        "final_face_gap_m": final_gap,
        "best_improvement_from_initial_m": best_gap - initial_gap,
        "best_shortfall_to_contact_band_m": best_shortfall,
        "final_shortfall_to_contact_band_m": final_shortfall,
        "cap040_best_shortfall_m": cap040_best_shortfall,
        "shortfall_delta_vs_cap040_m": best_shortfall - cap040_best_shortfall,
        "cap040_best_face_gap_m": cap040_best_gap,
        "face_gap_best_delta_vs_cap040_m": best_gap - cap040_best_gap,
        "cube_half_along_m": cube_half,
        "contact_band_m": contact_band,
        "initial_along_m": initial_gap - cube_half,
        "best_along_m": best_gap - cube_half,
        "final_along_m": final_gap - cube_half,
        "final_lateral_m": lateral,
        "final_vertical_offset_m": vertical,
        "lateral_ok": lateral_ok,
        "vertical_ok": vertical_ok,
        "along_gap_blocker": along_gap_blocker,
        "face_gap_moved_toward_band": face_gap_moved_toward_band,
        "face_gap_near_band": face_gap_near_band,
        "contact_seen": contact_seen,
        "reaction_signal": reaction_signal,
        "reaction_context": reaction_context,
        "reaction_seen": reaction_seen,
        "tap_success": tap_success,
        "overshoot": overshoot,
        "max_disp_along_m": max_disp_along,
        "max_speed_mps": max_speed,
        "professor_physical_reaction_evidence": "PASS" if professor_physical_reaction_evidence else "FAIL",
        "professor_physical_reaction_evidence_only": professor_physical_reaction_evidence
        and not direct_ik_apply_pass,
        "professor_physical_disp_evidence_threshold_m": PROFESSOR_PHYSICAL_DISP_EVIDENCE_M,
        "professor_physical_speed_evidence_threshold_mps": PROFESSOR_PHYSICAL_SPEED_EVIDENCE_MPS,
        "tcp_dist_min_m": _trace(direct, "cube_push_tcp_cube_dist_m", "min", _f(log, "cube_push_tcp_cube_dist_m")),
        "action_abs_max_trace": action_abs_max_trace,
        "joint_delta_abs_max_trace": joint_delta_abs_max_trace,
        "target_lead_abs_max_trace": target_lead_abs_max_trace,
        "joint_delta_cap_rate_trace": cap_rate_trace,
        "target_lead_limit_rate_trace": lead_limit_trace,
        "direct_ik_apply_pass": direct_ik_apply_pass,
        "wrapper_only_explanation_falsified": wrapper_only_explanation_falsified,
        "verdict": (
            "PROFESSOR_PHYSICAL_REACTION_PASS_RL_CONTACT_GATED_FAIL"
            if professor_physical_reaction_evidence and wrapper_only_explanation_falsified
            else (
                "DIRECT_IK_APPLY_FAILS_CONTACT_BAND_WRAPPER_ONLY_EXPLANATION_FALSIFIED"
                if wrapper_only_explanation_falsified
                else "REQUIRES_MORE_LOCAL_REVIEW"
            )
        ),
        "still_blocked": {
            "diffik_action_dataset": "BLOCKED",
            "tiny_action_dataset_dry_run": "BLOCKED",
            "ppo_rl_training": "BLOCKED",
            "large_dataset": "BLOCKED",
            "roarm": "BLOCKED",
        },
        "next": {
            "allowed_local_only": "review_target_geometry_kinematic_frame_and_reach_before_any_new_runtime",
            "not_allowed": "lead_cap_sweep_dataset_rl_roarm",
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
        "line1 artifact=cube10cm_tap_rl_direct_ik_apply_result_audit_v1 "
        "local_posthoc_visual_audit_only=YES gpu_runtime_launched_by_this_audit=NO "
        "dataset_generation=NO training=NO robot_control=NO ssh=NO b200=NO track_a=NO",
        (
            "line2 runtime_contract "
            f"direct_runtime_valid={direct_runtime_valid} status={direct.get('status')} "
            f"device={direct.get('device')} controller_mode={direct.get('controller_mode')} "
            f"direct_apply_active={direct_apply_active} rl_action_path_bypassed={rl_action_path_bypassed}"
        ),
        (
            "line3 ik_and_action_path "
            f"closed_loop_ik_ok_rate={_f(controller, 'closed_loop_ik_ok_rate'):.9f} "
            f"closed_loop_ik_err_mm_mean={_f(controller, 'closed_loop_ik_err_mm_mean'):.9f} "
            f"action_abs_max_trace={action_abs_max_trace:.9f} "
            f"joint_delta_abs_max_trace={joint_delta_abs_max_trace:.9f} "
            f"target_lead_abs_max_trace={target_lead_abs_max_trace:.9f} "
            f"cap_rate={cap_rate_trace:.9f} lead_limit_rate={lead_limit_trace:.9f}"
        ),
        (
            "line4 contact_outcome "
            f"contact_seen={contact_seen} reaction_signal={reaction_signal} "
            f"reaction_context={reaction_context} reaction_seen={reaction_seen} "
            f"tap_success={tap_success} overshoot={overshoot} "
            f"max_disp_along_m={max_disp_along:.9f} "
            f"max_speed_mps={max_speed:.9f} "
            f"professor_physical_reaction_evidence={result['professor_physical_reaction_evidence']}"
        ),
        (
            "line5 face_gap_trace "
            f"initial_face_gap_m={initial_gap:.9f} best_face_gap_m={best_gap:.9f} "
            f"worst_face_gap_m={worst_gap:.9f} final_face_gap_m={final_gap:.9f} "
            f"best_improvement_m={best_gap - initial_gap:.9f} "
            f"best_shortfall_m={best_shortfall:.9f} final_shortfall_m={final_shortfall:.9f} "
            f"face_gap_near_band={face_gap_near_band}"
        ),
        (
            "line6 cap040_comparison "
            f"cap040_best_shortfall_m={cap040_best_shortfall:.9f} "
            f"direct_best_shortfall_m={best_shortfall:.9f} "
            f"shortfall_delta_vs_cap040_m={best_shortfall - cap040_best_shortfall:.9f} "
            f"cap040_best_face_gap_m={cap040_best_gap:.9f} "
            f"direct_best_face_gap_m={best_gap:.9f} "
            f"face_gap_best_delta_vs_cap040_m={best_gap - cap040_best_gap:.9f}"
        ),
        (
            "line7 visual_axis "
            f"lateral_ok={lateral_ok} vertical_ok={vertical_ok} along_gap_blocker={along_gap_blocker} "
            f"final_lateral_m={lateral:.9f} final_vertical_offset_m={vertical:.9f} "
            f"tcp_dist_min_m={result['tcp_dist_min_m']:.9f}"
        ),
        (
            "line8 verdict "
            f"{result['verdict']} professor_physical_reaction_evidence={result['professor_physical_reaction_evidence']} "
            f"direct_ik_apply_pass={direct_ik_apply_pass} "
            f"wrapper_only_explanation_falsified={wrapper_only_explanation_falsified} "
            "diffik_action_dataset=BLOCKED tiny_action_dataset_dry_run=BLOCKED "
            "ppo_rl_training=BLOCKED large_dataset=BLOCKED roarm=BLOCKED"
        ),
        (
            "line9 outputs "
            f"html={args.out_html} svg={args.out_svg} png={args.out_png}"
        ),
        (
            "line10 next "
            "allowed_local_only=review_target_geometry_kinematic_frame_and_reach_before_any_new_runtime "
            "not_allowed=lead_cap_sweep_dataset_rl_roarm"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if wrapper_only_explanation_falsified else 2


if __name__ == "__main__":
    raise SystemExit(main())
