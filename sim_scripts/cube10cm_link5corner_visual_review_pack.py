"""Render local PNG review sheets for the cube10cm link5-corner contact evidence.

The existing HTML/SVG inspection is useful, but this script creates raster PNGs
that can be opened directly for visual review inside the local workspace. It
reads the existing trace-derived visual JSON only. It does not run IsaacLab, use
GPU, generate a dataset, train, control a robot, SSH, or mutate source traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"

DEFAULT_VISUAL_JSON = LOG_DIR / "cube10cm_link5corner_visual_proxy_contact_inspection.json"
DEFAULT_OUT_TOPDOWN = LOG_DIR / "cube10cm_link5corner_visual_review_topdown.png"
DEFAULT_OUT_SIDE = LOG_DIR / "cube10cm_link5corner_visual_review_side.png"
DEFAULT_OUT_FOCUS = LOG_DIR / "cube10cm_link5corner_visual_review_focus_env0.png"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_link5corner_visual_review_pack_summary.out"

BG = (252, 249, 242)
PANEL = (255, 253, 248)
INK = (30, 31, 29)
MUTED = (110, 105, 96)
CUBE = (247, 213, 229)
PROXY = (242, 165, 65)
TARGET = (209, 73, 91)
TCP = (51, 102, 204)
MAXDISP = (127, 74, 203)
FACE = (20, 20, 20)
GRID = (210, 204, 192)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_SMALL = _font(12)
FONT_MED = _font(15)
FONT_BIG = _font(20)


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=INK, font=FONT_SMALL) -> None:
    draw.text(xy, text, fill=fill, font=font)


def _point(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: tuple[int, int, int], r: int = 5) -> None:
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0))


def _project(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = bounds
    left, top, right, bottom = rect
    px = left + (x - xmin) * (right - left) / max(xmax - xmin, 1.0e-9)
    py = bottom - (y - ymin) * (bottom - top) / max(ymax - ymin, 1.0e-9)
    return px, py


def _topdown_values(env: dict[str, Any], which: str = "contact") -> dict[str, float]:
    m = env[which]
    return {
        "proxy_lat": _f(m.get("proxy_lateral_from_cube_center_m")),
        "proxy_gap": _f(m.get("proxy_gap_to_live_side_face_m")),
        "target_lat": _f(m.get("target_lateral_from_cube_center_m")),
        "target_gap": _f(m.get("target_gap_to_live_side_face_m")),
        "tcp_lat": _f(m.get("tcp_lateral_from_cube_center_m")),
        "tcp_gap": _f(m.get("tcp_gap_to_live_side_face_m")),
    }


def _side_values(env: dict[str, Any], which: str = "contact") -> dict[str, float]:
    m = env[which]
    return {
        "proxy_gap": _f(m.get("proxy_gap_to_live_side_face_m")),
        "proxy_z": _f(m.get("proxy_minus_live_cube_center_z_m")),
        "target_gap": _f(m.get("target_gap_to_live_side_face_m")),
        "target_z": _f(m.get("target_minus_live_cube_center_z_m")),
        "tcp_gap": _f(m.get("tcp_gap_to_live_side_face_m")),
        "tcp_z": _f(m.get("tcp_minus_live_cube_center_z_m")),
    }


def _draw_panel_chrome(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=7, fill=PANEL, outline=(214, 207, 195), width=1)
    _draw_text(draw, (x0 + 10, y0 + 8), title, font=FONT_MED)
    _draw_text(draw, (x0 + 10, y0 + 28), subtitle, fill=MUTED, font=FONT_SMALL)
    return (x0 + 16, y0 + 50, x1 - 16, y1 - 28)


def _draw_topdown_panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], env: dict[str, Any]) -> None:
    env_id = int(env.get("env_id", -1))
    contact = _topdown_values(env, "contact")
    maxdisp = _topdown_values(env, "maxdisp")
    title = f"env {env_id} top-down"
    subtitle = f"gap {contact['proxy_gap']*1000:.1f}mm, target {contact['target_gap']*1000:.1f}mm"
    area = _draw_panel_chrome(draw, rect, title, subtitle)
    bounds = (-0.060, 0.060, -0.018, 0.055)

    # Cube footprint in local coordinates: lateral +/-50mm, push depth 0..100mm.
    cube0 = _project(-0.050, 0.000, bounds, area)
    cube1 = _project(0.050, 0.055, bounds, area)
    draw.rectangle((cube0[0], cube1[1], cube1[0], cube0[1]), fill=CUBE, outline=FACE, width=1)
    y_face = _project(0.0, 0.0, bounds, area)[1]
    draw.line((area[0], y_face, area[2], y_face), fill=FACE, width=2)
    _draw_text(draw, (area[0], int(y_face) + 3), "live face", fill=INK, font=FONT_SMALL)

    for value_mm in (-10, 0, 10, 20, 40):
        y = _project(0.0, value_mm / 1000.0, bounds, area)[1]
        draw.line((area[0], y, area[2], y), fill=GRID, width=1)

    proxy_xy = _project(contact["proxy_lat"], contact["proxy_gap"], bounds, area)
    target_xy = _project(contact["target_lat"], contact["target_gap"], bounds, area)
    tcp_xy = _project(contact["tcp_lat"], contact["tcp_gap"], bounds, area)
    max_xy = _project(maxdisp["proxy_lat"], maxdisp["proxy_gap"], bounds, area)
    draw.line((target_xy[0], target_xy[1], proxy_xy[0], proxy_xy[1]), fill=(150, 92, 72), width=1)
    _point(draw, proxy_xy, PROXY, 5)
    _point(draw, target_xy, TARGET, 4)
    _point(draw, tcp_xy, TCP, 4)
    _point(draw, max_xy, MAXDISP, 4)


def _draw_side_panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], env: dict[str, Any]) -> None:
    env_id = int(env.get("env_id", -1))
    contact = _side_values(env, "contact")
    maxdisp = _side_values(env, "maxdisp")
    title = f"env {env_id} side"
    subtitle = f"z {contact['proxy_z']*1000:.1f}mm, below top {(0.05-contact['proxy_z'])*1000:.1f}mm"
    area = _draw_panel_chrome(draw, rect, title, subtitle)
    bounds = (-0.045, 0.020, -0.012, 0.058)

    cube0 = _project(0.0, -0.050, bounds, area)
    cube1 = _project(0.020, 0.050, bounds, area)
    draw.rectangle((cube0[0], cube1[1], cube1[0], cube0[1]), fill=CUBE, outline=FACE, width=1)
    x_face = _project(0.0, 0.0, bounds, area)[0]
    y_center = _project(0.0, 0.0, bounds, area)[1]
    y_top = _project(0.0, 0.050, bounds, area)[1]
    draw.line((x_face, area[1], x_face, area[3]), fill=FACE, width=2)
    draw.line((area[0], y_center, area[2], y_center), fill=(130, 130, 130), width=1)
    draw.line((area[0], y_top, area[2], y_top), fill=(130, 130, 130), width=1)
    _draw_text(draw, (int(x_face) + 3, area[3] - 16), "face", fill=INK, font=FONT_SMALL)
    _draw_text(draw, (area[0], int(y_center) + 2), "center", fill=MUTED, font=FONT_SMALL)
    _draw_text(draw, (area[0], int(y_top) + 2), "top", fill=MUTED, font=FONT_SMALL)

    proxy_xy = _project(contact["proxy_gap"], contact["proxy_z"], bounds, area)
    target_xy = _project(contact["target_gap"], contact["target_z"], bounds, area)
    tcp_xy = _project(contact["tcp_gap"], contact["tcp_z"], bounds, area)
    max_xy = _project(maxdisp["proxy_gap"], maxdisp["proxy_z"], bounds, area)
    draw.line((target_xy[0], target_xy[1], proxy_xy[0], proxy_xy[1]), fill=(150, 92, 72), width=1)
    _point(draw, proxy_xy, PROXY, 5)
    _point(draw, target_xy, TARGET, 4)
    _point(draw, tcp_xy, TCP, 4)
    _point(draw, max_xy, MAXDISP, 4)


def _legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    _draw_text(draw, (x, y), "Legend:", font=FONT_MED)
    labels = [("proxy", PROXY), ("target", TARGET), ("TCP", TCP), ("proxy@maxdisp", MAXDISP)]
    cursor = x + 70
    for label, color in labels:
        _point(draw, (cursor, y + 8), color, 5)
        _draw_text(draw, (cursor + 10, y), label, font=FONT_SMALL)
        cursor += 128
    _draw_text(draw, (x, y + 24), "Negative face gap is outside/grazing on the approach side.", fill=MUTED)


def _draw_grid_image(
    envs: list[dict[str, Any]],
    out_path: Path,
    mode: str,
    title: str,
) -> None:
    cols = 4
    rows = 4
    panel_w, panel_h = 330, 250
    margin = 24
    header = 84
    width = cols * panel_w + (cols + 1) * margin
    height = header + rows * panel_h + (rows + 1) * margin
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    _draw_text(draw, (margin, 22), title, font=FONT_BIG)
    _legend(draw, margin, 52)
    for idx, env in enumerate(envs[:16]):
        row = idx // cols
        col = idx % cols
        x0 = margin + col * (panel_w + margin)
        y0 = header + margin + row * (panel_h + margin)
        rect = (x0, y0, x0 + panel_w, y0 + panel_h)
        if mode == "topdown":
            _draw_topdown_panel(draw, rect, env)
        elif mode == "side":
            _draw_side_panel(draw, rect, env)
        else:
            raise ValueError(mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _draw_focus_image(env: dict[str, Any], out_path: Path) -> None:
    image = Image.new("RGB", (1120, 620), BG)
    draw = ImageDraw.Draw(image)
    _draw_text(draw, (24, 22), "cube10cm link5-corner focus env0 visual review", font=FONT_BIG)
    _legend(draw, 24, 54)
    _draw_topdown_panel(draw, (24, 104, 548, 576), env)
    _draw_side_panel(draw, (572, 104, 1096, 576), env)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual_json", type=Path, default=DEFAULT_VISUAL_JSON)
    parser.add_argument("--out_topdown_png", type=Path, default=DEFAULT_OUT_TOPDOWN)
    parser.add_argument("--out_side_png", type=Path, default=DEFAULT_OUT_SIDE)
    parser.add_argument("--out_focus_png", type=Path, default=DEFAULT_OUT_FOCUS)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    artifact = _load_json(args.visual_json)
    envs = [env for env in artifact.get("envs", []) if isinstance(env, dict)]
    if len(envs) != 16:
        raise SystemExit(f"expected 16 envs, got {len(envs)}")

    _draw_grid_image(envs, args.out_topdown_png, "topdown", "cube10cm link5-corner all-env top-down review")
    _draw_grid_image(envs, args.out_side_png, "side", "cube10cm link5-corner all-env side-height review")
    focus = next((env for env in envs if int(env.get("env_id", -1)) == 0), envs[0])
    _draw_focus_image(focus, args.out_focus_png)

    contact_metrics = artifact.get("contact_visual_metrics", {})
    verdict = artifact.get("verdict", {})
    side_center_count = sum(
        1 for env in envs if abs(_f(env["contact"].get("proxy_minus_live_cube_center_z_m"))) <= 0.005
    )
    outside_count = sum(
        1 for env in envs if _f(env["contact"].get("proxy_gap_to_live_side_face_m")) < -0.001
    )
    stop_same_count = sum(1 for env in envs if bool(env.get("contact_stop_same_rollout_step")))
    lines = [
        "line1 artifact=cube10cm_link5corner_visual_review_pack_v1 "
        "local_png_review_only=YES gpu_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 rendered "
            f"topdown_png={args.out_topdown_png} side_png={args.out_side_png} focus_png={args.out_focus_png}"
        ),
        (
            "line3 visual_counts "
            f"envs={len(envs)} side_center_proxy_count={side_center_count} "
            f"proxy_outside_live_face_count={outside_count} contact_stop_same_count={stop_same_count}"
        ),
        (
            "line4 source_metrics "
            f"proxy_gap_mean={_f(contact_metrics.get('proxy_gap_to_live_side_face_m', {}).get('mean')):.9f} "
            f"target_gap_mean={_f(contact_metrics.get('target_gap_to_live_side_face_m', {}).get('mean')):.9f} "
            f"proxy_z_center_mean={_f(contact_metrics.get('proxy_minus_live_cube_center_z_m', {}).get('mean')):.9f} "
            f"proxy_below_top_mean={_f(contact_metrics.get('proxy_below_live_cube_top_m', {}).get('mean')):.9f}"
        ),
        (
            "line5 verdict "
            f"side_center_proxy_visual_verified={verdict.get('side_center_proxy_visual_verified')} "
            f"grazing_or_outside_face_supported={verdict.get('grazing_or_outside_face_supported')} "
            f"early_freeze_supported={verdict.get('early_freeze_supported')} "
            f"clean_tap_strength_visual_verified={verdict.get('clean_tap_strength_visual_verified')}"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
