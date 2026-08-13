#!/usr/bin/env python3
"""Create a CPU-only lab-meeting video from the frozen P13 native trace."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/t3u_side_meeting1_mpl")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[4]
CASE = ROOT / "claudedocs/runtime_logs/grasp_track/g0b_d420"
PREFIX = "t3u_side_meeting1_trace_video"
TRACE_PATH = CASE / "t3u_side_preflight13_trace.npz"
RESULTS_PATH = CASE / "t3u_side_preflight13_results.json"
PLAN_PATH = CASE / "t3u_side_preflight13_plan.json"
PREREG_PATH = CASE / f"{PREFIX}_prereg.md"
SOURCE_PATH = CASE / f"{PREFIX}_render.py"
FRAME_DIR = CASE / f"{PREFIX}_frames"
MP4_PATH = CASE / f"{PREFIX}.mp4"
SHEET_PATH = CASE / f"{PREFIX}_contact_sheet.png"
MANIFEST_PATH = CASE / f"{PREFIX}_manifest.json"

INPUT_SHA256 = {
    TRACE_PATH.name: "ee67d3516a1c7871e5f48d455b420c3f5985ae889bceb097536904548e8134ee",
    RESULTS_PATH.name: "8324ed7a9682ccb297985dd733c9e91c480bed9ce65bb02672d5b40226eea6d5",
    PLAN_PATH.name: "d7fcfb47c26c38f4817ce7630671d915e0d77a4b3bcc1f2d7df40fd816f94f66",
}
PHASE_NAMES = ("settle", "approach", "stage", "descend", "close", "hold", "lift")
PHASE_STEPS = (120, 400, 400, 400, 400, 120, 500)
BODY_NAMES = ("link1", "link2", "link3", "link4", "link5", "gripper_link")
FRAME_INDICES = np.arange(0, 2340, 10, dtype=np.int64)
FPS = 20
WIDTH, HEIGHT = 1280, 720
DISCLAIMER = "POSTHOC TRACE VISUALIZATION — NOT RTX, NOT SCIENTIFIC AUTHORITY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def draw_cylinder(ax, center: np.ndarray, quat: np.ndarray) -> None:
    radius, half_height = 0.0145, 0.025
    theta = np.linspace(0, 2 * np.pi, 32)
    z = np.array([-half_height, half_height])
    tt, zz = np.meshgrid(theta, z)
    local = np.stack(
        [radius * np.cos(tt), radius * np.sin(tt), np.broadcast_to(zz, tt.shape)],
        axis=-1,
    )
    world = local @ quat_wxyz_to_matrix(quat).T + center
    ax.plot_surface(
        world[:, :, 0], world[:, :, 1], world[:, :, 2],
        color="#f59e0b", alpha=0.75, linewidth=0, shade=True,
    )
    for local_z in (-half_height, half_height):
        ring_local = np.column_stack(
            [radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, local_z)]
        )
        ring = ring_local @ quat_wxyz_to_matrix(quat).T + center
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#fde68a", lw=1.0)


def phase_spans() -> list[tuple[int, int, str]]:
    rows = []
    start = 0
    for name, count in zip(PHASE_NAMES, PHASE_STEPS):
        rows.append((start, start + count, name))
        start += count
    return rows


def style_time_axis(ax, title: str) -> None:
    colors = ("#334155", "#164e63", "#365314", "#78350f", "#7f1d1d", "#581c87", "#1e3a8a")
    for (start, end, _), color in zip(phase_spans(), colors):
        ax.axvspan(start, end, color=color, alpha=0.12, lw=0)
    ax.set_xlim(0, 2339)
    ax.set_title(title, loc="left", fontsize=10, color="#e2e8f0", pad=3)
    ax.grid(True, alpha=0.14, lw=0.6)
    ax.tick_params(labelsize=7, colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.set_facecolor("#0f172a")


def main() -> None:
    for path, expected in ((TRACE_PATH, INPUT_SHA256[TRACE_PATH.name]),
                           (RESULTS_PATH, INPUT_SHA256[RESULTS_PATH.name]),
                           (PLAN_PATH, INPUT_SHA256[PLAN_PATH.name])):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"INPUT_HASH_DRIFT {path.name} expected={expected} actual={actual}")
    if not PREREG_PATH.is_file():
        raise RuntimeError("PREREG_MISSING")
    for output in (FRAME_DIR, MP4_PATH, SHEET_PATH, MANIFEST_PATH):
        if output.exists():
            raise RuntimeError(f"FORWARD_ONLY_OUTPUT_ALREADY_EXISTS {output}")

    results = json.loads(RESULTS_PATH.read_text())
    plan = json.loads(PLAN_PATH.read_text())
    binding = results["representative_binding"]
    if binding != plan["representative_binding"]:
        raise RuntimeError("REPRESENTATIVE_BINDING_DRIFT")
    if binding != {
        "candidate_id": "side_sdg_005_raw_025092", "candidate_index": 5,
        "environment_slot": 0, "pinch_offset_index": 0,
        "selected_before_physics": True, "trial_id": "c05_o00",
    }:
        raise RuntimeError(f"REPRESENTATIVE_UNEXPECTED {binding}")
    representative = next(row for row in plan["trials"] if row["trial_id"] == binding["trial_id"])

    with np.load(TRACE_PATH, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    if trace["physics_step"].shape != (2340,) or not np.array_equal(trace["physics_step"], np.arange(1, 2341)):
        raise RuntimeError("TRACE_STEP_CONTRACT_FAIL")
    slot = int(trace["representative_environment_slot"])
    if slot != 0 or list(trace["trial_id"])[slot] != binding["trial_id"]:
        raise RuntimeError("TRACE_REPRESENTATIVE_BINDING_FAIL")
    if FRAME_INDICES.shape != (234,) or int(FRAME_INDICES[-1]) != 2330:
        raise RuntimeError("FRAME_INDEX_CONTRACT_FAIL")

    body = trace["moving_body_pos_m"][:, slot].astype(np.float64)
    fixed_base = trace["fixed_base_pos_m"][:, slot].astype(np.float64)
    obj = trace["object_pos_m"][:, slot].astype(np.float64)
    obj_q = trace["object_quat_wxyz"][:, slot].astype(np.float64)
    tcp = trace["tcp_pos_m"][:, slot].astype(np.float64)
    q5 = trace["joint_pos_deg"][:, slot, 5].astype(np.float64)
    q5_target = trace["joint_target_deg"][:, slot, 5].astype(np.float64)
    force_vec = trace["object_force_w_n"][:, slot].astype(np.float64)
    force_norm = np.linalg.norm(force_vec, axis=-1)
    fixed_force = force_norm[:, 5]
    moving_force = force_norm[:, 6]
    support_force = force_norm[:, 0]
    target_tcp = np.asarray(representative["tcp_grasp_m"], dtype=np.float64)
    target_mid = np.asarray(representative["antipodal_midpoint_base_m"], dtype=np.float64)
    initial_obj_z = float(obj[119, 2])
    success_count = int(sum(bool(v) for v in results["metrics"]["success"]))
    population_verdict = results["classification_summary"]["selected_verdict"]
    representative_class = results["classifications"][0]["label"]

    FRAME_DIR.mkdir(parents=False)
    plt.style.use("dark_background")
    frame_rows = []
    trail_stride = 20
    for output_index, source_index_raw in enumerate(FRAME_INDICES):
        source_index = int(source_index_raw)
        phase_id = int(trace["phase_id"][source_index])
        phase = PHASE_NAMES[phase_id]
        physics_step = int(trace["physics_step"][source_index])

        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#020617")
        gs = gridspec.GridSpec(3, 5, figure=fig, width_ratios=[1.25, 1.25, 1.25, 1, 1],
                               height_ratios=[1, 1, 1], wspace=0.40, hspace=0.46)
        ax3 = fig.add_subplot(gs[:, :3], projection="3d")
        points = np.vstack([fixed_base[source_index], body[source_index]])
        ax3.plot(points[:, 0], points[:, 1], points[:, 2], "-o", color="#38bdf8",
                 markerfacecolor="#e0f2fe", markersize=5, lw=3, label="actual body centers")
        ax3.scatter(*fixed_base[source_index], s=75, marker="s", color="#64748b")
        ax3.scatter(*tcp[source_index], s=80, marker="*", color="#f8fafc", label="actual TCP")
        ax3.scatter(*target_tcp, s=80, marker="x", color="#fb7185", linewidths=2.5,
                    label="planned grasp TCP")
        ax3.scatter(*target_mid, s=55, marker="+", color="#a78bfa", linewidths=2.5,
                    label="antipodal midpoint")
        draw_cylinder(ax3, obj[source_index], obj_q[source_index])
        trail_indices = np.arange(0, source_index + 1, trail_stride)
        if trail_indices.size:
            ax3.plot(obj[trail_indices, 0], obj[trail_indices, 1], obj[trail_indices, 2],
                     color="#f59e0b", alpha=0.55, lw=1.2, label="object trace")
            ax3.plot(tcp[trail_indices, 0], tcp[trail_indices, 1], tcp[trail_indices, 2],
                     color="#e2e8f0", alpha=0.25, lw=0.8)
        contacts = trace["object_contact_pos_m"][source_index, slot].astype(np.float64)
        for contact_index in range(7):
            magnitude = force_norm[source_index, contact_index]
            if magnitude > 0.01 and np.isfinite(contacts[contact_index]).all():
                vector = force_vec[source_index, contact_index]
                scale = 0.035 / max(0.3, magnitude)
                ax3.quiver(*contacts[contact_index], *(vector * scale), color="#ef4444",
                           linewidth=2.0, arrow_length_ratio=0.25)
        ax3.plot([-.03, .49], [-.08, -.08], [0, 0], alpha=0)
        gx, gy = np.meshgrid(np.linspace(-.03, .49, 2), np.linspace(-.08, .25, 2))
        ax3.plot_surface(gx, gy, np.zeros_like(gx), color="#334155", alpha=0.18, shade=False)
        ax3.set_xlim(-0.03, 0.49); ax3.set_ylim(-0.08, 0.25); ax3.set_zlim(0, 0.40)
        ax3.set_box_aspect((0.52, 0.33, 0.40))
        ax3.view_init(elev=25, azim=-123)
        ax3.set_xlabel("X [m]", fontsize=8); ax3.set_ylabel("Y [m]", fontsize=8); ax3.set_zlabel("Z [m]", fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.set_title("RoArm fixed-base PhysX trace\nbody-center skeleton (not robot mesh)",
                      loc="left", fontsize=13, color="#f8fafc", pad=12)
        ax3.legend(loc="upper left", fontsize=7, framealpha=0.45)

        x = trace["physics_step"]
        axq = fig.add_subplot(gs[0, 3:])
        style_time_axis(axq, "Gripper command tracking")
        axq.plot(x, q5_target, color="#fb7185", lw=1.0, label="target q5")
        axq.plot(x, q5, color="#38bdf8", lw=1.1, label="actual q5")
        axq.axvline(source_index, color="white", lw=1)
        axq.set_ylabel("degrees", fontsize=8); axq.legend(fontsize=7, ncol=2, loc="lower left")

        axf = fig.add_subplot(gs[1, 3:])
        style_time_axis(axf, "Object contact forces at jaws")
        axf.plot(x, fixed_force, color="#22c55e", lw=1.0, label="fixed jaw")
        axf.plot(x, moving_force, color="#f97316", lw=1.0, label="moving jaw")
        axf.axhline(0.01, color="#facc15", lw=0.8, ls="--", label="bilateral gate")
        axf.axvline(source_index, color="white", lw=1)
        axf.set_ylabel("N", fontsize=8); axf.legend(fontsize=7, ncol=3, loc="upper left")

        axz = fig.add_subplot(gs[2, 3:])
        style_time_axis(axz, "Vertical motion")
        axz.plot(x, (obj[:, 2] - initial_obj_z) * 1000, color="#f59e0b", lw=1.2,
                 label="object corrected lift")
        axz.plot(x, (tcp[:, 2] - tcp[1719, 2]) * 1000, color="#e2e8f0", lw=1.0,
                 label="TCP vs end-close")
        axz.axhline(6.0, color="#facc15", lw=0.8, ls="--", label="object gate 6 mm")
        axz.axvline(source_index, color="white", lw=1)
        axz.set_ylabel("mm", fontsize=8); axz.set_xlabel("physics step", fontsize=8)
        axz.legend(fontsize=7, ncol=2, loc="upper left")

        fig.text(0.015, 0.972, DISCLAIMER, color="#fbbf24", fontsize=10, weight="bold", va="top")
        fig.text(0.015, 0.935,
                 f"P13 actual trace | trial {binding['trial_id']} | step {physics_step}/2340 | "
                 f"phase {phase} ({int(trace['phase_step'][source_index])}/{PHASE_STEPS[phase_id]-1})",
                 color="#f8fafc", fontsize=11, va="top")
        fig.text(0.985, 0.972,
                 f"GRASP FAILED: {success_count}/5 success\n"
                 f"representative: {representative_class} | population: {population_verdict}",
                 color="#fca5a5", fontsize=9.5, weight="bold", va="top", ha="right")
        fig.text(0.985, 0.015,
                 f"now: q5={q5[source_index]:.1f}° target={q5_target[source_index]:.1f}° | "
                 f"jaw forces={fixed_force[source_index]:.3f}/{moving_force[source_index]:.3f} N | "
                 f"support={support_force[source_index]:.3f} N | "
                 f"object Δz={(obj[source_index,2]-initial_obj_z)*1000:+.3f} mm",
                 color="#cbd5e1", fontsize=8.5, ha="right", va="bottom")
        frame_path = FRAME_DIR / f"frame_{output_index:04d}.png"
        fig.savefig(frame_path, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        with Image.open(frame_path) as im:
            if im.size != (WIDTH, HEIGHT):
                raise RuntimeError(f"FRAME_SIZE_FAIL {frame_path} {im.size}")
        frame_rows.append({
            "frame_index": output_index,
            "source_trace_index": source_index,
            "physics_step": physics_step,
            "phase": phase,
            "sha256": sha256_file(frame_path),
            "bytes": frame_path.stat().st_size,
        })
        if output_index in (0, 59, 119, 179, 233):
            print(f"FRAME_PROGRESS {output_index + 1}/234 source_step={source_index} phase={phase}", flush=True)

    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS),
        "-i", str(FRAME_DIR / "frame_%04d.png"), "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(MP4_PATH),
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    sheet_indices = [0, 20, 52, 91, 131, 171, 183, 233]
    thumbs = []
    for index in sheet_indices:
        image = Image.open(FRAME_DIR / f"frame_{index:04d}.png").convert("RGB")
        image.thumbnail((640, 360), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    sheet = Image.new("RGB", (1280, 1440), "#020617")
    draw = ImageDraw.Draw(sheet)
    for position, (index, thumb) in enumerate(zip(sheet_indices, thumbs)):
        x0 = (position % 2) * 640
        y0 = (position // 2) * 360
        sheet.paste(thumb, (x0, y0))
        draw.rectangle((x0 + 5, y0 + 5, x0 + 235, y0 + 28), fill="#020617")
        draw.text((x0 + 12, y0 + 9), f"video frame {index:03d} / source step {int(FRAME_INDICES[index])}", fill="white")
    sheet.save(SHEET_PATH)

    probe_cmd = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries", "format=duration,size", "-of", "json", str(MP4_PATH),
    ]
    probe = json.loads(subprocess.run(probe_cmd, check=True, capture_output=True, text=True).stdout)
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(MP4_PATH), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    stream = probe["streams"][0]
    completion_checks = {
        "frozen_input_hashes_match": True,
        "trace_exact_2340_steps": True,
        "representative_binding_exact": True,
        "frame_count_exact_234": len(frame_rows) == 234 and len(list(FRAME_DIR.glob("frame_*.png"))) == 234,
        "frame_dimensions_1280x720": True,
        "codec_h264": stream.get("codec_name") == "h264",
        "pixel_format_yuv420p": stream.get("pix_fmt") == "yuv420p",
        "video_dimensions_1280x720": stream.get("width") == WIDTH and stream.get("height") == HEIGHT,
        "video_rate_20fps": stream.get("r_frame_rate") == "20/1" and stream.get("avg_frame_rate") == "20/1",
        "video_frames_234": stream.get("nb_frames") == "234" and stream.get("nb_read_frames") == "234",
        "video_duration_11p7s": abs(float(stream.get("duration", probe["format"]["duration"])) - 11.7) < 1e-6,
        "full_decode_exit_zero": decode.returncode == 0,
        "contact_sheet_nonempty": SHEET_PATH.stat().st_size > 0,
        "non_authority_label_required": True,
    }
    manifest = {
        "artifact": "T3U_SIDE_MEETING1_POSTHOC_TRACE_VIDEO_V1",
        "created_after_preregistration": True,
        "authority": "posthoc_trace_visualization_not_rtx_not_scientific_authority",
        "does_not_change_p13_science_or_results": True,
        "inputs": {
            name: {"path": str((CASE / name).relative_to(ROOT)), "sha256": expected}
            for name, expected in INPUT_SHA256.items()
        },
        "representative_binding": binding,
        "source_trace": {"steps": 2340, "stride": 10, "frame_indices_first_last": [0, 2330]},
        "frozen_outcome": {
            "success_count": success_count,
            "population_count": len(results["metrics"]["success"]),
            "representative_classification": representative_class,
            "population_selected_verdict": population_verdict,
            "representative_metrics": {
                key: results["metrics"][key][0]
                for key in ("preclose_jaw_max", "close_fixed_max", "close_moving_max",
                            "lift_fixed_max", "lift_moving_max", "lift_corrected_mm",
                            "lift_tcp_rise_mm", "final_tilt_deg", "success")
            },
        },
        "video": {
            "path": str(MP4_PATH.relative_to(ROOT)), "sha256": sha256_file(MP4_PATH),
            "bytes": MP4_PATH.stat().st_size, "ffprobe": probe,
            "full_decode_exit_code": decode.returncode, "full_decode_stderr": decode.stderr,
        },
        "contact_sheet": {
            "path": str(SHEET_PATH.relative_to(ROOT)), "sha256": sha256_file(SHEET_PATH),
            "bytes": SHEET_PATH.stat().st_size, "selected_frame_indices": sheet_indices,
        },
        "renderer": {"path": str(SOURCE_PATH.relative_to(ROOT)), "sha256": sha256_file(SOURCE_PATH)},
        "preregistration": {"path": str(PREREG_PATH.relative_to(ROOT)), "sha256": sha256_file(PREREG_PATH)},
        "frames": frame_rows,
        "phase_coverage": sorted({row["phase"] for row in frame_rows}),
        "completion_checks": completion_checks,
        "pass": all(completion_checks.values()),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not manifest["pass"]:
        raise RuntimeError(f"COMPLETION_GATE_FAIL {completion_checks}")
    print(json.dumps({
        "status": "PASS", "mp4": str(MP4_PATH), "contact_sheet": str(SHEET_PATH),
        "manifest": str(MANIFEST_PATH), "mp4_sha256": manifest["video"]["sha256"],
        "frames": 234, "duration_s": 11.7,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
