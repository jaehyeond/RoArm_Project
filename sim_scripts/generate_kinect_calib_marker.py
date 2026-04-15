#!/usr/bin/env python
"""Generate A4 printable red-circle marker for Kinect hand-eye calibration.

Marker: 20 mm diameter solid red circle on A4.
Print at 100% scale (disable 'fit to page' / 'scale to fit').
Verify the 10 mm scale bar with a caliper BEFORE cutting.

Rationale for 20 mm:
- Kinect RGB 720P @ ~1 m working distance => ~13-17 px diameter
  => HSV centroid uncertainty ~ +/- 1 mm (sub-pixel mass).
- link5 top flat patch is ~25-30 mm wide on RoArm-M3
  => 20 mm fits safely without touching wrist_roll drum curvature.

Warning: red (HSV ~0 deg) aliases with the pink sponge used in Step 2
(HSV 155-180 u 0-12). Remove the sponge from the Kinect view before
capturing hand-eye data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

A4_W_MM, A4_H_MM = 210.0, 297.0
MARKER_DIA_MM = 20.0
CUT_SIDE_MM = 40.0

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "claudedocs"
OUT_DIR.mkdir(exist_ok=True)

fig = plt.figure(figsize=(A4_W_MM / 25.4, A4_H_MM / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, A4_W_MM)
ax.set_ylim(0, A4_H_MM)
ax.set_aspect("equal")
ax.axis("off")

cx, cy = A4_W_MM / 2.0, 215.0

ax.add_patch(
    Circle((cx, cy), MARKER_DIA_MM / 2.0, facecolor=(1.0, 0.0, 0.0), edgecolor="none", zorder=3)
)
ax.plot([cx - 1.5, cx + 1.5], [cy, cy], color=(0.25, 0.25, 0.25), lw=0.3, zorder=4)
ax.plot([cx, cx], [cy - 1.5, cy + 1.5], color=(0.25, 0.25, 0.25), lw=0.3, zorder=4)

ax.add_patch(
    Rectangle(
        (cx - CUT_SIDE_MM / 2, cy - CUT_SIDE_MM / 2),
        CUT_SIDE_MM,
        CUT_SIDE_MM,
        facecolor="none",
        edgecolor=(0.5, 0.5, 0.5),
        lw=0.5,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
)
ax.text(
    cx + CUT_SIDE_MM / 2 + 2,
    cy + CUT_SIDE_MM / 2 - 2,
    "cut",
    fontsize=6,
    color=(0.4, 0.4, 0.4),
    ha="left",
    va="top",
)


def hbar(x0_mm: float, y_mm: float, length_mm: float, label: str) -> None:
    ax.plot([x0_mm, x0_mm + length_mm], [y_mm, y_mm], "k-", lw=1.0)
    ax.plot([x0_mm, x0_mm], [y_mm - 1.2, y_mm + 1.2], "k-", lw=1.0)
    ax.plot([x0_mm + length_mm, x0_mm + length_mm], [y_mm - 1.2, y_mm + 1.2], "k-", lw=1.0)
    ax.text(x0_mm + length_mm / 2, y_mm - 5, label, ha="center", fontsize=8)


hbar(20, 28, 10.0, "10 mm")
hbar(60, 28, 25.4, "1 inch (25.4 mm)")
hbar(120, 28, 50.0, "50 mm")

ax.text(
    A4_W_MM / 2,
    A4_H_MM - 20,
    "RoArm-M3 Kinect Hand-Eye Calibration Marker",
    ha="center",
    fontsize=12,
    weight="bold",
)
ax.text(
    A4_W_MM / 2,
    A4_H_MM - 32,
    f"{MARKER_DIA_MM:.0f} mm diameter solid red circle",
    ha="center",
    fontsize=10,
)
ax.text(
    A4_W_MM / 2,
    A4_H_MM - 42,
    "PRINT AT 100% (disable 'fit to page'). Verify scale bars with caliper BEFORE cutting.",
    ha="center",
    fontsize=8,
    style="italic",
)

instructions = [
    "Placement:",
    "  1. Cut along the dashed square.",
    "  2. Place marker CENTER on link5 top flat patch",
    "     (wrist_roll drum / gripper housing boundary).",
    "  3. Fix with clear scotch tape on the FOUR CORNERS only.",
    "     Do NOT tape over the red circle (reflection / color shift).",
    "",
    "Before capture:",
    "  4. Gripper = CLOSED (joint 5 ~ 0 deg). joint 5 does not affect",
    "     link5 FK, but CLOSED gives consistent silhouette and avoids",
    "     finger-catch on cables during the 28-pose sweep.",
    "  5. Measure hand_tcp <-> marker-center offset with a caliper",
    "     (URDF says Z=115.428 mm, photo analysis says ~66 mm;",
    "     the 50 mm discrepancy must be resolved by measurement).",
    "     Save in kinect_calib.yaml under [marker_offset].",
    "  6. Remove the pink sponge from the Kinect view",
    "     (red marker HSV aliases with sponge pink).",
]
y_top = 180
for i, line in enumerate(instructions):
    ax.text(20, y_top - i * 7.0, line, ha="left", fontsize=8.5, family="monospace")

pdf_path = OUT_DIR / "kinect_calib_marker_A4.pdf"
png_path = OUT_DIR / "kinect_calib_marker_A4.png"
fig.savefig(pdf_path, format="pdf")
fig.savefig(png_path, format="png", dpi=300)
plt.close(fig)

print(f"Saved PDF: {pdf_path}")
print(f"Saved PNG: {png_path}")
print(f"Marker spec: {MARKER_DIA_MM:.1f} mm diameter, solid red (255, 0, 0)")
print("Verify print scale with caliper against the 10 mm / 25.4 mm / 50 mm bars.")
