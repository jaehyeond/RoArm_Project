"""
figure_p6v12_labmeeting.py
C2 Analysis & Visualization Specialist — P6v12 Lab Meeting Figures

Usage:
    # With live log (after scp):
    python figure_p6v12_labmeeting.py --log /tmp/train_p6v12.out

    # Dry-run with embedded P6v12 iter-table data:
    python figure_p6v12_labmeeting.py --dry-run

Output: /home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/figures/
    p6v12_learning_curves_4panel.png
    p6v12_vs_p6v11_bar.png
    p6v12_stage_allocation.png
    p6v12_failure_diagnosis.png
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
import numpy as np
import pandas as pd
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COL_P6V11 = "#1f77b4"   # blue
COL_P6V12 = "#ff7f0e"   # orange
COL_REF   = "#999999"   # gray reference line
COL_WARN  = "#d62728"   # red highlight

OUT_DIR = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/figures")


# ── P6v11 final-iter reference values (from claudedocs result md) ──────────────
P6V11_FINAL = {
    "reward":         1059,
    "gripper_open":   0.0634,
    "is_on_target":   0.0161,
    "stage4":         0.0,
    "stage2_grasp":   0.849,
    "stage3_near":    0.0161,
    "xy_offset_m":    0.0671,
    "z_offset_m":     0.0479,
    "action_std":     1.44,
    "is_success_zone":0.526,
}

# ── P6v12 known iter-table (from claudedocs result md) ───────────────────────
# Columns: iter, reward, gripper_open, is_on_target, stage4, stage2_grasp,
#          stage3_near, xy_off, z_off, std
P6V12_TABLE = [
    # iter  reward  grip_open  on_tgt  stage4   s2      s3      xy      z     std
    (0,     None,   0.070,     0.019,  0.0003,  0.82,   0.019,  0.166,  0.054, 1.00),
    (65,    637,    0.072,     0.185,  0.0,     0.67,   0.185,  0.069,  0.044, 1.00),
    (247,   755,    0.069,     0.320,  0.0,     0.51,   0.320,  0.072,  0.046, 0.97),
    (351,   787,    0.069,     0.346,  0.0,     0.51,   0.346,  0.072,  0.044, 0.96),
    (532,   794,    0.062,     0.397,  0.0,     None,   0.397,  None,   None,  None),
    (750,   822,    0.064,     0.394,  0.0,     None,   0.394,  None,   None,  None),
    (966,   824,    0.064,     0.397,  0.0001,  None,   0.397,  None,   None,  None),
    (999,   854,    0.064,     0.406,  0.0002,  0.45,   0.406,  0.081,  0.048, 0.88),
]

COLS = ["iter","reward","gripper_open","is_on_target","stage4",
        "stage2_grasp","stage3_near","xy_off","z_off","std"]


# ── Parser (live log) ─────────────────────────────────────────────────────────
METRIC_PATTERNS = {
    "reward":         r"Mean reward\s*:\s*([\d.eE+\-]+)",
    "gripper_open":   r"gripper_open_rate\s*:\s*([\d.eE+\-]+)",
    "is_on_target":   r"is_on_target_rate\s*:\s*([\d.eE+\-]+)",
    "stage4":         r"stage4_success_frac\s*:\s*([\d.eE+\-]+)",
    "stage2_grasp":   r"stage2_grasp_frac\s*:\s*([\d.eE+\-]+)",
    "stage3_near":    r"stage3_neartgt_frac\s*:\s*([\d.eE+\-]+)",
    "xy_off":         r"xy_offset_mean\s*:\s*([\d.eE+\-]+)",
    "z_off":          r"z_offset_mean\s*:\s*([\d.eE+\-]+)",
    "std":            r"Mean action noise std\s*:\s*([\d.eE+\-]+)",
    "is_success_zone":r"is_success_zone_rate\s*:\s*([\d.eE+\-]+)",
    "near_target":    r"near_target_rate\s*:\s*([\d.eE+\-]+)",
}

def parse_log(path: str) -> pd.DataFrame:
    iter_pat = re.compile(r"Learning iteration\s+(\d+)/\d+")
    records = []
    current_iter = None
    current = {}

    with open(path, "r", errors="replace") as f:
        for line in f:
            # strip ANSI
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = iter_pat.search(clean)
            if m:
                if current_iter is not None and current:
                    current["iter"] = current_iter
                    records.append(current.copy())
                current_iter = int(m.group(1))
                current = {}
                continue
            for key, pat in METRIC_PATTERNS.items():
                m2 = re.search(pat, clean)
                if m2:
                    current[key] = float(m2.group(1))

    if current_iter is not None and current:
        current["iter"] = current_iter
        records.append(current)

    df = pd.DataFrame(records)
    if "iter" in df.columns:
        df = df.sort_values("iter").reset_index(drop=True)
    return df


def build_dry_run_df() -> pd.DataFrame:
    df = pd.DataFrame(P6V12_TABLE, columns=COLS)
    # fill forward for missing reward at iter 0
    df["reward"] = df["reward"].interpolate(method="linear", limit_direction="both")
    return df


# ── Helper ────────────────────────────────────────────────────────────────────
def regression_slope(iters, values):
    """Return linear regression slope and r-value."""
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept, r, p, se = stats.linregress(iters[mask], values[mask])
    return slope, r


def smooth(y, w=5):
    """Rolling mean, edge-padded."""
    y = np.array(y, dtype=float)
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


# ── PNG 1: 4-panel learning curves ───────────────────────────────────────────
def plot_4panel(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("P6v12 Learning Curves (η-v1 fix, 1000 iter)\nresume P6v11 model_999",
                 fontsize=14, fontweight="bold", y=1.01)

    iters = df["iter"].values

    # ── (a) Mean reward ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    vals = df["reward"].values.astype(float)
    ax.plot(iters, vals, color=COL_P6V12, lw=1.5, alpha=0.5, label="P6v12 raw")
    if len(iters) > 5:
        ax.plot(iters, smooth(vals), color=COL_P6V12, lw=2.5, label="P6v12 smooth")
    ax.axhline(P6V11_FINAL["reward"], color=COL_P6V11, lw=1.5, ls="--",
               label=f"P6v11 final ({P6V11_FINAL['reward']:.0f})")
    ax.set_title("(a) Mean Reward", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=9)
    # annotate final value
    ax.annotate(f"{vals[-1]:.0f}", xy=(iters[-1], vals[-1]),
                xytext=(-30, 8), textcoords="offset points",
                color=COL_P6V12, fontsize=10, fontweight="bold")
    ax.annotate("-19% vs P6v11", xy=(iters[-1], vals[-1]),
                xytext=(-30, -16), textcoords="offset points",
                color=COL_WARN, fontsize=9)

    # ── (b) gripper_open_rate ────────────────────────────────────────────────
    ax = axes[0, 1]
    vals = df["gripper_open"].values.astype(float)
    ax.plot(iters, vals, color=COL_P6V12, lw=2.0, marker="o", markersize=4,
            label="P6v12 gripper_open")
    ax.axhline(P6V11_FINAL["gripper_open"], color=COL_P6V11, lw=1.5, ls="--",
               label=f"P6v11 final ({P6V11_FINAL['gripper_open']:.3f})")
    ax.set_ylim(0.0, 0.20)
    ax.set_title("(b) Gripper Open Rate  ← KEY FAIL", fontweight="bold", color=COL_WARN)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=9)
    # regression slope
    slope, r = regression_slope(iters.astype(float), vals)
    ax.text(0.05, 0.90,
            f"Linear slope: {slope*1000:.4f}/1k iter\n(r={r:.3f}) → FLAT",
            transform=ax.transAxes, fontsize=9, color=COL_WARN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0", edgecolor=COL_WARN, alpha=0.8))
    # red shaded region emphasizing flatness
    ax.fill_between(iters, vals - 0.005, vals + 0.005, alpha=0.15, color=COL_WARN)
    ax.annotate("FLAT — release 학습 0\n(η-v1 transient: gripper_open gate 누락)",
                xy=(iters[len(iters)//2], vals[len(vals)//2]),
                xytext=(20, 40), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=COL_WARN),
                color=COL_WARN, fontsize=9)

    # ── (c) is_on_target_rate ────────────────────────────────────────────────
    ax = axes[1, 0]
    vals = df["is_on_target"].values.astype(float)
    ax.plot(iters, vals, color=COL_P6V12, lw=1.5, alpha=0.5)
    if len(iters) > 5:
        ax.plot(iters, smooth(vals), color=COL_P6V12, lw=2.5, label="P6v12 smooth")
    ax.axhline(P6V11_FINAL["is_on_target"], color=COL_P6V11, lw=1.5, ls="--",
               label=f"P6v11 final ({P6V11_FINAL['is_on_target']:.3f})")
    slope, r = regression_slope(iters.astype(float), vals)
    ax.set_title("(c) is_on_target Rate  ← 25× ↑", fontweight="bold", color="green")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=9)
    ax.annotate(f"25× ↑\n{P6V11_FINAL['is_on_target']:.3f} → {vals[-1]:.3f}",
                xy=(iters[-1], vals[-1]),
                xytext=(-60, -20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="green"),
                color="green", fontsize=10, fontweight="bold")
    ax.text(0.05, 0.55,
            f"Slope: +{slope*1000:.4f}/1k iter\n(r={r:.3f})",
            transform=ax.transAxes, fontsize=9, color="green",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fff0", edgecolor="green", alpha=0.8))

    # ── (d) stage4_success_frac ──────────────────────────────────────────────
    ax = axes[1, 1]
    vals = df["stage4"].values.astype(float)
    ax.plot(iters, vals, color=COL_P6V12, lw=2.0, marker="o", markersize=4,
            label="P6v12 stage4")
    ax.axhline(0, color=COL_P6V11, lw=1.5, ls="--", label="P6v11 final (0.0000)")
    ax.set_ylim(-0.00005, max(max(vals) * 2.5, 0.001))
    ax.set_title("(d) Stage-4 Success Frac  ← Still ~0", fontweight="bold", color=COL_WARN)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=9)
    ax.annotate(f"Final: {vals[-1]:.4f}\n(~1/4096 sporadic)",
                xy=(iters[-1], vals[-1]),
                xytext=(-60, 15), textcoords="offset points",
                color=COL_WARN, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=COL_WARN))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG1] Saved: {out_path}")


# ── PNG 2: P6v11 vs P6v12 bar chart ──────────────────────────────────────────
def plot_bar_comparison(out_path: Path):
    metrics = [
        ("Mean Reward",        P6V11_FINAL["reward"],       854,    "reward"),
        ("Gripper Open Rate",  P6V11_FINAL["gripper_open"], 0.064,  "frac"),
        ("is_on_target Rate",  P6V11_FINAL["is_on_target"], 0.406,  "frac"),
        ("Stage-4 Success",    0.0001,                      0.0002, "frac"),  # tiny P6v11 approx
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle("P6v11 vs P6v12 Final-Iter Comparison (iter 999)",
                 fontsize=14, fontweight="bold")

    for ax, (label, v11, v12, kind) in zip(axes, metrics):
        bars = ax.bar(["P6v11", "P6v12"], [v11, v12],
                      color=[COL_P6V11, COL_P6V12], width=0.5,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_ylabel("Value")

        # annotate values
        for bar, val in zip(bars, [v11, v12]):
            ypos = bar.get_height()
            fmt = ".0f" if kind == "reward" else ".4f"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    ypos + max(ypos * 0.03, 1e-5),
                    format(val, fmt), ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        # delta annotation
        if v11 > 0:
            ratio = v12 / v11
            if ratio >= 10:
                delta_str = f"{ratio:.0f}×"
                color = "green"
            elif ratio > 1.1:
                delta_str = f"+{(ratio-1)*100:.0f}%"
                color = "green"
            elif ratio < 0.9:
                delta_str = f"{(ratio-1)*100:.0f}%"
                color = COL_WARN
            else:
                delta_str = "FLAT"
                color = COL_WARN
            ax.text(0.5, 0.95, delta_str, transform=ax.transAxes,
                    ha="center", va="top", fontsize=12, fontweight="bold",
                    color=color)

        ax.set_ylim(0, max(v11, v12) * 1.3 + 1e-6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG2] Saved: {out_path}")


# ── PNG 3: Stage allocation stacked bar ──────────────────────────────────────
def plot_stage_allocation(out_path: Path):
    """
    Stage fraction = fraction of episode time the policy spent in each stage.
    P6v11 iter 999: stage1=0.135, stage2=0.849, stage3=0.016, stage4=0.0
    P6v12 iter 999: stage1=~0.135, stage2=0.45,  stage3=0.41,  stage4=0.0002
    (stage fracs sum ~1 with rounding)
    """
    labels = ["P6v11", "P6v12"]

    s1 = [0.135, 0.135]
    s2 = [0.849, 0.450]
    s3 = [0.016, 0.410]
    s4 = [0.000, 0.000]  # near zero for both

    colors = ["#aec7e8", "#1f77b4", "#ff7f0e", "#2ca02c"]
    stage_labels = ["Stage 1: Reach", "Stage 2: Grasp/Hold", "Stage 3: Near-Target", "Stage 4: Release/Place"]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Reward-Farm Location Shift: P6v11 → P6v12",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(labels))
    width = 0.45

    bottoms = np.zeros(len(labels))
    bars_all = []
    for arr, col, slabel in zip([s1, s2, s3, s4], colors, stage_labels):
        arr = np.array(arr, dtype=float)
        b = ax.bar(x, arr, width, bottom=bottoms, color=col, label=slabel,
                   edgecolor="white", linewidth=0.8)
        bars_all.append((arr, col, slabel))
        # label inside bar if big enough
        for xi, (val, bot) in enumerate(zip(arr, bottoms)):
            if val > 0.03:
                ax.text(xi, bot + val / 2, f"{val:.0%}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white")
        bottoms += arr

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
    ax.set_ylabel("Fraction of Episode Time", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)

    # annotation arrows showing the shift
    ax.annotate("", xy=(1, 0.135 + 0.245),   # midpoint of s2 in P6v12
                xytext=(0, 0.135 + 0.425),    # midpoint of s2 in P6v11
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(0.5, 0.58,
            "Stage 2 farming\n0.85 → 0.45 (↓47%)",
            ha="center", fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.annotate("", xy=(1, 0.135 + 0.45 + 0.205),  # midpoint of s3 in P6v12
                xytext=(0.12, 0.135 + 0.849 + 0.005),
                arrowprops=dict(arrowstyle="->", color=colors[2], lw=1.5))
    ax.text(1.12, 0.80,
            "Stage 3 farming\n~0 → 0.41 (NEW)",
            ha="center", fontsize=9, color=colors[2], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=colors[2], alpha=0.8))

    ax.text(0.5, -0.12,
            '"η-v1: Stage 2 cap으로 farming location을 Stage 3으로 이동시켰을 뿐,\n'
            'release(Stage 4) 학습은 여전히 0 — reward-farm 근본 해결 필요"',
            transform=ax.transAxes, ha="center", fontsize=9, style="italic",
            color="#555555")

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG3] Saved: {out_path}")


# ── PNG 4: Failure diagnosis — 1-step reward margin diagram ──────────────────
def plot_failure_diagnosis(out_path: Path):
    """
    Horizontal reward-margin diagram showing stage 3 close vs open margins.
    Illustrates η-v1 design flaw: transient fires regardless of gripper state.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("P6v12 η-v1 Design Flaw: 1-Step Reward Margin Diagram",
                 fontsize=13, fontweight="bold")

    # States along x, reward along y
    states = [
        "Stage 2\n(Far, hold\ngrasped)",
        "Stage 2\n(Near ≤10cm\nhold grasped)",
        "Stage 3\n(At zone\nCLOSED)",
        "Stage 3\n(At zone\nOPEN)",
        "Stage 4\n(Latched\nreleased)",
    ]
    # Base reward per step at each state
    rewards = [5.5, 2.0, 6.5, 7.0, 8.0]
    colors_bar = [COL_P6V11, COL_P6V11, COL_WARN, "green", "green"]

    x = np.arange(len(states))
    bars = ax.bar(x, rewards, color=colors_bar, width=0.55,
                  edgecolor="white", linewidth=1.0, alpha=0.85)

    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(states, fontsize=10)
    ax.set_ylabel("Reward per Step", fontsize=12)
    ax.set_ylim(0, 12)

    # -- Transient +10 annotation at Stage 3 CLOSED
    ax.bar([2], [10], bottom=[6.5], color=COL_WARN, width=0.55,
           alpha=0.45, hatch="///", edgecolor=COL_WARN, linewidth=1.0)
    ax.text(2, 6.5 + 5, "+10\n(transient,\nfires CLOSED too!)",
            ha="center", va="center", fontsize=9, color=COL_WARN, fontweight="bold")

    # -- Transient +10 annotation at Stage 3 OPEN (correct behavior)
    ax.bar([3], [10], bottom=[7.0], color="green", width=0.55,
           alpha=0.35, hatch="///", edgecolor="green", linewidth=1.0)
    ax.text(3, 7.0 + 5, "+10\n(transient,\nshould fire here)",
            ha="center", va="center", fontsize=9, color="green", fontweight="bold")

    # -- Margin annotations
    # Stage 2 near-cap → Stage 3 close (farming shift)
    ax.annotate("", xy=(1.27, 6.5), xytext=(1.27, 2.0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1.40, 4.25, "+4.5\n(stage3 close\nvs s2 near-cap)\n→ farm to s3",
            ha="left", fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    # Close vs Open margin at stage 3 (persistent, post-transient)
    ax.annotate("", xy=(2.27, 7.0), xytext=(2.27, 6.5),
                arrowprops=dict(arrowstyle="<->", color=COL_WARN, lw=2.0))
    ax.text(2.35, 6.75, "+0.5 only\n(insufficient!\n≈ noise)",
            ha="left", fontsize=9, color=COL_WARN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff0f0", edgecolor=COL_WARN, alpha=0.8))

    # Stage 4 vs Stage 3 open
    ax.annotate("", xy=(3.27, 8.0), xytext=(3.27, 7.0),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(3.35, 7.5, "+1.0\n(but sponge_stable\nhigh variance)",
            ha="left", fontsize=9, color="green",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0fff0", edgecolor="green", alpha=0.8))

    # Policy's learned behavior arrow
    ax.annotate("PPO learns:\nStage 3 CLOSED hover\n(farming optimum)",
                xy=(2, 6.5), xytext=(2, 11.0),
                arrowprops=dict(arrowstyle="->", color=COL_WARN, lw=2),
                ha="center", fontsize=10, color=COL_WARN, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0", edgecolor=COL_WARN))

    # η-v2 fix note
    ax.text(0.98, 0.08,
            "η-v2 Fix (P6v13): transient gate = is_on_target & gripper_open\n"
            "Stage 3 close-cap = 3.0 → close 3.0 vs open 7.0 = +4.0 margin",
            transform=ax.transAxes, ha="right", fontsize=9, style="italic",
            color="green",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fff0", edgecolor="green", alpha=0.8))

    ax.legend(handles=[
        mpatches.Patch(color=COL_P6V11, alpha=0.85, label="Hold/Grasp (blue = current farm)"),
        mpatches.Patch(color=COL_WARN, alpha=0.85, label="Problematic close-state"),
        mpatches.Patch(color="green", alpha=0.85, label="Target open-state"),
    ], loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG4] Saved: {out_path}")


# ── Statistical summary ───────────────────────────────────────────────────────
def print_statistics(df: pd.DataFrame):
    iters = df["iter"].values.astype(float)
    print("\n" + "="*60)
    print("  C2 STATISTICAL SUMMARY — P6v12 (1000 iter)")
    print("="*60)

    for col, label in [
        ("gripper_open", "gripper_open_rate"),
        ("is_on_target", "is_on_target_rate"),
        ("reward",       "Mean reward"),
        ("stage4",       "stage4_success_frac"),
    ]:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values.astype(float)
        iters_sub = df.loc[df[col].notna(), "iter"].values.astype(float)
        if len(vals) < 2:
            continue
        slope, r = regression_slope(iters_sub, vals)
        print(f"\n  [{label}]")
        print(f"    iter 0 value : {vals[0]:.5f}")
        print(f"    iter 999 val : {vals[-1]:.5f}")
        print(f"    Linear slope : {slope:.6f} /iter  ({slope*1000:.4f} /1k-iter)")
        print(f"    Pearson r    : {r:.4f}")
        if abs(slope) * 1000 < 0.005:
            print(f"    Verdict      : FLAT (|slope| < 0.005/1k-iter)")
        elif slope > 0:
            print(f"    Verdict      : RISING")
        else:
            print(f"    Verdict      : FALLING")

    print("\n  P6v11 vs P6v12 final-iter comparison:")
    comparisons = [
        ("reward",       P6V11_FINAL["reward"],       854),
        ("gripper_open", P6V11_FINAL["gripper_open"], 0.064),
        ("is_on_target", P6V11_FINAL["is_on_target"], 0.406),
        ("stage4",       0.0,                         0.0002),
    ]
    for metric, v11, v12 in comparisons:
        if v11 > 0:
            change = (v12 - v11) / v11 * 100
            print(f"    {metric:20s}: {v11:.4f} → {v12:.4f}  ({change:+.1f}%)")
        else:
            print(f"    {metric:20s}: {v11:.4f} → {v12:.4f}  (new signal)")
    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="", help="Path to train_p6v12.out")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use embedded P6v12 iter-table (no log file needed)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run or not args.log:
        print("[INFO] Using embedded P6v12 iter-table (dry-run mode)")
        df = build_dry_run_df()
    else:
        if not os.path.exists(args.log):
            print(f"[ERROR] Log file not found: {args.log}")
            print("Run: scp JHPark:/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/logs/phase1Balpha/train_p6v12.out /tmp/train_p6v12.out")
            sys.exit(1)
        print(f"[INFO] Parsing live log: {args.log}")
        df = parse_log(args.log)
        if df.empty:
            print("[WARN] Log parsed but empty — falling back to dry-run table")
            df = build_dry_run_df()

    print(f"[INFO] DataFrame shape: {df.shape}")
    print(df.head())

    p1 = OUT_DIR / "p6v12_learning_curves_4panel.png"
    p2 = OUT_DIR / "p6v12_vs_p6v11_bar.png"
    p3 = OUT_DIR / "p6v12_stage_allocation.png"
    p4 = OUT_DIR / "p6v12_failure_diagnosis.png"

    plot_4panel(df, p1)
    plot_bar_comparison(p2)
    plot_stage_allocation(p3)
    plot_failure_diagnosis(p4)

    print_statistics(df)

    print("\n" + "="*60)
    print("  OUTPUT FILES")
    print("="*60)
    for p in [p1, p2, p3, p4]:
        print(f"  {p}")
    print("="*60)


if __name__ == "__main__":
    main()
