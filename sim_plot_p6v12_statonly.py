"""sim_plot_p6v12_statonly.py — P6v12 failure mode visualization (NO Isaac Lab required).

대안 경로: B200에서 state rollout npz 추출 → 로컬에서 matplotlib plot.
Isaac Lab / USD 없이 4개 figure 생성 가능.

Usage:
  # Step 1 (B200): extract rollout data
  #   ssh JHPark
  #   cd /NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/
  #   source env.sh && micromamba activate envs/isaacsim_5_1
  #   export OMNI_KIT_ACCEPT_EULA=YES ROARM_M3_USD_PATH=assets/roarm_m3/usd/roarm_m3.usd
  #   python -m roarm_rl.eval_rollout_export \
  #       --checkpoint logs/roarm_rl/p6v12_eta_stage2cap_stage3transient_resumeP6v11/model_999.pt \
  #       --num_envs 256 --num_episodes 5 --out /tmp/p6v12_rollout.npz
  #   # scp back:
  #   exit
  #   scp JHPark:/tmp/p6v12_rollout.npz /tmp/p6v12_rollout.npz

  # Step 2 (local, conda env roarm or any Python with numpy/matplotlib):
  #   python sim_plot_p6v12_statonly.py --npz /tmp/p6v12_rollout.npz

  # OR: run with synthetic demo data to verify figure layout:
  #   python sim_plot_p6v12_statonly.py --demo

Output:
  claudedocs/figures/p6v12_trajectory_top.png
  claudedocs/figures/p6v12_trajectory_side.png
  claudedocs/figures/p6v12_failure_mode_snapshot.png
  claudedocs/figures/p6v12_diagnosis_diagram.png
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# Geometry constants (HARD RULE #19/#20, from roarm_stack_env.py)
TABLE_Z     = -0.012117
TARGET_XYZ  = (0.280, -0.0435, TABLE_Z + 0.047 / 2)   # L1.spot1
HOME_XYZ    = (0.0, 0.0, 0.2)   # approx TCP home position
ZONE_50MM   = 0.050
SPONGE_H    = 0.047

OUT_DIR     = "claudedocs/figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=None, help="npz file from B200 rollout export")
    p.add_argument("--demo", action="store_true",
                   help="Generate synthetic demo data (no B200 needed)")
    p.add_argument("--out_dir", default=OUT_DIR)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic demo data that mimics P6v12 failure mode
# ──────────────────────────────────────────────────────────────────────────────
def generate_demo_episode(n_steps=200, seed=0):
    """Synthetic trajectory: pick up sponge, transport to zone, fail to release."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_steps)

    # Sponge starts random in source region (R1/R2/R3/R4)
    sp_x0, sp_y0 = 0.20 + rng.uniform(-0.05, 0.05), -0.15 + rng.uniform(-0.03, 0.03)
    sp_z0 = TABLE_Z + SPONGE_H / 2.0

    # Phase 0-25%: approach sponge (TCP moves toward sponge)
    # Phase 25-40%: grasp (TCP on sponge)
    # Phase 40-90%: transport toward target
    # Phase 90-100%: at target, hover — gripper stays closed (failure)

    tp = t.copy()
    phase = np.zeros(n_steps)
    phase[int(n_steps*0.25):int(n_steps*0.40)] = 1  # grasping
    phase[int(n_steps*0.40):] = 2  # transport + hover

    # TCP trajectory
    def lerp(a, b, s): return a + (b - a) * np.clip(s, 0, 1)

    tcp_x = np.zeros(n_steps)
    tcp_y = np.zeros(n_steps)
    tcp_z = np.zeros(n_steps)

    # Approach: 0→sp_x, sp_y, z=0.12
    tcp_x[:int(n_steps*0.25)] = lerp(0.0, sp_x0, np.linspace(0, 1, int(n_steps*0.25)))
    tcp_y[:int(n_steps*0.25)] = lerp(0.0, sp_y0, np.linspace(0, 1, int(n_steps*0.25)))
    tcp_z[:int(n_steps*0.25)] = lerp(0.12, sp_z0 + 0.02, np.linspace(0, 1, int(n_steps*0.25)))

    # Grasp: hold at sponge
    i0, i1 = int(n_steps*0.25), int(n_steps*0.40)
    tcp_x[i0:i1] = sp_x0
    tcp_y[i0:i1] = sp_y0
    tcp_z[i0:i1] = lerp(sp_z0 + 0.02, sp_z0 + 0.05, np.linspace(0, 1, i1-i0))

    # Transport: sp → target
    i2 = int(n_steps*0.90)
    tcp_x[i1:i2] = lerp(sp_x0, TARGET_XYZ[0], np.linspace(0, 1, i2-i1))
    tcp_y[i1:i2] = lerp(sp_y0, TARGET_XYZ[1], np.linspace(0, 1, i2-i1))
    tcp_z[i1:i2] = lerp(sp_z0 + 0.05, TARGET_XYZ[2] + 0.07, np.linspace(0, 1, i2-i1))

    # Hover at target (failure: close gripper)
    tcp_x[i2:] = TARGET_XYZ[0] + rng.uniform(-0.01, 0.01, n_steps - i2)
    tcp_y[i2:] = TARGET_XYZ[1] + rng.uniform(-0.01, 0.01, n_steps - i2)
    tcp_z[i2:] = TARGET_XYZ[2] + 0.07

    # Add noise
    tcp_x += rng.normal(0, 0.003, n_steps)
    tcp_y += rng.normal(0, 0.003, n_steps)
    tcp_z += rng.normal(0, 0.002, n_steps)

    # Sponge trajectory: follows TCP during grasp+transport
    sp_x = np.full(n_steps, sp_x0)
    sp_y = np.full(n_steps, sp_y0)
    sp_z = np.full(n_steps, sp_z0)
    sp_x[i1:] = tcp_x[i1:]
    sp_y[i1:] = tcp_y[i1:]
    sp_z[i1:] = tcp_z[i1:] - 0.05

    # Gripper: stays closed throughout (failure mode)
    # ~6% random open spurts
    gripper_cmd = -0.5 * np.ones(n_steps)
    open_spurts = rng.choice(n_steps, size=int(n_steps * 0.064), replace=False)
    gripper_cmd[open_spurts] = 0.3
    gripper_open_flag = (gripper_cmd > 0).astype(float)

    # Grasped: True after grasp phase
    grasped = np.zeros(n_steps)
    grasped[i1:] = 1.0

    # Reward: stage1→stage2→stage3 hover
    reward = np.zeros(n_steps)
    reward[:i0] = 0.5 * np.linspace(0, 2, i0)          # reach reward
    reward[i0:i1] = 2.5                                  # grasp bonus
    reward[i1:i2] = 4.5 + 1.5 * np.linspace(0, 1, i2-i1)  # transport + stage2
    reward[i2:] = 5.8 + rng.normal(0, 0.1, n_steps-i2)    # hover at target

    # Dist to target
    sp_pos = np.stack([sp_x, sp_y, sp_z], axis=1)
    target = np.array(TARGET_XYZ)
    dist = np.linalg.norm(sp_pos - target, axis=1)

    return {
        "tcp_x": tcp_x, "tcp_y": tcp_y, "tcp_z": tcp_z,
        "sponge_x": sp_x, "sponge_y": sp_y, "sponge_z": sp_z,
        "gripper_cmd": gripper_cmd,
        "gripper_open_flag": gripper_open_flag,
        "grasped": grasped,
        "reward": reward,
        "sponge_to_target_dist": dist,
    }


def load_episodes(args):
    if args.demo:
        print("[plot] Using synthetic demo data (--demo flag)")
        return [generate_demo_episode(seed=i) for i in range(3)]
    elif args.npz is not None:
        print(f"[plot] Loading rollout data from: {args.npz}")
        data = np.load(args.npz, allow_pickle=True)
        # Expected keys: episodes (list of dicts)
        if "episodes" in data:
            return list(data["episodes"])
        else:
            # Flat format: one episode
            return [{k: data[k] for k in data.files}]
    else:
        print("[plot] No --npz provided. Using synthetic demo data.")
        return [generate_demo_episode(seed=i) for i in range(3)]


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_top_view(episodes, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(
        "P6v12 η — XY Trajectory (Top View)\n"
        "TCP (solid) + Sponge (dashed) | Star = Target L1.spot1",
        fontsize=12
    )
    COLORS = plt.cm.tab10.colors
    for i, ep in enumerate(episodes[:3]):
        c = COLORS[i]
        ax.plot(ep["tcp_x"], ep["tcp_y"], color=c, lw=1.4, alpha=0.9, label=f"Ep{i+1} TCP")
        ax.plot(ep["sponge_x"], ep["sponge_y"], color=c, lw=1.4, alpha=0.5, ls="--",
                label=f"Ep{i+1} Sponge")
        ax.scatter(ep["tcp_x"][0], ep["tcp_y"][0], color=c, marker="o", s=80, zorder=5)
        ax.scatter(ep["tcp_x"][-1], ep["tcp_y"][-1], color=c, marker="x", s=100, zorder=5)

    ax.scatter(TARGET_XYZ[0], TARGET_XYZ[1], color="red", marker="*", s=250, zorder=10,
               label="Target L1.spot1")
    ax.add_patch(mpatches.Circle((TARGET_XYZ[0], TARGET_XYZ[1]), ZONE_50MM,
                                  fill=False, color="red", ls="--", lw=1.5, alpha=0.6,
                                  label="50mm zone"))

    ax.text(0.02, 0.97,
            "o = start  x = end\nFailure: sponge enters zone\nbut gripper stays CLOSED",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("X (m) [forward from base]", fontsize=10)
    ax.set_ylabel("Y (m) [lateral]", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out = os.path.join(out_dir, "p6v12_trajectory_top.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {out}")


def plot_side_view(episodes, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    fig.suptitle("P6v12 η — Side View: Height + Sponge-to-Target Distance", fontsize=12)
    COLORS = plt.cm.tab10.colors

    for i, ep in enumerate(episodes[:3]):
        c = COLORS[i]
        T = np.arange(len(ep["tcp_z"]))
        axes[0].plot(T, ep["tcp_z"], color=c, lw=1.3, label=f"Ep{i+1} TCP-Z")
        axes[0].plot(T, ep["sponge_z"], color=c, lw=1.3, ls="--", alpha=0.6, label=f"Ep{i+1} Sponge-Z")

    axes[0].axhline(TARGET_XYZ[2], color="red", ls=":", lw=1.8, label=f"Target Z={TARGET_XYZ[2]*1000:.1f}mm")
    axes[0].axhline(TABLE_Z, color="gray", ls=":", lw=1.2, label=f"Table Z={TABLE_Z*1000:.1f}mm")
    axes[0].axhline(TABLE_Z + SPONGE_H, color="goldenrod", ls=":", lw=1,
                    label=f"Sponge top on table={( TABLE_Z+SPONGE_H)*1000:.1f}mm")
    axes[0].set_ylabel("Z (m)", fontsize=10)
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].grid(True, alpha=0.3)

    for i, ep in enumerate(episodes[:3]):
        c = COLORS[i]
        T = np.arange(len(ep["sponge_to_target_dist"]))
        axes[1].plot(T, ep["sponge_to_target_dist"] * 1000, color=c, lw=1.3, label=f"Ep{i+1}")

    axes[1].axhline(ZONE_50MM * 1000, color="red", ls="--", lw=1.8, label="50mm zone threshold")
    axes[1].fill_between(np.arange(len(episodes[0]["sponge_to_target_dist"])),
                          0, ZONE_50MM * 1000, alpha=0.08, color="green")
    axes[1].set_ylabel("Sponge-to-Target dist (mm)", fontsize=10)
    axes[1].set_xlabel("Step", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    out = os.path.join(out_dir, "p6v12_trajectory_side.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {out}")


def plot_failure_mode(episodes, out_dir):
    ep = episodes[0]
    T = np.arange(len(ep["reward"]))
    d = ep["sponge_to_target_dist"] * 1000

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        "P6v12 η — Failure Mode Analysis\n"
        "\"Policy grasps + transports, but never opens gripper\"\n"
        "B200 metrics: gripper_open 6.4% | is_on_target 40.6% | stage4_success 0.02%",
        fontsize=11, color="darkred", y=1.01
    )

    # Panel 1: Gripper action
    axes[0].plot(T, ep["gripper_cmd"], color="steelblue", lw=1.0, alpha=0.8)
    axes[0].axhline(0, color="black", ls="-", lw=0.5)
    axes[0].fill_between(T, 0, ep["gripper_cmd"],
                          where=np.array(ep["gripper_cmd"]) > 0,
                          alpha=0.4, color="green", label="open command")
    axes[0].fill_between(T, 0, ep["gripper_cmd"],
                          where=np.array(ep["gripper_cmd"]) < 0,
                          alpha=0.3, color="tomato", label="close command")
    axes[0].set_ylabel("Gripper action cmd\n(+open / -close)", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Flags (grasped + gripper_open)
    axes[1].fill_between(T, 0, ep["grasped"], step="post",
                          color="orange", alpha=0.6, label="grasped")
    axes[1].fill_between(T, 0, ep["gripper_open_flag"], step="post",
                          color="green", alpha=0.5, label="gripper_open (6.4%)")
    axes[1].set_ylabel("State flags", fontsize=9)
    axes[1].set_ylim(-0.05, 1.4)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Distance
    axes[2].plot(T, d, color="purple", lw=1.3)
    axes[2].axhline(ZONE_50MM * 1000, color="red", ls="--", lw=1.8, label="50mm zone")
    axes[2].fill_between(T, 0, ZONE_50MM * 1000, alpha=0.08, color="green")
    axes[2].set_ylabel("Sponge-to-Target (mm)", fontsize=9)
    axes[2].set_xlabel("Step", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    in_zone = np.where(d < ZONE_50MM * 1000)[0]
    if len(in_zone) > 0:
        first_in = in_zone[0]
        for ax in axes:
            ax.axvline(first_in, color="darkgreen", ls=":", lw=1.2, alpha=0.7)
        axes[2].text(first_in + 2, ZONE_50MM * 1000 + 5,
                     f"Enters zone\nstep {first_in}", fontsize=7.5, color="darkgreen")

    # Annotate: gripper stays closed after entering zone
    if len(in_zone) > 0:
        open_in_zone = ep["gripper_open_flag"][in_zone]
        open_rate_zone = open_in_zone.mean() * 100
        axes[1].text(0.65, 0.75,
                     f"In-zone gripper_open: {open_rate_zone:.1f}%\n(should be ~100% for release)",
                     transform=axes[1].transAxes, fontsize=8, color="darkred",
                     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out = os.path.join(out_dir, "p6v12_failure_mode_snapshot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {out}")


def plot_diagnosis_diagram(out_dir):
    """Static reward structure diagram explaining the failure mode."""
    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Left: Stage reward flow diagram
    ax_l = fig.add_subplot(gs[0])
    ax_l.set_xlim(0, 10)
    ax_l.set_ylim(0, 10)
    ax_l.axis("off")
    ax_l.set_title("P6 Reward Stage Flow\n(Root cause: stage3 hover dominates stage4 release)",
                   fontsize=10)

    stages = [
        (5.0, 9.0, "Stage 1\nReach sponge\n+0.5~2/step", "lightblue", ""),
        (5.0, 7.0, "Stage 2\nGrasp + lift\n+4/step (capped 2.0 by η)", "wheat", ""),
        (5.0, 5.0, "Stage 3\nHover near target\n+6.5/step (accumulated!)", "salmon", "← DOMINANT"),
        (5.0, 3.0, "Stage 4\nRelease (open gripper)\n+10 transient bonus (ONE TIME)", "lightgreen", "← RARELY REACHED"),
    ]
    for (x, y, txt, col, note) in stages:
        ax_l.add_patch(mpatches.FancyBboxPatch((x-2.5, y-0.7), 5, 1.4,
                                                boxstyle="round,pad=0.1",
                                                facecolor=col, edgecolor="gray", lw=1.2))
        ax_l.text(x, y, txt, ha="center", va="center", fontsize=7.5)
        if note:
            ax_l.text(x + 3.0, y, note, ha="left", va="center", fontsize=8, color="darkred",
                      fontweight="bold")
        if y > 3.0:
            ax_l.annotate("", xy=(x, y - 0.7), xytext=(x, y - 1.3),
                           arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    ax_l.text(5.0, 1.5,
              "η fix: capped stage2→2.0, added +10 transient\n"
              "BUT: 1-step advantage for close>>open remains\n"
              "→ PPO never explores release in 200-step episodes",
              ha="center", va="center", fontsize=8, color="darkred",
              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Right: Metrics comparison bar
    ax_r = fig.add_subplot(gs[1])
    labels = [
        "grasped\n(86%)",
        "is_success_zone\n50mm (54%)",
        "is_on_target\nstrict (41%)",
        "gripper_open\nrate (6.4%)",
        "stage4_success\n(0.02%)",
    ]
    values = [86.0, 54.1, 40.6, 6.4, 0.02]
    colors = ["orange", "steelblue", "cornflowerblue", "red", "tomato"]
    bars = ax_r.barh(labels, values, color=colors, edgecolor="gray", height=0.6)
    ax_r.set_xlim(0, 110)
    ax_r.axvline(100, color="gray", ls="--", lw=0.8, alpha=0.5)
    for bar, val in zip(bars, values):
        ax_r.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                  f"{val:.1f}%", va="center", fontsize=8.5)
    ax_r.set_xlabel("Rate (%)", fontsize=10)
    ax_r.set_title("P6v12 η — Key Metrics (B200, 1000 iter)\n"
                   "Policy: grasps well, transports, fails to release",
                   fontsize=10)
    ax_r.grid(axis="x", alpha=0.3)

    # Annotate gap
    ax_r.annotate("", xy=(6.4, 3), xytext=(54.1, 3),
                   arrowprops=dict(arrowstyle="<->", color="darkred", lw=1.5))
    ax_r.text(30, 3.35, "RELEASE GAP", ha="center", fontsize=8.5,
              color="darkred", fontweight="bold")

    fig.suptitle("P6v12 η — Failure Diagnosis for Lab Meeting 2026-05-13",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    out = os.path.join(out_dir, "p6v12_diagnosis_diagram.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {out}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    episodes = load_episodes(args)
    print(f"[plot] {len(episodes)} episodes loaded")

    plot_top_view(episodes, args.out_dir)
    plot_side_view(episodes, args.out_dir)
    plot_failure_mode(episodes, args.out_dir)
    plot_diagnosis_diagram(args.out_dir)

    print(f"\n[plot] DONE. All figures in: {args.out_dir}/")


if __name__ == "__main__":
    main()
