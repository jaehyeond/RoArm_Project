"""W25-A 그림 2장 (CPU, matplotlib, 원자료 = CPU 스텁 드라이런 NPZ/JSON · 물리 0).

① fig1_topview_robot_frame.png — 로봇 좌표(어깨축 원점, x 앞·y 왼쪽) 위에서 본 배치: 로봇 베이스축·상자 안쪽
   (rev34 선언 301×198 + 실제 드라이런 트레이)·컵·전 sync 립 궤적·취점·상자 좌표축 화살표. rev32 배치는 회색.
② fig2_schedule_door_lip.png — 문 관절각(서보 환산 오른쪽 축)·립 높이(펠릿면 기준) vs 물리시간, rev32 vs rev34.
usage: python figures_w25.py <rev32 run> <rev34 run> <out dir> [params_w25.json]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402

PHASES = ["initial_home", "settle", "approach", "descend", "close", "lift", "reclose",
          "transport", "discharge", "discharge_wait", "close_after_discharge", "return_home"]


def load(run):
    run = Path(run)
    res = json.load(open(run / "w13_cycle_seed460.json"))
    z = np.load(run / "w13_cycle_seed460.npz", allow_pickle=True)
    w = res.get("w25") or {}
    R = np.asarray((w.get("frame") or {}).get("R_robot_box") or np.eye(3), float)
    t = np.asarray(res["frames"]["adapter"]["t_robot_m"], float)
    return res, z, R, t


def to_robot(p, R, t):
    return (R @ np.asarray(p, float).T).T + t


def rect(ax, box, R, t, **kw):
    c = np.array([[box[0, 0], box[1, 0], 0], [box[0, 1], box[1, 0], 0], [box[0, 1], box[1, 1], 0],
                  [box[0, 0], box[1, 1], 0], [box[0, 0], box[1, 0], 0]])
    r = to_robot(c, R, t) * 1000
    ax.plot(r[:, 0], r[:, 1], **kw)


def fig1(r32, r34, out, P25):
    fig, ax = plt.subplots(figsize=(10, 9))
    for (res, z, R, t), col, lab in ((r32, "0.6", "rev32"), (r34, "tab:blue", "rev34")):
        box = np.asarray(z["box_bounds_m"], float)
        rect(ax, box, R, t, c=col, lw=2.2 if lab == "rev34" else 1.4,
             label=f"{lab} tray inner (run) {abs(box[0,1]-box[0,0])*1000:.0f}x{abs(box[1,1]-box[1,0])*1000:.0f} mm (box x x y)")
        tp = to_robot(np.asarray(z["tool_pos_m"], float), R, t) * 1000
        ax.plot(tp[:, 0], tp[:, 1], "-", c=col, lw=0.9, alpha=0.9, label=f"{lab} lip path (all syncs)")
        bc = np.asarray(res["fixtures"]["bin"]["center_xy_m"] + [0.0], float)
        bcr = to_robot(bc, R, t) * 1000
        ax.add_patch(plt.Circle(bcr[:2], res["fixtures"]["bin"]["inner_r_m"] * 1000, fill=False, ec=col, lw=1.8,
                                ls="--" if lab == "rev32" else "-"))
        ax.annotate(f"{lab} cup", bcr[:2], color=col, fontsize=8, ha="center", va="center")
        site = (res.get("w25") or {}).get("scoop_site_w11_xy_m") or [0.0, 0.0]
        sr = to_robot([site[0], site[1], 0.0], R, t) * 1000
        ax.plot(*sr[:2], "x", c=col, ms=11, mew=2.5, label=f"{lab} scoop lip site")
        cen = to_robot([0.0, 0.0, 0.0], R, t) * 1000
        ax.plot(*cen[:2], "+", c=col, ms=12, mew=1.5)
        if lab == "rev34":
            for k, nm, cc in ((0, "x_box", "tab:red"), (1, "y_box", "tab:green")):
                d = R[:2, k] * 70
                ax.annotate("", xy=cen[:2] + d, xytext=cen[:2], arrowprops=dict(arrowstyle="->", color=cc, lw=2))
                ax.text(*(cen[:2] + d * 1.18), nm, color=cc, fontsize=10, ha="center", va="center", weight="bold")
            if P25:
                L = np.asarray(P25["tray_inner_mm"], float) / 1000.0
                dbox = np.array([[-L[0] / 2, L[0] / 2], [-L[1] / 2, L[1] / 2], [0, 0]])
                rect(ax, dbox, R, t, c="tab:purple", lw=1.6, ls=":",
                     label=f"declared NTC106 inner {L[0]*1000:.0f}x{L[1]*1000:.0f} mm")
    ax.plot(0, 0, "k^", ms=12, label="robot base rotation axis (0,0)")
    ax.axhline(0, c="k", lw=0.4)
    ax.axvline(0, c="k", lw=0.4)
    ax.set_xlabel("robot x (forward) [mm]")
    ax.set_ylabel("robot y (left) [mm]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Top view in ROBOT frame (shoulder-axis origin). CPU stub dry-run geometry, no physics.\n"
                 "rev34: box long side (x_box, 301 mm) along robot y; box center at robot (250, 0) mm")
    fig.tight_layout()
    fig.savefig(out / "fig1_topview_robot_frame.png", dpi=130)
    plt.close(fig)


def fig2(r32, r34, out, runs_label):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8.5), gridspec_kw={"width_ratios": [1.35, 1]})
    for (res, z, R, t), col, lab in ((r32, "0.45", "rev32"), (r34, "tab:blue", "rev34")):
        tt = np.asarray(z["sync_t_s"], float)
        qa = np.asarray(z["door_actual_deg"], float)
        zs = float(res["scoop_site"]["surface_z_pre_settle_mm"])
        lz = np.asarray(z["tool_pos_m"], float)[:, 2] * 1000 - zs
        dec = {d["tag"]: d["sim_t"] for d in res["decisions"]}
        t0 = dec.get("approach_end", 0.0) - 0.8
        t1 = dec.get("reclose_end", tt[-1]) + 0.6
        for col_i, (lo, hi) in enumerate(((tt[0], tt[-1]), (t0, t1))):
            m = (tt >= lo) & (tt <= hi)
            axs[0, col_i].plot(tt[m], qa[m], c=col, lw=1.4, label=f"{lab} door joint (actual)")
            axs[1, col_i].plot(tt[m], lz[m], c=col, lw=1.4, label=f"{lab} lip z - pellet surface")
            for tag in ("approach_end", "descend_end", "close_stop", "lift_end", "reclose_end"):
                if tag in dec and lo <= dec[tag] <= hi and col_i == 1:
                    axs[0, 1].axvline(dec[tag], c=col, lw=0.6, ls=":")
                    axs[0, 1].text(dec[tag], 29 if lab == "rev34" else 31, tag.replace("_", " "), rotation=90,
                                   fontsize=6, color=col, va="top")
    for ax in axs[0]:
        ax.axhline(3.6 - 2.5, c="tab:red", lw=0.8, ls="--", label="chatter threshold servo 3.6 = joint 1.1")
        ax.set_ylabel("door joint angle [deg] (servo = joint + 2.5)")
        sec = ax.secondary_yaxis("right", functions=(lambda x: x + 2.5, lambda x: x - 2.5))
        sec.set_ylabel("servo [deg]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axs[1]:
        ax.axhline(0, c="tab:brown", lw=0.8, ls="-.", label="pellet surface (pre-settle, scoop site)")
        ax.axhline(-25, c="k", lw=0.6, ls=":", label="plunge -25 mm")
        ax.axhline(80, c="tab:green", lw=0.6, ls=":", label="real lift: surface +80 mm")
        ax.set_ylabel("lip z - surface [mm]")
        ax.set_xlabel("sim time [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axs[0, 0].set_title("full cycle")
    axs[0, 1].set_title("zoom: approach_end-0.8 s .. reclose_end+0.6 s")
    fig.suptitle(f"Procedure schedule rev32 vs rev34 — CPU kinematic stub (NOT DEME physics). {runs_label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig2_schedule_door_lip.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    P25 = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
    a, b = load(sys.argv[1]), load(sys.argv[2])
    fig1(a, b, out, P25)
    fig2(a, b, out, f"A={Path(sys.argv[1]).name} B={Path(sys.argv[2]).name}")
    print("saved", out)
