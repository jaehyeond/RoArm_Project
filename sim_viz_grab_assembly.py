#!/usr/bin/env python3
"""g18 그랩 조립 3D 시각 검수 — 실제 삼각형 메쉬(AABB 아님), link5 + 순정 가동 조 + 브래킷 + 링크 + 셸.

닫힘(서보 0°, 셸 0°)과 완전개방(서보 89°, 셸 44.5°)을 같은 카메라로, 4방향(등각·앞·옆·위).
순정 가동 조는 서보각만큼 실제로 돌려 놓는다(p37 G9 와 같은 변환). 링크는 linkage_pose 로 자세를 맞춘다.
사용: python sim_viz_grab_assembly.py <out_dir>   → assembly_closed.png / assembly_open.png / assembly_4view.png
"""
import sys, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = Path(__file__).resolve().parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "claudedocs/runtime_logs/grab_track/g18_nut_trap/viz")
sys.path.insert(0, str(REPO / "sim_scripts")); _a, sys.argv = sys.argv, ["x"]
import p37_g2_grab_v1_attach_probe as p
sys.argv = _a
G, P, K = p.G, p.P, p.K
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]; matplotlib.rcParams["axes.unicode_minus"] = False

T = p.placement(); R = T[:3, :3]
link5 = p.load_mm("link5.stl"); jaw0 = p.jaw_in_link5()
sL, nL = G.build_shell(P, -1); sR, nR = G.build_shell(P, +1)
br, nB = G.build_bracket(P); dr, nD, lk = G.build_linkage(P)
hinge_w = R @ np.array([0.0, 0.0, 1.0])
rows = lk["rows"]


def state(servo_deg):
    i = int(np.argmin([abs(r["servo_deg"] - servo_deg) for r in rows])); row = rows[i]
    Ts, Tr, Tk = G.linkage_pose(P, lk, i); Tsw = {"servocrank": Ts, "rod": Tr, "shellcrank": Tk}
    out = []                                             # (mesh(link5 frame), color, alpha)
    for m0, nm in zip(br, nB):
        m = m0.copy(); m.apply_transform(T); out.append((m, "#3a9d3a", 0.85))
    for m0, nm in zip(dr, nD):
        m = m0.copy(); m.apply_transform(T @ Tsw[G.linkage_group(nm)])
        out.append((m, "#e07b00" if G.linkage_group(nm) == "servocrank" else "#b05a00", 0.9))
    for parts, side, col in ((sL, -1, "#1f77b4"), (sR, +1, "#7fb3d5")):
        piv = R @ np.array([side * K["g"] / 2.0, 0.0, 0.0]) + T[:3, 3]
        Rj = p.rot_about(piv, hinge_w, math.radians(side * row["shell_deg"]))
        for m0 in parts:
            m = m0.copy(); m.apply_transform(Rj @ T); out.append((m, col, 0.55))
    jw = jaw0.copy(); jw.apply_transform(p.rot_about(p.GRIPPER_ORIGIN, p.GRIPPER_AXIS, math.radians(row["servo_deg"])))
    out.append((jw, "#555555", 0.35))
    return out, row


def add_mesh(ax, m, col, alpha, stride=1):
    f = m.faces[::stride]; v = m.vertices
    pc = Poly3DCollection(v[f], facecolor=col, edgecolor="none", alpha=alpha, linewidths=0)
    ax.add_collection3d(pc)


def draw(ax, items, elev, azim, title, lim=((-45, 45), (-45, 45), (55, 150))):
    l5 = link5.copy(); keep = l5.vertices[l5.faces].mean(1)[:, 2] > 40      # 손목~블레이드만
    l5.update_faces(keep)
    add_mesh(ax, l5, "#222222", 0.18, stride=2)
    for m, col, al in items:
        add_mesh(ax, m, col, al)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_box_aspect([lim[0][1] - lim[0][0], lim[1][1] - lim[1][0], lim[2][1] - lim[2][0]])
    ax.view_init(elev=elev, azim=azim); ax.set_title(title, fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    closed, r0 = state(0.0); opened, r1 = state(89.0)
    # 4방향 × 2상태
    fig = plt.figure(figsize=(22, 11))
    views = [(28, -60, "등각"), (0, -90, "앞(−Y→)"), (0, 0, "옆(+X→)"), (90, -90, "위(+Z→)")]
    for k, (items, row, tag) in enumerate(((closed, r0, "닫힘"), (opened, r1, "완전개방"))):
        for j, (el, az, vn) in enumerate(views):
            ax = fig.add_subplot(2, 4, k * 4 + j + 1, projection="3d")
            draw(ax, items, el, az, f"{tag} 서보 {row['servo_deg']:.0f}° 셸 {row['shell_deg']:.1f}° 입 {row['mouth_mm']:.1f} mm — {vn}")
    fig.suptitle("g18_nut_trap 조립 (link5 프레임, mm): 검정 link5 · 회색 순정 가동 조 · 초록 브래킷 · 주황 링크 · 파랑 셸 L/R", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "assembly_4view.png", dpi=110)
    # 근접: 체결부 (등각, 좁은 범위) 닫힘/개방
    fig2 = plt.figure(figsize=(16, 8))
    for k, (items, row, tag) in enumerate(((closed, r0, "닫힘"), (opened, r1, "완전개방"))):
        ax = fig2.add_subplot(1, 2, k + 1, projection="3d")
        draw(ax, items, 22, -135, f"체결부 근접 — {tag}", lim=((-35, 15), (-30, 25), (70, 130)))
    fig2.suptitle("브래킷 3점(초록 판·레일 슬롯)·크랭크판(주황)·순정 조(회색)·link5 플랜지(검정) 근접"); fig2.tight_layout()
    fig2.savefig(OUT / "assembly_fastening_closeup.png", dpi=120)
    print("saved:", OUT / "assembly_4view.png", OUT / "assembly_fastening_closeup.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
