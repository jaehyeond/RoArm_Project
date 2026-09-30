"""rev36-chain 진단 그림(CPU). 판정이 아니라 눈으로 보는 확인용(D324 시각 진단).
    site  <site_map.json> <out.png>      : 명령점 가능/벽/관절 제한 지도 + 가능 립 자리 + 상자 안쪽 + 로봇 방향
    row   <row_prefix> <out.png>         : 전·후·깎인 깊이(참값·카메라) + 명령점·립·관측 구덩이 중심
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def site(src, out):
    d = json.load(open(src)); b = np.asarray(d["box_inner_m"]) * 1000
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.add_patch(plt.Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0], b[1, 1] - b[1, 0], fill=False, lw=2, color="k"))
    cat = {"feasible0": ("#2a9d8f", "가능(롤 0)"), "feasible_roll": ("#f4a261", "롤 필요"), "wall": ("#e76f51", "벽 여유 부족"), "joint": ("#6c757d", "관절 제한 위반")}
    for key, (col, lab) in cat.items():
        pts = [r["cmd_box_xy_m"] for r in d["rows"] if
               (key == "feasible0" and r["feasible"] and r.get("feasible_roll0", True)) or
               (key == "feasible_roll" and r["feasible"] and not r.get("feasible_roll0", True)) or
               (key == "wall" and not r["feasible"] and (r["reason"] or "").startswith("벽")) or
               (key == "joint" and not r["feasible"] and "관절" in (r["reason"] or ""))]
        if pts:
            p = np.asarray(pts) * 1000; ax.scatter(p[:, 0], p[:, 1], s=28, c=col, label=f"명령점: {lab} ({len(pts)})")
    lip = np.asarray([r["lip_box_xy_m"] for r in d["rows"] if r["feasible"]]) * 1000
    ax.scatter(lip[:, 0], lip[:, 1], s=6, c="k", marker="x", label="가능 명령점의 FK 립")
    ax.annotate("로봇(규약 A: 상자 −y 쪽, 베이스 y = −250 mm)", xy=(0, b[1, 0]), xytext=(-120, b[1, 0] - 28),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlabel("상자 x (mm, 31 cm 변)"); ax.set_ylabel("상자 y (mm, 22 cm 변)"); ax.set_aspect("equal")
    ax.set_xlim(b[0, 0] - 20, b[0, 1] + 20); ax.set_ylim(b[1, 0] - 45, b[1, 1] + 20)
    ax.set_title(f"셀 위치 가능 지도 · {d['step_mm']:.0f} mm 격자 · 벽 여유 ≥ {d['wall_margin_mm']} mm · 가능 {d['n_feasible']}/{d['n_grid']}"
                 + (f" (롤 0 만 {d['n_feasible_roll0']})" if "n_feasible_roll0" in d else ""))
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def row(prefix, out):
    r = json.load(open(prefix + ".json")); z = np.load(prefix + ".npz")
    o = r["frame"]["origin_xy_m"]; sh = r["frame"]["shape"]; c = 5.0
    ext = [o[0] * 1000, o[0] * 1000 + sh[1] * c, o[1] * 1000, o[1] * 1000 + sh[0] * c]
    keys = [k for k in ("truth", "cam") if f"hm_pre_{k}" in z.files]
    fig, axs = plt.subplots(len(keys), 3, figsize=(15, 4.6 * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        pre, post = z[f"hm_pre_{k}"] * 1000, z[f"hm_post_{k}"] * 1000
        for j, (img, t, cm, lim) in enumerate(((pre, "전(settle_end)", "viridis", (0, 50)), (post, "후(reclose_end, 들린 알 제외)", "viridis", (0, 50)),
                                               (pre - post, "깎인 깊이 = 전 − 후", "RdBu_r", (-25, 25)))):
            ax = axs[i, j]; im = ax.imshow(img, origin="lower", extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1])
            if k == "cam" and j < 2:
                inv = ~z[f"hm_{'pre' if j == 0 else 'post'}_cam_valid"]
                if inv.any():
                    ax.contour(inv.astype(float), levels=[0.5], origin="lower", extent=ext, colors="m", linewidths=0.6)
            plt.colorbar(im, ax=ax, fraction=0.035, label="mm")
            a = r["action"]; cr = r["crater"]
            if a.get("cmd_box_xy_m"):
                ax.plot(a["cmd_box_xy_m"][0] * 1000, a["cmd_box_xy_m"][1] * 1000, "w+", ms=12, mew=2)
            ax.plot(a["lip_box_xy_m"][0] * 1000, a["lip_box_xy_m"][1] * 1000, "wx", ms=10, mew=2)
            if cr.get("observed_center_box_xy_m"):
                ax.plot(cr["observed_center_box_xy_m"][0] * 1000, cr["observed_center_box_xy_m"][1] * 1000, "o", mfc="none", mec="orange", ms=12, mew=2)
            ax.set_title(f"{k} · {t}", fontsize=10); ax.set_xlabel("상자 x (mm)"); ax.set_ylabel("상자 y (mm)")
    lb = r["label"]
    fig.suptitle(f"{r['row_id']} · 들린 알 {lb['lifted_count']} ({lb['lifted_mass_g']:.2f} g) · 구덩이 {r['crater']['removed_volume_ml']:.1f} mL · "
                 f"최대 {r['crater']['max_depth_mm']:.1f} mm   (+ 명령점 · x 립 · ○ 관측 중심)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


if __name__ == "__main__":
    from matplotlib import font_manager as fm
    _f = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
    if Path(_f).exists():
        fm.fontManager.addfont(_f)
        plt.rcParams["font.family"] = [fm.FontProperties(fname=_f).get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    {"site": site, "row": row}[sys.argv[1]](sys.argv[2], sys.argv[3])
