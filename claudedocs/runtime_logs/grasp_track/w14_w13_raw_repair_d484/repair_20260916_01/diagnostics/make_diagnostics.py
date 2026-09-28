"""단일 프레임 진단 PNG(D324 취지). 새 물리/렌더 0. 파생 NPZ + 원자료(읽기 전용)만 사용.
1) PF0/ID8 의 7구를 XZ 측면에서 바닥·±margin 선과 함께 그린다(최소 반례).
2) 프레임별 source/ambiguous 수: 기록(rev28) vs rev29 strict, phase 전환 11개(실선)·기록 25개(점선).
3) dense phase 코드 띠 + 전환 인덱스 25 vs 11.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))
import w14_paths as W                                   # noqa: E402
from independent_check import rotate_hamilton          # noqa: E402

PH = ["initial_home", "settle", "approach", "descend", "close", "lift", "reclose", "transport",
      "discharge", "discharge_wait", "close_after_discharge", "return_home"]
tpl = W.template()
off = np.asarray(tpl["offsets_m"], float); rad = np.asarray(tpl["sphere_radii_m"], float)
dz = np.load(W.DERIVED / "w13_cycle_seed460_rev29_derived.npz", allow_pickle=False)
with np.load(W.RAW, allow_pickle=False) as z:
    p8 = z["particle_pos_m"][0, 8].astype(float); q8 = z["particle_quat_xyzw"][0, 8].astype(float)
    pc = z["sync_phase_code"].astype(int); t = z["sync_t_s"]; pft = z["particle_frame_t_s"]
    box = np.asarray(z["box_bounds_m"], float)
S = rotate_hamilton(q8[None], off)[0] + p8
m = 0.0025
report = {}

# 1) PF0/ID8 side view
fig, ax = plt.subplots(figsize=(7, 5))
for c, r in zip(S, rad):
    ax.add_patch(plt.Circle((c[0] * 1000, c[2] * 1000), r * 1000, fill=False, lw=1.5, color="tab:blue"))
ax.axhline(0, color="k", lw=1.5, label="floor z=0 (source_bounds z_lo)")
ax.axhline(+m * 1000, color="tab:green", ls="--", label="floor + margin (+2.5 mm): rev29 requires every sphere bottom above")
ax.axhline(-m * 1000, color="tab:red", ls=":", label="floor − margin (−2.5 mm): rev28 compared sphere top against this")
lo = float((S[:, 2] - rad).min()); top = float((S[:, 2] + rad).min())
ax.axhline(lo * 1000, color="tab:blue", lw=0.8, alpha=0.6); ax.axhline(top * 1000, color="tab:orange", lw=0.8, alpha=0.6)
ax.text(p8[0] * 1000 + 3.2, lo * 1000 - 0.35, f"min sphere bottom = {lo*1000:.6f} mm", fontsize=8, color="tab:blue")
ax.text(p8[0] * 1000 + 3.2, top * 1000 + 0.15, f"min sphere top = {top*1000:.6f} mm", fontsize=8, color="tab:orange")
ax.set_aspect("equal"); ax.set_xlim(p8[0] * 1000 - 4, p8[0] * 1000 + 9); ax.set_ylim(-3.2, 5.2)
ax.set_xlabel("x [mm] (world)"); ax.set_ylabel("z [mm] (world)")
ax.set_title("W13 run_01 PF0/ID8: 7-sphere clump vs source floor\n(recorded rev28 = source, rev29 strict = ambiguous)", fontsize=10)
ax.legend(fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(HERE / "pf0_id8_floor_side_view.png", dpi=150); plt.close(fig)
report["pf0_id8"] = {"min_bottom_m": lo, "min_top_m": top, "strict_bottom_pass": lo > m, "rev28_top_pass": top > -m}

# 2) per-frame counts
cr, c9 = dz["per_frame_counts_recorded"], dz["per_frame_counts_rev29"]
tr11, tr25 = dz["transition_sync_index_phase_only"], dz["transition_sync_index_recorded_rev28"]
fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, (li, name) in zip(axs, ((0, "source"), (5, "ambiguous"))):
    ax.plot(pft, cr[:, li], label=f"recorded rev28 {name}", color="tab:red")
    ax.plot(pft, c9[:, li], label=f"rev29 strict {name}", color="tab:green")
    for i in tr25: ax.axvline(t[i], color="gray", ls=":", lw=0.6)
    for i in tr11: ax.axvline(t[i], color="k", ls="-", lw=0.8)
    ax.set_ylabel(f"{name} count"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
for i in tr11:
    axs[0].text(t[i], cr[:, 0].max(), PH[pc[i]], rotation=90, fontsize=6, va="top")
axs[1].set_xlabel("sim time [s]  (solid = 11 phase-only transitions, dotted = 25 recorded rev28 entries)")
axs[0].set_title("W13 run_01 inventory per particle frame: recorded (rev28 floor rule) vs rev29 strict floor rule")
fig.tight_layout(); fig.savefig(HERE / "inventory_counts_timeline.png", dpi=150); plt.close(fig)

# 3) phase strip
fig, ax = plt.subplots(figsize=(11, 2.8))
ax.step(np.arange(len(pc)), pc, where="post", color="tab:blue", lw=1)
ax.plot(tr25, pc[tr25] + 0.35, "v", color="gray", ms=5, label="recorded transition_sync_index (25)")
ax.plot(tr11, pc[tr11] - 0.35, "^", color="k", ms=5, label="rev29 phase-only (11)")
ax.set_yticks(range(12)); ax.set_yticklabels(PH, fontsize=6); ax.set_xlabel("dense sync row"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("sync_phase_code with transition indices: 25 recorded (row0 + subphase entries) vs 11 phase changes")
fig.tight_layout(); fig.savefig(HERE / "phase_transitions_strip.png", dpi=150); plt.close(fig)
report["counts"] = {"frames": int(len(pft)), "recorded_final": cr[-1].tolist(), "rev29_final": c9[-1].tolist(),
                    "max_source_drop_recorded_minus_rev29": int((cr[:, 0] - c9[:, 0]).max()),
                    "min_source_drop": int((cr[:, 0] - c9[:, 0]).min())}
report["transitions"] = {"n25": int(len(tr25)), "n11": int(len(tr11))}
report["files"] = [str(HERE / f) for f in ("pf0_id8_floor_side_view.png", "inventory_counts_timeline.png", "phase_transitions_strip.png")]
(HERE / "DIAGNOSTICS.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
