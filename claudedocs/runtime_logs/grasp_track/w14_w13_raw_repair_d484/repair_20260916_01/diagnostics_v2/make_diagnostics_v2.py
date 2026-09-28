"""v2 진단: 기록(rev28)/rev29 strict/rev30 support 의 프레임별 source·ambiguous·receiving_bin 수와 phase 전환. 새 물리 0."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
d = np.load(HERE.parent / "derived_v2" / "w13_cycle_seed460_rev30_derived_v2.npz", allow_pickle=False)
t = d["particle_frame_t_s"]; cr, c9, c0 = d["per_frame_counts_recorded"], d["per_frame_counts_rev29"], d["per_frame_counts_rev30"]
tr = d["transition_sync_index_phase_only"]
with np.load("/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/run_01/w13_cycle_seed460.npz", allow_pickle=False) as z:
    ts = z["sync_t_s"]; pc = z["sync_phase_code"].astype(int)
PH = ["initial_home","settle","approach","descend","close","lift","reclose","transport","discharge","discharge_wait","close_after_discharge","return_home"]
fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for ax, (li, name) in zip(axs, ((0, "source"), (5, "ambiguous"), (1, "receiving_bin"))):
    ax.plot(t, cr[:, li], color="tab:red", label=f"recorded rev28 {name}")
    ax.plot(t, c9[:, li], color="tab:green", label=f"rev29 strict {name}")
    ax.plot(t, c0[:, li], color="tab:blue", ls="--", label=f"rev30 support-floor (ERRATUM_04) {name}")
    for i in tr: ax.axvline(ts[i], color="k", lw=0.7)
    ax.set_ylabel(name); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
for i in tr: axs[0].text(ts[i], cr[:, 0].max(), PH[pc[i]], rotation=90, fontsize=6, va="top")
axs[2].set_xlabel("sim time [s]; vertical lines = 11 phase transitions")
axs[0].set_title("W13 run_01 inventory per frame: recorded vs rev29 strict vs rev30 support-floor")
fig.tight_layout(); fig.savefig(HERE / "inventory_counts_v2.png", dpi=150); plt.close(fig)
rb = c0[:, 1]; nz = np.flatnonzero(rb)
rep = {"files": [str(HERE / "inventory_counts_v2.png")], "receiving_bin_first_frame": int(nz[0]) if len(nz) else None,
       "receiving_bin_first_t_s": float(t[nz[0]]) if len(nz) else None, "receiving_bin_max": int(rb.max()),
       "final": {"recorded": cr[-1].tolist(), "rev29": c9[-1].tolist(), "rev30": c0[-1].tolist()},
       "source_rev30_minus_recorded_range": [int((c0[:, 0] - cr[:, 0]).min()), int((c0[:, 0] - cr[:, 0]).max())]}
(HERE / "DIAGNOSTICS_V2.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
