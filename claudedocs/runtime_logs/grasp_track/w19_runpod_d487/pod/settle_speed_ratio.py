"""settle 구간 실제 경과 시간(wall-clock) 요약 — W16 analyze_w16.py 와 같은 정의(sync_wall_elapsed_s 차분을 phase 별 합산). NPZ 만 읽는다."""
import json, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
m = json.loads(str(z["metadata_json"])) if "metadata_json" in z.files else None
if m is None:
    for k in z.files:
        if k.startswith("meta"):
            try: m = json.loads(str(z[k])); break
            except Exception: pass
inv = {v: k for k, v in m["phase_code"].items()}
ph = np.array([inv[int(c)] for c in z["sync_phase_code"]])
w = np.asarray(z["sync_wall_elapsed_s"], float); dw = np.diff(w, prepend=0.0)
t = np.asarray(z["sync_t_s"], float)
dt = np.diff(t, prepend=0.0)
out = {"npz": sys.argv[1], "n_sync": int(w.size), "total_wall_s": float(w[-1]), "total_sim_s": float(t[-1]), "phases": {}}
for p in dict.fromkeys(ph.tolist()):
    s = ph == p
    out["phases"][p] = {"n_sync": int(s.sum()), "wall_s": float(dw[s].sum()), "sim_s": float(dt[s].sum()),
                        "wall_per_sim_s": float(dw[s].sum() / max(dt[s].sum(), 1e-12))}
print(json.dumps(out, ensure_ascii=False, indent=1))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
