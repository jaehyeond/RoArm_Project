"""W26 셀 후처리(로컬 CPU, 원자료 읽기만). 셀 run 의 lift_end 기하 라벨·구덩이 모양을 W25 전체 사이클 두 run(podA/podB) lift_end 와
비교해 criteria_w26_cell.json 의 사전등록 관문(G_label·G_crater·G_run)을 판정하고, 단계별 실제 경과 시간을 낸다.
사용: python cell_post.py <cell_run_dir> <out_json>"""
import json, sys, numpy as np
from pathlib import Path
REF = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/exec_rev34_paperbox_20260929/runs")
CRIT = Path(__file__).resolve().parent.parent / "criteria_w26_cell.json"

def load(run):
    z = np.load(Path(run) / "w13_cycle_seed460.npz")
    tags = [str(x) for x in z["decision_tags"]]; fidx = dict(zip(tags, z["decision_particle_frame_index"].tolist()))
    return z, fidx

def lifted(z, fidx):
    P, C = z["particle_pos_m"], z["inventory_code"]
    fs, fl = fidx["settle_end"], fidx["lift_end"]
    top = float(P[fs][C[fs] == 0][:, 2].max())
    thr = top + 0.015
    n = int((P[fl][:, 2] > thr).sum())
    fr = fidx.get("reclose_end"); n_rc = int((P[fr][:, 2] > thr).sum()) if fr is not None else None
    return n, n_rc, top, thr

def hmap(z, f):
    P, C, box = z["particle_pos_m"], z["inventory_code"], z["box_bounds_m"]
    cell = 0.005; ny = int(np.ceil((box[1, 1] - box[1, 0]) / cell)); nx = int(np.ceil((box[0, 1] - box[0, 0]) / cell))
    q = P[f][C[f] == 0].astype(np.float64)
    ix = np.clip(((q[:, 0] - box[0, 0]) / cell).astype(int), 0, nx - 1); iy = np.clip(((q[:, 1] - box[1, 0]) / cell).astype(int), 0, ny - 1)
    H = np.full(ny * nx, -np.inf); np.maximum.at(H, iy * nx + ix, q[:, 2] + 0.00125); H[np.isinf(H)] = np.nan
    return H.reshape(ny, nx) * 1000

def phases(z):
    meta = json.loads(str(z["metadata_json"])); inv = {v: k for k, v in meta["phase_code"].items()}
    ph = z["sync_phase_code"]; w = np.diff(z["sync_wall_elapsed_s"], prepend=0.0); t = np.diff(z["sync_t_s"], prepend=0.0)
    return {inv[int(c)]: {"wall_s": round(float(w[ph == c].sum()), 1), "sim_s": round(float(t[ph == c].sum()), 4)} for c in sorted(set(ph.tolist()))}

def main(run, out):
    crit = json.load(open(CRIT))["w26_cell_gates"]
    zc, fc = load(run); za, fa = load(REF / "podA_4090/run_01"); zb, fb = load(REF / "podB_pro6000x2/run_01")
    la, lb, lc = lifted(za, fa), lifted(zb, fb), lifted(zc, fc)
    H0 = hmap(zb, fb["initial_home_end"]); Ha = hmap(za, fa["lift_end"]); Hb = hmap(zb, fb["lift_end"]); Hc = hmap(zc, fc["lift_end"])
    crater = (H0 - Hb) > 3.0; ref = (Ha + Hb) / 2
    d = (Hc - ref)[crater]; d = d[np.isfinite(d)]; rms = float(np.sqrt((d ** 2).mean()))
    dab = (Ha - Hb)[crater]; dab = dab[np.isfinite(dab)]
    rj = json.load(open(Path(run) / "w13_cycle_seed460.json")); ex = json.load(open(Path(run) / "EXECUTION_RECEIPT.json")) if (Path(run) / "EXECUTION_RECEIPT.json").exists() else {}
    g = crit["G_label"]; gl = g["low"] <= lc[0] <= g["high"]
    gc = rms <= crit["G_crater"]["value_mm"]
    gr = (rj.get("stopped_early_after_phase") == "reclose") and (rj.get("diverged") is False) and (rj.get("abort_class") is None) and (ex.get("rc", 0) == 0)
    res = {"artifact": "W26_CELL_POST_V1", "cell_run": str(run),
           "label_rule": crit["label_rule"],
           "lifted_lift_end": {"cell": lc[0], "podA_ref": la[0], "podB_ref": lb[0]},
           "lifted_reclose_end": {"cell": lc[1], "podA_ref": la[1], "podB_ref": lb[1]},
           "settle_top_z_m": {"cell": lc[2], "podA": la[2], "podB": lb[2]},
           "crater_cells": int(crater.sum()), "crater_rms_mm_cell_vs_refmean": round(rms, 3),
           "crater_rms_mm_A_vs_B": round(float(np.sqrt((dab ** 2).mean())), 3),
           "gates": {"G_label": gl, "G_crater": gc, "G_run": gr, "all_pass": bool(gl and gc and gr)},
           "run": {"stopped_early_after_phase": rj.get("stopped_early_after_phase"), "diverged": rj.get("diverged"), "abort_class": rj.get("abort_class"),
                   "runner_wall_s": ex.get("wall_s"), "sim_wall_seconds": rj.get("wall_seconds"), "w26_cell": rj.get("w26_cell")},
           "phase_wall": phases(zc), "chatter_log": (rj.get("w25") or {}).get("procedure", {}).get("chatter_log")}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("lifted_lift_end", "crater_rms_mm_cell_vs_refmean", "crater_rms_mm_A_vs_B", "gates")}, ensure_ascii=False))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
