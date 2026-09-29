#!/usr/bin/env python3
"""D324 시각 진단 — 단면·히스토그램 PNG. 생성 성공은 검수가 아니다(검수는 inspection.json).
측정값은 DIAGNOSTIC_MEASUREMENTS.json 에 같이 남겨 그림과 수치를 대조할 수 있게 한다."""
import argparse, json, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]
COL = ["#8c8c8c", "#1f77b4", "#d62728", "#9467bd", "#2ca02c", "#ff7f0e"]


def jdef(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(type(o)))


def main():
    ap = argparse.ArgumentParser()
    for k in ("raw", "meta", "derived", "outdir"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    res = json.load(open(a.meta)); z = np.load(a.raw, allow_pickle=False)
    d = np.load(a.derived, allow_pickle=False)
    meta = json.loads(str(z["metadata_json"])); P = res["params"]
    bf = res["fixtures"]["bin"]
    rec = np.asarray(z["inventory_code"]); F, N = rec.shape
    per = np.asarray(d["per_frame_counts_recorded"])
    per34 = np.asarray(d["per_frame_counts_rev34"])
    pft = np.asarray(z["particle_frame_t_s"], float)
    pfs = np.asarray(z["particle_frame_sync_index"]).astype(int)
    pcode = np.asarray(z["sync_phase_code"]).astype(int)
    phases = meta["phase_order"]
    door = np.asarray(z["door_actual_deg"], float)
    st = np.asarray(z["sync_t_s"], float)
    fin = np.asarray(z["final_positions_m"], float)
    tags = [str(t) for t in z["decision_tags"]]
    dpf = np.asarray(z["decision_particle_frame_index"]).astype(int)
    dsi = np.asarray(z["decision_sync_index"]).astype(int)
    M = {}

    # 1. 최종 용기 단면 (x-z), 용기 안벽/테두리 선 포함
    bc = np.asarray(bf["center_xy_m"], float)
    sel = np.hypot(fin[:, 0] - bc[0], fin[:, 1] - bc[1]) < 0.12
    f = plt.figure(figsize=(9, 6)); ax = f.add_subplot(111)
    for i, lab in enumerate(LABELS):
        m = sel & (rec[-1] == i)
        if m.any():
            ax.scatter(fin[m, 0] - bc[0], fin[m, 2], s=3, c=COL[i], label=f"{lab} ({int(m.sum())})", alpha=.6)
    apo = float(bf["apothem_m"]); zf = float(bf["floor_inner_z_m"]); zr = float(bf["rim_z_m"])
    for x in (-apo, apo):
        ax.axvline(x, color="k", lw=1.2, ls="--")
    ax.axhline(zf, color="k", lw=1.2, ls="--"); ax.axhline(zr, color="k", lw=1.2, ls=":")
    ax.set_xlabel("x - bin_center_x [m]"); ax.set_ylabel("z [m]")
    ax.set_title(f"01 final frame, bin neighbourhood section (apothem={apo:.5f} m, "
                 f"floor={zf} m, rim={zr} m)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.3)
    f.tight_layout(); f.savefig(out / "01_final_bin_section.png", dpi=130); plt.close(f)
    M["01_final_bin_section"] = {"n_within_0p12m_of_bin": int(sel.sum()),
                                 "counts_by_label": {LABELS[i]: int((sel & (rec[-1] == i)).sum())
                                                     for i in range(6)},
                                 "apothem_m": apo, "floor_inner_z_m": zf, "rim_z_m": zr,
                                 "max_z_of_bin_labelled_m":
                                     float(fin[rec[-1] == 1, 2].max()) if (rec[-1] == 1).any() else None}

    # 2. 프레임별 재고 곡선 + 결정 태그 표시
    f = plt.figure(figsize=(11, 6)); ax = f.add_subplot(111)
    for i, lab in enumerate(LABELS):
        if per[:, i].max() > 0:
            ax.plot(pft, per[:, i], color=COL[i], label=lab, lw=1.4)
    for i, t in enumerate(tags):
        ax.axvline(pft[dpf[i]], color="k", lw=.5, alpha=.4)
        ax.text(pft[dpf[i]], ax.get_ylim()[1] * .98, t, rotation=90, fontsize=6, va="top", ha="right")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("particle frame t [s]"); ax.set_ylabel("count (symlog)")
    ax.set_title("02 recorded inventory per saved particle frame (275 frames) with decision tags")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    f.tight_layout(); f.savefig(out / "02_inventory_per_frame.png", dpi=130); plt.close(f)
    M["02_inventory_per_frame"] = {"F": int(F),
                                   "max_receiving_bin": int(per[:, 1].max()),
                                   "frame_of_max_receiving_bin": int(per[:, 1].argmax()),
                                   "recorded_equals_rev34_per_frame": bool(np.array_equal(per, per34)),
                                   "final": {LABELS[i]: int(per[-1, i]) for i in range(6)}}

    # 3. 정착 창 속도 히스토그램 (창 안 receiving_bin 라벨 입자)
    win = float(P["settlement_window_s"])
    idx = [i for i, v in enumerate(pft) if float(v) >= float(pft[-1]) - win - 1e-9]
    V = np.asarray(z["particle_vel_m_s"])[idx].astype(float)
    Pw = np.asarray(z["particle_pos_m"])[idx].astype(float)
    stable = (rec[idx] == 1).all(0)
    spd = np.linalg.norm(V, axis=2).max(0)
    mov = np.linalg.norm(Pw - Pw[0], axis=2).max(0)
    f = plt.figure(figsize=(11, 4.6))
    ax = f.add_subplot(121)
    ax.hist(spd[stable] * 1000, bins=60, color="#1f77b4")
    ax.axvline(float(P["settle_speed_max_m_s"]) * 1000, color="r", ls="--",
               label=f"criterion {P['settle_speed_max_m_s']*1000} mm/s")
    ax.set_xlabel("max |v| over window [mm/s]"); ax.set_ylabel("particles"); ax.legend(fontsize=8)
    ax.set_title(f"03a window max speed, stable_bin n={int(stable.sum())}")
    ax = f.add_subplot(122)
    ax.hist(mov[stable] * 1000, bins=60, color="#2ca02c")
    ax.axvline(float(P["settle_move_max_m"]) * 1000, color="r", ls="--",
               label=f"criterion {P['settle_move_max_m']*1000} mm")
    ax.set_xlabel("max centre move over window [mm]"); ax.legend(fontsize=8)
    ax.set_title(f"03b window centre displacement (n_frames={len(idx)}, "
                 f"max gap {max(np.diff(pft[idx])) if len(idx)>1 else 0:.6f} s)")
    f.tight_layout(); f.savefig(out / "03_settlement_window.png", dpi=130); plt.close(f)
    M["03_settlement_window"] = {"n_window_frames": len(idx), "window_rows": [int(i) for i in idx],
                                 "window_times_s": [float(pft[i]) for i in idx],
                                 "max_gap_s": float(max(np.diff(pft[idx]))) if len(idx) > 1 else 0.0,
                                 "cadence_contract_min_frames": 6, "cadence_contract_max_gap_s": 0.05,
                                 "n_stable_bin": int(stable.sum()),
                                 "max_speed_of_stable_mm_s": float(spd[stable].max() * 1000),
                                 "max_move_of_stable_mm": float(mov[stable].max() * 1000)}

    # 4. 문 관절각 전체 + 채터링 구간 확대
    f = plt.figure(figsize=(11, 6))
    ax = f.add_subplot(211)
    ax.plot(st, door, lw=.7, color="#333")
    for s in res["door"]["stops"]:
        ax.plot(st[s["sync_index"]], door[s["sync_index"]], "rv", ms=4)
    ax.set_ylabel("door joint [deg]"); ax.grid(alpha=.3)
    ax.set_title("04 door joint angle over executed syncs (red = recorded stops)")
    ax2 = f.add_subplot(212)
    lo, hi = 6500, 13200
    ax2.plot(st[lo:hi], door[lo:hi], lw=.8, color="#333")
    ax2.axhline(float(P["w25_chatter_threshold_servo_deg"]) - float(P["servo_zero_offset_deg"]),
                color="b", ls="--", lw=1,
                label=f"chatter threshold joint {P['w25_chatter_threshold_servo_deg']}-"
                      f"{P['servo_zero_offset_deg']} deg")
    for s in res["door"]["stops"]:
        if lo <= s["sync_index"] < hi:
            ax2.plot(st[s["sync_index"]], door[s["sync_index"]], "rv", ms=6)
            ax2.text(st[s["sync_index"]], door[s["sync_index"]] + .12, s["subphase"], fontsize=6,
                     rotation=90, ha="center")
    ax2.set_xlabel("sim t [s]"); ax2.set_ylabel("door joint [deg]"); ax2.grid(alpha=.3)
    ax2.legend(fontsize=8)
    ax2.set_title("04b close + chatter window (sync 6500..13200)")
    f.tight_layout(); f.savefig(out / "04_door_angle_and_chatter.png", dpi=130); plt.close(f)
    M["04_door_angle_and_chatter"] = {"n_stops": len(res["door"]["stops"]),
                                      "door_min_deg": float(door.min()), "door_max_deg": float(door.max()),
                                      "close_stop_deg": float(door[7074]),
                                      "reclose_deg": float(door[13051]),
                                      "final_deg": float(door[-1]),
                                      "chatter_threshold_joint_deg":
                                          float(P["w25_chatter_threshold_servo_deg"]) -
                                          float(P["servo_zero_offset_deg"])}

    # 5. 흘림 입자 — 최초 spill 프레임 히스토그램 + 최종 z
    ever = np.flatnonzero((rec == 3).any(0))
    first = np.array([int(np.argmax(rec[:, i] == 3)) for i in ever])
    f = plt.figure(figsize=(11, 4.6))
    ax = f.add_subplot(121)
    ax.hist(pft[first], bins=30, color="#9467bd")
    for i, t in enumerate(tags):
        ax.axvline(pft[dpf[i]], color="k", lw=.5, alpha=.35)
    ax.set_xlabel("t of first spill label [s]"); ax.set_ylabel("particles")
    ax.set_title(f"05a first-spill time, n={len(ever)}")
    ax = f.add_subplot(122)
    ax.scatter(fin[ever, 0], fin[ever, 2] * 1000, s=14, c="#9467bd")
    ax.axhline(float(meta["spill_rest_z_m"]) * 1000, color="r", ls="--",
               label=f"spill_rest_z {meta['spill_rest_z_m']*1000} mm")
    ax.set_xlabel("final x [m]"); ax.set_ylabel("final z [mm]"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("05b spill cohort final position")
    f.tight_layout(); f.savefig(out / "05_spill_cohort.png", dpi=130); plt.close(f)
    M["05_spill_cohort"] = {"n_ever_spill": int(len(ever)),
                            "first_spill_frame_min": int(first.min()), "first_spill_frame_max": int(first.max()),
                            "first_spill_t_min_s": float(pft[first.min()]),
                            "first_spill_t_max_s": float(pft[first.max()]),
                            "phases": sorted({phases[int(pcode[int(pfs[fr])])] for fr in first}),
                            "final_z_mm_min": float(fin[ever, 2].min() * 1000),
                            "final_z_mm_max": float(fin[ever, 2].max() * 1000),
                            "spill_rest_z_mm": float(meta["spill_rest_z_m"]) * 1000}

    M["_meta"] = {"artifact": "W25_DIAGNOSTIC_MEASUREMENTS_V1",
                  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "wall_s": round(time.time() - t0, 3),
                  "note": "PNG 생성 성공은 검수가 아니다 — 실제로 열어 본 관찰은 inspection.json 에 적는다."}
    (out / "DIAGNOSTIC_MEASUREMENTS.json").write_text(json.dumps(M, ensure_ascii=False, indent=1, default=jdef))
    print(json.dumps(M, ensure_ascii=False, indent=1, default=jdef))


if __name__ == "__main__":
    main()
