#!/usr/bin/env python3
"""p32 — y3_d455 교차-arm 요약 + 대조 (prereg 37c11131 §7).

권위 = 각 arm의 {arm}_results.json (본 스크립트는 파생 요약일 뿐).
대조 3종: (i) 관측의 가치 = a5(blind) vs obs-gated 총 동작수,
(ii) release 브리지 = a8(y2 release) vs a1(v2) — pick 수열 동일성 검증 후 분산 대조,
(iii) 마스크 효과 = a7(stack_masked) vs y2 yp2(비마스크 stack) — H_max 위반.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
Y3 = REPO / "claudedocs/runtime_logs/yard_track/y3_d455"
Y2 = REPO / "claudedocs/runtime_logs/yard_track/y2_d454"
ARMS = ("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8")
OUT_JSON = Y3 / "y3_policy_summary.json"
OUT_PNG = Y3 / "y3_policy_summary.png"


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    if OUT_JSON.exists() or OUT_PNG.exists():
        raise SystemExit("WRITE_GUARD y3_policy_summary already exists")
    res = {a: json.loads((Y3 / f"{a}_results.json").read_text()) for a in ARMS}
    yp1 = json.loads((Y2 / "yp1_results.json").read_text())
    yp2 = json.loads((Y2 / "yp2_results.json").read_text())

    rows = {}
    for a in ARMS:
        m = res[a]["metrics"]
        rows[a] = {
            "policy": res[a]["policy"],
            "verdict": res[a]["verdict"]["code"],
            "total_actions": m["total_actions"], "n_noop": m["n_noop"],
            "dispersion_mm": m["dispersion_mm"],
            "reshape_sum": m["reshape_cells"]["sum"],
            "hmax_violations_final": m["hmax_violations_final"],
            "hmax_violations_any": m["hmax_violations_any"],
            "bin_hmax_final_mm": m["bin_hmax_final_mm"],
            "mask_events": m["mask_events"],
            "mask_exhausted_total": m["mask_exhausted_total"],
            "pred_err_mm": m["pred_err_mm"],
            "class_pick_order": m["class_pick_order"],
            "total_steps": m["total_steps"],
            "wall_seconds": res[a]["wall_seconds"]}

    def picks(r):
        return [tuple(x["pick_cell_rc"]) for x in r["actions"]
                if x["type"] == "pick_place"]

    def rocks_seq(r):
        return [x["rock"] for x in r["actions"] if x["type"] == "pick_place"]

    # (ii) release 브리지 통제 검증: a8과 a1은 pick 정책·소스 물리 동일 →
    # pick 수열이 bit-동일해야 대조가 격리됨 (아니면 대조 무효로 기록)
    bridge_controlled = (picks(res["a1"]) == picks(res["a8"])
                         and rocks_seq(res["a1"]) == rocks_seq(res["a8"]))
    d1, d8 = rows["a1"]["dispersion_mm"], rows["a8"]["dispersion_mm"]
    contrasts = {
        "obs_value_actions": {
            "a5_blind_total": rows["a5"]["total_actions"],
            "a5_noops": rows["a5"]["n_noop"],
            "gated_total": {a: rows[a]["total_actions"] for a in ARMS if a != "a5"},
            "ratio_blind_over_32": rows["a5"]["total_actions"] / 32.0},
        "release_bridge_a8_vs_a1": {
            "pick_sequence_identical": bool(bridge_controlled),
            "a1_v2": {"dispersion_mm": d1,
                      "viol_final": rows["a1"]["hmax_violations_final"],
                      "bin_hmax_mm": rows["a1"]["bin_hmax_final_mm"]},
            "a8_y2rel": {"dispersion_mm": d8,
                         "viol_final": rows["a8"]["hmax_violations_final"],
                         "bin_hmax_mm": rows["a8"]["bin_hmax_final_mm"]},
            "yp1_reference": {"dispersion_mm": yp1["metrics"]["dispersion_mm"],
                              "viol_final": yp1["metrics"]["hmax_violations_final"],
                              "bin_hmax_mm": yp1["metrics"]["bin_hmax_final_mm"]}},
        "mask_effect_a7_vs_yp2": {
            "a7_masked_stack": {
                "viol_final": rows["a7"]["hmax_violations_final"],
                "viol_any": rows["a7"]["hmax_violations_any"],
                "bin_hmax_mm": rows["a7"]["bin_hmax_final_mm"],
                "mask_exhausted": rows["a7"]["mask_exhausted_total"]},
            "yp2_unmasked_stack": {
                "viol_final": yp2["metrics"]["hmax_violations_final"],
                "viol_any": yp2["metrics"]["hmax_violations_any"],
                "bin_hmax_mm": yp2["metrics"]["bin_hmax_final_mm"]},
            "note": "yp2는 release=y2(0.20m)이므로 순수 마스크 단독 대조는 "
                    "a7 vs yp2가 아니라 방향성 증거 — release 통제 대조는 "
                    "a8 계열 후속 필요"},
        "pick_order_reshape": {
            a: {"reshape_sum": rows[a]["reshape_sum"],
                "class_first8": rows[a]["class_pick_order"][:8]}
            for a in ("a1", "a2", "a3", "a4", "a5")}}

    preds = {
        "pred1_gated_32_blind_gt_32": bool(
            all(rows[a]["total_actions"] == 32 for a in ARMS if a != "a5")
            and rows["a5"]["total_actions"] > 32),
        "pred2_a7_final_viol_le_2": bool(rows["a7"]["hmax_violations_final"] <= 2),
        "pred3_a8_disp_gt_a1": bool(d8["mean"] > d1["mean"] and d8["p95"] > d1["p95"]),
        "pred4_exploratory_reshape": {a: rows[a]["reshape_sum"]
                                      for a in ("a1", "a2", "a3", "a4")}}

    out = {"tool": "y3-policy-summary", "case": "y3_d455",
           "prereg": "y3_prereg.md (37c11131)",
           "inputs_sha16": {f"{a}_results.json": sha16(Y3 / f"{a}_results.json")
                            for a in ARMS}
           | {"yp1_results.json": sha16(Y2 / "yp1_results.json"),
              "yp2_results.json": sha16(Y2 / "yp2_results.json")},
           "arms": rows, "contrasts": contrasts, "predictions_check": preds}
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(ARMS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("total actions", [rows[a]["total_actions"] for a in labels], 32.0,
         "32 = rock count"),
        ("dispersion p95 [mm]", [rows[a]["dispersion_mm"]["p95"] for a in labels],
         yp1["metrics"]["dispersion_mm"]["p95"], "yp1 (0.20m drop)"),
        ("H_max violations final [cells]",
         [rows[a]["hmax_violations_final"] for a in labels],
         yp2["metrics"]["hmax_violations_final"], "yp2 unmasked stack"),
        ("reshape sum [cells]", [rows[a]["reshape_sum"] for a in labels],
         yp1["metrics"]["reshape_cells"]["mean"] * 32, "yp1"),
        ("bin h_max final [mm]", [rows[a]["bin_hmax_final_mm"] for a in labels],
         80.0, "H_max = 80mm"),
        ("mask exhausted [cycles]", [rows[a]["mask_exhausted_total"] for a in labels],
         None, None)]
    for ax, (title, vals, ref, refname) in zip(axes.ravel(), panels):
        ax.bar(labels, vals, color="#4878b0")
        if ref is not None:
            ax.axhline(ref, color="#c44e52", ls="--", lw=1,
                       label=f"{refname} = {ref:.1f}")
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("y3_d455 policy comparison — authority: a*_results.json "
                 "(this figure is derived)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)

    print(f"[y3_summary] {OUT_JSON.name} sha16={sha16(OUT_JSON)}")
    print(f"[y3_summary] bridge_controlled={bridge_controlled} "
          f"predictions={ {k: v for k, v in preds.items() if k != 'pred4_exploratory_reshape'} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
