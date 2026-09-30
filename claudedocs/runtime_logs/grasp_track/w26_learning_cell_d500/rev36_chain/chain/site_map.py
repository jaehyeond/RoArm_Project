"""rev36-chain 가능 위치 지도(CPU, 물리 0). 상자 좌표 명령점 격자마다 셀 시뮬과 **같은 함수**로
IK·관절 제한·벽 여유를 계산해, 오케스트레이터가 고를 수 있는 명령점 집합을 만든다.
또 명령점 = 상자 중심이 rev34 취점 자리·회전과 binary64 로 같은지 자체 검사한다.

사용: python site_map.py <params.json> <pile.npz> <out.json> [--step-mm 5]
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN)); sys.path.insert(0, str(SRC))
import sim_deme_scoop_s1 as W11SRC   # noqa: E402
import w13_fk as FK                  # noqa: E402
import w13_kinematics as K           # noqa: E402
import w26_cell_site as CS           # noqa: E402


def load_params(path):
    """셀 시뮬과 같은 병합 순서: W11 기본값 → W13 기본값 → params 파일."""
    P = dict(W11SRC.DEFAULT); P.update(K.W13_DEFAULT); P.update(json.load(open(path)))
    return P


def tool_setup(P, pile):
    z = np.load(pile, allow_pickle=True)
    box, _ = FK.w25_tray_bounds(np.asarray(z["box_bounds_m"], float), P, np.asarray(z["positions_m"], float),
                                np.asarray(z["radii_m"], float))
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]; q_end = P["close_end_joint_deg"]
    fixed_m, door_m, _, _, hinge_off, L5mm, _ = W11SRC.load_tool(P, q_open)
    P["lip_l5_mm"] = [float(v) for v in L5mm]
    axis_w = W11SRC.R_W @ np.array([0.0, 1.0, 0.0])
    return dict(box=box, q_open=q_open, q_end=q_end, fixed_v=np.asarray(fixed_m.vertices, float),
                door_v=np.asarray(door_m.vertices, float), hinge_off=hinge_off, axis_w=axis_w)


def reason_counts(rows):
    c = {}
    for r in rows:
        if not r["feasible"]:
            k = "벽 여유 부족" if (r["reason"] or "").startswith("벽 여유") else (r["reason"] or "").split(":")[0]
            c[k] = c.get(k, 0) + 1
    return c


def evaluate(P, T, cmd_xy, rolls=(0.0,)):
    """롤 후보마다 벽 여유를 재고, 가능한 것 중 여유 최대를 고른다(롤 0 이 가능하면 0 우선 = rev34 규약 유지)."""
    margin = float(P.get("w26_cell_wall_margin_mm", 2.0)) / 1000.0
    try:
        res = CS.cell_site_rolls(P, P["lip_l5_mm"], cmd_xy, rolls)
    except SystemExit as e:
        return {"cmd_box_xy_m": list(cmd_xy), "feasible": False, "feasible_roll0": False, "reason": str(e), "per_roll": None}
    per = {}
    for r, v in res.items():
        if isinstance(v, str):
            per[r] = {"ok": False, "reason": v.split(" (")[0]}
            continue
        site, R_cell, info = v
        w_min, _ = CS.wall_clearance(T["fixed_v"], T["door_v"], T["hinge_off"], T["axis_w"], T["q_open"],
                                     [T["q_end"], T["q_open"]], site, [0.0], R_cell, T["box"])
        per[r] = {"ok": bool(w_min >= margin), "wall_clearance_min_mm": w_min * 1000, "lip_box_xy_m": list(site),
                  "base_deg": info["base_deg"], "tool_yaw_box_deg": info["tool_yaw_box_deg"],
                  "R_fk_minus_ideal_max": info["R_fk_minus_ideal_max"]}
    ok = {r: v for r, v in per.items() if v["ok"]}
    if 0.0 in ok:
        best = 0.0
    elif ok:
        best = max(ok, key=lambda r: ok[r]["wall_clearance_min_mm"])
    else:
        best = None
    b = per[best] if best is not None else None
    r0 = per.get(0.0, {})
    reason = None if best is not None else ("벽 여유 부족(모든 롤)" if any("wall_clearance_min_mm" in v for v in per.values()) else r0.get("reason"))
    return {"cmd_box_xy_m": list(cmd_xy), "feasible": best is not None, "feasible_roll0": bool(r0.get("ok", False)),
            "best_roll_deg": best, "lip_box_xy_m": b["lip_box_xy_m"] if b else None,
            "base_deg": b["base_deg"] if b else None, "tool_yaw_box_deg": b["tool_yaw_box_deg"] if b else None,
            "wall_clearance_min_mm": b["wall_clearance_min_mm"] if b else None,
            "R_fk_minus_ideal_max": b["R_fk_minus_ideal_max"] if b else None,
            "reason": reason, "per_roll": {str(r): v for r, v in per.items()}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("params"); ap.add_argument("pile"); ap.add_argument("out")
    ap.add_argument("--step-mm", type=float, default=5.0)
    ap.add_argument("--rolls", default="0", help="손목 롤 후보(°) 쉼표 목록, 예 0,30,-30,60,-60,90,-90")
    a = ap.parse_args()
    rolls = [float(v) for v in a.rolls.split(",")]
    P = load_params(a.params); T = tool_setup(P, a.pile)
    sc = CS.self_check_center(P, P["lip_l5_mm"])
    box = T["box"]; st = a.step_mm / 1000.0
    xs = np.arange(math.ceil(box[0, 0] / st) * st, box[0, 1] + 1e-12, st)
    ys = np.arange(math.ceil(box[1, 0] / st) * st, box[1, 1] + 1e-12, st)
    rows = [evaluate(P, T, (round(float(x), 6), round(float(y), 6)), rolls) for y in ys for x in xs]
    feas = [r for r in rows if r["feasible"]]
    fx = np.array([r["cmd_box_xy_m"] for r in feas]) if feas else np.zeros((0, 2))
    roll_hist = {}
    for r in feas:
        roll_hist[str(r["best_roll_deg"])] = roll_hist.get(str(r["best_roll_deg"]), 0) + 1
    out = {"artifact": "W26_CELL_SITE_MAP_V2", "params": a.params, "pile": a.pile, "box_inner_m": box.tolist(),
           "step_mm": a.step_mm, "wall_margin_mm": float(P.get("w26_cell_wall_margin_mm", 2.0)), "rolls_deg": rolls,
           "self_check_center": sc, "n_grid": len(rows), "n_feasible": len(feas),
           "n_feasible_roll0": sum(1 for r in rows if r["feasible_roll0"]), "best_roll_hist": roll_hist,
           "feasible_cmd_bbox_m": {"x": [float(fx[:, 0].min()), float(fx[:, 0].max())],
                                   "y": [float(fx[:, 1].min()), float(fx[:, 1].max())]} if feas else None,
           "max_R_fk_minus_ideal": max((r.get("R_fk_minus_ideal_max") or 0.0) for r in rows),
           "reasons_infeasible": reason_counts(rows),
           "rows": rows,
           "frame": "시뮬 rev34 상자 규약(box_frame_convention 값 그대로, 원점 = 상자 안쪽 바닥 중심). "
                    "명령점 = 로봇에 주는 goto_xyz 목표의 상자 좌표, 립 = FK 립. best_roll = 가능한 롤 중 여유 최대(0 가능하면 0)."}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("self_check_center", "n_grid", "n_feasible", "n_feasible_roll0", "best_roll_hist",
                                          "feasible_cmd_bbox_m", "max_R_fk_minus_ideal", "reasons_infeasible")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
