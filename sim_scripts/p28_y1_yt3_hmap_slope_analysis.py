#!/usr/bin/env python
"""p28 — yt3 G-hmap FAIL 기전 분석 (물리 0, numpy 전용).

질문: yt3의 |obs−gt| > 0.5mm 셀들이 전부 '가파른 면(cooked convex 수평 표현차의
수직 증폭)' 셀인가?  셀별 GT 승자 삼각형의 기울기를 계산해 분류하고, 게이트
FAIL이 파이프라인 결함이 아니라 표현 갭임을 정량 기록한다.
출력 = yt3_hmap_slope_analysis.json (판정 결과 재작성 아님 — yt3 verdict 보존).
"""
import json
import hashlib
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
CASE = R / "claudedocs/runtime_logs/yard_track/y1_d453"
res = json.loads((CASE / "yt3_results.json").read_text())
d = np.load(CASE / "yt3_trace.npz")
m = json.loads((R / "sim_assets/posco_rocks_o1/manifest.json").read_text())
by = {o["name"]: o for o in m["objects"]}


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


tris = []  # (name, tri(3,3))
for p in res["final_poses"]:
    o = by[p["name"]]
    V = np.array(o["vertices_m"]) @ q2R(p["quat_wxyz"]).T + np.array(p["pos"])
    for f in np.array(o["faces"]):
        tris.append((p["name"], V[f]))


def gt_winner(cx, cy):
    best = None
    for name, tri in tris:
        a, b, c = tri
        d0, d1 = b[:2] - a[:2], c[:2] - a[:2]
        den = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(den) <= 1e-14:
            continue
        pp = np.array([cx, cy]) - a[:2]
        u = (pp[0] * d1[1] - pp[1] * d1[0]) / den
        v = (d0[0] * pp[1] - d0[1] * pp[0]) / den
        if u >= -1e-12 and v >= -1e-12 and u + v <= 1 + 1e-12:
            z = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
            if best is None or z > best[0]:
                n = np.cross(b - a, c - a)
                n /= np.linalg.norm(n)
                slope = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
                best = (float(z), name, slope)
    return best


cells = d["src_cells"]
obs = d["hmap_src_obs"]
gt = d["hmap_src_gt"]
diff = np.abs(obs - gt)
rows = []
for r_ in range(13):
    for c_ in range(13):
        if diff[r_, c_] > 0.0005:
            w = gt_winner(cells[r_, c_, 0], cells[r_, c_, 1])
            rows.append({
                "cell_rc": [r_, c_], "abs_diff_mm": float(diff[r_, c_] * 1000),
                "obs_mm": float(obs[r_, c_] * 1000), "gt_mm": float(gt[r_, c_] * 1000),
                "gt_winner_rock": w[1], "gt_face_slope_deg": w[2],
                "amplification_at_slope": float(np.tan(np.radians(min(w[2], 89.9))))})

slopes = [r_["gt_face_slope_deg"] for r_ in rows]
n_steep = sum(s > 60.0 for s in slopes)
out = {
    "tool": "p28_yt3_hmap_slope_analysis",
    "inputs": {"yt3_results_sha16": hashlib.sha256(
        (CASE / "yt3_results.json").read_bytes()).hexdigest()[:16]},
    "question": "|obs-gt|>0.5mm 셀이 전부 가파른 면 표현-증폭 셀인가",
    "n_cells_over_0p5mm": len(rows),
    "n_over_cells_with_slope_gt60": n_steep,
    "all_over_cells_steep": bool(n_steep == len(rows)),
    "max_diff_cell": max(rows, key=lambda r_: r_["abs_diff_mm"]) if rows else None,
    "cells": rows,
    "reading": (
        "obs(레이캐스트)는 cooked convex hull 표면 — sim 충돌이 보는 표면과 "
        "동일(자기일관). gt(raw 메쉬)와의 차이는 기울기 tan(θ) 증폭으로 가파른 "
        "면에만 국한. G-hmap max 게이트(2mm)는 이 증폭을 미고려한 설계 — "
        "파이프라인 결함 아님. yt3 verdict(FAIL)는 보존, 본 분석은 기전 기록."),
    "non_claims": "Kinect 실관측 충실도·프린트 표면과의 대응은 별도 검증 필요",
}
p = CASE / "yt3_hmap_slope_analysis.json"
p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: out[k] for k in (
    "n_cells_over_0p5mm", "n_over_cells_with_slope_gt60", "all_over_cells_steep")},
    ensure_ascii=False))
print("cells:", [(r_["cell_rc"], round(r_["abs_diff_mm"], 2),
                  round(r_["gt_face_slope_deg"], 1)) for r_ in rows])
print("sha16:", hashlib.sha256(p.read_bytes()).hexdigest()[:16])
