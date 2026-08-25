#!/usr/bin/env python
"""p30 — y2_d454 spread(yp1) vs stack(yp2) 대조 요약 (물리 0, numpy 전용).

프로포절 핵심 주장("place 선택이 다음 상태·제약에 영향")의 sim 실증치를
yp1/yp2 results에서 추출해 단일 대조 산출물로 영속화.  판정 재작성 아님.
출력 = y2_contrast_summary.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parents[1] / "claudedocs/runtime_logs/yard_track/y2_d454"


def load(tag):
    p = CASE / f"{tag}_results.json"
    return json.loads(p.read_text()), hashlib.sha256(p.read_bytes()).hexdigest()[:16]


yp1, sha1 = load("yp1")
yp2, sha2 = load("yp2")

pick1 = [(c["rock"], round(c["pick_h_mm"], 3)) for c in yp1["cycles"]]
pick2 = [(c["rock"], round(c["pick_h_mm"], 3)) for c in yp2["cycles"]]
resh1 = [c["reshape_cells_outside_fp"] for c in yp1["cycles"]]
resh2 = [c["reshape_cells_outside_fp"] for c in yp2["cycles"]]

out = {
    "tool": "p30_y2_contrast_summary",
    "inputs": {"yp1_results_sha16": sha1, "yp2_results_sha16": sha2},
    "pick_side_identical": {
        "pick_sequence_equal": pick1 == pick2,
        "reshape_sequence_equal": resh1 == resh2,
        "reading": "pick측(greedy argmax)·source 재형성이 place 패턴과 무관하게 "
                   "동일 — 트레이 간 물리 독립 + 결정론의 내적 일관성 검증"},
    "contrast": {
        "hmax_violation_cells_final": {"yp1_spread": yp1["metrics"]["hmax_violations_final"],
                                       "yp2_stack": yp2["metrics"]["hmax_violations_final"]},
        "hmax_violation_cells_any": {"yp1_spread": yp1["metrics"]["hmax_violations_any"],
                                     "yp2_stack": yp2["metrics"]["hmax_violations_any"]},
        "n_cycles_with_violation": {"yp1_spread": yp1["metrics"]["n_cycles_with_violation"],
                                    "yp2_stack": yp2["metrics"]["n_cycles_with_violation"]},
        "bin_hmax_final_mm": {"yp1_spread": yp1["metrics"]["bin_hmax_final_mm"],
                              "yp2_stack": yp2["metrics"]["bin_hmax_final_mm"]},
        "dispersion_mm": {"yp1_spread": yp1["metrics"]["dispersion_mm"],
                          "yp2_stack": yp2["metrics"]["dispersion_mm"]},
        "escapes_out": {"yp1_spread": yp1["metrics"]["end_counts"]["out"],
                        "yp2_stack": yp2["metrics"]["end_counts"]["out"]}},
    "readings": [
        "동일 pick 정책·동일 초기 더미에서 place 셀 선택만 바꾸면 H_max 위반"
        " 셀 1→11(최종)·3→18(any), bin 최대 높이 80.8→94.9mm — place 선택이"
        " 상태·제약 위반을 실측 가능하게 바꾼다 (프로포절 §5 프레임의 sim 실증)",
        "재형성(footprint 밖 |ΔH|>2mm)은 평균 2.9셀·최대 17셀, 더미가 높은"
        " 전반부에 집중 — '매 pick마다 재형성' 전제 실증 (RQ2 성립 근거)",
        "착지 분산 p95 43mm(spread)~63mm(stack) ≫ 셀 10mm — drop_z≈0.20m"
        " 프리미티브 기준의 상한이며, place 행동 해상도(셀 vs 존)와 release"
        " 높이 설계의 입력. 실기 release 높이는 더 낮으므로 과대추정 가능"
        " (비주장: 실기 분산)"],
    "non_claims": "정책 최적성 비교 아님(greedy 하나 고정); H_max는 회계만"
                  "(강제 없음); 파지 물리·실기 분산·Kinect 충실도 비주장",
}
p = CASE / "y2_contrast_summary.json"
p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(out["contrast"], indent=1))
print("pick_side_identical:", out["pick_side_identical"]["pick_sequence_equal"],
      out["pick_side_identical"]["reshape_sequence_equal"])
print("sha16:", hashlib.sha256(p.read_bytes()).hexdigest()[:16])
