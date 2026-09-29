"""표시 한계(재투영 최대 오차·선언 범위 밖 포즈 수)를 **렌더 없이** CPU 로 미리 계산한다.

정의는 post05 매니페스트와 **같다**(새 정의를 만들지 않는다):
  · `max_lip_reprojection_err_mm` = 표시 관절로 FK 한 립 위치와 저장된 립 위치 차의 최대값
    (`isaac_replay_w13_post05.py:525`).
  · `n_frames_with_limit_violations` = `FK.in_limits(q5)` 가 비지 않은 표시 레코드 수 (`:528`).
  · `joint_source_counts` = 정확 문자열 일치 집계 (`:520`, post04 결함 ③ 정정판).

어떻게: post05·post06 의 `mode_of`/`joints_for` **본문을 AST 로 꺼내 그대로 실행**한다. 프레임 선택은
본 실행 규칙(저장된 입자 프레임 전부, 1:1)을 따른다. 물리 0 · Isaac 0 · GPU 0 · 원자료 읽기만.

이 수치는 **기하 일관성 지표이지 구동 검증이 아니다**(post05 의 NOT_ACTUATION_VERIFICATION 계약 그대로).
"""
import ast
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True                    # 동결 worktree 에 __pycache__ 를 쓰지 않는다
POST04_SRC = Path("/home/cgxr/orca/workspaces/RoArm_Project/w19-replay/claudedocs/runtime_logs/grasp_track/"
                  "w19_runpod_d487/A_full_cycle/replay_20260918/rev/src")
sys.path.insert(0, str(POST04_SRC))
import w13_fk as FK                                                    # noqa: E402

HERE = Path(__file__).resolve().parent
POST05 = Path("/home/cgxr/orca/workspaces/RoArm_Project/w25-render-cad/claudedocs/research/"
              "w25_render_cad_20260928/isaac_replay_w13_post05.py")
POST06 = HERE / "isaac_replay_w13_post06.py"
RUNS = {
    "a_rev34_stub_convA": Path("/home/cgxr/orca/workspaces/RoArm_Project/w25-rev34-fullcycle/claudedocs/"
                               "runtime_logs/grasp_track/w25_realign_fullcycle_d498/rev34/dryrun/"
                               "paperbox_final_n67737"),
    "b_w19A": Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/"
                   "w19_runpod_d487/A_full_cycle/run_01"),
}
CART_SUB = {"align_to_w11_scoop_pose", "align_from_w11_scoop_pose", "door_open_at_approach",
            "plunge", "close", "lift", "reclose"}
PLATE_Z = 0.38


def grab(path, names):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))
    if set(out) != set(names):
        raise SystemExit(f"{path}: {set(names) - set(out)} 를 찾지 못했다")
    return out


def run_one(path, run_dir, use_rot):
    z = np.load(run_dir / "w13_cycle_seed460.npz", allow_pickle=True)
    res = json.load(open(run_dir / "w13_cycle_seed460.json"))
    meta = json.loads(str(z["metadata_json"]))
    t_robot = np.asarray(res["frames"]["adapter"]["t_robot_m"], float)
    W25F = meta.get("w25_frame")
    ROT = None
    if use_rot and W25F is not None:
        ROT = np.asarray(W25F["R_robot_box"], float)
        if float(np.abs(ROT - np.eye(3)).max()) == 0.0:
            ROT = None
    ns = {"np": np, "math": math, "FK": FK}
    names = ["mode_of", "joints_for"] + (["box_to_robot"] if "post06" in Path(path).name else [])
    for nm, src in grab(path, names).items():
        exec(compile(src, str(path), "exec"), ns)                       # noqa: S102
    ns.update({"tool_p": np.asarray(z["tool_pos_m"], float), "t_robot": t_robot, "ROT": ROT,
               "lip_owner": list(res["params"]["lip_l5_mm"]), "CART_SUB": CART_SUB,
               "z": {"sync_joint_deg": np.asarray(z["sync_joint_deg"], float)},
               "cmd_mode": ([str(v) for v in np.asarray(z["sync_cmd_mode"])]
                            if "sync_cmd_mode" in z.files else None),
               "sub": [str(v) for v in np.asarray(z["sync_subphase"])]})
    pf_s = np.asarray(z["particle_frame_sync_index"], int)
    recs, fails = [], []
    for fi in range(len(pf_s)):
        si = int(pf_s[fi]) if pf_s[fi] >= 0 else 0
        ik, how = ns["joints_for"](si)
        (recs if ik is not None else fails).append(ik if ik is not None else {"frame": fi, "reason": how})
    errs = [r["ik_err_mm"] for r in recs]
    return {"run": str(run_dir), "transform": "post06 (회전 적용)" if ROT is not None else
            ("post05 식(평행이동만)" if not use_rot else "회전 없음(w25_frame 부재 → post05 와 같은 식)"),
            "w25_frame_present": bool(W25F is not None), "rotation_applied": bool(ROT is not None),
            "n_particle_frames": int(len(pf_s)), "n_display_records": len(recs),
            "max_lip_reprojection_err_mm": round(max(errs or [0.0]), 4),
            "median_lip_reprojection_err_mm": round(float(np.median(errs)) if errs else 0.0, 4),
            "n_frames_with_limit_violations": sum(1 for r in recs if r["limit_violations"]),
            "limit_violation_joints": sorted({str(j["joint"]) for r in recs
                                              for j in (r["limit_violations"] or [])}),
            # 초과량 = 하한 미만이면 lo−v, 상한 초과면 v−hi (둘 다 아니면 0). 단위는 도(deg).
            "limit_violation_worst_overshoot_deg": round(max(
                [max(j["limit"][0] - j["value"], j["value"] - j["limit"][1], 0.0)
                 for r in recs for j in (r["limit_violations"] or [])] or [0.0]), 4),
            "joint_source_counts": {s: sum(1 for r in recs if r["joint_source"] == s)
                                    for s in sorted({r["joint_source"] for r in recs})},
            "n_ik_failures": len(fails), "ik_failure_reasons": sorted({f["reason"] for f in fails})}


def main():
    rep = {"artifact": "W25_POST06_DISPLAY_LIMITS_CPU_V1",
           "definitions_from": "isaac_replay_w13_post05.py:520-528 (정의 변경 0)",
           "not_a_claim": "기하 일관성 지표다. 서보 구동 검증도, 렌더 육안 검수도 아니다.",
           "cases": {}}
    rep["cases"]["a_rev34_stub_post06"] = run_one(POST06, RUNS["a_rev34_stub_convA"], True)
    rep["cases"]["a_rev34_stub_post05_unpatched"] = run_one(POST05, RUNS["a_rev34_stub_convA"], False)
    rep["cases"]["b_w19A_post06"] = run_one(POST06, RUNS["b_w19A"], True)
    rep["cases"]["b_w19A_post05"] = run_one(POST05, RUNS["b_w19A"], False)
    prior = json.load(open(HERE.parent / "render_post05_w19A_20260929" / "render_manifest.json"))
    pc = prior["robot_display_contract"]
    mine = rep["cases"]["b_w19A_post06"]
    rep["w19A_vs_prior_post05_render"] = {
        "prior_manifest": str(HERE.parent / "render_post05_w19A_20260929" / "render_manifest.json"),
        "prior_max_lip_reprojection_err_mm": pc["max_lip_reprojection_err_mm"],
        "mine_max_lip_reprojection_err_mm": mine["max_lip_reprojection_err_mm"],
        "prior_n_frames_with_limit_violations": pc["n_frames_with_limit_violations"],
        "mine_n_frames_with_limit_violations": mine["n_frames_with_limit_violations"],
        "prior_joint_source_counts": pc["joint_source_counts"],
        "mine_joint_source_counts": mine["joint_source_counts"],
        "all_match": bool(pc["max_lip_reprojection_err_mm"] == mine["max_lip_reprojection_err_mm"]
                          and pc["n_frames_with_limit_violations"] == mine["n_frames_with_limit_violations"]
                          and pc["joint_source_counts"] == mine["joint_source_counts"])}
    out = HERE / "display_limits_post06.json"
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=2)
    for k, v in rep["cases"].items():
        print(f"[{k}] {v['transform']} · 재투영 최대 {v['max_lip_reprojection_err_mm']} mm "
              f"(중앙값 {v['median_lip_reprojection_err_mm']}) · 범위 밖 포즈 "
              f"{v['n_frames_with_limit_violations']}/{v['n_display_records']} "
              f"{v['limit_violation_joints']} · IK 실패 {v['n_ik_failures']}")
    print(f"[대조] W19 A post06 결과 == 이전 post05 렌더 매니페스트: "
          f"{rep['w19A_vs_prior_post05_render']['all_match']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
