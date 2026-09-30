"""rev36-chain 연쇄 오케스트레이터. 같은 더미에서 셀을 K 번 이어 붙인다:
    더미_k → (명령점 선택) → params_k → 셀 run_k → 다음 더미_{k+1} + 데이터 행_k → …

셀마다 별도 프로세스(자기 세션). 실패하면 그 자리에서 멈추고 chain.json 에 사유를 남긴다(재시도 0).
모드
    stub : CPU 운동학 스텁(`checks/cpu_stub_dryrun_w26.py`, 물리 0) — 흐름·위치·저장·재시작·보존 검사용
    gpu  : 셀마다 COMMANDS json 을 쓰고 동결 러너(`run_w26.py`: 매니페스트 해시 대조·상한·재시도 0)를 부른다.
           ⚠ GPU 실행은 별도 사용자 승인 대상이다. 이 파일은 경로만 준비한다.
정책(명령점 선택): fixed_list(목록 순서) · random_feasible(지도에서 시드 고정 무작위) · **random / highest / footprint_volume**
    (policies.py — 2단계 결정 실험용 규칙 정책, 현재 더미 높이지도 참값 + 툴 발자국으로 고른다. 09-30 추가).

사용: python orchestrate.py <plan.json>
plan = {"chain_id", "mode": "stub"|"gpu", "base_params", "initial_pile", "steps", "out_dir",
        "policy": {"name": "fixed_list", "cmd_box_xy_m": [[x, y], ...]} | {"name": "random_feasible", "site_map": path, "seed": int},
        "cam": true, "gpu": {"run_w26": path, "manifest": path, "cap_s": 21600, "grace_s": 600, "max_wall_s": 21000}}
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REV = HERE.parent
PY = "/home/cgxr/miniconda3/envs/roarm/bin/python"
sys.path.insert(0, str(HERE))
import next_pile as NP   # noqa: E402
import row as ROW        # noqa: E402
import policies as PO    # noqa: E402
import site_map as SM    # noqa: E402

utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def choose(policy, k, rng, feas, ctx=None):
    """반환 (cmd_xy, roll_deg, info). fixed_list 항목 = [x, y] 또는 [x, y, roll]; random_feasible = 지도의 best_roll(0 가능하면 0);
    random/highest/footprint_volume = policies.py 규칙(현재 더미 높이지도 참값 + 툴 발자국)."""
    if policy["name"] == "fixed_list":
        e = [float(v) for v in policy["cmd_box_xy_m"][k]]
        return e[:2], (e[2] if len(e) > 2 else 0.0), {"policy": "fixed_list"}
    if policy["name"] == "random_feasible":
        r = feas[int(rng.integers(len(feas)))]
        return [float(v) for v in r["cmd_box_xy_m"]], float(r.get("best_roll_deg") or 0.0), {"policy": "random_feasible"}
    if policy["name"] in PO.POLICIES:
        H = PO.heightmap_from_pile(ctx["pile"], ctx["spec"])
        r, sc, i = PO.choose(policy["name"], H, ctx["spec"], feas, ctx["P"], ctx["fp"], rng)
        info = {"policy": policy["name"], "score": None if policy["name"] == "random" else float(sc[i]),
                "n_candidates": len(feas), "n_tied": None if policy["name"] == "random" else int((sc >= sc.max() - 1e-12).sum()),
                "h_ref_p10_m": float(np.percentile(H, 10)), "H_median_m": float(np.median(H))}
        return [float(v) for v in r["cmd_box_xy_m"]], float(r.get("best_roll_deg") or 0.0), info
    raise SystemExit(f"알 수 없는 정책: {policy['name']}")


def run_cell(plan, params_k, pile_k, out_k, log_k):
    if plan["mode"] == "stub":
        argv = [PY, "-B", str(REV / "checks/cpu_stub_dryrun_w26.py"), "--src", str(REV / "src"), "--params", str(params_k),
                "--pile", str(pile_k), "--out", str(out_k), "--stop-after-phase", "reclose"]
        with open(log_k, "w") as fo, open(str(log_k) + ".err", "w") as fe:
            p = subprocess.Popen(argv, stdout=fo, stderr=fe, start_new_session=True)
            rc = p.wait()
        return rc, argv
    g = plan["gpu"]
    cmd = {"artifact": "W26_CHAIN_COMMANDS_V1", "pod_tag": f"{plan['chain_id']}_{out_k.name}", "manifest": g["manifest"],
           "env": {"PYTHONDONTWRITEBYTECODE": "1"}, "auto_retry": False, "go_required_steps": ["cell"],
           "steps": {"cell": {"attempt_dir": str(out_k), "cap_s": g["cap_s"], "grace_s": g["grace_s"], "cwd": str(REV / "src"),
                              "argv": [PY, "-B", str(REV / "src/sim_w13_full_cycle.py"), "--params", str(params_k), "--pile",
                                       str(pile_k), "--out", str(out_k), "--seed", "460", "--stop-after-phase", "reclose",
                                       "--max-wall-s", str(g["max_wall_s"])]}}}
    cj = Path(str(out_k) + "_COMMANDS.json"); cj.write_text(json.dumps(cmd, ensure_ascii=False, indent=1) + "\n")
    argv = [PY, "-B", g["run_w26"], "--commands", str(cj), "--step", "cell", "--allow-go-step"]
    with open(log_k, "w") as fo, open(str(log_k) + ".err", "w") as fe:
        rc = subprocess.run(argv, stdout=fo, stderr=fe).returncode
    return rc, argv


def main(plan_path):
    plan = json.load(open(plan_path)); out = Path(plan["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    base = json.load(open(plan["base_params"]))
    rng = np.random.default_rng(plan["policy"].get("seed", 0))
    feas, ctx = None, None
    if plan["policy"]["name"] in ("random_feasible",) + PO.POLICIES:
        sm = json.load(open(plan["policy"]["site_map"]))
        feas = [r for r in sm["rows"] if r["feasible"]]
    if plan["policy"]["name"] in PO.POLICIES:
        P = SM.load_params(plan["base_params"]); T = SM.tool_setup(P, plan["initial_pile"])
        ctx = {"P": P, "fp": PO.Footprint(P, T), "spec": PO.grid_from_box(T["box"])}
    ch = {"artifact": "W26_CHAIN", "chain_id": plan["chain_id"], "plan": plan, "plan_sha256": NP.sha256(plan_path),
          "started_utc": utc(), "status": "running", "steps": []}
    cj = out / "chain.json"
    pile = Path(plan["initial_pile"])
    for k in range(int(plan["steps"])):
        st = {"k": k, "pile_in": str(pile), "pile_in_sha256": NP.sha256(pile), "t0_utc": utc()}
        try:
            if ctx is not None: ctx["pile"] = pile
            xy, roll, pinfo = choose(plan["policy"], k, rng, feas, ctx); st["cmd_box_xy_m"] = xy; st["roll_deg"] = roll; st["policy"] = pinfo
            pk = dict(base, w26_cell_start_at_scoop_pose=True, w26_cell_cmd_box_xy_m=xy, w26_cell_tool_roll_deg=roll)
            params_k = out / f"params_{k:03d}.json"; params_k.write_text(json.dumps(pk, ensure_ascii=False, indent=1) + "\n")
            st["params_sha256"] = NP.sha256(params_k)
            cell_k = out / f"cell_{k:03d}"; t0 = time.time()
            rc, argv = run_cell(plan, params_k, pile, cell_k, out / f"cell_{k:03d}.log")
            st.update(rc=rc, cell_wall_s=round(time.time() - t0, 1), argv=argv)
            if rc != 0:
                raise RuntimeError(f"셀 rc={rc}")
            rj = json.load(open(NP.cell_npz(cell_k)[:-4] + ".json"))
            if rj.get("stopped_early_after_phase") != "reclose" or rj.get("diverged") is not False:
                raise RuntimeError(f"셀 상태 이상: stopped={rj.get('stopped_early_after_phase')} diverged={rj.get('diverged')}")
            nxt = out / f"pile_{k+1:03d}.npz"
            rec = NP.build(cell_k, pile, params_k, nxt)
            r = ROW.build(cell_k, pile, params_k, out / f"row_{k:03d}",
                          chain={"chain_id": plan["chain_id"], "k": k, "policy": plan["policy"]["name"], "mode": plan["mode"]},
                          cam=bool(plan.get("cam", True)))
            st.update(lip_box_xy_m=r["action"]["lip_box_xy_m"], base_deg=r["action"]["base_deg"],
                      tool_yaw_box_deg=r["action"].get("tool_yaw_box_deg"),
                      lifted_count=r["label"]["lifted_count"], removed=rec["n_removed_lifted"],
                      removed_out_of_tray=rec["n_removed_out_of_tray"], n_kept=rec["n_kept"],
                      pile_out=str(nxt), pile_out_sha256=rec["out_sha256"], row=str(out / f"row_{k:03d}.json"),
                      crater_volume_ml=r["crater"]["removed_volume_ml"], sim_wall_s=rj.get("wall_seconds"),
                      post_settled_row=str(out / f"row_{k+1:03d}.json") if k + 1 < int(plan["steps"]) else None,
                      status="ok", t1_utc=utc())
            ch["steps"].append(st); cj.write_text(json.dumps(ch, ensure_ascii=False, indent=1) + "\n")
            pile = nxt
        except (Exception, SystemExit) as e:
            st.update(status="failed", error=f"{type(e).__name__}: {e}", t1_utc=utc())
            ch["steps"].append(st); ch["status"] = f"failed_at_step_{k}"; ch["ended_utc"] = utc()
            cj.write_text(json.dumps(ch, ensure_ascii=False, indent=1) + "\n")
            print(f"CHAIN_FAILED k={k}: {e}", flush=True); sys.exit(1)
    ch["status"] = "complete"; ch["ended_utc"] = utc()
    n = [s["n_kept"] + s["removed"] + s["removed_out_of_tray"] for s in ch["steps"]]
    ch["conservation"] = {"n_in_per_step": n, "ok": all(n[i + 1] == ch["steps"][i]["n_kept"] for i in range(len(n) - 1))}
    cj.write_text(json.dumps(ch, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"status": ch["status"], "steps": [{k: s.get(k) for k in ("k", "cmd_box_xy_m", "roll_deg", "lip_box_xy_m", "base_deg", "tool_yaw_box_deg",
                      "lifted_count", "n_kept", "cell_wall_s")} for s in ch["steps"]], "conservation": ch["conservation"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
