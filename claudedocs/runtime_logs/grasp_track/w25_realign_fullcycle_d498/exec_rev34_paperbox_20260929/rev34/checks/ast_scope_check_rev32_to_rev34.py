"""rev32 -> rev34 AST 변경 범위 검사 (CPU, 읽기 전용).

실행되는 코드(구문 트리)가 어느 파일·어느 함수에서 바뀌었는지 기록하고, 아래 허용 목록 밖 변경이 있으면 FAIL.
  허용: w13_fk.py(Adapter 3메서드 + 새 w25_* 함수/상수), sim_w13_full_cycle.py(run 한 함수),
        preflight_{geometry,rigid_hinge,snapshot}.py, verify_w13_self.py(check_science·main 의 pile sha 인자)
  금지: 그 밖의 모든 src 파일(물리·분류·bridge·재생·Rerun) — 바이트 동일이어야 한다.
또 criteria.json 이 rev32 와 sha256 동일한지, params_w25.json 이 rev32 params 에서 바꾼 키가 선언 목록뿐인지 본다.
"""
import ast
import difflib
import hashlib
import json
import sys
from pathlib import Path

R32 = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/"
           "w19_runpod_d487/rev32_frozen_copy")
R34 = Path(__file__).resolve().parent.parent

ALLOWED = {
    "w13_fk.py": {"Adapter", "build_adapter_w25", "w25_scoop_site", "w25_tray_bounds", "w25_frame",
                  "_w25_reference", "_w25_t_xy", "BOX_FRAME_CONVENTIONS", "BOX_ANCHORS"},
    "sim_w13_full_cycle.py": {"run"},
    "preflight_geometry.py": {"<module docstring>", "main", "<if __name__>"},
    "preflight_rigid_hinge.py": {"*"},
    "preflight_snapshot.py": {"*"},
    "verify_w13_self.py": {"check_science", "main"},
}
PARAM_KEYS_CHANGED_ALLOWED = {"declared_base_cm", "declared_pellet_cm", "declared_placement_provenance", "arm_radius_m",
                              "tray_wall_t_mm"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def label(n):
    if isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant) and isinstance(n.value.value, str):
        return "<module docstring>"
    if isinstance(n, ast.If):
        return "<if __name__>"
    if isinstance(n, ast.Assign):
        return ",".join(ast.unparse(t) for t in n.targets)
    return getattr(n, "name", None) or f"<{type(n).__name__} @L{n.lineno}>"


rep = {"artifact": "W25A_AST_SCOPE_REV32_TO_REV34", "files": {}, "violations": []}
for f in sorted((R32 / "src").glob("*.py")):
    g = R34 / "src" / f.name
    a_txt, b_txt = f.read_text(), g.read_text()
    row = {"rev32_sha256": sha(f), "rev34_sha256": sha(g), "bytes_identical": a_txt == b_txt}
    if not row["bytes_identical"]:
        A, B = ast.parse(a_txt), ast.parse(b_txt)
        # 문장 AST 열의 순서 diff(줄번호 이동에 둔감). 바뀐/추가/삭제 문장의 이름만 모은다.
        sa, sb = [ast.dump(x) for x in A.body], [ast.dump(x) for x in B.body]
        changed = set()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
            if tag != "equal":
                changed |= {label(x) for x in A.body[i1:i2]} | {label(x) for x in B.body[j1:j2]}
        changed = sorted(changed)
        row["changed_top_level"] = changed
        allow = ALLOWED.get(f.name, set())
        bad = [k for k in changed if "*" not in allow and k not in allow]
        if not allow or bad:
            rep["violations"].append({"file": f.name, "outside_allowed": bad or changed})
        if f.name == "sim_w13_full_cycle.py":
            ra = next(n for n in A.body if getattr(n, "name", None) == "run")
            rb = next(n for n in B.body if getattr(n, "name", None) == "run")
            fa = {getattr(n, "name", None): ast.dump(n) for n in ra.body if isinstance(n, ast.FunctionDef)}
            fb = {getattr(n, "name", None): ast.dump(n) for n in rb.body if isinstance(n, ast.FunctionDef)}
            row["run_inner_functions_changed"] = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
            row["run_stmt_count"] = {"rev32": len(ra.body), "rev34": len(rb.body)}
            # 물리·분류·기록 핵심 함수는 바뀌면 안 된다
            frozen_inner = ["servo", "q_from_pose", "classify", "inv_counts", "_int_steps", "flush",
                            "record_particle_frame", "decision", "sample", "step", "record_t0", "enter",
                            "joint_move", "z_move", "fine_on", "dwell", "align_move", "resume_fk_here",
                            "maybe_stop", "spheres", "actual_pose"]
            row["frozen_inner_functions_unchanged"] = {k: fa.get(k) == fb.get(k) for k in frozen_inner}
            if not all(row["frozen_inner_functions_unchanged"].values()):
                rep["violations"].append({"file": f.name, "frozen_inner_changed":
                                          [k for k, v in row["frozen_inner_functions_unchanged"].items() if not v]})
    rep["files"][f.name] = row

rep["criteria_json_sha256"] = {"rev32": sha(R32 / "criteria.json"), "rev34": sha(R34 / "criteria.json")}
rep["criteria_identical"] = rep["criteria_json_sha256"]["rev32"] == rep["criteria_json_sha256"]["rev34"]
p32 = json.load(open(R32 / "params_w13.json"))
p25 = json.load(open(R34 / "params_w25.json"))
chg = sorted(k for k in p32 if p25.get(k) != p32[k])
rep["params_w25_changed_rev32_keys"] = {k: {"rev32": p32[k], "rev34": p25.get(k)} for k in chg}
rep["params_w25_new_keys"] = sorted(k for k in p25 if k not in p32)
rep["params_changes_within_declared_list"] = set(chg) <= PARAM_KEYS_CHANGED_ALLOWED
rep["physics_keys_unchanged"] = {k: p25.get(k) == p32[k] for k in (
    "E_pa", "nu", "CoR", "mu", "Crr", "particle_density_kg_m3", "E_mesh_pa", "timestep_s", "dt_sync_descend_s",
    "dt_sync_close_s", "diag_fine_sync_s", "servo_torque_Nm", "servo_torque_fraction", "close_deg_s",
    "door_pinch_guard_N", "arm_force_max_N", "particle_frame_dt_s", "settlement_frame_dt_s", "settlement_window_s",
    "classify_margin_mm", "transport_speed_mm_s", "plunge_mm", "descend_mm_s")}
ok = (not rep["violations"] and rep["criteria_identical"] and rep["params_changes_within_declared_list"]
      and all(rep["physics_keys_unchanged"].values()))
rep["verdict"] = "PASS" if ok else "FAIL"
print(json.dumps(rep, ensure_ascii=False, indent=2))
sys.exit(0 if ok else 1)
