"""rev31 -> rev32 AST 변경 범위 검사.

목적: 편집이 (a) SC 버퍼 선언, (b) sample() 본문, (c) savez 의 metadata_json 인자
세 곳에만 닿았는지 **구문 트리**로 확인한다. 주석·문자열 diff 가 아니라 실행되는 코드의 범위를 본다.
"""
import ast, json, sys
from pathlib import Path

R31 = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/"
           "w14_w13_raw_repair_d484/repair_20260916_01/rev31/src/sim_w13_full_cycle.py")
R32 = Path(__file__).resolve().parent.parent / "src" / "sim_w13_full_cycle.py"

a, b = ast.parse(R31.read_text()), ast.parse(R32.read_text())
name = lambda n: getattr(n, "name", type(n).__name__)
top_a = [name(n) for n in a.body]
top_b = [name(n) for n in b.body]
rep = {"top_level_names_equal": top_a == top_b, "top_level_changed": [], "run_body_changed": [],
       "sample_body_changed": False, "other_funcs_changed": []}
assert top_a == top_b, (top_a, top_b)

for na, nb in zip(a.body, b.body):
    if ast.dump(na) != ast.dump(nb):
        rep["top_level_changed"].append(name(na))

run_a = next(n for n in a.body if getattr(n, "name", None) == "run")
run_b = next(n for n in b.body if getattr(n, "name", None) == "run")
assert len(run_a.body) == len(run_b.body), "run() 문장 개수가 달라졌다"
for i, (sa, sb) in enumerate(zip(run_a.body, run_b.body)):
    if ast.dump(sa) != ast.dump(sb):
        rep["run_body_changed"].append({"index": i, "node": type(sa).__name__,
                                        "name": getattr(sa, "name", None),
                                        "lineno_rev31": sa.lineno, "lineno_rev32": sb.lineno})

# sample() 안에서도 SC.append 3줄만 늘었는지
sam_a = next(n for n in run_a.body if getattr(n, "name", None) == "sample")
sam_b = next(n for n in run_b.body if getattr(n, "name", None) == "sample")
rep["sample_body_changed"] = ast.dump(sam_a) != ast.dump(sam_b)
rep["sample_stmt_count"] = {"rev31": len(sam_a.body), "rev32": len(sam_b.body)}
added = [ast.unparse(s) for s in sam_b.body if ast.dump(s) not in {ast.dump(x) for x in sam_a.body}]
removed = [ast.unparse(s) for s in sam_a.body if ast.dump(s) not in {ast.dump(x) for x in sam_b.body}]
rep["sample_added_stmts"], rep["sample_removed_stmts"] = added, removed

# 판정
ok = (rep["top_level_changed"] == ["run"]
      and len(rep["run_body_changed"]) == 3
      and {c["node"] for c in rep["run_body_changed"]} == {"Assign", "FunctionDef", "Expr"}
      and removed == []
      and len(added) == 4)          # perf_counter 시작 + append 3
rep["verdict"] = "PASS" if ok else "FAIL"
print(json.dumps(rep, ensure_ascii=False, indent=2))
sys.exit(0 if ok else 1)
