"""rev32 -> rev33b AST 변경 범위 검사.

왜 diff 가 아니라 AST 인가
--------------------------
주석·문자열 diff 는 "물리를 안 건드렸다"를 증명하지 못한다. 이 검사는 **구문 트리**로
(a) 어떤 최상위 정의가 바뀌었는지, (b) run() 본문에서 몇 번째 문장이 바뀌었는지,
(c) 물리 파라미터·제어 프리미티브·보호선·분류 판정식 함수가 **문자 그대로 그대로인지**를 본다.

rev33b 가 허용하는 변경 범위 (이 밖이면 FAIL) — rev33 과 같은 목록이며, 1 번 문장의 **값 출처**만
rev33b 에서 바뀐다(코드 상수 -> 이 revision 의 criteria.json). 문장 머리는 그대로 `settle_pf_dt =` 다:
  1. run() 안 지역변수 1개 추가      : settle_pf_dt (옵션, 기본 None. rev33b 는 criteria 에서 읽음)
  2. record_particle_frame()         : 다음 저장 시각 계산에 _pf_dt_now 도입 (None 이면 pf_dt 와 동일)
  3. np.savez_compressed(...) 문장    : metadata_json 선언 추가 (IG.semantics_metadata / 시간·epsilon 필드)
  4. flush(...) 호출 문장             : run_state 변수로 분리 (인자 값 동일)
  5. run() 끝에 manifest.json 기록 문장 추가
  6. __main__ argparse 에 --settle-window-frame-dt-s 추가 (rev33b 는 --criteria 도 같이 추가.
     게이트가 세는 것은 여전히 settle-window-frame-dt-s 1건이다)

불변 증명 대상(문자 그대로 동일해야 함): step/sample/joint_move/z_move/door_move/dwell/align_move/
enter/record_t0/decision/maybe_stop/classify 와 모듈 최상위 상수 전부.
"""
import ast, json, sys
from pathlib import Path

R32 = Path("/home/cgxr/orca/workspaces/RoArm_Project/w16-profiling/claudedocs/runtime_logs/"
           "grasp_track/w16_profile_d486/rev32/src/sim_w13_full_cycle.py")
R33 = Path(__file__).resolve().parent.parent / "src" / "sim_w13_full_cycle.py"   # rev33b

FROZEN_FUNCS = ["step", "sample", "joint_move", "z_move", "door_move", "dwell", "align_move",
                "enter", "record_t0", "decision", "maybe_stop", "classify", "actual_pose",
                "pose_of", "fine_on", "flush", "resume_fk_here", "q_from_pose"]

a, b = ast.parse(R32.read_text()), ast.parse(R33.read_text())
name = lambda n: getattr(n, "name", type(n).__name__)
top_a, top_b = [name(n) for n in a.body], [name(n) for n in b.body]
rep = {"artifact": "W20_REV33B_AST_SCOPE_CHECK_V1", "rev32": str(R32), "rev33b": str(R33),
       "top_level_names_equal": top_a == top_b, "top_level_changed": []}
assert top_a == top_b, (top_a, top_b)

for na, nb in zip(a.body, b.body):
    if ast.dump(na) != ast.dump(nb):
        rep["top_level_changed"].append(name(na))

run_a = next(n for n in a.body if getattr(n, "name", None) == "run")
run_b = next(n for n in b.body if getattr(n, "name", None) == "run")
rep["run_stmt_count"] = {"rev32": len(run_a.body), "rev33b": len(run_b.body)}

# 문장 단위 정렬 대조. **삽입**과 **치환**을 섞지 않는다 — 집합 차집합을 쓰면 치환 1건이
# 삽입 1 + 삭제 1 로 잘못 세어져 범위 판정이 무의미해진다. difflib 로 순서를 맞춘다.
import difflib
dump_a = [ast.dump(s) for s in run_a.body]
dump_b = [ast.dump(s) for s in run_b.body]
head = lambda s: ast.unparse(s).splitlines()[0][:120]
ins, dele, repl = [], [], []
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=dump_a, b=dump_b, autojunk=False).get_opcodes():
    if tag == "insert":
        ins += [head(s) for s in run_b.body[j1:j2]]
    elif tag == "delete":
        dele += [head(s) for s in run_a.body[i1:i2]]
    elif tag == "replace":
        repl += [{"rev32": head(x), "rev33b": head(y)}
                 for x, y in zip(run_a.body[i1:i2], run_b.body[j1:j2])]
        ins += [head(s) for s in run_b.body[j1 + (i2 - i1):j2]]
        dele += [head(s) for s in run_a.body[i1 + (j2 - j1):i2]]
rep["run_inserted_stmts"], rep["run_deleted_stmts"], rep["run_replaced_stmts"] = ins, dele, repl

# 동결 함수 대조 (run() 안의 중첩 def 포함, 이름으로 찾는다)
def by_name(tree):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(n.name, []).append(n)
    return out

fa, fb = by_name(a), by_name(b)
frozen_ok, frozen_detail = True, {}
for fn in FROZEN_FUNCS:
    if fn not in fa and fn not in fb:
        frozen_detail[fn] = "absent_in_both"
        continue
    da = [ast.dump(x) for x in fa.get(fn, [])]
    db = [ast.dump(x) for x in fb.get(fn, [])]
    same = da == db
    frozen_detail[fn] = "IDENTICAL" if same else "CHANGED"
    frozen_ok &= same
rep["frozen_functions"] = frozen_detail
rep["frozen_functions_all_identical"] = bool(frozen_ok)

# record_particle_frame 은 바뀌어야 하는 유일한 저장-경로 함수다. 무엇이 바뀌었는지 남긴다.
rpf_a = fa["record_particle_frame"][0]
rpf_b = fb["record_particle_frame"][0]
sa = {ast.dump(s): ast.unparse(s) for s in ast.walk(rpf_a) if isinstance(s, ast.stmt)}
sb = {ast.dump(s): ast.unparse(s) for s in ast.walk(rpf_b) if isinstance(s, ast.stmt)}
rep["record_particle_frame_added"] = [v for k, v in sb.items() if k not in sa]
rep["record_particle_frame_removed"] = [v for k, v in sa.items() if k not in sb]

# 모듈 최상위 상수 (PHASES/INV_NAMES/DECISIONS/PHASE_CODE 등) 불변
const_a = {ast.unparse(n) for n in a.body if isinstance(n, (ast.Assign, ast.AnnAssign))}
const_b = {ast.unparse(n) for n in b.body if isinstance(n, (ast.Assign, ast.AnnAssign))}
rep["module_constants_identical"] = const_a == const_b
rep["module_constants_diff"] = sorted(const_a ^ const_b)

# 새 CLI 인자
main_b = [n for n in b.body if isinstance(n, ast.If)][-1]
rep["new_cli_args"] = [ast.unparse(n)[:90] for n in ast.walk(main_b)
                       if isinstance(n, ast.Call) and ast.unparse(n).startswith("ap.add_argument")
                       and "settle-window-frame-dt-s" in ast.unparse(n)]

# 참고용(게이트 아님): rev33b 가 __main__ 에 가진 전체 인자 목록. --criteria 추가를 눈으로 확인한다.
rep["all_cli_args_rev33b"] = [ast.unparse(n).split("'")[1] for n in ast.walk(main_b)
                              if isinstance(n, ast.Call) and ast.unparse(n).startswith("ap.add_argument")]

# 허용된 치환 3건만: (1) record_particle_frame 정의, (2) savez_compressed 호출, (3) flush 호출.
# flush 호출은 인자식을 run_state 변수로 **분리**했으므로 rev32 쪽 머리와 rev33 쪽 머리가 다르다.
ALLOWED_REPL = [("def record_particle_frame(", "def record_particle_frame("),
                ("np.savez_compressed(", "np.savez_compressed("),
                ("flush(", "run_state =")]
repl_ok = (len(rep["run_replaced_stmts"]) == len(ALLOWED_REPL)
           and all(r["rev32"].startswith(k32) and r["rev33b"].startswith(k33)
                   for r, (k32, k33) in zip(rep["run_replaced_stmts"], ALLOWED_REPL)))
# 허용된 삽입 5건: 옵션 지역변수 · 분리된 flush 호출 · manifest 경로/목록/기록
ALLOWED_INS = ["settle_pf_dt =", "flush(run_state)", "_man_path =", "_man_files =", "json.dump("]
ins_ok = (len(rep["run_inserted_stmts"]) == len(ALLOWED_INS)
          and all(s.startswith(k) for s, k in zip(rep["run_inserted_stmts"], ALLOWED_INS)))
# record_particle_frame 안에서 실제로 바뀐 문장은 next_pf 계산 1줄 + _pf_dt_now 도입 1줄뿐이다
# (함수 정의 노드 자체가 통째로 같이 잡히므로 def 로 시작하는 항목은 빼고 센다).
rpf_add = [s for s in rep["record_particle_frame_added"] if not s.startswith("def ")]
rpf_del = [s for s in rep["record_particle_frame_removed"] if not s.startswith("def ")]
rep["record_particle_frame_body_added"] = rpf_add
rep["record_particle_frame_body_removed"] = rpf_del
rpf_ok = (len(rpf_add) == 2 and len(rpf_del) == 1
          and rpf_add[0].startswith("_pf_dt_now =")
          and rpf_add[1].startswith("state['next_pf'] =")
          and rpf_del[0].startswith("state['next_pf'] ="))

ok = (rep["top_level_names_equal"]
      and set(rep["top_level_changed"]) == {"run", "If"}
      and rep["frozen_functions_all_identical"]
      and rep["module_constants_identical"]
      and rep["run_deleted_stmts"] == []
      and ins_ok and repl_ok and rpf_ok
      and len(rep["new_cli_args"]) == 1)
rep["gate"] = {"top_level_changed_is_run_and_main": set(rep["top_level_changed"]) == {"run", "If"},
               "frozen_functions_all_identical": rep["frozen_functions_all_identical"],
               "module_constants_identical": rep["module_constants_identical"],
               "no_deleted_run_stmts": rep["run_deleted_stmts"] == [],
               "inserted_stmts_as_expected": ins_ok,
               "replaced_stmts_as_expected": repl_ok,
               "record_particle_frame_change_is_next_pf_only": rpf_ok,
               "one_new_cli_arg": len(rep["new_cli_args"]) == 1}
rep["verdict"] = "PASS" if ok else "FAIL"
rep["expected"] = {"top_level_changed": ["run", "If"], "run_inserted_stmts": ALLOWED_INS,
                   "run_replaced_stmts": ALLOWED_REPL, "run_deleted_stmts": 0,
                   "frozen_functions_all_identical": True, "module_constants_identical": True,
                   "new_cli_args": 1}
print(json.dumps(rep, ensure_ascii=False, indent=2))
sys.exit(0 if ok else 1)
