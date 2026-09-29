#!/usr/bin/env python3
"""rev34_copy = rev34(동결) + W14 rev29(헬퍼) 바이트 사본 증거. 식 변경 0 의 근거.
비교 3중: (a) 사본 sha == 원본 sha, (b) rev34 원본 sha == EXEC_PIN.json 기록 sha,
(c) rev34/src/inventory_geometry.py sha == W14 rev31 원본 sha(분류식 정본 동일 확인)."""
import hashlib, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
EX = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/"
          "w25_realign_fullcycle_d498/exec_rev34_paperbox_20260929")
W14 = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/"
           "w14_w13_raw_repair_d484/repair_20260916_01")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


pin = json.load(open(EX / "EXEC_PIN.json"))
exec_files = pin["files"]
rows, bad = {}, []
for cp in sorted((OUT / "rev34_copy").rglob("*")):
    if not cp.is_file():
        continue
    rel = cp.relative_to(OUT / "rev34_copy").as_posix()
    if rel.startswith("rev29/"):
        src = W14 / rel[len("rev29/"):].replace("src/", "rev29/src/", 1) if False else W14 / rel
    elif rel.startswith("rev34/"):
        src = EX / rel
    else:
        src = EX / rel
    s_copy = sha(cp)
    s_src = sha(src) if src.exists() else None
    ex_rel = rel if rel in exec_files else ("rev34/" + rel.split("rev34/", 1)[1] if "rev34/" in rel else None)
    ex_entry = exec_files.get(rel)
    s_exec = (ex_entry.get("sha256") if isinstance(ex_entry, dict) else ex_entry) if ex_entry else None
    row = {"copy": s_copy, "source_path": str(src), "source": s_src,
           "copy_equals_source": (s_copy == s_src) if s_src else None,
           "exec_pin": s_exec, "copy_equals_exec_pin": (s_copy == s_exec) if s_exec else None}
    rows[rel] = row
    if row["copy_equals_source"] is False or row["copy_equals_exec_pin"] is False:
        bad.append(rel)

ig34 = sha(OUT / "rev34_copy/rev34/src/inventory_geometry.py")
ig31 = sha(W14 / "rev31/src/inventory_geometry.py")
ig29 = sha(OUT / "rev34_copy/rev29/src/inventory_geometry.py")
main_src = Path("/home/cgxr/Documents/Robotics/RoArm_Project/sim_deme_scoop_s1.py")
res = {"artifact": "W25_POSTPROCESS_REVISION_PIN_V1",
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "purpose": "rev34_copy = 동결 rev34 + W14 rev29 헬퍼의 바이트 사본. 분류/전환 식 변경 0 의 증거.",
       "source_rev34": str(EX / "rev34"), "source_w14_rev29_src": str(W14 / "rev29/src"),
       "exec_pin": str(EX / "EXEC_PIN.json"), "exec_pin_sha256": sha(EX / "EXEC_PIN.json"),
       "n_files": len(rows), "n_bad": len(bad), "bad": bad,
       "formula_identity": {
           "rev34_inventory_geometry_sha256": ig34,
           "w14_rev31_inventory_geometry_sha256": ig31,
           "rev34_equals_w14_rev31": ig34 == ig31,
           "rev29_inventory_geometry_sha256": ig29,
           "main_sim_deme_scoop_s1_sha256": sha(main_src),
           "main_sim_deme_scoop_s1_equals_w14_pin":
               sha(main_src) == "2e40f7ed279dad42794d156c3e5e823d6ce0d8cb770ec9aad1b9280a33a7e933"},
       "files": rows}
(OUT / "rev34_copy/REVISION_PIN.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: res[k] for k in ("n_files", "n_bad", "bad", "formula_identity")}, ensure_ascii=False, indent=1))
