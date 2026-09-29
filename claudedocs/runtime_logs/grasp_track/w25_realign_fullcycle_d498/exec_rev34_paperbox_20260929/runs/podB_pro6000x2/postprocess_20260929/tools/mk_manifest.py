#!/usr/bin/env python3
"""출력 폴더 매니페스트 — 경로·바이트·sha256 + 항목별 요약. 원자료 무수정 증거 포함."""
import hashlib, json, time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


files = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name != "manifest.json":
        files[p.relative_to(OUT).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha(p)}

pre = json.load(open(OUT / "PRESERVATION_BEFORE.json"))
post = json.load(open(OUT / "PRESERVATION_AFTER.json"))
man = json.load(open(OUT / "derived_v2_w25/DERIVED_V2_MANIFEST.json"))
res = json.load(open(OUT / "tests/RESULTS_allframes_w25.json"))
sw = json.load(open(OUT / "derived_v2_w25/SETTLEMENT_WINDOW_RECOMPUTE.json"))
sc = json.load(open(OUT / "schema_check/RAW_SCHEMA_CHECK_W25.json"))
cc = json.load(open(OUT / "schema_check/CRITERIA_CHECK_W25.json"))
pin = json.load(open(OUT / "rev34_copy/REVISION_PIN.json"))

out = {
    "artifact": "W25_PODB_POSTPROCESS_MANIFEST_V1",
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "role": "raw-accountant", "cpu_only": True, "gpu_used": False, "new_physics_runs": 0,
    "raw_write_operations": 0,
    "subject_run": str(OUT.parent / "run_01"),
    "preservation": {"before": pre["verdict"], "after": post["verdict"],
                     "n_files": pre["n_observed_files"],
                     "n_mismatch_before": pre["n_mismatch"], "n_mismatch_after": post["n_mismatch"]},
    "revision_pin": {"n_files": pin["n_files"], "n_bad": pin["n_bad"],
                     "formula_identity": pin["formula_identity"]},
    "P2_reclassification": {"n_frames": man["n_frames"], "n_particles": man["n_particles"],
                            "n_cells": man["inventory"]["n_cells"],
                            "rev34_vs_recorded_mismatch": man["inventory"]["rev34_vs_recorded_mismatch"],
                            "rev29_vs_recorded_mismatch": man["inventory"]["rev29_vs_recorded_mismatch"],
                            "mismatch_by_class_recorded": man["inventory"]["mismatch_by_class_recorded"],
                            "final_json_equals_raw": man["inventory"]["final_json_equals_raw"],
                            "wall_s": man["wall_s"]},
    "P3_independent": {"mismatch_cells": res["cases"]["independent_vs_production_mismatch_cells"],
                       "n_cells": res["cases"]["n_cells"],
                       "checker_sha256": res["independent_checker_sha256"],
                       "checker_imports": res["independent_checker_imports"],
                       "wall_s": res["wall_s"]},
    "P4_settlement": {"n_items": sw["n_items"], "n_fail": sw["n_fail"],
                      "failures": [i["name"] for i in sw["items"] if i["verdict"] == "FAIL"],
                      "definite_n": sw["recomputed_recorded_labels"]["definite_delivered_n"],
                      "possible_n": sw["recomputed_recorded_labels"]["possible_delivered_n"],
                      "cadence_ok": sw["recomputed_recorded_labels"]["cadence_ok"]},
    "P5_schema": {"n_items": sc["n_items"], "n_pass": sc["n_pass"], "n_fail": sc["n_fail"],
                  "failures": sc["failures"], "verdict": sc["verdict"]},
    "P6_criteria": {"n_items": cc["n_items"], "n_pass": cc["n_pass"], "n_fail": cc["n_fail"],
                    "n_not_checkable": cc["n_not_checkable"], "failures": cc["failures"],
                    "hard_fail_failures": cc["hard_fail_failures"]},
    "non_claims": [
        "이 폴더는 파생이다 — run_01 원자료의 어떤 바이트도 바꾸지 않았다(PRE==POST 증명).",
        "결론은 '생산 회계 재현 여부 + 항목별 PASS/FAIL' 까지다. '전체 사이클 성공' 선언이 아니다(D490).",
        "FAIL 항목을 사후 허용값으로 완화하지 않았다(D485).",
        "차이는 관측이며 원인으로 읽지 않는다(n>=3 필요).",
        "새 물리·GPU·DEME·Isaac·Rerun 실행 0."],
    "n_files": len(files), "files": files}
(OUT / "manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: out[k] for k in ("preservation", "revision_pin", "P2_reclassification", "P3_independent",
                                      "P4_settlement", "P5_schema", "P6_criteria", "n_files")},
                 ensure_ascii=False, indent=1))
