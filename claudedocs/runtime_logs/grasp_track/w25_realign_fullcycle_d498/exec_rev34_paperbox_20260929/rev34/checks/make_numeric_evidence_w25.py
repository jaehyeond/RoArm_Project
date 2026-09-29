"""rev34 도메인용 수치 증거(numeric_inputs) 생성 — CPU 정적 재구성, DEME 초기화 0.

입력 = CPU 스텁 드라이런 결과 JSON(`engine.domain_x_m/…` — 러너가 DEME 에 넘길 도메인과 같은 코드로 계산된 값).
감사 도구 `recover_deme_lattice.reconstruct()` (W13 audit, sha 고정)를 그대로 부른다. 그 도구가 인증 분기
밖이라며 예외를 내면 REUSABLE_FOR_REV11=false 로 쓰고(fail-closed), 러너는 그 증거를 거부한다.

usage: python make_numeric_evidence_w25.py <stub result json> <out json> [--label TEMP]
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

AUDIT = Path("/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/"
             "w13_full_cycle_d484/resume_20260913/audit")
TOOL = AUDIT / "recover_deme_lattice.py"
TOOL_SHA = "931a518df75877942370dd552331e5967a237d11c4c49e674a985c23f3e81914"   # rev32 numeric_inputs.json 의 expected_sha256


def main(src_json, out_json, label):
    tool_sha = hashlib.sha256(TOOL.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("recover_deme_lattice", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = json.load(open(src_json))
    dom = {k: res["engine"][f"domain_{k}_m"] for k in ("x", "y", "z")}
    out = {"artifact": "W25A_NUMERIC_EVIDENCE_V1", "label": label,
           "purpose": "rev34 도메인의 설치본 격자 정적 재구성. 물리 파라미터가 아니다(params 불변).",
           "source_result_json": str(src_json),
           "source_result_sha256": hashlib.sha256(Path(src_json).read_bytes()).hexdigest(),
           "reproduction": {"script": str(TOOL), "sha256": tool_sha, "expected_sha256": TOOL_SHA,
                            "script_sha_matches_rev32_pin": tool_sha == TOOL_SHA},
           "rev11_recomputed_domain_m": dom}
    try:
        rec = mod.reconstruct(Path(src_json))
        out["reproduction"]["result"] = rec
        ok = (tool_sha == TOOL_SHA and rec["kind"] == "installed_binary_static_reconstruction_not_runtime_observation"
              and [list(v) for v in rec["user_domain_binary64_m"]] == [dom["x"], dom["y"], dom["z"]])
        out["REUSABLE_FOR_REV11"] = bool(ok)
        out["values_if_usable"] = {"l_m": rec["l_m"], "voxel_size_m": rec["voxel_size_m"],
                                   "absolute_coordinate_bound_m": rec["global_coordinate_abs_bound_m"],
                                   "axis_voxel_count_power2": rec["axis_voxel_count_power2"],
                                   "world_size_m": rec["world_size_m"], "world_max_m": rec["world_max_m"],
                                   "world_lbf_m": rec["target_box_min_binary32_m"],
                                   "kind": rec["kind"]} if ok else None
    except Exception as exc:                                    # noqa: BLE001 — 인증 분기 밖 = 거부
        out["REUSABLE_FOR_REV11"] = False
        out["values_if_usable"] = None
        out["reconstruct_error"] = repr(exc)
    out["rule"] = ("도메인이 binary64 로 동일하지 않거나 감사 스크립트가 인증 분기를 벗어나면 쓰지 않는다. "
                   "이 파일은 그 도메인에만 유효하다(다른 더미·배치로 바뀌면 다시 만든다).")
    out["non_claims"] = ["정적 재구성이며 솔버 초기화에서 관측한 값이 아니다.",
                         "수치 표현 범위일 뿐 충돌 없음·배출 성공의 증거가 아니다."]
    json.dump(out, open(out_json, "w"), ensure_ascii=False, indent=2)
    print(json.dumps({"REUSABLE_FOR_REV11": out["REUSABLE_FOR_REV11"], "domain": dom,
                      "values": out.get("values_if_usable"), "error": out.get("reconstruct_error")}, indent=1))
    return 0 if out["REUSABLE_FOR_REV11"] else 1


if __name__ == "__main__":
    lab = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "rev34"
    sys.exit(main(sys.argv[1], sys.argv[2], lab))
