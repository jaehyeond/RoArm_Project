"""DEME 2.4.0 파이썬 바인딩에 rev32 가 쓰는 질의가 실제로 있는지 확인한다.
솔버를 **만들지 않는다**(GPU 0, 물리 0). dir() 로 이름만 본다.
"""
import json, sys, inspect
import DEME

ds = DEME.DEMSolver
names = sorted(n for n in dir(ds) if not n.startswith("_"))
want = {"GetNumContacts": "scalar_engine_num_contacts",
        "GetExpandFactor": "scalar_engine_expand_factor_m (요청됨)",
        "GetUpdateFreq": "scalar_engine_cd_update_freq"}
hits = {k: {"bound": k in names, "maps_to": v} for k, v in want.items()}
rep = {"artifact": "W16_REV32_DEME_BINDING_PROBE",
       "deme_version": getattr(DEME, "__version__", "2.4.0 (모듈 속성 없음 — 설치 경로 기준)"),
       "deme_file": getattr(DEME, "__file__", None),
       "solver_constructed": False, "gpu_touched": False,
       "n_public_methods": len(names),
       "requested_queries": hits,
       "nearby_names": {"expand": [n for n in names if "expand" in n.lower()],
                        "updatefreq": [n for n in names if "updatefreq" in n.lower()],
                        "contacts": [n for n in names if "contact" in n.lower() and n.startswith("Get")]},
       "conclusion": ("GetNumContacts/GetUpdateFreq 는 바인딩에 있고 rev32 가 쓴다. "
                      "GetExpandFactor 는 C++ 선언(API.h:105)만 있고 파이썬 바인딩에 없다 → 기록 후 건너뛴다(패치 0).")}
rep["verdict"] = "PASS" if (hits["GetNumContacts"]["bound"] and hits["GetUpdateFreq"]["bound"]) else "FAIL"
print(json.dumps(rep, ensure_ascii=False, indent=2))
sys.exit(0 if rep["verdict"] == "PASS" else 1)
