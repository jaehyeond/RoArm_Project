"""W3b ① 귀속 표 + 게이트 JSON — 셀 결과(cell_*.json / *.partial.json / log_*.txt) 를 읽어 쓴다 (D476: 결과로 판정)."""
import json, hashlib, re
from pathlib import Path
B = Path(__file__).resolve().parent; REPO = B.parents[6]
sha16 = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
CELLS = [  # (셀, 무엇을 바꿨나, 후보)
    ("cut_base", "W3 셸 그대로(파팅면 법선 임의 → 4/12 뒤집힘)", "기준(결함)"),
    ("cut_fixnorm", "파팅면 법선을 바깥(+X)으로 수정", "(e) 수정"),
    ("e_flip_part", "수정판에서 파팅면 12면만 일부러 안쪽으로", "(e) 재현"),
    ("e_flip_all", "수정판에서 전체 법선 반전", "(e) 극단"),
    ("a_wall0", "벽 1.6 / 캡 2.0 mm (입자 반경 2.08 보다 얇음)", "(a)"),
    ("a_wall6", "벽 7.6 / 캡 8.0 mm (지름 이상)", "(a)"),
    ("b_vz5", "하강 5 mm/s (스텝당 관입 1/5), 잠김 6", "(b)"),
    ("b_dt5e6", "적분 dt 5e-6 (절반)", "(b)"),
    ("c_subdiv2", "삼각형 최대 변 2 mm 로 세분화(3.6k 삼각형)", "(c)"),
    ("d_outer_only", "안쪽 면·캡 제거(바깥 면+파팅면만)", "(d)"),
    ("d_no_caps", "캡 제거(안·바깥 면+파팅면)", "(d)"),
    ("f_cd5", "CD 주기 5", "참고"),
    ("f_E1e8", "입자 E 1e8 · dt 2e-6", "참고(W3 stiff 재현)"),
]
rows = []
for cell, what, cand in CELLS:
    full, part, log = B / f"cell_{cell}.json", B / f"cell_{cell}.partial.json", B / f"log_{cell}.txt"
    d = json.load(open(full)) if full.exists() else (json.load(open(part)) if part.exists() else None)
    txt = open(log).read() if log.exists() else ""
    m = re.search(r"max velocity is ([0-9.]+)", txt); rc = re.findall(r"^rc=(\d+)", txt, re.M)
    r = {"cell": cell, "changed": what, "candidate": cand, "source": full.name if full.exists() else (part.name if part.exists() else None), "rc": rc[-1] if rc else None}
    if d:
        rs = d["rows"]
        r.update({"insert_reached_mm": (rs[-1]["insert_mm"] if rs else None), "insert_target_mm": d.get("insert_target_mm"),
                  "single_max_N": round(max((x["single_N"] for x in rs), default=0.0), 3), "v_max_m_s": round(max((x["v_max"] for x in rs), default=0.0), 3),
                  "wall_s": d.get("wall_s"), "n_particles": d.get("n_particles"), "n_tri": d.get("n_tri")})
        r["diverged"] = bool(d.get("diverged")) or (m is not None)
        r["abort_velocity_m_s"] = float(m.group(1)) if m else None
    else:
        r.update({"diverged": None, "note": "결과 없음"})
    rows.append(r)
verdict = {
    "root_cause": "(e) 파팅면(x=8.1 평면) 삼각형 법선이 일부 안쪽 → DEME triangle_sphere_CD(양면) 가 그 면 발자국 안·뒤쪽의 정상 입자에 관입 r+|h| (2~4 mm, 30~80 N) 부여 → 입자 1 ms 안에 수십 m/s",
    "evidence": ["cut_base(결함) 8.8 mm 에서 21 m/s ↔ cut_fixnorm(수정) 20 mm 완주 0.37 m/s (같은 절편·물성·궤적)",
                 "e_flip_part(수정판에 결함 재주입) 11 mm 에서 DEME abort 63 m/s · e_flip_all 4.5 mm 에서 81 N",
                 "d_outer_only / d_no_caps: 면을 빼서 셸이 열리면 입자가 면 뒤쪽 발자국에 들어가 같은 기작으로 발산(17.7 / 19.4 mm)",
                 "a·b·c·f: 벽 두께 1.6→7.6, 하강 5→25 mm/s, dt 1e-5→5e-6, 삼각형 세분화(4.5 mm 까지), CD 5, E 1e8 전부 발산 0 → 무관",
                 "커널 원문 share/DEME/kernel/DEMCollisionKernels.cu triangle_sphere_CD: depth = h − radius (h<0 이면 r+|h|)"],
    "excluded": {"(a) 벽 두께": "a_wall0 완주(단일 0.145 N)", "(b) 스텝당 관입": "b_vz5·b_dt5e6 완주", "(c) 삼각형 크기": "c_subdiv2 4.5 mm 까지 이상 없음(느려서 중단)",
                 "(d) 문·보울 이중 접촉": "재현 셀은 고정 셸 1매(문 없음)에서 터졌으므로 문과 무관; 면 제거 셀의 발산은 (e) 기작"},
    "fix_applied": "sim_deme_scoop_s1.py half_bowl: 파팅면 outward = (−side, 0, 0) — 고정 +X, 문 −X (검증: 파팅 12면/반쪽 전부 일관)",
}
out = {"artifact": "W3B_DIVERGE_ATTRIBUTION", "date": "2026-09-10", "harness": "sim_deme_s1_diverge_min.py",
       "inputs_sha16": {str(p): sha16(p) for p in [REPO / "claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz",
                                                    REPO / "sim_deme_s1_diverge_min.py", REPO / "sim_deme_scoop_s1.py"]},
       "cells": rows, "verdict": verdict}
json.dump(out, open(B / "attribution_b_diverge.json", "w"), ensure_ascii=False, indent=1)
print("| 셀 | 바꾼 것 | 후보 | 잠김 도달/목표 mm | 발산 | 단일접촉 최대 N | 입자 최대 m/s | 벽시계 s |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    st = "완주" if not r.get("diverged") else ("🔴 발산" + (" (abort %s m/s)" % r["abort_velocity_m_s"] if r.get("abort_velocity_m_s") else ""))
    print("| %s | %s | %s | %s / %s | %s | %s | %s | %s |" % (r["cell"], r["changed"], r["candidate"], r.get("insert_reached_mm"),
          r.get("insert_target_mm"), st, r.get("single_max_N"), r.get("v_max_m_s"), r.get("wall_s")))
