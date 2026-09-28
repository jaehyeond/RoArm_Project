#!/usr/bin/env python3
"""rev30 파생 산출 v2 — ERRATUM_04(바닥=받침면) 규약으로 동결 W13 run_01 원자료를 재분류한다.
원시 좌표/ID/시간 무수정, 새 물리 0. run_01 의 옛 규약 verdict 는 소급 변경하지 않는다.
rev29 파생(strict) 과 기록(rev28) 둘 다에 대한 차이를 남긴다."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent                       # rev30/src
REV29_SRC = HERE.parent.parent / "rev29" / "src"


def import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D29 = import_from(REV29_SRC / "derive_repaired_raw.py", "derive_repaired_raw_rev29")   # helpers only
IG30 = import_from(HERE / "inventory_geometry.py", "inventory_geometry_rev30")
LABELS = list(IG30.INV_NAMES)
AUDIT_JSON = Path("/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/"
                  "w13_full_cycle_d484/resume_20260913/audit/REV28_PRODUCTION_PARTIAL_RAW_AUDIT_01.json")


def counts(code):
    return {n: int(v) for n, v in zip(LABELS, np.bincount(np.asarray(code).astype(int), minlength=len(LABELS)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(D29.DEFAULT_RAW))
    ap.add_argument("--meta", default=str(D29.DEFAULT_META))
    ap.add_argument("--pile", default=str(D29.DEFAULT_PILE))
    ap.add_argument("--rev29-derived", default=str(HERE.parent.parent / "derived" / "w13_cycle_seed460_rev29_derived.npz"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hashes = {"raw": D29.sha256_file(a.raw), "meta": D29.sha256_file(a.meta), "pile": D29.sha256_file(a.pile),
              "rev29_derived": D29.sha256_file(a.rev29_derived)}
    for k in ("raw", "meta", "pile"):
        if hashes[k] != D29.EXPECTED_SHA256[k]:
            raise SystemExit(f"입력 해시 불일치 {k}")
    assert IG30.REVISION == "rev31" and IG30.FLOOR_RULE == "support_surface_v2: min(center_z - r) > floor - margin" and IG30.CONTRACT_VERSION == "RAW_SCHEMA_REQUIRED + ERRATUM_04"
    res = json.load(open(a.meta)); tpl = D29.load_template(a.pile)
    z = np.load(a.raw, allow_pickle=False); meta = json.loads(str(z["metadata_json"]))
    cfg = D29.build_inv_cfg(res, z)
    rec = z["inventory_code"]; F, N = rec.shape
    d29 = np.load(a.rev29_derived, allow_pickle=False)
    code29 = d29["inventory_code_rev29_strict"]
    assert code29.shape == rec.shape and np.array_equal(d29["inventory_code_recorded_rev28"], rec)

    code30 = np.empty((F, N), np.int8)
    for fi in range(F):
        S, Rr, speed, p_tool, R_tool = D29.production_frame_inputs(z, fi, tpl)
        code30[fi] = IG30.classify_spheres(S, Rr, speed, p_tool, R_tool, cfg)
        if fi % 40 == 0:
            print(f"  frame {fi}/{F} rev30={counts(code30[fi])}", flush=True)
    per_rec = np.stack([np.bincount(r.astype(int), minlength=6) for r in rec]).astype(np.int64)
    per_29 = np.stack([np.bincount(r.astype(int), minlength=6) for r in code29]).astype(np.int64)
    per_30 = np.stack([np.bincount(r.astype(int), minlength=6) for r in code30]).astype(np.int64)
    assert (per_30.sum(1) == N).all()
    tags = [str(t) for t in z["decision_tags"]]
    pf_reclose = int(z["decision_particle_frame_index"][tags.index("reclose_end")])
    cohort = np.flatnonzero(rec[pf_reclose] == LABELS.index("tool_residual"))
    cand = []
    if AUDIT_JSON.exists():
        aud = json.load(open(AUDIT_JSON))
        for f in aud["findings"]:
            if f["name"] == "raw_bin_delivery_interval_and_settlement_limit":
                cand = [int(v) for v in f["evidence"]["possible_candidate_ids"]]
    S0, Rr0, _, _, _ = D29.production_frame_inputs(z, 0, tpl)
    box = cfg["box_bounds_m"]; m = cfg["margin_m"]
    ce = {"particle_frame_row": 0, "particle_id": 8, "recorded_label": LABELS[int(rec[0, 8])],
          "rev29_label": LABELS[int(code29[0, 8])], "rev30_label": LABELS[int(code30[0, 8])],
          "sphere_surface_min_z_m": float((S0[8, :, 2] - Rr0[8]).min()),
          "v2_floor_predicate_floor_minus_margin": bool(((S0[8, :, 2] - Rr0[8]) > box[2, 0] - m).all())}
    bin_any = int((code30 == LABELS.index("receiving_bin")).any(0).sum())
    prov = {"artifact": "W14_REV31_DERIVED_V2_SUPPORT_FLOOR", "derived_not_raw": True,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "contract": IG30.CONTRACT_VERSION, "floor_rule": IG30.FLOOR_RULE, "source_floor_rule": IG30.SOURCE_FLOOR_RULE,
            "erratum_draft": str(HERE.parent.parent / "contract" / "RAW_SCHEMA_REQUIRED_ERRATUM_04_DRAFT.md"),
            "erratum_registered": "/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/RAW_SCHEMA_REQUIRED_ERRATUM_04.md",
            "input_sha256": hashes, "rev31_inventory_geometry_sha256": D29.sha256_file(HERE / "inventory_geometry.py"),
            "semantics": IG30.semantics_metadata(len(tpl["sphere_radii_m"])),
            "non_claims": ["run_01 원자료의 옛 규약 FAIL 은 그대로다(소급 PASS 아님).",
                           "새 물리·GPU·렌더 0. 같은 원시 좌표에 대한 v2 규약 재분류다.",
                           "receiving_bin 라벨은 정착 판정이 아니다(settlement window 별도)."]}
    np.savez_compressed(out / "w13_cycle_seed460_rev31_derived_v2.npz",
                        inventory_code_rev30_support=code30, inventory_code_rev29_strict=code29,
                        inventory_code_recorded_rev28=rec.astype(np.int8), inventory_labels=np.asarray(LABELS),
                        transition_sync_index_phase_only=d29["transition_sync_index_phase_only"],
                        particle_frame_row=z["particle_frame_row"], particle_frame_sync_index=z["particle_frame_sync_index"],
                        particle_frame_t_s=z["particle_frame_t_s"], particle_ids=z["particle_ids"],
                        per_frame_counts_recorded=per_rec, per_frame_counts_rev29=per_29, per_frame_counts_rev30=per_30,
                        cohort_reclose_tool_ids_recorded=cohort, bin_candidate_ids_audit=np.asarray(cand, np.int64),
                        provenance_json=np.asarray(json.dumps(prov, ensure_ascii=False)))
    man = dict(prov)
    man.update({"wall_s": round(time.time() - t0, 3), "n_frames": int(F), "n_particles": int(N),
                "output_npz": str(out / "w13_cycle_seed460_rev31_derived_v2.npz"),
                "output_npz_sha256": D29.sha256_file(out / "w13_cycle_seed460_rev31_derived_v2.npz"),
                "array_sha256": {"inventory_code_rev30_support": D29.array_sha(code30),
                                 "inventory_code_rev29_strict": D29.array_sha(code29),
                                 "inventory_code_recorded": D29.array_sha(rec)},
                "inventory": {"rev30_vs_recorded_mismatch": int(np.count_nonzero(code30 != rec)),
                              "rev30_vs_rev29_mismatch": int(np.count_nonzero(code30 != code29)),
                              "rev29_vs_recorded_mismatch": int(np.count_nonzero(code29 != rec)),
                              "final_recorded": counts(rec[-1]), "final_rev29": counts(code29[-1]), "final_rev30": counts(code30[-1]),
                              "reclose_end_frame": pf_reclose, "reclose_rev30": counts(code30[pf_reclose]),
                              "cohort_n": int(len(cohort)), "cohort_final_rev30": counts(code30[-1, cohort]),
                              "cohort_final_rev29": counts(code29[-1, cohort]), "cohort_final_recorded": counts(rec[-1, cohort]),
                              "bin_candidates_audit_n": len(cand),
                              "bin_candidates_final_rev30": {str(i): LABELS[int(code30[-1, i])] for i in cand},
                              "bin_candidates_final_rev29": {str(i): LABELS[int(code29[-1, i])] for i in cand},
                              "n_ids_ever_receiving_bin_rev30": bin_any,
                              "max_receiving_bin_per_frame_rev30": int(per_30[:, 1].max()),
                              "frame_of_max_receiving_bin_rev30": int(per_30[:, 1].argmax()),
                              "particle_mass_g": float(z["particle_mass_kg"]) * 1000.0},
                "counterexample_pf0_id8": ce})
    (out / "DERIVED_V2_MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps({k: man[k] for k in ("inventory", "counterexample_pf0_id8", "wall_s")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
