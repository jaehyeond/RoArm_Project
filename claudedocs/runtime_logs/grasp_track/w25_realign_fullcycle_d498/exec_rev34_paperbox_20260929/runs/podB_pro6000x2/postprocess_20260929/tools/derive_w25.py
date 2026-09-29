#!/usr/bin/env python3
"""W25-A podB run_01 원자료 → rev29(strict floor) · rev31/rev34(ERRATUM_04 받침면) 파생 재분류.

식은 한 글자도 새로 쓰지 않는다. 분류/전개/전환 식은 전부 `rev34_copy` 의 **바이트 사본 모듈**에서 import 한다:
  · rev29/src/derive_repaired_raw.py  → build_inv_cfg · production_frame_inputs · array_sha · sha256_file
  · rev29/src/inventory_geometry.py   → classify_spheres (strict floor, rev29)
  · rev34/src/inventory_geometry.py   → classify_spheres (support floor; sha = W14 rev31 과 바이트 동일)
  · rev34/src/raw_transitions.py      → phase_only / legacy 전환 인덱스
이 드라이버가 하는 일은 (a) W13 하드코딩 경로/해시 대신 W25 경로·해시를 인자로 넘기고
(b) 275 프레임이 zip 을 한 번만 풀도록 배열을 미리 메모리에 올리고 (c) W25 회계표를 모으는 것뿐이다.
새 물리 0 · GPU 0 · 원자료 쓰기 0. (선례: W19 postprocess_20260918/tools/derive_w19.py)
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
COPY = HERE.parent / "rev34_copy"
REV29_SRC = COPY / "rev29/src"
REV34_SRC = COPY / "rev34/src"


def import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(REV29_SRC))
D29 = import_from(REV29_SRC / "derive_repaired_raw.py", "derive_repaired_raw_rev29")   # helpers only
IG29 = import_from(REV29_SRC / "inventory_geometry.py", "inventory_geometry_rev29")
IG34 = import_from(REV34_SRC / "inventory_geometry.py", "inventory_geometry_rev34")
RT = import_from(REV34_SRC / "raw_transitions.py", "raw_transitions_rev34")
LABELS = list(IG34.INV_NAMES)


class FrameView:
    """np.load NpzFile 은 키를 읽을 때마다 zip 전체를 다시 푼다(275프레임 × 3배열 = 수백 GB 재읽기).
    값은 바꾸지 않고 미리 메모리에 올린 배열을 같은 키로 돌려준다 — production_frame_inputs 입력 동일."""

    def __init__(self, z, keys):
        self._c = {k: np.asarray(z[k]) for k in keys}
        self._z = z

    def __getitem__(self, k):
        return self._c[k] if k in self._c else self._z[k]


def counts(code):
    return {n: int(v) for n, v in zip(LABELS, np.bincount(np.asarray(code).astype(int), minlength=len(LABELS)))}


def main():
    ap = argparse.ArgumentParser()
    for k in ("raw", "meta", "pile", "out", "expect-raw-sha256", "expect-pile-sha256", "expect-meta-sha256"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    hashes = {"raw": D29.sha256_file(a.raw), "meta": D29.sha256_file(a.meta), "pile": D29.sha256_file(a.pile),
              "main_sim_deme_scoop_s1": D29.sha256_file(D29.MAIN_REPO / "sim_deme_scoop_s1.py"),
              "rev29_inventory_geometry": D29.sha256_file(REV29_SRC / "inventory_geometry.py"),
              "rev34_inventory_geometry": D29.sha256_file(REV34_SRC / "inventory_geometry.py"),
              "rev34_raw_transitions": D29.sha256_file(REV34_SRC / "raw_transitions.py")}
    for key, exp in (("raw", a.expect_raw_sha256), ("meta", a.expect_meta_sha256), ("pile", a.expect_pile_sha256)):
        if hashes[key] != exp:
            raise SystemExit(f"{key} sha 불일치: {hashes[key]} != {exp}")
    if hashes["main_sim_deme_scoop_s1"] != D29.EXPECTED_SHA256["main_sim_deme_scoop_s1"]:
        raise SystemExit("동결 메인 소스(sim_deme_scoop_s1.py) 해시가 W14 pin 과 다르다")
    assert IG34.REVISION == "rev31" and IG34.CONTRACT_VERSION == "RAW_SCHEMA_REQUIRED + ERRATUM_04"
    assert IG34.FLOOR_RULE == "support_surface_v2: min(center_z - r) > floor - margin"

    res = json.load(open(a.meta))
    tpl = D29.load_template(a.pile)
    z0 = np.load(a.raw, allow_pickle=False)
    meta = json.loads(str(z0["metadata_json"]))
    cfg = D29.build_inv_cfg(res, z0)
    cfg_checks = {
        "v_settle_equals_meta_moving_threshold":
            abs(cfg["v_settle_m_s"] - float(meta["moving_threshold_m_s"])) < 1e-15,
        "margin_equals_meta_classify_margin":
            abs(cfg["margin_m"] - float(meta["classify_margin_m"])) < 1e-15,
        "raw_box_bounds_equals_w25_declared_tray":
            np.array_equal(np.asarray(z0["box_bounds_m"], float),
                           np.asarray(res["w25"]["tray"]["box_bounds_m"], float)),
        "inventory_labels_match": [str(v) for v in z0["inventory_labels"]] == LABELS,
        "spill_rest_z_equals_meta": abs(float(cfg["spill_rest_z_m"]) - float(meta["spill_rest_z_m"])) < 1e-15,
    }
    for k, v in cfg_checks.items():
        if not v:
            raise SystemExit(f"cfg 대조 실패: {k}")

    z = FrameView(z0, ["particle_pos_m", "particle_quat_xyzw", "particle_vel_m_s",
                       "particle_frame_sync_index", "tool_pos_m", "tool_quat_xyzw"])
    rec = np.asarray(z0["inventory_code"]); F, N = rec.shape

    # (1) 전환 인덱스 — 규약(phase-only) · rev28 legacy 재현 · 기록값
    trans_rec = np.asarray(z0["transition_sync_index"]).astype(np.int64)
    trans_phase = RT.phase_only_transition_indices(z0["sync_phase_code"])
    trans_legacy = RT.legacy_rev28_transition_indices(z0["sync_phase_code"], z0["sync_subphase"])

    # (2) 전 프레임 재분류 — rev29 strict, rev34(=rev31 식) support
    code29 = np.empty((F, N), np.int8)
    code34 = np.empty((F, N), np.int8)
    for fi in range(F):
        S, Rr, speed, p_tool, R_tool = D29.production_frame_inputs(z, fi, tpl)
        code29[fi] = IG29.classify_spheres(S, Rr, speed, p_tool, R_tool, cfg)
        code34[fi] = IG34.classify_spheres(S, Rr, speed, p_tool, R_tool, cfg)
        if fi % 25 == 0:
            print(f"  frame {fi}/{F} rec={counts(rec[fi])} rev34={counts(code34[fi])} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    per_rec = np.stack([np.bincount(r.astype(int), minlength=6) for r in rec]).astype(np.int64)
    per_29 = np.stack([np.bincount(r.astype(int), minlength=6) for r in code29]).astype(np.int64)
    per_34 = np.stack([np.bincount(r.astype(int), minlength=6) for r in code34]).astype(np.int64)
    assert (per_34.sum(1) == N).all() and (per_rec.sum(1) == N).all()

    # (3) 클래스별 불일치 — 기록 라벨 x 파생 라벨 혼동행렬
    conf34 = np.zeros((6, 6), np.int64); np.add.at(conf34, (rec.astype(int).ravel(), code34.astype(int).ravel()), 1)
    conf29 = np.zeros((6, 6), np.int64); np.add.at(conf29, (rec.astype(int).ravel(), code29.astype(int).ravel()), 1)
    per_frame_mismatch_34 = [int(v) for v in (code34 != rec).sum(1)]
    per_frame_mismatch_29 = [int(v) for v in (code29 != rec).sum(1)]

    # (4) 결정 태그별 재고 + 전이 행렬
    tags = [str(t) for t in z0["decision_tags"]]
    dpf = np.asarray(z0["decision_particle_frame_index"]).astype(int)
    per_tag = {t: {"particle_frame_row": int(dpf[i]), "sync_index": int(z0["decision_sync_index"][i]),
                   "t_s": float(z0["particle_frame_t_s"][dpf[i]]),
                   "recorded": counts(rec[dpf[i]]), "rev29": counts(code29[dpf[i]]), "rev34": counts(code34[dpf[i]]),
                   "rev34_vs_recorded_mismatch": int(np.count_nonzero(code34[dpf[i]] != rec[dpf[i]]))}
               for i, t in enumerate(tags)}
    trans_mats = []
    for k in range(1, len(tags)):
        a0 = code34[dpf[k - 1]].astype(int); a1 = code34[dpf[k]].astype(int)
        M = np.zeros((6, 6), np.int64); np.add.at(M, (a0, a1), 1)
        trans_mats.append({"from": tags[k - 1], "to": tags[k], "labels": LABELS,
                           "matrix": M.tolist(), "moved": int(M.sum() - np.trace(M))})

    # (5) 생산 JSON 기록값과의 직접 대조 (decisions[*].counts · delivery.inventory_final)
    prod_dec = {d["tag"]: d for d in res["decisions"]}
    dec_cmp = []
    for i, t in enumerate(tags):
        rc = counts(rec[dpf[i]])
        pj = prod_dec.get(t, {}).get("counts")
        dec_cmp.append({"tag": t, "particle_frame_row": int(dpf[i]),
                        "production_json_counts": pj, "raw_npz_recorded_counts": rc,
                        "json_equals_raw": (pj == rc) if pj is not None else None,
                        "rev34_recomputed_counts": counts(code34[dpf[i]]),
                        "rev34_vs_recorded_mismatch": int(np.count_nonzero(code34[dpf[i]] != rec[dpf[i]]))})
    fin_json = res["delivery"]["inventory_final"]
    fin_raw = counts(rec[-1])
    fin_34 = counts(code34[-1])
    fin_29 = counts(code29[-1])

    # (6) 재닫기(reclose_end) 종료 시 공구 내부 코호트와 그 운반 잔류 곡선
    pf_reclose = int(dpf[tags.index("reclose_end")])
    cohort_rec = np.flatnonzero(rec[pf_reclose] == LABELS.index("tool_residual"))
    cohort_34 = np.flatnonzero(code34[pf_reclose] == LABELS.index("tool_residual"))

    def curve(ids, code):
        return {"n": int(len(ids)),
                "tool_residual_per_frame": [int((code[f, ids] == LABELS.index("tool_residual")).sum())
                                            for f in range(F)],
                "final": counts(code[-1, ids])}
    coh = {"reclose_end_particle_frame_row": pf_reclose,
           "cohort_recorded_under_rev34": curve(cohort_rec, code34),
           "cohort_recorded_under_recorded_labels": curve(cohort_rec, rec),
           "cohort_rev34_n": int(len(cohort_34)),
           "cohort_rev34_equals_recorded": bool(np.array_equal(cohort_rec, cohort_34))}

    # (7) spill 코호트 — 이탈 단계(phase) 추적용 최초 spill 프레임 (기록 라벨 기준)
    sp = LABELS.index("spill")
    ever_spill_rec = np.flatnonzero((rec == sp).any(0))
    ever_spill_34 = np.flatnonzero((code34 == sp).any(0))
    first_spill_frame = {int(i): int(np.argmax(rec[:, i] == sp)) for i in ever_spill_rec}

    prov = {"artifact": "W25_REV34_DERIVED_V2_SUPPORT_FLOOR", "derived_not_raw": True,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_raw_npz": str(a.raw), "source_meta_json": str(a.meta), "pile_npz": str(a.pile),
            "contract": IG34.CONTRACT_VERSION, "floor_rule": IG34.FLOOR_RULE,
            "source_floor_rule": IG34.SOURCE_FLOOR_RULE, "transition_rule": RT.RULE,
            "legacy_transition_rule": RT.LEGACY_REV28_RULE,
            "input_sha256": hashes, "cfg_checks": cfg_checks,
            "semantics": IG34.semantics_metadata(len(tpl["sphere_radii_m"])),
            "non_claims": ["이 파일은 파생이다 — W25 원자료의 라벨/좌표/시간을 고치지 않는다.",
                           "새 물리·GPU·렌더 0. 같은 원시 좌표에 대한 재분류다.",
                           "receiving_bin 라벨은 정착 판정이 아니다(settlement window 별도).",
                           "이 재현은 '전체 사이클 성공' 선언이 아니다."]}
    npz_path = out / "w25_podB_seed460_rev34_derived.npz"
    np.savez_compressed(npz_path,
                        inventory_code_rev34_support=code34, inventory_code_rev29_strict=code29,
                        inventory_code_recorded_rev34=rec.astype(np.int8), inventory_labels=np.asarray(LABELS),
                        transition_sync_index_phase_only=trans_phase,
                        transition_sync_index_recorded=trans_rec,
                        transition_sync_index_legacy_rule_replay=trans_legacy,
                        particle_frame_row=z0["particle_frame_row"],
                        particle_frame_sync_index=z0["particle_frame_sync_index"],
                        particle_frame_t_s=z0["particle_frame_t_s"], particle_ids=z0["particle_ids"],
                        per_frame_counts_recorded=per_rec, per_frame_counts_rev29=per_29,
                        per_frame_counts_rev34=per_34,
                        confusion_recorded_vs_rev34=conf34, confusion_recorded_vs_rev29=conf29,
                        cohort_reclose_tool_ids_recorded=cohort_rec,
                        cohort_reclose_tool_ids_rev34=cohort_34,
                        ever_spill_ids_recorded=ever_spill_rec, ever_spill_ids_rev34=ever_spill_34,
                        provenance_json=np.asarray(json.dumps(prov, ensure_ascii=False)))
    man = dict(prov)
    man.update({"wall_s": round(time.time() - t0, 3), "n_frames": int(F), "n_particles": int(N),
                "output_npz": str(npz_path), "output_npz_sha256": D29.sha256_file(npz_path),
                "array_sha256": {"inventory_code_rev34_support": D29.array_sha(code34),
                                 "inventory_code_rev29_strict": D29.array_sha(code29),
                                 "inventory_code_recorded": D29.array_sha(rec)},
                "transitions": {"recorded": trans_rec.tolist(), "phase_only_rule": trans_phase.tolist(),
                                "legacy_rev28_rule_replay": trans_legacy.tolist(),
                                "n_recorded": int(len(trans_rec)), "n_phase_only": int(len(trans_phase)),
                                "n_legacy_replay": int(len(trans_legacy)),
                                "recorded_equals_phase_only": bool(np.array_equal(trans_rec, trans_phase)),
                                "recorded_equals_legacy_replay": bool(np.array_equal(trans_rec, trans_legacy))},
                "inventory": {"rev34_vs_recorded_mismatch": int(np.count_nonzero(code34 != rec)),
                              "rev29_vs_recorded_mismatch": int(np.count_nonzero(code29 != rec)),
                              "rev34_vs_rev29_mismatch": int(np.count_nonzero(code34 != code29)),
                              "n_cells": int(F) * int(N),
                              "confusion_recorded_rows_vs_rev34_cols": conf34.tolist(),
                              "confusion_recorded_rows_vs_rev29_cols": conf29.tolist(),
                              "mismatch_by_class_recorded": {LABELS[i]: int(conf34[i].sum() - conf34[i, i])
                                                             for i in range(6)},
                              "mismatch_by_class_rev34": {LABELS[j]: int(conf34[:, j].sum() - conf34[j, j])
                                                          for j in range(6)},
                              "per_frame_mismatch_rev34": per_frame_mismatch_34,
                              "per_frame_mismatch_rev29": per_frame_mismatch_29,
                              "n_frames_with_mismatch_rev34": int(sum(v > 0 for v in per_frame_mismatch_34)),
                              "final_recorded_raw_npz": fin_raw, "final_production_json": fin_json,
                              "final_json_equals_raw": fin_json == fin_raw,
                              "final_rev29": fin_29, "final_rev34": fin_34,
                              "n_ids_ever_receiving_bin_rev34":
                                  int((code34 == LABELS.index("receiving_bin")).any(0).sum()),
                              "max_receiving_bin_per_frame_rev34": int(per_34[:, 1].max()),
                              "frame_of_max_receiving_bin_rev34": int(per_34[:, 1].argmax()),
                              "particle_mass_g": float(z0["particle_mass_kg"]) * 1000.0},
                "decision_tag_inventory": per_tag, "decision_json_vs_raw_vs_rev34": dec_cmp,
                "decision_transition_matrices_rev34": trans_mats, "reclose_cohort": coh,
                "spill_cohort": {"n_ever_spill_recorded": int(len(ever_spill_rec)),
                                 "n_ever_spill_rev34": int(len(ever_spill_34)),
                                 "final_spill_recorded": int((rec[-1] == sp).sum()),
                                 "final_spill_rev34": int((code34[-1] == sp).sum()),
                                 "first_spill_particle_frame_row_by_id": first_spill_frame}})
    (out / "DERIVED_V2_MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps({"transitions": man["transitions"],
                      "inventory": {k: man["inventory"][k] for k in (
                          "rev34_vs_recorded_mismatch", "rev29_vs_recorded_mismatch", "rev34_vs_rev29_mismatch",
                          "n_cells", "n_frames_with_mismatch_rev34", "final_recorded_raw_npz",
                          "final_production_json", "final_json_equals_raw", "final_rev34",
                          "mismatch_by_class_recorded")},
                      "cohort": {k: coh[k] for k in ("reclose_end_particle_frame_row", "cohort_rev34_n",
                                                     "cohort_rev34_equals_recorded")},
                      "spill_cohort_n": man["spill_cohort"]["n_ever_spill_recorded"],
                      "wall_s": man["wall_s"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
