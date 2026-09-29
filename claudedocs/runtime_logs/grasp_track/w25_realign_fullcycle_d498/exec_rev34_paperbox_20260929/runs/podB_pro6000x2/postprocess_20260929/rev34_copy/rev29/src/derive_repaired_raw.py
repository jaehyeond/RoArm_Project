#!/usr/bin/env python3
"""rev29 파생 산출 — 동결 run_01 원자료(읽기 전용)에서 규약대로 다시 계산한
(1) phase-only `transition_sync_index`, (2) 전체-구체 strict source containment 재고 코드.

* 원시 좌표/ID/쿼터니언/속도/시간/입력 해시는 손대지 않는다. 출력은 **파생(derived)** 이며 run_01 의
  raw 규약 FAIL 을 소급 PASS 로 바꾸지 않는다. 새 물리 0 · GPU 0 · rev28/run_01/post03 무수정.
* 생산 경로(rev29 `inventory_geometry.classify_spheres` + 동결 메인 `sim_deme_scoop_s1.expand_spheres`)로
  계산한다. 독립 검사기(tests/independent_check.py)는 이 함수들을 import 하지 않는다.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MAIN_REPO = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN_REPO))
sys.path.insert(0, str(HERE))
import inventory_geometry as IG29                                       # noqa: E402  rev29 (이 폴더)
import raw_transitions as RT                                            # noqa: E402
import sim_deme_scoop_s1 as W11SRC                                      # noqa: E402  동결 메인 소스(수정 금지)
import w13_kinematics as K                                              # noqa: E402

IMPL28 = Path("/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/"
              "w13_full_cycle_d484/resume_20260913/implementation")
DEFAULT_RAW = IMPL28 / "run_01/w13_cycle_seed460.npz"
DEFAULT_META = IMPL28 / "run_01/w13_cycle_seed460.json"
DEFAULT_REV28_SRC = IMPL28 / "rev28/src"
DEFAULT_PILE = Path("/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/"
                    "pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz")
EXPECTED_SHA256 = {
    "raw": "529f422e962730b17b520c3c9af2261aae7c431ec5e078cd2c368dfab0c46b0f",
    "meta": "e482b939a4cf42e5284b9b15e36a9f3613e989161ec538475e2c05f4260d9340",
    "pile": "659d6b0bc771678a0c7209d91f550edc933d03e41922245ea0adb64eeb818812",
    "rev28_inventory_geometry": "d52a50016bab06b703950c2685a6fb3c41b22a7c124f0957f0b50cdc525ea4cf",
    "main_sim_deme_scoop_s1": "2e40f7ed279dad42794d156c3e5e823d6ce0d8cb770ec9aad1b9280a33a7e933",
}
LABELS = list(IG29.INV_NAMES)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    """감사 test_rev28_production_partial_results.array_sha 와 같은 규약(dtype+shape+bytes)."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def import_module_from(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_inv_cfg(res, z):
    """rev28 sim_w13_full_cycle.py:359-373 의 inv_cfg 구성을 그대로 재현(동결 파라미터에서만 파생)."""
    P = res["params"]
    bin_info = res["fixtures"]["bin"]
    box = np.asarray(z["box_bounds_m"], float)
    margin = float(P["classify_margin_mm"]) / 1000.0
    n_th = int(P["bin_n_theta"])
    th = np.linspace(0, 2 * math.pi, n_th, endpoint=False) + math.pi / n_th
    bin_nrm = np.stack([np.cos(th), np.sin(th)], 1)
    bin_apothem = bin_info["inner_r_m"] * math.cos(math.pi / n_th)
    bin_c = np.asarray(bin_info["center_xy_m"], float)
    v_settle = 9.81 * float(P["dt_sync_s"])
    return {"R_W": W11SRC.R_W,
            "lip_l5_m": np.asarray(P["lip_l5_mm"], float) / 1000.0,
            "bowl_center_l5_m": np.asarray(P["bowl_center_l5_mm"], float) / 1000.0,
            "bowl_r_in_m": P["bowl_r_in_mm"] / 1000.0,
            "cheek_half_y_m": P["cheek_half_y_mm"] / 1000.0,
            "bin_center_xy_m": bin_c, "bin_normals": bin_nrm, "bin_apothem_m": bin_apothem,
            "bin_floor_inner_z_m": bin_info["floor_inner_z_m"], "bin_rim_z_m": bin_info["rim_z_m"],
            "box_bounds_m": box, "box_top_m": float(box[2, 1]), "margin_m": margin,
            "v_settle_m_s": v_settle, "spill_rest_z_m": P["spill_rest_z_m"]}


def load_template(pile_path):
    with np.load(pile_path, allow_pickle=False) as pile:
        return json.loads(str(pile["clump_template_json"]))


def production_frame_inputs(z, fi, tpl):
    """생산 경로와 같은 입력: 동결 expand_spheres(scipy) 전개 + 실제 tracker 포즈(float32 원시)."""
    pp = np.asarray(z["particle_pos_m"][fi], float)
    oq = np.asarray(z["particle_quat_xyzw"][fi], float)
    vv = np.asarray(z["particle_vel_m_s"][fi], float)
    k = len(tpl["sphere_radii_m"])
    sp, sr = W11SRC.expand_spheres(pp, oq, tpl)
    n = len(pp)
    S = np.asarray(sp, float).reshape(n, k, 3)
    Rr = np.asarray(sr, float).reshape(n, k)
    fs = int(z["particle_frame_sync_index"][fi])
    p_tool = np.asarray(z["tool_pos_m"][fs], float)
    R_tool = K.quat_xyzw_to_mat(np.asarray(z["tool_quat_xyzw"][fs], float))
    return S, Rr, np.linalg.norm(vv, axis=1), p_tool, R_tool


def counts(code):
    return {n: int(v) for n, v in zip(LABELS, np.bincount(np.asarray(code).astype(int), minlength=len(LABELS)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--pile", default=str(DEFAULT_PILE))
    ap.add_argument("--rev28-src", default=str(DEFAULT_REV28_SRC))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    hashes = {"raw": sha256_file(a.raw), "meta": sha256_file(a.meta), "pile": sha256_file(a.pile),
              "rev28_inventory_geometry": sha256_file(Path(a.rev28_src) / "inventory_geometry.py"),
              "main_sim_deme_scoop_s1": sha256_file(MAIN_REPO / "sim_deme_scoop_s1.py")}
    for k, v in EXPECTED_SHA256.items():
        if hashes[k] != v:
            raise SystemExit(f"입력 해시 불일치 {k}: {hashes[k]} != {v} — 동결 입력이 아니면 진행하지 않는다")
    IG28 = import_module_from(Path(a.rev28_src) / "inventory_geometry.py", "inventory_geometry_rev28_frozen")
    res = json.load(open(a.meta))
    tpl = load_template(a.pile)
    z = np.load(a.raw, allow_pickle=False)
    meta = json.loads(str(z["metadata_json"]))
    cfg = build_inv_cfg(res, z)
    assert abs(cfg["v_settle_m_s"] - float(meta["moving_threshold_m_s"])) < 1e-15
    assert abs(cfg["margin_m"] - float(meta["classify_margin_m"])) < 1e-15
    labels = [str(v) for v in z["inventory_labels"]]
    assert labels == LABELS
    rec = z["inventory_code"]
    F, N = rec.shape

    # (1) transitions
    trans_rec = z["transition_sync_index"].astype(np.int64)
    trans_29 = RT.phase_only_transition_indices(z["sync_phase_code"])
    trans_legacy = RT.legacy_rev28_transition_indices(z["sync_phase_code"], z["sync_subphase"])

    # (2) inventory, all frames, rev28 (parity) and rev29 (strict floor)
    code28 = np.empty((F, N), np.int8)
    code29 = np.empty((F, N), np.int8)
    for fi in range(F):
        S, Rr, speed, p_tool, R_tool = production_frame_inputs(z, fi, tpl)
        code28[fi] = IG28.classify_spheres(S, Rr, speed, p_tool, R_tool, cfg)
        code29[fi] = IG29.classify_spheres(S, Rr, speed, p_tool, R_tool, cfg)
        if fi % 40 == 0:
            print(f"  frame {fi}/{F} rec={counts(rec[fi])} rev29={counts(code29[fi])}", flush=True)
    mism28 = int(np.count_nonzero(code28 != rec))
    mism29 = int(np.count_nonzero(code29 != rec))
    per_frame_rec = np.stack([np.bincount(r.astype(int), minlength=6) for r in rec]).astype(np.int64)
    per_frame_29 = np.stack([np.bincount(r.astype(int), minlength=6) for r in code29]).astype(np.int64)
    assert (per_frame_rec.sum(1) == N).all() and (per_frame_29.sum(1) == N).all()

    # cohort: reclose_end 프레임에서 기록상 tool_residual 이었던 ID 집합(생산 기록 그대로)
    tags = [str(t) for t in z["decision_tags"]]
    pf_reclose = int(z["decision_particle_frame_index"][tags.index("reclose_end")])
    cohort = np.flatnonzero(rec[pf_reclose] == LABELS.index("tool_residual"))
    cohort_29 = np.flatnonzero(code29[pf_reclose] == LABELS.index("tool_residual"))

    # counterexample PF0/ID8 (감사 finding 10 의 최소 반례) — 수치 재계산
    S0, Rr0, _, _, _ = production_frame_inputs(z, 0, tpl)
    box = cfg["box_bounds_m"]; m = cfg["margin_m"]
    ce = {"particle_frame_row": 0, "particle_id": 8,
          "recorded_label": LABELS[int(rec[0, 8])], "rev28_label": LABELS[int(code28[0, 8])],
          "rev29_label": LABELS[int(code29[0, 8])],
          "sphere_surface_min_z_m": float((S0[8, :, 2] - Rr0[8]).min()),
          "min_sphere_top_m": float((S0[8, :, 2] + Rr0[8]).min()),
          "floor_m": float(box[2, 0]), "margin_m": m,
          "strict_bottom_predicate": bool(((S0[8, :, 2] - Rr0[8]) > box[2, 0] + m).all()),
          "rev28_top_predicate": bool(((S0[8, :, 2] + Rr0[8]) > box[2, 0] - m).all())}

    provenance = {
        "artifact": "W14_REV29_DERIVED_RAW_REPAIR_V1", "derived_not_raw": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_raw_npz": str(a.raw), "source_meta_json": str(a.meta), "pile_npz": str(a.pile),
        "input_sha256": hashes, "rev29_src_dir": str(HERE),
        "rev29_inventory_geometry_sha256": sha256_file(HERE / "inventory_geometry.py"),
        "rev29_raw_transitions_sha256": sha256_file(HERE / "raw_transitions.py"),
        "transition_rule": RT.RULE, "legacy_rule": RT.LEGACY_REV28_RULE,
        "source_floor_rule": IG29.SOURCE_FLOOR_RULE,
        "semantics": IG29.semantics_metadata(len(tpl["sphere_radii_m"])),
        "non_claims": ["run_01 원자료의 raw 규약 FAIL 은 그대로다(소급 PASS 아님).",
                       "새 물리·GPU·렌더 0. 이 파일의 라벨은 같은 원시 좌표에 대한 파생 재분류다.",
                       "바닥에 놓인 알이 ambiguous 가 되는 것은 규약(모든 구체가 margin 안쪽) 의 결과이며 물리 판단이 아니다."],
    }
    np.savez_compressed(
        out / "w13_cycle_seed460_rev29_derived.npz",
        transition_sync_index_phase_only=trans_29,
        transition_sync_index_recorded_rev28=trans_rec,
        transition_sync_index_legacy_rule_replay=trans_legacy,
        inventory_code_rev29_strict=code29,
        inventory_code_recorded_rev28=rec.astype(np.int8),
        inventory_labels=np.asarray(LABELS),
        particle_frame_row=z["particle_frame_row"], particle_frame_sync_index=z["particle_frame_sync_index"],
        particle_frame_t_s=z["particle_frame_t_s"], particle_ids=z["particle_ids"],
        per_frame_counts_recorded=per_frame_rec, per_frame_counts_rev29=per_frame_29,
        cohort_reclose_tool_ids_recorded=cohort, cohort_reclose_tool_ids_rev29=cohort_29,
        provenance_json=np.asarray(json.dumps(provenance, ensure_ascii=False)))
    manifest = dict(provenance)
    manifest.update({
        "wall_s": round(time.time() - t0, 3), "n_frames": int(F), "n_particles": int(N),
        "output_npz": str(out / "w13_cycle_seed460_rev29_derived.npz"),
        "output_npz_sha256": sha256_file(out / "w13_cycle_seed460_rev29_derived.npz"),
        "array_sha256": {"inventory_code_recorded": array_sha(rec), "inventory_code_rev28_recomputed": array_sha(code28),
                         "inventory_code_rev29_strict": array_sha(code29),
                         "transition_sync_index_phase_only": array_sha(trans_29)},
        "transitions": {"recorded_rev28": trans_rec.tolist(), "legacy_rule_replay": trans_legacy.tolist(),
                        "phase_only_rev29": trans_29.tolist(),
                        "recorded_equals_legacy_replay": bool(np.array_equal(trans_rec, trans_legacy)),
                        "recorded_equals_phase_only": bool(np.array_equal(trans_rec, trans_29)),
                        "n_recorded": int(len(trans_rec)), "n_phase_only": int(len(trans_29))},
        "inventory": {"rev28_recomputed_vs_recorded_mismatch": mism28,
                      "rev29_strict_vs_recorded_mismatch": mism29,
                      "final_recorded": counts(rec[-1]), "final_rev29_strict": counts(code29[-1]),
                      "reclose_end_frame": pf_reclose,
                      "reclose_recorded": counts(rec[pf_reclose]), "reclose_rev29": counts(code29[pf_reclose]),
                      "cohort_recorded_n": int(len(cohort)), "cohort_rev29_n": int(len(cohort_29)),
                      "cohort_recorded_final_rev29_classes": counts(code29[-1, cohort]),
                      "cohort_recorded_final_recorded_classes": counts(rec[-1, cohort]),
                      "particle_mass_g": float(z["particle_mass_kg"]) * 1000.0},
        "counterexample_pf0_id8": ce,
    })
    (out / "DERIVED_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ("transitions", "inventory", "counterexample_pf0_id8", "wall_s")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
