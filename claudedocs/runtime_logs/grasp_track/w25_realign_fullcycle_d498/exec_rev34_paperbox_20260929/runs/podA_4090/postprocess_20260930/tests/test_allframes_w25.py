"""전 275프레임 회귀 — W25-A podB run_01.

기대값은 코드에 박지 않는다: 생산 수치는 실행 시 `w13_cycle_seed460.json`·원자료 NPZ 에서 읽어 비교한다.
독립식(Hamilton 곱 · AABB · groupby)은 생산 모듈(inventory_geometry / sim_deme_scoop_s1 / scipy / w13_*)을
import 하지 않는 `independent_check_v2.py`(W19 후처리 바이트 사본, sha 는 RESULTS JSON 에 기록) 이다.
"""
import json, time, unittest

import numpy as np

import w25_paths as W
from independent_check_v2 import classify_independent, phase_only_transitions_groupby

RESULT = {"artifact": "W25_ALLFRAMES_RESULTS_V1", "cases": {}}


class AllFramesW25(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t0 = time.time()
        cls.D = W.derive_module(); cls.IG = W.ig_rev34(); cls.tpl = W.template()
        cls.off = np.asarray(cls.tpl["offsets_m"], float)
        cls.rad = np.asarray(cls.tpl["sphere_radii_m"], float)
        cls.res = json.load(open(W.META))
        cls.z0 = np.load(W.RAW, allow_pickle=False)
        cls.meta = json.loads(str(cls.z0["metadata_json"]))
        cls.cfg = cls.D.build_inv_cfg(cls.res, cls.z0)
        cls.z = W.FrameView(cls.z0, ["particle_pos_m", "particle_quat_xyzw", "particle_vel_m_s",
                                     "particle_frame_sync_index", "tool_pos_m", "tool_quat_xyzw"])
        cls.dz = np.load(W.DERIVED_NPZ, allow_pickle=False)
        cls.man = json.load(open(W.DERIVED / "DERIVED_V2_MANIFEST.json"))

    def test_01_all_frames_independent(self):
        """생산식(rev34 사본) == 독립식 0 불일치 + 파생 NPZ/array_sha 재현."""
        rec = np.asarray(self.z0["inventory_code"]); F, N = rec.shape
        self.assertEqual(F, int(self.res["trajectory"]["n_particle_frames"]))
        self.assertEqual(N, int(self.res["particle"]["n"]))
        c_all = np.empty((F, N), np.int8); m_ind = 0; per_frame_ind = []
        for fi in range(F):
            S, Rr, speed, p_tool, R_tool = self.D.production_frame_inputs(self.z, fi, self.tpl)
            cp = self.IG.classify_spheres(S, Rr, speed, p_tool, R_tool, self.cfg); c_all[fi] = cp
            fs = int(self.z["particle_frame_sync_index"][fi])
            ci = classify_independent(self.z["particle_pos_m"][fi].astype(float),
                                      self.z["particle_quat_xyzw"][fi].astype(float),
                                      self.z["particle_vel_m_s"][fi].astype(float),
                                      self.z["tool_pos_m"][fs].astype(float),
                                      self.z["tool_quat_xyzw"][fs].astype(float),
                                      self.z0["bin_pos_m"], self.z0["bin_quat_xyzw"],
                                      self.off, self.rad, self.meta, self.res["fixtures"]["bin"],
                                      floor_rule="support")
            d = int(np.count_nonzero(cp != ci)); m_ind += d; per_frame_ind.append(d)
            self.assertTrue(((cp >= 0) & (cp <= 5)).all())
            self.assertEqual(int(np.bincount(cp.astype(int), minlength=6).sum()), N)
        RESULT["cases"]["independent_vs_production_mismatch_cells"] = m_ind
        RESULT["cases"]["independent_mismatch_frames"] = [i for i, v in enumerate(per_frame_ind) if v]
        RESULT["cases"]["n_frames"] = int(F); RESULT["cases"]["n_particles"] = int(N)
        RESULT["cases"]["n_cells"] = int(F) * int(N)
        RESULT["cases"]["production_recomputed_vs_recorded_mismatch_cells"] = int(np.count_nonzero(c_all != rec))
        RESULT["cases"]["final_recomputed"] = W.counts(c_all[-1])
        RESULT["cases"]["final_recorded_raw"] = W.counts(rec[-1])
        RESULT["cases"]["final_production_json"] = self.res["delivery"]["inventory_final"]
        self.assertEqual(m_ind, 0, f"독립식 불일치 {m_ind} 셀; 프레임={RESULT['cases']['independent_mismatch_frames']}")
        self.assertTrue(np.array_equal(c_all, self.dz["inventory_code_rev34_support"]))
        self.assertEqual(self.D.array_sha(c_all), self.man["array_sha256"]["inventory_code_rev34_support"])
        self.assertEqual(int(np.count_nonzero(c_all != rec)), self.man["inventory"]["rev34_vs_recorded_mismatch"])
        self.assertEqual(W.counts(c_all[-1]), self.man["inventory"]["final_rev34"])

    def test_02_transitions_independent(self):
        """전환 인덱스: 기록값 == 규약(phase-only, groupby 독립 구현)."""
        rec_t = np.asarray(self.z0["transition_sync_index"]).astype(int).tolist()
        ind_t = phase_only_transitions_groupby(self.z0["sync_phase_code"])
        RESULT["cases"]["transitions_recorded"] = rec_t
        RESULT["cases"]["transitions_independent_phase_only"] = ind_t
        self.assertEqual(rec_t, ind_t)
        self.assertEqual(self.man["transitions"]["phase_only_rule"], ind_t)
        pfs = set(int(v) for v in self.z0["particle_frame_sync_index"])
        missing = [i for i in rec_t if (i - 1) not in pfs]      # ERRATUM_04 §2 경계 프레임 i-1
        RESULT["cases"]["transition_boundary_frames_missing"] = missing
        self.assertEqual(missing, [])

    def test_03_frozen_inputs_and_outputs(self):
        """원자료 sha == 회수 영수증, 파생 sha == manifest, sync 행수 == 생산 JSON."""
        ref = W.receipt_sha()
        for name, p in (("run_01/w13_cycle_seed460.npz", W.RAW),
                        ("run_01/w13_cycle_seed460.json", W.META),
                        ("run_01/timeline_seed460.json", W.TIMELINE)):
            self.assertEqual(W.sha256_file(p), ref[name])
        self.assertEqual(W.sha256_file(W.DERIVED_NPZ), self.man["output_npz_sha256"])
        self.assertEqual(int(self.z0["sync_t_s"].shape[0]), int(self.res["trajectory"]["n_sync"]))
        RESULT["cases"]["n_sync"] = int(self.z0["sync_t_s"].shape[0])

    def test_04_reclose_cohort_and_json_parity(self):
        """decisions[*].counts(JSON) == 원자료 라벨 bincount, reclose 코호트 동일."""
        rec = np.asarray(self.z0["inventory_code"])
        tags = [str(t) for t in self.z0["decision_tags"]]
        dpf = np.asarray(self.z0["decision_particle_frame_index"]).astype(int)
        prod = {d["tag"]: d["counts"] for d in self.res["decisions"]}
        bad = []
        for i, t in enumerate(tags):
            if prod.get(t) != W.counts(rec[dpf[i]]):
                bad.append({"tag": t, "json": prod.get(t), "raw": W.counts(rec[dpf[i]])})
        RESULT["cases"]["decision_counts_json_vs_raw_mismatch"] = bad
        self.assertEqual(bad, [])
        pf = int(dpf[tags.index("reclose_end")])
        coh = np.flatnonzero(rec[pf] == W.LABELS.index("tool_residual"))
        RESULT["cases"]["reclose_cohort_n"] = int(len(coh))
        self.assertTrue(np.array_equal(coh, self.dz["cohort_reclose_tool_ids_recorded"]))

    @classmethod
    def tearDownClass(cls):
        RESULT["wall_s"] = round(time.time() - cls.t0, 3)
        RESULT["independent_checker_sha256"] = W.sha256_file(W.HERE / "independent_check_v2.py")
        RESULT["independent_checker_imports"] = ["itertools", "math", "numpy"]
        RESULT["independent_checker_provenance"] = (
            "W19 postprocess_20260918/tests/independent_check_v2.py 바이트 사본 — 규약 문구에서 별도 구현"
            "(Hamilton 곱 회전 · AABB 봉쇄 · groupby 전환). 생산 모듈/scipy import 0.")
        RESULT["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (W.POST / "tests/RESULTS_allframes_w25.json").write_text(
            json.dumps(RESULT, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    unittest.main()
