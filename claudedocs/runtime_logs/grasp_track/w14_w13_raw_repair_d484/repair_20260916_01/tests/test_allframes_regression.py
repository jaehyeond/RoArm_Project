"""전 283 프레임 회귀: rev28 재계산==기록(parity), rev29==독립식(0 불일치), rev29 vs 기록 불일치==감사 1,507,161,
매 프레임 보존, 최종/cohort 분류, 파생 NPZ 해시 일치. derive_repaired_raw.py 실행 후에 돈다."""
import json
import unittest

import numpy as np

import w14_paths as W
from independent_check import classify_independent


class AllFramesRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = W.derive_module()
        cls.IG28, cls.IG29 = W.ig_rev28(), W.ig_rev29()
        cls.tpl = W.template()
        cls.off = np.asarray(cls.tpl["offsets_m"], float); cls.rad = np.asarray(cls.tpl["sphere_radii_m"], float)
        cls.res = json.load(open(W.META))
        assert W.sha256_file(W.RAW) == W.EXPECTED["raw"] and W.sha256_file(W.PILE) == W.EXPECTED["pile"]
        cls.z = np.load(W.RAW, allow_pickle=False)
        cls.meta = json.loads(str(cls.z["metadata_json"]))
        cls.cfg = cls.D.build_inv_cfg(cls.res, cls.z)
        cls.dz = np.load(W.DERIVED / "w13_cycle_seed460_rev29_derived.npz", allow_pickle=False)
        cls.man = json.load(open(W.DERIVED / "DERIVED_MANIFEST.json"))

    def test_01_all_frames_three_way(self):
        z, rec = self.z, self.z["inventory_code"]
        F, N = rec.shape
        self.assertEqual((F, N), (283, 20000))
        m28 = m29_ind = m29_rec = 0
        c29_all = np.empty((F, N), np.int8)
        for fi in range(F):
            S, Rr, speed, p_tool, R_tool = self.D.production_frame_inputs(z, fi, self.tpl)
            c28 = self.IG28.classify_spheres(S, Rr, speed, p_tool, R_tool, self.cfg)
            c29 = self.IG29.classify_spheres(S, Rr, speed, p_tool, R_tool, self.cfg)
            fs = int(z["particle_frame_sync_index"][fi])
            ci = classify_independent(z["particle_pos_m"][fi].astype(float), z["particle_quat_xyzw"][fi].astype(float),
                                      z["particle_vel_m_s"][fi].astype(float), z["tool_pos_m"][fs].astype(float),
                                      z["tool_quat_xyzw"][fs].astype(float), z["bin_pos_m"], z["bin_quat_xyzw"],
                                      self.off, self.rad, self.meta, self.res["fixtures"]["bin"])
            m28 += int(np.count_nonzero(c28 != rec[fi]))
            m29_ind += int(np.count_nonzero(c29 != ci))
            m29_rec += int(np.count_nonzero(c29 != rec[fi]))
            c29_all[fi] = c29
            self.assertEqual(int(np.bincount(c29.astype(int), minlength=6).sum()), N)
            self.assertTrue(((c29 >= 0) & (c29 <= 5)).all())
        self.assertEqual(m28, 0, "rev28 재계산은 기록과 같아야 한다(parity)")
        self.assertEqual(m29_ind, 0, "rev29 는 독립식과 전 프레임 일치해야 한다")
        self.assertEqual(m29_rec, W.AUDIT_ALL_FRAME_MISMATCH, "rev29 vs 기록 불일치 수는 감사 값과 같아야 한다")
        self.assertTrue(np.array_equal(c29_all, self.dz["inventory_code_rev29_strict"]), "파생 NPZ 와 재계산 일치")
        self.assertEqual(self.D.array_sha(c29_all), self.man["array_sha256"]["inventory_code_rev29_strict"])
        self.assertEqual(W.counts(c29_all[-1]), W.AUDIT_STRICT_FINAL)
        self.assertEqual(W.counts(rec[-1]), W.AUDIT_RECORDED_FINAL)
        tags = [str(t) for t in z["decision_tags"]]
        pf = int(z["decision_particle_frame_index"][tags.index("reclose_end")])
        cohort = np.flatnonzero(rec[pf] == W.LABELS.index("tool_residual"))
        self.assertEqual(len(cohort), 144)
        self.assertEqual(W.counts(c29_all[-1, cohort]), W.AUDIT_COHORT_STRICT_FINAL)

    def test_02_derived_transitions_and_provenance(self):
        self.assertEqual(self.dz["transition_sync_index_phase_only"].tolist(), W.AUDIT_EXPECTED_PHASE_ONLY)
        self.assertEqual(self.dz["transition_sync_index_recorded_rev28"].tolist(), W.AUDIT_RECORDED_25)
        self.assertTrue(np.array_equal(self.dz["inventory_code_recorded_rev28"], self.z["inventory_code"]))
        self.assertTrue(np.array_equal(self.dz["particle_frame_sync_index"], self.z["particle_frame_sync_index"]))
        prov = json.loads(str(self.dz["provenance_json"]))
        self.assertTrue(prov["derived_not_raw"])
        self.assertEqual(prov["input_sha256"]["raw"], W.EXPECTED["raw"])
        self.assertEqual(self.man["inventory"]["rev29_strict_vs_recorded_mismatch"], W.AUDIT_ALL_FRAME_MISMATCH)
        self.assertEqual(self.man["inventory"]["rev28_recomputed_vs_recorded_mismatch"], 0)
        self.assertEqual(W.sha256_file(W.DERIVED / "w13_cycle_seed460_rev29_derived.npz"), self.man["output_npz_sha256"])

    def test_03_frozen_inputs_unchanged_after_work(self):
        self.assertEqual(W.sha256_file(W.RAW), W.EXPECTED["raw"])
        self.assertEqual(W.sha256_file(W.META), W.EXPECTED["meta"])
        self.assertEqual(W.sha256_file(W.PILE), W.EXPECTED["pile"])
        pin = json.load(open(W.IMPL28 / "rev28/REVISION_PIN.json"))
        for rel, h in pin["frozen_copies_sha256"].items():
            self.assertEqual(W.sha256_file(W.IMPL28 / "rev28" / rel), h, rel)


if __name__ == "__main__":
    unittest.main()
