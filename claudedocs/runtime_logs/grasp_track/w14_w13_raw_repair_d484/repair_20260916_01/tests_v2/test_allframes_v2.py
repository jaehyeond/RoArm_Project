"""전 283프레임: rev30 == 독립식 v2 (0), rev30 vs 기록/rev29 불일치 = manifest, 파생 NPZ·해시·보존."""
import json
import unittest

import numpy as np

import w14_paths as W
from independent_check_v2 import classify_independent


class AllFramesV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = W.derive_module(); cls.IG30 = W.ig_rev30(); cls.tpl = W.template()
        cls.off = np.asarray(cls.tpl["offsets_m"], float); cls.rad = np.asarray(cls.tpl["sphere_radii_m"], float)
        cls.res = json.load(open(W.META)); cls.z = np.load(W.RAW, allow_pickle=False)
        cls.meta = json.loads(str(cls.z["metadata_json"])); cls.cfg = cls.D.build_inv_cfg(cls.res, cls.z)
        cls.dz = np.load(W.DERIVED_V2 / "w13_cycle_seed460_rev30_derived_v2.npz", allow_pickle=False)
        cls.man = json.load(open(W.DERIVED_V2 / "DERIVED_V2_MANIFEST.json"))

    def test_01_all_frames(self):
        z, rec = self.z, self.z["inventory_code"]; F, N = rec.shape
        c29 = self.dz["inventory_code_rev29_strict"]; c30_all = np.empty((F, N), np.int8); m_ind = 0
        for fi in range(F):
            S, Rr, speed, p_tool, R_tool = self.D.production_frame_inputs(z, fi, self.tpl)
            c30 = self.IG30.classify_spheres(S, Rr, speed, p_tool, R_tool, self.cfg); c30_all[fi] = c30
            fs = int(z["particle_frame_sync_index"][fi])
            ci = classify_independent(z["particle_pos_m"][fi].astype(float), z["particle_quat_xyzw"][fi].astype(float),
                                      z["particle_vel_m_s"][fi].astype(float), z["tool_pos_m"][fs].astype(float),
                                      z["tool_quat_xyzw"][fs].astype(float), z["bin_pos_m"], z["bin_quat_xyzw"],
                                      self.off, self.rad, self.meta, self.res["fixtures"]["bin"], floor_rule="support")
            m_ind += int(np.count_nonzero(c30 != ci))
            self.assertEqual(int(np.bincount(c30.astype(int), minlength=6).sum()), N); self.assertTrue(((c30 >= 0) & (c30 <= 5)).all())
        self.assertEqual(m_ind, 0)
        self.assertTrue(np.array_equal(c30_all, self.dz["inventory_code_rev30_support"]))
        self.assertEqual(self.D.array_sha(c30_all), self.man["array_sha256"]["inventory_code_rev30_support"])
        self.assertEqual(int(np.count_nonzero(c30_all != rec)), self.man["inventory"]["rev30_vs_recorded_mismatch"])
        self.assertEqual(int(np.count_nonzero(c30_all != c29)), self.man["inventory"]["rev30_vs_rev29_mismatch"])
        self.assertEqual(W.counts(c30_all[-1]), self.man["inventory"]["final_rev30"])
        self.assertEqual(W.counts(c29[-1]), W.AUDIT_STRICT_FINAL); self.assertEqual(W.counts(rec[-1]), W.AUDIT_RECORDED_FINAL)

    def test_02_frozen_unchanged(self):
        self.assertEqual(W.sha256_file(W.RAW), W.EXPECTED["raw"]); self.assertEqual(W.sha256_file(W.META), W.EXPECTED["meta"])
        self.assertEqual(W.sha256_file(W.PILE), W.EXPECTED["pile"])
        self.assertEqual(W.sha256_file(W.DERIVED / "w13_cycle_seed460_rev29_derived.npz"), self.man["input_sha256"]["rev29_derived"])


if __name__ == "__main__":
    unittest.main()
