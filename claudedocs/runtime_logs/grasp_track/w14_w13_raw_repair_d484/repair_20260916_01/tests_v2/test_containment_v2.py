"""ERRATUM_04 v2(바닥=받침면): rev29 strict 와 rev30 support 의 차이를 합성 사례·PF0/ID8·표본 프레임에서 확인. 독립식 v2 대조."""
import json
import unittest

import numpy as np

import w14_paths as W
from independent_check_v2 import classify_independent, rotate_hamilton

BOX = np.array([[-0.155, 0.155], [-0.11, 0.11], [0.0, 0.13663304498235546]])
IDQ = np.array([0.0, 0.0, 0.0, 1.0])


class SupportFloorV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.IG29, cls.IG30 = W.ig_rev29(), W.ig_rev30()
        cls.tpl = W.template(); cls.off = np.asarray(cls.tpl["offsets_m"], float); cls.rad = np.asarray(cls.tpl["sphere_radii_m"], float)
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False) + np.pi / 48
        cls.cfg = {"R_W": np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float),
                   "lip_l5_m": np.array([0.0081, 0.0, 0.1696]), "bowl_center_l5_m": np.array([0.0081, 0.0, 0.145]),
                   "bowl_r_in_m": 0.02, "cheek_half_y_m": 0.0182,
                   "bin_center_xy_m": np.array([5.0, 5.0]), "bin_normals": np.stack([np.cos(th), np.sin(th)], 1),
                   "bin_apothem_m": 0.04 * np.cos(np.pi / 48), "bin_floor_inner_z_m": 0.003, "bin_rim_z_m": 0.073,
                   "box_bounds_m": BOX, "box_top_m": float(BOX[2, 1]), "margin_m": 0.0025,
                   "v_settle_m_s": 0.03924, "spill_rest_z_m": 0.02}
        cls.p_tool_far, cls.R_tool = np.array([10.0, 10.0, 10.0]), np.eye(3)
        cls.rmax = float(cls.rad.max())

    def one(self, IG, center, speed=0.0, quat=IDQ):
        S = rotate_hamilton(quat[None], self.off) + np.asarray(center, float)
        return W.LABELS[int(IG.classify_spheres(S, self.rad[None, :], np.array([speed]), self.p_tool_far, self.R_tool, self.cfg)[0])]

    def test_01_semantics_declared(self):
        self.assertEqual(self.IG30.REVISION, "rev30"); self.assertIn("support_surface_v2", self.IG30.FLOOR_RULE)
        self.assertIn("classify_floor_rule", self.IG30.semantics_metadata(7)); self.assertEqual(self.IG29.REVISION, "rev29")

    def test_02_resting_on_floor(self):
        zc = self.rmax                                   # 최하단 정확히 0
        self.assertEqual(self.one(self.IG29, [0, 0, zc]), "ambiguous")
        self.assertEqual(self.one(self.IG30, [0, 0, zc]), "source")

    def test_03_penetration_tolerance_and_beyond(self):
        self.assertEqual(self.one(self.IG30, [0.02, -0.03, -0.001]), "source")       # 최하단 −2.25 mm > −2.5
        self.assertEqual(self.one(self.IG30, [0.02, -0.03, self.rmax - 0.0025]), "ambiguous")   # 최하단 == −2.5 정확 경계(strict >)
        self.assertEqual(self.one(self.IG30, [0.02, -0.03, -0.003]), "ambiguous")    # 최하단 −4.25 < −2.5, 최상단 −1.75 > −2.5 → near
        self.assertEqual(self.one(self.IG30, [0.0, 0.0, -0.05]), "spill")

    def test_04_side_walls_top_unchanged(self):
        for IG in (self.IG29, self.IG30):
            self.assertEqual(self.one(IG, [0.155 - 0.0025, 0.0, 0.06]), "ambiguous")
            self.assertEqual(self.one(IG, [0.0, -0.11 + 0.001, 0.06]), "ambiguous")
            self.assertEqual(self.one(IG, [0.0, 0.0, BOX[2, 1] - 0.001]), "ambiguous")
            self.assertEqual(self.one(IG, [0.05, 0.02, 0.06]), "source")
            self.assertEqual(self.one(IG, [0.05, 0.02, 0.06], speed=0.05), "in_flight")

    def test_05_bin_floor_layer_becomes_receiving_bin(self):
        cfg = dict(self.cfg); cfg["bin_center_xy_m"] = np.array([0.5, 0.5])     # 용기를 상자 밖으로
        c = [0.5, 0.5, 0.003 + self.rmax]                                        # 용기 바닥(0.003)에 놓인 알
        for IG, exp in ((self.IG29, "ambiguous"), (self.IG30, "receiving_bin")):
            S = rotate_hamilton(IDQ[None], self.off) + np.asarray(c)
            self.assertEqual(W.LABELS[int(IG.classify_spheres(S, self.rad[None, :], np.array([0.0]), self.p_tool_far, self.R_tool, cfg)[0])], exp)
        S = rotate_hamilton(IDQ[None], self.off) + np.asarray(c)
        self.assertEqual(W.LABELS[int(self.IG30.classify_spheres(S, self.rad[None, :], np.array([0.05]), self.p_tool_far, self.R_tool, cfg)[0])], "in_flight")

    def test_06_raw_pf0_id8_and_sample_frames(self):
        D = W.derive_module(); res = json.load(open(W.META))
        with np.load(W.RAW, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"])); cfg = D.build_inv_cfg(res, z)
            for fi in (0, 107, 136, 282):
                S, Rr, speed, p_tool, R_tool = D.production_frame_inputs(z, fi, self.tpl)
                c30 = self.IG30.classify_spheres(S, Rr, speed, p_tool, R_tool, cfg)
                fs = int(z["particle_frame_sync_index"][fi])
                ci = classify_independent(z["particle_pos_m"][fi].astype(float), z["particle_quat_xyzw"][fi].astype(float),
                                          z["particle_vel_m_s"][fi].astype(float), z["tool_pos_m"][fs].astype(float),
                                          z["tool_quat_xyzw"][fs].astype(float), z["bin_pos_m"], z["bin_quat_xyzw"],
                                          self.off, self.rad, meta, res["fixtures"]["bin"], floor_rule="support")
                self.assertEqual(int(np.count_nonzero(c30 != ci)), 0, f"frame {fi}")
                if fi == 0:
                    self.assertEqual(W.LABELS[int(c30[8])], "source"); self.assertEqual(W.LABELS[int(ci[8])], "source")
                    self.assertEqual(W.LABELS[int(z["inventory_code"][0, 8])], "source")


if __name__ == "__main__":
    unittest.main()
