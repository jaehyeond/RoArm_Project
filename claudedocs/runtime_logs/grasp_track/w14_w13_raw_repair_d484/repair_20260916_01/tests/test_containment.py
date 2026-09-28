"""결함 2 — source 바닥 containment: rev28 FAIL(구 최상단 비교) → rev29 PASS(구 최하단 비교). 합성 사례 + PF0/ID8 + 표본 프레임 독립 대조."""
import json
import unittest

import numpy as np

import w14_paths as W
from independent_check import classify_independent, rotate_hamilton

BOX = np.array([[-0.155, 0.155], [-0.11, 0.11], [0.0, 0.13663304498235546]])
IDQ = np.array([0.0, 0.0, 0.0, 1.0])


class ContainmentRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.IG28, cls.IG29 = W.ig_rev28(), W.ig_rev29()
        cls.tpl = W.template()
        cls.off = np.asarray(cls.tpl["offsets_m"], float)
        cls.rad = np.asarray(cls.tpl["sphere_radii_m"], float)
        cls.k = len(cls.rad)
        # 합성 cfg: 공구·용기는 멀리 두고 상자/margin/속도창/spill 은 동결값
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False) + np.pi / 48
        cls.cfg = {"R_W": np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float),
                   "lip_l5_m": np.array([0.0081, 0.0, 0.1696]), "bowl_center_l5_m": np.array([0.0081, 0.0, 0.145]),
                   "bowl_r_in_m": 0.02, "cheek_half_y_m": 0.0182,
                   "bin_center_xy_m": np.array([5.0, 5.0]), "bin_normals": np.stack([np.cos(th), np.sin(th)], 1),
                   "bin_apothem_m": 0.04 * np.cos(np.pi / 48), "bin_floor_inner_z_m": 0.003, "bin_rim_z_m": 0.073,
                   "box_bounds_m": BOX, "box_top_m": float(BOX[2, 1]), "margin_m": 0.0025,
                   "v_settle_m_s": 0.03924, "spill_rest_z_m": 0.02}
        cls.p_tool_far, cls.R_tool = np.array([10.0, 10.0, 10.0]), np.eye(3)

    def one(self, IG, center, speed=0.0, quat=IDQ):
        S = rotate_hamilton(quat[None], self.off) + np.asarray(center, float)   # (1,k,3), 독립 회전식으로 전개
        code = IG.classify_spheres(S, self.rad[None, :], np.array([speed]), self.p_tool_far, self.R_tool, self.cfg)
        return W.LABELS[int(code[0])]

    def bottom_top(self, center, quat=IDQ):
        S = rotate_hamilton(quat[None], self.off)[0] + np.asarray(center, float)
        return float((S[:, 2] - self.rad).min()), float((S[:, 2] + self.rad).min())

    # ── 합성 사례: 규약 기대값 vs rev28/rev29 ────────────────────────────────
    def test_01_resting_on_floor_rev28_FAIL_rev29_PASS(self):
        zc = float(self.rad.max())                        # 가장 큰 구 최하단이 정확히 바닥 z=0 에 닿음
        lo, top = self.bottom_top([0.0, 0.0, zc])
        self.assertAlmostEqual(lo, 0.0, places=15)
        self.assertNotEqual(self.one(self.IG28, [0, 0, zc]), "ambiguous", "rev28 은 바닥에 놓인 알을 source 로 센다(결함 재현)")
        self.assertEqual(self.one(self.IG28, [0, 0, zc]), "source")
        self.assertEqual(self.one(self.IG29, [0, 0, zc]), "ambiguous", "규약: margin 밴드에 걸친 구가 있으면 ambiguous")

    def test_02_penetrating_floor_rev28_labels_source(self):
        # 중심 z=-1 mm: 최상단 > 바닥-margin 이라 rev28 은 source, rev29 는 ambiguous
        self.assertEqual(self.one(self.IG28, [0.02, -0.03, -0.001]), "source")
        self.assertEqual(self.one(self.IG29, [0.02, -0.03, -0.001]), "ambiguous")

    def test_03_fully_inside_both_source(self):
        zc = float(self.rad.max()) + 0.0025 + 1e-6
        self.assertGreater(self.bottom_top([0, 0, zc])[0], 0.0025)
        for IG in (self.IG28, self.IG29):
            self.assertEqual(self.one(IG, [0.05, 0.02, zc]), "source")
            self.assertEqual(self.one(IG, [0.05, 0.02, 0.06]), "source")

    def test_04_exact_floor_margin_boundary_is_strict(self):
        zc = float(self.rad.max()) + 0.0025                # 최하단 == floor+margin → '>' 이므로 안쪽 아님
        self.assertAlmostEqual(self.bottom_top([0, 0, zc])[0], 0.0025, places=15)
        self.assertEqual(self.one(self.IG29, [0, 0, zc]), "ambiguous")
        self.assertEqual(self.one(self.IG28, [0, 0, zc]), "source")

    def test_05_lateral_wall_band_both_ambiguous(self):
        for IG in (self.IG28, self.IG29):
            self.assertEqual(self.one(IG, [0.155 - 0.0025, 0.0, 0.06]), "ambiguous")
            self.assertEqual(self.one(IG, [0.0, -0.11 + 0.001, 0.06]), "ambiguous")

    def test_06_far_below_floor_spill_and_moving_in_flight(self):
        for IG in (self.IG28, self.IG29):
            self.assertEqual(self.one(IG, [0.0, 0.0, -0.05]), "spill")
            self.assertEqual(self.one(IG, [0.05, 0.02, 0.06], speed=0.05), "in_flight")
            self.assertEqual(self.one(IG, [0.05, 0.02, 0.06], speed=0.03924), "in_flight")   # >= 연산자
            self.assertEqual(self.one(IG, [0.05, 0.02, 0.06], speed=0.0392), "source")

    def test_07_rotated_clump_uses_all_spheres(self):
        q = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])   # x축 90°: 렌즈가 세로로 선다
        lo, _ = self.bottom_top([0, 0, 0.0025 + 0.0015], q)
        self.assertLess(lo, 0.0025)
        self.assertEqual(self.one(self.IG29, [0, 0, 0.0025 + 0.0015], quat=q), "ambiguous")
        self.assertEqual(self.one(self.IG29, [0, 0, 0.0025 + 0.0030], quat=q), "source")

    # ── 실제 원자료 PF0/ID8 (감사 finding 10 최소 반례) ─────────────────────
    def test_08_raw_pf0_id8_counterexample(self):
        D = W.derive_module()
        res = json.load(open(W.META))
        with np.load(W.RAW, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"]))
            cfg = D.build_inv_cfg(res, z)
            S, Rr, speed, p_tool, R_tool = D.production_frame_inputs(z, 0, self.tpl)
            rec0 = int(z["inventory_code"][0, 8])
            pos0, q0, v0 = z["particle_pos_m"][0], z["particle_quat_xyzw"][0], z["particle_vel_m_s"][0]
            fs0 = int(z["particle_frame_sync_index"][0])
            tp, tq, bp, bq = z["tool_pos_m"][fs0], z["tool_quat_xyzw"][fs0], z["bin_pos_m"], z["bin_quat_xyzw"]
        self.assertEqual(W.LABELS[rec0], "source")
        c28 = self.IG28.classify_spheres(S[8:9], Rr[8:9], speed[8:9], p_tool, R_tool, cfg)[0]
        c29 = self.IG29.classify_spheres(S[8:9], Rr[8:9], speed[8:9], p_tool, R_tool, cfg)[0]
        cind = classify_independent(pos0[8:9].astype(float), q0[8:9].astype(float), v0[8:9].astype(float),
                                    tp.astype(float), tq.astype(float), bp, bq, self.off, self.rad,
                                    meta, res["fixtures"]["bin"])[0]
        self.assertEqual(W.LABELS[int(c28)], "source")          # rev28 = 기록 재현 (FAIL)
        self.assertEqual(W.LABELS[int(c29)], "ambiguous")       # rev29 = 규약 (PASS)
        self.assertEqual(W.LABELS[int(cind)], "ambiguous")      # 독립식 동일
        lo = float((S[8, :, 2] - Rr[8]).min()); top = float((S[8, :, 2] + Rr[8]).min())
        self.assertAlmostEqual(lo, -1.2322870382021171e-06, places=12)
        self.assertAlmostEqual(top, 0.0023122574353417655, places=12)
        self.assertFalse(lo > 0.0025); self.assertTrue(top > -0.0025)

    # ── 표본 프레임 전수 대조(빠른 회귀) ─────────────────────────────────────
    def test_09_sample_frames_rev29_equals_independent_and_rev28_equals_recorded(self):
        D = W.derive_module()
        res = json.load(open(W.META))
        with np.load(W.RAW, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata_json"]))
            cfg = D.build_inv_cfg(res, z)
            for fi in (0, 107, 136, 282):
                S, Rr, speed, p_tool, R_tool = D.production_frame_inputs(z, fi, self.tpl)
                c28 = self.IG28.classify_spheres(S, Rr, speed, p_tool, R_tool, cfg)
                c29 = self.IG29.classify_spheres(S, Rr, speed, p_tool, R_tool, cfg)
                fs = int(z["particle_frame_sync_index"][fi])
                ci = classify_independent(z["particle_pos_m"][fi].astype(float), z["particle_quat_xyzw"][fi].astype(float),
                                          z["particle_vel_m_s"][fi].astype(float), z["tool_pos_m"][fs].astype(float),
                                          z["tool_quat_xyzw"][fs].astype(float), z["bin_pos_m"], z["bin_quat_xyzw"],
                                          self.off, self.rad, meta, res["fixtures"]["bin"])
                rec = z["inventory_code"][fi]
                self.assertEqual(int(np.count_nonzero(c28 != rec)), 0, f"frame {fi}: rev28 재계산 ≠ 기록")
                self.assertEqual(int(np.count_nonzero(c29 != ci)), 0, f"frame {fi}: rev29 ≠ 독립식")
                self.assertEqual(int(np.bincount(c29.astype(int), minlength=6).sum()), 20000)
                self.assertTrue(((c29 >= 0) & (c29 <= 5)).all())
                if fi == 282:
                    self.assertEqual(W.counts(c29), W.AUDIT_STRICT_FINAL)
                    self.assertEqual(W.counts(rec), W.AUDIT_RECORDED_FINAL)


if __name__ == "__main__":
    unittest.main()
