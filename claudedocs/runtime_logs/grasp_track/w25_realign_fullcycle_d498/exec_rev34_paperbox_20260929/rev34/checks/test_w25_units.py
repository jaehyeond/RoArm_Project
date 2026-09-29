"""rev34 신규 함수 단위 시험 (CPU, 읽기 전용). 결과 JSON: python test_w25_units.py <out json>

W23 F 독립 주장과 교차: 규약 A 에서 중심 자리 툴은 문 쪽 = +x_box, 힌지축 = −y_box (W23 REPORT.md:139),
베이스각 θ 에서 툴 yaw = Rz(90°+θ)·(rev32 자세) (W23 REPORT.md:48,55).
"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(SRC))
import sim_deme_scoop_s1 as W11SRC                                       # noqa: E402
import w13_fk as FK                                                      # noqa: E402
import w13_kinematics as K                                               # noqa: E402

R32_PARAMS = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/"
                  "w19_runpod_d487/rev32_frozen_copy/params_w13.json")
P25_PATH = SRC.parent / "params_w25.json"


def params(path):
    P = dict(W11SRC.DEFAULT)
    P.update(K.W13_DEFAULT)
    P.update(json.load(open(path)))
    q_open = P["door_open_servo_deg"] - P["servo_zero_offset_deg"]
    P["lip_l5_mm"] = [float(v) for v in W11SRC.load_tool(P, q_open)[5]]
    return P


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P32, cls.P25 = params(R32_PARAMS), params(P25_PATH)

    def test_rev32_path_bit_identical(self):
        """rev32 키면 build_adapter_w25 가 build_adapter 와 t·자세·owner_pose 가 비트 동일."""
        P = self.P32
        a0, i0 = FK.build_adapter(None, 0.0411807760, P["arm_radius_m"], P["lip_l5_mm"],
                                  P["declared_base_cm"], P["declared_pellet_cm"])
        a1, i1, w = FK.build_adapter_w25(None, 0.0411807760, P, P["lip_l5_mm"])
        self.assertTrue(w["rev32_path"])
        self.assertEqual(i0, i1)
        for q in (FK.HOME_Q5, FK.P1_Q5, [90.0] + FK.P1_Q5[1:]):
            p0, R0 = a0.owner_pose(q)
            p1, R1 = a1.owner_pose(q)
            self.assertEqual(p0.tobytes(), p1.tobytes())
            self.assertEqual(R0.tobytes(), R1.tobytes())
        self.assertEqual(FK.w25_scoop_site(P, P["lip_l5_mm"])[0], (0.0, 0.0))

    def test_convention_A_axes(self):
        R = FK.BOX_FRAME_CONVENTIONS["A"]
        np.testing.assert_array_equal(R @ [1, 0, 0], [0, -1, 0])      # x_box = 로봇 −y
        np.testing.assert_array_equal(R @ [0, 1, 0], [1, 0, 0])       # y_box = 로봇 +x
        self.assertAlmostEqual(np.linalg.det(R), 1.0)
        np.testing.assert_array_equal(FK.BOX_FRAME_CONVENTIONS["B"], R.T)

    def test_adapter_roundtrip(self):
        P = self.P25
        ad, info, w = FK.build_adapter_w25(None, 0.04, P, P["lip_l5_mm"])
        rng = np.random.default_rng(1)
        for _ in range(50):
            p = rng.uniform(-0.5, 0.5, 3)
            self.assertLess(np.abs(ad.to_world(ad.to_robot(p)) - p).max(), 1e-15)
        np.testing.assert_allclose(ad.to_robot([0, 0, 0])[:2], [0.25, 0.0], atol=0)

    def test_w23_door_side_and_hinge_axis(self):
        """W23 REPORT.md:139 — 규약 A, 중심 자리: 문 쪽 = +x_box, 힌지축 = −y_box."""
        P = self.P25
        ad, info, w = FK.build_adapter_w25(None, 0.04, P, P["lip_l5_mm"])
        R_sc = np.asarray(w["R_scoop_owner_box"], float)
        axis_owner = W11SRC.R_W @ np.array([0.0, 1.0, 0.0])
        door_side_owner = W11SRC.R_W @ np.array([1.0, 0.0, 0.0])          # 문 반쪽 = link5 +x (half_bowl side +1)
        np.testing.assert_allclose(R_sc @ axis_owner, [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(R_sc @ door_side_owner, [1, 0, 0], atol=1e-12)
        # 실제 FK 자세(기준 자세)도 같은 방향(기울기 0.02° 이내)
        _, R_fk = ad.owner_pose(info["reference_pose_q5"])
        ang = math.degrees(np.linalg.norm(K.rotvec_of(R_sc.T @ R_fk)))
        self.assertLess(ang, 0.1)

    def test_w23_yaw_equivariance(self):
        """W23 REPORT.md:48,55 — 베이스각 θ: owner 자세 = Rz(90+θ)·(rev32 θ=0 자세)."""
        P25, P32 = self.P25, self.P32
        a25, i25, _ = FK.build_adapter_w25(None, 0.04, P25, P25["lip_l5_mm"])
        a32, _, _ = FK.build_adapter_w25(None, 0.04, P32, P32["lip_l5_mm"])
        q = list(i25["reference_pose_q5"])
        _, R0 = a32.owner_pose(q)
        for th in (-40.0, -7.97, 0.0, 17.745, 40.0):
            qq = [th] + q[1:]
            _, Rt = a25.owner_pose(qq)
            self.assertLess(np.abs(Rt - K.rot_z(90.0 + th) @ R0).max(), 1e-12)

    def test_scoop_site_lateral_offset(self):
        """상자 중심 명령 → 실제 립은 x_box +8.1 mm(로봇 y −8.1 mm) 옆 (W23 REPORT.md:92,135 T=(8.10,−0.17))."""
        site, info = FK.w25_scoop_site(self.P25, self.P25["lip_l5_mm"])
        self.assertAlmostEqual(site[0] * 1000, 8.1, places=6)
        self.assertLess(abs(site[1] * 1000), 0.2)

    def test_tray_declared_fail_closed(self):
        P = dict(self.P25)
        npz_box = np.array([[-0.155, 0.155], [-0.11, 0.11], [0.0, 0.1366]])
        with self.assertRaises(SystemExit):
            FK.w25_tray_bounds(npz_box, P)
        ok_box = np.array([[-0.1505, 0.1505], [-0.099, 0.099], [0.0, 0.2]])
        b, info = FK.w25_tray_bounds(ok_box, P, np.array([[0.0, 0.0, 0.01]]), np.array([0.002]))
        np.testing.assert_allclose(b, [[-0.1505, 0.1505], [-0.099, 0.099], [0.0, 0.105]])
        with self.assertRaises(SystemExit):   # 알 하나가 선언 벽 밖
            FK.w25_tray_bounds(ok_box, P, np.array([[0.1500, 0.0, 0.01]]), np.array([0.002]))
        # 정착 더미의 µm 급 벽 접촉(tol 0.5 mm 안)은 통과, 출처 메모가 붙은 declared… 값도 선언 모드
        P2 = dict(P, tray_inner_source="declared_from_npz_convention (user outer 31x22, wall 0.2)")
        b2, i2 = FK.w25_tray_bounds(ok_box, P2, np.array([[0.1485 + 0.6e-6, 0.0, 0.01]]), np.array([0.002]))
        self.assertEqual(i2["n_spheres_outside_declared_xy"], 0)
        b32, _ = FK.w25_tray_bounds(npz_box, self.P32)
        self.assertIs(b32, npz_box)

    def test_bad_keys_rejected(self):
        for k, v in (("box_frame_convention", "C"), ("w25_box_anchor", "somewhere")):
            P = dict(self.P25)
            P[k] = v
            with self.assertRaises(SystemExit):
                FK.w25_frame(P)

    def test_chatter_units(self):
        P = self.P25
        self.assertAlmostEqual(P["w25_chatter_threshold_servo_deg"] - P["servo_zero_offset_deg"], 1.1)
        self.assertAlmostEqual(P["w25_chatter_open_servo_deg"] - P["servo_zero_offset_deg"], 5.5)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(T)
    names = [t._testMethodName for t in suite]
    res = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    json.dump({"artifact": "W25A_UNITTEST", "tests_run": res.testsRun,
               "failures": [[str(t), tb] for t, tb in res.failures], "errors": [[str(t), tb] for t, tb in res.errors],
               "names": names, "ok": res.wasSuccessful()},
              open(out, "w"), ensure_ascii=False, indent=2)
    sys.exit(0 if res.wasSuccessful() else 1)
