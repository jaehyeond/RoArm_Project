"""결함 1 — transition_sync_index: rev28 규칙 FAIL(25개) → rev29 규칙 PASS(11개). 독립식(groupby) 대조."""
import ast
import unittest

import numpy as np

import w14_paths as W
from independent_check import phase_only_transitions_groupby

PHASES = ["initial_home", "settle", "approach", "descend", "close", "lift", "reclose", "transport",
          "discharge", "discharge_wait", "close_after_discharge", "return_home"]


def extract_closures(sim_path):
    """sim_w13_full_cycle.py 에서 record_t0 / enter 함수 정의만 AST 로 뽑아 대역 상태로 실행한다(물리 0)."""
    tree = ast.parse(sim_path.read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ("record_t0", "enter")}
    assert set(fns) == {"record_t0", "enter"}, set(fns)
    ns = {"state": {"phase": None, "sub": None}, "trans_idx": [], "D": {"t": []}, "cmd": {},
          "pose_of": lambda cmd: None, "record_particle_frame": lambda force=False: None}
    ns["sample"] = lambda dts, tgt: ns["D"]["t"].append(dts) or (len(ns["D"]["t"]) - 1)
    exec(compile(ast.Module(body=[fns["record_t0"], fns["enter"]], type_ignores=[]), str(sim_path), "exec"), ns)
    return ns


def replay(sim_path, phase_codes, subphases):
    ns = extract_closures(sim_path)
    for i, (pc, sb) in enumerate(zip(phase_codes, subphases)):
        if i == 0:
            assert (PHASES[pc], sb) == ("initial_home", "t0_state")
            ns["record_t0"]()
        else:
            ns["enter"](PHASES[pc], sb)
            ns["sample"](0.004, None)
    assert len(ns["D"]["t"]) == len(phase_codes)
    return list(ns["trans_idx"])


class TransitionRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert W.sha256_file(W.RAW) == W.EXPECTED["raw"]
        with np.load(W.RAW, allow_pickle=False) as z:
            cls.pc = z["sync_phase_code"].astype(int)
            cls.sub = [str(v) for v in z["sync_subphase"]]
            cls.rec = z["transition_sync_index"].astype(int).tolist()
            cls.pf_sync = set(z["particle_frame_sync_index"].astype(int).tolist())
        cls.RT = W.import_from(W.REV29_SRC / "raw_transitions.py", "raw_transitions_rev29")
        cls.expected = phase_only_transitions_groupby(cls.pc)      # 독립식

    def test_01_independent_expected_equals_audit_prelisted(self):
        self.assertEqual(self.expected, W.AUDIT_EXPECTED_PHASE_ONLY)
        self.assertEqual(self.rec, W.AUDIT_RECORDED_25)

    def test_02_rev28_enter_reproduces_recorded_defect_and_FAILS_contract(self):
        got = replay(W.REV28_SRC / "sim_w13_full_cycle.py", self.pc, self.sub)
        self.assertEqual(got, self.rec, "rev28 enter()/record_t0 대역 재생이 기록 25개와 같아야 한다(결함 재현)")
        self.assertEqual(len(got), 25)
        self.assertNotEqual(got, self.expected, "rev28 규칙은 규약(phase-only) 과 달라야 한다 = FAIL 재현")

    def test_03_rev29_enter_matches_contract_PASS(self):
        got = replay(W.REV29_SRC / "sim_w13_full_cycle.py", self.pc, self.sub)
        self.assertEqual(got, self.expected)
        self.assertEqual(len(got), 11)

    def test_04_rev29_module_rules(self):
        self.assertEqual(self.RT.phase_only_transition_indices(self.pc).tolist(), self.expected)
        self.assertEqual(self.RT.legacy_rev28_transition_indices(self.pc, self.sub).tolist(), self.rec)

    def test_05_contract_properties(self):
        e = self.expected
        self.assertEqual(len(e), len(PHASES) - 1)
        self.assertNotIn(0, e)
        for i in e:
            self.assertNotEqual(self.pc[i], self.pc[i - 1])
        self.assertEqual(sorted(set(self.pc[[0] + e].tolist())), list(range(len(PHASES))), "12 phase 전부 1회씩")
        # 관찰(수정 범위 밖, 보고용): 생산 enter() 는 전환 직전 행(i-1, 경계 순간의 물리 상태)에 강제 입자 프레임을
        # 남기고, 전환 인덱스 i 는 새 phase 의 첫 행이다. 정확히 i 에 프레임이 있는 전환은 7336(결정 프레임) 뿐이다.
        # RAW_SCHEMA_REQUIRED.md:42 "Every phase transition sync must have a particle frame" 의 인덱스 규약이
        # 모호하다 — rev29 는 이 저장 동작을 바꾸지 않았고, 아래는 실제 저장 규약을 그대로 고정한다.
        for i in e:
            self.assertIn(i - 1, self.pf_sync, f"전환 {i}: 경계 순간(행 {i-1}) 입자 프레임이 있어야 한다")
        at_i = [i for i in e if i in self.pf_sync]
        self.assertEqual(at_i, [7336], "정확히 전환 행 i 에 프레임이 있는 경우는 reclose_end 결정 행뿐(관찰 고정)")
        extra = sorted(set(self.rec) - set(e))
        self.assertEqual(len(extra), 14)
        for i in extra:                                     # 초과 14개 = 행0 + subphase-only 13개
            self.assertTrue(i == 0 or (self.pc[i] == self.pc[i - 1] and self.sub[i] != self.sub[i - 1]))

    def test_06_synthetic_sequences(self):
        pc = np.array([0, 0, 1, 1, 1, 2, 2, 3])
        sb = ["a", "b", "b", "c", "c", "c", "d", "d"]
        self.assertEqual(self.RT.phase_only_transition_indices(pc).tolist(), [2, 5, 7])
        self.assertEqual(phase_only_transitions_groupby(pc), [2, 5, 7])
        self.assertEqual(self.RT.legacy_rev28_transition_indices(pc, sb).tolist(), [0, 1, 2, 3, 5, 6, 7])
        self.assertEqual(self.RT.phase_only_transition_indices(np.array([4])).tolist(), [])
        self.assertEqual(self.RT.phase_only_transition_indices(np.array([])).tolist(), [])


if __name__ == "__main__":
    unittest.main()
