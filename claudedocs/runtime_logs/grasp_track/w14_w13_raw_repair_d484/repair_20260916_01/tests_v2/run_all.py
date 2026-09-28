"""테스트 러너 — 결과를 RESULTS.json 에 남긴다(각 테스트 이름/상태/시간)."""
import json
import sys
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Recorder(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows = []
        self._t0 = {}

    def startTest(self, test):
        self._t0[test.id()] = time.time()
        super().startTest(test)

    def _row(self, test, status, detail=""):
        self.rows.append({"test": test.id(), "status": status, "wall_s": round(time.time() - self._t0[test.id()], 3),
                          "detail": detail[:2000]})

    def addSuccess(self, test):
        super().addSuccess(test); self._row(test, "PASS")

    def addFailure(self, test, err):
        super().addFailure(test, err); self._row(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err); self._row(test, "ERROR", self._exc_info_to_string(err, test))


def main(patterns):
    suite = unittest.TestSuite()
    for p in patterns:
        suite.addTests(unittest.defaultTestLoader.discover(str(HERE), pattern=p))
    runner = unittest.TextTestRunner(verbosity=2, resultclass=Recorder)
    res = runner.run(suite)
    out = {"artifact": "W14_CPU_TEST_RESULTS", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "python": sys.version, "patterns": patterns, "n_run": res.testsRun,
           "n_pass": sum(r["status"] == "PASS" for r in res.rows), "n_fail": len(res.failures), "n_error": len(res.errors),
           "all_pass": res.wasSuccessful(), "rows": res.rows}
    name = "RESULTS.json" if patterns == ["test_*.py"] else "RESULTS_" + "_".join(
        q.replace("test_", "").replace(".py", "").replace("*", "all") for q in patterns) + ".json"
    (HERE / name).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["test_*.py"]))
