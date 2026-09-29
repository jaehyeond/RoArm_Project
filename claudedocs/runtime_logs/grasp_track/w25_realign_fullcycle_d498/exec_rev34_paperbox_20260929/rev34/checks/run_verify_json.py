"""rev34 verify_w13_self.check_science 를 돌려 행 단위 결과를 JSON 으로 남긴다(읽기 전용, CPU).

검증기 본문은 바꾸지 않는다. 검증기가 도중에 예외로 죽으면(예: rev32 검증기의 bridge_precheck_columns 키 불일치)
그 전까지 쌓인 행과 예외를 그대로 기록하고 verdict = W13_SELF_CHECK_CRASHED 로 둔다.
usage: python run_verify_json.py <run_dir> <out_json> [pile_sha256]
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import verify_w13_self as V  # noqa: E402

_seen = []
_orig_init = V.Checks.__init__


def _init(self):
    _orig_init(self)
    _seen.append(self)


V.Checks.__init__ = _init

run, out = Path(sys.argv[1]), Path(sys.argv[2])
sha = sys.argv[3] if len(sys.argv) > 3 else V.PILE_SHA256
crash, summary = None, None
try:
    c, summary = V.check_science(run, sha)
except Exception:                                                   # noqa: BLE001
    crash = traceback.format_exc()
    c = _seen[-1]
rows = c.rows
verdict = "W13_SELF_CHECK_CRASHED" if crash else ("W13_SELF_CHECK_OK" if c.ok else "W13_SELF_CHECK_FAIL")
json.dump({"artifact": "W25A_VERIFY_SELF_JSON", "run": str(run), "verifier": str(Path(V.__file__).resolve()),
           "pile_sha256_expected": sha, "verdict": verdict, "crash": crash,
           "n_rows": len(rows), "n_fail": sum(not r["pass"] for r in rows),
           "failed": [r for r in rows if not r["pass"]], "rows": rows, "summary": summary},
          open(out, "w"), ensure_ascii=False, indent=2, default=float)
print(verdict, len(rows), "rows,", sum(not r["pass"] for r in rows), "fail:",
      [r["check"] for r in rows if not r["pass"]], "| crash:", (crash or "").strip().splitlines()[-1:] )
