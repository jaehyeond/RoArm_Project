"""pod 측 해시 대조 — BUNDLE_MANIFEST.json 의 모든 파일이 같은 절대경로에 같은 sha256 으로 있는지. 불일치/누락 → rc 3."""
import hashlib, json, sys, time
from pathlib import Path

man = json.load(open(sys.argv[1]))
out = Path(sys.argv[2]) if len(sys.argv) > 2 else None

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

rows, bad = [], []
for p, want in sorted(man["files"].items()):
    q = Path(p)
    got = sha(q) if q.exists() else None
    ok = got == want["sha256"]
    rows.append({"path": p, "expected": want["sha256"], "actual": got, "match": ok})
    if not ok:
        bad.append(p)
rec = {"artifact": "W19_POD_BUNDLE_VERIFY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "manifest": sys.argv[1], "n_checked": len(rows), "n_mismatch": len(bad), "mismatches": bad, "checks": rows}
if out:
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
print(f"verify: {len(rows)} files, mismatch {len(bad)}")
sys.exit(0 if not bad else 3)
