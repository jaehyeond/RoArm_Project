#!/usr/bin/env python3
"""P1 보존 영수증 — 원자료 sha256 을 RETRIEVAL_RECEIPT.json 과 대조. 읽기 전용.
podB 는 이 단계를 인라인으로 수행해 tools/ 에 파일이 없다. 형식은 podB PRESERVATION_*.json 과 동일.
"""
import argparse, datetime, hashlib, json
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
for k in ("run", "receipt", "out", "artifact"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
RUN = Path(a.run)
rec = json.load(open(a.receipt))["files"]
obs = {}
for p in sorted(RUN.rglob("*")):
    if p.is_file():
        obs["run_01/" + p.relative_to(RUN).as_posix()] = sha(p)
mism = []
for k, v in rec.items():
    if obs.get(k) != v:
        mism.append({"path": k, "receipt": v, "observed": obs.get(k)})
for k in obs:
    if k not in rec:
        mism.append({"path": k, "receipt": None, "observed": obs[k]})
out = {"artifact": a.artifact,
       "utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
       "raw_root": str(RUN), "receipt": str(a.receipt),
       "n_receipt_files": len(rec), "n_observed_files": len(obs),
       "n_mismatch": len(mism), "mismatch": mism,
       "verdict": "PASS" if not mism else "FAIL",
       "sha256_observed": obs}
Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: out[k] for k in ("artifact", "n_receipt_files", "n_observed_files", "n_mismatch", "verdict")},
                 ensure_ascii=False))
