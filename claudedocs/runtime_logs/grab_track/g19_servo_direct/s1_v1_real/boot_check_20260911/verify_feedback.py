"""Verify saved observations only. This file never opens a device."""
import hashlib
import json
import math
from pathlib import Path


def verify(events, summary):
    tx = [e["command"] for e in events if e["ev"] == "tx"]
    assert tx == [{"T": 105}] * 10, "unexpected transmitted command"
    assert events[0]["ev"] == "open" and events[-1]["ev"] == "closed"
    rows = [e["data"] for e in events if e["ev"] == "rx_json"
            and isinstance(e.get("data"), dict) and e["data"].get("T") == 1051]
    assert rows and len(rows) == summary["n_feedback"]
    keys = sorted(set().union(*(r.keys() for r in rows)))
    assert keys == summary["feedback_keys"]
    assert sum("tG" in r for r in rows) == summary["tG_present_rows"]
    assert rows[-1] == summary["last_raw"]
    for r in rows:
        assert all(k in r and math.isfinite(float(r[k])) for k in ("b", "s", "e", "t", "r", "g"))
    q = [math.degrees(float(rows[-1][k])) for k in ("b", "s", "e", "t", "r")]
    q.append(180 - math.degrees(float(rows[-1]["g"])))
    assert all(abs(a-b) < 1e-10 for a, b in zip(q, summary["last_sdk_angle_deg"]))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "feedback_summary.json").read_text())
    raw = root / Path(summary["raw_path"]).name
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == summary["sha256"]
    events = [json.loads(line) for line in raw.read_text().splitlines()]
    verify(events, summary)
    controls = {}
    for name in ("unexpected_motion", "missing_feedback", "wrong_field_count"):
        altered = json.loads(json.dumps(events))
        altered_summary = dict(summary)
        if name == "unexpected_motion":
            next(e for e in altered if e["ev"] == "tx")["command"] = {"T": 100}
        elif name == "missing_feedback":
            altered = [e for e in altered if e["ev"] != "rx_json"]
        else:
            altered_summary["tG_present_rows"] += 1
        try:
            verify(altered, altered_summary)
        except AssertionError:
            controls[name] = "rejected"
        else:
            raise AssertionError(f"negative control accepted: {name}")
    (root / "feedback_verification.json").write_text(json.dumps({
        "pass": True, "feedback_rows": summary["n_feedback"],
        "transmit_count": 10, "tG_present_rows": summary["tG_present_rows"],
        "negative_controls": controls, "raw_sha256": summary["sha256"],
        "scope": "file/hash/schema audit, no motion or temporal verdict"
    }, ensure_ascii=False, indent=2) + "\n")
    print("RAW_FEEDBACK_AUDIT_OK")
