"""두 CPU 스텁 드라이런(rev32 vs rev34 스위치 OFF)의 원자료가 바이트 동일한가 (읽기 전용).

NPZ: 모든 배열을 dtype·shape·바이트로 비교한다. 제외 = 실제 경과 시간(wall-clock) 두 개
(sync_wall_elapsed_s, scalar_engine_query_wall_s). metadata_json 은 rev34 가 **추가만** 한 키 `w25_frame` 을
뺀 뒤 비교한다. JSON: 결과의 제어·기하 절을 wall 필드를 지우고 비교한다.
usage: python compare_dryruns.py <run A> <run B> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np

WALL_KEYS = {"sync_wall_elapsed_s", "scalar_engine_query_wall_s"}
JSON_SECTIONS = ["frames", "trajectory", "door", "decisions", "transitions", "scoop_site", "fixtures", "engine",
                 "bridge_clearance", "delivery", "abort_class", "diverged", "fail_reason", "mesh_check", "particle"]


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in ("wall_s", "wall_seconds", "physics_steps_before_certify")
                and not (k == "path") and k not in ("inputs_sha256",)}
    if isinstance(o, list):
        return [strip(v) for v in o]
    if isinstance(o, str) and ("/dryrun/" in o):
        return o.split("/dryrun/")[1].split("/", 1)[-1]
    return o


def main(a, b, out):
    a, b = Path(a), Path(b)
    za = np.load(a / "w13_cycle_seed460.npz", allow_pickle=True)
    zb = np.load(b / "w13_cycle_seed460.npz", allow_pickle=True)
    rows, diff = {}, []
    for k in sorted(set(za.files) | set(zb.files)):
        if k not in za.files or k not in zb.files:
            rows[k] = "MISSING_IN_" + ("A" if k not in za.files else "B")
            diff.append(k)
            continue
        if k in WALL_KEYS:
            rows[k] = "EXCLUDED_WALL_CLOCK"
            continue
        x, y = za[k], zb[k]
        if k == "metadata_json":
            mx, my = json.loads(str(x)), json.loads(str(y))
            added = sorted(set(my) - set(mx))
            my = {kk: v for kk, v in my.items() if kk != "w25_frame"}
            same = mx == my
            rows[k] = f"{'EQUAL' if same else 'DIFF'} (B 추가 키 {added} 제외 비교)"
        elif k == "bridge_clearance_json":
            # 인증서 JSON 문자열 안의 실제 경과 시간 필드(wall_s)만 빼고 파싱 비교한다.
            same_raw = x.tobytes() == y.tobytes()
            same = strip(json.loads(str(x))) == strip(json.loads(str(y)))
            rows[k] = ("BYTES_EQUAL" if same_raw else
                       f"{'EQUAL' if same else 'DIFF'} after removing wall_s (raw bytes differ only if wall_s differs)")
        else:
            same = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            rows[k] = "BYTES_EQUAL" if same else f"DIFF dtype {x.dtype}/{y.dtype} shape {x.shape}/{y.shape}"
        if not same:
            diff.append(k)
    ja = json.load(open(a / "w13_cycle_seed460.json"))
    jb = json.load(open(b / "w13_cycle_seed460.json"))
    jrows = {}
    for s in JSON_SECTIONS:
        sa, sb = strip(ja.get(s)), strip(jb.get(s))
        jrows[s] = "EQUAL" if sa == sb else "DIFF"
    jdiff = [s for s, v in jrows.items() if v != "EQUAL"]
    n = int(np.asarray(za["sync_t_s"]).shape[0])
    rep = {"artifact": "W25A_DRYRUN_EQUALITY", "A": str(a), "B": str(b),
           "n_sync_A": n, "n_sync_B": int(np.asarray(zb["sync_t_s"]).shape[0]),
           "n_particle_frames_A": int(np.asarray(za["particle_frame_t_s"]).shape[0]),
           "n_particle_frames_B": int(np.asarray(zb["particle_frame_t_s"]).shape[0]),
           "abort_class": [ja.get("abort_class"), jb.get("abort_class")],
           "bridge_verdicts": [[c.get("verdict") for c in ja.get("bridge_clearance", [])],
                               [c.get("verdict") for c in jb.get("bridge_clearance", [])]],
           "npz_keys_compared": len([r for r in rows.values() if r != "EXCLUDED_WALL_CLOCK"]),
           "npz_diff_keys": diff, "npz_rows": rows, "json_rows": jrows, "json_diff_sections": jdiff,
           "w25_block_B": jb.get("w25", {}).get("frame", {}).get("rev32_path") if jb.get("w25") else None,
           "verdict": "IDENTICAL" if not diff and not jdiff else "DIFFERENT"}
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=2)
    print(rep["verdict"], "npz diff", diff, "json diff", jdiff, "n_sync", n)
    return 0 if rep["verdict"] == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
