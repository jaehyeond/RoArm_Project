"""Verify every recorded W11 geometry/scalar batch against its raw arrays.

Rerun coordinates are compared as Float32 display copies, never used for physics.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from rerun.experimental import RrdReader


KEYS = ("z_lip_mm", "q_deg", "lipF_N", "M_hinge_res_Nm", "v_particle_max", "n_door", "n_fixed", "Fz_fixed_up_N", "max_single_contact_N")


def verify(cell, rrd, negative_control=False):
    cell, rrd = Path(cell), Path(rrd)
    z = np.load(cell / "scoop_s1_seed460.npz")
    res = json.loads((cell / "scoop_s1_seed460.json").read_text())
    rows = json.loads((cell / "timeline_seed460.json").read_text())["rows"]
    rt = np.load(res["render_timeline"]["path"])
    # NpzFile indexing decompresses a whole array on every access: load once.
    z = {k: z[k] for k in ("frame_t_s", "nodes_F_m", "nodes_D_m", "contact_frame", "contact_point_m", "contact_force_N")}
    rt = {k: rt[k] for k in ("timeline_frame", "t_s", "clump_pos_m", "clump_quat_xyzw")}
    pile = np.load(next(k for k in res["inputs_sha16"] if k.endswith(".npz")), allow_pickle=True)
    tpl = json.loads(str(pile["clump_template_json"]))
    offsets = np.asarray(tpl["offsets_m"], float)
    n = len(z["frame_t_s"])
    reader = RrdReader(rrd)
    checks = {}
    contacts = [np.where(z["contact_frame"] == i)[0] for i in range(n)]
    rt_index = {int(f): i for i, f in enumerate(rt["timeline_frame"])}

    def particle_batch(f):
        i = rt_index[f]
        return (rt["clump_pos_m"][i, :, None, :] + np.einsum("nij,kj->nki", Rotation.from_quat(rt["clump_quat_xyzw"][i]).as_matrix(), offsets)).reshape(-1, 3)

    contracts = {
        "/geometry/tool/fixed_nodes": ("Points3D:positions", np.arange(n), lambda f: z["nodes_F_m"][f] + (0.001 if negative_control else 0)),
        "/geometry/tool/door_nodes": ("Points3D:positions", np.arange(n), lambda f: z["nodes_D_m"][f]),
        "/contacts/points": ("Points3D:positions", np.arange(n), lambda f: z["contact_point_m"][contacts[f]]),
        "/contacts/forces": ("Arrows3D:vectors", np.arange(n), lambda f: z["contact_force_N"][contacts[f]] * 0.005),
        "/geometry/pile/animated": ("Points3D:positions", rt["timeline_frame"], particle_batch),
    }
    for key in KEYS:
        contracts[f"/metrics/{key}"] = ("Scalars:scalars", np.arange(n), lambda f, k=key: [float(rows[f].get(k, 0.0))])
    for entity, (component, expected_frames, expected_batch) in contracts.items():
        frames, times, values_ok, counts_ok, max_abs = [], [], True, True, 0.0
        scalar = component == "Scalars:scalars"
        for chunk in reader.stream().filter(content=entity, has_timeline="frame", components=component):
            rb = chunk.to_record_batch()
            fields = [f.name for f in rb.schema if (f.metadata or {}).get(b"rerun:component", b"").decode() == component]
            assert len(fields) == 1, (entity, fields)
            batch_frames = rb.column("frame").to_pylist()
            batches = rb.column(fields[0]).to_pylist()
            frames.extend(batch_frames)
            times.extend(rb.column("sim_time_s").cast("int64").to_pylist())
            for f, batch in zip(batch_frames, batches):
                if int(f) not in expected_frames or batch is None:
                    values_ok = counts_ok = False
                    continue
                wanted = np.asarray(expected_batch(int(f)), dtype=np.float64 if scalar else np.float32)
                actual = np.asarray(batch, dtype=wanted.dtype)
                if actual.size == 0:
                    actual = actual.reshape(wanted.shape) if wanted.size == 0 else actual
                counts_ok &= actual.shape == wanted.shape
                if actual.shape != wanted.shape:
                    values_ok = False
                    continue
                values_ok &= np.array_equal(actual, wanted)
                if actual.size:
                    max_abs = max(max_abs, float(np.abs(actual.astype(float) - wanted.astype(float)).max()))
        order = np.argsort(frames)
        expected_order = np.argsort(expected_frames)
        frame_ok = np.array_equal(np.asarray(frames, np.int64)[order], np.asarray(expected_frames, np.int64)[expected_order])
        source_times = rt["t_s"] if entity.endswith("/animated") else z["frame_t_s"]
        time_ok = np.array_equal(np.asarray(times, np.int64)[order], np.rint(source_times * 1e9).astype(np.int64)[expected_order])
        checks[entity] = {"pass": bool(frame_ok and counts_ok and time_ok and values_ok), "observed_rows": len(frames),
                          "expected_rows": len(expected_frames), "frames_exact": bool(frame_ok), "batch_shapes_exact": bool(counts_ok),
                          "times_exact_nanoseconds": bool(time_ok), "values_exact_at_storage_precision": bool(values_ok), "max_abs_value_diff": max_abs}
    return {"artifact": "W11_RRD_RAW_ARRAY_COVERAGE", "rrd": str(rrd), "source_cell": str(cell), "source_syncs": n,
            "source_particle_frames": len(rt["t_s"]), "negative_control_fixed_nodes_shifted_1mm": negative_control,
            "checks": checks, "pass": all(c["pass"] for c in checks.values())}


if __name__ == "__main__":
    result = verify(sys.argv[1], sys.argv[2], "--negative-control" in sys.argv)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
