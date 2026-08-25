#!/usr/bin/env python3
"""p29 / yp1·yp2 — y2_d454 pick-place 전이 probe (전량 이송 에피소드).

Contract: claudedocs/runtime_logs/yard_track/y2_d454/y2_prereg.md (동결
3963212a...).  초기 상태 = y1_d453 yt3 웨이브 스폰 verbatim (D453 bit-재현).

cycle = pick(source 높이맵 argmax 셀 → 레이 hit 물체 → RemovePrim)
      + place(목표 bin 셀 상공 dynamic 신규 저작, z = max(0.20, h_bin+r+8mm))
      + 합동 정착(streak 30, cap 400) + 관측 + 지표.
측정: 재형성(footprint 밖 ΔH 셀 수) / 착지 분산 / H_max 회계 / spread vs stack.

--probe yp1 (9지점 라스터 순환) | yp2 (중앙 (6,6) 고정)
--rep 1 | 2 (yp1 전용 cold-start 재현성)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CASE_DIR = REPO / "claudedocs/runtime_logs/yard_track/y2_d454"
Y1_DIR = REPO / "claudedocs/runtime_logs/yard_track/y1_d453"
MANIFEST = REPO / "sim_assets/posco_rocks_o1/manifest.json"

PINS = {
    MANIFEST: "a1127acca8854a773e7c31f2855a3d753398473dea1d8fde4f69667c7a22008c",
    Y1_DIR / "y1_design.json":
        "a045c414bfeb381e60f84c39a71331d149ea0d06efeb7057542dbafbfb7f1f10",
}
PREREG = CASE_DIR / "y2_prereg.md"
NUMPY_PIN = "1.26.0"
PSUTIL_PIN = "5.9.8"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"

TAG = "yp1"      # main()에서 --probe로 설정
LOG = "y2_yp1"

SEED = 45300
CLASSES = (22, 26, 30, 34)
PER_CLASS = 8
N_ROCKS = 32
WAVE_OFF = 0.0275
WAVE_Z = 0.20
WAVE_GAP_STEPS = 45
N_WAVES = 8
GRAVITY = 9.81
DT = 1.0 / 60.0
V_EPS = 0.005
W_EPS = 0.1
INIT_T_MAX = 1200
INIT_STREAK = 60
CYCLE_CAP = 400
CYCLE_STREAK = 30
N_CYCLES = 32
T_ALLOC = INIT_T_MAX + N_CYCLES * CYCLE_CAP + 10
H_MAX = 0.080
H_TOL = 0.0005
PICK_MIN_H = 0.005
DROP_Z_MIN = 0.20
DROP_CLEAR = 0.008
RESHAPE_TOL = 0.002
FOOTPRINT_PAD = 0.010
PATTERN_SPREAD = ((2, 2), (2, 6), (2, 10), (6, 2), (6, 6), (6, 10),
                  (10, 2), (10, 6), (10, 10))
PATTERN_STACK = ((6, 6),)
STATIC_FRICTION = 0.40
DYNAMIC_FRICTION = 0.30
ROCK_RESTITUTION = 0.1
SUPPORT_FRICTION = 1.0
ROCKS_ROOT = "/World/rocks"
GROUND_PRIM = "/World/ground"


def out_files(rep: int) -> dict:
    if rep == 1:
        keys = ("results.json", "trace.npz", "timeline.rrd", "timeline.rbl",
                "rerun_validation.json", "inspection.png", "stdout.log",
                "stderr.log", "exit_status.txt", "script.py.txt", "argv.txt",
                "failure.json")
        return {k: CASE_DIR / f"{TAG}_{k}" for k in keys} | {
            "hmap_png": CASE_DIR / f"{TAG}_final_heightmap.png"}
    keys = ("results.json", "stdout.log", "stderr.log", "exit_status.txt",
            "argv.txt", "failure.json")
    return {k: CASE_DIR / f"{TAG}_rep2_{k}" for k in keys} | {
        "compare": CASE_DIR / f"{TAG}_repeat_compare.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_write(path: Path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_design() -> dict:
    d = json.loads((Y1_DIR / "y1_design.json").read_text())
    for reg in ("source", "bin"):
        if d[reg]["n_cells"] != 13 or abs(d[reg]["L_int"] - 0.13) > 1e-12:
            raise RuntimeError(f"DESIGN_MISMATCH {reg}")
    if abs(d["bin"]["h_max_m"] - H_MAX) > 1e-12:
        raise RuntimeError("H_MAX_MISMATCH")
    return d


def region_cells(center, n=13, cell=0.010) -> np.ndarray:
    half = n * cell / 2.0
    xs = center[0] - half + cell * (np.arange(n) + 0.5)
    ys = center[1] - half + cell * (np.arange(n) + 0.5)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([gx, gy], axis=-1)


def build_plan(design: dict, probe: str) -> dict:
    m = json.loads(MANIFEST.read_text())
    by = {(o["class_mm"], o["index"]): o for o in m["objects"]}
    rocks = []
    for k in range(PER_CLASS):
        for c in CLASSES:
            o = by[(c, k)]
            v = np.array(o["vertices_m"], dtype=np.float64)
            rocks.append({
                "name": o["name"], "class_mm": c, "index": k,
                "verts": v, "faces": np.array(o["faces"], dtype=np.int64),
                "mass_kg": o["mass_est_15infill_g"] / 1000.0,
                "r_bound_m": float(np.linalg.norm(v, axis=1).max()),
                "stl_sha16": o["stl_sha256_16"]})
    if len(rocks) != N_ROCKS:
        raise RuntimeError("SUBSET_COUNT")

    rng = np.random.default_rng(SEED)
    slot_perm = rng.permutation(N_ROCKS)
    quats = rng.normal(size=(N_ROCKS, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    quats[quats[:, 0] < 0] *= -1.0

    sc = design["source"]["center"]
    rb = np.array([r["r_bound_m"] for r in rocks])
    top2 = np.sort(rb)[-2:]
    order = [int(x) for x in slot_perm]
    wave_xy = [(sc[0] + sx * WAVE_OFF, sc[1] + sy * WAVE_OFF)
               for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1))]
    waves = []
    spawn = [None] * N_ROCKS
    for w in range(N_WAVES):
        members = []
        for j in range(4):
            ri = order[w * 4 + j]
            pos = np.array([wave_xy[j][0], wave_xy[j][1], WAVE_Z])
            members.append({"rock_idx": ri, "pos": pos.tolist()})
            spawn[ri] = {"pos": pos, "quat": quats[ri]}
        waves.append(members)
    margins = {
        "wall_band_clear_m": float(design["source"]["L_int"] / 2
                                   - (WAVE_OFF + float(rb.max()))),
        "wave_horiz_worst_m": float(2 * WAVE_OFF - top2.sum()),
        "wave_bottom_clear_m": float(WAVE_Z - rb.max()
                                     - design["source"]["wall_h_m"])}
    if min(margins.values()) <= 0:
        raise RuntimeError(f"SPAWN_CLEARANCE_FAIL {margins}")

    pattern = PATTERN_SPREAD if probe == "yp1" else PATTERN_STACK
    targets = [pattern[c % len(pattern)] for c in range(N_CYCLES)]
    return {"rocks": rocks, "spawn": spawn, "waves": waves, "margins": margins,
            "quats": quats.tolist(), "probe": probe,
            "place_targets_rc": [list(t) for t in targets]}


def preflight(rep: int, probe: str) -> tuple:
    import numpy
    import psutil
    if numpy.__version__ != NUMPY_PIN or psutil.__version__ != PSUTIL_PIN:
        raise RuntimeError(f"ENV_PIN numpy={numpy.__version__} psutil={psutil.__version__}")
    pins = {}
    for path, want in PINS.items():
        got = sha256(path)
        pins[path.name] = {"sha256": got, "match": got == want}
        if got != want:
            raise RuntimeError(f"SHA_DRIFT {path}")
    pins["y2_prereg.md"] = {"sha256": sha256(PREREG), "match": "recorded-only"}

    out = out_files(rep)
    guard_exempt = ("script.py.txt", "argv.txt", "stdout.log", "stderr.log")
    existing = [p.name for k, p in out.items()
                if p.exists() and k not in guard_exempt]
    if existing:
        raise RuntimeError(f"WRITE_GUARD existing={existing}")
    if rep == 2 and not (CASE_DIR / f"{TAG}_results.json").exists():
        raise RuntimeError("REP2_REQUIRES_REP1_RESULTS")

    design = load_design()
    plan = build_plan(design, probe)
    plan["pins"] = pins
    plan["env"] = {"numpy": numpy.__version__, "psutil": psutil.__version__,
                   "python": sys.version.split()[0]}
    return design, plan, out


def gt_heightmap(cells: np.ndarray, world_verts: list, faces_list: list) -> np.ndarray:
    flat = cells.reshape(-1, 2)
    h = np.zeros(flat.shape[0])
    for verts, faces in zip(world_verts, faces_list):
        tri = verts[faces]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        d0 = b[:, :2] - a[:, :2]
        d1 = c[:, :2] - a[:, :2]
        den = d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]
        ok = np.abs(den) > 1e-14
        for f in np.nonzero(ok)[0]:
            p = flat - a[f, :2]
            u = (p[:, 0] * d1[f, 1] - p[:, 1] * d1[f, 0]) / den[f]
            v = (d0[f, 0] * p[:, 1] - d0[f, 1] * p[:, 0]) / den[f]
            inside = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
            if inside.any():
                z = (a[f, 2] + u[inside] * (b[f, 2] - a[f, 2])
                     + v[inside] * (c[f, 2] - a[f, 2]))
                np.maximum.at(h, np.nonzero(inside)[0], z)
    return h.reshape(cells.shape[:2])


def run(rep: int, design: dict, plan: dict, out_paths: dict) -> int:
    t_start = time.time()
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    rc = 1
    try:
        rc = run_inner(rep, design, plan, out_paths, t_start)
    except BaseException as exc:  # noqa: BLE001  (D447)
        import traceback
        fsync_write(out_paths["failure.json"], json.dumps(
            {"tag": TAG, "rep": rep, "error": repr(exc),
             "traceback": traceback.format_exc(),
             "wall_seconds": round(time.time() - t_start, 1)}, indent=1))
        rc = 3
    finally:
        sentinel = (f"PRE_CLOSE_SENTINEL rc={rc} tag={TAG} rep={rep} "
                    f"wall={time.time() - t_start:.1f}s\n")
        fsync_write(out_paths["exit_status.txt"], sentinel)
        print(f"[{LOG}] {sentinel.strip()}", flush=True)
        sys.stdout.flush()
        app.close()
    return rc


def run_inner(rep: int, design: dict, plan: dict, OUT: dict, t_start: float) -> int:
    import asyncio
    import carb
    import omni.kit.app
    import omni.physx
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.replicator.grasping")
    import isaacsim.replicator.grasping.grasping_utils as grasping_utils

    out: dict = {"tool": f"{TAG}-transfer", "rep": rep, "case": "y2_d454",
                 "prereg": "y2_prereg.md (3963212a)",
                 "plan": {"margins": plan["margins"], "pins": plan["pins"],
                          "env": plan["env"], "seed": SEED,
                          "place_targets_rc": plan["place_targets_rc"]}}

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(GRAVITY)

    def bind_material(prim, path, sf, df, rest):
        mat = UsdShade.Material.Define(stage, path)
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr().Set(sf)
        pm.CreateDynamicFrictionAttr().Set(df)
        pm.CreateRestitutionAttr().Set(rest)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    def static_box(path, center, half):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr().Set(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(half[0] * 2, half[1] * 2, half[2] * 2))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        bind_material(cube.GetPrim(), "/World/materials/static_mat",
                      SUPPORT_FRICTION, SUPPORT_FRICTION, 0.0)
        return cube.GetPrim()

    static_box(GROUND_PRIM, (0.2, 0.0, -0.05), (0.5, 0.5, 0.05))
    for reg, tagr in (("source", "src"), ("bin", "bin")):
        r = design[reg]
        cx, cy = r["center"]
        hl = r["L_int"] / 2
        t = 0.005
        h = r["wall_h_m"]
        for wname, c, half in (
                ("n", (cx, cy + hl + t / 2, h / 2), (hl + t, t / 2, h / 2)),
                ("s", (cx, cy - hl - t / 2, h / 2), (hl + t, t / 2, h / 2)),
                ("e", (cx + hl + t / 2, cy, h / 2), (t / 2, hl, h / 2)),
                ("w", (cx - hl - t / 2, cy, h / 2), (t / 2, hl, h / 2))):
            static_box(f"/World/tray_{tagr}/wall_{wname}", c, half)

    physx_iface = omni.physx.get_physx_interface()
    physx_sim = omni.physx.get_physx_simulation_interface()
    sq = omni.physx.get_physx_scene_query_interface()

    ctxst = {"step": -1}
    step_counter = {"n": 0}

    def on_step(dt):
        ctxst["step"] += 1
        step_counter["n"] += 1

    step_sub = physx_iface.subscribe_physics_step_events(on_step)  # noqa: F841

    min_sep_by_step: dict = {}

    def on_contact(headers, data):
        st = ctxst["step"]
        for h in headers:
            a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            if not (a0.startswith(ROCKS_ROOT) or a1.startswith(ROCKS_ROOT)):
                continue
            for i in range(h.contact_data_offset,
                           h.contact_data_offset + h.num_contact_data):
                s = float(data[i].separation)
                if st not in min_sep_by_step or s < min_sep_by_step[st]:
                    min_sep_by_step[st] = s

    contact_sub = physx_sim.subscribe_contact_report_events(on_contact)  # noqa: F841

    def raycast_cell(x, y):
        hit = sq.raycast_closest(carb.Float3(x, y, 0.5), carb.Float3(0, 0, -1), 1.0)
        if not hit["hit"]:
            raise RuntimeError(f"RAYCAST_MISS {x} {y}")
        return float(hit["position"][2]), str(hit["collision"])

    src_cells = region_cells(design["source"]["center"])
    bin_cells = region_cells(design["bin"]["center"])

    def obs_region(cells):
        H = np.zeros((13, 13))
        hits = np.empty((13, 13), dtype=object)
        for r_ in range(13):
            for c_ in range(13):
                H[r_, c_], hits[r_, c_] = raycast_cell(cells[r_, c_, 0],
                                                       cells[r_, c_, 1])
        return H, hits

    def world_pose(path):
        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tr = xf.ExtractTranslation()
        q = xf.ExtractRotationQuat()
        return (np.array([tr[0], tr[1], tr[2]]),
                np.array([q.GetReal(), *q.GetImaginary()]))

    rock_paths = [f"{ROCKS_ROOT}/{r['name']}" for r in plan["rocks"]]
    path_to_idx = {p: i for i, p in enumerate(rock_paths)}
    spawned = np.zeros(N_ROCKS, dtype=bool)
    skip_delta_until = np.full(N_ROCKS, -1)

    def author_rock(i: int, pos, quat):
        rk = plan["rocks"][i]
        path = rock_paths[i]
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr().Set([Gf.Vec3f(*v) for v in rk["verts"]])
        mesh.CreateFaceVertexCountsAttr().Set([3] * len(rk["faces"]))
        mesh.CreateFaceVertexIndicesAttr().Set(
            [int(x) for f in rk["faces"] for x in f])
        b = rk["r_bound_m"]
        mesh.CreateExtentAttr().Set([(-b, -b, -b), (b, b, b)])
        mesh.AddTranslateOp().Set(Gf.Vec3d(*pos))
        mesh.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(quat[0]),
                     Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))))
        prim = mesh.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexHull)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(rk["mass_kg"])
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        prb.CreateSolverPositionIterationCountAttr().Set(8)
        prb.CreateSolverVelocityIterationCountAttr().Set(1)
        prb.CreateMaxAngularVelocityAttr().Set(10.0)
        prb.CreateMaxLinearVelocityAttr().Set(10.0)
        prb.CreateMaxDepenetrationVelocityAttr().Set(5.0)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        bind_material(prim, "/World/materials/rock_mat",
                      STATIC_FRICTION, DYNAMIC_FRICTION, ROCK_RESTITUTION)
        spawned[i] = True

    pos_tr = np.full((T_ALLOC + 1, N_ROCKS, 3), np.nan)
    quat_tr = np.full((T_ALLOC + 1, N_ROCKS, 4), np.nan)
    max_speed = np.full(T_ALLOC + 1, np.nan)
    max_angspeed = np.full(T_ALLOC + 1, np.nan)

    def sample(t):
        for i, p in enumerate(rock_paths):
            if spawned[i]:
                pos_tr[t, i], quat_tr[t, i] = world_pose(p)

    tcur = {"t": 0}
    result = {"steps_run": 0}

    def world_verts_of(i, t):
        R = quat_to_R(quat_tr[t, i])
        return plan["rocks"][i]["verts"] @ R.T + pos_tr[t, i]

    def centroid_of(i, t):
        return world_verts_of(i, t).mean(axis=0)

    async def sim_one():
        t = tcur["t"] + 1
        tcur["t"] = t
        await grasping_utils.simulate_physics_async(1, DT, None, render=False)
        sample(t)
        both = (np.isfinite(pos_tr[t]).all(axis=1)
                & np.isfinite(pos_tr[t - 1]).all(axis=1)
                & (skip_delta_until < t))
        if not both.any():
            raise RuntimeError(f"NO_TRACKED_BODIES t={t}")
        dpos = np.linalg.norm((pos_tr[t] - pos_tr[t - 1])[both], axis=1)
        dots = np.clip(np.abs((quat_tr[t] * quat_tr[t - 1])[both].sum(axis=1)), 0, 1)
        max_speed[t] = float(dpos.max() / DT)
        max_angspeed[t] = float(2.0 * np.arccos(dots).max() / DT)
        result["steps_run"] = t
        return t

    async def settle(streak_need, cap, label):
        streak = 0
        for k in range(cap):
            t = await sim_one()
            if max_speed[t] < V_EPS and max_angspeed[t] < W_EPS:
                streak += 1
            else:
                streak = 0
            if streak >= streak_need:
                return k + 1, True
        return cap, False

    sc = design["source"]["center"]
    hl_src = design["source"]["L_int"] / 2
    bc = design["bin"]["center"]
    hl_bin = design["bin"]["L_int"] / 2

    def region_of(i, t):
        c = centroid_of(i, t)
        if abs(c[0] - sc[0]) <= hl_src and abs(c[1] - sc[1]) <= hl_src:
            return "src"
        if abs(c[0] - bc[0]) <= hl_bin and abs(c[1] - bc[1]) <= hl_bin:
            return "bin"
        return "out"

    cycles_out = []

    async def orchestrate():
        # selfcheck (yt3 승계)
        step_counter["n"] = 0
        await grasping_utils.simulate_physics_async(5, DT, None, render=False)
        z_floor, _ = raycast_cell(bc[0], bc[1])
        z_open, _ = raycast_cell(0.2, 0.0)
        wall_y = bc[1] - hl_bin - 0.0025
        z_wall, hit_wall = raycast_cell(bc[0], wall_y)
        ok = (step_counter["n"] == 5 and abs(z_floor) < 5e-4 and abs(z_open) < 5e-4
              and abs(z_wall - design["bin"]["wall_h_m"]) < 1e-3
              and hit_wall.startswith("/World/tray_bin"))
        out["harness_selfcheck"] = {
            "step_events": step_counter["n"], "floor_z_m": z_floor,
            "open_ground_z_m": z_open, "wall_top_z_m": z_wall, "pass": bool(ok)}
        if not ok:
            raise RuntimeError(f"HARNESS_SELFCHECK_FAIL {out['harness_selfcheck']}")

        # 초기 스폰 (yt3 verbatim)
        ctxst["step"] = -1
        for w, members in enumerate(plan["waves"]):
            for mm in members:
                i_ = mm["rock_idx"]
                author_rock(i_, mm["pos"], plan["spawn"][i_]["quat"])
                p0, _ = world_pose(rock_paths[i_])
                if float(np.abs(p0 - np.array(mm["pos"])).max()) > 1e-9:
                    raise RuntimeError(f"WAVE_SPAWN_POSE_DEV w={w}")
                pos_tr[tcur["t"], i_] = p0
                quat_tr[tcur["t"], i_] = plan["spawn"][i_]["quat"]
            for k in range(WAVE_GAP_STEPS):
                t = await sim_one()
                if k == 0:
                    dz1 = min(float(mm["pos"][2] - pos_tr[t, mm["rock_idx"], 2])
                              for mm in members)
                    if dz1 < 0.0005:
                        raise RuntimeError(f"ROCKS_NOT_ACTIVATED_WAVE{w}")
        _, init_ok = await settle(INIT_STREAK, INIT_T_MAX - tcur["t"], "init")
        t0 = tcur["t"]
        regions0 = [region_of(i, t0) for i in range(N_ROCKS)]
        g_initial = bool(init_ok and all(r_ == "src" for r_ in regions0))
        out["initial"] = {"settled": init_ok, "settle_step": t0,
                          "n_in_src": regions0.count("src"), "pass": g_initial}
        if not g_initial:
            raise RuntimeError(f"G_INITIAL_FAIL {out['initial']}")
        print(f"[{LOG}] initial pile OK at step {t0} (32/32 src)", flush=True)

        H_src, _ = obs_region(src_cells)
        H_bin, _ = obs_region(bin_cells)

        # ---- 32 cycles ---- #
        for cyc in range(N_CYCLES):
            rr_, cc_ = np.unravel_index(int(np.argmax(H_src)), (13, 13))
            h_pick = float(H_src[rr_, cc_])
            px, py = src_cells[rr_, cc_]
            hit_z, hit_path = raycast_cell(px, py)
            pick_valid = (hit_path in path_to_idx and h_pick >= PICK_MIN_H
                          and abs(hit_z - h_pick) < 1e-6)
            if not pick_valid:
                cycles_out.append({"cycle": cyc, "pick_valid": False,
                                   "hit_path": hit_path, "h_pick": h_pick})
                raise RuntimeError(f"G_PICK_INVALID cyc={cyc} {hit_path}")
            i_ = path_to_idx[hit_path]
            t_pre = tcur["t"]
            wv_pre = world_verts_of(i_, t_pre)
            fp_min = wv_pre[:, :2].min(axis=0) - FOOTPRINT_PAD
            fp_max = wv_pre[:, :2].max(axis=0) + FOOTPRINT_PAD
            fp_mask = ((src_cells[..., 0] >= fp_min[0]) & (src_cells[..., 0] <= fp_max[0])
                       & (src_cells[..., 1] >= fp_min[1]) & (src_cells[..., 1] <= fp_max[1]))
            q_now = quat_tr[t_pre, i_].copy()

            # 목표 셀 + 낙하 z (관측의 결정론적 함수)
            tr_, tc_ = plan["place_targets_rc"][cyc]
            tx, ty = bin_cells[tr_, tc_]
            drop_z = max(DROP_Z_MIN,
                         float(H_bin.max()) + plan["rocks"][i_]["r_bound_m"] + DROP_CLEAR)

            # pick 제거 + place 저작
            stage.RemovePrim(rock_paths[i_])
            if stage.GetPrimAtPath(rock_paths[i_]).IsValid():
                raise RuntimeError(f"REMOVE_FAIL cyc={cyc}")
            spawned[i_] = False
            author_rock(i_, (tx, ty, drop_z), q_now)
            # trace 행은 덮어쓰지 않음 — 재배치는 t_pre→t_pre+1 사이 점프로
            # 정직하게 남기고, 속도 계산만 1 step 마스크
            skip_delta_until[i_] = tcur["t"] + 1

            n_steps, ok_settle = await settle(CYCLE_STREAK, CYCLE_CAP, f"cyc{cyc}")
            t_now = tcur["t"]
            dz_first = drop_z - pos_tr[min(t_pre + 1, t_now), i_, 2]
            if dz_first < 0.0005:
                raise RuntimeError(f"DROP_NOT_ACTIVATED cyc={cyc}")

            H_src_new, _ = obs_region(src_cells)
            H_bin_new, _ = obs_region(bin_cells)
            reshape_cells = int(((np.abs(H_src_new - H_src) > RESHAPE_TOL)
                                 & ~fp_mask).sum())
            com = centroid_of(i_, t_now)
            disp_mm = float(np.hypot(com[0] - tx, com[1] - ty) * 1000)
            viol = int((H_bin_new > H_MAX + H_TOL).sum())
            regs = [region_of(i2, t_now) for i2 in range(N_ROCKS)]
            row = {
                "cycle": cyc, "pick_valid": True,
                "pick_cell_rc": [int(rr_), int(cc_)], "pick_h_mm": h_pick * 1000,
                "rock": plan["rocks"][i_]["name"],
                "place_target_rc": [int(tr_), int(tc_)],
                "drop_z_m": drop_z,
                "settle_steps": n_steps, "settled": bool(ok_settle),
                "reshape_cells_outside_fp": reshape_cells,
                "dispersion_mm": disp_mm,
                "rock_region_end": region_of(i_, t_now),
                "hmax_violation_cells": viol,
                "bin_hmax_mm": float(H_bin_new.max() * 1000),
                "n_src": regs.count("src"), "n_bin": regs.count("bin"),
                "n_out": regs.count("out"),
            }
            cycles_out.append(row)
            print(f"[{LOG}] cyc {cyc:02d} {row['rock']}: pick({rr_},{cc_}) "
                  f"h={h_pick * 1000:.1f}mm -> place({tr_},{tc_}) disp={disp_mm:.1f}mm "
                  f"reshape={reshape_cells} viol={viol} settled={ok_settle} "
                  f"[src {row['n_src']} bin {row['n_bin']} out {row['n_out']}]",
                  flush=True)
            if not ok_settle:
                raise RuntimeError(f"G_CYCLE_SETTLE_FAIL cyc={cyc}")
            H_src, H_bin = H_src_new, H_bin_new

    task = asyncio.ensure_future(orchestrate())
    kit = omni.kit.app.get_app()
    guard = 0
    while not task.done():
        kit.update()
        guard += 1
        if guard > 4000000:
            raise RuntimeError("EVENT_LOOP_GUARD")
    if task.exception() is not None:
        # 부분 cycle 기록 영속화 후 재던짐 (에피소드 실패 증거)
        out["cycles_partial"] = cycles_out
        fsync_write(CASE_DIR / f"{TAG}_partial.json",
                    json.dumps(out, indent=2, default=str) + "\n")
        raise task.exception()

    T = result["steps_run"]

    # ---- 종료 상태 + G-final-hmap ---- #
    world_verts = [world_verts_of(i, T) for i in range(N_ROCKS)]
    faces_list = [rk["faces"] for rk in plan["rocks"]]
    H_src_f, _ = obs_region(src_cells)
    H_bin_f, _ = obs_region(bin_cells)
    gt_src = gt_heightmap(src_cells, world_verts, faces_list)
    gt_bin = gt_heightmap(bin_cells, world_verts, faces_list)
    all_diff = np.abs(np.concatenate(
        [(H_src_f - gt_src).ravel(), (H_bin_f - gt_bin).ravel()]))
    frac_tol = float((all_diff <= 0.0005).mean())
    g_hmap = bool(frac_tol >= 0.95)

    regs_f = [region_of(i, T) for i in range(N_ROCKS)]
    n_src_f, n_bin_f = regs_f.count("src"), regs_f.count("bin")
    g_pick = all(r_.get("pick_valid") for r_ in cycles_out)
    g_cycle = all(r_.get("settled") for r_ in cycles_out)
    g_transfer = bool(n_src_f == 0 and n_bin_f == N_ROCKS)

    disp = np.array([r_["dispersion_mm"] for r_ in cycles_out])
    resh = np.array([r_["reshape_cells_outside_fp"] for r_ in cycles_out])
    viols = np.array([r_["hmax_violation_cells"] for r_ in cycles_out])

    gates = {"g_initial": True, "g_pick_valid": bool(g_pick),
             "g_cycle_settle": bool(g_cycle), "g_final_hmap": g_hmap}
    if TAG == "yp1":
        gates["g_transfer_complete"] = g_transfer
    fails = [k for k, v in gates.items() if not v]
    code = (f"Y2_{TAG.upper()}_ALL_GATES_PASS" if not fails
            else f"Y2_{TAG.upper()}_FAIL_" + "_".join(sorted(fails)).upper())

    out["cycles"] = cycles_out
    out["metrics"] = {
        "dispersion_mm": {"mean": float(disp.mean()), "p95": float(np.percentile(disp, 95)),
                          "max": float(disp.max())},
        "reshape_cells": {"mean": float(resh.mean()), "max": int(resh.max()),
                          "n_cycles_nonzero": int((resh > 0).sum())},
        "hmax_violations_final": int(viols[-1]) if len(viols) else 0,
        "hmax_violations_any": int(viols.max()) if len(viols) else 0,
        "n_cycles_with_violation": int((viols > 0).sum()),
        "bin_hmax_final_mm": float(H_bin_f.max() * 1000),
        "end_counts": {"src": n_src_f, "bin": n_bin_f,
                       "out": regs_f.count("out")},
        "total_steps": T,
        "final_hmap": {"frac_within_0p5mm": frac_tol,
                       "max_abs_diff_mm": float(all_diff.max() * 1000)}}
    out["gates"] = gates
    out["verdict"] = {
        "code": code,
        "non_claims": "no claim about grasp physics (pick is an abstraction), "
                      "real masses/friction, Kinect fidelity, policy optimality, "
                      "H_max enforcement (accounting only)"}
    out["wall_seconds"] = round(time.time() - t_start, 1)

    if rep == 2:
        out["_final_bin_obs"] = H_bin_f.tolist()
        fsync_write(OUT["results.json"], json.dumps(out, indent=2, default=str) + "\n")
        rep1 = json.loads((CASE_DIR / f"{TAG}_results.json").read_text())
        dh = np.abs(np.array(rep1["_final_bin_obs"]) - H_bin_f)
        d1 = np.array([r_["dispersion_mm"] for r_ in rep1["cycles"]])
        branch = ("i_repeatable" if float(dh.max()) <= 0.002
                  else "ii_record_per_cycle_state")
        cmp_ = {"max_dh_bin_mm": float(dh.max() * 1000),
                "max_disp_diff_mm": float(np.abs(d1 - disp).max()),
                "rep1_total_steps": rep1["metrics"]["total_steps"],
                "rep2_total_steps": T, "branch": branch}
        fsync_write(OUT["compare"], json.dumps(cmp_, indent=2) + "\n")
        print(f"[{LOG}] REPEAT max_dh_bin={cmp_['max_dh_bin_mm']:.3f}mm "
              f"branch={branch}", flush=True)
        print(f"[{LOG}] Y2_{TAG.upper()}_REP2_VERDICT={code}", flush=True)
        return 0

    out["_final_bin_obs"] = H_bin_f.tolist()
    np.savez_compressed(
        OUT["trace.npz"], pos=pos_tr[:T + 1], quat=quat_tr[:T + 1],
        max_speed=max_speed[:T + 1], max_angspeed=max_angspeed[:T + 1],
        hmap_src_final=H_src_f, hmap_bin_final=H_bin_f,
        gt_src_final=gt_src, gt_bin_final=gt_bin,
        src_cells=src_cells, bin_cells=bin_cells,
        names=np.array([r["name"] for r in plan["rocks"]]))
    with open(OUT["trace.npz"], "rb") as f:
        os.fsync(f.fileno())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    hm = {"source": (H_src_f, gt_src), "bin": (H_bin_f, gt_bin)}
    for row_i, reg in enumerate(("source", "bin")):
        obs_, gt_ = hm[reg]
        vmax = max(0.01, obs_.max()) * 1000
        for col_i, (arr, ttl) in enumerate(((obs_ * 1000, "obs"), (gt_ * 1000, "gt"),
                                            ((obs_ - gt_) * 1000, "diff"))):
            a = axes[row_i, col_i]
            vm = 2.0 if ttl == "diff" else vmax
            im = a.imshow(arr, origin="lower",
                          cmap="coolwarm" if ttl == "diff" else "viridis",
                          vmin=(-vm if ttl == "diff" else 0), vmax=vm)
            a.set_title(f"{reg} {ttl} [mm] (max={np.abs(arr).max():.1f})")
            fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle(f"{TAG} final heightmaps — {code}")
    fig.tight_layout()
    fig.savefig(OUT["hmap_png"], dpi=140)
    plt.close(fig)

    # ---- rerun (D341) ---- #
    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        raise RuntimeError(f"RERUN_VERSION {rr.__version__} != {RERUN_VERSION}")
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    app_id = f"roarm_y2_d454_{TAG}"

    def tray_wire(reg):
        r = design[reg]
        cx, cy = r["center"]
        hl2 = r["L_int"] / 2
        h = r["wall_h_m"]
        pts = [(cx - hl2, cy - hl2), (cx + hl2, cy - hl2), (cx + hl2, cy + hl2),
               (cx - hl2, cy + hl2), (cx - hl2, cy - hl2)]
        strips = [[[x, y, 0.0] for x, y in pts], [[x, y, h] for x, y in pts]]
        strips += [[[x, y, 0.0], [x, y, h]] for x, y in pts[:4]]
        return strips

    def hull_edges_world():
        strips = []
        for wv, rk in zip(world_verts, plan["rocks"]):
            seen = set()
            for f in rk["faces"]:
                for a_, b_ in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                    e = (min(a_, b_), max(a_, b_))
                    if e not in seen:
                        seen.add(e)
                        strips.append([wv[e[0]].tolist(), wv[e[1]].tolist()])
        return strips

    with rr.RecordingStream(app_id, recording_id=f"y2_d454_{TAG}", make_default=False,
                            send_properties=True) as rec:
        rec.save(str(OUT["timeline.rrd"]), write_footer=True)
        rec.log("scene/tray_src", rr.LineStrips3D(
            tray_wire("source"), colors=[[70, 130, 230]], radii=0.0008), static=True)
        rec.log("scene/tray_bin", rr.LineStrips3D(
            tray_wire("bin"), colors=[[240, 150, 30]], radii=0.0008), static=True)
        rec.log("rocks/final_hulls", rr.LineStrips3D(
            hull_edges_world(), colors=[[150, 150, 160]], radii=0.0004), static=True)
        for ent, cells_, H_ in (("heightmap/src_final", src_cells, H_src_f),
                                ("heightmap/bin_final", bin_cells, H_bin_f)):
            rec.log(ent, rr.Points3D(
                np.column_stack([cells_.reshape(-1, 2), H_.ravel()]),
                colors=[60, 200, 90], radii=0.0012), static=True)

        for t in range(0, T + 1):
            rec.reset_time()
            rec.set_time("settle_step", sequence=t)
            fin = np.isfinite(pos_tr[t]).all(axis=1)
            rec.log("rocks/centers", rr.Points3D(
                pos_tr[t][fin], colors=[200, 200, 90], radii=0.004))
            if t >= 1 and np.isfinite(max_speed[t]):
                rec.log("plots/max_speed_mps", rr.Scalars(float(max_speed[t])))

        rec.reset_time()
        rec.set_time("settle_step", sequence=out["initial"]["settle_step"])
        rec.set_time("cycle", sequence=-1)
        rec.log("events/phase", rr.TextLog(
            f"INITIAL PILE settled at {out['initial']['settle_step']} (32/32 src)",
            level=rr.TextLogLevel.INFO))
        step_c = out["initial"]["settle_step"]
        for r_ in cycles_out:
            step_c += r_["settle_steps"]
            rec.reset_time()
            rec.set_time("settle_step", sequence=step_c)
            rec.set_time("cycle", sequence=r_["cycle"])
            rec.log("events/phase", rr.TextLog(
                f"cyc {r_['cycle']:02d} PICK {tuple(r_['pick_cell_rc'])} "
                f"{r_['rock']} h={r_['pick_h_mm']:.1f}mm -> PLACE "
                f"{tuple(r_['place_target_rc'])} disp={r_['dispersion_mm']:.1f}mm "
                f"reshape={r_['reshape_cells_outside_fp']} "
                f"viol={r_['hmax_violation_cells']}",
                level=rr.TextLogLevel.INFO if r_["settled"] else rr.TextLogLevel.WARN))
            rec.log("plots/dispersion_mm", rr.Scalars(float(r_["dispersion_mm"])))
            rec.log("plots/reshape_cells", rr.Scalars(float(r_["reshape_cells_outside_fp"])))
            rec.log("plots/bin_hmax_mm", rr.Scalars(float(r_["bin_hmax_mm"])))
            rec.log("plots/hmax_violation_cells",
                    rr.Scalars(float(r_["hmax_violation_cells"])))
        rec.reset_time()
        rec.set_time("settle_step", sequence=T)
        rec.set_time("cycle", sequence=N_CYCLES - 1)
        for k, v in gates.items():
            rec.log("events/verdict", rr.TextLog(
                f"{k}={v}", level=rr.TextLogLevel.INFO if v else rr.TextLogLevel.WARN))
        rec.log("events/verdict", rr.TextLog(
            f"VERDICT {code} | disp mean {disp.mean():.1f}/p95 "
            f"{np.percentile(disp, 95):.1f}mm | reshape mean {resh.mean():.1f} | "
            f"viol_any {int(viols.max())} | end src {n_src_f} bin {n_bin_f}",
            level=rr.TextLogLevel.INFO if not fails else rr.TextLogLevel.WARN))

        summary_md = (
            f"# y2_d454 {TAG} — pick-place full-transfer probe\n\n"
            f"**VERDICT: {code}**\n\n"
            f"32 cycles: pick=argmax src heightmap cell -> ray-hit rock removed; "
            f"place=drop-author at target cell "
            f"({'9-point raster' if TAG == 'yp1' else 'fixed center (6,6)'}), "
            f"joint settle (streak {CYCLE_STREAK}, cap {CYCLE_CAP}).\n\n"
            f"dispersion mean {disp.mean():.1f} / p95 {np.percentile(disp, 95):.1f} "
            f"/ max {disp.max():.1f} mm; reshape cells mean {resh.mean():.1f} "
            f"(nonzero {int((resh > 0).sum())}/32); H_max violations any-cycle "
            f"{int(viols.max())} cells, final {int(viols[-1])}; bin h_max "
            f"{H_bin_f.max() * 1000:.1f} mm; end src {n_src_f} / bin {n_bin_f} / "
            f"out {regs_f.count('out')}.\n\n"
            f"Authority = {TAG}_results.json + {TAG}_trace.npz; Rerun is "
            f"inspection evidence only (D341).  Masses are 15%-infill estimates.")
        rec.log("metadata/run", rr.TextDocument(summary_md, media_type=rr.MediaType.MARKDOWN),
                static=True)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | verdict"),
                    rrb.Spatial3DView(origin="/", contents=["/scene/**", "/rocks/**",
                                                            "/heightmap/**"],
                                      name="2 | transfer + final heightmaps"),
                    rrb.TextLogView(origin="/events", contents="/events/**",
                                    name="3 | cycles + gates"),
                    column_shares=[0.24, 0.5, 0.26]),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/plots", contents="/plots/**",
                                       name="4 | per-cycle metrics + speed")),
                row_shares=[0.6, 0.4]),
            auto_layout=False, auto_views=False, collapse_panels=True)
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=60.0)
    blueprint.save(app_id, str(OUT["timeline.rbl"]))

    expected_entities = [
        "metadata/run", "scene/tray_src", "scene/tray_bin", "rocks/final_hulls",
        "rocks/centers", "heightmap/src_final", "heightmap/bin_final",
        "plots/max_speed_mps", "plots/dispersion_mm", "plots/reshape_cells",
        "plots/bin_hmax_mm", "plots/hmax_violation_cells",
        "events/phase", "events/verdict"]
    pts3 = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    lin3 = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    components = {
        "metadata/run": ["TextDocument:text"],
        "scene/tray_src": lin3, "scene/tray_bin": lin3, "rocks/final_hulls": lin3,
        "rocks/centers": pts3, "heightmap/src_final": pts3, "heightmap/bin_final": pts3,
        "plots/max_speed_mps": ["Scalars:scalars"],
        "plots/dispersion_mm": ["Scalars:scalars"],
        "plots/reshape_cells": ["Scalars:scalars"],
        "plots/bin_hmax_mm": ["Scalars:scalars"],
        "plots/hmax_violation_cells": ["Scalars:scalars"],
        "events/phase": ["TextLog:text", "TextLog:level"],
        "events/verdict": ["TextLog:text", "TextLog:level"]}
    validation = validate_rerun_artifact(
        OUT["timeline.rrd"],
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        exact_timeline_names=["blueprint", "log_time", "settle_step", "cycle"],
        expected_entity_components=components,
        blueprint_path=OUT["timeline.rbl"],
        screenshot_path=OUT["inspection.png"],
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=300.0)
    fsync_write(OUT["rerun_validation.json"],
                json.dumps(validation, indent=2, default=str) + "\n")
    out["rerun_validation_pass"] = bool(validation.get("pass"))
    print(f"[{LOG}] rerun_validation pass={validation.get('pass')} "
          f"errors={validation.get('errors')}", flush=True)

    out["artifacts"] = {p.name: {"sha256_16": sha256(p)[:16], "bytes": p.stat().st_size}
                        for k, p in OUT.items()
                        if p.exists() and k not in ("results.json", "failure.json")}
    fsync_write(OUT["results.json"], json.dumps(out, indent=2, default=str) + "\n")
    print(f"[{LOG}] results.json={sha256(OUT['results.json'])[:16]} "
          f"bytes={OUT['results.json'].stat().st_size}", flush=True)
    print(f"[{LOG}] Y2_{TAG.upper()}_VERDICT={code} gates={gates}", flush=True)
    return 0


def main() -> int:
    global TAG, LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, choices=(1, 2), required=True)
    ap.add_argument("--probe", choices=("yp1", "yp2"), required=True)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    if args.probe == "yp2" and args.rep == 2:
        raise SystemExit("yp2 rep2 not in prereg scope")
    TAG = args.probe
    LOG = f"y2_{args.probe}"
    design, plan, out_paths = preflight(args.rep, args.probe)
    fsync_write(out_paths["argv.txt"], " ".join([sys.executable, *sys.argv]) + "\n")
    if args.rep == 1:
        shutil.copyfile(__file__, out_paths["script.py.txt"])
    print(f"[{LOG}] preflight OK rep={args.rep}: margins={plan['margins']}, "
          f"targets={len(plan['place_targets_rc'])}, pins verified", flush=True)
    if args.preflight_only:
        return 0
    return run(args.rep, design, plan, out_paths)


if __name__ == "__main__":
    sys.exit(main())
