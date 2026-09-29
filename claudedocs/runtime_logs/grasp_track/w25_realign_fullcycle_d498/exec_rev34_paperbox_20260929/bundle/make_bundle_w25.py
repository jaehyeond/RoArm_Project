"""W25 RunPod 전송 꾸러미 — W19 make_bundle.py(원칙 동일: 절대경로 그대로 tar, sha256 매니페스트, 코드 변경 0)에서
동결 revision = exec_rev34_paperbox_20260929(rev34 바이트 사본 + 새 criteria + COMMANDS + pod 스크립트), 더미 = 2차 NPZ(67,737알) 로 바꿨다.
외부 동결 입력 13개는 W19 BUNDLE_MANIFEST.json 의 sha 와 같아야 한다(W13/W16/W19 영수증 연속성)."""
import hashlib, json, os, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # <exec>/bundle
EXEC = HERE.parent
MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
ORCA = Path("/home/cgxr/orca/workspaces/RoArm_Project")
W11 = MAIN / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911"
S1V1 = MAIN / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1"
S1V0 = MAIN / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v0"
PILE20K = ORCA / "pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz"
NPZ = MAIN / "claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/pile_flat40_20260929/pile_lens6_a4p5_b3p8_c2p5_slab40_outer_310x220_n67737_rho0p503_seed460.npz"
NPZ_SHA = "31dd289731a144de30bbd3054ffd8c233d9897347f3cd4eab2e07f97032583c1"
W19MAN = json.load(open(MAIN / "claudedocs/runtime_logs/grasp_track/w19_runpod_d487/bundle/BUNDLE_MANIFEST.json"))

EXT13 = [
    S1V0 / "design.json", S1V1 / "design.json", S1V1 / "door_ALL_jawframe.stl", S1V1 / "fixed_ALL.stl",
    MAIN / "hw_s1_manual.py", MAIN / "hw_s1_scoop_probe.py",
    MAIN / "local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd",
    MAIN / "roarm_rl/heightmap.py", MAIN / "roarm_rl/rerun_contract.py", MAIN / "roarm_rl/viz_debug.py",
    MAIN / "sim_deme_scoop_s1.py", MAIN / "sim_scripts/roarm_kinematics.py", PILE20K,
]
EXPECT_EXT13 = {str(p): W19MAN["files"][str(p)]["sha256"] for p in EXT13}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def walk(d, keep=None, skip_dirs=()):
    out = []
    for r, ds, fs in os.walk(d):
        ds[:] = [x for x in ds if x != "__pycache__" and (Path(r) / x) not in skip_dirs]
        for f in fs:
            if f.endswith(".pyc"):
                continue
            p = Path(r) / f
            if keep is None or keep(p):
                out.append(p)
    return sorted(out)

EXEC_SKIP_FILES = {EXEC / "EXEC_PIN.json", EXEC / "RUNPOD_LOG_W25.md"}   # 꾸러미 뒤에 쓰는 파일(순환 방지)
groups = {
    "w25_exec_frozen": [p for p in walk(EXEC, skip_dirs=(EXEC / "bundle", EXEC / "runs")) if p not in EXEC_SKIP_FILES],
    "external_frozen_inputs_13": EXT13,
    "w25_pile_npz_n67737": [NPZ],
    "runtime_extra_urdf_and_meshes": [MAIN / "local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf",
                                      MAIN / "local_assets/roarm_m3/urdf/roarm_m3.urdf"]
                                     + walk(MAIN / "local_assets/roarm_m3/urdf/meshes"),
    "runtime_extra_roarm_rl_pkg": walk(MAIN / "roarm_rl", keep=lambda p: p.suffix in (".py", ".json", ".yaml", ".txt")),
    "smoke_sphere_regression_inputs_w10_G0": [
        MAIN / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop/b_diverge/params_fixnorm_plunge25.json",
        MAIN / "claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz"],
}
manifest = {"artifact": "W25_BUNDLE_MANIFEST_V1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rule": "pod 에서 같은 절대경로에 풀고 verify_bundle.py 로 전 파일 sha256 대조. 불일치 → 실행 0. 러너(run_w25.py)도 매 step 전 전수 대조.",
            "groups": {}, "files": {}}
missing = []
for g, files in groups.items():
    manifest["groups"][g] = [str(p) for p in files]
    for p in files:
        if not p.exists():
            missing.append(str(p)); continue
        manifest["files"][str(p)] = {"sha256": sha(p), "size": p.stat().st_size}
if missing:
    raise SystemExit(f"누락 {len(missing)}: {missing}")
bad = {k: (v, manifest["files"][k]["sha256"]) for k, v in EXPECT_EXT13.items() if manifest["files"].get(k, {}).get("sha256") != v}
manifest["external_13_match_w19_manifest"] = not bad
manifest["external_13_mismatch"] = bad
if bad:
    raise SystemExit(f"외부 동결 입력 해시 불일치(W19 대비): {bad}")
if manifest["files"][str(NPZ)]["sha256"] != NPZ_SHA:
    raise SystemExit("2차 NPZ sha 불일치")
manifest["pile_npz_n67737_sha256_match"] = True
tag = time.strftime("%Y%m%d_%H%M")
tar_path = HERE / f"w25_bundle_{tag}.tar.gz"
if tar_path.exists():
    raise SystemExit(f"이미 있다(덮어쓰지 않는다): {tar_path}")
with tarfile.open(tar_path, "w:gz") as tf:
    for p in sorted(manifest["files"]):
        tf.add(p, arcname=p.lstrip("/"), recursive=False)
manifest["tarball"] = {"path": str(tar_path), "sha256": sha(tar_path), "size": tar_path.stat().st_size,
                       "n_files": len(manifest["files"]), "extract": "tar -C / -xzf <tarball>  (절대경로 미러)"}
(HERE / "BUNDLE_MANIFEST_W25.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
print(json.dumps({k: len(v) for k, v in groups.items()}, ensure_ascii=False))
print("tar", tar_path, manifest["tarball"]["size"], manifest["tarball"]["sha256"])
