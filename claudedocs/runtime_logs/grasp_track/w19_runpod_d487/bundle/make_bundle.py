"""W19 RunPod 전송 꾸러미 — 로컬 동결 입력을 **절대경로 그대로** tar 로 묶고 sha256 매니페스트를 남긴다.

원칙
  · 코드 변경 0. rev32/src 와 외부 동결 입력(W13 run_01 영수증과 동일값)을 그대로 복사한다.
  · pod 에서는 같은 절대경로(/home/cgxr/...) 에 풀어 하드코딩 경로를 그대로 만족시킨다(경로 치환 금지).
  · 매니페스트의 sha256 은 pod 에서 verify_bundle.py 로 다시 대조한다. 불일치 → 실행 0.
"""
import hashlib, json, os, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
ORCA = Path("/home/cgxr/orca/workspaces/RoArm_Project")
REV32 = ORCA / "w16-profiling/claudedocs/runtime_logs/grasp_track/w16_profile_d486/rev32"
W11 = MAIN / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911"
S1V1 = MAIN / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1"
S1V0 = MAIN / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v0"
PILE = ORCA / "pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz"
W19 = MAIN / "claudedocs/runtime_logs/grasp_track/w19_runpod_d487"

# W13 run_01 / W16 영수증의 외부 동결 입력 13개 (MANIFEST_prospective_rev32.json 과 동일 목록)
EXT13 = [
    S1V0 / "design.json", S1V1 / "design.json", S1V1 / "door_ALL_jawframe.stl", S1V1 / "fixed_ALL.stl",
    MAIN / "hw_s1_manual.py", MAIN / "hw_s1_scoop_probe.py",
    MAIN / "local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd",
    MAIN / "roarm_rl/heightmap.py", MAIN / "roarm_rl/rerun_contract.py", MAIN / "roarm_rl/viz_debug.py",
    MAIN / "sim_deme_scoop_s1.py", MAIN / "sim_scripts/roarm_kinematics.py", PILE,
]
EXPECT_EXT13 = json.load(open(ORCA / "w16-profiling/claudedocs/runtime_logs/grasp_track/w16_profile_d486/MANIFEST_prospective_rev32.json"))["external_frozen_inputs_sha256"]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def walk(d, keep=None):
    out = []
    for r, ds, fs in os.walk(d):
        ds[:] = [x for x in ds if x != "__pycache__"]
        for f in fs:
            if f.endswith(".pyc"):
                continue
            p = Path(r) / f
            if keep is None or keep(p):
                out.append(p)
    return sorted(out)

groups = {
    "rev32_frozen_revision": walk(REV32),
    "external_frozen_inputs_13": EXT13,
    "runtime_extra_urdf_and_meshes": [MAIN / "local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf",
                                      MAIN / "local_assets/roarm_m3/urdf/roarm_m3.urdf"]
                                     + walk(MAIN / "local_assets/roarm_m3/urdf/meshes"),
    "runtime_extra_roarm_rl_pkg": walk(MAIN / "roarm_rl", keep=lambda p: p.suffix in (".py", ".json", ".yaml", ".txt")),
    "runB_inputs": [W11 / "params_w11_dt1e6.json"],
    "smoke_sphere_regression_inputs_w10_G0": [
        MAIN / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop/b_diverge/params_fixnorm_plunge25.json",
        MAIN / "claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz"],
    "w19_pod_scripts_and_params": walk(W19 / "pod") + walk(W19 / "B_scoop_repeat", keep=lambda p: p.suffix == ".json"),
}
# 자기 자신(꾸러미 산출물) 제외
for k in groups:
    groups[k] = [p for p in groups[k] if not str(p).startswith(str(HERE))]

manifest = {"artifact": "W19_BUNDLE_MANIFEST_V1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rule": "pod 에서 같은 절대경로에 풀고 verify_bundle.py 로 전 파일 sha256 대조. 불일치 → 실행 0.",
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
# 외부 13개는 W13/W16 영수증 값과 동일해야 한다
bad = {k: (v, manifest["files"][k]["sha256"]) for k, v in EXPECT_EXT13.items() if manifest["files"].get(k, {}).get("sha256") != v}
manifest["external_13_match_w13_w16_receipt"] = not bad
manifest["external_13_mismatch"] = bad
if bad:
    raise SystemExit(f"외부 동결 입력 해시 불일치: {bad}")

tag = time.strftime("%Y%m%d_%H%M")
tar_path = HERE / f"w19_bundle_{tag}.tar.gz"
if tar_path.exists():
    raise SystemExit(f"이미 있다(덮어쓰지 않는다): {tar_path}")
with tarfile.open(tar_path, "w:gz") as tf:
    for p in sorted(manifest["files"]):
        tf.add(p, arcname=p.lstrip("/"), recursive=False)
manifest["tarball"] = {"path": str(tar_path), "sha256": sha(tar_path), "size": tar_path.stat().st_size,
                       "n_files": len(manifest["files"]), "extract": "tar -C / -xzf <tarball>  (절대경로 미러)"}
(HERE / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
print(json.dumps({k: len(v) for k, v in groups.items()}, ensure_ascii=False))
print("tar", tar_path, manifest["tarball"]["size"], manifest["tarball"]["sha256"])
