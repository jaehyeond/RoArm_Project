"""pod 환경 영수증 — 실행 전 한 번. GPU/드라이버/CUDA/파이썬/패키지/DEME 링크를 JSON 으로 남긴다(관측만)."""
import hashlib, json, os, platform, subprocess, sys, sysconfig, time
from pathlib import Path

def run(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return {"rc": r.returncode, "stdout": r.stdout.strip()[:4000], "stderr": r.stderr.strip()[:1000]}
    except Exception as e:  # noqa: BLE001
        return {"rc": None, "error": f"{type(e).__name__}: {e}"}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

sp = Path(sysconfig.get_paths()["purelib"])
so = sp / "deme/_deme.cpython-311-x86_64-linux-gnu.so"
arc = sp / "lib64/libsimulator_multi_gpu.a"
nvrtc_candidates = [Path("/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12"), Path("/usr/local/cuda/lib64/libnvrtc.so.12")]
nvrtc = next((p for p in nvrtc_candidates if p.exists()), None)
rec = {
    "artifact": "W19_POD_ENV_RECEIPT", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(), "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"), "runpod_gpu_count": os.environ.get("RUNPOD_GPU_COUNT"),
    "python": sys.version, "python_exe": sys.executable, "site_packages": str(sp),
    "glibc": platform.libc_ver(), "uname": platform.uname()._asdict(),
    "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv"]),
    "nvcc": run(["/usr/local/cuda/bin/nvcc", "--version"]),
    "cuda_include_cuda_h": Path("/usr/local/cuda/include/cuda.h").exists(),
    "libnvrtc_path": str(nvrtc) if nvrtc else None, "libnvrtc_sha256": sha(nvrtc) if nvrtc else None,
    "libnvrtc_sha256_local_reference": "2bb82d1a34b9fefa46aca357299aed66763d4d6613d015d5a591224c00fa7e5a",
    "deme_so_ldd": run(["ldd", str(so)]) if so.exists() else None,
    "deme_static_archive_sha256": sha(arc) if arc.exists() else None,
    "deme_static_archive_sha256_local_reference": "c47ea1c4f0744a50602a139acb22e3db53df3363e6db23a260d050a1ca620529",
    "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]),
    "df_workspace": run(["df", "-h", "/workspace"]), "df_root": run(["df", "-h", "/"]),
    "nproc": os.cpu_count(), "mem_kb_total": next((l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")), None),
}
try:
    import DEME  # noqa: F401
    rec["deme_import"] = {"ok": True, "file": DEME.__file__}
except Exception as e:  # noqa: BLE001
    rec["deme_import"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
print(json.dumps({k: rec[k] for k in ("hostname", "runpod_pod_id", "python_exe", "deme_import", "deme_static_archive_sha256", "libnvrtc_sha256")}, ensure_ascii=False))
