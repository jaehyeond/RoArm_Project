#!/bin/bash
# W26 pod 부트스트랩(W25 판에서 경로만 rev35_cell 로) — W19 v5(sha e81802e3…, 실제 성공본)에서 **경로만** W25 로 바꿨다: 로그 /workspace/w26_bootstrap,
# 매니페스트 미러 경로, 출력 영속화 exec/runs → /workspace/w26_out/runs. 핀·deme 휠 sha·정적 라이브러리 sha·검사 순서 동일.
# RunPod 컨테이너 안에서 root 로 실행. 로컬과 **같은 절대경로** 를 만든다(경로 치환 금지).
# 사용: bash bootstrap_pod_w26.sh <bundle.tar.gz> <BUNDLE_MANIFEST_W26.json>
set -euo pipefail
export LC_ALL=C
BUNDLE=${1:?bundle tar path}; MANIFEST=${2:?manifest json path}
LOGDIR=/workspace/w26_bootstrap; mkdir -p "$LOGDIR"
exec 1>>"$LOGDIR/bootstrap.stdout.txt"
exec 2>>"$LOGDIR/bootstrap.stderr.txt"
echo "== [$(date -u +%FT%TZ)] 0. host facts"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
ldd --version | head -1
test -f /usr/local/cuda/include/cuda.h && echo "cuda.h OK"
ls /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12 /usr/local/cuda/lib64/libnvrtc.so.12 2>/dev/null || echo "WARN: libnvrtc.so.12 not at expected paths"
/usr/local/cuda/bin/nvcc --version | tail -1 || echo "WARN: nvcc missing"
command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)
echo "== 1. miniconda at /home/cgxr/miniconda3 (경로 미러)"
if [ ! -x /home/cgxr/miniconda3/bin/conda ]; then
  mkdir -p /home/cgxr /workspace/dl
  curl -fsSL -o /workspace/dl/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /workspace/dl/miniconda.sh -b -p /home/cgxr/miniconda3
fi
/home/cgxr/miniconda3/bin/conda --version
if [ ! -x /home/cgxr/miniconda3/envs/roarm/bin/python ]; then
  /home/cgxr/miniconda3/bin/conda create -y -q -n roarm --override-channels -c conda-forge python=3.11
fi
PY=/home/cgxr/miniconda3/envs/roarm/bin/python
$PY --version
echo "== 2. pip pins (binary only, deme 휠 sha256 고정)"
$PY -m pip install -q --upgrade pip
mkdir -p /workspace/wheels
$PY -m pip download -q --only-binary=:all: --no-deps -d /workspace/wheels deme==2.4.0
echo "acf8a02920c0e9f8adfcbf7c23e30a68d9fa7f8e5007e845f2a4de8ed6cd8bb3  /workspace/wheels/deme-2.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" | sha256sum -c -
$PY -m pip install -q --only-binary=:all: \
  /workspace/wheels/deme-2.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
  numpy==2.2.6 scipy==1.17.1 trimesh==4.11.5 matplotlib==3.10.8 pillow==12.0.0 networkx==3.6.1 rtree==1.4.1 manifold3d==3.5.2 gymnasium==1.2.3
SP=$($PY -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
echo "site-packages=$SP"
echo "c47ea1c4f0744a50602a139acb22e3db53df3363e6db23a260d050a1ca620529  $SP/lib64/libsimulator_multi_gpu.a" | sha256sum -c -
ldd "$SP/deme/_deme.cpython-311-x86_64-linux-gnu.so" | grep -E "cuda|nvrtc|not found" || true
$PY -c "import DEME, numpy, scipy, trimesh, matplotlib; print('imports ok', DEME.__file__, numpy.__version__, scipy.__version__, trimesh.__version__)"
echo "== 3. bundle extract at / (절대경로 미러) + 해시 대조"
tar -C / -xzf "$BUNDLE"
EXEC=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/rev35_cell
mkdir -p "$EXEC/bundle" && cp "$MANIFEST" "$EXEC/bundle/BUNDLE_MANIFEST_W26.json"
mkdir -p /workspace/w26_out/runs /workspace/w26_runs /workspace/w26_smokes
if [ ! -L "$EXEC/runs" ]; then
  if [ -d "$EXEC/runs" ]; then rmdir "$EXEC/runs"; fi
  ln -s /workspace/w26_out/runs "$EXEC/runs"
fi
$PY "$EXEC/pod/verify_bundle.py" "$MANIFEST" "$LOGDIR/W26_POD_BUNDLE_VERIFY.json"
(cd /home/cgxr/Documents/Robotics/RoArm_Project && $PY -c "import roarm_rl.heightmap; print('roarm_rl.heightmap import ok')")
echo "== 4. env receipt"
$PY "$EXEC/pod/env_receipt.py" "$LOGDIR/W26_POD_ENV_RECEIPT.json"
echo "== [$(date -u +%FT%TZ)] BOOTSTRAP_OK"
