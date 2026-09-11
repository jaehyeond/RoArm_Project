#!/usr/bin/env bash
# W1 (09-09) Isaac 단계 재현 명령. 반드시 repo 루트에서, 한 번에 하나씩(다른 워커와 GPU 공유 — 띄우기 전 여유 ≥ 6 GB).
# 사용: bash run_w1_isaac_steps.sh convert|probe_on|probe_off|render
set -u
PY=~/miniconda3/envs/isaaclab/bin/python; export OMNI_KIT_ACCEPT_EULA=YES
D=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w1_usd_v1
USD=local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 6144 ] || { echo "GPU free ${free} MiB < 6144 — 중단"; exit 3; }
case "$1" in
  convert)   timeout -k 30 1500 $PY -u sim_urdf_to_usd.py local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf $USD --collider convex_hull --headless > $D/convert_stdout.log 2> $D/convert_stderr.log ;;
  probe_on)  timeout -k 30 1500 $PY -u $D/door_close_probe_isaaclab.py --headless --usd $USD --out $D > $D/probe_on_stdout.log 2> $D/probe_on_stderr.log ;;
  probe_off) timeout -k 30 1500 $PY -u $D/door_close_probe_isaaclab.py --headless --usd $USD --out $D --no-self-collision > $D/probe_off_stdout.log 2> $D/probe_off_stderr.log ;;
  render)    timeout -k 30 1500 $PY -u sim_render_s1_closeup.py $USD $D/closeup > $D/render_stdout.log 2> $D/render_stderr.log ;;
  *) echo "unknown step $1"; exit 1 ;;
esac
echo "rc=$?"
