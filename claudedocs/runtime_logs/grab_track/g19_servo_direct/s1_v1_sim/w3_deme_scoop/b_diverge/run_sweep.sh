#!/bin/bash
# W3b ① 귀속 스윕 — 셀 하나씩 순차(GPU 공유), 셀당 320 s 가드. 사용: ./run_sweep.sh <cell-name> [args...]  또는 ./run_sweep.sh all
cd /home/cgxr/Documents/Robotics/RoArm_Project
B=claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop/b_diverge
PY=~/miniconda3/envs/roarm/bin/python
COMMON="--scenario cut --cut-half 40 --sync 1e-3 --max-wall-s 270 --plunge 20 --approach 3"
declare -A CELLS=(
  [a_wall0]="--wall-extra 0"            # (a) 벽 1.6/캡 2.0 (입자 반경 2.08 보다 얇음)
  [a_wall6]="--wall-extra 6"            # (a) 벽 7.6/캡 8.0 (지름 이상)
  [b_vz5]="--vz 5"                      # (b) 하강 5 mm/s (스텝당 관입 1/5)
  [b_dt5e6]="--dt 5e-6"                 # (b) 적분 dt 절반
  [c_subdiv2]="--subdiv 2.0"            # (c) 삼각형 최대 변 2 mm (입자 반경 이하)
  [d_outer_only]="--groups outer,part"  # (d) 안쪽 면·캡 제거 → 이중 접촉 불가
  [d_no_caps]="--groups inner,outer,part"
  [e_flip_part]="--flip-part"           # (e) 파팅면만 일부러 뒤집기 = W3 결함 재현 (발산 예상)
  [e_flip_all]="--flip"                 # (e) 전체 법선 뒤집기(참고)
  [f_cd5]="--cd 5"                      # CD 주기(참고)
  [f_E1e8]="--E 1e8 --dt 2e-6"          # 강성(참고, W3 stiff 재현)
)
run_cell(){ n=$1; shift; timeout -k 10 320 $PY sim_deme_s1_diverge_min.py --out $B --cell $n $COMMON "$@" > $B/log_$n.txt 2>&1; echo "rc=$?" >> $B/log_$n.txt; grep -E '^\[.*diverged' $B/log_$n.txt | tail -1; }
if [ "$1" == "all" ]; then for n in e_flip_part a_wall0 a_wall6 b_vz5 b_dt5e6 c_subdiv2 d_outer_only d_no_caps e_flip_all f_cd5 f_E1e8; do run_cell $n ${CELLS[$n]}; done
else n=$1; shift; run_cell $n "${CELLS[$n]}" "$@"; fi
echo SWEEP_DONE >> $B/sweep.log
