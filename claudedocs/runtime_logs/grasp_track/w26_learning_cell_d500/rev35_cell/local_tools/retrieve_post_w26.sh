#!/bin/bash
# 로컬: 셀(과 선택적으로 smoke) 회수(_obj 포함·해시 대조) → 사전등록 관문 판정(cell_post) → rev36 데이터 행·다음 더미·그림.
# pod 종료는 하지 않는다(영수증 확인 후 메인이 따로). 사용: bash retrieve_post_w26.sh <pod_id> <tag> <user@ip> <port> <run_names...>
set -u
POD=$1; TAG=$2; HOST=$3; PORT=$4; shift 4
R=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/rev35_cell
C=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/rev36_chain
RET=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w20_decisions_d492/tools/retrieve_run.sh
PY=/home/cgxr/miniconda3/envs/roarm/bin/python
PILE=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/pile_flat40_20260929/pile_lens6_a4p5_b3p8_c2p5_slab40_outer_310x220_n67737_rho0p503_seed460.npz
OUT=$R/runs/$TAG; mkdir -p $OUT/pod_logs
for RN in "$@"; do
  OUTDIR=$OUT POD_ID=$POD SSH_PORT=$PORT SSH_HOST=$HOST REMOTE_RUN=$OUT RUN_NAME=$RN bash $RET || { echo "RETRIEVE_FAIL $RN"; exit 1; }
  mv $OUT/RETRIEVAL_RECEIPT.json $OUT/RETRIEVAL_RECEIPT_$RN.json
done
scp -i $HOME/.ssh/id_ed25519 -P $PORT -q $HOST:/workspace/w26_smokes/chain.log $HOST:/workspace/w26_runs/*.runner.* $OUT/pod_logs/
case " $* " in *" cell_01 "*)
  PARAMS=$($PY -c "import json;a=json.load(open('$OUT/cell_01/EXECUTION_RECEIPT.json'))['argv'];print(a[a.index('--params')+1])")
  $PY -B $R/post/cell_post.py $OUT/cell_01 $OUT/POST_cell_01.json
  $PY -B $C/chain/row.py $OUT/cell_01 $PILE $PARAMS $OUT/ROW_cell_01 --chain-json "{\"note\":\"W26 검증 셀 $TAG, 연쇄 아님\"}"
  $PY -B $C/chain/next_pile.py $OUT/cell_01 $PILE $PARAMS $OUT/NEXT_PILE_cell_01.npz
  $PY -B $C/chain/figures.py row $OUT/ROW_cell_01 $OUT/ROW_cell_01.png
esac
echo "RETRIEVE_POST_DONE $TAG $(date -Is)"
