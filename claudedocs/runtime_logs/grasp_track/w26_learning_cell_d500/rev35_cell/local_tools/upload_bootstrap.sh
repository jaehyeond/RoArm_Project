#!/bin/bash
# 로컬 실행: 꾸러미·매니페스트·부트스트랩을 pod 에 올리고 sha 3/3 대조 뒤 부트스트랩을 분리 기동한다.
# 사용: SSH_HOST=root@IP SSH_PORT=NNNN bash upload_bootstrap.sh
set -u; : "${SSH_HOST:?}"; : "${SSH_PORT:?}"
R=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/rev35_cell
TAR=$(ls $R/bundle/w26_bundle_*.tar.gz | tail -1); MAN=$R/bundle/BUNDLE_MANIFEST_W26.json; BS=$R/pod/bootstrap_pod_w26.sh
SSHO="-i $HOME/.ssh/id_ed25519 -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes"
for i in $(seq 1 10); do ssh $SSHO $SSH_HOST 'echo SSH_OK; nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader' 2>/dev/null && break || { echo "ssh retry $i"; sleep 15; }; done
ssh $SSHO $SSH_HOST 'mkdir -p /workspace/w26_in' 2>/dev/null
scp -i $HOME/.ssh/id_ed25519 -P $SSH_PORT -o StrictHostKeyChecking=accept-new -q $TAR $MAN $BS $SSH_HOST:/workspace/w26_in/ || { echo UPLOAD_FAIL; exit 2; }
LOC=$(sha256sum $TAR $MAN $BS | awk '{print $1}' | sort); REM=$(ssh $SSHO $SSH_HOST "cd /workspace/w26_in && sha256sum $(basename $TAR) $(basename $MAN) $(basename $BS)" 2>/dev/null | awk '{print $1}' | sort)
[ "$LOC" = "$REM" ] && echo UPLOAD_SHA_3_OF_3 || { echo UPLOAD_SHA_MISMATCH; exit 3; }
ssh $SSHO $SSH_HOST "export LC_ALL=C; ( cd /workspace/w26_in && setsid nohup bash bootstrap_pod_w26.sh /workspace/w26_in/$(basename $TAR) /workspace/w26_in/$(basename $MAN) > /workspace/w26_in/bootstrap_launcher.out 2>&1 < /dev/null & ); sleep 3; head -3 /workspace/w26_bootstrap/bootstrap.stdout.txt" 2>/dev/null
echo BOOTSTRAP_LAUNCHED
