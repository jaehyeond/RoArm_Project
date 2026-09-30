#!/bin/bash
# 로컬 실행: pod 에서 명령을 **분리 실행**(서브셸 괄호 + setsid nohup + fd 전부 리다이렉트)해 ssh 채널이 매달리지 않게 한다(W25 함정 반영).
# 사용: SSH_HOST=root@IP SSH_PORT=NNNN LOG=/workspace/.../x.log bash pod_exec.sh '<원격 명령>'
set -u; : "${SSH_HOST:?}"; : "${SSH_PORT:?}"; : "${LOG:?}"
SSHO="-i $HOME/.ssh/id_ed25519 -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes"
ssh $SSHO "$SSH_HOST" "export LC_ALL=C PYTHONDONTWRITEBYTECODE=1; mkdir -p \$(dirname $LOG); ( setsid nohup bash -c '$1' > $LOG 2> $LOG.err < /dev/null & ) ; sleep 2; echo LAUNCHED; head -c 300 $LOG 2>/dev/null"
