#!/bin/bash
# 코디네이터: worker_done/escalation 만 깨움 조건으로 대기(생존 신호 무시). 배치가 오면 저장하고 종료(ack 는 메인이 읽은 뒤).
cd /home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/coordinator
read H RUN WT < ${IDS:-ids_priors.txt}
for i in $(seq 1 ${ROUNDS:-8}); do
  timeout 1000 orca-ide orchestration check --terminal "$H" --wait --types worker_done,escalation --timeout-ms 900000 --json > check_w_$i.json 2>/dev/null
  n=$(python3 - "$i" <<'PY'
import json,sys
txt=open(f'check_w_{sys.argv[1]}.json').read(); dec=json.JSONDecoder(); i=0; real=0
while i<len(txt):
    while i<len(txt) and txt[i] in ' \n\r\t': i+=1
    if i>=len(txt): break
    try:
        o,j=dec.raw_decode(txt,i); i=j
        for m in ((o.get('result') or {}).get('messages') or []):
            if not (m.get('type')=='heartbeat' or m.get('subject')=='alive'): real+=1
    except Exception:
        nl=txt.find('\n',i); i=nl+1 if nl>=0 else len(txt)
print(real)
PY
)
  echo "[$(date +%H:%M:%S)] round $i real=$n"
  if [ "${n:-0}" -ge 1 ]; then cp check_w_$i.json check_worker_result.json; echo WORKER_MESSAGE; exit 0; fi
done
echo NO_MESSAGE
