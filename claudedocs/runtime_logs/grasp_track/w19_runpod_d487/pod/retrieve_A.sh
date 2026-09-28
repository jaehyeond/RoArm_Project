#!/bin/bash
# A 회수: pod A2 run_01 → 로컬 미러 경로. _obj 제외. 회수 뒤 pod↔로컬 sha256 대조 → RETRIEVAL_RECEIPT.json. Terminate 는 이 스크립트가 하지 않는다(메인이 영수증 확인 후 MCP 로).
set -u
W19=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487
SSHO="-i $HOME/.ssh/id_ed25519 -p 11232 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes"; H=root@47.47.180.42
SCR=${SCR:-/tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-Project/5b4d278b-b94c-4472-a821-4387326bd3fe/scratchpad}
mkdir -p "$W19/A_full_cycle" "$SCR"
rsync -a -q -e "ssh $SSHO" --exclude "_obj" $H:/workspace/w19_out/A_full_cycle/run_01 "$W19/A_full_cycle/" && echo RSYNC_RUN01_OK
rsync -a -q -e "ssh $SSHO" $H:/workspace/w19_runs/ "$W19/A_full_cycle/podA2_runs_logs/" && echo RSYNC_LOGS_OK
ssh $SSHO $H 'export LC_ALL=C; cd /workspace/w19_out/A_full_cycle && find run_01 -type f -not -path "*/_obj/*" | sort | xargs sha256sum' 2>/dev/null > "$SCR/podA_run01_hashes.txt"
( cd "$W19/A_full_cycle" && find run_01 -type f -not -path "*/_obj/*" | sort | xargs sha256sum ) > "$SCR/localA_run01_hashes.txt"
python3 - "$SCR" "$W19" <<'PY'
import json, sys, time
S, W = sys.argv[1], sys.argv[2]
pod = {l.split()[1]: l.split()[0] for l in open(S + "/podA_run01_hashes.txt") if l.strip()}
loc = {l.split()[1]: l.split()[0] for l in open(S + "/localA_run01_hashes.txt") if l.strip()}
mism = [k for k in pod if loc.get(k) != pod[k]]; missing = [k for k in pod if k not in loc]
rec = {"artifact": "W19_A_RETRIEVAL_RECEIPT", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pod": "qfo6qnce4ti50d",
       "n_pod_files": len(pod), "n_local_files": len(loc), "n_mismatch": len(mism), "mismatch": mism, "missing_local": missing,
       "excluded": "_obj/", "files": pod}
open(W + "/A_full_cycle/RETRIEVAL_RECEIPT.json", "w").write(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
print("pod files", len(pod), "local", len(loc), "mismatch", len(mism), "missing", len(missing))
for k in sorted(pod):
    if k.endswith((".npz", ".json")) and "run_01/" in k and "HASH_VERIFICATION" not in k: print(" ", k, pod[k][:16])
PY
