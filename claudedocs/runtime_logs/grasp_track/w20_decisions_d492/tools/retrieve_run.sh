#!/bin/bash
# RunPod run 회수 — _obj 포함판 (W19 교훈 반영, 2026-09-19 D492).
#
# 왜 새 파일인가: W19 의 원본 `w19_runpod_d487/pod/retrieve_A.sh` 는 그 세션에서 실제로 돌아간
# 증거다. 내용을 고쳐 덮으면 "무엇이 실제로 실행됐나"의 기록이 깨진다. 그래서 원본은 그대로 두고
# 다음 실행부터 이 파일을 쓴다(폴더 forward-only 규칙과 같은 취지).
#
# W19 에서 잃은 것: 원본의 `--exclude "_obj"` 와 해시 대조 두 줄의 `-not -path "*/_obj/*"` 때문에
# 생산 중 만들어진 OBJ 4개(트레이·용기·고정부·문)가 회수되지 않았다. pod 종료 뒤 2개만
# 바이트 동일 로컬 사본으로 복원했고 2개는 미복원으로 남았다. W13 은 `_obj` 포함이었다.
# → 이 판은 `_obj` 를 회수하고 해시 대조에도 포함한다.
#
# Terminate 는 이 스크립트가 하지 않는다. 메인이 영수증을 확인한 뒤 별도로 한다.
#
# 사용법:
#   OUTDIR=<로컬 회수 루트> POD_ID=<id> SSH_PORT=<port> SSH_HOST=<user@ip> \
#   REMOTE_RUN=<pod 쪽 run 경로> RUN_NAME=<run_01 등> bash retrieve_run.sh
set -u

: "${OUTDIR:?OUTDIR 미지정}"; : "${POD_ID:?POD_ID 미지정}"
: "${SSH_PORT:?SSH_PORT 미지정}"; : "${SSH_HOST:?SSH_HOST 미지정}"
: "${REMOTE_RUN:?REMOTE_RUN 미지정 (예: /workspace/w19_out/A_full_cycle)}"
: "${RUN_NAME:=run_01}"
: "${REMOTE_LOGS:=}"

SSHO="-i $HOME/.ssh/id_ed25519 -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes"
SCR="${SCR:-$(mktemp -d)}"
mkdir -p "$OUTDIR" "$SCR"

# 1) 본체 회수 — 제외 없음(_obj 포함)
rsync -a -q -e "ssh $SSHO" "$SSH_HOST:$REMOTE_RUN/$RUN_NAME" "$OUTDIR/" && echo "RSYNC_${RUN_NAME}_OK"

# 2) 러너 로그(있으면)
if [ -n "$REMOTE_LOGS" ]; then
    rsync -a -q -e "ssh $SSHO" "$SSH_HOST:$REMOTE_LOGS/" "$OUTDIR/pod_runs_logs/" && echo RSYNC_LOGS_OK
fi

# 3) pod 쪽 / 로컬 해시 — 둘 다 _obj 포함
ssh $SSHO "$SSH_HOST" "export LC_ALL=C; cd $REMOTE_RUN && find $RUN_NAME -type f | sort | xargs sha256sum" \
    2>"$SCR/pod_hash.stderr.txt" > "$SCR/pod_hashes.txt"
( cd "$OUTDIR" && export LC_ALL=C && find "$RUN_NAME" -type f | sort | xargs sha256sum ) > "$SCR/local_hashes.txt"

python3 - "$SCR" "$OUTDIR" "$POD_ID" "$RUN_NAME" <<'PY'
import json, sys, time
S, W, POD, RUN = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
def load(p):
    d = {}
    for line in open(p):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        h, _, path = line.partition("  ")
        d[path] = h
    return d
pod, loc = load(S + "/pod_hashes.txt"), load(S + "/local_hashes.txt")
mism = sorted(k for k in pod if loc.get(k) != pod[k])
missing = sorted(k for k in pod if k not in loc)
extra = sorted(k for k in loc if k not in pod)
obj = sorted(k for k in pod if "/_obj/" in k)
rec = {"artifact": "RUN_RETRIEVAL_RECEIPT",
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "pod": POD, "run_name": RUN,
       "n_pod_files": len(pod), "n_local_files": len(loc),
       "n_mismatch": len(mism), "mismatch": mism,
       "missing_local": missing, "extra_local": extra,
       "excluded": None, "obj_included": True,
       "n_obj_files": len(obj), "obj_files": obj,
       "files": pod}
open(W + "/RETRIEVAL_RECEIPT.json", "w").write(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
print("pod", len(pod), "local", len(loc), "mismatch", len(mism),
      "missing", len(missing), "_obj", len(obj))
if mism or missing:
    print("RETRIEVAL_INCOMPLETE — pod 를 종료하지 말 것")
    sys.exit(1)
print("RETRIEVAL_COMPLETE")
PY
