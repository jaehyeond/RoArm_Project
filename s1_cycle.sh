#!/usr/bin/env bash
# S1 그랩 퍼내기·놓기 반복 실행기 — 횟수를 묻고 그만큼 above → scoop → place → above 를 돌린 뒤 HOME 으로 복귀.
# 사용:  ./s1_cycle.sh          (횟수 물어봄)      ./s1_cycle.sh 5     (바로 5회)
# 설정은 환경변수로:  PORT=/dev/ttyUSB0 BASE=38 PELLET=26 BOXTOP=38.5 PLACE=90 ./s1_cycle.sh
#   BASE=바닥→베이스판 윗면 cm · PELLET=바닥→펠릿면 cm · BOXTOP=바닥→상자 윗단 cm · PLACE=놓기 베이스 각(+y=90)
#   SIM=1 이면 로봇 없이 화면 시뮬. 로그 = claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/manual_<시각>.jsonl
cd "$(dirname "$0")" || exit 1
PORT=${PORT:-/dev/ttyUSB0}; BASE=${BASE:-38}; PELLET=${PELLET:-26}; BOXTOP=${BOXTOP:-38.5}; PLACE=${PLACE:-90}
PY=$HOME/miniconda3/envs/roarm/bin/python
if [ -n "$1" ]; then N=$1; else read -rp "몇 회 반복할까요? " N; fi
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then echo "1 이상의 정수를 입력하세요"; exit 1; fi
if [ -z "$SIM" ] && [ ! -e "$PORT" ]; then echo "포트 $PORT 가 없습니다 — 로봇 전원·USB 확인"; exit 1; fi
echo "설정: 포트 $PORT · 베이스판 $BASE · 펠릿면 $PELLET · 상자 윗단 $BOXTOP cm · 놓기 각 $PLACE° · 반복 $N 회"
if [ -z "$SIM" ]; then read -rp "팔 주변(더미 상자·놓는 자리·회전 반경 50 cm)을 비웠으면 Enter, 취소는 Ctrl-C: " _; fi
if [ -n "$SIM" ]; then CONN="--sim"; else CONN="--port $PORT"; fi
exec "$PY" -u hw_s1_manual.py $CONN --base-cm "$BASE" --pellet-cm "$PELLET" --boxtop-cm "$BOXTOP" --place-deg "$PLACE" --script "cycle $N; quit"
