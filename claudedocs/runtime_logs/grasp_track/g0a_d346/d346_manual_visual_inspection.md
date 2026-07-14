# D346 Rerun 실제 화면 확인 — FAIL 기록

두 PNG를 `view_image`, `detail=original`로 직접 열었다. Rerun 화면에는 여덟 개
공간 패널과 수치·사건 표가 보였지만, link5/gripper의 live instance 및 prototype
패널에는 callback 충돌 조각이 없었다. 표에는 측정 불가 값과 `live=FAIL` 경고가
보였고, viewer 알림이 오른쪽 위 패널 일부를 가렸다. 별도 판정 PNG도 256 witness와
128 live audit 실패 및 빈 형상 합집합을 거리 판정에 쓰지 않았다는 STOP을 표시했다.

따라서 수동 화면 완료 조건은 FAIL이다. 이는 attempt3 형상이 틀렸다는 판정이 아니라,
첫 callback 전에 검사 모듈 import가 실패한 상태와 화면이 일치한다는 확인이다.
`g0a_pass=false`를 유지한다.
