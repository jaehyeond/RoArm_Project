# D346 Rerun 실제 화면 확인 — FAIL 기록

- 확인 방식: 두 PNG를 `view_image`, `detail=original`로 직접 열었다.
- Rerun 화면: 실제 래스터 `4800x2800`; 여덟 개 공간 패널과 수치·사건 표는 보였다.
- 실패 표시: link5/gripper의 live instance와 prototype 패널에는 callback에서 얻은
  충돌 조각이 없었다. 표에는 측정 불가 기호와 `live=FAIL` 경고가 보였다.
- 가림: Rerun의 loading/gRPC 알림이 오른쪽 위 패널 일부를 가렸다.
- 별도 판정 그림: `1076x665` 화면에 256개 witness와 128개 live audit 실패,
  부분/빈 형상 합집합을 거리 판정에 쓰지 않았다는 STOP 문구가 읽혔다.

따라서 수동 화면 완료 조건은 **FAIL**이다. 이는 attempt3 형상이 틀렸다는 판정이
아니다. 첫 callback 전에 확장 기능 모듈 import가 실패해 실제 형상을 측정하지 못한
상태와 화면이 일치한다는 확인이다. 수치 판정은 화면이 아니라 JSON/callback/해시가
권위이며, `g0a_pass=false`를 유지한다.
