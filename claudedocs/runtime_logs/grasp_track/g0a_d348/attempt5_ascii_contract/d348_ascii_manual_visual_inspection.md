# D348 attempt5 Rerun 수동 화면 판독 — 통과

- 원본 해상도 `4800×2800` Rerun PNG와 attempt2 결정 그림을 직접 다시 확인했다.
- 조각 45의 네 형상과 전체 그리퍼의 네 형상이 모두 보였다.
- 정적 요약에서 `5% FROZEN`, `256/256 PASS`, `128/128 PASS`,
  `D347 HOME-near; q5=0 CLOSED`, `0 steps`, `D348 OFFLINE evidence replay`,
  `G0A=false`를 읽었다.
- 오른쪽 완료문 `PASS | D347 HOME-near | D348 offline | G0a=false`가 끝까지 보였다.
- 빈 완료 패널, `\\u...` 문자열, 누락 글리프 네모는 없었다.
- Rerun 0.34.1의 기계 화면 계약은 ASCII로 고정하고, 동일한 뜻의 한국어 설명은 세션
  문서와 사용자 브리핑에서 제공한다.
- 화면용 재기록에서 과학 재계산, PhysX 실행, 물리 step, cook 요청, 자산 쓰기, 목표
  질의는 모두 0회이며 `g0a_pass=false`를 유지한다.
