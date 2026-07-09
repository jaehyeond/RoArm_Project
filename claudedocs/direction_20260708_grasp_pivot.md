# D322 Grasp Pivot Direction

## 교수님 지시 (2026-07-08 랩미팅)

1. 변수 최소화: 완결된 case 하나를 먼저 성공시키고, 성공 시 변수를 1개씩 추가한다. 기존 마찰 randomization 선행은 순서 역전이었다.
2. VLA는 추후다. 지금 보고 싶은 것은 RL이 되는 것이다.
3. 마찰 보정/randomization은 최후 단계다.
4. grasp 피벗: 비대칭 그리퍼(고정 조 + 가동 조)를 물체 옆에 정확히 정렬하는 것이 1차 관문이다.
5. 핵심 목표 능력은 위치가 랜덤해도 잡기다. 고정 위치 태스크를 랜덤 위치로 일반화하는 것이 목표다.
6. 원통 등 잘 잡히는 형태부터 시작한다. 형태 다양화는 나중이며, 장기 질문은 "다양한 물체가 되는가"이다.
7. 로봇은 나중에 바뀔 수 있다. 최종 산출물은 특정 정책이 아니라 파이프라인과 데이터 스펙이다.
8. 색상은 당분간 무시한다. 재질/마찰은 변경하지 않는다.

## G-사다리

1. G0a: 정렬. 신규 변수는 grasp 기하다.
2. G0b: 원통 D34 x H90 파지 + 들어올림. 첫 완결 case이며 프로포절 트리거다.
3. G1a: 위치 grid 민감도 곡선.
4. G1b: standalone PPO scratch 0% -> X% 커브. zero-action 대조와 학습 전후 영상으로 "RL로 되는 것"을 증명한다.
5. G2: 형태 다양화.
6. G3: grid place.
7. 실기 전이: 캘리퍼 보정을 먼저 한다.

## 그리퍼 실측 계약

- 실기 최대 개방각: 88.3도. URDF 기준 1.571rad.
- 접촉 반경: 약 43mm, 대략 0.75mm/deg.
- 실용 개구: 약 40~45mm.
- 오프셋: 물체 중심은 TCP에서 가동 조 방향 +x로 `(D/2 - 8mm)` 떨어진다.
- cmd 0~5도는 서보 stall 때문에 금지한다.
- 30mm anchor stall: 37.88도.
- G0b 이후 sim 계약: gripper joint lower 0.09rad + effort limit로 stall을 재현한다. D322 G0a에서는 활성화하지 않는다.

## 그리퍼 좌표계 정본 (D323)

- Live runtime에는 `hand_tcp`가 별도 body로 존재하지 않는다. TCP는 env contract로 계산된다:
  `TCP = link5 position + link5 rotation * [0, 0, 0.115428]`.
- D323 frame audit에서 세 정지 자세 모두 `TCP in link5 = [0, 0, 0.115428]m`로 확인됐다. 오차는 `0.000044~0.000063mm` 수준이다.
- 도구축은 `link5 local +z`다. 조 분리축 계약은 `link5 local +x`이며, 가동 조는 `+x` 쪽으로 스윙한다.
- 고정 조 파지면 proxy는 `TCP - link5 local x * 0.008m`로 둔다. `gripper_link` body origin은 파지면이 아니다. D323 정지 감사에서 `gripper_link in link5`는 대략 `[0, 0.018821, 0.052035]m`였다.
- D323에서 검사한 strict side-grasp family는 도달 불가능했다: `link5 +z`를 수평 반경 방향으로, `link5 +x`를 수평 접선 방향으로 맞추는 목표는 best strict attempt도 TCP `35.729mm`, link5 `+z` `43.015deg` error로 gate를 넘지 못했다.
- 반대로 위치만 보면 같은 TCP side target은 `0.261mm` error로 도달 가능하다. 그러나 그때 link5 `+z`는 radial에서 `69.124deg` 벗어난다. 그러므로 다음 G0a repair는 오프셋 수치 반복 튜닝이 아니라, 이 reachable wrist-axis family를 반영한 alignment criterion 재정의여야 한다.

## Tap Track Freeze

Tap track은 D321 결과를 최종 산출물로 동결한다. D321 결과는 1,920 accepted episodes, combined acceptance 96.0%다. 인수 가능한 자산은 DiffIK 접근, D256 reset, 검증기 + 물리성 게이트, conveyor, 평가 규약, script 0~999 대조군이다.

이후 grasp track에서 tap track 조건을 계속 확장하지 않는다. 필요 아이디어는 `claudedocs/BACKLOG.md`에 적고 Active Case로 돌아온다.
