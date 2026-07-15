# D354 manual visual inspection

- 검사 일시: 2026-07-16 KST
- 검사 방법: `view_image` 원본 해상도(`original_resolution`)
- 판정 범위: 시각화 산출물이 비어 있거나 손상되지 않았고, 요구된 실제 Isaac/Rerun 판단 대상을 담는지만 확인했다. 이 검사는 자동 수치 판정이나 과학 verdict를 덮어쓰지 않는다.

## 원본 해상도 관찰

1. `d354_open_actual_physx_colliders.png` (1280x720): 실제 Isaac 장면에서 OPEN 그리퍼, 세워진 노란 실린더, 링크5와 이동 그리퍼를 감싼 초록 PhysX collider 외곽선이 함께 보인다. 프레임은 정상 렌더링됐고 빈 화면이나 잘린 핵심 대상이 없다.
2. `d354_decision_or_open_fallback_actual_physx_colliders.png` (1280x720): 마지막 clear 결정 자세의 실제 Isaac 장면과 초록 PhysX collider 외곽선이 보인다. OPEN 캡처와 파일 해시가 다르며, 그리퍼와 실린더의 결정 시점 상대 배치가 정상적으로 담겼다.
3. `d354_decision_or_open_fallback_colored_64plus64.png` (1280x720): 링크5 쪽 반투명 청록/보라 계열과 이동 그리퍼 쪽 회색/베이지 계열의 collider 묶음이 시각적으로 구분되고, 노란 실린더와 함께 보인다. 64+64 개별 collider 계약의 대상 두 집합을 육안으로 구분할 수 있다.
4. `d354_decision_or_open_fallback_side_geometry.png` (1280x720): 측면에서 실린더 상단/배럴 경계, 고정·이동 죠의 접촉 영역, 색상 오버레이로 표시된 inner patch 및 chord/witness 기하가 함께 보인다. 죠가 실린더 상단 림 근처에 기울어진 현재 자세도 분명하다.
5. `d354_zero_step_closure_geometry_rerun.png` (4800x2800): Rerun 3D 패널에 로봇/실린더, `live_jaw` 표면, 고정·이동 chord/patch와 q5 축·감소 방향 witness가 표시된다. 아래 Float64 metrics 패널에는 전체 q5 sample 타임라인과 raw/live 거리 곡선이, 오른쪽 events 패널에는 sample별 이벤트 행이 채워져 있다. 패널은 비어 있거나 손상되지 않았다.

## 수동 시각 계약

- actual Isaac OPEN 및 decision/fallback 자세: PASS
- 실제 PhysX collider 가시성: PASS
- link5 64와 moving gripper 64 색상 구분: PASS
- inner patch, fixed/moving chord, cylinder feature 가시성: PASS
- Rerun 전체 q5 동적 live surface/patch/witness 타임라인: PASS
- 빈 패널 또는 명백한 손상 없음: PASS

수동 시각 검사 결과는 PASS다. 단, 이는 관찰 가능성 계약만 통과했다는 뜻이며 `D354_CONTACT_ORDER_UNRESOLVED_FAIL_STOP` 과학 판정을 변경하지 않는다.
