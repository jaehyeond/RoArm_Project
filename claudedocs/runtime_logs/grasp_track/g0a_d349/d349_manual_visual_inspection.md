# D349 수동 화면 검사

- 검사일: 2026-07-14
- 방법: `view_image`의 `original` 해상도
- 원본 Rerun 화면: `4800x2800`
- decision PNG: `1500x1125`
- 보조 exact-summary 화면: `3200x1800`
- 판정: **PASS**

원본 화면에서 link5/gripper 각각 source·live instance·prototype·candidate의 8개
공간 패널, raw/live 네 endpoint/vector, cylinder, target/commanded/actual frame을
확인했다. 오른쪽 위 로딩 알림이 모서리 일부를 덮지만 판정 대상 형상은 남아 있다.
decision PNG에서도 두 body의 raw/live clear witness와 세 TCP marker가 읽힌다.

원본 embedded collision blueprint는 네 static summary row를 현재 event-table
viewport에 표시하지 않았다. 외부 RBL 전환 두 번은 실패했고 두 화면을 그대로
보존했다. 이어 원본 RRD의 네 TextLog 문자열을 CLI로 직접 확인한 뒤, 같은 문자열과
원본 RRD/measurement SHA만 담은 비권위 `d349_summary_text_only.rrd`를 만들었다.
그 화면에서 OPEN q5, `(7,11)mm`, `steps=0`, 두 raw/live/delta, `0.1/0.5mm`,
`G0a=false`, `settle=separate`가 모두 선명하게 읽혔다.

보조 화면은 관찰 편의를 위한 복사본일 뿐이다. 수치 권위는 원 Float64 JSON이고,
live 거리는 direct PhysX narrowphase가 아닌 D347 callback-face surface proxy다.
이 검사는 settle·grasp·G0a·G0b·RL·ladder를 승인하거나 증명하지 않는다.
