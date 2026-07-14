# D350 manual visual inspection

- 검사 방식: `functions.view_image`, `detail=original`
- 검사 대상: 실제 Isaac Viewer PNG 6장 + 새 full-RRD headless PNG 1장
- 결과: `PASS`

실제 Viewer의 두 PhysX 화면은 전체 로봇/툴/원통과 초록색 실제 collider wire를
보여준다. 네 colored 화면은 whole-oblique, top, side, oblique 시점에서 동일한 동결
자세의 link5/gripper 두 색상군, fixed-jaw 영역, 노란 원통을 보여준다. 빈 화면, 손상,
시점 간 pose 불일치, 경로 교환은 보이지 않았다.

Rerun 공간 화면은 조립된 128-part subject, 원통, fixed-jaw component, frame과
centerline/radial/tangent/normal/gap 화살표를 보여준다. `Float64 metrics` 패널 자체는
보이지만 headless가 선택한 `part_idx`에서는 숫자 trace가 읽히지 않는다. 따라서 정확한
수치와 `64+64` 개수는 화면에서 세지 않고 Float64 JSON 및 exact entity/component/
timeline 계약을 권위로 사용한다.

이 검사는 과학 판정을 바꾸지 않는다. D350은 `MEASURED`, `aligned_pass=null`,
`g0a_pass=false`이며 settle/G0b/RL/ladder는 계속 금지된다.
