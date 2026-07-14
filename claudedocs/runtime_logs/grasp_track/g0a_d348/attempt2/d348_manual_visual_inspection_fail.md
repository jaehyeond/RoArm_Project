# D348 attempt2 Rerun 수동 검사 — FAIL

- 검사 방법: 두 PNG를 `view_image(detail=original)`로 원본 해상도 확인
- 과학 판정 PNG: 수치와 `part_045` 두 형상 차이가 명확함
- Rerun 기하: `part_045` 네 패널과 gripper 네 패널이 모두 보임
- 실패 1: 기본 `part_idx` timeline 때문에 event 패널이 비어 HOME/최종 판정을 읽을 수 없음
- 실패 2: per-part dataframe이 기본 시점에서 `-`로 보여 5%, 256/256, 128/128을 읽을 수 없음
- 실패 3: 논리 창은 2400×1400이지만 HiDPI raster는 4800×2800이며 attempt2 checker가 이를 허용하지 않음
- 수동 판정: **FAIL**
- 과학 결과 변경: 없음
- 다음: attempt2 전부 보존, 과학 재계산 없이 Rerun 정적 요약/HiDPI 계약만 reactive repair
