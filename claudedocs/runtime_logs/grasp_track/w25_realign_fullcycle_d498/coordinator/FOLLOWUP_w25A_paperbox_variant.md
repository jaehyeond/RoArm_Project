# W25-A 후속 과제(같은 터미널 재사용) — 현재 실물 상자(종이 상자) + 4 cm 평평한 층 변형 params 와 CPU 사전검토

사용자 추가 입력(09-28 밤, HARD RULE #18): **현재 실물 상자는 임시 A4 종이 상자(윗단 바깥 31×22 cm·벽 0.2 cm·높이 23 cm)** 이고 그 안에 펠릿이 **벽까지 꽉 찬 상태로 바닥→윗면 약 4 cm**. NTC106 은 아직 구매 전 후보다. 재실행은 "지금 실물" 을 맞춰야 한다.

## 요구
1. `params_w25_paperbox.json` = `params_w25.json` 사본 + 다음만 변경(그 외 키 무변경): 트레이 안쪽 **310×220 mm**(기존 더미 NPZ `box_bounds_m` 규약 ±0.155/±0.11 과 일치; 실측 안쪽 ≈306×216 과의 4 mm 차는 `tray_inner_source: "declared_from_npz_convention (user outer 31x22, wall 0.2)"` 로 명시) · 트레이 높이 **230 mm** · 벽 두께 2 mm(선언) · 펠릿면 = 상자 바닥 + 40 mm(선언; `declared_pellet_cm` 는 받침 19.7 + 상자 바닥 두께(선언 0.2) + 4.0 = 23.9 로 두고 출처 명시) · 상자 중심 로봇 x 0.25 m · 규약 A 유지 · 절차 스위치 ON.
   - 두 번째 사본 `params_w25_paperbox_inner306.json`(안쪽 306×216) 도 만들어 두 경우 모두 사전검토.
2. 두 params 로 `preflight_geometry.py`·`preflight_rigid_hinge.py`·`preflight_snapshot.py` CPU 재실행 → 트레이 벽 230 mm 에 대한 간섭 여유(C10), 관절 제한(C11), 도메인 범위(x/y/z) 를 표로. 더미는 기존 20k NPZ 를 임시로 쓰되(표면 41 mm ≈ 4 cm) "TEMP" 표기; 40 mm 평평한 층 NPZ 는 W25-D 워커가 준비 중(경로 미정) — 있으면 그것으로.
3. `all_pass:false` 항목(C4 등)의 의미를 REPORT 에 한 줄씩(왜 false 인지, rev34 의도인지 결함인지).
4. `COMMANDS_w25_template.json` 에 paperbox 변형 실행 줄 추가(실행 금지).
5. 완료 시 `worker_done` 갱신 요약(성공/실패 명시) + `--report-path`.
금지 사항은 원 과제서 §0 그대로(GPU·DEME 물리·RunPod·설치·원장·다른 worktree 쓰기 0).
