# 새 세션 재개 — 원래 W13 판정 수정부터 / 교수 dt 질의는 별도 준비

작성: 2026-09-16. 최신 사용자 요청은 이 세션에서 질의 검토·LFS 보류/명단·다음 프롬프트 작성까지다. 실제 새 연구 작업은 새 세션에서 한다. 과거 GO/COMMANDS를 재실행하지 않는다.

## 반드시 읽기

1. AGENTS.md → START_HERE.md → claudedocs/DECISIONS_ACTIVE.md → LEDGER_RECENT.md → relay/from_codex.md. Codex는 from_claude.md도 읽되 옛 GPU 차단 상태를 현재로 쓰지 않는다.
2. [최신 종료 세션](session_20260916_professor_physics_lfs_defer.md), [상세 물리/렌더 보고](research/professor_review_20260916/REPORT.md), [실제 전수 파라미터](research/professor_review_20260916/PARAMETERS_ALL.md).
3. [원래 재개 문서](CONTINUE_20260914_W13_REPAIR_PERFORMANCE.md)를 전체 읽고 §2 원자료/독립 감사 및 §4 첫 수정 case를 따른다.
4. [LFS 나중 명단](LFS_DEFERRED_20260916.md), `research/professor_review_20260916/{LFS_BEFORE.json,LFS_AFTER.json,PARAMETER_AUDIT.json,OFFICIAL_SOURCE_CROSSCHECK.json}`.
5. git status/worktree/branch를 확인한다. 아래 staged D는 원본 삭제가 아니다. 실제 파일·해시부터 확인한다.

## 현재 사실 / 함정

- W13: TIMEOUT 부분 실행, 원자료 FAIL2/재생 FAIL3, 전체 성공 아님. raw/rev28/post03 불변.
- 첫 원래 작업: phase-only11 vs25, source 바닥 containment 반례를 재현 → 새 revision 수정 → 기존버그 FAIL/수정본 PASS CPU 회귀 → 독립 식·원본 해시 대조. 기존 raw verdict를 소급 PASS로 바꾸지 않는다.
- 그다음 원래 계획: 재생3결함 → 운반 보유 원인 → 같은 조건의 짧은 성능 측정. 물리 재실행과 코드 판정 수정을 섞지 않는다.
- 교수 제안10µs는 이미 W10직전 cell_DE_c에서 실패했다. timeline 마지막2.56118s/13.0303m/s와 C++예외22,291.97m/s/rc134를 구분한다. 완결 result 없음. 2µs/1µs는 각1회이며 수렴 증명 아님.
- 1ms는 기존0.1ms fine-sync보다 크다. nominal config 유지로 actual cadence가 유지된다고 하지 않는다. float32 dt 누적/overshoot는 PARAMETER_AUDIT의 CPU 산술과 독립 DEME_SOURCE_BOUND부터 읽는다. 실제 새 셀 측정은 아직0.
- 7구=한 알 접촉 형상. 20,000독립강체,140,000구. 실제 폭3.601263mm 대 목표3.8mm(-5.23%). density JSON만 바꿔 NPZ mass/MOI가 자동 변경된다고 가정 금지.
- Hertz+이력마찰+Crr, default EXTENDED_TAYLOR. 모든 강체 Euler항/PP 점탄성/실물 보정이 검증된 것은 아니다. 형상·회전 모델 변경은 별도 case.
- Isaac는 저장 상태 표시. W12 scene step vs W13 표시 render-only 구분. 카메라 `focal_mm`는 프로젝트 필드명이며 USD authoring 단위와 같다고 단정 금지. 실측40mm렌즈 아님.
- LFS 대상1016경로는5worktree의index에서만제외/ignore. 원본보존/HEAD불변. 과거커밋의LFS포인터는남으므로ignore만으로push문제해결아님. push보류.

## 작업 범위와 출력

- 새 세션에 아래 요청문을 전달하면 첫 CPU 판정 수정만 진행한다. `w14_w13_raw_repair_d484/<새ID>/`를 신규 경로로 사용하고 착수 때 Active Case에 명시한다. 이 문서를 작성한 세션은 그 폴더를 만들거나 수정 case를 시작하지 않았다.
- 교수 dt 비교는 **후속 별도 case 제안서**부터 작성한다. 1/2/10µs 기존 증거,10µs 재현 필요성,100µs/1ms호환성 시험을 구분한다. 새 E/형상/입자수/제어간격 변경을 한 번에 섞지 않는다.
- dt 구간·시드/초기조건·실제 sync cadence·비교 지표·중단조건·총 벽시계/정리예산·출력경로를 제시하고 새 GPU 실행은 명시 승인받는다. 수렴 합격선을 사후 만들지 않는다.
- RRD/기하·궤적 실검수 필요 시 D324/D341을 따른다. 코드/스키마·해시 감사로 끝나는 부분만 생략 사유를 기록한다.
- 기존미디어삭제/자동재생성/임의중간restart/학습/PBD하이브리드/A-B-C/실물/설치/commit/push금지.

## 붙여넣을 요청문

```text
/home/cgxr/Documents/Robotics/RoArm_Project에서 이어서 작업해.
AGENTS.md와 START_HERE.md의 부팅 순서를 따른 뒤
claudedocs/CONTINUE_20260916_PHYSICS_AUDIT_DT.md를 전체 읽어.
연결된 최신 session, 상세 물리/렌더 보고, LFS 명단과 원래 9/14 continuation을 읽고
실제 원자료·독립 감사로 현재 상태를 복원해.

먼저 원래 계획대로 W13 원자료 판정2결함을 재현하고,
동결rev28/run_01/post03을 건드리지 않는 새 revision에서 수정해.
CPU 단위/회귀 테스트로 기존버그FAIL→수정본PASS,
독립식 대조와 원자료 해시 보존까지 진행하고 여기서 보고해.
새 출력은 claudedocs/runtime_logs/grasp_track/w14_w13_raw_repair_d484/<새ID>/ 아래에만 저장해.

그다음 교수님이 요청한 dt확대 비교의 별도 실행안을 준비해.
과거10µs실패와 W10 2µs/W11 1µs를 먼저 대조하고,
100µs/1ms에서는 실제 제어/진단 간격이 어떻게 바뀌는지 검토해.
물성·형상·알수·경로·문속도·기존보호선을 함께 바꾸지 마.
구간/초기조건/실제시각/지표/중단조건/벽시계·정리예산/출력경로를 제시해.
새 DEME/Isaac GPU 실행·재렌더는 이 실행안의 명시 승인 뒤에 해.
원래 후속인 재생3결함→운반보유원인→성능계측도 별도 case로 유지해.

작업자를 쓰면 적절한 Orca worktree로 배정하고 여기서 보고 받아.
Claude는 claude-opus-5, Codex는 gpt-5.6-sol high를 실제 확인해.
메인만 START/상태원장/relay를 소유하고 worker 원장을 merge하지 마.
LFS 명단의 staged D는 추적제외이며 원본은 남아 있어. 지우거나 force-add하지 마.
과거 LFS 이력이 남은 branch의 push/이력재작성은 하지 마.
실물조회·구동·PID/토크·카메라수집·설치·학습·PBD하이브리드·A/B/C도 금지해.
관찰 가능한 절차→수치→근거파일→한계·다음승인 순서로 한국어로 보고하고
종료 시 START_HERE·새 session·relay를 갱신해.
```
