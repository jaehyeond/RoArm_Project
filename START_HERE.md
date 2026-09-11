# START_HERE.md

Last updated: 2026-09-11 — Codex790 대조/롤 실기 후 사용자 사진에 따른 배출 방향 재검토. 최신 세션: `claudedocs/session_20260911_790_tilt_execution.md`. 신규 결과 LEDGER`:587~588`; D484 이후 새 영속 규칙 추가 없음.

**먼저 현재 자세를 확인할 것: P1 복귀 상태가 아니다.** `release_roll_01`은 문을 연 배출 위치에서 사용자의 방향 재검토 질문에 따라 중단했다. 마지막 피드백 `[89.121094,75.058594,43.242188,64.863281,-4.746094,29.267578]°`, 마지막 토크 명령200. Ctrl-C 후 포트 닫힘; 롤0 복원/문닫기/자동복귀/토크OFF 명령0. 기존 `hw_measured_scoop.py`의 HOME/P1 시작 조건으로 곧바로 재시작하지 말 것.

- **790 대조 완료** `boot_check_20260911/torque790_01/`:71.812935s·3905피드백. 문4.218750→4.130859°, 리프트 최대 추가 개방0°. T12218/18 명시 문 목표 유지·전체 RRD/검수PASS. 수정900도 추가 개방0°; 초기 더미/잔류 미계측으로 인과 우열 미판정.
- **이번 질량**: 컵 포함22.28g, 사용자 빈 컵 약0.05g → 배출 약22.23g. 이전 사진의 빈 컵9.65g을 이번 값에 적용하지 않는다. 사용자 컵 자체 높이9cm·안지름7cm, 컵은 배출 위치에 맞춘다고 명시.9cm를 바닥 기준 절대 배출 높이로 쓰지 않는다.
- **작은 롤 실제 실행** `boot_check_20260911/release_roll_01/`: 새로 퍼지 않고 잔류만 운반, 문30°에서 롤−5° 1회. 실제 롤 변화−4.658203°, 출구 경사7.504859→7.762060°(Δ0.257201°). 사용자 새 사진에 고정 jaw 잔류가 보인다. 회전 후 무게 미입력(null); 추가 배출0g/전량 배출 성공으로 단정 금지.304.698172s·15496행 전체 RRD/명령 검증·근접 방향 화면 검수 보존.
- **현재 우선 과제**: 사용자 “회전 말고 옆으로 눕혀야 하나” 질문의 방향 검토. 문이 열린 쪽이 낮아지는 전체 공구 기울임이 후보. `direction_review.json`의 이상적5°는 좌표계 개념이며 검증된 관절 경로가 아니다. 단순 롤 확대·무조건 손목 피치 변경·90° 눕히기·관절 보호 해제 금지. 실제 기울임 경로 확정 뒤5회 계량; 완료5/7·미완료2(G5/G6)·포기0.
- 이전 P 비교/PID 원시1648행·수정900 원시3739행/사진 배출19.95g·잔류 사용자추정10~15알은 원본 그대로 보존. `torque_comparison_01.json`은 두 토크의 제한된 순차 비교다.
- Git 직전 게시 `dd0074c`는 `git@github.com:jaehyeond/RoArm_Project.git` `master`에서 원격 일치 확인. 사용자 commit/push 승인은 유지되며 이번 실기/사진/검토도 같은 저장소에 별도 게시한다. 다른 worktree로 push하지 않는다.

추가 브리핑: `session_20260911_video_w9_w10_review.md`의 지정 영상 워커/W9/W10 근거. W9는 W8F 재생이며 W10은 RRD, Isaac MP4 없음. W10 실물 정합은 아직 미확정.

## Current verified truth — 이번 W10

- 정본 폴더: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/`.
- `cell_DE_dt2e6_c`: E5e6 유지·dt2e-6, 13:13:04→13:44:34 KST, rc0/1890s/스톨0. **포획541개 / 10.9592g**.
- 첫 닫힘 q3.307°·저항1.783022N·m, 재닫기 q3.069°·저항1.765356N·m, 둘 다 **servo_stall**. 문 하한 없음, 3N 보호 정지 아님.
- 최종 **명목 관절각3.069° ≠ 서보 환산5.569°**. 실제 메시 관절각2.993°(서보5.493°). 립 등가 간격6.152mm, 최종 립 물림0개.
- W8 옵션F 중앙은154개/3.1196g·문 하한 정지. 두 조건의 차이 전체를 dt 효과로 단정 금지(dt 단일 변수의 직전 대조는 `cell_DE_c`).
- 현재 코드 구 회귀 `regression_sphere_resume_20260911`: **287개**, 허용268~362, 두 닫힘servo_stall. 과거 회귀 source와 달라 새로 실행했다.
- **과학 G0~G3 4/4 + D341 PASS**. Rerun0.34.1: 툴/접촉/스칼라2774sync·입자64frame RRD 읽기 대조, RRD/RBL/검증 JSON/스크린샷/실제 육안 검수 완료.
- dt-only가 발산하지 않아 조건부 E1e8 강성 셀은 미실행. 물리 코드·params·더미·S1 형상·사전 등록 조건 불변.
- 보고서 `.../w10_deme_close_fix/REPORT_w10.md`, 교차검증 `.../resume_20260911/verification.json`, 재생 `.../cell_DE_dt2e6_c/scoop_s1_seed460_w10.rrd`.

## Active Case — single source of truth

- **Active: `scoop_v0` — sim↔실물 정합(`s1_v1_sim/`, `pellet_model/`)**. 이번 재개 신규 변수는 **dt 1e-5→2e-6 s만**. W10 지정 실험·관측·보고 완료, 후속 범위 결정 대기.
- **현재 승인된 실물 단계**: 부팅/피드백 → 어깨 P8↔48 → 닫힘 상한900↔790 → 질량 계측. 신규 변수 P와 닫힘 토크 두 개. 첫 섭동의 문 목표 변경에 반응한 제어 계약 수정 포함(D483), 수정900 실기 검증 완료(D484),790 비교도 완료. 출구 기울임/조건 고정 질량5회가 남음. 출력 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/`. 추가 sim 변수 변경은 아직 미승인.
- **사용자 승인 후속 실물 단계(“진행해”)**:790 대조→단일 배출 기울임 경로 확인·실기→조건 고정5회. 현재 저장소 origin/master 커밋·push까지 요청됨.790 완료·롤 시험 후 방향 재검토로 진행 중이며 새 실기 결과는 별도 추가한다.
- **현재 재개 입력 반영**: 컵 자체9cm/안지름7cm·사용자 수동 수신 위치 조정. `torque790_01/` 완료 후 `release_roll_01/`을 실행·중단했고 `direction_visual_01/`로 방향을 재검토했다. 기존 관절/그리퍼 보호 범위 유지, 승인 반복 질문 없이 진행한다.
- **사용자 요청 배출 기울임 검토**: 잔류 사진·실제 배출 자세·S1 형상을 이용한 읽기/기구학 섭동 분석. 출력 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/release_tilt_review_01/`. 신규 검토 변수는 배출 회전 방향/각도이며 실제 배출 코드·형상·토크 조건은 이번 검토에서 변경하지 않는다.
- 실행 산출: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/{cell_DE_dt2e6_c,regression_sphere_resume_20260911,resume_20260911}/`.
- 확정 불변: S1 형상(입58·보울폭36.4·서보0~30°), PP 펠릿, **DEME 단독**, 배출 위치 고정, 로봇은 정해진 경로.
- 학습 대상 = 어디를 퍼는가(높이맵→양·남는 형상 예측→선택). 관절/그리퍼 제어는 학습 안 함.
- 동결: g18 전체, s1_v0, `y3_d455`, grasp track, W1~W9. W10 기존/이번 원자료도 덮어쓰지 말 것.

## Next concrete action / 승인 경계

1. 현재 열린 배출 자세를 시작점으로 복귀/출구 하향 기울임의 관절 경로와 간섭을 먼저 확인한다. 기존 실행 승인은 유효하므로 같은 승인을 다시 묻는 절차로 만들지 않는다. 사용자의 최신 요청은 사진/방향 검토이며 검증되지 않은 기울임을 즉시 명령하지 않았다.
2. 바닥의 출구쪽 경사를 실제로 키우는 단일 기울임을 시험하고 전후 무게/잔류를 기록한 뒤 고정 조건5회로 간다. 롤−5°는 출구 경사 변화가 작고 사진에 잔류가 있어 최종 배출법으로 고정하지 않았다. 잔류 질량·회전 후 무게가 없으면 null 유지.
3. W10 sim↔실물 물성 보정은 부은 각·단면각 실측 등 기존 다음 단계이며 이번에 새 sim 변수/학습/형상 수정은 도입하지 않는다. 결정층 연구의 기존 맥락/근거는 앞 세션 문서 그대로 보존한다.

## 기존 인계에서 유지하는 맥락 (이번에 재실행하지 않음)

- 실물 형상 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/`(79th). 자산 `local_assets/roarm_m3/{urdf/roarm_m3_s1_v1.urdf,usd_s1_v1/}` = inertial 없는 hand_tcp의 가짜1kg 제거판. D478 "팔 토크8.0 필요" 재사용 금지.
- 사용자 펠릿 실측4.5×3.8×2.5mm, 렌즈형. `claudedocs/runtime_logs/pellet_model/pellet_measured_20260910.json`.
- DEME 단독 물리, Isaac은 로봇/환경/렌더 담당. W9 기존 재생 완료. 원자료 위치와 이전 실험 맥락은 `claudedocs/relay/from_claude.md`·W8/W9 보고서.
- 조사 `claudedocs/research/survey_20260910/`: 상자에 평평하게 채운 초기 상태·최고점+층·열 규칙·단면각 보정 설계. 실물 PID는 `docs/reference/servo_pid_st3215.md`.

## Open risks / do-not-repeat

- GPU는 호스트 실행 필요: 샌드박스 NVML/CUDA 접근 실패를 드라이버 재고장으로 단정 금지. 호스트 검사 후 실행.
- 원장은 배타 소유. Codex의 상태/relay 갱신 종료 후 다음 세션이 이어받는다. 커밋은 사용자 요청 시만.
- 높이 추정 금지(줄자 실측), 손목90° 초과 금지, 맨 `{"T":106}` 금지, 그리퍼 관절 I 켜지 말 것.
- RRD 기본 커서는 시작 시각이고 중앙/오른쪽은 최종/첫 토크 정지 static 사본. 힌지 프레임 전체는 별도 decision_frames PNG. 실제 검수 한계는 inspection JSON 참조.
- 근접 RTX 검정면(D480)은 기존 미해결. Orca 코디네이터 바인딩 함정은 이전 relay 참조.

## Must read first

1. `AGENTS.md` → `claudedocs/DECISIONS_ACTIVE.md` / `LEDGER_RECENT.md`.
2. `claudedocs/session_20260911_790_tilt_execution.md`, `claudedocs/session_20260911_790_tilt_mass_git.md`, `claudedocs/session_20260911_real_boot_measurement.md`, `boot_check_20260911/GATES.md`, 실행별 README/plan/raw/analysis. W10은 `session_20260911_w10_reboot_resume.md`·`REPORT_w10.md`.
3. 다음 Claude는 `claudedocs/relay/from_codex.md`. 이전 `from_claude.md`·`RESUME_W10_20260911.md`의 GPU 차단/W10 ③' 미실행은 재개 전 기록이다.
4. 서보를 만지기 전 `docs/reference/servo_pid_st3215.md`와 `docs/reference/hardware.md`.

## Do not trust as current

`HANDOFF.md`·`TASKS.md`; `usd_s1/`(v0+hand_tcp1kg); D478 "토크8.0 필요"; "T:107=EPROM/부팅상한1000"; "펠릿=쌀알형·긴쪽3.8"; W8 유령 접촉 가설; 재부팅 전 GPU 차단/W10 다음 실행 상태.
