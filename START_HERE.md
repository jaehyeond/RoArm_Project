# START_HERE.md

Last updated: 2026-09-11 — Codex 후속 실기 준비·Git 게시 검증, **D484 `:30179`**. 최신 세션: `claudedocs/session_20260911_790_tilt_mass_git.md`.

후속 실행/commit/push는 사용자 승인됨. 사진 촬영 후 잔류 제거·실물 배치 복구 여부와 컵 입구 안지름·윗테두리 높이 입력 대기. `pre790_feedback_01/`은 T105 조회만 송신했고 부팅 문구 없이 응답107개를 보존했다. 이번 후속 단계의 이동 명령0;790·기울임·고정 조건5회는 아직 실행하지 않았다. Git은 이 저장소의 `origin/master`가 대상이며 게시 점검은 `boot_check_20260911/git_publish_01/`에 보존한다.

실물 절차는 승인됨. 사용자 정정: 앞 회차는 저울 미설치로 무게 미측정, 고정 jaw에 몇 알 잔류. “이번에는 무게를 재볼게 다시 해봐”로 수정900 1회 실행 완료. 실행 종료 시 P1 복귀·마지막 토크200·포트 닫힘. **사진 계량: 빈 컵9.65 g, 컵+펠릿29.60 g → 컵 배출19.95 g. 고정 jaw 잔류 약10~15알(사용자 추정), 회차별 변동 보고.** 사진 촬영 후 현재 로봇/컵 배치·잔류 제거 여부 미확인.
- P 비교 완료: 원시1648행, 두 교대 P8→48 어깨각 감소0.527344° 반복. 어깨 P16/I0 복귀 명령·부하 변화 관측(레지스터 읽기 아님). `boot_check_20260911/pid_hold_02/README.md`와 `visual_02/`에 CSV/RRD/PNG·검증·검수·실행 소스/해시 보존.
- 수정 전900 `torque900_02/`:3975행·문4.570→5.361°(+0.791°), 팔 이동 T122가 문 목표0→측정각으로 변경한 교란. 원시/기존 source는 그대로 보존.
- **수정900 `torque900_03/`**:3739행·71.7875s, 목표 유지 T12218/18 PASS·위반0. 문3.779→3.691°(−0.088°), 리프트 최대 추가 개방0°. RRD/CSV/PNG/실기 명령 검증·검수 완료. 계량/사진 정본 `torque900_03/operator_measurement_01.json`·`MEASUREMENT_01.md`. 다음은 배치/잔류 확인→동일 계약790→고정 조건 질량5회. G1~G3 완료3/G4~G5 미완료2. 초기 잔류·더미 동일성 미계측, 컵 배출량과 전체 포획량 구분.

배출 회전 검토 완료: `claudedocs/session_20260911_release_tilt_review.md`·`boot_check_20260911/release_tilt_review_01/REPORT.md`. 출구쪽5° 기울임 개념은 경사7.5→12.5°; 현재 손목 롤−5°는7.78°에 그침. 실기 배출 효과/실행 관절 경로는 미검증. 신규 로봇 구동0.

추가 브리핑: `claudedocs/session_20260911_video_w9_w10_review.md` — 지정 영상 워커 기록·W9/W10 원자료/화면 확인. W9는 W8F 재생, W10은 RRD이며 Isaac MP4 없음. 새 실험/구동 없음.

🟢 **재부팅 후 GPU 복구, W10 dt-only 토크 정지 완주.** 호스트 NVML/커널 580.178.04 일치, CUDA 연산 확인.
🟡 **실물 정합은 아직 미확정.** 최대속도5.329m/s·5m/s 초과1sync가 남았다. 이번 결과는 중앙 셀 한 번의 완주이며, 무조건 안정이나 실물 일치를 뜻하지 않는다.

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
- **현재 승인된 실물 단계**: 부팅/피드백 → 어깨 P8↔48 → 닫힘 상한900↔790 → 질량 계측. 신규 변수 P와 닫힘 토크 두 개. 첫 섭동의 문 목표 변경에 반응한 제어 계약 수정 포함(D483), 수정900 실기 검증 완료(D484), 배출19.95 g 1회 기록.790 비교/조건 고정 질량5회 대기. 출력 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/`. 추가 sim 변수 변경은 아직 미승인.
- **사용자 승인 후속 실물 단계(“진행해”)**:790 대조→단일 배출 기울임 경로 확인·실기→조건 고정5회. 이어 현재 저장소 origin/master 커밋·push까지 요청됨. 물리 배치 확인 입력 대기; 원격/브랜치 확인 완료.
- **사용자 요청 배출 기울임 검토**: 잔류 사진·실제 배출 자세·S1 형상을 이용한 읽기/기구학 섭동 분석. 출력 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/release_tilt_review_01/`. 신규 검토 변수는 배출 회전 방향/각도이며 실제 배출 코드·형상·토크 조건은 이번 검토에서 변경하지 않는다.
- 실행 산출: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/{cell_DE_dt2e6_c,regression_sphere_resume_20260911,resume_20260911}/`.
- 확정 불변: S1 형상(입58·보울폭36.4·서보0~30°), PP 펠릿, **DEME 단독**, 배출 위치 고정, 로봇은 정해진 경로.
- 학습 대상 = 어디를 퍼는가(높이맵→양·남는 형상 예측→선택). 관절/그리퍼 제어는 학습 안 함.
- 동결: g18 전체, s1_v0, `y3_d455`, grasp track, W1~W9. W10 기존/이번 원자료도 덮어쓰지 말 것.

## Next concrete action / 승인 경계

1. 수정900_03 사진 계량 기록 완료: 컵 배출19.95 g·잔류 약10~15알. 배치/잔류 제거 입력 후 같은 문 목표 유지 코드 `hw_measured_scoop.py --out .../torque790_01 --torque 790`로 1회 비교한다. 현재 측정 준비 입력 대기이며 재승인 질문이 아니다. 컵 안지름·윗테두리 높이는 그 다음 기울임 경로 확인용이다. 앞900_02 질량은 미측정(null), 이번 잔류 질량도 미측정. 조건 고정 반복5회에 자동 합산하지 않는다.
2. PID 및 수정900 실기 완료,790 비교/고정 조건 질량5회는 남음. 승인된 순차 절차는 재승인 없이 진행. 사용자 요청의 작은 배출 기울임은 오프라인 검토 완료. 실제 배출 동작 변경/형상 변경·추가 sim변수는 아직 도입하지 않았다.
3. 부은 각·렛지 각·한 입 뒤 절단면 각 실측 → 재료 보정. W10 -x 단면각은 후처리에 민감하므로 물성 정답으로 쓰지 않는다(단면 곡선/보고서 참조).
4. 결정층: 기존 산업 규칙(최고점+층·열)·실패 정의 도입은 후속 단계. 이번에 코드 구현하지 않음.
   영상의 후보 예측·선택 구조는 참고하되 현재 DEME는 미리 계산하는 검증/학습용(단일 W10 1890s). 즉석 후보 평가 속도를 확보했다고 가정하지 않는다.
5. 원장 잔여: 80th~81st의 W10 외 자산/실물/PID/문헌·산업 조사 항목은 아직 소급 등재하지 않았다. 이번 D482·LEDGER `:582`는 W10·구 회귀·관측만 다룬다.

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
2. `claudedocs/session_20260911_790_tilt_mass_git.md`, `claudedocs/session_20260911_real_boot_measurement.md`, `boot_check_20260911/GATES.md`, 실행별 README/plan/raw/analysis. W10은 `session_20260911_w10_reboot_resume.md`·`REPORT_w10.md`.
3. 다음 Claude는 `claudedocs/relay/from_codex.md`. 이전 `from_claude.md`·`RESUME_W10_20260911.md`의 GPU 차단/W10 ③' 미실행은 재개 전 기록이다.
4. 서보를 만지기 전 `docs/reference/servo_pid_st3215.md`와 `docs/reference/hardware.md`.

## Do not trust as current

`HANDOFF.md`·`TASKS.md`; `usd_s1/`(v0+hand_tcp1kg); D478 "토크8.0 필요"; "T:107=EPROM/부팅상한1000"; "펠릿=쌀알형·긴쪽3.8"; W8 유령 접촉 가설; 재부팅 전 GPU 차단/W10 다음 실행 상태.
