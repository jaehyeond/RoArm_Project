# START_HERE.md

Last updated: 2026-09-11 — 현재 열린 자세에서 출구 기울임 실제 실행. 최신 세션 `claudedocs/session_20260911_790_tilt_execution.md`, 최신 결과 LEDGER`:589`, 영속 결정D484 유지.

**현재 HOME/P1이 아니다. 문 열린 롤90 정렬/기울임 자세다.** 마지막 q=[89.121094,55.019531,87.626953,22.851563,89.648438,29.267578]°. `outlet_tilt_01/execution_01`의 마지막 `tilt_40` 이후 조작0·포트닫힘. 마지막 토크200. 기존 P1/롤±5 전제의 scoop/place 드라이버를 바로 시작하지 말 것.

- 사용자 “다음 동작 진행해 … 보다 맞는 방향으로 판단후 … 진행해”에 따라 **현재 열린 상태에서 실행**. 새 scoop·HOME 복귀 없이 직전 잔류로 시험했다.
- **실제 시험**: 명목5cm 상승→공구 수직을 유지하며 롤−5→90 축정렬→출구쪽20도 목표.64개 계획 T122 발행·58.143003초·2935피드백. 실제 공구 기울기14.501953도, 출구 경사7.762060→22.001692도. 문29.267578도 유지. 수치는 관절 FK 재구성이지 외부 위치 측정이 아니다.
- **마지막 정지**: 어깨 목표50.004953도/실제55.019531도, 편차5.014578도>기존5도 기준. 최종 목표는 발행했으나 `completed=false`; 추가동작·자동복귀·문닫기·토크OFF 없이 유지했다. 기준 완화/PID 변경0. 안전정지와 배출 성공을 혼동하지 않는다.
- **계량/잔류 입력 대기**: 기울임 후 컵 포함 무게와 고정 jaw 잔류 개수를 사용자에게 질문했다. 현재 후 값은null. 이전gross22.28g·이번 컵약0.05g는 역사적 기준으로 보존하되, 기울임 직전 새 독립 계량으로 가장하지 않는다. 이전 사진 컵9.65g를 이번 tare로 쓰지 말 것.
- **자료 검수**: 계획704보간 자세/RRD 및 실기2935행 전체 RRD/RBL/footer/entity/timeline/component/readback/실제 PNG3검수PASS. `outlet_tilt_01/REPORT.md`, 실행 `analysis.json`, `command_audit.json`, `visual_01/inspection.json` 참조. S1과 팔의 투영 분리 최소42.93mm는 명목 샘플 경로의 CAD 검사이며 컵/실물 접촉 보증이 아니다.
- **코드 범위**: 기본 `RecordedArm` 롤±5 보호는 유지. 별도 인스턴스가 검증된64개 패킷만 허용. S1직결형에는 g18 간섭 원인servocrank가 없다(`scoop_grab_s1_design.py:1`). 임의 롤 확대는 여전히 금지. 실기 동결 드라이버의 실패 shell0 문제는 정본에서2로 수정했고 장치 없는 회귀2종PASS.
- 기존790/900는 문 목표 유지 후 첫 리프트 추가 개방0°. 순차 더미와 잔류 미계측으로 토크/질량 인과 우열은 미판정. 어깨P8/P48 시험1648행·W10은 기존 증거 그대로 보존.
- **Git**: 출구 기울임 산출물 `d497fbc`를 `git@github.com:jaehyeond/RoArm_Project.git` master에 push완료·원격해시 일치 확인. LFS2개18MB·원본41파일 커밋해시 검사·LFS fsck PASS. 정본 `boot_check_20260911/outlet_tilt_publish_01.json`; 확인 기록은 다음 문서 커밋으로 게시한다.

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
- **사용자 승인 후속 실물 단계(“진행해”)**:790 대조→단일 배출 기울임 경로 확인·실기→조건 고정5회. 현재 저장소 origin/master 커밋·push까지 요청됨.790 완료·출구 기울임 실기 후 정지 기준 초과와 계량 입력을 확인하는 중이다.
- **현재 실물 신규 변수: [출구 하향 기울임 경로]**. 출력 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/outlet_tilt_01/`. 컵 자체9cm/안지름7cm·사용자 수동 수신 위치 조정. 토크/PID는 앞 단계를 마쳐 현재 고정하며, 후 질량/잔류 입력 대기. 실기 재승인 질문은 불필요하다.
- **이전 배출 기울임 검토(역사)**: 잔류 사진·실제 배출 자세·S1 형상을 이용한 읽기/기구학 섭동 분석. 출력 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/release_tilt_review_01/`. 신규 검토 변수는 배출 회전 방향/각도이며 실제 배출 코드·형상·토크 조건은 이번 검토에서 변경하지 않는다.
- 실행 산출: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/{cell_DE_dt2e6_c,regression_sphere_resume_20260911,resume_20260911}/`.
- 확정 불변: S1 형상(입58·보울폭36.4·서보0~30°), PP 펠릿, **DEME 단독**, 배출 위치 고정, 로봇은 정해진 경로.
- 학습 대상 = 어디를 퍼는가(높이맵→양·남는 형상 예측→선택). 관절/그리퍼 제어는 학습 안 함.
- 동결: g18 전체, s1_v0, `y3_d455`, grasp track, W1~W9. W10 기존/이번 원자료도 덮어쓰지 말 것.

## Next concrete action / 승인 경계

1. **기울임 후 컵 포함 무게와 고정 jaw 잔류 개수**를 받아 추가 배출 관찰을 기록한다. 사용자에게 비동기 질문했으며 아직 답 없음. 사진/사용자 관찰 없이 성공/0알/추가0g을 만들지 않는다.
2. 현재 열린 기울임 자세를 T105로 확인한 뒤 다음 복귀 경로를 계획한다. P1/롤±5 시작으로 가정하면 안 된다. 마지막 어깨 편차가 기준을 넘었으므로 기준을 올려서 반복하지 말고 실제 하중/자세를 확인한다. 승인된 다음 절차는 배출법 결정 후 조건 고정5회지만, 이 단일 시험을5회 완료로 보고하지 않는다.
3. 현재 출력 `outlet_tilt_01/`에만 후속 증거를 새 파일로 추가한다. 이전 원시/실기 동결 소스·plan·manifest를 덮어쓰지 않는다. 추가 sim 변수/학습/형상/PID 변경은 이번 단계에 넣지 않는다.

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
