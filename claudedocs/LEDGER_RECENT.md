# LEDGER_RECENT.md — 최근 실험 20건 요약 (부팅 read)

Last updated: 2026-09-14 — W13 단일 본 물리 TIMEOUT·원자료/부분 재생 실패 인계 반영. 원본 `EXPERIMENT_LEDGER.md` 595줄. 기존 prefix는 보존하고 끝에만 append.

## 권위와 읽는 방법

- 이 파일은 색인이다. 수치/판정 인용 전 `EXPERIMENT_LEDGER.md:<줄>`에서 링크된 세션/raw까지 확인한다.
- 현재 상태/다음 승인 범위는 `START_HERE.md`. 아래 과거 중간 자세나 “pending”은 당시 기록이며 현재 상태가 아니다.
- 선정: 날짜 행의 **append 순 마지막20개** = `:539~544` 6개 + `:582~595` 14개. 아래는 역순이다. 소급 등재 때문에 append 순은 시간순과 다를 수 있다.
- 재확인: `rg -n '^\\| 20' claudedocs/EXPERIMENT_LEDGER.md | tail -20`. 끝에 append하면 기존 줄 앵커는 움직이지 않는다.
- 원장 읽기는 필요한 행만 `sed -n '<줄>p' ...`; 전체 통독은 피한다. 세션 경로는 별도 표시가 없으면 `claudedocs/` 아래.

## 최근20건 — 신 → 구

- **:595 · 09-13~14 W13 재개 종료** — 단일 본 물리 정리 포함31,218.753855초·SIGNAL_STOP/runner124. 24.4868초·16,304sync·283PF, HOME 오차29.57491mm·hold0. 용기 확정 분류0/가능 상한11·정착 미확정. raw 사전규약2FAIL, 부분 재생3FAIL(문정지5개 PF282 오연결·결정PNG 불완전·Isaac 출처338/283). 영상283장/28.3초 root11표본 실제 검수; 독립14/16·12/15, root 같은 재생 검사rc1. 원본14/기존508·HEAD 보존.
  → **W13_SINGLE_RUN_TIMEOUT_PARTIAL__RAW_SCHEMA_FAIL2__PARTIAL_REPLAY_FAIL3__NO_FULL_CYCLE_PROMOTION**. 감사 정상 failed 인계 후 release, 생산은 권한누락2회거부 후 종료턴 확인·공식abandon/사용자 터미널 보존. 추가실행/학습/A-B-C/실물/commit/push0.
  근거 `session_20260913_w13_resume.md`, `runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/coordinator/REPORT_w13_resume_received.md` → 두 worktree 원자료/재생/감사, root 재현·실제시각·보존·Orca 인계 영수증.

- **:594 · 09-12 W13 부분 인계** — Claude Opus5 구현/실행, Codex gpt-5.6-sol high 독립감사. 단일dt1μs·20,000알·seed460 통 벽 시험128.352초/rc0, W11같은목표시각 위치max.041463mm/p99.000382mm·중심이탈0·입출력해시44/44·28/28 PASS. 최종속도.0539925m/s로완전정착아님. 메인원자료재검증·최종확대그림실제검수. 초기비정본dt시도제외·일부원로그공백명시.
  → **W13_WALL_SMOKE_PASS__FULL_CYCLE_NOT_RUN__LONG_RUN_APPROVAL_AND_INTEGRATION_PENDING**. 전체운반/배출·Isaac영상미완료, 약8시간은거친외삽·사용자선택대기. 두워커부분인계후터미널release·worktree보존.
  근거 `session_20260912_w13_full_cycle.md`, `runtime_logs/grasp_track/w13_full_cycle_d484/coordinator/REPORT_w13_partial.md` → 두worktree절대경로.

- **:593 · 09-12 W12** — 별도 Orca의 claude-opus-5 재생 / gpt-5.6-sol high 감사. W10→W11 Isaac각64프레임, 립최대1.873/1.867mm·원자료시간64쌍 최대차3.077ms. 541/517개·10.9592/10.4731g·재닫기정지원인차 재확인. 주석/RRD전체배열/실제PNG/보존 검수PASS, 워커터미널종료·산출보존. 재닫기연속입자영상 없음, 색은최종ID. 미사용보조값은정정JSON이정본.
  → **W12_ISAAC_REPLAY_VERIFIED__SOURCE_TIME_PAIRED__PHYSICS_VERDICTS_UNCHANGED**. 새물리·학습·실물0, 표시재현은실물서보가능성/dt수렴 증거아님.
  근거 `session_20260912_w12_isaac_replay.md`, `runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/coordinator/REPORT_w12_received.md` → 두worktree절대경로.

- **:592 · 09-12 W11** — 기존W10 2μs와신규1μs한셀. **541개/10.9592g→517개/10.4731g**, 저장최대속도5.3291→2.7255m/s·5m/s초과1→0. 재닫기는servo_stall→pinch_guard(3.0089N,기존3N),립물림진단0→5. post취점반경80mm MAE0.933979mm·최대49.162921mm. rc0·2730.0589초·Rerun2771sync/입자64frame·PNG4실제검수.
  → **W11_DT1E6_COMPLETE__SAMPLED_SPEED_WARNING_REDUCED__RECLOSE_PINCH_GUARD__CONVERGENCE_UNPROVEN**. 각dt1회·실물정합미확정·하드웨어/추가case/학습0.
  근거 `session_20260912_w11_dt_sensitivity.md`, `runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/{REPORT_w11.md,comparison.json,inspection.json}`.

- **:591 · 09-11 실물종료·계량/속도분석** — 컵9.66g·보고24g(총/순미확정),잔류약2알. 전송속도필드동일,마지막83.54초의정렬/원복/정착시간분해. 당시W11제안은이후:592에서실행됨.
  → **HARDWARE_CLOSED / MASS_BASIS_PENDING**. `session_20260911_hardware_closeout_next_sim.md`, `scoop_tilt_cycle_01/closeout_01/BRIEFING.md`.

- **:590 · 09-11 새scoop1회·기울임·HOME** — 중간정지4건후마지막83.54초/4215행추가정지0,전체11029행. 무중단성공아님. 계량미입력은당시상태,후속:591참조.
  → **ONE_SCOOP_DISCHARGE_MOTION_AND_HOME_COMPLETED_WITH_RECOVERIES**. `session_20260911_full_scoop_outlet_repeat.md`, `scoop_tilt_cycle_01/REPORT.md`.

- **:589 · 09-11 출구기울임 실기** — 2935행·58.14초,출구경사7.76→22.00°. 어깨편차5.01458°>5°로정지. 열린자세는이실험종료당시상태로현재아님.
  → **OUTLET_TILT_OBSERVED__FINAL_TRACKING_GATE_STOP__DISCHARGE_MASS_PENDING**. `session_20260911_790_tilt_execution.md`, `outlet_tilt_01/REPORT.md`.

- **:588 · 09-11 잔류 롤−5° 시험** — 실제롤−4.66° 변화에도출구경사+0.257°뿐,사용자사진에잔류. 열린상태중단은당시기록,이후HOME정본:590.
  → **SMALL_ROLL_EXECUTED__RESIDUE_REMAINS__PAIRED_MASS_INCOMPLETE**. `session_20260911_790_tilt_execution.md`, `release_roll_01/`.

- **:587 · 09-11 790 대조** — 3905행·71.81초,수정900/790모두리프트최대추가개방0°. 이회차총22.28g−당시추정컵0.05g≈22.23g;추정tare를다른회차에이월금지.
  → **MATCHED_TARGET_790_LIFT_NO_REOPEN_OBSERVED__SEQUENTIAL_PAIR_ONLY**. `session_20260911_790_tilt_execution.md`, `torque790_01/operator_measurement_01.json`.

- **:586 · 09-11 배출 회전 검토** — 실제배출51행·S1형상,출구하향5°개념은경사7.5→12.5°,작은손목롤은7.78°. 새실기0.
  → **OUTLET_DIRECTED_TILT_GEOMETRICALLY_PLAUSIBLE__PHYSICAL_DISCHARGE_UNTESTED**. `session_20260911_release_tilt_review.md`.

- **:585 · 09-11 수정900 계량 후속** — 같은torque900_03,빈컵9.65g·총29.60g→컵배출19.95g. 고정jaw잔류10~15알은사용자추정,새실기0.
  → **DELIVERED_MASS_19_95G_RECORDED__FIXED_JAW_RESIDUE_USER_ESTIMATED_10_TO_15**. `session_20260911_real_boot_measurement.md`, `torque900_03/operator_measurement_01.json`.

- **:584 · 09-11 수정900 실기 · D484** — T12218/18명시문목표유지,3739행·71.79초,리프트3.779→3.691°·추가개방0°. 당시계량대기는:585로보완.
  → **EXPLICIT_DOOR_TARGET_HARDWARE_VERIFIED__FIXED900_LIFT_NO_REOPEN_OBSERVED**. `session_20260911_real_boot_measurement.md`, `torque900_03/door_target_verification.json`.

- **:583 · 09-11 실물P/첫900 · D483** — P8↔48각도차0.527344°교대2회. 첫900되열림0.791°와T122문목표완화동시관측. 수정후결과는:584이후.
  → **P_COMMAND_RESPONSE_OBSERVED__SCOOP_TARGET_RELAXATION_FOUND**. `session_20260911_real_boot_measurement.md`, `pid_hold_02/`, `torque900_02/`.

- **:582 · 09-11 W10 재개 · D482** — 렌즈dt2μs/E5e6,두닫힘servo_stall·541개/10.9592g·rc0/1890초. 저장5m/s초과1sync·최대5.329m/s. 구회귀287개,과학4/4·Rerun2774sync/64frame검수.
  → **W10_DT2E6_TORQUE_STOP_COMPLETE__TRANSIENT_POP_WARNING__SIM_REAL_UNCALIBRATED**. `session_20260911_w10_reboot_resume.md`, `w10_deme_close_fix/REPORT_w10.md`.

- **:544 · 09-04~07 79th · D481** — S1 v0조립실패보존→v1재출력/조립,실물scoop-place 5/5. 당시손목피치펌웨어±90°확인,높이오판교훈.
  → **S1_REAL_SCOOP_PLACE_CYCLE_5_OF_5_OK__V0_ASSEMBLY_FAILED_V1_REPRINTED**. `session_20260904_79th_s1_v0_print_sent_assembly.md`.

- **:543 · 09-04 78th · D480** — 고정반쪽+서보직결문 S1전환,벤더STEP/실물/Isaac파지및슬라이스검증.
  → **S1_DESIGN_OK__ISAAC_GRASP_OK__PRINT_SLICED_NOT_SENT** (당시). `session_20260904_78th_s1_servo_direct_step_isaac_print.md`.

- **:542 · 09-03 77th후반4 · D479** — 순정서보조인트종속 구파지 재현,펌웨어에서맨T106은개방명령이지리셋아님확인.
  → **ISAACLAB_SPHERE_GRASP_SERVO_COUPLED_OK__HARDWARE_MD_T106_RESET_CLAIM_CORRECTED**. `g18_nut_trap/isaaclab_grasp_sphere_servo/`, `docs/reference/hardware.md`.

- **:541 · 09-03 77th후반3 · D478** — 구파지3차성공,1차토크상한/2차바닥박힘실패보존. 비물리8N·m·단일시행·펠릿0.
  → **ISAACLAB_SPHERE_GRASP_OK_RUN3__DEMO_TORQUE_8NM_NONPHYSICAL**. `g18_nut_trap/isaaclab_grasp_sphere/`.

- **:540 · 09-03 77th후반2 · D477** — 실메시/RTX근접및병렬스텝,셸mimic정정. writer폭주/close정지이력보존.
  → **ISAACLAB_PARALLEL_512_OK__SHELL_R_MIMIC_FLAG_INVERTED_FIXED_TO_FALSE**. `g18_nut_trap/{viz,isaaclab_smoke}/`.

- **:539 · 09-03 77th후반 · D476** — g18혼합방향너트체결로Phase2체결차단해소,p37/p38검증,URDF/USD갱신.
  → **PHASE2_FASTENING_RESOLVED_G18__3PT_MIXED_DIRECTION_NUT_TRAPS**. `session_20260903_77th_drive_extraction_fastener_probe_p38.md` §7 이후, `g18_nut_trap/`.

## 과거 원장 무결성 주의 / 갱신

- 원장 :529~531은4열드리프트 이력, :532~533은2026-08-26소급등재(57th/70th). `## Schema errata`에서누락열/보조판정토큰보완;원행무수정. D453~D455원문이판정정본.
- 56th는순수부트검증으로미등재사유가있어결함아님. 57th/70th누락은소급등재로해소. 물리0이어도중요산출/지속결정이있으면등재,미등재시세션에사유기록.
- 원장 :92~103의죽은 `Current Next Experiment Candidate` 및헤더없는옛표블록은현재상태로쓰지않는다. 원문은append-only라보존.
- 종료세션한개가6열(Date/Run/Goal/Result/Verdict/Source)로**파일끝에append**하고이색인을20건이하/200줄이하로갱신한다. 과거색인의“errata앞삽입·기존앵커이동”안내는append-only와충돌해따르지않는다.
