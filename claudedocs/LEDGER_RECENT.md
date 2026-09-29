# LEDGER_RECENT.md — 최근 실험 20건 요약 (부팅 read)

Last updated: 2026-09-29 21시 — W25 RunPod 병행 본 실행 2회 반영(:607). 원본 `EXPERIMENT_LEDGER.md` 607줄. 기존 prefix는 보존하고 끝에만 append.

## 권위와 읽는 방법

- 이 파일은 색인이다. 수치/판정 인용 전 `EXPERIMENT_LEDGER.md:<줄>`에서 링크된 세션/raw까지 확인한다.
- 현재 상태/다음 승인 범위는 `START_HERE.md`. 아래 과거 중간 자세나 “pending”은 당시 기록이며 현재 상태가 아니다.
- 선정: 날짜 행의 **append 순 마지막20개** = `:584~603` 20개. 아래는 역순이다(`:544`·`:582`·`:583` 은 목록에서 빠짐, 원장에는 그대로). 소급 등재 때문에 append 순은 시간순과 다를 수 있다.
- 재확인: `rg -n '^\\| 20' claudedocs/EXPERIMENT_LEDGER.md | tail -20`. 끝에 append하면 기존 줄 앵커는 움직이지 않는다.
- 원장 읽기는 필요한 행만 `sed -n '<줄>p' ...`; 전체 통독은 피한다. 세션 경로는 별도 표시가 없으면 `claudedocs/` 아래.

## 최근20건 — 신 → 구

- **:607 · 09-29 W25 3일차 RunPod 병행 본 실행 2회(rev34 종이 상자·67,737알·실물 절차 ON)** — 동결본·감사 PASS·스모크 R 2.556 → 둘 다 GO. podB 36,372.8 s·40.21 $, podA 52,707.9 s·10.62 $, 둘 다 completed_rc0·회수 14/14·terminate. R_full 1.449. 기하 라벨 836 vs 832. 원시 배출 585~665 vs 251~297(문 재닫기 2.39° vs 3.04°). podB 회계 불일치 0·cadence FAIL.
  → **W25_RUNPOD_PAIR_COMPLETED__R_FULL_1.449__GEOMETRIC_LIFT_LABEL_CONSISTENT_836_832__DELIVERY_DIVERGES_BY_DOOR_RECLOSE__SETTLEMENT_CADENCE_FAIL_INTERVAL_ONLY__NO_PROMOTION**. `session_20260928_w25_realign_fullcycle_prep.md` §13, D500.

- **:606 · 09-28 밤 W25 그리퍼 표시 원인·실물 정렬 전체 사이클 준비(rev34)·worktree 보관** — 물리·GPU·RunPod·실물·commit 0. 그리퍼 = 최신 S1 v1, "덜 그려짐" = 충돌 셸 + 문 CAD 숨김(post04 `gripper` 필터). 시뮬 상자 = 22 cm 벽이 로봇 쪽. B: CAD 배치식 16,813 sync ≤7.2e-5 mm, post05 초안 18/18. C/D: 4 cm 층 61,408~74,067알, 전체 사이클 15.0~26.4 h·11~20 $, 여백 4.6 mm 패치, 격자 인증 분기. E: dt 근거·{2,5} µs 사전등록(1대 1.8~2.9 h·1.4~2.2 $). A: rev34 OFF=rev32 바이트 동일, ON 관절 위반 0·여유 54/26 mm, C4 0.112 mm 미해결. worktree 10개 2단계 완료. 슬라이드 11장 정정.
  → **W25_GRIPPER_UNDERDRAWN_CONFIRMED_DISPLAY_LAYER__BOX_ORIENTATION_SIM_22CM_WALL_FACES_ROBOT__REV34_CPU_PREP_PASS__FLAT_LAYER_FULL_CYCLE_15_26H_EXCEEDS_CAP__NO_GPU_NO_RUNPOD**. `session_20260928_w25_realign_fullcycle_prep.md`, D498.

- **:605 · 09-19~23 W21 비용↔알 개수(로컬 GPU 전용)** — RunPod 0. **P2 n=5000 2회 = CELL_UNRUNNABLE**: 정지점 n_sync 556·물리 2.2205549943936376 repr 동일, 같은 순간 접촉쌍 46,964 vs 45,902 상이 → 결정적 제어 흐름 조건. 68스레드 futex_wait·rc 137·원자료 0. **부분 비용**: 접촉쌍 0.231×/0.226× vs 시간 0.433×/0.430× → **고정 부담, 2.3배(4배 아님)**. 변동폭 −2.2 %/−0.7 %(첫 기준선). ROI-300 스모크 증거 REUSABLE·게이트 3×2 검증. 🔴 알 축소는 case ① 문 토크를 통해 측정 오염.
  → **W21_COST_VS_N__P2_CELL_UNRUNNABLE_DETERMINISTIC_STALL__COST_FLOOR_2.3X__N_CONFOUNDS_CASE1**. `session_20260919_w21_cost_vs_particle_count.md`, D493.

- **:604 · 09-19 W20 사용자 결정 5항목 실행** — 물리 0·GPU 0·DEME 0·pod 0·commit 0. 결정 1 권고 **철회**(동결 `criteria.json` `policy.no_threshold_change_after_outcomes` = hard_fail) → **W19 A 배출량 = 272~327알(5.5100~6.6242 g), 정착 미증명**. 실행 상한 32,400 s 유지. case ② n=5,000 CPU 관문 **REUSABLE**(n=20,000 선검증 3축 binary64 exact, 음성 대조 SystemExit, 메인 재실행 stdout 바이트 동일 sha `3b563f94…c42e`, 읽기전용 50 재해시 0). rev33b 초안(재귀 diff 5자리, `frame_dt_s` 0.05→0.044, `retroactive:false`, wall cap 32400, AST PASS, 57/57, 읽기전용 78 재해시 0) — GPU 스모크 전이라 정본 아님. 역할 agent 4개 채택 + 원장 배타 hook(8경우 실측). 회수 스크립트 forward-only 신판.
  → **W20_DECISION_EXECUTION__CONTRACT_AMENDMENT_WITHDRAWN__DELIVERY_INTERVAL_272_327__PARTICLE_COUNT_GATE_REUSABLE__REV33B_NOT_PROMOTED**. `session_20260919_w20_decisions_execution.md`, D492.

- **:603 · 09-18 W19 A 후처리 1~3단계** — 회계 재현 0 불일치(5,760,000 라벨)·독립 검사 0·규약 27항목 21 PASS/6 FAIL(정착 cadence 5프레임<6·0.100025s, raw 메타 선언 누락, 영수증 해시·cap 43,200>32,400, 필드·visual_mapping·매니페스트 부재)·코호트 292→bin 152/spill 73/amb 57/src 6/tool 4. 감사1(Codex) 9/9. 재생 RRD 682MB rc0·DOOR_STOP 69/98/108/189/221·PNG 6 고유·Isaac 288·검사기 8/8·GPU 2,220s. 감사2(opus-5) 9/9. rev33 후보 AST PASS·30/30. `_obj` 2/4 미회수.
  → **W19_A_FULL_CYCLE_RAW_VERIFIED__ACCOUNTING_REPRODUCED_0_MISMATCH__REPLAY_8_8__AUDITS_9_9_AND_9_9__DEFINITE_DELIVERY_PROMOTION_DEFERRED_SETTLEMENT_CADENCE_CONTRACT**. `session_20260917_w19_runpod_afternoon.md` §10~§14, D491.

- **:602 · 09-17 W19 B RunPod 스쿱 반복** — W11 dt1e-6 셀 같은 입력 2회(pod 4090): 포획 517/10.4731g(W11 동일)·489/9.9059g, reclose 사유 servo_stall/servo_stall(W11 pinch_guard), 최대 1.103/0.906m/s, 각 ≈1,770s. sha 26/26.
  → **W19_B_SAME_INPUT_REPEATS_2_OF_2_RC0__CAPTURE_517_517_489__RECLOSE_STOP_REASON_VARIES__RUN_TO_RUN_VARIATION_OBSERVED_N3_NO_STATS**. `session_20260917_w19_runpod_afternoon.md` §8, D490.

- **:601 · 09-17 W19 A RunPod 전체 사이클** — rev32·W13 조건, pod 4090(drv 580) rc0 20,977.8s(5.83h), **물리 24.807s 12단계 완주**. 생산 회계 bin **272**(5.51g)/tool 11/spill 124/ambiguous 327, HOME 오차 0.00006mm, 정착 창 5프레임 안정 272 이나 cadence_ok false(간격 0.1s>0.05s). 재닫기 내부 292(W13 144). sha 10/10. 과금 4.08$.
  → **W19_A_FULL_CYCLE_COMPLETED_RC0__PRODUCTION_DELIVERY_272_HOME_0MM__SETTLEMENT_CADENCE_CONTRACT_UNMET__INDEPENDENT_AUDIT_PENDING**. 판정 승격은 rev31 회계·재생·독립 감사 후. `session_20260917_w19_runpod_afternoon.md` §5~§9, D490.

- **:600 · 09-17 W18 cohort 원인** — 운반 후보 144+235=379, 144개 PF110~136 전부 이탈(post_lift_travel 136, mouth_seam 129). 문 3.549° 유지·이음새 7.01mm > 펠릿 4.50mm; 가속도 상관 −0.016. 메인 재계산 일치.
  → **W18_TRANSPORT_LOSS_SUPPORTED_BY_OPEN_MOUTH_SEAM_7MM__NOT_ACCELERATION__NEXT_PHYSICS_VARIABLE_DOOR_CLOSURE**. `session_20260916_w14_raw_repair_dt_plan.md` §8, D488.

- **:599 · 09-17 W17 post04** — 재생 결함 3개 원인(루프 변수 누출/18자 접두 비교/스크린샷 로딩 경주) 규명·수정, 재렌더 rc0(2,677s/477s), 검사기 post03 5/8→post04 8/8, Codex 교차 12/13(B4=같은 시각 PNG 중복, 결함 아님). W13 판정 불변.
  → **W17_POST04_THREE_REPLAY_DEFECTS_RESOLVED__ROOT_CAUSES_FOUND__W13_SCIENTIFIC_VERDICT_UNCHANGED**. D489.

- **:598 · 09-17 W16 프로파일링** — rev32 진단 로그(물리 불변) + settle→reclose 1회: 9.0085s/11,014.7s, W13 대비 0.963. 잠재 접촉쌍 194k~237k(도구 접촉 0에서도 ~200k) → 비용은 더미 상시 부하, 손잡이 = 알 개수(dt는 D486 차단). Codex A1~A6 PASS.
  → **W16_COST_IS_PILE_INTERNAL_CONTACT_LOAD__PATH_OPTIMIZATION_CANNOT_REDUCE__NEXT_LEVER_PARTICLE_COUNT**. D487.

- **:597 · 09-16 W15 dt ladder** — timestep 10µs/100µs/1ms 단일 변수(코드 무수정·PREREG 선작성·순차·무재시도). 10µs: 2,790sync·3.1533s까지·저장최대19.22m/s·pinch_guard 2회·엔진 2.17e9m/s rc134·2,530s(≈802s/물리초). 100µs: 40s GPU OOM, 1ms: 10s 접촉탐색 커널 assertion — 둘 다 sync 0. 실측 sync 지속시간=float32 누산 예측 일치. 포획량 비교 불성립.
  → **W15_DT_LADDER_ALL_THREE_CELLS_ENGINE_ABORT__10US_FAILURE_REPRODUCED_DIFFERENT_LOCUS__100US_1MS_NOT_RUNNABLE_UNDER_CURRENT_CD_SETTINGS__NO_CONVERGENCE_CLAIM**. B′(cd_update_freq 스케일)·로그 추가·반복 셀은 별도 승인.
  근거 `session_20260916_w14_raw_repair_dt_plan.md` §7, `w15-dt-ladder/.../w15_dt_ladder_20260916/REPORT_w15.md`, D486.

- **:596 · 09-16 W14 raw repair** — W13 run_01 규약 FAIL2를 rev29 파생으로 수정. rev28 결함 재현(전환 대역재생 25=기록·분류 parity·PF0/ID8) → rev29 전환11개·바닥식 최하단. 파생NPZ `cdf11a36…`: vs 기록 1,507,161불일치(=감사값), 최종14350/0/0/73/0/5577, cohort144→132/7/5. CPU 18/18·독립식 0불일치·Codex gpt-5.6-sol high 8/8. 동결 해시 불변. dt 실행안: 10µs rc134 재검토·0.1ms 요청이 100µs→0.2ms/1ms→1.0ms, 셀 10µs/100µs/1ms ≤15,600s 제안(미실행).
  → **W14_RAW_JUDGMENT_REPAIR_REV29__REV28_FAIL2_REPRODUCED__CPU_18_18_PASS__INDEPENDENT_CODEX_8_8_PASS__RAW_VERDICT_UNCHANGED__DT_PLAN_DOC_ONLY_NO_GPU**. 규약 개정(바닥 containment·전환 프레임 i−1)·재생3결함·cohort 원인·W15 실행은 별도 승인. 새 물리/실물/commit/push 0.
  근거 `session_20260916_w14_raw_repair_dt_plan.md`, `runtime_logs/grasp_track/w14_w13_raw_repair_d484/repair_20260916_01/REPORT.md`, `research/dt_expansion_plan_20260916/DT_EXPANSION_PLAN.md`, D485.

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

## 과거 원장 무결성 주의 / 갱신

- 원장 :529~531은4열드리프트 이력, :532~533은2026-08-26소급등재(57th/70th). `## Schema errata`에서누락열/보조판정토큰보완;원행무수정. D453~D455원문이판정정본.
- 56th는순수부트검증으로미등재사유가있어결함아님. 57th/70th누락은소급등재로해소. 물리0이어도중요산출/지속결정이있으면등재,미등재시세션에사유기록.
- 원장 :92~103의죽은 `Current Next Experiment Candidate` 및헤더없는옛표블록은현재상태로쓰지않는다. 원문은append-only라보존.
- 종료세션한개가6열(Date/Run/Goal/Result/Verdict/Source)로**파일끝에append**하고이색인을20건이하/200줄이하로갱신한다. 과거색인의“errata앞삽입·기존앵커이동”안내는append-only와충돌해따르지않는다.
