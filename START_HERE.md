# START_HERE.md

Last updated: 2026-09-14 — 실행시간·최적화·학습 전략의 출력용 Markdown과 새 세션 재개 문서를 작성했다. 사용자 명시 승인으로 main/Orca commit·push 검수 진행 중. 최신 [session_20260914_research_closeout_git.md](claudedocs/session_20260914_research_closeout_git.md). W13 TIMEOUT/원자료 FAIL2/재생 FAIL3·D484 유지, 새 연구 실행 없음.

## Active Case — single source of truth

- **설명·재개 인계·Git 게시**: 이번 case의 신규 변수: []. 새 문서는 `research/closeout_20260914/20260915_W13결과_실행시간_최적화와학습전략_출력용.md`이며 Downloads 사본 전달. 새 세션은 [CONTINUE_20260914_W13_REPAIR_PERFORMANCE.md](claudedocs/CONTINUE_20260914_W13_REPAIR_PERFORMANCE.md)의 순서로 재개한다. 현재 연구 코드 수정/물리/렌더/학습 실행은 하지 않으며, 다음 수정 case도 아직 시작하지 않았다.
- **Git 권한 정정**: 최신 사용자 요청이 이번 자료 게시에 한해 commit/push를 명시 승인했다. 메인과6개 dirty worker branch를 따로 보존하며 임의 merge하지 않는다. 정확한 커밋·원격 확인·제외 파일은 [GIT_PUBLICATION.md](claudedocs/research/closeout_20260914/GIT_PUBLICATION.md). 과거 문서의 commit/push0은 당시 기록이지 이번 승인 취소가 아니다.
- **시간 개념 설명 문서 추가 전달 — 완료**: 이번 case의 신규 변수: []. `/home/cgxr/Downloads/20260915_Hz_dt_관측창_시간개념_설명_출력용.md`에 직전 설명의 8개 절·계산 예시·비교표·근거 링크를 보존했다. repo 사본은 `claudedocs/research/labmeeting_20260915/downloads_20260914/time_concepts/`이며 222줄/13,447bytes, 두 파일의 cmp/SHA256 일치. Markdown 전달이며 실제 인쇄/PDF 제작은 하지 않았다.
- **랩미팅 문서 전달 — 완료**: 이번 case의 신규 변수: []. Downloads의 `20260915_입자물리_DEME_Isaac_PBD_설명_출력용.md`는 직전 설명 본문 보존, `20260915_랩미팅PPT_연구실PC_제작인계.md`는 실제 초안6장→새본문8장 매핑·본문/노트/캡션·복사 경로·제작자 프롬프트를 포함한다. 원본 PPT SHA59d08134…b61c9 보존, PPT 제작·미디어 복사·연구실PC 접속은 하지 않았다.
- 최신 전달본/검증은 `claudedocs/research/labmeeting_20260915/downloads_20260914/`. 필수미디어6개18,226,968bytes·선택포함12개30,102,228bytes. 모든자산SHA·영상5개메타·로컬링크26개대조, 사진3개/W12PNG3개직접검수 후 배열사진2개채택. 실제NPZ 포획/질량 재계산 및 Downloads2개 바이트일치. 상세 검증기는 read-only `verify_delivery.mjs`.
- 계약 `claudedocs/research/labmeeting_20260915/CONTRACT.md`, 메인 출력 `claudedocs/research/labmeeting_20260915/coordinator/`. 각 worktree의 같은 research 경로 아래 ppt/physics/render/만 신규 작성. 실험 자료/코드/기존 worktree 삭제·이동·덮어쓰기 없음.
- 연구내용 정본 [REPORT_labmeeting_20260915.md](claudedocs/research/labmeeting_20260915/coordinator/REPORT_labmeeting_20260915.md), 독립 근거 `coordinator/{ROOT_NUMERIC_CROSSCHECK_01.json,ROOT_VISUAL_CHECK_01.json,ROOT_CROSS_REVIEW.md}`. 기존초안에 대응한 최신슬라이드배치는 Downloads 제작인계 문서다. 원래 render 자산12개/링크22개 검사와 이번 전달목록12개/링크26개 검사는 다른 범위이며, PNG 직접 관찰과 MP4 metadata 검수를 구분했다.
- Run `run_617e1f3f687f`: research-survey/ClaudeOpus5, pellet-model·w12-isaac-replay/Codexgpt-5.6-sol high. 초기3 Task + 발표 국소정정 Task `task_f215088027ff` 전부정상완료, 실제worker자원3개 release·active0·reclaimable0·메일함빈상태 확인 후 전용coordinator종료. `coordinator/ORCA_FINAL_ACCOUNTING.json`. 원ppt Dispatch retained 표시는 후속owner이전 이력이며 실제보유worker아님.
- 핵심 해석: actual7구 rigid clump20,000알·질량/MOI포함,9/10치수사진/적층관찰 존재·물성미보정. 입사방향효과는접촉속도분해에포함,단일각도측정없음. PBD미정착·관측조건민감으로전환했으며 readback/질량부재사유아님. W12는Isaac scene step이있는저장상태재생이며입자PhysX재계산/양방향결합아님.
- 상태/relay 종료 인계 후 다음 세션이 원장 소유. 새 실험·GPU 렌더·학습·A/B/C·실물조회/구동/PID/토크/카메라·설치 금지. commit/push는 위 최신 승인 범위에 한정한다. 이번 결과는 발표 근거 인수이며 기존 실험 실패나 수락 기준을 바꾸지 않는다.

## 보존된 W13 종료 상태 — 재실행 승인 아님

- **W13 재개 실행은 종료했고, 전체 성공 판정은 불가하다.** 이번 case의 신규 변수: [] — 기존 W13 경로 검사/전체 사이클 통합 재개이며 추가 dt·물성·전략 변수 없음.
- 순서: 실제 자세 연결 경로 검사 통합 → Isaac 준비 검수 → 정확 GO → 최대9시간 단일 본 물리 → 원자료 감사 → 단일 부분 Rerun/Isaac 재생 → root/독립 검수. 이 승인된 본 실행 기회는 이미 사용했으며 자동 재실행하지 않는다.
- 작업은 실제 별도 Orca worktree에서 진행했다. 생산 Claude claude-opus-5: /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle. 감사 Codex gpt-5.6-sol/high: /home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit. 메인만 상태/relay/원장 소유.
- 신규 출력은 각 worktree의 claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/ 및 audit/, 메인 같은 case의 coordinator/. 동결 revision·실패·기존 raw 이동/삭제/재사용 금지.
- 최신 물리 정본은 생산 implementation/run_01/{w13_cycle_seed460.json,w13_cycle_seed460.npz,EXECUTION_RECEIPT.json,RUN_STATUS.json}. 재생 정본은 implementation/partial_post_03/{execution,input/rerun,isaac}/. post01/02는 미실행 보존본이다.
- **물리 결과**: 09-13 21:25:01→09-14 06:05:20 KST, 정리 포함31,218.753855초≤32,400초. 예정 TERM15 뒤 SIGNAL_STOP, child0이나 timed_out=true·runner124. 물리24.486802938176766초·16,304sync·283입자 프레임. 원래 물리/후처리 PID 부재는 호스트에서 확인했다.
- **전체 HOME/정착 미완료**: HOME 립 목표 오차29.57491mm, home_hold0행. return_home_end는 중단 뒤 저장된 표식이므로 완주 증거가 아니다. 최종 확정 용기 분류0개/가능 상한11개(0.222831g), 마지막0.25초 창3프레임 최대간격0.100025초로 cadence 불충족. 정확한 정착 배출량은 미확정.
- **운반 보유 관측**: 재닫기 시 PF107에서 기록상 공구 내부144개를 ID 추적하면 PF136/t11.303116초에서0개다. 용기 도착 전 운반 중이며 최종 기록133source+7spill+4ambiguous, strict 식132+7+5. 이미지에서144개를 센 것이 아니고 원인/배출 성공 확정도 아니다.
- **원자료 규약 FAIL2개 유지**: phase-only 전환은11개 요구/실제25개, source 전체 구체 containment 식 불일치. 전체1,507,161 분류 불일치(입자 유실 수 아님), 최종 source기록19,712/strict14,350. 독립 REV28_PRODUCTION_PARTIAL_RAW_AUDIT_01.json 14/16; root는 마지막 프레임·ID8 반례·시간/속도/ID 추적을 별도 재현했다.
- **부분 재생 완료**: post03 06:43:04→07:09:46 KST. Rerun1,046.855042초/Isaac554.113737초·각rc0/timeout0·정리 포함. 원본14개 사후 보존 PASS. 영상283프레임·10fps·28.3초(원자료 시간24.4868초와 구분), root가 watch 추출11표본 실제 검사했다. 합성 프레임0·부분 실행 경고 유지.
- **재생 FAIL3개 유지**: DOOR_STOP5개 전부 마지막PF282에 오연결; Rerun 결정PNG는t0 더미만/빈 그래프; Isaac 관절 출처 요약338 대 실제283 중복집계. 독립 post03 12/15·root 같은 검사 재실행rc1/동일3실패. RRD/RBL 구조 PASS를 실제 결정 화면 PASS로 바꾸지 않는다.
- **표시 한계**: 어깨 범위 밖 표시20프레임·최대 립 재투영8.132mm. 동결 수치 합격선이 없어 사후 새 물리 FAIL 임계값으로 만들지 않았으며, 실물 구동 가능성은 미검증이다. 원자료 배열 해시는 렌더 픽셀/실제 USD geometry의 bit-exact 증명이 아니다.
- **경로 검사 범위**: 접촉 후 연결227개 구간에서 첫 물리호출 전 검사·실행 전 검사227/227·계획 대비 편차0. 계산 여유 최저65.661289mm/시작점 검사72.682252mm는 이 연결 구간의 기록이며 전체 운반/서보/배출 성공 증거가 아니다.
- **Orca 종료**: Run run_5cba1e55a775. 감사 ctx_e5023d7f09ed 정상 failed worker_done 후 release. 생산 ctx_006e285a14a4는 capability 누락으로 제출2회 거부, 실제 종료턴 확인 후 공식 worker-abandon → failed/abandoned. 생산은 user_owned/retained라 강제종료하지 않았다. reclaimable0, 감독 중인 새 작업0. 정본 coordinator/ORCA_FINAL_ACCOUNTING_01.json. 두 worktree/산출은 메인 Git push에 자동 포함되지 않는다.
- 실물 조회·구동·PID/토크·카메라 수집, 학습, A/B/C, 패키지 설치, commit/push는0. 기존 준비 단계 계약 초과/중간 메타 원문 보존 공백 등은 세션에 보존하며 완전 절차 준수로 과장하지 않는다.

## Next concrete action / 새 승인 경계

시간 개념 Markdown은 미리보기에서 수식·표·페이지 나눔을 확인한 뒤 출력한다. 기존 PPT 제작 인계는 사용자가 Downloads의 앞선 Markdown2개·원본PPT·제작인계 §4 필수 A01~A06 미디어를 연구실PC에 복사한 뒤 §8 요청문을 제작자에게 전달한다. 새 시간 개념 문서는 보충자료로 함께 전달할 수 있다. 선택미디어A07~A12는 필요할 때만. 실제 PPT 생성·외부링크 없는 영상재생 검수는 그 PC의 후속 작업이며 아직 수행하지 않았다. 새 물리/하이브리드 실행은 승인되지 않았다. 아래는 기존 W13 후속 승인 경계다.

1. 먼저 통합 보고서와 부분 Isaac 영상을 보고, **원자료 규약2개/재생3결함의 새 revision 수정·검수**를 별도 승인할지 결정한다. 기존 실패를 소급 수정하지 않는다.
2. 운반 중 보유 판정이 사라지는 원인 조사는 별도 case로 범위/변수를 정한 뒤 진행한다. 지금 경로·문 동작·허용 기준을 바꾸거나 새 장시간 물리/재렌더를 시작하지 않는다.
3. A잔류허용/B추가제거/C처음부터 통합배출은 BACKLOG의 별도 후속 case이며 미구현. 학습도 미승인. 이번 결과를 전체 사이클 성공/정착 효율/실물 정합/dt 수렴으로 승격하지 않는다.
4. 실물/T105 조회·구동·PID/토크·카메라 재개는 새 명시승인이 필요하다. 계량 총/순 기준 답 전 물성 피팅 금지.
5. 이번 게시 이후 연구 재개는 새 continuation §5 요청문을 사용한다. 원자료 판정 구현 수정과 CPU 테스트부터 별도 case로 진행하고, 이후 재생 결함·운반 원인·성능 측정을 분리한다. 새 물리 프로파일링은 구간·총시간/정리 예산·입력·출력을 먼저 승인받는다. 이번 Git 승인으로 후속 세션의 commit/push나 GO/COMMANDS 재실행이 자동 승인되지 않는다.

## 먼저 읽을 근거

- AGENTS.md → DECISIONS_ACTIVE.md → LEDGER_RECENT.md → relay/from_claude.md 및 최신 from_codex.md → session_20260914_research_closeout_git.md → CONTINUE_20260914_W13_REPAIR_PERFORMANCE.md. 앞선 전달은 session_20260914_time_concepts_md.md/session_20260914_labmeeting_downloads.md. 이전 조사 session_20260914_labmeeting_20260915.md/REPORT_labmeeting_20260915.md. W13 상세는 session_20260913_w13_resume.md/REPORT_w13_resume_received.md.
- 생산 최종 W13_FINAL_PRODUCER_REPORT.md, 독립 audit/REPORT.md 및 POST03_PARTIAL_REPLAY_RESULTS_AUDIT_02.json. 생산 보고가 놓친338/283 문제는 독립 보고와 메인 통합 보고를 따른다.
- root ROOT_PARTIAL_RAW_SPOT_REPRO_01.json / ROOT_RAW_COHORT_VISUAL_CUES_01.json / ROOT_POST03_AUDIT_REPRO_01.json / ROOT_RERUN_INSPECTION_01.json / ROOT_ISAAC_VIDEO_INSPECTION_01.json / PARTIAL_RAW_POSTCHECK_01.json.
- 관측은 JSON/NPZ 정본, Rerun Float32와 Isaac 영상은 검사층. coordinator/REFERENCE_RELEASE_CONTRACT_REV2.md는 다음 준비에 적용한 소유 해제 계약 정정이며 이전16_actual FAIL을 소급 변경하지 않는다.
- W13은 domain/유한벽/전체경로가 다른 별도 case다. W11 final NPZ는 전체 재시작 checkpoint가 아니므로 임의 이어붙임 금지.

## 과거 완료 결과 — W11/W12 유지

- W11은 기존 W10 dt2μs 대비 신규1μs만 실행했다. 포획541/10.9592g→517/10.4731g, 저장 최대속도5.3291→2.7255m/s·>5경고1→0. 재닫기는 servo_stall→pinch_guard(3.0089N), 립6.165mm·립물림 진단5개. 각dt1회라 수렴/동등성·실물 정합 미입증, 1μs를 기본값으로 자동 승격하지 않는다.
- W11 근거 session_20260912_w11_dt_sensitivity.md, runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/{REPORT_w11.md,comparison.json,cell_dt1e6_seed460/}. 명령 run_w10b.sh는 덮어쓰기 위험으로 재실행 금지.
- W12는 W10/W11 각64프레임을 동일 Isaac 장면에 재생했다. 립 최대1.873/1.867mm, 원자료시간 짝64개 최대차3.077ms. 두 run 모두 재닫기 단계의 연속 입자프레임이 없어 복원하지 않았다고 명시했다. 표시 재현 PASS는 실물 구동/dt 수렴 증거가 아니다.
- W12 근거 session_20260912_w12_isaac_replay.md, w12_isaac_w10_w11_compare_20260912/coordinator/REPORT_w12_received.md. 미사용 옛 보조값7.7822mm/° 무효, diagnostic_correction_01.json의2.057714892mm/°가 정정값이다.
- W9=W8의154알 Isaac 재생, W10=별도541알 실험. session_20260911_video_w9_w10_review.md 참고. W9/W10 숫자를 같은 실험의 전후 결과로 섞지 않는다.
- 장기 연구 방향은 고정 S1·배출 위치에서 높이/행 기준선→포획/사후맵 예측→취점 선택이다. 이번 실행 범위를 넘어 바로 구현/학습하지 않는다.

## 실물 종료 정본 / 유지할 관찰

- 정본 세션 `claudedocs/session_20260911_hardware_closeout_next_sim.md`; 근거 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/scoop_tilt_cycle_01/`.
- 마지막 기록 `execution_05/result.json`: completed/home_reached=true, HOME목표[0,0,90,0,0,0]. 문목표0·마지막토크명령200·포트닫힘. **이것은 과거 관측이며 이번 세션 조회 결과가 아니다.** 이전 열린 롤90/raise2 자세는 현재로 쓰지 않는다.
- 새scoop1회가 중간추종정지4건과 복구를 거쳐 끝났다. 여러 새scoop나 무중단 성공으로 세지 않는다. 전체 최종기록 `combined_02`; `combined_01`은 중간까지다.
- 최신 사용자 계량: 컵9.66g·보고24g·고정jaw잔류약2알. **24g의 컵포함/PP만 기준 미확정.** 컵포함이면PP14.34g, PP만이면24g. 과거 추정컵0.05g·직전0알 관찰 이월 금지. `closeout_01/operator_measurement_01.json`.
- 마지막 잔류정리→HOME83.538273초, 이동명령60개. 전송spd/acc는이전790과동일. 정렬19.334865+기울임/대기9.891083+방향원복26.204635+문닫기/HOME25.336758+초기2.770932초. `closeout_01/{BRIEFING.md,timing_analysis.json}`.
- 사용자 정정 유지: 전체 cycle 경계는 HOME→scoop→배출→HOME. 배출 효율 비교에는 같은 경계의 컵도착g/s·흘림·잔류가 필요하다. 이번 W11에는 배출이 없다.

## 신뢰하지 않을 과거 상태 / 환경

- CONTINUE_20260911_SIM_ONLY.md는 이전 실행 요청 프롬프트이며 그대로 재실행하지 않는다. HANDOFF.md/TASKS.md, 과거 재부팅 차단, 중간 열린/롤90/raise2 자세, 옛 비용 승인 대기는 현재 상태가 아니다.
- isaaclab numpy1.26.0·psutil5.9.8·Rerun0.34.1 유지, 설치0. 버전/공식 NVIDIA 근거는 최신 세션의 로컬 원문 링크를 먼저 읽는다.
- 이번 보고 세션은 DECISIONS.md/ACTIVE·EXPERIMENT_LEDGER·LEDGER_RECENT 미수정(D484 유지). 기존 W13 결과는 원장 :595/이전 세션 정본이며 새 실험 행이나 지속 규칙을 만들지 않았다. 상태 원장 소유권은 이 종료 인계 후 다음 세션이 넘겨받는다.
