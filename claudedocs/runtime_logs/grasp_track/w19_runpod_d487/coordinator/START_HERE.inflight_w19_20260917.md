# START_HERE.md

Last updated: 2026-09-17 16:3x KST — **W19 RunPod 본 실행 A·B 진행 중(사용자 GO 16:2x)**. 오전까지: 야간 W16/W17/W18 완료·검증·교차감사(D487~D489, 원장 :598~600). 최신 세션 문서 [session_20260917_w19_runpod_afternoon.md](claudedocs/session_20260917_w19_runpod_afternoon.md). 새 물리 = pod 에서만(로컬 GPU 0), 렌더·학습·실물·commit/push 0. 원장 append 는 결과 회수 후.

> 용어: 이 문서부터 "벽시계" 대신 **실제 경과 시간(wall-clock time)** 을 쓴다(시계로 잰 실제 흐른 시간 ≠ 시뮬 물리 시간).

## Active Case — single source of truth

- **진행 중(사용자 GO 2026-09-17 16:2x KST) — W19 RunPod 본 실행.** 이번 case 의 신규 변수: [실행 환경 = RunPod RTX 4090 Secure pod(드라이버 580 호스트)]. 물성·형상·알 수·경로·문 속도·보호선·dt 1 µs·cd_update_freq·seed 460 변경 0, 코드(rev32·`sim_deme_scoop_s1.py`) 변경 0. 출력은 `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/` 아래에만.
  · **A = 전체 사이클 1회 rev32**(W13 params, 20,000알, `--numeric-evidence`, `--max-wall-s 42600`, 러너 상한 43,200 s). pod **A2 `qfo6qnce4ti50d`**(`roarm_w19_A2_cuda13host_rev32`, 4090, 0.74 $/h, SSH `root@47.47.180.42 -p 11232`). 기동 **16:25:21 KST**, sim pid 2496. 출력 pod `/workspace/w19_out/A_full_cycle/run_01/`(미러 경로 `.../w19_runpod_d487/A_full_cycle/run_01` 심링크). 예상 종료 ≈ 21:5x KST(settle 속도비 1.57× 외삽, 추정), 상한 04:25 KST.
  · **B = 스쿱 구간 반복 seed461→462 순차**(W11 dt1e-6 params 사본, 변경 키 `render_timeline_path` 만; **seed 는 태그일 뿐 같은 입력 반복 = GPU 비결정 변동 측정**, `sim_deme_scoop_s1.py:377-378`·`scoop_y_range_mm=[0,0]`). pod **B `h0s018rxxn5phv`**(`roarm_w19_B_scoop_repeat`, 4090 EUR-IS-1, 0.74 $/h, SSH `root@157.157.221.29 -p 31894`). 기동 **16:26:47 KST**(`run_B_seq.sh`, 셀당 상한 7,200 s, 예상 ≈ 29 min/셀). 출력 `/workspace/w19_out/B_scoop_repeat/cell_dt1e6_seed46{1,2}/`.
  · **환경 검증 완료(양쪽)**: 부트스트랩 v5(miniconda 3.11.16 같은 절대경로, deme 2.4.0 cp311 휠 sha `acf8a029…`, 정적 아카이브 `c47ea1c4…`·libnvrtc `2bb82d1a…` 로컬 동일, 핀 + gymnasium 1.2.3), 꾸러미 577 파일 sha 0 불일치, smoke 3종 rc0(300알 settle / 구 회귀 W10 G0 3/3 PASS: A2 318알·B 309알 / 20,000알 settle 도메인 게이트 통과). 속도: settle 80.99 s(A2)·78.91 s(B) vs 로컬 W16 127.25 s.
  · **드라이버 570 호스트 함정**: 첫 pod `zwmwq17s80vdfb`(EU-RO-1, Open Kernel Module 570.211.01)에서 DEME 가 `DEMCubContactDetection.cu:339` illegal memory access 로 확정적으로 사망(두 파이프라인, CUDA_LAUNCH_BLOCKING 동일). 증거 `w19_runpod_d487/podA_zwmwq17s80vdfb_evidence/` 회수 후 16:11 Terminate. **pod 는 `minCudaVersion 13.0`(드라이버 ≥ 580) 으로만 만든다**(원인 미확정, 관측 기반 회피).
  · **랩 계정 금지 목록** `coordinator/DO_NOT_TOUCH_PODS.json`: `bmx3th83jy4rs1`(safe_coffee_porcupine, RTX PRO 6000, 2.09 $/h, 15:24 생성·소유자 미확인) · `hh5gae7qzsugro`(Geena L40S) · `xbp9doj5qmqtpd`(Geena 4090 EXITED). 이 세션은 조회만. 우리 pod 는 `POD_A2_CREATE.json`·`POD_B_CREATE.json` 의 id 만.
  · **결과 회수 절차(다음 행동)**: 종료 감지(Monitor 2개, 10분 폴링) → pod 에서 `RUN_STATUS.json`/`EXECUTION_RECEIPT.json` 확인(타임아웃≠성공) → `rsync`(`_obj` 제외 가능)로 `w19_runpod_d487/{A_full_cycle/run_01,B_scoop_repeat/cell_dt1e6_seed46x}/` 회수 → 로컬 sha256 대조(`manifest`/receipt) → **pod Terminate**(A2·B, 회수·해시 확인 뒤에만) → 로컬 회계(rev31 규약 v2 파생, W14 `derive_v2.py` 절차)·Rerun/Isaac 재생·독립 감사(Codex 는 금요일 한도 해제 후, 그 전엔 Claude opus-5)는 **다음 세션/별도 승인**.
- **완료(2026-09-17 오후) — 설계 브리핑 2건(문서만, 구현 0)**: Orca run `run_8bc9dc9d428f`, worker Claude claude-opus-5 high(`ctx_537893160d69`, released), worktree `w19-design-briefs`, 산출 `claudedocs/research/design_briefs_20260917/{DOOR_SEAM_CASE_DESIGN.md,PARTICLE_COUNT_COST_CASE_DESIGN.md,EVIDENCE_CHECK.json(101건)}`. ① 문 닫힘/이음새: W13 재닫기는 3.5487° 에서 servo_stall(물림 가드 미발동), 수단 후보 A1 chatter 1회(추천)/A2 torque_fraction 0.9→1.0/B 립 STL 변경 — **사용자 결정 필요**. ② 알 개수: 수단 (i) ROI 절단(`--max-particles`, numeric_inputs 도메인 게이트 fail-closed)/(ii) 소형 슬랩 재생성(n5000 NPZ 존재, footprint 절반) — **사용자 결정 필요**. 새 증거 파일·셀 표·사전 판정 기준은 문서에.
- **완료(2026-09-17 새벽~아침) — 야간 3건 W16/W17/W18**(D487~D489): rev32 = 다음 물리 정본(사용 중). W16 비용 = 더미 상시 접촉쌍 ~20만, 손잡이 = 알 개수. W17 post04 재생 결함 3건 해소. W18 운반 손실 = 문 3.55°·이음새 7.01 mm 누출. worktree `w16-profiling`·`w17-replay-fix`·`w18-cohort-cause` 보존(merge 없음). 상세는 세션 문서 `session_20260916_w14_raw_repair_dt_plan.md` §8.
- **완료 — W14 rev29~rev31(규약 v2 ERRATUM_04, 파생 정본 rev31 + `derived_v2_rev31/`)·W15 dt ladder(세 셀 rc134/OOM/assertion, D486)**. 원문은 이전 START_HERE 사본 `w19_runpod_d487/coordinator/START_HERE.before_w19_20260917.md` 와 세션 문서 §1~§7.
- 상태/relay 종료 인계 후 다음 세션이 원장 소유. 이전 GO/COMMANDS 이월 없음. LFS 로컬 보류(staged D 24는 추적 제외·원본 존재)·push/이력 재작성 금지 유지.

## 보존된 W13 종료 상태 — 재실행 승인 아님

- W13 재개 실행(09-13~14)은 31,218.75 s 상한에서 SIGNAL_STOP, 물리 24.4868 s·16,304 sync·283 입자 프레임, 전체 HOME/정착 미완료, 용기 확정 분류 0. 원자료 규약 FAIL2 는 rev29~rev31 파생으로만 수정(소급 PASS 아님), 재생 FAIL3 는 post04 로 해소. 표시 한계(어깨 범위 밖 20프레임·립 재투영 8.132 mm) 유지. 물리 정본 `w13-full-cycle/.../implementation/run_01/`(NPZ `529f422e…`), 재생 정본 `w17-replay-fix/.../post04_20260917/`.
- **W19 A 가 완주하면** 이 절의 "전체 사이클 미완" 상태를 대체할 첫 후보가 된다 — 단 회계(rev31 v2)·정착 창·용기 확정 분류가 끝나야 판정한다.

## Next concrete action / 새 승인 경계

1. W19 A/B 종료 → 회수 → 해시 대조 → **Terminate**(위 절차). 회수 전 pod 종료 금지. 타임아웃(`halted_timeout_not_success`)이면 부분 원자료로 보존, 성공 아님.
2. 회수 뒤 회계·Rerun·Isaac·독립 감사는 별도 승인(Codex 금요일 이후).
3. 문 닫힘/이음새 case(설계 문서 ①)·알 개수 case(②)의 수단 선택 = **사용자 결정**. 결정 전 구현 0.
4. RunPod 재사용 규칙: `minCudaVersion 13.0`, 부트스트랩 v5, 꾸러미 매니페스트 해시 대조, 러너 `--allow-go-step` 은 GO 뒤에만, 랩원 pod 는 조회만.
5. 실물/T105 조회·구동·PID/토크·카메라, 학습, PBD 하이브리드, A/B/C, 설치(로컬), commit/push, LFS push/이력 재작성은 새 명시 승인 전 금지.

## 먼저 읽을 근거

- AGENTS.md → DECISIONS_ACTIVE.md(D485~D489) → LEDGER_RECENT.md(`:596~600`) → relay/from_claude.md(9/17) → **session_20260917_w19_runpod_afternoon.md §1~§7** → `w19_runpod_d487/coordinator/{BUDGET_20260917.md,SPEED_RATIO_A2.json,DO_NOT_TOUCH_PODS.json,POD_A2_CREATE.json,POD_B_CREATE.json}` → 설계 브리핑 2건 → (필요 시) session_20260916_w14_raw_repair_dt_plan.md §7~§8 · 야간 보고 3건 · 교차감사.
- 관측은 JSON/NPZ 정본, Rerun Float32·Isaac 영상은 검사층. W13/W19 는 W11 과 domain/유한벽/전체경로가 다른 별도 case.

## 과거 완료 결과 — W11/W12 유지

- W11: dt 1 µs 1셀 포획 517/10.4731 g, 저장 최대속도 2.7255 m/s, 재닫기 pinch_guard. 각 dt 1회라 수렴 미입증. `run_w10b.sh` 재실행 금지. W12: Isaac 재생 립 최대 1.873/1.867 mm. 상세 `session_20260912_w1{1,2}_*.md`.

## 실물 종료 정본 / 유지할 관찰

- 정본 `claudedocs/session_20260911_hardware_closeout_next_sim.md`; 마지막 기록 HOME [0,0,90,0,0,0], 문 목표 0·토크 200·포트 닫힘 — 과거 관측. 최신 계량: 컵 9.66 g·보고 24 g(컵 포함 여부 미확정)·고정 jaw 잔류 약 2알.

## 신뢰하지 않을 과거 상태 / 환경

- CONTINUE_20260911/0914/0916/0917 의 과거 GO/COMMANDS·HANDOFF.md/TASKS.md·중간 자세·옛 비용 승인 대기는 현재 상태 아님. 첫 pod `zwmwq17s80vdfb` 는 종료됨(존재하지 않음).
- isaaclab numpy 1.26.0·psutil 5.9.8·Rerun 0.34.1·DEME 2.4.0 유지, 로컬 설치 0. LFS 로컬 보류(`LFS_DEFERRED_20260916.md`) 그대로.
