# START_HERE.md

Last updated: 2026-09-17 22:3x KST — **W19 RunPod 본 실행 A·B 완료·회수·pod 종료, 원장 D490·:601~602 append**. 최신 세션 문서 [session_20260917_w19_runpod_afternoon.md](claudedocs/session_20260917_w19_runpod_afternoon.md) §1~§9. 새 물리 = pod 에서만(로컬 GPU 0), 렌더·학습·실물·commit/push 0. 다음 세션 진입점 [CONTINUE_20260918_W19_POSTPROCESS.md](claudedocs/CONTINUE_20260918_W19_POSTPROCESS.md).

> 용어: "벽시계" 대신 **실제 경과 시간(wall-clock time)** 을 쓴다(시계로 잰 실제 흐른 시간 ≠ 시뮬 물리 시간).

## Active Case — single source of truth

- **완료(2026-09-17 오후~밤, 사용자 GO 16:2x) — W19 RunPod 본 실행.** 신규 변수: [실행 환경 = RunPod RTX 4090 Secure pod(드라이버 580 호스트)]. 물성·형상·알 수·경로·문 속도·보호선·dt 1 µs·cd_update_freq·seed 460 변경 0, 코드 변경 0. 출력 `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/`. **pod 3개 전부 Terminate**(우리 것만; 랩원 pod 3개는 조회만 했고 지금 전부 EXITED). 오늘 과금 5.39 달러.
  · **A = rev32 전체 사이클 첫 완주**(pod `qfo6qnce4ti50d`, 16:25:21→22:14:59 KST, rc 0, 실제 경과 20,977.8 s = 5.83 h, 물리 24.807 s 12단계, 16,813 sync·288 프레임). **생산 회계(rev32 규약 v2 프레임 분류, 독립 감사 전)**: 최종 source 19,266 / receiving_bin **272**(=5.51 g) / tool 11 / spill 124 / ambiguous 327, 가능 상한 327(6.62 g), 정착 창 5프레임 안정 272/272(최대 속도 2.39e-5 m/s·이동 0.0033 mm)이나 **cadence_ok false**(규약 ≥6프레임·≤0.05 s, 실제 간격 0.100025 s — W13 과 같은 기록 계약 한계), HOME 립 오차 **0.00006 mm**, 문 정지 close servo_stall 3.213°·reclose servo_stall 3.054°, 배출 개방/닫힘 정상, 최대 2.4597 m/s·5 m/s 초과 0. 재닫기 종료 공구 내부 292(W13 144), 운반 종료 tool 153·bin 18·spill 123. 회수 `A_full_cycle/run_01/`(377 MB, `_obj` 제외) sha 10/10 일치(`RETRIEVAL_RECEIPT.json`, NPZ `414633fb…`), 요약 `run_01/SUMMARY_A.json`.
  · **B = 같은 입력 반복 2셀**(pod `h0s018rxxn5phv`; W11 dt1e-6 params 사본, 변경 = 출력 경로만; seed 는 태그·DEME 미시드): 포획 **517/10.4731 g**(W11 로컬과 동일)·**489/9.9059 g**, reclose servo_stall 3.309°/3.399°(W11 pinch_guard 3.076°), 최대 1.103/0.906 m/s, 각 ≈1,770 s. 회수 `B_scoop_repeat/cell_dt1e6_seed46{1,2}/` sha 26/26. 구 회귀 smoke 3회 287/318/309알.
  · **환경·함정**: 부트스트랩 v5(miniconda 3.11 같은 절대경로·deme 휠 sha·핀+gymnasium)·꾸러미 577 sha·smoke 3종. **드라이버 570 호스트(첫 pod `zwmwq17s80vdfb`)에서 DEME 첫 접촉탐색 커널 illegal memory access → pod 는 `minCudaVersion 13.0` 으로만**(원인 미확정, 증거 `podA_zwmwq17s80vdfb_evidence/`). 속도: pod/로컬 단계별 1.18~1.72×, 전체 20,954 s(로컬 추정 30,768).
  · **원장**: D490(`:30229`, 앞 30,227줄 md5 `aa54d73a…` 불변, 백업 `DECISIONS.md.bak_20260917_pre_d490`), `EXPERIMENT_LEDGER.md:601~602`, ACTIVE/RECENT 갱신.
- **완료(2026-09-17 오후) — 설계 브리핑 2건(문서만)**: worktree `w19-design-briefs`, `claudedocs/research/design_briefs_20260917/{DOOR_SEAM_CASE_DESIGN.md,PARTICLE_COUNT_COST_CASE_DESIGN.md,EVIDENCE_CHECK.json}`. ① 문 닫힘/이음새: 수단 A1 chatter 1회(추천)/A2 torque_fraction/B 립 STL, 대조군 C0(W13 기록, **또는 W19 A 완주 기록으로 교체 검토**)+C1 반복 셀 권장, 판정 SUPPORTED/NOT/CLOSURE_NOT_ACHIEVED. ② 알 개수: 수단 (i) ROI 절단/(ii) 소형 슬랩(n5000 존재), 셀 P0/P1/P2, 새 numeric_inputs 증거 필요(재구성기 거부 가능). **둘 다 사용자 결정 필요, 구현 0.** 워커 제안: ② 먼저, 두 case 같은 N.
- **완료(2026-09-17 새벽~아침) — 야간 W16/W17/W18**(D487~D489), **W14 rev29~rev31·W15**(D485~D486): 이전 START_HERE 사본 `w19_runpod_d487/coordinator/START_HERE.before_w19_20260917.md`.
- 상태/relay 종료 인계 후 다음 세션이 원장 소유. 이전 GO/COMMANDS 이월 없음. LFS 로컬 보류(staged D 24는 추적 제외·원본 존재)·push/이력 재작성 금지 유지.

## W13 → W19 상태 변화 (판정은 아직 아님)

- W13 run_01(09-13~14)은 24.487 s 에서 상한 SIGNAL_STOP·HOME 미완·용기 확정 0. **W19 A 는 같은 설정(rev32 = rev31 규약 v2 + 진단 로그)으로 24.807 s 완주·HOME 0.00006 mm·용기 272**. 단 (a) 생산 회계일 뿐 rev31 파생 회계(`derive_v2.py` 절차)·독립 감사·Rerun/Isaac 재생 미실시, (b) 정착 cadence 규약 미충족, (c) 단일 실행. → "전체 사이클 성공" 판정은 위 (a)~(b) 처리 뒤 사용자 승인으로.
- 표시 한계(어깨 범위 밖 IK 포즈·립 재투영)는 W13 감사 기준을 그대로 적용해 재생 시 다시 본다.

## Next concrete action / 새 승인 경계

1. **W19 A 후처리(별도 승인)**: rev31 규약 v2 파생 회계(W14 `derive_v2.py`·독립식 검사기, CPU) → 원자료 규약 검사(`RAW_SCHEMA_REQUIRED` + ERRATUM 1~4, 전환 11개·바닥식) → post04 재생(Rerun/Isaac, 로컬 GPU, 각 5,400 s 상한) → 독립 감사(Codex 9/18 한도 해제 후, 그 전 Claude opus-5). 결과에 따라 "전체 사이클 완주" 판정·D 승격.
2. **정착 cadence 규약 결정**(사용자): 규약 문구를 프레임 간격 0.1 s 에 맞출지, 종료 창 프레임 간격을 0.05 s 로 촘촘히 기록하도록 코드를 바꿀지(물리 불변, 별도 revision).
3. **설계 브리핑 ①②의 수단 선택**(사용자). W19 A 완주 기록을 ①의 대조군으로 쓸지 포함.
4. **RunPod 재사용 규칙**: `minCudaVersion 13.0`, 부트스트랩 v5, 꾸러미 매니페스트 sha, 러너 `--allow-go-step` 은 GO 뒤에만, 랩원 pod 조회만, 회수·sha 대조 후 Terminate. 다른 GPU(5090/PRO 6000/2×4090) 실측은 새 GO.
5. 실물/T105 조회·구동·PID/토크·카메라, 학습, PBD 하이브리드, A/B/C, 설치(로컬), commit/push, LFS push/이력 재작성은 새 명시 승인 전 금지.

## 먼저 읽을 근거

- AGENTS.md → DECISIONS_ACTIVE.md(D485~D490) → LEDGER_RECENT.md(`:596~602`) → relay/from_claude.md(9/17 밤) → **session_20260917_w19_runpod_afternoon.md §1~§9** → `w19_runpod_d487/A_full_cycle/run_01/SUMMARY_A.json` · `coordinator/{SPEED_RATIO_A2.json,BUDGET_20260917.md,DO_NOT_TOUCH_PODS.json}` → 설계 브리핑 2건 → (필요 시) session_20260916_w14_raw_repair_dt_plan.md §7~§8 · 야간 보고 3건.
- 관측은 JSON/NPZ 정본, Rerun Float32·Isaac 영상은 검사층. W13/W19 는 W11 과 domain/유한벽/전체경로가 다른 별도 case.

## 과거 완료 결과 — W11/W12 유지

- W11: dt 1 µs 1셀 포획 517/10.4731 g(W19 B-461 과 동일값, B-462 는 489), 재닫기 pinch_guard. W12: Isaac 재생 립 최대 1.873/1.867 mm. `run_w10b.sh` 재실행 금지.

## 실물 종료 정본 / 유지할 관찰

- 정본 `claudedocs/session_20260911_hardware_closeout_next_sim.md`; 마지막 기록 HOME [0,0,90,0,0,0], 문 목표 0·토크 200·포트 닫힘 — 과거 관측. 최신 계량: 컵 9.66 g·보고 24 g(컵 포함 여부 미확정)·고정 jaw 잔류 약 2알.

## 신뢰하지 않을 과거 상태 / 환경

- CONTINUE_20260911/0914/0916/0917 의 과거 GO/COMMANDS·HANDOFF.md/TASKS.md·중간 자세·옛 비용 승인 대기는 현재 상태 아님. RunPod pod 는 **하나도 살아 있지 않다**(우리 3개 종료).
- isaaclab numpy 1.26.0·psutil 5.9.8·Rerun 0.34.1·DEME 2.4.0 유지, 로컬 설치 0. LFS 로컬 보류(`LFS_DEFERRED_20260916.md`) 그대로.
