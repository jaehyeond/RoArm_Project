# TASK_SPEC W25-C — 평평한 층 더미(NTC106) 사양 + 전체 사이클 RunPod 실행·비용 계획 (문서·산술만, 실행 0)

작성 2026-09-28 · 코디네이터 = 메인 Claude Code(Fable 5.1) · 이 과제서가 계약 정본. 보고는 **한국어**, 절차→수치→근거(file:line·sha256)→한계→다음 승인 경계 순, step-by-step.
⚠️ Codex 시작 시 "업데이트 가능" 안내가 뜨면 **건너뛰기**(설치·업데이트 금지, npm 전역 업데이트 금지).

## 0. 절대 금지
- GPU 0 · DEME 물리 0(더미 생성 실행 금지 — 명령 초안만) · Isaac 0 · RunPod API/pod 생성 0 · 로봇 0 · 설치 0 · git commit/push/merge 0 · 원장(START_HERE/DECISIONS*/EXPERIMENT_LEDGER/LEDGER_RECENT/BACKLOG/relay/session_*) 쓰기 0 · 다른 worktree·메인 repo 쓰기 0.
- 산출은 자기 worktree 의 `claudedocs/research/w25_pile_runpod_plan_20260928/` 아래에만.
- 모든 숫자는 파일 근거(file:line 또는 JSON 키)를 달고, 추정·외삽은 "추정" 표기. 기대 답을 맞추려 하지 말 것 — 메인이 재계산한다.

## 1. 배경(사실, 근거 경로)
- 실물 새 상자 후보 NTC106: **유효 안쪽 301×198×105 mm**(`START_HERE.md`, W24 N). 실물 더미 = 평평한 층. 실물 층 깊이·총 질량·알 1개 질량 **미실측**.
- 실물 절차 plunge 25 mm(`hw_s1_manual.py:157`) → 층 깊이 ≥ 30 mm 필요(W23 F REPORT §2-2).
- 더미 생성기: `/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/sim_deme_pile.py`(`--target-shape slab --target-depth-m … --box-width-mm … --box-length-mm … --shape lens --n-particles …`, `:1864-1890`). 기존 20,000알 더미 NPZ: `pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz`(경로는 W23 F `receipt.json`/`REPORT.md` §6), 알 질량 20.257 mg(템플릿, 밀도 905 가정), 정착 후 envelope 부피밀도 0.456 g/cm³·설정 0.55. 50,000알 생성은 1,200 s 제한에서 물리 0.40 s 미정착 rc 124(pellet-model `pile_lens_20260910/driver.log`).
- W23 F 예산표: `/home/cgxr/orca/workspaces/RoArm_Project/w23-sim-alignment/claudedocs/research/sim_alignment_20260928/{GPU_RUN_PLAN.md, out/pile_budget.json}`(310×220 상자 기준 — NTC106 발자국으로 다시 계산해야 함).
- 전체 사이클 실측 비용: W19 A(20,000알, pod 4090, dt 1 µs) 러너 20,977.8 s, 물리 24.807 s, 단계별 실제 경과(settle 79 · approach 4,957 · descend 1,153 · close 1,112 · lift 428 · reclose 12.5 · transport 4,760 · discharge 1,161 · wait 1,189 · close_after 1,420 · return_home 4,658 s) — `claudedocs/session_20260917_w19_runpod_afternoon.md` §9 표, 원자료 `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/run_01/{SUMMARY_A.json,timeline_seed460.json,RUN_STATUS.json}`. 로컬 W16 run_02 11,021.63 s(settle→reclose 1회). 과금 pod A2 4.08 $(0.74 $/h, RTX 4090 SECURE).
- 비용 모형(D493, `claudedocs/DECISIONS.md:30260`): 알 ¼ 로 줄이면 접촉쌍 0.23×인데 시간 0.43× → 고정 부담(floor). W16(D487): 비용은 더미 내부 잠재 접촉쌍(~20만)이 지배. 더미 생성 20k→50k 시간 지수 1.154(W23 F 비관 모형).
- pod 규약(D490·W19 §5~§9): RTX 4090 SECURE, **`minCudaVersion 13.0`**(드라이버 570 호스트에서 DEME 첫 접촉탐색 커널 사망), image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, disk 40 + persistent 30 GB, 부트스트랩 v5(conda-forge python 3.11 + deme 휠 + gymnasium 1.2.3 핀), 로컬과 같은 절대경로 미러, 입력 577 파일 sha 대조, 스모크 3종(import300·sphere_regression·settle_full) rc0 뒤 GO, `setsid nohup` + 바깥 watchdog, 회수 시 `_obj` 포함(D491), 실행 상한 32,400 s 유지(D492).
- 실행 시각 제약: 랩미팅 2026-09-29 오전. 지금(09-28 밤) 기동해도 12 h 넘는 실행은 회의 전 완주 불가 — 사실대로 적는다.

## 2. 목표(Change)
1. **더미 사양표**: NTC106 안쪽 301×198 발자국 × 층 깊이 {30, 35, 40} mm × 부피밀도 {0.456, 0.55} → 알 수·질량(g)·생성 명령 초안(`sim_deme_pile.py` 인자 그대로, 실행 금지)·생성 시간 추정(로컬 4090 Laptop, 50k rc124 전례를 근거로 cap 제안). 기존 20,000알을 그대로 쓰는 "축소 발자국" 대안도 1행(발자국 크기·왜 실물과 다른지).
2. **전체 사이클 비용표**: 각 알 수에 대해 세 모형(낙관 = D493 고정 부담, 중간 = 알 수 비례, 비관 = 지수 1.154)으로 pod 실제 경과 시간(h)·비용($)·watchdog cap(비관×2)·완주 예상 시각(KST, 기동 시각을 변수로). 20,000알 기준선은 W19 A 실측으로 고정.
3. **RunPod 실행 계획서**: pod 사양·부트스트랩(v5 재사용 여부·핀 목록)·미러 경로·입력 해시 게이트·스모크 3종·GO 조건·감시(10분 폴링, pgrep 자기매치 함정 `[`패턴)·중단 규칙(사전 고정: 발산·cap 초과·비용 상한)·회수(`_obj` 포함·sha 대조)·종료·비용 상한 제안. 실행 명령은 **템플릿**(실행 금지). 사용자가 결정할 항목을 표로(알 수·층 깊이·절차 스위치·비용 상한·기동 시각).
4. **산술 재현 스크립트**: `budget_w25.py`(순수 파이썬, 입력 = 위 근거 JSON 값, 출력 = `pile_budget_w25.json`) — 메인이 다시 돌려 표와 일치하는지 확인한다.

## 3. 산출(Observable acceptance)
- `PLAN.md`(한국어, 표 중심) + `pile_budget_w25.json` + `budget_w25.py` + `pile_generation_commands.md`(초안) + `RUNPOD_COMMANDS_template.md`(초안) + `receipt.json`(읽은 파일 sha256·줄).
- `REPORT.md`: 절차 → 수치 → 근거 → 한계(미실측 목록) → 다음 승인 경계(사용자 결정 항목).

## 4. 환경
- CPU 파이썬 `/home/cgxr/miniconda3/envs/roarm/bin/python`. 네트워크 조회는 하지 않는다(RunPod 단가는 W19 과금 실측값 0.74 $/h 를 근거로, "확인 필요" 표기).
- Orca: 질문은 preamble 의 `ask`. 완료 시 `worker_done` + `--report-path`.
