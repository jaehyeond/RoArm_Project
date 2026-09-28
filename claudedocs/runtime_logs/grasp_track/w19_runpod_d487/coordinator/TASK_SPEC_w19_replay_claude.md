# TASK_SPEC — W19 A 후처리 2단계: post04 재생(Rerun RRD/RBL·결정 화면·Isaac 관절 재생, 로컬 GPU) — Claude claude-opus-5 high
작성 2026-09-18, 코디네이터 = 메인. 사용자 승인 "다음 작업 각각 worktree 배정". 계약 정본 = 이 파일. 물리(DEME) 실행 0 — 저장된 W19 원자료를 **다시 그리는** 작업이다.
## Target
- 입력(읽기 전용): W19 A 원자료 `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/run_01/{w13_cycle_seed460.npz(sha 414633fbff8fc022…),w13_cycle_seed460.json,timeline_seed460.json}` · 1단계 산출 `/home/cgxr/orca/workspaces/RoArm_Project/w19-postprocess/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/postprocess_20260918/`(파생 라벨·규약 검사) · post04 재생 정본 `/home/cgxr/orca/workspaces/RoArm_Project/w17-replay-fix/claudedocs/runtime_logs/grasp_track/w17_replay_fix_d484/post04_20260917/{rev/(src·COMMANDS·REVISION_PIN·criteria),tools/,tests/,REPORT_post04.md}` · USD `/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd` · 규칙 AGENTS.md(D324 시각 정의, D341 Rerun 완료 계약, D489 헤드리스 캡처 규칙).
- 출력(이 worktree 안에만): `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/replay_20260918/{rev/(post04 src 바이트 사본+REVISION_PIN.json+필요 시 DIFF_paths_only.patch+AST 범위 검사), COMMANDS.json, tests/, execution/(러너 영수증·stdout/stderr), rerun/(RRD·RBL·verify·contract·screenshots), isaac/(mp4·frames 표본·joint source counts), evidence/(visual_mapping 등), inspection.json, REPORT_replay.md, manifest.json}`.
## Change (순서)
1. post04 `rev/`·`tools/`·`tests/` 바이트 복사 + REVISION_PIN(원본 pin 과 sha 대조). 입력/출력 경로는 COMMANDS/CLI 로만 바꾼다. 코드를 고치면 DIFF + AST 범위 검사로 "경로 외 변경 0" 증명.
2. **CPU 먼저**: post04 단위 테스트(W17 21/21 패턴)를 W19 원자료에 적용해 통과시킨다. 사전 해시 대조(입력 전부) 영수증.
3. **step2 Rerun(로컬 GPU/CPU)**: `w13_rerun_export.py` 로 RRD(전 입자 프레임 288·DOOR_STOP 행 결속·결정 이벤트) + 고정 청사진 내장 + `.rbl` 내보내기 → footer 포함 `rrd verify` PASS → 엔티티·타임라인·구성요소 정확 계약 검사 → `visual_mapping`(저장 입자 프레임당 1행 = 288) → 결정 시각 PNG(브리지 1 + 문 정지 5; `w13_rerun_screenshot.py` viewer-mcp 인제스트 완료 관측 후 캡처; 같은 시각이면 중복 정상으로 표기). 상한 5,400 s, setsid nohup, 러너 영수증.
4. **step3 Isaac(로컬 GPU)**: `isaac_replay_w13.py` 로 관절 재생(출처 fk/abs/w11 합 = 288, 표시 한계: 선언 어깨 범위 밖 포즈 수·립 재투영 최대 오차 mm 를 W13 감사와 같은 정의로 산출) + MP4 + 프레임 표본 PNG. 상한 5,400 s, setsid nohup. isaaclab env 핀(numpy 1.26.0·psutil 5.9.8) 무변경·설치 0. 실행 전 `nvidia-smi` 기록(현재 다른 프로세스가 약 3.3 GB 사용 중).
5. post04 검사기(8항목)를 W19 산출에 적용 → `tests/RESULTS_checker.json`.
6. PNG·MP4 프레임을 **실제로 열어** 본 관찰을 `inspection.json` 에 기록(생성됨≠검수, D341).
7. `REPORT_replay.md`(한국어, "실제 경과 시간(wall-clock)", 영어 용어 풀이): 무엇/왜 → 절차 → 수치+경로 → 일상어 판정·비주장·표시 한계·다음 승인 경계. **"전체 사이클 성공" 선언 금지**. `manifest.json`.
## Constraints
- DEME/새 물리 0, 설치 0, 로컬 DEME 와 동시 실행 금지(현재 없음). 메인 repo·상태 원장·다른 worktree 는 읽기만. W19 원자료 무수정(전후 sha 영수증). 총 4 h 상한, 초과 예상 시 preamble `ask`.
## Ownership
이 워커만 `replay_20260918/`. 메인이 원장·relay 소유.
## Observable acceptance
RRD+RBL verify PASS · 계약 검사 PASS · visual_mapping 288행 · 결정 PNG 6장(중복 시 사유) · Isaac 출처 합 288 + MP4 · 검사기 RESULTS · inspection.json 실제 관찰 · REPORT · manifest · worker_done(--report-path REPORT_replay.md, --files-modified, 3문장: 검사기 결과·표시 한계 수치·GPU 단계 실제 경과 시간).
