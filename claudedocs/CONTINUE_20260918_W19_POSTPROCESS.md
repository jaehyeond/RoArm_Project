# 새 세션 재개 — 2026-09-18 (W19 A 완주 원자료 후처리) / RunPod pod 없음

작성 2026-09-17 밤. W19 RunPod 본 실행 A(rev32 전체 사이클 첫 완주)·B(같은 입력 반복 2셀)가 끝나 로컬로 회수·해시 대조·pod 종료·원장(D490, :601~602)까지 마쳤다. 이 문서는 재개 안내이며 과거 GO/COMMANDS 를 재실행하지 않는다.

## 반드시 읽기
1. AGENTS.md → START_HERE.md → claudedocs/DECISIONS_ACTIVE.md(D485~D490) → LEDGER_RECENT.md(:596~602) → relay/from_claude.md(9/17 밤).
2. claudedocs/session_20260917_w19_runpod_afternoon.md §1~§9(직전 세션 정본).
3. `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/run_01/SUMMARY_A.json` + `RETRIEVAL_RECEIPT.json`, `B_scoop_repeat/cell_dt1e6_seed46{1,2}/scoop_s1_seed46x.json`.
4. 설계 브리핑 2건 `/home/cgxr/orca/workspaces/RoArm_Project/w19-design-briefs/claudedocs/research/design_briefs_20260917/`.
5. `git status --short`, `git worktree list`. LFS 보류·commit/push 금지 유지. Codex 주간 한도는 9/18(금) 해제.

## 현재 사실 / 함정
- W19 A: rc0·20,977.8 s·물리 24.807 s 완주. 생산 회계 bin 272(5.51 g)·HOME 0.00006 mm·정착 창 5프레임 안정 272 이나 **cadence_ok false**(≥6프레임·≤0.05 s 규약, 프레임 간격 0.1 s). 재닫기 내부 292(W13 144). **판정 승격 금지** — rev31 파생 회계·규약 검사·재생·독립 감사 뒤.
- W19 B: 포획 517/517/489(로컬 W11 포함 n=3), reclose 사유 pinch_guard/servo_stall/servo_stall. 통계 주장 없음.
- RunPod: pod 전부 종료. 재사용 시 `minCudaVersion 13.0`·부트스트랩 v5·꾸러미 sha·`--allow-go-step` 은 GO 뒤. 드라이버 570 호스트 DEME 사망(원인 미확정). 랩원 pod 조회만.
- 원장 무결성: D490 append 전 앞 30,227줄 md5 `aa54d73a410e83fee134026ba2fe738e`(백업 `.bak_20260917_pre_d490`).

## 다음 계획(각 항목 GO 필요)
1. **A 후처리 CPU**: W14 `derive_v2.py`·독립식 검사기를 W19 run_01 NPZ 에 적용(새 파생 폴더 `w19_runpod_d487/A_full_cycle/derived_v2_rev31/` 제안), `RAW_SCHEMA_REQUIRED`+ERRATUM 1~4 검사(전환 11개·바닥식), W13 대비 재고 전이 대조.
2. **A 재생(로컬 GPU)**: post04 재생 스크립트(`w17-replay-fix/.../post04_20260917/rev/`)로 Rerun RRD/RBL + 결정 PNG(viewer-mcp 캡처) + Isaac 관절 재생, 각 5,400 s 상한. 로컬 DEME 동시 실행 금지.
3. **독립 감사**: Codex gpt-5.6-sol high(9/18~) 또는 Claude opus-5, worktree `w13-cycle-audit` 재사용, NumPy-only.
4. 정착 cadence 규약 결정·설계 브리핑 ①② 수단 선택은 사용자.

## 붙여넣을 요청문
```text
/home/cgxr/Documents/Robotics/RoArm_Project에서 이어서 작업해.
AGENTS.md와 START_HERE.md의 부팅 순서를 따르고
claudedocs/CONTINUE_20260918_W19_POSTPROCESS.md를 전체 읽어.
직전 세션 문서 §5~§9와 SUMMARY_A.json을 읽어 W19 A 완주 원자료 상태를 복원해.

W19 A 후처리를 준비해: rev31 규약 v2 파생 회계(CPU)와 원자료 규약 검사 계획을 브리핑하고,
재생(Rerun/Isaac)·독립 감사는 내 GO 뒤에 시작해.
물성·형상·알 수·경로·문 속도·보호선·dt 1µs·cd_update_freq·원자료는 바꾸지 마.
작업자는 Orca worktree에 배정하고 여기서 보고만 받아(Claude claude-opus-5, Codex gpt-5.6-sol high 확인).
장시간 GPU 작업은 하네스 밖(setsid)에서 띄워. 메인만 상태 원장·relay를 소유하고 worker 원장을 merge하지 마.
LFS staged D·원본 삭제·force-add·이력 재작성·commit/push·실물·설치·학습·RunPod 신규 pod는 금지.
관찰 가능한 절차→수치→근거 파일→한계·다음 승인 순서로 한국어로 보고하고 종료 시 START_HERE·새 session·relay를 갱신해.
```
