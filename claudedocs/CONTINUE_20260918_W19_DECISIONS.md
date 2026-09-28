# 새 세션 재개 — W19 후처리 완료 뒤 사용자 결정 대기 (2026-09-18 15:3x 작성)

W19 A(rev32 전체 사이클 첫 완주) 원자료는 5층 검증(러너·생산 회계·rev31 재분류 0 불일치·재생 8/8·독립 감사 9/9+9/9)을 마쳤다(D491, 원장 :603). RunPod pod 없음, worktree 7개(핵심·활성만), 12개는 외장 SSD 아카이브 심링크. 과거 GO/COMMANDS 는 재실행하지 않는다.

## 반드시 읽기
1. AGENTS.md → START_HERE.md → DECISIONS_ACTIVE.md(D485~D491) → LEDGER_RECENT.md(:596~603) → relay/from_claude.md(9/18).
2. `claudedocs/session_20260917_w19_runpod_afternoon.md` §9~§14.
3. 1·2단계 보고 `w19-postprocess/.../postprocess_20260918/REPORT_postprocess.md`, `w19-replay/.../replay_20260918/REPORT_replay.md`, 감사 2건 JSON, `w19-rev33/.../rev33_20260918/REPORT_rev33.md`, 역할 agent README, 설계 브리핑 2건.
4. `git status --short`, `git worktree list`(7개), 외장 SSD 장착 여부(`ls /media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project`).

## 사용자 결정 항목(순서 제안)
1. 정착 cadence 규약: (a) 문구 개정(프레임 0.1 s 인정) / (b) rev33 옵션 ≤0.046 s 로 종료 창 촘촘 기록. + 상한 규약 32,400→43,200.
2. rev33 채택: GPU 300알 스모크 → 정본 승격.
3. 설계 case: ② 알 개수 먼저(수단 (i) ROI / (ii) 소형 슬랩), 같은 N 에서 ①. **① 은 재정의됨**(9/18 사용자 관찰: 실물 운반 유출 ≈0) — 시뮬 문 닫힘을 실물 관절 ≈1.2° 에 맞추는 보정 case, 성공 기준 관절 ≤1.3° + 유출 ≈0, 1순위 A1. 반복 셀 n≥3. 세션 문서 §15.
4. 역할 agent 4개 채택(3건 결정) → 본 repo `.claude/agents/`.
5. RunPod 재사용 시 `_obj` 포함 회수 스크립트 수정.

## 붙여넣을 요청문
```text
/home/cgxr/Documents/Robotics/RoArm_Project에서 이어서 작업해.
AGENTS.md와 START_HERE.md의 부팅 순서를 따르고 claudedocs/CONTINUE_20260918_W19_DECISIONS.md를 전체 읽어.
직전 세션 문서 §9~§14와 D491을 읽어 W19 검증 상태를 복원하고, 사용자 결정 5항목을 표로 브리핑한 뒤 내 결정을 기다려.
물성·형상·알 수·경로·문 속도·보호선·dt 1µs·cd_update_freq·원자료는 바꾸지 마. 새 물리·pod·commit/push·실물·설치·학습은 내 승인 뒤에.
작업자는 Orca worktree에 배정하고(새 worktree 우선, Claude claude-opus-5; Codex는 새 worktree에서만) 여기서 보고만 받아. 메인만 상태 원장·relay를 소유해.
관찰 가능한 절차→수치→근거 파일→한계·다음 승인 순서로 한국어로 보고하고 종료 시 START_HERE·새 session·relay를 갱신해.
```
