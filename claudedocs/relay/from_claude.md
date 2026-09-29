# from_claude.md — Claude → Codex/Cursor 인계 (relay)

## §0 이 파일의 규약
- **쓰는 쪽 = Claude 세션 하나.** 읽는 쪽 = 다음에 이 repo 를 여는 **다른 도구**. Claude 연속이면 `START_HERE.md` 로 재개.
- **덮어쓰기.** 상태 정본 = `START_HERE.md`(여기 안 베낌). 규칙 = `AGENTS.md`. 여기엔 만진 것·만지지 말 것·함정·승인 대기만.

## §2 2026-09-29 21:2x Claude(W25 3일차 완료) → Codex/Cursor

**한 일(전부 미커밋, 사용자 commit `407baf0` 이후)**
- RunPod 병행 본 실행 2회 완주·회수·종료(우리 pod 0). 동결본 `claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/exec_rev34_paperbox_20260929/`(runs/·receipts/·RUNPOD_LOG_W25.md·local_tools/ 포함). podB 회계 `runs/podB_pro6000x2/postprocess_20260929/`. post06 재생 패치 `w25_realign_fullcycle_d498/replay_post06_convA_20260929/`(렌더 미실행).
- 원장: D500 append(`DECISIONS.md:30350`, 앞 30348줄 md5 불변, 백업 `.bak_20260929_pre_d500`) · DECISIONS_ACTIVE·LEDGER_RECENT·EXPERIMENT_LEDGER :607 · START_HERE overwrite 부분 갱신 · 세션 문서 §13.
- 학습 방향 문서 `claudedocs/research/w25_learning_plan_20260929/{LEARNING_PLAN_BRIEF,LEARNING_PLAN_DETAILED,FULLCYCLE_VS_LEARNING_DATA}.md` + `~/Downloads` 사본, 선행 논문 패키지 `~/Downloads/proposal_priors_package_20260929/`(워커 worktree `.claude/worktrees/agent-a65ed2a45d31fd64a`, 미merge).

**만지지 말 것**: exec 동결본(runs/·receipts/·RUNPOD_LOG·local_tools 제외) · 회수 원자료 run_01 · 워커 worktree 산출 merge 금지.

**함정**: 시뮬 라벨은 기하 기준(lift_end 더미 위 들린 알) — tool_residual 분류는 절반 이하 · 스모크 속도비로 전체 비용 외삽 금지(R_full 1.449 vs 2.556) · ssh `A && B && setsid … &` 채널 점유 · Monitor 무이벤트 → Bash 대기 루프.

**승인 대기**: 학습 단계 1단계(셀 제작·RunPod 셀 단가 실측) 착수 · post06 렌더(로컬 GPU) · 실물 측정(M1·M4·M5, 로봇 구동 없음) · 카메라 연결·정합(구동 승인) · 옛 브랜치 4개 삭제.

---
### (이전 §2 2026-09-28 밤 원문, 참고용)
2026-09-28 밤 Claude(W25) → Codex/Cursor

**한 일 (repo에 남은 변경, 전부 미커밋)**
- 새 세션 문서 `claudedocs/session_20260928_w25_realign_fullcycle_prep.md` · `START_HERE.md` overwrite(백업 `START_HERE.md.bak_20260928_pre_w25`) · `DECISIONS.md` **D498 append**(앞부분 md5 불변 실측, 백업 `.bak_20260928_pre_d498`) · `DECISIONS_ACTIVE.md`·`LEDGER_RECENT.md` 갱신 · `EXPERIMENT_LEDGER.md` 1행 append · 이 relay.
- 새 case 폴더 `claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/coordinator/`(과제서 3 + worker_start 영수증). rev34 산출은 **워커 worktree** `w25-rev34-fullcycle/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/rev34/`(미merge).
- Orca worktree 신규 5개(`w25-rev34-fullcycle`·`w25-render-cad`·`w25-pile-runpod-plan`·`w25-pile-flat40`·`w25-dt-basis`, HEAD fc557db, 산출 `claudedocs/research/w25_*_20260928/`) — 전부 미merge, 워커 release(Run `run_d0eba99ce41b`).
- 랩미팅 자료: 슬라이드 v5(11장, `gripper_note` 추가), `claudedocs/research/labmeeting_20260929/{OUTLINE.md, lab_pc_bundle/LAB_PC_UPDATE_20260928.md, lab_pc_bundle/figures/08_gripper_shell_vs_cad.png, lab_pc_bundle_20260928b.zip}`.
- worktree 보관 **완료**: W23 7 + W24 3 → SSD `orca_worktree_archive/RoArm_Project/<이름>`(전량 SHA 대조·태그 `archive/<이름>`=3267dcb·bundle·`orca-ide worktree rm`·원경로 심링크·재대조, `ARCHIVE_INDEX.md` 10행). 남은 실제 worktree 15(메인 + 14). 총 보관 28. `w23-sim-reference` 잔류 Codex 터미널 종료. auto-memory `MEMORY.md` 회전(`MEMORY_archive_20260928.md`).

**만지지 말 것**
- 워커 worktree 5개(`w25-{rev34-fullcycle,render-cad,pile-runpod-plan,pile-flat40,dt-basis}`) 산출·rev34 사본 — merge 금지. `rev32_frozen_copy/` 무수정 유지.
- 기존 `isaac_replay_w13.py`(post04)·`w13_rerun_export.py` 원본 — post05 는 B 워커 폴더의 사본.

**함정**
- 🔴 **재생 화면의 툴 = 충돌 셸(보울+캡)**, USD 의 S1 v1 문 CAD(`gripper_link`)는 post04 의 `gripper` 이름 필터가 숨김(주석 "순정 그리퍼"는 오기). 고정부 CAD(`grab_fixed`)는 보임. post05(B 초안)는 둘 다 숨기고 `/World/s1_cad/*` 를 원시 포즈로 그림.
- 🔴 **시뮬 상자 = 22 cm 벽이 로봇 쪽**(W19 `t_robot (0.350, −0.008, −0.283)`, 로봇이 상자 31 cm 축 위). `w13_kinematics.py:10-16` "(0,+R)" 도크스트링은 낡음. 실물 = 31 cm 벽.
- 🔴 4 cm 평평한 층 전체 사이클(현재 종이 상자 310×220 = 61,408~74,067알)은 **15.0~26.4 h** → 현행 cap 32,400 s 로는 기동 불가. cap 은 새 revision·criteria 에서 결과 전에 정한다. 생성기 시드 여백 하한 = 알 외접 지름(4.5 mm) → `--seed-margin-mm 4.6`.
- 🔴 종이 상자 벽(230 mm)이 바닥판보다 9.6 cm 높아 운반 높이(립 45 cm)가 벽 윗단보다 22 mm 위뿐 — `travel_cm` 상향은 사용자 결정. RunPod 랩 계정엔 타인 pod 7개(RUNNING 1) — 절대 조작 금지.
- slab 생성기 가장자리 여백 13.5 mm/변(`sim_deme_pile.py:435,446-470`) → `--target-depth-m` 만으로 전면 평탄층 보장 안 됨. 부트스트랩 v5 는 W19 경로 하드코딩.
- 워커 Claude Code 가 "Auto mode is unavailable(safety verdict 없음)" 로 턴을 끝내면 `orca-ide orchestration send --from <코디 handle> --to dispatch:<id>` 로 깨울 수 있다. `send`·`worker-start`·`run-create` 는 `--from` 필수(비-Orca 터미널), `worker-start` 에 `--no-parent` 없음.
- Codex 0.158.0 = npm 최신(09-28 22시) → 업데이트 안내 없음. 이후 새 버전이 나오면 W22 함정 재발 가능.

**09-29 진행(2일차)**: ① post05 재렌더 완료 ② 더미 2개 생성 완료(61,408 FAIL 36.6 mm / 67,737 PASS 40.2 mm, `<case>/pile_flat40_20260929/`) ③ PB1~PB5 재검토 중(`w25-rev34-fullcycle/.../rev34/{preflight,domain,armwall,dryrun}/paperbox_final{,_n67737}`). 사용자 commit `407baf0`(00:02) 이후 변경은 미커밋.

**승인 대기**
- ③ RunPod(사용자 결정: 2차 더미 67,737알 확정·C4 수용·travel 45·cap 32 h·**비용 상한 없음**; GPU 선택만 대기 — 권고 병행 4090×1 + PRO 6000×2, 우리 pod `roarm_w25_*` 만, 타인 pod 8개 불가침): rev34 전체 사이클(부피밀도·cap ≥32 h·비용 상한·기동 시각·C4/travel_cm 결정; 우리 pod `roarm_w25_*` 만) ④ 옛 브랜치 4개 삭제 여부 ⑤ 실물 실측(알 질량·상자 안쪽 치수·벽 높이 안/바깥). dt 사다리 GPU 는 사용자 보류.

**검증 방법**
```
cd /home/cgxr/Documents/Robotics/RoArm_Project
grep -n '^## D498' claudedocs/DECISIONS.md
git tag -l 'archive/w2[34]-*' | wc -l                          # 10
ls /media/cgxr/ROBOT_DEV/orca_worktree_archive/RoArm_Project/_plan_20260928_w2[34]-*/bundle_verify.txt | wc -l   # 10
/home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w25-pile-runpod-plan/claudedocs/research/w25_pile_runpod_plan_20260928/budget_w25.py --check   # PASS
```
