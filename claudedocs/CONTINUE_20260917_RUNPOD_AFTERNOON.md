# 새 세션 재개 — 2026-09-17 오후 (RunPod 랩 계정 수령 후 본 실행) / 야간 3건은 완료·검증됨

작성: 2026-09-17 아침. 직전 세션(9/16 저녁~9/17 아침)은 W14 규약 수정(rev29~rev31), W15 dt ladder, 야간 W16/W17/W18과 교차감사, 원장 D485~D489·실험 원장 :596~:600 갱신까지 마쳤다. 이 문서는 컨텍스트 상한 때문에 남기는 재개 안내이며 과거 GO/COMMANDS를 재실행하지 않는다.

## 반드시 읽기
1. AGENTS.md → START_HERE.md → claudedocs/DECISIONS_ACTIVE.md(D485~D489) → LEDGER_RECENT.md(:596~600) → relay/from_claude.md(9/17).
2. claudedocs/session_20260916_w14_raw_repair_dt_plan.md §1~§8(직전 세션 정본).
3. 야간 보고 3건: `/home/cgxr/orca/workspaces/RoArm_Project/w16-profiling/claudedocs/runtime_logs/grasp_track/w16_profile_d486/REPORT_w16.md`, `.../w17-replay-fix/claudedocs/runtime_logs/grasp_track/w17_replay_fix_d484/post04_20260917/REPORT_post04.md`, `.../w18-cohort-cause/claudedocs/runtime_logs/grasp_track/w18_cohort_cause_d484/analysis_01/REPORT_w18.md`; 교차감사 `.../w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w16_w17_overnight_audit_20260917/`.
4. 실행안: claudedocs/research/dt_expansion_plan_20260916/DT_EXPANSION_PLAN.md, 사용자 읽기용 `~/Downloads/실험보고 9.29.md`(§12 분배안, §13 야간 요약; repo 사본 research/dt_expansion_plan_20260916/).
5. `git status --short`, `git worktree list`. 새 worktree 3개(w16/w17/w18)는 보존·미merge. LFS 보류·commit/push 금지 상태 유지.

## 현재 사실 / 함정
- 다음 물리 실행의 정본 후보 = **rev32**(`w16-profiling/.../w16_profile_d486/rev32/`, rev31 규약 v2 + sync별 GetNumContacts/GetUpdateFreq/질의비용; GetExpandFactor는 바인딩에 없음). `--max-particles`는 numeric_inputs 도메인 고정과 충돌(fail-closed) → 알 수 변경 case는 증거 파일을 새로 만든다.
- W16: 비용은 더미 상시 잠재 접촉쌍 ~20만이 지배, 물리 1초당 ≈1,200~1,700 s(이 노트북 GPU). 전체 사이클 ≈8.5 h(추정). 손잡이 = 알 개수(dt는 D486 차단).
- W18: 재닫기 후 문 3.55°·이음새 7.01 mm > 펠릿 4.5 mm → 운반 누출. 다음 물리 변수 = 문 닫힘/이음새 하나(승인 필요, 실물 토크 1.96×0.9 N·m·물림 3 N 아래 설계 결정).
- W17: 재생 결함 3건 해소(post04). 헤드리스 결정 화면은 viewer-mcp로 인제스트 완료 관측 후 캡처(D489). W13 과학 판정 불변.
- 하네스 함정: 워커 백그라운드 GPU 작업은 하네스 SIGTERM에 죽는다 → `setsid nohup`. settled Codex 터미널 재사용 실패 → `--retry-of` 새 터미널. Orca 코디네이터는 `terminal create` + `--from <handle>`; `check --wait` 파일에 heartbeat 줄.
- DEME 2.4.0은 PyPI cp311 manylinux 휠. RunPod pod에서 드라이버/NVRTC 호환은 구 회귀 smoke(~90 s)로 확인. Isaac 렌더는 로컬 전용.

## 오후 계획(사용자 분배안, 각 항목 GO 필요)
1. RunPod 계정 수령 → MCP/SSH로 pod 사양·요금 확인, 12 h 상한 예산 승인.
2. pod 환경: python 3.11 + `pip install deme==2.4.0` numpy scipy trimesh; 입력 전송(pile NPZ 659d6b0b…, s1_v1 STL/design.json, sim_deme_scoop_s1.py 2e40f7ed…, rev32 src+params, sim_scripts/roarm_kinematics.py) + 해시 대조 + smoke.
3. 본 실행 A: 전체 사이클 1회 rev32(dt 1 µs, 20,000알, seed 460, W13 params 동일, `--max-wall-s` 12 h 상한, 러너 영수증). 출력은 새 case 폴더(예: `grasp_track/w19_runpod_full_cycle_d487/`)에만.
4. 본 실행 B(병렬 pod): 스쿱 구간 seed 461·462(sim_deme_scoop_s1.py, params_w11_dt1e6 기반, timestep 1e-6, seed만 변경).
5. 결과 회수(rsync) 후 로컬에서 rev31/rev32 회계·Rerun·Isaac·Codex 독립 감사. pod Terminate.
6. 별도 승인 case: 문 닫힘/이음새 간격(W18), 알 개수 비용 곡선(W16), 규약 결정 잔여(이미 v2 채택), LFS/commit/push.

## 붙여넣을 요청문
```text
/home/cgxr/Documents/Robotics/RoArm_Project에서 이어서 작업해.
AGENTS.md와 START_HERE.md의 부팅 순서를 따르고
claudedocs/CONTINUE_20260917_RUNPOD_AFTERNOON.md를 전체 읽어.
직전 세션 문서 §8과 야간 보고 3건·교차감사를 읽어 현재 상태를 복원해.

RunPod 랩 계정을 받았으면 pod 사양·요금을 확인하고 예산을 나에게 보고한 뒤,
본 실행 A(전체 사이클 rev32, 12시간 상한)와 B(seed 461·462 스쿱 반복)를 준비해.
pod 환경 구축·smoke·입력 해시 대조까지는 진행하되 본 실행 시작은 내 GO 뒤에 해.
물성·형상·알 수·경로·문 속도·보호선·dt 1µs·cd_update_freq는 바꾸지 마.
문 닫힘/이음새 case와 알 개수 비용 case는 설계안만 브리핑하고 구현하지 마.

작업자를 쓰면 Orca worktree에 배정하고 여기서 보고만 받아(Claude claude-opus-5, Codex gpt-5.6-sol high 확인).
장시간 GPU 작업은 하네스 밖(setsid)에서 띄워. 메인만 상태 원장·relay를 소유하고 worker 원장을 merge하지 마.
LFS staged D·원본 삭제·force-add·이력 재작성·commit/push·실물·설치·학습은 금지.
관찰 가능한 절차→수치→근거 파일→한계·다음 승인 순서로 한국어로 보고하고 종료 시 START_HERE·새 session·relay를 갱신해.
```
