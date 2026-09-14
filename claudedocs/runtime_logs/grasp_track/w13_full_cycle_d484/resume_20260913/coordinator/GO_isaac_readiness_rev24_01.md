# 제한적 GO — rev24 Isaac 준비 표시 1회

발행 2026-09-13 19:37 KST. root / Run run_5cba1e55a775.
대상 producer ctx_006e285a14a4 / task_d3ad7187e682만. 아래 정확한 새 실행 1회 후 GPU HOLD.
사용자가 승인한 W13 준비 범위이며 DEME 본 실행 GO가 아니다. 자동 재시도는 없다.

## 읽고 인수한 근거

- root가 rev23→24 두 소스 diff와 `record_single_call` 전체, 생산 CPU 검사 파일 전체, 14_planned 계획 및 manifest 전체를 읽었다. 공개 `clear_instance()`가 마지막 표시 뒤 기존 probe/update/close보다 먼저 호출된다. 카메라/전역 정리나 명시적 stop은 추가하지 않았다. 이미 완료된 PHASE를 pre-close dump에 복사한다.
- root가 installed Isaac Lab 2.3 `simulation_context.py:638` 및 Isaac Sim base `simulation_context.py:211`과 공식 2.3 source를 대조했다. 콜백 제거는 설치 코드 사실이지만 rev23 hang의 실제 native 원인 및 이번 해결 효과는 아직 가설이다. 종료 때 backend/device 정리도 포함되며 실행 중 물리 조건 변경이 아니다.
- root 직접 재해시 source26/26, mini10/10, 외부입력15/15, criteria1/1 일치. 14_actual 부재. 기존508파일/HEAD 보존 rc0 PASS(19:28). COMMANDS 변경은 rev23→24 자기 경로뿐임을 실제 diff로 확인했다. physics/runner/launcher/params/numeric/criteria/visual 설정은 불변.
- rev24 PIN SHA `52e0fa7f6afaa56e27900b3dc7724e1c2174edb223a2b76deb42044b9136e20b`.
- renderer SHA `d27aa7ac1d48df0d35e83140fd628274b5e2e80d37d710dec93df6533a898060`.
- close_diagnostics SHA `2929db133ba6d837064a3cb800c9bcc729f4799cd2e1d268070c0378750a4d4c`.
- launcher SHA `3460dd6b461f4ae943b6d444b68428258df5e36d215b6c13b3e59163ef6106ac`(rev23 동일).
- 14_planned/plan/readiness_plan.json SHA `e4b107ab109cbbbe7f2cca420f06a8ffde39b27c9b6d0c397d2675e167f43bec`.
- planned mini PIN SHA `43718618b0646da0dc44c6e2f4ba115f5198f51d610bc02258d6ff1967bae9f2`.
- planned MANIFEST SHA `e3cbca0d40fa95f2a8beedd2ac5b2d7292071e46c912595243c599aec65339f5`.
- 독립 `REV24_CHANGED_PARTS_AUDIT_02.json` 전체 및 SHA `19cc89679c138f6f53aadbf709b1b554d337e2e5afb8448201f827374ac1b000` root 확인. 19/20, 실제 성공 판정 차단 결함0. 일반/제어 예외·진입 기록·양성 반환·AST 순서·공통 예산·저장 검사를 부분 인수했다. 설명 목록 `workflow_non_success_sources`의 clear 실패 이름 누락은 비차단 한계로 남긴다: rc96/실패 bool/전용 clear 영수증은 정확하다. 성공 기준을 완화하거나 이 목록을 새 필수 기준으로 추가하지 않는다.
- 감사 `order_swap` 대조는 실제 순서 교환이 아니라 probe 중복 추가를 검출했다. root가 source로 확인해 정정 요청했다. 해당 대조를 순서 교환 증명으로 세지 않으며 실제 AST 1371→1405→1479 및 root 직접 읽기로 호출 순서를 확인했다. 생산 22/22는 생산 보고이며 root가 pytest 전체를 재실행한 결과는 아니다.

## 정확한 실행 명령

cwd: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation`

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev24/src/readiness_launcher.py --run /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/implementation/attempts/smoke_wall_regression_01 --out /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_14_actual --budget-s 180 --max-frames 8 --synthetic-phases 6 --criteria /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev24/criteria.json --res 1600x900 --fps 10 --usd /home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd
```

직전 위 핀과 입력을 다시 대조하고 14_actual이 이미 존재하면 실행하지 않는다. populated 14_planned를 실행하지 않는다. 런처가 새 actual 사본을 만들 때 planned→actual 및 동일 바이트 criteria 사본 경로 차이만 허용한다. 옛 revision/실패/원시 출력은 수정하지 않는다.

## 실제 관찰과 중단 경계

1. 180초 정리 포함, child135/cleanup27/grace18, TERM162/KILL179.5. clear 호출부터 기존 cleanup 예산에 포함한다. 모든 6개 필수 시간 및 clear 전용 before/after/예외·preclose PHASE·최종 rc·그룹 정리를 확인한다. native가 멈추면 기존 outer 제한이 최종 경계이며 강제 rc0/skip_cleanup/수동 신호/예산 확대는 금지한다.
2. raw2행은 같은 t0/sync0이고 합성6은 기존 입자행 복사·선언된 자세 표시다. 새 DEME0. 초기 Isaac warmup8과 표시8프레임의 시계를 구분해 남긴다. 물리 전체 동작 결과로 부르지 않는다.
3. 시작 UTC/argv/PID·PGID, 최종 영수증 및 8PNG/MP4/매핑 해시를 보고한다. root가 8원본 PNG를 실제 열고 독립 감사는 최종 종료/매핑을 검수한다. probe 통과나 childrc0만으로 준비 성공을 선언하지 않는다.
4. 실행 후 GPU HOLD·자동 재시도0. 본 실행은 이 준비 인수와 최종 독립 사전검수 뒤 별도 root 기술 GO가 필요하다. 사용자 9시간 비용 승인은 유지한다.
5. 실물 조회/구동/PID/토크/카메라·학습·A/B/C·설치·commit/push 금지. 원장 쓰기는 root만 소유한다.
