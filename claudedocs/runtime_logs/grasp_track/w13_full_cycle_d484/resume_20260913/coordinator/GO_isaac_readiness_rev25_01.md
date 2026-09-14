# 제한적 GO — rev25 Isaac 준비 표시 1회

발행 2026-09-13 19:58 KST. root / Run run_5cba1e55a775.
대상 producer ctx_006e285a14a4 / task_d3ad7187e682. 아래 정확한 새 실행 1회 후 GPU HOLD. 자동 재시도 금지. DEME production GO가 아니다.

## 인수한 근거

- rev24 actual14는 정리 포함18.29642초·child/runner0·신호/timeout0·stderr0·Kit Stage closed로 끝났다. post-close 시간2개가 없으므로 전체 준비는 실패였지만 계속된 hang은 아니다. 기본 fast_shutdown=True와 종료 뒤 Python 계측 기대의 불일치를 확인했다. 원자료를 보완하거나 기준을 낮추지 않는다.
- rev25의 유일한 실행 코드 변경은 `AppLauncher(args, fast_shutdown=False)`다. 물리/구도/runner/launcher/6개 시간 필드/외부 시간제한 불변. root가 diff 전체, 독립 테스트 전체 및 보고서를 읽었다.
- 공식 Isaac Sim5.1/extension2.12.2 SimulationApp API와 설치 simulation_app.py:93,126,400의 기본 빠른 종료, Isaac Lab2.3 AppLauncher 설치 app_launcher.py:59,415,497,811의 공개 kwargs 전달을 대조했다. False는 정상 extension 정리를 요청하며 실제 반환 성공을 보장하지 않는다.
- 공식: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html ; https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/app/app_launcher.html
- 독립 `REV25_FAST_SHUTDOWN_DELTA_AUDIT_02.json` SHA `2a1abe7116cfa9b8258945996a50a23bb9372441e747fd41f40aa907dd036837`,13/13. root가 읽은 동일 테스트를 새 ROOT 산출 경로로 직접 재실행해13/13·rc0 확인: `REV25_ROOT_DELTA_RECHECK_01.json`. 실제 AST 호출의 false 전달, 기본값/True 음성 대조, 변경 범위, source26/mini10/external15/criteria1 및 실제 출력 경로 부재를 확인했다. native 실행 검증은 아직 아니다.
- root 보존 재검사19:57: `W13R_PRESERVATION_VERIFIED 508`, rc0.
- rev25 PIN SHA `8957361e62f32fbb64aa2911e32007f5fce304fb2510995db7d5ddff4b88e4be`.
- renderer SHA `870e78bff1891c75758b740b75e3ae88c8548f01aefad73bed780ef2b5e4979b`.
- launcher SHA `3460dd6b461f4ae943b6d444b68428258df5e36d215b6c13b3e59163ef6106ac`.
- close_diagnostics SHA `2929db133ba6d837064a3cb800c9bcc729f4799cd2e1d268070c0378750a4d4c`.
- 15_planned plan SHA `7714ceae739746f6a74656ce4d1a68f20bc4cdf56c77952d06e5b7d46c13ae85`, mini PIN `ef44764cca28bd7a21cbb0efb2b6ad3181294c45b77fd4b1f7042c9282271e45`, mini MANIFEST `dea6f129f138a25aa697e379791828915b2b5cda800e157bcd32ee9ab6ff3155`.

## 정확한 실행

cwd: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation`

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev25/src/readiness_launcher.py --run /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/implementation/attempts/smoke_wall_regression_01 --out /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_15_actual --budget-s 180 --max-frames 8 --synthetic-phases 6 --criteria /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev25/criteria.json --res 1600x900 --fps 10 --usd /home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd
```

직전 핀 재검사 및 actual15 부재 확인. populated planned는 실행하지 않는다. 새 actual 사본의 자기 경로와 동일바이트 criteria 사본 경로만 변경 허용.

## 관찰과 중단 경계

1. 총180초 정리 포함: child135/cleanup27/grace18, TERM162/KILL179.5. 6개 필수 시간과 실제 `/app/fastShutdown=False`, clear/probe/close 순서 및 최종 rc/owned group 부재를 확인한다. 강제 rc0/skip_cleanup/수동신호/예산확대 금지.
2. raw2행은 같은 t0/sync0, 합성6행은 기존 입자행 복사와 선언 자세다. 새 DEME0이며 전체 물리 결과가 아니다. Isaac 초기 warmup8과 표시8프레임 시계를 분리 기록한다.
3. UTC/argv/PID·PGID·원본 영수증/8PNG/MP4/매핑 해시 보고. root 직접 8원본 PNG 검사와 독립 종료/매핑 검수 후 인수 판단. childrc0만으로 성공 판정 금지.
4. 실행 후 HOLD. 이미 받은 사용자9시간 비용승인은 유지되며 준비 인수와 최종 preflight 뒤 별도 root 본실행 GO가 필요하다.
5. 실물조회/구동/PID/토크/카메라·학습·A/B/C·설치·commit/push 금지. 원장 소유 root. 옛실패/원자료 불변.
