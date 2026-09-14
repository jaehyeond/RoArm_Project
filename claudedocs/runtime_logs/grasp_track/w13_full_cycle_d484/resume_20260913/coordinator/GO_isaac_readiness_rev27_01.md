# 제한적 GO — rev27 Isaac 준비 표시 1회

발행 2026-09-13 20:33 KST. root / Run run_5cba1e55a775.
대상 producer ctx_006e285a14a4 / task_d3ad7187e682. 아래 정확한 새 실행 1회 후 GPU HOLD. 자동 재시도 금지. DEME production GO가 아니다.

## 인수한 근거와 한계

- actual15는 close21.208초 반환 및 필수6시간 저장 후 child SIGSEGV(-11)로 실패했다. 정확 PID2459761의 커널 기록은 PhysX tensors 모듈 fault를 입증하지만 특정 소멸자나 native 호출의 인과는 미확정이다.
- rev27은 rev25 대비 두 소스의 참조 해제 및 관측만 변경했다. 기존 clear_instance 반환 뒤 renderer 보유 cams/scene/robot/sim 셀을 None으로 바꾸고, 기존 probe/update/close는 그 뒤 그대로 실행한다. 실제 카메라2개도 weakref 관측에 포함한다. dict은 weakref 미지원임을 명시한다. 물리/구도/runner/launcher/6필드/외부 시간제한은 불변이다.
- root는 renderer/helper 전체 diff와 독립 시험·보고 전문을 읽고 동일 CPU 시험을 새 ROOT 경로에 재실행해16/16 rc0 확인했다. 별도 카메라/로봇 참조를 남긴 음성 대조는 각각 실패한다. 테스트는 실제 helper+가짜 객체의 실행이며 실제 Isaac 소멸자 효과나 실제 renderer 전체 AST 실행을 증명하지 않는다.
- 감사 `REV27_REFERENCE_RELEASE_DELTA_AUDIT_01.json` SHA `7474b652bb4f28b90d174bc3195c42eb80bb034c0df3bbf3b3f7b09c8e0584a5`. root 재실행 `REV27_ROOT_DELTA_RECHECK_01.json` SHA `941aef82bcfab0e619e997456e2ffaad2f13e7a84920ffd85997d874bbd659a3`.
- root20:32 직접 해시 재검사 source26/mini10/external15/criteria1 =52개 불일치0, actual16 및 run_01 부재. 기존508 보존 재검사 rc0 PASS.
- rev27 PIN `a631bb8bfc4faa4723fba3f63bdfee8ed0c0320292f8b246ab20e0ec482ae1b4`.
- renderer `f91274d52871a3411751cc0df3af61407aceb7da40231673b030da6ffabf7885`; helper `8c62898ae91587dd262ad0d2e2c73a41cd3467afcd072523c9435b75e070b3a9`; launcher `3460dd6b461f4ae943b6d444b68428258df5e36d215b6c13b3e59163ef6106ac`.
- 16planned plan `1f7dcb1d4a412b9ce320bd6ef8ec5280cb57cc8505db53043d3e1cfef1e27462`; mini PIN `24f68a23619be93cd6dcbfb2f78e6eadcaa285fdcdc1dd81129146506d7caa57`; mini MANIFEST `3c5d2b0a3bd7f300314cfd67c7bf1c8a66e114e803363dd02b23ba10b7d0134d`.
- 공식 근거: IsaacLab2.3 Camera published source 및 Adding sensors on a robot 예제의 main 반환 뒤 close 순서. 설치 camera.py161~169 및 renderer 마지막 객체사용/close 순서와 대조했다. 수명 순서 후보의 근거이지 충돌 해결 보장은 아니다.
  https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sensors/camera/camera.html
  https://isaac-sim.github.io/IsaacLab/v2.3.0/source/tutorials/04_sensors/add_sensors_on_robot.html

## 정확한 실행

cwd: `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation`

```bash
PYTHONDONTWRITEBYTECODE=1 /home/cgxr/miniconda3/envs/roarm/bin/python -B /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev27/src/readiness_launcher.py --run /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/implementation/attempts/smoke_wall_regression_01 --out /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_16_actual --budget-s 180 --max-frames 8 --synthetic-phases 6 --criteria /home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/rev27/criteria.json --res 1600x900 --fps 10 --usd /home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd
```

직전 핀 재검사 및 actual16 부재 확인. populated planned는 실행하지 않는다. 새 actual 사본의 자기 경로와 동일바이트 criteria 사본 경로만 변경 허용.

## 관찰·중단 경계

1. 총180초 정리 포함: child135/cleanup27/grace18, TERM162/KILL179.5. clear→참조해제→probe→close 순서, 5개 weakref 생존 여부와 실제 소요시간, 필수6시간, 실제 fastShutdown=False, 최종 rc 및 owned group 부재를 보고한다.
2. 참조 해제 실패는 기존rc96, close timeout은 기존97/외부124로 비성공 유지. 수동 신호/예산 확대/강제rc0/skip_cleanup/전역GC/직접소멸자 호출 금지.
3. raw2행은 t0/sync0이고 합성6행은 기존 입자 복사와 선언 자세다. 새 DEME0이며 전체 물리 결과가 아니다. 초기warmup8과 표시8프레임 시계를 분리한다.
4. UTC/argv/PID·PGID/원본 영수증/8PNG/MP4/매핑 해시 보고. root8원본PNG 직접검사 및 독립 종료·매핑 검수 후 인수 판단한다. 영상생성·childrc0만으로 성공 판정 금지.
5. 이후 HOLD. 기존 사용자9시간 비용승인은 유지되며 준비 인수와 최종 rev27 preflight 뒤 별도 root 본실행 GO가 필요하다. 실물/학습/A/B/C/설치/commit/push 금지, 원장 root 소유, 기존실패 불변.
