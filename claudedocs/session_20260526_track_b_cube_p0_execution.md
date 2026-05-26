# Session 2026-05-26 — Track B Cube P0 Execution (P0.1 + P0.2 done, blocked at hardware disconnect)

## TL;DR

Track B cube P0를 plan에서 **실행으로 전환**. 스크립트 3개 작성+dry-run, 포트
serial 검증, **P0.1 PASS**, **P0.2 anchor 실측 lock-in**, fixed-jaw URDF 확증,
Gauge 스크립트 stall auto-stop 개선. **세션 끝에 두 로봇 팔이 USB 레벨에서 동시
단선** → pose_ctrl smoke / P0.3 부터는 재연결 대기. Track A 미접촉.

## 신규/수정 파일 (코드)

1. `safety_p0_guards.py` (신규) — G1-G10 공유 헬퍼 + `DryRunArm` mock. 상수 전부
   소스 검증: JOINT_LIMITS/Z_FLOOR=-130/DIST_MAX=420/INIT_POS=[0,0,90,0,0,5].
   `move_joints`가 speed>200이면 ValueError (G10). G7 idempotent cleanup.
2. `trajectory_p0_gripper_sweep.py` (신규) — P0.2 Gauge sweep. **stall auto-stop**
   (연속 2 blocked cmd가 PLATEAU_TOL=0.6° 이내면 중단) + MAX_GAP=16 서보보호 +
   종료 시 jaw 열고 cube 제거 prompt (cleanup의 gripper→5 짓누름 방지).
3. `hw_p0_sanity.py` (신규) — P0.1: INIT_POS max_diff≤3° + Kinect 1-frame +
   pose_ctrl smoke(default-OFF, `--pose-ctrl-smoke`, FK guard + abort).
- 전부 py_compile OK, dry-run 검증, auto-stop 로직 실데이터 단위검증(cmd30 중단).

## 측정 결과 (실측, 재연결 후 재현 가능)

### 포트 매핑 (serial 검증, 가정 아님)
- ttyUSB0 ↔ serial `7842202ff8d9ef11b33f513dc8728757` = **Leader** ✅
- ttyUSB1 ↔ serial `ee7a06468e98ef1194edca63a8793231` = **Follower** ✅
- START_HERE P4(5/22) 기록과 일치. **재연결 시 재열거 가능 → by-id 재검증 필수**.

### P0.1 (PASS)
- Check1: torque→INIT_POS, state=[0.18,1.93,91.14,0.09,-0.09,4.66], **max_diff=1.93°≤3°**.
- Check2: Kinect 1280×720 저장 `logs/hw_sanity_p0/p0_kinect_frame.png`. 분홍 cube
  5개 HSV 검출, native ~33-38px. **224 square-resize 후 ~6px(가로)×11px(세로)**
  → ⚠️ **flag (P1 전 결정)**: cube가 모델 입력 해상도에서 작음(sponge 125mm 대비).
  실제 학습 resize 경로(square squash vs aspect-keep crop) 확인 + 필요시 카메라/
  cube 거리 조정. **P0.3~0.7 grasp 역학과는 무관 → calib 계속 가능**.

### P0.2 anchor (사용자 실측 Gauge sweep, lock-in)
- cmd40→state40.34(SETTLED), cmd35→37.97(TIMEOUT), cmd30→37.88, cmd25→37.88.
- **30mm cube → moving-jaw state ~37.88°에서 정지** (cmd30/25 재현). hold 전부 Y,
  drift +0.00 (cube 안 미끄러짐).
- **grip cmd target ~28** (stall-10). **cmd 0-5 절대 금지** (서보 stall, gap+33,
  v3 degenerate 위험). → tech_cube_grasp_anchors P1 "close cmd 0-5°" **수정됨**.
- deg→mm 검산: 0.75×37.88=28.4mm ≈ cube 30mm → **두꺼운 pad 없음 시사** (육안 미확인).
- caliper jaw_mm(state) 2-3점 실측 = offline TODO (object-agnostic curve 확정).

### fixed-jaw 확증 (사용자 지적 → URDF 검증)
- `local_assets/roarm_m3/urdf/roarm_m3.urdf`: movable gripper joint는
  **`link5_to_gripper_link` 단 1개** (revolute 0~1.571rad). moving jaw=gripper_link
  (link5 offset xyz=(0,18.8mm,52mm)), fixed jaw=link5 본체(hand_tcp 115mm 돌출).
- → cube center ≠ TCP 중심선, cube는 fixed jaw 쪽에 붙고 moving jaw가 clamp.
  **P0.3 approach + P0.4 grasp z + L-F는 fixed-jaw lateral offset 반영 필수.**
  Gauge "쑤셔넣기"로는 grasp 기하 못 잡음 (사용자 옳음).

## 사용자 비판 수용 (검토)
- "static Gauge 의미 있나?" → **반반**. 의미: stall점/grippable/grip cmd 수정/
  pad 없음. 사용자 옳음: 최적 grip은 L-F 데이터 평균에서 나옴, fixed-jaw grasp는
  Gauge 밖 → P0.3/0.4/L-F 몫, 나머지 sweep(20→0)은 무가치+서보발열 → auto-stop 개선.
- 결정(사용자): Gauge 개선(done) → P0.3(fixed-jaw 반영) → L-F 데이터.

## 하드웨어 단선 (세션 종료 사유)
- 세션 끝에 `/dev/ttyUSB*`+`/dev/serial/by-id/` 둘 다 소실, `lsusb` CP210x 둘 다
  없음, Kinect는 정상, cp210x 모듈 로드(0 devices). → **두 팔 동시 물리 단선**
  (공유 USB허브 전원 / 외부전원 / 케이블). 소프트웨어 아님.
- 사용자 확인 요청: 팔 전원 LED, USB허브/케이블. 재연결 후 by-id 재검증부터.

## 다음 step-by-step (재연결 후)
1. `ls /dev/serial/by-id/` → serial로 Follower 포트 확정 (USB0/USB1 가정 금지).
2. Follower read-only 응답 확인 (error default [180,-180,-90,-180,180,180] 아닌지).
3. `python hw_p0_sanity.py --pose-ctrl-smoke` → pose_ctrl IK RELIABLE/UNRELIABLE 판정
   = P0.3 구현 방식 게이팅 (RELIABLE→IK approach / UNRELIABLE→joint 직접/L-F).
4. P0.3 approach angle (wrist_p 75/60/45°, **fixed-jaw lateral offset 반영**, FK guard).
5. P0.4 grasp z (FK primary, +8/+12/+15mm tipping 비교) → P0.7 L-F single pick 5/5.
6. P1 250ep (grip cmd ~28 분포, cube 가시성 flag 해결 후).

## Track A 경계
본 세션 Track A 미접촉. Track A 최신 = v7 close_26 audit FAIL (static v8 design only).
