# t3t_prereg — T3 TILTED-approach physical grasp leg (43rd session, 2026-08-10)

작성 시각: 실행 **전**. 이 문서에 적힌 tuple 그대로 **1회** 실행한다.

## 0. 이 leg가 존재하는 이유 (승인 경위)

- 교수님이 **접근 기움을 허용**했다 (사용자 전달, 2026-08-10 43rd). D419의 "수직 상부 접근"
  중 **각도**만 완화되고 **파지점(상면 중심)은 불변**이다. HARD RULE #18 준수 —
  타깃을 Claude 단독으로 바꾼 것이 아니다.
- 사용자 우선순위 원문: "target object를 잡을 수 있는게 가장 중요해."
- 선행 4건이 모두 read-only로 끝나 있다: D424(수직 하강 목표 자체가 기하 위반) →
  D430(조립은 옳고 병목은 과제 사양) → D431(기움 6°면 물림 양수) →
  D432(기운 자세는 실제로 도달 가능, 지면 간섭 없음) →
  **t3r_n10_ctq5(43rd, 이 leg의 유일한 설계 입력)**.

## 1. 설계 입력 = `t3r_n10_ctq5_results.json` (READ-ONLY)

sha256(16) `236243d4cfaa58ae` / 1,437,511 B / verdict `COLLISION_ASSET_ADMITS_TILTED_BITE`.
**중요**: 이 수치들은 시각 메시가 아니라 **본 probe가 실제로 로드하는 동결 attempt3 충돌
USD**에서 측정됐다 (D428 #29).

| 항목 | 값 | 출처 |
|---|---|---|
| 기움 θ | **29.0°** | n10 band [15,29] 내 최대 물림 |
| 기움 월드 방위 ψ | **317.5142°** = `atan2(y,x)` of `seed0_S1` | D432 — 선택지가 아니라 기구학이 강제 |
| 공구 프레임 방위 φ* | **0.0°** | 41st 사다리 θ=29 행 |
| 하강 δ (TCP − cap, 축방향) | **−1.0997957078144082 mm** | n10 충돌 자산, q5 88.31~20.1° 전 구간 불변 |
| 양수 물림 q5 창 | **[14.70, 25.80]°** | n10 충돌 자산, 0.1° 해상도 |
| 닫힘 종점 | **24.5°** (물림 **+12.0411 mm**) | 상단 절벽 25.80에서 1.30° 안쪽, env 래치 바닥 22.918에서 1.58° 위 |

★ 종점을 **최댓값(25.80°)으로 잡지 않는 이유**: 물림이 상단에서 **계단**으로 끊긴다
(q5 26.0° → **−7.0669 mm**, 25.8° → **+12.1611 mm**). 최댓값은 절벽 위라 0.1° 오버슈트에
전부 잃는다. 24.5°는 물림 0.12 mm만 포기하고 1.30°의 여유를 산다.

★ **닫힘 바닥 23.0°는 물리 한계가 아니라 계측 한계**다: `roarm_stack_env.py:1185-1186`이
`q5 < grasp_gripper_thresh(0.4 rad = 22.918°)`를 "그리퍼 열림"으로 보고 `_grasped`를 지운다.
그 아래로 닫으면 조가 물리적으로 물고 있어도 D-2 marker가 False로 읽힌다.

## 2. 스크립트 = `sim_scripts/p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py`

동결 `p9`(sha `99c99c65da75d5b7`)의 **신규 파일 파생**. p9 원본 무수정. 변경은 **T-1~T-5 5개**뿐
(모듈 docstring에 사전등록):

- **T-1** 목표 공구축이 수직이 아니라 `d = (sinθcosψ, sinθsinψ, −cosθ)`. DLS 과제 오차의 축 2행이
  `axis·u`, `axis·v`를 죽인다. **θ=0, ψ=0에서 p9와 문자 그대로 동일** → 게이트 T3T-a.
- **T-2** ψ는 자유 파라미터가 아니다 (D432). 기본값 = 스폰 방위 자동.
- **T-3** approach/descend 목표가 `d` 방향으로 이동. **LIFT는 월드 수직(+z) 유지** — 이 phase의
  목적이 "중력에 맞서 물체가 따라오는가"이고 중력은 기울지 않았기 때문.
- **T-4** 하강 깊이·닫힘 종점을 상수로 옮겨 적지 않고 n10 산출물에서 **재확인**(게이트 T3T-b).
- **T-5** 손목 롤을 0으로 고정하지 않고 **닫힌 형식으로 푼다**(D432 N9f, 잔차 0). 롤은 공구축을
  중심으로 돌므로 TCP도 축도 안 움직이고, **원통이 공구 프레임에서 어느 방위로 기울어 보이는가**
  = D431의 φ*만 정한다.

나머지(Isaac 하네스, USD 가드 D-3/D-6, 스테이지 감사, marker 몽키패치 D-2, 물리 게이트 전체,
D341 Rerun 계약)는 **verbatim 승계**.

## 3. 실행 전 게이트 (전부 Isaac 기동 **전**, 실패 시 `PREFLIGHT_FAIL`로 즉시 종료)

| 게이트 | 내용 | 사전 측정값 |
|---|---|---|
| **T3T-a** | θ=0에서 p9 재현 (tilt·task_error 차이 0) | **0.0 / 0.0** (실행 전 확인) |
| **T3T-b** | n10 산출물과 (θ, φ*, δ, 종점, 창) 일치 + 하강 깊이가 전 스윕에서 불변 | — |
| **T3T-c** | 손목 롤이 φ*를 실제로 만든다 (잔차 < 1e-6°) | roll **+90.000°**, 잔차 **0.0** |

계획 IK 사전 확인(실행 전, Isaac 없이): approach/descend/lift **3/3 ok**,
위치오차 **(0.013, 0.023, 0.012) mm**, 축오차 **(0.200, 0.370, 0.200)°**,
`q_descend = [−42.49, 29.56, 117.00, 4.81, 90.00, 88.31]°`.

⚠️ 손목 롤 **+90.00°는 v6-clip 경계 위**(`roarm_kinematics.py:34`). URDF 한계 ±180°
(`roarm_m3.urdf:221`) 안이므로 안전 문제가 아니라 **분포 이탈 경고**다 (D432 ④).

## 4. 검정 질문 (실패 가능, 세션 진행 규칙 충족)

기움이 승인되고 물림이 기하적으로 양수이며 자세가 도달 가능할 때,
**(a)** descend가 통과하는가 (D424에서는 여기서 멈췄다),
**(b)** 닫힘 스윕에서 marker가 뜨는가,
**(c)** LIFT에서 물체가 **≥ 6 mm** 따라오는가.

- (a) 실패 → `DESCEND_FAIL`: 기움으로도 기하가 안 열린다 = n10의 정적 판정과 물리의 불일치.
- (b) 성공 + (c) 실패 → 조는 닿았으나 **force closure 없음** — D431이 미증명으로 남긴 바로 그것.
- (a)(b)(c) 전부 성공 → `GRASP_PASS`. **sim 결과이지 실물 주장 아님**(마찰 0.40/0.30 미실측).

## 5. tuple (정확히 이대로 1회)

```
cd /home/cgxr/Documents/Robotics/RoArm_Project
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py \
  --tag t3t_grasp \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3t_grasp_stdout.log \
  2> claudedocs/runtime_logs/grasp_track/g0b_d420/t3t_grasp_stderr.log
```

나머지 인자는 전부 스크립트 기본값(§1 표 = 기본값). 판정 권위 = **stdout의
`G0B_T3T_GRASP_VERDICT=` 라인 + `t3t_grasp_results.json`**. **exit code는 판정 채널 아님**
(D424 ③ 유지).

## 6. 주장하지 않는 것

- 실물 파지 주장 아님. `g0a_pass = false` 불변, T4~T7 대기, 프로포절 일정에 실물 파지 미포함.
- 마찰 μs 0.40 / μd 0.30 = **사전등록 가정, 미측정**(`t3_mass_friction_contract`).
- D427·D429·D430·D431·D432 불변 — 재실행 0, 재판정 0.
- 적대 재검증 미이행 상태 유지(D423 / D428 #25) — 40th·41st·42nd·43rd 네 판정 모두.

---

## 부록 A — `t3t_grasp2` leg 사전등록 (leg 1 결과 수령 후 발행, 실행 **전** 작성)

### A1. leg 1 결과 요약 (본 leg의 유일한 설계 입력)

`G0B_T3T_GRASP_VERDICT=LIFT_FAIL`, events_done 5/5, early_kill 0, exit 0, stderr 100 B.

- ✅ **approach 44wp 완주 · descend wp006 도달**(잔차 **1.993 mm**, 게이트 3 mm).
  D424/attempt1은 **같은 wp006에서 3.917 mm로 포화 정지**했다 ⇒ **기움이 D424가 특정한
  기하 위반을 물리 층에서 제거했다.**
- ✅ 닫힘 88.31 → **24.50°** 전 각도 `reached`, **`gripper_stalled=false` 전건**.
- ❌ LIFT `object_follow_delta_m = **−0.000373 m**`(게이트 +0.006) ⇒ 물체가 안 따라옴.
- 물체 반응은 **q5=60°에서 단 1회**(drift 0.759 mm, tilt 1.510°) 후 정지. 이후 drift ≤ 5.8 µm.
- `posewrite_calls=0` ⇒ 숨은 운동학적 핀 없음. 결과는 정직하다.

### A2. 원인 (leg 1 산출물 + 동결 n10 재질의로 특정, 신규 실행 0)

실행된 하강 깊이 **TCP − cap = −1.0997957078144082 mm**에서, 이동 조가 원통 벽에
**처음 간섭하는 각도 = q5 20.6°**(간섭 +0.0388 mm; 20.7°에서 정확히 0.0000).
**닫힘은 24.50°에서 멈췄다 ⇒ 접촉까지 3.9° 부족.**
`gripper_stalled` 전건 false가 이를 독립 확증한다 — 조는 저항을 만난 적이 없다.

⇒ **leg 1은 force closure를 검정하지 못했다. 애초에 닿지 않았다.**
D431 §5-1이 경고한 "bite > 0은 필요조건일 뿐"이 물리로 실증된 것이다.

### A3. 신규 델타 (leg 1 대비 2건, 둘 다 사전등록)

- **T-6 `cfg.grasp_gripper_thresh` 0.4 rad → 0.20 rad.** 이유: env의 이 상수는 env 자신의
  **cube 시대 규약(LARGE = CLOSED)** 하에서 저작됐다(`roarm_stack_env.py:242,380,1185-1186`).
  동결 grasp-track 규약은 **LARGE = OPEN**(D-1)이므로, 이 상수는 "조가 드디어 물체에 닿는
  각도에서 래치를 지우는 바닥"으로 작동한다 — `_verdict`가 `latch.grasped_seen`를 요구하므로
  **물리적으로 물고 있어도 `LATCH_FAIL`이 난다.** 이는 물리 실패가 아니라 **계측 인공물**이다.
  0.20 rad(11.46°)은 본 leg의 모든 지령각 아래이며 부호 규약을 바꾸지 않는다.
  ⚠️ **env 기본값 파일은 수정하지 않는다** — probe가 자기 `cfg` 인스턴스에만 설정한다
  (`cfg.reward_phase` / `cfg.curriculum_*` / `cfg.sponge.spawn`과 동일 계층).
- **T-7 닫힘 종점 24.50° → 19.50°** (실행 깊이 기준 지령 간섭 **+2.6260 mm**).
  스윕이 접촉 각도 20.6°를 통과하도록 20.5·20.0을 포함한다.

`--min_close_deg`는 T-6에 맞춰 12.0으로 내린다(가드 자체는 유지).

### A4. 검정 질문 (실패 가능)

조가 **실제로 닿은 상태**에서, 이 "고정 조가 테두리에 얹히고 이동 조가 벽을 후크하는"
한쪽 물림이 **24.83 g 원통을 중력에 맞서 6 mm 들어올리는가?**

- `GRASP_PASS` → sim에서 기운 파지가 성립. **실물 주장 아님**(마찰 미실측).
- `LIFT_FAIL` (닿았는데 안 들림) → **force closure 부재가 실증**된다. D431이 미증명으로
  남긴 바로 그 명제가 음의 방향으로 결정된다 ⇒ 다음은 파지 **자세**가 아니라 **접촉 형태**
  (양측 벽 파지 필요)가 문제라는 뜻이고, 그건 자산/과제 사양 층의 결정이다.
- `LATCH_FAIL` + drift 게이트 발동 → 닫힘이 물체를 밀어냈다(push-grasping) ⇒ 접근 깊이·각도
  재설계.

### A5. tuple (정확히 이대로 1회)

```
cd /home/cgxr/Documents/Robotics/RoArm_Project
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py \
  --tag t3t_grasp2 --min_close_deg 12.0 --grasp_latch_thresh_rad 0.20 \
  --close_deg 88.3096 60 45 41.40 35 30 25 22 21 20.5 20.0 19.5 \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3t_grasp2_stdout.log \
  2> claudedocs/runtime_logs/grasp_track/g0b_d420/t3t_grasp2_stderr.log
```

leg 1 산출물 8종은 **동결** — 본 leg는 별도 tag로만.

### A6. 자진 정정 — A5 tuple 1차 실행이 **자기 게이트에 막혔다** (실행 기록, 산출물 0)

1차 실행 = `G0B_T3T_GRASP_VERDICT=PREFLIGHT_FAIL a=True b=False c=True`, **Isaac 기동 전 종료,
산출물 0건**(로그 2건만). 원인: leg 1에 내가 넣은 T3T-b 하위 조건
**"하강 깊이가 전 닫힘 스윕에서 불변"** 이 발동했다 — q5 20.5°에서 깊이 −0.8253 vs 88.31°의
−1.0998 ⇒ 차 **0.2745 mm**.

★ **게이트가 옳았다.** leg 2는 바로 그 간섭을 **의도적으로 지령**하는 leg인데, 1차 tuple은
그 의도를 어디에도 선언하지 않았다. 게이트는 "종류가 바뀐 변경이 무선언으로 지나가는 것"을
정확히 막았다.

**추가 델타 T-8**: `--allow_closing_interference` 플래그를 신설한다. 세울 때만 그 하위 조건이
**하드 실패에서 기록 항목으로** 바뀌고, 닫힘 각도별 **지령 간섭량**이 `results.json`
(`commanded_interference_mm_by_close_angle` / `max_commanded_interference_mm` /
`close_angles_that_command_contact`)에 남는다. 기본값 False = leg 1 동작 그대로.

**정정된 tuple (이것이 실제 실행분)**:

```
cd /home/cgxr/Documents/Robotics/RoArm_Project
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py \
  --tag t3t_grasp2 --min_close_deg 12.0 --grasp_latch_thresh_rad 0.20 \
  --allow_closing_interference \
  --close_deg 88.3096 60 45 41.40 35 30 25 22 21 20.5 20.0 19.5 \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3t_grasp2_stdout.log \
  2> claudedocs/runtime_logs/grasp_track/g0b_d420/t3t_grasp2_stderr.log
```

⚠️ leg 1을 생산한 소스 리비전은 편집 **전에** `t3t_grasp_script.py.txt`
(sha `6861c35f94ed6427` / 104,700 B)로 동결했다. T-6~T-8 기본값은 전부 leg 1 동작과 동일하므로
leg 1은 현재 리비전으로도 그대로 재현된다.

---

## 부록 B — `t3t_grasp3` leg 사전등록 (leg 2 결과 수령 후 발행, 실행 **전** 작성)

### B1. leg 2 결과 (본 leg의 유일한 설계 입력)

`LIFT_FAIL`. 닫힘 88.31 → **19.5°** 전 각도 `reached`, **전 각도 `gripper_stalled=false`**,
20.5/20.0/19.5에서 물체 변위 **0.0004 / 0.0003 / 0.0004 mm**, `lift_follow −0.356 mm`.
HOLD·LIFT 실측 q5 = **19.532 / 19.527°**(err 0.03°) — 지령각에 실제로 도달했다.

★★ **정적 모델과 물리가 충돌한다.** 산출물 재계산:
- 지령 하강 δ = **−1.0998 mm**, **실제 도달 δ = +0.6885 mm**
  (`cap`=(0.213863,−0.195729,0.050), 실측 TCP=(0.213115,−0.194919,0.050178),
   `d`=(0.35752,−0.327444,−0.87462)) ⇒ **컨트롤러 잔차 1.788 mm가 전부 축방향**.
- 그 δ에서도 q5 19.53°의 정적 예측 간섭은 **+0.838 mm**인데 **접촉이 없다.**
⇒ **정적 볼록껍질 모델이 PhysX 접촉보다 최소 0.84 mm 낙관적이다.**
원인 후보(전부 [가설], 미검정): scipy 껍질 vs PhysX cook 차이 / **원통 프리미티브의 다면체
근사**(8분할이면 내접 오차 1.10 mm) / 도구축 각오차 0.37°의 레버 효과 / 닫힘 `reached` 밴드 0.75°.

### B2. 이번 leg가 하는 일

**접촉 각도를 추정하지 말고 물리에서 직접 측정한다.** 닫힘 스윕을 17.0°까지 연장하고
각 각도의 `gripper_stalled` / `max_object_drift_m`를 읽는다. **처음 stall이 뜨거나 물체가
움직이는 각도 = 실측 접촉 각도**이고, 그것이 정적 모델의 편향을 정량화한다.

- 종점 17.0°에서 접촉+파지 → `GRASP_PASS` 가능(sim 한정).
- 접촉은 생겼는데 안 들림 → **force closure 부재 실증**(D431 미증명 명제가 음으로 결정).
- drift 게이트(6 mm) 조기 종료 → 닫힘이 물체를 밀어냄(push-grasping) = 그것도 결과다.

### B3. 신규 델타 1건

**T-9 닫힘 종점 19.50° → 17.00°**, 스윕에 19.0·18.5·18.0·17.5 추가. 그 외 전부 leg 2와 동일
(T-6 래치 0.20 rad, T-8 간섭 선언 유지). 게이트 완화 **0건** — drift/speed/tilt/upright 전부 불변.

### B4. tuple (정확히 이대로 1회)

```
cd /home/cgxr/Documents/Robotics/RoArm_Project
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py \
  --tag t3t_grasp3 --min_close_deg 12.0 --grasp_latch_thresh_rad 0.20 \
  --allow_closing_interference \
  --close_deg 88.3096 60 45 41.40 35 30 25 22 21 20.5 20.0 19.5 19.0 18.5 18.0 17.5 17.0 \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3t_grasp3_stdout.log \
  2> claudedocs/runtime_logs/grasp_track/g0b_d420/t3t_grasp3_stderr.log
```
