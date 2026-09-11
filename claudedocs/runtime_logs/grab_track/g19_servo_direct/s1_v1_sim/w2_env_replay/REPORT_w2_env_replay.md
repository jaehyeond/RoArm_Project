# REPORT_w2_env_replay — 실물 환경 치수 Isaac Lab 씬 + 09-07 실물 5회 사이클 관절 명령 재생 (W2)

- 작성: 2026-09-09 (Claude, Orca worker task_9aa121c6b832) · 산출 폴더 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w2_env_replay/`
- 스크립트: `sim_isaaclab_s1_env_replay.py`(저장소 루트, 신규·미커밋) · 게이트 후처리 `w2_env_replay/w2_gates.py`
- 로봇 USD: `local_assets/roarm_m3/usd_s1/roarm_m3_s1.usd` (**v0 형상**, sha16 `bef6efeb42a241fe`). W1 의 `usd_s1_v1` 이 나오면 `--usd` 만 바꿔 재실행.
- 궤적: `s1_v1_real/manual_20260907_164725.jsonl` (sha16 `284e4d5776da9e39`, 180 이벤트 = goto 123 · door 31 · torque 16 · scoop_done 5 · place_done 5, 실물 소요 464.5 s)

## 0. 한 줄 판정

**G1~G4 전부 PASS.** 펠릿 없이 5 사이클 재생에서 그랩(문·고정 보울)·link5 는 상자(벽 4·바닥·받침)와 한 번도 접촉하지 않았고(표면 표본점 최소거리 = 문 립 ↔ 상자 −y 안쪽 벽, 아래 표), 시뮬 립 위치는 실물 FK 립과 중앙값 약 5 mm 로 맞았으며, 영상·키프레임·놓기 프레임을 냈다. 단, USD 에 **가짜 1 kg(hand_tcp)** 이 실려 있어 sim 어깨 처짐이 실물 처짐과 비슷한 크기로 나타나는 것은 우연이다(§6 발견 ①). 본 시행은 v0 USD 이며 v1 USD 재실행은 코디네이터 지시 대기.

## 1. 무엇을 / 왜

실물 09-07 `cycle 5`(5/5 성공, 93 s/회)에서 로봇이 받은 **관절 명령(cmd)** 을 그대로 Isaac Lab 의 로봇에 주고, 실물과 같은 치수의 상자·받침·펠릿면 높이·놓기 자리를 씬에 넣어
(a) 팔·그랩이 상자와 어디서 얼마나 가까운지(간섭 0 인지), (b) 시뮬 로봇의 립(보울 바깥 끝) 위치가 실물 로그의 FK 립과 맞는지(URDF/USD 기하·관절 규약 일치 확인), (c) 사람이 볼 영상을 얻는다.
펠릿 물리는 넣지 않았다(슬래브는 높이 표시용).

## 2. 규약 (읽은 원문 그대로)

- **SDK deg == URDF rad 부호·오프셋 없음** — `hw_s1_scoop_probe.chain` 이 `np.radians(q5)` 를 `roarm_kinematics._CHAIN` 에 그대로 넣는다. sim 관절 목표 = `radians(cmd)`. 손목 |q| ≤ 90 클램프(cmd 에 적용, 실제 cmd 는 전부 범위 안).
- **립** = link5 프레임 (8.1, 0, 166.6) mm (`hw_s1_scoop_probe.LIP_L5`). jsonl `lip` = `hw_s1_manual.lip_world(read)` = FK(read[:5]) 을 어깨축 기준(z − 0.122)으로 낸 값. 세계 z = lip_z + 0.122 + 0.38.
  스크립트 자체 검사: 180 이벤트 전부 FK(read) 재계산 vs jsonl lip 차 **최대 0.0017 mm** → 규약·체인·립 상수 일치.
- **문** = read[5] servo_deg → `radians` 직접(0 = 닫힘, 30 = 열림). door 이벤트 목표 = read(실물 정착값, 예 29.6/3.3).
- 시간: 실물 타이밍 재현 아님(실물 464.5 s, sim 396 s 는 우연히 비슷할 뿐). 관절 블렌드 60°/s(코사인, 세그먼트당 0.5~2.5 s), 정착 = 블렌드 후 ≥ 0.5 s + 전 관절 |속도| < 0.5°/s(상한 2 s). ⚠️ 속도 기준은 실효가 없었다(§6 ⑦) — 사실상 2.0 s 정착, run1(고정 0.5 s)과 립 위치 차 최대 0.041 mm.
- 액추에이터: 팔 `ImplicitActuatorCfg(stiffness 800, damping 40, effort_limit_sim 8.0)` = D478 계열 **비물리 데모값** · 문 (300, 30, **1.96**) = ST3215-HS 실물 상한. PhysX 읽기값으로 확인(`probe_gains/replay_result.json` `physx_dof_gains`).

## 3. 씬 (바닥 = z 0, 로봇 root = z 0.38 = 베이스판 윗면)

| 항목 | 값 (m) | 출처 |
|---|---|---|
| 베이스판 윗면 / 어깨축 | 0.38 / 0.502 (= +0.0701+0.05196, `SHOULDER_ABOVE_PLATE`) | 실측 09-07 |
| 펠릿 상자 안쪽 | x 0.31 × y 0.22, 중심 (0.35, 0), 윗단 0.385, 바닥(받침 윗면) 0.16 | 실측 |
| 상자 벽·바닥 두께 | **0.003 (가정, 실측 아님)** → 외곽 0.316 × 0.226 | 가정 |
| 받침 | 0 ~ 0.16, footprint = 상자 외곽(**가정**) | 가정 |
| 펠릿면 | 0.26 — 반투명 슬래브, **충돌 없음**(표시만) | 실측 |
| 놓기 자리(립) | (0.01, 0.34, 0.25) — 반투명 표식, 충돌 없음 | 실측 |
| 로봇 받침대 | 0.16 × 0.16 × 0.38 시각 전용 | 표시 |
| 카메라 | 측면 (0.95, −0.72, 0.78)→(0.28, 0.02, 0.33) focal 30 · 위 nadir (0.22, 0.10, 1.62)→z 0.35 focal 26, 640×400 각각 | D474: 대상 거리 1.09 / 1.27 m ≥ 0.85 |

충돌체 = 벽 4 + 상자 바닥 + 받침(고정 강체, `CuboidCfg` collision on). 상자 윗단 테두리·모서리 기둥(6 mm, 불투명)은 **시각 전용**(벽이 반투명이라 윤곽이 안 보여 추가, 안쪽으로 6 mm 튀어나오지만 충돌·거리 게이트에 없음).

## 5. 수치 (최종 실행 = 폴더 루트, `gates_w2_env_replay.json` 에서 생성)

| 게이트 | 기준 | 결과 | 판정 |
|---|---|---|---|
| G1 간섭 | door·grab_fixed·link5 vs 상자 최소거리 > 0, PhysX 접촉력 0 | 최소 **25.695 mm** = gripper_link ↔ `wall_yn`(−y 벽 안쪽 면), cycle 4 `scoop_plunge` t=252.5 s frame 2525, 점 (0.362, -0.084, 0.265); grab_fixed ≥ 50.132 · link5 ≥ 51.088 (하한); 접촉력 최대 0.0 / 0.0 / 0.0 N | **PASS** |
| G2 립 sim vs jsonl lip | goto 정착 123건 거리 중앙값 ≤ 10 mm | 중앙값 **4.86** · 평균 5.59 · p90 11.45 · **최대 16.21** mm (최대 = cycle 6 `home`); 축별 평균 편향(sim−real) x -0.31 y -0.2 z 3.05 mm | **PASS** |
| G3 완주 | scoop_done 5·place_done 5·NaN 0 | scoop_done 5 · place_done 5 · 비유한값 0 · sim 396.18 s · 프레임 3962; 관절 목표 대비 sim 읽기 편차 \|중앙값\| [base, 어깨, 팔꿈치, 손목P, 손목R] = [0.0, 1.356, 0.174, 0.012, 0.0]°, 최대 [0.001, 1.549, 5.072, 0.124, 0.0]° (최대 = home 팔꿈치), 문 편차 최대 0.002° | **PASS** |
| G4 매체 | mp4 + 키프레임 strip + 놓기 프레임 + 빈 프레임 없음 | `replay.mp4` 8.6 MB (3962 fr @ 10 fps) · `keyframe_strip_cycle1.png` 14장 · `place_frame.png`(frame 547, t 54.725 s, 립 sim [0.0081, 0.34655, 0.2517]) · `place_target_frame.png`; 빈 프레임 검사 61장 화소 std 최소 측면 51.3 / 위 42.9 | **PASS** |

### 5.1 G2 분해 (mm, goto 123건)

| 항목 | 중앙값 | 평균 | p90 | 최대 | 뜻 |
|---|---|---|---|---|---|
| sim 립 − jsonl lip (게이트) | 4.86 | 5.59 | 11.45 | 16.21 | |
| sim 립 − FK(cmd): sim 추종(처짐)+기하 | 8.43 | 6.12 | 9.83 | 34.84 | |
| FK(read) − FK(cmd): 실물 서보 편차, 시뮬 무관 | 8.61 | 9.89 | 20.83 | 21.94 | |

이름별 중앙값(sim−real, mm): above_move 3.46, above_up 4.16, home 16.21, p1 4.04, place_down 3.84, place_extend 3.2, place_retract 4.88, place_retract2 4.87, place_rot0 4.98, place_rot29 4.91, place_rot30 4.8, place_rot59 5.04, place_rot60 4.87, place_rot89 4.93, place_rot90 4.97, place_target 5.64, place_up 11.42, quit_up 4.11, scoop_5cm_above 4.1, scoop_lift8 12.41, scoop_plunge 8.28, scoop_surface 4.03, scoop_travel 11.89

놓기 자리(place_target, cycle 1) sim−real = [-3.14, 5.05, 1.5] mm; 놓기 개방 시 sim 립 = [0.0081, 0.34655, 0.2517] (사양 (0.01, 0.34, 0.25)).

정착: 블렌드 후 최소 0.5 s, 전 관절 |속도| < 0.5°/s 면 정착, 상한 2.0 s. 정착 시간 중앙값 2.0 s · 상한 도달 154/180 · 상한 시 잔류 속도 최대 7.858°/s.

### 5.2 부록 — 실물 어깨 부하 tS vs 중력 토크 모델

- n 123 · Pearson r(tS, M_model) = **-0.393** · tS 범위 [21.0, 285.0] · M_model(URDF 질량만) [0.142, 0.381] N·m · M_model_tcp(+hand_tcp 1.0 kg) [1.734, 3.824] N·m
- sim 처짐 역산 유효 강성 K_eff = |M_model_tcp| / 처짐각: 중앙값 **159.83** (범위 140.29~667.15) N·m/rad vs PhysX 읽기 강성 800.0 / 감쇠 40.0 / 상한 8.0
- sim 보고 applied_torque 가 상한 8.0 에 포화된 goto 비율 0.569 (비교 불가 → 모델 사용)
- 이름별 중앙값 (tS / M_model N·m / sim 어깨 처짐°): above_move 117/0.346/1.465; home 189/0.238/0.31; p1 161/0.142/0.478; place_rot90 197/0.142/0.479; place_target 141/0.381/1.411; place_up 277/0.346/1.465; scoop_lift8 277/0.371/1.405; scoop_plunge 29/0.379/1.356; scoop_surface 93/0.377/1.334; scoop_travel 277/0.345/1.442
- 그림 `appendix_loads_dmin.png` (좌: tS vs M_model 산점 · 우: G1 최소거리 시간열)

### 5.3 육안 검수 (직접 열어 본 것)

- `frames/f_02525.jpg` (G1 최소거리 순간, cycle 4 `scoop_plunge`): 측면 — 그랩이 상자 윗단 테두리 아래로 들어가 립이 펠릿 슬래브(갈색) 바로 위, 파란 문이 열려 −y(카메라 쪽) 벽을 향함, 벽 안쪽 면과 눈에 띄는 여유. 위 nadir — 그랩이 상자 footprint 안쪽 −y 쪽 절반에 있고 벽 선과 겹치지 않음. 라벨 dmin 26 mm 와 일치.
- `keyframe_strip_cycle1.png` (14장): 1 p1 툴 세움(상자 밖) → 2 above_move 상자 위 → 3·4 하강(립이 테두리 안으로) → 5 문 30° → 6 plunge → 7 닫힘 3.3° → 8 lift → 9 travel(테두리 위로 빠져나옴) → 10 base +90° 회전(위 화면에서 팔이 왼쪽 = +y) → 11 extend → 12 place_target(하강) → 13 문 30° 개방 → 14 retract. 회전·놓기 구간 내내 상자와 무관한 위치(dmin 138~195 mm).
- `place_frame.png` (cycle 1 놓기 개방, frame 547): 팔이 +y 로 뻗고 문이 열린 채 립 z ≈ 0.25(바닥 25 cm). 놓기 표식(반투명 초록 8×8 cm)은 렌더에서 식별되지 않음(너무 옅음) — 위치 검증은 수치(립 sim (0.0081, 0.3466, 0.2517) vs 사양 (0.01, 0.34, 0.25), 차 ≤ 7 mm)로 한다.
- 빈 프레임(D474) 없음: 61장 표본 화소 std 최소 측면 51 / 위 43.

## 4. 절차 (실행 순서대로)

1. 입력 읽기: `hw_s1_scoop_probe.py`(FK·립·HOME·P1, 읽기만), `sim_scripts/roarm_kinematics.py`(`_CHAIN`), `hw_s1_manual.py`(goto/door 로깅 = `lip_world(read)`), jsonl 180줄, `sim_isaaclab_grasp_sphere_s1.py`(골격), URDF `roarm_m3_s1.urdf`(링크·관절·질량), Isaac Lab 2.3.0 소스에서 API 이름 확인(`body_link_pos_w`, `step(render)`, `activate_contact_sensors`, `effort_limit_sim`).
2. 저장소 루트에 같은 이름의 **미커밋 초안**(`sim_isaaclab_s1_env_replay.py`, 09-09 15:44, 이전 시도 추정·`smoke/` 에 중단된 기동 로그)이 있었다. 사양과 맞아 **재사용하고 아래를 고쳤다**: 카메라 위치/초점(D474 거리 유지·확대), 상자 색·윗단 테두리·기둥(시각), 프레임 스텝 로그 누락 버그(`is_frame` 조건), 정착을 고정 0.5 s → 속도 기준 안정 판정, PhysX 질량·게인 읽기 기록, `--tcp-mass`(부록 변형용, 기본 미사용). configclass 안에 상수를 두면 asset 으로 오인되는 오류 1회 → 모듈 상수로 이동.
3. `--schedule-only`(Isaac 없이): 180 세그먼트, FK 자체검사 최대 0.0017 mm.
4. 스모크(12 이벤트, `smoke/`): 기동·렌더·거리·접촉 센서 동작 확인, 27 s wall / 10 s sim.
5. **본 실행 1**(`run1_fixed_settle/`, 정착 고정 0.5 s): 5 사이클 완주, 4 게이트 PASS. 영상에서 반투명 벽 때문에 상자 윤곽이 안 보여 §2 의 시각 보강 후 재실행.
6. **프로브**(`probe_gains/`, 2 이벤트): PhysX 가 실제 쓰는 질량·강성/감쇠/상한 읽기 → 발견 ①.
7. **본 실행 2**(폴더 루트, 최종): 아래 수치. 각 실행 전 `nvidia-smi` 여유 ≥ 6 GB 확인(사용 4.5 GB/16 GB), Isaac 앱 동시 1개, `timeout -k 30 1500`, 결과 JSON 선기록 + `os._exit` 워치독, Replicator 미사용(카메라 annotator 만).
8. `w2_gates.py`: 산출 JSON·프레임을 읽어 G1~G4 판정, mp4(ffmpeg libx264 10 fps), 키프레임 strip, 놓기 프레임, 빈 프레임 검사, 부록. 판정은 스크립트가 낸 **결과**를 읽어서 한다(D476).
9. 육안 검수: strip·최소거리 프레임·놓기 프레임을 직접 열어 봄(§5 말미).

## 6. 발견·주의

① **USD `hand_tcp` 링크에 질량 1.0 kg** — URDF 의 `hand_tcp` 는 무질량 프레임 링크인데 Isaac 임포터가 기본 1.0 kg 을 부여(PhysX 읽기값 `body_mass_kg`: hand_tcp 1.0, 나머지 URDF 값과 일치, 이동부 합 1.16 kg 중 1.0 이 가짜). link5 +Z 115.4 mm 툴 끝에 1 kg 이 달려 어깨 중력 모멘트가 URDF 기준 0.35 N·m → 3.7 N·m 로 약 10배. D478 에서 "USD maxForce 1.9 → 처짐" 이라 8.0 으로 올린 원인도 이것일 가능성이 크다(1.9 N·m 는 실물 서보 상한과 같은 크기). **W1 의 v1 USD 에서 hand_tcp 질량을 0 근처로 두거나 fixed joint 병합을 권고.** 본 시행은 USD 를 건드리지 않았다(`--tcp-mass` 는 기본 미사용).
② sim 어깨 처짐(≈1.4°, 뻗은 자세)과 실물 read−cmd 처짐(1.1~1.5°, 무적재)이 비슷한 것은 ①의 가짜 질량 때문이며 물리적 일치가 아니다. G2 는 사양대로 sim vs jsonl lip 로 판정했고, 분해값(sim vs FK(cmd) / FK(read) vs FK(cmd))을 같이 적었다. 기하·관절 규약 일치의 근거는 **FK 자체검사 0.0017 mm** 와 **처짐이 작은 자세(P1·회전 구간)의 sim vs FK(cmd) 차 ≈ 1.4 mm** 다.
③ sim 이 보고하는 `applied_torque` 는 Isaac Lab 의 PD 계산값이라 뻗은 자세에서 상한 8.0 에 포화된 값 — 물리 토크가 아니다. 부록의 토크 비교는 URDF 질량 정역학 모델로 했다. 처짐으로 역산한 유효 강성은 명목 800 N·m/rad 보다 훨씬 작다(부록) — 원인 미확인(PhysX 드라이브 단위/솔버). 이번 판정에 쓰이지 않으므로 추적하지 않았다.
④ 실물 tS(어깨 부하)는 자세의 중력 모멘트와 상관이 없다(r < 0): 펠릿 적재·펠릿면 접촉(plunge 에서 29)·접힌 자세(197) 등 상태가 지배. 단위·부호 미상이라 정성 비교만 가능.
⑤ 마지막 HOME 은 회전 구간 직후 큰 이동이라 정착 판정 상한에 걸릴 수 있다(run1 에서 팔꿈치 5° 잔류). 최종 실행의 정착 통계는 수치 표.
⑥ 가정 2건(벽 3 mm, 받침 footprint) 은 결과에 민감하지 않다: 최소거리는 벽 안쪽 면 기준이고 받침은 어떤 자세에서도 50 mm 이상 떨어져 있다.

⑦ 정착 속도 기준은 실효가 없었다: `robot.data.joint_vel` 최대값이 위치가 전혀 안 변하는 구간에서도 7~8°/s 로 읽혀(예 place_rot0 정착 중 관절·문 위치 변화 0.000°) 154/180 세그먼트가 상한 2 s 에 걸렸다. 판독 원인(어느 관절인지, PhysX 속도 산출)은 추적하지 않았다. run1(고정 0.5 s)과 최종 run 의 정착 립 위치 차 최대 0.041 mm → 0.5 s 에 이미 정지 상태였고 결과에 영향 없음.
⑧ 마지막 HOME([0,0,90,0,0]) 에서 sim 팔꿈치가 목표보다 5.07° 큰 정상 상태 오프셋(run1·run2 동일, 과도 아님) → sim vs FK(cmd) 34.8 mm, G2 최대값 16.2 mm 의 원인. 접힌 자세에서 툴 끝 가짜 1 kg 이 팔꿈치 모멘트를 키우는 것과 ③의 낮은 유효 강성이 겹친 것으로 본다. 실물 HOME read 도 어깨 1.93°·팔꿈치 0.97° 편차.
⑨ 놓기 표식이 렌더에서 안 보인다(불투명 패드로 바꾸면 보일 것, 재실행 시 반영 권고). 놓기 위치 자체는 수치로 검증됨(§5.3).

## 7. 판정 (일상어) + 다음 승인 경계

- 실물 09-07 명령을 v0 형상 로봇에 그대로 넣으면 **상자 벽과 가장 가까운 순간은 문 립이 −y 벽 안쪽 면에 약 2.6 cm** 까지 다가가는 plunge/닫기 구간이고, 접촉·관통은 없다. 놓기 자리·회전 구간은 상자와 10 cm 이상 떨어진다.
- 시뮬 립과 실물 FK 립은 중앙값 5 mm, 최대 1.6 cm(적재로 실물이 더 처진 lift 구간·마지막 HOME) 안에서 맞는다 → USD/URDF 기하와 SDK↔URDF 관절 규약이 일치한다.
- 다음: (1) W1 `usd_s1_v1` 나오면 `--usd` 바꿔 같은 명령으로 재실행(코디네이터 지시), (2) v1 USD 에서 hand_tcp 1 kg 제거 권고, (3) 펠릿 물리(DEME/입자)는 별도 case.

## 8. 산출 파일 (전부 `w2_env_replay/` 아래)

| 파일 | 내용 |
|---|---|
| `gates_w2_env_replay.json` | G1~G4 판정·수치·부록, 입력 sha256 앞 16자리(jsonl `284e4d5776da9e39` · USD `bef6efeb42a241fe` · URDF `d5a3647a5ea6b759` · `hw_s1_scoop_probe.py` `70096b2f2c6cbe77` · STL 5종), 산출 JSON sha |
| `replay_result.json` | 실행 요약(씬 치수·가정·액추에이터·PhysX 질량/게인·전 구간 최소거리·접촉력·FK 자체검사) |
| `settle_records.json` | 세그먼트 154건 정착 기록: cmd/read/sim 관절, 립 sim/real/FK(cmd), 거리 3종, 최소거리, 토크, 정착 시간 |
| `replay_log.json` | 4 스텝마다(30 Hz) 관절·문·최소거리·접촉력·토크 시간열 |
| `replay.mp4` · `frames/f_00000~03961.jpg` | 측면+위 나란히 1280×400, 10 fps, 396 s |
| `keyframe_strip_cycle1.png` · `place_frame.png` · `place_target_frame.png` · `appendix_loads_dmin.png` | 키프레임 14장 · 놓기 개방 프레임 · 놓기 하강 프레임 · 부록 그림 |
| `w2_gates.py` | 게이트 후처리(재실행: `python3 w2_gates.py --out <폴더>`) |
| `run1_fixed_settle/` | 본 실행 1(정착 고정 0.5 s, 테두리 없음) 전체 산출 — 수치 동일 |
| `probe_gains/` | PhysX 질량·게인 프로브(2 이벤트) |
| `smoke/` | 12 이벤트 스모크 |
| `stdout.log` · `stderr.log` | 최종 실행 로그(stderr 0줄) |

재실행 명령(v1 USD 는 `--usd` 만 교체):
```
OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 1500 ~/miniconda3/envs/isaaclab/bin/python -u sim_isaaclab_s1_env_replay.py --headless --enable_cameras \
  --usd local_assets/roarm_m3/usd_s1/roarm_m3_s1.usd --log claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/manual_20260907_164725.jsonl \
  --out claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w2_env_replay --fps 10
python3 claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w2_env_replay/w2_gates.py --out claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w2_env_replay
```
