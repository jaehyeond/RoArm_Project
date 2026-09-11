# REPORT_w2b_env_replay_v1 — 질량 수정된 v1 USD 로 실물 환경 재생 재실행 (W2b, 2026-09-10)

## 0. 한 줄 판정
**G1~G4 전부 PASS(v0 와 동일).** 가짜 hand_tcp 1 kg 을 뺀 v1 USD 에서는 시뮬 팔이 명령을 거의 그대로 따라가(립 sim−FK(cmd) 중앙값 **0.64 mm**, v0 8.43) 어깨 처짐이 1.36°→**0.10°** 로 줄었고, 그 결과 립 sim−실물(jsonl) 차는 4.86→**8.00 mm** 로 커졌다 — 남은 차이(중앙 8.6 · 최대 21.9 mm)는 **실물 서보가 부하 아래 명령보다 약 3° 처지는 것**(FK(read)−FK(cmd)) 이며 시뮬 기하 오차가 아니다. v0 의 4.86 mm 는 가짜 질량이 만든 우연(W2 §6 ②) 이었음이 수치로 확정됐다.

## 1. 무엇을 / 왜
- W1b 가 `usd_s1_v1` 의 hand_tcp 가짜 질량(1.0 kg → 1e-4 kg) 을 제거했다. W2 의 v0 실행은 이 가짜 질량 아래에서 나온 것이라 같은 명령·같은 씬으로 v1 USD 를 재생해 게이트를 다시 판정하고 v0 와 나란히 비교한다.
- 추가 지시 2건: ① 놓기 표식을 **불투명 패드**로 바꿔 렌더에 보이게(v0 는 반투명이라 식별 불가) ② 정착 판정을 관절 속도 대신 **위치 안정(두 번 연속 |Δq| < 0.3°)** 으로(W2 §6 ⑦: 속도 기준은 154/180 이 2 s 상한에 걸려 실효 없음).

## 2. 절차 (실행 순서)
1. `sim_isaaclab_s1_env_replay.py` 백업 `.bak_20260910_pre_w2b`(sha 6e6c87627f05e667) 후 수정(sha 08cc409e826c7415, diff 23줄):
   - 정착: 블렌드 후 최소 0.5 s(goto)/0.4 s(door), 이후 **0.4 s 간격**으로 전 관절 |Δq| 를 재고 **두 번 연속 < 0.3°** 면 정착, 상한 2.0 s — D481 ④ 실물 규약("0.4 s 두 번 연속 |Δ| < 0.3°") 을 그대로 옮김. `settle_records.json` 에 `settle_dq_last_deg`·`settle_n_stable` 기록.
   - 놓기 표식: 8×8 cm × **10 mm 불투명** 초록 패드, 윗면 = 놓기 립 z 0.25 (충돌 없음 그대로).
   - v1 USD 면 거리 표본 메시·URDF 를 v1 것(`s1_v1_door.stl`·`s1_v1_fixed.stl`·`roarm_m3_s1_v1.urdf`) 으로 자동 선택(`--urdf` 인자 추가). 그 외 씬·카메라·액추에이터(팔 800/40/**8.0 비물리 데모값**, 문 300/30/1.96)·블렌드 60°/s 는 v0 와 동일.
2. 실행(§8 명령에서 `--usd`·`--out` 만 교체): GPU 여유 12.2 GB·Isaac 앱 0 확인 →
   `OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 1500 ~/miniconda3/envs/isaaclab/bin/python -u sim_isaaclab_s1_env_replay.py --headless --enable_cameras --usd local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd --log claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/manual_20260907_164725.jsonl --out <이 폴더> --fps 10` → rc 0, stderr 0 B, ok, 약 6 분.
3. 게이트: `w2b_gates.py`(= `w2_gates.py` 사본, 정착 규약 문자열을 `replay_result.json` 에서 읽도록 2줄만 수정) `--out <이 폴더> --urdf roarm_m3_s1_v1.urdf` → `gates_w2_env_replay.json`, mp4, strip, 놓기 프레임. 비교표 `compare_v0_v1.py` → `compare_v0_v1.json`. v0 폴더(`w2_env_replay/` 루트) 는 읽기만 했다(mtime 09-09 16:51 유지).

## 3. 수치 + 경로 (v0 = `w2_env_replay/gates_w2_env_replay.json`, v1 = `v1/gates_w2_env_replay.json`)

| 항목 | v0 (hand_tcp **1.0 kg**, usd_s1) | v1 (hand_tcp **1e-4 kg**, usd_s1_v1) | 뜻 |
|---|---|---|---|
| 게이트 | G1~G4 PASS | G1~G4 **PASS** | |
| G1 상자 최소거리 gripper_link / grab_fixed / link5 (mm) | 25.695 / 50.132 / 51.088 | **25.625** / 50.199 / 50.796 | 둘 다 cycle 4 `scoop_plunge`, 문 립 ↔ −y 벽 안쪽(`wall_yn`), 점 (0.365, −0.084, 0.272); 접촉력 0 / 0 / 0 N |
| G2 립 sim − jsonl(FK(read)) 중앙 / p90 / 최대 (mm) | 4.86 / 11.45 / 16.21 | **8.00 / 19.99 / 20.96** | 게이트 ≤ 10 중앙값 PASS. 최대 = cycle 2 `scoop_travel`(실물 어깨 cmd 45.0 → read 47.99, tS 285) |
| 립 sim − FK(cmd) 중앙 / 최대 (mm) | 8.43 / 34.84 | **0.64 / 2.11** | sim 추종 오차 — 가짜 질량 제거로 13배 감소 |
| 립 FK(read) − FK(cmd) 중앙 / 최대 (mm) | 8.61 / 21.94 | 8.61 / 21.94 | 실물 서보 처짐의 립 환산(시뮬 무관) — v1 의 sim−real 과 거의 같음 |
| 축별 편향 sim−real 평균 x / y / z (mm) | −0.31 / −0.20 / 3.05 | 0.58 / 0.46 / **8.43** | v1 은 실물이 처진 만큼 sim 립이 위(+z) |
| 어깨 처짐 sim(|q_sim−q_cmd|) 중앙 / 최대 (°) | 1.356 / 1.549 | **0.097 / 0.111** | 뻗은 자세군 중앙 1.427 → 0.102 |
| 팔꿈치 편차 최대 (°) | 5.072 (마지막 HOME) | **0.284** | W2 §6 ⑧ 의 HOME 팔꿈치 5° 오프셋 = 가짜 질량 |
| 어깨 정역학 중력 모멘트, sim 이 실제 받는 값 M_model_tcp (N·m) | 1.734 ~ 3.824 | **0.133 ~ 0.362** | URDF 질량만의 모델 M_model 은 0.142~0.381(v0 URDF) / 0.133~0.361(v1 URDF) |
| 유효 강성 K_eff = M/처짐 (N·m/rad) 중앙 | 159.8 | 212.5 | 명목 800·PhysX 읽기 800/40/8.0 — 여전히 낮음(W2 ③ 미해결, 판정 미사용) |
| 실물 tS vs M_model 상관 r | −0.393 | −0.394 | 실물 부하는 자세 중력이 아니라 적재·접촉 지배(W2 ④ 동일) |
| 정착 | 속도 기준, 중앙 2.0 s, 상한 도달 154/180, 잔류 7.86°/s | **위치 안정**, 0.8(door)/1.2(goto) s, 상한 도달 **0**, 마지막 |Δq| 최대 **0.0046°** | |
| sim 시간 / 프레임 | 396.18 s / 3962 | 259.38 s / 2594 | |
| 놓기 개방(cycle 1 door30) sim 립 | (0.0081, 0.3466, 0.2517) | (0.0081, 0.3506, 0.2591) | 사양 (0.01, 0.34, 0.25): v1 은 +9 mm 위(실물은 처져서 사양 높이) |
| place_target sim−real (mm) | [−3.14, 5.05, 1.50] | [−3.14, 9.10, 8.86] | |
| G4 매체 | mp4 8.6 MB, strip 14, 빈 프레임 없음(std ≥ 42.9) | mp4 **6.4 MB**, strip 14, 빈 프레임 없음(측면 51.4 / 위 42.9) | |

이름별 sim−real 중앙값(mm, v0 → v1): scoop_plunge 8.28 → **1.82**(펠릿면·저부하 자세는 오히려 개선) · p1 4.04 → 5.31 · place_rot* ≈ 4.9 → 6.2 · above_move 3.46 → 10.14 · scoop_travel 11.89 → 20.51 · place_up 11.42 → 20.32 · scoop_lift8 12.41 → 20.00 · home 16.21 → 19.91. 실물 tS 가 큰 뻗은 자세(277~285)에서만 커진다 = 실물 처짐.

입력 sha256:16 (`gates_w2_env_replay.json` inputs): jsonl 284e4d5776da9e39 · USD 21dc4a11571d65f7(물리 레이어 8691734ead2d3e7f, W1b) · URDF v1 411ef28f78ead0f0 · `hw_s1_scoop_probe.py` 70096b2f2c6cbe77 · 메시 s1_v1_door 880547d19d611cac / s1_v1_fixed 2fea1c37b63ff36e / link5 1d63f374a78c1419. 산출 `replay_result.json` 430228b70c026507 · `settle_records.json` 0e845d5057744c3f · `replay.mp4` be20cb53b0d227ff · `compare_v0_v1.json` 5b114cfcd3b550e5.

## 4. 육안 검수 (직접 연 것)
- `place_frame.png`(cycle 1 door30, frame 363, t 36.3 s): **불투명 초록 패드가 측면·위 두 화면 모두에서 뚜렷**하다. 측면 — 팔이 +y 로 뻗어 열린 파란 문·초록 고정 보울이 패드 바로 위. 위 — 그랩이 패드 위에 겹쳐 있고 패드 중심(0.01, 0.34) 과 그랩 위치가 맞는다. v0 에서 "표식 식별 불가" 였던 문제 해소.
- `place_target_frame.png`(frame 348): 문 4.4° 로 닫힌 채 패드 위로 하강한 자세, 위 화면에서 그랩이 패드를 덮음.
- `frames/f_01659.jpg`(G1 최소거리 순간, cycle 4 `scoop_plunge`, dmin 26 mm): 측면 — 그랩이 상자 테두리 아래 펠릿면 위, 문이 −y 벽을 향해 열림, 벽과 눈에 보이는 여유. 위 — 그랩이 상자 안 −y 절반, 벽 선과 겹치지 않음. v0 f_02525 와 같은 구도.
- `keyframe_strip_cycle1.png`(14장): p1 → above_move → 하강 → 문 30° → plunge → 닫힘 3.3° → lift → travel → base +90° → extend → place_target → 문 30° → retract. 위 화면 전 장면에서 패드가 보인다. dmin 라벨 30~199 mm.

## 5. 판정 (일상어) + 다음 승인 경계
- **v1 USD 재생은 4 게이트 모두 통과하고, 시뮬 로봇은 이제 명령을 1 mm 이내로 따른다.** 상자와의 여유(약 26 mm)는 v0 와 같다.
- **시뮬 vs 실물 립 차이가 5 → 8 mm(최대 21 mm) 로 "커진" 것은 나빠진 게 아니다.** v0 는 툴 끝의 가짜 1 kg 이 시뮬 팔을 실물처럼 처지게 만들어 우연히 맞았던 것이고, 수정 후 남은 차이는 실물 서보가 부하 아래 명령보다 약 3° 처지는 양(FK(read)−FK(cmd) 중앙 8.6 mm) 과 일치한다. 즉 **시뮬 기하·관절 규약은 맞고, 실물 서보 컴플라이언스(그리고 펠릿 적재) 가 시뮬에 없다.**
- 다음 승인 경계(코디네이터 결정): ① 실물 처짐을 시뮬에 넣을지 — 방법은 (a) 팔 PD 강성을 실물 유효 강성으로 낮추기(현재 800 명목이나 K_eff 212 라 PhysX 드라이브 단위 문제부터 확인 필요, W2 ③), (b) cmd 대신 read 각도로 재생, (c) 실물 서보 처짐 모델(tS↔각) 별도 케이스. ② 팔 상한 8.0 을 실물 1.96 으로 표준화(W1b 파지 1회 성공). ③ 펠릿 물리는 W3 케이스.
- 만진 파일: `sim_isaaclab_s1_env_replay.py`(백업 `.bak_20260910_pre_w2b`) · 신규 `v1/`(REPORT·gates·compare·w2b_gates.py·replay 산출·mp4·프레임 2594). v0 폴더 루트·`w2_gates.py`·상태 원장·하드웨어 무접촉.

## 6. 산출 파일 (`w2_env_replay/v1/`)
| 파일 | 내용 |
|---|---|
| `gates_w2_env_replay.json` | G1~G4 + 부록, 입력 sha |
| `compare_v0_v1.json` · `compare_v0_v1.py` | v0 vs v1 비교표 원자료 |
| `replay_result.json` · `settle_records.json` · `replay_log.json` | 실행 요약 · 세그먼트 180건 정착 기록(위치 안정 값 포함) · 30 Hz 시간열 |
| `replay.mp4` · `frames/` | 측면+위 1280×400, 10 fps, 259 s, 2594장 |
| `keyframe_strip_cycle1.png` · `place_frame.png` · `place_target_frame.png` · `appendix_loads_dmin.png` | 키프레임·놓기·부록 그림 |
| `w2b_gates.py` | 게이트 후처리 사본(정착 규약 문자열만 동적) |
| `stdout.log` · `stderr.log`(0 B) | 실행 로그 |
