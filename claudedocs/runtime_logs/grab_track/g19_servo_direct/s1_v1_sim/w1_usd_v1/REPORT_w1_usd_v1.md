# REPORT_w1_usd_v1 — 실물 장착 형상 s1_v1 로 로봇+그랩 URDF/USD 재생성 + 시뮬 검증 (W1, 2026-09-09)

## 1. 무엇을 / 왜
- `local_assets/roarm_m3/usd_s1/` 의 기존 USD 는 **s1_v0**(조립 불가로 판명된 옛 설계) STL 로 만든 것이었다(`urdf/s1_meta.json` source_dir = s1_v0).
  로봇에 실제 달린 형상은 **s1_v1**(뺨 2.0 mm·고정부 구멍 3, D481) 이므로, v1 STL 로 URDF/USD 를 다시 만들고 v0 와 같은 구조인지, 질량·관절·문 접촉이 실물과 맞는지 검증했다.
- v0 산출물은 **무수정·무삭제**(compose 스크립트에 `--tag` 옵션을 추가해 이름을 분리). 검증 전후 v0 62개 파일 sha256 동일.

## 2. 절차 (실행 순서)
1. `compose_roarm_s1_urdf.py` 에 `--tag` 옵션 추가(argparse; `TAG = "s1_<tag>"` 로 출력 이름만 바뀜, 로직 무변경). meta 에 `base_urdf_sha16`(입력 벤더 URDF 해시, D470) 과 `wrist_pitch.firmware_clamp_deg: 90`(D481 ①) 추가.
   실행: `~/miniconda3/envs/isaaclab/bin/python compose_roarm_s1_urdf.py claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1 --tag v1`
   → `local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf`(링크 10·조인트 9) + `meshes/s1_v1_door.stl`·`s1_v1_fixed.stl` + `meshes/collision_s1_v1/`(문 34 + 고정 27 = 61 볼록 조각) + `urdf/s1_v1_meta.json`.
2. 콜라이더 방식: 지시문은 `convex_decomposition` 이었으나 v0 정본(D480 §5, `usd_s1/config.yaml`) 은 `convex_hull` 이고, compose 산출 collision 이 이미 조각별 볼록이라 조각=자기 hull 이 정확하다.
   **coordinator 에게 ask → "지시문 오기, convex_hull 로 하라(v0 동일)" 답변 → convex_hull 채택.** (VHACD 근사 잡음이 G4 접촉각 비교에 섞이지 않음)
   실행: `sim_urdf_to_usd.py local_assets/roarm_m3/urdf/roarm_m3_s1_v1.urdf local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd --collider convex_hull --headless` (8.8 s, rc 0)
   → `local_assets/roarm_m3/usd_s1_v1/{roarm_m3_s1_v1.usd, config.yaml, configuration/*_base.usd 2.32 MB}`. config: collider_type convex_hull · mimic false · merge_fixed_joints false.
3. G4 Isaac Lab 문 닫힘 프로브 `door_close_probe_isaaclab.py`(이 폴더): 팔 HOME·중력 ON·self-collision ON, 문 20° 열림에서 목표 0 rad, 4 s 정착. 문 액추에이터 PD 300/30·effort_limit_sim **1.96**(ST3215-HS), 팔 8.0(비물리 데모값, D478). 접촉 증거 = ContactSensor(gripper_link ↔ grab_fixed 필터). 대조군 = self-collision OFF(관절 하한 0 만 작용).
4. G6 근접 RTX 렌더 `sim_render_s1_closeup.py <usd> closeup/` — 자세 2(HOME/스쿱) × 개폐 2(0 / 0.511 rad) × 시점 → **10장**(스크립트 docstring 은 8 이라 적혀 있으나 스쿱 자세에 front 가 추가돼 실제 10장; 전부 저장).
5. 게이트 판정 `gates_w1_usd_v1.py` → `gates_w1_usd_v1.json` (읽은 입력 38개 path+sha16 포함, D470). 산출물을 직접 읽어 판정(D476).
- GPU 규약: 다른 워커(W2 replay) 의 Isaac 앱 종료를 기다린 뒤(약 35 분) 1개씩 순차 실행, 각 단계 전 여유 ≥ 6 GB 확인(`run_w1_isaac_steps.sh` 에 게이트 내장). ⚠️ 대기 루프의 `pgrep -f "envs/isaaclab/bin/python"` 이 자기 명령줄에 걸려 2회 오탐 타임아웃(실제 프로세스는 종료돼 있었음) — 다음엔 `pgrep -f "sim_[a-z_]*\.py"` 처럼 스크립트명으로 잡을 것.

## 3. 수치 + 경로
| 게이트 | 결과 | 값 | 근거 파일 |
|---|---|---|---|
| G1 meta source=s1_v1 · sha16 | **PASS** | source_dir tail `s1_v1`, door_ALL `32521db98db7e421`, fixed_ALL `2fea1c37b63ff36e` (meta = 파일 재계산 = 지시값) | `urdf/s1_v1_meta.json`, `gates_w1_usd_v1.json` G1 |
| G2 링크·조인트 이름 집합 = v0 | **PASS** | 링크 10·조인트 9 집합 동일. 문 관절 `link5_to_gripper_link` revolute, 축 0 0 1, 0~1.571 rad. USD 관절 6·바디 10(`grab_fixed`·`gripper_link` 포함) | G2 (USD 이름은 `door_close_selfcol_on.json` 의 joint_names/body_names) |
| G3 문+고정 질량 | **PASS** | URDF 18.26 + 18.47 = **36.73 g** vs design.json derived door_g+fixed_g = 36.73 g → **0.0 %**. ⚠️ 지시문의 "tool_mass_g 42.93(나사 제외)" 은 design.json 에서 **나사 6.2 g 포함값**(나사 제외 = 36.73). 42.93 기준이면 −14.4 %(범위 밖) — URDF 는 인쇄 부품만 모델링하므로 36.73 기준이 맞다 | G3 |
| G4 문 닫힘 정지각 | **PASS**(측정 성공·관통 0) — 단 **실물 대역 밖** | 시뮬 정지각 **0.0014°**(마지막 0.5 s 평균, min=max, 정착), 문↔고정부 접촉력 0.077 N(첫 접촉 t 0.94 s @ 0.002°), 드라이브 토크 −0.010 N·m. 대조군(self-collision OFF) 0.0004°·접촉력 0. 정지각에서 기하 최소 간격 0.002 mm·고정부 안 문 점 0·**관통 깊이 0.0 mm**. 실물 **2.5~3.5°** → 시뮬−실물 = **−2.5~−3.5°** | `door_close_selfcol_on.json`, `door_close_log_selfcol_on.json`, `door_close_selfcol_off.json`, G4 geometry_sweep |
| G5 손목 피치 | **PASS** | URDF `link3_to_link4` ±1.92 rad(±110°, 벤더값 유지) + meta `wrist_pitch.firmware_clamp_deg = 90` | `urdf/s1_v1_meta.json` |
| USD 실물 | **PASS** | `roarm_m3_s1_v1.usd` 1463 B(sha16 `21dc4a11571d65f7`) + base 2,319,323 B, asset_path = roarm_m3_s1_v1.urdf, collider convex_hull, mimic false | `usd_s1_v1/config.yaml` |
| G6 렌더 10장 | 저장·육안 완료 | 아래 §4 | `closeup/*.png`, `closeup/closeup_poses.json`(문 readback 0.0/0.511 정확) |

기하 스윕(문 각 vs 고정부 최소 간격, link5 mm; 표면 표본 30만+정점, KD-트리):
0° 0.000 (립 z 166.6 파팅면 접촉) · 0.5° 0.62 · 1° 1.25 · 2° 2.50 · 2.5° 3.12 · 3° 3.70 · 3.5° 3.72 · 5° 3.76. 관통 전 구간 0.
0~2.5° 의 최근접 쌍은 보울 바닥 쪽 파팅면(z≈123.4, 힌지에서 71 mm) — 립(114.85 mm) 보다 힌지에 가까워 간격이 먼저 닫힌다. 3° 이상은 문 뺨(z≈119.7)↔고정 보울 바닥의 고정 간격 3.7 mm 로 넘어간다.

해시(sha256 앞 16): `roarm_m3_s1_v1.urdf` fdca6b303c816a90 · `s1_v1_meta.json` 90e3d38fc8dab15a · `roarm_m3_s1_v1.usd` 21dc4a11571d65f7 · `*_base.usd` b53ec64a51164f2d · `compose_roarm_s1_urdf.py`(수정본) e590cd5de1eeeee1 · 벤더 `roarm_m3.urdf` 64dc8d082cbce9a1.

## 4. 렌더 육안 관찰 (`closeup/`, 1280×960, RTX annotator 1장씩)
- **문 열림 방향**: `home_open_mouth`·`scoop_open_mouth`·`scoop_open_front`·`home_open_side`·`scoop_open_side` 모두 파란 문(가동 반쪽 보울+포크)이 초록 고정 반쪽에서 **link5 +X 쪽으로 벌어져** 입이 열린다. 고정 반쪽은 스파인 쪽에 그대로. 설계(D480 "문은 link5 +X 로 열림") 와 일치.
- **립 접촉(닫힘)**: `home_closed_mouth`·`scoop_closed_mouth`·`home_closed_side`·`scoop_closed_side`·`scoop_closed_front` — 문·고정부 립이 맞닿아 보울이 닫힌 한 덩어리로 보이고, 틈이나 겹침선 없음. 기하 결과(간격 0, 관통 0) 와 부합.
- **관통**: 열림·닫힘 10장 어디에도 파란 문이 초록 고정부를 뚫고 나오는 곳 없음. 문 포크·암은 고정 판·스파인과 분리돼 보인다.
- ⚠️ 렌더 품질: 닫힌 보울 안쪽·원통 끝면·일부 문 뺨 면이 **검게** 나온다(`scoop_closed_front` 의 검은 원판 = 보울 끝 캡 면). D480 §5 에 기록된 미해결 RTX 검정 평면 현상이 v1 에서도 그대로이며, 기하 결함이 아니다(Isaac Lab Camera 센서 영상은 정상이라는 D480 판정 유지). 순정 서보 디스크 부위(`home_closed_side` 하단) 도 같은 현상.

## 5. 일상어 판정 + 다음 승인 경계
- **로봇+실물 그랩(s1_v1) USD 가 준비됐다.** 구조(링크·관절 이름) 는 v0 와 완전히 같아서 기존 스크립트(`sim_isaaclab_grasp_sphere_s1.py`, W2 replay 등) 에 `--usd local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd` 만 바꿔 끼우면 된다. 질량은 설계값과 0 % 차이, 손목 피치 펌웨어 클램프 90° 는 meta 에 기록됐다.
- **문 닫힘은 시뮬 0.00°, 실물 2.5~3.5°.** 시뮬에서는 설계대로 두 립이 정확히 파팅면(0°)에서 만나고 관절 하한(0) 도 거기라 접촉과 하한이 겹친다(관통 0). 실물이 약 3° 일찍 멈추는 것은 인쇄·조립 공차(또는 펠릿 물림) 때문이지 USD 결함이 아니다. 립 기준으로 환산하면 실물 닫힘 틈 ≈ 5~7 mm(114.85 mm × sin 2.5~3.5°), 보울 바닥 쪽은 ≈ 3~4 mm.
- **다음 승인 경계(코디네이터 결정)**: ① 환경 리플레이/제어에서 "닫힘" 을 0° 가 아니라 실측 2.5~3.5° 로 두는 오프셋 채택 여부, ② 실물 공차를 USD 에 반영(예: 립 두께 −0.3 mm 또는 문 하한 2.5°) 할지 — 둘 다 새 변수이므로 여기서는 구현하지 않았다(Variable Ladder). ③ RTX 검정 면은 D480 미해결 상태 유지.

## 6. 만진 파일
- 수정: `compose_roarm_s1_urdf.py`(`--tag`, meta 2키 추가; 기본 동작·v0 경로 불변).
- 신규: `local_assets/roarm_m3/urdf/{roarm_m3_s1_v1.urdf, s1_v1_meta.json, meshes/s1_v1_door.stl, meshes/s1_v1_fixed.stl, meshes/collision_s1_v1/(61)}`, `local_assets/roarm_m3/usd_s1_v1/`(USD 는 .gitignore), 이 폴더 `w1_usd_v1/`(스크립트 3·JSON 5·로그 8·렌더 10).
- 불변 확인: v0 산출 62개(`roarm_m3_s1.urdf`, `s1_meta.json`, `s1_door/fixed.stl`, `collision_s1/*`, 벤더 `roarm_m3.urdf`) sha256 전후 동일. 상태 원장·session·relay·hw_*·s1_v1 입력 폴더 무수정. 로봇 하드웨어 접근 0.

---

# 부록 W1b — v1 USD 의 가짜 링크 질량 제거 + 재검증 (2026-09-09, 산출 `fix_mass/`)

## 1. 무엇을 / 왜
- W2 가 v0 USD 에서 **`hand_tcp` 링크에 임포터 기본 질량 1.0 kg** 이 실려 어깨 중력 모멘트가 약 10배 부풀었다고 보고했다(`w2_env_replay/REPORT_w2_env_replay.md` §6 ①). v1 USD 도 같은 경로로 만들었으므로 확인하고 고쳤다.
- 근거(설치본 인용): IsaacLab 2.3.0 `sim/converters/urdf_converter_cfg.py:99-100` — `link_density: float = 0.0` "Default density in kg/m^3 for links whose inertial properties are missing in the URDF"; `urdf_converter.py:115-116` — `# default density used for links, use 0 to auto-compute` → `import_config.set_density(self.cfg.link_density)`. Isaac Sim 5.1 임포터 UI 툴팁 `isaacsim/exts/isaacsim.asset.importer.urdf/.../ui/UrdfOptionWidget.py:173` — "If a link doesn't have mass, use this density as backup, A density of 0.0 results in the physics engine automatically computing a default density". `hand_tcp` 는 URDF 에 inertial·visual·collision 이 모두 없는 빈 프레임 링크라 밀도로 계산할 형상도 없다 → 수정 전 USD 실측: `physics:mass` **미작성**, `physics:diagonalInertia` [1e-4]*3 만 작성(`PhysicsMassAPI` 적용), PhysX 읽기 질량 **1.0 kg**(`mass_static_pre.json` physx_bodies/usd_authored). "형상·질량 미지정 시 PhysX 기본 1.0 kg" 은 관측값에서의 추론이며 NVIDIA 문서 원문은 이번에 확인하지 않았다.
- 같은 이유로 루트 `world` 도 1.0 kg 이나, 모든 스크립트가 `fix_root_link=True` 로 루트를 고정하므로 관절 하중과 무관 → **미수정**(루트를 1e-4 kg 로 두면 base 비고정 사용 시 솔버 조건이 나빠질 수 있어 피함).

## 2. 절차
1. 백업: `urdf/roarm_m3_s1_v1.urdf.bak_pre_massfix`, `urdf/s1_v1_meta.json.bak_pre_massfix`, `usd_s1_v1.bak_pre_massfix/`(폴더 복사).
2. 수정 전 프로브 `fix_mass/mass_static_probe_isaaclab.py --tag pre`(백업 USD): PhysX `get_masses/get_inertias/get_coms` 표 + USD authored 속성 + HOME·스쿱 자세 정착 후 어깨(link1_to_link2)·팔꿈치(link2_to_link3) 중력 모멘트 정역학 재계산(Σ[(r_com−r_joint)×m·g]·axis, 축 = 자식 링크 z) + `applied_torque`.
3. `compose_roarm_s1_urdf.py` 수정(**`--tag` 경로에서만**): 자기닫힘 `<link name="X"/>` 중 `world` 를 뺀 링크(= `hand_tcp`)에 `<inertial>` 질량 **1e-4 kg**, 대각 관성 **1e-8 kg·m²** 주입. meta 에 `tiny_inertial_links` 기록. 벤더 `roarm_m3.urdf`·v0 산출(62파일 sha256) 무수정 재확인. 재생성 URDF 는 백업본과 **hand_tcp inertial 블록만** 다름(diff 확인).
4. USD 재변환(convex_hull, 8.7 s) → 같은 경로 `usd_s1_v1/`. 기하 레이어 `*_base.usd` sha 동일(b53ec64a51164f2d), 물리 레이어 `*_physics.usd` 6f2b31e46253283d → **8691734ead2d3e7f**(hand_tcp 질량 작성).
5. 수정 후 프로브 `--tag post`, G4 문 프로브(ON/OFF)·G6 렌더 10장을 `fix_mass/` 로 재실행, `gates_w1_usd_v1.py fix_mass`(결과 폴더 인자 추가) → G1~G6, `gates_fix_mass.py` → G7.
6. D478 확인: `sim_isaaclab_grasp_sphere_s1.py` 를 `fix_mass/grasp_sphere_s1_v1_arm196.py` 로 복사해 **팔 `effort_limit_sim` 8.0→1.96 한 줄만** 바꾸고 v1(수정 후) USD 로 1회 실행(`--headless --enable_cameras`, 카메라 270 프레임).
- Isaac 실행 6회 전부 동시 1개·여유 ≥ 6 GB 확인 후 실행(W3 DEME 는 Isaac 아님).

## 3. 수치 + 경로
**링크별 PhysX 질량표 (kg)** — `mass_static_{pre,post}.json` physx_bodies (관성 대각·COM 은 JSON):
| 링크 | URDF | pre(수정 전) | post(수정 후) |
|---|---|---|---|
| world(루트, 고정) | 없음 | 1.0 | 1.0 (미수정) |
| base_link | 0.25622 | 0.25622 | 0.25622 |
| link1 / link2 / link3 / link4 / link5 | 0.07292 / 0.07032 / 0.02161 / 0.00999 / 0.01539 | 동일 | 동일 |
| gripper_link(문) / grab_fixed | 0.01826 / 0.01846 | 동일 | 동일 |
| **hand_tcp** | 없음 → 1e-4 | **1.0** | **0.0001** |
| 합(world 제외) | 0.48317 | 1.48318 | 0.48328 |

**어깨·팔꿈치 정역학 중력 모멘트 (N·m, Isaac 측정각 기준)** — `gates_fix_mass.json` G7:
| 자세 | 관절 | pre PhysX 재계산 | post PhysX 재계산 | post URDF 정역학(순수 FK) | 편차 | 처짐 pre → post (°) |
|---|---|---|---|---|---|---|
| HOME | 어깨 link1_to_link2 | 0.5761 | **0.0407** | 0.0407 | 0.0 % | 0.299 → 0.011 |
| HOME | 팔꿈치 link2_to_link3 | 0.2352 | 0.0075 | 0.0075 | 0.0 % | 1.079 → 0.079 |
| 스쿱 | 어깨 | 2.6709 | **0.2326** | 0.2326 | 0.0 % | 1.531 → 0.111 |
| 스쿱 | 팔꿈치 | 1.3192 | 0.0963 | 0.0963 | 0.0 % | −0.082 → −0.002 |
hand_tcp 단독 기여(HOME 어깨): 0.530 → 0.00004 N·m. 가짜 1 kg 이 어깨 모멘트를 HOME 14×, 스쿱 11× 부풀렸다. `applied_torque`(Isaac Lab PD 계산값) 는 정역학과 맞지 않아(예: 스쿱 어깨 0.004 N·m) 참고로만 기록(W2 ③ 과 동일 관찰).

**G1~G6 재실행(재생성본)**: 전부 PASS — G4 문 정지각 0.0014°(접촉력 0.074 N, 관통 0), 대조군 OFF 0.0004°, 렌더 10장 재생성(`fix_mass/closeup/`, 기하 불변이라 §4 관찰 동일). **G7 PASS**(hand_tcp 1e-4 ≤ 1e-3 kg, 어깨 |편차| 0.0 % ≤ 5 %).

**D478 "팔 8.0 = 비물리 데모값" 확인** — `fix_mass/grasp_arm196/grasp_result.json` vs `s1_v0/isaaclab_grasp_sphere/`(D480, 팔 8.0 + 가짜 1 kg):
| 항목 | v1 수정 후 · 팔 **1.96** | v0 · 팔 8.0 · 가짜 1 kg |
|---|---|---|
| 파지 ok | **True** (구 z 끝 0.1701 m, 보울 xy 3.2 mm) | True (0.163 m, 3.2 mm) |
| 어깨 처짐 hold(down) / hold2(lift) | **0.110° / 0.165°** | 1.371° / 1.895° |
| 보울 z(down 끝) | 0.0289 (계획 0.0286) | 0.0237 (계획 0.0286, 처짐 4.9 mm) |
| 문 토크 최대 | 1.96 | 1.96 |
→ 가짜 질량을 빼면 **실물 라벨 1.96 N·m 로도 처짐 0.1~0.2°·파지 성공**. D478 에서 1.9 로 처졌던 것은 hand_tcp 1 kg 때문이었다고 보는 것이 맞다(단일 시행, 판정 아님·사실 보고).

해시(sha256:16): 재생성 `roarm_m3_s1_v1.urdf` 411ef28f78ead0f0 · `s1_v1_meta.json` 07995c667a548a8a · `compose_roarm_s1_urdf.py` 35d312a7392aa1c5 · `mass_static_pre/post.json` bf7888984695f052 / f799f50848ceeccb · `grasp_result.json` b253ffddea86e092. 입력 sha 전체는 `gates_fix_mass.json` inputs_sha16.

## 4. 판정 + 다음 승인 경계
- **v1 USD(`usd_s1_v1/`) 는 이제 URDF 질량만 싣는다**(툴 끝 가짜 1 kg 제거). W2 리플레이·구 파지 등 모든 v1 사용처는 재실행 시 어깨 처짐이 약 10배 줄어든다 — W2 §6 ②의 "sim 처짐 ≈ 실물 처짐" 은 우연이었음이 확정.
- 코디네이터 결정 사항: ① W2 리플레이를 수정 후 v1 USD 로 재실행할지, ② 팔 액추에이터 상한을 8.0(데모) 에서 실물 1.96 으로 내려 표준화할지(이번 1회 시행은 성공), ③ 루트 `world` 1 kg 도 제거할지(고정 루트라 현재는 무영향).
- 만진 파일: `compose_roarm_s1_urdf.py`(태그 경로 미소 관성 주입) · `w1_usd_v1/gates_w1_usd_v1.py`(결과 폴더 인자) · 재생성 `urdf/roarm_m3_s1_v1.urdf`·`s1_v1_meta.json`·`usd_s1_v1/`(백업 `.bak_pre_massfix`) · 신규 `fix_mass/`(스크립트 3·JSON 8·로그·렌더 10·프레임 270).
