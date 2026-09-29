# post06 — 규약 A(rev34) 원자료를 위한 재생 좌표 변환 준비 (CPU 전용)

작성 2026-09-29 · 역할 replay-renderer · **이 폴더는 준비 산출물이다. Isaac/DEME/GPU 프로세스 0 · RunPod 접근 0 ·
설치 0 · 기존 파일 수정 0 · 원자료 쓰기 0**(원자료는 sha256 전후 대조로 무변경 확인).

## 0. 한 줄 요약

post05 재생기는 DEME(상자) 좌표를 **평행이동만** 해서 그린다. rev34(W25-A)는 상자를 로봇 앞에 **90° 돌려**
놓으므로(규약 A) 같은 스크립트로 그리면 로봇·공구가 상자 대비 90° 틀린다. post06 은 **표시 좌표 변환만**
고쳐 원자료가 스스로 기록한 `metadata_json.w25_frame`(= `R_robot_box`, `t_robot_m`)을 적용하고,
그 키가 없는 옛 원자료(W19 A)에서는 post05 와 **같은 식**을 그대로 탄다. CPU 자체검사 9/9 PASS.

## 1. 파일과 해시

| 파일 | sha256 | 비고 |
|---|---|---|
| (원본) `/home/cgxr/orca/workspaces/RoArm_Project/w25-render-cad/claudedocs/research/w25_render_cad_20260928/isaac_replay_w13_post05.py` | `1a1c48c00ffe7181703793a7e8bca0aeb6a2f5ee4fcf15956471930cb947b0de` | `logs/post05_script_sha256.txt` 가 가리킨 파일. **읽기만**, 이 sha 를 대조하고 시작했다 |
| `isaac_replay_w13_post06.py` | `926d67ed113b67f599356a7ec7ff0ea13eb35ae402cdf9654fc2335a0c34252e` | 위 파일의 **바이트 사본에서 출발**해 좌표 변환만 패치(1,755 → 1,889 줄) |
| `cad_pose_math.py` | `bbf935fc373ee8ba81deaf539f2f69af3efbd995432ac1c4a11b773968045247` | 원본(w25-render-cad)의 **바이트 사본**. post05 는 `HERE`(스크립트와 같은 폴더)에서 이 모듈을 import 한다 → 스크립트를 옮기면 같이 있어야 한다. 내용 변경 0 |
| `DIFF_post05_to_post06.patch` | — | `diff -u` 전문(10 hunk, 추가 182 줄 / 삭제 49 줄 — 헤더 줄 제외 실측) |
| `selftest_post06.py` / `selftest_post06.json` | — | numpy 전용 자체검사 스크립트와 결과 (§3) |
| `display_limits_post06.py` / `display_limits_post06.json` | — | 표시 한계(재투영 최대 오차·범위 밖 포즈 수)를 **렌더 없이** CPU 로 계산 (§3-5) |

다른 형제 모듈(`arm_link_bounds`·`close_diagnostics`·`camera_framing`·`raw_row_identity`·`w13_fk`·
`w13_kinematics`)은 post05 가 **절대경로로 핀**해 둔 동결 post04 `rev/src` 에서 그대로 온다(`:51-59`, 변경 0).

원자료 sha256 (실행 전 대조 = 실행 후 재대조 `sha256sum -c` **OK**, 읽기만 했다):

```
d2b86281649e058876fc744c1ad53337d6398d043998ef8905a90b6c0cffc036  .../w25-rev34-fullcycle/.../rev34/dryrun/paperbox_final_n67737/w13_cycle_seed460.npz
414633fbff8fc022632524a39ac570b099b5b072e81bf3b5039d4c1589779074  .../w19_runpod_d487/A_full_cycle/run_01/w13_cycle_seed460.npz
```

## 2. 무엇을 바꿨나 (DEME→표시 좌표 변환만)

원자료가 변환을 **직접 기록**한다 — 추측하지 않는다:
`metadata_json.w25_frame = {R_robot_box, t_robot_m, box_frame_convention, box_anchor}`
(rev34 `src/sim_w13_full_cycle.py:1324-1327`, 주석 "p_robot = R_robot_box @ p_box + t_robot (소비자는 추측 금지)").
규약 정의는 `src/w13_fk.py:279-284` `BOX_FRAME_CONVENTIONS` (A = `Rz(−90°)`, B = `Rz(+90°)`).

| post06 줄 | 바뀐 것 | 어느 배열/대상 |
|---|---|---|
| `:358-409` | `w25_frame` 읽기 + **fail-closed 검사** 5종 + `Q_ROT`(자세용 쿼터니언) | — |
| `:415` | `origin_disp` 정의는 **그대로**(z 오프셋은 회전 뒤 한 번) | — |
| `:417-422` | `deme_to_disp`: `p + origin` → `p @ Rᵀ + origin` (끝 축 3 이면 `(3,)·(N,3)·(K,N,3)` 모두) | 입자 위치, 공구/문 CAD·충돌셸 정점, 용기 메시, 마커, 카메라 경계 점집합, 트레이/테두리 상자 |
| `:424-428` | `disp_box(lo,hi)`: 회전으로 뒤집힌 축의 lo/hi 를 성분별 min/max 로 되돌림 | 축정렬 표시 상자 **20 개**(tray 4·rim 4·post 4·floor_edge 4·wall_bottom 4) |
| `:430-440` | `disp_quat(q)`: `q_R ⊗ q_box`(Hamilton) | **입자 자세**(길쭉한 클럼프라 위치만 돌리면 방향이 어긋난다) |
| `:442-447` | `box_to_robot(p)`: `p + t` → `p @ Rᵀ + t` | `tool_pos_m[sync]` → IK/FK 입력 립 위치 (`:508`, `:531`) |
| `:449-454` | `robot_to_box(p)`: `p − t` → `(p − t) @ R` | 합성 fixture 의 FK 결과를 상자 좌표로 (`:1057`) |
| `:456-461` | `robot_to_box_rot(R_r)`: `Rᵀ · R_r` | 합성 fixture owner 자세 (`:1058`) — rev34 `w13_fk.build_adapter_w25` 의 `R_scoop_owner_box = R_box_robot` 과 같은 규칙 |
| `:1208` | 입자 자세에 `disp_quat` 적용 | `particle_quat_xyzw[frame]` |
| `:615-633` | `manifest["mapping"]` 에 회전 영수증(`w25_frame_present`, `R_robot_box_applied`, `rotation_applied_to`, `z_offset_applied_once`, `robot_joint_path_unrotated`) | 매니페스트 |
| `:563`, `:792` | 주석만 — 회전 불변 지점 명시(변경 0) | `door_r_max`(상자 좌표 안 거리), `CF.aabb_corners`(두 점의 8 조합이라 lo/hi 뒤집힘 무관) |

핵심 hunk (전문은 `DIFF_post05_to_post06.patch`):

```diff
-    def deme_to_disp(pw):
-        return np.asarray(pw, float) + origin_disp
+    def deme_to_disp(pw):
+        """상자(DEME) 좌표 → 표시 좌표. 끝 축이 3 이면 어떤 모양이든 된다((3,)·(N,3)·(K,N,3))."""
+        p = np.asarray(pw, float)
+        if ROT is None:
+            return p + origin_disp                      # post05 와 **같은 식**
+        return p @ ROT.T + origin_disp
```
```diff
     def joints_for(si):
-        lip_robot = np.asarray(tool_p[si], float) + t_robot
+        # post06: 저장된 `tool_pos_m` 은 **상자(DEME) 좌표**다 → 로봇 좌표로 옮겨야 IK/FK 가 맞는다.
+        lip_robot = box_to_robot(tool_p[si])
```
```diff
-        tray_xn = static_box("/World/tray/xn", deme_to_disp(TRAY["tray_xn"][0]),
-                             deme_to_disp(TRAY["tray_xn"][1]), TRAY_FACE_COLOR, TRAY_FACE_OPACITY)
+        tray_xn = static_box("/World/tray/xn", *disp_box(TRAY["tray_xn"][0],
+                             TRAY["tray_xn"][1]), TRAY_FACE_COLOR, TRAY_FACE_OPACITY)
```

**바꾸지 않은 것**: 물리·판정·게이트·프레임 1:1 계약·문 각도·`fixtures` 일치 게이트(상자 좌표 안에서만 비교하므로
회전 불변)·행 정체성 게이트·close 진단. 회전은 표시층에만 든다.

### 2-1. fail-closed 검사 (`:360-382`)

`w25_frame` 이 있을 때 다음 중 하나라도 어긋나면 **조용히 그리지 않고 `SystemExit`** 한다.
① `R_robot_box` 3×3 ② 직교·`det=+1` ③ **z 축 회전**(z 행·열이 단위) ④ **부호 있는 축 치환**
(축정렬 표시 상자를 축정렬로 유지할 수 있는 조건) ⑤ 결과 JSON `frames.adapter.t_robot_m` 과
`metadata.w25_frame.t_robot_m` 의 **완전 일치**(다르면 어느 쪽이 맞는지 추측하지 않는다).
기록이 단위행렬이면 `ROT=None` 으로 되돌려 rev32 경로와 동치로 만든다.

### 2-2. 로봇 관절(FK/IK)을 변환하지 않는 이유 — 코드 근거

관절각은 **로봇 좌표계의 양**이고, 표시 장면에서 로봇은 **회전 없이** 놓인다.

* `isaac_replay_w13_post06.py:893` — `init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, PLATE_Z), ...)`
  → 로봇 prim 은 표시 원점 위 z=`PLATE_Z` 에 **자세 지정 없이**(회전 항 없음) 놓인다. 즉 표시 축 = 로봇 축.
* `:812` — 카메라 경계 집합의 어깨(팔 원점) = `[0, 0, PLATE_Z + FK.SHOULDER_ABOVE_PLATE]`, `:848` — 팔 링크
  AABB 는 FK 결과에 `[0, 0, PLATE_Z + SHOULDER_ABOVE_PLATE]` **만** 더해 표시 좌표로 간다(xy 이동·회전 0).
* `:1293-1296` `set_pose` 는 `rec["ik"]["q5_deg"]` 를 라디안으로 바꿔 관절에 직접 쓴다 — 좌표 변환이 개입할 자리가 없다.

바뀌어야 하는 것은 **IK 입력**이다: 저장된 `tool_pos_m` 은 상자 좌표라서 로봇 좌표로 옮겨야 한다(`:508`, `:531`).
post05 는 이 자리에서 `+ t_robot` 만 했다 → 규약 A 에서 립 목표가 틀린 곳에 찍힌다(수치는 §3-4).

### 2-3. `t_robot` 의 z 와 `origin_disp` 의 z 가 이중 적용되지 않는 근거

* `origin_disp = [t_x, t_y, t_z + PLATE_Z + SHOULDER_ABOVE_PLATE]` (`:415`) — `t_robot[2]` 는 **여기서만** 더해진다.
* 규약 A/B 의 `R_robot_box` 는 **z 축 회전**이라 `(R·p)[2] == p[2]` — 회전이 z 에 기여하지 않는다.
  검사 ③ 이 이 전제를 강제하고, 자체검사가 실측한다: `rotation_changes_z = false`.
* 어느 헬퍼도 `t_robot[2]` 를 다시 더하지 않는다(`box_to_robot` 은 표시 z 오프셋 없이 `t_robot` 만, `deme_to_disp`
  는 `origin_disp` 만).
* 실측: 스텁 전 입자에 대해 `(표시 z − 상자 z) − origin_disp[2]` 의 **최대 절대 편차 = 0.0 m**,
  `origin_disp[2] = 0.2443098685822132 = (−0.2577501314177868) + 0.38 + 0.12206` (`t_robot[2] + PLATE_Z + 0.0701+0.05196`).

## 3. CPU 자체검사 결과 (`selftest_post06.py` → `selftest_post06.json`, 9/9 PASS)

검사 방식: 재구현이 아니라 **패치된 파일의 함수 본문을 AST 로 꺼내 그대로 실행**한다(post05 도 같은 방식).
따라서 아래 수치는 렌더가 실제로 쓸 식의 수치다. 실행 = 시스템 `python3` + numpy 2.4.2, 순수 CPU, 원자료 읽기만.
`SHOULDER_ABOVE_PLATE` 는 하드코딩하지 않고 동결 `w13_fk.py:43`(`0.0701 + 0.05196`)에서 읽는다.

### 3-1. (a) rev34 스텁 원자료 — 규약 A

입력 `.../rev34/dryrun/paperbox_final_n67737/w13_cycle_seed460.npz` (입자 프레임 264 × 알 67,737).
`w25_frame` = `R_robot_box = [[0,1,0],[−1,0,0],[0,0,1]]`(z 축 **−90.0°**), `t_robot = (0.25, 0, −0.2577501314177868)`,
`box_frame_convention = "A"`, `box_anchor = "declared_box_center"`. 상자 내부 310 × 220 × 230 mm.

**첫 입자 프레임을 로봇 좌표로 옮긴 결과** (기대값 단언 없이 계산한 값):

| 항목 | 상자 좌표 | 로봇 좌표 |
|---|---|---|
| xy 최소 | (−0.1537556, −0.1087566) m | (**0.1412434**, −0.1537602) m |
| xy 최대 | (0.1537602, 0.1087563) m | (**0.3587563**, 0.1537556) m |
| xy 중심 | (2.3e−06, −1.4e−07) m | (**0.249999858, −0.000002317**) m |
| xy 크기 | 307.5158 × 217.5130 mm | **217.5130 (x) × 307.5158 (y) mm** |
| 긴 변 방향 | x | **y** |
| z 범위 | 0.0012351 ~ 0.0416674 m | −0.2565150 ~ −0.2160827 m |

즉 **310 mm 긴 변이 로봇 y 축**, **220 mm 짧은 변이 로봇 x 축(로봇 정면 깊이)** 이고 더미 중심이 로봇 xy
(0.25, ≈0) m 에 온다 — `t_robot` 과 일치. 로봇 쪽 벽은 x = 0.14 m, 먼 벽은 x = 0.36 m.
표시 좌표 트레이 상자 모서리도 `lo ≤ hi` 로 정렬된다(`[0.14, −0.155, 0.24431] → [0.36, 0.155, 0.47431]`).

### 3-2. (b) W19 A 원자료 — `w25_frame` 없음 → post05 와 동일

입력 `.../w19_runpod_d487/A_full_cycle/run_01/w13_cycle_seed460.npz` (입자 프레임 288 × 알 20,000, sync 16,813).
post05 변환 vs post06 변환 **최대 절대 차 (8 항목 전부 정확히 0.0 m)**:

| 항목 | 최대 절대 차 |
|---|---|
| 입자 위치 frame0 (20,000×3) | `0.0` |
| 입자 자세 frame0 (20,000×4) | `0.0` |
| 공구 원점 전 sync (16,813×3) | `0.0` |
| 문 원점 전 sync (16,813×3) | `0.0` |
| `box_to_robot` vs `p + t_robot` | `0.0` |
| `robot_to_box` vs `p − t_robot` | `0.0` |
| `disp_box` lo / hi vs post05 | `0.0` / `0.0` |

`w25_frame` 이 없으면 `ROT is None` 분기라 **post05 와 같은 식**을 탄다(회전 분기 미진입).

### 3-3. (c) sync 0 owner 포즈와 CAD 배치식

| 원자료 | owner | 상자 좌표 위치 (m) | 로봇 좌표 (m) | 표시 좌표 (m) | `R_owner` 최대 변화 | CAD 항등식 최대 차 |
|---|---|---|---|---|---|---|
| rev34 스텁 | 고정부 | (0.0081, 0.1478410, 0.4794181) | (0.3978410, −0.0081, 0.2216680) | (0.3978410, −0.0081, 0.7237280) | **1.0** | 1.11e−16 m |
| rev34 스텁 | 문 | (−0.0, 0.0302760, 0.4605971) | (0.2802760, 0.0, 0.2028470) | (0.2802760, 0.0, 0.7049070) | **1.3488** | 1.11e−16 m |
| W19 A | 고정부 | (0.0478277, −0.0, 0.5049088) | (0.3978410, −0.0081, 0.2216680) | (0.3978410, −0.0081, 0.7237280) | **0.0** | 1.11e−16 m |
| W19 A | 문 | (−0.0697373, 0.0081000, 0.4860877) | (0.2802760, 3e−09, 0.2028470) | (0.2802760, 3e−09, 0.7049070) | **0.0** | 1.11e−16 m |

* **CAD 배치식은 같은 R·t 를 탄다**: `deme_to_disp(place(v, p, q)) == v·(R·R_owner)ᵀ + (R·p + origin)` 를
  owner-local 표본점 64 개로 확인 → 최대 차 **1.11e−16 m**(배정밀도 반올림). 즉 post05 의 CAD 배치식은
  정점을 상자 좌표로 만든 뒤 `deme_to_disp` 를 타므로, 위치·자세 모두 자동으로 같은 회전을 받는다.
* **조립 강체성 보존**: 문↔고정부 상대 회전 변화 `0.0`, 두 원점 거리 변화 `0.0 m`(스텁 0.11933721 m,
  W19 A 0.11933719 m) — 회전이 조립을 통째로 돌리지 부품을 흐트러뜨리지 않는다.
* **교차 확인(강한 증거)**: 두 원자료는 상자 좌표 위치가 완전히 다른데 **로봇 좌표에서는 같은 자리**다
  (고정부 x 0.397841007 vs 0.397840978, 차 2.9e−8 m). 규약 A 가 "립은 같은 로봇 자세, 상자만 90° 회전"
  이라는 뜻이므로 이 일치는 `box_to_robot` 에 R 을 넣은 해석이 원자료와 맞음을 보인다.

### 3-4. 패치하지 않았다면 얼마나 틀리나 (규약 A 원자료 기준)

| 항목 | 값 |
|---|---|
| sync 0 립 로봇 좌표 — post05 식 | (0.2581000, 0.1478410, 0.2216680) m |
| sync 0 립 로봇 좌표 — post06 | (0.3978410, −0.0081000, 0.2216680) m |
| 립 위치 어긋남 (18,864 sync) | sync0 **209.39 mm** · 평균 243.26 mm · **최대 507.73 mm** |
| 첫 프레임 입자 표시 위치 어긋남 | 평균 141.85 mm · **최대 265.71 mm** |
| owner 자세 어긋남 | **90.0°** |

### 3-5. 표시 한계 수치 — 렌더 전에 CPU 로 계산 (`display_limits_post06.py`)

정의는 **이전 감사와 같다**(post05 `:520-528`): `max_lip_reprojection_err_mm` = 표시 관절로 FK 한 립 위치와
저장된 립 위치 차의 최대값, `n_frames_with_limit_violations` = `FK.in_limits(q5)` 가 비지 않은 표시 레코드 수,
`joint_source_counts` = 정확 문자열 일치 집계. post05·post06 의 `mode_of`/`joints_for` 본문을 AST 로 꺼내
**그대로 실행**해 얻었다(동결 `w13_fk` import, `sys.dont_write_bytecode=True` → 동결 worktree 에 `__pycache__` 0).

| 경우 | 변환 | 재투영 최대(중앙값) | 선언 범위 밖 포즈 | 최대 초과 | IK 실패 | 관절 출처(abs/w11/fk) |
|---|---|---|---|---|---|---|
| rev34 스텁 | **post06(회전)** | **8.1001 mm** (0.0) | **0 / 264** | 0.0° | 0 | 2 / 52 / 210 = 264 |
| rev34 스텁 | post05(미패치) | **507.7214 mm** (157.7924) | 0 / 264 | 0.0° | 0 | 2 / 52 / 210 = 264 |
| W19 A | post06 | **8.132 mm** (0.0002) | **20 / 288** (shoulder) | 2.37° | 0 | 2 / 54 / 232 = 288 |
| W19 A | post05 | 8.132 mm (0.0002) | 20 / 288 (shoulder) | 2.37° | 0 | 2 / 54 / 232 = 288 |

* **이전 산출물과의 독립 대조**: W19 A 의 post06 수치가 2026-09-29 post05 실제 렌더의
  `render_post05_w19A_20260929/render_manifest.json` `robot_display_contract` 와 **세 항목 전부 일치**
  (8.132 mm · 20 프레임 · 출처 집계 2/54/232) → `all_match: true`. 즉 회귀 없음이 **렌더 영수증**으로도 확인된다.
* 8.1 mm 대의 잔차는 post05 가 이미 적어 둔 뜻 그대로 — stored-joint 프레임에서 **두꺼워진 owner 립 정의 차이**다
  (미해결 결함이 아니라 정의 차이). rev34 스텁의 중앙값이 **0.0 mm** 인 것은 립 목표와 저장 관절이 서로 맞는다는 뜻이다.
* post05 를 규약 A 에 그대로 쓰면 중앙값 157.8 mm·최대 507.7 mm — **이 패치가 없으면 재생 화면의 로봇이
  더미와 전혀 다른 곳을 집는다**. (범위 밖 포즈가 0 인 것은 IK 가 여전히 풀리기 때문이지 맞다는 뜻이 아니다.)
* 이 수치는 **기하 일관성 지표이지 구동 검증이 아니다**(post05 `NOT_ACTUATION_VERIFICATION` 계약 그대로).

## 4. 실행 명령 초안 (실행은 메인의 별도 승인 뒤에)

이 폴더는 **아무 렌더도 돌리지 않았다**. 아래는 초안이며, 출력 폴더는 **비어 있어야** 한다
(스크립트가 배타 소유를 강제하고 기존 항목이 있으면 거절한다).

2026-09-29 post05 실행과 **같은 기동 형태**를 쓴다(세션 로그 `session_20260928_w25_realign_fullcycle_prep.md:90`:
`setsid timeout -k 60 1800 isaaclab/python isaac_replay_w13_post05.py --run … --out … --headless --enable_cameras`).
`isaaclab/python` = `/home/cgxr/miniconda3/envs/isaaclab/bin/python`(post05 렌더 매니페스트 `versions`:
isaacsim 5.1.0-rc.19 · torch 2.7.0+cu128 · **numpy 1.26.0** · python 3.11.14 — 설치 0, 핀 무변경).

```bash
P=/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/replay_post06_convA_20260929
# 규약 A(rev34) 원자료 재생 — GPU 1개 전용, 다른 GPU 작업과 동시 실행 금지
setsid timeout -k 60 1800 /home/cgxr/miniconda3/envs/isaaclab/bin/python -B "$P/isaac_replay_w13_post06.py" \
  --run  <w13_cycle_seed460.{json,npz} 가 있는 rev34 run 폴더 절대경로> \
  --out  "$P/render_rev34_convA" \
  --headless --enable_cameras \
  > "$P/post06_render.stdout.txt" 2> "$P/post06_render.stderr.txt"
# 회귀 확인(선택, 같은 스크립트로 W19 A 재생 → post05 산출과 대조):
#   --run /home/cgxr/orca/workspaces/RoArm_Project/w19-replay/claudedocs/runtime_logs/grasp_track/\
#         w19_runpod_d487/A_full_cycle/replay_20260918   --out "$P/render_w19A_regress"
```

* `--run` 폴더는 `w13_cycle_seed460.json`·`.npz`(심링크 가능)를 가져야 하고, `--show-collision-shell` 을
  쓸 때만 `_obj/*.obj` 가 필요하다. `--out` 은 **비어 있는 새 폴더**여야 한다(스크립트가 거절한다).
* `timeout` 값 1800 s 는 post05 선례다 — W19 A 288 프레임에서 **실제 경과 시간(wall-clock) 506.018 s**
  (`render_post05_w19A_20260929/render_manifest.json` `phase_seconds.total_s`; 기동 30.3 + 렌더 450.2 +
  ffmpeg 5.3 + 정리 20.2, `close_timed_out=false`). **rev34 본 실행은 프레임 수가 다르므로 cap 을 결과 보기
  전에 먼저 정한다**(D485: 결과를 본 뒤 상한을 늘리지 않는다. 타임아웃은 성공이 아니다 — D486).

* `isaaclab` env 에 **설치 0**(`numpy==1.26.0`·`psutil==5.9.8` 핀 무변경 — D326). 이 폴더의 자체검사는
  시스템 python3 로 돌렸고 isaaclab env 를 건드리지 않았다.
* 실행 후 확인: `render_manifest.json` 의 `ok`·`failure_reasons`·`mapping`(회전 영수증)·
  `camera_framing.fits`, `visual_mapping.json` 의 행 수 = 저장 입자 프레임 수.
  ```bash
  python -B -c "import json;m=json.load(open('<out>/visual_mapping.json'));print(m['n_mapped_rows'], m['n_raw_particle_frames'])"
  python -B -c "import json;m=json.load(open('<out>/render_manifest.json'));print(m['ok'], m['mapping']['R_robot_box_applied'])"
  ```
* close 행(hang) 대비: 스크립트 자체 `--close-budget-s` 위에 **바깥 PGID 경계 타임아웃**을 둔다(D477·D486 —
  타임아웃은 성공이 아니다). 재시도 0.

## 5. D341 Rerun 완료 계약 — 이 준비 단계가 채운 것 / 못 채운 것

이 과제는 **Isaac 관절 재생 경로의 좌표 준비**이고 Rerun RRD 경로는 손대지 않았다. 정직하게 적는다.

| D341 항목 | 상태 | 근거 |
|---|---|---|
| SDK/CLI 버전 핀 | ✗ 미충족 | Rerun 을 돌리지 않았다. `rerun --version` 실행 0 |
| 파일 sink 를 첫 로그 이전에 부착 | ✗ 미충족(해당 없음, 이 단계) | RRD 생성 0 |
| `RecordingStream` 종료로 finalize | ✗ 미충족 | 같음 |
| footer 포함 `rrd verify` PASS | ✗ 미충족 | 같음 |
| 엔티티/타임라인/필수 구성요소 exact 계약 | ✗ 미충족 | 같음 |
| 고정 청사진 + `.rbl` 내보내기 검증 | ✗ 미충족 | 같음 |
| 헤드리스 결정 스크린샷 | ✗ 미충족 | GPU 기동 금지 과제 |
| **실제 육안 검수 기록** | ✗ 미충족 | **화면을 본 적이 없다.** 이 문서의 모든 수치는 좌표 계산값이다 |
| 결정 대상이 기록 안에 있을 것 | △ 준비만 | post05 의 결정 태그·입자·공구 엔티티 구조는 그대로 유지. 변환만 고침 |
| Rerun 은 비트 정확 권위가 아님 | ✓ 지킴 | 동일성 판정은 원자료 배열로만 했다(자체검사는 원본 npz 배열을 읽어 배정밀도로 비교). Float32 공간 사본을 과학 게이트에 재해싱 0 |
| 과학 판정 불변 | ✓ 지킴 | 물리·회계·판정 코드 변경 0. 표시층만 |
| 영수증·해시 대조 | ✓ 지킴 | §1 의 sha 전후 대조 OK |
| 사후 완화 금지 | ✓ 지킴 | 미충족 항목을 위에 그대로 남겼다 |

**D324 시각 진단** 도 같은 이유로 이 단계에서는 스냅샷이 없다(렌더 0). 다음 단계에서 렌더가 승인되면
결정 시점 스냅샷 경로를 보고서에 적어야 한다.

## 6. 남은 위험·미확인 (렌더 전에 알고 있어야 할 것)

1. **카메라 프레이밍은 자동이지만 검증은 렌더 뒤에만 가능하다.** 카메라는 변환된 점집합에서 최소 거리를
   닫힌 형태로 풀므로 상자가 돌아가도 따라간다(`CF.solve_framing`), 그리고 `fits`/far-clip 은 게이트다.
   다만 "들어온다"는 계산이지 **육안 검수가 아니다**(post05 가 이미 `non_claims` 에 적어 둔 한계).
2. **`--readiness --synthetic-phases` 경로는 CPU 검사로 검증하지 않았다.** 그 경로의 `robot_to_box`/
   `robot_to_box_rot` 는 코드로만 맞췄고 수치 확인은 못 했다(합성 fixture 는 본 실행에서 쓰이지 않는다).
3. **`w25_frame` 이 규약 B 이거나 축 치환이 아닌 R 이면 스크립트가 멈춘다**(fail-closed). 규약 B 원자료가
   생기면 그때 같은 방식으로 자체검사를 다시 돌려야 한다.
4. **스텁 원자료는 스텁이다.** (a) 의 수치는 rev34 `dryrun/paperbox_final_n67737` 스텁에서 나왔고
   본 실행 원자료가 아니다(`STUB_RECEIPT.json` 동봉 폴더). 본 실행이 나오면 같은 자체검사를 그 npz 로 다시 돌린다.
5. **rev34 스텁의 재투영 8.1001 mm 는 "정상"을 보증하지 않는다.** 정의상 두꺼워진 owner 립 차이가 섞인
   값이며, 본 실행 원자료에서 다시 재야 한다. 범위 밖 포즈 0 도 스텁 궤적 기준이다.
6. 이 폴더는 "전체 사이클 성공" 같은 선언을 하지 않는다. 표시 계약도 렌더·육안 검수 전에는 미충족이다.

## 7. 이 준비 단계에서 실제로 한 일 (감사 추적)

1. `logs/post05_script_sha256.txt` 를 읽고 그 경로의 sha256 을 **재계산해 일치 확인** → 바이트 사본.
2. 원자료 2종 sha256 기록 → 모든 CPU 작업 뒤 `sha256sum -c` **OK**(원자료 무변경, 쓰기 0).
3. 변환 패치(§2) → `python -m py_compile` OK · `ast.parse` OK · 치환 수 검증(static_box 20/20).
4. `selftest_post06.py` 9/9 PASS, `display_limits_post06.py` 로 표시 한계 계산 + 이전 렌더 영수증과 대조 일치.
5. 동결 worktree 에 `__pycache__` 를 만들지 않았다(`sys.dont_write_bytecode=True`, 사후 `ls` 로 0 확인).
6. Isaac/DEME/GPU 프로세스 기동 0 · RunPod 접근 0 · 설치 0 · git commit/push 0 · 상태 원장 쓰기 0.
