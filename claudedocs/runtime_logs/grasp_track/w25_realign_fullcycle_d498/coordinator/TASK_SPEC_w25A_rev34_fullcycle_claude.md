# TASK_SPEC W25-A — 실물 정렬 전체 사이클 revision(rev34) 초안 + CPU 사전검토 (실행 0)

작성 2026-09-28 · 코디네이터 = 메인 Claude Code(Fable 5.1) · 이 과제서가 계약 정본. 보고는 **한국어**, 절차→수치→근거(file:line·sha256)→한계→다음 승인 경계 순, step-by-step.

## 0. 절대 금지 (위반 시 실패 처리)
- GPU 사용 0(로컬 4090 포함) · DEME 물리 0(`DEME.DEMSolver()` 생성·Initialize 금지) · Isaac Sim 앱 기동 0 · RunPod 0 · 로봇/시리얼/카메라 0 · 설치 0(pip/conda/npm/apt 전부) · git commit/push/merge/branch 삭제 0.
- 원장 쓰기 0: `START_HERE.md`, `claudedocs/{DECISIONS,DECISIONS_ACTIVE,EXPERIMENT_LEDGER,LEDGER_RECENT,BACKLOG}.md`, `claudedocs/relay/`, `claudedocs/session_*.md` 는 읽기만. 다른 worktree·메인 repo 파일 쓰기 0.
- **동결 revision 원본 무수정**: `rev32_frozen_copy/` 는 바이트 사본을 뜬 뒤 사본만 고친다. 판정 문턱(criteria.json)은 바꾸지 않는다(D492: 결과를 본 뒤 임계 조정 금지, `policy.no_threshold_change_after_outcomes`).
- 산출은 **자기 worktree** 의 `claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/rev34/` 와 `claudedocs/research/w25_rev34_20260928/` 아래에만(forward-only, 기존 파일 이동·개명 금지).
- 파일명·주석·문서 문장을 근거로 형상·좌표를 판단하지 말고 **배열·수치를 직접 열어** 확인한다. 좌표를 말할 때는 기준(로봇/DEME/상자/실물)을 항상 명시한다.

## 1. 배경(사실, 근거 경로) — 메인이 원자료로 확인한 것
- W19 A 전체 사이클(rev32, RunPod 4090, 물리 24.807 s 완주)은 다음 세 가지가 **실물과 다르다**:
  1. **상자 방향**: 실행 로그 `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/run_01/runA_full_cycle.stdout.txt:2` 의 `t_robot [0.350013, -0.0081, -0.283241]`(DEME 원점 = 상자 바닥 중심의 **로봇 좌표**) 와 `w13_fk.py` Adapter 도크스트링("축은 평행, 원점만 평행이동") → 로봇 베이스는 DEME 좌표 (−0.35, +0.008) = 상자 **x(31 cm) 축 위**. 즉 시뮬은 **22 cm 벽이 로봇을 마주 봄**. 실물은 **31 cm 벽이 로봇을 마주 봄**(`START_HERE.md`, D496). 더미 NPZ `box_bounds_m` = x ±0.155·y ±0.11(W23 F 보고 §3-4).
  2. **더미**: 시뮬 = 알 20,000개 둔덕(가운데 41 mm→벽 5~8 mm), 실물 = 평평한 층.
  3. **동작 절차**: 실물 `hw_s1_manual.py scoop()`(`:157-169`)은 닫힌 채 접근 → **표면에서 문 30° 열기** → plunge 25 mm → 닫기(토크 900) → **닫힘 서보 읽기 > 3.6° 면 8° 열었다 재닫기 최대 3회(채터링)** → +8 cm 재닫기 → travel. 시뮬(rev32)은 문을 이미 연 채 표면 +10 mm 에서 시작하고 채터링이 없다.
- 그리퍼: 시뮬 물리 셸은 `sim_deme_scoop_s1.py:100-140 half_bowl()`(보울 벽+캡만, 부품당 204 삼각형)이고 S1 v1 STL(`claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/{fixed_ALL,door_ALL}.stl`, 1,704/4,832 삼각형)은 검사에만 쓴다(`load_tool()` 도크스트링). 실물 최신 그리퍼 = S1 v1(D481). **이 과제에서 충돌 셸은 바꾸지 않는다**(형상 변경은 별도 case). 단 §3-(5) 의 CAD 배치 변환은 문서화한다.
- 실물 배치 사실(사용자 실측·결정, `START_HERE.md`): 바닥→책상 윗면(로봇 검은 바닥판) **33.2 cm**, 상자 중심 = 로봇 회전축 앞 **25 cm**, 받침 19.7 cm, 펠릿면 바닥 **≈24 cm**(선언값·미실측), 새 상자 후보 **NTC106 유효 안쪽 301×198×105 mm**(벽 두께 미실측), 배출 용기 = 컵(rev32 픽스처 내경 r 40·깊이 70 유지), 배출 자리 = `hw_s1_manual.py place()` 기본(+y 90°).
- rev32 동결 사본: `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/rev32_frozen_copy/`(`src/*.py`, `params_w13.json`, `criteria.json`, `COMMANDS.json`, `numeric_inputs.json`, `checks/`, `REVISION_PIN.json`, `FROZEN_COPY_NOTE.md`). 실행 절차·스모크 3종·pod 규약은 `claudedocs/session_20260917_w19_runpod_afternoon.md` §3~§9.
- W23 F(시뮬 정렬 설계, CPU) 산출: `/home/cgxr/orca/workspaces/RoArm_Project/w23-sim-alignment/claudedocs/research/sim_alignment_20260928/{REPORT.md,GPU_RUN_PLAN.md,out/}` — 규약 A(x_box = 로봇 −y, y_box = 로봇 +x) 제안, 스쿱 셀용 `tool_yaw_deg` 패치(`out/sim_deme_scoop_s1_toolyaw.diff`), 정렬 게이트(`out/align_geometry.json`). ⚠️ 그 패치는 **스쿱 셀(`sim_deme_scoop_s1.py`, R_W 고정)** 용이다. 전체 사이클(rev32)의 툴 자세는 `w13_fk.py Adapter.owner_pose()` 가 실물 FK `R_l5` 를 쓰는지 **직접 확인**해 두 러너의 차이를 보고하라(추정 금지).

## 2. 목표(Change)
rev32 를 바이트 복사한 **rev34** 를 만들고, 아래를 **params 로 선언 가능**하게 최소 diff 로 구현한다. 기본값은 "실물 정렬" 이고, 각 항목은 params 만으로 rev32 동작으로 되돌릴 수 있어야 한다(A/B 비교 가능).
1. **상자 방향·격자**: 상자 긴 변(301 mm)이 로봇 좌우(y) 방향, 짧은 변(198)이 로봇 앞뒤(x) 방향이 되도록 트레이·더미·5 mm 격자·구덩이 좌표를 놓는다. 상자 좌표↔로봇/DEME 좌표 변환을 params 키(예: `box_frame_convention: "A"|"B"`)로 노출하고 결과 JSON 에 회전행렬을 그대로 기록한다. 더미 NPZ 는 W25-C 가 만들 **평평한 층(NTC106 발자국)** 을 받는다고 가정하되, 없으면 기존 20,000알 NPZ 를 90° 회전한 임시 입력으로 사전검토를 돌리고 "임시" 를 명시한다.
2. **배치**: `declared_base_cm 33.2`, `declared_pellet_cm 24.0`, 상자 중심 반경 0.25 m, 트레이 안쪽 301×198·높이 105 mm(벽 두께는 params, 기본값과 출처 명시), 배출 컵 픽스처·배출 자리 규약은 rev32 그대로(바뀌면 이유 기록). JOINT_LIMITS 준수 여부를 전 웨이포인트에서 검산(`hw_s1_manual.py` IK 규약·`w13_fk.py`).
3. **절차 정렬(params 스위치, 기본 ON)**: (a) 닫힌 채 접근 → 표면에서 문 열기 → plunge, (b) 닫힘 뒤 서보 환산각 > 3.6° 면 8° 열었다 재닫기 최대 3회(채터링), (c) lift 8 cm 재닫기. 서보각 = 관절각 + 2.5°(`servo_zero_offset_deg`) 규약을 지켜 **관절/서보 단위를 기록에 둘 다 남긴다**. 스위치 OFF 면 rev32 절차와 sync 격자·기록 수가 바이트 동일해야 한다(CPU 로 증명 가능한 범위: AST·일정표 `build_schedule` 출력 비교).
4. **기록 계약 유지**: rev32/rev33b 의 원자료 스키마(NPZ 키·프레임 분류·정착 창)를 깨지 않는다. `particle_frame_dt_s`·`settlement_frame_dt_s` 는 rev32 값 유지(계약 변경 금지).
5. **CAD 배치 변환 문서화**(렌더 워커 W25-B 용): 고정 owner 원점(두꺼워진 립, link5 mm (8.1, 0, 169.6))·문 owner 원점(힌지 (0, 18.821, 52.035))과 S1 v1 STL(link5 프레임 mm) 사이의 강체 변환을 수식과 수치로 적고, **sync 0 의 셸 노드(`nodes_F_m`/`nodes_D_m`, W19 A NPZ)** 를 그 변환으로 재현했을 때 최대 오차(mm)를 보고한다(CPU, NPZ 읽기만).

## 3. 산출(Observable acceptance — 전부 있어야 성공)
- `rev34/` : `src/`(사본+패치), `params_w25.json`, `criteria.json`(rev32 와 sha256 동일), `DIFF_rev32_to_rev34_src.patch`, `REVISION_PIN.json`, `COMMANDS_w25_template.json`(pod 실행 명령 초안, 실행 금지), `checks/`(rev32 `checks/` 의 AST scope·binding probe·import check 를 rev34 에 재실행한 결과 + 새 검사).
- `preflight/` : `preflight_geometry.py`·`preflight_rigid_hinge.py`·`preflight_snapshot.py`(rev32 도구) 를 새 배치로 CPU 실행한 결과 — 웨이포인트 IK 오차·관절 제한·툴 셸 vs 트레이/컵 간섭(`clearance_report`) 전부 표로. 하나라도 FAIL 이면 그대로 보고(수정 대행 금지).
- `unittest`: rev32 `verify_w13_self.py`·`tests` 상당을 rev34 에 실행(CPU) → 결과 JSON.
- 그림 2장 이상(CPU, matplotlib): ① 위에서 본 배치(로봇 베이스·상자 301×198·컵·전 웨이포인트 립 궤적·상자 좌표축 라벨) ② 절차 일정표(문 각·립 z vs 시간, rev32 vs rev34).
- `REPORT.md`(한국어): 절차 → 수치 → 근거(file:line, sha256) → 한계·미확인 → 다음 승인 경계. rev32 대비 **변수 목록**을 맨 위에(`이번 case 의 신규 변수: [...]`).
- 미확인·추정은 반드시 "미실측/추정" 이라고 표기. 기대 답을 맞추려 하지 말 것 — 메인이 원자료로 재계산해 교차 검산한다.

## 4. 환경
- 파이썬: `/home/cgxr/miniconda3/envs/roarm/bin/python`(DEME import 는 하되 `DEMSolver()` 생성 금지 — import 검사만). Rerun 이 필요하면 `/home/cgxr/miniconda3/envs/isaaclab/bin/python`(numpy 1.26.0·psutil 5.9.8 핀 유지, 설치 금지).
- 참고 코드: 메인 repo `sim_deme_scoop_s1.py`, `hw_s1_manual.py`, `roarm_rl/heightmap.py`, rev32 `src/`, W23 F 산출. 모두 읽기 전용.
- Orca: 과제 중 질문은 preamble 의 `ask` 로. 완료 시 `worker_done` 에 REPORT.md 절대경로를 `--report-path` 로.
