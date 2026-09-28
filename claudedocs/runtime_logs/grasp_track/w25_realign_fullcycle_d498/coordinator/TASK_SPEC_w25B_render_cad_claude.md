# TASK_SPEC W25-B — 재생(Isaac/Rerun)에 실제 S1 v1 CAD 그리퍼를 그리기 위한 준비 + CPU 검증 (실행 0)

작성 2026-09-28 · 코디네이터 = 메인 Claude Code(Fable 5.1) · 이 과제서가 계약 정본. 보고는 **한국어**, 절차→수치→근거(file:line·sha256)→한계→다음 승인 경계 순, step-by-step.

## 0. 절대 금지
- GPU 0(Isaac Sim 앱 기동·`SimulationApp` 금지, Rerun 뷰어 스크린샷 금지) · DEME 물리 0 · RunPod 0 · 로봇/카메라 0 · 설치 0 · git commit/push/merge 0 · 원장(START_HERE/DECISIONS*/EXPERIMENT_LEDGER/LEDGER_RECENT/BACKLOG/relay/session_*) 쓰기 0 · 다른 worktree·메인 repo 쓰기 0.
- 산출은 자기 worktree 의 `claudedocs/research/w25_render_cad_20260928/` 아래에만(forward-only).
- 파일명·주석으로 판단하지 말고 배열을 직접 열어 확인. 좌표 기준(link5/owner/DEME/로봇/표시) 항상 명시.

## 1. 배경(메인이 확인한 사실)
- W13/W19 재생 화면의 그리퍼는 **DEME 충돌 셸**(보울 벽+캡만, 고정 204·문 204 삼각형)이다: `isaac_replay_w13.py:740-772`(rev32 동결 사본 `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/rev32_frozen_copy/src/`) 가 `_obj/fixed_*.obj`·`door_*.obj` 의 위상에 NPZ `nodes_F_m`/`nodes_D_m` 를 매 sync 넣고, 경로에 `gripper` 가 든 USD prim 을 숨긴다(`:834-852`). 사용자 판정: "덜 그려진 것" — **실제 S1 v1 CAD 형상으로 그려야 한다.**
- S1 v1 CAD(실물 최신, D481): `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/{fixed_ALL.stl(1,704 tri), door_ALL.stl(4,832 tri), door_ALL_jawframe.stl, design.json}` — link5 프레임 mm(`sim_deme_scoop_s1.py:41-42, :150-193 load_tool` 참고). 실물 출력본 `s1_v1_print/{fixed,door_L,door_R}.stl` 은 출력 방향으로 회전된 같은 형상(V/F 동일).
- 물리 셸 규약: 고정 owner 원점 = 두꺼워진 립 link5 (8.1, 0, 169.6) mm, 문 owner 원점 = 힌지 link5 (0, 18.821, 52.035) mm, `R_W = [[0,-1,0],[-1,0,0],[0,0,-1]]`(link5→세계 열벡터), 문 OBJ 는 `q_open` 자세로 구워짐(`sim_deme_scoop_s1.py:190-193`, `isaac_replay_w13.py:775-830` 의 합성 포즈 주석). USD: `local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd`(문 관절 `link5_to_gripper_link`).
- 기존 재생 파이프라인(post04): `/home/cgxr/orca/workspaces/RoArm_Project/w19-replay/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/replay_20260918/{rev/src/, execution/EXECUTION_RECEIPT.json(argv), isaac/frames/, isaac/render_manifest.json, REPORT_replay.md}`. 원자료 NPZ: `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/A_full_cycle/run_01/w13_cycle_seed460.npz`(393 MB, 읽기만)·`w13_cycle_seed460.json`.
- 실물 상자 방향은 시뮬과 90° 다르며 새 revision(rev34, W25-A 워커)은 상자 301×198 을 로봇 좌우 방향으로 놓는다. 렌더는 **하드코딩 없이 실행 결과 JSON 의 fixtures(트레이·컵)** 를 읽어 그려야 한다.

## 2. 목표(Change)
1. **CAD 그리퍼 배치 수학**: NPZ 의 `tool_pos_m/tool_quat_xyzw`(고정 owner)·`door_pos_m/door_quat_xyzw`(문 owner) 로부터 S1 v1 STL 정점(link5 mm)을 세계(DEME) 좌표로 놓는 변환을 유도·구현(순수 numpy). **검증(CPU)**: W19 A NPZ 의 sync 0·문 열림 프레임(discharge)·재닫힘 프레임에서, 같은 변환을 충돌 셸 정점(`half_bowl` 로 재생성)에 적용한 결과가 저장된 `nodes_F_m`/`nodes_D_m` 와 일치하는지 최대 오차(mm)로 보고. 오차가 0.01 mm 를 넘으면 원인을 찾아 보고(수정은 하되 근거 필수).
2. **재생 스크립트 개정안(post05 초안)**: `isaac_replay_w13.py` 사본에 (a) `/World/s1_cad/fixed`·`/World/s1_cad/door` 로 CAD 메시를 추가하고 매 프레임 1 의 변환으로 갱신, (b) 충돌 셸은 반투명 오버레이 옵션(기본 off), (c) 트레이·컵은 결과 JSON `fixtures` 에서 읽어 그리기(rev32 도 이미 그렇다면 그 줄을 근거로 남기고 변경 0), (d) 오버레이 문구를 "tool = S1 v1 CAD (posed from raw owner poses) + collision shell (optional)" 로. **Isaac 기동은 하지 않는다** — AST 검사·`python -m py_compile`·순수 함수 단위 테스트까지만.
3. **CPU 미리보기 그림**(matplotlib Poly3DCollection, 라벨 한국어 폰트 `/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc`): W19 A 프레임 201(문 열림)과 sync 0 에서 ① 충돌 셸 ② S1 v1 CAD ③ 둘 겹침, 사선·정면·측면 3뷰. 로봇 팔은 그리지 않아도 됨(툴·트레이·컵·입자 표본 1,000개 정도).
4. **USD 은닉 prim 확인**: `strings`/바이너리 토큰 검색으로 `roarm_m3_s1_v1.usd` 안에 S1 v1 메시 prim(이름에 s1/fixed/door/gripper 등) 이 있는지, 재생이 `gripper` 필터로 무엇을 숨기는지 표로. pxr 미설치라 USD 파싱은 하지 않는다(설치 금지).
5. **Rerun(RRD) 쪽**: `w13_rerun_export.py` 가 툴을 어떻게 그리는지(셸 노드인지) 확인하고 CAD 엔티티 추가 지점을 diff 초안으로. RRD 생성은 CPU 로 가능하면 작은 표본(프레임 3개)만 만들고 `rerun rrd verify` 로 검증(Rerun 0.34.1 핀, `/home/cgxr/miniconda3/envs/isaaclab/bin/python`).

## 3. 산출(Observable acceptance)
- `cad_pose_math.py`(순수 numpy) + `cad_pose_check.json`(프레임별 최대 오차 mm, 사용 프레임 인덱스·sync·시각) + 음성 대조 1건(일부러 틀린 원점으로 넣으면 오차가 커짐을 보임).
- `isaac_replay_w13_post05.py`(사본) + `DIFF_post04_to_post05.patch` + `checks/`(py_compile·AST·단위 테스트 결과).
- 그림 ≥ 6장(PNG) + `figures_manifest.json`(각 그림의 데이터 출처·프레임).
- `usd_prims.md`(표), `rerun_cad_diff.patch`(초안), 표본 RRD + verify 로그(가능한 경우).
- `REPORT.md`(한국어): 절차 → 수치 → 근거 → 한계(예: Isaac 실제 렌더 미실행, GPU 승인 뒤 필요한 명령 1줄) → 다음 승인 경계.

## 4. 환경
- CPU 파이썬 `/home/cgxr/miniconda3/envs/roarm/bin/python`(trimesh·numpy·matplotlib 있음). Rerun 은 isaaclab env. 설치 0.
- Orca: 질문은 preamble 의 `ask`. 완료 시 `worker_done` + `--report-path`.
