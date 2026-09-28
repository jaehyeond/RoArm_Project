# TASK_SPEC W25-D — 실물처럼 "벽까지 꽉 찬 4 cm 평평한 층" 더미 생성 준비 + 새 도메인 격자 증거 가능성 확인 (CPU 전용, 실행 0)

작성 2026-09-28 · 코디네이터 = 메인 Claude Code(Fable 5.1) · 이 과제서가 계약 정본. 보고는 **한국어**, 절차→수치→근거(file:line·sha256)→한계→다음 승인 경계 순, step-by-step.

## 0. 절대 금지
- GPU 0 · DEME 물리 0(`DEMSolver()` 생성·더미 생성 실행 금지 — 명령 초안만) · Isaac 0 · RunPod 0 · 로봇 0 · 설치 0 · git commit/push/merge 0 · 원장(START_HERE/DECISIONS*/EXPERIMENT_LEDGER/LEDGER_RECENT/BACKLOG/relay/session_*) 쓰기 0 · 다른 worktree·메인 repo 쓰기 0.
- `pellet-model` worktree 의 `sim_deme_pile.py` 원본은 **무수정**. 패치는 자기 worktree 사본(`claudedocs/research/w25_pile_flat40_20260928/src/`)에만.
- 산출은 자기 worktree `claudedocs/research/w25_pile_flat40_20260928/` 아래에만. 파일명·주석으로 판단하지 말고 배열·코드를 열어 확인.

## 1. 배경(사실)
- **사용자 실측(09-28 밤, HARD RULE #18 우선)**: 실물 상자(현재 임시 A4 종이 상자, 윗단 바깥 31×22 cm·벽 0.2 cm·높이 23 cm)에 펠릿이 **벽면까지 꽉 찬 상태**, 부분 높이는 다르지만 **상자 바닥→펠릿 윗면 약 4 cm**. 이 층을 시뮬 초기 더미로 재현해야 한다.
- 기존 더미 NPZ: `/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz` — `box_bounds_m` x ±0.155·y ±0.11(= 310×220), 알 20,000 × 구 7(렌즈 a4.5 b3.8 c2.5 mm, 템플릿 질량 20.257 mg, 밀도 905 가정), 둔덕(가운데 41 mm→벽 5~8 mm). 생성기 `pellet-model/sim_deme_pile.py`(`--target-shape slab --target-depth-m … --box-width-mm --box-length-mm --n-particles --shape lens …`, `:1864-1919`).
- **생성기 함정(W25-C 확인)**: `sim_deme_pile.py:435,446-470` 의 `margin = max(3×bounding_diameter, 10 mm)`(=13.5 mm/변)가 시드 발자국을 상자보다 작게 잘라(최대 274×171 등) `--target-depth-m` 만으로는 벽까지 꽉 찬 평평한 층이 안 나온다. 50,000알 생성은 1,200 s 제한에서 물리 0.40 s 미정착 rc 124(`pile_lens_20260910/driver.log`, `cell_n50000.log`).
- 알 수 산술 참고(메인·C 일치): 부피 = 가로×세로×깊이, 질량 = 부피×부피밀도(0.456 정착 envelope / 0.55 설정), 알 수 = 질량 ÷ 20.257 mg. W23 F `out/pile_budget.json`(310×220 기준 40 mm: 61,462 / 74,067) 와 W25-C `pile_budget_w25.json`(NTC106 기준).
- 전체 사이클 러너(rev32/rev34)는 더미 NPZ 의 `box_bounds_m` 로 트레이·도메인을 만들고, **수치 증거(numeric evidence, 설치본 격자 재구성)** 를 도메인과 binary64 로 대조하는 fail-closed 게이트가 있다(`sim_w13_full_cycle.py:241-255`, 도구 `recover_deme_lattice.py` — W21 ROI-300 증거 `numeric_inputs.json` 참고, 경로는 `claudedocs/runtime_logs/grasp_track/w21_cost_vs_particle_count_d493/` 와 rev32 `numeric_inputs.json`). W23 F 는 스쿱 도메인에서 "uncertified figureOutNV remainder branch" 로 fail-closed 됐다(`w23-sim-alignment/.../REPORT.md` §3-6).
- rev34(W25-A 워커, 별도 worktree `w25-rev34-fullcycle`)는 상자를 90° 돌려 놓는다(31 cm 변이 로봇 좌우). 더미 NPZ 의 축 규약(x=31 cm) 을 그대로 두고 러너가 돌릴지, NPZ 자체를 돌릴지는 A 가 정한다 — **이 과제는 NPZ 축 규약을 바꾸지 않는다**(x = 긴 변 310 mm, y = 220 mm 유지). 단 rev34 가 요구할 수 있는 "회전된 상자 좌표" 변환 함수(순수 numpy, 위치·쿼터니언 z 둘레 90°)는 준비해 두되 기본 산출은 원 규약.

## 2. 목표(Change)
1. **생성기 패치(사본)**: 시드 발자국이 상자 안쪽 벽까지 닿도록 `margin` 을 CLI 인자(`--seed-margin-mm`, 기본 = 기존값 유지)로 노출하고, 0~2 mm 로 두었을 때 초기 배치가 벽·알 겹침 없이 만들어지는지 **CPU 로 시드 단계만**(DEME 호출 전) 검사한다. 원본 대비 diff·AST·`py_compile`·기본값에서의 바이트 동일성(패치 적용 후 `--seed-margin-mm` 미지정 시 기존 동작과 시드 좌표 동일) 증명.
2. **알 수·명령표**: 상자 310×220(현재 실물, 안쪽 306×216 인지 310×220 인지 — 사용자 실측은 "윗단 바깥 31×22·벽 0.2" 이므로 안쪽 ≈306×216; 두 경우 모두 표) × 깊이 40 mm(±5 mm 감도) × 부피밀도 {0.456, 0.55} → 알 수·질량·생성 명령 초안·생성 시간 추정(로컬 4090 Laptop; 50k rc124 전례 → cap = 추정×2, 최소 3,600 s)·VRAM 미확인 표기. NTC106(301×198)도 한 표.
3. **정착 높이 관문**: 생성 후 "벽까지 꽉 찬 40 mm 평평한 층" 을 판정할 기준을 미리 적는다(5 mm 격자 높이지도 중앙값·p95, 벽 인접 칸 커버리지, 사용자 4 cm 와의 허용차 ±5 mm) — `roarm_rl/heightmap.py` 의 함수로 CPU 계산 가능해야 함. 기존 20k NPZ 에 그 함수를 적용해 예시 출력(둔덕이라 FAIL 이 나와야 정상 = 음성 대조).
4. **격자 증거 가능성**: rev34 후보 도메인(A 워커 `params_w25.json`·preflight 산출을 읽어 도메인 x/y/z 범위를 가져오되, 없으면 W19 도메인 x[-0.445,0.22] y[-0.175,0.461] z[0,0.588] 에 층 40 mm·상자 90° 회전을 반영한 추정 도메인 2안)에 `recover_deme_lattice.py` 를 CPU 로 돌려 **인증 분기인지 fail-closed 인지** 보고. 양성 대조(W21 ROI-300 입력 → 기록값과 binary64 동일) 먼저.
5. NPZ 회전 유틸(순수 numpy, 옵션): 위치·쿼터니언 z 둘레 +90°/−90°, `box_bounds_m` x↔y 교환, 단위 테스트(왕복 항등·박스 포함 검사).

## 3. 산출(Observable acceptance)
- `src/sim_deme_pile_w25.py` + `DIFF_pile_generator.patch` + `checks/`(py_compile·AST·기본값 시드 동일성·시드 겹침 검사 JSON).
- `pile_spec_flat40.json` + `pile_generation_commands.md`(실행 금지 초안) + `settle_gate_spec.md` + `settle_gate_check_on_20k.json`(음성 대조).
- `lattice_feasibility.json`(양성 대조 + rev34 후보 도메인 결과) — fail-closed 면 그 오류 원문 그대로.
- `rotate_pile_npz.py` + 테스트 결과(선택 항목 5).
- `REPORT.md`(한국어): 절차 → 수치 → 근거 → 한계 → 다음 승인 경계(사용자 결정 항목: 안쪽 치수 실측, 부피밀도, 생성 실행 승인).

## 4. 환경
- CPU 파이썬 `/home/cgxr/miniconda3/envs/roarm/bin/python`(DEME import 는 가능하나 solver 생성 금지). 설치 0.
- Orca: 질문은 preamble 의 `ask`. 완료 시 `worker_done` + `--report-path`. 자동 모드 오류("safety verdict 없음")로 턴이 끝나면 코디네이터가 메시지로 깨운다 — 재개 후 이어서 진행.
