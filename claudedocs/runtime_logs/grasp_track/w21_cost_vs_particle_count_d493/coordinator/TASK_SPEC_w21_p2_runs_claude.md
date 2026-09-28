# TASK_SPEC — W21 P2 (알 5,000개) 로컬 GPU 실행 2회 — 사용자 승인 완료

작성 2026-09-19, 코디네이터 = Claude 메인 세션. 워커 = Claude `claude-opus-5`, 역할 = `deme-runner` 성격.
자기완결적이다. 여기 없는 행동은 하지 않는다.

## 0. 승인 범위 (명확히)

사용자가 **로컬 GPU 전용**으로 이 2회 실행을 승인했다. RunPod 은 쓰지 않는다 —
비교 기준(20,000알, W16 run_02)이 **이 로컬 GPU에서** 측정됐고, pod 은 로컬보다 물리 1초당
1.18~1.72배 빨라서(W19 실측) pod 에서 돌리면 기계 차이가 알 개수 효과에 섞인다.

승인된 것: **P2 셀 2회, 각 `settle`~`reclose`, rev32 동결 코드, 알 5,000개.**
승인 범위 밖(절대 금지): RunPod, 추가 셀, 다른 N, 다른 구간, 코드·파라미터 변경, 재시도,
상한 연장, 실물, 설치, commit/push.

**왜 2회인가**: 비교 기준의 비용 표본이 **하나뿐**이다(W16 run_01 은 `RUN_STATUS` 가 `running` 에서
멈춘 미완주 — Orca 하네스가 백그라운드 GPU 작업을 SIGTERM 한 그 사건이다). 변동폭 근거가 없으면
P2 와 기준의 차이를 판정에 쓸 수 없다. 기준을 반복하면 약 3시간, P2 를 반복하면 약 1.5시간으로
같은 정보를 얻는다. 포획량 표본도 2개가 되어 물리 감시 축에 쓰인다.

**용어(처음 나올 때 풀이)**
- **setsid** — 프로세스를 새 세션으로 분리해 띄우는 명령. 부모(에이전트 하네스)가 죽거나
  신호를 보내도 자식이 함께 죽지 않게 한다.
- **소프트 상한** — 시뮬 자신이 step 경계에서 우아하게 멈추고 원자료를 정상 마무리하는 한계.
  **하드 상한** — 러너가 프로세스 그룹에 SIGTERM → SIGKILL 하는 한계.
- **실제 경과 시간(wall-clock time)** — 사람이 시계로 재는 시간. 시뮬 속 물리 시간과 구분한다.

## 1. Target

**읽기 전용 입력 (수정 금지)**
- 동결 코드 `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/rev32_frozen_copy/`
  · `src/sim_w13_full_cycle.py` sha `fc8d4a872222ea553e6b8e9c00df5162686092ad8a0f5f9ea579bd39487059f3`
  · `params_w13.json`, `criteria.json`, `REVISION_PIN.json` — **전부 무수정**
- 수치 증거 `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w21_cost_vs_particle_count_d493/inputs/numeric_inputs_n5000_candidate.json`
  sha `35e9669333c798dee12813cd1a64855149f633137dedc915179c56f5c036fdbb`
  (W20 에서 격자 재구성 `REUSABLE` + 생산 게이트 통과 + 음성 대조까지 검증된 파일)
- 더미 NPZ (알 5,000개) `/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n5000_seed460.npz`
  sha `ac29d53370bd8379ffac554d991e884f4b8335b812a8740891a68d028aa9d294`
- 비교 기준(재실행 0, 읽기만) `/home/cgxr/orca/workspaces/RoArm_Project/w16-profiling/claudedocs/runtime_logs/grasp_track/w16_profile_d486/run_02/`
  (`EXECUTION_RECEIPT.json` rc 0 · `wall_s` 11021.63, `RUN_STATUS.json` `completed_rc0`)
- GPU 점유 기록 `.../w21_cost_vs_particle_count_d493/GPU_OCCUPANCY_BASELINE.json`

**쓰기**
`claudedocs/runtime_logs/grasp_track/w21_cost_vs_particle_count_d493/run_n5000_rep1/` 과 `.../run_n5000_rep2/`
(+ 같은 case 폴더의 `runs_meta/`)

## 2. Change — 실행 절차 (순서 엄수)

### 2-1. 사전점검 (GPU 0)
1. 위 읽기 전용 입력 전체 sha256 기록(`PRESERVATION_BEFORE.json`). **하나라도 불일치면 중단.**
2. 출력 폴더 `run_n5000_rep1`/`rep2` 가 **비어 있음**을 확인. 비어 있지 않으면 중단(덮어쓰기 금지).
3. 로컬에 다른 DEME/`sim_w13_full_cycle` 프로세스가 **없음**을 확인. 있으면 중단.
4. `nvidia-smi` 로 GPU 점유를 실행 직전 기록(`runs_meta/gpu_before_rep1.txt` 등).
   ⚠️ 현재 `mcp_memory_service.server` 7개가 각 390 MiB·합 2,900 MiB 를 물고 계산 점유 20 % 가 상시다.
   **이 프로세스들을 죽이지 마라** — 세션 도구가 의존할 수 있고, 코디네이터가 환경을 바꾸지 않기로 결정했다.
   기록만 한다.

### 2-2. rep1 실행 (GPU)
```sh
PYTHONDONTWRITEBYTECODE=1 setsid \
/home/cgxr/miniconda3/envs/roarm/bin/python -B \
  <REV32>/src/sim_w13_full_cycle.py \
  --params           <REV32>/params_w13.json \
  --pile             <PILE_N5000> \
  --out              <CASE>/run_n5000_rep1 \
  --seed             460 \
  --numeric-evidence <CASE>/inputs/numeric_inputs_n5000_candidate.json \
  --stop-after-phase reclose \
  --max-wall-s       13204 \
  > <CASE>/run_n5000_rep1/stdout.txt 2> <CASE>/run_n5000_rep1/stderr.txt
```
- **`setsid` 필수**(D487: Orca 하네스가 백그라운드 GPU 작업을 SIGTERM 한 전례 = W16 run_01 미완주).
- 소프트 상한 13,204 s = 기준 실측 11,003.3 s × 1.2. **연장·재시도 0.**
- 하드 상한은 워커가 별도 감시로 13,804 s(유예 600 s)에 프로세스 **그룹** SIGTERM → SIGKILL.
- 실행 중에는 진행만 관찰한다. 상한 초과·중단도 **유효한 결과**이며 그대로 보고한다.

### 2-3. rep2 실행
rep1 이 **완전히 끝난 뒤** 같은 명령으로 `run_n5000_rep2` 에 1회. **동시 실행 절대 금지.**
seed·입력·인자 전부 rep1 과 동일하다 — 바뀌는 것은 출력 폴더뿐이다(같은 입력 반복이 목적).

### 2-4. 실행 후 기록 (각 rep 마다)
`runs_meta/RECEIPT_rep{1,2}.json` 에: argv 전체, 시작·종료 UTC, 실제 경과 시간, 종료코드,
신호 수신 여부, stdout/stderr 의 sha256 과 바이트, 실행 전후 GPU 점유, 산출 파일 목록 sha256.
**타임아웃은 성공이 아니다** — 자동 종료 분류를 쓰지 말고 stderr 원문과 대조해 판정한다(D486).

### 2-5. 사전 등록 기록 항목 추출 (셀마다)
**비용 축 1~5**
1. 잠재 접촉쌍 (min / median / max) — `scalar_engine_num_contacts`.
2. 물리 1초당 실제 경과 시간 (단계별 + 합계).
3. 내부 step 1회당 마이크로초 (median) — 설치본 누산 재현값이지 엔진 카운터가 아니다.
   **셀 간 비교에만 쓰고 절대값을 성능 스펙으로 인용 금지.**
4. 충돌탐색 갱신주기 (min / median / max).
5. 총 실제 경과 시간 · rc · 완주 여부.

**물리 감시 축 6~9**
6. 포획량 — `reclose_end` 시점 공구 내부 클럼프 수와 질량.
7. 더미 표면 높이(취점 반경 안 heightmap median/max)와 `slab_settled_depth_estimate_m`.
8. 문 정지 각도와 정지 이유.
9. 최대 입자 속도, 5 m/s 경고 sync 수, 20 m/s 강제정지 발동 여부.

**사전 등록 판정** — 실행 뒤 문턱을 바꾸지 않는다.
- `COST_SCALES_WITH_N`: 접촉쌍 median 과 물리 1초당 실제 경과 시간이 **둘 다** N 감소에 따라 단조 감소.
- `COST_NOT_SCALING`: 둘 중 하나라도 단조성 깨짐.
- `CELL_UNRUNNABLE`: 상한 초과 / 엔진 중단 / fail-closed.
- 셀 수가 적어 **기울기나 `비용 ∝ N^α` 적합은 주장하지 않는다.** 단조성과 대략의 배율만.
- 포획량 ±15 % 는 **관찰 밴드이며 판정 문턱으로 승격하지 않는다**(사용자 미승인). 밴드 밖이면
  `PHYSICS_ALSO_CHANGED` 로 표시하고 비용 값은 보고하되 "N 을 줄여도 된다"의 근거로는 쓰지 않는다.

### 2-6. rep1 vs rep2 변동폭 + 기준 대비
- rep1 vs rep2 의 비용 축 차이를 **실행 간 변동폭**으로 제시한다(같은 조건 반복).
- 기준(20,000알 W16 run_02 = 11,021.63 s wall / 11,003.3 s 구간)과의 배율을 계산하되,
  ⚠️ **다음 한계를 반드시 함께 적는다**: 기준의 당시 GPU 점유는 기록이 없다. 지금은 20 % 상시 점유가 있다.
  따라서 기준 대비 배율은 이 confound 를 안고 있으며, rep1-rep2 내부 비교만 조건이 동일하다.

## 3. Constraints

- **RunPod 금지. 추가 셀 금지. 재시도 금지. 상한 연장 금지.**
- 물성·형상·알 크기·경로·문 속도·보호선·`timestep_s` 1e-06·`cd_update_freq`·seed 460 **불변**.
  이 case 의 신규 변수는 **알 개수 N 하나**다.
- 동결 사본·NPZ·수치 증거·기준 run 은 **읽기 전용**. 작업 전후 sha256 불변 증명.
- `isaaclab` 환경 설치 금지(D326). `roarm` 환경 사용. 새 패키지 설치 0.
- 상태 원장·relay 에 **한 줄도 쓰지 않는다**(hook 이 차단한다).
- 폴더 forward-only. 기존 파일 이동·개명 금지.
- **승격·성공 선언 금지.** "전체 사이클 성공"류 문장 금지. "없다/최초" 금지(HARD RULE #4).
- 단일 실행 차이를 변수 효과로 읽지 않는다(D490).
- `mcp_memory_service.server` 프로세스를 **죽이지 마라**.

## 4. Ownership

`w21_cost_vs_particle_count_d493/{run_n5000_rep1,run_n5000_rep2,runs_meta}/` 안에서만 쓴다.

## 5. Observable acceptance

1. `PRESERVATION_BEFORE.json` / `PRESERVATION_AFTER.json` — 불일치 0.
2. `run_n5000_rep1/`·`run_n5000_rep2/` 원자료 + stdout/stderr.
3. `runs_meta/RECEIPT_rep1.json`·`RECEIPT_rep2.json` + `gpu_before/after_rep{1,2}.txt`.
4. `runs_meta/PREREGISTERED_METRICS.json` — §2-5 의 9개 축, 셀별.
5. `runs_meta/VARIATION_AND_BASELINE.json` — §2-6 (변동폭 + 기준 배율 + confound 명시).
6. `REPORT_p2_runs.md` — 한국어, **관찰 가능한 절차 → 수치 → 근거 파일 → 한계·다음 승인** 순서.
   영어 약어는 처음 나올 때 풀어 쓴다. 사전 등록 판정 중 하나를 고르되 승격 선언은 하지 않는다.
7. `manifest.json` — 산출 전 파일 경로·바이트·sha256.

## 6. 보고

`worker_done` 정확히 한 번, 3문장 요약 + `--outcome succeeded|failed` + 두 lifecycle ID.
상한 초과·`CELL_UNRUNNABLE` 도 **`succeeded`** 다(정직한 실패는 유효한 결과). 과제를 못 끝낸 경우만 `failed`.
`--report-path` 는 6번 절대경로. 긴 실행 중에는 과제서가 정한 주기로만 heartbeat 를 보낸다.
