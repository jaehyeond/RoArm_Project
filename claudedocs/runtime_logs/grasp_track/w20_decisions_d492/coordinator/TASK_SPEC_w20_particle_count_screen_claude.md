# TASK_SPEC — W20 설계 case ② 알 개수: CPU 전용 사전 스크리닝 (GPU 실행 0)

작성 2026-09-19, 코디네이터 = Claude 메인 세션(RoArm_Project master). 워커 = Claude `claude-opus-5`.
이 과제서는 자기완결적이다. 여기 없는 행동은 하지 않는다.

## 0. 이 과제의 위치 (왜 지금 이것인가)

사용자가 설계 브리핑 ②(알 개수 비용 case)를 **수단 (ii) 소형 슬랩**으로, 설계 브리핑 ①(문 이음새)보다
**먼저** 실행하기로 승인했다. 그런데 브리핑 ② §9 위험 1 과 §8 이 명시한 대로, GPU 실행보다 **먼저**
CPU 로 걸러야 하는 관문이 하나 있다:

> 새 도메인의 DEME 격자(lattice) 재구성이 거부되면 그 셀은 **실행 불가**다.
> (`DEME_LATTICE_RECONSTRUCTION.md:81`~`:89`: "A different domain that triggers another
> `figureOutNV` branch is intentionally rejected ... requires a fresh installed-binary audit")

이 과제는 **그 관문만** 통과시키는 CPU 작업이다. 물리 실행도, 더미 생성도, 코드 변경도 하지 않는다.

**용어(처음 나올 때 풀이)**
- **도메인(domain)** — DEME 엔진이 계산할 3차원 상자의 범위. 더미 크기와 공구 스윕이 정한다.
- **격자(lattice) 재구성** — 설치된 DEME 이 그 도메인에서 내부적으로 쓰는 수치값(`l`, `voxelSize`,
  내부 step 수 `N` 등)을 소스 증거로부터 되살리는 CPU 절차. 이 값이 없으면 사전점검이
  **fail-closed**(증명 안 되면 추정하지 않고 그 자리에서 멈춤)된다.
- **실제 경과 시간(wall-clock time)** — 사람이 시계로 재는 시간. 시뮬 속 물리 시간과 구분한다.

## 1. Target (범위 안 파일·환경)

**읽기 전용 입력 (절대 수정 금지)**
- 동결 코드 정본: `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/rev32_frozen_copy/`
  (`src/`, `params_w13.json`, `numeric_inputs.json`, `criteria.json`, `REVISION_PIN.json`)
- 더미 NPZ 2개 (외장 SSD 아님, 로컬 worktree):
  · n=20,000 `/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz`
    sha256 `659d6b0bc771678a0c7209d91f550edc933d03e41922245ea0adb64eeb818812`
  · n=5,000 같은 폴더 `…_slab_n5000_seed460.npz`
    sha256 `ac29d53370bd8379ffac554d991e884f4b8335b812a8740891a68d028aa9d294`
  · 같은 폴더 `cells_table.json`, `run_cells.sh`
- 격자 재구성 도구: `/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/`
  의 `recover_deme_lattice.py`, `DEME_LATTICE_RECONSTRUCTION.md`, `SOURCE_EVIDENCE.md`
- fail-closed 실증 1줄: `/home/cgxr/orca/workspaces/RoArm_Project/w16-profiling/claudedocs/runtime_logs/grasp_track/w16_profile_d486/smoke_01/smoke.stderr.txt:1`
  (외장 SSD 심링크 경유 — 못 열면 "미확인"으로 적고 넘어간다)

**쓰기 (이 worktree 안에만)**
`claudedocs/runtime_logs/grasp_track/w20_decisions_d492/particle_count_screen/`

## 2. Change (만들어 낼 결과)

### 2-1. n=5,000 슬랩의 실제 도메인 확정
rev32 생산 코드가 더미 NPZ 로부터 도메인을 유도하는 **바로 그 경로**를 CPU 로 재현해
`dom_x`, `dom_y`, `dom_z` 3쌍을 얻는다. 경로는 브리핑 ② §2 후보 (i) 설명이 가리키는
`sim_w13_full_cycle.py` 의 더미 표면 높이(`:183`,`:185`) → 웨이포인트(`:200`) → 스윕 →
도메인(`:223`,`:224`,`:225`) → 대조 fail-closed(`:246`) 연쇄다.

⚠️ **알려진 함정**: `sim_w13_full_cycle.py` 는 모듈 수준에서 DEME 을 import 할 수 있다. GPU/DEME 을
띄우지 말고 도메인 유도부만 실행해야 한다. 어떻게 했는지(예: AST 로 함수 추출, import 스텁, 부분 실행)를
보고서에 **방법과 함께** 적고, **n=20,000 에 대해 같은 방법을 돌려 기존 `numeric_inputs.json` 의
도메인 값과 일치함을 먼저 증명**한다. 이 일치 증명이 없으면 n=5,000 결과는 신뢰할 수 없다 —
그 경우 "방법 미검증"으로 보고하고 멈춘다.

### 2-2. 격자 재구성 시도 (핵심 관문)
확정한 n=5,000 도메인에 대해 `recover_deme_lattice.py` 를 `DEME_LATTICE_RECONSTRUCTION.md:60`~`:71`
의 명령 형식 그대로 CPU 실행한다. 결과는 **둘 중 하나**이며 **어느 쪽이든 유효한 결과**다:
- **REUSABLE** — 새 `numeric_inputs.json` 후보를 이 worktree 산출 폴더에 쓴다
  (**`params_w13.json` 은 절대 건드리지 않는다** — 브리핑 ② §8 4항).
- **REJECTED** — 거부 사유 원문(`figureOutNV` 분기 등)을 그대로 인용해 기록한다.
  이것이 브리핑 ② §5 의 사전 등록 `CELL_UNRUNNABLE` 이다. **우회·추정·완화 금지.**

### 2-3. n=10,000 은 "무엇이 필요한지"만 적는다
n=10,000 더미 NPZ 는 **존재하지 않는다**. 생성은 DEME 동역학을 쓰므로 이 과제 범위 밖(GPU 승인 대상)이다.
생성하지 말고, 필요한 명령·예상 시간·상한만 적는다(W7 기록: n=5,000 147.06 s, n=20,000 800.40 s,
드라이버 상한 `run_cells.sh:15` `timeout 1200`).

### 2-4. GPU GO 꾸러미 초안 (실행하지 않는다)
2-2 가 REUSABLE 일 때만: 셀 P2(n=5,000, `settle`~`reclose`)를 돌릴 **정확한 명령문**, 상한 13,204 s,
사전 등록 기록 항목(브리핑 ② §5 의 비용 축 1~5 · 물리 감시 축 6~9), 산출 경로를 한 파일에 적는다.
**실행은 사용자 GO 뒤 별도 과제다.**

## 3. Constraints (불변·금지)

- **물리 실행 0 · GPU 0 · DEME 0 · 더미 생성 0 · 실물 0 · 설치 0 · commit/push 0.**
- 물성·펠릿 형상(7구 렌즈 4.5/3.8/2.5 ring 6)·경로·문 속도(22.5 °/s)·보호선(pinch 3.0 N,
  servo stall 1.96×0.9, pop 5 m/s, stop 20 m/s)·`timestep_s` 1e-06·`cd_update_freq` 상한 20·
  seed 460 은 **바꾸지 않는다**. 이 case 의 신규 변수는 **알 개수 N 하나**다.
- rev32 동결 사본과 원자료 NPZ/JSON 은 **읽기 전용**. 실행 전후 sha256 을 찍어 불변을 증명한다.
- `isaaclab` conda 환경에 **설치 금지**(핀 `numpy==1.26.0`·`psutil==5.9.8` 보호, D326).
- 상태 원장(`START_HERE.md`, `claudedocs/{DECISIONS,DECISIONS_ACTIVE,EXPERIMENT_LEDGER,
  LEDGER_RECENT}.md`, `claudedocs/relay/`)에 **한 줄도 쓰지 않는다**. 코디네이터만 소유한다.
- 폴더는 forward-only. 기존 파일·폴더 이동·개명 금지.
- "없다/최초" 류 주장 금지(HARD RULE #4). 단일 실행 차이를 변수 효과로 읽지 않는다(D490).
- 장시간 백그라운드 작업이 필요하면 하네스 밖 `setsid` 로 띄운다(D487). 다만 이 과제는 CPU 수분 규모다.

## 4. Ownership (편집 가능 경계)

이 worktree 의 `claudedocs/runtime_logs/grasp_track/w20_decisions_d492/particle_count_screen/` **안에서만**
파일을 만든다. 그 밖의 어떤 파일도 편집하지 않는다.

## 5. Observable acceptance (완료 증거)

아래 전부가 산출 폴더에 있어야 완료다.

1. `PRESERVATION_BEFORE.json` / `PRESERVATION_AFTER.json` — 읽기 전용 입력 전체의 sha256,
   **불일치 0** 이어야 한다.
2. `DOMAIN_DERIVATION.json` — n=20,000 검증(기존 `numeric_inputs.json` 도메인과 일치 여부)과
   n=5,000 도메인 3쌍. 방법 설명 포함.
3. `LATTICE_RECONSTRUCTION_N5000.json` — `REUSABLE` 또는 `REJECTED` + 도구 stdout/stderr 원문 인용,
   도구 파일 sha256, 실행 명령 argv.
4. REUSABLE 인 경우에만 `numeric_inputs_n5000_candidate.json`.
5. `N10000_PREREQUISITES.md` — 2-3 내용.
6. REUSABLE 인 경우에만 `GPU_GO_PACKAGE_P2.md` — 2-4 내용(실행 0).
7. `REPORT_particle_count_screen.md` — 한국어. 순서는 **관찰 가능한 절차 → 수치 → 근거 파일 →
   한계·다음 승인**. 영어 약어는 처음 나올 때 풀어 쓴다. 승격·성공 선언 금지.
8. `manifest.json` — 산출 전 파일의 경로·바이트·sha256.

## 6. 보고

`worker_done` 은 정확히 한 번, 3문장 요약 + `--outcome succeeded|failed` + 두 lifecycle ID.
**REJECTED 도 `succeeded`** 다(관문을 제대로 통과시킨 것이 과제다). 과제 자체를 못 끝낸 경우만 `failed`.
`--report-path` 는 위 7번 파일의 절대경로.
