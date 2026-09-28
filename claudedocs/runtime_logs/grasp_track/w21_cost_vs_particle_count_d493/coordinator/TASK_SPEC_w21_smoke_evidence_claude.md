# TASK_SPEC — W21 rev33b 스모크용 수치 증거 (CPU 전용, GPU 0)

작성 2026-09-19, 코디네이터 = Claude 메인 세션. 워커 = Claude `claude-opus-5`.
자기완결적이다. 여기 없는 행동은 하지 않는다.

## 0. 왜 이 과제가 생겼나 (코디네이터가 찾은 결함)

W20 의 `GPU_SMOKE_PACKAGE.md` 는 rev33b 스모크를 `--max-particles 300`(알 300개로 축소) +
`--numeric-evidence <rev33b/numeric_inputs.json>`(= 알 20,000개용 증거)로 짰다. **그대로 돌리면 죽는다.**

생산 코드 `sim_w13_full_cycle.py:241`~`:246` 은 증거의 고정 도메인과 이번 run 의 실제 도메인이
binary64 로 같지 않으면 `SystemExit("도메인이 수치 증거의 고정 입력과 다르다 — lattice 재사용 금지")` 다.
`--smoke` 도 `--max-particles` 도 이 검사를 **우회하지 않는다**(코디네이터가 소스로 확인).

코디네이터가 CPU 로 실측한 300알 도메인:

| 축 | 동결 증거(20,000알) | 실제(300알) |
|---|---|---|
| x | [-0.4449132573934248, 0.22] | 동일 |
| y 상한 | 0.4612066423643583 | **0.46122955972842306** |
| z 상한 | 0.5881087722936942 | **0.5877223000580789** |

이 두 값은 **W16 스모크가 실제로 실패했을 때 기록된 값과 정확히 같다**(브리핑 ② §2 가 인용한
`smoke_01/smoke.stderr.txt:1`). 즉 이 실패는 재현 확인된 것이다.

이어서 코디네이터가 300알 도메인에 `recover_deme_lattice.py` 를 돌려 **재구성 가능**을 확인했다
(rc 0, stderr 0바이트, `adjacent_additional_bits [0,0]` = 인증 분기 안).
따라서 이 과제는 **증거 파일 하나를 정식으로 만들고 게이트 통과를 증명**하면 끝난다.

**용어(처음 나올 때 풀이)**
- **수치 증거(numeric evidence)** — 설치된 DEME 이 그 도메인에서 쓰는 내부 격자값(`l`, `voxelSize` 등)을
  담은 별도 config. 물리 파라미터가 아니다.
- **fail-closed** — 증명되지 않으면 추정하지 않고 그 자리에서 멈추는 설계.
- **음성 대조(negative control)** — "틀린 입력을 넣으면 실제로 거부되는가"를 확인해, 통과가 공짜가 아님을 보이는 검사.
- **실제 경과 시간(wall-clock time)** — 사람이 시계로 재는 시간.

## 1. Target

**읽기 전용 입력 (수정 금지)**
- 동결 정본 `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w19_runpod_d487/rev32_frozen_copy/`
  (`src/sim_w13_full_cycle.py` sha `fc8d4a872222ea553e6b8e9c00df5162686092ad8a0f5f9ea579bd39487059f3`,
  `params_w13.json`, `numeric_inputs.json`)
- rev33b 미러 `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grasp_track/w21_cost_vs_particle_count_d493/inputs/rev33b_mirror/`
- 더미 NPZ (20,000알) `/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz`
  sha `659d6b0bc771678a0c7209d91f550edc933d03e41922245ea0adb64eeb818812`
- 격자 도구 `/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/{recover_deme_lattice.py,DEME_LATTICE_RECONSTRUCTION.md}`
- **참고 선례(같은 절차의 n=5,000 판)**: `/home/cgxr/orca/workspaces/RoArm_Project/w20-particle-count/claudedocs/runtime_logs/grasp_track/w20_decisions_d492/particle_count_screen/`
  의 `derive_domain_cpu.py`, `numeric_inputs_n5000_candidate.json`, `verify_candidate_gate.py`,
  `LATTICE_RECONSTRUCTION_N5000.json`, `CANDIDATE_GATE_VERIFICATION.json`.
  **이 절차와 파일 구조를 그대로 따른다**(키 이름·필드 구성 동일하게).

**쓰기 (이 worktree 안에만)**
`claudedocs/runtime_logs/grasp_track/w21_cost_vs_particle_count_d493/smoke_evidence/`

## 2. Change

1. **도메인 유도**: `derive_domain_cpu.py` 와 같은 방식(동결 소스 무수정 AST 절단 + 가짜 DEME)으로
   `max_particles=300` 조건의 도메인을 얻는다. **먼저 `max_particles=None`(20,000알)로 돌려
   동결 `numeric_inputs.json` 의 `rev11_recomputed_domain_m` 과 3축 binary64 일치를 증명**한 뒤
   300알로 넘어간다. 일치 증명이 없으면 "방법 미검증"으로 보고하고 멈춘다.
2. **격자 재구성**: 그 도메인으로 `recover_deme_lattice.py` 를 문서 명령 형식 그대로 CPU 실행.
   `REUSABLE` 이면 계속, `REJECTED` 면 거부 사유 원문을 인용해 기록하고 **거기서 끝낸다**(우회·추정 금지).
3. **증거 파일 작성**: `numeric_inputs_roi300_candidate.json` 을 n=5,000 판과 **같은 키 구조**로 만든다
   (`REUSABLE_FOR_REV11: true`, `rev11_recomputed_domain_m`, `values_if_usable`, 출처·비주장 필드 포함).
   **`params_w13.json` 은 절대 건드리지 않는다.**
4. **게이트 검증 + 음성 대조**: `verify_candidate_gate.py` 와 같은 방식으로
   생산 코드 `:236`~`:252` 대조를 실제로 태워서
   (a) 새 증거 + 300알 → **통과**
   (b) 동결 20,000알 증거 + 300알 → **SystemExit**
   (c) 새 증거 + 20,000알 → **SystemExit**
   세 경우를 모두 기록한다. (b)(c) 가 거부되지 않으면 게이트가 무력하다는 뜻이므로 **실패로 보고**한다.
5. **스모크 명령문 정정판**: `SMOKE_COMMAND_CORRECTED.md` 에 `--numeric-evidence` 를 새 파일로 바꾼
   최종 argv 를 적는다. `--settle-window-frame-dt-s` 는 **값을 받지 않는 플래그**임에 주의.
   상한은 sim 소프트 3,000 s / 러너 하드 3,600 s. **실행은 하지 않는다.**

## 3. Constraints

- **물리 실행 0 · GPU 0 · DEME 0 · 더미 생성 0 · 실물 0 · 설치 0 · commit/push 0.**
- 물성·형상·알 크기·경로·문 속도·보호선·`timestep_s` 1e-06·`cd_update_freq`·seed 460 **불변**.
  이 과제는 **수치 증거(별도 config)만** 만든다.
- 동결 사본·rev33b 미러·더미 NPZ·격자 도구는 **읽기 전용**. 작업 전후 sha256 으로 불변 증명.
- `isaaclab` conda 환경에 **설치 금지**(핀 `numpy==1.26.0`·`psutil==5.9.8`, D326). `roarm` 환경 사용.
- 상태 원장(`START_HERE.md`, `claudedocs/{DECISIONS,DECISIONS_ACTIVE,EXPERIMENT_LEDGER,LEDGER_RECENT}.md`,
  `claudedocs/relay/`)에 **한 줄도 쓰지 않는다**. hook 이 기계적으로 차단한다.
- 폴더 forward-only. 기존 파일 이동·개명 금지.
- "없다/최초" 주장 금지(HARD RULE #4). 승격·성공 선언 금지.

## 4. Ownership

`claudedocs/runtime_logs/grasp_track/w21_cost_vs_particle_count_d493/smoke_evidence/` **안에서만** 쓴다.

## 5. Observable acceptance

1. `PRESERVATION_BEFORE.json` / `PRESERVATION_AFTER.json` — 읽기 전용 입력 전체 sha256, **불일치 0**.
2. `DOMAIN_DERIVATION_ROI300.json` — 20,000알 방법 검증(3축 일치 여부) + 300알 도메인 + 방법 설명.
3. `LATTICE_RECONSTRUCTION_ROI300.json` — `REUSABLE`/`REJECTED` + 도구 stdout·stderr 원문 + 도구 sha + argv.
4. `numeric_inputs_roi300_candidate.json` — REUSABLE 인 경우에만.
5. `GATE_VERIFICATION_ROI300.json` — §2-4 의 (a)(b)(c) 세 경우 결과.
6. `SMOKE_COMMAND_CORRECTED.md` — 실행 0.
7. `REPORT_smoke_evidence.md` — 한국어, **관찰 가능한 절차 → 수치 → 근거 파일 → 한계·다음 승인** 순서.
   영어 약어는 처음 나올 때 풀어 쓴다.
8. `manifest.json` — 산출 전 파일 경로·바이트·sha256.

## 6. 보고

`worker_done` 정확히 한 번, 3문장 요약 + `--outcome succeeded|failed` + 두 lifecycle ID.
`REJECTED` 도 `succeeded`(관문을 제대로 통과시킨 것이 과제다). `--report-path` 는 7번 절대경로.
