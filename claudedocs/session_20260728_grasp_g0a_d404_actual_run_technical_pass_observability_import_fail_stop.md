# D404 actual run — 기술 체인 최초 전체 PASS, 관측성 층 import 결함 FAIL_STOP

Date: 2026-07-28 저녁 KST. 이번 case의 신규 변수:
`[authored_derivative_gate_contract_repair_v1]` (D404 static prep에서 승인, 이 문서는
그 runtime 결과). **D404 attempt1 소모·동결 (worker 1회, retry 0, 총 32.9s).**

## 1. 무엇을 왜

유저가 tuple SHA `0d06cc2d3995d80224aaa5289fde2b1e0dacf09ad54e45758fcd54d89220b196`을
명시 승인하여 D404 attempt1을 호스트 경계에서 1회 실행했다. 목적: D403에서 FAIL한
authored-derivative 게이트의 계약 수리 4건이 실제 Isaac 런타임에서 D400 preflight
체인을 통과시키는지 판정.

## 2. 실행 절차 (감사 가능 step-by-step)

1. 실행 직전 점검 재수행 — 4항목 전부 PASS: `/dev/nvidiactl` 존재(부팅 13:32 생성),
   셸 pid 1495600(호스트 범위), `HEAD==origin/master==a69a96d`, dirty 28 = prereg
   allowlist 28 완전 일치(양방향 차집합 0), 잔존 Isaac/Kit 프로세스 0.
2. isaaclab env `python -B` + `--approved-tuple-sha256 0d06cc2d...b196`으로
   controller 1회 실행. exit 1.
3. 사후 진단은 read-only로만 수행 (재실행 없음).

## 3. 정량 결과 (source: `claudedocs/runtime_logs/grasp_track/g0a_d404/attempt1_d403_authored_derivative_gate_contract_repair/`)

### 3.1 수리 4건 라이브 판정 — 전부 성공

| 게이트 (D403 실패 지점) | D403 결과 | D404 라이브 |
|---|---|---|
| typed authored readback | 실패 체크 2 / attr 2 | **PASS — 체크 9/9, attr 7/7** |
| composed semantic diff | 65 mismatch | **PASS — 0 mismatch** |

(`d400_worker_raw_summary.json` `derivative_asset.sdf_readback` /
`composed_semantic_diff_gate`; derivative pass=true, A64 64개 비활성.)

### 3.2 기술 체인 — 33 phase 사상 최초 전체 완주 (`technical_pass: true`)

- Isaac 기동 5.2s (phase 0.34→5.50s); derivative 저작 6.6s 시점; 수리된
  readback/semantic 게이트 구간 ~20s(6.60→26.52s) PASS.
- **SDF cook 실제 발생**: 태스크 136 스케줄/136 완료, cache hit 135·miss 1,
  running 0 드레인 (`cooking`, `post_query_cooking_gate` PASS).
- **PhysX property query 2회**: link5 collider 65, gripper_link collider 66 전부
  `VALID`; gripper SDF mesh collider(`node_STL_BINARY_/mesh`)도 VALID.
- mass invariance PASS, counter gate PASS(등록 36키; physics step/contact/q5 전부 0),
  cleanup/detach/stagecache erase 정상, worker `exception: None`.
- kit log 71행 오류 0; supervisor returncode 0, pgid 1496499, 잔존 서명 0;
  phase 감사 22/22 체크 PASS (순서·소유자·단조시간·ordinal 전부 정상).

### 3.3 FAIL 지점 — 관측성 층 (기술 완주 후)

- verdict: **`D400_OBSERVABILITY_OR_COMPLETION_INTEGRITY_FAIL_STOP`**
  (`pass:false`, `runtime_preflight_pass:false`, `technical_pass:true`).
- `observability_error`: D401 controller `run_runtime`(:581) →
  preflight.py:2779 `from roarm_rl.rerun_contract import validate_rerun_artifact`
  → `ModuleNotFoundError: No module named 'roarm_rl'`.
- decision board PNG(1920×1080, 526KB)는 정상 생성; RRD/RBL/validation/screenshot/
  manual inspection은 미도달 (`rerun: null`, `manual_inspection: null`).

### 3.4 사후 진단 (read-only)

- `roarm_rl/`은 repo 루트에 실재; isaaclab python이 cwd=repo 루트에서
  `import roarm_rl` 성공 (`roarm_rl/__init__.py`는 gymnasium import + lazy
  gym.register 4건뿐 — pxr/Isaac 부작용 없음).
- 원인: `python -B sim_scripts/xxx_controller.py` 경로 실행 시 `sys.path[0]`이
  `sim_scripts/`라 repo 루트 미포함 → launch 구성 결함 (모듈 부재 아님).
- 이 분기는 D400(정적)→D401(freeze 정지)→D402(GPU 정지)→D403(readback 게이트
  정지) 내내 **라이브 실행 0회** — D403 durable lesson의 층별 반복.

## 4. 판정과 다음 단계

- 과학 질문 여전히 미측정: `scientific_or_physics_verdict=null`, `g0a_pass=false`
  (D400 preflight authority 한계상 이 단계에 과학 판정 없음).
- 이 FAIL은 관측성 완주 계약(D341) 위반이며 Isaac/PhysX/SDF/수리 4건의 실패가
  아니다 — 그렇게 부르지 말 것.
- durable lesson은 DECISIONS **D404** 등재.
- 다음 최소 rung = **D405 [d404_observability_import_path_repair]** — 유저
  2026-07-28 순차 지시("다음 최소 승인할테니 step-by-step으로 순차적으로
  사고하면서 진행해")로 승인, D405 attempt1 1회로 소진.
