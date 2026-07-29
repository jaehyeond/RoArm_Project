# D404 static prep — authored-derivative 게이트 계약 수리 4건 구현·정적 검증 완료 (runtime 미실행, 승인 대기)

Date: 2026-07-28 밤 KST. 이번 case의 신규 변수: `[authored_derivative_gate_contract_repair_v1]`
(정확히 1개 — D403 관측 실패 4건에 대한 최소 reactive 수리 일괄).
**Runtime 실행 0회. 승인 범위는 "구현 착수 + 정적 준비까지" (유저 2026-07-28) — 소진 완료.
실제 1회 실행은 아래 tuple SHA에 대한 별도 명시 승인 필요.**

## 1. 무엇을 왜

D403 actual run은 동결 D400 worker의 `_author_sdf_derivative`(worker.py:1378)에서
게이트 코드 자체의 계약 결함 4건으로 FAIL_STOP했다 (Isaac/PhysX/SDF/드라이버 실패 0건,
DECISIONS D403). D404는 그 4건만을 D402 `_install_counter_authority_repair`와 동일한
**함수-객체 교체 패턴**으로 수리한다: D404 worker wrapper가 체인(D404→D403→D402→D401→D400)을
해시 pin으로 로드한 뒤, 로드된 동결 D400 worker 모듈의 `_sdf_prim_readback`과
`_normalize_allowlisted_semantics` **함수 객체 2개만 교체**한다. 동결 파일 수정 0,
check 키 이름/record 키 집합 변경 0, 과학·기하 변경 0.

## 2. 실행 순서 (감사 가능 step-by-step)

1. **정찰**: 동결 체인 wrapper 6종 + D400 worker 대상 함수 전문
   (worker.py:961-1520) + D401 controller merge 로직(169-207) + D400 preflight 승인
   게이트(462-595, REGISTERED_STATIC_NEGATIVE_IDS 102-121) + D403
   prereg/attestation 구조 정독. check 키 이름의 외부 소비자 부재 확인(grep 0건).
2. **설치 스키마 실측** (NVIDIA 검증 규칙): 설치 usdPhysics `schema.usda:285`
   `bool physics:collisionEnabled = true` — **uniform 키워드 없음(varying)**;
   `:293` `rel physics:simulationOwner`; `:317` `uniform token physics:approximation`.
   설치 PhysxSchema `schema.usda:1049-1129` — physxSDFMeshCollision 7속성 **전부 uniform**.
   두 파일 sha256을 prereg `installed_nvidia_primary_sources`에 신규 pin.
3. **prereg 작성** (빌더 스크립트, 상속 pin 전부 disk에서 재계산·assert 후 생성):
   `d404_preregistration.json` sha256 `4514e824a93902e1b69715df923d43a6c8b86790777b913f3e8c72434b254db0`.
   allowed_dirty_paths 28, additional_frozen_repo_inputs 23(=D403의 13 + D403 attempt 증거 10),
   inherited_science_contract 18(=D403의 13 + d403 layer 5), installed sources 7.
   authority: `actual_runtime_requires_new_explicit_approval=true` (순차 고속 지시는
   D403 attempt로 소진되었음을 명기).
4. **wrapper 2종 작성**:
   - `sim_scripts/cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_worker.py`
     — 수리 4건 + 3중 seam hook(`d403._load_frozen_d402_worker` →
     `d402._install_item_accessor_repair` → `d401._load_frozen_d400_worker`;
     D402 item 수리 먼저 설치 후 동일 모듈 객체에 D404 수리 추가 설치).
   - `..._controller.py` — 순수 경로/provenance rebind (D403 host gate 상속).
5. **정적 fixture 실행** (scratchpad runner, isaaclab python -B, read-only):
   **체크 29/29 + 양성 13/13 + 음성 43/43 전부 PASS**. 핵심은 §4.
6. **4-lens 적대적 리뷰** (workflow `wf_43c263d8-f07`, 4 agents, 607k tokens):
   **final blocker 0**, warnings 8건(§5, 전부 non-blocking, actionable 2건 조치 완료).
7. **attestation + tuple 작성** 후 동결 D400 `_validate_approval_tuple`(preflight.py:462-595)
   로직 **오프라인 복제 10/10 PASS**:
   - `d404_reviewed_script_attestation.json` sha256
     `d914c39d727f5fd0718ef77829580af825af16d4be4775b6123a444e262d28d6`
   - `d404_proposed_runtime_hash_tuple.json` sha256
     **`0d06cc2d3995d80224aaa5289fde2b1e0dacf09ad54e45758fcd54d89220b196`**
   - negative 52행(≥30, 등록 18 id superset), positive 13행, zero counters exact,
     controller/worker binding exact.

## 3. 수리 4건 (모두 설치 스키마/float32 의미론에서 도출 — D403 결정 준수)

| # | 동결 결함 (파일:라인) | 수리 (D404 worker wrapper) |
|---|---|---|
| 1 | collisionEnabled `uniform is True` 요구 (D400 worker.py:1480-1487) vs 설치 스키마 varying 선언(schema.usda:285) — 원리상 통과 불가 | 기대를 **varying**(`uniform is False`)으로 정렬. authored/non-custom/no-timesamples/no-connections 유지. approximation·7 SDF 속성의 uniform 기대는 스키마 실측대로 유지 |
| 2 | float 등호 `row["value"] == 0.01` (worker.py:1447-1460) — float32 왕복 0.009999999776482582라 통과 불가; bits 체크는 별도 PASS | float+bits 스펙 속성의 값 권위를 **`float32_bits_hex == expected_bits`**로. int/token/bool은 값 등호 유지 |
| 3 | normalizer가 attr `value`만 마스킹(worker.py:1072-1075), metadata `default`(true→false) 미마스킹 → 64행 유출 | A64 경로 collisionEnabled의 `default` metadata 항목 값을 동일 마커로 마스킹. **항목 부재/타 속성/타 경로는 여전히 fail-closed** |
| 4 | 관계(relationship) 모델 부재 — API Apply가 추가한 builtin `rel physics:simulationOwner`(schema.usda:293) → mesh 1행 유출 | mesh 경로에서 이름 일치 + **authored target 0개**인 그 관계만 필터. target 있으면/타 관계면/타 경로면 fail-closed |

교체 메커니즘: 동결 호출부는 전부 모듈-전역 이름 조회(worker.py:1119/1122의
`_normalize_allowlisted_semantics`, 1306/1634의 `_sdf_prim_readback`) — live readback
(1634, expected_live=True)에도 동일 수리가 적용되며, 이는 필수다(같은 스키마 변수성과
float32 저장이 live 스테이지에서도 성립).

## 4. 정적 검증 수치 (source: `g0a_d404/attempt1_*/d404_static_fixture_results.json`, sha256 `edacbfb439985a514ad68f46e7993bb9a314dedad8d3a6e7b951ba25753bdc17`)

- **실물 replay** (동결 D403 derivative를 pxr USD 24.05로 read-only 오픈, PhysxSchema
  plugInfo 등록; 스테이지 생성/저장 0):
  - 동결 readback → pass=False, 실패 체크 정확히 2개
    (`collision_enabled_authored_uniform_noncustom_default_only`, `all_seven_attrs_exact`),
    실패 attr 정확히 2개(sdfMargin, sdfNarrowBandThickness) — **D403 관측과 bit-일치 재현**
  - 관측: variability=`Sdf.VariabilityVarying`, margin=0.009999999776482582, bits=`0x3c23d70a`
  - 동결 semantic gate → **정확히 65 mismatch** 재현; 수리 후 → **0 mismatch**, base/deriv
    normalized digest 동일
  - simulationOwner rel 실측: **authored targets = []** (수리 4의 empty-targets 술어와 일치)
  - 모듈-전역 교체 유효성 replay로 확인 (`replay_module_global_replacement_effective`)
- 등록 18 negative id 중 **10개를 수리된 계약 수준에서 재실행**(resolution_255_257_512,
  remeshing_true, triangle_reduction_not_one, required_api_missing,
  approximation_convex_hull, sdf_mesh_collision_disabled, sdf_api_on_xform,
  instance_proxy_owner, semantic_runtime_ids_excluded, replay negative) — 수리가 기존
  거부력을 약화시키지 않음을 실증. 나머지 8개는 교체 함수가 건드리지 않는 서브시스템
  (stream/inventory/property-query/cook/mass/return-code/RRD footer)이라 체인 해시
  동일성 기반 재승인 (worker_internal_fail_return_zero는 D403 runtime에서 라이브 발화 전례).
- D402-layer 재실행: counter 11(3 accept/8 reject) + Item 8(2 accept/6 reject) 전부 PASS.
- 기타: -B 거부 2, host gate accept(호스트 가시 셸)/reject-shape, AST 2/2, 금지 import 0,
  체인 pin 8/8, `__pycache__` 신규 생성 0.

## 5. 4-lens 적대적 리뷰 (wf_43c263d8-f07; journal은 repo 밖 ~/.claude/.../subagents/workflows/wf_43c263d8-f07/journal.jsonl)

렌즈: repair correctness / chain freeze integrity / approval contract schema /
NVIDIA schema derivation. **Blocker 0.** Warnings 8건 전문 요지:

1. **[운영 — 최중요] dirty-path 원샷 취약성** (3개 렌즈 공통): freeze manifest의
   `no_unexpected_dirty_paths`는 첫 phase write 이후 평가되므로, allowlist(28) 밖 repo
   파일이 실행 시점에 하나라도 있으면 retry-0 attempt가 소모된다. prereg sha가 wrapper에
   embedded라 allowlist 확장은 전면 재작성 요구. → **실행 직전
   `git status --short --untracked-files=all` 전량 대조 필수, 리뷰/저널/스크립트 산출물은
   scratchpad에만**.
2. 수리 4가 rel 레코드 전체(메타데이터 포함)를 드롭 — 관측 서명(targets=[],
   registry metadata)보다 이론상 약간 넓음. 설치 스키마 pin 하에서 도달 불가; scope
   decision으로 기록, 스키마 pin 변경 시 재검토.
3. fixture runner가 scratchpad에만 존재 → **조치**: runner sha256
   `a47f13ce5e87d0dbbcfacdd28d57bb1060dd4d5bec0198c514c6255a87745c7b`를 attestation에
   pin + 본 문서 부록 A에 소스 전문 보존.
4. 9개 registered id의 hash-identity 재승인 근거를 attestation에 id별 명시 → **조치 완료**
   (`fixture_evidence_basis`).
5. `official_nvidia_sources[0]` URL 404 (올바른 철자
   `class_usd_physics_collision_a_p_i.html`, 라이브 fetch 확인) → **조치**: attestation
   `official_source_url_correction` + 본 문서에 정정 기록. prereg는 wrapper-embedded sha
   때문에 미수정 (설치 schema.usda pin이 권위, URL은 보조 — 리뷰어 판단 동일).
6. 세션 doc은 pinned 파일명으로만 생성할 것 → 본 문서가 그 파일명.
7. prereg `planned_static_stage_counters` 키 이름이 gate의 10키와 다름 — gate는
   attestation만 읽으므로 무해; attestation은 정확한 10키 사용 (확인됨).
8. check 키 이름 `..._uniform_...`이 varying 의미를 담게 됨 — record 스키마 바이트 호환을
   위한 의도적 결정 (prereg `record_schema_policy`에 문서화).

리뷰 검증 하이라이트: 3중 seam 합성 순서(D402 item 수리 먼저 → 동일 모듈에 D404 수리)
라인 단위 추적 확인; rebind 키 집합 D403 layer와 완전 합동(worker 9키/controller 25키+1);
sys.modules 미등록 fresh-load라 stale 캐시 불가; 세 API가 composed mesh prim에 기여하는
builtin은 정확히 attr 9개(=SEMANTIC_ALLOWED_MESH_ATTRIBUTES) + rel 1개(=D404 allowlist),
autoApplyTo 부재 — **미모델 builtin 없음**; float32 계산 재검증(0.01f→0x3c23d70a, 1.0f→0x3f800000).

## 6. 산출물과 해시 (전부 allowed_dirty_paths 내)

| 파일 | sha256 |
|---|---|
| `g0a_d404/attempt1_*/d404_preregistration.json` | `4514e824a93902e1b69715df923d43a6c8b86790777b913f3e8c72434b254db0` |
| `sim_scripts/cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_controller.py` | tuple의 `controller_script_sha256` |
| `sim_scripts/cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_worker.py` | tuple의 `worker_script_sha256` |
| `g0a_d404/attempt1_*/d404_static_fixture_results.json` | `edacbfb439985a514ad68f46e7993bb9a314dedad8d3a6e7b951ba25753bdc17` |
| `g0a_d404/attempt1_*/d404_reviewed_script_attestation.json` | `d914c39d727f5fd0718ef77829580af825af16d4be4775b6123a444e262d28d6` |
| `g0a_d404/attempt1_*/d404_proposed_runtime_hash_tuple.json` | **`0d06cc2d3995d80224aaa5289fde2b1e0dacf09ad54e45758fcd54d89220b196`** |

(scratchpad 도구 3종 sha256 — prereg 빌더 `93a8a2ec4b31...445266`, static runner
`a47f13ce...745c7b`(부록 A), attestation 빌더 `7dcd2ab8...346a1b`.)

## 7. 다음 단계 (유저 승인 경계)

**유저가 tuple sha `0d06cc2d3995d80224aaa5289fde2b1e0dacf09ad54e45758fcd54d89220b196`를
명시 승인하면** 호스트 경계에서 1회 실행 (one controller / one worker / retry 0):

```
cd /home/cgxr/Documents/Robotics/RoArm_Project && \
/home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_controller.py \
  --approved-tuple-sha256 0d06cc2d3995d80224aaa5289fde2b1e0dacf09ad54e45758fcd54d89220b196
```

실행 직전 필수 점검 (§5-1): `/dev/nvidiactl` 존재 + 셸 pid 호스트 범위(D402-R1),
`git status --short --untracked-files=all` 전 경로가 prereg allowlist 28개 내,
`HEAD==origin/master==a69a96d`, 잔존 Isaac/Kit 프로세스 0.
PASS 시 rung: cook/property-query 재개 → SDF 물리 A/B (D362 전도 재측정, 34×90/0.72kg 동결).

## 8. 경고 (다음 세션 필독)

- D400/D401/D402/D403 전 attempt 동결 유지. D404 attempt1은 **아직 미소모** (runtime 0회).
- 이 세션은 실행 실험 없음 — 사유: 유저 승인 범위가 "정적 준비까지"로 명시 한정
  (session progress rule 정당화). 실패 가능 요소는 정적 fixture(43 negative)와
  4-lens 리뷰였고 실제로 실행되어 전부 판정을 받았다.
- allowlist 밖 repo 파일 생성 절대 금지 (attempt 소모 위험, §5-1).
- DECISIONS.md 신규 항목 없음 — durable lesson은 D403에 이미 등재, D404 번호는
  runtime 결과 후 부여 예정.
- commit/push는 유저 요청 시에만 (현재 미요청).

## 부록 A — d404_static_runner.py 소스 전문 (sha256 `a47f13ce5e87d0dbbcfacdd28d57bb1060dd4d5bec0198c514c6255a87745c7b`)

```python
#!/usr/bin/env python3
"""D404 static fixture runner (offline, read-only, no Isaac/Kit launch).

Run with:
  LD_LIBRARY_PATH=<conda lib>:<omni.usd.libs bin>:<omni.usd.schema.physx bin> \
      /home/cgxr/miniconda3/envs/isaaclab/bin/python -B d404_static_runner.py \
      [--results-out PATH]

Stages:
  A  chain hash pins (10)
  B  AST parse + top-level import scan + embedded constants + no __pycache__
  C  python-without--B refusal subprocess fixtures (2 reject)
  D  host-boundary gate accept + reject-shape (inherited D403 gate)
  E  D402-layer re-execution: 11 counter fixtures + 8 Item fixtures + 5 positive
  F  D404 repaired readback-contract fixtures (pure, synthetic records)
  G  D404 extended-normalizer fixtures (pure, synthetic rows)
  H  read-only pxr replay against the frozen D403 attempt1 derivative

Loads only inert module definitions with python -B (no runtime main is called,
no Isaac/Kit/PhysX/Warp/CUDA/Rerun import, no repo write outside the optional
--results-out file).
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
SIM = REPO / "sim_scripts"
D400_CONTROLLER = SIM / "cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_preflight.py"
D400_WORKER = SIM / "cyl34_top_view_d400_gripper_link_sdf_res256_live_cook_articulation_worker.py"
D401_CONTROLLER = SIM / "cyl34_top_view_d401_d400_runtime_freeze_snapshot_order_repair_controller.py"
D401_WORKER = SIM / "cyl34_top_view_d401_d400_runtime_freeze_snapshot_order_repair_worker.py"
D402_CONTROLLER = SIM / "cyl34_top_view_d402_d401_runtime_stack_item_and_counter_order_authority_repair_controller.py"
D402_WORKER = SIM / "cyl34_top_view_d402_d401_runtime_stack_item_and_counter_order_authority_repair_worker.py"
D403_CONTROLLER = SIM / "cyl34_top_view_d403_d402_host_boundary_git_repin_rerun_controller.py"
D403_WORKER = SIM / "cyl34_top_view_d403_d402_host_boundary_git_repin_rerun_worker.py"
D404_CONTROLLER = SIM / "cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_controller.py"
D404_WORKER = SIM / "cyl34_top_view_d404_d403_authored_derivative_gate_contract_repair_worker.py"
D404_PREREG = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d404/"
    "attempt1_d403_authored_derivative_gate_contract_repair/"
    "d404_preregistration.json"
)
D403_DERIVATIVE_DIR = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d403/"
    "attempt1_d402_host_boundary_git_repin_rerun/collision_asset/"
    "roarm_m3_link5_a64_gripper_sdf_res256"
)

EXPECTED = {
    D400_CONTROLLER: "6d1f5014535fdffa1e9e63973b1037c14fb1c228a4d22f6a23f980a961ab3b17",
    D400_WORKER: "e5b4b764012258757a9086edb840af40bcc1637586bc05934fec2674ffbd0f0a",
    D401_CONTROLLER: "2807353bb36f3309ed7592bdd3b24f4214ebde8b204ab3e253443f51bf63296e",
    D401_WORKER: "fc019d0d74bc868a5f2cac928824f5de875e05783472f288873f01342775673d",
    D402_CONTROLLER: "af1940a57b05ad9f8afdf8359fc099437360a7ff43eb97259e1ada9eb158da52",
    D402_WORKER: "214d6dcf8e330aa3a6da8a01a614275092462fa337bb1c1fea649c3ec0d654c3",
    D403_CONTROLLER: "187d12f50415d8a33ead42c8cc851adea6614fed9ff777807a7378f757a99d22",
    D403_WORKER: "f594eb36940d25e48985b1ea5cdb1d8e19796353bd1103d61b9ea156b2277f05",
}

ALLOWED_TOP_IMPORTS = {
    "__future__", "argparse", "hashlib", "importlib", "importlib.util",
    "json", "sys", "traceback", "pathlib", "types", "typing",
}

EXT_USD = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
EXT_PHYSX_SCHEMA = Path(
    "/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353"
)

results: list[dict] = []
checks: list[dict] = []
observations: dict = {}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, ok: bool, observed=None) -> None:
    checks.append({"id": name, "pass": bool(ok), "observed": observed})
    if not ok:
        print(f"CHECK FAIL: {name}: {observed}")


def fixture(fid: str, expected: str, ok: bool, observed=None) -> None:
    results.append(
        {
            "id": fid,
            "expected": expected,
            "observed": (
                ("accepted" if ok else "not_accepted")
                if expected == "accept"
                else ("rejected" if ok else "not_rejected")
            ),
            "pass": bool(ok),
            "detail": observed,
        }
    )
    if not ok:
        print(f"FIXTURE FAIL: {fid}: {observed}")


def load_module(path: Path, name: str):
    assert sys.dont_write_bytecode, "runner must be started with python -B"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_sha(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# Stage A: chain hash pins
# --------------------------------------------------------------------------
def stage_a() -> None:
    for path, expected in EXPECTED.items():
        observed = sha(path)
        check(f"pin_exact:{path.name}", observed == expected, observed)
    check("pin_exists:d404_prereg", D404_PREREG.is_file(), str(D404_PREREG))
    observations["d404_prereg_sha256"] = sha(D404_PREREG)


# --------------------------------------------------------------------------
# Stage B: AST + import scan + embedded constants + pycache
# --------------------------------------------------------------------------
def stage_b(d404w, d404c) -> None:
    for path in (D404_CONTROLLER, D404_WORKER):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        check(f"ast_parse:{path.name}", True)
        bad = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                bad += [a.name for a in node.names if a.name.split(".")[0]
                        not in {x.split(".")[0] for x in ALLOWED_TOP_IMPORTS}]
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in {x.split(".")[0] for x in ALLOWED_TOP_IMPORTS}:
                    bad.append(node.module)
        check(f"no_forbidden_top_imports:{path.name}", not bad, bad)
    check(
        "worker_embeds_actual_prereg_sha",
        d404w.EXPECTED_PREREG_SHA256 == observations["d404_prereg_sha256"],
        d404w.EXPECTED_PREREG_SHA256,
    )
    check(
        "controller_embeds_actual_prereg_sha",
        d404c.EXPECTED_PREREG_SHA256 == observations["d404_prereg_sha256"],
        d404c.EXPECTED_PREREG_SHA256,
    )
    check(
        "worker_pins_frozen_d403_worker",
        d404w.EXPECTED_D403_WORKER_SHA256 == EXPECTED[D403_WORKER],
    )
    check(
        "controller_pins_frozen_d403_controller",
        d404c.EXPECTED_D403_CONTROLLER_SHA256 == EXPECTED[D403_CONTROLLER],
    )


# --------------------------------------------------------------------------
# Stage C: -B refusal subprocesses
# --------------------------------------------------------------------------
def stage_c() -> None:
    out = subprocess.run(
        [sys.executable, str(D404_WORKER)],
        capture_output=True, text=True, timeout=60,
    )
    fixture(
        "d404_worker_without_dash_b_rejected", "reject",
        out.returncode != 0 and "python -B" in (out.stderr + out.stdout),
        {"rc": out.returncode},
    )
    out = subprocess.run(
        [sys.executable, str(D404_CONTROLLER),
         "--approved-tuple-sha256", "0" * 64],
        capture_output=True, text=True, timeout=60,
    )
    fixture(
        "d404_controller_without_dash_b_rejected", "reject",
        out.returncode != 0 and "python -B" in (out.stderr + out.stdout),
        {"rc": out.returncode},
    )


# --------------------------------------------------------------------------
# Stage D: host-boundary gate (inherited D403 controller, hash-checked)
# --------------------------------------------------------------------------
def stage_d(d403c) -> None:
    import os
    try:
        d403c._host_boundary_gate()
        accepted = True
        detail = {"pid": os.getpid()}
    except RuntimeError as error:
        accepted = False
        detail = str(error)
    fixture("host_boundary_gate_accepts_on_host", "accept", accepted, detail)
    saved = d403c.GPU_DEVICE_NODES
    try:
        d403c.GPU_DEVICE_NODES = ("/dev/definitely_missing_gpu_node",)
        try:
            d403c._host_boundary_gate()
            rejected = False
        except RuntimeError:
            rejected = True
    finally:
        d403c.GPU_DEVICE_NODES = saved
    fixture("host_boundary_gate_missing_node_rejected", "reject", rejected)


# --------------------------------------------------------------------------
# Stage E: D402-layer counter + Item fixtures
# --------------------------------------------------------------------------
def extract_d400_controller_constants() -> dict:
    tree = ast.parse(D400_CONTROLLER.read_text(encoding="utf-8"))
    wanted = {"COUNTER_KEYS", "EXACT_COUNTERS", "ZERO_COUNTERS",
              "MAX_APP_UPDATE_PUMPS"}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    assert set(found) == wanted, f"missing constants: {wanted - set(found)}"
    return found


def stage_e(d402c, d402w, frozen_worker) -> None:
    consts = extract_d400_controller_constants()
    check("counter_keys_36", len(consts["COUNTER_KEYS"]) == 36)
    check("exact_counters_14", len(consts["EXACT_COUNTERS"]) == 14)
    check("zero_counters_21", len(consts["ZERO_COUNTERS"]) == 21)
    check(
        "counter_partition_exact",
        set(consts["COUNTER_KEYS"])
        == set(consts["EXACT_COUNTERS"])
        | set(consts["ZERO_COUNTERS"])
        | {"simulation_app_update_pumps"},
    )
    base = SimpleNamespace(
        COUNTER_KEYS=tuple(consts["COUNTER_KEYS"]),
        EXACT_COUNTERS=dict(consts["EXACT_COUNTERS"]),
        ZERO_COUNTERS=tuple(consts["ZERO_COUNTERS"]),
        MAX_APP_UPDATE_PUMPS=consts["MAX_APP_UPDATE_PUMPS"],
        _json_sha256=canonical_sha,
    )
    gate = lambda counters: d402c._counter_gate_registered_projection(
        base, counters
    )

    def baseline() -> dict:
        return {
            key: (1 if key == "simulation_app_update_pumps"
                  else consts["EXACT_COUNTERS"].get(key, 0))
            for key in consts["COUNTER_KEYS"]
        }

    fixture("counter_registered_insertion_order", "accept",
            gate(baseline())["pass"] is True)
    roundtrip = json.loads(json.dumps(baseline(), sort_keys=True))
    fixture("counter_sort_keys_json_roundtrip_with_different_physical_order",
            "accept", gate(roundtrip)["pass"] is True)
    reverse = dict(reversed(list(baseline().items())))
    fixture("counter_reverse_insertion_order_same_schema_and_values",
            "accept", gate(reverse)["pass"] is True)

    missing = baseline(); missing.pop("q5_samples")
    missing["q5_samplez"] = 0
    fixture("counter_missing_extra_or_misspelled_key_rejected", "reject",
            gate(missing)["pass"] is False)
    boolean = baseline(); boolean["q5_samples"] = False
    floaty = baseline(); floaty["contact_queries"] = 0.0
    stringy = baseline(); stringy["resets"] = "0"
    fixture("counter_bool_float_or_string_value_rejected", "reject",
            gate(boolean)["pass"] is False
            and gate(floaty)["pass"] is False
            and gate(stringy)["pass"] is False)
    wrong = baseline(); wrong["physx_stage_attaches"] = 2
    fixture("counter_wrong_exact_value_rejected", "reject",
            gate(wrong)["pass"] is False)
    nonzero = baseline(); nonzero["public_forwards"] = 1
    fixture("counter_nonzero_frozen_zero_rejected", "reject",
            gate(nonzero)["pass"] is False)
    pump0 = baseline(); pump0["simulation_app_update_pumps"] = 0
    pumphi = baseline()
    pumphi["simulation_app_update_pumps"] = consts["MAX_APP_UPDATE_PUMPS"] + 1
    fixture("counter_pump_zero_or_overflow_rejected", "reject",
            gate(pump0)["pass"] is False and gate(pumphi)["pass"] is False)
    physics = baseline(); physics["controlled_physics_steps"] = 1
    fixture("counter_controlled_physics_step_nonzero_rejected", "reject",
            gate(physics)["pass"] is False)
    nonmap_rejected = gate("not a mapping")["pass"] is False
    try:
        frozen_worker._json_no_duplicates('{"a":1,"a":2}')
        dup_rejected = False
    except ValueError:
        dup_rejected = True
    fixture("counter_non_mapping_or_duplicate_json_key_rejected", "reject",
            nonmap_rejected and dup_rejected)
    ordered_wrong = baseline(); ordered_wrong["q5_commands"] = 3
    fixture("counter_correct_physical_order_but_wrong_value_rejected",
            "reject", gate(ordered_wrong)["pass"] is False)

    version_of = d402w._extension_package_version
    expected_version = d402w.EXPECTED_OMNI_PHYSX_EXTENSION_VERSION

    class ItemLike:
        def __init__(self, mapping):
            self._mapping = mapping

        def __getitem__(self, key):
            value = self._mapping[key]
            return ItemLike(value) if isinstance(value, dict) and key != "version" else value

    item = ItemLike({"package": {"version": "107.3.26"}})
    fixture("item_non_dict_nested_indexing_exact_version", "accept",
            version_of(item) == expected_version)
    fixture("item_builtin_dict_compatibility", "accept",
            version_of({"package": {"version": "107.3.26"}})
            == expected_version)
    fixture("item_missing_package_rejected", "reject",
            version_of({}) is None)
    fixture("item_missing_version_rejected", "reject",
            version_of({"package": {}}) is None)
    fixture("item_non_string_version_rejected", "reject",
            version_of({"package": {"version": 107.326}}) is None)
    wrong_version = version_of({"package": {"version": "107.3.25"}})
    fixture("item_wrong_string_version_rejected", "reject",
            not (type(wrong_version) is str
                 and wrong_version == expected_version))
    fixture("item_extension_id_or_toml_fallback_rejected", "reject",
            version_of(None) is None)

    class Exploding:
        def __getitem__(self, key):
            raise RuntimeError("unexpected accessor failure")

    try:
        version_of(Exploding())
        not_swallowed = False
    except RuntimeError:
        not_swallowed = True
    fixture("item_unexpected_accessor_runtime_error_not_swallowed", "reject",
            not_swallowed)


# --------------------------------------------------------------------------
# Stage F: D404 repaired readback-contract fixtures (pure)
# --------------------------------------------------------------------------
def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


def clean_shape(uniform: bool) -> dict:
    return {
        "valid": True, "authored": True, "custom": False,
        "variability": ("Sdf.VariabilityUniform" if uniform
                        else "Sdf.VariabilityVarying"),
        "uniform": uniform, "time_samples": [], "connections": [],
    }


def make_record(specs: dict) -> dict:
    attrs = {}
    for name, (etype, evalue, ebits) in specs.items():
        value = f32(evalue) if etype == "float" else evalue
        attrs[name] = {
            **clean_shape(uniform=True),
            "usd_type": etype,
            "value": value,
            "float32_bits_hex": f32_bits(value) if etype == "float" else None,
        }
    return {
        "path": "/roarm_m3/gripper_link/collisions/gripper_link/"
                "node_STL_BINARY_/mesh",
        "applied_schemas": ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI",
                            "PhysxSDFMeshCollisionAPI"],
        "collision_enabled": True,
        "collision_enabled_shape": clean_shape(uniform=False),
        "approximation": "sdf",
        "approximation_shape": clean_shape(uniform=True),
        "attributes": attrs,
        "attribute_checks": {name: False for name in specs},
        "material_binding_targets": ["/expected/material"],
        "instance_flags": {"is_instanceable": False, "is_instance": False,
                           "is_instance_proxy": False},
        "checks": {
            "is_mesh": True,
            "collision_enabled_true": True,
            "collision_enabled_authored_uniform_noncustom_default_only": False,
            "approximation_sdf": True,
            "approximation_authored_uniform_noncustom_default_only": True,
            "required_apis_applied": True,
            "all_seven_attrs_exact": False,
            "material_binding_exact": True,
            "noninstance_state_when_live": True,
        },
        "pass": False,
    }


def frozen_attr_check(row: dict, spec: tuple) -> bool:
    """Transcription of frozen worker.py:1447-1460 semantics."""
    etype, evalue, ebits = spec
    return bool(
        row["valid"] and row["authored"] and row["custom"] is False
        and row["uniform"] is True and row["time_samples"] == []
        and row["connections"] == [] and row["usd_type"] == etype
        and row["value"] == evalue
        and (ebits is None or row["float32_bits_hex"] == ebits)
    )


def stage_f(d404w, frozen_worker) -> None:
    specs = frozen_worker.SDF_ATTRIBUTE_SPECS
    check("frozen_specs_7", len(specs) == 7, sorted(specs))
    check(
        "frozen_float_specs_bits",
        specs["physxSDFMeshCollision:sdfNarrowBandThickness"] == ("float", 0.01, "0x3c23d70a")
        and specs["physxSDFMeshCollision:sdfMargin"] == ("float", 0.01, "0x3c23d70a")
        and specs["physxSDFMeshCollision:sdfTriangleCountReductionFactor"] == ("float", 1.0, "0x3f800000"),
    )
    check("frozen_a64_64", len(frozen_worker.SOURCE_GRIPPER_A64_PATHS) == 64)

    repair = lambda record: d404w._repaired_readback_contract(record, specs)

    record = make_record(specs)
    frozen_float_reject = not frozen_attr_check(
        record["attributes"]["physxSDFMeshCollision:sdfMargin"],
        specs["physxSDFMeshCollision:sdfMargin"],
    )
    repaired = repair(copy.deepcopy(record))
    fixture(
        "readback_schema_conforming_derivative_accepted", "accept",
        repaired["pass"] is True
        and repaired["checks"]["collision_enabled_authored_uniform_noncustom_default_only"] is True
        and repaired["checks"]["all_seven_attrs_exact"] is True
        and frozen_float_reject,
        {"frozen_double_equality_would_reject": frozen_float_reject,
         "margin_value": record["attributes"]["physxSDFMeshCollision:sdfMargin"]["value"]},
    )

    bad = copy.deepcopy(record)
    bad["collision_enabled_shape"] = clean_shape(uniform=True)
    fixture("readback_uniform_collision_enabled_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["collision_enabled_shape"]["authored"] = False
    fixture("readback_unauthored_collision_enabled_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["collision_enabled_shape"]["time_samples"] = [0.0]
    fixture("readback_collision_enabled_time_samples_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["attributes"]["physxSDFMeshCollision:sdfNarrowBandThickness"]["float32_bits_hex"] = "0x3c23d70b"
    fixture("readback_wrong_float_bits_rejected", "reject",
            repair(bad)["pass"] is False)

    rejected_all = True
    for wrong_resolution in (255, 257, 512):
        bad = copy.deepcopy(record)
        bad["attributes"]["physxSDFMeshCollision:sdfResolution"]["value"] = wrong_resolution
        rejected_all &= repair(bad)["pass"] is False
    fixture("resolution_255_257_512_rejected", "reject", rejected_all)

    bad = copy.deepcopy(record)
    bad["attributes"]["physxSDFMeshCollision:sdfEnableRemeshing"]["value"] = True
    fixture("remeshing_true_rejected", "reject", repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    row = bad["attributes"]["physxSDFMeshCollision:sdfTriangleCountReductionFactor"]
    row["value"] = f32(0.9)
    row["float32_bits_hex"] = f32_bits(0.9)
    fixture("triangle_reduction_not_one_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["attributes"]["physxSDFMeshCollision:sdfMargin"]["usd_type"] = "double"
    fixture("readback_wrong_usd_type_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    del bad["attributes"]["physxSDFMeshCollision:sdfMargin"]
    del bad["attribute_checks"]["physxSDFMeshCollision:sdfMargin"]
    fixture("readback_missing_registered_attr_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["attributes"]["physxSDFMeshCollision:sdfBitsPerSubgridPixel"]["value"] = "BitsPerPixel8"
    fixture("readback_wrong_subgrid_token_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["checks"]["required_apis_applied"] = False
    fixture("required_api_missing_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["checks"]["approximation_sdf"] = False
    fixture("approximation_convex_hull_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["checks"]["collision_enabled_true"] = False
    fixture("sdf_mesh_collision_disabled_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["checks"]["is_mesh"] = False
    fixture("sdf_api_on_xform_rejected", "reject",
            repair(bad)["pass"] is False)

    bad = copy.deepcopy(record)
    bad["checks"]["noninstance_state_when_live"] = False
    fixture("instance_proxy_owner_rejected", "reject",
            repair(bad)["pass"] is False)

    invalid = {"path": "/x", "pass": False, "error": "invalid prim"}
    out = repair(copy.deepcopy(invalid))
    fixture("readback_invalid_prim_early_return_rejected", "reject",
            out == invalid and out["pass"] is False)


# --------------------------------------------------------------------------
# Stage G: D404 extended-normalizer fixtures (pure)
# --------------------------------------------------------------------------
def a64_row(path: str, default_value) -> dict:
    return {
        "path": path,
        "type_name": "Mesh",
        "active": True,
        "instanceable": False,
        "applied_schemas": ["PhysicsCollisionAPI"],
        "metadata": [["specifier", "def"]],
        "attributes": [
            {
                "name": "physics:collisionEnabled",
                "type_name": "bool",
                "value": "$ALLOWLIST_COLLISION_ENABLED_VALUE",
                "time_samples": [],
                "connections": [],
                "metadata": [["default", default_value],
                             ["typeName", "bool"]],
            },
            {
                "name": "physics:mass",
                "type_name": "float",
                "value": 0.1,
                "time_samples": [],
                "connections": [],
                "metadata": [["default", 0.1], ["typeName", "float"]],
            },
        ],
        "relationships": [],
    }


def mesh_row(mesh_path: str, relationships: list) -> dict:
    return {
        "path": mesh_path,
        "type_name": "Mesh",
        "active": True,
        "instanceable": False,
        "applied_schemas": [],
        "metadata": [["specifier", "def"]],
        "attributes": [
            {
                "name": "extent",
                "type_name": "float3[]",
                "value": [[0, 0, 0], [1, 1, 1]],
                "time_samples": [],
                "connections": [],
                "metadata": [["typeName", "float3[]"]],
            }
        ],
        "relationships": relationships,
    }


def stage_g(d404w, frozen_worker) -> None:
    a64_paths = frozenset(frozen_worker.SOURCE_GRIPPER_A64_PATHS)
    mesh_path = frozen_worker.SOURCE_MESH_PATH
    part0 = frozen_worker.SOURCE_GRIPPER_A64_PATHS[0]
    extend = lambda rows: d404w._extended_allowlist_normalization(
        rows, a64_paths, mesh_path
    )

    base_rows = [a64_row(part0, True)]
    deriv_rows = [a64_row(part0, False)]
    fixture(
        "normalizer_default_metadata_masked_equal", "accept",
        canonical_sha(extend(copy.deepcopy(base_rows)))
        == canonical_sha(extend(copy.deepcopy(deriv_rows))),
    )

    base_rows = [a64_row(part0, True)]
    absent = a64_row(part0, True)
    absent["attributes"][0]["metadata"] = [["typeName", "bool"]]
    fixture(
        "normalizer_default_metadata_presence_diff_rejected", "reject",
        canonical_sha(extend(copy.deepcopy(base_rows)))
        != canonical_sha(extend([absent])),
    )

    base_rows = [a64_row(part0, True)]
    other = a64_row(part0, True)
    other["attributes"][1]["metadata"] = [["default", 0.2],
                                          ["typeName", "float"]]
    fixture(
        "normalizer_other_attr_metadata_change_rejected", "reject",
        canonical_sha(extend(copy.deepcopy(base_rows)))
        != canonical_sha(extend([other])),
    )

    outside = "/roarm_m3/link5/collisions/d338_convex_parts/part_000"
    fixture(
        "normalizer_non_a64_path_default_change_rejected", "reject",
        canonical_sha(extend([a64_row(outside, True)]))
        != canonical_sha(extend([a64_row(outside, False)])),
    )

    owner_builtin = {"name": "physics:simulationOwner", "targets": [],
                     "metadata": [["variability", 0]]}
    fixture(
        "normalizer_simulation_owner_builtin_filtered_equal", "accept",
        canonical_sha(extend([mesh_row(mesh_path, [])]))
        == canonical_sha(extend([mesh_row(mesh_path, [owner_builtin])])),
    )

    owner_authored = {"name": "physics:simulationOwner",
                      "targets": ["/World/PhysicsScene"],
                      "metadata": [["variability", 0]]}
    fixture(
        "normalizer_simulation_owner_authored_target_rejected", "reject",
        canonical_sha(extend([mesh_row(mesh_path, [])]))
        != canonical_sha(extend([mesh_row(mesh_path, [owner_authored])])),
    )

    other_rel = {"name": "physics:filteredPairs", "targets": [],
                 "metadata": []}
    fixture(
        "normalizer_other_relationship_rejected", "reject",
        canonical_sha(extend([mesh_row(mesh_path, [])]))
        != canonical_sha(extend([mesh_row(mesh_path, [other_rel])])),
    )

    non_mesh = a64_row(part0, True)
    with_rel = copy.deepcopy(non_mesh)
    with_rel["relationships"] = [copy.deepcopy(owner_builtin)]
    fixture(
        "normalizer_non_mesh_path_relationship_rejected", "reject",
        canonical_sha(extend([copy.deepcopy(non_mesh)]))
        != canonical_sha(extend([with_rel])),
    )

    changed = mesh_row(mesh_path, [])
    changed["attributes"][0]["value"] = [[0, 0, 0], [2, 2, 2]]
    fixture(
        "semantic_runtime_ids_excluded_but_semantic_change_rejected",
        "reject",
        canonical_sha(extend([mesh_row(mesh_path, [])]))
        != canonical_sha(extend([changed])),
    )


# --------------------------------------------------------------------------
# Stage H: read-only pxr replay against the frozen D403 derivative
# --------------------------------------------------------------------------
def stage_h(d404w, frozen_worker) -> None:
    sys.path.insert(0, str(EXT_PHYSX_SCHEMA))
    sys.path.insert(0, str(EXT_USD))
    from pxr import Plug, Usd  # noqa: F401

    registered = Plug.Registry().RegisterPlugins(
        str(EXT_PHYSX_SCHEMA / "plugins/PhysxSchema/resources")
    )
    check("physx_schema_plugin_registered",
          any(p.name == "physxSchema" for p in registered)
          or Plug.Registry().GetPluginWithName("physxSchema") is not None)

    base_stage = Usd.Stage.Open(str(frozen_worker.BASE_ROOT_USD))
    deriv_root = D403_DERIVATIVE_DIR / "roarm_m3.usd"
    deriv_stage = Usd.Stage.Open(str(deriv_root))
    check("replay_stages_open", base_stage is not None
          and deriv_stage is not None)

    frozen_record = frozen_worker._sdf_prim_readback(
        deriv_stage, frozen_worker.SOURCE_MESH_PATH, expected_live=False
    )
    failing_checks = sorted(
        key for key, ok in frozen_record["checks"].items() if not ok
    )
    failing_attrs = sorted(
        key for key, ok in frozen_record["attribute_checks"].items() if not ok
    )
    observations["replay_frozen_failing_checks"] = failing_checks
    observations["replay_frozen_failing_attrs"] = failing_attrs
    observations["replay_collision_enabled_variability"] = (
        frozen_record["collision_enabled_shape"]["variability"]
    )
    observations["replay_margin_value"] = (
        frozen_record["attributes"]["physxSDFMeshCollision:sdfMargin"]["value"]
    )
    observations["replay_margin_bits"] = (
        frozen_record["attributes"]["physxSDFMeshCollision:sdfMargin"]["float32_bits_hex"]
    )
    fixture(
        "replay_frozen_readback_reproduces_defect", "reject",
        frozen_record["pass"] is False
        and failing_checks == [
            "all_seven_attrs_exact",
            "collision_enabled_authored_uniform_noncustom_default_only",
        ]
        and failing_attrs == [
            "physxSDFMeshCollision:sdfMargin",
            "physxSDFMeshCollision:sdfNarrowBandThickness",
        ],
        {"failing_checks": failing_checks, "failing_attrs": failing_attrs},
    )

    repaired_record = d404w._repaired_readback_contract(
        copy.deepcopy(frozen_record), frozen_worker.SDF_ATTRIBUTE_SPECS
    )
    fixture(
        "replay_repaired_readback_accepts_real_derivative", "accept",
        repaired_record["pass"] is True
        and all(repaired_record["checks"].values())
        and all(repaired_record["attribute_checks"].values()),
        {"margin_bits": observations["replay_margin_bits"]},
    )

    frozen_gate = frozen_worker._composed_semantic_diff_gate(
        base_stage, deriv_stage, D403_DERIVATIVE_DIR
    )
    observations["replay_frozen_mismatch_count"] = len(
        frozen_gate["nonallowlisted_mismatches"]
    )
    fixture(
        "replay_frozen_semantic_gate_reproduces_65_mismatches", "reject",
        frozen_gate["pass"] is False
        and len(frozen_gate["nonallowlisted_mismatches"]) == 65,
        {"count": len(frozen_gate["nonallowlisted_mismatches"])},
    )

    mesh_rows = [
        row
        for row in frozen_worker._composed_stage_rows(deriv_stage)
        if row["path"] == frozen_worker.SOURCE_MESH_PATH
    ]
    owner_rels = [
        rel for rel in mesh_rows[0]["relationships"]
        if rel["name"] == "physics:simulationOwner"
    ]
    fixture(
        "replay_simulation_owner_targets_empty_observed", "accept",
        len(owner_rels) == 1 and owner_rels[0]["targets"] == [],
        {"owner_rels": owner_rels},
    )

    d404w._install_repaired_gate_functions(frozen_worker)
    repaired_gate = frozen_worker._composed_semantic_diff_gate(
        base_stage, deriv_stage, D403_DERIVATIVE_DIR
    )
    fixture(
        "replay_repaired_semantic_gate_zero_mismatches", "accept",
        repaired_gate["pass"] is True
        and repaired_gate["nonallowlisted_mismatches"] == []
        and repaired_gate["checks"]["composed_prim_path_sequence_exact"],
        {"base_digest": repaired_gate["base_normalized_sha256"],
         "derivative_digest": repaired_gate["derivative_normalized_sha256"]},
    )
    repaired_live_readback = frozen_worker._sdf_prim_readback(
        deriv_stage, frozen_worker.SOURCE_MESH_PATH, expected_live=False
    )
    fixture(
        "replay_module_global_replacement_effective", "accept",
        repaired_live_readback["pass"] is True,
    )

    rows_base = frozen_worker._normalize_allowlisted_semantics(
        frozen_worker._composed_stage_rows(base_stage),
        frozen_worker.BASE_ASSET_DIR,
    )
    rows_deriv = frozen_worker._normalize_allowlisted_semantics(
        frozen_worker._composed_stage_rows(deriv_stage),
        D403_DERIVATIVE_DIR,
    )
    digest_base, _ = frozen_worker._rows_digest(rows_base)
    digest_deriv, _ = frozen_worker._rows_digest(rows_deriv)
    check("replay_repaired_digests_equal", digest_base == digest_deriv)
    rows_deriv[0]["active"] = False
    perturbed_digest, _ = frozen_worker._rows_digest(rows_deriv)
    fixture(
        "replay_injected_nonallowlisted_change_rejected", "reject",
        perturbed_digest != digest_base,
    )


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-out", default=None)
    args = parser.parse_args()

    assert sys.dont_write_bytecode, "start the runner with python -B"

    pycache_before = sorted(str(p) for p in SIM.glob("__pycache__/*"))

    stage_a()
    d404w = load_module(D404_WORKER, "_d404_worker_under_test")
    d404c = load_module(D404_CONTROLLER, "_d404_controller_under_test")
    stage_b(d404w, d404c)
    stage_c()
    d403c = load_module(D403_CONTROLLER, "_d403_controller_for_gate_fixture")
    stage_d(d403c)
    d402c = load_module(D402_CONTROLLER, "_d402_controller_for_fixtures")
    d402w = load_module(D402_WORKER, "_d402_worker_for_fixtures")
    frozen_worker = load_module(D400_WORKER, "_d400_worker_for_fixtures")
    stage_e(d402c, d402w, frozen_worker)
    stage_f(d404w, frozen_worker)
    stage_g(d404w, frozen_worker)
    stage_h(d404w, frozen_worker)

    pycache_after = sorted(str(p) for p in SIM.glob("__pycache__/*"))
    new_pyc = sorted(set(pycache_after) - set(pycache_before))
    chain_pyc = [p for p in pycache_after if "d40" in Path(p).name]
    check("no_pycache_created_by_runner", not new_pyc, new_pyc)
    check("no_chain_d40x_pycache_exists", not chain_pyc, chain_pyc)

    accepts = [r for r in results if r["expected"] == "accept"]
    rejects = [r for r in results if r["expected"] == "reject"]
    summary = {
        "artifact": "D404_STATIC_FIXTURE_RESULTS_V1",
        "runner": "scratchpad d404_static_runner.py, isaaclab python -B, "
                  "read-only (no repo write besides this results file)",
        "checks_passed": sum(1 for c in checks if c["pass"]),
        "checks_total": len(checks),
        "accept_fixtures_passed": sum(1 for r in accepts if r["pass"]),
        "accept_fixtures_total": len(accepts),
        "reject_fixtures_passed": sum(1 for r in rejects if r["pass"]),
        "reject_fixtures_total": len(rejects),
        "all_pass": all(c["pass"] for c in checks)
        and all(r["pass"] for r in results),
        "observations": observations,
        "checks": checks,
        "fixtures": results,
    }
    print(json.dumps(
        {k: summary[k] for k in (
            "checks_passed", "checks_total", "accept_fixtures_passed",
            "accept_fixtures_total", "reject_fixtures_passed",
            "reject_fixtures_total", "all_pass")},
        indent=2,
    ))
    print("observations:", json.dumps(observations, indent=2, default=str))
    if args.results_out:
        out = Path(args.results_out)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                  default=str) + "\n")
        print(f"wrote {out}")
        print(f"results sha256 {sha(out)}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```
